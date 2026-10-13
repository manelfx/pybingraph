"""Mode-aware instruction decoding and cached node inspection.

This layer exposes Capstone bytes without CFG discovery or target inference.
Node views are cached by object identity; replacing a block must not reuse the
old object's instruction stream. Callers can explicitly discard views at
project or analysis boundaries.
"""

from __future__ import annotations

from dataclasses import dataclass

from angr import Project
from capstone import CsInsn

# CFGNode equality can collide with a replacement at the same block address.
# Retaining the object and keying by identity prevents stale disassembly reuse.
_DECODED_NODE_CACHE: dict[int, tuple[object, DecodedNode]] = {}


def clear_decoded_node_cache() -> None:
    """Discard cached node views when the caller ends an analysis lifetime."""

    _DECODED_NODE_CACHE.clear()


def decode_raw_capstone_insns(
    project: Project,
    addr: int,
    size: int,
    *,
    count: int = 0,
) -> tuple[CsInsn, ...]:
    """Disassemble a requested byte span without discovering CFG boundaries.

    Select the ARM/Thumb engine from the execution address, read its physical
    bytes and preserve the tagged instruction addresses in the result. ``count``
    optionally limits the number of instructions. Unsupported engines or
    unreadable spans return an empty tuple, not a fabricated linear decode.
    """

    try:
        arch = project.arch
        try:
            is_thumb = arch.is_thumb(addr)
        except AttributeError:
            is_thumb = False

        # ARM/Thumb stores the execution mode in address bit 0. Read bytes at
        # the physical address, while preserving the tagged address in output.
        # ``capstone_thumb`` is an ARM-specific extension not declared on the
        # base archinfo ``Arch`` type.
        thumb_capstone = getattr(arch, "capstone_thumb", None)

        if is_thumb and thumb_capstone is not None:
            capstone = thumb_capstone
            memory_addr = addr & ~1
        else:
            capstone = arch.capstone
            memory_addr = addr
        data = project.loader.memory.load(memory_addr, size)
        return tuple(capstone.disasm(data, addr, count=count))
    except Exception:
        return ()


def decode_one(project: Project, addr: int, size: int) -> CsInsn | None:
    """Inspect one instruction using an exact-sized lift, then raw Capstone bytes.

    Prefer angr's block instruction view when available, but do not inherit
    previously discovered CFG boundaries. The fallback uses the same mode-aware
    raw decoder so lifting failures need not hide a valid Capstone instruction.
    Return ``None`` when both paths fail; callers decide whether a VEX-only
    linear fallback is permitted.
    """

    try:
        block = project.factory.block(
            addr,
            size=size,
            strict_block_end=True,
            cross_insn_opt=False,
        )
        capstone_insns = block.capstone.insns
        if capstone_insns:
            return capstone_insns[0].insn
    except Exception:
        pass

    fallback_insns = decode_raw_capstone_insns(project, addr, size, count=1)
    return fallback_insns[0] if fallback_insns else None


@dataclass(frozen=True)
class DecodedNode:
    """The Capstone instruction view of one CFG node, when it is available."""

    insns: tuple[CsInsn, ...] | None
    inspection_error: Exception | None = None

    @classmethod
    def from_node(cls, node) -> DecodedNode:
        """Read and cache Capstone instructions for one live CFG node.

        Target queries inspect the same CFG nodes repeatedly while the
        worklist discovers nearby blocks. ``node.block.capstone`` constructs a
        fresh angr Block each time, so retaining this immutable view avoids
        repeatedly disassembling unchanged node bytes. The cache is scoped to
        object identity because angr CFG nodes compare by block identity; a
        recovered replacement can otherwise collide with the stale node it
        replaced at the same address.
        """

        cache_key = id(node)
        cached = _DECODED_NODE_CACHE.get(cache_key)
        if cached is not None and cached[0] is node:
            return cached[1]

        try:
            block = node.block
            insns = tuple(item.insn for item in block.capstone.insns)
            if insns:
                decoded = cls(insns)
            else:
                project = block._project
                decoded = cls(decode_raw_capstone_insns(project, node.addr, node.size))

        except Exception as exc:
            decoded = cls(None, exc)

        _DECODED_NODE_CACHE[cache_key] = node, decoded
        return decoded

    @property
    def is_empty(self) -> bool:
        """Return whether Capstone found no instructions in the node."""

        return not self.insns

    @property
    def last(self) -> CsInsn | None:
        """Return the final decoded instruction, if one exists."""

        return self.insns[-1] if self.insns else None

    def has_exact_coverage(self, node) -> bool:
        """Return whether instructions exactly cover the node's declared range."""

        if node.size == 0 or self.insns is None:
            return False
        expected_addr = node.addr
        for insn in self.insns:
            if insn.address != expected_addr:
                return False
            expected_addr += insn.size
        return expected_addr == node.addr + node.size

    def contains_mid_instruction_addr(self, addr: int) -> bool:
        """Return whether ``addr`` falls strictly inside a decoded instruction."""

        if self.insns is None:
            return False
        return any(
            insn.address < addr < insn.address + insn.size for insn in self.insns
        )
