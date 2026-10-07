"""VEX-proven static data ranges used to bound construction."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from angr import Project
import pyvex

from bingraph.cfg.models import BlockSpec


def _execute_relative_instruction_addrs(vex: Any) -> frozenset[int]:
    """Return instructions whose VEX lift loads bytes for execution.

    Some architectures execute an instruction template fetched from memory.
    VEX represents that fetch as a concrete data reference and marks the
    instruction with ``Ijk_InvalICache``. It is code, not a literal pool.
    """

    instruction_addr: int | None = None
    execute_relative: set[int] = set()
    for stmt in vex.statements:
        if isinstance(stmt, pyvex.stmt.IMark):
            instruction_addr = stmt.addr
        elif (
            isinstance(stmt, pyvex.stmt.Exit)
            and stmt.jumpkind == "Ijk_InvalICache"
            and instruction_addr is not None
        ):
            execute_relative.add(instruction_addr)
    return frozenset(execute_relative)


@dataclass
class StaticDataRegions:
    """Track concrete non-code bytes read by already decoded blocks.

    VEX data references identify literal pools without assuming an
    architecture-specific encoding. Concrete reads are recorded regardless of
    VEX's data-type label: conditional loads are often labeled ``unknown``.
    Stores and execute-relative reads may be self-modifying or executable code
    and must stay decodable.
    """

    data_bytes: set[int] = field(default_factory=set)
    code_addrs: set[int] = field(default_factory=set)
    observed_blocks: set[tuple[int, int]] = field(default_factory=set)

    @staticmethod
    def _memory_addr(project: Project, addr: int) -> int:
        """Return the physical memory address for an execution address."""

        try:
            if project.arch.is_thumb(addr):
                return addr & ~1
        except AttributeError:
            pass
        return addr

    def claim_code(self, project: Project, addr: int) -> None:
        """Keep an explicit direct control-flow target executable."""

        self.code_addrs.add(self._memory_addr(project, addr))

    def contains(self, project: Project, addr: int) -> bool:
        """Return whether an address is data without an explicit code claim."""

        memory_addr = self._memory_addr(project, addr)
        return memory_addr in self.data_bytes and memory_addr not in self.code_addrs

    def record_block(self, project: Project, block: BlockSpec) -> bool:
        """Record concrete non-store reads from one exact VEX block lift."""

        key = block.addr, block.size
        if key in self.observed_blocks:
            return False
        self.observed_blocks.add(key)

        try:
            vex = project.factory.block(
                block.addr,
                size=block.size,
                strict_block_end=True,
                cross_insn_opt=False,
                collect_data_refs=True,
            ).vex
        except Exception:
            return False

        execute_relative = _execute_relative_instruction_addrs(vex)
        changed = False
        for ref in vex.data_refs or ():
            if (
                ref.data_size <= 0
                or "store" in ref.data_type_str.lower()
                or ref.ins_addr in execute_relative
            ):
                continue
            start = self._memory_addr(project, ref.data_addr)
            before = len(self.data_bytes)
            self.data_bytes.update(range(start, start + ref.data_size))
            changed |= len(self.data_bytes) != before
        return changed
