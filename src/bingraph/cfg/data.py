"""Concrete data-reference observations used to prevent code/data overlap.

Exact decoded blocks contribute non-store VEX reads; direct control-flow
claims override data classification at their addresses. Execute-relative
instruction templates stay eligible as code. These observations constrain
decoding/recovery, not indirect-target domains, and are scoped to a build.
"""

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
        """Normalize Thumb-tagged execution addresses to loader byte addresses.

        Code claims and data ranges must share this coordinate system, while
        actual CFG node addresses retain their instruction-set mode bit.
        """

        try:
            if project.arch.is_thumb(addr):
                return addr & ~1
        except AttributeError:
            pass
        return addr

    def claim_code(self, project: Project, addr: int) -> None:
        """Give an explicit code target precedence over a prior data observation.

        Record its physical address without deleting the data evidence. The
        membership query resolves the conflict; this method neither decodes
        instructions nor declares an arbitrary observed load executable.
        """

        self.code_addrs.add(self._memory_addr(project, addr))

    def contains(self, project: Project, addr: int) -> bool:
        """Test an observed data byte while respecting direct code precedence.

        Normalize mode-tagged addresses before comparing sets. A false result
        means decoding is not excluded here, not that an instruction boundary
        or executable destination has been proved.
        """

        memory_addr = self._memory_addr(project, addr)
        return memory_addr in self.data_bytes and memory_addr not in self.code_addrs

    def record_block(self, project: Project, block: BlockSpec) -> bool:
        """Collect data bytes from one exact lift, returning whether the set grew.

        Deduplicate observations by block start/size so redecoding identical
        spans does not repeat work. Include concrete reads even when VEX calls
        their type unknown, but exclude stores and execution-template fetches.
        Failed lifting contributes no observations; existing code claims remain
        authoritative without making data classification a target proof.
        """

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
