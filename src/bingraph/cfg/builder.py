"""Bounded CFG construction without CFGFast.

The builder starts at one function symbol and grows only through addresses
proven by decoded direct transfers. A shared leader set keeps recovered blocks
non-overlapping: whenever a newly discovered target falls inside an existing
block, that block is re-decoded with the target as a stop address.

Construction has four deliberate stages: decode direct flow, resolve exact
indirect transfers, discover disconnected code with an unknown entry, then
materialize the final graph. VEX-proven static jump tables contribute leaders
during the exact-resolution stage; remaining indirect transfers stay explicit
synthetic leaves. The builder never reads CFGFast's discovered regions.
"""

from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from dataclasses import replace
import re
from types import SimpleNamespace
from typing import Iterable, Mapping, cast

from angr import KnowledgeBase, Project
from angr.knowledge_plugins.cfg import CFGModel, CFGNode
from capstone import CS_OP_REG
from capstone.arm import (
    ARM_CC_AL,
    ARM_INS_BX,
    ARM_INS_IT,
    ARM_INS_POP,
    ARM_INS_PUSH,
    ARM_REG_LR,
    ARM_REG_PC,
    ARM_REG_SP,
)
from cle.backends.symbol import SymbolType
from loguru import logger
import networkx as nx

from bingraph.cfg.anomalies import _lookup_function_bounds
from bingraph.cfg.graph import CFGGraph, add_successor_edge, node_vex
from bingraph.cfg.jumps import (
    _jump_table_addr,
    _read_static_jump_table_targets,
    abi_static_register_transfer_targets,
    conditional_pc_dispatch_targets,
    is_memory_dependent_indirect_jump,
    plan_mips_pic_relative_jump_table,
    plan_static_jump_table,
    s390_table_loaded_branch,
    static_jump_target_rejection_reason,
    vex_has_computed_pc_transfer,
    vex_is_conditional_link_return,
)
from bingraph.cfg.models import (
    BlockSpec,
    EdgeJumpKind,
    FunctionBounds,
)
from bingraph.cfg.decode import (
    alternate_block_entry_rejoin_addr,
    decode_bounded_block,
    decode_raw_capstone_insns,
    is_valid_block_entry,
    target_is_known_nonreturning,
)
from bingraph.helpers.capstone import arch_has_delay_slot
from bingraph.helpers.symbols import plt_symbol_name

from .anomalies import find_custom_cfg_anomalies
from .data import StaticDataRegions
from .exceptions import ExceptionalCallSite, exceptional_call_sites_for_function
from .models import (
    CustomCFG,
    CustomCFGNode,
    CustomCFGStats,
    CustomCFGSummary,
)
from .shared_table_proof import (
    shared_register_targets,
    shared_table_targets,
    table_predecessor_facts,
)
from .sweep import (
    SweepBudgetExceeded,
    recover_executable_components,
    select_disconnected_components,
    validate_disconnected_baseline,
)
from .syscalls import ResolvedSyscall, resolve_static_syscall, unknown_syscall_target


_UNRESOLVABLE_CALL_ADDR = 0xFFFFFFFFFFFFFFD0
_UNRESOLVABLE_ENTRY_ADDR = 0xFFFFFFFFFFFFFFC0


def _thumb_mode(project: Project, addr: int) -> bool:
    """Return the execution mode encoded by an ARM/Thumb address."""

    try:
        return bool(project.arch.is_thumb(addr))
    except AttributeError:
        return False


def _node_name(bounds: FunctionBounds, addr: int) -> str:
    """Return the stable display name used for a recovered block."""

    if addr == bounds.addr:
        return bounds.name
    return f"{bounds.name}+0x{addr - bounds.addr:x}"


def _local_node_names(project: Project, addrs: Iterable[int]) -> dict[int, str]:
    """Choose local code labels at existing block starts, never new leaders.

    Untyped assembler labels precede local function aliases, then names sort
    lexically so symbol-table order cannot change the title. Object, section,
    and file symbols are not code labels; ARM mapping symbols only describe
    decoding regions. Normalize Thumb execution addresses to their byte address.
    """

    starts = {addr: addr - int(_thumb_mode(project, addr)) for addr in addrs}
    byte_addrs = set(starts.values())
    aliases: dict[int, tuple[bool, str]] = {}
    arm = project.arch.name.startswith("ARM") or project.arch.name == "AARCH64"
    for symbol in project.loader.main_object.symbols:
        if (
            not symbol.is_local
            or symbol.is_import
            or symbol.type not in (SymbolType.TYPE_NONE, SymbolType.TYPE_FUNCTION)
            or not symbol.name
        ):
            continue
        addr = symbol.rebased_addr
        addr -= int(_thumb_mode(project, addr))
        if addr not in byte_addrs or (
            arm and re.fullmatch(r"\$[adtx](?:\..*)?", symbol.name)
        ):
            continue
        rank = (symbol.is_function, symbol.name)
        if addr not in aliases or rank < aliases[addr]:
            aliases[addr] = rank
    return {
        addr: aliases[byte_addr][1]
        for addr, byte_addr in starts.items()
        if byte_addr in aliases
    }


def _make_block_node(
    model: CFGModel,
    project: Project,
    func_addr: int,
    bounds: FunctionBounds,
    block: BlockSpec,
    *,
    name: str | None = None,
) -> CFGNode:
    """Materialize one recovered normal CFG node."""

    return CustomCFGNode(
        block.addr,
        block.size,
        cfg=model,
        function_address=func_addr,
        block_id=block.addr,
        instruction_addrs=block.instruction_addrs,
        thumb=_thumb_mode(project, block.addr),
        name=name or _node_name(bounds, block.addr),
        vex_linear_instruction_sizes=dict(block.vex_linear_instruction_sizes),
    )


def _external_target_name(project: Project, addr: int) -> str:
    """Return the loader symbol name for an external target when known."""

    symbol = project.loader.find_symbol(addr)
    name = getattr(symbol, "name", None)
    if isinstance(name, str) and name:
        return name
    return plt_symbol_name(project, addr) or f"ExternalTarget_{addr:#x}"


def _make_leaf_node(
    model: CFGModel,
    func_addr: int,
    addr: int,
    name: str,
    *,
    is_syscall: bool = False,
) -> CFGNode:
    """Materialize a synthetic CFG leaf without making it a function block."""

    return CFGNode(
        addr,
        0,
        cfg=model,
        function_address=func_addr,
        block_id=addr,
        instruction_addrs=(),
        simprocedure_name=name,
        is_syscall=is_syscall,
        name=name,
    )


class _BuildSession:
    """Own the leader worklist and graph materialization for one function."""

    def __init__(
        self,
        project: Project,
        kb: KnowledgeBase,
        func_addr: int,
    ) -> None:
        self.project = project
        self.kb = kb
        self.func_addr = func_addr
        self.bounds = _lookup_function_bounds(project, func_addr)
        manager = SimpleNamespace(_kb=kb)
        self.model = CFGModel("CFGCustom", cfg_manager=manager)
        self.graph = cast(CFGGraph, self.model.graph)
        self.stats = CustomCFGStats()
        self.summary = CustomCFGSummary()
        self.leaders = {func_addr}
        self.rejected_leaders: set[int] = set()
        self.pending = deque([func_addr])
        self.pending_addrs = {func_addr}
        self.blocks: dict[int, BlockSpec] = {}
        self.data_regions = StaticDataRegions()
        self.data_regions.claim_code(project, func_addr)
        self.leaf_nodes: dict[tuple[int, str], CFGNode] = {}
        self.static_targets: dict[int, tuple[int, ...]] = {}
        self.exceptional_targets: dict[int, tuple[int, ...]] = {}
        self.continued_linear_direct_transfers: set[int] = set()
        self.resolved_syscalls: dict[int, ResolvedSyscall] = {}
        self._abi_analysis_blocks: dict[int, BlockSpec] | None = None
        self._shared_register_blocks: dict[int, tuple[BlockSpec, BlockSpec]] = {}
        # Presentation-only code must never enter _analysis_graph or its facts.
        self.recovered_blocks: dict[int, BlockSpec] = {}
        self.recovery_baseline: dict[int, BlockSpec] = {}
        self.recovery_source_addrs: dict[int, int] = {}
        self.recovered_roots: frozenset[int] = frozenset()
        self.recovery_table_bytes: set[int] = set()

    def _is_data_leader(self, addr: int) -> bool:
        """Return whether this prospective leader is VEX-proven data."""

        return self.data_regions.contains(self.project, addr)

    def _claim_code_target(self, addr: int) -> None:
        """Give direct control flow precedence over a data classification."""

        self.data_regions.claim_code(self.project, addr)
        self.rejected_leaders.discard(addr)

    def _reject_data_leader(self, addr: int) -> None:
        """Reject a literal-pool leader and remove impossible call returns."""

        self._reject_leader(addr, reason="data")
        self.stats.data_leaders_rejected += 1
        for start, block in tuple(self.blocks.items()):
            if block.jumpkind != "Ijk_Call" or block.fallthrough_addr != addr:
                continue
            self.blocks[start] = replace(block, fallthrough_addr=None)
            self.stats.post_decode_call_fallthroughs_suppressed += 1

    def _queue(self, addr: int) -> None:
        """Schedule one in-bounds block leader only once per pending round."""

        if not self.bounds.addr <= addr < self.bounds.end_addr:
            return
        if addr in self.pending_addrs:
            return
        self.pending.append(addr)
        self.pending_addrs.add(addr)

    def _reject_leader(self, addr: int, *, reason: str = "invalid_entry") -> None:
        """Forget an unsafe target that lands inside a decoded instruction."""

        if reason == "invalid_entry" and addr not in self.rejected_leaders:
            self.stats.leaders_rejected_invalid_entry += 1
        self.rejected_leaders.add(addr)
        self.leaders.discard(addr)
        self.blocks.pop(addr, None)
        self.static_targets.pop(addr, None)

    def _truncate_blocks_at_data(self) -> None:
        """Re-decode blocks that reached a newly proven data range."""

        for start, block in tuple(self.blocks.items()):
            if not any(
                self.data_regions.contains(self.project, insn_addr)
                for insn_addr in block.instruction_addrs[1:]
            ):
                continue
            del self.blocks[start]
            self.stats.block_redecodes += 1
            self.stats.blocks_redecoded_for_data += 1
            self._queue(start)

    def _add_leader(self, addr: int) -> bool:
        """Record one safe target and re-split only real instruction boundaries.

        Targets inside an instruction are never normal block leaders. The only
        supported exception is an alternate stream whose first instruction
        rejoins at the original instruction's end, avoiding duplicated tails.
        """

        if not self.bounds.addr <= addr < self.bounds.end_addr:
            return False
        if self._is_data_leader(addr):
            self._reject_data_leader(addr)
            return False
        self.rejected_leaders.discard(addr)

        covering_blocks = [
            block
            for start, block in self.blocks.items()
            if start < addr < start + block.size
        ]
        if covering_blocks and not all(
            is_valid_block_entry(self.project, block, addr) for block in covering_blocks
        ):
            self._reject_leader(addr)
            return False

        is_new_leader = addr not in self.leaders
        self.leaders.add(addr)
        if is_new_leader:
            self.stats.additional_leaders_discovered += 1
            for start, block in tuple(self.blocks.items()):
                if start < addr < start + block.size:
                    rejoin_addr = alternate_block_entry_rejoin_addr(
                        self.project, block, addr
                    )
                    if rejoin_addr is not None:
                        self._add_leader(rejoin_addr)
                        continue
                    del self.blocks[start]
                    self.stats.block_redecodes += 1
                    self.stats.blocks_redecoded_for_leader_split += 1
                    self.stats.leaders_split_existing_block += 1
                    self._queue(start)
        if is_new_leader or addr not in self.blocks:
            self._queue(addr)
        return True

    def _record_linear_direct_transfer(self, addr: int) -> None:
        """Count each direct next-instruction transfer kept within its block."""

        if addr not in self.continued_linear_direct_transfers:
            self.continued_linear_direct_transfers.add(addr)
            self.stats.linear_direct_transfers_continued += 1

    def _record_vex_linear_fallback(self, _addr: int) -> None:
        """Count a valid linear instruction unavailable from Capstone."""

        self.stats.vex_linear_fallbacks += 1

    def _resolve_syscall(self, block: BlockSpec) -> BlockSpec:
        """Apply a locally proven syscall target before discovering successors."""

        if block.jumpkind != "Ijk_Syscall":
            return block
        self.stats.static_syscall_resolution_attempts += 1
        resolved = resolve_static_syscall(self.project, block)
        if resolved is None:
            self.resolved_syscalls.pop(block.addr, None)
            return block

        self.resolved_syscalls[block.addr] = resolved
        self.stats.static_syscalls_resolved += 1
        if resolved.no_return and block.fallthrough_addr is not None:
            self.stats.static_syscall_fallthroughs_suppressed += 1
            return replace(block, fallthrough_addr=None)
        return block

    def _decode_all_blocks(self) -> None:
        """Drain discovered leaders until their block boundaries stabilize."""

        while self.pending:
            addr = self.pending.popleft()
            self.pending_addrs.remove(addr)
            if addr in self.rejected_leaders:
                continue
            if self._is_data_leader(addr):
                logger.info(
                    f"Custom CFG rejected literal-pool leader {addr:#x} for "
                    f"function {self.func_addr:#x}"
                )
                self._reject_data_leader(addr)
                continue
            block = decode_bounded_block(
                self.project,
                self.bounds,
                addr,
                self.leaders - {addr},
                preserve_conditional_return_fallthrough=True,
                split_syscall_blocks=True,
                resolve_declared_nonreturning=True,
                resolve_static_memory_calls=True,
                split_unclassified_indirect_vex_transfers=True,
                allow_vex_linear_fallback=True,
                on_linear_direct_transfer=self._record_linear_direct_transfer,
                on_vex_linear_fallback=self._record_vex_linear_fallback,
                stop_at_data=lambda target: self.data_regions.contains(
                    self.project, target
                ),
            )
            if block is None:
                self.stats.decode_failures += 1
                logger.warning(
                    f"Custom CFG could not decode block at {addr:#x} for "
                    f"function {self.func_addr:#x}"
                )
                continue
            if block.jumpkind == "Ijk_Ret" and self.project.arch.name == "S390X":
                node = _make_block_node(
                    self.model, self.project, self.func_addr, self.bounds, block
                )
                if s390_table_loaded_branch(node):
                    block = replace(block, jumpkind="Ijk_Boring")
            block = self._resolve_syscall(block)

            alternate_rejoins: set[int] = set()
            for leader in tuple(self.leaders):
                if not block.addr < leader < block.addr + block.size:
                    continue
                if leader in block.instruction_addrs:
                    continue
                rejoin_addr = alternate_block_entry_rejoin_addr(
                    self.project, block, leader
                )
                if rejoin_addr is None:
                    self._reject_leader(leader)
                else:
                    alternate_rejoins.add(rejoin_addr)

            covering_blocks = [
                (start, known_block)
                for start, known_block in self.blocks.items()
                if start < addr < start + known_block.size
            ]
            if any(
                not is_valid_block_entry(self.project, known_block, addr)
                for _, known_block in covering_blocks
            ):
                self._reject_leader(addr)
                continue
            for start, known_block in covering_blocks:
                if (
                    alternate_block_entry_rejoin_addr(self.project, known_block, addr)
                    is not None
                ):
                    continue
                del self.blocks[start]
                self.stats.block_redecodes += 1
                self.stats.blocks_redecoded_for_leader_split += 1
                self.stats.leaders_split_existing_block += 1
                self._queue(start)

            for target in block.direct_targets:
                self._claim_code_target(target)
            self.blocks[addr] = block
            data_bytes_before = len(self.data_regions.data_bytes)
            if self.data_regions.record_block(self.project, block):
                self.stats.data_region_observations += 1
                self.stats.data_bytes_discovered += (
                    len(self.data_regions.data_bytes) - data_bytes_before
                )
                self._truncate_blocks_at_data()
            self.stats.blocks_decoded += 1
            for rejoin_addr in alternate_rejoins:
                self._add_leader(rejoin_addr)
            for target in block.direct_targets:
                self._add_leader(target)
            if block.fallthrough_addr is not None:
                self._add_leader(block.fallthrough_addr)

    def _analysis_graph(
        self, static_targets: Mapping[int, tuple[int, ...]] | None = None
    ) -> tuple[CFGGraph, dict[int, CFGNode]]:
        """Build the exact-flow snapshot used by indirect-target proofs.

        The snapshot includes decoded direct/fallthrough edges and indirect
        edges proven by an earlier discovery round. It intentionally excludes
        unresolved and recovery edges: they cannot establish a must-reaching
        dataflow fact for a later resolver.
        """

        graph = cast(CFGGraph, nx.DiGraph())
        static_targets = static_targets or {}
        nodes = {
            addr: _make_block_node(
                self.model, self.project, self.func_addr, self.bounds, block
            )
            for addr, block in self.blocks.items()
        }
        for node in nodes.values():
            graph.add_node(node)
        for addr, block in self.blocks.items():
            source = nodes[addr]
            # Previously proven indirect targets become normal flow for later
            # proofs. A selector can survive an exact dispatcher before a
            # second table, or its carried target can feed a register jump.
            for target in (
                *block.direct_targets,
                *static_targets.get(addr, ()),
                block.fallthrough_addr,
            ):
                if target is None or target not in nodes:
                    continue
                add_successor_edge(
                    graph,
                    source,
                    nodes[target],
                    "Ijk_Call" if block.jumpkind == "Ijk_Call" else "Ijk_Boring",
                )
                if target in static_targets.get(addr, ()):
                    # Exact indirect edges constrain the jump-carrying register
                    # to this destination. Keep that provenance local to the
                    # immutable analysis snapshot, never on recovery edges.
                    edge = graph.get_edge_data(source, nodes[target])
                    if edge is not None:
                        edge["proven_dispatch"] = True
        return graph, nodes

    def _call_block_matches_lsda_site(
        self, block: BlockSpec, site: ExceptionalCallSite
    ) -> bool:
        """Match a recovered call instruction, excluding any delay slot."""

        if block.jumpkind != "Ijk_Call":
            return False
        insns = decode_raw_capstone_insns(self.project, block.addr, block.size)
        if not insns or insns[-1].address + insns[-1].size != block.addr + block.size:
            return False
        # The block's call classification can come from VEX when Capstone's
        # call group misses PPC branches, s390 BRASL, or MIPS BAL. MIPS also
        # includes one delay-slot instruction after the call.
        if arch_has_delay_slot(self.project.arch.name):
            if len(insns) < 2:
                return False
            call = insns[-2]
        else:
            call = insns[-1]
        return (
            site.start_addr <= call.address
            and call.address + call.size <= site.end_addr
        )

    def _discover_elf_exceptional_edges(self) -> None:
        """Queue LSDA-proven in-function landing pads through a small fixpoint.

        A landing pad may contain another call with its own cleanup action, so
        new leaders are decoded before their call-site records are considered.
        The LSDA is the sole authority here: ordinary calls without a matching
        record never gain an exceptional edge.
        """

        self.stats.exception_metadata_functions_scanned += 1
        call_sites = exceptional_call_sites_for_function(self.project, self.bounds)
        self.stats.exception_call_sites_discovered += len(call_sites)
        if not call_sites:
            return

        while True:
            targets_by_source: dict[int, set[int]] = {}
            for block in tuple(self.blocks.values()):
                for site in call_sites:
                    if not self._call_block_matches_lsda_site(block, site):
                        continue
                    target = site.landing_pad_addr
                    if not self.bounds.addr <= target < self.bounds.end_addr:
                        continue
                    self._claim_code_target(target)
                    self._add_leader(target)
                    if target not in self.leaders:
                        continue
                    targets_by_source.setdefault(block.addr, set()).add(target)
            if not self.pending:
                self.exceptional_targets = {
                    source: tuple(sorted(targets))
                    for source, targets in targets_by_source.items()
                }
                self.stats.exceptional_transfers_discovered = sum(
                    len(targets) for targets in targets_by_source.values()
                )
                return
            self._decode_all_blocks()

    def _restore_shared_register_blocks(self, addresses: Iterable[int]) -> None:
        """Withdraw a proof and its no-return side effect, not decoded facts."""

        for addr in addresses:
            original, proved = self._shared_register_blocks.pop(addr)
            if self.blocks.get(addr) != proved:
                continue
            if original.jumpkind == "Ijk_Call":
                self.stats.abi_static_call_targets_resolved -= 1
                self.stats.post_decode_call_fallthroughs_suppressed -= (
                    original.fallthrough_addr is not None
                    and proved.fallthrough_addr is None
                )
            else:
                self.stats.abi_static_jump_targets_resolved -= 1
            self.blocks[addr] = original

    def _resolve_abi_static_register_transfers(
        self, static_targets: Mapping[int, tuple[int, ...]] | None = None
    ) -> None:
        """Resolve carried targets over decoded flow and already-exact tables.

        Table recovery can expose predecessors carrying different static
        callees into a common epilogue. Give the ABI solvers those exact edges,
        but never speculative recovery edges. Cache the immutable input snapshot
        so an unchanged discovery round does not repeat the bounded analysis.
        Shared facts resolve calls on any architecture and jumps on AMD64.
        Only known ABIs preserve register facts across calls; local proofs
        need no preservation rule, even on architectures without an ABI adapter.
        MIPS also retains its independent private-frame analysis.
        Register proofs belong to their incoming-flow snapshot, not just their
        source block. Rebuild them from decoded facts when that snapshot changes.
        """

        originals = {
            addr: original
            for addr, (original, proved) in self._shared_register_blocks.items()
            if self.blocks.get(addr) == proved
        }
        inputs = self.blocks | originals
        snapshot = {
            addr: replace(block, direct_targets=static_targets[addr])
            if static_targets and addr in static_targets
            else block
            for addr, block in inputs.items()
        }
        if snapshot == self._abi_analysis_blocks:
            return
        self._abi_analysis_blocks = snapshot
        # Restore even the original call continuation: an invalidated proof
        # must not keep its nonreturning side effect or prove itself via its edge.
        self._restore_shared_register_blocks(tuple(self._shared_register_blocks))
        invalidated: set[int] = set()
        conflicts: set[int] = set()
        while True:
            analysis_blocks = {
                addr: replace(block, direct_targets=static_targets[addr])
                if static_targets and addr in static_targets
                else block
                for addr, block in self.blocks.items()
            }
            targets = {}
            lost: set[int] = set()
            shared_ran = False
            candidates = {
                addr
                for addr, block in analysis_blocks.items()
                if block.jumpkind in {"Ijk_Boring", "Ijk_Call"}
                and not block.direct_targets
            }
            shared_candidates = {
                addr
                for addr in candidates
                if self.project.arch.name == "AMD64"
                or analysis_blocks[addr].jumpkind == "Ijk_Call"
            }
            if shared_candidates or self._shared_register_blocks:
                graph, nodes = self._analysis_graph(static_targets)
                # A no-return conclusion must not remove a path while proving
                # its own callee. Query with the decoded continuations restored.
                for addr, (original, _) in self._shared_register_blocks.items():
                    if original.fallthrough_addr in nodes:
                        graph.add_edge(
                            nodes[addr],
                            nodes[original.fallthrough_addr],
                            jumpkind="Ijk_Boring",
                        )
                facts = table_predecessor_facts(self.project, graph, self.bounds)
                shared_ran = True
                for addr in sorted(
                    (shared_candidates | self._shared_register_blocks.keys())
                    - invalidated
                ):
                    exact = shared_register_targets(self.project, nodes[addr], facts)
                    if exact is not None:
                        targets[addr] = exact
                self.stats.shared_fact_steps += facts.steps
                self.stats.shared_fact_budget_exhausted += facts.exhausted
                self.stats.shared_target_rejection_attempts_by_reason = dict(
                    Counter(self.stats.shared_target_rejection_attempts_by_reason)
                    + facts.target_rejections
                )
                # Newly proved jumps also change incoming flow. Recheck earlier
                # answers, and do not oscillate by reinstating a withdrawn proof
                # after its supporting edge disappears in this same snapshot.
                lost = self._shared_register_blocks.keys() - targets.keys()
                changed = {
                    addr
                    for addr, (_, proved) in self._shared_register_blocks.items()
                    if addr in targets and proved.direct_targets != targets[addr]
                }
                targets = {
                    addr: exact
                    for addr, exact in targets.items()
                    if addr not in self._shared_register_blocks or addr in changed
                }
                self._restore_shared_register_blocks(lost | changed)
                invalidated.update(lost)
            shared_sources = set(targets)
            if self.project.arch.name.startswith("MIPS") and candidates:
                fallback, exhausted, ran = abi_static_register_transfer_targets(
                    self.project, self.bounds, analysis_blocks
                )
                shared_ran |= ran
                self.stats.abi_static_target_analysis_budget_exhausted += exhausted
                if not exhausted:
                    # Private-frame proofs have their own lifetime. Shared
                    # facts cannot revalidate spills, so track only their own
                    # answers. Conflicting exact proofs must fail closed.
                    conflicts.update(
                        addr
                        for addr in targets.keys() & fallback.keys()
                        if set(targets[addr]) != set(fallback[addr])
                    )
                    targets = {
                        addr: exact
                        for addr, exact in (fallback | targets).items()
                        if addr not in conflicts
                    }
                    shared_sources -= fallback.keys()
                    invalidated.update(conflicts)
            self.stats.abi_static_target_analysis_runs += shared_ran
            if not targets:
                if lost:
                    continue
                return

            discovered = False
            for addr, exact_targets in targets.items():
                block = self.blocks[addr]
                fallthrough = block.fallthrough_addr
                if block.jumpkind == "Ijk_Call" and all(
                    target_is_known_nonreturning(self.project, target)
                    for target in exact_targets
                ):
                    fallthrough = None
                    if block.fallthrough_addr is not None:
                        self.stats.post_decode_call_fallthroughs_suppressed += 1
                self.blocks[addr] = replace(
                    block, direct_targets=exact_targets, fallthrough_addr=fallthrough
                )
                if addr in shared_sources:
                    self._shared_register_blocks[addr] = block, self.blocks[addr]
                if block.jumpkind == "Ijk_Call":
                    self.stats.abi_static_call_targets_resolved += 1
                else:
                    self.stats.abi_static_jump_targets_resolved += 1
                for target in exact_targets:
                    if self.bounds.addr <= target < self.bounds.end_addr:
                        self._claim_code_target(target)
                        before = target in self.blocks or target in self.pending_addrs
                        if self._add_leader(target):
                            discovered |= not before
            if discovered:
                self._decode_all_blocks()
            # Each shared round either establishes a new site or permanently
            # withdraws one for this snapshot; copy cycles are never unrolled.

    def _recognize_saved_link_returns(self) -> None:
        """Recognize split ARM pops of the entry LR in an unchanged leaf frame.

        This is a provenance proof, not a guess based on a jump register's
        name. An unconditional entry push creates a private frame; a suffix
        of unconditional pops must restore the entire frame and load its LR
        slot into the final BX register. Other SP uses (including exporting a
        frame pointer), calls, IT predication, and unresolved intervening flow
        make this deliberately small proof inconclusive.

        As with private stack saves in the ABI target pass, ordinary argument
        pointers are assumed not to alias a newly allocated, unescaped frame.
        The scan is bounded and runs only for ARM functions with unknown exits.
        """

        if not self.project.arch.name.startswith("ARM"):
            return
        candidates = {
            addr
            for addr, block in self.blocks.items()
            if block.jumpkind == "Ijk_Boring"
            and not block.direct_targets
            and addr not in self.static_targets
            and block.fallthrough_addr is None
        }
        if not candidates:
            return
        entry = self.blocks[self.bounds.addr]
        entry_insns = decode_raw_capstone_insns(self.project, entry.addr, entry.size)
        if not entry_insns:
            return
        push = entry_insns[0]
        if push.id != ARM_INS_PUSH or push.cc != ARM_CC_AL:
            return
        saved = [op.reg for op in push.operands if op.type == CS_OP_REG]
        if ARM_REG_LR not in saved or {ARM_REG_SP, ARM_REG_PC}.intersection(saved):
            return

        graph, nodes = self._analysis_graph(self.static_targets)
        root = nodes[self.bounds.addr]
        reachable = {root, *nx.descendants(graph, root)}
        if graph.in_degree(root) or any(
            self.blocks[node.addr].jumpkind in {"Ijk_Call", "Ijk_Syscall"}
            or (
                self.blocks[node.addr].jumpkind == "Ijk_Boring"
                and not self.blocks[node.addr].direct_targets
                and node.addr not in self.static_targets
                and node.addr not in candidates
            )
            for node in reachable
        ):
            return
        # An undecodable internal successor could alter the frame before
        # rejoining an epilogue; a partial exact graph cannot prove a return.
        if any(
            target is not None
            and self.bounds.addr <= target < self.bounds.end_addr
            and target not in self.blocks
            for node in reachable
            for target in (
                *self.blocks[node.addr].direct_targets,
                *self.static_targets.get(node.addr, ()),
                self.blocks[node.addr].fallthrough_addr,
            )
        ):
            return
        insns = {}
        count = 0
        for node in reachable:
            block = self.blocks[node.addr]
            decoded = decode_raw_capstone_insns(self.project, block.addr, block.size)
            count += len(decoded)
            if (
                not decoded
                or count > 20000
                or decoded[0].address != block.addr
                or decoded[-1].address + decoded[-1].size != block.addr + block.size
            ):
                return
            insns[node.addr] = decoded

        epilogue_addrs = set()
        returns = set()
        for addr in candidates & insns.keys():
            decoded = insns[addr]
            branch = decoded[-1]
            if (
                branch.id != ARM_INS_BX
                or branch.cc != ARM_CC_AL
                or len(branch.operands) != 1
                or branch.operands[0].type != CS_OP_REG
            ):
                return
            start = len(decoded) - 1
            while start and decoded[start - 1].id == ARM_INS_POP:
                start -= 1
            consumed = 0
            restored_link = None
            for pop in decoded[start:-1]:
                registers = [op.reg for op in pop.operands if op.type == CS_OP_REG]
                if pop.cc != ARM_CC_AL or {ARM_REG_SP, ARM_REG_PC}.intersection(
                    registers
                ):
                    return
                if restored_link in registers:
                    restored_link = None
                slot = saved.index(ARM_REG_LR) - consumed
                if 0 <= slot < len(registers):
                    restored_link = registers[slot]
                consumed += len(registers)
            if consumed != len(saved) or restored_link != branch.operands[0].reg:
                return
            epilogue_addrs.update(insn.address for insn in decoded[start:-1])
            returns.add(addr)

        for decoded in insns.values():
            for insn in decoded:
                if insn.address == push.address or insn.address in epilogue_addrs:
                    continue
                reads, writes = insn.regs_access()
                if insn.id == ARM_INS_IT or ARM_REG_SP in (*reads, *writes):
                    return
        for addr in returns:
            self.blocks[addr] = replace(self.blocks[addr], jumpkind="Ijk_Ret")

    def _discover_static_jump_targets(self) -> None:
        """Iteratively discover exact indirect jump targets.

        Each changed exact-flow snapshot first feeds shared register facts
        and the remaining ABI fallback. Conditional-PC forms remain specialized.
        Shared finite table facts then get first refusal, before generic VEX
        tables and architecture adapters. Shared register facts handle remaining
        non-table transfers. The shared table proof receives only the exact-flow
        snapshot, never a legacy table plan or selector domain; successful
        proofs bypass legacy planning entirely.
        Only a fully bounded target set becomes ``static_targets``. Unbounded
        selectors retain their unresolved target; executable regions discovered
        later are presented with unknown entry, not guessed dispatcher edges.

        Each exact target is added as a leader. Because a new leader can split
        a block, plans are re-evaluated until the decode stabilizes. Plans from
        an earlier round survive only when their source ``BlockSpec`` is
        unchanged, preserving their proof while avoiding stale source edges.
        """

        retained_plans: dict[int, tuple[BlockSpec, tuple[int, ...], str]] = {}
        while True:
            retained_targets = {
                addr: targets
                for addr, (source, targets, _flavor) in retained_plans.items()
                if self.blocks.get(addr) == source
            }
            self._resolve_abi_static_register_transfers(retained_targets)
            # ABI-proven leaders can split a retained table's source. Only
            # unchanged sources may contribute exact edges to this round.
            retained_targets = {
                addr: targets
                for addr, targets in retained_targets.items()
                if self.blocks.get(addr) == retained_plans[addr][0]
            }
            retained_flavors = {
                addr: flavor
                for addr, (source, _targets, flavor) in retained_plans.items()
                if self.blocks.get(addr) == source
            }
            graph, nodes = self._analysis_graph(retained_targets)
            shared_facts = None
            discovered = False
            plans: dict[int, tuple[int, ...]] = {}
            plan_flavors: dict[int, str] = {}
            conditional_sources: set[int] = set()
            for addr, node in nodes.items():
                # Adding a target can split a later block from this snapshot.
                # Skip its now-stale node; the next round analyzes its decode.
                block = self.blocks.get(addr)
                if block is None:
                    continue
                if block.jumpkind != "Ijk_Boring" or block.direct_targets:
                    continue
                self.stats.static_jump_plan_attempts += 1
                # Exact resolvers share one leader worklist. A successful
                # plan can reveal code needed by a later resolver round.
                targets, reason = conditional_pc_dispatch_targets(
                    self.project,
                    self.bounds,
                    node,
                    table_data=self.recovery_table_bytes,
                )
                proof_flavor = "conditional_pc" if targets is not None else None
                plan = None
                if targets is not None:
                    conditional_sources.add(addr)
                if (
                    targets is None
                    and reason in {"not_conditional_pc", "no_vex"}
                    and node_vex(node) is not None
                ):
                    # Migrate complete single-level proofs before consulting
                    # the legacy recognizers. Unknown proofs still fall back,
                    # but those attempts and final winning flavors are visible
                    # in statistics rather than silently masking coverage gaps.
                    # One cache/budget belongs to this immutable graph round;
                    # neither plans nor facts survive a source block split.
                    if shared_facts is None:
                        shared_facts = table_predecessor_facts(
                            self.project, graph, self.bounds
                        )
                    self.stats.shared_table_attempts += 1
                    targets = shared_table_targets(self.project, node, shared_facts)
                    if targets is not None:
                        reason = None
                        proof_flavor = "shared_finite_table"
                if targets is None and reason in {"not_conditional_pc", "no_vex"}:
                    self.stats.legacy_table_fallback_attempts += 1
                    plan, reason = plan_static_jump_table(
                        self.project,
                        graph,
                        self.bounds,
                        node,
                        allow_inline_index_values=True,
                        allow_masked_index_values=True,
                        allow_static_bases=True,
                        allow_guarded_expression_indices=True,
                        allow_predecessor_clamped_indices=True,
                    )
                    if plan is None and reason == "no_table_shape":
                        plan, reason = plan_mips_pic_relative_jump_table(
                            self.project,
                            graph,
                            self.bounds,
                            node,
                            allow_predecessor_static_base=True,
                        )
                    if plan is not None:
                        targets = _read_static_jump_table_targets(
                            self.project,
                            plan.table,
                            plan.base_addr,
                            plan.entry_indices,
                        )
                        if targets is not None:
                            proof_flavor = plan.proof_flavor
                if targets is None and reason == "no_table_shape":
                    # The table query already ran against this snapshot. Do not
                    # repeat it after legacy planning or borrow its bounds.
                    # A previous table edge may carry its destination in a
                    # register which is adjusted before another jump. Reuse
                    # the same bounded facts instead of adding an ISA rule.
                    if shared_facts is None:
                        shared_facts = table_predecessor_facts(
                            self.project, graph, self.bounds
                        )
                    targets = shared_register_targets(self.project, node, shared_facts)
                    if targets is not None:
                        reason = None
                        proof_flavor = "shared_finite_register"
                if targets is None:
                    if reason == "no_table_shape" and is_memory_dependent_indirect_jump(
                        node
                    ):
                        # Preserve the historical diagnostic name for targets
                        # whose memory dependency prevents an exact proof.
                        reason = "dynamic_memory_target"
                    self.stats.static_jump_unresolved_dispatcher_attempts += 1
                    if reason is not None:
                        field = f"static_jump_{reason}"
                        if hasattr(self.stats, field):
                            setattr(self.stats, field, getattr(self.stats, field) + 1)
                    continue
                self.stats.static_jump_target_candidate_attempts += len(targets)
                rejected = [
                    static_jump_target_rejection_reason(self.project, target)
                    for target in targets
                    if not self.bounds.addr <= target < self.bounds.end_addr
                ]
                if any(rejected):
                    self.stats.static_jump_external_target_rejection_attempts += sum(
                        reason is not None for reason in rejected
                    )
                    continue
                accepted_targets: list[int] = []
                for target in targets:
                    if not self.bounds.addr <= target < self.bounds.end_addr:
                        accepted_targets.append(target)
                        continue
                    self._claim_code_target(target)
                    before = target in self.blocks or target in self.pending_addrs
                    added = self._add_leader(target)
                    if added or target in self.blocks or target in self.pending_addrs:
                        accepted_targets.append(target)
                        discovered |= added and not before
                plans[addr] = tuple(accepted_targets)
                # These bytes are evidence of data, not additional bounds
                # or targets. Keep them out of normal construction decisions.
                if plan is not None:
                    table_addr = _jump_table_addr(plan.base_addr, plan.table)
                    for index in plan.entry_indices:
                        address = table_addr + index * plan.table.entry_size
                        self.recovery_table_bytes.update(
                            range(address, address + plan.table.entry_size)
                        )
                if shared_facts is not None:
                    for address, size, _endness, _steps in shared_facts._table_rows:
                        self.recovery_table_bytes.update(range(address, address + size))
                assert proof_flavor is not None
                plan_flavors[addr] = proof_flavor
                self.stats.static_jump_targets_accepted += len(accepted_targets)

            if shared_facts is not None:
                self.stats.shared_fact_steps += shared_facts.steps
                self.stats.shared_fact_budget_exhausted += shared_facts.exhausted
                self.stats.shared_target_rejection_attempts_by_reason = dict(
                    Counter(self.stats.shared_target_rejection_attempts_by_reason)
                    + shared_facts.target_rejections
                )
            exact_edges_changed = any(
                targets and retained_targets.get(addr) != targets
                for addr, targets in plans.items()
            )
            proof_sources = dict(self.blocks)
            abi_changed = False
            if exact_edges_changed and not discovered:
                # Existing blocks can gain new exact predecessors without any
                # new leaders. Revisit ABI facts, but repeat table planning
                # only if that analysis actually changes the decoded flow.
                self._resolve_abi_static_register_transfers(retained_targets | plans)
                abi_changed = self.blocks != proof_sources
            if discovered or abi_changed:
                for addr, targets in plans.items():
                    source = proof_sources.get(addr)
                    if source is not None and targets:
                        retained_plans[addr] = (
                            source,
                            targets,
                            plan_flavors[addr],
                        )
                # Block splits invalidate plans built from this graph snapshot.
                # Retain a proof only if its source block is unchanged after
                # rebuilding. A split can otherwise orphan its target leaders.
                if discovered:
                    self.stats.static_jump_plans_in_discovery_rounds += len(plans)
                    self._decode_all_blocks()
                # Even when all table destinations already exist as blocks,
                # new exact edges can enable a carried-register ABI proof.
                continue
            retained_targets.update(plans)
            self.static_targets.update(retained_targets)
            final_flavors = retained_flavors | plan_flavors
            self.stats.exact_jump_proofs_by_flavor = dict(
                sorted(
                    Counter(final_flavors[addr] for addr in retained_targets).items()
                )
            )
            self.stats.conditional_pc_dispatches_resolved += len(conditional_sources)
            self.stats.conditional_pc_targets_recovered += sum(
                len(retained_targets[addr])
                for addr in conditional_sources
                if addr in retained_targets
            )
            self.stats.static_jump_plans_resolved += len(retained_targets)
            return

    def _leaf(self, addr: int, name: str, *, is_syscall: bool = False) -> CFGNode:
        """Return a unique synthetic leaf for one address/name pair."""

        key = (addr, name)
        node = self.leaf_nodes.get(key)
        if node is None:
            node = _make_leaf_node(
                self.model,
                self.func_addr,
                addr,
                name,
                is_syscall=is_syscall,
            )
            self.graph.add_node(node)
            self.leaf_nodes[key] = node
            self.stats.synthetic_leaves_created += 1
        else:
            self.stats.synthetic_leaves_reused += 1
        return node

    def _target_node(self, addr: int) -> CFGNode:
        """Return a recovered destination or a precise synthetic leaf."""

        node = self.nodes.get(addr)
        if node is not None:
            return node
        if self.bounds.addr <= addr < self.bounds.end_addr:
            self.stats.undecodable_target_references += 1
            return self._leaf(addr, "UndecodableInstructionTarget")
        self.stats.external_target_references += 1
        return self._leaf(addr, _external_target_name(self.project, addr))

    def _fallthrough_target_node(self, addr: int) -> CFGNode | None:
        """Return a valid continuation without inventing one past bad bytes."""

        node = self.nodes.get(addr)
        if node is not None:
            return node
        if self.bounds.addr <= addr < self.bounds.end_addr:
            # A known branch target may be useful as an explicit undecodable
            # leaf, but normal execution must not fall through to bytes that
            # failed bounded decoding. This matches the custom renderer's
            # existing treatment of invalid sequential continuations.
            return None
        return self._target_node(addr)

    def _materialize_edges(self) -> None:
        """Create final nodes and distinguish exact flow from unresolved flow.

        ``static_targets`` are exact edges produced by the resolver pipeline.
        Any remaining non-call indirect transfer receives one UJT leaf here;
        unknown-entry recovery may later add dashed edges from its own source,
        without guessing dispatcher targets or removing the UJT fallback.
        """

        # Labels affect final titles only, not proof snapshots or block discovery.
        blocks = self._output_blocks()
        local_names = _local_node_names(
            self.project, (addr for addr in blocks if addr != self.func_addr)
        )
        self.nodes = {
            addr: _make_block_node(
                self.model,
                self.project,
                self.func_addr,
                self.bounds,
                block,
                name=local_names.get(addr),
            )
            for addr, block in sorted(blocks.items())
        }
        for node in self.nodes.values():
            self.graph.add_node(node)

        # Exception metadata belongs to the call instruction, which may now
        # terminate a suffix fragment. Exact proof sources were kept pinned.
        exceptional_targets = {
            self.recovery_source_addrs.get(addr, addr): targets
            for addr, targets in self.exceptional_targets.items()
        }
        for addr, block in sorted(self._output_blocks().items()):
            source = self.nodes[addr]
            if block.jumpkind == "Ijk_Syscall":
                syscall_target = self.resolved_syscalls.get(addr)
                if syscall_target is None:
                    syscall_target = unknown_syscall_target(self.project)
                syscall = self._leaf(
                    syscall_target.addr,
                    syscall_target.name,
                    is_syscall=True,
                )
                syscall_jumpkind = cast(
                    EdgeJumpKind, block.syscall_jumpkind or "Ijk_Sys_syscall"
                )
                if add_successor_edge(self.graph, source, syscall, syscall_jumpkind):
                    self.summary.direct_edges += 1

            for target in block.direct_targets:
                destination = self._target_node(target)
                jumpkind = "Ijk_Call" if block.jumpkind == "Ijk_Call" else "Ijk_Boring"
                if add_successor_edge(self.graph, source, destination, jumpkind):
                    self.summary.direct_edges += 1

            for target in self.static_targets.get(addr, ()):
                destination = self._target_node(target)
                if add_successor_edge(self.graph, source, destination, "Ijk_Boring"):
                    self.stats.static_jump_target_edges_added += 1

            for target in exceptional_targets.get(addr, ()):
                destination = self._target_node(target)
                if add_successor_edge(
                    self.graph,
                    source,
                    destination,
                    "Ijk_Boring",
                    exceptional=True,
                ):
                    self.stats.exception_edges_added += 1

            if block.fallthrough_addr is not None:
                destination = self._fallthrough_target_node(block.fallthrough_addr)
                if destination is not None:
                    jumpkind = (
                        "Ijk_FakeRet"
                        if block.jumpkind in {"Ijk_Call", "Ijk_Syscall"}
                        else "Ijk_Boring"
                    )
                    if add_successor_edge(self.graph, source, destination, jumpkind):
                        self.summary.fallthrough_edges += 1

            if block.jumpkind == "Ijk_Call" and not block.direct_targets:
                unresolved = self._leaf(
                    _UNRESOLVABLE_CALL_ADDR,
                    "UnresolvableCallTarget",
                )
                if add_successor_edge(self.graph, source, unresolved, "Ijk_Call"):
                    self.stats.unresolved_call_targets += 1

            if (
                block.jumpkind == "Ijk_Boring"
                and not block.direct_targets
                and not self.static_targets.get(addr)
                and (
                    block.fallthrough_addr is None
                    or (
                        # The false-path continuation does not resolve a
                        # conditional computed-PC branch's taken target.
                        (vex := node_vex(source)) is not None
                        and vex_has_computed_pc_transfer(vex, block.fallthrough_addr)
                        # Caller-dependent return addresses are not UJTs.
                        and not vex_is_conditional_link_return(
                            vex, block.fallthrough_addr
                        )
                    )
                )
            ):
                unresolved = self._leaf(0xFFFFFFFFFFFFFFF0, "UnresolvableJumpTarget")
                if add_successor_edge(
                    self.graph,
                    source,
                    unresolved,
                    "Ijk_Boring",
                    unresolved_indirect=True,
                ):
                    self.stats.unresolved_indirect_targets += 1

    def _recover_disconnected_components(self) -> None:
        """Discover code with unknown entry, not an indirect-target proof.

        Only explicitly sized symbols are scanned. Alignment, execution mode,
        and known data restrict decoding; a 20K callback budget bounds work.
        Known blocks may only split losslessly at instruction boundaries;
        exact proof sources remain pinned. Other changes reject the scan. Regions
        must rejoin established code or have closed direct flow with a known
        return, trap or known non-returning/tail exit. Isolated undecodable or
        padding regions stay hidden.
        """

        if not any(
            block.jumpkind == "Ijk_Boring"
            and not block.direct_targets
            and block.fallthrough_addr is None
            and not self.static_targets.get(addr)
            for addr, block in self.blocks.items()
        ):
            return
        if not any(
            symbol.is_function
            and symbol.rebased_addr == self.bounds.addr
            and symbol.size == self.bounds.size
            and symbol.size > 0
            for symbol in self.project.loader.main_object.symbols
        ):
            return
        code_addrs = {
            address
            for block in self.blocks.values()
            for address in block.instruction_addrs
        }
        thumb = _thumb_mode(self.project, self.func_addr)
        alignment = 2 if thumb else (self.project.arch.instruction_alignment or 1)

        def excluded(address: int) -> bool:
            physical = StaticDataRegions._memory_addr(self.project, address)
            if address in code_addrs:
                return False
            return (
                physical % alignment != 0
                or _thumb_mode(self.project, address) != thumb
                or physical in self.recovery_table_bytes
                or self.data_regions.contains(self.project, address)
            )

        self.stats.disconnected_recovery_runs += 1
        try:
            sweep = recover_executable_components(
                self.project,
                self.bounds,
                self.blocks,
                stop_at_data=excluded,
                max_steps=20_000,
                resolve_static_memory_calls=True,
            )
        except SweepBudgetExceeded:
            self.stats.disconnected_recovery_budget_exhausted += 1
            return
        baseline = validate_disconnected_baseline(
            sweep,
            self.blocks,
            protected_sources=(
                self.static_targets.keys()
                | self.resolved_syscalls.keys()
                | self._shared_register_blocks.keys()
            ),
        )
        if baseline is None:
            self.stats.disconnected_recovery_rejected_changes += 1
            return
        selected = select_disconnected_components(
            self.project, sweep, baseline, bounds=self.bounds
        )
        self.recovered_blocks = dict(selected.blocks)
        self.recovered_roots = selected.roots
        self.stats.disconnected_regions = selected.component_count
        self.stats.disconnected_blocks = len(selected.blocks)
        if selected.blocks:
            self._set_recovery_partition(baseline | self.recovered_blocks)

    def _set_recovery_partition(self, blocks: Mapping[int, BlockSpec]) -> None:
        """Adopt validated display blocks while keeping proof inputs immutable."""

        original_insns = {a for b in self.blocks.values() for a in b.instruction_addrs}
        self.recovery_baseline = {
            addr: block for addr, block in blocks.items() if addr in original_insns
        }
        self.recovered_blocks = {
            addr: block for addr, block in blocks.items() if addr not in original_insns
        }
        terminals = {
            b.addr + b.size: addr for addr, b in self.recovery_baseline.items()
        }
        self.recovery_source_addrs = {
            addr: terminals[b.addr + b.size]
            for addr, b in self.blocks.items()
            if self.recovery_baseline.get(addr) != b
        }

    def _discover_recovered_elf_exceptional_edges(self) -> None:
        """Apply LSDA to displayed calls without feeding recovery into proofs.

        Reuse the ordinary LSDA/direct-decoding worklist on a private snapshot.
        Its fixed point handles calls inside newly decoded landing pads too;
        no indirect resolver runs. Only lossless partitions of the displayed
        code can be adopted, with original exact-proof sources still pinned.
        Metadata gives landing pads known incoming edges, so even a bare jump
        pad is valid here; it does not need an unknown-entry recovery root.
        """

        if not self.recovered_blocks or not exceptional_call_sites_for_function(
            self.project, self.bounds
        ):
            return
        displayed = self._output_blocks()
        late = _BuildSession(self.project, self.kb, self.func_addr)
        late.blocks = dict(displayed)
        late.leaders = set(displayed)
        late.pending.clear()
        late.pending_addrs.clear()
        late.data_regions = deepcopy(self.data_regions)
        late._discover_elf_exceptional_edges()
        validated = validate_disconnected_baseline(
            late.blocks,
            displayed,
            protected_sources=(
                self.static_targets.keys()
                | self.resolved_syscalls.keys()
                | self._shared_register_blocks.keys()
            ),
        )
        if validated is None:
            return
        self._set_recovery_partition(late.blocks)
        self.resolved_syscalls.update(late.resolved_syscalls)
        self.exceptional_targets = late.exceptional_targets
        self.stats.exceptional_transfers_discovered = sum(
            len(targets) for targets in self.exceptional_targets.values()
        )
        self.stats.disconnected_blocks = len(self.recovered_blocks)

    def _output_blocks(self) -> dict[int, BlockSpec]:
        """Return the rendered block partition without changing proof inputs."""

        return self.blocks | self.recovery_baseline | self.recovered_blocks

    def _attach_disconnected_components(self) -> None:
        """Explain discovered regions without inventing a dispatcher edge."""

        if not self.recovered_roots:
            return
        source = self._leaf(_UNRESOLVABLE_ENTRY_ADDR, "UnresolvableEntrySource")
        for address in sorted(self.recovered_roots):
            target = self.nodes[address]
            add_successor_edge(
                self.graph, source, target, "Ijk_Boring", unresolved_indirect=True
            )
            edge = self.graph.get_edge_data(source, target)
            assert edge is not None
            edge["recovered_entry"] = True

    def _summarize_output(self) -> None:
        """Record the final graph shape separately from construction decisions."""

        blocks = self._output_blocks()
        # Count stabilized call sites once, not every decode of their block.
        calls = [block for block in blocks.values() if block.jumpkind == "Ijk_Call"]
        self.stats.decoder_call_targets_by_kind = dict(
            sorted(
                Counter(
                    block.decoded_call_target_kind
                    for block in calls
                    if block.decoded_call_target_kind is not None
                ).items()
            )
        )
        self.stats.decoder_nonreturning_calls = sum(
            block.decoded_nonreturning_call for block in calls
        )
        self.summary.normal_blocks = len(blocks)
        self.summary.synthetic_leaves = len(self.leaf_nodes)
        self.summary.nodes = len(tuple(self.graph.nodes()))
        self.summary.edges = len(tuple(self.graph.edges()))
        for block in blocks.values():
            if block.jumpkind == "Ijk_Call":
                self.summary.calls += 1
            elif block.jumpkind == "Ijk_Syscall":
                self.summary.syscalls += 1
            elif block.jumpkind == "Ijk_Ret":
                self.summary.returns += 1
            elif block.jumpkind == "Ijk_Terminal":
                self.summary.terminal_blocks += 1
            elif block.direct_targets:
                self.summary.direct_branches += 1
                self.summary.conditional_branches += block.fallthrough_addr is not None

        # A dashed recovery edge is not a proven entry path. Count
        # its instructions as discovered, but not entry-connected coverage.
        connected: set[CFGNode] = set()
        pending = (
            deque([self.nodes[self.func_addr]])
            if self.func_addr in self.nodes
            else deque()
        )
        while pending:
            node = pending.popleft()
            if node in connected:
                continue
            connected.add(node)
            for target in self.graph.successors(node):
                data = self.graph.get_edge_data(node, target) or {}
                if not data.get("unresolved_indirect") and not data.get(
                    "recovered_entry"
                ):
                    pending.append(target)
        all_insns = {a for block in blocks.values() for a in block.instruction_addrs}
        connected_insns = {a for node in connected for a in node.instruction_addrs}
        self.summary.discovered_instructions = len(all_insns)
        self.summary.entry_connected_instructions = len(connected_insns & all_insns)
        self.summary.disconnected_instructions = len(all_insns - connected_insns)

    def build(self, *, recover_disconnected: bool = True) -> CustomCFG:
        """Run bounded construction in decode, proof, recovery, render order.

        Exact target discovery precedes any sweep so that a static table never
        depends on speculative recovered code. Rendering is deliberately last:
        it consumes the stabilized block set and records unresolved targets
        that the proof stages intentionally declined to resolve.
        The private ``recover_disconnected=False`` audit hook exposes the
        pre-recovery baseline; public construction always enables this phase.
        """

        # Stage 1: direct decoding establishes the initial bounded CFG.
        self._decode_all_blocks()
        # Stage 2: exact transfer proofs may add leaders and re-decode blocks.
        self._discover_static_jump_targets()
        self._recognize_saved_link_returns()
        # Stage 3: LSDA records establish known exceptional flow before any
        # unknown-entry discovery, including calls in other landing pads.
        self._discover_elf_exceptional_edges()
        # No proof query runs after this presentation-only discovery stage.
        if recover_disconnected:
            self._recover_disconnected_components()
            # Recovered calls can unwind too. LSDA may supply missing cleanup
            # leaders, but their code remains outside the original proof graph.
            self._discover_recovered_elf_exceptional_edges()
        # Stage 4: materialize the stabilized graph and its conservative edges.
        self._materialize_edges()
        self._attach_disconnected_components()
        function = self.kb.functions.function(self.func_addr, create=True)
        if function is not None:
            # angr treats names such as ``sub_119320`` as address selectors.
            # Create by the rebased address first, then set the display name.
            function.name = self.bounds.name
        self._summarize_output()

        anomalies = find_custom_cfg_anomalies(
            self.graph,
            self.bounds,
            self.func_addr,
            self._output_blocks(),
            project=self.project,
            recovered_roots=self.recovered_roots,
        )
        self.stats.output_anomaly_count = len(anomalies)
        self.stats.output_anomalies_by_kind = dict(
            sorted(Counter(anomaly.kind for anomaly in anomalies).items())
        )
        if anomalies:
            for anomaly in anomalies:
                logger.warning(anomaly.message)
        else:
            logger.info(
                f"Custom CFG for {self.func_addr:#x} passed structural validation"
            )
        return CustomCFG(
            graph=self.graph,
            model=self.model,
            functions=self.kb.functions,
            kb=self.kb,
            custom_stats=self.stats,
            custom_summary=self.summary,
        )


def build_custom_cfg(
    project: Project,
    kb: KnowledgeBase,
    func_addr: int,
) -> CustomCFG:
    """Build one bounded function CFG with recovery, without invoking CFGFast."""

    logger.info(f"Building custom CFG for function {func_addr:#x} without CFGFast")
    cfg = _BuildSession(project, kb, func_addr).build()
    logger.info(
        f"Custom CFG for {func_addr:#x}: "
        f"stats={cfg.custom_stats.as_dict()}, "
        f"summary={cfg.custom_summary.as_dict()}"
    )
    return cfg
