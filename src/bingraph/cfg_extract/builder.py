"""Bounded CFG extraction without CFGFast.

The extractor starts at one function symbol and grows only through addresses
proven by decoded direct transfers. A shared leader set keeps recovered blocks
non-overlapping: whenever a newly discovered target falls inside an existing
block, that block is re-decoded with the target as a stop address.

Extraction has four deliberate stages: decode direct flow, resolve exact
indirect transfers, conservatively reconnect only shape-free gaps, then
materialize the final graph. VEX-proven static jump tables contribute leaders
during the exact-resolution stage; remaining indirect transfers stay explicit
synthetic leaves. The extractor never reads CFGFast's discovered regions.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import replace
from types import SimpleNamespace
from typing import Mapping, cast

from angr import KnowledgeBase, Project
from angr.knowledge_plugins.cfg import CFGModel, CFGNode
from loguru import logger
import networkx as nx

from bingraph.cfg.anomalies import _lookup_function_bounds
from bingraph.cfg.graph import CFGGraph, add_successor_edge
from bingraph.cfg.jumps import (
    _read_static_jump_table_targets,
    abi_static_register_transfer_targets,
    conditional_pc_dispatch_targets,
    is_direct_memory_indirect_jump,
    plan_dynamic_selector_table_candidates,
    plan_mips_pic_relative_jump_table,
    plan_static_jump_table,
    static_jump_target_rejection_reason,
    unconditional_arithmetic_pc_dispatch_targets,
)
from bingraph.cfg.models import BlockSpec, EdgeJumpKind, FunctionBounds
from bingraph.cfg.decode import (
    alternate_block_entry_rejoin_addr,
    decode_bounded_block,
    decode_raw_capstone_insns,
    is_valid_block_entry,
)
from bingraph.helpers.capstone import InsnSemantics
from bingraph.helpers.symbols import plt_symbol_name

from .anomalies import find_extracted_cfg_anomalies
from .data import StaticDataRegions
from .exceptions import ExceptionalCallSite, exceptional_call_sites_for_function
from .models import (
    ExtractedCFG,
    ExtractedCFGNode,
    ExtractedCFGStats,
    ExtractedCFGSummary,
)
from .sweep import recover_executable_components, select_reconnecting_components
from .syscalls import ResolvedSyscall, resolve_static_syscall, unknown_syscall_target


_UNRESOLVABLE_CALL_ADDR = 0xFFFFFFFFFFFFFFD0
_RECONNECTING_SWEEP_REASONS = frozenset({"no_vex", "no_table_shape"})
# Candidate rows are decoded transitively, so keep their unproven frontier
# small. Exact static table plans are not subject to this limit.
_MAX_DYNAMIC_SELECTOR_CANDIDATE_TARGETS = 10


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


def _make_block_node(
    model: CFGModel,
    project: Project,
    func_addr: int,
    bounds: FunctionBounds,
    block: BlockSpec,
) -> CFGNode:
    """Materialize one recovered normal CFG node."""

    return ExtractedCFGNode(
        block.addr,
        block.size,
        cfg=model,
        function_address=func_addr,
        block_id=block.addr,
        instruction_addrs=block.instruction_addrs,
        thumb=_thumb_mode(project, block.addr),
        name=_node_name(bounds, block.addr),
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


class _ExtractionSession:
    """Own the leader worklist and graph materialization for one function."""

    def __init__(self, project: Project, kb: KnowledgeBase, func_addr: int) -> None:
        self.project = project
        self.kb = kb
        self.func_addr = func_addr
        self.bounds = _lookup_function_bounds(project, func_addr)
        manager = SimpleNamespace(_kb=kb)
        self.model = CFGModel("CFGExtract", cfg_manager=manager)
        self.graph = cast(CFGGraph, self.model.graph)
        self.stats = ExtractedCFGStats()
        self.summary = ExtractedCFGSummary()
        self.leaders = {func_addr}
        self.rejected_leaders: set[int] = set()
        self.pending = deque([func_addr])
        self.pending_addrs = {func_addr}
        self.blocks: dict[int, BlockSpec] = {}
        self.data_regions = StaticDataRegions()
        self.data_regions.claim_code(project, func_addr)
        self.leaf_nodes: dict[tuple[int, str], CFGNode] = {}
        self.static_targets: dict[int, tuple[int, ...]] = {}
        self.static_target_candidates: dict[int, tuple[int, ...]] = {}
        self.exceptional_targets: dict[int, tuple[int, ...]] = {}
        self.unresolved_dispatcher_reasons: dict[int, str | None] = {}
        self.sweep_dispatcher_addr: int | None = None
        self.sweep_component_roots: frozenset[int] = frozenset()
        self.continued_linear_direct_transfers: set[int] = set()
        self.resolved_syscalls: dict[int, ResolvedSyscall] = {}

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
            self.stats.call_fallthroughs_suppressed += 1

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
                    f"Extract CFG rejected literal-pool leader {addr:#x} for "
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
                    f"Extract CFG could not decode block at {addr:#x} for "
                    f"function {self.func_addr:#x}"
                )
                continue
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

        The snapshot includes decoded direct/fallthrough edges and static-table
        edges proven by an earlier discovery round. It intentionally excludes
        unresolved and candidate edges: they cannot establish a must-reaching
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
            # Previously proven table rows become normal flow for later
            # proofs. This permits a guarded selector to survive one exact
            # dispatcher before it indexes a second table.
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
        return graph, nodes

    def _call_block_matches_lsda_site(
        self, block: BlockSpec, site: ExceptionalCallSite
    ) -> bool:
        """Return whether this recovered call terminator lies in one LSDA range."""

        if block.jumpkind != "Ijk_Call":
            return False
        insns = decode_raw_capstone_insns(self.project, block.addr, block.size)
        call = next(
            (insn for insn in reversed(insns) if InsnSemantics(insn).is_call()),
            None,
        )
        return call is not None and (
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

    def _resolve_abi_static_register_transfers(self) -> None:
        """Materialize ABI-preserved register transfers proven by VEX dataflow."""

        while True:
            targets, exhausted, ran = abi_static_register_transfer_targets(
                self.project, self.bounds, self.blocks
            )
            self.stats.abi_static_target_analysis_runs += ran
            self.stats.abi_static_target_analysis_budget_exhausted += exhausted
            if exhausted or not targets:
                return

            discovered = False
            for addr, target in targets.items():
                block = self.blocks[addr]
                self.blocks[addr] = replace(block, direct_targets=(target,))
                if block.jumpkind == "Ijk_Call":
                    self.stats.abi_static_call_targets_resolved += 1
                else:
                    self.stats.abi_static_jump_targets_resolved += 1
                if self.bounds.addr <= target < self.bounds.end_addr:
                    self._claim_code_target(target)
                    before = target in self.blocks or target in self.pending_addrs
                    if self._add_leader(target):
                        discovered |= not before
            if not discovered:
                return
            self._decode_all_blocks()

    def _discover_static_jump_targets(self) -> None:
        """Iteratively discover exact and candidate indirect jump targets.

        Exact resolvers run in priority order: conditional-PC forms,
        arithmetic-PC forms, generic VEX tables, then the MIPS PIC fallback.
        Only a fully bounded table becomes ``static_targets``. A weaker,
        unbounded memory-selector table may contribute dashed candidates, but
        never replaces the unresolved target.

        Each exact target is added as a leader. Because a new leader can split
        a block, plans are re-evaluated until the decode stabilizes. Plans from
        an earlier round survive only when their source ``BlockSpec`` is
        unchanged, preserving their proof while avoiding stale source edges.
        """

        retained_plans: dict[int, tuple[BlockSpec, tuple[int, ...]]] = {}
        while True:
            retained_targets = {
                addr: targets
                for addr, (source, targets) in retained_plans.items()
                if self.blocks.get(addr) == source
            }
            graph, nodes = self._analysis_graph(retained_targets)
            discovered = False
            plans: dict[int, tuple[int, ...]] = {}
            candidate_plans: dict[int, tuple[int, ...]] = {}
            candidate_entry_counts: dict[int, int] = {}
            unresolved_reasons: dict[int, str | None] = {}
            conditional_sources: set[int] = set()
            # Candidate plans are collected first so accepting them is safe
            # only when exactly one unresolved table has this weak shape.
            candidate_table_plans = {
                addr: plan
                for addr, node in nodes.items()
                if (block := self.blocks.get(addr)) is not None
                and block.jumpkind == "Ijk_Boring"
                and not block.direct_targets
                and (
                    plan := plan_dynamic_selector_table_candidates(
                        self.project, graph, self.bounds, node
                    )
                )
                is not None
            }
            candidate_recovery_allowed = len(candidate_table_plans) == 1
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
                    self.project, self.bounds, node, graph
                )
                if targets is not None:
                    conditional_sources.add(addr)
                if targets is None and reason == "not_conditional_pc":
                    targets = unconditional_arithmetic_pc_dispatch_targets(
                        self.project, graph, self.bounds, node
                    )
                if targets is None and reason in {"not_conditional_pc", "no_vex"}:
                    plan, reason = plan_static_jump_table(
                        self.project,
                        graph,
                        self.bounds,
                        node,
                        allow_inline_index_values=True,
                        allow_masked_index_values=True,
                        allow_guarded_loads=True,
                        allow_static_bases=True,
                        allow_guarded_expression_indices=True,
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
                if targets is None and reason == "no_table_shape":
                    candidate_plan = candidate_table_plans.get(addr)
                    if candidate_plan is not None:
                        candidate_targets = _read_static_jump_table_targets(
                            self.project,
                            candidate_plan.table,
                            candidate_plan.base_addr,
                            candidate_plan.entry_indices,
                        )
                        if (
                            candidate_recovery_allowed
                            and candidate_targets is not None
                            and len(candidate_targets)
                            <= _MAX_DYNAMIC_SELECTOR_CANDIDATE_TARGETS
                            and all(
                                self.bounds.addr <= target < self.bounds.end_addr
                                and static_jump_target_rejection_reason(
                                    self.project, target
                                )
                                is None
                                for target in candidate_targets
                            )
                        ):
                            accepted_candidates: list[int] = []
                            for target in dict.fromkeys(candidate_targets):
                                self._claim_code_target(target)
                                before = (
                                    target in self.blocks
                                    or target in self.pending_addrs
                                )
                                added = self._add_leader(target)
                                if (
                                    added
                                    or target in self.blocks
                                    or target in self.pending_addrs
                                ):
                                    accepted_candidates.append(target)
                                    discovered |= added and not before
                            if accepted_candidates:
                                candidate_plans[addr] = tuple(accepted_candidates)
                                candidate_entry_counts[addr] = len(candidate_targets)
                if targets is None:
                    if reason == "no_table_shape" and is_direct_memory_indirect_jump(
                        self.project, node
                    ):
                        reason = "dynamic_memory_target"
                    unresolved_reasons[addr] = reason
                    self.stats.static_jump_unresolved_dispatcher_attempts += 1
                    if reason is not None:
                        field = f"static_jump_{reason}"
                        if hasattr(self.stats, field):
                            setattr(self.stats, field, getattr(self.stats, field) + 1)
                    continue
                self.stats.static_jump_table_entries_read += len(targets)
                rejected = [
                    static_jump_target_rejection_reason(self.project, target)
                    for target in targets
                    if not self.bounds.addr <= target < self.bounds.end_addr
                ]
                if any(rejected):
                    self.stats.static_jump_targets_rejected += sum(
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
                self.stats.static_jump_targets_accepted += len(accepted_targets)

            if discovered:
                for addr, targets in plans.items():
                    source = self.blocks.get(addr)
                    if source is not None and targets:
                        retained_plans[addr] = (source, targets)
                # Block splits invalidate plans built from this graph snapshot.
                # Retain a proof only if its source block is unchanged after
                # rebuilding. A split can otherwise orphan its target leaders.
                self.stats.static_jump_plans_invalidated += len(plans)
                self._decode_all_blocks()
                continue
            retained_targets.update(plans)
            self.static_targets.update(retained_targets)
            self.static_target_candidates.update(candidate_plans)
            self.unresolved_dispatcher_reasons = {
                addr: reason
                for addr, reason in unresolved_reasons.items()
                if addr not in retained_targets
            }
            self.stats.conditional_pc_dispatches_resolved += len(conditional_sources)
            self.stats.conditional_pc_targets_recovered += sum(
                len(retained_targets[addr])
                for addr in conditional_sources
                if addr in retained_targets
            )
            self.stats.static_jump_plans_resolved += len(retained_targets)
            self.stats.static_jump_candidate_plans += len(candidate_plans)
            self.stats.static_jump_candidate_entries_read += sum(
                candidate_entry_counts.values()
            )
            self.stats.static_jump_candidate_targets_accepted += sum(
                len(targets) for targets in candidate_plans.values()
            )
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
        candidate and sweep phases may later add dashed edges, but do not
        remove that explicit unknown-target fallback.
        """

        self.nodes = {
            addr: _make_block_node(
                self.model, self.project, self.func_addr, self.bounds, block
            )
            for addr, block in sorted(self.blocks.items())
        }
        for node in self.nodes.values():
            self.graph.add_node(node)

        for addr, block in sorted(self.blocks.items()):
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

            for target in self.exceptional_targets.get(addr, ()):
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
                and block.fallthrough_addr is None
                and not self.static_targets.get(addr)
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

    def _recover_reconnecting_components(self) -> None:
        """Attach leader-closed components behind one shape-free dispatcher.

        A failed static-table plan can still prove that an indirect branch has
        a table-like shape, for example an address with an unbounded index.
        Reconnecting arbitrary executable components behind that source would
        turn an incomplete proof into speculative targets. Sweep recovery is
        therefore reserved for dispatchers where VEX exposed no table shape at
        all; a recognized but unresolved table keeps only its explicit leaf.
        """

        dispatchers = [
            addr
            for addr, block in self.blocks.items()
            if (
                block.jumpkind == "Ijk_Boring"
                and not block.direct_targets
                and block.fallthrough_addr is None
                and not self.static_targets.get(addr)
            )
        ]
        if len(dispatchers) != 1:
            return

        dispatcher_addr = dispatchers[0]
        if (
            self.unresolved_dispatcher_reasons.get(dispatcher_addr)
            not in _RECONNECTING_SWEEP_REASONS
        ):
            self.stats.sweep_dispatchers_ineligible += 1
            return

        recovered_blocks = dict(self.blocks)
        sweep = recover_executable_components(
            self.project,
            self.bounds,
            recovered_blocks,
            stop_at_data=lambda target: self.data_regions.contains(
                self.project, target
            ),
        )
        audit = sweep.audit
        self.stats.sweep_runs += 1
        self.stats.sweep_candidate_blocks += audit.candidate_blocks
        self.stats.sweep_candidate_instructions += audit.candidate_instructions
        self.stats.sweep_candidate_components += audit.candidate_components
        self.stats.sweep_decode_failures += audit.decode_failures
        self.stats.sweep_non_executable_bytes += audit.non_executable_bytes
        selected = select_reconnecting_components(
            self.project,
            sweep,
            recovered_blocks,
            static_targets={
                addr: tuple(
                    dict.fromkeys((*self.static_targets.get(addr, ()), *candidates))
                )
                for addr in self.static_targets.keys()
                | self.static_target_candidates.keys()
                for candidates in (self.static_target_candidates.get(addr, ()),)
            },
        )
        if dispatcher_addr not in selected.blocks:
            # The sweep changed the source whose unknown targets would be
            # attached. Keep the original graph rather than mix a stale
            # dispatcher with speculative sweep components.
            return
        if not selected.blocks:
            return

        self.blocks = dict(selected.blocks)
        self.leaders = set(self.blocks)
        for block in tuple(self.blocks.values()):
            for target in block.direct_targets:
                self._claim_code_target(target)
                self._add_leader(target)
        self._decode_all_blocks()
        self.sweep_dispatcher_addr = dispatcher_addr
        self.sweep_component_roots = selected.roots
        self.stats.sweep_reconnecting_components += selected.component_count
        self.stats.sweep_reconnecting_blocks += selected.reconnecting_block_count
        logger.info(
            f"Extract CFG recovery for {self.func_addr:#x}: selected "
            f"{selected.reconnecting_block_count} block(s) from "
            f"{selected.component_count} "
            f"reconnecting component(s) behind {dispatchers[0]:#x}"
        )

    def _attach_reconnecting_components(self) -> None:
        """Connect selected roots directly while retaining the unknown leaf.

        One dispatcher makes the source of every selected component
        unambiguous, so draw the candidate edges directly from it. The
        ``UnresolvableJumpTarget`` edge remains as an explicit catch-all:
        executable-range recovery cannot prove that these are every possible
        target, including targets outside the bounded function region.
        """

        if self.sweep_dispatcher_addr is None:
            return
        dispatcher = self.nodes[self.sweep_dispatcher_addr]
        for addr in self.sweep_component_roots:
            if add_successor_edge(
                self.graph,
                dispatcher,
                self.nodes[addr],
                "Ijk_Boring",
                unresolved_indirect=True,
            ):
                self.stats.sweep_component_roots_attached += 1

    def _attach_static_jump_table_candidates(self) -> None:
        """Attach heuristic table rows while preserving their unknown target."""

        for dispatcher_addr, targets in self.static_target_candidates.items():
            dispatcher = self.nodes[dispatcher_addr]
            for target in targets:
                if add_successor_edge(
                    self.graph,
                    dispatcher,
                    self.nodes[target],
                    "Ijk_Boring",
                    unresolved_indirect=True,
                ):
                    self.stats.static_jump_candidate_edges_added += 1

    def _summarize_output(self) -> None:
        """Record the final graph shape separately from extraction decisions."""

        self.summary.normal_blocks = len(self.blocks)
        self.summary.synthetic_leaves = len(self.leaf_nodes)
        self.summary.nodes = len(tuple(self.graph.nodes()))
        self.summary.edges = len(tuple(self.graph.edges()))
        for block in self.blocks.values():
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

    def build(self) -> ExtractedCFG:
        """Run bounded extraction in decode, proof, recovery, render order.

        Exact target discovery precedes any sweep so that a static table never
        depends on speculative recovered code. Rendering is deliberately last:
        it consumes the stabilized block set and records unresolved targets
        that the proof stages intentionally declined to resolve.
        """

        # Stage 1: direct decoding establishes the initial bounded CFG.
        self._decode_all_blocks()
        # Stage 2: exact transfer proofs may add leaders and re-decode blocks.
        self._resolve_abi_static_register_transfers()
        self._discover_static_jump_targets()
        # Stage 3: only shape-free unresolved flow may gain dashed recovery.
        # Run this before LSDA discovery: cleanup pads can contain additional
        # unresolved jumps, but must not hide a pre-existing sweep dispatcher.
        self._recover_reconnecting_components()
        # LSDA records attach to the stabilized call blocks, including those
        # reached through a recovered component or another landing pad.
        self._discover_elf_exceptional_edges()
        # Stage 4: materialize the stabilized graph and its conservative edges.
        self._materialize_edges()
        self._attach_static_jump_table_candidates()
        self._attach_reconnecting_components()
        function = self.kb.functions.function(self.func_addr, create=True)
        if function is not None:
            # angr treats names such as ``sub_119320`` as address selectors.
            # Create by the rebased address first, then set the display name.
            function.name = self.bounds.name
        self._summarize_output()

        anomalies = find_extracted_cfg_anomalies(
            self.graph,
            self.bounds,
            self.func_addr,
            self.blocks,
            project=self.project,
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
                f"Extracted CFG for {self.func_addr:#x} passed structural validation"
            )
        return ExtractedCFG(
            graph=self.graph,
            model=self.model,
            functions=self.kb.functions,
            kb=self.kb,
            extract_stats=self.stats,
            extract_summary=self.summary,
        )


def build_extracted_cfg(
    project: Project, kb: KnowledgeBase, func_addr: int
) -> ExtractedCFG:
    """Build one experimental function CFG without invoking CFGFast."""

    logger.info(f"Extracting CFG for function {func_addr:#x} without CFGFast")
    cfg = _ExtractionSession(project, kb, func_addr).build()
    logger.info(
        f"Extracted CFG for {func_addr:#x}: "
        f"stats={cfg.extract_stats.as_dict()}, "
        f"summary={cfg.extract_summary.as_dict()}"
    )
    return cfg
