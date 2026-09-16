"""Read-only executable-range discovery used to audit unresolved dispatches."""

from __future__ import annotations

from collections.abc import Callable
from collections import deque
from dataclasses import dataclass
from typing import Mapping

from angr import Project
import networkx as nx

from bingraph.cfg.graph import vex_is_transparent_fallthrough_padding
from bingraph.cfg.jumps import static_jump_target_rejection_reason
from bingraph.cfg.models import BlockSpec, FunctionBounds
from bingraph.cfg.decode import (
    alternate_block_entry_rejoin_addr,
    decode_bounded_block,
    is_valid_block_entry,
)


@dataclass(frozen=True)
class ExecutableSweepAudit:
    """Candidate blocks found without changing the extracted CFG."""

    candidate_blocks: int
    candidate_instructions: int
    candidate_components: int
    decode_failures: int
    non_executable_bytes: int


@dataclass(frozen=True)
class ExecutableSweep:
    """Closed direct-flow components discovered outside extracted reachability."""

    blocks: Mapping[int, BlockSpec]
    reachable_addrs: frozenset[int]
    disconnected_addrs: frozenset[int]
    audit: ExecutableSweepAudit


@dataclass(frozen=True)
class ReconnectingComponents:
    """Disconnected components safe to expose behind one unknown dispatcher."""

    blocks: Mapping[int, BlockSpec]
    roots: frozenset[int]
    component_count: int


def _covering_end(blocks: Mapping[int, BlockSpec], addr: int) -> int | None:
    """Return the end of a recovered block covering ``addr``, if any."""

    for start, block in blocks.items():
        end = start + block.size
        if start <= addr < end:
            return end
    return None


def _direct_flow_graph(blocks: Mapping[int, BlockSpec]) -> nx.DiGraph:
    """Build the decoded direct-flow graph for a recovered block mapping."""

    graph = nx.DiGraph()
    graph.add_nodes_from(blocks)
    for addr, block in blocks.items():
        for target in (*block.direct_targets, block.fallthrough_addr):
            if target in blocks:
                graph.add_edge(addr, target)
    return graph


def _reachable_addrs(
    blocks: Mapping[int, BlockSpec], bounds: FunctionBounds
) -> set[int]:
    """Return block starts reachable from the function entry by direct flow."""

    graph = _direct_flow_graph(blocks)
    if bounds.addr not in graph:
        return set()
    return nx.descendants(graph, bounds.addr) | {bounds.addr}


def _disconnected_component_count(
    blocks: Mapping[int, BlockSpec], disconnected_addrs: set[int]
) -> int:
    """Count direct-flow components outside the function entry's reachability."""

    if not disconnected_addrs:
        return 0
    return nx.number_weakly_connected_components(
        _direct_flow_graph(blocks).subgraph(disconnected_addrs)
    )


def _is_transparent_fallthrough_padding(
    project: Project | None, block: BlockSpec
) -> bool:
    """Return whether a swept block is non-informative alignment padding."""

    if project is None:
        return False
    try:
        vex = project.factory.block(
            block.addr,
            size=block.size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception:
        return False
    return vex_is_transparent_fallthrough_padding(vex, block.addr, block.size)


def _recover_direct_closure(
    project: Project,
    bounds: FunctionBounds,
    blocks: dict[int, BlockSpec],
    leaders: set[int],
    stop_at_data: Callable[[int], bool] | None,
) -> int:
    """Close sweep targets using the extractor's safe leader invariant."""

    pending: deque[int] = deque()
    pending_addrs: set[int] = set()
    rejected_leaders: set[int] = set()
    decode_failures = 0

    def queue(addr: int) -> None:
        if addr not in pending_addrs:
            pending.append(addr)
            pending_addrs.add(addr)

    def reject(addr: int) -> None:
        rejected_leaders.add(addr)
        leaders.discard(addr)
        blocks.pop(addr, None)

    def add_leader(addr: int) -> None:
        if not bounds.addr <= addr < bounds.end_addr:
            return
        if addr in rejected_leaders:
            return
        if stop_at_data is not None and stop_at_data(addr):
            reject(addr)
            return

        covering_blocks = [
            block
            for start, block in blocks.items()
            if start < addr < start + block.size
        ]
        if covering_blocks and not all(
            is_valid_block_entry(project, block, addr) for block in covering_blocks
        ):
            reject(addr)
            return

        if addr not in leaders:
            leaders.add(addr)
            for start, block in tuple(blocks.items()):
                if start < addr < start + block.size:
                    rejoin_addr = alternate_block_entry_rejoin_addr(
                        project, block, addr
                    )
                    if rejoin_addr is not None:
                        add_leader(rejoin_addr)
                        continue
                    del blocks[start]
                    queue(start)
        if addr not in blocks:
            queue(addr)

    def normalize_inner_leaders(block: BlockSpec) -> None:
        for leader in tuple(leaders):
            if not block.addr < leader < block.addr + block.size:
                continue
            if leader in block.instruction_addrs:
                continue
            rejoin_addr = alternate_block_entry_rejoin_addr(project, block, leader)
            if rejoin_addr is None:
                reject(leader)
            else:
                add_leader(rejoin_addr)

    for block in tuple(blocks.values()):
        for target in (*block.direct_targets, block.fallthrough_addr):
            if target is not None:
                add_leader(target)

    while pending:
        addr = pending.popleft()
        pending_addrs.remove(addr)
        if addr in rejected_leaders:
            continue
        block = decode_bounded_block(
            project,
            bounds,
            addr,
            leaders - {addr},
            preserve_conditional_return_fallthrough=True,
            split_syscall_blocks=True,
            resolve_declared_nonreturning=True,
            allow_vex_linear_fallback=True,
            stop_at_data=stop_at_data,
        )
        if block is None or block.size <= 0:
            decode_failures += 1
            continue

        normalize_inner_leaders(block)
        covering_blocks = [
            (start, known_block)
            for start, known_block in blocks.items()
            if start < addr < start + known_block.size
        ]
        if any(
            not is_valid_block_entry(project, known_block, addr)
            for _, known_block in covering_blocks
        ):
            reject(addr)
            continue
        for start, known_block in covering_blocks:
            if (
                alternate_block_entry_rejoin_addr(project, known_block, addr)
                is not None
            ):
                continue
            del blocks[start]
            queue(start)

        blocks[addr] = block
        for target in (*block.direct_targets, block.fallthrough_addr):
            if target is not None:
                add_leader(target)

    return decode_failures


def recover_executable_components(
    project: Project,
    bounds: FunctionBounds,
    recovered_blocks: Mapping[int, BlockSpec],
    *,
    stop_at_data: Callable[[int], bool] | None = None,
) -> ExecutableSweep:
    """Recover and validate disconnected executable components without a CFG.

    The sweep starts from unclaimed executable bytes, then closes the decoded
    direct flow around each recovered target. It deliberately returns data only:
    callers decide whether unresolved indirect dispatches justify materializing
    these speculative components.
    """

    blocks = dict(recovered_blocks)
    leaders = set(blocks)
    cursor = bounds.addr
    decode_failures = 0
    non_executable_bytes = 0

    while cursor < bounds.end_addr:
        if stop_at_data is not None and stop_at_data(cursor):
            cursor += 1
            continue
        covered_end = _covering_end(blocks, cursor)
        if covered_end is not None:
            cursor = covered_end
            continue

        if static_jump_target_rejection_reason(project, cursor) is not None:
            non_executable_bytes += 1
            cursor += 1
            continue

        block = decode_bounded_block(
            project,
            bounds,
            cursor,
            leaders,
            preserve_conditional_return_fallthrough=True,
            split_syscall_blocks=True,
            resolve_declared_nonreturning=True,
            allow_vex_linear_fallback=True,
            stop_at_data=stop_at_data,
        )
        if block is None or block.size <= 0:
            decode_failures += 1
            cursor += 1
            continue

        blocks[block.addr] = block
        leaders.add(block.addr)
        cursor = block.addr + block.size

    decode_failures += _recover_direct_closure(
        project, bounds, blocks, leaders, stop_at_data
    )
    reachable_addrs = _reachable_addrs(blocks, bounds)
    disconnected_addrs = set(blocks) - reachable_addrs

    audit = ExecutableSweepAudit(
        candidate_blocks=len(disconnected_addrs),
        candidate_instructions=sum(
            len(blocks[addr].instruction_addrs) for addr in disconnected_addrs
        ),
        candidate_components=_disconnected_component_count(blocks, disconnected_addrs),
        decode_failures=decode_failures,
        non_executable_bytes=non_executable_bytes,
    )
    return ExecutableSweep(
        blocks,
        frozenset(reachable_addrs),
        frozenset(disconnected_addrs),
        audit,
    )


def select_reconnecting_components(
    project: Project | None,
    sweep: ExecutableSweep,
    recovered_blocks: Mapping[int, BlockSpec],
) -> ReconnectingComponents:
    """Select direct-flow components that rejoin known function code.

    Executable bytes alone do not prove an indirect-jump target. A component
    becomes a candidate only when its decoded direct flow reaches a block from
    the original extraction, it contains no additional unresolved indirect
    branch, and it does not target the middle of original code. The latter two
    cases need their own target evidence, not inherited trust from the outer
    dispatcher.
    """

    graph = _direct_flow_graph(sweep.blocks)
    disconnected_graph = graph.subgraph(sweep.disconnected_addrs)
    recovered_addrs = set(recovered_blocks)
    # Preserve leader-closed revisions of entry-reachable blocks. A selected
    # component can branch into the middle of an original block, so retaining
    # the pre-sweep block would undo the exact-target split.
    selected_blocks = {addr: sweep.blocks[addr] for addr in sweep.reachable_addrs}
    roots: set[int] = set()
    component_count = 0

    for component in nx.weakly_connected_components(disconnected_graph):
        has_rejoin = any(
            target in recovered_addrs
            for addr in component
            for target in graph.successors(addr)
        )
        has_nested_unresolved = any(
            sweep.blocks[addr].jumpkind == "Ijk_Boring"
            and not sweep.blocks[addr].direct_targets
            and sweep.blocks[addr].fallthrough_addr is None
            for addr in component
        )
        has_mid_block_target = any(
            block.jumpkind == "Ijk_Boring"
            and any(
                start < target < start + recovered_block.size
                for target in (*block.direct_targets, block.fallthrough_addr)
                if target is not None
                for start, recovered_block in recovered_blocks.items()
            )
            for block in (sweep.blocks[addr] for addr in component)
        )
        if not has_rejoin or has_nested_unresolved or has_mid_block_target:
            continue

        component_graph = disconnected_graph.subgraph(component)
        condensation = nx.condensation(component_graph)
        component_roots: set[int] = set()
        for source in condensation.nodes:
            if condensation.in_degree(source) != 0:
                continue
            members = condensation.nodes[source]["members"]
            component_roots.add(min(members))
        # A component whose only roots are swept alignment NOPs merely falls
        # through into recovered code; it supplies no indirect-target evidence.
        # Keep mixed-root components intact so no retained block becomes
        # unreachable through a suppressed entry root.
        if component_roots and all(
            _is_transparent_fallthrough_padding(project, sweep.blocks[root])
            for root in component_roots
        ):
            continue

        component_count += 1
        selected_blocks.update((addr, sweep.blocks[addr]) for addr in component)
        roots.update(component_roots)

    return ReconnectingComponents(selected_blocks, frozenset(roots), component_count)


def audit_executable_range(
    project: Project,
    bounds: FunctionBounds,
    recovered_blocks: Mapping[int, BlockSpec],
    *,
    stop_at_data: Callable[[int], bool] | None = None,
) -> ExecutableSweepAudit:
    """Return read-only statistics for disconnected executable components."""

    return recover_executable_components(
        project, bounds, recovered_blocks, stop_at_data=stop_at_data
    ).audit
