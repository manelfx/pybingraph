"""Read-only executable-range discovery used to audit unresolved dispatches."""

from __future__ import annotations

from bisect import bisect_left
from collections.abc import Callable, Collection
from collections import deque
from dataclasses import dataclass, replace
from typing import Mapping

from angr import Project
import networkx as nx

from bingraph.cfg.graph import vex_is_transparent_fallthrough_padding
from bingraph.cfg.jumps import static_jump_target_rejection_reason
from bingraph.cfg.models import BlockSpec, FunctionBounds
from bingraph.cfg.decode import (
    alternate_block_entry_rejoin_addr,
    decode_bounded_block,
    decode_raw_capstone_insns,
    is_valid_block_entry,
    target_is_known_nonreturning,
)
from bingraph.helpers.capstone import InsnSemantics


class SweepBudgetExceeded(Exception):
    """A speculative scan exceeded its deterministic decoding-work bound."""


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
    reconnecting_block_count: int = 0


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


def _reachable_with_static_targets(
    blocks: Mapping[int, BlockSpec],
    directly_reachable: frozenset[int],
    static_targets: Mapping[int, tuple[int, ...]],
) -> set[int]:
    """Return blocks reached by direct flow or already-proven table edges."""

    graph = _direct_flow_graph(blocks)
    reachable = set(directly_reachable)
    pending = deque(reachable)
    while pending:
        source = pending.popleft()
        targets = (*graph.successors(source), *static_targets.get(source, ()))
        for target in targets:
            if target not in graph or target in reachable:
                continue
            reachable.add(target)
            pending.append(target)
    return reachable


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
    *,
    resolve_static_memory_calls: bool = False,
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
            resolve_static_memory_calls=resolve_static_memory_calls,
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
    max_steps: int | None = None,
    resolve_static_memory_calls: bool = False,
) -> ExecutableSweep:
    """Recover and validate disconnected executable components without a CFG.

    The sweep starts from unclaimed executable bytes, then closes the decoded
    direct flow around each recovered target. It deliberately returns data only:
    callers decide whether unresolved indirect dispatches justify materializing
    these speculative components.
    Static-memory call decoding is opt-in so presentation-only recovery can
    match normal extraction without changing legacy sweep consumers.
    """

    if max_steps is not None:
        original_stop = stop_at_data
        remaining = max_steps

        def bounded_stop(addr: int) -> bool:
            nonlocal remaining
            remaining -= 1
            if remaining < 0:
                raise SweepBudgetExceeded
            return original_stop(addr) if original_stop is not None else False

        stop_at_data = bounded_stop

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
            resolve_static_memory_calls=resolve_static_memory_calls,
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
        project,
        bounds,
        blocks,
        leaders,
        stop_at_data,
        resolve_static_memory_calls=resolve_static_memory_calls,
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
    *,
    static_targets: Mapping[int, tuple[int, ...]] | None = None,
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
    known_reachable = _reachable_with_static_targets(
        sweep.blocks, sweep.reachable_addrs, static_targets or {}
    )
    disconnected_graph = graph.subgraph(set(sweep.blocks) - known_reachable)
    recovered_addrs = set(recovered_blocks)
    # Preserve leader-closed revisions of known blocks, including static-table
    # closure. A selected component can branch into the middle of an original
    # block, so retaining the pre-sweep block would undo an exact-target split.
    selected_blocks = {addr: sweep.blocks[addr] for addr in known_reachable}
    roots: set[int] = set()
    component_count = 0
    reconnecting_block_count = 0

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
        reconnecting_block_count += len(component)
        selected_blocks.update((addr, sweep.blocks[addr]) for addr in component)
        roots.update(component_roots)

    return ReconnectingComponents(
        selected_blocks,
        frozenset(roots),
        component_count,
        reconnecting_block_count,
    )


def validate_disconnected_baseline(
    sweep: ExecutableSweep | Mapping[int, BlockSpec],
    baseline: Mapping[int, BlockSpec],
    *,
    protected_sources: Collection[int] = (),
) -> dict[int, BlockSpec] | None:
    """Accept only lossless instruction-boundary partitions of known blocks.

    Prefix fragments must flow sequentially into the next fragment; the final
    fragment must retain the original terminal semantics and metadata. Exact
    proof sources stay pinned: splitting them would require independently
    revalidating their input facts, not trusting speculative predecessors.
    The returned mapping is for presentation, never resolver analysis.
    """

    # Both a sweep and the late LSDA decoder produce the same block facts.
    blocks = sweep.blocks if isinstance(sweep, ExecutableSweep) else sweep
    starts = sorted(blocks)
    validated: dict[int, BlockSpec] = {}
    for addr, original in baseline.items():
        if blocks.get(addr) == original:
            validated[addr] = original
            continue
        if addr in protected_sources:
            return None
        end = addr + original.size
        fragments = [
            blocks[start]
            for start in starts[bisect_left(starts, addr) : bisect_left(starts, end)]
        ]
        if (
            not fragments
            or fragments[0].addr != addr
            or fragments[-1].addr + fragments[-1].size != end
            or tuple(a for part in fragments for a in part.instruction_addrs)
            != original.instruction_addrs
        ):
            return None
        for prefix, following in zip(fragments, fragments[1:]):
            if (
                prefix.size <= 0
                or prefix.addr + prefix.size != following.addr
                or following.addr not in original.instruction_addrs
                or prefix.jumpkind != "Ijk_Fallthrough"
                or prefix.direct_targets
                or prefix.fallthrough_addr != following.addr
                or prefix.syscall_jumpkind is not None
                or prefix.vex_linear_instruction_sizes
            ):
                return None
        terminal = fragments[-1]
        if (
            terminal.size <= 0
            or replace(
                terminal,
                addr=addr,
                size=original.size,
                instruction_addrs=original.instruction_addrs,
            )
            != original
        ):
            return None
        validated.update((part.addr, part) for part in fragments)
    return validated


def select_disconnected_components(
    project: Project,
    sweep: ExecutableSweep,
    baseline: Mapping[int, BlockSpec],
    *,
    bounds: FunctionBounds | None = None,
) -> ReconnectingComponents:
    """Select meaningful code for unknown-entry presentation, never exact proofs.

    The baseline includes any instruction-boundary partitions accepted by
    ``validate_disconnected_baseline``. Every fragment is established code,
    not a newly recovered root, and must remain unchanged during selection.
    Regions may contain further unknown jumps, kept explicit in the output.
    Without a rejoin, require closed direct flow and evidence of a return,
    decoded trap, known non-returning call, or tail exit outside ``bounds``.
    An unknown exit is not such evidence.
    Neither case establishes reachability from an unresolved dispatcher, and
    alignment padding or isolated branch/return stubs are never payload.
    """

    if any(sweep.blocks.get(addr) != block for addr, block in baseline.items()):
        return ReconnectingComponents({}, frozenset(), 0)
    padding: dict[int, bool] = {}

    def is_padding(insn) -> bool:
        # Test single instructions: multi-instruction VEX lifts include
        # intermediate IP writes which obscure otherwise transparent padding.
        if insn.address not in padding:
            padding[insn.address] = insn.mnemonic in {"nop", "nop.w", "nop.n"} or (
                _is_transparent_fallthrough_padding(
                    project,
                    BlockSpec(
                        insn.address, insn.size, (insn.address,), "Ijk_Fallthrough"
                    ),
                )
            )
        return padding[insn.address]

    blocks = dict(sweep.blocks)
    graph = _direct_flow_graph(blocks)
    pending = deque(
        addr for addr in blocks if addr not in baseline and not graph.in_degree(addr)
    )
    while pending:
        addr = pending.popleft()
        block = blocks[addr]
        insns = decode_raw_capstone_insns(project, addr, block.size)
        prefix = 0
        for insn in insns:
            if not is_padding(insn):
                break
            prefix += insn.size
        if _is_transparent_fallthrough_padding(project, block) or prefix == block.size:
            successors = tuple(graph.successors(addr))
            graph.remove_node(addr)
            del blocks[addr]
            pending.extend(
                s for s in successors if s not in baseline and not graph.in_degree(s)
            )
        elif prefix:
            trimmed = replace(
                block,
                addr=addr + prefix,
                size=block.size - prefix,
                instruction_addrs=tuple(
                    a for a in block.instruction_addrs if a >= addr + prefix
                ),
            )
            del blocks[addr]
            blocks[trimmed.addr] = trimmed
    graph = _direct_flow_graph(blocks)
    disconnected = graph.subgraph(set(blocks) - baseline.keys())
    selected: dict[int, BlockSpec] = {}
    roots: set[int] = set()
    count = 0
    for component in nx.weakly_connected_components(disconnected):
        rejoins = any(
            target in baseline for a in component for target in graph.successors(a)
        )
        if not rejoins:
            # Return, trap and non-returning cases need not share an epilogue
            # with known code. A missing call fall-through alone is not proof:
            # the decoder may simply have reached the symbol's boundary.
            has_exit = False
            for addr in component:
                block = blocks[addr]
                external = tuple(
                    t
                    for t in block.direct_targets
                    if bounds is not None and not bounds.addr <= t < bounds.end_addr
                )
                nonreturning_call = (
                    block.jumpkind == "Ijk_Call"
                    and block.fallthrough_addr is None
                    and bool(block.direct_targets)
                    and all(
                        target_is_known_nonreturning(project, t)
                        for t in block.direct_targets
                    )
                )
                if (
                    block.jumpkind
                    not in {
                        "Ijk_Boring",
                        "Ijk_Call",
                        "Ijk_Fallthrough",
                        "Ijk_Ret",
                        "Ijk_Terminal",
                    }
                    or (
                        block.fallthrough_addr is not None
                        and block.fallthrough_addr not in blocks
                    )
                    or any(
                        t not in blocks and t not in external
                        for t in block.direct_targets
                    )
                    or (
                        block.jumpkind in {"Ijk_Call", "Ijk_Fallthrough"}
                        and block.fallthrough_addr is None
                        and not nonreturning_call
                    )
                ):
                    break
                has_exit |= (
                    nonreturning_call
                    or block.jumpkind in {"Ijk_Ret", "Ijk_Terminal"}
                    or (block.jumpkind == "Ijk_Boring" and bool(external))
                )
            else:
                if has_exit:
                    rejoins = True
            if not rejoins:
                continue
        payload = False
        for addr in component:
            block = blocks[addr]
            insns = decode_raw_capstone_insns(project, addr, block.size)
            if (
                tuple(i.address for i in insns) != block.instruction_addrs
                or sum(i.size for i in insns) != block.size
            ):
                break
            for insn in insns:
                semantic = InsnSemantics(insn)
                if (
                    not is_padding(insn)
                    and not (semantic.is_jump() and not semantic.is_conditional_jump())
                    and not semantic.is_ret()
                    and not (
                        block.jumpkind in {"Ijk_Ret", "Ijk_Terminal"}
                        and insn.address == block.instruction_addrs[-1]
                    )
                ):
                    payload = True
        else:
            if not payload:
                continue
            condensation = nx.condensation(disconnected.subgraph(component))
            # All members of a source SCC are possible presentation entries;
            # do not assert its lowest address is the actual runtime entry.
            for source in condensation:
                if not condensation.in_degree(source):
                    roots.update(condensation.nodes[source]["members"])
            selected.update((addr, blocks[addr]) for addr in component)
            count += 1
    return ReconnectingComponents(selected, frozenset(roots), count, len(selected))


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
