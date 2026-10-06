"""Structural validation for CFGs built by the independent extractor."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Iterable, Mapping

from angr import Project
from angr.knowledge_plugins.cfg import CFGNode

from bingraph.cfg.decode import alternate_block_entry_rejoin_addr
from bingraph.cfg.graph import CFGGraph, node_range_end, ranges_overlap
from bingraph.cfg.models import BlockSpec, FunctionBounds


@dataclass(frozen=True)
class ExtractedCFGAnomaly:
    """One invariant violation in a graph wholly constructed by extraction."""

    kind: str
    addr: int
    message: str


def _normal_nodes(graph: CFGGraph, func_addr: int) -> tuple[CFGNode, ...]:
    """Return this function's materialized nodes in stable address order."""

    return tuple(
        sorted(
            (
                node
                for node in graph.nodes()
                if not node.is_simprocedure and node.function_address == func_addr
            ),
            key=lambda node: node.addr,
        )
    )


def _reachable_nodes(graph: CFGGraph, entry: CFGNode) -> set[CFGNode]:
    """Return every graph node reachable from the extracted entry block."""

    reachable: set[CFGNode] = set()
    pending = deque([entry])
    while pending:
        node = pending.popleft()
        if node in reachable:
            continue
        reachable.add(node)
        pending.extend(graph.successors(node))
    return reachable


def _overlap_is_supported(
    project: Project | None,
    blocks: Mapping[int, BlockSpec],
    first: CFGNode,
    second: CFGNode,
) -> bool:
    """Return whether an overlap is a supported rejoining alternate stream."""

    if project is None:
        return False
    lower, upper = sorted((first, second), key=lambda node: node.addr)
    block = blocks.get(lower.addr)
    return block is not None and (
        alternate_block_entry_rejoin_addr(project, block, upper.addr) is not None
    )


def find_extracted_cfg_anomalies(
    graph: CFGGraph,
    bounds: FunctionBounds,
    func_addr: int,
    blocks: Mapping[int, BlockSpec],
    *,
    project: Project | None = None,
    recovered_roots: Iterable[int] = (),
) -> tuple[ExtractedCFGAnomaly, ...]:
    """Validate invariants that the extractor itself promises to establish.

    Unresolved indirect transfers are not anomalies: they remain explicit
    leaves until a table resolver proves their targets. The checks below cover
    only errors the bounded leader worklist should never leave behind.
    Explicit discovery roots exempt their regions from entry reachability,
    not from instruction, overlap, or edge validation.
    """

    anomalies: list[ExtractedCFGAnomaly] = []
    nodes = _normal_nodes(graph, func_addr)
    node_by_addr = {node.addr: node for node in nodes}
    instruction_owners: dict[int, CFGNode] = {}

    for index, node in enumerate(nodes):
        if node.size <= 0 or not node.instruction_addrs:
            anomalies.append(
                ExtractedCFGAnomaly(
                    "empty_block",
                    node.addr,
                    f"Extracted block {node.addr:#x} has no decoded instructions",
                )
            )
        if node.addr < bounds.addr or node_range_end(node) > bounds.end_addr:
            anomalies.append(
                ExtractedCFGAnomaly(
                    "block_out_of_bounds",
                    node.addr,
                    f"Extracted block {node.addr:#x} exceeds function bounds "
                    f"{bounds.addr:#x}-{bounds.end_addr:#x}",
                )
            )

        for other in nodes[index + 1 :]:
            if other.addr >= node_range_end(node):
                break
            if ranges_overlap(
                node.addr, node_range_end(node), other.addr, node_range_end(other)
            ) and not _overlap_is_supported(project, blocks, node, other):
                anomalies.append(
                    ExtractedCFGAnomaly(
                        "overlapping_blocks",
                        node.addr,
                        f"Extracted blocks at {node.addr:#x} and {other.addr:#x} overlap",
                    )
                )

        for insn_addr in node.instruction_addrs:
            previous = instruction_owners.setdefault(insn_addr, node)
            if previous is not node and not _overlap_is_supported(
                project, blocks, previous, node
            ):
                anomalies.append(
                    ExtractedCFGAnomaly(
                        "duplicate_instruction",
                        insn_addr,
                        f"Instruction {insn_addr:#x} appears in extracted blocks "
                        f"{previous.addr:#x} and {node.addr:#x}",
                    )
                )

        block = blocks.get(node.addr)
        if block is None:
            anomalies.append(
                ExtractedCFGAnomaly(
                    "missing_block_spec",
                    node.addr,
                    f"Extracted node {node.addr:#x} has no recovered block specification",
                )
            )
            continue

        if node.size != block.size:
            anomalies.append(
                ExtractedCFGAnomaly(
                    "block_size_mismatch",
                    node.addr,
                    f"Extracted block {node.addr:#x} has size {node.size:#x}, "
                    f"expected {block.size:#x}",
                )
            )
        if tuple(node.instruction_addrs) != block.instruction_addrs:
            anomalies.append(
                ExtractedCFGAnomaly(
                    "instruction_coverage_mismatch",
                    node.addr,
                    f"Extracted block {node.addr:#x} instruction coverage does not "
                    "match its recovered block specification",
                )
            )

        successors_by_addr = {
            successor.addr: successor for successor in graph.successors(node)
        }
        for target in block.direct_targets:
            destination = successors_by_addr.get(target)
            if destination is None:
                anomalies.append(
                    ExtractedCFGAnomaly(
                        "missing_direct_edge",
                        node.addr,
                        f"Extracted block {node.addr:#x} is missing direct edge to "
                        f"{target:#x}",
                    )
                )
            else:
                expected_jumpkind = (
                    "Ijk_Call" if block.jumpkind == "Ijk_Call" else "Ijk_Boring"
                )
                edge_data = graph.get_edge_data(node, destination) or {}
                if edge_data.get("jumpkind") != expected_jumpkind:
                    anomalies.append(
                        ExtractedCFGAnomaly(
                            "direct_edge_jumpkind_mismatch",
                            node.addr,
                            f"Extracted direct edge {node.addr:#x} -> {target:#x} has "
                            f"jumpkind {edge_data.get('jumpkind')!r}, expected "
                            f"{expected_jumpkind}",
                        )
                    )
            covering = next(
                (
                    candidate
                    for candidate in nodes
                    if candidate.addr < target < node_range_end(candidate)
                ),
                None,
            )
            if covering is not None and (
                target not in node_by_addr
                or not _overlap_is_supported(
                    project, blocks, covering, node_by_addr[target]
                )
            ):
                anomalies.append(
                    ExtractedCFGAnomaly(
                        "target_inside_block",
                        node.addr,
                        f"Extracted target {target:#x} from {node.addr:#x} lands "
                        f"inside block {covering.addr:#x}",
                    )
                )

        fallthrough = block.fallthrough_addr
        if fallthrough is not None and fallthrough in node_by_addr:
            destination = successors_by_addr.get(fallthrough)
            if destination is None:
                anomalies.append(
                    ExtractedCFGAnomaly(
                        "missing_fallthrough_edge",
                        node.addr,
                        f"Extracted block {node.addr:#x} is missing fallthrough edge "
                        f"to {fallthrough:#x}",
                    )
                )
            else:
                expected_jumpkind = (
                    "Ijk_FakeRet"
                    if block.jumpkind in {"Ijk_Call", "Ijk_Syscall"}
                    else "Ijk_Boring"
                )
                edge_data = graph.get_edge_data(node, destination) or {}
                if edge_data.get("jumpkind") != expected_jumpkind:
                    anomalies.append(
                        ExtractedCFGAnomaly(
                            "fallthrough_edge_jumpkind_mismatch",
                            node.addr,
                            f"Extracted fallthrough edge {node.addr:#x} -> "
                            f"{fallthrough:#x} has jumpkind "
                            f"{edge_data.get('jumpkind')!r}, expected "
                            f"{expected_jumpkind}",
                        )
                    )

    entry = node_by_addr.get(func_addr)
    if entry is None:
        anomalies.append(
            ExtractedCFGAnomaly(
                "missing_entry",
                func_addr,
                f"Extracted CFG has no entry at {func_addr:#x}",
            )
        )
    else:
        reachable = _reachable_nodes(graph, entry)
        for address in recovered_roots:
            if (root := node_by_addr.get(address)) is not None:
                reachable.update(_reachable_nodes(graph, root))
        for node in nodes:
            if node not in reachable:
                anomalies.append(
                    ExtractedCFGAnomaly(
                        "unreachable_block",
                        node.addr,
                        f"Extracted block {node.addr:#x} is unreachable from {func_addr:#x}",
                    )
                )

    return tuple(anomalies)
