"""Graph and node primitives shared by custom CFG validation and repair."""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from typing import Any, Protocol, cast

from angr.analyses.cfg import CFGBase
from angr.knowledge_plugins.cfg import CFGNode
import pyvex

from .models import EdgeJumpKind, FunctionBounds


class CFGGraph(Protocol):
    """Public graph operations shared by NetworkX and angr's SpillingCFG."""

    def nodes(self) -> Iterable[CFGNode]: ...

    def edges(
        self, data: bool = False
    ) -> Iterable[tuple[CFGNode, CFGNode, dict[str, Any]]]: ...

    def predecessors(self, node: CFGNode) -> Iterable[CFGNode]: ...

    def successors(self, node: CFGNode) -> Iterable[CFGNode]: ...

    def in_degree(self, node: CFGNode) -> int: ...

    def has_edge(self, src: CFGNode, dst: CFGNode) -> bool: ...

    def get_edge_data(self, src: CFGNode, dst: CFGNode) -> dict[str, Any] | None: ...

    def add_node(self, node: CFGNode) -> None: ...

    def add_edge(self, src: CFGNode, dst: CFGNode, **attrs: Any) -> None: ...

    def remove_edge(self, src: CFGNode, dst: CFGNode) -> None: ...

    def remove_node(self, node: CFGNode) -> None: ...


def cfg_graph(cfg: CFGBase) -> CFGGraph:
    """Return the CFG's public graph wrapper used by custom repair."""

    return cast(CFGGraph, cfg.graph)


def node_vex(node: CFGNode) -> Any | None:
    """Return VEX for a native CFG node, or None when angr cannot lift it."""

    try:
        return cast(Any, node.block).vex
    # CFGFast can retain zero-sized or otherwise unliftable seed nodes. VEX is
    # optional for the callers of this helper, so preserve their skip behavior.
    except Exception:
        return None


def node_range_end(node) -> int:
    """Return the closed-open end address of one node."""

    return node.addr + max(getattr(node, "size", 0), 0)


def ranges_overlap(start_a: int, end_a: int, start_b: int, end_b: int) -> bool:
    """Return True when two closed-open address ranges overlap."""

    return start_a < end_b and start_b < end_a


def node_is_placeholder(node) -> bool:
    """Return True when the node is a custom placeholder awaiting repair."""

    return getattr(node, "size", 0) == 0 and str(getattr(node, "name", "")).startswith(
        "placeholder_"
    )


def node_is_simprocedure(node) -> bool:
    """Return True when `node` is a synthetic/simprocedure CFG node."""

    return getattr(node, "is_simprocedure", False)


def is_unresolvable_jump_target(node) -> bool:
    """Return True for angr's synthetic unresolved indirect-jump target node."""

    return (
        is_unresolvable_control_target(node)
        and getattr(node, "simprocedure_name", None) == "UnresolvableJumpTarget"
    )


def is_unresolvable_control_target(node) -> bool:
    """Return True for angr's unresolved indirect-call or jump placeholder."""

    return node_is_simprocedure(node) and getattr(node, "simprocedure_name", None) in {
        "UnresolvableCallTarget",
        "UnresolvableJumpTarget",
    }


def node_ends_in_indirect_jump(node) -> bool:
    """Return whether VEX identifies ``node`` as an indirect boring jump."""

    vex = node_vex(node)
    return (
        vex is not None
        and vex.jumpkind == "Ijk_Boring"
        and not isinstance(vex.next, pyvex.expr.Const)
    )


def vex_is_transparent_fallthrough_padding(vex: Any, addr: int, size: int) -> bool:
    """Return whether VEX only self-assigns registers before falling through.

    Compilers commonly use instructions such as ``mov reg, reg`` and
    ``lea reg, [reg]`` for alignment.  Their VEX blocks contain only IMarks,
    temporary GETs, and PUTs that write the same register value back.  These
    blocks are safe to skip as *unresolved* indirect-jump candidates, but not
    when a known branch or table entry explicitly targets them.
    """

    if size <= 0:
        return False
    if vex.jumpkind != "Ijk_Boring":
        return False
    next_addr = getattr(getattr(vex.next, "con", None), "value", None)
    if next_addr != addr + size:
        return False

    definitions: dict[int, Any] = {}
    saw_instruction = False
    for statement in vex.statements:
        if isinstance(statement, pyvex.stmt.IMark):
            saw_instruction = True
            continue
        if isinstance(statement, pyvex.stmt.WrTmp):
            if not isinstance(statement.data, pyvex.expr.Get):
                return False
            definitions[statement.tmp] = statement.data
            continue
        if not isinstance(statement, pyvex.stmt.Put):
            return False

        value = statement.data
        while isinstance(value, pyvex.expr.RdTmp):
            value = definitions.get(value.tmp)
            if value is None:
                return False
        if not isinstance(value, pyvex.expr.Get) or value.offset != statement.offset:
            return False

    return saw_instruction


def node_is_transparent_fallthrough_padding(node) -> bool:
    """Return whether ``node`` is transparent padding before its fallthrough."""

    vex = node_vex(node)
    return vex is not None and vex_is_transparent_fallthrough_padding(
        vex, node.addr, getattr(node, "size", 0)
    )


def node_is_materialized_cfg_node(node) -> bool:
    """Return True for normal nodes that are neither simprocedures nor placeholders."""

    return not node_is_simprocedure(node) and not node_is_placeholder(node)


def node_intersects_bounds(node, bounds: FunctionBounds) -> bool:
    """Return True when a node overlaps the current function address range."""

    if node_is_simprocedure(node):
        return False
    return ranges_overlap(node.addr, node_range_end(node), bounds.addr, bounds.end_addr)


def iter_graph_bound_nodes(graph: CFGGraph, bounds: FunctionBounds):
    """Yield live graph nodes that overlap the current function bounds."""

    for node in graph.nodes():
        if node_intersects_bounds(node, bounds):
            yield node


def nodes_at_addr(graph: CFGGraph, bounds: FunctionBounds, addr: int) -> list[CFGNode]:
    """Return all non-simprocedure nodes in bounds that begin at ``addr``."""

    return [node for node in iter_graph_bound_nodes(graph, bounds) if node.addr == addr]


def covering_nodes(graph: CFGGraph, bounds: FunctionBounds, addr: int) -> list[CFGNode]:
    """Return all in-bounds node ranges that cover ``addr``."""

    return [
        node
        for node in iter_graph_bound_nodes(graph, bounds)
        if node.addr <= addr < node_range_end(node)
    ]


def add_successor_edge(
    graph: CFGGraph,
    src: CFGNode,
    dst: CFGNode,
    jumpkind: EdgeJumpKind,
    *,
    unresolved_indirect: bool = False,
) -> bool:
    """Add an edge if it is not already present with matching metadata."""

    if graph.has_edge(src, dst):
        edge_data = graph.get_edge_data(src, dst) or {}
        if (
            edge_data.get("jumpkind") == jumpkind
            and edge_data.get("unresolved_indirect", False) == unresolved_indirect
        ):
            return False
    graph.add_edge(
        src,
        dst,
        jumpkind=jumpkind,
        unresolved_indirect=unresolved_indirect,
    )
    return True


def remove_nodes(graph: CFGGraph, nodes: Iterable[CFGNode]) -> None:
    """Remove a batch of nodes through the graph wrapper's public API."""

    for node in nodes:
        graph.remove_node(node)


def cleanup_unreachable_function_nodes(
    graph: CFGGraph, bounds: FunctionBounds, func_addr: int
) -> int:
    """Remove unreachable in-bounds nodes and return how many were pruned."""

    entry_nodes = nodes_at_addr(graph, bounds, func_addr)
    if not entry_nodes:
        return 0

    reachable: set[CFGNode] = set()
    queue: deque[CFGNode] = deque(entry_nodes)
    while queue:
        node = queue.popleft()
        if node in reachable:
            continue
        reachable.add(node)
        queue.extend(graph.successors(node))

    stale_nodes = [
        node
        for node in list(graph.nodes())
        if node_intersects_bounds(node, bounds) and node not in reachable
    ]
    remove_nodes(graph, stale_nodes)
    return len(stale_nodes)
