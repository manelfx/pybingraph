"""Graph and node primitives shared by custom CFG construction and validation."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any, Protocol, cast

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
    """Return whether an empty node carries a placeholder label."""

    return getattr(node, "size", 0) == 0 and str(getattr(node, "name", "")).startswith(
        "placeholder_"
    )


def node_is_simprocedure(node) -> bool:
    """Return True when `node` is a synthetic/simprocedure CFG node."""

    return getattr(node, "is_simprocedure", False)


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


def add_successor_edge(
    graph: CFGGraph,
    src: CFGNode,
    dst: CFGNode,
    jumpkind: EdgeJumpKind,
    *,
    unresolved_indirect: bool = False,
    exceptional: bool = False,
) -> bool:
    """Add an edge if it is not already present with matching metadata."""

    if graph.has_edge(src, dst):
        edge_data = graph.get_edge_data(src, dst) or {}
        if (
            edge_data.get("jumpkind") == jumpkind
            and edge_data.get("unresolved_indirect", False) == unresolved_indirect
            and edge_data.get("exceptional", False) == exceptional
        ):
            return False
    graph.add_edge(
        src,
        dst,
        jumpkind=jumpkind,
        unresolved_indirect=unresolved_indirect,
        exceptional=exceptional,
    )
    return True
