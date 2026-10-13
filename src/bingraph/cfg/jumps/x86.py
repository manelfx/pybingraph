"""x86 PC-thunk verification and predecessor evidence for PIC table bases.

These predecessor queries establish the existing base and guard facts; generic
table planning owns row reads and the retained extent-estimation fallback.
Reusable thunk-body verification lives in helpers.abi rather than this CFG
provenance layer."""

from __future__ import annotations

import pyvex
from angr import Project
from angr.knowledge_plugins.cfg import CFGNode

from bingraph.cfg.graph import CFGGraph
from bingraph.cfg.graph import iter_graph_bound_nodes as _iter_graph_bound_nodes
from bingraph.cfg.graph import node_intersects_bounds as _node_intersects_bounds
from bingraph.cfg.graph import (
    node_is_materialized_cfg_node as _node_is_materialized_cfg_node,
)
from bingraph.cfg.graph import node_vex as _node_vex
from bingraph.cfg.jumps.expressions import (
    MAX_STATIC_JUMPTABLE_ENTRIES,
    _vex_register_with_displacement,
)
from bingraph.cfg.jumps.guards import _vex_guarded_index_upper_bound
from bingraph.cfg.models import FunctionBounds, StaticJumpTable
from bingraph.helpers.vex import (
    _vex_const_value,
    temporary_definitions,
)


def _x86_pc_thunk_predecessors(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    table: StaticJumpTable,
) -> tuple[CFGNode, ...]:
    """Locate an immediate i386 thunk continuation for the selected base register.

    Require a materialized in-function call ending exactly at ``node`` and a
    constant callee symbol with the matching GCC register-specific name. This
    predecessor relation serves the retained PIC planner; helpers.abi provides
    separate byte-level thunk verification for shared predecessor facts.
    """

    if project.arch.name != "X86" or project.arch.bits != 32:
        return ()
    if table.base_register_offset is None:
        return ()
    register_name = project.arch.register_names.get(table.base_register_offset)
    if register_name is None:
        return ()
    # GCC names 32-bit x86 thunks after the 16-bit register suffix: ``ebx``
    # is initialized by ``__x86.get_pc_thunk.bx``.
    thunk_register = register_name.removeprefix("e")
    expected_name = f"__x86.get_pc_thunk.{thunk_register}"

    predecessors: list[CFGNode] = []
    for predecessor in graph.predecessors(node):
        if not _node_is_materialized_cfg_node(predecessor):
            continue
        if not _node_intersects_bounds(predecessor, bounds):
            continue
        if predecessor.addr + predecessor.size != node.addr:
            continue
        vex = _node_vex(predecessor)
        if vex is None or vex.jumpkind != "Ijk_Call":
            continue
        definitions = temporary_definitions(vex)
        call_target = _vex_const_value(vex.next, definitions)
        if call_target is None:
            continue
        symbol = project.loader.find_symbol(call_target)
        if symbol is not None and symbol.name == expected_name:
            predecessors.append(predecessor)

    return tuple(predecessors)


def _x86_pc_thunk_base_addr(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    table: StaticJumpTable,
) -> int | None:
    """Recover the retained PIC base from a matching thunk and optional local add.

    Prefer an immediate matching call, whose continuation address is the thunk
    result. Otherwise inspect in-function thunk continuations for the supported
    base-register addition and require one agreeing value. This retains the
    adapter's local planning assumptions, not the shared engine's general
    must-reaching register analysis.
    """

    if project.arch.name != "X86" or project.arch.bits != 32:
        return None
    if table.base_register_offset is None:
        return None
    register_name = project.arch.register_names.get(table.base_register_offset)
    if register_name is None:
        return None
    # GCC names 32-bit x86 thunks after the 16-bit register suffix: ``ebx``
    # is initialized by ``__x86.get_pc_thunk.bx``.
    thunk_register = register_name.removeprefix("e")
    expected_name = f"__x86.get_pc_thunk.{thunk_register}"

    if _x86_pc_thunk_predecessors(project, graph, bounds, node, table):
        return node.addr

    values: set[int] = set()

    # Tables are often far from the prologue that initializes their PIC base.
    # GCC emits ``call __x86.get_pc_thunk.<reg>; add $offset, %reg``: the
    # fallthrough address is the thunk's result, so the following Add defines
    # the exact table base for every later use of the register in this function.
    for thunk_call in _iter_graph_bound_nodes(graph, bounds):
        vex = _node_vex(thunk_call)
        try:
            is_thunk_call = vex is not None and vex.jumpkind == "Ijk_Call"
        except AttributeError:
            continue
        if not is_thunk_call:
            continue
        definitions = temporary_definitions(vex)
        call_target = _vex_const_value(vex.next, definitions)
        symbol = project.loader.find_symbol(call_target) if call_target else None
        if symbol is None or symbol.name != expected_name:
            continue
        for successor in graph.successors(thunk_call):
            if not _node_is_materialized_cfg_node(successor):
                continue
            if successor.addr != thunk_call.addr + thunk_call.size:
                continue
            successor_vex = _node_vex(successor)
            if successor_vex is None:
                continue
            try:
                successor_defs = temporary_definitions(successor_vex)
            except (AttributeError, TypeError):
                continue
            for stmt in successor_vex.statements:
                if (
                    not isinstance(stmt, pyvex.stmt.Put)
                    or stmt.offset != table.base_register_offset
                ):
                    continue
                base = _vex_register_with_displacement(
                    stmt.data, successor_defs, successor_vex
                )
                if base is None or base[0] != (
                    table.base_register_offset,
                    table.base_bits,
                ):
                    continue
                values.add(successor.addr + base[1])

    return next(iter(values)) if len(values) == 1 else None


def _x86_pc_thunk_guarded_entry_count(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    table: StaticJumpTable,
) -> int | None:
    """Carry an unsigned bound through the matching thunk continuation.

    Inspect guards immediately before admitted thunk calls and match the
    table's index register/width. Require one agreeing inclusive endpoint,
    convert it to a row count and apply the common cap. Unrelated prologue
    checks or multiple conflicting bounds cannot authorize table reads.
    """

    if table.index_register_offset is None or table.index_bits is None:
        return None
    index_key = table.index_register_offset, table.index_bits
    upper_bounds: set[int] = set()
    for thunk_call in _x86_pc_thunk_predecessors(project, graph, bounds, node, table):
        for predecessor in graph.predecessors(thunk_call):
            if not _node_is_materialized_cfg_node(predecessor):
                continue
            if not _node_intersects_bounds(predecessor, bounds):
                continue
            vex = _node_vex(predecessor)
            if vex is None:
                continue
            upper_bound = _vex_guarded_index_upper_bound(
                vex, thunk_call.addr, index_key
            )
            if upper_bound is not None:
                upper_bounds.add(upper_bound)

    if len(upper_bounds) != 1:
        return None
    entry_count = next(iter(upper_bounds)) + 1
    return entry_count if entry_count <= MAX_STATIC_JUMPTABLE_ENTRIES else None
