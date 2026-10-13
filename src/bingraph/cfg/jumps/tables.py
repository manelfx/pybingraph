"""Generic finite jump-table planning and representation-aware target reads.

Compose local shapes with independently established bounds and base facts.
Architecture adapters keep their existing priority; this module returns plans
without inserting leaders or edges. The retained i386 extent approximation
is used only through its existing PC-thunk planning path."""

from __future__ import annotations

from collections import deque

import pyvex
from angr import Project
from angr.knowledge_plugins.cfg import CFGNode
from loguru import logger

from bingraph.cfg.graph import CFGGraph
from bingraph.cfg.graph import node_intersects_bounds as _node_intersects_bounds
from bingraph.cfg.graph import (
    node_is_materialized_cfg_node as _node_is_materialized_cfg_node,
)
from bingraph.cfg.graph import node_vex as _node_vex
from bingraph.cfg.jumps.expressions import (
    MAX_STATIC_JUMPTABLE_ENTRIES,
    _vex_direct_jump_table,
    _vex_relative_jump_table,
    is_direct_target_valid,
)
from bingraph.cfg.jumps.guards import (
    _guarded_jump_table_entry_count,
    _predecessor_clamped_index_values,
)
from bingraph.cfg.jumps.ppc import _plan_ppc64_toc_relative_ctr_table
from bingraph.cfg.jumps.x86 import (
    _x86_pc_thunk_base_addr,
    _x86_pc_thunk_guarded_entry_count,
)
from bingraph.cfg.models import FunctionBounds, StaticJumpTable, StaticJumpTablePlan
from bingraph.helpers.vex import _vex_const_value, temporary_definitions


def _constant_register_from_predecessors(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    register_offset: int,
) -> int | None:
    """Search backwards for one agreeing local constant base definition.

    Stop each path at its last matching PUT; a nonconstant write or failed
    lift declines the query. Traverse only materialized in-function nodes and
    deduplicate visits to loops. This helper recognizes constant expressions,
    not ABI call summaries or the validity of the eventual table contents.
    """

    definitions: set[int] = set()
    queue: deque[CFGNode] = deque([node])
    seen: set[CFGNode] = set()
    while queue:
        current = queue.popleft()
        if current in seen:
            continue
        seen.add(current)
        vex = _node_vex(current)
        if vex is None:
            return None

        tmp_definitions = temporary_definitions(vex)
        found_definition = False
        for stmt in reversed(vex.statements):
            if not isinstance(stmt, pyvex.stmt.Put) or stmt.offset != register_offset:
                continue
            value = _vex_const_value(stmt.data, tmp_definitions)
            if value is None:
                return None
            definitions.add(value)
            found_definition = True
            break

        if found_definition:
            continue
        queue.extend(
            predecessor
            for predecessor in graph.predecessors(current)
            if _node_is_materialized_cfg_node(predecessor)
            and _node_intersects_bounds(predecessor, bounds)
        )

    return next(iter(definitions)) if len(definitions) == 1 else None


def _in_function_jump_table_entry_count(
    project: Project,
    bounds: FunctionBounds,
    table: StaticJumpTable,
    base_addr: int,
) -> int | None:
    """Apply the retained i386 PIC extent approximation to adjacent rows.

    Decode at most the row cap with recorded signedness/endianness, stopping
    at the first destination outside this function. Require at least two
    admitted entries. This is not proof that targets are sorted or that later
    rows cannot return in-function; the planner restricts this estimate to its
    existing PC-thunk fallback rather than generalizing it to other tables.
    """

    table_addr = _jump_table_addr(base_addr, table)
    try:
        raw = project.loader.memory.load(
            table_addr, MAX_STATIC_JUMPTABLE_ENTRIES * table.entry_size
        )
    except Exception as exc:
        logger.debug(f"Custom CFG could not read jump table at {table_addr:#x}: {exc}")
        return None

    byteorder = "little" if table.endness == "Iend_LE" else "big"
    count = 0
    for offset in range(0, len(raw), table.entry_size):
        entry = int.from_bytes(
            raw[offset : offset + table.entry_size],
            byteorder=byteorder,
            signed=table.signed_entries,
        )
        target = _jump_table_target_addr(base_addr, table, entry)
        if not is_direct_target_valid(bounds, target):
            break
        count += 1
    return count if count >= 2 else None


def plan_static_jump_table(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node: CFGNode,
    *,
    allow_inline_index_values: bool = False,
    allow_masked_index_values: bool = False,
    allow_static_bases: bool = False,
    allow_guarded_expression_indices: bool = False,
    allow_predecessor_clamped_indices: bool = False,
) -> tuple[StaticJumpTablePlan | None, str | None]:
    """Return one fully proven static-table read plan for an indirect branch.

    This deliberately owns only architecture-neutral VEX recognition and the
    evidence needed to read a finite table. The builder adds proven targets
    as block leaders before its final graph materialization.

    Try the PPC adapter first, then relative/direct VEX shapes with the caller's
    opt-in policies. Compose finite shape domains and guard bounds, locate a
    static or reaching register base, and retain sparse row sets. The i386
    PC-thunk fallback may use its historical extent estimate when explicit
    bounds are absent; that exception is documented separately above.

    Return ``(plan, None)`` on success or an explanatory reason such as
    ``no_table_shape``, ``unknown_base`` or ``unbounded_index``. Reasons affect
    builder fallback/UJT handling; failed opt-in recognition must not erase
    the unresolved transfer. No leaders or edges are modified here.
    """

    vex = _node_vex(node)
    if vex is None:
        return None, "no_vex"

    ppc_plan = _plan_ppc64_toc_relative_ctr_table(project, graph, bounds, node, vex)
    if ppc_plan is not None:
        return ppc_plan, None

    table = _vex_relative_jump_table(
        vex,
        allow_inline_index_values=allow_inline_index_values,
        allow_masked_index_values=allow_masked_index_values,
        allow_guarded_expression_index=allow_guarded_expression_indices,
    ) or _vex_direct_jump_table(
        vex,
        allow_inline_index_values=allow_inline_index_values,
        allow_masked_index_values=allow_masked_index_values,
        allow_static_base=allow_static_bases,
        allow_guarded_expression_index=allow_guarded_expression_indices,
    )
    if table is None:
        # A table index naturally has the architecture's full register width
        # on 64-bit targets. Recognition remains safe because table reads
        # still require a separate finite range proof.
        table = _vex_relative_jump_table(
            vex,
            allow_full_width_index=True,
            allow_inline_index_values=allow_inline_index_values,
            allow_masked_index_values=allow_masked_index_values,
            allow_guarded_expression_index=allow_guarded_expression_indices,
        ) or _vex_direct_jump_table(
            vex,
            allow_full_width_index=True,
            allow_inline_index_values=allow_inline_index_values,
            allow_masked_index_values=allow_masked_index_values,
            allow_static_base=allow_static_bases,
            allow_guarded_expression_index=allow_guarded_expression_indices,
        )

    pic_base_addr = None
    if project.arch.name == "X86" and project.arch.bits == 32 and table is not None:
        pic_base_addr = _x86_pc_thunk_base_addr(project, graph, bounds, node, table)
    if table is None or table.base_bits != project.arch.bits:
        return None, "no_table_shape"

    entry_indices = table.index_values
    entry_count = _guarded_jump_table_entry_count(graph, bounds, node, table)
    if table.index_expression is not None and entry_count is None:
        # The optional expression matcher is only safe when its matching
        # predecessor guard proves a finite table. An s390 table-loaded
        # register branch has a known shape but an unbounded selector; it
        # must not seed speculative component reconnection.
        reason = (
            "unbounded_index"
            if vex.arch.name == "S390X" and vex.jumpkind == "Ijk_Ret"
            else "no_table_shape"
        )
        return None, reason
    if entry_indices is None:
        if entry_count is None and pic_base_addr is not None:
            entry_count = _x86_pc_thunk_guarded_entry_count(
                project, graph, bounds, node, table
            )
        if entry_count is not None:
            entry_indices = tuple(range(entry_count))
        elif allow_predecessor_clamped_indices:
            entry_indices = _predecessor_clamped_index_values(
                graph, bounds, node, table
            )
    elif entry_count is not None:
        # Independent finite facts compose by intersection. A bit mask limits
        # the index value and a predecessor guard can narrow it further.
        entry_indices = tuple(index for index in entry_indices if index < entry_count)

    base_addr = (
        table.static_base_addr if table.static_base_addr is not None else pic_base_addr
    )
    base_register_offset = table.base_register_offset
    if base_addr is None:
        if base_register_offset is None:
            return None, "unknown_base"
        base_addr = _constant_register_from_predecessors(
            graph, bounds, node, base_register_offset
        )
    if base_addr is None:
        return (
            None,
            "no_table_shape" if table.preserve_unresolved_fallback else "unknown_base",
        )

    if (
        entry_indices is None
        and table.entries_are_relative
        and pic_base_addr is not None
    ):
        entry_count = _in_function_jump_table_entry_count(
            project, bounds, table, base_addr
        )
        if entry_count is not None:
            entry_indices = tuple(range(entry_count))
    if not entry_indices:
        return (
            None,
            "no_table_shape"
            if table.preserve_unresolved_fallback
            else "unbounded_index",
        )

    return StaticJumpTablePlan(table, base_addr, entry_indices), None


def _jump_table_target_addr(base_addr: int, table: StaticJumpTable, entry: int) -> int:
    """Apply lifted target arithmetic to one correctly decoded table entry.

    Absolute entries already are addresses. Relative entries add the proven
    base/displacement with width wrapping and an optional alignment mask.
    The caller supplies entry signedness and validates the resulting target.
    """

    if not table.entries_are_relative:
        return entry

    mask = (1 << table.base_bits) - 1
    target = (base_addr + table.target_displacement + entry) & mask
    return target if table.target_and_mask is None else target & table.target_and_mask


def _jump_table_addr(base_addr: int, table: StaticJumpTable) -> int:
    """Compute the first-row address separately from branch-target arithmetic.

    Wrap base plus table displacement at its recorded address width. Relative
    tables can have different load and target displacements; conflating those
    would read unrelated bytes despite otherwise correct target arithmetic.
    """

    return (base_addr + table.table_displacement) & ((1 << table.base_bits) - 1)


def _read_static_jump_table_targets(
    project: Project,
    table: StaticJumpTable,
    base_addr: int,
    entry_indices: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Read every admitted row with exact width, signedness and endianness.

    Check row indices against the common cap and deduplicate/sort targets only
    after all reads succeed. One unreadable row rejects the entire result,
    rather than returning a misleading subset. Executable-target validation
    and code/data claims are separate caller responsibilities.
    """

    if (
        not entry_indices
        or len(entry_indices) > MAX_STATIC_JUMPTABLE_ENTRIES
        or entry_indices[0] < 0
        or entry_indices[-1] >= MAX_STATIC_JUMPTABLE_ENTRIES
    ):
        return None

    table_addr = _jump_table_addr(base_addr, table)
    byteorder = "little" if table.endness == "Iend_LE" else "big"
    targets: set[int] = set()
    try:
        for index in entry_indices:
            raw = project.loader.memory.load(
                table_addr + index * table.entry_size, table.entry_size
            )
            entry = int.from_bytes(
                raw,
                byteorder=byteorder,
                signed=table.signed_entries,
            )
            targets.add(_jump_table_target_addr(base_addr, table, entry))
    except Exception as exc:
        logger.debug(f"Custom CFG could not read jump table at {table_addr:#x}: {exc}")
        return None

    return tuple(sorted(targets))
