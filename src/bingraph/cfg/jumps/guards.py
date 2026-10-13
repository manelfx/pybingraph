"""Finite selector bounds from guards, clamping and exact reaching definitions.

Taken and default edges retain their original polarity. Cross-block and stack
queries reject unknown incoming paths rather than narrowing them by guesses.
All walks retain their original node and table-domain limits."""

from __future__ import annotations

from collections import deque
from typing import Any

import pyvex
from angr.knowledge_plugins.cfg import CFGNode

from bingraph.cfg.graph import CFGGraph
from bingraph.cfg.graph import node_intersects_bounds as _node_intersects_bounds
from bingraph.cfg.graph import (
    node_is_materialized_cfg_node as _node_is_materialized_cfg_node,
)
from bingraph.cfg.graph import node_vex as _node_vex
from bingraph.cfg.jumps.expressions import (
    MAX_STATIC_JUMPTABLE_ENTRIES,
    _vex_expr_key,
    _vex_guard_width_view_key,
    _vex_index_key,
    _vex_is_zero_extension_from,
    _vex_low_bits_source,
    _vex_low_masked_source,
    _vex_low_register_view_keys,
    _vex_static_int,
)
from bingraph.cfg.models import FunctionBounds, StaticJumpTable
from bingraph.helpers.vex import (
    _resolve_vex_expr,
    _vex_const_value,
    _vex_get_key,
    _vex_last_put,
    _vex_width_conversion,
    temporary_definitions,
)

# Bound must-reaching walks independently of the number of admitted table rows.
MAX_CROSS_BLOCK_SELECTOR_PROOF_NODES = 128


def _vex_guarded_index_upper_bound(
    vex, target_addr: int, index_key: tuple[int, int]
) -> int | None:
    """Return a proven unsigned bound for a VEX path entering ``target_addr``.

    A compiler may encode the dispatcher as either the taken ``Exit`` or the
    default ``NEXT`` path.  In the latter form, the exit guard is the inverse
    of the table-domain condition, such as ``limit <u index``.
    """

    definitions = temporary_definitions(vex)
    for exit_index, stmt in enumerate(vex.statements):
        if not isinstance(stmt, pyvex.stmt.Exit):
            continue
        if getattr(stmt.dst, "value", None) != target_addr:
            continue

        upper_bound = _vex_guard_upper_bound(
            stmt.guard,
            index_key,
            definitions,
            vex,
            vex.statements[:exit_index],
            index_on_left=True,
        )
        if upper_bound is not None:
            return upper_bound

    if _vex_const_value(vex.next, definitions) != target_addr:
        return None

    bounds = {
        upper_bound
        for exit_index, stmt in enumerate(vex.statements)
        if isinstance(stmt, pyvex.stmt.Exit)
        if (
            upper_bound := _vex_guard_upper_bound(
                stmt.guard,
                index_key,
                definitions,
                vex,
                vex.statements[:exit_index],
                index_on_left=False,
            )
        )
        is not None
    }
    return next(iter(bounds)) if len(bounds) == 1 else None


def _vex_guarded_expression_upper_bound(
    vex, target_addr: int, index_expression: tuple[Any, ...]
) -> int | None:
    """Collect a consistent bound for an expression on the dispatcher edge.

    Match taken Exits and complementary constraints when NEXT reaches
    ``target_addr``. Require the guarded expression to match the selector's
    structural key, not merely its register spelling. Return one inclusive
    upper bound, or ``None`` without usable agreeing evidence.
    """

    definitions = temporary_definitions(vex)
    bounds: set[int] = set()
    for exit_index, stmt in enumerate(vex.statements):
        if not isinstance(stmt, pyvex.stmt.Exit):
            continue
        if getattr(stmt.dst, "value", None) != target_addr:
            continue
        upper_bound = _vex_guard_expression_upper_bound(
            stmt.guard,
            index_expression,
            definitions,
            vex,
            index_on_left=True,
        )
        if upper_bound is not None:
            bounds.add(upper_bound)

    if _vex_const_value(vex.next, definitions) == target_addr:
        for stmt in vex.statements:
            if not isinstance(stmt, pyvex.stmt.Exit):
                continue
            upper_bound = _vex_guard_expression_upper_bound(
                stmt.guard,
                index_expression,
                definitions,
                vex,
                index_on_left=False,
            )
            if upper_bound is not None:
                bounds.add(upper_bound)

    return next(iter(bounds)) if len(bounds) == 1 else None


def _vex_unsigned_guard_comparison(guard, definitions: dict[int, Any]):
    """Unwrap VEX's nonzero test around an unsigned branch comparison."""

    guard = _resolve_vex_expr(guard, definitions)
    while isinstance(guard, pyvex.expr.Unop):
        guard = _resolve_vex_expr(guard.args[0], definitions)
    if not isinstance(guard, pyvex.expr.Binop) or not guard.op.startswith("Iop_CmpNE"):
        return guard

    left, right = guard.args
    left_value = _vex_const_value(left, definitions)
    right_value = _vex_const_value(right, definitions)
    if left_value == 0 and right_value is None:
        condition = right
    elif right_value == 0 and left_value is None:
        condition = left
    else:
        return guard
    condition = _resolve_vex_expr(condition, definitions)
    while isinstance(condition, pyvex.expr.Unop):
        condition = _resolve_vex_expr(condition.args[0], definitions)
    return condition


def _vex_guard_upper_bound(
    guard,
    index_key: tuple[int, int],
    definitions: dict[int, Any],
    vex,
    preceding_statements: tuple[Any, ...] | list[Any],
    *,
    index_on_left: bool,
) -> int | None:
    """Match an unsigned guard to the index view and derive its upper endpoint.

    ``index_on_left`` selects the supported taken/complementary interpretation.
    Normalize compatible shifts, then verify register/view identity against
    preceding writes. Require a constant bound and decline negative endpoints;
    the machine register name alone does not prove value equivalence.
    """

    guard = _vex_unsigned_guard_comparison(guard, definitions)
    if not isinstance(guard, pyvex.expr.Binop) or not guard.op.endswith("U"):
        return None

    index_expr, bound_expr = (
        (guard.args[0], guard.args[1])
        if index_on_left
        else (guard.args[1], guard.args[0])
    )
    index_expr, bound_expr = _vex_unsigned_shifted_compare_operands(
        index_expr, bound_expr, definitions, vex
    )
    if not _vex_guard_matches_index_register(
        index_expr, index_key, definitions, vex, preceding_statements
    ):
        return None
    return _unsigned_comparison_upper_bound(
        guard.op, _vex_static_int(bound_expr, definitions), index_on_left
    )


def _unsigned_comparison_upper_bound(
    op: str, bound: int | None, index_on_left: bool = True
) -> int | None:
    """Convert an unsigned comparison into its inclusive selector bound.

    With the selector on the right, callers are bounding the false/default
    path rather than the taken path. Matching operands remains their job.
    """

    if bound is None:
        return None
    if "CmpLT" in op:
        upper_bound = bound - 1 if index_on_left else bound
    elif "CmpLE" in op:
        upper_bound = bound if index_on_left else bound - 1
    else:
        return None
    return upper_bound if upper_bound >= 0 else None


def _vex_guard_range(guard, definitions: dict[int, Any]) -> tuple[int, ...] | None:
    """Expand a matched unsigned upper bound into a complete capped domain.

    The caller owns selector identity and polarity. Compute the constant bound
    and expand ``0..upper``; empty or oversized intervals return ``None`` rather
    than an arbitrary prefix that might suppress a real unresolved target.
    """

    upper_bound = _unsigned_comparison_upper_bound(
        guard.op, _vex_static_int(guard.args[1], definitions)
    )
    if upper_bound is None or upper_bound >= MAX_STATIC_JUMPTABLE_ENTRIES:
        return None
    return tuple(range(upper_bound + 1))


def _vex_unsigned_shifted_compare_operands(expr, bound, definitions, vex):
    """Unwrap equal-width left shifts used for narrow x86 unsigned flags.

    VEX represents an x86 comparison such as ``cmp ax, 43`` as a 64-bit
    unsigned comparison after shifting both operands left by 48 bits.  The
    shift preserves unsigned ordering when the unshifted value fits below the
    discarded high bits, so recover the original operands before matching the
    guarded table index.
    """

    expr = _resolve_vex_expr(expr, definitions)
    bound = _resolve_vex_expr(bound, definitions)
    if not (
        isinstance(expr, pyvex.expr.Binop)
        and isinstance(bound, pyvex.expr.Binop)
        and expr.op.startswith("Iop_Shl")
        and bound.op == expr.op
    ):
        return expr, bound
    shift = _vex_const_value(expr.args[1], definitions)
    if shift is None or shift != _vex_const_value(bound.args[1], definitions):
        return expr, bound
    source = _resolve_vex_expr(expr.args[0], definitions)
    source_bits = source.result_size(vex.tyenv)
    while (
        (conversion := _vex_width_conversion(source)) is not None
        and conversion[2] == "U"
        and conversion[1] == source_bits
    ):
        source_bits = conversion[0]
        source = _resolve_vex_expr(source.args[0], definitions)
    result_bits = expr.result_size(vex.tyenv)
    if shift <= 0 or shift >= result_bits or source_bits > result_bits - shift:
        return expr, bound
    return source, _resolve_vex_expr(bound.args[0], definitions)


def _vex_guard_expression_upper_bound(
    guard,
    index_expression: tuple[Any, ...],
    definitions: dict[int, Any],
    vex,
    *,
    index_on_left: bool,
) -> int | None:
    """Return an unsigned guard bound for one exact non-register selector."""

    guard = _resolve_vex_expr(guard, definitions)
    while isinstance(guard, pyvex.expr.Unop):
        guard = _resolve_vex_expr(guard.args[0], definitions)
    if not isinstance(guard, pyvex.expr.Binop) or not guard.op.endswith("U"):
        return None

    index_expr, bound_expr = (
        (guard.args[0], guard.args[1])
        if index_on_left
        else (guard.args[1], guard.args[0])
    )
    guarded_key = _vex_normalize_guard_expression_key(
        _vex_expr_key(index_expr, definitions)
    )
    if guarded_key != index_expression and guarded_key != _vex_key_at_block_exit(
        index_expression, definitions, vex
    ):
        return None
    return _unsigned_comparison_upper_bound(
        guard.op, _vex_static_int(bound_expr, definitions), index_on_left
    )


def _vex_normalize_guard_expression_key(
    key: tuple[Any, ...] | None,
) -> tuple[Any, ...] | None:
    """Normalize the zero-extended value VEX narrows again for a guard."""

    if (
        key is not None
        and key[:2] == ("unop", "Iop_64to32")
        and key[2][:2] == ("unop", "Iop_32Uto64")
    ):
        return key[2]
    return key


def _vex_key_at_block_exit(
    key: tuple[Any, ...], definitions: dict[int, Any], vex
) -> tuple[Any, ...] | None:
    """Project an expression through the predecessor's final register writes.

    This associates a dispatch read with a guard's earlier lifted expression.
    Replace outer GETs once: nested GETs in a replacement observe assignment-
    time state and must not be rewritten using later PUTs. Unsupported key
    forms remain unknown instead of becoming a fabricated matching expression.
    """

    register_values = {
        statement.offset: _vex_expr_key(statement.data, definitions)
        for statement in vex.statements
        if isinstance(statement, pyvex.stmt.Put)
    }

    def rewrite(value: tuple[Any, ...], seen: frozenset[int]) -> tuple[Any, ...] | None:
        """Rewrite outer reads without changing assignment-time nested operands."""

        tag = value[0]
        if tag == "get":
            offset = value[1]
            replacement = register_values.get(offset)
            if replacement is None or offset in seen:
                return value
            # ``replacement`` reads the register state at the assignment, not
            # the block exit. Do not rewrite nested GETs through later PUTs.
            return replacement
        if tag == "const":
            return value
        if tag in {"unop", "load"}:
            prefix, argument = value[:-1], value[-1]
            rewritten = rewrite(argument, seen)
            return (*prefix, rewritten) if rewritten is not None else None
        if tag in {"binop", "ccall"}:
            prefix = value[:2]
            arguments = tuple(rewrite(argument, seen) for argument in value[2:])
            return (*prefix, *arguments) if all(arguments) else None
        if tag == "ite":
            arguments = tuple(rewrite(argument, seen) for argument in value[1:])
            return (tag, *arguments) if all(arguments) else None
        return None

    return rewrite(key, frozenset())


def _vex_guarded_index_values(
    guard,
    index_key: tuple[int, int],
    definitions: dict[int, Any],
    vex,
) -> tuple[int, ...] | None:
    """Return the finite domain proven by one unsigned index guard."""

    guard = _resolve_vex_expr(guard, definitions)
    while isinstance(guard, pyvex.expr.Unop):
        guard = _resolve_vex_expr(guard.args[0], definitions)
    if not isinstance(guard, pyvex.expr.Binop) or not guard.op.endswith("U"):
        return None
    if not _vex_guard_matches_index_register(
        guard.args[0], index_key, definitions, vex, vex.statements
    ):
        return None
    return _vex_guard_range(guard, definitions)


def _vex_guarded_expression_values(
    guard,
    index_expr,
    definitions: dict[int, Any],
) -> tuple[int, ...] | None:
    """Return a finite domain when an unsigned guard bounds one exact expression."""

    guard = _resolve_vex_expr(guard, definitions)
    while isinstance(guard, pyvex.expr.Unop):
        guard = _resolve_vex_expr(guard.args[0], definitions)
    if not isinstance(guard, pyvex.expr.Binop) or not guard.op.endswith("U"):
        return None
    guarded_expr = _resolve_vex_expr(guard.args[0], definitions)
    index_expr = _resolve_vex_expr(index_expr, definitions)
    if guarded_expr is None or index_expr is None:
        return None
    guarded_key = _vex_expr_key(guarded_expr, definitions)
    index_key = _vex_expr_key(index_expr, definitions)
    if guarded_key is None or index_key is None or guarded_key != index_key:
        return None
    return _vex_guard_range(guard, definitions)


def _vex_guard_matches_index_register(
    expr,
    index_key: tuple[int, int],
    definitions: dict[int, Any],
    vex,
    preceding_statements: tuple[Any, ...] | list[Any],
) -> bool:
    """Match a guard to the dispatch register using explicit view evidence.

    Recognize supported reads, masks, zero extensions, narrow views and shifts.
    Consult preceding PUTs when equivalence depends on an assignment. Retain
    evidence for the consumed width; comparing low bits does not establish
    unknown high bits of another, wider selector value.
    """

    if _vex_index_key(expr, definitions, vex, allow_full_width=True) == index_key:
        return True

    resolved_expr = _resolve_vex_expr(expr, definitions)
    if resolved_expr is None:
        return False
    guard_bits = resolved_expr.result_size(vex.tyenv)
    if getattr(getattr(vex, "arch", None), "name", None) in {"AMD64", "X86"} and (
        masked_source := _vex_low_masked_source(resolved_expr, definitions, vex.tyenv)
    ):
        resolved_expr, guard_bits = masked_source
    guard_source = _vex_low_bits_source(
        resolved_expr, definitions, vex.tyenv, guard_bits
    )
    low_view_key = _vex_get_key(guard_source, definitions, vex)
    low_views = _vex_low_register_view_keys(index_key, guard_bits, vex)
    later_alias_write = False
    for stmt in reversed(preceding_statements):
        if not isinstance(stmt, pyvex.stmt.Put):
            continue
        if stmt.offset != index_key[0]:
            if index_key[0] < stmt.offset < index_key[0] + index_key[1] // 8:
                later_alias_write = True
            continue
        value = _resolve_vex_expr(stmt.data, definitions)
        if value is resolved_expr or value == resolved_expr:
            return True
        if (
            _vex_width_conversion(resolved_expr) == (index_key[1], guard_bits, "U")
            and _resolve_vex_expr(resolved_expr.args[0], definitions) == value
        ):
            # The branch compares the zero-extended value just written to
            # this narrow register view (for example s390 ``clijle``).
            return True
        if _vex_is_zero_extension_from(value, definitions, guard_bits, index_key[1]):
            if _vex_guard_width_view_key(
                value, definitions, vex.tyenv, guard_bits
            ) == _vex_guard_width_view_key(
                resolved_expr, definitions, vex.tyenv, guard_bits
            ):
                return True
        if (
            not later_alias_write
            and low_view_key in low_views
            and low_view_key[1] == guard_bits
            and _vex_is_zero_extension_from(
                value, definitions, guard_bits, index_key[1]
            )
        ):
            # A later low-register read observes the zero-extended full write.
            return True
        if guard_source is None:
            return False
        value_source = _vex_low_bits_source(value, definitions, vex.tyenv, guard_bits)
        if value_source is not guard_source and value_source != guard_source:
            return False
        if _vex_is_zero_extension_from(value, definitions, guard_bits, index_key[1]):
            return True
        return _vex_is_right_shift_narrowed_value(
            value, definitions, vex.tyenv, guard_bits, index_key[1]
        )
    return False


def _vex_is_right_shift_narrowed_value(
    expr, definitions: dict[int, Any], tyenv, bits: int, register_bits: int
) -> bool:
    """Return whether a logical right shift leaves at most ``bits`` values."""

    expr = _resolve_vex_expr(expr, definitions)
    if not isinstance(expr, pyvex.expr.Binop) or not expr.op.startswith("Iop_Shr"):
        return False
    shift = _vex_const_value(expr.args[1], definitions)
    return (
        shift is not None
        and expr.result_size(tyenv) == register_bits
        and register_bits - shift <= bits
    )


def _guarded_jump_table_entry_count(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    table: StaticJumpTable,
) -> int | None:
    """Choose the supported guard/reaching-definition proof for a row count.

    Expression indices, zero-extended stack reloads and register indices have
    different evidence requirements. Register checks inspect compatible views
    and local guards, with a bounded reaching-zero-extension path where
    supported. Require one consistent admitted bound and the common row cap;
    this helper does not infer extent from adjacent destination-looking bytes.
    """

    if table.index_register_offset is None and table.index_expression is None:
        return None
    bounds_found: set[int] = set()
    for predecessor in graph.predecessors(node):
        if not _node_is_materialized_cfg_node(predecessor):
            continue
        if not _node_intersects_bounds(predecessor, bounds):
            continue
        vex = _node_vex(predecessor)
        if vex is None:
            continue
        if table.index_expression is not None:
            upper_bound = _vex_guarded_expression_upper_bound(
                vex, node.addr, table.index_expression
            )
        elif table.index_register_offset is not None and table.index_bits is not None:
            index_key = (table.index_register_offset, table.index_bits)
            upper_bound = _vex_guarded_index_upper_bound(vex, node.addr, index_key)
            if upper_bound is None and table.index_low_bits is not None:
                alias_bounds = {
                    bound
                    for alias in _vex_low_register_view_keys(
                        index_key, table.index_low_bits, vex
                    )
                    if (bound := _vex_guarded_index_upper_bound(vex, node.addr, alias))
                    is not None
                }
                if len(alias_bounds) == 1:
                    upper_bound = next(iter(alias_bounds))
        else:
            continue
        if upper_bound is not None:
            bounds_found.add(upper_bound)

    if len(bounds_found) == 1:
        upper_bound = next(iter(bounds_found))
        entry_count = upper_bound + 1
        return entry_count if entry_count <= MAX_STATIC_JUMPTABLE_ENTRIES else None
    if (
        cross_block_entry_count := _cross_block_zero_extended_selector_entry_count(
            graph, bounds, node, table
        )
    ) is not None:
        return cross_block_entry_count
    if table.index_expression is None:
        return None
    return _zero_extended_stack_selector_entry_count(
        graph, bounds, node, table.index_expression
    )


def _vex_clamped_index_values(expr, definitions) -> tuple[int, ...] | None:
    """Prove the finite domain of an unsigned conditional-move clamp.

    For ``index >= limit ? fallback : index``, the unchanged arm can only
    contain values below ``limit``. A finite fallback completes the proof.
    """

    expr = _resolve_vex_expr(expr, definitions)
    if not isinstance(expr, pyvex.expr.ITE):
        return None
    condition = _resolve_vex_expr(expr.cond, definitions)
    while (
        isinstance(condition, pyvex.expr.Unop)
        and (conversion := _vex_width_conversion(condition)) is not None
        and 1 in conversion[:2]
    ):
        condition = _resolve_vex_expr(condition.args[0], definitions)
    if not isinstance(condition, pyvex.expr.Binop):
        return None
    if "CmpLE" in condition.op and condition.op.endswith("U"):
        inclusive = False
    elif "CmpLT" in condition.op and condition.op.endswith("U"):
        inclusive = True
    else:
        return None
    limit = _vex_static_int(condition.args[0], definitions)
    selector = condition.args[1]
    unchanged = expr.iffalse
    fallback = _vex_static_int(expr.iftrue, definitions)
    count = limit + int(inclusive) if limit is not None else None
    if (
        count is None
        or not 0 < count <= MAX_STATIC_JUMPTABLE_ENTRIES
        or not isinstance(selector, pyvex.expr.RdTmp)
        or not isinstance(unchanged, pyvex.expr.RdTmp)
        or unchanged.tmp != selector.tmp
        or fallback is None
        or not 0 <= fallback < MAX_STATIC_JUMPTABLE_ENTRIES
    ):
        return None
    return tuple(sorted({*range(count), fallback}))


def _predecessor_clamped_index_values(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    table: StaticJumpTable,
) -> tuple[int, ...] | None:
    """Union clamp domains only when every immediate predecessor proves one.

    Each materialized in-function boring predecessor must write the full index
    before any side exit using the supported conditional-select clamp form.
    Missing writes, unsupported transfers or an oversized union decline the
    whole proof; one clamp cannot erase an unbounded incoming path.
    """

    if table.index_register_offset is None or table.index_bits is None:
        return None
    predecessors = tuple(graph.predecessors(node))
    if not predecessors:
        return None
    values: set[int] = set()
    for predecessor in predecessors:
        if not _node_is_materialized_cfg_node(
            predecessor
        ) or not _node_intersects_bounds(predecessor, bounds):
            return None
        vex = _node_vex(predecessor)
        if vex is None or vex.jumpkind != "Ijk_Boring":
            return None
        definitions = temporary_definitions(vex)
        index_write = None
        saw_exit = False
        for statement in vex.statements:
            if isinstance(statement, pyvex.stmt.Exit):
                saw_exit = True
            if (
                isinstance(statement, pyvex.stmt.Put)
                and statement.offset == table.index_register_offset
            ):
                if saw_exit:
                    return None
                index_write = statement
        if (
            index_write is None
            or index_write.data.result_size(vex.tyenv) != table.index_bits
        ):
            return None
        proven = _vex_clamped_index_values(index_write.data, definitions)
        if proven is None:
            return None
        values.update(proven)
        if len(values) > MAX_STATIC_JUMPTABLE_ENTRIES:
            return None
    return tuple(sorted(values))


def _cross_block_zero_extended_selector_entry_count(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    table: StaticJumpTable,
) -> int | None:
    """Bound a later table index materialized after earlier exact dispatches.

    A compiler can guard a narrow register, zero-extend it into the eventual
    table index, and pass through another already-proven jump table before the
    second dispatcher. This bounded reverse proof accepts that shape only when
    every in-function path to the later table reaches one zext definition
    without another index write or a call boundary.
    """

    if table.index_register_offset is None or table.index_bits is None:
        return None
    index_key = (table.index_register_offset, table.index_bits)
    definition = _must_reaching_zero_extended_register_definition(
        graph, bounds, node, index_key
    )
    if definition is None:
        return None
    definition_node, source_key = definition

    upper_bounds: set[int] = set()
    predecessors = tuple(graph.predecessors(definition_node))
    if not predecessors:
        return None
    for predecessor in predecessors:
        if not (
            _node_is_materialized_cfg_node(predecessor)
            and _node_intersects_bounds(predecessor, bounds)
        ):
            return None
        vex = _node_vex(predecessor)
        if vex is None:
            return None
        upper_bound = _vex_guarded_index_upper_bound(
            vex, definition_node.addr, source_key
        )
        if upper_bound is None:
            return None
        upper_bounds.add(upper_bound)

    if len(upper_bounds) != 1:
        return None
    entry_count = next(iter(upper_bounds)) + 1
    return entry_count if entry_count <= MAX_STATIC_JUMPTABLE_ENTRIES else None


def _must_reaching_zero_extended_register_definition(
    graph: CFGGraph,
    bounds: FunctionBounds,
    dispatcher,
    index_key: tuple[int, int],
) -> tuple[CFGNode, tuple[int, int]] | None:
    """Find agreeing zero-extension definitions by bounded backward traversal.

    Stop at a matching register write, call/foreign-node barrier or entry
    without a definition. Traversed paths must identify the same supported
    narrow source. The node cap makes large or cyclic searches inconclusive,
    not permission to reuse an arbitrary non-reaching definition.
    """

    pending = deque(graph.predecessors(dispatcher))
    seen = set()
    definitions: set[tuple[CFGNode, tuple[int, int]]] = set()
    while pending:
        candidate = pending.popleft()
        if candidate in seen:
            continue
        seen.add(candidate)
        if len(seen) > MAX_CROSS_BLOCK_SELECTOR_PROOF_NODES:
            return None
        if not (
            _node_is_materialized_cfg_node(candidate)
            and _node_intersects_bounds(candidate, bounds)
        ):
            return None
        vex = _node_vex(candidate)
        if vex is None or vex.jumpkind != "Ijk_Boring":
            return None

        source_key = _vex_zero_extended_register_source(vex, index_key)
        if source_key is not None:
            definitions.add((candidate, source_key))
            continue
        if _vex_last_put(vex, index_key[0]) is not None:
            return None
        if candidate.addr == bounds.addr:
            return None
        predecessors = tuple(graph.predecessors(candidate))
        if not predecessors:
            return None
        pending.extend(predecessors)

    return next(iter(definitions)) if len(definitions) == 1 else None


def _vex_zero_extended_register_source(
    vex, index_key: tuple[int, int]
) -> tuple[int, int] | None:
    """Return the narrow source of a block's final full-width zext register put."""

    statement = _vex_last_put(vex, index_key[0])
    if statement is None:
        return None
    definitions = temporary_definitions(vex)
    expression = _resolve_vex_expr(statement.data, definitions)
    current_bits = index_key[1]
    while True:
        conversion = _vex_width_conversion(expression)
        if conversion is None:
            return None
        source_bits, destination_bits, signedness = conversion
        if destination_bits != current_bits or signedness != "U":
            return None
        expression = _resolve_vex_expr(expression.args[0], definitions)
        source_key = _vex_get_key(expression, definitions, vex)
        if source_key is not None:
            return source_key if source_key[1] == source_bits else None
        if source_bits >= current_bits:
            return None
        current_bits = source_bits


def _zero_extended_stack_selector_entry_count(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    index_expression: tuple[Any, ...],
) -> int | None:
    """Combine a narrow guard with a must-reaching zero-extended stack save.

    Match the reloaded slot to a supported bounded store path and compatible
    unsigned guard evidence. SP changes and slot writes are barriers to reusing
    its identity or high-bit contents. The narrow storage type alone does not
    authorize scanning every representable value as a table row.
    """

    if (
        len(index_expression) != 4
        or index_expression[0] != "load"
        or not isinstance(index_expression[2], int)
        or index_expression[2] <= 8
    ):
        return None
    endness, selector_bits, address_key = index_expression[1:]
    dispatcher_vex = _node_vex(node)
    if dispatcher_vex is None or _vex_writes_stack_pointer(dispatcher_vex):
        return None

    guard_nodes = [
        predecessor
        for predecessor in graph.predecessors(node)
        if _node_is_materialized_cfg_node(predecessor)
        and _node_intersects_bounds(predecessor, bounds)
    ]
    if not guard_nodes:
        return None

    guard_proofs: set[tuple[int, int]] = set()
    for guard_node in guard_nodes:
        guard_vex = _node_vex(guard_node)
        if guard_vex is None or _vex_writes_stack_pointer(guard_vex):
            return None
        guard_bounds = {
            (narrow_bits, upper_bound)
            for narrow_bits in (8, 16, 32)
            if narrow_bits < selector_bits
            if (
                upper_bound := _vex_guarded_stack_load_upper_bound(
                    guard_vex,
                    node.addr,
                    ("load", endness, narrow_bits, address_key),
                )
            )
            is not None
        }
        if len(guard_bounds) != 1:
            return None
        guard_proofs.update(guard_bounds)

    if len(guard_proofs) != 1:
        return None
    source_bits, upper_bound = next(iter(guard_proofs))
    if (
        _zero_extended_stack_selector_store_node(
            graph,
            bounds,
            node,
            guard_nodes,
            address_key,
            endness,
            source_bits,
            selector_bits,
        )
        is None
    ):
        return None
    entry_count = upper_bound + 1
    return entry_count if entry_count <= MAX_STATIC_JUMPTABLE_ENTRIES else None


def _vex_writes_stack_pointer(vex) -> bool:
    """Return whether a block changes its stack-pointer register."""

    sp_offset = getattr(getattr(vex, "arch", None), "sp_offset", None)
    return sp_offset is not None and any(
        isinstance(statement, pyvex.stmt.Put) and statement.offset == sp_offset
        for statement in vex.statements
    )


def _vex_last_stack_store_is_zero_extension(
    vex,
    address_key: tuple[Any, ...],
    endness: str,
    source_bits: int,
    destination_bits: int,
) -> bool:
    """Return whether the final write to one slot zero-extends its narrow value."""

    definitions = temporary_definitions(vex)
    stores = [
        statement
        for statement in vex.statements
        if isinstance(statement, pyvex.stmt.Store)
        and _vex_expr_key(statement.addr, definitions) == address_key
    ]
    if not stores:
        return False
    store = stores[-1]
    return (
        store.end == endness
        and store.data.result_size(vex.tyenv) == destination_bits
        and _vex_is_zero_extension_from(
            store.data, definitions, source_bits, destination_bits
        )
    )


def _zero_extended_stack_selector_store_node(
    graph: CFGGraph,
    bounds: FunctionBounds,
    dispatcher,
    guard_nodes: list[CFGNode],
    address_key: tuple[Any, ...],
    endness: str,
    source_bits: int,
    destination_bits: int,
):
    """Return a must-reaching zext slot store through a bounded predecessor walk."""

    pending = deque(
        predecessor
        for guard in guard_nodes
        for predecessor in graph.predecessors(guard)
    )
    seen = set()
    definitions = set()
    while pending:
        candidate = pending.popleft()
        if candidate in seen:
            continue
        seen.add(candidate)
        if len(seen) > MAX_CROSS_BLOCK_SELECTOR_PROOF_NODES:
            return None
        if not (
            _node_is_materialized_cfg_node(candidate)
            and _node_intersects_bounds(candidate, bounds)
        ):
            return None
        vex = _node_vex(candidate)
        if (
            vex is None
            or vex.jumpkind not in {"Ijk_Boring", "Ijk_Call"}
            # Calls return to the caller's frame. Ignore VEX's synthetic
            # return-address push while rejecting selector-slot writes below.
            or (vex.jumpkind != "Ijk_Call" and _vex_writes_stack_pointer(vex))
        ):
            return None
        if _vex_writes_stack_slot(vex, address_key):
            if not _vex_last_stack_store_is_zero_extension(
                vex, address_key, endness, source_bits, destination_bits
            ):
                return None
            definitions.add(candidate)
            continue
        if candidate.addr == bounds.addr:
            return None
        pending.extend(graph.predecessors(candidate))
    return next(iter(definitions)) if len(definitions) == 1 else None


def _vex_writes_stack_slot(vex, address_key: tuple[Any, ...]) -> bool:
    """Return whether one block stores to the selected stack-slot address."""

    definitions = temporary_definitions(vex)
    return any(
        isinstance(statement, pyvex.stmt.Store)
        and _vex_expr_key(statement.addr, definitions) == address_key
        for statement in vex.statements
    )


def _vex_guarded_stack_load_upper_bound(
    vex, target_addr: int, narrow_load_key: tuple[Any, ...]
) -> int | None:
    """Return a guard bound for one narrowed load of a stack selector slot."""

    definitions = temporary_definitions(vex)
    bounds_found: set[int] = set()
    for statement in vex.statements:
        if not isinstance(statement, pyvex.stmt.Exit):
            continue
        if getattr(statement.dst, "value", None) == target_addr:
            upper_bound = _vex_guard_stack_load_upper_bound(
                statement.guard,
                narrow_load_key,
                definitions,
                vex,
                index_on_left=True,
            )
            if upper_bound is not None:
                bounds_found.add(upper_bound)
        elif _vex_const_value(vex.next, definitions) == target_addr:
            upper_bound = _vex_guard_stack_load_upper_bound(
                statement.guard,
                narrow_load_key,
                definitions,
                vex,
                index_on_left=False,
            )
            if upper_bound is not None:
                bounds_found.add(upper_bound)
    return next(iter(bounds_found)) if len(bounds_found) == 1 else None


def _vex_guard_stack_load_upper_bound(
    guard,
    narrow_load_key: tuple[Any, ...],
    definitions: dict[int, Any],
    vex,
    *,
    index_on_left: bool,
) -> int | None:
    """Return an unsigned bound when a guard reads one narrowed stack slot."""

    guard = _vex_unsigned_guard_comparison(guard, definitions)
    if not isinstance(guard, pyvex.expr.Binop) or not guard.op.endswith("U"):
        return None
    index_expr, bound_expr = (
        (guard.args[0], guard.args[1])
        if index_on_left
        else (guard.args[1], guard.args[0])
    )
    index_expr, bound_expr = _vex_unsigned_shifted_compare_operands(
        index_expr, bound_expr, definitions, vex
    )
    if not _vex_guard_reads_stack_load(index_expr, narrow_load_key, definitions, vex):
        return None
    return _unsigned_comparison_upper_bound(
        guard.op, _vex_static_int(bound_expr, definitions), index_on_left
    )


def _vex_guard_reads_stack_load(
    expr,
    narrow_load_key: tuple[Any, ...],
    definitions: dict[int, Any],
    vex,
) -> bool:
    """Return whether one guard reads the specified narrow stack load."""

    expr = _resolve_vex_expr(expr, definitions)
    if _vex_expr_key(expr, definitions) == narrow_load_key:
        return True
    masked_source = _vex_low_masked_source(expr, definitions, vex.tyenv)
    if masked_source is not None:
        expr, bits = masked_source
    else:
        bits = narrow_load_key[2]
    if bits != narrow_load_key[2]:
        return False
    source = _vex_low_bits_source(expr, definitions, vex.tyenv, bits)
    return _vex_expr_key(source, definitions) == narrow_load_key
