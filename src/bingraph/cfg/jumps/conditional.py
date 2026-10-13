"""Conditional-PC dispatch and link-return recognition from focused VEX lifts.

Keep default continuations distinct from computed dispatch targets. Only the
existing supported forms are accepted; failed lifts or incomplete bounds leave
the unresolved transfer in place. The builder owns target materialization."""

from __future__ import annotations

from typing import Any

import pyvex
from angr import Project
from angr.knowledge_plugins.cfg import CFGNode

from bingraph.cfg.jumps.expressions import (
    _vex_direct_table_from_load,
    _vex_expr_key,
    is_direct_target_valid,
)
from bingraph.cfg.jumps.guards import _vex_guarded_index_values
from bingraph.cfg.jumps.tables import _jump_table_addr, _read_static_jump_table_targets
from bingraph.cfg.models import FunctionBounds, StaticJumpTable
from bingraph.helpers.vex import (
    _resolve_vex_expr,
    _vex_add_terms,
    _vex_const_value,
    _vex_get_key,
    temporary_definitions,
)


def _conditional_pc_dispatch_shape(vex, fallthrough_addr: int):
    """Return a VEX-proven conditional non-linear program-counter update.

    A conditional write to the program counter is represented as ``next =
    ITE(condition, taken, current_pc)`` plus an ``Exit`` for the ordinary
    continuation.  This is a VEX control-flow shape, not an instruction-set
    convention: ARM's predicated ``ldr pc`` and ``add pc`` are two examples.
    """

    if vex.jumpkind != "Ijk_Boring":
        return None

    definitions = temporary_definitions(vex)
    next_expr = _resolve_vex_expr(vex.next, definitions)
    if not isinstance(next_expr, pyvex.expr.ITE):
        return None

    if not any(
        isinstance(statement, pyvex.stmt.Exit)
        and getattr(statement.dst, "value", None) == fallthrough_addr
        for statement in vex.statements
    ):
        return None

    taken = _resolve_vex_expr(next_expr.iftrue, definitions)
    if taken is None and isinstance(next_expr.iftrue, pyvex.expr.RdTmp):
        # LoadG destinations are statements, not WrTmps, and therefore do
        # not appear in the normal temporary-definition mapping.
        taken = next_expr.iftrue
    if taken is None:
        return None
    return definitions, next_expr.cond, taken


def vex_is_conditional_link_return(vex, fallthrough_addr: int) -> bool:
    """Recognize a conditional branch to the unchanged ABI link register.

    SystemZ conditional returns have ``Ijk_Boring`` rather than ``Ijk_Ret``.
    Require a direct link-register read and no overlapping writes in this
    block: a reassigned link register may instead hold a computed jump target.
    """

    link_offset = getattr(vex.arch, "lr_offset", None)
    target = _resolve_vex_expr(vex.next, temporary_definitions(vex))
    if not isinstance(target, pyvex.expr.Get) or target.offset != link_offset:
        return False
    link_end = link_offset + vex.arch.bytes
    return any(
        isinstance(statement, pyvex.stmt.Exit)
        and statement.jumpkind == "Ijk_Boring"
        and getattr(statement.dst, "value", None) == fallthrough_addr
        for statement in vex.statements
    ) and not any(
        isinstance(statement, pyvex.stmt.Put)
        and statement.offset < link_end
        and link_offset < statement.offset + statement.data.result_size(vex.tyenv) // 8
        for statement in vex.statements
    )


def vex_has_computed_pc_transfer(vex, fallthrough_addr: int) -> bool:
    """Return whether VEX proves a computed program-counter transfer.

    A conditional transfer needs its explicit ordinary continuation to avoid
    mistaking VEX's internal predication machinery for a CFG boundary. An
    unconditional ``NEXT`` that remains non-constant after resolving its
    temporary definitions is a computed branch by definition. Decoders use
    this solely to choose a basic-block boundary; target recovery remains
    separate and requires a finite VEX-derived target domain.
    """

    if _conditional_pc_dispatch_shape(vex, fallthrough_addr) is not None:
        return True
    if vex.jumpkind != "Ijk_Boring":
        return False

    definitions = temporary_definitions(vex)
    next_expr = _resolve_vex_expr(vex.next, definitions)
    return next_expr is not None and not isinstance(
        next_expr, (pyvex.expr.Const, pyvex.expr.ITE)
    )


def _conditional_pc_load_table(
    vex,
    definitions: dict[int, Any],
    condition,
    taken,
) -> StaticJumpTable | None:
    """Describe a guarded absolute table load selected as the next PC."""

    if not isinstance(taken, pyvex.expr.RdTmp):
        return None

    condition_key = _vex_expr_key(condition, definitions)
    if condition_key is None:
        return None
    for statement in vex.statements:
        if not isinstance(statement, pyvex.stmt.LoadG) or statement.dst != taken.tmp:
            continue
        guard_key = _vex_expr_key(statement.guard, definitions)
        if guard_key != condition_key:
            continue
        entry_size = vex.tyenv.sizeof(statement.dst) // 8
        if statement.cvt != f"ILGop_Ident{entry_size * 8}":
            continue
        # The outer ITE and fallthrough Exit prove this absolute load updates
        # the PC. Its bound is recovered from the enclosing VEX condition.
        return _vex_direct_table_from_load(
            vex,
            definitions,
            statement.addr,
            entry_size,
            statement.end,
            allow_full_width_index=True,
            allow_inline_index_values=True,
            allow_masked_index_values=True,
            allow_static_base=True,
        )
    return None


def _conditional_pc_arithmetic_dispatch(
    vex,
    definitions: dict[int, Any],
    taken,
) -> tuple[tuple[int, int], int, int] | None:
    """Describe a ``constant + (register << shift)`` PC target expression."""

    if not isinstance(taken, pyvex.expr.Binop) or not taken.op.startswith("Iop_Add"):
        return None

    base_addr = None
    index_key = None
    shift = None
    for term in _vex_add_terms(taken, definitions) or ():
        value = _vex_const_value(term, definitions)
        if value is not None and base_addr is None:
            base_addr = value
            continue
        shifted = _resolve_vex_expr(term, definitions)
        if not isinstance(shifted, pyvex.expr.Binop) or not shifted.op.startswith(
            "Iop_Shl"
        ):
            return None
        candidate_shift = _vex_const_value(shifted.args[1], definitions)
        candidate_index = _vex_get_key(shifted.args[0], definitions, vex)
        if candidate_shift is None or candidate_index is None or index_key is not None:
            return None
        index_key = candidate_index
        shift = candidate_shift

    if base_addr is None or index_key is None or shift is None:
        return None
    return index_key, base_addr, shift


def conditional_pc_dispatch_targets(
    project: Project,
    bounds: FunctionBounds,
    node: CFGNode,
    *,
    table_data: set[int] | None = None,
) -> tuple[tuple[int, ...] | None, str | None]:
    """Return finite targets of one VEX-proven conditional PC dispatcher.

    It supports either an absolute-address table load or an inline arithmetic
    dispatch. A same-block unsigned guard must bound the index; predecessor
    paths are not symbolically replayed. It never infers targets from arbitrary
    code addresses, so callers can safely feed its output into a leader
    worklist.
    """

    try:
        # CFGNode.block may use angr's cross-instruction optimization, which
        # substitutes an index register with an earlier temporary. Re-lift the
        # already bounded block without that optimization so a guard and its
        # indexed PC update retain their shared register expression.
        vex = project.factory.block(
            node.addr,
            size=node.size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception:
        return None, "no_vex"

    shape = _conditional_pc_dispatch_shape(vex, node.addr + node.size)
    if shape is None:
        return None, "not_conditional_pc"
    definitions, condition, taken = shape

    table = _conditional_pc_load_table(vex, definitions, condition, taken)
    if table is not None:
        if table.index_register_offset is None or table.index_bits is None:
            return None, "unbounded_index"
        index_values = _vex_guarded_index_values(
            condition,
            (table.index_register_offset, table.index_bits),
            definitions,
            vex,
        )
        if index_values is None or table.static_base_addr is None:
            return None, "unbounded_index"
        targets = _read_static_jump_table_targets(
            project,
            table,
            table.static_base_addr,
            index_values,
        )
        if targets is None:
            return None, "table_unreadable"
        if not all(is_direct_target_valid(bounds, target) for target in targets):
            return None, "invalid_target"
        if table_data is not None:
            address = _jump_table_addr(table.static_base_addr, table)
            for index in index_values:
                table_data.update(
                    range(
                        address + index * table.entry_size,
                        address + (index + 1) * table.entry_size,
                    )
                )
        return targets, None

    arithmetic = _conditional_pc_arithmetic_dispatch(vex, definitions, taken)
    if arithmetic is None:
        return None, "no_dispatch_shape"
    index_key, base_addr, shift = arithmetic
    index_values = _vex_guarded_index_values(condition, index_key, definitions, vex)
    if index_values is None:
        return None, "unbounded_index"
    mask = (1 << project.arch.bits) - 1
    targets = tuple(
        sorted({(base_addr + (index << shift)) & mask for index in index_values})
    )
    if not all(is_direct_target_valid(bounds, target) for target in targets):
        return None, "invalid_target"
    return targets, None
