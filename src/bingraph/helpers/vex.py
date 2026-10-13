"""Stateless VEX expressions and scalar-read primitives.

These helpers neither traverse CFGs nor establish selector bounds or targets.
Callers retain memory permissions, alias handling, ABI assumptions and budgets."""

from __future__ import annotations

from types import MappingProxyType
from typing import Any

import pyvex
from angr import Project

STATIC_INTEGER_BYTES = frozenset({1, 2, 4, 8})


X86_CONDITION_HELPERS = MappingProxyType(
    {
        "AMD64": "amd64g_calculate_condition",
        "X86": "x86g_calculate_condition",
    }
)


X86_CONDITION_HELPER_NAMES = frozenset(X86_CONDITION_HELPERS.values())


def temporary_definitions(vex: Any) -> dict[int, Any]:
    """Return local WrTmp expressions without resolving or evaluating them.

    This intentionally excludes LoadG destinations. Analyses requiring read
    positions or provenance must keep their richer maps, not substitute this
    context-free view for reaching-definition facts.
    """

    return {
        stmt.tmp: stmt.data
        for stmt in vex.statements
        if isinstance(stmt, pyvex.stmt.WrTmp)
    }


def is_call_like_jumpkind(jumpkind: object) -> bool:
    """Identify VEX calls/system transfers, not fake returns or internal labels."""

    return jumpkind == "Ijk_Call" or (
        isinstance(jumpkind, str) and jumpkind.startswith("Ijk_Sys_")
    )


def _resolve_vex_expr(expr, definitions: dict[int, Any]):
    """Follow local temporary aliases until reaching a non-temporary expression.

    Detect missing/cyclic chains rather than repeatedly chasing them. The
    result is an expression, not necessarily a constant or a reaching fact:
    callers needing read positions must use their richer statement-aware map.
    """

    seen: set[int] = set()
    while isinstance(expr, pyvex.expr.RdTmp):
        if expr.tmp in seen:
            return None
        seen.add(expr.tmp)
        expr = definitions.get(expr.tmp)
        if expr is None:
            return None
    return expr


def _vex_const_value(expr, definitions: dict[int, Any]) -> int | None:
    """Return a VEX constant's value after resolving local temporaries."""

    expr = _resolve_vex_expr(expr, definitions)
    if not isinstance(expr, pyvex.expr.Const):
        return None
    value = expr.con.value
    return value if isinstance(value, int) else None


def _vex_get_key(expr, definitions: dict[int, Any], vex) -> tuple[int, int] | None:
    """Return ``(register_offset, bits)`` for a VEX register read expression."""

    expr = _resolve_vex_expr(expr, definitions)
    if not isinstance(expr, pyvex.expr.Get):
        return None
    return expr.offset, expr.result_size(vex.tyenv)


def _vex_add_terms(expr, definitions: dict[int, Any]) -> list[Any] | None:
    """Flatten resolved integer additions while retaining every opaque operand.

    This lets a caller separately identify constant, base-register and selector
    terms. Do not evaluate memory, discard unknown operands or treat the sum
    as a target proof. Unsupported non-additive expressions remain terms for
    the caller's shape-specific policy to accept or reject.
    """

    expr = _resolve_vex_expr(expr, definitions)
    if expr is None:
        return None
    if isinstance(expr, pyvex.expr.Binop) and expr.op.startswith("Iop_Add"):
        left = _vex_add_terms(expr.args[0], definitions)
        right = _vex_add_terms(expr.args[1], definitions)
        if left is None or right is None:
            return None
        return [*left, *right]
    return [expr]


def _vex_width_conversion(
    expr,
) -> tuple[int, int, str | None] | None:
    """Describe one VEX integer-width conversion without accepting arithmetic."""

    if not isinstance(expr, pyvex.expr.Unop):
        return None
    conversion = expr.op.removeprefix("Iop_")
    source, separator, destination = conversion.partition("to")
    if not separator or not destination.isdecimal():
        return None
    signedness = source[-1:] if source[-1:] in {"S", "U"} else None
    source_bits = source[:-1] if signedness is not None else source
    if not source_bits.isdecimal():
        return None
    return int(source_bits), int(destination), signedness


def _vex_last_put(vex, offset: int):
    """Return the last local VEX register definition for one offset."""

    return next(
        (
            statement
            for statement in reversed(vex.statements)
            if isinstance(statement, pyvex.stmt.Put) and statement.offset == offset
        ),
        None,
    )


def read_static_integer(
    project: Project,
    slot_addr: int,
    entry_size: int,
    endness: str,
) -> int | None:
    """Read an unsigned 1/2/4/8-byte integer using VEX's endianness.

    Callers own address and memory-permission proofs. Unreadable memory is
    unknown, not zero; this helper neither validates targets nor signs entries.
    """

    if entry_size not in STATIC_INTEGER_BYTES:
        return None
    try:
        raw_target = project.loader.memory.load(slot_addr, entry_size)
    except Exception:
        return None

    byteorder = "little" if endness == "Iend_LE" else "big"
    return int.from_bytes(raw_target, byteorder=byteorder)
