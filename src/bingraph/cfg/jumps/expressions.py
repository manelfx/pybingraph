"""Local dispatch shapes, selector expressions and executable-target checks.

Recognition describes lifted expressions, not complete reachable target sets.
Table planners must separately prove bounds and bases before reading entries.
These primitives do not import architecture planners or builder state."""

from __future__ import annotations

from typing import Any

import pyvex
from angr import Project
from angr.knowledge_plugins.cfg import CFGNode

from bingraph.cfg.graph import node_vex as _node_vex
from bingraph.cfg.models import FunctionBounds, StaticJumpTable
from bingraph.helpers.vex import (
    STATIC_INTEGER_BYTES,
    _resolve_vex_expr,
    _vex_add_terms,
    _vex_const_value,
    _vex_get_key,
    _vex_width_conversion,
    temporary_definitions,
)

# Shared finite-domain ceiling, not evidence for an unguarded table's extent.
MAX_STATIC_JUMPTABLE_ENTRIES = 256


def is_direct_target_valid(bounds: FunctionBounds, target: int | None) -> bool:
    """Check a direct target against the supplied closed-open function range.

    This only answers an address-range question. It does not validate executable
    memory, instruction boundaries or external destinations; those have
    different policies in planning, decoding and final materialization.
    """

    return target is not None and bounds.addr <= target < bounds.end_addr


def static_jump_target_rejection_reason(project: Project, target: int) -> str | None:
    """Return why one static-table entry is unsafe to materialize, if any.

    A table shape and finite index prove that entries are consulted, but do not
    prove arbitrary values read from the table are executable CFG destinations.
    Executable targets outside the function remain valid external leaves; data
    and CLE's synthetic extern-address space remain unresolved.
    """

    obj = project.loader.find_object_containing(target)
    if obj is None:
        return "unmapped"
    if obj is getattr(project.loader, "extern_object", None):
        return "synthetic"

    find_section = getattr(obj, "find_section_containing", None)
    section = find_section(target) if callable(find_section) else None
    if section is not None:
        return None if getattr(section, "is_executable", False) else "non_executable"

    find_segment = getattr(obj, "find_segment_containing", None)
    segment = find_segment(target) if callable(find_segment) else None
    if segment is not None:
        return None if getattr(segment, "is_executable", False) else "non_executable"

    return "non_executable"


def is_memory_dependent_indirect_jump(node: CFGNode) -> bool:
    """Detect a memory read in the locally lifted computed target expression.

    This is a recovery safeguard, not a target proof. After exact resolution
    fails, neither a vtable load nor an unproved static-table load justifies
    edges to arbitrary disconnected code. Follow VEX temporaries and target
    arithmetic, but ignore loads unrelated to NEXT and do not trace other
    blocks. Decoding may classify an s390 table-loaded branch as Ijk_Ret;
    the caller must already have identified the machine transfer as a jump.
    """

    vex = _node_vex(node)
    if vex is None or vex.jumpkind not in {"Ijk_Boring", "Ijk_Ret"}:
        return False
    key = _vex_expr_key(vex.next, temporary_definitions(vex))
    return key is not None and _vex_key_reads_memory(key)


def _vex_register_with_displacement(
    expr, definitions: dict[int, Any], vex
) -> tuple[tuple[int, int], int] | None:
    """Recognize one base register plus additive constants after local resolution.

    Flatten additions and accumulate constants, but require exactly one
    register term. Return its offset/width key and displacement without reading
    memory or following predecessors. Two registers or an unsupported term
    make the address ambiguous and return ``None``.
    """

    register_key = _vex_get_key(expr, definitions, vex)
    if register_key is not None:
        return register_key, 0

    terms = _vex_add_terms(expr, definitions)
    if terms is None:
        return None

    displacement = 0
    register_key = None
    for term in terms:
        value = _vex_const_value(term, definitions)
        if value is not None:
            displacement += value
            continue
        key = _vex_get_key(term, definitions, vex)
        if key is None or register_key is not None:
            return None
        register_key = key
    return (register_key, displacement) if register_key is not None else None


def _vex_index_key(
    expr,
    definitions: dict[int, Any],
    vex,
    *,
    allow_full_width: bool = False,
) -> tuple[int, int] | None:
    """Identify a selector's underlying register, not its complete value domain.

    Follow temporaries and the supported conversion/mask wrappers so a narrowed
    guard can be related to the address register. Wider sources are opt-in;
    this recognition does not prove unknown high bits zero. A separate guard
    or finite-domain proof is required before table rows can be read.
    """

    expr = _resolve_vex_expr(expr, definitions)
    # A range guard often compares a narrowed view of the register used to
    # address the table (for example, x86 ``cmp r8d, limit`` before indexing
    # with ``r8``). Preserve the source register across integer-width casts.
    while (
        isinstance(expr, pyvex.expr.Unop)
        and expr.op.startswith("Iop_")
        and "to" in expr.op
    ):
        expr = _resolve_vex_expr(expr.args[0], definitions)
    if isinstance(expr, pyvex.expr.Binop) and expr.op.startswith("Iop_And"):
        left = _vex_index_key(expr.args[0], definitions, vex)
        right = _vex_index_key(expr.args[1], definitions, vex)
        return left or right
    key = _vex_get_key(expr, definitions, vex)
    if key is not None and key[1] > 0 and (allow_full_width or key[1] <= 8):
        return key
    return None


def _vex_finite_index_values(
    expr, definitions: dict[int, Any]
) -> tuple[int, ...] | None:
    """Return a small exact integer domain represented by one VEX expression.

    This accepts only a constant or an ITE whose true arm is bounded by its
    unsigned comparison and whose false arm is independently finite. It is a
    table-index proof, not a general VEX evaluator.
    """

    expr = _resolve_vex_expr(expr, definitions)
    value = _vex_const_value(expr, definitions)
    if value is not None:
        return (value,)
    if not isinstance(expr, pyvex.expr.ITE):
        return None

    condition = _resolve_vex_expr(expr.cond, definitions)
    while isinstance(condition, pyvex.expr.Unop):
        condition = _resolve_vex_expr(condition.args[0], definitions)
    if not isinstance(condition, pyvex.expr.Binop) or not condition.op.endswith("U"):
        return None
    bound = _vex_static_int(condition.args[1], definitions)
    if bound is None:
        return None
    if "CmpLT" in condition.op:
        upper_bound = bound - 1
    elif "CmpLE" in condition.op:
        upper_bound = bound
    else:
        return None
    if upper_bound < 0 or upper_bound >= MAX_STATIC_JUMPTABLE_ENTRIES:
        return None

    guard_value = _resolve_vex_expr(condition.args[0], definitions)
    true_value = _resolve_vex_expr(expr.iftrue, definitions)
    if guard_value is None or true_value is None:
        return None
    if guard_value is not true_value and guard_value != true_value:
        return None
    false_values = _vex_finite_index_values(expr.iffalse, definitions)
    if false_values is None:
        return None

    values = tuple(sorted({*range(upper_bound + 1), *false_values}))
    if not values or len(values) > MAX_STATIC_JUMPTABLE_ENTRIES:
        return None
    if values[0] < 0 or values[-1] >= MAX_STATIC_JUMPTABLE_ENTRIES:
        return None
    return values


def _vex_masked_index_values(
    expr,
    definitions: dict[int, Any],
) -> tuple[int, ...] | None:
    """Return every value permitted by a constant VEX bit-mask index.

    ``index & mask`` has a finite, architecture-neutral domain even when the
    original register is unconstrained. Enumerating the mask's submasks keeps
    non-contiguous masks exact rather than treating their numerical maximum as
    a range. The shared table limit prevents this local proof from becoming an
    unbounded symbolic evaluator.
    """

    expr = _resolve_vex_expr(expr, definitions)
    while (
        isinstance(expr, pyvex.expr.Unop)
        and expr.op.startswith("Iop_")
        and "to" in expr.op
    ):
        expr = _resolve_vex_expr(expr.args[0], definitions)
    if not isinstance(expr, pyvex.expr.Binop) or not expr.op.startswith("Iop_And"):
        return None

    left, right = expr.args
    left_value = _vex_const_value(left, definitions)
    right_value = _vex_const_value(right, definitions)
    if (left_value is None) == (right_value is None):
        return None
    mask, index_expr = (
        (left_value, right) if left_value is not None else (right_value, left)
    )
    if mask is None:
        return None
    if _resolve_vex_expr(index_expr, definitions) is None:
        return None

    possible_values = 1 << mask.bit_count()
    if possible_values > MAX_STATIC_JUMPTABLE_ENTRIES:
        return None

    values: list[int] = []
    value = mask
    while True:
        values.append(value)
        if value == 0:
            break
        value = (value - 1) & mask
    values.sort()
    return tuple(values)


def _vex_table_index(
    expr,
    definitions: dict[int, Any],
    vex,
    *,
    allow_full_width: bool,
    allow_inline_index_values: bool,
    allow_masked_index_values: bool,
) -> tuple[tuple[int, int] | None, tuple[int, ...] | None]:
    """Select an enabled selector representation without combining incompatible ones.

    Prefer an exact mask domain when enabled, then a supported register key,
    then an opt-in finite conditional expression. Return ``(register, values)``
    with at most one populated representation; callers must supply any guard
    needed by a register and retain relative/direct shape-specific policies.
    """

    if allow_masked_index_values:
        values = _vex_masked_index_values(
            expr,
            definitions,
        )
        if values is not None:
            return None, values
    register = _vex_index_key(expr, definitions, vex, allow_full_width=allow_full_width)
    if register is not None:
        return register, None
    if not allow_inline_index_values:
        return None, None
    return None, _vex_finite_index_values(expr, definitions)


def _vex_scaled_table_index(
    expr, definitions: dict[int, Any]
) -> tuple[Any, int, tuple[int, ...] | None, int | None] | None:
    """Normalize a VEX-scaled table selector to its source and shift.

    Besides an ordinary left shift, s390x lifts ``risbg``/``risbgn`` as a
    rotate followed by a mask. Accept only the form whose mask removes the
    rotated-in low bits and retains a contiguous low selector domain. That is
    exactly a masked left shift, not a general rotate-based computation.
    """

    expr = _resolve_vex_expr(expr, definitions)
    if isinstance(expr, pyvex.expr.Binop) and expr.op.startswith("Iop_Shl"):
        shift = _vex_const_value(expr.args[1], definitions)
        return (expr.args[0], shift, None, None) if shift is not None else None

    if not isinstance(expr, pyvex.expr.Binop) or not expr.op.startswith("Iop_And"):
        return None
    left, right = expr.args
    left_value = _vex_const_value(left, definitions)
    right_value = _vex_const_value(right, definitions)
    if (left_value is None) == (right_value is None):
        return None
    mask, rotated = (
        (left_value, right) if left_value is not None else (right_value, left)
    )
    rotated = _resolve_vex_expr(rotated, definitions)
    if (
        mask is None
        or not isinstance(rotated, pyvex.expr.Binop)
        or not rotated.op.startswith("Iop_Or")
    ):
        return None

    shifts = tuple(_resolve_vex_expr(term, definitions) for term in rotated.args)
    left_shift = next(
        (
            term
            for term in shifts
            if isinstance(term, pyvex.expr.Binop) and term.op.startswith("Iop_Shl")
        ),
        None,
    )
    right_shift = next(
        (
            term
            for term in shifts
            if isinstance(term, pyvex.expr.Binop) and term.op.startswith("Iop_Shr")
        ),
        None,
    )
    if left_shift is None or right_shift is None:
        return None
    shift = _vex_const_value(left_shift.args[1], definitions)
    inverse_shift = _vex_const_value(right_shift.args[1], definitions)
    source = _resolve_vex_expr(left_shift.args[0], definitions)
    inverse_source = _resolve_vex_expr(right_shift.args[0], definitions)
    width = source.result_size(None) if source is not None else 0
    if (
        shift is None
        or inverse_shift is None
        or source is None
        or inverse_source is None
        or source != inverse_source
        or shift <= 0
        or shift >= width
        or inverse_shift != width - shift
    ):
        return None

    if mask & ((1 << shift) - 1):
        return None
    index_mask = mask >> shift
    if index_mask <= 0 or index_mask & (index_mask + 1):
        return None
    if mask != index_mask << shift:
        return None
    values = (
        tuple(range(index_mask + 1))
        if index_mask + 1 <= MAX_STATIC_JUMPTABLE_ENTRIES
        else None
    )
    return source, shift, values, index_mask.bit_length()


def _vex_low_register_view_keys(
    register_key: tuple[int, int], minimum_bits: int, vex
) -> tuple[tuple[int, int], ...]:
    """Enumerate supported low register views from archinfo's offset/size metadata.

    Respect register endianness instead of matching names such as AL/EAX/RAX.
    These views are candidates for width-compatible guard matching; finding
    an alias does not make the unconsumed high bits known.
    """

    if minimum_bits <= 0 or minimum_bits > register_key[1]:
        return ()
    arch = getattr(vex, "arch", None)
    registers = getattr(arch, "registers", {})
    parent_offset, parent_bits = register_key
    parent_bytes = parent_bits // 8
    candidates: set[tuple[int, int]] = set()
    for offset, size in registers.values():
        view_bits = size * 8
        if not minimum_bits <= view_bits < parent_bits:
            continue
        view_offset = (
            parent_offset + parent_bytes - size
            if getattr(arch, "register_endness", None) == "Iend_BE"
            else parent_offset
        )
        if offset == view_offset:
            candidates.add((offset, view_bits))
    return tuple(sorted(candidates))


def _vex_static_int(expr, definitions: dict[int, Any]) -> int | None:
    """Fold the supported constant guard expressions with machine-width wrapping.

    Resolve local temporaries and a restricted set of integer operations,
    declining any nonconstant operand. This deliberately cannot chase memory
    or reaching register definitions to rescue an otherwise unrelated guard.
    """

    value = _vex_const_value(expr, definitions)
    if value is not None:
        return value

    expr = _resolve_vex_expr(expr, definitions)
    if (
        isinstance(expr, pyvex.expr.Unop)
        and expr.op.startswith("Iop_")
        and "to" in expr.op
    ):
        value = _vex_static_int(expr.args[0], definitions)
        result_bits = expr.op.rsplit("to", maxsplit=1)[-1]
        if value is None or not result_bits.isdecimal():
            return None
        return value & ((1 << int(result_bits)) - 1)
    if not isinstance(expr, pyvex.expr.Binop):
        return None
    left = _vex_static_int(expr.args[0], definitions)
    right = _vex_static_int(expr.args[1], definitions)
    if left is None or right is None:
        return None
    if expr.op.startswith("Iop_And"):
        return left & right
    return None


def _vex_low_bits_source(expr, definitions: dict[int, Any], tyenv, bits: int):
    """Strip supported width wrappers while the queried low bits remain unchanged.

    Decline when a conversion or source width cannot preserve the requested
    slice. This aids structural guard matching, not a claim that truncation
    preserves the numeric value of an unknown wider register.
    """

    while True:
        expr = _resolve_vex_expr(expr, definitions)
        conversion = _vex_width_conversion(expr)
        if conversion is None:
            return expr if expr.result_size(tyenv) >= bits else None
        source_bits, destination_bits, _ = conversion
        if destination_bits < bits or source_bits < bits:
            return None
        expr = expr.args[0]


def _vex_low_masked_source(expr, definitions: dict[int, Any], tyenv):
    """Recognize a contiguous low-bit mask and report its source and bit width.

    Require the mask's exact all-ones form after local constant resolution.
    Sparse masks have different domains and must use the submask enumerator;
    neither form proves that discarded high bits were originally zero.
    """

    expr = _resolve_vex_expr(expr, definitions)
    if not isinstance(expr, pyvex.expr.Binop) or not expr.op.startswith("Iop_And"):
        return None
    left, right = expr.args
    left_value = _vex_const_value(left, definitions)
    right_value = _vex_const_value(right, definitions)
    mask, source = (
        (left_value, right) if left_value is not None else (right_value, left)
    )
    if mask is None or mask <= 0 or mask & (mask + 1):
        return None
    bits = mask.bit_length()
    source = _vex_low_bits_source(source, definitions, tyenv, bits)
    return (source, bits) if source is not None else None


def _vex_guard_width_view_key(
    expr, definitions: dict[int, Any], tyenv, bits: int
) -> tuple[Any, ...] | None:
    """Key the supported guard view after removing redundant width wrappers.

    Match zero-extension followed by compatible narrowing without losing the
    source's width evidence. The key is for local expression identity; it does
    not replace reaching-definition checks across blocks or intervening PUTs.
    """

    while True:
        expr = _resolve_vex_expr(expr, definitions)
        conversion = _vex_width_conversion(expr)
        if conversion is None:
            return _vex_expr_key(expr, definitions)
        source_bits, destination_bits, signedness = conversion
        if destination_bits == bits and source_bits > bits:
            # A guard observes only this low-width view of a wider value.
            expr = expr.args[0]
            continue
        if source_bits == bits and destination_bits > bits and signedness == "U":
            expr = expr.args[0]
            continue
        return _vex_expr_key(expr, definitions)


def _vex_is_zero_extension_from(
    expr, definitions: dict[int, Any], bits: int, register_bits: int
) -> bool:
    """Require an explicit conversion from the specified narrow to full width.

    A correctly sized register read or storage type alone is insufficient.
    This supplies high-bit evidence for a separate selector proof, whose
    reaching paths and finite bounds must still be established by the caller.
    """

    expr = _resolve_vex_expr(expr, definitions)
    current_bits = register_bits
    while True:
        conversion = _vex_width_conversion(expr)
        if conversion is None:
            return False
        source_bits, destination_bits, signedness = conversion
        if destination_bits != current_bits or signedness != "U":
            return False
        if source_bits == bits:
            return True
        if source_bits < bits:
            return False
        current_bits = source_bits
        expr = _resolve_vex_expr(expr.args[0], definitions)


def _vex_normalized_table_entry_load(
    expr, definitions: dict[int, Any]
) -> tuple[pyvex.expr.Load, bool] | None:
    """Return a table load and effective signedness through width casts.

    Lifters can express a signed 32-bit table entry as a zero extension, a
    truncation, and a final sign extension. Normalize only conversion chains
    that return to the original load width before one final direct extension.
    This accepts representation-only casts without mistaking arithmetic for a
    jump-table entry.
    """

    casts: list[tuple[int, int, str | None]] = []
    while True:
        expr = _resolve_vex_expr(expr, definitions)
        if isinstance(expr, pyvex.expr.Load):
            break
        if not isinstance(expr, pyvex.expr.Unop):
            return None

        conversion = _vex_width_conversion(expr)
        if conversion is None:
            return None
        casts.append(conversion)
        expr = expr.args[0]

    entry_bits = expr.result_size(None)
    casts.reverse()
    current_bits = entry_bits
    for source_bits, destination_bits, _ in casts:
        if source_bits != current_bits:
            return None
        current_bits = destination_bits

    # An extension followed by a truncation back to the load width preserves
    # the original entry bits. Discard these detours before deciding whether
    # the final value is signed or unsigned.
    normalized: list[tuple[int, int, str | None]] = []
    cursor = 0
    while cursor < len(casts):
        start = cursor
        current_bits = entry_bits
        while cursor < len(casts):
            _, current_bits, _ = casts[cursor]
            cursor += 1
            if current_bits == entry_bits:
                break
        if current_bits == entry_bits:
            continue
        normalized.extend(casts[start:])
        break

    if not normalized:
        return expr, False
    if len(normalized) != 1:
        return None
    source_bits, destination_bits, signedness = normalized[0]
    if source_bits != entry_bits or destination_bits <= entry_bits:
        return None
    return expr, signedness == "S"


def _vex_scaled_table_selector(
    term,
    definitions: dict[int, Any],
    vex,
    entry_size: int,
    *,
    allow_full_width_index: bool,
    allow_inline_index_values: bool,
    allow_masked_index_values: bool,
    allow_guarded_expression_index: bool,
    relative: bool = False,
) -> dict[str, Any] | None:
    """Describe one scaled selector using the existing table-model fields.

    This is shape recognition, not a complete target proof: register and
    expression selectors still need their caller's bounds. ``relative`` keeps
    that recognizer's existing masked-register fallback and UJT requirement;
    neither policy is inherited by direct tables. Base recognition, selector
    uniqueness, entry signedness and target arithmetic remain with callers.
    """

    scaled = _vex_scaled_table_index(term, definitions)
    if scaled is None:
        return None
    source, shift, scaled_values, low_bits = scaled
    key, values = _vex_table_index(
        source,
        definitions,
        vex,
        allow_full_width=allow_full_width_index,
        allow_inline_index_values=allow_inline_index_values,
        allow_masked_index_values=allow_masked_index_values,
    )
    if relative and key is None and low_bits is not None:
        # Relative rotate-mask tables retain the full register so predecessor
        # guards on narrower aliases can constrain the low-bit domain.
        key = _vex_index_key(source, definitions, vex, allow_full_width=True)
    if values is None:
        values = scaled_values
    expression = None
    if key is None and values is None and allow_guarded_expression_index:
        expression = _vex_expr_key(source, definitions)
        if expression is None or not _vex_key_reads_memory(expression):
            expression = None
    if (
        shift is None
        or (key is None and values is None and expression is None)
        or 1 << shift != entry_size
    ):
        return None
    return {
        "index_register_offset": key[0] if key is not None else None,
        "index_bits": key[1] if key is not None else None,
        "index_values": values,
        "index_low_bits": low_bits,
        "index_expression": expression,
        # An oversized rotate-mask domain still needs an unresolved fallback,
        # even when another representation of its selector was recognized.
        "preserve_unresolved_fallback": relative
        and low_bits is not None
        and scaled_values is None,
    }


def _vex_relative_jump_table(
    vex,
    *,
    allow_full_width_index: bool = False,
    allow_inline_index_values: bool = False,
    allow_masked_index_values: bool = False,
    allow_guarded_expression_index: bool = False,
) -> StaticJumpTable | None:
    """
    Describe a bounded relative jump table encoded in one VEX indirect jump.

    The accepted form is intentionally narrow: ``next`` must add a register or
    constant base to a loaded (optionally sign-extended) entry, while the load
    address must be that same base plus an index scaled by the entry size. The
    optional expression form accepts a memory-reading selector only when a
    predecessor separately proves its finite range.

    Share selector parsing with direct tables, but retain this recognizer's
    same-base requirement, signed-entry normalization and masked fallback.
    """

    if vex.jumpkind != "Ijk_Boring":
        return None

    definitions = temporary_definitions(vex)
    next_expr = _resolve_vex_expr(vex.next, definitions)
    if not isinstance(next_expr, pyvex.expr.Binop) or not next_expr.op.startswith(
        "Iop_Add"
    ):
        return None

    left, right = (
        _resolve_vex_expr(next_expr.args[0], definitions),
        _resolve_vex_expr(next_expr.args[1], definitions),
    )
    candidates = ((left, right), (right, left))
    for entry_expr, base_expr in candidates:
        base = _vex_register_with_displacement(base_expr, definitions, vex)
        static_base_addr = None
        if base is None:
            static_base_addr = _vex_const_value(base_expr, definitions)
            if static_base_addr is None:
                continue
            base_key = None
            target_displacement = 0
            base_bits = base_expr.result_size(vex.tyenv)
        else:
            base_key, target_displacement = base
            base_bits = base_key[1]

        if base_bits <= 0:
            continue

        entry_expr = _resolve_vex_expr(entry_expr, definitions)
        normalized_entry = _vex_normalized_table_entry_load(entry_expr, definitions)
        if normalized_entry is None:
            continue
        entry_expr, signed_entries = normalized_entry

        entry_size = entry_expr.result_size(vex.tyenv) // 8
        if entry_size not in STATIC_INTEGER_BYTES:
            continue

        address_terms = _vex_add_terms(entry_expr.addr, definitions)
        if address_terms is None:
            continue

        constant_total = 0
        displacement = 0
        saw_base = False
        selector = None
        for term in address_terms:
            value = _vex_const_value(term, definitions)
            if value is not None:
                constant_total += value
                continue
            if (
                base_key is not None
                and _vex_get_key(term, definitions, vex) == base_key
            ):
                saw_base = True
                continue
            candidate = _vex_scaled_table_selector(
                term,
                definitions,
                vex,
                entry_size,
                allow_full_width_index=allow_full_width_index,
                allow_inline_index_values=allow_inline_index_values,
                allow_masked_index_values=allow_masked_index_values,
                allow_guarded_expression_index=allow_guarded_expression_index,
                relative=True,
            )
            # Two selector terms are ambiguous even if each is recognizable.
            if candidate is None or selector is not None:
                break
            selector = candidate
        else:
            if static_base_addr is not None:
                mask = (1 << base_bits) - 1
                displacement = (constant_total - static_base_addr) & mask
                saw_base = True
            else:
                displacement = constant_total
            if saw_base and selector is not None:
                offset = base_key[0] if base_key is not None else None
                table_displacement = displacement & ((1 << base_bits) - 1)
                return StaticJumpTable(
                    base_register_offset=offset,
                    base_bits=base_bits,
                    table_displacement=table_displacement,
                    entry_size=entry_size,
                    endness=entry_expr.end,
                    signed_entries=signed_entries,
                    target_displacement=target_displacement,
                    static_base_addr=static_base_addr,
                    **selector,
                )

    return None


def _vex_direct_jump_table(
    vex,
    *,
    allow_full_width_index: bool = False,
    allow_inline_index_values: bool = False,
    allow_masked_index_values: bool = False,
    allow_static_base: bool = False,
    allow_guarded_expression_index: bool = False,
) -> StaticJumpTable | None:
    """
    Describe a bounded table whose entries are absolute jump destinations.

    The accepted VEX form is ``next = Load(base + index * entry_size + disp)``.
    Unlike relative tables, the loaded entry is itself the target address. The
    selector can be a register, an opt-in finite domain, or a supported
    memory-reading expression. Recognizing the shape is not enough to read
    entries: the table planner must separately establish its finite domain.
    """

    # VEX classifies s390 ``br`` through a register as a return even when
    # that register holds an absolute jump-table entry.
    if vex.jumpkind != "Ijk_Boring" and not (
        vex.arch.name == "S390X" and vex.jumpkind == "Ijk_Ret"
    ):
        return None

    definitions = temporary_definitions(vex)
    entry_expr = _resolve_vex_expr(vex.next, definitions)
    if isinstance(entry_expr, pyvex.expr.Load):
        return _vex_direct_table_from_load(
            vex,
            definitions,
            entry_expr.addr,
            entry_expr.result_size(vex.tyenv) // 8,
            entry_expr.end,
            allow_full_width_index=allow_full_width_index,
            allow_inline_index_values=allow_inline_index_values,
            allow_masked_index_values=allow_masked_index_values,
            allow_static_base=allow_static_base,
            allow_guarded_expression_index=allow_guarded_expression_index,
        )

    # Predicated LoadG program-counter writes are handled by the dedicated
    # conditional-PC resolver, which also validates the ordinary continuation.
    return None


def s390_table_loaded_branch(node: CFGNode) -> bool:
    """Distinguish a table-loaded s390 ``br`` from a saved-address return.

    VEX labels both as ``Ijk_Ret``. This check only identifies the static
    table shape; the normal table planner must still prove a finite selector.
    """

    vex = _node_vex(node)
    return bool(
        vex is not None
        and vex.arch.name == "S390X"
        and vex.jumpkind == "Ijk_Ret"
        and _vex_direct_jump_table(
            vex,
            allow_full_width_index=True,
            allow_inline_index_values=True,
            allow_masked_index_values=True,
            allow_static_base=True,
            allow_guarded_expression_index=True,
        )
        is not None
    )


def _vex_direct_table_from_load(
    vex,
    definitions: dict[int, Any],
    address_expr,
    entry_size: int,
    endness: str,
    *,
    allow_full_width_index: bool,
    allow_inline_index_values: bool,
    allow_masked_index_values: bool,
    allow_static_base: bool = False,
    allow_guarded_expression_index: bool = False,
) -> StaticJumpTable | None:
    """Describe an absolute-address table load from its VEX address expression.

    Parse constants and the first bare register as the address base, then
    delegate scaled selector terms to the shared shape helper. Unlike the
    relative recognizer, do not add masked-register or UJT-preservation rules.
    Static bases remain opt-in, and multiple selectors reject the whole shape.
    """

    if entry_size not in STATIC_INTEGER_BYTES:
        return None
    address_terms = _vex_add_terms(address_expr, definitions)
    if address_terms is None:
        return None

    base_bits = address_expr.result_size(vex.tyenv)
    displacement = 0
    base_key = None
    selector = None
    for term in address_terms:
        value = _vex_const_value(term, definitions)
        if value is not None:
            displacement += value
            continue

        register_key = _vex_get_key(term, definitions, vex)
        if register_key is not None and base_key is None:
            base_key = register_key
            continue

        candidate = _vex_scaled_table_selector(
            term,
            definitions,
            vex,
            entry_size,
            allow_full_width_index=allow_full_width_index,
            allow_inline_index_values=allow_inline_index_values,
            allow_masked_index_values=allow_masked_index_values,
            allow_guarded_expression_index=allow_guarded_expression_index,
        )
        if candidate is None or selector is not None:
            return None
        selector = candidate

    if selector is None:
        return None
    if base_key is None and not allow_static_base:
        # Absolute bases are opt-in. Conditional-PC recognition enables them
        # only after identifying the guarded load and its ordinary continuation.
        return None

    mask = (1 << base_bits) - 1
    static_base_addr = None
    table_displacement = displacement & mask
    if base_key is None:
        # A VEX constant in the effective address is already the table base.
        static_base_addr = table_displacement
        table_displacement = 0
    else:
        base_bits = base_key[1]
        table_displacement = displacement & ((1 << base_bits) - 1)

    return StaticJumpTable(
        base_register_offset=base_key[0] if base_key is not None else None,
        base_bits=base_bits,
        table_displacement=table_displacement,
        entry_size=entry_size,
        endness=endness,
        signed_entries=False,
        entries_are_relative=False,
        static_base_addr=static_base_addr,
        **selector,
    )


def _vex_key_reads_memory(key: tuple[Any, ...]) -> bool:
    """Detect a load anywhere in a supported structural expression key.

    Recursively inspect the keyed operations, not every instruction in a block.
    Consumers use this to distinguish memory-derived NEXT values from pure
    register arithmetic; a load's presence is not proof of a static table.
    """

    tag = key[0]
    if tag == "load":
        return True
    if tag in {"const", "get"}:
        return False
    if tag == "unop":
        return _vex_key_reads_memory(key[2])
    if tag in {"binop", "ccall"}:
        return any(_vex_key_reads_memory(argument) for argument in key[2:])
    if tag == "ite":
        return any(_vex_key_reads_memory(argument) for argument in key[1:])
    return False


def _vex_expr_key(expr, definitions: dict[int, Any]) -> tuple[Any, ...] | None:
    """Return a structural key for a local VEX expression.

    VEX temporary numbers are local to one lifted block, so matching branch
    predicates across adjacent blocks requires recursively replacing them with
    their definitions. Unsupported expressions deliberately return ``None``:
    callers use this only as a proof, never as a best-effort guess.
    """

    expr = _resolve_vex_expr(expr, definitions)
    if expr is None:
        return None
    if isinstance(expr, pyvex.expr.Const):
        value = expr.con.value
        return ("const", value) if isinstance(value, int) else None
    if isinstance(expr, pyvex.expr.Get):
        return ("get", expr.offset, expr.result_size(None))
    if isinstance(expr, pyvex.expr.Load):
        address = _vex_expr_key(expr.addr, definitions)
        return (
            (
                "load",
                expr.end,
                expr.result_size(None),
                address,
            )
            if address is not None
            else None
        )
    if isinstance(expr, pyvex.expr.Unop):
        argument = _vex_expr_key(expr.args[0], definitions)
        return ("unop", expr.op, argument) if argument is not None else None
    if isinstance(expr, pyvex.expr.Binop):
        arguments = tuple(_vex_expr_key(arg, definitions) for arg in expr.args)
        return ("binop", expr.op, *arguments) if all(arguments) else None
    if isinstance(expr, pyvex.expr.ITE):
        condition = _vex_expr_key(expr.cond, definitions)
        if_true = _vex_expr_key(expr.iftrue, definitions)
        if_false = _vex_expr_key(expr.iffalse, definitions)
        if condition is None or if_true is None or if_false is None:
            return None
        return ("ite", condition, if_true, if_false)
    if isinstance(expr, pyvex.expr.CCall):
        arguments = tuple(_vex_expr_key(arg, definitions) for arg in expr.args)
        if not all(arguments):
            return None
        return ("ccall", expr.cee.name, *arguments)
    return None
