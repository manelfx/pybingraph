"""Jump-target primitives shared by custom CFG reconstruction.

Static jump-table recovery is deliberately VEX-driven and therefore portable
across architectures when their lifted transfer has a recognized shape.  It
supports direct and relative entries, one-, two-, four-, and eight-byte table
entries, VEX endianness and signed-entry semantics, guard-derived index bounds,
constant-mask index domains, constant bases, and bases proven from predecessor
register definitions. The 32-bit x86 PC-thunk helper is a narrow supplement for
PIC code whose table base is not retained as a VEX constant.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable
from dataclasses import replace
from typing import Any, Literal

from angr import Project, options as angr_options
from angr.knowledge_plugins.cfg import CFGNode
from loguru import logger
import pyvex

from bingraph.helpers.capstone import (
    InsnSemantics,
    arch_has_delay_slot,
    control_transfer_index,
    proven_unconditional_direct_target,
)
from .decode import DecodedNode, lift_instruction_vex
from .graph import (
    CFGGraph,
    iter_graph_bound_nodes as _iter_graph_bound_nodes,
    node_intersects_bounds as _node_intersects_bounds,
    node_is_materialized_cfg_node as _node_is_materialized_cfg_node,
    node_is_placeholder as _node_is_placeholder,
    node_vex as _node_vex,
)
from .models import (
    BlockSpec,
    CFGAnomaly,
    FunctionBounds,
    JumpSuccessorAnalysis,
    JumpSuccessorExpectation,
    StaticJumpTable,
    StaticJumpTablePlan,
)


# Static table recovery is deliberately bounded. Larger index domains require a
# stronger range proof than the local VEX matcher currently provides.
MAX_STATIC_JUMPTABLE_ENTRIES = 256
MAX_ABI_STATIC_TARGET_WORKLIST_UPDATES = 4096


def is_direct_target_valid(bounds: FunctionBounds, target: int | None) -> bool:
    """Return whether one direct target remains inside the function range."""

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


def _vex_tmp_definitions(vex) -> dict[int, Any]:
    """Return the local VEX temporary definitions used to unfold expressions."""

    return {
        stmt.tmp: stmt.data
        for stmt in vex.statements
        if isinstance(stmt, pyvex.stmt.WrTmp)
    }


def _resolve_vex_expr(expr, definitions: dict[int, Any]):
    """Follow local VEX temporary references until reaching a concrete expression."""

    seen: set[int] = set()
    while isinstance(expr, pyvex.expr.RdTmp):
        if expr.tmp in seen:
            return None
        seen.add(expr.tmp)
        expr = definitions.get(expr.tmp)
        if expr is None:
            return None
    return expr


def is_direct_memory_indirect_jump(project: Project, node: CFGNode) -> bool:
    """Return whether an indirect jump loads its target from dynamic memory.

    A direct load through a register or stack address is a dynamic dispatch,
    such as a vtable slot, rather than evidence that arbitrary disconnected
    code is a target.  A load whose effective address has a mapped static base
    remains eligible for static-table recovery, even when its finite index
    proof is not available yet.
    """

    vex = _node_vex(node)
    if vex is None or vex.jumpkind != "Ijk_Boring":
        return False
    definitions = _vex_tmp_definitions(vex)
    next_expr = _resolve_vex_expr(vex.next, definitions)
    if not isinstance(next_expr, pyvex.expr.Load):
        return False

    address_terms = _vex_add_terms(next_expr.addr, definitions)
    if address_terms is None:
        return True
    address_mask = (1 << next_expr.addr.result_size(vex.tyenv)) - 1
    static_base = (
        sum(
            value
            for term in address_terms
            if (value := _vex_const_value(term, definitions)) is not None
        )
        & address_mask
    )
    return project.loader.find_object_containing(static_base) is None


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


def _vex_register_with_displacement(
    expr, definitions: dict[int, Any], vex
) -> tuple[tuple[int, int], int] | None:
    """Return one register plus its static displacement from a VEX expression."""

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


def _vex_add_terms(expr, definitions: dict[int, Any]) -> list[Any] | None:
    """Flatten a VEX integer-addition expression into its non-additive terms."""

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


def _amd64_sysv_register_layout(
    project: Project,
) -> tuple[dict[int, int], set[int]] | None:
    """Return AMD64 SysV alias writes and canonical callee-saved offsets."""

    if project.arch.name != "AMD64":
        return None
    object_os = getattr(project.loader.main_object, "os", "")
    if not isinstance(object_os, str) or not object_os.startswith("UNIX"):
        return None

    general_registers = [
        register
        for register in project.arch.register_list
        if register.general_purpose and register.name != "rip"
    ]
    offsets = {register.name: register.vex_offset for register in general_registers}
    if not {"rbx", "rbp", "r12", "r13", "r14", "r15"} <= offsets.keys():
        return None

    alias_writes: dict[int, int] = {}
    for register in general_registers:
        alias_writes[register.vex_offset] = register.vex_offset
        for _, relative_offset, _ in register.subregisters:
            alias_writes[register.vex_offset + relative_offset] = register.vex_offset
    return alias_writes, {
        offsets[name] for name in ("rbx", "rbp", "r12", "r13", "r14", "r15")
    }


def _abi_static_target_is_valid(project: Project, target: int) -> bool:
    """Return whether one propagated address is a materializable code target."""

    rejection = static_jump_target_rejection_reason(project, target)
    if rejection is None:
        return True
    if rejection != "synthetic":
        return False
    symbol = project.loader.find_symbol(target)
    return bool(symbol is not None and getattr(symbol, "is_function", False))


def _abi_static_pointer_target(project: Project, vex, load) -> int | None:
    """Read one exact static pointer load when it names executable code."""

    if not isinstance(load.addr, pyvex.expr.Const):
        return None
    slot_addr = load.addr.con.value
    entry_size = load.result_size(vex.tyenv) // 8
    if not isinstance(slot_addr, int) or entry_size not in {4, 8}:
        return None
    try:
        raw = project.loader.memory.load(slot_addr, entry_size)
    except Exception:
        return None
    byteorder = "little" if "LE" in load.end else "big"
    target = int.from_bytes(raw, byteorder=byteorder)
    return target if _abi_static_target_is_valid(project, target) else None


def _abi_static_expression_target(
    project: Project,
    vex,
    expr,
    definitions: dict[int, Any],
    state: dict[int, int],
    alias_writes: dict[int, int],
) -> int | None:
    """Evaluate one VEX expression in the small static-target domain."""

    expr = _resolve_vex_expr(expr, definitions)
    if isinstance(expr, pyvex.expr.Const):
        value = expr.con.value
        return (
            value
            if isinstance(value, int) and _abi_static_target_is_valid(project, value)
            else None
        )
    if isinstance(expr, pyvex.expr.Get):
        canonical_offset = alias_writes.get(expr.offset)
        if (
            canonical_offset != expr.offset
            or expr.result_size(vex.tyenv) != project.arch.bits
        ):
            return None
        return state.get(canonical_offset)
    if isinstance(expr, pyvex.expr.Load):
        return _abi_static_pointer_target(project, vex, expr)
    return None


def _abi_transfer_static_targets(
    project: Project,
    block: BlockSpec,
    alias_writes: dict[int, int],
    state: dict[int, int],
) -> tuple[dict[int, int], int | None]:
    """Apply one block's VEX register writes and return its indirect target."""

    try:
        vex = project.factory.block(
            block.addr,
            size=block.size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception:
        return {}, None

    definitions = _vex_tmp_definitions(vex)
    output = dict(state)
    for statement in vex.statements:
        if not isinstance(statement, pyvex.stmt.Put):
            continue
        canonical_offset = alias_writes.get(statement.offset)
        if canonical_offset is None:
            continue
        if statement.data.result_size(vex.tyenv) != project.arch.bits:
            output.pop(canonical_offset, None)
            continue
        target = _abi_static_expression_target(
            project, vex, statement.data, definitions, output, alias_writes
        )
        if target is None:
            output.pop(canonical_offset, None)
        else:
            output[canonical_offset] = target
    next_expr = _resolve_vex_expr(vex.next, definitions)
    if not isinstance(next_expr, pyvex.expr.Get):
        return output, None
    target = _abi_static_expression_target(
        project, vex, next_expr, definitions, output, alias_writes
    )
    return output, target


def _abi_successor_addrs(
    blocks: dict[int, BlockSpec], block: BlockSpec
) -> tuple[int, ...]:
    """Return intraprocedural continuations used by must-dataflow analysis."""

    if block.jumpkind == "Ijk_Call":
        candidates = (block.fallthrough_addr,)
    else:
        candidates = (*block.direct_targets, block.fallthrough_addr)
    return tuple(target for target in candidates if target in blocks)


def _abi_join_static_states(states: Iterable[dict[int, int]]) -> dict[int, int]:
    """Keep only target values established by every incoming path."""

    iterator = iter(states)
    try:
        joined = dict(next(iterator))
    except StopIteration:
        return {}
    for state in iterator:
        joined = {
            offset: target
            for offset, target in joined.items()
            if state.get(offset) == target
        }
    return joined


def abi_static_register_transfer_targets(
    project: Project,
    bounds: FunctionBounds,
    blocks: dict[int, BlockSpec],
) -> tuple[dict[int, int], bool, bool]:
    """Resolve register transfers proven by the active function ABI.

    The analysis is deliberately a must-analysis over exact code addresses.
    Unknown values and disagreeing predecessors are discarded, while calls
    preserve only registers guaranteed by the selected ABI.  It returns no
    partial answers when its bounded worklist is exhausted.  The final flag
    reports whether the ABI profile and at least one candidate were present.
    """

    profile = _amd64_sysv_register_layout(project)
    if profile is None:
        return {}, False, False
    alias_writes, preserved_offsets = profile
    candidates = {
        addr: block
        for addr, block in blocks.items()
        if block.jumpkind in {"Ijk_Boring", "Ijk_Call"} and not block.direct_targets
    }
    if not candidates:
        return {}, False, False

    successors = {
        addr: _abi_successor_addrs(blocks, block) for addr, block in blocks.items()
    }
    predecessors: dict[int, set[int]] = {addr: set() for addr in blocks}
    for source, targets in successors.items():
        for target in targets:
            predecessors[target].add(source)

    in_states: dict[int, dict[int, int]] = {bounds.addr: {}}
    out_states: dict[int, dict[int, int]] = {}
    transfer_targets: dict[int, int] = {}
    pending = deque([bounds.addr])
    updates = 0
    while pending:
        addr = pending.popleft()
        updates += 1
        if updates > MAX_ABI_STATIC_TARGET_WORKLIST_UPDATES:
            return {}, True, True
        block = blocks[addr]
        output, target = _abi_transfer_static_targets(
            project, block, alias_writes, in_states[addr]
        )
        if block.jumpkind == "Ijk_Call":
            output = {
                offset: value
                for offset, value in output.items()
                if offset in preserved_offsets
            }
        if out_states.get(addr) == output:
            continue
        out_states[addr] = output
        if addr in candidates and target is not None:
            transfer_targets[addr] = target
        else:
            transfer_targets.pop(addr, None)
        for successor in successors[addr]:
            if successor == bounds.addr:
                continue
            incoming = [
                out_states[pred]
                for pred in predecessors[successor]
                if pred in out_states
            ]
            if not incoming:
                continue
            joined = _abi_join_static_states(incoming)
            if in_states.get(successor) != joined:
                in_states[successor] = joined
                pending.append(successor)
    return transfer_targets, False, True


def _mips_entry_global_pointer(project: Project, bounds: FunctionBounds) -> int | None:
    """Resolve the MIPS PIC global pointer established from entry ``$t9``."""

    if not project.arch.name.startswith("MIPS"):
        return None
    try:
        gp_offset = project.arch.registers["gp"][0]
        t9_offset = project.arch.registers["t9"][0]
        entry_vex = project.factory.block(bounds.addr).vex
    except (AttributeError, KeyError):
        return None
    except Exception:
        return None

    definitions = _vex_tmp_definitions(entry_vex)
    for statement in entry_vex.statements:
        if not isinstance(statement, pyvex.stmt.Put) or statement.offset != gp_offset:
            continue
        expression = _resolve_vex_expr(statement.data, definitions)
        if not isinstance(expression, pyvex.expr.Binop) or not expression.op.startswith(
            "Iop_Add"
        ):
            continue
        left, right = expression.args
        constant, register = (
            (left, right) if isinstance(left, pyvex.expr.Const) else (right, left)
        )
        if not isinstance(constant, pyvex.expr.Const):
            continue
        if _vex_get_key(register, definitions, entry_vex) != (
            t9_offset,
            project.arch.bits,
        ):
            continue
        return (bounds.addr + constant.con.value) & ((1 << project.arch.bits) - 1)
    return None


def _vex_index_key(
    expr,
    definitions: dict[int, Any],
    vex,
    *,
    allow_full_width: bool = False,
) -> tuple[int, int] | None:
    """Return the original register identity for a table index expression."""

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
    """Describe a table index as either a register or a finite value domain."""

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
    """Return byte-addressable aliases representing a register's low bits."""

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


def _vex_affine_difference_index(
    expr, definitions: dict[int, Any], vex
) -> tuple[tuple[int, int], tuple[int, int], int] | None:
    """Describe ``2^n - 1 + left - right`` table indices.

    The bounded proof for this form is intentionally separate from ordinary
    register indices.  Arithmetic alone does not make a finite table domain;
    callers must still prove masked operands and their ordering on every path.
    """

    expr = _resolve_vex_expr(expr, definitions)
    if not isinstance(expr, pyvex.expr.Binop) or not expr.op.startswith("Iop_Sub"):
        return None

    left = _vex_register_with_displacement(expr.args[0], definitions, vex)
    right = _vex_get_key(expr.args[1], definitions, vex)
    if left is None or right is None:
        return None
    left_key, displacement = left
    if left_key[1] != right[1] or displacement <= 0:
        return None
    if (
        displacement + 1 > MAX_STATIC_JUMPTABLE_ENTRIES
        or (displacement + 1) & displacement
    ):
        return None
    return left_key, right, displacement


def _vex_static_int(expr, definitions: dict[int, Any]) -> int | None:
    """Evaluate the small constant-only VEX expressions used in branch guards."""

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


def _vex_low_bits_source(expr, definitions: dict[int, Any], tyenv, bits: int):
    """Return the expression supplying ``bits`` unchanged low-order bits."""

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
    """Return the source and width of an all-ones low-bit mask expression."""

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
    """Return a structural key after removing guard-width zext/truncation."""

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
    """Return whether ``expr`` zero-extends exactly ``bits`` into a register."""

    expr = _resolve_vex_expr(expr, definitions)
    conversion = _vex_width_conversion(expr)
    return conversion == (bits, register_bits, "U")


def _vex_guarded_index_upper_bound(
    vex, target_addr: int, index_key: tuple[int, int]
) -> int | None:
    """Return a proven unsigned bound for a VEX path entering ``target_addr``.

    A compiler may encode the dispatcher as either the taken ``Exit`` or the
    default ``NEXT`` path.  In the latter form, the exit guard is the inverse
    of the table-domain condition, such as ``limit <u index``.
    """

    definitions = _vex_tmp_definitions(vex)
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
    """Return a bound when a predecessor guards one exact selector expression."""

    definitions = _vex_tmp_definitions(vex)
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
    """Return an unsigned bound implied when ``guard`` has the given truth."""

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
    bound = _vex_static_int(bound_expr, definitions)
    if bound is None:
        return None

    if "CmpLT" in guard.op:
        upper_bound = bound - 1 if index_on_left else bound
    elif "CmpLE" in guard.op:
        upper_bound = bound if index_on_left else bound - 1
    else:
        return None
    return upper_bound if upper_bound >= 0 else None


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
    bound = _vex_static_int(bound_expr, definitions)
    if bound is None:
        return None

    if "CmpLT" in guard.op:
        upper_bound = bound - 1 if index_on_left else bound
    elif "CmpLE" in guard.op:
        upper_bound = bound if index_on_left else bound - 1
    else:
        return None
    return upper_bound if upper_bound >= 0 else None


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
    """Rewrite register reads in one key through the predecessor's last PUT."""

    register_values = {
        statement.offset: _vex_expr_key(statement.data, definitions)
        for statement in vex.statements
        if isinstance(statement, pyvex.stmt.Put)
    }

    def rewrite(value: tuple[Any, ...], seen: frozenset[int]) -> tuple[Any, ...] | None:
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
    bound = _vex_static_int(guard.args[1], definitions)
    if bound is None:
        return None
    if "CmpLE" in guard.op:
        upper_bound = bound
    elif "CmpLT" in guard.op:
        upper_bound = bound - 1
    else:
        return None
    if upper_bound < 0 or upper_bound >= MAX_STATIC_JUMPTABLE_ENTRIES:
        return None
    return tuple(range(upper_bound + 1))


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
    bound = _vex_static_int(guard.args[1], definitions)
    if bound is None:
        return None
    if "CmpLE" in guard.op:
        upper_bound = bound
    elif "CmpLT" in guard.op:
        upper_bound = bound - 1
    else:
        return None
    if upper_bound < 0 or upper_bound >= MAX_STATIC_JUMPTABLE_ENTRIES:
        return None
    return tuple(range(upper_bound + 1))


def _vex_guard_matches_index_register(
    expr,
    index_key: tuple[int, int],
    definitions: dict[int, Any],
    vex,
    preceding_statements: tuple[Any, ...] | list[Any],
) -> bool:
    """Return whether a guard expression is the index register's current value."""

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
    for stmt in reversed(preceding_statements):
        if not isinstance(stmt, pyvex.stmt.Put) or stmt.offset != index_key[0]:
            continue
        value = _resolve_vex_expr(stmt.data, definitions)
        if value is resolved_expr or value == resolved_expr:
            return True
        if _vex_is_zero_extension_from(value, definitions, guard_bits, index_key[1]):
            if _vex_guard_width_view_key(
                value, definitions, vex.tyenv, guard_bits
            ) == _vex_guard_width_view_key(
                resolved_expr, definitions, vex.tyenv, guard_bits
            ):
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


def _vex_register_value_at_block_entry(
    vex, register_key: tuple[int, int]
) -> tuple[int, int] | None:
    """Trace one unchanged register value through a short VEX block."""

    definitions = _vex_tmp_definitions(vex)
    statement = _vex_last_put(vex, register_key[0])
    if statement is None:
        return register_key

    expression = _resolve_vex_expr(statement.data, definitions)
    while (conversion := _vex_width_conversion(expression)) is not None:
        source_bits, destination_bits, _ = conversion
        if source_bits <= 0 or destination_bits <= 0:
            return None
        expression = _resolve_vex_expr(expression.args[0], definitions)
    key = _vex_get_key(expression, definitions, vex)
    return key if key is not None and key[1] >= register_key[1] else None


def _vex_low_masked_register(
    expr, definitions: dict[int, Any], vex
) -> tuple[tuple[int, int], int] | None:
    """Return a register and width for a value masked to its low ``n`` bits."""

    expression = _resolve_vex_expr(expr, definitions)
    while expression is not None:
        masked = _vex_low_masked_source(expression, definitions, vex.tyenv)
        if masked is not None:
            source, bits = masked
            key = _vex_get_key(source, definitions, vex)
            if key is not None:
                return key, bits
        conversion = _vex_width_conversion(expression)
        if conversion is None:
            return None
        expression = _resolve_vex_expr(expression.args[0], definitions)
    return None


def _vex_low_width_key(
    expr, definitions: dict[int, Any], vex, bits: int
) -> tuple[Any, ...] | None:
    """Return an expression key after preserving only its low ``bits`` bits."""

    source = _vex_low_bits_source(expr, definitions, vex.tyenv, bits)
    return _vex_expr_key(source, definitions) if source is not None else None


def _x86_cc_offsets(vex) -> tuple[int, int, int] | None:
    """Return the VEX flag pseudo-register offsets for an x86-family block."""

    arch = getattr(vex, "arch", None)
    if arch is None or getattr(arch, "name", None) not in {"AMD64", "X86"}:
        return None
    try:
        registers = arch.registers
        return tuple(registers[name][0] for name in ("cc_op", "cc_dep1", "cc_dep2"))
    except (AttributeError, KeyError):
        return None


def _x86_unsigned_branch_relation(vex, target_addr: int) -> str | None:
    """Return the unsigned relation selected by one x86 conditional edge."""

    offsets = _x86_cc_offsets(vex)
    if offsets is None:
        return None
    definitions = _vex_tmp_definitions(vex)
    exits = [
        statement
        for statement in vex.statements
        if isinstance(statement, pyvex.stmt.Exit)
    ]
    if len(exits) != 1:
        return None
    exit_statement = exits[0]
    exit_target = getattr(exit_statement.dst, "value", None)
    if target_addr == exit_target:
        taken = True
    elif target_addr == _vex_const_value(vex.next, definitions):
        taken = False
    else:
        return None
    if any(_vex_last_put(vex, offset) is not None for offset in offsets):
        return None

    guard = _resolve_vex_expr(exit_statement.guard, definitions)
    while isinstance(guard, pyvex.expr.Unop):
        guard = _resolve_vex_expr(guard.args[0], definitions)
    if not isinstance(guard, pyvex.expr.CCall):
        return None
    if getattr(guard.callee, "name", None) not in {
        "amd64g_calculate_condition",
        "x86g_calculate_condition",
    }:
        return None
    condition = _vex_const_value(guard.args[0], definitions)
    if len(guard.args) != 5 or condition not in {
        2,
        3,
        6,
        7,
    }:
        return None
    flag_keys = tuple(
        _vex_get_key(argument, definitions, vex) for argument in guard.args[1:4]
    )
    if (
        any(key is None for key in flag_keys)
        or tuple(key[0] for key in flag_keys if key is not None) != offsets
    ):
        return None

    assert condition is not None
    relation = {2: "lt", 3: "ge", 6: "le", 7: "gt"}[condition]
    if taken:
        return relation
    return {"lt": "ge", "ge": "lt", "le": "gt", "gt": "le"}[relation]


def _x86_affine_compare_anchor(
    vex, successor_addr: int
) -> tuple[tuple[int, int], tuple[int, int], int, bool] | None:
    """Return masked ``cmp`` operands and whether its fallthrough excludes equality."""

    offsets = _x86_cc_offsets(vex)
    if offsets is None:
        return None
    definitions = _vex_tmp_definitions(vex)
    cc_op, cc_dep1, cc_dep2 = offsets
    op_statement = _vex_last_put(vex, cc_op)
    dep1_statement = _vex_last_put(vex, cc_dep1)
    dep2_statement = _vex_last_put(vex, cc_dep2)
    if op_statement is None or dep1_statement is None or dep2_statement is None:
        return None
    # x86 VEX encodes byte, word, dword, and qword subtraction as 5 through 8.
    if _vex_static_int(op_statement.data, definitions) not in {5, 6, 7, 8}:
        return None
    dep1 = _vex_low_masked_register(dep1_statement.data, definitions, vex)
    dep2 = _vex_low_masked_register(dep2_statement.data, definitions, vex)
    if dep1 is None or dep2 is None or dep1[1] != dep2[1]:
        return None

    equality_guarded = False
    if _vex_const_value(vex.next, definitions) == successor_addr:
        dep1_key = _vex_low_width_key(dep1_statement.data, definitions, vex, dep1[1])
        dep2_key = _vex_low_width_key(dep2_statement.data, definitions, vex, dep2[1])
        for statement in vex.statements:
            if not isinstance(statement, pyvex.stmt.Exit):
                continue
            guard = _resolve_vex_expr(statement.guard, definitions)
            while isinstance(guard, pyvex.expr.Unop):
                guard = _resolve_vex_expr(guard.args[0], definitions)
            if not isinstance(guard, pyvex.expr.Binop) or not guard.op.startswith(
                "Iop_CmpEQ"
            ):
                continue
            left = _vex_low_width_key(guard.args[0], definitions, vex, dep1[1])
            right = _vex_low_width_key(guard.args[1], definitions, vex, dep1[1])
            if (left, right) in {(dep1_key, dep2_key), (dep2_key, dep1_key)}:
                equality_guarded = True
                break
    return dep1[0], dep2[0], dep1[1], equality_guarded


def _x86_affine_index_path_is_ordered(
    graph: CFGGraph,
    bounds: FunctionBounds,
    predecessor,
    node_addr: int,
    left_key: tuple[int, int],
    right_key: tuple[int, int],
    mask_bits: int,
) -> bool:
    """Prove one dispatcher predecessor leaves a masked affine index ordered."""

    predecessor_vex = _node_vex(predecessor)
    if predecessor_vex is None:
        return False
    left_at_entry = _vex_register_value_at_block_entry(predecessor_vex, left_key)
    right_at_entry = _vex_register_value_at_block_entry(predecessor_vex, right_key)
    if left_at_entry is None or right_at_entry is None:
        return False

    branch = predecessor
    relation = _x86_unsigned_branch_relation(predecessor_vex, node_addr)
    if relation is None:
        branch_predecessors = [
            candidate
            for candidate in graph.predecessors(predecessor)
            if _node_is_materialized_cfg_node(candidate)
            and _node_intersects_bounds(candidate, bounds)
        ]
        if len(branch_predecessors) != 1:
            return False
        branch = branch_predecessors[0]
        branch_vex = _node_vex(branch)
        if branch_vex is None:
            return False
        relation = _x86_unsigned_branch_relation(branch_vex, predecessor.addr)
        if relation is None:
            return False

    anchor_predecessors = [
        candidate
        for candidate in graph.predecessors(branch)
        if _node_is_materialized_cfg_node(candidate)
        and _node_intersects_bounds(candidate, bounds)
    ]
    if len(anchor_predecessors) != 1:
        return False
    anchor_vex = _node_vex(anchor_predecessors[0])
    if anchor_vex is None:
        return False
    anchor = _x86_affine_compare_anchor(anchor_vex, branch.addr)
    if anchor is None:
        return False
    dep1, dep2, bits, excludes_equality = anchor
    if bits != mask_bits:
        return False

    if relation == "le" and excludes_equality:
        relation = "lt"
    elif relation == "ge" and excludes_equality:
        relation = "gt"
    if (left_at_entry, right_at_entry) == (dep1, dep2):
        return relation == "lt"
    if (left_at_entry, right_at_entry) == (dep2, dep1):
        return relation == "gt"
    return False


def _guarded_affine_difference_index_values(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    table: StaticJumpTable,
) -> tuple[int, ...] | None:
    """Return exact values for a masked, ordered x86 affine table index."""

    difference = table.index_affine_difference
    if difference is None:
        return None
    left_key, right_key, displacement = difference
    if (
        displacement + 1 > MAX_STATIC_JUMPTABLE_ENTRIES
        or (displacement + 1) & displacement
    ):
        return None
    mask_bits = displacement.bit_length()
    predecessors = [
        predecessor
        for predecessor in graph.predecessors(node)
        if _node_is_materialized_cfg_node(predecessor)
        and _node_intersects_bounds(predecessor, bounds)
    ]
    if not predecessors:
        return None
    if not all(
        _x86_affine_index_path_is_ordered(
            graph,
            bounds,
            predecessor,
            node.addr,
            left_key,
            right_key,
            mask_bits,
        )
        for predecessor in predecessors
    ):
        return None
    return tuple(range(displacement))


def _guarded_jump_table_entry_count(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    table: StaticJumpTable,
) -> int | None:
    """Return the bounded table length proven by a predecessor branch."""

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

    if len(bounds_found) != 1:
        return None
    upper_bound = next(iter(bounds_found))
    entry_count = upper_bound + 1
    return entry_count if entry_count <= MAX_STATIC_JUMPTABLE_ENTRIES else None


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

    The accepted form is intentionally narrow: ``next`` must add a register
    base to a loaded (optionally sign-extended) table entry, while the load
    address must be that same base plus an index scaled by the entry size. The
    optional expression form accepts a memory-reading selector only when a
    predecessor separately proves its finite range.
    """

    if vex.jumpkind != "Ijk_Boring":
        return None

    definitions = _vex_tmp_definitions(vex)
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
        if entry_size not in {1, 2, 4, 8}:
            continue

        address_terms = _vex_add_terms(entry_expr.addr, definitions)
        if address_terms is None:
            continue

        constant_total = 0
        displacement = 0
        saw_base = False
        index_key: tuple[int, int] | None = None
        index_bits: int | None = None
        index_values: tuple[int, ...] | None = None
        index_expression: tuple[Any, ...] | None = None
        index_affine_difference: tuple[tuple[int, int], tuple[int, int], int] | None = (
            None
        )
        preserve_unresolved_fallback = False
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
            scaled_index = _vex_scaled_table_index(term, definitions)
            if scaled_index is None:
                break
            index_expr, shift, bounded_values, masked_index_bits = scaled_index
            candidate_key, candidate_values = _vex_table_index(
                index_expr,
                definitions,
                vex,
                allow_full_width=allow_full_width_index,
                allow_inline_index_values=allow_inline_index_values,
                allow_masked_index_values=allow_masked_index_values,
            )
            if candidate_key is None and masked_index_bits is not None:
                # The rotate-mask pattern establishes a finite low-bit domain.
                # Retain its full register solely to intersect that domain with
                # a predecessor guard expressed through a byte-sized alias.
                candidate_key = _vex_index_key(
                    index_expr, definitions, vex, allow_full_width=True
                )
            if candidate_values is None:
                candidate_values = bounded_values
            preserve_unresolved_fallback |= (
                masked_index_bits is not None and bounded_values is None
            )
            candidate_affine_difference = None
            if candidate_key is None and candidate_values is None:
                candidate_affine_difference = _vex_affine_difference_index(
                    index_expr, definitions, vex
                )
            candidate_expression = None
            if (
                candidate_key is None
                and candidate_values is None
                and candidate_affine_difference is None
                and allow_guarded_expression_index
            ):
                candidate_expression = _vex_expr_key(index_expr, definitions)
                if candidate_expression is None or not _vex_key_reads_memory(
                    candidate_expression
                ):
                    candidate_expression = None
            if (
                shift is None
                or (
                    candidate_key is None
                    and candidate_values is None
                    and candidate_affine_difference is None
                    and candidate_expression is None
                )
                or 1 << shift != entry_size
                or index_key is not None
                or index_values is not None
                or index_affine_difference is not None
                or index_expression is not None
            ):
                break
            index_key = candidate_key
            index_bits = candidate_key[1] if candidate_key is not None else None
            index_values = candidate_values
            index_affine_difference = candidate_affine_difference
            index_expression = candidate_expression
        else:
            if static_base_addr is not None:
                mask = (1 << base_bits) - 1
                displacement = (constant_total - static_base_addr) & mask
                saw_base = True
            else:
                displacement = constant_total
            if saw_base and (
                index_key is not None
                or index_values is not None
                or index_affine_difference is not None
                or index_expression is not None
            ):
                offset = base_key[0] if base_key is not None else None
                table_displacement = displacement & ((1 << base_bits) - 1)
                return StaticJumpTable(
                    base_register_offset=offset,
                    base_bits=base_bits,
                    table_displacement=table_displacement,
                    index_register_offset=(
                        index_key[0] if index_key is not None else None
                    ),
                    index_bits=index_bits,
                    entry_size=entry_size,
                    endness=entry_expr.end,
                    signed_entries=signed_entries,
                    target_displacement=target_displacement,
                    static_base_addr=static_base_addr,
                    index_values=index_values,
                    index_low_bits=masked_index_bits,
                    index_expression=index_expression,
                    index_affine_difference=index_affine_difference,
                    preserve_unresolved_fallback=preserve_unresolved_fallback,
                )

    return None


def _vex_static_compact_table_load(
    vex,
    definitions: dict[int, Any],
    entry_expr,
    *,
    allow_full_width_index: bool = False,
    allow_masked_index_values: bool = False,
) -> StaticJumpTable | None:
    """Describe an 8- or 16-bit table loaded from a static base plus an index.

    Larger entries are already handled by the ordinary direct and relative
    table matchers. Compact entries need a separate target scale, which this
    helper leaves to the enclosing PC-target matcher.
    """

    entry = _vex_normalized_table_entry_load(entry_expr, definitions)
    if entry is None:
        return None
    load, signed_entries = entry
    entry_size = load.result_size(vex.tyenv) // 8
    if entry_size not in {1, 2}:
        return None

    address_terms = _vex_add_terms(load.addr, definitions)
    if address_terms is None:
        return None

    static_base_addr = 0
    index_key = None
    index_values = None
    for term in address_terms:
        value = _vex_const_value(term, definitions)
        if value is not None:
            static_base_addr += value
            continue

        candidate_index, candidate_values = _vex_table_index(
            term,
            definitions,
            vex,
            allow_full_width=allow_full_width_index,
            allow_inline_index_values=False,
            allow_masked_index_values=allow_masked_index_values,
        )
        if candidate_index is None and candidate_values is None:
            term = _resolve_vex_expr(term, definitions)
            if not isinstance(term, pyvex.expr.Binop) or not term.op.startswith(
                "Iop_Shl"
            ):
                return None
            shift = _vex_const_value(term.args[1], definitions)
            candidate_index, candidate_values = _vex_table_index(
                term.args[0],
                definitions,
                vex,
                allow_full_width=allow_full_width_index,
                allow_inline_index_values=False,
                allow_masked_index_values=allow_masked_index_values,
            )
            if shift is None or 1 << shift != entry_size:
                return None
        elif entry_size != 1:
            return None
        if (
            (candidate_index is None and candidate_values is None)
            or index_key is not None
            or index_values is not None
        ):
            return None
        index_key = candidate_index
        index_values = candidate_values

    if index_key is None and index_values is None:
        return None
    address_bits = load.addr.result_size(vex.tyenv)
    if address_bits <= 0:
        return None

    return StaticJumpTable(
        base_register_offset=None,
        base_bits=address_bits,
        table_displacement=0,
        index_register_offset=index_key[0] if index_key is not None else None,
        index_bits=index_key[1] if index_key is not None else None,
        entry_size=entry_size,
        endness=load.end,
        signed_entries=signed_entries,
        static_base_addr=static_base_addr & ((1 << address_bits) - 1),
        index_values=index_values,
    )


def _vex_scaled_relative_jump_table(
    vex,
    *,
    allow_full_width_index: bool = False,
    allow_masked_index_values: bool = False,
) -> StaticJumpTable | None:
    """Describe a compact static table whose scaled entries update the PC.

    Accept only ``next = (base + (LoadN(base + index * N) << shift)) | mask``
    for 8- or 16-bit entries. The duplicated static base and VEX load, shift,
    and or operations prove the table address, target scale, and target-mode
    bits without using an instruction-set mnemonic.
    """

    if vex.jumpkind != "Ijk_Boring":
        return None

    definitions = _vex_tmp_definitions(vex)
    next_expr = _resolve_vex_expr(vex.next, definitions)
    target_or_mask = 0
    if isinstance(next_expr, pyvex.expr.Binop) and next_expr.op.startswith("Iop_Or"):
        left, right = (
            _resolve_vex_expr(next_expr.args[0], definitions),
            _resolve_vex_expr(next_expr.args[1], definitions),
        )
        left_value = _vex_const_value(left, definitions)
        right_value = _vex_const_value(right, definitions)
        if left_value is not None and right_value is None:
            next_expr, target_or_mask = right, left_value
        elif right_value is not None and left_value is None:
            next_expr, target_or_mask = left, right_value
        else:
            return None

    terms = _vex_add_terms(next_expr, definitions)
    if terms is None:
        return None

    target_base = 0
    shifted_entry = None
    for term in terms:
        value = _vex_const_value(term, definitions)
        if value is not None:
            target_base += value
            continue
        term = _resolve_vex_expr(term, definitions)
        if (
            not isinstance(term, pyvex.expr.Binop)
            or not term.op.startswith("Iop_Shl")
            or shifted_entry is not None
        ):
            return None
        shift = _vex_const_value(term.args[1], definitions)
        if shift is None or shift < 0 or shift >= term.result_size(vex.tyenv):
            return None
        shifted_entry = term.args[0], shift

    if shifted_entry is None:
        return None
    entry_expr, shift = shifted_entry
    table = _vex_static_compact_table_load(
        vex,
        definitions,
        entry_expr,
        allow_full_width_index=allow_full_width_index,
        allow_masked_index_values=allow_masked_index_values,
    )
    if table is None or table.static_base_addr != target_base:
        return None

    target_mask = (1 << table.base_bits) - 1
    return replace(
        table,
        entries_are_relative=True,
        target_scale=1 << shift,
        target_or_mask=target_or_mask & target_mask,
    )


def _vex_direct_jump_table(
    vex,
    *,
    allow_full_width_index: bool = False,
    allow_inline_index_values: bool = False,
    allow_masked_index_values: bool = False,
    allow_guarded_loads: bool = False,
    allow_static_base: bool = False,
    allow_guarded_expression_index: bool = False,
) -> StaticJumpTable | None:
    """
    Describe a bounded table whose entries are absolute jump destinations.

    The accepted VEX form is ``next = Load(base + index * entry_size + disp)``.
    Unlike relative tables, the loaded entry is itself the target address. The
    index must be a register. The caller separately requires a matching
    predecessor guard before it reads any finite number of table entries.
    """

    if vex.jumpkind != "Ijk_Boring":
        return None

    definitions = _vex_tmp_definitions(vex)
    entry_expr = _resolve_vex_expr(vex.next, definitions)
    if isinstance(entry_expr, pyvex.expr.Load):
        return _vex_direct_table_from_load(
            vex,
            definitions,
            entry_expr.addr,
            entry_expr.result_size(vex.tyenv) // 8,
            entry_expr.end,
            guard=None,
            allow_full_width_index=allow_full_width_index,
            allow_inline_index_values=allow_inline_index_values,
            allow_masked_index_values=allow_masked_index_values,
            allow_static_base=allow_static_base,
            allow_guarded_expression_index=allow_guarded_expression_index,
        )

    if not allow_guarded_loads:
        return None
    return _vex_guarded_load_pc_table(
        vex,
        definitions,
        entry_expr,
        allow_full_width_index=allow_full_width_index,
        allow_inline_index_values=allow_inline_index_values,
        allow_masked_index_values=allow_masked_index_values,
    )


def _vex_guarded_load_pc_table(
    vex,
    definitions: dict[int, Any],
    next_expr,
    *,
    allow_full_width_index: bool,
    allow_inline_index_values: bool,
    allow_masked_index_values: bool,
) -> StaticJumpTable | None:
    """Describe a guarded VEX ``LoadG`` value selected as the next PC.

    VEX uses ``LoadG`` for a conditional memory load. Some instruction sets
    use that load as a computed program counter, yielding ``next =
    ITE(guard, LoadG(...), old_pc)`` plus an ordinary fall-through ``Exit``.
    This is a portable VEX shape: no instruction mnemonic or architecture
    register name is required here.
    """

    if not isinstance(next_expr, pyvex.expr.ITE):
        return None
    selected = next_expr.iftrue
    if not isinstance(selected, pyvex.expr.RdTmp):
        return None

    condition = _resolve_vex_expr(next_expr.cond, definitions)
    if condition is None:
        return None
    for statement in vex.statements:
        if not isinstance(statement, pyvex.stmt.LoadG) or statement.dst != selected.tmp:
            continue
        guard = _resolve_vex_expr(statement.guard, definitions)
        guard_key = _vex_expr_key(guard, definitions)
        condition_key = _vex_expr_key(condition, definitions)
        if guard_key is None or condition_key is None or guard_key != condition_key:
            continue
        entry_size = vex.tyenv.sizeof(statement.dst) // 8
        if statement.cvt != f"ILGop_Ident{entry_size * 8}":
            continue
        return _vex_direct_table_from_load(
            vex,
            definitions,
            statement.addr,
            entry_size,
            statement.end,
            guard=condition,
            allow_full_width_index=allow_full_width_index,
            allow_inline_index_values=allow_inline_index_values,
            allow_masked_index_values=allow_masked_index_values,
        )
    return None


def _vex_direct_table_from_load(
    vex,
    definitions: dict[int, Any],
    address_expr,
    entry_size: int,
    endness: str,
    *,
    guard,
    allow_full_width_index: bool,
    allow_inline_index_values: bool,
    allow_masked_index_values: bool,
    allow_static_base: bool = False,
    allow_guarded_expression_index: bool = False,
) -> StaticJumpTable | None:
    """Describe an absolute-address table load from its VEX address expression."""

    if entry_size not in {1, 2, 4, 8}:
        return None
    address_terms = _vex_add_terms(address_expr, definitions)
    if address_terms is None:
        return None

    base_bits = address_expr.result_size(vex.tyenv)
    displacement = 0
    base_key = None
    index_key = None
    index_bits = None
    index_values = None
    index_expression = None
    for term in address_terms:
        value = _vex_const_value(term, definitions)
        if value is not None:
            displacement += value
            continue

        register_key = _vex_get_key(term, definitions, vex)
        if register_key is not None and base_key is None:
            base_key = register_key
            continue

        term = _resolve_vex_expr(term, definitions)
        if not isinstance(term, pyvex.expr.Binop) or not term.op.startswith("Iop_Shl"):
            return None
        shift = _vex_const_value(term.args[1], definitions)
        candidate_index, candidate_values = _vex_table_index(
            term.args[0],
            definitions,
            vex,
            allow_full_width=allow_full_width_index,
            allow_inline_index_values=allow_inline_index_values,
            allow_masked_index_values=allow_masked_index_values,
        )
        if candidate_index is None and candidate_values is None and guard is not None:
            candidate_values = _vex_guarded_expression_values(
                guard, term.args[0], definitions
            )
        candidate_expression = None
        if (
            candidate_index is None
            and candidate_values is None
            and allow_guarded_expression_index
        ):
            candidate_expression = _vex_expr_key(term.args[0], definitions)
            if candidate_expression is None or not _vex_key_reads_memory(
                candidate_expression
            ):
                candidate_expression = None
        if (
            (
                candidate_index is None
                and candidate_values is None
                and candidate_expression is None
            )
            or shift is None
            or 1 << shift != entry_size
            or index_key is not None
            or index_values is not None
            or index_expression is not None
        ):
            return None
        index_key = candidate_index
        index_bits = candidate_index[1] if candidate_index is not None else None
        index_values = candidate_values
        index_expression = candidate_expression

    if index_key is None and index_values is None and index_expression is None:
        return None
    if base_key is None and guard is None and not allow_static_base:
        # Preserve the existing direct-table policy. An absolute base is only
        # safe here when the guarded LoadG has proved a PC-table dispatch.
        return None
    if guard is not None and index_key is not None:
        index_values = _vex_guarded_index_values(guard, index_key, definitions, vex)
    if guard is not None and index_values is None:
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
        index_register_offset=index_key[0] if index_key is not None else None,
        index_bits=index_bits,
        entry_size=entry_size,
        endness=endness,
        signed_entries=False,
        entries_are_relative=False,
        static_base_addr=static_base_addr,
        index_values=index_values,
        index_expression=index_expression,
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

    definitions = _vex_tmp_definitions(vex)
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

    definitions = _vex_tmp_definitions(vex)
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
            guard=None,
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


def _unique_entry_path(
    graph: CFGGraph, bounds: FunctionBounds, node: CFGNode
) -> tuple[CFGNode, ...] | None:
    """Return one acyclic in-function path from entry to ``node``.

    Symbolic successor recovery is deliberately limited to this strict shape.
    Multiple predecessor paths would require joining path constraints, which is
    beyond a local static jump-target proof.
    """

    path = [node]
    seen = {node}
    current = node
    while current.addr != bounds.addr:
        predecessors = [
            predecessor
            for predecessor in graph.predecessors(current)
            if _node_is_materialized_cfg_node(predecessor)
            and _node_intersects_bounds(predecessor, bounds)
        ]
        if len(predecessors) != 1:
            return None
        current = predecessors[0]
        if current in seen:
            return None
        seen.add(current)
        path.append(current)
    return tuple(reversed(path))


def _path_constrained_pc_targets(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node: CFGNode,
) -> tuple[int, ...] | None:
    """Resolve one dispatch by replaying its unique VEX path from entry.

    Replaying a unique path lets angr retain predecessor constraints without
    turning extraction into unrestricted symbolic execution. The caller first
    proves that the final block has a conditional arithmetic PC update.
    """

    path = _unique_entry_path(graph, bounds, node)
    if path is None:
        return None
    if any(_node_vex(path_node) is None for path_node in path):
        return None

    try:
        state = project.factory.blank_state(
            addr=path[0].addr,
            add_options={angr_options.SYMBOL_FILL_UNCONSTRAINED_REGISTERS},
        )
        for current, successor in zip(path, path[1:], strict=False):
            successors = project.factory.successors(
                state,
                addr=current.addr,
                size=current.size,
            )
            matching = [
                next_state
                for next_state in successors.flat_successors
                if next_state.addr == successor.addr
            ]
            if len(matching) != 1:
                return None
            state = matching[0]

        successors = project.factory.successors(
            state,
            addr=node.addr,
            size=node.size,
        )
    except Exception:
        return None

    targets = {
        successor.addr
        for successor in successors.flat_successors
        if isinstance(successor.addr, int)
        and successor.addr != node.addr + node.size
        and is_direct_target_valid(bounds, successor.addr)
    }
    if not targets or len(targets) > MAX_STATIC_JUMPTABLE_ENTRIES:
        return None
    return tuple(sorted(targets))


def _unconditional_pc_arithmetic_dispatch_shape(vex) -> bool:
    """Return whether one computed-PC update has a static arithmetic shape.

    The final VEX ``next`` may read a register directly or set an architecture
    mode bit on it. The register itself must have been assigned from a static
    base plus a shifted register in the same block. This deliberately excludes
    values loaded from memory, even when a path solver could otherwise choose
    a finite set of concrete targets.
    """

    if vex.jumpkind != "Ijk_Boring":
        return False

    definitions = _vex_tmp_definitions(vex)
    next_expr = _resolve_vex_expr(vex.next, definitions)
    target_key = _vex_get_key(next_expr, definitions, vex)
    if target_key is None and isinstance(next_expr, pyvex.expr.Binop):
        if not next_expr.op.startswith("Iop_Or"):
            return False
        register_terms = [
            _vex_get_key(term, definitions, vex) for term in next_expr.args
        ]
        constant_terms = [
            _vex_const_value(term, definitions) for term in next_expr.args
        ]
        keys = [key for key in register_terms if key is not None]
        constants = [value for value in constant_terms if value is not None]
        if len(keys) != 1 or constants != [1]:
            return False
        target_key = keys[0]
    if target_key is None:
        return False

    for index in range(len(vex.statements) - 1, -1, -1):
        statement = vex.statements[index]
        if (
            not isinstance(statement, pyvex.stmt.Put)
            or statement.offset != target_key[0]
        ):
            continue
        assignment = _resolve_vex_expr(statement.data, definitions)
        if not isinstance(assignment, pyvex.expr.Binop) or not assignment.op.startswith(
            "Iop_Add"
        ):
            return False

        terms = _vex_add_terms(assignment, definitions)
        if terms is None or len(terms) != 2:
            return False
        shifted_terms = []
        base_terms = []
        for term in terms:
            resolved = _resolve_vex_expr(term, definitions)
            if isinstance(resolved, pyvex.expr.Binop) and resolved.op.startswith(
                "Iop_Shl"
            ):
                shifted_terms.append(resolved)
            else:
                base_terms.append(term)
        if len(shifted_terms) != 1 or len(base_terms) != 1:
            return False
        shifted = shifted_terms[0]
        if (
            _vex_get_key(shifted.args[0], definitions, vex) is None
            or _vex_const_value(shifted.args[1], definitions) is None
        ):
            return False

        base = base_terms[0]
        if _vex_const_value(base, definitions) is not None:
            return True
        base_key = _vex_get_key(base, definitions, vex)
        if base_key is None:
            return False
        for preceding in reversed(vex.statements[:index]):
            if (
                isinstance(preceding, pyvex.stmt.Put)
                and preceding.offset == base_key[0]
            ):
                return _vex_const_value(preceding.data, definitions) is not None
        return False
    return False


def unconditional_arithmetic_pc_dispatch_targets(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node: CFGNode,
) -> tuple[int, ...] | None:
    """Return finite targets for a uniquely constrained arithmetic PC update."""

    try:
        vex = project.factory.block(
            node.addr,
            size=node.size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception:
        return None
    if not _unconditional_pc_arithmetic_dispatch_shape(vex):
        return None

    targets = _path_constrained_pc_targets(project, graph, bounds, node)
    if targets is None or not all(
        is_direct_target_valid(bounds, target) for target in targets
    ):
        return None
    return targets


def conditional_pc_dispatch_targets(
    project: Project,
    bounds: FunctionBounds,
    node: CFGNode,
    graph: CFGGraph | None = None,
) -> tuple[tuple[int, ...] | None, str | None]:
    """Return finite targets of one VEX-proven conditional PC dispatcher.

    It supports either an absolute-address table load or an inline arithmetic
    dispatch. The preferred proof is a same-block unsigned guard that bounds
    the index. For arithmetic dispatches only, a unique predecessor path can
    instead provide the necessary constraints through VEX replay. It never
    infers targets from arbitrary code addresses, so callers can safely feed
    its output into a leader worklist.
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
        return targets, None

    arithmetic = _conditional_pc_arithmetic_dispatch(vex, definitions, taken)
    if arithmetic is None:
        return None, "no_dispatch_shape"
    index_key, base_addr, shift = arithmetic
    index_values = _vex_guarded_index_values(condition, index_key, definitions, vex)
    if index_values is not None:
        mask = (1 << project.arch.bits) - 1
        targets = tuple(
            sorted({(base_addr + (index << shift)) & mask for index in index_values})
        )
    else:
        targets = (
            _path_constrained_pc_targets(project, graph, bounds, node)
            if graph is not None
            else None
        )
        if targets is None:
            return None, "unbounded_index"
    if not all(is_direct_target_valid(bounds, target) for target in targets):
        return None, "invalid_target"
    return targets, None


def _constant_register_from_predecessors(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    register_offset: int,
) -> int | None:
    """Return one unambiguous constant register definition reaching ``node``."""

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

        tmp_definitions = _vex_tmp_definitions(vex)
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


def _unique_static_register_value(
    graph: CFGGraph,
    bounds: FunctionBounds,
    register_offset: int,
) -> int | None:
    """Return one in-bounds VEX-proven static value assigned to a register."""

    values: set[int] = set()
    for node in _iter_graph_bound_nodes(graph, bounds):
        if not _node_is_materialized_cfg_node(node):
            continue
        try:
            vex = node.block.vex
        except Exception:
            continue
        definitions = _vex_tmp_definitions(vex)
        for stmt in vex.statements:
            if not isinstance(stmt, pyvex.stmt.Put) or stmt.offset != register_offset:
                continue
            value = _vex_static_int(stmt.data, definitions)
            if value is not None:
                values.add(value)

    return next(iter(values)) if len(values) == 1 else None


def _x86_pc_thunk_predecessors(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    table: StaticJumpTable,
) -> tuple[CFGNode, ...]:
    """Return matching x86 PC-thunk calls whose fake return enters ``node``."""

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
        definitions = _vex_tmp_definitions(vex)
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
    """Return an x86 PIC table base register value proven by a PC thunk call."""

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
        definitions = _vex_tmp_definitions(vex)
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
                successor_defs = _vex_tmp_definitions(successor_vex)
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
    """Return a table length proven by a guard immediately before an x86 thunk."""

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


def _in_function_jump_table_entry_count(
    project: Project,
    bounds: FunctionBounds,
    table: StaticJumpTable,
    base_addr: int,
) -> int | None:
    """Estimate a table extent from contiguous in-function target entries."""

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


def _mips_static_value(
    project: Project,
    vex,
    expr,
    definitions: dict[int, Any],
    global_pointer: int,
) -> int | None:
    """Evaluate a MIPS PIC data expression rooted in the function's ``$gp``."""

    expr = _resolve_vex_expr(expr, definitions)
    if expr is None:
        return None
    value = _vex_const_value(expr, definitions)
    if value is not None:
        return value

    try:
        gp_offset = project.arch.registers["gp"][0]
    except KeyError:
        return None
    if _vex_get_key(expr, definitions, vex) == (gp_offset, project.arch.bits):
        return global_pointer
    if any(
        isinstance(statement, pyvex.stmt.Put)
        and statement.offset == gp_offset
        and _resolve_vex_expr(statement.data, definitions) == expr
        for statement in vex.statements
    ):
        return global_pointer

    if isinstance(expr, pyvex.expr.Binop) and expr.op.startswith("Iop_Add"):
        left = _mips_static_value(
            project, vex, expr.args[0], definitions, global_pointer
        )
        right = _mips_static_value(
            project, vex, expr.args[1], definitions, global_pointer
        )
        if left is None or right is None:
            return None
        return (left + right) & ((1 << project.arch.bits) - 1)

    if not isinstance(expr, pyvex.expr.Load):
        return None
    addr = _mips_static_value(project, vex, expr.addr, definitions, global_pointer)
    entry_size = expr.result_size(vex.tyenv) // 8
    if addr is None or entry_size not in {1, 2, 4, 8}:
        return None
    try:
        raw = project.loader.memory.load(addr, entry_size)
    except Exception:
        return None
    byteorder = "little" if expr.end == "Iend_LE" else "big"
    return int.from_bytes(raw, byteorder=byteorder)


def _mips_static_register_from_immediate_predecessors(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    expr,
    global_pointer: int,
) -> int | None:
    """Resolve one dispatcher register when every predecessor writes it."""

    if not isinstance(expr, pyvex.expr.Get):
        return None

    values: set[int] = set()
    predecessors = tuple(graph.predecessors(node))
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
        definitions = _vex_tmp_definitions(vex)
        for statement in reversed(vex.statements):
            if (
                not isinstance(statement, pyvex.stmt.Put)
                or statement.offset != expr.offset
            ):
                continue
            value = _mips_static_value(
                project, vex, statement.data, definitions, global_pointer
            )
            if value is None:
                return None
            values.add(value)
            break
        else:
            return None
    return next(iter(values)) if len(values) == 1 else None


def _mips_static_register_at_node(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    expr,
    global_pointer: int,
) -> int | None:
    """Return a register's must-constant value at one in-function CFG node."""

    if not isinstance(expr, pyvex.expr.Get):
        return None

    unknown = object()
    unreached = object()
    nodes = tuple(
        candidate
        for candidate in graph.nodes()
        if _node_is_materialized_cfg_node(candidate)
        and _node_intersects_bounds(candidate, bounds)
    )
    if node not in nodes:
        return None

    def merge(values: list[object]) -> object:
        if not values:
            return unreached
        if unknown in values or len(set(values)) != 1:
            return unknown
        return values[0]

    def transfer(candidate, incoming: object) -> object:
        if incoming is unreached:
            return unreached
        vex = _node_vex(candidate)
        if vex is None:
            return unknown
        definitions = _vex_tmp_definitions(vex)
        value = incoming
        for statement in vex.statements:
            if (
                not isinstance(statement, pyvex.stmt.Put)
                or statement.offset != expr.offset
            ):
                continue
            static_value = _mips_static_value(
                project, vex, statement.data, definitions, global_pointer
            )
            value = static_value if static_value is not None else unknown
        return value

    incoming_values = {candidate: unreached for candidate in nodes}
    outgoing_values = {candidate: unreached for candidate in nodes}
    while True:
        changed = False
        for candidate in nodes:
            values = []
            if candidate.addr == bounds.addr:
                values.append(unknown)
            for predecessor in graph.predecessors(candidate):
                if predecessor not in outgoing_values:
                    values.append(unknown)
                    continue
                predecessor_value = outgoing_values[predecessor]
                if predecessor_value is not unreached:
                    values.append(predecessor_value)
            incoming = merge(values)
            outgoing = transfer(candidate, incoming)
            if (
                incoming_values[candidate] != incoming
                or outgoing_values[candidate] != outgoing
            ):
                incoming_values[candidate] = incoming
                outgoing_values[candidate] = outgoing
                changed = True
        if not changed:
            break

    value = incoming_values[node]
    return value if isinstance(value, int) else None


def _mips_static_register_from_predecessors(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    expr,
    global_pointer: int,
) -> tuple[int | None, bool]:
    """Resolve a dispatcher register and report whether propagation was needed."""

    immediate_value = _mips_static_register_from_immediate_predecessors(
        project, graph, bounds, node, expr, global_pointer
    )
    if immediate_value is not None:
        return immediate_value, False
    return (
        _mips_static_register_at_node(
            project, graph, bounds, node, expr, global_pointer
        ),
        True,
    )


def _mips_inverted_unsigned_guard(guard, definitions: dict[int, Any]):
    """Unwrap ``CmpEQ(1UtoN(Cmp*u(...)), 0)`` into its unsigned compare."""

    guard = _resolve_vex_expr(guard, definitions)
    if not isinstance(guard, pyvex.expr.Binop) or not guard.op.startswith("Iop_CmpEQ"):
        return None
    left, right = guard.args
    boolean, zero = (
        (left, right) if _vex_const_value(right, definitions) == 0 else (right, left)
    )
    if _vex_const_value(zero, definitions) != 0:
        return None
    boolean = _resolve_vex_expr(boolean, definitions)
    return _mips_unsigned_comparison_from_boolean(boolean, definitions)


def _mips_unsigned_comparison_from_boolean(boolean, definitions: dict[int, Any]):
    """Return the unsigned comparison stored in one widened boolean value."""

    conversion = _vex_width_conversion(boolean)
    if conversion is None or conversion[0] != 1 or conversion[2] != "U":
        return None
    comparison = _resolve_vex_expr(boolean.args[0], definitions)
    if not isinstance(comparison, pyvex.expr.Binop) or not comparison.op.endswith("U"):
        return None
    return comparison


def _mips_inverted_unsigned_guard_from_fallthrough_predecessor(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    guard,
    definitions: dict[int, Any],
):
    """Resolve a zero test through one predecessor delay-slot register write."""

    comparison = _mips_inverted_unsigned_guard(guard, definitions)
    if comparison is not None:
        return comparison

    guard = _resolve_vex_expr(guard, definitions)
    if not isinstance(guard, pyvex.expr.Binop) or not guard.op.startswith("Iop_CmpEQ"):
        return None
    left, right = guard.args
    value, zero = (
        (left, right) if _vex_const_value(right, definitions) == 0 else (right, left)
    )
    if _vex_const_value(zero, definitions) != 0:
        return None
    value = _resolve_vex_expr(value, definitions)
    if not isinstance(value, pyvex.expr.Get):
        return None

    predecessors = tuple(graph.predecessors(node))
    if len(predecessors) != 1:
        return None
    predecessor = predecessors[0]
    if not (
        _node_is_materialized_cfg_node(predecessor)
        and _node_intersects_bounds(predecessor, bounds)
    ):
        return None
    predecessor_vex = _node_vex(predecessor)
    if (
        predecessor_vex is None
        or _vex_const_value(predecessor_vex.next, _vex_tmp_definitions(predecessor_vex))
        != node.addr
    ):
        return None
    predecessor_definitions = _vex_tmp_definitions(predecessor_vex)
    for statement in reversed(predecessor_vex.statements):
        if isinstance(statement, pyvex.stmt.Put) and statement.offset == value.offset:
            return _mips_unsigned_comparison_from_boolean(
                _resolve_vex_expr(statement.data, predecessor_definitions),
                predecessor_definitions,
            )
    return None


def _mips_guarded_index_entry_count(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    index_key: tuple[int, int],
) -> int | None:
    """Return a predecessor-proven table count for one MIPS selector."""

    bounds_found: set[int] = set()
    for predecessor in graph.predecessors(node):
        if not _node_is_materialized_cfg_node(predecessor):
            continue
        if not _node_intersects_bounds(predecessor, bounds):
            continue
        vex = _node_vex(predecessor)
        if (
            vex is None
            or _vex_const_value(vex.next, _vex_tmp_definitions(vex)) != node.addr
        ):
            continue
        definitions = _vex_tmp_definitions(vex)
        upper_bounds = {
            upper_bound
            for exit_index, statement in enumerate(vex.statements)
            if isinstance(statement, pyvex.stmt.Exit)
            if (
                comparison
                := _mips_inverted_unsigned_guard_from_fallthrough_predecessor(
                    graph, bounds, predecessor, statement.guard, definitions
                )
            )
            is not None
            if (
                upper_bound := _vex_guard_upper_bound(
                    comparison,
                    index_key,
                    definitions,
                    vex,
                    vex.statements[:exit_index],
                    index_on_left=True,
                )
            )
            is not None
        }
        if len(upper_bounds) == 1:
            bounds_found.update(upper_bounds)

    if len(bounds_found) != 1:
        return None
    entry_count = next(iter(bounds_found)) + 1
    return entry_count if entry_count <= MAX_STATIC_JUMPTABLE_ENTRIES else None


def _mips_scaled_index_entry_count(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    scaled_index_key: tuple[int, int],
    entry_size: int,
) -> int | None:
    """Return a predecessor-proven count for a MIPS byte-scaled selector."""

    expression_entry_counts: set[int] = set()
    bounded_expression_predecessors = 0
    relevant_predecessors = 0
    index_keys: set[tuple[int, int]] = set()
    for predecessor in graph.predecessors(node):
        if not (
            _node_is_materialized_cfg_node(predecessor)
            and _node_intersects_bounds(predecessor, bounds)
        ):
            continue
        vex = _node_vex(predecessor)
        if (
            vex is None
            or _vex_const_value(vex.next, _vex_tmp_definitions(vex)) != node.addr
        ):
            continue
        relevant_predecessors += 1
        definitions = _vex_tmp_definitions(vex)
        predecessor_expression_counts: set[int] = set()
        for statement_index, statement in enumerate(vex.statements):
            if not isinstance(statement, pyvex.stmt.Put):
                continue
            if statement.offset != scaled_index_key[0]:
                continue
            scaled = _resolve_vex_expr(statement.data, definitions)
            if not isinstance(scaled, pyvex.expr.Binop) or not scaled.op.startswith(
                "Iop_Shl"
            ):
                continue
            shift = _vex_const_value(scaled.args[1], definitions)
            if shift is None or 1 << shift != entry_size:
                continue
            source = _resolve_vex_expr(scaled.args[0], definitions)
            expression_counts = {
                len(values)
                for statement in vex.statements
                if isinstance(statement, pyvex.stmt.Exit)
                if (
                    comparison
                    := _mips_inverted_unsigned_guard_from_fallthrough_predecessor(
                        graph, bounds, predecessor, statement.guard, definitions
                    )
                )
                is not None
                if (
                    values := _vex_guarded_expression_values(
                        comparison, source, definitions
                    )
                )
                is not None
            }
            if len(expression_counts) == 1:
                predecessor_expression_counts.update(expression_counts)
            index_key = _vex_get_key(source, definitions, vex)
            if index_key is None:
                for previous in reversed(vex.statements[:statement_index]):
                    if not isinstance(previous, pyvex.stmt.Put):
                        continue
                    value = _resolve_vex_expr(previous.data, definitions)
                    if value is source or value == source:
                        index_key = (previous.offset, value.result_size(vex.tyenv))
                        break
            if index_key is not None:
                index_keys.add(index_key)
        if len(predecessor_expression_counts) == 1:
            expression_entry_counts.update(predecessor_expression_counts)
            bounded_expression_predecessors += 1

    if (
        relevant_predecessors
        and bounded_expression_predecessors == relevant_predecessors
        and len(expression_entry_counts) == 1
    ):
        return next(iter(expression_entry_counts))
    if expression_entry_counts:
        return None
    if len(index_keys) != 1:
        return None
    return _mips_guarded_index_entry_count(graph, bounds, node, next(iter(index_keys)))


def _mips_inline_scaled_index_key(
    expr, definitions: dict[int, Any], vex, entry_size: int
) -> tuple[int, int] | None:
    """Return the unscaled register from one exact in-dispatcher shift."""

    expr = _resolve_vex_expr(expr, definitions)
    if not isinstance(expr, pyvex.expr.Binop) or not expr.op.startswith("Iop_Shl"):
        return None
    shift = _vex_const_value(expr.args[1], definitions)
    if shift is None or 1 << shift != entry_size:
        return None
    return _vex_get_key(expr.args[0], definitions, vex)


def plan_mips_pic_relative_jump_table(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    *,
    allow_predecessor_static_base: bool = False,
) -> tuple[StaticJumpTablePlan | None, str | None]:
    """Plan a MIPS ``$gp``-relative table of offsets to branch targets.

    MIPS PIC dispatchers commonly load a table base through the GOT, add a
    byte-scaled selector, read a relative entry, then add ``$gp`` before
    ``jr``. The ordinary matcher intentionally rejects that two-base form.
    Accept it only when the function entry proves ``$gp`` and the immediate
    predecessor proves the unscaled selector's finite range. Extraction may
    additionally opt in to a table base established before that predecessor,
    including a delay slot, when every in-function path agrees on its value.
    """

    if not project.arch.name.startswith("MIPS"):
        return None, "no_table_shape"
    vex = _node_vex(node)
    global_pointer = _mips_entry_global_pointer(project, bounds)
    if vex is None or global_pointer is None or vex.jumpkind != "Ijk_Boring":
        return None, "no_table_shape"
    try:
        gp_offset = project.arch.registers["gp"][0]
    except KeyError:
        return None, "no_table_shape"

    definitions = _vex_tmp_definitions(vex)
    next_expr = _resolve_vex_expr(vex.next, definitions)
    if not isinstance(next_expr, pyvex.expr.Binop) or not next_expr.op.startswith(
        "Iop_Add"
    ):
        return None, "no_table_shape"

    for entry_expr, target_base in (
        (next_expr.args[0], next_expr.args[1]),
        (next_expr.args[1], next_expr.args[0]),
    ):
        if _vex_get_key(target_base, definitions, vex) != (
            gp_offset,
            project.arch.bits,
        ):
            continue
        normalized_entry = _vex_normalized_table_entry_load(entry_expr, definitions)
        if normalized_entry is None:
            continue
        entry, signed_entries = normalized_entry
        entry_size = entry.result_size(vex.tyenv) // 8
        if entry_size not in {1, 2, 4, 8}:
            continue

        address_terms = _vex_add_terms(entry.addr, definitions)
        if address_terms is None:
            continue
        table_addr = 0
        scaled_index_key = None
        inline_index_key = None
        used_propagated_static_base = False
        for term in address_terms:
            value = _mips_static_value(project, vex, term, definitions, global_pointer)
            if value is not None:
                table_addr += value
                continue
            if allow_predecessor_static_base:
                key = _mips_inline_scaled_index_key(term, definitions, vex, entry_size)
                if key is not None:
                    if scaled_index_key is not None or inline_index_key is not None:
                        break
                    inline_index_key = key
                    continue
                value, propagated = _mips_static_register_from_predecessors(
                    project, graph, bounds, node, term, global_pointer
                )
                if value is not None:
                    table_addr += value
                    used_propagated_static_base |= propagated
                    continue
            key = _vex_get_key(term, definitions, vex)
            if key is not None and scaled_index_key is None:
                scaled_index_key = key
                continue
            if allow_predecessor_static_base:
                break
            key = _mips_inline_scaled_index_key(term, definitions, vex, entry_size)
            if key is None or scaled_index_key is not None:
                break
            inline_index_key = key
        else:
            if scaled_index_key is None and inline_index_key is None:
                continue
            if scaled_index_key is not None:
                entry_count = _mips_scaled_index_entry_count(
                    graph, bounds, node, scaled_index_key, entry_size
                )
            else:
                assert inline_index_key is not None
                entry_count = _mips_guarded_index_entry_count(
                    graph, bounds, node, inline_index_key
                )
            if entry_count is None:
                if used_propagated_static_base:
                    # The opt-in base proof must not change a previously
                    # shape-free dispatcher unless it can resolve the table.
                    return None, "no_table_shape"
                return None, "unbounded_index"
            mask = (1 << project.arch.bits) - 1
            table = StaticJumpTable(
                base_register_offset=None,
                base_bits=project.arch.bits,
                table_displacement=(table_addr - global_pointer) & mask,
                index_register_offset=None,
                index_bits=None,
                entry_size=entry_size,
                endness=entry.end,
                signed_entries=signed_entries,
                static_base_addr=global_pointer,
            )
            return StaticJumpTablePlan(
                table, global_pointer, tuple(range(entry_count))
            ), None

    return None, "no_table_shape"


def plan_static_jump_table(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node: CFGNode,
    *,
    allow_inline_index_values: bool = False,
    allow_masked_index_values: bool = False,
    allow_guarded_loads: bool = False,
    allow_static_bases: bool = False,
    allow_guarded_expression_indices: bool = False,
) -> tuple[StaticJumpTablePlan | None, str | None]:
    """Return one fully proven static-table read plan for an indirect branch.

    This deliberately owns only architecture-neutral VEX recognition and the
    evidence needed to read a finite table. Callers decide how proven targets
    are materialized: CFGFast fixup queues repairs, while independent
    extraction adds new block leaders before its final graph materialization.
    """

    vex = _node_vex(node)
    if vex is None:
        return None, "no_vex"

    table = (
        _vex_relative_jump_table(
            vex,
            allow_inline_index_values=allow_inline_index_values,
            allow_masked_index_values=allow_masked_index_values,
            allow_guarded_expression_index=allow_guarded_expression_indices,
        )
        or _vex_scaled_relative_jump_table(
            vex,
            allow_masked_index_values=allow_masked_index_values,
        )
        or _vex_direct_jump_table(
            vex,
            allow_inline_index_values=allow_inline_index_values,
            allow_masked_index_values=allow_masked_index_values,
            allow_guarded_loads=allow_guarded_loads,
            allow_static_base=allow_static_bases,
            allow_guarded_expression_index=allow_guarded_expression_indices,
        )
    )
    if table is None:
        # A table index naturally has the architecture's full register width
        # on 64-bit targets. Recognition remains safe because table reads
        # still require a separate finite range proof.
        table = (
            _vex_relative_jump_table(
                vex,
                allow_full_width_index=True,
                allow_inline_index_values=allow_inline_index_values,
                allow_masked_index_values=allow_masked_index_values,
                allow_guarded_expression_index=allow_guarded_expression_indices,
            )
            or _vex_scaled_relative_jump_table(
                vex,
                allow_full_width_index=True,
                allow_masked_index_values=allow_masked_index_values,
            )
            or _vex_direct_jump_table(
                vex,
                allow_full_width_index=True,
                allow_inline_index_values=allow_inline_index_values,
                allow_masked_index_values=allow_masked_index_values,
                allow_guarded_loads=allow_guarded_loads,
                allow_static_base=allow_static_bases,
                allow_guarded_expression_index=allow_guarded_expression_indices,
            )
        )

    pic_base_addr = None
    if project.arch.name == "X86" and project.arch.bits == 32 and table is not None:
        pic_base_addr = _x86_pc_thunk_base_addr(project, graph, bounds, node, table)
    if table is None or table.base_bits != project.arch.bits:
        return None, "no_table_shape"

    entry_indices = table.index_values
    if table.index_affine_difference is not None:
        entry_indices = _guarded_affine_difference_index_values(
            graph, bounds, node, table
        )
        if entry_indices is None:
            return None, "no_table_shape"
    entry_count = _guarded_jump_table_entry_count(graph, bounds, node, table)
    if table.index_expression is not None and entry_count is None:
        # The optional expression matcher is only safe when its matching
        # predecessor guard proves a finite table. Otherwise let extraction
        # use its ordinary component-recovery path.
        return None, "no_table_shape"
    if entry_indices is None:
        if entry_count is None and pic_base_addr is not None:
            entry_count = _x86_pc_thunk_guarded_entry_count(
                project, graph, bounds, node, table
            )
        if entry_count is not None:
            entry_indices = tuple(range(entry_count))
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
    if base_addr is None and base_register_offset is not None:
        # Disconnected dispatchers can lack a full predecessor path. Accept a
        # base only if every bounded VEX definition agrees on its value.
        base_addr = _unique_static_register_value(graph, bounds, base_register_offset)
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
    """Apply the architecture-width arithmetic used by a relative table jump."""

    if not table.entries_are_relative:
        return entry

    mask = (1 << table.base_bits) - 1
    target = (base_addr + table.target_displacement + entry * table.target_scale) & mask
    return (target | table.target_or_mask) & mask


def _jump_table_addr(base_addr: int, table: StaticJumpTable) -> int:
    """Apply the table-address arithmetic in the architecture's address width."""

    return (base_addr + table.table_displacement) & ((1 << table.base_bits) - 1)


def _read_static_jump_table_targets(
    project: Project,
    table: StaticJumpTable,
    base_addr: int,
    entry_indices: tuple[int, ...],
) -> tuple[int, ...] | None:
    """Read targets from one VEX-proven table, or None when memory is unreadable."""

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


def _vex_key_reads_memory(key: tuple[Any, ...]) -> bool:
    """Return whether one structural VEX key contains a memory read."""

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


def _vex_scaled_register_target(
    vex,
) -> tuple[tuple[Any, ...], tuple[int, int], int, int] | None:
    """Describe one conditional ``base + (register << shift)`` VEX target."""

    definitions = _vex_tmp_definitions(vex)
    next_expr = _resolve_vex_expr(vex.next, definitions)
    if not isinstance(next_expr, pyvex.expr.ITE):
        return None

    target_expr = _resolve_vex_expr(next_expr.iftrue, definitions)
    if not isinstance(target_expr, pyvex.expr.Binop) or not target_expr.op.startswith(
        "Iop_Add"
    ):
        return None

    base_addr = None
    index_key = None
    shift = None
    for term in _vex_add_terms(target_expr, definitions) or ():
        value = _vex_const_value(term, definitions)
        if value is not None and base_addr is None:
            base_addr = value
            continue
        term = _resolve_vex_expr(term, definitions)
        if not isinstance(term, pyvex.expr.Binop) or not term.op.startswith("Iop_Shl"):
            return None
        candidate_shift = _vex_const_value(term.args[1], definitions)
        candidate_index = _vex_get_key(term.args[0], definitions, vex)
        if candidate_shift is None or candidate_index is None or index_key is not None:
            return None
        index_key = candidate_index
        shift = candidate_shift

    condition = _vex_expr_key(next_expr.cond, definitions)
    if condition is None or base_addr is None or index_key is None or shift is None:
        return None
    return condition, index_key, base_addr, shift


def _vex_conditionally_scaled_register(
    vex,
    register_key: tuple[int, int],
    condition_key: tuple[Any, ...],
) -> int | None:
    """Return a conditionally assigned register multiplier, if VEX proves one."""

    definitions = _vex_tmp_definitions(vex)
    register_expr = ("get", *register_key)
    for statement in reversed(vex.statements):
        if (
            not isinstance(statement, pyvex.stmt.Put)
            or statement.offset != register_key[0]
        ):
            continue
        assignment = _resolve_vex_expr(statement.data, definitions)
        if not isinstance(assignment, pyvex.expr.ITE):
            return None
        if _vex_expr_key(assignment.cond, definitions) != condition_key:
            return None

        unchanged = _vex_expr_key(assignment.iffalse, definitions)
        scaled_expr = _resolve_vex_expr(assignment.iftrue, definitions)
        if unchanged != register_expr or not isinstance(scaled_expr, pyvex.expr.Binop):
            return None
        if not scaled_expr.op.startswith("Iop_Add"):
            return None

        terms = _vex_add_terms(scaled_expr, definitions)
        if terms is None or len(terms) != 2:
            return None
        direct_reads = [
            _vex_expr_key(term, definitions) == register_expr for term in terms
        ]
        shifted_terms = [
            _resolve_vex_expr(term, definitions)
            for term, is_direct_read in zip(terms, direct_reads, strict=True)
            if not is_direct_read
        ]
        if direct_reads.count(True) != 1 or len(shifted_terms) != 1:
            return None

        shifted = shifted_terms[0]
        if not isinstance(shifted, pyvex.expr.Binop) or not shifted.op.startswith(
            "Iop_Shl"
        ):
            return None
        if _vex_expr_key(shifted.args[0], definitions) != register_expr:
            return None
        shift = _vex_const_value(shifted.args[1], definitions)
        return 1 + (1 << shift) if shift is not None else None
    return None


def _immediate_linear_predecessor(
    graph: CFGGraph, bounds: FunctionBounds, node
) -> CFGNode | None:
    """Return one sole fallthrough predecessor ending immediately before ``node``."""

    predecessors = [
        predecessor
        for predecessor in graph.predecessors(node)
        if _node_is_materialized_cfg_node(predecessor)
        and _node_intersects_bounds(predecessor, bounds)
        and predecessor.addr + predecessor.size == node.addr
        and tuple(graph.successors(predecessor)) == (node,)
    ]
    return predecessors[0] if len(predecessors) == 1 else None


def arithmetic_pc_dispatch_targets(
    project: Project, graph: CFGGraph, bounds: FunctionBounds, node
) -> tuple[int, ...] | None:
    """Return a proven target subset for one arithmetic computed-PC dispatch.

    This covers a VEX-level pattern where a conditional indirect branch writes
    ``base + (index << shift)`` to the program counter, and a sole linear
    predecessor conditionally scales that same index under the exact same VEX
    predicate. CFGFast may conservatively fan this out to every instruction
    boundary. We retain only the stride-aligned candidates it already found;
    the function never invents targets or applies architecture-specific
    mnemonic rules.
    """

    vex = _node_vex(node)
    if vex is None:
        return None
    dispatch = _vex_scaled_register_target(vex)
    if dispatch is None:
        return None
    condition_key, index_key, base_addr, shift = dispatch

    predecessor = _immediate_linear_predecessor(graph, bounds, node)
    # A neutral instruction may separate the scale and computed branch. Walk
    # only unique linear predecessors, so no unproven control-flow path is
    # included in the proof.
    for _ in range(3):
        if predecessor is None:
            return None
        predecessor_vex = _node_vex(predecessor)
        if predecessor_vex is not None:
            multiplier = _vex_conditionally_scaled_register(
                predecessor_vex, index_key, condition_key
            )
            if multiplier is None:
                # CFGFast may group the scale with the instruction that sets
                # condition flags, allowing VEX to simplify its predicate.
                # Re-lift only the last instruction to compare the preserved
                # condition-code form used by the computed-PC branch.
                last_insn = DecodedNode.from_node(predecessor).last
                if last_insn is not None:
                    single_insn_vex = lift_instruction_vex(project, last_insn)
                    if single_insn_vex is not None:
                        multiplier = _vex_conditionally_scaled_register(
                            single_insn_vex, index_key, condition_key
                        )
            if multiplier is not None:
                stride = (1 << shift) * multiplier
                break
        predecessor = _immediate_linear_predecessor(graph, bounds, predecessor)
    else:
        return None

    direct_exit_targets = {
        statement.dst.value
        for statement in vex.statements
        if isinstance(statement, pyvex.stmt.Exit)
        and isinstance(getattr(statement.dst, "value", None), int)
    }
    candidates = {
        successor.addr
        for successor in graph.successors(node)
        if _node_is_materialized_cfg_node(successor)
        and _node_intersects_bounds(successor, bounds)
        and successor.addr not in direct_exit_targets
    }
    if base_addr not in candidates:
        return None
    upper_bound = max(candidates)
    expected_targets = tuple(range(base_addr, upper_bound + 1, stride))
    if not expected_targets or not set(expected_targets).issubset(candidates):
        return None
    return expected_targets


def _seed_node_expected_successors(node) -> tuple[tuple[int, ...], int | None]:
    """
    Return the local direct targets and fallthrough encoded in one seed node.

    CFGFast may already have stitched a malformed region incorrectly, so when we
    reconnect a preserved predecessor into repaired blocks we prefer the
    predecessor's own lifted block semantics over the old graph edges.
    """

    try:
        decoded = DecodedNode.from_node(node)
    except Exception:
        return (), None
    if decoded.insns is None:
        return (), None

    try:
        vex = node.block.vex
    except Exception:
        return (), None
    if vex.jumpkind == "Ijk_NoDecode":
        return (), None

    if decoded.is_empty:
        return (), None

    last_insn = decoded.last
    if last_insn is None:
        return (), None

    last_semantic = InsnSemantics(last_insn)
    if not last_semantic.is_control_transfer():
        return (), None

    last_addrs = {last_insn.address}

    direct_targets: list[int] = []
    for ins_addr, _, stmt in vex.exit_statements:
        if ins_addr not in last_addrs:
            continue
        target = getattr(stmt.dst, "value", None)
        if isinstance(target, int):
            direct_targets.append(target)

    fallthrough_addr = None
    if isinstance(vex.next, pyvex.expr.Const):
        target = vex.next.con.value
        if isinstance(target, int):
            fallthrough_addr = target

    return tuple(direct_targets), fallthrough_addr


def _seed_graph_direct_targets(graph: CFGGraph, node) -> tuple[int, ...]:
    """
    Return direct branch targets that are explicitly present in the seed graph.

    For unrepaired seed nodes we only want to preserve leaders that are backed
    by a concrete branch edge already materialized in CFGFast. This is narrower
    than trusting the node's fallthrough layout and avoids swallowing real
    branch-target leaders such as `0x806a85a` in `__strcasecmp_l_sse4_2`.
    """

    try:
        decoded = DecodedNode.from_node(node)
    except Exception:
        return ()

    insns = decoded.insns
    if not insns:
        return ()

    transfer_index = control_transfer_index(
        node.block.arch.name, list(insns), strict=False
    )
    if transfer_index is None:
        return ()

    transfer = InsnSemantics(insns[transfer_index])
    if not transfer.is_jump():
        return ()

    target = transfer.direct_target()
    if not isinstance(target, int):
        return ()

    successor_addrs = {
        succ.addr for succ in graph.successors(node) if hasattr(succ, "addr")
    }
    if target not in successor_addrs:
        return ()

    return (target,)


def _resolve_direct_branch_target(
    project: Project,
    bounds: FunctionBounds,
    capstone_target: int | None,
    vex_exit_targets: Iterable[int],
) -> int | None:
    """Choose a direct branch target from Capstone with a bounded VEX fallback.

    Capstone normally supplies the concrete target, but some architectures
    expose a PC-relative displacement instead. When that value is not mapped
    in the loaded binary and VEX supplies exactly one mapped exit, the VEX
    target is the unambiguous absolute address. This includes a direct branch
    outside the current symbol bounds, which must become an external target
    rather than repeatedly repairing an impossible relative displacement.
    Mapped Capstone targets may point outside this function, so retain them:
    malformed CFGFast VEX metadata can still be stale.
    """

    if capstone_target is not None and project.loader.find_object_containing(
        capstone_target
    ):
        return capstone_target

    mapped_exits = {
        target
        for target in vex_exit_targets
        if project.loader.find_object_containing(target)
    }
    if len(mapped_exits) == 1:
        return next(iter(mapped_exits))

    return capstone_target


def _analyze_jump_successors(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> JumpSuccessorAnalysis | None:
    """
    Return expected vs present successors for one decoded jump block.

    VEX helps recognize conditional control flow, while Capstone remains
    authoritative for the direct target because malformed CFGFast nodes can
    expose stale VEX exit addresses. `InsnSemantics` selects the final immediate
    operand so compare-and-branch instructions do not use a condition value,
    such as S390's `-1`, as the branch address.
    """

    from .anomalies import _can_decode_block_at, node_has_decoding_coverage_mismatch

    try:
        decoded = DecodedNode.from_node(node)
    except Exception:
        return None

    if decoded.is_empty or node_has_decoding_coverage_mismatch(node):
        return None

    insns = decoded.insns
    if not insns:
        return None

    # On delayed-branch architectures, the final decoded instruction is the
    # delay slot. Classify the branch itself, but retain the final instruction
    # for the architectural fall-through address after that delay slot.
    try:
        arch_name = node.block.arch.name
    except AttributeError:
        arch_name = ""
    transfer_index = (
        control_transfer_index(arch_name, list(insns), strict=False)
        if arch_has_delay_slot(arch_name)
        else len(insns) - 1
    )
    if transfer_index is None:
        return None
    transfer_insn = insns[transfer_index]
    final_insn = insns[-1]

    try:
        vex = node.block.vex
    except Exception:
        vex = None

    transfer = InsnSemantics(transfer_insn)
    if not transfer.is_jump():
        return None
    # Calls may be members of Capstone's generic jump group. They have their
    # own call/fake-return edge semantics and must not be checked as branches.
    if transfer.is_call() or (vex is not None and vex.jumpkind == "Ijk_Call"):
        return None

    proven_direct_target = proven_unconditional_direct_target(
        arch_name, list(insns), transfer_index
    )

    # Some instruction encodings, including RISC-V ``c.jr ra``, are exposed
    # by Capstone as generic jumps rather than returns. Re-lift just the
    # terminator to avoid treating an old CFG node's stale fallthrough as a
    # successor required by the recovered block.
    terminator_vex = lift_instruction_vex(project, transfer_insn)
    terminator_jumpkind = getattr(terminator_vex, "jumpkind", "")
    if terminator_jumpkind == "Ijk_Ret" or terminator_jumpkind.startswith("Ijk_Sig"):
        return None

    # A normalized CFGFast node can span many instructions. Its full-block
    # VEX ``next`` may therefore describe an earlier stale split rather than
    # the final branch Capstone decoded above. Use a fresh lift of exactly that
    # terminator for its unconditional target, but retain the full node's exit
    # statements for conditional classification: some ARM encodings expose
    # predicated direct branches as generic ``b`` instructions in Capstone.
    fresh_vex_has_control_flow = (
        terminator_vex is not None and terminator_jumpkind != "Ijk_NoDecode"
    )
    target_vex = terminator_vex if fresh_vex_has_control_flow else vex
    vex_has_control_flow = vex is not None and vex.jumpkind != "Ijk_NoDecode"
    exit_targets: list[int] = []
    if vex_has_control_flow:
        for ins_addr, _, stmt in vex.exit_statements:
            if ins_addr not in {transfer_insn.address, final_insn.address}:
                continue
            target = getattr(stmt.dst, "value", None)
            if isinstance(target, int) and target not in exit_targets:
                exit_targets.append(target)

    # A one-instruction VEX lift on a delay-slot ISA can omit the branch Exit
    # entirely. Preserve only an explicit Capstone condition operand here: a
    # one-target branch can be unconditional and must not gain a fallthrough.
    is_conditional = proven_direct_target is None and (
        bool(exit_targets)
        or (arch_has_delay_slot(arch_name) and transfer.has_explicit_branch_condition())
    )
    if not is_conditional and transfer.is_conditional_jump():
        # A malformed CFG node can retain stale VEX without the final branch
        # exit. Re-lift only the ambiguous terminator: this keeps genuine x86
        # conditionals conditional, while correctly classifying S390 `j` as a
        # direct jump instead of inventing a fallthrough edge. Keep the fresh
        # targets too: some architectures expose a PC-relative Capstone
        # operand, while the short VEX lift provides the resolved address.
        fresh_vex = terminator_vex
        if fresh_vex is None:
            # Preserve the conservative Capstone classification if a fresh
            # lift is unavailable; repair is safer than silently omitting an
            # actual branch successor.
            is_conditional = True
        else:
            exit_targets = [
                target
                for ins_addr, _, stmt in fresh_vex.exit_statements
                if ins_addr in {transfer_insn.address, final_insn.address}
                if isinstance(target := getattr(stmt.dst, "value", None), int)
            ]
            is_conditional = bool(exit_targets)

    vex_branch_targets = list(exit_targets)
    if (
        proven_direct_target is None
        and not is_conditional
        and vex_has_control_flow
        and target_vex is not None
        and isinstance(getattr(target_vex, "next", None), pyvex.expr.Const)
        and isinstance(target_vex.next.con.value, int)
    ):
        # An unconditional branch is represented by VEX's default successor,
        # not an Exit statement. This matters for architectures whose
        # Capstone operands retain a PC-relative displacement.
        vex_branch_targets.append(target_vex.next.con.value)

    direct_target = _resolve_direct_branch_target(
        project,
        bounds,
        proven_direct_target
        if proven_direct_target is not None
        else transfer.direct_target(),
        vex_branch_targets,
    )
    if direct_target is None:
        return None

    expected: list[JumpSuccessorExpectation] = []
    kind: Literal["conditional", "direct"]
    fallthrough_addr = final_insn.address + final_insn.size
    if is_conditional:
        expected.append(JumpSuccessorExpectation(direct_target, "Ijk_Boring", True))
        if _can_decode_block_at(project, bounds, fallthrough_addr):
            expected.append(
                JumpSuccessorExpectation(fallthrough_addr, "Ijk_Boring", False)
            )
        kind = "conditional"
    else:
        expected.append(JumpSuccessorExpectation(direct_target, "Ijk_Boring", True))
        kind = "direct"

    present = frozenset(
        succ.addr for succ in graph.successors(node) if not _node_is_placeholder(succ)
    )
    return JumpSuccessorAnalysis(
        kind=kind,
        expected=tuple(expected),
        present=present,
    )


def _missing_jump_successor_anomaly(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> CFGAnomaly | None:
    """Return the direct-jump successor anomaly for ``node``, if any."""

    analysis = _analyze_jump_successors(project, graph, bounds, node)
    if analysis is None:
        return None

    expected_addrs = {item.addr for item in analysis.expected}
    if analysis.present == expected_addrs:
        return None

    label = "conditional branch" if analysis.kind == "conditional" else "direct jump"
    return CFGAnomaly(
        "missing_jump_successor",
        node.addr,
        f"Node {node.addr:#x} is missing {label} successor(s) or shows "
        f"unexpected ones: expected {', '.join(hex(t) for t in sorted(expected_addrs))}, "
        f"got {', '.join(hex(t) for t in sorted(analysis.present)) if analysis.present else '<none>'}",
    )


def _missing_jump_successors(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> tuple[JumpSuccessorExpectation, ...]:
    """Return the subset of expected jump successors still missing in the graph."""

    analysis = _analyze_jump_successors(project, graph, bounds, node)
    if analysis is None:
        return ()

    return tuple(
        item for item in analysis.expected if item.addr not in analysis.present
    )
