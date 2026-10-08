"""Architecture-neutral finite dispatch proofs using shared bounded facts.

Table proofs establish complete entry-address sets independently of legacy
selector domains. Register proofs reuse the same facts for non-table transfers.
"""

from __future__ import annotations

from angr import Project
from cle.backends.elf.relocation.generic import MipsLocalReloc
import pyvex

from bingraph.cfg.graph import CFGGraph, node_vex
from bingraph.cfg.decode import _is_static_pointer_call_target
from bingraph.cfg.jumps import (
    _mips_entry_global_pointer,
    _resolve_vex_expr,
    _vex_tmp_definitions,
    _vex_width_conversion,
    static_jump_target_rejection_reason,
)
from bingraph.cfg.models import FunctionBounds

from .shared_facts import PredecessorFacts


_MAX_EXPRESSION_DEPTH = 24
# Operation, output width, scalar operand/source width, and load-side polarity.
_TargetStep = tuple[str, int, int, bool]


def _table_expression(node, expr, facts, before, depth=0):
    """Reuse completed programs within this immutable fact snapshot."""

    if depth >= _MAX_EXPRESSION_DEPTH or facts.exhausted:
        return None
    key = node, expr, before
    if key in facts._table_expressions:
        return facts._table_expressions[key]
    result = _compile_table_expression(node, expr, facts, before, depth)
    if result is not None and not facts.exhausted:
        facts._table_expressions[key] = result
    return result


def _compile_table_expression(node, expr, facts, before, depth):
    """Compile one table load and exact surrounding integer operations.

    Each other operand must be a must-reaching scalar fact. Preserve VEX
    operation order, widths and read positions, rather than stripping an
    architecture's alignment mask or guessing its instruction sequence.
    Unknown operands, two variable loads, and unsupported operations decline.
    """

    if not facts._step():
        return None
    expr, before = facts._definition(node, expr, before)
    if isinstance(expr, pyvex.expr.Load):
        return expr, before, ()
    if (conversion := _vex_width_conversion(expr)) is not None:
        result = _table_expression(node, expr.args[0], facts, before, depth + 1)
        if result is not None:
            entry, position, steps = result
            source, destination, signed = conversion
            op = "signed" if signed == "S" else "unsigned"
            return entry, position, (*steps, (op, destination, source, False))
    if not isinstance(expr, pyvex.expr.Binop) or len(expr.args) != 2:
        return None
    vex, _ = facts._block(node)
    bits = expr.result_size(vex.tyenv)
    if expr.op not in {
        f"Iop_{op}{bits}" for op in ("Add", "Sub", "And", "Or", "Shl", "Shr")
    }:
        return None
    for index, operand in enumerate(expr.args):
        result = _table_expression(node, operand, facts, before, depth + 1)
        if result is not None:
            # Identify the entry-bearing operand before querying the other
            # side. Probing a dynamic table load as a scalar wastes backward
            # searches when the base happens to be the left operand.
            constant = facts.value(node, expr.args[1 - index], before)
            if constant is None:
                continue
            entry, position, steps = result
            return entry, position, (*steps, (expr.op, bits, constant, index == 0))
    return None


def _table_target(value: int, steps: tuple[_TargetStep, ...], facts) -> int | None:
    """Apply compiled operations with the shared budget and machine arithmetic."""

    for op, bits, constant, value_on_left in steps:
        if op in {"signed", "unsigned"}:
            if not facts._step():
                return None
            if op == "signed" and value & (1 << (constant - 1)):
                value -= 1 << constant
            value &= (1 << bits) - 1
        else:
            operands = (frozenset({value}), frozenset({constant}))
            if not value_on_left:
                operands = tuple(reversed(operands))
            result = facts._combine(op, bits, *operands)
            if result is None:
                return None
            value = next(iter(result))
    return value


def shared_table_targets(project: Project, node, facts: PredecessorFacts):
    """Prove a dispatch from finite addresses, without a legacy index set.

    Evaluate one loaded entry with exact scaling, casts, arithmetic and masks.
    The fact engine must cover every possible entry address. Every row must
    reside wholly in immutable mapped data; one unknown row rejects the whole
    proof. Neither plausible destinations nor adjacent table bytes establish
    selector bounds. Validate all destinations before returning a complete
    proof, so rejecting one row cannot accidentally suppress the UJT fallback.
    """

    vex = node_vex(node)
    if vex is None or vex.jumpkind != "Ijk_Boring":
        return None
    expression = _table_expression(node, vex.next, facts, len(vex.statements))
    if expression is None:
        return None
    entry, before, steps = expression
    size = entry.result_size(vex.tyenv) // 8
    if size not in {1, 2, 4, 8}:
        return None
    addresses = facts.values(node, entry.addr, before)
    if not addresses:
        return None
    targets = []
    for address in sorted(addresses):
        # Cache complete targets, not raw entries. Distinct operation chains
        # may scale, sign-extend or mask the same immutable row differently.
        key = address, size, entry.end, steps
        if key in facts._table_rows:
            targets.append(facts._table_rows[key])
            continue
        if not facts._step():
            return None
        section = project.loader.find_section_containing(address)
        if (
            section is None
            or section.is_writable
            or address + size - 1 > section.max_addr
        ):
            return None
        try:
            data = project.loader.memory.load(address, size)
        except Exception:
            return None
        value = int.from_bytes(data, "little" if entry.end == "Iend_LE" else "big")
        target_addr = _table_target(value, steps, facts)
        if target_addr is None:
            return None
        reason = static_jump_target_rejection_reason(project, target_addr)
        if reason is not None:
            facts.target_rejections[reason] += 1
            return None
        facts._table_rows[key] = target_addr
        targets.append(target_addr)
    return tuple(sorted(set(targets)))


def shared_register_targets(project: Project, node, facts: PredecessorFacts):
    """Resolve register-derived calls or jumps from shared predecessor facts.

    Exact incoming dispatch edges can establish the carried destination, which
    local arithmetic may adjust. Every incoming path must remain bounded and
    every resulting target valid; one unknown path or rejected target preserves
    the unresolved target. Try scalar must facts first: unchanged-register
    loops can establish a unique target without finite-domain recursion.
    Calls additionally allow CLE-identified synthetic function targets, using
    the same validation as static-memory calls. No callee CFG is analyzed.
    This shares the table resolver's budget and leader/redecode rules.
    """

    vex = node_vex(node)
    if vex is None or vex.jumpkind not in {"Ijk_Boring", "Ijk_Call"}:
        return None
    target = _resolve_vex_expr(vex.next, _vex_tmp_definitions(vex))
    # VEX may fold a same-block register call into Const even though the
    # decoder still classifies its machine instruction as indirect. Ordinary
    # constant NEXT jumps, however, include linear fallthroughs, not dispatches.
    if isinstance(target, pyvex.expr.Const) and vex.jumpkind == "Ijk_Boring":
        return None
    scalar = facts.value(node, vex.next)
    values = frozenset({scalar}) if scalar is not None else facts.values(node, vex.next)
    if not values or facts.exhausted:
        return None
    for address in values:
        if vex.jumpkind == "Ijk_Call":
            if not _is_static_pointer_call_target(project, address):
                facts.target_rejections["invalid_call_target"] += 1
                return None
        else:
            reason = static_jump_target_rejection_reason(project, address)
            if reason is not None:
                facts.target_rejections[reason] += 1
                return None
    return tuple(sorted(values))


def _known_registers(project: Project, bounds: FunctionBounds, node) -> dict[int, int]:
    """Seed ABI-proven base registers without encoding dispatch instructions."""

    registers: dict[int, int] = {}
    if project.arch.name.startswith("MIPS"):
        gp = _mips_entry_global_pointer(project, bounds)
        if gp is not None:
            registers[project.arch.registers["gp"][0]] = gp
    elif project.arch.name == "PPC64":
        obj = project.loader.find_object_containing(node.addr)
        toc = obj.get_symbol(".TOC.") if obj is not None else None
        if toc is not None:
            registers[project.arch.registers["r2"][0]] = toc.rebased_addr
    return registers


def table_predecessor_facts(
    project: Project, graph: CFGGraph, bounds: FunctionBounds
) -> PredecessorFacts:
    """Create shared facts once for the stabilized table-analysis snapshot."""

    entry = next((node for node in graph.nodes() if node.addr == bounds.addr), None)
    seeds = _known_registers(project, bounds, entry) if entry is not None else {}
    if project.arch.name.startswith("MIPS"):
        seeds[project.arch.registers["t9"][0]] = bounds.addr
    obj = project.loader.find_object_containing(bounds.addr)
    # CLE distinguishes local MIPS GOT entries from interposable symbols.
    # Their loader-fixed module addresses are ABI linkage roots; other writable
    # memory, including global/lazy GOT slots, remains unknown to shared facts.
    linkage_slots = frozenset(
        reloc.rebased_addr
        for reloc in getattr(obj, "relocs", ())
        if isinstance(reloc, MipsLocalReloc) and reloc.resolved
    )
    return PredecessorFacts(project, graph, bounds, seeds, linkage_slots=linkage_slots)
