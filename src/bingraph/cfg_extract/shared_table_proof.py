"""Architecture-neutral finite dispatch proofs and legacy shadow comparisons.

The primary table proof establishes all entry addresses using shared finite facts.
The shadow comparison separately borrows the legacy selector domain, allowing
incremental migration of table shapes the functional proof cannot yet cover.
"""

from __future__ import annotations

from typing import Any

from angr import Project
from cle.backends.elf.relocation.generic import MipsLocalReloc
import pyvex

from bingraph.cfg.graph import CFGGraph, node_vex
from bingraph.cfg.jumps import (
    _jump_table_addr,
    _mips_entry_global_pointer,
    _read_static_jump_table_targets,
    _resolve_vex_expr,
    _vex_const_value,
    _vex_normalized_table_entry_load,
    _vex_tmp_definitions,
    static_jump_target_rejection_reason,
)
from bingraph.cfg.models import FunctionBounds, StaticJumpTable, StaticJumpTablePlan

from .shared_facts import PredecessorFacts


_MAX_EXPRESSION_DEPTH = 24


def shared_table_targets(project: Project, node, facts: PredecessorFacts):
    """Prove a dispatch from finite addresses, without a legacy index set.

    Recognize only a loaded pointer or a base plus a normalized table entry.
    The fact engine must cover every possible entry address. Every row must
    reside wholly in immutable mapped data; one unknown row rejects the whole
    proof. Neither plausible destinations nor adjacent table bytes establish
    selector bounds. Validate all destinations before returning a complete
    proof, so rejecting one row cannot accidentally suppress the UJT fallback.
    """

    vex = node_vex(node)
    if vex is None or vex.jumpkind != "Ijk_Boring":
        return None
    definitions = _vex_tmp_definitions(vex)
    target = _resolve_vex_expr(vex.next, definitions)
    candidates = [(target, None)]
    if (
        isinstance(target, pyvex.expr.Binop)
        and target.op == f"Iop_Add{project.arch.bits}"
    ):
        candidates = [
            (target.args[0], target.args[1]),
            (target.args[1], target.args[0]),
        ]
    for entry_expr, base_expr in candidates:
        normalized = _vex_normalized_table_entry_load(entry_expr, definitions)
        if normalized is None:
            continue
        entry, signed = normalized
        size = entry.result_size(vex.tyenv) // 8
        if size not in {1, 2, 4, 8}:
            continue
        base = 0 if base_expr is None else facts.value(node, base_expr)
        if base is None:
            continue
        addresses = facts.values(node, entry.addr)
        if not addresses:
            continue
        targets = []
        for address in sorted(addresses):
            key = address, size, entry.end, signed, base
            if key in facts._table_rows:
                targets.append(facts._table_rows[key])
                continue
            if not facts._step():
                break
            section = project.loader.find_section_containing(address)
            if (
                section is None
                or section.is_writable
                or address + size - 1 > section.max_addr
            ):
                break
            try:
                data = project.loader.memory.load(address, size)
            except Exception:
                break
            value = int.from_bytes(
                data, "little" if entry.end == "Iend_LE" else "big", signed=signed
            )
            target_addr = (base + value) & ((1 << project.arch.bits) - 1)
            if static_jump_target_rejection_reason(project, target_addr) is not None:
                break
            # Dispatchers often share rows. Cache only fully validated immutable
            # rows, never selector bounds or partially completed table proofs.
            facts._table_rows[key] = target_addr
            targets.append(target_addr)
        else:
            return tuple(sorted(set(targets)))
    return None


def shared_register_targets(project: Project, node, facts: PredecessorFacts):
    """Resolve a finite register-derived jump using the shared predecessor facts.

    Exact incoming dispatch edges can establish the carried destination, which
    local arithmetic may adjust. Every incoming path must remain bounded and
    every resulting target valid; one unknown path or rejected target preserves
    the UJT. This shares the table resolver's budget and leader/redecode rules.
    """

    vex = node_vex(node)
    if vex is None or vex.jumpkind != "Ijk_Boring":
        return None
    target = _resolve_vex_expr(vex.next, _vex_tmp_definitions(vex))
    if isinstance(target, pyvex.expr.Const):
        return None
    values = facts.values(node, vex.next)
    if not values or any(
        static_jump_target_rejection_reason(project, address) is not None
        for address in values
    ):
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


def _address_terms(expr, definitions: dict[int, Any]) -> list[Any]:
    """Flatten additions while retaining temporary read positions for facts."""

    resolved = _resolve_vex_expr(expr, definitions)
    if isinstance(resolved, pyvex.expr.Binop) and resolved.op.startswith("Iop_Add"):
        return [
            *_address_terms(resolved.args[0], definitions),
            *_address_terms(resolved.args[1], definitions),
        ]
    return [expr]


def _exact_value(
    project: Project,
    vex,
    expr,
    definitions: dict[int, Any],
    registers: dict[int, int],
    *,
    depth: int = 0,
) -> int | None:
    """Evaluate a bounded VEX expression using exact constants and file bytes."""

    if depth >= _MAX_EXPRESSION_DEPTH:
        return None
    expr = _resolve_vex_expr(expr, definitions)
    if isinstance(expr, pyvex.expr.Const):
        return expr.con.value if isinstance(expr.con.value, int) else None
    if isinstance(expr, pyvex.expr.Get):
        return registers.get(expr.offset)
    if isinstance(expr, pyvex.expr.Load):
        address = _exact_value(
            project, vex, expr.addr, definitions, registers, depth=depth + 1
        )
        size = expr.result_size(vex.tyenv) // 8
        if address is None or size not in {1, 2, 4, 8}:
            return None
        try:
            raw = project.loader.memory.load(address, size)
        except Exception:
            return None
        return int.from_bytes(raw, "little" if expr.end == "Iend_LE" else "big")
    if not isinstance(expr, pyvex.expr.Binop) or len(expr.args) != 2:
        return None
    left = _exact_value(
        project, vex, expr.args[0], definitions, registers, depth=depth + 1
    )
    right = _exact_value(
        project, vex, expr.args[1], definitions, registers, depth=depth + 1
    )
    if left is None or right is None:
        return None
    bits = expr.result_size(vex.tyenv)
    mask = (1 << bits) - 1
    if expr.op == f"Iop_Add{bits}":
        return (left + right) & mask
    if expr.op == f"Iop_Sub{bits}":
        return (left - right) & mask
    if expr.op == f"Iop_And{bits}":
        return left & right
    if expr.op == f"Iop_Or{bits}":
        return left | right
    if expr.op == f"Iop_Shl{bits}" and right < bits:
        return (left << right) & mask
    return None


def shadow_relative_table_targets(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    reference: StaticJumpTablePlan,
    facts: PredecessorFacts | None = None,
) -> tuple[int, ...] | None:
    """Reprove a relative table's concrete targets using its proven index set.

    This is diagnostic only: the legacy resolver still owns index completeness,
    and a disagreement must never change the CFG.
    """

    vex = node_vex(node)
    if vex is None or vex.jumpkind != "Ijk_Boring":
        return None
    definitions = _vex_tmp_definitions(vex)
    facts = (
        facts if facts is not None else table_predecessor_facts(project, graph, bounds)
    )
    registers = _known_registers(project, bounds, node)
    if any(
        isinstance(statement, pyvex.stmt.Put) and statement.offset in registers
        for statement in vex.statements
    ):
        return None
    target = _resolve_vex_expr(vex.next, definitions)
    target_mask = None
    if isinstance(target, pyvex.expr.Binop) and target.op == "Iop_And64":
        masked = tuple(_vex_const_value(arg, definitions) for arg in target.args)
        if sum(value is not None for value in masked) != 1:
            return None
        target_mask = next(value for value in masked if value is not None)
        target = target.args[0 if masked[0] is None else 1]
        target = _resolve_vex_expr(target, definitions)
    if (
        not isinstance(target, pyvex.expr.Binop)
        or target.op != f"Iop_Add{project.arch.bits}"
    ):
        return None

    for entry_expr, base_expr in (
        (target.args[0], target.args[1]),
        (target.args[1], target.args[0]),
    ):
        normalized = _vex_normalized_table_entry_load(entry_expr, definitions)
        if normalized is None:
            continue
        entry, signed = normalized
        entry_size = entry.result_size(vex.tyenv) // 8
        if entry_size not in {1, 2, 4, 8}:
            continue
        target_base = _exact_value(project, vex, base_expr, definitions, registers)
        if target_base is None:
            target_base = facts.value(node, base_expr)
            if target_base is None:
                continue
        terms = _address_terms(entry.addr, definitions)
        table_addr = 0
        unknown_terms = 0
        for term in terms:
            value = _exact_value(project, vex, term, definitions, registers)
            if value is None:
                value = facts.value(node, term)
            if value is None:
                unknown_terms += 1
            else:
                table_addr += value
        if unknown_terms != 1:
            continue
        mask = (1 << project.arch.bits) - 1
        reference_table = reference.table
        if (
            table_addr & mask != _jump_table_addr(reference.base_addr, reference_table)
            or entry_size != reference_table.entry_size
            or entry.end != reference_table.endness
            or signed != reference_table.signed_entries
            or reference_table.target_scale != 1
            or reference_table.target_or_mask != 0
            or target_base
            != (reference.base_addr + reference_table.target_displacement) & mask
            or target_mask != reference_table.target_and_mask
        ):
            continue
        table = StaticJumpTable(
            base_register_offset=None,
            base_bits=project.arch.bits,
            table_displacement=(table_addr - target_base) & mask,
            index_register_offset=None,
            index_bits=None,
            entry_size=entry_size,
            endness=entry.end,
            signed_entries=signed,
            static_base_addr=target_base,
            target_and_mask=target_mask,
        )
        return _read_static_jump_table_targets(
            project, table, target_base, reference.entry_indices
        )
    return None
