"""PPC64 TOC-relative CTR table planning with width-aware selector evidence.

Prove the scaled selector and high bits before reading signed relative rows.
TOC roots, alignment masks and guarded preludes retain the existing semantics
and limits; plausible destinations are not evidence for table boundaries."""

from __future__ import annotations

from collections import deque

import pyvex
from angr import Project
from angr import options as angr_options
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
    _vex_expr_key,
    _vex_index_key,
    _vex_is_zero_extension_from,
    _vex_normalized_table_entry_load,
    _vex_register_with_displacement,
    _vex_scaled_table_index,
)
from bingraph.cfg.jumps.guards import MAX_CROSS_BLOCK_SELECTOR_PROOF_NODES
from bingraph.cfg.models import FunctionBounds, StaticJumpTable, StaticJumpTablePlan
from bingraph.helpers.vex import (
    _resolve_vex_expr,
    _vex_const_value,
    _vex_last_put,
    temporary_definitions,
)

# No-return pruning can leave a larger guard prelude; lifting remains bounded.
MAX_PPC64_GUARD_BLOCK_BYTES = 128


def _ppc64_index_is_zero_extended_on_all_paths(
    graph: CFGGraph, bounds: FunctionBounds, node: CFGNode, offset: int
) -> bool:
    """Find byte-to-64-bit definitions on every traversed index path.

    A low-word guard cannot bound an unmasked 64-bit byte offset by itself.
    Walk backwards through unchanged boring blocks to supported zero-extension
    writes. Calls, foreign/empty roots, conflicting writes and the node cap
    decline the proof; unknown high bits must not survive table scaling.
    """

    pending = deque(graph.predecessors(node))
    seen = set()
    while pending:
        predecessor = pending.popleft()
        if predecessor in seen:
            continue
        seen.add(predecessor)
        if len(seen) > MAX_CROSS_BLOCK_SELECTOR_PROOF_NODES or not (
            _node_is_materialized_cfg_node(predecessor)
            and _node_intersects_bounds(predecessor, bounds)
        ):
            return False
        vex = _node_vex(predecessor)
        if vex is None or vex.jumpkind != "Ijk_Boring":
            return False
        definition = _vex_last_put(vex, offset)
        if definition is not None:
            if not _vex_is_zero_extension_from(
                definition.data, temporary_definitions(vex), 8, 64
            ):
                return False
            continue
        if predecessor.addr == bounds.addr:
            return False
        parents = tuple(graph.predecessors(predecessor))
        if not parents:
            return False
        pending.extend(parents)
    return bool(seen)


def _ppc64_prescaled_ctr_index_source(
    graph: CFGGraph, bounds: FunctionBounds, node: CFGNode, offset: int
) -> tuple[int, int] | None:
    """Find one preserved source for a four-byte table offset in a register.

    The definition may precede a loop backedge, but every incoming path must
    reach the same masked scale without changing its source register. Calls
    invalidate the proof because the offset register is caller-saved.
    """

    pending = deque(graph.predecessors(node))
    seen: set[CFGNode] = set()
    traversed = []
    source: tuple[int, int] | None = None
    while pending:
        predecessor = pending.popleft()
        if predecessor in seen:
            continue
        seen.add(predecessor)
        if len(seen) > MAX_CROSS_BLOCK_SELECTOR_PROOF_NODES or not (
            _node_is_materialized_cfg_node(predecessor)
            and _node_intersects_bounds(predecessor, bounds)
        ):
            return None
        vex = _node_vex(predecessor)
        if vex is None or vex.jumpkind != "Ijk_Boring":
            return None
        definition = _vex_last_put(vex, offset)
        if definition is not None:
            definitions = temporary_definitions(vex)
            scaled = _vex_scaled_table_index(definition.data, definitions)
            if scaled is None or scaled[1] != 2:
                return None
            low_bits = scaled[3]
            # The guard compares the source's low 32 bits; higher bits must
            # not survive the scale into the table's byte offset.
            if low_bits is None or not 0 < low_bits <= 32:
                return None
            key = _vex_index_key(scaled[0], definitions, vex, allow_full_width=True)
            if key is None or key[1] != 64 or key[0] == offset:
                return None
            candidate = (key[0], low_bits)
            if source is not None and source != candidate:
                return None
            source = candidate
            traversed.append(vex)
            continue
        if predecessor.addr == bounds.addr:
            return None
        parents = tuple(graph.predecessors(predecessor))
        if not parents:
            return None
        traversed.append(vex)
        pending.extend(parents)

    if source is None:
        return None
    selector_offset = source[0]
    if any(_vex_last_put(vex, selector_offset) is not None for vex in traversed):
        return None
    return source


def _plan_ppc64_toc_relative_ctr_table(
    project: Project, graph: CFGGraph, bounds: FunctionBounds, node: CFGNode, vex
) -> StaticJumpTablePlan | None:
    """Resolve a guarded PPC64LE CTR dispatch through a TOC-relative offset table.

    The same TOC-derived pointer must base both the signed-entry load and the
    target addition. Local symbolic execution enumerates the finite index
    domain on each incoming edge, only after the exact VEX shape is matched.
    """

    if (
        project.arch.name != "PPC64"
        or project.arch.memory_endness != "Iend_LE"
        or vex.jumpkind != "Ijk_Boring"
    ):
        return None
    # PPC64EL's r2 names the containing ELF object's TOC, not a process-wide
    # address. Accept both direct TOC arithmetic and a pointer loaded from it.
    obj = project.loader.find_object_containing(node.addr)
    toc = obj.get_symbol(".TOC.") if obj is not None else None
    if toc is None:
        return None

    definitions = temporary_definitions(vex)
    next_expr = _resolve_vex_expr(vex.next, definitions)
    if not isinstance(next_expr, pyvex.expr.Binop) or next_expr.op != "Iop_And64":
        return None
    mask = (1 << 64) - 4
    masked_terms = tuple(_resolve_vex_expr(arg, definitions) for arg in next_expr.args)
    target_expr = next(
        (term for term in masked_terms if _vex_const_value(term, definitions) is None),
        None,
    )
    if (
        target_expr is None
        or mask not in (_vex_const_value(term, definitions) for term in masked_terms)
        or not isinstance(target_expr, pyvex.expr.Binop)
        or target_expr.op != "Iop_Add64"
    ):
        return None
    ctr_offset = project.arch.registers["ctr"][0]
    ctr_write = _vex_last_put(vex, ctr_offset)
    if ctr_write is None or _vex_expr_key(ctr_write.data, definitions) != _vex_expr_key(
        target_expr, definitions
    ):
        return None

    for entry, base in (target_expr.args, tuple(reversed(target_expr.args))):
        base = _resolve_vex_expr(base, definitions)
        if base is None or base.result_size(vex.tyenv) != 64:
            continue
        got_addr = None
        if isinstance(base, pyvex.expr.Load):
            toc_base = _vex_register_with_displacement(base.addr, definitions, vex)
            if base.end != "Iend_LE":
                continue
            if toc_base is not None:
                got_addr = (toc.rebased_addr + toc_base[1]) & ((1 << 64) - 1)
        else:
            toc_base = _vex_register_with_displacement(base, definitions, vex)
        if toc_base is None or toc_base[0] != (project.arch.registers["r2"][0], 64):
            continue
        normalized = _vex_normalized_table_entry_load(entry, definitions)
        if normalized is None:
            continue
        load, signed = normalized
        if load.result_size(vex.tyenv) != 32 or not signed:
            continue
        address = _resolve_vex_expr(load.addr, definitions)
        if not isinstance(address, pyvex.expr.Binop) or address.op != "Iop_Add64":
            continue
        base_key = _vex_expr_key(base, definitions)
        base_terms = [
            term
            for term in address.args
            if _vex_expr_key(term, definitions) == base_key
        ]
        if len(base_terms) != 1:
            continue
        index_expr = next(term for term in address.args if term is not base_terms[0])
        scaled = _vex_scaled_table_index(index_expr, definitions)
        if scaled is not None and scaled[1] == 2:
            index_key = _vex_index_key(
                scaled[0], definitions, vex, allow_full_width=True
            )
            if index_key is None or index_key[1] != 64:
                continue
            index_low_bits = scaled[3]
            zero_extended = _ppc64_index_is_zero_extended_on_all_paths(
                graph, bounds, node, index_key[0]
            )
        else:
            offset_key = _vex_index_key(
                index_expr, definitions, vex, allow_full_width=True
            )
            if offset_key is None or offset_key[1] != 64:
                continue
            prescaled = _ppc64_prescaled_ctr_index_source(
                graph, bounds, node, offset_key[0]
            )
            if prescaled is None:
                continue
            index_key = (prescaled[0], 64)
            index_low_bits = prescaled[1]
            zero_extended = False

        predecessors = tuple(graph.predecessors(node))
        if not predecessors or len(predecessors) > 4:
            continue
        entry_indices: set[int] = set()
        proved_paths = 0
        for predecessor in predecessors:
            predecessor_vex = _node_vex(predecessor)
            if not (
                _node_is_materialized_cfg_node(predecessor)
                and _node_intersects_bounds(predecessor, bounds)
                and predecessor.size <= MAX_PPC64_GUARD_BLOCK_BYTES
                and predecessor_vex is not None
                and any(
                    isinstance(statement, pyvex.stmt.Exit)
                    for statement in predecessor_vex.statements
                )
            ):
                break
            try:
                state = project.factory.blank_state(
                    addr=predecessor.addr,
                    add_options={
                        angr_options.SYMBOL_FILL_UNCONSTRAINED_REGISTERS,
                        angr_options.SYMBOL_FILL_UNCONSTRAINED_MEMORY,
                    },
                )
                successors = project.factory.successors(state, size=predecessor.size)
                entering = [
                    successor
                    for successor in successors.flat_successors
                    if successor.addr == node.addr
                ]
                if len(entering) != 1:
                    break
                index_bits = index_low_bits
                if zero_extended:
                    index_bits = min(8, index_bits or 8)
                else:
                    # A full-width comparison in this predecessor can bound
                    # an unmasked selector without an upstream byte load.
                    index_bits = index_bits or 64
                if index_bits > 64:
                    break
                index = entering[0].registers.load(index_key[0], 8) & (
                    (1 << index_bits) - 1
                )
                if entering[0].solver.satisfiable(
                    extra_constraints=[index >= MAX_STATIC_JUMPTABLE_ENTRIES - 1]
                ):
                    break
                # One extra model detects an oversized domain without treating
                # all rows below a maximum as reachable. A blank incoming
                # state overapproximates actual caller states, so this cannot
                # omit a feasible row on the proven predecessor path.
                values = entering[0].solver.eval_upto(
                    index, MAX_STATIC_JUMPTABLE_ENTRIES + 1
                )
                if (
                    not values
                    or len(values) > MAX_STATIC_JUMPTABLE_ENTRIES
                    or max(values) >= MAX_STATIC_JUMPTABLE_ENTRIES - 1
                ):
                    break
                entry_indices.update(values)
                proved_paths += 1
            except Exception as exc:
                logger.debug(
                    f"PPC64 CTR table guard proof failed at {predecessor.addr:#x}: {exc}"
                )
                break
        if proved_paths != len(predecessors) or not entry_indices:
            continue

        if got_addr is None:
            base_addr = (toc.rebased_addr + toc_base[1]) & ((1 << 64) - 1)
        else:
            try:
                base_addr = int.from_bytes(
                    project.loader.memory.load(got_addr, 8), "little"
                )
            except Exception:
                continue
        table = StaticJumpTable(
            base_register_offset=None,
            base_bits=64,
            table_displacement=0,
            index_register_offset=index_key[0],
            index_bits=64,
            entry_size=4,
            endness=load.end,
            signed_entries=True,
            target_and_mask=mask,
            static_base_addr=base_addr,
            index_low_bits=index_low_bits,
        )
        return StaticJumpTablePlan(
            table,
            base_addr,
            tuple(sorted(entry_indices)),
            proof_flavor="ppc64_toc_ctr_table",
        )
    return None
