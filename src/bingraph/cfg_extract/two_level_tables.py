"""Exact, bounded recovery of byte-map to pointer-table indirect jumps.

The dispatcher must load a byte from immutable memory, use that byte to load
a pointer from immutable memory, and jump through the pointer. We replay only
its immediate guarded predecessors and the dispatcher. Unknown input remains
symbolic; no path through the rest of the function is executed.
"""

from __future__ import annotations

from collections import deque
from typing import Any

import claripy
from angr import BP_AFTER, BP_BEFORE, Project, options as angr_options
from angr.knowledge_plugins.cfg import CFGNode
from loguru import logger
import pyvex

from bingraph.cfg.graph import (
    CFGGraph,
    node_intersects_bounds,
    node_is_materialized_cfg_node,
    node_vex,
)
from bingraph.cfg.jumps import (
    MAX_STATIC_JUMPTABLE_ENTRIES,
    _vex_const_value,
    _vex_expr_key,
    _vex_tmp_definitions,
    static_jump_target_rejection_reason,
)
from bingraph.cfg.models import FunctionBounds


_MAX_PREDECESSORS = 4
_MAX_PROBE_BLOCK_BYTES = 160
_MAX_BASE_PROOF_NODES = 256
_SOLVER_TIMEOUT_MS = 2000


class _ProbeFinished(Exception):
    """Stop VEX replay after observing the final computed jump."""


class _ProbeRejected(Exception):
    """Reject a proof that needs an unknown value or exceeds its budget."""


def _loads_in_key(key: tuple[Any, ...]) -> tuple[tuple[Any, ...], ...]:
    """Return every load in a structural VEX expression, including nested loads."""

    loads = (key,) if key[0] == "load" else ()
    for child in key[1:]:
        if isinstance(child, tuple):
            loads += _loads_in_key(child)
    return loads


def _load_site(vex, key: tuple[Any, ...]) -> int | None:
    """Find the sole instruction that performs this exact VEX memory read."""

    definitions = _vex_tmp_definitions(vex)
    current_addr = None
    sites: list[int] = []
    for stmt in vex.statements:
        if isinstance(stmt, pyvex.stmt.IMark):
            current_addr = stmt.addr
        elif (
            current_addr is not None
            and isinstance(stmt, pyvex.stmt.WrTmp)
            and isinstance(stmt.data, pyvex.expr.Load)
            and _vex_expr_key(stmt.data, definitions) == key
        ):
            sites.append(current_addr)
    return sites[0] if len(sites) == 1 else None


def _two_level_load_sites(
    vex, pointer_bits: int
) -> tuple[int, int, tuple[Any, ...]] | None:
    """Require the final PC expression to contain one nested byte-map read."""

    if any(
        isinstance(stmt, (pyvex.stmt.Exit, pyvex.stmt.StoreG))
        for stmt in vex.statements
    ):
        return None
    key = _vex_expr_key(vex.next, _vex_tmp_definitions(vex))
    if key is None:
        return None
    outer = [
        load
        for load in _loads_in_key(key)
        if load[2] == pointer_bits
        and any(child[2] == 8 for child in _loads_in_key(load[3]))
    ]
    if len(outer) != 1 or outer[0][1] != vex.arch.memory_endness:
        return None
    inner = [load for load in _loads_in_key(outer[0][3]) if load[2] == 8]
    if len(inner) != 1:
        return None
    map_site = _load_site(vex, inner[0])
    pointer_site = _load_site(vex, outer[0])
    if map_site is None or pointer_site is None or map_site >= pointer_site:
        return None
    return map_site, pointer_site, inner[0][3]


def _source_atoms(
    key: tuple[Any, ...],
    vex,
    statements: tuple[Any, ...] | list[Any],
    seen: frozenset[int] = frozenset(),
) -> set[tuple[Any, ...]]:
    """Find input registers/loads, looking through predecessor register copies.

    This is only a cheap necessary condition for symbolic replay. Alias writes
    may broaden the dependency, but cannot make the final proof unsound.
    """

    if key[0] == "load":
        return {key}
    if key[0] == "get":
        offset, bits = key[1:]
        if offset in seen:
            return {key}
        for stmt in reversed(statements):
            if not isinstance(stmt, pyvex.stmt.Put):
                continue
            end = stmt.offset + stmt.data.result_size(vex.tyenv) // 8
            if not (stmt.offset <= offset and end >= offset + bits // 8):
                continue
            replacement = _vex_expr_key(stmt.data, _vex_tmp_definitions(vex))
            if replacement is not None:
                return _source_atoms(replacement, vex, statements, seen | {offset})
        return {key}
    return {
        atom
        for child in key[1:]
        if isinstance(child, tuple)
        for atom in _source_atoms(child, vex, statements, seen)
    }


def _guard_depends_on_map_index(
    vex, exit_index: int, map_address_key: tuple[Any, ...]
) -> bool:
    """Skip guards that cannot constrain any input to the byte-map address."""

    stmt = vex.statements[exit_index]
    guard_key = _vex_expr_key(stmt.guard, _vex_tmp_definitions(vex))
    if guard_key is None:
        return False
    preceding = vex.statements[:exit_index]
    guard_atoms = _source_atoms(guard_key, vex, preceding)
    map_atoms = _source_atoms(map_address_key, vex, preceding)
    for guard_atom in guard_atoms:
        for map_atom in map_atoms:
            if guard_atom == map_atom:
                return True
            if guard_atom[0] == map_atom[0] == "get":
                guard_start, guard_bits = guard_atom[1:]
                map_start, map_bits = map_atom[1:]
                if (
                    guard_start < map_start + map_bits // 8
                    and map_start < guard_start + guard_bits // 8
                ):
                    return True
    return False


def _register_reads(key: tuple[Any, ...]) -> set[tuple[int, int]]:
    """Identify register inputs to a VEX memory-address expression."""

    if key[0] == "get":
        return {(key[1], key[2])}
    return {
        register
        for child in key[1:]
        if isinstance(child, tuple)
        for register in _register_reads(child)
    }


def _must_reaching_constant(
    graph: CFGGraph, bounds: FunctionBounds, node: CFGNode, offset: int, bits: int
) -> int | None:
    """Require every bounded incoming path to define the same full register."""

    pending = deque([node])
    seen: set[CFGNode] = set()
    values: set[int] = set()
    register_end = offset + bits // 8
    while pending:
        current = pending.popleft()
        if current in seen:
            continue
        seen.add(current)
        if len(seen) > _MAX_BASE_PROOF_NODES or not (
            node_is_materialized_cfg_node(current)
            and node_intersects_bounds(current, bounds)
        ):
            return None
        vex = node_vex(current)
        if vex is None:
            return None
        definitions = _vex_tmp_definitions(vex)
        matching = False
        for stmt in reversed(vex.statements):
            if not isinstance(stmt, pyvex.stmt.Put):
                continue
            write_end = stmt.offset + stmt.data.result_size(vex.tyenv) // 8
            if stmt.offset >= register_end or write_end <= offset:
                continue
            if stmt.offset != offset or write_end != register_end:
                return None
            value = _vex_const_value(stmt.data, definitions)
            if value is None:
                return None
            values.add(value)
            matching = True
            break
        if matching:
            continue
        parents = tuple(graph.predecessors(current))
        if not parents or current.addr == bounds.addr:
            return None
        pending.extend(parents)
    return next(iter(values)) if len(values) == 1 else None


def _seed_proven_bases(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    predecessor: CFGNode,
    state,
) -> None:
    """Supply only constants already proved from the entry or ABI metadata."""

    vex = node_vex(predecessor)
    if vex is None:
        return
    definitions = _vex_tmp_definitions(vex)
    writes = {
        stmt.offset for stmt in vex.statements if isinstance(stmt, pyvex.stmt.Put)
    }
    reads: set[tuple[int, int]] = set()
    for stmt in vex.statements:
        if not isinstance(stmt, pyvex.stmt.WrTmp) or not isinstance(
            stmt.data, pyvex.expr.Load
        ):
            continue
        key = _vex_expr_key(stmt.data.addr, definitions)
        if key is not None:
            reads.update(_register_reads(key))

    for offset, bits in reads:
        if bits != project.arch.bits or offset in writes:
            continue
        value = _must_reaching_constant(graph, bounds, predecessor, offset, bits)
        if value is not None:
            state.registers.store(offset, claripy.BVV(value, bits))


def _immutable_values(
    project: Project, addresses: tuple[int, ...], size: int
) -> frozenset[int]:
    """Read a finite immutable table domain, rejecting writable/unknown bytes."""

    values: set[int] = set()
    byteorder = "little" if project.arch.memory_endness == "Iend_LE" else "big"
    for addr in addresses:
        obj = project.loader.find_object_containing(addr)
        if obj is None or obj is getattr(project.loader, "extern_object", None):
            raise _ProbeRejected
        section = obj.find_section_containing(addr)
        if (
            section is None
            or section is not obj.find_section_containing(addr + size - 1)
            or not section.is_readable
            or section.is_writable
        ):
            raise _ProbeRejected
        raw = project.loader.memory.load(addr, size)
        values.add(int.from_bytes(raw, byteorder))
    return frozenset(values)


def _finite_values(state, expr) -> tuple[int, ...]:
    """Enumerate a complete domain or decline an oversized one."""

    values = state.solver.eval_upto(expr, MAX_STATIC_JUMPTABLE_ENTRIES + 1)
    if not values or len(values) > MAX_STATIC_JUMPTABLE_ENTRIES:
        raise _ProbeRejected
    return tuple(sorted(values))


def _prove_one_predecessor(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    predecessor: CFGNode,
    node: CFGNode,
    sites: tuple[int, int, tuple[Any, ...]],
) -> tuple[int, ...] | None:
    """Enumerate both static loads under one predecessor's taken guard."""

    pred_vex = node_vex(predecessor)
    if pred_vex is None:
        return None
    exits = [
        index
        for index, stmt in enumerate(pred_vex.statements)
        if isinstance(stmt, pyvex.stmt.Exit) and stmt.dst.value == node.addr
    ]
    if len(exits) != 1 or not _guard_depends_on_map_index(pred_vex, exits[0], sites[2]):
        return None

    state = project.factory.blank_state(
        addr=predecessor.addr,
        add_options={
            angr_options.SYMBOL_FILL_UNCONSTRAINED_REGISTERS,
            angr_options.SYMBOL_FILL_UNCONSTRAINED_MEMORY,
        },
    )
    state.solver._solver.timeout = _SOLVER_TIMEOUT_MS
    _seed_proven_bases(project, graph, bounds, predecessor, state)
    successors = project.factory.successors(
        state, addr=predecessor.addr, size=predecessor.size
    )
    entering = [succ for succ in successors.flat_successors if succ.addr == node.addr]
    if len(entering) != 1:
        return None
    state = entering[0]
    state.solver._solver.timeout = _SOLVER_TIMEOUT_MS
    map_site, pointer_site, _ = sites
    map_value = claripy.BVS("static_byte_map_value", 8)
    pointer_value = claripy.BVS("static_pointer_value", project.arch.bits)
    domains: dict[str, frozenset[int]] = {}
    result: tuple[int, ...] | None = None

    def before_read(current) -> None:
        site = current.scratch.ins_addr
        if site not in {map_site, pointer_site}:
            return
        if site == pointer_site and "map" not in domains:
            raise _ProbeRejected
        address = current.inspect.attrs.mem_read_address
        if site == pointer_site and address.variables != map_value.variables:
            # Otherwise an unknown base or an unguarded second index could
            # silently add pointer-table rows beyond this map's value set.
            raise _ProbeRejected
        addresses = _finite_values(current, address)
        size = 1 if site == map_site else project.arch.bytes
        domains["map" if site == map_site else "pointer"] = _immutable_values(
            project, addresses, size
        )
        # The actual symbolic read is unnecessary: its exact finite value
        # domain is supplied in after_read, avoiding address concretization.
        current.inspect.attrs.mem_read_address = claripy.BVV(
            addresses[0], project.arch.bits
        )

    def after_read(current) -> None:
        site = current.scratch.ins_addr
        if site == map_site:
            values = domains.get("map")
            if not values:
                raise _ProbeRejected
            current.inspect.attrs.mem_read_expr = map_value
            current.solver.add(claripy.Or(*(map_value == value for value in values)))
        elif site == pointer_site:
            values = domains.get("pointer")
            if not values:
                raise _ProbeRejected
            current.inspect.attrs.mem_read_expr = pointer_value
            current.solver.add(
                claripy.Or(*(pointer_value == value for value in values))
            )

    def before_exit(current) -> None:
        nonlocal result
        if "pointer" not in domains:
            raise _ProbeRejected
        target = current.inspect.attrs.exit_target
        if target.variables != pointer_value.variables:
            raise _ProbeRejected
        targets = _finite_values(current, target)
        if any(
            static_jump_target_rejection_reason(project, addr) is not None
            for addr in targets
        ):
            raise _ProbeRejected
        result = targets
        raise _ProbeFinished

    state.inspect.b("mem_read", when=BP_BEFORE, action=before_read)
    state.inspect.b("mem_read", when=BP_AFTER, action=after_read)
    state.inspect.b("exit", when=BP_BEFORE, action=before_exit)
    try:
        project.factory.successors(state, addr=node.addr, size=node.size)
    except _ProbeFinished:
        return result
    except _ProbeRejected:
        return None
    return None


def exact_two_level_table_targets(
    project: Project, graph: CFGGraph, bounds: FunctionBounds, node: CFGNode
) -> tuple[int, ...] | None:
    """Resolve only fully guarded, immutable byte-map/pointer-table jumps.

    All incoming edges must be explicit guarded branches. Each is replayed
    separately from an unconstrained state, so successful enumeration is an
    overapproximation of real callers, never a sampled execution path.
    """

    vex = node_vex(node)
    if (
        vex is None
        or node.size > _MAX_PROBE_BLOCK_BYTES
        or vex.jumpkind not in {"Ijk_Boring", "Ijk_Ret"}
    ):
        return None
    sites = _two_level_load_sites(vex, project.arch.bits)
    if sites is None:
        return None
    predecessors = tuple(graph.predecessors(node))
    if not 0 < len(predecessors) <= _MAX_PREDECESSORS or node.addr == bounds.addr:
        return None
    targets: set[int] = set()
    for predecessor in predecessors:
        if not (
            node_is_materialized_cfg_node(predecessor)
            and node_intersects_bounds(predecessor, bounds)
            and predecessor.size <= _MAX_PROBE_BLOCK_BYTES
        ):
            return None
        try:
            proved = _prove_one_predecessor(
                project, graph, bounds, predecessor, node, sites
            )
        except Exception as exc:
            logger.debug(f"Two-level jump-table proof failed at {node.addr:#x}: {exc}")
            return None
        if not proved:
            return None
        targets.update(proved)
    return tuple(sorted(targets)) if targets else None
