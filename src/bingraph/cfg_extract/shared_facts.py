"""Bounded, demand-driven must facts over an exact intraprocedural graph.

Facts describe integers, rather than instruction patterns. Scalar queries
require all incoming paths to agree; finite-domain queries union different
bounded values, but never drop an unknown path. Both follow VEX definitions
at their actual read positions and share alias, ABI and work-budget rules.
Scalar searches can pass unchanged registers through loops; finite-domain
searches conservatively stop at recursive definitions instead of unrolling.
"""

from __future__ import annotations

from typing import Any

from angr import Project
import pyvex

from bingraph.cfg.graph import CFGGraph, node_vex
from bingraph.cfg.jumps import (
    MAX_STATIC_JUMPTABLE_ENTRIES,
    _resolve_vex_expr,
    _vex_const_value,
    _vex_tmp_definitions,
    _vex_width_conversion,
    _x86_pic_thunk_reads_return_address,
)
from bingraph.cfg.models import FunctionBounds


# These are preserved integer registers under the ELF ABIs used by the corpus.
# Unknown ABIs retain no register facts across calls. SP is deliberately absent:
# VEX's call-side stack update is not the stack state after a callee returns.
_PRESERVED = {
    "AMD64": ("rbx", "rbp", "r12", "r13", "r14", "r15"),
    "X86": ("ebx", "ebp", "esi", "edi"),
    "MIPS32": tuple(f"s{i}" for i in range(8)) + ("fp",),
    "MIPS64": tuple(f"s{i}" for i in range(8)) + ("fp",),
    "PPC64": tuple(f"r{i}" for i in range(14, 32)),
    "S390X": tuple(f"r{i}" for i in range(6, 14)),
    "ARMEL": tuple(f"r{i}" for i in range(4, 12)),
    "ARMHF": tuple(f"r{i}" for i in range(4, 12)),
}


class PredecessorFacts:
    """Cache exact facts for one graph snapshot within a shared work budget.

    ``seeds`` describes only function-entry register values. Reads use their
    actual statement positions, so a temporary retains an old register value
    even if a later PUT overwrites that register. Cached answers are valid only
    for this immutable snapshot; callers must create a new instance after a
    block split or an edge change.

    ``linkage_slots`` is an adapter-supplied set of ABI-fixed local relocation
    slots. These are address roots even in a writable GOT; arbitrary writable
    loads and private stack spills are deliberately still unknown.

    ``values`` additionally tracks finite supersets for masks, conversions of
    proven values, arithmetic and branch-constrained registers. These are not
    feasible path enumerations: unsupported correlations may lose precision,
    but cannot justify removing a possible value. Domains never exceed the
    table-row cap.
    """

    def __init__(
        self,
        project: Project,
        graph: CFGGraph,
        bounds: FunctionBounds,
        seeds: dict[int, int],
        *,
        max_steps: int = 20000,
        linkage_slots: frozenset[int] = frozenset(),
    ) -> None:
        self.project, self.graph, self.bounds = project, graph, bounds
        self.seeds = seeds
        self.max_steps = max_steps
        self.linkage_slots = linkage_slots
        self.steps = 0
        self.exhausted = False
        self._blocks: dict[Any, tuple[Any, dict[int, tuple[int, Any]]]] = {}
        self._cache: dict[tuple[Any, int, int, int], int | None] = {}
        self._write_cache: dict[tuple[Any, int, int, int], int] = {}
        self._active: set[tuple[Any, int, int, int]] = set()
        self._domains: dict[tuple[Any, int, int, int], frozenset[int] | None] = {}
        self._domain_active: set[tuple[Any, int, int, int]] = set()
        self._combinations: dict[tuple, frozenset[int]] = {}
        elf = getattr(project.loader.main_object, "os", "") == "UNIX - System V"
        self.preserved = {
            project.arch.registers[name][0]
            for name in _PRESERVED.get(project.arch.name, ())
            if elf and name in project.arch.registers
        }

    def _step(self) -> bool:
        if self.exhausted:
            return False
        if self.steps >= self.max_steps:
            self.exhausted = True
            return False
        self.steps += 1
        return True

    def _block(self, node):
        if node not in self._blocks:
            vex = node_vex(node)
            definitions = {
                stmt.tmp: (index, stmt.data)
                for index, stmt in enumerate(vex.statements if vex is not None else ())
                if isinstance(stmt, pyvex.stmt.WrTmp)
            }
            self._blocks[node] = vex, definitions
        return self._blocks[node]

    def value(
        self, node, expr, before: int | None = None, depth: int = 0
    ) -> int | None:
        """Evaluate a use without substituting later register definitions."""

        if depth >= 32 or not self._step():
            return None
        vex, definitions = self._block(node)
        if vex is None:
            return None
        before = len(vex.statements) if before is None else before
        if isinstance(expr, pyvex.expr.RdTmp):
            definition = definitions.get(expr.tmp)
            if definition is None or definition[0] >= before:
                return None
            return self.value(node, definition[1], definition[0], depth + 1)
        if isinstance(expr, pyvex.expr.Const):
            return expr.con.value if isinstance(expr.con.value, int) else None
        if isinstance(expr, pyvex.expr.Get):
            return self._register(
                node, before, expr.offset, expr.result_size(vex.tyenv)
            )
        if isinstance(expr, pyvex.expr.Load):
            address = self.value(node, expr.addr, before, depth + 1)
            size = expr.result_size(vex.tyenv) // 8
            if address is None or size not in {1, 2, 4, 8}:
                return None
            section = self.project.loader.find_section_containing(address)
            linkage = address in self.linkage_slots and size == self.project.arch.bytes
            if not linkage and (
                section is None
                or section.is_writable
                or address + size - 1 > section.max_addr
            ):
                return None
            try:
                raw = self.project.loader.memory.load(address, size)
            except Exception:
                return None
            return int.from_bytes(raw, "little" if expr.end == "Iend_LE" else "big")
        conversion = _vex_width_conversion(expr)
        if conversion is not None:
            source, destination, signed = conversion
            value = self.value(node, expr.args[0], before, depth + 1)
            if value is None:
                return None
            if signed == "S" and value & (1 << (source - 1)):
                value -= 1 << source
            return value & ((1 << destination) - 1)
        if not isinstance(expr, pyvex.expr.Binop) or len(expr.args) != 2:
            return None
        left = self.value(node, expr.args[0], before, depth + 1)
        right = self.value(node, expr.args[1], before, depth + 1)
        if left is None or right is None:
            return None
        bits = expr.result_size(vex.tyenv)
        operations = {
            f"Iop_Add{bits}": lambda: left + right,
            f"Iop_Sub{bits}": lambda: left - right,
            f"Iop_And{bits}": lambda: left & right,
            f"Iop_Or{bits}": lambda: left | right,
            f"Iop_Shl{bits}": lambda: left << right if right < bits else None,
        }
        operation = operations.get(expr.op)
        result = operation() if operation is not None else None
        return result & ((1 << bits) - 1) if result is not None else None

    def _call_value(self, node, offset: int, bits: int) -> int | None:
        """Summarize a verified i386 PC thunk, without following its CFG."""

        if self.project.arch.name != "X86" or bits != 32:
            return None
        vex, _ = self._block(node)
        target = _vex_const_value(vex.next, _vex_tmp_definitions(vex))
        symbol = self.project.loader.find_symbol(target) if target is not None else None
        name = self.project.arch.register_names.get(offset, "")
        if (
            target is not None
            and symbol is not None
            and symbol.name == f"__x86.get_pc_thunk.{name.removeprefix('e')}"
            and _x86_pic_thunk_reads_return_address(self.project, target, offset)
        ):
            return node.addr + node.size
        return None

    def _nearest_write(self, node, before: int, offset: int, bits: int) -> int | None:
        """Share completed local scans, not path-dependent register values.

        Return the first overlapping write or barrier, or -1 for no write.
        An interrupted scan returns None and is never cached. Position and
        register width are part of the key to preserve read/alias semantics.
        """

        key = node, before, offset, bits
        if key in self._write_cache:
            return self._write_cache[key]
        vex, _ = self._block(node)
        for index in range(before - 1, -1, -1):
            if not self._step():
                return None
            stmt = vex.statements[index]
            if isinstance(stmt, (pyvex.stmt.Dirty, pyvex.stmt.PutI)):
                break
            if isinstance(stmt, pyvex.stmt.Put):
                size = stmt.data.result_size(vex.tyenv) // 8
                if stmt.offset < offset + bits // 8 and offset < stmt.offset + size:
                    break
        else:
            index = -1
        self._write_cache[key] = index
        return index

    def _register(self, node, before: int, offset: int, bits: int) -> int | None:
        """Meet all reaching definitions, stopping at aliases and call clobbers.

        Search stops at each first overlapping write. A loop with no write
        adds no definition; a loop-carried computation recursively querying
        itself is unknown. Entry and disconnected roots remain real unknown
        inputs, even if another predecessor provides a constant.
        """

        key = node, before, offset, bits
        if key in self._active or len(self._active) >= 32 or self.exhausted:
            return None
        if key in self._cache:
            return self._cache[key]
        self._active.add(key)
        pending = [(node, before)]
        seen = set()
        values: set[int] = set()
        unknown = False
        while pending and not unknown:
            current, position = pending.pop()
            if (current, position) in seen:
                continue
            seen.add((current, position))
            vex, _ = self._block(current)
            if vex is None or not self._step():
                unknown = True
                break
            index = self._nearest_write(current, position, offset, bits)
            if index is None:
                unknown = True
                break
            if index >= 0:
                stmt = vex.statements[index]
                if not isinstance(stmt, pyvex.stmt.Put):
                    unknown = True
                    break
                size = stmt.data.result_size(vex.tyenv) // 8
                value = (
                    self.value(current, stmt.data, index)
                    if stmt.offset == offset and size * 8 == bits
                    else None
                )
                if value is None:
                    unknown = True
                else:
                    values.add(value)
            else:
                incoming = self._inputs(current, offset, bits)
                if incoming is None:
                    unknown = True
                else:
                    positions, seeds = incoming
                    pending.extend(positions)
                    values.update(seeds)
            if len(values) > 1:
                unknown = True
        self._active.remove(key)
        result = next(iter(values)) if not unknown and len(values) == 1 else None
        self._cache[key] = result
        return result

    def _inputs(self, node, offset: int, bits: int):
        """Share exact edge, ABI, and statement-position rules between domains."""

        predecessors = tuple(self.graph.predecessors(node))
        seeds: set[int] = set()
        pending = []
        if node.addr == self.bounds.addr or not predecessors:
            seed = self.seeds.get(offset) if node.addr == self.bounds.addr else None
            if seed is None or bits != self.project.arch.bits:
                return None
            seeds.add(seed)
        for predecessor in predecessors:
            vex, _ = self._block(predecessor)
            edge = self.graph.get_edge_data(predecessor, node) or {}
            if (
                vex is None
                or not self.bounds.addr <= predecessor.addr < self.bounds.end_addr
                or edge.get("candidate")
                or edge.get("unresolved_indirect")
            ):
                return None
            if vex.jumpkind == "Ijk_Call":
                if node.addr != predecessor.addr + predecessor.size:
                    return None
                value = self._call_value(predecessor, offset, bits)
                if value is not None:
                    seeds.add(value)
                    continue
                if offset not in self.preserved:
                    return None
            elif (
                vex.jumpkind not in {"Ijk_Boring", "Ijk_Ret"}
                or edge.get("jumpkind") != "Ijk_Boring"
            ):
                return None
            if edge.get("proven_dispatch") and self._dispatch_carries_register(
                predecessor, node, offset, bits
            ):
                # This exact edge establishes one value, not the union of every
                # destination in the source table. Other incoming edges still
                # participate normally in the scalar meet or finite union.
                seeds.add(node.addr)
                continue
            # Taken exits observe state before the Exit; NEXT observes the
            # completed block. Keep both when both paths reach this successor.
            positions = [
                index
                for index, stmt in enumerate(vex.statements)
                if isinstance(stmt, pyvex.stmt.Exit) and stmt.dst.value == node.addr
            ]
            if (
                not isinstance(vex.next, pyvex.expr.Const)
                or vex.next.con.value == node.addr
                or vex.jumpkind == "Ijk_Call"
            ):
                positions.append(len(vex.statements))
            if not positions:
                return None
            pending.extend((predecessor, position) for position in positions)
        return pending, seeds

    def _dispatch_carries_register(self, source, target, offset, bits):
        """Check that the register at an exact edge equals its jump destination.

        Compare actual VEX read positions: a jump can use an old temporary even
        after the register was overwritten. Partial aliases and early exits to
        the same successor cannot establish the completed block's value.
        """

        vex, _ = self._block(source)
        if bits != self.project.arch.bits or vex.jumpkind != "Ijk_Boring":
            return False
        if isinstance(vex.next, pyvex.expr.Const) or any(
            isinstance(stmt, pyvex.stmt.Exit) and stmt.dst.value == target.addr
            for stmt in vex.statements
        ):
            return False
        position = len(vex.statements)
        write = self._nearest_write(source, position, offset, bits)
        if write is None:
            return False
        value = pyvex.expr.Get(offset, f"Ity_I{bits}")
        if write >= 0:
            stmt = vex.statements[write]
            if (
                not isinstance(stmt, pyvex.stmt.Put)
                or stmt.offset != offset
                or stmt.data.result_size(vex.tyenv) != bits
            ):
                return False
            value, position = stmt.data, write
        return self._same_value(
            source, value, position, vex.next, len(vex.statements), bits
        )

    def values(self, node, expr, before: int | None = None, depth: int = 0):
        """Return a bounded superset of integer values, or unknown.

        Unlike scalar must facts, finite domains union different incoming
        values. Every path still needs a proof. Arithmetic uses width-correct
        wrapping and bounded Cartesian products, without assuming correlations
        between operands. Explicit masks and guards can bound unknown inputs;
        writable memory is never read as if its initial bytes were constant.
        """

        if depth >= 32 or not self._step():
            return None
        vex, definitions = self._block(node)
        if vex is None:
            return None
        before = len(vex.statements) if before is None else before
        bits = expr.result_size(vex.tyenv)
        if isinstance(expr, pyvex.expr.RdTmp):
            definition = definitions.get(expr.tmp)
            if definition is None or definition[0] >= before:
                return None
            return self.values(node, definition[1], definition[0], depth + 1)
        if isinstance(expr, pyvex.expr.Get):
            result = self._register_values(node, before, expr.offset, bits, depth)
        elif isinstance(expr, pyvex.expr.Const):
            result = frozenset({expr.con.value})
        elif (conversion := _vex_width_conversion(expr)) is not None:
            source, destination, signed = conversion
            values = self.values(node, expr.args[0], before, depth + 1)
            result = (
                None
                if values is None
                else frozenset(
                    (
                        value - (1 << source)
                        if signed == "S" and value & (1 << (source - 1))
                        else value
                    )
                    & ((1 << destination) - 1)
                    for value in values
                )
            )
        elif isinstance(expr, pyvex.expr.Binop) and len(expr.args) == 2:
            left = self.values(node, expr.args[0], before, depth + 1)
            right = self.values(node, expr.args[1], before, depth + 1)
            # A small mask bounds even an otherwise completely unknown value.
            masks = [v for v in (left, right) if v is not None and len(v) == 1]
            if (
                expr.op == f"Iop_And{bits}"
                and masks
                and min(next(iter(v)) for v in masks) < MAX_STATIC_JUMPTABLE_ENTRIES
            ):
                mask = min(next(iter(v)) for v in masks)
                result = frozenset(i & mask for i in range(mask + 1))
                if left is not None and right is not None:
                    result = self._combine(expr.op, bits, left, right)
            else:
                result = self._combine(expr.op, bits, left, right)
        else:
            value = self.value(node, expr, before)
            result = frozenset({value}) if value is not None else None
        if self.exhausted:
            return None
        # Storage width alone does not establish the intended dispatch domain.
        # Keep unknown narrow values unknown instead of reading adjacent tables.
        return result

    def _combine(self, op, bits, left, right):
        """Cache only completed, bounded, machine-width Cartesian products."""

        if (
            left is None
            or right is None
            or len(left) * len(right) > MAX_STATIC_JUMPTABLE_ENTRIES
        ):
            return None
        operations = {
            f"Iop_Add{bits}": lambda a, b: a + b,
            f"Iop_Sub{bits}": lambda a, b: a - b,
            f"Iop_And{bits}": lambda a, b: a & b,
            f"Iop_Or{bits}": lambda a, b: a | b,
            f"Iop_Shl{bits}": lambda a, b: a << b if b < bits else None,
            f"Iop_Shr{bits}": lambda a, b: a >> b if b < bits else None,
        }
        operation = operations.get(op)
        if operation is None:
            return None
        # Distinct dispatchers often apply the same arithmetic to the same
        # bounded remainder. These pure results are independent of CFG paths.
        key = op, bits, left, right
        if key in self._combinations:
            return self._combinations[key]
        values = set()
        for a in left:
            for b in right:
                if not self._step() or (value := operation(a, b)) is None:
                    return None
                values.add(value & ((1 << bits) - 1))
        result = frozenset(values)
        self._combinations[key] = result
        return result

    def _definition(self, node, expr, position):
        """Retain a temporary's read position when matching a branch operand."""

        _, definitions = self._block(node)
        for _ in range(32):
            if not isinstance(expr, pyvex.expr.RdTmp):
                return expr, position
            definition = definitions.get(expr.tmp)
            if definition is None or definition[0] >= position:
                break
            position, expr = definition
        return None, position

    def _same_value(self, node, expr, before, other, other_before, bits):
        """Match a VEX definition or register reads with the same reaching write."""

        vex, _ = self._block(node)
        expr, before = self._definition(node, expr, before)
        other, other_before = self._definition(node, other, other_before)
        if (
            expr is None
            or other is None
            or expr.result_size(vex.tyenv) != bits
            or other.result_size(vex.tyenv) != bits
        ):
            return False
        if isinstance(expr, pyvex.expr.Get) and isinstance(other, pyvex.expr.Get):
            if expr.offset != other.offset:
                return False
            write = self._nearest_write(node, before, expr.offset, bits)
            return write is not None and write == self._nearest_write(
                node, other_before, other.offset, bits
            )
        return expr is other

    def _edge_domain(self, node, position, offset, bits, domain=None):
        """Carry edge guard constraints into the outgoing register domain.

        Identical temporary definitions are sufficient even after a mask PUT.
        GETs also match when no intervening overlapping write changed their
        value. A bounded guard operand can flow through supported arithmetic
        into the outgoing register: the guard need not observe its final value.
        Signed comparisons only filter already proven finite operands; they
        never invent a domain. Differing width views and unsupported Boolean
        wrappers are declined rather than interpreted heuristically.
        """

        vex, definitions = self._block(node)
        write = self._nearest_write(node, position, offset, bits)
        query = pyvex.expr.Get(offset, f"Ity_I{bits}")
        query_position = position
        if write is None:
            return domain
        if write >= 0:
            stmt = vex.statements[write]
            if (
                not isinstance(stmt, pyvex.stmt.Put)
                or stmt.offset != offset
                or stmt.data.result_size(vex.tyenv) != bits
            ):
                return domain
            query, query_position = stmt.data, write
        query, query_position = self._definition(node, query, query_position)

        def matches(expr, before, other, other_before):
            return self._same_value(node, expr, before, other, other_before, bits)

        tmp_definitions = {k: v[1] for k, v in definitions.items()}
        exits = [
            (i, s)
            for i, s in enumerate(vex.statements)
            if isinstance(s, pyvex.stmt.Exit)
        ]
        for index, stmt in exits:
            if index > position or stmt.jumpkind != "Ijk_Boring":
                continue
            if not self._step():
                return None
            guard = _resolve_vex_expr(stmt.guard, tmp_definitions)
            while (
                isinstance(guard, pyvex.expr.Unop)
                and _vex_width_conversion(guard) is not None
            ):
                guard = _resolve_vex_expr(guard.args[0], tmp_definitions)
            if not isinstance(guard, pyvex.expr.Binop) or guard.op not in {
                f"Iop_CmpLT{bits}U",
                f"Iop_CmpLE{bits}U",
                f"Iop_CmpLT{bits}S",
                f"Iop_CmpLE{bits}S",
                f"Iop_CmpEQ{bits}",
                f"Iop_CmpNE{bits}",
            }:
                continue
            taken = index == position
            equality = "CmpEQ" in guard.op or "CmpNE" in guard.op
            signed = guard.op.endswith("S")
            selector, bound_expr = (
                guard.args if taken or equality or signed else guard.args[::-1]
            )
            bound, _ = self._definition(node, bound_expr, index)
            reversed_operands = False
            if (equality or signed) and not isinstance(bound, pyvex.expr.Const):
                bound, _ = self._definition(node, selector, index)
                selector = bound_expr
                reversed_operands = True
            if not isinstance(bound, pyvex.expr.Const):
                continue
            if signed:
                # A wrapped remainder can cross zero after arithmetic. Keep
                # its signed branch constraint before reading any table rows,
                # rather than unioning negative indices with positive ones.
                operands = self.values(node, selector, index)
                if operands is None:
                    continue
                signed_bound = bound.con.value
                if signed_bound & (1 << (bits - 1)):
                    signed_bound -= 1 << bits
                constraint = set()
                for value in operands:
                    if not self._step():
                        return None
                    signed_value = value - (1 << bits) if value >> (bits - 1) else value
                    left, right = (
                        (signed_bound, signed_value)
                        if reversed_operands
                        else (signed_value, signed_bound)
                    )
                    passes = left < right if "CmpLT" in guard.op else left <= right
                    if passes == taken:
                        constraint.add(value)
                constraint = frozenset(constraint)
            elif equality:
                if ("CmpEQ" in guard.op) == taken:
                    constraint = frozenset({bound.con.value})
                else:
                    # Excluding one input cannot be translated by excluding the
                    # same output: arithmetic need not be one-to-one.
                    if domain is not None and matches(
                        query, query_position, selector, index
                    ):
                        domain -= {bound.con.value}
                    continue
            else:
                upper = bound.con.value - int(("CmpLT" in guard.op) == taken)
                if not 0 <= upper < MAX_STATIC_JUMPTABLE_ENTRIES:
                    continue
                constraint = frozenset(range(upper + 1))
            mapped = self._map_guard_domain(
                node,
                query,
                query_position,
                lambda expr, before: matches(expr, before, selector, index),
                constraint,
            )
            if mapped is not None:
                domain = mapped if domain is None else domain & mapped
        return domain

    def _map_guard_domain(self, node, expr, before, matches, constraint, depth=0):
        """Replay local arithmetic over a guard-proven operand domain.

        For example, a borrow guard proves the old value is below N, although
        the outgoing PUT stores old-N. Preserve machine-width wrapping when
        mapping that set; a later add of N can then restore the small domain.
        This walks only local definitions, never loop iterations or unknown
        memory. An unrelated input, unsupported operation or exhausted budget
        declines the proof rather than dropping possible values.
        """

        if depth >= 32 or not self._step():
            return None
        expr, before = self._definition(node, expr, before)
        if expr is None:
            return None
        if matches(expr, before):
            return constraint
        if isinstance(expr, pyvex.expr.Const):
            return frozenset({expr.con.value})
        if not isinstance(expr, pyvex.expr.Binop) or len(expr.args) != 2:
            return None
        vex, _ = self._block(node)
        left = self._map_guard_domain(
            node, expr.args[0], before, matches, constraint, depth + 1
        )
        right = self._map_guard_domain(
            node, expr.args[1], before, matches, constraint, depth + 1
        )
        return self._combine(expr.op, expr.result_size(vex.tyenv), left, right)

    def _register_values(self, node, before, offset, bits, depth):
        key = node, before, offset, bits
        if (
            key in self._domain_active
            or len(self._domain_active) >= 32
            or self.exhausted
        ):
            return None
        if key in self._domains:
            return self._domains[key]
        self._domain_active.add(key)
        vex, _ = self._block(node)
        index = self._nearest_write(node, before, offset, bits)
        result = None
        if index is not None and index >= 0:
            stmt = vex.statements[index]
            if (
                isinstance(stmt, pyvex.stmt.Put)
                and stmt.offset == offset
                and stmt.data.result_size(vex.tyenv) == bits
            ):
                result = self.values(node, stmt.data, index, depth + 1)
        elif index == -1 and (incoming := self._inputs(node, offset, bits)) is not None:
            pending, seeds = incoming
            result = frozenset(seeds)
            for predecessor, position in pending:
                domain = self._edge_domain(predecessor, position, offset, bits)
                if domain is None:
                    domain = self.values(
                        predecessor,
                        pyvex.expr.Get(offset, f"Ity_I{bits}"),
                        position,
                        depth + 1,
                    )
                    domain = self._edge_domain(
                        predecessor, position, offset, bits, domain
                    )
                if (
                    domain is None
                    or len(result | domain) > MAX_STATIC_JUMPTABLE_ENTRIES
                ):
                    result = None
                    break
                result |= domain
        self._domain_active.remove(key)
        self._domains[key] = result
        return result
