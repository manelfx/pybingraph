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
from angr.engines.vex.claripy.ccall import data as x86_cc_data
import pyvex

from bingraph.cfg.graph import CFGGraph, node_vex
from bingraph.cfg.jumps import (
    MAX_STATIC_JUMPTABLE_ENTRIES,
    _vex_const_value,
    _vex_tmp_definitions,
    _vex_width_conversion,
    _x86_pic_thunk_reads_return_address,
)
from bingraph.cfg.models import FunctionBounds
from bingraph.cfg_extract.shared_relations import RelationalValues


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
        self._expressions: dict[tuple[Any, Any, int], frozenset[int]] = {}
        self._table_rows: dict[
            tuple[int, int, str, tuple[tuple[str, int, int, bool], ...]], int
        ] = {}
        self._table_expressions: dict[
            tuple[Any, Any, int],
            tuple[Any, int, tuple[tuple[str, int, int, bool], ...]],
        ] = {}
        self._domain_active: set[tuple[Any, int, int, int]] = set()
        self._combinations: dict[tuple, frozenset[int]] = {}
        self._register_views = frozenset(project.arch.registers.values())
        self._predicates: dict[tuple[Any, int], Any] = {}
        self._guard_domains: dict[tuple[Any, int, bool], frozenset[int]] = {}
        self._guard_constants: dict[tuple[Any, Any, int], int] = {}
        self._guard_operands: dict[tuple[Any, Any, int], tuple[Any, int]] = {}
        self._guard_views: dict[tuple[Any, Any, int, int], tuple] = {}
        self._guard_widths: set[int] = set()
        self._relations = RelationalValues(self)
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
        scanned = []
        for index in range(before - 1, -1, -1):
            suffix = node, index + 1, offset, bits
            if suffix in self._write_cache:
                index = self._write_cache[suffix]
                break
            if not self._step():
                return None
            scanned.append(suffix)
            stmt = vex.statements[index]
            if isinstance(stmt, (pyvex.stmt.Dirty, pyvex.stmt.PutI)):
                break
            if isinstance(stmt, pyvex.stmt.Put):
                size = stmt.data.result_size(vex.tyenv) // 8
                if stmt.offset < offset + bits // 8 and offset < stmt.offset + size:
                    break
        else:
            index = -1
        # Every scanned position reaches the same first write. Cache the whole
        # completed suffix so queries at nearby old-temp positions do not walk
        # the same statements again. Interrupted scans cache nothing.
        for suffix in scanned:
            self._write_cache[suffix] = index
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
        wrapping and bounded Cartesian products. Additive expressions can be
        refined by verified, bounded definition/edge correlations; unsupported
        relations keep the Cartesian superset. Masks and guards can bound
        unknown inputs; writable bytes are never assumed constant.
        """

        if depth >= 32 or self.exhausted:
            return None
        vex, _ = self._block(node)
        if vex is None:
            return None
        before = len(vex.statements) if before is None else before
        key = node, expr, before
        if key in self._expressions:
            return self._expressions[key]
        result = self._expression_values(node, expr, before, depth)
        # Repeated table/guard queries share completed finite supersets, not
        # unknown or interrupted answers. Read positions distinguish old temps
        # and aliases; a new graph round always owns a fresh expression cache.
        if result is not None and not self.exhausted:
            self._expressions[key] = result
        return result

    def _expression_values(self, node, expr, before, depth):
        if not self._step():
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
            masks = [
                (next(iter(domain)), 1 - index)
                for index, domain in enumerate((left, right))
                if domain is not None and len(domain) == 1
            ]
            if (
                expr.op == f"Iop_And{bits}"
                and masks
                and min(masks)[0] < MAX_STATIC_JUMPTABLE_ENTRIES
            ):
                mask, operand_index = min(masks)
                operand = expr.args[operand_index]
                result = self._masked_values(node, operand, mask, before, depth + 1)
                if result is None:
                    result = frozenset(i & mask for i in range(mask + 1))
                if left is not None and right is not None:
                    combined = self._combine(expr.op, bits, left, right)
                    result = None if combined is None else result & combined
            else:
                result = self._combine(expr.op, bits, left, right)
                if (
                    result is not None
                    and expr.op in {f"Iop_Add{bits}", f"Iop_Sub{bits}"}
                    and left is not None
                    and len(left) > 1
                    and right is not None
                    and len(right) > 1
                ):
                    # Query only when independent domains actually lose a
                    # possible correlation. The optional joint evaluator uses
                    # the same immutable snapshot, alias/ABI rules and budget.
                    related = self._relations.values(node, expr, before)
                    if related is not None:
                        result &= related
        else:
            value = self.value(node, expr, before)
            result = frozenset({value}) if value is not None else None
        if self.exhausted:
            return None
        # Storage width alone does not establish the intended dispatch domain.
        # Keep unknown narrow values unknown instead of reading adjacent tables.
        return result

    def _masked_values(self, node, expr, mask, before, depth):
        """Query only bits consumed by a mask, never unknown discarded bits.

        Push the bit demand through logical shifts/OR (including lifted
        rotates). A contained register view can then use its own guard and
        reaching write. This does not turn an unknown full register into a
        constant or infer a selector domain from storage width alone.
        """

        if depth >= 32 or not self._step():
            return None
        if mask == 0:
            return frozenset({0})
        expr, before = self._definition(node, expr, before)
        if expr is None:
            return None
        vex, _ = self._block(node)
        bits = expr.result_size(vex.tyenv)
        if isinstance(expr, pyvex.expr.Const):
            return frozenset({expr.con.value & mask})
        if isinstance(expr, pyvex.expr.Get):
            views = sorted(
                (size, offset)
                for offset, size in self._register_views
                if mask.bit_length() <= size * 8 <= bits
                and offset == self._low_view_offset(expr.offset, bits, size * 8)
            )
            result = None
            full_known = (
                self._domains.get((node, before, expr.offset, bits)) is not None
            )
            for index, (size, offset) in enumerate(views):
                # An unknown full value needs a smallest-view probe to discover
                # narrow guards. Otherwise try only observed guard widths and
                # the original (already queried) view, not every alias.
                if (
                    (index != 0 or full_known)
                    and size * 8 != bits
                    and size * 8 not in self._guard_widths
                ):
                    continue
                values = self.values(
                    node, pyvex.expr.Get(offset, f"Ity_I{size * 8}"), before, depth + 1
                )
                if values is None:
                    continue
                result = frozenset(v & mask for v in values)
                # A guard may test a wider view than the smallest consumed
                # slice. Stop once a view improves the mask-only superset.
                if len(result) < 1 << mask.bit_count():
                    break
            return result
        if (conversion := _vex_width_conversion(expr)) is not None:
            source, _destination, _signed = conversion
            if mask.bit_length() <= source:
                return self._masked_values(node, expr.args[0], mask, before, depth + 1)
        if not isinstance(expr, pyvex.expr.Binop):
            return None
        if expr.op == f"Iop_Or{bits}":
            left, right = (
                self._masked_values(node, arg, mask, before, depth + 1)
                for arg in expr.args
            )
            return self._combine(expr.op, bits, left, right)
        if expr.op in {f"Iop_Shl{bits}", f"Iop_Shr{bits}"}:
            shift = self.value(node, expr.args[1], before)
            if shift is not None and 0 <= shift < bits:
                demand = (
                    mask >> shift
                    if expr.op == f"Iop_Shl{bits}"
                    else (mask << shift) & ((1 << bits) - 1)
                )
                values = self._masked_values(
                    node, expr.args[0], demand, before, depth + 1
                )
                result = self._combine(expr.op, bits, values, frozenset({shift}))
                return None if result is None else frozenset(v & mask for v in result)
        return None

    def _low_view_offset(self, offset, source_bits, bits):
        """Locate a low register slice in VEX's endian-specific byte layout."""

        if self.project.arch.register_endness == "Iend_BE":
            return offset + (source_bits - bits) // 8
        return offset

    def _written_view(self, node, stmt, offset, bits):
        """Project a containing PUT; partial writes never define wider reads."""

        vex, _ = self._block(node)
        if not isinstance(stmt, pyvex.stmt.Put):
            return None
        source_bits = stmt.data.result_size(vex.tyenv)
        if not (
            bits >= 8
            and stmt.offset <= offset
            and offset + bits // 8 <= stmt.offset + source_bits // 8
        ):
            return None
        shift = (offset - stmt.offset) * 8
        if self.project.arch.register_endness == "Iend_BE":
            shift = source_bits - bits - shift
        expr = stmt.data
        if shift:
            expr = pyvex.expr.Binop(
                f"Iop_Shr{source_bits}", [expr, pyvex.expr.Const(pyvex.const.U8(shift))]
            )
        if bits < source_bits:
            expr = pyvex.expr.Unop(f"Iop_{source_bits}to{bits}", [expr])
        return expr

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

    def _guard_comparison(self, node, expr, before, depth=0):
        """Recover a typed predicate and its polarity through Boolean wrappers.

        Conversions are transparent only around a proven Boolean comparison.
        Likewise EQ/NE with zero may invert its truth, but an arbitrary integer
        nonzero test is not reinterpreted as an underlying ordered comparison.
        """

        # The outer predicate is charged by _edge_domain. Additional wrapper
        # expansion consumes work, but a direct comparison is not charged twice.
        if depth >= 32 or self.exhausted or (depth > 0 and not self._step()):
            return None
        expr, before = self._definition(node, expr, before)
        if expr is None:
            return None
        if _vex_width_conversion(expr) is not None:
            return self._guard_comparison(node, expr.args[0], before, depth + 1)
        if isinstance(expr, pyvex.expr.CCall):
            return self._x86_guard_comparison(node, expr, before)
        if not isinstance(expr, pyvex.expr.Binop) or expr.op not in {
            f"Iop_Cmp{relation}{bits}{sign}"
            for bits in (1, 8, 16, 32, 64)
            for relation, signs in (("LT", "US"), ("LE", "US"), ("EQ", ""), ("NE", ""))
            for sign in (signs or ("",))
        }:
            return None
        if expr.op.startswith(("Iop_CmpEQ", "Iop_CmpNE")):
            for value, condition in (expr.args, expr.args[::-1]):
                if self._guard_constant(node, value, before) != 0:
                    continue
                nested = self._guard_comparison(node, condition, before, depth + 1)
                if nested is not None:
                    comparison, position, inverted = nested
                    return comparison, position, inverted ^ ("CmpEQ" in expr.op)
        return expr, before, False

    def _x86_guard_comparison(self, node, expr, before):
        """Translate verified subtraction flags into the common predicate form.

        A split cmp/jcc reads saved VEX flag operands. The operation must be a
        must-reaching SUB of known width; calls or mixed reaching operations
        therefore decline. No instruction pattern or function identity is used.
        Other flag producers (ADD, LOGIC, COPY, etc.) remain unsupported.
        """

        arch = self.project.arch.name
        helper = {
            "AMD64": "amd64g_calculate_condition",
            "X86": "x86g_calculate_condition",
        }
        if expr.callee.name != helper.get(arch) or len(expr.args) != 5:
            return None
        condition = self._guard_constant(node, expr.args[0], before)
        operation = self.value(node, expr.args[1], before)
        data = x86_cc_data[arch]
        widths = {
            data["OpTypes"][f"G_CC_OP_SUB{suffix}"]: bits
            for suffix, bits in (("B", 8), ("W", 16), ("L", 32), ("Q", 64))
            if data["OpTypes"].get(f"G_CC_OP_SUB{suffix}") is not None
        }
        bits = widths.get(operation)
        predicates = {
            data["CondTypes"][f"Cond{name}"]: (op, sign, inverted)
            for name, op, sign, inverted in (
                ("Z", "EQ", "", False),
                ("NZ", "EQ", "", True),
                ("B", "LT", "U", False),
                ("NB", "LT", "U", True),
                ("BE", "LE", "U", False),
                ("NBE", "LE", "U", True),
                ("L", "LT", "S", False),
                ("NL", "LT", "S", True),
                ("LE", "LE", "S", False),
                ("NLE", "LE", "S", True),
            )
        }
        predicate = predicates.get(condition)
        if bits is None or predicate is None:
            return None
        vex, _ = self._block(node)
        operands = []
        for arg in expr.args[2:4]:
            source_bits = arg.result_size(vex.tyenv)
            if source_bits != self.project.arch.bits or bits > source_bits:
                return None
            operands.append(
                arg
                if source_bits == bits
                else pyvex.expr.Unop(f"Iop_{source_bits}to{bits}", [arg])
            )
        op, sign, inverted = predicate
        return pyvex.expr.Binop(f"Iop_Cmp{op}{bits}{sign}", operands), before, inverted

    def _guard_constant(self, node, expr, before, depth=0):
        """Fold local constant-only guard terms without predecessor searches."""

        key = node, expr, before
        if key in self._guard_constants:
            return self._guard_constants[key]
        result = self._fold_guard_constant(node, expr, before, depth)
        if result is not None and not self.exhausted:
            self._guard_constants[key] = result
        return result

    def _fold_guard_constant(self, node, expr, before, depth):
        if depth >= 32 or self.exhausted:
            return None
        expr, before = self._definition(node, expr, before)
        if isinstance(expr, pyvex.expr.Const):
            return expr.con.value
        conversion = _vex_width_conversion(expr)
        masked = isinstance(expr, pyvex.expr.Binop) and expr.op.startswith("Iop_And")
        if (conversion is None and not masked) or not self._step():
            return None
        if conversion is not None:
            value = self._guard_constant(node, expr.args[0], before, depth + 1)
            if value is None:
                return None
            source, destination, signed = conversion
            if signed == "S" and value >> (source - 1):
                value -= 1 << source
            return value & ((1 << destination) - 1)
        if masked:
            left, right = (
                self._guard_constant(node, arg, before, depth + 1) for arg in expr.args
            )
            return None if left is None or right is None else left & right
        return None

    def _unsigned_operand(self, node, expr, before, depth=0):
        """Remove only injective zero extensions and redundant upper masks."""

        key = node, expr, before
        if key in self._guard_operands:
            return self._guard_operands[key]
        result = self._normalize_unsigned_operand(node, expr, before, depth)
        if result[0] is not None and not self.exhausted:
            self._guard_operands[key] = result
        return result

    def _normalize_unsigned_operand(self, node, expr, before, depth):
        if depth >= 32 or not self._step():
            return None, before
        expr, before = self._definition(node, expr, before)
        if expr is None:
            return None, before
        conversion = _vex_width_conversion(expr)
        if conversion is not None and conversion[2] == "U":
            if conversion[0] < conversion[1]:
                return self._unsigned_operand(node, expr.args[0], before, depth + 1)
        if isinstance(expr, pyvex.expr.Binop) and expr.op.startswith("Iop_And"):
            vex, _ = self._block(node)
            for value, mask in (expr.args, expr.args[::-1]):
                mask_value = self._guard_constant(node, mask, before)
                if mask_value is None:
                    continue
                operand, position = self._unsigned_operand(
                    node, value, before, depth + 1
                )
                if operand is not None:
                    full = (1 << operand.result_size(vex.tyenv)) - 1
                    if mask_value & full == full:
                        return operand, position
        return expr, before

    def _guard_view_key(self, node, expr, before, bits, depth=0):
        """Identify a low view using its reaching definition, not its name.

        Local containing writes and explicit truncations can describe the same
        slice in different VEX forms. Never project away bits from the queried
        value itself: callers first require equivalent operand widths.
        """

        key = node, expr, before, bits
        if key in self._guard_views:
            return self._guard_views[key]
        result = self._build_guard_view_key(node, expr, before, bits, depth)
        if result is not None and not self.exhausted:
            self._guard_views[key] = result
        return result

    def _build_guard_view_key(self, node, expr, before, bits, depth):
        if depth >= 32 or not self._step():
            return None
        expr, before = self._definition(node, expr, before)
        if expr is None:
            return None
        vex, _ = self._block(node)
        source_bits = expr.result_size(vex.tyenv)
        if source_bits < bits:
            return None
        if isinstance(expr, pyvex.expr.Const):
            return "const", bits, expr.con.value & ((1 << bits) - 1)
        if isinstance(expr, pyvex.expr.Get):
            offset = self._low_view_offset(expr.offset, source_bits, bits)
            write = self._nearest_write(node, before, offset, bits)
            if write is None:
                return None
            if write < 0:
                return "get", offset, bits
            value = self._written_view(node, vex.statements[write], offset, bits)
            return (
                None
                if value is None
                else self._guard_view_key(node, value, write, bits, depth + 1)
            )
        if (conversion := _vex_width_conversion(expr)) is not None:
            if bits <= conversion[0]:
                return self._guard_view_key(node, expr.args[0], before, bits, depth + 1)
            key = self._guard_view_key(
                node, expr.args[0], before, conversion[0], depth + 1
            )
            return None if key is None else ("extend", conversion, bits, key)
        if isinstance(expr, pyvex.expr.Binop) and expr.op.startswith("Iop_And"):
            keys = [
                self._guard_view_key(node, arg, before, bits, depth + 1)
                for arg in expr.args
            ]
            return None if None in keys else ("and", bits, *keys)
        return ("expression", id(expr), bits) if source_bits == bits else None

    def _same_guard_value(self, node, expr, before, other, other_before):
        """Match numeric values without confusing narrow views with full reads."""

        expr, before = self._definition(node, expr, before)
        other, other_before = self._definition(node, other, other_before)
        vex, _ = self._block(node)
        if expr is None or other is None:
            return False
        bits = expr.result_size(vex.tyenv)
        if other.result_size(vex.tyenv) == bits:
            # Preserve the inexpensive same-width path used by most proofs.
            # A shared definition observes its operands at one read position.
            if expr is other and before == other_before:
                return True
            if isinstance(expr, pyvex.expr.Get) and isinstance(other, pyvex.expr.Get):
                # Same-width register copying remains the predecessor search's
                # job, not a reason to scan unrelated registers for every guard.
                return self._same_value(node, expr, before, other, other_before, bits)
        expr, before = self._unsigned_operand(node, expr, before)
        other, other_before = self._unsigned_operand(node, other, other_before)
        vex, _ = self._block(node)
        if expr is None or other is None:
            return False
        self._guard_widths.add(other.result_size(vex.tyenv))
        bits = expr.result_size(vex.tyenv)
        if other.result_size(vex.tyenv) != bits:
            return False
        if expr is other and before == other_before:
            return True
        key = self._guard_view_key(node, expr, before, bits)
        return key is not None and key == self._guard_view_key(
            node, other, other_before, bits
        )

    def _edge_domain(self, node, position, offset, bits, domain=None):
        """Carry edge guard constraints into the outgoing register domain.

        Identical temporary definitions are sufficient even after a mask PUT.
        GETs also match when no intervening overlapping write changed their
        value. A bounded guard operand can flow through supported arithmetic
        into the outgoing register: the guard need not observe its final value.
        Signed comparisons only filter already proven finite operands; they
        never invent a domain. Typed predicates connect equivalent zero-extended
        views, but a guard on low bits never bounds unknown high register bits.
        """

        vex, _ = self._block(node)
        write = self._nearest_write(node, position, offset, bits)
        query = pyvex.expr.Get(offset, f"Ity_I{bits}")
        query_position = position
        if write is None:
            return domain
        if write >= 0:
            query = self._written_view(node, vex.statements[write], offset, bits)
            if query is None:
                return domain
            query_position = write
        query, query_position = self._definition(node, query, query_position)
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
            key = node, index
            if key not in self._predicates:
                predicate = self._guard_comparison(node, stmt.guard, index)
                if not self.exhausted:
                    self._predicates[key] = predicate
            else:
                predicate = self._predicates[key]
            if predicate is None:
                continue
            guard, guard_position, inverted = predicate
            taken = (index == position) != inverted
            signed = guard.op.endswith("S")
            guard_bits = guard.args[0].result_size(vex.tyenv)
            selector, bound_expr = guard.args
            bound = self._guard_constant(node, bound_expr, guard_position)
            reversed_operands = False
            if bound is None:
                bound = self._guard_constant(node, selector, guard_position)
                selector = bound_expr
                reversed_operands = True
            if bound is None:
                continue

            def matches(expr, before, other, other_before):
                if signed:
                    return self._same_value(
                        node, expr, before, other, other_before, guard_bits
                    )
                return self._same_guard_value(node, expr, before, other, other_before)

            def constraint():
                # Search predecessors only once the outgoing expression has
                # been shown to depend on this guard. Many guards observe an
                # unrelated register; eager searches waste the shared budget.
                domain_key = node, index, taken
                if domain_key not in self._guard_domains:
                    result = self._comparison_domain(
                        node,
                        selector,
                        guard_position,
                        guard.op,
                        bound,
                        taken,
                        reversed_operands,
                    )
                    # Unknown and interrupted answers are not reusable proofs.
                    if result is not None and not self.exhausted:
                        self._guard_domains[domain_key] = result
                    return result
                return self._guard_domains[domain_key]

            mapped = self._map_guard_domain(
                node,
                query,
                query_position,
                lambda expr, before: matches(expr, before, selector, guard_position),
                constraint,
            )
            if mapped is not None:
                domain = mapped if domain is None else domain & mapped
        return domain

    def _comparison_domain(
        self, node, selector, before, op, bound, taken, reversed_operands
    ):
        """Prove a small interval, or filter an independently finite operand.

        Upper unsigned bounds and equality can bound otherwise unknown inputs.
        Lower bounds, exclusions and signed ordering cannot: they only refine
        an existing finite superset. Filter before arithmetic so a subtraction
        does not turn impossible inputs into wrapped-negative table indices.
        """

        equality = "CmpEQ" in op or "CmpNE" in op
        signed = op.endswith("S")
        if equality and ("CmpEQ" in op) == taken:
            return frozenset({bound})
        if not equality and not signed and taken != reversed_operands:
            upper = bound - int(("CmpLT" in op) == taken)
            if 0 <= upper < MAX_STATIC_JUMPTABLE_ENTRIES:
                return frozenset(range(upper + 1))

        operands = self.values(node, selector, before)
        if operands is None:
            return None
        vex, _ = self._block(node)
        bits = selector.result_size(vex.tyenv)

        def numeric(value):
            return value - (1 << bits) if signed and value >> (bits - 1) else value

        result = set()
        for value in operands:
            if not self._step():
                return None
            left, right = numeric(value), numeric(bound)
            if reversed_operands:
                left, right = right, left
            if equality:
                passes = (left == right) == ("CmpEQ" in op)
            else:
                passes = left < right if "CmpLT" in op else left <= right
            if passes == taken:
                result.add(value)
        return frozenset(result)

    def _map_guard_domain(self, node, expr, before, matches, constraint, depth=0):
        """Replay local arithmetic over a guard-proven operand domain.

        For example, a borrow guard proves the old value is below N, although
        the outgoing PUT stores old-N. Preserve machine-width wrapping when
        mapping that set; a later add of N can then restore the small domain.
        This walks only local definitions, never loop iterations or unknown
        memory. The constraint is lazy: a predecessor search is needed only
        when a matched operand occurs. An unrelated input, unsupported operation
        or exhausted budget declines the proof rather than dropping values.
        """

        if depth >= 32 or not self._step():
            return None
        expr, before = self._definition(node, expr, before)
        if expr is None:
            return None
        if matches(expr, before):
            constraint = constraint()
            if constraint is None:
                return None
            vex, _ = self._block(node)
            # A widened equality can request an impossible narrow value.
            # Filter it, rather than wrapping it into a different selector.
            operand, _ = self._unsigned_operand(node, expr, before)
            if operand is None:
                return None
            limit = 1 << operand.result_size(vex.tyenv)
            return frozenset(v for v in constraint if v < limit)
        if isinstance(expr, pyvex.expr.Const):
            return frozenset({expr.con.value})
        if (conversion := _vex_width_conversion(expr)) is not None:
            values = self._map_guard_domain(
                node, expr.args[0], before, matches, constraint, depth + 1
            )
            if values is None:
                return None
            source, destination, signed = conversion
            return frozenset(
                (
                    value - (1 << source)
                    if signed == "S" and value >> (source - 1)
                    else value
                )
                & ((1 << destination) - 1)
                for value in values
            )
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
            value = self._written_view(node, vex.statements[index], offset, bits)
            if value is not None:
                result = self.values(node, value, index, depth + 1)
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
