"""Bounded joint domains using angr's VEX semantics and Claripy constraints.

Only already-finite definitions become symbolic atoms. Exact-edge traversal
still owns aliases, call barriers and loop rejection; angr evaluates expressions,
not callees or whole functions. Unsupported guards lose precision, never paths.
"""

from __future__ import annotations

from itertools import product
from math import prod
from types import SimpleNamespace
from typing import TYPE_CHECKING

from angr import sim_options
from angr.engines.vex.claripy.datalayer import ClaripyDataMixin
from angr.errors import SimError
import claripy
import pyvex

from bingraph.cfg.jumps import MAX_STATIC_JUMPTABLE_ENTRIES

if TYPE_CHECKING:
    from bingraph.cfg.shared_facts import PredecessorFacts


class RelationalValues(ClaripyDataMixin):
    """Delegate expressions to angr without creating execution states.

    State-dependent reads use existing must facts. Claripy DAGs replace the
    handwritten term language, arithmetic and Cartesian predicate interpreter.
    """

    def __init__(self, facts: PredecessorFacts):
        super().__init__(facts.project)
        self.facts = facts
        self.state = SimpleNamespace(options={sim_options.EXTENDED_IROP_SUPPORT})
        self.terms = {}
        self.atoms = {}
        self.reads = {}
        self.answers = {}
        self.active = set()
        self._substitutions = {}
        self._depth = 0
        self.context = None, 0
        self._guarding = False

    def values(self, node, expr, before):
        key = node, expr, before
        if key in self.active or len(self.active) >= 32 or self.facts.exhausted:
            return None
        if key in self.answers:
            return self.answers[key]
        self.active.add(key)
        try:
            query = self._term(node, expr, before)
            result = (
                None
                if query is None
                else self._walk(node, before, query, frozenset(), {})
            )
        finally:
            self.active.remove(key)
        if not self.facts.exhausted:
            self.answers[key] = result
        return result

    def _term(self, node, expr, before):
        if self._depth >= 32 or not self.facts._step():
            return None
        expr, before = self.facts._definition(node, expr, before)
        if not isinstance(
            expr,
            (
                pyvex.expr.Const,
                pyvex.expr.Get,
                pyvex.expr.Load,
                pyvex.expr.Unop,
                pyvex.expr.Binop,
                pyvex.expr.Triop,
                pyvex.expr.Qop,
                pyvex.expr.CCall,
                pyvex.expr.ITE,
            ),
        ):
            return None
        key = node, expr, before
        if key in self.terms:
            return self.terms[key]
        previous = self.context
        self.context = node, before
        self._depth += 1
        try:
            result = super()._handle_vex_expr(expr)
        except (SimError, claripy.ClaripyError, KeyError, NotImplementedError):
            result = None
        finally:
            self._depth -= 1
            self.context = previous
        if (
            isinstance(result, claripy.ast.BV)
            and result.depth <= 32
            and not self.facts.exhausted
        ):
            if not self._guarding:
                self.terms[key] = result
            return result
        return None

    def _handle_vex_expr(self, expr):
        node, before = self.context
        result = self._term(node, expr, before)
        if result is None:
            raise SimError("Unknown expression in bounded joint proof")
        return result

    def _handle_vex_expr_Get(self, expr):
        node, before = self.context
        vex, _ = self.facts._block(node)
        bits = expr.result_size(vex.tyenv)
        write = self.facts._nearest_write(node, before, expr.offset, bits)
        if write is None:
            raise SimError("Interrupted register search")
        if write >= 0:
            value = self.facts._written_view(
                node, vex.statements[write], expr.offset, bits
            )
            if value is None:
                raise SimError("Overlapping register write")
            return self._term(node, value, write)
        # Preserve the consumed view across edges. Reaching containing PUTs
        # establish alias relationships through the existing _written_view.
        name = f"joint_get_{expr.offset}_{bits}"
        self.reads[name] = expr.offset, bits
        return claripy.BVS(name, bits, explicit_name=True)

    def _handle_vex_expr_Load(self, expr):
        node, before = self.context
        value = self.facts.value(node, expr, before)
        if value is None:
            raise SimError("Unproven memory read")
        vex, _ = self.facts._block(node)
        return claripy.BVV(value, expr.result_size(vex.tyenv))

    def _handle_vex_expr_Op(self, expr):
        node, before = self.context
        if self._guarding:
            # A constraint must keep its input dependencies. Query definitions
            # already cached as finite atoms are reused by _term above.
            return super()._handle_vex_expr_Op(expr)
        if expr.op.startswith("Iop_And") and any(
            (mask := self.facts._guard_constant(node, arg, before)) is not None
            and mask < MAX_STATIC_JUMPTABLE_ENTRIES
            for arg in expr.args
        ):
            domain = self.facts.values(node, expr, before)
            if domain is None:
                raise SimError("Unproven finite definition")
            name = f"joint_atom_{id(node)}_{before}_{id(expr)}"
            self.atoms[name] = domain
            vex, _ = self.facts._block(node)
            return claripy.BVS(name, expr.result_size(vex.tyenv), explicit_name=True)
        return super()._handle_vex_expr_Op(expr)

    def _handle_vex_expr_CCall(self, expr):
        if expr.cee.name not in {
            "amd64g_calculate_condition",
            "x86g_calculate_condition",
        }:
            raise SimError("Unsupported condition helper")
        node, before = self.context
        # A concrete operation selector is required by angr's flag helpers.
        # Other inputs remain symbolic and retain their own ABI/barrier rules.
        operation = self.facts.value(node, expr.args[1], before)
        if operation is None:
            raise SimError("Unknown flag producer")
        vex, _ = self.facts._block(node)
        args = [
            self._handle_vex_expr(arg)
            if i != 1
            else claripy.BVV(operation, arg.result_size(vex.tyenv))
            for i, arg in enumerate(expr.args)
        ]
        return self._perform_vex_expr_CCall(expr.cee.name, expr.retty, args)

    def _substitute(self, term, node, position):
        key = term, node, position
        if key not in self._substitutions:
            replacements = {}
            for name in term.variables & self.reads.keys():
                offset, bits = self.reads[name]
                value = self._term(
                    node, pyvex.expr.Get(offset, f"Ity_I{bits}"), position
                )
                if value is None:
                    return None
                replacements[claripy.BVS(name, bits, explicit_name=True).hash()] = value
            self._substitutions[key] = claripy.replace_dict(term, replacements)
        return self._substitutions[key]

    def _guards(self, node, position):
        vex, _ = self.facts._block(node)
        for index, stmt in enumerate(vex.statements[: position + 1]):
            if isinstance(stmt, pyvex.stmt.Exit) and stmt.jumpkind == "Ijk_Boring":
                previous, self._guarding = self._guarding, True
                try:
                    value = self._term(node, stmt.guard, index)
                finally:
                    self._guarding = previous
                if value is not None:
                    yield value != 0 if index == position else value == 0

    def _walk(self, node, position, query, guards, cache, active=frozenset()):
        if node in active or len(active) >= 32 or not self.facts._step():
            return None
        guards |= frozenset(self._guards(node, position))
        roots = query.variables & self.reads.keys()
        if not roots:
            return self._evaluate(query, guards)
        key = node, position, query, guards
        if key in cache:
            return cache[key]
        cache[key] = None
        incoming = [
            self.facts._inputs(node, *self.reads[name]) for name in sorted(roots)
        ]
        if any(paths is None or paths[1] for paths in incoming):
            return None
        paths = incoming[0][0]
        if any(other[0] != paths for other in incoming[1:]):
            return None
        transportable = [
            guard
            for guard in guards
            if all(
                self.facts._inputs(node, *self.reads[name]) == (paths, set())
                for name in guard.variables & self.reads.keys()
            )
        ]
        result = frozenset()
        for predecessor, before in paths:
            term = self._substitute(query, predecessor, before)
            if term is None:
                return None
            constraints = frozenset(
                rewritten
                for guard in transportable
                if (rewritten := self._substitute(guard, predecessor, before))
                is not None
            )
            values = self._walk(
                predecessor, before, term, constraints, cache, active | {node}
            )
            if values is None or len(result | values) > MAX_STATIC_JUMPTABLE_ENTRIES:
                return None
            result |= values
        cache[key] = result
        return result

    def _evaluate(self, query, guards):
        names = sorted(query.variables)
        if len(names) > 2 or any(name not in self.atoms for name in names):
            return None
        assignments = prod(len(self.atoms[name]) for name in names)
        if assignments > MAX_STATIC_JUMPTABLE_ENTRIES:
            return None
        # Domains are already finite: concrete Claripy evaluation avoids SMT
        # state creation, timeouts and repeated model enumeration. Angr still
        # owns every VEX operation and flag helper used to build these DAGs.
        atoms = {leaf.args[0]: leaf for leaf in query.leaf_asts() if leaf.op == "BVS"}
        predicate = claripy.And(
            *(guard for guard in guards if guard.variables <= set(names))
        )
        result = set()
        for assignment in product(*(sorted(self.atoms[name]) for name in names)):
            if not self.facts._step():
                return None
            replacements: dict[int, claripy.ast.Base] = {
                atoms[name].hash(): claripy.BVV(value, atoms[name].length)
                for name, value in zip(names, assignment, strict=True)
            }
            try:
                if claripy.backends.concrete.eval(
                    claripy.replace_dict(predicate, replacements), 1
                )[0]:
                    value = claripy.replace_dict(query, replacements)
                    result.add(claripy.backends.concrete.eval(value, 1)[0])
            except claripy.ClaripyError:
                return None
        return frozenset(result)
