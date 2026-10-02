"""Query-driven correlations between at most two bounded VEX values.

Backward substitution preserves copies, swaps and old temporary reads. Small
finite definitions become typed atoms, rather than independent domains at each
use. Edge predicates filter assignments to those atoms before arithmetic is
evaluated. Unknown paths decline the refinement; there is no path unrolling,
mutable-memory model, or unconstrained symbolic solver.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import TYPE_CHECKING

import pyvex

from bingraph.cfg.jumps import MAX_STATIC_JUMPTABLE_ENTRIES, _vex_width_conversion

if TYPE_CHECKING:
    from bingraph.cfg_extract.shared_facts import PredecessorFacts


@dataclass(frozen=True, eq=False)
class _Term:
    """Interned DAG node: hashing never recursively expands copied subtrees."""

    parts: tuple
    depth: int

    def __getitem__(self, index):
        return self.parts[index]


class RelationalValues:
    """Share completed expressions and queries within one fact snapshot."""

    def __init__(self, facts: PredecessorFacts):
        self.facts = facts
        self.terms = {}
        self.atoms = {}
        self.answers = {}
        self.active = set()
        self._intern = {}
        self._leaf_cache = {}
        self._substitutions = {}

    def _make(self, *parts):
        if parts in self._intern:
            return self._intern[parts]
        args = (
            parts[-1]
            if parts[0] in {"op", "cmp"}
            else (parts[-1],)
            if parts[0] in {"convert", "not"}
            else ()
        )
        depth = 1 + max((arg.depth for arg in args), default=0)
        if depth > 32 or not self.facts._step():
            return None
        term = _Term(parts, depth)
        self._intern[parts] = term
        return term

    def values(self, node, expr, before):
        key = node, expr, before
        if key in self.active or len(self.active) >= 32:
            return None
        if key in self.answers:
            return self.answers[key]
        self.active.add(key)
        term = self._term(node, expr, before)
        result = (
            None if term is None else self._walk(node, before, term, frozenset(), 0, {})
        )
        self.active.remove(key)
        if not self.facts.exhausted:
            self.answers[key] = result
        return result

    def _term(self, node, expr, before, depth=0):
        """Expand local definitions, retaining entry reads and finite atoms."""

        facts = self.facts
        if depth >= 32 or not facts._step():
            return None
        expr, before = facts._definition(node, expr, before)
        if expr is None:
            return None
        key = node, expr, before
        if key in self.terms:
            return self.terms[key]
        vex, _ = facts._block(node)
        bits = expr.result_size(vex.tyenv)
        result = None
        if isinstance(expr, pyvex.expr.Const):
            result = self._make("const", expr.con.value)
        elif isinstance(expr, pyvex.expr.Get):
            write = facts._nearest_write(node, before, expr.offset, bits)
            if write == -1:
                result = self._make("get", expr.offset, bits)
            elif write is not None:
                value = facts._written_view(
                    node, vex.statements[write], expr.offset, bits
                )
                if value is not None:
                    result = self._term(node, value, write, depth + 1)
        elif (conversion := _vex_width_conversion(expr)) is not None:
            arg = self._term(node, expr.args[0], before, depth + 1)
            if arg is not None:
                result = self._make("convert", conversion, arg)
        elif isinstance(expr, pyvex.expr.Binop) and expr.op in {
            f"Iop_{op}{bits}" for op in ("Add", "Sub", "And", "Or", "Shl", "Shr")
        }:
            # A small mask can bound an unknown input without enumerating its
            # machine-width domain. Every copy of this definition shares the
            # same atom; its operand is NOT equated with the masked result.
            masks = [facts._guard_constant(node, arg, before) for arg in expr.args]
            if expr.op == f"Iop_And{bits}" and any(
                mask is not None and mask < MAX_STATIC_JUMPTABLE_ENTRIES
                for mask in masks
            ):
                domain = facts.values(node, expr, before)
                if domain is not None:
                    self.atoms[key] = domain
                    result = self._make("atom", key)
            else:
                args = tuple(
                    self._term(node, arg, before, depth + 1) for arg in expr.args
                )
                if None not in args:
                    result = self._make("op", expr.op, bits, args)
        else:
            # Static scalar loads may be constants. Never turn an unknown load,
            # unsupported operation or private-frame value into a finite atom.
            value = facts.value(node, expr, before)
            if value is not None:
                result = self._make("const", value)
        if result is not None and not facts.exhausted:
            self.terms[key] = result
        return result

    def _leaves(self, term, kind):
        key = term, kind
        if key in self._leaf_cache:
            return self._leaf_cache[key]
        if term[0] == kind:
            result = frozenset({term})
        elif term[0] in {"convert", "not"}:
            result = self._leaves(term[-1], kind)
        elif term[0] in {"op", "cmp"}:
            result = frozenset().union(*(self._leaves(arg, kind) for arg in term[-1]))
        else:
            result = frozenset()
        self._leaf_cache[key] = result
        return result

    def _substitute(self, term, node, position):
        """Move entry reads together across one exact predecessor edge."""

        key = term, node, position
        if key in self._substitutions:
            return self._substitutions[key]
        if not self.facts._step():
            return None
        result = self._rewrite(term, node, position)
        if not self.facts.exhausted:
            self._substitutions[key] = result
        return result

    def _rewrite(self, term, node, position):
        if term[0] == "get":
            return self._term(
                node, pyvex.expr.Get(term[1], f"Ity_I{term[2]}"), position
            )
        if term[0] in {"convert", "not"}:
            arg = self._substitute(term[-1], node, position)
            return None if arg is None else self._make(*term[:-1], arg)
        if term[0] in {"op", "cmp"}:
            args = tuple(self._substitute(arg, node, position) for arg in term[-1])
            return None if None in args else self._make(*term[:-1], args)
        return term

    def _guards(self, node, position):
        facts = self.facts
        vex, _ = facts._block(node)
        guards = set()
        for index, stmt in enumerate(vex.statements[: position + 1]):
            if not isinstance(stmt, pyvex.stmt.Exit) or stmt.jumpkind != "Ijk_Boring":
                continue
            if not facts._step():
                break
            key = node, index
            if key not in facts._predicates:
                predicate = facts._guard_comparison(node, stmt.guard, index)
                if not facts.exhausted:
                    facts._predicates[key] = predicate
            predicate = facts._predicates.get(key)
            if predicate is None:
                continue
            expr, before, inverted = predicate
            args = tuple(self._term(node, arg, before) for arg in expr.args)
            if None in args:
                continue
            bits = expr.args[0].result_size(vex.tyenv)
            guard = self._make("cmp", expr.op, bits, args)
            if guard is not None and (index == position) == inverted:
                guard = self._make("not", guard)
            if guard is not None:
                guards.add(guard)
        return frozenset(guards)

    def _walk(self, node, position, query, guards, depth, cache, active=frozenset()):
        """Union independently refined paths, never omit an unknown input."""

        facts = self.facts
        # Even a different statement position in a revisited block denotes a
        # new loop iteration. Static definition identities must not equate the
        # masked values produced by two executions of that same instruction.
        if node in active or depth >= 32 or not facts._step():
            return None
        active = active | {node}
        guards |= self._guards(node, position)
        roots = self._leaves(query, "get")
        if not roots:
            return self._evaluate(query, guards)
        key = node, position, query, guards
        if key in cache:
            return cache[key]
        # Mark active states unknown. A repeated state or loop-carried rewrite
        # cannot justify excluding an iteration from the joint domain.
        cache[key] = None
        incoming = [facts._inputs(node, root[1], root[2]) for root in roots]
        if any(paths is None or paths[1] for paths in incoming):
            return None
        paths = incoming[0][0]
        if any(other[0] != paths for other in incoming[1:]):
            return None
        result = frozenset()
        transportable = {
            guard
            for guard in guards
            if all(
                facts._inputs(node, root[1], root[2]) == (paths, set())
                for root in self._leaves(guard, "get")
            )
        }
        for predecessor, before in paths:
            term = self._substitute(query, predecessor, before)
            if term is None:
                return None
            # Unsupported predicates are optional precision, unlike query
            # inputs: omitting such a predicate only keeps additional values.
            constraints = frozenset(
                rewritten
                # A predicate's extra inputs have their own ABI/barrier rules.
                # Preserved query registers cannot preserve a volatile flag or
                # operand merely because it was not part of the query.
                for guard in transportable
                if (rewritten := self._substitute(guard, predecessor, before))
                is not None
            )
            values = self._walk(
                predecessor, before, term, constraints, depth + 1, cache, active
            )
            if values is None or len(result | values) > MAX_STATIC_JUMPTABLE_ENTRIES:
                return None
            result |= values
        cache[key] = result
        return result

    def _evaluate(self, query, guards):
        """Enumerate only query atoms; unrelated predicates cannot narrow them."""

        atoms = tuple(self._leaves(query, "atom"))
        if len(atoms) > 2:
            return None
        domains = [self.atoms[atom[1]] for atom in atoms]
        if (
            len(domains) == 2
            and len(domains[0]) * len(domains[1]) > MAX_STATIC_JUMPTABLE_ENTRIES
        ):
            return None
        known = set(atoms)
        predicates = [
            guard
            for guard in guards
            if not self._leaves(guard, "get") and self._leaves(guard, "atom") <= known
        ]
        result = set()
        for assignment in product(*domains):
            if not self.facts._step():
                return None
            cache = dict(zip(atoms, assignment, strict=True))
            truth = [self._eval(guard, cache) for guard in predicates]
            if None in truth:
                return None
            if all(truth):
                value = self._eval(query, cache)
                if value is None:
                    return None
                result.add(value)
        return frozenset(result)

    def _eval(self, term, cache):
        if term in cache:
            return cache[term]
        if not self.facts._step():
            return None
        kind = term[0]
        if kind == "const":
            value = term[1]
        elif kind in {"convert", "not"}:
            value = self._eval(term[-1], cache)
            if value is None:
                return None
            if kind == "not":
                value = int(not value)
            else:
                source, destination, signed = term[1]
                if signed == "S" and value >> (source - 1):
                    value -= 1 << source
                value &= (1 << destination) - 1
        else:
            op, bits, args = term[1:]
            left, right = (self._eval(arg, cache) for arg in args)
            if left is None or right is None:
                return None
            if kind == "cmp":
                if op.endswith("S"):
                    left = left - (1 << bits) if left >> (bits - 1) else left
                    right = right - (1 << bits) if right >> (bits - 1) else right
                value = int(
                    left == right
                    if "CmpEQ" in op
                    else left != right
                    if "CmpNE" in op
                    else left < right
                    if "CmpLT" in op
                    else left <= right
                )
            else:
                values = self.facts._combine(
                    op, bits, frozenset({left}), frozenset({right})
                )
                if values is None:
                    return None
                value = next(iter(values))
        cache[term] = value
        return value
