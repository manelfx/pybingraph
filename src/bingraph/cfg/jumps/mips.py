"""MIPS PIC table proofs and bounded private-frame register propagation.

Two complementary mechanisms live here. Register-transfer propagation replays
decoded blocks with small finite register domains and private-frame saves, so
PIC callees spilled around calls can be recovered. Table planning recognizes
GP-relative offsets whose selector bounds and table base are established by
local guards or agreeing predecessor definitions.

Both start with helpers.abi's entry GP proof, but neither treats that root as
evidence that GP survives an arbitrary call. Register propagation kills call
clobbers and overlapping saves; computed memory reads must stay in immutable
data, while GP-relative linkage reads use the existing PIC linkage assumptions.
The private-frame model assumes ordinary argument pointers do not alias this
function's own stack saves; it is not general-purpose memory alias analysis.

These are fallbacks to shared snapshot facts and local transfer decoding, not
a second CFG builder. Return target maps or table plans to the caller, which
owns executable-target validation, leader insertion, redecoding and statistics.
The register pass has explicit block, value, worklist and cycle-proof limits;
table counts use the common row cap and never guessed table boundaries.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Iterable, Set
from dataclasses import dataclass
from typing import Any

import pyvex
from angr import Project

from bingraph.cfg.calls import target_is_known_nonreturning
from bingraph.cfg.graph import CFGGraph
from bingraph.cfg.graph import node_intersects_bounds as _node_intersects_bounds
from bingraph.cfg.graph import (
    node_is_materialized_cfg_node as _node_is_materialized_cfg_node,
)
from bingraph.cfg.graph import node_vex as _node_vex
from bingraph.cfg.jumps.expressions import (
    MAX_STATIC_JUMPTABLE_ENTRIES,
    _vex_expr_key,
    _vex_masked_index_values,
    _vex_normalized_table_entry_load,
    _vex_register_with_displacement,
    static_jump_target_rejection_reason,
)
from bingraph.cfg.jumps.guards import (
    _vex_guard_upper_bound,
    _vex_guarded_expression_values,
)
from bingraph.cfg.models import (
    BlockSpec,
    FunctionBounds,
    StaticJumpTable,
    StaticJumpTablePlan,
)
from bingraph.helpers.abi import mips_entry_global_pointer
from bingraph.helpers.vex import (
    STATIC_INTEGER_BYTES,
    _resolve_vex_expr,
    _vex_add_terms,
    _vex_const_value,
    _vex_get_key,
    _vex_width_conversion,
    read_static_integer,
    temporary_definitions,
)

MAX_ABI_STATIC_TARGET_WORKLIST_UPDATES = 4096
MAX_ABI_STATIC_TARGET_VALUES = 8
MAX_MIPS_PIC_TARGET_BLOCKS = 512
MAX_MIPS_NORETURN_CYCLE_PROOFS = 8


def _abi_static_target_is_valid(project: Project, target: int) -> bool:
    """Check that a propagated value can become a concrete transfer target.

    Use the normal executable-region validator for loaded objects. For CLE's
    synthetic extern object, additionally allow a loader symbol identified as
    a function: imported calls can legitimately target those addresses. A
    numeric value alone, an unmapped address or a data symbol is insufficient.
    """

    rejection = static_jump_target_rejection_reason(project, target)
    if rejection is None:
        return True
    if rejection != "synthetic":
        return False
    symbol = project.loader.find_symbol(target)
    return bool(symbol is not None and getattr(symbol, "is_function", False))


def _abi_successor_addrs(
    blocks: dict[int, BlockSpec], block: BlockSpec
) -> tuple[int, ...]:
    """Select decoded successors for propagation inside this function.

    Calls contribute only their continuation, never the callee's blocks;
    other transfers contribute their direct targets and optional fallthrough.
    Keep only addresses already present in ``blocks``. This builds the local
    analysis graph without introducing speculative targets or following calls.
    """

    if block.jumpkind == "Ijk_Call":
        candidates = (block.fallthrough_addr,)
    else:
        candidates = (*block.direct_targets, block.fallthrough_addr)
    return tuple(target for target in candidates if target in blocks)


def abi_static_register_transfer_targets(
    project: Project,
    bounds: FunctionBounds,
    blocks: dict[int, BlockSpec],
) -> tuple[dict[int, tuple[int, ...]], bool, bool]:
    """Resolve MIPS PIC transfers using its private-frame must analysis.

    AMD64 transfers use shared predecessor facts in the custom builder.
    MIPS retains this pass until the shared engine supports frame-private saves.

    Return ``(targets_by_source, budget_exhausted, analysis_attempted)``. An
    empty map need not mean exhaustion: unsupported architectures or missing
    entry evidence can skip the pass, and an attempted proof may find nothing.
    Never publish partial results when the worklist budget is exhausted.
    """

    if project.arch.name.startswith("MIPS"):
        return _mips_pic_register_transfer_targets(project, bounds, blocks)

    return {}, False, False


@dataclass
class _MipsPicState:
    """Known finite domains at one decoded block boundary.

    ``registers`` uses VEX offsets; ``stack`` uses signed displacements from
    the current SP for word-sized private saves. A missing key means unknown,
    not zero or unreachable. A present domain may contain several possible
    values, but only when each contributing incoming state knows that key.
    Frame changes invalidate stack entries rather than rebasing guessed saves.
    """

    registers: dict[int, frozenset[int]]
    stack: dict[int, frozenset[int]]


def _mips_finite_union(values: Iterable[frozenset[int]]) -> frozenset[int] | None:
    """Combine finite alternatives without exceeding the register-domain cap.

    The union represents possible values, not a choice of the most plausible
    target. Return ``None`` for an empty or oversized union so callers discard
    the fact instead of truncating it. Incoming-path completeness is checked
    by the state join; this helper only combines supplied known domains.
    """

    merged: set[int] = set()
    for item in values:
        merged.update(item)
        if len(merged) > MAX_ABI_STATIC_TARGET_VALUES:
            return None
    return frozenset(merged) if merged else None


def _mips_join_states(states: Iterable[_MipsPicState]) -> _MipsPicState:
    """Merge incoming states by intersecting keys and unioning their domains.

    Keep a register or save only if every supplied predecessor has a fact for
    it. Different known values remain alternatives; one missing fact or an
    oversized union removes the key. During worklist iteration only processed
    predecessors are supplied, so final transfer proofs separately require
    every immediate predecessor to have been processed.
    """

    incoming = tuple(states)
    if not incoming:
        return _MipsPicState({}, {})

    def join_map(name: str) -> dict[int, frozenset[int]]:
        """Apply the same completeness rule to registers and private saves."""

        maps = [getattr(state, name) for state in incoming]
        common = set.intersection(*(set(mapping) for mapping in maps))
        joined = {}
        for key in common:
            values = _mips_finite_union(mapping[key] for mapping in maps)
            if values is not None:
                joined[key] = values
        return joined

    return _MipsPicState(join_map("registers"), join_map("stack"))


def _mips_register_displacement(
    key: tuple[Any, ...] | None, offset: int, bits: int
) -> int | None:
    """Recognize a keyed full-width base register plus one signed constant.

    Accept the register alone or either operand order of an addition. Convert
    the constant's bit pattern to a signed byte displacement for stack/GOT
    addressing. Reject casts, extra operands and other arithmetic rather than
    inferring an alias from a partially understood address expression.
    """

    if key == ("get", offset, bits):
        return 0
    if key is None or key[:2] != ("binop", f"Iop_Add{bits}"):
        return None
    left, right = key[2:]
    for base, displacement in ((left, right), (right, left)):
        if base == ("get", offset, bits) and displacement[0] == "const":
            value = displacement[1]
            return value if value < 1 << (bits - 1) else value - (1 << bits)
    return None


def _mips_pic_transfer(
    project: Project,
    block: BlockSpec,
    incoming: _MipsPicState,
    gp_offset: int,
    sp_offset: int,
    preserved: Set[int],
) -> tuple[_MipsPicState, frozenset[int] | None]:
    """Replay one exact decoded block and return its state and target domain.

    Lift the supplied byte span with instruction optimization disabled so
    delay-slot writes are included and temporaries retain read-time values.
    Evaluate supported word-width arithmetic, private stack reloads and known
    linkage/static loads over capped finite domains. SP changes clear saves;
    overlapping stores kill them before a new word-sized save is recorded.

    Extract a supported register-carried NEXT target before killing call-
    clobbered registers. Calls preserve only the supplied ABI offsets, while
    private saves remain subject to the frame-privacy assumption. Unsupported
    memory side effects, failed lifting or writes after an early side exit
    invalidate the whole replay: one final state cannot describe both paths.
    Return unknown domains as ``None``/missing keys, never as guessed values.
    """

    try:
        vex = project.factory.block(
            block.addr, size=block.size, strict_block_end=True, cross_insn_opt=False
        ).vex
    except Exception:
        return _MipsPicState({}, {}), None

    bits = project.arch.bits
    mask = (1 << bits) - 1
    definitions = temporary_definitions(vex)
    # These effects cannot be represented by this word-save model. Declining
    # the block is safer than carrying old facts through an unmodeled write.
    if any(
        isinstance(
            statement,
            (pyvex.stmt.Dirty, pyvex.stmt.CAS, pyvex.stmt.LLSC, pyvex.stmt.StoreG),
        )
        for statement in vex.statements
    ):
        return _MipsPicState({}, {}), None
    # A side exit observes an earlier state. Later stores/register writes
    # would require path-specific transfer states, which this pass does not keep.
    for index, statement in enumerate(vex.statements):
        if isinstance(statement, pyvex.stmt.Exit) and any(
            isinstance(later, pyvex.stmt.Store)
            or isinstance(later, pyvex.stmt.Put)
            and later.offset != project.arch.ip_offset
            for later in vex.statements[index + 1 :]
        ):
            return _MipsPicState({}, {}), None
    registers = dict(incoming.registers)
    stack = dict(incoming.stack)
    temps: dict[int, frozenset[int]] = {}

    def evaluate(expr) -> frozenset[int] | None:
        """Evaluate supported word expressions using the current replay state.

        Temporaries freeze earlier reads; register GETs observe current writes.
        Unknown operands propagate unknown except for a provably finite mask.
        Every admitted memory address must be readable, and computed non-GP
        loads additionally require an entirely immutable mapped word.
        """

        if isinstance(expr, pyvex.expr.RdTmp):
            return temps.get(expr.tmp)
        if isinstance(expr, pyvex.expr.Const):
            return frozenset({expr.con.value & mask})
        if isinstance(expr, pyvex.expr.Get):
            return (
                registers.get(expr.offset)
                if expr.result_size(vex.tyenv) == bits
                else None
            )
        if isinstance(expr, pyvex.expr.Binop) and expr.op == f"Iop_And{bits}":
            left = evaluate(expr.args[0])
            right = evaluate(expr.args[1])
            if left is not None and right is not None:
                return _mips_finite_union(
                    frozenset({a & b}) for a in left for b in right
                )
            masked = _vex_masked_index_values(expr, definitions)
            if masked is not None and len(masked) <= MAX_ABI_STATIC_TARGET_VALUES:
                return frozenset(masked)
            return None
        if isinstance(expr, pyvex.expr.Binop) and expr.op == f"Iop_Shl{bits}":
            left = evaluate(expr.args[0])
            right = evaluate(expr.args[1])
            if left is None or right is None:
                return None
            return _mips_finite_union(
                frozenset({(a << b) & mask}) for a in left for b in right
            )
        if isinstance(expr, pyvex.expr.Binop) and expr.op in {
            f"Iop_Add{bits}",
            f"Iop_Sub{bits}",
        }:
            left = evaluate(expr.args[0])
            right = evaluate(expr.args[1])
            if left is None or right is None:
                return None
            operation = (
                (lambda a, b: a + b)
                if expr.op == f"Iop_Add{bits}"
                else (lambda a, b: a - b)
            )
            return _mips_finite_union(
                frozenset({operation(a, b) & mask}) for a in left for b in right
            )
        if not isinstance(expr, pyvex.expr.Load):
            return None
        if expr.result_size(vex.tyenv) != bits:
            return None
        key = _vex_expr_key(expr.addr, definitions)
        stack_offset = _mips_register_displacement(key, sp_offset, bits)
        if stack_offset is not None:
            return stack.get(stack_offset)
        gp_displacement = _mips_register_displacement(key, gp_offset, bits)
        if gp_displacement is not None:
            gp_values = registers.get(gp_offset)
            if gp_values is None:
                return None
            addresses = frozenset((gp + gp_displacement) & mask for gp in gp_values)
        else:
            addresses = evaluate(expr.addr)
            # Computed table reads must remain entirely in immutable data.
            if addresses is None or any(
                (section := project.loader.find_section_containing(addr)) is None
                or section.is_writable
                or addr + project.arch.bytes - 1 > section.max_addr
                for addr in addresses
            ):
                return None
        byteorder = "little" if expr.end == "Iend_LE" else "big"
        values: set[int] = set()
        for addr in addresses:
            try:
                value = int.from_bytes(
                    project.loader.memory.load(addr, project.arch.bytes), byteorder
                )
            except Exception:
                return None
            values.add(value)
        return _mips_finite_union(frozenset({value}) for value in values)

    for statement in vex.statements:
        if isinstance(statement, pyvex.stmt.WrTmp):
            value = evaluate(statement.data)
            if value is not None:
                temps[statement.tmp] = value
        elif isinstance(statement, pyvex.stmt.Store):
            key = _vex_expr_key(statement.addr, definitions)
            displacement = _mips_register_displacement(key, sp_offset, bits)
            if displacement is None:
                # The ABI treats this function's own stack frame as private.
                continue
            size = statement.data.result_size(vex.tyenv) // 8
            for saved_offset in tuple(stack):
                if (
                    displacement < saved_offset + project.arch.bytes
                    and saved_offset < displacement + size
                ):
                    stack.pop(saved_offset)
            value = evaluate(statement.data)
            if statement.data.result_size(vex.tyenv) == bits and value is not None:
                stack[displacement] = value
        elif isinstance(statement, pyvex.stmt.Put):
            if statement.offset == sp_offset:
                stack.clear()
            value = (
                evaluate(statement.data)
                if statement.data.result_size(vex.tyenv) == bits
                else None
            )
            if value is None:
                registers.pop(statement.offset, None)
            else:
                registers[statement.offset] = value

    target_expr = _resolve_vex_expr(vex.next, definitions)
    # Capture the callee before applying its clobbers to the return state.
    target = evaluate(vex.next) if isinstance(target_expr, pyvex.expr.Get) else None
    if block.jumpkind == "Ijk_Call":
        registers = {
            offset: value for offset, value in registers.items() if offset in preserved
        }
    return _MipsPicState(registers, stack), target


def _mips_pic_register_transfer_targets(
    project: Project, bounds: FunctionBounds, blocks: dict[int, BlockSpec]
) -> tuple[dict[int, tuple[int, ...]], bool, bool]:
    """Propagate private-frame PIC facts and prove complete transfer domains.

    Validate the decoded entry against the ABI GP root, then replay a worklist
    rooted only in incoming T9. State joins keep small domains known on all
    supplied paths; final candidates additionally require every predecessor
    to have an output state and every target to pass executable validation.
    Unknown or excessive value sets are discarded, not approximated by a
    subset. Repeated identical output states do not reschedule successors.

    A bounded secondary proof handles a call reached again through its own
    apparent return. Remove only that continuation in an alternative analysis
    and accept a unique known non-returning callee only if it is still reached.
    Both normal and alternative analyses share the function's update budget;
    exhaustion discards all results. Return the same target-map/exhausted/ran
    contract as ``abi_static_register_transfer_targets``.
    """

    if len(blocks) > MAX_MIPS_PIC_TARGET_BLOCKS or bounds.addr not in blocks:
        return {}, False, False
    gp = mips_entry_global_pointer(project, bounds.addr)
    if gp is None:
        return {}, False, False
    registers = project.arch.registers
    try:
        gp_offset = registers["gp"][0]
        sp_offset = registers["sp"][0]
        t9_offset = registers["t9"][0]
        preserved = {
            registers[name][0] for name in (*[f"s{i}" for i in range(8)], "s8", "sp")
        }
    except KeyError:
        return {}, False, False

    candidates = {
        addr
        for addr, block in blocks.items()
        if block.jumpkind in {"Ijk_Call", "Ijk_Boring"}
        and not block.direct_targets
        and not (block.jumpkind == "Ijk_Boring" and block.fallthrough_addr is not None)
    }
    if not candidates:
        return {}, False, False

    successors = {
        addr: _abi_successor_addrs(blocks, block) for addr, block in blocks.items()
    }
    initial = _MipsPicState({t9_offset: frozenset({bounds.addr})}, {})
    # Validate the prologue's GP before an entry-ending call discards it.
    # This temporary exemption is only for validation; propagation below
    # still drops GP at calls and requires an explicit restore to use it again.
    entry_output, _ = _mips_pic_transfer(
        project,
        blocks[bounds.addr],
        initial,
        gp_offset,
        sp_offset,
        preserved | {gp_offset},
    )
    if entry_output.registers.get(gp_offset) != frozenset({gp}):
        return {}, False, True
    updates_left = MAX_ABI_STATIC_TARGET_WORKLIST_UPDATES

    def propagate(edges: dict[int, tuple[int, ...]]):
        """Reach stable boundary states using the shared remaining budget.

        Reject an incoming edge to the entry: its seeded T9 state describes
        a fresh invocation, not a loop back to arbitrary register contents.
        Return states and predecessor sets together for final completeness
        checks; ``None`` means this propagation cannot supply a usable proof.
        """

        nonlocal updates_left
        predecessors: dict[int, set[int]] = {addr: set() for addr in blocks}
        for source, targets in edges.items():
            for target in targets:
                predecessors[target].add(source)
        if predecessors[bounds.addr]:
            return None

        in_states = {bounds.addr: initial}
        out_states: dict[int, _MipsPicState] = {}
        pending = deque([bounds.addr])
        while pending:
            updates_left -= 1
            if updates_left < 0:
                return None
            addr = pending.popleft()
            output, _ = _mips_pic_transfer(
                project, blocks[addr], in_states[addr], gp_offset, sp_offset, preserved
            )
            if out_states.get(addr) == output:
                continue
            out_states[addr] = output
            for successor in edges[addr]:
                incoming = [
                    out_states[pred]
                    for pred in predecessors[successor]
                    if pred in out_states
                ]
                joined = _mips_join_states(incoming)
                if in_states.get(successor) != joined:
                    in_states[successor] = joined
                    pending.append(successor)
        return in_states, out_states, predecessors

    normal = propagate(successors)
    if normal is None:
        return {}, updates_left < 0, True
    in_states, out_states, predecessors = normal

    proved = {}
    # Worklist joins can be provisional. Do not use a candidate whose last
    # incoming state was never analyzed, even if its observed paths agree.
    for addr in candidates & in_states.keys():
        if not predecessors[addr] <= out_states.keys():
            continue
        _, target = _mips_pic_transfer(
            project, blocks[addr], in_states[addr], gp_offset, sp_offset, preserved
        )
        if target and all(
            _abi_static_target_is_valid(project, value) for value in target
        ):
            proved[addr] = tuple(sorted(target))

    def return_reaches_call(addr: int) -> bool:
        """Detect a decoded continuation cycle without following callee CFGs."""

        continuation = blocks[addr].fallthrough_addr
        if continuation not in blocks:
            return False
        pending = deque([continuation])
        seen: set[int] = set()
        while pending:
            current = pending.popleft()
            if current == addr:
                return True
            if current in seen:
                continue
            seen.add(current)
            pending.extend(successors[current])
        return False

    attempts = 0
    for addr in sorted(candidates - proved.keys()):
        if (
            attempts >= MAX_MIPS_NORETURN_CYCLE_PROOFS
            or blocks[addr].jumpkind != "Ijk_Call"
            or not return_reaches_call(addr)
        ):
            continue
        attempts += 1
        # The first encounter of this call cannot have traversed its own
        # return. If it must target a non-returning callee on those paths,
        # the apparent return and every cycle through it are impossible.
        without_return = dict(successors)
        without_return[addr] = ()
        alternative = propagate(without_return)
        if alternative is None:
            return {}, True, True
        alternate_in, _, _ = alternative
        if addr not in alternate_in:
            continue
        _, target = _mips_pic_transfer(
            project, blocks[addr], alternate_in[addr], gp_offset, sp_offset, preserved
        )
        if target and len(target) == 1:
            value = next(iter(target))
            if target_is_known_nonreturning(project, value):
                proved[addr] = (value,)
    return proved, False, True


def _mips_static_value(
    project: Project,
    vex,
    expr,
    definitions: dict[int, Any],
    global_pointer: int,
) -> int | None:
    """Evaluate the table planner's restricted GP-rooted scalar language.

    Resolve local temporaries, constants, GP reads, additive expressions and
    scalar loads whose addresses recursively resolve from those roots. Values
    assigned to GP by the lift can represent its proven root even when VEX
    folds away the explicit GET. Addition wraps at architecture width.

    This helper does not follow predecessor definitions or validate memory
    permissions/branch destinations; callers own those policies. ``None``
    means an unsupported or unreadable expression, distinct from integer zero.
    """

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
    if addr is None:
        return None
    return read_static_integer(project, addr, entry_size, expr.end)


def _mips_static_register_from_immediate_predecessors(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    expr,
    global_pointer: int,
) -> int | None:
    """Try the cheap base proof: every immediate predecessor writes one value.

    For a register GET, inspect each in-function materialized predecessor's
    final matching PUT and evaluate it in the GP-rooted scalar language.
    Return the value only when every predecessor supplies the same constant.
    A missing write, failed lift, foreign predecessor or disagreement leaves
    the base unknown; an arbitrary definition elsewhere in the graph cannot
    substitute for a reaching write.
    """

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
        definitions = temporary_definitions(vex)
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
    """Find a table-base register's constant by finite in-function iteration.

    This fallback carries a scalar through blocks that do not write the
    register; matching PUTs replace it with a GP-rooted constant or unknown.
    Distinguish unreached nodes from reachable unknown inputs, and seed the
    function entry/foreign incoming edges as unknown. Merge only agreeing
    reaching values and iterate until boundary states stop changing.

    Return the incoming value at ``node``, not its value after executing that
    node. This is the table planner's restricted constant propagation, not
    the separate private-frame/register-call analysis above.
    """

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
        """Keep an agreeing reached scalar; distinguish no path from unknown."""

        if not values:
            return unreached
        if unknown in values or len(set(values)) != 1:
            return unknown
        return values[0]

    def transfer(candidate, incoming: object) -> object:
        """Carry one register or replace it at its matching local writes."""

        if incoming is unreached:
            return unreached
        vex = _node_vex(candidate)
        if vex is None:
            return unknown
        definitions = temporary_definitions(vex)
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
    """Try immediate writes before the more expensive reaching-base fallback.

    Return ``(value, used_propagation)``; the second flag describes the route
    attempted even when it returns unknown. The planner uses this provenance
    to avoid changing a previously shape-free dispatcher when an optional
    propagated base cannot also establish a complete table bound.
    """

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
    """Recover an unsigned range check hidden behind a widened zero test.

    MIPS often computes the comparison into a boolean and branches on zero
    rather than exposing the range check directly as an Exit guard. Match
    either zero-test operand order and delegate only the unsigned one-bit
    widening form. This returns the underlying comparison, not its negation;
    the consumer supplies the taken/fallthrough range interpretation.
    """

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
    """Recognize a one-bit unsigned comparison widened to an integer flag.

    Require an unsigned conversion whose source width is exactly one bit and
    an underlying unsigned comparison operation. Signed flags or arbitrary
    integer zero tests cannot justify an unsigned selector bound.
    """

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
    """Recover the range flag locally or through one exact fallthrough block.

    A delay slot can materialize the widened comparison in a register while
    the next block performs the zero test. Follow that form only through one
    materialized in-function predecessor whose constant NEXT is this block,
    using its final matching PUT. Multiple predecessors, other transfers or
    an unsupported flag producer provide no comparison evidence.
    """

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
        or _vex_const_value(
            predecessor_vex.next, temporary_definitions(predecessor_vex)
        )
        != node.addr
    ):
        return None
    predecessor_definitions = temporary_definitions(predecessor_vex)
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
    """Collect a unique guarded row count for an unscaled register selector.

    Inspect in-function predecessors whose NEXT falls through to the
    dispatcher, recover their widened comparison guards and ask the generic
    guard helper for an inclusive upper bound on this register view. Require
    one consistent recognized bound, convert it to a count, and apply the row
    cap. This local recognizer is not a general path-feasibility analysis;
    missing usable or conflicting guard evidence returns ``None``.
    """

    bounds_found: set[int] = set()
    for predecessor in graph.predecessors(node):
        if not _node_is_materialized_cfg_node(predecessor):
            continue
        if not _node_intersects_bounds(predecessor, bounds):
            continue
        vex = _node_vex(predecessor)
        if (
            vex is None
            or _vex_const_value(vex.next, temporary_definitions(vex)) != node.addr
        ):
            continue
        definitions = temporary_definitions(vex)
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
    """Relate a byte-offset register to its guarded, unscaled row selector.

    Search fallthrough predecessors for writes that shift by exactly the
    entry size. First try a guarded expression domain: every relevant
    predecessor must establish the same count. Otherwise, when no expression
    count was found, identify one unscaled register and use its local guarded
    bound. Partial expression evidence cannot fall back to an unrelated
    register bound; ambiguous scales, sources or counts leave the table open.
    """

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
            or _vex_const_value(vex.next, temporary_definitions(vex)) != node.addr
        ):
            continue
        relevant_predecessors += 1
        definitions = temporary_definitions(vex)
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


def _mips_inline_scaled_index_source(
    expr, definitions: dict[int, Any], entry_size: int
):
    """Recover a selector shifted inside the dispatcher rather than upstream.

    Accept only a constant left shift whose multiplier equals ``entry_size``.
    Return its resolved source expression without claiming that source is a
    register or bounded. Subsequent helpers must supply that separate evidence.
    """

    expr = _resolve_vex_expr(expr, definitions)
    if not isinstance(expr, pyvex.expr.Binop) or not expr.op.startswith("Iop_Shl"):
        return None
    shift = _vex_const_value(expr.args[1], definitions)
    if shift is None or 1 << shift != entry_size:
        return None
    return _resolve_vex_expr(expr.args[0], definitions)


def _mips_inline_scaled_index_key(
    expr, definitions: dict[int, Any], vex, entry_size: int
) -> tuple[int, int] | None:
    """Require the inline shift's source to be a concrete register-width key.

    Expression selectors use a separate guarded-reload proof. Returning a key
    here only associates the shifted address with a register; it does not
    establish its row count or prove the surrounding table base.
    """

    source = _mips_inline_scaled_index_source(expr, definitions, entry_size)
    return _vex_get_key(source, definitions, vex) if source is not None else None


def _mips_guarded_reloaded_index_entry_count(
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
    source,
    definitions: dict[int, Any],
) -> int | None:
    """Bind a dispatcher reload to an adjacent guard on the same frame slot.

    Require one exact fallthrough predecessor, a load rooted in SP/FP/S8,
    identical keyed guard/load expressions and one finite comparison domain.
    Neither block may rewrite the address register or perform a memory side
    effect. Without alias analysis even an apparently unrelated store is a
    barrier. This narrow exception avoids guessing that two frame-relative
    loads retain the same selector across arbitrary control flow.
    """

    vex = _node_vex(node)
    source = _resolve_vex_expr(source, definitions)
    if vex is None or not isinstance(source, pyvex.expr.Load):
        return None
    address = _vex_register_with_displacement(source.addr, definitions, vex)
    if address is None:
        return None
    address_register = address[0][0]
    frame_registers = {
        vex.arch.registers[name][0]
        for name in ("sp", "s8", "fp")
        if name in vex.arch.registers
    }
    if address_register not in frame_registers:
        return None
    source_key = _vex_expr_key(source, definitions)
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
    if predecessor_vex is None:
        return None
    predecessor_defs = temporary_definitions(predecessor_vex)
    if _vex_const_value(predecessor_vex.next, predecessor_defs) != node.addr:
        return None

    # This local proof deliberately rejects even unrelated stores: without
    # alias analysis, any write could change the reloaded selector.
    for block_vex in (predecessor_vex, vex):
        if any(
            (
                isinstance(statement, pyvex.stmt.Put)
                and statement.offset == address_register
            )
            or isinstance(
                statement,
                (
                    pyvex.stmt.Store,
                    pyvex.stmt.StoreG,
                    pyvex.stmt.CAS,
                    pyvex.stmt.LLSC,
                    pyvex.stmt.Dirty,
                ),
            )
            for statement in block_vex.statements
        ):
            return None

    counts = {
        len(values)
        for statement in predecessor_vex.statements
        if isinstance(statement, pyvex.stmt.Exit)
        if (
            comparison := _mips_inverted_unsigned_guard(
                statement.guard, predecessor_defs
            )
        )
        is not None
        if _vex_expr_key(comparison.args[0], predecessor_defs) == source_key
        if (
            values := _vex_guarded_expression_values(
                comparison, comparison.args[0], predecessor_defs
            )
        )
        is not None
    }
    return next(iter(counts)) if len(counts) == 1 else None


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
    predecessor proves the unscaled selector's finite range. Construction may
    additionally opt in to a table base established before that predecessor,
    including a delay slot, when every in-function path agrees on its value.

    Split the entry address into static terms and exactly one supported
    selector. Choose the matching bound proof for an upstream scaled register,
    an inline unscaled register or a guarded frame reload. The returned plan
    records GP as the target base and the table's independent displacement;
    generic target reading later applies signedness, endianness and row indices.

    Return ``(plan, None)`` only after establishing a finite selector count.
    Missing shape/base evidence returns ``no_table_shape``; a recognized but
    unbounded selector normally returns ``unbounded_index`` so the builder
    preserves an unresolved target. Optional propagated-base recognition keeps
    the earlier shape-free result unless it can complete the whole proof.
    This function never reads target rows or modifies leaders/edges itself.
    """

    if not project.arch.name.startswith("MIPS"):
        return None, "no_table_shape"
    vex = _node_vex(node)
    global_pointer = mips_entry_global_pointer(project, bounds.addr)
    if vex is None or global_pointer is None or vex.jumpkind != "Ijk_Boring":
        return None, "no_table_shape"
    try:
        gp_offset = project.arch.registers["gp"][0]
    except KeyError:
        return None, "no_table_shape"

    definitions = temporary_definitions(vex)
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
        if entry_size not in STATIC_INTEGER_BYTES:
            continue

        address_terms = _vex_add_terms(entry.addr, definitions)
        if address_terms is None:
            continue
        table_addr = 0
        scaled_index_key = None
        inline_index_key = None
        inline_index_source = None
        used_propagated_static_base = False
        for term in address_terms:
            value = _mips_static_value(project, vex, term, definitions, global_pointer)
            if value is not None:
                table_addr += value
                continue
            if allow_predecessor_static_base:
                source = _mips_inline_scaled_index_source(term, definitions, entry_size)
                if source is not None:
                    if scaled_index_key is not None or inline_index_source is not None:
                        break
                    inline_index_key = _vex_get_key(source, definitions, vex)
                    inline_index_source = source
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
            if (
                scaled_index_key is None
                and inline_index_key is None
                and inline_index_source is None
            ):
                continue
            if scaled_index_key is not None:
                entry_count = _mips_scaled_index_entry_count(
                    graph, bounds, node, scaled_index_key, entry_size
                )
            elif inline_index_key is not None:
                entry_count = _mips_guarded_index_entry_count(
                    graph, bounds, node, inline_index_key
                )
            else:
                entry_count = _mips_guarded_reloaded_index_entry_count(
                    graph, bounds, node, inline_index_source, definitions
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
                table,
                global_pointer,
                tuple(range(entry_count)),
                proof_flavor="mips_pic_table",
            ), None

    return None, "no_table_shape"
