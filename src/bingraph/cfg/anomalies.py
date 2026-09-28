"""CFG anomaly detection, reporting, and function-bound helpers."""

from __future__ import annotations

from functools import lru_cache

from angr import Project
from angr.analyses.cfg import CFGBase
from angr.knowledge_plugins.cfg import CFGNode
from loguru import logger
import pyvex

from bingraph.helpers.capstone import (
    InsnSemantics,
    arch_has_delay_slot,
    control_transfer_index,
)
from bingraph.helpers.symbols import list_function_symbols
from .decode import (
    DecodedNode,
    call_fallthrough_addr as _call_fallthrough_addr,
    decode_one,
    decode_raw_capstone_insns,
    lift_instruction_vex,
    target_is_hooked_nonreturning,
    vex_jumpkind_is_terminal as _vex_jumpkind_is_terminal,
)
from .graph import (
    CFGGraph,
    cfg_graph as _cfg_graph,
    node_is_placeholder as _node_is_placeholder,
    node_is_simprocedure as _node_is_simprocedure,
    node_range_end as _node_range_end,
    node_vex as _node_vex,
)
from .jumps import (
    _constant_register_from_predecessors,
    _missing_jump_successor_anomaly,
    _missing_jump_successors,
    _read_static_jump_table_targets,
    plan_static_jump_table,
    s390_table_loaded_branch,
    static_jump_target_rejection_reason,
)
from .models import (
    CFGAnomaly,
    FunctionBounds,
    JumpSuccessorExpectation,
)


def _node_has_forced_split(node, forced_block_starts: set[int]) -> bool:
    """Return whether a required leader falls inside ``node``."""

    node_end = _node_range_end(node)
    return any(node.addr < addr < node_end for addr in forced_block_starts)


def _lookup_function_bounds(
    project: Project,
    func_addr: int,
    *,
    display_name: str | None = None,
) -> FunctionBounds:
    """
    Return function bounds from the symbol view used across bingraph.

    The custom repair pass needs a stable upper bound even when CFGFast itself
    missed blocks. `kb.functions[addr].size` is derived from currently
    discovered CFG blocks, so it can shrink along with a malformed CFG. By
    reusing `list_function_symbols()` we inherit the project's existing symbol
    parsing and size-inference logic instead.
    """

    function = next(
        (sym for sym in list_function_symbols(project) if sym.addr == func_addr), None
    )
    if function is None:
        raise KeyError(f"Function {func_addr:#x} not found in binary")

    end_addr = func_addr + function.size
    return FunctionBounds(
        addr=func_addr,
        end_addr=end_addr,
        size=end_addr - func_addr,
        symbol=function,
        display_name=display_name,
    )


def _iter_seed_function_nodes(seed_cfg: CFGBase, func_addr: int):
    """Yield non-simprocedure nodes from the seed CFG for one function."""

    for node in _cfg_graph(seed_cfg).nodes():
        if getattr(node, "function_address", None) != func_addr:
            continue
        if getattr(node, "is_simprocedure", False):
            continue
        yield node


def iter_function_nodes(cfg: CFGBase, func_addr: int):
    """Yield non-simprocedure nodes that belong to one function."""

    yield from _iter_seed_function_nodes(cfg, func_addr)


def _call_has_known_nonreturning_target(
    project: Project, graph: CFGGraph, node
) -> bool:
    """Return whether a call targets an explicitly non-returning hook.

    Immediate calls carry their address in Capstone. Resolved indirect calls,
    such as MIPS ``jalr $t9`` through a GOT slot, instead expose the concrete
    target through CFGFast's existing ``Ijk_Call`` successor.
    """

    decoded = DecodedNode.from_node(node)
    last_insn = decoded.last
    if last_insn is not None:
        target = InsnSemantics(last_insn).direct_target()
        if target is not None and target_is_hooked_nonreturning(project, target):
            return True

    for successor in graph.successors(node):
        edge_data = graph.get_edge_data(node, successor) or {}
        if edge_data.get("jumpkind") != "Ijk_Call":
            continue

        addr = getattr(successor, "addr", None)
        if isinstance(addr, int) and target_is_hooked_nonreturning(project, addr):
            return True

    return False


@lru_cache(maxsize=10_000)
def _can_decode_block_at_cached(
    project: Project,
    start_addr: int,
    end_addr: int,
    addr: int,
) -> bool:
    """Return whether ``addr`` begins one complete instruction within bounds."""

    if not start_addr <= addr < end_addr:
        return False

    # This is an entry-validity probe, not a request to recover the remainder
    # of the function. Lifting through ``end_addr`` made every cache miss
    # disassemble an entire suffix of large functions.
    max_bytes = min(getattr(project.arch, "max_inst_bytes", 16), end_addr - addr)
    insn = decode_one(project, addr, max_bytes)
    return insn is not None and insn.address + insn.size <= end_addr


def _can_decode_block_at(project: Project, bounds: FunctionBounds, addr: int) -> bool:
    """Return cached decodeability for one address in a function's bounds."""

    return _can_decode_block_at_cached(project, bounds.addr, bounds.end_addr, addr)


def _inval_icache_loaded_register_offset(vex) -> int | None:
    """Return the register used as one invalidated block's instruction address."""

    definitions = {
        statement.tmp: statement.data
        for statement in vex.statements
        if isinstance(statement, pyvex.stmt.WrTmp)
    }
    for statement in vex.statements:
        if not (
            isinstance(statement, pyvex.stmt.WrTmp)
            and isinstance(statement.data, pyvex.expr.Load)
        ):
            continue
        address = statement.data.addr
        seen: set[int] = set()
        while isinstance(address, pyvex.expr.RdTmp):
            if address.tmp in seen:
                break
            seen.add(address.tmp)
            address = definitions.get(address.tmp)
        if isinstance(address, pyvex.expr.Get):
            return address.offset
    return None


def _proven_inval_icache_fallthrough(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> int | None:
    """Return a safe fallthrough for a statically resolved invalidated block.

    VEX uses ``Ijk_InvalICache`` for instructions that execute an instruction
    selected at runtime. Most such transfers must remain unknown. A self-edge
    is recoverable only when the target register has one VEX-proven constant
    definition and the selected instruction itself lifts as plain linear code.
    """

    vex = _node_vex(node)
    if vex is None:
        return None
    edge_data = graph.get_edge_data(node, node) or {}
    if (
        vex.jumpkind != "Ijk_InvalICache"
        or edge_data.get("jumpkind") != "Ijk_InvalICache"
    ):
        return None

    register_offset = _inval_icache_loaded_register_offset(vex)
    if register_offset is None:
        return None
    target = _constant_register_from_predecessors(graph, bounds, node, register_offset)
    if target is None:
        return None

    insn = decode_one(project, target, getattr(project.arch, "max_inst_bytes", 16))
    if insn is None or InsnSemantics(insn).is_control_transfer():
        return None
    target_vex = lift_instruction_vex(project, insn)
    next_addr = getattr(getattr(target_vex, "next", None), "con", None)
    if (
        target_vex is None
        or target_vex.jumpkind != "Ijk_Boring"
        or getattr(next_addr, "value", None) != target + insn.size
    ):
        return None

    fallthrough_addr = node.addr + node.size
    return (
        fallthrough_addr
        if _can_decode_block_at(project, bounds, fallthrough_addr)
        else None
    )


def _inval_icache_self_loop_anomaly(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> CFGAnomaly | None:
    """Return an anomaly for a statically provable invalidated self-loop."""

    fallthrough_addr = _proven_inval_icache_fallthrough(project, graph, bounds, node)
    if fallthrough_addr is None:
        return None
    return CFGAnomaly(
        "inval_icache_self_loop",
        node.addr,
        f"Node {node.addr:#x} has a resolved Ijk_InvalICache self-loop; "
        f"execution falls through to {fallthrough_addr:#x}",
    )


@lru_cache(maxsize=10_000)
def _has_complete_capstone_block_at_cached(
    project: Project,
    start_addr: int,
    end_addr: int,
    addr: int,
) -> bool:
    """Return whether Capstone reaches a block boundary from ``addr``."""

    if not start_addr <= addr < end_addr:
        return False

    current_addr = addr
    while current_addr < end_addr:
        max_bytes = min(
            getattr(project.arch, "max_inst_bytes", 16), end_addr - current_addr
        )
        insn = decode_one(project, current_addr, max_bytes)
        if insn is None:
            return False
        current_addr += insn.size
        if InsnSemantics(insn).is_control_transfer():
            return True

    return current_addr == end_addr


def _has_complete_capstone_block_at(
    project: Project, bounds: FunctionBounds, addr: int
) -> bool:
    """Return whether a fake-return continuation reaches executable boundary."""

    return _has_complete_capstone_block_at_cached(
        project, bounds.addr, bounds.end_addr, addr
    )


@lru_cache(maxsize=10_000)
def _raw_capstone_avx512_vex_decode_gap_at(project: Project, addr: int) -> bool:
    """Return whether an AVX-512 instruction is decodable only by Capstone."""

    max_inst_bytes = getattr(project.arch, "max_inst_bytes", 16)
    insns = decode_raw_capstone_insns(project, addr, max_inst_bytes * 2, count=2)
    if not insns or not InsnSemantics(insns[0]).is_avx512():
        return False

    lift_size = insns[0].size
    if (
        arch_has_delay_slot(project.arch.name)
        and InsnSemantics(insns[0]).is_control_transfer()
        and len(insns) > 1
    ):
        # VEX needs the executed delay slot to lift MIPS transfers correctly.
        lift_size += insns[1].size

    try:
        vex = project.factory.block(
            addr,
            size=lift_size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception:
        return False
    if vex.jumpkind != "Ijk_NoDecode":
        return False
    return True


def _missing_call_fallthrough_anomaly(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> CFGAnomaly | None:
    """Return the missing fake-return anomaly for ``node``, if any."""

    if node_has_decoding_coverage_mismatch(node):
        return None

    decoded = DecodedNode.from_node(node)
    last_insn = decoded.last
    if last_insn is None:
        return None
    last_vex = lift_instruction_vex(project, last_insn)
    is_call = InsnSemantics(last_insn).is_call() or (
        last_vex is not None and last_vex.jumpkind == "Ijk_Call"
    )
    if not is_call or _call_has_known_nonreturning_target(project, graph, node):
        return None

    next_addr = _node_range_end(node)
    fallthrough_addr = _call_fallthrough_addr(project, bounds, next_addr)
    if fallthrough_addr is None:
        return None
    # Symbol-size inference can include literal pools. Require Capstone to
    # reach a complete block boundary before adding a call continuation.
    if (
        bounds.addr <= next_addr < bounds.end_addr
        and not _has_complete_capstone_block_at(project, bounds, next_addr)
    ):
        return None

    for successor in graph.successors(node):
        if successor.addr != fallthrough_addr:
            continue
        edge_data = graph.get_edge_data(node, successor) or {}
        if edge_data.get("jumpkind") == "Ijk_FakeRet":
            return None

    return CFGAnomaly(
        "missing_call_fallthrough",
        node.addr,
        f"Call node {node.addr:#x} is missing fake-return successor "
        f"{fallthrough_addr:#x}",
    )


def _missing_linear_fallthrough_anomaly(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    node,
) -> CFGAnomaly | None:
    """Return the straight-line fallthrough anomaly for ``node``, if any.

    CFGFast can stop at the next instruction when VEX cannot lift it, then
    retain a stale successor from a speculative decode beginning in the middle
    of that instruction stream. Limit this check to an AVX-512 fallthrough
    address that Capstone decodes while VEX reports ``Ijk_NoDecode``. Other
    straight-line-looking blocks can carry valid architecture-specific
    transfer semantics that Capstone does not classify as branches.
    """

    if node_has_decoding_coverage_mismatch(node):
        return None

    decoded = DecodedNode.from_node(node)
    if decoded.is_empty or decoded.insns is None:
        return None
    if (
        control_transfer_index(project.arch.name, list(decoded.insns), strict=False)
        is not None
    ):
        return None

    fallthrough_addr = _node_range_end(node)
    if not _can_decode_block_at(project, bounds, fallthrough_addr):
        return None
    if not _raw_capstone_avx512_vex_decode_gap_at(project, fallthrough_addr):
        return None

    successors = list(graph.successors(node))
    if len(successors) != 1 or successors[0].addr != fallthrough_addr:
        return CFGAnomaly(
            "missing_linear_fallthrough",
            node.addr,
            f"Node {node.addr:#x} is missing straight-line successor "
            f"{fallthrough_addr:#x}",
        )
    edge_data = graph.get_edge_data(node, successors[0]) or {}
    if edge_data.get("jumpkind") == "Ijk_Boring":
        return None
    return CFGAnomaly(
        "missing_linear_fallthrough",
        node.addr,
        f"Node {node.addr:#x} is missing straight-line successor {fallthrough_addr:#x}",
    )


def _stale_linear_execution_mode_transition_anomaly(
    graph: CFGGraph, node
) -> CFGAnomaly | None:
    """Return an anomaly for an ARM/Thumb switch without a branch instruction.

    CFGFast can retain a speculative alternate-mode stream and connect it with
    an ``Ijk_Boring`` edge from an ARM or Thumb block. A normal mode switch
    requires a control-transfer instruction such as ``bx`` or ``blx``; a
    linear edge between different execution modes is therefore stale. Queue
    the source for ordinary recovery so bounded decoding rebuilds its real
    terminator and physical fall-through.
    """

    source_mode = getattr(node, "thumb", None)
    if not isinstance(source_mode, bool):
        return None

    decoded = DecodedNode.from_node(node)
    if decoded.insns is None:
        return None
    if (
        control_transfer_index(node.block.arch.name, list(decoded.insns), strict=False)
        is not None
    ):
        return None

    for successor in graph.successors(node):
        if _node_is_simprocedure(successor):
            continue
        successor_mode = getattr(successor, "thumb", None)
        edge_data = graph.get_edge_data(node, successor) or {}
        if (
            isinstance(successor_mode, bool)
            and successor_mode != source_mode
            and edge_data.get("jumpkind") == "Ijk_Boring"
        ):
            return CFGAnomaly(
                "stale_linear_execution_mode_transition",
                node.addr,
                f"Node {node.addr:#x} falls through from "
                f"{'Thumb' if source_mode else 'ARM'} to "
                f"{'Thumb' if successor_mode else 'ARM'} at {successor.addr:#x}",
            )
    return None


def node_has_linear_merge_successor(
    graph: CFGGraph, node, protected_starts: set[int] | None = None
) -> bool:
    """Return whether ``node`` should absorb its only straight-line successor.

    This targets the specific malformed shape where CFGFast left an artificial
    split inside one linear byte range: A has one `Ijk_Boring` successor B, B
    has exactly one predecessor, B starts exactly where A ends, and A itself
    does not end in a control-transfer instruction.
    """

    successors = list(graph.successors(node))
    if len(successors) != 1:
        return False

    succ = successors[0]
    if protected_starts is not None and succ.addr in protected_starts:
        return False
    if getattr(succ, "is_simprocedure", False):
        return False
    if _node_is_placeholder(succ):
        return False
    # Synthetic unresolved-jump fallbacks preserve disconnected regions but do
    # not represent real branch targets. They must not prevent a normal linear
    # merge between two adjacent materialized blocks.
    materialized_predecessors = [
        predecessor
        for predecessor in graph.predecessors(succ)
        if not _node_is_simprocedure(predecessor)
    ]
    if len(materialized_predecessors) != 1:
        return False

    edge_data = graph.get_edge_data(node, succ) or {}
    if edge_data.get("jumpkind") != "Ijk_Boring":
        return False
    if _node_range_end(node) != succ.addr:
        return False

    try:
        decoded = DecodedNode.from_node(node)
    except Exception:
        return False

    insns = decoded.insns
    if not insns:
        return False

    # On MIPS the final instruction may be the delay slot. Look for the
    # effective terminator instead of treating that trailing instruction as a
    # straight-line fallthrough.
    if (
        control_transfer_index(node.block.arch.name, list(insns), strict=False)
        is not None
    ):
        return False

    return True


def _decoding_coverage_anomaly(node) -> CFGAnomaly | None:
    """Return the byte-coverage anomaly for ``node``, when present."""

    if node.size == 0:
        return CFGAnomaly(
            "decoding_coverage_mismatch",
            node.addr,
            f"Node {node.addr:#x} has size zero",
        )

    decoded = DecodedNode.from_node(node)
    if decoded.insns is None:
        return CFGAnomaly(
            "decoding_coverage_mismatch",
            node.addr,
            f"Capstone inspection failed for node {node.addr:#x}: "
            f"{type(decoded.inspection_error).__name__}: {decoded.inspection_error}",
        )

    if decoded.has_exact_coverage(node):
        return None

    expected_addr = node.addr
    for insn in decoded.insns:
        if insn.address != expected_addr:
            return CFGAnomaly(
                "decoding_coverage_mismatch",
                node.addr,
                f"Node {node.addr:#x} decodes instruction at {insn.address:#x} "
                f"instead of expected {expected_addr:#x}",
            )
        expected_addr += insn.size

    node_end = node.addr + node.size
    if expected_addr != node_end:
        return CFGAnomaly(
            "decoding_coverage_mismatch",
            node.addr,
            f"Node {node.addr:#x} decoded instructions end at {expected_addr:#x}, "
            f"but node size extends to {node_end:#x}",
        )

    return None


def node_has_decoding_coverage_mismatch(node) -> bool:
    """Return True when a CFG node clearly covers bytes incorrectly."""

    return _decoding_coverage_anomaly(node) is not None


def node_has_decode_gap(node) -> bool:
    """
    Return True when a CFG node has a real lifting-only gap worth flagging.

    Once a block's Capstone instruction stream covers the node span exactly, we
    treat it as structurally decodable even if VEX still reports
    `Ijk_NoDecode`. This keeps the anomaly checker focused on malformed blocks
    and missing coverage instead of on VEX-specific complaints for blocks we
    can already render correctly.
    """

    if node_has_decoding_coverage_mismatch(node):
        return False

    decoded = DecodedNode.from_node(node)
    if not decoded.is_empty:
        return False

    try:
        return node.block.vex.jumpkind == "Ijk_NoDecode"
    except Exception:
        return False


def _node_ends_in_undefined_instruction_trap(node) -> bool:
    """Return whether exact Capstone coverage ends in x86's ``ud2`` trap."""

    decoded = DecodedNode.from_node(node)
    last_insn = decoded.last
    return (
        last_insn is not None
        and decoded.has_exact_coverage(node)
        and InsnSemantics(last_insn).is_undefined_instruction_trap()
    )


def node_has_truncated_leaf(
    project: Project,
    graph: CFGGraph,
    bounds: FunctionBounds,
    func_addr: int,
    node,
) -> bool:
    """Return True when a CFG node stops before a real terminator and has no exits."""

    decoded = DecodedNode.from_node(node)
    if decoded.insns is None or decoded.is_empty:
        return False

    # VEX reports x86 ``ud2`` as Ijk_NoDecode even though Capstone decodes its
    # complete two-byte trap encoding. Treat that intentional synchronous trap
    # as a valid leaf, while retaining ordinary Ijk_NoDecode blocks as repair
    # candidates.
    last_insn = decoded.last
    if _node_ends_in_undefined_instruction_trap(node):
        return False

    try:
        if _vex_jumpkind_is_terminal(node.block.vex.jumpkind):
            return False
    except Exception:
        pass

    insns = list(decoded.insns)
    if last_insn is None:
        return False

    # MIPS executes one delay-slot instruction after a branch. The final
    # instruction can therefore be ordinary arithmetic even though the block
    # already has a real control transfer and is not a truncated leaf.
    if control_transfer_index(project.arch.name, insns, strict=False) is not None:
        return False

    last = InsnSemantics(last_insn)
    if last.is_control_transfer():
        return False

    if any(True for _ in graph.successors(node)):
        return False

    if not _can_decode_block_at(project, bounds, _node_range_end(node)):
        return False

    has_later_function_node = any(
        other is not node
        and getattr(other, "function_address", None) == func_addr
        and not getattr(other, "is_simprocedure", False)
        and other.addr > node.addr
        for other in graph.nodes()
    )
    return has_later_function_node


def _terminal_successor_anomaly(
    graph: CFGGraph,
    node,
    *,
    project: Project | None = None,
    bounds: FunctionBounds | None = None,
) -> CFGAnomaly | None:
    """Return an anomaly when VEX-terminal code retains a stale successor.

    A conditional return can have ``Ijk_Ret`` as its default VEX jumpkind and
    an ``Ijk_Boring`` exit to the next instruction for its not-taken path.
    An s390 table-loaded ``br`` can also lift as ``Ijk_Ret``. Only its proven
    table targets are valid; other successors are still stale.
    """

    try:
        vex = node.block.vex
        jumpkind = vex.jumpkind
    except Exception:
        return None

    undefined_instruction_trap = _node_ends_in_undefined_instruction_trap(node)
    if not _vex_jumpkind_is_terminal(jumpkind) and not undefined_instruction_trap:
        return None

    allowed_successor_addrs = (
        set()
        if undefined_instruction_trap
        else {
            target
            for _, _, stmt in getattr(vex, "exit_statements", ())
            if getattr(stmt, "jumpkind", None) == "Ijk_Boring"
            if isinstance(
                target := getattr(getattr(stmt, "dst", None), "value", None), int
            )
            if target == _node_range_end(node)
        }
    )
    successors = tuple(
        successor
        for successor in graph.successors(node)
        if successor.addr not in allowed_successor_addrs
    )
    if not successors:
        return None

    if project is not None and bounds is not None and s390_table_loaded_branch(node):
        plan, _ = plan_static_jump_table(project, graph, bounds, node)
        if plan is not None:
            targets = _read_static_jump_table_targets(
                project, plan.table, plan.base_addr, plan.entry_indices
            )
            if targets and all(
                static_jump_target_rejection_reason(project, target) is None
                for target in targets
            ):
                successors = tuple(
                    successor
                    for successor in successors
                    if successor.addr not in targets
                )
                if not successors:
                    return None

    targets = ", ".join(f"{successor.addr:#x}" for successor in successors)
    terminal_kind = (
        "instruction ud2" if undefined_instruction_trap else f"VEX jumpkind {jumpkind}"
    )
    return CFGAnomaly(
        "terminal_successor",
        node.addr,
        f"Node {node.addr:#x} has terminal {terminal_kind} but retains "
        f"successor(s): {targets}",
    )


def overlapping_instruction_entries(
    graph: CFGGraph, nodes: tuple[CFGNode, ...]
) -> tuple[tuple[CFGNode, tuple[int, ...]], ...]:
    """Return independently entered instruction starts covered by each node.

    CFGFast can retain a large linear node while also creating a separate node
    at an instruction it discovered through another path. The latter address
    must become a block leader when the covering node is repaired; otherwise
    the instruction is rendered twice. Entries inside an instruction are
    intentionally excluded because they can represent valid alternate x86 or
    Thumb instruction streams.
    """

    materialized_nodes = tuple(
        node
        for node in nodes
        if not _node_is_simprocedure(node) and not _node_is_placeholder(node)
    )
    nodes_by_addr: dict[int, list[CFGNode]] = {}
    for node in materialized_nodes:
        nodes_by_addr.setdefault(node.addr, []).append(node)

    overlaps: list[tuple[CFGNode, tuple[int, ...]]] = []
    for node in materialized_nodes:
        decoded = DecodedNode.from_node(node)
        if decoded.insns is None:
            continue

        node_end = _node_range_end(node)
        starts = {
            candidate.addr
            for insn in decoded.insns
            if node.addr < insn.address < node_end
            for candidate in nodes_by_addr.get(insn.address, ())
            if candidate is not node
            and any(
                predecessor is not node for predecessor in graph.predecessors(candidate)
            )
        }
        if starts:
            overlaps.append((node, tuple(sorted(starts))))

    return tuple(overlaps)


def node_has_foreign_function_owner(
    graph: CFGGraph,
    func_addr: int,
    node,
) -> bool:
    """Return True for an in-function successor mis-owned by CFGFast."""

    if node.function_address == func_addr:
        return False

    if not any(
        predecessor.function_address == func_addr
        for predecessor in graph.predecessors(node)
    ):
        return False

    return True


class CFGAnomalyDetector:
    """Classify node-local CFG anomalies and report each one only once."""

    def __init__(
        self,
        project: Project,
        graph: CFGGraph,
        bounds: FunctionBounds,
        func_addr: int,
        protected_starts: set[int] | None = None,
    ) -> None:
        """Bind anomaly checks to one live CFG graph and function range."""

        self.project = project
        self.graph = graph
        self.bounds = bounds
        self.func_addr = func_addr
        self.protected_starts = (
            protected_starts if protected_starts is not None else set()
        )
        self.reported_anomalies: set[tuple[str, int]] = set()

    def _report(self, anomaly: CFGAnomaly) -> None:
        """Emit ``anomaly`` once for this detector."""

        key = anomaly.kind, anomaly.addr
        if key in self.reported_anomalies:
            return
        self.reported_anomalies.add(key)
        logger.warning(anomaly.message)

    def check_decoding_coverage_mismatch(self, node) -> bool:
        """Check byte coverage and report the precise mismatch once."""

        anomaly = _decoding_coverage_anomaly(node)
        if anomaly is None:
            return False
        self._report(anomaly)
        return True

    def check_missing_jump_successor(self, node) -> bool:
        """Check direct branch edges and report missing or unexpected targets."""

        anomaly = _missing_jump_successor_anomaly(
            self.project, self.graph, self.bounds, node
        )
        if anomaly is None:
            return False
        self._report(anomaly)
        return True

    def check_terminal_successor(self, node) -> bool:
        """Check that a terminal VEX node has no stale CFG successor."""

        anomaly = _terminal_successor_anomaly(
            self.graph, node, project=self.project, bounds=self.bounds
        )
        if anomaly is None:
            return False
        self._report(anomaly)
        return True

    def missing_jump_successors(self, node) -> tuple[JumpSuccessorExpectation, ...]:
        """Return missing direct branch targets for immediate edge recovery."""

        return _missing_jump_successors(self.project, self.graph, self.bounds, node)

    def check_missing_call_fallthrough(self, node) -> bool:
        """Check fake-return coverage for calls and report a missing edge once."""

        anomaly = _missing_call_fallthrough_anomaly(
            self.project, self.graph, self.bounds, node
        )
        if anomaly is None:
            return False
        self._report(anomaly)
        return True

    def check_missing_linear_fallthrough(self, node) -> bool:
        """Check straight-line blocks for their required next-byte successor."""

        anomaly = _missing_linear_fallthrough_anomaly(
            self.project, self.graph, self.bounds, node
        )
        if anomaly is None:
            return False

        self._report(anomaly)
        return True

    def proven_inval_icache_fallthrough(self, node) -> int | None:
        """Return a statically proved fallthrough for one invalidated node."""

        return _proven_inval_icache_fallthrough(
            self.project, self.graph, self.bounds, node
        )

    def check_inval_icache_self_loop(self, node) -> bool:
        """Check for a dynamic-execution self-loop with a proven continuation."""

        anomaly = _inval_icache_self_loop_anomaly(
            self.project, self.graph, self.bounds, node
        )
        if anomaly is None:
            return False
        self._report(anomaly)
        return True

    def check_stale_linear_execution_mode_transition(self, node) -> bool:
        """Check that a linear edge does not change ARM/Thumb execution mode."""

        anomaly = _stale_linear_execution_mode_transition_anomaly(self.graph, node)
        if anomaly is None:
            return False
        self._report(anomaly)
        return True

    def check_linear_merge_successor(self, node) -> bool:
        """Check for an artificial linear split and report it once."""

        if not node_has_linear_merge_successor(self.graph, node, self.protected_starts):
            return False
        successor = next(iter(self.graph.successors(node)))
        self._report(
            CFGAnomaly(
                "linear_merge_successor",
                node.addr,
                f"Node {node.addr:#x} is split from straight-line successor "
                f"{successor.addr:#x}",
            )
        )
        return True

    def check_foreign_function_owner(self, node) -> bool:
        """Check for an in-bounds node retained under another function owner."""

        if not node_has_foreign_function_owner(self.graph, self.func_addr, node):
            return False
        self._report(
            CFGAnomaly(
                "foreign_function_owner",
                node.addr,
                f"Node {node.addr:#x} is reached from function {self.func_addr:#x} "
                f"but is owned by CFGFast function {node.function_address:#x}",
            )
        )
        return True

    def node_needs_repair(self, node) -> bool:
        """Return True when ``node`` violates a repair invariant."""

        return (
            self.check_decoding_coverage_mismatch(node)
            or node_has_decode_gap(node)
            or node_has_truncated_leaf(
                self.project, self.graph, self.bounds, self.func_addr, node
            )
            or self.check_terminal_successor(node)
            or self.check_missing_jump_successor(node)
            or self.check_missing_call_fallthrough(node)
            or self.check_missing_linear_fallthrough(node)
            or self.check_inval_icache_self_loop(node)
            or self.check_stale_linear_execution_mode_transition(node)
            or self.check_linear_merge_successor(node)
        )

    def node_is_acceptable(self, forced_block_starts: set[int], node) -> bool:
        """Return True when an existing node may remain unchanged in the graph."""

        if _node_is_placeholder(node):
            return False
        if _node_has_forced_split(node, forced_block_starts):
            return False
        if getattr(node, "function_address", None) != self.func_addr:
            return False
        return not self.node_needs_repair(node)


def _has_decoding_coverage_mismatch(cfg: CFGBase, node) -> bool:
    """Return True when decoded instructions do not cover the full node span."""

    if not node_has_decoding_coverage_mismatch(node):
        return False

    logger.warning(
        f"CFG anomaly for function {node.function_address:#x}:"
        f" {'zero_sized_block' if node.size == 0 else 'malformed_block'}"
        f" at {node.addr:#x}"
    )
    return True


def _has_truncated_leaf(cfg: CFGBase, func_addr: int, node) -> bool:
    """Return True when a block stops before a real terminator and has no exits."""

    project = getattr(cfg, "project", None)
    if project is None:
        project = getattr(getattr(cfg, "kb", None), "_project", None)
    if project is None:
        return False

    if not node_has_truncated_leaf(
        project,
        _cfg_graph(cfg),
        _lookup_function_bounds(project, func_addr),
        func_addr,
        node,
    ):
        return False

    try:
        insns = list(node.block.capstone.insns)
    except (AttributeError, KeyError):
        insn_text = "<unknown>"
    else:
        if insns:
            last = insns[-1]
            insn_text = f"{last.mnemonic} {last.op_str}".strip()
        else:
            insn_text = "<empty>"

    logger.warning(
        f"CFG anomaly for function {func_addr:#x}: truncated_leaf at {node.addr:#x}: "
        f"block ends with non-terminating instruction {insn_text} and has no CFG successors"
    )
    return True


def _has_decode_gap(cfg: CFGBase, func_addr: int) -> bool:
    """Return True when the CFG contains true decoding/lifting failures."""

    if getattr(getattr(cfg, "model", None), "ident", "") == "CFGFastCustom":
        # The custom fallback is intentionally capstone-driven. VEX lifting can
        # still complain about some recovered nodes, but at that point the
        # custom graph should be judged by decoded instruction coverage instead
        # of by whether pyvex likes every block.
        return False

    for node in iter_function_nodes(cfg, func_addr):
        if _has_decoding_coverage_mismatch(cfg, node):
            continue

        if node_has_decode_gap(node):
            logger.warning(
                f"CFG anomaly for function {func_addr:#x}: decode_gap at {node.addr:#x}: "
                "node ended with Ijk_NoDecode, which points to a lifting/decoding failure"
            )
            return True

    return False


def _has_weird_graph(cfg: CFGBase, func_addr: int) -> bool:
    """Return True when the CFG shows malformed structure without a decode gap."""

    project = getattr(cfg, "project", None)
    if project is None:
        project = getattr(getattr(cfg, "kb", None), "_project", None)

    for node in iter_function_nodes(cfg, func_addr):
        if _has_decoding_coverage_mismatch(cfg, node):
            return True
        if _has_truncated_leaf(cfg, func_addr, node):
            return True
        if project is not None and _missing_jump_successor_anomaly(
            project,
            _cfg_graph(cfg),
            _lookup_function_bounds(project, func_addr),
            node,
        ):
            logger.warning(
                f"CFG anomaly for function {func_addr:#x}: missing_jump_successor "
                f"at {node.addr:#x}: direct jump block is missing one or more CFG edges"
            )
            return True
        if project is not None and _missing_call_fallthrough_anomaly(
            project,
            _cfg_graph(cfg),
            _lookup_function_bounds(project, func_addr),
            node,
        ):
            logger.warning(
                f"CFG anomaly for function {func_addr:#x}: "
                f"missing_call_fallthrough at {node.addr:#x}: call block is missing "
                "an in-function fake-return edge"
            )
            return True
        if project is not None and _missing_linear_fallthrough_anomaly(
            project,
            _cfg_graph(cfg),
            _lookup_function_bounds(project, func_addr),
            node,
        ):
            logger.warning(
                f"CFG anomaly for function {func_addr:#x}: "
                f"missing_linear_fallthrough at {node.addr:#x}: straight-line block "
                f"does not continue at {_node_range_end(node):#x}"
            )
            return True
        if node_has_linear_merge_successor(_cfg_graph(cfg), node):
            logger.warning(
                f"CFG anomaly for function {func_addr:#x}: linear_split at {node.addr:#x}: "
                "straight-line successor should be merged into the current block"
            )
            return True
    return False


def log_cfg_status(cfg: CFGBase, func_addr: int, cfg_label: str) -> None:
    """Log whether a CFG still shows the anomaly classes we currently track."""

    has_weird_graph = _has_weird_graph(cfg, func_addr)
    has_decode_gap = _has_decode_gap(cfg, func_addr)
    if not has_weird_graph and not has_decode_gap:
        logger.info(
            f"{cfg_label} for function {func_addr:#x} no longer shows known CFG anomalies"
        )
    else:
        logger.warning(
            f"{cfg_label} for function {func_addr:#x} still shows CFG anomalies"
        )
