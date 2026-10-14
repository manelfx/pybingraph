# Portions adapted from bingraphvis (BSD-2-Clause); see NOTICE.

"""Presentation-only edge classification from exact transfers and provenance.

Use decoded tails and focused VEX lifts to classify existing edges, including
delay slots, exceptional flow and system calls. Classification neither changes
the graph nor supplies proof facts to discovery or recovery."""

from __future__ import annotations

from loguru import logger

from bingraph.cfg.decode import vex_jumpkind_is_terminal
from bingraph.core.vis import EdgeAnnotator, Node
from bingraph.helpers import get_style
from bingraph.helpers.capstone import (
    InsnSemantics,
    control_transfer_index,
    proven_unconditional_direct_target,
)
from bingraph.helpers.disassembly import decode_one
from bingraph.helpers.vex import is_call_like_jumpkind


def _is_unresolvable_jump_target(node: Node) -> bool:
    """Return whether ``node`` is angr's unresolved indirect-jump placeholder."""

    return (
        node.obj.is_simprocedure
        and node.obj.simprocedure_name == "UnresolvableJumpTarget"
    )


def _control_transfer_context(edge):
    """Locate the actual source transfer used to style one existing edge.

    Prefer precise instruction provenance attached by angr, otherwise find the
    architecture-aware terminator including delay slots. Reconstruct the native
    instruction view when VEX-only linear spans precede it. Missing inspection
    evidence leaves classification unknown; this presentation fallback cannot
    contribute targets or leaders to discovery.
    """

    source_node = edge.src.obj
    try:
        vex_sizes = getattr(source_node, "vex_linear_instruction_sizes", None)
        if vex_sizes:
            project = edge.src.project
            max_bytes = getattr(project.arch, "max_inst_bytes", 16)
            insns = []
            for addr in source_node.instruction_addrs:
                if addr in vex_sizes:
                    continue
                insn = decode_one(project, addr, max_bytes)
                if insn is None:
                    return None
                insns.append(insn)
        else:
            insns = [wrapped.insn for wrapped in source_node.block.capstone.insns]
        ins_addr = edge.meta.get("ins_addr")
        if isinstance(ins_addr, int):
            for index, insn in enumerate(insns):
                if insn.address != ins_addr:
                    continue
                if InsnSemantics(insn).is_control_transfer():
                    # CFGFast associates an edge with the instruction that
                    # created it. Prefer that precise provenance over a
                    # full-block VEX lift, which can stop at an older split.
                    return insns, index
                break
        terminator_index = control_transfer_index(edge.src.project.arch.name, insns)
        if terminator_index is None:
            return None
        return insns, terminator_index
    except (AttributeError, KeyError, RuntimeError):
        return None


def _control_transfer_tail(edge):
    """Return a block's control-transfer instruction and any delay-slot tail."""

    context = _control_transfer_context(edge)
    if context is None:
        return None
    insns, terminator_index = context
    return insns[terminator_index:]


def _lift_control_transfer_tail(edge, tail):
    """Lift the exact native terminator/delay-slot span for edge presentation.

    A whole-block lift may have stopped earlier or retained an old split. The
    focused tail can explain the already-existing edge without rebuilding its
    source block. Failed lifting returns no evidence, not a rendering failure
    or a reason to invent a new successor.
    """

    try:
        return edge.src.project.factory.block(
            tail[0].address,
            size=sum(insn.size for insn in tail),
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception:
        # This is a presentation-only recovery attempt. A failed tail lift
        # means the edge is genuinely unclassifiable, not a render failure.
        return None


def _tail_lift_confirms_unconditional_direct_branch(
    edge, tail, target_addr: int
) -> bool:
    """Return whether a focused VEX lift confirms one direct target and no exits."""

    tail_vex = _lift_control_transfer_tail(edge, tail)
    if tail_vex is None or tail_vex.jumpkind != "Ijk_Boring":
        return False
    try:
        next_addr = tail_vex.next.con.value
    except AttributeError:
        return False
    return next_addr == target_addr and not tail_vex.exit_statements


def _capstone_direct_branch_edge_type(edge, exit_targets: set[int]) -> str | None:
    """Classify a direct branch obscured by VEX block semantics.

    VEX can constant-fold a branch or assign a non-branch jumpkind to a block
    because of an earlier instruction such as x86 ``pause`` or ``syscall``.
    Preserve the branch only when Capstone's exact immediate target matches
    the CFG edge being styled.
    """

    context = _control_transfer_context(edge)
    if context is None:
        return None
    insns, terminator_index = context
    tail = insns[terminator_index:]
    semantics = InsnSemantics(tail[0])
    target_addr = semantics.direct_target_for_arch(edge.src.project.arch.name)
    if not semantics.is_jump() or target_addr is None:
        return None

    if (
        proven_unconditional_direct_target(
            edge.src.project.arch.name, insns, terminator_index
        )
        == target_addr
    ):
        fallthrough_addr = tail[-1].address + tail[-1].size
        if target_addr == fallthrough_addr:
            return None
        return "UNCONDITIONAL" if edge.dst.obj.addr == target_addr else None

    is_conditional = semantics.is_conditional_jump()
    if is_conditional:
        try:
            successor_addrs = {
                successor.addr for successor in edge.src.graph.successors(edge.src.obj)
            }
        except (AttributeError, KeyError):
            successor_addrs = set()

        if successor_addrs == {target_addr}:
            if _tail_lift_confirms_unconditional_direct_branch(edge, tail, target_addr):
                # Several architectures encode an unconditional direct branch
                # as a one-operand generic jump. Capstone cannot distinguish
                # it from a flag-conditioned branch, but a focused VEX lift
                # has no Exit for its sole target.
                is_conditional = False

    if not is_conditional:
        # Prefer VEX when it still exposes an explicit conditional exit. A
        # Thumb branch can look unconditional to Capstone while VEX retains
        # architecture-specific condition semantics for the CFG edge.
        if exit_targets and not _tail_lift_confirms_unconditional_direct_branch(
            edge, tail, target_addr
        ):
            return None
        fallthrough_addr = tail[-1].address + tail[-1].size
        if target_addr == fallthrough_addr:
            return None
        return "UNCONDITIONAL" if edge.dst.obj.addr == target_addr else None

    fallthrough_addr = tail[-1].address + tail[-1].size
    # Some control-transfer instructions, such as x86 XBEGIN with a zero
    # displacement, encode the sequential address as their only target. They
    # do not represent a distinct taken edge in the rendered CFG.
    if target_addr == fallthrough_addr:
        return None
    try:
        successor_addrs = {
            successor.addr for successor in edge.src.graph.successors(edge.src.obj)
        }
    except (AttributeError, KeyError):
        return None
    if {target_addr, fallthrough_addr} - successor_addrs:
        return None

    # A long normalized node can retain exits from a VEX lift that stopped at
    # an earlier instruction. They are not evidence about this edge when they
    # reach neither successor of the exact Capstone branch that CFGFast tagged
    # in ``ins_addr``.
    if exit_targets.intersection({target_addr, fallthrough_addr}):
        return None
    if edge.dst.obj.addr == target_addr:
        return "CONDITIONAL_TRUE"
    if edge.dst.obj.addr == fallthrough_addr:
        return "CONDITIONAL_FALSE"
    return None


def _capstone_linear_tail_edge_type(edge, vex) -> str | None:
    """Classify a sole fall-through after VEX stops inside a recovered block.

    VEX can stop lifting before a decoded block's actual end. When
    Capstone finds no control transfer in the complete block, VEX's default
    target remains inside its byte range, and the source has one successor at
    that decoded end, the edge is the architectural fall-through. A VEX
    warning jumpkind may instead preserve the decoded next address exactly;
    that narrow case is also linear. Both patterns occur on multiple
    architectures when bounded decoding continues beyond VEX's first boundary.
    """

    source_node = edge.src.obj
    try:
        insns = [wrapped.insn for wrapped in source_node.block.capstone.insns]
        if not insns:
            return None
        if control_transfer_index(edge.src.project.arch.name, insns) is not None:
            return None
        next_addr = vex.next.con.value
        fallthrough_addr = insns[-1].address + insns[-1].size
        successor_addrs = {
            successor.addr for successor in edge.src.graph.successors(source_node)
        }
    except (AttributeError, KeyError, RuntimeError):
        return None

    if not isinstance(next_addr, int):
        return None
    next_is_internal = source_node.addr <= next_addr < fallthrough_addr
    next_is_error_fallthrough = (
        vex.jumpkind in {"Ijk_EmWarn", "Ijk_EmFail"} and next_addr == fallthrough_addr
    )
    if not (next_is_internal or next_is_error_fallthrough):
        return None
    if successor_addrs != {fallthrough_addr}:
        return None
    return "NEXT" if edge.dst.obj.addr == fallthrough_addr else None


def _vex_boring_edge_type(edge) -> str:
    """Interpret an existing boring edge using lifted exits and native tail evidence.

    Distinguish taken/continuation branches, unconditional direct transfers and
    unresolved indirect flow, including stale whole-block lifts and linear
    VEX-only prefixes. Native fallback evidence must agree with the edge's
    actual target; uncertainty stays a visual unknown category, not a new CFG
    fact. Exceptional and recovered-entry metadata have their own edge policy.
    """

    source_node = edge.src.obj
    try:
        vex = source_node.block.vex
    except (AttributeError, KeyError):
        return "UNKNOWN"

    if vex.jumpkind == "Ijk_NoDecode":
        # Lift only the branch tail when preceding SIMD/extension instructions
        # make VEX reject the complete block.
        tail = _control_transfer_tail(edge)
        if tail is None:
            if edge.dst.obj.addr == source_node.addr + source_node.size:
                return "NEXT"
            return "UNKNOWN"
        vex = _lift_control_transfer_tail(edge, tail)
        if vex is None:
            return "UNKNOWN"

    if vex.jumpkind == "Ijk_Call":
        try:
            call_target = vex.next.con.value
        except AttributeError:
            call_target = None
        # CFGFast occasionally retains a direct call target as Ijk_Boring even
        # though VEX identifies the source transfer as a call. Only recover
        # the style when the edge lands at that exact lifted target.
        if edge.dst.obj.addr == call_target:
            return "CALL"

        # A conditional call has an explicit boring exit to its architectural
        # fall-through, while VEX's default target remains the call itself.
        # This also covers conditional indirect calls, whose call target is
        # not a concrete VEX value.
        fallthrough_addr = source_node.addr + source_node.size
        if edge.dst.obj.addr != fallthrough_addr:
            return "UNKNOWN"
        for _, _, stmt in vex.exit_statements:
            if stmt.jumpkind != "Ijk_Boring":
                continue
            try:
                if stmt.dst.value == fallthrough_addr:
                    return "CONDITIONAL_FALSE"
            except AttributeError:
                continue
        return "UNKNOWN"

    try:
        next_addr = vex.next.con.value
    except AttributeError:
        next_addr = None

    # VEX records explicit Exit statements for conditional branches. Depending
    # on the lifter, either the exit or the default `next` can be the taken
    # destination, so both are valid non-fall-through successors.
    exit_targets: set[int] = set()
    for _, _, stmt in vex.exit_statements:
        if stmt.jumpkind != "Ijk_Boring":
            continue
        try:
            target = stmt.dst.value
        except AttributeError:
            continue
        if isinstance(target, int):
            exit_targets.add(target)

    capstone_branch_type = _capstone_direct_branch_edge_type(edge, exit_targets)
    if capstone_branch_type is not None:
        return capstone_branch_type

    capstone_linear_type = _capstone_linear_tail_edge_type(edge, vex)
    if capstone_linear_type is not None:
        return capstone_linear_type

    terminal_default = vex_jumpkind_is_terminal(vex.jumpkind)
    if vex.jumpkind != "Ijk_Boring" and not terminal_default:
        return "UNKNOWN"

    if exit_targets:
        if terminal_default:
            # Conditional return instructions can use a terminal default VEX
            # jumpkind together with an explicit Ijk_Boring exit for their
            # non-returning path. The terminal default is the taken return,
            # so the explicit next-instruction exit is the red not-taken path.
            fallthrough_addr = source_node.addr + source_node.size
            if (
                edge.dst.obj.addr == fallthrough_addr
                and edge.dst.obj.addr in exit_targets
            ):
                return "CONDITIONAL_FALSE"
            if edge.dst.obj.addr in exit_targets:
                return "CONDITIONAL_TRUE"
            if edge.dst.obj.addr == next_addr:
                return "CONDITIONAL_FALSE"
            return "UNKNOWN"

        fallthrough_addr = source_node.addr + source_node.size
        if (
            edge.dst.obj.addr == fallthrough_addr
            and exit_targets == {source_node.addr}
            and _control_transfer_tail(edge) is None
        ):
            # VEX models some atomic x86 instructions with a self-targeting
            # internal Exit. Capstone sees no control transfer, so the only
            # CFG successor is the ordinary next instruction.
            return "NEXT"
        if edge.dst.obj.addr == fallthrough_addr:
            return "CONDITIONAL_FALSE"
        if edge.dst.obj.addr in exit_targets or edge.dst.obj.addr == next_addr:
            return "CONDITIONAL_TRUE"
        if next_addr is None:
            # A conditional computed-PC write has an explicit not-taken exit
            # plus a dynamic taken destination. CFG recovery may resolve that
            # destination into concrete table targets, which remain indirect.
            return "INDIRECT"
        return "UNKNOWN"

    if next_addr is None:
        # A non-constant VEX `next` is an indirect branch. Recovered table
        # entries are concrete edges, but the dispatch itself remains indirect.
        return "INDIRECT"

    if not isinstance(next_addr, int) or edge.dst.obj.addr != next_addr:
        return "UNKNOWN"
    if next_addr == source_node.addr + source_node.size:
        return "NEXT"
    return "UNCONDITIONAL"


def _edge_type(edge) -> str:
    """Select a visual transfer category without changing the underlying graph.

    Honor explicit exceptional, recovery and fake-return metadata before
    ordinary call/jump interpretation. System transfers use call styling while
    retaining their distinct analysis jumpkind. Unknown classification remains
    visible; edge colors/styles never establish discovery or proof semantics.
    """

    # ELF LSDA metadata proves that a call transfers to this landing pad only
    # while unwinding. Keep that known exceptional path distinct from both the
    # normal call edge and its fake-return continuation.
    if edge.meta.get("exceptional"):
        return "EXCEPTION"

    # Candidate edges carry unresolved semantics even when their destination
    # is a decoded block rather than an unknown-target synthetic leaf.
    if edge.meta.get("unresolved_indirect"):
        return "UNRESOLVED_INDIRECT"

    # Both sides of this synthetic node express unresolved control flow: the
    # incoming edge is the unresolved jump and outgoing edges are candidates.
    if _is_unresolvable_jump_target(edge.src) or _is_unresolvable_jump_target(edge.dst):
        return "UNRESOLVED_INDIRECT"

    jumpkind = edge.meta.get("jumpkind")
    if jumpkind == "Ijk_Ret":
        return "RET"
    if jumpkind == "Ijk_FakeRet":
        return "FAKE_RET"
    # System transfers such as x86 ``int 0x80`` cross into an OS service just
    # like calls. Keep their explicit fake-return edges distinct below.
    if is_call_like_jumpkind(jumpkind):
        return "CALL"
    if jumpkind == "Ijk_Boring":
        return _vex_boring_edge_type(edge)

    logger.warning(
        f"Unexpected {jumpkind!r} edge type for "
        f"{edge.src.obj.addr:#x} -> {edge.dst.obj.addr:#x}"
    )
    return "UNKNOWN"


class ColorEdgesVex(EdgeAnnotator):
    """Apply the style of an existing edge's proven or unknown transfer kind."""

    def annotate_edge(self, edge) -> None:
        """Style one edge without inferring branch kind from successor count."""

        get_style().make_edge(edge, _edge_type(edge))
