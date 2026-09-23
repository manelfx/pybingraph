from abc import abstractmethod
from typing import Any
from loguru import logger

from bingraph.helpers import get_style
from bingraph.helpers.capstone import (
    InsnSemantics,
    control_transfer_index,
    proven_unconditional_direct_target,
)
from bingraph.cfg.decode import vex_jumpkind_is_terminal
from .vis import NodeAnnotator, ContentAnnotator, EdgeAnnotator, Node


class ColorSimprocedures(NodeAnnotator):
    def annotate_node(self, node: Node) -> None:
        if not node.obj.is_simprocedure:
            return

        node.pydot.set_style("filled")
        if node.obj.simprocedure_name in [
            "PathTerminator",
            "ReturnUnconstrained",
            "UnresolvableTarget",
        ]:
            node.pydot.set_fillcolor("#ffcccc")
        else:
            node.pydot.set_fillcolor("#dddddd")


class CommentsAnnotator(ContentAnnotator):
    name: str = "asm"
    column: str = "comment"

    @abstractmethod
    def get_comments_by_addr(self, node: Node) -> dict[int, list[str]]:
        pass

    def annotate_content(self, node: Node, content: dict[str, Any]):
        if node.obj.is_simprocedure or node.obj.is_syscall:
            return

        comments_by_addr = self.get_comments_by_addr(node)

        for k in content["data"]:
            ins = k["_ins"]
            ins_addr = ins.address if ins is not None else k.get("_addr")
            comments = list(k.get("_comments", ()))
            if ins_addr is not None:
                comments.extend(comments_by_addr.get(ins_addr, ()))
            if comments:
                k["comment"] = {"content": " ; " + "\n".join(comments)}
                k["comment"]["color"] = "gray"
                k["comment"]["align"] = "LEFT"


class CommentsDataRef(CommentsAnnotator):
    @staticmethod
    def _symbol_name_at(node: Node, addr: int) -> str | None:

        # Check if it maps to a known internal label
        project = node.project
        if addr in project.kb.labels:
            return project.kb.labels[addr]

        # Check if it maps to a global symbol or imported function
        sym = project.loader.find_symbol(addr)
        if sym:
            return sym.name

        return None

    @staticmethod
    def _truncate_comment(text: str, max_len: int = 64) -> str:
        text = text.replace("\n", "\\n").replace("\r", "\\r")

        if len(text) <= max_len:
            return text

        return text[: max_len - 3] + "..."

    def _format_memory_data_comment(self, node: Node, md) -> str:

        content = getattr(md, "content", None)
        if isinstance(content, (bytes, bytearray)):
            text = content.decode("utf-8", errors="ignore").strip("\x00")
            if text:
                text = self._truncate_comment(text)
                return f'"{text}"'

        addr = getattr(md, "addr", None)
        sort = str(getattr(md, "sort", "data")).lower()

        if addr is not None:
            symbol = self._symbol_name_at(node, addr)
            if symbol:
                return symbol

        if addr is None:
            return f"data: {sort}"

        if "pointer" in sort or sort == "ptr":
            target = getattr(md, "pointer_addr", None)
            if isinstance(target, int):
                target_name = self._symbol_name_at(node, target)
                if target_name:
                    return f"ptr -> {target_name}"
                return f"ptr -> {hex(target)}"

            return f"ptr @ {hex(addr)}"

        if "string" in sort:
            return f"string @ {hex(addr)}"

        if "jumptable" in sort or "jump table" in sort:
            return f"jump table @ {hex(addr)}"

        if "integer" in sort or "int" in sort:
            return f"int @ {hex(addr)}"

        return f"{sort} @ {hex(addr)}"

    def _format_address_comment(self, node: "Node", addr: int) -> str:
        symbol = self._symbol_name_at(node, addr)
        if symbol:
            return symbol

        return f"ref {hex(addr)}"

    def _format_xref_comment(self, node: Node, xref) -> str | None:

        md = getattr(xref, "memory_data", None)
        if md is not None:
            # Case 1: angr resolved a MemoryData object
            return self._format_memory_data_comment(node, md)

        dst = getattr(xref, "dst", None)
        if isinstance(dst, int):
            # Case 2: raw destination address only
            return self._format_address_comment(node, dst)

        return None

    def get_comments_by_addr(self, node: Node) -> dict[int, list[str]]:
        comments_by_addr: dict[int, list[str]] = {}

        kb_xrefs = getattr(node.kb, "xrefs", None)
        if kb_xrefs is None:
            return comments_by_addr

        block_start = node.obj.addr
        block_end = block_start + node.obj.size

        xrefs = kb_xrefs.get_xrefs_by_ins_addr_region(block_start, block_end)
        if not xrefs:
            # Do not use "accessed_data_refernces" sice CFGFast is required
            # e.g., xrefs = list(getattr(node.obj, "accessed_data_references", []))
            for instr_addr in node.obj.instruction_addrs:
                xrefs.update(kb_xrefs.get_xrefs_by_ins_addr(instr_addr))
        if len(xrefs):
            logger.info(
                f"Found {len(xrefs)} reference(s) in block {hex(block_start)}-{hex(block_end)}"
            )

        def _xref_sort_key(xref):
            md = getattr(xref, "memory_data", None)
            return (
                getattr(xref, "ins_addr", -1),
                getattr(xref, "dst", -1),
                getattr(md, "addr", -1) if md is not None else -1,
                str(getattr(md, "sort", "")) if md is not None else "",
            )

        # Keep comment emission deterministic across runs. angr xref iteration
        # order is not stable enough for golden-file tests when an instruction
        # accumulates multiple references/comments.
        for xref in sorted(xrefs, key=_xref_sort_key):
            comment = self._format_xref_comment(node, xref)
            if not comment:
                continue

            # add some space for pretty printing on BB boundary
            comment += " "

            # Merge multiple references originating from the same instruction
            ins_addr = xref.ins_addr
            if ins_addr in comments_by_addr:
                if comment not in comments_by_addr[ins_addr]:
                    comments_by_addr[ins_addr].append(comment)
            else:
                comments_by_addr[ins_addr] = [comment]

        for comments in comments_by_addr.values():
            comments.sort()

        return comments_by_addr


def _is_unresolvable_jump_target(node: Node) -> bool:
    """Return whether ``node`` is angr's unresolved indirect-jump placeholder."""

    return (
        node.obj.is_simprocedure
        and node.obj.simprocedure_name == "UnresolvableJumpTarget"
    )


def _control_transfer_context(edge):
    """Return a block's instructions and the edge-producing transfer index."""

    source_node = edge.src.obj
    try:
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
    """Lift a Capstone-discovered control-transfer tail for classification."""

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

    CFGFast can cap VEX lifting before a recovered block's actual end. When
    Capstone finds no control transfer in the complete block, VEX's default
    target remains inside its byte range, and the source has one successor at
    that decoded end, the edge is the architectural fall-through. A VEX
    warning jumpkind may instead preserve the decoded next address exactly;
    that narrow case is also linear. Both patterns occur on multiple
    architectures after custom repair merges CFGFast fragments into one block.
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
    """Classify one ordinary edge from its source block's lifted terminator."""

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
    """Return the visual category for one CFG edge."""

    # ELF LSDA metadata proves that a call transfers to this landing pad only
    # while unwinding. Keep that known exceptional path distinct from both the
    # normal call edge and its fake-return continuation.
    if edge.meta.get("exceptional"):
        return "EXCEPTION"

    # Custom CFG repair may flatten an UnresolvableJumpTarget placeholder into
    # direct candidate edges. The marker keeps that unresolved semantics
    # visible after the synthetic endpoint itself has been removed.
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
    if jumpkind == "Ijk_Call" or (
        isinstance(jumpkind, str) and jumpkind.startswith("Ijk_Sys_")
    ):
        return "CALL"
    if jumpkind == "Ijk_Boring":
        return _vex_boring_edge_type(edge)

    logger.warning(
        f"Unexpected {jumpkind!r} edge type for "
        f"{edge.src.obj.addr:#x} -> {edge.dst.obj.addr:#x}"
    )
    return "UNKNOWN"


class ColorEdgesVex(EdgeAnnotator):
    """Apply semantic edge styles derived from VEX and repair metadata."""

    def annotate_edge(self, edge) -> None:
        """Style one edge without inferring branch kind from successor count."""

        get_style().make_edge(edge, _edge_type(edge))
