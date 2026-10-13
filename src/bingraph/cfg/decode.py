"""Function-bounded block decoding and normalized control-flow terminators.

Instruction bytes and cached node views come from helpers.disassembly; local
callee evidence and non-returning declarations come from cfg.calls. This layer
owns leader boundaries, delay slots, exceptional ISA transfers and BlockSpec
construction, without discovering a whole-function graph or resolving tables.
"""

from __future__ import annotations

from collections.abc import Callable

import pyvex
from angr import Project
from capstone import CsInsn
from loguru import logger

from bingraph.cfg.calls import (
    _is_known_synthetic_function_target,
    _is_static_pointer_call_target,
    _is_trusted_direct_call_target,
    _mips_gp_relative_indirect_call_target,
    _mips_gp_relative_indirect_jump_target,
    known_nonreturning_call_target,
    static_memory_indirect_call_target,
    static_memory_indirect_jump_target,
)
from bingraph.cfg.models import (
    BlockSpec,
    DecodedCallTargetKind,
    FunctionBounds,
    TerminatorInfo,
)
from bingraph.helpers.capstone import (
    InsnSemantics,
    arch_has_delay_slot,
    control_transfer_index,
    instruction_is_conditionally_executed,
    proven_unconditional_direct_target,
)
from bingraph.helpers.disassembly import decode_one, decode_raw_capstone_insns
from bingraph.helpers.vex import temporary_definitions


def decode_linear_vex_span(
    project: Project, addr: int, size: int
) -> tuple[int, int] | None:
    """Return one VEX-proven linear instruction Capstone cannot decode.

    This deliberately accepts only an unambiguous one-instruction lift with no
    side exits. It gives the custom builder a narrow fallback for valid
    ISA instructions missing from Capstone without using VEX to infer ordinary
    branch semantics.
    """

    try:
        block = project.factory.block(
            addr,
            size=size,
            num_inst=1,
            strict_block_end=True,
            cross_insn_opt=False,
        )
        vex = block.vex
    except Exception:
        return None

    marks = [
        statement
        for statement in vex.statements
        if isinstance(statement, pyvex.stmt.IMark)
    ]
    if len(marks) != 1:
        return None

    mark = marks[0]
    next_addr = getattr(getattr(vex.next, "con", None), "value", None)
    if (
        mark.addr != addr
        or mark.len <= 0
        or block.size != mark.len
        or vex.jumpkind != "Ijk_Boring"
        or vex.exit_statements
        or next_addr != addr + mark.len
    ):
        return None
    return mark.addr, mark.len


def is_post_prefix_instruction_entry(insn: CsInsn, addr: int) -> bool:
    """Return whether ``addr`` enters immediately after all instruction prefixes.

    Some x86 binaries deliberately branch after an instruction prefix, such as
    ``LOCK``. The remaining opcode bytes form a valid alternate instruction
    stream, unlike arbitrary entries into the middle of an instruction.
    """

    prefix_size = sum(1 for prefix in getattr(insn, "prefix", ()) if prefix)
    return (
        prefix_size > 0
        and addr == insn.address + prefix_size
        and addr < insn.address + insn.size
    )


def _block_insns(project: Project, block: BlockSpec) -> tuple[CsInsn, ...]:
    """Decode a recovered block with Capstone without constructing an angr Block."""

    return decode_raw_capstone_insns(project, block.addr, block.size)


def _containing_block_insn(
    project: Project,
    block: BlockSpec,
    addr: int,
) -> CsInsn | None:
    """Return the recovered instruction that strictly contains ``addr``."""

    return next(
        (
            insn
            for insn in _block_insns(project, block)
            if insn.address < addr < insn.address + insn.size
        ),
        None,
    )


def _max_instruction_bytes(project: Project) -> int:
    """Return the architecture's maximum instruction width with a safe default."""

    try:
        return project.arch.max_inst_bytes
    except AttributeError:
        return 16


def alternate_block_entry_rejoin_addr(
    project: Project,
    block: BlockSpec,
    addr: int,
) -> int | None:
    """Return the shared tail of a bounded alternate instruction stream.

    An entry inside an instruction is only accepted when decoding one
    instruction at that address reaches the original instruction's end. This
    covers x86 post-prefix streams and valid Thumb halfword alternate streams,
    while rejecting arbitrary mid-instruction targets.
    """

    insn = _containing_block_insn(project, block, addr)
    if insn is None:
        return None

    alternate = decode_raw_capstone_insns(
        project,
        addr,
        _max_instruction_bytes(project),
        count=1,
    )
    if len(alternate) != 1:
        return None

    rejoin_addr = insn.address + insn.size
    return (
        rejoin_addr if alternate[0].address + alternate[0].size == rejoin_addr else None
    )


def is_valid_block_entry(project: Project, block: BlockSpec, addr: int) -> bool:
    """Return whether ``addr`` is a normal or supported alternate leader."""

    return addr in block.instruction_addrs or (
        alternate_block_entry_rejoin_addr(project, block, addr) is not None
    )


def lift_instruction_vex(project: Project, insn: CsInsn):
    """Lift one instruction without inheriting CFG-node block boundaries."""

    try:
        return project.factory.block(
            insn.address,
            size=insn.size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception:
        return None


def vex_jumpkind_is_terminal(jumpkind: str) -> bool:
    """Return whether VEX marks a block as a return or synchronous trap."""

    return jumpkind == "Ijk_Ret" or jumpkind.startswith("Ijk_Sig")


def vex_jumpkind_is_syscall(jumpkind: str) -> bool:
    """Return whether VEX marks a transfer into an operating-system service."""

    return jumpkind.startswith("Ijk_Sys_")


def call_fallthrough_addr(
    project: Project, bounds: FunctionBounds, next_addr: int
) -> int | None:
    """Return a call continuation inside this function or at a known next one."""

    if bounds.addr <= next_addr < bounds.end_addr:
        return next_addr

    symbol = project.loader.find_symbol(next_addr)
    if symbol is not None and symbol.rebased_addr == next_addr and symbol.is_function:
        return next_addr

    return None


def _vex_exit_guard_reads_register(vex, exit_statement, register_offset: int) -> bool:
    """Return whether a VEX exit guard transitively reads one register."""

    definitions = temporary_definitions(vex)
    visited_temps: set[int] = set()

    def _reads_register(expression) -> bool:
        if isinstance(expression, pyvex.expr.Get):
            return expression.offset == register_offset
        if isinstance(expression, pyvex.expr.RdTmp):
            if expression.tmp in visited_temps:
                return False
            visited_temps.add(expression.tmp)
            definition = definitions.get(expression.tmp)
            return definition is not None and _reads_register(definition)
        return any(_reads_register(child) for child in expression.child_expressions)

    return _reads_register(exit_statement.guard)


def _is_unproven_itstate_fallthrough(
    project: Project,
    vex,
    exit_statements: list[pyvex.stmt.Exit],
    insns: list[CsInsn],
    terminator_index: int,
) -> bool:
    """Return whether VEX's next exit is only its generic inactive-IT path.

    libVEX models Thumb execution through the architectural ``itstate``
    register even when no preceding IT instruction predicates the current
    instruction. Its generic guard must not invent a return fall-through;
    Capstone independently proves whether the source instruction is actually
    predicated. Other architectures have no ``itstate`` register and retain
    their genuine conditional-return exits.
    """

    try:
        itstate_offset = project.arch.registers["itstate"][0]
    except KeyError:
        return False

    return not instruction_is_conditionally_executed(
        project.arch.name, insns, terminator_index
    ) and any(
        _vex_exit_guard_reads_register(vex, statement, itstate_offset)
        for statement in exit_statements
    )


def _exceptional_instruction_vex_jumpkind(project: Project, insn: CsInsn) -> str | None:
    """Return a terminal or syscall jumpkind for one exceptional instruction."""

    semantic = InsnSemantics(insn)
    if not semantic.may_have_nonfallthrough_vex_semantics():
        return None

    try:
        vex = project.factory.block(
            insn.address,
            size=insn.size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception as exc:
        logger.warning(
            f"CFG decoder could not lift exceptional instruction at "
            f"{insn.address:#x}: {exc}"
        )
        return None

    if vex_jumpkind_is_terminal(vex.jumpkind) or vex_jumpkind_is_syscall(vex.jumpkind):
        return vex.jumpkind
    return None


def _vex_sigill_is_linear(vex, insn: CsInsn) -> bool:
    """Keep a valid linear instruction when VEX cannot model its operation."""

    semantic = InsnSemantics(insn)
    return (
        vex.jumpkind == "Ijk_SigILL"
        and not semantic.is_control_transfer()
        and not semantic.is_undefined_instruction_trap()
        and vex.instruction_addresses
        and vex.instruction_addresses[-1] == insn.address
        and isinstance(vex.next, pyvex.expr.Const)
        and vex.next.con.value == insn.address + insn.size
    )


def _instruction_has_unclassified_vex_transfer(
    project: Project,
    insn: CsInsn,
    *,
    include_indirect: bool = False,
) -> str | None:
    """Return whether VEX proves control flow absent from Capstone's groups.

    ``include_indirect`` recognizes VEX's guarded ``LoadG``-to-PC shape. It
    is opt-in for callers that can recover a bounded table from that shape.
    """

    semantic = InsnSemantics(insn)
    if semantic.is_control_transfer():
        return None

    target = semantic.direct_target()
    has_executable_direct_target = False
    if target is not None:
        obj = project.loader.find_object_containing(target)
        section = obj.find_section_containing(target) if obj is not None else None
        has_executable_direct_target = bool(section and section.is_executable)
    if not include_indirect:
        if not has_executable_direct_target:
            return None

    lift_size = insn.size
    if arch_has_delay_slot(project.arch.name) and (
        not include_indirect or has_executable_direct_target
    ):
        # VEX needs the executed delay-slot instruction to classify MIPS BAL
        # and similar branch-and-link instructions as calls.
        delay_insn = decode_one(
            project,
            insn.address + insn.size,
            getattr(project.arch, "max_inst_bytes", 16),
        )
        if delay_insn is not None:
            lift_size += delay_insn.size

    try:
        vex = project.factory.block(
            insn.address,
            size=lift_size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex
    except Exception as exc:
        logger.warning(
            f"CFG decoder could not lift possible control transfer at "
            f"{insn.address:#x}: {exc}"
        )
        return None

    if _vex_sigill_is_linear(vex, insn):
        return None
    if vex.jumpkind == "Ijk_Call" or vex_jumpkind_is_terminal(vex.jumpkind):
        return vex.jumpkind

    if not include_indirect or vex.jumpkind != "Ijk_Boring":
        return None

    # Keep conditional computed-PC detection VEX-driven. This covers both a
    # guarded table load into PC and arithmetic PC dispatch without embedding
    # instruction-set-specific mnemonic rules in the decoder.
    from bingraph.cfg.jumps.conditional import vex_has_computed_pc_transfer

    return (
        vex.jumpkind
        if vex_has_computed_pc_transfer(vex, insn.address + insn.size)
        else None
    )


def _native_vex_transfer_end(
    project: Project,
    bounds: FunctionBounds,
    start_addr: int,
    *,
    split_syscall_blocks: bool = False,
) -> int | None:
    """Return a native VEX call or terminal boundary absent from Capstone groups."""

    try:
        block = project.factory.block(start_addr)
    except Exception:
        return None

    size = getattr(block, "size", None)
    jumpkind = getattr(block.vex, "jumpkind", None)
    if not isinstance(size, int) or not isinstance(jumpkind, str):
        return None

    end_addr = start_addr + size
    if not start_addr < end_addr <= bounds.end_addr:
        return None
    if jumpkind == "Ijk_SigILL" and block.vex.instruction_addresses:
        last_addr = block.vex.instruction_addresses[-1]
        insn = decode_one(
            project, last_addr, getattr(project.arch, "max_inst_bytes", 16)
        )
        if insn is not None and _vex_sigill_is_linear(block.vex, insn):
            return None
    if (
        jumpkind == "Ijk_Call"
        or vex_jumpkind_is_terminal(jumpkind)
        or (split_syscall_blocks and vex_jumpkind_is_syscall(jumpkind))
    ):
        return end_addr
    return None


def lift_block_terminator(
    project: Project,
    bounds: FunctionBounds,
    block_insns: list[CsInsn],
    has_nonfallthrough_vex_terminator: bool = False,
    unclassified_vex_terminator_addr: int | None = None,
    preserve_conditional_return_fallthrough: bool = False,
    split_syscall_blocks: bool = False,
    resolve_declared_nonreturning: bool = False,
    resolve_static_memory_calls: bool = False,
) -> TerminatorInfo:
    """Normalize one decoded block's actual transfer into TerminatorInfo.

    Locate the machine terminator, including delay-slot instructions, and
    prefer a focused tail lift over a possibly stale full-block VEX boundary.
    Distinguish direct destinations, genuine continuations, returns, traps,
    system transfers and unresolved register/memory targets. Opt-in call
    policies can resolve supported pointers and suppress a declared no-return
    continuation; they do not invoke whole-function or callee analysis.

    Failed control-transfer lifts raise a diagnostic RuntimeError rather than
    silently classifying unknown bytes as ordinary fallthrough. The result
    describes addresses and provenance, not graph nodes or rendered edges.
    """

    # Jump-table support consumes the basic decoding utilities in this module.
    # Delay the reverse dependency until terminator classification to avoid an
    # import cycle while still sharing its target-validity policy.
    from bingraph.cfg.jumps.expressions import (
        is_direct_target_valid,
        static_jump_target_rejection_reason,
    )

    block_end_addr = block_insns[-1].address + block_insns[-1].size

    term_idx = control_transfer_index(project.arch.name, block_insns)
    if term_idx is None and unclassified_vex_terminator_addr is not None:
        term_idx = next(
            (
                index
                for index, insn in enumerate(block_insns)
                if insn.address == unclassified_vex_terminator_addr
            ),
            None,
        )

    if term_idx is None:
        if has_nonfallthrough_vex_terminator:
            return TerminatorInfo(jumpkind="Ijk_Terminal")

        next_addr = block_end_addr
        fallthrough_addr = next_addr if next_addr < bounds.end_addr else None
        return TerminatorInfo(
            jumpkind="Ijk_Fallthrough", fallthrough_addr=fallthrough_addr
        )

    tail_insns = block_insns[term_idx:]
    last = tail_insns[0]
    semantic = InsnSemantics(last)
    next_addr = block_end_addr

    tail_addr = tail_insns[0].address
    tail_size = sum(insn.size for insn in tail_insns)
    block_addr = block_insns[0].address
    block_size = sum(insn.size for insn in block_insns)

    def _lift(addr: int, size: int):
        """Lift exact supplied bytes without cross-instruction optimization."""

        return project.factory.block(
            addr,
            size=size,
            strict_block_end=True,
            cross_insn_opt=False,
        ).vex

    try:
        vex = _lift(tail_addr, tail_size)
    except Exception:
        try:
            vex = _lift(block_addr, block_size)
        except Exception as exc:
            raise RuntimeError(
                f"CFG decoder failed lifting control-transfer block at {block_addr:#x} "
                f"(terminator {last.address:#x}: {last.mnemonic} {last.op_str})"
            ) from exc

    terminator_addrs = {last.address}
    if arch_has_delay_slot(project.arch.name) and len(tail_insns) > 1:
        terminator_addrs.add(tail_insns[1].address)

    exit_targets: list[int] = []
    next_exit_statements: list[pyvex.stmt.Exit] = []
    for ins_addr, _, stmt in vex.exit_statements:
        if ins_addr not in terminator_addrs:
            continue
        target = getattr(stmt.dst, "value", None)
        if isinstance(target, int):
            exit_targets.append(target)
            if target == next_addr:
                next_exit_statements.append(stmt)

    default_target: int | None = None
    if isinstance(vex.next, pyvex.expr.Const):
        target = vex.next.con.value
        if isinstance(target, int):
            default_target = target

    # A terminal VEX return can still carry an explicit ordinary Exit to the
    # next instruction. This represents the not-taken path of a conditional
    # return; the taken return target is caller-dependent and is not drawn.
    if (
        preserve_conditional_return_fallthrough
        and vex.jumpkind == "Ijk_Ret"
        and next_addr in exit_targets
        and not _is_unproven_itstate_fallthrough(
            project,
            vex,
            next_exit_statements,
            block_insns,
            term_idx,
        )
    ):
        fallthrough_addr = next_addr if next_addr < bounds.end_addr else None
        return TerminatorInfo(jumpkind="Ijk_Boring", fallthrough_addr=fallthrough_addr)

    if semantic.is_ret() or vex.jumpkind == "Ijk_Ret":
        return TerminatorInfo(jumpkind="Ijk_Ret")

    if split_syscall_blocks and vex_jumpkind_is_syscall(vex.jumpkind):
        fallthrough_addr = next_addr if next_addr < bounds.end_addr else None
        return TerminatorInfo(
            jumpkind="Ijk_Syscall",
            fallthrough_addr=fallthrough_addr,
            syscall_jumpkind=vex.jumpkind,
        )

    if vex_jumpkind_is_terminal(vex.jumpkind):
        return TerminatorInfo(jumpkind="Ijk_Terminal")

    if semantic.is_call() or vex.jumpkind == "Ijk_Call":
        direct_targets: tuple[int, ...] = ()
        target_kind: DecodedCallTargetKind | None = None
        if isinstance(default_target, int) and _is_trusted_direct_call_target(
            project, default_target
        ):
            # A constant VEX call target is precise even when it has no loader
            # symbol. The builder materializes an ExternalTarget leaf for
            # such unnamed callees, allowing later render policy to decide
            # whether it should be visible.
            direct_targets = (default_target,)
            target_kind = "vex_constant"
        call_vex = vex
        if project.arch.name.startswith("MIPS") and tail_addr != block_addr:
            # The tail lift intentionally excludes preceding instructions, but
            # MIPS PIC calls load $t9 through $gp before the call itself.
            try:
                call_vex = _lift(block_addr, block_size)
            except Exception:
                pass
        static_memory_target = (
            static_memory_indirect_call_target(project, call_vex)
            if resolve_static_memory_calls
            else None
        )
        memory_kind: DecodedCallTargetKind = "static_memory"
        if (
            static_memory_target is None
            and resolve_static_memory_calls
            and project.arch.name.startswith("MIPS")
        ):
            candidate = _mips_gp_relative_indirect_call_target(
                project, bounds, call_vex
            )
            if candidate is not None and _is_static_pointer_call_target(
                project, candidate
            ):
                static_memory_target = candidate
                memory_kind = "mips_gp"
        if static_memory_target is not None and not direct_targets:
            # A constant-address pointer load proves the same exact callee as
            # a direct VEX call while preserving the ordinary FakeRet edge.
            direct_targets = (static_memory_target,)
            target_kind = memory_kind
        nonreturning_vex = call_vex
        nonreturning_target = known_nonreturning_call_target(
            project,
            bounds,
            nonreturning_vex,
            resolve_declared_nonreturning=resolve_declared_nonreturning,
        )
        if nonreturning_target is not None and not direct_targets:
            # This static GOT-like call target is precise enough to retain as
            # a normal call edge, while suppressing its impossible FakeRet.
            direct_targets = (nonreturning_target,)
            target_kind = "nonreturning_memory"
        fallthrough_addr = (
            None
            if nonreturning_target is not None
            else call_fallthrough_addr(project, bounds, next_addr)
        )
        return TerminatorInfo(
            jumpkind="Ijk_Call",
            direct_targets=direct_targets,
            fallthrough_addr=fallthrough_addr,
            decoded_call_target_kind=target_kind,
            decoded_nonreturning_call=nonreturning_target is not None,
        )

    unconditional_target = proven_unconditional_direct_target(
        project.arch.name, block_insns, term_idx
    )
    if unconditional_target is not None:
        if unconditional_target == next_addr:
            # An adjacent direct jump still terminates the block, but its only
            # architectural successor is the ordinary linear continuation.
            return TerminatorInfo(
                jumpkind="Ijk_Fallthrough",
                fallthrough_addr=(next_addr if next_addr < bounds.end_addr else None),
            )
        return TerminatorInfo(
            jumpkind="Ijk_Boring", direct_targets=(unconditional_target,)
        )

    if exit_targets or semantic.is_conditional_jump():
        if last.address in exit_targets:
            direct_target = semantic.direct_target()
            if direct_target is None:
                direct_target = (
                    default_target
                    if isinstance(default_target, int)
                    and is_direct_target_valid(bounds, default_target)
                    else None
                )
            fallthrough_addr = next_addr if next_addr < bounds.end_addr else None
            return TerminatorInfo(
                jumpkind="Ijk_Boring",
                direct_targets=(direct_target,) if direct_target is not None else (),
                fallthrough_addr=fallthrough_addr,
            )

        all_targets: list[int] = list(exit_targets)
        if isinstance(default_target, int) and default_target not in all_targets:
            all_targets.append(default_target)
        fallthrough_addr = (
            next_addr
            if next_addr in all_targets and next_addr < bounds.end_addr
            else None
        )
        return TerminatorInfo(
            jumpkind="Ijk_Boring",
            direct_targets=tuple(
                target for target in all_targets if target != fallthrough_addr
            ),
            fallthrough_addr=fallthrough_addr,
        )

    if (
        semantic.is_jump()
        and isinstance(default_target, int)
        and is_direct_target_valid(bounds, default_target)
    ):
        return TerminatorInfo(jumpkind="Ijk_Boring", direct_targets=(default_target,))

    if semantic.is_jump():
        jump_vex = vex
        if project.arch.name.startswith("MIPS") and tail_addr != block_addr:
            # MIPS PIC tail jumps load $t9 before ``jr $t9``. Use the whole
            # bounded block so the exact local assignment remains available.
            try:
                jump_vex = _lift(block_addr, block_size)
            except Exception:
                pass
        static_memory_target = static_memory_indirect_jump_target(project, jump_vex)
        if static_memory_target is None:
            static_memory_target = _mips_gp_relative_indirect_jump_target(
                project,
                bounds,
                jump_vex,
            )
        rejection_reason = (
            static_jump_target_rejection_reason(project, static_memory_target)
            if static_memory_target is not None
            else None
        )
        is_known_import = (
            static_memory_target is not None
            and rejection_reason == "synthetic"
            and _is_known_synthetic_function_target(project, static_memory_target)
        )
        if static_memory_target is not None and (
            rejection_reason is None or is_known_import
        ):
            # Keep exact tail targets as normal boring edges. The renderer's
            # ``cfg_exits=jump`` policy then exposes out-of-function jumps,
            # while unresolved memory-derived targets still take the UJT path.
            return TerminatorInfo(
                jumpkind="Ijk_Boring", direct_targets=(static_memory_target,)
            )

    direct_target = semantic.direct_target()
    if semantic.is_jump() and direct_target is not None:
        return TerminatorInfo(jumpkind="Ijk_Boring", direct_targets=(direct_target,))

    direct_targets: tuple[int, ...] = ()
    if isinstance(default_target, int) and is_direct_target_valid(
        bounds, default_target
    ):
        direct_targets = (default_target,)
    return TerminatorInfo(jumpkind="Ijk_Boring", direct_targets=direct_targets)


def decode_bounded_block(
    project: Project,
    bounds: FunctionBounds,
    start_addr: int,
    stop_addrs: set[int],
    *,
    preserve_conditional_return_fallthrough: bool = False,
    split_syscall_blocks: bool = False,
    resolve_declared_nonreturning: bool = False,
    resolve_static_memory_calls: bool = False,
    split_unclassified_indirect_vex_transfers: bool = False,
    allow_vex_linear_fallback: bool = False,
    on_linear_direct_transfer: Callable[[int], None] | None = None,
    on_vex_linear_fallback: Callable[[int], None] | None = None,
    stop_at_data: Callable[[int], bool] | None = None,
) -> BlockSpec | None:
    """Recover one BlockSpec while respecting function bounds and leader stops.

    Decode instruction-by-instruction until a transfer, known leader, data
    barrier or native VEX boundary is reached. Include supported branch delay
    slots and optionally preserve narrowly proven VEX-only linear instructions
    missing from Capstone. Policy flags are forwarded to terminator handling;
    callbacks record decoding events without supplying branch evidence.

    If a direct target enters the recovered span, recursively decode with that
    address as a new stop so the returned block does not cover its own leader.
    Return ``None`` when no Capstone instruction was recovered; lifting errors
    from transfer normalization remain explicit. The caller closes successors
    and handles overlap with already established blocks.
    """

    max_inst_bytes = getattr(project.arch, "max_inst_bytes", 16)
    cur = start_addr
    insns: list[CsInsn] = []
    instruction_addrs: list[int] = []
    decoded_size = 0
    vex_linear_instruction_sizes: list[tuple[int, int]] = []
    last_was_vex_fallback = False
    has_delay_slot = arch_has_delay_slot(project.arch.name)
    has_nonfallthrough_vex_terminator = False
    unclassified_vex_terminator_addr: int | None = None
    # Capstone does not consistently group trap instructions as control flow.
    # Let VEX provide a native boundary on every architecture. Recognized
    # delay-slot branches still take the explicit delay-slot path below.
    native_vex_transfer_end = _native_vex_transfer_end(
        project,
        bounds,
        start_addr,
        split_syscall_blocks=split_syscall_blocks,
    )

    while bounds.addr <= cur < bounds.end_addr:
        if insns and cur in stop_addrs:
            break
        if insns and stop_at_data is not None and stop_at_data(cur):
            break

        insn = decode_one(project, cur, max_inst_bytes)
        if insn is None:
            vex_span = (
                decode_linear_vex_span(project, cur, max_inst_bytes)
                if allow_vex_linear_fallback
                else None
            )
            if vex_span is not None:
                _, vex_size = vex_span
                instruction_addrs.append(cur)
                decoded_size += vex_size
                vex_linear_instruction_sizes.append((cur, vex_size))
                last_was_vex_fallback = True
                if on_vex_linear_fallback is not None:
                    on_vex_linear_fallback(cur)
                cur += vex_size
                continue
            logger.warning(f"CFG decoder could not decode instruction at {cur:#x}")
            break

        insns.append(insn)
        instruction_addrs.append(insn.address)
        decoded_size += insn.size
        last_was_vex_fallback = False
        semantic = InsnSemantics(insn)
        next_addr = insn.address + insn.size

        is_linear_direct_transfer = semantic.is_linear_direct_jump(project.arch.name)
        if is_linear_direct_transfer and on_linear_direct_transfer is not None:
            on_linear_direct_transfer(insn.address)
        if semantic.is_control_transfer() and not is_linear_direct_transfer:
            if has_delay_slot and bounds.addr <= next_addr < bounds.end_addr:
                delay_insn = decode_one(project, next_addr, max_inst_bytes)
                if delay_insn is not None:
                    insns.append(delay_insn)
                    instruction_addrs.append(delay_insn.address)
                    decoded_size += delay_insn.size
            break

        if semantic.is_undefined_instruction_trap():
            # VEX reports x86 UD2 as Ijk_NoDecode. It is nevertheless an
            # intentional synchronous trap, never a linear fallthrough.
            has_nonfallthrough_vex_terminator = True
            break

        exceptional_jumpkind = _exceptional_instruction_vex_jumpkind(project, insn)
        if exceptional_jumpkind is not None:
            if split_syscall_blocks and vex_jumpkind_is_syscall(exceptional_jumpkind):
                unclassified_vex_terminator_addr = insn.address
                break
            if not vex_jumpkind_is_syscall(exceptional_jumpkind):
                has_nonfallthrough_vex_terminator = True
                break

        unclassified_vex_jumpkind = _instruction_has_unclassified_vex_transfer(
            project,
            insn,
            include_indirect=split_unclassified_indirect_vex_transfers,
        )
        if unclassified_vex_jumpkind is not None:
            unclassified_vex_terminator_addr = insn.address
            if (
                has_delay_slot
                and (
                    not split_unclassified_indirect_vex_transfers
                    or not vex_jumpkind_is_terminal(unclassified_vex_jumpkind)
                )
                and bounds.addr <= next_addr < bounds.end_addr
            ):
                delay_insn = decode_one(project, next_addr, max_inst_bytes)
                if delay_insn is not None:
                    insns.append(delay_insn)
                    instruction_addrs.append(delay_insn.address)
                    decoded_size += delay_insn.size
            break

        if next_addr == native_vex_transfer_end:
            unclassified_vex_terminator_addr = insn.address
            break

        cur = next_addr

    if not insns:
        return None

    if last_was_vex_fallback:
        fallthrough_addr = cur if cur < bounds.end_addr else None
        terminator = TerminatorInfo(
            jumpkind="Ijk_Fallthrough", fallthrough_addr=fallthrough_addr
        )
    else:
        terminator = lift_block_terminator(
            project,
            bounds,
            insns,
            has_nonfallthrough_vex_terminator=has_nonfallthrough_vex_terminator,
            unclassified_vex_terminator_addr=unclassified_vex_terminator_addr,
            preserve_conditional_return_fallthrough=preserve_conditional_return_fallthrough,
            split_syscall_blocks=split_syscall_blocks,
            resolve_declared_nonreturning=resolve_declared_nonreturning,
            resolve_static_memory_calls=resolve_static_memory_calls,
        )
    block = BlockSpec(
        addr=start_addr,
        size=decoded_size,
        instruction_addrs=tuple(instruction_addrs),
        jumpkind=terminator.jumpkind,
        vex_linear_instruction_sizes=tuple(vex_linear_instruction_sizes),
        direct_targets=terminator.direct_targets,
        fallthrough_addr=terminator.fallthrough_addr,
        syscall_jumpkind=terminator.syscall_jumpkind,
        decoded_call_target_kind=terminator.decoded_call_target_kind,
        decoded_nonreturning_call=terminator.decoded_nonreturning_call,
    )

    block_end = block.addr + block.size
    internal_targets = sorted(
        target
        for target in block.direct_targets
        if block.addr < target < block_end and target not in stop_addrs
    )
    if internal_targets:
        return decode_bounded_block(
            project,
            bounds,
            start_addr,
            stop_addrs | {internal_targets[0]},
            preserve_conditional_return_fallthrough=preserve_conditional_return_fallthrough,
            split_syscall_blocks=split_syscall_blocks,
            resolve_declared_nonreturning=resolve_declared_nonreturning,
            resolve_static_memory_calls=resolve_static_memory_calls,
            split_unclassified_indirect_vex_transfers=(
                split_unclassified_indirect_vex_transfers
            ),
            allow_vex_linear_fallback=allow_vex_linear_fallback,
            on_linear_direct_transfer=on_linear_direct_transfer,
            on_vex_linear_fallback=on_vex_linear_fallback,
            stop_at_data=stop_at_data,
        )

    return block
