"""Capstone-backed node inspection used by custom CFG reconstruction."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import re

from angr import Project
from angr.procedures.definitions import SIM_LIBRARIES, SimSyscallLibrary
from capstone import CsInsn
from loguru import logger
import pyvex

from bingraph.helpers.capstone import (
    InsnSemantics,
    arch_has_delay_slot,
    control_transfer_index,
    instruction_is_conditionally_executed,
    proven_unconditional_direct_target,
)
from bingraph.helpers.symbols import plt_symbol_name

from .models import BlockSpec, FunctionBounds, TerminatorInfo


# CFGNode equality is address/block-ID based, so a recovered replacement can
# compare equal to the stale node it supersedes. Keep the object alive in each
# entry and key by identity so replacement nodes never reuse stale decoding.
_DECODED_NODE_CACHE: dict[int, tuple[object, DecodedNode]] = {}


def clear_decoded_node_cache() -> None:
    """Discard decoded-node views from the preceding custom CFG build."""

    _DECODED_NODE_CACHE.clear()


def decode_raw_capstone_insns(
    project: Project,
    addr: int,
    size: int,
    *,
    count: int = 0,
) -> tuple[CsInsn, ...]:
    """Decode bytes directly through the project's architecture Capstone engine."""

    try:
        arch = project.arch
        try:
            is_thumb = arch.is_thumb(addr)
        except AttributeError:
            is_thumb = False

        # ARM/Thumb stores the execution mode in address bit 0. Read bytes at
        # the physical address, while preserving the tagged address in output.
        # ``capstone_thumb`` is an ARM-specific extension not declared on the
        # base archinfo ``Arch`` type.
        thumb_capstone = getattr(arch, "capstone_thumb", None)

        if is_thumb and thumb_capstone is not None:
            capstone = thumb_capstone
            memory_addr = addr & ~1
        else:
            capstone = arch.capstone
            memory_addr = addr
        data = project.loader.memory.load(memory_addr, size)
        return tuple(capstone.disasm(data, addr, count=count))
    except Exception:
        return ()


def decode_one(project: Project, addr: int, size: int) -> CsInsn | None:
    """Decode one instruction with mode-aware Capstone and a raw fallback."""

    try:
        block = project.factory.block(
            addr,
            size=size,
            strict_block_end=True,
            cross_insn_opt=False,
        )
        capstone_insns = block.capstone.insns
        if capstone_insns:
            return capstone_insns[0].insn
    except Exception:
        pass

    fallback_insns = decode_raw_capstone_insns(project, addr, size, count=1)
    return fallback_insns[0] if fallback_insns else None


def decode_linear_vex_span(
    project: Project, addr: int, size: int
) -> tuple[int, int] | None:
    """Return one VEX-proven linear instruction Capstone cannot decode.

    This deliberately accepts only an unambiguous one-instruction lift with no
    side exits. It gives the independent extractor a narrow fallback for valid
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


def _is_trusted_direct_call_target(project: Project, addr: int) -> bool:
    """Return whether a constant VEX call target is safe to materialize.

    Address zero can be a valid code address in a deliberately zero-based
    image. In an image that does not map zero, however, it commonly denotes an
    unresolved weak function reference. Do not turn such a reference into a
    concrete external callee merely because another zero-valued symbol exists.
    """

    if addr != 0:
        return True
    try:
        return project.loader.find_object_containing(addr) is not None
    except AttributeError:
        return False


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


def _library_family(name: str) -> str:
    """Return a stable family name for a versioned shared-library filename."""

    basename = name.rsplit("/", 1)[-1]
    match = re.match(
        r"(?P<family>.+?)(?:-[0-9][0-9A-Za-z._-]*)?\\.so(?:\\..*)?$", basename
    )
    return match.group("family") if match is not None else basename


_LINKED_NONRETURNING_RUNTIME_SYMBOLS = frozenset(
    {"__assert_fail", "__libc_assert_fail", "__malloc_assert", "__stack_chk_fail"}
)


def _symbol_is_declared_nonreturning(project: Project, addr: int) -> bool:
    """Recognize exact no-return runtime symbols and compatible declarations."""

    symbol = project.loader.find_symbol(addr)
    if symbol is None or symbol.rebased_addr != addr or not symbol.is_function:
        return False

    # A statically linked runtime has an application filename, so its libc
    # declarations cannot be matched by the owning object's library name.
    if (
        symbol.owner is project.loader.main_object
        and not symbol.is_import
        and symbol.name in _LINKED_NONRETURNING_RUNTIME_SYMBOLS
    ):
        return True

    binary = project.loader.find_object_containing(addr)
    if binary is None:
        return False

    binary_names = (binary.provides, binary.binary)
    families = {_library_family(name) for name in binary_names if name}
    for library_name, libraries in SIM_LIBRARIES.items():
        if _library_family(library_name) not in families:
            continue
        for library in libraries:
            if isinstance(library, SimSyscallLibrary):
                continue
            if library.has_prototype(symbol.name) and not library.is_returning(
                symbol.name
            ):
                return True
    return False


def target_is_hooked_nonreturning(project: Project, addr: int) -> bool:
    """Return whether angr exposes a concrete target as a no-return hook."""

    return bool(
        project.is_hooked(addr) and getattr(project.hooked_by(addr), "NO_RET", False)
    )


def target_is_known_nonreturning(project: Project, addr: int) -> bool:
    """Return whether static information declares a target non-returning."""

    return (
        target_is_hooked_nonreturning(project, addr)
        or _symbol_is_declared_nonreturning(project, addr)
        or plt_symbol_name(project, addr) == "_Unwind_Resume"
    )


def _temporary_definitions(vex) -> dict[int, object]:
    """Return the VEX expressions defining temporary values in ``vex``."""

    return {
        statement.tmp: statement.data
        for statement in vex.statements
        if isinstance(statement, pyvex.stmt.WrTmp)
    }


def _vex_exit_guard_reads_register(vex, exit_statement, register_offset: int) -> bool:
    """Return whether a VEX exit guard transitively reads one register."""

    definitions = _temporary_definitions(vex)
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


def _mips_entry_global_pointer(project: Project, bounds: FunctionBounds) -> int | None:
    """Resolve a MIPS PIC global pointer initialized from the entry ``$t9``.

    MIPS PIC functions commonly establish ``$gp`` as ``$t9 + constant`` at
    entry, where ABI rules guarantee that the incoming ``$t9`` is the function
    address. Restrict this to the exact VEX shape so calls reached through
    arbitrary register state remain unresolved.
    """

    if not project.arch.name.startswith("MIPS"):
        return None

    try:
        gp_offset = project.arch.registers["gp"][0]
        t9_offset = project.arch.registers["t9"][0]
        entry_vex = project.factory.block(bounds.addr).vex
    except (AttributeError, KeyError):
        return None
    except Exception:
        return None

    definitions = _temporary_definitions(entry_vex)
    for statement in entry_vex.statements:
        if not isinstance(statement, pyvex.stmt.Put) or statement.offset != gp_offset:
            continue
        if not isinstance(statement.data, pyvex.expr.RdTmp):
            continue

        expression = definitions.get(statement.data.tmp)
        if not isinstance(expression, pyvex.expr.Binop) or not expression.op.startswith(
            "Iop_Add"
        ):
            continue

        left, right = expression.args
        constant, register = (
            (left, right) if isinstance(left, pyvex.expr.Const) else (right, left)
        )
        if not isinstance(constant, pyvex.expr.Const):
            continue
        if not isinstance(register, pyvex.expr.RdTmp):
            continue

        source = definitions.get(register.tmp)
        if not isinstance(source, pyvex.expr.Get) or source.offset != t9_offset:
            continue

        mask = (1 << project.arch.bits) - 1
        return (bounds.addr + constant.con.value) & mask

    return None


def _mips_gp_relative_indirect_slot(
    project: Project, bounds: FunctionBounds, vex
) -> tuple[int, str] | None:
    """Resolve a MIPS PIC indirect transfer through ``Load($gp + offset)``."""

    target_expr = _indirect_target_load(project, vex)
    if target_expr is None:
        return None
    return _mips_gp_relative_slot_from_load(project, bounds, vex, target_expr)


def _mips_gp_relative_slot_from_load(
    project: Project,
    bounds: FunctionBounds,
    vex,
    target_expr: pyvex.expr.Load,
) -> tuple[int, str] | None:
    """Return a static ``$gp``-relative load slot used for a MIPS target."""

    global_pointer = _mips_entry_global_pointer(project, bounds)
    if global_pointer is None:
        return None

    try:
        gp_offset = project.arch.registers["gp"][0]
    except KeyError:
        return None

    if not isinstance(target_expr.addr, pyvex.expr.RdTmp):
        return None

    definitions = _temporary_definitions(vex)
    slot_expr = definitions.get(target_expr.addr.tmp)
    if not isinstance(slot_expr, pyvex.expr.Binop) or not slot_expr.op.startswith(
        "Iop_Add"
    ):
        return None

    left, right = slot_expr.args
    constant, register = (
        (left, right) if isinstance(left, pyvex.expr.Const) else (right, left)
    )
    if not isinstance(constant, pyvex.expr.Const):
        return None
    if not isinstance(register, pyvex.expr.RdTmp):
        return None

    source = definitions.get(register.tmp)
    if not isinstance(source, pyvex.expr.Get) or source.offset != gp_offset:
        return None

    mask = (1 << project.arch.bits) - 1
    return ((global_pointer + constant.con.value) & mask, target_expr.end)


def _mips_gp_relative_adjusted_indirect_jump_target(
    project: Project, bounds: FunctionBounds, vex
) -> int | None:
    """Resolve ``lw $t9, offset($gp); addiu $t9, $t9, imm; jr $t9``."""

    if vex.jumpkind != "Ijk_Boring" or not project.arch.name.startswith("MIPS"):
        return None
    try:
        t9_offset = project.arch.registers["t9"][0]
    except KeyError:
        return None

    definitions = _temporary_definitions(vex)
    if not isinstance(vex.next, pyvex.expr.RdTmp):
        return None
    target_expr = definitions.get(vex.next.tmp)
    if not isinstance(target_expr, pyvex.expr.Get) or target_expr.offset != t9_offset:
        return None

    assignments = [
        statement.data.tmp
        for statement in vex.statements
        if isinstance(statement, pyvex.stmt.Put)
        and statement.offset == t9_offset
        and isinstance(statement.data, pyvex.expr.RdTmp)
    ]
    if len(assignments) < 2:
        return None

    adjusted = definitions.get(assignments[-1])
    if not isinstance(adjusted, pyvex.expr.Binop) or not adjusted.op.startswith(
        "Iop_Add"
    ):
        return None
    left, right = adjusted.args
    constant, register = (
        (left, right) if isinstance(left, pyvex.expr.Const) else (right, left)
    )
    if not isinstance(constant, pyvex.expr.Const) or not isinstance(
        register, pyvex.expr.RdTmp
    ):
        return None
    register_source = definitions.get(register.tmp)
    if not isinstance(register_source, pyvex.expr.Get) or (
        register_source.offset != t9_offset
    ):
        return None

    load = definitions.get(assignments[-2])
    if not isinstance(load, pyvex.expr.Load):
        return None
    slot = _mips_gp_relative_slot_from_load(project, bounds, vex, load)
    if slot is None:
        return None

    slot_addr, endness = slot
    target = _read_static_pointer_target(
        project, slot_addr, project.arch.bytes, endness
    )
    if target is None:
        return None
    mask = (1 << project.arch.bits) - 1
    return (target + constant.con.value) & mask


def _indirect_target_load(project: Project, vex) -> pyvex.expr.Load | None:
    """Return a direct VEX load supplying an indirect transfer target."""

    if not isinstance(vex.next, pyvex.expr.RdTmp):
        return None

    definitions = _temporary_definitions(vex)
    target_expr = definitions.get(vex.next.tmp)
    if isinstance(target_expr, pyvex.expr.Load):
        return target_expr

    # Strict MIPS lifts retain the call target as GET($t9), while the preceding
    # instruction writes that register from the PIC GOT. Follow just this one
    # register assignment; arbitrary register-derived calls stay unresolved.
    if not project.arch.name.startswith("MIPS"):
        return None
    try:
        t9_offset = project.arch.registers["t9"][0]
    except KeyError:
        return None
    if not isinstance(target_expr, pyvex.expr.Get) or target_expr.offset != t9_offset:
        return None

    assigned = next(
        (
            statement.data
            for statement in reversed(vex.statements)
            if isinstance(statement, pyvex.stmt.Put)
            and statement.offset == t9_offset
            and isinstance(statement.data, pyvex.expr.RdTmp)
        ),
        None,
    )
    if assigned is None:
        return None
    load = definitions.get(assigned.tmp)
    return load if isinstance(load, pyvex.expr.Load) else None


def _read_static_pointer_target(
    project: Project,
    slot_addr: int,
    entry_size: int,
    endness: str,
) -> int | None:
    """Read one statically addressed pointer with VEX-proven representation."""

    if entry_size not in {1, 2, 4, 8}:
        return None
    try:
        raw_target = project.loader.memory.load(slot_addr, entry_size)
    except Exception:
        return None

    byteorder = "little" if endness == "Iend_LE" else "big"
    return int.from_bytes(raw_target, byteorder=byteorder)


def _is_known_synthetic_function_target(project: Project, addr: int) -> bool:
    """Return whether CLE identifies a synthetic address as an exact function."""

    try:
        symbol = project.loader.find_symbol(addr)
    except Exception:
        return False
    return bool(
        symbol is not None
        and getattr(symbol, "rebased_addr", None) == addr
        and getattr(symbol, "is_function", False)
    )


def _is_static_pointer_call_target(project: Project, addr: int) -> bool:
    """Return whether a static pointer supplies a safe concrete call target."""

    if not _is_trusted_direct_call_target(project, addr):
        return False
    try:
        obj = project.loader.find_object_containing(addr)
    except Exception:
        return False
    if obj is None:
        return False
    if obj is getattr(project.loader, "extern_object", None):
        return _is_known_synthetic_function_target(project, addr)

    for method_name in ("find_section_containing", "find_segment_containing"):
        method = getattr(obj, method_name, None)
        region = method(addr) if callable(method) else None
        if region is not None:
            return bool(getattr(region, "is_executable", False))
    return False


def _static_memory_indirect_target(project: Project, vex, jumpkind: str) -> int | None:
    """Resolve an exact constant-slot indirect transfer target, if present.

    This recognizes only ``next = Load(Const(slot))``.  In particular, it does
    not follow register-derived addresses or table indices, which remain the
    responsibility of the bounded static jump-table planner.
    """

    if vex.jumpkind != jumpkind or not isinstance(vex.next, pyvex.expr.RdTmp):
        return None

    definitions = _temporary_definitions(vex)
    load = definitions.get(vex.next.tmp)
    if not isinstance(load, pyvex.expr.Load) or not isinstance(
        load.addr, pyvex.expr.Const
    ):
        return None

    slot_addr = load.addr.con.value
    entry_size = load.result_size(vex.tyenv) // 8
    if not isinstance(slot_addr, int):
        return None

    return _read_static_pointer_target(project, slot_addr, entry_size, load.end)


def static_memory_indirect_jump_target(project: Project, vex) -> int | None:
    """Resolve an exact static-memory indirect jump target, if present."""

    return _static_memory_indirect_target(project, vex, "Ijk_Boring")


def static_memory_indirect_call_target(project: Project, vex) -> int | None:
    """Resolve an exact static-memory indirect call target, if present."""

    target = _static_memory_indirect_target(project, vex, "Ijk_Call")
    if target is None or not _is_static_pointer_call_target(project, target):
        return None
    return target


def _mips_gp_relative_indirect_jump_target(
    project: Project, bounds: FunctionBounds, vex
) -> int | None:
    """Resolve an exact MIPS PIC ``jr $t9`` target through its GOT slot."""

    if vex.jumpkind != "Ijk_Boring":
        return None
    slot = _mips_gp_relative_indirect_slot(project, bounds, vex)
    if slot is None:
        return _mips_gp_relative_adjusted_indirect_jump_target(project, bounds, vex)

    slot_addr, endness = slot
    return _read_static_pointer_target(project, slot_addr, project.arch.bytes, endness)


def _mips_gp_relative_indirect_call_target(
    project: Project, bounds: FunctionBounds, vex
) -> int | None:
    """Resolve an exact MIPS PIC ``jalr $t9`` target through its GOT slot."""

    if vex.jumpkind != "Ijk_Call":
        return None
    slot = _mips_gp_relative_indirect_slot(project, bounds, vex)
    if slot is None:
        return None

    slot_addr, endness = slot
    return _read_static_pointer_target(project, slot_addr, project.arch.bytes, endness)


def _static_memory_nonreturning_call_target(
    project: Project,
    bounds: FunctionBounds,
    vex,
    *,
    resolve_declared_nonreturning: bool,
) -> int | None:
    """Resolve a call through one constant-address pointer to a no-return target.

    This intentionally covers only the simple GOT-like VEX shape where the
    call's ``next`` value comes directly from ``LDle/LDbe(Const(slot))``. It
    avoids treating arbitrary memory-derived indirect calls as resolved.
    """

    load = _indirect_target_load(project, vex)
    if load is None:
        return None

    slot: tuple[int, str] | None = None
    if isinstance(load.addr, pyvex.expr.Const):
        slot_addr = load.addr.con.value
        if isinstance(slot_addr, int):
            slot = (slot_addr, load.end)
    if slot is None:
        slot = _mips_gp_relative_indirect_slot(project, bounds, vex)
    if slot is None:
        return None

    target = _read_static_pointer_target(
        project,
        slot[0],
        project.arch.bytes,
        slot[1],
    )
    if target is None:
        return None

    predicate = (
        target_is_known_nonreturning
        if resolve_declared_nonreturning
        else target_is_hooked_nonreturning
    )
    return target if predicate(project, target) else None


def known_nonreturning_call_target(
    project: Project,
    bounds: FunctionBounds,
    vex,
    *,
    resolve_declared_nonreturning: bool = False,
) -> int | None:
    """Return a conservatively resolved non-returning call target, if any."""

    predicate = (
        target_is_known_nonreturning
        if resolve_declared_nonreturning
        else target_is_hooked_nonreturning
    )
    if isinstance(vex.next, pyvex.expr.Const):
        target = vex.next.con.value
        if isinstance(target, int) and predicate(project, target):
            return target

    return _static_memory_nonreturning_call_target(
        project,
        bounds,
        vex,
        resolve_declared_nonreturning=resolve_declared_nonreturning,
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


def _instruction_has_unclassified_vex_transfer(
    project: Project,
    insn: CsInsn,
    *,
    include_indirect: bool = False,
) -> str | None:
    """Return whether VEX proves control flow absent from Capstone's groups.

    ``include_indirect`` recognizes VEX's guarded ``LoadG``-to-PC shape. It
    is opt-in because CFG extraction can recover a bounded table from that
    shape, whereas CFGFast repair deliberately keeps its existing conservative
    decoding behavior.
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

    if vex.jumpkind == "Ijk_Call" or vex_jumpkind_is_terminal(vex.jumpkind):
        return vex.jumpkind

    if not include_indirect or vex.jumpkind != "Ijk_Boring":
        return None

    # Keep conditional computed-PC detection VEX-driven. This covers both a
    # guarded table load into PC and arithmetic PC dispatch without embedding
    # instruction-set-specific mnemonic rules in the decoder.
    from .jumps import vex_has_computed_pc_transfer

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
    """Lift decoded block bytes and derive their control-flow shape."""

    # Jump-table support consumes the basic decoding utilities in this module.
    # Delay the reverse dependency until terminator classification to avoid an
    # import cycle while still sharing its target-validity policy.
    from .jumps import is_direct_target_valid, static_jump_target_rejection_reason

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
        if isinstance(default_target, int) and _is_trusted_direct_call_target(
            project, default_target
        ):
            # A constant VEX call target is precise even when it has no loader
            # symbol. The extractor materializes an ExternalTarget leaf for
            # such unnamed callees, allowing later render policy to decide
            # whether it should be visible.
            direct_targets = (default_target,)
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
        if static_memory_target is not None and not direct_targets:
            # A constant-address pointer load proves the same exact callee as
            # a direct VEX call while preserving the ordinary FakeRet edge.
            direct_targets = (static_memory_target,)
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
        fallthrough_addr = (
            None
            if nonreturning_target is not None
            else call_fallthrough_addr(project, bounds, next_addr)
        )
        return TerminatorInfo(
            jumpkind="Ijk_Call",
            direct_targets=direct_targets,
            fallthrough_addr=fallthrough_addr,
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
    """Decode one bounded block until control flow or a known leader stops it."""

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


@dataclass(frozen=True)
class DecodedNode:
    """The Capstone instruction view of one CFG node, when it is available."""

    insns: tuple[CsInsn, ...] | None
    inspection_error: Exception | None = None

    @classmethod
    def from_node(cls, node) -> DecodedNode:
        """Read and cache Capstone instructions for one live CFG node.

        Anomaly checks inspect the same CFGFast nodes repeatedly while the
        worklist repairs nearby blocks. ``node.block.capstone`` constructs a
        fresh angr Block each time, so retaining this immutable view avoids
        repeatedly disassembling unchanged node bytes. The cache is scoped to
        object identity because angr CFG nodes compare by block identity; a
        recovered replacement can otherwise collide with the stale node it
        replaced at the same address.
        """

        cache_key = id(node)
        cached = _DECODED_NODE_CACHE.get(cache_key)
        if cached is not None and cached[0] is node:
            return cached[1]

        try:
            block = node.block
            insns = tuple(item.insn for item in block.capstone.insns)
            if insns:
                decoded = cls(insns)
            else:
                project = block._project
                decoded = cls(decode_raw_capstone_insns(project, node.addr, node.size))

        except Exception as exc:
            decoded = cls(None, exc)

        _DECODED_NODE_CACHE[cache_key] = node, decoded
        return decoded

    @property
    def is_empty(self) -> bool:
        """Return whether Capstone found no instructions in the node."""

        return not self.insns

    @property
    def last(self) -> CsInsn | None:
        """Return the final decoded instruction, if one exists."""

        return self.insns[-1] if self.insns else None

    def has_exact_coverage(self, node) -> bool:
        """Return whether instructions exactly cover the node's declared range."""

        if node.size == 0 or self.insns is None:
            return False
        expected_addr = node.addr
        for insn in self.insns:
            if insn.address != expected_addr:
                return False
            expected_addr += insn.size
        return expected_addr == node.addr + node.size

    def contains_mid_instruction_addr(self, addr: int) -> bool:
        """Return whether ``addr`` falls strictly inside a decoded instruction."""

        if self.insns is None:
            return False
        return any(
            insn.address < addr < insn.address + insn.size for insn in self.insns
        )
