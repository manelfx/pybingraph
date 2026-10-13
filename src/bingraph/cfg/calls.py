"""Local call-target evidence and non-returning runtime declarations.

Resolve only supported static-memory and MIPS GP-relative expressions; no
callee CFG or interprocedural execution is required. Entry GP recognition uses
the same ABI root proof as the jump planner and shared predecessor facts.
Non-return classification is independent of call-edge presentation."""

from __future__ import annotations

import re
from functools import lru_cache

import pyvex
from angr import Project
from angr.procedures.definitions import SIM_LIBRARIES, SimSyscallLibrary
from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from bingraph.cfg.models import FunctionBounds
from bingraph.helpers.abi import mips_entry_global_pointer
from bingraph.helpers.symbols import plt_symbol_name
from bingraph.helpers.vex import read_static_integer, temporary_definitions


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


def _library_family(name: str) -> str:
    """Return a stable family name for a versioned shared-library filename."""

    basename = name.rsplit("/", 1)[-1]
    match = re.match(
        r"(?P<family>.+?)(?:-[0-9][0-9A-Za-z._-]*)?\\.so(?:\\..*)?$", basename
    )
    return match.group("family") if match is not None else basename


_LINKED_NONRETURNING_RUNTIME_SYMBOLS = frozenset(
    {
        "abort",
        "_Unwind_Resume",
        "__assert_fail",
        "__libc_assert_fail",
        "__malloc_assert",
        "__stack_chk_fail",
    }
)


@lru_cache(maxsize=32)
def _ppc64_nonreturning_local_entries(project: Project) -> frozenset[int]:
    """Recognize metadata-defined ELFv2 aliases of linked no-return runtimes.

    PPC64 calls can enter after a function's TOC setup instead of at its global
    symbol address. Read the local-entry displacement from verified ELFv2
    symbols in the owning image, requiring the known runtime name and a valid
    in-symbol offset. Arbitrary interior addresses are not non-return aliases.
    Cache this file metadata per project; unreadable files contribute no aliases.
    """

    obj = project.loader.main_object
    if project.arch.name != "PPC64" or not obj.binary:
        return frozenset()

    entries: set[int] = set()
    try:
        with open(obj.binary, "rb") as stream:
            elf = ELFFile(stream)
            if elf["e_machine"] != "EM_PPC64" or elf["e_flags"] & 3 != 2:
                return frozenset()
            for section in elf.iter_sections():
                if not isinstance(section, SymbolTableSection):
                    continue
                for raw_symbol in section.iter_symbols():
                    if (
                        raw_symbol.name not in _LINKED_NONRETURNING_RUNTIME_SYMBOLS
                        or raw_symbol["st_info"]["type"] != "STT_FUNC"
                    ):
                        continue
                    symbol = obj.get_symbol(raw_symbol.name)
                    if (
                        symbol is None
                        or symbol.is_import
                        or symbol.relative_addr
                        != raw_symbol["st_value"] - obj.linked_base
                    ):
                        continue
                    encoded_offset = raw_symbol["st_other"]["local"]
                    # ELFv2 encodes the local entry's byte displacement in
                    # st_other; this is not an arbitrary in-function alias.
                    offset = ((1 << encoded_offset) >> 2) << 2
                    if 0 < offset < raw_symbol["st_size"]:
                        entries.add(symbol.rebased_addr + offset)
    except OSError:
        return frozenset()
    return frozenset(entries)


def _symbol_is_declared_nonreturning(project: Project, addr: int) -> bool:
    """Match a concrete target to the supported runtime/declaration evidence.

    Require an exact function symbol, or a validated PPC local-entry alias.
    Statically linked runtime names use the explicit supported set; imported
    declarations must belong to a compatible owning library with an angr
    non-returning prototype. Do not infer no-return behavior from control-flow
    shape, a fuzzy nearby symbol or an unrelated library's same-named function.
    """

    symbol = project.loader.find_symbol(addr)
    if symbol is None or symbol.rebased_addr != addr or not symbol.is_function:
        return addr in _ppc64_nonreturning_local_entries(project)

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
    """Use an installed SimProcedure's explicit NO_RET property at this address.

    This is the minimal hook-only policy. Symbol/declaration lookup is kept
    separate so decoder callers can opt into broader runtime evidence without
    changing the behavior of callers that request only installed hooks.
    """

    return bool(
        project.is_hooked(addr) and getattr(project.hooked_by(addr), "NO_RET", False)
    )


def target_is_known_nonreturning(project: Project, addr: int) -> bool:
    """Combine supported no-return evidence without analyzing the callee body.

    Accept an explicit hook, compatible runtime declaration or exact reverse-
    PLT identity in the known runtime set. The caller may then suppress a call
    continuation, but must establish its callee target independently; an
    unresolved call is never non-returning merely because one candidate is.
    """

    return (
        target_is_hooked_nonreturning(project, addr)
        or _symbol_is_declared_nonreturning(project, addr)
        or plt_symbol_name(project, addr) in _LINKED_NONRETURNING_RUNTIME_SYMBOLS
    )


def _mips_gp_relative_indirect_slot(
    project: Project, bounds: FunctionBounds, vex
) -> tuple[int, str] | None:
    """Find the supported indirect target load, then describe its GP-relative slot.

    Local VEX evidence must associate the load with NEXT or the admitted T9
    assignment. Return the slot address and endianness, not the loaded callee;
    reading the pointer and validating its destination are separate steps.
    """

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
    """Combine the shared entry GP root with the admitted local load displacement.

    Require the narrow VEX ``Load(RdTmp(Add(Const, RdTmp(Get(gp)))))`` address
    shape in either operand order. Wrap the slot at architecture width and
    preserve load endianness. More complex register-derived addresses require
    the separate bounded propagation/table analyses; this does not prove GP
    preservation along arbitrary paths or inspect the pointer value itself.
    """

    global_pointer = mips_entry_global_pointer(project, bounds.addr)
    if global_pointer is None:
        return None

    try:
        gp_offset = project.arch.registers["gp"][0]
    except KeyError:
        return None

    if not isinstance(target_expr.addr, pyvex.expr.RdTmp):
        return None

    definitions = temporary_definitions(vex)
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
    """Resolve the admitted GP-slot load followed by a local T9 adjustment.

    Match ``lw $t9, offset($gp); addiu $t9, $t9, imm; jr $t9`` through lifted
    assignments, requiring the earlier value to come from the identified load.
    Read its pointer and apply the constant adjustment at machine width.
    This local extension does not follow predecessor writes or arbitrary
    arithmetic chains, and callers still validate the final branch address.
    """

    if vex.jumpkind != "Ijk_Boring" or not project.arch.name.startswith("MIPS"):
        return None
    try:
        t9_offset = project.arch.registers["t9"][0]
    except KeyError:
        return None

    definitions = temporary_definitions(vex)
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
    target = read_static_integer(project, slot_addr, project.arch.bytes, endness)
    if target is None:
        return None
    mask = (1 << project.arch.bits) - 1
    return (target + constant.con.value) & mask


def _indirect_target_load(project: Project, vex) -> pyvex.expr.Load | None:
    """Return a direct VEX load supplying an indirect transfer target."""

    if not isinstance(vex.next, pyvex.expr.RdTmp):
        return None

    definitions = temporary_definitions(vex)
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


def _is_known_synthetic_function_target(project: Project, addr: int) -> bool:
    """Check CLE's exact synthetic function identity for an imported call target.

    The extern object is not ordinary executable bytes, but a resolved function
    symbol there can legitimately identify a callee. Require exact rebasing and
    function type; a merely present, nearby or data symbol cannot validate it.
    """

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
    """Validate a loaded call pointer as executable code or an exact extern function.

    Reject unmapped weak-zero targets. For normal objects require executable
    section/segment evidence; for CLE extern addresses require its exact
    synthetic function symbol. This validates a destination, not the stability
    of a memory slot or the contents of an unproved indirect-call expression.
    """

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

    definitions = temporary_definitions(vex)
    load = definitions.get(vex.next.tmp)
    if not isinstance(load, pyvex.expr.Load) or not isinstance(
        load.addr, pyvex.expr.Const
    ):
        return None

    slot_addr = load.addr.con.value
    entry_size = load.result_size(vex.tyenv) // 8
    if not isinstance(slot_addr, int):
        return None

    return read_static_integer(project, slot_addr, entry_size, load.end)


def static_memory_indirect_jump_target(project: Project, vex) -> int | None:
    """Read the pointer for the supported constant-slot ordinary jump shape.

    Return the loaded integer or unknown. Unlike the call helper below, this
    does not validate code destinations itself; jump consumers apply their
    executable-region and function/external-target policies before using it.
    """

    return _static_memory_indirect_target(project, vex, "Ijk_Boring")


def static_memory_indirect_call_target(project: Project, vex) -> int | None:
    """Read and validate one callee from the supported constant-slot call shape.

    Reuse the same scalar-read representation as jumps, then require executable
    or exact synthetic function evidence. An unreadable pointer or invalid
    destination remains unresolved instead of becoming an external callee leaf.
    """

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
    return read_static_integer(project, slot_addr, project.arch.bytes, endness)


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
    return read_static_integer(project, slot_addr, project.arch.bytes, endness)


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

    target = read_static_integer(
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
    """Resolve only a concrete target with the selected no-return evidence policy.

    Try constant NEXT first, then the admitted static/GP-relative pointer-load
    form. By default trust installed hooks only; opting into declarations also
    enables supported runtime/PLT metadata. Return ``None`` when either target
    recovery or the no-return predicate is inconclusive, preserving the normal
    continuation rather than analyzing arbitrary callees to infer termination.
    """

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
