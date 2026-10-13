"""Local ABI roots and register-preservation facts shared by binary analyses.

These helpers inspect entry lifts, verified thunk bodies and loader metadata,
not CFG paths or callees. Consumers still own reaching definitions, aliases,
call barriers and proof budgets. In particular, a known entry GP does not mean
that GP survives a call or remains unchanged at a later use.
"""

from __future__ import annotations

import pyvex
from angr import Project

from .vex import (
    _resolve_vex_expr,
    _vex_add_terms,
    _vex_const_value,
    _vex_get_key,
    _vex_last_put,
    temporary_definitions,
)

# ELF integer preservation only. SP's call-side VEX update is not the returned
# stack state; private-frame analyses must supply their own stack policy.
_MIPS_PRESERVED = tuple(f"s{i}" for i in range(8)) + ("fp",)
_ARM_PRESERVED = tuple(f"r{i}" for i in range(4, 12))
_PRESERVED = {
    "AMD64": ("rbx", "rbp", "r12", "r13", "r14", "r15"),
    "X86": ("ebx", "ebp", "esi", "edi"),
    "MIPS32": _MIPS_PRESERVED,
    "MIPS64": _MIPS_PRESERVED,
    "PPC64": tuple(f"r{i}" for i in range(14, 32)),
    "S390X": tuple(f"r{i}" for i in range(6, 14)),
    "ARMEL": _ARM_PRESERVED,
    "ARMHF": _ARM_PRESERVED,
}


def callee_preserved_register_offsets(project: Project) -> set[int]:
    """Return integer-register offsets preserved by a supported ELF ABI.

    Use archinfo's register layout rather than hardcoded offsets so consumers
    can track VEX definitions. Unsupported object formats and architectures
    yield no preservation facts. SP is deliberately excluded: VEX's call-side
    stack update does not describe the stack after return. Alias widths and
    private-frame saves remain the consuming analysis's responsibility.
    """

    elf = getattr(project.loader.main_object, "os", "") == "UNIX - System V"
    return {
        project.arch.registers[name][0]
        for name in _PRESERVED.get(project.arch.name, ())
        if elf and name in project.arch.registers
    }


def mips_entry_global_pointer(project: Project, entry_addr: int) -> int | None:
    """Prove the final entry GP assignment is entry T9 plus constant terms.

    Under the existing MIPS PIC ABI assumption, incoming T9 names this entry.
    Require one full-width T9 term, unchanged before the final GP write; all
    other additive terms must be constants. Support split additions, but never
    reuse an earlier GP assignment after a later write. Unknown terms, missing
    registers and failed lifts leave the root unresolved. This is not a proof
    of GP's value elsewhere in the function or across calls.
    """

    if not project.arch.name.startswith("MIPS"):
        return None
    try:
        gp_offset = project.arch.registers["gp"][0]
        t9_offset = project.arch.registers["t9"][0]
        entry_vex = project.factory.block(entry_addr).vex
    except (AttributeError, KeyError):
        return None
    except Exception:
        return None

    definitions = temporary_definitions(entry_vex)
    gp_writes = [
        (index, statement.data)
        for index, statement in enumerate(entry_vex.statements)
        if isinstance(statement, pyvex.stmt.Put) and statement.offset == gp_offset
    ]
    if not gp_writes or any(
        isinstance(statement, pyvex.stmt.Put) and statement.offset == t9_offset
        for statement in entry_vex.statements[: gp_writes[-1][0]]
    ):
        return None
    terms = _vex_add_terms(gp_writes[-1][1], definitions)
    if terms is None:
        return None
    t9_key = t9_offset, project.arch.bits
    t9_reads = sum(
        _vex_get_key(term, definitions, entry_vex) == t9_key for term in terms
    )
    constants = [_vex_const_value(term, definitions) for term in terms]
    if t9_reads != 1 or any(
        value is None and _vex_get_key(term, definitions, entry_vex) != t9_key
        for term, value in zip(terms, constants, strict=True)
    ):
        return None
    displacement = sum(value for value in constants if value is not None)
    return (entry_addr + displacement) & ((1 << project.arch.bits) - 1)


def entry_base_registers(project: Project, entry_addr: int) -> dict[int, int]:
    """Return supported MIPS GP/PPC TOC roots without assuming path preservation.

    MIPS uses the entry-prologue proof; PPC64 reads the containing object's
    relocated ``.TOC.`` symbol. Return VEX offsets mapped to concrete addresses,
    not register names. The consumer decides whether its graph contains the
    entry, supplies additional roots such as MIPS T9, and handles later writes
    and call barriers. Missing evidence yields no root for that architecture.
    """

    registers: dict[int, int] = {}
    if project.arch.name.startswith("MIPS"):
        gp = mips_entry_global_pointer(project, entry_addr)
        if gp is not None:
            registers[project.arch.registers["gp"][0]] = gp
    elif project.arch.name == "PPC64":
        obj = project.loader.find_object_containing(entry_addr)
        toc = obj.get_symbol(".TOC.") if obj is not None else None
        if toc is not None:
            registers[project.arch.registers["r2"][0]] = toc.rebased_addr
    return registers


def x86_pic_thunk_reads_return_address(
    project: Project, target: int, register_offset: int
) -> bool:
    """Verify a PC thunk's 32-bit stack load and return, not just its name.

    Resolve the selected register's final VEX assignment through temporary
    definitions and require a 32-bit load from SP with a return jumpkind. This
    supports a caller-owned PC summary without trusting a thunk-like symbol
    name. Callers select the x86 ABI and expected register; this helper does
    not summarize other side effects or establish reaching definitions at the
    call. A missing assignment or failed lift supplies no evidence.
    """

    try:
        vex = project.factory.block(target).vex
    except Exception:
        return False
    assignment = _vex_last_put(vex, register_offset)
    if assignment is None or vex.jumpkind != "Ijk_Ret":
        return False
    definitions = temporary_definitions(vex)
    value = _resolve_vex_expr(assignment.data, definitions)
    return (
        isinstance(value, pyvex.expr.Load)
        and value.result_size(vex.tyenv) == 32
        and _vex_get_key(value.addr, definitions, vex) == (project.arch.sp_offset, 32)
    )
