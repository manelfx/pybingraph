"""Local ABI roots: prologue overwrites, register preservation and thunk bodies."""

from types import SimpleNamespace

import archinfo
import pytest
import pyvex

from bingraph.cfg.calls import _mips_gp_relative_slot_from_load
from bingraph.helpers.abi import (
    callee_preserved_register_offsets,
    entry_base_registers,
    mips_entry_global_pointer,
    x86_pic_thunk_reads_return_address,
)


def _mips_entry(architecture, kind):
    """Build a small lifted entry; only the selected provenance fact varies."""

    arch = archinfo.arch_from_id(architecture)
    bits = arch.bits // 2 if kind == "narrow" else arch.bits

    def constant(value):
        return pyvex.expr.Const(
            getattr(pyvex.const, f"U{bits}")(value & ((1 << bits) - 1))
        )

    gp, t9 = arch.registers["gp"][0], arch.registers["t9"][0]
    env = pyvex.IRTypeEnv(arch, [f"Ity_I{bits}"] * 3)
    operands = [pyvex.expr.RdTmp(0), constant(0x100)]
    if kind == "reversed":
        operands.reverse()
    elif kind == "duplicate":
        operands[1] = pyvex.expr.RdTmp(0)
    read = arch.registers["v0"][0] if kind == "unknown" else t9
    statements = [
        pyvex.stmt.WrTmp(0, pyvex.expr.Get(read, f"Ity_I{bits}")),
        pyvex.stmt.WrTmp(1, pyvex.expr.Binop(f"Iop_Add{bits}", operands)),
    ]
    last = 1
    if kind == "split":
        statements.append(
            pyvex.stmt.WrTmp(
                2,
                pyvex.expr.Binop(
                    f"Iop_Add{bits}", [pyvex.expr.RdTmp(1), constant(0x200)]
                ),
            )
        )
        last = 2
    elif kind == "cycle":
        statements = [
            pyvex.stmt.WrTmp(0, pyvex.expr.RdTmp(1)),
            pyvex.stmt.WrTmp(1, pyvex.expr.RdTmp(0)),
        ]
    statements.append(pyvex.stmt.Put(pyvex.expr.RdTmp(last), gp))
    if kind == "t9_before":
        statements.insert(0, pyvex.stmt.Put(constant(0x900000), t9))
    elif kind == "t9_after":
        statements.append(pyvex.stmt.Put(constant(0x900000), t9))
    elif kind == "gp_after":
        statements.append(pyvex.stmt.Put(constant(0x1234), gp))
    elif kind == "gp_before":
        statements.insert(0, pyvex.stmt.Put(constant(0x1234), gp))
    elif kind == "missing":
        statements = []
    vex = SimpleNamespace(statements=statements, tyenv=env)
    return SimpleNamespace(
        arch=arch, factory=SimpleNamespace(block=lambda _addr: SimpleNamespace(vex=vex))
    )


@pytest.mark.parametrize("architecture", ["MIPS32", "MIPS64"])
@pytest.mark.parametrize(
    "kind,displacement",
    [
        ("simple", 0x100),
        ("reversed", 0x100),
        ("split", 0x300),
        ("gp_before", 0x100),
        ("t9_after", 0x100),
        ("t9_before", None),
        ("gp_after", None),
        ("unknown", None),
        ("duplicate", None),
        ("cycle", None),
        ("missing", None),
        ("narrow", None),
    ],
)
def test_mips_root_and_decoder_slot_share_the_final_gp_proof(
    architecture, kind, displacement
):
    project = _mips_entry(architecture, kind)
    entry = 0x400000
    expected = entry + displacement if displacement is not None else None
    assert mips_entry_global_pointer(project, entry) == expected
    gp = project.arch.registers["gp"][0]
    assert entry_base_registers(project, entry) == (
        {gp: expected} if expected is not None else {}
    )

    # Exercise the decoder consumer, not just the ABI helper: a split prologue
    # can now supply a slot, while stale GP/T9 roots cannot resolve a callee.
    bits = project.arch.bits
    use = SimpleNamespace(
        statements=[
            pyvex.stmt.WrTmp(0, pyvex.expr.Get(gp, f"Ity_I{bits}")),
            pyvex.stmt.WrTmp(
                1,
                pyvex.expr.Binop(
                    f"Iop_Add{bits}",
                    [
                        pyvex.expr.RdTmp(0),
                        pyvex.expr.Const(getattr(pyvex.const, f"U{bits}")(16)),
                    ],
                ),
            ),
        ]
    )
    load = pyvex.expr.Load("Iend_LE", f"Ity_I{bits}", pyvex.expr.RdTmp(1))
    assert _mips_gp_relative_slot_from_load(
        project, SimpleNamespace(addr=entry), use, load
    ) == ((expected + 16, "Iend_LE") if expected is not None else None)


def test_mips_root_wraps_and_declines_unavailable_entry_evidence():
    project = _mips_entry("MIPS32", "simple")
    assert mips_entry_global_pointer(project, 0xFFFFFF80) == 0x80

    def unavailable(_addr):
        raise RuntimeError("lift unavailable")

    project.factory.block = unavailable
    assert mips_entry_global_pointer(project, 0x400000) is None
    project.arch = SimpleNamespace(name="MIPS64", registers={})
    assert mips_entry_global_pointer(project, 0x400000) is None
    project.arch = SimpleNamespace(name="AMD64")
    assert mips_entry_global_pointer(project, 0x400000) is None


@pytest.mark.parametrize(
    "architecture,preserved,clobbered",
    [
        ("AMD64", "rbx", "rax"),
        ("X86", "ebx", "eax"),
        ("MIPS32", "s0", "t9"),
        ("MIPS64", "s0", "t9"),
        ("PPC64", "r14", "r3"),
        ("S390X", "r6", "r2"),
        ("ARMEL", "r4", "r0"),
        ("ARMHF", "r4", "r0"),
    ],
)
def test_elf_integer_preservation_excludes_call_clobbers_and_stack_pointer(
    architecture, preserved, clobbered
):
    project = SimpleNamespace(
        arch=archinfo.arch_from_id(architecture),
        loader=SimpleNamespace(main_object=SimpleNamespace(os="UNIX - System V")),
    )
    offsets = callee_preserved_register_offsets(project)
    assert project.arch.registers[preserved][0] in offsets
    assert project.arch.registers[clobbered][0] not in offsets
    assert project.arch.sp_offset not in offsets
    project.loader.main_object.os = "unknown ABI"
    assert callee_preserved_register_offsets(project) == set()


@pytest.mark.parametrize("has_toc", [False, True])
def test_ppc_entry_base_requires_loader_toc_metadata(has_toc):
    project = SimpleNamespace(
        arch=archinfo.ArchPPC64(),
        loader=SimpleNamespace(
            find_object_containing=lambda _addr: SimpleNamespace(
                get_symbol=lambda _name: (
                    SimpleNamespace(rebased_addr=0x1000) if has_toc else None
                )
            )
        ),
    )
    assert entry_base_registers(project, 0x400000) == (
        {project.arch.registers["r2"][0]: 0x1000} if has_toc else {}
    )


@pytest.mark.parametrize("kind", ["valid", "not_return", "wrong_load_width"])
def test_x86_pc_thunk_requires_its_actual_return_address_load(kind):
    arch = archinfo.ArchX86()
    register = arch.registers["ebx"][0]
    load = pyvex.expr.Load(
        "Iend_LE",
        "Ity_I64" if kind == "wrong_load_width" else "Ity_I32",
        pyvex.expr.Get(arch.sp_offset, "Ity_I32"),
    )
    vex = SimpleNamespace(
        statements=[pyvex.stmt.Put(load, register)],
        jumpkind="Ijk_Boring" if kind == "not_return" else "Ijk_Ret",
        tyenv=pyvex.IRTypeEnv(arch, []),
    )
    project = SimpleNamespace(
        arch=arch, factory=SimpleNamespace(block=lambda _addr: SimpleNamespace(vex=vex))
    )
    assert x86_pic_thunk_reads_return_address(project, 0x400000, register) is (
        kind == "valid"
    )
