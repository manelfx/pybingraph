"""Instruction-level semantic expectations, independent of golden appearance."""

import angr
from pathlib import Path
import networkx as nx
import pydot
import pytest

from bingraph.ddg import get_ddg, render_ddg
from bingraph.core import load_project


def analyze(code, memory="always"):
    data = bytes.fromhex(code)
    project = angr.load_shellcode(data, arch="amd64", load_address=0x1000)
    return get_ddg(project, 0x1000, len(data), memory)


def kinds(ddg):
    return {(edge.source, edge.target, edge.kind) for edge in ddg.dependencies}


def test_all_dependency_kinds_and_no_self_edges():
    # mov eax,1; mov ebx,eax; mov eax,2; mov ecx,eax
    ddg = analyze("b80100000089c3b80200000089c1")
    assert {(0, 1, "RAW"), (1, 2, "WAR"), (0, 2, "WAW"), (2, 3, "RAW")} <= kinds(ddg)
    assert (0, 3, "RAW") not in kinds(ddg)
    assert all(edge.source < edge.target for edge in ddg.dependencies)
    graph = nx.DiGraph((edge.source, edge.target) for edge in ddg.dependencies)
    assert nx.is_directed_acyclic_graph(graph)


def test_aliases_and_zeroing_idiom():
    # mov al,1; mov ah,2; mov bl,al: ah must not produce bl.
    assert (1, 2, "RAW") not in kinds(analyze("b001b40288c3"))
    # mov eax,1; xor eax,eax; mov ebx,eax: xor does not consume old eax.
    edges = kinds(analyze("b80100000031c089c3"))
    assert (0, 1, "RAW") not in edges
    assert (1, 2, "RAW") in edges
    # Equal current values in *different* registers are not a zeroing idiom.
    edges = kinds(analyze("b801000000bb0100000031d8"))
    assert {(0, 2, "RAW"), (1, 2, "RAW")} <= edges


def test_flags_cross_instruction_temporaries():
    assert (0, 1, "RAW") in kinds(analyze("39d80f94c1"))  # cmp eax,ebx; sete cl


@pytest.mark.parametrize("budget", [0, 64])
@pytest.mark.parametrize(
    "code,source", [("d3ff7f1153f57f11", "ft11"), ("d3fd8d09d3fd8d09", "fs11")]
)
def test_riscv_float_ites_and_typed_unknowns_keep_operand_dependencies(
    monkeypatch, budget, code, source
):
    from bingraph.ddg import effects

    # FMUL.S FT11,FT11,FS7; FMUL.S FA0,FT11,FS7, or two FSUB.S FS11,FS11,FS8.
    # VEX's NaN-boxing checks select between register bits and an FP32 NaN.
    monkeypatch.setattr(effects, "MAX_VALUE_NODES", budget)
    data = bytes.fromhex(code)
    project = angr.load_shellcode(data, arch="riscv64", load_address=0x1000)
    graph = get_ddg(project, 0x1000, len(data))
    assert any(
        d.source == 0
        and d.target == 1
        and d.kind == "RAW"
        and any(r.startswith(source) for r in d.resources)
        for d in graph.dependencies
    )


@pytest.mark.parametrize(
    "width,bits,arithmetic",
    [
        (32, 0x3FC00000, True),
        (64, 0x3FF8000000000000, True),
        (32, 0x80000000, False),  # Signed zero and NaN payloads must survive moves.
        (64, 0x8000000000000000, False),
        (32, 0x7FC12345, False),
        (64, 0x7FF8123456789ABC, False),
        (32, 0x3FC00000, None),  # Unknown rounding must not invent an exact result.
        (64, 0x3FF8000000000000, None),
    ],
)
def test_float_register_bitcasts_and_ites_preserve_exact_values(
    monkeypatch, width, bits, arithmetic
):
    from types import SimpleNamespace
    import pyvex
    from bingraph.ddg.effects import instruction_effects

    # Keep the original GPR storage as IEEE bits for 1.5, interpret a VEX FP
    # register read, merge it with 1.5, then add 0.5. The result is the bits of
    # 2.0, not the integer 2, even when used as an address after reinterpreting.
    data = (
        bytes.fromhex("48b8") + bits.to_bytes(8, "little") + bytes.fromhex("4889c7891f")
    )
    project = angr.load_shellcode(data, arch="amd64", load_address=0x1000)
    vex = project.factory.block(
        0x1000, size=len(data), opt_level=0, cross_insn_opt=False, strict_block_end=True
    ).vex
    result = pyvex.expr.Get(project.arch.registers["rax"][0], f"Ity_F{width}")
    if arithmetic is not False:
        constant = getattr(pyvex.const, f"F{width}")
        guard = pyvex.expr.Binop(
            "Iop_CmpNE64",
            [
                pyvex.expr.Get(project.arch.registers["r8"][0], "Ity_I64"),
                pyvex.expr.Const(pyvex.const.U64(0)),
            ],
        )
        merged = pyvex.expr.ITE(guard, result, pyvex.expr.Const(constant(1.5)))
        result = pyvex.expr.Triop(
            f"Iop_AddF{width}",
            [
                pyvex.expr.Const(pyvex.const.U32(0))
                if arithmetic is True
                else pyvex.expr.Get(project.arch.registers["r9"][0], "Ity_I32"),
                merged,
                pyvex.expr.Const(constant(0.5)),
            ],
        )
    result = pyvex.expr.Unop(f"Iop_ReinterpF{width}asI{width}", [result])
    if width == 32:
        result = pyvex.expr.Unop("Iop_32Uto64", [result])
    target = next(
        s
        for s in vex.statements
        if isinstance(s, pyvex.stmt.Put)
        and s.offset == project.arch.registers["rdi"][0]
    )
    target.data = result
    monkeypatch.setattr(
        project.factory, "block", lambda *a, **k: SimpleNamespace(vex=vex)
    )
    graph = get_ddg(project, 0x1000, len(data))
    effect = instruction_effects(project, graph.instructions, len(data))[-1]
    address = next(r.address for r in effect.writes if r.kind == "memory")
    if arithmetic is None:
        assert not address.concrete
    else:
        expected = (
            (0x40000000 if width == 32 else 0x4000000000000000) if arithmetic else bits
        )
        assert address.concrete_value == expected
    assert {(0, 1, "RAW"), (1, 2, "RAW")} <= kinds(graph)


@pytest.mark.parametrize("code", ["ed0000300012a774035a", "b3430000"])
def test_unavailable_s390_float128_values_keep_architectural_reads(code):
    from bingraph.ddg.effects import instruction_effects

    # TCXB F0,0x30; JNE, or LCXBR F0,F0. angr's F128HItoF64 evaluator raises
    # IndexError: retain the F0/F2 operands rather than pretending to compute it.
    data = bytes.fromhex(code)
    project = angr.load_shellcode(data, arch="s390x", load_address=0x1000)
    graph = get_ddg(project, 0x1000, len(data))
    effect = instruction_effects(project, graph.instructions, len(data))[0]
    for name in ("f0", "f2"):
        offset, size = project.arch.registers[name]
        assert any(
            r.kind == "register"
            and r.offset <= offset
            and offset + size <= r.offset + r.size
            for r in effect.reads
        )
    if len(graph.instructions) == 2:
        assert (0, 1, "RAW") in kinds(graph)


@pytest.mark.parametrize(
    "architecture,offset,width,expected",
    [
        ("amd64", 16, 4, "eax"),
        ("amd64", 17, 1, "ah"),
        ("x86", 168, 8, "xmm0[127:64]"),
        ("s390x", 583, 1, "r0[7:0]"),
        ("s390x", 590, 1, "r1[15:8]"),
        ("s390x", 736, 8, "cc_op"),
        ("s390x", 744, 8, "cc_dep1"),
        ("riscv64", 284, 4, "ft0[63:32]"),
        ("riscv64", 536, 4, "guest[0x218:4]"),
        # A range crossing the end of xmm7 must not be called a slice of xmm7.
        ("x86", 284, 8, "guest[0x11c:8]"),
    ],
)
def test_numeric_register_labels_use_metadata_and_endian_slices(
    architecture, offset, width, expected
):
    from archinfo import arch_from_id
    from bingraph.ddg.effects import _register_label

    assert _register_label(arch_from_id(architecture), offset, width) == expected


def test_register_naming_does_not_change_effects_or_dependency_topology(monkeypatch):
    from bingraph.ddg import effects
    from bingraph.helpers.disassembly import decode_raw_capstone_insns

    # movdqa xmm0,xmm1; pmovmskb eax,xmm0 reads the unnamed upper 64-bit slice.
    code = bytes.fromhex("660f6fc1660fd7c0")
    project = angr.load_shellcode(code, arch="x86", load_address=0x1000)
    insns = decode_raw_capstone_insns(project, 0x1000, len(code))
    named = effects.instruction_effects(project, insns, len(code))
    graph = get_ddg(project, 0x1000, len(code))
    monkeypatch.setattr(
        effects,
        "_register_label",
        lambda arch, offset, width: arch.translate_register_name(offset, width),
    )
    original = effects.instruction_effects(project, insns, len(code))
    # A different cache key gets a fresh graph without clearing shared caches.
    before = get_ddg(project, 0x1000, len(code), "may")
    assert kinds(graph) == kinds(before)
    assert any("xmm0[127:64]" in edge.resources for edge in graph.dependencies)

    def identity(resource):
        return resource.kind, resource.offset, resource.size, resource.address

    for left, right in zip(named, original, strict=True):
        for attribute in ("reads", "writes"):
            assert list(map(identity, getattr(left, attribute))) == list(
                map(identity, getattr(right, attribute))
            )


@pytest.mark.parametrize(
    "opcode,width,reads",
    [("d2", 8, 1), ("d2", 256, 1), ("d7", 80, 0), ("d7", 81, 0), ("d6", 2, 2)],
)
def test_instruction_loops_cover_complete_memory_ranges(opcode, width, reads):
    from bingraph.ddg.effects import instruction_effects
    from bingraph.helpers.disassembly import decode_raw_capstone_insns
    from pyvex.vex_ffi import guest_offsets

    # MVC copies, XC self-zeros, and OC reads/modifies memory. VEX implements
    # them with different strides and even two subfields of a restart counter.
    source = "f0f8" if opcode == "d7" else "1028"
    code = bytes.fromhex(f"{opcode}{width - 1:02x}f0f8{source}")
    project = angr.load_shellcode(code, arch="s390x", load_address=0x1000)
    insns = decode_raw_capstone_insns(project, 0x1000, len(code))
    effect = instruction_effects(project, insns, len(code))[0]
    assert [r.size for r in effect.reads if r.kind == "memory"] == [width] * reads
    assert [r.size for r in effect.writes if r.kind == "memory"] == [width]
    counter = guest_offsets["s390x", "counter"]
    assert not any(
        r.kind == "register" and counter <= r.offset < counter + 8
        for r in effect.reads + effect.writes
    )


@pytest.mark.parametrize(
    "code,expected",
    [
        # Store to the final source byte, MVC eight bytes, then read final dest byte.
        ("4220102fd207f0f810284330f0ff", {(0, 1, "RAW"), (1, 2, "RAW")}),
        # A fully overwritten store must not remain a producer of the last byte.
        ("e320f0f80024d207f0f810284330f0ff", {(0, 1, "WAW"), (1, 2, "RAW")}),
        # XC clears 80 bytes, not just the first 8-byte VEX store.
        ("d74ff0a8f0a84320f0f7", {(0, 1, "RAW")}),
    ],
)
def test_instruction_loops_recover_last_byte_dependencies(code, expected):
    data = bytes.fromhex(code)
    project = angr.load_shellcode(data, arch="s390x", load_address=0x1000)
    assert kinds(get_ddg(project, 0x1000, len(data))) == expected


@pytest.mark.parametrize("failure", ["budget", "symbolic", "multiple exits"])
def test_instruction_loops_fail_instead_of_returning_partial_effects(
    monkeypatch, failure
):
    import pyvex
    from types import SimpleNamespace
    from bingraph.ddg import effects

    code = bytes.fromhex("d207f0f81028")
    project = angr.load_shellcode(code, arch="s390x", load_address=0x1000)
    if failure == "budget":
        monkeypatch.setattr(effects, "MAX_INSTRUCTION_ITERATIONS", 1)
    else:
        vex = project.factory.block(
            0x1000,
            size=len(code),
            opt_level=0,
            cross_insn_opt=False,
            strict_block_end=True,
        ).vex
        loop = next(s for s in vex.statements if isinstance(s, pyvex.stmt.Exit))
        if failure == "symbolic":
            loop.guard = pyvex.expr.Binop(
                "Iop_CmpNE64",
                [
                    pyvex.expr.Get(project.arch.registers["r2"][0], "Ity_I64"),
                    pyvex.expr.Const(pyvex.const.U64(0)),
                ],
            )
        else:
            vex.statements.insert(
                1,
                pyvex.stmt.Exit(
                    pyvex.expr.Const(pyvex.const.U1(0)),
                    pyvex.const.U64(0x1006),
                    "Ijk_Boring",
                    project.arch.ip_offset,
                ),
            )
        monkeypatch.setattr(
            project.factory, "block", lambda *a, **k: SimpleNamespace(vex=vex)
        )
    with pytest.raises(ValueError, match="instruction-internal loop"):
        get_ddg(project, 0x1000, len(code))


def test_architectural_self_branch_is_not_replayed():
    assert len(analyze("90ebfe").instructions) == 2  # nop; jmp to itself


@pytest.mark.parametrize(
    "arch,direction,opcode,width,displacement,accesses",
    [
        ("amd64", "fc", "f3a6", 10, "", 2),
        ("amd64", "fd", "f3a6", 10, "-0x9", 2),
        ("amd64", "", "f3a6", 19, "-0x9", 2),
        ("amd64", "fc", "66f3a7", 20, "", 2),
        ("amd64", "fc", "f3a7", 40, "", 2),
        ("amd64", "fc", "f348a7", 80, "", 2),
        ("amd64", "fc", "f2ae", 10, "", 1),
        ("x86", "fc", "f2ae", 10, "", 1),
    ],
)
def test_repeated_comparisons_use_maximum_read_footprints(
    arch, direction, opcode, width, displacement, accesses
):
    from bingraph.ddg.effects import instruction_effects

    # MOV ECX,10; REP CMPS/SCAS; NOP. The following instruction must survive
    # VEX's split at REP without interpreting a data-dependent comparison loop.
    data = bytes.fromhex(direction + "b90a000000" + opcode + "90")
    project = angr.load_shellcode(data, arch=arch, load_address=0x1000)
    graph = get_ddg(project, 0x1000, len(data))
    effects = instruction_effects(project, graph.instructions, len(data))
    reads = [r for r in effects[-2].reads if r.kind == "memory"]
    assert len(reads) == accesses
    assert all(r.size == width and r.maximum_footprint for r in reads)
    if arch == "amd64":
        assert any(f"entry_rdi{displacement}]" in r.label for r in reads)
    assert graph.instructions[-1].mnemonic == "nop"
    assert not any(r.kind == "memory" for e in effects for r in e.writes)


@pytest.mark.parametrize("policy", ["always", "may", "must"])
def test_repeat_footprints_reuse_memory_policies_and_compact_labels(policy):
    # CLD; MOV ECX,10; write [rdi+9], [rdi+10], [rsi]; REPNE SCASB.
    graph = analyze("fcb90a000000c6470901c6470a01c60601f2ae", policy)
    edges = kinds(graph)
    assert (2, 5, "RAW") in edges  # Inside the maximum, not only the first byte.
    assert ((3, 5, "RAW") in edges) == (policy == "always")
    assert ((4, 5, "RAW") in edges) == (policy != "must")
    # Maximum is a tooltip qualifier, not an oversized visible edge caption.
    from bingraph.ddg.render import DependencyEdges

    resource = next(
        d.resources[0]
        for d in graph.dependencies
        if d.source == 2 and d.target == 5 and d.kind == "RAW"
    )
    assert "maximum footprint" in resource
    assert DependencyEdges._resource_label(resource) == "[rdi]"
    # Preserve the qualifier for a later store's WAR edge too, even though
    # its own destination is an ordinary exact-width memory resource.
    later_store = analyze("fcb90a0000004889fbf2aec6430901", policy)
    assert any(
        d.source == 3
        and d.target == 4
        and d.kind == "WAR"
        and "maximum footprint" in d.resources[0]
        for d in later_store.dependencies
    )


def test_repeat_outputs_are_unknown_but_zero_count_preserves_state():
    from bingraph.ddg.effects import instruction_effects

    for prefix, repeats in [("fcb90a000000", True), ("31c9", False)]:
        # Repeat; SETA AL; load [rdi+rcx]. An early stop must not leave an
        # exact one-iteration pointer/count, nor claim ten iterations completed.
        data = bytes.fromhex(prefix + "f3a60f97c0488b140f")
        project = angr.load_shellcode(data, arch="amd64", load_address=0x1000)
        graph = get_ddg(project, 0x1000, len(data))
        effects = instruction_effects(project, graph.instructions, len(data))
        repeat = next(e for e in effects if e.insn.mnemonic == "repe cmpsb")
        last_read = next(r for r in effects[-1].reads if r.kind == "memory")
        if repeats:
            assert all(
                name.startswith("ddg_unknown_") for name in last_read.address.variables
            )
        else:
            assert not repeat.writes
            assert [r.label for r in repeat.reads] == ["rcx"]
            assert last_read.label == "mem[entry_rdi]:8"
            assert (0, 2, "RAW") in kinds(graph)  # XOR, not empty REP, produces flags.
            assert (1, 2, "RAW") not in kinds(graph)


def test_repeat_address_override_and_unsupported_symbolic_count():
    from bingraph.ddg.effects import instruction_effects

    # RCX has nonzero high bits, but address-size override uses ECX=2. A
    # possible 32-bit address wrap cannot be represented as one 64-bit interval.
    data = bytes.fromhex("fc48b9020000000100000067f3a6")
    project = angr.load_shellcode(data, arch="amd64", load_address=0x1000)
    graph = get_ddg(project, 0x1000, len(data))
    effects = instruction_effects(project, graph.instructions, len(data))
    reads = [r for r in effects[-1].reads if r.kind == "memory"]
    # Both opaque footprints coalesce as "unknown memory", not equal pointers.
    assert len(reads) == 1
    assert all(r.address is None and r.size == 2 for r in reads)
    with pytest.raises(ValueError, match="symbolic repeat count"):
        analyze("f3a6")
    assert analyze("f20fc2c100").instructions[0].mnemonic == "cmpeqsd"


def test_large_repeat_count_does_not_require_iteration():
    from bingraph.ddg.effects import instruction_effects

    # CLD; MOV RCX,-1; REPNE SCASB is a common string-length idiom.
    data = bytes.fromhex("fc48b9fffffffffffffffff2ae")
    project = angr.load_shellcode(data, arch="amd64", load_address=0x1000)
    graph = get_ddg(project, 0x1000, len(data))
    effects = instruction_effects(project, graph.instructions, len(data))
    read = next(r for r in effects[-1].reads if r.kind == "memory")
    assert read.maximum_footprint and read.size == (1 << 64) - 1


@pytest.mark.parametrize("adc,producer", [("0000a0e0", 0), ("0000b0e0", 1)])
def test_arm_conditions_use_the_actual_flag_writer(adc, producer):
    # cmp r3,r1,lsl #31; adc/adcs r0,r0,r0; subhs r3,r3,r1,lsl #31.
    # ADC reads carry but does not update flags; ADCS does. Capstone's
    # update_flags field is unreliable here, so rely on the VEX effects.
    project = angr.load_shellcode(
        bytes.fromhex("810f53e1" + adc + "813f4320"), arch="armel", load_address=0x1000
    )
    graph = get_ddg(project, 0x1000, 12)
    assert (0, 1, "RAW") in kinds(graph)
    assert (producer, 2, "RAW") in kinds(graph)
    assert (1 - producer, 2, "RAW") not in kinds(graph)
    assert not any("lifted value" in d.resources for d in graph.dependencies)
    dot = render_ddg(project, 0x1000, 12, format="raw")
    edges = pydot.graph_from_dot_data(dot)[0].get_edges()
    edge = next(
        e
        for e in edges
        if e.get_source().strip('"') == hex(graph.instructions[producer].address)
        and e.get_destination().strip('"') == hex(graph.instructions[2].address)
    )
    assert edge.get("label").strip('"') == "flags"


def test_value_evaluation_budget_does_not_drop_dependencies(monkeypatch):
    from bingraph.ddg import effects

    monkeypatch.setattr(effects, "MAX_VALUE_NODES", 4)
    graph = analyze("b801000000" + "01c0" * 100)  # mov eax,1; repeated add eax,eax
    assert len(graph.instructions) == 101
    assert all((i, i + 1, "RAW") in kinds(graph) for i in range(100))


@pytest.mark.parametrize("call", ["0320f809", "04110001"])
@pytest.mark.parametrize("padding", [0, 98])
def test_mips_delay_slot_effects_belong_to_the_slot(call, padding):
    # li t0,1; jalr t9 / bal next; sw t0,16(sp). VEX fuses the last two
    # IMarks; Capstone omits the generic call group for BAL.
    project = angr.load_shellcode(
        bytes.fromhex("24080001" + "00000000" * padding + call + "afa80010"),
        arch="mips",
        load_address=0x1000,
    )
    ddg = get_ddg(project, 0x1000, 12 + 4 * padding, "may")
    assert len(ddg.instructions) == 3 + padding
    assert ddg.instructions[-1].address == 0x1008 + 4 * padding
    assert (0, 2 + padding, "RAW") in kinds(ddg)
    assert (0, 1 + padding, "RAW") not in kinds(ddg)


def test_widened_writes_partial_overlap_and_vectors():
    # eax writes clear the whole rax; the previous high bytes are not live.
    edges = kinds(analyze("48b80100000000000000b8020000004889c3"))
    assert (0, 2, "RAW") not in edges
    assert (1, 2, "RAW") in edges
    # Both byte writes produce the later ax read.
    edges = kinds(analyze("b001b4026689c3"))
    assert {(0, 2, "RAW"), (1, 2, "RAW")} <= edges
    # movdqa xmm0,xmm1; paddd xmm0,xmm2; movdqa xmm3,xmm0
    edges = kinds(analyze("660f6fc1660ffec2660f6fd8"))
    assert {(0, 1, "RAW"), (1, 2, "RAW")} <= edges
    assert (0, 1, "RAW") not in kinds(analyze("660f6fc1660fefc0"))  # pxor xmm0,xmm0


def test_unknown_stores_do_not_kill_earlier_possible_memory_producers():
    # store [rdi]; store [rsi]; load [rdi]: rsi may or may not alias rdi.
    edges = kinds(analyze("8907891e8b0f", "may"))
    assert {(0, 2, "RAW"), (1, 2, "RAW")} <= edges
    # Overlapping ranges are not treated as disjoint stack slots.
    assert (0, 1, "RAW") in kinds(analyze("48894424088b5c240c", "may"))
    # A dead loaded value does not make its memory access disappear.
    edges = kinds(analyze("488b0731c0891f", "may"))
    assert (0, 2, "WAR") in edges
    assert (0, 1, "RAW") not in edges


def test_unsupported_semantics_and_multiple_blocks_fail_explicitly():
    with pytest.raises(ValueError, match="Unsupported DDG"):
        analyze("0fa2")  # CPUID is an opaque VEX helper, not a no-op.
    with pytest.raises(ValueError, match="one fully liftable"):
        analyze("eb0090")  # jmp followed by bytes from a different block
    with pytest.raises(ValueError, match="one fully liftable"):
        analyze("740090")  # conditional branch also falls through to the next byte


@pytest.mark.parametrize(
    "arch,code,address",
    [
        ("amd64", "0f0b", 0x1000),
        ("x86", "0f0b", 0x1000),
        ("armel", "f201f0e7", 0x1000),
        ("armel", "00de", 0x1001),
        ("armhf", "f0f700a0", 0x1001),
        ("ARMCortexM", "00de", 0x1001),
    ],
)
def test_explicit_terminal_traps_remain_visible_without_fake_dependencies(
    arch, code, address
):
    from bingraph.ddg.effects import instruction_effects

    data = bytes.fromhex(code)
    project = angr.load_shellcode(data, arch=arch, load_address=address & ~1)
    graph = get_ddg(project, address, len(data))
    assert len(graph.instructions) == 1 and not graph.dependencies
    effect = instruction_effects(project, graph.instructions, len(data))[0]
    assert not effect.reads and not effect.writes
    assert "ud" in render_ddg(project, address, len(data), format="raw")


@pytest.mark.parametrize(
    "arch,prefix,trap,address",
    [
        ("amd64", "89f889c3", "0f0b", 0x1000),  # MOV EAX,EDI; MOV EBX,EAX.
        ("armel", "0100a0e10020a0e1", "f201f0e7", 0x1000),  # MOV R0,R1; MOV R2,R0.
        ("armel", "08460246", "00de", 0x1001),  # Thumb MOV R0,R1; MOV R2,R0.
    ],
)
def test_terminal_traps_preserve_the_preceding_instruction_effects(
    arch, prefix, trap, address
):
    from bingraph.ddg.effects import instruction_effects

    data = bytes.fromhex(prefix + trap)
    project = angr.load_shellcode(data, arch=arch, load_address=address & ~1)
    graph = get_ddg(project, address, len(data))
    before = get_ddg(project, address, len(bytes.fromhex(prefix)))
    assert (0, 1, "RAW") in kinds(graph)
    assert [d for d in graph.dependencies if d.target < 2] == list(before.dependencies)
    actual = instruction_effects(project, graph.instructions, len(data))
    expected = instruction_effects(
        project, before.instructions, len(bytes.fromhex(prefix))
    )
    # VEX may renumber temporaries when the IRSB grows. Architectural effects
    # and the actual prefix dependencies above must remain unchanged.
    for left, right in zip(actual[:2], expected, strict=True):
        for attribute in ("reads", "writes"):
            assert {r for r in getattr(left, attribute) if r.kind != "temporary"} == {
                r for r in getattr(right, attribute) if r.kind != "temporary"
            }


def test_thumb_trap_retains_any_it_state_effects_emitted_by_vex():
    from bingraph.ddg.effects import instruction_effects

    # VEX emits a zero-length UDF marker after IT EQ, including IT-state
    # accesses. Accept the trap without dropping that partially lifted IR.
    data = bytes.fromhex("08bf00de")
    project = angr.load_shellcode(data, arch="armel", load_address=0x1000)
    graph = get_ddg(project, 0x1001, len(data))
    effect = instruction_effects(project, graph.instructions, len(data))[1]
    assert any(r.label == "itstate" for r in effect.reads)
    assert (0, 1, "RAW") in kinds(graph)


@pytest.mark.parametrize(
    "arch,code,address",
    [
        ("amd64", "0f0b90", 0x1000),  # No instructions may follow UD2.
        (
            "amd64",
            "0f0b0f0b",
            0x1000,
        ),  # An earlier trap cannot hide behind a later one.
        ("amd64", "eb000f0b", 0x1000),  # A jump still ends the preceding block.
        ("amd64", "0fffc00f0b", 0x1000),  # An earlier UD0 is not the accepted UD2.
        ("armel", "f201f0e70000a0e1", 0x1000),
        ("armel", "00de00bf", 0x1001),
    ],
)
def test_trap_exception_does_not_hide_earlier_failures_or_following_code(
    arch, code, address
):
    data = bytes.fromhex(code)
    project = angr.load_shellcode(data, arch=arch, load_address=address & ~1)
    with pytest.raises(ValueError, match="DDG"):
        get_ddg(project, address, len(data))


@pytest.mark.parametrize("failure", ["fault PC", "extent", "emulation failure"])
def test_terminal_trap_requires_matching_vex_stop_metadata(monkeypatch, failure):
    from types import SimpleNamespace
    import pyvex

    data = bytes.fromhex("89f80f0b")
    project = angr.load_shellcode(data, arch="amd64", load_address=0x1000)
    vex = SimpleNamespace(
        size=3 if failure == "extent" else len(data),
        next=pyvex.expr.Const(
            pyvex.const.U64(0x1000 if failure == "fault PC" else 0x1002)
        ),
        jumpkind="Ijk_EmFail" if failure == "emulation failure" else "Ijk_NoDecode",
    )
    monkeypatch.setattr(
        project.factory, "block", lambda *a, **k: SimpleNamespace(vex=vex)
    )
    with pytest.raises(ValueError, match="one fully liftable"):
        get_ddg(project, 0x1000, len(data))


@pytest.mark.parametrize(
    "code,width,access",
    [
        ("0878", 1, "reads"),
        ("0888", 2, "reads"),
        ("0870", 1, "writes"),
        ("0880", 2, "writes"),
        ("0860", 4, "writes"),
    ],
)
def test_constant_guarded_thumb_memory_keeps_its_access_width(code, width, access):
    from bingraph.ddg.effects import instruction_effects

    # IT EQ; NOP exhausts an explicit IT state before a load/store r0,[r1].
    # A raw Thumb entry alone does not prove its incoming IT state is zero.
    project = angr.load_shellcode(
        bytes.fromhex("08bf00bf" + code), arch="armel", load_address=0x1000
    )
    graph = get_ddg(project, 0x1001, 6)
    effect = instruction_effects(project, graph.instructions, 6)[-1]
    assert [r.size for r in getattr(effect, access) if r.kind == "memory"] == [width]


@pytest.mark.parametrize("guard", [True, False, None])
@pytest.mark.parametrize("policy", ["always", "may", "must"])
def test_guarded_stores_require_proof_before_replacing_producers(
    monkeypatch, guard, policy
):
    from types import SimpleNamespace
    import pyvex
    from bingraph.ddg.effects import instruction_effects

    # Store EAX, guarded store EBX, load, definite store ESI, load. A possible
    # write keeps both producers; the later definite write replaces both.
    data = bytes.fromhex("8907891f8b0f89378b17")
    project = angr.load_shellcode(data, arch="amd64", load_address=0x1000)
    vex = project.factory.block(
        0x1000, size=len(data), opt_level=0, cross_insn_opt=False, strict_block_end=True
    ).vex
    stores = [
        i for i, s in enumerate(vex.statements) if isinstance(s, pyvex.stmt.Store)
    ]
    store = vex.statements[stores[1]]
    condition = (
        pyvex.expr.Const(pyvex.const.U1(int(guard)))
        if guard is not None
        else pyvex.expr.Binop(
            "Iop_CmpNE64",
            [
                pyvex.expr.Get(project.arch.registers["rsi"][0], "Ity_I64"),
                pyvex.expr.Const(pyvex.const.U64(0)),
            ],
        )
    )
    vex.statements[stores[1]] = pyvex.stmt.StoreG(
        store.end, store.addr, store.data, condition
    )
    monkeypatch.setattr(
        project.factory, "block", lambda *a, **k: SimpleNamespace(vex=vex)
    )
    graph = get_ddg(project, 0x1000, len(data), policy)
    assert ((0, 2, "RAW") in kinds(graph)) == (not guard)
    assert ((1, 2, "RAW") in kinds(graph)) == (guard is not False)
    assert ((0, 1, "WAW") in kinds(graph)) == (guard is not False)
    assert (3, 4, "RAW") in kinds(graph)
    assert not {(0, 4, "RAW"), (1, 4, "RAW")} & kinds(graph)
    effect = instruction_effects(project, graph.instructions, len(data))[1]
    if guard is False:
        assert not effect.reads
        assert all(r.kind == "temporary" for r in effect.writes)
    else:
        write = next(r for r in effect.writes if r.kind == "memory")
        assert write.conditional == (guard is None)
        assert write.size == 4
        if guard is None:
            assert any(r.label == "rsi" for r in effect.reads)
            assert any(
                "(conditional)" in r
                for d in graph.dependencies
                if d.source == 1 and d.target == 2 and d.kind == "RAW"
                for r in d.resources
            )


@pytest.mark.parametrize("guard", [True, False, None])
def test_guarded_loads_keep_the_fallback_and_skip_inactive_addresses(
    monkeypatch, guard
):
    from types import SimpleNamespace
    import pyvex
    from bingraph.ddg.effects import instruction_effects

    # MOV ECX,1; guarded load ECX,[RDI] (otherwise keep ECX); MOV EDX,ECX.
    data = bytes.fromhex("b9010000008b0f89ca")
    project = angr.load_shellcode(data, arch="amd64", load_address=0x1000)
    vex = project.factory.block(
        0x1000, size=len(data), opt_level=0, cross_insn_opt=False, strict_block_end=True
    ).vex
    index = next(
        n
        for n, s in enumerate(vex.statements)
        if isinstance(s, pyvex.stmt.WrTmp) and isinstance(s.data, pyvex.expr.Load)
    )
    statement = vex.statements[index]
    condition = (
        pyvex.expr.Const(pyvex.const.U1(int(guard)))
        if guard is not None
        else pyvex.expr.Binop(
            "Iop_CmpNE64",
            [
                pyvex.expr.Get(project.arch.registers["rsi"][0], "Ity_I64"),
                pyvex.expr.Const(pyvex.const.U64(0)),
            ],
        )
    )
    vex.statements[index] = pyvex.stmt.LoadG(
        statement.data.end,
        "ILGop_Ident32",
        statement.tmp,
        statement.data.addr,
        pyvex.expr.Get(project.arch.registers["ecx"][0], "Ity_I32"),
        condition,
    )
    monkeypatch.setattr(
        project.factory, "block", lambda *a, **k: SimpleNamespace(vex=vex)
    )
    graph = get_ddg(project, 0x1000, len(data))
    assert ((0, 1, "RAW") in kinds(graph)) == (guard is not True)
    assert (1, 2, "RAW") in kinds(graph)
    effect = instruction_effects(project, graph.instructions, len(data))[1]
    memory = [r for r in effect.reads if r.kind == "memory"]
    assert len(memory) == int(guard is not False)
    if memory:
        assert memory[0].conditional == (guard is None)
        assert memory[0].size == 4
    assert any(r.label == "rdi" for r in effect.reads) == (guard is not False)
    assert any(r.label == "rsi" for r in effect.reads) == (guard is None)


@pytest.mark.parametrize("policy", ["always", "may", "must"])
def test_arm_predicated_load_keeps_guard_fallback_and_memory_dependencies(policy):
    from bingraph.ddg.effects import instruction_effects
    from bingraph.ddg.render import DependencyEdges

    # MOV R0,1; MOV R1,R2; CMP R3,0; STR R5,[R1]; LDRNE R0,[R1]; ADD R4,R0,1.
    data = bytes.fromhex("0100a0e30210a0e1000053e3005081e500009115014080e2")
    project = angr.load_shellcode(data, arch="armel", load_address=0x1000)
    graph = get_ddg(project, 0x1000, len(data), policy)
    assert {
        (0, 4, "RAW"),
        (1, 4, "RAW"),
        (2, 4, "RAW"),
        (3, 4, "RAW"),
        (4, 5, "RAW"),
    } <= kinds(graph)
    effect = instruction_effects(project, graph.instructions, len(data))[4]
    assert next(r for r in effect.reads if r.kind == "memory").conditional
    label = next(
        r
        for d in graph.dependencies
        if d.source == 3 and d.target == 4 and d.kind == "RAW"
        for r in d.resources
        if r.startswith("mem")
    )
    assert "(conditional)" in label
    assert DependencyEdges._resource_label(label) == "[r2]"


@pytest.mark.parametrize("policy", ["always", "may", "must"])
@pytest.mark.parametrize(
    "store,unknown_alias", [("00208115", True), ("04208015", False)]
)
def test_conditional_stores_use_the_existing_alias_policy(policy, store, unknown_alias):
    # STR R4,[R0]; STRNE R2,[R1] (unknown alias) or [R0+4] (disjoint); LDR R3,[R0].
    data = bytes.fromhex("004080e5" + store + "003090e5")
    project = angr.load_shellcode(data, arch="armel", load_address=0x1000)
    graph = get_ddg(project, 0x1000, len(data), policy)
    assert (0, 2, "RAW") in kinds(graph)
    expected = policy == "always" or (unknown_alias and policy == "may")
    assert ((1, 2, "RAW") in kinds(graph)) == expected


@pytest.mark.parametrize(
    "arch,code,width,inputs,outputs",
    [
        ("amd64", "f00fb01f", 1, {"rdi", "al", "bl"}, {"al"}),
        ("amd64", "66f00fb11f", 2, {"rdi", "ax", "bx"}, {"ax"}),
        ("amd64", "f00fb11f", 4, {"rdi", "eax", "ebx"}, {"rax"}),
        ("amd64", "f0480fb11f", 8, {"rdi", "rax", "rbx"}, {"rax"}),
        (
            "amd64",
            "f00fc70f",
            8,
            {"rdi", "eax", "edx", "ebx", "ecx"},
            {"rax", "rdx"},
        ),
        (
            "amd64",
            "f0480fc70f",
            16,
            {"rdi", "rax", "rdx", "rbx", "rcx"},
            {"rax", "rdx"},
        ),
        ("s390x", "ba123000", 4, {"r3", "r1_32", "r2_32"}, {"r1_32"}),
        ("s390x", "eb1230000030", 8, {"r3", "r1", "r2"}, {"r1"}),
        (
            "s390x",
            "bb024000",
            8,
            {"r4", "r0[31:0]", "r1_32", "r2_32", "r3_32"},
            {"r0[31:0]", "r1_32"},
        ),
        (
            "s390x",
            "eb024000003e",
            16,
            {"r4", "r0", "r1", "r2", "r3"},
            {"r0", "r1"},
        ),
    ],
)
def test_cas_uses_vex_widths_operands_and_outputs(arch, code, width, inputs, outputs):
    from bingraph.ddg.effects import instruction_effects

    # Native CMPXCHG/CS families cover single/double CAS in both byte orders.
    # VEX, not mnemonic-specific handling, maps old-value words to registers.
    data = bytes.fromhex(code)
    project = angr.load_shellcode(data, arch=arch, load_address=0x1000)
    graph = get_ddg(project, 0x1000, len(data))
    effect = instruction_effects(project, graph.instructions, len(data))[0]
    reads = [r for r in effect.reads if r.kind == "memory"]
    writes = [r for r in effect.writes if r.kind == "memory"]
    assert len(reads) == len(writes) == 1
    assert reads[0].size == writes[0].size == width
    assert reads[0].address.identical(writes[0].address)
    assert not reads[0].conditional and writes[0].conditional
    assert inputs <= {r.label for r in effect.reads if r.kind == "register"}
    assert outputs <= {r.label for r in effect.writes if r.kind == "register"}


@pytest.mark.parametrize("policy", ["always", "may", "must"])
def test_cas_preserves_possible_memory_producers_and_register_dependencies(policy):
    # Set address/expected/replacement registers; store; CAS; load; SETE;
    # consume old EAX; definite store; load. The CAS cannot kill the first
    # store, but the later definite overwrite must kill both possible writers.
    graph = analyze("4889f789d089cb892ff00fb11f8b0f0f94c24189c0892f8b07", policy)
    assert {
        (0, 4, "RAW"),
        (1, 4, "RAW"),
        (2, 4, "RAW"),
        (3, 4, "RAW"),
        (3, 4, "WAW"),
        (3, 5, "RAW"),
        (4, 5, "RAW"),
        (4, 6, "RAW"),
        (4, 7, "RAW"),
        (8, 9, "RAW"),
    } <= kinds(graph)
    assert not {(3, 9, "RAW"), (4, 9, "RAW")} & kinds(graph)
    assert all(d.source < d.target for d in graph.dependencies)
    assert any(
        "(conditional)" in resource
        for d in graph.dependencies
        if (d.source, d.target, d.kind) == (4, 5, "RAW")
        for resource in d.resources
    )


def test_cas_does_not_infer_old_memory_from_expected_or_stored_constants():
    from bingraph.ddg.effects import instruction_effects

    # EAX=1; EBX=2; [RDI]=1; CMPXCHG [RDI],EBX; load [RDI+RAX].
    # Like ordinary loads, CAS returns unknown memory even after a known store.
    data = bytes.fromhex("b801000000bb0200000048c70701000000f0480fb11f8b1407")
    project = angr.load_shellcode(data, arch="amd64", load_address=0x1000)
    graph = get_ddg(project, 0x1000, len(data))
    effect = instruction_effects(project, graph.instructions, len(data))[-1]
    address = next(r.address for r in effect.reads if r.kind == "memory")
    assert any(v.startswith("ddg_unknown_") for v in address.variables)


@pytest.mark.parametrize("code", ["f00fc107", "8707", "f0ff07"])
def test_atomic_cas_retry_loops_still_fail_instead_of_reporting_one_iteration(code):
    # XADD, XCHG and INC lower to CAS with symbolic retry exits. CAS support
    # must not make the first attempted iteration appear to be complete.
    with pytest.raises(ValueError, match="symbolic instruction-internal loop"):
        analyze(code)


@pytest.mark.parametrize(
    "arch,load,store,width",
    [
        ("armel", "9f1fd2e1", "967fc1e1", 1),
        ("armel", "9f1ff2e1", "967fe1e1", 2),
        ("armel", "9f1f92e1", "967f81e1", 4),
        ("armel", "9f0fb3e1", "964fa1e1", 8),
        ("aarch64", "697c5f88", "677c0b88", 4),
        ("aarch64", "697c5fc8", "677c0bc8", 8),
        ("mips64", "c0640000", "e2210000", 4),
        ("mipsel", "2c3cc9c0", "2c3ccbe0", 4),
        ("ppc32", "7d40e028", "7cc0e12d", 4),
        ("ppc64le", "a948607c", "add9007d", 8),
    ],
)
def test_llsc_uses_vex_transfer_widths_not_success_flag_width(arch, load, store, width):
    from bingraph.ddg.effects import instruction_effects

    # Exercise native LL/SC in both byte orders, including paired ARM words.
    # SC may begin a block: its reservation can have been set before entry.
    for code, linked in [(load, True), (store, False)]:
        data = bytes.fromhex(code)
        project = angr.load_shellcode(data, arch=arch, load_address=0x1000)
        graph = get_ddg(project, 0x1000, len(data))
        effect = instruction_effects(project, graph.instructions, len(data))[0]
        reads = [r for r in effect.reads if r.kind == "memory"]
        writes = [r for r in effect.writes if r.kind == "memory"]
        assert len(reads) == int(linked) and len(writes) == int(not linked)
        access = (reads if linked else writes)[0]
        assert access.size == width and access.conditional == (not linked)
        assert any(r.kind == "register" for r in effect.reads)
        assert any(r.kind == "register" for r in effect.writes)
        assert any(r.kind == "reservation" for r in effect.reads) == (not linked)
        assert any(r.kind == "reservation" for r in effect.writes)


@pytest.mark.parametrize("policy", ["always", "may", "must"])
def test_store_conditional_preserves_possible_producers_and_success_dependencies(
    policy,
):
    from bingraph.ddg.effects import instruction_effects

    # Store; LL (its data is unused); SC with independent data; load; consume
    # SC status; definite store; load. SC does not kill the original producer.
    data = bytes.fromhex("004081e59f0f91e1932f81e1005091e5000052e3004081e5006091e5")
    project = angr.load_shellcode(data, arch="armel", load_address=0x1000)
    graph = get_ddg(project, 0x1000, len(data), policy)
    assert {
        (0, 1, "RAW"),
        (1, 2, "RAW"),
        (0, 3, "RAW"),
        (2, 3, "RAW"),
        (2, 4, "RAW"),
        (5, 6, "RAW"),
    } <= kinds(graph)
    assert not {(0, 6, "RAW"), (2, 6, "RAW")} & kinds(graph)
    effect = instruction_effects(project, graph.instructions, len(data))[2]
    assert {"r1", "r3", "reservation"} <= {r.label for r in effect.reads}
    assert any(
        "(conditional)" in resource
        for d in graph.dependencies
        if (d.source, d.target, d.kind) == (2, 3, "RAW")
        for resource in d.resources
    )
    assert all(d.source < d.target for d in graph.dependencies)


@pytest.mark.parametrize("cancel", [False, True])
@pytest.mark.parametrize("policy", ["always", "may", "must"])
def test_reservation_order_is_independent_of_alias_policy_and_tracks_cancellation(
    cancel, policy
):
    from bingraph.ddg.effects import instruction_effects

    # LL R1,[R2]; optional CLREX; SC R7,R6,[R3]. Neither SC's address nor
    # its data uses the LL result, and the memory policy may exclude overlap.
    data = bytes.fromhex("9f1f92e1" + ("1ff07ff5" if cancel else "") + "967f83e1")
    project = angr.load_shellcode(data, arch="armel", load_address=0x1000)
    graph = get_ddg(project, 0x1000, len(data), policy)
    edges = {
        (d.source, d.target, d.kind)
        for d in graph.dependencies
        if "reservation" in d.resources
    }
    assert (int(cancel), 1 + int(cancel), "RAW") in edges
    if cancel:
        assert (0, 1, "WAW") in edges and (0, 2, "RAW") not in edges
        effect = instruction_effects(project, graph.instructions, len(data))[1]
        assert not effect.reads  # CLREX cancels state; it is not a memory fence.
        assert {r.kind for r in effect.writes} == {"reservation"}


@pytest.mark.parametrize("operation", ["9f7f91e1", "967f81e1"])
def test_linked_memory_and_conditional_success_stay_unknown(operation):
    from bingraph.ddg.effects import instruction_effects

    # Matching LL; LL R7,[R1] or SC R7,R6,[R1]; load [R1+R7]. Even a local
    # reservation cannot establish SC success in the presence of interference.
    data = bytes.fromhex("9f0f91e1" + operation + "070091e7")
    project = angr.load_shellcode(data, arch="armel", load_address=0x1000)
    graph = get_ddg(project, 0x1000, len(data))
    effect = instruction_effects(project, graph.instructions, len(data))[-1]
    address = next(r.address for r in effect.reads if r.kind == "memory")
    assert any(v.startswith("ddg_unknown_") for v in address.variables)


def test_conditional_accesses_do_not_prove_overwrites_or_merge_into_definite_ranges():
    from dataclasses import replace
    from angr import claripy
    from bingraph.ddg.memory import coalesce_memory, covers
    from bingraph.ddg.models import Resource

    definite = Resource("memory", 0, 4, "mem[p]:4", claripy.BVV(0x1000, 64))
    possible = replace(definite, conditional=True)
    assert not covers(possible, definite)
    assert covers(definite, possible)
    assert coalesce_memory([definite, possible]) == [possible, definite]


@pytest.mark.parametrize(
    "prefix,code,keep_itstate",
    [
        ("00bf" * 9, "012002210332", False),  # Proven outside any IT region.
        ("", "012002210332", True),  # Incoming IT state is unknown.
        ("00bf" * 9, "08bf014600220132", True),  # IT EQ; conditional MOV.
        ("00bf" * 9, "0120022108bf", True),  # IT starts at the end of the block.
    ],
)
def test_only_proven_reset_only_itstate_is_omitted(prefix, code, keep_itstate):
    from bingraph.ddg.effects import instruction_effects

    data = bytes.fromhex(code)
    project = angr.load_shellcode(
        bytes.fromhex(prefix) + data, arch="armel", load_address=0x1000
    )
    address = (0x1000 + len(bytes.fromhex(prefix))) | 1
    graph = get_ddg(project, address, len(data))
    effects = instruction_effects(project, graph.instructions, len(data))
    assert (
        any(r.label == "itstate" for e in effects for r in e.reads + e.writes)
        == keep_itstate
    )
    if not keep_itstate:
        # MOVS still overwrites real flags; removing bookkeeping must not
        # discard output dependencies on architectural condition codes.
        assert any(
            d.kind == "WAW" and "cc_op" in d.resources for d in graph.dependencies
        )
    elif prefix == "" or code.startswith("08bf"):
        assert any(
            d.kind == "RAW" and "itstate" in d.resources for d in graph.dependencies
        )


@pytest.mark.parametrize("partial", [False, True])
def test_symbolic_or_partial_itstate_writes_prevent_reset_filter(monkeypatch, partial):
    from types import SimpleNamespace
    import pyvex
    from bingraph.ddg.effects import instruction_effects

    project = angr.load_shellcode(
        bytes.fromhex("00bf" * 9 + "01200221"), arch="armel", load_address=0x1000
    )
    address, size = 0x1013, 4
    block = project.factory.block(
        address, size=size, opt_level=0, cross_insn_opt=False, strict_block_end=True
    )
    offset, _ = project.arch.registers["itstate"]
    value = (
        pyvex.expr.Const(pyvex.const.U8(0))
        if partial
        else pyvex.expr.Get(project.arch.registers["r0"][0], "Ity_I32")
    )
    block.vex.statements.append(pyvex.stmt.Put(value, offset + int(partial)))
    monkeypatch.setattr(
        project.factory, "block", lambda *a, **k: SimpleNamespace(vex=block.vex)
    )
    graph = get_ddg(project, address, size)
    effects = instruction_effects(project, graph.instructions, size)
    assert all(any(r.label == "itstate" for r in e.writes) for e in effects)


def test_dependencies_span_lifter_instruction_limits():
    # mov eax,1; 98 nops; mov ebx,eax; mov eax,2. The reads and writes
    # after VEX's first 99 instructions must retain earlier producers/readers.
    graph = analyze("b801000000" + "90" * 98 + "89c3b802000000")
    assert len(graph.instructions) == 101
    assert {(0, 99, "RAW"), (99, 100, "WAR"), (0, 100, "WAW")} <= kinds(graph)


def test_arm_division_block_includes_return_after_lifter_limit():
    project = load_project(Path("angr-binaries/tests/armel/test_division"))
    graph = get_ddg(project, 0x86BC, 400)
    assert len(graph.instructions) == 100
    assert graph.instructions[-1].address == 0x8848
    assert graph.instructions[-1].mnemonic == "bx"
    assert all(edge.source < edge.target for edge in graph.dependencies)
    assert not any("lifted value" in d.resources for d in graph.dependencies)
    assert (1, 2, "RAW") in kinds(graph)
    assert (1, 3, "RAW") in kinds(graph)
    assert (2, 3, "RAW") not in kinds(graph)


def test_analysis_cache_and_address_labels_are_deterministic():
    data = bytes.fromhex("894424088b5c2410")
    p = angr.load_shellcode(data, arch="amd64", load_address=0x1000)
    first = get_ddg(p, 0x1000, len(data), "always")
    assert get_ddg(p, 0x1000, len(data), "always") is first
    other = angr.load_shellcode(data, arch="amd64", load_address=0x1000)
    assert (
        get_ddg(other, 0x1000, len(data), "always").dependencies == first.dependencies
    )


def test_memory_policies_and_unknown_pointers():
    # mov [rsp+8],eax; mov ebx,[rsp+16]
    code = "894424088b5c2410"
    assert (0, 1, "RAW") in kinds(analyze(code))
    assert (0, 1, "RAW") not in kinds(analyze(code, "may"))
    project = angr.load_shellcode(
        bytes.fromhex(code), arch="amd64", load_address=0x1000
    )
    assert (
        get_ddg(project, 0x1000, 8).dependencies
        == get_ddg(project, 0x1000, 8, "must").dependencies
    )
    assert render_ddg(project, 0x1000, 8, format="raw") == render_ddg(
        project, 0x1000, 8, "must", "raw", "raw,war,waw"
    )
    # Different unknown registers do not prove disjoint memory.
    assert (0, 1, "RAW") in kinds(analyze("89078b1e", "may"))
    # A known overwritten store is not the producer of the following load.
    edges = kinds(analyze("8907891f8b0f", "may"))
    assert (0, 2, "RAW") not in edges
    assert (1, 2, "RAW") in edges


@pytest.mark.parametrize(
    "policy,unknown,disjoint",
    [("always", True, True), ("may", True, False), ("must", False, False)],
)
def test_memory_policies_distinguish_unknown_and_proven_overlap(
    policy, unknown, disjoint
):
    for kind, different, same in (
        ("RAW", "89078b1e", "89078b1f"),
        ("WAR", "8b07891e", "8b07891f"),
        ("WAW", "8907891e", "8907891f"),
    ):
        # rdi and rsi might alias, but equal rdi-relative ranges definitely do.
        assert ((0, 1, kind) in kinds(analyze(different, policy))) == unknown
        assert (0, 1, kind) in kinds(analyze(same, policy))
    assert ((0, 1, "RAW") in kinds(analyze("894424088b5c2410", policy))) == disjoint
    # Partial overlap is sufficient; an entire access need not be covered.
    assert (0, 1, "RAW") in kinds(analyze("48894424088b5c240c", policy))
    # A register copy can prove aliasing even with different operand names.
    assert (1, 2, "RAW") in kinds(analyze("4889fe89078b1e", policy))


def test_must_policy_does_not_turn_omitted_aliases_into_killing_proofs():
    # store [rdi]; store [rsi]; load [rdi]. Omitting rsi's possible edges
    # does not prove anything about rsi or kill the known rdi producer.
    edges = kinds(analyze("8907891e8b0f", "must"))
    assert (0, 2, "RAW") in edges
    assert (1, 2, "RAW") not in edges


def test_render_uses_existing_pipeline_and_retains_isolated_instructions():
    data = bytes.fromhex("b80100000089c3b80200000090")
    project = angr.load_shellcode(data, arch="amd64", load_address=0x1000)
    dot = render_ddg(
        project, 0x1000, len(data), format="raw", dependencies="raw,war,waw"
    )
    edges = pydot.graph_from_dot_data(dot)[0].get_edges()
    assert {e.get("style") for e in edges} == {"solid", "dashed", "dotted"}
    assert all(
        not e.get("label").strip('"').startswith(("RAW:", "WAR:", "WAW:"))
        for e in edges
    )
    assert "nop" in dot


def test_dependency_filters_preserve_nodes_and_cached_analysis():
    data = bytes.fromhex("b80100000089c3b80200000090")
    project = angr.load_shellcode(data, arch="amd64", load_address=0x1000)
    graph = get_ddg(project, 0x1000, len(data))
    before = graph.dependencies
    for selection, styles in (
        ("raw", {"solid"}),
        ("war", {"dashed"}),
        ("waw", {"dotted"}),
        ("raw,war", {"solid", "dashed"}),
        ("raw,waw", {"solid", "dotted"}),
        ("war,waw", {"dashed", "dotted"}),
        ("raw,war,waw", {"solid", "dashed", "dotted"}),
        (" WAW, raw,raw ", {"solid", "dotted"}),
    ):
        dot = render_ddg(
            project, 0x1000, len(data), format="raw", dependencies=selection
        )
        parsed = pydot.graph_from_dot_data(dot)[0]
        assert {e.get("style") for e in parsed.get_edges()} == styles
        assert len([n for n in parsed.get_nodes() if n.get("label")]) == 4
        assert get_ddg(project, 0x1000, len(data)) is graph
        assert graph.dependencies == before
    default = render_ddg(project, 0x1000, len(data), format="raw")
    assert default == render_ddg(
        project, 0x1000, len(data), format="raw", dependencies="raw,war,waw"
    )
    # Invalid selections must not silently produce an edgeless graph.
    with pytest.raises(ValueError, match="Dependencies must be"):
        render_ddg(project, 0x1000, len(data), dependencies="raw,unknown")


def test_memory_labels_are_compact_without_losing_alias_details():
    project = angr.load_shellcode(
        bytes.fromhex("89078b1e"), arch="amd64", load_address=0x1000
    )  # store [rdi]; load [rsi]: possible alias even under may policy
    graph = get_ddg(project, 0x1000, 4, "may")
    before = graph.dependencies
    dot = render_ddg(project, 0x1000, 4, "may", "raw")
    edge = pydot.graph_from_dot_data(dot)[0].get_edges()[0]
    assert edge.get("label").strip('"') == "[rsi]"
    assert "RAW:" in edge.get("tooltip")
    assert "mem[entry_rsi]:4 (may alias)" in edge.get("tooltip")
    assert graph.dependencies == before
    # Concrete and signed base offsets keep their assembly-like appearance.
    from bingraph.ddg.render import DependencyEdges

    for original, expected in (
        ("mem[entry_rsp-0x28]:8", "[rsp-0x28]"),
        ("mem[entry_rdi+0x40]:8", "[rdi+0x40]"),
        ("mem[0x400000]:4", "[0x400000]"),
        ("mem? (may alias)", "[?]"),
    ):
        assert DependencyEdges._resource_label(original) == expected


@pytest.mark.parametrize(
    "code,target,label",
    [
        ("891e488b07894010", 2, "[rax+0x10]"),
        ("891e488b078940f0", 2, "[rax-0x10]"),
        ("891e488b074883c008894010", 3, "[rax+0x10]"),
        ("891e488b074889c131c0894110", 4, "[rcx+0x10]"),
        ("891e488b2750", 2, "[rsp-0x8]"),
        ("891e488b07488b0a890408", 3, "[?]"),
    ],
)
def test_loaded_pointer_labels_use_read_registers_without_changing_aliases(
    code, target, label
):
    # Start with a store through rsi. A later store through a loaded pointer
    # may alias it, irrespective of the friendly name used for that pointer.
    data = bytes.fromhex(code)
    project = angr.load_shellcode(data, arch="amd64", load_address=0x1000)
    graph = get_ddg(project, 0x1000, len(data), "may")
    dependency = next(
        d
        for d in graph.dependencies
        if (d.source, d.target, d.kind) == (0, target, "WAW")
    )
    assert dependency.display_resources == (label,)
    assert "ddg_unknown_" in dependency.resources[0]
    assert "may alias" in dependency.resources[0]
    dot = render_ddg(project, 0x1000, len(data), memory="may", format="raw")
    edge = next(
        e
        for e in pydot.graph_from_dot_data(dot)[0].get_edges()
        if e.get_source().strip('"') == "0x1000"
        and e.get_destination().strip('"') == hex(graph.instructions[target].address)
    )
    assert edge.get("label").strip('"') == label
    assert dependency.resources[0] in edge.get("tooltip")


def test_mips_loaded_pointer_label_uses_s2_instead_of_its_unknown_id():
    project = load_project(Path("angr-binaries/tests/mipsel/mips_syscall_demo"))
    graph = get_ddg(project, 0x4064F4, 112, "may")
    dependencies = [
        d
        for d in graph.dependencies
        if graph.instructions[d.target].address == 0x40654C
        and any("ddg_unknown_" in resource for resource in d.resources)
    ]
    assert dependencies
    assert all(d.display_resources == ("[s2]",) for d in dependencies)


def test_visible_raw_subsumes_only_parallel_selected_ordering_edges():
    # mul ebx; xchg eax,ebx. RAW/WAW on eax, WAR on the input-only ebx.
    project = angr.load_shellcode(
        bytes.fromhex("f7e393"), arch="amd64", load_address=0x1000
    )
    graph = get_ddg(project, 0x1000, 3)
    before = graph.dependencies
    assert kinds(graph) == {(0, 1, "RAW"), (0, 1, "WAR"), (0, 1, "WAW")}
    for selection in (
        "raw",
        "raw,war",
        "raw,waw",
        "raw,war,waw",
        "war",
        "waw",
        "war,waw",
    ):
        dot = render_ddg(project, 0x1000, 3, format="raw", dependencies=selection)
        edges = pydot.graph_from_dot_data(dot)[0].get_edges()
        selected = set(selection.upper().split(","))
        if "RAW" in selected:
            assert len(edges) == 1
            assert edges[0].get("style") == "solid"
        else:
            assert len(edges) == len(selected)
            assert {e.get("style") for e in edges} == {
                {"WAR": "dashed", "WAW": "dotted"}[kind] for kind in selected
            }
        tooltip = " ".join(e.get("tooltip") for e in edges)
        for kind in ("RAW", "WAR", "WAW"):
            assert (kind + ":" in tooltip) == (kind in selected)
        assert graph.dependencies == before


@pytest.mark.parametrize("size", [0, -1, 4097, 1])
def test_invalid_bounds(size):
    project = angr.load_shellcode(
        bytes.fromhex("b801000000"), arch="amd64", load_address=0x1000
    )
    with pytest.raises(ValueError):
        get_ddg(project, 0x1000, size)
