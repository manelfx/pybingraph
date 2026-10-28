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
    project = angr.load_shellcode(
        bytes.fromhex("00009115"), arch="armel", load_address=0x1000
    )  # ldrne r0,[r1]: genuinely conditional memory remains unsupported.
    with pytest.raises(ValueError, match="Unsupported DDG conditional LoadG"):
        get_ddg(project, 0x1000, 4)


@pytest.mark.parametrize("code,width", [("0878", 1), ("0888", 2)])
def test_constant_guarded_thumb_loads_keep_their_memory_width(code, width):
    from bingraph.ddg.effects import instruction_effects

    # IT EQ; NOP exhausts an explicit IT state before LDRB/LDRH r0,[r1].
    # A raw Thumb entry alone does not prove its incoming IT state is zero.
    project = angr.load_shellcode(
        bytes.fromhex("08bf00bf" + code), arch="armel", load_address=0x1000
    )
    graph = get_ddg(project, 0x1001, 6)
    reads = instruction_effects(project, graph.instructions, 6)[-1].reads
    assert [r.size for r in reads if r.kind == "memory"] == [width]


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
