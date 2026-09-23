"""Tests for the experimental CFG extractor independent of CFGFast."""

from __future__ import annotations

from collections import deque
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from angr import KnowledgeBase
from elftools.common.exceptions import ELFRelocationError
import networkx as nx

from bingraph.cfg_extract import build_extracted_cfg
from bingraph.cfg_extract.anomalies import find_extracted_cfg_anomalies
from bingraph.cfg_extract import builder as builder_module
from bingraph.cfg_extract import exceptions as exceptions_module
from bingraph.cfg_extract.exceptions import (
    ExceptionalCallSite,
    exceptional_call_sites_for_function,
)
from bingraph.cfg_extract.sweep import (
    ExecutableSweep,
    ExecutableSweepAudit,
    ReconnectingComponents,
    recover_executable_components,
    select_reconnecting_components,
)
from bingraph.cfg.models import BlockSpec, FunctionBounds, StaticJumpTable
from bingraph.cfg.models import StaticJumpTablePlan
from bingraph.cfg.decode import (
    decode_bounded_block,
    decode_raw_capstone_insns,
    lift_block_terminator,
    target_is_known_nonreturning,
)
from bingraph.cfg_extract.models import ExtractedCFGStats, ExtractedCFGSummary
from bingraph.core import project as project_module


def test_extract_builder_decodes_a_bounded_function_without_cfgfast() -> None:
    """Build normal function blocks without requesting an angr CFG analysis."""

    project = project_module.load_project(
        Path("angr-binaries/tests/samples/ais3_crackme")
    )
    kb = KnowledgeBase(project)

    with patch.object(project.analyses, "CFGFast", side_effect=AssertionError):
        cfg = build_extracted_cfg(project, kb, 0x40043C)

    nodes = [node for node in cfg.graph.nodes() if not node.is_simprocedure]
    assert [node.addr for node in nodes] == [0x40043C, 0x40044C, 0x40044E]
    assert cfg.functions.get(0x40043C) is not None
    assert cfg.extract_summary == ExtractedCFGSummary(
        normal_blocks=3,
        synthetic_leaves=1,
        nodes=4,
        edges=4,
        calls=1,
        direct_branches=1,
        conditional_branches=1,
        returns=1,
        direct_edges=1,
        fallthrough_edges=2,
    )


def test_extract_mode_bypasses_fast_cfg(monkeypatch) -> None:
    """Route the public extract mode directly to independent construction."""

    project = project_module.load_project(
        Path("angr-binaries/tests/samples/ais3_crackme")
    )
    project_module.get_cfg.cache_clear()
    monkeypatch.setattr(
        project_module,
        "_get_fast_cfg",
        lambda *_args: (_ for _ in ()).throw(AssertionError("CFGFast called")),
    )

    cfg = project_module.get_cfg(project, 0x40043C, "extract")

    assert sum(not node.is_simprocedure for node in cfg.graph.nodes()) == 3


def test_extract_recovers_elf_lsda_landing_pads() -> None:
    """Recover Rust cleanup blocks from exact LSDA call-site metadata."""

    project = project_module.load_project(Path("angr-binaries/tests/x86_64/fmt-rust"))
    kb = KnowledgeBase(project)
    session = builder_module._ExtractionSession(project, kb, 0x4B1040)

    assert exceptional_call_sites_for_function(project, session.bounds) == (
        ExceptionalCallSite(0x4B106E, 0x4B1073, 0x4B10A5),
        ExceptionalCallSite(0x4B107C, 0x4B1081, 0x4B1094),
        ExceptionalCallSite(0x4B109E, 0x4B10B4, 0x4B10BC),
    )

    cfg = session.build()
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    assert {0x4B1094, 0x4B10A5, 0x4B10BC} <= nodes.keys()
    for source_addr, target_addr in (
        (0x4B1067, 0x4B10A5),
        (0x4B1075, 0x4B1094),
        (0x4B1094, 0x4B10BC),
        (0x4B10A5, 0x4B10BC),
    ):
        edge = cfg.graph.get_edge_data(nodes[source_addr], nodes[target_addr])
        assert edge is not None
        assert edge["jumpkind"] == "Ijk_Boring"
        assert edge["exceptional"] is True


def test_extract_names_nonreturning_unwind_plt_call() -> None:
    """Keep the unwind call while omitting its impossible continuation."""

    project = project_module.load_project(Path("angr-binaries/tests/x86_64/fmt-rust"))
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x4B1040
    )
    cfg = session.build()
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    successors = tuple(cfg.graph.successors(nodes[0x4B10B4]))

    assert target_is_known_nonreturning(project, 0x5725A0)
    assert not target_is_known_nonreturning(project, 0x572590)
    assert len(successors) == 1
    assert successors[0].addr == 0x5725A0
    assert successors[0].name == "_Unwind_Resume"
    assert (
        cfg.graph.get_edge_data(nodes[0x4B10B4], successors[0])["jumpkind"]
        == "Ijk_Call"
    )
    assert 0x4B10BC in nodes


def test_extract_keeps_lsda_cleanup_without_assertion_fakeret() -> None:
    """An assertion failure cannot return into a separate cleanup landing pad."""

    cases = (
        ("x86_64/static", 0x40FD30, 0x401750, 0x4102B4, 0x410175, 0x4102CD),
        (
            "i386/bronze_ropchain",
            0x8050750,
            0x806F0B0,
            0x8050D9E,
            0x8050CD7,
            0x8050DA3,
        ),
        (
            "mipsel/mips_syscall_demo",
            0x43EFD4,
            0x42A660,
            0x43F79C,
            0x43F4E0,
            0x43F7BC,
        ),
    )
    for binary, entry, callee, assertion_block, call_block, landing_pad in cases:
        project = project_module.load_project(Path("angr-binaries/tests") / binary)
        cfg = build_extracted_cfg(project, KnowledgeBase(project), entry)
        nodes = {
            node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure
        }

        assert target_is_known_nonreturning(project, callee)
        assert (
            cfg.graph.get_edge_data(nodes[assertion_block], nodes[landing_pad]) is None
        )
        edge = cfg.graph.get_edge_data(nodes[call_block], nodes[landing_pad])
        assert edge is not None and edge["exceptional"] is True


def test_extract_skips_unlinked_elf_exception_metadata() -> None:
    """PPC relocatable objects cannot supply linked LSDA addresses."""

    project = project_module.load_project(Path("angr-binaries/tests/ppc/partial.o"))
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x400000
    )
    assert exceptional_call_sites_for_function(project, session.bounds) == ()
    cfg = session.build()
    assert cfg.extract_stats.exception_edges_added == 0


def test_extract_recovers_lsda_on_other_elf_architectures() -> None:
    """Keep exceptional-flow recovery for linked i386 and MIPS binaries."""

    for binary, function_addr in (
        ("angr-binaries/tests/i386/bronze_ropchain", 0x804FCD0),
        ("angr-binaries/tests/mipsel/mips_syscall_demo", 0x407470),
    ):
        project = project_module.load_project(Path(binary))
        session = builder_module._ExtractionSession(
            project, KnowledgeBase(project), function_addr
        )
        assert exceptional_call_sites_for_function(project, session.bounds)
        assert session.build().extract_stats.exception_edges_added > 0


def test_extract_skips_elf_relocation_errors() -> None:
    """An unsupported ELF relocation must not abort CFG extraction."""

    project = project_module.load_project(Path("angr-binaries/tests/x86_64/fmt-rust"))
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x4B1040
    )
    with patch.object(
        exceptions_module,
        "_exception_sites_by_elf",
        side_effect=ELFRelocationError("Unsupported relocation type: 26"),
    ):
        assert exceptional_call_sites_for_function(project, session.bounds) == ()


def test_extract_preserves_conditional_return_fallthrough() -> None:
    """Keep the non-returning paths of VEX conditional returns recoverable."""

    project = project_module.load_project(Path("angr-binaries/tests/armel/btrfs.ko"))
    bxeq_bounds = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x44D480
    ).bounds
    popeq_bounds = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x473818
    ).bounds

    bxeq = decode_bounded_block(
        project,
        bxeq_bounds,
        0x44D480,
        set(),
        preserve_conditional_return_fallthrough=True,
    )
    popeq = decode_bounded_block(
        project,
        popeq_bounds,
        0x473828,
        set(),
        preserve_conditional_return_fallthrough=True,
    )

    assert bxeq is not None
    assert bxeq.jumpkind == "Ijk_Boring"
    assert bxeq.fallthrough_addr == 0x44D488
    assert popeq is not None
    assert popeq.jumpkind == "Ijk_Boring"
    assert popeq.fallthrough_addr == 0x473848


def test_extract_does_not_fall_through_from_an_unconditional_thumb_return() -> None:
    """Ignore VEX's generic inactive-IT exit after an ordinary Thumb return."""

    project = project_module.load_project(
        Path("angr-binaries/tests/armel/Nucleo_read_hyperterminal.elf")
    )
    bounds = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x80023A5
    ).bounds

    block = decode_bounded_block(
        project,
        bounds,
        0x80023AD,
        set(),
        preserve_conditional_return_fallthrough=True,
    )

    assert block is not None
    assert block.jumpkind == "Ijk_Ret"
    assert block.fallthrough_addr is None


def test_extract_preserves_powerpc_conditional_return_fallthrough() -> None:
    """Keep a non-ARM VEX conditional-return continuation."""

    project = project_module.load_project(
        Path("angr-binaries/tests/ppc64el/fauxware_static")
    )
    bounds = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x10019100
    ).bounds

    block = decode_bounded_block(
        project,
        bounds,
        0x10019124,
        set(),
        preserve_conditional_return_fallthrough=True,
    )

    assert block is not None
    assert block.jumpkind == "Ijk_Boring"
    assert block.fallthrough_addr == 0x10019140


def test_extract_keeps_powerpc_pc_materialization_in_one_block() -> None:
    """Retain a branch-and-link to its next instruction as linear code."""

    project = project_module.load_project(Path("angr-binaries/tests/ppc/ld.so.1"))
    bounds = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x40A320
    ).bounds

    continued_transfers: list[int] = []
    block = decode_bounded_block(
        project,
        bounds,
        0x40A320,
        set(),
        on_linear_direct_transfer=continued_transfers.append,
    )

    assert block is not None
    assert 0x40A32C in block.instruction_addrs
    assert 0x40A330 in block.instruction_addrs
    assert continued_transfers == [0x40A32C]

    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x40A320)

    assert cfg.extract_stats.linear_direct_transfers_continued == 1


def test_extract_retains_external_call_target() -> None:
    """Keep a resolved direct callee even when it lies outside function bounds."""

    project = project_module.load_project(Path("angr-binaries/tests/armel/btrfs.ko"))
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x44D480
    )

    block = decode_bounded_block(project, session.bounds, 0x44D534, set())

    assert block is not None
    assert block.jumpkind == "Ijk_Call"
    assert block.direct_targets == (0x500048,)


def test_extract_retains_unnamed_external_call_target() -> None:
    """Keep a direct callee whose address has no loader symbol."""

    project = project_module.load_project(
        Path("angr-binaries/tests/ppc64el/fauxware_static")
    )
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x10028860
    )

    block = decode_bounded_block(project, session.bounds, 0x10028BC0, set())

    assert project.loader.find_symbol(0x10026988) is None
    assert block is not None
    assert block.jumpkind == "Ijk_Call"
    assert block.direct_targets == (0x10026988,)


def test_extract_models_an_unmapped_zero_call_as_unresolved() -> None:
    """Do not confuse an unresolved weak call with another zero-valued symbol."""

    project = project_module.load_project(
        Path("angr-binaries/tests/s390x/test-instr_s390x")
    )
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x80046660
    )

    block = decode_bounded_block(project, session.bounds, 0x80046680, set())

    assert block is not None
    assert block.jumpkind == "Ijk_Call"
    assert block.direct_targets == ()
    assert block.fallthrough_addr == 0x8004668C

    cfg = session.build()
    source = next(node for node in cfg.graph.nodes() if node.addr == 0x80046680)
    unresolved = next(
        node
        for node in cfg.graph.successors(source)
        if node.is_simprocedure and node.name == "UnresolvableCallTarget"
    )
    assert cfg.graph.get_edge_data(source, unresolved)["jumpkind"] == "Ijk_Call"
    assert cfg.extract_stats.unresolved_call_targets == 1


def test_extract_models_syscalls_as_call_like_block_terminators() -> None:
    """End at a syscall and retain both its service and returning paths."""

    project = project_module.load_project(
        Path("angr-binaries/tests/ppc64el/fauxware_static")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x1000ED70)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    syscall_block = nodes[0x1000F09C]

    assert tuple(syscall_block.instruction_addrs) == (0x1000F09C,)
    successors = tuple(cfg.graph.successors(syscall_block))
    syscall = next(node for node in successors if node.is_syscall)
    assert syscall.addr == 0x10300494
    assert syscall.name == "sys_293"
    continuation = next(node for node in successors if node.addr == 0x1000F0A0)
    assert (
        cfg.graph.get_edge_data(syscall_block, syscall)["jumpkind"] == "Ijk_Sys_syscall"
    )
    assert (
        cfg.graph.get_edge_data(syscall_block, continuation)["jumpkind"]
        == "Ijk_FakeRet"
    )


def test_extract_resolves_a_static_nonreturning_syscall() -> None:
    """Use the active syscall ABI when one local block fixes its number."""

    project = project_module.load_project(
        Path("angr-binaries/tests/mipsel/mips_syscall_demo")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x400EFC)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    syscall_block = nodes[0x400FBC]

    successors = tuple(cfg.graph.successors(syscall_block))
    assert len(successors) == 1
    syscall = successors[0]
    assert syscall.is_syscall
    assert syscall.name == "exit"
    assert syscall.addr == 0x700004
    assert 0x400FC4 not in nodes
    assert 0x400FD0 not in nodes
    assert cfg.extract_stats.static_syscalls_resolved == 1
    assert cfg.extract_stats.static_syscall_fallthroughs_suppressed == 1


def test_extract_models_vex_traps_without_a_linear_successor() -> None:
    """Keep MIPS ``break`` as VEX's synchronous trap instead of falling through."""

    project = project_module.load_project(Path("angr-binaries/tests/mipsel/busybox"))
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x409C38
    )

    block = decode_bounded_block(project, session.bounds, 0x409C64, set())

    assert block is not None
    assert block.jumpkind == "Ijk_Terminal"
    assert block.fallthrough_addr is None


def test_extract_models_ud2_as_a_terminal_trap() -> None:
    """Keep x86 ``ud2`` as a trap despite VEX's ``Ijk_NoDecode`` result."""

    project = project_module.load_project(
        Path("angr-binaries/tests/x86_64/rust_hello_world")
    )
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x427680
    )

    block = decode_bounded_block(project, session.bounds, 0x427771, set())

    assert block is not None
    assert block.instruction_addrs == (0x427771,)
    assert block.jumpkind == "Ijk_Terminal"
    assert block.fallthrough_addr is None


def test_extract_suppresses_fakeret_for_a_static_nonreturning_call() -> None:
    """Resolve a GOT-loaded ``abort`` target without constructing CFGFast."""

    project = project_module.load_project(
        Path("angr-binaries/tests/x86_64/rust_hello_world")
    )
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x41FB60
    )

    block = decode_bounded_block(project, session.bounds, 0x41FFB5, set())

    assert block is not None
    assert block.jumpkind == "Ijk_Call"
    assert block.direct_targets == (0x500020,)
    assert block.fallthrough_addr is None


def test_extract_resolves_returning_static_memory_call_targets() -> None:
    """Keep exact returning calls made through constant GOT slots."""

    cases = (
        (
            "angr-binaries/tests/x86_64/"
            "1cbbf108f44c8f4babde546d26425ca5340dccf878d306b90eb0fbec2f83ab51",
            0x41DC10,
            0x41DC53,
            0x500190,
        ),
        (
            "angr-binaries/tests/x86_64/rust_hello_world",
            0x4075D0,
            0x407662,
            0x5000F0,
        ),
        (
            "angr-binaries/tests/mips/dir",
            0x419AC0,
            0x419B28,
            0x500064,
        ),
        (
            "angr-binaries/tests/mipsel/btrfs-tools_btrfs-calc-size",
            0x41A380,
            0x41A3E8,
            0x50011C,
        ),
    )

    for binary, function_addr, call_addr, target in cases:
        project = project_module.load_project(Path(binary))
        session = builder_module._ExtractionSession(
            project, KnowledgeBase(project), function_addr
        )
        block = decode_bounded_block(
            project,
            session.bounds,
            call_addr,
            set(),
            resolve_static_memory_calls=True,
        )

        assert block is not None
        assert block.jumpkind == "Ijk_Call"
        assert block.direct_targets == (target,)
        assert block.fallthrough_addr is not None

        cfg = session.build()
        source = next(node for node in cfg.graph.nodes() if node.addr == call_addr)
        assert target in {node.addr for node in cfg.graph.successors(source)}


def test_extract_models_static_memory_tail_jump_as_explicit_exit() -> None:
    """Render a proven GOT tail target as an explicit jump exit."""

    project = project_module.load_project(
        Path("angr-binaries/tests/x86_64/rust_hello_world")
    )
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x424060
    )

    block = decode_bounded_block(project, session.bounds, 0x424110, set())

    assert block is not None
    assert block.jumpkind == "Ijk_Boring"
    assert block.direct_targets == (0x408A80,)

    cfg = session.build()
    source = next(node for node in cfg.graph.nodes() if node.addr == 0x424110)
    assert [node.addr for node in cfg.graph.successors(source)] == [0x408A80]
    assert cfg.extract_stats.unresolved_indirect_targets == 0

    # The same exact static-load form remains an ordinary in-function edge.
    tail_block = project.factory.block(
        0x424110,
        size=0x17,
        strict_block_end=True,
        cross_insn_opt=False,
    )
    executable_section = SimpleNamespace(is_executable=True)
    in_function_project = SimpleNamespace(
        arch=SimpleNamespace(name=project.arch.name),
        factory=SimpleNamespace(block=lambda *_args, **_kwargs: tail_block),
        loader=SimpleNamespace(
            memory=SimpleNamespace(
                load=lambda _addr, _size: (0x424127).to_bytes(8, byteorder="little")
            ),
            find_object_containing=lambda _addr: SimpleNamespace(
                find_section_containing=lambda _target: executable_section
            ),
            extern_object=object(),
        ),
    )
    in_function = lift_block_terminator(
        in_function_project,
        session.bounds,
        [entry.insn for entry in tail_block.capstone.insns],
    )

    assert in_function is not None
    assert in_function.jumpkind == "Ijk_Boring"
    assert in_function.direct_targets == (0x424127,)

    extern_object = object()
    unknown_synthetic_project = SimpleNamespace(
        arch=SimpleNamespace(name=project.arch.name),
        factory=SimpleNamespace(block=lambda *_args, **_kwargs: tail_block),
        loader=SimpleNamespace(
            memory=SimpleNamespace(
                load=lambda _addr, _size: (0x500000).to_bytes(8, byteorder="little")
            ),
            find_object_containing=lambda _addr: extern_object,
            find_symbol=lambda _addr: SimpleNamespace(
                rebased_addr=0x500000,
                is_function=False,
            ),
            extern_object=extern_object,
        ),
    )
    unknown_synthetic = lift_block_terminator(
        unknown_synthetic_project,
        session.bounds,
        [entry.insn for entry in tail_block.capstone.insns],
    )

    assert unknown_synthetic.jumpkind == "Ijk_Boring"
    assert unknown_synthetic.direct_targets == ()


def test_extract_suppresses_fakeret_for_a_mips_pic_nonreturning_call() -> None:
    """Resolve MIPS ``$gp``-relative ``$t9`` calls to known no-return targets."""

    project = project_module.load_project(Path("angr-binaries/tests/mips/dir"))
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x40DB70
    )

    block = decode_bounded_block(project, session.bounds, 0x40DC74, set())

    assert block is not None
    assert block.jumpkind == "Ijk_Call"
    assert block.direct_targets == (0x5001AC,)
    assert block.fallthrough_addr is None


def test_extract_models_mips_pic_tail_jump_as_explicit_exit() -> None:
    """Render a resolved MIPS ``jr $t9`` tail call as a jump exit."""

    project = project_module.load_project(
        Path("angr-binaries/tests/mipsel/btrfs-tools_btrfs-calc-size")
    )
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x40D684
    )

    block = decode_bounded_block(project, session.bounds, 0x40D740, set())

    assert block is not None
    assert block.jumpkind == "Ijk_Boring"
    assert block.direct_targets == (0x42237C,)

    cfg = session.build()
    source = next(node for node in cfg.graph.nodes() if node.addr == 0x40D740)
    assert [node.addr for node in cfg.graph.successors(source)] == [0x42237C]
    assert cfg.extract_stats.unresolved_indirect_targets == 0

    full_block = project.factory.block(
        0x40D740,
        size=0x14,
        strict_block_end=True,
        cross_insn_opt=False,
    )
    tail_block = project.factory.block(
        0x40D74C,
        size=0x8,
        strict_block_end=True,
        cross_insn_opt=False,
    )
    executable_section = SimpleNamespace(is_executable=True)
    in_function_project = SimpleNamespace(
        arch=project.arch,
        factory=SimpleNamespace(
            block=lambda addr, **_kwargs: (
                project.factory.block(session.bounds.addr)
                if addr == session.bounds.addr
                else full_block
                if addr == 0x40D740
                else tail_block
            )
        ),
        loader=SimpleNamespace(
            memory=SimpleNamespace(
                load=lambda _addr, _size: (0x40D6A0).to_bytes(4, byteorder="little")
            ),
            find_object_containing=lambda _addr: SimpleNamespace(
                find_section_containing=lambda _target: executable_section
            ),
            extern_object=object(),
        ),
    )
    in_function = lift_block_terminator(
        in_function_project,
        session.bounds,
        [entry.insn for entry in full_block.capstone.insns],
    )

    assert in_function.jumpkind == "Ijk_Boring"
    assert in_function.direct_targets == (0x40D6A0,)


def test_extract_resolves_adjusted_mips_pic_tail_jump() -> None:
    """Resolve a MIPS PIC tail jump with one post-load target adjustment."""

    project = project_module.load_project(
        Path("angr-binaries/tests/mipsel/btrfs-tools_btrfs-calc-size")
    )
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x43B414
    )
    block = decode_bounded_block(project, session.bounds, 0x43B528, set())

    assert block is not None
    assert block.jumpkind == "Ijk_Boring"
    assert block.direct_targets == (0x43B100,)

    cfg = session.build()
    source = next(node for node in cfg.graph.nodes() if node.addr == 0x43B528)
    assert {node.addr for node in cfg.graph.successors(source)} == {0x43B100}


def test_extract_models_mips_pic_import_tail_jump_as_explicit_exit() -> None:
    """Render a MIPS GOT import tail callee as an explicit jump exit."""

    project = project_module.load_project(Path("angr-binaries/tests/mips/dir"))
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x40D80C
    )

    block = decode_bounded_block(project, session.bounds, 0x40D9B0, set())

    assert block is not None
    assert block.jumpkind == "Ijk_Boring"
    assert block.direct_targets == (0x500070,)


def test_extract_recovers_mips_pic_relative_jump_table() -> None:
    """Recover a bounded MIPS PIC table of ``$gp``-relative branch offsets."""

    project = project_module.load_project(Path("angr-binaries/tests/mipsel/busybox"))
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x40FDC0
    )

    session._decode_all_blocks()
    session._discover_static_jump_targets()

    targets = session.static_targets[0x40FFD4]
    assert len(targets) == 27
    assert targets[0] == 0x40FFF0
    assert targets[-1] == 0x410518
    assert session.unresolved_dispatcher_reasons.get(0x40FFD4) is None


def test_extract_recovers_a_masked_affine_x86_relative_jump_table() -> None:
    """Recover only the ordered low-nibble entries of an x86-64 table."""

    project = project_module.load_project(Path("angr-binaries/tests/x86_64/static"))
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x42C6B0
    )

    session._decode_all_blocks()
    session._discover_static_jump_targets()

    assert session.static_targets[0x42C723] == (
        0x42C7E0,
        0x42C900,
        0x42CA20,
        0x42CB40,
        0x42CC60,
        0x42CD80,
        0x42CEA0,
        0x42CFC0,
        0x42D0E0,
        0x42D200,
        0x42D320,
        0x42D440,
        0x42D560,
        0x42D680,
        0x42D7A0,
    )
    assert session.unresolved_dispatcher_reasons.get(0x42C723) is None


def test_extract_recovers_mips_pic_table_with_inline_scaled_index() -> None:
    """Accept a selector shifted in the dispatcher, not only its predecessor."""

    for path, function_addr, dispatcher_addr, expected_count in (
        ("mipsel/busybox", 0x412898, 0x413654, 6),
        ("mips/dir", 0x416960, 0x416A3C, 10),
    ):
        project = project_module.load_project(Path("angr-binaries/tests") / path)
        session = builder_module._ExtractionSession(
            project, KnowledgeBase(project), function_addr
        )

        session._decode_all_blocks()
        session._discover_static_jump_targets()

        assert len(session.static_targets[dispatcher_addr]) == expected_count
        assert session.unresolved_dispatcher_reasons.get(dispatcher_addr) is None


def test_extract_suppresses_fakeret_for_a_declared_nonreturning_symbol() -> None:
    """Use angr's libc declaration for an in-image ``__stack_chk_fail`` call."""

    project = project_module.load_project(
        Path("angr-binaries/tests/armel/libc-2.31.so")
    )
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x47A4E9
    )

    block = decode_bounded_block(
        project,
        session.bounds,
        0x47A551,
        set(),
        resolve_declared_nonreturning=True,
    )

    assert block is not None
    assert block.jumpkind == "Ijk_Call"
    assert block.direct_targets == (0x4AA5FD,)
    assert block.fallthrough_addr is None


def test_extract_does_not_fall_through_to_a_verified_literal_pool() -> None:
    """Reject a no-decode call continuation proven to be static data."""

    project = project_module.load_project(
        Path("angr-binaries/tests/armel/ld-linux.so.3")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x4165E8)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}

    assert 0x416674 not in nodes
    assert all(
        successor.addr != 0x416674
        for successor in cfg.graph.successors(nodes[0x416654])
    )
    assert cfg.extract_stats.data_leaders_rejected == 1
    assert cfg.extract_stats.data_region_observations > 0
    assert cfg.extract_stats.data_bytes_discovered > 0
    assert cfg.extract_stats.call_fallthroughs_suppressed == 1


def test_extract_stops_before_a_decodable_thumb_literal_pool() -> None:
    """Do not execute literal bytes merely because Capstone can decode them."""

    project = project_module.load_project(
        Path("angr-binaries/tests/armhf/ld-linux-armhf.so.3")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x40DD51)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    continuation = nodes[0x40DDCB]

    assert continuation.size == 2
    assert [insn.mnemonic for insn in continuation.block.capstone.insns] == ["nop"]
    assert 0x40DDCD not in nodes


def test_extract_stops_at_an_unknown_thumb_literal_pool() -> None:
    """Treat a conditional VEX ``unknown`` load as literal-pool evidence."""

    project = project_module.load_project(
        Path("angr-binaries/tests/armel/lwip_udpecho_bm.elf")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x451)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    literal_predecessor = nodes[0x4A3]

    assert [insn.mnemonic for insn in literal_predecessor.block.capstone.insns] == [
        "nop"
    ]
    assert 0x4A5 not in nodes


def test_extract_reclaims_direct_targets_previously_seen_as_data() -> None:
    """Keep a direct branch target executable after a data-reference conflict."""

    project = project_module.load_project(
        Path("angr-binaries/tests/armel/Nucleo_read_hyperterminal.elf")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x80064B5)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    target = nodes[0x80066F5]

    assert [insn.mnemonic for insn in target.block.capstone.insns] == ["cmp.w", "beq"]


def test_extract_keeps_s390_execute_relative_instruction_templates() -> None:
    """Keep inline instructions fetched by S/390's execute-relative opcode."""

    project = project_module.load_project(
        Path("angr-binaries/tests/s390x/test-instr_s390x")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x80014A30)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}

    template = nodes[0x80014E08]
    assert [insn.mnemonic for insn in template.block.capstone.insns] == [
        "xc",
        "xc",
        "xc",
    ]


def test_extract_leader_is_not_requeued_after_recovery() -> None:
    """Keep a cycle from repeatedly scheduling an unchanged completed block."""

    session = object.__new__(builder_module._ExtractionSession)
    session.bounds = SimpleNamespace(addr=0x1000, end_addr=0x1100)
    session.leaders = {0x1000}
    session.rejected_leaders = set()
    session.blocks = {0x1000: SimpleNamespace(size=4)}
    session.static_targets = {}
    session.pending = []
    session.pending_addrs = set()
    session.stats = SimpleNamespace(block_redecodes=0)
    session.project = SimpleNamespace()
    session.data_regions = SimpleNamespace(contains=lambda *_args: False)

    session._add_leader(0x1000)

    assert session.pending == []


def test_extract_static_table_discovery_discards_stale_snapshot_plans(
    monkeypatch,
) -> None:
    """Rebuild table planning when a recovered target splits a later block."""

    session = object.__new__(builder_module._ExtractionSession)
    session.bounds = SimpleNamespace(addr=0x1000, end_addr=0x1200)
    session.leaders = {0x1000, 0x1100}
    session.rejected_leaders = set()
    session.pending = deque()
    session.pending_addrs = set()
    session.blocks = {
        0x1000: BlockSpec(0x1000, 0x20, (0x1000, 0x1008), "Ijk_Boring"),
        0x1100: BlockSpec(0x1100, 4, (0x1100,), "Ijk_Boring"),
    }
    session.static_targets = {}
    session.static_target_candidates = {}
    session.stats = ExtractedCFGStats()
    session.project = SimpleNamespace()
    session.data_regions = SimpleNamespace(
        contains=lambda *_args: False,
        claim_code=lambda *_args: None,
    )

    dispatcher = object()
    stale_node = object()
    monkeypatch.setattr(
        session,
        "_analysis_graph",
        lambda *_args: (nx.DiGraph(), {0x1100: dispatcher, 0x1000: stale_node}),
    )
    monkeypatch.setattr(
        builder_module,
        "plan_static_jump_table",
        lambda *_args, **_kwargs: (
            StaticJumpTablePlan(
                StaticJumpTable(
                    base_register_offset=None,
                    base_bits=32,
                    table_displacement=0,
                    index_register_offset=0,
                    index_bits=32,
                    entry_size=4,
                    endness="Iend_LE",
                    signed_entries=False,
                ),
                0x2000,
                (0,),
            ),
            None,
        ),
    )
    monkeypatch.setattr(
        builder_module,
        "_read_static_jump_table_targets",
        lambda *_args: (0x1008,),
    )
    monkeypatch.setattr(
        builder_module,
        "static_jump_target_rejection_reason",
        lambda *_args: None,
    )
    monkeypatch.setattr(session, "_decode_all_blocks", lambda: None)

    session._discover_static_jump_targets()

    assert 0x1000 not in session.blocks
    assert session.static_targets == {0x1100: (0x1008,)}
    assert session.stats.static_jump_plan_attempts == 2
    assert session.stats.static_jump_plans_invalidated == 1
    assert session.stats.static_jump_plans_resolved == 1
    assert session.stats.static_jump_table_entries_read == 2
    assert session.stats.static_jump_targets_accepted == 2


def test_extract_recovers_reconnecting_components_from_one_dispatcher() -> None:
    """Attach only reconnecting components behind one unresolved dispatcher."""

    project = project_module.load_project(
        Path("angr-binaries/tests/i386/bronze_ropchain")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x80A7DB0)

    assert cfg.extract_stats.sweep_runs == 1
    assert cfg.extract_stats.sweep_candidate_blocks > 100
    assert cfg.extract_stats.sweep_candidate_components > 0
    assert cfg.extract_stats.sweep_reconnecting_components > 0
    assert cfg.extract_stats.sweep_reconnecting_blocks > 0
    assert cfg.extract_stats.sweep_component_roots_attached > 0
    assert cfg.extract_stats.output_anomaly_count == 0
    assert cfg.extract_stats.output_anomalies_by_kind == {}

    dispatcher = next(
        node
        for node in cfg.graph.nodes()
        if node.addr == 0x80A7E0B and not node.is_simprocedure
    )
    unresolved = next(
        node
        for node in cfg.graph.nodes()
        if node.is_simprocedure and node.name == "UnresolvableJumpTarget"
    )
    successors = set(cfg.graph.successors(dispatcher))
    assert unresolved in successors
    assert len(successors) == cfg.extract_stats.sweep_component_roots_attached + 1
    assert not tuple(cfg.graph.successors(unresolved))


def test_extract_retains_static_targets_during_reconnecting_recovery() -> None:
    """Keep proven table closure when another dispatcher remains unresolved."""

    cases = (
        (
            "x86_64/rust_hello_world",
            0x4207F0,
            0x4208F7,
            (0x4208A8, 0x420A0C),
            41,
            True,
        ),
        (
            "x86_64/cvs",
            0x47F600,
            0x47FBD0,
            (0x47FD00, 0x47FE60),
            6,
            True,
        ),
        (
            "x86_64/1cbbf108f44c8f4babde546d26425ca5340dccf878d306b90eb0fbec2f83ab51",
            0x427320,
            0x42A0FA,
            (0x42A125, 0x42A171),
            5,
            False,
        ),
    )
    for (
        binary,
        function_addr,
        dispatcher_addr,
        expected_targets,
        successor_count,
        uses_sweep,
    ) in cases:
        project = project_module.load_project(Path("angr-binaries/tests") / binary)
        cfg = build_extracted_cfg(project, KnowledgeBase(project), function_addr)
        nodes = {
            node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure
        }
        successors = tuple(cfg.graph.successors(nodes[dispatcher_addr]))

        assert len(successors) == successor_count
        assert set(expected_targets) <= nodes.keys()
        assert all(not successor.is_simprocedure for successor in successors)
        assert not any(
            node.is_simprocedure and node.name == "UndecodableInstructionTarget"
            for node in cfg.graph.nodes()
        )
        assert bool(cfg.extract_stats.sweep_runs) is uses_sweep


def test_extract_retains_static_table_plan_after_leader_splits() -> None:
    """Reattach table plans and propagate a guarded selector across them."""

    project = project_module.load_project(
        Path(
            "angr-binaries/tests/x86_64/"
            "1cbbf108f44c8f4babde546d26425ca5340dccf878d306b90eb0fbec2f83ab51"
        )
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x423690)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    source = nodes[0x4239B8]

    assert {node.addr for node in cfg.graph.successors(source)} == {
        0x4239C1,
        0x423A00,
        0x423A50,
        0x423AA0,
        0x423B86,
    }
    later_source = nodes[0x42443D]
    assert {node.addr for node in cfg.graph.successors(later_source)} == {
        0x424456,
        0x4244A4,
        0x4244BF,
        0x4244DF,
        0x4244F5,
    }
    assert sum(len(node.instruction_addrs) for node in nodes.values()) == 969
    assert not any(node.is_simprocedure for node in cfg.graph.nodes())
    assert all(
        node.addr == 0x423690 or tuple(cfg.graph.predecessors(node))
        for node in nodes.values()
    )


def test_extract_does_not_reconnect_an_unbounded_table_dispatcher() -> None:
    """Keep unknown targets behind a recognized but unbounded jump table."""

    project = project_module.load_project(Path("angr-binaries/tests/x86_64/static"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x40D230)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    source = nodes[0x40D3DA]

    successors = tuple(cfg.graph.successors(source))
    assert len(successors) == 1
    assert successors[0].simprocedure_name == "UnresolvableJumpTarget"
    assert cfg.extract_stats.sweep_runs == 0
    assert cfg.extract_stats.sweep_dispatchers_ineligible == 1


def test_extract_recovers_memory_selector_table_candidates() -> None:
    """Recover bounded candidate rows without claiming an enum table is exact."""

    project = project_module.load_project(Path("angr-binaries/tests/x86_64/fmt-rust"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x4B1040)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    source = nodes[0x4B1040]
    successors = tuple(cfg.graph.successors(source))

    assert {0x4B1057, 0x4B1067, 0x4B1075} <= {
        node.addr for node in successors if not node.is_simprocedure
    }
    assert any(
        node.is_simprocedure and node.name == "UnresolvableJumpTarget"
        for node in successors
    )
    assert all(
        cfg.graph.get_edge_data(source, node)["unresolved_indirect"]
        for node in successors
    )
    assert cfg.extract_stats.static_jump_candidate_plans == 1
    assert cfg.extract_stats.static_jump_candidate_targets_accepted == 3
    assert cfg.extract_stats.sweep_runs == 1


def test_extract_skips_ambiguous_memory_selector_table_candidates() -> None:
    """LSDA landing pads must not suppress an existing dispatcher sweep."""

    project = project_module.load_project(Path("angr-binaries/tests/x86_64/fmt-rust"))
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x4F2AC0
    )
    cfg = session.build()

    assert cfg.extract_stats.static_jump_candidate_plans == 0
    assert cfg.extract_stats.sweep_reconnecting_blocks == 26
    assert cfg.extract_stats.exceptional_transfers_discovered == 149
    assert {0x4F2ED5, 0x4F31FB, 0x4F360F, 0x4F42E6} <= session.blocks.keys()
    covered = {
        insn.address
        for block in session.blocks.values()
        for insn in decode_raw_capstone_insns(project, block.addr, block.size)
    }
    assert len(covered) >= 1333


def test_extract_bounds_memory_selector_table_candidates() -> None:
    """Keep large unproven table frontiers on the ordinary recovery path."""

    cases = (
        (
            "x86_64/1cbbf108f44c8f4babde546d26425ca5340dccf878d306b90eb0fbec2f83ab51",
            0x431900,
        ),
        ("x86_64/rust_hello_world", 0x423360),
    )
    for binary, function_addr in cases:
        project = project_module.load_project(Path("angr-binaries/tests") / binary)
        cfg = build_extracted_cfg(project, KnowledgeBase(project), function_addr)

        assert cfg.extract_stats.static_jump_candidate_plans == 0


def test_extract_preserves_sweep_for_unbounded_rotated_table_index() -> None:
    """Keep the prior fallback when a new rotate matcher lacks a range proof."""

    project = project_module.load_project(Path("angr-binaries/tests/s390x/cfg_2"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x400840)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    dispatcher = nodes[0x4008AC]

    assert 0x4008C2 in nodes
    assert (
        len({addr for node in nodes.values() for addr in node.instruction_addrs}) == 59
    )
    assert len(tuple(cfg.graph.successors(dispatcher))) == 12
    assert cfg.extract_stats.static_jump_no_table_shape == 1
    assert cfg.extract_stats.sweep_runs == 1


def test_extract_does_not_reconnect_dynamic_memory_dispatch() -> None:
    """Keep vtable-style jumps behind their unresolved target leaf."""

    project = project_module.load_project(
        Path("angr-binaries/tests/i386/bronze_ropchain")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x8055B80)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    dispatcher = nodes[0x8055BFA]

    successors = tuple(cfg.graph.successors(dispatcher))
    assert len(successors) == 1
    assert successors[0].simprocedure_name == "UnresolvableJumpTarget"
    assert 0x8055C06 not in nodes
    assert cfg.extract_stats.static_jump_dynamic_memory_target == 1
    assert cfg.extract_stats.sweep_runs == 0
    assert cfg.extract_stats.sweep_dispatchers_ineligible == 1


def test_extract_keeps_static_memory_dispatch_eligible_for_recovery() -> None:
    """Do not mistake a statically based jump table for a dynamic vtable."""

    project = project_module.load_project(Path("angr-binaries/tests/x86_64/static"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x451F40)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    dispatcher = nodes[0x45279C]

    successors = tuple(cfg.graph.successors(dispatcher))
    assert len(successors) > 1
    assert any(
        node.simprocedure_name == "UnresolvableJumpTarget" for node in successors
    )
    assert cfg.extract_stats.static_jump_dynamic_memory_target == 0
    assert cfg.extract_stats.sweep_runs == 1


def test_extract_resolves_abi_preserved_register_tail_target() -> None:
    """Propagate a static AMD64 SysV callback target across its calls."""

    project = project_module.load_project(
        Path(
            "angr-binaries/tests/x86_64/"
            "1cbbf108f44c8f4babde546d26425ca5340dccf878d306b90eb0fbec2f83ab51"
        )
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x413630)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}

    first_call = nodes[0x413645]
    tail_jump = nodes[0x41367F]
    assert 0x41C030 in {node.addr for node in cfg.graph.successors(first_call)}
    assert {node.addr for node in cfg.graph.successors(tail_jump)} == {0x41C030}
    assert cfg.extract_stats.abi_static_target_analysis_runs == 1
    assert cfg.extract_stats.abi_static_call_targets_resolved == 8
    assert cfg.extract_stats.abi_static_jump_targets_resolved == 1
    assert cfg.extract_stats.abi_static_target_analysis_budget_exhausted == 0


def test_extract_rejects_swept_transparent_padding_root() -> None:
    """Do not attach a scanned alignment NOP as an indirect-jump candidate."""

    project = project_module.load_project(Path("angr-binaries/tests/x86_64/static"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x47A4D0)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}

    dispatcher = nodes[0x47A508]
    successors = tuple(cfg.graph.successors(dispatcher))
    assert len(successors) == 1
    assert successors[0].is_simprocedure
    assert successors[0].name == "UnresolvableJumpTarget"
    assert 0x47A513 not in nodes


def test_extract_resolves_a_guarded_x86_64_expression_table() -> None:
    """Resolve a zero-extended memory selector narrowed by its guard."""

    project = project_module.load_project(Path("angr-binaries/tests/x86_64/bomb"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x400F43)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    source = nodes[0x400F71]

    assert nodes[0x400F7C]
    successors = tuple(cfg.graph.successors(source))

    assert len(successors) == 8
    assert all(not successor.is_simprocedure for successor in successors)
    assert cfg.extract_stats.static_jump_plans_resolved == 1
    assert cfg.extract_stats.static_jump_unbounded_index == 0
    assert cfg.extract_stats.unresolved_indirect_targets == 0


def test_extract_resolves_guarded_post_decrement_byte_tables() -> None:
    """Use range guards on byte selectors after their index normalization."""

    project = project_module.load_project(
        Path(
            "angr-binaries/tests/x86_64/"
            "1cbbf108f44c8f4babde546d26425ca5340dccf878d306b90eb0fbec2f83ab51"
        )
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x427320)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}

    for source_addr in (0x4284F5, 0x4291FE):
        successors = tuple(cfg.graph.successors(nodes[source_addr]))
        assert len(successors) == 5
        assert all(
            successor.simprocedure_name != "UnresolvableJumpTarget"
            for successor in successors
        )

    assert cfg.extract_stats.static_jump_plans_resolved >= 2


def test_extract_resolves_16_bit_guarded_relative_tables() -> None:
    """Recover signed-relative tables guarded through x86's ``ax`` view."""

    project = project_module.load_project(
        Path(
            "angr-binaries/tests/x86_64/"
            "1cbbf108f44c8f4babde546d26425ca5340dccf878d306b90eb0fbec2f83ab51"
        )
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x421770)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}

    for source_addr, successor_count in ((0x4217AF, 44), (0x421864, 5)):
        successors = tuple(cfg.graph.successors(nodes[source_addr]))
        assert len(successors) == successor_count
        assert all(not successor.is_simprocedure for successor in successors)

    assert cfg.extract_stats.static_jump_plans_resolved >= 2


def test_extract_resolves_mips_pic_table_guarded_before_local_scale() -> None:
    """Use a range guard and table base carried by the predecessor delay slot."""

    project = project_module.load_project(Path("angr-binaries/tests/mipsel/darpa_ping"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x404120)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    successors = tuple(cfg.graph.successors(nodes[0x404158]))

    assert len(successors) == 22
    assert all(
        successor.simprocedure_name != "UnresolvableJumpTarget"
        for successor in successors
    )
    assert cfg.extract_stats.static_jump_plans_resolved == 1


def test_extract_resolves_normalized_mips_pic_table_selectors() -> None:
    """Use the guarded expression before scaling, not only a plain register."""

    project = project_module.load_project(
        Path("angr-binaries/tests/mipsel/mips_syscall_demo")
    )
    for function_addr, source_addr, expected_successors in (
        (0x4064F4, 0x406580, 8),
        (0x42C460, 0x42CE94, 6),
        (0x43C030, 0x43C0C4, 8),
    ):
        cfg = build_extracted_cfg(project, KnowledgeBase(project), function_addr)
        nodes = {
            node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure
        }
        successors = tuple(cfg.graph.successors(nodes[source_addr]))

        assert len(successors) == expected_successors
        assert all(
            successor.simprocedure_name != "UnresolvableJumpTarget"
            for successor in successors
        )
        assert cfg.extract_stats.static_jump_unbounded_index == 0


def test_extract_propagates_mips_pic_table_base_across_split_blocks() -> None:
    """Retain a must-constant table base after target-driven block splitting."""

    project = project_module.load_project(
        Path("angr-binaries/tests/mipsel/mips_syscall_demo")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x467380)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    successors = tuple(cfg.graph.successors(nodes[0x467548]))

    assert len(successors) == 6
    assert all(
        successor.simprocedure_name != "UnresolvableJumpTarget"
        for successor in successors
    )
    assert cfg.extract_stats.static_jump_plans_resolved == 1


def test_extract_resolves_mips_pic_table_guarded_across_delay_slot() -> None:
    """Carry a MIPS guard through its predecessor delay-slot register write."""

    project = project_module.load_project(
        Path("angr-binaries/tests/mipsel/mips_syscall_demo")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x4669AC)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    successors = tuple(cfg.graph.successors(nodes[0x466A68]))

    assert len(successors) == 27
    assert all(
        successor.simprocedure_name != "UnresolvableJumpTarget"
        for successor in successors
    )
    assert 0x466B34 in nodes
    assert cfg.extract_stats.static_jump_plans_resolved == 1
    assert cfg.extract_stats.static_jump_target_edges_added == 27
    assert cfg.extract_stats.sweep_runs == 0
    assert cfg.extract_stats.static_jump_unbounded_index == 0


def test_extract_resolves_a_guarded_x86_64_stack_selector() -> None:
    """Normalize VEX's narrowed zero-extended stack selector in a guard."""

    project = project_module.load_project(
        Path("angr-binaries/tests/x86_64/cfg_switches")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x40052D)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    successors = tuple(cfg.graph.successors(nodes[0x40053A]))

    assert len(successors) == 7
    assert all(not successor.is_simprocedure for successor in successors)
    assert cfg.extract_stats.static_jump_plans_resolved == 1
    assert cfg.extract_stats.unresolved_indirect_targets == 0


def test_extract_resolves_guarded_memory_relative_tables() -> None:
    """Use a predecessor range proof for a stack-loaded relative-table index."""

    cases = (
        (
            "x86_64/traffic_light_addsensor_x86-64/Traffic_Light_addsensor_x86-64.so",
            0x406AA1,
            0x406AB2,
            20,
        ),
        ("x86_64/multiarch_main_main.o", 0x403173, 0x403188, 21),
    )
    for binary, function_addr, source_addr, successor_count in cases:
        project = project_module.load_project(Path("angr-binaries/tests") / binary)
        cfg = build_extracted_cfg(project, KnowledgeBase(project), function_addr)
        nodes = {
            node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure
        }
        successors = tuple(cfg.graph.successors(nodes[source_addr]))

        assert len(successors) == successor_count
        assert all(not successor.is_simprocedure for successor in successors)
        assert cfg.extract_stats.static_jump_plans_resolved == 1
        assert cfg.extract_stats.unresolved_indirect_targets == 0


def test_extract_resolves_zero_extended_stack_selector_table() -> None:
    """Use a narrow stack guard after its value was stored as a full zext."""

    project = project_module.load_project(
        Path(
            "angr-binaries/tests/x86_64/"
            "1cbbf108f44c8f4babde546d26425ca5340dccf878d306b90eb0fbec2f83ab51"
        )
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x427320)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    successors = tuple(cfg.graph.successors(nodes[0x42AD91]))

    assert {successor.addr for successor in successors} == {
        0x42ADA8,
        0x42ADC4,
        0x42ADE7,
        0x42AE0F,
        0x42AE48,
        0x42BE0B,
    }
    assert all(not successor.is_simprocedure for successor in successors)
    assert cfg.extract_stats.static_jump_plans_resolved >= 1


def test_extract_resolves_s390x_rotated_relative_table_index() -> None:
    """Normalize s390x's masked rotate encoding of an eight-byte index."""

    project = project_module.load_project(Path("angr-binaries/tests/s390x/libc.so.6"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x48EE08)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    successors = tuple(cfg.graph.successors(nodes[0x48EE94]))

    assert len(successors) == 9
    assert all(not successor.is_simprocedure for successor in successors)
    assert cfg.extract_stats.static_jump_plans_resolved == 1
    assert cfg.extract_stats.unresolved_indirect_targets == 0


def test_extract_honors_s390x_rotated_table_predecessor_guard() -> None:
    """Do not read adjacent data past a guarded rotated-index table."""

    project = project_module.load_project(
        Path("angr-binaries/tests/s390x/test-instr_s390x")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x80067160)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    successors = tuple(cfg.graph.successors(nodes[0x80067188]))

    assert len(successors) == 8
    assert all(not successor.is_simprocedure for successor in successors)
    assert not {0x800674A4, 0x80067572, 0x800675B8} & nodes.keys()
    assert cfg.extract_stats.static_jump_plans_resolved == 1


def test_extract_resolves_a_guarded_sign_extended_byte_selector() -> None:
    """Use the unsigned guard on a sign-extended byte table selector."""

    project = project_module.load_project(
        Path("angr-binaries/tests/x86_64/dir_gcc_-O0")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x404D02)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    successors = tuple(cfg.graph.successors(nodes[0x404DF6]))

    assert len(successors) == 14
    assert all(not successor.is_simprocedure for successor in successors)
    assert cfg.extract_stats.static_jump_plans_resolved >= 3
    assert cfg.extract_stats.unresolved_indirect_targets == 0


def test_extract_resolves_constant_masked_jump_table_indices() -> None:
    """Recover concrete targets when VEX masks an otherwise unbounded index."""

    project = project_module.load_project(Path("angr-binaries/tests/x86_64/static"))
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x43DA00)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}

    for source_addr in (0x43DAC8, 0x43DB00):
        successors = tuple(cfg.graph.successors(nodes[source_addr]))
        assert len(successors) == 10
        assert all(not successor.is_simprocedure for successor in successors)

    assert cfg.extract_stats.static_jump_plans_resolved >= 2
    assert cfg.extract_stats.unresolved_indirect_targets == 0


def test_executable_sweep_closes_direct_targets_before_reporting_components() -> None:
    """Audit components retain the normal extractor's exact-leader invariant."""

    project = project_module.load_project(
        Path("angr-binaries/tests/i386/bronze_ropchain")
    )
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x80A7DB0
    )
    session._decode_all_blocks()
    session._discover_static_jump_targets()

    sweep = recover_executable_components(project, session.bounds, session.blocks)

    assert sweep.audit.candidate_blocks > 100
    assert sweep.audit.decode_failures == 0
    for block in sweep.blocks.values():
        for target in block.direct_targets:
            if session.bounds.addr <= target < session.bounds.end_addr:
                assert target in sweep.blocks


def test_extract_rejects_sweep_targets_inside_thumb_instructions() -> None:
    """Do not reconnect scanned data through an unsafe Thumb halfword."""

    project = project_module.load_project(
        Path("angr-binaries/tests/armel/lwip_udpecho_bm.elf")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x41DD)

    assert cfg.extract_stats.output_anomaly_count == 0


def test_extract_accepts_thumb_alternate_instruction_stream() -> None:
    """Decode a branch whose target enters a Thumb wide instruction's tail."""

    project = project_module.load_project(
        Path("angr-binaries/tests/armel/libc-2.31.so")
    )
    session = builder_module._ExtractionSession(
        project, KnowledgeBase(project), 0x46CB25
    )
    block = decode_bounded_block(project, session.bounds, 0x46CCDF, set())

    assert block is not None
    assert block.direct_targets == (0x46CC79,)


def test_extract_discards_thumb_tail_lift_fallthrough() -> None:
    """Keep a Capstone-proven Thumb branch independent of VEX IT state."""

    project = project_module.load_project(Path("angr-binaries/tests/armel/efm32gg.elf"))
    session = builder_module._ExtractionSession(project, KnowledgeBase(project), 0x641)
    block = decode_bounded_block(project, session.bounds, 0x701, set())

    assert block is not None
    assert block.direct_targets == (0x6E5,)
    assert block.fallthrough_addr is None


def test_extract_factors_x86_post_prefix_shared_tail() -> None:
    """Keep a LOCK and non-LOCK stream distinct until their shared tail."""

    project = project_module.load_project(
        Path("angr-binaries/tests/i386/bronze_ropchain")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x8049BA0)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}

    assert tuple(nodes[0x8049D62].instruction_addrs) == (0x8049D62,)
    assert tuple(nodes[0x8049D63].instruction_addrs) == (0x8049D63,)
    assert tuple(nodes[0x8049D6A].instruction_addrs) == (0x8049D6A,)
    assert set(cfg.graph.successors(nodes[0x8049D62])) >= {nodes[0x8049D6A]}
    assert set(cfg.graph.successors(nodes[0x8049D63])) >= {nodes[0x8049D6A]}
    owners = [
        node.addr for node in nodes.values() if 0x8049D6A in node.instruction_addrs
    ]
    assert owners == [0x8049D6A]


def test_extract_keeps_rebased_function_address_for_sub_name() -> None:
    """Do not let angr parse a synthetic sub-name as a linked address."""

    project = project_module.load_project(
        Path("angr-binaries/tests/i386/calling_convention_0.o")
    )
    cfg = build_extracted_cfg(project, KnowledgeBase(project), 0x400049)

    function = cfg.functions.get(0x400049, None)
    assert function is not None
    assert function.name == "sub_119320"


def test_reconnecting_component_cycle_gets_a_dispatcher_root() -> None:
    """Attach source strongly connected components with no zero-indegree node."""

    blocks = {
        0x1000: BlockSpec(0x1000, 1, (0x1000,), "Ijk_Boring", (0x1010,)),
        0x1010: BlockSpec(0x1010, 1, (0x1010,), "Ijk_Boring"),
        0x1020: BlockSpec(0x1020, 1, (0x1020,), "Ijk_Boring", (0x1030,)),
        0x1030: BlockSpec(
            0x1030,
            1,
            (0x1030,),
            "Ijk_Boring",
            (0x1020, 0x1040),
        ),
        0x1040: BlockSpec(0x1040, 1, (0x1040,), "Ijk_Ret"),
    }
    sweep = ExecutableSweep(
        blocks,
        frozenset({0x1000, 0x1010, 0x1040}),
        frozenset({0x1020, 0x1030}),
        ExecutableSweepAudit(2, 2, 1, 0, 0),
    )

    selected = select_reconnecting_components(
        None,
        sweep,
        {addr: blocks[addr] for addr in (0x1000, 0x1010, 0x1040)},
    )

    assert selected.roots == frozenset({0x1020})
    assert set(selected.blocks) == set(blocks)


def test_extract_keeps_original_graph_when_sweep_loses_dispatcher(monkeypatch) -> None:
    """Do not attach components from a dispatcher removed by speculative sweep."""

    dispatcher = BlockSpec(0x1000, 1, (0x1000,), "Ijk_Boring")
    selected_block = BlockSpec(0x1010, 1, (0x1010,), "Ijk_Ret")
    audit = ExecutableSweepAudit(1, 1, 1, 0, 0)
    sweep = ExecutableSweep(
        {0x1010: selected_block}, frozenset({0x1010}), frozenset(), audit
    )
    session = object.__new__(builder_module._ExtractionSession)
    session.project = SimpleNamespace()
    session.bounds = SimpleNamespace(addr=0x1000, end_addr=0x1020)
    session.func_addr = 0x1000
    session.blocks = {0x1000: dispatcher}
    session.static_targets = {}
    session.static_target_candidates = {}
    session.unresolved_dispatcher_reasons = {0x1000: "no_table_shape"}
    session.leaders = {0x1000}
    session.stats = ExtractedCFGStats()
    session.data_regions = SimpleNamespace(contains=lambda *_args: False)
    session.sweep_dispatcher_addr = None
    session.sweep_component_roots = frozenset()
    monkeypatch.setattr(
        builder_module, "recover_executable_components", lambda *_args, **_kwargs: sweep
    )
    monkeypatch.setattr(
        builder_module,
        "select_reconnecting_components",
        lambda *_, **__: ReconnectingComponents(
            {0x1010: selected_block}, frozenset({0x1010}), 1
        ),
    )

    session._recover_reconnecting_components()

    assert session.blocks == {0x1000: dispatcher}
    assert session.sweep_dispatcher_addr is None


class _Node:
    """Minimal hashable extracted-node stand-in for structural checks."""

    def __init__(self, addr: int, size: int, instruction_addrs: tuple[int, ...]):
        self.addr = addr
        self.size = size
        self.instruction_addrs = instruction_addrs
        self.is_simprocedure = False
        self.function_address = 0x1000


def test_extract_validation_rejects_targets_inside_other_blocks() -> None:
    """Require every direct target to become an exact block leader."""

    source = _Node(0x1000, 4, (0x1000,))
    covering = _Node(0x1002, 4, (0x1002,))
    graph = nx.DiGraph([(source, covering)])
    bounds = FunctionBounds(0x1000, 0x1010, 0x10, SimpleNamespace(name="f"))
    blocks = {
        0x1000: BlockSpec(0x1000, 4, (0x1000,), "Ijk_Boring", (0x1003,)),
        0x1002: BlockSpec(0x1002, 4, (0x1002,), "Ijk_Ret"),
    }

    anomalies = find_extracted_cfg_anomalies(graph, bounds, 0x1000, blocks)

    assert {anomaly.kind for anomaly in anomalies} == {
        "overlapping_blocks",
        "missing_direct_edge",
        "target_inside_block",
    }


def test_extract_validation_checks_block_spec_coverage_and_bounds() -> None:
    """Require normal nodes to exactly represent their recovered block spec."""

    node = _Node(0x1000, 4, (0x1000,))
    graph = nx.DiGraph()
    graph.add_node(node)
    bounds = FunctionBounds(0x1000, 0x1003, 3, SimpleNamespace(name="f"))
    blocks = {0x1000: BlockSpec(0x1000, 3, (0x1000, 0x1001), "Ijk_Ret")}

    anomalies = find_extracted_cfg_anomalies(graph, bounds, 0x1000, blocks)

    assert {anomaly.kind for anomaly in anomalies} == {
        "block_out_of_bounds",
        "block_size_mismatch",
        "instruction_coverage_mismatch",
    }


def test_extract_validation_checks_direct_and_fallthrough_jumpkinds() -> None:
    """Require materialized call and fake-return edges to retain their semantics."""

    source = _Node(0x1000, 4, (0x1000,))
    fallthrough = _Node(0x1004, 4, (0x1004,))
    callee = _Node(0x1010, 4, (0x1010,))
    graph = nx.DiGraph()
    graph.add_edge(source, fallthrough, jumpkind="Ijk_Boring")
    graph.add_edge(source, callee, jumpkind="Ijk_Boring")
    bounds = FunctionBounds(0x1000, 0x1020, 0x20, SimpleNamespace(name="f"))
    blocks = {
        0x1000: BlockSpec(
            0x1000,
            4,
            (0x1000,),
            "Ijk_Call",
            direct_targets=(0x1010,),
            fallthrough_addr=0x1004,
        ),
        0x1004: BlockSpec(0x1004, 4, (0x1004,), "Ijk_Ret"),
        0x1010: BlockSpec(0x1010, 4, (0x1010,), "Ijk_Ret"),
    }

    anomalies = find_extracted_cfg_anomalies(graph, bounds, 0x1000, blocks)

    assert {anomaly.kind for anomaly in anomalies} == {
        "direct_edge_jumpkind_mismatch",
        "fallthrough_edge_jumpkind_mismatch",
    }
