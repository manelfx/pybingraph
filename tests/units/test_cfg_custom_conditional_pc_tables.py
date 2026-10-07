"""Regression coverage for VEX-guarded memory loads used as the next PC."""

from __future__ import annotations

from pathlib import Path

from angr import KnowledgeBase

from bingraph.cfg.decode import decode_bounded_block
from bingraph.cfg import build_custom_cfg
from bingraph.cfg import builder as builder_module
from bingraph.core.project import load_project


def test_custom_recovers_a_guarded_pc_load_table() -> None:
    """Keep both paths of ``ldrls pc, [pc, r3, lsl #2]`` in the CFG."""

    project = load_project(Path("angr-binaries/tests/armel/btrfs.ko"))
    bounds = builder_module._BuildSession(
        project, KnowledgeBase(project), 0x446F2C
    ).bounds

    block = decode_bounded_block(
        project,
        bounds,
        0x446F58,
        set(),
        split_unclassified_indirect_vex_transfers=True,
    )

    assert block is not None
    assert block.instruction_addrs == (0x446F58, 0x446F5C, 0x446F60)
    assert block.fallthrough_addr == 0x446F64

    cfg = build_custom_cfg(project, KnowledgeBase(project), 0x446F2C)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    source = nodes[0x446F58]
    successors = {node.addr for node in cfg.graph.successors(source)}

    assert successors == {0x446F64, 0x446F78, 0x4471B0}
    assert cfg.custom_stats.static_jump_plans_resolved == 1
    assert cfg.custom_stats.exact_jump_proofs_by_flavor == {"conditional_pc": 1}
    assert cfg.custom_stats.static_jump_target_edges_added == 2
    assert not any(
        node.simprocedure_name == "UnresolvableJumpTarget"
        for node in cfg.graph.nodes()
        if node.is_simprocedure
    )


def test_custom_recovers_a_guarded_arithmetic_pc_dispatch() -> None:
    """Recover a finite ``addls pc, pc, r2, lsl #2`` target range."""

    project = load_project(Path("angr-binaries/tests/armel/libc.so.6"))
    cfg = build_custom_cfg(project, KnowledgeBase(project), 0x47E114)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    source = nodes[0x47E114]
    successors = {node.addr for node in cfg.graph.successors(source)}

    assert successors == {
        0x47E140,
        0x47E144,
        0x47E148,
        0x47E14C,
        0x47E150,
        0x47E154,
        0x47E158,
        0x47E15C,
        0x47E160,
    }
    assert cfg.custom_stats.conditional_pc_dispatches_resolved == 1
    assert cfg.custom_stats.conditional_pc_targets_recovered == 8


def test_custom_keeps_an_unbounded_conditional_pc_branch_explicit() -> None:
    """A known false path must not hide the unresolved taken branch."""

    project = load_project(Path("angr-binaries/tests/armel/test_division"))
    session = builder_module._BuildSession(project, KnowledgeBase(project), 0x8670)
    session._decode_all_blocks()
    session._materialize_edges()
    nodes = {node.addr: node for node in session.graph if not node.is_simprocedure}
    source = nodes[0x86A0]
    successors = tuple(session.graph.successors(source))

    assert len(successors) == 2
    assert nodes[0x86BC] in successors
    unknown = next(node for node in successors if node.is_simprocedure)
    assert unknown.simprocedure_name == "UnresolvableJumpTarget"
    assert session.graph[source][unknown]["unresolved_indirect"]
    assert session.stats.unresolved_indirect_targets == 1


def test_custom_recovers_an_unconditional_static_pc_load_table() -> None:
    """Stop at and resolve a VEX-only absolute table load into the PC."""

    project = load_project(
        Path("angr-binaries/tests/armel/i2c_master_read-nucleol152re.elf")
    )
    bounds = builder_module._BuildSession(
        project, KnowledgeBase(project), 0x800B401
    ).bounds

    block = decode_bounded_block(
        project,
        bounds,
        0x800BB47,
        set(),
        split_unclassified_indirect_vex_transfers=True,
    )

    assert block is not None
    assert block.instruction_addrs == (0x800BB47, 0x800BB49)
    assert block.fallthrough_addr is None

    cfg = build_custom_cfg(project, KnowledgeBase(project), 0x800B401)
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    source = nodes[0x800BB47]
    successors = {node.addr for node in cfg.graph.successors(source)}

    assert successors == {0x800B489, 0x800BC93, 0x800BCCB}
    assert cfg.custom_stats.static_jump_plans_resolved >= 1
    assert cfg.custom_stats.static_jump_target_edges_added >= 3
