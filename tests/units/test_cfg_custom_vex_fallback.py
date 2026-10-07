"""Tests for VEX-only linear decoding in the independent CFG builder."""

from __future__ import annotations

from pathlib import Path

from angr import KnowledgeBase

from bingraph.cfg.decode import decode_bounded_block
from bingraph.cfg import build_custom_cfg
from bingraph.cfg import builder as builder_module
from bingraph.core import project as project_module
from bingraph.core.render import render_cfg


def test_custom_recovers_mips_fpu_compare_missing_from_capstone() -> None:
    """Use VEX for one linear MIPS FPU compare Capstone cannot decode."""

    project = project_module.load_project(Path("angr-binaries/tests/mips/dir"))
    bounds = builder_module._BuildSession(
        project, KnowledgeBase(project), 0x40F6F4
    ).bounds

    without_fallback = decode_bounded_block(project, bounds, 0x40F86C, set())
    with_fallback = decode_bounded_block(
        project,
        bounds,
        0x40F86C,
        set(),
        allow_vex_linear_fallback=True,
    )

    assert without_fallback is not None
    assert without_fallback.instruction_addrs == (0x40F86C,)
    assert with_fallback is not None
    assert 0x40F870 in with_fallback.instruction_addrs
    assert 0x40F874 in with_fallback.instruction_addrs
    assert 0x40F878 in with_fallback.instruction_addrs
    assert with_fallback.direct_targets == (0x40F890,)

    cfg = build_custom_cfg(project, KnowledgeBase(project), 0x40F6F4)

    assert cfg.custom_stats.vex_linear_fallbacks >= 1
    assert any(
        node.addr <= 0x40F870 < node.addr + node.size
        for node in cfg.graph.nodes()
        if not node.is_simprocedure
    )

    render_cfg.cache_clear()
    rendered = render_cfg(
        project,
        0x40F6F4,
        dfs_rank=False,
        comments=True,
        cfg_mode="custom",
        cfg_exits="jump",
        format="raw",
    )
    assert "0x0040f870" in rendered
    assert ".word" in rendered
    assert "0x4600113e" in rendered
    assert "VEX linear decode" in rendered
    assert "[VEX]" not in rendered
    assert "0x0040f874" in rendered

    without_comments = render_cfg(
        project,
        0x40F6F4,
        dfs_rank=False,
        comments=False,
        cfg_mode="custom",
        cfg_exits="jump",
        format="raw",
    )
    assert "VEX linear decode" not in without_comments
