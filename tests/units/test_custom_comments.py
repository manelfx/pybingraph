"""Reference annotations must enrich custom CFGs without reconstructing them."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch

from angr import Project
from pyvex.data_ref import DataRef
import pytest

from bingraph.core.annotators import CommentsDataRef
from bingraph.core.comments import collect_custom_comments
from bingraph.core.project import get_cfg
from bingraph.core.render import render_cfg
from bingraph.core.vis import Node


@pytest.fixture
def switch_cfg():
    """Use a fresh project so previous analyses cannot supply reference metadata."""

    project = Project("angr-binaries/tests/x86_64/switch", auto_load_libs=False)
    return project, get_cfg(project, 0x400544, "custom")


def _comments(cfg):
    """Compare instruction addresses rather than mode-dependent block splits."""

    comments = {}
    for obj in cfg.graph.nodes:
        if obj.is_simprocedure or obj.is_syscall:
            continue
        comments.update(
            {
                addr: text
                for addr, text in CommentsDataRef()
                .get_comments_by_addr(Node(obj, cfg.graph))
                .items()
                if addr in obj.instruction_addrs
            }
        )
    return comments


def test_switch_comments_are_collected_without_cfgfast(switch_cfg):
    """Restore strings and the pointer-table note without changing CFG facts."""

    project, cfg = switch_cfg
    nodes = list(cfg.graph.nodes)
    edges = [
        (src, dst, deepcopy(data)) for src, dst, data in cfg.graph.edges(data=True)
    ]
    stats = cfg.custom_stats.as_dict()
    summary = cfg.custom_summary.as_dict()
    assert not _comments(cfg)

    with patch(
        "angr.analyses.cfg.cfg_fast.CFGFast.__init__", side_effect=AssertionError
    ):
        collect_custom_comments(cfg)

    actual = _comments(cfg)
    assert len(actual) == 42
    assert actual[0x400580] == ['"0 is 0" ']
    assert actual[0x400576] == ["ptr @ 0x400a20 "]
    assert list(cfg.graph.nodes) == nodes
    assert list(cfg.graph.edges(data=True)) == edges
    assert cfg.custom_stats.as_dict() == stats
    assert cfg.custom_summary.as_dict() == summary
    assert not project.kb.xrefs.get_xrefs_by_ins_addr(0x400580)


def test_comment_collection_is_lazy_cached_and_does_not_change_baseline(switch_cfg):
    """Disabled renders avoid collection; later enabled renders reuse metadata."""

    project, cfg = switch_cfg
    with patch(
        "bingraph.core.render.collect_custom_comments", side_effect=AssertionError
    ):
        before = render_cfg(project, 0x400544, False, False, "custom", "jump", "raw")
    assert not cfg._comments_collected
    enabled = render_cfg(project, 0x400544, False, True, "custom", "jump", "raw")
    assert cfg._comments_collected
    assert ' ; "0 is 0" ' in enabled
    with patch.object(type(cfg.model), "_guess_data_type", side_effect=AssertionError):
        collect_custom_comments(cfg)
    render_cfg.cache_clear()
    after = render_cfg(project, 0x400544, False, False, "custom", "jump", "raw")
    assert after == before


def test_zero_based_thumb_constants_do_not_become_header_string_comments():
    """Scalar constants in division must not produce references into ELF headers."""

    project = Project(
        "angr-binaries/tests/armhf/float_int_conversion.elf", auto_load_libs=False
    )
    cfg = get_cfg(project, 0xEF19, "custom")
    collect_custom_comments(cfg)
    comments = _comments(cfg)
    assert comments == {0xEF3F: ["code reference @ 0xef50 "]}


def test_comment_collection_skips_a_failed_lift_and_collects_later_blocks(switch_cfg):
    """A presentation-only decoder failure must not abort the graph render."""

    project, cfg = switch_cfg
    lift = project.factory.block

    def fail_entry(addr, *args, **kwargs):
        if addr == 0x400544:
            raise ValueError("unsupported instruction")
        return lift(addr, *args, **kwargs)

    with patch.object(project.factory, "block", side_effect=fail_entry):
        collect_custom_comments(cfg)
    assert _comments(cfg)[0x400580] == ['"0 is 0" ']


def test_executable_literal_pool_is_not_classified_as_decoded_code(switch_cfg):
    """Data within executable sections is eligible when no CFG block covers it."""

    project, cfg = switch_cfg
    find_section = project.loader.find_section_containing

    def executable_region(addr):
        if addr == 0x4008C0:
            region = find_section(addr)
            return SimpleNamespace(
                is_readable=True,
                is_executable=True,
                vaddr=region.vaddr,
                memsize=region.memsize,
            )
        return find_section(addr)

    with patch.object(
        project.loader, "find_section_containing", side_effect=executable_region
    ):
        collect_custom_comments(cfg)
    assert cfg.model.memory_data[0x4008C0].sort == "string"
    assert _comments(cfg)[0x400580] == ['"0 is 0" ']


@pytest.mark.parametrize(
    ("data_type", "sort", "size"),
    [(0x9001, "integer", 2), (0x9003, "integer", 8), (0x9002, "fp", 4)],
)
def test_typed_access_upgrades_an_earlier_untyped_hint(
    switch_cfg, data_type, sort, size
):
    """Retain native load/store/float types and widths regardless of hint order."""

    project, cfg = switch_cfg
    address = project.loader.main_object.sections_map[".bss"].vaddr
    lift = project.factory.block

    def typed_entry(addr, *args, **kwargs):
        if addr != 0x400544:
            return lift(addr, *args, **kwargs)
        return SimpleNamespace(
            vex=SimpleNamespace(
                statements=[],
                data_refs=[
                    DataRef(address, 0, 0x9000, 0, addr),
                    DataRef(address, size, data_type, 1, addr),
                ],
            )
        )

    with patch.object(project.factory, "block", side_effect=typed_entry):
        collect_custom_comments(cfg)
    md = cfg.model.memory_data[address]
    assert md.sort == sort
    assert md.size == size


def test_flag_update_constant_is_not_a_pointer_in_a_zero_based_section():
    """A mapped scalar emitted by VEX bookkeeping must not become a comment."""

    project = Project(
        "angr-binaries/tests/armel/RTOSDemo.axf.issue_685", auto_load_libs=False
    )
    cfg = get_cfg(project, 0x1555, "custom")
    collect_custom_comments(cfg)
    assert 0x158D not in _comments(cfg)
    assert all(ref.ins_addr != 0x158D for ref in cfg.kb.xrefs.get_xrefs_by_dst(5))


def test_string_view_is_not_truncated_by_a_reference_to_its_suffix():
    """Independent references must not split 'Autotalent' into 'Autotale'."""

    project = Project(
        "angr-binaries/tests/riscv/autotalent-autotalent.so", auto_load_libs=False
    )
    cfg = get_cfg(project, 0x403C50, "custom")
    before = render_cfg(project, 0x403C50, False, False, "custom", "jump", "raw")
    collect_custom_comments(cfg)
    assert _comments(cfg)[0x403C96] == ['"Autotalent" ']
    assert cfg.model.memory_data[0x405048].content == b"nt"
    render_cfg.cache_clear()
    assert (
        render_cfg(project, 0x403C50, False, False, "custom", "jump", "raw") == before
    )


@pytest.mark.parametrize("size", [1, 2, 4, 8])
def test_string_classification_is_independent_of_access_width(switch_cfg, size):
    """Reading characters must not replace the full string with an integer note."""

    project, cfg = switch_cfg
    lift = project.factory.block

    def typed_entry(addr, *args, **kwargs):
        if addr != 0x400544:
            return lift(addr, *args, **kwargs)
        return SimpleNamespace(
            vex=SimpleNamespace(
                statements=[],
                data_refs=[
                    DataRef(0x4008C0, size, 0x9001, 0, addr),
                    DataRef(0x4008C0, 0, 0x9000, 1, addr),
                ],
            )
        )

    with patch.object(project.factory, "block", side_effect=typed_entry):
        collect_custom_comments(cfg)
    md = cfg.model.memory_data[0x4008C0]
    assert md.sort == "string"
    assert md.content == b"0 is 0"
    assert md.size == len(md.content) + 1
    assert _comments(cfg)[0x400544] == ['"0 is 0" ']


def test_relocatable_literal_pool_keeps_its_accepted_section_boundary():
    """CLE's warm section hit must survive later cache changes during guessing."""

    project = Project("angr-binaries/tests/armel/btrfs.ko", auto_load_libs=False)
    cfg = get_cfg(project, 0x400F50, "custom")
    # ELF relocatable sections overlap in this image. The full corpus primes
    # this cache with earlier functions; reproduce that without rendering them.
    section = project.loader.main_object.sections_map[".text"]
    project.loader.main_object._last_section = section
    with patch("bingraph.core.comments.logger.warning") as warning:
        collect_custom_comments(cfg)
    warning.assert_not_called()
    for addr in (0x401098, 0x40109C):
        md = cfg.model.memory_data[addr]
        assert md.max_size == section.vaddr + section.memsize - addr
        assert cfg.kb.xrefs.get_xrefs_by_dst(addr)


@pytest.mark.parametrize(
    ("binary", "function", "data_addr", "size", "instructions"),
    [
        (
            "x86_64/g_game.o",
            0x400040,
            0x4035EC,
            4,
            (
                0x400887,
                0x400892,
                0x40089D,
                0x4008AA,
                0x4008B5,
                0x4008C0,
                0x4008CB,
                0x4008D8,
            ),
        ),
        ("x86_64/calc", 0x401FB3, 0x4054C0, 4, (0x4023A1,)),
        ("x86_64/static", 0x4009E0, 0x4001E0, 4, (0x400AD0,)),
        ("x86_64/static", 0x459E50, 0x4B2DA0, 16, (0x45A33A, 0x45A710)),
        ("x86_64/rust_hello_world", 0x4465D0, 0x44E5C0, 16, (0x446986, 0x446A85)),
    ],
)
def test_printable_numeric_accesses_do_not_become_strings(
    binary, function, data_addr, size, instructions
):
    """Keep numeric/vector access metadata even when the bytes resemble text."""

    project = Project(f"angr-binaries/tests/{binary}", auto_load_libs=False)
    cfg = get_cfg(project, function, "custom")
    collect_custom_comments(cfg)
    md = cfg.model.memory_data[data_addr]
    assert md.sort == "integer"
    assert md.size == size
    assert md.content is None
    comments = _comments(cfg)
    for insn in instructions:
        assert comments[insn] == [f"int @ {data_addr:#x} "]
