"""Reference annotations must enrich custom CFGs without reconstructing them."""

from copy import deepcopy
import re
from types import SimpleNamespace
from unittest.mock import patch
from xml.etree import ElementTree

from angr import Project, load_shellcode
from angr.knowledge_plugins.cfg import CFGNode
from angr.knowledge_plugins.xrefs import XRefType
from archinfo import arch_from_id
from cle.backends.symbol import SymbolType
import networkx as nx
import pyvex
from pyvex.data_ref import DataRef
import pytest

from bingraph.core.annotators import CommentsDataRef
from bingraph.core.comments import (
    _is_scalar_reference,
    _reference_uses,
    collect_custom_comments,
)
from bingraph.core.contents import NodeAsm
from bingraph.core.labels import MAX_LABEL_LENGTH
from bingraph.core.outputs import COMMENT_COLUMN_GAP, DotOutput
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


@pytest.mark.parametrize("addr", [0x5, 0x400561, 0x100400561])
@pytest.mark.parametrize("vex_only", [False, True])
def test_instruction_addresses_are_compact_hex_without_changing_operands(
    addr, vex_only
):
    class RenderNode(CFGNode):
        pass

    project = load_shellcode(
        b"\x90\xb8\x50\x04\x40\x00\xc3", "AMD64", load_address=addr
    )
    obj = RenderNode(
        addr,
        7,
        SimpleNamespace(project=project, _iropt_level=0),
        block_id=addr,
        instruction_addrs=[addr, addr + 1, addr + 6],
    )
    if vex_only:
        obj.vex_linear_instruction_sizes = {addr: 1}
    node = Node(obj)
    NodeAsm().gen_render(node)
    rows = node.content["asm"]["data"]
    assert [row["addr"] for row in rows] == [
        {"content": f"{a:#x}:\t", "align": "LEFT"} for a in (addr, addr + 1, addr + 6)
    ]
    assert [row["_addr"] for row in rows] == [addr, addr + 1, addr + 6]
    assert rows[0]["mnemonic"]["content"] == (".word" if vex_only else "nop")
    assert rows[1]["operands"]["content"] == "eax, 0x400450"


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
    assert actual[0x400576] == ["ref 0x400a20 "]
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


@pytest.mark.parametrize(
    "comments", [[], ["one"], ["one", "two"], ["one", "two", "<three>"]]
)
def test_comment_lines_have_aligned_prefixes_and_explicit_column_spacing(
    monkeypatch, comments
):
    node = SimpleNamespace(obj=SimpleNamespace(is_simprocedure=False, is_syscall=False))
    row = {
        "mnemonic": {"content": "mov", "align": "LEFT"},
        "_ins": SimpleNamespace(address=0x400000),
    }
    monkeypatch.setattr(
        CommentsDataRef, "get_comments_by_addr", lambda self, node: {0x400000: comments}
    )
    CommentsDataRef().annotate_content(node, {"data": [row]})
    rendered = DotOutput(fname="unused").render_row(row, ["mnemonic", "comment"])
    spacer = f'<TD WIDTH="{COMMENT_COLUMN_GAP}"></TD>'
    assert rendered.count(spacer) == 1
    if not comments:
        assert "comment" not in row
        assert rendered == f'<TR><TD ALIGN="LEFT">mov</TD>{spacer}<TD></TD></TR>'
        return
    assert row["comment"]["content"] == "\n".join(" ; " + c for c in comments)
    assert rendered.count('<BR ALIGN="LEFT"/>') == (
        len(comments) if len(comments) > 1 else 0
    )
    assert rendered.count(" ; ") == len(comments)
    assert '<FONT COLOR="gray">' in rendered
    assert rendered.count('VALIGN="TOP"') == (2 if len(comments) > 1 else 0)
    assert "valign" not in row["mnemonic"]
    if len(comments) == 1:
        assert rendered == (
            '<TR><TD ALIGN="LEFT">mov</TD>'
            + spacer
            + '<TD ALIGN="LEFT"><FONT COLOR="gray"> ; one</FONT></TD></TR>'
        )
    if len(comments) == 3:
        assert "&#60;three&#62;" in rendered


def test_malloc_set_state_comments_render_on_separate_aligned_svg_lines():
    project = Project("angr-binaries/tests/x86_64/static", auto_load_libs=False)
    svg = render_cfg(project, 0x41F190, False, True, "custom", "jump", "svg")
    root = ElementTree.fromstring(svg)
    for node in root.findall(".//{*}g[@class='node']"):
        texts = node.findall("{*}text")
        for i, text in enumerate(texts):
            if (text.text or "").strip() != "0x41f621:":
                continue
            instruction = texts[i : i + 3]
            comments = texts[i + 3 : i + 5]
            assert [t.text.strip() for t in comments] == [
                "; malloc_check @ 0x41bf40",
                "; ptr slot __malloc_hook @ 0x6c9788",
            ]
            first_y, second_y = (float(t.attrib["y"]) for t in comments)
            assert comments[0].attrib["x"] == comments[1].attrib["x"]
            # XDOT exposes Graphviz's measured text width, unlike the SVG.
            xdot = render_cfg(project, 0x41F190, False, True, "custom", "jump", "xdot")
            operands = re.search(
                r"T ([\d.]+) [\d.]+ -1 ([\d.]+) \d+ -" + re.escape(instruction[2].text),
                xdot.replace("\\\n", ""),
            )
            assert operands is not None
            operand_end = float(operands[1]) + float(operands[2])
            assert float(comments[0].attrib["x"]) - operand_end >= COMMENT_COLUMN_GAP
            assert all(float(t.attrib["y"]) == first_y for t in instruction)
            assert second_y > first_y
            assert float(texts[i + 5].attrib["y"]) > second_y
            return
    pytest.fail("Expected malloc-hook store instruction missing from SVG")


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


@pytest.mark.parametrize(
    ("binary", "function", "rejected", "retained"),
    [
        (
            "armel/lwip_udpecho_bm.elf",
            0x5CB9,
            (0x5CB9, 0x5CBB, 0x5CBF, 0x5CC1, 0x5CC3, 0x5D41),
            None,
        ),
        ("armel/RTOSDemo.axf.issue_685", 0x9781, (0x9783,), 0x978B),
        ("mipsel/jumptable_0", 0x4080B0, (0x408100,), None),
    ],
)
def test_scalar_stack_and_link_values_are_not_absolute_data_references(
    binary, function, rejected, retained
):
    project = Project(f"angr-binaries/tests/{binary}", auto_load_libs=False)
    cfg = get_cfg(project, function, "custom")
    before = render_cfg(project, function, False, False, "custom", "jump", "raw")
    with patch(
        "angr.analyses.cfg.cfg_fast.CFGFast.__init__", side_effect=AssertionError
    ):
        collect_custom_comments(cfg)
    comments = _comments(cfg)
    assert not set(rejected).intersection(comments)
    if retained is not None:
        assert comments[retained]
    render_cfg.cache_clear()
    assert (
        render_cfg(project, function, False, False, "custom", "jump", "raw") == before
    )


def test_reference_validation_preserves_absolute_zero_and_scopes_address_uses():
    load = pyvex.IRStmt.WrTmp(
        0,
        pyvex.IRExpr.Load("Iend_LE", "Ity_I32", pyvex.IRExpr.Const(pyvex.const.U32(0))),
    )
    vex = SimpleNamespace(statements=[load])
    assert not _is_scalar_reference(
        DataRef(0, 4, 0x9001, 0, 0x1000), vex, set(), set(), {}
    )
    vex.statements = [
        pyvex.IRStmt.WrTmp(
            0,
            pyvex.IRExpr.Binop(
                "Iop_Add32",
                [pyvex.IRExpr.RdTmp(1), pyvex.IRExpr.Const(pyvex.const.U32(4))],
            ),
        )
    ]
    for stmt_idx in (0, 1):
        assert _is_scalar_reference(
            DataRef(4, 0, 0x9000, stmt_idx, 0x1000),
            SimpleNamespace(statements=vex.statements * 2),
            set(),
            set(),
            {0: {1}},
        ) == (stmt_idx != 0)


def test_failed_address_hint_classification_does_not_abort_annotations(switch_cfg):
    _, cfg = switch_cfg
    with patch.object(
        type(cfg.model), "_guess_data_type", side_effect=ValueError("bad data")
    ):
        collect_custom_comments(cfg)
    assert cfg._comments_collected


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
    """Shared string metadata survives a load-specific bounded byte preview."""

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
    expected = 'bytes @ 0x4008c0: "0 is" ' if size == 4 else '"0 is 0" '
    assert _comments(cfg)[0x400544] == [expected]


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
        expected = (
            "forwardmove+0x4 @ 0x4035ec "
            if binary == "x86_64/g_game.o"
            else 'bytes @ 0x4054c0: "(knN" '
            if binary == "x86_64/calc"
            else f"ref {data_addr:#x} "
        )
        assert comments[insn] == [expected]


@pytest.mark.parametrize(
    ("data", "width", "flags", "expected"),
    [
        (b"ib/debugNEIGHBOR", 8, {}, 'bytes @ 0x500000: "ib/debug"'),
        (b'abc"\\def', 8, {}, 'bytes @ 0x500000: "abc\\"\\\\def"'),
        (b"a" * 32, 32, {}, 'bytes @ 0x500000: "' + "a" * 32 + '"'),
        (b"ABC", 3, {}, "ref 0x500000"),
        (b"a" * 33, 33, {}, "ref 0x500000"),
        (b"V\0\0\0", 4, {}, "ref 0x500000"),
        (b"ABC\n", 4, {}, "ref 0x500000"),
        (b"ABC\xff", 4, {}, "ref 0x500000"),
        (b"ABCDEFGH", 8, {"is_writable": True}, "ref 0x500000"),
        (b"ABCDEFGH", 8, {"is_executable": True}, "ref 0x500000"),
        (b"ABCDEFGH", 8, {"is_readable": False}, "ref 0x500000"),
        (b"ABCDEFGH", 8, {"name": ".got.plt"}, "ref 0x500000"),
        (b"ABCDEFGH", 8, {"name": ".igot.plt"}, "ref 0x500000"),
        (b"ABCDEFGH", 8, {"memsize": 4}, "ref 0x500000"),
        (b"ABC", 8, {}, "ref 0x500000"),
    ],
)
def test_integer_byte_previews_are_bounded_printable_readonly_data(
    data, width, flags, expected
):
    region = SimpleNamespace(
        **{
            "name": ".rodata",
            "vaddr": 0x500000,
            "memsize": 64,
            "is_readable": True,
            "is_writable": False,
            "is_executable": False,
            **flags,
        }
    )
    loader = SimpleNamespace(
        find_symbol=lambda addr, fuzzy=False: None,
        find_object_containing=lambda addr: None,
        find_section_containing=lambda addr: region,
        memory=SimpleNamespace(load=lambda addr, size: data[:size]),
    )
    project = SimpleNamespace(kb=SimpleNamespace(labels={}), loader=loader)
    node = SimpleNamespace(project=project, kb=project.kb)
    md = SimpleNamespace(addr=0x500000, sort="integer", size=width, content=None)
    annotation = CommentsDataRef()
    assert annotation._format_memory_data_comment(node, md, is_read=True) == expected
    assert md.sort == "integer" and md.content is None
    # Address hints and stores are not load previews.
    assert annotation._format_memory_data_comment(node, md) == "ref 0x500000"
    md.sort = "pointer-array"
    assert (
        annotation._format_memory_data_comment(node, md, is_read=True) == "ref 0x500000"
    )


def test_gimli_packed_path_loads_show_bytes_without_guessing_string_boundaries():
    project = Project(
        "angr-binaries/tests/x86_64/"
        "1cbbf108f44c8f4babde546d26425ca5340dccf878d306b90eb0fbec2f83ab51",
        auto_load_libs=False,
    )
    cfg = get_cfg(project, 0x438E10, "custom")
    nodes, edges = list(cfg.graph.nodes), list(cfg.graph.edges(data=True))
    stats, summary = cfg.custom_stats.as_dict(), cfg.custom_summary.as_dict()
    with patch(
        "angr.analyses.cfg.cfg_fast.CFGFast.__init__", side_effect=AssertionError
    ):
        collect_custom_comments(cfg)
    comments = _comments(cfg)
    assert comments[0x43D067] == ['bytes @ 0x45ef39: "ib/debug" ']
    assert comments[0x43D073] == ['bytes @ 0x45ef33: "/usr/lib" ']
    md = cfg.model.memory_data[0x45EF39]
    assert md.sort == "integer" and md.size == 8 and md.content is None
    # Another instruction takes the address of the complete pool string.
    # Keep that shared classification, but bound this load's preview to 8 bytes.
    assert cfg.model.memory_data[0x45EF33].sort == "string"
    for instruction, address in ((0x43D067, 0x45EF39), (0x43D073, 0x45EF33)):
        ref = next(
            ref
            for ref in cfg.kb.xrefs.get_xrefs_by_ins_addr(instruction)
            if ref.dst == address
        )
        assert ref.memory_data.sort == "integer"
        assert ref.memory_data.size == 8 and ref.memory_data.content is None
    assert list(cfg.graph.nodes) == nodes
    assert list(cfg.graph.edges(data=True)) == edges
    assert cfg.custom_stats.as_dict() == stats
    assert cfg.custom_summary.as_dict() == summary


def test_integer_previews_use_each_load_width_not_the_shared_object(switch_cfg):
    project, cfg = switch_cfg
    node = next(n for n in cfg.graph.nodes if n.addr == 0x400544)
    cfg.graph = nx.DiGraph()
    cfg.graph.add_node(node)
    node.instruction_addrs = [0x400544, 0x40054A]
    project.loader.memory.store(0x4008C0, b"ABCDEFGH\0")
    # Two loads at the same address have different widths; no opcode matcher
    # participates in collection or rendering of their reference metadata.
    encoding = b"\x8b\x05\x76\x03\0\0\x48\x8b\x05\x6f\x03\0\0\xc3"
    lift = project.factory.block
    with patch.object(
        project.factory,
        "block",
        side_effect=lambda addr, **kwargs: lift(addr, byte_string=encoding, **kwargs),
    ):
        collect_custom_comments(cfg)
    comments = _comments(cfg)
    assert comments[0x400544] == ['bytes @ 0x4008c0: "ABCD" ']
    assert comments[0x40054A] == ['bytes @ 0x4008c0: "ABCDEFGH" ']
    assert {
        (ref.ins_addr, ref.memory_data.size)
        for ref in cfg.kb.xrefs.get_xrefs_by_dst(0x4008C0)
        if ref.type == XRefType.Read
    } == {(0x400544, 4), (0x40054A, 8)}


@pytest.mark.parametrize(
    ("binary", "function", "instruction", "address", "content"),
    [
        ("calc", 0x401FB3, 0x402206, 0x405280, b"0123456789ABCDEF"),
        ("calc", 0x402A87, 0x402EB0, 0x405280, b"0123456789ABCDEF"),
        ("bomb", 0x401062, 0x401099, 0x4024B0, b"maduiersnfotvbyl"),
        ("dir_gcc_-O0", 0x408F4E, 0x408FB2, 0x41A198, b"?pcdb-lswd"),
        ("static", 0x46CDA0, 0x46CDFC, 0x4B44D6, b"ORIGIN"),
    ],
)
def test_indexed_byte_tables_keep_string_comments_with_object_boundaries(
    binary, function, instruction, address, content
):
    project = Project(f"angr-binaries/tests/x86_64/{binary}", auto_load_libs=False)
    cfg = get_cfg(project, function, "custom")
    before = render_cfg(project, function, False, False, "custom", "jump", "raw")
    with patch(
        "angr.analyses.cfg.cfg_fast.CFGFast.__init__", side_effect=AssertionError
    ):
        collect_custom_comments(cfg)
    md = cfg.model.memory_data[address]
    assert md.sort == "string"
    assert md.content == content
    assert f'"{content.decode()}"' in _comments(cfg)[instruction][0]
    render_cfg.cache_clear()
    assert (
        render_cfg(project, function, False, False, "custom", "jump", "raw") == before
    )


@pytest.mark.parametrize("function", [0x457400, 0x488960, 0x48BB80, 0x48E9F0])
def test_word_strided_numeric_table_address_does_not_imply_string(function):
    """A pointer to the integer 10 must not turn it into a newline string."""

    project = Project("angr-binaries/tests/x86_64/static", auto_load_libs=False)
    cfg = get_cfg(project, function, "custom")
    collect_custom_comments(cfg)
    md = cfg.model.memory_data[0x4B7688]
    assert md.sort not in ("string", "unicode")
    assert md.content is None
    references = [
        comment
        for comments in _comments(cfg).values()
        for comment in comments
        if "__tens+0x8 @ 0x4b7688" in comment
    ]
    assert references
    assert all(comment == "__tens+0x8 @ 0x4b7688 " for comment in references)


@pytest.mark.parametrize(
    "encoding",
    [
        "8a85c0084000",  # mov al, [rbp + 0x4008c0]
        "8a8424c0084000",  # mov al, [rsp + 0x4008c0]
        "648a80c0084000",  # mov al, fs:[rax + 0x4008c0]
        "8a05c0084000",  # mov al, [rip + 0x4008c0]
    ],
)
def test_stack_segment_and_rip_displacements_are_not_absolute_table_bases(
    switch_cfg, encoding
):
    project, cfg = switch_cfg
    lift = project.factory.block

    def displaced_entry(addr, *args, **kwargs):
        if addr != 0x400544:
            return lift(addr, *args, **kwargs)
        return lift(
            addr,
            byte_string=bytes.fromhex(encoding),
            collect_data_refs=True,
            opt_level=1,
            cross_insn_opt=False,
        )

    with patch.object(project.factory, "block", side_effect=displaced_entry):
        collect_custom_comments(cfg)
    assert 0x400544 not in _comments(cfg)


@pytest.mark.parametrize(
    "architecture", ["AMD64", "X86", "ARMEL", "MIPS32", "PPC64", "S390X"]
)
@pytest.mark.parametrize("scaled", [False, True])
def test_vex_reference_uses_track_address_widths_without_instruction_patterns(
    architecture, scaled
):
    arch = arch_from_id(architecture)
    offsets = {r.vex_offset for r in arch.register_list if r.general_purpose} - {
        arch.sp_offset,
        arch.bp_offset,
        arch.ip_offset,
    }
    offset = min(offsets)
    const = pyvex.const.U64 if arch.bits == 64 else pyvex.const.U32
    word_type = f"Ity_I{arch.bits}"
    statements = [pyvex.IRStmt.WrTmp(0, pyvex.IRExpr.Get(offset, word_type))]
    index = pyvex.IRExpr.RdTmp(0)
    if scaled:
        statements.append(
            pyvex.IRStmt.WrTmp(
                1,
                pyvex.IRExpr.Binop(
                    f"Iop_Shl{arch.bits}",
                    [index, pyvex.IRExpr.Const(pyvex.const.U8(3))],
                ),
            )
        )
        index = pyvex.IRExpr.RdTmp(1)
    source = len(statements)
    statements.extend(
        [
            pyvex.IRStmt.WrTmp(
                2,
                pyvex.IRExpr.Binop(
                    f"Iop_Add{arch.bits}", [index, pyvex.IRExpr.Const(const(0x5000))]
                ),
            ),
            pyvex.IRStmt.WrTmp(
                3,
                pyvex.IRExpr.Load(arch.memory_endness, "Ity_I8", pyvex.IRExpr.RdTmp(2)),
            ),
            pyvex.IRStmt.WrTmp(
                4,
                pyvex.IRExpr.Load(
                    arch.memory_endness, word_type, pyvex.IRExpr.RdTmp(2)
                ),
            ),
            pyvex.IRStmt.Put(pyvex.IRExpr.RdTmp(2), offset),
        ]
    )
    assert _reference_uses(SimpleNamespace(statements=statements), offsets) == {
        source: {0} if scaled else {1, arch.bytes}
    }


@pytest.mark.parametrize("role", ["sp_offset", "bp_offset", "ip_offset"])
def test_vex_role_based_address_rejection_without_register_names(role):
    arch = arch_from_id("AMD64")
    offsets = {r.vex_offset for r in arch.register_list if r.general_purpose} - {
        arch.sp_offset,
        arch.bp_offset,
        arch.ip_offset,
    }
    statements = [
        pyvex.IRStmt.WrTmp(0, pyvex.IRExpr.Get(getattr(arch, role), "Ity_I64")),
        pyvex.IRStmt.WrTmp(
            1,
            pyvex.IRExpr.Binop(
                "Iop_Add64",
                [pyvex.IRExpr.RdTmp(0), pyvex.IRExpr.Const(pyvex.const.U64(0x5000))],
            ),
        ),
        pyvex.IRStmt.WrTmp(
            2, pyvex.IRExpr.Load("Iend_LE", "Ity_I8", pyvex.IRExpr.RdTmp(1))
        ),
    ]
    assert _reference_uses(SimpleNamespace(statements=statements), offsets) == {}


def test_vex_reference_provenance_is_bounded_and_does_not_follow_cycles():
    statements = [pyvex.IRStmt.WrTmp(0, pyvex.IRExpr.Get(16, "Ity_I64"))]
    for tmp in range(1, 40):
        statements.append(
            pyvex.IRStmt.WrTmp(
                tmp,
                pyvex.IRExpr.Binop(
                    "Iop_Add64",
                    [
                        pyvex.IRExpr.RdTmp(tmp - 1),
                        pyvex.IRExpr.Const(pyvex.const.U64(tmp)),
                    ],
                ),
            )
        )
    statements.append(
        pyvex.IRStmt.WrTmp(
            40, pyvex.IRExpr.Load("Iend_LE", "Ity_I8", pyvex.IRExpr.RdTmp(39))
        )
    )
    assert _reference_uses(SimpleNamespace(statements=statements), {16}) == {}
    assert (
        _reference_uses(
            SimpleNamespace(
                statements=[
                    pyvex.IRStmt.WrTmp(0, pyvex.IRExpr.RdTmp(0)),
                    pyvex.IRStmt.WrTmp(
                        1, pyvex.IRExpr.Load("Iend_LE", "Ity_I8", pyvex.IRExpr.RdTmp(0))
                    ),
                ]
            ),
            {16},
        )
        == {}
    )


@pytest.mark.parametrize("alias_write", [b"", b"\xb0\x00", b"\xb4\x00"])
def test_vex_register_provenance_invalidates_aliased_pointer_writes(alias_write):
    """AL/AH writes must not retain the earlier RAX address annotation."""

    project = load_shellcode(
        b"\x48\xb8\x00\x50\x00\x00\x00\x00\x00\x00" + alias_write + b"\x8a\x08\xc3",
        "AMD64",
        load_address=0x400000,
    )
    vex = project.factory.block(
        0x400000, opt_level=1, cross_insn_opt=False, collect_data_refs=True
    ).vex
    offsets = {
        r.vex_offset for r in project.arch.register_list if r.general_purpose
    } - {project.arch.sp_offset, project.arch.bp_offset, project.arch.ip_offset}
    source = next(
        i
        for i, stmt in enumerate(vex.statements)
        if isinstance(stmt, pyvex.IRStmt.Put)
        and isinstance(stmt.data, pyvex.IRExpr.Const)
        and stmt.data.con.value == 0x5000
    )
    assert (1 in _reference_uses(vex, offsets).get(source, ())) == (not alias_write)


def test_vex_guarded_accesses_use_addresses_not_store_values():
    arch = arch_from_id("AMD64")
    tyenv = pyvex.IRTypeEnv(arch)
    tyenv.add("Ity_I64")
    tyenv.add("Ity_I64")
    tyenv.add("Ity_I32")

    def const(value):
        return pyvex.IRExpr.Const(pyvex.const.U64(value))

    statements = [
        pyvex.IRStmt.WrTmp(0, pyvex.IRExpr.Get(16, "Ity_I64")),
        pyvex.IRStmt.WrTmp(
            1, pyvex.IRExpr.Binop("Iop_Add64", [pyvex.IRExpr.RdTmp(0), const(0x5000)])
        ),
        pyvex.IRStmt.LoadG(
            "Iend_LE",
            "ILGop_8Uto32",
            2,
            pyvex.IRExpr.RdTmp(1),
            pyvex.IRExpr.Const(pyvex.const.U32(0)),
            pyvex.IRExpr.Const(pyvex.const.U1(1)),
        ),
        pyvex.IRStmt.StoreG(
            "Iend_LE",
            pyvex.IRExpr.RdTmp(1),
            pyvex.IRExpr.Const(pyvex.const.U8(10)),
            pyvex.IRExpr.Const(pyvex.const.U1(1)),
        ),
    ]
    assert _reference_uses(
        SimpleNamespace(statements=statements, tyenv=tyenv), {16}
    ) == {1: {1}}


def test_symbolic_call_comments_preserve_operands_and_cfg(switch_cfg):
    project, cfg = switch_cfg
    before = render_cfg(project, 0x400544, False, False, "custom", "jump", "raw")
    edges = [
        (src, dst, deepcopy(data)) for src, dst, data in cfg.graph.edges(data=True)
    ]
    stats, summary = cfg.custom_stats.as_dict(), cfg.custom_summary.as_dict()
    with patch(
        "angr.analyses.cfg.cfg_fast.CFGFast.__init__", side_effect=AssertionError
    ):
        enabled = render_cfg(project, 0x400544, False, True, "custom", "jump", "raw")
    assert '<TD ALIGN="LEFT">0x400450</TD>' in enabled
    assert '<FONT COLOR="gray"> ; atoi </FONT>' in enabled
    assert '<FONT COLOR="gray"> ; puts </FONT>' in enabled
    assert list(cfg.graph.edges(data=True)) == edges
    assert cfg.custom_stats.as_dict() == stats
    assert cfg.custom_summary.as_dict() == summary
    assert not project.kb.functions
    render_cfg.cache_clear()
    assert (
        render_cfg(project, 0x400544, False, False, "custom", "jump", "raw") == before
    )


@pytest.mark.parametrize(
    ("name", "displayed"),
    [
        ("atoi", "atoi"),
        ("f" * MAX_LABEL_LENGTH, "f" * MAX_LABEL_LENGTH),
        ("f" * (MAX_LABEL_LENGTH + 1), "f" * MAX_LABEL_LENGTH + "..."),
        ("f" * 80 + "+0xc4", "f" * MAX_LABEL_LENGTH + "...+0xc4"),
        ("f" * 80 + "-0xABC", "f" * MAX_LABEL_LENGTH + "...-0xABC"),
    ],
)
def test_symbolic_transfer_comments_shorten_only_displayed_names(
    switch_cfg, name, displayed
):
    project, cfg = switch_cfg
    project.kb.labels[0x400450] = name
    obj = next(n for n in cfg.graph.nodes if n.addr == 0x400544)
    node = Node(obj, cfg.graph)
    node.kb.labels[0x400450] = name
    insn = next(project.arch.capstone.disasm(bytes.fromhex("e8ebfeffff"), 0x400560))
    annotation = CommentsDataRef()
    assert annotation.get_instruction_comments(node, insn) == [displayed + " "]
    assert annotation._symbol_name_at(node, 0x400450) == name
    assert project.kb.labels[0x400450] == name
    assert node.kb.labels[0x400450] == name
    assert insn.op_str == "0x400450"


@pytest.mark.parametrize("sort", ["unknown", "integer", "pointer-array", "string"])
@pytest.mark.parametrize("offset", ["", "+0x123", "-0xABC"])
def test_named_reference_comments_preserve_addresses_offsets_and_contents(sort, offset):
    name = "s" * 80 + offset
    displayed = "s" * MAX_LABEL_LENGTH + "..." + offset
    project = SimpleNamespace(kb=SimpleNamespace(labels={0x500000: name}))
    node = SimpleNamespace(project=project, kb=project.kb)
    data = SimpleNamespace(addr=0x500000, sort=sort)
    annotation = CommentsDataRef()
    location = f"{displayed} @ 0x500000"
    if sort == "string":
        data.content = b"a" * 40
        expected = f'{location}: "' + "a" * 40 + '"'
    elif sort == "pointer-array":
        expected = f"ptr slot {location}"
    else:
        expected = location
    assert annotation._format_memory_data_comment(node, data) == expected
    assert annotation._format_address_comment(node, 0x500000) == displayed
    assert annotation._symbol_name_at(node, 0x500000) == name
    # Byte previews must use the same shortened symbol, without shortening data.
    project.loader = SimpleNamespace(
        find_section_containing=lambda addr: SimpleNamespace(
            name=".rodata",
            vaddr=addr,
            memsize=8,
            is_readable=True,
            is_writable=False,
            is_executable=False,
        ),
        memory=SimpleNamespace(load=lambda addr, size: b"ABCDEFGH"),
    )
    data.sort, data.size = "integer", 8
    assert annotation._format_memory_data_comment(node, data, is_read=True) == (
        f'bytes {location}: "ABCDEFGH"'
    )
    assert project.kb.labels[0x500000] == name


@pytest.mark.parametrize(
    ("encoding", "expected"),
    [
        ("e8ebfeffff", ["atoi "]),  # call 0x400450
        ("e9ebfeffff", ["atoi "]),  # jmp 0x400450
        ("b850044000", []),  # mov eax, 0x400450 is not a proved pointer use
        ("3d50044000", []),  # cmp eax, 0x400450
        ("0550044000", []),  # add eax, 0x400450
        ("ffd0", []),  # call rax
        ("ff1500040000", []),  # call [rip + 0x400]: no implied callee
        ("e800000000", ["main+0x21 "]),  # call inside a sized function symbol
        ("e8ffffff7f", []),  # call an unnamed address
    ],
)
def test_only_direct_transfer_operands_receive_callee_names(
    switch_cfg, encoding, expected
):
    project, cfg = switch_cfg
    obj = next(n for n in cfg.graph.nodes if n.addr == 0x400544)
    insn = next(project.arch.capstone.disasm(bytes.fromhex(encoding), 0x400560))
    assert (
        CommentsDataRef().get_instruction_comments(Node(obj, cfg.graph), insn)
        == expected
    )


@pytest.mark.parametrize("mode", ["generated", "local", "explicit_offset", "hidden"])
def test_internal_branch_comments_suppress_only_repeated_generated_headers(
    switch_cfg, mode
):
    project, cfg = switch_cfg
    source = next(n for n in cfg.graph.nodes if 0x40056D in n.instruction_addrs)
    insn = source.block.capstone.insns[-1]
    destination = next(n for n in cfg.graph.nodes if n.addr == 0x4007BA)
    name = CommentsDataRef._symbol_name_at(Node(source, cfg.graph), destination.addr)
    graph = nx.DiGraph()
    graph.add_nodes_from(cfg.graph.nodes)
    graph.add_edges_from(cfg.graph.edges)
    if mode == "local":
        name = "error_path"
        project.kb.labels[destination.addr] = name
    elif mode == "explicit_offset":
        project.kb.labels[destination.addr] = name
    elif mode == "hidden":
        graph.remove_node(destination)
    expected = [] if mode == "generated" else [name + " "]
    assert (
        CommentsDataRef().get_instruction_comments(Node(source, graph), insn)
        == expected
    )


@pytest.mark.parametrize("label", ["$d", "$d.0", "$x.marker", "$data_table"])
def test_arm_mapping_metadata_is_not_a_symbolic_operand_name(label):
    symbol = SimpleNamespace(
        name=label, type=SymbolType.TYPE_NONE, size=0, is_import=False
    )
    project = SimpleNamespace(
        arch=SimpleNamespace(name="ARMEL"),
        kb=SimpleNamespace(labels={0x500000: label}),
        loader=SimpleNamespace(
            find_symbol=lambda addr, fuzzy=False: symbol,
            find_object_containing=lambda addr: None,
        ),
    )
    node = SimpleNamespace(project=project, kb=project.kb)
    assert CommentsDataRef._symbol_name_at(node, 0x500000) == (
        label if label == "$data_table" else None
    )


@pytest.mark.parametrize("sort", ["integer", "unknown", "pointer-array"])
@pytest.mark.parametrize("explicit", [False, True])
def test_unnamed_reference_notes_add_only_missing_addresses(
    monkeypatch, sort, explicit
):
    project = SimpleNamespace(
        kb=SimpleNamespace(labels={}),
        loader=SimpleNamespace(
            find_symbol=lambda addr, fuzzy=False: None,
            find_object_containing=lambda addr: None,
        ),
    )
    node = SimpleNamespace(
        project=project,
        kb=project.kb,
        obj=SimpleNamespace(is_simprocedure=False, is_syscall=False),
    )
    annotation = CommentsDataRef()
    comment = annotation._format_memory_data_comment(
        node, SimpleNamespace(addr=0x500000, sort=sort)
    )
    assert comment == "ref 0x500000"
    monkeypatch.setattr(
        CommentsDataRef,
        "get_comments_by_addr",
        lambda self, node: {0x400000: [comment, "variable @ 0x500000", '"hello"']},
    )
    row = {
        "_ins": SimpleNamespace(
            address=0x400000,
            op_str=("qword ptr [0x500000]" if explicit else "qword ptr [rip + 0x20]"),
        )
    }
    annotation.annotate_content(node, {"data": [row]})
    assert ("ref 0x500000" in row["comment"]["content"]) != explicit
    assert "variable @ 0x500000" in row["comment"]["content"]
    assert '"hello"' in row["comment"]["content"]


@pytest.mark.parametrize(
    ("size", "offset", "kind", "expected"),
    [
        (8, 4, SymbolType.TYPE_OBJECT, "buffer+0x4"),
        (8, 4, SymbolType.TYPE_FUNCTION, "buffer+0x4"),
        (8, 8, SymbolType.TYPE_OBJECT, None),
        (8, 9, SymbolType.TYPE_OBJECT, None),
        (0, 4, SymbolType.TYPE_OBJECT, None),
        (8, 4, SymbolType.TYPE_SECTION, None),
    ],
)
def test_symbol_offsets_require_real_symbol_bounds(size, offset, kind, expected):
    symbol = SimpleNamespace(
        name="buffer", rebased_addr=0x500000, size=size, type=kind, is_import=False
    )
    project = SimpleNamespace(
        kb=SimpleNamespace(labels={}),
        loader=SimpleNamespace(
            find_symbol=lambda addr, fuzzy=False: symbol if fuzzy else None,
            find_object_containing=lambda addr: None,
        ),
    )
    node = SimpleNamespace(project=project, kb=SimpleNamespace(labels={}))
    assert CommentsDataRef._symbol_name_at(node, 0x500000 + offset) == expected
    node.kb.labels[0x500000 + offset] = "exact_local_label"
    assert (
        CommentsDataRef._symbol_name_at(node, 0x500000 + offset) == "exact_local_label"
    )


def test_pointer_metadata_names_the_slot_not_an_inferred_callee():
    project = SimpleNamespace(
        kb=SimpleNamespace(labels={0x500000: "callback", 0x400450: "atoi"})
    )
    node = SimpleNamespace(project=project, kb=project.kb)
    data = SimpleNamespace(addr=0x500000, sort="pointer-array", pointer_addr=0x400450)
    comment = CommentsDataRef()._format_memory_data_comment(node, data)
    assert comment == "ptr slot callback @ 0x500000"
    assert "atoi" not in comment
    data.sort, data.content = "string", b"hello"
    assert CommentsDataRef()._format_memory_data_comment(node, data) == (
        'callback @ 0x500000: "hello"'
    )


def test_s390_native_call_metadata_supplies_missing_capstone_call_group():
    project = Project(
        "angr-binaries/tests/s390x/object_sensitivity_0", auto_load_libs=False
    )
    cfg = get_cfg(project, 0x401488, "custom")
    node = Node(next(n for n in cfg.graph.nodes if n.addr == 0x401488), cfg.graph)
    asm = NodeAsm()
    asm.add_annotator(CommentsDataRef())
    asm.render(node)
    call = node.content["asm"]["data"][-1]
    assert call["mnemonic"]["content"] == "brasl"
    assert call["operands"]["content"] == "%r14, 0x400fa0"
    assert call["comment"]["content"] == " ; __sprintf_chk "


def test_riscv_relative_call_uses_destination_label_not_displacement():
    project = Project(
        "angr-binaries/tests/riscv/autotalent-autotalent.so", auto_load_libs=False
    )
    cfg = get_cfg(project, 0x403C50, "custom")
    node = Node(next(n for n in cfg.graph.nodes if n.addr == 0x403C50), cfg.graph)
    call = node.obj.block.capstone.insns[-1]
    assert call.address == 0x403C58
    assert call.op_str == "-0x3278"
    project.kb.labels[0x4009E0] = "known_target"
    project.kb.labels[-0x3278] = "not_a_target"
    assert CommentsDataRef().get_instruction_comments(node, call) == ["known_target "]
