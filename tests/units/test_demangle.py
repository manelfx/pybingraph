"""Demangling is a display choice, never a change to symbol or CFG identity."""

import re
from html import escape, unescape
from types import SimpleNamespace
from typing import cast
from xml.etree import ElementTree

import pytest
from angr import Project
from angr.knowledge_plugins.cfg import CFGNode
from fastapi.testclient import TestClient

import bingraph.api.app as app_module
import bingraph.helpers.settings as settings_module
from bingraph.core.annotators.comments import CommentsDataRef
from bingraph.core.contents import NodeHead
from bingraph.core.labels import MAX_LABEL_LENGTH, short_label
from bingraph.core.outputs import escape as dot_escape
from bingraph.core.project import get_cfg
from bingraph.core.render import render_cfg
from bingraph.core.vis import Node
from bingraph.helpers.settings import GlobalSettings, Settings
from bingraph.helpers.symbols import FunctionSymbol
from bingraph.helpers.utils import demangle

METHOD = "a_very_long_method_name_for_formatting"
MANGLED = f"_ZN3Foo{len(METHOD)}{METHOD}Ev"
DEMANGLED = f"Foo::{METHOD}()"
SHORT = DEMANGLED[:MAX_LABEL_LENGTH] + "..."


@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("", ""),
        ("ordinary_name", "ordinary_name"),
        ("_Zbad", "_Zbad"),
        ("_Rbad", "_Rbad"),
        ("_Z3foov", "foo()"),
        (MANGLED, DEMANGLED),
        ("_Z3foov@@VERSION", "foo()@@VERSION"),
        ("_Z3foov.cold", "foo() (.cold)"),
        ("_ZN3foo3bar17h0123456789abcdefE", "foo::bar"),
        ("_ZN3foo3bar17h0123456789abcdefE.0", "foo::bar.0"),
        ("_ZN4test5$LT$A17h0123456789abcdefE", "test::<A"),
        ("_RNvC6_123foo3bar", "123foo::bar"),
    ],
)
def test_demangle_uses_existing_cpp_and_rust_support(name, expected):
    assert demangle(name) == expected


def test_demangle_setting_defaults_cli_and_environment(tmp_path, monkeypatch):
    monkeypatch.delenv("BINGRAPH_DEMANGLE", raising=False)
    options = {"root": tmp_path, "_env_file": None}
    assert not GlobalSettings(**options, _cli_parse_args=False).demangle
    assert GlobalSettings(**options, _cli_parse_args=["--demangle"]).demangle
    monkeypatch.setenv("BINGRAPH_DEMANGLE", "true")
    assert GlobalSettings(**options, _cli_parse_args=False).demangle
    assert not GlobalSettings(**options, _cli_parse_args=["--no-demangle"]).demangle


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("offset", ["", "+0x123", "-0xABC"])
@pytest.mark.parametrize("simprocedure", [False, True])
def test_headers_demangle_before_shortening_without_losing_offsets(
    enabled, offset, simprocedure
):
    original = MANGLED + offset
    obj = SimpleNamespace(
        addr=0x400000,
        block_id=0x400000,
        name=original,
        simprocedure_name=original if simprocedure else None,
        is_simprocedure=simprocedure,
        is_syscall=False,
        no_ret=False,
    )
    node = Node(cast(CFGNode, obj))
    NodeHead(demangle=enabled).gen_render(node)
    expected = SHORT if enabled else MANGLED[:MAX_LABEL_LENGTH] + "..."
    assert node.content["head"]["data"][0]["name"]["content"] == expected + offset
    assert obj.name == original and node.seq == "0x400000"
    assert short_label(None, demangle=enabled) is None


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("sort", ["unknown", "pointer-array", "string", "integer"])
def test_reference_comments_demangle_names_not_addresses_or_contents(enabled, sort):
    original = MANGLED + "+0xc4"
    kb = SimpleNamespace(labels={0x500000: original})
    region = SimpleNamespace(
        name=".rodata",
        vaddr=0x500000,
        memsize=8,
        is_readable=True,
        is_writable=False,
        is_executable=False,
    )
    project = SimpleNamespace(
        kb=kb,
        loader=SimpleNamespace(
            find_section_containing=lambda addr: region,
            memory=SimpleNamespace(load=lambda addr, size: b"dataHERE"[:size]),
        ),
    )
    node = SimpleNamespace(project=project, kb=kb)
    md = SimpleNamespace(addr=0x500000, sort=sort, size=8)
    annotation = CommentsDataRef(demangle=enabled)
    name = (SHORT if enabled else MANGLED[:MAX_LABEL_LENGTH] + "...") + "+0xc4"
    location = name + " @ 0x500000"
    expected = location
    if sort == "pointer-array":
        expected = "ptr slot " + location
    elif sort == "string":
        md.content = b"_Z3foov is literal text"
        expected += ': "_Z3foov is literal text"'
    elif sort == "integer":
        expected = "bytes " + location + ': "dataHERE"'
    assert annotation._format_memory_data_comment(node, md, is_read=True) == expected
    assert annotation._format_address_comment(node, md.addr) == name
    assert annotation._symbol_name_at(node, md.addr) == original


@pytest.mark.parametrize("endpoint", ["/symtab", "/api/symtab", "/cfg", "/api/cfg"])
@pytest.mark.parametrize("global_enabled", [False, True])
def test_demangle_api_default_and_request_overrides(
    tmp_path, monkeypatch, endpoint, global_enabled
):
    binary = tmp_path / "binary"
    binary.touch()
    settings = Settings.model_construct(
        root=tmp_path, demangle=global_enabled, server=None, client=None
    )
    monkeypatch.setattr(settings_module, "_settings", settings)
    monkeypatch.setattr(app_module, "load_project", lambda path: object())
    symbols = [
        FunctionSymbol(
            addr=0x400000, name=MANGLED, size=16, is_import=False, origin="symtab"
        )
    ]
    monkeypatch.setattr(app_module, "list_function_symbols", lambda project: symbols)
    calls = []
    monkeypatch.setattr(
        app_module, "render_cfg", lambda *args: calls.append(args) or "graph"
    )
    client = TestClient(app_module.create_app())
    for override in (None, True, False):
        params = {"filepath": "binary", "function": "0x400000", "format": "raw"}
        if override is not None:
            params["demangle"] = str(override).lower()
        expected = global_enabled if override is None else override
        response = client.get(endpoint, params=params)
        assert response.status_code == 200
        if endpoint == "/api/symtab":
            assert response.json()["items"] == [
                {**symbols[0].model_dump(), "name": DEMANGLED if expected else MANGLED}
            ]
        elif endpoint == "/symtab":
            assert escape(DEMANGLED if expected else MANGLED) in response.text
            if override is not None:
                assert f"&demangle={str(override).lower()}" in response.text
        else:
            assert calls[-1][-1] is expected
        assert symbols[0].name == MANGLED


def test_cfg_render_demangles_headers_and_calls_with_separate_cache_entries():
    project = Project("angr-binaries/tests/x86_64/switch", auto_load_libs=False)
    cfg = get_cfg(project, 0x400544, "custom")
    obj = next(n for n in cfg.graph.nodes if n.addr == 0x400544)
    obj._name = MANGLED
    cfg.kb.labels[0x400450] = MANGLED
    stats = cfg.custom_stats.as_dict()
    annotation = CommentsDataRef(demangle=True)
    insn = next(project.arch.capstone.disasm(bytes.fromhex("e8ebfeffff"), 0x400560))
    assert annotation.get_instruction_comments(Node(obj, cfg.graph), insn) == [
        SHORT + " "
    ]
    plain = render_cfg(project, 0x400544, False, True, "custom", "always", "raw")
    decoded = render_cfg(
        project, 0x400544, False, True, "custom", "always", "raw", True
    )
    assert plain != decoded and SHORT in unescape(decoded)
    assert MANGLED[:MAX_LABEL_LENGTH] + "..." in plain
    assert (
        render_cfg(project, 0x400544, False, True, "custom", "always", "raw") == plain
    )
    # Only display names change: instruction rows, graph edges and stats stay put.
    instruction = re.compile(r'<TR>(<TD ALIGN="LEFT">0x.*?</TD><TD.*?</TD><TD.*?</TD>)')
    assert instruction.findall(plain) == instruction.findall(decoded)
    edges = re.compile(r"^.* -> .*$", re.MULTILINE)
    assert edges.findall(plain) == edges.findall(decoded)
    assert cfg.custom_stats.as_dict() == stats
    assert obj.name == MANGLED and cfg.kb.labels[0x400450] == MANGLED


def test_demangled_reference_and_template_names_are_valid_svg():
    project = Project("angr-binaries/tests/x86_64/switch", auto_load_libs=False)
    cfg = get_cfg(project, 0x400544, "custom")
    obj = next(n for n in cfg.graph.nodes if n.addr == 0x400544)
    obj._name = "_Z3fooRi"
    cfg.kb.labels[0x400450] = "_Z3fooRi"
    svg = render_cfg(project, 0x400544, False, True, "custom", "jump", "svg", True)
    tree = ElementTree.fromstring(svg)
    assert "foo(int&)" in "".join(tree.itertext())
    assert "&amp;" in svg
    assert (
        ElementTree.fromstring("<text>" + dot_escape("foo<Bar>(int&)") + "</text>").text
        == "foo<Bar>(int&)"
    )
