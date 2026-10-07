"""Node-header shortening is presentation-only and preserves block offsets."""

from types import SimpleNamespace
from typing import cast

from angr.knowledge_plugins.cfg import CFGNode
from archinfo import ArchAMD64, ArchARM, ArchAArch64
from cle.backends.symbol import SymbolType
import pytest

from bingraph.cfg.builder import _local_node_names
from bingraph.core.contents import MAX_NODE_LABEL_LENGTH, NodeHead
from bingraph.core.outputs import DotOutput
from bingraph.core.vis import Node


_LONG_NAME = "_ZN13fluent_bundle5types11FluentValue5write17hd5d59ec48b6606d8E"
_SHORT_NAME = _LONG_NAME[:MAX_NODE_LABEL_LENGTH] + "..."


@pytest.mark.parametrize("arch", [ArchAMD64(), ArchARM(), ArchAArch64()])
def test_local_labels_have_stable_alias_priority(arch) -> None:
    def symbol(name, kind=SymbolType.TYPE_NONE, **kwargs):
        return SimpleNamespace(
            rebased_addr=0x400010,
            name=name,
            type=kind,
            is_function=kind == SymbolType.TYPE_FUNCTION,
            **{"is_local": True, "is_import": False, **kwargs},
        )

    symbols = [
        symbol("zzz_label"),
        symbol("aaa_function", SymbolType.TYPE_FUNCTION),
        symbol("LoopStart"),
        symbol("aaa_object", SymbolType.TYPE_OBJECT),
        symbol("aaa_section", SymbolType.TYPE_SECTION),
        symbol("aaa_file", SymbolType.TYPE_OTHER),
        symbol("aaa_global", is_local=False),
        symbol("aaa_import", is_import=True),
        symbol(""),
    ]
    project = SimpleNamespace(
        arch=arch, loader=SimpleNamespace(main_object=SimpleNamespace(symbols=symbols))
    )
    assert _local_node_names(project, [0x400010, 0x400014]) == {0x400010: "LoopStart"}
    symbols.reverse()
    assert _local_node_names(project, [0x400010]) == {0x400010: "LoopStart"}
    project.loader.main_object.symbols = [
        symbol("function_label", SymbolType.TYPE_FUNCTION)
    ]
    assert _local_node_names(project, [0x400010]) == {0x400010: "function_label"}


@pytest.mark.parametrize("arch", [ArchARM(), ArchAArch64()])
@pytest.mark.parametrize("mapping", ["$a", "$d", "$t", "$x", "$t.1", "$d.42"])
def test_arm_mapping_symbols_are_not_node_titles(arch, mapping) -> None:
    symbol = SimpleNamespace(
        rebased_addr=0x400010,
        name=mapping,
        type=SymbolType.TYPE_NONE,
        is_function=False,
        is_local=True,
        is_import=False,
    )
    project = SimpleNamespace(
        arch=arch, loader=SimpleNamespace(main_object=SimpleNamespace(symbols=[symbol]))
    )
    assert _local_node_names(project, [0x400010]) == {}
    symbol.name = "$table_loop"
    assert _local_node_names(project, [0x400010]) == {0x400010: "$table_loop"}


@pytest.mark.parametrize("symbol_addr", [0x400010, 0x400011])
def test_thumb_labels_match_normalized_block_starts(symbol_addr) -> None:
    symbol = SimpleNamespace(
        rebased_addr=symbol_addr,
        name="ThumbLoop",
        type=SymbolType.TYPE_NONE,
        is_function=False,
        is_local=True,
        is_import=False,
    )
    project = SimpleNamespace(
        arch=ArchARM(),
        loader=SimpleNamespace(main_object=SimpleNamespace(symbols=[symbol])),
    )
    assert _local_node_names(project, [0x400010, 0x400011, 0x400015]) == {
        0x400010: "ThumbLoop",
        0x400011: "ThumbLoop",
    }


@pytest.mark.parametrize(
    ("name", "displayed"),
    [
        (None, None),
        ("", ""),
        ("short_name", "short_name"),
        ("f" * MAX_NODE_LABEL_LENGTH, "f" * MAX_NODE_LABEL_LENGTH),
        ("f" * (MAX_NODE_LABEL_LENGTH + 1), "f" * MAX_NODE_LABEL_LENGTH + "..."),
        (_LONG_NAME, _SHORT_NAME),
        (_LONG_NAME + "+0xc4", _SHORT_NAME + "+0xc4"),
        (_LONG_NAME + "-0xABC", _SHORT_NAME + "-0xABC"),
        (
            "f" * MAX_NODE_LABEL_LENGTH + "+0x12345",
            "f" * MAX_NODE_LABEL_LENGTH + "+0x12345",
        ),
        ("operator+0xabc<long_template_name>", "operator+0xabc<long_template_n..."),
    ],
)
def test_header_shortens_only_the_label(name, displayed) -> None:
    obj = SimpleNamespace(
        addr=0x4AD824,
        block_id=0x4AD824,
        name=name,
        simprocedure_name=name,
        is_simprocedure=True,
        is_syscall=True,
        no_ret=True,
    )
    node = Node(cast(CFGNode, obj))

    NodeHead().gen_render(node)

    header = node.content["head"]["data"][0]
    assert header["name"] == {"content": displayed, "style": "B"}
    assert header["addr"]["content"] == "(0x4ad824)"
    assert header["attributes"]["content"] == " SIMP  SYSC  NORET"
    assert obj.name == name


@pytest.mark.parametrize(
    ("name", "procedure_name", "is_simprocedure", "show_addr"),
    [
        ("UnresolvableJumpTarget", "UnresolvableJumpTarget", True, False),
        ("UnresolvableCallTarget", "UnresolvableCallTarget", True, False),
        ("UnresolvableEntrySource", "UnresolvableEntrySource", True, False),
        ("UnresolvableTarget", "UnresolvableTarget", True, False),
        ("renamed", "UnresolvableFutureTarget", True, False),
        ("ExternalTarget_0x400000", "ExternalTarget_0x400000", True, True),
        ("sys_0", "sys_0", True, True),
        ("normal_function", None, False, True),
        ("UnresolvableJumpTarget", None, False, True),
    ],
)
def test_unknown_simprocedure_headers_omit_only_the_displayed_address(
    name, procedure_name, is_simprocedure, show_addr
) -> None:
    address = 0xFFFFFFFFFFFFFFD0
    obj = SimpleNamespace(
        addr=address,
        block_id=address,
        name=name,
        simprocedure_name=procedure_name,
        is_simprocedure=is_simprocedure,
        is_syscall=False,
        no_ret=False,
    )
    node = Node(cast(CFGNode, obj))

    NodeHead().gen_render(node)

    header = node.content["head"]
    assert ("addr" in header["columns"]) == show_addr
    assert (
        "(0xffffffffffffffd0)" in DotOutput(fname="unused").render_content(header)
    ) == show_addr
    assert header["data"][0]["attributes"]["content"] == (
        " SIMP" if is_simprocedure else ""
    )
    assert node.seq == hex(address)
    assert node.pydot.get_name() == hex(address)
    assert obj.addr == address and obj.name == name


@pytest.mark.parametrize(
    ("name", "is_simprocedure", "displayed"),
    [
        ("ExternalTarget_0x400000", True, "ExternalTarget"),
        ("ExternalTarget_0xABCDEF", True, "ExternalTarget"),
        ("ExternalTarget_0xfffffffffffffff0", True, "ExternalTarget"),
        ("ExternalTarget_0x400000", False, "ExternalTarget_0x400000"),
        ("ExternalTarget", True, "ExternalTarget"),
        ("ExternalTarget_0xnothex", True, "ExternalTarget_0xnothex"),
        ("ExternalTarget_0x123_suffix", True, "ExternalTarget_0x123_suffix"),
        ("memcpy", True, "memcpy"),
    ],
)
def test_external_target_headers_do_not_repeat_the_address(
    name, is_simprocedure, displayed
) -> None:
    obj = SimpleNamespace(
        addr=0x400000,
        block_id=0x400000,
        name=name,
        simprocedure_name=name if is_simprocedure else None,
        is_simprocedure=is_simprocedure,
        is_syscall=False,
        no_ret=False,
    )
    node = Node(cast(CFGNode, obj))

    NodeHead().gen_render(node)

    header = node.content["head"]
    assert header["data"][0]["name"] == {"content": displayed, "style": "B"}
    assert header["data"][0]["addr"]["content"] == "(0x400000)"
    assert "addr" in header["columns"]
    assert node.seq == "0x400000"
    assert obj.name == name
