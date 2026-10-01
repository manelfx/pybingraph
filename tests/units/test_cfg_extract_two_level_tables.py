"""Focused proofs for guarded byte-map to pointer-table dispatches."""

from pathlib import Path

from angr import KnowledgeBase
import pytest

from bingraph.cfg_extract.builder import _ExtractionSession
from bingraph.cfg_extract.two_level_tables import _ProbeRejected, _immutable_values
from bingraph.core.project import load_project


@pytest.mark.parametrize(
    ("binary", "function", "dispatcher", "new_target", "minimum_coverage"),
    [
        (
            "s390x/test-instr_s390x",
            0x80029818,
            0x8002A3DE,
            0x8002A6DE,
            1800,
        ),
        ("x86_64/static", 0x451F40, 0x45279C, 0x452909, 1800),
    ],
)
def test_extract_proves_two_level_immutable_tables(
    binary: str,
    function: int,
    dispatcher: int,
    new_target: int,
    minimum_coverage: int,
) -> None:
    """A complete guard replaces the UJT and exposes real code, not guesses."""

    project = load_project(Path("angr-binaries/tests") / binary)
    session = _ExtractionSession(project, KnowledgeBase(project), function)
    cfg = session.build()

    targets = session.static_targets[dispatcher]
    assert len(targets) == 14
    assert new_target in targets
    assert cfg.extract_stats.exact_jump_proofs_by_flavor.get("two_level_table", 0) >= 1
    nodes = {node.addr: node for node in cfg.graph.nodes() if not node.is_simprocedure}
    assert {node.addr for node in cfg.graph.successors(nodes[dispatcher])} == set(
        targets
    )
    if binary == "x86_64/static":
        assert 0x452FDF not in nodes  # Former dashed target was a padding NOP.
    assert (
        len(
            {
                addr
                for block in session.blocks.values()
                for addr in block.instruction_addrs
            }
        )
        >= minimum_coverage
    )


@pytest.mark.parametrize(
    ("binary", "function", "dispatcher"),
    [
        ("i386/bronze_ropchain", 0x807DE20, 0x807EAE0),
        ("mipsel/mips_syscall_demo", 0x402320, 0x402B98),
        ("ppc64el/fauxware_static", 0x1004F5F0, 0x100500C0),
    ],
)
def test_two_level_table_keeps_unproven_dispatchers_unresolved(
    binary: str, function: int, dispatcher: int
) -> None:
    """An unproved base or guard cannot turn candidate rows into solid edges."""

    project = load_project(Path("angr-binaries/tests") / binary)
    session = _ExtractionSession(project, KnowledgeBase(project), function)
    session.build()

    assert dispatcher not in session.static_targets


def test_two_level_table_rejects_writable_map() -> None:
    """A finite selector is insufficient if the table can change at runtime."""

    project = load_project(Path("angr-binaries/tests/x86_64/static"))
    section = next(
        section
        for section in project.loader.main_object.sections
        if section.is_readable and section.is_writable and section.memsize
    )
    with pytest.raises(_ProbeRejected):
        _immutable_values(project, (section.min_addr,), 1)
