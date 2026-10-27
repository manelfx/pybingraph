"""Offline DDG selection is bounded, deterministic and protects regression cases."""

from collections import Counter
import csv
import importlib.util
from pathlib import Path
import sys

import pytest


@pytest.fixture
def corpus():
    path = Path(__file__).resolve().parents[2] / "scripts/ddg.py"
    spec = importlib.util.spec_from_file_location("ddg_corpus", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_balancing_enforces_caps_and_skips_duplicate_shapes(corpus):
    blocks = [
        corpus.Block(
            f"{arch}/{binary}",
            address,
            size=4,
            architecture=arch,
            function=str(address // 3),
            shape=f"{arch}/{binary}/{address // 2}",
        )
        for arch in ("a", "b")
        for binary in ("first", "second")
        for address in range(12)
    ]
    seed = blocks[0]
    selected = corpus.balanced(blocks, 10, per_binary=3, per_function=2, initial=[seed])
    assert len(selected) == 10
    assert selected[0] == seed
    assert len({b.key for b in selected}) == len({b.shape for b in selected}) == 10
    assert max(Counter(b.filepath for b in selected).values()) <= 3
    assert max(Counter((b.filepath, b.function) for b in selected).values()) <= 2
    assert {b.architecture for b in selected} == {"a", "b"}


def test_checkpoints_preserve_seeds_cover_features_and_prefer_small_blocks(corpus):
    seed = corpus.Block("a/bin", 1, 4, (1,), "seed", "a", frozenset({"old"}))
    small = corpus.Block("b/bin", 2, 8, (2, 3), "small", "b", frozenset({"new"}))
    large = corpus.Block(
        "b/bin", 3, 40, tuple(range(10)), "large", "b", frozenset({"new"})
    )
    assert corpus.checkpoints([large, small, seed], [seed], 2) == [seed, small]
    assert corpus.checkpoints([seed, small, large], [seed], 2) == [seed, small]
    with pytest.raises(ValueError, match="protected seed"):
        corpus.checkpoints([seed], [seed], 0)


def test_inventory_uses_assembly_boundaries_and_deduplicates_functions(
    corpus, tmp_path
):
    functions = tmp_path / "functions.csv"
    functions.write_text('filepath\n"x86_64/a,b"\n')
    # Keep Thumb bits and HTML-encoded colons; synthetic headers are not code.
    node = (
        '"0x101" [label=<<TD>(0x101)</TD>'
        + "".join(
            f"<TD>0x{address:x}&#58;&nbsp;</TD>"
            for address in (0x101, 0x103, 0x105, 0x107)
        )
        + '>];\n"0x200" [label=<<TD>UnresolvableJumpTarget</TD>>];\n'
    )
    for address in (0x100, 0x200):
        (tmp_path / f"x86_64,a,b,0x{address:x},fn.dot").write_text(node)
    blocks, counts = corpus.inventory(tmp_path, functions)
    assert counts == {"x86_64": 1}
    assert len(blocks) == 1
    assert blocks[0].filepath == "x86_64/a,b"
    assert blocks[0].addresses == (0x101, 0x103, 0x105, 0x107)


def test_seed_flags_preserve_only_curated_rows(corpus, tmp_path):
    manifest = tmp_path / "blocks.csv"
    manifest.write_text(
        "filepath,address,size,checkpoint,purpose\na/bin,0x10,4,true,regression\na/bin,0x20,8,false,ordinary\n"
    )
    assert [b.key for b in corpus.read_seeds(manifest)] == [("a/bin", 16, 4)]


def test_frozen_manifest_is_unique_and_has_a_real_checkpoint_subset():
    manifest = Path(__file__).resolve().parents[1] / "playground_blocks.csv"
    with manifest.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    keys = {(r["filepath"], int(r["address"], 0), int(r["size"])) for r in rows}
    assert len(keys) == len(rows)
    assert 80 <= sum(r["checkpoint"] == "true" for r in rows) <= 120
    assert len(rows) >= 1000
    assert all(r["checkpoint"] in {"true", "false"} and r["purpose"] for r in rows)
    assert all(0 < size <= 4096 and address >= 0 for _, address, size in keys)
