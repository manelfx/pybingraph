"""Frozen DDG corpus with a separately marked checkpoint subset.

The manifest records exact bytes, not function-discovery guesses. By default,
each block runs once with ddg_default, leaving memory/dependencies unspecified.
BINGRAPH_GOLDEN_CONFIGS selects other named configurations or a mixed CFG/DDG
list. Compare writes separate candidates per configuration; promotion copies
those exact files and never rerenders. Missing baselines require user review.
"""

import csv
import os
from pathlib import Path

from fastapi.testclient import TestClient
import pytest

from bingraph.api.app import create_app
from bingraph.helpers.settings import Settings
from bingraph.helpers import settings as settings_module
from tests.goldens.configs import DDG_CONFIGS, DEFAULT_DDG_CONFIGS, selected_configs

TESTS = Path(__file__).resolve().parents[1]
ROOT = TESTS.parent / "angr-binaries/tests"
with (TESTS / "playground_blocks.csv").open(newline="") as stream:
    ROWS = list(csv.DictReader(stream))
ACTIVE_CONFIGS = selected_configs(DDG_CONFIGS, DEFAULT_DDG_CONFIGS)


@pytest.mark.parametrize(
    "config,row",
    [
        pytest.param(
            config,
            row,
            id=f"{config.name}-{row['filepath']}-{row['address']}-{row['size']}",
            marks=pytest.mark.checkpoint if row["checkpoint"] == "true" else (),
        )
        for config in ACTIVE_CONFIGS
        for row in ROWS
    ],
)
def test_render_ddg_goldens(config, row, monkeypatch):
    name = row["filepath"].replace("/", ",") + f",{row['address']},{row['size']}.dot"
    relative = Path(config.name) / name
    actual, golden = TESTS / "_actual" / relative, TESTS / "goldens" / relative
    mode = os.environ.get("BINGRAPH_GOLDEN_MODE", "compare")
    if mode == "promote":
        assert actual.is_file(), f"No reviewed candidate: {actual}"
        golden.parent.mkdir(parents=True, exist_ok=True)
        golden.write_bytes(actual.read_bytes())
        return
    assert mode == "compare", "BINGRAPH_GOLDEN_MODE must be compare or promote"
    monkeypatch.setattr(
        settings_module,
        "_settings",
        Settings.model_construct(root=ROOT, comments=False, server=None, client=None),
    )
    response = TestClient(create_app()).get(
        "/api/ddg",
        params={
            "filepath": row["filepath"],
            "address": row["address"],
            "size": row["size"],
            "format": "raw",
            # None means inherit the real API default, not a frozen test value.
            **{
                key: value
                for key in ("memory", "dependencies")
                if (value := getattr(config, key)) is not None
            },
        },
    )
    assert response.status_code == 200, response.text
    dot = response.json()["graph"]
    actual.parent.mkdir(parents=True, exist_ok=True)
    actual.write_text(dot)
    assert golden.is_file(), f"Missing DDG golden; review candidate: {actual}"
    assert dot == golden.read_text(), f"DDG differs: {actual}"
