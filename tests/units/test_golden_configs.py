"""Graph configuration filtering and isolated DDG comparison/promotion paths."""

from types import SimpleNamespace

import pytest

from tests.goldens import configs
from tests.goldens import test_ddg_goldens as ddg_suite


@pytest.mark.parametrize(
    "raw,cfg,ddg",
    [
        (None, ["cfg_mode_custom"], ["ddg_default"]),
        ("  ", ["cfg_mode_custom"], ["ddg_default"]),
        ("cfg_mode_custom", ["cfg_mode_custom"], []),
        ("ddg_memory_may", [], ["ddg_memory_may"]),
        (
            "ddg_dependencies_raw, cfg_mode_none,ddg_default,ddg_dependencies_raw",
            ["cfg_mode_none"],
            ["ddg_dependencies_raw", "ddg_default"],
        ),
    ],
)
def test_configuration_filter_preserves_order_and_suite_boundaries(
    monkeypatch, raw, cfg, ddg
):
    if raw is None:
        monkeypatch.delenv(configs.CONFIGS_ENV, raising=False)
    else:
        monkeypatch.setenv(configs.CONFIGS_ENV, raw)
    assert [
        c.name
        for c in configs.selected_configs(
            configs.CFG_CONFIGS, configs.DEFAULT_CFG_CONFIGS
        )
    ] == cfg
    assert [
        c.name
        for c in configs.selected_configs(
            configs.DDG_CONFIGS, configs.DEFAULT_DDG_CONFIGS
        )
    ] == ddg


def test_unknown_configuration_is_not_silently_filtered(monkeypatch):
    monkeypatch.setenv(configs.CONFIGS_ENV, "cfg_mode_custom,ddg_dedendencies_raw")
    for family, defaults in (
        (configs.CFG_CONFIGS, configs.DEFAULT_CFG_CONFIGS),
        (configs.DDG_CONFIGS, configs.DEFAULT_DDG_CONFIGS),
    ):
        with pytest.raises(ValueError, match="unknown config.*ddg_dedendencies_raw"):
            configs.selected_configs(family, defaults)


@pytest.mark.parametrize("config", configs.DDG_CONFIGS, ids=lambda c: c.name)
def test_ddg_overrides_and_artifacts_are_isolated(monkeypatch, tmp_path, config):
    """Use temporary artifacts only; defaults must not become explicit overrides."""
    monkeypatch.setattr(ddg_suite, "TESTS", tmp_path)
    monkeypatch.setattr(ddg_suite, "create_app", lambda: object())
    monkeypatch.setenv("BINGRAPH_GOLDEN_MODE", "compare")
    requests = []

    def get(url, params):
        requests.append((url, params))
        return SimpleNamespace(
            status_code=200, json=lambda: {"graph": "reviewed DOT\n"}, text=""
        )

    monkeypatch.setattr(ddg_suite, "TestClient", lambda app: SimpleNamespace(get=get))
    row = dict(filepath="a/bin", address="0x10", size="8")
    name = "a,bin,0x10,8.dot"
    golden = tmp_path / "goldens" / config.name / name
    golden.parent.mkdir(parents=True)
    golden.write_text("reviewed DOT\n")
    ddg_suite.test_render_ddg_goldens(config, row, monkeypatch)
    expected = dict(filepath="a/bin", address="0x10", size="8", format="raw")
    if config.memory is not None:
        expected["memory"] = config.memory
    if config.dependencies is not None:
        expected["dependencies"] = config.dependencies
    assert requests == [("/api/ddg", expected)]
    actual = tmp_path / "_actual" / config.name / name
    assert actual.read_bytes() == golden.read_bytes()

    # Promotion copies this configuration's candidate, never invoking the API.
    actual.write_text("new reviewed DOT\n")
    monkeypatch.setenv("BINGRAPH_GOLDEN_MODE", "promote")
    monkeypatch.setattr(
        ddg_suite, "create_app", lambda: pytest.fail("Promotion rerendered")
    )
    ddg_suite.test_render_ddg_goldens(config, row, monkeypatch)
    assert golden.read_bytes() == actual.read_bytes()
    assert len(requests) == 1
