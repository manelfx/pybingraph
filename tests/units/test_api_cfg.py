from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
import pytest

import bingraph.api.app as app_module
import bingraph.helpers.settings as settings_module
from bingraph.helpers import Settings
from bingraph.helpers.settings import GlobalSettings
from bingraph.core import project as project_module
from bingraph.core import render as render_module


def test_cfg_api_exits_query_overrides_the_default(monkeypatch, tmp_path) -> None:
    """Pass the request exit-display policy through to the renderer."""

    binary = tmp_path / "binary"
    binary.touch()
    settings = Settings.model_construct(
        root=tmp_path,
        cfg_mode="custom",
        cfg_exits="jump",
        comments=False,
        dfs_rank=False,
        log_level="INFO",
        debug=False,
        server=None,
        client=None,
    )
    monkeypatch.setattr(settings_module, "_settings", settings)
    monkeypatch.setattr(app_module, "load_project", lambda _: object())

    rendered_calls: list[tuple[object, ...]] = []

    def render_cfg(*args: object) -> str:
        rendered_calls.append(args)
        return "digraph G {}"

    monkeypatch.setattr(app_module, "render_cfg", render_cfg)
    client = TestClient(app_module.create_app())

    response = client.get(
        "/api/cfg",
        params={
            "filepath": binary.name,
            "function": "0x10",
            "format": "raw",
            "exits": "always",
        },
    )

    assert response.status_code == 200
    assert response.json() == {"graph": "digraph G {}"}
    assert [args[5] for args in rendered_calls] == ["always"]


@pytest.mark.parametrize("endpoint", ["/cfg", "/api/cfg"])
@pytest.mark.parametrize("default", [False, True])
def test_cfg_api_recovery_overrides_default(
    monkeypatch, tmp_path, endpoint, default
) -> None:
    """Omission inherits settings; explicit false must disable recovery."""

    binary = tmp_path / "binary"
    binary.touch()
    settings = Settings.model_construct(
        root=tmp_path,
        cfg_mode="extract",
        cfg_recovery=default,
        server=None,
        client=None,
    )
    monkeypatch.setattr(settings_module, "_settings", settings)
    monkeypatch.setattr(app_module, "load_project", lambda _: object())
    calls = []
    monkeypatch.setattr(
        app_module, "render_cfg", lambda *args: calls.append(args) or "graph"
    )
    client = TestClient(app_module.create_app())
    params = {"filepath": binary.name, "function": "0x10", "format": "raw"}
    for override, expected in [(None, default), ("true", True), ("false", False)]:
        query = params if override is None else {**params, "recovery": override}
        assert client.get(endpoint, params=query).status_code == 200
        assert calls[-1][7] is expected
    assert (
        client.get(endpoint, params={**params, "recovery": "invalid"}).status_code
        >= 400
    )
    assert len(calls) == 3


def test_cfg_recovery_settings_support_cli_and_config_file(
    monkeypatch, tmp_path
) -> None:
    """Expose the knob through normal settings, not a special runtime flag."""

    monkeypatch.delenv("BINGRAPH_CFG_RECOVERY", raising=False)
    config = tmp_path / ".bingraphenv"
    config.write_text("BINGRAPH_CFG_RECOVERY=true\n")
    assert (
        GlobalSettings(
            root=tmp_path, _env_file=config, _cli_parse_args=False
        ).cfg_recovery
        is True
    )
    assert (
        GlobalSettings(
            root=tmp_path, _env_file=config, _cli_parse_args=["--no-cfg-recovery"]
        ).cfg_recovery
        is False
    )
    assert (
        GlobalSettings(
            root=tmp_path, _env_file=None, _cli_parse_args=["--cfg-recovery"]
        ).cfg_recovery
        is True
    )
    assert (
        GlobalSettings(
            root=tmp_path, _env_file=None, _cli_parse_args=False
        ).cfg_recovery
        is False
    )
    monkeypatch.setenv("BINGRAPH_CFG_RECOVERY", "true")
    assert (
        GlobalSettings(
            root=tmp_path, _env_file=None, _cli_parse_args=False
        ).cfg_recovery
        is True
    )


def test_cfg_api_recovery_switches_real_extract_graph(monkeypatch) -> None:
    """Request overrides must reach extraction and both cache layers."""

    settings = Settings.model_construct(
        root=Path("angr-binaries/tests"),
        cfg_mode="extract",
        cfg_recovery=True,
        comments=False,
        server=None,
        client=None,
    )
    monkeypatch.setattr(settings_module, "_settings", settings)
    project_module.get_cfg.cache_clear()
    render_module.render_cfg.cache_clear()
    client = TestClient(app_module.create_app())
    params = {
        "filepath": "i386/bronze_ropchain",
        "function": "0x807b160",
        "format": "raw",
    }
    try:
        old_response = client.get("/api/cfg", params={**params, "recovery": "false"})
        assert old_response.status_code == 200
        original = old_response.json()["graph"]
        assert "UnresolvedEntrySource" not in original
        new_response = client.get("/api/cfg", params=params)
        assert new_response.status_code == 200
        recovered = new_response.json()["graph"]
        assert "UnresolvedEntrySource" in recovered
        assert '"0xffffffffffffffc0" ->' in recovered
        assert "color=orange, style=dashed" in recovered
        assert (
            client.get("/api/cfg", params={**params, "recovery": "false"}).json()[
                "graph"
            ]
            == original
        )
        assert (
            client.get("/api/cfg", params={**params, "recovery": "true"}).json()[
                "graph"
            ]
            == recovered
        )
    finally:
        project_module.get_cfg.cache_clear()
        render_module.render_cfg.cache_clear()
