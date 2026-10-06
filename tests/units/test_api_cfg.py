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
def test_cfg_api_has_no_recovery_parameter(monkeypatch, tmp_path, endpoint) -> None:
    """Recovery is not a public route parameter or renderer cache dimension."""

    binary = tmp_path / "binary"
    binary.touch()
    settings = Settings.model_construct(
        root=tmp_path,
        cfg_mode="extract",
        server=None,
        client=None,
    )
    monkeypatch.setattr(settings_module, "_settings", settings)
    monkeypatch.setattr(app_module, "load_project", lambda _: object())
    calls = []
    monkeypatch.setattr(
        app_module, "render_cfg", lambda *args: calls.append(args) or "graph"
    )
    app = app_module.create_app()
    client = TestClient(app)
    params = {"filepath": binary.name, "function": "0x10", "format": "raw"}
    assert client.get(endpoint, params=params).status_code == 200
    assert len(calls) == 1 and len(calls[0]) == 7
    names = {p["name"] for p in app.openapi()["paths"][endpoint]["get"]["parameters"]}
    assert "recovery" not in names


@pytest.mark.parametrize("flag", ["--cfg-recovery", "--no-cfg-recovery"])
def test_cfg_recovery_setting_and_cli_flags_are_removed(tmp_path, flag) -> None:
    """Retire both directions of the flag, rather than retain a hidden knob."""

    assert "cfg_recovery" not in GlobalSettings.model_fields
    with pytest.raises(SystemExit) as exc:
        GlobalSettings(root=tmp_path, _env_file=None, _cli_parse_args=[flag])
    assert exc.value.code == 2


def test_cfg_api_extract_always_recovers_disconnected_code(monkeypatch) -> None:
    """Default requests recover code, and obsolete overrides cannot disable it."""

    monkeypatch.setenv("BINGRAPH_CFG_RECOVERY", "false")
    settings = Settings.model_construct(
        root=Path("angr-binaries/tests"),
        cfg_mode="extract",
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
        new_response = client.get("/api/cfg", params=params)
        assert new_response.status_code == 200
        recovered = new_response.json()["graph"]
        assert "UnresolvableEntrySource" in recovered
        assert "UnresolvedEntrySource" not in recovered
        assert '"0xffffffffffffffc0" ->' in recovered
        assert "color=orange, style=dashed" in recovered
        assert (
            client.get("/api/cfg", params={**params, "recovery": "false"}).json()[
                "graph"
            ]
            == recovered
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
