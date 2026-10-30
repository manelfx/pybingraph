from __future__ import annotations

from pathlib import Path

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

import bingraph
import bingraph.api.app as app_module
import bingraph.helpers.settings as settings_module
from bingraph.helpers import Settings
from bingraph.helpers.settings import GlobalSettings
from bingraph.core import project as project_module
from bingraph.core import render as render_module


def test_openapi_reports_package_version_and_text_formats(monkeypatch, tmp_path):
    settings = Settings.model_construct(root=tmp_path, server=None, client=None)
    monkeypatch.setattr(settings_module, "_settings", settings)
    schema = app_module.create_app().openapi()
    assert schema["info"]["version"] == bingraph.__version__
    parameter = next(
        item
        for item in schema["paths"]["/api/cfg"]["get"]["parameters"]
        if item["name"] == "format"
    )
    assert parameter["schema"]["enum"] == ["svg", "dot", "raw", "xdot"]


@pytest.mark.parametrize("endpoint", ["/cfg", "/api/cfg"])
@pytest.mark.parametrize("debug", [False, True])
def test_error_messages_escape_only_nondebug_html(
    monkeypatch, tmp_path, endpoint, debug
):
    """Render exception text literally without changing JSON or debug propagation."""

    (tmp_path / "binary").touch()
    settings = Settings.model_construct(
        root=tmp_path, debug=debug, server=None, client=None
    )
    monkeypatch.setattr(settings_module, "_settings", settings)
    message = '<script>alert("error")</script> & missing binary'

    def fail_loading(_):
        raise ValueError(message)

    monkeypatch.setattr(app_module, "load_project", fail_loading)
    client = TestClient(app_module.create_app())
    params = {"filepath": "binary", "function": "0x10", "format": "raw"}
    if debug:
        with pytest.raises(ValueError) as exc:
            client.get(endpoint, params=params)
        assert str(exc.value) == message
        return

    response = client.get(endpoint, params=params)
    assert response.status_code == 400
    if endpoint.startswith("/api"):
        assert response.json() == {"error": message}
    else:
        assert "<script>" not in response.text
        assert (
            "<p>&lt;script&gt;alert(&quot;error&quot;)&lt;/script&gt; &amp; missing binary</p>"
            in response.text
        )


@pytest.mark.parametrize("format", ["png", "pdf", "unknown"])
def test_nontext_formats_are_rejected_before_loading_or_analysis(
    monkeypatch, tmp_path, format
):
    settings = Settings.model_construct(root=tmp_path, server=None, client=None)
    monkeypatch.setattr(settings_module, "_settings", settings)

    def unexpected(*args, **kwargs):
        pytest.fail("Unsupported format reached binary loading or CFG analysis")

    monkeypatch.setattr(app_module, "load_project", unexpected)
    monkeypatch.setattr(render_module, "get_cfg", unexpected)
    response = TestClient(app_module.create_app()).get(
        "/api/cfg", params={"filepath": "missing", "function": "0x10", "format": format}
    )
    assert response.status_code == 400
    assert "Unsupported CFG format" in response.json()["error"]
    with pytest.raises(ValueError, match="Unsupported CFG format"):
        render_module.render_cfg(object(), 0x10, False, False, "custom", "jump", format)


def test_cfg_settings_and_cli_only_support_current_modes(tmp_path) -> None:
    """The default and CLI advertise only bounded custom construction or CFGFast."""

    settings = GlobalSettings(root=tmp_path, _env_file=None, _cli_parse_args=False)
    assert settings.cfg_mode == "custom"
    for mode in ("none", "custom"):
        settings = GlobalSettings(
            root=tmp_path,
            _env_file=None,
            _cli_parse_args=["--cfg-mode", mode],
        )
        assert settings.cfg_mode == mode
    with pytest.raises(ValidationError):
        GlobalSettings(
            root=tmp_path, cfg_mode="extract", _env_file=None, _cli_parse_args=False
        )
    with pytest.raises(ValidationError):
        GlobalSettings(
            root=tmp_path, _env_file=None, _cli_parse_args=["--cfg-mode", "extract"]
        )


@pytest.mark.parametrize("endpoint", ["/cfg", "/api/cfg"])
def test_cfg_api_rejects_the_retired_mode(monkeypatch, tmp_path, endpoint) -> None:
    """Do not retain an implicit compatibility alias in either API route."""

    settings = Settings.model_construct(root=tmp_path, server=None, client=None)
    monkeypatch.setattr(settings_module, "_settings", settings)
    client = TestClient(app_module.create_app())
    response = client.get(
        endpoint,
        params={
            "filepath": "binary",
            "function": "0x10",
            "format": "raw",
            "mode": "extract",
        },
    )
    # Query validation errors are client errors, not internal analysis failures.
    assert response.status_code == 422
    assert "Input should be" in response.text
    assert "none" in response.text and "custom" in response.text


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
        cfg_mode="custom",
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
    assert len(calls) == 1 and len(calls[0]) == 9
    names = {p["name"] for p in app.openapi()["paths"][endpoint]["get"]["parameters"]}
    assert "recovery" not in names


@pytest.mark.parametrize("flag", ["--cfg-recovery", "--no-cfg-recovery"])
def test_cfg_recovery_setting_and_cli_flags_are_removed(tmp_path, flag) -> None:
    """Retire both directions of the flag, rather than retain a hidden knob."""

    assert "cfg_recovery" not in GlobalSettings.model_fields
    with pytest.raises(SystemExit) as exc:
        GlobalSettings(root=tmp_path, _env_file=None, _cli_parse_args=[flag])
    assert exc.value.code == 2


def test_cfg_api_custom_always_recovers_disconnected_code(monkeypatch) -> None:
    """Default requests recover code, and obsolete overrides cannot disable it."""

    monkeypatch.setenv("BINGRAPH_CFG_RECOVERY", "false")
    settings = Settings.model_construct(
        root=Path("angr-binaries/tests"),
        cfg_mode="custom",
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
