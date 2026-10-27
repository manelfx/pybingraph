"""DDG API selection and CFG link metadata, without whole-corpus rendering."""

from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from unittest.mock import Mock

from fastapi.testclient import TestClient
from pydantic import ValidationError
import pytest

from bingraph.api import app as app_module
from bingraph.core.annotators.nodes import LinkDDG
from bingraph.core.vis import Node
from bingraph.helpers import settings as settings_module
from bingraph.helpers.settings import ClientSettings, Settings, GlobalSettings


def test_ddg_default_override_and_real_api(monkeypatch):
    monkeypatch.setattr(
        settings_module,
        "_settings",
        Settings.model_construct(
            root=Path("angr-binaries/tests"),
            server=None,
            client=None,
        ),
    )
    client = TestClient(app_module.create_app())
    params = {
        "filepath": "x86_64/switch",
        "address": "0x400544",
        "size": 34,
        "format": "raw",
    }
    response = client.get("/api/ddg", params=params)
    assert response.status_code == 200, response.text
    assert "RAW:" in response.json()["graph"]
    assert "WAR:" in response.json()["graph"]
    assert "WAW:" in response.json()["graph"]
    filtered = client.get("/api/ddg", params={**params, "dependencies": "raw"})
    assert filtered.status_code == 200
    assert "WAR:" not in filtered.json()["graph"]
    assert "WAW:" not in filtered.json()["graph"]
    always = client.get("/api/ddg", params={**params, "memory": "always"})
    assert always.status_code == 200
    assert always.json()["graph"] != response.json()["graph"]
    assert (
        client.get("/ddg", params=params)
        .headers["content-type"]
        .startswith("image/svg+xml")
    )
    cfg = client.get(
        "/api/cfg",
        params={"filepath": "x86_64/switch", "function": "0x400544", "format": "raw"},
    )
    assert cfg.status_code == 200
    assert (
        "/ddg?filepath=x86_64%2Fswitch&address=0x400544&size=34" in cfg.json()["graph"]
    )


def test_links_encode_paths_and_skip_simps():
    node = Node(
        SimpleNamespace(addr=0x10, size=8, block_id=0x10, is_simprocedure=False)
    )
    LinkDDG("a b/file&name").annotate_node(node)
    query = parse_qs(urlsplit(node.pydot.get("URL")).query)
    assert query == {"filepath": ["a b/file&name"], "address": ["0x10"], "size": ["8"]}
    assert node.pydot.get("target") is None
    node.obj.is_simprocedure = True
    synthetic = Node(node.obj)
    LinkDDG("binary").annotate_node(synthetic)
    assert synthetic.pydot.get("URL") is None


def test_ddg_cli_memory_setting(tmp_path):
    settings = GlobalSettings(
        root=tmp_path,
        _env_file=None,
        _cli_parse_args=[
            "--ddg-memory",
            "always",
            "--ddg-dependencies",
            "WAW, raw,raw",
        ],
    )
    assert settings.ddg_memory == "always"
    assert settings.ddg_dependencies == "raw,waw"
    assert (
        GlobalSettings(
            root=tmp_path, _env_file=None, _cli_parse_args=False
        ).ddg_dependencies
        == "raw,war,waw"
    )
    assert (
        GlobalSettings(root=tmp_path, _env_file=None, _cli_parse_args=False).ddg_memory
        == "must"
    )
    client = ClientSettings(endpoint="/ddg", filepath="binary", dependencies="WAR,raw")
    assert client.dependencies == "raw,war"
    for policy in ("always", "may", "must"):
        assert (
            GlobalSettings(
                root=tmp_path, _env_file=None, _cli_parse_args=["--ddg-memory", policy]
            ).ddg_memory
            == policy
        )
    with pytest.raises(ValidationError, match="Dependencies must be"):
        GlobalSettings(
            root=tmp_path,
            ddg_dependencies="raw,bad",
            _env_file=None,
            _cli_parse_args=False,
        )


@pytest.mark.parametrize("endpoint", ["/ddg", "/api/ddg"])
def test_dependency_selection_overrides_global_default(monkeypatch, tmp_path, endpoint):
    monkeypatch.setattr(
        settings_module,
        "_settings",
        Settings.model_construct(
            root=tmp_path, ddg_dependencies="war", server=None, client=None
        ),
    )
    binary = tmp_path / "binary"
    binary.touch()
    monkeypatch.setattr(app_module, "load_project", lambda _: object())
    render = Mock(return_value="digraph {}")
    monkeypatch.setattr(app_module, "render_ddg", render)
    client = TestClient(app_module.create_app())
    params = {"filepath": "binary", "address": "0x10", "size": 8}
    assert client.get(endpoint, params=params).status_code == 200
    assert render.call_args.args[-1] == "war"
    assert (
        client.get(
            endpoint, params={**params, "dependencies": " WAW,raw,raw "}
        ).status_code
        == 200
    )
    assert render.call_args.args[-1] == "raw,waw"


@pytest.mark.parametrize(
    "params", [{"address": "bad"}, {"filepath": "../outside"}, {"format": "png"}]
)
def test_ddg_bad_requests_do_not_analyze(monkeypatch, tmp_path, params):
    monkeypatch.setattr(
        settings_module,
        "_settings",
        Settings.model_construct(root=tmp_path, server=None, client=None),
    )
    monkeypatch.setattr(
        app_module,
        "load_project",
        lambda *args: pytest.fail("Invalid request loaded a binary"),
    )
    response = TestClient(app_module.create_app()).get(
        "/api/ddg",
        params={"filepath": "binary", "address": "0x10", "size": 8, **params},
    )
    assert response.status_code == 400


@pytest.mark.parametrize("endpoint", ["/ddg", "/api/ddg"])
@pytest.mark.parametrize(
    "params",
    [
        {"size": 0},
        {"size": 4097},
        {"memory": "bad"},
        {"memory": "smart"},
        {"dependencies": ""},
        {"dependencies": "raw,"},
        {"dependencies": "raw,bad"},
    ],
)
def test_ddg_invalid_query_bounds_do_not_analyze(
    monkeypatch, tmp_path, params, endpoint
):
    monkeypatch.setattr(
        settings_module,
        "_settings",
        Settings.model_construct(root=tmp_path, server=None, client=None),
    )
    monkeypatch.setattr(
        app_module,
        "load_project",
        lambda *args: pytest.fail("Invalid query loaded a binary"),
    )
    response = TestClient(app_module.create_app()).get(
        endpoint,
        params={"filepath": "binary", "address": "0x10", "size": 8, **params},
    )
    assert response.status_code == 422
