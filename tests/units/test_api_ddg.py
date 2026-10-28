"""DDG API selection and CFG link metadata, without whole-corpus rendering."""

from pathlib import Path
from types import SimpleNamespace
from urllib.parse import parse_qs, urlsplit
from unittest.mock import Mock
from xml.etree import ElementTree

from fastapi.testclient import TestClient
from angr.knowledge_plugins.cfg import CFGNode
import networkx as nx
from pydantic import ValidationError
import pytest

from bingraph.api import app as app_module
from bingraph.core.annotators.nodes import LinkDDG
from bingraph.core.outputs import DotOutput
from bingraph.core.vis import Graph, Node
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
    cfg_svg = client.get(
        "/cfg", params={"filepath": "x86_64/switch", "function": "0x400544"}
    )
    assert cfg_svg.status_code == 200, cfg_svg.text
    link = next(
        node
        for node in ElementTree.fromstring(cfg_svg.text).iter(
            "{http://www.w3.org/2000/svg}a"
        )
        if node.get("{http://www.w3.org/1999/xlink}href")
        == "/ddg?filepath=x86_64%2Fswitch&address=0x400544&size=34"
    )
    assert link.get("{http://www.w3.org/1999/xlink}title") == (
        "Block 0x400544 (10 instructions, 34 bytes)"
    )


def test_ddg_repeated_comparison_in_real_cvs_block(monkeypatch):
    monkeypatch.setattr(
        settings_module,
        "_settings",
        Settings.model_construct(
            root=Path("angr-binaries/tests"), server=None, client=None
        ),
    )
    client = TestClient(app_module.create_app())
    params = {"filepath": "x86_64/cvs", "address": "0x4092c0", "size": 72}
    response = client.get("/api/ddg", params={**params, "format": "raw"})
    assert response.status_code == 200, response.text
    assert "repe cmpsb" in response.json()["graph"]
    assert "0x409302" in response.json()["graph"]
    svg = client.get("/ddg", params=params)
    assert svg.status_code == 200, svg.text
    assert svg.headers["content-type"].startswith("image/svg+xml")


def test_links_encode_paths_and_skip_simps():
    node = Node(
        SimpleNamespace(
            addr=0x10,
            size=8,
            block_id=0x10,
            is_simprocedure=False,
            instruction_addrs=(0x10, 0x14),
        )
    )
    LinkDDG("a b/file&name").annotate_node(node)
    query = parse_qs(urlsplit(node.pydot.get("URL")).query)
    assert query == {"filepath": ["a b/file&name"], "address": ["0x10"], "size": ["8"]}
    assert node.pydot.get("target") is None
    assert node.pydot.get("tooltip") == "Block 0x10 (2 instructions, 8 bytes)"
    # Rendered rows take precedence over a lifter's marker count, which can
    # combine a branch and delay slot or differ from VEX fallback assembly.
    node.content["asm"] = {"data": [{}, {}, {}]}
    LinkDDG("binary").annotate_node(node)
    assert node.pydot.get("tooltip") == "Block 0x10 (3 instructions, 8 bytes)"
    node.content["asm"]["data"] = [{}]
    node.obj.size = 1
    LinkDDG("binary").annotate_node(node)
    assert node.pydot.get("tooltip") == "Block 0x10 (1 instruction, 1 byte)"
    node.obj.is_simprocedure = True
    synthetic = Node(node.obj)
    LinkDDG("binary").annotate_node(synthetic)
    assert synthetic.pydot.get("URL") is None
    assert synthetic.pydot.get("tooltip") is None


@pytest.mark.parametrize("separator", ["&", "&amp;"])
def test_svg_urls_escape_once_without_changing_raw_links(tmp_path, separator):
    # Ubuntu's Graphviz 2.42.2 does not escape SVG URL ampersands. Check the
    # DOT passed to Graphviz as well as the SVG, since modern dot fixes it too.
    node = Node(CFGNode(0x10, 1, Mock(), block_id=0x10))
    url = f"/ddg?filepath=a%26b{separator}address=0x10{separator}size=1"
    node.pydot.set("URL", url)
    node.pydot.set("tooltip", "Block 0x10 (1 instruction, 1 byte)")
    graph = Graph(SimpleNamespace(graph=nx.DiGraph()), nodes={node})
    output = DotOutput(str(tmp_path / "block"), format="svg", dfs_rank=False)
    for _ in range(2):
        dot = output.generate(graph)
        assert "&amp;address=0x10&amp;size=1" in dot
        svg = ElementTree.fromstring((tmp_path / "block.svg").read_text())
        link = next(svg.iter("{http://www.w3.org/2000/svg}a"))
        assert link.get("{http://www.w3.org/1999/xlink}href") == (
            "/ddg?filepath=a%26b&address=0x10&size=1"
        )
    assert node.pydot.get("URL") == url
    raw = DotOutput(str(tmp_path / "block"), format="raw", dfs_rank=False).generate(
        graph
    )
    assert url in raw


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
