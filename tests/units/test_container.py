"""Offline checks for safe container staging and failed-server cleanup.

Actual Docker/native analysis is exercised separately by make image and CI,
not multiplied across the fast unit suite or dependent on a running daemon.
"""

import importlib.util
from http.client import RemoteDisconnected
import json
from pathlib import Path
from zipfile import ZipFile

import pytest


@pytest.fixture
def container_tools():
    path = Path(__file__).resolve().parents[2] / "scripts/container.py"
    spec = importlib.util.spec_from_file_location("container_tools", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def expected():
    return {"version": "0.2.3", "commit": "a" * 40}


def test_build_stages_only_audited_inputs(
    container_tools, tmp_path, monkeypatch, expected
):
    wheel = tmp_path / "pybingraph.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr("bingraph/version.json", json.dumps(expected))
    (tmp_path / "private.env").write_text("not build input")
    commands, contexts = [], []

    def run(command, **kwargs):
        assert kwargs["check"]
        commands.append(command)
        if command[0] == "uv":
            assert kwargs["cwd"] == container_tools.ROOT
            assert {"--locked", "--no-emit-project", "--no-dev"} <= set(command)
            Path(command[-1]).write_text("angr==10.0.4 --hash=sha256:example\n")
        else:
            context = Path(command[-1])
            contexts.append(context)
            assert {path.name for path in context.iterdir()} == {
                "Dockerfile",
                "LICENSE",
                "NOTICE",
                "requirements.txt",
                wheel.name,
            }
            assert (context / wheel.name).read_bytes() == wheel.read_bytes()

    monkeypatch.setattr(container_tools.subprocess, "run", run)
    container_tools.build("bingraph:test", tmp_path, expected, "linux/arm64")
    assert "VERSION=0.2.3" in commands[1] and "REVISION=" + "a" * 40 in commands[1]
    assert commands[1][-3:-1] == ["--platform", "linux/arm64"]
    assert len(contexts) == 1 and not contexts[0].exists()


@pytest.mark.parametrize("problem", ["missing", "multiple", "provenance"])
def test_invalid_wheels_fail_before_docker(
    container_tools, tmp_path, expected, problem
):
    if problem != "missing":
        for index in range(2 if problem == "multiple" else 1):
            with ZipFile(tmp_path / f"package{index}.whl", "w") as archive:
                archive.writestr("bingraph/version.json", "{}")
    with pytest.raises(ValueError, match="exactly one|provenance"):
        container_tools.build("bingraph:test", tmp_path, expected, None)


@pytest.mark.parametrize("transient", [False, True])
def test_failed_smoke_removes_its_container(
    container_tools, monkeypatch, expected, transient
):
    commands = []
    attempts = []

    def docker(*args):
        commands.append(args)
        if "--version" in args:
            return "bingraph 0.2.3"
        if args[0] == "run":
            assert {
                "--read-only",
                "--cap-drop=ALL",
                "--security-opt=no-new-privileges",
            } <= set(args)
            assert "127.0.0.1::8000" in args
            assert next(
                value for value in args if value.startswith("type=bind")
            ).endswith(",readonly")
            return "our-container"
        if args[0] == "port":
            return "127.0.0.1:12345"
        return ""

    def unavailable(*args, **kwargs):
        attempts.append(args)
        if transient and len(attempts) == 1:
            # A published Docker port can accept connections before Uvicorn is ready.
            raise RemoteDisconnected("Server is still starting")
        raise RuntimeError("HTTP probe failed")

    monkeypatch.setattr(container_tools, "docker", docker)
    monkeypatch.setattr(container_tools.subprocess, "run", lambda *a, **k: None)
    monkeypatch.setattr(container_tools, "urlopen", unavailable)
    monkeypatch.setattr(container_tools.time, "sleep", lambda *args: None)
    with pytest.raises(RuntimeError, match="HTTP probe failed"):
        container_tools.smoke_test("bingraph:test", expected)
    assert commands[-1] == ("rm", "--force", "our-container")
    assert len(attempts) == (2 if transient else 1)
