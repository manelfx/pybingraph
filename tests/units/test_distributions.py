"""Packaging smoke checks and non-destructive release attachment decisions."""

import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import textwrap
from zipfile import ZipFile

import pytest


@pytest.fixture
def checker():
    path = Path(__file__).resolve().parents[2] / "scripts/distributions.py"
    spec = importlib.util.spec_from_file_location("distribution_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("count", [0, 2])
def test_missing_or_stale_wheels_are_rejected(checker, tmp_path, count):
    for index in range(count):
        (tmp_path / f"package-{index}.whl").touch()
    with pytest.raises(ValueError, match="exactly one"):
        checker._one(tmp_path, "*.whl")


@pytest.mark.parametrize(
    "name",
    [
        "tests/test_api.py",
        "bingraph/tests/test_api.py",
        "scripts/build.py",
        "angr-binaries/sample",
    ],
)
def test_wheels_reject_development_payloads(checker, tmp_path, name):
    wheel = tmp_path / "package.whl"
    with ZipFile(wheel, "w") as archive:
        archive.writestr(name, "not distributable")
    with pytest.raises(ValueError, match="development file"):
        checker._check_wheel_contents(wheel)


@pytest.fixture
def licensed_wheel(tmp_path):
    """Build small archives exercising licensing without installing packages."""

    def build(expression="BSD-2-Clause", declared=("LICENSE", "NOTICE"), missing=None):
        wheel = tmp_path / "package.whl"
        root = "pybingraph-0.1.10.dist-info/"
        metadata = "Metadata-Version: 2.4\nName: pybingraph\nVersion: 0.1.10\n"
        if expression is not None:
            metadata += f"License-Expression: {expression}\n"
        metadata += "".join(f"License-File: {name}\n" for name in declared)
        with ZipFile(wheel, "w") as archive:
            archive.writestr(root + "METADATA", metadata)
            for name in ("LICENSE", "NOTICE"):
                if name != missing:
                    archive.writestr(root + "licenses/" + name, "License notice\n")
        return wheel

    return build


def test_wheel_contains_project_and_upstream_license_notices(checker, licensed_wheel):
    checker._check_wheel_contents(licensed_wheel())


@pytest.mark.parametrize("expression", [None, "MIT"])
def test_wheel_rejects_missing_or_incorrect_license_expression(
    checker, licensed_wheel, expression
):
    with pytest.raises(ValueError, match="License-Expression"):
        checker._check_wheel_contents(licensed_wheel(expression=expression))


@pytest.mark.parametrize("name", ["LICENSE", "NOTICE"])
def test_wheel_rejects_missing_license_notices(checker, licensed_wheel, name):
    with pytest.raises(ValueError, match="Missing or empty wheel license file"):
        checker._check_wheel_contents(licensed_wheel(missing=name))


def test_wheel_rejects_undeclared_upstream_notice(checker, licensed_wheel):
    with pytest.raises(ValueError, match="declare both LICENSE and NOTICE"):
        checker._check_wheel_contents(licensed_wheel(declared=("LICENSE",)))


@pytest.mark.parametrize("python_version", [sys.executable, "3.14"])
def test_complete_install_requests_dependencies_and_runtime_probe(
    checker, tmp_path, monkeypatch, python_version
):
    expected = {"version": "0.1.10", "commit": "a" * 40}
    commands = []

    def uv(*args, cwd):
        commands.append(args)
        if "-c" in args:
            return json.dumps({**expected, "metadata_version": expected["version"]})
        return "bingraph 0.1.10" if "--version" in args else ""

    monkeypatch.setattr(checker, "_uv", uv)
    checker._check_install(
        tmp_path / "package.whl",
        tmp_path,
        expected,
        runtime=True,
        python_version=python_version,
    )
    assert commands[0][:3] == ("venv", "--python", python_version)
    installation = next(command for command in commands if command[0] == "pip")
    assert "--no-deps" not in installation
    assert "--expected-version" in commands[-1]
    assert any(argument.endswith("runtime.py") for argument in commands[-1])


@pytest.mark.parametrize("failing_version", [None, "3.13"])
def test_compatibility_reuses_wheel_and_checks_all_versions(
    checker, licensed_wheel, monkeypatch, capsys, failing_version
):
    wheel = licensed_wheel()
    expected = {"version": "0.1.10", "commit": "a" * 40}
    installations = []

    def check(candidate, directory, provenance, runtime, python_version):
        assert candidate == wheel.resolve() and provenance == expected and runtime
        assert directory.is_dir()
        installations.append((directory, python_version))
        if python_version == failing_version:
            raise RuntimeError("native dependency unavailable")

    monkeypatch.setattr(checker, "_check_install", check)
    versions = ["3.12", "3.13", "3.14", "3.15"]
    if failing_version:
        with pytest.raises(ValueError, match="Python compatibility failed: 3.13"):
            checker.validate_compatibility(wheel.parent, expected, versions)
    else:
        checker.validate_compatibility(wheel.parent, expected, versions)
    assert [version for directory, version in installations] == versions
    assert len({directory for directory, version in installations}) == len(versions)
    assert all(not directory.exists() for directory, version in installations)
    output = capsys.readouterr()
    assert "PASS Python 3.15" in output.out
    if failing_version:
        assert "FAIL Python 3.13: native dependency unavailable" in output.err


@pytest.mark.parametrize("field", ["version", "commit", "metadata_version", "cli"])
def test_install_check_rejects_provenance_or_console_drift(
    checker, tmp_path, monkeypatch, field
):
    expected = {"version": "0.1.10", "commit": "a" * 40}
    actual = {**expected, "metadata_version": expected["version"]}
    if field != "cli":
        actual[field] = "incorrect"

    def uv(*args, cwd):
        if "-c" in args:
            return json.dumps(actual)
        return "incorrect" if field == "cli" else ""

    monkeypatch.setattr(checker, "_uv", uv)
    with pytest.raises(ValueError, match="provenance|console version"):
        checker._check_install(tmp_path / "package.whl", tmp_path, expected)


@pytest.mark.parametrize(
    "existing,immutable,allowed",
    [
        ("none", False, True),
        ("matching", False, True),
        ("matching", True, True),
        ("different", False, False),
        ("none", True, False),
    ],
)
def test_release_upload_preserves_existing_assets(
    tmp_path, monkeypatch, existing, immutable, allowed
):
    workflow = Path(__file__).resolve().parents[2] / ".github/workflows/packages.yml"
    script = workflow.read_text().split("uv run --no-project python - <<'PY'\n", 1)[1]
    script = textwrap.dedent(script.rsplit("          PY", 1)[0])
    dist = tmp_path / "dist"
    dist.mkdir()
    files = [dist / "package.whl", dist / "package.tar.gz"]
    for path in files:
        path.write_bytes(b"package bytes")
    assets = (
        []
        if existing == "none"
        else [
            {
                "name": path.name,
                "digest": "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()
                if existing == "matching"
                else "sha256:different",
            }
            for path in files
        ]
    )
    (tmp_path / "release.json").write_text(
        json.dumps({"assets": assets, "immutable": immutable})
    )
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("RELEASE_TAG", "v0.1.10")
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: calls.append(args))
    if allowed:
        exec(compile(script, str(workflow), "exec"), {})
        assert len(calls) == (2 if existing == "none" else 0)
    else:
        with pytest.raises(SystemExit, match="refusing to overwrite|draft"):
            exec(compile(script, str(workflow), "exec"), {})
        assert not calls
