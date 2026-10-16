"""Smoke-test distributable packages without relying on the source checkout.

Rebuild the source archive into a wheel, then install both wheels independently
without dependencies. Check frozen provenance, installed package metadata and
the console entry point in isolated environments outside the repository. With
``--runtime``, additionally install declared dependencies and exercise actual
analysis and UI routes. This does not replace unit tests or the CFG golden corpus.
``--python-versions`` audits the original wheel's complete runtime across selected
interpreters, without rebuilding it or changing the development environment.
"""

from __future__ import annotations

import argparse
from email.parser import Parser
import json
import os
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
from zipfile import ZipFile


def _uv(*args: str, cwd: Path) -> str:
    """Run a uv command in the audit directory, preserving useful failure logs."""

    result = subprocess.run(["uv", *args], cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"uv {' '.join(args)} failed:\n{result.stdout}{result.stderr}"
        )
    return result.stdout.strip()


def _one(directory: Path, pattern: str) -> Path:
    """Reject missing or stale build outputs rather than picking an arbitrary file."""

    matches = list(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly one {pattern} in {directory}; found {len(matches)}"
        )
    return matches[0].resolve()


def _check_wheel_contents(wheel: Path) -> None:
    """Require licensing metadata/notices and reject development-only payloads."""

    with ZipFile(wheel) as archive:
        names = archive.namelist()
        for name in names:
            parts = Path(name).parts
            if any(part in {"tests", "scripts", "angr-binaries"} for part in parts):
                raise ValueError(f"Unexpected development file in wheel: {name}")
        metadata_files = [
            name for name in names if name.endswith(".dist-info/METADATA")
        ]
        if len(metadata_files) != 1:
            raise ValueError("Expected exactly one wheel METADATA file")
        metadata_path = metadata_files[0]
        metadata = Parser().parsestr(archive.read(metadata_path).decode("utf-8"))
        if metadata["License-Expression"] != "BSD-2-Clause":
            raise ValueError("Wheel must declare License-Expression: BSD-2-Clause")
        if set(metadata.get_all("License-File", [])) != {"LICENSE", "NOTICE"}:
            raise ValueError("Wheel must declare both LICENSE and NOTICE")
        # PEP 639 places these files under .dist-info/licenses, not the package.
        root = metadata_path.rsplit("/", 1)[0] + "/licenses/"
        for filename in ("LICENSE", "NOTICE"):
            name = root + filename
            if name not in names or not archive.read(name).strip():
                raise ValueError(f"Missing or empty wheel license file: {filename}")


def _check_install(
    wheel: Path,
    directory: Path,
    expected: dict[str, str],
    runtime: bool = False,
    python_version: str = sys.executable,
) -> None:
    """Verify a non-editable installation's metadata, import and console version.

    Normally skip dependencies: the version query must work without angr, a
    binary root or application initialization. ``runtime`` exercises a complete
    install too, catching incompatible dependency resolutions. Python ignores source
    paths and user site packages, preventing an editable install from masking a
    broken wheel. Each wheel gets its own environment to avoid installer caches
    skipping a same-version reinstall. An explicit interpreter request uses uv's
    normal discovery: reuse an installed version, downloading only when needed.
    """

    environment = directory / "environment"
    _uv("venv", "--python", python_version, str(environment), cwd=directory)
    binaries = environment / ("Scripts" if os.name == "nt" else "bin")
    python = binaries / ("python.exe" if os.name == "nt" else "python")
    interpreter = _uv(
        "run",
        "--no-project",
        "--python",
        str(python),
        "python",
        "--version",
        cwd=directory,
    )
    print(f"Testing {wheel.name} with {interpreter}", flush=True)
    _uv(
        "pip",
        "install",
        *(("--no-deps",) if not runtime else ()),
        "--python",
        str(python),
        str(wheel),
        cwd=directory,
    )
    query = (
        "import json,bingraph;from importlib.metadata import version;"
        "print(json.dumps({'version':bingraph.__version__,"
        "'commit':bingraph.__commit__,'metadata_version':version('pybingraph')}))"
    )
    actual = json.loads(
        _uv(
            "run",
            "--no-project",
            "--python",
            str(python),
            "python",
            "-I",
            "-c",
            query,
            cwd=directory,
        )
    )
    if actual != {**expected, "metadata_version": expected["version"]}:
        raise ValueError(
            f"Installed {wheel.name} has incorrect version/provenance: {actual}"
        )
    cli = binaries / ("bingraph.exe" if os.name == "nt" else "bingraph")
    output = _uv(
        "run",
        "--no-project",
        "--python",
        str(python),
        str(cli),
        "--version",
        cwd=directory,
    )
    if output != f"bingraph {expected['version']}":
        raise ValueError(f"Installed console version disagrees: {output}")
    if runtime:
        script = Path(__file__).resolve().with_name("runtime.py")
        _uv(
            "run",
            "--no-project",
            "--python",
            str(python),
            "python",
            "-I",
            str(script),
            "--expected-version",
            expected["version"],
            cwd=directory,
        )


def validate_distributions(
    directory: Path, expected: dict[str, str], runtime: bool = False
) -> None:
    """Validate exactly one wheel and source archive against source provenance.

    Build from the archive itself, not a Git checkout. The rebuilt wheel must
    preserve the original commit and version. Temporary environments and rebuilt
    files never touch the user's environment, source tree or golden artifacts.
    """

    wheel = _one(directory, "*.whl")
    source = _one(directory, "*.tar.gz")
    with TemporaryDirectory(prefix="bingraph-distributions-") as temporary:
        audit = Path(temporary)
        rebuilt = audit / "rebuilt"
        _uv("build", str(source), "--wheel", "--out-dir", str(rebuilt), cwd=audit)
        for index, candidate in enumerate((wheel, _one(rebuilt, "*.whl"))):
            _check_wheel_contents(candidate)
            install = audit / f"install-{index}"
            install.mkdir()
            _check_install(candidate, install, expected, runtime)


def validate_compatibility(
    directory: Path, expected: dict[str, str], versions: list[str]
) -> None:
    """Audit one built wheel on every requested Python, reporting all failures.

    Each interpreter gets a clean full installation, including native dependencies,
    and the same runtime probe used by release validation. No playground or golden
    files are needed. Installation failures count as incompatibilities, not skips.
    """

    wheel = _one(directory, "*.whl")
    _check_wheel_contents(wheel)
    failures = []
    with TemporaryDirectory(prefix="bingraph-compatibility-") as temporary:
        for index, version in enumerate(versions):
            install = Path(temporary) / str(index)
            install.mkdir()
            print(f"Checking Python {version}", flush=True)
            try:
                _check_install(wheel, install, expected, True, version)
            except (RuntimeError, ValueError) as error:
                failures.append(version)
                print(f"FAIL Python {version}: {error}", file=sys.stderr, flush=True)
            else:
                print(f"PASS Python {version}", flush=True)
    if failures:
        raise ValueError(f"Python compatibility failed: {', '.join(failures)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--provenance", type=Path, default=Path("version.json"))
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--runtime",
        action="store_true",
        help="audit a full runtime installation (requires Clang and dot)",
    )
    mode.add_argument(
        "--python-versions",
        nargs="+",
        help="audit the existing wheel on these Python versions (requires Clang and dot)",
    )
    args = parser.parse_args()
    expected = json.loads(args.provenance.read_text())
    if args.python_versions:
        validate_compatibility(args.dist, expected, args.python_versions)
    else:
        validate_distributions(args.dist, expected, args.runtime)
    print(
        f"Validated Python distributions for {expected['version']} ({expected['commit']})"
    )
