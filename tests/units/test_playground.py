"""Playground setup supplies missing samples without replacing local fixtures."""

import os
from pathlib import Path
import shutil
import subprocess
import textwrap

import pytest


@pytest.mark.parametrize("state", ["fresh", "matching", "conflicting"])
def test_sample_setup_is_idempotent_and_preserves_local_files(tmp_path, state):
    root = tmp_path / "project"
    (root / "scripts").mkdir(parents=True)
    script = root / "scripts/fetch_playground.sh"
    shutil.copyfile(
        Path(__file__).resolve().parents[2] / script.relative_to(root), script
    )
    upstream = tmp_path / "upstream/examples/samples"
    upstream.mkdir(parents=True)
    (upstream / "ais3_crackme").write_bytes(b"binary\x00fixture")
    (upstream / "simple0").write_bytes(b"another fixture")
    # Stub only Git's network operations; exercise the real shell, archive/copy
    # operations and conflict handling without downloading or touching the corpus.
    tools = tmp_path / "bin"
    tools.mkdir()
    git = tools / "git"
    git.write_text(
        textwrap.dedent("""\
        #!/usr/bin/env bash
        set -eu
        if [ "$1" = -C ]; then shift 2; fi
        case "$1" in
          clone) mkdir -p "${!#}/.git" ;;
          init|fetch) ;;
          archive) tar -cf - -C "$FAKE_UPSTREAM" examples/samples ;;
          *) exit 1 ;;
        esac
        """)
    )
    git.chmod(0o755)
    samples = root / "angr-binaries/tests/samples"
    if state != "fresh":
        (root / "angr-binaries/.git").mkdir(parents=True)
        samples.mkdir(parents=True)
        (samples / "user-file").write_bytes(b"keep me")
        name = "simple0" if state == "conflicting" else "ais3_crackme"
        (samples / name).write_bytes(
            b"local changes"
            if state == "conflicting"
            else (upstream / name).read_bytes()
        )
    env = {
        **os.environ,
        "PATH": f"{tools}{os.pathsep}{os.environ['PATH']}",
        "FAKE_UPSTREAM": str(upstream.parents[1]),
        "TMPDIR": str(tmp_path),
    }
    result = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True
    )
    if state == "conflicting":
        assert result.returncode != 0 and "refusing to overwrite" in result.stderr
        assert (samples / "simple0").read_bytes() == b"local changes"
        assert not (samples / "ais3_crackme").exists()
    else:
        assert result.returncode == 0, result.stderr
        before = {
            path.name: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in samples.iterdir()
        }
        subprocess.run(["bash", str(script)], env=env, check=True, capture_output=True)
        assert before == {
            path.name: (path.read_bytes(), path.stat().st_mtime_ns)
            for path in samples.iterdir()
        }
        for sample in upstream.iterdir():
            assert (samples / sample.name).read_bytes() == sample.read_bytes()
    if state != "fresh":
        assert (samples / "user-file").read_bytes() == b"keep me"
    assert not list(tmp_path.glob("bingraph-samples.*"))
