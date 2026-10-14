"""Exercise the installed package's real CLI, API, template and CFG renderer.

Compile a tiny x86-64 ELF object with Clang's cross target, so the same fixture
works on macOS and Linux without a playground checkout or cross-linker. Run this
script with the audited wheel's isolated interpreter and installed dependencies;
it must never import Bingraph from the source checkout. Clang and Graphviz are
test tools, not extra requirements for installing the Python package.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
from tempfile import TemporaryDirectory

import bingraph
from bingraph.api.app import create_app
from bingraph.helpers import Settings
import bingraph.helpers.settings as settings_module
from fastapi.testclient import TestClient


def smoke_test(expected_version: str) -> None:
    """Check actual native analysis and both frontend paths in a temporary root."""

    assert bingraph.__version__ == expected_version
    assert (
        not Path(bingraph.__file__)
        .resolve()
        .is_relative_to(Path(__file__).resolve().parents[1])
    )
    compiler = shutil.which("clang")
    if compiler is None or shutil.which("dot") is None:
        raise RuntimeError("Runtime packaging audit requires Clang and Graphviz dot")
    with TemporaryDirectory(prefix="bingraph-runtime-") as temporary:
        root = Path(temporary)
        source, binary = root / "sample.c", root / "sample.o"
        source.write_text(
            "int inspected(int value) { if (value) return value + 1; return -1; }\n"
        )
        subprocess.run(
            [
                compiler,
                "--target=x86_64-linux-gnu",
                "-O0",
                "-c",
                str(source),
                "-o",
                str(binary),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        settings_module._settings = Settings.model_construct(
            root=root, server=None, client=None, comments=False, log_level="ERROR"
        )
        with TestClient(create_app()) as client:
            response = client.get("/api/symtab", params={"filepath": binary.name})
            assert response.status_code == 200, response.text
            symbol = next(
                item for item in response.json()["items"] if item["name"] == "inspected"
            )
            params = {"filepath": binary.name, "function": hex(symbol["addr"])}
            response = client.get("/symtab", params={"filepath": binary.name})
            assert response.status_code == 200 and "inspected" in response.text
            assert "text/html" in response.headers["content-type"]
            for format in ("raw", "dot", "xdot"):
                response = client.get("/api/cfg", params={**params, "format": format})
                assert response.status_code == 200, response.text
                assert (
                    "digraph" in response.json()["graph"]
                    and "inspected" in response.json()["graph"]
                )
            response = client.get("/cfg", params=params)
            assert response.status_code == 200 and "<svg" in response.text
            assert "image/svg+xml" in response.headers["content-type"]
            response = client.get("/api/cfg", params={**params, "format": "png"})
            assert (
                response.status_code == 400
                and "Unsupported CFG format" in response.text
            )
            assert (
                client.get("/openapi.json").json()["info"]["version"]
                == expected_version
            )

        # Invoke the installed console script in a separate process as a user would.
        cli = Path(sys.executable).parent / "bingraph"
        result = subprocess.run(
            [
                "uv",
                "run",
                "--no-project",
                "--python",
                sys.executable,
                str(cli),
                "--root",
                str(root),
                "--no-comments",
                "--log-level",
                "ERROR",
                "client",
                "--endpoint",
                "/cfg",
                "--filepath",
                binary.name,
                "--function",
                params["function"],
                "--format",
                "raw",
            ],
            cwd=root,
            check=True,
            capture_output=True,
            text=True,
        )
        assert "digraph" in json.loads(result.stdout)["graph"]
    print(f"Installed runtime smoke test passed for {expected_version}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-version", required=True)
    smoke_test(parser.parse_args().expected_version)
