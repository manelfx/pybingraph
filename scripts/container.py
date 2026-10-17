"""Build or audit the installed Bingraph container, without publishing it.

Builds stage only one prebuilt wheel, hash-locked runtime requirements, the
Dockerfile and license notices in a temporary context. Smoke checks compile an
ELF fixture on the host, mount it read-only and exercise the actual HTTP server
with a non-root user and read-only container filesystem. Docker is required for
both commands; only the smoke check needs host Clang. No playground is used.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from tempfile import TemporaryDirectory
import time
from urllib.error import URLError
from urllib.parse import urlencode
from urllib.request import urlopen
from zipfile import ZipFile

ROOT = Path(__file__).resolve().parents[1]


def docker(*args: str) -> str:
    """Run a bounded Docker operation, surfacing daemon errors to the caller."""

    result = subprocess.run(
        ["docker", *args], capture_output=True, text=True, timeout=120
    )
    if result.returncode:
        raise RuntimeError(f"Docker {args[0]} failed:\n{result.stdout}{result.stderr}")
    return result.stdout.strip()


def build(
    image: str, dist: Path, expected: dict[str, str], platform: str | None
) -> None:
    """Install the existing wheel with lockfile dependencies, never source code.

    Reject ambiguous outputs or mixed wheel/provenance inputs before contacting
    Docker. A temporary allowlisted context avoids sending the checkout, local
    configuration, golden artifacts or binaries to the build daemon.
    """

    wheels = list(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise ValueError("Container build requires exactly one wheel")
    with ZipFile(wheels[0]) as archive:
        if json.loads(archive.read("bingraph/version.json")) != expected:
            raise ValueError("Wheel and container provenance disagree")
    with TemporaryDirectory(prefix="bingraph-image-") as temporary:
        context = Path(temporary)
        for source in (
            wheels[0],
            ROOT / "LICENSE",
            ROOT / "NOTICE",
            ROOT / "docker/Dockerfile",
        ):
            shutil.copy2(source, context / source.name)
        subprocess.run(
            [
                "uv",
                "export",
                "--locked",
                "--no-dev",
                "--no-default-groups",
                "--no-emit-project",
                "--no-header",
                "--no-annotate",
                "--quiet",
                "--output-file",
                str(context / "requirements.txt"),
            ],
            cwd=ROOT,
            check=True,
        )
        subprocess.run(
            [
                "docker",
                "build",
                "--tag",
                image,
                "--build-arg",
                f"VERSION={expected['version']}",
                "--build-arg",
                f"REVISION={expected['commit']}",
                *(["--platform", platform] if platform else []),
                str(context),
            ],
            check=True,
        )


def smoke_test(image: str, expected: dict[str, str]) -> None:
    """Check provenance, demangling and real analysis behind a loopback-only port.

    Use a random host port to avoid conflicting with a developer's server. Keep
    scratch state in /tmp, never in the mounted input directory. Always remove
    our own container, including when startup or an HTTP assertion fails.
    """

    assert (
        docker("run", "--rm", image, "--version") == f"bingraph {expected['version']}"
    )
    with TemporaryDirectory(prefix="bingraph-container-") as temporary:
        root = Path(temporary)
        root.chmod(0o755)  # The container UID must be able to read the host fixture.
        source, binary = root / "sample.c", root / "sample.o"
        source.write_text("int inspected(int n) { if (n) return n + 1; return -1; }\n")
        subprocess.run(
            [
                "clang",
                "--target=x86_64-linux-gnu",
                "-O0",
                "-c",
                str(source),
                "-o",
                str(binary),
            ],
            check=True,
        )
        original = {path.name: path.read_bytes() for path in root.iterdir()}
        container = docker(
            "run",
            "--detach",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            "--tmpfs",
            "/tmp:rw,nosuid,nodev,size=128m",
            "--mount",
            f"type=bind,src={root},dst=/binaries,readonly",
            "--publish",
            "127.0.0.1::8000",
            image,
        )
        try:
            port = docker("port", container, "8000/tcp").rsplit(":", 1)[1]

            def get(path: str, **params: str) -> bytes:
                """Request a real published port, bounding every HTTP operation."""

                with urlopen(
                    f"http://127.0.0.1:{port}{path}?{urlencode(params)}", timeout=10
                ) as response:
                    return response.read()

            deadline = time.monotonic() + 60
            while True:
                try:
                    schema = json.loads(get("/openapi.json"))
                    break
                except (URLError, ConnectionError, TimeoutError):
                    if time.monotonic() >= deadline:
                        raise TimeoutError("Container server did not become ready")
                    time.sleep(0.5)
            assert schema["info"]["version"] == expected["version"]
            symbols = json.loads(get("/api/symtab", filepath="sample.o"))["items"]
            symbol = next(item for item in symbols if item["name"] == "inspected")
            params = {"filepath": "sample.o", "function": hex(symbol["addr"])}
            assert b"inspected" in get("/symtab", filepath="sample.o")
            assert b"digraph" in get("/api/cfg", **params, format="raw")
            assert b"<svg" in get("/cfg", **params)
            query = (
                "import json,os,bingraph,cxxfilt;from pathlib import Path;"
                "assert os.getuid()!=0;assert cxxfilt.demangle('_Z3foov')=='foo()';"
                "assert all((Path('/usr/share/doc/pybingraph')/p).stat().st_size "
                "for p in ('LICENSE','NOTICE'));"
                "print(json.dumps({'version':bingraph.__version__,'commit':bingraph.__commit__}))"
            )
            assert (
                json.loads(docker("exec", container, "python", "-c", query)) == expected
            )
            assert {path.name: path.read_bytes() for path in root.iterdir()} == original
        except Exception:
            # Docker puts the server's stderr logs on stderr too; stream both.
            subprocess.run(["docker", "logs", container], check=False, timeout=10)
            raise
        finally:
            docker("rm", "--force", container)
    print(f"Container smoke test passed: {image} ({expected['version']})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("build", "check"))
    parser.add_argument("--image", default="pybingraph:local")
    parser.add_argument("--dist", type=Path, default=Path("dist"))
    parser.add_argument("--provenance", type=Path, default=Path("version.json"))
    parser.add_argument("--platform", choices=("linux/amd64", "linux/arm64"))
    args = parser.parse_args()
    expected = json.loads(args.provenance.read_text())
    if args.command == "build":
        build(args.image, args.dist, expected, args.platform)
    else:
        smoke_test(args.image, expected)
