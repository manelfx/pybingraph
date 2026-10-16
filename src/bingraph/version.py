"""One Git-derived version for development, package metadata and images.

An explicit baseline prevents ordinary release tags from renumbering
existing commits. Count all commits reachable from HEAD but not the baseline,
including merge commits; branch names have no effect. Built distributions carry
the resulting version and SHA as package data, so they never need Git at runtime.
Editable checkouts do not carry that data and query their current HEAD instead.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import subprocess

BASE_VERSION = (0, 2, 0)
# Revision zero belongs to this fixed version baseline, not a moving HEAD.
BASE_COMMIT = "5129b28ec26160fa47f2b712f05c2ecb84d3802c"


def get_version_info(root: Path | None = None) -> tuple[str, str]:
    """Return ``(major.minor.revision, full_commit_sha)`` without guessing.

    Prefer frozen distribution data; otherwise require this package's own Git
    checkout and complete history containing the baseline. Pin HEAD once
    so concurrent commits cannot mix a revision count with another SHA. Neither
    dirty working trees nor tags change the numeric version. ``root`` selects
    an explicit source checkout for build tooling and isolated tests.
    """

    package = Path(__file__).resolve().parent if root is None else root / "src/bingraph"
    frozen = package / "version.json"
    if frozen.is_file():
        info = json.loads(frozen.read_text())
        version, commit = info["version"], info["commit"]
        if not isinstance(version, str) or not re.fullmatch(r"\d+\.\d+\.\d+", version):
            raise RuntimeError("Invalid packaged bingraph version")
        if not isinstance(commit, str) or not re.fullmatch(r"[0-9a-f]{40}", commit):
            raise RuntimeError("Invalid packaged bingraph commit")
        return version, commit

    root = package.parents[1]
    if not (root / ".git").exists():
        raise RuntimeError(
            "Bingraph needs its Git checkout or a built distribution with version data"
        )

    def git(*args: str) -> str:
        """Run bounded, read-only queries in the package's checkout, not cwd."""

        try:
            return subprocess.run(
                ["git", "-C", str(root), *args],
                check=True,
                capture_output=True,
                text=True,
                timeout=5,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError) as error:
            raise RuntimeError(
                "Cannot determine bingraph version; Git and complete history "
                f"containing baseline {BASE_COMMIT} are required"
            ) from error

    if git("rev-parse", "--is-shallow-repository") != "false":
        raise RuntimeError(
            "Bingraph version requires complete history; unshallow the clone"
        )
    commit = git("rev-parse", "HEAD")
    git("merge-base", "--is-ancestor", BASE_COMMIT, commit)
    distance = int(git("rev-list", "--count", f"{BASE_COMMIT}..{commit}"))
    major, minor, revision = BASE_VERSION
    return f"{major}.{minor}.{revision + distance}", commit


def get_version() -> str:
    """Supply Hatch metadata from the same calculation used by Python and CLI."""

    return get_version_info()[0]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json", action="store_true", help="include the full commit SHA"
    )
    args = parser.parse_args()
    version, commit = get_version_info()
    print(json.dumps({"version": version, "commit": commit}) if args.json else version)
