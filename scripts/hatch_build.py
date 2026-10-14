"""Freeze version provenance into wheels and source distributions, not sources.

Hatch loads this script through the explicit custom-hook path in pyproject.toml.

The code version source supplies package metadata. This hook carries the same
version and SHA into artifacts so installing or rebuilding an sdist needs no
Git history. Editable wheels intentionally omit frozen data to keep tracking
the source checkout. Temporary payloads cannot stale or dirty that checkout.
"""

import json
from pathlib import Path
import runpy
from tempfile import TemporaryDirectory

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class VersionBuildHook(BuildHookInterface):
    """Attach version data using Hatch's target-aware forced inclusion paths."""

    def initialize(self, version, build_data):
        """Create isolated data only for distributable, non-editable artifacts."""

        if self.target_name == "wheel" and version == "editable":
            return
        namespace = runpy.run_path(str(Path(self.root) / "src/bingraph/version.py"))
        package_version, commit = namespace["get_version_info"]()
        if package_version != self.metadata.version:
            raise RuntimeError(
                "Git changed during build; retry to keep versions consistent"
            )
        self._version_directory = TemporaryDirectory(prefix="bingraph-version-")
        payload = Path(self._version_directory.name) / "version.json"
        payload.write_text(json.dumps({"version": package_version, "commit": commit}))
        destination = (
            "bingraph/version.json"
            if self.target_name == "wheel"
            else "src/bingraph/version.json"
        )
        build_data["force_include"][str(payload)] = destination

    def finalize(self, version, build_data, artifact_path):
        """Remove the private payload after Hatch has assembled the artifact."""

        directory = getattr(self, "_version_directory", None)
        if directory is not None:
            directory.cleanup()
