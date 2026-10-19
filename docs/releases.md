# Releases

This guide covers versioning, packaging and publication procedures. See
[AGENTS.md](../AGENTS.md) for the instruction entry point and [Git](git.md) for
checkpoint and commit rules.
[packages.yml](../.github/workflows/packages.yml) is authoritative for CI gates
and publication conditions; [Testing](testing.md) covers local checks.

## Versioning

- Versions are `major.minor.revision`; constants live only in
  `src/bingraph/version.py`. Revision counts commits since `BASE_COMMIT`, including
  merges, plus the baseline revision. Change major/minor via `BASE_VERSION`;
  preserve the baseline and do not add routine version-bump commits or hooks.
- Versions are commit-based, not branch/tag-based; the SHA is the exact identity.
  Require full Git history (`fetch-depth: 0` in CI). Built distributions must
  freeze version/SHA and work without Git; editable installs track HEAD.
- Publish official releases from `main` using annotated `v<version>` tags matching
  the calculated version. The tagged commit must be reachable from `origin/main`,
  which must already contain the release workflow and packaging changes.

Fast-forwarding `main` to `dev` preserves the version because both then reference
the same commit. A merge creates a new commit and changes the revision. Tags name
an existing version; creating a release does not calculate a new one. The README
documents CLI and Python commands for inspecting the version and source commit.

## Preflight

Before a release, run `make check`, applicable goldens and packaging audits using
an empty build output directory. Only run the full golden corpus when requested,
as required by the working agreements. The runtime audit requires Clang and
Graphviz; it does not replace unit or golden tests.

Run the following from the repository root with `dist/` empty or absent:

```bash
uv run --no-project python src/bingraph/version.py --json > version.json
uv build
uv run --no-project python scripts/distributions.py
uv run --no-project python scripts/distributions.py --runtime
```

Do not delete existing build output without checking ownership. For an isolated
audit, use a fresh temporary directory with `uv build --out-dir`, and pass its
paths through the checker's `--dist` and `--provenance` options.

## Draft and Publication

1. Commit and push the approved changes to `dev`, and verify CI succeeds.
2. Fast-forward `main` to the validated commit and push it. Confirm its package
   validation succeeds before continuing.
3. Calculate the version at that commit and create an annotated `v<version>` tag.
   Push the tag and create a draft release with reviewed release notes. Do not
   move a published release tag to different code.
4. Run the Python packages workflow from `main` with `release_tag` set to that tag.
   Leave `test_pypi` disabled unless a trial upload is explicitly intended.
   For example, replace `v<version>` below with the actual tag:

   ```bash
   gh workflow run packages.yml --ref main -f release_tag='v<version>'
   ```

5. Verify all validation jobs and the attachment job pass. Check the wheel and
   source distribution versions, commit provenance and attached hashes. Keep
   the release in draft while its description or assets are being reviewed.
6. Publish only after approval. With immutable releases enabled, attachments must
   already be complete; existing assets must never be replaced with different
   files. Identical files are skipped on retries.
7. Review the publication workflow triggered by the published release. Approve
   the production PyPI environment when ready, then verify PyPI file hashes and
   Docker image tags and platforms. The wheel and source archive on PyPI must
   match the audited distributions, not a rebuild.

Draft preparation validates containers but does not publish them. Normal pushes
build and validate distributions and containers without publishing production
packages. Python distributions are retained as workflow artifacts; development
container images are not retained. A GitHub release also shows GitHub-generated
source archives; Docker images live in GHCR, not in the release's asset list.

## Containers

`make image` builds and HTTP-smoke-tests a local native image using a temporary
wheel and hash-locked runtime export. `IMAGE=...` overrides its name;
`IMAGE_PLATFORM=linux/amd64` selects a different platform. Docker and host Clang
are required; the resulting image needs neither Git nor Clang.

The packages workflow tests native AMD64 and ARM64 images. Published releases
produce customer-facing version and SHA tags after both platforms and release
attachments pass validation. `latest` tracks GitHub's latest stable release;
rerunning an older release must not move it backwards.

Preserve version image tags: differing existing images fail instead of being
replaced. Refresh the pinned Docker base digest deliberately when updating it.
Preserve `LICENSE` and upstream attributions in `NOTICE` in the image and Python
distributions.

After first publication, verify that the GHCR package is public and permits
anonymous pulls. Change its visibility in package settings only if necessary;
a public repository alone is not a reason to assume anonymous image access.

## PyPI Configuration and Trials

PyPI publishing uses Trusted Publishing (OIDC), never stored API tokens. Register
`pybingraph` on PyPI and TestPyPI with owner `manelfx`, repository `pybingraph`,
workflow `packages.yml`, and environments `pypi` and `testpypi`, respectively.
TestPyPI needs its own account and publisher.

Keep manual approval on the `pypi` GitHub environment; allow self-review for a sole
maintainer. Restrict `pypi` to `v*` tags and `testpypi` to the `dev` and `main`
branches. Only published stable releases upload to PyPI, after validation and
attachment gates; development pushes and drafts never do.

Trial uploads use the packages workflow's manual `test_pypi` checkbox, or:

```bash
gh workflow run packages.yml --ref dev -F test_pypi=true
```

Leave `release_tag` empty to avoid touching release assets. Verify the trial
before approving production publication.

Both publishing jobs upload existing audited distributions without rebuilding.
The staging check skips identical indexed files, rejects differing hashes and
fails on index errors other than 404. Do not enable `skip-existing` on the upload
action: retries must verify and preserve existing files, not hide conflicts.
