# Testing

The tooling and test-selection rules in [AGENTS.md](../AGENTS.md#python-tooling)
apply to this guide. Golden ownership and promotion rules below are mandatory.
See [Git](git.md) for working-tree checkpoints and [Releases](releases.md) for
packaging audits.

Use the [Makefile](../Makefile) for local checks and
[packages.yml](../.github/workflows/packages.yml) for the current CI job order and
runner configuration. Keep procedures here aligned with those implementations,
rather than maintaining a second list of dependency versions or fixture pins.

## Development Checks and Fixtures

Install locked development dependencies with `uv sync --locked --extra dev`.
`make check` validates the Python badge, formatting, lint, source typing and unit
tests.

Some unit and golden tests require the playground. CI checks out the pinned
`angr/binaries` revision from the packages workflow, then runs
[`scripts/fetch_playground.sh`](../scripts/fetch_playground.sh) to fetch pinned
supplemental samples. The script preserves existing fixture content rather than
overwriting it.

Package builds run `make check` followed by `make goldens-checkpoint` before
building distributions. Failed checkpoint candidates are uploaded for review;
CI never promotes goldens.

## Golden Artifacts

- Do not modify user-managed golden artifacts or `tests/_actual/` unless the
  user explicitly requests generation, promotion, or cleanup.
- Preserve the current working tree and validate focused checkpoint goldens
  when making CFG changes.

### Golden CFG Regression Tests

Golden tests render CFGs for the corpus in
[`tests/playground_functions.csv`](../tests/playground_functions.csv). Compare
mode writes fresh candidates under `tests/_actual/` and compares them with the
committed artifacts under `tests/goldens/`.

The complete corpus can take a long time. Use the curated checkpoint suite for
quick custom-CFG regression coverage:

```bash
BINGRAPH_GOLDEN_CONFIGS=cfg_mode_custom make goldens-checkpoint
```

To run every selected corpus entry:

```bash
BINGRAPH_GOLDEN_CONFIGS=cfg_mode_custom make goldens
```

For an isolated audit of demangled names, including instruction comments, use
`BINGRAPH_GOLDEN_CONFIGS=cfg_demangle_on make goldens`. Its artifacts are written
under `tests/_actual/cfg_demangle_on/`; existing configurations and goldens
keep demangling disabled.

Promotion never rerenders CFGs. After reviewing a successful compare-mode run,
copy its existing candidates into the golden tree with:

```bash
BINGRAPH_GOLDEN_CONFIGS=cfg_mode_custom make goldens-promote
```

Review the resulting Git diff before committing promoted artifacts. Additional
selection controls are documented at the top of
[`tests/goldens/test_cfg_goldens.py`](../tests/goldens/test_cfg_goldens.py),
including `BINGRAPH_GOLDEN_MIN_BBS` and `BINGRAPH_GOLDEN_LIMIT`.

## Python Compatibility

Run `make compatibility` to smoke-test one wheel on the Makefile's
`PYTHON_VERSIONS`, or select versions with:

```bash
make compatibility PYTHON_VERSIONS="3.12 3.14"
```

Clang and Graphviz are required. Checks use temporary environments and download
missing Python interpreters without upgrading existing ones. They test installed
package behavior without multiplying the complete unit or golden suites across
Python versions.

The packages workflow audits its build interpreter during packaging and reuses
the same wheel for the remaining versions selected by `make python-matrix`.
After changing `PYTHON_VERSIONS`, run `make badges` and commit the SVG with the
Makefile. `make badges-check` validates it without writing; both `make check`
and CI run that validation.
