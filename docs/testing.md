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

## CFG and DDG Suites

`make goldens` runs both full graph suites and the independent `goldens-symbols`
audit; `make goldens-checkpoint` runs both curated graph suites. Use `goldens-cfg`,
`goldens-ddg`, `goldens-cfg-checkpoint`, or `goldens-ddg-checkpoint` to select one
graph type without running the symbol audit. `BINGRAPH_GOLDEN_CONFIGS` filters
both graph suites using comma-separated configuration names. Unset or empty
selects `cfg_mode_custom` and `ddg_default`; an explicit list selects only those
names, without enabling the other suite's default. Unknown names are errors.
CI explicitly selects both defaults for `make goldens-checkpoint`.

`goldens-symbols` compares file-owned function names, addresses and sizes against
`tests/playground_functions.csv`, which is its baseline rather than a separate
artifact tree. It has no promotion mode. Intentional symbol-discovery changes
require regenerating that CSV with `scripts/export_playground_functions_csv.py`
and reviewing the diff; this can also change the selected CFG corpus.

The frozen DDG manifest is `tests/playground_blocks.csv`: binary-relative path,
start address, byte size, checkpoint flag and selection rationale. Each selected
configuration runs all 3,000 blocks, or 100 curated blocks for checkpoints:

- `ddg_default`: inherit the actual memory/dependency defaults.
- `ddg_memory_may`: override only `memory=may`.
- `ddg_dependencies_raw`: override only `dependencies=raw`.

Add future configurations in `tests/goldens/configs.py`, the shared registry.
Candidates live under `tests/_actual/<config>/` and goldens under
`tests/goldens/<config>/`; basenames include binary, address and size.
Initial baselines require review like CFG candidates.
Promotion uses `BINGRAPH_GOLDEN_MODE=promote`, copies existing artifacts and never
rerenders. Earlier `ddg/` and policy-specific candidates are not moved or deleted
automatically; rerun the selected DDG suite to populate the named directories.

```bash
BINGRAPH_GOLDEN_CONFIGS=ddg_memory_may,ddg_dependencies_raw make goldens-ddg
BINGRAPH_GOLDEN_CONFIGS=cfg_mode_custom,ddg_default make goldens-checkpoint
```

To explicitly survey existing custom CFG goldens and refresh the manifest:

```bash
uv run python scripts/ddg.py --report /tmp/ddg-selection.json --write
```

Without `--write`, the command only surveys and reports. It does not discover new
CFGs or generate/promote DOT files. Selection balances architectures, limits
contributions per binary/function, and removes repeated instruction/dependency
shapes. Ordinary blocks have 4-64 instructions; protected regression cases may
be larger. Checkpoints favor small representatives of distinct architecture,
mode, instruction-group and dependency features. Existing checkpoint rows are
protected, so edit their flags deliberately when replacing a regression case.
The report lists surveyed unsupported blocks instead of hiding them; the survey
is bounded rather than an exhaustive compatibility audit. Normal golden runs
read the committed manifest without resampling or silently skipping failures.

CFG links change DOT/SVG node metadata only. Audit nodes, assembly and edges
after removing URL/target attributes before accepting a bulk link-only promotion.

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
BINGRAPH_GOLDEN_CONFIGS=cfg_mode_custom make goldens-cfg-checkpoint
```

To run every selected corpus entry:

```bash
BINGRAPH_GOLDEN_CONFIGS=cfg_mode_custom make goldens-cfg
```

For an isolated audit of demangled names, including instruction comments, use
`BINGRAPH_GOLDEN_CONFIGS=cfg_demangle_on make goldens-cfg`. Its artifacts are written
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
