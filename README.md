# pybingraph

`pybingraph` is a binary-inspection service and library built on
[angr](https://angr.io/). It reads function symbols from a binary and renders
per-function control-flow graphs (CFGs) as SVG, DOT, or other Graphviz output.

The default CFG strategy starts with angr's bounded `CFGFast` analysis. Custom
mode retains that graph when it is well-formed and applies localized repairs
when it detects decoding gaps, missing direct successors, stale split blocks,
or selected indirect-jump targets.

## Requirements

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- The Graphviz `dot` executable for rendered image formats

On macOS:

```bash
brew install graphviz
```

## Setup

Install the runtime dependencies:

```bash
uv sync
```

For development tools such as pytest, Ruff, and Ty:

```bash
uv sync --extra dev
```

### Playground binaries

The optional playground corpus is used by the golden CFG tests and the examples
below. Fetch it with:

```bash
./scripts/fetch_playground.sh
```

It is cloned into `./angr-binaries/`; the binaries themselves live under
`./angr-binaries/tests/`.

## Run Bingraph

All binary paths are relative to `--root` and are constrained to remain below
that directory.

### Server

Start the local web server:

```bash
uv run bingraph --root ./angr-binaries/tests server
```

Then open a symbol table or CFG in a browser:

```text
http://127.0.0.1:8000/symtab?filepath=samples/ais3_crackme
http://127.0.0.1:8000/cfg?filepath=samples/ais3_crackme&function=0x40043c
```

Useful server options include `--port`, `--debug`, `--no-comments`, and
`--cfg-mode none|custom`. Run `uv run bingraph --help` for the full CLI.

### Client

The `client` subcommand makes an in-process request, so it is useful for
scripting and does not require a separately running server:

```bash
uv run bingraph --no-comments --root ./angr-binaries/tests \
  client --endpoint /cfg --filepath samples/ais3_crackme \
  --function 0x40043c --format raw
```

## HTTP API

The HTML routes are intended for browser use, while `/api/...` routes return
JSON.

| Route | Result |
| --- | --- |
| `GET /symtab?filepath=<path>` | HTML table of function symbols |
| `GET /api/symtab?filepath=<path>` | `{"items": [...]}` function-symbol response |
| `GET /cfg?filepath=<path>&function=<addr>` | SVG CFG |
| `GET /api/cfg?filepath=<path>&function=<addr>&format=dot` | `{"graph": "..."}` CFG response |

`function` accepts either decimal or `0x`-prefixed hexadecimal addresses. CFG
routes also accept optional `mode=none|custom|extract`, `comments=true|false`,
`dfs=true|false`, and `exits=never|jump|always` query
parameters. A request value overrides the corresponding
application setting for that render only.

## Configuration

Settings are Pydantic settings: they can be supplied as CLI options,
`BINGRAPH_`-prefixed environment variables, or values in `.bingraphenv`.

For example:

```bash
BINGRAPH_ROOT=./angr-binaries/tests \
BINGRAPH_CFG_MODE=custom \
BINGRAPH_COMMENTS=false \
uv run bingraph server
```

CFG modes:

- `custom` (default): build a bounded `CFGFast` graph and repair known local
  structural anomalies when present.
- `none`: return the bounded `CFGFast` graph without custom repair; useful for
  comparison and diagnostics.
- `extract`: independently decode and prove bounded control flow, then recover
  eligible disconnected code without treating it as proven entry-reachable code.

Disconnected-code recovery is part of extract mode, including API requests and
golden corpus runs. It scans only eligible explicitly sized functions with
unresolved jumps, rejects padding and known data, and shows selected rejoining
or validated return/trap/non-returning-exit regions behind dashed-orange
`UnresolvableEntrySource` edges. These are discovery hints, not proven jump targets.
Extract summaries separate discovered, entry-connected, and disconnected
instruction counts.
Recovery may split established blocks at existing instruction boundaries while
preserving their instructions and terminal transfers. Exact proof sources stay
pinned, and recovered code never contributes resolver facts.
Recovered calls are also checked against ELF LSDA metadata; proven cleanup
edges remain dotted gray and can discover additional landing pads.
The former recovery setting, CLI flags and API override have been removed.
Remove `BINGRAPH_CFG_RECOVERY` entries from `.bingraphenv` and omit
`--cfg-recovery`/`--no-cfg-recovery` from commands; no enable flag is needed.

## Development

The [Makefile](Makefile) wraps the standard development commands:

```bash
make                 # fast unit tests
make format          # rewrite source and tests with Ruff formatting
make check           # formatting, linting, type checking, and unit tests
```

The commands behind `make check` are also available independently:

```bash
make format-check
make lint
make typecheck
make units
make coverage
```

`make coverage` runs both unit tests and curated golden checkpoints, prints a
terminal summary, and writes a browsable report to `htmlcov/index.html`.

### Golden CFG regression tests

Golden tests render CFGs for the corpus in
[`tests/playground_functions.csv`](tests/playground_functions.csv). Compare
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

Promotion never rerenders CFGs. After reviewing a successful compare-mode run,
copy its existing candidates into the golden tree with:

```bash
BINGRAPH_GOLDEN_CONFIGS=cfg_mode_custom make goldens-promote
```

Review the resulting Git diff before committing promoted artifacts. Additional
selection controls are documented at the top of
[`tests/goldens/test_cfg_goldens.py`](tests/goldens/test_cfg_goldens.py),
including `BINGRAPH_GOLDEN_MIN_BBS` and `BINGRAPH_GOLDEN_LIMIT`.

## Troubleshooting

### Optional Unicorn warning on macOS

If startup reports:

```text
angr.state_plugins.unicorn_engine | failed loading "unicornlib.dylib"
```

Unicorn is an optional angr acceleration dependency. The bounded `CFGFast` and
custom CFG workflows used by Bingraph still work without it. Install a
compatible Unicorn build only if you specifically need Unicorn-backed symbolic
execution.
