# pybingraph

[![Latest release](https://img.shields.io/github/v/release/manelfx/pybingraph?label=latest%20release&color=blue)](https://github.com/manelfx/pybingraph/releases)
[![Development version](https://manelfx.github.io/pybingraph/version.svg)](https://github.com/manelfx/pybingraph/tree/dev)
[![Python versions](https://raw.githubusercontent.com/manelfx/pybingraph/dev/docs/images/python-versions.svg)](https://github.com/manelfx/pybingraph/actions/workflows/packages.yml)

`pybingraph` is a binary-inspection service and library built on
[angr](https://angr.io/). It reads function symbols from a binary and renders
per-function control-flow graphs (CFGs) as SVG or DOT.

Supported binary formats include ELF executables, shared libraries and
relocatable objects, and Windows PE executables and DLLs. The corpus covers
x86/x86-64, ARM/Thumb/AArch64, MIPS, PowerPC, RISC-V and s390x. Instruction
decoding and indirect-transfer proofs depend on the architecture and binary;
not every format/architecture combination has the same analysis coverage.

The default `custom` strategy constructs a function-bounded CFG directly from
Capstone decoding and VEX control-flow proofs, without a `CFGFast` seed graph.
It resolves proven indirect transfers, recovers ELF exception paths, and keeps
unresolved targets and additionally discovered code explicit.

## CFG Examples

These are real custom-mode CFGs from the playground binaries. Comments are
disabled for compact previews, matching the golden-test configuration; they
will be shown in the browser as block annotations. Click a thumbnail to
inspect the full-size SVG.

| Example | Description | Preview |
| --- | --- | --- |
| **Resolved Switch Dispatch** | `func_0` in `x86_64/cfg_switches`. Red and green edges distinguish the two conditional outcomes; orange edges identify proven indirect-jump targets. Here the cases converge on a shared return block; the same branch colors also make loops and separate return paths visible in other CFGs. | [<img src="https://raw.githubusercontent.com/manelfx/pybingraph/main/docs/images/cfg-switch-x86-64.svg" width="240" alt="x86-64 switch CFG with resolved jump-table targets">](https://github.com/manelfx/pybingraph/blob/main/docs/images/cfg-switch-x86-64.svg) |
| **Exception Paths and Unresolved Control Flow** | `drop_in_place<ReferenceKind>` in `x86_64/fmt-rust`. Dotted gray edges show exception paths. Dashed orange edges keep unresolved targets and disconnected-code recovery hints explicit rather than claiming proven reachability. | [<img src="https://raw.githubusercontent.com/manelfx/pybingraph/main/docs/images/cfg-unwind-rust.svg" width="240" alt="Rust CFG with exception paths, an unresolved jump and recovered regions">](https://github.com/manelfx/pybingraph/blob/main/docs/images/cfg-unwind-rust.svg) |
| **Vectorized String Search** | `__strchr_evex` at `0x42b200` in `x86_64/langdetect_clang`. AVX-512 vector loads and comparisons, mask registers and a scanning loop remain visible in the disassembly. | [<img src="https://raw.githubusercontent.com/manelfx/pybingraph/main/docs/images/cfg-simd-x86-64.svg" width="240" alt="x86-64 AVX-512 string-search CFG">](https://github.com/manelfx/pybingraph/blob/main/docs/images/cfg-simd-x86-64.svg) |
| **RISC-V Control Flow** | `sodium_allocarray` at `0x3208` in `riscv/server_eapp.eapp_riscv`. RV64 compressed instructions, allocation-size guards, calls and return paths use the same CFG presentation. | [<img src="https://raw.githubusercontent.com/manelfx/pybingraph/main/docs/images/cfg-riscv.svg" width="240" alt="RISC-V allocation CFG with compressed instructions and conditional branches">](https://github.com/manelfx/pybingraph/blob/main/docs/images/cfg-riscv.svg) |

TODOs for this library are:
- Render call graph (CG) of all functions in a binary.
- Render data dependency graph (DDG) for instructions in a basic block.
- Graph coloring from execution counts, after adding support for profile information.

## Requirements

For the container option below, only Docker is required on the host. A native
Python installation requires:

- Python 3.12 or later
- [uv](https://docs.astral.sh/uv/)
- The Graphviz `dot` executable for rendered image formats

On macOS:

```bash
brew install graphviz
```

## Setup

### Run with Docker

Release images are published to [GitHub Container Registry](https://github.com/manelfx/pybingraph/pkgs/container/pybingraph)
for Linux AMD64 and ARM64. Docker chooses the appropriate platform automatically.
They include Python, runtime dependencies and Graphviz, but no playground binaries.

```bash
docker run --rm --name bingraph \
  -p 127.0.0.1:9000:8000 \
  --mount type=bind,src=/absolute/path/to/binaries,dst=/binaries,readonly \
  ghcr.io/manelfx/pybingraph:latest
```

Replace the host directory with your binary directory and `9000` with the local
port you want. Open `http://127.0.0.1:9000/docs` for interactive API documentation,
or `/symtab?filepath=your-binary` for the symbol table. Paths remain relative to
the mounted directory. The image runs as a non-root user; mounted files and their
parent directories must be readable/searchable by that user.

Use a numeric release tag instead of `latest` to select a specific version.
`docker run --rm ghcr.io/manelfx/pybingraph:latest --version` reports its version.
Configure global settings with environment flags such as
`-e BINGRAPH_DEMANGLE=true` or `-e BINGRAPH_COMMENTS=false` before the image name.
The server listens on `0.0.0.0:8000` **inside** the container; the example publishes
it only on the host's loopback interface. It has no authentication and should not
be exposed directly to an untrusted network.

### Install a release

Published releases can be installed from [PyPI](https://pypi.org/project/pybingraph/)
with their Python dependencies:

```bash
uv venv --python 3.12
uv pip install pybingraph
```

On macOS/Linux, activate that environment and run the installed command:

```bash
source .venv/bin/activate
bingraph --version
bingraph --root /path/to/your/binaries server
```

No repository checkout is required. Install the Graphviz `dot` executable
separately; wheels do not bundle system tools or the optional playground corpus.
Alternatively, download a wheel (`.whl`) from a GitHub release and install it
with `uv pip install ./pybingraph-*.whl` in the environment above.

### Work from a checkout

Install the runtime dependencies:

```bash
uv sync
```

For development tools such as pytest, Ruff, and Ty:

```bash
uv sync --extra dev
```

### Playground binaries

The optional playground corpus is used by unit tests, golden CFG tests and the
examples below. Fetch it with:

```bash
./scripts/fetch_playground.sh
```

It is cloned into `./angr-binaries/`; the binaries themselves live under
`./angr-binaries/tests/`. The helper also fetches pinned example fixtures from
[angr-utils](https://github.com/axt/angr-utils/tree/master/examples/samples) into
`tests/samples/` within that corpus. Existing samples are verified, not overwritten.

## Run Bingraph

All binary paths are relative to `--root` and are constrained to remain below
that directory.

Check the version from the CLI or Python:

```bash
uv run bingraph --version
uv run python -c 'import bingraph; print(bingraph.__version__, bingraph.__commit__)'
```

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

Use the global `--demangle` option (or `BINGRAPH_DEMANGLE=true`) to display
demangled C++ and Rust names in symbol tables, CFG headers, and comments.
It defaults to false. CFG names are shortened after demangling, keeping
block offsets; symbol-table names remain full. Addresses, assembly operands,
and analysis metadata are unchanged. Unsupported names are kept as-is.

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
JSON responses support `format=svg|dot|raw|xdot`; `raw` is unprocessed DOT,
`dot` runs Graphviz's DOT output backend, and `xdot` includes drawing metadata.
Binary formats such as PNG/PDF are not supported by the string-returning API.

CFG routes also accept optional `mode=none|custom`, `comments=true|false`,
`dfs=true|false`, and `exits=never|jump|always` query
parameters. A request value overrides the corresponding
application setting for that render only.
Both symbol-table and CFG routes also accept `demangle=true|false`, overriding
the global setting for that request. Symbol-table links retain this override.

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

- `custom` (default): independently decode and prove bounded control flow, then
  recover eligible disconnected code without treating it as proven
  entry-reachable code.
- `none`: return the bounded `CFGFast` graph without custom construction; useful
  for comparison and diagnostics.

Disconnected-code recovery is part of custom mode.
It scans only eligible explicitly sized functions with
unresolved jumps, rejects padding and known data, and shows selected rejoining
or validated return/trap/non-returning-exit regions behind dashed-orange
`UnresolvableEntrySource` edges. These are discovery hints, not proven jump targets.
Custom summaries separate discovered, entry-connected, and disconnected
instruction counts.

## Development

The [Makefile](https://github.com/manelfx/pybingraph/blob/main/Makefile) wraps the standard development commands:

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

## License

pybingraph's original code is licensed under the [BSD 2-Clause License](https://github.com/manelfx/pybingraph/blob/main/LICENSE).
Copied or adapted code retains the upstream copyright and license notices in
[NOTICE](https://github.com/manelfx/pybingraph/blob/main/NOTICE). Dependencies, playground binaries and their disassembly
examples remain subject to their respective upstream terms.

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
