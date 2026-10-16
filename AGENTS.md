# Agent Workflow

## Checkpoints and Git

- Before presenting any code changes for review, create a named stash checkpoint
  and immediately re-apply it so the working tree remains available:

  ```sh
  git stash push -m "codex checkpoint: <description>"
  git stash apply stash@{0}
  ```

- Do not include untracked files in checkpoints unless the user explicitly asks.
- Never use destructive Git commands such as `git reset --hard` or
  `git checkout --` without explicit user approval.
- After a successful push, clear the checkpoint stack with `git stash clear`,
  unless the user asks to preserve a stash entry. The stash stack is shared:
  inspect it first and never remove user-created entries.
- Before applying, dropping, or clearing a stash, verify that the relevant
  entry is a Codex checkpoint. Do not assume `stash@{0}` belongs to Codex.
- Treat unexpected worktree changes as user changes. Do not revert or remove
  them unless explicitly asked.

## Python Tooling

- Run Python commands, tests, linters, and formatters through `uv run`.
- Use the repository Makefile targets where they cover the requested check.
- Run focused tests first; do not run the full golden corpus unless requested.

Run `make compatibility` to smoke-test one wheel on the Makefile's
`PYTHON_VERSIONS`, or select versions with
`make compatibility PYTHON_VERSIONS="3.12 3.14"`. Clang and Graphviz are required.
Checks use temporary environments and download missing Python interpreters
without upgrading existing ones.
On pushes and release runs, the Python packages workflow audits 3.12 during
packaging and the same wheel on the remaining versions before attaching assets.
It first runs `make check` followed by `make goldens-checkpoint` on 3.12 with
locked development dependencies and a pinned playground checkout plus samples from
`scripts/fetch_playground.sh`; failures block packaging and release attachments.
Failed checkpoint candidates are uploaded for review; CI never promotes goldens.
After changing `PYTHON_VERSIONS`, run `make badges` and commit the SVG with the
Makefile. `make badges-check` validates it without writing; both `make check`
and CI run that validation. CI's additional Python matrix uses the same list.

## Source Organization

Paths below are relative to `src/bingraph/`. Update this section as part of the
same reviewed change when moving modules or changing their responsibilities.
Keep the golden-test guidance current when its commands or selection controls
change. Documentation maintenance must preserve the user's workflow rules.

- `cfg/builder.py` owns function discovery, leader splitting and final graph
  construction. `cfg/decode.py` normalizes bounded block terminators;
  `cfg/calls.py` supplies local callee and non-returning evidence.
- `cfg/jumps/` separates dispatch expressions, guard bounds, table planning,
  conditional-PC transfers and the retained MIPS, PPC and x86 adapters.
- `cfg/shared/{facts,relations,proofs}.py` implements snapshot-scoped predecessor
  facts, bounded angr/Claripy evaluation and exact table/register target proofs.
  Disconnected recovery hints never become exact proof facts.
- `cfg/nodes.py` constructs and names existing block leaders and synthetic
  leaves. `cfg/returns.py` provides read-only saved-link return proofs; the
  builder decides when to apply them.
- `helpers/disassembly.py` and `helpers/vex.py` provide reusable instruction
  decoding and stateless IR primitives, without CFG discovery or rendering.
- `helpers/abi.py` supplies shared entry-base roots, ELF register preservation
  and verified x86 PC-thunk semantics. Entry roots do not imply preservation
  across calls or unknown paths.
- `core/annotators/` separates node styling, assembly comments and edge styling.
  These presentation operations do not change discovered control flow.

Module moves must preserve resolver priority, proof budgets, cache lifetime and
unknown-target handling. Keep internal imports directed at their implementation
modules; package entry points expose the public construction and annotation API.

## Commits

- Use a concise subject line, followed by a blank line and an explanatory body
  when a body is useful.
- Keep every commit-message line at or below 80 columns.
- Use real line breaks in commit messages, never literal `\\n` text.
- Inspect the final commit message with `git log -1 --format=fuller` before
  pushing when the message has been amended or rewritten.

## Golden Artifacts

- Do not modify user-managed golden artifacts or `tests/_actual/` unless the
  user explicitly requests generation, promotion, or cleanup.
- Preserve the current working tree and validate focused checkpoint goldens
  when making CFG changes.

### Golden CFG Regression Tests

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
[`tests/goldens/test_cfg_goldens.py`](tests/goldens/test_cfg_goldens.py),
including `BINGRAPH_GOLDEN_MIN_BBS` and `BINGRAPH_GOLDEN_LIMIT`.

## Versioning and Releases

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
- Preserve existing release assets: skip identical files, never overwrite differing
  ones. For immutable releases, create a draft, run the Python packages workflow
  from `main` with `release_tag`, verify attachments, then publish.
- Keep runtime dependency bounds compatible with the tested stack. Retain the angr
  pin and `bitstring<5` safeguard; dependency upgrades need unit/golden validation.
- Preserve `LICENSE` and upstream attributions in `NOTICE`; both must accompany
  wheels and source distributions.

Before release, run `make check`, applicable goldens, and packaging audits using
an empty build output directory. The runtime audit requires Clang and Graphviz;
it does not replace unit or golden tests.

```bash
uv run --no-project python src/bingraph/version.py --json > version.json
uv build
uv run --no-project python scripts/distributions.py
uv run --no-project python scripts/distributions.py --runtime
```
