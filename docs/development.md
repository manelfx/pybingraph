# Development

This guide covers architecture and dependency maintenance. See
[AGENTS.md](../AGENTS.md) for the instruction entry point,
[Testing](testing.md) for validation, [Git](git.md) for checkpoints and commits,
and [Releases](releases.md) for packaging and publication.

## Source Organization

Paths below are relative to `src/bingraph/`. Update this map in the same reviewed
change when moving modules or changing their responsibilities.

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

## Dependency Changes

Keep runtime dependency bounds compatible with the tested stack. Retain the angr
pin and `bitstring<5` safeguard unless deliberately upgrading and validating those
dependencies. Dependency upgrades need unit and applicable golden validation;
packaging or interpreter-compatibility checks do not replace CFG regression tests.
Use the [testing guide](testing.md) to select validation for the changed stack.
