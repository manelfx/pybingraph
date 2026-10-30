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
  CFG-to-DDG links include plain-text address and size tooltips; instruction
  counts reuse rendered assembly rows rather than triggering another decode.
- `ddg/effects.py` extracts instruction-local VEX effects and delegates scalar
  arithmetic to angr/Claripy. Unoptimized IR preserves architectural flag
  producers; pure temporary reads are collected at their consumers, while
  actual loads remain effects even when their values are unused. Optional
  values have an expanded-expression budget independent of dependency facts.
  Scalar FP reads use angr-compatible bitcasts from register/memory storage;
  optional values that the arithmetic backend cannot evaluate become typed
  unknowns without discarding operand dependencies. Unknown explicit rounding
  modes remain unknown values rather than using angr's default-mode assumption.
  Large blocks span bounded VEX lifts only across
  proven straight-line fallthroughs, retaining state and separate temporary
  namespaces. `ddg/memory.py` handles conservative byte-range
  aliases; `ddg/builder.py` computes RAW/WAR/WAW ordering. Neither discovers
  blocks or follows callees.
  Instruction-internal VEX self-loops replay only concrete guards within a
  fixed iteration budget, using the same evaluator. Completed memory accesses
  merge only across proven contiguous ranges; symbolic/multi-exit loops fail
  rather than silently exposing partial effects. VEX restart counters are
  internal state, not architectural dependency resources.
  Explicit terminal UD2/UDF traps require matching Capstone architecture/opcode
  metadata and a VEX NoDecode stop/fault PC at that final instruction. Retain
  any prefix/trap IR effects; never skip an earlier undecodable instruction or
  accept instructions following the trap.
  Repeated x86 CMPS/SCAS with concrete counts instead summarize maximum read
  footprints from one VEX iteration, without exploring early exits. Unknown
  direction covers both directions; final registers become unknown. Zero counts
  leave memory, pointers and flags untouched. Symbolic counts remain unsupported.
- `ddg/render.py` adapts immutable cached analysis records to one-instruction
  CFG nodes and the existing `Vis`/`NodeAsm`/`DotOutput` pipeline. DDG edge styles
  describe dependency kinds, not CFG branch outcomes. Real CFG blocks receive
  root-relative DDG links through a presentation-only annotator.

DDG analysis caching includes project, start address, exact size and memory
policy; rendering additionally keys on the dependency selection and format,
filters only edges, and uses fresh wrappers. Register aliases share archinfo
byte ranges, including widened architectural writes. Memory roots are symbolic
and versioned; different pointer registers never imply disjointness. Unsupported stateful IR
must fail explicitly. Keep external block inputs implicit (no synthetic input
nodes), and never connect independent instructions just to simplify layout.
DDG register captions retain Archinfo aliases, then use PyVEX field names or
endianness-aware slices of declared parent registers. Unmapped slots show their
guest offset and byte width explicitly; naming never changes resource bounds.
Inactive Thumb IT-state resets are omitted only when the entire requested block
has no consumed IT-state reads and all writes to that field are full-width zero
resets. Incoming, active, symbolic or partial IT-state effects remain intact.

Memory policy selects edges, not alias facts: `may` retains unknown relations,
whereas default `must` omits them. Do not use that omission to infer pointer
independence, remove possible producers, or claim safe instruction reordering.
For repeated comparisons, `must` proves overlap of modeled maximum footprints,
not that an early-terminating instruction actually accesses every modeled byte.
Guarded VEX loads/stores retain guard reads and mark possible memory accesses as
conditional, independently of alias policy. Symbolic loads combine the loaded
and fallback values within the existing expression budget. Conditional stores
never kill earlier producers or merge into unconditional write ranges; proven
false guards consume no memory, address or store-data resources.

Module moves must preserve resolver priority, proof budgets, cache lifetime and
unknown-target handling. Keep internal imports directed at their implementation
modules; package entry points expose the public construction and annotation API.

## Dependency Changes

Keep runtime dependency bounds compatible with the tested stack. Retain the angr
pin and the `bitstring<5` and `pycparser<3.12` safeguards unless deliberately
upgrading and validating those dependencies. The parser bound preserves the
private token-stream API used by angr 10.0.4; fresh installs otherwise fail
during angr import even though the locked environment works.
Dependency upgrades need unit and applicable golden validation;
packaging or interpreter-compatibility checks do not replace CFG regression tests.
Use the [testing guide](testing.md) to select validation for the changed stack.
