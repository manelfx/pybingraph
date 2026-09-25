"""Shared data models for custom CFG analysis and repair."""

from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Literal

from angr import KnowledgeBase
from angr.knowledge_plugins.cfg import CFGModel, CFGNode

from bingraph.helpers.symbols import FunctionSymbol


# Internal repair terminology. These labels are converted to normal angr
# jumpkinds before they are materialized as graph edges.
TerminatorKind = Literal[
    "Ijk_Boring",
    "Ijk_Call",
    "Ijk_Fallthrough",
    "Ijk_Ret",
    "Ijk_Syscall",
    "Ijk_Terminal",
]
EdgeJumpKind = Literal["Ijk_Boring", "Ijk_Call", "Ijk_FakeRet"]
EntryResolutionPolicy = Literal["queued", "immediate"]


@dataclass(frozen=True)
class BlockSpec:
    """A basic-block candidate recovered from bounded disassembly."""

    addr: int
    size: int
    instruction_addrs: tuple[int, ...]
    jumpkind: TerminatorKind
    direct_targets: tuple[int, ...] = ()
    fallthrough_addr: int | None = None
    syscall_jumpkind: str | None = None
    vex_linear_instruction_sizes: tuple[tuple[int, int], ...] = ()


@dataclass(frozen=True)
class TerminatorInfo:
    """Normalized control-flow summary for one recovered block terminator."""

    jumpkind: TerminatorKind
    direct_targets: tuple[int, ...] = ()
    fallthrough_addr: int | None = None
    syscall_jumpkind: str | None = None


@dataclass(frozen=True)
class FunctionBounds:
    """Closed-open function bounds derived from the symbol table."""

    addr: int
    end_addr: int
    size: int
    symbol: FunctionSymbol
    display_name: str | None = None

    @property
    def name(self) -> str:
        """Return the stable label chosen for this CFG request."""

        return self.display_name or self.symbol.name


@dataclass(frozen=True)
class StaticJumpTable:
    """A high-confidence static jump table described by a VEX terminator."""

    base_register_offset: int | None
    base_bits: int
    table_displacement: int
    index_register_offset: int | None
    index_bits: int | None
    entry_size: int
    endness: str
    signed_entries: bool
    target_displacement: int = 0
    target_scale: int = 1
    target_or_mask: int = 0
    target_and_mask: int | None = None
    entries_are_relative: bool = True
    static_base_addr: int | None = None
    index_values: tuple[int, ...] | None = None
    index_low_bits: int | None = None
    index_expression: tuple[Any, ...] | None = None
    index_affine_difference: tuple[tuple[int, int], tuple[int, int], int] | None = None
    preserve_unresolved_fallback: bool = False


@dataclass(frozen=True)
class StaticJumpTablePlan:
    """A VEX-proven table, its concrete base, and selected entry indices."""

    table: StaticJumpTable
    base_addr: int
    entry_indices: tuple[int, ...]

    @property
    def entry_count(self) -> int:
        """Return the number of concrete table entries selected by the proof."""

        return len(self.entry_indices)


@dataclass(frozen=True)
class CFGAnomaly:
    """One node-local CFG invariant violation found during analysis or repair."""

    kind: str
    addr: int
    message: str


@dataclass
class CustomCFGStats:
    """Transformation and shape counters for one custom CFG repair session."""

    input_blocks: int = 0
    input_edges: int = 0
    input_anomalies: int = 0
    output_blocks: int = 0
    output_edges: int = 0
    output_anomalies: int = 0
    worklist_obligations: int = 0
    blocks_redecoded: int = 0
    blocks_replaced: int = 0
    linear_block_merges: int = 0
    shared_instruction_tails_factored: int = 0
    explicit_splits: int = 0
    placeholders_created: int = 0
    external_targets_created: int = 0
    undecodable_targets_created: int = 0
    edges_added: int = 0
    static_jump_tables_resolved: int = 0
    static_jump_targets_added: int = 0
    arithmetic_pc_dispatches_pruned: int = 0
    arithmetic_pc_targets_removed: int = 0
    static_jump_dispatchers_unresolved: int = 0
    static_jump_no_vex: int = 0
    static_jump_no_table_shape: int = 0
    static_jump_unknown_base: int = 0
    static_jump_unbounded_index: int = 0
    static_jump_table_unreadable: int = 0
    static_jump_table_empty: int = 0
    static_jump_tables_rejected_targets: int = 0
    static_jump_targets_read: int = 0
    static_jump_targets_accepted: int = 0
    static_jump_targets_external_code: int = 0
    static_jump_targets_unmapped: int = 0
    static_jump_targets_non_executable: int = 0
    static_jump_targets_synthetic: int = 0
    unresolved_jump_edges_removed: int = 0
    unresolved_fallback_edges_added: int = 0
    unresolved_fallbacks_flattened: int = 0
    unresolved_candidate_edges_flattened: int = 0
    inval_icache_self_loops_resolved: int = 0
    unreachable_blocks_removed: int = 0
    placeholders_pruned: int = 0
    orphan_simprocedures_pruned: int = 0
    function_owners_canonicalized: int = 0
    cleanup_rounds: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return a stable log-friendly view of the collected counters."""

        return {field: getattr(self, field) for field in self.__dataclass_fields__}


@dataclass
class BlockLeaderRegistry:
    """Reasons that an address must remain a basic-block entry during recovery."""

    reasons: dict[int, set[str]]

    def copy(self) -> BlockLeaderRegistry:
        """Return an independent snapshot suitable for one recovery attempt."""

        return BlockLeaderRegistry(
            {addr: set(reasons) for addr, reasons in self.reasons.items()}
        )

    def add(self, addr: int, reason: str) -> bool:
        """Record one leader reason and return whether the registry changed."""

        reasons = self.reasons.setdefault(addr, set())
        if reason in reasons:
            return False
        reasons.add(reason)
        return True

    def starts(self) -> set[int]:
        """Return all addresses currently required to begin a block."""

        return set(self.reasons)

    def starts_with_reason(self, reason: str) -> set[int]:
        """Return leader addresses that carry one particular reason."""

        return {addr for addr, reasons in self.reasons.items() if reason in reasons}


@dataclass(frozen=True)
class RepairObligation:
    """One request to recover or reconcile a CFG address."""

    addr: int
    reason: str
    action: Literal["recover", "reconcile"] = "recover"
    source_node: CFGNode | None = None
    jumpkind: EdgeJumpKind = "Ijk_Boring"
    preserve_exact_addr: bool = False
    resolution_policy: EntryResolutionPolicy = "queued"


@dataclass(frozen=True)
class EdgeClaim:
    """One required edge from one exact live source node into an obligation."""

    source_node: CFGNode
    jumpkind: EdgeJumpKind


@dataclass
class PendingObligation:
    """Merged queued work for one action at one CFG address."""

    addr: int
    action: Literal["recover", "reconcile"]
    reasons: set[str]
    edge_claims: set[EdgeClaim]
    preserve_exact_addr: bool = False

    @classmethod
    def from_request(cls, request: RepairObligation) -> PendingObligation:
        """Create pending state from one first-in request."""

        claims = set()
        if request.source_node is not None:
            claims.add(EdgeClaim(request.source_node, request.jumpkind))
        return cls(
            addr=request.addr,
            action=request.action,
            reasons={request.reason},
            edge_claims=claims,
            preserve_exact_addr=request.preserve_exact_addr,
        )

    def merge(self, request: RepairObligation) -> None:
        """Accumulate another request without changing queue order."""

        self.reasons.add(request.reason)
        if request.source_node is not None:
            self.edge_claims.add(EdgeClaim(request.source_node, request.jumpkind))
        self.preserve_exact_addr |= request.preserve_exact_addr

    def fingerprint(self) -> tuple[bool, tuple[tuple[int, EdgeJumpKind], ...]]:
        """Return the repair-relevant state used to detect a stalled requeue."""

        claims = tuple(
            sorted(
                (id(claim.source_node), claim.jumpkind) for claim in self.edge_claims
            )
        )
        return self.preserve_exact_addr, claims


@dataclass(frozen=True)
class JumpSuccessorExpectation:
    """One expected direct jump successor and its repair metadata."""

    addr: int
    jumpkind: EdgeJumpKind
    preserve_exact_addr: bool


@dataclass(frozen=True)
class JumpSuccessorAnalysis:
    """Expected and present successors for one decoded jump-terminating block."""

    kind: Literal["conditional", "direct"]
    expected: tuple[JumpSuccessorExpectation, ...]
    present: frozenset[int]


class CustomCFG(SimpleNamespace):
    """Small CFG-like wrapper exposing the attributes bingraph actually uses."""

    graph: object
    model: CFGModel
    functions: object
    kb: KnowledgeBase
    custom_stats: CustomCFGStats
