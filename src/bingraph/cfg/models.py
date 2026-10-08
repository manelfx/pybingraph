"""Shared data models for bounded custom CFG construction."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any, Literal

from angr import KnowledgeBase
from angr.knowledge_plugins.cfg import CFGModel, CFGNode

from bingraph.helpers.symbols import FunctionSymbol


# Internal decoding terminology. These labels are converted to normal angr
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
DecodedCallTargetKind = Literal[
    "vex_constant", "static_memory", "mips_gp", "nonreturning_memory"
]


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
    # Audit provenance must not affect source equality or proof revalidation.
    decoded_call_target_kind: DecodedCallTargetKind | None = field(
        default=None, compare=False
    )
    decoded_nonreturning_call: bool = field(default=False, compare=False)


@dataclass(frozen=True)
class TerminatorInfo:
    """Normalized control-flow summary for one recovered block terminator."""

    jumpkind: TerminatorKind
    direct_targets: tuple[int, ...] = ()
    fallthrough_addr: int | None = None
    syscall_jumpkind: str | None = None
    decoded_call_target_kind: DecodedCallTargetKind | None = field(
        default=None, compare=False
    )
    decoded_nonreturning_call: bool = field(default=False, compare=False)


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
    target_and_mask: int | None = None
    entries_are_relative: bool = True
    static_base_addr: int | None = None
    index_values: tuple[int, ...] | None = None
    index_low_bits: int | None = None
    index_expression: tuple[Any, ...] | None = None
    preserve_unresolved_fallback: bool = False


@dataclass(frozen=True)
class StaticJumpTablePlan:
    """A VEX-proven table, its concrete base, and selected entry indices."""

    table: StaticJumpTable
    base_addr: int
    entry_indices: tuple[int, ...]
    proof_flavor: str = "generic_vex_table"

    @property
    def entry_count(self) -> int:
        """Return the number of concrete table entries selected by the proof."""

        return len(self.entry_indices)


@dataclass
class CustomCFGStats:
    """Audit construction work separately from final-site contributions.

    Attempt/work counters can revisit a site after graph changes; they are
    neither unique instructions nor evidence of incorrect proofs. Decoder
    counters count final displayed call blocks, including recovered code.
    Zero rejections in a corpus do not establish that validation is redundant.
    """

    additional_leaders_discovered: int = 0
    leaders_rejected_invalid_entry: int = 0
    leaders_split_existing_block: int = 0
    blocks_decoded: int = 0
    block_redecodes: int = 0
    blocks_redecoded_for_leader_split: int = 0
    blocks_redecoded_for_data: int = 0
    decode_failures: int = 0
    vex_linear_fallbacks: int = 0
    data_leaders_rejected: int = 0
    data_region_observations: int = 0
    data_bytes_discovered: int = 0
    # Post-decoder actions (data rejection or ABI proofs), net of withdrawals.
    post_decode_call_fallthroughs_suppressed: int = 0
    decoder_call_targets_by_kind: dict[str, int] = field(default_factory=dict)
    # Known no-return calls, not necessarily removed in-function continuations.
    decoder_nonreturning_calls: int = 0
    static_syscall_resolution_attempts: int = 0
    static_syscalls_resolved: int = 0
    static_syscall_fallthroughs_suppressed: int = 0
    abi_static_target_analysis_runs: int = 0
    abi_static_target_analysis_budget_exhausted: int = 0
    abi_static_call_targets_resolved: int = 0
    abi_static_jump_targets_resolved: int = 0
    linear_direct_transfers_continued: int = 0
    unresolved_indirect_targets: int = 0
    unresolved_call_targets: int = 0
    external_target_references: int = 0
    undecodable_target_references: int = 0
    synthetic_leaves_created: int = 0
    synthetic_leaves_reused: int = 0
    static_jump_plan_attempts: int = 0
    static_jump_plans_resolved: int = 0
    exact_jump_proofs_by_flavor: dict[str, int] = field(default_factory=dict)
    shared_table_attempts: int = 0
    legacy_table_fallback_attempts: int = 0
    shared_fact_steps: int = 0
    shared_fact_budget_exhausted: int = 0
    # All plans in rounds that discover leaders, even if their sources survive.
    static_jump_plans_in_discovery_rounds: int = 0
    # Deduplicated destinations per successful query, not physical table rows.
    static_jump_target_candidate_attempts: int = 0
    static_jump_targets_accepted: int = 0
    static_jump_target_edges_added: int = 0
    exception_metadata_functions_scanned: int = 0
    exception_call_sites_discovered: int = 0
    exceptional_transfers_discovered: int = 0
    exception_edges_added: int = 0
    static_jump_unresolved_dispatcher_attempts: int = 0
    static_jump_no_vex: int = 0
    static_jump_no_table_shape: int = 0
    static_jump_dynamic_memory_target: int = 0
    static_jump_unknown_base: int = 0
    static_jump_unbounded_index: int = 0
    static_jump_table_unreadable: int = 0
    # Builder validation covers external candidates only. Shared queries can
    # reject earlier; their first failing target is counted per proof attempt.
    static_jump_external_target_rejection_attempts: int = 0
    shared_target_rejection_attempts_by_reason: dict[str, int] = field(
        default_factory=dict
    )
    conditional_pc_dispatches_resolved: int = 0
    conditional_pc_targets_recovered: int = 0
    disconnected_recovery_runs: int = 0
    disconnected_recovery_budget_exhausted: int = 0
    disconnected_recovery_rejected_changes: int = 0
    disconnected_regions: int = 0
    disconnected_blocks: int = 0
    output_anomaly_count: int = 0
    output_anomalies_by_kind: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int | dict[str, int]]:
        """Return stable log-friendly construction counters."""

        return asdict(self)


@dataclass
class CustomCFGSummary:
    """Describe the final graph materialized by one CFG construction."""

    normal_blocks: int = 0
    synthetic_leaves: int = 0
    nodes: int = 0
    edges: int = 0
    calls: int = 0
    syscalls: int = 0
    direct_branches: int = 0
    conditional_branches: int = 0
    returns: int = 0
    terminal_blocks: int = 0
    direct_edges: int = 0
    fallthrough_edges: int = 0
    discovered_instructions: int = 0
    entry_connected_instructions: int = 0
    disconnected_instructions: int = 0

    def as_dict(self) -> dict[str, int]:
        """Return a stable log-friendly view of the materialized graph."""

        return asdict(self)


class CustomCFGNode(CFGNode):
    """A normal CFG node with custom-builder VEX fallback rendering spans."""

    __slots__ = ("vex_linear_instruction_sizes",)

    def __init__(
        self, *args: Any, vex_linear_instruction_sizes: dict[int, int], **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vex_linear_instruction_sizes = vex_linear_instruction_sizes


class CustomCFG(SimpleNamespace):
    """Small CFGBase-compatible surface consumed by bingraph rendering."""

    graph: Any
    model: CFGModel
    functions: Any
    kb: KnowledgeBase
    custom_stats: CustomCFGStats
    custom_summary: CustomCFGSummary
    _comments_collected: bool = False
