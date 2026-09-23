"""Data models exposed by the experimental independent CFG extractor."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from types import SimpleNamespace
from typing import Any

from angr import KnowledgeBase
from angr.knowledge_plugins.cfg import CFGModel, CFGNode


@dataclass
class ExtractedCFGStats:
    """Audit the decisions and transformations of one CFG extraction."""

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
    call_fallthroughs_suppressed: int = 0
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
    static_jump_plans_invalidated: int = 0
    static_jump_table_entries_read: int = 0
    static_jump_targets_accepted: int = 0
    static_jump_target_edges_added: int = 0
    static_jump_candidate_plans: int = 0
    static_jump_candidate_entries_read: int = 0
    static_jump_candidate_targets_accepted: int = 0
    static_jump_candidate_edges_added: int = 0
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
    static_jump_targets_rejected: int = 0
    conditional_pc_dispatches_resolved: int = 0
    conditional_pc_targets_recovered: int = 0
    sweep_runs: int = 0
    sweep_candidate_blocks: int = 0
    sweep_candidate_instructions: int = 0
    sweep_candidate_components: int = 0
    sweep_decode_failures: int = 0
    sweep_non_executable_bytes: int = 0
    sweep_dispatchers_ineligible: int = 0
    sweep_reconnecting_components: int = 0
    sweep_reconnecting_blocks: int = 0
    sweep_component_roots_attached: int = 0
    output_anomaly_count: int = 0
    output_anomalies_by_kind: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, int | dict[str, int]]:
        """Return stable log-friendly extraction counters."""

        return asdict(self)


@dataclass
class ExtractedCFGSummary:
    """Describe the final graph materialized by one CFG extraction."""

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

    def as_dict(self) -> dict[str, int]:
        """Return a stable log-friendly view of the materialized graph."""

        return asdict(self)


class ExtractedCFGNode(CFGNode):
    """A normal CFG node with extractor-only VEX fallback rendering spans."""

    __slots__ = ("vex_linear_instruction_sizes",)

    def __init__(
        self, *args: Any, vex_linear_instruction_sizes: dict[int, int], **kwargs: Any
    ) -> None:
        super().__init__(*args, **kwargs)
        self.vex_linear_instruction_sizes = vex_linear_instruction_sizes


class ExtractedCFG(SimpleNamespace):
    """Small CFGBase-compatible surface consumed by bingraph rendering."""

    graph: Any
    model: CFGModel
    functions: Any
    kb: KnowledgeBase
    extract_stats: ExtractedCFGStats
    extract_summary: ExtractedCFGSummary
