"""Exercise custom proof accounting without running the golden corpus."""

from dataclasses import replace
from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

import networkx as nx

from bingraph.cfg.builder import _BuildSession
from bingraph.cfg.models import BlockSpec, CustomCFGStats, CustomCFGSummary


_GOLDEN_TEST_PATH = Path(__file__).resolve().parents[1] / "goldens/test_cfg_goldens.py"
_SPEC = spec_from_file_location("_cfg_goldens_stats_test", _GOLDEN_TEST_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_GOLDEN_TEST = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _GOLDEN_TEST
_SPEC.loader.exec_module(_GOLDEN_TEST)


def test_custom_summary_counts_actions_functions_and_architectures() -> None:
    state = _GOLDEN_TEST.ConfigRunState(expected_files={Path("summary.json")})
    _GOLDEN_TEST._record_custom_cfg_stats(
        state,
        SimpleNamespace(
            custom_stats=CustomCFGStats(
                static_jump_plans_resolved=2,
                legacy_table_fallback_attempts=4,
                exact_jump_proofs_by_flavor={"mips_pic_table": 2},
                decoder_call_targets_by_kind={"mips_gp": 3},
                decoder_nonreturning_calls=1,
                shared_target_rejection_attempts_by_reason={"non_executable": 2},
            )
        ),
        "mips64",
    )
    _GOLDEN_TEST._record_custom_cfg_stats(
        state,
        SimpleNamespace(
            custom_stats=CustomCFGStats(
                static_jump_plans_resolved=1,
                exact_jump_proofs_by_flavor={"mips_pic_table": 1},
                decoder_call_targets_by_kind={"mips_gp": 2},
            )
        ),
        "mipsel",
    )

    summary = _GOLDEN_TEST._summary_payload(
        _GOLDEN_TEST.GoldenConfig("cfg_mode_custom", "custom"), state, None
    )
    custom = summary["custom_cfg_stats"]
    assert custom["runs"] == 2
    assert custom["totals"]["static_jump_plans_resolved"] == 3
    assert custom["affected_functions"]["static_jump_plans_resolved"] == 2
    assert custom["totals"]["legacy_table_fallback_attempts"] == 4
    assert custom["affected_functions"]["legacy_table_fallback_attempts"] == 1
    assert custom["totals"]["exact_jump_proofs_by_flavor.mips_pic_table"] == 3
    assert custom["totals"]["decoder_call_targets_by_kind.mips_gp"] == 5
    assert custom["affected_functions"]["decoder_call_targets_by_kind.mips_gp"] == 2
    assert custom["totals"]["decoder_nonreturning_calls"] == 1
    assert (
        custom["totals"]["shared_target_rejection_attempts_by_reason.non_executable"]
        == 2
    )
    assert custom["proofs_by_architecture"] == {
        "mips64": {"mips_pic_table": 2},
        "mipsel": {"mips_pic_table": 1},
    }

    unchanged = _GOLDEN_TEST._summary_payload(
        _GOLDEN_TEST.GoldenConfig("cfg_mode_none", "none"), state, None
    )
    assert "proofs_by_architecture" not in unchanged["custom_cfg_stats"]


def test_decoder_provenance_does_not_change_proof_source_equality() -> None:
    block = BlockSpec(0x1000, 4, (0x1000,), "Ijk_Call", (0x2000,))
    annotated = replace(
        block, decoded_call_target_kind="static_memory", decoded_nonreturning_call=True
    )
    assert annotated == block
    assert hash(annotated) == hash(block)
    assert replace(annotated, direct_targets=(0x3000,)) != block


def test_decoder_counters_use_final_partition_not_decode_attempts() -> None:
    session = object.__new__(_BuildSession)
    original = BlockSpec(
        0x1000,
        8,
        (0x1000, 0x1004),
        "Ijk_Call",
        (0x2000,),
        decoded_call_target_kind="static_memory",
    )
    session.blocks = {original.addr: original}
    # Recovery splits the original block: its call appears only in the tail.
    session.recovery_baseline = {
        0x1000: BlockSpec(0x1000, 4, (0x1000,), "Ijk_Fallthrough"),
        0x1004: replace(original, addr=0x1004, size=4, instruction_addrs=(0x1004,)),
    }
    session.recovered_blocks = {
        0x1010: BlockSpec(
            0x1010,
            4,
            (0x1010,),
            "Ijk_Call",
            (0x3000,),
            decoded_call_target_kind="vex_constant",
            decoded_nonreturning_call=True,
        ),
        0x1020: BlockSpec(0x1020, 4, (0x1020,), "Ijk_Call"),
    }
    session.stats = CustomCFGStats(blocks_decoded=20, block_redecodes=17)
    session.summary = CustomCFGSummary()
    session.leaf_nodes = {}
    session.nodes = {}
    session.func_addr = 0x1000
    session.graph = nx.DiGraph()

    session._summarize_output()

    assert session.stats.decoder_call_targets_by_kind == {
        "static_memory": 1,
        "vex_constant": 1,
    }
    assert session.stats.decoder_nonreturning_calls == 1
    assert session.summary.calls == 3  # Includes the unresolved recovered call.
    assert session.stats.post_decode_call_fallthroughs_suppressed == 0
