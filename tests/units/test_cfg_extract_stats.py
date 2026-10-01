"""Exercise extract proof accounting without running the golden corpus."""

from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path
import sys
from types import SimpleNamespace

from bingraph.cfg_extract.models import ExtractedCFGStats


_GOLDEN_TEST_PATH = Path(__file__).resolve().parents[1] / "goldens/test_cfg_goldens.py"
_SPEC = spec_from_file_location("_cfg_goldens_stats_test", _GOLDEN_TEST_PATH)
assert _SPEC is not None and _SPEC.loader is not None
_GOLDEN_TEST = module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _GOLDEN_TEST
_SPEC.loader.exec_module(_GOLDEN_TEST)


def test_extract_summary_counts_actions_functions_and_architectures() -> None:
    state = _GOLDEN_TEST.ConfigRunState(expected_files={Path("summary.json")})
    _GOLDEN_TEST._record_extract_cfg_stats(
        state,
        SimpleNamespace(
            extract_stats=ExtractedCFGStats(
                static_jump_plans_resolved=2,
                exact_jump_proofs_by_flavor={"mips_pic_table": 2},
            )
        ),
        "mips64",
    )
    _GOLDEN_TEST._record_extract_cfg_stats(
        state,
        SimpleNamespace(
            extract_stats=ExtractedCFGStats(
                static_jump_plans_resolved=1,
                exact_jump_proofs_by_flavor={"mips_pic_table": 1},
            )
        ),
        "mipsel",
    )

    summary = _GOLDEN_TEST._summary_payload(
        _GOLDEN_TEST.GoldenConfig("cfg_mode_extract", "extract"), state, None
    )
    extract = summary["extract_cfg_stats"]
    assert extract["runs"] == 2
    assert extract["totals"]["static_jump_plans_resolved"] == 3
    assert extract["affected_functions"]["static_jump_plans_resolved"] == 2
    assert extract["totals"]["exact_jump_proofs_by_flavor.mips_pic_table"] == 3
    assert extract["proofs_by_architecture"] == {
        "mips64": {"mips_pic_table": 2},
        "mipsel": {"mips_pic_table": 1},
    }

    custom = _GOLDEN_TEST._summary_payload(
        _GOLDEN_TEST.GoldenConfig("cfg_mode_custom", "custom"), state, None
    )
    assert "extract_cfg_stats" not in custom
