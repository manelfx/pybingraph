"""Golden regression tests for CFG rendering.

How this module works:

1. Test matrix
   Pytest parametrizes over every row in `playground_functions.csv` and the
   selected CFG-mode configurations. By default it uses `cfg_mode_custom`; set
   `BINGRAPH_GOLDEN_CONFIGS` to select another configuration explicitly.

2. Fresh render output
   Each test patches `get_settings()` so the app uses the requested test
   configuration, then extracts the real nested `_render_cfg()` function from
   `create_app()` and renders one CFG as raw DOT text.

3. Compare mode
   Default behavior is compare mode (`BINGRAPH_GOLDEN_MODE=compare`, or unset).
   The freshly rendered output is always written under `tests/_actual/...`.
   The test then compares that new output against the committed golden file
   under `tests/goldens/<config-name>/...`.

4. Promote mode
   When `BINGRAPH_GOLDEN_MODE=promote`, the module does not rerender CFGs.
   Instead, it copies the already generated files from `tests/_actual/...`
   into the committed golden directories under `tests/goldens/<config-name>/...`.
   This keeps promotion fast and makes it an explicit "accept what compare mode
   already produced" workflow. Promotion normalizes the transient comparison
   counters in `summary.json`, so one promotion is sufficient after adding a
   previously missing golden.

5. Summary files
   A run that selects the per-config summary test writes one `summary.json`
   file per config under `tests/_actual/<config-name>/...` and compares it
   with the committed summary. Custom-mode summaries aggregate repair counters;
   extract-mode summaries aggregate extraction counters and exact-proof usage.
   Checkpoint and other partial runs deliberately leave that full-corpus
   summary and the other `_actual` artifacts untouched.
   In promote mode, the full summary is also copied into the golden directory.

6. First-time bootstrap
   To create goldens for the first time:
   - Run compare mode. It will render files into `tests/_actual/...` and fail
     because no committed goldens exist yet.
   - Inspect the generated `_actual` files.
   - Run `BINGRAPH_GOLDEN_MODE=promote` to copy `_actual` into the golden
     directories.

7. Review workflow
   - Run compare mode to detect regressions.
   - Inspect files under `tests/_actual/...` when a mismatch occurs.
   - If the new output is correct, rerun with
     `BINGRAPH_GOLDEN_MODE=promote` and review the resulting git diff.

Useful environment variables:
   - `BINGRAPH_GOLDEN_MODE=compare|promote`
   - `BINGRAPH_GOLDEN_CONFIGS=name1,name2,...` to run only selected configs
   - `BINGRAPH_GOLDEN_MIN_BBS=<N>` to test only rows with at least `N` BBs
     (defaults to `10`)
   - `BINGRAPH_GOLDEN_LIMIT=<N>` to limit the filtered rows during local smoke tests
"""

from __future__ import annotations

import csv
import fnmatch
import json
import os
import re
import traceback
from dataclasses import asdict, dataclass, field
import shutil
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator
from unittest.mock import patch

import pytest

import bingraph.helpers as helpers_module
from bingraph.api import app as app_module
from bingraph.core import project as project_module
from bingraph.core import render as render_module
from bingraph.helpers import settings as settings_module


TESTS_DIR = Path(__file__).resolve().parents[1]
PROJECT_ROOT = TESTS_DIR.parent
CSV_PATH = TESTS_DIR / "playground_functions.csv"
PLAYGROUND_ROOT = PROJECT_ROOT / "angr-binaries" / "tests"
SUMMARY_NAME = "summary.json"
ACTUAL_ROOT = TESTS_DIR / "_actual"
GOLDENS_ROOT = TESTS_DIR / "goldens"
MODE_ENV = "BINGRAPH_GOLDEN_MODE"
CONFIGS_ENV = "BINGRAPH_GOLDEN_CONFIGS"
LIMIT_ENV = "BINGRAPH_GOLDEN_LIMIT"
MIN_BBS_ENV = "BINGRAPH_GOLDEN_MIN_BBS"
SKIPPED_BINARIES = [
    # These binaries currently make CFG golden runs disproportionately slow.
    # Keep them out of the parametrized corpus until we revisit the analysis
    # strategy for them. Curated checkpoint artifacts remain selected so this
    # does not remove regression coverage for these binary families.
    "x86_64/ALLSTAR*",
    "x86_64/decompiler*",
    "x86_64/langdetect*",
    "*lib*",
]


@dataclass(frozen=True)
class GoldenConfig:
    name: str
    cfg_mode: str
    cfg_exits: str = "jump"


CONFIGS = [
    # Compare the runtime CFG selection modes while inheriting the rest of the
    # application defaults from the real Settings model.
    GoldenConfig(name="cfg_mode_none", cfg_mode="none"),
    GoldenConfig(name="cfg_mode_custom", cfg_mode="custom"),
    # The independent extractor is experimental. Its baseline starts as a
    # copy of custom artifacts so checkpoint tests show extractor differences.
    GoldenConfig(name="cfg_mode_extract", cfg_mode="extract"),
    # Extract mode with info about external calls
    GoldenConfig(name="cfg_mode_always", cfg_mode="extract", cfg_exits="always"),
]

# Custom repair is the production CFG path and therefore the default golden
# suite. Keep the other configuration available for explicit comparisons.
DEFAULT_CONFIGS = [
    next(config for config in CONFIGS if config.name == "cfg_mode_custom")
]

# Curated regression cases that exercise repair behavior we want to protect
# while keeping a quick developer-facing golden suite. Configuration selection
# remains the responsibility of BINGRAPH_GOLDEN_CONFIGS.
CHECKPOINT_ARTIFACTS = frozenset(
    {
        "armel,lwip_udpecho_bm.elf,0x705,__udivmoddi4.dot",
        "armel,lwip_udpecho_bm.elf,0x39f1,tcp_alloc.dot",
        "armel,lwip_udpecho_bm.elf,0x5f65,dhcp_bind.dot",
        "armel,btrfs.ko,0x401154,btrfs_parse_options.dot",
        "armel,btrfs.ko,0x4230a4,btrfs_del_root.dot",
        "armel,btrfs.ko,0x438184,btrfs_writepage_fixup_worker.dot",
        "armel,btrfs.ko,0x44449c,btrfs_get_blocks_direct.dot",
        "armel,libc-2.31.so,0x46cb25,mbrtoc16.dot",
        "armel,libc-2.31.so,0x47a4e9,alarm.dot",
        "armel,libc.so.6,0x47e114,__mempcpy_small.dot",
        "armel,Nucleo_read_hyperterminal.elf,0x80080d1,__aeabi_ddiv.dot",
        "i386,bronze_ropchain,0x8049930,plural_eval.dot",
        "i386,bronze_ropchain,0x80572f0,_IO_list_lock.dot",
        "i386,bronze_ropchain,0x805ec40,__memset_sse2.dot",
        "i386,bronze_ropchain,0x8060ca0,__strcmp_sse4_2.dot",
        "i386,bronze_ropchain,0x8062690,__memcmp_sse4_2.dot",
        "i386,bronze_ropchain,0x806a770,__strcasecmp_l_sse4_2.dot",
        "i386,bronze_ropchain,0x806f870,_dl_aux_init.dot",
        "i386,bronze_ropchain,0x806b8e0,handle_amd.dot",
        "i386,bronze_ropchain,0x809cc70,_dl_mcount.dot",
        "i386,bronze_ropchain,0x80a7db0,execute_stack_op.dot",
        "mips,dir,0x40f6f4,hash_initialize.dot",
        "mipsel,mips_syscall_demo,0x401390,__libc_setup_tls.dot",
        "mipsel,mips_syscall_demo,0x420240,__gconv_db_freemem.dot",
        "mipsel,btrfs-tools_btrfs-calc-size,0x4162c8,btrfs_find_block_group.isra.14.dot",
        "ppc64el,fauxware_static,0x100014a0,__libc_check_standard_fds.dot",
        "ppc64el,fauxware_static,0x10002390,plural_eval.dot",
        "ppc64el,fauxware_static,0x1000ed70,abort.dot",
        "ppc64el,fauxware_static,0x1000f0c0,msort_with_tmp.part.0.dot",
        "ppc64el,fauxware_static,0x10019100,flush_cleanup.dot",
        "ppc64el,fauxware_static,0x1001e4c0,malloc_consolidate.dot",
        "ppc64el,fauxware_static,0x1003ac00,__gconv_release_step.dot",
        "ppc,libc.so.6,0x43f300,initstate.dot",
        "riscv,server_eapp.eapp_riscv,0x1830,channel_init.dot",
        "riscv,autotalent-autotalent.so,0x403c50,_init.dot",
        "riscv,server_eapp.eapp_riscv,0x4416,crypto_core_salsa.dot",
        "riscv,server_eapp.eapp_riscv,0x7e54,crypto_scalarmult_curve25519_ref10.dot",
        "riscv,server_eapp.eapp_riscv,0xe60c,crypto_generichash_blake2b__init_salt_personal.dot",
        "riscv,server_eapp.eapp_riscv,0xeb6c,blake2b_compress_ref.dot",
        "s390x,test-instr_s390x,0x8001d140,__gconv.dot",
        "s390x,test-instr_s390x,0x800555f8,_IO_vfscanf.dot",
        "s390x,libc.so.6,0x48ee08,__libc_mallopt.dot",
        "x86_64,cvs,0x485f00,vasnprintf.dot",
        "x86_64,calc,0x403c51,__wait.dot",
        "x86_64,decompiler,clientloop.o,0x405f00,client_loop.dot",
        "x86_64,elf_with_static_libc_ubuntu_2004,0x445970,__memset_avx512_no_vzeroupper.dot",
        "x86_64,elf_with_static_libc_ubuntu_2004,0x48ef40,execute_stack_op.dot",
        "x86_64,langdetect_clang,0x408ce0,msort_with_tmp.part.0.dot",
        "x86_64,langdetect_clang,0x420c40,__memcpy_avx512_unaligned_erms.dot",
        "x86_64,langdetect_clang,0x423590,__memset_avx512_no_vzeroupper.dot",
        "x86_64,langdetect_clang,0x420450,__memcpy_avx512_no_vzeroupper.dot",
        "x86_64,langdetect_clang,0x435640,__strstr_avx512.dot",
        "x86_64,langdetect_clang,0x465ee0,_dl_mcount.dot",
        "x86_64,langdetect_clang,0x46b680,__lll_lock_elision.dot",
        "x86_64,langdetect_clang,0x4741a0,execute_cfa_program.dot",
        "x86_64,langdetect_clang,0x475e90,_Unwind_Resume_or_Rethrow.dot",
        "x86_64,rust_hello_world,0x4207f0,_ZN3std3env11current_exe17hfb9bee2aecec296fE.dot",
        "x86_64,libc.so.6,0x4370d0,sigwait.dot",
        "x86_64,static,0x40dc00,abort.dot",
        "x86_64,veritesting_skm,0x4018a0,lexer_get_next_token.dot",
    }
)


@dataclass
class ConfigRunState:
    """Accumulate render and CFG-analysis results for one configuration."""

    expected_files: set[Path]
    entries: int = 0
    render_successes: int = 0
    render_failures: int = 0
    golden_matches: int = 0
    golden_mismatches: int = 0
    missing_goldens: int = 0
    render_crashes: int = 0
    render_recoveries: int = 0
    custom_cfg_runs: int = 0
    custom_cfg_stats: dict[str, int] = field(default_factory=dict)
    extract_cfg_runs: int = 0
    extract_cfg_stats: dict[str, int] = field(default_factory=dict)
    extract_cfg_affected_functions: dict[str, int] = field(default_factory=dict)
    extract_proofs_by_architecture: dict[str, dict[str, int]] = field(
        default_factory=dict
    )


def _iter_rows(limit: int | None = None) -> Iterator[dict[str, Any]]:
    """Yield normalized CSV rows for parametrized golden tests."""

    min_bbs = _env_min_bbs()
    yielded = 0
    seen_function_formats: set[tuple[str, str]] = set()

    # Only non-duplicate rows take part in the golden suite. Apply the BB-count
    # filter before the optional smoke-test limit so the limit reflects the
    # exact number of collected test rows, not raw CSV line numbers.
    with CSV_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, row in enumerate(reader, start=1):
            normalized_row = {
                "index": index,
                "filepath": row["filepath"],
                "function_name": row["funcname"],
                "function_addr": row["funcaddr"],
                "num_bbs": int(row["num_bbs"]),
            }
            is_checkpoint = _is_checkpoint(normalized_row)

            # Checkpoints are a curated regression contract. They must remain
            # part of the normal full golden matrix even when their binary is
            # skipped, their CSV row is duplicate, or it falls below the BB
            # threshold used to reduce broad corpus coverage.
            if (
                any(
                    fnmatch.fnmatch(row["filepath"], pattern)
                    for pattern in SKIPPED_BINARIES
                )
                and not is_checkpoint
            ):
                continue
            if _csv_bool(row["duplicate"]) and not is_checkpoint:
                continue
            if int(row["num_bbs"]) < min_bbs and not is_checkpoint:
                continue

            # Keep one deterministic representative for each format/name pair.
            # This runs after the BB threshold so a small first variant cannot
            # hide a later variant that is otherwise eligible for CFG testing.
            # Curated checkpoint cases remain selected in addition to the
            # ordinary representative for their format/name group.
            function_format = (row["fileformat"], row["funcname"])
            if function_format in seen_function_formats and not is_checkpoint:
                continue
            seen_function_formats.add(function_format)

            yielded += 1
            yield normalized_row
            if limit is not None and yielded >= limit:
                return


def _csv_bool(value: str) -> bool:
    """Parse a lowercase CSV boolean field into a Python bool."""

    return value.strip().lower() == "true"


def _sanitize_filename(value: str, max_length: int = 80) -> str:
    """Convert arbitrary text into a filesystem-safe path fragment."""

    sanitized = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip()).strip("._")
    if not sanitized:
        sanitized = "unknown"
    return sanitized[:max_length]


def _sanitize_flat_artifact_part(value: str, max_length: int = 160) -> str:
    """Convert text into a flat filename fragment while preserving commas."""

    sanitized = re.sub(r"[^A-Za-z0-9,._-]+", "_", value.strip()).strip(".,")
    if not sanitized:
        sanitized = "unknown"
    return sanitized[:max_length]


def _artifact_relative_path(row: dict[str, Any]) -> Path:
    """Build the stable relative artifact path for one CSV entry."""

    # Keep artifacts flat under each config directory so the original CSV path
    # can be inferred directly from the filename, without recreating folders.
    filepath_part = _sanitize_flat_artifact_part(row["filepath"].replace("/", ","))
    funcaddr_part = _sanitize_flat_artifact_part(row["function_addr"])
    funcname_part = _sanitize_flat_artifact_part(row["function_name"])
    return Path(f"{filepath_part},{funcaddr_part},{funcname_part}.dot")


def _is_checkpoint(row: dict[str, Any]) -> bool:
    """Return whether one corpus row belongs to the curated smoke suite."""

    return _artifact_relative_path(row).as_posix() in CHECKPOINT_ARTIFACTS


def _assert_all_checkpoints_selected(rows: Iterable[dict[str, Any]]) -> None:
    """Fail collection when a full golden run omits a declared checkpoint."""

    selected = {_artifact_relative_path(row).as_posix() for row in rows}
    missing = sorted(CHECKPOINT_ARTIFACTS - selected)
    if missing:
        raise ValueError(
            "Full golden selection omitted checkpoint artifact(s): "
            f"{', '.join(missing)}"
        )


def _summary_payload(
    config: GoldenConfig,
    state: ConfigRunState,
    limit: int | None,
) -> dict[str, Any]:
    """Serialize one configuration run-state into summary.json fields."""

    payload = {
        "config": asdict(config),
        "csv": str(CSV_PATH.relative_to(PROJECT_ROOT)),
        "root": str(PLAYGROUND_ROOT.relative_to(PROJECT_ROOT)),
        "entries": state.entries,
        "render_successes": state.render_successes,
        "render_failures": state.render_failures,
        "golden_matches": state.golden_matches,
        "golden_mismatches": state.golden_mismatches,
        "missing_goldens": state.missing_goldens,
        "render_crashes": state.render_crashes,
        "render_recoveries": state.render_recoveries,
        # These counters are aggregated from completed custom-repair sessions,
        # not inferred from DOT differences, so they distinguish repair work
        # from rendering-only layout or label changes.
        "custom_cfg_stats": {
            "runs": state.custom_cfg_runs,
            "totals": dict(sorted(state.custom_cfg_stats.items())),
        },
        "format": "raw",
        "artifact_extension": ".dot",
        "limit": limit,
        "min_bbs": _env_min_bbs(),
    }
    if config.cfg_mode == "extract":
        payload["extract_cfg_stats"] = {
            "runs": state.extract_cfg_runs,
            "totals": dict(sorted(state.extract_cfg_stats.items())),
            "affected_functions": dict(
                sorted(state.extract_cfg_affected_functions.items())
            ),
            "proofs_by_architecture": {
                arch: dict(sorted(flavors.items()))
                for arch, flavors in sorted(
                    state.extract_proofs_by_architecture.items()
                )
            },
        }
    return payload


def _serialize_summary(payload: dict[str, Any]) -> str:
    """Render summary metadata as deterministic JSON text."""

    return json.dumps(payload, indent=2, sort_keys=True) + "\n"


def _error_artifact(exc: Exception) -> str:
    """Convert a render exception into a persisted pseudo-artifact."""

    # Persist the full traceback alongside the exception summary so a failed run
    # can be debugged directly from the generated artifact without reproducing it.
    traceback_text = "".join(
        traceback.format_exception(type(exc), exc, exc.__traceback__)
    )
    return f"# render-error\n{type(exc).__name__}: {exc}\n\n{traceback_text}"


def _is_error_artifact(text: str) -> bool:
    """Return True when an artifact contains a captured render exception."""

    return text.startswith("# render-error\n")


def _clear_caches() -> None:
    """Reset cached CFG helpers before pytest switches to another config."""

    # The app and CFG helpers are cached globally. Clear them when pytest moves
    # into a different golden configuration so rows inside the same config can
    # reuse the already-loaded project and CFG objects.
    render_module.render_cfg.cache_clear()
    project_module._get_project.cache_clear()
    project_module.get_cfg.cache_clear()


def _record_custom_cfg_stats(state: ConfigRunState, cfg: object) -> None:
    """Add one completed custom CFG wrapper's repair counters to run state."""

    stats = getattr(cfg, "custom_stats", None)
    if stats is None:
        return

    state.custom_cfg_runs += 1
    for name, value in stats.as_dict().items():
        state.custom_cfg_stats[name] = state.custom_cfg_stats.get(name, 0) + value


def _record_extract_cfg_stats(
    state: ConfigRunState, cfg: object, architecture: str
) -> None:
    """Count completed extract actions and the functions that use each one."""

    stats = getattr(cfg, "extract_stats", None)
    if stats is None:
        return

    state.extract_cfg_runs += 1
    for name, value in stats.as_dict().items():
        entries = value.items() if isinstance(value, dict) else ((None, value),)
        for subname, count in entries:
            key = f"{name}.{subname}" if subname is not None else name
            state.extract_cfg_stats[key] = state.extract_cfg_stats.get(key, 0) + count
            if count:
                state.extract_cfg_affected_functions[key] = (
                    state.extract_cfg_affected_functions.get(key, 0) + 1
                )
                if name == "exact_jump_proofs_by_flavor":
                    by_arch = state.extract_proofs_by_architecture.setdefault(
                        architecture, {}
                    )
                    by_arch[subname] = by_arch.get(subname, 0) + count


def _extract_render_cfg() -> Callable[..., str]:
    """Extract the nested `_render_cfg` callable from the real FastAPI app."""

    # `_render_cfg` is nested inside `create_app()`, so pull it out from the
    # `/api/cfg` route closure instead of duplicating application logic here.
    app = app_module.create_app()
    for route in app.routes:
        if getattr(route, "path", None) != "/api/cfg":
            continue
        for cell in route.endpoint.__closure__ or ():
            candidate = cell.cell_contents
            if (
                callable(candidate)
                and getattr(candidate, "__name__", "") == "_render_cfg"
            ):
                return candidate

    raise AssertionError("Unable to extract _render_cfg from create_app()")


def _existing_files(config_dir: Path) -> set[Path]:
    """Return every file currently present under a generated artifact directory."""

    if not config_dir.exists():
        return set()
    return {
        path.relative_to(config_dir) for path in config_dir.rglob("*") if path.is_file()
    }


def _prune_stale_files(config_dir: Path, expected_files: set[Path]) -> None:
    """Delete generated files that are no longer expected for a given run."""

    # Keep generated directories tidy when row limits change or old artifacts disappear.
    for stale_path in sorted(
        _existing_files(config_dir) - expected_files, reverse=True
    ):
        (config_dir / stale_path).unlink()
    for directory in sorted(
        (path for path in config_dir.rglob("*") if path.is_dir()), reverse=True
    ):
        try:
            directory.rmdir()
        except OSError:
            pass


def _env_limit() -> int | None:
    """Read the optional CSV row limit used for small local smoke tests."""

    raw_limit = os.getenv(LIMIT_ENV)
    if raw_limit is None or raw_limit == "":
        return None
    return int(raw_limit)


def _env_min_bbs() -> int:
    """Read the minimum BB threshold applied before test parametrization."""

    raw_value = os.getenv(MIN_BBS_ENV)
    if raw_value is None or raw_value == "":
        return 10
    return int(raw_value)


def _golden_mode() -> str:
    """Return the active golden workflow mode from the environment."""

    # Two explicit workflows:
    # - compare: write `_actual`, compare against committed goldens
    # - promote: copy previously generated `_actual` files into the golden tree
    mode = os.getenv(MODE_ENV, "compare").strip().lower()
    valid_modes = {"compare", "promote"}
    if mode not in valid_modes:
        raise ValueError(f"{MODE_ENV} must be one of: {', '.join(sorted(valid_modes))}")
    return mode


def _selected_configs() -> list[GoldenConfig]:
    """Return the active config subset requested through the environment."""

    raw_configs = os.getenv(CONFIGS_ENV, "").strip()
    if not raw_configs:
        return DEFAULT_CONFIGS

    available = {config.name: config for config in CONFIGS}
    selected_names = [name.strip() for name in raw_configs.split(",") if name.strip()]
    unknown = [name for name in selected_names if name not in available]
    if unknown:
        raise ValueError(
            f"{CONFIGS_ENV} contains unknown config(s): {', '.join(unknown)}. "
            f"Valid values: {', '.join(sorted(available))}"
        )

    # Preserve the order requested by the user while silently deduplicating repeats.
    selected: list[GoldenConfig] = []
    seen: set[str] = set()
    for name in selected_names:
        if name not in seen:
            selected.append(available[name])
            seen.add(name)
    return selected


def _row_id(row: dict[str, Any]) -> str:
    """Build a readable pytest id for one CSV-driven row case."""

    return (
        f"{_sanitize_filename(row['filepath'].replace('/', ','), 32)}-"
        f"{_sanitize_filename(row['function_addr'], 18)}-"
        f"{_sanitize_filename(row['function_name'], 24)}"
    )


def _copy_tree_contents(src_dir: Path, dst_dir: Path) -> None:
    """Copy one generated `_actual` tree into the committed golden tree."""

    # Copy the generated `_actual` tree into the committed golden tree, replacing
    # files in place and removing stale golden files that no longer exist in `_actual`.
    expected_files = _existing_files(src_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for relpath in expected_files:
        target = dst_dir / relpath
        target.parent.mkdir(parents=True, exist_ok=True)
        if relpath == Path(SUMMARY_NAME):
            target.write_text(
                _normalized_promoted_summary_text(src_dir / relpath),
                encoding="utf-8",
            )
            continue
        shutil.copy2(src_dir / relpath, target)
    _prune_stale_files(dst_dir, expected_files)


def _normalized_promoted_summary_text(actual_summary_path: Path) -> str:
    """Return a summary representing the successful run promotion establishes."""

    payload = json.loads(actual_summary_path.read_text(encoding="utf-8"))
    entries = payload.get("entries")
    if not isinstance(entries, int):
        raise ValueError(
            f"Cannot promote malformed summary without integer entries: "
            f"{actual_summary_path}"
        )

    # `_actual` records the comparison result of the run that produced it.
    # Once its artifacts become goldens, the next equivalent run must instead
    # report an all-matching comparison. Preserve render results (including
    # expected error artifacts), but reset only golden-comparison counters.
    payload.update(
        golden_matches=entries,
        golden_mismatches=0,
        missing_goldens=0,
        render_crashes=0,
        render_recoveries=0,
    )
    return _serialize_summary(payload)


def _build_test_settings(config: GoldenConfig) -> settings_module.Settings:
    """Build mocked settings from the real Settings model and its defaults."""

    # Reuse the production settings model so test defaults track the real app
    # defaults automatically. We only override the fields that must differ for
    # the golden-suite environment. Golden CFG comparisons currently run with
    # comments disabled so comment-label churn does not hide structural CFG
    # differences between modes.
    return settings_module.Settings.model_construct(
        root=PLAYGROUND_ROOT,
        cfg_mode=config.cfg_mode,
        cfg_exits=config.cfg_exits,
        comments=False,
        server=None,
        client=None,
    )


CURRENT_MODE = _golden_mode()
ACTIVE_CONFIGS = _selected_configs()
ROWS = tuple(_iter_rows(_env_limit())) if CURRENT_MODE == "compare" else ()
if CURRENT_MODE == "compare" and _env_limit() is None:
    _assert_all_checkpoints_selected(ROWS)


def _golden_cases():
    """Build ordered `(config, row)` cases with checkpoint marks where needed."""

    return tuple(
        pytest.param(
            config,
            row,
            id=f"{config.name}-{_row_id(row)}",
            marks=pytest.mark.checkpoint if _is_checkpoint(row) else (),
        )
        for config in ACTIVE_CONFIGS
        for row in ROWS
    )


GOLDEN_CASES = _golden_cases()
RUN_STATE: dict[str, ConfigRunState] = {}
ACTIVE_CACHE_CONFIG: str | None = None


def _summary_test_is_selected(request: pytest.FixtureRequest) -> bool:
    """Return whether this pytest invocation selected the summary test."""

    summary_nodeid = f"{request.node.nodeid}::test_config_summary_matches_golden"
    return any(item.nodeid.startswith(summary_nodeid) for item in request.session.items)


@pytest.fixture(scope="module", autouse=True)
def _manage_summary_files(request: pytest.FixtureRequest) -> Iterator[None]:
    """Initialize run state and always materialize actual summaries at teardown."""

    global RUN_STATE, ACTIVE_CACHE_CONFIG

    mode = CURRENT_MODE
    if mode != "compare":
        # Promote mode is a pure filesystem copy and should not touch run stats.
        yield
        return

    # Track per-config aggregate stats as the parametrized row tests execute.
    RUN_STATE = {
        config.name: ConfigRunState(expected_files={Path(SUMMARY_NAME)})
        for config in ACTIVE_CONFIGS
    }

    yield

    if not _summary_test_is_selected(request):
        # A marker or explicit node selection can run only part of the corpus.
        # Do not replace the complete summary with partial counters or prune
        # the remaining full-corpus `_actual` artifacts in that case.
        RUN_STATE = {}
        ACTIVE_CACHE_CONFIG = None
        _clear_caches()
        return

    for config in ACTIVE_CONFIGS:
        _write_config_summary(config, validate=False)

    RUN_STATE = {}
    ACTIVE_CACHE_CONFIG = None
    _clear_caches()


def _write_config_summary(config: GoldenConfig, *, validate: bool) -> list[str]:
    """Write one actual summary and optionally return committed-summary differences."""

    state = RUN_STATE[config.name]
    golden_dir = GOLDENS_ROOT / config.name
    actual_dir = ACTUAL_ROOT / config.name
    actual_summary_path = actual_dir / SUMMARY_NAME
    golden_summary_path = golden_dir / SUMMARY_NAME
    summary_text = _serialize_summary(
        _summary_payload(
            config=config,
            state=state,
            limit=_env_limit(),
        )
    )

    actual_dir.mkdir(parents=True, exist_ok=True)
    actual_summary_path.write_text(summary_text, encoding="utf-8")
    _prune_stale_files(actual_dir, state.expected_files)

    if not validate or _env_limit() is not None:
        # Limited smoke runs intentionally do not validate committed summaries:
        # their counters represent only a partial view of the selected corpus.
        return []

    failures: list[str] = []
    if not golden_summary_path.exists():
        failures.append(f"missing {golden_summary_path.relative_to(TESTS_DIR)}")
    elif golden_summary_path.read_text(encoding="utf-8") != summary_text:
        failures.append(f"mismatch in {golden_summary_path.relative_to(TESTS_DIR)}")

    unexpected_files = sorted(_existing_files(golden_dir) - state.expected_files)
    failures.extend(f"unexpected golden file {path}" for path in unexpected_files)
    return failures


if CURRENT_MODE == "compare":

    @pytest.fixture(scope="function")
    def _config_cache_scope(config: GoldenConfig) -> Iterator[None]:
        """Reset cached project state only when pytest switches config groups."""

        global ACTIVE_CACHE_CONFIG

        # Pytest iterates config first, then row, so this clears caches once per
        # config and lets every row inside that config reuse the same warm caches.
        if ACTIVE_CACHE_CONFIG != config.name:
            _clear_caches()
            ACTIVE_CACHE_CONFIG = config.name

        yield

    @pytest.mark.slow
    @pytest.mark.parametrize(("config", "row"), GOLDEN_CASES)
    def test_render_cfg_goldens(
        config: GoldenConfig,
        row: dict[str, Any],
        _config_cache_scope: None,
    ) -> None:
        """Render one CFG case, store `_actual`, and compare it against its golden."""

        golden_dir = GOLDENS_ROOT / config.name
        actual_dir = ACTUAL_ROOT / config.name

        # Build the runtime settings object that the real app code will consult.
        settings = _build_test_settings(config)
        artifact_relpath = _artifact_relative_path(row)
        golden_path = golden_dir / artifact_relpath
        actual_path = actual_dir / artifact_relpath
        state = RUN_STATE[config.name]
        state.entries += 1
        state.expected_files.add(artifact_relpath)

        # Patch every module that imports `get_settings()` directly so the render path
        # sees a coherent configuration from app entrypoint down to CFG generation.
        with (
            patch.object(settings_module, "get_settings", return_value=settings),
            patch.object(helpers_module, "get_settings", return_value=settings),
            patch.object(app_module, "get_settings", return_value=settings),
            patch.object(project_module, "get_settings", return_value=settings),
        ):
            render_cfg = _extract_render_cfg()
            original_build_custom_cfg = project_module.build_custom_cfg
            original_build_extracted_cfg = project_module.build_extracted_cfg

            def build_custom_cfg_with_stats(*args: Any, **kwargs: Any) -> Any:
                """Preserve repair counters while delegating to real CFG building."""

                cfg = original_build_custom_cfg(*args, **kwargs)
                _record_custom_cfg_stats(state, cfg)
                return cfg

            def build_extracted_cfg_with_stats(*args: Any, **kwargs: Any) -> Any:
                """Preserve extraction counters while delegating to real CFG building."""

                cfg = original_build_extracted_cfg(*args, **kwargs)
                architecture = row["filepath"].split("/", 1)[0]
                _record_extract_cfg_stats(state, cfg, architecture)
                return cfg

            try:
                with (
                    patch.object(
                        project_module,
                        "build_custom_cfg",
                        side_effect=build_custom_cfg_with_stats,
                    ),
                    patch.object(
                        project_module,
                        "build_extracted_cfg",
                        side_effect=build_extracted_cfg_with_stats,
                    ),
                ):
                    artifact_text = render_cfg(
                        row["filepath"], row["function_addr"], format="raw"
                    )
                state.render_successes += 1
            except Exception as exc:  # pragma: no cover - exercised against real corpus
                artifact_text = _error_artifact(exc)
                state.render_failures += 1

        # Always keep the newly rendered candidate output on disk for inspection.
        actual_path.parent.mkdir(parents=True, exist_ok=True)
        actual_path.write_text(artifact_text, encoding="utf-8")

        # Compare mode requires an existing committed golden file.
        if not golden_path.exists():
            state.missing_goldens += 1
            pytest.fail(
                f"Missing golden file for {config.name}: {artifact_relpath}\n"
                f"Review {actual_path.relative_to(TESTS_DIR)} and rerun with {MODE_ENV}=promote to create or update goldens."
            )
        expected_text = golden_path.read_text(encoding="utf-8")
        if expected_text != artifact_text:
            state.golden_mismatches += 1
            # Distinguish between three important cases in test output:
            # - the new run crashed
            # - the old golden expected a crash but the run recovered
            # - both are outputs, but the output changed
            if _is_error_artifact(artifact_text):
                state.render_crashes += 1
                pytest.fail(
                    f"Render crashed for {config.name}: {artifact_relpath}\n"
                    f"Inspect {actual_path.relative_to(TESTS_DIR)} for the captured exception.\n"
                    f"Promote with {MODE_ENV}=promote only if this error artifact is the new expected result."
                )
            if _is_error_artifact(expected_text):
                state.render_recoveries += 1
                pytest.fail(
                    f"Render recovered for {config.name}: {artifact_relpath}\n"
                    f"The committed golden currently expects an error artifact, but the new run produced output.\n"
                    f"Inspect {actual_path.relative_to(TESTS_DIR)} and promote with {MODE_ENV}=promote if recovery is expected."
                )
            pytest.fail(
                f"Render output changed for {config.name}: {artifact_relpath}\n"
                f"Inspect {actual_path.relative_to(TESTS_DIR)} and promote with {MODE_ENV}=promote if the new output is correct."
            )
        # Only exact output matches count as successful golden comparisons.
        state.golden_matches += 1

    @pytest.mark.slow
    @pytest.mark.parametrize("config", ACTIVE_CONFIGS, ids=lambda config: config.name)
    def test_config_summary_matches_golden(config: GoldenConfig) -> None:
        """Compare one config summary after all of its artifact cases complete."""

        failures = _write_config_summary(config, validate=True)
        if not failures:
            return

        details = "\n".join(f"- {failure}" for failure in failures)
        pytest.fail(
            f"Golden summary validation failed for {config.name}.\n"
            "This usually follows a corpus or checkpoint selection change; "
            "inspect the generated summary and promote it with the artifacts if intended.\n"
            f"{details}"
        )


@pytest.mark.slow
def test_promote_actual_to_goldens() -> None:
    """Copy previously generated `_actual` artifacts into the golden directories."""

    if CURRENT_MODE != "promote":
        pytest.skip("Promotion copy step only runs when BINGRAPH_GOLDEN_MODE=promote.")

    # Promote mode assumes compare mode already generated the `_actual` tree.
    assert ACTUAL_ROOT.exists(), (
        f"Missing {ACTUAL_ROOT.relative_to(TESTS_DIR)} directory.\n"
        f"Run compare mode first to generate candidate outputs before promoting."
    )

    for config in ACTIVE_CONFIGS:
        actual_dir = ACTUAL_ROOT / config.name
        golden_dir = GOLDENS_ROOT / config.name
        assert actual_dir.exists(), (
            f"Missing {actual_dir.relative_to(TESTS_DIR)}.\n"
            f"Run compare mode first so there is something to promote."
        )
        assert (actual_dir / SUMMARY_NAME).exists(), (
            f"Missing summary file in {actual_dir.relative_to(TESTS_DIR)}.\n"
            f"Run compare mode first so promotion copies a complete artifact set."
        )
        # Promotion is a pure copy operation: no rerendering, just accept what
        # compare mode already wrote into `_actual`.
        _copy_tree_contents(actual_dir, golden_dir)
