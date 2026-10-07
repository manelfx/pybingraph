from __future__ import annotations
from functools import lru_cache
from pathlib import Path
import traceback
from typing import Any, cast

from angr import Project, KnowledgeBase
from angr.analyses import CFGFast
from angr.analyses.cfg import CFGBase
from loguru import logger

from bingraph.helpers import time_it, get_settings, CfgMode
from bingraph.cfg import build_custom_cfg
from bingraph.cfg.decode import decode_raw_capstone_insns
from bingraph.helpers.capstone import InsnSemantics
from bingraph.helpers.symbols import list_function_symbols


@lru_cache
def _get_project(spath: str, mtime: float) -> Project:
    """
    Return an angr project from a cached binary image.

    Args:
        spath (str): The file path of the binary to analyze.
        mtime (float): Binary modification time -- note this is not used but is still cached,
            so updated binaries are not incorrectly cached.

    Returns:
        Project: The loaded angr project.
    """
    return Project(spath, auto_load_libs=False)


def load_project(path: Path) -> Project:
    """
    Load an angr project from the specified path.

    Args:
        path (Path): The file path of the binary to analyze.

    Returns:
        Project: The loaded angr project.
    """
    return _get_project(str(path), path.stat().st_mtime)


def _terminal_unmapped_return_callee(
    project: Project,
    func_addr: int,
    func_size: int,
    known_function_addrs: set[int],
) -> int | None:
    """Return a known terminal callee whose fake return would be unmapped.

    CFGFast models a direct call as returning until it proves otherwise. If the
    call is the final instruction in a function and the architectural return
    address is not mapped, its post-analysis tries to lift that impossible
    continuation and raises ``SimEngineError``. Predeclaring the callee as
    non-returning avoids creating that invalid fake-return job. Restrict this
    to known direct callees so ordinary calls keep CFGFast's normal return
    inference.
    """

    func_end = func_addr + func_size
    insns = decode_raw_capstone_insns(project, func_addr, func_size)
    if not insns:
        return None

    last_insn = insns[-1]
    if last_insn.address + last_insn.size != func_end:
        return None

    semantics = InsnSemantics(last_insn)
    callee_addr = semantics.direct_target()
    if not semantics.is_call() or callee_addr is None or callee_addr == func_addr:
        return None
    if callee_addr not in known_function_addrs:
        return None

    try:
        project.loader.memory.load(func_end, 1)
    except Exception:
        return callee_addr
    return None


def _get_fast_cfg(project: Project, kb: KnowledgeBase, func_addr: int) -> CFGFast:
    """
    Build or retrieve the fast control flow graph (CFGFast) for the given project.

    Args:
        project (Project): The angr project for which to build the fast CFG.
        kb (KnowledgeBase): Shared knowledge base reused across CFG strategies.
        func_addr (int): Address of the target function.

    Returns:
        CFGFast: The fast control flow graph of the project.
    """
    # Find the function symbol first so we can bound CFGFast to its address range.
    function_symbols = list_function_symbols(project)
    function = next((sym for sym in function_symbols if sym.addr == func_addr), None)
    if not function:
        raise KeyError(f"Function {func_addr:#x} not found binary")
    regions = [(func_addr, func_addr + function.size)]
    logger.info(
        f"Region for CFG reconstruct will be {[(hex(a), hex(b)) for a, b in regions]}"
    )

    nonreturning_terminal_callee = _terminal_unmapped_return_callee(
        project,
        func_addr,
        function.size,
        {symbol.addr for symbol in function_symbols},
    )

    def _should_retry_cfgfast_with_safer_settings(exc: Exception) -> bool:
        """
        Return True when CFGFast should be retried with safer bounded settings.

        We have seen two angr failure modes on our region-bounded CFGFast runs:

        1. Smart-scan post-processing can dereference a `None` block
           (`AttributeError: 'NoneType' object has no attribute 'addr'`).
        2. Data-reference collection can fail inside Clinic/StackPointerTracker
           with a `KeyError(<callee-addr>)` when the bounded knowledge base does
           not contain metadata for out-of-region callees.

        3. Syscall resolution can assert while applying missing numerical
           metadata to an angr syscall stub.

        In all cases we retry once with `force_smart_scan=False`, which also
        disables `data_references` in `build_cfg()`. The syscall case also
        disables indirect-jump resolution to avoid its failing resolver.
        """
        if isinstance(exc, AttributeError):
            return "'NoneType' object has no attribute 'addr'" in str(exc)

        if isinstance(exc, KeyError):
            return True

        if not isinstance(exc, AssertionError):
            return False
        return any(
            frame.name == "_apply_numerical_metadata"
            and frame.filename.endswith("angr/procedures/definitions/__init__.py")
            for frame in traceback.extract_tb(exc.__traceback__)
        )

    # Create a fresh knowledge base so this analysis does not pollute the project state.
    def build_cfg(*, force_smart_scan: bool, resolve_indirect_jumps: bool) -> CFGFast:
        if nonreturning_terminal_callee is not None:
            # The return site is unmapped, so this terminal call cannot have a
            # valid in-image continuation. Tell CFGFast before it creates a
            # fake-return job that would later try to lift the missing bytes.
            callee = kb.functions.function(nonreturning_terminal_callee, create=True)
            if callee is not None:
                callee.returning = False

        return project.analyses.CFGFast(
            kb=kb,
            # we already know the exact entry point we want
            function_starts=[func_addr],
            # big performance win, do not analyze the full binary
            regions=regions,
            # avoid extra function discovery heuristics, already gave the function start explicitly
            eh_frame=False,
            exceptions=False,
            force_complete_scan=False,
            function_prologues=False,
            start_at_entry=False,
            symbols=False,
            # Enable smarter basic-block discovery, but keep it constrained to the
            # requested function region.
            data_references=force_smart_scan,
            force_smart_scan=force_smart_scan,
            resolve_indirect_jumps=resolve_indirect_jumps,
            # stable, clean function graphs for rendering
            normalize=True,
        )

    # Prefer the smarter region-bounded scan, but retry without it for the
    # specific angr post-processing crash patterns observed in the corpus.
    try:
        cfg = build_cfg(force_smart_scan=True, resolve_indirect_jumps=True)
    except Exception as exc:
        if not _should_retry_cfgfast_with_safer_settings(exc):
            raise
        logger.warning(
            f"Retrying CFGFast with safer settings for {func_addr:#x} after angr crash: {exc}"
        )
        return build_cfg(
            force_smart_scan=False,
            resolve_indirect_jumps=not isinstance(exc, AssertionError),
        )

    # Smart scanning can split a valid ARM function into interior functions
    # despite the explicit start address. Rendering requires the requested
    # function to exist in the knowledge base, so retry the bounded scan with
    # the conservative strategy when that postcondition is not met.
    if cfg.kb.functions.get(func_addr) is None:
        logger.warning(
            f"Retrying CFGFast without smart scan because {func_addr:#x} "
            "was not retained as a function entry"
        )
        return build_cfg(force_smart_scan=False, resolve_indirect_jumps=True)

    return cfg


def get_cfg(
    project: Project,
    func_addr: int,
    cfg_mode: CfgMode | None = None,
) -> CFGBase:
    """Resolve the CFG mode before caching its graph."""

    return _get_cfg(project, func_addr, cfg_mode or get_settings().cfg_mode)


@lru_cache
@time_it
def _get_cfg(project: Project, func_addr: int, resolved_cfg_mode: CfgMode) -> CFGBase:
    """Build a graph with fully resolved settings forming its cache key."""

    logger.info(
        f"Getting CFG for function {func_addr:#x} with mode '{resolved_cfg_mode}'"
    )
    # Isolate analysis metadata between CFG requests and strategies.
    kb = KnowledgeBase(project)
    if resolved_cfg_mode == "none":
        return _get_fast_cfg(project, kb, func_addr)
    if resolved_cfg_mode == "custom":
        # CustomCFG intentionally exposes the CFGBase subset consumed by the
        # rest of bingraph, but angr's nominal type hierarchy cannot express it.
        return cast(CFGBase, build_custom_cfg(project, kb, func_addr))
    raise ValueError(f"Unsupported cfg mode: {resolved_cfg_mode}")


# Preserve the cache-control interface used by corpus runners and callers.
cast(Any, get_cfg).cache_clear = _get_cfg.cache_clear
cast(Any, get_cfg).cache_info = _get_cfg.cache_info
