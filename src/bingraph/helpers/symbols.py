"""Function-symbol discovery and size inference for loaded binaries."""

from __future__ import annotations
from functools import lru_cache
from itertools import groupby

from loguru import logger
from pydantic import BaseModel
from angr import Project, KnowledgeBase


UNKNOWN_FUNC = "**UNKNOWN**"


def plt_symbol_name(project: Project, addr: int) -> str | None:
    """Resolve a PLT stub address to its imported function name."""

    obj = project.loader.find_object_containing(addr)
    if obj is None:
        return None
    name = getattr(obj, "reverse_plt", {}).get(addr)
    return name if isinstance(name, str) and name else None


class FunctionSymbol(BaseModel):
    addr: int
    name: str
    size: int
    is_import: bool
    origin: str


@lru_cache
def list_function_symbols(project: Project) -> list[FunctionSymbol]:

    logger.info("Getting function symbols from symbol table")

    # grab functions from binary symbol table
    symbols = [
        FunctionSymbol(
            name=symbol.name or UNKNOWN_FUNC,
            addr=symbol.rebased_addr,
            size=symbol.size,
            is_import=symbol.is_import,
            origin="symtab",
        )
        for symbol in project.loader.main_object.symbols
        if symbol.is_function
    ]

    if not symbols:
        logger.info(
            "No function symbols were found (stripped binary?), analyzing binary "
        )

        # create clean knowledge base object, so project is not pulluted with this analysis
        kb = KnowledgeBase(project)

        # Super-fast CFG reconstruction, by avoiding fancy reconstruct heuristics.
        # Still slower than reading symbol table, so this might take a while for big binaries.
        # Note we don't need to cache since output symbols will be cached anyway.
        cfg = project.analyses.CFGFast(
            kb=kb,
            force_smart_scan=False,
            resolve_indirect_jumps=False,
            data_references=False,
        )

        # grab functions from analysis output
        symbols = [
            FunctionSymbol(
                name=func.name or UNKNOWN_FUNC,
                addr=func.addr,
                size=func.size,
                is_import=func.is_plt,
                origin="cfg",
            )
            for func in cfg.kb.functions.values()
            if not (func.is_simprocedure or func.is_alignment or func.is_syscall)
        ]

    # sort symbols
    symbols = sorted(symbols, key=lambda s: (s.addr, s.name))

    # remove duplicaties (it happens sometimes the CLE loader duplicate symbols)
    symbols = [
        next(group) for _, group in groupby(symbols, key=lambda s: (s.addr, s.name))
    ]

    # CLE often leaves function sizes unset. Infer them from the next distinct
    # symbol address, while keeping the inferred span inside its section. The
    # distinct-address lookup preserves aliases that share one function entry.
    for idx, symbol in enumerate(symbols):
        if symbol.size != 0:
            continue

        next_addr = next(
            (
                candidate.addr
                for candidate in symbols[idx + 1 :]
                if candidate.addr > symbol.addr
            ),
            None,
        )
        if next_addr is not None:
            inferred_size = next_addr - symbol.addr
            find_section = getattr(
                project.loader.main_object, "find_section_containing", None
            )
            section = find_section(symbol.addr) if callable(find_section) else None
            if section is not None:
                section_end = section.vaddr + section.memsize
                inferred_size = min(inferred_size, section_end - symbol.addr)
            symbol.size = inferred_size

    symbols = [symbol for symbol in symbols if symbol.size > 0]

    logger.info(f"Obtained {len(symbols)} symbols")
    return symbols
