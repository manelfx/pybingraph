"""Construction and naming of recovered CFG blocks and synthetic leaves.

Labels decorate existing leaders; symbols must not create new control flow.
Thumb-tagged addresses retain execution mode while local-label lookup uses
physical byte addresses. These factories do not run any discovery analysis."""

from __future__ import annotations

from typing import Iterable

from angr import Project
from angr.knowledge_plugins.cfg import CFGModel, CFGNode
from cle.backends.symbol import SymbolType

from bingraph.cfg.models import BlockSpec, CustomCFGNode, FunctionBounds
from bingraph.helpers.symbols import is_arm_mapping_symbol, plt_symbol_name


def _thumb_mode(project: Project, addr: int) -> bool:
    """Return the execution mode encoded by an ARM/Thumb address."""

    try:
        return bool(project.arch.is_thumb(addr))
    except AttributeError:
        return False


def _node_name(bounds: FunctionBounds, addr: int) -> str:
    """Return the stable display name used for a recovered block."""

    if addr == bounds.addr:
        return bounds.name
    return f"{bounds.name}+0x{addr - bounds.addr:x}"


def _local_node_names(project: Project, addrs: Iterable[int]) -> dict[int, str]:
    """Choose local code labels at existing block starts, never new leaders.

    Untyped assembler labels precede local function aliases, then names sort
    lexically so symbol-table order cannot change the title. Object, section,
    and file symbols are not code labels; ARM mapping symbols only describe
    decoding regions. Normalize Thumb execution addresses to their byte address.
    """

    starts = {addr: addr - int(_thumb_mode(project, addr)) for addr in addrs}
    byte_addrs = set(starts.values())
    aliases: dict[int, tuple[bool, str]] = {}
    for symbol in project.loader.main_object.symbols:
        if (
            not symbol.is_local
            or symbol.is_import
            or symbol.type not in (SymbolType.TYPE_NONE, SymbolType.TYPE_FUNCTION)
            or not symbol.name
        ):
            continue
        addr = symbol.rebased_addr
        addr -= int(_thumb_mode(project, addr))
        if addr not in byte_addrs or is_arm_mapping_symbol(
            project.arch.name, symbol.name
        ):
            continue
        rank = (symbol.is_function, symbol.name)
        if addr not in aliases or rank < aliases[addr]:
            aliases[addr] = rank
    return {
        addr: aliases[byte_addr][1]
        for addr, byte_addr in starts.items()
        if byte_addr in aliases
    }


def _make_block_node(
    model: CFGModel,
    project: Project,
    func_addr: int,
    bounds: FunctionBounds,
    block: BlockSpec,
    *,
    name: str | None = None,
) -> CFGNode:
    """Materialize one recovered normal CFG node."""

    return CustomCFGNode(
        block.addr,
        block.size,
        cfg=model,
        function_address=func_addr,
        block_id=block.addr,
        instruction_addrs=block.instruction_addrs,
        thumb=_thumb_mode(project, block.addr),
        name=name or _node_name(bounds, block.addr),
        vex_linear_instruction_sizes=dict(block.vex_linear_instruction_sizes),
    )


def _external_target_name(project: Project, addr: int) -> str:
    """Return the loader symbol name for an external target when known."""

    symbol = project.loader.find_symbol(addr)
    name = getattr(symbol, "name", None)
    if isinstance(name, str) and name:
        return name
    return plt_symbol_name(project, addr) or f"ExternalTarget_{addr:#x}"


def _make_leaf_node(
    model: CFGModel,
    func_addr: int,
    addr: int,
    name: str,
    *,
    is_syscall: bool = False,
) -> CFGNode:
    """Materialize a synthetic CFG leaf without making it a function block."""

    return CFGNode(
        addr,
        0,
        cfg=model,
        function_address=func_addr,
        block_id=addr,
        instruction_addrs=(),
        simprocedure_name=name,
        is_syscall=is_syscall,
        name=name,
    )
