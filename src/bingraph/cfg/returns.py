"""Read-only saved-link return proofs over an exact intraprocedural graph.

Recognize the supported ARM private-frame restore without changing blocks.
The builder applies successful proofs; unsupported flow, calls, stack uses or
ambiguous provenance leave the original unresolved transfer intact."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import networkx as nx
from angr import Project
from angr.knowledge_plugins.cfg import CFGNode
from capstone import CS_OP_REG
from capstone.arm import (
    ARM_CC_AL,
    ARM_INS_BX,
    ARM_INS_IT,
    ARM_INS_POP,
    ARM_INS_PUSH,
    ARM_REG_LR,
    ARM_REG_PC,
    ARM_REG_SP,
)

from bingraph.cfg.graph import CFGGraph
from bingraph.cfg.models import BlockSpec, FunctionBounds
from bingraph.helpers.disassembly import decode_raw_capstone_insns


def saved_link_return_addrs(
    project: Project,
    bounds: FunctionBounds,
    blocks: Mapping[int, BlockSpec],
    static_targets: Mapping[int, tuple[int, ...]],
    analysis_graph: Callable[[], tuple[CFGGraph, Mapping[int, CFGNode]]],
) -> set[int]:
    """Recognize split ARM pops of the entry LR in an unchanged leaf frame.

    This is a provenance proof, not a guess based on a jump register's
    name. An unconditional entry push creates a private frame; a suffix
    of unconditional pops must restore the entire frame and load its LR
    slot into the final BX register. Other SP uses (including exporting a
    frame pointer), calls, IT predication, and unresolved intervening flow
    make this deliberately small proof inconclusive.

    As with private stack saves in the ABI target pass, ordinary argument
    pointers are assumed not to alias a newly allocated, unescaped frame.
    The scan is bounded and runs only for ARM functions with unknown exits.

    Return source block addresses without mutating the supplied snapshot.
    Request its exact graph lazily through ``analysis_graph`` only after the
    entry-frame checks pass. An empty result means no return was proved, not
    that the candidate transfers cannot return.
    """

    if not project.arch.name.startswith("ARM"):
        return set()
    candidates = {
        addr
        for addr, block in blocks.items()
        if block.jumpkind == "Ijk_Boring"
        and not block.direct_targets
        and addr not in static_targets
        and block.fallthrough_addr is None
    }
    if not candidates:
        return set()
    entry = blocks[bounds.addr]
    entry_insns = decode_raw_capstone_insns(project, entry.addr, entry.size)
    if not entry_insns:
        return set()
    push = entry_insns[0]
    if push.id != ARM_INS_PUSH or push.cc != ARM_CC_AL:
        return set()
    saved = [op.reg for op in push.operands if op.type == CS_OP_REG]
    if ARM_REG_LR not in saved or {ARM_REG_SP, ARM_REG_PC}.intersection(saved):
        return set()

    graph, nodes = analysis_graph()
    root = nodes[bounds.addr]
    reachable = {root, *nx.descendants(graph, root)}
    if graph.in_degree(root) or any(
        blocks[node.addr].jumpkind in {"Ijk_Call", "Ijk_Syscall"}
        or (
            blocks[node.addr].jumpkind == "Ijk_Boring"
            and not blocks[node.addr].direct_targets
            and node.addr not in static_targets
            and node.addr not in candidates
        )
        for node in reachable
    ):
        return set()
    # An undecodable internal successor could alter the frame before
    # rejoining an epilogue; a partial exact graph cannot prove a return.
    if any(
        target is not None
        and bounds.addr <= target < bounds.end_addr
        and target not in blocks
        for node in reachable
        for target in (
            *blocks[node.addr].direct_targets,
            *static_targets.get(node.addr, ()),
            blocks[node.addr].fallthrough_addr,
        )
    ):
        return set()
    insns = {}
    count = 0
    for node in reachable:
        block = blocks[node.addr]
        decoded = decode_raw_capstone_insns(project, block.addr, block.size)
        count += len(decoded)
        if (
            not decoded
            or count > 20000
            or decoded[0].address != block.addr
            or decoded[-1].address + decoded[-1].size != block.addr + block.size
        ):
            return set()
        insns[node.addr] = decoded

    epilogue_addrs = set()
    returns = set()
    for addr in candidates & insns.keys():
        decoded = insns[addr]
        branch = decoded[-1]
        if (
            branch.id != ARM_INS_BX
            or branch.cc != ARM_CC_AL
            or len(branch.operands) != 1
            or branch.operands[0].type != CS_OP_REG
        ):
            return set()
        start = len(decoded) - 1
        while start and decoded[start - 1].id == ARM_INS_POP:
            start -= 1
        consumed = 0
        restored_link = None
        for pop in decoded[start:-1]:
            registers = [op.reg for op in pop.operands if op.type == CS_OP_REG]
            if pop.cc != ARM_CC_AL or {ARM_REG_SP, ARM_REG_PC}.intersection(registers):
                return set()
            if restored_link in registers:
                restored_link = None
            slot = saved.index(ARM_REG_LR) - consumed
            if 0 <= slot < len(registers):
                restored_link = registers[slot]
            consumed += len(registers)
        if consumed != len(saved) or restored_link != branch.operands[0].reg:
            return set()
        epilogue_addrs.update(insn.address for insn in decoded[start:-1])
        returns.add(addr)

    for decoded in insns.values():
        for insn in decoded:
            if insn.address == push.address or insn.address in epilogue_addrs:
                continue
            reads, writes = insn.regs_access()
            if insn.id == ARM_INS_IT or ARM_REG_SP in (*reads, *writes):
                return set()
    return returns
