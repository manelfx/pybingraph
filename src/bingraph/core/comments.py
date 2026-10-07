"""Collect custom's presentation-only data references without running CFGFast."""

from typing import Any, cast

from angr.knowledge_plugins.cfg.memory_data import MemoryDataSort
from angr.knowledge_plugins.xrefs import XRef, XRefType
import pyvex

# angr installs this native module dynamically, without an importable type stub.
from angr.rustylib import SegmentList  # ty: ignore[unresolved-import]
from loguru import logger

from bingraph.cfg.models import CustomCFG


def _is_scalar_reference(
    ref: Any, vex: Any, register_offsets: set[int], address_tmps: set[int]
) -> bool:
    """Exclude bookkeeping constants and partial operands, not memory accesses.

    VEX also reports constants used to update condition codes or as operands
    of arithmetic. Those are not complete pointer values, even when their
    numbers happen to lie inside a mapped section of a zero-based image.
    """

    if ref.data_size or not 0 <= ref.stmt_idx < len(vex.statements):
        return False
    stmt = vex.statements[ref.stmt_idx]
    if isinstance(stmt, pyvex.IRStmt.Put):
        return stmt.offset not in register_offsets
    # An indexed memory access still has a useful static base-address hint.
    if isinstance(stmt, pyvex.IRStmt.WrTmp) and stmt.tmp in address_tmps:
        return False
    expr = getattr(stmt, "data", None)
    return isinstance(expr, pyvex.IRExpr.Binop) and any(
        isinstance(arg, pyvex.IRExpr.Const) and arg.con.value == ref.data_addr
        for arg in expr.args
    )


def collect_custom_comments(cfg: CustomCFG) -> None:
    """Enrich the finalized blocks once, only when a render requests comments.

    VEX supplies instruction-addressed references; angr's existing data
    classifier supplies their string/pointer descriptions. Neither operation
    discovers blocks or contributes facts to the builder's exact proofs.
    """

    if cfg._comments_collected:
        return
    project = cfg.model.project
    assert project is not None
    nodes = sorted(
        (
            node
            for node in cfg.graph.nodes
            if not node.is_simprocedure and not node.is_syscall and node.size
        ),
        key=lambda node: node.addr,
    )
    code = SegmentList()
    register_offsets = {
        reg.vex_offset for reg in project.arch.register_list if reg.general_purpose
    }
    reference_ends: dict[int, int] = {}
    address_hints: set[int] = set()
    for node in nodes:
        # Thumb nodes carry their mode bit in the address, not in memory.
        addr = node.addr & ~1 if project.arch.name.startswith("ARM") else node.addr
        code.occupy(addr, node.size, "code")

    for node in nodes:
        try:
            block = project.factory.block(
                node.addr,
                size=node.size,
                opt_level=1,
                cross_insn_opt=False,
                strict_block_end=True,
                collect_data_refs=True,
                load_from_ro_regions=True,
            )
            vex = block.vex
            refs = vex.data_refs or ()
            if not refs:
                continue
            address_tmps = {
                expr.addr.tmp
                for stmt in vex.statements
                for expr in (stmt, *stmt.expressions)
                if isinstance(getattr(expr, "addr", None), pyvex.IRExpr.RdTmp)
            }
        except Exception as exc:
            # Annotation failure must not prevent an otherwise valid CFG render.
            logger.warning(f"Cannot collect comments for {node.addr:#x}: {exc}")
            continue
        instruction_addrs = set(node.instruction_addrs)
        for ref in refs:
            if _is_scalar_reference(ref, vex, register_offsets, address_tmps):
                continue
            region = project.loader.find_section_containing(ref.data_addr)
            if region is None:
                # Do not interpret ELF headers as data just because an integer
                # constant happens to fall inside a zero-based load segment.
                if project.loader.main_object.sections:
                    continue
                region = project.loader.find_segment_containing(ref.data_addr)
            if (
                region is None
                or not region.is_readable
                or ref.ins_addr not in instruction_addrs
            ):
                continue
            # Executable sections may contain genuine literal pools. Only
            # known decoded instructions are classified as code references.
            sort = (
                MemoryDataSort.CodeReference
                if code.occupied_by_sort(ref.data_addr) == "code"
                else ref.data_type_str.removesuffix("(store)")
            )
            size = 0 if sort == MemoryDataSort.CodeReference else ref.data_size
            if sort == MemoryDataSort.Unknown and not size:
                address_hints.add(ref.data_addr)
            cfg.model.add_memory_data(
                ref.data_addr,
                cast(MemoryDataSort, sort),
                data_size=size,
            )
            memory_data = cfg.model.memory_data[ref.data_addr]
            # A bare address hint may precede a typed access to the same data.
            # Keep the actual access width/type rather than whichever came first.
            if memory_data.sort == MemoryDataSort.Unknown and size:
                memory_data.sort, memory_data.size = sort, size
            reference_ends[ref.data_addr] = region.vaddr + region.memsize
            cfg.kb.xrefs.add_xref(
                XRef(
                    ins_addr=ref.ins_addr,
                    block_addr=node.addr,
                    stmt_idx=ref.stmt_idx,
                    memory_data=memory_data,
                    xref_type=XRefType.Offset,
                )
            )

    for addr, region_end in sorted(reference_ends.items()):
        try:
            memory_data = cfg.model.memory_data[addr]
            memory_data.max_size = region_end - addr
            if memory_data.sort not in (MemoryDataSort.Unknown, MemoryDataSort.Integer):
                continue
            # An access width is not the object's size: a byte load can read
            # a character of a longer string. Keep the native type/width as
            # fallback, but let angr inspect the section-bounded object view.
            # Reuse the accepted boundary rather than tidy_data_references:
            # its second CLE lookup can lose relocatable-image section hits,
            # and adjacent references are not reliable string boundaries.
            content: list[bytes] = []
            sort, size = cfg.model._guess_data_type(
                addr, memory_data.max_size, content_holder=content
            )
            # Printable scalar/vector bytes alone do not imply text (e.g.
            # integer 50 starts with ASCII '2'). Require a separate address
            # hint before overriding a typed access with a string description.
            if (
                sort in (MemoryDataSort.String, MemoryDataSort.UnicodeString)
                and memory_data.sort == MemoryDataSort.Integer
                and addr not in address_hints
            ):
                continue
            if sort is not None:
                memory_data.sort, memory_data.size = sort, size
            if content:
                memory_data.content = content[0]
        except Exception as exc:
            logger.warning(f"Cannot classify custom comment data at {addr:#x}: {exc}")
    cfg._comments_collected = True
