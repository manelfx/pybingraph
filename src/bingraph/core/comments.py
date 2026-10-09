"""Collect custom's presentation-only data references without running CFGFast."""

from typing import Any, NamedTuple, cast

from angr.knowledge_plugins.cfg.memory_data import MemoryDataSort
from angr.knowledge_plugins.xrefs import XRef, XRefType
from cle.backends.symbol import SymbolType
import pyvex

# angr installs this native module dynamically, without an importable type stub.
from angr.rustylib import SegmentList  # ty: ignore[unresolved-import]
from loguru import logger

from bingraph.cfg.models import CustomCFG


class _ReferenceValue(NamedTuple):
    sources: frozenset[int]
    scaled: bool = False


def _reference_uses(vex: Any, index_offsets: set[int]) -> dict[int, set[int]]:
    """Find address uses of additive constants in one bounded VEX pass.

    Sources are statement indices, not guessed pointer values. Width 0 means
    an indirect transfer or scaled indexing, neither of which proves text.
    Only a byte memory access through an unscaled expression supplies width 1.
    Unsupported expressions and oversized provenance lose evidence safely.
    """

    temps: dict[int, _ReferenceValue | None] = {}
    registers: dict[tuple[int, int], _ReferenceValue | None] = {}
    uses: dict[int, set[int]] = {}

    def record(value: _ReferenceValue | None, width: int) -> None:
        if value is not None:
            for source in value.sources:
                uses.setdefault(source, set()).add(0 if value.scaled else width)

    def describe(expr: Any, source: int, depth: int = 0) -> _ReferenceValue | None:
        if depth >= 32:
            return None
        if isinstance(expr, pyvex.IRExpr.Const):
            return _ReferenceValue(frozenset({source}))
        if isinstance(expr, pyvex.IRExpr.Get):
            if expr.offset not in index_offsets:
                return None
            return registers.get(
                (expr.offset, pyvex.get_type_size(expr.ty) // 8),
                _ReferenceValue(frozenset()),
            )
        if isinstance(expr, pyvex.IRExpr.RdTmp):
            # SSA temporaries only refer backwards: no retries or cyclic walk.
            return temps.get(expr.tmp)
        if isinstance(expr, pyvex.IRExpr.Load):
            record(
                describe(expr.addr, source, depth + 1),
                pyvex.get_type_size(expr.ty) // 8,
            )
            # Loaded values may be indices, but their address is not their value.
            return _ReferenceValue(frozenset())
        if isinstance(expr, pyvex.IRExpr.Unop):
            value = describe(expr.args[0], source, depth + 1)
            if value is None or "to" in expr.op:
                return value
            return _ReferenceValue(frozenset(), value.scaled)
        if isinstance(expr, pyvex.IRExpr.ITE):
            left = describe(expr.iftrue, source, depth + 1)
            right = describe(expr.iffalse, source, depth + 1)
            if left is None or right is None or len(left.sources | right.sources) > 32:
                return None
            return _ReferenceValue(
                left.sources | right.sources, left.scaled or right.scaled
            )
        if isinstance(expr, pyvex.IRExpr.Binop):
            args = [describe(arg, source, depth + 1) for arg in expr.args]
            if any(arg is None for arg in args):
                return None
            values = cast(list[_ReferenceValue], args)
            # A shift count or arithmetic mask is not a static address base.
            if expr.op.startswith(("Iop_Add", "Iop_Sub")):
                sources = frozenset().union(*(arg.sources for arg in values))
            elif expr.op.startswith(("Iop_Or", "Iop_And")):
                sources = frozenset().union(
                    *(
                        value.sources
                        for arg, value in zip(expr.args, values)
                        if not isinstance(arg, pyvex.IRExpr.Const)
                    )
                )
            else:
                sources = frozenset()
            if len(sources) > 32:
                return None
            return _ReferenceValue(
                sources,
                expr.op.startswith(("Iop_Shl", "Iop_Mul"))
                or any(arg.scaled for arg in values),
            )
        return None

    for index, stmt in enumerate(vex.statements):
        if isinstance(stmt, pyvex.IRStmt.WrTmp):
            temps[stmt.tmp] = describe(stmt.data, index)
        elif isinstance(stmt, pyvex.IRStmt.Put):
            value = describe(stmt.data, index)
            if hasattr(vex, "tyenv"):
                width = stmt.data.result_size(vex.tyenv) // 8
                # Aliased writes invalidate overlapping views, rather than
                # letting a byte write leave a stale full-register pointer.
                for offset, size in list(registers):
                    if offset < stmt.offset + width and stmt.offset < offset + size:
                        del registers[offset, size]
                if stmt.offset in index_offsets:
                    registers[stmt.offset, width] = value
            if value is not None and value.scaled and stmt.offset in index_offsets:
                # Retain strided table bases as non-text hints. An ordinary
                # assignment or scalar increment is not an address use.
                record(value, 0)
        elif isinstance(stmt, (pyvex.IRStmt.Store, pyvex.IRStmt.StoreG)):
            record(describe(stmt.addr, index), stmt.data.result_size(vex.tyenv) // 8)
        elif isinstance(stmt, pyvex.IRStmt.LoadG):
            record(
                describe(stmt.addr, index), pyvex.get_type_size(stmt.cvt_types[0]) // 8
            )
            temps[stmt.dst] = _ReferenceValue(frozenset())
    if getattr(vex, "next", None) is not None:
        record(describe(vex.next, len(vex.statements)), 0)
    return uses


def _is_scalar_reference(
    ref: Any,
    vex: Any,
    register_offsets: set[int],
    register_tmps: set[int],
    address_uses: dict[int, set[int]],
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
    expr = getattr(stmt, "data", None)
    if isinstance(expr, pyvex.IRExpr.Const):
        return expr.con.value != ref.data_addr
    if isinstance(expr, pyvex.IRExpr.ITE):
        # Predicated ARM writes have ITEs too. Only the assigned value is a
        # candidate address, not the condition or condition-code bookkeeping.
        return not (
            isinstance(stmt, pyvex.IRStmt.WrTmp)
            and stmt.tmp in register_tmps
            and isinstance(expr.iftrue, pyvex.IRExpr.Const)
            and expr.iftrue.con.value == ref.data_addr
        )
    # Keep additive bases with address uses, not arbitrary arithmetic operands.
    return (
        isinstance(expr, pyvex.IRExpr.Binop)
        and any(
            isinstance(arg, pyvex.IRExpr.Const) and arg.con.value == ref.data_addr
            for arg in expr.args
        )
        and ref.stmt_idx not in address_uses
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
    nontext_table_hints: set[int] = set()
    hint_sorts: dict[tuple[int, int], Any] = {}
    typed_reads: list[tuple[XRef, int]] = []

    def hint_sort(address: int, end: int) -> Any:
        key = address, end
        if key not in hint_sorts:
            try:
                hint_sorts[key], _ = cfg.model._guess_data_type(address, end - address)
            except Exception as exc:
                logger.warning(f"Cannot classify address hint at {address:#x}: {exc}")
                hint_sorts[key] = None
        return hint_sorts[key]

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
            offsets = register_offsets.copy()
            if (
                getattr(vex, "jumpkind", None) == "Ijk_Call"
                and project.arch.lr_offset is not None
            ):
                # The link register receives a continuation, not a data pointer.
                offsets.discard(project.arch.lr_offset)
            register_tmps = {
                stmt.data.tmp
                for stmt in vex.statements
                if isinstance(stmt, pyvex.IRStmt.Put)
                and stmt.offset in offsets
                and isinstance(stmt.data, pyvex.IRExpr.RdTmp)
            }
            address_uses = _reference_uses(
                vex,
                offsets
                - {
                    project.arch.sp_offset,
                    project.arch.bp_offset,
                    project.arch.ip_offset,
                },
            )
        except Exception as exc:
            # Annotation failure must not prevent an otherwise valid CFG render.
            logger.warning(f"Cannot collect comments for {node.addr:#x}: {exc}")
            continue
        instruction_addrs = set(node.instruction_addrs)
        for ref in refs:
            if _is_scalar_reference(ref, vex, offsets, register_tmps, address_uses):
                continue
            expr = (
                getattr(vex.statements[ref.stmt_idx], "data", None)
                if 0 <= ref.stmt_idx < len(vex.statements)
                else None
            )
            partial_address = (
                not ref.data_size
                and isinstance(expr, pyvex.IRExpr.Binop)
                and any(
                    isinstance(arg, pyvex.IRExpr.Const)
                    and arg.con.value == ref.data_addr
                    for arg in expr.args
                )
            )
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
            if partial_address and 1 not in address_uses.get(ref.stmt_idx, ()):
                symbol = project.loader.find_symbol(
                    ref.data_addr, fuzzy=0 in address_uses[ref.stmt_idx]
                )
                bounded_object = (
                    symbol is not None
                    and symbol.type == SymbolType.TYPE_OBJECT
                    and symbol.size > 0
                    and symbol.rebased_addr
                    <= ref.data_addr
                    < symbol.rebased_addr + symbol.size
                )
                if not bounded_object and not (
                    0 in address_uses[ref.stmt_idx]
                    and hint_sort(ref.data_addr, region.vaddr + region.memsize)
                    == MemoryDataSort.PointerArray
                ):
                    # A numeric displacement or strided scalar calculation is
                    # not a table without object bounds or pointer-array data.
                    continue
            # Executable sections may contain genuine literal pools. Only
            # known decoded instructions are classified as code references.
            sort = (
                MemoryDataSort.CodeReference
                if code.occupied_by_sort(ref.data_addr) == "code"
                else ref.data_type_str.removesuffix("(store)")
            )
            size = 0 if sort == MemoryDataSort.CodeReference else ref.data_size
            if not ref.data_size and 0 <= ref.stmt_idx < len(vex.statements):
                # A mapped immediate alone is not a pointer. Accept deliberate
                # address construction, sized symbols, or non-code string data;
                # never guess strings from executable bytes of scalar operands.
                symbol = project.loader.find_symbol(ref.data_addr)
                named_object = (
                    symbol is not None
                    and symbol.size > 0
                    and symbol.type
                    in (SymbolType.TYPE_OBJECT, SymbolType.TYPE_FUNCTION)
                )
                if ref.stmt_idx not in address_uses and not named_object:
                    enclosing = project.loader.find_symbol(ref.data_addr, fuzzy=True)
                    if code.occupied_by_sort(ref.data_addr) == "code" or (
                        enclosing is not None
                        and enclosing.type == SymbolType.TYPE_FUNCTION
                        and enclosing.rebased_addr
                        <= ref.data_addr
                        < enclosing.rebased_addr + enclosing.size
                    ):
                        continue
                    guessed = hint_sort(ref.data_addr, region.vaddr + region.memsize)
                    if guessed not in (
                        MemoryDataSort.String,
                        MemoryDataSort.UnicodeString,
                    ):
                        continue
            if sort == MemoryDataSort.Unknown and not size:
                if partial_address and 1 not in address_uses.get(ref.stmt_idx, ()):
                    nontext_table_hints.add(ref.data_addr)
                else:
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
            end = region.vaddr + region.memsize
            if 1 in address_uses.get(ref.stmt_idx, ()):
                symbol = project.loader.find_symbol(ref.data_addr)
                if (
                    symbol is not None
                    and symbol.type == SymbolType.TYPE_OBJECT
                    and symbol.size > 0
                ):
                    # Character arrays need not be NUL-terminated; do not
                    # append a neighboring object to their displayed content.
                    end = min(end, symbol.rebased_addr + symbol.size)
            reference_ends[ref.data_addr] = min(
                reference_ends.get(ref.data_addr, end), end
            )
            is_read = ref.data_size > 0 and not ref.data_type_str.endswith("(store)")
            xref = XRef(
                ins_addr=ref.ins_addr,
                block_addr=node.addr,
                stmt_idx=ref.stmt_idx,
                memory_data=memory_data,
                xref_type=XRefType.Read if is_read else XRefType.Offset,
            )
            cfg.kb.xrefs.add_xref(xref)
            if is_read and ref.data_type_str == MemoryDataSort.Integer:
                typed_reads.append((xref, ref.data_size))

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
                and (
                    memory_data.sort == MemoryDataSort.Integer
                    or addr in nontext_table_hints
                )
                and addr not in address_hints
            ):
                continue
            if sort is not None:
                memory_data.sort, memory_data.size = sort, size
            if content:
                memory_data.content = content[0]
        except Exception as exc:
            logger.warning(f"Cannot classify custom comment data at {addr:#x}: {exc}")
    for xref, width in typed_reads:
        md = xref.memory_data
        if md is not None and (
            md.sort == MemoryDataSort.Integer
            or (
                md.sort == MemoryDataSort.String
                and 4 <= width <= 32
                and md.content is not None
                and len(md.content) >= width
                and all(32 <= byte <= 126 for byte in md.content[:width])
            )
        ):
            # The shared object can have several access widths. A preview must
            # use this load's width, even if a separate address hint identified
            # a larger string, without changing the shared object's classification.
            xref.memory_data = md.copy()
            xref.memory_data.sort = MemoryDataSort.Integer
            xref.memory_data.size = width
            xref.memory_data.content = None
    cfg._comments_collected = True
