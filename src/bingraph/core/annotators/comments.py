"""Assembly annotations that preserve original operands and graph structure.

Render collected references and symbolic names as bounded comments. Comment
formatting, deduplication and classification here never establish CFG edges."""

from __future__ import annotations

import json
import re
from abc import abstractmethod
from typing import Any

from angr.knowledge_plugins.xrefs import XRefType
from cle.backends.symbol import SymbolType
from loguru import logger

from bingraph.core.labels import short_label
from bingraph.core.vis import ContentAnnotator, Node
from bingraph.helpers.capstone import RISCV_ARCHES, InsnSemantics, arch_has_delay_slot
from bingraph.helpers.symbols import is_arm_mapping_symbol, plt_symbol_name


class CommentsAnnotator(ContentAnnotator):
    """Append deduplicated comment rows while retaining assembly operands."""

    name: str = "asm"
    column: str = "comment"

    @abstractmethod
    def get_comments_by_addr(self, node: Node) -> dict[int, list[str]]:
        pass

    def get_instruction_comments(self, node: Node, insn: Any) -> list[str]:
        """Supply operand annotations without replacing the disassembly text."""

        return []

    def annotate_content(self, node: Node, content: dict[str, Any]):
        if node.obj.is_simprocedure or node.obj.is_syscall:
            return

        comments_by_addr = self.get_comments_by_addr(node)

        for k in content["data"]:
            ins = k["_ins"]
            ins_addr = ins.address if ins is not None else k.get("_addr")
            comments = list(k.get("_comments", ()))
            if ins_addr is not None:
                comments.extend(comments_by_addr.get(ins_addr, ()))
            if ins is not None:
                comments.extend(self.get_instruction_comments(node, ins))
                # Raw address notes add nothing when that address is already an
                # operand. Names and contents remain useful even in that case.
                operands = {
                    int(value, 0)
                    for value in re.findall(
                        r"(?<![\w-])(?:0x[0-9a-fA-F]+|\d+)(?!\w)",
                        getattr(ins, "op_str", ""),
                    )
                }
                comments = [
                    comment
                    for comment in comments
                    if not (
                        (
                            match := re.fullmatch(
                                r"ref (0x[0-9a-fA-F]+)", comment.strip()
                            )
                        )
                        and int(match[1], 16) in operands
                    )
                ]
            comments = list(dict.fromkeys(comments))
            if comments:
                k["comment"] = {
                    "content": "\n".join(" ; " + comment for comment in comments)
                }
                k["comment"]["color"] = "gray"
                k["comment"]["align"] = "LEFT"


class CommentsDataRef(CommentsAnnotator):
    """Display bounded data and symbol references, with optional demangling."""

    def __init__(self, *, demangle: bool = False) -> None:
        self.demangle = demangle

    @staticmethod
    def _symbol_name_at(node: Node, addr: int) -> str | None:
        """Prefer exact labels; infer offsets only inside a sized ELF symbol."""

        project = node.project

        def meaningful(name):
            arch = getattr(getattr(project, "arch", None), "name", "")
            return name and not is_arm_mapping_symbol(arch, name)

        for kb in (node.kb, project.kb):
            if addr in kb.labels and meaningful(kb.labels[addr]):
                return kb.labels[addr]

        name = plt_symbol_name(project, addr)
        if name:
            return name
        sym = project.loader.find_symbol(addr)
        if sym and meaningful(sym.name):
            return sym.name

        # A preceding symbol alone is not enough: gaps, section symbols and
        # zero-sized labels must not absorb unrelated addresses.
        sym = project.loader.find_symbol(addr, fuzzy=True)
        if (
            sym
            and meaningful(sym.name)
            and not sym.is_import
            and sym.type in (SymbolType.TYPE_FUNCTION, SymbolType.TYPE_OBJECT)
            and sym.rebased_addr < addr < sym.rebased_addr + sym.size
        ):
            return f"{sym.name}+{addr - sym.rebased_addr:#x}"
        return None

    def get_instruction_comments(self, node: Node, insn: Any) -> list[str]:
        """Name direct transfers, never scalar immediates or indirect callees.

        Use the actual instruction operand, not resolved indirect edges. The
        original graph's call metadata covers missing Capstone call groups
        (e.g. s390 brasl) without lifting or discovering any new code.
        """

        if not hasattr(insn, "groups"):
            return []
        semantic = InsnSemantics(insn)
        target = semantic.direct_target_for_arch(node.project.arch.name)
        if target is None:
            return []
        if (
            node.project.arch.name in RISCV_ARCHES
            and semantic.is_call()
            and not semantic.is_jump()
        ):
            # The shared helper normalizes jumps; RISC-V calls are relative too.
            target += insn.address
        direct = semantic.is_call() or semantic.is_jump()
        if (
            not direct
            and not arch_has_delay_slot(node.project.arch.name)
            and insn.address == node.obj.instruction_addrs[-1]
        ):
            graph = node.obj._cfg_model.graph
            direct = any(
                data.get("jumpkind") == "Ijk_Call"
                and data.get("ins_addr", insn.address) == insn.address
                and dst.addr == target
                for _, dst, data in graph.out_edges(node.obj, data=True)
            )
        if not direct:
            return []
        name = self._symbol_name_at(node, target)
        if name is None and node.project.arch.name.startswith("ARM"):
            # CLE may retain a Thumb function's mode bit in its symbol address.
            name = self._symbol_name_at(node, target | 1)
        if (
            name
            and semantic.is_jump()
            and not semantic.is_call()
            and re.search(r"[+-]0x[0-9a-fA-F]+$", name)
            and not any(
                kb.labels.get(target) == name for kb in (node.kb, node.project.kb)
            )
            and getattr(node.project.loader.find_symbol(target), "name", None) != name
        ):
            # Keep meaningful local labels and hidden/external targets. A plain
            # function+offset at an existing successor merely repeats its header.
            if any(
                not dst.is_simprocedure
                and dst.addr in {target, target | 1}
                and dst.function_address == node.obj.function_address
                and dst.name == name
                for dst in node.graph.successors(node.obj)
            ):
                return []
        return [f"{short_label(name, demangle=self.demangle)} "] if name else []

    @staticmethod
    def _truncate_comment(text: str, max_len: int = 64) -> str:
        text = text.replace("\n", "\\n").replace("\r", "\\r")

        if len(text) <= max_len:
            return text

        return text[: max_len - 3] + "..."

    def _format_memory_data_comment(self, node: Node, md, *, is_read=False) -> str:
        """Name referenced storage without claiming the value of its contents."""

        addr = getattr(md, "addr", None)
        sort = str(getattr(md, "sort", "data")).lower()
        symbol = self._symbol_name_at(node, addr) if addr is not None else None
        location = (
            f"{short_label(symbol, demangle=self.demangle)} @ {addr:#x}"
            if symbol
            else hex(addr)
            if addr is not None
            else ""
        )

        if addr is not None and ("pointer" in sort or sort == "ptr"):
            # pointer_addr is the origin of a reference to this data, not
            # its stored value. Do not turn it into a purported callee.
            return f"ptr slot {location}" if symbol else f"ref {location}"

        width = getattr(md, "size", 0)
        if is_read and sort == "integer" and addr is not None and 4 <= width <= 32:
            region = node.project.loader.find_section_containing(addr)
            if (
                region is not None
                and region.is_readable
                and not region.is_writable
                and not region.is_executable
                and not region.name.startswith((".got", ".igot"))
                and addr + width <= region.vaddr + region.memsize
            ):
                try:
                    data = node.project.loader.memory.load(addr, width)
                except KeyError:
                    data = b""
                if len(data) == width and all(32 <= byte <= 126 for byte in data):
                    # This describes exact bytes, not a guessed string object.
                    text = json.dumps(data.decode("ascii"))
                    return f"bytes {location if symbol else '@ ' + location}: {text}"

        content = getattr(md, "content", None)
        if isinstance(content, (bytes, bytearray)):
            text = content.decode("utf-8", errors="ignore").strip("\x00")
            if text:
                text = self._truncate_comment(text)
                return f'{location}: "{text}"' if symbol else f'"{text}"'

        if symbol:
            return location

        if addr is None:
            return f"data: {sort}"

        if "jumptable" in sort or "jump table" in sort:
            return f"jump table @ {hex(addr)}"

        if sort == "code reference":
            return f"code reference @ {hex(addr)}"

        return f"ref {hex(addr)}"

    def _format_address_comment(self, node: "Node", addr: int) -> str:
        symbol = self._symbol_name_at(node, addr)
        if symbol:
            return f"{short_label(symbol, demangle=self.demangle)}"

        return f"ref {hex(addr)}"

    def _format_xref_comment(self, node: Node, xref) -> str | None:

        md = getattr(xref, "memory_data", None)
        if md is not None:
            # Case 1: angr resolved a MemoryData object
            return self._format_memory_data_comment(
                node, md, is_read=getattr(xref, "type", None) == XRefType.Read
            )

        dst = getattr(xref, "dst", None)
        if isinstance(dst, int):
            # Case 2: raw destination address only
            return self._format_address_comment(node, dst)

        return None

    def get_comments_by_addr(self, node: Node) -> dict[int, list[str]]:
        comments_by_addr: dict[int, list[str]] = {}

        kb_xrefs = getattr(node.kb, "xrefs", None)
        if kb_xrefs is None:
            return comments_by_addr

        block_start = node.obj.addr
        block_end = block_start + node.obj.size

        xrefs = kb_xrefs.get_xrefs_by_ins_addr_region(block_start, block_end)
        if not xrefs:
            # Do not use "accessed_data_refernces" sice CFGFast is required
            # e.g., xrefs = list(getattr(node.obj, "accessed_data_references", []))
            for instr_addr in node.obj.instruction_addrs:
                xrefs.update(kb_xrefs.get_xrefs_by_ins_addr(instr_addr))
        if len(xrefs):
            logger.info(
                f"Found {len(xrefs)} reference(s) in block {hex(block_start)}-{hex(block_end)}"
            )

        def _xref_sort_key(xref):
            md = getattr(xref, "memory_data", None)
            return (
                getattr(xref, "ins_addr", -1),
                getattr(xref, "dst", -1),
                getattr(md, "addr", -1) if md is not None else -1,
                str(getattr(md, "sort", "")) if md is not None else "",
            )

        # Keep comment emission deterministic across runs. angr xref iteration
        # order is not stable enough for golden-file tests when an instruction
        # accumulates multiple references/comments.
        read_sites = {
            (xref.ins_addr, xref.dst) for xref in xrefs if xref.type == XRefType.Read
        }
        for xref in sorted(xrefs, key=_xref_sort_key):
            # A load's exact width takes precedence over an address hint from
            # the same instruction; do not also emit an unbounded string guess.
            if xref.type == XRefType.Offset and (xref.ins_addr, xref.dst) in read_sites:
                continue
            comment = self._format_xref_comment(node, xref)
            if not comment:
                continue

            # add some space for pretty printing on BB boundary
            comment += " "

            # Merge multiple references originating from the same instruction
            ins_addr = xref.ins_addr
            if ins_addr in comments_by_addr:
                if comment not in comments_by_addr[ins_addr]:
                    comments_by_addr[ins_addr].append(comment)
            else:
                comments_by_addr[ins_addr] = [comment]

        for comments in comments_by_addr.values():
            comments.sort()

        return comments_by_addr
