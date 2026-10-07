import re
from typing import Any

from angr import Project
from angr.knowledge_plugins.cfg import CFGNode
from bingraph.cfg.decode import decode_one

from .vis import Content, Node
from archinfo.archerror import ArchError
from loguru import logger


MAX_NODE_LABEL_LENGTH = 30


def _short_node_name(name: str | None) -> str | None:
    """Limit the displayed label while preserving its trailing block offset."""

    if name is None:
        return None
    offset_match = re.search(r"[+-]0x[0-9a-fA-F]+$", name)
    label = name[: offset_match.start()] if offset_match else name
    offset = offset_match.group() if offset_match else ""
    if len(label) > MAX_NODE_LABEL_LENGTH:
        label = label[:MAX_NODE_LABEL_LENGTH] + "..."
    return label + offset


class NodeHead(Content):
    name: str = "head"
    columns: list[str] = ["addr", "name", "attributes"]

    def gen_render(self, node: Node) -> None:
        cfg_node = node.obj
        display_name = cfg_node.name
        if cfg_node.is_simprocedure and re.fullmatch(
            r"ExternalTarget_0x[0-9a-fA-F]+", display_name or ""
        ):
            display_name = "ExternalTarget"
        show_addr = not (
            cfg_node.is_simprocedure
            and (cfg_node.simprocedure_name or "").startswith("Unresolvable")
        )
        attributes = []
        if cfg_node.is_simprocedure:
            attributes.append(" SIMP")
        if cfg_node.is_syscall:
            attributes.append(" SYSC")
        if cfg_node.no_ret:
            attributes.append(" NORET")

        node.content[self.name] = {
            "data": [
                {
                    "addr": {
                        "content": "({:#08x})".format(cfg_node.addr)
                        if show_addr
                        else None,
                    },
                    "name": {"content": _short_node_name(display_name), "style": "B"},
                    "attributes": {"content": " ".join(attributes)},
                }
            ],
            "columns": self.columns if show_addr else self.columns[1:],
        }


class NodeAsm(Content):
    name: str = "asm"
    columns: list[str] = ["addr", "mnemonic", "operands"]

    @staticmethod
    def _vex_word(project: Any, addr: int, size: int) -> str:
        """Return a raw instruction word for a VEX-only decoded instruction."""

        try:
            raw = project.loader.memory.load(addr, size)
        except Exception:
            return "<unavailable>"
        return f"0x{raw.hex()}"

    def _custom_insns(self, node: Node) -> list[dict[str, Any]] | None:
        """Render builder VEX spans without losing later Capstone instructions."""

        cfg_node = node.obj
        vex_sizes = getattr(cfg_node, "vex_linear_instruction_sizes", None)
        if not vex_sizes:
            return None
        project = getattr(getattr(cfg_node, "block", None), "_project", None)
        if not isinstance(project, Project):
            return None

        data: list[dict[str, Any]] = []
        for addr in cfg_node.instruction_addrs:
            vex_size = vex_sizes.get(addr)
            if vex_size is not None:
                data.append(
                    {
                        "addr": {"content": "0x%08x:\t" % addr, "align": "LEFT"},
                        "mnemonic": {"content": ".word", "align": "LEFT"},
                        "operands": {
                            "content": self._vex_word(project, addr, vex_size),
                            "align": "LEFT",
                        },
                        "_ins": None,
                        "_addr": addr,
                        "_comments": ["VEX linear decode"],
                    }
                )
                continue

            insn = decode_one(
                project, addr, getattr(project.arch, "max_inst_bytes", 16)
            )
            if insn is None:
                continue
            data.append(
                {
                    "addr": {"content": "0x%08x:\t" % insn.address, "align": "LEFT"},
                    "mnemonic": {"content": insn.mnemonic, "align": "LEFT"},
                    "operands": {"content": insn.op_str, "align": "LEFT"},
                    "_ins": insn,
                    "_addr": insn.address,
                }
            )
        return data

    def gen_render(self, node: Node) -> None:
        cfg_node: Any = node.obj

        if isinstance(cfg_node, CFGNode):
            is_syscall = cfg_node.is_syscall
            is_simprocedure = cfg_node.is_simprocedure
        elif type(cfg_node).__name__ == "CodeLocation":
            is_syscall = False
            is_simprocedure = cfg_node.sim_procedure is not None
        elif type(cfg_node).__name__ == "ProgramVariable":
            is_syscall = False
            is_simprocedure = cfg_node.location.sim_procedure is not None
        elif type(cfg_node).__name__ == "BlockNode":
            is_syscall = False
            is_simprocedure = False
        elif type(cfg_node).__name__ == "HookNode":
            return
        elif type(cfg_node).__name__ == "Function":
            return
        elif type(cfg_node).__name__ == "Block":
            is_syscall = False
            is_simprocedure = False
        else:
            return

        if is_simprocedure or is_syscall:
            return None

        custom_data = self._custom_insns(node)
        if custom_data is not None:
            node.content[self.name] = {
                "data": custom_data,
                "columns": self.columns,
            }
            return

        try:
            # FIXME -- pp writes "call <fn>" instead of "call <addr>"
            # print(node.obj.block.pp())
            insns = cfg_node.block.capstone.insns
        except (ArchError, KeyError) as e:
            logger.error(str(e))
            insns = []
        except Exception as e:
            logger.exception(e)
            insns = []

        data = []
        for ins in insns:
            data.append(
                {
                    "addr": {"content": "0x%08x:\t" % ins.address, "align": "LEFT"},
                    "mnemonic": {"content": ins.mnemonic, "align": "LEFT"},
                    "operands": {"content": ins.op_str, "align": "LEFT"},
                    "_ins": ins,
                    "_addr": ins.address,
                }
            )

        node.content[self.name] = {
            "data": data,
            "columns": self.columns,
        }
