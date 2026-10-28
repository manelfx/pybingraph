# Portions adapted from bingraphvis (BSD-2-Clause); see NOTICE.

"""Node styling and navigation metadata, without modifying analysis graphs."""

from __future__ import annotations

from urllib.parse import urlencode
from bingraph.core.vis import Node, NodeAnnotator


class ColorSimprocedures(NodeAnnotator):
    """Style existing synthetic leaves without changing their analysis names."""

    def annotate_node(self, node: Node) -> None:
        if not node.obj.is_simprocedure:
            return

        node.pydot.set_style("filled")
        if node.obj.simprocedure_name in [
            "PathTerminator",
            "ReturnUnconstrained",
            "UnresolvableTarget",
        ]:
            node.pydot.set_fillcolor("#ffcccc")
        else:
            node.pydot.set_fillcolor("#dddddd")


class LinkDDG(NodeAnnotator):
    """Link real BBs to DDGs and show plain-text address/size hover statistics.

    Reuse rendered assembly rows so instruction counts include delay slots and
    VEX fallback rows without relifting. Explicit tooltips prevent Graphviz
    from displaying the node's HTML table markup as link hover text.
    """

    def __init__(self, filepath: str):
        self.filepath = filepath

    def annotate_node(self, node: Node) -> None:
        if node.obj.is_simprocedure or not node.obj.size:
            return
        query = urlencode(
            {
                "filepath": self.filepath,
                "address": hex(node.obj.addr),
                "size": node.obj.size,
            }
        )
        node.pydot.set("URL", "/ddg?" + query)
        instructions = node.content.get("asm", {}).get("data")
        if instructions is None:
            instructions = node.obj.instruction_addrs
        count = len(instructions) if instructions is not None else "unknown"
        node.pydot.set(
            "tooltip",
            f"Block {node.obj.addr:#x} ({count} instruction{'s' if count != 1 else ''}, "
            f"{node.obj.size} byte{'s' if node.obj.size != 1 else ''})",
        )
