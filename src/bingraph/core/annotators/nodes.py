# Portions adapted from bingraphvis (BSD-2-Clause); see NOTICE.

"""Synthetic-node coloring for graph presentation, without modifying CFGs."""

from __future__ import annotations

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
