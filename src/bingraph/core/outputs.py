from dataclasses import dataclass
from typing import Any
from pydot import Dot, Node as PydotNode, Subgraph
from .vis import Edge, Graph, Node, Output


escape_map = {
    "!": "&#33;",
    "#": "&#35;",
    ":": "&#58;",
    "{": "&#123;",
    "}": "&#125;",
    "<": "&#60;",
    ">": "&#62;",
    "\t": "&nbsp;",
    "&": "&amp",
    "|": "&#124;",
}


def escape(text: str) -> str:
    return "".join(escape_map.get(c, c) for c in text)


default_node_attributes = {
    "shape": "Mrecord",
    "fontname": "monospace",
    "fontsize": "8",
}


default_edge_attributes = {
    "fontname": "monospace",
    "fontsize": "8",
}

COMMENT_COLUMN_GAP = 8


@dataclass
class DotOutput(Output):
    fname: str
    format: str = "png"
    dfs_rank: bool = True
    entry_addr: int | None = None

    def render_cell(self, key: str, data: dict[str, Any] | None) -> str:
        if (
            data is not None
            and data["content"] is not None
            and data["content"].strip() != ""
        ):
            ret = (
                "<TD "
                + ('bgcolor="' + data["bgcolor"] + '" ' if "bgcolor" in data else "")
                + ('ALIGN="' + data["align"] + '"' if "align" in data else "")
                + (' VALIGN="' + data["valign"] + '"' if "valign" in data else "")
                + ">"
            )
            if "color" in data:
                ret += '<FONT COLOR="' + data["color"] + '">'
            if "style" in data:
                ret += "<" + data["style"] + ">"

            if isinstance(data["content"], list):
                ret += '<TABLE BORDER="0">'
                for c in data["content"]:
                    ret += (
                        "<TR><TD "
                        + ('ALIGN="' + data["align"] + '"' if "align" in data else "")
                        + ">"
                    )
                    ret += escape(c)
                    ret += "</TD></TR>"
                ret += "</TABLE>"
            else:
                line_break = '<BR ALIGN="' + data.get("align", "CENTER") + '"/>'
                ret += line_break.join(
                    escape(line) for line in data["content"].split("\n")
                )
                if "\n" in data["content"]:
                    # BR alignment applies to the preceding line, including the last.
                    ret += line_break
            if "style" in data:
                ret += "</" + data["style"] + ">"
            if "color" in data:
                ret += "</FONT>"
            ret += "</TD>"
            return ret
        else:
            return "<TD></TD>"

    def render_row(self, row: dict[str, Any], colmeta: list[str]) -> str:
        # A multiline comment expands the row; keep assembly beside its first line.
        comment = row.get("comment", {}).get("content")
        multiline_comment = isinstance(comment, str) and "\n" in comment
        ret = "<TR>"
        for k in colmeta:
            if k == "comment":
                # Reserve space independently of whitespace and text/font metrics.
                ret += f'<TD WIDTH="{COMMENT_COLUMN_GAP}"></TD>'
            cell = row.get(k)
            if multiline_comment and cell is not None:
                cell = {**cell, "valign": "TOP"}
            ret += self.render_cell(k, cell)
        ret += "</TR>"
        return ret

    def render_content(self, c: dict[str, Any]) -> str:
        ret = ""
        if len(c["data"]) > 0:
            ret = '<TABLE BORDER="0" CELLPADDING="1" ALIGN="LEFT">'
            for r in c["data"]:
                ret += self.render_row(r, c["columns"])
            ret += "</TABLE>"
        return ret

    def set_node_label(self, node: Node):

        label = " | ".join([self.render_content(c) for c in node.content.values()])
        if label:
            node.pydot.set("label", "<{ %s }>" % label)

    def _pin_entry_to_source_rank(
        self, digraph: Dot, edges: list[Edge], nodes: list[Node]
    ) -> None:
        """Keep re-entered function entries at the top of rendered layouts."""

        if self.entry_addr is None:
            return

        entry_node = next(
            (node for node in nodes if node.obj.addr == self.entry_addr), None
        )
        if entry_node is None:
            return

        if not any(edge.dst == entry_node for edge in edges):
            return

        # Recursive or re-entered CFGs can give the entry block predecessors,
        # leaving Graphviz no natural source node. Ordinary entries already
        # have that placement, and forcing a rank there distorts the layout.
        rank_group = Subgraph(graph_name="entry_rank", rank="source")
        rank_group.add_node(PydotNode(entry_node.seq))
        digraph.add_subgraph(rank_group)

    def _mark_back_edges_nonconstraining(
        self, edges: list[Edge], nodes: list[Node]
    ) -> None:
        """Keep DFS back-edges visible without letting them drive DOT ranks."""

        outgoing: dict[Node, list[Edge]] = {node: [] for node in nodes}
        for edge in edges:
            outgoing.setdefault(edge.src, []).append(edge)

        visited: set[Node] = set()
        active: set[Node] = set()

        def visit(node: Node) -> None:
            """Mark edges that close this depth-first traversal as back-edges."""

            visited.add(node)
            active.add(node)
            for edge in outgoing.get(node, []):
                if edge.dst in active:
                    edge.pydot.set("constraint", "false")
                elif edge.dst not in visited:
                    visit(edge.dst)
            active.remove(node)

        entry_node = next(
            (node for node in nodes if node.obj.addr == self.entry_addr), None
        )
        if entry_node is not None:
            visit(entry_node)

        # The source may intentionally omit unreachable nodes. Lay out any
        # remaining components deterministically without changing their edges.
        for node in nodes:
            if node not in visited:
                visit(node)

    def generate(self, graph: Graph) -> str:

        digraph = Dot(graph_type="digraph", rankdir="TB")
        digraph.set_node_defaults(**default_node_attributes)
        digraph.set_edge_defaults(**default_edge_attributes)

        # add nodes, sorted by node (addr)
        nodes = sorted(graph.nodes, key=lambda n: n.obj.addr)
        # Stable edge order makes Graphviz's layout tie-breaking repeatable.
        edges = sorted(graph.edges, key=lambda e: (e.src.obj.addr, e.dst.obj.addr))
        for node in nodes:
            self.set_node_label(node)
            digraph.add_node(node.pydot)

        self._pin_entry_to_source_rank(digraph, edges, nodes)
        if self.dfs_rank:
            self._mark_back_edges_nonconstraining(edges, nodes)

        # Add edges in the same deterministic order used by the DFS pass.
        for edge in edges:
            digraph.add_edge(edge.pydot)

        # write graph to output file
        digraph.write("{}.{}".format(self.fname, self.format), format=self.format)

        return digraph.to_string()
