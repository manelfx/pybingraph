from dataclasses import dataclass
from typing import Any

from angr.analyses.cfg import CFGBase

from bingraph.helpers import CfgExits

from .vis import Edge, Graph, Node, Source, VisError


_EXTERNAL_FRONTIER_JUMPKINDS = frozenset({"Ijk_Boring", "Ijk_Call", "Ijk_FakeRet"})


def _is_call_like_exit(jumpkind: object) -> bool:
    """Return whether an exit is optional under the call-leaf policy."""

    return jumpkind == "Ijk_Call" or (
        isinstance(jumpkind, str) and jumpkind.startswith("Ijk_Sys_")
    )


def _is_fake_return_only_leaf(cfg_graph: Any, node: Any) -> bool:
    """Return whether a synthetic leaf is reached only after calls return."""

    predecessors = tuple(cfg_graph.predecessors(node))
    if not node.is_simprocedure or not predecessors:
        return False
    return all(
        cfg_graph.get_edge_data(predecessor, node).get("jumpkind") == "Ijk_FakeRet"
        for predecessor in predecessors
    )


def _is_path_terminator(node: Any) -> bool:
    """Return whether ``node`` is angr's artificial path-end marker."""

    return node.is_simprocedure and node.simprocedure_name == "PathTerminator"


def _is_structural_simprocedure(
    cfg_graph: Any, node: Any, selected_nodes: set[Any]
) -> bool:
    """Return whether a synthetic node connects selected function blocks."""

    return node.is_simprocedure and any(
        successor in selected_nodes for successor in cfg_graph.successors(node)
    )


def _simprocedures_by_addr(nodes: set[Any]) -> dict[int, Any]:
    """Return selected synthetic nodes by their target address."""

    return {node.addr: node for node in nodes if node.is_simprocedure}


def _canonical_render_node(node: Any, simprocedures: dict[int, Any]) -> Any:
    """Prefer a selected SIMP leaf over its foreign normal-node twin."""

    return simprocedures.get(node.addr, node)


def _selected_cfg_edges(cfg_graph: Any, selected_nodes: set[Any]):
    """Yield selected edges after canonicalizing external target twins."""

    simprocedures = _simprocedures_by_addr(selected_nodes)
    for source, destination, data in cfg_graph.edges(data=True):
        source = _canonical_render_node(source, simprocedures)
        destination = _canonical_render_node(destination, simprocedures)
        if source in selected_nodes and destination in selected_nodes:
            yield source, destination, data


def _select_cfg_nodes(
    cfg_graph: Any,
    func_addr: int | None,
    exits: CfgExits = "jump",
) -> set[Any]:
    """Choose function nodes and the one-hop exits needed to explain them.

    CFGFast may attach targets that belong to another function or a synthetic
    procedure. Keep only nodes owned by the requested function, plus selected
    direct semantic exits. Structural simprocedures that connect selected
    function blocks remain visible independently of the exit-display policy.
    """

    if func_addr is None:
        return {node for node in cfg_graph if not _is_path_terminator(node)}

    selected = {
        node
        for node in cfg_graph
        if (
            not _is_path_terminator(node)
            and not node.is_simprocedure
            and node.function_address == func_addr
        )
    }
    synthetic_targets = _simprocedures_by_addr(
        {
            node
            for node in cfg_graph
            if (
                not _is_path_terminator(node)
                and node.is_simprocedure
                and node.function_address == func_addr
            )
        }
    )
    # Unknown-entry sources have outgoing edges only, so the normal one-hop
    # exit walk cannot find them. Structural sources explain selected blocks
    # regardless of the external-call display policy.
    selected.update(
        node
        for node in synthetic_targets.values()
        if _is_structural_simprocedure(cfg_graph, node, selected)
    )

    for source in tuple(selected):
        for destination in cfg_graph.successors(source):
            if _is_path_terminator(destination):
                continue

            edge_data = cfg_graph.get_edge_data(source, destination)
            destination = _canonical_render_node(destination, synthetic_targets)
            jumpkind = edge_data.get("jumpkind")
            is_semantic_exit = (
                destination.is_simprocedure or jumpkind in _EXTERNAL_FRONTIER_JUMPKINDS
            )
            if _is_structural_simprocedure(cfg_graph, destination, selected):
                selected.add(destination)
            elif is_semantic_exit and (
                exits == "always"
                or (
                    exits == "jump"
                    and not _is_call_like_exit(jumpkind)
                    and not _is_fake_return_only_leaf(cfg_graph, destination)
                )
            ):
                selected.add(destination)

    return selected


@dataclass
class CFGSource(Source):
    func_addr: int | None = None
    exits: CfgExits = "jump"

    def parse(self, cfg: CFGBase) -> Graph:
        """Convert the selected function and its direct exits to render nodes."""

        obj = cfg.graph
        graph = Graph(cfg)
        selected_nodes = _select_cfg_nodes(obj, self.func_addr, self.exits)
        lookup = {}

        for n in selected_nodes:
            if n in lookup:
                raise VisError("Duplicate node %s" % str(n))

            wn = Node(n, obj)
            lookup[n] = wn
            graph.add_node(wn)

        for src, dst, data in _selected_cfg_edges(obj, selected_nodes):
            graph.add_edge(Edge(lookup[src], lookup[dst], data))

        return graph
