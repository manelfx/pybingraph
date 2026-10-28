"""Adapt dependency records to the existing Vis/NodeAsm/DotOutput pipeline."""

from collections import defaultdict
from functools import lru_cache
from pathlib import Path
import re
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any

from angr import KnowledgeBase, Project
from angr.knowledge_plugins.cfg import CFGModel, CFGNode
import networkx as nx

from bingraph.core.contents import NodeAsm
from bingraph.core.outputs import DotOutput, validate_text_cfg_format
from bingraph.core.vis import Edge, EdgeAnnotator, Graph, Node, Source, Vis
from bingraph.helpers.settings import DependencySelection, normalize_dependencies
from .builder import get_ddg
from .models import MemoryPolicy


class InstructionNode(CFGNode):
    """One decoded instruction with the CFG-node interface used by NodeAsm.

    Rendering must not relift a branch without its delay slot. Its assembly
    bytes are already validated by analysis, so expose that exact decode.
    """

    __slots__ = ("insn",)

    def __init__(self, model: CFGModel, insn: Any):
        super().__init__(
            insn.address,
            insn.size,
            model,
            block_id=insn.address,
            instruction_addrs=(insn.address,),
        )
        self.insn = insn

    @property
    def block(self) -> Any:
        return SimpleNamespace(
            _project=self._cfg_model.project,
            capstone=SimpleNamespace(insns=(self.insn,)),
        )


class DDGSource(Source):
    """Materialize fresh instruction wrappers and parallel dependency edges."""

    def parse(self, cfg: Any) -> Graph:
        graph = Graph(cfg)
        lookup = {}
        for node in cfg.graph.nodes():
            lookup[node] = Node(node, cfg.graph)
            graph.add_node(lookup[node])
        for source, target, data in cfg.graph.edges(data=True):
            graph.add_edge(Edge(lookup[source], lookup[target], data))
        return graph


class DependencyEdges(EdgeAnnotator):
    """Keep resource labels compact; retain analysis details in hover tooltips.

    Kind is already encoded by line style. Memory widths, symbolic entry roots
    and may-alias qualifiers remain available without crowding the graph, and
    no presentation abbreviation changes the cached dependency records.
    """

    @staticmethod
    def _resource_label(resource: str) -> str:
        """Show memory like assembly operands, not internal symbolic names."""
        resource = resource.removesuffix(" (may alias)")
        resource = resource.removesuffix(" (maximum footprint)")
        if resource in {"cc_op", "cc_dep1", "cc_dep2", "cc_ndep"}:
            return "flags"
        if resource == "mem?":
            return "[?]"
        if match := re.fullmatch(r"mem\[(.*)\]:\d+", resource):
            return "[" + re.sub(r"\bentry_", "", match[1]) + "]"
        return resource

    def annotate_edge(self, edge: Edge) -> None:
        kind = edge.meta["kind"]
        edge.pydot.set(
            "style", {"RAW": "solid", "WAR": "dashed", "WAW": "dotted"}[kind]
        )
        resources = edge.meta["resources"]
        edge.pydot.set(
            "label", ", ".join(dict.fromkeys(map(self._resource_label, resources)))
        )
        edge.pydot.set("tooltip", edge.meta["tooltip"])


@lru_cache(maxsize=128)
def render_ddg(
    project: Project,
    address: int,
    size: int,
    memory: MemoryPolicy = "must",
    format: str = "svg",
    dependencies: DependencySelection = "raw,war,waw",
) -> str:
    """Analyze/cache only the requested block and render through shared output.

    Cached analysis has no mutable presentation state. Stable instruction and
    dependency ordering preserves deterministic golden DOT. A visible RAW edge
    subsumes parallel WAR/WAW ordering; their details remain in its tooltip.
    Filter edge kinds only here: the analysis cache retains every dependency,
    and independent instructions remain visible under every selection.
    """
    validate_text_cfg_format(format)
    selected = set(normalize_dependencies(dependencies).upper().split(","))
    ddg = get_ddg(project, address, size, memory)
    kb = KnowledgeBase(project)
    model = CFGModel("DDG", cfg_manager=kb.cfgs)
    nodes = [InstructionNode(model, insn) for insn in ddg.instructions]
    raw = nx.MultiDiGraph()
    raw.add_nodes_from(nodes)
    pairs = defaultdict(list)
    for dependency in ddg.dependencies:
        if dependency.kind in selected:
            pairs[dependency.source, dependency.target].append(dependency)
    for (source, target), parallel in pairs.items():
        # Suppress only parallel constraints, never other paths. Compute this
        # after filtering: WAR/WAW must remain visible when RAW is not selected.
        has_raw = any(dependency.kind == "RAW" for dependency in parallel)
        for dependency in parallel:
            if has_raw and dependency.kind != "RAW":
                continue
            details = parallel if has_raw else [dependency]
            raw.add_edge(
                nodes[source],
                nodes[target],
                kind=dependency.kind,
                resources=dependency.display_resources or dependency.resources,
                tooltip="; ".join(
                    d.kind + ": " + ", ".join(d.resources) for d in details
                ),
            )
    cfg = SimpleNamespace(graph=raw)
    with TemporaryDirectory() as directory:
        name = str(Path(directory) / "ddg")
        vis = Vis(
            DDGSource(),
            DotOutput(name, format=format, dfs_rank=False),
            contents=[NodeAsm()],
            annotators=[DependencyEdges()],
        )
        vis.process(cfg)
        return Path(name + "." + format).read_text()
