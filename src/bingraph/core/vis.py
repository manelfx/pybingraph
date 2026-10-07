from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, cast

from angr.analyses.cfg import CFGBase
from angr.knowledge_plugins.cfg import CFGNode
from pydot import Node as PydotNode, Edge as PydotEdge

from bingraph.helpers import time_it


class VisError(Exception):
    pass


class Node:
    def __init__(self, obj: CFGNode, graph: Any | None = None) -> None:
        """Wrap one CFG node with the graph selected for this render pass."""

        self.obj = obj
        # Render against the selected live graph, which need not be the graph
        # registered on a node's angr model.
        self._graph = graph
        # Pydot creates dynamic ``set_<attribute>`` methods at runtime that
        # are absent from its static stub surface.
        self.pydot: Any = PydotNode(self.seq)
        self.content = {}

    @property
    def graph(self):
        """Returns NetworkX graph."""
        return self._graph or self.obj._cfg_model.graph

    @property
    def project(self):
        """Returns static project info, no change between analyses."""
        return self.obj._cfg_model.project

    @property
    def kb(self):
        """Returns Angr Knowledge base info, analysis dependent."""
        cfg_manager = cast(Any, self.obj._cfg_model._cfg_manager)
        return cfg_manager._kb

    @property
    def seq(self):
        """Return a readable, stable DOT identifier for this CFG node."""

        block_id = self.obj.block_id
        if isinstance(block_id, int):
            return hex(block_id)
        return str(block_id)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Node):
            return False
        # CFGNode's equality does not recognize custom-builder subclasses,
        # even when both wrappers hold the exact same underlying object.
        return self is other or self.obj is other.obj or self.obj == other.obj

    def __hash__(self) -> int:
        return hash(self.obj)


class Edge:
    def __init__(
        self, src: Node, dst: Node, meta: dict[str, Any] | None = None
    ) -> None:
        self.src = src
        self.dst = dst
        self.pydot = PydotEdge(src.seq, dst.seq)
        self.meta = meta or {}

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Edge):
            return False
        return self.src == other.src and self.dst == other.dst

    def __hash__(self) -> int:
        return hash((self.src, self.dst))


class Annotator(ABC):
    """Base class for all annotators."""

    pass


class NodeAnnotator(Annotator):
    @abstractmethod
    def annotate_node(self, node: Node) -> None:
        pass


class EdgeAnnotator(Annotator):
    @abstractmethod
    def annotate_edge(self, edge: Edge) -> None:
        pass


class ContentAnnotator(Annotator):
    name: str
    column: str

    @abstractmethod
    def annotate_content(self, node: Node, content: dict[str, Any]) -> None:
        pass


class Content(ABC):
    name: str
    columns: list[str]

    def __init__(self) -> None:
        """Create per-render mutable state from subclass metadata."""

        # Subclasses declare their columns as class-level presentation metadata.
        # Copy it so annotators never modify another Content instance's schema.
        self.columns = list(self.columns)
        self.annotators: list[ContentAnnotator] = []

    def append_column(self, column: str) -> None:
        if column not in self.columns:
            self.columns.append(column)

    def add_annotator(self, obj: ContentAnnotator) -> None:
        self.append_column(obj.column)
        self.annotators.append(obj)

    @abstractmethod
    def gen_render(self, node: Node):
        pass

    def render(self, n: Node) -> None:
        self.gen_render(n)
        for an in self.annotators:
            if self.name in n.content:
                an.annotate_content(n, n.content[self.name])


class Graph:
    def __init__(
        self,
        cfg: CFGBase,
        nodes: set[Node] | None = None,
        edges: list[Edge] | None = None,
    ) -> None:
        self.cfg = cfg
        self.obj = cfg.graph
        self.nodes = nodes if nodes is not None else set()
        self.edges = edges if edges is not None else []

    def add_node(self, node: Node) -> None:
        self.nodes.add(node)

    def add_edge(self, edge: Edge) -> None:
        self.edges.append(edge)

    def remove_node(self, node: Node) -> None:
        self.nodes.remove(node)
        self.edges = list(
            filter(lambda edge: edge.src != node and edge.dst != node, self.edges)
        )

    def remove_edge(self, edge: Edge) -> None:
        self.edges.remove(edge)

    def filter_nodes(self, node_filter: Callable[[Node], bool]) -> None:
        new_graph = self.filtered_view(node_filter)
        self.nodes = new_graph.nodes
        self.edges = new_graph.edges

    def filtered_view(self, node_filter: Callable[[Node], bool]) -> "Graph":
        nodes = {node for node in self.nodes if node_filter(node)}
        edges = list(
            filter(
                lambda edge: node_filter(edge.src) and node_filter(edge.dst), self.edges
            )
        )
        return Graph(self.cfg, nodes, edges)


class Source(ABC):
    @abstractmethod
    def parse(self, cfg: CFGBase) -> Graph:
        pass


class Transformer(ABC):
    @abstractmethod
    def transform(self, graph: Graph) -> None:
        pass


class Output(ABC):
    @abstractmethod
    def generate(self, graph: Graph) -> str:
        pass


@dataclass
class Vis:
    source: Source
    output: Output

    transformers: list[Transformer] = field(default_factory=list)
    contents: list[Content] = field(default_factory=list)
    annotators: list[Annotator] = field(default_factory=list)

    def __post_init__(self) -> None:
        """Build render-time lookup tables after constructing the pipeline."""

        self.setup()

    def setup(self) -> None:
        """Validate and categorize the configured content and annotators."""

        # create a content dictionary out of the content list
        self._contents = {obj.name: obj for obj in self.contents}

        # split annotators by type
        self._node_annotators = []
        self._edge_annotators = []
        self._content_annotators = {}

        for annotator in self.annotators:
            if isinstance(annotator, NodeAnnotator):
                self._node_annotators.append(annotator)
            elif isinstance(annotator, EdgeAnnotator):
                self._edge_annotators.append(annotator)
            elif isinstance(annotator, ContentAnnotator):
                if annotator.name not in self._contents:
                    raise VisError(
                        f"Content '{annotator.name}' not found, required by annotator '{type(annotator)}'"
                    )
                self._contents[annotator.name].add_annotator(annotator)
            else:
                VisError(f"Unexpected annotator of type {type(annotator)}")

    @time_it
    def process(self, cfg: CFGBase) -> str:

        # parse input graph
        graph = self.source.parse(cfg)

        # apply graph transformations
        for t in self.transformers:
            t.transform(graph)

        for n in graph.nodes:
            for c in self._contents.values():
                c.render(n)
            for na in self._node_annotators:
                na.annotate_node(n)

        for e in graph.edges:
            for ea in self._edge_annotators:
                ea.annotate_edge(e)

        return self.output.generate(graph)
