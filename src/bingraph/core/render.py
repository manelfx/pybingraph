# Portions adapted from angr-utils (BSD-2-Clause); see NOTICE.

"""Render cached CFGs through presentation-only sources, contents and annotators.

Discovery belongs to core.project/cfg.builder. Comments can populate reference
metadata, and exit filtering/layout change the visible view, but none of these
operations creates proof edges or changes decoded instruction coverage.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

from angr import Project
from angr.analyses.cfg import CFGBase
from loguru import logger

from bingraph.cfg.models import CustomCFG
from bingraph.core import get_cfg
from bingraph.core.annotators.comments import CommentsDataRef
from bingraph.core.annotators.edges import ColorEdgesVex
from bingraph.core.annotators.nodes import ColorSimprocedures
from bingraph.core.comments import collect_custom_comments
from bingraph.core.contents import NodeAsm, NodeHead
from bingraph.core.outputs import DotOutput, validate_text_cfg_format
from bingraph.core.sources import CFGSource
from bingraph.core.vis import Annotator, Vis
from bingraph.helpers import CfgExits, CfgMode


def plot_cfg(
    cfg: CFGBase,
    fname: str,
    func_addr: int,
    dfs_rank: bool,
    comments: bool,
    exits: CfgExits,
    format: str,
    demangle: bool = False,
) -> None:
    """Write one CFG view using the requested annotation, exit and layout policies.

    Collect custom references only when comments are requested, then attach
    node/assembly content and style existing edges through the Vis pipeline.
    Demangling changes displayed names before shortening, never analysis
    identifiers. DotOutput owns format-specific Graphviz output at ``fname``.
    """

    logger.info(f"Start CFG plotting function at {hex(func_addr)} as {format}")

    annotators: list[Annotator] = []
    annotators.append(ColorSimprocedures())
    if comments:
        if isinstance(cfg, CustomCFG):
            collect_custom_comments(cfg)
        annotators.append(CommentsDataRef(demangle=demangle))
    annotators.append(ColorEdgesVex())

    vis = Vis(
        source=CFGSource(func_addr=func_addr, exits=exits),
        output=DotOutput(
            fname=fname, format=format, dfs_rank=dfs_rank, entry_addr=func_addr
        ),
        transformers=[],
        contents=[NodeHead(demangle=demangle), NodeAsm()],
        annotators=annotators,
    )

    vis.process(cfg)


@lru_cache
def render_cfg(
    project: Project,
    func_addr: int,
    dfs_rank: bool,
    comments: bool,
    cfg_mode: CfgMode,
    cfg_exits: CfgExits,
    format: str,
    demangle: bool = False,
) -> str:
    """Return textual Graphviz output for a function and its full view settings.

    Obtain the strategy-specific CFG, require its requested function in the
    knowledge base, and render through plot_cfg in a temporary directory. Read
    the resulting UTF-8 file exactly as generated. ``format`` accepts ``svg``,
    ``dot``, ``raw`` or ``xdot``; binary formats are rejected before discovery
    starts.

    The cache includes mode, comments, exit policy, layout, output format and
    demangling so one presentation cannot leak into another. Missing functions
    raise KeyError; analysis, rendering and file errors remain visible to the
    API/corpus caller. Temporary output is removed after the text is read.
    """
    validate_text_cfg_format(format)
    # Obtain the CFG first. We still prefer its block when the lift succeeds,
    # because it carries richer metadata than a disassembly-only graph.
    cfg = get_cfg(project, func_addr, cfg_mode)
    # Check if function is found in CFG.
    function = cfg.functions.get(func_addr)
    if not function:
        raise KeyError(f"Function {func_addr:#x} not found in CFG")

    # logger.info(serialize_function_summary(function))
    # logger.info(serialize_function(function))

    # Create SVG graph in temporal file
    with TemporaryDirectory() as tmp_dir:
        output_base = Path(tmp_dir) / f"cfg_{func_addr:x}"
        plot_cfg(
            cfg,
            str(output_base),
            func_addr=func_addr,
            dfs_rank=dfs_rank,
            comments=comments,
            exits=cfg_exits,
            format=format,
            demangle=demangle,
        )
        output_path = output_base.with_suffix(f".{format}")
        output = output_path.read_text(encoding="utf-8")

    # svg_start = output.find("<svg")
    # if svg_start != -1:
    #    output = output[svg_start:]
    return output
