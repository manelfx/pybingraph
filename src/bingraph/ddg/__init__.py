"""Bounded instruction dependencies, independent of CFG discovery."""

from .builder import get_ddg
from .render import render_ddg

__all__ = ["get_ddg", "render_ddg"]
