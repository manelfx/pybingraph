"""Presentation-only label formatting shared by headers and annotations."""

import re
from bingraph.helpers.utils import demangle as demangle_name


MAX_LABEL_LENGTH = 30


def short_label(name: str | None, *, demangle: bool = False) -> str | None:
    """Optionally demangle, then shorten a name while preserving its offset."""

    if name is None:
        return None
    offset_match = re.search(r"[+-]0x[0-9a-fA-F]+$", name)
    label = name[: offset_match.start()] if offset_match else name
    offset = offset_match.group() if offset_match else ""
    if demangle:
        label = demangle_name(label)
    if len(label) > MAX_LABEL_LENGTH:
        label = label[:MAX_LABEL_LENGTH] + "..."
    return label + offset
