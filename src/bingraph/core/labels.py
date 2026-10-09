"""Presentation-only label formatting shared by headers and annotations."""

import re


MAX_LABEL_LENGTH = 30


def short_label(name: str | None) -> str | None:
    """Shorten a displayed name, preserving a trailing hexadecimal offset."""

    if name is None:
        return None
    offset_match = re.search(r"[+-]0x[0-9a-fA-F]+$", name)
    label = name[: offset_match.start()] if offset_match else name
    offset = offset_match.group() if offset_match else ""
    if len(label) > MAX_LABEL_LENGTH:
        label = label[:MAX_LABEL_LENGTH] + "..."
    return label + offset
