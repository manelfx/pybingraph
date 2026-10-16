"""Generate the committed Python badge, or check it without modifying files.

The Makefile supplies its canonical PYTHON_VERSIONS list. Keep generation offline
and deterministic so each branch and release displays its own tested versions.
"""

import argparse
from html import escape
from pathlib import Path


BADGE = Path(__file__).resolve().parents[1] / "docs/images/python-versions.svg"


def python_badge(versions: list[str]) -> str:
    """Render a self-contained SVG with enough space for each version string."""

    value = " | ".join(versions)
    label_width, value_width = 56, len(value) * 7 + 16
    width = label_width + value_width
    text = escape(value, quote=True)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="20"
     role="img" aria-label="Python: {text}">
  <title>Python: {text}</title>
  <clipPath id="rounded"><rect width="{width}" height="20" rx="3"/></clipPath>
  <g clip-path="url(#rounded)">
    <rect width="{label_width}" height="20" fill="#555"/>
    <rect x="{label_width}" width="{value_width}" height="20" fill="#007ec6"/>
  </g>
  <g fill="white" font-family="DejaVu Sans Mono,monospace" font-size="11"
     text-anchor="middle">
    <text x="{label_width / 2:g}" y="14">python</text>
    <text x="{label_width + value_width / 2:g}" y="14">{text}</text>
  </g>
</svg>
'''


def update_badge(versions: list[str], check: bool = False) -> None:
    """Reject a missing/stale badge in check mode; only explicit generation writes."""

    expected = python_badge(versions)
    if check:
        if not BADGE.is_file() or BADGE.read_text(encoding="utf-8") != expected:
            raise SystemExit("Python badge is missing or stale; run `make badges`.")
        print("Python badge is up to date.")
    else:
        BADGE.write_text(expected, encoding="utf-8")
        print(f"Generated {BADGE.name}.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="validate without writing")
    parser.add_argument("versions", nargs="+", help="Python versions from the Makefile")
    args = parser.parse_args()
    update_badge(args.versions, args.check)
