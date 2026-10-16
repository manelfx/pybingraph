"""The version badge is deterministic, accessible and checked without rewriting."""

import importlib.util
from pathlib import Path
from xml.etree import ElementTree

import pytest


@pytest.fixture
def badges(tmp_path):
    path = Path(__file__).resolve().parents[2] / "scripts/badges.py"
    spec = importlib.util.spec_from_file_location("badge_checks", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.BADGE = tmp_path / "python.svg"
    return module


def test_badge_generation_and_read_only_validation(badges):
    versions = ["3.12", "3.13", "3.14", "3.15"]
    badges.update_badge(versions)
    original = badges.BADGE.read_bytes()
    modified = badges.BADGE.stat().st_mtime_ns
    badges.update_badge(versions, check=True)
    assert badges.BADGE.read_bytes() == original
    assert badges.BADGE.stat().st_mtime_ns == modified
    svg = ElementTree.fromstring(original)
    assert svg.attrib["aria-label"] == "Python: " + " | ".join(versions)
    assert (
        svg.find("{http://www.w3.org/2000/svg}title").text == svg.attrib["aria-label"]
    )
    assert svg.findall(".//{http://www.w3.org/2000/svg}text")[-1].text == " | ".join(
        versions
    )


@pytest.mark.parametrize("existing", [None, "stale badge"])
def test_badge_check_rejects_missing_or_stale_without_writing(badges, existing):
    if existing is not None:
        badges.BADGE.write_text(existing)
    with pytest.raises(SystemExit, match="make badges"):
        badges.update_badge(["3.12"], check=True)
    if existing is None:
        assert not badges.BADGE.exists()
    else:
        assert badges.BADGE.read_text() == existing


def test_badge_escapes_xml_and_adapts_to_version_list(badges):
    svg = ElementTree.fromstring(badges.python_badge(["3.12", "<future>&"]))
    assert svg.attrib["aria-label"] == "Python: 3.12 | <future>&"
    smaller = ElementTree.fromstring(badges.python_badge(["3.12"]))
    assert int(smaller.attrib["width"]) < int(svg.attrib["width"])
