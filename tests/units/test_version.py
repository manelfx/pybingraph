"""Version identity follows commit reachability and survives packaging."""

import json
from pathlib import Path
import runpy
import subprocess
import sys

import pytest

import bingraph
from bingraph import version


def git(root, *args):
    """Keep test histories local and independent of the developer's Git identity."""

    return subprocess.run(
        ["git", "-C", str(root), *args], check=True, capture_output=True, text=True
    ).stdout.strip()


@pytest.fixture
def repository(tmp_path, monkeypatch):
    root = tmp_path / "repository"
    (root / "src/bingraph").mkdir(parents=True)
    git(root, "init", "--initial-branch=main")
    git(root, "config", "user.name", "Version Test")
    git(root, "config", "user.email", "version@example.invalid")
    git(root, "config", "commit.gpgsign", "false")
    git(root, "commit", "--allow-empty", "-m", "Baseline")
    monkeypatch.setattr(version, "BASE_VERSION", (0, 1, 0))
    monkeypatch.setattr(version, "BASE_COMMIT", git(root, "rev-parse", "HEAD"))
    return root


def test_revision_changes_for_each_commit(repository):
    assert version.get_version_info(repository)[0] == "0.1.0"
    git(repository, "commit", "--allow-empty", "-m", "First")
    assert version.get_version_info(repository) == (
        "0.1.1",
        git(repository, "rev-parse", "HEAD"),
    )
    git(repository, "commit", "--allow-empty", "-m", "Second")
    assert version.get_version_info(repository)[0] == "0.1.2"


def test_baseline_revision_is_added(repository, monkeypatch):
    monkeypatch.setattr(version, "BASE_VERSION", (2, 3, 7))
    git(repository, "commit", "--allow-empty", "-m", "Next")
    assert version.get_version_info(repository)[0] == "2.3.8"


def test_fast_forward_preserves_dev_version(repository):
    git(repository, "switch", "-c", "dev")
    git(repository, "commit", "--allow-empty", "-m", "Development")
    expected = version.get_version_info(repository)
    git(repository, "switch", "main")
    git(repository, "merge", "--ff-only", "dev")
    assert version.get_version_info(repository) == expected
    git(repository, "switch", "--detach")
    assert version.get_version_info(repository) == expected


def test_merge_counts_both_histories_and_merge_commit(repository):
    git(repository, "switch", "-c", "dev")
    git(repository, "commit", "--allow-empty", "-m", "Development")
    assert version.get_version_info(repository)[0] == "0.1.1"
    git(repository, "switch", "main")
    git(repository, "commit", "--allow-empty", "-m", "Hotfix")
    git(repository, "merge", "--no-ff", "dev", "-m", "Merge")
    assert version.get_version_info(repository)[0] == "0.1.3"


def test_release_tags_and_dirty_sources_do_not_renumber(repository):
    git(repository, "commit", "--allow-empty", "-m", "Development")
    expected = version.get_version_info(repository)
    git(repository, "tag", "v0.1.1")
    (repository / "uncommitted.txt").write_text("pending changes")
    assert version.get_version_info(repository) == expected


def test_query_uses_package_checkout_not_current_directory(repository, monkeypatch):
    monkeypatch.chdir(repository.parent)
    monkeypatch.setattr(
        version, "__file__", str(repository / "src/bingraph/version.py")
    )
    assert version.get_version_info()[1] == git(repository, "rev-parse", "HEAD")


def test_frozen_distribution_needs_neither_git_nor_baseline(tmp_path, monkeypatch):
    package = tmp_path / "src/bingraph"
    package.mkdir(parents=True)
    (package / "version.json").write_text(
        json.dumps({"version": "1.2.34", "commit": "a" * 40})
    )
    monkeypatch.setenv("PATH", "")
    assert version.get_version_info(tmp_path) == ("1.2.34", "a" * 40)


@pytest.mark.parametrize(
    "info",
    [{"version": "1.2", "commit": "a" * 40}, {"version": "1.2.3", "commit": "bad"}],
)
def test_invalid_frozen_data_is_rejected(tmp_path, info):
    package = tmp_path / "src/bingraph"
    package.mkdir(parents=True)
    (package / "version.json").write_text(json.dumps(info))
    with pytest.raises(RuntimeError, match="Invalid packaged"):
        version.get_version_info(tmp_path)


def test_missing_git_data_does_not_use_a_parent_repository(tmp_path):
    (tmp_path / "src/bingraph").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="Git checkout or a built distribution"):
        version.get_version_info(tmp_path)


def test_git_unavailable_reports_actionable_error(repository, monkeypatch):
    monkeypatch.setenv("PATH", "")
    with pytest.raises(RuntimeError, match="complete history"):
        version.get_version_info(repository)


def test_baseline_must_be_an_ancestor(repository, monkeypatch):
    monkeypatch.setattr(version, "BASE_COMMIT", "0" * 40)
    with pytest.raises(RuntimeError, match="baseline"):
        version.get_version_info(repository)


def test_shallow_clone_is_rejected(repository, tmp_path):
    git(repository, "commit", "--allow-empty", "-m", "Development")
    clone = tmp_path / "shallow"
    git(tmp_path, "clone", "--depth=1", repository.as_uri(), str(clone))
    (clone / "src/bingraph").mkdir(parents=True)
    with pytest.raises(RuntimeError, match="unshallow"):
        version.get_version_info(clone)


def test_cli_version_does_not_import_application_dependencies(monkeypatch, capsys):
    from bingraph.api.cli import main

    monkeypatch.setattr(sys, "argv", ["bingraph", "--version"])
    monkeypatch.setitem(sys.modules, "bingraph.api.app", None)
    monkeypatch.setitem(sys.modules, "pydantic_settings", None)
    main()
    assert capsys.readouterr().out == f"bingraph {bingraph.__version__}\n"


def test_hatch_code_source_has_no_runtime_dependency_imports():
    namespace = runpy.run_path(str(Path(version.__file__)))
    assert namespace["get_version"]() == bingraph.__version__
    assert version.get_version_info()[1] == bingraph.__commit__


def test_cli_help_lists_version_and_existing_options(monkeypatch, capsys):
    from bingraph.api.cli import main

    monkeypatch.setattr(sys, "argv", ["bingraph", "--help"])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 0
    output = capsys.readouterr().out
    assert "--version" in output
    assert "--cfg-mode" in output
