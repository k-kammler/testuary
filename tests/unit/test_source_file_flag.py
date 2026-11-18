"""``--source-files``, direct target injection: the targets
``orchestrator._source_file_top_files`` builds for the target phases, and
``orchestrator._select_pipeline_targets`` on the injected list."""
from __future__ import annotations

from pathlib import Path, PurePosixPath

import pytest

import testuary.__main__ as main_mod
from testuary import orchestrator as orch_mod
from testuary.settings import ConfigurationError

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _args(**over):
    ns = main_mod._build_parser().parse_args([])
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def _project(tmp_path: Path, names=("alpha.py", "beta.py", "gamma.py")):
    root = tmp_path / "proj"
    root.mkdir()
    for n in names:
        (root / n).write_text("def f(x):\n    return x\n", encoding="utf-8")
    return root


# _source_file_top_files

def test_top_files_are_built_in_list_order(tmp_path):
    """The named files become targets in list order, each with its ``rank`` and
    path only. Nothing is written to disk."""
    root = _project(tmp_path)
    args = _args(source_files="gamma.py,alpha.py")
    top = orch_mod._source_file_top_files(args, projects_root=root)
    assert [Path(t["file_path"]).name for t in top] == ["gamma.py", "alpha.py"]
    assert [t["target_number"] for t in top] == [1, 2]
    assert all(set(t) == {"target_number", "file_path"} for t in top)


def test_the_selection_line_goes_to_the_log_not_the_console(tmp_path, monkeypatch, capsys):
    """The report before the run names the files, so the selection line is
    a log line: a terminal display leaves it out, as it leaves out every
    log line, and plain output shows it with the others."""
    from testuary import _progress

    monkeypatch.setattr(orch_mod._log, "logger", None)
    monkeypatch.setattr(_progress, "_DETAIL_ECHO", False)
    written = []
    monkeypatch.setattr(orch_mod, "write_plain_log", lambda message, level=0: written.append(message))
    root = _project(tmp_path)
    orch_mod._source_file_top_files(_args(source_files="alpha.py"), projects_root=root)
    shown = capsys.readouterr()
    assert "target(s) selected" not in shown.out + shown.err
    assert written == ["1 target(s) selected."]
    monkeypatch.setattr(_progress, "_DETAIL_ECHO", True)
    orch_mod._source_file_top_files(_args(source_files="alpha.py"), projects_root=root)
    assert "target(s) selected" in capsys.readouterr().out


def test_top_files_container_keeps_posix_path_verbatim(tmp_path, monkeypatch, in_run):
    """In container mode an in-container POSIX --source-files path stays as
    written (pathlib on a Windows host would rewrite /home/... to
    C:\\home\\...), and the host stat is skipped, since the file is in the
    container."""
    monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
    in_run()
    posix = "/home/user/sympy/sympy/codegen/pyutils.py"
    args = _args(source_files=posix)
    top = orch_mod._source_file_top_files(
        args, projects_root="/home/user/sympy",
    )
    assert len(top) == 1
    assert top[0]["file_path"] == posix
    assert "\\" not in top[0]["file_path"]  # never Windows-mangled


def test_selection_reads_the_injected_list_in_order(tmp_path):
    """The injected candidates pass through ``_select_pipeline_targets`` and
    yield the files in list order, not alphabetical order, with no CSV
    read."""
    root = _project(tmp_path)
    args = _args(source_files="gamma.py,alpha.py,beta.py", targets="1-3")
    top = orch_mod._source_file_top_files(args, projects_root=root)
    picked = orch_mod._select_pipeline_targets(
        args, source_targets=top,
    )
    assert [Path(t["file_path"]).name for t in picked] == [
        "gamma.py", "alpha.py", "beta.py",
    ]
    assert [t["target_number"] for t in picked] == [1, 2, 3]


def test_selection_without_targets_takes_every_file(tmp_path):
    root = _project(tmp_path)
    args = _args(source_files="gamma.py,alpha.py,beta.py")
    top = orch_mod._source_file_top_files(args, projects_root=root)
    picked = orch_mod._select_pipeline_targets(args, source_targets=top)
    assert [t["target_number"] for t in picked] == [1, 2, 3]


def test_selection_refuses_a_blank_targets_value(tmp_path):
    root = _project(tmp_path)
    args = _args(source_files="alpha.py,beta.py", targets=" ")
    top = orch_mod._source_file_top_files(args, projects_root=root)
    with pytest.raises(ConfigurationError):
        orch_mod._select_pipeline_targets(args, source_targets=top)


def test_selection_slices_by_target_numbers(tmp_path):
    root = _project(tmp_path)
    args = _args(source_files="gamma.py,alpha.py,beta.py", targets="-2")
    top = orch_mod._source_file_top_files(args, projects_root=root)
    picked = orch_mod._select_pipeline_targets(
        args, source_targets=top,
    )
    assert [Path(t["file_path"]).name for t in picked] == ["gamma.py", "alpha.py"]


def test_top_files_relative_and_absolute_paths(tmp_path):
    root = _project(tmp_path)
    abs_path = root / "alpha.py"
    args = _args(source_files=f"beta.py,{abs_path}")
    top = orch_mod._source_file_top_files(args, projects_root=root)
    assert len(top) == 2


def test_top_files_dir_stops(tmp_path):
    root = _project(tmp_path)
    (root / "pkg").mkdir()
    args = _args(source_files="alpha.py,pkg")
    with pytest.raises(ConfigurationError, match="pkg is a directory, not a source file"):
        orch_mod._source_file_top_files(args, projects_root=root)


def test_top_files_missing_file_stops(tmp_path):
    root = _project(tmp_path)
    args = _args(source_files="alpha.py,missing.py")
    with pytest.raises(ConfigurationError, match="does not exist"):
        orch_mod._source_file_top_files(args, projects_root=root)


def test_top_files_skips_host_exists_check_in_container(tmp_path, monkeypatch, in_run):
    """In container mode the source lives in the container, so no host stat: a
    path that does not exist here is accepted and recorded as the target."""
    monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
    in_run()
    args = _args(source_files="/home/user/repo/pkg/mod.py")
    top = orch_mod._source_file_top_files(
        args, projects_root=PurePosixPath("/home/user/repo"),
    )
    assert len(top) == 1
    assert Path(top[0]["file_path"]).name == "mod.py"


def test_top_files_unknown_extension_stops(tmp_path):
    root = _project(tmp_path)
    (root / "notes.txt").write_text("x", encoding="utf-8")
    args = _args(source_files="notes.txt")
    with pytest.raises(ConfigurationError, match="no installed language"):
        orch_mod._source_file_top_files(args, projects_root=root)


def test_top_files_duplicate_listing_stops(tmp_path):
    root = _project(tmp_path)
    args = _args(source_files="alpha.py,alpha.py")
    with pytest.raises(ConfigurationError, match="selected twice"):
        orch_mod._source_file_top_files(args, projects_root=root)


def test_top_files_empty_list_stops(tmp_path):
    args = _args(source_files=" , ")
    with pytest.raises(ConfigurationError, match="no target selected"):
        orch_mod._source_file_top_files(args, projects_root=tmp_path)


def test_top_files_take_more_than_100_files(tmp_path):
    names = [f"f{i}.py" for i in range(101)]
    root = _project(tmp_path, names)
    top = orch_mod._source_file_top_files(_args(source_files=",".join(names)), projects_root=root)
    assert [Path(t["file_path"]).name for t in top] == names
    assert [t["target_number"] for t in top] == list(range(1, 102))


# --resume

@pytest.mark.parametrize("checking", [False, True])
def test_an_option_prefix_is_not_taken_for_the_option(checking, capsys):
    """--source-file is not --source-files: options are matched whole."""
    parser = main_mod._build_parser(checking=checking)
    with pytest.raises(SystemExit):
        parser.parse_args(["--source-file", "mod.py"])
    assert "unrecognized arguments: --source-file" in capsys.readouterr().err


@pytest.mark.parametrize("command, old, new", [
    ("testuary", "--source-folder", "--source-dir"),
    ("testuary-web", "--folder", "--dir"),
])
def test_a_renamed_option_is_refused_naming_the_new(command, old, new, capsys):
    """An old option name is refused as any unknown option is, naming the
    option that reads its "folder" as "dir"."""
    from testuary import web

    with pytest.raises(SystemExit):
        if command == "testuary":
            main_mod._build_parser().parse_args([old, "src"])
        else:
            web.main([old, "src"])
    err = capsys.readouterr().err
    assert "unrecognized arguments: " in err and f". The nearest known option is '{new}'" in err


def test_resume_parses():
    """--resume resumes the phases from their first verifiably
    incomplete unit."""
    ns = main_mod._build_parser().parse_args(
        ["--source-files", "pkg/mod.py", "--resume", "--targets", "1-2",
         "--output-dir", "out"],
    )
    assert ns.resume is True
    assert ns.source_files == "pkg/mod.py"
    assert ns.targets == "1-2"
    assert ns.output_dir == "out"


