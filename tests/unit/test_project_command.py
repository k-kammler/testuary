"""``testuary [FILE ...]`` and ``testuary check [FILE ...]`` in a project:
the files named, relative to the directory the command starts in, or else the
project's ``[project] source_files``."""
from __future__ import annotations

import os
import sys

import pytest

from testuary import __main__ as cli
from testuary import config, orchestrator
from testuary import settings as settings_module
from testuary.settings import ConfigurationError, resolve_settings

PROJECT = "[project]\nsource_files = pkg/a.py, pkg/b.py\nlanguage = python\n\n[api]\nmodel = local_qwen\n"


@pytest.fixture
def project_dir(tmp_path):
    (tmp_path / ".testuary").mkdir()
    (tmp_path / ".testuary" / "project.ini").write_text(PROJECT, encoding="utf-8")
    (tmp_path / "pkg").mkdir()
    for name in ("pkg/a.py", "pkg/b.py", "pkg/c.py", "x.py", "Calc.java"):
        (tmp_path / name).write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def _files(project_dir, named, *, start=None, source_files=None):
    settings = resolve_settings(base_dir=start or project_dir, environment={})
    return cli._project_files(named, source_files, settings, start=start or project_dir)


def test_without_named_files_a_run_takes_the_projects_files(project_dir):
    assert _files(project_dir, []) == (("pkg/a.py", "pkg/b.py"), True)


def test_named_files_are_relative_to_the_starting_dir(project_dir):
    assert _files(project_dir, ["pkg/c.py"]) == (("pkg/c.py",), False)
    assert _files(project_dir, ["c.py", "../x.py"], start=project_dir / "pkg") == (("pkg/c.py", "x.py"), False)
    assert _files(project_dir, [str(project_dir / "x.py")]) == (("x.py",), False)


def test_source_files_named_with_the_option_stay_as_given(project_dir):
    assert _files(project_dir, [], source_files="pkg/c.py") == (("pkg/c.py",), False)
    with pytest.raises(ConfigurationError, match="once"):
        _files(project_dir, ["x.py"], source_files="pkg/c.py")


def test_a_file_outside_the_project_or_of_another_language_is_refused(project_dir, tmp_path_factory):
    outside = tmp_path_factory.mktemp("outside") / "y.py"
    outside.write_text("y = 1\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="outside the source directory"):
        _files(project_dir, [str(outside)])
    with pytest.raises(ConfigurationError, match="not a python source file"):
        _files(project_dir, ["Calc.java"])


def test_a_project_without_files_leaves_the_refusal_to_the_run(tmp_path):
    assert cli._project_files([], None, resolve_settings(base_dir=tmp_path, environment={}),
                              start=tmp_path) == ((), False)


def _entry(project_dir, monkeypatch, argv):
    monkeypatch.chdir(project_dir)
    seen = []
    real = settings_module.resolve_settings

    def resolve(**kwargs):
        seen.append(kwargs.get("overrides") or {})
        return real(**{**kwargs, "environment": {}})

    monkeypatch.setattr(settings_module, "resolve_settings", resolve)

    def run(args, *rest, **kwargs):
        seen.append(args)
        return 0

    monkeypatch.setattr(orchestrator, "_run_pipeline", run)
    monkeypatch.setattr(sys, "argv", ["testuary", *argv, "--display", "plain"])
    try:
        return cli.main(), seen
    except SystemExit as stop:
        return stop.code, seen


def test_the_entry_runs_the_projects_files_after_listing_them(project_dir, monkeypatch, capsys):
    status, seen = _entry(project_dir, monkeypatch, [])
    assert status == 0
    overrides, args = seen
    assert args.source_files == ("pkg/a.py", "pkg/b.py")
    out = capsys.readouterr().out
    assert f"Project {project_dir.name}: 2 targets, each a full run:" in out


def test_the_entry_runs_named_files_after_its_flags(project_dir, monkeypatch, capsys):
    status, seen = _entry(project_dir, monkeypatch, ["--stop-after-phase", "starter", "pkg/c.py"])
    assert status == 0
    _overrides, args = seen
    assert args.source_files == ("pkg/c.py",) and args.stop_after_phase == "starter"
    assert "each a full run" not in capsys.readouterr().out


def test_generate_and_check_take_files_too(project_dir, monkeypatch):
    status, seen = _entry(project_dir, monkeypatch, ["generate", "x.py"])
    assert status == 0 and seen[-1].source_files == ("x.py",)


def test_a_named_file_with_a_comma_in_its_name_is_one_target(project_dir, monkeypatch):
    """A file name may hold a comma: the run, and the run request that its
    pre-run summary and check read, take it as one target."""
    (project_dir / "pkg" / "a,b.py").write_text("x = 1\n", encoding="utf-8")
    status, seen = _entry(project_dir, monkeypatch, ["pkg/a,b.py"])
    assert status == 0
    with config.use_run_settings(resolve_settings(base_dir=project_dir, environment={})):
        targets = orchestrator._source_file_top_files(seen[-1], projects_root=project_dir)
    path = str((project_dir / "pkg" / "a,b.py").resolve())
    assert targets == [{"target_number": 1, "file_path": path}]
    assert orchestrator._run_request(seen[-1]).source_files == ("pkg/a,b.py",)


def test_a_word_that_is_no_file_is_refused_before_any_work(project_dir, monkeypatch, capsys):
    status, seen = _entry(project_dir, monkeypatch, ["demo"])
    assert status == 2 and not any(not isinstance(item, dict) for item in seen)
    assert "demo" in capsys.readouterr().err


def test_without_a_project_and_files_the_command_shows_its_usage(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["testuary"])
    assert cli.main() == 2
    assert "usage: testuary" in capsys.readouterr().out


def test_the_project_option_is_only_a_label(project_dir):
    settings = resolve_settings(base_dir=project_dir, environment={},
                                overrides={"project": {"name": "elsewhere"}})
    from testuary import config as config_module

    with config_module.use_run_settings(settings):
        config = config_module.get_project_config()
    assert config["project_name"] == "elsewhere"
    assert config["source_dir"] == str(project_dir.resolve())
    assert os.path.isdir(config["source_dir"])
