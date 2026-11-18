"""The check reports each item of [main] extensions: the quick check runs
no extension code, and the tool check loads each item and its hints."""
from __future__ import annotations

import json
import sys
import uuid

import pytest
from langadapter import names, unregister

from testuary import config, orchestrator
from testuary._words import plain_name
from testuary.orchestrator import RunRequest
from testuary.settings import ConfigurationError, resolve_settings


def _project(tmp_path, extensions):
    """A project with one Python target whose project.ini names ``extensions``,
    and a marker file the extension files write when they run."""
    project = tmp_path / "proj"
    (project / ".testuary").mkdir(parents=True)
    (project / "calc.py").write_text("def double(x):\n    return x * 2\n", encoding="utf-8")
    (project / ".testuary" / "project.ini").write_text(f"[main]\nextensions = {extensions}\n",
                                                      encoding="utf-8")
    return project


def _extension(project, name, body):
    marker = project.parent / "ran.txt"
    (project / ".testuary" / name).write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran', encoding='utf-8')\n{body}",
        encoding="utf-8")
    return marker


def _settings(project):
    return resolve_settings(base_dir=project, models={}, overrides={
        "project": {"source_dir": str(project)},
        "api": {"model": "example", "provider": "openai", "api_url": "https://example.test/v1"}})


def test_the_quick_check_reports_each_item_and_runs_no_extension_code(tmp_path, monkeypatch):
    project = _project(tmp_path, "found_ext.py, missing_ext.py, onlyimport_pkg.child")
    marker = _extension(project, "found_ext.py", "")
    # A package whose child module only an import of the package finds.
    package = tmp_path / "lib" / "onlyimport_pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("__path__.append('elsewhere')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path / "lib"))
    monkeypatch.chdir(project)
    report = orchestrator.check(RunRequest(source_files=(str(project / "calc.py"),)),
                                settings=_settings(project))
    items = {item.selected: item for item in report.items if item.key.startswith("configuration.extensions.")}
    named = f"It is named in {project / '.testuary' / 'project.ini'}."
    found, missing, only = items["found_ext.py"], items["missing_ext.py"], items["onlyimport_pkg.child"]
    assert (found.status, found.tried_by) == ("unverified", "verification")
    assert found.reason == (f"It is {(project / '.testuary' / 'found_ext.py').resolve()}. This check runs "
                            f"no extension code. {named}")
    assert missing.status == "missing_setup" and missing.reason.endswith(named)
    assert (only.status, only.tried_by) == ("unverified", "verification")
    assert not marker.exists()
    assert plain_name(found) == "extension"


def test_the_tool_check_loads_each_item_and_reports_bad_hints(tmp_path):
    from testuary.readiness import _extension_admission, _extension_check_items

    project = _project(tmp_path, "good_ext.py, bad_hints_ext.py")
    marker = _extension(project, "good_ext.py", "testuary_hints = {'coverage': {'general': 'x'}}\n")
    _extension(project, "bad_hints_ext.py", "testuary_hints = {'coverage': {'bogus_cell': 'x'}}\n")
    settings = _settings(project)
    with config.use_run_settings(settings):
        # As the tool check's report: the admission loads, then the items.
        _extension_admission([], settings, verify=True)
        items = {item.selected: item for item in _extension_check_items(settings, verify=True)}
    assert items["good_ext.py"].status == "verified" and marker.exists()
    assert items["bad_hints_ext.py"].status == "failed" and "bogus_cell" in items["bad_hints_ext.py"].reason


def test_the_tool_check_refuses_settings_that_name_a_broken_extension(tmp_path, monkeypatch, capsys):
    """testuary check --verify loads the extensions while it resolves the
    settings, so a file that raises refuses them, naming it."""
    from testuary import __main__ as cli

    project = _project(tmp_path, "broken_ext.py")
    _extension(project, "broken_ext.py", "raise RuntimeError('broken on purpose')\n")
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.chdir(project)
    monkeypatch.setattr(sys, "argv", ["testuary", "check", "--json", "--verify", "--home", str(home),
                                      "--source-files", "calc.py"])
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    report = json.loads(capsys.readouterr().out)
    assert (stopped.value.code, report["status"]) == (2, "refused")
    assert "[main] extensions: broken_ext.py: RuntimeError: broken on purpose" in json.dumps(report)


def _language_project(tmp_path, language_line):
    """A project whose settings name toy_lang.py, which adds the language
    of a fresh name, and set ``language_line``, with a target of that
    language. Returns the project, the language and the target."""
    language = f"toy{uuid.uuid4().hex[:8]}"
    project = _project(tmp_path, "toy_lang.py")
    with (project / ".testuary" / "project.ini").open("a", encoding="utf-8") as settings:
        settings.write(f"[project]\n{language_line.format(language)}\n")
    marker = _extension(project, "toy_lang.py", (
        "from langadapter import LangAdapter\n"
        f"class Toy(LangAdapter):\n    language = {language!r}\n    file_extensions = ('.{language}',)\n"
        "langadapter_language = Toy\n"))
    target = project / f"calc.{language}"
    target.write_text("double(x) = x * 2\n", encoding="utf-8")
    return project, language, target, marker


def test_the_quick_check_leaves_an_extension_language_to_the_tool_check(tmp_path):
    project, language, target, marker = _language_project(tmp_path, "language = {}")
    report = orchestrator.check(RunRequest(source_files=(str(target),)), settings=_settings(project))
    items = {item.key: item for item in report.items}
    left, setting = items["configuration.target.1"], items["configuration.unchecked.1"]
    assert (left.status, left.tried_by, left.selected) == ("unverified", "verification", str(target))
    assert (setting.status, setting.tried_by) == ("unverified", "verification")
    assert setting.selected == f"[project] language = {language}"
    assert items["source.targets"].status != "verified"
    assert (plain_name(left), plain_name(setting)) == ("target", "setting")
    assert report.status != "refused" and not marker.exists() and language not in names()


def test_the_tool_check_loads_the_extensions_and_refuses_a_misspelled_language(tmp_path):
    project, language, target, marker = _language_project(tmp_path, "language = {}x")
    try:
        with pytest.raises(ConfigurationError, match=f"{language}x"):
            orchestrator.check(RunRequest(source_files=(str(target),)), settings=_settings(project),
                               verify=True)
        assert marker.exists() and language in names()
    finally:
        if language in names():
            unregister(language)


def test_the_quick_check_takes_a_named_file_of_an_extension_language(tmp_path, monkeypatch, capsys):
    """testuary check FILE does not compare the file with a [project]
    language left to the tool check."""
    from testuary import __main__ as cli

    project, language, target, marker = _language_project(tmp_path, "language = {}")
    home = tmp_path / "home"
    home.mkdir()
    (home / "models.ini").write_text(
        "[example]\nprovider = openai\nmodel = example\napi_url = https://example.test/v1\n", encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.setattr(sys, "argv", ["testuary", "check", "--json", "--home", str(home), "--model", "example",
                                      target.name])
    try:
        cli._main()
    except SystemExit as stopped:
        assert stopped.code != 2
    report = json.loads(capsys.readouterr().out)
    keys = {item["key"] for item in report["items"]}
    assert report["status"] != "refused" and not marker.exists()
    assert {"configuration.target.1", "configuration.unchecked.1"} <= keys
