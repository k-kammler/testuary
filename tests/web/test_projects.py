"""Projects are project directories: the interface lists the recent ones, kept in
Testuary's user directory, changes a project's settings file through
Testuary's setup, and starts its runs and checks in its directory, as
``testuary [FILES]`` and ``testuary check`` run there."""
from __future__ import annotations

import json
import os
from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import wait_for
from testuary.web import _data as catalogues
from testuary.web import _data as projects
import configparser
import re
import shutil
import subprocess
import sys
import threading
import time
from conftest import SOURCE


@pytest.fixture(autouse=True)
def no_machine_settings(monkeypatch):
    for name in list(os.environ):
        if name.startswith("TESTUARY_") and name != "TESTUARY_HOME":
            monkeypatch.delenv(name)


@pytest.fixture
def calc(tmp_path):
    project_dir = tmp_path / "calc"
    (project_dir / ".testuary").mkdir(parents=True)
    (project_dir / ".testuary" / "project.ini").write_text(
        "# Mine.\n[project]\nsource_files = calc.py\nlanguage = python\n\n[api]\nmodel = local\n",
        encoding="utf-8")
    (project_dir / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    return project_dir


def _recent(user):
    return json.loads((user / "web" / "projects.json").read_text(encoding="utf-8"))


def test_a_project_dir_is_added_listed_and_forgotten(serve, calc, user_dir):
    app, server, client = serve()
    status, view = client.json("GET", "/api/projects")
    assert status == 200 and view["projects"] == []
    projects.add(str(calc))
    assert _recent(user_dir) == {"project_dirs": [str(calc.resolve())]}
    status, view = client.json("GET", "/api/projects")
    (entry,) = view["projects"]
    assert entry["project_dir"] == str(calc.resolve()) and entry["name"] == "calc"
    assert entry["fields"] == {"source_files": "calc.py", "language": "python", "model": "local"}
    # Testuary's plain names of the settings, for the page.
    assert entry["labels"]["source_files"] == "Source files" and entry["labels"]["model"] == "Model"
    assert entry["settings_file"] == str(calc.resolve() / ".testuary" / "project.ini")
    assert "python" in view["languages"]
    status, _ = client.json("POST", "/api/projects/forget", {"project_dir": str(calc)})
    assert status == 200 and _recent(user_dir) == {"project_dirs": []}
    # Forgetting a project never touches its directory.
    assert (calc / ".testuary" / "project.ini").is_file() and (calc / "calc.py").is_file()


@pytest.mark.parametrize("project_dir, message", [
    ("relative/path", "absolute"), ("{missing}", "no directory"), ("//server/share", "local"),
])
def test_only_an_existing_local_dir_is_added(tmp_path, user_dir, project_dir, message):
    with pytest.raises(catalogues.Refusal, match=message) as refused:
        projects.add(project_dir.format(missing=tmp_path / "missing"))
    assert [field for field, _ in refused.value.problems] == ["project_dir"]
    assert not (user_dir / "web" / "projects.json").exists()


def test_projects_added_at_the_same_time_are_all_listed(tmp_path, user_dir, monkeypatch):
    """The server adds projects from its request threads: none loses
    another's addition, even when each reads the list slowly."""
    read = projects.recent

    def slow_recent():
        entries = read()
        time.sleep(0.05)
        return entries

    monkeypatch.setattr(projects, "recent", slow_recent)
    project_dirs = [tmp_path / f"project{index}" for index in range(8)]
    for project_dir in project_dirs:
        project_dir.mkdir()
    threads = [threading.Thread(target=projects.add, args=(str(project_dir),)) for project_dir in project_dirs]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert sorted(_recent(user_dir)["project_dirs"]) == sorted(str(project_dir.resolve()) for project_dir in project_dirs)


def test_an_addition_waits_for_a_reading_of_the_list(tmp_path, user_dir, monkeypatch):
    """Windows cannot replace a file another thread has open: an addition
    waits while a listing request reads the list."""
    first, second = tmp_path / "first", tmp_path / "second"
    first.mkdir()
    second.mkdir()
    projects.add(str(first))
    reading = threading.Event()
    read_text = Path.read_text

    def slow_read_text(path, *args, **kwargs):
        if path.name != "projects.json" or threading.current_thread() is not reader:
            return read_text(path, *args, **kwargs)
        with open(path, encoding="utf-8") as handle:
            reading.set()
            time.sleep(0.5)
            return handle.read()

    monkeypatch.setattr(Path, "read_text", slow_read_text)
    reader = threading.Thread(target=projects.recent)
    reader.start()
    assert reading.wait(5)
    projects.add(str(second))
    reader.join()
    assert _recent(user_dir)["project_dirs"] == [str(second.resolve()), str(first.resolve())]


def test_a_project_joins_the_list_only_through_its_setup(serve, calc):
    """The page adds a project by writing its settings through Testuary's
    setup, which lists it. No request lists a directory on its own."""
    app, server, client = serve()
    assert client.json("POST", "/api/projects/add", {"project_dir": str(calc)})[0] == 404
    assert client.json("GET", "/api/projects")[1]["projects"] == []


def test_the_working_dir_project_is_listed_first(serve, calc):
    app, server, client = serve(calc / "src")
    status, view = client.json("GET", "/api/projects")
    assert [entry["project_dir"] for entry in view["projects"]] == [str(calc.resolve())]
    assert view["projects"][0]["working"] is True


def _with_settings(project_dir):
    (project_dir / ".testuary").mkdir(parents=True)
    (project_dir / ".testuary" / "project.ini").write_text("[project]\nlanguage = python\n",
                                                      encoding="utf-8")
    return project_dir


def test_a_dir_without_a_project_lists_the_projects_directly_below_it(serve, tmp_path):
    """Started in a directory that belongs to no project, such as one that
    holds several projects, the interface lists the projects directly below it, by
    name, before the recent ones. Runs and checks take them as listed
    projects."""
    parent = tmp_path / "parent"
    tenacity, jmustache = _with_settings(parent / "tenacity"), _with_settings(parent / "jmustache")
    (parent / "wheels").mkdir()
    _with_settings(parent / "wheels" / "deeper")
    app, server, client = serve(parent)
    status, view = client.json("GET", "/api/projects")
    assert status == 200
    assert [(entry["name"], entry["working"], entry["below"]) for entry in view["projects"]] == [
        ("jmustache", False, True), ("tenacity", False, True)]
    assert view["projects"][1]["fields"] == {"language": "python"}
    assert projects.known(parent, str(tenacity)) == tenacity.resolve()
    assert set(projects.marks(parent)) == {"jmustache", "tenacity"}
    # A recent project follows them, and one of them that is also recent
    # stays where it is found.
    projects.add(str(jmustache))
    projects.add(str(_with_settings(tmp_path / "elsewhere")))
    names = [entry["name"] for entry in client.json("GET", "/api/projects")[1]["projects"]]
    assert names == ["jmustache", "tenacity", "elsewhere"]


def test_a_projects_dir_lists_no_project_below_it(serve, calc):
    """Inside a project the interface lists that project, not the ones in
    its subdirectories."""
    _with_settings(calc / "vendored")
    app, server, client = serve(calc)
    status, view = client.json("GET", "/api/projects")
    assert [(entry["name"], entry["working"], entry["below"]) for entry in view["projects"]] == [
        ("calc", True, False)]
    with pytest.raises(catalogues.Refusal):
        projects.known(calc, str(calc / "vendored"))


def test_a_projects_form_offers_each_language_setting_once_by_its_name(serve, calc):
    """Each language's settings come once each, in the order of Testuary's
    description, with Testuary's plain name."""
    app, server, client = serve()
    status, view = client.json("GET", "/api/projects")
    offered = view["language_settings"]
    python = [setting["name"] for setting in offered["python"]]
    assert python[:3] == ["python_import_root", "python_test_environment", "python_executable"]
    assert "python_version" in python and len(python) == len(set(python))
    assert [setting["label"] for setting in offered["python"][:3]] == [
        "Python import root", "Python test environment", "Python executable"]
    java = [setting["name"] for setting in offered["java"]]
    assert {"java_home", "maven", "gradle"} <= set(java) and len(java) == len(set(java))
    assert all(setting["label"] for settings in offered.values() for setting in settings)


def test_a_projects_settings_change_through_testuarys_setup(serve, calc):
    app, server, client = serve()
    projects.add(str(calc))
    status, _ = client.json("POST", "/api/project", {
        "project_dir": str(calc), "values": {"source_files": "calc.py, more.py", "model": ""},
        "remove": ["language"]})
    assert status == 200
    assert (calc / ".testuary" / "project.ini").read_text(encoding="utf-8") == (
        "# Mine.\n[project]\nsource_files = calc.py, more.py\n\n[api]\n")
    status, refused = client.json("POST", "/api/project", {
        "project_dir": str(calc), "values": {"language": "cobol"}})
    assert status == 400 and [field for field, _ in refused["problems"]] == ["language"]


def test_a_path_the_setup_refuses_is_refused_beside_its_field(serve, calc, tmp_path):
    """An interpreter or import directory that does not exist is refused as
    Testuary's setup refuses it, by the field, and nothing is written."""
    app, server, client = serve()
    projects.add(str(calc))
    before = (calc / ".testuary" / "project.ini").read_bytes()
    status, refused = client.json("POST", "/api/project", {"project_dir": str(calc), "values": {
        "python_executable": str(tmp_path / "nope" / "python.exe"), "python_import_root": "nope"}})
    assert status == 400
    assert dict(refused["problems"]) == {"python_executable": "this interpreter does not run",
                                         "python_import_root": "no such directory"}
    assert (calc / ".testuary" / "project.ini").read_bytes() == before


def test_a_dir_that_is_no_recent_project_is_refused(serve, calc, tmp_path):
    app, server, client = serve()
    for path, body in (("/api/project", {"project_dir": str(calc), "values": {}}),
                       ("/api/run", {"project_dir": str(calc)}),
                       ("/api/check", {"project_dir": str(calc)})):
        status, refused = client.json("POST", path, body)
        assert status == 400 and refused["field"] == "project_dir", path


def test_a_run_and_a_check_start_in_the_projects_dir(serve, calc, stand_in):
    app, server, client = serve()
    projects.add(str(calc))
    status, job = client.json("POST", "/api/run", {
        "project_dir": str(calc), "files": ["calc.py"], "model": "other", "runs": "2"})
    assert status == 201
    assert job["arguments"][:4] == ["calc.py", "--display", "plain", "--no-color"]
    assert "--model=other" in job["arguments"] and "--runs=2" in job["arguments"]
    assert "--limits-profile" not in job["arguments"]
    assert job["project"] == str(calc.resolve())
    assert wait_for(lambda: client.json("GET", f"/api/jobs/{job['id']}")[1]["status"] != "running")
    status, job = client.json("POST", "/api/check", {"project_dir": str(calc), "verify": True})
    assert status == 201 and job["arguments"] == ["check", "--json", "--verify"]
    assert job["project"] == str(calc.resolve())


def test_a_check_after_a_run_judges_the_dir_the_next_run_uses(serve, calc, stand_in):
    """Testuary refuses a new run in a run directory that holds a saved
    run, so the check judges what a run from the Run page does by default:
    it continues a saved run with unfinished files, else it starts fresh in
    the free directory beside it."""
    from testuary.presentation.terminal import FileCompletion, RunCompletion, write_completion

    app, server, client = serve()
    projects.add(str(calc))
    status, job = client.json("POST", "/api/check", {"project_dir": str(calc)})
    assert status == 201 and job["arguments"] == ["check", "--json"]
    ended = lambda job: client.json("GET", f"/api/jobs/{job['id']}")[1]["status"] != "running"  # noqa: E731
    assert wait_for(lambda: ended(job))
    results = calc / ".testuary" / "results" / "local" / "results"
    write_completion(RunCompletion(files=(
        FileCompletion("calc.py", "complete"),), results_directory=str(results)))
    status, job = client.json("POST", "/api/check", {"project_dir": str(calc), "verify": True})
    assert status == 201
    assert job["arguments"] == ["check", "--json", "--verify", "--output-dir=.testuary/results/local-2"]
    assert wait_for(lambda: ended(job))
    write_completion(RunCompletion(files=(
        FileCompletion("calc.py", "no_complete_suite", reason="no tests"),),
        results_directory=str(results)))
    status, job = client.json("POST", "/api/check", {"project_dir": str(calc)})
    assert status == 201 and job["arguments"] == ["check", "--json", "--resume"]


def test_a_file_name_that_reads_as_an_option_is_refused(serve, calc):
    app, server, client = serve()
    projects.add(str(calc))
    status, refused = client.json("POST", "/api/run", {"project_dir": str(calc), "files": ["--help"]})
    assert status == 400 and refused["field"] == "files"


def test_settings_at_both_levels_with_their_sources(serve, calc, user_dir):
    (user_dir / "config.ini").write_text("[main]\nseed = 3\n", encoding="utf-8")
    app, server, client = serve()
    projects.add(str(calc))
    status, view = client.json("GET", f"/api/settings?project_dir={calc}")
    assert status == 200 and view["error"] is None
    assert view["files"] == {"global": str(user_dir.resolve() / "config.ini"),
                             "project": str(calc.resolve() / ".testuary" / "project.ini")}
    rows = {(row["section"], row["name"]): row for section in view["sections"]
            for row in section["settings"]}
    assert (rows[("main", "seed")]["value"], rows[("main", "seed")]["source"]) == (
        "3", str(user_dir.resolve() / "config.ini"))
    assert rows[("api", "model")]["source"] == str(calc.resolve() / ".testuary" / "project.ini")
    status, _ = client.json("POST", "/api/settings", {
        "scope": "project", "project_dir": str(calc), "section": "main", "values": {"seed": "5"}})
    assert status == 200
    assert "[main]\nseed = 5\n" in (calc / ".testuary" / "project.ini").read_text(encoding="utf-8")
    status, _ = client.json("POST", "/api/settings", {
        "scope": "global", "section": "main", "remove": ["seed"]})
    assert status == 200 and (user_dir / "config.ini").read_text(encoding="utf-8") == "[main]\n"


def test_the_project_scope_refuses_testuary_dir_in_the_runs_words(serve, calc):
    """The Settings page refuses [main] testuary_dir in a project's file
    with the run's own words, and names the field."""
    app, server, client = serve()
    projects.add(str(calc))
    before = (calc / ".testuary" / "project.ini").read_bytes()
    status, refused = client.json("POST", "/api/settings", {
        "scope": "project", "project_dir": str(calc), "section": "main",
        "values": {"testuary_dir": "elsewhere"}})
    assert status == 400
    assert refused["error"].endswith("[main] testuary_dir locates this file. Set it in the global "
                                     "configuration, the environment or on the command line")
    assert refused["field"] == "testuary_dir"
    assert (calc / ".testuary" / "project.ini").read_bytes() == before


def test_a_global_value_a_project_overrides_stays_resettable(serve, calc, user_dir):
    """The row names every file that sets the key, so each level's value can
    be reset while another level's value is in force."""
    global_file = str(user_dir.resolve() / "config.ini")
    project_file = str(calc.resolve() / ".testuary" / "project.ini")
    (user_dir / "config.ini").write_text("[main]\nseed = 3\n", encoding="utf-8")
    # The interface started in the project directory.
    app, server, client = serve(calc)
    projects.add(str(calc))
    assert client.json("POST", "/api/settings", {
        "scope": "project", "project_dir": str(calc), "section": "main", "values": {"seed": "5"}})[0] == 200

    def seed(query=f"?project_dir={calc}"):
        view = client.json("GET", f"/api/settings{query}")[1]
        return next(row for section in view["sections"] for row in section["settings"]
                    if (row["section"], row["name"]) == ("main", "seed"))

    row = seed()
    assert (row["value"], row["source"], row["sources"]) == ("5", project_file, [global_file, project_file])
    # The global page reads the settings without the project: the global
    # file's own value, which a save there writes back unchanged.
    row = seed("")
    assert (row["value"], row["source"], row["sources"]) == ("3", global_file, [global_file])
    assert client.json("POST", "/api/settings", {
        "scope": "global", "section": "main", "remove": ["seed"]})[0] == 200
    row = seed()
    assert (row["value"], row["source"], row["sources"]) == ("5", project_file, [project_file])
    assert (user_dir / "config.ini").read_text(encoding="utf-8") == "[main]\n"
    assert client.json("POST", "/api/settings", {
        "scope": "project", "project_dir": str(calc), "section": "main", "remove": ["seed"]})[0] == 200
    assert seed()["sources"] == []


def test_the_extensions_row_edits_each_levels_own_items(serve, calc, user_dir):
    """[main] extensions adds each level's items: a page's row edits the
    items its own file names and shows the combined list in force, and a
    project's save writes the project's own items alone."""
    global_file = str(user_dir.resolve() / "config.ini")
    project_file = calc.resolve() / ".testuary" / "project.ini"
    (user_dir / "config.ini").write_text("[main]\nextensions = user_ext.py\n", encoding="utf-8")
    _, _, client = serve(calc)
    projects.add(str(calc))
    assert client.json("POST", "/api/settings", {
        "scope": "project", "project_dir": str(calc), "section": "main",
        "values": {"extensions": "my_hints.py"}})[0] == 200
    written = configparser.ConfigParser(interpolation=None)
    written.read(project_file, encoding="utf-8")
    assert written["main"]["extensions"] == "my_hints.py"

    def row(query=f"?project_dir={calc}"):
        view = client.json("GET", f"/api/settings{query}")[1]
        return next(row for section in view["sections"] for row in section["settings"]
                    if (row["section"], row["name"]) == ("main", "extensions"))

    project_row, global_row = row(), row("")
    assert (project_row["value"], project_row["combined"], project_row["sources"]) == (
        "my_hints.py", "user_ext.py, my_hints.py", [global_file, str(project_file)])
    assert (global_row["value"], global_row["combined"]) == ("user_ext.py", "user_ext.py")


def test_a_project_card_names_the_extensions_its_settings_file_names(serve, tmp_path):
    """A settings file that names only [main] extensions is not a project
    without settings of its own: its card lists the extensions."""
    project = tmp_path / "tool"
    (project / ".testuary").mkdir(parents=True)
    (project / ".testuary" / "project.ini").write_text("[main]\nextensions = my_hints.py\n", encoding="utf-8")
    _, _, client = serve()
    projects.add(str(project))
    (entry,) = client.json("GET", "/api/projects")[1]["projects"]
    assert entry["fields"] == {"extensions": "my_hints.py"}
    assert entry["labels"] == {"extensions": "Extensions"}


def test_a_language_an_extension_may_add_is_saved_and_left_to_the_tool_check(serve, calc):
    """A project that names an extension can set a language no registered
    language has: the page saves it, loads no extension and says the tool
    check checks it."""
    marker = calc / "ran.txt"
    (calc / ".testuary").mkdir(exist_ok=True)
    (calc / ".testuary" / "toy_lang.py").write_text(
        f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran', encoding='utf-8')\n", encoding="utf-8")
    _, _, client = serve(calc)
    projects.add(str(calc))
    for section, values in (("main", {"extensions": "toy_lang.py"}), ("project", {"language": "toyxyz"})):
        assert client.json("POST", "/api/settings", {
            "scope": "project", "project_dir": str(calc), "section": section, "values": values})[0] == 200
    view = client.json("GET", f"/api/settings?project_dir={calc}")[1]
    assert view["unchecked"] == ["[project] language = toyxyz"] and not marker.exists()


def test_each_settings_page_reads_and_links_its_own_level():
    """The project page reads the project's settings. The global page reads
    the settings without a project, and with the project it came from as
    context only. Choosing the global settings from a project's page keeps
    that project as context."""
    from conftest import page_value

    project = {"project_dir": "C:/p q", "name": "calc"}
    requests = page_value(["settingsRequests"], "[%s].map(([scope, project]) => settingsRequests(scope, project))"
                          % ", ".join(json.dumps(case) for case in (
                              ["global", None], ["project", project], ["global", project])))
    assert requests == [["/api/settings"], ["/api/settings?project_dir=C%3A%2Fp%20q"],
                        ["/api/settings", "/api/settings?project_dir=C%3A%2Fp%20q"]]
    links = page_value(["settingsLink"], "[%s].map(([choice, project]) => settingsLink(choice, project))"
                       % ", ".join(json.dumps(case) for case in (
                           ["", None], ["", project], ["C:/p q", None], ["D:/r", project])))
    assert links == ["#/settings", "#/settings?project_dir=C%3A%2Fp%20q&scope=global",
                     "#/settings?project_dir=C%3A%2Fp%20q", "#/settings?project_dir=D%3A%2Fr"]


def test_the_global_page_says_in_text_when_the_projects_value_wins():
    """On the global page, the project it came from shows only as text: the
    value in force there when it differs from the global page's source."""
    from conftest import page_value

    files = {"global": "G/config.ini", "project": "P/project.ini"}
    cases = [  # (the global page's row, the project's row)
        ({"value": "3", "source": "G/config.ini"}, {"value": "5", "source": "P/project.ini"}),
        ({"value": "3", "source": "default"}, {"value": "5", "source": "P/project.ini"}),
        ({"value": "3", "source": "G/config.ini"}, {"value": "3", "source": "G/config.ini"}),
        ({"value": "7", "source": "environment"}, {"value": "7", "source": "environment"}),
        ({"value": "3", "source": "G/config.ini"}, None),
    ]
    script = ("%s.map(([setting, other]) => contextNote(setting, other, { files: %s, context: 'calc' }))"
              % (json.dumps(cases), json.dumps(files)))
    assert page_value(["sourceWords", "contextNote"], script) == [
        "In the project calc, the value in force is 5, from its own settings.",
        "In the project calc, the value in force is 5, from its own settings.",
        None, None, None]


def test_a_row_offers_its_levels_reset_and_says_when_another_level_wins():
    """Reset is offered for the file the page changes whenever that file
    sets the key, also when another level's value is in force, and the row
    then says which level wins."""
    from conftest import page_value

    files = {"global": "G/config.ini", "project": "P/project.ini"}
    cases = [  # (value's source, files that set it, file the page changes)
        ("P/project.ini", ["G/config.ini", "P/project.ini"], "G/config.ini"),
        ("P/project.ini", ["G/config.ini", "P/project.ini"], "P/project.ini"),
        ("environment", ["P/project.ini", "environment"], "P/project.ini"),
        ("default", [], "G/config.ini"),
        ("P/project.ini", ["P/project.ini"], "G/config.ini"),
    ]
    script = ("%s.map(([source, sources, file]) => { const setting = { source, sources };"
              " const target = { file, files: %s };"
              " return [canReset(setting, target), overrideNote(setting, target)]; })"
              % (json.dumps(cases), json.dumps(files)))
    assert page_value(["sourceWords", "canReset", "levelWords", "overrideNote"], script) == [
        [True, "The global settings set this too. The value in force comes from this project."],
        [True, None],
        [True, "This project's settings set this too. The value in force comes from the environment."],
        [False, None],
        [False, None],
    ]


def test_the_results_of_a_recent_project_are_shown(serve, calc):
    run = calc / ".testuary" / "results" / "local"
    (run / "results").mkdir(parents=True)
    (run / "logs").mkdir()
    app, server, client = serve()
    status, refused = client.json("GET", f"/api/results?run_dir={run}")
    assert status == 404
    projects.add(str(calc))
    status, view = client.json("GET", f"/api/results?run_dir={run}")
    assert status == 200, view


def _summarized(output, line):
    """A run directory whose summaries Testuary's own writer wrote from
    one run of calc.py."""
    from testuary import results as run_summary

    (output / "results" / "calc").mkdir(parents=True)
    (output / "logs").mkdir()
    columns = "target_number,run,file_path,line_coverage,mutation_score"
    (output / "results" / "tests.csv").write_text(f"{columns}\n1,0,calc.py,{line},80.0\n",
                                                  encoding="utf-8")
    run_summary.write_file_summaries(output / "results", "calc", 1)
    run_summary.write_project_summary(output / "results")


def test_each_project_shows_its_newest_summarized_results(serve, calc):
    app, server, client = serve()
    projects.add(str(calc))
    status, view = client.json("GET", "/api/projects")
    assert view["projects"][0]["latest"] is None
    results = calc / ".testuary" / "results"
    _summarized(results / "older", "50.0")
    _summarized(results / "local", "75.0")
    newer = results / "local" / "results" / "summary.json"
    stamp = (results / "older" / "results" / "summary.json").stat().st_mtime + 10
    os.utime(newer, (stamp, stamp))
    status, view = client.json("GET", "/api/projects")
    latest = view["projects"][0]["latest"]
    assert latest["run_dir"] == str(results / "local")
    (item,) = latest["files"]
    assert (item["file"], item["runs"], item["fully_measured_runs"]) == ("calc.py", 1, 1)
    assert item["values"]["line_coverage"]["mean"]["text"] == "75%"
    assert [value["id"] for value in latest["values"]] == [
        "line_coverage", "branch_coverage", "mutation_score"]
    status, shown = client.json("GET", f"/api/results?run_dir={quote(latest['run_dir'])}")
    assert status == 200 and shown["runs"][0]["files"][0]["runs"][0]["name"] == "r0"


def test_results_that_cannot_be_read_are_an_error_not_no_results(serve, calc, monkeypatch):
    """Only a missing run directory or summary means no results. A
    summary that is not one, or a changed Testuary, reaches the page with
    its reason."""
    app, server, client = serve()
    projects.add(str(calc))
    status, view = client.json("GET", "/api/projects")
    assert (view["projects"][0]["latest"], view["projects"][0]["latest_error"]) == (None, None)
    written = calc / ".testuary" / "results" / "local" / "results" / "summary.json"
    written.parent.mkdir(parents=True)
    written.write_text("not json", encoding="utf-8")
    status, view = client.json("GET", "/api/projects")
    entry = view["projects"][0]
    assert entry["latest"] is None and "is not a summary Testuary wrote" in entry["latest_error"]
    from testuary import results as run_summary

    monkeypatch.delattr(run_summary, "SUMMARY")
    status, view = client.json("GET", "/api/projects")
    assert "SUMMARY" in view["projects"][0]["latest_error"]


def test_a_projects_files_and_runs_are_listed_for_the_page(serve, calc):
    app, server, client = serve()
    projects.add(str(calc))
    (calc / "more.py").write_text("def more():\n    return 1\n", encoding="utf-8")
    status, files = client.json("GET", f"/api/project/files?project_dir={quote(str(calc))}")
    assert status == 200
    assert (files["language"], files["project_files"]) == ("python", ["calc.py"])
    assert files["files"] == {"python": ["calc.py", "more.py"]}
    status, runs = client.json("GET", f"/api/project/results?project_dir={quote(str(calc))}")
    assert (status, runs["runs"]) == (200, [])
    _summarized(calc / ".testuary" / "results" / "local", "75.0")
    status, runs = client.json("GET", f"/api/project/results?project_dir={quote(str(calc))}")
    (run,) = runs["runs"]
    assert (run["name"], run["files"], run["fully_measured_runs"]) == ("local", 1, 1)
    status, refused = client.json("GET", f"/api/project/files?project_dir={quote(str(calc.parent))}")
    assert status == 400 and refused["field"] == "project_dir"


def test_a_project_without_a_language_has_its_files_listed_by_language(serve, calc):
    """A project whose settings name no language runs the source files of
    every installed language: the page gets them by language."""
    app, server, client = serve()
    (calc / ".testuary" / "project.ini").write_text("[project]\n", encoding="utf-8")
    java = calc / "src" / "main" / "java" / "demo"
    java.mkdir(parents=True)
    (java / "Calc.java").write_text("package demo;\n\npublic class Calc {\n}\n", encoding="utf-8")
    projects.add(str(calc))
    status, files = client.json("GET", f"/api/project/files?project_dir={quote(str(calc))}")
    assert (status, files["language"]) == (200, "")
    assert files["files"] == {"java": ["src/main/java/demo/Calc.java"], "python": ["calc.py"]}


def test_the_settings_view_has_plain_titles_and_grays_what_a_run_does_not_use(serve, calc, user_dir):
    (user_dir / "models.ini").write_text("[local]\nprovider = ollama\nmodel = m\n", encoding="utf-8")
    app, server, client = serve()
    projects.add(str(calc))
    status, view = client.json("GET", f"/api/settings?project_dir={quote(str(calc))}")
    sections = {section["name"]: section for section in view["sections"]}
    assert sections["coverage"]["title"] == "Coverage phase"
    openai = sections["openai"]["settings"][0]
    assert (openai["active"], openai["inactive"]) == (False, "the model's provider is ollama")
    assert sections["ollama"]["settings"][0]["active"] is True
    assert sections["main"]["settings"][0]["label"][0].isupper()


def test_a_saved_run_in_the_run_dir_is_named_with_testuarys_choices(serve, calc, stand_in):
    """A second run with the same model meets the first one's run directory,
    which Testuary refuses for a new run. The page learns the directory, the
    files continuing would finish and a free directory for a fresh start, and
    a run can continue the saved one."""
    from testuary.presentation.terminal import FileCompletion, RunCompletion, write_completion

    app, server, client = serve()
    projects.add(str(calc))
    ask = lambda **values: client.json("GET", "/api/run/saved?" + "&".join(  # noqa: E731
        f"{name}={quote(value)}" for name, value in {"project_dir": str(calc), **values}.items()))[1]
    assert ask() == {"occupied": False, "run_dir": ".testuary/results/local"}
    results = calc / ".testuary" / "results" / "local" / "results"
    # Saved work without an end-of-run summary: no file to name.
    results.mkdir(parents=True)
    (results / "tests.csv").write_bytes(b"file\n")
    unread = ask()
    assert (unread["occupied"], unread["summary"], unread["unfinished"]) == (True, False, [])
    write_completion(RunCompletion(files=(
        FileCompletion("calc.py", "no_complete_suite", reason="no tests"),),
        results_directory=str(results)))
    saved = ask()
    assert saved["summary"] is True
    assert (saved["occupied"], saved["run_dir"], saved["unfinished"], saved["fresh"]) == (
        True, ".testuary/results/local", ["calc.py"], ".testuary/results/local-2")
    # Another model, or a run directory of one's own, is free.
    assert ask(model="other")["occupied"] is False
    assert ask(output_dir="elsewhere")["occupied"] is False
    stand_in.setenv("STAND_IN_MODE", "exit0")
    status, job = client.json("POST", "/api/run", {"project_dir": str(calc), "resume": True})
    assert status == 201 and job["arguments"][-1] == "--resume"
    assert wait_for(lambda: client.json("GET", f"/api/jobs/{job['id']}")[1]["status"] != "running")
    status, refused = client.json("POST", "/api/run", {"project_dir": str(calc), "resume": "yes"})
    assert (status, refused["field"]) == (400, "resume")


def _missing_items(items):
    """missingItem as the page runs it, evaluated by Node, for each item."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    function = re.search(r"^function missingItem\(text\) \{\n.*?^\}\n", source, re.S | re.M)
    assert function, "app.js defines missingItem"
    script = (function.group(0) + "process.stdout.write(JSON.stringify("
              + json.dumps(items) + ".map(missingItem)));")
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def _settings_dirs(cases):
    """settingsDir as the page runs it, evaluated by Node, for each
    (address query, listed projects) case."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    function = re.search(r"^function settingsDir\(query, projects\) \{\n.*?^\}\n", source, re.S | re.M)
    assert function, "app.js defines settingsDir"
    script = (function.group(0) + "process.stdout.write(JSON.stringify(" + json.dumps(cases)
              + ".map(([query, projects]) => settingsDir(new URLSearchParams(query), projects))));")
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_settings_page_opens_on_the_working_dirs_project():
    """Without a project in its address the Settings page opens on the
    working directory's project. The global settings stay one choice away, and
    a directory of none opens on them."""
    working = {"project_dir": "C:\\parent\\tenacity", "working": True}
    other = {"project_dir": "C:\\elsewhere", "working": False}
    assert _settings_dirs([
        ["", [working, other]],
        ["project_dir=C%3A%5Celsewhere", [working, other]],
        ["project_dir=C%3A%5Cparent%5Ctenacity&scope=global", [working, other]],
        ["", [other]],
        ["", []],
    ]) == ["C:\\parent\\tenacity", "C:\\elsewhere", "C:\\parent\\tenacity", "", ""]


def test_an_open_answer_without_a_chosen_path_asks_for_one():
    """A setup question with paths to choose from and another path to type
    (the project's interpreter, a JDK) chooses none when the setup found no
    default. Next then asks for a path on the answer's field instead of
    failing in the page's script."""
    from conftest import page_value

    unchosen = page_value(["openAnswer"], "(() => { try { return openAnswer(null, { value: '' }); }"
                                          " catch (error) { return error; } })()")
    assert unchosen == {"field": "answer", "message": "Choose one of the paths above, or another path."}
    assert page_value(["openAnswer"], "[openAnswer({ value: 'C:/env/python.exe' }, { value: '' }),"
                                      " openAnswer({ value: '' }, { value: 'D:/other/python.exe' })]") == [
        "C:/env/python.exe", "D:/other/python.exe"]


def test_a_missing_item_keeps_its_path_and_drops_the_console_command():
    """The setup's list of what is still missing ends some items with the
    console command that adds it. The page drops that command and keeps the
    rest whole, a Windows path with its drive letter included."""
    assert _missing_items([
        r"install coverage, cosmic-ray in C:\venv\Scripts\python.exe",
        "a model alias: run testuary setup model",
        "this machine's workers (defaults apply): run testuary setup machine",
        r"a pom.xml or build.gradle in D:\work\calc",
    ]) == [
        r"install coverage, cosmic-ray in C:\venv\Scripts\python.exe",
        "a model alias",
        "this machine's workers (defaults apply)",
        r"a pom.xml or build.gradle in D:\work\calc",
    ]


def _project(tmp_path):
    project_dir = tmp_path / "calcproj"
    (project_dir / "calc").mkdir(parents=True)
    (project_dir / "calc" / "ops.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (project_dir / "calc" / "io.py").write_text("def show(x):\n    return str(x)\n", encoding="utf-8")
    return project_dir


def test_the_setup_asks_testuarys_questions_and_writes_its_file(serve, tmp_path, user_dir):
    (user_dir / "models.ini").write_text("[local]\nprovider = ollama\nmodel = qwen3:8b\n",
                                            encoding="utf-8")
    project_dir = _project(tmp_path)
    app, server, client = serve()
    answers = {}
    asked = []
    while True:
        status, view = client.json("POST", "/api/project/setup",
                                   {"project_dir": str(project_dir), "answers": answers})
        assert status == 200, view
        if "question" not in view:
            break
        question = view["question"]
        assert view["problem"] is None, view
        asked.append(question["key"])
        if question["key"] == "project.language":
            assert question["default"] == "python" and "python" in question["choices"]
        if question["key"] == "python.interpreter":
            # What the setup found, and another path besides.
            assert question["open"] and sys.executable in question["choices"]
        if question["key"] == "project.confirm":
            # The settings it is about to write, by Testuary's plain names.
            pending = {item["name"]: (item["label"], item["value"]) for item in view["pending"]}
            assert pending["language"] == ("Language", "python")
            assert pending["source_files"] == ("Source files", "calc/ops.py")
            assert pending["model"] == ("Model", "local")
        if question["key"] == "targets":
            # The files the question numbers, as Testuary lists them.
            assert view["files"] == ["calc/io.py", "calc/ops.py"]
        answers[question["key"]] = {"python.interpreter": sys.executable, "targets": "2",
                                    "model": "local"}.get(question["key"], "")
    assert asked == ["project.language", "python.import_root", "python.interpreter", "targets",
                     "model", "project.confirm"]
    written = project_dir / ".testuary" / "project.ini"
    assert view["written"] == [str(written)]
    parser = configparser.ConfigParser()
    parser.read(written, encoding="utf-8")
    assert dict(parser["project"])["source_files"] == "calc/ops.py"
    assert parser["project"]["language"] == "python"
    assert parser["api"]["model"] == "local"
    recent = json.loads((user_dir / "web" / "projects.json").read_text(encoding="utf-8"))
    assert recent["project_dirs"] == [str(project_dir.resolve())]


def test_the_page_sends_the_setup_the_body_the_server_reads(serve, tmp_path, user_dir):
    """Add a project's setup step, as app.js runs it, sends a body the
    server's setup takes: the server asks its first question and refuses
    nothing."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    function = re.search(r"^async function setupStep\(.*?^\}\n", source, re.S | re.M)
    assert function, "app.js defines setupStep"
    project_dir = _project(tmp_path)
    script = (
        "let sent = null;\n"
        "async function call(path, body) { sent = { path, body }; throw new Error('sent'); }\n"
        "function showError() {}\nfunction el() { return {}; }\n"
        + function.group(0)
        + f"setupStep({{ replaceChildren() {{}} }}, {{ isConnected: false }}, {json.dumps(str(project_dir))}, {{}})"
        ".then(() => process.stdout.write(JSON.stringify(sent)));")
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    sent = json.loads(done.stdout)
    app, server, client = serve()
    status, view = client.json("POST", sent["path"], sent["body"])
    assert status == 200, view
    assert view["question"]["key"] == "project.language" and view["problem"] is None


SETUP_STUBS = """
function el(tag, props = {}, ...children) {
  return { tag, text: props.text || "", role: props.role, disabled: false, textContent: props.text || "",
    handlers: {}, children: children.flat(Infinity).filter(Boolean),
    append(...parts) { this.children.push(...parts.flat(Infinity).filter(Boolean)); },
    addEventListener(type, handler) { this.handlers[type] = handler; }, querySelector() { return null; },
    focus() {} };
}
function find(node, test) {
  if (test(node)) return node;
  for (const child of node.children || []) {
    const found = typeof child === "object" && find(child, test);
    if (found) return found;
  }
  return null;
}
const box = { replaceChildren(shown) { this.shown = shown; } };
const replies = [{ said: [], project_dir: "/work/calc", problem: null, missing: [],
  question: { key: "targets", prompt: "Which files?", choices: [] } }];
let release;
function call() {
  return replies.length ? Promise.resolve(replies.shift()) : new Promise((resolve) => { release = resolve; });
}
function setupInput() { return [null, () => "1"]; }
function showError() {}
function render() {}
"""


def test_a_setup_steps_buttons_wait_while_testuarys_setup_answers():
    """While a setup step's answer goes to Testuary's setup, Next and Back
    are disabled and the step says so, so that no second step starts and no
    later answer replaces the form. They work again after the answer."""
    from conftest import page_value

    states = page_value(["askingSetup", "setupStep"], """(async () => {
  await setupStep(box, { isConnected: false }, "/work/calc", {});
  const form = box.shown;
  const next = find(form, (node) => node.tag === "button" && node.text === "Next");
  const back = find(form, (node) => node.tag === "button" && node.text === "Back");
  const busy = find(form, (node) => node.role === "status");
  const waiting = form.handlers.submit({ preventDefault() {} });
  const during = [next.disabled, back.disabled, busy.textContent];
  release({ said: [], project_dir: "/work/calc", refused: "stopped" });
  await waiting;
  return [during, [next.disabled, back.disabled, busy.textContent]];
})()""", before=SETUP_STUBS)
    assert states == [[True, True, "Asking Testuary's setup."], [False, False, ""]]

def test_a_refused_answer_comes_back_with_testuarys_reason(serve, tmp_path, user_dir):
    project_dir = _project(tmp_path)
    app, server, client = serve()
    status, view = client.json("POST", "/api/project/setup", {
        "project_dir": str(project_dir), "answers": {"project.language": "no_such_language"}})
    assert status == 200
    assert view["question"]["key"] == "project.language"
    assert view["problem"].startswith("choose one of")
    assert not (project_dir / ".testuary").exists()
    assert not (user_dir / "web" / "projects.json").exists()


def test_the_setup_takes_only_an_existing_local_dir_and_text_answers(serve, tmp_path):
    project_dir = _project(tmp_path)
    app, server, client = serve()
    for body in ({"project_dir": str(tmp_path / "missing")}, {"project_dir": "relative"},
                 {"project_dir": str(project_dir), "answers": {"targets": 2}},
                 {"project_dir": str(project_dir), "answers": ["targets"]}):
        status, refused = client.json("POST", "/api/project/setup", body)
        assert status == 400, (body, refused)


def _delivered(calc):
    """A run of calc.py in the project's default run directory whose suite directory
    holds the delivered test file, the measurement helper and a record."""
    output = calc / ".testuary" / "results" / "local"
    _summarized(output, "75.0")
    run = output / "results" / "calc" / "r0"
    suite = run / "tests"
    suite.mkdir(parents=True)
    (suite / "test_calc.py").write_bytes(b"from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n")
    (suite / "conftest.py").write_bytes(b"# the measurement helper\n")
    (suite / "kill_matrix.json").write_bytes(b"{}")
    return output, suite


def test_a_listed_projects_run_names_its_project_for_the_settings_link(serve, calc):
    """A run of a listed project names that project, wherever its Testuary
    directory is and whatever [main] testuary_dir names it, and the page's
    settings link opens that project's settings. A run in the working
    directory outside every listed project names none and links the
    settings page."""
    from conftest import page_value

    app, server, client = serve()
    projects.add(str(calc))
    output = calc / "runs" / "first"
    _summarized(output, "75.0")
    status, view = client.json("GET", f"/api/results?run_dir={quote(str(output))}")
    assert status == 200 and view["project_dir"] == str(calc.resolve())
    other = app.base_dir / "elsewhere"
    _summarized(other, "75.0")
    status, outside = client.json("GET", f"/api/results?run_dir={quote(str(other))}")
    assert status == 200 and outside["project_dir"] is None
    links = page_value(["settingsOf"], f"[settingsOf({json.dumps(view['project_dir'])}), settingsOf(null)]")
    assert links == [f"#/settings?project_dir={quote(str(calc.resolve()), safe='')}", "#/settings"]


def test_a_jobs_warnings_link_its_projects_settings_or_the_projects_of_its_run():
    """An ended job's warnings link the settings of the job's project, and
    for a job without one, such as an extension's, the settings of the
    listed project its run directory belongs to, as /api/results names it."""
    from conftest import page_value

    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    closure = re.search(r"^  const showWarnings = async \(\) => \{\n.*?^  \};\n", source, re.S | re.M)
    assert closure, "the job page defines showWarnings"
    stubs = """
const current = { ended: 1, run_dir: "/work/runs/r1", project: null };
async function call() {
  return { project_dir: "/work/calc", runs: [{ files: [{ name: "calc.py", runs: [{ name: "r0", warnings: [{}] }] }] }] };
}
function el() { return {}; }
const links = [];
function warningList(warnings, settings) { links.push(settings); return {}; }
const warningsBox = { replaceChildren() {} };
"""
    links = page_value(["settingsOf"], "(async () => {\n" + closure.group(0)
                       + "await showWarnings(); current.project = '/work/own'; await showWarnings(); return links; })()",
                       before=stubs)
    assert links == ["#/settings?project_dir=%2Fwork%2Fcalc", "#/settings?project_dir=%2Fwork%2Fown"]

def test_a_runs_suite_names_its_delivered_test_file_and_its_setup_file(serve, calc):
    """The results view names the suite's test file, by the language's own
    rule for the file Testuary delivers, without the measurement helper,
    and the setup file beside it."""
    app, server, client = serve()
    projects.add(str(calc))
    output, suite = _delivered(calc)
    status, view = client.json("GET", f"/api/results?run_dir={quote(str(output))}")
    item = view["runs"][0]["files"][0]["runs"][0]
    assert status == 200, view
    assert item["suite"]["tests"] == ["results/calc/r0/tests/test_calc.py"]
    # The measurement helper beside it is the language's setup file.
    assert item["suite"]["setup"] == ["results/calc/r0/tests/conftest.py"]


def test_the_delivered_test_file_is_copied_into_the_project_where_its_tests_are(serve, calc):
    """Copying offers the directory of the project's own tests, writes the
    delivered test file there as it is, and never writes over a file, a
    file other than the delivered test, outside the project or into its
    Testuary directory."""
    app, server, client = serve()
    projects.add(str(calc))
    output, suite = _delivered(calc)
    (calc / "tests").mkdir()
    (calc / "tests" / "test_other.py").write_bytes(b"def test_x():\n    pass\n")
    path = "results/calc/r0/tests/test_calc.py"
    query = f"run_dir={quote(str(output))}&path={quote(path)}"
    status, offer = client.json("GET", f"/api/results/adopt?{query}")
    assert (status, offer["test_dir"], offer["file"]) == (200, str(calc / "tests"), "test_calc.py")
    status, done = client.json("POST", "/api/results/adopt", {"run_dir": str(output), "path": path, "test_dir": "tests"})
    assert status == 200, done
    assert done["written"] == str(calc / "tests" / "test_calc.py")
    assert (calc / "tests" / "test_calc.py").read_bytes() == (suite / "test_calc.py").read_bytes()
    (calc / "tests" / "test_calc.py").write_bytes(b"# changed by the person\n")
    for body, field, words in (
            ({"path": path, "test_dir": "tests"}, "test_dir", "exists already"),
            ({"path": "results/calc/r0/tests/kill_matrix.json", "test_dir": "tests"}, None,
             "not a delivered test file"),
            ({"path": path, "test_dir": str(calc.parent)}, "test_dir", "inside the project"),
            ({"path": path, "test_dir": ".testuary/mine"}, "test_dir", "Testuary directory")):
        status, refused = client.json("POST", "/api/results/adopt", {"run_dir": str(output), **body})
        assert status == 400 and words in refused["error"] and refused["field"] == field, refused
    assert (calc / "tests" / "test_calc.py").read_bytes() == b"# changed by the person\n"
    assert not (calc / ".testuary" / "mine").exists()
    # A new directory in the project is made.
    status, done = client.json("POST", "/api/results/adopt", {"run_dir": str(output), "path": path,
                                                              "test_dir": "tests/new"})
    assert status == 200 and (calc / "tests" / "new" / "test_calc.py").is_file()


def test_the_setup_file_beside_the_test_file_is_copied_where_the_test_file_goes(serve, calc):
    """A run's tests can need the setup file beside them (the measurement
    helper, conftest.py): it is offered the test file's directory and copied
    there as it is, never over the project's own."""
    app, server, client = serve()
    projects.add(str(calc))
    output, suite = _delivered(calc)
    (calc / "tests").mkdir()
    (calc / "tests" / "test_other.py").write_bytes(b"def test_x():\n    pass\n")
    path = "results/calc/r0/tests/conftest.py"
    status, offer = client.json("GET", f"/api/results/adopt?run_dir={quote(str(output))}&path={quote(path)}")
    assert (status, offer["test_dir"], offer["file"]) == (200, str(calc / "tests"), "conftest.py")
    status, done = client.json("POST", "/api/results/adopt", {"run_dir": str(output), "path": path, "test_dir": "tests"})
    assert status == 200 and (calc / "tests" / "conftest.py").read_bytes() == (suite / "conftest.py").read_bytes()
    status, refused = client.json("POST", "/api/results/adopt", {"run_dir": str(output), "path": path, "test_dir": "tests"})
    assert status == 400 and "exists already" in refused["error"], refused


def test_a_project_without_tests_is_offered_its_own_dir(serve, calc):
    app, server, client = serve()
    projects.add(str(calc))
    output, suite = _delivered(calc)
    path = "results/calc/r0/tests/test_calc.py"
    status, offer = client.json("GET", f"/api/results/adopt?run_dir={quote(str(output))}&path={quote(path)}")
    assert (status, offer["test_dir"]) == (200, str(calc))


def test_a_run_of_a_project_below_the_working_dir_goes_into_that_project(serve, tmp_path):
    """Started in a directory that belongs to no project (one that holds
    several projects), the results view reads a run of a project listed
    below it with that project's settings, and its test file is copied
    into that project, as
    for every listed project."""
    parent = tmp_path / "parent"
    tenacity = _with_settings(parent / "tenacity")
    app, server, client = serve(parent)
    output, suite = _delivered(tenacity)
    assert app.results_base(str(output)) == tenacity.resolve()
    path = "results/calc/r0/tests/test_calc.py"
    status, offer = client.json("GET", f"/api/results/adopt?run_dir={quote(str(output))}&path={quote(path)}")
    assert (status, offer.get("test_dir")) == (200, str(tenacity.resolve())), offer
    status, done = client.json("POST", "/api/results/adopt", {"run_dir": str(output), "path": path, "test_dir": "tests"})
    assert status == 200, done
    assert (tenacity / "tests" / "test_calc.py").read_bytes() == (suite / "test_calc.py").read_bytes()


# Adding a project from a GitHub HTTPS address: git clones it into a
# new directory named after it in the directory the user chooses, then the usual
# setup follows on the page.

def _cloned(monkeypatch, *, returncode=0, stderr=""):
    calls = []

    def clone(address, destination):
        calls.append((address, destination))
        if returncode == 0:
            (destination / "src").mkdir(parents=True)
        return subprocess.CompletedProcess(["git"], returncode, "", stderr)

    monkeypatch.setattr(projects, "_git", lambda: "git")
    monkeypatch.setattr(projects, "_git_clone", clone)
    return calls


@pytest.mark.parametrize("address", ["https://github.com/jd/tenacity", "https://github.com/jd/tenacity.git",
                                     "https://github.com/jd/tenacity/", " https://github.com/jd/tenacity "])
def test_a_github_address_is_cloned_into_a_new_dir(serve, tmp_path, monkeypatch, address):
    calls = _cloned(monkeypatch)
    parent = tmp_path / "work"
    parent.mkdir()
    app, server, client = serve()
    status, made = client.json("POST", "/api/projects/clone", {"url": address, "parent_dir": str(parent)})
    assert status == 200 and made["project_dir"] == str((parent / "tenacity").resolve())
    assert calls == [("https://github.com/jd/tenacity.git", parent.resolve() / "tenacity")]


@pytest.mark.parametrize("address", [
    "http://github.com/jd/tenacity", "git@github.com:jd/tenacity.git", "ssh://git@github.com/jd/tenacity.git",
    "git://github.com/jd/tenacity.git", "file:///C:/repo", "https://gitlab.com/jd/tenacity",
    "ext::sh -c touch% x", "https://github.com/jd", "https://github.com/jd/tenacity/tree/main",
    "https://github.com/jd/..", "https://user:pw@github.com/jd/tenacity", "", None])
def test_only_a_github_https_address_is_cloned(serve, tmp_path, monkeypatch, address):
    calls = _cloned(monkeypatch)
    parent = tmp_path / "work"
    parent.mkdir()
    app, server, client = serve()
    status, refused = client.json("POST", "/api/projects/clone", {"url": address, "parent_dir": str(parent)})
    assert status == 400 and refused["field"] == "url" and "https://github.com/" in refused["error"]
    assert calls == [] and list(parent.iterdir()) == []


def test_cloning_without_git_says_so(serve, tmp_path, monkeypatch):
    calls = _cloned(monkeypatch)
    monkeypatch.setattr(projects, "_git", lambda: None)
    app, server, client = serve()
    status, refused = client.json("POST", "/api/projects/clone",
                                  {"url": "https://github.com/jd/tenacity", "parent_dir": str(tmp_path)})
    assert status == 400 and refused["field"] == "url"
    assert "needs git" in refused["error"] and "git-scm.com" in refused["error"]
    assert calls == []


def test_a_clone_never_goes_into_a_dir_that_exists(serve, tmp_path, monkeypatch):
    calls = _cloned(monkeypatch)
    (tmp_path / "tenacity").mkdir()
    app, server, client = serve()
    status, refused = client.json("POST", "/api/projects/clone",
                                  {"url": "https://github.com/jd/tenacity", "parent_dir": str(tmp_path)})
    assert status == 400 and refused["field"] == "parent_dir" and "exists" in refused["error"]
    status, refused = client.json("POST", "/api/projects/clone",
                                  {"url": "https://github.com/jd/tenacity", "parent_dir": "relative"})
    assert status == 400 and refused["field"] == "project_dir"
    assert calls == []


def test_a_failed_clone_says_what_git_said(serve, tmp_path, monkeypatch):
    _cloned(monkeypatch, returncode=128, stderr="Cloning into 'x'...\nfatal: repository not found\n")
    app, server, client = serve()
    status, refused = client.json("POST", "/api/projects/clone",
                                  {"url": "https://github.com/jd/nothing", "parent_dir": str(tmp_path)})
    assert status == 400 and refused["field"] == "url" and "fatal: repository not found" in refused["error"]


def test_git_clones_over_https_only_and_asks_for_nothing(tmp_path):
    command, environment = projects._clone_command("https://github.com/jd/tenacity.git", tmp_path / "t")
    assert command[1:] == ["-c", "protocol.allow=never", "-c", "protocol.https.allow=always", "clone",
                           "--no-recurse-submodules", "--", "https://github.com/jd/tenacity.git",
                           str(tmp_path / "t")]
    assert environment["GIT_TERMINAL_PROMPT"] == "0" and environment["GIT_ALLOW_PROTOCOL"] == "https"


def test_the_add_form_clones_a_github_address_then_sets_the_clone_up():
    """The Add a project form takes an optional GitHub address: with one, it
    clones into the directory named below it and the setup follows on the new
    directory. Without, the setup takes the directory itself."""
    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    start = source.index("function addProjectForm(state) {")
    form = source[start:source.index("\n}\n", start)]
    assert 'field({ id: "url", label: "GitHub address (optional)"' in form
    clone = form.index('call("/api/projects/clone", { url, parent_dir: chosen })')
    assert clone < form.index("setupStep(box, form, chosen, {})")


def test_the_project_scope_refuses_an_address_beside_its_field(serve, calc):
    """The Settings page refuses [openai] api_url in a project's file, which
    would decide where the model's key goes, in the run's words beside the
    address's own field."""
    app, server, client = serve()
    projects.add(str(calc))
    before = (calc / ".testuary" / "project.ini").read_bytes()
    status, refused = client.json("POST", "/api/settings", {
        "scope": "project", "project_dir": str(calc), "section": "openai",
        "values": {"api_url": "http://127.0.0.1:9/v1"}})
    assert status == 400
    assert refused["error"].endswith("[openai] api_url decides where the model's key goes. Set the "
                                     "address in your global configuration, models.ini, the "
                                     "environment or on the command line")
    assert refused["field"] == "api_url"
    assert (calc / ".testuary" / "project.ini").read_bytes() == before
