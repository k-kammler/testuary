"""The results views read documented files by name and show a run's table,
its delivered suites and every other file of a run or demonstration. They
show nothing else: no directory outside the working directory, nothing below a
link or junction and no Testuary configuration file, so a planted key
reaches no view."""
from __future__ import annotations

import json
import os
import re
import sys
from urllib.parse import quote

import pytest

from testuary.web import _results

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

KEY = "canary-key-0123456789"


@pytest.fixture(autouse=True)
def no_machine_settings(monkeypatch):
    """The views resolve the working directory's settings through Testuary, which
    reads this process's TESTUARY_ variables. The machine's are left out."""
    for name in list(os.environ):
        if name.startswith("TESTUARY_"):
            monkeypatch.delenv(name)


def _run_dir(root, *, demo=False):
    run = root / "run" if demo else root
    suite = run / "results" / "calc_ops" / "r0" / "tests"
    suite.mkdir(parents=True)
    (run / "results" / "calc_ops" / "r0" / "archive" / "coverage").mkdir(parents=True)
    table = ("\ufefftarget_number,run,file_path,line_coverage,new_column,mutation_score\r\n"
             "1,0,calc/ops.py,96.5,x,\r\n")
    (run / "results" / "tests.csv").write_bytes(table.encode("utf-8"))
    (run / "results" / "calc_ops" / "phase_state.json").write_text(
        json.dumps({"target_number": 1, "target_file_path": "calc/ops.py"}), encoding="utf-8")
    (suite / "test_ops.py").write_bytes(b"def test_add():\n    assert 1 + 1 == 2\n")
    (run / "logs").mkdir()
    (run / "logs" / "run.log").write_bytes(b"model fake_alias answered\n")
    if demo:
        (root / "DEMO.json").write_text(json.dumps({"statement": "DEMO", "result": {
            "outcome": "completed", "answers": 3}}), encoding="utf-8")
    return root


def _plant(path):
    """A model catalogue at ``path`` that holds the planted key."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(("[fake_alias]\nprovider = openai\nmodel = fake-model\n"
                      f"api_key = {KEY}\n").encode("ascii"))
    return path


def _suites(view):
    """The files of every suite directory the dashboard lists."""
    return [listed["path"] for run in view["runs"] for file in run["files"]
            for item in file["runs"] if item["suite"]
            for listed in item["suite"]["files"]]


def _every_view(run_dir):
    """The directory's summary and every file it lists, read, as one text."""
    view = _results.summary(run_dir)
    listed = [item["path"] for item in view["files"]] + _suites(view)
    texts = [json.dumps(view)] + [json.dumps(_results.read(run_dir, path)) for path in listed]
    return view, "\n".join(texts)


def test_the_table_is_read_by_column_name(tmp_path):
    run_dir = _results.locate(tmp_path, str(_run_dir(tmp_path / "out")))
    view = _results.summary(run_dir)
    assert view["demo"] is False and "demo_record" not in view
    assert view["rows"] == [{"target_number": "1", "run": "0", "file_path": "calc/ops.py",
                             "line_coverage": "96.5", "new_column": "x", "mutation_score": ""}]
    (run,) = view["runs"][0]["files"][0]["runs"]
    assert run["suite"] == {"suite_dir": "results/calc_ops/r0/tests", "files": [
        {"path": "results/calc_ops/r0/tests/test_ops.py", "size": 38}],
        "tests": ["results/calc_ops/r0/tests/test_ops.py"], "setup": []}
    assert view["log"] == "logs/run.log"


def test_the_page_reads_only_keys_a_summarys_run_holds(tmp_path):
    """The keys of a summary's run that the page reads are keys the server
    sends: a settings link built from another name opens the wrong scope."""
    from conftest import SOURCE

    run_dir = _results.locate(tmp_path, str(_run_dir(tmp_path / "out")))
    run = _results.summary(run_dir)["runs"][0]
    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    read = set(re.findall(r"\bsummary\.runs\[0\]\.(\w+)", source))
    assert read and read - set(run) == set()


def test_the_page_reads_every_part_of_a_files_details():
    """The parts of a file's details the server sends are those the page
    reads: a part under another name shows "none" in every row."""
    from conftest import SOURCE

    sent = _results._details({"phases": {"starter": {"line_coverage": 90.0}},
                             "finishing": {"line_coverage": 95.0}, "starter": {}, "time": {}})
    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert set(re.findall(r"\bdetails\.(\w+)", source)) == set(sent)


def test_a_run_shows_every_file_logs_included(tmp_path):
    _run_dir(tmp_path / "out")
    run_dir = _results.locate(tmp_path, "out")
    assert _results.read(run_dir, "results/calc_ops/r0/tests/test_ops.py")["text"].startswith("def test")
    assert _results.read(run_dir, "logs/run.log")["text"] == "model fake_alias answered\n"
    assert {item["path"] for item in _results.summary(run_dir)["files"]} >= {
        "logs/run.log", "results/tests.csv"}


def test_a_demonstration_shows_every_file(tmp_path):
    root = _run_dir(tmp_path / "demo", demo=True)
    for opened in (root, root / "run", root / "run" / "results"):
        run_dir = _results.locate(tmp_path, str(opened))
        assert run_dir.root == root and run_dir.demo
    view = _results.summary(run_dir)
    assert view["demo_record"]["result"]["outcome"] == "completed"
    assert view["log"] == "run/logs/run.log"
    assert "run/logs/run.log" in {item["path"] for item in view["files"]}
    assert _results.read(run_dir, "run/logs/run.log")["text"] == "model fake_alias answered\n"


def test_nothing_outside_the_dir_is_read(tmp_path):
    root = _run_dir(tmp_path / "demo", demo=True)
    (tmp_path / "secret.txt").write_text("outside", encoding="utf-8")
    run_dir = _results.locate(tmp_path, str(root))
    for path in ("../secret.txt", str(tmp_path / "secret.txt"), "run/../../secret.txt",
                 "run\\..\\..\\secret.txt", "", "missing.txt", "run", "run/", "./DEMO.json"):
        with pytest.raises(_results.NotShown, match="no such file"):
            _results.read(run_dir, path)


def test_other_dirs_are_not_results(tmp_path):
    (tmp_path / "plain").mkdir()
    (tmp_path / "half" / "results").mkdir(parents=True)
    for name in ("plain", "half", "missing", ""):
        with pytest.raises(_results.NotShown):
            _results.locate(tmp_path, name)


def test_a_dir_that_does_not_exist_is_named_so(tmp_path):
    """A run directory that does not exist, such as a run's that ended
    before making it, says so, not that a link may be on the way."""
    (tmp_path / "project" / ".testuary" / "results").mkdir(parents=True)
    named = str(tmp_path / "project" / ".testuary" / "results" / "fake-2")
    with pytest.raises(_results.NotShown) as refused:
        _results.locate(tmp_path, named)
    assert str(refused.value) == f"The directory {named} does not exist."
    (tmp_path / "file.txt").write_bytes(b"")
    with pytest.raises(_results.NotShown, match="is not a directory"):
        _results.locate(tmp_path, "file.txt")


def test_only_dirs_below_the_working_dir_are_opened(tmp_path, monkeypatch):
    base = tmp_path / "work"
    _plant(base / "models.ini")
    _run_dir(base)
    _run_dir(tmp_path / "outside")

    def opened(*args, **kwargs):
        raise AssertionError("a refused path was opened")

    # Each path is refused as text, before anything is opened: a network
    # path would contact its host.
    monkeypatch.setattr(_results, "_child", opened)
    texts = [".", "..", "../outside", "out/../..", str(base), str(tmp_path / "outside"),
             str(base / ".." / "outside"), "//server/share/out"]
    if os.name == "nt":
        texts += ["\\\\server\\share\\out", "\\\\?\\" + str(base / "out"),
                  "\\\\.\\pipe\\out", base.drive + "out", "\\out"]
    for text in texts:
        with pytest.raises(_results.NotShown, match="below the working directory or a listed project's directory"):
            _results.locate(base, text)


def test_no_configuration_file_is_listed_or_read(tmp_path):
    base = tmp_path / "work"
    _plant(base / "models.ini")
    root = _run_dir(base / "out")
    planted = ["models.ini", "results/MODELS.INI", "logs/projects.ini",
               "results/calc_ops/r0/tests/config.ini", "results/calc_ops/r0/tests/Models.Ini"]
    for relative in planted:
        _plant(root / relative)
    run_dir = _results.locate(base, "out")
    view, shown = _every_view(run_dir)
    assert KEY not in shown
    names = {item["path"].rsplit("/", 1)[-1].casefold() for item in view["files"]}
    assert not names & _results.CONFIGURATION_NAMES
    assert _suites(view) == ["results/calc_ops/r0/tests/test_ops.py"]
    for relative in planted:
        with pytest.raises(_results.NotShown, match="configuration"):
            _results.read(run_dir, relative)


def test_testuarys_own_rule_leaves_its_configuration_out_under_any_name(tmp_path, monkeypatch,
                                                                       user_dir):
    base = tmp_path / "work"
    catalogue = _plant(user_dir / "models.ini")
    root = _run_dir(base / "out")
    (root / "logs" / "settings.ini").write_bytes(f"[main]\n# {KEY}\n".encode("ascii"))
    monkeypatch.setenv("TESTUARY_CONFIG_FILE", str(root / "logs" / "settings.ini"))
    hidden = ["logs/settings.ini"]
    try:
        os.link(catalogue, root / "logs" / "catalogue.txt")
        hidden.append("logs/catalogue.txt")
    except OSError:
        pass
    run_dir = _results.locate(base, "out")
    view, shown = _every_view(run_dir)
    assert KEY not in shown
    for relative in hidden:
        assert relative not in {item["path"] for item in view["files"]}
        with pytest.raises(_results.NotShown, match="configuration"):
            _results.read(run_dir, relative)
    assert "logs/run.log" in {item["path"] for item in view["files"]}


def test_unreadable_settings_close_the_views(tmp_path, user_dir):
    base = tmp_path / "work"
    _run_dir(base / "out")
    (user_dir / "config.ini").write_bytes(b"[no_such_section]\nkey = 1\n")
    with pytest.raises(_results.NotShown, match="settings"):
        _results.locate(base, "out")


def test_a_demonstrations_own_catalogue_is_not_shown(tmp_path):
    base = tmp_path / "work"
    root = _run_dir(base / "demo", demo=True)
    _plant(root / "workspace" / "models.ini")
    (root / "workspace" / "preset.ini").write_bytes(b"[main]\n")
    run_dir = _results.locate(base, "demo")
    view, shown = _every_view(run_dir)
    assert KEY not in shown
    paths = {item["path"] for item in view["files"]}
    assert "workspace/preset.ini" in paths and "workspace/models.ini" not in paths
    with pytest.raises(_results.NotShown, match="configuration"):
        _results.read(run_dir, "workspace/models.ini")


def test_no_link_or_junction_is_followed(tmp_path):
    base = tmp_path / "work"
    outside = _run_dir(tmp_path / "outside")
    _plant(outside / "logs" / "secret.txt")
    _plant(base / "models.ini")
    root = _run_dir(base / "out")
    made = []

    def link(target, path, directory=True):
        try:
            os.symlink(target, path, target_is_directory=directory)
        except OSError:
            return
        made.append(path)

    link(outside, base / "linked-run")
    link(base, root / "logs" / "up")
    link(base / "models.ini", root / "logs" / "catalogue.txt", directory=False)
    link(outside / "logs" / "secret.txt", root / "logs" / "secret.txt", directory=False)
    if os.name == "nt":
        import _winapi

        for target, path in ((outside, base / "junction-run"), (base, root / "logs" / "junction-up")):
            _winapi.CreateJunction(str(target), str(path))
            made.append(path)
    if not made:
        pytest.skip("this system lets the test create neither links nor junctions")
    for path in made:
        if path.parent == base:
            with pytest.raises(_results.NotShown, match="link"):
                _results.locate(base, path.name)
    run_dir = _results.locate(base, "out")
    view, shown = _every_view(run_dir)
    assert KEY not in shown
    listed = {item["path"] for item in view["files"]}
    for path in made:
        if path.parent != base:
            relative = path.relative_to(root).as_posix()
            assert not {item for item in listed if item == relative
                        or item.startswith(relative + "/")}
            for name in (relative, relative + "/logs/secret.txt", relative + "/run.log"):
                with pytest.raises(_results.NotShown, match="no such file"):
                    _results.read(run_dir, name)


def test_binary_and_large_files_are_named_not_shown(tmp_path, monkeypatch):
    root = _run_dir(tmp_path / "demo", demo=True)
    (root / "data.gz").write_bytes(b"\x1f\x8b\x00\x01")
    run_dir = _results.locate(tmp_path, str(root))
    assert _results.read(run_dir, "data.gz") == {"path": "data.gz", "size": 4, "text": None,
                                               "note": "This file is not text."}
    monkeypatch.setattr(_results, "MAX_TEXT_BYTES", 10)
    assert _results.read(run_dir, "run/logs/run.log")["note"] == "This file is too large to show here."


def test_the_interface_serves_the_views(serve, tmp_path):
    app, server, client = serve(tmp_path / "work")
    _run_dir(tmp_path / "work" / "out")
    status, view = client.json("GET", "/api/results?run_dir=out")
    assert status == 200 and view["rows"][0]["line_coverage"] == "96.5"
    status, file = client.json("GET", "/api/results/file?run_dir=out&path=results/calc_ops/r0/tests/test_ops.py")
    assert status == 200 and "assert 1 + 1 == 2" in file["text"]
    status, log = client.json("GET", "/api/results/file?run_dir=out&path=logs/run.log")
    assert status == 200 and log["text"] == "model fake_alias answered\n"
    status, refused = client.json("GET", "/api/results/file?run_dir=out&path=../outside.txt")
    assert status == 404 and "no such file" in refused["error"]
    assert client.json("GET", "/api/results?run_dir=nowhere")[0] == 404


def test_the_interface_serves_no_key(serve, tmp_path):
    base = tmp_path / "work"
    _plant(base / "models.ini")
    _plant(base / "projects.ini")
    _run_dir(base)
    root = _run_dir(base / "out")
    _plant(root / "models.ini")
    demo = _run_dir(base / "demo", demo=True)
    _plant(demo / "workspace" / "models.ini")
    app, server, client = serve(base)
    bodies = []
    for directory in (".", "", "..", str(base), "out", str(root), "out/results", "demo",
                      "demo/run", str(demo / "workspace")):
        bodies.append(client.request("GET", f"/api/results?run_dir={quote(directory)}")[2])
        for path in ("models.ini", "MODELS.INI", "projects.ini", "../models.ini",
                     "workspace/models.ini", "run/../models.ini", str(base / "models.ini"),
                     "logs/../../models.ini", "models.ini::$DATA", "models.ini.", "MODELS~1.INI"):
            bodies.append(client.request(
                "GET", f"/api/results/file?run_dir={quote(directory)}&path={quote(path)}")[2])
    assert not [body for body in bodies if KEY.encode("ascii") in body]
    # The working directory itself as a run directory shows no file.
    status, refused = client.json("GET", "/api/results/file?run_dir=.&path=models.ini")
    assert status == 404 and "below the working directory" in refused["error"]


def test_without_testuarys_settings_no_dir_opens(tmp_path, monkeypatch):
    _run_dir(tmp_path / "out")
    monkeypatch.setitem(sys.modules, "testuary.settings", None)
    with pytest.raises(_results.NotShown, match="settings"):
        _results.locate(tmp_path, "out")


def test_a_java_test_goes_below_the_projects_test_root_in_its_package(tmp_path):
    """The root of a project's own tests is their directory without the package
    path their files declare. A delivered test goes below it in its own."""
    from testuary.languages import get_backend

    backend = get_backend("java")
    project = tmp_path / "jsonproject"
    own = project / "src" / "test" / "java" / "org" / "example"
    own.mkdir(parents=True)
    (own / "FooTest.java").write_text("package org.example;\n\npublic class FooTest {}\n", encoding="utf-8")
    (own / "BarTest.java").write_text("package org.example;\n\npublic class BarTest {}\n", encoding="utf-8")
    delivered = tmp_path / "JSONCompareTest.java"
    delivered.write_text("package org.skyscreamer.jsonassert;\n\npublic class JSONCompareTest {}\n",
                         encoding="utf-8")
    assert _results._tests_dir(project, delivered, backend) == (
        project / "src" / "test" / "java" / "org" / "skyscreamer" / "jsonassert")
