"""With Testuary itself: a demonstration through the interface, from its
progress events to its results, and a check whose model names and keys
reach no response and no log of the interface."""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.parse import quote

import pytest

import fake_model
from conftest import SOURCE, wait_for
from testuary import web as web_app
from testuary.web import _data as projects

pytestmark = pytest.mark.slow

#: The limits of a small first run, as Testuary's getting-started guide
#: shows them.
SMALL_RUN = ("[starter]\nsuite_count = 3\nattempts_per_suite = 2\n\n"
             "[coverage]\nmax_requests_per_phase = 20\nmax_output_tokens_per_phase = 200000\n"
             "max_wall_clock_seconds_per_phase = 1800\n\n"
             "[mutation]\nmax_requests_per_phase = 20\nmax_output_tokens_per_phase = 200000\n"
             "max_wall_clock_seconds_per_phase = 1800\n")


@pytest.fixture
def testuary(monkeypatch):
    """The children find this source tree and the Testuary under test, and no
    Testuary setting of the machine: only the test's user directory."""
    paths = [str(SOURCE)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))
    for name in list(os.environ):
        if name.startswith("TESTUARY_") and name != "TESTUARY_HOME":
            monkeypatch.delenv(name)
    return monkeypatch


def _works():
    """Testuary's command line answers: a broken command fails these
    tests, never skips them."""
    version = web_app._testuary_version(sys.executable)
    assert version.startswith("testuary "), version


def test_a_demonstration_through_the_interface(serve, testuary, tmp_path):
    _works()
    work = tmp_path / "work"
    work.mkdir()
    (work / "models.ini").write_bytes(
        b"[fake_alias]\nprovider = openai\nmodel = fake-model-name\n"
        b"api_key = canary-in-models-ini\n")
    app, server, client = serve(work)
    status, job = client.json("POST", "/api/demo", {"language": "python", "pace": "instant"})
    assert status == 201 and job["events"]
    stream, payload = client.stream(job["id"])
    end = stream[-1][1]
    text = "\n".join(data["line"] for kind, data in stream if kind == "text")
    assert (end["status"], end["exit_code"]) == ("finished", 0), text[-3000:]
    views = [data for kind, data in stream if kind == "view"]
    logged = [data for kind, data in stream if kind == "log"]
    assert len(views) > 3 and len(logged) > 30
    assert views[-1]["title"] == "Testuary | demo playback"
    assert "SUCCEEDED" in views[-1]["layers"][0]["text"]
    assert "Demo complete" in text
    status, summary = client.json("GET", f"/api/results?run_dir={quote(end['output_dir'])}")
    assert status == 200 and summary["demo"]
    assert summary["demo_record"]["result"]["outcome"] == "completed"
    assert summary["rows"][0]["line_coverage"] and summary["rows"][0]["demo"] == "demo"
    (run,) = summary["runs"][0]["files"][0]["runs"]
    assert (run["name"], run["row"], run["summarized"]) == ("r0", 0, True)
    assert run["values"]["line_coverage"]["text"].endswith("%")
    suite = [item["path"] for item in run["suite"]["files"]]
    test_file = next(path for path in suite if path.endswith(".py") and "/test_" in path)
    status, file = client.json(
        "GET", f"/api/results/file?run_dir={quote(end['output_dir'])}&path={quote(test_file)}")
    assert status == 200 and "def test" in file["text"]
    # The key planted in the working directory's catalogue reaches neither the
    # stream nor any results view, and the demonstration's own catalogue is
    # not shown.
    listed = [item["path"] for item in summary["files"]]
    assert "workspace/models.ini" not in listed and "workspace/preset.ini" in listed
    bodies = [payload, client.request("GET", f"/api/results?run_dir={quote(end['output_dir'])}")[2]]
    for path in listed + ["workspace/models.ini", "../models.ini"]:
        bodies.append(client.request(
            "GET", f"/api/results/file?run_dir={quote(end['output_dir'])}&path={quote(path)}")[2])
    assert not [body for body in bodies if b"canary-in-models-ini" in body]


def test_a_demonstration_is_paced_over_the_configured_length(serve, testuary):
    """The form's default plays at normal speed over [demo]
    playback_seconds, here set by Testuary's own variable."""
    _works()
    testuary.setenv("TESTUARY_DEMO_PLAYBACK_SECONDS", "15")
    app, server, client = serve()
    status, job = client.json("POST", "/api/demo", {"language": "python"})
    assert status == 201
    stream, _ = client.stream(job["id"])
    end = stream[-1][1]
    assert (end["status"], end["exit_code"]) == ("finished", 0)
    assert end["elapsed_s"] >= 15
    status, summary = client.json("GET", f"/api/results?run_dir={quote(end['output_dir'])}")
    assert summary["demo_record"]["result"]["playback"].startswith(
        "paced, at least about 15 seconds")


def test_keys_never_reach_the_page_and_model_names_do(serve, testuary, tmp_path, user_dir, capsys):
    work = tmp_path / "work"
    (work / "proj" / ".testuary").mkdir(parents=True)
    (user_dir / "models.ini").write_bytes(
        b"[fake_alias]\nprovider = openai\nmodel = fake-model-name\n"
        b"api_url = http://127.0.0.1:9/v1\napi_key = canary-in-models-ini\n")
    (user_dir / "config.ini").write_bytes(b"[api]\nmodel = fake_alias\n")
    (work / "proj" / "mod.py").write_bytes(b"def add(a, b):\n    return a + b\n")
    (work / "proj" / ".testuary" / "project.ini").write_text(
        f"[project]\npython_executable = {sys.executable}\n", encoding="utf-8")
    testuary.setenv("TESTUARY_OPENAI_API_KEY", "canary-in-environment")
    app, server, client = serve(work)
    projects.add(str(work / "proj"))
    status, job = client.json("POST", "/api/check", {"project_dir": str(work / "proj"),
                                                     "files": ["mod.py"]})
    assert status == 201
    stream, payload = client.stream(job["id"])
    end = stream[-1][1]
    report = end["report"]
    assert report is not None, end["errors"]
    assert report["credentials_configured"] is True and report["exit_code"] == 0
    # Testuary's own texts for the report, as the console writes them.
    assert end["report_text"]["summary"] and end["report_text"]["detailed"]
    models = [item for item in report["items"] if item["key"].startswith("configuration.model.")]
    assert models and all(item["status"] == "verified" for item in models)
    # Model names are shown, as the console shows them.
    assert (report["model_alias"], report["model"]) == ("fake_alias", "fake-model-name")
    bodies = [payload, client.request("GET", f"/api/jobs/{job['id']}")[2],
              client.request("GET", "/api/state")[2], capsys.readouterr().err.encode("utf-8"),
              (app.jobs.get(job["id"]).job_dir / "stdout.txt").read_bytes()]
    for body in bodies:
        for key in (b"canary-in-models-ini", b"canary-in-environment"):
            assert key not in body


def test_a_check_after_a_run_passes_with_testuary(serve, testuary, tmp_path, user_dir):
    """With Testuary itself: the project's default run directory holds a finished
    run, which Testuary refuses for a new run, and the check still passes:
    it judges the free directory a fresh run would use."""
    project = tmp_path / "proj"
    (project / ".testuary").mkdir(parents=True)
    (user_dir / "models.ini").write_bytes(
        b"[fake_alias]\nprovider = openai\nmodel = fake-model-name\n"
        b"api_url = http://127.0.0.1:9/v1\n")
    (project / "mod.py").write_bytes(b"def add(a, b):\n    return a + b\n")
    (project / ".testuary" / "project.ini").write_text(
        f"[project]\npython_executable = {sys.executable}\nsource_files = mod.py\n\n"
        "[api]\nmodel = fake_alias\n", encoding="utf-8")
    testuary.setenv("TESTUARY_OPENAI_API_KEY", "loopback-test-key")
    saved = project / ".testuary" / "results" / "fake_alias" / "results"
    saved.mkdir(parents=True)
    (saved / "summary.csv").write_bytes(b"file\n")
    app, server, client = serve(project)
    status, job = client.json("POST", "/api/check", {"project_dir": str(project)})
    assert status == 201 and "--output-dir=.testuary/results/fake_alias-2" in job["arguments"]
    stream, payload = client.stream(job["id"])
    end = stream[-1][1]
    report = end["report"]
    assert report is not None, end["errors"]
    assert report["exit_code"] == 0, [(item["key"], item["reason"]) for item in report["items"]]
    assert report["output_dir"].endswith("fake_alias-2")


def test_a_demonstration_stops_as_ctrl_c_stops_it(serve, testuary):
    _works()
    app, server, client = serve()
    status, job = client.json("POST", "/api/demo", {"language": "python", "pace": "paced",
                                                    "seconds": "120"})
    assert status == 201
    events = app.jobs.get(job["id"]).events
    # Stop once the playback has begun.
    assert wait_for(lambda: events.exists()
                    and len(events.read_text(encoding="utf-8").splitlines()) > 5, 120)
    assert client.json("POST", f"/api/jobs/{job['id']}/stop", {})[0] == 200
    stream, _ = client.stream(job["id"])
    end = stream[-1][1]
    assert (end["status"], end["exit_code"]) == ("cancelled", 130)
    assert end["elapsed_s"] < 100
    status, summary = client.json("GET", f"/api/results?run_dir={quote(end['output_dir'])}")
    assert status == 200 and summary["demo_record"]["result"]["outcome"] == "cancelled"


@pytest.mark.mutation_tool
def test_a_run_through_the_interface_with_a_loopback_model(serve, testuary):
    """A real run of a small project: Testuary's pipeline with real pytest,
    coverage and mutation tools against a model service on 127.0.0.1. The
    run's key reaches no response."""
    _works()
    # A short directory: the project copies' paths stay within Windows' limit.
    root = Path(tempfile.mkdtemp(prefix="tw", dir=tempfile.gettempdir()))
    model = fake_model.start()
    try:
        work = root / "w"
        (work / "calcproj" / "calc").mkdir(parents=True)
        (work / "calcproj" / "calc" / "__init__.py").write_bytes(b"")
        (work / "calcproj" / "calc" / "ops.py").write_bytes(fake_model.OPS.encode("ascii"))
        port = model.server_address[1]
        user = Path(os.environ["TESTUARY_HOME"])
        (user / "models.ini").write_bytes(
            f"[fake]\nprovider = openai\nmodel = fake-model\n"
            f"api_url = http://127.0.0.1:{port}/v1\n".encode("ascii"))
        # The project's settings file: its files, interpreter and model, and
        # the getting-started guide's limits of a small first run.
        (work / "calcproj" / ".testuary").mkdir()
        (work / "calcproj" / ".testuary" / "project.ini").write_text(
            f"[project]\nsource_files = calc/ops.py\npython_executable = {sys.executable}\n\n"
            "[api]\nmodel = fake\n\n" + SMALL_RUN, encoding="utf-8")
        testuary.setenv("TESTUARY_OPENAI_API_KEY", "loopback-test-key")
        app, server, client = serve(work)
        projects.add(str(work / "calcproj"))
        status, job = client.json("POST", "/api/run", {"project_dir": str(work / "calcproj")})
        assert status == 201, job
        stream, payload = client.stream(job["id"])
        end = stream[-1][1]
        text = "\n".join(data["line"] for kind, data in stream if kind == "text")
        assert (end["status"], end["exit_code"]) == ("finished", 0), text[-4000:]
        views = [data for kind, data in stream if kind == "view"]
        assert views and views[-1]["log_path"], "the run named no log"
        run_dir = Path(views[-1]["log_path"]).parent.parent
        status, summary = client.json("GET", f"/api/results?run_dir={quote(str(run_dir))}")
        assert status == 200 and summary["rows"], summary
        assert summary["rows"][0].get("line_coverage")
        (run,) = summary["runs"]
        (first,) = run["files"][0]["runs"]
        # As Testuary's lines write a percentage.
        from testuary.presentation.view import percent_text

        assert first["values"]["line_coverage"]["text"] == (
            f"{percent_text(float(summary['rows'][0]['line_coverage']))}%")
        # The end-of-run summary comes from the run directory's completion.json.
        assert (run_dir / "results" / "completion.json").is_file()
        assert run["summary"]["job"] is None and run["summary"]["lines"]
        bodies = [payload, client.request("GET", "/api/state")[2],
                  client.request("GET", f"/api/jobs/{job['id']}")[2]]
        assert not [body for body in bodies if b"loopback-test-key" in body]
    finally:
        model.shutdown()
        model.server_close()
        shutil.rmtree(root, ignore_errors=True)


def _console_frame(events_file):
    """The console's final compact frame for an event file: Testuary's own
    presenter, fed the same observations, on a terminal wide enough that no
    row wraps. One line per row, the bar glyphs left out."""
    import io
    import json as json_module

    from rich.console import Console

    from testuary.presentation import DisplayEvent, TerminalPresenter

    ticks = iter(range(10 ** 6))
    console = Console(file=io.StringIO(), width=1000, height=60, force_terminal=True,
                      color_system=None, legacy_windows=False)
    with TerminalPresenter("compact", console=console, clock=lambda: next(ticks) * 0.1) as presenter:
        for line in events_file.read_text(encoding="utf-8").splitlines():
            record = json_module.loads(line)
            fields = record.get("event") if "event" in record else record if "level" in record else None
            if fields is not None:
                presenter(DisplayEvent.from_dict(fields))
        mirror = Console(file=io.StringIO(), width=1000, height=60, record=True, color_system=None)
        mirror.print(presenter.frame())
    lines = []
    for line in mirror.export_text().splitlines():
        text = " ".join(line.replace("\u2588", " ").replace("\u2591", " ").split())
        if text:
            lines.append(text)
    return lines


def _web_view(view):
    """The web's final view, in the console frame's order, one line per row."""
    lines = [view["title"]]
    for layer in view["layers"]:
        lines.append(layer["text"])
        if layer["bar"]:
            lines.append(layer["bar"]["figures"])
    for operation in view["operations"]:
        work = operation["bar"]["figures"] if operation["bar"] else operation["progress"]
        lines.append(" ".join([operation["state"], operation["name"], work, operation["elapsed"]]))
    if view["hidden_active"]:
        lines.append(f"{view['hidden_active']} additional active operations")
    lines += [text for text in (view["message"], view["metrics"], view["usage"]) if text]
    lines += view["issues"]
    return [" ".join(line.split()) for line in lines if line.strip()]


@pytest.mark.parametrize("language", ["python", "java"])
def test_the_final_view_is_the_consoles_final_frame(serve, testuary, language):
    """A demonstration's final web view shows what the console's final
    frame shows, and nothing else."""
    _works()
    app, server, client = serve()
    status, job = client.json("POST", "/api/demo", {"language": language, "pace": "instant"})
    assert status == 201
    stream, _ = client.stream(job["id"])
    assert stream[-1][1]["status"] == "finished"
    final = [data for kind, data in stream if kind == "view"][-1]
    assert _web_view(final) == _console_frame(app.jobs.get(job["id"]).events)
