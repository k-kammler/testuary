"""Jobs run one at a time, each in a process of its own, and stop as Ctrl+C
stops them: the stand-in command ends with status 130 and keeps its output.
Force stop comes only after Testuary's cleanup time and ends every process
the job started, and an exit status is explained in Testuary's own words
(testuary.exits)."""
from __future__ import annotations

import os
import re
import subprocess
import sys

import psutil
import pytest

from conftest import wait_for
from testuary.web import _jobs
from testuary.web._registry import Registry
from testuary.web import _data as projects
import json


def _start(app, mode, stand_in, *, events=True):
    stand_in.setenv("STAND_IN_MODE", mode)
    job = app.jobs.start("demo", "Stand-in", ["--demo"], events=events)
    assert wait_for(lambda: "stand-in started" in job.stdout.read_text(encoding="utf-8")
                    if job.stdout.exists() else False)
    return job


def test_a_run_that_wrote_no_results_says_so_with_its_own_error(serve, stand_in, tmp_path):
    """A run refused before its run directory exists, as one with a project
    interpreter that does not exist is, offers no run directory and names why
    in Testuary's own words, its setup report's. A run that wrote its
    directory offers it."""
    project = tmp_path / "calc"
    (project / ".testuary").mkdir(parents=True)
    (project / ".testuary" / "project.ini").write_text("[project]\nsource_files = calc.py\n",
                                                       encoding="utf-8")
    app, server, client = serve()
    projects.add(str(project))

    def ended(mode, output):
        stand_in.setenv("STAND_IN_MODE", mode)
        status, job = client.json("POST", "/api/run", {"project_dir": str(project), "output_dir": output})
        assert status == 201, job
        assert wait_for(lambda: client.json("GET", f"/api/jobs/{job['id']}")[1]["status"] != "running")
        return client.json("GET", f"/api/jobs/{job['id']}")[1]

    refused = ended("noresults", ".testuary/results/fake-4")
    assert refused["run_dir"] is None and refused["output_dir"] and refused["exit_code"] == 2
    assert refused["error"].startswith("required tools are missing for this run:\n")
    assert "python_executable: points at" in refused["error"]
    assert refused["error"].endswith("No phase ran.")
    wrote = ended("results", "out")
    assert wrote["run_dir"] == str(project / "out")


def test_a_job_names_its_run_directory_once_and_a_check_names_none(serve, stand_in):
    """A job's view names its run directory once, as run_dir: where a run
    writes while it runs and the run it wrote once it ended (the tests
    above), and none for a check, which writes no run."""
    app, server, client = serve()
    project = app.base_dir.parent / "calcproj"
    project.mkdir()
    projects.add(str(project))
    stand_in.setenv("STAND_IN_MODE", "exit0")
    status, check = client.json("POST", "/api/check", {"project_dir": str(project)})
    assert status == 201, check
    assert wait_for(lambda: app.jobs.get(check["id"]).ended is not None)
    viewed = client.json("GET", f"/api/jobs/{check['id']}")[1]
    assert viewed["run_dir"] is None and "dir" not in viewed


def test_a_job_runs_in_its_own_process_and_reports_its_status(serve, stand_in):
    app, server, client = serve()
    job = _start(app, "exit3", stand_in)
    assert wait_for(lambda: job.ended is not None)
    assert (job.status, job.exit_code) == ("ended", 3)
    view = job.view()
    assert view["command"] == "testuary --demo" and view["force_after_s"] is None
    assert job.events is not None and '"stand-in"' in job.events.read_text(encoding="utf-8")


def test_stop_ends_a_job_as_ctrl_c_does(serve, stand_in):
    app, server, client = serve()
    job = _start(app, "wait", stand_in)
    status, view = client.json("POST", f"/api/jobs/{job.id}/stop", {})
    assert status == 200 and view["status"] in ("stopping", "cancelled")
    assert wait_for(lambda: job.ended is not None, 60)
    assert (job.status, job.exit_code) == ("cancelled", 130)
    assert "stand-in interrupted" in job.stdout.read_text(encoding="utf-8")


def test_a_stopped_job_ends_stopped_whatever_its_status(serve, stand_in):
    """A job the person stopped ends as stopped, without an error of its
    own, also when stopped while Python still starts (the system's status
    for Ctrl+C and a KeyboardInterrupt)."""
    app, server, client = serve()
    job = _start(app, "early", stand_in)
    client.json("POST", f"/api/jobs/{job.id}/stop", {})
    assert wait_for(lambda: job.ended is not None, 60)
    assert (job.status, job.exit_code, job.error) == ("cancelled", 3, None)


def test_a_stop_while_testuary_starts_ends_quietly_as_interrupted(serve, stand_in):
    """A stop before Testuary's own Ctrl+C handling ends as Testuary ends an
    interrupt, without a traceback, and a requested stop reads as
    interrupted, no failure, in the job's view."""
    app, server, client = serve()
    stand_in.setenv("STAND_IN_MODE", "importstop")
    job = app.jobs.start("run", "Run", ["calc.py"], events=True)
    assert wait_for(lambda: job.ended is not None, 60)
    errors = job.stderr.read_text(encoding="utf-8")
    assert job.exit_code == 130 and "Traceback" not in errors and "Interrupted." in errors
    early = _start(app, "early", stand_in)
    client.json("POST", f"/api/jobs/{early.id}/stop", {})
    assert wait_for(lambda: early.ended is not None, 60)
    view = client.json("GET", f"/api/jobs/{early.id}")[1]
    assert (view["status"], view["word"], view["failure"]) == ("cancelled", "interrupted", False)


def test_a_run_refused_before_its_progress_says_why(serve, stand_in):
    """A configuration Testuary refuses before any progress event leaves
    only its setup report in the event file and no ERROR line: the job's
    error is that report's blocking reason, which the cause panel shows."""
    app, server, client = serve()
    job = _start(app, "refused", stand_in)
    assert wait_for(lambda: job.ended is not None, 60)
    view = client.json("GET", f"/api/jobs/{job.id}")[1]
    assert (view["exit_code"], view["word"]) == (2, "refused")
    assert view["error"] == "[coverage] goal: expected a percentage from 0 to 100"


def test_one_job_at_a_time(serve, stand_in):
    app, server, client = serve()
    job = _start(app, "wait", stand_in)
    with pytest.raises(_jobs.Busy):
        app.jobs.start("demo", "Second", ["--demo"])
    assert client.json("POST", "/api/demo", {"language": "python"})[0] == 409
    app.jobs.stop(job.id)
    assert wait_for(lambda: job.ended is not None, 60)


def _alive(process) -> bool:
    try:
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False


def test_force_comes_only_after_testuarys_cleanup_time(serve, stand_in):
    """Force stop comes after Testuary's cleanup time and ends every process
    the job started, also a command in a session of its own."""
    from testuary import exits

    app, server, client = serve()
    cleanup = app.jobs.cleanup_seconds
    assert cleanup == exits.CLEANUP_SECONDS
    job = _start(app, "stubborn", stand_in)
    assert wait_for(lambda: "stand-in child" in job.stdout.read_text(encoding="utf-8"))
    child = psutil.Process(int(re.search(r"stand-in child (\d+)", job.stdout.read_text(encoding="utf-8"))[1]))
    with pytest.raises(_jobs.NotAllowed, match="Stop the job first"):
        app.jobs.force(job.id)
    app.jobs.stop(job.id)
    assert not wait_for(lambda: job.ended is not None, 2)
    assert client.json("POST", f"/api/jobs/{job.id}/force", {})[0] == 409
    assert 0 < job.force_after() <= cleanup
    job.stop_requested -= cleanup + 1
    app.jobs.force(job.id)
    assert wait_for(lambda: job.ended is not None, 60)
    try:
        assert wait_for(lambda: not _alive(child), 10)
    finally:
        if _alive(child):
            child.kill()
    # A forced end's status is the system's, which Testuary does not explain.
    assert job.status == "stopped" and app.job_view(job)["meaning"] is None


@pytest.mark.skipif(os.name == "nt", reason="taskkill /T ends the tree by its parent links")
def test_force_ends_what_a_listed_group_leader_starts_after_the_listing(serve, tmp_path, monkeypatch):
    """A command the job started leads a group of its own, and a process
    joins that group after Force stop listed the job's processes, as a
    mutation runner starts its next run: the group ends with its leader."""
    job_process = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"],
                                   start_new_session=True)
    # The leader starts a process of its own group and names it.
    leader = subprocess.Popen(
        [sys.executable, "-c", "import subprocess, sys, time\n"
         "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'])\n"
         "print(child.pid, flush=True)\n"
         "time.sleep(300)\n"],
        start_new_session=True, stdout=subprocess.PIPE, text=True)
    started = [psutil.Process(job_process.pid), psutil.Process(leader.pid),
               psutil.Process(int(leader.stdout.readline()))]
    # The listing names the leader alone: its process came after it.
    monkeypatch.setattr(psutil.Process, "children", lambda self, recursive=False: [started[1]])
    app, server, client = serve()
    job = _jobs.Job("id", "run", "Run", [], tmp_path, None, None, process=job_process)
    try:
        app.jobs.kill(job)
        assert wait_for(lambda: not any(_alive(process) for process in started), 10)
    finally:
        for process in started:
            if _alive(process):
                process.kill()
        for process in (job_process, leader):
            process.wait()
        leader.stdout.close()


def test_without_a_cleanup_time_force_stop_comes_at_once(tmp_path):
    job = _jobs.Job("id", "run", "Run", [], tmp_path, None, None, process=None)
    assert job.force_after(now=100.0) is None
    job.stop_requested = 100.0
    assert job.force_after(now=100.0) == 0.0
    job.cleanup_seconds = 120.0
    assert job.force_after(now=130.0) == 90.0


def test_an_ended_job_has_testuarys_word_for_its_status(serve, stand_in, tmp_path):
    """Each ended job has Testuary's word for its command and status and
    whether that is a failure: a check that found a problem or a refused
    setting is not a failure of the job."""
    app, server, client = serve()
    project = tmp_path / "proj"
    project.mkdir()
    projects.add(str(project))
    seen = []
    for mode, path in (("exit1", "/api/check"), ("exit2", "/api/run"), ("exit1", "/api/run"),
                       ("exit0", "/api/check"), ("exit7", "/api/check")):
        stand_in.setenv("STAND_IN_MODE", mode)
        status, started = client.json("POST", path, {"project_dir": str(project)})
        assert status == 201, started
        assert wait_for(lambda: app.jobs.get(started["id"]).ended is not None)
        view = client.json("GET", f"/api/jobs/{started['id']}")[1]
        seen.append((view["word"], view["failure"]))
    assert seen == [("problem", False), ("refused", False), ("failed", True),
                    ("ready", False), ("failed", True)]
    # The jobs list says the same.
    listed = client.json("GET", "/api/state")[1]["jobs"]
    assert [(job["word"], job["failure"]) for job in reversed(listed)] == seen


def test_an_exit_status_is_explained_in_testuarys_words(serve, stand_in):
    from testuary import exits

    app, server, client = serve()
    project = app.base_dir.parent / "calcproj"
    project.mkdir()
    projects.add(str(project))
    job = _start(app, "exit3", stand_in)
    assert wait_for(lambda: job.ended is not None)
    assert app.job_view(job)["meaning"] == exits.meaning(exits.DEMO, 3)
    body = {"project_dir": str(project)}
    for mode, path, command in (("exit2", "/api/check", exits.CHECK),
                                ("exit0", "/api/run", exits.RUN),
                                ("exit7", "/api/run", exits.RUN)):
        stand_in.setenv("STAND_IN_MODE", mode)
        status, started = client.json("POST", path, body)
        assert status == 201, started
        assert wait_for(lambda: app.jobs.get(started["id"]).ended is not None)
        status, view = client.json("GET", f"/api/jobs/{started['id']}")
        assert view["meaning"] == exits.meaning(command, int(mode[4:]))
    # A status Testuary does not document gets its plain sentence.
    assert view["meaning"] == "The command ended with status 7."


def test_the_stream_sends_events_text_and_the_end(serve, stand_in):
    app, server, client = serve()
    job = _start(app, "exit0", stand_in)
    events, _ = client.stream(job.id)
    kinds = [kind for kind, _ in events]
    assert kinds[-1] == "end" and events[-1][1]["status"] == "finished"
    # The event file's observation, through Testuary's view model: its row
    # and the line the console's log records.
    views = [data for kind, data in events if kind == "view"]
    assert views and views[-1]["message"] == "stand-in"
    assert [layer["name"] for layer in views[-1]["layers"]][:1] == ["Overall"]
    assert [data for kind, data in events if kind == "log"]
    assert {"stream": "stdout", "line": "stand-in started"} in [
        data for kind, data in events if kind == "text"]


def test_an_ended_jobs_stream_replays_its_whole_event_file(serve, stand_in):
    """A job's stream opened after the job ended replays an event file
    longer than the server reads at a time to its last record."""
    app, server, client = serve()
    job = _start(app, "exit0", stand_in)
    assert wait_for(lambda: job.ended is not None)
    event = {"run_id": "r", "target": "all targets", "phase": "pipeline", "operation": "run",
             "state": "running", "elapsed_s": 0.0, "message": "", "level": "run"}
    with open(job.events, "a", encoding="utf-8") as stream:
        for step in range(3 * _jobs.READ_CHUNK // 200):
            stream.write(json.dumps({"event": {**event, "message": f"Step {step}"}}) + "\n")
        stream.write(json.dumps({"event": {**event, "state": "succeeded", "message": "Run completed"}}) + "\n")
    events, _ = client.stream(job.id)
    views = [data for kind, data in events if kind == "view"]
    assert events[-1][0] == "end" and views[-1]["message"] == "Run completed"


def test_a_run_always_writes_its_progress_events(stand_in, tmp_path):
    """The interface runs the Testuary it belongs to, whose generate always
    has --events: no probe decides it, and a run or a demonstration always
    writes its progress events."""
    manager = _jobs.JobManager(tmp_path / "work")
    stand_in.setenv("STAND_IN_MODE", "exit0")
    job = manager.start("demo", "Stand-in", ["--demo"], events=True)
    assert wait_for(lambda: job.ended is not None)
    assert job.events is not None and '"stand-in"' in job.events.read_text(encoding="utf-8")


def test_read_lines_waits_for_complete_lines(tmp_path):
    path = tmp_path / "growing.txt"
    assert _jobs.read_lines(path, 0) == ([], 0)
    path.write_bytes(b"one\r\ntwo\nthr")
    lines, offset = _jobs.read_lines(path, 0)
    assert lines == ["one", "two"] and offset == 9
    with open(path, "ab") as stream:
        stream.write(b"ee\n")
    assert _jobs.read_lines(path, offset) == (["three"], 15)


def test_a_demonstration_is_paced_as_at_a_terminal_unless_asked(serve, stand_in):
    """Normal speed over Testuary's configured length by default. A given
    length goes to Testuary unchecked."""
    app, server, client = serve()
    stand_in.setenv("STAND_IN_MODE", "exit0")
    commands = []
    for request in ({"language": "python"}, {"language": "java", "pace": "paced"},
                    {"language": "python", "pace": "paced", "seconds": "45.5"},
                    {"language": "python", "pace": "paced", "seconds": "--help"},
                    {"language": "python", "pace": "instant", "seconds": "45"}):
        status, job = client.json("POST", "/api/demo", request)
        assert status == 201, job
        assert wait_for(lambda: app.jobs.get(job["id"]).ended is not None)
        commands.append(job["command"].split()[-3:])
    assert commands == [
        ["--no-color", "--demo-speed", "1"],
        ["--no-color", "--demo-speed", "1"],
        ["--demo-speed", "1", "--demo-playback-seconds=45.5"],
        ["--demo-speed", "1", "--demo-playback-seconds=--help"],
        ["--no-color", "--demo-speed", "instant"]]
    assert client.json("POST", "/api/demo", {"pace": "paced", "seconds": 45})[0] == 400


def _plant_modules(module_dir):
    """Modules with standard library names in the working directory, as a
    project can have them: importing any of them ends the process."""
    module_dir.mkdir(parents=True, exist_ok=True)
    for name in ("json", "signal", "runpy", "secrets", "threading"):
        (module_dir / f"{name}.py").write_bytes(b"raise SystemExit(99)\n")


def test_a_job_imports_nothing_from_the_working_dir(serve, stand_in, tmp_path):
    _plant_modules(tmp_path / "work")
    app, server, client = serve(tmp_path / "work")
    job = _start(app, "exit0", stand_in)
    assert wait_for(lambda: job.ended is not None)
    assert (job.status, job.exit_code) == ("finished", 0)


def test_a_relative_path_entry_does_not_reach_into_the_working_dir(serve, stand_in, tmp_path):
    """A relative entry of PYTHONPATH resolves against the job's directory, so
    the job leaves it out: the modules planted there are not imported."""
    import os

    _plant_modules(tmp_path / "work" / "lib")
    stand_in.setenv("PYTHONPATH", os.pathsep.join(["lib", os.environ["PYTHONPATH"]]))
    app, server, client = serve(tmp_path / "work")
    job = _start(app, "exit0", stand_in)
    assert wait_for(lambda: job.ended is not None)
    assert (job.status, job.exit_code) == ("finished", 0)


def test_testuarys_version_probe_imports_nothing_from_the_current_dir(stand_in, tmp_path):
    from testuary import web as web_app

    _plant_modules(tmp_path / "here")
    stand_in.chdir(tmp_path / "here")
    stand_in.setenv("STAND_IN_MODE", "exit0")
    assert web_app._testuary_version(sys.executable) == "stand-in"


def test_a_continued_run_is_named_so_once(serve, stand_in):
    """A run that continues an earlier one is named "..., continued" once,
    whether the Run page started it with --resume or a job's Continue did:
    the name follows the --resume argument, not the earlier name."""
    app, server, client = serve()
    project = app.base_dir.parent / "calcproj"
    project.mkdir()
    projects.add(str(project))
    stand_in.setenv("STAND_IN_MODE", "incomplete")
    status, job = client.json("POST", "/api/run", {"project_dir": str(project), "output_dir": "out1",
                                                   "resume": True})
    assert status == 201, job
    assert job["title"] == "Run of calcproj, continued"
    assert wait_for(lambda: app.jobs.get(job["id"]).ended is not None)
    status, again = client.json("POST", f"/api/jobs/{job['id']}/continue", {})
    assert status == 201, again
    assert again["title"] == "Run of calcproj, continued"
    assert wait_for(lambda: app.jobs.get(again["id"]).ended is not None)


def test_an_incomplete_run_states_its_outcome_and_can_be_continued(serve, stand_in):
    """A run whose file got no suite that meets the minimums exits 0 but
    ends incomplete, with Testuary's one-sentence outcome. Continue starts
    the same run with --resume in the same run directory, and only while
    files are left."""
    app, server, client = serve()
    project = app.base_dir.parent / "calcproj"
    project.mkdir()
    projects.add(str(project))
    stand_in.setenv("STAND_IN_MODE", "incomplete")
    status, job = client.json("POST", "/api/run", {"project_dir": str(project), "output_dir": "out1"})
    assert status == 201, job
    assert wait_for(lambda: app.jobs.get(job["id"]).ended is not None)
    status, ended = client.json("GET", f"/api/jobs/{job['id']}")
    assert (ended["status"], ended["exit_code"]) == ("finished", 0)
    assert ended["outcome"] == {
        "incomplete": True, "unfinished": 1,
        "headline": "Run finished: 0 of 1 target with a suite that meets the minimums, 1 without one."}
    stand_in.setenv("STAND_IN_MODE", "exit0")
    status, continued = client.json("POST", f"/api/jobs/{job['id']}/continue", {})
    assert status == 201, continued
    assert continued["command"] == "testuary --display plain --no-color --output-dir=out1 --resume"
    assert (continued["title"], continued["output_dir"]) == ("Run of calcproj, continued", job["output_dir"])
    assert wait_for(lambda: app.jobs.get(continued["id"]).ended is not None)
    # A run with nothing left to finish is not continued.
    status, refused = client.json("POST", f"/api/jobs/{continued['id']}/continue", {})
    assert status == 409, refused


def test_the_jobs_list_names_each_runs_dir_model_and_values(serve, stand_in):
    """A run's entry names its model as the setup report names it, also
    while it runs, its run directory and each file's values as its
    end-of-run summary states them: cut to one decimal, never rounded up,
    and a value the run did not measure as None."""
    app, server, client = serve()
    project = app.base_dir.parent / "calcproj"
    project.mkdir()
    projects.add(str(project))
    stand_in.setenv("STAND_IN_MODE", "setup")
    status, job = client.json("POST", "/api/run", {"project_dir": str(project)})
    assert status == 201, job

    def entry():
        return next(item for item in client.json("GET", "/api/state")[1]["jobs"] if item["id"] == job["id"])

    assert wait_for(lambda: entry()["model"] == "fake")
    assert entry()["run_dir"] == str(project / "out" / "fake-2") and entry()["values"] is None
    client.json("POST", f"/api/jobs/{job['id']}/stop", {})
    assert wait_for(lambda: app.jobs.get(job["id"]).ended is not None)
    stand_in.setenv("STAND_IN_MODE", "measured")
    status, job = client.json("POST", "/api/run", {"project_dir": str(project)})
    assert wait_for(lambda: app.jobs.get(job["id"]).ended is not None)
    ended = entry()
    assert (ended["model"], ended["run_dir"]) == ("fake", str(project / "out" / "fake-2"))
    assert ended["values"] == [{"file": "calc/ops.py", "outcome": "complete",
                                "outcome_text": "a suite that meets the minimums", "values": [
        {"name": "Line coverage", "text": "100%"}, {"name": "Branch coverage", "text": "100%"},
        {"name": "Mutation score", "text": "91.8%"}, {"name": "Mutation score", "text": None}]}]


def test_a_run_passes_each_option_to_testuary_as_given(serve, stand_in):
    app, server, client = serve()
    project = app.base_dir.parent / "calcproj"
    project.mkdir()
    projects.add(str(project))
    stand_in.setenv("STAND_IN_MODE", "exit0")
    status, job = client.json("POST", "/api/run", {
        "project_dir": str(project), "files": ["calc/ops.py"], "model": "--help",
        "runs": "2", "run": "1", "output_dir": "out1"})
    assert status == 201, job
    assert job["command"] == (
        "testuary calc/ops.py --display plain --no-color "
        "--model=--help --runs=2 --run=1 --output-dir=out1")
    assert job["events"] and job["project"] == str(project.resolve())
    assert job["output_dir"] == str(project.resolve() / "out1")
    assert wait_for(lambda: app.jobs.get(job["id"]).ended is not None)
    for body, field in (({}, "project_dir"), ({"project_dir": "../x"}, "project_dir"),
                        ({"project_dir": str(project), "files": ["-x"]}, "files"),
                        ({"project_dir": str(project), "runs": 2}, "runs")):
        status, refused = client.json("POST", "/api/run", body)
        assert (status, refused["field"]) == (400, field)


def test_the_report_is_kept_as_printed():
    printed = {"status": "unverified", "exit_code": 0, "model_alias": "fake_alias",
               "provider": "openai", "model": "fake-model-name", "items": []}
    assert _jobs.check_report(json.dumps(printed)) == printed


def test_the_page_reads_only_fields_the_readiness_record_holds():
    """Each field of a check report that app.js reads is a field of the
    readiness record that testuary check --json and an events file hold, so
    no report fact shows as unknown for a name the record lacks."""
    from dataclasses import fields

    from conftest import SOURCE
    from testuary.readiness import RunReadiness

    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    read = set(re.findall(r"\breport\.([a-z_]+)", source))
    assert read and read - {field.name for field in fields(RunReadiness)} == set()


def test_output_that_is_no_report_is_refused():
    for text in ("", "not json", "[]", json.dumps({"status": "failed"})):
        with pytest.raises(ValueError):
            _jobs.check_report(text)


def _report():
    from testuary.readiness import ReadinessItem, ReadinessLimit, RunReadiness

    return RunReadiness(
        scope="inspection", status="missing_setup", exit_code=2, source_dir="proj",
        execution="local", source_files=("calc/ops.py",), phases=("starter", "coverage"),
        model_alias="fake_alias", provider="openai", model="fake-model",
        credentials_configured=False, output_dir="results/fake/proj",
        output_mode="fresh",
        limits=(ReadinessLimit("coverage", "max_requests_per_phase", 20, "requests", "configured"),),
        items=(ReadinessItem("configuration.model.coverage", True, "missing_setup",
                             reason="no key was found", action="Set the model's key."),))


def test_each_check_item_has_the_reports_words():
    """The check's page names each item as the report above it does: its
    plain name and its status word, the identifier small beside it."""
    from conftest import SOURCE

    texts = _jobs.check_texts(_report().as_dict(), Registry())
    assert texts["items"] == {"configuration.model.coverage": {
        "name": "model settings for the coverage phase", "status": "missing"}}
    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    start = source.index("function reportView(")
    assert "texts.items" in source[start:source.index("\n}\n", start)]


def test_the_texts_are_testuarys_presenter_output():
    from rich.console import Console

    from testuary.presentation import TerminalPresenter, present_readiness

    report = _report()
    texts = _jobs.check_texts(report.as_dict(), Registry())
    for mode, lines in (("plain", texts["summary"]), ("detailed", texts["detailed"])):
        stream = __import__("io").StringIO()
        console = Console(file=stream, width=10000, force_terminal=False, color_system=None)
        with TerminalPresenter(mode, console=console) as presenter:
            present_readiness(report, presenter=presenter, steps=_jobs._page_steps(Registry()))
        assert lines == stream.getvalue().splitlines() and lines
    assert any("Next step" in line for line in texts["summary"])


def _texts(report, steps=None):
    """A report's plain and detailed lines through Testuary's presenter."""
    from rich.console import Console

    from testuary.presentation import TerminalPresenter, present_readiness

    found = []
    for mode in ("plain", "detailed"):
        stream = __import__("io").StringIO()
        console = Console(file=stream, width=10000, force_terminal=False, color_system=None)
        with TerminalPresenter(mode, console=console) as presenter:
            present_readiness(report, presenter=presenter, **({"steps": steps} if steps else {}))
        found += stream.getvalue().splitlines()
    return "\n".join(found)


def test_the_check_names_the_pages_controls_where_the_terminal_names_commands():
    """On the page, the check's next steps are the page's: a check with the
    tools started, the detailed report below, a run on the Run page. While
    an extension replaces or hides Check and hides Run, they name neither
    page nor the Check page's box. The terminal keeps its commands."""
    from dataclasses import replace

    from testuary.readiness import ReadinessItem
    from testuary.web.extensions import WebExtension, WebTab

    tried = ReadinessItem("native.python.item.pytest", True, "unverified", version="8.1.1",
                          reason="installed, not yet tried", action="run the verification check to try it")
    ready = replace(_report(), status="verified", exit_code=0, items=(tried,))
    problem = ReadinessItem("configuration", True, "failed", reason="no model")
    reports_seen = [ready, replace(ready, scope="verification", items=()),
                    replace(_report(), items=(problem,)), replace(_report(), items=(problem, replace(
                        problem, key="output.admission", reason="occupied")))]

    def page_of(registry):
        return "\n".join(_texts(report, _jobs._page_steps(registry)) for report in reports_seen)

    page = page_of(Registry())
    terminal = "\n".join(_texts(report) for report in reports_seen)
    assert not re.search(r"--[a-z]|testuary (check|generate)|same command", page), page
    for words in ("Tool check: also start each tool once and try the project in a copy",
                  "the detailed report below", "the Run page",
                  "check again"):
        assert words in page, words
    for words in ("testuary check --verify with the same options", "--display detailed",
                  "testuary generate with the same options", "run the same command again"):
        assert words in terminal, words
    assert ('The tool check is the check with "Tool check: also start each tool once and try the project'
            ' in a copy" ticked.') in page
    own_check = WebTab("own", "Own check", "own.js", replaces="check")
    for changes in ({"tabs": (own_check,), "hidden_tabs": ("run",)}, {"hidden_tabs": ("check", "run")}):
        other = page_of(Registry([(WebExtension(api=1, id="own", **changes), "own")]))
        assert "Tool check:" not in other and "Run page" not in other, other
        for words in ("The tool check is Testuary's check that also starts each tool once and tries the"
                      " project in a copy.", "Next step: check again with Testuary's tool check (tries the"
                      " installed tools",
                      "Next step: start a run. It sends model requests"):
            assert words in other, words


def test_a_report_this_testuary_cannot_read_has_no_texts():
    report = _report().as_dict()
    report["a_field_from_a_later_version"] = 1
    texts = _jobs.check_texts(report, Registry())
    assert (texts["summary"], texts["detailed"]) == (None, None)


def test_the_phases_are_named_by_testuarys_view_model():
    from testuary.presentation.view import phase_label

    report = _report().as_dict()
    texts = _jobs.check_texts(report, Registry())
    named = [*report["phases"], *(limit["phase"] for limit in report["limits"])]
    assert named and texts["phases"] == {phase: phase_label(phase) for phase in named}
    assert texts["phases"]["coverage"] != "coverage"


def test_the_page_finds_each_error_the_command_prints(tmp_path, monkeypatch, capsys, default_run):
    """The page reads a missing tool before any phase from the setup
    report the command records, and the command's own error by the starts
    of its lines: a phase that stopped short of the requested stop, and the
    command line's error, as the command prints them."""
    from testuary import __main__ as cli
    from testuary import languages, orchestrator
    from testuary.readiness import _configuration_error_report
    from testuary.settings import ConfigurationError

    class Broken:
        def capabilities(self, source_dir=None):
            raise RuntimeError("no interpreter")

        def check_dependencies(self):
            return []

    monkeypatch.setattr(languages, "get_backend", lambda language: Broken())
    with pytest.raises(ConfigurationError) as missing:
        orchestrator._preflight_run_dependencies({"python"}, tmp_path)
    report = _configuration_error_report(missing.value, before_run=True)
    assert _jobs.blocking_reason(report.as_dict()) == str(missing.value)

    def stopped():
        raise orchestrator._PhaseStopNotReached("the coverage phase did not reach its stop")

    monkeypatch.setattr(cli, "run_command", stopped)
    assert cli.main() == 1
    line = capsys.readouterr().err
    assert _jobs.run_error("", line) == line.strip()
    with pytest.raises(SystemExit):
        cli._build_parser().parse_args(["--no-such-option"])
    refused = capsys.readouterr().err
    assert _jobs.run_error("", refused) == refused.strip().splitlines()[-1]


def test_a_commands_own_error_is_found_in_its_output():
    """The last block that starts with ERROR:, Stopped: or the command
    line's error:, with the lines that continue it, else the exception line
    of a traceback. Progress lines and blank lines end a block."""
    stdout = ("stand-in started\nERROR: required tools are missing for this run:\n"
              "  [python] capability report failed\nInstall them and re-run; no phase ran.\n\nlater\n")
    assert _jobs.run_error(stdout, "") == (
        "ERROR: required tools are missing for this run:\n"
        "  [python] capability report failed\nInstall them and re-run; no phase ran.")
    assert _jobs.run_error("", "00:00:01 RUNNING | Run\nStopped: phase failed. The run's log names why.\n"
                                 "00:00:02 FAILED | Run") == "Stopped: phase failed. The run's log names why."
    assert _jobs.run_error("", "usage: testuary [-h]\ntestuary: error: unrecognized arguments: --x\n") == (
        "testuary: error: unrecognized arguments: --x")
    traceback = "Traceback (most recent call last):\n  File \"x.py\", line 1\n    boom()\nRuntimeError: boom\n"
    assert _jobs.run_error("", traceback) == "RuntimeError: boom"
    assert _jobs.run_error("all good\n", "00:00:01 SUCCEEDED | Run\n") is None
