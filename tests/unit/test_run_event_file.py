"""``run_pipeline(..., events=PATH)`` writes a program's run into an event
file as ``testuary --events`` writes it. Only the phase body is stubbed:
the file loop, the plan, the completion and the events are real."""
from __future__ import annotations

import json
import subprocess
from types import SimpleNamespace

import pytest

import test_generation_reuse_interruptions as harness
from testuary import __main__ as cli
from testuary import config, orchestrator, settings as settings_module
from testuary import target_phases as phases
from testuary.orchestrator import RunRequest
from testuary.settings import ConfigurationError, RunSettings


@pytest.fixture(autouse=True)
def no_native_or_provider(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native process"))
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_check_canonical_against_git", lambda *a: [])


def _starter_only(ctx):
    phases._display_phase(ctx, "starter", "running")
    phases._display_phase(ctx, "starter", "succeeded")
    return phases.TestPipelineResult(stopped_after_phase="starter")


def _records(path, base=None):
    """The file's records, with ``base`` written BASE, and without the
    run's id and clock, which differ between two runs."""
    text = path.read_text(encoding="utf-8")
    if base is not None:
        text = text.replace(json.dumps(str(base))[1:-1], "BASE").replace(base.as_posix(), "BASE")
    records = [json.loads(line) for line in text.splitlines()]
    for record in records:
        for value in record.values():
            if isinstance(value, dict):
                value.pop("run_id", None)
                value.pop("elapsed_s", None)
    return records


def _kind(record):
    return next(iter(record))


def test_a_program_names_the_event_file_and_the_web_interface_shows_it(tmp_path, monkeypatch):
    """Without an observer the file holds the header, every event with the
    plan before the event that announces it and the setup report before
    the first target starts, as the command line's file does, the
    completion without a resume command and the last run event, and the
    web interface's view of the file shows the plan's phases and the run as
    ended."""
    from testuary.web import _jobs

    _, _, _, settings = harness.setup_case(tmp_path)
    monkeypatch.setattr(phases, "run_target_phases", _starter_only)
    path = tmp_path / "events.jsonl"
    request = RunRequest(("module.py",), stop_after_phase="starter")
    assert orchestrator.run_pipeline(request, settings=settings, events=path) == 0
    records = _records(path)
    kinds = [_kind(record) for record in records]
    started = next(index for index, record in enumerate(records)
                   if _kind(record) == "event" and record["event"]["message"] == "Target started")
    assert kinds[0] == "schema_version" and kinds.count("readiness") == 1
    assert kinds.index("plan") < kinds.index("readiness") < started
    assert kinds[-2:] == ["completion", "event"] and kinds.count("plan") == 1
    assert records[kinds.index("plan") + 1]["event"]["message"] == "Run planned"
    assert records[-2]["completion"]["resume_command"] is None
    # Ended after the requested phase, the target meets no minimum: an
    # outcome, not an execution failure, and the run exits 0.
    last = records[-1]["event"]
    assert (last["level"], last["state"], last["failure"]) == ("run", "failed", "outcome")
    progress = _jobs.Progress()
    for line in path.read_text(encoding="utf-8").splitlines():
        progress.add(line)
    diagram = progress.diagram()
    assert diagram["phases"] and all(phase["state"] != "running" for phase in diagram["phases"])
    assert progress.summary


def test_one_run_both_ways_writes_the_same_records(tmp_path, monkeypatch, capsys):
    """The same run from the command line with --events and from a program
    with events: the same records in the same order, the program's
    completion without the command that resumes the run."""
    monkeypatch.setattr(phases, "run_target_phases", _starter_only)
    project, _, _, settings = harness.setup_case(tmp_path / "command")
    monkeypatch.chdir(project)
    monkeypatch.setattr("sys.argv", ["testuary", "--display", "plain", "--no-color", "--events",
                                     str(tmp_path / "command.jsonl"), "--stop-after-phase", "starter",
                                     "module.py"])
    monkeypatch.setattr(settings_module, "resolve_settings", lambda **kw: settings)
    assert cli._main() == 0
    capsys.readouterr()
    _, _, _, settings = harness.setup_case(tmp_path / "program")
    request = RunRequest(("module.py",), stop_after_phase="starter")
    assert orchestrator.run_pipeline(request, settings=settings, events=tmp_path / "program.jsonl") == 0
    command = _records(tmp_path / "command.jsonl", tmp_path / "command")
    program = _records(tmp_path / "program.jsonl", tmp_path / "program")
    assert command[-2]["completion"].pop("resume_command")
    assert program[-2]["completion"].pop("resume_command") is None
    assert program == command


def test_a_stopped_run_with_the_file_alone(tmp_path, monkeypatch):
    """Stopped during a target, the file ends with the stopped run's
    completion and a last run event in the state cancelled."""
    def stopped(ctx):
        phases._display_phase(ctx, "starter", "running")
        raise KeyboardInterrupt

    _, _, _, settings = harness.setup_case(tmp_path)
    monkeypatch.setattr(phases, "run_target_phases", stopped)
    path = tmp_path / "events.jsonl"
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run_pipeline(RunRequest(("module.py",)), settings=settings, events=path)
    *_, completion, last = _records(path)
    assert completion["completion"]["files"][0]["outcome"] == "cancelled"
    assert (last["event"]["level"], last["event"]["state"]) == ("run", "cancelled")


def test_a_credential_in_an_errors_text_is_removed(tmp_path, monkeypatch):
    """A key that no pattern recognizes is removed by the run's own
    settings, as the command line's display removes it."""
    planted = "zq81Plain7731value"
    resolved = settings_module.resolve_settings(
        base_dir=tmp_path, use_environment=False,
        models={"m": {"provider": "openai", "model": "x", "api_url": "http://127.0.0.1:9/v1",
                      "api_key_env": "TUTORIAL_PLANTED_KEY"}},
        environment={"TUTORIAL_PLANTED_KEY": planted})

    def execute(args, languages):
        args._progress.emit(target="calc.py", phase="coverage", operation="request/1",
                            state="failed", message=f"API key: {planted} was refused",
                            failure="execution")
        return 0

    monkeypatch.setattr(orchestrator, "_run_pipeline", execute)
    path = tmp_path / "events.jsonl"
    assert orchestrator.run_pipeline(SimpleNamespace(), settings=resolved, events=path) == 0
    written = path.read_text(encoding="utf-8")
    assert "API key: [redacted] was refused" in written and planted not in written


@pytest.mark.parametrize("make, message", [
    (lambda root: root / "missing" / "events.jsonl", "the directory"),
    (lambda root: root, "is a directory"),
])
def test_the_path_is_checked_before_the_run(tmp_path, monkeypatch, make, message):
    def never(*args, **kwargs):
        raise AssertionError("the run started")

    monkeypatch.setattr(orchestrator, "_run_pipeline", never)
    with pytest.raises(ConfigurationError, match=message):
        orchestrator.run_pipeline(SimpleNamespace(), settings=RunSettings(), events=make(tmp_path))
    assert not (tmp_path / "missing").exists()


def test_a_relative_path_starts_from_the_current_dir(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda args, languages: 0)
    assert orchestrator.run_pipeline(SimpleNamespace(), settings=RunSettings(), events="events.jsonl") == 0
    assert _kind(_records(tmp_path / "events.jsonl")[0]) == "schema_version"


def test_a_call_refused_before_the_file_opens_writes_nothing(tmp_path, monkeypatch):
    """A request without resolved settings, settings that cannot be
    resolved and an overlapping call raise as without events, create no
    file and leave a running call's file as it was."""
    new = tmp_path / "new.jsonl"
    with pytest.raises(ConfigurationError, match="RunRequest requires explicitly resolved RunSettings"):
        orchestrator.run_pipeline(RunRequest(("module.py",)), events=new)

    def unresolved(**kwargs):
        raise ValueError("[api] model: no such model")

    monkeypatch.setattr(settings_module, "resolve_settings", unresolved)
    with pytest.raises(ConfigurationError, match="no such model"):
        orchestrator.run_pipeline(SimpleNamespace(), events=new)
    assert not new.exists()
    running = tmp_path / "running.jsonl"
    running.write_text("the running call's records\n", encoding="utf-8")
    with config.use_run_settings(RunSettings()):
        with pytest.raises(RuntimeError, match="already active"):
            orchestrator.run_pipeline(SimpleNamespace(), settings=RunSettings(), events=running)
    assert running.read_text(encoding="utf-8") == "the running call's records\n"


def test_a_run_refused_at_admission_ends_its_file_with_the_setup_report(tmp_path, monkeypatch):
    """The refused run event, then the refusal's setup report, and the
    caller receives the same error."""
    def execute(args, languages):
        # As the run's dependency check refuses a missing tool.
        raise settings_module._MissingSetup(
            "required tools are missing for this run:\n  [python] the test tool is not installed")

    monkeypatch.setattr(orchestrator, "_run_pipeline", execute)
    path = tmp_path / "events.jsonl"
    with pytest.raises(ConfigurationError, match="required tools are missing"):
        orchestrator.run_pipeline(SimpleNamespace(), settings=RunSettings(), events=path)
    header, *records = _records(path)
    events = [record["event"] for record in records if "event" in record]
    assert [(event["state"], event["message"]) for event in events] == [
        ("running", "Checking run inputs"), ("refused", "Setup refused")]
    assert _kind(records[-1]) == "readiness"
    report = records[-1]["readiness"]
    assert (report["status"], report["exit_code"]) == ("refused", 2)
    assert report["items"][0]["status"] == "missing_setup"


def test_an_observer_beside_the_file_receives_what_the_file_holds(tmp_path, monkeypatch):
    """The observer receives every display event the file holds, in the same
    order, and a later call without events writes no file."""
    _, _, _, settings = harness.setup_case(tmp_path)
    monkeypatch.setattr(phases, "run_target_phases", _starter_only)
    path = tmp_path / "events.jsonl"
    seen = []
    request = RunRequest(("module.py",), stop_after_phase="starter")
    assert orchestrator.run_pipeline(request, settings=settings, event_sink=seen.append, events=path) == 0
    written = [record["event"] for record in _records(path) if "event" in record]
    assert [(event.operation, event.state, event.message) for event in seen] == [
        (event["operation"], event["state"], event["message"]) for event in written]
    before = path.read_bytes()
    _, _, _, again = harness.setup_case(tmp_path / "again")
    assert orchestrator.run_pipeline(request, settings=again, event_sink=seen.append) == 0
    assert path.read_bytes() == before
