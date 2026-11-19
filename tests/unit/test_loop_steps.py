"""The loops' steps are display observations: each graph mode has its step
(the start's read and measurement, the cycle's revise, validate and
assess), a run with a display shows them, and the model receives the same
input with or without a display."""
from __future__ import annotations

import re
import shutil
from types import SimpleNamespace

import calc_language
import pytest
import test_run_dirs as run_dir_tests
from testuary import models as model_source
from testuary import orchestrator
from testuary.agent._base import CodingAgent


def test_every_graph_mode_has_a_step():
    agent = SimpleNamespace(forces_initial_read=True, forces_initial_measurement=True,
                            INITIAL_READ_MODE=CodingAgent.INITIAL_READ_MODE,
                            INITIAL_MEASURE_MODE=CodingAgent.INITIAL_MEASURE_MODE,
                            initial_read_tools=lambda reads: reads)
    modes = CodingAgent.tool_mode_sets(agent, [], [], "run_tests", "measure")
    assert set(modes) == set(CodingAgent.MODE_STEPS)
    # Assess is the step of two modes: with and without the edit tool after
    # a STANDS ALONE note.
    assert [step for step, _ in CodingAgent.MODE_STEPS.values()] == [
        "read", "measure", "revise", "validate", "assess", "assess"]


def _calc_run(tmp_path, monkeypatch, sink, coverage="goal = 90\ngoal_metric = line\n"):
    """One whole Calc run with the scripted model through run_pipeline, with
    ``sink`` as its display or none and ``coverage`` as its coverage
    settings. Returns every text the model received."""
    from testuary.__main__ import _build_parser

    source = run_dir_tests._Recording()
    calc_language.register()
    try:
        project = tmp_path / "proj"
        project.mkdir(exist_ok=True)
        (project / "calc.calc").write_text(run_dir_tests.calc.SOURCE, encoding="utf-8")
        (tmp_path / "models.ini").write_text(
            "[scripted]\nprovider = ollama\nmodel = scripted\n", encoding="utf-8")
        (project / ".testuary").mkdir(exist_ok=True)
        (project / ".testuary" / "project.ini").write_text(
            "[project]\nlanguage = calc\n", encoding="utf-8")
        (tmp_path / "config.ini").write_text(
            "[project]\n[starter]\nsuite_count = 3\nattempts_per_suite = 2\n"
            f"[coverage]\n{coverage}"
            "[mutation]\ngoal = 80\n", encoding="utf-8")
        monkeypatch.chdir(project)
        monkeypatch.setenv("TESTUARY_HOME", str(tmp_path))
        monkeypatch.setenv("TESTUARY_TIMING_RUN_COUNT", "1")
        shutil.rmtree(tmp_path / "out", ignore_errors=True)
        args = _build_parser().parse_args([
            "--config", str(tmp_path / "config.ini"), "--model", "scripted",
            "--project-name", "calcproj", "--source-files", "calc.calc",
            "--output-dir", str(tmp_path / "out"), "--no-color"])
        with model_source.installed(source):
            orchestrator.run_pipeline(args, event_sink=sink)
    finally:
        calc_language.unregister()
    return source.seeds, source.texts


def _timeless(texts):
    return [re.sub(r"\d+(\.\d+)?s\b", "<seconds>", text) for text in texts]


def test_a_runs_loop_steps_are_shown_and_the_model_input_is_unchanged(tmp_path, monkeypatch):
    events = []
    shown = _calc_run(tmp_path, monkeypatch, events.append)
    unshown = _calc_run(tmp_path, monkeypatch, None)
    # The same seeds and texts. Only the measured durations in the test
    # runs' reports differ between two runs.
    assert shown[0] == unshown[0]
    assert _timeless(shown[1]) == _timeless(unshown[1]) and shown[1]
    steps = [(event.phase, event.operation, event.state) for event in events
             if event.level == "step" and event.operation in {
                 f"step/{step}" for step, _ in CodingAgent.MODE_STEPS.values()}]
    coverage = [operation for phase, operation, state in steps
                if phase == "coverage" and state == "running"]
    assert coverage[:2] == ["step/read", "step/measure"] and "step/revise" in coverage
    # Every step that starts ends.
    for phase in ("coverage", "mutation"):
        started = sum(1 for p, _, state in steps if p == phase and state == "running")
        ended = sum(1 for p, _, state in steps if p == phase and state != "running")
        assert started == ended
    assert {event.label for event in events if event.operation == "step/revise"} == {"Edit tests"}
    # The starter suites' steps, in the order they run, each ended.
    from testuary import starters

    starter = [(operation, state) for phase, operation, state in (
        (event.phase, event.operation, event.state) for event in events if event.level == "step")
        if phase == "starter"]
    begun = [operation for operation, state in starter if state == "running"]
    order = [f"step/{step}" for step, _ in starters.STEPS]
    assert begun[:3] == order[:3] and begun == sorted(begun, key=order.index)
    assert sum(state == "running" for _, state in starter) == sum(state != "running" for _, state in starter)
    assert {event.label for event in events
            if event.phase == "starter" and event.operation == "step/consolidation"} == {"Consolidation"}
    # The check of the chosen suite under its delivered name is part of it.
    assert begun[-1] == "step/consolidation"
    # The loops' phase end is the same consolidation, under the same name.
    assert {event.label for event in events
            if event.phase in ("coverage", "mutation")
            and event.operation == "step/consolidation"} == {"Consolidation"}
    # The Measure step counts the accepted suites measured, from none to all.
    from testuary.presentation.view import STARTER_MEASUREMENT

    counted = [(event.completed, event.total, event.state) for event in events
               if event.operation == STARTER_MEASUREMENT]
    total = counted[0][1]
    assert total and counted == [(done, total, "succeeded" if done == total else "running")
                                 for done in range(total + 1)]


def test_a_loop_that_ends_at_its_request_limit_reads_limit_reached(tmp_path, monkeypatch):
    """A loop's request limit is an end its settings chose, not a failure:
    the request refused at the limit and the step it ends read
    limit_reached, and nothing in the loop reads failed."""
    events = []
    _calc_run(tmp_path, monkeypatch, events.append,
              coverage="goal = 100\ngoal_metric = line\nmax_requests_per_phase = 1\n")
    ended = [event for event in events if event.phase == "coverage"
             and event.operation.startswith(("request/", "step/")) and event.state != "running"]
    limit = [event for event in ended if event.state == "limit_reached"]
    assert [event.operation.split("/")[0] for event in limit] == ["request", "step"]
    assert limit[0].message == "Request limit reached (1 of 1)"
    assert [event.state for event in ended if event.state == "failed"] == []


@pytest.mark.parametrize("setting, words", [
    ("max_requests_per_phase = 1", "Request limit reached"),
    ("max_total_tokens_per_phase = 1", "Token limit reached"),
])
def test_the_step_a_limit_ends_names_the_limit(tmp_path, monkeypatch, setting, words):
    """The step a limit ends names which limit, in the words of the limit's
    progress line without its count."""
    events = []
    _calc_run(tmp_path, monkeypatch, events.append,
              coverage=f"goal = 100\ngoal_metric = line\n{setting}\n")
    ended = [event.message for event in events if event.phase == "coverage"
             and event.operation.startswith("step/") and event.state == "limit_reached"]
    assert ended == [words]


def test_the_request_a_limit_stops_is_not_shown_as_sent(tmp_path, monkeypatch):
    """The limit stops the loop before its next request is sent. The line
    names the phase's model requests and the limit, not the number of a
    request never sent ("Model request 2" beside "model requests 1 (limit
    1)"). A request that was sent keeps its number."""
    from testuary.presentation.view import RunView

    events = []
    _calc_run(tmp_path, monkeypatch, events.append,
              coverage="goal = 100\ngoal_metric = line\nmax_requests_per_phase = 1\n")
    requests = [event for event in events if event.phase == "coverage"
                and event.operation.startswith("request/") and event.state != "running"]
    limit = [event for event in requests if event.state == "limit_reached"]
    assert len(limit) == 1 and limit[0].usage.logical_requests == 1
    line = RunView().add(limit[0]).line
    assert "Model requests | Request limit reached (1 of 1)" in line and "Model request 2" not in line
    assert [RunView().add(event).line.count("Model request 1 ") for event in requests
            if event.state == "succeeded"] == [1]


def test_a_loop_its_request_limit_stopped_is_a_warning_of_the_end_summary(tmp_path, monkeypatch):
    """A loop the request limit stopped before its goal is named in the end
    summary's warnings, which completion.json keeps, and in the run's
    summary.json, which the web shows. Before, only time limits made
    warnings, and the run read as a complete suite."""
    import json

    _calc_run(tmp_path, monkeypatch, None,
              coverage="goal = 100\ngoal_metric = line\nmax_requests_per_phase = 1\n")
    completion, = (tmp_path / "out").rglob("completion.json")
    files = json.loads(completion.read_text(encoding="utf-8"))["completion"]["files"]
    sentence = ("The coverage phase stopped at its request limit before reaching its goal "
                "(coverage.max_requests_per_phase = 1).")
    assert [warning for item in files for warning in item["warnings"]] == [sentence]
    summary, = (completion.parent).glob("*/r0/summary.json")
    assert [(warning["kind"], warning["setting"], warning["text"]) for warning in json.loads(
        summary.read_text(encoding="utf-8"))["warnings"]] == [
        ("loop_limit", "coverage.max_requests_per_phase", sentence)]


def test_a_phases_steps_follow_each_other_and_an_observer_error_never_reaches_it(monkeypatch):
    """Entering a step ends the one before it, the phase's end ends the last
    (failed for a phase that did not succeed), and a display that raises
    leaves the phase and its result alone."""
    from types import SimpleNamespace

    from testuary import _progress

    shown = []

    def observe(**fields):
        shown.append((fields["operation"], fields["state"]))
        if fields["state"] == "running":
            raise RuntimeError("the display broke")

    monkeypatch.setattr(_progress, "phase_observer", lambda phase: observe)

    def phase():
        _progress.enter_step("generate", "Generate")
        _progress.enter_step("generate", "Generate")
        _progress.enter_step("measure", "Measure")
        return SimpleNamespace(status="cancelled_zero_passing")

    result = _progress.run_with_steps("starter", lambda done: done.status == "ok", phase)
    assert result.status == "cancelled_zero_passing"
    assert shown == [("step/generate", "running"), ("step/generate", "succeeded"),
                     ("step/measure", "running"), ("step/measure", "failed")]
    # Outside a phase that shows its steps, nothing is shown.
    _progress.enter_step("verify", "Verify")
    assert len(shown) == 4
