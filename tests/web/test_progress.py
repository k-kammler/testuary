"""The progress view takes its rows, its log and the end-of-run summary from
Testuary's own view model and presenter."""
from __future__ import annotations

import dataclasses
import io
import json


from testuary.web import _jobs



def _line(record):
    return json.dumps(record)


def _readiness(*items, status="failed", exit_code=1):
    """A setup report with ``items``, as the event file holds it."""
    from testuary.readiness import RunReadiness

    return RunReadiness(
        scope="inspection", status=status, exit_code=exit_code, source_dir=None, execution=None,
        source_files=(), phases=(), model_alias=None, provider=None, model=None,
        credentials_configured=None, output_dir=None, output_mode="fresh", limits=(),
        items=items).as_dict()


def test_the_rows_and_log_are_the_view_models():
    from testuary.presentation import DisplayEvent

    event = DisplayEvent(run_id="r", target="all targets", phase="pipeline", operation="run",
                         state="running", elapsed_s=1.0, message="Run planned", completed=1,
                         total=4, work_unit="phases done", level="run",
                         log_path="C:/work/out/logs/run.log")
    view = _jobs.Progress()
    assert not view.add(_line({"schema_version": 2, "testuary": "test"}))
    assert view.add(_line({"event": dataclasses.asdict(event)}))
    snapshot = view.snapshot()
    assert snapshot["layers"][0]["name"] == "Overall"
    assert snapshot["layers"][0]["bar"]["figures"].startswith(" 25%  1 of 4 phases done")
    assert snapshot["message"] == "Run planned" and snapshot["log_path"] == event.log_path
    assert view.log and "Run planned" in view.log[0]
    # A line that is not Testuary's event record changes nothing.
    assert not _jobs.Progress().add(_line(dataclasses.asdict(event)))


def test_the_summary_is_testuarys_own_text():
    from rich.console import Console

    from testuary.presentation import DisplayEvent, Metric, TerminalPresenter, Usage
    from testuary.presentation.terminal import FileCompletion, RunCompletion, present_completion

    completion = RunCompletion(
        files=(FileCompletion("calc/ops.py", "complete", suite=("tests/test_ops.py",),
                              metrics=(Metric("mutation score", 87.7193, "%"),),
                              usage=Usage(logical_requests=20, physical_requests=20)),
               FileCompletion("calc/io.py", "no_complete_suite", reason="below the minimum")),
        results_directory="out", log_path="out/logs/run.log", resume_command="testuary calc --resume")
    last = DisplayEvent(run_id="r", target="all targets", phase="pipeline", operation="run",
                        state="succeeded", elapsed_s=9.0, message="Run completed", level="run",
                        provenance="synthetic")
    view = _jobs.Progress()
    view.add(_line({"event": dataclasses.asdict(last)}))
    assert view.add(_line({"completion": dataclasses.asdict(completion)}))
    stream = io.StringIO()
    console = Console(file=stream, width=10000, force_terminal=False, color_system=None)
    with TerminalPresenter("plain", console=console) as presenter:
        presenter.view.add(last)
        present_completion(dataclasses.replace(completion, resume_command=None), presenter=presenter)
    assert view.summary == stream.getvalue().splitlines()
    assert any("87.7% [demo]" in line for line in view.summary)
    # The page offers Continue in place of the console's command.
    assert not any(line.startswith("Next step") for line in view.summary)


def _event(**fields):
    from testuary.presentation import DisplayEvent

    values = dict(run_id="r", target="calc/ops.py", phase="coverage", operation="phase/run-1",
                  state="running", elapsed_s=1.0)
    values.update(fields)
    return DisplayEvent(**values)


def _plan():
    from testuary.presentation.events import PlannedLimit, PlannedPhase, PlannedStep, RunPlan

    loop = tuple(PlannedStep(step, step.title()) for step in ("read", "measure", "revise", "validate", "assess"))
    return RunPlan(files=("calc/ops.py",), runs=("r0",), phases=(
        PlannedPhase("starter", "Starter-suite generation"),
        PlannedPhase("coverage", "Coverage phase", steps=loop, limits=(PlannedLimit("goal", 90.0, "%"),), cycle=2),
        PlannedPhase("mutation", "Mutation phase", "off", steps=loop, cycle=2)),
        finishing=PlannedStep("finishing", "Finishing"))


def test_the_diagram_is_the_plan_in_the_states_the_run_shows(tmp_path):
    """The run's plan from the event file, each phase and step in the
    state its latest observation shows, else the plan's."""
    from testuary.presentation.view import FINISHING

    view = _jobs.Progress(tmp_path)
    assert view.snapshot()["diagram"] is None
    assert view.add(_line({"plan": dataclasses.asdict(_plan())}))
    view.add(_line({"event": dataclasses.asdict(_event(phase="starter", level="phase", state="succeeded"))}))
    view.add(_line({"event": dataclasses.asdict(_event(level="phase"))}))
    for operation, state in (("step/read", "succeeded"), ("step/measure", "running")):
        view.add(_line({"event": dataclasses.asdict(_event(operation=operation, level="step", state=state))}))
    diagram = view.snapshot()["diagram"]
    assert [(phase["phase"], phase["state"]) for phase in diagram["phases"]] == [
        ("starter", "succeeded"), ("coverage", "running"), ("mutation", "off")]
    coverage = diagram["phases"][1]
    assert coverage["cycle"] == 2 and coverage["limits"] == [{"name": "goal", "value": 90.0, "unit": "%"}]
    assert [step["state"] for step in coverage["steps"]] == [
        "succeeded", "running", "planned", "planned", "planned"]
    assert diagram["finishing"]["state"] == "planned" and diagram["target"] == "calc/ops.py"
    # A finished loop that did not need a step, and an ended run, mark what never ran.
    view.add(_line({"event": dataclasses.asdict(_event(level="phase", state="succeeded"))}))
    view.add(_line({"event": dataclasses.asdict(_event(
        phase="pipeline", operation="run", level="run", target="all targets", state="succeeded"))}))
    ended = view.snapshot()["diagram"]
    assert [step["state"] for step in ended["phases"][1]["steps"]][2:] == ["skipped"] * 3
    assert ended["finishing"]["state"] == "skipped"
    view.add(_line({"event": dataclasses.asdict(_event(phase="finishing", operation=FINISHING,
                                                       level="step", state="succeeded"))}))
    assert view.snapshot()["diagram"]["finishing"]["state"] == "succeeded"


def test_a_step_a_limit_ended_names_the_limit_in_the_diagram(tmp_path):
    """The step a limit ended carries its event's words for the limit; every
    other step carries none."""
    view = _jobs.Progress(tmp_path)
    view.add(_line({"plan": dataclasses.asdict(_plan())}))
    view.add(_line({"event": dataclasses.asdict(_event(level="phase"))}))
    view.add(_line({"event": dataclasses.asdict(_event(operation="step/read", level="step",
                                                       state="succeeded"))}))
    view.add(_line({"event": dataclasses.asdict(_event(
        operation="step/measure", level="step", state="limit_reached",
        message="Token limit reached"))}))
    steps = view.snapshot()["diagram"]["phases"][1]["steps"]
    assert [(step["state"], step["reached"]) for step in steps[:3]] == [
        ("succeeded", None), ("limit_reached", "Token limit reached"), ("planned", None)]


def test_the_diagram_carries_each_phases_end_and_its_counters(tmp_path):
    """Each phase's end group and the view model's counters: the starter
    suites generated and measured on their steps, a loop's requests against
    its limit, and its measured values once it has ended."""
    from testuary.presentation import Metric, Usage
    from testuary.presentation.events import PlannedPhase, PlannedStep

    starter = tuple(PlannedStep(step, step.title()) for step in ("generate", "measure", "consolidation"))
    loop = tuple(PlannedStep(step, step.title()) for step in ("read", "revise", "validate", "consolidation"))
    plan = dataclasses.replace(_plan(), phases=(
        PlannedPhase("starter", "Starter phase", steps=starter, cycle=2, end=2),
        PlannedPhase("coverage", "Coverage phase", steps=loop, cycle=1, end=3)))
    view = _jobs.Progress(tmp_path)
    view.add(_line({"plan": dataclasses.asdict(plan)}))
    view.add(_line({"event": dataclasses.asdict(_event(phase="starter", level="phase"))}))
    view.add(_line({"event": dataclasses.asdict(_event(
        phase="starter", operation="suite-processing", completed=2, total=3))}))
    view.add(_line({"event": dataclasses.asdict(_event(level="phase"))}))
    view.add(_line({"event": dataclasses.asdict(_event(
        operation="request/4", state="succeeded", usage=Usage(logical_requests=4, logical_request_limit=20)))}))
    first, loop_phase = view.snapshot()["diagram"]["phases"]
    assert (first["cycle"], first["end"], loop_phase["cycle"], loop_phase["end"]) == (2, 2, 1, 3)
    assert [step.get("count") for step in first["steps"]] == ["2 of 3 suites", None, None]
    assert (loop_phase["count"], loop_phase["values"]) == ("model requests 4 of 20", [])
    view.add(_line({"event": dataclasses.asdict(_event(
        level="phase", state="succeeded", metrics=(Metric("line coverage", 98.25, "%"),)))}))
    assert view.snapshot()["diagram"]["phases"][1]["values"] == ["line coverage: 98.2% [measured]"]


def test_finishing_says_what_it_does_in_testuarys_words(tmp_path):
    """The page's finishing line takes Finishing's label from the
    plan and its purpose from the view model, which the console shows too."""
    from testuary.presentation.view import FINISHING_PURPOSE

    view = _jobs.Progress(tmp_path)
    view.add(_line({"plan": dataclasses.asdict(_plan())}))
    finishing = view.snapshot()["diagram"]["finishing"]
    assert finishing["label"] == "Finishing" and finishing["purpose"] == FINISHING_PURPOSE


def test_finishing_carries_its_step_and_its_values(tmp_path):
    """Finishing's box gets the step the run observes in it and, once it has
    ended, the values it measured, as a loop's box does."""
    from testuary.presentation import Metric
    from testuary.presentation.view import FINISHING

    def finishing(operation, label, state="running", metrics=()):
        event = _event(phase="finishing", operation=operation, level="step", label=label, state=state,
                       metrics=metrics)
        view.add(_line({"event": dataclasses.asdict(event)}))
        return view.snapshot()["diagram"]["finishing"]

    view = _jobs.Progress(tmp_path)
    view.add(_line({"plan": dataclasses.asdict(_plan())}))
    assert finishing(FINISHING, "Finishing")["steps"] == []
    shown = finishing("step/delivered-measurement", "Checking the delivered suite")
    assert [(step["label"], step["state"]) for step in shown["steps"]] == [("Checking the delivered suite", "running")]
    assert shown["values"] == []
    finishing("step/delivered-measurement", "Checking the delivered suite", "succeeded")
    shown = finishing(FINISHING, "Finishing", "succeeded",
                      (Metric("line coverage", 100.0, "%"), Metric("mutation score", None, "%")))
    assert [(step["label"], step["state"]) for step in shown["steps"]] == [("Checking the delivered suite", "succeeded")]
    assert shown["values"] == ["line coverage: 100% [measured]"]


def test_a_failed_runs_cause_and_a_log_path_below_the_project(tmp_path):
    view = _jobs.Progress(tmp_path)
    log = tmp_path / ".testuary" / "results" / "m" / "logs" / "run.log"
    view.add(_line({"event": dataclasses.asdict(_event(
        phase="pipeline", operation="run", level="run", target="all targets", log_path=str(log)))}))
    view.add(_line({"event": dataclasses.asdict(_event(phase="starter", level="phase"))}))
    view.add(_line({"event": dataclasses.asdict(_event(phase="starter", operation="request/1", state="failed",
                                                       message="The model service refused. It said 401."))}))
    view.add(_line({"event": dataclasses.asdict(_event(
        phase="pipeline", operation="run", level="run", target="all targets", state="failed",
        message="Run ended with an execution error", log_path=str(log)))}))
    snapshot = view.snapshot()
    assert snapshot["cause"] == "The model service refused."
    phase = next(layer for layer in snapshot["layers"] if layer["name"] == "Phase")
    assert "FAILED" in phase["text"]
    path = next(layer for layer in snapshot["layers"] if layer["name"] == "Log")
    assert (path["path"], path["short"]) == (str(log), ".testuary/results/m/logs/run.log")


def test_a_refused_runs_cause_is_its_setup_reports_blocking_reason():
    view = _jobs.Progress()
    view.add(_line({"event": dataclasses.asdict(_event(
        phase="pipeline", operation="run", level="run", target="all targets", state="failed",
        message="Configuration refused"))}))
    assert view.snapshot()["cause"] == "Configuration refused"
    from testuary.readiness import ReadinessItem

    assert not view.add(_line({"readiness": _readiness(
        ReadinessItem("configuration.model", True, "verified", reason="fine"),
        ReadinessItem("configuration.settings", True, "failed",
                      reason="output directory is occupied: out; no previous files were removed"))}))
    assert view.snapshot()["cause"] == "output directory is occupied: out; no previous files were removed"


def test_a_refused_runs_cause_is_the_problem_its_setup_summary_lists_first():
    """The cause is the problem the setup report's summary lists first, as
    the terminal shows it: a required item, the most severe first."""
    from testuary.readiness import ReadinessItem

    view = _jobs.Progress()
    view.add(_line({"event": dataclasses.asdict(_event(
        phase="pipeline", operation="run", level="run", target="all targets", state="failed",
        message="Setup refused"))}))
    view.add(_line({"readiness": _readiness(
        ReadinessItem("native.python.item.cache", False, "failed", reason="an optional cache failed"),
        ReadinessItem("native.python.item.runtime", True, "missing_setup", reason="no interpreter was found"),
        ReadinessItem("configuration", True, "refused", reason="[coverage] goal: expected a percentage"),
        status="refused", exit_code=2)}))
    assert view.snapshot()["cause"] == "[coverage] goal: expected a percentage"


def test_a_refused_run_is_over_and_names_its_refused_item(tmp_path):
    """Testuary ends a configuration refusal with the state "refused", no
    failure, and its report's item with the status "refused". The diagram
    takes the run as over, and the cause is that item's reason."""
    reason = "[coverage] goal: expected a percentage from 0 to 100"
    view = _jobs.Progress(tmp_path)
    view.add(_line({"plan": dataclasses.asdict(_plan())}))
    from testuary.readiness import ReadinessItem

    view.add(_line({"readiness": _readiness(
        ReadinessItem("configuration", True, "refused", reason=reason), status="refused", exit_code=2)}))
    view.add(_line({"event": dataclasses.asdict(_event(
        phase="pipeline", operation="run", level="run", target="all targets", state="refused",
        message="Configuration refused"))}))
    snapshot = view.snapshot()
    assert snapshot["cause"] == reason
    assert snapshot["diagram"]["finishing"]["state"] == "skipped"
    assert snapshot["layers"][0]["tone"] == "refused"


def test_a_loop_that_ran_no_step_says_why_in_its_own_words(tmp_path):
    """A loop that ended at its first measurement, without any step, shows
    the sentence its end carries, such as the goal its inherited suite met.
    A loop that ran its steps shows none."""
    met = "The suite it received already met the goal, so no model request was needed"
    view = _jobs.Progress(tmp_path)
    view.add(_line({"plan": dataclasses.asdict(_plan())}))
    view.add(_line({"event": dataclasses.asdict(_event(level="phase", state="succeeded", message=met))}))
    view.add(_line({"event": dataclasses.asdict(_event(
        phase="pipeline", operation="run", level="run", target="all targets", state="succeeded"))}))
    phases = view.snapshot()["diagram"]["phases"]
    assert phases[1]["note"] == met and phases[0]["note"] is None


def test_a_loop_that_ran_only_its_phase_end_still_says_why(tmp_path):
    """The phase end runs whether or not the loop did: a loop whose
    consolidation ran but none of its own steps still shows its sentence."""
    from testuary.presentation.events import PlannedPhase, PlannedStep

    met = "The suite it received already met the goal, so no model request was needed"
    loop = tuple(PlannedStep(step, step.title()) for step in ("read", "revise", "consolidation"))
    plan = dataclasses.replace(_plan(), phases=(
        PlannedPhase("coverage", "Coverage phase", steps=loop, cycle=1, end=2),))
    view = _jobs.Progress(tmp_path)
    view.add(_line({"plan": dataclasses.asdict(plan)}))
    view.add(_line({"event": dataclasses.asdict(_event(
        operation="step/consolidation", level="step", state="succeeded"))}))
    view.add(_line({"event": dataclasses.asdict(_event(level="phase", state="succeeded", message=met))}))
    view.add(_line({"event": dataclasses.asdict(_event(
        phase="pipeline", operation="run", level="run", target="all targets", state="succeeded"))}))
    (phase,) = view.snapshot()["diagram"]["phases"]
    assert [step["state"] for step in phase["steps"]] == ["skipped", "skipped", "succeeded"]
    assert phase["note"] == met


def test_a_step_that_ended_long_ago_keeps_its_state(tmp_path):
    """The view model keeps only recent finished work in view. A step that
    ended before many other observations still shows as done."""
    view = _jobs.Progress(tmp_path)
    view.add(_line({"plan": dataclasses.asdict(_plan())}))
    view.add(_line({"event": dataclasses.asdict(_event(level="phase"))}))
    view.add(_line({"event": dataclasses.asdict(_event(operation="step/read", level="step", state="succeeded"))}))
    for index in range(200):
        view.add(_line({"event": dataclasses.asdict(_event(
            operation=f"tool/{index}", level="operation", state="succeeded"))}))
    view.add(_line({"event": dataclasses.asdict(_event(
        phase="pipeline", operation="run", level="run", target="all targets", state="succeeded"))}))
    steps = view.snapshot()["diagram"]["phases"][1]["steps"]
    assert steps[0]["state"] == "succeeded" and steps[1]["state"] == "skipped"


def test_an_ended_runs_run_dir_is_its_logs_dir(tmp_path):
    """The run directory is the directory of its log, as its latest run
    event names it. Without a log path there is none."""
    events = tmp_path / "events.jsonl"
    log = tmp_path / "out" / "logs" / "run.log"
    events.write_text("\n".join(json.dumps(record) for record in (
        {"schema_version": 2, "testuary": "test"},
        {"event": {"level": "run", "log_path": ""}},
        {"event": {"level": "step", "log_path": str(tmp_path / "other" / "logs" / "run.log")}},
        {"event": {"level": "run", "log_path": str(log)}})) + "\n", encoding="utf-8")
    assert _jobs.run_dir(events) == tmp_path / "out"
    events.write_text(json.dumps({"event": {"level": "run", "log_path": ""}}) + "\n", encoding="utf-8")
    assert _jobs.run_dir(events) is None


def test_a_rows_tone_is_its_state_and_a_scientific_outcome_a_warning():
    """The page colors a row as the terminal does: by its event's state, and
    a failure the producer marks as an outcome as incomplete, never as a
    failure."""
    assert _jobs._tone(None) is None
    assert _jobs._tone(_event(state="running")) == "running"
    assert _jobs._tone(_event(state="failed")) == "failed"
    assert _jobs._tone(_event(state="failed", failure="outcome")) == "incomplete"


def test_an_event_files_records_are_its_json_objects(tmp_path):
    """The web's readers of an event file share one parser: a line is a
    record when it is a JSON object; a cut, empty or other line is none."""
    assert _jobs.record('{"event": {"level": "run"}}') == {"event": {"level": "run"}}
    assert _jobs.record(b'{"completion": {}}') == {"completion": {}}
    assert [_jobs.record(line) for line in ("", "{cut", "[1, 2]", "3")] == [None] * 4
    events = tmp_path / "events.jsonl"
    events.write_text('{"schema_version": 2}\n{cut\n[1]\n{"event": {"level": "run"}}\n', encoding="utf-8")
    assert list(_jobs.records(events)) == [{"schema_version": 2}, {"event": {"level": "run"}}]


def test_the_view_carries_the_runs_clock_and_when_the_current_step_began(tmp_path):
    """The run's clock at its newest observation, which a running row's time
    follows on the page, and when each step under way began, which the
    diagram's current step counts from. A step that ended carries none."""
    view = _jobs.Progress(tmp_path)
    view.add(_line({"plan": dataclasses.asdict(_plan())}))
    view.add(_line({"event": dataclasses.asdict(_event(level="phase", elapsed_s=10.0))}))
    view.add(_line({"event": dataclasses.asdict(_event(operation="step/read", level="step",
                                                       state="succeeded", elapsed_s=12.0))}))
    view.add(_line({"event": dataclasses.asdict(_event(operation="step/measure", level="step",
                                                       elapsed_s=99.0))}))
    view.add(_line({"event": dataclasses.asdict(_event(operation="mutants", work_unit="mutants",
                                                       completed=4, total=37, elapsed_s=127.5))}))
    snapshot = view.snapshot()
    assert snapshot["clock_s"] == 127.5
    steps = snapshot["diagram"]["phases"][1]["steps"]
    assert [(step["step"], step["since_s"]) for step in steps[:3]] == [
        ("read", None), ("measure", 99.0), ("revise", None)]


def test_a_cause_the_model_service_gave_is_marked_for_the_page():
    """The sentence of a request given up after its retries marks the cause as
    the model service's, for the page's link to the Models page. Another
    cause carries no mark."""
    from testuary import model_requests

    sentence = model_requests.exhausted_text(
        model_requests.TransportExhausted("gone", TimeoutError("slow"), attempt=3), "http://127.0.0.1:9/v1")
    view = _jobs.Progress()
    view.add(_line({"event": dataclasses.asdict(_event(
        phase="pipeline", operation="run", level="run", target="all targets"))}))
    view.add(_line({"event": dataclasses.asdict(_event(phase="starter", level="phase"))}))
    view.add(_line({"event": dataclasses.asdict(_event(phase="starter", operation="request/1", state="failed",
                                                       message=sentence))}))
    view.add(_line({"event": dataclasses.asdict(_event(
        phase="pipeline", operation="run", level="run", target="all targets", state="failed",
        message="Run ended with an execution error"))}))
    snapshot = view.snapshot()
    assert snapshot["cause"].startswith("The model service at http://127.0.0.1:9/v1 did not answer in time")
    assert snapshot["cause_kind"] == "service"
    other = _jobs.Progress()
    other.add(_line({"event": dataclasses.asdict(_event(
        phase="pipeline", operation="run", level="run", target="all targets", state="failed",
        message="Configuration refused"))}))
    assert other.snapshot()["cause_kind"] is None
