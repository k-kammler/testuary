"""Pipeline observers do not own results, limits or persistence."""
from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from io import StringIO
from types import SimpleNamespace

import pytest
from agent_log import console_records
from rich.console import Console

from testuary import orchestrator
from testuary.agent import _helpers
from testuary import config
from testuary import _progress
from testuary._progress import RunProgress, demonstration_display
from testuary import target_phases as phases
from testuary.presentation import Metric, TerminalPresenter
from testuary.settings import RunSettings
import test_orchestrator_phases as harness


@pytest.mark.parametrize("status", [0, 1])
def test_pipeline_observer_does_not_change_return_or_caller_arguments(monkeypatch, status):
    events = []
    args = SimpleNamespace()
    seen = []
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda args, languages: seen.append(args) or status)
    result = orchestrator.run_pipeline(args, settings=RunSettings(), event_sink=events.append)
    assert result == status
    assert vars(args) == {}
    assert seen[0]._progress is not None
    assert [event.state for event in events] == ["running", "succeeded" if status == 0 else "failed"]
    assert len({event.run_id for event in events}) == 1
    assert all(event.completed is None and event.total is None and event.usage is None for event in events)
    assert config.active_settings() is None


@pytest.mark.parametrize("exception", [RuntimeError("private provider detail"), KeyboardInterrupt()])
def test_pipeline_failure_or_cancellation_is_observed_and_propagates(monkeypatch, exception):
    def fail(args, languages):
        raise exception

    events = []
    monkeypatch.setattr(orchestrator, "_run_pipeline", fail)
    with pytest.raises(type(exception)) as caught:
        orchestrator.run_pipeline(SimpleNamespace(), settings=RunSettings(), event_sink=events.append)
    assert caught.value is exception
    assert events[-1].state == ("cancelled" if isinstance(exception, KeyboardInterrupt) else "failed")
    assert "private provider detail" not in events[-1].message
    assert config.active_settings() is None


def test_broken_observer_does_not_retry_or_fail_execution(monkeypatch, capsys):
    calls = []
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda *a: calls.append("pipeline") or 0)

    def fail(event):
        calls.append("observer")
        raise RuntimeError("private observer detail")

    assert orchestrator.run_pipeline(SimpleNamespace(), settings=RunSettings(), event_sink=fail) == 0
    assert calls == ["observer", "pipeline"]
    err = capsys.readouterr().err
    assert "Display observer failed" in err
    assert "private observer detail" not in err


def test_sequential_runs_do_not_retain_the_previous_observer(monkeypatch):
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda *a: 0)
    first, second = [], []
    args = SimpleNamespace()
    orchestrator.run_pipeline(args, settings=RunSettings(), event_sink=first.append)
    orchestrator.run_pipeline(args, settings=RunSettings(), event_sink=second.append)
    orchestrator.run_pipeline(args, settings=RunSettings())
    assert len(first) == len(second) == 2
    assert first[0].run_id != second[0].run_id


@pytest.mark.parametrize("provenance", ["observed", "recorded", "derived", "synthetic"])
def test_demo_observer_captures_provenance_for_worker_events(monkeypatch, provenance):
    clock = [20.0]
    monkeypatch.setattr(_progress, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    events = []
    scripted_timing = Metric("suite runtime", 43200.0, "s")
    with demonstration_display():
        progress = RunProgress(events.append)
    clock[0] = 22.5
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(progress.emit, target="fixture", phase="finishing", operation="benchmark",
                    state="succeeded", metrics=(scripted_timing,), provenance=provenance).result()
    assert events[0].provenance == "synthetic"
    assert events[0].elapsed_s == 2.5
    assert events[0].metrics == (scripted_timing,)

    ordinary = RunProgress(events.append)
    ordinary.emit(target="fixture", phase="finishing", operation="benchmark",
                  state="succeeded", provenance=provenance)
    assert events[1].provenance == provenance


def test_nested_demo_scope_restores_its_parent_and_does_not_leak_to_another_thread():
    events = []

    def observe():
        RunProgress(events.append).emit(target="fixture", phase="starter", operation="seed",
                                        state="running")

    with demonstration_display():
        with demonstration_display():
            observe()
        observe()
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(observe).result()
    observe()
    assert [event.provenance for event in events] == ["synthetic", "synthetic", "observed", "observed"]


@pytest.mark.parametrize("width", [40, 100])
def test_every_demo_frame_keeps_its_badge_after_the_banner_is_gone(width):
    console = Console(file=StringIO(), width=width, height=25, force_terminal=True,
                      color_system=None)
    with TerminalPresenter("detailed", console=console) as presenter, demonstration_display():
        progress = RunProgress(presenter)
        for phase, state in [("starter", "running"), ("starter", "succeeded"),
                             ("coverage", "running"), ("mutation", "failed")]:
            progress.emit(target="fixture", phase=phase, operation="phase", state=state,
                          metrics=(Metric("mutation score", 75.0, "%"),))
            with console.capture() as frame:
                console.print(presenter.frame())
            assert "demo playback" in " ".join(frame.get().split())
    assert "[demo]" in console.file.getvalue()


def test_plain_demo_events_keep_the_badge_without_a_banner():
    stream = StringIO()
    with TerminalPresenter("plain", stream=stream, width=100) as presenter, demonstration_display():
        progress = RunProgress(presenter)
        for state in ["running", "succeeded"]:
            progress.emit(target="fixture", phase="coverage", operation="phase", state=state)
    states = [line for line in stream.getvalue().splitlines()
              if "RUNNING" in line or "SUCCEEDED" in line]
    assert len(states) == 2
    assert all("[demo]" in line for line in states)


def test_start_event_follows_the_start_checkpoint(monkeypatch, tmp_path, default_run):
    order = []
    ctx = harness._make_ctx(tmp_path, progress=RunProgress(lambda event: order.append(event)))
    harness._observe_checkpoints(monkeypatch, start=lambda phase: order.append("persisted"))
    phases._start_phase(ctx, "coverage")
    assert order[0] == "persisted"
    assert order[1].state == "running"
    assert order[1].phase == "coverage"


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
@pytest.mark.parametrize("value", [0.0, None])
def test_restored_scores_are_marked_reused_without_new_usage(tmp_path, phase, value):
    events = []
    ctx = harness._make_ctx(tmp_path, progress=RunProgress(events.append))
    result = phases.TestPipelineResult(
        starter_result=phases.starters.StarterResult(baseline_line_coverage=value),
        test_result=harness._coverage_result(line_coverage=value),
        mutation_result=harness._mutation_result(mutation_score=value),
    )
    phases._observe_restored_phase(ctx, phase, result)
    event = events[0]
    assert event.provenance == "recorded" and event.state == "succeeded"
    assert event.metrics[0].value == value
    assert event.metrics[0].basis == ("reused" if value is not None else "unavailable")
    assert event.usage is None and event.total is None


def test_completion_event_follows_persistence_and_keeps_zero_and_unknown_distinct(monkeypatch, tmp_path, default_run):
    marker = tmp_path / "completion.json"
    events = []

    def checkpoint(phase, success, artifacts, metrics):
        assert success
        marker.write_text(json.dumps(metrics), encoding="utf-8")

    def observe(event):
        assert marker.is_file()
        events.append(event)

    ctx = harness._make_ctx(tmp_path, progress=RunProgress(observe))
    harness._observe_checkpoints(monkeypatch, done=checkpoint)
    (ctx.work_tests / "test_foo.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    result = phases.TestPipelineResult(
        test_file_name="test_foo.py", phase_outcomes={"coverage": "ok"},
        test_result=harness._coverage_result(line_coverage=0.0, branch_coverage=None),
    )
    phases._checkpoint_phase(ctx, result, "coverage")
    assert len(events) == 1
    assert events[0].state == "succeeded"
    values = {metric.name: metric for metric in events[0].metrics}
    assert values["line coverage"].value == 0.0
    assert values["line coverage"].basis == "measured"
    assert values["branch coverage"].value is None
    assert values["branch coverage"].basis == "unavailable"


def _finishing_measurement(mutation_score):
    from testuary.measurement import FinishingMetrics

    return phases.TestPipelineResult(
        test_file_name="test_foo.py", phase_outcomes={"finishing": "ok"},
        finishing_result=harness._metrics_result(
            coverage_final_line=100.0, coverage_final_branch=100.0,
            finishing=FinishingMetrics(mutation_score_full=mutation_score)))


@pytest.mark.parametrize("score", [87.72, 0.0, None])
def test_the_finishing_measurement_shows_its_mutation_score_and_records_as_before(
        monkeypatch, tmp_path, score, default_run):
    """The finishing event carries every final metric finishing measured,
    the run log's "final standard suite" score included. The recorded metrics
    stay as they were: the score is shown, not recorded here."""
    recorded, events = [], []
    ctx = harness._make_ctx(tmp_path, progress=RunProgress(events.append))
    harness._observe_checkpoints(monkeypatch, done=lambda phase, success, artifacts, metrics:
                                 recorded.append(dict(metrics)))
    (ctx.work_tests / "test_foo.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    phases._checkpoint_phase(ctx, _finishing_measurement(score), "finishing")
    assert "final_mutation_score" not in recorded[0]
    assert set(recorded[0]) == {"final_line_coverage", "final_branch_coverage",
                                "finishing_settings"}
    values = {metric.name: metric for metric in events[-1].metrics}
    assert list(values) == ["line coverage", "branch coverage", "mutation score"]
    assert values["mutation score"].value == score
    assert values["mutation score"].unit == "%"
    assert values["mutation score"].basis == ("measured" if score is not None else "unavailable")
    assert values["mutation score"].goal is None


def test_checkpoint_failure_emits_no_success_event(monkeypatch, tmp_path):
    def fail(*args):
        raise OSError("cannot persist")

    events = []
    ctx = harness._make_ctx(tmp_path, progress=RunProgress(events.append))
    harness._observe_checkpoints(monkeypatch, done=fail)
    result = phases.TestPipelineResult(phase_outcomes={"coverage": "failed"})
    with pytest.raises(OSError, match="cannot persist"):
        phases._checkpoint_phase(ctx, result, "coverage")
    assert events == []


def test_an_invalid_display_value_does_not_change_canonical_execution(tmp_path, default_run):
    events = []
    ctx = harness._make_ctx(tmp_path, progress=RunProgress(events.append))
    phases._display_phase(ctx, "coverage", "succeeded", metrics={"coverage": float("nan")})
    assert events[0].metrics[0].value is None
    assert events[0].metrics[0].basis == "failed"


def test_failed_phase_has_no_headline_scores(tmp_path, default_run):
    events = []
    ctx = harness._make_ctx(tmp_path, progress=RunProgress(events.append))
    result = phases.TestPipelineResult(phase_outcomes={"coverage": "failed"})
    phases._checkpoint_phase(ctx, result, "coverage")
    assert events[0].state == "failed"
    assert all(metric.value is None for metric in events[0].metrics)


def test_plain_presenter_does_not_redirect_stdout(monkeypatch, capsys):
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda *a: print("machine output") or 0)
    with TerminalPresenter("plain", no_color=True) as presenter:
        orchestrator.run_pipeline(SimpleNamespace(), settings=RunSettings(), event_sink=presenter)
    streams = capsys.readouterr()
    assert streams.out == "machine output\n"
    assert "SUCCEEDED" in streams.err
    assert "\x1b" not in streams.err


@pytest.mark.parametrize("options", [{}, {"no_color": True}, {"display": "plain"}])
def test_the_agent_logger_respects_redirects_and_display_preferences(monkeypatch, tmp_path, capsys, options):
    logger = _helpers.AgentLogger("fixture")
    logger.init()
    path = logger.enable_streaming(str(tmp_path), "log.json")

    def run(*args):
        logger.log("Diagnostic persisted")
        return 0

    monkeypatch.setattr(orchestrator, "_run_pipeline", run)
    orchestrator.run_pipeline(SimpleNamespace(**options), settings=RunSettings())
    streams = capsys.readouterr()
    assert "Diagnostic persisted" in streams.out
    assert "\x1b" not in streams.out
    assert console_records(path)[-1]["message"] == "Diagnostic persisted"


@pytest.mark.parametrize("options", [{"no_color": True}, {"display": "plain"}])
def test_the_agent_logger_preferences_apply_to_a_tty_and_are_restored(monkeypatch, capsys, options):
    import sys

    logger = _helpers.AgentLogger("fixture")
    monkeypatch.setattr(sys.stdout, "isatty", lambda: True)
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda *a: logger.log("during") or 0)
    orchestrator.run_pipeline(SimpleNamespace(**options), settings=RunSettings())
    during = capsys.readouterr().out
    assert "\x1b" not in during
    logger.log("after")
    assert "\x1b" in capsys.readouterr().out


@pytest.mark.parametrize("display", [False, True])
def test_actual_phase_orchestration_keeps_results_and_checkpoint_order(monkeypatch, tmp_path, display, default_run):
    events, persisted = [], []
    progress = RunProgress(events.append) if display else None
    if progress is not None:
        progress.plan(files=1, phases=4, repetitions=1)
    ctx = harness._make_ctx(tmp_path, progress=progress)
    harness._observe_checkpoints(monkeypatch, start=lambda phase: persisted.append((phase, "start")),
                                 done=lambda phase, success, *a: persisted.append((phase, success)))
    harness._green_pipeline(monkeypatch, ctx)
    monkeypatch.setattr(phases.measurement, "run_finishing", lambda *a, **k:
                        harness.FinishingMetrics(mutation_score_full=75.0))
    monkeypatch.setattr(phases, "_verify_restored_phase_validity", lambda *a: (True, ""))
    result = phases.run_target_phases(ctx)
    assert result.final_line_coverage == 92.0
    assert result.final_mutation_score == 75.0
    assert [(phase, state) for phase, state in persisted] == [
        (phase, state) for phase in ("starter", "coverage", "mutation", "finishing")
        for state in ("start", True)
    ]
    if display:
        # Four phases, counted. Finishing's events are at the step level.
        phase_events = [event for event in events if event.level == "phase"]
        assert [(event.phase, event.state) for event in phase_events] == [
            (phase, state) for phase in ("starter", "coverage", "mutation")
            for state in ("running", "succeeded")
        ]
        for phase in ("starter", "coverage", "mutation"):
            assert len({event.identity for event in phase_events if event.phase == phase}) == 1
        assert [(event.completed, event.total) for event in phase_events] == [
            (0, 4), (1, 4), (1, 4), (2, 4), (2, 4), (3, 4)]
        assert [(event.completed, event.total) for event in events if event.level == "run"] == [
            (1, 4), (2, 4), (3, 4), (4, 4)]
        # The other events are named steps that start and end inside their phase.
        assert {event.level for event in events} == {"run", "phase", "step"}
        steps = [(event.phase, event.operation, event.state) for event in events if event.level == "step"]
        assert steps == [
            ("coverage", "step/consolidation", "running"), ("coverage", "step/consolidation", "succeeded"),
            ("mutation", "step/consolidation", "running"), ("mutation", "step/consolidation", "succeeded"),
            ("finishing", "step/finishing", "running"),
            ("finishing", "step/delivered-measurement", "running"),
            ("finishing", "step/delivered-measurement", "succeeded"),
            ("finishing", "step/finishing", "succeeded")]
        finishing = [event for event in events if event.operation == "step/finishing"]
        assert [(event.label, event.completed, event.total) for event in finishing] == [
            ("Finishing", 3, 4), ("Finishing", 4, 4)]
    else:
        assert events == []


@pytest.mark.parametrize("requests, message", [
    (0, "The suite it received already met the goal, so no model request was needed"),
    (3, "Phase finished")])
def test_a_loop_that_asked_no_model_says_its_inherited_suite_met_the_goal(tmp_path, requests, message, default_run):
    """A loop that ended at its first measurement, before any model
    request, says why it needed none of its steps."""
    events = []
    ctx = harness._make_ctx(tmp_path, progress=RunProgress(events.append))
    (ctx.work_tests / "test_foo.py").write_text("def test_ok(): pass\n", encoding="utf-8")
    result = phases.TestPipelineResult(
        test_file_name="test_foo.py", phase_outcomes={"coverage": "ok"},
        test_result=harness._coverage_result(line_coverage=95.0, logical_requests=requests))
    phases._checkpoint_phase(ctx, result, "coverage")
    assert (events[-1].state, events[-1].message) == ("succeeded", message)
