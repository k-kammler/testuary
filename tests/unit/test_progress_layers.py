"""Layered progress data for the display: known totals, plain step names, the
selected source path as target, and the log's console lines only when asked."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from testuary import orchestrator
from testuary.agent import _helpers
from testuary._progress import (
    RunProgress, console_preferences, echoes_log_line, observe_target, observed_step,
)
from testuary import target_phases as phases


def _progress():
    events = []
    return RunProgress(events.append), events


def test_emit_carries_level_and_label():
    progress, events = _progress()
    progress.emit(target="src/mod.py", phase="pipeline", operation="target", state="running",
                  level="target", label="src/mod.py")
    assert (events[0].level, events[0].label) == ("target", "src/mod.py")


@pytest.mark.parametrize(("files", "repetitions", "unit"), [
    (1, 1, "phases done"), (2, 1, "phases done (2 targets)"),
    (1, 3, "phases done (3 repetitions)"), (2, 3, "phases done (2 targets x 3 repetitions)")])
def test_the_run_count_says_phases_done_and_which_files_and_repetitions_it_spans(
        files, repetitions, unit):
    progress, events = _progress()
    progress.plan(files=files, phases=4, repetitions=repetitions)
    progress.report_run()
    assert (events[-1].completed, events[-1].total, events[-1].work_unit) == (
        0, 4 * files * repetitions, unit)


def test_counts_are_unknown_until_the_run_is_planned():
    progress, events = _progress()
    assert progress.phase_counts("a.py", 1, finished=True)["total"] is None
    progress.report_run()
    assert events == []


def test_only_done_phase_runs_count_and_only_a_complete_skip_tops_up():
    progress, events = _progress()
    progress.plan(files=3, phases=3, repetitions=2)
    assert progress.run_counts() == {"completed": 0, "total": 18,
                                     "work_unit": "phases done (3 targets x 2 repetitions)"}
    assert progress.phase_counts("a.py", 1, finished=True) == {
        "completed": 1, "total": 3, "work_unit": "phases"}
    assert progress.phase_counts("a.py", 2) == {"completed": 0, "total": 3, "work_unit": "phases"}
    # A file that stopped early leaves its other phase runs undone.
    assert progress.file_counts("a.py", finished=True) == {
        "completed": 1, "total": 3, "work_unit": "targets"}
    assert progress.run_counts()["completed"] == 1
    # A file a resume skipped as already complete counts its recorded runs.
    progress.file_counts("b.py", finished=True, already_complete=True)
    assert progress.run_counts()["completed"] == 7
    progress.report_run()
    assert (events[-1].level, events[-1].completed, events[-1].total) == ("run", 7, 18)


def test_steps_report_running_then_their_end_inside_the_target_scope():
    progress, events = _progress()
    with observe_target(progress, "src/mod.py"):
        with observed_step("coverage", "consolidation", "Consolidation"):
            pass
        with pytest.raises(RuntimeError):
            with observed_step("mutation", "consolidation", "Consolidation"):
                raise RuntimeError("merge failed")
    assert [(event.target, event.phase, event.operation, event.level, event.label, event.state)
            for event in events] == [
        ("src/mod.py", "coverage", "step/consolidation", "step", "Consolidation", "running"),
        ("src/mod.py", "coverage", "step/consolidation", "step", "Consolidation", "succeeded"),
        ("src/mod.py", "mutation", "step/consolidation", "step", "Consolidation", "running"),
        ("src/mod.py", "mutation", "step/consolidation", "step", "Consolidation", "failed"),
    ]
    assert all(event.completed is None and event.total is None for event in events)
    with observed_step("finishing", "delivered-measurement", "Measuring the delivered suite"):
        pass
    assert len(events) == 4


# The run plan's phases are the three generation phases; Finishing follows
# them in its step-level events and counts as the fourth.
@pytest.mark.parametrize("stop,reuse,expected", [
    (None, {}, ("starter", "coverage", "mutation")),
    ("starter", {}, ("starter",)),
    (None, {"reuse_finishing": True}, ("starter", "coverage", "mutation")),
    (None, {"reuse_starter": True, "reuse_coverage": True, "reuse_mutation": True,
        "reuse_finishing": True}, ("starter", "coverage", "mutation")),
])
def test_planned_phases_are_the_three_and_follow_the_stop(stop, reuse, expected):
    args = SimpleNamespace(stop_after_phase=stop, **reuse)
    assert orchestrator._display_phases(args) == expected


def test_a_phase_stop_plans_one_repetition():
    assert orchestrator._display_repetitions(SimpleNamespace(stop_after_phase="coverage", runs=3)) == 1
    assert orchestrator._display_repetitions(SimpleNamespace(stop_after_phase=None, runs=3)) == 3
    assert orchestrator._display_repetitions(SimpleNamespace(stop_after_phase=None, runs=1, run=4)) == 1


@pytest.mark.parametrize("stop,counted", [(None, 4), ("finishing", 4), ("mutation", 3), ("starter", 1)])
def test_finishing_counts_as_the_fourth_phase_unless_the_run_stops_before_it(stop, counted):
    assert orchestrator._counted_phases(SimpleNamespace(stop_after_phase=stop)) == counted


def test_finishing_completes_the_count_in_its_step_events(default_run):
    """Finishing's events stay at the step level, carry the file run's phase
    count and complete it: a run is 4 of 4 only after Finishing."""
    from testuary.presentation.view import FINISHING

    progress, events = _progress()
    progress.plan(files=1, phases=4, repetitions=1)
    ctx = SimpleNamespace(progress=progress, workspace="rank1", run_idx=1, display_target="a.py")
    for phase in ("starter", "coverage", "mutation"):
        phases._display_phase(ctx, phase, "succeeded", message="Phase finished")
    assert progress.run_counts()["completed"] == 3
    phases._display_phase(ctx, "finishing", "running", message="Phase started")
    phases._display_phase(ctx, "finishing", "succeeded", message="Phase finished")
    started, finished = [event for event in events if event.phase == "finishing"]
    assert {(event.level, event.operation) for event in (started, finished)} == {("step", FINISHING)}
    assert (started.completed, started.total, started.message) == (3, 4, "Phase started")
    assert (finished.completed, finished.total, finished.message) == (4, 4, "Phase finished")
    assert progress.run_counts()["completed"] == 4


def test_a_reused_finishing_completes_the_count_too(default_run):
    from testuary.presentation.view import FINISHING

    progress, events = _progress()
    progress.plan(files=1, phases=4, repetitions=1)
    ctx = SimpleNamespace(progress=progress, workspace="rank1", run_idx=1, display_target="a.py")
    phases._observe_restored_phase(ctx, "finishing", SimpleNamespace(finishing_result=None))
    assert [(event.level, event.operation, event.completed, event.total)
            for event in events if event.phase == "finishing"] == [("step", FINISHING, 1, 4)]
    assert progress.run_counts()["completed"] == 1


def test_a_failed_phase_does_not_count_as_done():
    progress, events = _progress()
    progress.plan(files=1, phases=4, repetitions=1)
    ctx = SimpleNamespace(progress=progress, workspace="rank1", run_idx=1, display_target="a.py")
    phases._display_phase(ctx, "starter", "failed", message="Phase did not complete")
    assert (events[-1].level, events[-1].completed) == ("phase", 0)
    assert progress.run_counts()["completed"] == 0


@pytest.mark.parametrize("selected,goals", [
    ("branch", (None, 90.0)), ("line", (90.0, None)), ("either", (90.0, 90.0)), ("both", (90.0, 90.0)),
])
def test_phase_events_name_the_source_path_and_the_configured_goals(tmp_path, monkeypatch, selected, goals):
    from testuary import config

    progress, events = _progress()
    progress.plan(files=1, phases=4, repetitions=1)
    configuration = {"test_coverage": {"goal": 90.0, "goal_metric": selected},
                     "test_mutation": {"goal": 80.0}}
    monkeypatch.setattr(config, "get_agent_config",
                        lambda section, parent=None: configuration[section])
    ctx = SimpleNamespace(progress=progress, workspace="rank1_src_mod", run_idx=1,
                          display_target="src/mod.py")
    phases._display_phase(ctx, "coverage", "running", message="Phase started")
    phases._display_phase(ctx, "coverage", "succeeded", message="Phase completed",
                             metrics={"coverage": 75.0, "branch_coverage": 60.0})
    phases._display_phase(ctx, "mutation", "succeeded", metrics={"mutation_score": 70.0})
    started, finished, run, mutation, _ = events
    assert {event.target for event in (started, finished, mutation)} == {"src/mod.py"}
    assert (started.level, started.completed, started.total, started.work_unit) == ("phase", 0, 4, "phases")
    assert (finished.completed, run.level, run.completed, run.total) == (1, "run", 1, 4)
    assert tuple(metric.goal for metric in finished.metrics) == goals
    assert [metric.value for metric in finished.metrics] == [75.0, 60.0]
    assert [(metric.name, metric.goal) for metric in mutation.metrics] == [("mutation score", 80.0)]


def test_workspace_name_remains_the_target_without_a_source_path():
    progress, events = _progress()
    ctx = SimpleNamespace(progress=progress, workspace="rank1_src_mod", run_idx=1)
    phases._display_phase(ctx, "finishing", "running")
    assert events[0].target == "rank1_src_mod"


@pytest.mark.parametrize("line,shown", [
    ("[starter] starter suites: 4", False),
    ("ERROR: target 1 crashed unexpectedly", True),
    ("WARNING: model reported no usage", True),
    ("WARN: git-drift check raised (OSError: x). Skipping.", True),
    ("[mutation] WARNING: HTTP worker setup FAILED", True),
    ("[CRITICAL] mutation measurement INCONCLUSIVE: 0 useful outcomes", True),
    ("[measure_mutation] [CRITICAL] 3/4 mutants timed out", True),
    ("  FATAL: uncaught exception ended the run", True),
    ("WARNINGS are summarized below", False),
])
def test_only_warnings_errors_and_critical_lines_reach_the_console_while_a_display_runs(line, shown):
    with console_preferences(detail_echo=False):
        assert echoes_log_line(line) is shown
        assert echoes_log_line(line, echo=True) is True
    with console_preferences(detail_echo=True):
        assert echoes_log_line(line) is True


def test_outcome_lines_are_echoed_without_the_log_timestamp(capsys):
    from testuary import orchestrator as pipeline

    logger = _helpers.AgentLogger("fixture", agent_name="main")
    logger.init()
    previous = pipeline.get_main_logger()
    pipeline.set_main_logger(logger)
    try:
        with console_preferences(detail_echo=False, no_color=True):
            pipeline.log_print("Targets without a suite that meets the minimums (1):", echo=True)
            pipeline.log_print("[starter] starter suites: 4")
    finally:
        pipeline.set_main_logger(previous)
    printed = capsys.readouterr().out.splitlines()
    assert printed == ["Targets without a suite that meets the minimums (1):"]


def test_saved_logs_keep_every_line_the_console_omits(tmp_path, capsys):
    logger = _helpers.AgentLogger("fixture", agent_name="coverage")
    logger.init()
    with console_preferences(detail_echo=False, no_color=True):
        logger.log("[coverage] measured 80%")
        logger.log("WARNING: the suite timed out once")
    printed = capsys.readouterr().out
    assert "measured 80%" not in printed and "WARNING: the suite timed out once" in printed
    assert [entry["message"] for entry in logger._console_logs] == [
        "[coverage] measured 80%", "WARNING: the suite timed out once"]


def test_the_plain_log_keeps_every_line_with_one_timestamp_style(tmp_path):
    from testuary import _progress

    path = tmp_path / "run.log"
    _progress.write_plain_log("before the log opens")
    assert _progress.open_plain_log(path) == str(path) and _progress.plain_log_path() == str(path)
    try:
        _progress.write_plain_log("\x1b[92m[starter] starter suites: 4\x1b[0m", level=1)
        _progress.write_plain_log("first\nsecond")
    finally:
        _progress.close_plain_log()
    _progress.write_plain_log("after it closes")
    lines = path.read_text(encoding="utf-8").splitlines()
    assert [line.split(" ", 1)[1] for line in lines] == ["  [starter] starter suites: 4", "first", "second"]
    assert all(len(line.split(" ", 1)[0]) == len("00:00:00.000") for line in lines)
    assert _progress.plain_log_path() == ""
    _progress.open_plain_log(path)
    _progress.write_plain_log("a resumed run appends")
    _progress.close_plain_log()
    assert path.read_text(encoding="utf-8").splitlines()[-1].endswith("a resumed run appends")


def test_the_plain_log_times_each_line_from_the_run_start(tmp_path, monkeypatch):
    """run.log's text: the time since the run started (a kept line timed
    from begin_plain_log), two spaces per level, no color codes, and one
    stamped line per line of a message."""
    from testuary import _progress

    clock = iter([100.0, 100.25, 3700.5])
    monkeypatch.setattr(_progress, "time", SimpleNamespace(monotonic=lambda: next(clock)))
    path = tmp_path / "run.log"
    _progress.begin_plain_log()
    try:
        _progress.write_plain_log("[main] early", level=1)
        _progress.open_plain_log(path)
        _progress.write_plain_log("\x1b[1mbold\x1b[0m\nnext", level=2)
    finally:
        _progress.close_plain_log()
    assert path.read_text(encoding="utf-8") == (
        "00:00:00.250   [main] early\n01:00:00.500     bold\n01:00:00.500     next\n")


def test_lines_before_the_log_opens_are_written_first_and_diagnostics_once(tmp_path, capsys, monkeypatch):
    from testuary import _progress
    from testuary.agent._helpers import AgentLogger

    path = tmp_path / "logs" / "run.log"
    path.parent.mkdir()
    _progress.write_plain_log("before the run starts")
    _progress.begin_plain_log()
    try:
        _progress.write_plain_log("[main] early line", level=1)
        _progress.diagnostic_print("ERROR: early diagnostic", flush=True)
        monkeypatch.setattr(_progress, "_PLAIN_LOG_PENDING_LIMIT", 3)
        _progress.write_plain_log("kept as the third")
        _progress.write_plain_log("beyond the limit")
        _progress.open_plain_log(path)
        logger = AgentLogger(project_name="fixture", agent_name="main", config={})
        logger.init()
        logger.log("WARNING: echoed and logged once")
        _progress.diagnostic_print("a later diagnostic")
    finally:
        _progress.close_plain_log()
    _progress.begin_plain_log()
    _progress.close_plain_log()
    _progress.write_plain_log("after the run")
    lines = [line.split(" ", 1)[1] for line in path.read_text(encoding="utf-8").splitlines()]
    assert lines == ["  [main] early line", "ERROR: early diagnostic", "kept as the third",
                     "WARNING: echoed and logged once", "a later diagnostic"]
    assert "WARNING: echoed and logged once" in capsys.readouterr().out
