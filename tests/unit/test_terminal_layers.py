"""Layered progress: run, file, phase, step and operation rows from supplied events.

The events are written here in the display's event shape. They test
presentation only, not native measurements.
"""
from __future__ import annotations

import io
import re

import pytest
from rich.console import Console

from testuary.presentation import DisplayEvent, Metric, TerminalPresenter, Usage, present_readiness
from testuary.readiness import ReadinessItem, RunReadiness


@pytest.fixture(autouse=True)
def terminal_environment(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")
    monkeypatch.delenv("NO_COLOR", raising=False)


def _event(**fields):
    values = dict(run_id="run-7", target="src/calc.py", phase="coverage",
                  operation="phase/run-1", state="running", elapsed_s=72)
    values.update(fields)
    return DisplayEvent(**values)


_LAYERED = (
    _event(target="all targets", phase="pipeline", operation="run", level="run",
           completed=5, total=12, work_unit="phases done"),
    _event(phase="pipeline", operation="target", level="target", label="src/calc.py",
           completed=1, total=2, work_unit="targets"),
    _event(level="phase", label="Coverage phase", completed=1, total=4, work_unit="phases"),
    _event(operation="step/consolidation", level="step", label="Consolidation",
           completed=3, total=7, work_unit="candidates"),
    _event(operation="measure/mutants", level="operation", label="Mutation measurement",
           completed=120, total=350, work_unit="mutants"),
)


def _frame(presenter, width):
    snapshot = Console(file=io.StringIO(), width=width, color_system=None,
                       force_terminal=True, highlight=False, markup=False)
    with snapshot.capture() as captured:
        snapshot.print(presenter.frame())
    return captured.get()


@pytest.mark.parametrize("width", [40, 100])
def test_layers_show_overall_file_phase_step_and_operation_with_bars(width):
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, width=width, height=40,
                      color_system=None, highlight=False, markup=False)
    with TerminalPresenter("compact", console=console) as presenter:
        for item in _LAYERED:
            presenter(item)
        frame = _frame(presenter, width)
    lines = frame.splitlines()
    text = " ".join(frame.split())
    assert lines[0] == "Testuary"
    for row in ("Overall RUNNING", "41% 5 of 12 phases done", "Target src/calc.py", "50% 1 of 2 targets",
                "Phase Coverage phase | RUNNING", "25% 1 of 4 phases",
                "Step Consolidation | RUNNING", "42% 3 of 7 candidates",
                "Mutation measurement", "34% 120 of 350 mutants"):
        assert row in text
    assert text.index("Overall") < text.index("Target") < text.index("Phase") < text.index("Step")
    # Five bars where they fit: overall, file, phase, step and the operation row.
    # Block characters, or # and - on an ASCII-only console; narrow rows may
    # show the numbers only.
    bars = [line for line in lines if re.search("[" + "\u2588\u2591#-" + "]{6,} +[0-9]+%", line)]
    assert len(bars) == (5 if width >= 100 else len(bars)) and bars
    assert "unknown" not in frame
    for internal in ("run-7", "phase/run-1", "step/consolidation", "measure/mutants"):
        assert internal not in frame
    assert all(len(line) <= width for line in lines)


def _plan(files=("src/calc.py",), runs=("r0",)):
    from testuary.presentation.events import PlannedPhase, PlannedStep, RunPlan

    return RunPlan(files=files, runs=runs, phases=(
        PlannedPhase("starter", "Starter phase"), PlannedPhase("coverage", "Coverage phase"),
        PlannedPhase("mutation", "Mutation phase")), finishing=PlannedStep("finishing", "Finishing"))


def _bars(frame):
    return [line for line in frame.splitlines() if re.search("[" + "\u2588\u2591#-" + "]{6,} +[0-9]+%", line)]


_ONE_FILE_RUN = (
    _event(target="all targets", phase="pipeline", operation="run", level="run",
           completed=1, total=3, work_unit="phases done"),
    _event(phase="pipeline", operation="target", level="target", label="src/calc.py",
           completed=0, total=1, work_unit="targets"),
)


@pytest.mark.parametrize("phase, place", [
    (dict(phase="starter", state="running", completed=0), "Starter phase | phase 1 of 3 | RUNNING"),
    (dict(phase="starter", state="succeeded", completed=1), "Starter phase | phase 1 of 3 | SUCCEEDED"),
    (dict(state="running", completed=1), "Coverage phase | phase 2 of 3 | RUNNING"),
    (dict(operation="restore/run-1", state="succeeded", completed=2, provenance="recorded"),
     "Coverage phase (reused) | phase 2 of 3 | SUCCEEDED"),
    (dict(phase="mutation", state="failed", completed=2), "Mutation phase | phase 3 of 3 | FAILED"),
    (dict(phase="mutation", state="succeeded", completed=3), "Mutation phase | phase 3 of 3 | SUCCEEDED"),
])
def test_a_single_file_run_has_one_bar_the_overall_one(phase, place):
    """In a run of one file and one run, the Target and Phase bars would repeat
    the Overall count, so only Overall keeps its bar: Target names the file,
    Phase its phase, its place among the phases and its state."""
    _, presenter = _compact()
    with presenter:
        presenter.plan(_plan())
        for item in _ONE_FILE_RUN + (_event(level="phase", total=3, work_unit="phases", **phase),):
            presenter(item)
        frame = _frame(presenter, 120)
        # The rows above a failure's issue line, which is the log's line.
        text = " ".join(frame.split()).split(" Recent issue")[0]
        assert "33% 1 of 3 phases done" in text
        assert "Target src/calc.py" in text and "targets" not in text
        assert f"Phase {place}" in text and "phases" not in text.split("Target src/calc.py")[1]
        assert len(_bars(frame)) == 1
        # The browser draws the view's own rows, which keep every bar.
        assert [layer.name for layer in presenter.view.snapshot(4).layers if layer.bar] == [
            "Overall", "Target", "Phase"]


@pytest.mark.parametrize("files, runs", [(("src/calc.py", "src/io.py"), ("r0",)),
                                         (("src/calc.py",), ("r0", "r1"))])
def test_several_file_runs_keep_the_file_and_phase_bars(files, runs):
    _, presenter = _compact()
    with presenter:
        presenter.plan(_plan(files, runs))
        for item in _LAYERED[:3]:
            presenter(item)
        frame = _frame(presenter, 120)
    text = " ".join(frame.split())
    for row in ("41% 5 of 12 phases done", "Target src/calc.py", "50% 1 of 2 targets",
                "Phase Coverage phase | RUNNING", "25% 1 of 4 phases"):
        assert row in text
    assert len(_bars(frame)) == 3


def test_unknown_appears_only_where_a_count_has_no_total():
    with TerminalPresenter("plain", stream=io.StringIO()) as presenter:
        presenter(_event(operation="suite-processing", completed=2, work_unit="responses"))
        presenter(_event(operation="request/9", state="waiting"))
        frame = _frame(presenter, 100)
    assert "Generate" in frame and "2 responses, total unknown" in frame
    assert "Model request 9" in frame
    assert frame.count("unknown") == 1


def test_compact_log_records_layer_starts_and_outcomes_not_running_operations():
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, width=120, height=40,
                      color_system=None, highlight=False, markup=False)
    with TerminalPresenter("compact", console=console) as presenter:
        presenter(_event(level="phase", label="Coverage phase", message="Phase started"))
        presenter(_event(operation="request/1", message="Sending model request"))
        presenter(_event(operation="request/1", state="succeeded", message="Model response received"))
    log = console.export_text() if console.record else stream.getvalue()
    assert "RUNNING | src/calc.py | Coverage phase | Phase started" in log
    assert "SUCCEEDED | src/calc.py | Coverage phase | Model request 1 | Model response received" in log
    assert "RUNNING | src/calc.py | Coverage phase | Model request 1 | Sending model request" not in log


@pytest.mark.parametrize("tty", [True, False])
def test_debug_level_adds_identities_and_keeps_every_readiness_item(tty):
    stream = io.StringIO()
    if tty:
        console = Console(file=stream, force_terminal=True, width=200, color_system=None,
                          highlight=False, markup=False)
        presenter = TerminalPresenter("debug", console=console)
    else:
        presenter = TerminalPresenter("debug", stream=stream, width=200)
    assert presenter.mode == ("detailed" if tty else "plain") and presenter.debug
    report = RunReadiness(
        scope="inspection", status="unverified", exit_code=0, source_dir="/work/p",
        execution="local", source_files=("a.py",), phases=("starter",), model_alias=None,
        provider=None, model=None, credentials_configured=None, output_dir=None,
        output_mode="fresh", limits=(),
        items=(ReadinessItem("native.python.item.runtime", True, "verified", "/bin/python", "3.12"),))
    with presenter:
        presenter(_event(operation="request/x/2"))
        present_readiness(report, presenter=presenter)
    text = stream.getvalue()
    assert "run-7 | src/calc.py | coverage / request/x/2" in text
    assert "Model request 2" in text
    assert "runtime (native.python.item.runtime, required): found" in text


def test_unlabeled_step_reads_as_its_name():
    with TerminalPresenter("plain", stream=io.StringIO()) as presenter:
        presenter(_event(operation="step/suite_timing", level="step"))
        frame = _frame(presenter, 100)
    assert "Step     Suite timing | RUNNING" in frame and "step/" not in frame


def test_request_limit_and_goal_are_limits_beside_values_not_progress():
    stream = io.StringIO()
    with TerminalPresenter("plain", stream=stream) as presenter:
        presenter(_event(operation="request/6", usage=Usage(logical_requests=6, logical_request_limit=8),
                         metrics=(Metric("Line coverage", 80, "%", goal=100),
                                  Metric("Mutation score", None, "%", "unavailable", goal=90))))
        presenter(_event(operation="request/7", state="waiting",
                         usage=Usage(logical_request_limit=8)))
        frame = _frame(presenter, 100)
    text = stream.getvalue()
    assert "model requests 6 of 8" in text and "request limit 8" in text
    assert "Line coverage: 80% (goal 100%) [measured]" in text
    assert "6/8" not in text + frame and "(75%)" not in text + frame
    bars = [line for line in frame.splitlines()
            if line.strip() and set(line.strip()) <= set("\u2501\u2578\u257a-")]
    assert not bars


def _compact():
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, width=120, height=40,
                      color_system=None, highlight=False, markup=False)
    return stream, TerminalPresenter("compact", console=console)


def test_step_row_belongs_to_the_current_phase_and_repetition():
    _, presenter = _compact()
    with presenter:
        presenter(_event(level="phase"))
        presenter(_event(operation="step/consolidation", level="step", label="Consolidation",
                         state="succeeded"))
        assert "Step     Consolidation" in _frame(presenter, 120)
        presenter(_event(phase="mutation", level="phase"))
        assert "Step" not in _frame(presenter, 120)
        presenter(_event(phase="mutation", operation="step/consolidation", level="step",
                         label="Consolidation"))
        assert "Step     Consolidation | RUNNING" in _frame(presenter, 120)
        # A new repetition of the same phase starts without the old step.
        presenter(_event(phase="mutation", operation="phase/run-2", level="phase"))
        assert "Step" not in _frame(presenter, 120)


def test_a_restored_phase_that_never_ran_does_not_adopt_older_steps_or_usage():
    _, presenter = _compact()
    with presenter:
        presenter(_event(operation="step/consolidation", level="step", label="Consolidation",
                         state="succeeded"))
        presenter(_event(operation="request/7", state="succeeded",
                         usage=Usage(logical_requests=7, logical_request_limit=8)))
        # A restored phase appears for the first time already finished.
        presenter(_event(operation="restore/run-1", level="phase", state="succeeded"))
        frame = _frame(presenter, 120)
    assert "Phase    Coverage phase (reused) | SUCCEEDED" in frame
    assert "Step" not in frame and "Usage" not in frame


def test_usage_row_belongs_to_the_current_file_and_phase():
    _, presenter = _compact()
    with presenter:
        presenter(_event(level="phase"))
        presenter(_event(operation="request/3", state="succeeded",
                         usage=Usage(logical_requests=3, logical_request_limit=8)))
        assert "Model use, Coverage phase: model requests 3 of 8" in _frame(presenter, 120)
        presenter(_event(phase="mutation", level="phase"))
        assert "Usage" not in _frame(presenter, 120)
        presenter(_event(phase="mutation", operation="request/1", state="succeeded",
                         usage=Usage(logical_requests=1, logical_request_limit=8)))
        assert "Model use, Mutation phase: model requests 1 of 8" in _frame(presenter, 120)
        presenter(_event(target="src/other.py", level="phase"))
        assert "Usage" not in _frame(presenter, 120)


def test_overall_time_appears_once_the_run_has_ended():
    _, presenter = _compact()
    with presenter:
        run = _event(target="all targets", phase="pipeline", operation="run", level="run",
                     elapsed_s=10, completed=0, total=4, work_unit="phases done")
        presenter(run)
        presenter(_event(operation="request/1", elapsed_s=50))
        # The view redraws only on observations, so no standing clock is shown.
        assert "Overall  RUNNING\n" in _frame(presenter, 120)
        presenter(_event(operation="request/1", elapsed_s=60, state="succeeded"))
        presenter(_event(target="all targets", phase="pipeline", operation="run", level="run",
                         elapsed_s=61, state="succeeded", completed=4, total=4,
                         work_unit="phases done"))
        assert "Overall  SUCCEEDED 00:01:01" in _frame(presenter, 120)


def test_compact_log_records_a_step_once_at_start_and_once_at_its_outcome():
    stream, presenter = _compact()
    with presenter:
        presenter(_event(level="phase", message="Phase started"))
        for done in range(21):
            presenter(_event(operation="step/consolidation", level="step", label="Consolidation",
                             completed=done, total=20, work_unit="candidates", elapsed_s=72 + done))
        presenter(_event(operation="step/consolidation", level="step", label="Consolidation",
                         state="succeeded", elapsed_s=95))
        presenter(_event(level="phase", message="Phase started", elapsed_s=140))
    # Log records start with a timestamp once cursor controls are removed;
    # the pinned view's own rows never do.
    text = re.sub(r"\x1b\[[0-?]*[ -/]*[@-~]|\r", "\n", stream.getvalue())
    records = [line for line in text.splitlines() if re.match(r"\d{2}:\d{2}:\d{2} ", line)]
    steps = [line for line in records if "| Consolidation" in line]
    assert len(steps) == 2 and "RUNNING" in steps[0] and "SUCCEEDED" in steps[1]
    assert len([line for line in records if "Phase started" in line]) == 1


def test_goal_stays_visible_while_the_value_is_not_measured():
    stream = io.StringIO()
    with TerminalPresenter("plain", stream=stream) as presenter:
        presenter(_event(level="phase", metrics=(
            Metric("Line coverage", None, "%", "unavailable", goal=100),
            Metric("Branch coverage", None, "%", "unavailable"))))
        frame = _frame(presenter, 120)
    for text in (stream.getvalue(), frame):
        assert "Line coverage: goal 100%" in text
        assert "Branch coverage" not in text and "Line coverage: unavailable" not in text


def test_time_left_only_for_evenly_sized_measurement_work_after_a_few_items():
    _, presenter = _compact()
    with presenter:
        mutants = dict(operation="measure/mutants", level="operation", label="Mutation measurement",
                       total=20, work_unit="mutants")
        presenter(_event(**mutants, completed=0, elapsed_s=10))
        presenter(_event(**mutants, completed=2, elapsed_s=20))
        assert "left" not in _frame(presenter, 120)
        presenter(_event(**mutants, completed=4, elapsed_s=30))
        # 20 s for 4 mutants: about 80 s for the remaining 16.
        assert "about 1 min left" in _frame(presenter, 120)
        presenter(_event(operation="suite-processing", completed=3, total=10,
                         work_unit="starter suites accepted", elapsed_s=40))
        presenter(_event(operation="suite-processing", completed=6, total=10,
                         work_unit="starter suites accepted", elapsed_s=90))
        frame = _frame(presenter, 120)
    assert "Generate" in frame and frame.count("left") == 1


def test_log_location_is_shown_from_the_first_frame_with_the_detailed_hint():
    stream, presenter = _compact()
    plain = io.StringIO()
    run = _event(target="all targets", phase="pipeline", operation="run", level="run",
                 completed=0, total=4, work_unit="phases done", log_path="C:/runs/out put/logs")
    with presenter:
        presenter(run)
        frame = _frame(presenter, 120)
    assert "Log      C:/runs/out put/logs (every step: --display detailed)" in frame
    with TerminalPresenter("plain", stream=plain) as plain_presenter:
        plain_presenter(run)
    assert "Log: C:/runs/out put/logs" in plain.getvalue()


@pytest.mark.parametrize("width, hint", [(70, True), (40, False)])
def test_a_long_log_path_is_shortened_from_the_front_with_a_mark(width, hint):
    """A Log row wider than the terminal keeps the path's end, the log's
    directories and file name, behind "...", cut at a directory. A narrow terminal
    drops the hint first. The event and the plain log keep the whole path."""
    path = ("/home/user/projects/examples/demonstrations/testuary/"
            "testuary-demo-tenacity_wait-20261028T050632Z/run/logs/run.log")
    console = Console(file=io.StringIO(), force_terminal=True, width=width, height=40,
                      color_system=None, highlight=False, markup=False)
    run = _event(target="all targets", phase="pipeline", operation="run", level="run",
                 completed=0, total=4, work_unit="phases done", log_path=path)
    with TerminalPresenter("compact", console=console) as presenter:
        presenter(run)
        frame = _frame(presenter, width)
        assert presenter.view.snapshot(4).layers[1].text.endswith(path + " (every step: --display detailed)")
    row = next(line for line in frame.splitlines() if line.startswith("Log"))
    assert row.startswith("Log      .../") and len(row) <= width
    assert row.endswith("/logs/run.log" + (" (every step: --display detailed)" if hint else ""))


def test_bar_and_percentage_reach_full_only_when_all_work_is_done():
    _, presenter = _compact()
    plain = io.StringIO()
    work = dict(operation="measure/mutants", level="operation", label="Mutation measurement",
                total=200, work_unit="mutants")
    with presenter:
        presenter(_event(**work, completed=199))
        frame = _frame(presenter, 120)
    # Rounded down, with one empty cell left while one mutant is still open.
    assert " 99%  199 of 200 mutants" in frame and "100%" not in frame
    assert chr(0x2591) in frame or "-" in frame.split("99%")[0]
    with TerminalPresenter("plain", stream=plain) as plain_presenter:
        plain_presenter(_event(**work, completed=199))
        plain_presenter(_event(**work, completed=200, state="succeeded"))
    lines = plain.getvalue().splitlines()
    assert "199 of 200 mutants (99%)" in lines[0] and "200 of 200 mutants (100%)" in lines[1]


def test_time_left_counts_only_items_finished_since_the_first_sighting():
    _, presenter = _compact()
    work = dict(operation="measure/mutants", level="operation", label="Mutation measurement",
                total=100, work_unit="mutants")
    with presenter:
        # First seen half done: the 50 earlier mutants took time the view never saw.
        presenter(_event(**work, completed=50, elapsed_s=100))
        presenter(_event(**work, completed=51, elapsed_s=130))
        assert "left" not in _frame(presenter, 120)
        presenter(_event(**work, completed=53, elapsed_s=160))
        # 60 s for 3 mutants: about 940 s for the remaining 47.
        assert "about 16 min left" in _frame(presenter, 120)


def test_time_left_restarts_when_the_counter_starts_again():
    _, presenter = _compact()
    work = dict(operation="measure/mutants", level="operation", label="Mutation measurement",
                total=754, work_unit="mutants")
    with presenter:
        presenter(_event(**work, completed=0, elapsed_s=0))
        presenter(_event(**work, completed=754, elapsed_s=600))
        # The next cycle counts again from 0: 70 mutants in 70 s, not in 53 min.
        presenter(_event(**work, completed=0, elapsed_s=3110))
        presenter(_event(**work, completed=70, elapsed_s=3180))
        frame = _frame(presenter, 120)
    assert "70 of 754 mutants" in frame
    # 70 s for 70 mutants: about 684 s for the remaining 684.
    assert "about 11 min left" in frame


def test_a_file_without_a_complete_suite_is_a_warning_not_an_execution_failure():
    outcome = dict(phase="pipeline", operation="target", level="target", state="failed")
    plain = io.StringIO()
    with TerminalPresenter("plain", stream=plain) as presenter:
        presenter(_event(**outcome, label="src/calc.py", failure="outcome",
                         message="No complete suite: no seed passed"))
        presenter(_event(**outcome, target="src/other.py", label="src/other.py",
                         failure="execution", message="Target failed to execute: crash"))
    lines = plain.getvalue().splitlines()
    assert "MINIMUMS NOT MET | src/calc.py | No complete suite: no seed passed" in lines[0]
    assert "FAILED | src/other.py | Target failed to execute: crash" in lines[1]
    # Colors: the outcome is yellow, the execution failure keeps its red.
    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, width=120, height=40,
                      color_system="standard", highlight=False, markup=False)
    with TerminalPresenter("detailed", console=console) as presenter:
        presenter(_event(**outcome, label="src/calc.py", failure="outcome",
                         message="No complete suite"))
        presenter(_event(**outcome, target="src/other.py", label="src/other.py",
                         failure="execution", message="Target failed to execute"))
    colored = stream.getvalue().splitlines()
    assert "[1;33m" in colored[0] and "MINIMUMS NOT MET" in colored[0]
    assert "[1;31m" in colored[1] and "FAILED" in colored[1]


def test_failure_kind_is_validated():
    with pytest.raises(ValueError, match="failed observation only"):
        _event(state="succeeded", failure="outcome")
    with pytest.raises(ValueError, match="execution or outcome"):
        _event(state="failed", failure="scientific")


def test_layer_and_label_are_validated():
    with pytest.raises(ValueError, match="level"):
        _event(level="file")
    with pytest.raises(ValueError, match="label"):
        _event(label=None)
    assert _event().level is None and _event().label == ""
    for goal in (float("nan"), float("inf"), True):
        with pytest.raises(ValueError, match="goal"):
            Metric("Line coverage", 80, "%", goal=goal)
    with pytest.raises(ValueError, match="logical_request_limit"):
        Usage(logical_request_limit=-1)
