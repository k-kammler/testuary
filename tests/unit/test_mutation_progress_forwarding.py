"""The backend's mutation progress reaches the display's operation layer.

The display's lowest row: ``MutationRequest.progress`` records become
"Mutation testing" observations of the target and phase that run them,
and the row ends with its run.
"""
from __future__ import annotations

import threading

import pytest

from langadapter.mutation import MutationProgress
from testuary._progress import RunProgress, mutation_measurement, observe_target


def _observed(events, target="src/calc.py", phase="coverage"):
    progress = RunProgress(events.append)
    progress.emit(target=target, phase=phase, operation=f"phase/{phase}", state="running",
                  level="phase", label=phase)
    return progress


def test_records_become_operation_rows_of_the_running_phase_and_end_with_the_run():
    events = []
    progress = _observed(events)
    with observe_target(progress, "src/calc.py"), mutation_measurement() as forward:
        # The backend calls back from its own thread.
        worker = threading.Thread(target=forward, args=(MutationProgress("mutants", 3, 40, 2.0),))
        worker.start()
        worker.join()
        forward(MutationProgress("mutants", 12, 40, 6.0))
    rows = [event for event in events if event.level == "operation"]
    assert [(row.state, row.completed, row.total, row.phase, row.target) for row in rows] == [
        ("running", 3, 40, "coverage", "src/calc.py"),
        ("running", 12, 40, "coverage", "src/calc.py"),
        ("succeeded", 12, 40, "coverage", "src/calc.py"),
    ]
    assert {row.label for row in rows} == {"Mutation testing"}
    assert all(row.metrics == () and row.run_id == progress.run_id for row in rows)
    assert rows[-1].message == "Mutation testing done"


@pytest.mark.parametrize("error, state", [(RuntimeError("engine"), "failed"),
                                          (KeyboardInterrupt(), "cancelled")])
def test_a_failed_or_interrupted_run_ends_its_row_and_keeps_the_error(error, state):
    events = []
    progress = _observed(events, phase="mutation")
    with pytest.raises(type(error)):
        with observe_target(progress, "src/calc.py"), mutation_measurement() as forward:
            forward(MutationProgress("mutants", 1, None, 1.0))
            raise error
    last = [event for event in events if event.level == "operation"][-1]
    assert (last.state, last.total, last.phase) == (state, None, "mutation")


def test_an_engine_without_progress_shows_no_row_and_no_observer_gives_no_callback():
    events = []
    progress = _observed(events)
    with observe_target(progress, "src/calc.py"), mutation_measurement() as forward:
        assert callable(forward)
    assert not [event for event in events if event.level == "operation"]
    with mutation_measurement() as forward:
        assert forward is None


def test_a_failing_display_never_reaches_the_backend_thread(capsys):
    def broken(event):
        if event.level == "operation":
            raise ValueError("display failed")

    progress = RunProgress(broken)
    with observe_target(progress, "src/calc.py"), mutation_measurement() as forward:
        forward(MutationProgress("mutants", 1, 2, 0.5))
        forward(MutationProgress("mutants", 2, 2, 0.9))
    assert "Display observer failed" in capsys.readouterr().err


@pytest.mark.parametrize("outcome, incomplete, state, message", [
    ("COMPLETED", None, "succeeded", "Mutation testing done"),
    ("TIMED_OUT", {"unresolved": 3}, "failed", "Mutation testing not fully measured: the run's time limit ended it"),
    # A completed run with untested mutants hit no time limit.
    ("COMPLETED", {"reason": "untested_mutants_in_catalogue"}, "failed",
     "Mutation testing not fully measured: some mutants have no result"),
    ("INFRASTRUCTURE_FAILURE", None, "failed", "Mutation testing failed"),
])
def test_the_row_ends_with_the_runs_stated_outcome(outcome, incomplete, state, message):
    # A run the wall clock cut, or a failed run, returns normally, and its
    # row does not end as succeeded.
    from types import SimpleNamespace

    from langadapter import Outcome

    events = []
    progress = _observed(events, phase="mutation")
    with observe_target(progress, "src/calc.py"), mutation_measurement() as forward:
        forward(MutationProgress("mutants", 2, 5, 1.0))
        forward.report(SimpleNamespace(outcome=getattr(Outcome, outcome), incomplete=incomplete))
    last = [event for event in events if event.level == "operation"][-1]
    assert (last.state, last.message) == (state, message)


def test_a_complete_run_ends_its_row_with_every_mutant_measured():
    """The engine reports progress at intervals, so its last record can lag
    the run: a run that completed with a result for every mutant ends its
    row at the total, not at 36 of 37 beside a finished run."""
    from types import SimpleNamespace

    from langadapter import Outcome

    events = []
    progress = _observed(events, phase="mutation")
    with observe_target(progress, "src/calc.py"), mutation_measurement() as forward:
        forward(MutationProgress("mutants", 36, 37, 30.0))
        forward.report(SimpleNamespace(outcome=Outcome.COMPLETED, incomplete=None))
    last = [event for event in events if event.level == "operation"][-1]
    assert (last.state, last.completed, last.total) == ("succeeded", 37, 37)


def test_a_per_suite_run_with_an_incomplete_suite_ends_incomplete():
    # A per-suite run states no completeness of its own: any suite's run
    # with unmeasured mutants makes the measurement incomplete.
    from types import SimpleNamespace

    from langadapter import Outcome

    events = []
    progress = _observed(events, phase="mutation")
    run = SimpleNamespace(outcome=Outcome.COMPLETED, runs={
        "suite_001": SimpleNamespace(incomplete=None),
        "suite_002": SimpleNamespace(incomplete={"reason": "untested_mutants_in_catalogue"})})
    with observe_target(progress, "src/calc.py"), mutation_measurement() as forward:
        forward(MutationProgress("mutants", 2, 5, 1.0))
        forward.report(run)
    last = [event for event in events if event.level == "operation"][-1]
    assert (last.state, last.message) == (
        "failed", "Mutation testing not fully measured: some mutants have no result")


@pytest.mark.parametrize("outcome, measurement, verification", [
    ("COMPLETED", ("succeeded", "Mutation testing done"),
     ("succeeded", "Re-checking caught mutants done")),
    ("TIMED_OUT", ("failed", "Mutation testing not fully measured: the run's time limit ended it"),
     ("failed", "Re-checking caught mutants failed")),
])
def test_each_kind_of_progress_ends_its_own_row(outcome, measurement, verification):
    from types import SimpleNamespace

    from langadapter import Outcome

    events = []
    progress = _observed(events, phase="mutation")
    with observe_target(progress, "src/calc.py"), mutation_measurement() as forward:
        forward(MutationProgress("mutants", 40, 40, 5.0))
        forward(MutationProgress("kill verifications", 0, 6, 6.0))
        forward(MutationProgress("kill verifications", 4, 6, 7.0))
        forward.report(SimpleNamespace(outcome=getattr(Outcome, outcome), incomplete=None))
    ended = {event.label: (event.state, event.message, event.completed)
             for event in events if event.level == "operation" and event.state != "running"}
    assert ended == {"Mutation testing": (*measurement, 40),
                     "Re-checking caught mutants": (*verification, 4)}
