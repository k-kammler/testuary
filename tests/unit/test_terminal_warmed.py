"""Warmed suite timing records from the public backend types, rendered as observations.

The records are constructed here with the backend's own validation. They are
display inputs, not native measurements.
"""
from __future__ import annotations

import io
import math

import pytest
from rich.console import Console

from langadapter import ExecutionIdentity, Outcome, ToolchainIdentity
from langadapter.runs import (
    WarmedBenchmarkIdentity, WarmedBenchmarkProgress, WarmedFork, WarmedIteration,
    WarmedSuiteBenchmark, WarmedTestCounts, WarmedTestObservation,
)
from testuary.presentation import TerminalPresenter
from testuary.presentation.measurement import warmed_progress_event, warmed_result_event


@pytest.fixture(autouse=True)
def terminal_environment(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")


def _identity(forks, runs):
    return WarmedBenchmarkIdentity(
        suite=(), project_inputs=(), toolchain=ToolchainIdentity("java", "21"),
        execution=ExecutionIdentity("local"), build_tool=None, build_version=None,
        engine_version=None, jvm_args=(), runner_options=(), mode="warmed_jvm",
        timing_boundary="junit_launcher_execute", state_policy="native_lifecycle",
        forks=forks, warmup_runs=1, runs=runs, iteration_timeout_s=2, total_timeout_s=60,
        startup_timeout_s=30, requested_cpus=None, serialize=True)


def _complete(samples_ms, *, outcome=Outcome.COMPLETED, **headlines):
    status = "successful" if outcome is Outcome.COMPLETED else "failed"
    tests = (WarmedTestObservation("[engine:junit]/[method:m]", "m", "test", status),)
    counts = WarmedTestCounts(passed=1) if outcome is Outcome.COMPLETED else WarmedTestCounts(failed=1)
    iterations, forks = [], []
    for index, values in enumerate(samples_ms):
        samples = tuple(int(value * 1_000_000) for value in values)
        iterations.append(WarmedIteration(index, "warmup", 0, 30_000_000, outcome, True, counts, tests))
        iterations.extend(WarmedIteration(index, "measurement", repetition, value, outcome, True,
                                          counts, tests)
                          for repetition, value in enumerate(samples))
        forks.append(WarmedFork(index, 1, len(samples), True, True, None, samples))
    count, runs = len(samples_ms), len(samples_ms[0])
    return WarmedSuiteBenchmark(
        outcome=outcome, reason="", validity="complete", requested_forks=count,
        requested_warmup_runs=1, requested_runs=runs, forks_started=count,
        forks_completed=count, warmup_completed=count, measurements_completed=count * runs,
        iterations=tuple(iterations), forks=tuple(forks), operation_elapsed_s=12.5,
        identity=_identity(count, runs), requested_cpus=None, allowed_cpus=None,
        cancelled=False, disposal_confirmed=True, **headlines)


def _incomplete(**changes):
    values = dict(outcome=Outcome.TIMED_OUT, reason="iteration bound reached", validity="incomplete",
                  requested_forks=2, requested_warmup_runs=1, requested_runs=2, forks_started=1,
                  forks_completed=0, warmup_completed=0, measurements_completed=0, iterations=(),
                  forks=(), operation_elapsed_s=None, identity=_identity(2, 2),
                  requested_cpus=None, allowed_cpus=None, cancelled=False, disposal_confirmed=True)
    values.update(changes)
    return WarmedSuiteBenchmark(**values)


def _plain(*events):
    stream = io.StringIO()
    with TerminalPresenter("plain", stream=stream) as presenter:
        for item in events:
            presenter(item)
    return stream.getvalue()


def _progress(stage, **changes):
    values = dict(stage=stage, fork=None, kind=None, repetition=None, requested_forks=2,
                  requested_warmup_runs=1, requested_runs=2, forks_started=0, forks_completed=0,
                  warmup_completed=0, measurements_completed=0)
    values.update(changes)
    return WarmedBenchmarkProgress(**values)


def test_progress_counts_finished_iterations_and_keeps_raw_samples_out_of_summaries():
    records = (
        _progress("setup"),
        _progress("fork_start", fork=0, forks_started=1),
        _progress("iteration_start", fork=0, kind="warmup", repetition=0, forks_started=1),
        _progress("iteration_end", fork=0, kind="warmup", repetition=0, forks_started=1,
                  warmup_completed=1, elapsed_ns=30_000_000, outcome=Outcome.COMPLETED,
                  operation_elapsed_s=1.5),
        _progress("iteration_end", fork=0, kind="measurement", repetition=0, forks_started=1,
                  warmup_completed=1, measurements_completed=1, elapsed_ns=8_000_000,
                  outcome=Outcome.TEST_FAILURES, operation_elapsed_s=2.0),
    )
    events = [warmed_progress_event(item, run_id="r", target="CalcTest") for item in records]
    assert [event.completed for event in events] == [0, 0, 0, 1, 2]
    assert all(event.total == 6 and event.work_unit == "iterations" for event in events)
    assert all(event.state == "running" and event.metrics == () for event in events)
    text = _plain(*events)
    assert "Warmed suite timing" in text
    assert "Fork 1 of 2: warmup 1 of 1 running" in text
    assert "Fork 1 of 2: warmup 1 of 1 finished in 30.000 ms (raw)" in text
    assert "measurement 1 of 2 finished in 8.000 ms (raw), outcome test failures" in text
    assert "Mean of fork means" not in text and "warmed-timing" not in text


@pytest.mark.parametrize("samples,headlines,expected", [
    (((1, 5), (2, 10)),
     dict(mean_of_fork_means_ms=4.5, median_of_fork_medians_ms=4.5, sample_min_ms=1.0,
          sample_max_ms=10.0, between_fork_mean_sd_ms=3 / math.sqrt(2)),
     ["Mean of fork means: 4.5 ms [measured]", "Between-fork sample SD: 2.12132 ms [measured]"]),
    (((1, 2, 10, 11),),
     dict(mean_of_fork_means_ms=6.0, median_of_fork_medians_ms=6.0, sample_min_ms=1.0,
          sample_max_ms=11.0, between_fork_mean_sd_ms=None),
     ["Mean of fork means: 6 ms [measured]", "Between-fork sample SD: unavailable"]),
])
def test_complete_result_shows_the_backend_summaries(samples, headlines, expected):
    result = _complete(samples, **headlines)
    event = warmed_result_event(result, run_id="r", target="CalcTest")
    assert event.state == "succeeded"
    assert (event.completed, event.total) == (result.measurements_completed,
                                              result.requested_forks * result.requested_runs)
    text = _plain(event)
    for item in expected:
        assert item in text
    assert "validity complete" in text and "comparison not assessed" in text
    assert "around JUnit Launcher.execute" in text and "no timing summary" not in text


@pytest.mark.parametrize("result,state,words", [
    (lambda: _complete(((1, 5),), outcome=Outcome.TEST_FAILURES), "failed",
     ("outcome test failures", "validity complete", "no timing summary")),
    (lambda: _incomplete(), "failed",
     ("outcome timed out", "validity incomplete", "no timing summary", "iteration bound reached")),
    (lambda: _incomplete(cancelled=True), "cancelled", ("validity incomplete", "no timing summary")),
    (lambda: _incomplete(disposal_confirmed=False), "failed", ("process disposal not confirmed",)),
])
def test_unusable_results_report_validity_without_a_number(result, state, words):
    event = warmed_result_event(result(), run_id="r", target="CalcTest")
    assert event.state == state and event.metrics == ()
    text = _plain(event)
    for item in words:
        assert item in text
    assert "Mean of fork means" not in text and " ms [measured]" not in text


def test_zero_durations_are_noted_as_timer_resolution_not_invented_values():
    result = _complete(((0, 2), (0, 4)), mean_of_fork_means_ms=1.5, median_of_fork_medians_ms=1.5,
                       sample_min_ms=0.0, sample_max_ms=4.0,
                       between_fork_mean_sd_ms=1 / math.sqrt(2))
    text = _plain(warmed_result_event(result, run_id="r", target="CalcTest"))
    assert "zero-duration samples are limited by timer resolution" in text
    assert "Sample minimum: 0 ms [measured]" in text


def test_compact_view_draws_a_measurement_bar_and_rejects_other_records():
    result = _complete(((1, 5), (2, 10)), mean_of_fork_means_ms=4.5,
                       median_of_fork_medians_ms=4.5, sample_min_ms=1.0, sample_max_ms=10.0,
                       between_fork_mean_sd_ms=3 / math.sqrt(2))
    console = Console(file=io.StringIO(), force_terminal=True, width=100, height=40,
                      color_system=None, highlight=False, markup=False)
    with TerminalPresenter("compact", console=console) as presenter:
        presenter(warmed_result_event(result, run_id="r", target="CalcTest"))
        snapshot = Console(file=io.StringIO(), width=100, color_system=None,
                           force_terminal=True, highlight=False, markup=False)
        with snapshot.capture() as captured:
            snapshot.print(presenter.frame())
    frame = captured.get()
    assert "Warmed suite timing" in frame and "100%  4 of 4 measurements" in frame
    with pytest.raises(TypeError, match="WarmedBenchmarkProgress"):
        warmed_progress_event(result, run_id="r", target="CalcTest")
    with pytest.raises(TypeError, match="WarmedSuiteBenchmark"):
        warmed_result_event(_progress("setup"), run_id="r", target="CalcTest")
