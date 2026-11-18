"""The benchmark display, from hand-written observations.

The dictionaries carry the display's fields and exercise the renderer only.
They are not native benchmark observations.
"""
from __future__ import annotations

from dataclasses import asdict, replace
from io import StringIO
import socket
import subprocess

import pytest
from rich.console import Console

from testuary.presentation import DisplayEvent, Metric, TerminalPresenter


_FRESH_BOUNDARY = "fresh-process: JVM startup, discovery and test execution"
_WARMED_BOUNDARY = (
    "warmed: System.nanoTime around Launcher.execute; startup/discovery excluded"
)
_SAMPLES = (
    (0, "warmup", 0, 30_000_000),
    (0, "warmup", 1, 20_000_000),
    (0, "measurement", 0, 8_000_000),
    (0, "measurement", 1, 10_000_000),
    (1, "warmup", 0, 40_000_000),
    (1, "warmup", 1, 25_000_000),
    (1, "measurement", 0, 10_000_000),
    (1, "measurement", 1, 12_000_000),
)
_CASES = ("fresh", "complete", "failed_complete", "incomplete", "non_repeatable",
          "no_samples", "unsupported", "lost_report", "cancelled", "zero", "unresolved_teardown")
_HEADLINES = (
    ("mean_of_fork_means_ms", "Mean of fork means"),
    ("median_of_fork_medians_ms", "Median of fork medians"),
    ("sample_min_ms", "Sample minimum"),
    ("sample_max_ms", "Sample maximum"),
    ("between_fork_mean_sd_ms", "Between-fork sample SD"),
)


def _event(operation, state, elapsed_s, *, message, completed=None, total=None,
           work_unit="iterations", metrics=(), language="java"):
    return DisplayEvent(
        run_id="bm1-offline-rendering",
        target="example/suite.py" if language == "python" else "example/SuiteTest.java",
        phase="suite timing", operation=operation, state=state,
        elapsed_s=elapsed_s, message=message, completed=completed, total=total,
        work_unit=work_unit, metrics=metrics, provenance="synthetic",
    )


def _progress(fork, kind, repetition, elapsed_ns, *, entered,
              outcome="completed", warmup_completed=0, measurements_completed=0):
    ordinal = fork * 8 + repetition * 2 + (kind == "measurement") * 4
    return {
        "stage": "iteration_start" if entered else "iteration_end",
        "fork": fork, "kind": kind, "repetition": repetition,
        "requested_forks": 2, "requested_warmup_runs": 2, "requested_runs": 2,
        "forks_started": fork + 1, "forks_completed": fork,
        "warmup_completed": warmup_completed,
        "measurements_completed": measurements_completed,
        "elapsed_ns": None if entered else elapsed_ns,
        "operation_elapsed_s": 1 + ordinal + (not entered),
        "outcome": None if entered else outcome, "reason": "",
    }


def _iteration(progress):
    fork, kind, repetition = (progress[key] for key in ("fork", "kind", "repetition"))
    entered = progress["stage"] == "iteration_start"
    state = "running" if entered else "succeeded" if progress["outcome"] == "completed" else "failed"
    return _event(
        f"warmed/fork-{fork}/{kind}", state, progress["operation_elapsed_s"],
        completed=repetition if entered else repetition + 1,
        total=progress["requested_warmup_runs" if kind == "warmup" else "requested_runs"],
        work_unit=f"complete {kind} reports in this fork",
        message=(
            f"{_WARMED_BOUNDARY}; native_lifecycle; fork {fork + 1}/2; "
            f"{kind} iteration {repetition + 1}/2; "
            + ("entered, no native outcome yet" if entered else
               f"native outcome {progress['outcome']}; raw elapsed {progress['elapsed_ns']} ns")
            + ("; warmup excluded from headline summaries" if kind == "warmup" else "")
        ),
        metrics=(Metric("Iteration elapsed", progress["elapsed_ns"], " ns",
                        "unavailable" if entered else "progress"),
                 Metric("Recorded warmups", progress["warmup_completed"], basis="progress"),
                 Metric("Recorded measurements", progress["measurements_completed"], basis="progress")),
    )


def _result(case):
    """Display-only subset of planned result fields, with authored native records."""
    retained = _SAMPLES
    if case in ("incomplete", "cancelled"):
        retained = _SAMPLES[:3]
    if case == "non_repeatable":
        retained = _SAMPLES[:4]
    if case in ("lost_report", "unsupported"):
        retained = ()
    if case == "zero":
        retained = tuple((fork, kind, repetition, 0) for fork, kind, repetition, _ in _SAMPLES)
    iterations = []
    for fork, kind, repetition, elapsed_ns in retained:
        failed = case == "failed_complete" or case == "non_repeatable" and kind == "measurement" and repetition == 1
        empty = case == "no_samples"
        iterations.append({
            "fork": fork, "kind": kind, "repetition": repetition,
            "elapsed_ns": elapsed_ns, "outcome": "failed" if failed else "completed",
            "report_complete": True,
            "counts": {"discovered": 0 if empty else 1, "started": 0 if empty else 1,
                       "passed": 0 if empty or failed else 1, "skipped": 0, "aborted": 0,
                       "failed": int(failed), "container_failures": 0},
            "tests": () if empty else ({"unique_id": "[engine:junit-jupiter]/[class:Example]/[method:check()]",
                                       "display_name": "check()", "kind": "test",
                                       "status": "failed" if failed else "successful", "reason": ""},),
            "reason": "authored native failure" if failed else "", "diagnostics": (),
        })
    validity = ("complete" if case in ("complete", "failed_complete", "zero") else
                "non_repeatable" if case == "non_repeatable" else
                "no_samples" if case in ("no_samples", "unsupported") else "incomplete")
    outcome = ("completed" if case in ("complete", "zero", "no_samples") else
               "unsupported" if case == "unsupported" else
               "cancelled" if case == "cancelled" else
               "timed_out" if case == "incomplete" else "failed")
    successful = validity == "complete" and outcome == "completed"
    forks_started = 0 if case == "unsupported" else 1 if case == "lost_report" else len({r["fork"] for r in iterations})
    forks = []
    for fork in range(forks_started):
        records = [record for record in iterations if record["fork"] == fork]
        warmups = sum(r["kind"] == "warmup" and r["report_complete"] for r in records)
        measurements = sum(r["kind"] == "measurement" and r["report_complete"] for r in records)
        disposed = not (case == "unresolved_teardown" and fork == 1)
        forks.append({
            "fork": fork, "warmup_completed": warmups, "measurements_completed": measurements,
            "complete": warmups == 2 and measurements == 2 and disposed,
            "disposal_confirmed": disposed,
            "allowed_cpus": None,
            "samples_ns": tuple(r["elapsed_ns"] for r in records if r["kind"] == "measurement"),
            "mean_ms": (0 if case == "zero" else 9 + 2 * fork) if successful else None,
            "median_ms": (0 if case == "zero" else 9 + 2 * fork) if successful else None,
            "min_ms": (0 if case == "zero" else 8 + 2 * fork) if successful else None,
            "max_ms": (0 if case == "zero" else 10 + 2 * fork) if successful else None,
        })
    values = (10, 10, 8, 12, 1.4142135623730951)
    return {
        "outcome": outcome, "validity": validity, "mode": "warmed_jvm",
        "timing_boundary": "junit_launcher_execute", "clock": "java.lang.System.nanoTime",
        "unit": "ns", "state_policy": "native_lifecycle",
        "requested_forks": 2, "requested_warmup_runs": 2, "requested_runs": 2,
        "forks_started": forks_started,
        "forks_completed": sum(f["complete"] and f["disposal_confirmed"] for f in forks),
        "warmup_completed": sum(f["warmup_completed"] for f in forks),
        "measurements_completed": sum(f["measurements_completed"] for f in forks),
        "iterations": tuple(iterations), "forks": tuple(forks),
        "operation_elapsed_s": 20, "comparison": "not_assessed",
        "cancelled": case == "cancelled",
        "disposal_confirmed": all(f["disposal_confirmed"] for f in forks),
        **{key: (0 if case == "zero" else value) if successful else None
           for (key, _), value in zip(_HEADLINES, values)},
    }


def _requested(result):
    return (Metric("Requested forks", result["requested_forks"], basis="progress"),
            Metric("Warmups per fork", result["requested_warmup_runs"], basis="progress"),
            Metric("Measurements per fork", result["requested_runs"], basis="progress"))


def _result_event(result, case):
    successful = result["validity"] == "complete" and result["outcome"] == "completed"
    basis = "measured" if successful else "unsupported" if case == "unsupported" else "unavailable"
    diagnostic = ("resolution-limited zero; no epsilon invented" if case == "zero" else
                  "warmup performed is not JIT convergence" if successful else
                  "native result missing; no complete reports received" if case == "lost_report" else
                  "executed population empty" if case == "no_samples" else
                  "native test population changed; not repeatable" if case == "non_repeatable" else
                  "Python warmed mode unavailable; no tool executed" if case == "unsupported" else
                  "owned teardown unresolved; complete reports are not disposal confirmation" if case == "unresolved_teardown" else
                  "retained reports are diagnostics, not valid samples")
    last = result["iterations"][-1]["counts"] if result["iterations"] else {}
    return _event(
        "warmed/result", "succeeded" if successful else "cancelled" if case == "cancelled" else "failed",
        result["operation_elapsed_s"], completed=result["measurements_completed"],
        total=result["requested_forks"] * result["requested_runs"],
        language="python" if case == "unsupported" else "java",
        work_unit="complete measurement reports across forks",
        message=(f"{_WARMED_BOUNDARY}; {result['state_policy']}; mode {result['mode']}; "
                 f"boundary {result['timing_boundary']}; clock {result['clock']}; unit {result['unit']}; "
                 f"outcome {result['outcome']}; validity {result['validity']}; {diagnostic}; "
                 f"disposal {'confirmed' if result['disposal_confirmed'] else 'unconfirmed'}; "
                 "complete reports are not necessarily valid samples; comparison not assessed"),
        metrics=(
            *_requested(result),
            Metric("Forks started", result["forks_started"], basis="progress"),
            Metric("Forks completed", result["forks_completed"], basis="progress"),
            Metric("Recorded warmups", result["warmup_completed"], basis="progress"),
            Metric("Recorded measurements", result["measurements_completed"], basis="progress"),
            Metric("Last native failures", last.get("failed"), basis="progress" if last else "unavailable"),
            Metric("Last container failures", last.get("container_failures"), basis="progress" if last else "unavailable"),
            *(Metric(label, result[key], " ms", "unavailable" if successful and result[key] is None else basis)
              for key, label in _HEADLINES),
        ),
    )


def _observations(case):
    """Authored facts only, without timing, result parser or statistic estimator."""
    if case == "fresh":
        return (
            _event("fresh/result", "succeeded", 8, completed=2, total=2,
                   work_unit="fresh-process measurements",
                   message=f"{_FRESH_BOUNDARY}; new JVM each repetition; comparison not assessed",
                   metrics=(Metric("Fresh mean", 10, " ms"),)),
        )
    if case not in _CASES:
        raise ValueError(f"unknown authored fixture {case}")
    result = _result(case)
    events = [_event(
        "warmed/setup", "running", 0,
        message=(f"{_WARMED_BOUNDARY}; native_lifecycle; display setup; "
                 "requested counts, not passed samples"),
        metrics=_requested(result), language="python" if case == "unsupported" else "java",
    )]
    warmups = measurements = 0
    for record in result["iterations"]:
        fork, kind, repetition, elapsed_ns = (record[key] for key in ("fork", "kind", "repetition", "elapsed_ns"))
        events.append(_iteration(_progress(fork, kind, repetition, elapsed_ns, entered=True,
                                          warmup_completed=warmups, measurements_completed=measurements)))
        warmups += kind == "warmup"
        measurements += kind == "measurement"
        events.append(_iteration(_progress(fork, kind, repetition, elapsed_ns, entered=False,
                                          outcome=record["outcome"], warmup_completed=warmups,
                                          measurements_completed=measurements)))
    # A started iteration is not an additional completed native record.
    if case in ("incomplete", "cancelled", "lost_report"):
        kind, repetition = ("warmup", 0) if case == "lost_report" else ("measurement", 1)
        entered = _iteration(_progress(0, kind, repetition, None, entered=True,
                                      warmup_completed=warmups, measurements_completed=measurements))
        events.append(entered)
        events.append(replace(
            entered, state="cancelled" if case == "cancelled" else "failed",
            elapsed_s=entered.elapsed_s + 1,
            message=entered.message + "; stopped without a complete additional report",
        ))
    events.append(_result_event(result, case))
    return tuple(events)


@pytest.fixture
def no_execution(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("benchmark rendering fixture must not execute or open a connection")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setenv("TERM", "xterm-256color")


@pytest.mark.parametrize("case", _CASES)
@pytest.mark.parametrize("mode", ("plain", "detailed", "compact"))
@pytest.mark.parametrize("width", (40, 100))
def test_authored_benchmark_states_render_offline(case, mode, width, no_execution, tmp_path):
    events = _observations(case)
    before = tuple(asdict(event) for event in events)
    stream = StringIO()
    console = Console(file=stream, force_terminal=True, color_system=None,
                      width=width, height=35, record=True, markup=False, highlight=False)
    with TerminalPresenter(mode, console=console, clock=lambda: 0) as presenter:
        for event in events:
            presenter(event)
        snapshot = Console(file=StringIO(), width=width, record=True,
                           markup=False, highlight=False)
        snapshot.print(presenter.frame())
        final_frame = snapshot.export_text(clear=False)
        snapshot.save_html(str(tmp_path / f"{case}-{mode}-{width}.html"), clear=False)
    rendered = console.export_text(clear=False)
    flattened = " ".join(rendered.split())
    final = " ".join(final_frame.split())
    boundary = _FRESH_BOUNDARY if case == "fresh" else _WARMED_BOUNDARY
    assert boundary in final
    assert "demo playback" in final
    assert "comparison not assessed" in final
    assert max(map(len, final_frame.splitlines())) <= width
    assert tuple(asdict(event) for event in events) == before
    assert all(event.usage is None and event.provenance == "synthetic" for event in events)
    assert not any(word in flattened.lower() for word in ("speedup", "improved", "faster"))
    if case == "fresh":
        assert "new JVM each repetition" in final and "Fresh mean: 10 ms [demo]" in final
    else:
        assert "Warmups per fork: 2 [progress]" in final
        assert "Measurements per fork: 2 [progress]" in final
        assert "validity " in final and "outcome " in final
        if case in ("complete", "zero"):
            value = 10 if case == "complete" else 0
            assert f"Mean of fork means: {value} ms [demo]" in final
            assert "4 of 4 complete measurement" in final
        else:
            assert "Mean of fork means:" in final and " ms [demo]" not in final
    if mode == "plain":
        assert "\x1b" not in stream.getvalue()


def test_entered_iteration_does_not_claim_an_outcome_or_increment_counts():
    progress = _progress(0, "measurement", 1, None, entered=True,
                         warmup_completed=2, measurements_completed=1)
    entered = _iteration(progress)
    assert progress["fork"] == 0 and progress["repetition"] == 1
    assert progress["outcome"] is None and progress["elapsed_ns"] is None
    assert progress["measurements_completed"] == 1
    assert entered.state == "running" and entered.completed == 1 and entered.total == 2
    assert entered.metrics[0].value is None and entered.metrics[0].basis == "unavailable"
    assert "no native outcome yet" in entered.message
    assert "fork 1/2; measurement iteration 2/2" in entered.message
    assert entered.operation == "warmed/fork-0/measurement"
    assert "raw elapsed" not in entered.message


@pytest.mark.parametrize("case", ("failed_complete", "incomplete", "non_repeatable", "no_samples",
                                  "unsupported", "lost_report", "cancelled", "unresolved_teardown"))
def test_unusable_measurements_keep_headline_unavailable(case):
    events = _observations(case)
    result = events[-1]
    headline = next(m for m in result.metrics if m.name == "Mean of fork means")
    assert headline.value is None and headline.basis != "measured"
    assert result.state != "succeeded"
    if case == "unsupported":
        assert all(event.target.endswith(".py") for event in events)
    if case == "lost_report":
        assert result.completed == 0
        values = {m.name: m.value for m in result.metrics}
        assert values["Recorded measurements"] == 0 and values["Recorded warmups"] == 0
        assert values["Last native failures"] is None
        assert values["Last container failures"] is None


def test_raw_nanoseconds_and_display_elapsed_clock_are_distinct():
    progress = _progress(0, "measurement", 0, 8_000_000, entered=False,
                         warmup_completed=2, measurements_completed=1)
    event = _iteration(progress)
    assert event.elapsed_s == 6 and event.metrics[0].value == 8_000_000
    assert progress["operation_elapsed_s"] == 6 and progress["elapsed_ns"] == 8_000_000
    assert event.metrics[0].unit == " ns" and event.metrics[0].basis == "progress"
    assert "raw elapsed 8000000 ns" in event.message


def test_incomplete_result_retains_partial_counts_without_refilling_summary():
    events = _observations("incomplete")
    result = events[-1]
    assert result.completed == 1 and result.total == 4
    values = {m.name: m.value for m in result.metrics}
    assert values["Recorded warmups"] == 2 and values["Recorded measurements"] == 1
    assert values["Mean of fork means"] is None
    assert any(m.value == 8_000_000 for event in events for m in event.metrics)


def test_zero_is_resolution_limited_not_unavailable_or_free_execution():
    result = _observations("zero")[-1]
    headline = next(m for m in result.metrics if m.name == "Mean of fork means")
    assert headline.value == 0 and headline.basis == "measured"
    assert "resolution-limited zero" in result.message
    assert "no epsilon invented" in result.message


def test_single_fork_has_no_between_fork_uncertainty(no_execution):
    result = _result("complete")
    result.update(requested_forks=1, forks_started=1, forks_completed=1,
                  warmup_completed=2, measurements_completed=2,
                  iterations=result["iterations"][:4], forks=result["forks"][:1],
                  mean_of_fork_means_ms=9, median_of_fork_medians_ms=9,
                  sample_min_ms=8, sample_max_ms=10, between_fork_mean_sd_ms=None)
    single = _result_event(result, "complete")
    stream = StringIO()
    with TerminalPresenter("plain", stream=stream, width=100) as presenter:
        presenter(single)
    rendered = " ".join(stream.getvalue().split())
    assert "Between-fork sample SD: unavailable" in rendered
    assert "Between-fork sample SD: 0" not in rendered


def test_failed_complete_protocol_counts_reports_not_successful_samples():
    result = _result("failed_complete")
    event = _result_event(result, "failed_complete")
    assert result["validity"] == "complete" and result["outcome"] == "failed"
    assert result["forks_completed"] == 2
    assert result["warmup_completed"] == 4 and result["measurements_completed"] == 4
    assert all(r["report_complete"] and r["outcome"] == "failed" for r in result["iterations"])
    assert all(r["counts"]["failed"] == 1 for r in result["iterations"])
    assert event.completed == 4 and event.total == 4 and event.state == "failed"
    assert "complete reports are not necessarily valid samples" in event.message
    assert all(result[key] is None for key, _ in _HEADLINES)
    assert all(f[key] is None for f in result["forks"] for key in ("mean_ms", "median_ms", "min_ms", "max_ms"))
    assert result["forks"][0]["samples_ns"] == (8_000_000, 10_000_000)
    assert any(e.state == "failed" and e.completed == 2 for e in _observations("failed_complete")[:-1])


def test_non_repeatable_completed_failure_retains_raw_report_and_counter():
    result = _result("non_repeatable")
    last = result["iterations"][-1]
    assert last["fork"] == 0 and last["repetition"] == 1 and last["kind"] == "measurement"
    assert last["report_complete"] and last["outcome"] == "failed"
    assert last["counts"]["failed"] == 1 and last["tests"][0]["status"] == "failed"
    assert result["measurements_completed"] == 2 and result["forks_completed"] == 1
    assert result["validity"] == "non_repeatable"
    assert all(result[key] is None for key, _ in _HEADLINES)


def test_empty_native_execution_can_complete_protocol_without_valid_samples():
    result = _result("no_samples")
    assert result["outcome"] == "completed" and result["validity"] == "no_samples"
    assert result["measurements_completed"] == 4 and result["warmup_completed"] == 4
    assert result["forks_completed"] == 2
    assert all(not r["tests"] and all(value == 0 for value in r["counts"].values())
               for r in result["iterations"])
    assert all(result[key] is None for key, _ in _HEADLINES)
    assert all(f["mean_ms"] is None and f["samples_ns"] for f in result["forks"])


@pytest.mark.parametrize("case", tuple(case for case in _CASES if case != "fresh"))
def test_authored_result_uses_exact_units_indices_and_summary_gate(case):
    result = _result(case)
    assert result["mode"] == "warmed_jvm" and result["timing_boundary"] == "junit_launcher_execute"
    assert result["clock"] == "java.lang.System.nanoTime" and result["unit"] == "ns"
    assert result["state_policy"] == "native_lifecycle" and result["comparison"] == "not_assessed"
    assert result["requested_runs"] == 2 and result["requested_warmup_runs"] == 2
    assert all(r["fork"] in (0, 1) and r["repetition"] in (0, 1) for r in result["iterations"])
    assert result["measurements_completed"] == sum(
        r["kind"] == "measurement" and r["report_complete"] for r in result["iterations"])
    if result["outcome"] != "completed" or result["validity"] != "complete":
        assert all(result[key] is None for key, _ in _HEADLINES)
        assert all(f[key] is None for f in result["forks"] for key in ("mean_ms", "median_ms", "min_ms", "max_ms"))


def test_missing_report_keeps_unknown_native_fields_distinct_from_report_counts():
    result = _result("lost_report")
    assert result["forks_started"] == 1 and result["forks_completed"] == 0
    assert not result["iterations"] and not result["forks"][0]["samples_ns"]
    progress = _progress(0, "warmup", 0, None, entered=True)
    assert progress["warmup_completed"] == 0 and progress["elapsed_ns"] is None
    event = _result_event(result, "lost_report")
    values = {m.name: m.value for m in event.metrics}
    assert values["Recorded warmups"] == 0 and values["Recorded measurements"] == 0
    assert values["Last native failures"] is None and values["Last container failures"] is None


def test_native_container_failure_is_not_manufactured_as_a_test_failure():
    result = _result("failed_complete")
    last = result["iterations"][-1]
    last["counts"].update(discovered=0, started=0, passed=0, failed=0, container_failures=1)
    last["tests"] = ({"unique_id": "[engine:junit-jupiter]/[class:Example]",
                      "display_name": "Example", "kind": "container", "status": "failed",
                      "reason": "authored native container failure"},)
    event = _result_event(result, "failed_complete")
    values = {m.name: m.value for m in event.metrics}
    assert values["Last native failures"] == 0 and values["Last container failures"] == 1
    assert values["Mean of fork means"] is None
    assert last["tests"][0]["kind"] == "container"


def test_unconfirmed_teardown_does_not_complete_a_fork_or_supply_headlines():
    result = _result("unresolved_teardown")
    assert result["measurements_completed"] == 4 and result["warmup_completed"] == 4
    assert result["forks_started"] == 2 and result["forks_completed"] == 1
    assert result["forks"][0]["complete"] and not result["forks"][1]["complete"]
    assert result["forks"][0]["disposal_confirmed"]
    assert not result["forks"][1]["disposal_confirmed"] and not result["disposal_confirmed"]
    assert result["validity"] == "incomplete" and result["outcome"] == "failed"
    assert all(result[key] is None for key, _ in _HEADLINES)
    event = _result_event(result, "unresolved_teardown")
    assert event.completed == 4 and event.state == "failed"
    assert "disposal unconfirmed" in event.message
