"""RunView: what a display shows of a run, as plain data, and the events
rebuilt from their JSON lines."""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pytest

from testuary.presentation import DisplayEvent, Metric, Usage
from testuary.presentation.view import RunView, phase_label


def _event(**fields):
    values = dict(run_id="run-1", target="calc.py", phase="coverage", operation="tool",
                  state="running", elapsed_s=1.0)
    values.update(fields)
    return DisplayEvent(**values)


def test_an_event_comes_back_from_its_json_line():
    event = _event(
        level="phase", label="Coverage phase", message="Phase finished",
        state="failed", failure="outcome", completed=2, total=4, work_unit="phases",
        metrics=(Metric("line coverage", 96.5, "%", "measured", 90.0),
                 Metric("branch coverage", None, "%", "unavailable")),
        usage=Usage(logical_requests=3, physical_requests=4, input_tokens=100,
                    output_tokens=20, cached_input_tokens=5, cost_usd=0.25,
                    logical_request_limit=20),
        provenance="synthetic", log_path="out/logs/run.log")
    assert DisplayEvent.from_dict(asdict(event)) == event
    assert DisplayEvent.from_dict(json.loads(json.dumps(asdict(event)))) == event
    assert DisplayEvent.from_dict(asdict(_event())) == _event()


def test_a_line_that_is_no_event_is_refused():
    with pytest.raises(ValueError):
        DisplayEvent.from_dict({**asdict(_event()), "state": "unknown"})
    with pytest.raises(TypeError):
        DisplayEvent.from_dict({**asdict(_event()), "extra": 1})


def test_the_snapshot_holds_the_layers_operations_and_usage():
    view = RunView()
    assert view.snapshot(4).empty
    view.add(_event(phase="pipeline", operation="run", level="run", completed=1, total=4,
                    work_unit="phases done", target="all targets", log_path="logs/run.log"))
    view.add(_event(operation="target", level="target", completed=0, total=1,
                    work_unit="targets", phase="pipeline"))
    view.add(_event(operation="phase/run-1", level="phase", completed=1, total=4,
                    work_unit="phases"))
    view.add(_event(operation="request/1", message="Sending model request",
                    usage=Usage(logical_requests=1, physical_requests=1,
                                logical_request_limit=20)))
    snapshot = view.snapshot(4)
    assert [layer.name for layer in snapshot.layers] == ["Overall", "Log", "Target", "Phase"]
    assert snapshot.layers[0].text.startswith("Overall  RUNNING")
    assert snapshot.layers[0].bar.ratio == 0.25 and "1 of 4 phases done" in snapshot.layers[0].bar.figures
    assert snapshot.layers[1].dim and "logs/run.log" in snapshot.layers[1].text
    assert snapshot.layers[3].text == "Phase    Coverage phase | RUNNING"
    assert [(item.state, item.name) for item in snapshot.operations] == [
        ("RUNNING", "Model request 1")]
    assert snapshot.message == "Sending model request" and snapshot.metrics is None
    assert snapshot.usage == "Model use, Coverage phase: model requests 1 of 20"


def test_an_ended_run_shows_no_phase_use():
    """Once the run has ended, the run's summary holds its model use: the
    pinned view no longer shows the last phase's."""
    view = RunView()
    view.add(_event(phase="pipeline", operation="run", level="run", target="all targets"))
    view.add(_event(operation="phase/run-1", level="phase"))
    view.add(_event(operation="request/1", usage=Usage(logical_requests=1, physical_requests=1,
                                                       logical_request_limit=20)))
    assert view.snapshot(4).usage == "Model use, Coverage phase: model requests 1 of 20"
    view.add(_event(phase="pipeline", operation="run", level="run", target="all targets",
                    state="succeeded"))
    assert view.snapshot(4).usage is None


def test_every_text_has_its_credentials_removed():
    view = RunView(secrets=("plain-known-value",))
    planted = "api_key=sk-test-123 https://u:pw@example.test/v1?token=abc plain-known-value"
    shown = view.add(_event(operation="request/1", state="failed", message=planted))
    snapshot = view.snapshot(4)
    for text in (shown.line, snapshot.message, *snapshot.issues):
        for secret in ("sk-test-123", "u:pw", "token=abc", "plain-known-value"):
            assert secret not in text


def test_every_text_passes_the_displays_clean():
    """A key with no recognizable name is removed by the display's own
    clean, which the terminal fills with the run's redaction."""
    planted = "zq81Plain7731value"
    view = RunView(clean=lambda value: value.replace(planted, "[redacted]"))
    shown = view.add(_event(operation="request/1", state="failed",
                            message=f"API key: {planted} was refused"))
    snapshot = view.snapshot(4)
    texts = (shown.line, snapshot.message, *snapshot.issues,
             *(operation.name for operation in snapshot.operations))
    assert all(planted not in text for text in texts)
    assert "API key: [redacted] was refused" in shown.line


def test_the_terminal_hands_its_redaction_to_the_view():
    import io

    from rich.console import Console

    from testuary.presentation.terminal import TerminalPresenter

    planted = "zq81Plain7731value"
    stream = io.StringIO()
    console = Console(file=stream, width=100, force_terminal=False, color_system=None)
    with TerminalPresenter("plain", console=console) as presenter:
        presenter.redact_with(lambda value: value.replace(planted, "[redacted]"))
        presenter(_event(operation="request/1", state="failed",
                         message=f"API key: {planted} was refused"))
        snapshot = presenter.view.snapshot(4)
    assert planted not in stream.getvalue() and "[redacted]" in stream.getvalue()
    assert planted not in (snapshot.message or "")


def test_the_view_decides_what_is_shown_and_logged():
    view = RunView()
    first = view.add(_event(operation="step/consolidation", level="step"))
    assert first.shown and first.logged
    assert not view.add(_event(operation="step/consolidation", level="step")).shown
    operation = view.add(_event(operation="request/2"))
    assert operation.shown and not operation.logged
    finished = view.add(_event(operation="request/2", state="succeeded"))
    assert finished.shown and finished.logged
    # 1 and 2 of 1000 are both 0%: a chronological view waits for the next ten percent.
    counted = dict(operation="mutants", work_unit="mutants", total=1000)
    view.add(_event(completed=1, **counted), throttle_counts=True)
    assert not view.add(_event(completed=2, elapsed_s=2.0, **counted),
                        throttle_counts=True).shown
    assert view.add(_event(completed=2, elapsed_s=2.0, **counted)).shown


def test_a_retired_observation_leaves_no_record_behind():
    """Only the newest 64 finished observations stay in view, and every
    record kept per observation goes with the ones that retire."""
    view = RunView()
    for index in range(80):
        view.add(_event(operation=f"tool/{index}", state="running", completed=0, total=2))
        view.add(_event(operation=f"tool/{index}", state="succeeded", completed=2, total=2))
    kept = set(view.latest())
    assert len(kept) == 64
    for record in (view._printed, view._updated, view._started, view._began,
                   view._done_before):
        assert set(record) <= kept


def test_a_phase_is_labeled_as_the_console_labels_it():
    view = RunView()
    view.add(_event(operation="phase"))
    assert phase_label("coverage") == "Coverage phase"
    assert phase_label("coverage") in json.dumps(asdict(view.snapshot(4)))
    assert phase_label("seed_processing") == "Seed processing"


def test_finishing_is_the_fourth_phase_in_the_phase_row():
    from testuary.presentation.view import FINISHING

    view = RunView()
    view.add(_event(phase="pipeline", operation="run", level="run", completed=3, total=4,
                    work_unit="phases done", target="all targets"))
    view.add(_event(phase="mutation", operation="phase/run-1", level="phase", state="succeeded",
                    completed=3, total=4, work_unit="phases"))
    view.add(_event(phase="finishing", operation=FINISHING, level="step",
                    label="Finishing", message="Phase started", completed=3, total=4,
                    work_unit="phases"))
    snapshot = view.snapshot(4)
    assert [layer.name for layer in snapshot.layers] == ["Overall", "Target", "Phase"]
    assert snapshot.layers[2].text == (
        "Phase    Finishing: measuring the delivered suite and writing the results | RUNNING")
    assert "3 of 4 phases" in snapshot.layers[2].bar.figures
    assert "3 of 4 phases done" in snapshot.layers[0].bar.figures
    line = view.add(_event(phase="finishing", operation=FINISHING, level="step",
                           label="Finishing", state="succeeded")).line
    assert "calc.py | Finishing" in line and "Finishing | Finishing" not in line


def test_a_failed_run_ends_what_it_left_running_and_names_its_cause():
    view = RunView()
    view.add(_event(phase="pipeline", operation="run", level="run", target="all targets"))
    view.add(_event(operation="target", level="target", phase="pipeline"))
    view.add(_event(phase="starter", operation="phase/run-1", level="phase"))
    view.add(_event(phase="starter", operation="request/1", message="Sending model request"))
    view.add(_event(phase="starter", operation="suite-processing", state="failed",
                    message="The model service refused the request. It answered 401."))
    view.add(_event(phase="pipeline", operation="run", level="run", target="all targets",
                    state="failed", message="Run ended with an execution error"))
    snapshot = view.snapshot(4)
    phase = next(layer for layer in snapshot.layers if layer.name == "Phase")
    assert phase.text == "Phase    Starter phase | FAILED"
    assert not [item for item in snapshot.operations if item.state == "RUNNING"]
    assert snapshot.cause == "The model service refused the request."
    view = RunView()
    view.add(_event(phase="pipeline", operation="run", level="run", target="all targets",
                    state="failed", message="Run ended with an execution error. More text."))
    assert view.snapshot(4).cause == "Run ended with an execution error."
    # A run names no failure while it goes on, and none once it has
    # succeeded: what failed on its way stays in the log.
    view = RunView()
    view.add(_event(phase="starter", operation="request/1", state="failed",
                    message="Request limit reached"))
    assert view.snapshot(4).cause is None
    view.add(_event(phase="pipeline", operation="run", level="run", target="all targets",
                    state="succeeded", message="Run completed"))
    assert view.snapshot(4).cause is None
    quiet = RunView()
    quiet.add(_event(phase="pipeline", operation="run", level="run", target="all targets",
                     state="succeeded", message="Run completed"))
    assert quiet.snapshot(4).cause is None
    # A run without a suite that meets the minimums names a file's own
    # reason, not a failure it recovered from before.
    incomplete = RunView()
    incomplete.add(_event(phase="coverage", operation="step/revise", level="step", state="failed",
                          message="Request limit reached"))
    incomplete.add(_event(phase="pipeline", operation="target", level="target", state="failed",
                          failure="outcome", message="No complete suite: coverage 66.7% is below 75%"))
    incomplete.add(_event(phase="pipeline", operation="run", level="run", target="all targets",
                          state="failed", failure="outcome", message="Run finished: 0 of 1 file."))
    assert incomplete.snapshot(4).cause == "No complete suite: coverage 66.7% is below 75%"
    # A file without a suite because a phase failed: the failure it traces
    # back to, such as a model service that could not be reached.
    traced = RunView()
    traced.add(_event(phase="starter", operation="request/1", state="failed",
                      message="The model service could not be reached after 3 attempts: check it."))
    traced.add(_event(phase="starter", operation="phase/run-1", level="phase", state="failed",
                      message="Phase failed"))
    traced.add(_event(phase="pipeline", operation="target", level="target", state="failed",
                      failure="outcome", message="No complete suite: no tests from the coverage phase"))
    traced.add(_event(phase="pipeline", operation="run", level="run", target="all targets",
                      state="failed", failure="outcome", message="Run finished: 0 of 1 file."))
    assert traced.snapshot(4).cause == "The model service could not be reached after 3 attempts: check it."


def test_a_log_path_below_the_working_dir_is_shown_relative(tmp_path, monkeypatch):
    from testuary.presentation.view import shown_path

    monkeypatch.chdir(tmp_path)
    inside = tmp_path / ".testuary" / "results" / "local" / "logs" / "run.log"
    outside = tmp_path.parent / "elsewhere" / "run.log"
    assert shown_path(str(inside)) == str(Path(".testuary", "results", "local", "logs", "run.log"))
    assert shown_path(str(outside)) == str(outside)
    view = RunView()
    view.add(_event(phase="pipeline", operation="run", level="run", target="all targets",
                    log_path=str(inside)))
    log = next(layer for layer in view.snapshot(4).layers if layer.name == "Log")
    assert str(tmp_path) not in log.text and "run.log" in log.text


def test_a_running_files_time_follows_the_runs_clock():
    """A file's last event is often its start: while it runs, its time is
    the run's latest observation less its start, not zero. A finished one
    keeps its own."""
    view = RunView()
    view.add(_event(phase="pipeline", operation="run", level="run", target="all targets",
                    state="running", elapsed_s=0.0))
    view.add(_event(phase="pipeline", operation="target", level="target", state="running",
                    elapsed_s=0.0, completed=0, total=1, work_unit="targets"))
    view.add(_event(phase="coverage", operation="phase/run-1", level="phase", state="running",
                    elapsed_s=1.0, completed=0, total=3, work_unit="phases"))
    view.add(_event(phase="coverage", operation="tool", state="running", elapsed_s=91.0))
    rows = {layer.name: layer for layer in view.snapshot(4).layers}
    assert "00:01:31" in rows["Target"].bar.figures and "00:01:30" in rows["Phase"].bar.figures
    view.add(_event(phase="pipeline", operation="target", level="target", state="succeeded",
                    elapsed_s=95.0, completed=1, total=1, work_unit="targets"))
    view.add(_event(phase="coverage", operation="tool", state="running", elapsed_s=200.0))
    assert "00:01:35" in {layer.name: layer for layer in view.snapshot(4).layers}["Target"].bar.figures


def test_a_diagram_counts_the_starter_suites_and_the_loops_requests_and_values():
    view = RunView()
    assert view.step_counts("starter") == {"steps": {}, "phase": None, "values": []}
    view.add(_event(phase="starter", operation="phase", level="phase", elapsed_s=0.5))
    for done in (1, 2):
        view.add(_event(phase="starter", operation="suite-processing", completed=done, total=3,
                        work_unit="starter suites accepted"))
    view.add(_event(phase="starter", operation="starter-measurement", completed=1, total=2,
                    work_unit="starter suites measured"))
    assert view.step_counts("starter")["steps"] == {"generate": "2 of 3 suites", "measure": "1 of 2 suites"}
    view.add(_event(phase="coverage", operation="phase", level="phase", elapsed_s=2.0))
    view.add(_event(phase="coverage", operation="request/4", state="succeeded",
                    usage=Usage(logical_requests=4, logical_request_limit=20)))
    view.add(_event(phase="coverage", operation="phase", level="phase", state="succeeded",
                    elapsed_s=9.0, metrics=(Metric("line coverage", 98.25, "%"),
                                            Metric("branch coverage", None, "%", "unavailable"))))
    counts = view.step_counts("coverage")
    assert counts["phase"] == "model requests 4 of 20"
    # Only measured values, with their basis, and nothing for an unavailable one.
    assert counts["values"] == ["line coverage: 98.2% [measured]"]
    # A loop without a configured limit names its requests alone.
    view.add(_event(phase="mutation", operation="request/1", state="running",
                    usage=Usage(logical_requests=1)))
    assert view.step_counts("mutation")["phase"] == "1 model request"
    # The next run of the file starts the phase's counters again.
    view.add(_event(phase="starter", operation="phase", level="phase", state="succeeded",
                    elapsed_s=10.0))
    view.add(_event(phase="starter", operation="phase", level="phase", elapsed_s=11.0))
    assert view.step_counts("starter")["steps"] == {}
    assert view.step_counts("coverage", target="other.py")["phase"] is None


def test_the_step_a_limit_ends_names_the_limit():
    """The step names the limit in its message's words, without the
    request count. An older event file's "Limit reached" stays generic."""
    view = RunView()
    view.add(_event(phase="mutation", operation="step/assess", level="step"))
    lines = [view.add(_event(phase="mutation", operation="step/assess", level="step",
                             state="limit_reached", message=message, elapsed_s=2.0)).line
             for message in ("Request limit reached (1 of 1)", "Token limit reached",
                             "Time limit reached", "Limit reached",
                             "Reply received. Token limit reached")]
    assert ["REQUEST LIMIT REACHED" in lines[0], "(1 OF 1)" in lines[0]] == [True, False]
    assert "TOKEN LIMIT REACHED" in lines[1] and "TIME LIMIT REACHED" in lines[2]
    assert "LIMIT REACHED" in lines[3]
    assert not any(word in lines[3] for word in ("REQUEST", "TOKEN", "TIME LIMIT"))
    assert "TOKEN LIMIT REACHED" in lines[4] and "REPLY RECEIVED" not in lines[4]


def test_a_loop_that_ends_at_its_limit_reads_limit_reached_and_is_no_issue():
    """A limit the run's settings set is an ordinary end: the refused request
    and the step the loop ends on say LIMIT REACHED in the log, and no
    Recent issue line names them. A failure still does."""
    view = RunView()
    view.add(_event(phase="mutation", operation="step/assess", level="step"))
    request = view.add(_event(phase="mutation", operation="request/21", state="limit_reached",
                              message="Request limit reached", elapsed_s=2.0))
    step = view.add(_event(phase="mutation", operation="step/assess", level="step",
                           state="limit_reached", message="Step ended", elapsed_s=2.0))
    assert request.logged and "LIMIT REACHED" in request.line and "FAILED" not in request.line
    assert step.logged and "LIMIT REACHED" in step.line
    assert view.snapshot(4).issues == ()
    failed = view.add(_event(phase="mutation", operation="request/22", state="failed",
                             message="Model request ended with an error", elapsed_s=3.0))
    assert "FAILED" in failed.line
    assert view.snapshot(4).issues == (f"Recent issue: {failed.line}",)


@pytest.mark.parametrize("ending", ["succeeded", "failed", "cancelled"])
def test_only_a_run_that_succeeded_ends_without_its_recent_issues(ending):
    """A run that succeeded recovered from every failure it showed, whose
    lines stay in the log above the view: its final screen names no Recent
    issue beside Overall SUCCEEDED. A run that ended otherwise keeps them."""
    view = RunView()
    view.add(_event(phase="pipeline", operation="run", level="run", target="all targets"))
    failed = view.add(_event(operation="step/edit", level="step", label="Model edits the tests",
                             state="failed", message="Edit refused", elapsed_s=2.0))
    assert failed.logged and "FAILED" in failed.line
    assert view.snapshot(4).issues == (f"Recent issue: {failed.line}",)
    view.add(_event(phase="pipeline", operation="run", level="run", target="all targets",
                    state=ending, elapsed_s=3.0))
    issues = view.snapshot(4).issues
    if ending == "succeeded":
        assert issues == ()
    else:
        assert f"Recent issue: {failed.line}" in issues


def test_a_mutation_measurements_count_reads_as_mutants_run_not_as_a_score():
    """The measurement's count is of mutants run, not killed: no percentage
    stands beside it in the line or the bar, where it read as a mutation
    score (36 of 37 mutants (97%) beside a score of 91.9%). Other counts keep
    their percentage."""
    from langadapter.mutation import MutationProgress

    from testuary.presentation.measurement import mutation_progress_event

    view = RunView()
    shown = view.add(mutation_progress_event(MutationProgress("mutants", 36, 37, 12.0), run_id="run-1",
                                             target="calc.py", elapsed_s=30.0, phase="mutation"))
    assert "36 of 37 mutants run" in shown.line and "%" not in shown.line
    (row,) = view.snapshot(4).operations
    assert row.progress == "36 of 37 mutants run" and "%" not in row.bar.figures
    counted = view.add(_event(operation="suite-processing", completed=1, total=4,
                              work_unit="starter suites accepted", elapsed_s=31.0))
    assert "1 of 4 starter suites accepted (25%)" in counted.line


def test_a_refused_final_answer_is_one_line_and_a_usable_one_adds_none():
    """The final answer's check is no pinned row: a refused answer is one
    logged line when it happens, a usable answer prints nothing, and the
    loop's end says only when no usable answer came."""
    view = RunView()
    view.add(_event(operation="phase", level="phase"))
    refused = view.add(_event(operation="output-validation", state="retrying", label="Final answer",
                              message="The model's final answer was not usable. It was told why "
                                      "and continues."))
    assert refused.shown and refused.logged and "Final answer | The model's final answer" in refused.line
    assert view.snapshot(4).operations == ()
    assert not view.add(_event(operation="output-validation", state="succeeded", label="Final answer",
                               message="Usable final answer received", elapsed_s=2.0)).shown
    assert view.snapshot(4).operations == ()
    ended = view.add(_event(operation="output-validation", state="limit_reached", label="Final answer",
                            message="Request limit reached before a usable final answer", elapsed_s=3.0))
    assert ended.shown and "Request limit reached before a usable final answer" in ended.line
    assert "REQUEST LIMIT REACHED" in ended.line


def test_requests_read_as_model_requests_against_the_limit_and_sent_only_when_they_differ():
    from testuary.presentation.view import usage_text

    assert usage_text(Usage(logical_requests=11, physical_requests=11, logical_request_limit=20,
                            input_tokens=3618, output_tokens=33)) == (
        "model requests 11 of 20, 3,618 input tokens, 33 output tokens")
    assert usage_text(Usage(logical_requests=11, physical_requests=13, logical_request_limit=20)) == (
        "model requests 11 of 20 (13 sent, 2 sent again after connection errors)")
    assert usage_text(Usage(logical_requests=11, physical_requests=11, logical_request_limit=20),
                      both_counts=True) == "model requests 11 of 20 (11 sent)"
    assert usage_text(Usage(logical_requests=25, input_tokens=282729, cached_input_tokens=148910,
                            output_tokens=133271)) == (
        "25 model requests, 282,729 input tokens (148,910 cache-hit tokens), 133,271 output tokens")
    assert usage_text(Usage(logical_request_limit=8)) == "request limit 8"
    assert usage_text(None) == "not reported"
    # A progress line labels a model use that reports nothing itself.
    assert RunView().add(_event(usage=Usage())).line.endswith("| Model use not reported")
    limit = _event(operation="request/21", state="limit_reached", message="Request limit reached",
                   usage=Usage(logical_requests=20, logical_request_limit=20))
    line = RunView().add(limit).line
    assert "| Model requests | Request limit reached | model requests 20 of 20" in line
    assert "21" not in line


def test_a_tool_row_names_the_tool_in_plain_words_and_its_own_name_only_in_detail():
    shown = _event(operation="tool/1f2e", label="Model reads the tests", message="Running read_tests")
    assert RunView().add(shown).line.endswith("| Model reads the tests")
    assert RunView(detailed=True).add(shown).line.endswith("| Model reads the tests | Running read_tests")
    view = RunView()
    view.add(shown)
    assert view.snapshot(4).message is None
