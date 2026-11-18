"""Plain streams, Rich snapshots, failures, concurrency and lifecycle isolation."""
from __future__ import annotations

import io
import json
import threading
from dataclasses import asdict, replace

import pytest
from rich.console import Console
from rich.text import Text

from testuary.presentation import DisplayEvent, Metric, TerminalPresenter, Usage
from testuary.presentation.terminal import safe_text


@pytest.fixture(autouse=True)
def terminal_environment(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")


def event(**overrides):
    fields = dict(run_id="run-1", target="src/calc.py", phase="mutation",
                  operation="measure", state="running", elapsed_s=4)
    return DisplayEvent(**{**fields, **overrides})


def work(presenter, item):
    """The work cell the compact view draws for ``item``'s operation."""
    presenter.view.add(item)
    return presenter._work(presenter.view.snapshot(1).operations[-1])


def console(width=120, *, tty=False):
    stream = io.StringIO()
    return Console(file=stream, width=width, height=32, force_terminal=tty,
                   color_system=None, no_color=True,
                   record=True, highlight=False, markup=False), stream


@pytest.mark.parametrize("mode", ["auto", "compact", "detailed", "plain"])
def test_redirected_modes_are_plain_without_cursor_sequences(mode):
    output, stream = console()
    with TerminalPresenter(mode, console=output) as presenter:
        assert presenter.mode == "plain"
        presenter(event(state="waiting", message="Waiting for model"))
        presenter(event(state="retrying", message="Connection retry"))
        presenter(event(state="failed", message="Native measurement failed"))
    text = stream.getvalue()
    assert "WAITING" in text and "RETRYING" in text and "FAILED" in text
    assert "\x1b" not in text and "\r" not in text
    assert "0%" not in text and "100%" not in text


@pytest.mark.parametrize("mode", ["compact", "detailed", "plain"])
def test_rendering_does_not_mutate_observations_accounting_or_global_streams(mode):
    import sys

    output, _ = console(tty=True)
    observation = event(completed=3, total=4, metrics=(Metric("Mutation", 0, "%"),),
                        usage=Usage(logical_requests=1, physical_requests=3, input_tokens=0))
    before = json.dumps(asdict(observation), sort_keys=True)
    stdout, stderr = sys.stdout, sys.stderr
    with TerminalPresenter(mode, console=output) as presenter:
        presenter(observation)
        assert sys.stdout is stdout and sys.stderr is stderr
    assert json.dumps(asdict(observation), sort_keys=True) == before
    assert observation.usage.physical_requests == 3


def test_missing_and_incomplete_metrics_are_not_displayed_as_zero():
    output, stream = console(width=200)
    with TerminalPresenter(console=output) as presenter:
        presenter(event(metrics=(Metric("Line coverage", 0, "%"),
                                 Metric("Mutation", None, "%", "incomplete")),
                        usage=Usage(input_tokens=0)))
    text = stream.getvalue()
    assert "Line coverage: 0% [measured]" in text
    assert "Mutation: incomplete" in text
    assert "Mutation: 0" not in text
    assert "0 input tokens" in text
    assert "cache-hit tokens" not in text and "cost $0" not in text


def test_plain_snapshot_names_actual_work_not_request_limit():
    output, stream = console(width=200)
    with TerminalPresenter("plain", console=output) as presenter:
        presenter(event(completed=2, total=8, work_unit="mutants"))
    assert stream.getvalue() == (
        "00:00:04 RUNNING | src/calc.py | Mutation phase | Measure | 2 of 8 mutants (25%)\n")


@pytest.mark.parametrize("width", [30, 60])
def test_known_work_has_a_proportional_bar_with_percentage_and_count(width):
    stream = io.StringIO()
    output = Console(file=stream, width=width, force_terminal=True, color_system=None,
                     highlight=False, markup=False, legacy_windows=False)
    presenter = TerminalPresenter(console=output)
    with output.capture() as captured:
        output.print(work(presenter, event(completed=2, total=8)))
    line = captured.get().rstrip("\n")
    bar, _, stats = line.partition(" ")
    assert stats.strip() == "25%  2 of 8 items" and len(line) <= width
    full = bar.count("\u2588") + bar.count("#")
    # Rounded down: a quarter of the cells, never more.
    assert full == len(bar) // 4 and len(bar) >= 6


@pytest.mark.parametrize("width", [40, 100])
@pytest.mark.parametrize("completed", [2, 8])
def test_work_bar_cannot_overwrite_the_next_frame_message(width, completed):
    output, stream = console(width)
    presenter = TerminalPresenter(console=output)
    message = "BOUNDARY starts here and remains visible"
    presenter(event(completed=completed, total=8, message=message,
                    metrics=(Metric("Measured value", 7, " ms"),)))
    stream.seek(0)
    stream.truncate(0)
    output.print(presenter.frame())
    rendered = " ".join(stream.getvalue().split())
    assert message in rendered and "Measured value: 7 ms [measured]" in rendered


@pytest.mark.parametrize("counts", [{}, {"completed": 1}, {"total": 8},
                                   {"completed": 0, "total": 0}])
def test_unknown_or_empty_work_never_uses_a_progress_bar(counts):
    output, _ = console()
    assert isinstance(work(TerminalPresenter(console=output), event(**counts)), Text)


@pytest.mark.parametrize("width", [20, 44, 80])
def test_known_work_bar_and_labels_fit_the_terminal(width):
    output, stream = console(width)
    presenter = TerminalPresenter(console=output)
    presenter(event(completed=2, total=8, work_unit="mutants"))
    stream.seek(0)
    stream.truncate(0)
    output.print(presenter.frame())
    assert all(len(line) <= width for line in stream.getvalue().splitlines())
    assert "25%" in stream.getvalue()


@pytest.mark.parametrize("counts,expected", [
    ({"completed": 3}, "3 items, total unknown"),
    ({"total": 7}, "? of 7 items"), ({"completed": 0, "total": 0}, "0 of 0 items"),
])
def test_unknown_or_empty_work_has_no_fabricated_percent(counts, expected):
    output, stream = console()
    with TerminalPresenter(console=output) as presenter:
        presenter(event(**counts))
    assert expected in stream.getvalue() and "%" not in stream.getvalue()


def test_observations_without_counts_show_no_work_field():
    output, stream = console()
    with TerminalPresenter(console=output) as presenter:
        presenter(event(message="Phase started"))
        frame_output, frame_stream = console()
        frame_output.print(presenter.frame())
    line = stream.getvalue()
    assert "Mutation phase | Measure | Phase started" in line
    for text in (line, frame_stream.getvalue()):
        assert "work total unknown" not in text and "%" not in text


@pytest.mark.parametrize("operation,label", [
    ("request/2893a3fc34b0499bb11f37988d374717/1", "Model request 1"),
    ("request/3", "Model request 3"),
    ("tool/1b4e28ba-2fa1-11d2-883f-0016d3cca427", "Tool"),
    ("phase/run-1", None),
])
def test_internal_identities_appear_only_at_the_debug_level(operation, label):
    observation = event(operation=operation, message="Sending model request")
    shown = {}
    for mode in ("plain", "debug"):
        output, stream = console()
        with TerminalPresenter(mode, console=output) as presenter:
            presenter(observation)
            frame_output, frame_stream = console(width=160)
            frame_output.print(presenter.frame())
        shown[mode] = stream.getvalue() + frame_stream.getvalue()
    readable = label or "Mutation phase"
    assert readable in shown["plain"] and readable in shown["debug"]
    assert operation not in shown["plain"] and "run-1" not in shown["plain"]
    assert f"run-1 | src/calc.py | mutation / {operation}" in shown["debug"]


def test_distinct_identities_with_the_same_label_stay_separate():
    output, _ = console(tty=True)
    with TerminalPresenter("compact", console=output) as presenter:
        first = event(operation="request/a/1")
        presenter(first)
        presenter(replace(first, operation="request/b/1"))
        assert len(presenter.view.latest()) == 2


@pytest.mark.parametrize("state", ["running", "waiting", "retrying", "succeeded", "failed", "cancelled"])
def test_not_yet_measured_metrics_are_omitted_only_while_work_runs(state):
    metrics = (Metric("Line coverage", None, "%", "unavailable"),
               Metric("Mutation", 40, "%", "progress"),
               Metric("Branch coverage", None, "%", "incomplete"))
    output, stream = console()
    with TerminalPresenter(console=output) as presenter:
        presenter(event(state=state, metrics=metrics))
        frame_output, frame_stream = console(width=200)
        frame_output.print(presenter.frame())
    for text in (stream.getvalue(), frame_stream.getvalue()):
        assert "Mutation: 40% [progress]" in text and "Branch coverage: incomplete" in text
        assert ("Line coverage: unavailable" in text) is (state in ("succeeded", "failed", "cancelled"))


def test_duplicate_polling_is_suppressed_but_sparse_wait_heartbeat_is_retained():
    output, stream = console(width=200)
    with TerminalPresenter(console=output) as presenter:
        waiting = event(state="waiting", message="Waiting for provider", elapsed_s=1)
        for elapsed in (1, 2, 3, 10, 31, 32):
            presenter(replace(waiting, elapsed_s=elapsed))
        for _ in range(5):
            presenter(replace(waiting, state="failed", elapsed_s=33))
    assert stream.getvalue().count("WAITING") == 2
    assert stream.getvalue().count("FAILED") == 1


@pytest.mark.parametrize("width", [20, 32, 48, 80, 120])
def test_static_frame_fits_narrow_and_normal_terminals(width):
    output, stream = console(width)
    presenter = TerminalPresenter(console=output)
    presenter(event(target="a/very/long/package/name/Calculator.java",
                    state="failed", message="The native tool did not produce a complete result",
                    metrics=(Metric("Mutation", None, "%", "incomplete"),)))
    stream.truncate(0)
    stream.seek(0)
    output.print(presenter.frame())
    text = stream.getvalue()
    assert all(len(line) <= width for line in text.splitlines())
    assert "FAILED" in text and "incomplete" in text


def test_failure_remains_visible_when_an_independent_operation_starts():
    output, stream = console(tty=True)
    with TerminalPresenter("compact", console=output) as presenter:
        presenter(event(state="failed", message="Native report missing"))
        presenter(event(target="src/other.py", state="running", elapsed_s=8))
        snapshot, snapshot_stream = console()
        snapshot.print(presenter.frame())
        assert "Native report missing" in snapshot_stream.getvalue()
    assert "Native report missing" in stream.getvalue()


@pytest.mark.parametrize("width", [44, 80])
@pytest.mark.parametrize("height", [4, 32])
@pytest.mark.parametrize("state", ["succeeded", "failed", "cancelled"])
def test_latest_terminal_outcome_keeps_a_row_without_finishing_other_operations(
    width, height, state,
):
    output, _ = console(width)
    output.height = height
    presenter = TerminalPresenter("plain", console=output)
    running = [event(operation=f"seed-{i}") for i in range(5)]
    outcome = event(phase="pipeline", operation="finalize", state=state)
    before = [asdict(item) for item in (*running, outcome)]
    for item in (*running, outcome):
        presenter(item)

    # The frame draws the operations of the view's snapshot at the
    # terminal's row limit.
    capacity = max(1, min(4, height // 4))
    selected = [row.event for row in presenter.view.snapshot(capacity).operations]
    snapshot, stream = console(width)
    snapshot.print(presenter.frame())

    expected = running[-(capacity - 1):] + [outcome] if capacity > 1 else [outcome]
    assert selected == expected
    hidden = len(running) - (capacity - 1)
    assert f"{hidden} more running" in stream.getvalue()
    assert [asdict(item) for item in presenter.view.latest().values()] == before
    assert all(item.state == "running" for item in running)


def test_independent_concurrent_operations_keep_their_identities():
    output, _ = console()
    with TerminalPresenter(console=output) as presenter:
        threads = [threading.Thread(target=presenter,
                                    args=(event(operation=f"seed-{i}"),))
                   for i in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        snapshot, stream = console(width=150)
        snapshot.print(presenter.frame())
    assert all(f"Seed {i}" in stream.getvalue() for i in range(3))


@pytest.mark.parametrize("exception,marker", [(RuntimeError("native failure"), "FAILED"),
                                             (KeyboardInterrupt(), "STOPPED")])
def test_execution_exceptions_and_cancellation_propagate_and_restore_terminal(exception, marker):
    output, stream = console(tty=True)
    presenter = TerminalPresenter("compact", console=output)
    with pytest.raises(type(exception)):
        with presenter:
            presenter(event())
            raise exception
    assert presenter._live is None and presenter._closed
    assert marker in stream.getvalue()


@pytest.mark.parametrize("mode", ["plain", "detailed", "compact"])
@pytest.mark.parametrize("failure_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_shutdown_diagnostic_failure_does_not_replace_primary_exception(mode, failure_type, monkeypatch):
    output, _ = console(tty=True)
    presenter = TerminalPresenter(mode, console=output)
    primary = failure_type(37) if failure_type is SystemExit else failure_type("native failure")
    calls = []

    def failed_diagnostic(message):
        calls.append(message)
        raise RuntimeError("shutdown formatter failed")

    with pytest.raises(failure_type) as raised:
        with presenter:
            presenter(event())
            monkeypatch.setattr(presenter, "_write_plain", failed_diagnostic)
            raise primary
    assert raised.value is primary
    assert presenter._closed and presenter._live is None
    if failure_type is SystemExit:
        # A deliberate exit keeps its status and adds no line after its report.
        assert raised.value.code == 37 and calls == []
    else:
        assert len(calls) == 1
        assert calls[0].startswith("STOPPED" if failure_type is KeyboardInterrupt else "FAILED")


@pytest.mark.parametrize("mode", ["plain", "detailed", "compact"])
def test_unprintable_primary_exception_still_propagates_after_display_closes(mode):
    class UnprintableError(RuntimeError):
        def __str__(self):
            raise LookupError("exception formatting failed")

    output, _ = console(tty=True)
    presenter = TerminalPresenter(mode, console=output)
    primary = UnprintableError("native failure")
    with pytest.raises(UnprintableError) as raised:
        with presenter:
            presenter(event())
            raise primary
    assert raised.value is primary
    assert presenter._closed and presenter._live is None


def test_renderer_failure_falls_back_without_changing_result_or_hiding_diagnostics(monkeypatch):
    output, stream = console(tty=True)
    monkeypatch.setattr(output, "print", lambda *a, **k: (_ for _ in ()).throw(OSError("bad terminal")))
    with TerminalPresenter("detailed", console=output) as presenter:
        presenter(event(state="failed", message="Measurement report missing"))
        assert presenter.mode == "plain"
    assert "using plain output" in stream.getvalue()
    assert "Measurement report missing" in stream.getvalue()


@pytest.mark.parametrize("mode", ["compact", "detailed", "plain"])
def test_message_markup_terminal_controls_and_credentials_are_not_executed_or_leaked(mode):
    output, stream = console(tty=True)
    with TerminalPresenter(mode, console=output, secrets=("secret-opaque-value",)) as presenter:
        presenter(event(state="failed", message=(
            "[red]literal[/red] \x1b[31mfailed\x1b[0m Authorization: Bearer hidden-bearer "
            "api_key=hidden-key https://user:hidden-pass@example.com/v1?token=hidden-query "
            "secret-opaque-value")))
    text = stream.getvalue() if mode == "plain" else output.export_text()
    assert "[red]literal[/red]" in text
    assert all(secret not in text for secret in ("hidden-bearer", "hidden-key", "hidden-pass",
                                                "hidden-query", "secret-opaque-value"))
    assert "[redacted]" in text
    if mode == "plain":
        assert "\x1b" not in stream.getvalue()


def test_ascii_only_output_stream_is_supported():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii")
    with TerminalPresenter("plain", stream=stream) as presenter:
        presenter(event(message="native path /tmp/caf\u00e9.py \u2713"))
    assert b"caf?.py ?" in raw.getvalue()


def test_ascii_only_compact_progress_does_not_require_block_characters():
    raw = io.BytesIO()
    stream = io.TextIOWrapper(raw, encoding="ascii")
    output = Console(file=stream, force_terminal=True, color_system=None,
                     no_color=True, width=80, highlight=False, markup=False)
    with TerminalPresenter("compact", console=output) as presenter:
        presenter(event(completed=2, total=8, work_unit="mutants"))
        assert presenter.mode == "compact"
    assert b"25%  2 of 8 mutants" in raw.getvalue() and b"##-" in raw.getvalue()


def test_url_query_redaction_handles_percent_encoded_credentials():
    assert "private" not in safe_text("https://u:private@example.com/?api_key=private%2Bvalue")


@pytest.mark.parametrize("control", ["\x1b[31m", "\x1b]0;title\x07", "\x9b"])
def test_known_secrets_remain_redacted_after_control_normalization(control):
    secret = "opaque-credential"
    assert safe_text("opaque-" + control + "credential", (secret,)) == "[redacted]"


@pytest.mark.parametrize("mode", ["compact", "detailed", "plain"])
def test_normalized_known_secrets_do_not_reach_any_rendered_field(mode):
    output, stream = console(tty=True)
    secret = "opaque-credential"
    concealed = "opaque-\x1b[31mcredential\x1b[0m"
    supplied = event(run_id=concealed, target=concealed, phase=concealed,
                     operation=concealed, work_unit=concealed, completed=1, total=3,
                     message=concealed, metrics=(Metric(concealed, 1, concealed),))
    before = asdict(supplied)
    with TerminalPresenter(mode, console=output, secrets=(secret,)) as presenter:
        presenter(supplied)
        assert presenter.view.latest()[supplied.identity] is supplied
    text = stream.getvalue() if mode == "plain" else output.export_text()
    assert secret not in text and "[redacted]" in text
    assert asdict(supplied) == before


def test_sanitizer_removes_c0_delete_and_c1_controls_without_removing_unicode_text():
    controls = "".join(chr(code) for code in (*range(32), *range(127, 160)))
    original = "before" + controls + " after caf\u00e9.py"
    cleaned = safe_text(original)
    assert not any(ord(char) < 32 or 127 <= ord(char) < 160 for char in cleaned)
    assert "before" in cleaned and "after caf\u00e9.py" in cleaned
    assert controls in original


@pytest.mark.parametrize("mode", ["compact", "detailed", "plain"])
def test_c1_controls_in_every_display_field_do_not_reach_the_terminal(mode):
    output, stream = console(tty=True)
    supplied = event(run_id="run\x9b2J", target="caf\u00e9.py\x9d", phase="mutation\x90",
                     operation="measure\x9c", work_unit="mutants\x9e", completed=1, total=3,
                     message="retained\x9f", metrics=(Metric("Mutation\x98", 1, "%\x91"),))
    with TerminalPresenter(mode, console=output) as presenter:
        presenter(supplied)
        assert presenter.view.latest()[supplied.identity] is supplied
    text = stream.getvalue() if mode == "plain" else output.export_text()
    assert not any(127 <= ord(char) < 160 for char in text)
    assert "caf\u00e9.py" in text and "retained" in text
    assert supplied.target == "caf\u00e9.py\x9d"


@pytest.mark.parametrize("basis", ["failed", "incomplete", "unavailable", "unsupported"])
def test_diagnostic_or_stale_values_do_not_become_headline_scores(basis):
    output, stream = console(width=200)
    with TerminalPresenter(console=output) as presenter:
        presenter(event(metrics=(Metric("Mutation", 83, "%", basis),)))
    assert f"Mutation: {basis}" in stream.getvalue()
    assert "83%" not in stream.getvalue()


def test_dumb_terminal_does_not_get_a_live_display(monkeypatch):
    monkeypatch.setenv("TERM", "dumb")
    output, stream = console(tty=True)
    with TerminalPresenter("compact", console=output) as presenter:
        assert presenter.mode == "plain"
        presenter(event())
    assert "\x1b" not in stream.getvalue()


def test_live_render_failure_restores_terminal_and_reports_operation(monkeypatch):
    output, stream = console(tty=True)
    with TerminalPresenter("compact", console=output) as presenter:
        presenter(event())
        live = presenter._live
        monkeypatch.setattr(live, "update", lambda *a, **k: (_ for _ in ()).throw(
            OSError("failed refresh")))
        presenter(event(state="failed", message="Native tool failed"))
        assert presenter.mode == "plain" and presenter._live is None
    assert "Native tool failed" in stream.getvalue()
    assert "using plain output" in stream.getvalue()


def test_completed_display_history_is_bounded_but_active_work_is_retained():
    output, _ = console()
    with TerminalPresenter("plain", console=output) as presenter:
        presenter(event(operation="active"))
        for i in range(100):
            presenter(event(operation=f"finished-{i}", state="succeeded"))
        assert len(presenter.view.latest()) == 65
        assert any(item.operation == "active" for item in presenter.view.latest().values())
        assert len(presenter.view._printed) == 65


@pytest.mark.parametrize("environment,explicit,colored", [
    ({}, False, True), ({"NO_COLOR": "1"}, False, False), ({}, True, False),
])
def test_color_follows_no_color_convention_and_explicit_request(monkeypatch, environment, explicit, colored):
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("FORCE_COLOR", "1")  # an interactive terminal without a real TTY
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    stream = io.StringIO()
    with TerminalPresenter("detailed", stream=stream, no_color=explicit, width=40) as presenter:
        assert presenter.mode == "detailed"
        presenter(event(state="failed", message="Native report missing in /work/out put/run log.json"))
    text = stream.getvalue()
    # Detailed records are not split for the 40-column width either.
    assert len(text.splitlines()) == 1 and "/work/out put/run log.json" in text
    # The convention concerns color. Emphasis such as bold may remain.
    assert ("31m" in text) is colored


@pytest.mark.parametrize("width", [0, -1, 1.5, True])
def test_invalid_programmatic_width_is_rejected(width):
    with pytest.raises(ValueError, match="width"):
        TerminalPresenter(width=width)


def test_a_demonstration_shows_playback_time_and_synthetic_values():
    """Synthetic observations: the clock is the playback's and no value reads as measured."""
    output, stream = console(width=200)
    with TerminalPresenter(console=output) as presenter:
        presenter(event(elapsed_s=65, provenance="synthetic",
                        metrics=(Metric("Line coverage", 100, "%", goal=100),)))
        presenter(event(elapsed_s=66, provenance="observed",
                        metrics=(Metric("Line coverage", 90, "%"),)))
    lines = stream.getvalue().splitlines()
    assert lines[0].startswith("00:01:05 RUNNING [demo] [playback time] | ")
    assert "Line coverage: 100% (goal 100%) [demo]" in lines[0] and "[measured]" not in lines[0]
    assert lines[1].startswith("00:01:06 RUNNING | ") and "Line coverage: 90% [measured]" in lines[1]


def test_the_compact_view_of_a_demonstration_labels_its_clock_as_playback():
    output, _ = console(width=120, tty=True)
    with TerminalPresenter("compact", console=output) as presenter:
        presenter(event(phase="pipeline", operation="run", elapsed_s=3, provenance="synthetic"))
        presenter(event(operation="request/1", elapsed_s=7, provenance="synthetic",
                        metrics=(Metric("Mutation score", 80, "%"),)))
        log = " ".join(output.export_text().split())
        frame = Console(width=120, record=True, file=io.StringIO())
        frame.print(presenter.frame())
        view = " ".join(frame.export_text().split())
    assert "Playback" in view and "playback 00:00:07" in view and "Elapsed" not in view
    assert "Mutation score: 80% [demo]" in view
    # The run's start is logged above the view with the playback clock too.
    assert "00:00:03 RUNNING [demo] [playback time] | Run" in log


def test_a_percentage_shows_at_most_one_decimal():
    """A measured percentage shows at most one decimal, cut, never rounded
    up: 91.8%, not 91.8919% or 91.9%. A whole one stays whole, and other
    units keep their value."""
    from testuary.presentation.view import metric_text, percent_text

    assert metric_text(Metric("Mutation score", 91.891891, "%", "measured")) == "Mutation score: 91.8% [measured]"
    assert metric_text(Metric("Line coverage", 100.0, "%", "measured")) == "Line coverage: 100% [measured]"
    assert metric_text(Metric("Measured value", 7.25, " ms", "measured")) == "Measured value: 7.25 ms [measured]"
    assert [percent_text(value) for value in (87.719298, 66.666, 0.04, 99.96)] == ["87.7", "66.6", "0", "99.9"]


def test_a_refused_configuration_reads_refused_not_failed():
    """A refused configuration (exit status 2) is no failed run: its line and
    the compact view's Overall row say REFUSED, never FAILED."""
    output, _ = console(width=120, tty=True)
    with TerminalPresenter("compact", console=output) as presenter:
        presenter(event(target="all targets", phase="pipeline", operation="run",
                        level="run", elapsed_s=1, message="Checking run inputs"))
        presenter(event(target="all targets", phase="pipeline", operation="run",
                        level="run", state="refused", elapsed_s=2,
                        message="Configuration refused"))
        frame = Console(width=120, record=True, file=io.StringIO())
        frame.print(presenter.frame())
        shown = frame.export_text()
    log = output.export_text()
    assert "Overall  REFUSED" in shown
    assert "REFUSED" in log and "FAILED" not in shown + log


def test_a_cancelled_run_reads_stopped_as_the_web_interface_says():
    output, stream = console()
    with TerminalPresenter("plain", console=output) as presenter:
        presenter(event(state="cancelled", message="Run stopped by you"))
        operation = presenter.view.snapshot(1).operations[-1]
    assert stream.getvalue().split(" | ")[0].endswith("STOPPED")
    assert operation.state == "STOPPED" and "CANCELLED" not in stream.getvalue()
