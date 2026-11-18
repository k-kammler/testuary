"""Explicit diagnostic writes preserve their stream, bytes and exceptions."""
from __future__ import annotations

import io
import sys
import threading
from dataclasses import asdict, replace

import pytest
from rich.console import Console
from rich.live import Live

from testuary.presentation import DisplayEvent, TerminalPresenter
from testuary.presentation import terminal


@pytest.fixture(autouse=True)
def terminal_environment(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")


def event(**overrides):
    values = dict(run_id="run-1", target="src/calc.py", phase="coverage",
                  operation="measure", state="running", elapsed_s=1)
    return DisplayEvent(**{**values, **overrides})


def console():
    stream = io.StringIO()
    return Console(file=stream, force_terminal=True, color_system=None,
                   no_color=True, width=100, height=35, markup=False,
                   highlight=False), stream


@pytest.mark.parametrize("mode", ["compact", "detailed", "plain"])
def test_diagnostic_preserves_original_bytes_streams_and_observations(mode, monkeypatch):
    output, display = console()
    stdout = io.StringIO()
    monkeypatch.setattr(sys, "stdout", stdout)
    original_streams = sys.stdout, sys.stderr
    observation = event()
    before = asdict(observation)
    payload = "\x1b[36mNative progress\x1b[0m\nsecond line\n"
    calls = []
    with TerminalPresenter(mode, console=output) as presenter:
        presenter(observation)
        original_live = presenter._live

        def write():
            calls.append(1)
            assert (sys.stdout, sys.stderr) == original_streams
            assert presenter._live is None
            print(payload, end="", file=sys.stdout, flush=True)

        assert presenter.write_diagnostic(write) is None
        assert calls == [1] and stdout.getvalue() == payload
        assert presenter.view.latest()[observation.identity] is observation
        if mode == "compact":
            assert presenter._live is not None and presenter._live is not original_live
        else:
            assert presenter._live is None
    assert payload not in display.getvalue()
    assert asdict(observation) == before
    assert (sys.stdout, sys.stderr) == original_streams


def test_compact_suspends_transiently_then_creates_a_fresh_live_after_write(monkeypatch):
    trace = []

    class TracedLive(Live):
        def start(self, refresh=False):
            trace.append(("start", self))
            super().start(refresh=refresh)

        def stop(self):
            trace.append(("stop", self, self.transient))
            super().stop()

    monkeypatch.setattr(terminal, "Live", TracedLive)
    output, _ = console()
    with TerminalPresenter("compact", console=output) as presenter:
        presenter(event())
        previous = presenter._live
        trace.clear()
        presenter.write_diagnostic(lambda: trace.append(("write", presenter._live)))
        current = presenter._live
        assert trace == [("stop", previous, True), ("write", None), ("start", current)]
        assert current is not previous and not current.transient
        assert not previous.transient


def test_partial_diagnostic_ends_on_its_own_line_before_the_new_frame():
    output, stream = console()
    payload = "direct partial diagnostic"
    with TerminalPresenter("compact", console=output) as presenter:
        presenter(event())
        presenter.write_diagnostic(lambda: print(payload, end="", file=stream, flush=True))
        before, after = stream.getvalue().split(payload)
        assert before and after.startswith("\n")
        assert "Testuary" in after


@pytest.mark.parametrize("failure", [OSError("write failed"), RuntimeError("write failed"),
                                    KeyboardInterrupt(), SystemExit(37)])
def test_actual_write_exception_propagates_once_and_live_is_restored(failure):
    output, _ = console()
    calls = []
    with TerminalPresenter("compact", console=output) as presenter:
        presenter(event())

        def write():
            calls.append(1)
            raise failure

        with pytest.raises(type(failure)) as raised:
            presenter.write_diagnostic(write)
        assert raised.value is failure and calls == [1]
        assert presenter._live is not None and presenter.mode == "compact"
    assert presenter._live is None


@pytest.mark.parametrize("point", ["suspend", "restart"])
def test_display_failure_does_not_duplicate_or_prevent_the_write(point, monkeypatch):
    output, display = console()
    calls = []
    with TerminalPresenter("compact", console=output) as presenter:
        observation = event()
        presenter(observation)
        live = presenter._live
        if point == "suspend":
            monkeypatch.setattr(live, "refresh", lambda: (_ for _ in ()).throw(
                OSError("display refresh failed")))
        else:
            class FailedStart(Live):
                def start(self, refresh=False):
                    super().start(refresh=refresh)
                    raise OSError("display start failed")

            monkeypatch.setattr(terminal, "Live", FailedStart)
        presenter.write_diagnostic(lambda: calls.append(1))
        assert calls == [1]
        assert presenter.mode == "plain" and presenter._live is None
        assert presenter.view.latest()[observation.identity] is observation
        position = display.tell()
        output.print("ordinary console output")
        assert display.getvalue()[position:] == "ordinary console output\n"
    assert "using plain output" in display.getvalue()


def test_restart_failure_never_replaces_an_actual_write_failure(monkeypatch):
    output, _ = console()
    failure = RuntimeError("original write failure")
    calls = []
    with TerminalPresenter("compact", console=output) as presenter:
        presenter(event())

        class FailedStart(Live):
            def start(self, refresh=False):
                raise OSError("display start failed")

        monkeypatch.setattr(terminal, "Live", FailedStart)

        def write():
            calls.append(1)
            raise failure

        with pytest.raises(RuntimeError) as raised:
            presenter.write_diagnostic(write)
        assert raised.value is failure and calls == [1]
        assert presenter.mode == "plain" and presenter._live is None


@pytest.mark.parametrize("state", ["before_event", "closed"])
def test_diagnostic_without_an_active_display_writes_once_without_starting_live(state):
    output, display = console()
    presenter = TerminalPresenter("compact", console=output)
    if state == "closed":
        presenter.close()
    calls = []
    presenter.write_diagnostic(lambda: calls.append(1))
    assert calls == [1] and presenter._live is None
    assert display.getvalue() == ""


def test_nested_writes_and_observations_resume_only_after_the_outer_write():
    output, _ = console()
    calls = []
    with TerminalPresenter("compact", console=output) as presenter:
        supplied = event()
        presenter(supplied)

        def outer_write():
            calls.append("outer start")
            presenter.write_diagnostic(lambda: calls.append("inner"))
            presenter(replace(supplied, state="succeeded", elapsed_s=2))
            assert presenter._live is None
            calls.append("outer end")

        presenter.write_diagnostic(outer_write)
        assert calls == ["outer start", "inner", "outer end"]
        assert presenter._live is not None
        assert presenter.view.latest()[supplied.identity].state == "succeeded"


def test_parallel_writes_and_observations_cannot_refresh_inside_a_diagnostic():
    output, _ = console()
    entered = threading.Event()
    release = threading.Event()
    attempted = [threading.Event(), threading.Event()]
    calls, failures = [], []
    with TerminalPresenter("compact", console=output) as presenter:
        supplied = event()
        presenter(supplied)

        def first_write():
            calls.append("first start")
            entered.set()
            assert release.wait(5)
            assert presenter._live is None
            calls.append("first end")

        def run(action):
            try:
                action()
            except BaseException as exc:
                failures.append(exc)

        def observe():
            attempted[0].set()
            presenter(replace(supplied, state="succeeded", elapsed_s=2))
            calls.append("observed")

        def write_second():
            attempted[1].set()
            presenter.write_diagnostic(lambda: calls.append("second"))

        threads = [threading.Thread(target=run, args=(action,)) for action in (
            lambda: presenter.write_diagnostic(first_write), observe, write_second)]
        threads[0].start()
        try:
            assert entered.wait(5)
            for thread in threads[1:]:
                thread.start()
            assert all(item.wait(5) for item in attempted)
        finally:
            release.set()
            for thread in threads:
                if thread.ident is not None:
                    thread.join(5)
        assert all(not thread.is_alive() for thread in threads)
        assert not failures
        assert calls[:2] == ["first start", "first end"]
        assert sorted(calls[2:]) == ["observed", "second"]
        assert presenter.view.latest()[supplied.identity].state == "succeeded"
