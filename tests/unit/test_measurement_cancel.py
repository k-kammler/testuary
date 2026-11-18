"""A Ctrl+C while an agent's graph runs cancels the measurement a tool runs
on another thread, then interrupts as usual."""
from __future__ import annotations

import concurrent.futures
import signal
import threading
import time

import pytest

from testuary.languages import mutation


class _Native:
    def __init__(self, cancellable=True):
        self.cancels = 0
        if not cancellable:
            self.cancel = None

    def cancel(self):
        self.cancels += 1


@pytest.fixture(autouse=True)
def _no_cancelled_session_left():
    yield
    mutation._CANCELLED.clear()


def _session(native):
    return mutation.MutationSession(runner=None, session=native, workers=1, use_http=False)


def test_a_ctrl_c_during_the_graph_cancels_the_measurement_in_flight_then_interrupts():
    native = _Native()
    session = _session(native)
    before = signal.getsignal(signal.SIGINT)
    with pytest.raises(KeyboardInterrupt):
        with mutation.cancel_on_interrupt():
            with session.worker():
                assert session in mutation._IN_FLIGHT
                signal.raise_signal(signal.SIGINT)
    # The handler and the wakeup watcher both cancel, which is idempotent.
    assert native.cancels >= 1
    assert session not in mutation._IN_FLIGHT
    assert signal.getsignal(signal.SIGINT) is before


def test_a_session_without_an_operation_is_not_cancelled():
    native = _Native()
    _session(native)
    with pytest.raises(KeyboardInterrupt):
        with mutation.cancel_on_interrupt():
            signal.raise_signal(signal.SIGINT)
    assert native.cancels == 0


def test_a_second_ctrl_c_while_the_cancelled_measurement_runs_is_held(capsys):
    """Raised, the second Ctrl+C would cut the wait for the tool's thread,
    and the interpreter would end that thread midway through its cleanup."""
    native = _Native()
    session = _session(native)
    interrupts = 0
    with mutation.cancel_on_interrupt():
        with session.worker():
            for _ in range(3):
                try:
                    signal.raise_signal(signal.SIGINT)
                except KeyboardInterrupt:
                    interrupts += 1
        assert interrupts == 1 and native.cancels >= 3
        # The held presses are answered once.
        assert capsys.readouterr().err.count("Stopping: the work in flight ends") == 1
        # With the operation ended, a Ctrl+C interrupts again.
        with pytest.raises(KeyboardInterrupt):
            signal.raise_signal(signal.SIGINT)


def test_the_hook_changes_nothing_off_the_main_thread_or_when_ignored():
    seen = []

    def elsewhere():
        before = signal.getsignal(signal.SIGINT)
        with mutation.cancel_on_interrupt():
            seen.append(signal.getsignal(signal.SIGINT) is before)

    thread = threading.Thread(target=elsewhere)
    thread.start()
    thread.join()
    assert seen == [True]
    previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
    try:
        with mutation.cancel_on_interrupt():
            assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
    finally:
        signal.signal(signal.SIGINT, previous)


def test_a_custom_handler_in_place_decides_how_to_interrupt():
    native = _Native()
    session = _session(native)
    calls = []
    previous = signal.signal(signal.SIGINT, lambda signum, frame: calls.append(signum))
    try:
        with mutation.cancel_on_interrupt():
            with session.worker():
                signal.raise_signal(signal.SIGINT)
    finally:
        signal.signal(signal.SIGINT, previous)
    assert native.cancels >= 1 and calls == [signal.SIGINT]


def test_cancel_forwards_to_the_engine_and_tolerates_one_without_it():
    native = _Native()
    _session(native).cancel()
    assert native.cancels == 1
    _session(_Native(cancellable=False)).cancel()


def test_the_session_in_flight_is_cancelled_while_its_tool_still_runs():
    """A stop raised while a tool runs on another thread reaches its session
    before the tool ends: by the handler where the main thread can run it,
    and by the wakeup watcher where the main thread is blocked in a lock
    (Windows, waiting for the tool's thread). The next test covers a main
    thread that is already blocked."""
    import _thread

    native = _Native()
    session = _session(native)
    seen = []

    def tool():
        with session.worker():
            _thread.interrupt_main()
            for _ in range(100):
                if native.cancels:
                    break
                threading.Event().wait(0.05)
            seen.append(native.cancels)

    thread = threading.Thread(target=tool)
    with pytest.raises(KeyboardInterrupt):
        with mutation.cancel_on_interrupt():
            thread.start()
            thread.join()
    thread.join()
    assert seen and seen[0] >= 1


@pytest.mark.parametrize("wait", ["join", "future"])
def test_the_session_is_cancelled_while_the_main_thread_is_already_blocked(wait):
    """The main thread blocks in its wait before the stop, as ToolNode
    waits on a future. Only the wakeup watcher reaches the session here."""
    import _thread

    native = _Native()
    session = _session(native)
    seen = []

    def tool():
        with session.worker():
            # Let the main thread block in its wait first.
            time.sleep(0.5)
            _thread.interrupt_main()
            for _ in range(60):
                if native.cancels:
                    break
                time.sleep(0.05)
            seen.append(native.cancels)

    thread = threading.Thread(target=tool)
    with pytest.raises(KeyboardInterrupt):
        with mutation.cancel_on_interrupt():
            if wait == "join":
                thread.start()
                thread.join()
            else:
                with concurrent.futures.ThreadPoolExecutor(1) as executor:
                    executor.submit(tool).result()
    if wait == "join":
        thread.join()
    assert seen and seen[0] >= 1, seen


class _Closing(_Native):
    def __init__(self, during_close=None):
        super().__init__()
        self.closes = 0
        self.during_close = during_close

    def close(self):
        if self.during_close is not None:
            self.during_close()
        self.closes += 1


def test_a_cancelled_sessions_close_waits_for_the_operations_cleanup():
    native = _Closing()
    session = _session(native)
    inside, cleaned = threading.Event(), []

    def tool():
        with session.worker():
            inside.set()
            time.sleep(0.5)
            cleaned.append(True)

    thread = threading.Thread(target=tool)
    thread.start()
    inside.wait()
    session.cancel()
    session.close()
    thread.join()
    assert cleaned == [True] and native.closes == 1 and session._closed
    assert session not in mutation._CANCELLED


def test_an_uncancelled_sessions_close_still_refuses_an_active_operation():
    session = _session(_Closing())
    inside, leave = threading.Event(), threading.Event()

    def tool():
        with session.worker():
            inside.set()
            leave.wait()

    thread = threading.Thread(target=tool)
    thread.start()
    inside.wait()
    try:
        with pytest.raises(RuntimeError, match="operation is active"):
            session.close()
    finally:
        leave.set()
        thread.join()


def test_a_ctrl_c_during_a_close_is_held_until_the_close_ends():
    native = _Closing(during_close=lambda: signal.raise_signal(signal.SIGINT))
    session = _session(native)
    with pytest.raises(KeyboardInterrupt):
        session.close()
    assert native.closes == 1 and session._closed
    # During an interrupt already under way, the held one is not raised again.
    native = _Closing(during_close=lambda: signal.raise_signal(signal.SIGINT))
    session = _session(native)
    session.close(primary=KeyboardInterrupt())
    assert native.closes == 1 and session._closed


def test_the_command_line_closes_a_cancelled_session_the_interrupt_skipped():
    from testuary import __main__ as cli

    native = _Closing()
    session = _session(native)
    session.cancel()
    assert cli._retry_interrupted_cleanup(KeyboardInterrupt()) is None
    assert native.closes == 1 and session._closed and not mutation._CANCELLED
    # From the exit's cleanup on, a further Ctrl+C neither cuts the cleanup
    # short nor kills the exit (the conftest restores the handler).
    assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN


def test_a_ctrl_c_cancels_every_tool_call_in_progress():
    """Every agent tool call runs under a cancel scope: a Ctrl+C stops what
    the call started, coverage and suite runs included, and the call ends
    with no result."""
    from langadapter import OperationCancelled

    seen = []
    with pytest.raises(KeyboardInterrupt):
        with mutation.cancel_on_interrupt():
            try:
                with mutation.tool_scope():
                    assert len(mutation._SCOPES) == 1
                    try:
                        signal.raise_signal(signal.SIGINT)
                    except KeyboardInterrupt:
                        seen.append("interrupt")
            except OperationCancelled:
                seen.append("cancelled")
                raise
    assert seen == ["interrupt", "cancelled"] and not mutation._SCOPES


def test_every_bound_tool_runs_in_a_scope():
    from testuary.agent import _base

    seen = []

    class _Agent(_base.CodingAgent):
        def __init__(self):
            pass

        def tool(self) -> str:
            """A tool."""
            seen.append(len(mutation._SCOPES))
            return "done"

    bound = _Agent()._bind_tool(_Agent().tool)
    assert bound.invoke({}) == "done" and seen == [1] and not mutation._SCOPES
