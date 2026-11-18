"""Paced demonstration playback: the speed choices, the even waits,
stepping and cancellation. No pipeline runs here."""
from __future__ import annotations

import io
import queue
import threading
import time
from types import SimpleNamespace

import pytest

from testuary._progress import RunProgress, paced_display
from testuary.demo import pacing
from testuary.presentation import DisplayEvent


@pytest.mark.parametrize("text, factor", [
    ("auto", None), ("instant", 0.0), ("1", 1.0), ("2", 2.0), ("0.25", 0.25), (" Instant ", 0.0)])
def test_speeds(text, factor):
    assert pacing.parse_speed(text) == factor


@pytest.mark.parametrize("text", ["fast", "normal", "0", "-1", "nan", "inf", ""])
def test_unusable_speeds_are_refused(text):
    with pytest.raises(ValueError):
        pacing.parse_speed(text)


class _Stream(io.StringIO):
    def __init__(self, terminal):
        super().__init__()
        self.terminal = terminal

    def isatty(self):
        return self.terminal


@pytest.mark.parametrize("terminal, environ, expected", [
    (True, {"TERM": "xterm"}, True),
    (False, {"TERM": "xterm"}, False),
    (True, {"TERM": "dumb"}, False),
    (True, {"TERM": "xterm", "CI": "true"}, False),
    (True, {"TERM": "xterm", "PYTEST_CURRENT_TEST": "x"}, False),
])
def test_only_a_person_at_a_terminal_gets_paced_playback_by_default(terminal, environ, expected):
    assert pacing.interactive(_Stream(terminal), environ) is expected


class _Clock:
    """Time that passes only while the pacer waits."""

    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now

    def wait(self, seconds):
        self.now += seconds
        return False


def _pacer(answers, seconds, factor=1.0, **kwargs):
    clock = _Clock()
    pacer = pacing.Pacer(answers, seconds, factor, clock=clock, **kwargs)
    pacer._cancelled.wait = clock.wait
    return pacer, clock


def _phase(name, state):
    return DisplayEvent(run_id="r", target="a.py", phase=name, operation="phase",
                        state=state, elapsed_s=0)


def _answer(segment, kind="model.agent"):
    return SimpleNamespace(segment=segment, kind=kind)


def _observations():
    """Observations of every level: the run, phases, a step, an operation."""
    run = dict(run_id="r", target="pipeline", phase="pipeline", operation="run", elapsed_s=0)
    return [DisplayEvent(state="running", **run), _phase("starter", "running"),
            DisplayEvent(run_id="r", target="a.py", phase="starter", operation="step/verify",
                         state="running", elapsed_s=0, level="step"),
            DisplayEvent(run_id="r", target="a.py", phase="starter", operation="measure",
                         state="running", elapsed_s=0, completed=1, total=3),
            _phase("starter", "succeeded"), _phase("finishing", "running"),
            DisplayEvent(state="succeeded", **run)]


@pytest.mark.parametrize("seconds, factor", [(21.0, 1.0), (21.0, 2.0), (120.0, 0.5)])
def test_the_waits_split_the_length_evenly_over_the_observations(seconds, factor):
    events = _observations()
    pacer, clock = _pacer(len(events), seconds, factor)
    share = seconds / factor / len(events)
    assert pacer.share == pytest.approx(share)
    for event in events:
        before = clock.now
        pacer.on_event(event)
        # Every observation, at every level, waits the same share.
        assert clock.now - before == pytest.approx(share)
    assert clock.now == pytest.approx(seconds / factor)


def test_answers_do_not_wait_in_a_paced_playback():
    pacer, clock = _pacer(4, 40.0)
    for kind in ("model.seed", "backend.run_test_files", "cover.exact", "model.agent"):
        pacer.on_answer(_answer("starter", kind))
    assert clock.now == 0


def test_concurrent_observations_wait_one_after_another():
    pacer = pacing.Pacer(3, 0.6)
    threads = [threading.Thread(target=pacer.on_event, args=(_phase("starter", "running"),))
               for _ in range(3)]
    began = time.monotonic()
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(5)
    assert time.monotonic() - began >= 0.55


def test_cancelling_ends_a_pause_at_once_in_another_thread():
    pacer = pacing.Pacer(1, 3600.0)
    worker = threading.Thread(target=pacer.on_event, args=(_phase("coverage", "running"),))
    worker.start()
    time.sleep(0.2)
    assert worker.is_alive()
    began = time.monotonic()
    pacer.cancel()
    worker.join(2)
    assert not worker.is_alive() and time.monotonic() - began < 0.5
    # Later observations do not wait at all.
    began = time.monotonic()
    pacer.on_event(_phase("mutation", "running"))
    assert time.monotonic() - began < 0.1


def test_stepping_advances_on_each_key_and_ends_on_cancel():
    keys = SimpleNamespace(pressed=queue.Queue())
    pacer = pacing.Pacer(10, 50.0, step=True, keys=keys)
    done = threading.Event()

    def answer():
        pacer.on_answer(_answer("coverage"))
        done.set()

    threading.Thread(target=answer, daemon=True).start()
    assert not done.wait(0.3)
    keys.pressed.put("\r")
    assert done.wait(1)
    done.clear()
    threading.Thread(target=answer, daemon=True).start()
    assert not done.wait(0.2)
    pacer.cancel()
    assert done.wait(1)


def test_stepping_waits_at_model_answers_and_each_phase_bound_once():
    keys = SimpleNamespace(pressed=queue.Queue())
    pacer = pacing.Pacer(10, 50.0, step=True, keys=keys)
    # A measurement does not wait for a key.
    pacer.on_answer(_answer("starter", "backend.run_test_files"))
    for event in (_phase("starter", "running"), _phase("starter", "succeeded")):
        done = threading.Event()

        def observe(event=event):
            pacer.on_event(event)
            done.set()

        threading.Thread(target=observe, daemon=True).start()
        assert not done.wait(0.2)
        keys.pressed.put("\r")
        assert done.wait(1)
    # A repeated observation of the same bound does not wait again.
    pacer.on_event(_phase("starter", "running"))
    pacer.cancel()


def test_stepping_ignores_keys_pressed_while_the_run_worked():
    keys = SimpleNamespace(pressed=queue.Queue())
    pacer = pacing.Pacer(10, 50.0, step=True, keys=keys)
    # Pressed during a long stretch of work before the step: no step passes.
    for _ in range(3):
        keys.pressed.put("\r")
    done = threading.Event()

    def answer():
        pacer.on_answer(_answer("coverage"))
        done.set()

    threading.Thread(target=answer, daemon=True).start()
    assert not done.wait(0.3)
    keys.pressed.put("\r")
    assert done.wait(1)
    pacer.cancel()


def test_the_progress_reporter_paces_before_display_and_shows_the_playback_clock():
    shown, paced = [], []

    def pace(event):
        paced.append((event.operation, len(shown)))
        time.sleep(0.2)

    with paced_display(pace):
        progress = RunProgress(shown.append)
    progress.emit(target="a.py", phase="coverage", operation="phase", state="running")
    # Called before the display, and the shown event reads when it was shown.
    assert paced == [("phase", 0)] and len(shown) == 1
    assert shown[0].elapsed_s >= 0.15
    # Without pacing, nothing waits.
    plain = []
    RunProgress(plain.append).emit(target="a.py", phase="coverage", operation="phase",
                                   state="running")
    assert plain[0].elapsed_s < 0.15


def test_a_failing_pacer_never_stops_the_display():
    shown = []

    def pace(event):
        raise RuntimeError("pacing broke")

    with paced_display(pace):
        progress = RunProgress(shown.append)
    for _ in range(2):
        progress.emit(target="a.py", phase="coverage", operation="phase", state="running")
    assert len(shown) == 2


def test_stepping_treats_finishing_as_a_phase():
    from testuary.presentation.view import FINISHING

    finishing = DisplayEvent(run_id="r", target="a.py", phase="finishing", operation=FINISHING,
                             state="running", elapsed_s=0, level="step", label="Finishing")
    other = DisplayEvent(run_id="r", target="a.py", phase="finishing", operation="step/other",
                         state="running", elapsed_s=0, level="step")
    assert pacing._is_phase(finishing) and not pacing._is_phase(other)


def test_one_key_reader_serves_the_process(monkeypatch):
    """The setup demonstration's typist and its playback ask for the keys in
    one process. Both get the one reader, taken up again after its close,
    so no Enter press goes to a reader nobody waits on."""
    made = []

    class Reader:
        def __init__(self):
            self.resumed = 0
            made.append(self)

        def resume(self):
            self.resumed += 1

        def close(self):
            pass

    monkeypatch.setattr(pacing, "_Keys", Reader)
    monkeypatch.setattr(pacing, "_READER", None, raising=False)
    first = pacing.keys()
    first.close()
    assert pacing.keys() is first
    assert len(made) == 1 and first.resumed == 1
