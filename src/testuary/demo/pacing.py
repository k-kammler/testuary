"""Paced playback of a demonstration: presentation only.

Instant playback finishes in seconds, which shows that the scenario plays,
not what a run looks like. Paced playback holds the same run before each
observation it displays. Nothing else changes: the answers, their order, the
results, the request counts and the tokens are those of instant playback,
and every record is written as it would be, but for the run log's lines of
each model request's elapsed time.

The playback's length is divided evenly over the observations: each waits
the same share, one after another, whatever the recorded run took. Every
phase, Finishing included, so moves at a steady rate, and the
run's own work only adds to the length, which is therefore a minimum. A
playback is the same run every time, so its number of observations is known
for each installed data set (``OBSERVATIONS``).

A pause waits in short steps on an event that Ctrl+C sets, so an interrupt
ends every pause at once and the run stops as a cancelled run does.
"""
from __future__ import annotations

import math
import os
import queue
import sys
import threading
import time
from typing import Any, Callable, Mapping, Optional, TextIO

from testuary.phase_state import PHASES

#: The observations a playback of each installed data set displays, which
#: tests/integration/test_demo_pacing.py counts in each playback.
OBSERVATIONS = {"python": 219, "java": 223, "toy": 46, "tutorial": 112}
_STEP = 0.05


def parse_speed(text: str) -> Optional[float]:
    """``auto``, ``instant`` or a positive factor of the configured length's
    speed (2 plays in half the time). Returns the factor, 0 for instant and
    None for auto. Raises ValueError otherwise."""
    value = str(text).strip().lower()
    if value == "auto":
        return None
    if value == "instant":
        return 0.0
    try:
        factor = float(value)
    except ValueError:
        raise ValueError(f"unknown demonstration speed: {text!r}") from None
    if not math.isfinite(factor) or factor <= 0:
        raise ValueError(f"a demonstration speed factor must be positive: {text!r}")
    return factor


def interactive(stream: TextIO, environ: Mapping[str, str]) -> bool:
    """Whether output reaches a person at a terminal: a terminal that is not
    dumb, outside a test run and outside continuous integration."""
    try:
        terminal = bool(stream.isatty())
    except (AttributeError, ValueError, OSError):
        terminal = False
    return (terminal and environ.get("TERM", "") != "dumb"
            and "PYTEST_CURRENT_TEST" not in environ and not environ.get("CI"))


class _Keys:
    """Enter presses read by a daemon thread, without echo, so a step wait
    can end on a key or on cancellation alike."""

    def __init__(self) -> None:
        self.pressed: "queue.Queue[str]" = queue.Queue()
        self._restore: Optional[Callable[[], None]] = None
        if os.name == "nt":
            import msvcrt

            def read() -> str:
                return msvcrt.getwch()
        else:
            def read() -> str:
                return sys.stdin.read(1)
        self.resume()

        def loop() -> None:
            while True:
                try:
                    key = read()
                except (OSError, ValueError):
                    return
                if not key:
                    return
                if key in ("\r", "\n"):
                    self.pressed.put(key)

        threading.Thread(target=loop, name="demo-step-keys", daemon=True).start()

    def resume(self) -> None:
        """Take keys one at a time without echo until ``close``: on POSIX
        the terminal's cbreak mode, in which Ctrl+C still interrupts."""
        if os.name == "nt" or self._restore is not None:
            return
        import termios
        import tty

        descriptor = sys.stdin.fileno()
        saved = termios.tcgetattr(descriptor)
        tty.setcbreak(descriptor)
        self._restore = lambda: termios.tcsetattr(descriptor, termios.TCSADRAIN, saved)

    def close(self) -> None:
        """Restore the terminal's settings. The reader thread stays for the
        next ``keys``."""
        if self._restore is not None:
            self._restore()
            self._restore = None


class Pacer:
    """Holds the playback before each of its ``observations`` for the same
    share of ``seconds`` at ``factor`` times speed 1, or, stepping, until
    Enter is pressed at each model answer, phase start, phase result and the
    run's end. ``on_event`` receives the display observations, ``on_answer``
    the scenario's answers. Both only wait. ``cancel`` ends every wait, now
    and later."""

    def __init__(self, observations: int, seconds: float, factor: float = 1.0, *,
                 step: bool = False, clock: Callable[[], float] = time.monotonic,
                 keys: Any = None) -> None:
        if not step and factor <= 0:
            raise ValueError("a paced playback needs a positive factor")
        self.step = step
        #: Each observation's wait in seconds: the length at this speed, split
        #: evenly.
        self.share = (seconds / factor / observations
                      if observations > 0 and not step else 0.0)
        self._clock = clock
        self._keys = keys
        self._cancelled = threading.Event()
        self._lock = threading.Lock()
        # Concurrent observations wait one after another, each for its share.
        self._share_lock = threading.Lock()
        self._step_lock = threading.Lock()
        self._phase_start: set = set()
        self._phase_done: set = set()
        self._ended = False

    def cancel(self) -> None:
        self._cancelled.set()

    def close(self) -> None:
        self.cancel()
        if self._keys is not None and hasattr(self._keys, "close"):
            self._keys.close()

    def _wait_until(self, deadline: float) -> None:
        while not self._cancelled.is_set():
            remaining = deadline - self._clock()
            if remaining <= 0:
                return
            self._cancelled.wait(min(remaining, _STEP))

    def _wait_key(self) -> None:
        # One step at a time: concurrent answers wait for their own key.
        with self._step_lock:
            wait_for_key(self._keys, self._cancelled)

    def on_event(self, event: Any) -> None:
        """Called before an observation is displayed: it waits its share,
        or, stepping, Enter at each phase's start and result and at the
        run's end, once each."""
        if self._cancelled.is_set():
            return
        if not self.step:
            with self._share_lock:
                self._wait_until(self._clock() + self.share)
            return
        if _is_run_end(event):
            with self._lock:
                if self._ended or not self._phase_done:
                    return
                self._ended = True
            self._wait_key()
            return
        if not _is_phase(event) or event.phase not in PHASES:
            return
        seen = (self._phase_start if event.state == "running" else
                self._phase_done if event.state in ("succeeded", "failed") else None)
        if seen is None:
            return
        with self._lock:
            if event.phase in seen:
                return
            seen.add(event.phase)
        self._wait_key()

    def on_answer(self, entry: Any) -> None:
        """Stepping holds a model answer until Enter. A paced playback does
        not wait at answers."""
        if self.step and not self._cancelled.is_set() and str(entry.kind).startswith("model."):
            self._wait_key()


def _is_run_end(event: Any) -> bool:
    from testuary.presentation.view import level_of

    return level_of(event) == "run" and event.state in ("succeeded", "failed", "cancelled")


def _is_phase(event: Any) -> bool:
    """A phase's observation, or Finishing's, which stepping treats
    as a phase."""
    from testuary.presentation.view import is_finishing, level_of

    return ((level_of(event) == "phase" or is_finishing(event))
            and not str(event.operation).startswith("restore/"))


#: The process's one key reader, made at the first ``keys``.
_READER: Optional[_Keys] = None
_READER_LOCK = threading.Lock()


def keys() -> _Keys:
    """Enter presses read from the terminal, for stepping. One reader serves
    the process, such as the setup demonstration's typist and then its
    playback, since a second reading thread would take presses nobody
    waits for."""
    global _READER
    with _READER_LOCK:
        if _READER is None:
            _READER = _Keys()
        else:
            _READER.resume()
        return _READER


def wait_for_key(keys: _Keys, cancelled: Optional[threading.Event] = None) -> None:
    """Wait for a key from ``keys`` until ``cancelled`` is set. A key pressed
    while the run worked before the wait would let a step pass unseen: only
    a key pressed during the wait counts. Short waits keep Ctrl+C working."""
    while True:
        try:
            keys.pressed.get_nowait()
        except queue.Empty:
            break
    while cancelled is None or not cancelled.is_set():
        try:
            keys.pressed.get(timeout=_STEP)
            return
        except queue.Empty:
            continue
