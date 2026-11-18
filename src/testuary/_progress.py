"""Read-only progress events for display clients, the console preferences
and the run's plain-text log."""
from __future__ import annotations

import logging
import sys
import re
import threading
import time
import uuid
from asyncio import CancelledError as AsyncCancelledError
from collections.abc import Callable
from concurrent.futures import CancelledError as FutureCancelledError
from contextlib import contextmanager
from dataclasses import replace
from contextvars import ContextVar

from langchain_core.callbacks import BaseCallbackHandler
from testuary.presentation import DisplayEvent, Metric, Usage


_COLOR_ENABLED = True
_DIAGNOSTIC_WRITER = None
_DETAIL_ECHO = True
_ANSI = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
_DIAGNOSTIC_LINE = re.compile(
    r"\s*(?:\[[^\]\n]*\]\s*)*(?:\[CRITICAL\]|(?:ERROR|WARNING|WARN|FATAL|CRITICAL)\b)")
_DEMONSTRATION = ContextVar("testuary_display_demonstration", default=False)
_PACING = ContextVar("testuary_display_pacing", default=None)
_TARGET_OBSERVER = ContextVar("testuary_target_observer", default=None)
#: The interruptions that end work as stopped, not failed.
_CANCELLED = (KeyboardInterrupt, AsyncCancelledError, FutureCancelledError)


def _ended(exc: BaseException) -> str:
    """How work that ``exc`` ended ends: cancelled or failed."""
    return "cancelled" if isinstance(exc, _CANCELLED) else "failed"


@contextmanager
def observe_target(progress, target: str):
    """Report the phases run inside the block for ``target`` to ``progress``
    (None reports nothing), in this context only."""
    token = _TARGET_OBSERVER.set((progress, target) if progress is not None else None)
    try:
        yield
    finally:
        _TARGET_OBSERVER.reset(token)


def phase_observer(phase: str):
    scope = _TARGET_OBSERVER.get()
    if scope is None:
        return None
    progress, target = scope
    return lambda **fields: progress.emit(target=target, phase=phase, **fields)


_STEPS = ContextVar("testuary_phase_steps", default=None)


class _StepSequence:
    """A phase's named steps, one after the other: entering a step ends the
    one before it, and the phase's end ends the last. Display only: an
    observer's error never reaches the phase."""

    def __init__(self, phase: str):
        self.phase = phase
        self.observe = phase_observer(phase)
        self.open = None

    def _show(self, state: str, message: str, step: str, label: str) -> None:
        if self.observe is None:
            return
        try:
            self.observe(state=state, message=message, operation=f"step/{step}", level="step",
                         label=label)
        except Exception:
            pass

    def enter(self, step: str, label: str) -> None:
        if self.open is not None and self.open[0] == step:
            return
        self.close("succeeded")
        self.open = (step, label)
        self._show("running", "Step started", step, label)

    def close(self, state: str) -> None:
        if self.open is None:
            return
        step, label = self.open
        self.open = None
        self._show(state, {"succeeded": "Step done", "cancelled": "Step stopped"}.get(
            state, "Step failed"), step, label)


def run_with_steps(phase: str, succeeded, function, /, **arguments):
    """``function(**arguments)``, showing the steps it begins with
    ``enter_step`` in order. The last step ends with the phase: succeeded
    when ``succeeded(result)`` holds, else failed, and failed or cancelled
    when the phase raises."""
    sequence = _StepSequence(phase)
    token = _STEPS.set(sequence)
    try:
        result = function(**arguments)
    except BaseException as exc:
        sequence.close(_ended(exc))
        raise
    finally:
        _STEPS.reset(token)
    sequence.close("succeeded" if succeeded(result) else "failed")
    return result


def enter_step(step: str, label: str) -> None:
    """Begin the named step of the phase ``run_with_steps`` runs, ending the
    step before it. Nothing happens outside such a phase."""
    sequence = _STEPS.get()
    if sequence is not None:
        sequence.enter(step, label)


@contextmanager
def observed_step(phase: str, step: str, label: str):
    """Show one named step of a phase while it runs, without counting work.

    The step ends succeeded when its body returns, whatever result it
    records. A raised error or interruption ends it failed or cancelled.
    Inside ``run_with_steps`` for the same phase, the step is one of its
    sequence instead and ends when the next begins.
    """
    sequence = _STEPS.get()
    if sequence is not None and sequence.phase == phase:
        sequence.enter(step, label)
        yield
        return
    single = _StepSequence(phase)
    single.enter(step, label)
    try:
        yield
    except BaseException as exc:
        single.close(_ended(exc))
        raise
    single.close("succeeded")


@contextmanager
def mutation_measurement():
    """Show one mutation run's progress from LangAdapter as rows.

    Yields the callback for ``MutationRequest.progress``, or None outside an
    observed target. Records arrive from LangAdapter's own thread. The row
    "Mutation testing" shows the count, never a score, and ends with the
    run if the engine reported any progress (Java and guest runs report
    none, so no row appears for them). The caller passes the finished run to
    ``report``, so the row ends with the run's stated outcome: a run the wall
    clock cut or that failed does not end as succeeded.
    """
    scope = _TARGET_OBSERVER.get()
    if scope is None:
        yield None
        return
    progress, target = scope
    forward = _MutationForward(progress, target, progress.current_phase(target) or "mutation")
    try:
        yield forward
    except BaseException as exc:
        forward.end(_ended(exc))
        raise
    forward.end(*forward.reported)


class _MutationForward:
    """The ``MutationRequest.progress`` callback of one run, and its end:
    each kind of record (the measurement, the kill verification) is a row
    of its own and ends with the run."""

    def __init__(self, progress, target, phase):
        self._progress, self._target, self._phase = progress, target, phase
        # The last record of each kind, in the order the kinds appeared.
        self._last = {}
        self.reported = ("succeeded", None)
        # Whether the finished run stated a result for every mutant.
        self._complete = False

    def __call__(self, record) -> None:
        self._last[getattr(record, "kind", None)] = record
        self._progress.mutation_progress(self._target, self._phase, record)

    def report(self, run) -> None:
        """Take the end of the row from the finished run's stated outcome."""
        from langadapter import Outcome

        outcome = getattr(run, "outcome", None)
        incomplete = getattr(run, "incomplete", None)
        if incomplete is None and hasattr(run, "runs"):
            # A per-suite run is incomplete when any suite's run is.
            incomplete = any(getattr(suite, "incomplete", None)
                             for suite in run.runs.values())
        if outcome is Outcome.COMPLETED and not incomplete:
            self.reported = ("succeeded", None)
            self._complete = True
        elif outcome is Outcome.TIMED_OUT:
            self.reported = ("failed", "Mutation testing not fully measured: the run's time limit ended it")
        elif outcome is Outcome.COMPLETED:
            # The run finished, but some mutants of the catalogue have no result.
            self.reported = ("failed", "Mutation testing not fully measured: some mutants have no result")
        else:
            self.reported = ("failed", None)

    def end(self, state: str, message: str | None = None) -> None:
        from testuary.presentation.measurement import mutation_progress_ending

        for kind, record in self._last.items():
            # The engine reports progress at intervals, so its last record
            # can lag the run: a run with a result for every mutant ends the
            # measurement's row at the total.
            if (kind == "mutants" and state == "succeeded" and self._complete
                    and getattr(record, "total", None) is not None):
                record = replace(record, finished=record.total)
            # The run's stated reason belongs to the measurement's row.
            text = (message if message and kind == "mutants"
                    else mutation_progress_ending(kind, state))
            self._progress.mutation_progress(self._target, self._phase, record,
                                             state=state, message=text)


def _tool_label(name: str) -> str:
    """A model tool's row name in plain words."""
    from testuary.presentation.view import TOOL_LABELS

    return TOOL_LABELS.get(name, "Tool")


class ToolObserver(BaseCallbackHandler):
    """Show each LangChain tool run as a row named in plain words, and how it
    ended, without its arguments or output. The tool's own name is in the
    message, which only the detailed view shows."""
    def __init__(self, observer):
        self._observe = observer
        self._names = {}
        self._lock = threading.Lock()

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        name = (serialized or {}).get("name", "tool")
        name = name if isinstance(name, str) and name.isidentifier() else "tool"
        with self._lock:
            self._names[run_id] = name
        self._observe(operation=f"tool/{run_id}", state="running", message=f"Running {name}",
                      label=_tool_label(name))

    def _finish(self, run_id, state):
        with self._lock:
            name = self._names.pop(run_id, "tool")
        self._observe(operation=f"tool/{run_id}", state=state, label=_tool_label(name),
                      message=f"{name} " + {"succeeded": "done", "failed": "failed",
                                          "cancelled": "stopped"}[state])

    def on_tool_end(self, output, *, run_id, **kwargs):
        self._finish(run_id, "failed" if getattr(output, "status", None) == "error" else "succeeded")

    def on_tool_error(self, error, *, run_id, **kwargs):
        self._finish(run_id, _ended(error))


@contextmanager
def demonstration_display():
    """Mark the observations shown inside the block as a demonstration's
    playback."""
    token = _DEMONSTRATION.set(True)
    try:
        yield
    finally:
        _DEMONSTRATION.reset(token)


def demonstrating() -> bool:
    """Whether the current display is a demonstration's playback."""
    return _DEMONSTRATION.get()


@contextmanager
def paced_display(pace):
    """Call ``pace`` with every observation before it is displayed, for a
    demonstration's paced playback. It may only wait: execution and saved
    evidence are unchanged."""
    token = _PACING.set(pace)
    try:
        yield
    finally:
        _PACING.reset(token)


@contextmanager
def console_preferences(*, no_color: bool = False, plain: bool = False, diagnostic_writer=None,
                        detail_echo: bool = True):
    """Scope console preferences inside the exclusive invocation guard.

    ``detail_echo`` False keeps ordinary log lines off the console while a
    terminal display shows the run. Warnings and errors are still echoed, and
    saved logs are unchanged.
    """
    global _COLOR_ENABLED, _DIAGNOSTIC_WRITER, _DETAIL_ECHO
    if diagnostic_writer is not None and not callable(diagnostic_writer):
        raise TypeError("diagnostic_writer must be callable")
    previous = _COLOR_ENABLED, _DIAGNOSTIC_WRITER, _DETAIL_ECHO
    try:
        _COLOR_ENABLED = not (no_color or plain)
        _DIAGNOSTIC_WRITER = diagnostic_writer
        _DETAIL_ECHO = bool(detail_echo)
        yield
    finally:
        _COLOR_ENABLED, _DIAGNOSTIC_WRITER, _DETAIL_ECHO = previous


def echoes_log_line(message: str, *, echo: bool = False) -> bool:
    """Whether a plain log line also goes to the console.

    Log lines carry no typed severity. Warnings, errors and critical findings
    are the lines that start with the established ERROR, WARNING, WARN, FATAL
    or CRITICAL prefix, optionally after bracketed tags such as
    ``[mutation]``. A caller passes ``echo`` for an outcome line the person
    running Testuary must see, such as the summary of targets without a
    suite that meets the minimums.
    """
    return _DETAIL_ECHO or echo or bool(_DIAGNOSTIC_LINE.match(message))


# The run's plain-text log (run.log): a logger that writes only there,
# through a file handler while the log is open and, before it opens, a
# handler that keeps the run's first lines.
_RUN_LOG = logging.getLogger("testuary.run")
_RUN_LOG.propagate = False
_RUN_LOG.setLevel(logging.INFO)
_PLAIN_LOG_PENDING_LIMIT = 2000


class _RunLogFormat(logging.Formatter):
    """A record as run.log writes it: the time since the run started, two
    spaces per level, no color codes, one stamped line per message line."""

    def __init__(self, started: float) -> None:
        super().__init__()
        self.started = started

    def format(self, record: logging.LogRecord) -> str:
        hours, rest = divmod(max(0.0, record.at - self.started), 3600)
        minutes, seconds = divmod(rest, 60)
        prefix = (f"{int(hours):02d}:{int(minutes):02d}:{seconds:06.3f} "
                  + "  " * max(0, int(record.depth or 0)))
        return "\n".join(prefix + line
                         for line in _ANSI.sub("", record.getMessage()).splitlines() or [""])


class _RunLogFile(logging.FileHandler):
    def handleError(self, record: logging.LogRecord) -> None:
        pass  # the JSON logs keep the record and the text copy is a convenience


class _RunLogStart(logging.Handler):
    """The run's start and its lines before the log opens, up to the limit."""

    def __init__(self) -> None:
        super().__init__()
        self.started, self.records = time.monotonic(), []

    def emit(self, record: logging.LogRecord) -> None:
        if len(self.records) < _PLAIN_LOG_PENDING_LIMIT:
            self.records.append(record)


def begin_plain_log() -> None:
    """Mark the run's start: lines written before ``open_plain_log`` are kept
    and written first when the log opens, timed from this start."""
    close_plain_log()
    _RUN_LOG.addHandler(_RunLogStart())


def open_plain_log(path) -> str:
    """Start the run's plain-text log at ``path``, appending to an earlier run's
    lines (a resumed run continues its log). Every plain log line of every
    logger is written there with one run-relative timestamp style, also those
    the console leaves out while a display runs, and the lines kept since
    ``begin_plain_log``."""
    start = next((h for h in _RUN_LOG.handlers if isinstance(h, _RunLogStart)), None)
    close_plain_log()
    handler = _RunLogFile(path, mode="a", encoding="utf-8")
    handler.path = str(path)
    handler.setFormatter(_RunLogFormat(start.started if start else time.monotonic()))
    for record in start.records if start else ():
        handler.handle(record)
    _RUN_LOG.addHandler(handler)
    return handler.path


def close_plain_log() -> None:
    for handler in list(_RUN_LOG.handlers):
        _RUN_LOG.removeHandler(handler)
        handler.close()


def plain_log_path() -> str:
    return next((h.path for h in _RUN_LOG.handlers if isinstance(h, _RunLogFile)), "")


def write_plain_log(message: str, level: int = 0) -> None:
    """Append one log line to the run's plain-text log, when one is open, or
    keep it for the log while the run has started and the log is not open."""
    if _RUN_LOG.handlers:
        _RUN_LOG.info(message, extra={"at": time.monotonic(), "depth": level})


def display_owns_timestamps() -> bool:
    """While a terminal display shows the run, echoed log lines leave out
    their elapsed-time prefix, because the display shows its own clock."""
    return not _DETAIL_ECHO


def diagnostic_print(*args, logged: bool = False, **kwargs) -> None:
    """Write an explicit diagnostic unchanged around a scoped terminal refresh.

    The diagnostic also goes to the run's plain-text log. ``logged`` is for a
    caller that already wrote the line there.
    """
    if not logged and kwargs.get("file") in (None, sys.stdout, sys.stderr):
        write_plain_log(kwargs.get("sep", " ").join(str(value) for value in args))
    writer = _DIAGNOSTIC_WRITER
    if writer is None:
        print(*args, **kwargs)
    else:
        writer(lambda: print(*args, **kwargs))


def console_text(text: str) -> str:
    """``text`` without ANSI control sequences, unless color is on and the
    output is a terminal."""
    if _COLOR_ENABLED and getattr(sys.stdout, "isatty", lambda: False)():
        return text
    return _ANSI.sub("", text)


class RunProgress:
    """The progress events of one invocation, sent to ``sink``, with the
    counts of planned and finished work the layered display shows."""

    def __init__(self, sink: Callable[[DisplayEvent], None]) -> None:
        if not callable(sink):
            raise TypeError("sink must be callable")
        self.run_id = uuid.uuid4().hex[:12]
        self._started = time.monotonic()
        self._sink = sink
        # Workers share this observer, not their caller's context variables.
        self._demonstration = _DEMONSTRATION.get()
        self._pacing = _PACING.get()
        self._lock = threading.RLock()
        self._disabled = False
        # Display counts only: planned and finished work for the layered view.
        self._files = self._phases = self._repetitions = self._phase_runs = None
        self._files_done = self._phase_runs_done = 0
        self._phases_done: dict[tuple[str, int], int] = {}
        self._target_runs_done: dict[str, int] = {}
        self._current_phase: dict[str, str] = {}
        #: The run's plain-text log once it is open. Later run events name it.
        self.log_path = ""

    def plan(self, *, files: int, phases: int, repetitions: int) -> None:
        """Record the planned work once the targets and phases are selected."""
        with self._lock:
            self._files, self._phases, self._repetitions = int(files), int(phases), int(repetitions)
            self._phase_runs = self._files * self._phases * self._repetitions

    def run_counts(self) -> dict:
        """The run's phases done of all its files and repetitions, such as
        "5 of 8 phases done (2 targets)" or "(2 targets x 3 repetitions)"."""
        with self._lock:
            return dict(completed=self._phase_runs_done if self._phase_runs is not None else None,
                        total=self._phase_runs, work_unit="phases done" + self._spread())

    def _spread(self) -> str:
        """The files and repetitions the run's phase count spans, when more
        than one."""
        if self._phase_runs is None or self._files * self._repetitions <= 1:
            return ""
        files = f"{self._files} target{'' if self._files == 1 else 's'}"
        if self._repetitions == 1:
            return f" ({files})"
        repetitions = f"{self._repetitions} repetitions"
        return f" ({repetitions})" if self._files == 1 else f" ({files} x {repetitions})"

    def file_counts(self, target: str = "", *, finished: bool = False,
                    already_complete: bool = False) -> dict:
        """Files finished so far, counting ``target`` when it ends.

        Only phase runs that succeeded or were restored count as done. A file a
        resume skipped as already complete counts its planned phase runs as
        restored, since they are recorded. Any other file that stops early
        leaves its remaining phase runs undone, so the run ends short of its
        total.
        """
        with self._lock:
            if finished:
                self._files_done += 1
                if already_complete and self._phase_runs is not None:
                    planned = self._phases * self._repetitions
                    missing = planned - self._target_runs_done.get(target, 0)
                    if missing > 0:
                        self._target_runs_done[target] = planned
                        self._phase_runs_done += missing
            return dict(completed=self._files_done if self._files is not None else None,
                        total=self._files, work_unit="targets")

    def phase_counts(self, target: str, run_idx: int, *, finished: bool = False) -> dict:
        """Phases done in one file run. ``finished`` is true only for a phase
        that succeeded or was restored. A done phase also advances the run's
        phase-run count (``report_run`` shows it)."""
        with self._lock:
            key = (target, run_idx)
            if finished:
                self._phases_done[key] = self._phases_done.get(key, 0) + 1
                self._target_runs_done[target] = self._target_runs_done.get(target, 0) + 1
                self._phase_runs_done += 1
            return dict(completed=self._phases_done.get(key, 0) if self._phases is not None else None,
                        total=self._phases, work_unit="phases")

    def current_phase(self, target: str) -> str | None:
        """The phase most recently started for ``target``."""
        with self._lock:
            return self._current_phase.get(target)

    def mutation_progress(self, target: str, phase: str, record, *, state: str = "running",
                          message: str | None = None) -> None:
        """Show one ``MutationProgress`` record from LangAdapter, or the end of
        its run."""
        from testuary.presentation.measurement import mutation_progress_ending, mutation_progress_event

        with self._lock:
            if self._disabled:
                return
            try:
                event = mutation_progress_event(
                    record, run_id=self.run_id, target=target, phase=phase,
                    elapsed_s=max(0.0, time.monotonic() - self._started),
                    provenance="synthetic" if self._demonstration else "observed")
                if state != "running":
                    event = replace(event, state=state, message=message or
                                    mutation_progress_ending(record.kind, state))
            except Exception:
                return  # an unusable record never changes the run
            self._deliver(event)

    def report_run(self) -> None:
        """Show the run's count of phases done after a phase or file event.
        Before the run is planned it shows nothing."""
        if self._phase_runs is not None:
            self.emit(target="all targets", phase="pipeline", operation="run", level="run",
                      state="running", message="Run in progress", log_path=self.log_path,
                      **self.run_counts())

    def emit(self, *, target: str, phase: str, operation: str, state: str,
             message: str = "", metrics: tuple[Metric, ...] = (),
             provenance: str = "observed", completed: int | None = None,
             total: int | None = None, work_unit: str = "items", usage: Usage | None = None,
             level: str | None = None, label: str = "", log_path: str = "",
             failure: str | None = None) -> None:
        with self._lock:
            if self._disabled:
                return
            if level == "phase" and state == "running":
                self._current_phase[target] = phase
            try:
                event = DisplayEvent(
                    run_id=self.run_id, target=target, phase=phase,
                    operation=operation, state=state,
                    elapsed_s=max(0.0, time.monotonic() - self._started),
                    message=message, metrics=metrics, completed=completed, total=total,
                    work_unit=work_unit, usage=usage,
                    provenance="synthetic" if self._demonstration else provenance,
                    level=level, label=label, log_path=log_path, failure=failure,
                )
            except Exception:
                self._fail()
                return
            self._deliver(event)

    def _deliver(self, event) -> None:
        # Before the display, under the lock: a pause keeps the frame on
        # screen and holds up other workers' observations meanwhile, as the
        # Pacer serializes its waits anyway. An interrupt ends the pause and
        # the run alike.
        pace = self._pacing
        if pace is not None and not self._disabled:
            try:
                pace(event)
            except Exception:
                self._pacing = None
            # The playback clock reads when the frame is shown.
            event = replace(event, elapsed_s=max(event.elapsed_s,
                                                 time.monotonic() - self._started))
        with self._lock:
            if self._disabled:
                return
            try:
                self._sink(event)
            except Exception:
                self._fail()

    def _fail(self) -> None:
        # Observer failure must not change results or trigger a retry.
        with self._lock:
            self._disabled = True
        try:
            sys.stderr.write("Display observer failed. Execution and saved logs continue.\n")
        except (OSError, ValueError):
            pass
