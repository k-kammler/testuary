"""Compact, chronological and redirected views of the same observations,
and the end-of-run summary.

The summary is rendered from facts the producer supplies: each file's
outcome, the delivered suite's paths it found, the final metrics with their
basis and the model use. Nothing is checked, measured or inferred here. A
file without a suite that meets the minimums is a result, kept apart from a
file that failed to execute.
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Literal, Mapping, TextIO

from rich.console import Console, Group
from rich.live import Live
from rich.measure import Measurement
from rich.padding import Padding
from rich.table import Table
from rich.text import Text

from testuary import _words
from testuary.presentation.events import DisplayEvent, Metric, Usage, record_from_dict
from testuary.presentation.view import (
    LOG_HINT,
    TERMINAL_STATES,
    Bar,
    Operation,
    RunView,
    is_outcome,
    metric_text,
    safe_text,
    usage_text,
)

Mode = Literal["auto", "compact", "detailed", "plain", "debug"]
_STYLES = {"running": "cyan", "waiting": "yellow", "retrying": "yellow",
           "succeeded": "green", "failed": "bold red", "cancelled": "yellow", "refused": "yellow",
           "limit_reached": "yellow"}


def _style(event: DisplayEvent) -> str:
    # A result of the method is a warning, never the red of an execution failure.
    return "bold yellow" if is_outcome(event) else _STYLES[event.state]


class _BarLine:
    """A progress bar that fills the width it is given, followed by its numbers.

    Block characters keep the unfinished part visible without color. An ASCII
    console gets # and -. Too little room shows the numbers only.
    """

    def __init__(self, ratio: float, stats: str, style: str, ascii_only: bool) -> None:
        self.ratio, self.stats, self.style = ratio, stats, style
        self.full, self.empty = ("#", "-") if ascii_only else (chr(0x2588), chr(0x2591))

    def __rich_console__(self, console, options):
        size = min(50, options.max_width - len(self.stats) - 1)
        if size < 6:
            yield Text(self.stats)
            return
        # Rounded down, with one empty cell until the work is complete.
        filled = size if self.ratio >= 1 else min(int(size * self.ratio), size - 1)
        yield Text.assemble((self.full * filled, self.style),
                            (self.empty * (size - filled), "dim"), " " + self.stats)

    def __rich_measure__(self, console, options):
        return Measurement(len(self.stats), min(options.max_width, len(self.stats) + 51))


#: The mark of a Log row shortened from the front to fit the terminal.
_SHORTENED = "..."


def _fitted_log(text: str, width: int) -> str:
    """The Log row, fitted to ``width``: a path too long for the terminal
    keeps its end, the log's directories and file name, behind a mark, cut at a
    directory. A narrow terminal drops the hint first. The row's own text, the
    event and the plain log keep the whole path."""
    if len(text) <= width:
        return text
    prefix, rest = text[:9], text[9:]
    hint = LOG_HINT if rest.endswith(LOG_HINT) else ""
    path = rest[:len(rest) - len(hint)]
    for kept in (hint, ""):
        room = width - len(prefix) - len(_SHORTENED) - len(kept)
        if room >= 12:
            tail = path[-room:]
            cuts = [found for found in (tail.find("/", 1), tail.find("\\", 1)) if found > 0]
            return prefix + _SHORTENED + (tail[min(cuts):] if cuts else tail) + kept
    return text


class TerminalPresenter:
    """A callable observer that draws ``view``, a ``RunView``, with Rich. It
    installs no hooks and redirects no global stream.

    As a context manager it restores the terminal on success, failure and
    Ctrl+C, and no display error hides an execution error. Output goes to
    stderr, so stdout stays machine-readable. A redirected or dumb terminal
    gets plain output whatever the display mode. Debug is the detailed view with
    internal identities.
    """

    def __init__(self, mode: Mode = "auto", *, stream: TextIO | None = None,
                 console: Console | None = None, no_color: bool = False,
                 secrets: tuple[str, ...] = (), width: int | None = None,
                 clock: Callable[[], float] = time.monotonic) -> None:
        if mode not in ("auto", "compact", "detailed", "plain", "debug"):
            raise ValueError(f"unknown display mode: {mode}")
        self.debug = mode == "debug"
        if console is not None and stream is not None:
            raise ValueError("choose console or stream, not both")
        if width is not None and (type(width) is not int or width < 1):
            raise ValueError("width must be a positive integer or None")
        # Without an explicit request, Rich applies the NO_COLOR convention itself.
        self.console = console or Console(
            file=stream if stream is not None else sys.stderr, width=width,
            no_color=True if no_color else None, color_system=None if no_color else "auto",
            highlight=False, markup=False, safe_box=True)
        interactive = self.console.is_terminal and not self.console.is_dumb_terminal
        # The requested detail survives the plain fallback of redirected output.
        self.requested_mode = mode
        shown = {"auto": "compact", "debug": "detailed"}.get(mode, mode)
        self.mode = shown if interactive else "plain"
        self._secrets = tuple(secrets)
        self._redact: Callable[[str], str] | None = None
        self._clock = clock
        self.view = RunView(secrets=self._secrets, debug=self.debug,
                            detailed=mode in ("detailed", "debug"), clean=self.clean)
        # A plan of one file and one run: the compact view's one bar is Overall's.
        self._one_bar = False
        self._live: Live | None = None
        self._last_refresh = 0.0
        self._closed = False
        self._diagnostic_depth = 0
        self._lock = threading.RLock()

    def __enter__(self) -> TerminalPresenter:
        return self

    def __exit__(self, exc_type, exc, traceback) -> bool:
        self.close()
        # A deliberate exit follows its own report, and Python prints a text
        # exit message itself. Repeating the status would follow the next step.
        if exc is not None and not isinstance(exc, SystemExit):
            try:
                state = "STOPPED" if isinstance(exc, KeyboardInterrupt) else "FAILED"
                message = "Stopped by you" if state == "STOPPED" else str(exc)
                self._write_plain(f"{state} | {message}")
            except Exception:
                # A shutdown diagnostic must not replace the execution error.
                pass
        return False

    def write_diagnostic(self, write: Callable[[], None]) -> None:
        """Run one diagnostic write unchanged, with the compact view paused
        around it and resumed on a fresh line. Writes and observations are
        serialized, and a nested write resumes the view only after the outer
        one. A display failure falls back to plain output, while ``write`` runs
        once and its exceptions propagate. Writes outside this call are not
        covered.
        """
        with self._lock:
            self._diagnostic_depth += 1
            try:
                if self._diagnostic_depth == 1 and self.mode == "compact":
                    try:
                        self._suspend_live()
                    except Exception:
                        self._fallback_to_plain()
                write()
            finally:
                self._diagnostic_depth -= 1
                if (not self._diagnostic_depth and not self._closed
                        and self.mode == "compact" and self.view.showing()):
                    try:
                        self.console.line()
                        self._start_live()
                        self._last_refresh = self._clock()
                    except Exception:
                        self._fallback_to_plain()

    def plan(self, plan) -> None:
        """Show the run's plan (``RunPlan``) once, in one line, before the
        run's work. A plan of one file and one run leaves the compact view
        one bar, the Overall row's (``RunView.snapshot``)."""
        with self._lock:
            if self._closed:
                return
            self._one_bar = len(plan.files) * len(plan.runs) == 1
            line = f"Plan: {plan.summary()}"
            try:
                if self.mode == "plain":
                    self._write_plain(line)
                else:
                    self.console.print(Text(self._safe(line)), soft_wrap=True)
            except Exception:
                self._fallback_to_plain()
                self._write_plain(line)

    @property
    def provenance(self) -> str:
        """The provenance of the newest observation, observed before any."""
        return self.view.provenance

    def redact_with(self, function: Callable[[str], str]) -> None:
        """Also remove the run's own credentials with ``function``, the
        resolved settings' redaction, from everything shown from now on. The
        patterns alone miss a key that carries no recognizable name."""
        with self._lock:
            self._redact = function

    def clean(self, value: str, *, preserve_whitespace: bool = False) -> str:
        """``value`` as it may be shown: the run's own credentials removed,
        then terminal controls and recognizable credentials."""
        if self._redact is not None:
            value = self._redact(value)
        return safe_text(value, self._secrets, preserve_whitespace=preserve_whitespace)

    def _safe(self, value: str) -> str:
        return self.clean(value)

    def _bar_line(self, bar: Bar, event: DisplayEvent) -> "_BarLine":
        """A tqdm-style line: wide bar, then its numbers from the view."""
        ascii_only = not self.console.encoding.lower().startswith("utf")
        return _BarLine(bar.ratio, bar.figures, _style(event), ascii_only)

    def _work(self, operation: Operation):
        if operation.bar is not None:
            return self._bar_line(operation.bar, operation.event)
        return Text(operation.progress)

    def _write_plain(self, message: str, *, preserve_whitespace: bool = False) -> None:
        # One newline record per message. A terminal wraps long records for
        # display. Inserted breaks would split paths and commands when copied.
        message = self.clean(message, preserve_whitespace=preserve_whitespace)
        text = message + "\n"
        stream = self.console.file
        encoding = getattr(stream, "encoding", None) or "utf-8"
        text = text.encode(encoding, errors="replace").decode(encoding)
        try:
            stream.write(text)
            stream.flush()
        except (OSError, ValueError):
            if stream is not sys.stderr:
                try:
                    sys.stderr.write(text)
                    sys.stderr.flush()
                except (OSError, ValueError):
                    pass

    def frame(self) -> Group:
        """What the compact view draws now, also for looking at it in a test."""
        with self._lock:
            return self._frame()

    def _frame(self) -> Group:
        snapshot = self.view.snapshot(max(1, min(4, self.console.height // 4)), one_bar=self._one_bar)
        if snapshot.empty:
            return Group(Text("Testuary", style="bold"), Text("Waiting for observations"))
        rows: list = [Text(snapshot.title, style="bold")]
        for layer in snapshot.layers:
            style = "dim" if layer.dim else _style(layer.event) if layer.styled else ""
            text = _fitted_log(layer.text, self.console.width) if layer.name == "Log" else layer.text
            rows.append(Text(text, style=style))
            if layer.bar is not None:
                rows.append(Padding(self._bar_line(layer.bar, layer.event), (0, 0, 0, 9)))
        if self.console.width < 60:
            for operation in snapshot.operations:
                rows.append(Text(operation.name))
                rows.append(Text(self._safe(f"{operation.state} | {operation.elapsed}"),
                                 style=_style(operation.event)))
                rows.append(self._work(operation))
        elif snapshot.operations:
            table = Table(box=None, padding=(0, 1), expand=True, show_edge=False)
            table.add_column("State", width=10)
            table.add_column("Operation", ratio=2)
            table.add_column("Progress", ratio=3)
            table.add_column("Playback" if snapshot.playback else "Elapsed",
                             width=17 if snapshot.playback else 8)
            for operation in snapshot.operations:
                table.add_row(Text(operation.state, style=_style(operation.event)),
                              Text(operation.name), self._work(operation),
                              Text(operation.elapsed))
            rows.append(table)
        if snapshot.hidden_active:
            rows.append(Text(f"{snapshot.hidden_active} more running", style="dim"))
        if snapshot.message is not None:
            rows.append(Text(snapshot.message))
        if snapshot.metrics is not None:
            rows.append(Text(snapshot.metrics))
        if snapshot.usage is not None:
            rows.append(Text(snapshot.usage, style="dim"))
        rows.extend(Text(issue, style="yellow") for issue in snapshot.issues)
        return Group(*rows)

    def __call__(self, event: DisplayEvent) -> None:
        with self._lock:
            if self._closed:
                return
            observed = self.view.add(event, throttle_counts=self.mode in ("plain", "detailed"))
            if not observed.shown:
                return
            if self.mode == "compact" and self._diagnostic_depth:
                return
            try:
                if self.mode == "plain":
                    self._write_plain(observed.line)
                elif self.mode == "detailed":
                    # One record per line, as in plain output. The terminal wraps it.
                    self.console.print(Text(observed.line, style=_style(event)),
                                       soft_wrap=True)
                else:
                    self._start_live()
                    # The view says what the log above the pinned view records.
                    if observed.logged:
                        self._live.update(self.frame(), refresh=False)
                        # Printed above the live view, so only the view's own height
                        # matters for redrawing. The terminal wraps this record.
                        self.console.print(Text(observed.line, style=_style(event)),
                                           soft_wrap=True)
                    now = self._clock()
                    refresh = event.state in TERMINAL_STATES or now - self._last_refresh >= .25
                    self._live.update(self.frame(), refresh=refresh)
                    if refresh:
                        self._last_refresh = now
            except Exception:
                self._fallback_to_plain()
                self._write_plain(observed.line)

    def _start_live(self) -> None:
        if self._live is None:
            self._live = Live(self.frame(), console=self.console,
                              auto_refresh=False, redirect_stdout=False,
                              redirect_stderr=False, screen=False, transient=False)
            self._live.start(refresh=True)

    def _suspend_live(self) -> None:
        if self._live is not None:
            live = self._live
            transient = live.transient
            live.transient = True
            try:
                live.stop()
            finally:
                live.transient = transient
                self._live = None

    def _fallback_to_plain(self) -> None:
        self._stop_live()
        self.mode = "plain"
        try:
            self._write_plain("Presentation unavailable, using plain output.")
        except Exception:
            pass

    def _stop_live(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def close(self) -> None:
        with self._lock:
            if not self._closed:
                self._stop_live()
                self._closed = True


Outcome = Literal["complete", "no_complete_suite", "failed", "cancelled", "already_complete",
                  "stopped"]
OUTCOMES = ("complete", "no_complete_suite", "failed", "cancelled", "already_complete", "stopped")
#: The outcomes the headline counts after those with a suite that meets the
#: minimums, in its words.
_COUNTED = {"no_complete_suite": "without one", "failed": "failed to execute",
            "cancelled": "stopped before finishing", "stopped": "ended after the requested phase"}
# Outcomes a resume could continue. A file stopped on request delivered nothing,
# but it is neither a failure nor unfinished work.
_UNFINISHED = ("no_complete_suite", "failed", "cancelled")


@dataclass(frozen=True)
class FileCompletion:
    """One selected file at the end of the run, as the producer observed it.

    ``resumable`` is False for a file that a resume would skip, such as a result
    below its configured minimum. The resume step then leaves it out.
    """

    path: str
    outcome: Outcome
    reason: str = ""
    suite: tuple[str, ...] = ()
    metrics: tuple[Metric, ...] = ()
    usage: Usage | None = None
    resumable: bool = True
    #: The fallbacks that changed the method for this file, each a sentence.
    fallbacks: tuple[str, ...] = ()
    #: The warnings of this file's runs, such as a time limit reached, each
    #: a sentence.
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.path, str) or not self.path:
            raise ValueError("a file completion needs its path")
        if self.outcome not in OUTCOMES:
            raise ValueError("unrecognized file outcome")
        object.__setattr__(self, "suite", tuple(self.suite))
        object.__setattr__(self, "metrics", tuple(self.metrics))
        object.__setattr__(self, "fallbacks", tuple(self.fallbacks))
        object.__setattr__(self, "warnings", tuple(self.warnings))
        if any(not isinstance(path, str) for path in self.suite):
            raise ValueError("suite paths must be text")
        if any(not isinstance(metric, Metric) for metric in self.metrics):
            raise ValueError("metrics must be Metric values")
        if self.usage is not None and not isinstance(self.usage, Usage):
            raise ValueError("usage must be a Usage value or None")
        if type(self.resumable) is not bool:
            raise ValueError("resumable must be True or False")


@dataclass(frozen=True)
class RunCompletion:
    """The whole invocation: its files, where results and logs are, how to continue.

    ``resume_command`` is the producer's exact command for continuing this
    run, or None when there is nothing to continue or no such command.
    """

    files: tuple[FileCompletion, ...]
    results_directory: str | None = None
    log_path: str | None = None
    resume_command: str | None = None
    cancelled: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "files", tuple(self.files))
        if any(not isinstance(item, FileCompletion) for item in self.files):
            raise ValueError("files must be FileCompletion values")
        for name in ("results_directory", "log_path", "resume_command"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be text or None")

    @property
    def unfinished(self) -> tuple[FileCompletion, ...]:
        """The files a resume would continue: those without a suite that
        meets the minimums that the producer did not mark as not resumable."""
        return tuple(item for item in self.files if item.outcome in _UNFINISHED and item.resumable)

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "RunCompletion":
        """The completion a mapping of its fields describes
        (``record_from_dict``)."""
        return record_from_dict(cls, data)


#: The end-of-run summary's file in a run's results directory, beside
#: ``summary.json``, and its format's version.
COMPLETION_FILE = "completion.json"
COMPLETION_SCHEMA = 1


def write_completion(completion: RunCompletion, *, provenance: str = "observed",
                     clean: Callable[[str], str] | None = None) -> Path | None:
    """Write the end-of-run summary into its results directory as
    ``completion.json``, in one step, so a reader sees the old or the new
    file: the format's version, the provenance of the run's values (demo
    values are ``synthetic``) and the completion. Every text has its
    credentials removed (``clean``, else the recognizable ones). A resumed
    run rewrites it. None when the completion names no results directory."""
    if not completion.results_directory:
        return None

    def safe(value):
        if isinstance(value, str):
            return clean(value) if clean is not None else safe_text(value, ())
        if isinstance(value, dict):
            return {key: safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        return value

    record = {"schema_version": COMPLETION_SCHEMA, "provenance": provenance,
              "completion": safe(asdict(completion))}
    results_dir = Path(completion.results_directory)
    results_dir.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".completion.", suffix=".tmp", dir=str(results_dir))
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(record, stream, indent=2, allow_nan=False)
            stream.write("\n")
        os.replace(temporary, results_dir / COMPLETION_FILE)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)
    return results_dir / COMPLETION_FILE


def read_completion(path: str | Path) -> tuple[RunCompletion, str]:
    """The end-of-run summary at ``path``, a run's
    ``results/completion.json``, as the pair of its ``RunCompletion`` and
    its provenance as the file records it, ``synthetic`` for a
    demonstration's values. Only the file is read.

    Raises ValueError when the file is not JSON or does not hold this
    Testuary's summary format version and a completion, ValueError or
    TypeError when the completion's fields do not match this version's,
    and OSError when the file cannot be read."""
    record = json.loads(Path(path).read_text(encoding="utf-8"))
    if (not isinstance(record, dict) or record.get("schema_version") != COMPLETION_SCHEMA
            or not isinstance(record.get("completion"), dict)):
        raise ValueError(f"{path} is not an end-of-run summary of this Testuary")
    return RunCompletion.from_dict(record["completion"]), str(record.get("provenance") or "observed")


def headline(completion: RunCompletion) -> str:
    """The run's outcome in one sentence: how many files have a suite that
    meets the minimums, and how many ended otherwise."""
    files = completion.files
    done = sum(item.outcome in ("complete", "already_complete") for item in files)
    others = {outcome: sum(item.outcome == outcome for item in files) for outcome in _COUNTED}
    status = ("Run stopped" if completion.cancelled else
              "Run finished" if not others["failed"] else "Run finished with failures")
    parts = [f"{done} of {len(files)} target{'s' if len(files) != 1 else ''} "
             "with a suite that meets the minimums"]
    parts += [f"{count} {_COUNTED[outcome]}" for outcome, count in others.items() if count]
    return f"{status}: " + ", ".join(parts) + "."


def _lines(completion: RunCompletion, provenance: str = "observed") -> list[str]:
    lines = [headline(completion)]
    if completion.results_directory:
        lines.append(f"Results: {completion.results_directory}")
    if completion.log_path:
        lines.append(f"Log: {completion.log_path}")
    for item in completion.files:
        lines.append("")
        lines.append(f"{item.path}: {_words.OUTCOME_WORDS[item.outcome]}"
                     + (f": {item.reason}" if item.reason else ""))
        if item.suite:
            lines.append("  Suite: " + ", ".join(item.suite))
        if item.metrics:
            lines.append("  " + " | ".join(metric_text(metric, provenance) for metric in item.metrics))
        if item.usage is not None:
            lines.append(f"  Model use: {usage_text(item.usage)}")
        for fallback in item.fallbacks:
            lines.append(f"  Fallback: {fallback}")
        for warning in item.warnings:
            lines.append(f"  Warning: {warning}")
    # Only files a resume would continue: the producer marks the others.
    unfinished = completion.unfinished
    if completion.resume_command and unfinished:
        lines += ["", f"Next step: {completion.resume_command} continues the unfinished "
                      f"target{'s' if len(unfinished) != 1 else ''}."]
    return lines


def present_completion(completion: RunCompletion, *, presenter: TerminalPresenter) -> None:
    """Write the end-of-run summary through the presenter's diagnostic boundary."""
    if not isinstance(completion, RunCompletion):
        raise TypeError("completion must be a RunCompletion")
    # A demonstration's summary keeps its observations' provenance: its
    # values read as demo values, never as measured.
    lines = _lines(completion, presenter.provenance)

    def write() -> None:
        for line in lines:
            presenter._write_plain(line, preserve_whitespace=True)

    presenter.write_diagnostic(write)
