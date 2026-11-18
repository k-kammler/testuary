"""What a display shows of a run, without drawing it.

``RunView`` takes the observations of one run (``DisplayEvent``) and keeps
what the terminal's views are built from: the latest observation of each
identity, when each started and changed, and the issues to keep in view. It
decides which observations a chronological view prints and which the compact
view's log records, and it gives the compact view's rows as plain data: the
layers (Overall, Log, Target, Phase, Step), the visible operations with their
bars, the message, metrics, model usage and recent issues. Every text has its
credentials removed with ``safe_text``.

``TerminalPresenter`` draws these rows with Rich. Other displays, such as a
browser, draw the same rows, so the layout and the texts exist once.
"""
from __future__ import annotations

import os
import re
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass, replace
from typing import Callable
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from .events import DisplayEvent, Metric, Usage

TERMINAL_STATES = {"succeeded", "failed", "cancelled", "refused", "limit_reached"}


def is_outcome(event: DisplayEvent) -> bool:
    """A failed observation the producer marks as a result, not an execution failure."""
    return event.state == "failed" and event.failure == "outcome"


#: A limit's words in a message, as _langgraph._limit_words writes them:
#: "Reply received. Token limit reached" gives "Token limit reached".
_REACHED_LIMIT = re.compile(r"\b\w+ limit reached", re.IGNORECASE)


def _reached_limit(event: DisplayEvent) -> str:
    """Which limit ended ``event``'s work, in its message's words, else
    "limit reached" for a message that names no limit."""
    match = _REACHED_LIMIT.search(event.message or "")
    return match.group(0) if match else "limit reached"


def _state_word(event: DisplayEvent) -> str:
    # A cancelled run was stopped, as the web interface says too. Work that
    # stopped at a limit its settings set names that limit. A file without a
    # suite that meets the minimums is a result, not a failure to execute.
    if is_outcome(event):
        return "MINIMUMS NOT MET"
    if event.state == "limit_reached":
        return _reached_limit(event).upper()
    return {"cancelled": "STOPPED"}.get(event.state, event.state.upper())


_ANSI = re.compile(r"\x1b\][^\x07]*(?:\x07|\x1b\\)|\x1b\[[0-?]*[ -/]*[@-~]")
# A credential name may follow any character but a letter or digit, so the
# names in OPENAI_API_KEY, MY_TOKEN and client_secret count.
_CREDENTIAL = re.compile(
    r"(?i)((?<![^\W_])(?:api[_-]?key|access[_-]?token|token|password|secret)"
    r"\s*['\"]?\s*[:=]\s*['\"]?)([^\s'\",;&]+)")
_AUTH = re.compile(r"(?i)(\b(?:bearer|basic)\s+)([^\s'\",;]+)")
_URL = re.compile(r"https?://[^\s<>\"']+")
_HEX_ID = re.compile(r"\b[0-9a-f]{8}(?:-?[0-9a-f]{4}){3}-?[0-9a-f]{12}\b|\b[0-9a-f]{16,}\b")
_SECRET_NAMES = {"apikey", "api_key", "api-key", "access_token", "token",
                 "password", "secret", "key"}


def safe_text(value: str, secrets: tuple[str, ...] = (), *,
              preserve_whitespace: bool = False) -> str:
    """Console text without terminal controls and recognizable credentials.

    ``secrets`` are removed as well, for a credential without a recognizable
    name. ``preserve_whitespace`` keeps the spaces in paths and commands, not
    line breaks or other controls. What a run writes to its logs is the run's
    own concern.
    """
    value = str(value)
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[redacted]")
    value = _ANSI.sub("", value)
    value = (re.sub(r"[\t\r\n\v\f]", " ", value) if preserve_whitespace
             else " ".join(value.split()))
    value = "".join(c for c in value if ord(c) >= 32 and not 127 <= ord(c) < 160)
    # Stripping controls can join parts of a previously nonmatching secret.
    for secret in secrets:
        if secret:
            value = value.replace(secret, "[redacted]")

    def clean_url(match: re.Match) -> str:
        try:
            parts = urlsplit(match.group())
            host = parts.netloc.rsplit("@", 1)[-1]
            query = urlencode([(key, "[redacted]" if key.lower() in _SECRET_NAMES
                                else val) for key, val in parse_qsl(
                                    parts.query, keep_blank_values=True)])
            fragment = "[redacted]" if parts.fragment else ""
            return urlunsplit((parts.scheme, host, parts.path, query, fragment))
        except ValueError:
            return "[redacted URL]"

    value = _URL.sub(clean_url, value)
    value = _CREDENTIAL.sub(r"\1[redacted]", value)
    return _AUTH.sub(r"\1[redacted]", value)


def _elapsed(seconds: float, provenance: str = "observed") -> str:
    """A clock reading. Synthetic observations come from a demonstration,
    whose clock is the playback's, never the time the recorded run took."""
    minutes, seconds = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    clock = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
    return f"playback {clock}" if provenance == "synthetic" else clock


def _progress(event: DisplayEvent) -> str:
    if event.total is None:
        return (f"{event.completed} {event.work_unit}, total unknown"
                if event.completed is not None else "work total unknown")
    if event.completed is None:
        return f"? of {event.total} {event.work_unit}"
    label = f"{event.completed} of {event.total} {event.work_unit}"
    if event.total > 0 and event.operation not in WORK_COUNTS:
        label += f" ({_percent(event)}%)"
    return label


def _count_step_pending(previous, event: DisplayEvent) -> bool:
    """A running count update that would not yet be worth a line of its own.

    Lines follow ten percent steps. Without a known total, at most one every 30 s.
    Anything else that changed (state, message, metrics, usage) prints at once.
    """
    if previous is None or event.state != "running":
        return False
    last, printed_at = previous
    if (last.state != "running" or last.completed is None or event.completed is None
            or (last.message, last.total, last.work_unit, last.metrics, last.usage,
                last.provenance) != (event.message, event.total, event.work_unit,
                                     event.metrics, event.usage, event.provenance)):
        return False
    if event.elapsed_s - printed_at >= 30:
        return False
    return not event.total or _percent(last) // 10 == _percent(event) // 10


def _percent(event: DisplayEvent) -> int:
    """Whole percent rounded down: 100% only once all work is done."""
    return 100 * event.completed // event.total


_PHASE_LABELS = {"starter": "Starter phase", "coverage": "Coverage phase",
                 "mutation": "Mutation phase", "finishing": "Finishing"}
#: Finishing, the fourth phase: the operation of its events, at the step
#: level, as ``RunPlan.finishing`` plans it.
FINISHING = "step/finishing"
#: What Finishing does, as its row says after its name.
FINISHING_PURPOSE = "measuring the delivered suite and writing the results"
#: The starter phase's count of its suites generated and accepted, and of
#: its accepted suites measured: the operations whose counts the plan's
#: Generate and Measure suites steps show.
STARTER_GENERATION = "suite-processing"
STARTER_MEASUREMENT = "starter-measurement"
_STEP_COUNTS = {STARTER_GENERATION: "generate", STARTER_MEASUREMENT: "measure"}
#: The mutation measurement's progress rows (presentation.measurement). Their
#: counts are of mutants run and kills checked, never of mutants killed, so
#: no percentage stands beside them, where it read as a mutation score.
MUTATION_MEASUREMENT = "mutation-measurement"
KILL_VERIFICATION = "kill-verification"
WORK_COUNTS = frozenset({MUTATION_MEASUREMENT, KILL_VERIFICATION})
#: A loop's check of the model's final answer. A refused answer is one line
#: when it happens, never a pinned row, and a usable answer adds no line: the
#: loop's end says only when no usable answer came.
FINAL_ANSWER = "output-validation"


#: What the Log row adds after the log's path.
LOG_HINT = " (every step: --display detailed)"


def shown_path(path: str) -> str:
    """A path as a row shows it: relative to the working directory when it lies
    below it, else as given. The event keeps the full path."""
    if not os.path.isabs(path):
        return path
    try:
        relative = os.path.relpath(path)
    except ValueError:
        return path
    return path if relative == os.pardir or relative.startswith(os.pardir + os.sep) else relative


def is_finishing(event: DisplayEvent) -> bool:
    """Whether ``event`` observes Finishing, the run's fourth phase."""
    return event.operation == FINISHING
_OPERATION_LABELS = {STARTER_GENERATION: "Generate",
                     STARTER_MEASUREMENT: "Measure suites"}
_REQUEST = re.compile(r"request/(?:[^/]+/)?(\d+)")
#: The model's tools in plain words, as their rows name them. The tool
#: names are the model's own and show only in the detailed view.
TOOL_LABELS = {"read_tests": "Model reads the tests", "read_source": "Model reads the source",
               "edit_tests": "Model edits the tests", "run_tests": "Tests run",
               "measure_coverage": "Coverage measured",
               "measure_mutation": "Mutation score measured"}


def level_of(event: DisplayEvent) -> str:
    """The display layer. An observation without a level is classified by its
    identity: the run event, phase events, everything else an operation."""
    if event.level is not None:
        return event.level
    if event.phase == "pipeline" and event.operation == "run":
        return "run"
    if event.operation == "phase" or event.operation.startswith(("phase/", "restore/")):
        return "phase"
    return "operation"


def _words(name: str) -> str:
    name = name.replace("_", " ").replace("-", " ").strip()
    return name[:1].upper() + name[1:]


def phase_label(phase: str) -> str:
    """A phase's name as the console shows it, such as ``Coverage phase`` for
    ``coverage``: the label other views of a run, like a web page, use."""
    return _PHASE_LABELS.get(phase) or _words(phase)


def own_finishing(label: str) -> bool:
    """Whether a step of Finishing labeled ``label`` is Testuary's own, which
    says what it does: unlabeled or labeled Finishing. Another program's
    last step has a label of its own."""
    return label in ("", phase_label("finishing"))


def _label(event: DisplayEvent) -> str:
    """Plain words for an observation. Internal identifiers are left to debug."""
    if event.label:
        return event.label
    level = level_of(event)
    if level == "run":
        return "Run"
    if level == "target":
        return event.target
    if event.operation.startswith("restore/"):
        return f"{phase_label(event.phase)} (reused)"
    if level == "phase":
        return phase_label(event.phase)
    match = _REQUEST.fullmatch(event.operation)
    if match:
        # The limit names the phase's model requests, not a number past it.
        return "Model requests" if event.state == "limit_reached" else f"Model request {match.group(1)}"
    if event.operation.startswith("tool/"):
        return "Tool"
    known = _OPERATION_LABELS.get(event.operation)
    if known:
        return known
    name = event.operation.removeprefix("step/") if level == "step" else event.operation
    return _words(_HEX_ID.sub("", name).strip("/").replace("/", " ")) or "Operation"


def _identity(event: DisplayEvent) -> str:
    """Full identity, for the debug display level."""
    return f"{event.run_id} | {event.target} | {event.phase} / {event.operation}"


# Evenly sized, repeated measurement work: the only units that get a time-left
# estimate. Model phases never do. Their request durations vary too much.
_ESTIMATED_UNITS = {"mutants", "mutants run", "iterations", "measurements"}


def _has_bar(event: DisplayEvent) -> bool:
    return event.completed is not None and event.total is not None and event.total > 0


def _phase_place(event: DisplayEvent) -> str:
    """A phase's place in its file run, such as ``phase 1 of 4``, from its
    count of phases done: a phase that succeeded counts itself."""
    place = event.completed if event.state == "succeeded" else event.completed + 1
    return f"phase {min(max(place, 1), event.total)} of {event.total}"


def _duration(seconds: float) -> str:
    if seconds < 60:
        return f"{max(1, round(seconds))} s"
    if seconds < 3600:
        return f"{round(seconds / 60)} min"
    hours, rest = divmod(seconds, 3600)
    return f"{int(hours)} h {round(rest / 60)} min"


def _time_left(event: DisplayEvent, item_elapsed: float, done_before: int) -> str:
    """An "about" estimate for evenly sized work, after a few items finished in view.

    Only items finished since the first sighting count: work done before it
    took time that ``item_elapsed`` does not include.
    """
    done = (event.completed or 0) - done_before
    if (item_elapsed <= 0 or event.work_unit not in _ESTIMATED_UNITS
            or not _has_bar(event) or done < 3 or event.completed >= event.total):
        return ""
    return f"about {_duration(item_elapsed / done * (event.total - event.completed))} left"


def _shown_metrics(event: DisplayEvent) -> tuple[Metric, ...]:
    """While work runs, a metric without a value that is unavailable is not
    measured yet and is left out unless it carries a goal. Terminal
    observations keep every reported state."""
    if event.state in TERMINAL_STATES:
        return event.metrics
    return tuple(metric for metric in event.metrics
                 if metric.value is not None or metric.basis != "unavailable"
                 or metric.goal is not None)


def _running_metric_text(metric: Metric, running: bool, provenance: str = "observed") -> str:
    """A not-yet-measured metric of running work shows only its goal."""
    if running and metric.value is None and metric.basis == "unavailable" and metric.goal is not None:
        return f"{metric.name}: goal {metric.goal:g}{metric.unit}"
    return metric_text(metric, provenance)


def _provenance_word(provenance: str) -> str:
    """How a line names an observation's provenance: a demonstration's
    synthetic observations read as demo."""
    return "demo" if provenance == "synthetic" else provenance


def percent_text(value: float) -> str:
    """A percentage as Testuary's lines and pages show it: at most one
    decimal, cut and never rounded up (91.8, not 91.8919 or 91.9), so a
    shown value never reads as a goal met that is not. A whole number has
    no decimal."""
    from testuary import measurement

    return f"{measurement.floor_pct(value, 1):g}"


def metric_text(metric: Metric, provenance: str = "observed") -> str:
    """A metric with its basis. In a demonstration (synthetic observations)
    a value reads as a demo value, never as measured, so a line stays true
    when it is read on its own."""
    goal = "" if metric.goal is None else f" (goal {metric.goal:g}{metric.unit})"
    if metric.basis not in ("measured", "progress", "reused"):
        return f"{metric.name}: {metric.basis}{goal}"
    if metric.value is None:
        status = "unavailable" if metric.basis == "measured" else metric.basis
        return f"{metric.name}: {status}{goal}"
    basis = "demo" if provenance == "synthetic" and metric.basis == "measured" else metric.basis
    value = percent_text(metric.value) if metric.unit == "%" else f"{metric.value:g}"
    return f"{metric.name}: {value}{metric.unit}{goal} [{basis}]"


def _requests_text(counted: int, limit: int | None) -> str:
    """The model requests a limit counts: "model requests 11 of 20" against
    a set limit, else "11 model requests"."""
    return (f"model requests {counted:,} of {limit:,}" if limit is not None
            else f"{counted:,} model request{'' if counted == 1 else 's'}")


#: usage_text's words for a model use with nothing reported.
_NOT_REPORTED = "not reported"


def usage_text(usage: Usage | None, *, both_counts: bool = False) -> str:
    """Model use in one line: the model requests the limit counts, against
    the limit when one is set, then the tokens, such as "model requests 11
    of 20, 3,618 input tokens, 33 output tokens". The requests sent follow
    the count when they differ ("(13 sent, 2 sent again after connection
    errors)"), and always with ``both_counts``, as the detailed view shows
    them."""
    if usage is None:
        return _NOT_REPORTED
    counted, sent, limit = usage.logical_requests, usage.physical_requests, usage.logical_request_limit
    values = []
    if counted is not None:
        # A configured ceiling, not expected work: never drawn as progress.
        text = _requests_text(counted, limit)
        if sent is not None and sent > counted:
            text += f" ({sent:,} sent, {sent - counted:,} sent again after connection errors)"
        elif sent is not None and (sent != counted or both_counts):
            text += f" ({sent:,} sent)"
        values.append(text)
    else:
        if limit is not None:
            values.append(f"request limit {limit:,}")
        if sent is not None:
            values.append(f"{sent:,} sent")
    if usage.input_tokens is not None:
        cached = ("" if usage.cached_input_tokens is None
                  else f" ({usage.cached_input_tokens:,} cache-hit tokens)")
        values.append(f"{usage.input_tokens:,} input tokens{cached}")
    elif usage.cached_input_tokens is not None:
        values.append(f"{usage.cached_input_tokens:,} cache-hit tokens")
    if usage.output_tokens is not None:
        values.append(f"{usage.output_tokens:,} output tokens")
    if usage.cost_usd is not None:
        values.append(f"cost ${usage.cost_usd:.4f}")
    return ", ".join(values) or _NOT_REPORTED


@dataclass(frozen=True)
class Observation:
    """What one observation changes for a display: whether a chronological
    view prints it now, its line, and whether the compact view's log above
    the pinned rows records it."""

    shown: bool
    line: str = ""
    logged: bool = False


_UNSHOWN = Observation(False)


@dataclass(frozen=True)
class Bar:
    """A progress bar: the done share, and its numbers (percent, count,
    elapsed and, for evenly sized measurement work, the time left)."""

    ratio: float
    figures: str


@dataclass(frozen=True)
class Layer:
    """One row of the layers above the operations. ``text`` is the whole
    line as shown, name included. A styled row takes the color of its
    event's state, a dim row is secondary. ``bar`` belongs to ``event``."""

    name: str
    text: str
    event: DisplayEvent | None = None
    styled: bool = False
    dim: bool = False
    bar: Bar | None = None


@dataclass(frozen=True)
class Operation:
    """One visible operation: its state word, name, clock reading, and its
    work as a bar or, without a total, as counted text."""

    event: DisplayEvent
    state: str
    name: str
    elapsed: str
    bar: Bar | None
    progress: str


@dataclass(frozen=True)
class Snapshot:
    """The compact view's rows at one moment, as plain data. ``empty`` holds
    before the first observation. ``playback`` names the clock of the
    newest observation: playback time in a demonstration. ``message``,
    ``metrics`` and ``usage`` are None when the newest observation has none."""

    empty: bool
    title: str = ""
    layers: tuple[Layer, ...] = ()
    operations: tuple[Operation, ...] = ()
    playback: bool = False
    hidden_active: int = 0
    message: str | None = None
    metrics: str | None = None
    usage: str | None = None
    issues: tuple[str, ...] = ()
    cause: str | None = None


class RunView:
    """The observations of one run as a display shows them.

    ``add`` takes each observation in order and says whether and how a
    display shows it. ``snapshot`` gives the compact view's rows. Both run
    under the view's own lock, so observations from several threads keep
    their order.
    """

    def __init__(self, *, secrets: tuple[str, ...] = (), debug: bool = False,
                 detailed: bool = False, clean: Callable[[str], str] | None = None) -> None:
        self.debug = debug
        # The detailed view (debug is one) also names the model's tools and
        # both request counts.
        self.detailed = detailed or debug
        self._secrets = tuple(secrets)
        # Every text passes through ``clean`` before it is shown. A display
        # passes its own, which also removes the run's own credentials
        # (TerminalPresenter.clean). By default the recognizable credentials
        # and ``secrets`` are removed.
        self._clean = clean if clean is not None else self._safe_text
        self._latest: OrderedDict[tuple[str, ...], DisplayEvent] = OrderedDict()
        self._printed: dict[tuple[str, ...], tuple[DisplayEvent, float]] = {}
        # Observation order: when each identity last changed and last started
        # running, so rows can be tied to the phase that is current now.
        self._sequence = 0
        self._updated: dict[tuple[str, ...], int] = {}
        self._started: dict[tuple[str, ...], int] = {}
        self._began: dict[tuple[str, ...], float] = {}
        #: The run's clock: the latest elapsed time any observation carried.
        self._clock = 0.0
        self._done_before: dict[tuple[str, ...], int] = {}
        self._issues: deque[str] = deque(maxlen=3)
        # The failures below the run, in order: the first is the cause.
        self._printed_failures: deque[DisplayEvent] = deque(maxlen=8)
        # Each target's phases' counters for a diagram (``step_counts``), and
        # the target counted last.
        self._counts: dict[tuple[str, str], dict] = {}
        self._counted_target: str | None = None
        self._lock = threading.RLock()

    def add(self, event: DisplayEvent, *, throttle_counts: bool = False) -> Observation:
        """Take one observation. A repeated observation is not shown again
        within 30 seconds. With ``throttle_counts``, as for chronological
        views, a running count is shown only at ten percent steps."""
        with self._lock:
            prior = self._latest.get(event.identity)
            self._sequence += 1
            self._clock = max(self._clock, event.elapsed_s)
            self._updated[event.identity] = self._sequence
            start = event.state == "running" and (prior is None or prior.state != "running")
            # The first sighting in any state also starts an identity: a restored
            # phase may never report running.
            if start or prior is None:
                self._started[event.identity] = self._sequence
                self._began[event.identity] = event.elapsed_s
                self._done_before[event.identity] = event.completed or 0
            elif (event.completed is not None and prior.completed is not None
                    and event.completed < prior.completed):
                # A counter that starts again, such as the next measurement
                # cycle under the same operation, restarts its time left.
                self._began[event.identity] = event.elapsed_s
                self._done_before[event.identity] = event.completed
            previous = self._printed.get(event.identity)
            comparable = (event.state, event.message, event.completed, event.total,
                          event.work_unit, event.metrics, event.usage, event.provenance)
            same = previous is not None and comparable == (
                previous[0].state, previous[0].message, previous[0].completed,
                previous[0].total, previous[0].work_unit, previous[0].metrics,
                previous[0].usage, previous[0].provenance)
            self._latest[event.identity] = event
            self._latest.move_to_end(event.identity)
            self._count_event(event, start)
            if event.state in ("failed", "cancelled", "refused") and level_of(event) in ("run", "target"):
                self._end_open_work(event)
            # Only the newest 64 ended identities stay: the logs keep the history.
            retired = [key for key, item in self._latest.items()
                       if item.state in TERMINAL_STATES][:-64]
            for key in retired:
                del self._latest[key]
                self._printed.pop(key, None)
                self._updated.pop(key, None)
                self._started.pop(key, None)
                self._began.pop(key, None)
                self._done_before.pop(key, None)
            if same and (event.state in TERMINAL_STATES or event.elapsed_s - previous[1] < 30):
                return _UNSHOWN
            if event.operation == FINAL_ANSWER and event.state == "succeeded":
                return _UNSHOWN
            if throttle_counts and _count_step_pending(previous, event):
                return _UNSHOWN
            self._printed[event.identity] = event, event.elapsed_s
            line = self._line(event)
            if event.state in ("failed", "cancelled", "refused"):
                self._issues.append(line)
            if event.state == "failed" and level_of(event) != "run":
                self._printed_failures.append(event)
            # The log above the pinned view records outcomes and the start
            # of each run, file, phase and step. Count updates and running
            # operations stay in the pinned view.
            return Observation(True, line, event.state != "running"
                               or (start and level_of(event) != "operation"))

    def _end_open_work(self, ended: DisplayEvent) -> None:
        """A failed or cancelled run or file ends the phases, steps and
        operations it left running, in its state: none of them goes on."""
        for key, item in list(self._latest.items()):
            if (item.state not in TERMINAL_STATES and level_of(item) in ("phase", "step", "operation")
                    and (level_of(ended) == "run" or item.target == ended.target)):
                self._latest[key] = replace(
                    item, state=ended.state, elapsed_s=max(item.elapsed_s, ended.elapsed_s),
                    message="Ended with the run" if level_of(ended) == "run" else "Ended with the target")

    def _cause(self) -> str | None:
        runs = [event for event in self._latest.values() if level_of(event) == "run"]
        if not runs or runs[-1].state not in TERMINAL_STATES:
            return None
        failed = [event for event in self._printed_failures if event.message]
        if is_outcome(runs[-1]) and not any(
                level_of(event) == "phase" and not is_outcome(event) for event in failed):
            # A run without a suite that meets the minimums, whose phases all
            # ended: a file's own reason says why, before a failure a phase
            # recovered from.
            # A failed phase traces it back to the run's first failure.
            failed = [event for event in failed if is_outcome(event) and level_of(event) == "target"] or failed
        if not failed and runs[-1].state not in ("failed", "refused"):
            return None
        text = (failed[0] if failed else runs[-1]).message
        first = re.split(r"(?<=[.!?])\s+", text.strip(), maxsplit=1)[0]
        return self._safe(first) if first else None

    def latest(self) -> OrderedDict[tuple[str, ...], DisplayEvent]:
        """The latest observation of each identity still in view, oldest first."""
        with self._lock:
            return OrderedDict(self._latest)

    @property
    def provenance(self) -> str:
        """The provenance of the newest observation, observed before any."""
        with self._lock:
            return next(reversed(self._latest.values())).provenance if self._latest else "observed"

    def showing(self) -> bool:
        """Whether a pinned view has something to show: an observation, and
        the run not ended."""
        with self._lock:
            runs = [event for event in self._latest.values() if level_of(event) == "run"]
            return bool(self._latest) and not (bool(runs) and runs[-1].state in TERMINAL_STATES)

    def _count_event(self, event: DisplayEvent, start: bool) -> None:
        """Keep what ``step_counts`` shows of ``event``. A phase that starts
        again, as in the next run of a file, starts its counters again."""
        level = level_of(event)
        key = (event.target, event.phase)
        if level == "phase" and start:
            self._counts.pop(key, None)
        step = _STEP_COUNTS.get(event.operation)
        usage = event.usage
        if step is not None and event.completed is not None and event.total is not None:
            text = f"{event.completed} of {event.total} suites"
        elif (event.operation.startswith("request/") and event.phase in ("coverage", "mutation")
                and usage is not None and usage.logical_requests is not None):
            text = _requests_text(usage.logical_requests, usage.logical_request_limit)
        elif level == "phase" and event.state in TERMINAL_STATES and event.metrics:
            text = None
        else:
            return
        counts = self._counts.setdefault(key, {"steps": {}, "phase": None, "values": []})
        self._counted_target = event.target
        if step is not None:
            counts["steps"][step] = text
        elif text is not None:
            counts["phase"] = text
        else:
            counts["values"] = [metric_text(metric, event.provenance) for metric in event.metrics
                                if metric.value is not None]

    def step_counts(self, phase: str, target: str | None = None) -> dict:
        """The counters a diagram shows for ``phase`` of ``target`` (by
        default the target counted last), from the observations so far:
        ``steps`` maps a planned step to its count (the starter phase's
        suites generated and measured), ``phase`` is a loop's model requests
        against its limit, and ``values`` the loop's measured values once it
        has ended. Only counts the observations carry: nothing is estimated."""
        with self._lock:
            target = self._counted_target if target is None else target
            counts = self._counts.get((target or "", phase))
            if counts is None:
                return {"steps": {}, "phase": None, "values": []}
            return {"steps": dict(counts["steps"]), "phase": counts["phase"],
                    "values": list(counts["values"])}

    def snapshot(self, operations: int, *, one_bar: bool = False) -> Snapshot:
        """The compact view's rows, with at most ``operations`` operations:
        the active ones, and the newest when it just finished. ``one_bar``
        is for a run of one file run, whose Target and Phase bars would repeat
        the Overall count: they keep their text without a bar, and the Phase
        row names its place, such as "phase 1 of 4"."""
        with self._lock:
            if not self._latest:
                return Snapshot(empty=True)
            events = list(self._latest.values())
            latest = events[-1]
            label = ("" if latest.provenance == "observed" else " | demo playback"
                     if latest.provenance == "synthetic" else f" | {latest.provenance} observations")
            title = f"Testuary | {latest.run_id}{label}" if self.debug else f"Testuary{label}"
            work = [e for e in events if level_of(e) == "operation" and e.operation != FINAL_ANSWER]
            active = [e for e in work if e.state not in TERMINAL_STATES]
            finished = latest.state in TERMINAL_STATES and latest in work
            candidates = active + [latest] if finished else active
            visible = candidates[-max(1, operations):]
            shown = tuple(Operation(
                event=event, state=_state_word(event),
                name=self._safe(self._operation_name(event)),
                elapsed=_elapsed(event.elapsed_s, event.provenance),
                bar=self._bar(event, elapsed=False), progress=self._progress(event))
                for event in visible)
            metrics = _shown_metrics(latest)
            running = latest.state not in TERMINAL_STATES
            usage = self._current_usage(events)
            message = self._message(latest)
            return Snapshot(
                empty=False, title=self._safe(title), layers=tuple(self._layers(events, one_bar)),
                operations=shown, playback=latest.provenance == "synthetic",
                hidden_active=len(active) - sum(e.state not in TERMINAL_STATES for e in visible),
                message=self._safe(message) if message else None,
                metrics=self._safe(" | ".join(_running_metric_text(m, running, latest.provenance)
                                              for m in metrics)) if metrics else None,
                usage=None if usage is None else self._safe(usage),
                issues=tuple(self._safe(f"Recent issue: {issue}") for issue in self._issues),
                cause=self._cause())

    def _message(self, event: DisplayEvent) -> str:
        """An observation's message as this view shows it: a tool row's,
        which names the model's tool, only in the detailed view."""
        if event.operation.startswith("tool/") and not self.detailed:
            return ""
        return event.message

    def _safe_text(self, value: str) -> str:
        return safe_text(value, self._secrets)

    def _safe(self, value: str) -> str:
        return self._clean(value)

    def _item_elapsed(self, event: DisplayEvent) -> float:
        # Work still running has run until the run's latest observation,
        # not only until its own: a file's start is its last event for long.
        end = event.elapsed_s if event.state in TERMINAL_STATES else max(event.elapsed_s, self._clock)
        return max(0.0, end - self._began[event.identity])

    def _bar(self, event: DisplayEvent, *, elapsed: bool = True) -> Bar | None:
        """A tqdm-style line's numbers: percentage, count, elapsed and, for
        evenly sized measurement work, an "about" time-left estimate."""
        if not _has_bar(event):
            return None
        item_elapsed = self._item_elapsed(event)
        share = "" if event.operation in WORK_COUNTS else f"{_percent(event):3d}%  "
        figures = f"{share}{event.completed} of {event.total} {self._safe(event.work_unit)}"
        if elapsed:
            figures += f"  {_elapsed(item_elapsed, event.provenance)}"
        left = _time_left(event, item_elapsed, self._done_before.get(event.identity, 0))
        if left:
            figures += f"  {left}"
        return Bar(event.completed / event.total, figures)

    def _progress(self, event: DisplayEvent) -> str:
        text = _progress(event) if event.completed is not None or event.total is not None else ""
        return self._safe(text)

    def _where(self, event: DisplayEvent) -> list[str]:
        """Plain-words position: target file, phase and step or operation."""
        level = level_of(event)
        if level == "run":
            return ["Run"]
        if level == "target":
            return [_label(event)]
        values = [event.target]
        if level == "phase" or event.phase == "pipeline" or is_finishing(event):
            return values + [_label(event)]
        return values + [phase_label(event.phase), _label(event)]

    def _line(self, event: DisplayEvent) -> str:
        label = "" if event.provenance == "observed" else f" [{_provenance_word(event.provenance)}]"
        if event.provenance == "synthetic":
            # A demonstration's clock: the line starts with it as every line does.
            label += " [playback time]"
        values = [f"{_elapsed(event.elapsed_s)} {_state_word(event)}{label}"]
        if self.debug:
            values.append(_identity(event))
        values.extend(self._where(event))
        # Without any work count there is nothing to report. No percent is invented.
        if event.completed is not None or event.total is not None:
            values.append(_progress(event))
        message = self._message(event)
        if message:
            values.append(message)
        running = event.state not in TERMINAL_STATES
        values.extend(_running_metric_text(metric, running, event.provenance)
                      for metric in _shown_metrics(event))
        if event.usage is not None:
            # The reported values name themselves. Nothing reported needs the label.
            text = usage_text(event.usage, both_counts=self.detailed)
            values.append(f"Model use {text}" if text == _NOT_REPORTED else text)
        if event.log_path:
            values.append(f"Log: {shown_path(event.log_path)}")
        return self._safe(" | ".join(values))

    def _operation_name(self, event: DisplayEvent) -> str:
        name = _label(event)
        if event.target != self._current_target():
            name = f"{event.target} | {name}"
        return f"{name} | {_identity(event)}" if self.debug else name

    def _current_target(self) -> str | None:
        work = [e for e in self._latest.values() if level_of(e) != "run"]
        return work[-1].target if work else None

    def _layers(self, events: list[DisplayEvent], one_bar: bool = False) -> list[Layer]:
        """Overall, target, phase and step rows above the low-level
        operations. With ``one_bar`` the Target and Phase rows show no count."""
        rows: list[Layer] = []

        def add(name: str, text: str, event: DisplayEvent | None, styled: bool = False,
                elapsed: bool = True, counts: bool = True) -> None:
            counted = counts and event is not None and not _has_bar(event) and (
                event.completed is not None or event.total is not None)
            line = f"{text} | {_progress(event)}" if counted else text
            bar = self._bar(event, elapsed=elapsed) if counts and event is not None else None
            rows.append(Layer(name, f"{name:<9}" + self._safe(line), event, styled, bar=bar))

        run = next((e for e in reversed(events) if level_of(e) == "run"), None)
        if run is not None:
            # The view redraws only on observations, so a running clock would
            # stand still. The total time is shown once the run has ended.
            state = _state_word(run)
            if run.state in TERMINAL_STATES:
                state += f" {_elapsed(max(e.elapsed_s for e in events), run.provenance)}"
            add("Overall", state, run, True, elapsed=False)
            log = next((e.log_path for e in reversed(events) if level_of(e) == "run" and e.log_path), "")
            if log:
                rows.append(Layer("Log", f"{'Log':<9}" + self._safe(shown_path(log)) + LOG_HINT,
                                  dim=True))
        target, phase, phase_name, since = self._current_phase(events)
        if target is None:
            return rows
        work = [e for e in events if level_of(e) != "run" and e.target == target]
        file_event = next((e for e in reversed(work) if level_of(e) == "target"), None)
        add("Target", _label(file_event) if file_event else target, file_event, counts=not one_bar)
        # Finishing, the fourth phase, takes the Phase row once it has begun.
        # It has no steps of its own.
        finishing = next((e for e in reversed(work) if is_finishing(e)), None)
        if finishing is not None and (phase is None or self._started.get(finishing.identity, 0)
                                      > self._started.get(phase.identity, 0)):
            place = f" | {_phase_place(finishing)}" if one_bar and _has_bar(finishing) else ""
            purpose = f": {FINISHING_PURPOSE}" if own_finishing(finishing.label) else ""
            add("Phase", f"{_label(finishing)}{purpose}{place} | {_state_word(finishing)}",
                finishing, True, counts=not one_bar)
            return rows
        if phase is not None:
            place = f" | {_phase_place(phase)}" if one_bar and _has_bar(phase) else ""
            add("Phase", f"{_label(phase)}{place} | {_state_word(phase)}", phase, True,
                counts=not one_bar)
        elif phase_name != "pipeline":
            add("Phase", phase_label(phase_name), None)
        step = next((e for e in reversed(work) if level_of(e) == "step" and e.phase == phase_name
                     and self._updated.get(e.identity, 0) > since), None)
        if step is not None:
            add("Step", f"{_label(step)} | {_state_word(step)}", step, True)
        return rows

    def _current_phase(self, events: list[DisplayEvent]):
        """The current file, its phase event, phase name and the phase's start order."""
        target = self._current_target()
        if target is None:
            return None, None, None, 0
        work = [e for e in events if level_of(e) != "run" and e.target == target]
        phase = next((e for e in reversed(work) if level_of(e) == "phase"), None)
        if phase is not None:
            return target, phase, phase.phase, self._started.get(phase.identity, 0)
        # Without a phase event, only what happened since the file started counts.
        file_event = next((e for e in reversed(work) if level_of(e) == "target"), None)
        since = self._started.get(file_event.identity, 0) if file_event is not None else 0
        return target, None, work[-1].phase, since

    def _current_usage(self, events: list[DisplayEvent]) -> str | None:
        """Model use of the current file and phase only, labeled with that phase.
        An ended run's summary holds its model use, so none is shown then."""
        target, _, phase_name, since = self._current_phase(events)
        run = next((e for e in reversed(events) if level_of(e) == "run"), None)
        if target is None or (run is not None and run.state in TERMINAL_STATES):
            return None
        usage = next((e.usage for e in reversed(events)
                      if e.usage is not None and e.target == target and e.phase == phase_name
                      and self._updated.get(e.identity, 0) > since), None)
        if usage is None:
            return None
        return f"Model use, {phase_label(phase_name)}: {usage_text(usage, both_counts=self.detailed)}"
