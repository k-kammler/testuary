"""Framework-independent scaffolding for the per-phase agents:

1. Tool-result helpers: tool messages, the phase log lines, the metric
   log lines, truncation that keeps banners and elapsed-time formatting.
2. Edit and write helpers: the read views' line numbers,
   ``assert_write_under``, ``source_structure_error`` and the
   unsound-async check of a passing run. The edit tool's matching is in
   _edit_tool.py.
3. The coding-agent session state, the loop breaker and the best passing
   snapshot.
4. ``AgentLogger``: console output and JSON log streaming. Messages render
   through ``_chat_log._extract_message_for_log``.

Agent-runtime code, partly shared with the phase orchestration (the
logger, the phase log lines, the metric lines and the structure
message). Helpers that only ``__main__.py`` or the run orchestrator call
belong in config.py or project.py instead.
"""
from __future__ import annotations

import contextlib
import logging
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from pythonjsonlogger.json import JsonFormatter

from testuary.agent import _limits


# Tool-result helpers (pure functions, no state).


def tool_msg(prompts: Dict[str, Any], key: str, **subs: Any) -> str:
    """The tool-facing message ``prompts['tool_messages'][key]`` with the
    runtime values ``subs`` substituted through ``string.Template``
    (``$var``, ``${var}``), so the prompt file's text stays data. Every agent
    keeps its model-facing error, rejection and hint texts there (its
    ``prompts/<agent>.json``). A missing ``key`` raises ``KeyError``, so a
    prompt-file typo is loud. ``$$`` writes one dollar sign, with values or
    without.
    """
    msgs = prompts.get("tool_messages", {})
    return Template(msgs[key]).substitute(**subs)


def log_phase_started(log_fn: Any, name: str) -> None:
    """Emit a phase's START log line through ``log_fn``.
    Single source of the format so every phase's log lines cannot drift."""
    log_fn(f">>> phase START: [{name}]")


def log_phase_finished(log_fn: Any, name: str, outcome: str = "ok") -> None:
    """Emit a phase's END log line through ``log_fn``."""
    log_fn(f"<<< phase END:   [{name}] outcome={outcome}")


# Metric reporting: one source, so every phase reports the same shape.

def _fmt_pct(value: Optional[float]) -> str:
    """A percentage to two decimals, or ``n/a`` when it was not measured."""
    return f"{value:.2f}" if value is not None else "n/a"


def format_metric_distribution(
    label: str, stats: Optional[Dict[str, float]],
) -> str:
    """One metric's per-unit distribution as a single aligned line:
    ``label  mean=.. median=.. std=.. min=.. max=..``. ``label  n/a`` when the
    distribution is empty (no measured unit), so a reader sees the metric was
    requested but not available rather than a missing line."""
    if not stats:
        return f"{label:<8} n/a"
    return (
        f"{label:<8} "
        f"mean={stats.get('mean', float('nan')):.2f} "
        f"median={stats.get('median', float('nan')):.2f} "
        f"std={stats.get('std', float('nan')):.2f} "
        f"min={stats.get('min', float('nan')):.2f} "
        f"max={stats.get('max', float('nan')):.2f}"
    )


def log_metric_distribution(
    log_fn: Any, phase: str, n: int,
    line_stats: Optional[Dict[str, float]],
    branch_stats: Optional[Dict[str, float]],
    mutation_stats: Optional[Dict[str, float]],
) -> None:
    """Log the per-unit distribution of the three metrics under one header,
    so every phase with a population of units (the starter phase's
    suites) reports line, branch and mutation in the same shape: mean,
    median, std, min and max."""
    log_fn(f"[{phase}] per-suite distribution before merge (n={n}):")
    log_fn(f"[{phase}]   {format_metric_distribution('line', line_stats)}")
    log_fn(f"[{phase}]   {format_metric_distribution('branch', branch_stats)}")
    log_fn(f"[{phase}]   {format_metric_distribution('mutation', mutation_stats)}")


def log_phase_metrics(
    log_fn: Any, phase: str, *, line: Optional[float],
    branch: Optional[float], mutation: Optional[float],
    tests: Optional[int] = None, note: str = "",
) -> None:
    """Log a phase's single result across all three metrics in one line, so
    every phase end reports line / branch / mutation together (an unmeasured
    metric reads ``n/a`` rather than being silently absent). ``note`` carries a
    short phase-specific prefix (e.g. the selection counts)."""
    head = f"[{phase}] {note}".rstrip()
    tail = f" tests={tests}" if tests is not None else ""
    log_fn(
        f"{head} line={_fmt_pct(line)} branch={_fmt_pct(branch)} "
        f"mutation={_fmt_pct(mutation)}{tail}"
    )


#: The marker the history trimmer puts in place of elided write-tool
#: payloads. It is deliberately opaque (no prose, no instructions, no
#: plausible-content variants): thinking-mode models take a prose-shaped
#: placeholder for a real payload and echo it into a new edit call.
#: ``_context.dedup_stub`` emits it, ``contains_history_marker`` detects
#: it, and the prompts show it through the ``elided_sentinel`` slot.
ELIDED_SENTINEL = "<<<TESTUARY_ELIDED>>>"

#: Detection matches the opening so a tail truncated at a context
#: boundary still trips the write-tool validators across all agents.
TRIMMED_HISTORY_MARKER = ELIDED_SENTINEL.rstrip(">")


def contains_history_marker(payload: Optional[str]) -> bool:
    """True if ``payload`` contains the history-trim placeholder
    (``TRIMMED_HISTORY_MARKER``). Write-tool validators reject such
    payloads, so the marker never lands on disk as test or source content.
    """
    if not payload:
        return False
    return TRIMMED_HISTORY_MARKER in payload


def looks_placeholder_text(
    text: Optional[str], fragments: "Sequence[str]", *, min_len: int = 12,
) -> bool:
    """The output validators' content check: True when ``text`` is empty,
    shorter than ``min_len`` characters after stripping, or echoes one of
    the agent's ``fragments`` (the field-description phrases of its prompt
    that a lazy model copies back). One implementation keeps the agents'
    result-field checks in sync. ``min_len=0`` is the looser check for
    short legitimate values (function names)."""
    if not text:
        return True
    t = text.strip().lower()
    if min_len and len(t) < min_len:
        return True
    return any(fragment in t for fragment in fragments)


# Edit and write helpers: line numbers, write safety, structure.


def number_line(number: int, text: str) -> str:
    """One line of the read and edit views: its 1-based number, a tab, then
    the line. ``strip_line_number_prefixes`` removes exactly this form."""
    return f"{number}\t{text}"


def coerce_paths_to_str(data: Any) -> Any:
    """A pydantic ``model_validator(mode='before')`` helper: stringifies every
    ``pathlib.Path`` value of the input dict, also inside a list or tuple,
    so a ``Deps`` field declared ``str`` accepts the orchestrator's Paths
    without a ``str(...)`` at every call site. A non-dict input (a sub-model
    being built) is returned unchanged.
    """
    if not isinstance(data, dict):
        return data
    out: Dict[str, Any] = {}
    for k, v in data.items():
        if isinstance(v, Path):
            out[k] = str(v)
        elif isinstance(v, (list, tuple)) and not hasattr(v, "_fields"):
            # A NamedTuple (marked by ``_fields``) cannot be rebuilt from a
            # generator, so only plain lists and tuples are coerced.
            kind = type(v)
            out[k] = kind(str(x) if isinstance(x, Path) else x for x in v)
        else:
            out[k] = v
    return out


class SandboxEscapeError(Exception):
    """Raised when a write would land outside the agent's test output directory.

    The directory is ``deps.test_output_dir``, the tests directory beside the
    target's copy of the project
    (``<outdir>/_work[/<job_ns>]/<workspace>/tests``, in container mode the
    work directory's ``tests``). Every write of the edit tool (the loop's test
    file) must land under it. Anywhere else (a stale absolute path, a path
    resolved through a symlink, a deps object the orchestrator did not
    override) is presumed a bug and refused. The check knows only the
    directory the agents may touch, so it depends on resolving no other path.
    """


def assert_write_under(target: str, test_output_dir: str) -> None:
    """Refuse the write when ``target`` is not under ``test_output_dir``.

    Both paths are resolved first (symlinks followed, ``..`` collapsed), so
    ``<tests>/../../projects/sympy/foo.py`` counts as escaping. Raises
    ``SandboxEscapeError`` on an escape, and when ``test_output_dir`` is empty
    or a path cannot be resolved, since the sandbox cannot be proven then.
    Tool callers catch it at the write site and answer with a refusal. They
    never write anyway, since a write outside the directory could corrupt
    the canonical source.
    """
    if not test_output_dir:
        raise SandboxEscapeError(
            f"sandbox check refused: the test output folder is empty; "
            f"target={target!r}. The agent's deps must set "
            f"test_output_dir to the target's tests folder."
        )
    try:
        target_abs = Path(target).resolve()
        output_root = Path(test_output_dir).resolve()
    except Exception as exc:
        raise SandboxEscapeError(
            f"sandbox check refused: could not resolve paths "
            f"(target={target!r}, test_output_dir={test_output_dir!r}): "
            f"{type(exc).__name__}: {exc}"
        ) from exc
    try:
        target_abs.relative_to(output_root)
    except ValueError:
        raise SandboxEscapeError(
            f"sandbox check REFUSED write: target {target_abs} is NOT "
            f"under the test output folder {output_root}. A write "
            f"outside it can corrupt the canonical source; "
            f"every agent write must land inside ``test_output_dir``. "
            f"Check whether the orchestrator forgot to override "
            f"target_file_path / test_output_dir."
        )


def structure_message(result: Any) -> str:
    """A structural parser's verdict: the kind and place, then the offending
    source quoted as written, its line breaks and indentation read as single
    spaces, cut to ``syntax_snippet_chars`` from its start with the marker
    outside the quotes. A missing token quotes an empty source (``: ''``).
    A language that gives no source (None) reads as the bare message."""
    whole = getattr(result, "snippet", None)
    if whole is None:
        return result.message
    flat = " ".join(whole.split())
    limit = _limits.value("syntax_snippet_chars")
    shown = flat if limit is None else flat[:limit]
    return f"{result.message}: '{shown}'" + (
        " " + _limits.marker(len(flat) - len(shown), "chars", "the source")
        if len(shown) < len(flat) else "")


def source_structure_error(content: str, language: str) -> Optional[str]:
    """None if ``content`` is syntactically valid for ``language``, else the
    formatted error with the offending line and column in a five-line
    context window.

    It goes through the language backend's ``check_structure``
    (tree-sitter), so every supported language gets the same gate. A
    language without a backend returns None (code that cannot be parsed is
    never rejected). Write-tool validators use it to refuse a malformed
    payload at the write instead of two tool calls later as a failing run.
    The parse takes milliseconds even on files of thousands of lines.
    """
    # Local import: importing this module does not load the language
    # backends.
    from testuary import languages

    backend = languages.get_backend(language)
    if backend is None:
        return None
    result = backend.check_structure(content or "")
    if result.ok:
        return None
    window = _source_window(content, result.line or 0, result.col or 0, "→ ")
    return f"{structure_message(result)}\n\n{window}"


def source_validation_error(content: str, validation) -> Optional[str]:
    """Return a formatted authoritative target-validator diagnostic."""
    if validation.ok:
        return None
    window = _source_window(content, validation.line or 0, validation.col or 0, "> ")
    return f"{validation.status}: {validation.message}\n\n{window}"


def _source_window(content: str, line_no: int, col: int, marker: str) -> str:
    """Two lines of context before and after ``line_no``, that line marked
    with ``marker`` and a caret under the column ``col`` when it lies on
    the line, or "(empty file)" when no line is in range."""
    lines = (content or "").splitlines()
    snippet: List[str] = []
    for i in range(max(1, line_no - 2), min(len(lines), line_no + 2) + 1):
        snippet.append(f"{marker if i == line_no else '  '}{i:4d}: {lines[i - 1]}")
        if i == line_no and 0 < col <= len(lines[i - 1]) + 1:
            snippet.append("       " + (" " * (col - 1)) + "^")
    return "\n".join(snippet) if snippet else "(empty file)"


def stopping_tests(run: Any) -> Tuple[List[str], bool]:
    """The tests a run under a measurement tool names as failing, and whether
    the run is red: it names a failure, a failed test or an error."""
    stopping = [failure.test for failure in getattr(run, "failures", ()) or ()]
    return stopping, bool(stopping) or int(run.failed or 0) + int(run.errors or 0) > 0


def coverage_counts_inconsistent(run: Any) -> bool:
    """Whether a run that collected tests left coverage counts the coverage
    tool could not read: its ``coverage_counts_inconsistent`` diagnostic.
    A run that collected no test has another cause."""
    return bool(run.collected) and any(
        entry.kind == "coverage_counts_inconsistent" for entry in run.diagnostics)


def unsound_async_rejection(
    run: Any,
    tool_msg: Callable[..., str],
    msg_key: str = "tests_strict_green_unsound",
) -> Optional[str]:
    """The strict-green check of a passing run: when ``run`` (a
    ``TestRunResult``) has returncode 0 and unsound-async test warnings
    (its ``unsound`` entries), the rejection
    text that turns it failing and names the offending tests, else None.

    A leaked-async test (an unawaited coroutine, a leaked task, an
    unraisable exception) can pass falsely, which would poison the best
    passing snapshot or a runtime measurement. Counting it as failing keeps
    it out, and the named tests let the model fix them. A nonzero
    returncode already fails, so it returns None. A language without the
    concept reports no entries. ``tool_msg`` resolves ``msg_key`` with a
    ``sources_block`` substitution.
    """
    if run.returncode != 0 or not run.unsound:
        return None
    sources = [(test, warn) for test, warn in run.unsound if test is not None]
    if sources:
        sources_block = (
            "Tests that emitted unsound-async warnings:\n  - "
            + "\n  - ".join(_limits.cut_lines(
                [f"{test}  ({warn})" for test, warn in sources],
                _limits.value("unsound_async_items"), what="further tests"))
            + "\n"
        )
    else:
        sources_block = (
            "Detected warnings (test names not parsed):\n  - "
            + "\n  - ".join(_limits.cut_lines(
                [warn for _test, warn in run.unsound], _limits.value("unsound_async_warnings"),
                what="further warnings"))
            + "\n"
        )
    return tool_msg(msg_key, sources_block=sources_block)


def format_thinking_elapsed(seconds: float) -> Optional[str]:
    """Format ``seconds`` as ``"2m45s"`` / ``"3.4s"`` / ``None`` (<1s suppressed)."""
    if seconds >= 60:
        return f"{int(seconds) // 60}m{int(seconds) % 60:02d}s"
    if seconds >= 1:
        return f"{seconds:.1f}s"
    return None


# Retry-prompt fragments (the agent's retry prompt).

TRUNCATION_HINT = (
    "\n\n[Framework note: your previous response was cut at "
    "max_tokens (finish_reason='length'). The same generation "
    "shape will likely truncate again. Be more concise on this "
    "turn -- emit one focused tool call rather than a long "
    "thinking block followed by a large payload. If you intended "
    "a big edit, split it into several smaller edits instead.]"
)


# JSON Lines logs (the agent logger's and the trace logger's).


class JsonLinesHandler(logging.Handler):
    """Appends each record to ``path`` as one line, the JSON object that
    python-json-logger makes of the record's dict message, ended by the
    platform's line end. The file is opened for each record, so no handle
    stays open and a reader can follow it while it grows. A record that
    cannot be written goes to ``on_error`` instead of logging's printed
    traceback: a log never stops a run."""

    def __init__(self, path: Path,
                 on_error: Optional[Callable[[BaseException], None]] = None) -> None:
        super().__init__()
        self.path = path
        self._on_error = on_error
        self.setFormatter(JsonFormatter([], json_ensure_ascii=False, json_default=str))

    def emit(self, record: logging.LogRecord) -> None:
        try:
            line = self.format(record)
            with open(self.path, "a", encoding="utf-8") as stream:
                stream.write(line + "\n")
        except Exception as exc:
            if self._on_error is not None:
                self._on_error(exc)


def json_lines_logger(path: Path, on_error: Optional[Callable[[BaseException], None]] = None
                      ) -> logging.Logger:
    """A logger whose records, each a dict, go to ``path`` as JSON lines.
    It is outside logging's registry, so it ends with its owner."""
    logger = logging.Logger(f"testuary.{path.name}")
    logger.addHandler(JsonLinesHandler(path, on_error))
    return logger


# Agent logger (console output and the JSON Lines log).

class _Colors:
    """ANSI color codes for terminal output."""
    RESET = "\033[0m"
    BRIGHT_GREEN = "\033[92m"


@dataclass
class AgentLogger:
    """Elapsed-time console output and the JSON Lines log of one agent run.

    The log holds one JSON object per line, in order, each naming its kind
    in ``record``: ``start`` (the project, the agent, the clock domain, the
    start time and the configuration), ``console`` (a logged line: its level
    and message, ``debug`` for a detail entry, and its timing fields),
    ``analysis`` (a finished agent run with its conversation) and ``end``
    (the end time, which ``save`` writes).

    Usage:
        logger = AgentLogger("my_project", agent_name="coverage")
        logger.init()
        logger.enable_streaming(str(logs_dir), "log_${project_name}.jsonl")
        logger.log("Starting analysis")
        logger.log_analysis(file_path, duration_ms, messages)
        logger.save(str(logs_dir), "log_${project_name}.jsonl")
    """

    project_name: str
    agent_name: str = "agent"
    config: Dict[str, Any] = field(default_factory=dict)
    #: The reasoning fields a logged conversation keeps, which the agent
    #: sets from its model's settings. None keeps the default ones.
    reasoning_field_names: Optional[Sequence[str]] = field(default=None, init=False)

    _console_logs: List[Dict] = field(default_factory=list, init=False)
    _analyses: List[Dict] = field(default_factory=list, init=False)
    _initialized: bool = field(default=False, init=False)
    _streaming_path: Optional[Path] = field(default=None, init=False)
    # Reentrant, so the same thread can re-enter (log_analysis, then the
    # flush). Needed because the tool calls of one model response and the
    # starter phase's suite generations log from worker threads.
    _lock: threading.RLock = field(default_factory=threading.RLock, init=False)
    _clock_start: float = field(default_factory=time.monotonic, init=False)
    _finished_timing: Optional[Dict[str, Any]] = field(default=None, init=False)
    _log: Optional[logging.Logger] = field(default=None, init=False)

    def init(self) -> None:
        """Reset per-run state."""
        with self._lock:
            self._console_logs.clear()
            self._analyses.clear()
            self._clock_start = time.monotonic()
            self._finished_timing = None
            self._initialized = True

    def enable_streaming(self, output_dir: str, filename_template: str) -> str:
        """Write the log from now on, one line per event, starting with
        the start record, in place of a log at the same path. Returns its
        path. ``output_dir`` is an absolute directory."""
        path = self._log_path(output_dir, filename_template)
        with self._lock:
            path.write_text("", encoding="utf-8")
            self._streaming_path = path
            self._log = json_lines_logger(path, self._write_failed)
            self._log.info(self._start_record())
        return str(path)

    def _write_whole(self, path: Path, records: List[Dict[str, Any]]) -> bool:
        """Write ``records`` as the log at ``path`` into a file beside it,
        renamed over it once every line is written, so a failed write leaves
        a prior log as it was and removes the file beside it. Returns whether
        it was written. A line that cannot be written is reported as a
        warning naming ``path``. OSError when the file beside it cannot be
        made or renamed."""
        from testuary._progress import diagnostic_print

        lines = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
        lines.touch(exist_ok=False)
        failed: List[BaseException] = []
        try:
            writer = json_lines_logger(lines, failed.append)
            for record in records:
                writer.info(record)
            if failed:
                diagnostic_print(f"WARNING: the agent log {path} could not be written: {failed[0]}")
                return False
            lines.replace(path)
            return True
        finally:
            with contextlib.suppress(OSError):
                lines.unlink(missing_ok=True)

    def _log_path(self, output_dir: str, filename_template: str) -> Path:
        """The log file for ``filename_template`` in ``output_dir``, an
        absolute directory, which is created if needed."""
        log_dir = Path(output_dir)
        if not log_dir.is_absolute():
            raise ValueError(f"the log directory {output_dir} is not an absolute path")
        log_dir.mkdir(parents=True, exist_ok=True)
        return log_dir / Template(filename_template).substitute(project_name=self.project_name)

    def _start_record(self) -> Dict[str, Any]:
        from testuary.phase_state import clock_domain_id
        return {"record": "start", "project": self.project_name, "agent": self.agent_name,
                "clock_domain_id": clock_domain_id(), "start_t_mono_s": self._clock_start,
                "config": self.config}

    def _end_record(self) -> Dict[str, Any]:
        return {"record": "end", "end_t_mono_s": self._finished_timing["t_mono_s"],
                "end_clock_domain_id": self._finished_timing["clock_domain_id"]}

    def _write_failed(self, exc: BaseException) -> None:
        from testuary._progress import diagnostic_print

        diagnostic_print(f"WARNING: the agent log {self._streaming_path} could not be written: {exc}")

    def _stream(self, record: Dict[str, Any]) -> None:
        if self._log is not None:
            self._log.info(record)

    def _format_elapsed(self) -> str:
        from testuary.phase_state import elapsed_label
        return elapsed_label(self._clock_start)

    def log(self, message: str, level: int = 0, color: Optional[str] = None, *,
            echo: bool = False, **fields: Any) -> None:
        """Log to console + internal buffer. ``level`` indents 2 spaces each.
        ``echo`` keeps an outcome line on the console while a display runs.
        ``fields`` go into the line's console record, after its message."""
        from testuary._progress import (
            console_text, diagnostic_print, display_owns_timestamps, echoes_log_line,
            write_plain_log,
        )
        ts_str = self._format_elapsed()
        indent = "  " * level
        ts_color = color or _Colors.BRIGHT_GREEN
        self._record(level, message, **fields)
        write_plain_log(message, level)
        if echoes_log_line(message, echo=echo):
            if display_owns_timestamps():
                diagnostic_print(console_text(f"{indent}{message}"), flush=True, logged=True)
            else:
                diagnostic_print(console_text(f"{ts_color}{ts_str}{_Colors.RESET} {indent}{message}"),
                                 flush=True, logged=True)

    def debug(self, message: str, level: int = 0) -> None:
        """Record a detail entry in the saved log without printing it, so the
        console stays scannable while the streamed JSON keeps the full trail
        (measurement internals, decision reasons, intermediate state). The
        entries share the console timeline, in order, marked
        ``"debug": True`` for filtering."""
        self._record(level, message, debug=True)

    def _record(self, level: int, message: str, **extra: Any) -> None:
        """Append one console-log entry (``level``, ``message``, ``extra``,
        then the timing fields) and stream the log."""
        from testuary.phase_state import timing_fields
        with self._lock:
            entry = {"level": level, "message": message, **extra, **timing_fields()}
            self._console_logs.append(entry)
            self._stream({"record": "console", **entry})

    def extract_messages(self, messages: List[Any]) -> List[dict]:
        """Render a LangChain conversation as JSON-loggable dicts, in order,
        with ``reasoning_field_names``. A message that fails to render becomes
        an ``unknown`` entry.
        """
        if not messages:
            return []
        # Lazy import: keep AgentLogger usable from non-LG callers.
        from testuary._chat_log import _extract_message_for_log

        extracted = []
        for msg in messages:
            try:
                rendered = _extract_message_for_log(msg, reasoning_field_names=self.reasoning_field_names)
            except Exception:
                rendered = {"type": "unknown", "content": str(msg)[:200]}
            extracted.append(rendered)
        return extracted

    def log_analysis(
        self,
        file_path: str,
        duration_ms: float,
        messages: Optional[List[Any]] = None,
        error: Optional[str] = None,
    ) -> None:
        """Log a finished agent run of ``file_path`` with its conversation."""
        # Lazy import: avoid circular dep with project.
        try:
            from testuary import project
            stored_path = project.relativize_path(file_path)
        except Exception:
            stored_path = file_path
        from testuary.phase_state import timing_fields
        entry = {
            "project": self.project_name,
            "file_path": stored_path,
            "file_name": Path(file_path).name,
            "duration_ms": round(duration_ms, 2),
            "error": error,
            "conversation": self.extract_messages(messages) if messages else None,
            **timing_fields(),
        }
        with self._lock:
            self._analyses.append(entry)
            self._stream({"record": "analysis", **entry})

    def save(self, output_dir: str, filename_template: str) -> str:
        """End the log with its end record. Returns its path, or "" when
        nothing was logged. A log streamed elsewhere, or not at all, is
        written whole at the given path."""
        if not self._console_logs and not self._analyses:
            return ""
        log_path = self._log_path(output_dir, filename_template)
        from testuary.phase_state import timing_fields
        with self._lock:
            self._finished_timing = timing_fields()
            if self._log is not None and log_path == self._streaming_path:
                self._log.info(self._end_record())
            else:
                self._write_whole(log_path, [
                    self._start_record(),
                    *({"record": "console", **entry} for entry in self._console_logs),
                    *({"record": "analysis", **entry} for entry in self._analyses),
                    self._end_record()])
        return str(log_path)


def log_line(logger: Any, message: str, level: int = 0, *, echo: bool = False,
             unlogged: str = "printed", **fields: Any) -> None:
    """``logger.log``, or before a logger exists the console policy
    ``unlogged``: "echoed" prints only a line the console shows anyway
    (``echoes_log_line``), "printed" prints every line as a diagnostic (which
    also goes to the plain log), and "plain" writes the line with its level
    to the plain log and prints it. ``echo`` keeps an outcome line on the
    console while a display runs. ``fields`` go into the line's record in
    the logger's log. Without a logger there is no record."""
    if logger is not None:
        if echo:
            logger.log(message, level=level, echo=True, **fields)
        else:
            logger.log(message, level=level, **fields)
        return
    from testuary._progress import diagnostic_print, echoes_log_line, write_plain_log

    if unlogged == "plain":
        write_plain_log(message, level)
        diagnostic_print(message, logged=True)
    elif unlogged == "printed" or echoes_log_line(message, echo=echo):
        diagnostic_print(message)


class PhaseLogger:
    """A module's ``AgentLogger`` lifecycle: each agent module's, finishing's
    and the run's main log. A module holds one instance, which exists before
    any agent, and exposes its ``init`` / ``log`` / ``save`` as the
    module-level ``init_logging`` / ``log_print`` / ``save_logs`` the
    orchestrator drives. ``unlogged`` is ``log_line``'s console policy for a
    line logged before ``init``. ``CodingAgent.log`` routes to the logger handed to
    the agent at construction."""

    def __init__(self, agent_name: str, *, unlogged: str = "echoed") -> None:
        self.agent_name = agent_name
        # The phase's settings for the log's start record, which the phase
        # sets before init (its model and loop attempts).
        self._config: Dict[str, Any] = {}
        self.unlogged = unlogged
        self.logger: Optional[Any] = None
        # Public so a phase with extra logging can resolve the same
        # directory the save lands in.
        self.log_dir_override: Optional[str] = None
        self.log_filename_override: Optional[str] = None

    def init(self, project_name: str, log_dir: str, log_filename: str) -> None:
        self.log_dir_override = log_dir
        self.log_filename_override = log_filename
        self.logger = AgentLogger(project_name=project_name, agent_name=self.agent_name,
                                  config=self._config)
        self.logger.init()
        self.logger.enable_streaming(log_dir, log_filename)

    def log(self, message: str, level: int = 0, *, echo: bool = False, **fields: Any) -> None:
        log_line(self.logger, message, level, echo=echo, unlogged=self.unlogged, **fields)

    def debug(self, message: str, level: int = 0) -> None:
        """Record a detail entry in the streamed log only, never on the
        console. Before ``init`` there is no log, so it is dropped."""
        if self.logger is not None:
            self.logger.debug(message, level=level)

    def save(self) -> str:
        if self.logger is None:
            return ""
        return self.logger.save(output_dir=self.log_dir_override,
                                filename_template=self.log_filename_override)


def run_summary_line(result: str) -> str:
    """The line of a run_tests result that states its outcome: the
    ``TESTS:`` summary, which notes such as ``[STANDS ALONE, NOTE]``
    precede, else the first line."""
    lines = result.strip().splitlines()
    return next((ln for ln in lines if ln.startswith("TESTS: ")),
                lines[0] if lines else "")


def text_lines(text: str) -> List[str]:
    """``text``'s lines as the reads and the edit views number them: split at
    "\\n" only, so a form feed or a lone carriage return stays inside its
    line, and a final newline ends the last line. Empty text has no line."""
    lines = text.split("\n")
    if lines[-1] == "":
        lines.pop()
    return lines


def line_count(text: str) -> int:
    """The number of ``text_lines``."""
    return len(text_lines(text))


def suspect_names(suspects: Sequence[str]) -> str:
    """The tests named as suspects after a timeout, within the
    ``timeout_suspects`` limit."""
    if not suspects:
        return "(could not identify)"
    named, omitted = _limits.cut_items(suspects, _limits.value("timeout_suspects"),
                                       what="further tests")
    return ", ".join(named) + (f" {omitted}" if omitted else "")


def tests_added_since(backend: Any, current: str, baseline: str) -> List[str]:
    """The functions ``current`` defines that ``baseline`` does not, left
    out those nested in another function: the tests added since the suite
    last ran within its time limit, hence the likely suspects for a
    suite-level timeout. A function nested in a function is a helper the
    runner never collects, so it is no suspect. Functions come from the
    backend's per-function fingerprints (functions only, by qualified
    name) and the baseline's definitions from its skeleton.

    A backend without per-function fingerprints (no such operation, or
    none for a source its skeleton reads) gives its skeleton's added
    definitions instead, left out those nested in another added one, so a
    new test is never lost from the suspects. Returns [] on any trouble
    (a parse error yields [] on its own)."""
    try:
        defined = set(backend.skeleton(current))
        before = set(backend.skeleton(baseline))
        fingerprints = getattr(backend, "function_fingerprints", None)
        functions = set(fingerprints(current)) if callable(fingerprints) else set()
    except Exception:
        return []
    if not functions and defined:
        # Without fingerprints the added definitions stand for the
        # functions: a definition under another added one is its helper.
        functions = defined - before

    def nested(name: str) -> bool:
        parts = name.split(".")
        return any(".".join(parts[:i]) in functions for i in range(1, len(parts)))

    return sorted(name for name in functions - before if not nested(name))


# Loop breaker (any agent). Two counters: the cross-tool no-progress
# counter rises on every tool call and resets only on a successful edit,
# catching loops across tools that never change the file. The per-tool
# counter rises when a tool returns its cached result, catching single-tool
# loops. Each agent owns its LoopBreakerState with its own wording slots
# (file kind, edit and return actions). The loop breaker only nudges the
# agent: only the run's limits (``UsageTracker``) and the designed stop
# conditions end a run.


@dataclass
class CodingAgentSession:
    """Per-target stop, progress and loop-breaker state shared by coverage and mutation agents."""

    # Whether the latest run_tests passed: the classifier routes on it and
    # the output validator refuses a result while it is False.
    last_tests_passed: bool = False
    # Whether the latest green run of the working file carried a STANDS
    # ALONE note naming tests: the edit tool is then offered beside the
    # measurement, and a goal end waits until a green run carries none.
    stands_alone_noted: bool = False
    # That note, which the goal-wait reply repeats. "" without one.
    stands_alone_note: str = ""
    # Flips true on the agent's designated read (``designated_read_name``),
    # or on any read when it names none. An agent that forces an initial
    # read stays in the read-only initial mode until this is set, so the
    # file enters as a dedup-able read instead of prompt text.
    has_read: bool = False
    # Flips true when the measurement tool answers its first call, the
    # start's second step after the designated read. An agent that forces
    # it stays in the measure-only start mode until then.
    initial_measured: bool = False
    # Consecutive measurements without a metric gain. Reset on a gain. Past
    # max_measurements_without_improvement the phase ends.
    measurements_without_improvement: int = 0
    # Consecutive premature result submissions (validator rejections with no
    # new measurement since the last attempt). At the
    # ``max_final_answers_without_measurement`` cap the framework accepts
    # the return.
    early_submission_rejections: int = 0
    # Measurement count captured at the last validation, used to tell a new
    # measurement (real progress) from a re-submission. None before the
    # first submission.
    measures_at_last_submission: Optional[int] = None
    # Consecutive suite timeouts (a hung or too-slow suite, not a failing
    # run). Reset when the suite next completes in time.
    consecutive_timeouts: int = 0
    # Designed early-stop: a tool detects a goal / measurements without improvement / repeated-
    # timeout end, synthesizes the canonical result here plus a short reason,
    # and the entry point substitutes it for the model's output.
    early_stop_result: Optional[Any] = None
    early_stop_reason: Optional[str] = None
    # Per-step logging + output-truncation tracking, shared so every tool
    # and the progress logger read the same counters on every editing agent.
    session_tool_calls: int = 0
    # Real test executions (a cache-skipped run_tests call does not count). The
    # coverage CSV writer surfaces it as the "runs" metric.
    run_tests_calls: int = 0
    # The most recent run_tests payload, replayed by the cached-skip when the
    # suite is unchanged since the last real run. "" until the first run.
    last_run_tests_result: str = ""
    # The measuring tool ("coverage" or "mutation") and its reply, when a
    # measurement it refused set the red state: the suite passed its normal
    # run, so the replies while that reply is the latest result name the
    # tool, not a failing run_tests. Empty otherwise.
    refusal: Tuple[str, str] = ("", "")
    # Whether the suite changed since the last run_tests. True at start so the
    # first call always runs. Set False after a run, True on a write.
    # Coverage and mutation use it for cached-run checks.
    tests_changed_since_run_tests: bool = True
    # The run's usage tracker, stashed by run_phase so a measure tool can
    # record cumulative token totals on its history entries. None until the
    # run starts.
    usage_tracker_ref: Optional[Any] = None
    current_tool_name: Optional[str] = None
    last_tool_event_at: Optional[float] = None
    # Phase clock. The entry point arms it (``time.perf_counter()``) before
    # framework baseline preparation, so the stamped ``agent_duration_s``
    # spans preparation through terminal finalization. The model loop is
    # reported separately. None leaves the duration at the model-loop span.
    phase_clock_start: Optional[float] = None
    # Wall-clock seconds the framework spent preparing the baseline before
    # the model loop (the derive, or the fallback engine measurement).
    framework_prep_seconds: float = 0.0
    # The reverse-order diagnostic's runs in this session by the suite's
    # content hash, so an unchanged suite is not run again.
    delivered_verdicts: Dict[str, Any] = field(default_factory=dict)
    # Per-read-tool immediate-consecutive-call counters for the cached
    # short-read header. Keyed by tool name, cleared by any non-read tool.
    consecutive_reads_by_tool: Dict[str, int] = field(default_factory=dict)
    # The line ranges already read by each read tool in the current streak (no
    # edit / run between), keyed by tool name. A re-read of a range already in
    # the set returns the cached header, so paging a file and then re-reading a
    # section already seen is caught, not just a whole-file re-read. Cleared by
    # any non-read tool alongside the counter above.
    read_ranges_seen_by_tool: Dict[str, set] = field(default_factory=dict)
    # The hash of the bytes the last completed measurement (or the mutation
    # phase's derive) scored: a measurement counts for the working file only
    # while its bytes match, so an engine failure cannot pass an unmeasured
    # file.
    measured_suite_hash: Optional[str] = None
    # Measured suite execution time (ms) of the phase's last measured run
    # (the coverage run, the mutation phase's normal run).
    last_suite_exec_ms: Optional[float] = None
    writes_done: int = 0
    # The best passing snapshot (the recovery state, BestGreenState below).
    best_green: "BestGreenState" = field(default_factory=lambda: BestGreenState())
    # The session's loop guard, per session so two sessions never share
    # counters. Each agent's factory names its result. The forward
    # reference resolves at instantiation, after LoopBreakerState below.
    loop_breaker: "LoopBreakerState" = field(default_factory=lambda: LoopBreakerState())


@dataclass
class LoopBreakerState:
    """Per-agent loop-detection state. One instance per agent run."""

    # Cross-tool no-progress counter.
    calls_since_last_edit: int = 0
    no_progress_threshold: int = 8

    # Per-tool consecutive-cached-call counters.
    consecutive_cached_calls: Dict[str, int] = field(default_factory=dict)

    # The escalation messages' wording, the editing loops' by default. Each
    # loop names its result in ``return_action_text``.
    file_kind: str = "working file"
    edit_action_text: str = "edit with ``edit_tests`` to insert, replace, or delete a block"
    return_action_text: str = "return the result if exit conditions are met"


#: How the no-progress banner starts, which ``track_output`` puts before a
#: tool's answer, ending in "]" and a blank line.
NO_PROGRESS_BANNER = "[STUCK: "


def bump_no_progress(state: LoopBreakerState) -> str:
    """Cross-tool no-progress counter. Call from every tool right before
    returning. Returns an empty string until ``no_progress_threshold`` is
    hit, then returns a banner to prepend to the tool output.

    ``CodingAgent.progress_made`` resets it after a real (byte-changing)
    edit. Rejected writes increment but do not reset. The message is built
    from the state's wording slots.
    """
    state.calls_since_last_edit += 1
    if state.calls_since_last_edit < state.no_progress_threshold:
        return ""
    return (
        f"{NO_PROGRESS_BANNER}{state.calls_since_last_edit} tool calls since the last "
        f"successful edit. The {state.file_kind} on disk has not changed "
        f"since. Repeating a previous read, run, or measurement cannot "
        f"produce new information. Pick one:\n"
        f"  (a) {state.edit_action_text},\n"
        f"  (b) use a different currently offered tool for information you "
        f"have not read or measured,\n"
        f"  (c) {state.return_action_text}.]\n\n"
    )


def loop_breaker_check(
    state: LoopBreakerState, tool_key: str, cached_payload: str,
) -> str:
    """Count a cached call of ``tool_key`` and return the cached payload
    under a header that escalates at the third and sixth call. The header
    nudges the model to edit or switch tools and never tells it to end the
    run: only the run's limits and designed stop conditions end it.
    """
    state.consecutive_cached_calls[tool_key] = (
        state.consecutive_cached_calls.get(tool_key, 0) + 1
    )
    n = state.consecutive_cached_calls[tool_key]
    if n >= 6:
        header = (
            f"[STUCK: this is identical call #{n} to `{tool_key}` with NO "
            f"edit between calls. The framework can NEVER return a new value "
            f"for this tool until the {state.file_kind} on disk changes. STOP "
            f"calling `{tool_key}`. Required next action -- pick one:\n"
            f"  (a) {state.edit_action_text},\n"
            f"  (b) call a different tool you haven't called yet,\n"
            f"  (c) {state.return_action_text}.]"
        )
    elif n >= 3:
        header = (
            f"[REPEATED CALL #{n} to `{tool_key}` -- the {state.file_kind} is "
            f"byte-identical to the version that produced the cached output "
            f"below. You will keep getting this exact result until you edit "
            f"the file. Pick one: a targeted edit, a DIFFERENT currently "
            f"offered tool, or return the result if exit conditions are met.]"
        )
    else:
        header = (
            f"[no new result: `{tool_key}` was already called and the "
            f"{state.file_kind} hasn't changed since. The cached result "
            f"below is identical. Edit the file before re-calling, or pick "
            f"a different tool.]"
        )
    return header + "\n\n" + cached_payload


# Best passing snapshot, for agents that rewrite a test file. Each
# ``update_best_green`` call checks whether the on-disk content, with the
# caller's fresh scores, beats the earlier best, which is also written to
# the sibling dotfile ``.best_passing_<test-file-name>.bak``. A clean run
# restores from the in-memory snapshot. After a hard kill (SIGKILL) the
# dotfile is the only record of the best passing version, kept for
# diagnosis. The ``.bak`` suffix keeps it out of every test runner (pytest
# skips dotfiles, javac and Surefire take only ``*.java`` and
# ``*Test.java``), and the original name with its extension shows the
# language.
#
# ``green_snapshot_key`` ranks by mutation score, then line coverage. Only
# a strictly better score replaces the snapshot, so equal scores keep the
# earlier one: the loop times nothing and counts no tests, and runtime is
# the phase-end consolidation's criterion. An unmeasured mutation score
# sorts below any measurement (a measured low value tells more), an
# unmeasured coverage as 0.


@dataclass
class BestGreenState:
    """An agent's own snapshot state: created on entry, updated by
    ``update_best_green`` after a passing run and its measurement, and
    restored by ``restore_best_green`` at the end of the run or on recovery.
    """

    snapshot: Optional[Dict[str, Any]] = None
    consecutive_failed_runs: int = 0
    restored_from_green: bool = False
    # Restores that wrote the snapshot back to disk. A caller compares it
    # before and after a step to tell whether that step restored.
    restores: int = 0


def green_snapshot_key(
    snapshot: Optional[Dict[str, Any]],
) -> tuple:
    """Sort key for ranking snapshots by their scores, lexicographic,
    higher is better: ``mutation``, then ``coverage``. Equal keys keep the
    earlier snapshot (``update_best_green`` replaces it only on a strictly
    higher key). Runtime and test count are not part of it. An
    unmeasured mutation score sorts as -1, below any real measurement,
    and an unmeasured coverage as 0. A missing (``None``) snapshot sorts
    below every snapshot.
    """
    if snapshot is None:
        return (-2.0, -1.0)
    return (
        snapshot.get("mutation") if snapshot.get("mutation") is not None else -1.0,
        snapshot.get("coverage") if snapshot.get("coverage") is not None else 0.0,
    )


def update_best_green(
    state: BestGreenState,
    test_file_path: Path,
    *,
    coverage: Optional[float] = None,
    branch_coverage: Optional[float] = None,
    mutation: Optional[float] = None,
    test_count: Optional[int] = None,
    test_source_observation: Optional[Dict[str, Any]] = None,
    exec_time_ms: Optional[float] = None,
    persist_to_disk: bool = True,
    rank_key=None,
    log_fn=None,
    loop_state: Optional[Dict[str, Any]] = None,
) -> bool:
    """Take a new snapshot when the file at ``test_file_path``, with the given
    scores, ranks strictly above ``state.snapshot`` by ``rank_key`` (default
    the shared mutation-first ``green_snapshot_key``). True when it did.
    ``persist_to_disk`` (default on) also writes it to the sibling
    ``.best_passing_<test-file-name>.bak``. ``log_fn(msg, level=int)``, when
    given, gets the diagnostic line.

    Pass only scores measured fresh against the current on-disk content,
    others as None, so stale values are not credited (each agent checks
    freshness in its measurement tools). ``loop_state`` is the agent's
    record of the state these bytes left the loop in (whether they were
    measured and what that left), so a restore puts the loop back exactly
    there. A file with the snapshot's own bytes that ranks no higher takes
    no new snapshot, but its measurement fills the snapshot's missing
    scores and an unmeasured ``loop_state``.
    """
    if rank_key is None:
        rank_key = green_snapshot_key
    if not test_file_path.exists():
        return False
    try:
        content = test_file_path.read_text(encoding="utf-8")
    except Exception:
        return False

    candidate = {
        "content": content,
        "coverage": coverage,
        "branch_coverage": branch_coverage,
        "mutation": mutation,
        "test_count": test_count,
        "exec_time_ms": exec_time_ms,
        "snapshot_clock_s": time.monotonic(),
    }
    if test_source_observation is not None:
        candidate["test_source_observation"] = dict(test_source_observation)
    if loop_state is not None:
        candidate["loop_state"] = dict(loop_state)
    if rank_key(candidate) <= rank_key(state.snapshot):
        snapshot = state.snapshot
        if snapshot is not None and snapshot.get("content") == content:
            # The same bytes measured again: the measurement belongs to the
            # snapshot as much as to the live file.
            for name in ("coverage", "branch_coverage", "mutation"):
                if snapshot.get(name) is None and candidate[name] is not None:
                    snapshot[name] = candidate[name]
            measured = (loop_state or {}).get("measured")
            if measured and not (snapshot.get("loop_state") or {}).get("measured"):
                snapshot["loop_state"] = dict(loop_state)
        return False

    state.snapshot = candidate
    if log_fn is not None:
        # One decimal, so float noise (``84.30000000001``) does not clutter
        # either agent's log line.
        def _fmt(v: Optional[float]) -> str:
            return f"{v:.1f}" if v is not None else "n/a"
        log_fn(
            f"[snapshot] New best-green: "
            f"mutation={_fmt(mutation)}, "
            f"coverage={_fmt(coverage)}, "
            f"tests={test_count if test_count is not None else 'n/a'}",
            level=2,
        )
    if persist_to_disk:
        try:
            best_path = (
                test_file_path.parent / f".best_passing_{test_file_path.name}.bak"
            )
            best_path.write_text(content, encoding="utf-8")
        except Exception as exc:
            if log_fn is not None:
                log_fn(
                    f"[snapshot] Could not persist best-green to disk: "
                    f"{type(exc).__name__}: {exc}. In-memory snapshot still "
                    "updated. The on-disk write is the post-mortem copy.",
                    level=2,
                )
    return True


def restore_best_green(
    state: BestGreenState,
    test_file_path: Path,
    *,
    current_is_green: bool,
    live_candidate: Optional[Dict[str, Any]] = None,
    rank_key=None,
    force: bool = False,
    log_fn=None,
) -> bool:
    """Revert the file to ``state.snapshot`` unless the live file has earned
    its place. True when a restore happened.

    ``current_is_green`` is the caller's "the last run_tests passed" flag.
    A passing live file is kept outright without a ``live_candidate``. With
    one, only when its ``rank_key`` is strictly higher than the snapshot's,
    so a passing but unmeasured, lower or equally scored final edit is never
    delivered over the earlier best. ``force=True`` skips that gate: the
    caller measured the live file as a regression, so the snapshot wins even
    when the cached comparison values are stale.

    A restore writes the snapshot content to disk, sets
    ``state.restored_from_green`` and counts ``state.restores``. The caller
    re-syncs its own previous-measurement values to the snapshot's scores,
    so later reports match the saved file, and marks its state passing (the
    snapshot passed by construction).
    """
    if rank_key is None:
        rank_key = green_snapshot_key
    if state.snapshot is None:
        return False
    if current_is_green and not force and live_candidate is None:
        return False
    if not test_file_path.exists():
        return False
    snapshot_content = state.snapshot["content"]
    try:
        current_content = test_file_path.read_text(encoding="utf-8")
    except Exception:
        return False
    if current_content == snapshot_content:
        return False
    if (current_is_green and not force
            and rank_key(live_candidate) > rank_key(state.snapshot)):
        return False
    try:
        test_file_path.write_text(snapshot_content, encoding="utf-8")
    except Exception:
        return False
    state.restored_from_green = True
    state.restores += 1
    if log_fn is not None:
        snap = state.snapshot
        log_fn(
            f"[restore] Reverted test file to best-green snapshot "
            f"(end-of-run artifact was not the selected best-green snapshot). "
            f"Snapshot: mutation="
            f"{snap.get('mutation') if snap.get('mutation') is not None else 'n/a'}, "
            f"coverage="
            f"{snap.get('coverage') if snap.get('coverage') is not None else 'n/a'}, "
            f"tests={snap.get('test_count')}.",
            level=1,
        )
    return True


def regression_revert_decision(
    *, regression_count: int, is_regression: bool, n_restore: int,
) -> tuple[bool, int]:
    """The in-loop ``N_restore`` gate the editing agents share. Given the
    consecutive-regression count, whether the latest passing measurement is
    a regression (strictly worse than the best by the agent's comparator)
    and the ``n_restore`` threshold, returns ``(should_revert, new_count)``:

      - A non-regression breaks the streak: ``(False, 0)``. Equal or lateral
        measurements (a setup test that does not move the metric) end here,
        so setup is never reverted.
      - ``n_restore <= 0`` disables the in-loop revert: ``(False, count)``.
      - A regression raises the count. At ``n_restore`` it reverts and
        resets, ``(True, 0)``, else ``(False, new_count)``.

    The agent owns the comparator and the revert itself (which artifact,
    restored how). This owns only the counter arithmetic.
    """
    if not is_regression:
        return False, 0
    if n_restore <= 0:
        return False, regression_count
    new_count = regression_count + 1
    if new_count >= n_restore:
        return True, 0
    return False, new_count
