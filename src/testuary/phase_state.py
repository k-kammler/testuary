"""The resume checkpoints of one target.

A target's work is a sequence of resumable phases, ``starter``
(the starter phase), ``coverage``, ``mutation`` and
``finishing``. ``phase_state.json`` in the target's directory, the only
place that survives between runs, records which of them are verifiably
complete. Per phase:

1. The phase runs.
2. The caller decides its success from the pipeline's signals, not from a
   file existing.
3. On success the caller extracts its artifacts to disk.
4. Then ``mark_complete`` records the phase with each artifact's hash
   (``mark_started`` already hashed the target's canonical source). The
   checkpoint is written last, only on success, and atomically.

On resume ``validate`` reuses a phase only when its status is ``complete``,
its recorded ``target_file_path`` and canonical-source hash match the
current target and ``projects/<name>/<file>``, its execution context
(execution mode and, in a container, the container command and the
project's runtime) still matches, and every artifact exists with its
recorded hash. Otherwise the phase is redone. The settings a phase was made
under (model, prompts, limits) are compared by the resume preflight's
generation-condition check instead (``_generation_conditions``), except
that the ``finishing`` phase is also redone when its recorded finishing
settings changed. Since an artifact is on disk before its checkpoint exists
and the checkpoint is checked against it on read, a checkpoint never vouches for
a partial or stale artifact.
"""
from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Any, Dict, FrozenSet, List, Optional, Tuple, Union
import contextlib
import contextvars
import hashlib
import json
import tempfile


_CLOCK_DOMAIN_ENV = "TESTUARY_CLOCK_DOMAIN_ID"
#: The kernel's ID of the current boot, where the system has one.
_BOOT_ID = Path("/proc/sys/kernel/random/boot_id")
#: The clock of this process, for a system without a boot ID.
_PROCESS_CLOCK = f"{os.getpid()}-{uuid.uuid4().hex[:12]}"


def clock_domain_id() -> str:
    """Identify the monotonic clock used by the current process: the boot's
    where the system has a boot ID, else this process's, labeled with
    ``[main] device_name`` when the run sets it. The computer's own name
    never appears. TESTUARY_CLOCK_DOMAIN_ID replaces the whole label."""
    from testuary import config

    configured = os.environ.get(_CLOCK_DOMAIN_ENV, "").strip()
    if configured:
        return configured
    try:
        value = _BOOT_ID.read_text(encoding="ascii").strip()
    except OSError:
        value = ""
    device = config.device_name()
    label = f"{device}-" if device else ""
    if value:
        return f"boot-{label}{value}"
    return f"process-{label}{_PROCESS_CLOCK}"


def timing_fields() -> Dict[str, Union[str, float]]:
    """Return the current monotonic point and its comparison domain."""
    return {
        "t_mono_s": time.monotonic(),
        "clock_domain_id": clock_domain_id(),
    }


def elapsed_label(start_t_mono_s: float) -> str:
    """Render elapsed monotonic time as ``H:MM:SS.mmm`` for console logs."""
    elapsed = max(0.0, time.monotonic() - float(start_t_mono_s))
    hours, remainder = divmod(elapsed, 3600.0)
    minutes, seconds = divmod(remainder, 60.0)
    return f"{int(hours)}:{int(minutes):02d}:{seconds:06.3f}"


# The resumable phases, in dependency order. The caller's resume walk trusts
# a later phase only when every earlier one validates.
PHASES = ("starter", "coverage", "mutation", "finishing")

IN_PROGRESS = "in_progress"
COMPLETE = "complete"
# ``failed``: the phase ran to its end but its outcome was not usable (a red
# baseline, retries used up). ``in_progress``: the process stopped during
# the phase. ``validate`` redoes both, and ``failed`` carries a ``reason``
# that says how the phase ended.
FAILED = "failed"

#: Why a failed phase failed, where the phase knows it: a cause outside
#: the method (a provider refusal, the ceiling on requests sent again, a measurement
#: tool that produced no result) or the method's own outcome (no usable suite
#: within the limits). A failed record without the field is unjudged.
INFRASTRUCTURE_FAILURE = "infrastructure"
METHOD_FAILURE = "method"
FAILURE_KINDS = frozenset({INFRASTRUCTURE_FAILURE, METHOD_FAILURE})

_STATE_FILENAME = "phase_state.json"


def _named_timing(prefix: str) -> Dict[str, Any]:
    current = timing_fields()
    return {
        f"{prefix}_t_mono_s": current["t_mono_s"],
        f"{prefix}_clock_domain_id": current["clock_domain_id"],
    }


def state_path(target_dir: Path) -> Path:
    """The target's phase_state.json, which holds its checkpoints."""
    return Path(target_dir) / _STATE_FILENAME


def sha256_file(path) -> Optional[str]:
    """Hex SHA-256 of a file's bytes, or None when it cannot be read. None
    never equals a recorded hash, so a missing or unreadable artifact fails
    validation and its phase is redone.
    """
    p = Path(path)
    if not p.is_file():
        return None
    try:
        h = hashlib.sha256()
        with open(p, "rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _sha256_canonical_source(path) -> Optional[str]:
    """Hex SHA-256 of the target's canonical source. In container mode the
    source lies inside the container, so it is read through the executor.
    Artifacts lie in the host's run directory and are hashed directly."""
    direct = sha256_file(path)
    if direct is not None:
        return direct
    from testuary import config
    from testuary import project
    if config.project_exec_mode() != "container":
        return None
    return project.sha256_project_file(str(path))


def _environment_identity(target_file_path: Optional[str] = None) -> Optional[Dict[str, str]]:
    """The container a checkpoint was made in, from the settings that name
    it: the container command (TESTUARY_PROJECT_CONTAINER_RUNNER), the
    project's Python interpreter inside the container (python_executable),
    and the runtime the target's language describes for the container
    (``_described_runtime``) where it is not that interpreter, such as a
    JDK's java or a compiler. The container's own name is not part of it: a
    site usually starts a new container of the same setup for each job.
    None outside container mode."""
    from testuary import config
    if config.project_exec_mode() != "container":
        return None
    identity = {
        "runner": config.environment_value("TESTUARY_PROJECT_CONTAINER_RUNNER", "").strip() or "docker",
        "python_executable": str(config.get_project_config().get("python_executable") or "").strip(),
    }
    runtime = _described_runtime(target_file_path)
    if runtime and runtime != identity["python_executable"]:
        identity["runtime"] = runtime
    return identity


def _described_runtime(target_file_path: Optional[str]) -> str:
    """The runtime the adapter of ``target_file_path``'s language describes
    from the run's settings for the container (``describe_runtime``), or ""
    when the language, the run's settings or a runtime is not known, or the
    adapter refuses the selection (the context is then incomplete)."""
    from langadapter import language_for, registration
    from langadapter.errors import InvalidRequest, UnsupportedLanguage

    from testuary import config
    from testuary import project as project_module
    from testuary.languages import language_setting_values

    settings = config.active_settings()
    if settings is None or not target_file_path:
        return ""
    try:
        language = language_for(str(target_file_path))
    except UnsupportedLanguage:
        return ""
    describe = getattr(registration(language).factory, "describe_runtime", None)
    if describe is None:
        return ""
    project = config.get_project_config()
    try:
        choice = describe(str(project.get("source_dir") or ""),
                          language_setting_values(language, project, settings), container=True,
                          environment=settings.environment,
                          resolve_path=project_module.resolve_host_path)
    except InvalidRequest:
        return ""
    return str(getattr(choice.interpreter, "path", None) or "")


def _current_resume_context(target_file_path: Optional[str] = None) -> Dict[str, Any]:
    """The context a newly created checkpoint of ``target_file_path`` is
    safe to resume under."""
    from testuary import config
    return {
        "schema_version": 1,
        "project_exec": config.project_exec_mode(),
        "environment": _environment_identity(target_file_path),
    }


def _resume_context_complete(context: Any) -> bool:
    """Whether a schema 1 context contains every required identity. Any
    other context is incomplete."""
    if not isinstance(context, dict) or context.get("schema_version") != 1:
        return False
    mode = context.get("project_exec")
    if not mode:
        return False
    if mode != "container":
        return True
    environment = context.get("environment")
    if not isinstance(environment, dict):
        return False
    return (environment.get("runner") not in (None, "")
            and any(environment.get(key) not in (None, "")
                    for key in ("python_executable", "runtime")))


def load(target_dir: Path) -> Dict[str, Any]:
    """The target's phase records, or ``{}`` when they are absent or
    unreadable, which means nothing is complete and everything is redone.
    """
    p = state_path(target_dir)
    if not p.is_file():
        return {}
    try:
        with open(p, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else {}
    except (json.JSONDecodeError, OSError):
        return {}


def save(target_dir: Path, state: Dict[str, Any]) -> None:
    """Write the phase records atomically, through a temporary file in the same
    directory and ``os.replace``, so a crash never leaves a damaged file and
    the old one stays until the new one is complete.
    """
    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    target = state_path(target_dir)
    fd, tmp_name = tempfile.mkstemp(
        prefix=".phase_state.", suffix=".tmp", dir=str(target_dir)
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(state, fh, indent=2, sort_keys=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp_name, target)
    finally:
        if os.path.exists(tmp_name):
            try:
                os.unlink(tmp_name)
            except OSError:
                pass


def _ensure_shell(
    state: Dict[str, Any],
    *,
    target_number: int,
    stem: str,
    target_file_path: str,
) -> Dict[str, Any]:
    """``state`` with the target's identity fields. Each run block records
    its own path and canonical-source hash (``mark_started``)."""
    state.setdefault("target_number", target_number)
    state.setdefault("stem", stem)
    state["target_file_path"] = str(target_file_path)
    return state


def _run_key(run_idx: int) -> str:
    """A run's key in ``state[phase]``: its index as a string, since JSON
    object keys are strings."""
    return str(int(run_idx))


def _get_run_block(state: Dict[str, Any], phase: str, run_idx: int) -> Dict[str, Any]:
    """``state[phase][run_key]`` as a dict, made with its parents where
    they are missing. Readers use ``_read_run_block``."""
    block = state.get(phase)
    if not isinstance(block, dict):
        block = {}
        state[phase] = block
    run_block = block.get(_run_key(run_idx))
    if not isinstance(run_block, dict):
        run_block = {}
        block[_run_key(run_idx)] = run_block
    return run_block


def _read_run_block(
    state: Dict[str, Any], phase: str, run_idx: int,
) -> Optional[Dict[str, Any]]:
    """``state[phase][run_key]`` when present and well formed, else None."""
    block = state.get(phase)
    if not isinstance(block, dict):
        return None
    run_block = block.get(_run_key(run_idx))
    if not isinstance(run_block, dict):
        return None
    return run_block


def phase_metrics(
    state: Dict[str, Any], phase: str, run_idx: int = 0,
) -> Dict[str, Any]:
    """The ``metrics`` recorded for ``phase[run_idx]``, or ``{}`` without
    them. ``run_idx`` defaults to 0, the run whose artifacts a restore from
    disk uses.
    """
    run_block = _read_run_block(state, phase, run_idx)
    if run_block is None:
        return {}
    metrics = run_block.get("metrics")
    return metrics if isinstance(metrics, dict) else {}


def mark_started(
    target_dir: Path,
    phase: str,
    *,
    target_number: int,
    stem: str,
    target_file_path: str,
    canonical_source_path,
    run_idx: int = 0,
    generation_condition: Optional[Dict[str, Any]] = None,
) -> None:
    """Record ``phase[run_idx]=in_progress`` before its first tool call.
    A crash leaves it so, and ``validate`` then has the phase redone (the
    caller removes the partial artifact first). Each run has its own block,
    so a later run keeps an earlier one's completion record, with the target
    and the canonical source (``projects/<name>/<file>``, never the
    target's edited copy) it was made from.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    state = load(target_dir)
    state = _ensure_shell(
        state,
        target_number=target_number,
        stem=stem,
        target_file_path=target_file_path,
    )
    state.setdefault(phase, {})
    prior = state[phase].get(_run_key(run_idx))
    # A retry after a crash keeps the earlier attempts on record as discarded
    # work. Restarting a COMPLETE phase discards the artifact it described,
    # and its time describes that artifact, so the two go together.
    carried = []
    if isinstance(prior, dict) and prior.get("status") != COMPLETE:
        carried = [x for x in (prior.get("segments") or [])
                   if isinstance(x, dict)]
    state[phase][_run_key(run_idx)] = {
        "status": IN_PROGRESS,
        "target_file_path": str(target_file_path),
        "target_source_sha256": _sha256_canonical_source(canonical_source_path),
        "resume_context": _current_resume_context(target_file_path),
        "segments": carried,
        **_named_timing("started"),
    }
    if generation_condition is not None:
        state[phase][_run_key(run_idx)]["generation_condition"] = generation_condition
    save(target_dir, state)


#: One run's pipeline, as a segment key under the run's block. It is no
#: phase and has no checkpoint: it times the run's work
#: between its phases (its work tree, its restores, its CSV write), which is
#: its time less its phases'. Named apart from the ``target_number`` field, the
#: target's number.
SEGMENT_PIPELINE = "pipeline"

#: The whole target file in one invocation, as a segment key under the first
#: block: its runs and the work they share, before the first run and after
#: the last (resume planning, the first project copy, engine teardown). What
#: it holds beyond its runs' pipelines is the file-level figure
#: ``shared_seconds``, added to no run.
SEGMENT_FILE = "file"

#: What ``open_segment`` and its callers accept.
SEGMENT_KEYS = PHASES + (SEGMENT_PIPELINE, SEGMENT_FILE)

#: A segment's outcome. ``ran`` is the only one whose time produced the
#: artifact the block holds: a crashed attempt re-runs from scratch and an
#: abandoned one was never closed, so neither bought anything.
SEGMENT_RAN = "ran"
SEGMENT_CRASHED = "crashed"
SEGMENT_ABANDONED = "abandoned"
SEGMENT_RESTORED = "restored"
_SEGMENT_OPEN = "open"


#: What a phase's time went on, one operation each: a model round trip, a
#: mutation engine run, a test run, a merge, a cover solve. Their spans do
#: not overlap, so their sum is at most the segment's time and the
#: difference is the rest of the phase.
TIME_MODEL = "model"
TIME_MUTATION = "mutation"
TIME_COVERAGE = "coverage"
TIME_TEST = "test"
TIME_MERGE = "merge"
TIME_COVER = "cover"
#: Copying the project for a measurement, seeding and harvesting its
#: bytecode and removing the copy: work around a measurement, not of it.
TIME_PREPARATION = "preparation"
TIME_PARTS = (TIME_MODEL, TIME_MUTATION, TIME_COVERAGE, TIME_TEST,
              TIME_MERGE, TIME_COVER, TIME_PREPARATION)

#: The segment that takes attributed time now, per context, so concurrent
#: work each sees its own and work outside a segment records nothing.
_ACTIVE_SEGMENT: contextvars.ContextVar = contextvars.ContextVar(
    "testuary_active_segment", default=None)

#: Which parts have a span open in this context, so an operation reached
#: through another of the same kind is counted once.
_OPEN_PARTS: contextvars.ContextVar = contextvars.ContextVar(
    "testuary_open_time_parts", default=frozenset())


def discard_segment(phase: str) -> None:
    """Mark the open segment of ``phase`` as an attempt whose work is not
    kept, such as a failed phase whose artifact the next attempt replaces:
    it closes as ``crashed``, so its time counts for nothing the block
    holds. Nothing happens when no segment of ``phase`` is open."""
    segment = _ACTIVE_SEGMENT.get()
    if segment is not None and segment._args[1] == phase:
        segment.discarded = True


def record_time(part: str, seconds: float) -> None:
    """Attribute ``seconds`` of the open segment to ``part``, from the code
    that runs the operation. Nothing happens when no segment is open, so it
    is safe outside a phase."""
    if part not in TIME_PARTS:
        raise ValueError(f"unknown time part {part!r}")
    segment = _ACTIVE_SEGMENT.get()
    if segment is None or seconds is None:
        return
    try:
        value = float(seconds)
    except (TypeError, ValueError):
        return
    if value > 0:
        segment.parts[part] = segment.parts.get(part, 0.0) + value


class _TimedPart:
    """Time one operation and attribute it to a part on the open segment.

    Entered again for a part that already has a span open, it measures
    nothing: an engine's one-call entry point opens a session and runs inside
    it, so the outer span already holds the inner one's time and adding both
    would count it twice."""

    def __init__(self, part: str) -> None:
        self._part = part
        self._t0 = None
        self._token = None
        self._handed: Dict[str, float] = {}

    def hand_over(self, part: str, seconds: Optional[float]) -> None:
        """Attribute ``seconds`` of this span to ``part`` instead: work the
        operation reports it did besides its own, such as a mutation run's
        check of where the tests import the target from. A span reached
        through another of its kind hands over nothing: the outer span
        holds the time."""
        if part not in TIME_PARTS:
            raise ValueError(f"unknown time part {part!r}")
        if self._token is not None and seconds:
            self._handed[part] = self._handed.get(part, 0.0) + float(seconds)

    def __enter__(self) -> "_TimedPart":
        import time as _time
        open_parts = _OPEN_PARTS.get()
        if self._part in open_parts:
            return self
        self._token = _OPEN_PARTS.set(open_parts | {self._part})
        self._t0 = _time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        import time as _time
        if self._token is not None:
            elapsed = _time.perf_counter() - self._t0
            handed = sum(self._handed.values())
            # Never more than the span held.
            scale = min(1.0, elapsed / handed) if handed else 1.0
            record_time(self._part, elapsed - handed * scale)
            for part, seconds in self._handed.items():
                record_time(part, seconds * scale)
            _OPEN_PARTS.reset(self._token)
            self._token = None
        return False


def timed_part(part: str) -> _TimedPart:
    """Attribute the wrapped operation's elapsed time to ``part``.

    The time is recorded on the way out, so an operation that raises still
    reports what it spent before failing."""
    if part not in TIME_PARTS:
        raise ValueError(f"unknown time part {part!r}")
    return _TimedPart(part)


def timing_open() -> bool:
    """Whether a timed part's span is open in this context, so the time
    spent now is already attributed."""
    return bool(_OPEN_PARTS.get())


class _MeasuredPart:
    """Attribute to a part the time an operation reports it measured itself
    (``observed``), not the span around it: a measurement in an owned copy
    spends time on the copy too, which is a part of its own. Inside
    ``timing``, an operation that reports no time (or raises) is charged
    its elapsed time. A part whose span is already open in this context
    counts that time already."""

    def __init__(self, part: str) -> None:
        self._part = part
        self._seconds = None

    def observed(self, seconds: Optional[float]) -> None:
        if seconds is not None:
            self._seconds = seconds

    @contextlib.contextmanager
    def timing(self):
        import time as _time
        started = _time.perf_counter()
        try:
            yield self
        finally:
            if self._seconds is None:
                self._seconds = _time.perf_counter() - started

    def __enter__(self) -> "_MeasuredPart":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        if self._seconds is not None and self._part not in _OPEN_PARTS.get():
            record_time(self._part, self._seconds)
        return False


def measured_part(part: str) -> _MeasuredPart:
    """Attribute to ``part`` the time the wrapped operation reports through
    ``observed``."""
    if part not in TIME_PARTS:
        raise ValueError(f"unknown time part {part!r}")
    return _MeasuredPart(part)


def _segment_list(run_block: Dict[str, Any]) -> list:
    segments = run_block.get("segments")
    if not isinstance(segments, list):
        segments = []
        run_block["segments"] = segments
    return segments


def open_segment(
    target_dir: Path,
    phase: str,
    *,
    run_idx: int = 0,
    engine_workers: Optional[int] = None,
    suites_reused: bool = False,
) -> None:
    """Record that work on ``phase[run_idx]`` has begun.

    Written before the work rather than after it, because a process that is
    killed cannot close anything and the open record is what survives it.
    Any segment still open for this key belongs to such a run and is closed
    as ``abandoned`` here, so recovery needs no separate sweep and repeated
    kills cannot accumulate open segments."""
    if phase not in SEGMENT_KEYS:
        raise ValueError(f"unknown phase {phase!r}")
    state = load(target_dir)
    run_block = _get_run_block(state, phase, run_idx)
    segments = _segment_list(run_block)
    for seg in segments:
        if seg.get("outcome") == _SEGMENT_OPEN:
            seg["outcome"] = SEGMENT_ABANDONED
            # How far that run got is not knowable, and a guess from a file's
            # modification time would read as measured.
            seg["seconds"] = None
            seg.update(_named_timing("ended"))
    opened = timing_fields()
    segments.append({
        "seconds": None,
        "model_seconds": None,
        "outcome": _SEGMENT_OPEN,
        "engine_workers": engine_workers,
        "starter_suites_reused": bool(suites_reused),
        "start_t_mono_s": opened["t_mono_s"],
        "clock_domain_id": opened["clock_domain_id"],
    })
    save(target_dir, state)


def close_segment(
    target_dir: Path,
    phase: str,
    *,
    outcome: str,
    seconds: Optional[float],
    run_idx: int = 0,
    parts: Optional[Dict[str, float]] = None,
    discarded: bool = False,
) -> None:
    """Close this key's open segment with the time it took.

    ``outcome`` describes the segment, not the phase: whether the block ran to
    completion is this call's fact, whether the phase is complete stays
    ``mark_complete``'s. Keeping the two apart is what lets this write land
    before the checkpoint without contradicting it. ``discarded`` marks an
    attempt that finished but whose work is not kept (``discard_segment``):
    the run spent its time, which no phase output holds."""
    if phase not in SEGMENT_KEYS:
        raise ValueError(f"unknown phase {phase!r}")
    state = load(target_dir)
    run_block = _get_run_block(state, phase, run_idx)
    for seg in reversed(_segment_list(run_block)):
        if seg.get("outcome") == _SEGMENT_OPEN:
            seg["outcome"] = outcome
            if discarded:
                seg["discarded"] = True
            seg["seconds"] = (
                None if seconds is None else round(float(seconds), 3))
            model_seconds = (parts or {}).get(TIME_MODEL)
            if model_seconds is not None:
                seg["model_seconds"] = round(float(model_seconds), 3)
            if parts is not None:
                named = {k: round(float(v), 3)
                         for k, v in parts.items() if k in TIME_PARTS}
                seg["parts"] = named
                if seg["seconds"] is not None:
                    # The residual: time no part claimed, which also shows
                    # an operation that records no part.
                    seg["other_seconds"] = round(
                        max(0.0, seg["seconds"] - sum(named.values())), 3)
            seg.update(_named_timing("ended"))
            break
    save(target_dir, state)


def record_restored(
    target_dir: Path, phase: str, *, run_idx: int = 0,
) -> None:
    """Record that this run reused ``phase[run_idx]`` and did no work for
    it. The phase keeps the cost it had when it was produced, marked as not
    from this run."""
    if phase not in SEGMENT_KEYS:
        raise ValueError(f"unknown phase {phase!r}")
    state = load(target_dir)
    run_block = _get_run_block(state, phase, run_idx)
    point = timing_fields()
    _segment_list(run_block).append({
        "seconds": 0.0,
        "model_seconds": None, "outcome": SEGMENT_RESTORED,
        "engine_workers": None, "starter_suites_reused": False,
        "start_t_mono_s": point["t_mono_s"],
        "clock_domain_id": point["clock_domain_id"],
        "ended_t_mono_s": point["t_mono_s"],
        "ended_clock_domain_id": point["clock_domain_id"],
    })
    save(target_dir, state)


def output_seconds(
    target_dir: Path, phase: str, *, run_idx: int = 0,
) -> Optional[float]:
    """What this phase cost to produce the artifact it holds: its ``ran``
    segments only. None when nothing ran."""
    state = load(target_dir)
    run_block = _read_run_block(state, phase, run_idx)
    if run_block is None:
        return None
    total = [seg.get("seconds") for seg in _segment_list(run_block)
             if seg.get("outcome") == SEGMENT_RAN
             and isinstance(seg.get("seconds"), (int, float))]
    return round(sum(total), 3) if total else None


def output_time_parts(
    target_dir: Path, phase: str, *, run_idx: int = 0, discarded: bool = False,
) -> Dict[str, float]:
    """Where this phase's producing time went, part by part, over the same
    segments as ``output_seconds``, so a discarded attempt counts in
    neither. With ``discarded`` also over the attempts whose work was not
    kept, which spent the run's time (``target_time_parts``)."""
    totals: Dict[str, float] = {}
    for seg in _segment_list(_read_run_block(load(target_dir), phase, run_idx) or {}):
        if seg.get("outcome") != SEGMENT_RAN and not (discarded and seg.get("discarded")):
            continue
        for part, value in (seg.get("parts") or {}).items():
            if part in TIME_PARTS and isinstance(value, (int, float)):
                totals[part] = totals.get(part, 0.0) + float(value)
    return {part: round(value, 3) for part, value in totals.items()}


def target_time_parts(target_dir: Path, run_idx: int = 0) -> Dict[str, float]:
    """Run ``run_idx``'s producing time, part by part, with the rest of the
    run's own work under ``other``.

    The parts come from the run's phases, since a phase segment is inside the
    run's pipeline one and time is attributed to the innermost. ``other`` is
    the run's pipeline total minus all of them: its work tree, its restores,
    its CSV write and whatever else sits between its phases. It is present
    only when the run's pipeline was timed, so a run that recorded no pipeline
    segment reports no remainder rather than reporting zero. The work the
    file's runs share is in none of them (``shared_seconds``). An attempt
    whose work was not kept spent the run's time, so its parts stay here. No
    part when the run's time is not known (``unknown_times``), never a
    partial one."""
    if _run_time_unknown(load(target_dir), run_idx):
        return {}
    totals: Dict[str, float] = {}
    for phase in PHASES:
        for part, value in output_time_parts(target_dir, phase, run_idx=run_idx,
                                             discarded=True).items():
            totals[part] = totals.get(part, 0.0) + value
    parts = {part: round(value, 3) for part, value in totals.items()}
    total = target_seconds(target_dir, run_idx)
    if total is not None:
        parts["other"] = round(max(0.0, total - sum(totals.values())), 3)
    return parts


def target_seconds(target_dir: Path, run_idx: int = 0) -> Optional[float]:
    """Run ``run_idx``'s own wall-clock: its phases plus the work between them.
    None when the run holds reused work, a restored phase or a starter run
    again with its saved suites, and one of its pipelines was not counted
    (abandoned by a kill, or crashed): the work may come from that pipeline,
    whose time is unknown, so the total is missing, never partial."""
    if _reused_beside_uncounted(load(target_dir), run_idx, ("restored", "starter")):
        return None
    return output_seconds(target_dir, SEGMENT_PIPELINE, run_idx=run_idx)


def unknown_times(target_dir: Path, run_idx: int = 0) -> FrozenSet[str]:
    """The times of run ``run_idx`` that are not known, so a row shows them
    empty, never partial: ``target``, the run's total and where it went, when
    its pipeline was recorded but none of it counts (a crash) or the run holds
    work an uncounted pipeline left, and ``starter`` when the starter ran
    again with suites such a pipeline saved."""
    state = load(target_dir)
    unknown = set()
    if _run_time_unknown(state, run_idx):
        unknown.add("target")
    if _reused_beside_uncounted(state, run_idx, ("starter",)):
        unknown.add("starter")
    return frozenset(unknown)


def _run_time_unknown(state: Dict[str, Any], run_idx: int) -> bool:
    """Whether the run's pipeline was recorded but its time is not known:
    no segment of it counts, or it holds reused work beside an uncounted
    one."""
    segments = _segment_list(_read_run_block(state, SEGMENT_PIPELINE, run_idx) or {})
    return bool(segments) and (
        not any(seg.get("outcome") == SEGMENT_RAN for seg in segments)
        or _reused_beside_uncounted(state, run_idx, ("restored", "starter")))


def _reused_beside_uncounted(state: Dict[str, Any], run_idx: int, kinds: Tuple[str, ...]) -> bool:
    """Whether the run holds work that one of its uncounted pipelines
    (abandoned by a kill, or crashed) may have produced: a restored phase
    (``restored`` in ``kinds``) or a starter run again with its saved suites
    (``starter``)."""
    pipeline = _read_run_block(state, SEGMENT_PIPELINE, run_idx) or {}
    if not any(seg.get("outcome") in (SEGMENT_ABANDONED, SEGMENT_CRASHED)
               for seg in _segment_list(pipeline)):
        return False
    return any(("restored" in kinds and seg.get("outcome") == SEGMENT_RESTORED)
               or ("starter" in kinds and seg.get("starter_suites_reused"))
               for phase in PHASES
               for seg in _segment_list(_read_run_block(state, phase, run_idx) or {}))


def shared_seconds(target_dir: Path) -> Optional[float]:
    """The file-level figure: what the file's invocations spent outside its
    runs, the work its runs share, such as resume planning and the first copy
    of the project. Added to no run. For a file with one run it and the run's
    ``target_seconds`` add up to the file's time. None when no file segment was
    recorded."""
    total = output_seconds(target_dir, SEGMENT_FILE)
    if total is None:
        return None
    state = load(target_dir)
    runs = sum(output_seconds(target_dir, SEGMENT_PIPELINE, run_idx=run_idx) or 0.0
               for run_idx in _recorded_runs(state, SEGMENT_PIPELINE))
    return round(max(0.0, total - runs), 3)


def _recorded_runs(state: Dict[str, Any], phase: str) -> list:
    block = state.get(phase)
    if not isinstance(block, dict):
        return []
    runs = []
    for key in block:
        try:
            runs.append(int(key))
        except (TypeError, ValueError):
            continue
    return sorted(runs)


class _TimedSegment:
    """Open on entry, close on either exit path."""

    def __init__(self, target_dir, phase, run_idx, engine_workers,
                 suites_reused):
        self._args = (target_dir, phase, run_idx)
        self._engine_workers = engine_workers
        self._suites_reused = suites_reused
        self._t0 = None
        self._token = None
        #: Attributed time, filled by ``record_time``.
        self.parts: Dict[str, float] = {}
        #: Set by ``discard_segment``: the attempt's work is not kept.
        self.discarded = False

    def __enter__(self) -> "_TimedSegment":
        import time as _time
        target_dir, phase, run_idx = self._args
        open_segment(target_dir, phase, run_idx=run_idx,
                     engine_workers=self._engine_workers,
                     suites_reused=self._suites_reused)
        self._t0 = _time.perf_counter()
        self._token = _ACTIVE_SEGMENT.set(self)
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        import time as _time
        elapsed = _time.perf_counter() - self._t0
        if self._token is not None:
            _ACTIVE_SEGMENT.reset(self._token)
            self._token = None
        target_dir, phase, run_idx = self._args
        close_segment(
            target_dir, phase, run_idx=run_idx,
            outcome=SEGMENT_CRASHED if exc_type is not None or self.discarded else SEGMENT_RAN,
            seconds=elapsed,
            parts=self.parts,
            discarded=self.discarded,
        )
        return False


def timed_segment(
    target_dir: Path,
    phase: str,
    *,
    run_idx: int = 0,
    engine_workers: Optional[int] = None,
    suites_reused: bool = False,
) -> _TimedSegment:
    """Time one attempt at ``phase[run_idx]``.

    The elapsed time comes from a monotonic clock, so a clock adjustment
    mid-phase cannot move it."""
    return _TimedSegment(target_dir, phase, run_idx, engine_workers,
                         suites_reused)


def mark_complete(
    target_dir: Path,
    phase: str,
    *,
    artifacts: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, Any]] = None,
    run_idx: int = 0,
    generation_condition: Optional[Dict[str, Any]] = None,
) -> None:
    """Record ``phase[run_idx]=complete``, the phase's last write, after
    the caller extracted its artifacts to disk. ``artifacts`` maps a
    relative key (``tests/test_foo.py``) to the file's absolute path, and
    each is hashed now for ``validate`` to compare on resume. Finishing
    (``metrics``) has no file and passes ``artifacts=None``.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    state = load(target_dir)
    hashed: Dict[str, Optional[str]] = {}
    for rel, abspath in (artifacts or {}).items():
        hashed[rel] = sha256_file(abspath)
    run_block = _get_run_block(state, phase, run_idx)
    run_block.update(
        {
            "status": COMPLETE,
            "artifacts": hashed,
            "metrics": metrics or {},
            **_named_timing("completed"),
        }
    )
    if generation_condition is not None:
        run_block["generation_condition"] = generation_condition
    save(target_dir, state)


def repoint_artifact(
    target_dir: Path,
    phase: str,
    *,
    old_rel: str,
    new_rel: str,
    new_abspath,
    run_idx: int = 0,
) -> None:
    """Point a completed phase's artifact record at another file, hashed
    again. The mutation phase revises the delivered ``tests/`` file in place,
    and its consolidation keeps the inherited and the mutation sources under
    ``archive/mutation/pre_merge``, so the starter and coverage checkpoints move
    to the kept inherited copy. The coverage phase does the same for the
    starter checkpoint. A resume then validates those phases against their
    unchanged files, and a consolidated target resumes where it stopped.
    Nothing happens without a run block for ``run_idx``, and a block without
    an artifact record gets one."""
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    state = load(target_dir)
    block = state.get(phase, {}).get(_run_key(run_idx))
    if not block:
        return
    artifacts = block.get("artifacts") or {}
    artifacts.pop(old_rel, None)
    artifacts[new_rel] = sha256_file(new_abspath)
    block["artifacts"] = artifacts
    save(target_dir, state)


def mark_failed(
    target_dir: Path,
    phase: str,
    *,
    reason: str,
    run_idx: int = 0,
    failure_kind: Optional[str] = None,
) -> None:
    """Record ``phase[run_idx]=failed`` with its reason and, when the
    phase knows its cause, its ``failure_kind`` (``FAILURE_KINDS``): the
    phase ran to its end without a usable artifact (a red baseline, retries
    used up, a model error). A resume redoes it, and phase_state.json shows
    how it ended instead of ``in_progress``.
    """
    if phase not in PHASES:
        raise ValueError(f"unknown phase {phase!r}")
    if failure_kind is not None and failure_kind not in FAILURE_KINDS:
        raise ValueError(f"unknown failure kind {failure_kind!r}")
    state = load(target_dir)
    run_block = _get_run_block(state, phase, run_idx)
    run_block.pop("failure_kind", None)
    run_block.update(
        {
            "status": FAILED,
            "reason": (reason or "")[:500],
            **({"failure_kind": failure_kind} if failure_kind is not None else {}),
            **_named_timing("failed"),
        }
    )
    save(target_dir, state)


def running_phase(target_dir: Path, *, run_idx: int = 0) -> Optional[str]:
    """The phase whose ``run_idx`` block is still ``in_progress``, the
    last one in ``PHASES`` order when several are, else None: the phase
    a crash interrupted."""
    state = load(target_dir)
    running = [phase for phase in PHASES
               if (_read_run_block(state, phase, run_idx) or {}).get("status")
               == IN_PROGRESS]
    return running[-1] if running else None


def validate(
    state: Dict[str, Any],
    phase: str,
    *,
    target_dir: Path,
    target_file_path: str,
    canonical_source_path,
    run_idx: int = 0,
) -> bool:
    """Whether ``phase[run_idx]`` is verifiably complete and reusable,
    which needs every one of:

    * the run's block with the status ``complete``,
    * the recorded ``target_file_path`` and the current target equal,
    * the recorded canonical-source hash and the current
      ``projects/<name>/<file>`` equal (tests of a changed source are wrong
      to reuse, even when they pass),
    * the recorded context and the current execution mode equal, and in a
      container the container command and the project's runtime (the
      settings are the resume preflight's generation-condition check),
    * for the ``finishing`` phase, its recorded finishing settings and the
      current ones equal,
    * every recorded artifact present at ``target_dir/<rel>`` with its hash.

    A recorded hash of ``None`` never matches. The run's block holds the
    path and hash. The rule is ``rejection_reason``'s.
    """
    return (isinstance(state, dict) and _read_run_block(state, phase, run_idx) is not None
            and rejection_reason(state, phase, target_dir=target_dir, target_file_path=target_file_path,
                                 canonical_source_path=canonical_source_path, run_idx=run_idx) is None)


def _finishing_changed(phase: str, run_block: Dict[str, Any]) -> List[str]:
    """The finishing settings that differ from those the ``finishing`` phase was
    measured under (every one for a phase that recorded none), none for another
    phase. A resume then measures the delivered suites again instead of
    keeping numbers measured another way."""
    if phase != "finishing":
        return []
    from testuary.evidence import finishing_settings

    current = finishing_settings()
    recorded = (run_block.get("metrics") or {}).get("finishing_settings") or {}
    return sorted(key for key in current if recorded.get(key) != current[key])


def rejection_reason(
    state: Dict[str, Any],
    phase: str,
    *,
    target_dir: Path,
    target_file_path: str,
    canonical_source_path,
    run_idx: int = 0,
) -> Optional[str]:
    """Why ``validate`` refused this phase, in one sentence, or None: the
    resume planner says which recorded fact stopped matching, since a phase
    redone without a reason looks like a fault. None also for a phase
    without a record, which has nothing to reuse."""
    if not isinstance(state, dict):
        return "the phase state is unreadable"
    run_block = _read_run_block(state, phase, run_idx)
    if run_block is None:
        return None
    status = run_block.get("status")
    if status != COMPLETE:
        return f"the phase is recorded as {status or 'not started'}"
    if str(run_block.get("target_file_path")) != str(target_file_path):
        return (f"the target moved: recorded {run_block.get('target_file_path')!r}, "
                f"now {target_file_path!r}")
    recorded_source = run_block.get("target_source_sha256")
    if recorded_source is None:
        return ("the canonical source's hash is not recorded: the source could not be read "
                "when the phase started, or an earlier version recorded the phase")
    if recorded_source != _sha256_canonical_source(canonical_source_path):
        return "the canonical source changed since the phase ran"
    recorded = run_block.get("resume_context")
    current = _current_resume_context(target_file_path)
    if not _resume_context_complete(recorded):
        return "the recorded execution context is incomplete"
    if not _resume_context_complete(current):
        return "the current execution context is incomplete"
    differing = [k for k in set(recorded) | set(current)
                 if recorded.get(k) != current.get(k)]
    if differing:
        named = {
            "environment": "the container command or the project's runtime changed",
            "project_exec": "the execution mode changed",
        }
        return ", ".join(named.get(k, f"{k} changed")
                         for k in sorted(differing))
    changed = _finishing_changed(phase, run_block)
    if changed:
        return "Finishing's settings changed: " + ", ".join(changed)
    for rel, recorded_hash in (run_block.get("artifacts") or {}).items():
        if recorded_hash is None:
            return f"the artifact {rel} was recorded without a hash"
        if sha256_file(Path(target_dir) / rel) != recorded_hash:
            return f"the artifact {rel} is missing or changed"
    return None


def no_resume_skips() -> Dict[str, bool]:
    """A fresh "skip nothing" resume map keyed by every phase."""
    return dict.fromkeys(PHASES, False)


_RESUME_REQUIRED_CSV_FIELDS = (
    "starter_status",
    "starter_suites_generated",
    "starter_suites_passing",
    "starter_merged_tests",
    "starter_delivered_tests",
    "starter_requests",
    "starter_input_tokens",
    "starter_cache_hit_tokens",
    "starter_output_tokens",
    "coverage_merged_tests",
    "coverage_delivered_tests",
    "mutation_merged_tests",
    "mutation_delivered_tests",
    "line_coverage",
    "branch_coverage",
    "mutation_score",
)


def resume_csv_complete(target_results_dir: Path, expected_runs: int, target_number: Optional[int] = None,
                         *, run: Optional[int] = None) -> bool:
    """Return whether Finishing produced one structurally complete row per run.

    tests.csv is shared by every file of the run, so with ``target_number`` only that
    file's rows count. With ``run`` (counted from 0, as in the ``run``
    column) only that run's row counts, so a resume of one run does not
    depend on the others."""
    from testuary.results import _read_csv
    try:
        rows = _read_csv(target_results_dir.parent / "tests.csv")[1]
    except OSError:
        return False
    rows = [row for row in rows
            if _selected(row, "target_number", target_number) and _selected(row, "run", run)]
    if run is not None:
        expected_runs = 1
    by_run = {str(row.get("run", "")): row for row in rows}
    if len(rows) != expected_runs or len(by_run) != expected_runs:
        return False
    return all(_row_complete(target_results_dir, by_run.get(str(index)), int(index))
               for index in ((run,) if run is not None else range(expected_runs)))


def _selected(row: Dict[str, str], column: str, value: Optional[int]) -> bool:
    """Whether ``row`` is in the selection: ``value`` in ``column``, or any
    row without a value."""
    return value is None or str(row.get(column, "")).strip() == str(value)


def _row_complete(target_results_dir: Path, row: Optional[Dict[str, str]], run_idx: int) -> bool:
    """Whether a run's row holds every required cell, an empty one only
    where the run's own record holds no value for it."""
    return row is not None and not any(
        (row.get(field) is None or str(row[field]).strip() == "")
        and not _recorded_without_value(target_results_dir, row, field, run_idx)
        for field in _RESUME_REQUIRED_CSV_FIELDS)


def _recorded_without_value(target_results_dir: Path, row: Dict[str, str], field: str,
                            run_idx: int) -> bool:
    """Whether a required cell is empty because the run's own record holds no
    value for it, so the row is complete as it is: a target without
    branches (Finishing recorded no branch coverage), or a
    final mutation measurement that completed without a valid mutant. A
    resume measures neither again. Any other empty cell is missing."""
    if field == "branch_coverage":
        try:
            recorded = phase_metrics(
                load(target_results_dir), "finishing", run_idx=run_idx)
        except Exception:
            return False
        return "final_branch_coverage" in recorded and recorded["final_branch_coverage"] is None
    if field == "mutation_score":
        return (str(row.get("finishing_mutation_measurement") or "").strip() == "complete"
                and str(row.get("finishing_total") or "").strip() == "0")
    return False


def plan_test_resume(
    *,
    target_results_dir: Path,
    rel_target: Path,
    canonical_path: Path,
    run_idx: int,
    generation_skipped: Optional[set] = None,
    log: Optional[Any] = None,
    target_number: Optional[int] = None,
) -> Tuple[bool, Dict[str, bool], bool]:
    """Decide, from the target file's validated checkpoints, what the resume of
    run ``run_idx`` (0 for r0) skips. The run resumes from its own checkpoints
    and tests.csv row, so from what it completed itself.

    Returns ``(skip_target, resume_skips, restore_tests)``:

    * ``skip_target``: every phase validates and the run's tests.csv
      row is complete. The whole target is done, so it is skipped entirely
      (no project copy, no CSV writes).
    * ``resume_skips``: per phase
      (starter, coverage, mutation, finishing) booleans for the contiguous
      validated prefix. The walk stops at the first phase that
      does not validate, so a later checkpoint is never trusted past an
      earlier gap (the dependency chain).
    * ``restore_tests``: whether any phase is being skipped
      and so the durable test artifacts must be copied back into the
      target's copy before the orchestrator restores-from-disk.

    ``generation_skipped`` names phases this run skips by flag
    (a reused starter phase in a later run). A skipped phase writes no
    checkpoint, so the walk passes over it instead of treating the gap as
    the end of the validated prefix. The phase contributes no
    artifact the chain depends on.
    """
    state = load(target_results_dir)
    skipped = generation_skipped or set()

    # Validate the dependency-ordered prefix, excluding skipped units.
    # Mutation consolidates into the delivered coverage-named file.
    # Finishing follows generation.
    walk = [sp for sp in PHASES if sp not in skipped]
    prefix = _validated_prefix(state, walk, log, target_dir=target_results_dir,
                               target_file_path=rel_target.as_posix(),
                               canonical_source_path=canonical_path, run_idx=run_idx)

    all_tests_ok = len(prefix) == len(walk)
    if all_tests_ok and not resume_csv_complete(target_results_dir, 1, target_number, run=run_idx):
        # All expensive generation phases are reusable. Re-enter only
        # Finishing so the ordinary CSV writer reconstructs a missing or
        # partial row from the authoritative phase checkpoints.
        if prefix and prefix[-1] == "finishing":
            prefix.pop()
        all_tests_ok = False
        if log is not None:
            log(
                "[resume] every test-generation phase is complete, but this "
                "target's tests.csv row is missing or incomplete. Restoring "
                "the completed phases and measuring the suite again.",
                level=0,
            )
    skip_target = all_tests_ok

    resume_skips = no_resume_skips()
    restore_tests = False
    if prefix and not skip_target:
        # Phase names are the skip-flag keys.
        for sp in prefix:
            resume_skips[sp] = True
        restore_tests = True

    return skip_target, resume_skips, restore_tests


def _validated_prefix(state: Dict[str, Any], walk: List[str], log: Optional[Any],
                      **where: Any) -> List[str]:
    """The phases of ``walk`` that validate (``validate`` with ``where``), up
    to the first that does not. Redoing a finished phase is expensive, so
    ``log`` hears which recorded fact stopped matching there, rather than
    the operator inferring it from the work starting over."""
    from testuary import project

    prefix: List[str] = []
    for sp in walk:
        if validate(state, sp, **where):
            prefix.append(sp)
            continue
        reason = rejection_reason(state, sp, **where)
        if reason and log is not None:
            log(f"[resume] {sp} in repetition {project.run_name(where['run_idx'])} is not reusable: "
                f"{reason}. It and everything after it will run again.", level=0)
        break
    return prefix


def resume_generation_skipped(args, run_idx: int) -> set:
    """The phases run ``run_idx``'s resume walk passes over, since this
    invocation skips them by flag and the run recorded no checkpoint for them:
    in a later run a reused starter phase, which only the first run
    generated. A resume takes no other reuse (``check_run_options``)."""
    if run_idx > 0 and getattr(args, "reuse_starter", False):
        return {"starter"}
    return set()
