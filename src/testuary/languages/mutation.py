"""Testuary's mutation measurements through the language's public mutation
sessions, and its readings of kill matrices.

A session opens through the language's session attempts for Testuary's
mutation settings (tool caches under Testuary's output directory), in a
prepared copy of the project. Each run sends a ``MutationRequest`` under the
phase's mutation timer and reads the ``MutationRun`` it returns. A Ctrl+C
cancels the sessions in flight.

The readings of a run or a stored matrix give the counts the gates,
summaries and alarms use, the survivors' text for the model and the kill
matrix with Testuary's timing fields. The counts are the engine's before
the per-test journal attributed a timeout to a failing test. ``score`` is
the in-loop percentage, killed over valid mutants with an untested one
counted as not killed, 0.0 without one. The final score comes from the
kill matrix, which withholds it for an incomplete run. The matrix schema,
statuses, scores, counts and a subset's derivation are LangAdapter's
(``langadapter.mutation``), re-exported here.

The base ``MutationRunner`` is Cosmic Ray's: its named operator sets come
from ``langadapter.python.cosmic_ray`` and the pytest setup files a suite
loads from ``langadapter.python.project``. ``GenericMutationRunner``
(``languages.generic``) takes both from the language. Cleanup owners are
read from ``cleanup_handle``, from LangAdapter's ``recovery_owners`` and
from this layer's own markers on an exception, which this module sets
when a prepared copy fails.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, replace
from pathlib import Path
import threading
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple, Set, Union
from langadapter import ProjectContext, create, prepare_project, recovery_owners
from langadapter.mutation import BudgetedSuite, PerSuiteRequest, PerSuiteRun
from langadapter.mutation import Catalogue, MutationRequest, TimeoutRule
from langadapter.mutation import ATTRIBUTIONS, FAILED_INSTRUMENT_CHECK, FAILED_WORKERS, FROM_DEFAULT, MutationRun
from langadapter.mutation import MutationCleanupError as NativeMutationCleanupError
from langadapter.outcomes import Outcome
from langadapter.python.cosmic_ray import operator_set, operator_set_names
from langadapter.python.project import loaded_setup_files
from langadapter.suites import Suite
from testuary import phase_state, project
from testuary.languages import preparation
from testuary.languages.preparation import _PREPARED_MARKERS, _plan, _project_path, _suite
from langadapter.mutation import INCOMPLETE_KEY, DETECTED_STATUSES, KILLED_STATUS, NON_VIABLE_STATUS, SKIPPED_STATUS, SURVIVED_STATUS, TIMEOUT_STATUS, has_operators, killed_ids, killed_under, matrix_lower_bound_percent, matrix_score_percent, refresh_summary, restrict_kill_matrix, survivors_from_matrix, timeout_kill_count, unattributed_kill_fraction, unresolved_kill_count, upgrade_kill_matrix, valid_ids  # noqa: F401  (re-exported)


#: Cosmic Ray, the one Python mutation engine of this distribution.
ENGINE = "cosmic_ray"


class MutationInstrumentFault(RuntimeError):
    """A run the backend rejected because its evidence shows the per-mutant
    instrument gave no verdicts. A repeated model request repeats the broken
    instrument, so it is never retried as a transport failure.
    ``stopping_tests`` holds the id objects of the tests the run names, such
    as Python's origin refusal names them, for a place where no model is
    active."""

    stopping_tests: Tuple[Any, ...] = ()


def test_texts(tests: Any) -> Tuple[str, ...]:
    """The texts of ``tests`` as a reply and a record name them: a test id's
    ``text``, a text as it is."""
    return tuple(getattr(test, "text", str(test)) for test in tests)


class _LogRelay:
    """The session's log function: forwards each line to the logger its
    current caller set, so the progress lines of a pooled session reach the
    target that runs it."""

    def __init__(self, target: Optional[Callable[..., None]] = None) -> None:
        self.target = target

    def __call__(self, message: str, level: int = 1) -> None:
        if self.target is not None:
            self.target(message, level)


# ---- readings of a run -------------------------------------------------------

def summary_counts(rows) -> Dict[str, int]:
    """The counts a mutation summary shows, from the kill-matrix rows of a
    live run or a stored matrix alike. A timeout the journal attributed to a
    failing test (``killed`` with ``pre_attribution_status``) counts as a
    timeout, as the session recorded it. A skipped mutant counts nowhere."""
    counts = {"killed": 0, "timeout": 0, "non_viable": 0,
              "survived": 0, "untested": 0}
    for row in rows:
        status = row.get("status")
        if status == SKIPPED_STATUS:
            continue
        if row.get("pre_attribution_status") is not None or status == TIMEOUT_STATUS:
            counts["timeout"] += 1
        elif status == KILLED_STATUS:
            counts["killed"] += 1
        elif status == NON_VIABLE_STATUS:
            counts["non_viable"] += 1
        elif status == SURVIVED_STATUS:
            counts["survived"] += 1
        else:
            counts["untested"] += 1
    return counts


@dataclass(frozen=True)
class MutationMeasurement:
    """Testuary's reading of one mutation run. ``total`` counts the mutants
    with a result, ``truncated`` says the wall clock ended the run, ``run``
    is the backend's result (survivors and kill matrix), and ``reason`` the
    backend's reason when the run did not complete, "" otherwise."""

    killed: int
    survived: int
    timeout: int
    non_viable: int
    untested: int
    total: int
    score: float
    truncated: bool = False
    run: Optional[MutationRun] = None
    requested_target: Optional[str] = None
    reason: str = ""
    #: The engine tested no mutant because the suite failed without one.
    suite_failed: bool = False
    #: The tests that failed then, as the test id objects the run names,
    #: empty when it names none. Their texts are ``test_texts`` of them.
    stopping_tests: Tuple[Any, ...] = ()
    #: The score counts untested mutants as not killed (a partial run).
    lower_bound: bool = False

    @classmethod
    def empty(cls, *, truncated: bool = False, reason: str = "",
              suite_failed: bool = False,
              stopping_tests: Tuple[Any, ...] = ()) -> "MutationMeasurement":
        """A run that tested nothing, with the backend's reason when it
        gave one and the tests that stopped it."""
        return cls(0, 0, 0, 0, 0, 0, 0.0, truncated, reason=reason,
                   suite_failed=suite_failed, stopping_tests=tuple(stopping_tests))

    @classmethod
    def of(cls, run: MutationRun) -> "MutationMeasurement":
        """The reading of ``run``. A run the instrument check rejected, or
        whose worker raised for every mutant, raises
        ``MutationInstrumentFault``. Any other run that measured nothing
        (the tool did not initialize, its session could not be read) tested
        nothing, naming the tests that stopped it where the adapter's run
        gives them (``stopping_tests``, absent from older adapters)."""
        if not run.measured:
            stopping = tuple(getattr(run, "stopping_tests", ()) or ())
            if run.failure in (FAILED_INSTRUMENT_CHECK, FAILED_WORKERS):
                fault = MutationInstrumentFault(run.reason)
                fault.stopping_tests = stopping
                raise fault
            return cls.empty(reason=run.reason or str(run.outcome.value),
                             suite_failed=run.outcome is Outcome.TEST_FAILURES,
                             stopping_tests=stopping)
        counts = summary_counts(mutant.row for mutant in run.mutants)
        total = (counts["killed"] + counts["survived"] + counts["timeout"]
                 + counts["non_viable"])
        # An untested mutant counts as not killed, so the score is then a
        # lower bound and reaches a goal only if the suite truly does.
        progress = run.progress_score()
        lower_bound = progress is not None and bool(counts["untested"])
        return cls(
            total=total,
            score=progress if progress is not None else 0.0,
            truncated=run.outcome is Outcome.TIMED_OUT, run=run,
            reason="" if run.outcome is Outcome.COMPLETED else (run.reason or run.outcome.value),
            lower_bound=lower_bound, **counts)

    def survivors_text(self, max_mutants: Optional[int] = 10,
                       max_chars: Optional[int] = 240) -> str:
        """Up to ``max_mutants`` surviving mutants' changes for the model, each
        compacted to ``max_chars`` (None for no limit), by source line, led by
        the list of every survivor when there are more, or empty without
        survivors."""
        if self.survived == 0 or self.run is None:
            return ""
        survivors = [{"id": shown_mutant_id(m.row, m.engine_id), "line": m.key.line,
                      "diff": m.diff, "operator": m.key.operator}
                     for m in self.run.survivors()]
        return render_survivors(survivors, max_mutants=max_mutants, max_chars=max_chars)

    def timeout_rule(self) -> Optional[Dict[str, Any]]:
        """The per-mutant timeout the engine applied (``applied_timeout``),
        None without a run."""
        return applied_timeout(self.run)

    def kill_matrix(self) -> Optional[Dict[str, Any]]:
        """The run's kill matrix with Testuary's timing fields and the
        per-mutant timeout the engine applied (``per_mutant_timeout``), or
        None for a run that measured nothing."""
        if self.run is None or not self.run.measured:
            return None
        from testuary.phase_state import timing_fields
        fields = dict(timing_fields())
        applied = self.timeout_rule()
        if applied is not None:
            fields["per_mutant_timeout"] = applied
        payload = self.run.to_kill_matrix(fields)
        if self.requested_target is not None:
            payload["target_file"] = self.requested_target
        return payload


def applied_timeout(run: Any) -> Optional[Dict[str, Any]]:
    """The per-mutant timeout a run applied, as the engine states it (its
    effective ``per_mutant_timeout`` setting): ``value``, ``source``
    (``request``, or ``default`` for the engine's own rule) and the rule's
    ``inputs``. PIT's rule bounds each test of a mutant's run at ``factor``
    times its normal time plus ``constant_s``. None when the run states
    none."""
    setting = (getattr(run, "settings", None) or {}).get("per_mutant_timeout")
    return setting.to_dict() if setting is not None else None


def timeout_rule_text(applied: Optional[Dict[str, Any]]) -> Optional[str]:
    """``applied`` (``applied_timeout``) for a log line or an alarm, None
    without it. PIT bounds each test of a mutant, by its rule or by a fixed
    constant. The other engines bound a mutant's tests together."""
    if not applied:
        return None
    inputs = applied.get("inputs") or {}
    if inputs.get("rule") == "pit":
        return (f"{inputs.get('factor'):g} x each test's normal time + "
                f"{inputs.get('constant_s')}s per test (PIT's rule)")
    if inputs.get("rule") == "fixed" and "constant_s" in inputs:
        return f"{applied.get('value')}s per test"
    return f"{applied.get('value')}s per mutant"


def engine_timeout_rule(applied: Optional[Dict[str, Any]]) -> bool:
    """Whether the engine's own rule set the per-mutant timeout, rather
    than the timeout Testuary requested."""
    return bool(applied) and applied.get("source") == FROM_DEFAULT


def timeout_advice(applied: Optional[Dict[str, Any]]) -> str:
    """The setting that changes the per-mutant timeout ``applied`` names.
    The engine's own rule takes neither Testuary's factor nor its minimum,
    and only a fixed ``per_mutant_timeout_seconds`` replaces it."""
    if engine_timeout_rule(applied):
        return ("the engine applies its own rule, which "
                "per_mutant_timeout_factor and min_per_mutant_timeout_seconds "
                "do not change. A fixed per_mutant_timeout_seconds replaces it.")
    return ("raise per_mutant_timeout_factor or "
            "min_per_mutant_timeout_seconds, or fix per_mutant_timeout_seconds.")


def write_kill_matrix(measurement: MutationMeasurement, out_path: Path,
                      log: Optional[Callable[..., None]] = None) -> bool:
    """Write the measurement's kill matrix to ``out_path`` as JSON, or return
    False when the run has none. A matrix it cannot write returns False,
    with a warning to ``log``: it costs the matrix, never the run."""
    import json
    try:
        payload = measurement.kill_matrix()
        if payload is None:
            return False
        out = Path(out_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        if log is not None:
            log(f"WARNING: the mutation tool's per-test kill record could not be saved "
                f"({type(exc).__name__}: {exc})", 2)
        return False
    if log is not None:
        summary = payload.get("summary") or {}
        log(f"[kill-matrix] captured {summary.get('total')} mutants "
            f"(detected={summary.get('detected')}, "
            f"survived={summary.get('survived')}) -> {out.name}", 2)
    return True


def shown_mutant_id(row: Dict[str, Any], engine_id: Any = None) -> Any:
    """The id a mutant is shown and recorded by: the row's ``mutant_id``,
    the same for the same mutant in every run over the same source and
    catalogue (Cosmic Ray mints its engine id per session), else
    ``engine_id``, else the row's ``id``. Kill tokens keep the row's
    ``id``."""
    if row.get("mutant_id") is not None:
        return row["mutant_id"]
    return engine_id if engine_id is not None else row.get("id")


def _order_key(survivor: Dict[str, Any]) -> Tuple[Any, ...]:
    return (survivor.get("line") is not None, survivor.get("line") or 0,
            str(survivor.get("id")))


def spread_sample(survivors: Sequence[Dict[str, Any]], count: int) -> List[Dict[str, Any]]:
    """Up to ``count`` of ``survivors`` (``{id, line, operator, ...}``),
    spread over the target's lines and the operators, deterministically.
    Each round, from the top of the file, takes one mutant from each line
    with one left, or from lines evenly spaced over them when fewer places
    remain. On a line it takes the mutant whose operator the sample holds
    least often (then by operator and id)."""
    by_line: Dict[Any, List[Dict[str, Any]]] = {}
    for survivor in sorted(survivors, key=_order_key):
        by_line.setdefault(survivor.get("line"), []).append(survivor)
    chosen: List[Dict[str, Any]] = []
    shown_ops: Dict[Any, int] = {}
    while len(chosen) < count:
        lines = [line for line, pool in by_line.items() if pool]
        if not lines:
            break
        places = count - len(chosen)
        if places < len(lines):
            lines = [lines[(2 * i + 1) * len(lines) // (2 * places)] for i in range(places)]
        for line in lines:
            pool = by_line[line]
            pick = min(pool, key=lambda s: (shown_ops.get(s.get("operator"), 0),
                                            str(s.get("operator")), str(s.get("id"))))
            pool.remove(pick)
            chosen.append(pick)
            shown_ops[pick.get("operator")] = shown_ops.get(pick.get("operator"), 0) + 1
    return chosen


def render_survivors(survivors: Sequence[Dict[str, Any]], *, max_mutants: Optional[int],
                     max_chars: Optional[int]) -> str:
    """Surviving mutants (``{id, line, diff, operator}``) for the model, by
    source line and then id: ``--- mutant <id> (line <n>) ---`` and the
    compacted change, blank-line separated, for survivors with a change.
    Over ``max_mutants`` the changes are a sample (``spread_sample``), led
    by the complete id@line list and how to read the rest, so the model
    still sees every survivor. Empty without survivors."""
    from testuary.agent import _limits

    if not survivors:
        return ""
    ordered = sorted(survivors, key=_order_key)
    with_change = [s for s in ordered if s.get("diff")]
    shown = (with_change if max_mutants is None
             else sorted(spread_sample(with_change, max(int(max_mutants), 0)), key=_order_key))
    diffs = []
    for s in shown:
        line = s.get("line")
        line_tag = f" (line {line})" if line is not None else ""
        diffs.append(f"--- mutant {s.get('id')}{line_tag} ---\n"
                     f"{compact_diff(s.get('diff') or '', max_chars)}")
    body = "\n\n".join(diffs)
    if len(ordered) > len(shown):
        listed = ", ".join(
            f"{s.get('id')}@L{s.get('line')}" if s.get("line") is not None
            else str(s.get("id")) for s in ordered)
        head = f"All {len(ordered)} surviving mutants (id@line): {listed}\n\n"
        if shown:
            head += _limits.marker(
                len(ordered) - len(shown), "items", "further mutant changes",
                "read_source around the lines listed above and strengthen those tests") + "\n\n"
        body = head + body
    return body


def compact_diff(diff_text: str, max_chars: Optional[int] = 240) -> str:
    """The hunk headers and changed lines of a unified diff, within
    ``max_chars`` (None for no limit), the omission marked."""
    from testuary.agent import _limits
    if not diff_text:
        return ""
    lines: List[str] = []
    for line in diff_text.splitlines():
        if line.startswith("@@") or (
            (line.startswith("-") or line.startswith("+"))
            and not line.startswith("---")
            and not line.startswith("+++")
        ):
            lines.append(line)
    if not lines:
        lines = [ln for ln in diff_text.splitlines() if ln.strip()][:6]
    compact = "\n".join(lines).strip()
    return _limits.cut_text(compact, max_chars, what="the change", keep="head")


# ---- settings ----------------------------------------------------------------

def configured_catalogue(config: Dict[str, Any]) -> Catalogue:
    """The catalogue of the required ``cosmic_ray_operators`` setting: empty
    for the full catalogue, a named operator set, or a comma list of
    operators."""
    configured = str(config["cosmic_ray_operators"] or "").strip()
    if not configured:
        return Catalogue()
    if "," not in configured and configured in operator_set_names():
        return Catalogue(name=configured)
    return Catalogue(operators=tuple(
        op.strip() for op in configured.split(",") if op.strip()))


# ---- sessions ----------------------------------------------------------------

def mutation_cleanup_pending(error: BaseException) -> bool:
    """Whether a setup fallback would abandon a recorded cleanup owner: a
    native cleanup error or its public ``cleanup_handle``, the
    ``mutation_cleanup_handle`` a failed session close attaches, the owners
    LangAdapter's ``recovery_owners`` returns (a prepared copy, a command or
    an execution scope), or this layer's own markers on a prepared copy.
    They are recovery evidence, never proof that nothing runs, and no
    disposal is inferred from a PID. A launch that failed and left nothing
    running carries no owner: its measurement has no result, and the
    session goes on.
    """
    return (isinstance(error, NativeMutationCleanupError)
            # An interruption during a native cleanup keeps its owner here.
            or getattr(error, "cleanup_handle", None) is not None
            or getattr(error, "mutation_cleanup_handle", None) is not None
            or any(getattr(error, name, None) is not None for name in _PREPARED_MARKERS)
            or bool(recovery_owners(error)))


def _retain_session_cleanup(error: BaseException, cleanup: BaseException,
                            session: Any) -> None:
    error.mutation_cleanup_handle = session
    error.mutation_cleanup_error = cleanup
    if error is not cleanup and hasattr(error, "add_note"):
        with contextlib.suppress(BaseException):
            error.add_note("Mutation session cleanup remains unresolved: " + str(cleanup))


def close_mutation_session(session: Any, *, primary: Optional[BaseException] = None) -> None:
    """Close ``session``, any object with a no-argument ``close``. A close
    failure is raised, or, with ``primary``, attached to it without
    replacing it: the caller still raises ``primary``.
    """
    try:
        session.close()
    except BaseException as cleanup:
        _retain_session_cleanup(cleanup if primary is None else primary, cleanup, session)
        if primary is None:
            raise


class UndeclaredSuiteFolder(ValueError):
    """A Python mutation run of a suite in a directory outside the project that
    its session's caller did not declare when opening the session: the run
    would write into that directory."""


class _PreparedMutationLifetime:
    """One prepared copy of the project for a native session's whole life,
    not one per run. Declared suite directories outside the project are copied
    beside it, and each run's suite is staged there."""

    def __init__(self, backend, suite_dirs=()):
        self.original = backend.context
        self.language = backend.language
        self.copies_external_suite = preparation._fact(backend, "copies_external_suite")
        self.declared = tuple(dict.fromkeys(Path(suite_dir).resolve() for suite_dir in suite_dirs))
        self.owner = prepare_project(self.original, **_plan(self.original),
                                     **preparation.session_companions(backend, self.declared),
                                     **preparation.bytecode_store(backend))
        self.thread = threading.get_ident()
        self._use = self.owner.use()
        self._entered = False
        self._pending_session = None
        try:
            self.context = self._use.__enter__()
            self._entered = True
            self.backend = create(backend.language, self.context)
        except BaseException as error:
            error.prepared_project = self.owner
            if self._entered:
                self.release_use(primary=error)
            raise

    def request(self, request, *, suite=None, absent_files=()):
        self._require_declared(request.suite if suite is None else suite)
        target = _project_path(request.target, self.original, self.context)
        staged = self.owner.stage_suite(
            self.context, request.suite if suite is None else suite,
            protected_paths=(target,), absent_files=absent_files)
        return replace(request, target=target, suite=staged)

    def _require_declared(self, suite):
        """Refuse a suite outside the project whose directory the caller did not
        declare, for an adapter that copies such a directory beside the project
        (``copies_external_suite``): undeclared, it would run in place, in
        that directory."""
        if not self.copies_external_suite:
            return
        suite_dir = Path(suite.root).resolve()
        host = preparation._host_root(self.original)
        if host is not None and suite_dir.is_relative_to(host):
            return
        if any(suite_dir.is_relative_to(declared) for declared in self.declared):
            return
        named = ", ".join(str(declared) for declared in self.declared) or "none"
        raise UndeclaredSuiteFolder(
            f"the suite directory {suite_dir} lies outside the project and the directories this "
            f"mutation session declared ({named}): declare it when opening the session")

    def release_use(self, *, primary=None):
        if threading.get_ident() != self.thread:
            raise RuntimeError("Prepared mutation lifetime closes on its opening thread")
        if self._entered:
            try:
                self._use.__exit__(type(primary) if primary is not None else None,
                                   primary, primary.__traceback__ if primary is not None else None)
            except BaseException as cleanup:
                if primary is None:
                    raise
                primary.preparation_cleanup_error = cleanup
            finally:
                self._entered = False

    def close(self):
        if threading.get_ident() != self.thread:
            raise RuntimeError("Prepared mutation lifetime closes on its opening thread")
        if self._pending_session is not None:
            self._pending_session.close()
            self._pending_session = None
        self.release_use()
        self.owner.close()


#: Sessions with an operation in flight on a tool's thread, which a stop
#: in the main thread cancels (``cancel_on_interrupt``).
_IN_FLIGHT: "set[MutationSession]" = set()
#: Sessions cancelled and not yet closed. The command line closes them when
#: an interrupt ends the run, whatever close the interrupt skipped.
_CANCELLED: "set[MutationSession]" = set()
_IN_FLIGHT_LOCK = threading.Lock()

#: How long a close waits for a cancelled operation's own cleanup (processes
#: stopped, copies removed) before it reports the session as still open.
CANCEL_CLEANUP_WAIT_S = 120.0


#: The cancel scopes of the tool calls in progress (``tool_scope``), with
#: the thread each one runs on.
_SCOPES: "dict[Any, int]" = {}


@contextlib.contextmanager
def tool_scope() -> Iterator[None]:
    """One tool call, or the starter phase's checks of one response, under a
    LangAdapter cancel scope: a Ctrl+C stops every process the call started
    (a suite run, a coverage run, a measurement), and the call then raises
    ``OperationCancelled`` after the processes' cleanup, with no result."""
    from langadapter import CancelScope

    scope = CancelScope()
    with _IN_FLIGHT_LOCK:
        _SCOPES[scope] = threading.get_ident()
    try:
        with scope:
            yield
    finally:
        with _IN_FLIGHT_LOCK:
            _SCOPES.pop(scope, None)


def cancel_in_flight() -> bool:
    """Cancel every tool call and every session with an operation in
    flight. Each operation then raises the engine's cancellation after its
    own cleanup. True when one was in flight."""
    with _IN_FLIGHT_LOCK:
        sessions = list(_IN_FLIGHT)
        scopes = list(_SCOPES)
    for scope in scopes:
        scope.cancel()
    for session in sessions:
        session.cancel()
    return bool(sessions or scopes)


def _operation_in_flight() -> bool:
    with _IN_FLIGHT_LOCK:
        return bool(_IN_FLIGHT or _SCOPES)



def close_cancelled() -> List[BaseException]:
    """Close every cancelled session not yet closed, each after its
    operation's cleanup. Returns the failures of the closes that did not
    succeed."""
    with _IN_FLIGHT_LOCK:
        sessions = list(_CANCELLED)
    failures: List[BaseException] = []
    for session in sessions:
        try:
            session.close()
        except BaseException as error:
            failures.append(error)
    return failures


def _say_held(why: str) -> None:
    import sys

    print(f"Stopping: {why}, then the run ends.", file=sys.stderr, flush=True)


@contextlib.contextmanager
def _held_interrupts(deliver: bool = True):
    """Hold a Ctrl+C on the main thread until the block ends, then deliver
    it, unless the block ends by an exception or ``deliver`` is false (an
    interrupt is already under way): a close a Ctrl+C cut short would leave
    the session's processes and copies behind."""
    import signal

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    if previous is signal.SIG_IGN or previous is None:
        yield
        return
    held: List[int] = []

    def hold(signum, frame):
        if not held:
            _say_held("the mutation session closes first")
        held.append(signum)

    signal.signal(signal.SIGINT, hold)
    try:
        yield
    finally:
        signal.signal(signal.SIGINT, previous)
    if held and deliver:
        if callable(previous):
            previous(signal.SIGINT, None)
        else:
            raise KeyboardInterrupt


@contextlib.contextmanager
def cancel_on_interrupt():
    """While an agent's graph runs, a Ctrl+C first cancels the tool calls
    and measurements in flight on other threads, then interrupts as usual:
    a terminal's Ctrl+C does not reach the processes LangAdapter starts in
    their own sessions. A further Ctrl+C while a cancelled call cleans up
    cancels again without raising, since raising would cut the wait for the
    tool's thread and leave its processes and copies behind. Off the main
    thread, or where the interrupt is ignored, nothing changes."""
    import signal

    if threading.current_thread() is not threading.main_thread():
        yield
        return
    previous = signal.getsignal(signal.SIGINT)
    if previous is signal.SIG_IGN:
        yield
        return
    stopping: List[bool] = []

    def interrupt(signum, frame):
        if stopping and _operation_in_flight():
            cancel_in_flight()
            if len(stopping) == 1:
                stopping.append(True)
                _say_held("the work in flight ends and cleans up first")
            return
        if cancel_in_flight():
            stopping.append(True)
        # The handler in place decides how to interrupt (Python's default
        # raises KeyboardInterrupt). SIG_DFL would end the process unclean.
        if callable(previous):
            previous(signum, frame)
            return
        raise KeyboardInterrupt

    signal.signal(signal.SIGINT, interrupt)
    watcher = _SignalWatcher.start()
    try:
        yield
    finally:
        if watcher is not None:
            watcher.stop()
        signal.signal(signal.SIGINT, previous)


class _SignalWatcher:
    """Cancels the sessions in flight the moment a SIGINT arrives. Python
    runs a signal handler only when the main thread executes, and a main
    thread blocked in a lock (on Windows, waiting for a tool's thread) runs
    only after that thread ends. The interpreter writes the signal's number
    to the wakeup socket at once, and this thread reads it."""

    def __init__(self, reader, writer, previous_fd) -> None:
        self.reader, self.writer, self.previous_fd = reader, writer, previous_fd
        self.done = threading.Event()
        self.thread = threading.Thread(target=self._watch, name="testuary-cancel", daemon=True)

    @classmethod
    def start(cls) -> "Optional[_SignalWatcher]":
        import signal
        import socket

        reader, writer = socket.socketpair()
        writer.setblocking(False)
        try:
            previous_fd = signal.set_wakeup_fd(writer.fileno(), warn_on_full_buffer=False)
        except (ValueError, OSError):
            reader.close()
            writer.close()
            return None
        if previous_fd != -1:
            # Another user of the wakeup socket (an event loop) keeps it.
            signal.set_wakeup_fd(previous_fd)
            reader.close()
            writer.close()
            return None
        watcher = cls(reader, writer, previous_fd)
        watcher.thread.start()
        return watcher

    def _watch(self) -> None:
        import select
        import signal

        while not self.done.is_set():
            ready, _, _ = select.select([self.reader], [], [], 0.2)
            if not ready:
                continue
            try:
                received = self.reader.recv(64)
            except OSError:
                return
            if int(signal.SIGINT) in received:
                cancel_in_flight()

    def stop(self) -> None:
        import signal

        signal.set_wakeup_fd(self.previous_fd)
        self.done.set()
        self.thread.join(timeout=2)
        self.reader.close()
        self.writer.close()


class MutationSession:
    """One open mutation session of Testuary's project: ``workers`` and
    ``use_http`` state the fleet the session runs, for the phases' logs."""

    def __init__(self, runner: "MutationRunner", session: Any,
                 workers: int, use_http: bool) -> None:
        self.runner = runner
        self._session = session
        self.workers = workers
        self.use_http = use_http
        self._closed = False
        self._cleanup_error: Optional[BaseException] = None
        self._prepared: Optional[_PreparedMutationLifetime] = None
        self._native_closed = False
        self._operation = threading.Lock()
        self._worker_thread = None

    @contextlib.contextmanager
    def worker(self):
        """Scope one whole tool call to the session's prepared copy. A
        nested call on the same thread reuses the scope, and a competing
        call raises instead of waiting. No copy is made again.
        """
        if self._worker_thread == threading.get_ident():
            self._require_open()
            yield
            return
        if not self._operation.acquire(blocking=False):
            raise RuntimeError("Another mutation session operation is active")
        try:
            self._require_open()
            scope = (self._prepared.owner.worker(self._prepared.context)
                     if self._prepared is not None else contextlib.nullcontext())
            with scope:
                self._worker_thread = threading.get_ident()
                with _IN_FLIGHT_LOCK:
                    _IN_FLIGHT.add(self)
                try:
                    yield
                finally:
                    with _IN_FLIGHT_LOCK:
                        _IN_FLIGHT.discard(self)
                    self._worker_thread = None
        finally:
            self._operation.release()

    def _require_open(self):
        if self._closed:
            raise MutationSetupError("the mutation session is closed")
        if self._cleanup_error is not None:
            error = RuntimeError("mutation session cleanup must be retried before further work")
            _retain_session_cleanup(error, self._cleanup_error, self)
            raise error

    def run(self, *, target: Path, test_dir: Path, test_files: Sequence[str],
            per_mutant_timeout: int, wall_clock_timeout: int,
            operators: Optional[Sequence[str]] = None,
            attribution: Optional[str] = None,
            ) -> MutationMeasurement:
        """Measure ``test_files`` of ``test_dir`` against ``target`` with
        ``operators``, or the configured catalogue when None, and with
        ``attribution`` when given, else the runner's."""
        try:
            with self.worker():
                request = self.runner.request(
                    target=target, test_dir=test_dir, test_files=test_files,
                    per_mutant_timeout=per_mutant_timeout,
                    wall_clock_timeout=wall_clock_timeout, operators=operators,
                    attribution=attribution)
                asked = request
                if self._prepared is not None:
                    with phase_state.timed_part(phase_state.TIME_PREPARATION):
                        suite, absent = self.runner._prepared_suite_inputs(
                            request.suite, self._prepared)
                        request = self._prepared.request(request, suite=suite,
                                                         absent_files=absent)
                from testuary._progress import mutation_measurement

                # On this thread: the observed target is a context variable.
                with phase_state.timed_part(phase_state.TIME_MUTATION) as span, \
                        mutation_measurement() as forward:
                    if forward is not None:
                        request = replace(request, progress=forward)
                    run = self._session.run(request)
                    # The check of where the tests import the target from is
                    # preparation, not part of the measurement.
                    span.hand_over(phase_state.TIME_PREPARATION, run.origin_run_s)
                    if forward is not None:
                        # The row ends with the run's stated outcome, not with the return.
                        forward.report(run)
                measurement = MutationMeasurement.of(run)
                return (replace(measurement, requested_target=asked.target)
                        if self._prepared is not None else measurement)
        except BaseException as error:
            if mutation_cleanup_pending(error):
                self._cleanup_error = error
                _retain_session_cleanup(error, error, self)
            raise

    def per_suite(self, *, target: Path,
                  suites: Sequence[Tuple[str, Path, Sequence[str], int]],
                  wall_clock_timeout: int,
                  operators: Optional[Sequence[str]] = None) -> PerSuiteRun:
        """Each suite's verdict for every mutant of ``target``. Every entry of
        ``suites`` is ``(name, test_dir, test_files, per_mutant_timeout)``,
        and each suite's verdicts are those of the run ``run`` would make
        for it alone: the same request, staged the same way. Suites that
        would put a test file at the same place in the project copy are
        refused, since one would replace the other."""
        try:
            with self.worker():
                requests = [(name, self.runner.request(
                    target=target, test_dir=test_dir, test_files=test_files,
                    per_mutant_timeout=timeout, wall_clock_timeout=wall_clock_timeout,
                    operators=operators)) for name, test_dir, test_files, timeout in suites]
                if not requests:
                    raise ValueError("per_suite needs at least one suite")
                asked = requests[0][1].target
                if self._prepared is not None:
                    staged, placed = [], {}
                    for name, request in requests:
                        with phase_state.timed_part(phase_state.TIME_PREPARATION):
                            suite, absent = self.runner._prepared_suite_inputs(
                                request.suite, self._prepared)
                            request = self._prepared.request(request, suite=suite,
                                                             absent_files=absent)
                        for path in request.suite.files()[:len(request.suite.test_files)]:
                            other = placed.setdefault(Path(path).resolve(), name)
                            if other != name:
                                raise ValueError(
                                    f"suites {other} and {name} put a test file at the "
                                    f"same place in the project copy: {path}")
                        staged.append((name, request))
                    requests = staged
                first = requests[0][1]
                request = PerSuiteRequest(
                    target=first.target,
                    suites=tuple(BudgetedSuite(name, item.suite, item.per_mutant_timeout,
                                               item.passing_suite_s)
                                 for name, item in requests),
                    catalogue=first.catalogue, attribution=first.attribution,
                    wall_clock_limit_s=first.wall_clock_limit_s, workers=first.workers,
                    fail_on_unresolved=first.fail_on_unresolved,
                    timeout_journal=first.timeout_journal)
                from testuary._progress import mutation_measurement

                with phase_state.timed_part(phase_state.TIME_MUTATION) as span, \
                        mutation_measurement() as forward:
                    if forward is not None:
                        request = replace(request, progress=forward)
                    result = self._session.per_suite(request)
                    span.hand_over(phase_state.TIME_PREPARATION, sum(
                        run.origin_run_s or 0.0 for run in result.runs.values()))
                    if forward is not None:
                        forward.report(result)
                return replace(result, target=str(asked)) if self._prepared is not None else result
        except BaseException as error:
            if mutation_cleanup_pending(error):
                self._cleanup_error = error
                _retain_session_cleanup(error, error, self)
            raise

    def cancel(self) -> None:
        """Stop the engine's work in flight, from any thread: the operation
        raises the engine's cancellation after its own cleanup, and the
        session admits no new launch. The owner still closes the session.
        An engine without a cancel leaves the operation to end by itself."""
        cancel = getattr(self._session, "cancel", None)
        if cancel is not None:
            with _IN_FLIGHT_LOCK:
                if not self._closed:
                    _CANCELLED.add(self)
            cancel()

    def close(self, *, primary: Optional[BaseException] = None) -> None:
        """Close the session, which a later close leaves as it is. A close
        failure is raised, or, with ``primary``, attached to it. A failed
        close can be retried and admits no further measurement. A cancelled
        session's close first waits up to ``CANCEL_CLEANUP_WAIT_S`` for the
        operation in flight to end its cleanup. On the main thread a Ctrl+C
        during the close is held until it ends.
        """
        if self._closed:
            return
        with _held_interrupts(deliver=not isinstance(primary, KeyboardInterrupt)):
            self._close(primary)

    def _close(self, primary: Optional[BaseException]) -> None:
        acquired = False
        try:
            acquired = self._operation.acquire(blocking=False)
            with _IN_FLIGHT_LOCK:
                cancelled = self in _CANCELLED
            if (not acquired and cancelled
                    and self._worker_thread != threading.get_ident()):
                acquired = self._operation.acquire(timeout=CANCEL_CLEANUP_WAIT_S)
                if not acquired:
                    raise RuntimeError(
                        f"Cannot close while a mutation session operation is active: the "
                        f"cancelled operation did not end within {CANCEL_CLEANUP_WAIT_S:.0f} s")
            if not acquired:
                raise RuntimeError("Cannot close while a mutation session operation is active")
            if self._prepared is not None and threading.get_ident() != self._prepared.thread:
                raise RuntimeError("Prepared mutation lifetime closes on its opening thread")
            if not self._native_closed:
                with phase_state.timed_part(phase_state.TIME_MUTATION):
                    self._session.close()
                self._native_closed = True
            if self._prepared is not None:
                with phase_state.timed_part(phase_state.TIME_PREPARATION):
                    self._prepared.close()
        except BaseException as cleanup:
            self._cleanup_error = cleanup
            if (acquired and self._prepared is not None
                    and threading.get_ident() == self._prepared.thread):
                self._prepared.release_use(primary=cleanup if primary is None else primary)
            _retain_session_cleanup(cleanup if primary is None else primary, cleanup, self)
            if primary is None:
                raise
        else:
            self._closed = True
            self._cleanup_error = None
            with _IN_FLIGHT_LOCK:
                _CANCELLED.discard(self)
        finally:
            if acquired:
                self._operation.release()

    def __enter__(self) -> "MutationSession":
        if self._closed:
            raise MutationSetupError("the mutation session is closed")
        if self._cleanup_error is not None:
            error = RuntimeError("mutation session cleanup remains unresolved")
            _retain_session_cleanup(error, self._cleanup_error, self)
            raise error
        return self

    def __exit__(self, exc_type: Any, error: Any, traceback: Any) -> None:
        self.close(primary=error)


class MutationSetupError(RuntimeError):
    """The mutation session could not be opened. The message says why."""


def cosmic_ray_distributor(configured: Any, workers: int) -> Optional[str]:
    """The Cosmic Ray distributor for ``workers`` configured workers: the
    configured ``local`` or ``http``, or for ``auto`` (the default) http
    with more than one worker and local with one. None for a configured
    local with more than one worker, which would run one mutant at a time."""
    distributor = str(configured or "auto").strip().lower()
    if distributor == "auto":
        return "http" if workers > 1 else "local"
    if distributor == "local" and workers > 1:
        return None
    return distributor


class MutationRunner:
    """Testuary's mutation runs for one project: the configuration, the
    sessions and the operator catalogues."""

    name = ENGINE
    #: How a request records killing tests unless its run names it:
    #: "first", or "all" for the phase end's union measurements.
    attribution = "first"

    def __init__(self, backend: Any, config: Dict[str, Any], *,
                 log: Optional[Callable[..., None]] = None) -> None:
        self.backend = backend
        self.config = dict(config)
        self.log = _LogRelay(log)
        self._full: Optional[List[str]] = None
        self._full_loaded = False

    def engine_sets_timeout(self) -> bool:
        """Whether a request leaves the per-mutant timeout to the engine's own
        rule, so the timeout Testuary derives does not apply. Cosmic Ray
        takes the requested timeout."""
        return False

    def requested_timeout(self, per_mutant_timeout: int) -> Union[int, TimeoutRule, None]:
        """What a request carries as the per-mutant timeout: the derived
        timeout ``per_mutant_timeout`` in seconds, or None where the engine's
        own rule applies (``engine_sets_timeout``)."""
        return None if self.engine_sets_timeout() else int(per_mutant_timeout)

    def requested_timeout_text(self, per_mutant_timeout: int) -> str:
        """What a request asks for as the per-mutant timeout
        (``requested_timeout``), for the line logged before the run. The
        run's result states what was applied (``applied_timeout``)."""
        requested = self.requested_timeout(per_mutant_timeout)
        if requested is None:
            return f"the {self.name} engine's own rule"
        if isinstance(requested, TimeoutRule):
            cap = f", at most {requested.cap_s}s" if requested.cap_s is not None else ""
            return (f"{requested.factor:g} x the {self.name} engine's own run of the "
                    f"passing suite, at least {requested.minimum_s}s{cap}")
        return f"{requested}s"

    # ---- catalogues ----------------------------------------------------------

    def full_operators(self) -> Optional[List[str]]:
        """Every operator the engine registers in the project environment, or
        None when the catalogue cannot be read."""
        if not self._full_loaded:
            catalogue = self.backend.mutations.operators(self.name)
            self._full = (list(catalogue.operators)
                          if catalogue.operators is not None else None)
            self._full_loaded = True
        return list(self._full) if self._full is not None else None

    def catalogue(self) -> Catalogue:
        """The configured catalogue (``cosmic_ray_operators``)."""
        return configured_catalogue(self.config)

    def configured_operators(self) -> Optional[List[str]]:
        """The configured catalogue's operators, or None for the full one."""
        catalogue = self.catalogue()
        if catalogue.kind == "full":
            return None
        if catalogue.kind == "named":
            return list(operator_set(catalogue.name))
        return list(catalogue.operators)

    def request(self, *, target: Path, test_dir: Path,
                test_files: Sequence[str], per_mutant_timeout: int,
                wall_clock_timeout: int,
                operators: Optional[Sequence[str]] = None,
                attribution: Optional[str] = None,
                ) -> MutationRequest:
        """The request for one run, in this configuration: ``operators`` as
        listed (the full catalogue as ``full_operators`` lists it), or the
        configured catalogue when None. ``attribution`` ("first" or "all")
        overrides the runner's for this run: a pooled starter run
        records every killing test."""
        root = Path(test_dir)
        files = tuple(str(Path(f)) for f in test_files)
        catalogue = (Catalogue(operators=tuple(operators))
                     if operators is not None
                     else self.catalogue())
        if attribution is None:
            attribution = self.attribution
        elif attribution not in ATTRIBUTIONS:
            raise ValueError(f"attribution must be {' or '.join(ATTRIBUTIONS)}, "
                             f"not {attribution!r}")
        return MutationRequest(
            target=str(target),
            suite=Suite(str(root), files),
            catalogue=catalogue,
            per_mutant_timeout=self.requested_timeout(per_mutant_timeout),
            wall_clock_limit_s=int(wall_clock_timeout),
            attribution=attribution,
            fail_on_unresolved=False,
        )

    # ---- sessions ------------------------------------------------------------

    def _prepared_suite_inputs(self, suite, lifetime):
        """The suite with the pytest setup files Cosmic Ray loads beside it
        declared before staging, and those of the copy whose originals are
        gone."""
        current = loaded_setup_files(Path(suite.root), list(suite.files()[:len(suite.test_files)]))
        selected = replace(suite, setup_files=tuple(dict.fromkeys(
            (*suite.setup_files, *(str(path) for path in current)))))
        # A suite in a declared directory outside the project lies in the copy's
        # companion of that directory.
        copied = _suite(suite, lifetime.original, lifetime.context, lifetime.owner)
        previous = loaded_setup_files(Path(copied.root), list(copied.files()[:len(copied.test_files)]))
        absent = []
        for path in previous:
            original = Path(suite.root) / path.relative_to(Path(copied.root))
            if not original.exists():
                absent.append(str(original))
        return selected, tuple(absent)

    def open(self, workers: int, *, suite_dirs: Sequence[Any]) -> MutationSession:
        """Open one owned project use for the native session's whole lifetime.
        ``suite_dirs`` are the directories the session's suites lie in: each
        one outside the project is copied beside the project's copy, and a
        run of a suite outside the project and these directories is refused
        (``UndeclaredSuiteFolder``)."""
        if not isinstance(getattr(self.backend, "context", None), ProjectContext):
            # A backend without a project context (a replay) needs no copy.
            return self._open_backend(workers, self.backend)
        # The owned copy the session runs in, and its bytecode, are work around
        # the measurement, not part of it.
        with phase_state.timed_part(phase_state.TIME_PREPARATION):
            lifetime = _PreparedMutationLifetime(self.backend, suite_dirs)
        try:
            session = self._open_backend(workers, lifetime.backend)
        except BaseException as error:
            if mutation_cleanup_pending(error):
                lifetime._pending_session = getattr(error, "cleanup_handle", None)
                lifetime.release_use(primary=error)
                error.prepared_project = lifetime.owner
                _retain_session_cleanup(error, error, lifetime)
            else:
                try:
                    lifetime.close()
                except BaseException as cleanup:
                    error.prepared_project = lifetime.owner
                    error.preparation_cleanup_error = cleanup
                    _retain_session_cleanup(error, cleanup, lifetime)
            raise
        session._prepared = lifetime
        return session

    def _open_backend(self, workers: int, backend: Any) -> MutationSession:
        """The first session the language's attempts open, with the workers
        each records. Before a fallback attempt its warning is logged with
        the failed session's reason."""
        configured = int(workers or 1)
        try:
            attempts = backend.mutation_session_attempts(
                self.config, configured_workers=configured,
                effective_workers=configured,
                cache_base=str(project.output_dir()), log=self.log)
        except ValueError as refused:
            raise MutationSetupError(str(refused)) from None
        failed = None
        for attempt in attempts:
            if failed is not None and attempt.warning:
                self.log(attempt.warning.replace("{reason}", failed.reason or ""), 1)
            with phase_state.timed_part(phase_state.TIME_MUTATION):
                session = backend.mutations.open_session(self.name, **attempt.options)
            if session.outcome is Outcome.COMPLETED:
                return MutationSession(self, session, attempt.workers, attempt.parallel)
            with phase_state.timed_part(phase_state.TIME_MUTATION):
                session.close()
            failed = session
        raise MutationSetupError((failed.reason or str(failed.outcome)) if failed is not None
                                 else f"{self.name} offers no mutation session")

    @contextlib.contextmanager
    def session(self, workers: int, *,
                suite_dirs: Sequence[Any]) -> Iterator[MutationSession]:
        with self.open(workers, suite_dirs=suite_dirs) as session:
            yield session

    def run_once(self, *, workers: int, **run: Any) -> MutationMeasurement:
        """One run in a session of its own, which declares the run's suite
        directory."""
        with self.session(workers, suite_dirs=(run["test_dir"],)) as session:
            return session.run(**run)


def suite_test_files(backend: Any, test_dir: Path) -> List[str]:
    """The test files of a suite directory, as its green check names them:
    the files with the backend's file extensions that are not setup files."""

    root = Path(test_dir)
    setup = {Path(path).resolve()
             for path in backend.shared_setup_files(str(root))}
    return [str(path) for path in sorted(root.iterdir())
            if path.is_file() and path.suffix in backend.test_file_extensions()
            and path.resolve() not in setup]


#: A matrix whose kills are largely unattributed is worth a second look. The
#: threshold sits far above a clean run's few percent, and the floor keeps a
#: tiny all-breaking matrix from tripping it.
_UNATTRIBUTED_ALARM_FRACTION = 0.5
_UNATTRIBUTED_ALARM_MIN_KILLED = 8


def _killed_count(matrix: Optional[Dict[str, Any]]) -> int:
    """The matrix's killed mutant rows, the count the alarm texts name."""
    return sum(1 for m in (matrix or {}).get("mutants") or ()
               if m.get("status") in DETECTED_STATUSES)


def timeout_kill_alarm(matrix: Optional[Dict[str, Any]]) -> Optional[str]:
    """A warning when any kill came from the per-mutant clock, which depends
    on load, else None. It names the applied timeout the matrix records
    (``per_mutant_timeout``) and the setting that changes it. Diagnostic
    only: the score is the matrix's cr-rate either way."""

    n = timeout_kill_count(matrix)
    if n == 0:
        return None
    killed = _killed_count(matrix)
    applied = (matrix or {}).get("per_mutant_timeout")
    rule = timeout_rule_text(applied)
    return (
        f"[mutation] {n}/{killed} kills are per-mutant timeouts "
        f"(status timeout"
        + (f", timeout {rule}" if rule else "")
        + "): these verdicts came from the clock, not assertions, and can "
        f"flip under load. Consider re-measuring with a larger per-mutant "
        f"timeout to confirm the score: {timeout_advice(applied)}"
    )


def unattributed_kill_alarm(
    matrix: Optional[Dict[str, Any]],
) -> Optional[str]:
    """A warning when a matrix's kills are largely unattributed to a test
    (the sign of a broken baseline or a contaminated run), else None.
    Diagnostic only: it never changes a score."""
    frac = unattributed_kill_fraction(matrix)
    if frac is None or frac < _UNATTRIBUTED_ALARM_FRACTION:
        return None
    killed = _killed_count(matrix)
    if killed < _UNATTRIBUTED_ALARM_MIN_KILLED:
        return None
    unattributed = round(frac * killed)
    # Scaled before the division, as the backend's percentages are: 23 of 40
    # is 57.5%, which the fraction times 100 gives as 57.49999.
    return (
        f"[mutation] {unattributed}/{killed} kills ({100 * unattributed / killed:.0f}%) "
        f"have no named killer test. A mutant of status killed with no killer is a kill every "
        f"test triggers, so a score this heavily unattributed is worth reviewing "
        f"for a broken baseline or a contaminated measurement."
    )


def unresolved_kill_alarm(matrix: Optional[Dict[str, Any]]) -> Optional[str]:
    """A warning when a matrix holds kills whose verification gave no
    measurement evidence, neither confirmed nor refuted, else None. The
    remedy is re-measuring the suite. Diagnostic only: the score is the
    matrix's cr-rate either way."""
    n = unresolved_kill_count(matrix)
    if n == 0:
        return None
    killed = _killed_count(matrix)
    return (
        f"[mutation] {n}/{killed} kills are UNRESOLVED: recorded killed with "
        f"no attributed killer, and the verification re-run also died before "
        f"pytest could report. The score may contain environment-induced "
        f"kills; re-measure this suite on a quiet node to confirm."
    )


def render_survivors_from_matrix(
    matrix: Optional[Dict[str, Any]],
    *,
    operators: Optional[Set[str]] = None,
    max_mutants: Optional[int] = 10,
    max_chars: Optional[int] = 240,
) -> str:
    """The survivors of a stored matrix, rendered as a live measurement
    renders them (``render_survivors``), of ``operators`` when given, so
    the selective view's survivors come from the same full run. "" without
    survivors. It needs only the matrix, no engine session."""

    rows = {m.get("id"): m for m in (matrix or {}).get("mutants") or ()}
    survivors = [{**s, "id": shown_mutant_id(rows.get(s.get("id")) or {}, s.get("id")),
                  "operator": (rows.get(s.get("id")) or {}).get("operator")}
                 for s in survivors_from_matrix(matrix, operators=operators)]
    return render_survivors(survivors, max_mutants=max_mutants, max_chars=max_chars)


def matrix_counts(
    matrix: Optional[Dict[str, Any]],
    *,
    operators: Optional[Set[str]] = None,
) -> Dict[str, Any]:
    """The score and counts of one kill matrix, of its mutants of
    ``operators`` when given: ``mutation_score`` (percent, None without a
    valid mutant or for a matrix stamped incomplete),
    ``mutation_score_lower_bound`` (every untested mutant counted as not
    killed, also for a stamped matrix), and ``killed``, ``survived``,
    ``untested`` and ``total`` over the valid mutants. Nothing is run."""
    valid = valid_ids(matrix, operators=operators)
    killed = killed_ids(matrix, operators=operators) & valid
    survived = {item.get("id") for item in survivors_from_matrix(matrix, operators=operators)} & valid
    return {
        "mutation_score": matrix_score_percent(matrix, operators=operators),
        "mutation_score_lower_bound": matrix_lower_bound_percent(matrix, operators=operators),
        "killed": len(killed),
        "survived": len(survived),
        "untested": len(valid - killed - survived),
        "total": len(valid),
    }
