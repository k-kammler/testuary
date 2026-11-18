"""Finishing's measurements of the delivered suite, and the mutation phase
end's merge and selection of its suites.

``measure_combined`` measures target coverage, source and test complexity,
test counts, assertions and fixtures, with lint and suite timing optional.
``run_finishing`` adds the final mutation score and coverage. The test
phase calls both and attaches the per-phase measurements. The engine session
helpers also serve the phase ends' union and per-suite measurements. Nothing
here makes a model request."""
from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional, Any, Dict, List, Tuple, TYPE_CHECKING
from langadapter.mutation import (
    DETECTED_STATUSES, KILLED_STATUS, NON_VIABLE_STATUS, UNTESTED_MUTANTS_IN_CATALOGUE,
    UNTESTED_STATUS, VERDICTS)
from langadapter.outcomes import Outcome
from langadapter.sources import SourceObservation, SourceStatus, TestMetrics
from testuary import languages, normal_run, project
import math
import json
import os
import shutil
import tempfile
import time
from dataclasses import dataclass, field, asdict
from testuary import config as config_module
from testuary.languages import mutation
from testuary.agent._helpers import PhaseLogger
from testuary.generation_evidence import source_sha256
if TYPE_CHECKING:
    from testuary.consolidation import ReductionSettings


def observe_file(
    path, *, backend=None, read_text: Optional[Callable[[str], str]] = None,
) -> SourceObservation[TestMetrics]:
    """The source observation of a host file, or of a target read via ``read_text``."""
    if path is None:
        return SourceObservation(
            SourceStatus.INFRASTRUCTURE_FAILURE, reason="no delivered source file",
        )
    try:
        text = (read_text(str(path)) if read_text is not None
                else Path(path).read_text(encoding="utf-8"))
    except (OSError, UnicodeError) as exc:
        return SourceObservation(
            SourceStatus.INFRASTRUCTURE_FAILURE,
            reason=f"source could not be read: {type(exc).__name__}: {exc}",
        )
    if backend is None:
        backend = languages.get_backend(languages.detect_language(str(path)))
    if backend is None:
        return SourceObservation(
            SourceStatus.UNSUPPORTED, reason="no language adapter for source",
        )
    try:
        return backend.test_metrics_observation(text)
    except NotImplementedError:
        return SourceObservation(
            SourceStatus.UNSUPPORTED, reason="backend does not provide source metrics",
        )


def record(observation: SourceObservation[TestMetrics]) -> dict:
    """The record to persist: status, reason, parser and counts. A measured
    zero stays distinct from an absent count (None)."""
    value = observation.value
    return {
        "status": observation.status.value,
        "reason": observation.reason,
        "parser": observation.parser,
        **{name: getattr(value, name) if value is not None else None
           for name in ("code_lines", "tests", "assertions", "fixtures",
                        "parametrized_cases")},
    }


def floor_pct(value: Optional[float], decimals: int = 1) -> Optional[float]:
    """``value`` truncated toward zero to ``decimals`` places, for the
    coverage and mutation percentages the phase agents give the model, the
    display shows and the minimum messages name: rounding shows 89.55 as
    89.6, read as 90% and the goal met, and the model stops short."""
    if value is None:
        return None
    factor = 10 ** decimals
    return math.floor(value * factor) / factor


def compute_file_complexity_stats(
    file_path,
) -> "tuple[Optional[int], Optional[int]]":
    """``(sum, max)`` of the McCabe complexity over every function lizard finds
    in the file (language from the file extension). ``(None, None)`` is not
    measured (the file missing, unreadable or unparseable, or lizard not
    importable), an empty CSV cell. ``(0, 0)`` is a file without functions, a
    real zero."""
    try:
        import lizard
    except ImportError:
        return None, None
    p = Path(file_path)
    if not p.exists() or not p.is_file():
        return None, None
    try:
        info = lizard.analyze_file(str(p))
    except Exception:
        return None, None
    ccs = [f.cyclomatic_complexity for f in info.function_list]
    if not ccs:
        return 0, 0
    return sum(ccs), max(ccs)


def compute_file_complexity(file_path) -> Optional[int]:
    """The sum of per-function McCabe complexity, or None when not measured."""
    total, _ = compute_file_complexity_stats(file_path)
    return total


def level_tolerant_log(fn):
    """``fn`` adapted to the mutation session's ``log(msg, level=...)`` calls:
    a callable without a ``level`` keyword or ``**kwargs`` (such as the
    one-argument lambdas callers pass) is wrapped to drop the level, any
    other is returned unchanged."""
    if fn is None:
        return None
    import inspect
    try:
        params = inspect.signature(fn).parameters
        if "level" in params or any(
            p.kind == p.VAR_KEYWORD for p in params.values()
        ):
            return fn
    except (TypeError, ValueError):
        return fn
    return lambda msg, level=1: fn(msg)


_AGENT_CONFIG = config_module.PhaseConfiguration("test_metrics", parent="test_coverage")
# The raw section: the curated phase settings lack some measurement flags
# (measure_lint among them).
_RAW_METRICS_CONFIG = config_module.PhaseConfiguration("test_metrics", raw=True)
_TIMING_CONFIG = config_module.PhaseConfiguration("timing", raw=True)


def _bool_flag(config, key: str) -> bool:
    """Whether the switch ``key`` of ``config`` is on."""
    return str(config[key]).lower() in ("true", "1", "yes")

# The phases whose output suite has columns in tests.csv: "starter" (the
# merged starter suites), "coverage" and "mutation" (the merged suite).
_PHASE_ORIGINAL_PHASES = ("starter", "coverage", "mutation")

#: Attempts, with backoff, of the final un-mutated green check. "No tests
#: collected" (pytest rc 5) or a runner usage error (rc 4) from a suite that
#: holds tests is a transient seen only after a phase's cosmic-ray runs in the
#: full-pipeline container. It clears within tens of seconds, which the backoff
#: spans so the mutation score is not lost.
_GREEN_CHECK_ATTEMPTS = 5
_GREEN_CHECK_BACKOFF_S = (5, 10, 15, 20)


def phase_scores_mode() -> str:
    """Which recorded per-phase measurements the per-phase columns report:
    ``full`` (coverage and full mutation), ``coverage`` or ``off`` (snapshots
    only, no columns). ``target_phases._attach_phase_scores`` reads the
    archived outputs and measures nothing again. The run's ``[finishing]
    phase_scores`` applies, ``full`` by default, and an unknown value reads
    as ``full``."""
    val = str(config_module.load_config("test_metrics").get("phase_scores", "full"))
    val = val.strip().lower()
    return val if val in ("full", "coverage", "off") else "full"


def phase_original_path(originals_dir: Path, base_stem: str,
                        suffix: str, phase: str) -> Path:
    """The immutable snapshot path of one phase's output suite under
    ``originals_dir``, named from the suite's stem (``test_groupby_coverage.py``).
    The phase joins with ``_``: a dotted stem makes pytest import
    ``test_groupby.coverage`` as a package, so the snapshot never collects. A
    class language must keep its public class name matching this file name."""
    return Path(originals_dir) / f"{base_stem}_{phase}{suffix}"


# Logging

_log = PhaseLogger("test_metrics", unlogged="printed")

init_logging = _log.init
log_print = _log.log
save_logs = _log.save


# Result records

@dataclass
class TestFileMetrics:
    """One test file's metrics: the backend's counts and lizard's complexity."""
    name: str
    loc: Optional[int] = None
    test_count: Optional[int] = None
    assertion_count: Optional[int] = None
    parametrize_case_count: Optional[int] = None
    fixture_count: Optional[int] = None
    cyclomatic_complexity_sum: Optional[int] = None
    cyclomatic_complexity_max: Optional[int] = None
    source_observation: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CombinedTestMetrics:
    """Finishing's measurements of the delivered suite, from
    ``measure_combined``, for the target's tests.csv row."""
    target_file_path: str
    target_complexity_sum: Optional[int] = None
    target_loc: Optional[int] = None
    target_source_observation: Dict[str, Any] = field(default_factory=dict)
    coverage_final_line: Optional[float] = None
    coverage_final_branch: Optional[float] = None
    lint_smells_target: Optional[int] = None
    lint_smells_tests: Optional[int] = None
    # The suite timing, None unless its measure is on: warmup and timed runs
    # without coverage under the process-wide benchmark lock, in columns apart
    # from coverage and mutation.
    suite_timing_median_ms: Optional[float] = None
    suite_timing_samples_ms: Optional[str] = None
    suite_timing_run_count: Optional[int] = None
    suite_timing_warmup_count: Optional[int] = None
    # The processors requested for the timed runs, and those a process started
    # like them may use (its affinity, not the ones used), comma-separated.
    suite_timing_requested_cpus: Optional[str] = None
    suite_timing_allowed_cpus: Optional[str] = None
    test_files: List[TestFileMetrics] = field(default_factory=list)
    duration_s: Optional[float] = None
    status: str = "success"
    error: Optional[str] = None
    # The delivered suite's standalone mutation and coverage measurement, None
    # when it did not run.
    finishing: Optional["FinishingMetrics"] = None
    # Each phase's delivered suite, by phase: "mutation" reuses
    # ``finishing``, "starter" and "coverage" are read from those phases'
    # archived matrices and are absent when phase_scores is off or their
    # matrices are absent.
    phase_finishing: Dict[str, "FinishingMetrics"] = field(
        default_factory=dict
    )

    def to_csv_dict(self) -> Dict[str, Any]:
        """The tests.csv row's columns. A test file the backend classifies
        fills the ``_standard`` columns."""
        out: Dict[str, Any] = {
            "target_complexity_sum": self.target_complexity_sum,
            "target_loc": self.target_loc,
            "coverage_final_line": self.coverage_final_line,
            "coverage_final_branch": self.coverage_final_branch,
            "lint_smells_target": self.lint_smells_target,
            "lint_smells_tests": self.lint_smells_tests,
            "suite_timing_median_ms": self.suite_timing_median_ms,
            "suite_timing_samples_ms": self.suite_timing_samples_ms,
            "suite_timing_run_count": self.suite_timing_run_count,
            "suite_timing_warmup_count": self.suite_timing_warmup_count,
            "suite_timing_requested_cpus": self.suite_timing_requested_cpus,
            "suite_timing_allowed_cpus": self.suite_timing_allowed_cpus,
            "finishing_duration_s": self.duration_s,
            "finishing_status": self.status,
            "finishing_error": self.error,
        }
        observations = {}
        if self.target_source_observation:
            observations["target"] = self.target_source_observation
        for tfm in self.test_files:
            # Coverage and mutation revise one delivered file, so every file
            # the backend classifies is the standard suite.
            kind = "unknown"
            backend = languages.get_backend(
                languages.detect_language(tfm.name)
            )
            if backend is not None:
                try:
                    kind = backend.classify_test_kind(tfm.name)
                except NotImplementedError:
                    pass
            if kind == "unknown":
                continue
            role = "delivered"
            if tfm.source_observation:
                observations[role] = tfm.source_observation
            out[f"test_loc_{role}"] = tfm.loc
            out[f"test_count_{role}"] = tfm.test_count
            out[f"assertion_count_{role}"] = tfm.assertion_count
            out[f"parametrize_count_{role}"] = tfm.parametrize_case_count
            out[f"fixture_count_{role}"] = tfm.fixture_count
            out[f"test_complexity_sum_{role}"] = tfm.cyclomatic_complexity_sum
            out[f"test_complexity_max_{role}"] = tfm.cyclomatic_complexity_max
        if observations:
            out["source_observations"] = observations
        if self.finishing is not None:
            out.update(self.finishing.to_csv_dict())
        for phase in _PHASE_ORIGINAL_PHASES:
            fm = self.phase_finishing.get(phase)
            out[f"{phase}_original_line"] = (
                fm.line_coverage if fm is not None else None
            )
            out[f"{phase}_original_branch"] = (
                fm.branch_coverage if fm is not None else None
            )
            out[f"{phase}_original_mutation_full"] = (
                fm.mutation_score_full if fm is not None else None
            )
            out[f"{phase}_original_mutation_selective"] = (
                fm.mutation_score_selective if fm is not None else None
            )
        return out


# Test file analysis

def _analyze_test_file(path: Path) -> TestFileMetrics:
    """One test file's structural counts and the backend's observation record."""
    metrics = TestFileMetrics(name=path.name)
    observed = record(observe_file(path))
    metrics.source_observation = observed
    metrics.loc = observed["code_lines"]
    metrics.test_count = observed["tests"]
    metrics.assertion_count = observed["assertions"]
    metrics.fixture_count = observed["fixtures"]
    metrics.parametrize_case_count = observed["parametrized_cases"]
    return metrics


# Coverage

def _cov_via_backend(
    target_file: str,
    source_dir: str,
    pytest_targets: List[str],
    tag: str,
) -> tuple[Optional[float], Optional[float], bool]:
    """``(line_pct, branch_pct, timed_out)`` of ``target_file`` from running
    exactly the files ``pytest_targets`` (never a directory) through the
    backend's ``run_test_files``. The percentages are None for a failed
    measurement: no files, no backend, a timeout, or a red suite (the delivered
    suite is expected green, so partial coverage would mislead), an empty
    column rather than an error. ``timed_out`` tells a used-up coverage
    timeout from a structural failure."""
    if not pytest_targets:
        return None, None, False
    backend = languages.get_backend(languages.detect_language(target_file))
    if backend is None:
        return None, None, False
    cov_timeout = int(_AGENT_CONFIG["coverage_timeout_seconds"])
    res = backend.run_test_files(
        list(pytest_targets), source_dir,
        timeout=cov_timeout, measure_coverage_of=target_file,
    )
    if res.timed_out:
        log_print(f"[metrics] {tag} coverage run TIMED OUT", level=1)
        from testuary import results

        log_print("WARNING: " + results.note(
            "finishing", "finishing.coverage_timeout_seconds", cov_timeout))
        return None, None, True
    if not res.ok:
        tail = (res.output or "").strip()
        if len(tail) > 3000:
            tail = "..." + tail[-3000:]
        log_print(
            f"[metrics] {tag} coverage run failed "
            f"(exit={res.returncode}), so {tag} coverage is unmeasurable. "
            f"Runner output tail:\n{tail}",
            level=1,
        )
        return None, None, False
    return res.line_coverage, res.branch_coverage, False


# Suite timing

def _measure_suite_time(
    result: "CombinedTestMetrics",
    backend: Any,
    test_paths: List[str],
    source_dir: str,
    target_file_path: Optional[str] = None,
) -> None:
    """Time the standard suite without coverage through
    ``backend.benchmark_tests`` (warmup and every configured timed run, under
    the process-wide benchmark lock) and record the median and samples on
    ``result``. A backend without it, a red suite, a timeout or an error
    leaves the columns None. Each run's timeout is the combined coverage
    timeout."""
    benchmark = getattr(backend, "benchmark_tests", None)
    if benchmark is None:
        return
    timeout = int(_AGENT_CONFIG["coverage_timeout_seconds"])
    try:
        stats = benchmark(
            source_dir=source_dir,
            test_file_paths=list(test_paths),
            warmup_runs=int(_TIMING_CONFIG["warmup_count"]),
            benchmark_runs=int(_TIMING_CONFIG["run_count"]),
            timeout=timeout,
            suite_label="delivered",
            progress_fn=lambda msg: log_print(f"[metrics] {msg}", level=1),
            instrument_target=target_file_path,
        )
    except Exception as exc:
        log_print(f"[metrics] suite runtime measurement failed: {exc}", level=1)
        return
    if not stats.get("ok"):
        log_print(
            "[metrics] suite runtime unmeasurable "
            f"({stats.get('error', 'suite did not run green')})",
            level=1,
        )
        return
    samples = stats.get("times_ms") or []
    result.suite_timing_median_ms = stats.get("median_ms")
    result.suite_timing_samples_ms = ";".join(f"{v:.4f}" for v in samples)
    result.suite_timing_run_count = stats.get("timing_run_count")
    result.suite_timing_warmup_count = stats.get("warmup_runs")
    result.suite_timing_requested_cpus = stats.get("requested_cpus")
    allowed = stats.get("allowed_cpus")
    result.suite_timing_allowed_cpus = (
        ",".join(str(cpu) for cpu in allowed) if allowed else None)
    if result.suite_timing_median_ms is not None:
        log_print(
            f"[metrics] suite runtime: median={result.suite_timing_median_ms:.0f}ms "
            f"over {result.suite_timing_run_count} reps "
            f"(warmup {result.suite_timing_warmup_count})",
            level=1,
        )


# Entry point

def measure_combined(
    target_file_path: str,
    test_output_dir: str,
    source_dir: str,
    test_file_names: List[str],
    recorded_coverage: Optional[Tuple[Optional[float], Optional[float]]] = None,
) -> CombinedTestMetrics:
    """Finishing's measurements of the delivered suite: the files
    ``test_file_names`` in ``test_output_dir`` (normally one
    ``test_<stem>.py``), run from ``source_dir``. ``recorded_coverage`` is
    the line and branch coverage the phase end that delivered the suite
    measured. Given, it is taken without a coverage run. A field whose
    measurement failed stays None."""
    start = time.perf_counter()
    log_print(f"[metrics] measure_combined start for {Path(target_file_path).name}")

    result = CombinedTestMetrics(target_file_path=target_file_path)

    # In container mode the source lives in the container: the complexity
    # tool, which analyses a path, gets a host temporary copy.
    target_path = Path(target_file_path)
    target_backend = languages.get_backend(languages.detect_language(target_file_path))
    target_observation = observe_file(
        target_path, backend=target_backend, read_text=project.read_project_text,
    )
    result.target_source_observation = record(target_observation)
    result.target_loc = result.target_source_observation["code_lines"]
    if config_module.project_exec_mode() == "container":
        try:
            text = project.read_project_text(target_file_path)
        except OSError:
            text = None
        if text is not None and target_observation.completed:
            tmp = None
            try:
                with tempfile.NamedTemporaryFile(
                    "w", suffix=(target_path.suffix or ".txt"),
                    delete=False, encoding="utf-8",
                ) as tf:
                    tf.write(text)
                    tmp = tf.name
                cc_sum, _ = compute_file_complexity_stats(tmp)
            finally:
                if tmp:
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass
            result.target_complexity_sum = cc_sum
            log_print(
                f"[metrics] target metrics: loc={result.target_loc}, "
                f"cc_sum={cc_sum}",
                level=1,
            )
    elif target_observation.completed:
        cc_sum, _ = compute_file_complexity_stats(target_path)
        result.target_complexity_sum = cc_sum
        log_print(
            f"[metrics] target metrics: loc={result.target_loc}, "
            f"cc_sum={cc_sum}",
            level=1,
        )

    # Per-test-file metrics of the delivered suite.
    combined_paths: List[str] = []
    for name in test_file_names:
        path = Path(test_output_dir) / name
        tfm = _analyze_test_file(path)
        result.test_files.append(tfm)
        if not path.exists():
            continue
        combined_paths.append(str(path))
        if tfm.source_observation["status"] == "completed":
            cc_sum, cc_max = compute_file_complexity_stats(path)
            tfm.cyclomatic_complexity_sum = cc_sum
            tfm.cyclomatic_complexity_max = cc_max
        log_print(
            f"[metrics] {name}: loc={tfm.loc}, tests={tfm.test_count}, "
            f"asserts={tfm.assertion_count}, params={tfm.parametrize_case_count}, "
            f"fixtures={tfm.fixture_count}, cc_sum={tfm.cyclomatic_complexity_sum}",
            level=1,
        )

    # Coverage of the delivered suite's existing files: its phase end's
    # measurement, else a run of its own.
    if recorded_coverage is not None:
        line_pct, branch_pct = recorded_coverage
        log_print("[metrics] combined coverage: the delivering phase end's "
                  "measurement", level=1)
    else:
        line_pct, branch_pct, _ = _cov_via_backend(
            target_file_path, source_dir, combined_paths, "combined",
        )
    result.coverage_final_line = line_pct
    result.coverage_final_branch = branch_pct
    if line_pct is not None:
        log_print(
            f"[metrics] combined coverage: line={line_pct:.1f}%, "
            f"branch={branch_pct:.1f}%" if branch_pct is not None else
            f"[metrics] combined coverage: line={line_pct:.1f}%",
            level=1,
        )

    if target_backend is not None and combined_paths:
        _write_coverage_matrix(target_backend, target_file_path, source_dir,
                               combined_paths, test_output_dir)

    # The suite timing, off by default.
    if _bool_flag(_TIMING_CONFIG, "measure") and combined_paths and target_backend is not None:
        _measure_suite_time(
            result, target_backend, combined_paths, source_dir,
            target_file_path=target_file_path,
        )

    # Lint, off by default: the language's linter (ruff for Python), no
    # counts for a language without one.
    if _bool_flag(_RAW_METRICS_CONFIG, "measure_lint") and target_backend is not None:
        result.lint_smells_target = target_backend.code_smells([target_path])
        test_paths = [Path(test_output_dir) / n for n in test_file_names]
        test_paths = [p for p in test_paths if p.exists()]
        result.lint_smells_tests = target_backend.code_smells(test_paths) if test_paths else 0
        log_print(
            f"[metrics] lint smells: target={result.lint_smells_target}, "
            f"tests={result.lint_smells_tests}",
            level=1,
        )

    result.duration_s = time.perf_counter() - start
    log_print(f"[metrics] done in {result.duration_s:.1f}s")
    return result


# The mutation phase end: merge and select its suites


@dataclass
class MergeResult:
    """The mutation phase end's selected suite. ``merged_path`` is the
    delivered test file (it keeps the inherited file's name). The two input
    suites are kept in a sibling ``_pre_merge`` directory, outside the test
    directory, so neither pytest nor the Java compiler picks them up."""

    merged_path: str
    pre_merge_coverage_path: str
    pre_merge_mutation_path: str
    #: The full-catalogue kill matrix of the adopted suite, as consolidation
    #: measured or derived it. None when no matrix was measured.
    kill_matrix: Optional[dict] = None
    #: For the CSV: the test counts of the merged pool and of the delivered
    #: suite (None when not measured).
    union_tests: Optional[int] = None
    delivered_tests: Optional[int] = None
    winner: str = "merged"
    winner_source_label: Optional[str] = None


def _atomic_write_text(path: Path, content: str) -> None:
    """Write ``content`` to ``path`` through a temporary file and
    ``os.replace``, so a crash never leaves it half-written."""
    fd, tmp = tempfile.mkstemp(dir=str(path.parent), suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, str(path))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def _bound_to_delivered(evidence: Dict[str, Any], suite: Path) -> Dict[str, Any]:
    """``evidence`` stamped, as the phase archive's matrices are, with the
    delivered suite ``suite`` it describes: the hash of its text, which the
    phase records hash, and its file name."""
    from testuary.consolidation import _bound_to_source

    return _bound_to_source(evidence, source_sha256(suite.read_text(encoding="utf-8")),
                            suite.name)


def _write_coverage_matrix(backend: Any, target_file_path: str, source_dir: str,
                           suite_paths: List[str], test_output_dir: str) -> None:
    """The delivered suite's per-line coverage matrix in
    ``test_output_dir``, the coverage counterpart of kill_matrix.json,
    stamped like it with the suite (``suite_paths``, one file). Nothing
    gates on it. A backend without per-line detail writes none, and an
    earlier attempt's matrix is removed first, so only this run's is
    stamped. A failure is logged."""
    matrix_path = Path(test_output_dir) / "coverage_matrix.json"
    try:
        matrix_path.unlink(missing_ok=True)
        backend.capture_coverage_matrix(
            target=Path(target_file_path), source_dir=Path(source_dir),
            test_paths=suite_paths, out_path=matrix_path,
            timeout=int(config_module.get_agent_config("test_coverage")["suite_run_timeout_seconds"]),
        )
        if matrix_path.is_file():
            _atomic_write_text(matrix_path, json.dumps(_bound_to_delivered(
                json.loads(matrix_path.read_text(encoding="utf-8")), Path(suite_paths[0])),
                indent=2))
    except Exception as exc:
        log_print(f"[metrics] coverage-matrix capture failed: {exc}", level=1)


from contextlib import ExitStack, contextmanager, nullcontext  # noqa: E402


def union_attribution(config) -> str:
    """How the phase-end (union) measurement attributes kills: "all" (every
    killing test, so the runtime cover keeps the cheapest) unless
    ``cover_attribution`` is "first". Its matrices carry this in their
    measurement identity."""
    return ("first" if str(config.get("cover_attribution", "full")).strip().lower()
            == "first" else "all")


@contextmanager
def union_mutation_runner(
    *,
    language: str,
    target_file: str,
    source_dir: str,
    config: Dict[str, Any],
    log_fn: Optional[Any] = None,
):
    """One engine session for one full-catalogue measurement of a suite
    union, yielding the ``run_mutation_on`` callable, or None when the engine
    cannot start (logged: with phase_end_mutation on, the caller then fails,
    else it goes on without kill evidence). The session closes on exit. A
    session whose cleanup failed ends the caller's work: its error propagates
    from the opening, the body or the closing, and an error that ended the
    body is never replaced by the closing's."""
    from testuary.languages.mutation import close_mutation_session, mutation_cleanup_pending
    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    session = None
    runner = None
    scratch = ExitStack()
    try:
        # The logger makes this multi-minute pass log its progress. The
        # session calls log(msg, level), so a one-argument log_fn is adapted.
        mutation = languages.get_backend(language).mutation_runner(
            config, source_dir=source_dir,
            log=(level_tolerant_log(log_fn)
                 if log_fn is not None else None),
        )
        # The phase end's engine drops -x, so every killer is recorded and the
        # runtime cover keeps the cheapest. The loops keep -x.
        mutation.attribution = union_attribution(config)
        # The session declares the directory its suites are copied into, so a
        # run writes nothing beside a caller's suite.
        suites_root = Path(scratch.enter_context(
            project.scratch_dir(prefix="testuary_mutation_suites_")))
        session = mutation.open(
            int(config["engine_workers"]),
            suite_dirs=(suites_root,))
        runner = _session_mutation_fn(
            session, suites_root, mutation.full_operators(), target_file=target_file,
            source_dir=source_dir, config=config, language=language, log_fn=log_fn)
    except Exception as exc:
        if mutation_cleanup_pending(exc):
            try:
                if session is not None:
                    close_mutation_session(session, primary=exc)
            finally:
                scratch.close()
            raise
        consequence = (
            "The setting phase_end_mutation needs kill evidence, so this step fails"
            if config_module.consolidation_measure_mutation()
            else "This step goes on without kill evidence"
        )
        _log(
            f"[consolidate] the mutation engine did not start for the union "
            f"measurement ({type(exc).__name__}). {consequence}"
        )
        runner = None
    try:
        yield runner
    except BaseException as exc:
        if session is not None:
            close_mutation_session(session, primary=exc)
        raise
    else:
        if session is not None:
            close_mutation_session(session)
    finally:
        scratch.close()


@contextmanager
def phase_end_runner(phase: str, *, always: bool = False, **runner):
    """The union mutation runner (``union_mutation_runner`` with ``runner``)
    of ``phase``'s phase end: None when the phase end measures no kills,
    unless it ``always`` opens its engine. With phase_end_mutation on, an
    engine that cannot start fails the file."""
    if not always and not config_module.consolidation_measure_mutation():
        yield None
        return
    with union_mutation_runner(**runner) as run_mutation_on:
        if run_mutation_on is None and config_module.consolidation_measure_mutation():
            raise RuntimeError(
                f"the {phase} phase end needs kill evidence "
                "(phase_end_mutation is on) and the mutation "
                "engine could not start, so this target fails rather than "
                "being reduced on coverage alone"
            )
        yield run_mutation_on


#: Schema of the persisted per-suite attribution record: each starter suite
#: measured alone, from per-suite verdicts.
STARTER_POOL_ATTRIBUTION_SCHEMA = "testuary.starter_pool_attribution/3"


def _with_suites_not_measured(union: Dict[str, Any], names, own: Dict[str, Dict[str, Any]]) -> None:
    """Add the suites ``names``, which never entered the per-suite request
    (not green), to ``union`` as the backend adds a suite whose run measured
    nothing: each has no result for any mutant, so a row no suite killed is
    untested and the union is stamped incomplete. A non_viable row stays
    non_viable when a measured suite's own matrix in ``own`` shows that its
    mutant did not load (the C engine's native status ``load_error``), since
    the load error belongs to the mutant, not to a suite."""

    did_not_load = {json.dumps(row["key"], sort_keys=True)
                    for matrix in own.values() for row in matrix.get("mutants") or []
                    if row.get("native_status") == "load_error" and row.get("key")}
    for row in union.get("mutants") or []:
        verdicts = row.setdefault("suite_verdicts", {})
        for name in names:
            verdicts.setdefault(name, {"verdict": UNTESTED_STATUS, "killers": []})
        kept = row.get("status") == KILLED_STATUS or (
            row.get("status") == NON_VIABLE_STATUS
            and json.dumps(row.get("key"), sort_keys=True) in did_not_load)
        if not kept:
            row["status"] = UNTESTED_STATUS
    summary = mutation.refresh_summary(union)["summary"]
    if summary["untested"]:
        union[mutation.INCOMPLETE_KEY] = {
            "reason": UNTESTED_MUTANTS_IN_CATALOGUE,
            "untested": summary["untested"], "total": summary["total"]}


def per_suite_attribution_record(measured: Optional[Dict[str, Any]], *, slots_by_name: Dict[str, int],
                                 requested_slots: int, mode: str) -> Optional[Dict[str, Any]]:
    """The version 3 record of starter suites each measured alone, from a
    per-suite measurement ``measured`` (``run_mutation_on.per_suite``). A
    slot's ``valid`` and ``killed`` count its suite's own matrix as a
    standalone measurement scores it, and ``verdicts`` counts the union
    matrix's verdicts for that suite. A slot without an accepted or measured
    suite has no counts, never zeros. One whose suite was in the request
    carries its run's ``outcome`` (a run the wall clock cut shows as such)."""

    if measured is None:
        return None
    rows = (measured.get("union") or {}).get("mutants") or []
    by_slot = {slot: name for name, slot in slots_by_name.items()}
    entries = []
    for slot in range(1, requested_slots + 1):
        name = by_slot.get(slot)
        own = measured["suites"].get(name) if name is not None else None
        entry: Dict[str, Any] = {"slot": slot, "accepted": name is not None, "measured": own is not None}
        if name in (measured.get("outcomes") or {}):
            entry["outcome"] = measured["outcomes"][name]
        if own is None:
            entry["killed"] = None
        else:
            valid = mutation.valid_ids(own)
            entry["valid"] = len(valid)
            entry["killed"] = sum(1 for m in own.get("mutants") or []
                                  if m.get("id") in valid and m.get("status") in DETECTED_STATUSES)
            counts = dict.fromkeys(VERDICTS, 0)
            for row in rows:
                verdict = ((row.get("suite_verdicts") or {}).get(name) or {}).get("verdict")
                counts[verdict if verdict in counts else UNTESTED_STATUS] += 1
            entry["verdicts"] = counts
        entries.append(entry)
    return {
        "schema": STARTER_POOL_ATTRIBUTION_SCHEMA,
        "basis": "per_suite_verdicts",
        "implementation": measured.get("implementation"),
        "attribution_mode": mode,
        "union_mutants": len(rows),
        "slots": entries,
    }


def per_suite_scores(measured: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Each measured suite's own mutation score, as a standalone run gives it."""

    scores = {}
    for name, own in ((measured or {}).get("suites") or {}).items():
        score = mutation.matrix_score_percent(own)
        if score is not None:
            scores[name] = score
    return scores


def consolidate_test_suite(
    *,
    test_output_dir: str,
    coverage_file_name: str,
    language: str,
    target_file_path: str,
    source_dir: str,
    candidate1_source: str,
    candidate2_source: str,
    run_mutation_on: Optional[Any] = None,
    candidate1_km: Optional[dict] = None,
    candidate1_coverage: Optional[dict] = None,
    candidate1_measurement_identity: Optional[dict] = None,
    candidate2_km: Optional[dict] = None,
    candidate2_source_sha256: Optional[str] = None,
    candidate2_execution_name: Optional[str] = None,
    candidate2_measurement_identity: Optional[dict] = None,
    measurement_identity: Optional[dict] = None,
    timeout: int = 300,
    reduction: "ReductionSettings",
    matrices_dir: Optional[str] = None,
    selective_operators: Optional[List[str]] = None,
    log_fn: Optional[Any] = None,
    generation_scope: Optional[dict] = None,
    generation_inputs: Optional[dict] = None,
) -> Optional[MergeResult]:
    """Merge the inherited and the agent's suite (separate candidates even
    when one delivered path produced both), then select the phase's delivered
    suite. Existing measurements are reused only when the caller binds them to
    the exact source and execution name. An inherited matrix of another
    measurement identity is measured again. The merge and its reduction carry
    this call's measurements, so selection repeats no coverage or mutation
    work. Returns None without a backend. Raises RuntimeError
    when the backend lacks a needed operation or no candidate passes with
    complete required scores."""
    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    backend = languages.get_backend(language)
    if backend is None:
        return None
    out_dir = Path(test_output_dir)
    coverage_path = out_dir / coverage_file_name
    coverage_src = candidate1_source
    mutation_src = candidate2_source
    execution_name = coverage_path.name

    # The merged suite is measured in its own namespace: full-catalogue
    # mutation gives the cover's kill tokens, per-test coverage its line and
    # branch tokens. The reduced suite is also measured on its own.
    from testuary.consolidation import measure_candidate_standalone

    candidate1_remeasured = None
    # The inherited suite as the coverage phase delivered it, and the removals
    # of its place of the removal fallback.
    inherited_src = coverage_src
    candidate1_removals: List[Dict[str, Any]] = []
    if (
        candidate1_km is not None
        and candidate1_measurement_identity != measurement_identity
    ):
        _log(
            "[consolidate] inherited mutation matrix has a stale measurement "
            "identity, so the unchanged inherited suite is measured again"
        )
        candidate1_remeasured = measure_candidate_standalone(
            backend,
            coverage_src,
            target_file_path=target_file_path,
            source_dir=source_dir,
            timeout=timeout,
            mutation_runner=run_mutation_on,
            stage_as_delivered=True,
            log=_log,
            expected_measurement_identity=measurement_identity,
            candidate_label="standalone_0",
            candidate_name="candidate-1",
            place="phase_end.inherited",
            removals=candidate1_removals,
        )
        coverage_src = candidate1_remeasured.source
        candidate1_source = coverage_src
        candidate1_km = candidate1_remeasured.kill_matrix
        candidate1_coverage = candidate1_remeasured.coverage_evidence
        candidate1_measurement_identity = measurement_identity

    from testuary.consolidation import (
        InheritedMatrix,
        PhaseEnd,
        PhaseEndContext,
        PhaseEndOptions,
        _note_removals,
        run_phase_end,
        scored_from_measurement,
        source_test_count,
    )
    inherited_changed = any("removed" in entry for entry in candidate1_removals)

    def _with_inputs(phase, consolidated):
        # The two input suites, each measured on its own, follow the
        # consolidation's candidates.
        candidates = list(consolidated)
        if candidate1_remeasured is not None:
            candidates.append(candidate1_remeasured)
            _note_removals(
                PhaseEndContext(backend, target_file_path, source_dir, timeout), phase,
                "phase_end.inherited", candidate1_removals, inherited_src,
                candidate1_remeasured.source,
                archive_name=f"candidate_standalone_0_removed{coverage_path.suffix}")
        else:
            candidate1_green = bool(
                candidate1_coverage
                and (run_mutation_on is None or candidate1_km is not None)
            )
            candidates.append(scored_from_measurement(
                label="standalone_0",
                source=coverage_src,
                kill_matrix=candidate1_km,
                coverage=candidate1_coverage,
                execution_name=execution_name,
                test_count=source_test_count(backend, coverage_src),
                green=candidate1_green,
                status=("passing" if candidate1_green else "incomplete"),
                target_revision=phase.target_revision,
                measurement_identity=(
                    candidate1_measurement_identity or phase.measurement_identity
                ),
            ))
        if phase.candidate2_candidate is not None:
            candidates.append(phase.candidate2_candidate)
        else:
            candidates.append(scored_from_measurement(
                label="standalone_1",
                source=(phase.candidate2_source or mutation_src),
                kill_matrix=phase.candidate2_km,
                coverage=phase.candidate2_coverage,
                execution_name=(
                    phase.candidate2_execution_name or execution_name
                ),
                test_count=source_test_count(backend, phase.candidate2_source or mutation_src),
                green=phase.candidate2_green,
                status=("passing" if phase.candidate2_green else "incomplete"),
                status_reason=phase.candidate2_status_reason,
                target_revision=phase.target_revision,
                measurement_identity=phase.measurement_identity,
            ))
        return candidates

    # The one persistence point, on every outcome.
    outcome = run_phase_end(
        PhaseEnd(
            name="mutation", backend=backend, sources=[coverage_src, mutation_src],
            target_file_path=target_file_path, source_dir=source_dir,
            timeout=timeout, matrices_dir=matrices_dir, execution_name=execution_name,
            log=_log,
            options=PhaseEndOptions(
                mutation_method=config_module.consolidation_mutation_method(),
                reduction=reduction,
                candidate_evidence=[
                    (candidate1_coverage or {}).get("per_test_tokens") or None,
                    None,
                ],
                inherited1=InheritedMatrix(
                    candidate1_km, source_sha256(coverage_src), execution_name,
                    candidate1_measurement_identity),
                inherited2=InheritedMatrix(
                    candidate2_km, candidate2_source_sha256, candidate2_execution_name,
                    candidate2_measurement_identity),
                measurement_identity=measurement_identity,
            ),
            generation_scope=generation_scope,
            generation_inputs=generation_inputs,
            unresolved_bound=True,
        ),
        nullcontext(run_mutation_on),
        own_candidates=_with_inputs,
        no_winner=lambda _names: (
            "the mutation phase end produced no passing candidate with "
            "complete required score measurements"
        ),
        single_suite="standalone_",
        delivered_tests=lambda winner: (
            winner.test_count if winner.test_count is not None
            else source_test_count(backend, winner.source)
        ),
    )
    phase, winner = outcome.phase, outcome.winner
    km_ = phase.merged_km
    if km_ is not None and matrices_dir is not None and selective_operators:
        # The selective operator set's union view, beside the full union matrix.
        sel = set(selective_operators)
        mutants = [m for m in (km_.get("mutants") or [])
                   if m.get("operator") in sel]
        _atomic_write_text(
            Path(matrices_dir) / "merged_kill_matrix_selective.json",
            json.dumps({
                "operators": sorted(sel),
                "mutants": mutants,
                "killed": len(mutation.killed_ids(km_, operators=sel)),
                "total": len(mutation.valid_ids(km_, operators=sel)),
                "score": mutation.matrix_score_percent(km_, operators=sel),
            }, indent=2))
    if winner is None:
        raise RuntimeError(outcome.detail)

    merged = winner.source
    kill_matrix = winner.kill_matrix
    winner_label = outcome.label
    winner_source_label = outcome.source_label
    delivered_n = outcome.delivered_tests

    pre_dir = out_dir.parent / "_pre_merge"
    pre_dir.mkdir(parents=True, exist_ok=True)
    pre_cov = pre_dir / f"candidate_inherited{coverage_path.suffix}"
    pre_mut = pre_dir / f"candidate_agent{coverage_path.suffix}"
    _atomic_write_text(pre_cov, inherited_src if inherited_changed else coverage_src)
    _atomic_write_text(pre_mut, mutation_src)
    _atomic_write_text(coverage_path, merged)
    _log(
        f"[consolidate] the selected candidate {winner.label} is now "
        f"{coverage_file_name}. The two input suites are kept under "
        f"{pre_dir.name}/"
    )
    return MergeResult(
        merged_path=str(coverage_path),
        pre_merge_coverage_path=str(pre_cov),
        pre_merge_mutation_path=str(pre_mut),
        kill_matrix=kill_matrix,
        union_tests=phase.pool_tests,
        delivered_tests=delivered_n,
        winner=winner_label,
        winner_source_label=winner_source_label,
    )


# Finishing: the delivered suite's standalone measurement


@dataclass
class FinishingMetrics:
    """Finishing's measurement of the delivered suite, standalone: mutation
    and coverage. With an engine that knows selective catalogues the run uses
    the full catalogue: the ``_full`` scores cover it, the ``_selective``
    ones filter the same run by operator. Other engines fill ``_full`` from
    their configured run and leave ``_selective`` None. The counts
    (``killed``, ``survived``, ``total``, ``untested``) come from the same
    kill matrix.

    ``mutation_measurement`` is ``complete``, ``incomplete`` when the engine
    stamped the catalogue unresolved (no score, counts of the resolved
    mutants only), ``failed`` without a matrix, or None when no mutation was
    measured (the explicit opt-out). A failed measurement leaves its fields
    None."""

    mutation_score_full: Optional[float] = None
    #: Kills by the per-mutant timeout (timeout status) in the standard matrix.
    #: Nonzero flags the score as timeout-dependent: such verdicts can flip
    #: under load, so the row calls for a re-measure with a longer timeout.
    timeout_kills: Optional[int] = None
    mutation_score_selective: Optional[float] = None
    mutation_measurement: Optional[str] = None
    killed: Optional[int] = None
    survived: Optional[int] = None
    total: Optional[int] = None
    line_coverage: Optional[float] = None
    branch_coverage: Optional[float] = None
    #: A measurement that left mutants untested (``mutation_measurement``
    #: incomplete): its score counting every untested mutant as not
    #: killed, and how many were untested. None otherwise.
    mutation_score_lower_bound: Optional[float] = None
    untested: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_csv_dict(self) -> Dict[str, Any]:
        """The tests.csv columns of finishing's measurement: the delivered
        suite's coverage and mutation metrics (``CombinedTestMetrics`` writes
        the ``coverage_final_*`` columns), each score with a full and a
        selective column. The counts cover the catalogue as far as it
        resolved. ``finishing_mutation_measurement`` says whether it did."""
        return {
            "finishing_mutation_score_full": self.mutation_score_full,
            "finishing_timeout_kills": self.timeout_kills,
            "finishing_mutation_score_selective": self.mutation_score_selective,
            "finishing_mutation_measurement": self.mutation_measurement,
            "finishing_killed": self.killed,
            "finishing_survived": self.survived,
            "finishing_total": self.total,
            "finishing_line_coverage": self.line_coverage,
            "finishing_branch_coverage": self.branch_coverage,
            "finishing_mutation_score_lower_bound": self.mutation_score_lower_bound,
            "finishing_untested": self.untested,
        }


def _clean_dir(path: Path) -> None:
    """Remove ``path`` and everything under it, if present: the per-suite
    scratch, before a run (a crashed run's leftovers) and after it."""
    if path.exists():
        shutil.rmtree(path, ignore_errors=True)


def _green_check_with_retry(backend, test_dir, source_dir, timeout, log, *,
                            test_paths, instrument_target=None):
    """The un-mutated suite's final green check: its normal run
    (:mod:`testuary.normal_run`), or for several files one run of them as
    the mutation tool runs them, in one session, since its time sizes the
    per-mutant timeout. Retried with
    backoff while it returns "no tests collected" (returncode 5) or a runner
    error (4): a hiccup under load, not a failing test (1), on which refusing
    to mutate would lose the mutation score. A red or empty suite stays not
    ok and the caller refuses to mutate. Returns the run's result, or None
    when the run raised (a TypeError or a pending session cleanup propagates)."""
    pre = None
    for _attempt in range(_GREEN_CHECK_ATTEMPTS):
        try:
            if len(test_paths) > 1:
                pre = backend.run_test_files(
                    list(test_paths), source_dir, timeout=int(timeout),
                    instrument_target=instrument_target, tb="line")
            else:
                pre = normal_run.run(
                    backend, list(test_paths), source_dir, timeout=int(timeout),
                    target=instrument_target, tb="line")
        except TypeError:
            # A signature mismatch is a wiring bug, not a suite outcome: caught,
            # it would give "no mutation score" for a reason no record explains.
            raise
        except Exception as exc:
            if mutation.mutation_cleanup_pending(exc):
                raise
            log(f"[finishing] baseline suite run raised ({exc}), so "
                "mutation is skipped and no score is reported")
            return None
        if pre.ok or pre.returncode not in (4, 5):
            return pre
        tail = "\n".join(
            (getattr(pre, "output", "") or "").splitlines()[-20:])
        log(f"[finishing] un-mutated green check returned returncode="
            f"{pre.returncode} (spurious, attempt {_attempt + 1}/"
            f"{_GREEN_CHECK_ATTEMPTS}). Output tail:\n{tail}")
        if _attempt + 1 == _GREEN_CHECK_ATTEMPTS:
            # What the run was pointed at, since a run collecting nothing says
            # nothing else. On the last attempt only, so a recovered hiccup is quiet.
            try:
                listing = sorted(p.name for p in Path(test_dir).iterdir())
            except OSError as exc:
                listing = [f"<unreadable: {type(exc).__name__}: {exc}>"]
            log(f"[finishing] the green check never passed. It ran "
                f"{test_dir} (contents: {listing}) against source directory "
                f"{source_dir}, collected="
                f"{getattr(pre, 'collected', None)} "
                f"passed={getattr(pre, 'passed', None)}")
        if _attempt + 1 < _GREEN_CHECK_ATTEMPTS:
            delay = _GREEN_CHECK_BACKOFF_S[
                min(_attempt, len(_GREEN_CHECK_BACKOFF_S) - 1)]
            time.sleep(delay)
    return pre


def _build_suite_dir(dest: Path, suite_file: Path, shared_files: List[Path]) -> Path:
    """Create ``dest`` afresh with one suite's test file and the shared setup
    files (conftest, pom) and return it: the engine runs only the tests it
    finds there, so the suite's score is standalone."""
    _clean_dir(dest)
    dest.mkdir(parents=True, exist_ok=True)
    suite_file = Path(suite_file)
    shutil.copy2(str(suite_file), str(dest / suite_file.name))
    for shared in shared_files:
        shared = Path(shared)
        if shared.is_file():
            shutil.copy2(str(shared), str(dest / shared.name))
    return dest


def _mutation_measurement(matrix: Optional[Dict[str, Any]]) -> str:
    """How a final mutation measurement ended: ``failed`` without a matrix,
    ``incomplete`` when the engine stamped its catalogue unresolved, else
    ``complete``."""
    if not matrix:
        return "failed"
    if matrix.get(mutation.INCOMPLETE_KEY):
        return "incomplete"
    return "complete"


def measure_finishing(
    *,
    scratch_root: Path,
    merged_test_path: Path,
    shared_setup_paths: List[Path],
    run_mutation_on: Optional[Callable[[Path], Optional[Dict[str, Any]]]],
    standard_coverage: Tuple[Optional[float], Optional[float]],
    selective_operators: Optional[set] = None,
    kill_matrix_out: Optional[Path] = None,
    inherited_standard_km: Optional[dict] = None,
    log_fn: Optional[Any] = None,
) -> FinishingMetrics:
    """The delivered suite's standalone measurement: mutation of the standard
    (merged) suite in its own clean directory, so the engine scores it alone,
    and its coverage. ``run_mutation_on(test_dir)`` runs the full catalogue
    and returns a parsed kill matrix or None. None for the callable measures
    no mutation (the explicit opt-out), and ``inherited_standard_km`` is
    used instead of a run. Under the opt-out the caller passes neither.
    ``selective_operators`` gives the ``_selective``
    score by filtering the same matrix by operator, when it carries
    operators. ``standard_coverage`` is the suite's ``(line, branch)`` as
    measured before. ``run_mutation_on`` handles its own errors. The scratch
    is removed regardless. A failed write of ``kill_matrix_out`` raises
    RuntimeError."""

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    scratch_root = Path(scratch_root)
    merged_test_path = Path(merged_test_path)

    _clean_dir(scratch_root)
    try:
        std_dir = _build_suite_dir(
            scratch_root / "delivered", merged_test_path, shared_setup_paths,
        )
        measured = True
        if inherited_standard_km is not None:
            _log("[finishing] mutation: standard suite reuses the "
                 "consolidation's kill matrix")
            standard_km = inherited_standard_km
        elif run_mutation_on is not None:
            _log("[finishing] mutation: standard suite (full catalogue)")
            standard_km = run_mutation_on(std_dir)
        else:
            standard_km = None
            measured = False

        # Log a matrix with mostly unattributed, timeout or unresolved kills.
        # The score is its kill rate either way.
        alarm = mutation.unattributed_kill_alarm(standard_km)
        if alarm:
            _log(alarm)
        t_alarm = mutation.timeout_kill_alarm(standard_km)
        if t_alarm:
            _log(t_alarm)
        u_alarm = mutation.unresolved_kill_alarm(standard_km)
        if u_alarm:
            _log(u_alarm)

        # The standard suite's matrix as kill_matrix.json, stamped with the suite.
        # Only run_target_phases, which owns the tests directory, passes a path.
        if kill_matrix_out is not None and standard_km:
            try:
                _atomic_write_text(
                    Path(kill_matrix_out),
                    json.dumps(_bound_to_delivered(standard_km, merged_test_path), indent=2),
                )
            except Exception as exc:
                _log(f"[finishing] persisting kill_matrix.json failed: {exc}")
                raise RuntimeError(
                    "[finishing] could not persist kill_matrix.json"
                ) from exc

        # Over an incomplete catalogue the counts cover the resolved mutants.
        # Without a matrix there are none.
        full = mutation.matrix_counts(standard_km)
        metrics = FinishingMetrics(
            mutation_score_full=full["mutation_score"],
            timeout_kills=(
                mutation.timeout_kill_count(standard_km)
                if standard_km else None),
            mutation_measurement=(
                _mutation_measurement(standard_km) if measured else None),
        )
        if standard_km:
            metrics.killed = full["killed"]
            metrics.survived = full["survived"]
            metrics.total = full["total"]
        if metrics.mutation_measurement == "incomplete":
            # It left mutants untested: no score, and its lower bound.
            metrics.mutation_score_lower_bound = full["mutation_score_lower_bound"]
            metrics.untested = full["untested"]
        # The selective score filters the same run by operator, when a
        # selective set is given and the matrix carries operators.
        if selective_operators and mutation.has_operators(standard_km):
            metrics.mutation_score_selective = mutation.matrix_counts(
                standard_km, operators=selective_operators)["mutation_score"]

        std_line, std_branch = standard_coverage
        metrics.line_coverage = std_line
        metrics.branch_coverage = std_branch
        if log_fn is not None:
            from testuary.agent import _helpers as agent_helpers
            agent_helpers.log_phase_metrics(
                log_fn, "finishing",
                line=metrics.line_coverage,
                branch=metrics.branch_coverage,
                mutation=metrics.mutation_score_full,
                note="final standard suite |",
            )
        return metrics
    finally:
        _clean_dir(scratch_root)


def finishing_csv_columns() -> List[str]:
    """The ordered ``FinishingMetrics`` CSV column names (the keys
    ``to_csv_dict`` emits)."""
    return list(FinishingMetrics().to_csv_dict().keys())


def phase_finishing_csv_columns() -> List[str]:
    """The per-phase column names of tests.csv: for each phase's output suite,
    its line and branch coverage and full and selective mutation score (the
    one source for the schema and the row writer)."""
    cols: List[str] = []
    for phase in _PHASE_ORIGINAL_PHASES:
        cols += [
            f"{phase}_original_line",
            f"{phase}_original_branch",
            f"{phase}_original_mutation_full",
            f"{phase}_original_mutation_selective",
        ]
    return cols


def mutation_config() -> Dict[str, Any]:
    """Finishing's mutation engine settings: ``[mutation]`` over
    ``[coverage]``, with the resolved maximum per-mutant timeout."""
    cfg = dict(config_module.load_config("test_mutation"))
    parent = config_module.load_config("test_coverage")
    for key, value in parent.items():
        cfg.setdefault(key, value)
    cfg["mutation_max_timeout"] = config_module.resolve_mutation_max_timeout(cfg)
    return cfg


def run_finishing(
    *,
    target_file: str,
    source_dir: str,
    language: str,
    merged_test_path: Path,
    shared_setup_paths: List[Path],
    scratch_root: Path,
    config: Dict[str, Any],
    standard_coverage: Tuple[Optional[float], Optional[float]],
    kill_matrix_out: Optional[Path] = None,
    inherited_standard_km: Optional[dict] = None,
    log_fn: Optional[Any] = None,
) -> FinishingMetrics:
    """Finishing's measurement of the delivered suite: full and derived
    selective mutation, and coverage. The standard suite is mandatory and must
    work: a missing suite, one not green, or an engine that cannot start
    raises RuntimeError rather than give numbers measured against a broken
    verifier. The green check is the coverage the target's phases already
    measured (``standard_coverage``), so it adds no run. The one opt-out is
    ``measure_mutation = false`` (coverage only). The full operator catalogue
    runs, and the selective score (the engine's configured set) filters it.
    With the mutation phase end's matrix (``inherited_standard_km``) no
    engine session opens: Finishing takes that matrix."""
    from testuary.languages.mutation import close_mutation_session, mutation_cleanup_pending
    if not Path(merged_test_path).exists():
        raise RuntimeError(
            f"[finishing] the standard test suite is mandatory but is missing "
            f"at {merged_test_path}."
        )

    if standard_coverage[0] is None:
        # No line coverage: the run was not green or timed out, and every
        # final number would be measured against a broken verifier.
        raise RuntimeError(
            "[finishing] the standard test suite (this file's correctness "
            "verifier) did not run green or its final coverage run timed "
            "out -- its coverage came back unmeasurable, so the mutation / "
            "coverage numbers cannot be measured against it. Fix the "
            "standard suite or raise its timeout, then re-run."
        )

    mutation_mandatory = config_module.finishing_measures_mutation()
    inherited = inherited_standard_km if mutation_mandatory else None

    session = None
    scratch = ExitStack()
    run_mutation_on: Optional[Callable[[Path], Optional[Dict[str, Any]]]] = None
    selective_operators: Optional[set] = None
    if mutation_mandatory:
        try:
            cfg = dict(config)
            # The session calls log(msg, level), so a one-argument log_fn is adapted.
            mutation = languages.get_backend(language).mutation_runner(
                cfg, source_dir=source_dir,
                log=(level_tolerant_log(log_fn)
                     if log_fn is not None else None),
            )
            # Finishing's measurement records every killing test.
            mutation.attribution = "all"
            full_ops = mutation.full_operators()
            configured_ops = mutation.configured_operators()
            selective_operators = (
                set(configured_ops) if full_ops and configured_ops else None
            )
            if inherited is None:
                suites_root = Path(scratch.enter_context(
                    project.scratch_dir(prefix="testuary_mutation_suites_")))
                session = mutation.open(int(config["engine_workers"]),
                                        suite_dirs=(suites_root,))
                run_mutation_on = _session_mutation_fn(
                    session, suites_root, full_ops, target_file=target_file,
                    source_dir=source_dir, config=config, language=language,
                    log_fn=log_fn)
        except Exception as exc:
            # No fallback: the standard suite's mutation score is mandatory.
            if session is not None:
                close_mutation_session(session, primary=exc)
            scratch.close()
            if mutation_cleanup_pending(exc):
                raise
            raise RuntimeError(
                f"[finishing] the standard suite's mutation score is mandatory "
                f"but the mutation engine could not be started "
                f"({type(exc).__name__}: {exc}). Fix the engine and re-run, or "
                f"set [{config_module.phase_section('finishing')}] measure_mutation = false "
                f"to opt out of mutation metrics."
            ) from exc

    # The session closes without replacing an error that ended the measurement.
    # A cleanup that fails after it is reported, as in the phases' sessions.
    try:
        result = measure_finishing(
            scratch_root=scratch_root,
            merged_test_path=merged_test_path,
            shared_setup_paths=shared_setup_paths,
            run_mutation_on=run_mutation_on,
            standard_coverage=standard_coverage,
            selective_operators=selective_operators,
            kill_matrix_out=kill_matrix_out,
            inherited_standard_km=inherited,
            log_fn=log_fn,
        )
    except BaseException as exc:
        if session is not None:
            close_mutation_session(session, primary=exc)
        raise
    else:
        if session is not None:
            close_mutation_session(session)
        return result
    finally:
        scratch.close()


def _requested_seconds(session: Any, per_mutant_timeout: int) -> Optional[int]:
    """The derived seconds the session's requests carry, None where they
    carry the engine's own rule or the auto rule (``requested_timeout``). A
    runner that does not say carries the seconds."""
    requested = getattr(getattr(session, "runner", None), "requested_timeout", None)
    value = requested(per_mutant_timeout) if callable(requested) else per_mutant_timeout
    return value if isinstance(value, int) else None


def _requested_timeout(session: Any, per_mutant_timeout: int) -> str:
    """What the session's requests ask for as the per-mutant timeout, for a
    log line: the derived timeout, or the engine's own rule."""
    describe = getattr(getattr(session, "runner", None), "requested_timeout_text", None)
    return describe(per_mutant_timeout) if callable(describe) else f"{per_mutant_timeout}s"


def _session_mutation_fn(session, suites_root, operators, *, target_file, source_dir,
                         config, language, log_fn):
    """``make_finishing_mutation_fn`` for ``session`` over ``operators``,
    with the per-mutant and wall-clock limits ``config`` sets."""
    return make_finishing_mutation_fn(
        session=session,
        suites_root=suites_root,
        target_file=target_file,
        source_dir=source_dir,
        operators=operators,
        configured_per_mutant_timeout=config["per_mutant_timeout_seconds"],
        wall_clock_timeout=int(config["measurement_timeout_seconds"]),
        per_mutant_timeout_max=int(config["mutation_max_timeout"]),
        per_mutant_timeout_min=int(config["min_per_mutant_timeout_seconds"]),
        per_mutant_timeout_factor=float(config["per_mutant_timeout_factor"]),
        language=language,
        log_fn=log_fn,
    )


def make_finishing_mutation_fn(
    *,
    session: Any,
    suites_root: Optional[Path] = None,
    target_file: str,
    source_dir: str,
    operators: Optional[List[str]] = None,
    configured_per_mutant_timeout: object,
    wall_clock_timeout: int,
    per_mutant_timeout_max: int,
    per_mutant_timeout_min: int,
    per_mutant_timeout_factor: float,
    language: str = "python",
    log_fn: Optional[Any] = None,
) -> Callable[[Path], Optional[Dict[str, Any]]]:
    """The ``run_mutation_on`` callable for ``measure_finishing``: each call
    runs the live ``session`` on one clean per-suite directory and returns its
    parsed kill matrix, or None on any failure (empty columns, not an error).
    A session whose cleanup failed admits no further run, so its error ends
    the work. An interruption propagates. With ``suites_root``, the directory
    the session declared, each measured directory is copied into a fresh one there
    under its own name and measured from the copy.

    ``operators`` None runs the engine's configured catalogue. As in the
    loops, a positive ``configured_per_mutant_timeout`` fixes the per-mutant
    timeout. ``0`` scales it as ``per_mutant_timeout_factor`` times the
    suite's green run time, within ``[per_mutant_timeout_min,
    per_mutant_timeout_max]``."""

    def _log(msg: str) -> None:
        if log_fn is not None:
            log_fn(msg)

    @contextmanager
    def _declared(test_dir: Path):
        if suites_root is None:
            yield Path(test_dir)
            return
        parent = Path(tempfile.mkdtemp(prefix="suite_", dir=str(suites_root)))
        staged = parent / Path(test_dir).name
        shutil.copytree(str(test_dir), str(staged))
        try:
            yield staged
        finally:
            shutil.rmtree(parent, ignore_errors=True)

    def _green_limit(test_dir: Path):
        """The suite's test files and its per-mutant timeout, or None when its
        un-mutated run is not green. The same rule sizes a standalone run and
        each suite of a per-suite measurement."""
        # The un-mutated run sizes the automatic timeout and gates the mutation:
        # cosmic-ray scores every mutant killed when the baseline cannot run
        # green, a false 100% that looks like a real one.
        _log("[finishing] checking the un-mutated suite runs green + timing it")
        # The normal run, as every other verdict on this target. Several
        # files run in one session, as the tool runs them.
        backend = languages.get_backend(language)
        setup = {Path(path).resolve() for path in backend.shared_setup_files(str(test_dir))}
        test_paths = [str(path) for path in sorted(Path(test_dir).iterdir())
                      if path.is_file() and path.suffix in backend.test_file_extensions()
                      and path.resolve() not in setup]
        pre = _green_check_with_retry(
            backend, test_dir, source_dir, int(wall_clock_timeout), _log,
            instrument_target=target_file, test_paths=test_paths)
        if pre is None:
            return None
        if not pre.ok:
            _keep_names(test_dir, pre, timed_out=bool(getattr(pre, "timed_out", False)))
            _log(f"[finishing] un-mutated suite is NOT green "
                 f"(returncode={pre.returncode}), so no mutant is run: a failing "
                 "suite would kill every mutant. No mutation score.")
            return None
        pmt = config_module.fixed_per_mutant_timeout(configured_per_mutant_timeout)
        if pmt <= 0:
            pmt = config_module.per_mutant_timeout(
                "auto", pre.elapsed_s, cap_s=per_mutant_timeout_max,
                floor_s=per_mutant_timeout_min,
                multiplier=per_mutant_timeout_factor)
        # The requested seconds, None where a rule takes their place. The run
        # records what it applied (``per_mutant_timeout``).
        limits[str(test_dir)] = {"baseline_seconds": float(pre.elapsed_s or 0.0),
                                 "per_mutant_timeout_seconds": _requested_seconds(session, pmt),
                                 "per_mutant_timeout": None}
        return test_paths, pmt

    def _note_cut(measurement) -> None:
        # A measurement its clock cut notes the time limit in the phase that
        # ran it, Finishing included.
        from testuary import results

        cut = (getattr(measurement, "truncated", False)
               or str(getattr(getattr(measurement, "outcome", None), "value", "")) == "timed_out")
        phase = results.current_phase()
        if cut and phase:
            _log("WARNING: " + results.note(
                phase, "mutation.measurement_timeout_seconds", wall_clock_timeout))

    #: Per measured directory: the un-mutated run time, the requested per-mutant
    #: timeout (None where the engine's rule applies) and the timeout the
    #: engine states it applied.
    limits: Dict[str, Dict[str, Any]] = {}
    #: Each run the mutation tool's origin check stopped, in order: the suite,
    #: the reason and every stopping test's name.
    stops: List[Dict[str, Any]] = []
    #: Per measured directory, the id objects of the tests its last run's
    #: green check or check before the first mutant named, for a place where
    #: no model is active: a red green check or origin, or an instrument fault
    #: that names them. A timed-out run names none.
    named: Dict[str, Tuple[Any, ...]] = {}

    def _keep_names(test_dir: Path, failed: Any, *, timed_out: bool = False) -> None:
        """Keep the tests ``failed`` names for ``test_dir``, none for a
        timed-out run."""
        named[str(test_dir)] = () if timed_out else tuple(
            getattr(failed, "stopping_tests", ()) or ())

    def _stopped(name: str, reason: str, tests) -> str:
        """Log and keep why the origin check stopped suite ``name``: a test
        failed without a mutant, so no mutant ran. Returns the logged text."""
        tests = list(tests)
        text = f"{reason} (stopping tests: {', '.join(tests) if tests else 'none named'})"
        _log(f"[finishing] the mutation tool's origin check stopped {name}: {text}")
        stops.append({"suite": name, "reason": reason, "stopping_tests": tests})
        return text

    def _measure_once(test_dir: Path, attribution: Optional[str] = None) -> Optional[Dict[str, Any]]:
        with _declared(test_dir) as staged:
            result = _measure_staged(staged, attribution)
            if str(staged) in limits:
                limits[str(test_dir)] = limits[str(staged)]
            named[str(test_dir)] = named.pop(str(staged), ())
            return result

    def _measure_staged(test_dir: Path, attribution: Optional[str] = None) -> Optional[Dict[str, Any]]:
        named.pop(str(test_dir), None)
        limit = _green_limit(test_dir)
        if limit is None:
            return None
        test_paths, pmt = limit
        _log(f"[finishing] full-catalogue mutation on {Path(test_dir).name}: "
             f"per-mutant timeout {_requested_timeout(session, pmt)}")
        try:
            measurement = session.run(
                target=Path(target_file),
                test_dir=Path(test_dir),
                test_files=test_paths,
                operators=operators,
                per_mutant_timeout=pmt,
                wall_clock_timeout=wall_clock_timeout,
                # A run that overrides the configured attribution, as the
                # pooled run of the starter suites records every killer.
                **({"attribution": attribution} if attribution is not None else {}),
            )
        except Exception as exc:
            if mutation.mutation_cleanup_pending(exc):
                raise
            _log(f"[finishing] mutation run failed on {Path(test_dir).name}: {exc}")
            _keep_names(test_dir, exc)
            return None
        if getattr(measurement, "suite_failed", False):
            _keep_names(test_dir, measurement)
            _stopped(Path(test_dir).name, measurement.reason,
                     mutation.test_texts(measurement.stopping_tests))
            return None
        limits[str(test_dir)]["per_mutant_timeout"] = (
            measurement.timeout_rule() if hasattr(measurement, "timeout_rule") else None)
        _note_cut(measurement)
        fd, raw = tempfile.mkstemp(suffix="_km.json")
        os.close(fd)
        km_path = Path(raw)
        try:
            mutation.write_kill_matrix(
                measurement, km_path,
                level_tolerant_log(log_fn) if log_fn is not None
                else None)
            if km_path.exists() and km_path.stat().st_size > 0:
                return json.loads(km_path.read_text(encoding="utf-8"))
            return None
        except Exception as exc:
            _log(f"[finishing] kill-matrix capture failed on {Path(test_dir).name}: {exc}")
            return None
        finally:
            if km_path.exists():
                km_path.unlink()

    def _measure_per_suite(suites: Dict[str, Path]) -> Optional[Dict[str, Any]]:
        """Measure each suite directory alone against every mutant in one
        per-suite request, each with its own green-run timeout, as a standalone
        run would. Returns the union matrix, each measured suite's matrix, each
        requested suite's run outcome, the implementation, each green suite's
        timeout and the suites left unmeasured (not green, or measuring
        nothing: no score, never a zero), or None when nothing could be
        measured. A run the wall clock cut keeps its matrix and outcome. The
        tests a suite's checks named are ``named_of`` its directory."""
        with ExitStack() as staging:
            staged = {name: staging.enter_context(_declared(test_dir))
                      for name, test_dir in suites.items()}
            for test_dir in staged.values():
                named.pop(str(test_dir), None)
            try:
                return _measure_suites(staged)
            finally:
                for name, test_dir in suites.items():
                    named[str(test_dir)] = named.pop(str(staged[name]), ())

    def _measure_suites(suites: Dict[str, Path]) -> Optional[Dict[str, Any]]:
        entries, unmeasured = [], []
        for name, test_dir in suites.items():
            limit = _green_limit(test_dir)
            if limit is None:
                unmeasured.append(name)
                continue
            test_paths, pmt = limit
            entries.append((name, Path(test_dir), test_paths, pmt))
        if not entries:
            return None
        _log(f"[finishing] per-suite mutation of {len(entries)} suite(s), each alone")
        try:
            run = session.per_suite(target=Path(target_file), suites=entries,
                                    wall_clock_timeout=wall_clock_timeout, operators=operators)
        except (AttributeError, TypeError):
            raise
        except Exception as exc:
            if mutation.mutation_cleanup_pending(exc):
                raise
            _log(f"[finishing] per-suite mutation run failed: {exc}")
            return None
        own, outcomes, stopped = {}, {}, {}
        suite_limits = {name: limits.get(str(test_dir))
                        for name, test_dir, _paths, _pmt in entries}
        for name, suite_run in run.runs.items():
            outcomes[name] = suite_run.outcome.value
            _note_cut(suite_run)
            if suite_limits.get(name) is not None:
                suite_limits[name]["per_mutant_timeout"] = mutation.applied_timeout(suite_run)
            if not suite_run.measured:
                _keep_names(suites[name], suite_run)
                if suite_run.outcome is Outcome.TEST_FAILURES:
                    stopped[name] = _stopped(name, suite_run.reason, mutation.test_texts(
                        getattr(suite_run, "stopping_tests", ()) or ()))
                else:
                    _log(f"[finishing] suite {name} measured nothing "
                         f"({suite_run.outcome.value}: {suite_run.reason})")
                continue
            own[name] = run.kill_matrix(name)
        union = run.kill_matrix()
        if unmeasured:
            _with_suites_not_measured(union, unmeasured, own)
        return {"union": union, "suites": own, "outcomes": outcomes, "stopped": stopped,
                "implementation": run.implementation, "budgets": suite_limits,
                "unmeasured": sorted({*unmeasured, *(set(suites) - set(own))})}

    def _fn(test_dir: Path, *, attribution: Optional[str] = None) -> Optional[Dict[str, Any]]:
        return _measure_once(test_dir, attribution)

    #: Each suite of several measured alone in one request (starter suites).
    _fn.per_suite = _measure_per_suite
    #: The limits of a measured directory, None when it was not measured.
    _fn.limit_of = lambda test_dir: limits.get(str(test_dir))
    _fn.stops = stops
    #: The tests a measured directory's run named, as id objects, () for none.
    _fn.named_of = lambda test_dir: named.get(str(test_dir), ())
    return _fn
