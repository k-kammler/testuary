"""tests.csv's rows, one per target and run, with stable column names and
order, and the run, file and project summaries written from them. The
coverage phase, the mutation phase and finishing each contribute columns. A
row written into a file with other columns rewrites the file with the
current columns."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Iterator
from testuary import languages, phase_state, project
from testuary import measurement
import contextlib
import secrets
import tempfile
import threading
import csv
import statistics


#: The cells finishing's mutation measurement leaves empty when it gives no
#: score: mutation_score, the final scores and the mutation phase's snapshot
#: columns, which reuse finishing's measurement.
_WITHOUT_A_FINAL_SCORE = (
    "mutation_score",
    "finishing_mutation_score_full",
    "finishing_mutation_score_selective",
    "mutation_original_mutation_full",
    "mutation_original_mutation_selective",
)

TESTS_FIELDNAMES: List[str] = [
    # Identification
    "target_number", "run", "file_path", "score",
    # Coverage: line_coverage and branch_coverage are finishing's values,
    # else the coverage phase's last measurement.
    "line_coverage", "branch_coverage",
    # Mutation: ``mutation_score`` is finishing's score of the delivered suite
    # when that measurement ran (empty when it gave no score), else the
    # mutation phase's last measurement.
    "mutation_score",
    # The switch from the selective operator set to the full catalogue
    # (full_after_selective_goal): whether the loop switched, where
    # the selective score stood and the full catalogue's score at the switch.
    # Empty without a switch.
    "mutation_switched", "mutation_selective_final", "mutation_full_at_switch",
    # Per-file counts and lines of code
    "test_count_delivered", "mutation_test_count",
    "test_loc_delivered", "mutation_test_loc",
    # Finishing: assertion density, per test file
    "assertion_count_delivered", "mutation_assertion_count",
    "parametrize_count_delivered", "mutation_parametrize_count",
    "fixture_count_delivered", "mutation_fixture_count",
    # Finishing: cyclomatic complexity per test file (lizard)
    "test_complexity_sum_delivered", "test_complexity_max_delivered",
    # Finishing: target metrics
    "target_loc", "target_complexity",
    # Finishing: lint findings (ruff), only with ``measure_lint = true``.
    "lint_smells_target", "lint_smells_tests",
    # Finishing: the timing of the delivered suite (a warm-up and N
    # timed runs under the benchmark lock), only with
    # the suite timing's ``measure = true``. These columns never overwrite the
    # coverage and mutation cells. ``suite_timing_samples_ms`` holds the
    # timed runs, separated by semicolons.
    "suite_timing_median_ms", "suite_timing_samples_ms",
    "suite_timing_run_count", "suite_timing_warmup_count",
    # Where the timed runs ran: the processor set requested and the set a
    # process started like them was allowed to run on (not the processors
    # they used).
    "suite_timing_requested_cpus", "suite_timing_allowed_cpus",
    # Finishing's own measures of the delivered suite (full and selective
    # mutation per suite, merged counts and overlap, merged coverage) are
    # appended by write_tests_row from ``_finishing_columns()``.
    # The starter suites' generation: merged is the pool before the lossless
    # cover, delivered the keep-best winner on disk, as in the matrix files.
    "starter_status", "starter_suites_generated", "starter_suites_passing",
    # Passing starter suites without complete numbers: the coverage tool or
    # the mutation tool could not measure them. Each left the starter
    # suites. Its slot was generated again if attempts remained.
    "starter_suites_unmeasured_coverage", "starter_suites_unmeasured_mutation",
    "starter_merged_tests", "starter_delivered_tests", "starter_delivered_from",
    "starter_baseline_line", "starter_baseline_branch",
    # The starter suites' statistics before the merge, over green starter
    # suites only. Line and branch are measured per suite, each on its own.
    # Mutation is measured or derived as starter_suite_mutation_method says.
    # The per-slot counts and outcomes are in the starter archive's
    # starter_pool_attribution.json and starter_pool_derivation.json.
    "starter_suite_line_mean", "starter_suite_line_median",
    "starter_suite_line_std", "starter_suite_line_min", "starter_suite_line_max",
    "starter_suite_branch_mean", "starter_suite_branch_median",
    "starter_suite_branch_std", "starter_suite_branch_min",
    "starter_suite_branch_max",
    "starter_suite_mutation_mean", "starter_suite_mutation_median",
    "starter_suite_mutation_std", "starter_suite_mutation_min",
    "starter_suite_mutation_max",
    # How the starter_suite_mutation_* columns were obtained: measure (each
    # starter suite measured alone), derive (from one pooled run) or both
    # (measured, with the derivation recorded beside it). A derive or both
    # that could not derive reads measure. unavailable: the per-suite
    # measurement produced no result, so the columns are empty. Empty when
    # no mutation was measured.
    "starter_suite_mutation_method",
    "starter_requests", "starter_input_tokens", "starter_output_tokens",
    # Cached-prefix prompt tokens (provider prompt-cache hits) per phase:
    # a subset of the input tokens billed at the lower cache-hit rate.
    "starter_cache_hit_tokens",
    "starter_duration_s",
    # The consolidation stages of the coverage and mutation phases' ends, with
    # the same merged and delivered names. delivered_from names the keep-best
    # pick (merged, or the candidate that stood).
    "coverage_merged_tests", "coverage_delivered_tests",
    "coverage_delivered_from",
    "mutation_merged_tests", "mutation_delivered_tests",
    "mutation_delivered_from",
    # Per phase: the loops' requests and tool calls
    "coverage_turns", "mutation_turns",
    "coverage_requests", "mutation_requests",
    "coverage_duration_s", "mutation_duration_s", "finishing_duration_s",
    # The run's own wall-clock, and where it went. ``duration_s_target``
    # exceeds the sum of the run's phases by the work between them, which is
    # what ``time_other_s`` holds. The parts are summed over the run's phases:
    # a phase's time is attributed to the operation that spent it, so the same
    # second is never in two of these columns. ``time_preparation_s`` is the
    # measurement copies of the project (copy, bytecode seed and harvest,
    # close), kept out of the test, coverage and mutation times.
    # ``duration_s_file_shared`` is the file-level figure: the work the file's
    # runs share, outside every run, the same in each of the file's rows.
    "duration_s_target", "duration_s_file_shared",
    "time_model_s", "time_mutation_s", "time_coverage_s", "time_test_s",
    "time_merge_s", "time_cover_s", "time_preparation_s", "time_other_s",
    # The coverage and mutation phases' own time: model round trips and tool
    # runs, less than the phase, which also pays for its entry measurement
    # and its consolidation.
    "coverage_agent_duration_s", "mutation_agent_duration_s",
    # Per phase: input and output tokens over all its model requests, from
    # the loop's ``UsageTracker``. Empty for finishing, which uses no model.
    "coverage_input_tokens", "coverage_output_tokens",
    "coverage_cache_hit_tokens",
    "mutation_input_tokens", "mutation_output_tokens",
    "mutation_cache_hit_tokens",
    # Per-phase status
    "coverage_status", "mutation_status", "finishing_status",
    "error",
    # Request accounting. The request columns
    # above count requests sent, every attempt. Model requests are
    # the loops' turns and the starter attempts charged against the phase's limit. The two
    # differ only in a run with transport failures. Unknown-usage requests
    # are those whose token usage the provider did not report, so the token
    # columns undercount by their usage. An empty cell is unknown, not zero.
    "starter_logical_requests", "starter_unknown_usage_requests",
    "coverage_logical_requests", "coverage_unknown_usage_requests",
    "mutation_logical_requests", "mutation_unknown_usage_requests",
    # Optional structural counts with status/reason/parser.
    "source_observations",
    # Requests with reported token usage whose cache split the provider did
    # not report. The cache_hit_tokens columns sum only the others.
    "starter_unknown_cache_requests", "coverage_unknown_cache_requests",
    "mutation_unknown_cache_requests",
    # Why a suite ended below its configured minimum, a finished result a
    # resume does not continue. ``error`` can also carry other notes, such as
    # a coverage phase ended by measurements without improvement.
    "below_minimum_reason",
]

TESTS_FLATTEN_FIELDS: List[str] = ["error"]


# Helpers

def _get(obj: Any, name: str, default: Any = None) -> Any:
    """The attribute, or ``default`` when it is missing or None."""
    if obj is None:
        return default
    val = getattr(obj, name, default)
    return default if val is None else val


def _to_dict(obj: Any) -> Dict[str, Any]:
    """A result's ``to_csv_dict()``, or an empty dict for None (an absent
    phase result), whose columns then stay empty.
    """
    if obj is None:
        return {}
    return obj.to_csv_dict()


def _finishing_columns() -> List[str]:
    """tests.csv's ``final_*`` columns, finishing's ``FinishingMetrics``
    fields, so a new field becomes a column without an edit here. Imported
    when called, so this module does not import the phases at load."""
    from testuary.measurement import finishing_csv_columns
    return finishing_csv_columns()


def _phase_finishing_columns() -> List[str]:
    """tests.csv's per-phase snapshot columns: coverage and full and
    selective mutation of the starter, coverage and mutation suites, named
    as the finishing module names them."""
    from testuary.measurement import phase_finishing_csv_columns
    return phase_finishing_csv_columns()


# The tests.csv writer

def _read_csv(path: Path) -> Tuple[List[str], List[Dict[str, Any]]]:
    """A CSV file's columns and rows, read as tests.csv is written (UTF-8
    with a byte order mark)."""
    with open(path, "r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        rows = [dict(row) for row in reader]
        return list(reader.fieldnames or []), rows


def _rewrite_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    """Write ``rows`` under a header in place of ``path`` atomically, through
    a temporary file beside it, so a crash keeps the file as it was."""
    tmp = path.with_name(f"{path.name}.{secrets.token_hex(4)}.tmp")
    handle = open(tmp, "x", newline="", encoding="utf-8-sig")
    try:
        with handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\r\n",
                                    extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(OSError):
            tmp.unlink(missing_ok=True)


def _read_tests_row(
    path: Path, target_number: int, run_idx: int,
) -> Optional[Dict[str, Any]]:
    """tests.csv's row for target ``target_number`` and run ``run_idx`` (its ``run``
    column, from 0) as a dict, or None. ``write_tests_row`` starts from it
    with ``preserve_existing``, so a measurement of a reused suite keeps the
    columns of the phases it did not run again."""
    if not path.exists():
        return None
    return next((row for row in _read_csv(path)[1]
                 if str(row.get("target_number")) == str(target_number)
                 and str(row.get("run")) == str(run_idx)), None)


def _overlay_durable_durations(
    row: Dict[str, Any], target_dir: Path, *, run_idx: int,
) -> set:
    """Write the phase durations the target's own record holds, which
    survives a crash and a resume: a phase that ran in two segments reports
    both, and a reused phase what producing it cost. The time already in
    ``row`` stays where no segment was recorded, as after a crash before the
    phase. A time that is not known (``phase_state.unknown_times``) is
    written empty, never partial, and returned among the cells this write
    empties on purpose."""
    from testuary import phase_state

    unknown = phase_state.unknown_times(target_dir, run_idx)
    emptied = {"starter_duration_s"} if "starter" in unknown else set()
    if "target" in unknown:
        emptied |= {"duration_s_target",
                    *(f"time_{part}_s" for part in (*phase_state.TIME_PARTS, "other"))}
    for phase in ("starter", "coverage", "mutation", "finishing"):
        seconds = phase_state.output_seconds(
            target_dir, phase, run_idx=run_idx)
        if seconds is not None:
            row[f"{phase}_duration_s"] = seconds

    target_seconds = phase_state.target_seconds(target_dir, run_idx)
    if target_seconds is not None:
        row["duration_s_target"] = target_seconds
    for part, seconds in phase_state.target_time_parts(target_dir, run_idx).items():
        row[f"time_{part}_s"] = seconds
    shared = phase_state.shared_seconds(target_dir)
    if shared is not None:
        row["duration_s_file_shared"] = shared
    for column in emptied:
        row[column] = ""
    return emptied


def refresh_tests_row_durations(
    results_dir: Path, *, workspace: str, target_number: int,
) -> None:
    """Write a target's timing cells once its pipeline segment closes.
    ``write_tests_row`` runs inside that segment, so its results show at
    once, but the target's total exists only after it. This second atomic
    write changes only the timing cells.
    """
    path = results_dir / "tests.csv"
    if not path.is_file():
        return
    fieldnames, rows = _read_csv(path)
    if not fieldnames:
        return

    changed = False
    for row in rows:
        if str(row.get("target_number")) != str(target_number):
            continue
        try:
            run_idx = int(row.get("run") or 0)
        except (TypeError, ValueError):
            continue
        _overlay_durable_durations(
            row, results_dir / workspace, run_idx=run_idx,
        )
        changed = True
    if changed:
        _rewrite_csv(path, fieldnames, rows)


def _suite_column(measure: str, role: str) -> str:
    """The tests.csv column of a suite's measure: the mutation phase's suite
    names its phase first, as every phase's column does. The delivered
    suite names its role last."""
    return f"mutation_{measure}" if role == "mutation" else f"{measure}_{role}"


#: The starter phase's columns, by the attribute of its result that fills
#: them. Its request and token columns follow from ``_usage_columns``.
_STARTER_COLUMNS = (
    ("starter_status", "status"),
    ("starter_suites_generated", "n_generated"),
    ("starter_suites_passing", "n_green"),
    ("starter_suites_unmeasured_coverage", "n_unmeasured_coverage"),
    ("starter_suites_unmeasured_mutation", "n_unmeasured_mutation"),
    ("starter_merged_tests", "pool_tests"),
    ("starter_delivered_tests", "selected_tests"),
    ("starter_baseline_line", "baseline_line_coverage"),
    ("starter_baseline_branch", "baseline_branch_coverage"),
    ("starter_suite_mutation_method", "suite_mutation_method"),
    ("starter_requests", "requests_used"),
)

#: The coverage phase's coverage and the delivered file's measures.
_COVERAGE_COLUMNS = (
    ("line_coverage", "line_coverage"),
    ("branch_coverage", "branch_coverage"),
    ("test_count_delivered", "test_count_standard"),
    ("test_loc_delivered", "test_loc_standard"),
    ("target_loc", "target_loc"),
    ("target_complexity", "target_complexity"),
)

#: The mutation phase's switch to the full catalogue and its file's measures.
#: mutation_score is written only when the loop has one.
_MUTATION_COLUMNS = (
    ("mutation_switched", "mutation_switched"),
    ("mutation_selective_final", "mutation_selective_final"),
    ("mutation_full_at_switch", "mutation_full_at_switch"),
    ("mutation_test_count", "test_count_mutation"),
    ("mutation_test_loc", "test_loc_mutation"),
)

#: Finishing's values that replace the loops' where it measured them, by
#: its key. Its own measures follow from ``_finishing_columns()`` and
#: ``_phase_finishing_columns()``, so a new measure needs no edit here.
_FINISHING_COLUMNS = (
    ("line_coverage", "coverage_final_line"),
    ("branch_coverage", "coverage_final_branch"),
    ("target_complexity", "target_complexity_sum"),
    ("finishing_duration_s", "finishing_duration_s"),
) + tuple((name, name) for name in (
    # Per test file: structural measures and complexity (both roles).
    "assertion_count_delivered", "mutation_assertion_count",
    "parametrize_count_delivered", "mutation_parametrize_count",
    "fixture_count_delivered", "mutation_fixture_count",
    "test_loc_delivered", "mutation_test_loc",
    "test_count_delivered", "mutation_test_count",
    "test_complexity_sum_delivered", "test_complexity_max_delivered",
    "target_loc",
    # The optional measurements: lint findings and the suite timing, which
    # never overwrites coverage and mutation cells, and a run without it
    # never empties them.
    "lint_smells_target", "lint_smells_tests",
    "suite_timing_median_ms", "suite_timing_samples_ms",
    "suite_timing_run_count", "suite_timing_warmup_count",
    "suite_timing_requested_cpus", "suite_timing_allowed_cpus",
))


def _usage_columns(phase: str, *loop_columns: Tuple[str, str]) -> List[Tuple[str, str]]:
    """A model phase's request and token columns, ``<phase>_<attribute>``,
    and a loop's own columns given as ``(column, attribute)``."""
    attributes = ("logical_requests", "unknown_usage_requests", "unknown_cache_requests",
                  "input_tokens", "cache_hit_tokens", "output_tokens")
    return ([(f"{phase}_{column}", attribute) for column, attribute in loop_columns]
            + [(f"{phase}_{attribute}", attribute) for attribute in attributes])


#: The coverage and mutation phases' tool calls, requests and own time.
_LOOP_COLUMNS = (("turns", "agent_iterations"), ("requests", "model_requests"),
                 ("agent_duration_s", "agent_duration_s"))


def _copy_attributes(row: Dict[str, Any], result: Any, columns) -> None:
    """Fill ``row`` from a phase result by ``(column, attribute)``, empty
    where the result carries none: absent is not zero."""
    for column, attribute in columns:
        row[column] = _get(result, attribute, "")


def _join_error(row: Dict[str, Any], error: Any) -> None:
    """Add ``error`` to the row's ``error`` cell, joined by " / "."""
    if error:
        existing = row.get("error", "")
        row["error"] = f"{existing} / {error}" if existing else error


def _loop_error(result: Any) -> Any:
    """A loop's error for the ``error`` cell: none when a request, token or
    time limit the settings set ended it, a designed ending its status names."""
    return "" if _get(result, "status", "") == "budget_exhausted" else _get(result, "error", "")


def write_tests_row(
    results_dir: Path,
    *,
    workspace: str,
    target_number: int,
    file_path: str,
    test_result: Optional[Any] = None,
    starter_result: Optional[Any] = None,
    mutation_result: Optional[Any] = None,
    finishing_result: Optional[Any] = None,
    work_tests: Optional[Path] = None,
    phase_durations: Optional[Dict[str, float]] = None,
    run_idx: int = 0,
    preserve_existing: bool = False,
    below_minimum_reason: Optional[str] = None,
    consolidation_stats: Optional[Dict[str, Dict[str, Any]]] = None,
    crashed: Optional[Tuple[Optional[str], str]] = None,
    coverage_reused: bool = False,
    refused: Optional[Tuple[str, str]] = None,
) -> None:
    """Append one target file's row to tests.csv. All result objects are optional.
    None passes its columns through as empty CSV cells, and a count, token,
    request, iteration or duration a result does not carry is written empty:
    absent is not zero.

    ``below_minimum_reason`` marks a target whose suite ended below the
    configured coverage or mutation minimum: a mutation phase that never ran
    reads ``skipped_below_minimum``, and the reason (the metric, the value
    and the minimum) goes to ``below_minimum_reason`` and ``error``.

    ``crashed`` is the phase that raised (None when no phase was running)
    and the error when the test pipeline crashed: that phase's status reads
    ``crashed`` and the error lands in ``error``.

    ``coverage_reused`` says the caller reused the coverage phase instead of
    running it, so a row without its result reads ``reused``. A coverage
    phase that did not run (the starter suites failed, the pipeline crashed)
    leaves the status empty.

    ``refused`` is a phase whose reuse it refused and why (its restored
    suite missing, not collecting or unlike its record): that phase's status
    reads ``reuse_refused`` and the reason joins ``error``.

    The row is built in this order, a later source winning on a key: the
    identification and the ``score`` column, the coverage phase
    (``test_result``) with the delivered file's measures, the mutation phase
    (``mutation_result``) with mutation_score, and finishing
    (``finishing_result``), whose coverage replaces the coverage phase's, and
    its structural measures.

    ``preserve_existing`` starts the row from the earlier ``(target_number, run_idx)``
    row. A measurement of a reused suite (``--reuse coverage``, ``mutation``
    or ``test``) passes None for the phases it did not run again, whose
    columns are then kept, and finishing's new values fill in. A Finishing
    result with a mutation measurement writes each of Finishing's metrics
    columns, empty where that measurement gave no value, and a write with a
    coverage, mutation or Finishing result writes ``error`` afresh. A fresh
    run builds the row from nothing.
    """
    row: Dict[str, Any] = {}
    preserved_row: Dict[str, Any] = {}
    # Cells this write empties on purpose, which a preserved row must not
    # refill, such as mutation_score when finishing's measurement did not
    # finish.
    cleared: set = set()
    if preserve_existing:
        existing = _read_tests_row(results_dir / "tests.csv", target_number, run_idx)
        if existing is not None:
            preserved_row = existing
            row.update(existing)
    row.update({
        "target_number": target_number,
        # The run as its directory names it: 0 for r0.
        "run": run_idx,
        "file_path": project.relativize_path(file_path),
    })
    if any(result is not None for result in (test_result, mutation_result, finishing_result)):
        # The phases this write records give the error afresh, so an earlier
        # attempt's error does not stay beside their statuses.
        row["error"] = ""
        cleared.add("error")

    # The starter suites' generation: every attempt, failed ones included,
    # counts in the request and token numbers.
    if starter_result is not None:
        _copy_attributes(row, starter_result,
                         _STARTER_COLUMNS + tuple(_usage_columns("starter")))
        recorded_winner = _get(starter_result, "winner", "")
        row["starter_delivered_from"] = recorded_winner or (
            "single_suite"
            if _get(starter_result, "baseline_from_single_suite", False)
            else "merged"
        )
        # The starter suites' statistics before the merge: each {mean, median,
        # std, min, max} dict in its five columns, "" for a value none gave.
        for _metric, _attr in (
            ("line", "suite_line_stats"),
            ("branch", "suite_branch_stats"),
            ("mutation", "suite_mutation_stats"),
        ):
            _d = _get(starter_result, _attr, {}) or {}
            for _s in ("mean", "median", "std", "min", "max"):
                _v = _d.get(_s) if isinstance(_d, dict) else None
                row[f"starter_suite_{_metric}_{_s}"] = (
                    round(float(_v), 4) if _v is not None else "")
        _duration = _get(starter_result, "duration_s", None)
        row["starter_duration_s"] = (
            "" if _duration is None else round(float(_duration), 3))

    # The consolidation stages of the coverage and mutation phases: merged and
    # delivered test counts and the keep-best winner, by phase. A phase that
    # did not run leaves its columns empty.
    for _phase, _stats in (consolidation_stats or {}).items():
        for _k in ("merged_tests", "delivered_tests", "delivered_from"):
            if _stats.get(_k) is not None:
                row[f"{_phase}_{_k}"] = _stats[_k]

    # The coverage phase: coverage, the delivered file's measures, requests
    # and tool calls. mutation_score comes from the mutation phase below.
    if test_result is not None:
        _copy_attributes(row, test_result,
                         _COVERAGE_COLUMNS + tuple(_usage_columns("coverage", *_LOOP_COLUMNS)))
        row["coverage_status"] = _get(test_result, "status", "unknown")
        row["error"] = _loop_error(test_result)
    elif coverage_reused:
        row["coverage_status"] = "reused"

    # The mutation phase: mutation_score, the mutation file's measures,
    # requests and tool calls. Errors are joined by " / ". The field is in the
    # flatten list.
    if mutation_result is not None:
        new_mut = _get(mutation_result, "mutation_score", None)
        if new_mut is not None:
            row["mutation_score"] = new_mut
        _copy_attributes(row, mutation_result,
                         _MUTATION_COLUMNS + tuple(_usage_columns("mutation", *_LOOP_COLUMNS)))
        row["mutation_status"] = _get(mutation_result, "status", "unknown")
        _join_error(row, _loop_error(mutation_result))

    if crashed is not None:
        phase, error = crashed
        if phase is not None:
            row[f"{phase}_status"] = "crashed"
        row["error"] = error

    if refused is not None:
        phase, reason = refused
        row[f"{phase}_status"] = "reuse_refused"
        _join_error(row, reason)

    if below_minimum_reason:
        row["below_minimum_reason"] = below_minimum_reason
        if mutation_result is None and not row.get("mutation_status"):
            row["mutation_status"] = "skipped_below_minimum"
        _join_error(row, below_minimum_reason)

    # Finishing: its coverage, structural measures and optional measurements
    # of the delivered suite replace the coverage phase's coverage, complexity
    # and target_loc where present.
    if finishing_result is not None:
        fin = _to_dict(finishing_result)
        if "coverage_final_line" in fin and fin["coverage_final_line"] is None:
            # Finishing's coverage run gave no value (it timed out or ran
            # red). The coverage phase's values describe another suite, so the
            # coverage cells stay empty.
            for key in ("line_coverage", "branch_coverage"):
                row[key] = ""
                cleared.add(key)
        own = [(k, k) for k in _finishing_columns() + _phase_finishing_columns()]
        for column, key in _FINISHING_COLUMNS + tuple(own):
            if fin.get(key) is not None:
                row[column] = fin[key]
        if fin.get("finishing_mutation_measurement") is not None:
            # A new mutation measurement describes itself alone: a column of
            # it without a value is emptied, so no earlier measurement's lower
            # bound or counts stay beside it. A write that measured no
            # mutation, such as a timing pass, keeps the earlier ones.
            for key in _finishing_columns():
                if fin.get(key) is None:
                    row[key] = ""
                    cleared.add(key)
        # mutation_score is finishing's score of the delivered suite over the
        # full catalogue, so it matches the kept kill matrix, not the mutation
        # phase's best. A finishing measurement without a score (its catalogue
        # did not resolve, or it gave no matrix) leaves mutation_score and the
        # final scores empty: no loop value, partial ratio, placeholder or
        # earlier score stands in. A write without finishing's mutation
        # measurement keeps what the row had.
        if fin.get("finishing_mutation_measurement"):
            final_score = fin.get("finishing_mutation_score_full")
            if final_score is not None:
                row["mutation_score"] = final_score
            else:
                for key in _WITHOUT_A_FINAL_SCORE:
                    row[key] = ""
                    cleared.add(key)
        if fin.get("finishing_status"):
            row["finishing_status"] = fin["finishing_status"]
        _join_error(row, fin.get("finishing_error"))

    # The orchestrator's time fills ``<phase>_duration_s`` when the row has
    # none (a failed phase that returned None, an error inside the loop), so
    # a failed phase's time is reported too. With ``workspace`` the phase
    # record below wins where it holds a time.
    durations = phase_durations or {}
    for _phase in ("coverage", "mutation", "finishing"):
        if not row.get(f"{_phase}_duration_s") and durations.get(_phase):
            row[f"{_phase}_duration_s"] = round(durations[_phase], 3)

    observations = {}
    final_observations = {}
    for result in (test_result, mutation_result):
        observations.update(_get(result, "source_observations", {}))
    if finishing_result is not None:
        final_fields = _to_dict(finishing_result)
        final_observations = final_fields.get("source_observations", {})
        observations.update(final_observations)
        # An attempted structural measurement in finishing replaces stale
        # complexity too. A timing-only update has no source observations.
        for role in final_observations:
            fields = ({"target_complexity": "target_complexity_sum"} if role == "target"
                      else {_suite_column(f"test_complexity_{stat}", role): f"test_complexity_{stat}_{role}"
                            for stat in ("sum", "max")})
            for key, source in fields.items():
                value = final_fields.get(source)
                row[key] = value if value is not None else ""
                if value is None:
                    cleared.add(key)

    # A reuse finds the delivered path through the language backend. A
    # timing-only update with kept counts is no new source observation.
    if work_tests is not None and "delivered" not in observations and (
        test_result is None or row.get("test_count_delivered") in (None, "")
    ) and (
        not preserve_existing or row.get("test_count_delivered") in (None, "")
    ):
        backend = languages.get_backend(languages.detect_language(file_path))
        standard_path = (backend.find_target_test_file(work_tests, file_path)
                         if backend is not None else None)
        observations["delivered"] = measurement.record(
            measurement.observe_file(standard_path, backend=backend),
        )
        if "target" not in observations:
            observations["target"] = measurement.record(measurement.observe_file(
                file_path, backend=backend, read_text=project.read_project_text,
            ))
            row["target_complexity"] = measurement.compute_file_complexity(file_path)

    if observations:
        prior_observations = row.get("source_observations", "")
        try:
            combined = json.loads(prior_observations) if prior_observations else {}
        except (TypeError, ValueError):
            combined = {}
        combined.update(observations)
        row["source_observations"] = json.dumps(combined, sort_keys=True)
        for role, observed in observations.items():
            fields = ({"target_loc": "code_lines"} if role == "target" else {
                _suite_column("test_loc", role): "code_lines",
                _suite_column("test_count", role): "tests",
                _suite_column("assertion_count", role): "assertions",
                _suite_column("fixture_count", role): "fixtures",
                _suite_column("parametrize_count", role): "parametrized_cases",
            })
            for key, metric in fields.items():
                value = observed.get(metric)
                row[key] = value if value is not None else ""
                if value is None:
                    cleared.add(key)
            if observed.get("status") != "completed":
                keys = (["target_complexity"] if role == "target" else [
                    _suite_column("test_complexity_sum", role),
                    _suite_column("test_complexity_max", role),
                ])
                for key in keys:
                    row[key] = ""
                    cleared.add(key)

    # The ``score`` column: the mutation score, 0 for a failed suite, else
    # empty. A preserved row (preserve_existing) holds mutation_score as text,
    # so it is read back as a number first.
    mut = row.get("mutation_score")
    if isinstance(mut, str):
        try:
            mut = float(mut)
        except ValueError:
            mut = None
    if isinstance(mut, (int, float)):
        row["score"] = mut
    elif "mutation_score" in cleared:
        row["score"] = ""
        cleared.add("score")
    elif row.get("coverage_status") == "tests_failed":
        row["score"] = 0
    else:
        row["score"] = ""

    cleared |= _overlay_durable_durations(row, results_dir / workspace, run_idx=run_idx)

    # A preserved row keeps an earlier value where this write leaves the
    # column empty (a crash row holds only the crashed status, a partial run
    # touches some phases). This write's values still win.
    if preserve_existing and preserved_row:
        for key, prior in preserved_row.items():
            if key in cleared:
                continue
            if row.get(key) in (None, "") and prior not in (None, ""):
                row[key] = prior

    append_csv_row(
        results_dir / "tests.csv",
        TESTS_FIELDNAMES + _finishing_columns() + _phase_finishing_columns(),
        row,
        flatten_text_fields=TESTS_FLATTEN_FIELDS,
        dedup_key=("target_number", "run"),
    )


# Appending a row

def append_csv_row(
    csv_path,
    fieldnames: List[str],
    row: Dict[str, Any],
    flatten_text_fields: Optional[List[str]] = None,
    dedup_key: Optional[Tuple[str, ...]] = None,
) -> None:
    """Append one row to ``csv_path``, with a header when the file is new,
    so the rows of a run can be followed as they come (``tail -f
    tests.csv``). A plain append is not atomic: a crash while writing loses
    at most that row.

    With ``dedup_key``, the key's columns such as ``("target_number", "run")``, a
    row with the same key is replaced: the file is rewritten
    atomically with the new row last. ``flatten_text_fields`` names columns
    whose line breaks become ``" / "``, such as ``["error"]``.
    """
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    flatten = set(flatten_text_fields or [])

    def _flat(text):
        if text is None:
            return ""
        if not isinstance(text, str):
            return text
        return text.replace("\r\n", " / ").replace("\r", " / ").replace("\n", " / ")

    cleaned = {}
    for k in fieldnames:
        v = row.get(k)
        if v is None:
            cleaned[k] = ""
        elif k in flatten:
            cleaned[k] = _flat(v)
        else:
            cleaned[k] = v

    file_existed = csv_path.exists() and csv_path.stat().st_size > 0

    if dedup_key is not None and file_existed:
        # Rewrite the file without an earlier row of the same key, then add
        # the new one, atomically, so a crash keeps the existing rows.
        new_key = tuple(str(cleaned.get(col, "")) for col in dedup_key)
        try:
            existing = _read_csv(csv_path)[1]
        except OSError:
            existing = []
        kept = [row for row in existing
                if tuple(str(row.get(col, "")) for col in dedup_key) != new_key]
        _rewrite_csv(csv_path, fieldnames, kept + [cleaned])
        return

    with open(csv_path, "a", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f, fieldnames=fieldnames, lineterminator="\r\n",
            extrasaction="ignore",
        )
        if not file_existed:
            writer.writeheader()
        writer.writerow(cleaned)


def parse_csv_field(raw: str) -> List[str]:
    """Split a comma-separated config value into a list of stripped
    tokens."""
    if not raw:
        return []
    return [tok.strip() for tok in raw.split(",") if tok.strip()]


#: The notes' file in a run directory.
FILE = "time_limits.json"

_LOCK = threading.Lock()
_ACTIVE: Optional[Path] = None
_PHASE: Optional[str] = None

#: What reaches each limit, for the sentence.
_WHAT = {
    "measurement_timeout_seconds": "The mutation measurement",
    "suite_run_timeout_seconds": "A test run",
    "coverage_timeout_seconds": "The final coverage run",
}
#: What follows from reaching it, when something does.
_EFFECT = {
    "measurement_timeout_seconds": " Untested mutants count as not killed.",
}


@contextlib.contextmanager
def collecting(run_dir: Path) -> Iterator[None]:
    """Note the limits reached while the block runs in ``run_dir``."""
    global _ACTIVE
    previous, _ACTIVE = _ACTIVE, Path(run_dir)
    try:
        yield
    finally:
        _ACTIVE = previous


@contextlib.contextmanager
def in_phase(phase: str) -> Iterator[None]:
    """Name ``phase`` as the one a shared measurement belongs to while the
    block runs (``current_phase``)."""
    global _PHASE
    previous, _PHASE = _PHASE, phase
    try:
        yield
    finally:
        _PHASE = previous


def current_phase() -> Optional[str]:
    """The phase whose work is running, or None outside a phase."""
    return _PHASE


def sentence(setting: str, value: float, count: int = 1) -> str:
    """The plain sentence for a limit reached ``count`` times."""
    name = setting.rsplit(".", 1)[-1]
    times = "" if count == 1 else f" {count} times"
    return (f"{_WHAT.get(name, 'A run')} reached its time limit{times} "
            f"({setting} = {value:g} s).{_EFFECT.get(name, '')}")


def note(phase: str, setting: str, value: float) -> str:
    """Record that ``phase`` reached the limit ``setting`` = ``value`` seconds
    and return the sentence for the console."""
    run_dir = _ACTIVE
    if run_dir is not None:
        with _LOCK:
            notes = _read(run_dir)
            notes.append({"phase": phase, "setting": setting, "value": value})
            run_dir.mkdir(parents=True, exist_ok=True)
            handle, temporary = tempfile.mkstemp(prefix=".time_limits.", suffix=".tmp",
                                                 dir=str(run_dir))
            try:
                with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
                    json.dump(notes, stream, indent=2)
                    stream.write("\n")
                os.replace(temporary, run_dir / FILE)
            finally:
                if os.path.exists(temporary):
                    os.remove(temporary)
    return sentence(setting, value)


def _read(run_dir: Path) -> List[Dict[str, Any]]:
    try:
        notes = json.loads((Path(run_dir) / FILE).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    return notes if isinstance(notes, list) else []


def warnings(run_dir: Path) -> List[Dict[str, Any]]:
    """The run's warnings: the time limits it reached, one per phase and
    setting in the order first reached, with the value, how often and the
    sentence, then each loop one of its limits stopped before its goal
    (``loop_limits``). Empty for a run that reached no limit."""
    grouped: Dict[tuple, Dict[str, Any]] = {}
    for item in _read(run_dir):
        if not isinstance(item, dict):
            continue
        key = (str(item.get("phase")), str(item.get("setting")))
        entry = grouped.setdefault(key, {
            "kind": "time_limit", "phase": key[0], "setting": key[1],
            "value": item.get("value"), "unit": "s", "count": 0})
        entry["count"] += 1
    for entry in grouped.values():
        entry["text"] = sentence(entry["setting"], entry["value"], entry["count"])
    return list(grouped.values()) + loop_limits(run_dir)


#: A loop's limits by the setting its stop names: the limit in words and the
#: value's unit.
_LOOP_LIMITS = {
    "max_requests_per_phase": ("request limit", None),
    "max_input_tokens_per_phase": ("input token limit", None),
    "max_output_tokens_per_phase": ("output token limit", None),
    "max_total_tokens_per_phase": ("token limit", None),
    "max_wall_clock_seconds_per_phase": ("time limit", "s"),
}
#: The setting and value in a limit's stop, such as
#: "max_wall_clock_seconds_per_phase (120s)".
_LOOP_STOP = re.compile(r"UsageLimitExceeded: (\w+) \((\d+(?:\.\d+)?)s?\)")


def loop_limits(run_dir: Path) -> List[Dict[str, Any]]:
    """A warning for each loop of the run in ``run_dir`` that one of its
    limits stopped before its goal, from the stop the file's phase record
    holds: the setting and its value when the stop names them. A loop that
    reached its goal ends as a success."""
    run_dir = Path(run_dir)
    if not (run_dir.name[:1] == "r" and run_dir.name[1:].isdigit()):
        return []
    state = phase_state.load(run_dir.parent)
    found = []
    for phase in ("coverage", "mutation"):
        metrics = phase_state.phase_metrics(state, phase, int(run_dir.name[1:]))
        if metrics.get("status") != "budget_exhausted":
            continue
        stop = _LOOP_STOP.search(str(metrics.get("error") or ""))
        known = _LOOP_LIMITS.get(stop.group(1)) if stop else None
        entry: Dict[str, Any] = {"kind": "loop_limit", "phase": phase, "setting": None,
                                 "value": None, "unit": None, "count": 1}
        if known is None:
            entry["text"] = f"The {phase} phase stopped at one of its limits before reaching its goal."
        else:
            words, unit = known
            number = float(stop.group(2))
            entry.update(setting=f"{phase}.{stop.group(1)}",
                         value=int(number) if number.is_integer() else number, unit=unit)
            entry["text"] = (f"The {phase} phase stopped at its {words} before reaching its goal "
                             f"({entry['setting']} = {stop.group(2)}{' s' if unit else ''}).")
        found.append(entry)
    return found


SUMMARY = "summary.json"
#: The final values a summary reports, named as in tests.csv.
VALUES = ("line_coverage", "branch_coverage", "mutation_score")
#: The values a fully measured repetition has measured.
FULLY_MEASURED_WITH = ("line_coverage", "mutation_score")
#: The phase outcomes a run summary carries, named as in tests.csv.
STATUSES = ("starter_status", "coverage_status", "mutation_status")
#: Each phase's values for the suite it delivered and its model use and
#: time, by the summary's name and the tests.csv column: the result-file
#: reference describes each column.
PHASE_VALUES = {
    "starter": {"line_coverage": "starter_original_line", "branch_coverage": "starter_original_branch",
                "mutation_score": "starter_original_mutation_full",
                "mutation_score_selective": "starter_original_mutation_selective",
                "tests": "starter_delivered_tests", "logical_requests": "starter_logical_requests",
                "physical_requests": "starter_requests", "input_tokens": "starter_input_tokens",
                "output_tokens": "starter_output_tokens",
                "cached_input_tokens": "starter_cache_hit_tokens", "seconds": "starter_duration_s"},
    "coverage": {"line_coverage": "coverage_original_line", "branch_coverage": "coverage_original_branch",
                 "mutation_score": "coverage_original_mutation_full",
                 "mutation_score_selective": "coverage_original_mutation_selective",
                 "tests": "coverage_delivered_tests", "logical_requests": "coverage_logical_requests",
                 "physical_requests": "coverage_requests", "input_tokens": "coverage_input_tokens",
                 "output_tokens": "coverage_output_tokens",
                 "cached_input_tokens": "coverage_cache_hit_tokens", "turns": "coverage_turns",
                 "seconds": "coverage_duration_s"},
    "mutation": {"line_coverage": "mutation_original_line", "branch_coverage": "mutation_original_branch",
                 "mutation_score": "mutation_original_mutation_full",
                 "mutation_score_selective": "mutation_original_mutation_selective",
                 "tests": "mutation_delivered_tests", "logical_requests": "mutation_logical_requests",
                 "physical_requests": "mutation_requests", "input_tokens": "mutation_input_tokens",
                 "output_tokens": "mutation_output_tokens",
                 "cached_input_tokens": "mutation_cache_hit_tokens", "turns": "mutation_turns",
                 "seconds": "mutation_duration_s"},
}
#: Finishing's own values and time.
FINAL_VALUES = {"line_coverage": "finishing_line_coverage",
                "branch_coverage": "finishing_branch_coverage",
                "mutation_score": "finishing_mutation_score_full",
                "mutation_score_selective": "finishing_mutation_score_selective",
                "mutation_score_lower_bound": "finishing_mutation_score_lower_bound",
                "mutants_untested": "finishing_untested",
                "seconds": "finishing_duration_s"}
#: The accepted starter suites' statistics of each value, before the loops:
#: tests.csv's starter_suite_<value>_<statistic> columns.
STARTER = {"line_coverage": "line", "branch_coverage": "branch", "mutation_score": "mutation"}
STATISTICS = ("mean", "median", "min", "max", "std")
#: Where a file's time went, in seconds: no second is counted twice.
TIME = {"total": "duration_s_target", "model_requests": "time_model_s", "test_runs": "time_test_s",
        "coverage_measurement": "time_coverage_s", "mutation_measurement": "time_mutation_s",
        "merging": "time_merge_s", "cover_solver": "time_cover_s",
        "project_copies": "time_preparation_s", "other": "time_other_s"}
#: Each name a summary uses, in plain words, for displays.
LABELS = {
    "line_coverage": "Line coverage", "branch_coverage": "Branch coverage",
    "mutation_score": "Mutation score", "mutation_score_selective": "Mutation score, selective catalogue",
    "mutation_score_lower_bound": "Mutation score, lower bound (untested mutants count as not killed)",
    "mutants_untested": "Mutants not tested (the measurement gave no result for them)",
    "tests": "Tests in the suite",
    "logical_requests": "Model requests (counted against the limit)",
    "physical_requests": "Requests sent, retries included", "input_tokens": "Input tokens",
    "output_tokens": "Output tokens", "cached_input_tokens": "Cache-hit tokens",
    "turns": "Tool calls (accepted edits not counted)", "seconds": "Time", "mean": "Mean", "median": "Median", "min": "Lowest",
    "max": "Highest", "std": "Standard deviation", "total": "Total for the target",
    "model_requests": "Model requests", "test_runs": "Test runs",
    "coverage_measurement": "Coverage measurement", "mutation_measurement": "Mutation measurement",
    "merging": "Merging suites", "cover_solver": "Reducing the suite",
    "project_copies": "Copying the project", "other": "Other",
    "starter": "Starter phase", "coverage": "Coverage phase", "mutation": "Mutation phase",
    "warnings": "Warnings", "time_limit": "Time limit reached",
    "loop_limit": "Limit reached before the goal",
    "warning_runs": "Runs with a warning",
    "finishing": "Finishing",
}

#: Each phase outcome a run summary or a phase record names, in plain words:
#: the phases' own outcomes and the phase records' states.
OUTCOMES = {
    "ok": "done", "success": "goal reached",
    "stagnation": "stopped before its goal: measurements without improvement",
    "budget_exhausted": "request, token or time limit reached",
    "tests_failed": "ended with failing tests",
    "context_overflow": "the model's context was full", "agent_error": "ended on an error",
    "inconclusive": "the mutation tool gave no usable result", "skipped": "skipped",
    "skipped_inherited_red": "skipped: the suite it received has failing tests",
    "skipped_inherited_slow": "skipped: the suite it received runs longer than its time limit",
    "skipped_below_minimum": "skipped: a suite ended below its configured minimum",
    "inherited_not_portable": "skipped: the suite it received does not pass when run on its own",
    "crashed": "stopped by an unexpected error", "reused": "reused from an earlier run",
    "reuse_refused": "its saved results could not be reused",
    "unknown": "not reported",
    "validation_unresolved": "generated starter suites could not be checked",
    "cancelled_zero_passing": "no starter suite passed with complete measurements",
    "failed_model_request": "a model request failed",
    "failed_phase_selection": "no suite was selected",
    "complete": "finished", "failed": "failed", "not_started": "not started",
    "in_progress": "did not finish",
}


def outcome_text(status: str) -> str:
    """A phase outcome in plain words, an unknown one as it is."""
    return OUTCOMES.get(status, status)


#: Why a failed phase failed, as its record names it, in plain words.
FAILURE_KINDS = {"infrastructure": "a model service or tool problem",
                 "method": "no usable suite within the limits"}


def failure_kind_text(kind: str) -> str:
    """A failed phase's kind of failure in plain words, an unknown one as it is."""
    return FAILURE_KINDS.get(kind, kind)


def _number(text: Any) -> Optional[float]:
    try:
        value = float(text)
    except (TypeError, ValueError):
        return None
    return value if value == value else None


def run_summary(row: Dict[str, str]) -> Dict[str, Any]:
    """One run's summary from its tests.csv row."""
    values = {name: _number(row.get(name)) for name in VALUES}
    return {
        "run": project.run_name(int(row["run"])),
        "file": row.get("file_path", ""),
        "fully_measured": all(values[name] is not None for name in FULLY_MEASURED_WITH),
        **values,
        **{name: row.get(name) or None for name in STATUSES},
        "phases": {phase: {name: _number(row.get(column)) for name, column in columns.items()}
                   for phase, columns in PHASE_VALUES.items()},
        "finishing": {name: _number(row.get(column)) for name, column in FINAL_VALUES.items()},
        "starter": {name: {statistic: _number(row.get(f"starter_suite_{short}_{statistic}"))
                           for statistic in STATISTICS} for name, short in STARTER.items()},
        "time": {name: _number(row.get(column)) for name, column in TIME.items()},
        # The time limits the run reached and the loops a limit stopped
        # (warnings), filled from the run directory when the summary is written.
        "warnings": [],
    }


def file_summary(file: str, runs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The summary across a file's runs, from their run summaries: each
    final value's mean, lowest and highest over the runs that measured it,
    and the mean over the runs of the starter suites' median of each value
    (the result before the loops)."""
    summary: Dict[str, Any] = {
        "file": file,
        "runs": [run["run"] for run in runs],
        "fully_measured_runs": sum(1 for run in runs if run["fully_measured"]),
        "warning_runs": sum(1 for run in runs if run.get("warnings")),
    }
    for name in VALUES:
        measured = [run[name] for run in runs if run[name] is not None]
        summary[name] = {
            "runs": len(measured),
            "mean": statistics.fmean(measured) if measured else None,
            "minimum": min(measured) if measured else None,
            "maximum": max(measured) if measured else None,
        }
    # The runs whose mutation measurement in Finishing left mutants untested: no score,
    # but the lower bound, the untested mutants counted as not killed.
    bounds = [run["finishing"]["mutation_score_lower_bound"] for run in runs
              if run.get("mutation_score") is None
              and (run.get("finishing") or {}).get("mutation_score_lower_bound") is not None]
    summary["mutation_score_lower_bound"] = {
        "runs": len(bounds),
        "mean": statistics.fmean(bounds) if bounds else None,
        "minimum": min(bounds) if bounds else None,
        "maximum": max(bounds) if bounds else None,
    }
    starters = {name: [run.get("starter", {}).get(name, {}).get("median") for run in runs]
                for name in VALUES}
    summary["starter_median"] = {
        name: statistics.fmean(found) if (found := [value for value in values if value is not None])
        else None for name, values in starters.items()}
    return summary


def statistics_of(values: List[Optional[float]]) -> Dict[str, Any]:
    """Statistics of the values that are known: how many, their mean,
    median, lowest, highest and standard deviation (of a sample, None for
    fewer than two), and the values themselves, for a distribution."""
    known = [value for value in values if value is not None]
    return {
        "count": len(known),
        "mean": statistics.fmean(known) if known else None,
        "median": statistics.median(known) if known else None,
        "minimum": min(known) if known else None,
        "maximum": max(known) if known else None,
        "std": statistics.stdev(known) if len(known) > 1 else None,
        "values": known,
    }


def _write(path: Path, data: Dict[str, Any]) -> None:
    """Write ``data`` as JSON in one step, so a reader never sees half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(prefix=".summary.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(data, stream, indent=2)
            stream.write("\n")
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.remove(temporary)


def _rows(results_dir: Path, target_number: int) -> List[Dict[str, str]]:
    try:
        rows = [row for row in _read_csv(results_dir / "tests.csv")[1]
                if str(row.get("target_number", "")).strip() == str(target_number)]
    except OSError:
        return []
    return sorted((row for row in rows if str(row.get("run", "")).strip().isdigit()),
                  key=lambda row: int(row["run"]))


def write_file_summaries(results_dir: Path, workspace: str, target_number: int) -> Optional[Dict[str, Any]]:
    """Write the summary of each run of target ``target_number`` that tests.csv records
    and the summary across them. Returns the summary across runs, or None
    when tests.csv records no run of the file."""
    rows = _rows(Path(results_dir), target_number)
    if not rows:
        return None
    target_dir = Path(results_dir) / workspace
    runs = []
    for row in rows:
        run_dir = project.run_dir(target_dir, int(row["run"]))
        summary = {**run_summary(row), "warnings": warnings(run_dir)}
        _write(run_dir / SUMMARY, summary)
        runs.append(summary)
    across = file_summary(runs[-1]["file"], runs)
    _write(target_dir / SUMMARY, across)
    return across


def project_summary(files: List[Dict[str, Any]]) -> Dict[str, Any]:
    """The project's summary from its file summaries: the files, and for
    each final value the statistics across the files of each file's mean
    over its runs, beside those of the starter suites' median (the
    result before the loops)."""
    return {
        "files": files,
        "values": {name: statistics_of([(item.get(name) or {}).get("mean") for item in files])
                   for name in VALUES},
        "starter_median": {name: statistics_of([(item.get("starter_median") or {}).get(name)
                                                for item in files]) for name in VALUES},
        # The files with no mutation score whose measurement left mutants
        # untested: statistics of their lower bounds.
        "mutation_score_lower_bound": statistics_of([
            (item.get("mutation_score_lower_bound") or {}).get("mean") for item in files
            if (item.get("mutation_score") or {}).get("mean") is None]),
    }


def write_project_summary(results_dir: Path) -> None:
    """Write ``results/summary.json`` from the file summaries in the results
    directory (``project_summary``)."""
    files = []
    for path in sorted(Path(results_dir).glob(f"*/{SUMMARY}")):
        try:
            files.append(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, ValueError):
            continue
    _write(Path(results_dir) / SUMMARY, project_summary(files))


def recorded_source_sha256(output_dir, target: str) -> Optional[str]:
    """The SHA-256, in hex, of a target file's raw bytes that the run in
    ``output_dir`` recorded when the target's phases started, or None when
    its records hold none or its phases recorded different ones. ``target``
    is the file's path relative to the run's source directory, as the
    run's completion names it. A phase that recorded none, such as one
    the run reused, does not count. A caller compares the result with
    ``hashlib.sha256`` of the file's bytes read in binary. Only the
    records are read."""
    state = phase_state.load(Path(output_dir) / "results" / project.workspace_name(target))
    hashes = set()
    for phase in phase_state.PHASES:
        runs = state.get(phase)
        for block in runs.values() if isinstance(runs, dict) else ():
            value = block.get("target_source_sha256") if isinstance(block, dict) else None
            if isinstance(value, str) and value:
                hashes.add(value)
    return hashes.pop() if len(hashes) == 1 else None
