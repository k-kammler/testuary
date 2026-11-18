"""The normal test run: the one run that decides whether tests pass.

Every question whether a suite's tests pass, and which of them fail
(``failing_ids``), goes here. The coverage and mutation tools only measure:
their runs never turn a test that passes here into a failing one.

A language whose runs as delivered take a session seed (Python's) has the
run as delivered as its normal run, with the fixed session seed
``SESSION_SEED``. A language without runs as delivered, or whose runs as
delivered take no seed (Java's, C's), runs the backend's
``run_test_files``.
"""
from __future__ import annotations

from typing import Any, Optional, Sequence

#: The session seed of the normal run as delivered.
SESSION_SEED = 1


def run(backend: Any, test_paths: Sequence[Any], source_dir: str, *, timeout: int,
        target: Optional[str] = None, fail_fast: bool = False, tb: str = "short",
        record_leaks: bool = False, timeout_journal: bool = False) -> Any:
    """The normal run of ``test_paths``: a ``TestRunResult`` whose ``ok`` says
    whether they pass. ``target`` is the module under test. ``fail_fast``
    and ``tb`` shape the run's output, ``record_leaks`` asks for the tests
    that left the module under test or a library setting changed (the run
    as delivered observes them without restoring anything), and
    ``timeout_journal`` for the test still running when a timeout ended the
    run."""
    paths = [str(path) for path in test_paths]
    if _through_run_test_files(backend, source_dir):
        return backend.run_test_files(
            paths, source_dir, timeout=timeout, instrument_target=target,
            fail_fast=fail_fast, tb=tb, record_leaks=record_leaks,
            **({"timeout_journal": True} if timeout_journal else {}))
    return _result_of(backend.run_as_delivered(
        paths, source_dir, timeout=timeout, seed=SESSION_SEED, target=target,
        fail_fast=fail_fast, tb=tb, record_leaks=record_leaks,
        timeout_journal=timeout_journal))


def failing_ids(backend: Any, test_paths: Sequence[Any], source_dir: str, *,
                timeout: int, target: Optional[str] = None,
                report: Optional[dict] = None) -> Optional[list]:
    """The tests the normal run of ``test_paths`` fails, as the backend's
    failing-test listing names them (``failing_test_ids``), or None when the
    run gave no test outcome. ``report`` receives why, as there."""
    paths = [str(path) for path in test_paths]
    if _through_run_test_files(backend, source_dir):
        return backend.failing_test_ids(
            paths, source_dir, timeout=timeout, instrument_target=target, report=report)
    return backend.failing_test_ids(paths, source_dir, timeout=timeout,
                                    instrument_target=target, report=report,
                                    delivered_seed=SESSION_SEED)


def _through_run_test_files(backend: Any, source_dir: str) -> bool:
    """Whether the normal run is the backend's ``run_test_files``: for a
    language without runs as delivered, or whose runs as delivered take no
    session seed (``seeds_delivered_runs``)."""
    delivers = getattr(backend, "runs_as_delivered", None)
    seeds = getattr(backend, "seeds_delivered_runs", None)
    return not (callable(delivers) and delivers(source_dir)
                and callable(seeds) and seeds(source_dir))


def _result_of(delivered: Any) -> Any:
    """The run as delivered read as a ``TestRunResult``: its cases'
    outcomes, failures and output, the id objects of its failed tests, its
    leaks and the test running at its end, no coverage."""
    from testuary.languages import TestFailure, TestRunResult

    cases = tuple(delivered.cases)
    failures = tuple(TestFailure(case.test, case.outcome, case.message)
                     for case in cases if case.outcome in ("failed", "error"))
    decided = delivered.failing is not None
    return TestRunResult(
        ok=delivered.ok,
        returncode=0 if delivered.ok else 1 if decided else -1,
        timed_out=bool(getattr(delivered, "timed_out", False)),
        elapsed_s=getattr(delivered, "elapsed_s", None),
        collected=len(cases),
        passed=sum(1 for case in cases if case.outcome == "passed"),
        failed=sum(1 for case in cases if case.outcome == "failed"),
        errors=sum(1 for case in cases if case.outcome == "error"),
        output=delivered.output or delivered.reason or "",
        line_coverage=None, branch_coverage=None,
        leaks=getattr(delivered, "leaks", None), failures=failures,
        active_at_end=getattr(delivered, "active_at_end", None),
        unsound=delivered.unsound,
        stopping_tests=tuple(getattr(delivered, "stopping_tests", ()) or ()))
