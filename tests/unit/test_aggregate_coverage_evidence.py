"""``testuary.consolidation.aggregate_coverage_evidence``: the dict it returns
for a run with an exact coverage matrix and the ValueError each check of the
matrix raises, in the order the checks run, and the merged suite's refusal of
a matrix that is not exact."""
from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from testuary.consolidation import (
    PhaseConsolidation,
    _record_aggregate_coverage,
    aggregate_coverage_evidence,
    coverage_disagreement,
    coverage_percents,
)


def _matrix(**changes):
    """A consistent matrix for a target with lines 1 to 4, line 3 missing,
    and branches 1->2 (taken) and 2->4 (missing)."""
    matrix = {
        "schema_version": 1,
        "target_file": "mod.py",
        "lines": [
            {"line": 2, "status": "covered"},
            {"line": 1, "status": "covered"},
            {"line": 3, "status": "missing"},
            {"line": 4, "status": "covered"},
        ],
        "branches": [
            {"from": 1, "to": 2, "status": "taken"},
            {"from": 2, "to": 4, "status": "missing"},
        ],
        "summary": {"total_lines": 4, "covered_lines": 3, "missing_lines": 1,
                    "covered_branches": 1, "missing_branches": 1},
    }
    matrix.update(changes)
    return matrix


def _run(**changes):
    run = {"line_coverage": 75.0, "branch_coverage": 50.0, "collected": 3,
           "passed": 2, "failed": 1, "errors": 0}
    run.update(changes)
    return SimpleNamespace(**run)


def test_a_target_without_lines_has_no_coverage_and_its_run_agrees():
    """LangAdapter reports no line or branch coverage for a target without
    statements, and the exact matrix of such a target agrees with that
    run."""
    empty = _matrix(lines=[], branches=[], summary={
        "total_lines": 0, "covered_lines": 0, "missing_lines": 0,
        "covered_branches": 0, "missing_branches": 0})
    run = _run(line_coverage=None, branch_coverage=None)
    assert aggregate_coverage_evidence(run, empty, target_file="mod.py") == {
        "line": None, "branch": None, "target_total_statements": 0,
        "covered_statement_lines": [], "target_total_branches": 0,
        "covered_branch_arcs": []}


def test_a_run_with_an_exact_matrix_gives_its_sorted_evidence():
    assert aggregate_coverage_evidence(_run(), _matrix(), target_file="mod.py") == {
        "line": 75.0,
        "branch": 50.0,
        "target_total_statements": 4,
        "covered_statement_lines": [1, 2, 4],
        "target_total_branches": 2,
        "covered_branch_arcs": ["1->2"],
    }


@pytest.mark.parametrize("content", [None, "not json", json.dumps({"lines": None}),
                                     json.dumps({"lines": [], "branches": {}})])
def test_the_merged_suite_without_an_exact_matrix_records_no_coverage(tmp_path, content):
    path = tmp_path / "aggregate.json"
    if content is not None:
        path.write_text(content, encoding="utf-8")
    res = PhaseConsolidation()
    assert _record_aggregate_coverage(None, res, _run(), path, merged="",
                                      target_file_path="mod.py") is False
    assert res.detail == "the merged suite's aggregate coverage matrix could not be persisted"
    assert res.merged_incomplete == res.detail
    assert res.line is None and res.suite_covered_lines is None


@pytest.mark.parametrize("matrix, run, message", [
    # Each matrix breaks the check named and every later one, so the
    # first check to fail decides the message.
    (_matrix(schema_version=2, target_file="other.py",
             lines=[{"line": 1, "status": "partial"}]), _run(),
     "raw aggregate coverage schema is unsupported"),
    (_matrix(target_file="other.py", lines=[{"line": 1, "status": "partial"}]), _run(),
     "raw aggregate coverage target disagrees"),
    (_matrix(lines=[{"line": 1, "status": "covered"}, {"line": 1, "status": "covered"}],
             branches=[{"from": 1, "to": 2, "status": "partial"}]), _run(),
     "raw aggregate coverage matrix is malformed"),
    (_matrix(lines=[{"line": 1, "status": "covered"}, {"line": 1, "status": "covered"}],
             summary={}), _run(),
     "raw aggregate coverage matrix has duplicates"),
    (_matrix(branches=[{"from": 1, "to": 2, "status": "taken"},
                       {"from": 1, "to": 2, "status": "missing"}], summary={}), _run(),
     "raw aggregate coverage matrix has duplicates"),
    (_matrix(summary={"total_lines": 4, "covered_lines": 3, "missing_lines": 1,
                      "covered_branches": 1, "missing_branches": 0}),
     _run(line_coverage=10.0), "raw aggregate coverage summary disagrees"),
    (_matrix(), _run(line_coverage=70.0), "raw and scalar aggregate coverage disagree"),
    (_matrix(), _run(branch_coverage=None), "raw and scalar aggregate coverage disagree"),
])
def test_each_matrix_check_raises_its_own_message_in_order(matrix, run, message):
    with pytest.raises(ValueError, match=f"^{message}$"):
        aggregate_coverage_evidence(run, matrix, target_file="mod.py")


@pytest.mark.parametrize("counts, percents", [
    ((3, 4, 1, 2), (75.0, 50.0)),
    # A target without lines or branches has no line or branch coverage, as
    # LangAdapter reports it.
    ((0, 0, 0, 0), (None, None)),
    ((None, 4, None, 2), (None, None)),
    ((3, None, 1, None), (None, None)),
])
def test_one_rule_gives_a_suite_s_coverage_percents(counts, percents):
    assert coverage_percents(*counts) == percents


@pytest.mark.parametrize("first, second, differ", [
    ((75.0, 50.0), (75.0, 50.0), False),
    ((75.0, 50.0), (75.0 + 1e-12, 50.0), False),
    ((75.0, 50.0), (75.1, 50.0), True),
    ((75.0, 50.0), (75.0, 50.1), True),
    ((75.0, None), (75.0, 50.0), True),
    ((75.0, None), (75.0, None), False),
    ((None, None), (None, None), False),
    ((None, None), (75.0, None), True),
])
def test_one_rule_compares_two_coverage_measurements(first, second, differ):
    assert coverage_disagreement(*first, *second) is differ
