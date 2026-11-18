"""The canonical phase output: the delivered suite, persisted under names
that do not depend on which candidate won.

Per-candidate sidecars describe their own candidate, so
`reduced_coverage_matrix.json` describes the delivered suite only when the
reduction was delivered. The canonical artifacts are written for every winner and
carry the delivered bytes' hash, so a reader never derives the delivered suite
from the recorded winner.

Everything below the mutation engine is production code: the boundary,
merging, cover, selection and persistence.
"""
from __future__ import annotations

import hashlib
import json

import pytest

import test_coverage_boundary_identity as boundary
from testuary.consolidation import (
    ScoredSuite, selected_coverage_payload,
)

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

CANONICAL = ["selected_coverage_matrix.json", "selected_kill_matrix.json"]


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Through the real coverage boundary, for every outcome its harness can reach
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("agent_suite", [
    boundary.AGENT_SUITE_REDUNDANT,
    boundary.AGENT_SUITE_IRREDUCIBLE,
    boundary.AGENT_SUITE_MERGE_RED,
], ids=["redundant", "irreducible", "merge_red"])
def test_the_canonical_artifacts_describe_the_delivered_suite(
    tmp_path, monkeypatch, agent_suite,
):
    """Whatever candidate wins, the canonical coverage and kill matrix carry
    the delivered suite's hash, unlike the reduced sidecar."""
    run = boundary._drive_coverage_boundary(tmp_path, monkeypatch, agent_suite)

    delivered = run.suite_path.read_text(encoding="utf-8")
    record = _json(run.archive / "consolidation_record.json")
    assert record["selected_source_sha256"] == _digest(delivered)

    coverage = _json(run.archive / "selected_coverage_matrix.json")
    assert coverage["source_sha256"] == _digest(delivered)
    assert coverage["execution_name"] == "test_calc.py"
    assert coverage["candidate_label"] == record["delivered_from"]
    # The per-test evidence names the delivered suite's own tests.
    assert coverage["per_test_tokens"], coverage
    assert sorted(coverage["per_test_tokens"]) == coverage["kept_tests"]
    for test_id in coverage["per_test_tokens"]:
        assert test_id.startswith("test_calc.")

    matrix = _json(run.archive / "selected_kill_matrix.json")
    assert matrix["source_sha256"] == _digest(delivered)
    assert matrix["execution_name"] == "test_calc.py"
    assert record["shipped_has_matrix"] is True
    assert record["artifacts"][:0] == []
    for name in CANONICAL:
        assert name in record["artifacts"]


def test_the_canonical_coverage_counts_the_delivered_tests(
    tmp_path, monkeypatch,
):
    """The canonical coverage's per-test entries match the record's delivered
    test count, which is below the merged suite's after the reduction."""
    run = boundary._drive_coverage_boundary(
        tmp_path, monkeypatch, boundary.AGENT_SUITE_REDUNDANT)

    record = _json(run.archive / "consolidation_record.json")
    coverage = _json(run.archive / "selected_coverage_matrix.json")

    assert len(coverage["per_test_tokens"]) == record["delivered_tests"]
    assert record["delivered_tests"] < record["merged_tests"]


def test_the_canonical_matrix_is_written_for_a_phase_that_ships_no_reduction(
    tmp_path, monkeypatch,
):
    """A ``single_suite`` phase end, where the reduced sidecar is least
    trustworthy, also writes the canonical matrix.

    This harness persists its initial boundary without the winning candidate
    the starter phase passes, so it has no canonical coverage. The tests
    above assert the coverage side through the coverage boundary."""
    run = boundary._drive_coverage_boundary(
        tmp_path, monkeypatch, boundary.AGENT_SUITE_REDUNDANT)
    initial = run.ctx.results_dir / run.ctx.workspace / "r0" / "archive" / "starter"

    record = _json(initial / "consolidation_record.json")
    assert record["delivered_from"] == "single_suite"
    matrix = _json(initial / "selected_kill_matrix.json")

    assert matrix["source_sha256"] == record["selected_source_sha256"]
    assert matrix["mutants"]
    assert "selected_kill_matrix.json" in record["artifacts"]


# ---------------------------------------------------------------------------
# The payload builder
# ---------------------------------------------------------------------------

class _Phase:
    structural_parser_identity = "tree-sitter test"
    target_validator_identity = "CPython test"
    target_total_lines = 10
    target_total_branches = 4
    per_test_cost_ms = {"t::a": 1.5, "t::b": 2.5, "t::dropped": 9.0}
    effective_cost_us = {"t::a": 1500.0, "t::b": 2500.0, "t::dropped": 9000.0}


def _candidate(**kw):
    defaults = dict(
        label="agent", source="def test_a():\n    assert True\n",
        line=80.0, branch=50.0,
        per_test_tokens={"t::a": frozenset({"L:1"}), "t::b": frozenset({"L:2"})},
        covered_lines=frozenset({1, 2}), covered_branches=frozenset({"1->2"}),
    )
    defaults.update(kw)
    return ScoredSuite(**defaults)


def test_the_payload_describes_the_candidate_it_is_given():
    payload = selected_coverage_payload(_Phase(), _candidate(), "calc.py")

    assert payload["line"] == 80.0
    assert payload["branch"] == 50.0
    assert payload["candidate_label"] == "agent"
    assert sorted(payload["per_test_tokens"]) == ["t::a", "t::b"]
    assert payload["per_test_tokens"]["t::a"] == ["L:1"]
    assert payload["kept_tests"] == ["t::a", "t::b"]


def test_the_payload_restricts_the_cost_maps_to_the_candidates_tests():
    """The cost maps hold only the candidate's own tests, so a reader never
    prices a test that was not delivered."""
    payload = selected_coverage_payload(_Phase(), _candidate(label="reduced"), "calc.py")

    assert sorted(payload["per_test_runtime_ms"]) == ["t::a", "t::b"]
    assert sorted(payload["effective_cost_us"]) == ["t::a", "t::b"]
    assert "t::dropped" not in payload["per_test_runtime_ms"]


def test_another_winner_records_no_merged_run_times():
    """The merged run's per-test times are keyed by the merged suite's test
    ids, which can name another candidate's renamed collisions, so a winner
    the merged run did not measure records none."""
    payload = selected_coverage_payload(_Phase(), _candidate(label="agent"), "calc.py")

    assert payload["per_test_runtime_ms"] is None
    assert payload["effective_cost_us"] is None
    assert payload["per_test_tokens"]["t::a"] == ["L:1"]


def test_the_payload_falls_back_to_the_candidates_own_evidence():
    """A candidate measured standalone carries its coverage in
    ``coverage_evidence`` rather than in the phase's merged fields."""
    candidate = _candidate(
        per_test_tokens=None, covered_lines=None, covered_branches=None,
        coverage_evidence={
            "per_test_tokens": {"t::a": ["L:1"]},
            "covered_statement_lines": [1],
            "covered_branch_arcs": ["1->2"],
            "target_total_statements": 7,
            "target_total_branches": 2,
        },
    )

    payload = selected_coverage_payload(_Phase(), candidate, "calc.py")

    assert payload["per_test_tokens"] == {"t::a": ["L:1"]}
    assert payload["covered_statement_lines"] == [1]
    assert payload["target_total_statements"] == 7
    assert payload["target_total_branches"] == 2


def test_a_java_candidate_s_coverage_run_records_the_target_s_branch_total():
    """JaCoCo counts a partly taken branch line as one untaken-branch entry,
    "7->1 of 2 branches untaken". The run's totals come from its coverage
    observation, so the candidate's evidence and the delivered suite's
    payload record the target's branch total, not that count."""
    from types import SimpleNamespace

    from testuary.consolidation import coverage_evidence_of_run

    run = SimpleNamespace(line_coverage=80.0, branch_coverage=50.0,
                          covered_lines=[1, 2, 7], covered_branches=["7->1"],
                          missing_lines=(), missing_branches=("7->1 of 2 branches untaken",),
                          total_statements=5, total_branches=2)
    evidence = coverage_evidence_of_run(run, "Calc.java")
    assert (evidence["target_total_statements"], evidence["target_total_branches"]) == (5, 2)
    candidate = _candidate(per_test_tokens=None, covered_lines=None, covered_branches=None,
                           coverage_evidence=evidence)
    payload = selected_coverage_payload(_Phase(), candidate, "Calc.java")
    assert (payload["target_total_statements"], payload["target_total_branches"]) == (5, 2)


def test_a_run_without_totals_leaves_the_payload_the_phase_end_s():
    """A run that measured no totals records None in the candidate's
    evidence, and the delivered suite's payload takes the phase end's."""
    from types import SimpleNamespace

    from testuary.consolidation import coverage_evidence_of_run

    run = SimpleNamespace(line_coverage=80.0, branch_coverage=50.0,
                          covered_lines=[1, 2], covered_branches=["1->2"],
                          missing_lines=(), missing_branches=())
    evidence = coverage_evidence_of_run(run, "calc.py")
    assert (evidence["target_total_statements"], evidence["target_total_branches"]) == (None, None)
    candidate = _candidate(per_test_tokens=None, covered_lines=None, covered_branches=None,
                           coverage_evidence=evidence)
    payload = selected_coverage_payload(_Phase(), candidate, "calc.py")
    assert (payload["target_total_statements"], payload["target_total_branches"]) == (10, 4)


def test_a_candidate_without_coverage_yields_no_payload():
    """A phase that measured no coverage writes no canonical sidecar rather
    than an empty one that a reader would take as evidence of zero."""
    candidate = _candidate(
        line=None, branch=None, per_test_tokens=None,
        covered_lines=None, covered_branches=None, coverage_evidence=None,
    )

    assert selected_coverage_payload(_Phase(), candidate, "calc.py") is None
