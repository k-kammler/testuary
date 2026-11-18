"""Pure tests for the shared phase-end candidate selector."""

from __future__ import annotations

import math

import pytest

from dataclasses import replace

from testuary.consolidation import ScoredSuite, choose_best, _PHASE_END_AXES


def _c(
    label: str,
    *,
    mutation=50.0,
    branch=50.0,
    line=50.0,
    runtime_ms=100.0,
    green=True,
    kills=("m1",),
):
    matrix = None if mutation is None else {
        "mutants": [
            {"id": mutant, "status": "killed"}
            for mutant in kills
        ]
    }
    return ScoredSuite(
        label=label,
        source=f"# {label}\n",
        mutation=mutation,
        branch=branch,
        line=line,
        runtime_ms=runtime_ms,
        green=green,
        kill_matrix=matrix,
    )


@pytest.mark.parametrize("axis", ["mutation", "branch", "line"])
def test_a_strict_score_win_selects_that_candidate(axis):
    baseline = dict(mutation=50.0, branch=50.0, line=50.0)
    ahead = dict(baseline)
    ahead[axis] = 60.0
    winner = choose_best([
        _c("baseline", **baseline),
        _c("ahead", runtime_ms=10_000.0, **ahead),
    ], required_axes=_PHASE_END_AXES)
    assert winner.label == "ahead"


def test_mutation_outranks_branch_and_line():
    winner = choose_best([
        _c("coverage", mutation=49.0, branch=100.0, line=100.0),
        _c("mutation", mutation=50.0, branch=1.0, line=1.0),
    ], required_axes=_PHASE_END_AXES)
    assert winner.label == "mutation"


def test_branch_breaks_mutation_tie_before_line():
    winner = choose_best([
        _c("line", mutation=50.0, branch=49.0, line=100.0),
        _c("branch", mutation=50.0, branch=50.0, line=1.0),
    ], required_axes=_PHASE_END_AXES)
    assert winner.label == "branch"


def test_non_passing_and_incompletely_measured_candidates_are_excluded():
    winner = choose_best([
        _c("red", mutation=100.0, branch=100.0, line=100.0, green=False),
        _c("no_mutation", mutation=None, branch=100.0, line=100.0),
        _c("no_branch", mutation=100.0, branch=None, line=100.0),
        _c("no_line", mutation=100.0, branch=100.0, line=None),
        _c("complete", mutation=1.0, branch=1.0, line=1.0),
    ], required_axes=_PHASE_END_AXES)
    assert winner.label == "complete"


def test_no_complete_passing_candidate_returns_none():
    assert choose_best([
        _c("red", green=False),
        _c("partial", mutation=None),
    ], required_axes=_PHASE_END_AXES) is None


def test_an_empty_branch_denominator_is_still_a_measurement():
    candidate = _c("branchless", branch=None)
    candidate = replace(candidate, covered_branches=frozenset())
    assert choose_best([candidate], required_axes=_PHASE_END_AXES) == candidate


def test_an_empty_mutant_catalogue_is_still_a_measurement():
    candidate = _c("mutant_free", mutation=0.0)
    candidate = replace(candidate, mutation=None, kill_matrix={"mutants": []})
    assert choose_best([candidate], required_axes=_PHASE_END_AXES) == candidate


def test_a_tied_reduced_suite_is_preferred_without_benchmarking():
    seen = []
    winner = choose_best([
        _c("input"),
        _c("reduced", runtime_ms=9_000.0),
    ], preferred_label="reduced", runtime_of=lambda candidate: seen.append(
        candidate.label
    ), required_axes=_PHASE_END_AXES)
    assert winner.label == "reduced"
    assert seen == []


def test_a_non_reduced_tie_requires_a_timing_benchmark():
    with pytest.raises(RuntimeError, match="no suite timing"):
        choose_best([_c("baseline"), _c("agent")],
                    preferred_label="reduced", required_axes=_PHASE_END_AXES)


def test_the_measured_timing_decides_a_non_reduced_tie():
    values = {"baseline": 900.0, "agent": 100.0}
    winner = choose_best(
        [_c("baseline", runtime_ms=1.0),
         _c("agent", runtime_ms=10_000.0)],
        preferred_label="reduced",
        runtime_of=lambda candidate: values[candidate.label], required_axes=_PHASE_END_AXES,
    )
    assert winner.label == "agent"


def test_an_invalid_timing_measurement_fails_explicitly():
    for value in (-1.0, math.inf, math.nan):
        with pytest.raises(RuntimeError, match="invalid median"):
            choose_best(
                [_c("baseline"), _c("agent")],
                runtime_of=lambda _candidate, result=value: result, required_axes=_PHASE_END_AXES,
            )


def test_exact_evidence_identity_is_not_an_eligibility_gate():
    # The higher mutation score wins even though its killed-mutant identity
    # differs. That difference belongs in the phase-end audit record.
    winner = choose_best([
        _c("old", mutation=50.0, kills=("m1",)),
        _c("new", mutation=51.0, kills=("m2",)),
    ], required_axes=_PHASE_END_AXES)
    assert winner.label == "new"


def test_measured_timing_tie_breaks_by_input_order():
    first = _c("first")
    second = _c("second")
    winner = choose_best(
        [first, second], runtime_of=lambda _candidate: 5.0, required_axes=_PHASE_END_AXES,
    )
    assert winner is first
