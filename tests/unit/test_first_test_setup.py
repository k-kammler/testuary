"""The run's first test is not charged the run's one-time setup.

A per-test evidence run pays its one-time setup (the runner's and the
session's first-use costs) in whichever test runs first. Charged to that
test, the setup would make the reduction avoid it.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from testuary.languages.generic import EvictionData
from testuary import reduction
from testuary.reduction import EQUAL_COST_MS, ExactCoverError, reduction_costs, solve_cover

from test_consolidate_phase_end import SUITE_A, SUITE_B, _project
from phase_end_calls import phase_end

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


TOKENS = {
    "t.test_first": frozenset({"a", "b"}),
    "t.test_left": frozenset({"a"}),
    "t.test_right": frozenset({"b"}),
}


def test_the_first_test_costs_at_most_the_slowest_other_test():
    times = {"t.test_first": 250.0, "t.test_left": 3.0, "t.test_right": 4.0}

    assert reduction.first_test_setup(TOKENS, times, "t.test_first") == {
        "test": "t.test_first", "measured_ms": 250.0, "cost_ms": 4.0,
        "setup_ms": 246.0,
    }
    assert reduction_costs(TOKENS, times, first_test="t.test_first") == (
        {"t.test_left": 3.0, "t.test_right": 4.0}, {"t.test_first": 4.0})


def test_a_first_test_faster_than_the_others_keeps_its_time():
    times = {"t.test_first": 2.0, "t.test_left": 3.0, "t.test_right": 4.0}

    assert reduction.first_test_setup(TOKENS, times, "t.test_first")["cost_ms"] == 2.0
    assert reduction_costs(TOKENS, times, first_test="t.test_first")[1] == {
        "t.test_first": 2.0}


@pytest.mark.parametrize("first, times", [
    (None, {"t.test_first": 250.0, "t.test_left": 3.0}),
    ("t.test_gone", {"t.test_first": 250.0, "t.test_left": 3.0}),
    ("t.test_first", {"t.test_left": 3.0}),
    ("t.test_first", {"t.test_first": 250.0}),
], ids=["unknown", "not-a-test", "untimed", "no-other-time"])
def test_without_a_comparable_time_nothing_is_adjusted(first, times):
    assert reduction.first_test_setup(TOKENS, times, first) is None


def test_an_untimed_test_is_not_imputed_the_setup():
    times = {"t.test_first": 250.0, "t.test_left": 3.0}

    measured, imputed = reduction_costs(TOKENS, times, first_test="t.test_first")

    assert measured == {"t.test_left": 3.0}
    assert imputed == {"t.test_first": 3.0, "t.test_right": 3.0}
    assert reduction_costs(TOKENS, {}, first_test="t.test_first") == (
        {}, {test: EQUAL_COST_MS for test in TOKENS})


@pytest.mark.parametrize("solver", ["exact", "greedy"])
def test_the_reduction_keeps_the_first_test_once_its_setup_is_not_charged(solver):
    times = {"t.test_first": 250.0, "t.test_left": 3.0, "t.test_right": 4.0}
    measured, imputed = reduction_costs(TOKENS, times, first_test="t.test_first")

    cover = solve_cover(TOKENS, measured, imputed_cost_ms=imputed, solver=solver)

    assert cover.keep == ("t.test_first",)


@pytest.mark.parametrize("solver", ["exact", "greedy"])
def test_a_millisecond_clock_with_zero_times_keeps_the_ties_decidable(solver):
    """Java reports whole milliseconds, so other tests often measure 0 ms
    and the first test then costs 0 ms too. The solver's minimum cost makes
    the tests equal, not free: the fewest tests win, then name order."""
    tokens = {
        "t.test_first": frozenset({"a"}),
        "t.test_second": frozenset({"a"}),
        "t.test_third": frozenset({"b"}),
    }
    times = {"t.test_first": 41.0, "t.test_second": 0.0, "t.test_third": 0.0}
    measured, imputed = reduction_costs(tokens, times, first_test="t.test_first")

    assert imputed == {"t.test_first": 0.0}
    cover = solve_cover(tokens, measured, imputed_cost_ms=imputed, solver=solver)

    assert cover.keep == ("t.test_first", "t.test_third")


def test_the_phase_end_charges_the_setup_to_no_test(tmp_path, monkeypatch):
    backend, target, root = _project(tmp_path)

    def evidence(_self, *_args, **_kwargs):
        return SimpleNamespace(
            per_test_tokens={
                "test_calc.test_add_a": frozenset({"L:calc:2"}),
                "test_calc.test_add_b": frozenset({"L:calc:2"}),
            },
            per_test_time_ms={"test_calc.test_add_a": 250.0,
                              "test_calc.test_add_b": 3.0},
            # Real target coverage: 3 of 8 lines, none of the 4 branches.
            suite_time_ms=253.0, includes_kills=False, target_total_lines=8,
            target_total_branches=4, suite_covered_lines=frozenset({1, 2, 5}),
            suite_covered_branches=frozenset(),
            first_test="test_calc.test_add_a",
        )

    seen = {}

    def solver(per_test_tokens, per_test_cost_ms, **kwargs):
        seen.update(costs=dict(per_test_cost_ms),
                    imputed=dict(kwargs.get("imputed_cost_ms") or {}))
        raise ExactCoverError("the costs are what this test checks", {"solver": "exact"})

    monkeypatch.setattr(type(backend), "collect_eviction_data", evidence)
    monkeypatch.setattr("testuary.reduction.solve_cover", solver)
    with pytest.raises(ExactCoverError):
        phase_end(
            backend, [SUITE_A, SUITE_B], target_file_path=target, source_dir=root,
            timeout=120, mutation_runner=None, solver="exact", log=lambda _message: None,
        )
    assert seen == {"costs": {"test_calc.test_add_b": 3.0},
                    "imputed": {"test_calc.test_add_a": 3.0}}


def test_the_demonstration_evidence_keeps_the_first_test():
    from testuary.demo.recorded import eviction_data, eviction_data_json

    data = EvictionData({"t.test_b": frozenset({"x"}), "t.test_a": frozenset({"y"})},
                        {"t.test_b": 90.0, "t.test_a": 1.0}, 91.0, False,
                        first_test="t.test_b")

    assert eviction_data(eviction_data_json(data)).first_test == "t.test_b"
    assert EvictionData({}, {}, 0.0, False).first_test is None
