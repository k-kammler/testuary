"""The demonstration's scripted exact reduction: the recorded answer rebuilds
the solver's reduction, an answer that does not solve the package's problem
stops the run, and the key tells problems apart."""
from __future__ import annotations

import dataclasses
import json

import pytest

from testuary import reduction
from testuary.demo import backend as playback
from testuary.demo import recorded as cover
from testuary.demo.scenario import ScenarioStop

# Two covers are optimal in both tiers ({a, c} and {b, c}: 3 ms, two tests),
# so the solver's choice between them is what the scenario records.
TOKENS = {
    "t.py::test_a": frozenset({"L:1", "L:2"}),
    "t.py::test_b": frozenset({"L:1", "L:2"}),
    "t.py::test_c": frozenset({"L:3", "K:m1"}),
    "t.py::test_d": frozenset({"L:3"}),
}
COSTS = {"t.py::test_a": 1.0, "t.py::test_b": 1.0, "t.py::test_c": 2.0,
         "t.py::test_d": 2.5}
SETTINGS = {"forced_ids": frozenset(), "time_limit_s": 60.0,
            "minimum_test_runtime_ms": 0.001, "random_seed": 1,
            "tie_break": "solver"}


def _check(answer, forced_ids=frozenset()):
    return cover.check_answer(answer, TOKENS, COSTS, forced_ids=forced_ids,
                              minimum_test_runtime_ms=0.001)


@pytest.fixture(scope="module")
def solved():
    return reduction.exact_cover(TOKENS, COSTS, **SETTINGS)


def test_the_answer_rebuilds_the_reduction_the_solver_returned(solved):
    # The scenario stores its answers as JSON.
    answer = json.loads(json.dumps(cover.answer_of(solved)))
    assert _check(answer) is None
    rebuilt = cover.reduction_of(answer, TOKENS, COSTS, minimum_test_runtime_ms=0.001)
    for field in dataclasses.fields(reduction.LosslessReduction):
        if field.name != "solver_metadata":
            assert getattr(rebuilt, field.name) == getattr(solved, field.name), (
                field.name)
    ours = dict(rebuilt.solver_metadata)
    assert ours.pop("demonstration") == cover.NOTE
    theirs = {k: v for k, v in solved.solver_metadata.items() if k != "elapsed_s"}
    theirs["tiers"] = [
        {k: v for k, v in tier.items() if k not in ("elapsed_s", "time_limit_s")}
        for tier in solved.solver_metadata["tiers"]
    ]
    assert ours == theirs


def test_the_answer_leaves_out_the_machine_and_what_the_problem_determines(solved):
    answer = cover.answer_of(solved)
    assert set(answer) == {"keep", "runtime_objective_us",
                           "test_count_objective", "solver_metadata"}
    meta = answer["solver_metadata"]
    assert not {"raw_cost_ms", "effective_cost_us", "elapsed_s"} & set(meta)
    assert [set(tier) & {"elapsed_s", "time_limit_s"}
            for tier in meta["tiers"]] == [set(), set()]
    assert meta["selected_ids"] == answer["keep"] == list(solved.keep)


def test_authored_solver_provenance_is_not_replaced_with_a_cbc_claim(solved):
    answer = cover.answer_of(solved)
    note = "exhaustively enumerated synthetic example; no CBC ran"
    answer["solver_metadata"]["demonstration"] = note
    result = cover.reduction_of(answer, TOKENS, COSTS, minimum_test_runtime_ms=0.001)
    assert result.solver_metadata["demonstration"] == note


@pytest.mark.parametrize("change, forced, problem", [
    ({"keep": ["t.py::test_a", "t.py::test_c", "t.py::test_x"]}, (),
     "names tests the problem does not hold"),
    ({"keep": ["t.py::test_c", "t.py::test_a"]}, (),
     "names tests the problem does not hold"),
    ({}, ("t.py::test_d",), "leaves out a forced test"),
    ({"keep": ["t.py::test_a"], "test_count_objective": 1}, (),
     "does not cover every token"),
    ({"runtime_objective_us": 2999}, (),
     "does not meet the recorded runtime optimum"),
    ({"keep": ["t.py::test_a", "t.py::test_b", "t.py::test_c"]}, (),
     "does not meet the recorded runtime optimum"),
    ({"test_count_objective": 3}, (),
     "does not meet the recorded test-count optimum"),
])
def test_an_answer_that_does_not_solve_the_problem_is_named(change, forced, problem):
    answer = {"keep": ["t.py::test_a", "t.py::test_c"],
              "runtime_objective_us": 3000, "test_count_objective": 2,
              "solver_metadata": {}}
    assert _check(answer) is None
    answer.update(change)
    assert _check(answer, frozenset(forced)) == problem


class _Oracle:
    def __init__(self, answer):
        self.value = answer
        self.calls = []

    def answer(self, call):
        self.calls.append(call)
        return self.value


def test_the_source_asks_for_the_problem_and_returns_its_reduction(solved):
    oracle = _Oracle(json.loads(json.dumps(cover.answer_of(solved))))
    result = playback.ScenarioCoverSource(oracle).solve_exact(TOKENS, COSTS, **SETTINGS)
    (call,) = oracle.calls
    assert call.kind == cover.KIND
    assert call.key == cover.problem_key(TOKENS, COSTS, **SETTINGS)
    assert result.keep == solved.keep and result.remove == solved.remove
    assert result.solver_metadata["demonstration"] == cover.NOTE


def test_an_answer_that_does_not_fit_stops_the_run(solved):
    answer = cover.answer_of(solved)
    answer["keep"] = answer["keep"][:1]
    source = playback.ScenarioCoverSource(_Oracle(answer))
    with pytest.raises(ScenarioStop) as stop:
        source.solve_exact(TOKENS, COSTS, **SETTINGS)
    assert stop.value.call.kind == cover.KIND
    assert "does not cover every token" in stop.value.reason
    assert "regenerated" in stop.value.reason


def test_the_key_tells_problems_apart():
    key = cover.problem_key(TOKENS, COSTS, **SETTINGS)
    reordered = {t: frozenset(sorted(TOKENS[t], reverse=True))
                 for t in reversed(list(TOKENS))}
    assert cover.problem_key(reordered, dict(reversed(list(COSTS.items()))),
                             **SETTINGS) == key
    # A forced test the problem does not hold constrains nothing.
    assert cover.problem_key(TOKENS, COSTS, **{
        **SETTINGS, "forced_ids": frozenset({"other.py::test_z"})}) == key
    changed_tokens = {**TOKENS, "t.py::test_d": frozenset({"L:3", "L:4"})}
    changed = [
        cover.problem_key(changed_tokens, COSTS, **SETTINGS),
        cover.problem_key(TOKENS, {**COSTS, "t.py::test_d": 2.4}, **SETTINGS),
        cover.problem_key(TOKENS, COSTS, **{
            **SETTINGS, "forced_ids": frozenset({"t.py::test_d"})}),
        cover.problem_key(TOKENS, COSTS, **{**SETTINGS, "time_limit_s": 30.0}),
        cover.problem_key(TOKENS, COSTS, **{
            **SETTINGS, "minimum_test_runtime_ms": 0.5}),
        cover.problem_key(TOKENS, COSTS, **{**SETTINGS, "random_seed": 2}),
        cover.problem_key(TOKENS, COSTS, **{
            **SETTINGS, "tie_break": "name_order"}),
    ]
    assert all(other != key for other in changed)
    assert len({json.dumps(other, sort_keys=True) for other in changed}) == len(changed)
