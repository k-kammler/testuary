"""The exact and greedy lossless set-cover reducers: redundancy removal, cost
weighting, tie-breaks, forced tests and solver records, and the invariants the
caller relies on (the token union is kept and every removed test is redundant
against the kept set)."""
from __future__ import annotations

import math
import random
from concurrent.futures import ThreadPoolExecutor

import pytest

from testuary.reduction import (
    EQUAL_COST_MS, ExactCoverError, exact_cover,
    lossless_cover, reduction_costs, solve_cover,
)


def _cost(ids_to_cost):
    return dict(ids_to_cost)


def test_redundant_test_is_removed_and_tokens_preserved():
    """C's tokens are a subset of A union B, so C is removed and the token
    union is unchanged."""
    tokens = {
        "A": frozenset({"L:1", "L:2"}),
        "B": frozenset({"L:3"}),
        "C": frozenset({"L:1", "L:3"}),  # covered by A and B
    }
    cost = _cost({"A": 1.0, "B": 1.0, "C": 1.0})
    r = lossless_cover(tokens, cost)
    assert "C" in r.remove
    assert set(r.keep) == {"A", "B"}
    assert len(_union(tokens, r.keep)) == 3


def test_cost_weighting_prefers_cheaper_test_for_same_coverage():
    """Two tests cover the same tokens. The cheaper one is kept."""
    tokens = {
        "expensive": frozenset({"L:1", "L:2"}),
        "cheap": frozenset({"L:1", "L:2"}),
    }
    cost = _cost({"expensive": 10.0, "cheap": 1.0})
    r = lossless_cover(tokens, cost)
    assert r.keep == ("cheap",)
    assert r.remove == ("expensive",)
    assert len(_union(tokens, r.keep)) == 2


def test_tie_break_is_deterministic_by_id():
    """With equal tokens and cost the lexicographically smaller id is kept."""
    tokens = {"b_test": frozenset({"L:1"}), "a_test": frozenset({"L:1"})}
    cost = _cost({"a_test": 1.0, "b_test": 1.0})
    first = lossless_cover(tokens, cost)
    again = lossless_cover(tokens, cost)
    assert first == again
    assert first.keep == ("a_test",)
    assert first.remove == ("b_test",)


def test_zero_token_test_is_always_removed():
    """A test that holds no tokens is pure overhead and is removed."""
    tokens = {
        "real": frozenset({"L:1"}),
        "empty": frozenset(),
    }
    cost = _cost({"real": 5.0, "empty": 1.0})
    r = lossless_cover(tokens, cost)
    assert "empty" in r.remove
    assert "real" in r.keep


def test_zero_cost_test_covering_new_tokens_is_kept():
    """A zero-time test that covers a unique token still gets selected (the
    cost floor prevents division by zero and ranks free coverage first)."""
    tokens = {"free": frozenset({"L:9"}), "other": frozenset({"L:1"})}
    cost = _cost({"free": 0.0, "other": 1.0})
    r = lossless_cover(tokens, cost)
    assert set(r.keep) == {"free", "other"}
    assert r.remove == ()


def test_disjoint_tests_are_all_kept():
    """Each test covers a unique token, so none is redundant."""
    tokens = {
        "A": frozenset({"L:1"}),
        "B": frozenset({"L:2"}),
        "C": frozenset({"L:3"}),
    }
    cost = _cost({"A": 1.0, "B": 1.0, "C": 1.0})
    r = lossless_cover(tokens, cost)
    assert set(r.keep) == {"A", "B", "C"}
    assert r.remove == ()


def _kept_cost(cover, cost):
    """The summed cost of the tests ``cover`` keeps."""
    return sum(cost.get(test, 0.0) for test in cover.keep)


def test_empty_input():
    r = lossless_cover({}, {})
    assert (r.keep, r.remove) == ((), ())


@pytest.mark.parametrize("cost", [float("nan"), float("inf"), -1.0])
def test_the_greedy_cover_refuses_a_cost_the_exact_cover_refuses(cost):
    """A cost that is no finite number at or above zero would leave the
    test out of every comparison, and the cover empty."""
    with pytest.raises(ValueError, match="invalid time for reduction test 'a'"):
        lossless_cover({"a": frozenset({"L:f:1"})}, {"a": cost})


def test_a_time_tier_answer_that_leaves_a_line_uncovered_is_named_by_time(monkeypatch):
    """A solver answer whose least-time cover leaves a line uncovered ends the
    exact cover with an error that names the time tier."""
    import pulp

    monkeypatch.setattr(pulp.LpVariable, "value", lambda self: 0.0)
    with pytest.raises(ExactCoverError) as refused:
        exact_cover({"a": frozenset({"L:1"}), "b": frozenset({"L:2"})}, {"a": 1.0, "b": 1.0})
    assert str(refused.value) == "time tier returned selected ids that do not cover the universe"
    assert refused.value.metadata["failure"] == "runtime_selected_ids_do_not_cover_universe"


def test_a_chosen_cover_whose_time_does_not_replay_is_named_by_time(monkeypatch):
    """A chosen cover whose summed time differs from the least time ends the
    exact cover with an error that names the time objective."""
    import testuary.reduction as reduction

    monkeypatch.setattr(reduction, "_break_ties_by_name", lambda keep_set, **_: {"b"})
    tokens = {"a": frozenset({"L:1"}), "b": frozenset({"L:1", "L:2"}), "c": frozenset({"L:2"})}
    with pytest.raises(ExactCoverError) as refused:
        exact_cover(tokens, {"a": 1.0, "b": 5.0, "c": 1.0})
    assert str(refused.value) == "exact cover time objective does not replay from selected ids"
    assert refused.value.metadata["failure"] == "objective_replay_mismatch"


def test_removed_tests_are_redundant_against_kept_set():
    """The safety invariant the caller relies on: every removed test's
    tokens are a subset of the kept tests' union, and the token count is
    preserved, while time does not increase."""
    tokens = {
        "t1": frozenset({"L:1", "L:2", "B:1->2"}),
        "t2": frozenset({"L:2", "L:3"}),
        "t3": frozenset({"L:1"}),            # subset of t1
        "t4": frozenset({"L:3", "B:1->2"}),  # subset of t1 union t2
        "t5": frozenset({"M:7"}),            # unique mutant token
    }
    cost = _cost({"t1": 2.0, "t2": 2.0, "t3": 1.0, "t4": 1.0, "t5": 3.0})
    r = lossless_cover(tokens, cost)

    kept_union = set()
    for t in r.keep:
        kept_union |= tokens[t]
    for t in r.remove:
        assert tokens[t] <= kept_union, f"{t} would lose tokens"

    assert kept_union == set().union(*tokens.values())
    # t5 holds the only M:7 token, so it can never be removed.
    assert "t5" in r.keep


# Invariants checked across random instances.


def _random_instance(rng: random.Random):
    """A random per-test token table and ms-scale costs, including an empty
    universe, all-redundant tests and zero costs."""
    n_tests = rng.randint(1, 8)
    n_tokens = rng.randint(0, 10)
    universe = [f"k{i}" for i in range(n_tokens)]
    tokens, cost = {}, {}
    for i in range(n_tests):
        tid = f"t{i}"
        k = rng.randint(0, n_tokens)
        tokens[tid] = frozenset(rng.sample(universe, k)) if n_tokens else frozenset()
        cost[tid] = rng.choice([0.0, 5_000.0, 15_000.0, 40_000.0]) * rng.random()
    return tokens, cost


def _union(tokens, ids):
    out = set()
    for t in ids:
        out |= set(tokens[t])
    return out


def test_lossless_cover_invariants_hold_on_random_inputs():
    """Across random token tables the cover (a) partitions the ids, (b) keeps
    the full token union, and (c) leaves every removed test redundant against
    the kept set."""
    rng = random.Random(20260608)
    for _ in range(3000):
        tokens, cost = _random_instance(rng)
        res = lossless_cover(tokens, cost)
        universe = _union(tokens, tokens)
        # (a) keep / remove partition the input ids
        assert set(res.keep) | set(res.remove) == set(tokens)
        assert not (set(res.keep) & set(res.remove))
        # (b) lossless: the kept set is an exact cover of the universe
        kept_union = _union(tokens, res.keep)
        assert kept_union == universe
        # (c) every removed test is redundant against the kept set
        for t in res.remove:
            assert set(tokens[t]) <= kept_union


def _optimal_cover_cost(tokens, cost):
    """The minimum cost of a cover of the universe, by brute force
    (exponential, small inputs only)."""
    ids = list(tokens)
    universe = _union(tokens, ids)
    best = math.inf
    for mask in range(1 << len(ids)):
        chosen = [ids[i] for i in range(len(ids)) if mask & (1 << i)]
        if _union(tokens, chosen) == universe:
            best = min(best, sum(cost[t] for t in chosen))
    return best


def test_lossless_cover_is_within_the_chvatal_bound_of_optimal():
    """Greedy weighted set cover is within ``H(n) * OPT`` of the optimal cover
    (Chvatal 1979, n = universe size, H the harmonic number), checked against a
    brute-forced optimum on small instances with strictly positive costs."""
    rng = random.Random(99)
    checked = 0
    for _ in range(4000):
        n_tests = rng.randint(1, 6)
        n_tokens = rng.randint(1, 6)
        universe_ids = [f"k{i}" for i in range(n_tokens)]
        tokens, cost = {}, {}
        for i in range(n_tests):
            tid = f"t{i}"
            k = rng.randint(0, n_tokens)
            tokens[tid] = frozenset(rng.sample(universe_ids, k))
            cost[tid] = rng.uniform(0.1, 10.0)  # strictly positive: the bound's domain
        universe = _union(tokens, tokens)
        if not universe:
            continue
        res = lossless_cover(tokens, cost)
        opt = _optimal_cover_cost(tokens, cost)
        n = len(universe)
        h_n = sum(1.0 / i for i in range(1, n + 1))
        assert _kept_cost(res, cost) <= h_n * opt + 1e-9
        checked += 1
    assert checked > 500  # the random draw produced plenty of coverable instances


# The exact cover, the solver dispatch and forced tests.


def test_exact_cover_is_lossless_and_at_most_greedy_cost_on_random_inputs():
    """exact_cover keeps the token union, partitions the ids, and never costs
    more than the greedy cover, since it solves the same problem optimally."""
    rng = random.Random(2024)
    cheaper = 0
    for _ in range(300):
        tokens, cost = _random_instance(rng)
        e = exact_cover(tokens, cost)
        g = lossless_cover(tokens, cost)
        universe = _union(tokens, tokens)
        assert set(e.keep) | set(e.remove) == set(tokens)
        assert not (set(e.keep) & set(e.remove))
        assert _union(tokens, e.keep) == universe
        assert _kept_cost(e, cost) <= _kept_cost(g, cost) + 1e-9
        if _kept_cost(e, cost) < _kept_cost(g, cost) - 1e-9:
            cheaper += 1
    assert cheaper > 0  # on some instances exact is strictly cheaper than greedy


def test_exact_cover_respects_forced_ids():
    """Forced tests are always kept, so a removable test whose tokens a forced
    test covers is removed. A test in a file the eviction may not rewrite is
    forced in."""
    tokens = {
        "F": frozenset({"1", "2"}),  # not rewritable by this pass, forced in
        "T": frozenset({"1", "2"}),  # rewritable, redundant given F
        "U": frozenset({"3"}),
    }
    cost = {"F": 5.0, "T": 1.0, "U": 1.0}
    e = exact_cover(tokens, cost, forced_ids=frozenset({"F"}))
    assert "F" in e.keep            # forced kept despite being the costlier holder
    assert "T" in e.remove          # redundant given the forced F
    assert _union(tokens, e.keep) == {"1", "2", "3"}


def test_greedy_cover_respects_forced_ids():
    tokens = {
        "F": frozenset({"1", "2"}), "T": frozenset({"1", "2"}),
        "U": frozenset({"3"}),
    }
    cost = {"F": 5.0, "T": 1.0, "U": 1.0}
    g = lossless_cover(tokens, cost, forced_ids=frozenset({"F"}))
    assert "F" in g.keep and "T" in g.remove
    assert _union(tokens, g.keep) == {"1", "2", "3"}


def test_exact_cover_is_deterministic():
    rng = random.Random(5)
    tokens, cost = _random_instance(rng)
    a = exact_cover(tokens, cost)
    b = exact_cover(tokens, cost)
    assert a.keep == b.keep and a.remove == b.remove


def test_exact_cover_is_stable_under_input_key_order():
    """Among equal-cost optimal covers the choice does not depend on the input
    mapping's key order, so processes with different hash seeds agree."""
    # Two equal-cost covers exist: {A, B} and {C, D} both cost 2.0 and both
    # cover {1, 2}. Without canonical ordering the pick would track key order.
    base = {
        "A": frozenset({"1"}), "B": frozenset({"2"}),
        "C": frozenset({"1", "2"}), "D": frozenset({"1", "2"}),
    }
    cost = {"A": 1.0, "B": 1.0, "C": 2.0, "D": 2.0}
    keeps = set()
    for order in (["A", "B", "C", "D"], ["D", "C", "B", "A"],
                  ["C", "A", "D", "B"], ["B", "D", "A", "C"]):
        shuffled = {k: base[k] for k in order}
        keeps.add(exact_cover(shuffled, cost).keep)
    assert len(keeps) == 1, keeps


def test_exact_cover_minimizes_test_count_after_timing():
    tokens = {
        "A": frozenset({"1"}),
        "B": frozenset({"2"}),
        "C": frozenset({"1", "2"}),
    }
    result = exact_cover(tokens, {"A": 1.0, "B": 1.0, "C": 2.0})
    assert result.keep == ("C",)
    assert result.runtime_objective_us == 2000
    assert result.test_count_objective == 1


def test_exact_cover_costs_are_integer_microseconds_with_a_floor():
    tokens = {
        "rounded_down": frozenset({"1"}),
        "rounded_up": frozenset({"2"}),
        "zero": frozenset({"3"}),
    }
    result = exact_cover(
        tokens,
        {"rounded_down": 0.0004, "rounded_up": 0.0015, "zero": 0.0},
    )
    assert dict(result.effective_cost_us) == {
        "rounded_down": 1,
        "rounded_up": 2,
        "zero": 1,
    }
    assert result.runtime_objective_us == 4


@pytest.mark.parametrize(
    "costs, match",
    [
        ({}, "missing time"),
        ({"A": -1.0}, "invalid time"),
        ({"A": math.inf}, "invalid time"),
        ({"A": math.nan}, "invalid time"),
    ],
)
def test_exact_cover_rejects_incomplete_or_invalid_costs(costs, match):
    with pytest.raises(ValueError, match=match):
        exact_cover({"A": frozenset({"1"})}, costs)


def test_reduction_costs_keep_measured_times_and_impute_the_rest():
    """Measured times stay as measured. A test without one costs the slowest
    measured time, and without any time every test costs the same."""
    tokens = {"a": frozenset({"x"}), "b": frozenset({"x"}),
              "c": frozenset({"y"})}
    assert reduction_costs(tokens, {"b": 3.0, "c": 10.0}) == (
        {"b": 3.0, "c": 10.0}, {"a": 10.0})
    assert reduction_costs(tokens, {"a": 1.0, "b": 3.0, "c": 10.0}) == (
        {"a": 1.0, "b": 3.0, "c": 10.0}, {})
    assert reduction_costs(tokens, {}) == (
        {}, {t: EQUAL_COST_MS for t in tokens})


def test_a_test_without_a_time_is_not_free():
    """The untimed duplicate costs the slowest measured time, so the measured,
    cheaper duplicate is kept. The record holds measured and imputed costs
    apart. A cost of 0.0 ms would keep the untimed test."""
    tokens = {"a": frozenset({"x"}), "b": frozenset({"x"}),
              "c": frozenset({"y"})}
    measured, imputed = reduction_costs(tokens, {"b": 3.0, "c": 10.0})
    cover = exact_cover(tokens, measured, imputed_cost_ms=imputed)
    assert cover.keep == ("b", "c")
    metadata = cover.solver_metadata
    assert metadata["raw_cost_ms"] == {"b": 3.0, "c": 10.0}
    assert metadata["imputed_cost_ms"] == {"a": 10.0}
    assert metadata["cost_basis"] == "imputed"
    greedy = solve_cover(tokens, measured, imputed_cost_ms=imputed, solver="greedy")
    assert greedy.keep == ("b", "c")
    # The greedy cover records the costs that decided it, as the exact one.
    assert greedy.solver_metadata == {
        "solver": "greedy", "raw_cost_ms": {"b": 3.0, "c": 10.0},
        "imputed_cost_ms": {"a": 10.0}, "cost_basis": "imputed",
    }
    assert exact_cover(tokens, {"a": 0.0, "b": 3.0, "c": 10.0}).keep == (
        "a", "c")


def test_without_any_time_the_reduction_minimizes_the_count():
    tokens = {"a": frozenset({"x"}), "b": frozenset({"x", "y"}),
              "c": frozenset({"y"})}
    measured, imputed = reduction_costs(tokens, {})
    cover = exact_cover(tokens, measured, imputed_cost_ms=imputed)
    assert cover.keep == ("b",)
    metadata = cover.solver_metadata
    assert metadata["raw_cost_ms"] == {}
    assert metadata["cost_basis"] == "equal"
    assert metadata["test_count_objective"] == 1


def test_measured_costs_leave_the_record_as_it_was():
    cover = exact_cover({"a": frozenset({"x"})}, {"a": 2.0})
    assert "imputed_cost_ms" not in cover.solver_metadata
    assert "cost_basis" not in cover.solver_metadata


def test_exact_cover_records_replayable_solver_metadata():
    result = exact_cover(
        {"A": frozenset({"1"}), "B": frozenset({"1"})},
        {"A": 2.0, "B": 1.0},
        tie_break="solver",
    )
    metadata = result.solver_metadata
    assert metadata is not None
    assert metadata["selected_ids"] == ["B"]
    assert metadata["runtime_objective_us"] == 1000
    assert metadata["test_count_objective"] == 1
    assert [tier["name"] for tier in metadata["tiers"]] == [
        "runtime", "test_count",
    ]
    assert all(tier["problem_status"] == "Optimal"
               for tier in metadata["tiers"])
    assert all(tier["solution_status"] == "Optimal Solution Found"
               for tier in metadata["tiers"])
    assert metadata["pulp_version"] == "3.3.2"
    assert metadata["cbc_version"] == "2.10.3"
    assert metadata["cbc_options"] == [
        "randomSeed 1", "randomCbcSeed 1",
    ]


# The tie-break among equally good covers.


def test_name_order_counts_exact_duplicates_once_and_keeps_the_first():
    """B and C hold the same tokens at the same cost: they are one candidate,
    the first by name."""
    tokens = {"A": frozenset({"1"}), "C": frozenset({"2"}),
              "B": frozenset({"2"})}
    result = exact_cover(tokens, {"A": 1.0, "B": 1.0, "C": 1.0})
    assert result.keep == ("A", "B")
    assert result.remove == ("C",)
    assert result.solver_metadata["duplicate_tests"] == {"B": ["C"]}


def test_name_order_keeps_a_forced_duplicate():
    """A forced test stands for its duplicates, and the unforced one goes."""
    tokens = {"A": frozenset({"1"}), "F": frozenset({"1"}),
              "U": frozenset({"2"})}
    result = exact_cover(tokens, {"A": 1.0, "F": 1.0, "U": 1.0},
                         forced_ids=frozenset({"F"}))
    assert result.keep == ("F", "U")
    assert result.solver_metadata["duplicate_tests"] == {"F": ["A"]}


def test_name_order_prefers_the_least_sum_of_name_ranks():
    """Three covers of two tests cost the same. {a, b} has the least rank sum,
    and one more solve proves no other cover equals it."""
    tokens = {"a": frozenset({"1", "2"}), "b": frozenset({"1", "3"}),
              "c": frozenset({"2", "3"})}
    result = exact_cover(tokens, {"a": 1.0, "b": 1.0, "c": 1.0})
    metadata = result.solver_metadata
    assert result.keep == ("a", "b")
    assert metadata["tie_break"] == "name_order"
    assert metadata["rank_sum_objective"] == 3
    assert metadata["rank_square_sum_objective"] == 5
    assert metadata["uniqueness"] == "proven"
    assert metadata["equally_good_selections"] == 1
    assert [tier["name"] for tier in metadata["tiers"]] == [
        "runtime", "test_count", "rank_sum", "rank_square_sum", "uniqueness",
    ]
    uniqueness = metadata["tiers"][-1]
    assert uniqueness["problem_status"] == "Infeasible"
    assert all("preprocess off" in tier["cbc_options"]
               for tier in metadata["tiers"][1:])


def test_name_order_breaks_an_equal_rank_sum_by_squares():
    """{a, d} and {b, c} are the only covers of two tests, with the same rank
    sum 5. The sums of squares, 17 and 13, decide."""
    tokens = {"a": frozenset({"1", "2"}), "b": frozenset({"1", "3"}),
              "c": frozenset({"2", "4"}), "d": frozenset({"3", "4"})}
    result = exact_cover(tokens, {t: 1.0 for t in tokens})
    assert result.keep == ("b", "c")
    assert result.solver_metadata["rank_sum_objective"] == 5
    assert result.solver_metadata["rank_square_sum_objective"] == 13
    assert result.solver_metadata["uniqueness"] == "proven"


# Two exact covers equal in all four objectives: ranks {1, 5, 6} and
# {2, 3, 7} both sum to 12 with squares summing to 62. t4 is never needed.
_FOUR_WAY_TIE = {
    "t1": frozenset({"u1", "u2"}), "t5": frozenset({"u3", "u4"}),
    "t6": frozenset({"u5", "u6"}), "t2": frozenset({"u1", "u3"}),
    "t3": frozenset({"u2", "u5"}), "t7": frozenset({"u4", "u6"}),
    "t4": frozenset({"u1"}),
}


def test_name_order_lists_the_covers_equal_in_every_tier():
    """The uniqueness solve finds the other cover, every equally good cover
    is listed, each step proven, and the first in name order is chosen."""
    result = exact_cover(_FOUR_WAY_TIE, {t: 1.0 for t in _FOUR_WAY_TIE})
    metadata = result.solver_metadata
    assert result.keep == ("t1", "t5", "t6")
    assert metadata["rank_sum_objective"] == 12
    assert metadata["rank_square_sum_objective"] == 62
    assert metadata["uniqueness"] == "resolved_by_name_order"
    assert metadata["equally_good_selections"] == 2
    names = [tier["name"] for tier in metadata["tiers"]]
    assert names[:5] == [
        "runtime", "test_count", "rank_sum", "rank_square_sum", "uniqueness",
    ]
    assert names[5:] == ["name_order_1"]
    assert metadata["tiers"][-1]["problem_status"] == "Infeasible"


def test_name_order_does_not_depend_on_input_order():
    costs = {t: 1.0 for t in _FOUR_WAY_TIE}
    keeps = {
        exact_cover(dict(reversed(list(_FOUR_WAY_TIE.items()))), costs).keep,
        exact_cover(dict(sorted(_FOUR_WAY_TIE.items())), costs).keep,
    }
    assert keeps == {("t1", "t5", "t6")}


def _unsolved_when_named(monkeypatch, *names):
    """Every solve of a problem with one of ``names`` returns without a
    result, as a solve that ran out of time does."""
    import pulp

    original = pulp.LpProblem.solve

    def solve(self, solver=None, **kwargs):
        if self.name in names:
            self.assignStatus(pulp.LpStatusNotSolved,
                              pulp.LpSolutionNoSolutionFound)
            return self.status
        return original(self, solver, **kwargs)

    monkeypatch.setattr(pulp.LpProblem, "solve", solve)


def test_the_solver_metadata_names_cbc_by_its_version_not_its_path():
    """The records hold CBC's version and no path of the machine."""
    import pulp

    path = str(pulp.PULP_CBC_CMD(msg=False, _skip_v4_deprecation=True).path)
    metadata = exact_cover({"a": frozenset({"1"})}, {"a": 1.0}).solver_metadata
    assert "cbc_version" in metadata
    assert "cbc_path" not in metadata
    assert path not in repr(metadata)


def test_a_cbc_that_fails_to_run_raises_exact_cover_error(monkeypatch):
    """A CBC that cannot run raises ExactCoverError naming the tier, with
    the metadata the phase end's failure record holds."""
    import pulp

    def solve(self, solver=None, **kwargs):
        raise pulp.PulpSolverError(
            "Pulp: Error while executing /opt/solvers/cbc in /home/someone/work")

    monkeypatch.setattr(pulp.LpProblem, "solve", solve)
    with pytest.raises(ExactCoverError) as caught:
        exact_cover({"a": frozenset({"1"})}, {"a": 1.0})
    # The recorded error names no path of the machine.
    assert str(caught.value) == "CBC failed in runtime (PulpSolverError)"
    assert caught.value.metadata["failure"] == "solver_error"
    assert caught.value.metadata["failed_tier"] == "runtime"


def test_the_cbc_version_probe_is_not_charged_to_the_cover_time(monkeypatch):
    """An exact solve probes CBC's version before the cover time starts, so
    a process's first cover time holds the solve alone."""
    from testuary import phase_state, reduction

    probed = []

    def probe(executable):
        probed.append(phase_state.timing_open())
        return "2.10.3", ""

    monkeypatch.setattr(reduction, "cbc_version", probe)
    result = exact_cover({"a": frozenset({"1"})}, {"a": 1.0})
    assert probed == [False]
    assert result.solver_metadata["cbc_version"] == "2.10.3"


def test_an_unproven_uniqueness_solve_is_not_a_proof(monkeypatch):
    """Without a result on uniqueness the equally good covers are listed.
    Here the listing proves that the chosen one is the only one."""
    _unsolved_when_named(monkeypatch, "uniqueness")
    tokens = {"a": frozenset({"1", "2"}), "b": frozenset({"1", "3"}),
              "c": frozenset({"2", "3"})}
    result = exact_cover(tokens, {"a": 1.0, "b": 1.0, "c": 1.0})
    metadata = result.solver_metadata
    assert result.keep == ("a", "b")
    assert metadata["uniqueness"] == "resolved_by_name_order"
    assert metadata["equally_good_selections"] == 1
    assert [tier["name"] for tier in metadata["tiers"]][-2:] == [
        "uniqueness", "name_order_1",
    ]


def test_a_tie_without_a_proof_fails_the_exact_reduction(monkeypatch):
    """A listing step without a result leaves the tie open: the exact
    reduction fails and names what it could not prove."""
    _unsolved_when_named(monkeypatch, "uniqueness", "name_order_1")
    with pytest.raises(ExactCoverError, match="first in name order") as info:
        exact_cover(_FOUR_WAY_TIE, {t: 1.0 for t in _FOUR_WAY_TIE})
    assert info.value.metadata["failure"] == "tie_not_resolved"
    assert info.value.metadata["failed_tier"] == "name_order_1"


def test_a_time_limit_spent_before_the_uniqueness_tier_is_recorded_as_such(
        monkeypatch):
    """When the time limit is spent before the uniqueness tier, the failure
    names the time limit and that tier, not an open tie of a tier that
    never ran."""
    import time
    from types import SimpleNamespace

    import pulp
    from testuary import reduction

    offset = [0.0]
    monkeypatch.setattr(reduction, "time",
                        SimpleNamespace(monotonic=lambda: time.monotonic() + offset[0]))
    original = pulp.LpProblem.solve

    def solve(self, solver=None, **kwargs):
        status = original(self, solver, **kwargs)
        if self.name == "min_rank_square_sum_optimal_set_cover":
            offset[0] += 1000.0
        return status

    monkeypatch.setattr(pulp.LpProblem, "solve", solve)
    with pytest.raises(ExactCoverError, match="time limit before uniqueness") as info:
        exact_cover(_FOUR_WAY_TIE, {t: 1.0 for t in _FOUR_WAY_TIE})
    assert info.value.metadata["failure"] == "time_limit_exhausted"
    assert info.value.metadata["failed_tier"] == "uniqueness"


def test_solver_tie_break_keeps_the_two_tiers_and_every_test():
    """The solver tie-break: no duplicate is removed from the model and
    the solver's optimal cover stands."""
    tokens = {"A": frozenset({"1"}), "B": frozenset({"1"}),
              "C": frozenset({"2"})}
    result = exact_cover(tokens, {"A": 1.0, "B": 1.0, "C": 1.0},
                         tie_break="solver")
    metadata = result.solver_metadata
    assert metadata["tie_break"] == "solver"
    assert [tier["name"] for tier in metadata["tiers"]] == [
        "runtime", "test_count",
    ]
    assert "duplicate_tests" not in metadata
    assert "uniqueness" not in metadata
    assert result.test_count_objective == 2 and "C" in result.keep


def test_an_unknown_tie_break_is_refused():
    with pytest.raises(ValueError, match="reduction_tie_break"):
        exact_cover({"A": frozenset({"1"})}, {"A": 1.0}, tie_break="first")


def test_solve_cover_passes_the_tie_break_to_the_exact_solver():
    tokens = {"a": frozenset({"1", "2"}), "b": frozenset({"1", "3"}),
              "c": frozenset({"2", "3"})}
    costs = {"a": 1.0, "b": 1.0, "c": 1.0}
    assert solve_cover(tokens, costs).solver_metadata["tie_break"] == (
        "name_order")
    assert solve_cover(tokens, costs, tie_break="solver").solver_metadata[
        "tie_break"] == "solver"


@pytest.mark.parametrize("tie_break, solves", [
    ("solver", 2),       # runtime, test count
    ("name_order", 5),   # and rank sum, sum of squares, uniqueness
])
def test_exact_cover_passes_pinned_cbc_settings_and_shared_timeout(
    monkeypatch, tie_break, solves,
):
    import pulp

    real = pulp.PULP_CBC_CMD
    calls = []

    def recording_solver(*args, **kwargs):
        calls.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(pulp, "PULP_CBC_CMD", recording_solver)
    exact_cover({"A": frozenset({"1"})}, {"A": 1.0}, time_limit_s=5.0,
                tie_break=tie_break)
    solve_calls = [call for call in calls if "timeLimit" in call]
    assert len(solve_calls) == solves
    # No -threads: CBC's serial search, which has no worker to lose a wakeup.
    assert all(call.get("threads") is None for call in solve_calls)
    assert solve_calls[0]["options"] == [
        "randomSeed 1", "randomCbcSeed 1",
    ]
    assert all(call["options"] == [
        "randomSeed 1", "randomCbcSeed 1", "preprocess off",
    ] for call in solve_calls[1:])
    limits = [call["timeLimit"] for call in solve_calls]
    assert limits == sorted(limits, reverse=True)
    assert 0 < limits[-1] and limits[0] <= 5


def test_exact_cover_requires_optimal_problem_and_solution_status(monkeypatch):
    import pulp

    def feasible_only(self, *args, **kwargs):
        self.status = pulp.LpStatusOptimal
        self.sol_status = pulp.LpSolutionIntegerFeasible

    monkeypatch.setattr(pulp.LpProblem, "solve", feasible_only)
    with pytest.raises(ExactCoverError) as caught:
        exact_cover({"A": frozenset({"1"})}, {"A": 1.0})
    assert caught.value.metadata["failed_tier"] == "runtime"
    assert caught.value.metadata["tiers"][0]["problem_status"] == "Optimal"
    assert caught.value.metadata["tiers"][0]["solution_status"] == "Solution Found"


def test_solve_cover_dispatches_and_rejects_unknown_solver():
    tokens = {"A": frozenset({"1"}), "B": frozenset({"1"})}
    cost = {"A": 2.0, "B": 1.0}
    assert solve_cover(tokens, cost, solver="greedy").keep == ("B",)
    assert solve_cover(tokens, cost, solver="exact").keep == ("B",)
    assert solve_cover(tokens, cost).solver == "exact"
    with pytest.raises(ValueError):
        solve_cover(tokens, cost, solver="bogus")


def test_exact_cover_hard_fails_without_pulp(monkeypatch):
    """exact is the default: without PuLP it raises and never falls back to
    the greedy heuristic."""
    import builtins

    real_import = builtins.__import__

    def no_pulp(name, *args, **kwargs):
        if name == "pulp":
            raise ImportError("pulp not installed")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", no_pulp)
    with pytest.raises(ImportError):
        exact_cover({"A": frozenset({"1"})}, {"A": 1.0})


def test_exact_cover_cost_equals_the_bruteforce_optimum_on_random_inputs():
    """The exact cover's cost equals the brute-forced minimum over all covering
    subsets. Zero costs make the optimal set non-unique, so the cost is
    compared. Forced ids restrict the brute force to supersets of them."""
    rng = random.Random(20260612)
    strictly_below_greedy = 0
    for _ in range(150):
        n_tests = rng.randint(1, 7)
        n_tokens = rng.randint(0, 7)
        universe_ids = [f"k{i}" for i in range(n_tokens)]
        tokens, cost = {}, {}
        for i in range(n_tests):
            k = rng.randint(0, n_tokens)
            tokens[f"t{i}"] = (
                frozenset(rng.sample(universe_ids, k))
                if n_tokens else frozenset()
            )
            cost[f"t{i}"] = rng.choice([0.0, rng.uniform(0.1, 10.0)])
        forced = frozenset(t for t in tokens if rng.random() < 0.2)

        e = exact_cover(tokens, cost, forced_ids=forced)

        ids = list(tokens)
        universe = _union(tokens, ids)
        best = math.inf
        for mask in range(1 << len(ids)):
            chosen = {ids[i] for i in range(len(ids)) if mask & (1 << i)}
            if not forced <= chosen:
                continue
            if _union(tokens, chosen) == universe:
                best = min(best, sum(cost[t] for t in chosen))

        assert forced <= set(e.keep)
        assert _union(tokens, e.keep) == universe
        assert abs(_kept_cost(e, cost) - best) <= 1e-6

        g = lossless_cover(tokens, cost, forced_ids=forced)
        if _kept_cost(g, cost) > _kept_cost(e, cost) + 1e-9:
            strictly_below_greedy += 1
    assert strictly_below_greedy > 0


def test_exact_cover_is_stable_across_serial_and_concurrent_solves():
    tokens = {
        "A": frozenset({"L:1", "B:1->2"}),
        "B": frozenset({"L:1", "K:m1"}),
        "C": frozenset({"B:1->2", "K:m1"}),
    }
    costs = {"A": 2.0, "B": 2.0, "C": 2.0}

    def solve(_label):
        result = exact_cover(tokens, costs)
        return result.keep, result.runtime_objective_us, result.test_count_objective

    serial = [solve(0), solve(1)]
    with ThreadPoolExecutor(max_workers=2) as pool:
        concurrent = list(pool.map(solve, (2, 3)))

    assert len(set(serial + concurrent)) == 1


def test_exact_cover_raises_when_the_solve_yields_no_solution(monkeypatch):
    """A solve without an incumbent (every variable None) raises instead of
    returning an incomplete cover."""
    import pulp

    monkeypatch.setattr(
        pulp.LpProblem, "solve", lambda self, *a, **k: None,
    )
    with pytest.raises(ExactCoverError, match="did not establish"):
        exact_cover({"A": frozenset({"1"})}, {"A": 1.0})


def test_exact_cover_does_not_modify_warning_filters(monkeypatch):
    import warnings

    import pulp

    filters = warnings.filters
    entries = list(filters)
    showwarning = warnings.showwarning
    real = pulp.PULP_CBC_CMD
    calls = []

    def solver(*args, **kwargs):
        assert warnings.filters is filters
        assert warnings.filters == entries
        assert warnings.showwarning is showwarning
        calls.append(dict(kwargs))
        return real(*args, **kwargs)

    monkeypatch.setattr(pulp, "PULP_CBC_CMD", solver)
    result = exact_cover({"A": frozenset({"1"})}, {"A": 1.0})

    assert result.keep == ("A",)
    assert len(calls) == 6      # the path probe and five solves
    assert all(call["_skip_v4_deprecation"] is True for call in calls)
    assert warnings.filters is filters
    assert warnings.filters == entries
    assert warnings.showwarning is showwarning


def test_exact_cover_keeps_warning_state_during_overlapping_calls(monkeypatch):
    import threading
    import warnings

    import pulp

    first_entered = threading.Event()
    second_entered = threading.Event()
    first_finished = threading.Event()
    local = threading.local()
    original = pulp.LpVariable
    filters = warnings.filters
    entries = list(filters)
    showwarning = warnings.showwarning

    def variable(*args, **kwargs):
        if not local.entered:
            local.entered = True
            if local.label == "first":
                first_entered.set()
                assert second_entered.wait(60)
            else:
                second_entered.set()
                assert first_finished.wait(60)
        return original(*args, **kwargs)

    def solve(label):
        local.label, local.entered = label, False
        try:
            result = exact_cover(
                {"A": frozenset({"L:1", "B:1->2"}),
                 "B": frozenset({"L:1", "K:m1"}),
                 "C": frozenset({"B:1->2", "K:m1"})},
                {"A": 2.0, "B": 2.0, "C": 2.0},
            )
            return (result.keep, result.runtime_objective_us,
                    result.test_count_objective)
        finally:
            if label == "first":
                first_finished.set()

    monkeypatch.setattr(pulp, "LpVariable", variable)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(solve, "first")
        assert first_entered.wait(60)
        second = pool.submit(solve, "second")
        results = [first.result(), second.result()]

    # Three covers of two tests cost the same. The least name-rank sum wins.
    assert results == [(("A", "B"), 4000, 2)] * 2
    assert warnings.filters is filters
    assert warnings.filters == entries
    assert warnings.showwarning is showwarning
