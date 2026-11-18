"""Deterministic test-suite reduction over an opaque per-test token table.

Given each test's evidence tokens and cost, it selects a subset whose tokens
together equal the full suite's. The configured solver is an exact binary
program that minimizes the time in integer microseconds, then the number of
tests. By default it chooses among equally good selections by name order,
with a proof that the choice is unique or a proven list of every equally good
selection, so the same input always gives the same suite. A deterministic
greedy heuristic is the other solver setting.

The module knows nothing about coverage, mutation or any language: tokens are
opaque strings and costs are floats. The caller builds both maps from the
per-test evidence, acts on the returned ids, keeps a suite from becoming empty
and decides when to reduce.
"""
from __future__ import annotations

from typing import (
    FrozenSet,
    Mapping,
    Optional,
    Protocol,
    Any,
    Callable,
    Dict,
    List,
    Set,
    Tuple,
)
import functools
import math
import subprocess
import time
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from testuary import phase_state
from testuary.settings import _CHOICES, Slot




class CoverSource(Protocol):
    """What an installed source provides."""

    def solve_exact(
        self,
        per_test_tokens: Mapping[str, FrozenSet[str]],
        per_test_cost_ms: Mapping[str, float],
        *,
        forced_ids: FrozenSet[str],
        time_limit_s: float,
        minimum_test_runtime_ms: float,
        random_seed: int,
        tie_break: str,
        imputed_cost_ms: Optional[Mapping[str, float]],
    ) -> "LosslessReduction":
        """The exact reduction of this problem, in the shape
        ``reduction.exact_cover`` returns."""


_SOURCE: Slot[CoverSource] = Slot("cover source")
install, uninstall, current, installed = (
    _SOURCE.install, _SOURCE.uninstall, _SOURCE.current, _SOURCE.installed)


# Cost floor so a zero-time test that still covers a new token ranks ahead of
# any positive-cost test (free coverage), rather than dividing by zero.
_MIN_COST = 1e-9


def _timed_cover(fn):
    """Charge a solve to the running phase's cover time. It wraps the solvers,
    not ``solve_cover``, so a caller that calls a solver directly is measured
    the same way."""
    @functools.wraps(fn)
    def solve(*args, **kwargs):
        with phase_state.timed_part(phase_state.TIME_COVER):
            return fn(*args, **kwargs)
    return solve


@dataclass(frozen=True)
class LosslessReduction:
    """Result of a reduction, :func:`exact_cover` or :func:`lossless_cover`.

    ``keep`` and ``remove`` partition the input test ids (both sorted). The
    kept tests hold every token. Every removed test is safe to drop: all its
    tokens are held by some kept test."""

    keep: Tuple[str, ...]
    remove: Tuple[str, ...]
    solver: str = "greedy"
    effective_cost_us: Tuple[Tuple[str, int], ...] = ()
    runtime_objective_us: Optional[int] = None
    test_count_objective: Optional[int] = None
    solver_metadata: Optional[Dict[str, Any]] = None


#: The cost of each test of a reduction none of whose tests has a measured
#: time: the same for every test, so the reduction minimizes the number of
#: tests.
EQUAL_COST_MS = 1.0


def first_test_setup(
    per_test_tokens: Mapping[str, FrozenSet[str]],
    per_test_time_ms: Mapping[str, float],
    first_test: Optional[str],
) -> Optional[Dict[str, Any]]:
    """The cost of the run's first test, whose measured time also holds the
    run's one-time setup, or None when there is no such test to adjust.

    A run pays its one-time setup (the runner's and the session's first-use
    costs) in whichever test it runs first, and a reduced suite pays it again
    in its own first test. It is a cost of the suite, not of that test, so
    it is charged to no test. The first test's own time is not observable in
    the run: its measured time is an upper bound of it. It costs the smaller
    of that bound and the slowest other measured test's time, the cost
    imputed to a test without a measured time, so it is neither charged the
    setup nor made cheaper than every other test. A run with no other
    measured test keeps the measured time. The record holds the test, its
    measured time, its cost and the difference charged to no test."""
    if first_test is None or first_test not in per_test_tokens:
        return None
    if first_test not in per_test_time_ms:
        return None
    others = [float(per_test_time_ms[test]) for test in per_test_tokens
              if test != first_test and test in per_test_time_ms]
    if not others:
        return None
    measured = float(per_test_time_ms[first_test])
    cost = min(measured, max(others))
    return {"test": first_test, "measured_ms": measured, "cost_ms": cost,
            "setup_ms": measured - cost}


def reduction_costs(
    per_test_tokens: Mapping[str, FrozenSet[str]],
    per_test_time_ms: Mapping[str, float],
    *,
    first_test: Optional[str] = None,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """The cost of each test a reduction weighs:
    ``(measured, imputed)``. ``measured`` holds the measured times of the
    tests that have one, unchanged. A test without one is not free: it gets
    the time of the slowest measured test, an imputed cost, not an upper
    bound. When no test has a time, every test gets ``EQUAL_COST_MS``, so the
    reduction minimizes the number of tests. ``first_test``, the test the run
    executed first, costs what :func:`first_test_setup` gives, an imputed
    cost, and the slowest measured time is read with that cost in place of
    its measured time. The caller records the imputed costs apart from the
    measured times."""
    measured = {test: float(per_test_time_ms[test]) for test in per_test_tokens
                if test in per_test_time_ms}
    imputed: Dict[str, float] = {}
    setup = first_test_setup(per_test_tokens, per_test_time_ms, first_test)
    if setup is not None:
        del measured[setup["test"]]
        imputed[setup["test"]] = setup["cost_ms"]
    missing = sorted(test for test in per_test_tokens
                     if test not in per_test_time_ms)
    if missing:
        weighed = [*measured.values(), *imputed.values()]
        cost = max(weighed) if weighed else EQUAL_COST_MS
        imputed.update(dict.fromkeys(missing, cost))
    return measured, imputed


class ExactCoverError(RuntimeError):
    """An exact solve that did not prove the required optimum."""

    def __init__(self, message: str, metadata: Mapping[str, Any]):
        super().__init__(message)
        self.metadata = dict(metadata)


def _cost_ms(test_id: str, value: object) -> Decimal:
    """A test's cost in milliseconds, a finite number not below zero, else
    ValueError naming the test."""
    try:
        decimal_ms = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(
            f"invalid time for reduction test {test_id!r}: {value!r}"
        ) from exc
    if not decimal_ms.is_finite() or decimal_ms < 0:
        raise ValueError(
            f"invalid time for reduction test {test_id!r}: {value!r}"
        )
    return decimal_ms


def _effective_costs_us(
    tests: List[str],
    per_test_cost_ms: Mapping[str, float],
    *,
    minimum_ms: float,
) -> Tuple[Dict[str, float], Dict[str, int]]:
    """Validate millisecond costs and convert them to integer microseconds."""
    try:
        minimum_us = int(
            (Decimal(str(minimum_ms)) * Decimal(1000)).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
        )
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(
            f"invalid reduction_min_test_time_ms {minimum_ms!r}"
        ) from exc
    if minimum_us < 1:
        raise ValueError(
            "reduction_min_test_time_ms must round to at least one "
            "microsecond"
        )

    raw: Dict[str, float] = {}
    effective: Dict[str, int] = {}
    for test_id in tests:
        if test_id not in per_test_cost_ms:
            raise ValueError(
                f"missing time for reduction test {test_id!r}"
            )
        decimal_ms = _cost_ms(test_id, per_test_cost_ms[test_id])
        cost_us = int(
            (decimal_ms * Decimal(1000)).quantize(
                Decimal("1"), rounding=ROUND_HALF_UP
            )
        )
        raw[test_id] = float(decimal_ms)
        effective[test_id] = max(minimum_us, cost_us)
    return raw, effective


#: CBC's version by executable, once a probe found it: an executable's
#: version does not change while the process runs, so an exact cover's
#: record adds no probe to its timed solve after the first.
_CBC_VERSIONS: Dict[str, str] = {}


def cbc_version(executable: str) -> Tuple[Optional[str], str]:
    """CBC's version at ``executable``, or None with why the probe failed:
    the error that stopped it, or the exit code and the first output line
    when CBC printed no version. A version found is kept for the process
    (``_CBC_VERSIONS``). A failed probe is not kept, so the next call
    probes again."""
    if executable in _CBC_VERSIONS:
        return _CBC_VERSIONS[executable], ""
    try:
        # CBC reads commands from stdin. An inherited open pipe would keep
        # the probe waiting until its timeout.
        completed = subprocess.run(
            [executable, "-version"], capture_output=True, text=True,
            stdin=subprocess.DEVNULL, timeout=10, check=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"unavailable ({type(exc).__name__}: {exc})"
    output = "\n".join((completed.stdout or "", completed.stderr or ""))
    lines = [line.strip() for line in output.splitlines() if line.strip()]
    for line in lines:
        if line.lower().startswith("version:"):
            _CBC_VERSIONS[executable] = line.split(":", 1)[1].strip()
            return _CBC_VERSIONS[executable], ""
    first = repr(lines[0][:80]) if lines else "no output"
    return None, f"unknown (exit code {completed.returncode}, {first})"


@_timed_cover
def lossless_cover(
    per_test_tokens: Mapping[str, FrozenSet[str]],
    per_test_cost_ms: Mapping[str, float],
    *,
    forced_ids: FrozenSet[str] = frozenset(),
    imputed_cost_ms: Optional[Mapping[str, float]] = None,
) -> LosslessReduction:
    """Greedy weighted set-cover: the kept / removed partition that preserves
    the full token union at reduced cost.

    Selection repeatedly takes the test maximizing (newly covered tokens) /
    cost, breaking ties by test id (ascending) so the result is
    deterministic. A test that covers no new token is never selected, so a
    pure-overhead test (empty token set, or fully redundant) lands in
    ``remove``. With an empty token universe every test is redundant and
    ``keep`` is empty. The caller keeps a suite from becoming empty.

    ``forced_ids`` are kept and cover their tokens first, so only the other
    tests are chosen for the rest. The loops' per-test redundancy report uses
    this for the tests in suite files other than the loop's own test file:
    their tokens count, but they are not candidates for removal, so a test of
    the loop's file whose tokens they already cover counts as redundant. A
    polynomial-time heuristic (the H(n) bound, not the optimum). See
    :func:`exact_cover` for the optimal-cost solver. A test's cost that is
    not a finite number at or above zero raises ValueError. A test without a
    cost costs the minimum. ``imputed_cost_ms`` are weighed like measured
    costs, and the metadata keeps them apart as :func:`exact_cover`'s does:
    ``raw_cost_ms`` the measured times, ``imputed_cost_ms`` and
    ``cost_basis`` the imputed costs."""
    tests = list(per_test_tokens)
    raw_cost, metadata = _greedy_costs(tests, per_test_cost_ms, imputed_cost_ms)
    token_sets: Dict[str, Set[str]] = {
        t: set(per_test_tokens[t]) for t in tests
    }
    universe: Set[str] = set()
    for toks in token_sets.values():
        universe |= toks


    forced = [t for t in tests if t in forced_ids]
    covered: Set[str] = set()
    for t in forced:
        covered |= token_sets[t]
    keep: List[str] = list(forced)
    remaining = set(tests) - set(forced)
    while covered != universe:
        best_id = None
        best_ratio = -1.0
        # sorted() makes the tie-break "lowest id wins" (strict > below).
        for t in sorted(remaining):
            new = len(token_sets[t] - covered)
            if new == 0:
                continue
            cost = max(raw_cost.get(t, 0.0), _MIN_COST)
            ratio = new / cost
            if ratio > best_ratio:
                best_ratio = ratio
                best_id = t
        if best_id is None:
            # No remaining test covers a new token. The rest are redundant.
            break
        keep.append(best_id)
        covered |= token_sets[best_id]
        remaining.discard(best_id)

    keep_set = set(keep)
    remove = [t for t in tests if t not in keep_set]
    return LosslessReduction(
        keep=tuple(sorted(keep_set)),
        remove=tuple(sorted(remove)),
        solver_metadata=metadata,
    )


def _greedy_costs(tests, per_test_cost_ms: Mapping[str, float],
                  imputed_cost_ms: Optional[Mapping[str, float]],
                  ) -> Tuple[Dict[str, float], Dict[str, Any]]:
    """The costs :func:`lossless_cover` weighs, each validated
    (:func:`_cost_ms`), and its metadata recording them."""
    imputed = dict(imputed_cost_ms or {})
    costs = {**per_test_cost_ms, **imputed}
    raw_cost = {t: float(_cost_ms(t, costs[t])) for t in tests if t in costs}
    return raw_cost, {
        "solver": "greedy",
        "raw_cost_ms": {t: c for t, c in raw_cost.items() if t not in imputed},
        **_imputed_costs(tests, raw_cost, imputed),
    }


def _imputed_costs(tests, raw_cost: Mapping[str, float],
                   imputed: Mapping[str, float]) -> Dict[str, Any]:
    """The imputed part of a reduction's cost record: ``imputed_cost_ms``
    and ``cost_basis``, ``equal`` when every test's cost is imputed. Empty
    without imputed costs."""
    if not imputed:
        return {}
    return {
        "imputed_cost_ms": {t: raw_cost[t] for t in tests if t in imputed},
        "cost_basis": "equal" if len(imputed) == len(tests) else "imputed",
    }


#: How the exact reduction chooses among covers that are equally good in
#: time and test count. ``name_order`` counts exact duplicate tests once,
#: prefers tests earlier in name order, with a proof that the choice is
#: unique or a proven list of every equally good cover.
#: ``solver`` keeps whichever optimal cover the solver returns, which can
#: differ between CBC builds. The setting ``reduction_tie_break`` offers
#: these.
TIE_BREAKS = _CHOICES["reduction_tie_break"]


def _collapse_duplicates(
    tests: List[str],
    token_sets: Mapping[str, Set[str]],
    effective_cost: Mapping[str, int],
    forced_ids: FrozenSet[str],
) -> Tuple[List[str], Dict[str, List[str]]]:
    """Exact duplicates, the same tokens at the same cost in microseconds,
    are interchangeable for the solver, and an optimal cover holds at most
    one of them. Each group keeps its first test by name as the candidate,
    or its forced tests when it has any. Returns the candidates, sorted, and
    for each candidate that stands for others the tests it replaces."""
    groups: Dict[Tuple[FrozenSet[str], int], List[str]] = {}
    for t in tests:
        key = (frozenset(token_sets[t]), effective_cost[t])
        groups.setdefault(key, []).append(t)
    candidates: List[str] = []
    duplicates: Dict[str, List[str]] = {}
    for members in groups.values():
        kept = [t for t in members if t in forced_ids] or members[:1]
        candidates.extend(kept)
        dropped = [t for t in members if t not in kept]
        if dropped:
            duplicates[kept[0]] = dropped
    return sorted(candidates), duplicates


def exact_cover(
    per_test_tokens: Mapping[str, FrozenSet[str]],
    per_test_cost_ms: Mapping[str, float],
    *,
    forced_ids: FrozenSet[str] = frozenset(),
    imputed_cost_ms: Optional[Mapping[str, float]] = None,
    time_limit_s: float = 120.0,
    minimum_test_runtime_ms: float = 0.001,
    random_seed: int = 1,
    tie_break: str = "name_order",
) -> LosslessReduction:
    """Solve the exact set-cover program: least time first, then fewest tests.

    With ``tie_break="name_order"`` two more tiers
    choose among the covers equal in time and test count, over the tests
    left after exact duplicates are counted once: the least sum of the tests'
    ranks in name order, then the least sum of their squares. One more solve
    then asks for another cover equal in all four. Only a proof that none
    exists makes the choice unique. If one exists, or the solver ends without
    a proof, every equally good cover is listed, each listing step proven,
    and the first in name order is chosen. ``tie_break="solver"`` keeps the
    two tiers and the solver's choice.

    CBC runs its serial search: no ``-threads`` option is passed. With
    ``-threads 1`` CBC 2.10.3 starts one worker thread and signals it without
    holding its mutex before joining it, so a signal that arrives before the
    worker waits is lost and the solve stalls for the worker's fixed 10 s
    timeout. The serial search has no worker and is the mode CBC documents as
    repeatable.

    ``imputed_cost_ms`` holds the costs that are not a test's own measured
    time (:func:`reduction_costs`): those of the tests without one, and the
    first test's, whose measured time also holds the run's setup. They are
    weighed like measured ones, but the metadata keeps them apart:
    ``raw_cost_ms`` holds the measured times only, ``imputed_cost_ms`` the
    imputed costs, and ``cost_basis`` says ``imputed`` or, when no test has
    a time, ``equal``.

    The bundled CBC's version is probed before the cover time starts, so
    a process's first cover time holds the solve alone."""
    return _exact_cover(
        per_test_tokens, per_test_cost_ms, _bundled_cbc(),
        forced_ids=forced_ids, imputed_cost_ms=imputed_cost_ms,
        time_limit_s=time_limit_s,
        minimum_test_runtime_ms=minimum_test_runtime_ms,
        random_seed=random_seed, tie_break=tie_break,
    )


def _bundled_cbc() -> Optional[str]:
    """The version of the CBC that PuLP runs. A process's first call
    probes it (:func:`cbc_version`)."""
    import pulp

    # PuLP is pinned to 3.3.2. Its constructor-level migration-warning flag
    # retains the bundled solver without changing process-wide warning filters.
    return cbc_version(
        str(pulp.PULP_CBC_CMD(msg=False, _skip_v4_deprecation=True).path))[0]


@_timed_cover
def _exact_cover(
    per_test_tokens: Mapping[str, FrozenSet[str]],
    per_test_cost_ms: Mapping[str, float],
    cbc: Optional[str],
    *,
    forced_ids: FrozenSet[str],
    imputed_cost_ms: Optional[Mapping[str, float]],
    time_limit_s: float,
    minimum_test_runtime_ms: float,
    random_seed: int,
    tie_break: str,
) -> LosslessReduction:
    """:func:`exact_cover`'s solve, with ``cbc``, the version of the CBC
    that PuLP runs."""
    import pulp

    if tie_break not in TIE_BREAKS:
        raise ValueError(
            f"unknown reduction_tie_break {tie_break!r}; use "
            + " or ".join(repr(name) for name in TIE_BREAKS)
        )
    tests = sorted(per_test_tokens)
    token_sets: Dict[str, Set[str]] = {
        t: set(per_test_tokens[t]) for t in tests
    }
    universe: Set[str] = set()
    for toks in token_sets.values():
        universe |= toks
    imputed = dict(imputed_cost_ms or {})
    raw_cost, effective_cost = _effective_costs_us(
        tests, {**per_test_cost_ms, **imputed}, minimum_ms=minimum_test_runtime_ms,
    )
    if not math.isfinite(time_limit_s) or time_limit_s <= 0:
        raise ValueError("reduction solver time limit must be positive")
    if random_seed <= 0:
        raise ValueError("the exact reduction contract requires a positive seed")

    if tie_break == "name_order":
        candidates, duplicates = _collapse_duplicates(
            tests, token_sets, effective_cost, forced_ids,
        )
    else:
        candidates, duplicates = tests, {}
    holders: Dict[str, List[str]] = {tok: [] for tok in sorted(universe)}
    for t in candidates:
        for tok in sorted(token_sets[t]):
            holders[tok].append(t)

    started = time.monotonic()
    base_metadata: Dict[str, Any] = {
        "solver": "exact",
        "tie_break": tie_break,
        "time_limit_s": float(time_limit_s),
        "cbc_search": "serial",
        "random_seed": int(random_seed),
        "cbc_options": [
            f"randomSeed {random_seed}",
            f"randomCbcSeed {random_seed}",
        ],
        "minimum_test_runtime_ms": float(minimum_test_runtime_ms),
        "raw_cost_ms": {t: raw_cost[t] for t in tests if t not in imputed},
        "effective_cost_us": {t: effective_cost[t] for t in tests},
        "selected_ids": None,
        "runtime_objective_us": None,
        "test_count_objective": None,
        "tiers": [],
        "pulp_version": str(pulp.__version__),
    }
    base_metadata.update(_imputed_costs(tests, raw_cost, imputed))

    # The records name CBC by its version, never by the machine's path.
    base_metadata["cbc_version"] = cbc

    def _tier_options(tier_name: str) -> List[str]:
        options = list(base_metadata["cbc_options"])
        if tier_name != "runtime":
            # CBC 2.10.3's integer preprocessor can falsely classify a
            # feasible model as infeasible when the proven runtime bound has
            # large integer coefficients. The unprocessed model returns the
            # same feasible optimum and keeps the proof status inspectable.
            options.append("preprocess off")
        return options

    def _solver(remaining_s: float, tier_name: str):
        return pulp.PULP_CBC_CMD(
            msg=False,
            timeLimit=max(0.001, remaining_s),
            options=_tier_options(tier_name),
            _skip_v4_deprecation=True,
        )

    def _failure(failure: str, **extra: Any) -> Dict[str, Any]:
        return {
            **base_metadata,
            "elapsed_s": time.monotonic() - started,
            "failure": failure,
            **extra,
        }

    def _run_tier(prob, name: str) -> Optional[Dict[str, Any]]:
        """Solve one tier within the shared time limit and record it. None when
        the time limit is spent. A CBC that fails to run raises
        ExactCoverError, so the phase end records the failure."""
        remaining = float(time_limit_s) - (time.monotonic() - started)
        if remaining <= 0:
            return None
        tier_started = time.monotonic()
        try:
            prob.solve(_solver(remaining, name))
        except pulp.PulpSolverError as exc:
            # PuLP's text names CBC's path and the working directory, which
            # the records hold nowhere. It stays on the chained exception.
            raise ExactCoverError(
                f"CBC failed in {name} ({type(exc).__name__})",
                _failure("solver_error", failed_tier=name),
            ) from exc
        tier = {
            "name": name,
            "problem_status_code": int(prob.status),
            "problem_status": pulp.LpStatus.get(prob.status, str(prob.status)),
            "solution_status_code": int(prob.sol_status),
            "solution_status": pulp.LpSolution.get(
                prob.sol_status, str(prob.sol_status)
            ),
            "time_limit_s": remaining,
            "elapsed_s": time.monotonic() - tier_started,
            "cbc_options": _tier_options(name),
        }
        base_metadata["tiers"].append(tier)
        return tier

    def _solve_tier(prob, name: str) -> Dict[str, Any]:
        tier = _run_tier(prob, name)
        if tier is None:
            raise ExactCoverError(
                f"exact cover exhausted its total time limit before {name}",
                _failure("time_limit_exhausted", failed_tier=name),
            )
        if (
            prob.status != pulp.LpStatusOptimal
            or prob.sol_status != pulp.LpSolutionOptimal
        ):
            raise ExactCoverError(
                "exact cover did not establish both optimal statuses for "
                f"{name}: problem={tier['problem_status']}, "
                f"solution={tier['solution_status']}",
                _failure("optimality_not_established", failed_tier=name),
            )
        return tier

    def _new_model(name: str):
        prob = pulp.LpProblem(name, pulp.LpMinimize)
        variables = {
            t: pulp.LpVariable(
                f"x_{i}", cat="Binary", _skip_v4_deprecation=True,
            )
            for i, t in enumerate(candidates)
        }
        for token in sorted(holders):
            prob += pulp.lpSum(
                variables[t] for t in holders[token]
            ) >= 1
        for t in sorted(forced_ids):
            if t in variables:
                prob += variables[t] == 1
        return prob, variables

    def _selected(variables) -> Set[str]:
        return {
            t for t in candidates
            if variables[t].value() is not None
            and variables[t].value() > 0.5
        }

    def _covers(selection: Set[str]) -> bool:
        covered: Set[str] = set()
        for t in selection:
            covered |= token_sets[t]
        return covered == universe and {
            t for t in forced_ids if t in token_sets
        } <= selection

    runtime_prob, runtime_x = _new_model("min_runtime_set_cover")
    runtime_objective = pulp.lpSum(
        effective_cost[t] * runtime_x[t] for t in candidates
    )
    runtime_prob += runtime_objective
    _solve_tier(runtime_prob, "runtime")
    runtime_keep = _selected(runtime_x)
    runtime_tokens = set().union(*(
        token_sets[t] for t in runtime_keep
    )) if runtime_keep else set()
    if runtime_tokens != universe:
        raise ExactCoverError(
            "time tier returned selected ids that do not cover the "
            "universe",
            _failure("runtime_selected_ids_do_not_cover_universe",
                     selected_ids=sorted(runtime_keep)),
        )
    # Replay the integer objective from the returned binary assignment.
    # Reading the affine expression through floating solver values can
    # round to a neighboring integer and make the exact second-tier
    # equality spuriously infeasible.
    runtime_optimum = sum(effective_cost[t] for t in runtime_keep)
    base_metadata["runtime_objective_us"] = runtime_optimum

    # Build the second tier as a fresh model. Replacing a solved PuLP
    # model's objective in place can leave CBC with stale presolve state
    # and has produced a false infeasible verdict on a feasible optimum.
    count_prob, x = _new_model("min_count_runtime_optimal_set_cover")
    count_runtime = pulp.lpSum(
        effective_cost[t] * x[t] for t in candidates
    )
    # The first tier proves that no feasible cover is cheaper, so this
    # upper bound is mathematically equivalent to equality. CBC is more
    # numerically reliable with the inequality when integer microsecond
    # coefficients are large.
    count_prob += count_runtime <= runtime_optimum
    test_count_objective = pulp.lpSum(x[t] for t in candidates)
    count_prob.setObjective(test_count_objective)
    _solve_tier(count_prob, "test_count")
    test_count_optimum = int(round(float(
        pulp.value(test_count_objective) or 0
    )))
    base_metadata["test_count_objective"] = test_count_optimum
    keep_set = _selected(x)

    if tie_break == "name_order":
        keep_set = _break_ties_by_name(
            keep_set, candidates=candidates, duplicates=duplicates,
            effective_cost=effective_cost, runtime_optimum=runtime_optimum,
            test_count_optimum=test_count_optimum,
            new_model=_new_model, selected=_selected, covers=_covers,
            solve_tier=_solve_tier, run_tier=_run_tier, failure=_failure,
            metadata=base_metadata,
        )

    remove = [t for t in tests if t not in keep_set]
    tokens_after: Set[str] = set()
    for t in keep_set:
        tokens_after |= token_sets[t]
    if tokens_after != universe:
        raise ExactCoverError(
            "exact cover returned selected ids that do not cover the universe",
            _failure("selected_ids_do_not_cover_universe",
                     selected_ids=sorted(keep_set)),
        )
    replay_runtime_us = sum(effective_cost[t] for t in keep_set)
    if replay_runtime_us != base_metadata["runtime_objective_us"]:
        raise ExactCoverError(
            "exact cover time objective does not replay from selected ids",
            _failure("objective_replay_mismatch",
                     selected_ids=sorted(keep_set),
                     replayed_runtime_objective_us=replay_runtime_us),
        )
    if len(keep_set) != base_metadata["test_count_objective"]:
        raise ExactCoverError(
            "exact cover test-count objective does not replay from selected ids",
            _failure("test_count_replay_mismatch",
                     selected_ids=sorted(keep_set)),
        )
    base_metadata["selected_ids"] = sorted(keep_set)
    base_metadata["elapsed_s"] = time.monotonic() - started
    return LosslessReduction(
        keep=tuple(sorted(keep_set)),
        remove=tuple(sorted(remove)),
        solver="exact",
        effective_cost_us=tuple((t, effective_cost[t]) for t in tests),
        runtime_objective_us=int(base_metadata["runtime_objective_us"]),
        test_count_objective=int(base_metadata["test_count_objective"]),
        solver_metadata=base_metadata,
    )


def _break_ties_by_name(
    keep_set: Set[str],
    *,
    candidates: List[str],
    duplicates: Dict[str, List[str]],
    effective_cost: Mapping[str, int],
    runtime_optimum: int,
    test_count_optimum: int,
    new_model: Callable[[str], Any],
    selected: Callable[[Any], Set[str]],
    covers: Callable[[Set[str]], bool],
    solve_tier: Callable[[Any, str], Dict[str, Any]],
    run_tier: Callable[[Any, str], Optional[Dict[str, Any]]],
    failure: Callable[..., Dict[str, Any]],
    metadata: Dict[str, Any],
) -> Set[str]:
    """The name-order tiers after time and test count: least
    sum of name ranks, least sum of their squares, then a proof that no
    other cover is equal in all four, or the first of all such covers in
    name order. Every selection is replayed in integers before it counts."""
    import pulp

    rank = {t: i + 1 for i, t in enumerate(candidates)}

    def bounded(name: str, bounds: Tuple[int, ...]):
        prob, v = new_model(name)
        prob += pulp.lpSum(
            effective_cost[t] * v[t] for t in candidates) <= runtime_optimum
        prob += pulp.lpSum(v[t] for t in candidates) <= test_count_optimum
        if len(bounds) > 0:
            prob += pulp.lpSum(rank[t] * v[t] for t in candidates) <= bounds[0]
        if len(bounds) > 1:
            prob += pulp.lpSum(
                rank[t] ** 2 * v[t] for t in candidates) <= bounds[1]
        return prob, v

    def objectives(selection: Set[str]) -> Tuple[int, int, int, int]:
        return (
            sum(effective_cost[t] for t in selection),
            len(selection),
            sum(rank[t] for t in selection),
            sum(rank[t] ** 2 for t in selection),
        )

    def replayed(selection: Set[str], name: str, expected: Tuple[int, ...]):
        if not covers(selection) or objectives(selection)[:len(expected)] != expected:
            raise ExactCoverError(
                f"exact cover {name} tier does not replay from selected ids",
                failure("tie_break_replay_mismatch", failed_tier=name,
                        selected_ids=sorted(selection)),
            )
        return selection

    replayed(keep_set, "test_count", (runtime_optimum, test_count_optimum))
    prob, v = bounded("min_rank_sum_optimal_set_cover", ())
    prob.setObjective(pulp.lpSum(rank[t] * v[t] for t in candidates))
    solve_tier(prob, "rank_sum")
    first = replayed(selected(v), "rank_sum",
                     (runtime_optimum, test_count_optimum))
    rank_sum = objectives(first)[2]
    prob, v = bounded("min_rank_square_sum_optimal_set_cover", (rank_sum,))
    prob.setObjective(pulp.lpSum(rank[t] ** 2 * v[t] for t in candidates))
    solve_tier(prob, "rank_square_sum")
    chosen = replayed(selected(v), "rank_square_sum",
                      (runtime_optimum, test_count_optimum, rank_sum))
    optimum = objectives(chosen)
    metadata["duplicate_tests"] = duplicates
    metadata["rank_sum_objective"] = optimum[2]
    metadata["rank_square_sum_objective"] = optimum[3]

    def another(excluded: List[Set[str]], name: str):
        """Ask for a cover equal in all four objectives that differs from
        every excluded selection: ("none", None) with a proof, ("found", s),
        ("unproven", None), or ("exhausted", None) when the time limit was
        spent before the tier could run."""
        prob, v = bounded(name, optimum[2:])
        for selection in excluded:
            prob += pulp.lpSum(v[t] for t in sorted(selection)) <= (
                len(selection) - 1)
        prob.setObjective(pulp.lpSum(v[t] for t in candidates))
        tier = run_tier(prob, name)
        if tier is None:
            return "exhausted", None
        if prob.status == pulp.LpStatusInfeasible:
            return "none", None
        if prob.sol_status in (
            pulp.LpSolutionOptimal, pulp.LpSolutionIntegerFeasible,
        ):
            return "found", selected(v)
        return "unproven", None

    tier_name = "uniqueness"
    verdict, other = another([chosen], tier_name)
    if verdict == "none":
        metadata["uniqueness"] = "proven"
        metadata["equally_good_selections"] = 1
        return chosen

    # Not proven unique: list every equally good cover, each step proven,
    # and take the first in name order. A returned assignment that does not
    # replay all four objectives is excluded without counting as one.
    equal: List[Set[str]] = [chosen]
    excluded: List[Set[str]] = [chosen]
    step = 0
    while True:
        if verdict == "exhausted":
            raise ExactCoverError(
                f"exact cover exhausted its total time limit before {tier_name}",
                failure("time_limit_exhausted", failed_tier=tier_name,
                        equal_selections=[sorted(s) for s in equal]),
            )
        if verdict == "found":
            excluded.append(other)
            if covers(other) and objectives(other) == optimum:
                equal.append(other)
        elif verdict == "unproven" and step > 0:
            raise ExactCoverError(
                "exact cover could not prove which of the equally good "
                "covers comes first in name order",
                failure("tie_not_resolved",
                        failed_tier=tier_name,
                        equal_selections=[sorted(s) for s in equal]),
            )
        step += 1
        tier_name = f"name_order_{step}"
        verdict, other = another(excluded, tier_name)
        if verdict == "none":
            break
    metadata["uniqueness"] = "resolved_by_name_order"
    metadata["equally_good_selections"] = len(equal)
    return min(equal, key=lambda s: sorted(rank[t] for t in s))


def solve_cover(
    per_test_tokens: Mapping[str, FrozenSet[str]],
    per_test_cost_ms: Mapping[str, float],
    *,
    forced_ids: FrozenSet[str] = frozenset(),
    imputed_cost_ms: Optional[Mapping[str, float]] = None,
    solver: str = "exact",
    time_limit_s: float = 120.0,
    minimum_test_runtime_ms: float = 0.001,
    random_seed: int = 1,
    tie_break: str = "name_order",
) -> LosslessReduction:
    """Dispatch to the explicitly configured cover solver. A cover source
    installed with :func:`installed` (the demonstration's scripted solver)
    answers the exact reduction in place of CBC. A normal run installs none.
    ``imputed_cost_ms`` are the costs that are not a test's own measured
    time (:func:`reduction_costs`)."""
    if solver == "exact":
        source = current()
        if source is not None:
            with phase_state.timed_part(phase_state.TIME_COVER):
                return source.solve_exact(
                    per_test_tokens, per_test_cost_ms,
                    forced_ids=forced_ids, time_limit_s=time_limit_s,
                    minimum_test_runtime_ms=minimum_test_runtime_ms,
                    random_seed=random_seed,
                    tie_break=tie_break,
                    imputed_cost_ms=imputed_cost_ms,
                )
        return exact_cover(
            per_test_tokens, per_test_cost_ms,
            forced_ids=forced_ids, imputed_cost_ms=imputed_cost_ms,
            time_limit_s=time_limit_s,
            minimum_test_runtime_ms=minimum_test_runtime_ms,
            random_seed=random_seed,
            tie_break=tie_break,
        )
    if solver == "greedy":
        return lossless_cover(
            per_test_tokens, per_test_cost_ms,
            forced_ids=forced_ids, imputed_cost_ms=imputed_cost_ms,
        )
    raise ValueError(
        f"unknown reduction_solver {solver!r}; use "
        + " or ".join(repr(name) for name in _CHOICES["reduction_solver"])
    )
