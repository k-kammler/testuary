"""Mutation phase helpers.

Module functions such as ``_canonical_mutation_summary_line`` are called
directly. Stateful helpers are ``MutationAgent`` methods, called on an
instance without a model.
"""
from __future__ import annotations

from pathlib import Path


from agent_fixtures import mutation_agent

from testuary.agent import _helpers as agent_helpers
from testuary.agent import mutation_agent as tm
from phase_values import set_phase_value
import pytest

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


# target_path: the test-file path builder.

def test_target_path_joins_dir_and_filename(tmp_path: Path) -> None:
    """``target_path`` names the test file the mutation phase writes, the
    same under either path separator."""
    agent = mutation_agent(tmp_path, mutation_test_file_name="test_foo.py")
    assert agent.target_path() == tmp_path / "test_foo.py"


# The mutation phase's file is the delivered suite, named by ``target_test_path``.

def test_mutation_filename_python_via_backend(tmp_path) -> None:
    """The mutation phase edits the delivered suite the backend names."""
    from testuary.languages import get_backend
    p = get_backend("python").target_test_path(
        str(tmp_path), "/abs/proj/encoder.py",
    )
    assert p.name == "test_encoder.py"


# after_write: cache invalidation.

def test_after_write_invalidates_both_caches(tmp_path: Path) -> None:
    """Both ``tests_changed_since_mutation`` and
    ``tests_changed_since_run_tests`` become True, so the next measurement
    runs instead of returning cached output."""
    agent = mutation_agent(tmp_path)
    agent.session.tests_changed_since_mutation = False
    agent.session.tests_changed_since_run_tests = False
    agent.after_write()
    assert agent.session.tests_changed_since_mutation is True
    assert agent.session.tests_changed_since_run_tests is True


# A new run is a new instance with clean state.

def test_fresh_session_starts_with_cleared_counters() -> None:
    """Each run builds a fresh ``MutationAgent`` with a fresh
    ``TestMutationSession``, so no counter carries one target's
    measurements into the next."""
    s = tm.TestMutationSession()
    assert s.best_mutation_score is None
    assert s.prev_mutation_score is None
    assert s.measurements_without_improvement == 0
    assert s.mutation_post_goal_count == 0
    assert s.seen_mutation_goal is False
    assert s.mutation_measurements == 0
    assert s.writes_done == 0
    assert s.run_tests_calls == 0
    assert s.session_tool_calls == 0
    # Cache flags start True so the first measurement runs.
    assert s.tests_changed_since_mutation is True
    assert s.tests_changed_since_run_tests is True
    assert s.last_mutation_result == ""
    assert s.last_run_tests_result == ""
    assert s.last_tests_passed is False


# _synthesize_mutation_result: the MutationTestResult of an early stop.

def test_synthesize_mutation_result_goal_met_status_is_success(tmp_path: Path) -> None:
    """A stop reason ending in ``_goal_met`` gives status success, any other
    reason stagnation. The status fills the CSV's status column."""
    agent = mutation_agent(tmp_path, mutation_test_file_name="test_foo_mutation.py")
    agent.session.best_mutation_score = 87.5
    agent.session.mutation_measurements = 3
    agent.session.writes_done = 4
    r = agent._synthesize_mutation_result("mutation_goal_met")
    assert r.status == "success"
    assert r.mutation_score == 87.5
    assert r.mutation_test_file_name == "test_foo_mutation.py"
    assert "87.5%" in r.explanation


def test_synthesize_mutation_result_stagnation_status(tmp_path: Path) -> None:
    """A 'stagnation' stop gives status stagnation, so the CSV tells it apart
    from a met goal."""
    agent = mutation_agent(tmp_path)
    agent.session.best_mutation_score = 42.0
    r = agent._synthesize_mutation_result("stagnation")
    assert r.status == "stagnation"


def test_synthesize_mutation_result_unmeasured_omits_score_line(tmp_path: Path) -> None:
    """Without a measured mutation score (best=None) the explanation omits
    the score line instead of printing 'None%'."""
    agent = mutation_agent(tmp_path)
    r = agent._synthesize_mutation_result("stagnation")
    assert r.mutation_score is None
    assert "None" not in r.explanation


# The MUTATION: summary line, shaped like coverage's TESTS: line.

def test_canonical_mutation_summary_line_success() -> None:
    """With killed and survived mutants and no truncation the line has the
    shape of coverage's ``TESTS: ...`` line."""
    from testuary.languages.mutation import MutationMeasurement
    res = MutationMeasurement(
        score=67.0, killed=2020, survived=993,
        timeout=0, non_viable=0, untested=0, total=3013,
        truncated=False,
    )
    line = tm._canonical_mutation_summary_line(res, elapsed_s=363.2)
    assert line == (
        "MUTATION: SUCCESS | mutants 3013 | "
        "killed 2020 | survived 993 | timeout 0 | "
        "suspicious 0 | untested 0 | "
        "score 67.0% | elapsed 363.2s"
    )


def test_canonical_mutation_summary_line_truncated() -> None:
    """When wall_clock_timeout ends the run with partial results, the status
    is TRUNCATED, marking the numbers as not final."""
    from testuary.languages.mutation import MutationMeasurement
    res = MutationMeasurement(
        score=50.0, killed=100, survived=100,
        timeout=10, non_viable=5, untested=2000, total=200,
        truncated=True,
    )
    line = tm._canonical_mutation_summary_line(res, elapsed_s=3600.0)
    assert line.startswith("MUTATION: TRUNCATED | ")
    assert "untested 2000" in line


def test_canonical_mutation_summary_line_inconclusive() -> None:
    """No killed or survived mutant among attempted ones (every mutant was
    non-viable, untested or timed out) gives INCONCLUSIVE."""
    from testuary.languages.mutation import MutationMeasurement
    res = MutationMeasurement(
        score=0.0, killed=0, survived=0,
        timeout=10, non_viable=500, untested=200, total=0,
        truncated=False,
    )
    line = tm._canonical_mutation_summary_line(res, elapsed_s=1800.0)
    assert line.startswith("MUTATION: INCONCLUSIVE | ")


def test_canonical_mutation_summary_line_no_mutants() -> None:
    """An engine run with no mutants (empty catalogue, or initialization
    failed before any work item) has a status of its own."""
    from testuary.languages.mutation import MutationMeasurement
    res = MutationMeasurement(
        score=0.0, killed=0, survived=0,
        timeout=0, non_viable=0, untested=0, total=0,
        truncated=False,
    )
    line = tm._canonical_mutation_summary_line(res, elapsed_s=12.3)
    assert line.startswith("MUTATION: NO_MUTANTS | ")


# The mutation phase's terminal _settle_best_green keeps a red or unverified
# last edit out of the merged suite, since the merge has no green gate.


def _seed_best_green(agent, content: str) -> None:
    agent.session.best_green = agent_helpers.BestGreenState(snapshot={
        "content": content,
        "mutation": 80.0,
        "coverage": 90.0,
        "test_count": 1,
        "time": 1.0,
    })


def test_settle_best_green_reverts_unverified_end(tmp_path: Path) -> None:
    """A red or unverified mutation file at phase exit is reverted to the
    best-green snapshot, so the merge takes the green file, not the broken
    last edit."""
    mut_name = "test_mod_mutation.py"
    agent = mutation_agent(tmp_path, mutation_test_file_name=mut_name)
    mut_file = tmp_path / mut_name
    mut_file.write_text("def test_broken(): assert False\n", encoding="utf-8")
    best = "def test_good(): assert True\n"
    _seed_best_green(agent, best)
    agent.session.last_tests_passed = False
    agent.session.tests_changed_since_run_tests = True

    agent._settle_best_green(agent.target_path(), log_fn=agent.log)

    assert mut_file.read_text(encoding="utf-8") == best
    assert agent.session.best_green.restored_from_green is True


def test_settle_best_green_keeps_verified_green_end(tmp_path: Path) -> None:
    """A green current file whose fresh measurement outranks the snapshot is
    kept, never replaced by the older snapshot."""
    mut_name = "test_mod_mutation.py"
    agent = mutation_agent(tmp_path, mutation_test_file_name=mut_name)
    mut_file = tmp_path / mut_name
    live = "def test_live(): assert True\n"
    mut_file.write_text(live, encoding="utf-8")
    _seed_best_green(agent, "def test_other(): assert True\n")
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_run_tests = False
    agent.session.tests_changed_since_mutation = False
    agent.session.prev_mutation_score = 85.0  # fresh, above the snapshot's 80
    import hashlib as _hashlib
    agent.session.measured_suite_hash = _hashlib.sha256(
        live.encode("utf-8")
    ).hexdigest()

    agent._settle_best_green(agent.target_path(), log_fn=agent.log)

    assert mut_file.read_text(encoding="utf-8") == live


def test_settle_best_green_no_snapshot_is_noop(tmp_path: Path) -> None:
    """Without a green snapshot there is nothing to restore."""
    mut_name = "test_mod_mutation.py"
    agent = mutation_agent(tmp_path, mutation_test_file_name=mut_name)
    mut_file = tmp_path / mut_name
    red = "def test_broken(): assert False\n"
    mut_file.write_text(red, encoding="utf-8")
    agent.session.best_green = agent_helpers.BestGreenState(snapshot=None)
    agent.session.last_tests_passed = False
    agent.session.tests_changed_since_run_tests = True

    agent._settle_best_green(agent.target_path(), log_fn=agent.log)

    assert mut_file.read_text(encoding="utf-8") == red


def test_a_failed_terminal_restore_is_logged_as_a_restore(
        tmp_path: Path, monkeypatch) -> None:
    """The mutation phase logs a terminal restore that raises under the tag
    of every best-green restore line, ``[restore]``, and keeps its file."""
    mut_name = "test_mod_mutation.py"
    agent = mutation_agent(tmp_path, mutation_test_file_name=mut_name)
    mut_file = tmp_path / mut_name
    red = "def test_broken(): assert False\n"
    mut_file.write_text(red, encoding="utf-8")
    _seed_best_green(agent, "def test_good(): assert True\n")
    agent.session.last_tests_passed = False
    agent.session.tests_changed_since_run_tests = True

    def failing_restore(*_args, **_kwargs):
        raise OSError("disk full")

    monkeypatch.setattr(agent_helpers, "restore_best_green", failing_restore)
    lines = []
    monkeypatch.setattr(agent, "log", lambda message, *args, **fields: lines.append(message))

    agent._settle_best_green(agent.target_path(), log_fn=agent.log)

    assert [line for line in lines if "restore failed" in line] == [
        "[restore] best-green restore failed silently: OSError: disk full"]
    assert mut_file.read_text(encoding="utf-8") == red


# _maybe_revert_on_regression reverts to best-green after N green measurements
# strictly worse than the best. An equal score is tolerated.


def _seed_best_green_for_regress(agent, best: str) -> Path:
    agent.session.best_green = agent_helpers.BestGreenState(snapshot={
        "content": best, "mutation": 80.0, "coverage": 90.0,
        "test_count": 2, "time": 1.0,
    })
    agent.session.mutation_regression_count = 0
    return agent.target_path()


def test_regression_reverts_at_n_restore_1(tmp_path: Path, monkeypatch) -> None:
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "restore_count", 1)
    agent = mutation_agent(tmp_path, mutation_test_file_name="test_mod_mutation.py")
    best = "def test_best(): assert True\n"
    mut = _seed_best_green_for_regress(agent, best)
    mut.write_text("def test_regressed(): assert True  # fewer kills\n", encoding="utf-8")

    reverted = agent._maybe_revert_on_regression(score=70.0, prev_best=80.0)

    assert reverted is not None
    assert mut.read_text(encoding="utf-8") == best
    assert agent.session.mutation_regression_count == 0


def test_lateral_equal_score_is_tolerated(tmp_path: Path, monkeypatch) -> None:
    """An equal score, such as after a setup test, is not a regression. It is
    never reverted and resets the counter."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "restore_count", 1)
    agent = mutation_agent(tmp_path, mutation_test_file_name="test_mod_mutation.py")
    mut = _seed_best_green_for_regress(agent, "def test_best(): assert True\n")
    agent.session.mutation_regression_count = 3
    live = "def test_setup(): assert True\n"
    mut.write_text(live, encoding="utf-8")

    reverted = agent._maybe_revert_on_regression(score=80.0, prev_best=80.0)

    assert reverted is None
    assert mut.read_text(encoding="utf-8") == live
    assert agent.session.mutation_regression_count == 0


def test_improvement_resets_counter_no_revert(tmp_path: Path, monkeypatch) -> None:
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "restore_count", 1)
    agent = mutation_agent(tmp_path, mutation_test_file_name="test_mod_mutation.py")
    mut = _seed_best_green_for_regress(agent, "def test_best(): assert True\n")
    agent.session.mutation_regression_count = 2
    live = "def test_better(): assert True\n"
    mut.write_text(live, encoding="utf-8")

    reverted = agent._maybe_revert_on_regression(score=85.0, prev_best=80.0)

    assert reverted is None
    assert mut.read_text(encoding="utf-8") == live
    assert agent.session.mutation_regression_count == 0


def test_n_restore_disabled_never_reverts(tmp_path: Path, monkeypatch) -> None:
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "restore_count", -1)
    agent = mutation_agent(tmp_path, mutation_test_file_name="test_mod_mutation.py")
    mut = _seed_best_green_for_regress(agent, "BEST")
    live = "def test_regressed(): assert True\n"
    mut.write_text(live, encoding="utf-8")

    reverted = agent._maybe_revert_on_regression(score=70.0, prev_best=80.0)

    assert reverted is None
    assert mut.read_text(encoding="utf-8") == live


def test_n_restore_2_reverts_on_second_regression(tmp_path: Path, monkeypatch) -> None:
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "restore_count", 2)
    agent = mutation_agent(tmp_path, mutation_test_file_name="test_mod_mutation.py")
    best = "def test_best(): assert True\n"
    mut = _seed_best_green_for_regress(agent, best)

    mut.write_text("first regression\n", encoding="utf-8")
    assert agent._maybe_revert_on_regression(70.0, 80.0) is None
    assert mut.read_text(encoding="utf-8") == "first regression\n"

    mut.write_text("second regression\n", encoding="utf-8")
    assert agent._maybe_revert_on_regression(70.0, 80.0) is not None
    assert mut.read_text(encoding="utf-8") == best


# build_state_marker: the task anchor and live mutation score, kept by keep-N.

def test_state_marker_reports_no_score_before_first_measurement(tmp_path: Path) -> None:
    """Before any measure_mutation the marker reports no score."""
    agent = mutation_agent(tmp_path)
    assert "mutation=none" in agent.build_state_marker(None)


def test_state_marker_carries_best_mutation_score(tmp_path: Path) -> None:
    """The marker carries the best mutation score so keep-N never drops it.
    The score is floored, never rounded up, so the marker cannot overstate it."""
    agent = mutation_agent(tmp_path)
    agent.session.best_mutation_score = 60.0
    assert "mutation=60.0%" in agent.build_state_marker(None)


def test_state_marker_carries_stall_count(tmp_path: Path) -> None:
    """A non-zero no-improvement count surfaces as a stall hint."""
    agent = mutation_agent(tmp_path)
    agent.session.best_mutation_score = 60.0
    agent.session.measurements_without_improvement = 3
    assert "(stalled 3 measurements)" in agent.build_state_marker(None)
