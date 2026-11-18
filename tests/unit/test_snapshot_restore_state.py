"""A restored best-green snapshot restores the loop state its bytes left.

When a snapshot's bytes were measured, every restore (the timeout revert,
the revert after failed runs, the regression revert, the terminal restore)
puts back the measured flag, the values, the measured hash, the cached result
and the uncovered regions or survivors. The loop is then in Revise and does
not measure that suite again. An unmeasured snapshot stays unmeasured. Both
loops follow the same rule.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_fixtures import SOURCE, mutation_agent, project_agent

from offered_tools import offered_tools
from testuary.agent import _helpers as ah
from testuary import languages as _languages
from testuary.languages.generic import GenericBackend
from testuary.agent import coverage_agent as tc
from testuary.agent import mutation_agent as tm
from phase_values import set_phase_value

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

SNAPSHOT = "def test_a():\n    assert True\n"
SLOW = SNAPSHOT + "\n\ndef test_slow():\n    while True:\n        pass\n"
LATER = SNAPSHOT + "\n\ndef test_b():\n    assert 1\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


# Coverage


def _coverage_agent(tmp_path: Path) -> "tc.CoverageAgent":
    agent = project_agent(tmp_path, source="def f(x):\n    return x\n", test_source=SNAPSHOT)
    agent.session.has_read = True
    agent.session.initial_measured = True  # past the start's measurement
    return agent


def _target_test(agent) -> Path:
    return Path(agent.deps.test_output_dir) / "test_foo.py"


def _measure_snapshot(agent) -> None:
    """The state a completed measurement of SNAPSHOT leaves, captured into
    best-green the way measure_coverage captures it."""
    s = agent.session
    s.writes_done = 1
    s.last_tests_passed = True
    s.tests_changed_since_run_tests = False
    s.tests_changed_since_measure["coverage"] = False
    s.prev_coverage_pct = 50.0
    s.prev_branch_coverage_pct = 25.0
    s.best_coverage_pct = 50.0
    s.best_branch_coverage_pct = 25.0
    s.measured_suite_hash = _sha(SNAPSHOT)
    s.last_coverage_result = "--- Coverage Summary ---\nLine coverage: 50.0% (snapshot)"
    s.last_uncovered_ranges = "2"
    s.last_suite_exec_ms = 10.0
    agent._maybe_update_best_green(agent.deps.test_output_dir, agent.deps.target_file_path)
    assert (s.best_green.snapshot["loop_state"] or {}).get("measured") is True


def _later_measurement(agent) -> None:
    """A later suite, measured: the live state belongs to LATER."""
    s = agent.session
    _target_test(agent).write_text(LATER, encoding="utf-8")
    s.measured_suite_hash = _sha(LATER)
    s.prev_coverage_pct = 40.0
    s.prev_branch_coverage_pct = 20.0
    s.last_coverage_result = "--- Coverage Summary ---\nLine coverage: 40.0% (later)"
    s.last_uncovered_ranges = "2, 9"


def _assert_coverage_in_snapshot_state(agent) -> None:
    s = agent.session
    assert _target_test(agent).read_text(encoding="utf-8") == SNAPSHOT
    assert s.last_tests_passed is True
    assert s.tests_changed_since_run_tests is False
    assert s.tests_changed_since_measure["coverage"] is False
    assert s.measured_suite_hash == _sha(SNAPSHOT)
    assert (s.prev_coverage_pct, s.prev_branch_coverage_pct) == (50.0, 25.0)
    assert s.last_coverage_result.endswith("(snapshot)")
    assert s.last_uncovered_ranges == "2"
    assert agent.resolve_agent_mode() == "needs_editing"


def _timeout_backend(target_test: Path):
    real = GenericBackend("python", (".py",))
    return SimpleNamespace(
        find_target_test_file=lambda *a, **kw: target_test,
        run_test_files=lambda *a, **kw: _languages.TestRunResult(
            ok=False, returncode=-1, timed_out=True, elapsed_s=None,
            collected=0, passed=0, failed=0, errors=0, output="",
            line_coverage=None, branch_coverage=None),
        skeleton=real.skeleton,
        function_fingerprints=real.function_fingerprints,
    )


def test_coverage_timeout_revert_restores_the_measured_snapshot_state(tmp_path, monkeypatch):
    agent = _coverage_agent(tmp_path)
    _measure_snapshot(agent)
    _later_measurement(agent)
    monkeypatch.setattr(tc.languages, "get_backend",
                        lambda *a, **kw: _timeout_backend(_target_test(agent)))
    _target_test(agent).write_text(SLOW, encoding="utf-8")
    agent.session.tests_changed_since_run_tests = True
    agent.session.tests_changed_since_measure["coverage"] = True
    out = agent.run_tests()
    assert "did not finish" in out
    _assert_coverage_in_snapshot_state(agent)


def test_coverage_restore_after_failed_runs_restores_the_measured_snapshot_state(
        tmp_path, monkeypatch):
    agent = _coverage_agent(tmp_path)
    _measure_snapshot(agent)
    _later_measurement(agent)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_failed_test_attempts", 1)
    monkeypatch.setattr(agent, "_canonical_view", lambda res, w, rc=None: res.output)
    _target_test(agent).write_text("def test_red():\n    assert False\n", encoding="utf-8")
    agent.session.tests_changed_since_measure["coverage"] = True
    out = agent.after_run_tests(SimpleNamespace(output="1 failed", returncode=1), 1, None)
    assert "reverted" in out
    _assert_coverage_in_snapshot_state(agent)


def test_coverage_regression_revert_restores_the_snapshot_measurement(tmp_path, monkeypatch):
    agent = _coverage_agent(tmp_path)
    _measure_snapshot(agent)
    _later_measurement(agent)
    # LATER was measured just now: green, unchanged since its run and measure.
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "restore_count", 1)
    note = agent._maybe_revert_on_regression(
        agent.deps.target_file_path, True, current_goal=20.0, best_goal=25.0)
    assert note and "[REVERTED]" in note
    _assert_coverage_in_snapshot_state(agent)


def test_coverage_restore_of_an_unmeasured_snapshot_stays_unmeasured(tmp_path, monkeypatch):
    agent = _coverage_agent(tmp_path)
    s = agent.session
    s.writes_done = 1
    s.best_green.snapshot = {"content": SNAPSHOT, "coverage": None, "test_count": 1,
                             "loop_state": {"measured": False}}
    _later_measurement(agent)
    s.tests_changed_since_measure["coverage"] = False
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_failed_test_attempts", 1)
    monkeypatch.setattr(agent, "_canonical_view", lambda res, w, rc=None: res.output)
    _target_test(agent).write_text("def test_red():\n    assert False\n", encoding="utf-8")
    agent.after_run_tests(SimpleNamespace(output="1 failed", returncode=1), 1, None)
    assert _target_test(agent).read_text(encoding="utf-8") == SNAPSHOT
    assert s.tests_changed_since_measure["coverage"] is True
    assert s.measured_suite_hash is None
    assert agent.resolve_agent_mode() == "needs_measurement"


def test_coverage_restore_of_an_unmeasured_snapshot_credits_none_of_the_discarded_values(
        tmp_path):
    """The discarded suite was measured. The restored snapshot was not:
    the previous-measurement values (the final coverage a run reports) are
    the snapshot's own, None, not the discarded suite's."""
    agent = _coverage_agent(tmp_path)
    s = agent.session
    s.writes_done = 1
    s.best_green.snapshot = {"content": SNAPSHOT, "coverage": None, "branch_coverage": None,
                             "test_count": 1, "loop_state": {"measured": False}}
    _later_measurement(agent)
    s.last_tests_passed = False  # the live file ended red
    agent._maybe_restore_last_green(agent.deps.test_output_dir, agent.deps.target_file_path)
    assert _target_test(agent).read_text(encoding="utf-8") == SNAPSHOT
    assert (s.prev_coverage_pct, s.prev_branch_coverage_pct) == (None, None)
    metrics = agent.get_run_metrics()
    assert metrics["final_line_coverage"] is None


def test_coverage_terminal_restore_reports_the_snapshot_measurement(tmp_path):
    agent = _coverage_agent(tmp_path)
    _measure_snapshot(agent)
    _later_measurement(agent)
    agent.session.last_tests_passed = False  # the live file ended red
    agent._maybe_restore_last_green(agent.deps.test_output_dir, agent.deps.target_file_path)
    _assert_coverage_in_snapshot_state(agent)
    metrics = agent.get_run_metrics()
    assert (metrics["final_line_coverage"], metrics["final_branch_coverage"]) == (50.0, 25.0)


def test_coverage_prelude_revert_is_not_measured_again(tmp_path, monkeypatch):
    """measure_coverage's auto-prelude restores the snapshot after failed
    runs: the call reports the revert and runs no coverage measurement."""
    agent = _coverage_agent(tmp_path)
    _measure_snapshot(agent)
    _later_measurement(agent)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_failed_test_attempts", 1)
    calls = []
    real = GenericBackend("python", (".py",))

    def run_test_files(*a, measure_coverage_of=None, **kw):
        calls.append(measure_coverage_of)
        return _languages.TestRunResult(
            ok=False, returncode=1, timed_out=False, elapsed_s=0.1,
            collected=1, passed=0, failed=1, errors=0, output="1 failed",
            line_coverage=None, branch_coverage=None)

    fake = SimpleNamespace(
        find_target_test_file=lambda *a, **kw: _target_test(agent),
        run_test_files=run_test_files,
        skeleton=real.skeleton,
        test_metrics_observation=real.test_metrics_observation,
    )
    monkeypatch.setattr(tc.languages, "get_backend", lambda *a, **kw: fake)
    _target_test(agent).write_text("def test_red():\n    assert False\n", encoding="utf-8")
    agent.session.tests_changed_since_run_tests = True
    agent.session.tests_changed_since_measure["coverage"] = True
    out = agent.measure_coverage()
    assert "reverted" in out
    assert calls == [None]  # the prelude run only, no coverage run
    _assert_coverage_in_snapshot_state(agent)


def test_same_bytes_measured_later_complete_an_unmeasured_snapshot(tmp_path):
    """A snapshot taken before its bytes were measured takes the
    measurement of those bytes when it ranks no lower, so a later restore
    knows them measured."""
    path = tmp_path / "test_x.py"
    path.write_text(SNAPSHOT, encoding="utf-8")
    state = ah.BestGreenState()
    ah.update_best_green(state, path, test_count=1, exec_time_ms=5.0,
                         persist_to_disk=False, loop_state={"measured": False})
    took = ah.update_best_green(
        state, path, coverage=0.0, test_count=1, exec_time_ms=9.0,
        persist_to_disk=False, loop_state={"measured": True, "x": 1})
    assert took is False
    assert state.snapshot["coverage"] == 0.0
    assert state.snapshot["loop_state"] == {"measured": True, "x": 1}
    assert state.snapshot["exec_time_ms"] == 5.0


# Mutation


def _mutation_agent(tmp_path: Path) -> "tm.MutationAgent":
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    source = tmp_path / "src" / "foo.py"
    source.parent.mkdir()
    source.write_text(SOURCE, encoding="utf-8")
    agent = mutation_agent(tmp_path, test_output_dir=str(test_dir), mutation_test_file_name="test_foo.py",
                           coverage_baseline_pct=85.0)
    agent.session.has_read = True
    agent.session.initial_measured = True  # past the start's measurement
    return agent


def _mutation_snapshot(agent) -> None:
    s = agent.session
    agent.target_path().write_text(SNAPSHOT, encoding="utf-8")
    s.writes_done = 1
    s.last_tests_passed = True
    s.tests_changed_since_run_tests = False
    s.tests_changed_since_mutation = False
    s.prev_mutation_score = 70.0
    s.best_mutation_score = 70.0
    s.measured_suite_hash = _sha(SNAPSHOT)
    s.last_mutation_result = "Mutation score: 70.0% (snapshot)"
    s.last_surviving_sample = "--- mutant a (line 1) ---"
    s.last_mutation_counts = (7, 3)
    ah.update_best_green(
        s.best_green, agent.target_path(), mutation=70.0, test_count=1,
        persist_to_disk=False, loop_state=agent._snapshot_loop_state(SNAPSHOT))
    assert s.best_green.snapshot["loop_state"]["measured"] is True
    # A later suite, measured lower.
    agent.target_path().write_text(LATER, encoding="utf-8")
    s.prev_mutation_score = 60.0
    s.measured_suite_hash = _sha(LATER)
    s.last_mutation_result = "Mutation score: 60.0% (later)"
    s.last_surviving_sample = "--- mutant b (line 1) ---"
    s.last_mutation_counts = (6, 4)


def _assert_mutation_in_snapshot_state(agent) -> None:
    s = agent.session
    assert agent.target_path().read_text(encoding="utf-8") == SNAPSHOT
    assert s.last_tests_passed is True
    assert s.tests_changed_since_run_tests is False
    assert s.tests_changed_since_mutation is False
    assert s.measured_suite_hash == _sha(SNAPSHOT)
    assert s.prev_mutation_score == 70.0
    assert s.last_mutation_result.endswith("(snapshot)")
    assert s.last_surviving_sample == "--- mutant a (line 1) ---"
    assert s.last_mutation_counts == (7, 3)
    assert agent.resolve_agent_mode() == "needs_editing"


def test_mutation_regression_revert_restores_the_measured_snapshot_state(tmp_path, monkeypatch):
    agent = _mutation_agent(tmp_path)
    _mutation_snapshot(agent)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "restore_count", 1)
    note = agent._maybe_revert_on_regression(60.0, 70.0)
    assert note and "[REVERTED]" in note
    _assert_mutation_in_snapshot_state(agent)


def test_mutation_restore_after_failed_runs_restores_the_measured_snapshot_state(
        tmp_path, monkeypatch):
    agent = _mutation_agent(tmp_path)
    _mutation_snapshot(agent)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "max_failed_test_attempts", 1)
    monkeypatch.setattr(agent, "_canonical_view", lambda res, w, rc=None: res.output)
    agent.target_path().write_text("def test_red():\n    assert False\n", encoding="utf-8")
    agent.session.tests_changed_since_mutation = True
    out = agent.after_run_tests(SimpleNamespace(output="1 failed", returncode=1), 1, None)
    assert "reverted" in out
    _assert_mutation_in_snapshot_state(agent)


def test_mutation_timeout_revert_restores_the_measured_snapshot_state(tmp_path, monkeypatch):
    agent = _mutation_agent(tmp_path)
    _mutation_snapshot(agent)
    agent.target_path().write_text(SLOW, encoding="utf-8")
    agent.session.tests_changed_since_mutation = True
    out = agent.on_timeout(30, "run_tests")
    assert "did not finish" in out
    _assert_mutation_in_snapshot_state(agent)


def test_mutation_terminal_restore_restores_the_snapshot_state(tmp_path):
    agent = _mutation_agent(tmp_path)
    _mutation_snapshot(agent)
    agent.session.last_tests_passed = False  # the live file ended red
    agent._settle_best_green(agent.target_path(), log_fn=agent.log)
    _assert_mutation_in_snapshot_state(agent)


def test_mutation_restore_of_an_unmeasured_snapshot_credits_none_of_the_discarded_score(
        tmp_path):
    agent = _mutation_agent(tmp_path)
    s = agent.session
    s.best_green.snapshot = {"content": SNAPSHOT, "mutation": None, "coverage": 85.0,
                             "test_count": 1, "loop_state": {"measured": False}}
    agent.target_path().write_text(LATER, encoding="utf-8")
    s.prev_mutation_score = 60.0  # the discarded suite's
    s.last_tests_passed = False
    agent._settle_best_green(agent.target_path(), log_fn=agent.log)
    assert agent.target_path().read_text(encoding="utf-8") == SNAPSHOT
    assert s.prev_mutation_score is None
    assert s.tests_changed_since_mutation is True


def test_a_selective_measurement_is_not_restored_onto_the_full_catalogue(tmp_path, monkeypatch):
    agent = _mutation_agent(tmp_path)
    _mutation_snapshot(agent)
    agent.session.mutation_mode = "full"
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "restore_count", 1)
    agent._maybe_revert_on_regression(60.0, 70.0)
    s = agent.session
    assert agent.target_path().read_text(encoding="utf-8") == SNAPSHOT
    assert s.tests_changed_since_mutation is True
    assert s.measured_suite_hash is None
    assert agent.resolve_agent_mode() == "needs_measurement"


# A timeout without a revert: the bytes on disk ran into the timeout. When no
# revert follows (no best-green on record, the file already holds the
# snapshot, or a named hang left in place), the file counts as run. Both loops
# return to Revise and do not offer run_tests again on the unchanged suite.

_REVISE = {"read_source", "read_tests", "edit_file"}
_NO_REVERT_CASES = ("no_snapshot", "file_is_the_snapshot", "named_hang")

def _hanging_backend(target_test: Path, active_test):
    backend = _timeout_backend(target_test)
    backend.run_test_files = lambda *a, **kw: _languages.TestRunResult(
        ok=False, returncode=-1, timed_out=True, elapsed_s=None,
        collected=0, passed=0, failed=0, errors=0, output="",
        line_coverage=None, branch_coverage=None, active_at_end=active_test)
    return backend

def _assert_back_in_revise(agent) -> None:
    s = agent.session
    assert s.last_tests_passed is False
    assert s.tests_changed_since_run_tests is False
    assert agent.effective_agent_mode() == "needs_editing"
    assert {t.__name__ for t in offered_tools(agent)} == _REVISE

@pytest.mark.parametrize("case", _NO_REVERT_CASES)
def test_coverage_timeout_without_a_revert_returns_to_revise(tmp_path, monkeypatch, case):
    agent = _coverage_agent(tmp_path)
    s = agent.session
    if case == "no_snapshot":
        _target_test(agent).write_text(SLOW, encoding="utf-8")
    else:
        _measure_snapshot(agent)
        if case == "named_hang":
            _target_test(agent).write_text(SLOW, encoding="utf-8")
    active = "test_foo.py::test_slow" if case == "named_hang" else None
    monkeypatch.setattr(tc.languages, "get_backend",
                        lambda *a, **kw: _hanging_backend(_target_test(agent), active))
    s.writes_done = 1
    s.last_tests_passed = True
    s.tests_changed_since_run_tests = True
    s.tests_changed_since_measure["coverage"] = True
    out = agent.run_tests()
    assert "TIMED OUT" in out or "did not finish" in out
    assert "run_tests" not in {t.__name__ for t in offered_tools(agent)}
    _assert_back_in_revise(agent)

@pytest.mark.parametrize("case", _NO_REVERT_CASES)
def test_mutation_timeout_without_a_revert_returns_to_revise(tmp_path, monkeypatch, case):
    agent = _mutation_agent(tmp_path)
    s = agent.session
    if case == "no_snapshot":
        agent.target_path().write_text(SLOW, encoding="utf-8")
    else:
        _mutation_snapshot(agent)
        agent.target_path().write_text(
            SNAPSHOT if case == "file_is_the_snapshot" else SLOW, encoding="utf-8")
    active = "test_foo.py::test_slow" if case == "named_hang" else None
    monkeypatch.setattr(tm.languages, "get_backend",
                        lambda *a, **kw: _hanging_backend(agent.target_path(), active))
    s.writes_done = 1
    s.last_tests_passed = True
    s.tests_changed_since_run_tests = True
    s.tests_changed_since_mutation = True
    out = agent.run_tests()
    assert "TIMED OUT" in out or "did not finish" in out
    _assert_back_in_revise(agent)


def test_coverage_measurement_timeout_without_a_revert_returns_to_revise(tmp_path, monkeypatch):
    # The measured run itself hangs on a run green suite, with no best-green
    # to revert to.
    agent = _coverage_agent(tmp_path)
    s = agent.session
    monkeypatch.setattr(tc.languages, "get_backend",
                        lambda *a, **kw: _hanging_backend(_target_test(agent), None))
    s.writes_done = 1
    s.last_tests_passed = True
    s.last_run_tests_result = "1 passed"
    s.tests_changed_since_run_tests = False
    s.tests_changed_since_measure["coverage"] = True
    assert agent.effective_agent_mode() == "needs_measurement"
    out = agent.measure_coverage()
    assert "TIMED OUT" in out or "did not finish" in out
    _assert_back_in_revise(agent)


def test_mutation_measurement_timeout_without_a_revert_returns_to_revise(tmp_path, monkeypatch):
    # The measurement's normal run hangs on a run green suite, with no
    # best-green to revert to.
    agent = _mutation_agent(tmp_path)
    s = agent.session
    agent.target_path().write_text(SLOW, encoding="utf-8")
    monkeypatch.setattr(tm.languages, "get_backend",
                        lambda *a, **kw: _hanging_backend(agent.target_path(), None))
    s.writes_done = 1
    s.last_tests_passed = True
    s.last_run_tests_result = "1 passed"
    s.tests_changed_since_run_tests = False
    s.tests_changed_since_mutation = True
    assert agent.effective_agent_mode() == "needs_measurement"
    out = agent.measure_mutation()
    assert "TIMED OUT" in out or "did not finish" in out
    _assert_back_in_revise(agent)
