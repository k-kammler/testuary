"""The coverage phase's best-green snapshot: its ranking, its updates and the
terminal restore (which artifact is kept, with which metrics, and when the
file on disk is reverted).
"""
from __future__ import annotations

from pathlib import Path

from agent_fixtures import project_agent

from testuary.agent import coverage_agent as tc
from phase_values import set_phase_value
import pytest

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _make_agent(tmp_path: Path) -> "tc.CoverageAgent":
    agent = project_agent(tmp_path)
    agent.session.active_backend = tc.languages.get_backend("python")
    return agent


def _snap(coverage=None, branch=None, test_count=1, exec_ms=None,
          snapshot_clock_s=1.0,
          content="def test_a():\n    assert True\n"):
    return {
        "content": content,
        "coverage": coverage,
        "branch_coverage": branch,
        "test_count": test_count,
        "exec_time_ms": exec_ms,
        "snapshot_clock_s": snapshot_clock_s,
    }


# ---------------------------------------------------------------------------
# The ranking key, per goal-metric mode
# ---------------------------------------------------------------------------

def test_key_ranks_by_the_configured_goal_metric(monkeypatch):
    """The key leads with the goal metric: line, branch, the higher (either)
    or the lower (both)."""
    a = _snap(coverage=90.0, branch=50.0)
    b = _snap(coverage=80.0, branch=70.0)
    expectations = {
        "line": True,
        "branch": False,
        "either": True,
        "both": False,
    }
    for mode, a_wins in expectations.items():
        set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", mode)
        assert (
            tc._green_snapshot_key(a) > tc._green_snapshot_key(b)
        ) is a_wins, mode


def test_key_unmeasured_sorts_below_any_measured_snapshot(monkeypatch):
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "line")
    measured = _snap(coverage=50.0)
    unmeasured = _snap(coverage=None, branch=None)
    assert tc._green_snapshot_key(measured) > tc._green_snapshot_key(unmeasured)
    assert tc._green_snapshot_key(unmeasured) > tc._green_snapshot_key(None)


def test_key_has_no_tie_break(monkeypatch):
    """At an equal goal value, timing, test count and capture time order
    nothing, so the earlier snapshot is kept."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "line")
    slow = _snap(coverage=70.0, exec_ms=900.0)
    fast = _snap(coverage=70.0, exec_ms=100.0)
    assert tc._green_snapshot_key(fast) == tc._green_snapshot_key(slow)
    unmeasured_rt = _snap(coverage=70.0, exec_ms=None)
    assert tc._green_snapshot_key(fast) == tc._green_snapshot_key(unmeasured_rt)
    lean = _snap(coverage=70.0, exec_ms=100.0, test_count=3)
    bloated = _snap(coverage=70.0, exec_ms=100.0, test_count=9)
    assert tc._green_snapshot_key(lean) == tc._green_snapshot_key(bloated)
    early = _snap(
        coverage=70.0, exec_ms=100.0, test_count=3, snapshot_clock_s=1.0,
    )
    late = _snap(
        coverage=70.0, exec_ms=100.0, test_count=3, snapshot_clock_s=9.0,
    )
    assert tc._green_snapshot_key(early) == tc._green_snapshot_key(late)


# ---------------------------------------------------------------------------
# The update flow: which artifact the snapshot holds
# ---------------------------------------------------------------------------

def test_update_credits_fresh_metrics_and_ignores_stale(monkeypatch, tmp_path):
    """A green run with a fresh measurement is snapshotted, and a stale one
    cannot displace it."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "line")
    agent = _make_agent(tmp_path)
    s = agent.session
    test_path = Path(agent.deps.test_output_dir) / "test_foo.py"

    import hashlib as _hashlib

    def _measured(text: str) -> None:
        s.tests_changed_since_measure["coverage"] = False
        s.measured_suite_hash = _hashlib.sha256(
            text.encode("utf-8")
        ).hexdigest()

    v1 = "def test_a():\n    assert True\n"
    test_path.write_text(v1, encoding="utf-8")
    s.prev_coverage_pct = 60.0
    s.prev_branch_coverage_pct = 40.0
    _measured(v1)
    agent._maybe_update_best_green(
        agent.deps.test_output_dir, agent.deps.target_file_path,
    )
    assert s.best_green.snapshot is not None
    assert s.best_green.snapshot["content"] == v1
    assert s.best_green.snapshot["coverage"] == 60.0
    assert s.best_green.snapshot["branch_coverage"] == 40.0
    bak = test_path.parent / f".best_passing_{test_path.name}.bak"
    assert bak.read_text(encoding="utf-8") == v1

    v2 = v1 + "\ndef test_b():\n    assert True\n"
    test_path.write_text(v2, encoding="utf-8")
    s.tests_changed_since_measure["coverage"] = True
    agent._maybe_update_best_green(
        agent.deps.test_output_dir, agent.deps.target_file_path,
    )
    assert s.best_green.snapshot["content"] == v1

    s.prev_coverage_pct = 75.0
    s.prev_branch_coverage_pct = 55.0
    _measured(v2)
    agent._maybe_update_best_green(
        agent.deps.test_output_dir, agent.deps.target_file_path,
    )
    assert s.best_green.snapshot["content"] == v2
    assert s.best_green.snapshot["coverage"] == 75.0
    assert bak.read_text(encoding="utf-8") == v2


# ---------------------------------------------------------------------------
# The terminal restore: an unmeasured or worse final edit never ships
# ---------------------------------------------------------------------------

def _seed_snapshot(agent, content, coverage=70.0, branch=60.0):
    agent.session.best_green.snapshot = {
        "content": content,
        "coverage": coverage,
        "branch_coverage": branch,
        "test_count": 2,
        "exec_time_ms": 100.0,
        "time": 1.0,
    }


def test_terminal_restore_reverts_a_green_unmeasured_final_edit(
    monkeypatch, tmp_path,
):
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "line")
    agent = _make_agent(tmp_path)
    s = agent.session
    test_path = Path(agent.deps.test_output_dir) / "test_foo.py"
    best = "def test_best():\n    assert True\n"
    _seed_snapshot(agent, best)
    test_path.write_text(
        "def test_unmeasured():\n    assert True\n", encoding="utf-8",
    )
    s.last_tests_passed = True
    s.tests_changed_since_measure["coverage"] = True
    agent._maybe_restore_last_green(
        agent.deps.test_output_dir, agent.deps.target_file_path,
    )
    assert test_path.read_text(encoding="utf-8") == best
    assert s.best_green.restored_from_green is True
    assert s.prev_coverage_pct == 70.0
    assert s.prev_branch_coverage_pct == 60.0
    assert s.last_tests_passed is True


def test_terminal_restore_keeps_a_green_measured_better_final_edit(
    monkeypatch, tmp_path,
):
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "line")
    agent = _make_agent(tmp_path)
    s = agent.session
    test_path = Path(agent.deps.test_output_dir) / "test_foo.py"
    _seed_snapshot(agent, "def test_best():\n    assert True\n", coverage=70.0)
    live = "def test_better():\n    assert True\n"
    test_path.write_text(live, encoding="utf-8")
    s.last_tests_passed = True
    s.tests_changed_since_run_tests = False
    s.prev_coverage_pct = 85.0
    s.prev_branch_coverage_pct = 70.0
    s.tests_changed_since_measure["coverage"] = False
    import hashlib as _hashlib
    s.measured_suite_hash = _hashlib.sha256(
        live.encode("utf-8")
    ).hexdigest()
    agent._maybe_restore_last_green(
        agent.deps.test_output_dir, agent.deps.target_file_path,
    )
    assert test_path.read_text(encoding="utf-8") == live
    assert s.best_green.restored_from_green is False


def test_terminal_restore_reverts_a_red_final_edit(monkeypatch, tmp_path):
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "line")
    agent = _make_agent(tmp_path)
    s = agent.session
    test_path = Path(agent.deps.test_output_dir) / "test_foo.py"
    best = "def test_best():\n    assert True\n"
    _seed_snapshot(agent, best)
    test_path.write_text("def test_red():\n    assert False\n", encoding="utf-8")
    s.last_tests_passed = False
    agent._maybe_restore_last_green(
        agent.deps.test_output_dir, agent.deps.target_file_path,
    )
    assert test_path.read_text(encoding="utf-8") == best


def test_forced_restore_overrides_the_green_short_circuit(
    monkeypatch, tmp_path,
):
    """force=True (the in-loop regression revert) restores the snapshot even
    when stale cached values claim the live file is better."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "line")
    agent = _make_agent(tmp_path)
    s = agent.session
    test_path = Path(agent.deps.test_output_dir) / "test_foo.py"
    best = "def test_best():\n    assert True\n"
    _seed_snapshot(agent, best, coverage=70.0)
    test_path.write_text("def test_live():\n    assert True\n", encoding="utf-8")
    s.last_tests_passed = True
    s.prev_coverage_pct = 99.0
    s.tests_changed_since_measure["coverage"] = False
    agent._maybe_restore_last_green(
        agent.deps.test_output_dir, agent.deps.target_file_path, force=True,
    )
    assert test_path.read_text(encoding="utf-8") == best


def test_terminal_restore_no_snapshot_is_a_no_op(tmp_path):
    agent = _make_agent(tmp_path)
    test_path = Path(agent.deps.test_output_dir) / "test_foo.py"
    live = "def test_live():\n    assert True\n"
    test_path.write_text(live, encoding="utf-8")
    assert agent.session.best_green.snapshot is None
    agent._maybe_restore_last_green(
        agent.deps.test_output_dir, agent.deps.target_file_path,
    )
    assert test_path.read_text(encoding="utf-8") == live
