"""The agent loops' best-green snapshot at equal scores.

In the coverage and the mutation phase, a later passing suite replaces the
snapshot only with a strictly better score. At equal scores the earlier one
stays: nothing is timed and no test count decides. Timing is the criterion of
the phase-end consolidation only.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from agent_fixtures import SOURCE, mutation_agent, project_agent

from testuary.agent import _helpers as ah
from testuary import consolidation as consolidate
from testuary.agent import coverage_agent as tc
from testuary.agent import mutation_agent as tm

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

# The later suite has fewer tests and runs faster: neither wins a tie.
EARLY = "def test_a():\n    assert True\n\n\ndef test_b():\n    assert 1\n"
LATE = "def test_c():\n    assert True\n"


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


@pytest.fixture
def no_benchmark(monkeypatch):
    """Fails the test if anything times suites."""
    calls = []

    def refuse(*args, **kwargs):
        calls.append((args, kwargs))
        raise AssertionError("the loop timed suites")

    monkeypatch.setattr(consolidate, "measure_interleaved", refuse)
    return calls


# ---------------------------------------------------------------------------
# Coverage
# ---------------------------------------------------------------------------


def _coverage_agent(tmp_path: Path) -> "tc.CoverageAgent":
    return project_agent(tmp_path, source="def f(x):\n    return x\n")


def _coverage_measured(agent, text: str, pct: float) -> None:
    """``text`` on disk, green and measured at ``pct`` line and branch."""
    (Path(agent.deps.test_output_dir) / "test_foo.py").write_text(text, encoding="utf-8")
    s = agent.session
    s.last_tests_passed = True
    s.tests_changed_since_run_tests = False
    s.tests_changed_since_measure["coverage"] = False
    s.prev_coverage_pct = pct
    s.prev_branch_coverage_pct = pct
    s.measured_suite_hash = _sha(text)
    s.last_suite_exec_ms = 1.0 if text == LATE else 900.0


def _snapshot_content(agent) -> str:
    return agent.session.best_green.snapshot["content"]


def test_coverage_equal_score_keeps_the_earlier_snapshot(tmp_path, no_benchmark):
    agent = _coverage_agent(tmp_path)
    _coverage_measured(agent, EARLY, 50.0)
    agent._maybe_update_best_green(agent.deps.test_output_dir, agent.deps.target_file_path)
    _coverage_measured(agent, LATE, 50.0)
    agent._maybe_update_best_green(agent.deps.test_output_dir, agent.deps.target_file_path)
    assert _snapshot_content(agent) == EARLY
    assert no_benchmark == []


def test_coverage_a_strictly_better_score_replaces_the_snapshot(tmp_path, no_benchmark):
    agent = _coverage_agent(tmp_path)
    _coverage_measured(agent, EARLY, 50.0)
    agent._maybe_update_best_green(agent.deps.test_output_dir, agent.deps.target_file_path)
    _coverage_measured(agent, LATE, 60.0)
    agent._maybe_update_best_green(agent.deps.test_output_dir, agent.deps.target_file_path)
    assert _snapshot_content(agent) == LATE
    assert no_benchmark == []


def test_coverage_restore_after_equal_scores_returns_the_earlier_snapshot(
        tmp_path, no_benchmark):
    agent = _coverage_agent(tmp_path)
    _coverage_measured(agent, EARLY, 50.0)
    agent._maybe_update_best_green(agent.deps.test_output_dir, agent.deps.target_file_path)
    _coverage_measured(agent, LATE, 50.0)
    agent._maybe_restore_last_green(agent.deps.test_output_dir, agent.deps.target_file_path)
    path = Path(agent.deps.test_output_dir) / "test_foo.py"
    assert path.read_text(encoding="utf-8") == EARLY
    assert agent.session.best_green.restored_from_green is True
    assert no_benchmark == []


# ---------------------------------------------------------------------------
# Mutation
# ---------------------------------------------------------------------------


def _mutation_agent(tmp_path: Path) -> "tm.MutationAgent":
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    source = tmp_path / "src" / "foo.py"
    source.parent.mkdir()
    source.write_text(SOURCE, encoding="utf-8")
    return mutation_agent(tmp_path, test_output_dir=str(test_dir), mutation_test_file_name="test_foo.py",
                          coverage_baseline_pct=85.0)


def _mutation_measured(agent, text: str, score: float) -> None:
    """``text`` on disk, green and measured at ``score``, stamped into
    best-green the way the mutation measurement stamps it."""
    agent.target_path().write_text(text, encoding="utf-8")
    s = agent.session
    s.last_tests_passed = True
    s.tests_changed_since_run_tests = False
    s.tests_changed_since_mutation = False
    s.prev_mutation_score = score
    s.measured_suite_hash = _sha(text)
    s.last_suite_exec_ms = 1.0 if text == LATE else 900.0
    ah.update_best_green(
        s.best_green, agent.target_path(), coverage=85.0, mutation=score,
        test_count=text.count("def test_"), exec_time_ms=s.last_suite_exec_ms,
        persist_to_disk=False, loop_state=agent._snapshot_loop_state(text))


def test_mutation_equal_score_keeps_the_earlier_snapshot(tmp_path, no_benchmark):
    agent = _mutation_agent(tmp_path)
    _mutation_measured(agent, EARLY, 70.0)
    _mutation_measured(agent, LATE, 70.0)
    assert agent.session.best_green.snapshot["content"] == EARLY
    assert no_benchmark == []


def test_mutation_a_strictly_better_score_replaces_the_snapshot(tmp_path, no_benchmark):
    agent = _mutation_agent(tmp_path)
    _mutation_measured(agent, EARLY, 70.0)
    _mutation_measured(agent, LATE, 80.0)
    assert agent.session.best_green.snapshot["content"] == LATE
    assert no_benchmark == []


def test_mutation_restore_after_equal_scores_returns_the_earlier_snapshot(
        tmp_path, no_benchmark):
    agent = _mutation_agent(tmp_path)
    _mutation_measured(agent, EARLY, 70.0)
    _mutation_measured(agent, LATE, 70.0)
    agent._settle_best_green(agent.target_path(), log_fn=agent.log)
    assert agent.target_path().read_text(encoding="utf-8") == EARLY
    assert agent.session.best_green.restored_from_green is True
    assert no_benchmark == []


def test_the_shared_key_holds_scores_only():
    base = {"mutation": 75.0, "coverage": 80.0, "test_count": 10,
            "exec_time_ms": 900.0, "snapshot_clock_s": 5.0}
    other = {**base, "test_count": 1, "exec_time_ms": 1.0, "snapshot_clock_s": 1.0}
    assert ah.green_snapshot_key(base) == ah.green_snapshot_key(other)
    assert ah.green_snapshot_key({**base, "coverage": 81.0}) > ah.green_snapshot_key(base)
