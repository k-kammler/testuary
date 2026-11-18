"""Coverage-goal, stagnation and recovery feedback: _format_trend, goal and
stall checks, stall hints and the coverage agent's recovery-state block."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_fixtures import coverage_agent

from testuary.agent import coverage_agent as tc
from phase_values import set_phase_value

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture(autouse=True)
def _pin_line_metric(monkeypatch):
    # These tests read ``best_coverage_pct`` (line) only, so metric=line keeps
    # the bundled default ``min`` from requiring both metrics in every test.
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "line")


# _format_trend, a pure function

def test_format_trend_first_measurement() -> None:
    """Without a prior value the line marks ``(first measurement)``."""
    assert tc._format_trend("Coverage", None, 75.0) == "Coverage: 75.0% (first measurement)"


def test_format_trend_improvement() -> None:
    """A positive delta gets a ``+`` sign and shows the previous value, so
    the model sees whether its last edit helped."""
    out = tc._format_trend("Coverage", 70.0, 80.0)
    assert "70.0%" in out
    assert "80.0%" in out
    assert "+10.0%" in out


def test_format_trend_regression() -> None:
    """A negative delta has no extra sign, since the number carries its
    minus."""
    out = tc._format_trend("Mutation", 80.0, 70.0)
    assert "-10.0%" in out
    assert "(+" not in out


def test_format_trend_custom_unit() -> None:
    """``unit`` overrides the default ``%`` suffix."""
    out = tc._format_trend("Time", 100.0, 80.0, unit="ms")
    assert "ms" in out
    assert "%" not in out


# _coverage_goal_met

def test_coverage_goal_met_not_yet(tmp_path) -> None:
    """In a fresh state the goal is not met."""
    assert coverage_agent(tmp_path)._coverage_goal_met() is False


def test_coverage_goal_met_below_goal(tmp_path) -> None:
    """Coverage below the goal does not meet it."""
    a = coverage_agent(tmp_path)
    a.session.best_coverage_pct = 80.0
    assert a._coverage_goal_met() is False


def test_coverage_goal_met_at_or_above_goal(tmp_path) -> None:
    """Accept coverage at or above the configured goal."""
    a = coverage_agent(tmp_path)
    a.session.best_coverage_pct = 95.0
    assert a._coverage_goal_met() is True


def test_coverage_goal_met_boundary_exactly_at_goal(tmp_path) -> None:
    """Coverage exactly at the goal threshold is treated as 'met'."""
    cov_goal = float(tc._AGENT_CONFIG.get("goal", 90))
    a = coverage_agent(tmp_path)
    a.session.best_coverage_pct = cov_goal
    assert a._coverage_goal_met() is True


# _coverage_stalled_below_goal

def test_coverage_stalled_below_goal_returns_false_when_improving(tmp_path) -> None:
    """A stagnation streak of 0 is not a stall."""
    assert coverage_agent(tmp_path)._coverage_stalled_below_goal() is False


def test_coverage_stalled_below_goal_requires_streak_AND_below_goal(tmp_path) -> None:
    """A stall needs both a stagnation streak and coverage below the goal.
    Either alone gives False."""
    max_measurements_without_improvement = int(tc._AGENT_CONFIG.get("max_measurements_without_improvement", 10))
    a = coverage_agent(tmp_path)
    a.session.measurements_without_improvement = max_measurements_without_improvement
    a.session.best_coverage_pct = 50.0   # below 90 goal
    assert a._coverage_stalled_below_goal() is True


def test_coverage_stalled_above_goal_does_not_count_as_stalled(tmp_path) -> None:
    """High coverage with a high stagnation streak is success, not a stall."""
    max_measurements_without_improvement = int(tc._AGENT_CONFIG.get("max_measurements_without_improvement", 10))
    a = coverage_agent(tmp_path)
    a.session.measurements_without_improvement = max_measurements_without_improvement
    a.session.best_coverage_pct = 95.0
    assert a._coverage_stalled_below_goal() is False


# _coverage_stall_hint

def test_coverage_stall_hint_goal_met_directs_return(tmp_path) -> None:
    """When the coverage goal is met, the hint tells the model to
    return a CoverageTestResult."""
    a = coverage_agent(tmp_path)
    a.session.best_coverage_pct = 92.0
    hint = a._coverage_stall_hint()
    assert "return" in hint.lower()


def test_coverage_stall_hint_stalled_below_goal_with_early_stop(tmp_path, monkeypatch) -> None:
    """Stalled below the goal with allow_early_stop=True, the hint
    tells the model to return what it has."""
    max_measurements_without_improvement = int(tc._AGENT_CONFIG.get("max_measurements_without_improvement", 10))
    a = coverage_agent(tmp_path)
    a.session.measurements_without_improvement = max_measurements_without_improvement
    a.session.best_coverage_pct = 50.0
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "allow_early_stop", "true")
    hint = a._coverage_stall_hint()
    assert "return" in hint.lower() or "have" in hint.lower()


# recovery_state_block

def test_build_recovery_state_block_fresh_state(tmp_path: Path) -> None:
    """In a fresh state the block shows 'NONE YET' for the test file and
    run_tests as not called, and names the framework's run at the loop's
    start, since the status beside it is that run's."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    block = coverage_agent(tmp_path, test_output_dir=str(test_dir)).recovery_state_block()
    assert "NONE YET" in block
    assert ("  Last run_tests:    (not called yet; the framework ran the suite at the "
            "loop's start)\n") in block
    assert "  Last test status:  GREEN at the loop's start (the framework's measurement)\n" in block
    assert "source is unchanged and already in your context" not in block
    assert "tool offered in the current state" in block


def test_build_recovery_state_block_includes_metrics(tmp_path: Path) -> None:
    """When best_coverage_pct is set, it appears in the recovery
    block alongside its goal."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    a = coverage_agent(tmp_path, test_output_dir=str(test_dir))
    a.session.best_coverage_pct = 75.0
    a.session.writes_done = 3
    block = a.recovery_state_block()
    assert "75" in block
    assert "3 write" in block


def test_build_recovery_state_block_red_marker(tmp_path: Path) -> None:
    """A red run_tests (last_tests_passed=False) puts a 'RED' marker in the
    block."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    a = coverage_agent(tmp_path, test_output_dir=str(test_dir))
    a.session.last_tests_passed = False
    a.session.last_run_tests_result = (
        "TESTS: FAILED | collected 1 | passed 0 | failed 1 | errors 0 | elapsed 0.10s")
    block = a.recovery_state_block()
    assert "RED" in block or "fail" in block.lower()


def test_marker_and_recovery_state_the_goal_metric_with_both_values(
    tmp_path: Path, monkeypatch,
) -> None:
    """Under ``min`` the goal is the lower of line and branch: best line
    100% alone does not meet a 100% goal while branch is 98.5%. The state
    marker and the recovery snapshot give both values, the goal metric and
    the goal, never line coverage as the goal."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "both")
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal", 100.0)
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    a = coverage_agent(tmp_path, test_output_dir=str(test_dir))
    a.session.best_coverage_pct = 100.0
    a.session.best_branch_coverage_pct = 98.5
    a.session.measurements_without_improvement = 2

    expected = "line 100.0%, branch 98.5%; goal metric (lower of line and branch) 98.5%, goal 100%"
    assert f"  Best coverage:     {expected}\n" in a.recovery_state_block()
    assert "Best line coverage" not in a.recovery_state_block()
    assert (
        "coverage=line 100%, branch 98%; goal metric (lower of line and "
        "branch) 98%, goal 100% (stalled 2 measurements) |"
    ) in a.build_state_marker()


def test_marker_without_branches_judges_line_coverage(tmp_path: Path, monkeypatch) -> None:
    """A target without branches is judged by its line coverage, whatever
    the configured metric."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "both")
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal", 90)
    a = coverage_agent(tmp_path)
    a.session.best_coverage_pct = 92.25

    assert (
        "coverage=line 92%, no branches; goal metric (line) 92%, goal 90% |"
    ) in a.build_state_marker()


def test_recovery_snapshot_counts_lines_as_read_tests_and_shows_the_run_summary(
    tmp_path: Path,
) -> None:
    """A file ending in a newline has as many lines as read_tests reports,
    and a run result that a note precedes is shown by its TESTS summary,
    not cut inside the note."""
    from testuary.agent import _context

    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    a = coverage_agent(tmp_path, test_output_dir=str(test_dir))
    content = "def test_a():\n    assert True\n"
    (test_dir / "test_foo.py").write_text(content, encoding="utf-8")
    a.session.last_run_tests_result = (
        "[STANDS ALONE, NOTE] The file passes on its own, but some tests leave\n"
        "shared state changed, so a test run alone or in another order can fail\n"
        "because of them:\n\nTESTS: PASSED | collected 2 | passed 2 | failed 0 | "
        "errors 0 | elapsed 0.10s\n\n2 passed in 0.05s")

    block = a.recovery_state_block()

    assert "# test_foo.py (2 lines)" in _context.render_read(content, name="test_foo.py", max_chars=None)
    assert "  Working file on disk: test_foo.py (2 lines, 0 write(s) so far)\n" in block
    assert ("  Last run_tests:    TESTS: PASSED | collected 2 | passed 2 | failed 0 | "
            "errors 0 | elapsed 0.10s\n") in block
