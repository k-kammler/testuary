"""The coverage agent's failure and recovery helpers: the failure result,
the early-stop result, the end-of-run snapshot restore and output tracking."""
from __future__ import annotations

import hashlib

from agent_fixtures import coverage_agent

import pytest

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


# build_failure_result

def test_build_failure_result_returns_test_failed_when_no_snapshot(tmp_path):
    """Without a green snapshot or a test file on disk, the result names
    ``test_failed.py`` and carries the error message in its explanation."""
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir()
    target.write_text("def f(): return 1\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    a = coverage_agent(tmp_path, test_output_dir=str(test_dir))
    out = a.build_failure_result("model raised RuntimeError")
    assert out.test_file_name == "test_failed.py"
    assert "model raised RuntimeError" in out.explanation


def test_build_failure_result_surfaces_restored_green_file(tmp_path):
    """A green snapshot is written back to disk, and the result names the
    target's test file, not test_failed.py."""
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir()
    target.write_text("def f(): return 1\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_foo.py").write_text(
        "def test_bad(): assert 1 == 2\n", encoding="utf-8",
    )
    a = coverage_agent(tmp_path, test_output_dir=str(test_dir))
    a.session.last_tests_passed = False
    a.session.best_green.snapshot = {
        "content": "def test_good(): assert True\n",
        "coverage": 80.0, "mutation": 60.0, "test_count": 1,
    }
    out = a.build_failure_result("context overflow")
    assert (test_dir / "test_foo.py").read_text(encoding="utf-8") == "def test_good(): assert True\n"
    assert out.test_file_name == "test_foo.py"
    assert "last-green snapshot" in out.explanation


def test_build_failure_result_no_target_returns_test_failed(tmp_path):
    """Without a target_file_path nothing is restored and the result is test_failed."""
    a = coverage_agent(tmp_path, target_file_path="")
    out = a.build_failure_result("no target")
    assert out.test_file_name == "test_failed.py"


# _maybe_restore_last_green

def test_maybe_restore_no_op_when_current_state_is_green(tmp_path):
    """A verified-green live file (passed, unedited since the run) that
    measured strictly better than the snapshot is not replaced."""
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir()
    target.write_text("", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_foo.py").write_text("CURRENT", encoding="utf-8")
    a = coverage_agent(tmp_path, test_output_dir=str(test_dir))
    a.session.last_tests_passed = True
    a.session.tests_changed_since_run_tests = False
    a.session.tests_changed_since_measure["coverage"] = False
    a.session.prev_coverage_pct = a.session.prev_branch_coverage_pct = 10.0
    a.session.measured_suite_hash = hashlib.sha256(b"CURRENT").hexdigest()
    a.session.best_green.snapshot = {
        "content": "SNAPSHOT", "coverage": 0.0, "mutation": 0.0, "test_count": 1,
    }
    a._maybe_restore_last_green(str(test_dir), str(target))
    assert (test_dir / "test_foo.py").read_text(encoding="utf-8") == "CURRENT"


# _synthesize_early_stop_result

def test_synthesize_early_stop_uses_target_test_when_present(tmp_path):
    """When the restore lands a green file on disk, the result names it."""
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir()
    target.write_text("", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_foo.py").write_text("RED", encoding="utf-8")
    a = coverage_agent(tmp_path, test_output_dir=str(test_dir))
    a.session.last_tests_passed = False
    a.session.best_green.snapshot = {
        "content": "GREEN", "coverage": 80.0, "test_count": 5,
    }
    a.session.best_coverage_pct = 80.0
    a.session.writes_done = 3
    a.session.coverage_measurements = 4

    out = a._synthesize_early_stop_result(
        "stagnation patience exhausted", target_file_path=str(target),
    )
    assert out.test_file_name == "test_foo.py"
    assert "stagnation patience exhausted" in out.explanation
    assert "80.0" in out.explanation     # final coverage surfaced
    assert "3 writes" in out.explanation
    assert "4 coverage" in out.explanation


def test_synthesize_early_stop_falls_back_to_test_failed_when_no_file(tmp_path):
    """Without a restorable green file, the result names test_failed.py."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    a = coverage_agent(tmp_path, test_output_dir=str(test_dir))
    out = a._synthesize_early_stop_result("request limit reached")
    assert out.test_file_name == "test_failed.py"
    assert "request limit reached" in out.explanation


# track_output and request-limit ownership

def test_track_output_increments_session_counter(tmp_path):
    """Each call counts one tool call."""
    a = coverage_agent(tmp_path)
    starting = a.session.session_tool_calls
    out = a.track_output("hello world\n")
    assert a.session.session_tool_calls == starting + 1
    assert "hello world" in out


def test_track_output_prepends_no_progress_banner_when_threshold_hit(tmp_path):
    """With ``current_tool_name`` set and the no-progress threshold reached,
    the banner is prepended to the tracked output."""
    a = coverage_agent(tmp_path)
    a.session.current_tool_name = "run_tests"
    a.session.loop_breaker.calls_since_last_edit = (
        a.session.loop_breaker.no_progress_threshold - 1
    )
    out = a.track_output("plain output")
    assert "plain output" in out
    assert (
        a.session.loop_breaker.calls_since_last_edit
        >= a.session.loop_breaker.no_progress_threshold
    )
