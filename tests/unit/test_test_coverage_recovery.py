"""Tests for the coverage agent's recovery helpers: early stop on the goal or
stagnation, best-green snapshots, restore and revert on regression, suite
timeouts, and the pure uncovered-code rendering and cleanup.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_fixtures import coverage_agent, passing_delivery, project_agent, run_result

from testuary.languages.generic import GenericBackend
from testuary.agent import coverage_agent as tc
from phase_values import set_phase_value

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _make_agent(tmp_path: Path) -> "tc.CoverageAgent":
    source = "def f(x):\n    if x > 0:\n        return x\n    return 0\n"
    return project_agent(tmp_path, source=source, test_source="def test_a(): assert True\n")


def _bare_agent(root: Path) -> "tc.CoverageAgent":
    """An agent with dummy deps in ``root``, for tests that spy the restore."""
    return coverage_agent(root, target_file_path=str(root / "foo.py"), target_file_content="x",
                          test_output_dir=str(root / "tests"))


# _maybe_set_early_stop

def test_maybe_set_early_stop_no_metrics_yet_returns_none(tmp_path: Path) -> None:
    """With nothing measured the helper returns None and changes no state."""
    a = _make_agent(tmp_path)
    assert a._maybe_set_early_stop(source_tool="measure_coverage") is None
    assert a.session.early_stop_result is None
    assert a.session.early_stop_reason is None


def test_maybe_set_early_stop_coverage_goal_met_sets_state(tmp_path) -> None:
    """Coverage at or above the goal with extra_measurements_after_goal=0 (default): the
    helper synthesizes the result on state, sets reason ``coverage_goal_met``
    and returns a directive string."""
    cov_goal = float(tc._AGENT_CONFIG.get("goal", 90))
    a = _make_agent(tmp_path)
    a.session.best_coverage_pct = cov_goal
    a.session.best_branch_coverage_pct = cov_goal
    msg = a._maybe_set_early_stop(source_tool="measure_coverage")
    assert msg is not None
    assert "FRAMEWORK STOP" in msg
    assert a.session.early_stop_reason == "coverage_goal_met"
    assert a.session.early_stop_result is not None


def test_maybe_set_early_stop_goal_met_extra_measurements(monkeypatch, tmp_path) -> None:
    """With extra_measurements_after_goal > 0, the first goal-met measurement sets
    ``seen_coverage_goal=True`` but returns None. Later measurements at or
    above the goal count up until patience is exhausted."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "extra_measurements_after_goal", 2)
    cov_goal = float(tc._AGENT_CONFIG.get("goal", 90))
    a = _make_agent(tmp_path)
    a.session.best_coverage_pct = cov_goal + 5.0
    a.session.best_branch_coverage_pct = cov_goal + 5.0
    assert a._maybe_set_early_stop(source_tool="measure_coverage") is None
    assert a.session.seen_coverage_goal is True
    assert a._maybe_set_early_stop(source_tool="measure_coverage") is None
    msg = a._maybe_set_early_stop(source_tool="measure_coverage")
    assert msg is not None
    assert a.session.early_stop_reason == "coverage_goal_met"


def test_maybe_set_early_stop_no_improvement_below_goal(monkeypatch, tmp_path) -> None:
    """Coverage stuck below the goal with measurements_without_improvement >= patience and
    early_stop=True: the helper sets state with reason ``coverage_stagnation``."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "allow_early_stop", "true")
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_measurements_without_improvement", 3)
    a = _make_agent(tmp_path)
    a.session.best_coverage_pct = 50.0   # below 90 goal
    a.session.measurements_without_improvement = 3
    msg = a._maybe_set_early_stop(source_tool="measure_coverage")
    assert msg is not None
    assert "stagn" in a.session.early_stop_reason.lower()


def test_maybe_set_early_stop_disabled_returns_none(monkeypatch, tmp_path) -> None:
    """With ``allow_early_stop=False`` a stall below the goal returns
    None, and the run continues until usage_limits or the goal."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "allow_early_stop", False)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_measurements_without_improvement", 3)
    a = _make_agent(tmp_path)
    a.session.best_coverage_pct = 50.0
    a.session.measurements_without_improvement = 5
    assert a._maybe_set_early_stop(source_tool="measure_coverage") is None
    assert a.session.early_stop_result is None


# _synthesize_early_stop_result

def test_synthesize_early_stop_result_uses_on_disk_test_file(tmp_path) -> None:
    """When the target's test file exists, the synthesized result names it."""
    src = tmp_path / "src" / "foo.py"
    src.parent.mkdir()
    src.write_text("def f(): return 1\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_foo.py").write_text("def test_a(): pass\n", encoding="utf-8")
    a = coverage_agent(tmp_path, test_output_dir=str(test_dir))
    result = a._synthesize_early_stop_result("coverage_goal_met", target_file_path=str(src))
    assert result.test_file_name == "test_foo.py"
    assert "coverage_goal_met" in result.explanation.lower() or "goal" in result.explanation.lower()


def test_synthesize_early_stop_result_no_test_file_returns_failed_marker(tmp_path) -> None:
    """Without an on-disk test file, the synthesized result uses the
    ``test_failed.py`` framework marker."""
    src = tmp_path / "src" / "foo.py"
    src.parent.mkdir()
    src.write_text("def f(): return 1\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    a = coverage_agent(tmp_path, test_output_dir=str(test_dir))
    result = a._synthesize_early_stop_result("stagnation", target_file_path=str(src))
    assert result.test_file_name == "test_failed.py"


# _render_uncovered_code, a pure function

def test_render_uncovered_code_empty_when_no_missing(tmp_path: Path) -> None:
    """No uncovered lines in the backend's normalized ``missing_lines``
    render as ""."""
    a = _make_agent(tmp_path)
    assert tc._render_uncovered_code(a.deps.target_file_path, ()) == ""
    assert tc._render_uncovered_code(a.deps.target_file_path, []) == ""


def test_render_uncovered_code_renders_excerpt_with_markers(tmp_path: Path) -> None:
    """Given uncovered line numbers, the target source is read directly and
    the uncovered lines get ``>>>`` markers with surrounding context."""
    a = _make_agent(tmp_path)
    out = tc._render_uncovered_code(a.deps.target_file_path, (4,))
    assert "UNCOVERED" in out
    assert ">>>" in out
    assert "return 0" in out          # line 4's content
    assert "   4 |" in out or ">>>    4 |" in out
    assert "read_source" not in out
    assert "too many" not in out


def test_render_uncovered_code_shows_all_ranges_and_suggests_reads(tmp_path: Path, monkeypatch) -> None:
    """Past the inline character limit the complete range map is still
    shown, and regions that do not fit inline get read_source suggestions
    instead of being truncated."""
    defaults = tc._limits.defaults()
    monkeypatch.setattr(tc._limits, "value",
                        lambda name: 600 if name == "uncovered_chars" else defaults[name])
    src = tmp_path / "big.py"
    lines = []
    for i in range(40):
        lines.append(f"def f{i}(x):")
        lines.append(
            f"    if x: return x + {i} + "
            "'padding wide enough that these regions blow the small limit'"
        )
        lines.append("    return 0")
        lines.append("")
        lines.append("")
        lines.append("")
    src.write_text("\n".join(lines), encoding="utf-8")
    missing = [6 * i + 2 for i in range(40)]
    out = tc._render_uncovered_code(str(src), missing)
    assert "further uncovered regions omitted, read them with read_source" in out
    assert "\nComplete uncovered ranges: " in out
    assert "read_source(start_line=" in out
    assert str(missing[-1]) in out


def test_render_uncovered_code_empty_when_source_unreadable(tmp_path: Path) -> None:
    """An unreadable target file renders as "" even with uncovered lines."""
    out = tc._render_uncovered_code(str(tmp_path / "does_not_exist.py"), (1, 2))
    assert out == ""


# _render_uncovered_branches (pure): branch detail like the line render

_BRANCH_SRC = (
    "def f(x):\n"        # 1
    "    if x > 0:\n"    # 2  decision
    "        return 1\n"  # 3
    "    return 0\n"     # 4
)


def test_render_uncovered_branches_empty_when_no_arcs(tmp_path: Path) -> None:
    """No uncovered arcs render as "", so a branch metric already at its goal
    adds nothing beside the other metric's detail."""
    src = tmp_path / "m.py"
    src.write_text(_BRANCH_SRC, encoding="utf-8")
    assert tc._render_uncovered_branches(str(src), ()) == ""
    assert tc._render_uncovered_branches(str(src), []) == ""


def test_render_uncovered_branches_marks_decision_line_and_destination(tmp_path) -> None:
    """An uncovered arc renders the decision line with a marker and annotates
    the untaken destination, the same level of detail as the line render."""
    src = tmp_path / "m.py"
    src.write_text(_BRANCH_SRC, encoding="utf-8")
    out = tc._render_uncovered_branches(str(src), ("2->4",))
    assert "UNCOVERED BRANCHES" in out
    assert ">>>    2 |" in out
    assert "if x > 0" in out
    # The untaken destination is named with the code it leads to, not just the
    # bare line number, so the model sees what the missing branch reaches.
    assert "<- branch to line 4" in out and "not taken" in out
    assert "return 0" in out                      # line 4's source snippet


def test_render_uncovered_branches_renders_exit_destination(tmp_path: Path) -> None:
    """A jump out of the function (``exit``) is rendered as such, not a line."""
    src = tmp_path / "m.py"
    src.write_text(_BRANCH_SRC, encoding="utf-8")
    out = tc._render_uncovered_branches(str(src), ("2->exit",))
    assert "branch to exit" in out and "control leaves the function" in out


def test_render_uncovered_branches_reads_java_branch_counts(tmp_path: Path) -> None:
    """Java names no destination: its "line->k of n branches untaken" entry
    is shown as the line's untaken count, not as a destination line."""
    src = tmp_path / "M.java"
    src.write_text("class M {\n  int f(int x) {\n    if (x > 0) return 1;\n"
                   "    return 0;\n  }\n}\n", encoding="utf-8")
    out = tc._render_uncovered_branches(str(src), ("3->1 of 2 branches untaken",))
    assert ">>>    3 |     if (x > 0) return 1;    <- 1 of 2 branches untaken\n" in out
    assert "branch to" not in out and "not taken" not in out


def test_render_uncovered_branches_empty_when_source_unreadable(tmp_path: Path) -> None:
    out = tc._render_uncovered_branches(str(tmp_path / "nope.py"), ("2->4",))
    assert out == ""


# _maybe_update_best_green and cleanup

def test_maybe_update_best_green_no_op_when_no_target(tmp_path: Path) -> None:
    """``target_file_path=None`` does nothing."""
    a = _make_agent(tmp_path)
    a._maybe_update_best_green(str(tmp_path), None)
    assert a.session.best_green.snapshot is None


def test_cleanup_removes_best_green_sidecar_any_extension(tmp_path: Path) -> None:
    """End-of-target cleanup removes the best-green sidecar regardless of the
    language extension carried in its name (extension-agnostic glob)."""
    py_side = tmp_path / ".best_passing_test_foo.py.bak"
    java_side = tmp_path / ".best_passing_FooTest.java.bak"
    py_side.write_text("x", encoding="utf-8")
    java_side.write_text("y", encoding="utf-8")
    tc._cleanup_temp_files(str(tmp_path))
    assert not py_side.exists()
    assert not java_side.exists()


def test_maybe_update_best_green_takes_first_snapshot(tmp_path: Path) -> None:
    """The first green call takes a snapshot of the current file content and
    scores. A score is credited only when
    ``tests_changed_since_measure[metric]=False``."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_foo.py"
    test_file.write_text("def test_a():\n    assert True\n", encoding="utf-8")
    target = tmp_path / "foo.py"
    target.write_text("def f(): pass\n", encoding="utf-8")
    a = coverage_agent(tmp_path, target_file_path=str(target),
                       target_file_content="def f(): pass\n", test_output_dir=str(test_dir))
    a.session.prev_coverage_pct = 85.0
    a.session.tests_changed_since_measure["coverage"] = False  # fresh
    import hashlib as _hashlib
    a.session.measured_suite_hash = _hashlib.sha256(
        test_file.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    a._maybe_update_best_green(str(test_dir), str(target))
    assert a.session.best_green.snapshot is not None
    assert a.session.best_green.snapshot["coverage"] == 85.0


# _maybe_restore_last_green

def test_maybe_restore_last_green_no_op_when_live_beats_snapshot(tmp_path: Path) -> None:
    """A live file that is green and at least as good as the snapshot is not
    restored."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_foo.py"
    test_file.write_text(
        "def test_a(): assert True\ndef test_b(): assert True\n", encoding="utf-8",
    )
    a = coverage_agent(tmp_path, target_file_path=str(tmp_path / "foo.py"),
                       target_file_content="x", test_output_dir=str(test_dir))
    a.session.last_tests_passed = True
    a.session.tests_changed_since_run_tests = False
    a.session.prev_coverage_pct = 95.0
    a.session.prev_branch_coverage_pct = 92.0
    a.session.tests_changed_since_measure = {"coverage": False, "mutation": False}
    import hashlib as _hashlib
    a.session.measured_suite_hash = _hashlib.sha256(
        test_file.read_text(encoding="utf-8").encode("utf-8")
    ).hexdigest()
    a.session.best_green.snapshot = {
        "content": "old snapshot\n",
        "coverage": 80.0, "branch_coverage": 80.0,
        "test_count": 2, "time": 1.0,
    }
    a._maybe_restore_last_green(str(test_dir), str(tmp_path / "foo.py"))
    assert "test_a" in test_file.read_text(encoding="utf-8")
    assert a.session.best_green.restored_from_green is False


def test_maybe_restore_last_green_reverts_bloated_file_to_snapshot(tmp_path: Path) -> None:
    """The model added tests without improving any metric. The live file is
    green at the same metric but has more tests, so its
    ``_green_snapshot_key`` is worse and the restore delivers the smaller
    suite."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_foo.py"
    bloated_content = "\n".join(f"def test_{i}(): assert True" for i in range(245)) + "\n"
    test_file.write_text(bloated_content, encoding="utf-8")
    a = coverage_agent(tmp_path, target_file_path=str(tmp_path / "foo.py"),
                       target_file_content="x", test_output_dir=str(test_dir))
    a.session.last_tests_passed = True
    a.session.prev_coverage_pct = 93.87
    a.session.prev_branch_coverage_pct = 89.9
    a.session.tests_changed_since_measure = {"coverage": False, "mutation": False}
    snapshot_content = "\n".join(f"def test_{i}(): assert True" for i in range(241)) + "\n"
    a.session.best_green.snapshot = {
        "content": snapshot_content,
        "coverage": 93.87, "branch_coverage": 89.9,
        "test_count": 241, "time": 1.0,
    }
    a._maybe_restore_last_green(str(test_dir), str(tmp_path / "foo.py"))
    assert test_file.read_text(encoding="utf-8") == snapshot_content
    assert a.session.best_green.restored_from_green is True
    assert a.session.prev_coverage_pct == pytest.approx(93.87)
    assert a.session.prev_branch_coverage_pct == pytest.approx(89.9)


def test_revert_on_regression_forces_restore_despite_stale_coverage(tmp_path: Path) -> None:
    """A confirmed regression reverts the file to the snapshot even when the
    green short-circuit's cached prev_coverage_pct is stale and high. The
    fresh measurement decides is_regression, and force=True makes the
    snapshot win regardless of the cache, so a "[REVERTED]" note never comes
    with the regressed suite still on disk."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_foo.py"
    # Live file: small and green, but with low coverage (the model rewrote
    # the suite down to a handful of tests).
    live = "\n".join(f"def test_{i}(): assert True" for i in range(3)) + "\n"
    test_file.write_text(live, encoding="utf-8")
    a = coverage_agent(tmp_path, target_file_path=str(tmp_path / "foo.py"),
                       target_file_content="x", test_output_dir=str(test_dir))
    a.session.last_tests_passed = True
    # Stale cache: prev_* still show the earlier high reading, which rates
    # the small live file as good as the snapshot.
    a.session.prev_coverage_pct = 78.8
    a.session.prev_branch_coverage_pct = 73.6
    a.session.tests_changed_since_measure = {"coverage": False, "mutation": False}
    snapshot_content = (
        "\n".join(f"def test_big_{i}(): assert True" for i in range(331)) + "\n"
    )
    a.session.best_green.snapshot = {
        "content": snapshot_content,
        "coverage": 78.8, "branch_coverage": 73.6, "test_count": 331, "time": 1.0,
    }
    # is_regression=True comes from the fresh measurement (37.3 < best 73.6).
    note = a._maybe_revert_on_regression(
        str(tmp_path / "foo.py"), True, current_goal=37.3, best_goal=73.6,
    )
    assert note is not None  # the model is told it was reverted
    # and the file holds the snapshot.
    assert test_file.read_text(encoding="utf-8") == snapshot_content
    assert a.session.best_green.restored_from_green is True


def test_maybe_restore_last_green_no_snapshot_no_op(tmp_path: Path) -> None:
    """No green snapshot was taken in this run, so nothing is restored."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_foo.py"
    test_file.write_text("broken\n", encoding="utf-8")
    a = coverage_agent(tmp_path, target_file_path=str(tmp_path / "foo.py"),
                       target_file_content="x", test_output_dir=str(test_dir))
    a.session.last_tests_passed = False
    a.session.best_green.snapshot = None
    a._maybe_restore_last_green(str(test_dir), str(tmp_path / "foo.py"))
    assert test_file.read_text(encoding="utf-8") == "broken\n"


def test_maybe_restore_last_green_reverts_red_file_to_snapshot(tmp_path: Path) -> None:
    """A red file with a snapshot is reverted: restored_from_green and
    last_tests_passed become True (the snapshot is green), and the
    snapshot's measurement becomes the current one."""
    snapshot_content = "def test_one():\n    assert True\n"
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir()
    target.write_text("def f(): pass\n", encoding="utf-8")
    test_file = test_dir / "test_foo.py"
    test_file.write_text("def broken_test(:\n    syntax err\n", encoding="utf-8")
    a = coverage_agent(tmp_path, target_file_path=str(target),
                       target_file_content="def f(): pass\n", test_output_dir=str(test_dir))
    a.session.last_tests_passed = False
    a.session.best_green.snapshot = {
        "content": snapshot_content,
        "coverage": 90.0, "test_count": 1, "time": 1.0,
        "loop_state": {"measured": True, "prev_coverage_pct": 90.0,
                       "prev_branch_coverage_pct": None},
    }
    a._maybe_restore_last_green(str(test_dir), str(target))
    assert test_file.read_text(encoding="utf-8") == snapshot_content
    assert a.session.best_green.restored_from_green is True
    assert a.session.last_tests_passed is True
    assert a.session.prev_coverage_pct == 90.0
    assert a.session.prev_branch_coverage_pct is None


def test_maybe_restore_last_green_same_content_no_write(tmp_path: Path) -> None:
    """A file that already matches the snapshot is not rewritten."""
    content = "def test_one(): assert True\n"
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir()
    target.write_text("def f(): pass\n", encoding="utf-8")
    test_file = test_dir / "test_foo.py"
    test_file.write_text(content, encoding="utf-8")
    a = coverage_agent(tmp_path, target_file_path=str(target),
                       target_file_content="def f(): pass\n", test_output_dir=str(test_dir))
    a.session.last_tests_passed = False
    a.session.best_green.snapshot = {
        "content": content, "coverage": 90.0, "test_count": 1, "time": 1.0,
    }
    a._maybe_restore_last_green(str(test_dir), str(target))
    assert a.session.best_green.restored_from_green is False


def test_maybe_restore_last_green_no_target_path_no_op(tmp_path: Path) -> None:
    """Without ``target_file_path`` the target's test file is unknown, so
    nothing is restored."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    a = coverage_agent(tmp_path, target_file_path=str(tmp_path / "foo.py"),
                       target_file_content="x", test_output_dir=str(test_dir))
    a.session.last_tests_passed = False
    a.session.best_green.snapshot = {
        "content": "ok\n", "coverage": 90.0, "test_count": 1, "time": 1.0,
    }
    a._maybe_restore_last_green(str(test_dir), None)
    assert a.session.best_green.restored_from_green is False


# _maybe_revert_on_regression: the counter and check that decide when to
# revert. The revert itself goes through _maybe_restore_last_green.


def _spy_restore(agent, monkeypatch):
    calls = []
    # A restore that wrote the snapshot, as the revert's note needs.
    monkeypatch.setattr(
        agent, "_maybe_restore_last_green",
        lambda d, t, *, force=False: calls.append((d, t, force)) or True,
    )
    return calls


def test_restore_count_reverts_on_regression(monkeypatch, tmp_path) -> None:
    a = _bare_agent(tmp_path)
    a.session.best_green.snapshot = {"content": "x", "coverage": 90.0, "test_count": 1, "time": 1.0}
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "restore_count", 1)
    calls = _spy_restore(a, monkeypatch)
    assert a._maybe_revert_on_regression("t.py", is_regression=True) is not None
    assert len(calls) == 1
    # The confirmed-regression path forces the restore so a stale coverage cache
    # cannot make the green short-circuit silently decline it.
    assert calls[0][2] is True
    assert a.session.coverage_regression_count == 0


def test_coverage_lateral_tolerated_resets_counter(monkeypatch, tmp_path) -> None:
    a = _bare_agent(tmp_path)
    a.session.best_green.snapshot = {"content": "x"}
    a.session.coverage_regression_count = 4
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "restore_count", 1)
    calls = _spy_restore(a, monkeypatch)
    assert a._maybe_revert_on_regression("t.py", is_regression=False) is None
    assert calls == []
    assert a.session.coverage_regression_count == 0


def test_restore_count_disabled(monkeypatch, tmp_path) -> None:
    a = _bare_agent(tmp_path)
    a.session.best_green.snapshot = {"content": "x"}
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "restore_count", -1)
    calls = _spy_restore(a, monkeypatch)
    assert a._maybe_revert_on_regression("t.py", is_regression=True) is None
    assert calls == []


def test_restore_count_2_reverts_on_second(monkeypatch, tmp_path) -> None:
    a = _bare_agent(tmp_path)
    a.session.best_green.snapshot = {"content": "x"}
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "restore_count", 2)
    calls = _spy_restore(a, monkeypatch)
    assert a._maybe_revert_on_regression("t.py", is_regression=True) is None
    assert calls == []
    assert a._maybe_revert_on_regression("t.py", is_regression=True) is not None
    assert len(calls) == 1


def test_coverage_no_snapshot_no_revert(monkeypatch, tmp_path) -> None:
    a = _bare_agent(tmp_path)
    a.session.best_green.snapshot = None
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "restore_count", 1)
    calls = _spy_restore(a, monkeypatch)
    assert a._maybe_revert_on_regression("t.py", is_regression=True) is None
    assert calls == []


# Suite-level timeout: revert to last green, name suspects, stop after N,
# but only with a best-green to return.

def _fake_backend_timeout(target_test):
    from testuary.languages.generic import GenericBackend
    real = GenericBackend("python", (".py",))
    return SimpleNamespace(
        find_target_test_file=lambda *a, **kw: target_test,
        run_test_files=lambda *a, **kw: run_result(ok=False, returncode=-1, timed_out=True),
        skeleton=real.skeleton,
        function_fingerprints=real.function_fingerprints,
    )


def test_run_tests_timeout_reverts_names_suspects_then_stops(tmp_path, monkeypatch):
    """A suite timeout reverts to the last in-time green, names the tests
    added since (suspects), and after suite_timeout_stop_after timeouts ends
    the phase with the best-green result."""
    a = _make_agent(tmp_path)
    target_test = Path(a.deps.test_output_dir) / "test_foo.py"
    snapshot = "def test_a(): assert True\n"
    slow = (
        "def test_a(): assert True\n\n"
        "def test_slow():\n    while True:\n        pass\n"
    )
    a.session.best_green.snapshot = {
        "content": snapshot, "coverage": 66.0, "test_count": 1, "time": 60.0,
    }
    a.session.best_coverage_pct = 66.0
    monkeypatch.setattr(tc.languages, "get_backend", lambda *a, **kw: _fake_backend_timeout(target_test))
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "suite_timeout_stop_after", 3)

    for n in (1, 2):
        target_test.write_text(slow, encoding="utf-8")
        a.session.tests_changed_since_run_tests = True
        out = a.run_tests()
        assert a.session.consecutive_timeouts == n
        assert "test_slow" in out                                 # suspect named
        assert target_test.read_text(encoding="utf-8") == snapshot   # reverted
        assert a.session.early_stop_result is None

    target_test.write_text(slow, encoding="utf-8")
    a.session.tests_changed_since_run_tests = True
    out = a.run_tests()
    assert a.session.consecutive_timeouts == 3
    assert a.session.early_stop_reason == "repeated_test_timeout"
    assert a.session.early_stop_result is not None
    assert "FRAMEWORK STOP" in out


def test_run_tests_timeout_stop_disabled_at_zero(tmp_path, monkeypatch):
    """``suite_timeout_stop_after <= 0`` disables the timeout stop: timeouts
    keep reverting to best-green but never end the phase."""
    a = _make_agent(tmp_path)
    target_test = Path(a.deps.test_output_dir) / "test_foo.py"
    snapshot = "def test_a(): assert True\n"
    slow = (
        "def test_a(): assert True\n\n"
        "def test_slow():\n    while True:\n        pass\n"
    )
    a.session.best_green.snapshot = {
        "content": snapshot, "coverage": 66.0, "test_count": 1, "time": 60.0,
    }
    a.session.best_coverage_pct = 66.0
    monkeypatch.setattr(tc.languages, "get_backend", lambda *a, **kw: _fake_backend_timeout(target_test))
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "suite_timeout_stop_after", -1)

    for n in (1, 2, 3, 4):
        target_test.write_text(slow, encoding="utf-8")
        a.session.tests_changed_since_run_tests = True
        a.run_tests()
        assert a.session.consecutive_timeouts == n
        assert a.session.early_stop_result is None
        assert target_test.read_text(encoding="utf-8") == snapshot


def test_run_tests_timeout_without_green_keeps_trying(tmp_path, monkeypatch):
    """Without a best-green to return, a timeout cannot end the phase."""
    a = _make_agent(tmp_path)
    target_test = Path(a.deps.test_output_dir) / "test_foo.py"
    a.session.best_green.snapshot = None
    monkeypatch.setattr(tc.languages, "get_backend", lambda *a, **kw: _fake_backend_timeout(target_test))
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "suite_timeout_stop_after", 2)
    for _ in range(3):                                # past the threshold
        a.session.tests_changed_since_run_tests = True
        a.run_tests()
    assert a.session.consecutive_timeouts == 3
    assert a.session.early_stop_result is None      # never green, never stops


def test_measure_coverage_timeout_surfaces_suspects(tmp_path, monkeypatch):
    """A timeout in measure_coverage's auto-prelude surfaces the suspects
    through the red-suite path instead of silently measuring the reverted
    suite."""
    a = _make_agent(tmp_path)
    target_test = Path(a.deps.test_output_dir) / "test_foo.py"
    snapshot = "def test_a(): assert True\n"
    slow = (
        "def test_a(): assert True\n\n"
        "def test_slow():\n    while True:\n        pass\n"
    )
    a.session.best_green.snapshot = {
        "content": snapshot, "coverage": 66.0, "test_count": 1, "time": 60.0,
    }
    a.session.best_coverage_pct = 66.0
    monkeypatch.setattr(tc.languages, "get_backend", lambda *a, **kw: _fake_backend_timeout(target_test))
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "suite_timeout_stop_after", 5)
    target_test.write_text(slow, encoding="utf-8")
    a.session.tests_changed_since_run_tests = True
    out = a.measure_coverage()
    assert "test_slow" in out             # suspects surfaced, not swallowed
    assert "Line coverage" not in out     # did not silently measure


def test_measure_coverage_instrumented_timeout_is_the_coverage_tools_refusal(tmp_path, monkeypatch):
    """A suite whose normal run passes in time but whose run under the
    coverage tool does not finish is the tool's refusal, not a suite
    timeout: no timeout counts, the file is not reverted to the snapshot,
    and the reply says the tool's run did not finish."""
    a = _make_agent(tmp_path)
    target_test = Path(a.deps.test_output_dir) / "test_foo.py"
    snapshot = "def test_a(): assert True\n"
    slow = (
        "def test_a(): assert True\n\n"
        "def test_heavy():\n    assert sum(range(3)) == 3\n"
    )
    a.session.best_green.snapshot = {
        "content": snapshot, "coverage": 66.0, "test_count": 1, "time": 1.0,
    }
    a.session.best_coverage_pct = 66.0
    real = GenericBackend("python", (".py",))

    def run_test_files(*a_, measure_coverage_of=None, **kw):
        if measure_coverage_of is not None:                  # instrumented run
            return run_result(ok=False, returncode=-1, timed_out=True)
        return run_result(collected=2, passed=2, output="2 passed")  # prelude: green

    fake = SimpleNamespace(
        find_target_test_file=lambda *a_, **kw: target_test,
        run_test_files=run_test_files,
        run_as_delivered=passing_delivery,
        skeleton=real.skeleton,
        function_fingerprints=real.function_fingerprints,
        test_metrics_observation=real.test_metrics_observation,
    )
    monkeypatch.setattr(tc.languages, "get_backend", lambda *a_, **kw: fake)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "suite_timeout_stop_after", 5)
    target_test.write_text(slow, encoding="utf-8")
    a.session.tests_changed_since_run_tests = True
    out = a.measure_coverage()
    assert a.session.consecutive_timeouts == 0
    assert "coverage tool's run of them did not finish within" in out
    assert target_test.read_text(encoding="utf-8") == slow
    assert a.refused_by() == "coverage"


def test_terminal_restore_treats_edited_after_green_as_unverified(tmp_path):
    """A file edited after its last green run is unverified at delivery:
    even against a scoreless snapshot, the live file does not pass the
    green gate, the same conjunct the mutation terminal applies."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_foo.py"
    test_file.write_text("def test_live(): assert True\n", encoding="utf-8")
    a = coverage_agent(tmp_path, target_file_path=str(tmp_path / "foo.py"),
                       target_file_content="x", test_output_dir=str(test_dir))
    a.session.last_tests_passed = True
    a.session.tests_changed_since_run_tests = True  # edited after green
    a.session.best_green.snapshot = {
        "content": "def test_snap(): assert True\n",
        "coverage": None, "branch_coverage": None,
        "test_count": 1, "time": 1.0,
    }
    a._maybe_restore_last_green(str(test_dir), str(tmp_path / "foo.py"))
    assert test_file.read_text(encoding="utf-8") == (
        "def test_snap(): assert True\n"
    )


def test_render_uncovered_branches_counts_java_untaken_branches(tmp_path: Path) -> None:
    """Java reports one entry per decision line with its untaken count: the
    header counts the untaken branches, not the entries, and never calls
    them arcs."""
    src = tmp_path / "M.java"
    src.write_text("class M {\n  int f(int x) {\n    if (x > 0) return 1;\n"
                   "    if (x < -5 || x == -2) return 2;\n    return 0;\n  }\n}\n",
                   encoding="utf-8")
    out = tc._render_uncovered_branches(
        str(src), ("3->1 of 2 branches untaken", "4->3 of 4 branches untaken"))
    assert out.startswith("=== UNCOVERED BRANCHES (4 untaken branches, 2 decision lines) ===")
    assert "arc" not in out


def test_render_uncovered_branches_names_the_function_an_exit_leaves(tmp_path: Path) -> None:
    """Python names the code object an exit arc leaves by its first line
    ("src->exit:N", the def or its first decorator): the render names that
    function, so an exit of a never-called stub shows as such. A plain
    "exit" without a line still renders."""
    src = tmp_path / "m.py"
    src.write_text("import functools\n\n@functools.lru_cache\ndef stub(x):\n"
                   "    if x:\n        return 1\n", encoding="utf-8")
    out = tc._render_uncovered_branches(str(src), ("5->exit:3",))
    assert ("<- branch to exit from def stub(x): (control leaves the function "
            "here) not taken") in out
    plain = tc._render_uncovered_branches(str(src), ("5->exit",))
    assert "branch to exit (control leaves the function / loop here)" in plain
