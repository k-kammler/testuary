"""``CoverageAgent._run_tests_uncached``, the test run behind ``run_tests``
and the ``measure_coverage`` prelude: strict green, collection errors, the
revert after repeated failures and the state each run updates.
"""
from __future__ import annotations

from pathlib import Path

from agent_fixtures import coverage_agent, install_backend, project_agent, run_result

from testuary.agent import coverage_agent as tc
from phase_values import set_phase_value
import pytest

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _make_agent(tmp_path: Path) -> "tc.CoverageAgent":
    # The method resolves the target's test file, so one is created first.
    return project_agent(tmp_path, test_source="def test_x():\n    assert True\n")


def test_run_tests_uncached_green_run_updates_state(monkeypatch, tmp_path) -> None:
    """A green run sets last_tests_passed=True, resets
    consecutive_failed_runs and increments run_tests_calls."""
    install_backend(monkeypatch, run_result(ok=True, returncode=0, output="3 passed in 0.1s"))
    a = _make_agent(tmp_path)
    a.session.best_green.consecutive_failed_runs = 5   # pre-existing streak
    passed, output = a._run_tests_uncached(log_prefix="run_tests")
    assert passed is True
    assert a.session.last_tests_passed is True
    assert a.session.best_green.consecutive_failed_runs == 0
    assert a.session.run_tests_calls == 1


def test_run_tests_uncached_red_run_increments_failures(monkeypatch, tmp_path) -> None:
    """A non-zero exit sets last_tests_passed=False and increments
    consecutive_failed_runs."""
    install_backend(monkeypatch, run_result(
        ok=False, returncode=1,
        output="FAILED tests/test_foo.py::test_x - AssertionError"))
    a = _make_agent(tmp_path)
    passed, output = a._run_tests_uncached(log_prefix="run_tests")
    assert passed is False
    assert a.session.last_tests_passed is False
    assert a.session.best_green.consecutive_failed_runs == 1


def test_run_tests_uncached_collection_error_returncode_2(monkeypatch, tmp_path) -> None:
    """Exit code 2 without passed or failed counts means pytest could not
    load the file. The output is annotated with a collection-error tool_msg."""
    bad_out = (
        "collected 0 items / 1 error\n"
        "================== ERRORS ==================\n"
        "ERROR collecting tests/test_foo.py\n"
        "E   ImportError: cannot import name 'foo'\n"
        "tests/test_foo.py:3: error\n"
    )
    install_backend(monkeypatch, run_result(ok=False, returncode=2, output=bad_out))
    a = _make_agent(tmp_path)
    passed, output = a._run_tests_uncached(log_prefix="run_tests")
    assert passed is False
    assert "Collection error" in output or "collection" in output.lower()


def test_run_tests_uncached_strict_green_unsound_warnings_red(monkeypatch, tmp_path) -> None:
    """A pytest exit of 0 with unsound-async warnings counts as red
    (strict-green override)."""
    out = (
        "5 passed in 0.4s\n"
        "============================== warnings summary ==============================\n"
        "tests/test_foo.py::test_x\n"
        "  /x:1: RuntimeWarning: coroutine 'g' was never awaited\n"
        "============== \n"
    )
    install_backend(monkeypatch, run_result(ok=True, returncode=0, output=out)._replace(
        unsound=(("tests/test_foo.py::test_x", "coroutine 'g' was never awaited"),)))
    a = _make_agent(tmp_path)
    passed, output = a._run_tests_uncached(log_prefix="run_tests")
    assert passed is False
    assert a.session.last_tests_passed is False
    assert "[STRICT GREEN]" in output


def test_a_run_without_a_failure_section_shows_the_reported_failures(monkeypatch, tmp_path) -> None:
    """A runner whose console has no failure section (JUnit's launcher)
    reaches the model as the report's failing tests, every one named."""
    from testuary.languages import TestFailure
    failures = (TestFailure("demo.CalcTest.a", "failed", "expected:<1> but was:<2>"),
                TestFailure("demo.CalcTest.b", "error", "boom"))
    install_backend(monkeypatch, run_result(ok=False, returncode=1, output=(
        "Test run finished after 9 ms\n[         2 tests failed          ]\n"))._replace(
        collected=3, passed=1, failed=1, errors=1, failures=failures))
    a = _make_agent(tmp_path)
    _passed, output = a._run_tests_uncached(log_prefix="run_tests")
    assert ("Failing tests (2):\n  - demo.CalcTest.a: expected:<1> but was:<2>\n"
            "  - demo.CalcTest.b (error): boom") in output


def test_strict_green_override_reads_failed_with_the_red_footer(monkeypatch, tmp_path) -> None:
    """A green run the strict-green rules turned red does not say "TESTS:
    PASSED" under the [STRICT GREEN] block, and gets the red footer."""
    out = (
        "5 passed in 0.4s\n"
        "============================== warnings summary ==============================\n"
        "tests/test_foo.py::test_x\n"
        "  /x:1: RuntimeWarning: coroutine 'g' was never awaited\n"
    )
    install_backend(monkeypatch, run_result(ok=True, returncode=0, output=out)._replace(
        collected=5, passed=5,
        unsound=(("tests/test_foo.py::test_x", "coroutine 'g' was never awaited"),)))
    a = _make_agent(tmp_path)
    _passed, output = a._run_tests_uncached(log_prefix="run_tests")
    assert "TESTS: FAILED | collected 5 | passed 5 | failed 0 | errors 0" in output
    assert "TESTS: PASSED" not in output
    assert ("-> tests are RED. No test failed, but the run counts as failed for "
            "the reason stated above.") in output


def test_run_tests_uncached_caches_result_in_state(monkeypatch, tmp_path) -> None:
    """After a run the formatted output is stored in
    ``last_run_tests_result`` and the cache flag is False."""
    install_backend(monkeypatch, run_result(ok=True, returncode=0, output="2 passed in 0.1s"))
    a = _make_agent(tmp_path)
    _, output = a._run_tests_uncached(log_prefix="run_tests")
    assert a.session.last_run_tests_result == output
    assert a.session.tests_changed_since_run_tests is False


def test_run_tests_uncached_timeout_returns_clear_payload(monkeypatch, tmp_path) -> None:
    """A timed-out run gives the pytest_timeout tool_msg, sets
    last_tests_passed=False and caches the output."""
    install_backend(monkeypatch, run_result(timed_out=True, ok=False))
    a = _make_agent(tmp_path)
    passed, output = a._run_tests_uncached(log_prefix="run_tests")
    assert passed is False
    assert a.session.last_tests_passed is False


def test_run_tests_uncached_no_test_file_returns_directive(monkeypatch, tmp_path) -> None:
    """Without the target's test file on disk the method returns a 'no test
    file' directive without running a subprocess."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "foo.py"
    src.write_text("def f(): return 1\n", encoding="utf-8")
    empty_test_dir = tmp_path / "no_tests"
    empty_test_dir.mkdir()
    a = coverage_agent(tmp_path, test_output_dir=str(empty_test_dir))
    runs = {"n": 0}
    install_backend(monkeypatch, run_result(), counter=runs)
    passed, output = a._run_tests_uncached(log_prefix="run_tests")
    assert runs["n"] == 0  # find_target_test_file None -> no run
    assert passed is False
    assert a.session.last_tests_passed is False


def test_run_tests_uncached_unsupported_language_returns_directive(monkeypatch, tmp_path) -> None:
    """A language other than Python or Java gets a directive without a
    subprocess run."""
    # The real get_backend returns None for go.
    a = coverage_agent(tmp_path, target_file_path=str(tmp_path / "foo.go"),
                       target_file_content="package main\n",
                       test_output_dir=str(tmp_path / "tests"), language="go")
    passed, output = a._run_tests_uncached(log_prefix="run_tests")
    assert passed is False
    assert a.session.last_tests_passed is False


def test_run_tests_uncached_auto_revert_after_threshold(monkeypatch, tmp_path) -> None:
    """After ``max_failed_test_attempts`` consecutive failures with a
    best-green snapshot available, the file is reverted to the snapshot and
    ``restored_from_green`` is True."""
    install_backend(monkeypatch, run_result(ok=False, returncode=1, output="FAILED ..."))
    a = _make_agent(tmp_path)
    test_path = Path(a.deps.test_output_dir) / "test_foo.py"
    good_content = "def test_one():\n    assert True\n"
    a.session.best_green.snapshot = {
        "content": good_content,
        "mutation": 80.0, "coverage": 90.0, "test_count": 1, "time": 1.0,
    }
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_failed_test_attempts", 1)
    test_path.write_text("def test_broken(:\n    syntax error\n", encoding="utf-8")
    a._run_tests_uncached(log_prefix="run_tests")
    assert a.session.best_green.restored_from_green is True
    assert test_path.read_text(encoding="utf-8") == good_content


def test_auto_revert_return_value_matches_in_state(monkeypatch, tmp_path) -> None:
    """After a revert the returned ``(passed, output)`` agrees with
    ``last_tests_passed``, both green."""
    install_backend(monkeypatch, run_result(ok=False, returncode=1, output="FAILED ..."))
    a = _make_agent(tmp_path)
    good_content = "def test_one():\n    assert True\n"
    a.session.best_green.snapshot = {
        "content": good_content,
        "mutation": 80.0, "coverage": 90.0, "test_count": 1, "time": 1.0,
    }
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_failed_test_attempts", 1)
    passed, _output = a._run_tests_uncached(log_prefix="run_tests")
    assert a.session.best_green.restored_from_green is True
    assert a.session.last_tests_passed is True
    assert passed is True


def test_auto_revert_leaves_the_snapshot_run_state(monkeypatch, tmp_path) -> None:
    """After auto-revert the loop is in the restored snapshot's state: the
    snapshot is green by construction and unchanged since its run, so the
    failed candidate's run is not replayed for it and no run of the
    restored bytes is needed."""
    install_backend(monkeypatch, run_result(ok=False, returncode=1, output="FAILED ..."))
    a = _make_agent(tmp_path)
    good_content = "def test_one():\n    assert True\n"
    a.session.best_green.snapshot = {
        "content": good_content,
        "mutation": 80.0, "coverage": 90.0, "test_count": 1, "time": 1.0,
    }
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_failed_test_attempts", 1)
    a._run_tests_uncached(log_prefix="run_tests")
    assert a.session.best_green.restored_from_green is True
    assert a.session.last_tests_passed is True
    assert a.session.tests_changed_since_run_tests is False


def test_non_restored_failure_leaves_cache_valid(monkeypatch, tmp_path) -> None:
    """A failing run without restoration (no snapshot) leaves
    ``tests_changed_since_run_tests = False``, so later run_tests calls hit
    the cache."""
    install_backend(monkeypatch, run_result(ok=False, returncode=1, output="FAILED ..."))
    a = _make_agent(tmp_path)
    a.session.best_green.snapshot = None
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_failed_test_attempts", 1)
    a._run_tests_uncached(log_prefix="run_tests")
    assert a.session.best_green.restored_from_green is False
    assert a.session.tests_changed_since_run_tests is False


def test_run_tests_uncached_red_limit_zero_is_unlimited(monkeypatch, tmp_path) -> None:
    """``max_failed_test_attempts <= 0`` is an unlimited red repair limit:
    the test file is never reverted, so the model can keep repairing a broken
    edit within the request limit."""
    install_backend(monkeypatch, run_result(ok=False, returncode=1, output="FAILED ..."))
    a = _make_agent(tmp_path)
    test_path = Path(a.deps.test_output_dir) / "test_foo.py"
    a.session.best_green.snapshot = {
        "content": "def test_one():\n    assert True\n",
        "mutation": 80.0, "coverage": 90.0, "test_count": 1, "time": 1.0,
    }
    a.session.best_green.consecutive_failed_runs = 50  # far past any finite limit
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_failed_test_attempts", 0)
    broken = "def test_broken(:\n    syntax error\n"
    test_path.write_text(broken, encoding="utf-8")
    a._run_tests_uncached(log_prefix="run_tests")
    assert a.session.best_green.restored_from_green is False
    assert test_path.read_text(encoding="utf-8") == broken  # never reverted
