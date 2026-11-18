"""A suite timeout names the test still running (``TestRunResult.active_at_end``,
from the backend's per-test timeout journal). Like a red run, a named hang is
fixed by editing: the file is not reverted and the model is told the test.
Without a name the loop reverts to the best-green snapshot and names the tests
added since, which are runnable definitions, never helpers nested in a test.
"""
from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

from agent_fixtures import SOURCE, mutation_agent, project_agent

import pytest
from langadapter import Outcome, TestOperations

from testuary.agent import _helpers as ah
from testuary import languages as _languages
from testuary import normal_run
from testuary.languages.generic import GenericBackend
from testuary.agent import coverage_agent as tc
from testuary.agent import mutation_agent as tm

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

GREEN = "def test_a():\n    assert True\n"
HANGING = (GREEN + "\n\ndef test_b():\n    assert 1\n"
           "\n\ndef test_slow():\n    def wait():\n        while True:\n            pass\n"
           "    wait()\n")


@pytest.mark.slow
@pytest.mark.parametrize("through", ["normal run", "run_test_files"])
def test_a_real_run_names_the_test_a_timeout_cut_off(tmp_path, through):
    """Python's normal run, the run as delivered, and run_test_files, the
    coverage tool's, both name the test a timeout cut off. Each run waits
    out its 20 s timeout, so the test is slow."""
    suite = tmp_path / "test_hang.py"
    suite.write_text(
        "import time\n\n"
        "def test_fast():\n    assert True\n\n"
        "def test_slow():\n    time.sleep(120)\n",
        encoding="utf-8")
    backend = _languages.get_backend("python")
    assert backend.offers_timeout_journal() is True
    if through == "normal run":
        res = normal_run.run(backend, [str(suite)], str(tmp_path), timeout=20,
                             timeout_journal=True)
    else:
        res = backend.run_test_files([str(suite)], str(tmp_path), timeout=20,
                                     timeout_journal=True)
    assert res.timed_out is True
    assert res.active_at_end == "test_slow"


def _timed_out_while(monkeypatch, name):
    """LangAdapter's suite runs come back as a timeout would end them
    while the test ``name`` ran: each real run of a quick suite is
    reported timed out, with that test as the journal's."""
    def ending(real):
        def run(self, request, **options):
            done = real(self, request, **options)
            test = next(case.test for case in done.tests if case.test.qualname == name)
            return dataclasses.replace(done, outcome=Outcome.TIMED_OUT, active_at_end=test)
        return run

    for method in ("run_suite", "run_as_delivered"):
        monkeypatch.setattr(TestOperations, method, ending(getattr(TestOperations, method)))


@pytest.mark.parametrize("through", ["normal run", "run_test_files", "failing listing"])
def test_each_entry_point_names_the_test_the_journal_reports(tmp_path, monkeypatch, through):
    """The normal run, run_test_files and the failing-test listing each name
    the test the timeout journal reports, as their real runs in the slow
    step do, here for a quick run LangAdapter reports as ended by the
    timeout while test_slow ran."""
    suite = tmp_path / "test_hang.py"
    suite.write_text("def test_fast():\n    assert True\n\n\n"
                     "def test_slow():\n    assert True\n", encoding="utf-8")
    _timed_out_while(monkeypatch, "test_slow")
    backend = _languages.get_backend("python")
    if through == "failing listing":
        report = {}
        assert backend.failing_test_ids([str(suite)], str(tmp_path), timeout=60,
                                        report=report) is None
        assert report == {"outcome": "timed_out", "active_at_end": "test_hang.test_slow"}
        return
    if through == "normal run":
        res = normal_run.run(backend, [str(suite)], str(tmp_path), timeout=60,
                             timeout_journal=True)
    else:
        res = backend.run_test_files([str(suite)], str(tmp_path), timeout=60,
                                     timeout_journal=True)
    assert res.timed_out is True
    assert res.active_at_end == "test_slow"


def test_suspects_leave_out_helpers_nested_in_a_test():
    backend = GenericBackend("python", (".py",))
    assert ah.tests_added_since(backend, HANGING, GREEN) == ["test_b", "test_slow"]


def _timeout_backend(target_test: Path, calls: list):
    real = GenericBackend("python", (".py",))

    def run_test_files(*a, **kw):
        calls.append(kw)
        return _languages.TestRunResult(
            ok=False, returncode=-1, timed_out=True, elapsed_s=None,
            collected=0, passed=0, failed=0, errors=0, output="",
            line_coverage=None, branch_coverage=None,
            active_at_end="test_slow" if kw.get("timeout_journal") else None)

    return SimpleNamespace(
        find_target_test_file=lambda *a, **kw: target_test,
        run_test_files=run_test_files,
        offers_timeout_journal=lambda source_dir=None: True,
        skeleton=real.skeleton,
        function_fingerprints=real.function_fingerprints,
    )


def test_coverage_names_the_hanging_test_and_keeps_the_file(tmp_path, monkeypatch):
    agent = project_agent(tmp_path, source="def f(x):\n    return x\n")
    target_test = tmp_path / "tests" / "test_foo.py"
    s = agent.session
    s.has_read = True
    s.initial_measured = True  # past the start's measurement
    s.writes_done = 1
    s.best_green.snapshot = {"content": GREEN, "coverage": 50.0, "test_count": 1,
                             "loop_state": {"measured": False}}
    calls = []
    monkeypatch.setattr(tc.languages, "get_backend",
                        lambda *a, **kw: _timeout_backend(target_test, calls))
    target_test.write_text(HANGING, encoding="utf-8")
    s.tests_changed_since_run_tests = True
    out = agent.run_tests()
    assert calls and calls[0].get("timeout_journal") is True
    assert "test_slow" in out and "unchanged" in out
    assert target_test.read_text(encoding="utf-8") == HANGING   # test_b kept
    assert s.last_tests_passed is False
    assert s.consecutive_timeouts == 1
    assert s.best_green.consecutive_failed_runs == 1
    assert agent.resolve_agent_mode() == "needs_editing"


def test_mutation_names_the_hanging_test_and_keeps_the_file(tmp_path, monkeypatch):
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    source = tmp_path / "src" / "foo.py"
    source.parent.mkdir()
    source.write_text(SOURCE, encoding="utf-8")
    agent = mutation_agent(tmp_path, test_output_dir=str(test_dir), mutation_test_file_name="test_foo.py")
    s = agent.session
    s.has_read = True
    s.initial_measured = True  # past the start's measurement
    s.writes_done = 1
    s.best_green.snapshot = {"content": GREEN, "mutation": 50.0, "test_count": 1,
                             "loop_state": {"measured": False}}
    calls = []
    monkeypatch.setattr(tm.languages, "get_backend",
                        lambda *a, **kw: _timeout_backend(agent.target_path(), calls))
    agent.target_path().write_text(HANGING, encoding="utf-8")
    s.tests_changed_since_run_tests = True
    out = agent.run_tests()
    assert calls and calls[0].get("timeout_journal") is True
    assert "test_slow" in out and "unchanged" in out
    assert agent.target_path().read_text(encoding="utf-8") == HANGING
    assert s.last_tests_passed is False
    assert s.best_green.consecutive_failed_runs == 1
    assert agent.resolve_agent_mode() == "needs_editing"
