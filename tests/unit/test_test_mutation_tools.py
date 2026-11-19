"""Mutation phase tools with a faked backend and mutation engine.

Per-run state lives on a ``MutationAgent`` instance (``session`` and
``deps``), so each test builds one with synthetic deps and a fresh
``TestMutationSession``, fakes ``run_test_files`` and the engine, calls the
tool and checks the returned string and the session. The cases cover
caching, output compaction, state updates and the switch to the full mutant
catalogue.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from agent_fixtures import logs_in, mutation_agent, passing_delivery, run_result, SOURCE
from agent_prompts import c_prompts
from langadapter.runs import RunDiagnostic

from offered_tools import offered_tools
from testuary.agent import _helpers as agent_helpers
from testuary.agent import mutation_agent as tm
from phase_values import set_phase_value
from session_fields import written_fields

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

#: The fake engine the tests install: ``(language, config) -> engine``,
#: an object with the engine's ``run``, ``full_operators`` and ``name``.
_ENGINE_FACTORY = None


def _set_engine(monkeypatch, factory) -> None:
    monkeypatch.setattr(sys.modules[__name__], "_ENGINE_FACTORY", factory)


class _FakeRunner:
    """The agent's mutation runner over a fake engine: a run of its own
    session is the engine's ``run``, and the catalogue is the engine's."""

    def __init__(self, engine, log=None) -> None:
        self.engine = engine
        self.name = getattr(engine, "name", "fake_engine")
        self.log = SimpleNamespace(target=log)

    def full_operators(self):
        return self.engine.full_operators()

    def configured_operators(self):
        return None

    def run_once(self, *, workers, **run):
        return self.engine.run(workers=workers, **run)


def _runner(language, config, log):
    if _ENGINE_FACTORY is None:
        raise AssertionError("the test installed no mutation engine")
    return _FakeRunner(_ENGINE_FACTORY(language, config), log)


def _trr(rc: int = 0, out: str = "", *, timed_out: bool = False):
    """A run_result in which one test passed when ``rc`` is 0 and one failed
    otherwise. A timed-out run has the exit code -1."""
    return run_result(ok=(rc == 0 and not timed_out), returncode=(-1 if timed_out else rc),
                      timed_out=timed_out, passed=(1 if rc == 0 else 0),
                      failed=(0 if rc == 0 else 1), output=out)


def _install_backend(monkeypatch, result_or_fn, *, under_coverage=None):
    """Route the mutation phase's test runs (run_tests and the
    measure_mutation's normal run) through a fake backend. ``result_or_fn`` is
    a TestRunResult or a callable ``(test_paths) -> TestRunResult``.
    ``under_coverage``, when given, is the result of a run under the
    coverage tool."""
    def _run(test_paths, source_dir, *, timeout,
             measure_coverage_of=None, instrument_target=None,
             fail_fast=False, tb="short", record_leaks=False):
        if under_coverage is not None and measure_coverage_of is not None:
            return under_coverage
        if callable(result_or_fn):
            return result_or_fn(list(test_paths))
        return result_or_fn
    monkeypatch.setattr(
        tm.languages, "get_backend",
        lambda lang: SimpleNamespace(
            run_test_files=_run,
            run_as_delivered=passing_delivery,
            shared_setup_files=lambda test_dir: (),
            file_extensions=(".py",), test_file_extensions=lambda: (".py",),
            mutation_runner=lambda config, source_dir=None, log=None,
            _l=lang: _runner(_l, config, log),
        ),
    )


def _make_agent(tmp_path: Path, *, mutation_session=None) -> "tm.MutationAgent":
    """A MutationAgent over src/foo.py on disk and the directory tests,
    with a coverage baseline of 85%."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    source = tmp_path / "src" / "foo.py"
    source.parent.mkdir(parents=True, exist_ok=True)
    source.write_text(SOURCE, encoding="utf-8")
    return mutation_agent(tmp_path, mutation_session=mutation_session,
                          test_output_dir=str(test_dir), coverage_baseline_pct=85.0)


# read_source: the cache for an immediate re-read.

def test_read_source_caches_immediate_re_read(tmp_path: Path) -> None:
    """A second consecutive read_source with no other tool call between
    returns the cached header instead of a fresh render, as the other
    loops' read tools do."""
    agent = _make_agent(tmp_path)
    first = agent.read_source()
    assert "def f()" in first  # full source rendered on the first read
    second = agent.read_source()
    assert "Cached read #2" in second
    assert "def f()" not in second  # cached header only, no re-render


# run_tests with a faked backend.

def test_run_tests_passing_run_updates_state(monkeypatch, tmp_path: Path) -> None:
    """A green run sets last_tests_passed, shows the pass summary,
    increments both call counters and clears the cache flag."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.42s"))
    agent = _make_agent(tmp_path)
    out = agent.run_tests()
    assert "PASSED" in out
    assert "3 passed" in out
    assert agent.session.last_tests_passed is True
    assert agent.session.run_tests_calls == 1
    assert agent.session.session_tool_calls == 1
    # A re-call on identical bytes is then answered from the cache.
    assert agent.session.tests_changed_since_run_tests is False


def test_synthesized_result_records_iteration_count(tmp_path: Path) -> None:
    """The synthesized end-of-run result carries agent_iterations from the
    session's tool-call count, so tests.csv records the mutation phase's
    request count, not zero."""
    agent = _make_agent(tmp_path)
    agent.session.session_tool_calls = 7
    agent.session.best_mutation_score = 88.3
    result = agent._synthesize_mutation_result("mutation_goal_met")
    assert result.agent_iterations == 7
    assert result.mutation_score == 88.3
    assert result.status == "success"


def test_invalid_mutation_edit_preserves_working_bytes_and_measurement(
    tmp_path: Path,
) -> None:
    agent = _make_agent(tmp_path)
    path = Path(agent.deps.test_output_dir) / agent.deps.mutation_test_file_name
    original = "import src.foo as foo\n\ndef test_f():\n    assert foo.f() == 1\n"
    path.write_text(original, encoding="utf-8")
    agent.session.prev_mutation_score = 75.0
    agent.session.measured_suite_hash = "existing-measurement"
    out = agent.edit_file(
        "    assert foo.f() == 1",
        "    assert open(foo.__file__).read()",
    )
    assert "REJECTED" in out and "measurements are unchanged" in out
    assert path.read_text(encoding="utf-8") == original
    assert agent.session.prev_mutation_score == 75.0
    assert agent.session.measured_suite_hash == "existing-measurement"
    assert agent.session.writes_done == 0


def test_run_tests_failing_run_records_failure(monkeypatch, tmp_path: Path) -> None:
    """A non-zero exit clears ``last_tests_passed`` and shows the TESTS:
    FAILED line. The validator refuses a result while last_tests_passed is
    False."""
    _install_backend(monkeypatch, _trr(
        1, "FAILED tests/test_x.py::test_a - AssertionError",
    ))
    agent = _make_agent(tmp_path)
    out = agent.run_tests()
    assert "TESTS: FAILED" in out
    assert agent.session.last_tests_passed is False


def _passing_result() -> "tm.MutationTestResult":
    """A MutationTestResult with a real explanation, so it passes the content
    check and a validator test reaches the green or goal branch."""
    return tm.MutationTestResult(
        mutation_test_file_name="test_src_mutation.py",
        explanation="Targeted the surviving arithmetic and boundary mutants",
        targeted_mutants=["src.py:3:ReplaceBinaryOperator_Add_Sub"],
        mutation_score=0.0,
        status="success",
    )


def test_validate_output_rejects_red_suite(monkeypatch, tmp_path: Path) -> None:
    """The output validator refuses a MutationTestResult while the suite is
    red, with the validator_not_green message from the prompt JSON."""
    agent = _make_agent(tmp_path)
    agent.session.early_stop_result = None
    agent.session.best_mutation_score = None
    agent.session.last_tests_passed = False
    out = agent.validate_output(None, _passing_result())
    assert out is not None
    assert "MutationTestResult" in out
    assert "run_tests" in out


def test_validate_output_rejects_below_goal(monkeypatch, tmp_path: Path) -> None:
    """Below the mutation goal with a green suite and patience left, the
    validator rejects with validator_stagnation_not_exhausted, and the
    message names all three permitted exits."""
    agent = _make_agent(tmp_path)
    agent.session.early_stop_result = None
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_mutation = False
    agent.session.best_mutation_score = 50.0
    agent.session.mutation_measurements = 1
    agent.session.measures_at_last_submission = 1
    out = agent.validate_output(None, _passing_result())
    assert out is not None
    assert "cannot return MutationTestResult yet" in out
    assert "patience threshold" in out
    assert "request budget" in out


def test_validator_early_stop_defers_while_switch_pending(tmp_path: Path) -> None:
    """An exhausted patience streak does not open the voluntary-return gate
    while a selective-to-full switch is pending. The plateau switches
    catalogues on the next measurement instead of ending the phase on the
    selective score."""
    agent = _make_agent(tmp_path)
    s = agent.session
    s.last_tests_passed = True
    s.tests_changed_since_mutation = False
    s.best_mutation_score = 70.0
    s.mutation_measurements = 3
    s.measures_at_last_submission = 3
    s.measurements_without_improvement = 99
    s.selective_switch_possible = True
    assert agent.validate_output(None, _passing_result()) is not None
    s.selective_switch_possible = False
    assert agent.validate_output(None, _passing_result()) is None


def test_run_tests_timeout_records_timeout_payload(
    monkeypatch, tmp_path: Path,
) -> None:
    """A timed-out run returns a timeout message from ``tool_messages`` and
    clears the cache flag, so the next call does not retry blindly."""
    _install_backend(monkeypatch, _trr(timed_out=True))
    agent = _make_agent(tmp_path)
    out = agent.run_tests()
    assert agent.session.last_tests_passed is False
    assert agent.session.tests_changed_since_run_tests is False
    # The timeout handler's message is the cached result.
    assert out == agent.session.last_run_tests_result


def test_run_tests_cached_re_call_short_circuits(
    monkeypatch, tmp_path: Path,
) -> None:
    """Of two calls without an edit between them, the second returns the
    cached rejection without a second run."""
    call_count = {"n": 0}

    def responder(test_paths):
        call_count["n"] += 1
        return _trr(0, "2 passed in 0.1s")

    _install_backend(monkeypatch, responder)
    agent = _make_agent(tmp_path)
    agent.run_tests()  # primes the cache
    second = agent.run_tests()
    # Only the first call ran the tests.
    assert call_count["n"] == 1
    # The cached rejection contains the prior result.
    assert "2 passed" in second


def test_run_tests_repeated_no_change_calls_escalate(
    monkeypatch, tmp_path: Path,
) -> None:
    """A repeated run_tests without a change returns the cached result with
    an escalating header (REPEATED CALL at the third identical call, STUCK at
    the sixth), as in the other loops."""
    _install_backend(monkeypatch, _trr(0, "2 passed in 0.1s"))
    agent = _make_agent(tmp_path)
    agent.run_tests()  # real run, primes the cache
    outs = [agent.run_tests() for _ in range(6)]  # cached re-calls
    assert "2 passed" in outs[0]       # served from the cache
    assert "REPEATED CALL" in outs[2]  # third identical call escalates
    assert "STUCK" in outs[5]          # sixth escalates further


def test_run_tests_compacts_huge_output(monkeypatch, tmp_path: Path) -> None:
    """A test output of 50K characters is compacted by the shared
    ``_compact_test_output`` helper to the failures section and short
    summary, so a verbose failure does not fill the model's context."""
    huge = "x" * 50_000
    _install_backend(monkeypatch, _trr(1, huge))
    agent = _make_agent(tmp_path)
    out = agent.run_tests()
    assert "TESTS: FAILED" in out
    assert len(out) < 50_000


def test_run_tests_counts_real_executions_vs_invocations(
    monkeypatch, tmp_path: Path,
) -> None:
    """session_tool_calls counts every run_tests call, since a cached call
    is still a model turn. run_tests_calls counts only real runs, so the two
    cached re-runs do not advance it."""
    _install_backend(monkeypatch, _trr(0, "1 passed in 0.1s"))
    agent = _make_agent(tmp_path)
    agent.run_tests()
    agent.run_tests()
    agent.run_tests()
    assert agent.session.run_tests_calls == 1
    assert agent.session.session_tool_calls == 3


def test_run_tests_strict_green_reclassifies_unsound_as_red(
    monkeypatch, tmp_path: Path,
) -> None:
    """A green run that emits unsound-async warnings is reclassified as red
    with the offending tests named, as in the other loops."""
    agent = _make_agent(tmp_path)
    monkeypatch.setattr(
        tm.languages, "get_backend",
        lambda lang: SimpleNamespace(
            run_test_files=lambda *a, **kw: _trr(0, "5 passed in 0.2s")._replace(
                unsound=(("test_async_thing", "coroutine never awaited"),)),
        ),
    )
    out = agent.run_tests()
    assert agent.session.last_tests_passed is False
    assert "STRICT GREEN" in out
    assert "test_async_thing" in out
    # The summary under the block reads the outcome, not the exit code.
    assert "TESTS: FAILED | collected 1 | passed 1" in out
    assert "TESTS: PASSED" not in out


def test_run_tests_shows_the_reported_failures(monkeypatch, tmp_path: Path) -> None:
    """The mutation phase shows a JUnit run's failing tests from the report,
    as the coverage phase does."""
    from testuary.languages import TestFailure
    agent = _make_agent(tmp_path)
    failures = (TestFailure("demo.CalcTest.a", "failed", "expected:<1> but was:<2>"),
                TestFailure("demo.CalcTest.b", "failed", "boom"))
    _install_backend(monkeypatch, _trr(1, "[         2 tests failed          ]\n")._replace(
        failed=2, failures=failures))
    out = agent.run_tests()
    assert ("Failing tests (2):\n  - demo.CalcTest.a: expected:<1> but was:<2>\n"
            "  - demo.CalcTest.b: boom") in out


def test_measure_mutation_refuses_a_normal_run_with_unsound_warnings(
    monkeypatch, tmp_path: Path,
) -> None:
    """The normal run applies run_tests' strict-green rule. A green suite that
    emitted unsound asynchronous warnings is refused before any mutant runs,
    with the offending test named."""
    agent = _make_agent(tmp_path)
    agent.target_path().write_text("def test_async_thing():\n    pass\n", encoding="utf-8")
    engine_calls = []
    monkeypatch.setattr(
        tm.languages, "get_backend",
        lambda lang: SimpleNamespace(
            run_test_files=lambda *a, **kw: _trr(0, "5 passed in 0.2s")._replace(
                unsound=(("test_async_thing", "coroutine never awaited"),)),
            run_as_delivered=passing_delivery,
            shared_setup_files=lambda test_dir: (),
            file_extensions=(".py",), test_file_extensions=lambda: (".py",),
            mutation_runner=lambda *a, **kw: engine_calls.append(1),
        ),
    )
    agent.session.has_read = True
    agent.session.initial_measured = True  # past the start's measurement
    agent.session.writes_done = 1
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_run_tests = False
    agent.session.tests_changed_since_mutation = True
    out = agent.measure_mutation()
    assert "Tests are not green; mutation cannot be" in out
    assert "STRICT GREEN" in out and "test_async_thing" in out
    assert engine_calls == []
    # As after a red run_tests: the next turn offers the edit tool, not the
    # measurement.
    assert agent.session.last_tests_passed is False
    offered = [tool.__name__ for tool in offered_tools(agent)]
    assert "edit_file" in offered and "measure_mutation" not in offered


def test_a_red_normal_run_returns_the_loop_to_editing(monkeypatch, tmp_path: Path) -> None:
    """A normal run with failing tests leaves the suite red: the next turn
    offers the edit tool, not the measurement."""
    agent = _make_agent(tmp_path)
    agent.target_path().write_text("def test_x():\n    assert False\n", encoding="utf-8")
    _install_backend(monkeypatch, _trr(1, "FAILED test_foo_mutation.py::test_x\n1 failed in 0.1s\n"))
    agent.session.has_read = True
    agent.session.initial_measured = True  # past the start's measurement
    agent.session.writes_done = 1
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_run_tests = False
    agent.session.tests_changed_since_mutation = True
    out = agent.measure_mutation()
    assert "Tests are not green; mutation cannot be" in out
    assert agent.session.last_tests_passed is False
    offered = [tool.__name__ for tool in offered_tools(agent)]
    assert "edit_file" in offered and "measure_mutation" not in offered


def test_run_tests_strict_green_setting_off_keeps_green(
    monkeypatch, tmp_path: Path,
) -> None:
    """With reject_unsound_async_green disabled, an exit-0 run with unsound
    warnings stays green."""
    agent = _make_agent(tmp_path)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "reject_unsound_async_green", False)
    monkeypatch.setattr(
        tm.languages, "get_backend",
        lambda lang: SimpleNamespace(
            run_test_files=lambda *a, **kw: _trr(0, "5 passed in 0.2s"),
            run_as_delivered=passing_delivery,
        ),
    )
    agent.run_tests()
    assert agent.session.last_tests_passed is True


def test_edit_rejects_history_marker(tmp_path: Path) -> None:
    """old_text or new_text echoing the history sentinel is rejected. The
    file exists because the gate reports a missing file first."""
    agent = _make_agent(tmp_path)
    agent.target_path().write_text("def test_a():\n    assert True\n", encoding="utf-8")
    out = agent.edit_file("assert True <<<TESTUARY_ELIDED>>>", "bar")
    assert "TESTUARY_ELIDED" in out or "history-trim" in out


def test_edit_loose_match_tolerates_indentation(tmp_path: Path) -> None:
    """old_text at a different uniform indent than the file still matches
    through the whitespace-loose fallback, as in the other editing loops."""
    agent = _make_agent(tmp_path)
    agent.target_path().write_text(
        "def test_beta():\n    if True:\n        x = 1\n        assert x == 1\n",
        encoding="utf-8",
    )
    # old_text uses a 4-space indent, the file 8. Only the per-line lstrip
    # match finds it.
    agent.edit_file(
        "    x = 1\n    assert x == 1",
        "    x = 2\n    assert x == 2",
    )
    body = agent.target_path().read_text(encoding="utf-8")
    assert "x = 2" in body and "assert x == 2" in body
    assert "x = 1" not in body


def test_read_source_is_line_numbered_and_pageable(tmp_path: Path) -> None:
    """read_source renders a line-numbered body through render_read and
    honors start_line and end_line."""
    agent = _make_agent(tmp_path)  # target content = "def f(): return 1\n"
    out = agent.read_source()
    assert "1\tdef f(): return 1" in out
    assert "of 1]" in out  # the trailing newline ends line 1
    paged = agent.read_source(start_line=1, end_line=1)
    assert "1\tdef f(): return 1" in paged
    assert "2\t" not in paged  # only line 1 in this page


# edit_file: the shared CodingAgent edit gate.

def test_edit_unique_match_updates_file_and_counter(
    tmp_path: Path,
) -> None:
    """A unique ``old_text`` match in an existing file is replaced once on
    disk, increments writes_done and marks the tests changed."""
    agent = _make_agent(tmp_path)
    path = agent.target_path()
    path.write_text("def test_a():\n    assert 1 == 1\n", encoding="utf-8")
    agent.edit_file("1 == 1", "2 == 2")
    assert "2 == 2" in path.read_text(encoding="utf-8")
    assert agent.session.writes_done == 1
    assert agent.session.tests_changed_since_mutation is True


def test_edit_missing_file_returns_error(tmp_path: Path) -> None:
    """Without a mutation file on disk the error names the missing baseline
    and no counter advances."""
    agent = _make_agent(tmp_path)
    out = agent.edit_file("foo", "bar")
    assert agent.session.writes_done == 0
    assert "No working file" in out


def test_edit_zero_matches_returns_not_found(tmp_path: Path) -> None:
    """An absent ``old_text`` gives the not-found message without a write or
    a counter increment. Every substitution resolves and the stale-view
    recovery fires."""
    agent = _make_agent(tmp_path)
    path = agent.target_path()
    path.write_text("def test_a(): pass\n", encoding="utf-8")
    out = agent.edit_file("nonexistent_block", "new")
    assert agent.session.writes_done == 0
    assert path.read_text(encoding="utf-8") == "def test_a(): pass\n"
    assert "${" not in out
    assert "STALE" in out and "from memory" in out


def test_edit_ambiguous_match_returns_count(tmp_path: Path) -> None:
    """An ``old_text`` that appears more than once is rejected as ambiguous,
    and the file stays untouched."""
    agent = _make_agent(tmp_path)
    path = agent.target_path()
    path.write_text(
        "def test_a(): assert True\n"
        "def test_b(): assert True\n",
        encoding="utf-8",
    )
    out = agent.edit_file("assert True", "assert 1 == 1")
    assert agent.session.writes_done == 0
    assert "2 times" in out or "appears 2" in out
    assert "assert True" in path.read_text(encoding="utf-8")  # untouched


def test_edit_syntax_error_after_edit_rejected(tmp_path: Path) -> None:
    """A replacement that produces a SyntaxError is refused. The file keeps
    its previous valid state and no counter advances."""
    agent = _make_agent(tmp_path)
    path = agent.target_path()
    original = "def test_a():\n    assert 1 == 1\n"
    path.write_text(original, encoding="utf-8")
    out = agent.edit_file("1 == 1", "1 ==")   # breaks syntax
    assert agent.session.writes_done == 0
    assert path.read_text(encoding="utf-8") == original
    assert "SyntaxError" in out or "syntax" in out.lower()


def test_edit_rejected_by_target_parser_preserves_file_and_measurement(
    tmp_path: Path,
) -> None:
    """Tree-sitter can represent an empty function body, but the target
    Python cannot execute it. The target gate rejects that edit before the
    file or its source-bound mutation state changes."""
    agent = _make_agent(tmp_path)
    path = agent.target_path()
    original = "def test_a():\n    assert True\n"
    path.write_text(original, encoding="utf-8")
    agent.session.last_mutation_result = "prior measured result"
    agent.session.mutation_measurements = 1
    agent.session.tests_changed_since_mutation = False

    out = agent.edit_file("    assert True", "")

    assert agent.test_backend().check_structure("def test_a():\n").ok
    assert path.read_text(encoding="utf-8") == original
    assert agent.session.writes_done == 0
    assert agent.session.last_mutation_result == "prior measured result"
    assert agent.session.mutation_measurements == 1
    assert agent.session.tests_changed_since_mutation is False
    assert "target parser" in out.lower() or "indent" in out.lower()


# measure_mutation with a faked normal run and engine.

class _FakeEngine:
    """Test double for a mutation engine. ``run`` returns a canned
    measurement. ``render_surviving_diffs`` is a no-op."""
    name = "fake_engine"

    def __init__(self, results: tm.mutation.MutationMeasurement) -> None:
        self._results = results
        self.run_calls = 0

    def run(self, **kwargs) -> Any:
        self.run_calls += 1
        return self._results

    def full_operators(self):
        # A single mutant catalogue without a selective operator set, so the
        # switch never fires.
        return None

    def render_surviving_diffs(self, **kwargs) -> str:
        return ""


def _install_fake_engine(monkeypatch, results) -> _FakeEngine:
    """Install a fake engine that returns ``results``, for measure_mutation
    tests on the per-call path without a session."""
    fake = _FakeEngine(results)
    _set_engine(monkeypatch, lambda lang, config: fake)
    return fake


def test_measure_mutation_red_suite_returns_short_circuit(
    monkeypatch, tmp_path: Path,
) -> None:
    """A failing normal run returns ``measure_mutation_red_suite``, clears the
    cache flag and never starts the engine. On a red suite every mutant
    would count as killed."""
    engine_run_count = {"n": 0}

    def fake_get_engine(*a, **kw):
        engine_run_count["n"] += 1
        return _FakeEngine(None)

    _set_engine(monkeypatch, fake_get_engine)
    _install_backend(monkeypatch, _trr(
        1, "FAILED tests/test_x.py::test_a - AssertionError",
    ))
    agent = _make_agent(tmp_path)
    # After the model's own edits a red normal run is feedback it can act
    # on. The early exit fires only when writes_done == 0.
    agent.session.writes_done = 1
    agent.measure_mutation()
    assert engine_run_count["n"] == 0
    assert agent.session.tests_changed_since_mutation is False
    assert agent.session.mutation_measurements == 0
    assert agent.session.early_stop_result is None


def test_measure_mutation_red_suite_inherited_red_early_exits(
    monkeypatch, tmp_path: Path,
) -> None:
    """A red normal run before any edit means the inherited suite arrived red
    (a flaky test or a changed environment). The tool sets a skipped
    MutationTestResult as ``early_stop_result`` and returns a
    [FRAMEWORK STOP] directive."""
    engine_run_count = {"n": 0}

    def fake_get_engine(*a, **kw):
        engine_run_count["n"] += 1
        return _FakeEngine(None)

    _set_engine(monkeypatch, fake_get_engine)
    _install_backend(monkeypatch, _trr(
        1, "ERROR tests/test_x.py - NameError: name 'lo' is not defined",
    ))
    agent = _make_agent(tmp_path)
    assert agent.session.writes_done == 0  # entry state
    out = agent.measure_mutation()
    assert engine_run_count["n"] == 0
    assert "[FRAMEWORK STOP]" in out
    assert agent.session.early_stop_reason == "inherited_red"
    assert agent.session.early_stop_result is not None
    assert agent.session.early_stop_result.status == "skipped_inherited_red"
    assert agent.session.early_stop_result.mutation_score is None


def test_measure_mutation_normal_run_timeout_returns_payload(
    monkeypatch, tmp_path: Path,
) -> None:
    """A normal run timeout goes through the timeout handler and returns its
    message. No mutation measurement runs."""
    _install_backend(monkeypatch, _trr(timed_out=True))
    agent = _make_agent(tmp_path)
    out = agent.measure_mutation()
    assert agent.session.mutation_measurements == 0
    assert out  # the handler's message is surfaced


def test_zero_mutants_tested_passes_the_engine_s_reason(
    monkeypatch, tmp_path: Path,
) -> None:
    """A run that tested no mutant because the engine failed, or because
    its clock ran out, gives the model the engine's reason with the
    zero-mutant message."""
    from langadapter.mutation import MutationRun
    from langadapter.outcomes import Outcome

    failed = MutationRun.failed("calc.py", Outcome.INFRASTRUCTURE_FAILURE,
                                "PIT could not execute: no JDK")
    cut = MutationRun(target_file="calc.py", mutants=(), summary={},
                      requested_catalogue="full", requested_operators=(),
                      outcome=Outcome.TIMED_OUT,
                      reason="the wall clock limit of 60 s ended the run")
    for run, reason in ((failed, "PIT could not execute: no JDK"),
                        (cut, "the wall clock limit of 60 s ended the run")):
        _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
        _install_fake_engine(monkeypatch, tm.mutation.MutationMeasurement.of(run))
        agent = _make_agent(tmp_path / reason[:3])
        out = agent.measure_mutation()
        assert out.startswith("ERROR: 0 mutants were tested")
        assert f"The mutation engine's reason: {reason}" in out


def test_measure_mutation_zero_mutants_returns_payload(
    monkeypatch, tmp_path: Path,
) -> None:
    """With every count at 0 (empty catalogue or an immediate engine failure)
    the 'no mutants generated' message is returned, mutation_measurements
    stays 0 and the cache flag is cleared. ``mutation_inconclusive`` stays
    False, since no mutant was attempted."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    zero_results = tm.mutation.MutationMeasurement(
        score=0.0, killed=0, survived=0, timeout=0, non_viable=0,
        untested=0, total=0,
    )
    _install_fake_engine(monkeypatch, zero_results)
    agent = _make_agent(tmp_path)
    agent.measure_mutation()
    assert agent.session.mutation_measurements == 0
    assert agent.session.tests_changed_since_mutation is False
    assert agent.session.mutation_inconclusive is False
    assert agent.session.early_stop_result is None


def test_measure_mutation_green_normal_run_resets_timeout_streak(
    monkeypatch, tmp_path: Path,
) -> None:
    """A green normal run means the full suite ran in time, so the
    consecutive-timeout streak clears, the same reset run_tests does on a
    green run."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    results = tm.mutation.MutationMeasurement(
        score=80.0, killed=8, survived=2, timeout=0, non_viable=0,
        untested=0, total=10,
    )
    _install_fake_engine(monkeypatch, results)
    agent = _make_agent(tmp_path)
    agent.session.consecutive_timeouts = 3  # streak from prior timeouts
    agent.measure_mutation()
    assert agent.session.consecutive_timeouts == 0


def test_the_summary_counts_and_delta_agree_with_the_score(
    monkeypatch, tmp_path: Path,
) -> None:
    """The killed count is the score's numerator (a timeout and an invalid
    mutant count as killed, as in the score), and the change from the best
    is the difference of the shown values, so the shown numbers add up."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    results = tm.mutation.MutationMeasurement(
        score=70.0, killed=5, survived=3, timeout=1, non_viable=1,
        untested=3, total=10,
    )
    _install_fake_engine(monkeypatch, results)
    agent = _make_agent(tmp_path)
    agent.session.best_mutation_score = 100.0 * 2 / 3      # shown as 66.6%
    out = agent.measure_mutation()
    assert ("Mutation score: 70.0% (killed 7/10, 1 of them by timeout, "
            "1 of them invalid; survived 3; untested 3)") in out
    assert "(prev best: 66.6%, delta: +3.4pp)" in out    # 66.6 + 3.4 = 70.0
    assert agent.session.last_mutation_counts == (7, 3)


def test_incomplete_mutation_measurement_keeps_explicit_partial_loop_progress(
    monkeypatch, tmp_path: Path,
) -> None:
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    # The engine's lower bound: 8 killed of 10 scored and 5 untested.
    results = tm.mutation.MutationMeasurement(
        score=100.0 * 8 / 15, killed=8, survived=2, timeout=0, non_viable=0,
        untested=5, total=10, truncated=True, lower_bound=True,
    )
    _install_fake_engine(monkeypatch, results)
    agent = _make_agent(tmp_path)
    output = agent.measure_mutation()
    assert agent.session.prev_mutation_score == 100.0 * 8 / 15
    assert agent.session.best_mutation_score == 100.0 * 8 / 15
    assert agent.session.mutation_measurements == 1
    assert "Mutation score: 53.3% (killed 8/15; survived 2; untested 5)" in output
    limit = int(tm._AGENT_CONFIG["measurement_timeout_seconds"])
    assert agent.tool_msg("measure_mutation_time_limit", limit=limit, untested=5) in output
    assert "TRUNCATED" not in output
    assert not agent.session.mutation_inconclusive


def test_measure_mutation_threads_the_timeout_factor(
    monkeypatch, tmp_path: Path,
) -> None:
    """The configured per-mutant multiplier reaches the timeout rule from the
    in-loop measurement. Every mutation run (loop, Finishing, baseline)
    resolves the same timeout, so a raised factor applies to all of them."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    results = tm.mutation.MutationMeasurement(
        score=80.0, killed=8, survived=2, timeout=0, non_viable=0,
        untested=0, total=10,
    )
    _install_fake_engine(monkeypatch, results)
    set_phase_value(monkeypatch,
        tm._AGENT_CONFIG, "per_mutant_timeout_factor", 7.0)
    seen = {}
    real = tm.config_module.per_mutant_timeout

    def capture(configured, suite_exec_s, **kw):
        seen.update(kw)
        return real(configured, suite_exec_s, **kw)

    monkeypatch.setattr(tm.config_module, "per_mutant_timeout", capture)
    agent = _make_agent(tmp_path)
    agent.measure_mutation()
    assert seen.get("multiplier") == 7.0


def test_measure_mutation_inconclusive_fires_framework_stop(
    monkeypatch, tmp_path: Path,
) -> None:
    """Mutants that all end non-viable, untested or timed out (killed and
    survived both 0) give a 0% score that measures nothing, for example after
    a worker import error, a broken test command or a too low per-mutant
    timeout. ``mutation_inconclusive`` becomes True, ``early_stop_result``
    has status 'inconclusive', the message starts with the FRAMEWORK STOP
    directive and ``mutation_measurements`` stays 0.
    """
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    # As Cosmic Ray counts: total = covered = killed + survived + timeout = 0
    inconclusive = tm.mutation.MutationMeasurement(
        score=0.0, killed=0, survived=0, timeout=0,
        non_viable=3203, untested=3055, total=0,
    )
    _install_fake_engine(monkeypatch, inconclusive)
    agent = _make_agent(tmp_path)
    out = agent.measure_mutation()
    assert agent.session.mutation_inconclusive is True
    assert agent.session.mutation_measurements == 0
    assert agent.session.early_stop_result is not None
    assert getattr(agent.session.early_stop_result, "status", None) == "inconclusive"
    assert "FRAMEWORK STOP" in out


def test_measure_mutation_inconclusive_when_total_field_nonzero(
    monkeypatch, tmp_path: Path,
) -> None:
    """The same inconclusive path with ``total`` as the full attempted count,
    as mutmut counts. The rule depends only on killed and survived being 0
    with at least one attempted mutant, not on the engine's ``total``."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    inconclusive = tm.mutation.MutationMeasurement(
        score=0.0, killed=0, survived=0, timeout=10,
        non_viable=50, untested=40, total=100,
    )
    _install_fake_engine(monkeypatch, inconclusive)
    agent = _make_agent(tmp_path)
    out = agent.measure_mutation()
    assert agent.session.mutation_inconclusive is True
    assert "FRAMEWORK STOP" in out


class _SeqEngine:
    """Fake engine whose ``run`` returns a queued sequence of results and
    records the operator set each call received, so a test can assert the
    selective-then-full two-phase measurement."""
    name = "fake_seq_engine"

    def __init__(self, seq, full_ops):
        self._seq = list(seq)
        self._last = self._seq[-1] if self._seq else None
        self._full_ops = full_ops
        self.ops_seen = []

    def run(self, **kwargs):
        self.ops_seen.append(kwargs.get("operators"))
        return self._seq.pop(0) if self._seq else self._last

    def full_operators(self):
        return list(self._full_ops) if self._full_ops is not None else None

    def render_surviving_diffs(self, **kwargs):
        return ""


def test_measure_mutation_switches_to_full_catalogue_at_selective_goal(
    monkeypatch, tmp_path: Path,
) -> None:
    """With full_after_selective_goal on, a selective score at the
    threshold makes measure_mutation re-measure the same suite with the full
    catalogue in the same call, switch to full only, reset the best to the
    lower full score and tell the model about the switch."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "full_after_selective_goal", True)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "selective_switch_threshold", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=100.0, killed=10, survived=0, timeout=0, non_viable=0,
        untested=0, total=10)
    full = tm.mutation.MutationMeasurement(
        score=88.0, killed=88, survived=12, timeout=0, non_viable=0,
        untested=0, total=100)
    fake = _SeqEngine([selective, full], ["FULL_OP"])
    _set_engine(monkeypatch,
                        lambda lang, config: fake)

    agent = _make_agent(tmp_path)
    out = agent.measure_mutation()

    assert agent.session.mutation_mode == "full"
    assert agent.session.selective_final_score == 100.0
    assert agent.session.best_mutation_score == 88.0
    # The switch happens on the first model-triggered cycle. Both engine runs
    # of that cycle are recorded, and the cycle counts once.
    assert agent.session.switch_iteration == 1
    assert agent.session.engine_runs == 2
    assert agent.session.mutation_measurements == 1
    assert "[CATALOGUE SWITCH]" in out
    # The first measurement is selective (operators=None, the configured
    # default). The re-measure uses the full operator set.
    assert fake.ops_seen == [None, ["FULL_OP"]]
    # The switch analysis fields land on the result for tests.csv.
    res = agent._synthesize_mutation_result("mutation_goal_met")
    assert res.mutation_switched is True
    assert res.mutation_selective_final == 100.0
    assert res.mutation_full_at_switch == 88.0


@pytest.mark.parametrize("score, streak_before, switches", [(50.0, 1, True), (60.0, 2, False)])
def test_the_selective_plateau_switch_counts_the_current_measurement(
    monkeypatch, tmp_path: Path, score, streak_before, switches,
) -> None:
    """The plateau is judged with the measurement just taken counted: the
    one that brings the measurements without improvement to the limit
    switches to the full catalogue at once, and one that improves does not,
    whatever the streak before it."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "full_after_selective_goal", True)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "selective_switch_threshold", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "max_measurements_without_improvement", 2)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    killed = int(score / 10)
    selective = tm.mutation.MutationMeasurement(
        score=score, killed=killed, survived=10 - killed, timeout=0, non_viable=0, untested=0, total=10)
    full = tm.mutation.MutationMeasurement(
        score=40.0, killed=40, survived=60, timeout=0, non_viable=0, untested=0, total=100)
    _set_engine(monkeypatch, lambda lang, config: _SeqEngine([selective, full], ["FULL_OP"]))
    agent = _make_agent(tmp_path)
    agent.session.best_mutation_score = 50.0
    agent.session.measurements_without_improvement = streak_before
    agent.measure_mutation()
    assert agent.session.mutation_mode == ("full" if switches else "selective")


def test_the_switch_restarts_best_green_on_the_full_catalogue(
    monkeypatch, tmp_path: Path,
) -> None:
    """A selective score never outranks a full one. At the switch the
    best-green snapshot restarts with the suite just measured on the full
    catalogue, so a revert or the phase-end restore cannot return a
    pre-switch suite over full-catalogue progress."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "full_after_selective_goal", True)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "selective_switch_threshold", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=100.0, killed=10, survived=0, timeout=0, non_viable=0,
        untested=0, total=10)
    full = tm.mutation.MutationMeasurement(
        score=88.0, killed=88, survived=12, timeout=0, non_viable=0,
        untested=0, total=100)
    _set_engine(monkeypatch,
                lambda lang, config: _SeqEngine([selective, full], ["FULL_OP"]))
    from testuary.languages.generic import GenericBackend
    real = GenericBackend("python", (".py",))
    monkeypatch.setattr(
        tm.measurement, "observe_file",
        lambda path, **kw: real.test_metrics_observation(
            Path(path).read_text(encoding="utf-8")))
    agent = _make_agent(tmp_path)
    current = "def test_now():\n    assert True\n"
    agent.target_path().write_text(current, encoding="utf-8")
    agent.session.best_green.snapshot = {
        "content": "def test_old():\n    assert True\n", "mutation": 95.0,
        "test_count": 1, "loop_state": {"measured": True, "mode": "selective"},
    }
    agent.measure_mutation()
    snap = agent.session.best_green.snapshot
    assert agent.session.mutation_mode == "full"
    assert snap["content"] == current
    assert snap["mutation"] == 88.0
    assert snap["loop_state"]["mode"] == "full"


def test_switch_run_zero_mutants_writes_no_switch_state(
    monkeypatch, tmp_path: Path,
) -> None:
    """A full-catalogue switch run that attempted nothing is an engine
    failure, not a switch. No switch state is written and the loop stays on
    the selective operator set."""
    set_phase_value(monkeypatch,
        tm._AGENT_CONFIG, "full_after_selective_goal", True)
    set_phase_value(monkeypatch,
        tm._AGENT_CONFIG, "selective_switch_threshold", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=100.0, killed=10, survived=0, timeout=0, non_viable=0,
        untested=0, total=10)
    zero = tm.mutation.MutationMeasurement(
        score=0.0, killed=0, survived=0, timeout=0, non_viable=0,
        untested=0, total=0)
    fake = _SeqEngine([selective, zero], ["FULL_OP"])
    _set_engine(monkeypatch,
                        lambda lang, config: fake)
    agent = _make_agent(tmp_path)
    out = agent.measure_mutation()
    s = agent.session
    assert s.mutation_mode == "selective"
    assert s.switch_iteration is None
    assert s.full_score_at_switch is None
    assert "0 mutants" in out


def test_switch_run_inconclusive_fires_framework_stop(
    monkeypatch, tmp_path: Path,
) -> None:
    """A full-catalogue switch run without a killed or survived mutant is
    inconclusive. The framework stop fires and the all-timeout score is
    never accepted."""
    set_phase_value(monkeypatch,
        tm._AGENT_CONFIG, "full_after_selective_goal", True)
    set_phase_value(monkeypatch,
        tm._AGENT_CONFIG, "selective_switch_threshold", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=100.0, killed=10, survived=0, timeout=0, non_viable=0,
        untested=0, total=10)
    all_timeout = tm.mutation.MutationMeasurement(
        score=100.0, killed=0, survived=0, timeout=50, non_viable=0,
        untested=0, total=50)
    fake = _SeqEngine([selective, all_timeout], ["FULL_OP"])
    _set_engine(monkeypatch,
                        lambda lang, config: fake)
    agent = _make_agent(tmp_path)
    agent.measure_mutation()
    s = agent.session
    assert s.mutation_inconclusive is True
    assert s.early_stop_reason == "engine_inconclusive"
    assert s.mutation_mode == "selective"
    assert s.full_score_at_switch is None


@pytest.mark.parametrize("switch", [False, True])
def test_an_inconclusive_run_logs_its_counts_and_the_first_run_a_likely_cause(
    monkeypatch, tmp_path: Path, switch,
) -> None:
    """An engine run without a killed or survived mutant logs a CRITICAL line
    with its counts. The first run's line names the per-mutant timeout as a
    likely cause, the catalogue switch's run's line names none."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "full_after_selective_goal", switch)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "selective_switch_threshold", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=100.0, killed=10, survived=0, timeout=0, non_viable=0,
        untested=0, total=10)
    inconclusive = tm.mutation.MutationMeasurement(
        score=0.0, killed=0, survived=0, timeout=2, non_viable=1,
        untested=0, total=3)
    engine = _SeqEngine([selective, inconclusive] if switch else [inconclusive], ["FULL_OP"])
    _set_engine(monkeypatch, lambda lang, config: engine)
    agent = _make_agent(tmp_path)
    lines = []
    monkeypatch.setattr(agent, "log", lambda message, *args, **fields: lines.append(message))
    assert agent.measure_mutation() == agent.tool_msg("measure_mutation_inconclusive")
    (critical,) = [line for line in lines if line.startswith("[CRITICAL]")]
    counts = "0 useful outcomes from 3 attempted (suspicious=1, timeout=2, untested=0)."
    if switch:
        assert critical == f"[CRITICAL] full-catalogue switch measurement INCONCLUSIVE: {counts}"
    else:
        assert critical.startswith(
            f"[CRITICAL] mutation measurement INCONCLUSIVE: {counts} Likely: worker "
            "import error, broken test_command, or suite too slow for the per-mutant "
            "timeout (")
        assert critical.endswith("s).")


def test_measure_mutation_at_goal_does_not_stop_while_switch_pending(
    monkeypatch, tmp_path: Path,
) -> None:
    """The goal is judged on the full catalogue, never on the selective
    operator set.
    A selective score at or above the goal but below the switch threshold
    keeps the selective mode with the switch pending, so the full catalogue
    is measured before a goal-met stop can fire."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "full_after_selective_goal", True)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "selective_switch_threshold", 85)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 70)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=75.0, killed=75, survived=25, timeout=0, non_viable=0,
        untested=0, total=100)
    fake = _SeqEngine([selective], ["FULL_OP"])
    _set_engine(monkeypatch,
                        lambda lang, config: fake)

    agent = _make_agent(tmp_path)
    out = agent.measure_mutation()

    # 75% selective reaches goal 70 but not threshold 85: no switch, and no
    # goal stop before the full catalogue is scored.
    assert agent.session.mutation_mode == "selective"
    assert agent.session.selective_switch_possible is True
    assert agent.session.early_stop_result is None
    assert "FRAMEWORK STOP" not in out
    assert fake.ops_seen == [None]          # only the selective set ran


def test_measure_mutation_selective_phase_frames_score_as_approximation(
    monkeypatch, tmp_path: Path,
) -> None:
    """While scoring the selective operator set the model is told that the score
    approximates the full catalogue's, so it does not take it as final."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "full_after_selective_goal", True)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "selective_switch_threshold", 85)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=50.0, killed=50, survived=50, timeout=0, non_viable=0,
        untested=0, total=100)
    fake = _SeqEngine([selective], ["FULL_OP"])
    _set_engine(monkeypatch,
                        lambda lang, config: fake)

    agent = _make_agent(tmp_path)
    out = agent.measure_mutation()

    assert agent.session.mutation_mode == "selective"
    assert "[SELECTIVE APPROXIMATION]" in out


def test_measure_mutation_no_switch_when_feature_disabled(
    monkeypatch, tmp_path: Path,
) -> None:
    """With the switch off (the bundled default), reaching the selective goal
    neither re-measures nor changes the mode."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "full_after_selective_goal", False)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=100.0, killed=10, survived=0, timeout=0, non_viable=0,
        untested=0, total=10)
    fake = _install_fake_engine(monkeypatch, selective)
    agent = _make_agent(tmp_path)
    agent.measure_mutation()
    assert agent.session.mutation_mode == "selective"
    assert agent.session.switch_iteration is None
    assert fake.run_calls == 1


def test_measure_mutation_no_switch_when_engine_has_no_full_catalogue(
    monkeypatch, tmp_path: Path,
) -> None:
    """An engine without a selective operator set (full_operators() is None, as for
    mutmut and PIT) has no fuller catalogue, so it never switches, even with
    the switch on and the selective goal reached."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "full_after_selective_goal", True)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "selective_switch_threshold", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=100.0, killed=10, survived=0, timeout=0, non_viable=0,
        untested=0, total=10)
    fake = _SeqEngine([selective], None)
    _set_engine(monkeypatch,
                        lambda lang, config: fake)
    agent = _make_agent(tmp_path)
    out = agent.measure_mutation()
    assert agent.session.mutation_mode == "selective"
    assert agent.session.switch_iteration is None
    assert agent.session.selective_final_score is None
    assert "[CATALOGUE SWITCH]" not in out
    assert fake.ops_seen == [None]


def test_measure_mutation_switches_to_full_catalogue_on_selective_plateau(
    monkeypatch, tmp_path: Path,
) -> None:
    """A selective phase whose stagnation streak reaches patience below the
    switch threshold switches to the full catalogue instead of ending, so the
    rest of the request limit improves the full score. The full catalogue is
    measured in the same call and the best is reset to the full score."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "full_after_selective_goal", True)
    # The 60% selective score never reaches this threshold, so only the
    # plateau can switch.
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "selective_switch_threshold", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "max_measurements_without_improvement", 3)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=60.0, killed=6, survived=4, timeout=0, non_viable=0,
        untested=0, total=10)
    full = tm.mutation.MutationMeasurement(
        score=44.0, killed=44, survived=56, timeout=0, non_viable=0,
        untested=0, total=100)
    fake = _SeqEngine([selective, full], ["FULL_OP"])
    _set_engine(monkeypatch,
                        lambda lang, config: fake)
    agent = _make_agent(tmp_path)
    # A selective plateau at 60% over prior measurements. This measurement,
    # no improvement either, brings the streak to the limit.
    agent.session.best_mutation_score = 60.0
    agent.session.measurements_without_improvement = 2
    out = agent.measure_mutation()

    assert agent.session.mutation_mode == "full"
    assert agent.session.selective_final_score == 60.0
    assert agent.session.best_mutation_score == 44.0
    assert agent.session.switch_iteration is not None
    assert agent.session.selective_switch_possible is False
    assert "[CATALOGUE SWITCH]" in out
    assert fake.ops_seen == [None, ["FULL_OP"]]


def test_early_stop_suppressed_until_full_catalogue(
    monkeypatch, tmp_path: Path,
) -> None:
    """The stagnation stop does not fire while a selective-to-full switch is
    still possible, since a selective plateau switches instead. It fires on
    the full catalogue."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "allow_early_stop", True)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "max_measurements_without_improvement", 3)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "extra_measurements_after_goal", 0)
    agent = _make_agent(tmp_path)
    agent.session.best_mutation_score = 60.0   # below goal
    agent.session.measurements_without_improvement = 3        # at patience

    # With the switch pending a plateau switches and never stops.
    agent.session.selective_switch_possible = True
    assert agent._maybe_set_mutation_stop(source_tool="t") is None
    assert agent.session.early_stop_result is None

    # On the full catalogue stagnation stops.
    agent.session.selective_switch_possible = False
    out = agent._maybe_set_mutation_stop(source_tool="t")
    assert out is not None and "[FRAMEWORK STOP]" in out
    assert agent.session.early_stop_reason == "mutation_stagnation"


def test_a_stagnant_measurement_shows_no_suite_report(
    monkeypatch, tmp_path: Path,
) -> None:
    """A mutation measurement long into stagnation reports the score
    and nothing about the suite's per-test redundancy or cost, though the
    backend could give per-test evidence with kills."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "full_after_selective_goal", False)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "max_measurements_without_improvement", 1000)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "restore_count", 999)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    fake = tm.languages.get_backend("python")
    fake.collect_eviction_data = lambda *args, **kwargs: SimpleNamespace(
        per_test_tokens={"test_a": frozenset({"K:m1"}), "test_b": frozenset({"K:m1"})},
        per_test_time_ms={"test_a": 2.0, "test_b": 3.0}, suite_time_ms=5.0, includes_kills=True,
    )
    monkeypatch.setattr(tm.languages, "get_backend", lambda lang: fake)
    res = tm.mutation.MutationMeasurement(
        score=50.0, killed=5, survived=5, timeout=0, non_viable=0,
        untested=0, total=10)
    _install_fake_engine(monkeypatch, res)
    agent = _make_agent(tmp_path)
    agent.target_path().write_text(
        "def test_a(): assert True\n\ndef test_b(): assert True\n", encoding="utf-8")
    # A prior best above the new score makes this measurement stagnant.
    agent.session.best_mutation_score = 90.0
    agent.session.measurements_without_improvement = 10

    summary = agent.measure_mutation()

    assert "50.0" in summary
    assert "[suite report]" not in summary and "add no unique" not in summary


def test_measure_mutation_switch_is_one_way(
    monkeypatch, tmp_path: Path,
) -> None:
    """After the switch the mode stays full. A later full measurement at the
    threshold does not switch again, the selective convergence point is
    recorded once, and only the first switch is announced to the model."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "full_after_selective_goal", True)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "selective_switch_threshold", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "max_measurements_without_improvement", 100)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=100.0, killed=10, survived=0, timeout=0, non_viable=0,
        untested=0, total=10)
    full1 = tm.mutation.MutationMeasurement(
        score=88.0, killed=88, survived=12, timeout=0, non_viable=0,
        untested=0, total=100)
    full2 = tm.mutation.MutationMeasurement(
        score=92.0, killed=92, survived=8, timeout=0, non_viable=0,
        untested=0, total=100)
    fake = _SeqEngine([selective, full1, full2], ["FULL_OP"])
    _set_engine(monkeypatch,
                        lambda lang, config: fake)
    agent = _make_agent(tmp_path)
    agent.measure_mutation()                      # selective, switch, full1
    first_switch_iter = agent.session.switch_iteration
    assert first_switch_iter is not None
    agent.session.tests_changed_since_mutation = True
    out2 = agent.measure_mutation()               # full2, no re-switch
    assert agent.session.mutation_mode == "full"
    assert agent.session.switch_iteration == first_switch_iter
    assert agent.session.selective_final_score == 100.0
    assert "[CATALOGUE SWITCH]" not in out2
    # Selective (None), then the full operators on both later measurements.
    assert fake.ops_seen == [None, ["FULL_OP"], ["FULL_OP"]]


def test_measure_mutation_full_best_not_reduced_by_lower_version(
    monkeypatch, tmp_path: Path,
) -> None:
    """The full score at the switch is the best, and a later lower full
    measurement does not replace it."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "full_after_selective_goal", True)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "selective_switch_threshold", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "max_measurements_without_improvement", 100)
    # Keep the in-loop regression revert off so only the best is tested.
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "restore_count", 999)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=100.0, killed=10, survived=0, timeout=0, non_viable=0,
        untested=0, total=10)
    full_high = tm.mutation.MutationMeasurement(
        score=88.0, killed=88, survived=12, timeout=0, non_viable=0,
        untested=0, total=100)
    full_low = tm.mutation.MutationMeasurement(
        score=80.0, killed=80, survived=20, timeout=0, non_viable=0,
        untested=0, total=100)
    fake = _SeqEngine([selective, full_high, full_low], ["FULL_OP"])
    _set_engine(monkeypatch,
                        lambda lang, config: fake)
    agent = _make_agent(tmp_path)
    agent.measure_mutation()                      # selective, switch, 88
    assert agent.session.best_mutation_score == 88.0
    agent.session.tests_changed_since_mutation = True
    agent.measure_mutation()                      # full_low = 80
    assert agent.session.best_mutation_score == 88.0   # not reduced
    assert agent.session.mutation_mode == "full"


def test_measure_mutation_no_switch_below_threshold(
    monkeypatch, tmp_path: Path,
) -> None:
    """With the switch on but the selective score below the threshold there
    is no switch and no re-measure, so the switch costs nothing extra."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "full_after_selective_goal", True)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG,
                        "selective_switch_threshold", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=72.0, killed=72, survived=28, timeout=0, non_viable=0,
        untested=0, total=100)
    fake = _SeqEngine([selective], ["FULL_OP"])
    _set_engine(monkeypatch,
                        lambda lang, config: fake)
    agent = _make_agent(tmp_path)
    out = agent.measure_mutation()
    assert agent.session.mutation_mode == "selective"
    assert agent.session.switch_iteration is None
    assert "[CATALOGUE SWITCH]" not in out
    assert fake.ops_seen == [None]


def test_measure_mutation_session_reuse_wires_log_fn_into_engine(
    monkeypatch, tmp_path: Path,
) -> None:
    """With a reused mutation session, the loop routes the session's log lines
    to its own logger before calling ``session.run(...)``.

    The session is opened before this target's logger is bound, and the
    engine's progress poller and inconclusive-abort observer run only with a
    logger. Without it a mutation run stays silent for its whole
    wall_clock_timeout. The fake session records the runner's log target at
    call time."""
    class _CapturingSession:
        """Minimal mutation-session double."""

        def __init__(self):
            self.workers = 48
            self.use_http = True
            self.runner = SimpleNamespace(
                log=SimpleNamespace(target=None),
                name="fake_cosmic_ray",
                full_operators=lambda: None,
            )
            self.log_fn_at_run_time = None

        def run(self, **kwargs):
            # Record the runner's log target before the run starts.
            self.log_fn_at_run_time = self.runner.log.target
            return tm.mutation.MutationMeasurement(
                score=70.0, killed=7, survived=3,
                timeout=0, non_viable=0, untested=0, total=10,
            )

    fake_session = _CapturingSession()
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    agent = _make_agent(tmp_path, mutation_session=fake_session)
    agent.measure_mutation()
    assert fake_session.log_fn_at_run_time == agent.log, (
        "Session-reuse path failed to wire log_fn into the engine config "
        "before calling session.run. The progress poller is disabled and "
        "the inconclusive-abort gate has no observer."
    )


def test_measure_mutation_happy_path_updates_state(
    monkeypatch, tmp_path: Path,
) -> None:
    """A green normal run and a measured score set best_mutation_score,
    increment mutation_measurements, reset measurements_without_improvement, clear the cache
    flag and fill last_mutation_result."""
    _install_backend(monkeypatch, _trr(0, "3 passed"))
    results = tm.mutation.MutationMeasurement(
        score=72.5, killed=29, survived=11, timeout=0, non_viable=0,
        untested=0, total=40,
    )
    _install_fake_engine(monkeypatch, results)
    agent = _make_agent(tmp_path)
    out = agent.measure_mutation()
    assert agent.session.mutation_measurements == 1
    assert agent.session.best_mutation_score == 72.5
    assert agent.session.prev_mutation_score == 72.5
    assert agent.session.measurements_without_improvement == 0
    assert agent.session.tests_changed_since_mutation is False
    assert "72.5%" in out
    assert agent.session.last_mutation_result == out


def test_measure_mutation_lower_score_increments_no_improvement(
    monkeypatch, tmp_path: Path,
) -> None:
    """A measurement below the recorded best keeps best_mutation_score and
    increments measurements_without_improvement. Both scores lie below goal (80%),
    so the goal-met stop does not fire."""
    _install_backend(monkeypatch, _trr(0, ""))
    results = tm.mutation.MutationMeasurement(
        score=65.0, killed=26, survived=14, timeout=0, non_viable=0,
        untested=0, total=40,
    )
    _install_fake_engine(monkeypatch, results)
    agent = _make_agent(tmp_path)
    # A higher earlier best, still below the goal.
    agent.session.best_mutation_score = 70.0
    agent.session.mutation_measurements = 1
    agent.session.last_mutation_result = "prior result"
    agent.session.tests_changed_since_mutation = True   # not cached
    agent.measure_mutation()
    assert agent.session.best_mutation_score == 70.0       # unchanged
    assert agent.session.prev_mutation_score == 65.0
    assert agent.session.measurements_without_improvement == 1
    assert agent.session.mutation_measurements == 2


def test_measure_mutation_below_minimum_any_gain_resets_counter(
    monkeypatch, tmp_path: Path,
) -> None:
    """Below ``minimum`` any gain resets the stagnation counter,
    even one under ``min_improvement_points``, so the phase continues while
    the score still rises."""
    _install_backend(monkeypatch, _trr(0, ""))
    results = tm.mutation.MutationMeasurement(
        score=50.2, killed=20, survived=20, timeout=0, non_viable=0,
        untested=0, total=40,
    )
    _install_fake_engine(monkeypatch, results)
    agent = _make_agent(tmp_path)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "min_improvement_points", 0.5)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "minimum", 65)
    agent.session.best_mutation_score = 50.0
    agent.session.measurements_without_improvement = 4
    agent.session.tests_changed_since_mutation = True
    agent.measure_mutation()
    assert agent.session.best_mutation_score == 50.2
    assert agent.session.measurements_without_improvement == 0


def test_measure_mutation_at_minimum_configured_delta_applies(
    monkeypatch, tmp_path: Path,
) -> None:
    """At or above the minimum a sub-threshold gain counts as stagnation."""
    _install_backend(monkeypatch, _trr(0, ""))
    results = tm.mutation.MutationMeasurement(
        score=70.2, killed=28, survived=12, timeout=0, non_viable=0,
        untested=0, total=40,
    )
    _install_fake_engine(monkeypatch, results)
    agent = _make_agent(tmp_path)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "min_improvement_points", 0.5)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "minimum", 65)
    agent.session.best_mutation_score = 70.0
    agent.session.measurements_without_improvement = 2
    agent.session.tests_changed_since_mutation = True
    agent.measure_mutation()
    assert agent.session.best_mutation_score == 70.2
    assert agent.session.measurements_without_improvement == 3


def test_measure_mutation_minimum_zero_disables_the_band(
    monkeypatch, tmp_path: Path,
) -> None:
    """``minimum = 0`` leaves the configured delta in force at any
    score."""
    _install_backend(monkeypatch, _trr(0, ""))
    results = tm.mutation.MutationMeasurement(
        score=50.2, killed=20, survived=20, timeout=0, non_viable=0,
        untested=0, total=40,
    )
    _install_fake_engine(monkeypatch, results)
    agent = _make_agent(tmp_path)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "min_improvement_points", 0.5)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "minimum", 0)
    agent.session.best_mutation_score = 50.0
    agent.session.measurements_without_improvement = 2
    agent.session.tests_changed_since_mutation = True
    agent.measure_mutation()
    assert agent.session.best_mutation_score == 50.2
    assert agent.session.measurements_without_improvement == 3


def test_finalize_result_stamps_measured_score_on_voluntary_return(
    monkeypatch, tmp_path: Path,
) -> None:
    """A voluntary model return leaves the [auto] mutation_score null. The
    finalizer sets the session's best, so the CSV row and the mutation minimum
    read the measured value as on the synthesized early-stop path."""
    agent = _make_agent(tmp_path)
    agent.session.best_mutation_score = 72.5
    monkeypatch.setattr(agent, "_settle_best_green", lambda *_args, **_kwargs: False)
    voluntary = tm.MutationTestResult(
        mutation_test_file_name="test_foo_mutation.py",
        explanation="done",
        targeted_mutants=[],
    )
    assert voluntary.mutation_score is None
    out = agent.finalize_result(
        voluntary, error_msg=None, duration_ms=1000.0,
        usage_tracker=SimpleNamespace(
            input_tokens_used=10, output_tokens_used=5,
        ),
        final_messages=[],
    )
    assert out.mutation_score == 72.5


def test_finalize_result_stamps_the_session_score(
    monkeypatch, tmp_path: Path,
) -> None:
    """Finalization sets the session's best on every path, so the recorded
    score is always the measured one. The synthesizer reads the same session
    state."""
    agent = _make_agent(tmp_path)
    agent.session.best_mutation_score = 70.0
    monkeypatch.setattr(agent, "_settle_best_green", lambda *_args, **_kwargs: False)
    carried = tm.MutationTestResult(
        mutation_test_file_name="test_foo_mutation.py",
        explanation="stopped early",
        targeted_mutants=[],
        mutation_score=68.0,
    )
    out = agent.finalize_result(
        carried, error_msg=None, duration_ms=1000.0,
        usage_tracker=SimpleNamespace(
            input_tokens_used=10, output_tokens_used=5,
        ),
        final_messages=[],
    )
    assert out.mutation_score == 70.0


def test_measure_mutation_cached_recall_skips_subprocess(
    monkeypatch, tmp_path: Path,
) -> None:
    """With tests_changed_since_mutation False and a cached result,
    measure_mutation returns the cached rejection without a test run or an
    engine run."""
    sub_calls = {"n": 0}
    eng_calls = {"n": 0}

    def responder(test_paths):
        sub_calls["n"] += 1
        return _trr(0, "")

    def fake_eng(*a, **kw):
        eng_calls["n"] += 1
        return _FakeEngine(None)

    _install_backend(monkeypatch, responder)
    _set_engine(monkeypatch, fake_eng)
    agent = _make_agent(tmp_path)
    agent.session.tests_changed_since_mutation = False
    agent.session.last_mutation_result = "Mutation score: 50.0% (killed 5/10)"
    agent.session.best_mutation_score = 50.0
    out = agent.measure_mutation()
    assert sub_calls["n"] == 0
    assert eng_calls["n"] == 0
    assert "50.0%" in out  # the cached result


def test_measure_mutation_red_normal_run_stores_red_in_cache_slot(
    monkeypatch, tmp_path: Path,
) -> None:
    """A red normal run stores its failure detail in ``last_mutation_result``,
    so a later cache hit does not serve an earlier measurement that would
    present the suite as green."""
    _install_backend(monkeypatch, _trr(rc=1, out="FAILED test_a"))
    agent = _make_agent(tmp_path)
    # An earlier measurement is cached, and the suite is red.
    agent.session.best_mutation_score = 80.0
    agent.session.last_mutation_result = "Mutation score: 80.0% (stale)"
    agent.session.tests_changed_since_mutation = True  # not cached
    out = agent.measure_mutation()
    # The message is the red normal run's detail, not the cached 80%.
    assert "red" in out.lower() or "FAILED" in out
    assert "80.0%" not in agent.session.last_mutation_result
    assert agent.session.tests_changed_since_mutation is False


def test_measure_mutation_refuses_when_tests_known_red_no_edit(
    monkeypatch, tmp_path: Path,
) -> None:
    """When run_tests reported failures and the mutation file has not been
    edited since, ``measure_mutation`` refuses without running the
    normal run and points the model to an edit."""
    sub_calls = {"n": 0}
    eng_calls = {"n": 0}

    def responder(test_paths):
        sub_calls["n"] += 1
        return _trr(0, "")

    def fake_eng(*a, **kw):
        eng_calls["n"] += 1
        return None
    _install_backend(monkeypatch, responder)
    _set_engine(monkeypatch, fake_eng)
    agent = _make_agent(tmp_path)
    agent.session.last_tests_passed = False
    agent.session.tests_changed_since_run_tests = False
    agent.session.last_run_tests_result = "FAILED test_x"
    out = agent.measure_mutation()
    assert sub_calls["n"] == 0
    assert eng_calls["n"] == 0
    assert "REFUSED" in out
    assert "edit" in out.lower()


def test_measure_mutation_cached_after_red_uses_red_variant(
    monkeypatch, tmp_path: Path,
) -> None:
    """After a red normal run stored its detail in the cache, a cache hit
    returns ``measure_mutation_cached_red``, not the regular cached message
    that assumes survivors."""
    agent = _make_agent(tmp_path)
    agent.session.tests_changed_since_mutation = False
    agent.session.last_mutation_result = "REJECTED: pre-flight pytest is RED"
    agent.session.best_mutation_score = None  # no real measurement happened
    out = agent.measure_mutation()
    assert "never measured" in out.lower()
    assert "surviving mutants from" not in out.lower()


# A suite timeout reverts the mutation file, names suspects and stops after N
# timeouts. Before any mutation test is written it ends the phase.

def _timeout_backend(monkeypatch):
    """Backend whose run_test_files always times out, with a real skeleton
    and function fingerprints for the suspects diff."""
    from testuary.languages.generic import GenericBackend
    real = GenericBackend("python", (".py",))

    def _run(*a, **kw):
        return run_result(ok=False, returncode=-1, timed_out=True)
    monkeypatch.setattr(
        tm.languages, "get_backend",
        lambda *a, **kw: SimpleNamespace(
            run_test_files=_run, skeleton=real.skeleton,
            function_fingerprints=real.function_fingerprints),
    )


def test_run_tests_timeout_reverts_names_then_stops(monkeypatch, tmp_path) -> None:
    """With mutation tests written, a full-suite timeout reverts the
    mutation file to best-green, names the mutation tests added since, and
    after suite_timeout_stop_after timeouts ends with the best score."""
    agent = _make_agent(tmp_path)
    mut_path = agent.target_path()
    snapshot = "def test_k1(): assert True\n"
    slow = (
        "def test_k1(): assert True\n\n"
        "def test_slow():\n    while True:\n        pass\n"
    )
    # The snapshot as a measurement leaves it: its bytes are the measured ones.
    agent.session.best_green.snapshot = {
        "content": snapshot, "mutation": 60.0, "coverage": 85.0,
        "test_count": 1, "time": 1.0,
        "loop_state": {"measured": True, "mode": agent.session.mutation_mode,
                       "measured_suite_hash": hashlib.sha256(snapshot.encode("utf-8")).hexdigest()},
    }
    agent.session.best_mutation_score = 60.0
    agent.session.writes_done = 3
    _timeout_backend(monkeypatch)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "suite_timeout_stop_after", 3)

    for n in (1, 2):
        mut_path.write_text(slow, encoding="utf-8")
        agent.session.tests_changed_since_run_tests = True
        out = agent.run_tests()
        assert agent.session.consecutive_timeouts == n
        assert "test_slow" in out
        assert mut_path.read_text(encoding="utf-8") == snapshot  # reverted
        assert agent.session.early_stop_result is None

    mut_path.write_text(slow, encoding="utf-8")
    agent.session.tests_changed_since_run_tests = True
    out = agent.run_tests()
    assert agent.session.consecutive_timeouts == 3
    assert agent.session.early_stop_reason == "repeated_test_timeout"
    assert agent.session.early_stop_result.status == "stagnation"
    assert "FRAMEWORK STOP" in out


def test_repeated_timeouts_without_a_measured_snapshot_end_with_failing_tests(monkeypatch, tmp_path) -> None:
    """Repeated timeouts that leave an unmeasured best-green snapshot end
    the phase as tests_failed, as the coverage phase records the same stop."""
    agent = _make_agent(tmp_path)
    mut_path = agent.target_path()
    snapshot = "def test_k1(): assert True\n"
    agent.session.best_green.snapshot = {"content": snapshot, "mutation": None, "test_count": 1}
    agent.session.writes_done = 3
    _timeout_backend(monkeypatch)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "suite_timeout_stop_after", 1)
    mut_path.write_text("def test_slow():\n    while True:\n        pass\n", encoding="utf-8")
    agent.session.tests_changed_since_run_tests = True
    agent.run_tests()
    assert agent.session.early_stop_reason == "repeated_test_timeout"
    assert agent.session.early_stop_result.status == "tests_failed"


def test_run_tests_timeout_stop_disabled_at_zero(monkeypatch, tmp_path) -> None:
    """``suite_timeout_stop_after <= 0`` disables the timeout stop. Each
    timeout still reverts to best-green, but the phase never ends on it."""
    agent = _make_agent(tmp_path)
    mut_path = agent.target_path()
    snapshot = "def test_k1(): assert True\n"
    slow = (
        "def test_k1(): assert True\n\n"
        "def test_slow():\n    while True:\n        pass\n"
    )
    agent.session.best_green.snapshot = {
        "content": snapshot, "mutation": 60.0, "coverage": 85.0,
        "test_count": 1, "time": 1.0,
    }
    agent.session.best_mutation_score = 60.0
    agent.session.writes_done = 3
    _timeout_backend(monkeypatch)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "suite_timeout_stop_after", -1)

    for n in (1, 2, 3, 4):
        mut_path.write_text(slow, encoding="utf-8")
        agent.session.tests_changed_since_run_tests = True
        agent.run_tests()
        assert agent.session.consecutive_timeouts == n
        assert agent.session.early_stop_result is None
        assert mut_path.read_text(encoding="utf-8") == snapshot


def test_run_tests_timeout_before_any_edit_ends_phase(monkeypatch, tmp_path) -> None:
    """Before any edit the timeout comes from the inherited suite itself,
    which added tests cannot fix, so the phase ends at once."""
    agent = _make_agent(tmp_path)
    agent.session.writes_done = 0
    agent.session.best_mutation_score = None
    _timeout_backend(monkeypatch)
    agent.session.tests_changed_since_run_tests = True
    out = agent.run_tests()
    assert agent.session.early_stop_reason == "inherited_suite_too_slow"
    assert agent.session.early_stop_result.status == "skipped_inherited_slow"
    assert "inherited suite" in out


def test_measure_mutation_normal_run_timeout_uses_handler(monkeypatch, tmp_path) -> None:
    """A timeout in measure_mutation's normal run goes through the same
    handler (revert and suspects), not a bare timeout return."""
    agent = _make_agent(tmp_path)
    mut_path = agent.target_path()
    snapshot = "def test_k1(): assert True\n"
    slow = (
        "def test_k1(): assert True\n\n"
        "def test_slow():\n    while True:\n        pass\n"
    )
    agent.session.best_green.snapshot = {
        "content": snapshot, "mutation": 60.0, "coverage": 85.0,
        "test_count": 1, "time": 1.0,
    }
    agent.session.best_mutation_score = 60.0
    agent.session.writes_done = 3
    agent.session.last_tests_passed = True             # pass the green gate
    agent.session.tests_changed_since_mutation = True   # not cached
    _timeout_backend(monkeypatch)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "suite_timeout_stop_after", 5)
    mut_path.write_text(slow, encoding="utf-8")
    out = agent.measure_mutation()
    assert agent.session.consecutive_timeouts == 1
    assert "test_slow" in out
    assert mut_path.read_text(encoding="utf-8") == snapshot  # reverted


# measure_mutation: measurement accounting across the counters.

def test_measure_mutation_visible_call_counts_once(
    monkeypatch, tmp_path: Path,
) -> None:
    """A visible engine-backed call counts one tool call, one visible
    assessment, one engine run, and one model-triggered measurement."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    results = tm.mutation.MutationMeasurement(
        score=80.0, killed=8, survived=2, timeout=0, non_viable=0,
        untested=0, total=10,
    )
    _install_fake_engine(monkeypatch, results)
    agent = _make_agent(tmp_path)
    agent.measure_mutation()
    s = agent.session
    assert s.session_tool_calls == 1
    assert s.measure_mutation_calls == 1
    assert s.engine_runs == 1
    assert s.mutation_measurements == 1
    assert s.framework_prep_runs == 0
    assert s.cached_measure_returns == 0


def test_measure_mutation_refused_and_cached_count_the_call(
    monkeypatch, tmp_path: Path,
) -> None:
    """Refused and cached returns go through track_output. The tool-call
    and visible-assessment counters advance, the cached counter records the
    cache hit, and no engine runs."""
    _install_backend(monkeypatch, _trr(0, "1 passed"))
    agent = _make_agent(tmp_path)
    agent.session.last_tests_passed = False
    agent.session.tests_changed_since_run_tests = False
    agent.session.last_run_tests_result = "FAILED tests/test_x.py::test_a"
    agent.measure_mutation()
    assert agent.session.session_tool_calls == 1
    assert agent.session.measure_mutation_calls == 1
    assert agent.session.cached_measure_returns == 0
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_mutation = False
    agent.session.last_mutation_result = "Mutation score: 60.0%"
    agent.session.best_mutation_score = 60.0
    out = agent.measure_mutation()
    assert agent.session.session_tool_calls == 2
    assert agent.session.measure_mutation_calls == 2
    assert agent.session.cached_measure_returns == 1
    assert agent.session.engine_runs == 0
    assert "60.0" in out


@pytest.mark.parametrize("setting,expected", [(True, 1), (False, 0)])
def test_measure_mutation_cached_call_count_repeated_measurements(
    monkeypatch, tmp_path: Path, setting: bool, expected: int,
) -> None:
    """The cached path advances the no-improvement counter exactly when
    ``count_repeated_measurements`` is on, the same rule
    measure_coverage applies to its cached calls."""
    set_phase_value(monkeypatch,
        tm._AGENT_CONFIG, "count_repeated_measurements", setting,
    )
    agent = _make_agent(tmp_path)
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_mutation = False
    agent.session.last_mutation_result = "Mutation score: 60.0%"
    agent.session.best_mutation_score = 60.0
    agent.measure_mutation()
    assert agent.session.measurements_without_improvement == expected


def test_measure_mutation_cached_red_does_not_advance_measurements_without_improvement(
    monkeypatch, tmp_path: Path,
) -> None:
    """A cached return without a measurement on record (a red normal run's
    cache) never advances the counter, since stagnation counts measured
    results without improvement."""
    set_phase_value(monkeypatch,
        tm._AGENT_CONFIG, "count_repeated_measurements", True,
    )
    agent = _make_agent(tmp_path)
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_mutation = False
    agent.session.last_mutation_result = "normal run red detail"
    assert agent.session.best_mutation_score is None
    agent.measure_mutation()
    assert agent.session.measurements_without_improvement == 0


def test_framework_baseline_measure_is_not_model_activity(
    monkeypatch, tmp_path: Path,
) -> None:
    """The fallback baseline runs the engine as framework preparation. The
    engine cost is recorded as a preparation run and an engine run, the
    model-facing counters and per-step state stay untouched, and the
    mutation session serves the cache as the derivation leaves it."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    results = tm.mutation.MutationMeasurement(
        score=70.0, killed=7, survived=3, timeout=0, non_viable=0,
        untested=0, total=10,
    )
    fake = _install_fake_engine(monkeypatch, results)
    agent = _make_agent(tmp_path)
    agent.framework_baseline_measure()
    s = agent.session
    assert fake.run_calls == 1
    assert s.framework_prep_runs == 1
    assert s.engine_runs == 1
    assert s.mutation_measurements == 0
    assert s.measure_mutation_calls == 0
    assert s.session_tool_calls == 0
    assert s.last_tool_event_at is None
    assert s.best_mutation_score == 70.0
    assert s.tests_changed_since_mutation is False
    assert s.last_mutation_result
    out = agent.measure_mutation()
    assert fake.run_calls == 1  # served from the prepared cache
    assert s.cached_measure_returns == 1
    assert "70.0" in out


def test_framework_baseline_switch_records_iteration_zero(
    monkeypatch, tmp_path: Path,
) -> None:
    """A selective baseline at the switch threshold during framework
    preparation switches immediately and records the switch at measurement
    index 0, matching the derived immediate switch. Both engine executions
    are preparation cost."""
    set_phase_value(monkeypatch,
        tm._AGENT_CONFIG, "full_after_selective_goal", True,
    )
    set_phase_value(monkeypatch,
        tm._AGENT_CONFIG, "selective_switch_threshold", 100,
    )
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=100.0, killed=10, survived=0, timeout=0, non_viable=0,
        untested=0, total=10)
    full = tm.mutation.MutationMeasurement(
        score=88.0, killed=88, survived=12, timeout=0, non_viable=0,
        untested=0, total=100)
    fake = _SeqEngine([selective, full], ["FULL_OP"])
    _set_engine(monkeypatch,
                        lambda lang, config: fake)
    agent = _make_agent(tmp_path)
    agent.framework_baseline_measure()
    s = agent.session
    assert s.mutation_mode == "full"
    assert s.switch_iteration == 0
    assert s.engine_runs == 2
    assert s.framework_prep_runs == 1
    assert s.mutation_measurements == 0
    assert s.measure_mutation_calls == 0


# Guards shared with the coverage phase: read loop breaker, collection error,
# timeout advisory, stall hint and the placeholder-explanation validator.

def test_read_source_loop_breaker_prefix_escalates(tmp_path: Path) -> None:
    """Repeated whole-file re-reads escalate to the loop-breaker prefix at
    the same threshold as the coverage phase's read_source."""
    agent = _make_agent(tmp_path)
    agent.read_source()
    agent.read_source()
    agent.read_source()
    fourth = agent.read_source()
    assert "LOOP BREAKER" in fourth
    assert "measure_mutation (when the passing suite is unmeasured)" in fourth


def test_read_tests_missing_file_names_the_working_file(tmp_path: Path) -> None:
    """read_tests on a not-yet-written working file returns the shared
    no-file message."""
    agent = _make_agent(tmp_path)
    out = agent.read_tests()
    assert "[FRAMEWORK ERROR]" in out
    assert "required suite from the preceding phase" in out
    assert "``edit_tests`` remains unavailable" in out
    assert "Retry ``read_tests`` once" in out


def test_run_tests_collection_error_promotes_specific_errors(
    monkeypatch, tmp_path: Path,
) -> None:
    """A collection error (exit 2, no test results) leads with the
    extracted error lines, the same view the coverage phase builds."""
    out_text = (
        "E   ImportError: cannot import name 'foo' from 'bar'\n"
        "collected 0 items / 1 error\n"
    )
    _install_backend(monkeypatch, _trr(2, out_text))
    agent = _make_agent(tmp_path)
    agent.target_path().write_text(
        "import missing_module\n", encoding="utf-8",
    )
    out = agent.run_tests()
    assert "[Discovery / compile error.]" in out
    assert "ImportError: cannot import name 'foo' from 'bar'" in out


def test_timeout_without_snapshot_returns_advisory(
    monkeypatch, tmp_path: Path,
) -> None:
    """A suite timeout without a best-green snapshot has nothing to revert
    to. The advisory claims no restore and the phase continues."""
    _install_backend(monkeypatch, _trr(timed_out=True))
    agent = _make_agent(tmp_path)
    agent.session.writes_done = 2
    assert agent.session.best_green.snapshot is None
    out = agent.run_tests()
    assert "TIMED OUT after" in out
    assert "Reverted" not in out
    assert agent.session.consecutive_timeouts == 1
    assert agent.session.early_stop_result is None


def test_measure_mutation_stall_hint_when_early_stop_disabled(
    monkeypatch, tmp_path: Path,
) -> None:
    """With allow_early_stop off and the streak past patience, the
    measurement summary carries the keep-iterating hint. With the stop on
    the stop directive replaces the hint."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "allow_early_stop", False)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "max_measurements_without_improvement", 2)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    results = tm.mutation.MutationMeasurement(
        score=50.0, killed=5, survived=5, timeout=0, non_viable=0,
        untested=0, total=10,
    )
    _install_fake_engine(monkeypatch, results)
    agent = _make_agent(tmp_path)
    agent.session.best_mutation_score = 50.0
    agent.session.measurements_without_improvement = 1
    out = agent.measure_mutation()
    assert agent.session.measurements_without_improvement == 2
    assert "keep iterating" in out
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "allow_early_stop", True)
    agent2 = _make_agent(tmp_path)
    agent2.session.best_mutation_score = 50.0
    agent2.session.measurements_without_improvement = 1
    out2 = agent2.measure_mutation()
    assert "keep iterating" not in out2


def test_validator_placeholder_explanation_uses_the_shared_key(
    tmp_path: Path,
) -> None:
    """A placeholder explanation is rejected by the same content gate the
    coverage validator applies."""
    agent = _make_agent(tmp_path)
    parsed = tm.MutationTestResult(
        mutation_test_file_name="test_foo_mutation.py",
        explanation="Brief explanation of which surviving mutants you targeted and how.",
        targeted_mutants=["m.py:1:AOR"],
    )
    out = agent.validate_result_fields(parsed)
    assert out is not None
    assert "looks like placeholder text" in out
    assert "MutationTestResult.explanation" in out


# Best-green at phase end: an unmeasured or worse final edit is not delivered.

def _seed_best_green(agent, content: str) -> None:
    agent.session.best_green.snapshot = {
        "content": content, "mutation": 70.0, "coverage": 85.0,
        "test_count": 1, "exec_time_ms": 100.0, "time": 1.0,
    }


def test_finalize_reverts_a_green_unmeasured_final_edit(tmp_path: Path) -> None:
    """A final edit that ran green but was never measured ranks below any
    measured snapshot, so the phase-end restore delivers the best measured
    green file."""
    agent = _make_agent(tmp_path)
    s = agent.session
    best = "def test_best():\n    assert True\n"
    _seed_best_green(agent, best)
    agent.target_path().write_text(
        "def test_unmeasured():\n    assert True\n", encoding="utf-8",
    )
    s.last_tests_passed = True
    s.tests_changed_since_run_tests = False
    s.tests_changed_since_mutation = True
    agent._settle_best_green(agent.target_path(), log_fn=agent.log)
    assert agent.target_path().read_text(encoding="utf-8") == best
    assert s.best_green.restored_from_green is True


def test_finalize_reverts_a_green_measured_equal_final_edit(tmp_path: Path) -> None:
    """A green final edit measured at the snapshot's score does not replace
    the earlier snapshot, which the phase-end restore returns."""
    import hashlib as _hashlib

    agent = _make_agent(tmp_path)
    s = agent.session
    _seed_best_green(agent, "def test_best():\n    assert True\n")
    live = "def test_live():\n    assert True\n"
    agent.target_path().write_text(live, encoding="utf-8")
    s.last_tests_passed = True
    s.tests_changed_since_run_tests = False
    s.tests_changed_since_mutation = False
    s.prev_mutation_score = 70.0
    s.last_suite_exec_ms = 100.0
    s.measured_suite_hash = _hashlib.sha256(live.encode("utf-8")).hexdigest()
    agent._settle_best_green(agent.target_path(), log_fn=agent.log)
    assert agent.target_path().read_text(encoding="utf-8") == (
        "def test_best():\n    assert True\n")
    assert s.best_green.restored_from_green is True


def test_finalize_keeps_a_green_measured_better_final_edit(tmp_path: Path) -> None:
    """A green final edit measured strictly better than the snapshot is the
    live copy the phase delivers."""
    import hashlib as _hashlib

    agent = _make_agent(tmp_path)
    s = agent.session
    _seed_best_green(agent, "def test_best():\n    assert True\n")
    live = "def test_live():\n    assert True\n"
    agent.target_path().write_text(live, encoding="utf-8")
    s.last_tests_passed = True
    s.tests_changed_since_run_tests = False
    s.tests_changed_since_mutation = False
    s.prev_mutation_score = 80.0
    s.last_suite_exec_ms = 100.0
    s.measured_suite_hash = _hashlib.sha256(live.encode("utf-8")).hexdigest()
    agent._settle_best_green(agent.target_path(), log_fn=agent.log)
    assert agent.target_path().read_text(encoding="utf-8") == live
    assert s.best_green.restored_from_green is False


def test_finalize_reverts_a_green_measured_worse_final_edit(tmp_path: Path) -> None:
    """The phase-end restore reverts a green final edit whose fresh score is
    below the snapshot's."""
    import hashlib as _hashlib

    agent = _make_agent(tmp_path)
    s = agent.session
    best = "def test_best():\n    assert True\n"
    _seed_best_green(agent, best)
    worse = "def test_worse():\n    assert True\n"
    agent.target_path().write_text(worse, encoding="utf-8")
    s.last_tests_passed = True
    s.tests_changed_since_run_tests = False
    s.tests_changed_since_mutation = False
    s.prev_mutation_score = 50.0
    s.last_suite_exec_ms = 100.0
    s.measured_suite_hash = _hashlib.sha256(worse.encode("utf-8")).hexdigest()
    agent._settle_best_green(agent.target_path(), log_fn=agent.log)
    assert agent.target_path().read_text(encoding="utf-8") == best


def test_finalize_reverts_when_the_flag_cleared_without_a_measurement(
    tmp_path: Path,
) -> None:
    """An engine failure (no mutant attempted) clears the changed flag
    without scoring the current bytes. The phase-end gate does not credit
    the earlier score to that file and reverts to the best measured green."""
    import hashlib as _hashlib

    agent = _make_agent(tmp_path)
    s = agent.session
    best = "def test_best():\n    assert True\n"
    _seed_best_green(agent, best)
    unmeasured = "def test_after_engine_failure():\n    assert True\n"
    agent.target_path().write_text(unmeasured, encoding="utf-8")
    s.last_tests_passed = True
    s.tests_changed_since_run_tests = False
    s.tests_changed_since_mutation = False  # cleared by the failure path
    s.prev_mutation_score = 70.0            # belongs to earlier bytes
    s.last_suite_exec_ms = 50.0
    s.measured_suite_hash = _hashlib.sha256(best.encode("utf-8")).hexdigest()
    agent._settle_best_green(agent.target_path(), log_fn=agent.log)
    assert agent.target_path().read_text(encoding="utf-8") == best


# At entry the working file starts as the coverage suite.

def test_the_loop_enters_with_the_inherited_suite_green(tmp_path, monkeypatch):
    """The framework ran the inherited suite green before the loop, so the
    loop enters green, as the coverage phase does. The start's measurement
    counts as the loop's first: a final answer right after it meets
    the goal and stagnation rules, not a refusal for lack of a measurement
    or as a suite that failed run_tests. A measure call while editing is
    told that the file is already measured."""
    src = tmp_path / "src" / "foo.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "src" / "pyproject.toml").write_text("", encoding="utf-8")
    seen = {}

    def fake_run_phase(self, *, project_name, user_prompt):
        seen["entered_green"] = self.session.last_tests_passed
        # The model's start: the designated read, then the start's measurement.
        self.session.has_read = True
        self.measure_mutation()
        parsed = tm.MutationTestResult(
            mutation_test_file_name=self.deps.mutation_test_file_name,
            explanation="Nothing to add", targeted_mutants=["foo.py:2:ReplaceReturn"],
            mutation_score=41.3,
            status="success",
        )
        seen["finishing"] = self.validate_output(state={}, parsed=parsed)
        seen["stagnation"] = self.early_stop_reject_message()
        seen["measure"] = self.unbound_tool_reply("measure_mutation")
        seen["agent"] = self
        return parsed

    def fake_measure(self, framework_prep=False):
        self.session.prev_mutation_score = 41.3
        self.session.best_mutation_score = 41.3
        self.session.last_mutation_counts = (5, 7)
        self.session.tests_changed_since_mutation = False
        return "measured"

    monkeypatch.setattr(tm.MutationAgent, "run_phase", fake_run_phase)
    monkeypatch.setattr(tm.MutationAgent, "_measure_mutation_engine", fake_measure)
    monkeypatch.setattr(tm, "_seed_mutation_baseline_from_matrix", lambda _agent: False)
    logs_in(monkeypatch, tmp_path / "logs")
    tm.generate_mutation_tests(
        project_name="p", target_file_path=str(src),
        test_output_dir=str(tmp_path / "tests_out"),
        preseed={"content": "def test_cov():\n    assert True\n", "file_stem": "test_foo"},
        coverage_baseline_pct=88.5,
    )
    agent = seen["agent"]
    assert seen["entered_green"] is True
    assert seen["finishing"] == seen["stagnation"]
    assert seen["finishing"] != agent.tool_msg("validator_no_mutation_measurement")
    assert seen["finishing"] != agent.tool_msg("validator_not_green")
    assert "has not passed run_tests" not in seen["measure"]
    assert "already measured" in seen["measure"]


def test_entry_preseeds_working_file_and_best_green(tmp_path, monkeypatch):
    """generate_mutation_tests writes the coverage suite into the working
    file before the loop starts and initializes best-green with it, so the
    framework restores to the inherited suite on any regression."""
    src = tmp_path / "src" / "foo.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "src" / "pyproject.toml").write_text("", encoding="utf-8")
    captured = {}

    def fake_run_phase(self, *, project_name, user_prompt):
        captured["agent"] = self
        captured["user_prompt"] = user_prompt
        return tm.MutationTestResult(
            mutation_test_file_name=self.deps.mutation_test_file_name,
            explanation="x", targeted_mutants=[], mutation_score=None,
        )

    monkeypatch.setattr(tm.MutationAgent, "run_phase", fake_run_phase)

    def fake_measure(self, framework_prep=False):
        self.session.prev_mutation_score = 41.3
        self.session.last_mutation_counts = (5, 7)
        self.session.last_surviving_sample = "- mutant diff sample"
        self.session.tests_changed_since_mutation = False
        return "measured"

    monkeypatch.setattr(
        tm.MutationAgent, "_measure_mutation_engine", fake_measure,
    )
    for key, value in {"max_measurements_without_improvement": 10, "min_improvement_points": 0.5,
                       "minimum": 65}.items():
        set_phase_value(monkeypatch, tm._AGENT_CONFIG, key, value)
    coverage_suite = "def test_cov():\n    assert True\n"
    logs_in(monkeypatch, tmp_path / "logs")
    tm.generate_mutation_tests(
        project_name="p",
        target_file_path=str(src),
        test_output_dir=str(tmp_path / "tests_out"),
        preseed={"content": coverage_suite, "file_stem": "test_foo"},
        coverage_baseline_pct=88.5,
    )
    # The prompt states the stagnation threshold and its exception.
    assert ("If the measurement stalls (no gain of at least 0.5\n"
            "  points across 10 consecutive measurements; while the mutation score\n"
            "  is below 65%, any gain counts)") in captured["user_prompt"]
    agent = captured["agent"]
    working = (Path(agent.deps.test_output_dir)
               / agent.deps.mutation_test_file_name)
    assert working.read_text(encoding="utf-8") == coverage_suite
    snap = agent.session.best_green.snapshot
    assert snap is not None
    assert snap["content"] == coverage_suite
    assert snap["coverage"] == 88.5
    assert "already on disk in your working file" in captured["user_prompt"]
    assert "41.3% (5 killed / 7 surviving)" in captured["user_prompt"]
    # The survivor sample comes with the start's measurement, not the
    # task prompt.
    assert "- mutant diff sample" not in captured["user_prompt"]
    assert "$frozen_test_file_name" not in captured["user_prompt"]
    # As in the coverage phase, the inherited suite is the designated first
    # read.
    assert agent.designated_read_name == "read_tests"
    # The baseline ran as framework preparation. The preparation counter
    # records it, the model-facing counters stay zero, and the phase clock
    # runs for the duration.
    assert agent.session.framework_prep_runs == 1
    assert agent.session.measure_mutation_calls == 0
    assert agent.session.mutation_measurements == 0
    assert agent.session.session_tool_calls == 0
    assert agent.session.phase_clock_start is not None
    assert agent.session.framework_prep_seconds >= 0.0


def test_entry_baseline_stop_stamps_phase_cost(tmp_path, monkeypatch):
    """A phase that ends at the baseline never reaches run_phase, so the
    entry records the result itself: the clock span, zero tokens and the
    inherited file's counts."""
    src = tmp_path / "src" / "foo.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "src" / "pyproject.toml").write_text("", encoding="utf-8")

    def fake_seed(agent):
        agent.session.early_stop_result = tm.MutationTestResult(
            mutation_test_file_name=agent.deps.mutation_test_file_name,
            explanation="baseline at goal", targeted_mutants=[],
            mutation_score=100.0, status="success",
        )
        agent.session.early_stop_reason = "mutation_goal_met"
        return True

    monkeypatch.setattr(
        tm, "_seed_mutation_baseline_from_matrix", fake_seed,
    )
    coverage_suite = "def test_cov():\n    assert True\n"
    logs_in(monkeypatch, tmp_path / "logs")
    result = tm.generate_mutation_tests(
        project_name="p",
        target_file_path=str(src),
        test_output_dir=str(tmp_path / "tests_out"),
        preseed={"content": coverage_suite, "file_stem": "test_foo"},
        coverage_baseline_pct=88.5,
    )
    assert result.status == "success"
    assert result.agent_duration_s is not None
    assert result.input_tokens == 0
    assert result.output_tokens == 0
    assert result.test_count_mutation == 1
    assert result.test_loc_mutation == 2


def test_entry_without_preseed_fails_before_model_call(tmp_path, monkeypatch):
    src = tmp_path / "src" / "foo.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def f():\n    return 1\n", encoding="utf-8")
    captured = {}

    def fake_run_phase(self, *, project_name, user_prompt):
        captured["agent"] = self
        return tm.MutationTestResult(
            mutation_test_file_name=self.deps.mutation_test_file_name,
            explanation="x", targeted_mutants=[], mutation_score=None,
        )

    monkeypatch.setattr(tm.MutationAgent, "run_phase", fake_run_phase)

    def fake_measure(self, framework_prep=False):
        self.session.tests_changed_since_mutation = False
        return "measured"

    monkeypatch.setattr(
        tm.MutationAgent, "_measure_mutation_engine", fake_measure,
    )
    with pytest.raises(RuntimeError, match="nonempty validated suite"):
        tm.generate_mutation_tests(
            project_name="p",
            target_file_path=str(src),
            test_output_dir=str(tmp_path / "tests_out"),
            preseed=None,
        )
    assert captured == {}
    assert not (tmp_path / "tests_out" / "test_foo.py").exists()


@pytest.mark.parametrize("coverage, stated", [
    (88.56, " It measured 88.5% line coverage. Strengthen it in place."),
    (None, " Its line coverage was not measured. Strengthen it in place."),
])
def test_entry_states_the_inherited_line_coverage_or_its_absence(
        tmp_path, monkeypatch, coverage, stated):
    """The mutation prompt states the coverage phase's line coverage floored
    to one decimal, and says it was not measured instead of "unknown%"."""
    src = tmp_path / "src" / "foo.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "src" / "pyproject.toml").write_text("", encoding="utf-8")
    captured = {}

    def fake_run_phase(self, *, project_name, user_prompt):
        captured["user_prompt"] = user_prompt
        return tm.MutationTestResult(
            mutation_test_file_name=self.deps.mutation_test_file_name,
            explanation="x", targeted_mutants=[], mutation_score=None,
        )

    def fake_measure(self, framework_prep=False):
        self.session.prev_mutation_score = 41.3
        self.session.last_mutation_counts = (5, 7)
        self.session.last_surviving_sample = "- mutant diff sample"
        self.session.tests_changed_since_mutation = False
        return "measured"

    monkeypatch.setattr(tm.MutationAgent, "run_phase", fake_run_phase)
    monkeypatch.setattr(tm.MutationAgent, "_measure_mutation_engine", fake_measure)
    logs_in(monkeypatch, tmp_path / "logs")
    tm.generate_mutation_tests(
        project_name="p",
        target_file_path=str(src),
        test_output_dir=str(tmp_path / "tests_out"),
        preseed={"content": "def test_cov():\n    assert True\n", "file_stem": "test_foo"},
        coverage_baseline_pct=coverage,
    )
    prompt = captured["user_prompt"].replace("\n", " ")
    assert stated in prompt
    assert "unknown%" not in prompt


def test_a_coverage_tool_refusal_at_the_baseline_stands_when_the_loop_starts(
        tmp_path, monkeypatch):
    """The coverage tool's refusal at the baseline is not replaced by the
    inherited suite's green, so the loop starts by editing."""
    src = tmp_path / "src" / "foo.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "src" / "pyproject.toml").write_text("", encoding="utf-8")
    seen = {}

    def fake_run_phase(self, *, project_name, user_prompt):
        seen.update(passed=self.session.last_tests_passed, refused_by=self.refused_by())
        return tm.MutationTestResult(
            mutation_test_file_name=self.deps.mutation_test_file_name,
            explanation="x", targeted_mutants=[], mutation_score=None,
        )

    def fake_measure(self, framework_prep=False):
        self.session.prev_mutation_score = 41.3
        self.session.last_mutation_counts = (5, 7)
        self.session.last_surviving_sample = "- mutant diff sample"
        self.session.tests_changed_since_mutation = False
        self.record_refusal("coverage", "the coverage tool refused the suite")
        return "measured"

    monkeypatch.setattr(tm.MutationAgent, "run_phase", fake_run_phase)
    monkeypatch.setattr(tm.MutationAgent, "_measure_mutation_engine", fake_measure)
    logs_in(monkeypatch, tmp_path / "logs")
    tm.generate_mutation_tests(
        project_name="p",
        target_file_path=str(src),
        test_output_dir=str(tmp_path / "tests_out"),
        preseed={"content": "def test_cov():\n    assert True\n", "file_stem": "test_foo"},
        coverage_baseline_pct=88.56,
    )
    assert seen == {"passed": False, "refused_by": "coverage"}


def test_a_suite_red_only_while_measured_is_named_so_and_counts_as_a_failed_run(
    monkeypatch, tmp_path: Path,
) -> None:
    """When the normal run passed but the measuring engine refused the suite
    as failing, the model is told so in general words without the engine's
    reason, and the attempt counts as a failed run."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    results = tm.mutation.MutationMeasurement.empty(
        reason="the suite fails without a mutant", suite_failed=True)
    _install_fake_engine(monkeypatch, results)
    agent = _make_agent(tmp_path)
    output = agent.measure_mutation()
    assert agent.tool_msg("measure_mutation_red_when_measured") in output
    assert "engine's reason" not in output and "0 mutants were tested" not in output
    assert agent.session.best_green.consecutive_failed_runs == 1


def test_a_suite_red_only_while_measured_names_the_tests_the_tool_gives(
    monkeypatch, tmp_path: Path,
) -> None:
    """When the mutation tool names the tests that failed without a mutant,
    the reply names them, and the attempt counts as a failed run."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    results = tm.mutation.MutationMeasurement.empty(
        reason="the suite fails without a mutant", suite_failed=True,
        stopping_tests=("tests/test_foo_mutation.py::test_loaded",))
    _install_fake_engine(monkeypatch, results)
    agent = _make_agent(tmp_path)
    output = agent.measure_mutation()
    assert "tests/test_foo_mutation.py::test_loaded" in output
    assert "did not name the failing" not in output
    assert agent.session.best_green.consecutive_failed_runs == 1


def test_a_suite_red_under_the_coverage_tool_names_its_tests_and_counts_as_a_failed_run(
    monkeypatch, tmp_path: Path,
) -> None:
    """The measurement runs the coverage tool after the green normal run: a
    test that fails only under it is named, the mutation tool does not
    start, the suite reads red and the attempt counts as a failed run."""
    from testuary.languages.generic import TestFailure

    failing = "test_foo_mutation.py::test_loaded"
    red = _trr(1, f"FAILED {failing} - ImportError: loaded twice")._replace(
        failures=(TestFailure(failing, "failed", "ImportError: loaded twice"),))
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"), under_coverage=red)
    engine = _install_fake_engine(monkeypatch, tm.mutation.MutationMeasurement.empty(
        reason="not reached"))
    agent = _make_agent(tmp_path)
    output = agent.measure_mutation()
    assert engine.run_calls == 0
    assert "failed while coverage was being measured" in output and failing in output
    # The suite passed run_tests: the refusal is the coverage tool's.
    assert agent.refused_by() == "coverage"
    again = agent.measure_mutation()
    assert "the coverage tool could not measure" in again and "FAILING" not in again
    assert engine.run_calls == 0
    assert agent.session.best_green.consecutive_failed_runs == 1
    assert agent.session.last_tests_passed is False


def _logged_fields(monkeypatch, agent) -> list:
    """The fields each of the agent's log lines carries, beside its level."""
    logged = []
    monkeypatch.setattr(agent, "log", lambda message, *args, **fields: logged.append(
        {key: value for key, value in fields.items() if key != "level"}))
    return logged


def test_a_measurement_the_mutation_tool_refused_records_the_tests_that_stopped_it(
    monkeypatch, tmp_path: Path,
) -> None:
    """The refused measurement's log line carries the tool and the tests it
    named as fields of the line's record, which the results page reads."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    _install_fake_engine(monkeypatch, tm.mutation.MutationMeasurement.empty(
        reason="the suite fails without a mutant", suite_failed=True,
        stopping_tests=("tests/test_foo_mutation.py::test_loaded",)))
    agent = _make_agent(tmp_path)
    logged = _logged_fields(monkeypatch, agent)
    agent.measure_mutation()
    assert {"tool": "mutation",
            "stopping_tests": ["tests/test_foo_mutation.py::test_loaded"]} in logged


def test_a_measurement_the_coverage_tool_refused_records_the_tests_that_stopped_it(
    monkeypatch, tmp_path: Path,
) -> None:
    from testuary.languages.generic import TestFailure

    failing = "test_foo_mutation.py::test_loaded"
    red = _trr(1, f"FAILED {failing} - ImportError: loaded twice")._replace(
        failures=(TestFailure(failing, "failed", "ImportError: loaded twice"),))
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"), under_coverage=red)
    _install_fake_engine(monkeypatch, tm.mutation.MutationMeasurement.empty(
        reason="not reached"))
    agent = _make_agent(tmp_path)
    logged = _logged_fields(monkeypatch, agent)
    agent.measure_mutation()
    assert {"tool": "coverage", "stopping_tests": [failing]} in logged


def test_a_coverage_run_without_a_test_result_stops_the_measurement_as_the_tools_refusal(
    monkeypatch, tmp_path: Path,
) -> None:
    """A run under the coverage tool that collected nothing, or ended
    without counting a failure, after the green normal run is the coverage
    tool's refusal: the mutation tool does not start, the model sees the
    run's output, the suite needs an edit, the attempt counts and the record
    says the tool named no test."""
    empty = _trr(4, "ERROR: the run ended under the coverage tool")._replace(failed=0)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"), under_coverage=empty)
    engine = _install_fake_engine(monkeypatch, tm.mutation.MutationMeasurement.empty(
        reason="not reached"))
    agent = _make_agent(tmp_path)
    logged = _logged_fields(monkeypatch, agent)
    output = agent.measure_mutation()
    assert engine.run_calls == 0
    assert "ended without a test result" in output
    assert "the run ended under the coverage tool" in output
    assert agent.session.best_green.consecutive_failed_runs == 1
    assert agent.session.last_tests_passed is False and agent.refused_by() == "coverage"
    assert {"tool": "coverage", "stopping_tests": []} in logged


def test_a_coverage_run_that_does_not_finish_is_the_tools_refusal_not_a_slow_suite(
    monkeypatch, tmp_path: Path,
) -> None:
    """The normal run finished in time and only the run under the coverage
    tool did not: at the framework's baseline, before any edit, the phase
    does not end as an inherited suite too slow, and no suite timeout
    counts. The coverage tool's refusal names the running test."""
    running = "tests/test_foo_mutation.py::test_slow"
    slow = _trr(timed_out=True)._replace(active_at_end=running)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"), under_coverage=slow)
    engine = _install_fake_engine(monkeypatch, tm.mutation.MutationMeasurement.empty(
        reason="not reached"))
    agent = _make_agent(tmp_path)
    logged = _logged_fields(monkeypatch, agent)
    agent.framework_baseline_measure()
    assert engine.run_calls == 0
    assert agent.session.writes_done == 0
    assert agent.session.early_stop_reason is None
    assert agent.session.consecutive_timeouts == 0
    assert agent.refused_by() == "coverage"
    assert "did not finish within" in agent.session.last_mutation_result
    assert running in agent.session.last_mutation_result
    assert {"tool": "coverage", "stopping_tests": [running]} in logged


def _under_coverage(monkeypatch, tmp_path, covered):
    """measure_mutation on a green normal run whose run under the coverage
    tool gives ``covered``: the agent, its reply, the session fields the call
    wrote, and the engine, which must not start."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"), under_coverage=covered)
    engine = _install_fake_engine(monkeypatch, tm.mutation.MutationMeasurement.empty(
        reason="not reached"))
    agent = _make_agent(tmp_path)
    output, written = written_fields(agent.session, agent.measure_mutation)
    assert engine.run_calls == 0
    return agent, output, written


def _coverage_refusal_fields(reply):
    """The fields a measurement writes when the coverage tool refuses after a
    green normal run: the suite reads red with the refusal as the tool's,
    the reply answers the unchanged file, and the attempt counts as a
    failed run."""
    return {
        "measure_mutation_calls": 1, "last_suite_exec_ms": 100.0,
        "last_run_tests_result": reply, "tests_changed_since_run_tests": False,
        "refusal": ("coverage", reply), "last_mutation_result": reply,
        "tests_changed_since_mutation": False,
        "best_green": agent_helpers.BestGreenState(consecutive_failed_runs=1),
    }


@pytest.mark.parametrize("active", [None, "tests/test_foo_mutation.py::test_slow"])
def test_the_coverage_tools_timeout_reply_and_the_fields_it_writes(
    monkeypatch, tmp_path: Path, active,
) -> None:
    """A run under the coverage tool that does not finish after a green
    normal run answers the timeout reply, with the running test when the
    journal names it, and writes the refusal's fields only."""
    agent, output, written = _under_coverage(
        monkeypatch, tmp_path, _trr(timed_out=True)._replace(active_at_end=active))
    timeout = int(tm._AGENT_CONFIG["suite_run_timeout_seconds"])
    reply = (agent.tool_msg("measure_mutation_timeout_under_coverage_named",
                            timeout=timeout, test=active) if active
             else agent.tool_msg("measure_mutation_timeout_under_coverage", timeout=timeout))
    assert output == reply
    assert written == _coverage_refusal_fields(output)


def test_the_coverage_tools_red_reply_and_the_fields_it_writes(
    monkeypatch, tmp_path: Path,
) -> None:
    """A run under the coverage tool that fails after a green normal run
    answers the red reply with the run's canonical view."""
    from testuary.languages.generic import TestFailure

    failing = "test_foo_mutation.py::test_loaded"
    red = _trr(1, f"FAILED {failing} - ImportError: loaded twice")._replace(
        failures=(TestFailure(failing, "failed", "ImportError: loaded twice"),))
    agent, output, written = _under_coverage(monkeypatch, tmp_path, red)
    assert output == agent.tool_msg("measure_mutation_red_under_coverage",
                                    tests_output=agent._canonical_view(red, None, 1))
    assert written == _coverage_refusal_fields(output)


def test_the_coverage_tools_reply_without_a_test_result_and_the_fields_it_writes(
    monkeypatch, tmp_path: Path,
) -> None:
    """A run under the coverage tool that ends without a test result after a
    green normal run answers with the tail of its output."""
    empty = _trr(4, "ERROR: the run ended under the coverage tool")._replace(failed=0)
    agent, output, written = _under_coverage(monkeypatch, tmp_path, empty)
    assert output == agent.tool_msg(
        "measure_mutation_unmeasured_under_coverage",
        output="ERROR: the run ended under the coverage tool")
    assert written == _coverage_refusal_fields(output)


def test_the_coverage_tools_reply_when_its_counts_cannot_be_read(
    monkeypatch, tmp_path: Path,
) -> None:
    """A run under the coverage tool whose coverage counts the tool could not
    read answers with its own reply, in the language's words for C, starts
    no mutant and writes the refusal's fields only."""
    unread = _trr(-1, "gcovr: negative count")._replace(
        failed=0, collected=3, passed=3,
        diagnostics=(RunDiagnostic("coverage_counts_inconsistent", "negative count"),))
    words = c_prompts("test_mutation")
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"), under_coverage=unread)
    engine = _install_fake_engine(monkeypatch, tm.mutation.MutationMeasurement.empty(
        reason="not reached"))
    agent = _make_agent(tmp_path)
    agent.prompts = words
    output, written = written_fields(agent.session, agent.measure_mutation)
    assert engine.run_calls == 0
    assert output == (
        "[measure_mutation] The tests passed, but the coverage tool could not\n"
        "read the coverage counts their run left, so the suite cannot be\n"
        "measured and no mutant was tested.\n"
        "Counts that cannot be read usually mean that a test corrupts memory or relies "
        "on undefined behavior.\n"
        "If your last edit added or changed tests, check those first.\n"
        "Change the test so that it uses only valid values and memory, or remove it.\n"
        "\n"
        "WORKFLOW: edit the tests with edit_tests and call run_tests, then\n"
        "measure_mutation again.")
    assert written == _coverage_refusal_fields(output)


def test_the_coverage_tools_red_reply_wins_over_unreadable_counts(
    monkeypatch, tmp_path: Path,
) -> None:
    """A run under the coverage tool whose tests fail and whose coverage
    counts the tool could not read gets the red reply, in C's words too: its
    failing tests decide."""
    red = _trr(1, "1 failed")._replace(
        collected=2, passed=1,
        diagnostics=(RunDiagnostic("coverage_counts_inconsistent", "negative count"),))
    words = c_prompts("test_mutation")
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"), under_coverage=red)
    _install_fake_engine(monkeypatch, tm.mutation.MutationMeasurement.empty(
        reason="not reached"))
    agent = _make_agent(tmp_path)
    agent.prompts = words
    output, _ = written_fields(agent.session, agent.measure_mutation)
    assert output == agent.tool_msg("measure_mutation_red_under_coverage",
                                    tests_output=agent._canonical_view(red, None, 1))


class _UnimplementedEngine(_FakeEngine):
    """An engine whose run raises NotImplementedError."""

    def run(self, **kwargs) -> Any:
        self.run_calls += 1
        raise NotImplementedError("no mutation engine for this language")


def test_an_engine_that_cannot_run_answers_its_error_and_writes_no_measurement(
    monkeypatch, tmp_path: Path,
) -> None:
    """An engine run that raises NotImplementedError answers the error, on
    the per-call path, and records no engine run or score."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    engine = _UnimplementedEngine(None)
    _set_engine(monkeypatch, lambda lang, config: engine)
    agent = _make_agent(tmp_path)
    output, written = written_fields(agent.session, agent.measure_mutation)
    assert engine.run_calls == 1
    assert output == agent.tool_msg("measure_mutation_engine_not_implemented",
                                    exc="no mutation engine for this language")
    assert written == {"measure_mutation_calls": 1, "last_suite_exec_ms": 100.0}


def test_a_pooled_session_that_cannot_run_answers_its_error(
    monkeypatch, tmp_path: Path,
) -> None:
    """The pooled session's run that raises NotImplementedError answers the
    same reply as the per-call engine's."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))

    def unimplemented(**kwargs):
        raise NotImplementedError("no mutation engine for this language")

    session = SimpleNamespace(workers=2, use_http=False, run=unimplemented,
                              runner=SimpleNamespace(name="fake", log=SimpleNamespace(target=None),
                                                     full_operators=lambda: None))
    agent = _make_agent(tmp_path, mutation_session=session)
    output, written = written_fields(agent.session, agent.measure_mutation)
    assert output == agent.tool_msg("measure_mutation_engine_not_implemented",
                                    exc="no mutation engine for this language")
    assert written == {"measure_mutation_calls": 1, "last_suite_exec_ms": 100.0}


def test_a_switch_run_that_cannot_run_raises_its_error(monkeypatch, tmp_path: Path) -> None:
    """The catalogue switch's second engine run has no handler of its own: a
    NotImplementedError there leaves measure_mutation, unlike the first
    run's."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "full_after_selective_goal", True)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "selective_switch_threshold", 100)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    selective = tm.mutation.MutationMeasurement(
        score=100.0, killed=10, survived=0, timeout=0, non_viable=0,
        untested=0, total=10)

    class _SelectiveThenUnimplemented(_SeqEngine):
        def run(self, **kwargs):
            if self.ops_seen:
                raise NotImplementedError("no full catalogue")
            return super().run(**kwargs)

    engine = _SelectiveThenUnimplemented([selective], ["FULL_OP"])
    _set_engine(monkeypatch, lambda lang, config: engine)
    agent = _make_agent(tmp_path)
    with pytest.raises(NotImplementedError, match="no full catalogue"):
        agent.measure_mutation()
    assert engine.ops_seen == [None]


def test_a_measurement_the_mutation_tool_refused_leaves_the_suite_needing_an_edit(
    monkeypatch, tmp_path: Path,
) -> None:
    """The mutation tool's own refusal reads as the tool's: the suite needs
    an edit, the reply is the measurement's result, and measuring the
    unchanged file again is refused without starting the tool."""
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    engine = _install_fake_engine(monkeypatch, tm.mutation.MutationMeasurement.empty(
        reason="the suite fails without a mutant", suite_failed=True,
        stopping_tests=("tests/test_foo_mutation.py::test_loaded",)))
    agent = _make_agent(tmp_path)
    output = agent.measure_mutation()
    assert agent.session.last_tests_passed is False and agent.refused_by() == "mutation"
    assert agent.session.last_mutation_result == output
    runs = engine.run_calls
    again = agent.measure_mutation()
    assert "the mutation tool could not measure" in again
    assert engine.run_calls == runs

def test_a_run_that_tested_nothing_for_another_reason_keeps_its_message(
    monkeypatch, tmp_path: Path,
) -> None:
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    results = tm.mutation.MutationMeasurement.empty(reason="no mutants were generated")
    _install_fake_engine(monkeypatch, results)
    agent = _make_agent(tmp_path)
    output = agent.measure_mutation()
    assert "0 mutants were tested" in output and "no mutants were generated" in output
    assert agent.session.best_green.consecutive_failed_runs == 0



def test_the_goal_stop_log_reports_the_reached_score(monkeypatch, tmp_path: Path) -> None:
    """The stop's log reports the score that reached the goal."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 80)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "extra_measurements_after_goal", 0)
    agent = _make_agent(tmp_path)
    agent.session.best_mutation_score = 85.0
    agent.session.selective_switch_possible = False
    logs = []
    monkeypatch.setattr(agent, "log", lambda msg, *args, **kwargs: logs.append(msg))
    assert agent._maybe_set_mutation_stop(source_tool="t") is not None
    assert any("goal reached (85.0% >= 80%)" in line for line in logs)
    assert not any("first measurement" in line for line in logs)


def test_a_snapshot_restore_that_fails_is_logged(monkeypatch, tmp_path: Path) -> None:
    """A best-green restore that raises returns False, and the log says
    why."""
    agent = _make_agent(tmp_path)
    agent.session.best_green.snapshot = SimpleNamespace()

    def restore(*args, **kwargs):
        raise OSError("the disk is full")

    monkeypatch.setattr(agent_helpers, "restore_best_green", restore)
    logs = []
    monkeypatch.setattr(agent, "log", lambda msg, *args, **kwargs: logs.append(msg))
    assert agent._restore_snapshot(agent.target_path()) is False
    assert any("OSError: the disk is full" in line for line in logs)


class _WithSurvivors(tm.mutation.MutationMeasurement):
    """A measurement whose surviving mutants render through the language
    layer's own renderer, as a real run's do."""

    def survivors_text(self, max_mutants=10, max_chars=240):
        survivors = [{"id": "m2", "line": 1, "diff": "-    return 1\n+    return 2\n",
                      "operator": "core/NumberReplacer"}]
        return tm.mutation.render_survivors(
            survivors, max_mutants=max_mutants, max_chars=max_chars)


def test_the_main_measurement_reply_keeps_its_bytes(monkeypatch, tmp_path: Path) -> None:
    """measure_mutation's reply after a second measurement, the model's input
    on the main path: the summary with the change against the previous best,
    the goal and the streak, and the surviving mutants. Pinned byte for byte,
    so a change of its text is a decision, not a side effect."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 100)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    first = tm.mutation.MutationMeasurement(
        score=60.0, killed=6, survived=4, timeout=0, non_viable=0, untested=0, total=10)
    second = _WithSurvivors(
        score=75.0, killed=6, survived=2, timeout=0, non_viable=0, untested=0, total=8)
    engine = _SeqEngine([first, second], None)
    _set_engine(monkeypatch, lambda lang, config: engine)
    agent = _make_agent(tmp_path)
    agent.measure_mutation()
    agent.session.tests_changed_since_mutation = True
    reply = agent.measure_mutation()
    assert reply == (
        'Mutation score: 75.0% (killed 6/8; survived 2) (prev best: 60.0%, delta: +15.0pp)\n'
        'Goal: 100%. Stagnation: 0/10.\n'
        '\n'
        '--- Surviving mutants (sample) ---\n'
        '--- mutant m2 (line 1) ---\n'
        '-    return 1\n'
        '+    return 2'
    )


def test_a_fractional_goal_is_stated_as_set_in_the_measurement_reply(
        monkeypatch, tmp_path: Path) -> None:
    """The goal is a float setting: the reply states 87.5 as set, not
    rounded."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 87.5)
    _install_backend(monkeypatch, _trr(0, "3 passed in 0.1s"))
    engine = _SeqEngine([tm.mutation.MutationMeasurement(
        score=60.0, killed=6, survived=4, timeout=0, non_viable=0, untested=0, total=10)], None)
    _set_engine(monkeypatch, lambda lang, config: engine)
    assert "Goal: 87.5%." in _make_agent(tmp_path).measure_mutation()
