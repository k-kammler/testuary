"""``CoverageAgent.measure_coverage`` paths that return early or from the
cache, testable without a mocked full pytest and coverage run.
test_test_coverage_measure_full.py covers the full runs."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent_fixtures import coverage_agent, project_agent, run_result
from agent_prompts import c_prompts, python_prompts
from langadapter.runs import RunDiagnostic

from offered_tools import offered_tools
from testuary import prompts as prompts_module
from testuary.agent import coverage_agent as tc
from phase_values import set_phase_value
from session_fields import written_fields
import pytest

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

# The diagnostic of coverage counts the coverage tool could not read.
_UNREAD = (RunDiagnostic("coverage_counts_inconsistent", "negative count"),)


def _trr(*, collected=1, **fields):
    """A run_result in which all ``collected`` tests passed, with the output "out"."""
    return run_result(collected=collected, passed=collected, output="out", **fields)


def _install_backend(monkeypatch, *, cov=None, counter=None):
    # A fake backend serves measure_coverage's prelude and coverage run: the
    # coverage call (measure_coverage_of set) returns ``cov``, the prelude's
    # correctness call returns green.
    backend = tc.languages.get_backend("python")

    def _run(test_paths, source_dir, *, timeout,
             measure_coverage_of=None, instrument_target=None,
             fail_fast=False, tb="short", record_leaks=False):
        if counter is not None:
            counter["n"] += 1
        if measure_coverage_of is not None:
            return cov if cov is not None else _trr(line_coverage=90.0)
        return _trr(ok=True)

    def _delivered(test_paths, source_dir, *, timeout, reverse=False):
        from testuary.languages.generic import DeliveredCase, DeliveredRun

        return DeliveredRun(
            (DeliveredCase("test_x", "test_x", "passed"),), (), None, "out",
            reverse=reverse,
        )

    monkeypatch.setattr(
        tc.languages, "get_backend",
        lambda lang: SimpleNamespace(
            run_test_files=_run,
            find_target_test_file=backend.find_target_test_file,
            test_metrics_observation=backend.test_metrics_observation,
            run_as_delivered=_delivered,
        ),
    )


def _make_agent(tmp_path: Path) -> "tc.CoverageAgent":
    return project_agent(tmp_path, test_source="def test_x(): assert True\n")


def _accounted_call(agent, invoke):
    calls = agent.session.session_tool_calls
    output = invoke()
    assert agent.session.session_tool_calls == calls + 1
    return output


def test_a_measured_run_that_fails_records_no_coverage(monkeypatch, tmp_path):
    """The prelude was green, but the measured run itself failed (a test
    that fails under the instrument): its coverage is not recorded and the
    failure detail is returned."""
    red = _trr(ok=False, returncode=1, line_coverage=40.0)._replace(
        passed=0, failed=1,
        output="FAILED tests/test_foo.py::test_x - assert 1 == 2\n1 failed in 0.1s\n")
    _install_backend(monkeypatch, cov=red)
    agent = _make_agent(tmp_path)
    output = agent.measure_coverage()
    assert "passed in the normal run but\nfailed while coverage was being measured" in output
    assert "TESTS: FAILED | collected 1 | passed 0 | failed 1" in output
    assert "Coverage Summary" not in output and "Line coverage: 40" not in output
    assert agent.session.best_coverage_pct is None


def test_a_measured_run_that_fails_returns_the_loop_to_editing(monkeypatch, tmp_path):
    """After a red measured run the suite reads red, as after a red
    run_tests: the next turn offers the edit tool and not the measurement,
    and measuring the unchanged file again is refused as known red."""
    red = _trr(ok=False, returncode=1, line_coverage=40.0)._replace(
        passed=0, failed=1,
        output="FAILED tests/test_foo.py::test_x - assert 1 == 2\n1 failed in 0.1s\n")
    _install_backend(monkeypatch, cov=red)
    agent = _make_agent(tmp_path)
    agent.session.has_read = True
    agent.session.initial_measured = True  # past the start's measurement
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = True
    agent.session.tests_changed_since_measure["coverage"] = True
    agent.measure_coverage()
    assert agent.session.last_tests_passed is False
    assert "tests=red" in agent.build_state_marker()
    offered = [tool.__name__ for tool in offered_tools(agent)]
    assert "edit_file" in offered and "measure_coverage" not in offered
    agent.session.tests_changed_since_measure["coverage"] = True
    assert agent.measure_coverage().startswith("REFUSED: this measure_coverage call is invalid")
    assert agent.session.run_tests_calls == 1


def test_the_replies_after_a_refused_measurement_name_the_coverage_tool(monkeypatch, tmp_path):
    """The suite passed run_tests and only the coverage tool's run failed, so
    while that refusal is the latest result the replies name the tool, not a
    failing run_tests. A run of the edited file ends it."""
    red = _trr(ok=False, returncode=1, line_coverage=40.0)._replace(
        passed=0, failed=1,
        output="FAILED tests/test_foo.py::test_x - assert 1 == 2\n1 failed in 0.1s\n")
    _install_backend(monkeypatch, cov=red)
    agent = _make_agent(tmp_path)
    agent.session.has_read = True
    agent.session.initial_measured = True
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = True
    agent.session.tests_changed_since_measure["coverage"] = True
    agent.measure_coverage()
    assert agent.refused_by() == "coverage"
    offer = agent.unbound_tool_reply("measure_coverage")
    assert "the coverage tool could not measure the working file" in offer
    assert "has not passed run_tests" not in offer
    agent.session.tests_changed_since_measure["coverage"] = True
    again = agent.measure_coverage()
    assert "the coverage tool could not measure" in again and "FAILING" not in again
    assert "test_x" in again
    cached = agent.run_tests()
    assert "REJECTED: no new run" in cached
    assert "the coverage tool then could not" in cached
    state = agent.working_state_block(None, "coverage")
    assert "Last run_tests:    passed; the coverage tool then could not measure" in state
    assert "NOT MEASURABLE" in state and "RED or unknown" not in state
    agent.session.tests_changed_since_run_tests = True
    agent.run_tests()
    assert agent.session.last_tests_passed is True and agent.refused_by() == ""


def test_a_stagnant_measurement_shows_no_suite_report(monkeypatch, tmp_path):
    """A measurement long into stagnation reports the coverage and
    nothing about the suite's per-test redundancy or cost, though the backend
    could give per-test evidence. The suite stays as it was."""
    _install_backend(monkeypatch, cov=_trr(line_coverage=70.0, branch_coverage=60.0))
    fake = tc.languages.get_backend("python")
    fake.collect_eviction_data = lambda *args, **kwargs: SimpleNamespace(
        per_test_tokens={"test_x": frozenset({"L:foo:1"}), "test_y": frozenset({"L:foo:1"})},
        per_test_time_ms={"test_x": 2.0, "test_y": 3.0}, suite_time_ms=5.0, includes_kills=True,
    )
    monkeypatch.setattr(tc.languages, "get_backend", lambda lang: fake)
    agent = _make_agent(tmp_path)
    for key, value in {"goal": 100, "restore_count": 999,
                       "max_measurements_without_improvement": 1000}.items():
        set_phase_value(monkeypatch, tc._AGENT_CONFIG, key, value)
    agent.session.best_coverage_pct = 90.0
    agent.session.best_branch_coverage_pct = 90.0
    agent.session.measurements_without_improvement = 10
    test_file = Path(agent.deps.test_output_dir) / "test_foo.py"
    before = test_file.read_bytes()

    output = agent.measure_coverage()

    assert "Line coverage: 70.0%" in output
    assert "[suite report]" not in output and "add no unique" not in output
    assert test_file.read_bytes() == before
    assert agent.session.prev_coverage_pct == 70.0
    assert agent.session.prev_branch_coverage_pct == 60.0


def test_a_target_without_branches_is_reported_as_measured(monkeypatch, tmp_path):
    """Line coverage without branch coverage is a target without branches:
    the summary says so, never that coverage was not measurable."""
    _install_backend(monkeypatch, cov=_trr(line_coverage=100.0, branch_coverage=None))
    output = _make_agent(tmp_path).measure_coverage()
    assert "Line coverage: 100.0%" in output
    assert "The target has no branches." in output
    assert "not measurable" not in output and "None" not in output


def test_a_run_without_any_coverage_is_reported_as_not_measured(monkeypatch, tmp_path):
    _install_backend(monkeypatch, cov=_trr(line_coverage=None, branch_coverage=None))
    output = _make_agent(tmp_path).measure_coverage()
    assert "(coverage was not measurable this run)" in output
    assert "no branches" not in output


# measure_coverage: cache short-circuit

def test_measure_coverage_cache_short_circuit_no_subprocess(monkeypatch, tmp_path) -> None:
    """With ``tests_changed_since_measure['coverage']=False`` and a cached
    result, the cached payload returns and the no-improvement counter
    advances, without a subprocess."""
    sub_calls = {"n": 0}
    _install_backend(monkeypatch, counter=sub_calls)
    a = _make_agent(tmp_path)
    a.session.tests_changed_since_measure["coverage"] = False
    a.session.last_coverage_result = "Cached: line 88%, branch 70%"
    a.session.best_coverage_pct = 88.0
    out = _accounted_call(a, a.measure_coverage)
    assert sub_calls["n"] == 0  # cache short-circuits before any run
    assert "88" in out


def test_measure_coverage_advances_no_improvement_counter(monkeypatch, tmp_path) -> None:
    """The cached path increments ``measurements_without_improvement`` when a best exists,
    but only when ``count_repeated_measurements`` is on."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "count_repeated_measurements", "true")
    _install_backend(monkeypatch)
    a = _make_agent(tmp_path)
    a.session.tests_changed_since_measure["coverage"] = False
    a.session.last_coverage_result = "Cached"
    a.session.best_coverage_pct = 85.0
    starting = a.session.measurements_without_improvement
    _accounted_call(a, a.measure_coverage)
    assert a.session.measurements_without_improvement == starting + 1


# measure_coverage: missing-test-file early return

def test_measure_coverage_no_test_file_returns_directive(monkeypatch, tmp_path) -> None:
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src = src_dir / "foo.py"
    src.write_text("def f(): return 1\n", encoding="utf-8")
    empty_test_dir = tmp_path / "no_tests"
    empty_test_dir.mkdir()
    a = coverage_agent(tmp_path, test_output_dir=str(empty_test_dir))
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "tests passed"))
    out = _accounted_call(a, a.measure_coverage)
    assert "baseline suite" in out


def test_measure_coverage_red_suite_returns_failure_detail(monkeypatch, tmp_path) -> None:
    """When _ensure_tests_evaluated reports red, measure_coverage returns
    the failure detail and does not measure coverage."""
    a = _make_agent(tmp_path)
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (False, "FAILED tests/test_x.py::test_a"))
    out = _accounted_call(a, a.measure_coverage)
    assert "red" in out.lower() or "FAILED" in out or "fix" in out.lower()


def test_measure_coverage_cached_after_red_uses_red_variant(monkeypatch, tmp_path) -> None:
    """After a RED automatic prelude, with no measurement, a later cache hit
    returns the cached-RED variant, not the cached_rejected ``same numbers``
    text, since there are no numbers."""
    a = _make_agent(tmp_path)
    a.session.tests_changed_since_measure["coverage"] = False
    a.session.last_coverage_result = "[measure_coverage] Tests are not green; FAILED test_x"
    a.session.best_coverage_pct = None  # no real measurement ever ran
    out = _accounted_call(a, a.measure_coverage)
    assert "never measured" in out.lower()
    assert "same numbers" not in out.lower()


def test_measure_coverage_refuses_when_tests_known_red_no_edit(monkeypatch, tmp_path) -> None:
    """With ``last_tests_passed = False`` and no edit since the last
    run_tests, ``measure_coverage`` refuses before running the automatic
    prelude."""
    sub_calls = {"n": 0}
    _install_backend(monkeypatch, counter=sub_calls)
    a = _make_agent(tmp_path)
    a.session.last_tests_passed = False
    a.session.tests_changed_since_run_tests = False
    a.session.last_run_tests_result = "FAILED test_x"
    out = _accounted_call(a, a.measure_coverage)
    assert sub_calls["n"] == 0
    assert "REFUSED" in out
    assert "edit" in out.lower()


def test_measure_coverage_runs_when_file_edited_since_run_tests(monkeypatch, tmp_path) -> None:
    """With ``last_tests_passed = False`` but an edit since
    (``tests_changed_since_run_tests = True``), the refusal does not apply."""
    a = _make_agent(tmp_path)
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "all passed"))
    _install_backend(monkeypatch)
    a.session.last_tests_passed = False
    a.session.tests_changed_since_run_tests = True  # edited since
    a.session.last_run_tests_result = "stale failure detail"
    out = _accounted_call(a, a.measure_coverage)
    assert "REFUSED" not in out


def test_measure_coverage_cached_after_real_measurement_uses_standard_variant(monkeypatch, tmp_path) -> None:
    """After a real measurement, the cached path uses the ``cached_rejected``
    variant."""
    a = _make_agent(tmp_path)
    a.session.tests_changed_since_measure["coverage"] = False
    a.session.last_coverage_result = "Line coverage: 88.0% (branch 70.0%)"
    a.session.best_coverage_pct = 88.0
    out = _accounted_call(a, a.measure_coverage)
    assert "same numbers" in out.lower() or "no new measurement" in out.lower()
    assert "never measured" not in out.lower()


def test_measure_coverage_empty_collection_is_accounted_once(
        monkeypatch, tmp_path) -> None:
    a = _make_agent(tmp_path)
    monkeypatch.setattr(
        a,
        "_ensure_tests_evaluated",
        lambda *, log_prefix: (True, "all passed"),
    )
    _install_backend(
        monkeypatch,
        cov=_trr(ok=False, collected=0, returncode=5),
    )

    out = _accounted_call(a, a.measure_coverage)

    assert "ended without a test result" in out


def test_goal_is_judged_on_one_suites_line_and_branch(monkeypatch, tmp_path) -> None:
    """The best line and the best branch value can come from two suites:
    line 100 / branch 90, then line 95 / branch 98. Under ``min`` no suite
    reached 98, so a 98% goal is not met. The best values are the best
    suite's own pair (95 / 98, goal metric 95)."""
    results = iter([
        _trr(line_coverage=100.0, branch_coverage=90.0),
        _trr(line_coverage=95.0, branch_coverage=98.0),
    ])
    a = _make_agent(tmp_path)
    for key, value in {
        "goal": 98, "goal_metric": "both",
        "restore_count": 999, "max_measurements_without_improvement": 1000,
    }.items():
        set_phase_value(monkeypatch, tc._AGENT_CONFIG, key, value)
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "all passed"))
    backend = tc.languages.get_backend("python")
    monkeypatch.setattr(tc.languages, "get_backend", lambda lang: SimpleNamespace(
        run_test_files=lambda *args, **kw: next(results),
        find_target_test_file=backend.find_target_test_file,
        test_metrics_observation=backend.test_metrics_observation,
    ))

    for _ in range(2):
        a.session.tests_changed_since_measure["coverage"] = True
        a.measure_coverage()

    assert (a.session.best_coverage_pct, a.session.best_branch_coverage_pct) == (95.0, 98.0)
    assert a._best_goal_metric_value() == 95.0
    assert not a._coverage_goal_met()


def test_the_measurement_counts_java_untaken_branches(monkeypatch, tmp_path) -> None:
    """The summary counts untaken branches and their decision lines, so a
    Java entry per line ("k of n branches untaken") is not counted as one
    arc."""
    a = _make_agent(tmp_path)
    run = _trr(line_coverage=90.0, branch_coverage=50.0)._replace(
        missing_branches=("1->2 of 2 branches untaken", "1->exit"))
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "all passed"))
    backend = tc.languages.get_backend("python")
    monkeypatch.setattr(tc.languages, "get_backend", lambda lang: SimpleNamespace(
        run_test_files=lambda *args, **kw: run,
        find_target_test_file=backend.find_target_test_file,
        test_metrics_observation=backend.test_metrics_observation,
    ))
    a.session.tests_changed_since_measure["coverage"] = True

    out = a.measure_coverage()

    assert "Uncovered branches: 3 untaken on 1 decision line" in out
    assert "arc" not in out.split("Uncovered branches:")[1].splitlines()[0]


def _red_while_measured():
    return _trr(ok=False, returncode=1, collected=2)._replace(passed=1, failed=1)


def test_a_suite_red_only_while_measured_is_named_so_and_counts_as_a_failed_run(
        monkeypatch, tmp_path) -> None:
    """The normal run passed, the measuring run failed: the model is told
    the tests fail only while measured, and the attempt counts as a failed
    run, so the restore applies."""
    _install_backend(monkeypatch, cov=_red_while_measured())
    a = _make_agent(tmp_path)
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "all passed"))
    out = _accounted_call(a, a.measure_coverage)
    assert out.startswith("[measure_coverage] The tests below passed in the normal run but")
    assert "Tests are not green" not in out
    assert a.session.best_green.consecutive_failed_runs == 1
    assert a.session.last_tests_passed is False


def test_a_measurement_the_coverage_tool_refused_records_the_tests_that_stopped_it(
        monkeypatch, tmp_path) -> None:
    """The refusal's line in the coverage phase's log carries the tool and the
    failing tests as fields of its record, which the results page reads."""
    import json

    from testuary.agent import _helpers as agent_helpers
    from testuary.languages.generic import TestFailure

    failing = "test_mod.py::test_loaded"
    _install_backend(monkeypatch, cov=_red_while_measured()._replace(
        failures=(TestFailure(failing, "failed", "ImportError: loaded twice"),)))
    a = _make_agent(tmp_path)
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "all passed"))
    phase_log = agent_helpers.PhaseLogger("coverage")
    phase_log.init("proj", str(tmp_path / "logs"), "coverage.jsonl")
    monkeypatch.setattr(tc, "log_print", phase_log.log)
    _accounted_call(a, a.measure_coverage)
    records = [json.loads(line) for line in
               (tmp_path / "logs" / "coverage.jsonl").read_text(encoding="utf-8").splitlines()]
    assert [(row["tool"], row["stopping_tests"]) for row in records
            if row["record"] == "console" and "stopping_tests" in row] == [("coverage", [failing])]


def _console_refusals(tmp_path, monkeypatch):
    """The coverage phase's log as the results page reads it: the console
    records that carry the tests that stopped a tool."""
    import json

    from testuary.agent import _helpers as agent_helpers

    phase_log = agent_helpers.PhaseLogger("coverage")
    phase_log.init("proj", str(tmp_path / "logs"), "coverage.jsonl")
    monkeypatch.setattr(tc, "log_print", phase_log.log)

    def read():
        lines = (tmp_path / "logs" / "coverage.jsonl").read_text(encoding="utf-8").splitlines()
        return [(row["tool"], row["stopping_tests"]) for row in map(json.loads, lines)
                if row["record"] == "console" and "stopping_tests" in row]
    return read


def test_a_measured_run_without_a_test_result_is_the_coverage_tools_refusal(
        monkeypatch, tmp_path) -> None:
    """A run under the coverage tool that collected nothing, or ended
    without counting a failure, after the green normal run is the tool's
    refusal as a failing one is: the model sees the run's output, the
    attempt counts as a failed run, the suite needs an edit, the record says
    the tool named no test, and measuring the unchanged file again is
    refused rather than answered as already measured."""
    for collected in (0, 3):
        empty = _trr(ok=False, returncode=4, collected=collected)._replace(
            passed=0, output="ERROR: the run ended under the coverage tool")
        _install_backend(monkeypatch, cov=empty)
        a = _make_agent(tmp_path / str(collected))
        monkeypatch.setattr(a, "_ensure_tests_evaluated",
                            lambda *, log_prefix: (True, "all passed"))
        refusals = _console_refusals(tmp_path / str(collected), monkeypatch)
        out = _accounted_call(a, a.measure_coverage)
        assert "ended without a test result" in out
        assert "the run ended under the coverage tool" in out
        assert a.session.best_green.consecutive_failed_runs == 1
        assert a.session.last_tests_passed is False and a.refused_by() == "coverage"
        assert "tests=red" in a.build_state_marker()
        assert refusals() == [("coverage", [])]
        # The tool named no test, so no reply says it did.
        a.session.has_read = a.session.initial_measured = True
        offer = a.unbound_tool_reply("measure_coverage")
        assert "could not measure" in offer and "it named" not in offer
        assert a.measure_coverage().startswith("REFUSED: this measure_coverage call is invalid")


def test_a_refusal_that_restores_the_snapshot_leaves_the_restored_file_green(
        monkeypatch, tmp_path) -> None:
    """The restore after max_failed_test_attempts refusals brings back the
    state of the snapshot's bytes: they passed, so they are not marked with
    the discarded file's refusal."""
    _install_backend(monkeypatch, cov=_red_while_measured())
    a = _make_agent(tmp_path)
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "all passed"))

    def restore(target):
        a.session.last_tests_passed = True
        return True

    monkeypatch.setattr(a, "_restore_snapshot", restore)
    limit = int(tc._AGENT_CONFIG.get("max_failed_test_attempts", 10))
    a.session.best_green.consecutive_failed_runs = limit - 1
    _accounted_call(a, a.measure_coverage)
    assert a.session.last_tests_passed is True and a.refused_by() == ""


def test_repeated_red_measured_runs_restore_the_best_green_snapshot(monkeypatch, tmp_path) -> None:
    _install_backend(monkeypatch, cov=_red_while_measured())
    a = _make_agent(tmp_path)
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "all passed"))
    restored = []
    monkeypatch.setattr(a, "_restore_snapshot", lambda target: restored.append(target) or True)
    limit = int(tc._AGENT_CONFIG.get("max_failed_test_attempts", 10))
    a.session.best_green.consecutive_failed_runs = limit - 1
    out = _accounted_call(a, a.measure_coverage)
    assert len(restored) == 1
    note = a.tool_msg("tests_restore_note", max_attempts=limit, output="<OUTPUT>")
    assert note.split("<OUTPUT>")[0] in out


def test_a_measured_run_that_only_times_out_is_the_coverage_tools_refusal(
        monkeypatch, tmp_path) -> None:
    """The coverage tool can make the measured run slower than the normal
    one. When only the measured run does not finish, the tool could not
    measure the suite: the model is told so, with the running test when the
    journal names it, the attempt counts as a failed run and no suite
    timeout counts."""
    for active in (None, "tests/test_foo.py::test_slow"):
        slow = _trr(ok=False, returncode=-1, timed_out=True)._replace(active_at_end=active)
        _install_backend(monkeypatch, cov=slow)
        case_dir = tmp_path / ("named" if active else "plain")
        a = _make_agent(case_dir)
        monkeypatch.setattr(a, "_ensure_tests_evaluated",
                            lambda *, log_prefix: (True, "all passed"))
        refusals = _console_refusals(case_dir, monkeypatch)
        out = _accounted_call(a, a.measure_coverage)
        assert "passed in the normal run, but the\ncoverage tool's run of them did not finish" in out
        assert "TIMED OUT" not in out
        assert (active in out) if active else ("still running" not in out)
        assert a.session.consecutive_timeouts == 0
        assert a.session.best_green.consecutive_failed_runs == 1
        assert a.session.early_stop_reason is None
        assert a.refused_by() == "coverage"
        assert refusals() == [("coverage", [active] if active else [])]



def test_a_run_that_ends_without_output_leaves_a_result_the_model_can_act_on(monkeypatch, tmp_path) -> None:
    """A run_tests call whose run ends before it produced output leaves a
    result that names what the model can do, which is to edit the working
    file and call run_tests again. The model cannot read the run's log."""
    backend = tc.languages.get_backend("python")

    def _run(*args, **kwargs):
        raise OSError("the runner did not start")

    monkeypatch.setattr(tc.languages, "get_backend", lambda lang: SimpleNamespace(
        run_test_files=_run, find_target_test_file=backend.find_target_test_file))
    a = _make_agent(tmp_path)
    try:
        a._run_tests_uncached("run_tests")
    except OSError:
        pass
    result = a.session.last_run_tests_result
    assert "call run_tests again" in result
    assert "log" not in result


def test_a_measurement_before_any_run_names_run_tests(tmp_path) -> None:
    """Before any run_tests result exists, the measurement's test part says
    so in the tool's own name, which every language's loop offers, rather
    than in one test runner's name."""
    a = _make_agent(tmp_path)
    a.session.tests_changed_since_run_tests = False
    _passed, output = a._ensure_tests_evaluated(log_prefix="measure_coverage")
    assert "run_tests" in output
    assert "pytest" not in output


def test_a_measurement_without_coverage_logs_that_none_was_measured(monkeypatch, tmp_path) -> None:
    """A measured run that yields no line or branch coverage is logged as
    such, never as a report that could not be parsed."""
    _install_backend(monkeypatch, cov=_trr(line_coverage=None))
    a = _make_agent(tmp_path)
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "all passed"))
    logs = []
    monkeypatch.setattr(tc, "log_print", lambda msg, *args, **kwargs: logs.append(msg))
    _accounted_call(a, a.measure_coverage)
    assert "[measure_coverage] the run measured no coverage" in logs


def test_the_goal_and_early_stops_log_what_ended_the_phase(monkeypatch, tmp_path) -> None:
    """The goal can be reached on any measurement: the stop's log reports
    the value that reached it. A stagnation stop's log says what follows."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "line")
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal", 90)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "extra_measurements_after_goal", 0)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_measurements_without_improvement", 2)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "allow_early_stop", True)
    logs = []
    monkeypatch.setattr(tc, "log_print", lambda msg, *args, **kwargs: logs.append(msg))
    reached = _make_agent(tmp_path)
    reached.session.best_coverage_pct = 92.5
    reached.session.coverage_measurements = 3
    assert reached._maybe_set_early_stop(source_tool="measure_coverage") is not None
    assert any("goal reached (92.5% >= 90%)" in line for line in logs)
    assert not any("first measurement" in line for line in logs)
    stalled = _make_agent(tmp_path / "b")
    stalled.session.best_coverage_pct = 50.0
    stalled.session.measurements_without_improvement = 2
    assert stalled._maybe_set_early_stop(source_tool="measure_coverage") is not None
    assert stalled.session.early_stop_reason == "coverage_stagnation"
    assert any(line.endswith("below goal -> handing off to the mutation phase.")
               for line in logs)


# The replies of measure_coverage's paths that end before a coverage value,
# exact, with the session fields each writes beside every call's own.

def _coverage_refusal_fields(reply):
    """The fields a measurement writes when the coverage tool refuses after a
    green normal run: the suite reads red with the refusal as the tool's,
    the reply answers the unchanged file, and the attempt counts as a
    failed run."""
    return {
        "tests_changed_since_measure": {"coverage": False}, "last_coverage_result": reply,
        "last_run_tests_result": reply, "tests_changed_since_run_tests": False,
        "refusal": ("coverage", reply),
        "best_green": tc.agent_helpers.BestGreenState(consecutive_failed_runs=1),
    }


def _measured_with(monkeypatch, tmp_path, cov):
    """measure_coverage after a green prelude whose measured run gives
    ``cov``: the agent, its reply and the session fields the call wrote."""
    _install_backend(monkeypatch, cov=cov)
    a = _make_agent(tmp_path)
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "all passed"))
    output, written = written_fields(a.session, a.measure_coverage)
    return a, output, written


def test_the_preludes_revert_answers_the_revert_and_measures_nothing(monkeypatch, tmp_path) -> None:
    """A prelude that reverted the file to the best-green snapshot answers
    the revert as it is and runs no measured run."""
    counter = {"n": 0}
    _install_backend(monkeypatch, counter=counter)
    a = _make_agent(tmp_path)
    revert = "REVERTED to the best-green snapshot"

    def reverted(*, log_prefix):
        a.session.best_green.restores += 1
        return True, revert

    monkeypatch.setattr(a, "_ensure_tests_evaluated", reverted)
    output, written = written_fields(a.session, a.measure_coverage)
    assert output == revert
    assert counter["n"] == 0
    assert written == {
        "tests_changed_since_measure": {"coverage": False},
        "last_coverage_result": a.tool_msg("measure_coverage_no_prior_result"),
        "best_green": tc.agent_helpers.BestGreenState(restores=1)}


@pytest.mark.parametrize("active", [None, "tests/test_foo.py::test_slow"])
def test_the_measured_runs_timeout_reply_and_the_fields_it_writes(monkeypatch, tmp_path, active) -> None:
    slow = _trr(ok=False, returncode=-1, timed_out=True)._replace(active_at_end=active)
    a, output, written = _measured_with(monkeypatch, tmp_path, slow)
    timeout = int(tc._AGENT_CONFIG["suite_run_timeout_seconds"])
    assert output == (a.tool_msg("measure_coverage_timeout_when_measured_named",
                                 timeout=timeout, test=active) if active
                      else a.tool_msg("measure_coverage_timeout_when_measured", timeout=timeout))
    assert written == _coverage_refusal_fields(output)


def test_the_measured_runs_red_reply_and_the_fields_it_writes(monkeypatch, tmp_path) -> None:
    red = _red_while_measured()
    a, output, written = _measured_with(monkeypatch, tmp_path, red)
    assert output == a.tool_msg("measure_coverage_red_when_measured",
                                tests_output=a._canonical_view(red, None, 1))
    assert written == _coverage_refusal_fields(output)


@pytest.mark.parametrize("diagnostics", [(), _UNREAD])
def test_the_measured_runs_reply_without_a_test_result_and_the_fields_it_writes(
        monkeypatch, tmp_path, diagnostics) -> None:
    """A run that collected no test gets the unmeasured reply, also when its
    coverage counts could not be read."""
    empty = _trr(ok=False, returncode=4, collected=0)._replace(
        output="the run ended", diagnostics=diagnostics)
    a, output, written = _measured_with(monkeypatch, tmp_path, empty)
    assert output == a.tool_msg("measure_coverage_unmeasured", output="the run ended")
    assert written == _coverage_refusal_fields(output)


def test_the_measured_runs_reply_when_its_coverage_counts_cannot_be_read(
        monkeypatch, tmp_path) -> None:
    """A measured run whose coverage counts the coverage tool could not read
    answers with its own reply, in the language's words for C, and counts as
    a failed run like every refusal."""
    unread = _trr(ok=False, returncode=-1)._replace(
        output="gcovr: negative count", diagnostics=_UNREAD)
    words = c_prompts("test_coverage")
    _install_backend(monkeypatch, cov=unread)
    a = _make_agent(tmp_path)
    a.prompts = words
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "all passed"))
    output, written = written_fields(a.session, a.measure_coverage)
    assert output == (
        "[measure_coverage] The tests passed, but the coverage tool could not\n"
        "read the coverage counts their run left, so coverage cannot be\n"
        "measured.\n"
        "Counts that cannot be read usually mean that a test corrupts memory or relies "
        "on undefined behavior.\n"
        "If your last edit added or changed tests, check those first.\n"
        "Change the test so that it uses only valid values and memory, or remove it.\n"
        "\n"
        "WORKFLOW: edit the tests with edit_tests and call run_tests, then\n"
        "measure_coverage again.")
    assert written == _coverage_refusal_fields(output)


def test_a_measured_run_with_failing_tests_and_unreadable_counts_takes_the_red_reply(
        monkeypatch, tmp_path) -> None:
    """A measured run whose tests fail and whose coverage counts the
    coverage tool could not read gets the red reply, in C's words too: its
    failing tests decide."""
    red = _red_while_measured()._replace(diagnostics=_UNREAD)
    words = c_prompts("test_coverage")
    _install_backend(monkeypatch, cov=red)
    a = _make_agent(tmp_path)
    a.prompts = words
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "all passed"))
    output, _ = written_fields(a.session, a.measure_coverage)
    assert output == a.tool_msg("measure_coverage_red_when_measured",
                                tests_output=a._canonical_view(red, None, 1))


def test_a_language_that_names_no_counts_cause_gets_no_counts_reply() -> None:
    """Python's coverage reports no counts it cannot read, and its vocabulary
    names no cause for them: its prompts carry neither reply."""
    assert "measure_coverage_counts_inconsistent" not in python_prompts(
        "test_coverage")["tool_messages"]
    assert "measure_mutation_counts_inconsistent" not in python_prompts(
        "test_mutation")["tool_messages"]


def test_an_empty_counts_entry_leaves_no_line(monkeypatch) -> None:
    """A language whose vocabulary names the cause and the fix of coverage
    counts the coverage tool cannot read as empty, as C's extension mode
    does, gets that reply without a line for either."""
    own = prompts_module._own
    monkeypatch.setattr(prompts_module, "_own", lambda backend, source_dir: {
        **own(backend, source_dir), "lang_coverage_counts_inconsistent_cause": "",
        "lang_coverage_counts_inconsistent_fix": ""})
    assert prompts_module.resolve_agent_prompts("test_coverage", "python")["tool_messages"][
        "measure_coverage_counts_inconsistent"] == (
        "[measure_coverage] The tests passed, but the coverage tool could not\n"
        "read the coverage counts their run left, so coverage cannot be\n"
        "measured.\n"
        "If your last edit added or changed tests, check those first.\n"
        "\n"
        "WORKFLOW: edit the tests with edit_tests and call run_tests, then\n"
        "measure_coverage again.")


def test_a_run_without_coverage_logs_the_runs_own_diagnosis(monkeypatch, tmp_path) -> None:
    """A measured run without coverage whose output carries the JaCoCo
    diagnosis logs that line, not the general one."""
    logs = []
    monkeypatch.setattr(tc, "log_print", lambda msg, *args, **kwargs: logs.append(msg))
    diagnosis = "[testuary][jacoco-miss] no execution data for foo"
    _measured_with(monkeypatch, tmp_path,
                   _trr(line_coverage=None)._replace(output=f"run\n  {diagnosis}  \nend\n"))
    assert f"[measure_coverage] {diagnosis}" in logs
    assert "[measure_coverage] the run measured no coverage" not in logs


def test_the_streak_counts_a_small_gain_once_the_minimum_is_reached(monkeypatch, tmp_path) -> None:
    """Below ``minimum`` any rise of the goal metric resets the streak of
    measurements without improvement. From there only a rise by at least
    ``min_improvement_points`` points does, and no rise adds one."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "min_improvement_points", 5)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "minimum", 50)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_measurements_without_improvement", 10)
    a = _make_agent(tmp_path)
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "all passed"))
    streaks = []
    for line in (40.0, 42.0, 55.0, 57.0, 57.0, 50.0, 63.0):
        _install_backend(monkeypatch, cov=_trr(line_coverage=line))
        a.session.tests_changed_since_measure["coverage"] = True
        a.measure_coverage()
        streaks.append(a.session.measurements_without_improvement)
    assert streaks == [0, 0, 0, 1, 2, 3, 0]


def test_the_main_measurement_reply_keeps_its_bytes(monkeypatch, tmp_path) -> None:
    """measure_coverage's reply, the model's input on the main path: the
    summary, the uncovered lines and the uncovered branches in their source
    context. Pinned byte for byte, so a change of its text is a decision,
    not a side effect."""
    # x is always true in the tests: the last line and the branch to it stay
    # uncovered.
    covered = _trr(line_coverage=75.0, branch_coverage=50.0, missing_lines=(4,))._replace(
        missing_branches=("2->4",))
    _install_backend(monkeypatch, cov=covered)
    a = _make_agent(tmp_path)
    (tmp_path / "src" / "foo.py").write_text(
        "def f(x):\n    if x:\n        return 1\n    return 0\n", encoding="utf-8")
    monkeypatch.setattr(a, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "all passed"))
    reply = a.measure_coverage()
    assert reply == (
        '--- Coverage Summary ---\n'
        'Line coverage: 75.0%\n'
        'Branch coverage: 50.0%\n'
        'Uncovered lines: 4\n'
        'The uncovered source is appended below.\n'
        'Uncovered branches: 1 untaken on 1 decision line -- the decision lines and untaken directions are appended below.\n'
        '\n'
        '=== UNCOVERED (1 lines, 1 regions) ===\n'
        '\n'
        '--- Lines 4-4 ---\n'
        '       2 |     if x:\n'
        '       3 |         return 1\n'
        '>>>    4 |     return 0\n'
        '       5 | \n'
        '\n'
        '\n'
        '\n'
        '=== UNCOVERED BRANCHES (1 untaken branches, 1 decision lines) ===\n'
        '\n'
        '--- Lines 2-2 ---\n'
        '       1 | def f(x):\n'
        '>>>    2 |     if x:    <- branch to line 4 (return 0) not taken\n'
        '       3 |         return 1\n'
        '       4 |     return 0\n'
        '\n'
    )
