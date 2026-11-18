"""The start of both loops: the designated read, then the measurement as
the only offered tool, answered from the stored measurement of the exact
suite, then Revise."""
from __future__ import annotations

import hashlib
from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_fixtures import mutation_agent, project_agent, run_result

from offered_tools import offered_tools
from testuary.agent import coverage_agent as tc
from testuary.agent import mutation_agent as tm
from testuary.prompts import resolve_agent_prompts

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

SUITE = "def test_x():\n    assert True\n"


def _run():
    """A green run of one test at 50% line and branch coverage, with line 2
    and the branch 1->2 not covered."""
    return run_result(collected=1, passed=1, output="out", line_coverage=50.0,
                      branch_coverage=50.0, missing_lines=(2,), missing_branches=("1->2",))


def _coverage_agent(tmp_path: Path) -> "tc.CoverageAgent":
    return project_agent(tmp_path, source="def f(x):\n    return x\n", test_source=SUITE)


def _backend(monkeypatch, run):
    """The coverage backend with ``run`` answering every suite run."""
    real = tc.languages.get_backend("python")
    calls = []

    def run_test_files(*args, **kwargs):
        calls.append(kwargs)
        if isinstance(run, Exception):
            raise run
        return run

    monkeypatch.setattr(tc.languages, "get_backend", lambda lang: SimpleNamespace(
        run_test_files=run_test_files,
        find_target_test_file=real.find_target_test_file,
        test_metrics_observation=real.test_metrics_observation,
    ))
    return calls


def _seed(agent, **baseline):
    tc._seed_baseline(agent.session, {"content": SUITE, "line_coverage": 50.0,
                                      "branch_coverage": 50.0, "test_count": 1,
                                      **baseline}, agent.deps.target_file_path)


def test_coverage_modes_come_read_then_measure_then_revise(tmp_path, monkeypatch):
    agent = _coverage_agent(tmp_path)
    _seed(agent, missing_lines=(2,), missing_branches=("1->2",))
    _backend(monkeypatch, AssertionError("no run expected"))
    assert agent.effective_agent_mode() == agent.INITIAL_READ_MODE
    agent.read_tests()
    assert agent.effective_agent_mode() == agent.INITIAL_MEASURE_MODE
    agent.measure_coverage()
    assert agent.effective_agent_mode() == "needs_editing"


def test_mutation_modes_come_read_then_measure_then_revise(tmp_path, monkeypatch):
    agent = _mutation_agent(tmp_path)
    agent.session.measured_suite_hash = hashlib.sha256(SUITE.encode("utf-8")).hexdigest()
    agent.session.last_mutation_result = "stored"
    assert agent.effective_agent_mode() == agent.INITIAL_READ_MODE
    agent.read_tests()
    assert agent.effective_agent_mode() == agent.INITIAL_MEASURE_MODE
    agent.measure_mutation()
    assert agent.effective_agent_mode() == "needs_editing"


@pytest.mark.parametrize("build, measure", [
    (lambda p: _coverage_agent(p), "measure_coverage"),
    (lambda p: _mutation_agent(p), "measure_mutation"),
])
def test_the_measure_mode_binds_only_the_measurement(tmp_path, build, measure):
    agent = build(tmp_path)
    modes = agent.tool_mode_sets(*agent.phase_tools())
    assert [t.name for t in modes[agent.INITIAL_MEASURE_MODE]] == [measure]
    agent.session.has_read = True
    assert [t.__name__ for t in offered_tools(agent)] == [measure]


def test_a_stored_coverage_measurement_answers_without_a_run(tmp_path, monkeypatch):
    agent = _coverage_agent(tmp_path)
    _seed(agent, missing_lines=(2,), missing_branches=("1->2",))
    calls = _backend(monkeypatch, AssertionError("no run expected"))
    agent.read_tests()

    answer = agent.measure_coverage()

    assert calls == []
    assert answer == agent.session.stored_measurement[1]
    assert agent.session.coverage_measurements == 0


def test_the_stored_coverage_answer_is_the_tools_own(tmp_path, monkeypatch):
    """The start's stored answer equals the live measurement of that suite."""
    stored = _coverage_agent(tmp_path / "a")
    _seed(stored, missing_lines=(2,), missing_branches=("1->2",))
    _backend(monkeypatch, AssertionError("no run expected"))
    stored.read_tests()
    start = stored.measure_coverage()

    live = _coverage_agent(tmp_path / "b")
    _seed(live)
    _backend(monkeypatch, _run())
    live.session.has_read = True
    live.session.initial_measured = True
    live.session.writes_done = 1
    live.session.tests_changed_since_measure["coverage"] = True
    answer = live.measure_coverage()

    assert answer.startswith(start)


def test_an_unstored_coverage_suite_is_measured_at_the_start(tmp_path, monkeypatch):
    """A suite with no stored measurement of its bytes (here no uncovered
    record) is measured by the start's call, outside the loop's counters."""
    agent = _coverage_agent(tmp_path)
    _seed(agent)
    calls = _backend(monkeypatch, _run())
    agent.read_tests()

    answer = agent.measure_coverage()

    assert len(calls) == 1
    assert answer.startswith("--- Coverage Summary ---\nLine coverage: 50.0%")
    assert ">>>    2 |     return x" in answer
    assert agent.session.coverage_measurements == 0
    assert agent.session.measurements_without_improvement == 0
    assert agent.effective_agent_mode() == "needs_editing"


def test_the_starts_measurement_credits_an_unmeasured_suite_as_the_best(tmp_path, monkeypatch):
    """A suite handed over without its measurement is measured by the
    start's call, and that measurement credits it as the best-green
    snapshot, so a later lower suite is a regression against it and a
    revert or the phase's end can return to it."""
    agent = _coverage_agent(tmp_path)
    _seed(agent, line_coverage=None, branch_coverage=None)
    assert agent.session.best_green.snapshot["coverage"] is None
    _backend(monkeypatch, _run())
    agent.read_tests()

    agent.measure_coverage()

    snapshot = agent.session.best_green.snapshot
    assert (snapshot["content"], snapshot["coverage"], snapshot["branch_coverage"]) == (SUITE, 50.0, 50.0)
    assert agent.measured_bytes(snapshot["content"])


def _mutation_agent(tmp_path: Path) -> "tm.MutationAgent":
    source = "def f(x): return x\n"
    (tmp_path / "src.py").write_text(source, encoding="utf-8")
    agent = mutation_agent(tmp_path, target_file_path=str(tmp_path / "src.py"), target_file_content=source,
                           mutation_test_file_name="test_src_mutation.py", coverage_baseline_pct=80.0)
    agent.target_path().write_text(SUITE, encoding="utf-8")
    return agent


def test_a_stored_mutation_measurement_answers_without_an_engine_run(tmp_path, monkeypatch):
    agent = _mutation_agent(tmp_path)
    agent.session.measured_suite_hash = hashlib.sha256(SUITE.encode("utf-8")).hexdigest()
    agent.session.last_mutation_result = "Mutation score: 50.0% (killed 1/2; survived 1)"
    monkeypatch.setattr(agent, "_measure_mutation_engine",
                        lambda **kw: pytest.fail("no engine run expected"))
    agent.read_tests()

    assert agent.measure_mutation() == agent.session.last_mutation_result


def test_an_unstored_mutation_suite_is_measured_at_the_start(tmp_path, monkeypatch):
    agent = _mutation_agent(tmp_path)
    runs = []
    monkeypatch.setattr(agent, "_measure_mutation_engine",
                        lambda framework_prep=False: runs.append(framework_prep) or "measured")
    agent.read_tests()

    assert agent.measure_mutation() == "measured"
    assert runs == [True]
    assert agent.session.mutation_measurements == 0


def test_the_derived_mutation_summary_is_the_tools_own():
    """A derived baseline summarizes its matrix with the tool's summary
    line: the same counts give the same words."""
    from testuary.languages.mutation import MutationMeasurement

    matrix = {"mutants": [
        {"id": 1, "status": "killed"}, {"id": 2, "status": "timeout"},
        {"id": 3, "status": "survived"}, {"id": 4, "status": "untested"}]}
    # As MutationMeasurement.of reads it: the untested mutant counts as not
    # killed, so 2 of 4.
    live = MutationMeasurement(killed=1, survived=1, timeout=1, non_viable=0,
                               untested=1, total=3, score=50.0, lower_bound=True)
    derived = tm._matrix_measurement(matrix, None, 50.0)
    assert tm._mutation_summary_line(derived) == tm._mutation_summary_line(live)


def _stored_baseline(tmp_path, monkeypatch, matrix):
    """A mutation agent whose coverage phase stored ``matrix`` as the baseline
    of its suite, and a backend whose runner has no second catalogue."""
    import json

    agent = _mutation_agent(tmp_path)
    (tmp_path / ".mutation_baseline_matrix.json").write_text(json.dumps({
        "matrix": matrix, "suite_sha256": tm.source_sha256(SUITE),
        "measurement_identity": tm.mutation_measurement_identity(
            target_file_path=agent.deps.target_file_path, language="python",
            execution_name=agent.target_path().name)}), encoding="utf-8")
    monkeypatch.setattr(tm.languages, "get_backend", lambda language: SimpleNamespace(
        mutation_runner=lambda config, source_dir=None: SimpleNamespace(
            full_operators=lambda: None)))
    return agent


def test_a_stored_baseline_matrix_of_version_2_gives_the_counts_of_its_words(
        tmp_path, monkeypatch):
    """A baseline matrix the coverage phase stored at kill-matrix version 2,
    read by a run resumed after that phase, gives the model its counts in
    this version's words, never every mutant untested."""
    agent = _stored_baseline(tmp_path, monkeypatch, {"schema_version": 2, "mutants": [
        {"id": 1, "status": "ok_killed"}, {"id": 2, "status": "bad_survived"}]})

    assert tm._seed_mutation_baseline_from_matrix(agent)
    assert agent.session.last_mutation_result.startswith(
        "Mutation score: 50.0% (killed 1/2; survived 1)")


def test_a_stored_baseline_matrix_of_a_later_version_is_measured_instead(
        tmp_path, monkeypatch):
    """A baseline matrix of a kill-matrix version this release does not read
    is refused, as an unbound one is, and the baseline is measured."""
    agent = _stored_baseline(tmp_path, monkeypatch, {"schema_version": 99, "mutants": []})

    assert tm._seed_mutation_baseline_from_matrix(agent) is False
    assert agent.session.last_mutation_result == ""


@pytest.mark.parametrize("build, measure", [
    (lambda p: _coverage_agent(p), "measure_coverage"),
    (lambda p: _mutation_agent(p), "measure_mutation"),
])
def test_the_start_modes_say_what_comes_next(tmp_path, build, measure):
    agent = build(tmp_path)
    assert agent.unbound_tool_reply("edit_tests") == (
        f"edit_tests is not offered yet: call read_tests first, then {measure}. "
        "The editing tools are offered after them.")
    agent.session.has_read = True
    assert agent.unbound_tool_reply("edit_tests") == (
        f"edit_tests is not offered yet: call {measure} now. The editing "
        "tools are offered after it.")


def _start_paragraph(agent):
    system = resolve_agent_prompts(agent, "python")["system_prompt"]
    start = system.index("Tool availability is phase-gated")
    return system[start:system.index("not available yet.", start)]


def test_the_start_is_described_in_the_same_words_in_both_phases():
    coverage = _start_paragraph("test_coverage")
    assert coverage == _start_paragraph("test_mutation")
    assert "first\nread the current tests" in coverage
    assert "measure them" in coverage and "Testuary's record of that suite" in coverage


def test_the_derived_and_the_live_summary_count_a_timeout_kill_alike():
    """A timeout the per-test journal attributed to a failing test is an
    ``killed`` row keeping its ``pre_attribution_status``. The live reading
    and the derive from the stored matrix count it as a timeout through the
    same function, so the start's answer and a live measurement read alike."""
    from testuary.languages import mutation as mutation_runs
    from testuary.languages.mutation import MutationMeasurement

    rows = [
        {"id": 1, "status": "killed"},
        {"id": 2, "status": "killed", "pre_attribution_status": "timeout"},
        {"id": 3, "status": "timeout"},
        {"id": 4, "status": "non_viable"},
        {"id": 5, "status": "survived"},
        {"id": 6, "status": "untested"},
    ]
    run = SimpleNamespace(
        measured=True, complete=True, failure=None, reason="",
        outcome=mutation_runs.Outcome.COMPLETED,
        mutants=[SimpleNamespace(row=row, classification=None) for row in rows],
        progress_score=lambda: 100.0 * 4 / 6)
    live = MutationMeasurement.of(run)
    derived = tm._matrix_measurement({"mutants": rows}, None, 100.0 * 4 / 6)

    # The untested mutant counts as not killed, in both readings.
    assert tm._mutation_summary_line(derived) == tm._mutation_summary_line(live)
    assert tm._mutation_summary_line(live) == (
        "Mutation score: 66.6% (killed 4/6, 2 of them by timeout, 1 of them "
        "invalid; survived 1; untested 1)")
