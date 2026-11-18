"""Records and alarms state the per-mutant timeout the engine applied.

PIT bounds each test of a mutant by its own rule (1.25 times the test's normal
time plus 4 s) unless per_mutant_timeout_seconds fixes one. Testuary's derived
timeout (factor times the suite time, at least min_per_mutant_timeout_seconds) does not
apply then, so logs, kill matrix, finishing limits and alarms must not
state it or advise changing its settings. Mull runs each C test of a
mutant in a process of its own and applies Testuary's rule to its own run
of the tests, so a C request carries the rule and not seconds.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from langadapter.mutation import Catalogue, EffectiveSetting, TimeoutRule
from langadapter.mutation import MutationRun

from testuary.languages import mutation as mutation_runs
from testuary.languages.generic import GenericMutationRunner
from testuary.languages import mutation as _kill_matrix
from testuary import measurement as finishing
from testuary.agent import mutation_agent as tm

PIT_RULE = EffectiveSetting(4, "default", {"rule": "pit", "factor": 1.25, "constant_s": 4})
PIT_TEXT = "1.25 x each test's normal time + 4s per test (PIT's rule)"


def _pit_run() -> MutationRun:
    return MutationRun(target_file="Calc.java", mutants=(), summary={},
                       requested_catalogue="full", requested_operators=(),
                       settings={"per_mutant_timeout": PIT_RULE})


def test_the_kill_matrix_records_the_rule_pit_applied():
    measurement = mutation_runs.MutationMeasurement.of(_pit_run())
    matrix = measurement.kill_matrix()
    assert matrix["per_mutant_timeout"] == PIT_RULE.to_dict()
    assert mutation_runs.timeout_rule_text(matrix["per_mutant_timeout"]) == PIT_TEXT


def test_the_runner_says_when_the_engine_sets_the_timeout():
    backend = SimpleNamespace(own_mutant_timeout_rule=True)
    runner = GenericMutationRunner(backend, {"per_mutant_timeout_seconds": "auto"}, engine="pit")
    assert runner.engine_sets_timeout() is True
    assert runner.requested_timeout_text(30) == "the pit engine's own rule"
    fixed = GenericMutationRunner(backend, {"per_mutant_timeout_seconds": 45}, engine="pit")
    assert fixed.engine_sets_timeout() is False
    assert fixed.requested_timeout_text(45) == "45s"


def test_the_loop_alarm_states_pit_s_rule_and_the_setting_that_changes_it():
    applied = PIT_RULE.to_dict()
    msg = tm._timeout_alarm_message(5, 5, 5, 0, mutation_runs.timeout_rule_text(applied), applied)
    assert PIT_TEXT in msg
    assert "30s" not in msg
    assert "do not change" in msg and "fixed per_mutant_timeout_seconds" in msg


def test_the_matrix_alarm_states_pit_s_rule():
    matrix = {"mutants": [{"id": 1, "status": "killed"},
                          {"id": 2, "status": "timeout"}],
              "per_mutant_timeout": PIT_RULE.to_dict()}
    msg = _kill_matrix.timeout_kill_alarm(matrix)
    assert PIT_TEXT in msg and "do not change" in msg


def test_the_finishing_step_records_the_applied_rule_not_the_derived_timeout(
        tmp_path, monkeypatch):
    monkeypatch.setattr(
        finishing.languages, "get_backend",
        lambda lang: SimpleNamespace(
            file_extensions=(".java",), shared_setup_files=lambda root: [],
            run_test_files=lambda paths, root, **kw: SimpleNamespace(
                ok=True, elapsed_s=0.1, returncode=0)))
    runner = GenericMutationRunner(SimpleNamespace(own_mutant_timeout_rule=True),
                                   {"per_mutant_timeout_seconds": "auto"}, engine="pit")
    measurement = mutation_runs.MutationMeasurement.of(_pit_run())
    logged = []
    session = SimpleNamespace(runner=runner, run=lambda **kw: measurement)
    fn = finishing.make_finishing_mutation_fn(
        session=session, target_file="Calc.java", source_dir=str(tmp_path),
        configured_per_mutant_timeout="auto", wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0, language="java",
        log_fn=logged.append)
    test_dir = tmp_path / "delivered"
    test_dir.mkdir()
    fn(test_dir)
    limits = fn.limit_of(test_dir)
    assert limits["per_mutant_timeout_seconds"] is None
    assert limits["per_mutant_timeout"] == PIT_RULE.to_dict()
    assert any("the pit engine's own rule" in str(line) for line in logged)
    assert not any("per-mutant timeout 30s" in str(line) for line in logged)


#: The bundled per-mutant timeout settings.
TIMEOUT_SETTINGS = {"per_mutant_timeout_seconds": "auto", "per_mutant_timeout_factor": 10.0,
                    "min_per_mutant_timeout_seconds": 30, "max_per_mutant_timeout_factor": 1.5,
                    "suite_run_timeout_seconds": 600}
#: The rule Mull applied to its own run of the tests alone, which took 4.2 s.
MULL_RULE = EffectiveSetting(42, "request", {"factor": 10.0, "minimum_s": 30, "cap_s": 900,
                                             "passing_suite_s": 4.2})


def _runner(engine: str, configured: object = "auto") -> GenericMutationRunner:
    backend = SimpleNamespace(own_mutant_timeout_rule=False,
                              mutation_catalogue=lambda config: Catalogue())
    return GenericMutationRunner(
        backend, {**TIMEOUT_SETTINGS, "per_mutant_timeout_seconds": configured}, engine=engine)


def _requested(runner: GenericMutationRunner, seconds: int):
    return runner.request(target=Path("src/calc.c"), test_dir=Path("tests"),
                          test_files=["test_calc.c"], per_mutant_timeout=seconds,
                          wall_clock_timeout=3600).per_mutant_timeout


def test_the_c_engine_gets_the_rule_for_its_own_run_of_the_tests_alone():
    """Mull runs each test of a mutant in a process of its own, and
    Testuary's normal run runs them in one program, so a C request carries
    the rule, which the engine applies to its own run, and not seconds from
    Testuary's run. A fixed value and Cosmic Ray's requests keep the
    seconds."""
    assert _requested(_runner("mull"), 30) == TimeoutRule(factor=10.0, minimum_s=30, cap_s=900)
    assert _runner("mull").requested_timeout_text(30) == (
        "10 x the mull engine's own run of the passing suite, at least 30s, at most 900s")
    assert _requested(_runner("mull", 45), 45) == 45
    assert _runner("mull", 45).requested_timeout_text(45) == "45s"
    assert _requested(_runner("cosmic_ray"), 30) == 30


def test_a_c_run_states_its_timeout_per_mutant():
    """The C engine gives a mutant's tests one time limit together, fixed or
    from the rule, so the logs and alarms state it per mutant."""
    assert mutation_runs.timeout_rule_text(MULL_RULE.to_dict()) == "42s per mutant"
    fixed = EffectiveSetting(45, "request", {"rule": "fixed"})
    assert mutation_runs.timeout_rule_text(fixed.to_dict()) == "45s per mutant"


def test_the_finishing_step_records_the_rule_a_c_run_applied(tmp_path, monkeypatch):
    monkeypatch.setattr(
        finishing.languages, "get_backend",
        lambda lang: SimpleNamespace(
            file_extensions=(".c",), shared_setup_files=lambda root: [],
            run_test_files=lambda paths, root, **kw: SimpleNamespace(
                ok=True, elapsed_s=0.1, returncode=0)))
    run = MutationRun(target_file="src/calc.c", mutants=(), summary={},
                      requested_catalogue="full", requested_operators=(),
                      settings={"per_mutant_timeout": MULL_RULE})
    logged = []
    session = SimpleNamespace(runner=_runner("mull"),
                              run=lambda **kw: mutation_runs.MutationMeasurement.of(run))
    fn = finishing.make_finishing_mutation_fn(
        session=session, target_file="src/calc.c", source_dir=str(tmp_path),
        configured_per_mutant_timeout="auto", wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0, language="c", log_fn=logged.append)
    test_dir = tmp_path / "delivered"
    test_dir.mkdir()
    fn(test_dir)
    limits = fn.limit_of(test_dir)
    assert limits["per_mutant_timeout_seconds"] is None
    assert limits["per_mutant_timeout"] == MULL_RULE.to_dict()
    assert any("10 x the mull engine's own run" in str(line) for line in logged)
    assert not any("per-mutant timeout 30s" in str(line) for line in logged)
