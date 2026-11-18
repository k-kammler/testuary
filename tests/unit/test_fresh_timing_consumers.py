"""Testuary keeps fresh-process timing when warmed timing is available."""
from dataclasses import asdict
from pathlib import Path
import subprocess
import sys

from langadapter import Outcome
from langadapter.outcomes import ExecutionIdentity, ObservationIdentity, ToolchainIdentity
from langadapter.java.adapter import JavaAdapter as ProjectJavaBackend
from langadapter.python.adapter import PythonAdapter as ProjectPythonBackend
from langadapter.runs import Benchmark, BenchmarkProgress, BenchmarkRequest, WALL_TIME_SCOPE
from langadapter.operations import TestOperations
import pytest

from test_prepared_benchmarks import _project
from testuary import config as config_module
from testuary import project
from testuary.evidence import current_condition
from testuary.__main__ import _build_parser
from testuary.languages.generic import GenericBackend
from testuary.languages import preparation
from testuary import consolidation as consolidate
from testuary import measurement as metrics
from testuary.settings import resolve_settings
from phase_values import set_phase_value

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture
def fresh_timing(tmp_path, monkeypatch, request):
    language = request.param
    root, target, test = _project(tmp_path, language, False)
    calls = []

    def native_forbidden(*args, **kwargs):
        pytest.fail("Controlled timing consumers must not execute native tools")

    monkeypatch.setattr(subprocess, "Popen", native_forbidden)
    monkeypatch.setattr(subprocess, "run", native_forbidden)
    # The bytecode store's helper is a native command; its own tests run it.
    monkeypatch.setattr(preparation, "bytecode_store", lambda backend: {})
    monkeypatch.setattr(config_module, "project_exec_mode", lambda: "local")
    monkeypatch.setattr(project, "project_work_dir", lambda: None)
    # Each language's runtime is this interpreter, selected in the settings,
    # and the timing lock is on, its default.
    monkeypatch.setattr(config_module, "load_config", lambda section: {
        "pytest_config": "none", "frozen_module_strategy": "skip",
        "python_executable": sys.executable, "java_home": sys.executable,
        "serialize": "true"})

    def warmed_forbidden(*args, **kwargs):
        pytest.fail("Testuary must not select warmed suite timing")

    def observed(backend, benchmark_request):
        assert type(benchmark_request) is BenchmarkRequest
        benchmark_request.validate()
        assert benchmark_request.serialize is False
        assert benchmark_request.cpus is None
        text = Path(benchmark_request.suite.test_files[0]).read_text(encoding="utf-8")
        samples = (899.0, 900.0, 901.0) if "baseline" in text else (99.0, 100.0, 101.0)
        if benchmark_request.runs == 1:
            # One execution of an interleaved timing round.
            samples = samples[1:2]
        calls.append(benchmark_request)
        if benchmark_request.warmup_runs:
            benchmark_request.progress(BenchmarkProgress("warmup", 1, benchmark_request.warmup_runs, 7.0))
        return Benchmark(
            Outcome.COMPLETED, "", samples, benchmark_request.warmup_runs, None,
            WALL_TIME_SCOPE,
            ObservationIdentity(None, benchmark_request.suite.identity(),
                                ToolchainIdentity(language, "controlled"), None,
                                ExecutionIdentity("local")),
            allowed_cpus=(1, 3), returncode=0,
        )

    native = ProjectPythonBackend if language == "python" else ProjectJavaBackend
    monkeypatch.setattr(native, "benchmark", observed)
    # The warmed operation may be absent (raising=False). These tests check
    # that it is never selected, not warmed execution.
    monkeypatch.setattr(TestOperations, "benchmark_warmed", warmed_forbidden, raising=False)
    monkeypatch.setattr(native, "benchmark_warmed", warmed_forbidden, raising=False)
    if language == "java":
        monkeypatch.setattr(ProjectJavaBackend, "file_rule", lambda self, source: "CandidateTest")
    adapter = GenericBackend("python", (".py",)) if language == "python" else GenericBackend("java", (".java",))
    return adapter, root, target, test, calls


def _unchanged_scores(result):
    return {key: value for key, value in asdict(result).items()
            if not key.startswith("suite_timing_")}


@pytest.mark.parametrize("fresh_timing", ["python", "java"], indirect=True)
@pytest.mark.parametrize("warmup_runs", [0, 1])
def test_metrics_keeps_fresh_requests_samples_and_scores(fresh_timing, warmup_runs, monkeypatch):
    adapter, root, target, test, calls = fresh_timing
    set_phase_value(monkeypatch, metrics._TIMING_CONFIG, "warmup_count", warmup_runs)
    set_phase_value(monkeypatch, metrics._TIMING_CONFIG, "run_count", 3)
    set_phase_value(monkeypatch, metrics._AGENT_CONFIG, "coverage_timeout_seconds", 19)
    result = metrics.CombinedTestMetrics(target_file_path=str(target))
    before = _unchanged_scores(result)

    metrics._measure_suite_time(result, adapter, [str(test)], str(root), str(target))

    assert len(calls) == 1
    assert (calls[0].runs, calls[0].warmup_runs, calls[0].timeout_s) == (3, warmup_runs, 19)
    assert result.suite_timing_median_ms == 100.0
    assert result.suite_timing_samples_ms == "99.0000;100.0000;101.0000"
    assert result.suite_timing_run_count == 3 and result.suite_timing_warmup_count == warmup_runs
    assert result.suite_timing_requested_cpus is None
    assert result.suite_timing_allowed_cpus == "1,3"
    assert _unchanged_scores(result) == before


@pytest.mark.parametrize("fresh_timing", ["python", "java"], indirect=True)
@pytest.mark.parametrize("warmup_runs", [0, 1])
def test_phase_end_tie_uses_fresh_median_not_saved_runtime(fresh_timing, warmup_runs):
    adapter, root, target, _, calls = fresh_timing
    candidates = [consolidate.ScoredSuite(
        label=label, source=f"// {label}\npublic class CandidateTest {{}}\n"
        if target.suffix == ".java" else f"# {label}\ndef test_ok(): pass\n",
        line=50.0, branch=50.0, mutation=50.0, green=True, test_count=1,
        runtime_ms=1.0 if label == "baseline" else 10_000.0,
    ) for label in ("baseline", "agent")]
    runtime_of = consolidate.TiedCandidateTimer(
        adapter, source_dir=str(root), target_file_path=str(target), timeout=19,
        warmup_runs=warmup_runs, benchmark_runs=3, stage_as_delivered=False, log=lambda message: None,
    )

    winner = consolidate.pick_phase_result(candidates, runtime_of=runtime_of, required_axes=consolidate._PHASE_END_AXES)

    # Each round executes both candidates once, the first one rotating.
    executions = 2 * (warmup_runs + 3)
    assert winner.label == "agent"
    assert len(calls) == executions
    assert all((call.runs, call.warmup_runs, call.timeout_s) == (1, 0, 19)
               for call in calls)
    rounds = [{"round": number + 1, "warmup": number < warmup_runs,
               "order": ["baseline", "agent"] if number % 2 == 0 else ["agent", "baseline"]}
              for number in range(warmup_runs + 3)]
    assert winner.suite_timing == {
        "warmup_runs": warmup_runs, "timing_run_count": 3,
        "samples_ms": [100.0, 100.0, 100.0], "median_ms": 100.0,
        "order": "interleaved", "batch": 1, "rounds": rounds,
    }
    assert runtime_of(winner) == 100.0 and len(calls) == executions
    assert (winner.line, winner.branch, winner.mutation) == (50.0, 50.0, 50.0)


def test_bundled_timing_defaults_and_public_cli_do_not_select_warmed_mode(tmp_path, in_run):
    settings = resolve_settings(base_dir=tmp_path, config_files=[], use_environment=False,
                                models={})
    in_run.leave()
    with config_module.use_run_settings(settings):
        assert metrics._bool_flag(metrics._TIMING_CONFIG, "measure") is False
        assert int(metrics._TIMING_CONFIG["warmup_count"]) == 1
        assert int(metrics._TIMING_CONFIG["run_count"]) == 5
    assert "warmed" not in _build_parser().format_help().lower()


@pytest.mark.parametrize("setting,value", [("runtime_mode", "warmed"), ("forks", 3),
                                          ("state_policy", "native_lifecycle")])
def test_warmed_backend_controls_are_not_testuary_settings(tmp_path, setting, value):
    with pytest.raises(ValueError, match="unknown setting"):
        resolve_settings(base_dir=tmp_path, config_files=[], use_environment=False,
                         models={}, overrides={"finishing": {setting: value}})


@pytest.mark.parametrize("language", ["python", "java"])
@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_existing_timing_settings_do_not_change_generation_origin(tmp_path, language, phase, in_run):
    models = {"chosen": {"provider": "openai", "model": "controlled",
                         "api_url": "https://example.test/v1", "max_output_tokens": 80}}
    base = {"api": {"model": "chosen"}, "main": {"hints_file": "", "seed": 37}}
    conditions = []
    for timing in ({}, {"measure": True, "warmup_count": 0, "run_count": 30, "cpuset": "1,3"}):
        settings = resolve_settings(
            base_dir=tmp_path, config_files=[], use_environment=False, models=models,
            overrides={**base, "timing": timing},
        )
        in_run.leave()
        with config_module.use_run_settings(settings):
            conditions.append(current_condition(
                phase, language, str(tmp_path),
                templates={"system_prompt": "controlled", "user_prompt_template": "source"},
            ).export())
    assert conditions[0] == conditions[1]
