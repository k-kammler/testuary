"""Language-neutral structural observations through ranking and result writers."""

import csv
import json
from types import SimpleNamespace

import pytest
from langadapter.sources import SourceObservation, SourceStatus, TestMetrics

from agent_prompts import python_prompts

from testuary import languages
from testuary.agent import _helpers
from testuary import measurement as _source_metrics
from testuary import results as csv_writer
from testuary import config
from testuary import project
from testuary.agent import coverage_agent as coverage, mutation_agent as mutation
from testuary import measurement

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture(params=["python", "java"])
def suite(request, tmp_path):
    language = request.param
    backend = languages.get_backend(language)
    target = tmp_path / ("thing.py" if language == "python" else "Thing.java")
    target.write_text(
        "# comment\ndef thing():\n    return 1\n" if language == "python"
        else "// comment\n/* comment */\nclass Thing { int value() { return 1; } }\n",
        encoding="utf-8",
    )
    delivered = backend.target_test_path(str(tmp_path), str(target))

    def source(count):
        if language == "python":
            return "# comment\n" + "".join(
                f"def test_{i}():\n    assert True\n" for i in range(count)
            )
        return ("// comment\n/* comment */\nimport org.junit.jupiter.api.Test;\n"
                "class ThingTest {\n" + "".join(
                    f"    @Test void test{i}() {{}}\n" for i in range(count)
                ) + "}\n")

    delivered.write_text(source(2), encoding="utf-8")
    return SimpleNamespace(language=language, backend=backend, target=target,
                           delivered=delivered, source=source)


def make_coverage(suite):
    deps = coverage.CoverageTestDeps(
        target_file_path=str(suite.target),
        target_file_content=suite.target.read_text(encoding="utf-8"),
        source_dir=str(suite.target.parent),
        test_output_dir=str(suite.delivered.parent), language=suite.language,
    )
    session = coverage.TestCoverageSession()
    session.active_backend = suite.backend
    return coverage.CoverageAgent(prompts=python_prompts("test_coverage"), deps=deps, session=session)


def read_row(directory):
    with (directory / "tests.csv").open(encoding="utf-8-sig") as handle:
        return next(csv.DictReader(handle))


@pytest.mark.parametrize("count", [0, 2])
def test_source_counts_keep_observed_zero_and_language_comments(suite, count):
    suite.delivered.write_text(suite.source(count), encoding="utf-8")
    observed = measurement._analyze_test_file(suite.delivered)
    assert observed.test_count == count
    assert observed.source_observation["status"] == "completed"
    assert observed.source_observation["parser"]
    assert observed.loc == (2 * count if suite.language == "python" else 3 + count)


@pytest.mark.parametrize("failure", ["missing", "unreadable", "parse", "unsupported"])
def test_unavailable_source_counts_retain_reason(suite, monkeypatch, failure):
    if failure == "missing":
        suite.delivered.unlink()
    elif failure == "parse":
        suite.delivered.write_text("{ { {", encoding="utf-8")
    elif failure == "unreadable":
        suite.delivered.write_bytes(b"\xff\xfe")
    else:
        monkeypatch.setattr(type(suite.backend), "test_metrics_observation", lambda self, _: SourceObservation(
            SourceStatus.UNSUPPORTED, reason="counts are not supported", parser="custom",
        ))
    observed = measurement._analyze_test_file(suite.delivered)
    assert all(getattr(observed, key) is None for key in (
        "loc", "test_count", "assertion_count", "fixture_count", "parametrize_case_count",
    ))
    assert observed.source_observation["status"] != "completed"
    assert observed.source_observation["reason"]


def test_best_green_and_live_restore_use_backend_test_counts(suite):
    """The snapshot records the backend's test count. The count decides
    nothing: a green live file with fewer tests and an equal score is
    restored to the earlier snapshot."""
    agent = make_coverage(suite)
    agent._maybe_update_best_green(str(suite.delivered.parent), str(suite.target))
    snapshot = agent.session.best_green.snapshot
    assert snapshot["test_count"] == 2
    assert snapshot["test_source_observation"]["status"] == "completed"
    suite.delivered.write_text(suite.source(1), encoding="utf-8")
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_run_tests = False
    agent._maybe_restore_last_green(str(suite.delivered.parent), str(suite.target))
    assert suite.delivered.read_text(encoding="utf-8") == suite.source(2)


@pytest.mark.parametrize("ranker", [coverage._green_snapshot_key, _helpers.green_snapshot_key])
def test_test_count_does_not_rank_a_suite(ranker):
    """A known, zero or unknown test count gives the same key: equal
    scores keep the earlier snapshot whatever the counts."""
    base = {"coverage": 90, "branch_coverage": 90, "mutation": 90,
            "exec_time_ms": 1, "snapshot_clock_s": 1}
    assert ranker({**base, "test_count": 0}) == ranker({**base, "test_count": 2})
    assert ranker({**base, "test_count": 2}) == ranker({**base, "test_count": None})
    assert ranker({**base, "test_count": 2}) == ranker(base)


def test_unavailable_snapshot_retains_none_and_reason(suite, monkeypatch):
    monkeypatch.setattr(type(suite.backend), "test_metrics_observation", lambda self, _: SourceObservation(
        SourceStatus.UNSUPPORTED, reason="unsupported counting", parser="custom",
    ))
    agent = make_coverage(suite)
    agent._maybe_update_best_green(str(suite.delivered.parent), str(suite.target))
    assert agent.session.best_green.snapshot["test_count"] is None
    assert agent.session.best_green.snapshot["test_source_observation"]["reason"]


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
@pytest.mark.parametrize("available", [True, False])
def test_actual_finalizers_record_structural_availability(suite, monkeypatch, phase, available):
    if not available:
        monkeypatch.setattr(type(suite.backend), "test_metrics_observation", lambda self, _: SourceObservation(
            SourceStatus.UNSUPPORTED, reason="unsupported counting", parser="custom",
        ))
    if phase == "coverage":
        agent = make_coverage(suite)
        output = coverage.CoverageTestResult(
            test_file_name="model.py", explanation="", targeted_functions=[],
        )
        role = "delivered"
    else:
        deps = mutation.MutationTestDeps(
            target_file_path=str(suite.target),
            target_file_content=suite.target.read_text(encoding="utf-8"),
            source_dir=str(suite.target.parent),
            test_output_dir=str(suite.delivered.parent), language=suite.language,
            mutation_test_file_name=suite.delivered.name,
        )
        agent = mutation.MutationAgent(
            prompts=python_prompts("test_mutation"), deps=deps, session=mutation.TestMutationSession(),
        )
        output = mutation.MutationTestResult(
            mutation_test_file_name="model.py", explanation="", targeted_mutants=[],
        )
        role = "mutation"
    monkeypatch.setattr(agent, "assert_target_has_no_shadowed_definitions", lambda _: None)
    usage = SimpleNamespace(input_tokens_used=0, output_tokens_used=0, cache_hit_tokens_used=0)
    result = agent.finalize_result(output, error_msg=None, duration_ms=1000,
                                   usage_tracker=usage, final_messages=[])
    field = "test_count_standard" if phase == "coverage" else "test_count_mutation"
    assert getattr(result, field) == (2 if available else None)
    observation = result.source_observations[role]
    assert observation["status"] == ("completed" if available else "unsupported")
    assert bool(observation["reason"]) is not available
    assert "source_observations" not in type(result).model_json_schema()["properties"]


def test_csv_reuse_discovers_language_deliverable_and_keeps_metadata(suite, tmp_path):
    csv_writer.write_tests_row(
        tmp_path, workspace="rank1_csv", target_number=1, file_path=str(suite.target),
        work_tests=suite.delivered.parent,
    )
    row = read_row(tmp_path)
    assert row["test_count_delivered"] == "2"
    assert row["target_loc"] == ("2" if suite.language == "python" else "1")
    observations = json.loads(row["source_observations"])
    assert observations["delivered"]["tests"] == 2
    assert observations["target"]["status"] == "completed"


@pytest.mark.parametrize("role", ["delivered", "mutation", "target"])
def test_csv_new_unavailable_observation_clears_stale_counts(suite, tmp_path, role):
    csv_writer.write_tests_row(
        tmp_path, workspace="rank1_csv", target_number=1, file_path=str(suite.target),
        test_result=SimpleNamespace(test_count_standard=99, target_loc=99,
                                    target_complexity=99),
        mutation_result=SimpleNamespace(test_count_mutation=99),
    )
    observed = _source_metrics.record(SourceObservation(
        SourceStatus.PARSE_ERROR, reason="invalid source", parser="custom",
    ))
    finishing = measurement.CombinedTestMetrics(target_file_path=str(suite.target))
    if role == "target":
        finishing.target_source_observation = observed
    else:
        result = SimpleNamespace(source_observations={role: observed})
    csv_writer.write_tests_row(
        tmp_path, workspace="rank1_csv", target_number=1, file_path=str(suite.target), preserve_existing=True,
        test_result=result if role == "delivered" else None,
        mutation_result=result if role == "mutation" else None,
        finishing_result=finishing if role == "target" else None,
    )
    row = read_row(tmp_path)
    key = {"target": "target_loc", "delivered": "test_count_delivered",
           "mutation": "mutation_test_count"}[role]
    assert row[key] == ""
    assert json.loads(row["source_observations"])[role]["reason"] == "invalid source"
    if role == "target":
        assert row["target_complexity"] == ""


def test_csv_timing_only_update_keeps_prior_structural_observation(suite, tmp_path):
    observed = measurement._analyze_test_file(suite.delivered)
    finishing = measurement.CombinedTestMetrics(target_file_path=str(suite.target), test_files=[observed])
    csv_writer.write_tests_row(tmp_path, workspace="rank1_csv", target_number=1, file_path=str(suite.target),
                               finishing_result=finishing)
    before = read_row(tmp_path)
    suite.delivered.write_text("bad source {", encoding="utf-8")
    timing = measurement.CombinedTestMetrics(target_file_path=str(suite.target),
                                         suite_timing_median_ms=5)
    csv_writer.write_tests_row(tmp_path, workspace="rank1_csv", target_number=1, file_path=str(suite.target),
                               preserve_existing=True, finishing_result=timing,
                               work_tests=suite.delivered.parent)
    after = read_row(tmp_path)
    assert after["test_count_delivered"] == before["test_count_delivered"]
    assert after["source_observations"] == before["source_observations"]
    assert after["suite_timing_median_ms"] == "5"


def test_csv_measured_zero_is_not_replaced_from_a_different_source(suite, tmp_path):
    observed = _source_metrics.record(suite.backend.test_metrics_observation(suite.source(0)))
    result = SimpleNamespace(test_count_standard=0, source_observations={"delivered": observed})
    csv_writer.write_tests_row(tmp_path, workspace="rank1_csv", target_number=1, file_path=str(suite.target),
                               test_result=result,
                               work_tests=suite.delivered.parent)
    row = read_row(tmp_path)
    assert row["test_count_delivered"] == "0"
    assert json.loads(row["source_observations"])["delivered"]["status"] == "completed"


@pytest.mark.parametrize("value", [None, 0, 3])
def test_complexity_wrapper_preserves_unavailable_and_zero(monkeypatch, value):
    monkeypatch.setattr(_source_metrics, "compute_file_complexity_stats", lambda _: (value, value))
    assert _source_metrics.compute_file_complexity("target.py") == value


def test_completed_partial_observation_keeps_missing_counts(suite, monkeypatch):
    monkeypatch.setattr(type(suite.backend), "test_metrics_observation", lambda self, _: SourceObservation(
        SourceStatus.COMPLETED, TestMetrics(parsed=True, parser="custom", code_lines=4),
        parser="custom",
    ))
    observed = measurement._analyze_test_file(suite.delivered)
    assert observed.loc == 4
    assert observed.test_count is None
    assert observed.source_observation["status"] == "completed"


@pytest.mark.parametrize("container", [False, True])
def test_actual_finishing_step_uses_target_language_counts(suite, monkeypatch, container):
    monkeypatch.setattr(config, "project_exec_mode", lambda: "container" if container else "local")
    target_text = suite.target.read_text(encoding="utf-8")
    monkeypatch.setattr(project, "read_project_text", lambda _: target_text)
    monkeypatch.setattr(_source_metrics, "compute_file_complexity_stats", lambda _: (None, None))
    monkeypatch.setattr(measurement, "_RAW_METRICS_CONFIG", {"measure_lint": "false"})
    monkeypatch.setattr(measurement, "_cov_via_backend", lambda *args: (80, 70, False))
    measured = measurement.measure_combined(
        target_file_path=str(suite.target),
        test_output_dir=str(suite.delivered.parent), source_dir=str(suite.target.parent),
        test_file_names=[suite.delivered.name],
    )
    assert measured.target_loc == (2 if suite.language == "python" else 1)
    assert measured.target_complexity_sum is None
    assert measured.test_files[0].test_count == 2
    assert measured.test_files[0].cyclomatic_complexity_sum is None
    assert measured.coverage_final_line == 80
    assert measured.coverage_final_branch == 70
    row = measured.to_csv_dict()
    assert row["source_observations"]["target"]["status"] == "completed"
    assert row["source_observations"]["delivered"]["tests"] == 2


def test_csv_failed_complexity_measurement_does_not_keep_old_value(suite, tmp_path):
    csv_writer.write_tests_row(tmp_path, workspace="rank1_csv", target_number=1, file_path=str(suite.target),
                               test_result=SimpleNamespace(target_complexity=9))
    finishing = measurement.CombinedTestMetrics(
        target_file_path=str(suite.target), target_source_observation=_source_metrics.record(
            _source_metrics.observe_file(suite.target),
        ), target_complexity_sum=None,
    )
    csv_writer.write_tests_row(tmp_path, workspace="rank1_csv", target_number=1, file_path=str(suite.target),
                               preserve_existing=True, finishing_result=finishing)
    row = read_row(tmp_path)
    assert row["target_complexity"] == ""
    assert row["target_loc"] == ("2" if suite.language == "python" else "1")


def test_missing_language_is_unavailable_without_python_fallback(tmp_path):
    path = tmp_path / "test_suite.unknown"
    path.write_text("def test_a():\n    assert True\n", encoding="utf-8")
    result = measurement._analyze_test_file(path)
    assert result.loc is None
    assert result.test_count is None
    assert result.source_observation["status"] == "unsupported"
