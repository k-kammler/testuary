"""Testuary's Java language over the language backend's Java operations:
the file and class naming without a toolchain, then, with a JDK and Maven
(``LANGADAPTER_TEST_JAVA_HOMES``, ``LANGADAPTER_TEST_MAVEN``), suite runs,
coverage, failing tests, per-test evidence with kills, PIT through
Testuary's mutation runner, a merge and a benchmark on a small project."""
from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest
from langadapter.context import Interpreter, ProjectContext
from langadapter.java.adapter import JavaAdapter

from java_fixtures import MAIN, POM, jdk_and_maven

from testuary.languages import get_backend
from testuary.languages.generic import GenericBackend

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _java_project(root, **tests):
    """Java's adapter of the project at ``root``, its test group answering
    ``tests``."""
    project = JavaAdapter(ProjectContext(str(root), Interpreter(path=sys.executable)))
    project._test_operations = types.SimpleNamespace(supports=lambda name: True, **tests)
    return project


def test_collection_diagnostic_uses_discovery_not_execution(tmp_path, monkeypatch):
    from types import SimpleNamespace

    from langadapter.outcomes import Outcome
    from langadapter.runs import CollectionRequest

    test = tmp_path / "ExampleTest.java"
    test.write_text("class ExampleTest {}", encoding="utf-8")
    requests = []

    def collect(request):
        requests.append(request)
        return SimpleNamespace(outcome=Outcome.COMPLETED, reason="", output="")

    java = GenericBackend("java", (".java",))
    project = _java_project(tmp_path, collect_suite=collect)
    monkeypatch.setattr(java, "project_backend", lambda root: project)
    assert java.describe_collection_failure([str(test)], str(tmp_path), timeout=12) == ""
    assert len(requests) == 1
    assert isinstance(requests[0], CollectionRequest)
    assert requests[0].timeout_s == 12


def test_delivery_delegates_to_public_backend_without_changing_native_seed(tmp_path, monkeypatch):
    from contextlib import contextmanager
    from types import SimpleNamespace

    from langadapter.outcomes import Outcome
    from testuary.languages import generic as java_language

    test = tmp_path / "ExampleTest.java"
    test.write_text("class ExampleTest {}", encoding="utf-8")
    requests, prepared = [], []

    def delivered(request, *, sources):
        requests.append((request, sources))
        return SimpleNamespace(outcome=Outcome.COMPLETED, reason="", output="native", tests=(),
                               wall_time_s=None, leaks=None, active_at_end=None, diagnostics=())

    project = _java_project(tmp_path, run_as_delivered=delivered)

    @contextmanager
    def measurement(backend, suite, *paths):
        # Delivery runs in a prepared copy: the staged suite reaches the copy's backend.
        prepared.append((backend, suite, paths))
        yield project, suite, ()

    java = GenericBackend("java", (".java",))
    monkeypatch.setattr(java, "project_backend", lambda root: project)
    monkeypatch.setattr(java_language, "prepared_measurement", measurement)
    sources = {str(test): "class ExampleTest { int x; }"}
    result = java.run_as_delivered([str(test)], str(tmp_path), timeout=12, seed=7, sources=sources)
    assert result.failing == ()
    assert result.output == "native"
    assert len(requests) == 1 and len(prepared) == 1
    assert prepared[0][0] is project and prepared[0][1].test_files == (str(test.resolve()),)
    assert requests[0][0].suite is prepared[0][1] and requests[0][0].timeout_s == 12
    assert requests[0][0].session_seed is None
    assert requests[0][1] == {"ExampleTest.java": sources[str(test)]}


def test_a_delivered_run_without_a_verdict_shows_its_detail_to_the_limit(tmp_path, monkeypatch):
    """The reason's detail lines follow the run_reason_lines limit."""
    from contextlib import contextmanager
    from types import SimpleNamespace

    from langadapter.outcomes import Outcome
    from testuary.agent import _limits
    from testuary.languages import generic as java_language

    test = tmp_path / "ExampleTest.java"
    test.write_text("class ExampleTest {}", encoding="utf-8")
    run = SimpleNamespace(outcome=Outcome.INFRASTRUCTURE_FAILURE, output="native", tests=(), wall_time_s=None,
                          leaks=None, active_at_end=None, diagnostics=(),
                          reason_text=lambda lines: f"no verdict, {lines} lines")
    project = _java_project(tmp_path, run_as_delivered=lambda request, *, sources: run)

    @contextmanager
    def measurement(backend, suite, *paths):
        yield project, suite, ()

    java = GenericBackend("java", (".java",))
    monkeypatch.setattr(java, "project_backend", lambda root: project)
    monkeypatch.setattr(java_language, "prepared_measurement", measurement)
    monkeypatch.setattr(_limits, "value", lambda name: {"run_reason_lines": 1}[name])
    result = java.run_as_delivered([str(test)], str(tmp_path), timeout=12)
    assert result.reason == "no verdict, 1 lines"

TEST = """\
package demo;

import static org.junit.jupiter.api.Assertions.assertEquals;

import org.junit.jupiter.api.Test;

class CalcTest {
    @Test
    void positive() {
        assertEquals(1, Calc.sign(5));
    }

    @Test
    void negative() {
        assertEquals(-1, Calc.sign(-5));
    }

    @Test
    void clampsHigh() {
        assertEquals(10, Calc.clamp(50, 0, 10));
    }
}
"""


def test_the_java_language_is_registered_with_its_naming(tmp_path):
    java = get_backend("java")
    assert isinstance(java, GenericBackend) and java.file_extensions == (".java",)
    assert java.target_test_path(str(tmp_path), "/p/src/main/java/demo/Calc.java") \
        == tmp_path / "CalcTest.java"
    assert [java.classify_test_kind(n) for n in (
        "CalcTest.java", "Calc.java", "test_calc.py")] == [
        "coverage", "unknown", "unknown"]
    (tmp_path / "CalcTest.java").write_text(TEST, encoding="utf-8")
    assert java.find_target_test_file(str(tmp_path), "/p/Calc.java") \
        == tmp_path / "CalcTest.java"
    assert java.scratch_test_path(str(tmp_path), TEST, "/p/Calc.java",
                                  stage_as_delivered=False) \
        == tmp_path / "CalcTest.java"


def test_the_static_analyses_need_no_jdk(tmp_path, monkeypatch, in_run):
    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.delenv("TESTUARY_PROJECT_JAVA_HOME", raising=False)
    in_run()
    java = GenericBackend("java", (".java",))
    with pytest.raises(RuntimeError, match="no JDK"):
        java.project_backend(str(tmp_path))
    assert java.check_structure(TEST).ok
    assert java.test_metrics(TEST)["test_count"] == 3
    assert java.skeleton(TEST)[0] == "CalcTest"
    assert java.is_test_file(Path("CalcTest.java"))


@pytest.fixture
def project(tmp_path, monkeypatch, in_run):
    jdk, maven = jdk_and_maven()
    root = tmp_path / "calc"
    (root / "src/main/java/demo").mkdir(parents=True)
    (root / "src/test/java/demo").mkdir(parents=True)
    (root / "src/main/java/demo/Calc.java").write_text(MAIN, encoding="utf-8")
    (root / "pom.xml").write_text(POM, encoding="utf-8")
    monkeypatch.setenv("TESTUARY_PROJECT_JAVA_HOME", str(jdk))
    monkeypatch.setenv("TESTUARY_PROJECT_MAVEN", maven)
    in_run()
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "CalcTest.java").write_text(TEST, encoding="utf-8")
    return root, tests


def test_suite_runs_coverage_and_failing_tests(project):
    root, tests = project
    java = GenericBackend("java", (".java",))
    report = java.capabilities(str(root))
    assert report.capability("suite_runs").available, report.reason
    target = str(root / "src/main/java/demo/Calc.java")
    result = java.run_test_files([str(tests / "CalcTest.java")], str(root),
                                 timeout=300, measure_coverage_of=target)
    assert result.ok and (result.passed, result.failed) == (3, 0), result.output
    assert result.missing_lines == (14,)
    assert result.missing_branches == ("11->1 of 2 branches untaken",)
    assert result.covered_lines == (8, 9, 11, 12, 18)
    red = tests / "CalcRedTest.java"
    red.write_text(TEST.replace("class CalcTest", "class CalcRedTest").replace(
        "assertEquals(1, Calc.sign(5))", "assertEquals(2, Calc.sign(5))"),
        encoding="utf-8")
    assert java.failing_test_ids([str(red)], str(root), timeout=300) == [
        "demo.CalcRedTest.positive"]
    pruned = java.prune_tests(red.read_text(encoding="utf-8"),
                              ["demo.CalcRedTest.positive"])
    assert "void positive()" not in pruned
    delivered = java.run_as_delivered([str(tests / "CalcTest.java")], str(root),
                                      timeout=300)
    assert delivered.ok and len(delivered.cases) == 3
    assert java.run_as_delivered([str(red)], str(root), timeout=300,
                                 reverse=True).reason


@pytest.mark.mutation_tool
def test_evidence_with_kills_and_pit_through_testuarys_runner(project):
    root, tests = project
    java = GenericBackend("java", (".java",))
    target = root / "src/main/java/demo/Calc.java"
    config = {"pit_mutators": "", "engine": "cosmic_ray"}
    runner = java.mutation_runner(config, source_dir=str(root))
    assert runner.configured_operators() is None
    assert len(runner.full_operators()) == 11
    measurement = runner.run_once(
        workers=1, target=target, test_dir=tests, test_files=["CalcTest.java"],
        per_mutant_timeout=10, wall_clock_timeout=600)
    assert (measurement.killed, measurement.survived) == (5, 2)
    assert "changed conditional boundary" in measurement.survivors_text()
    matrix = tests.parent / "kill_matrix.json"
    matrix.write_text(json.dumps(measurement.kill_matrix()), encoding="utf-8")
    data = java.collect_eviction_data(
        [str(tests / "CalcTest.java")], str(root),
        measure_coverage_of=str(target), timeout=300, kill_matrix_path=str(matrix))
    assert data is not None and data.includes_kills
    assert set(data.per_test_tokens) == {
        "demo.CalcTest.positive", "demo.CalcTest.negative", "demo.CalcTest.clampsHigh"}
    assert any(t.startswith("K:") for t in data.per_test_tokens["demo.CalcTest.negative"])
    assert data.target_covered_lines == 5


def test_a_merge_validation_and_a_benchmark(project):
    root, tests = project
    java = GenericBackend("java", (".java",))
    second = TEST.replace("class CalcTest", "class CalcMoreTest").replace(
        "Calc.sign(5)", "Calc.sign(7)")
    resolution = java.resolve_test_collisions([TEST, second], localize_autouse=True)
    assert resolution.ejected == ()
    assert resolution.renames[1] == {"positive": "positive_2"}
    assert resolution.deduped[1] == ("negative", "clampsHigh")
    merged = java.merge_test_files(*resolution.sources)
    assert merged is not None and java.validate_test_source(merged, str(root)).ok
    normalized = java.normalize_shadowed_definitions(merged)
    assert not normalized.changed and normalized.parse_ok
    stats = java.benchmark_tests(source_dir=str(root),
                                 test_file_paths=[str(tests / "CalcTest.java")],
                                 warmup_runs=1, benchmark_runs=2, timeout=300)
    assert stats["ok"] and len(stats["times_ms"]) == 2, stats
    fragments = java.prompt_fragments(str(root))
    assert fragments["lang_test_framework"] == "JUnit 5"
    assert fragments["lang_code_fence"] == "java"


def test_pits_own_timeout_rule_unless_a_value_is_fixed(tmp_path):
    """``auto`` leaves the per-mutant timeout to PIT's rule (each test of a
    mutant's run: 1.25 times its normal time plus 4 seconds). A configured
    value stays fixed."""
    from langadapter import ProjectContext
    from langadapter.java.adapter import JavaAdapter
    from testuary.languages.generic import GenericMutationRunner

    def request(configured, computed):
        runner = GenericMutationRunner(
            JavaAdapter(ProjectContext(str(tmp_path))),
            {"pit_mutators": "", "per_mutant_timeout_seconds": configured}, engine="pit")
        return runner.request(target=tmp_path / "Calc.java", test_dir=tmp_path,
                              test_files=["CalcTest.java"],
                              per_mutant_timeout=computed, wall_clock_timeout=600)

    assert request("auto", 30).per_mutant_timeout is None
    assert request("45", 45).per_mutant_timeout == 45


def test_a_java_test_id_prefix_is_its_class_package_qualified_name(tmp_path):
    """A Java test's id names its class with the package, which finds a
    failing test's file in a pooled run."""
    path = tmp_path / "CalcTest.java"
    path.write_text(
        "package demo;\n\nimport org.junit.jupiter.api.Test;\n\n"
        "class CalcTest {\n    @Test\n    void good() {\n    }\n\n"
        "    @Test\n    void bad() {\n    }\n}\n", encoding="utf-8")
    java = GenericBackend("java", (".java",))
    assert java.test_id_prefix(str(path)) == "demo.CalcTest"


def test_the_project_of_a_java_target_is_its_module(tmp_path):
    """In a multi-module build the project is the target's module, which
    the backend resolves with the modules it uses: the nearest build file
    above the target, a Kotlin Gradle file included."""
    from testuary import project as project_module

    for build_file in ("pom.xml", "build.gradle", "build.gradle.kts"):
        root = tmp_path / build_file.replace(".", "_")
        source = root / "core" / "src" / "main" / "java" / "demo"
        source.mkdir(parents=True)
        top = "pom.xml" if build_file == "pom.xml" else "settings.gradle"
        (root / top).write_text("", encoding="utf-8")
        (root / "core" / build_file).write_text("", encoding="utf-8")
        target = source / "Calc.java"
        target.write_text("class Calc {}\n", encoding="utf-8")
        assert project_module.resolve_source_dir(str(target)) == str(
            (root / "core").resolve())
