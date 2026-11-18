"""Native execution adapters forward public group requests and observations."""
import contextlib
import dataclasses
import sys

import pytest

from langadapter import Interpreter, Outcome, ProjectContext, TestOperations
from langadapter.java.adapter import JavaAdapter
from langadapter.python.adapter import PythonAdapter
from langadapter.outcomes import ExecutionIdentity, FileIdentity, ObservationIdentity, ToolchainIdentity
from langadapter.runs import (
    Benchmark, BenchmarkProgress, CollectionRun, CoverageObservation, KILLS_NOT_REQUESTED,
    PerTestEvidence, RunCounts, RunDiagnostic, SuiteRun, TestEvidence, TestOutcome,
    WALL_TIME_SCOPE,
)
from langadapter.suites import LineItem, TestId

from testuary import normal_run
from testuary.languages.generic import GenericBackend


class RecordingTests(TestOperations):
    def __init__(self, language, test):
        self.requests = []
        self.identity = ObservationIdentity(None, (), ToolchainIdentity(language, "fixture"),
                                            None, ExecutionIdentity("local"))
        self.coverage = CoverageObservation(FileIdentity("target", "fixture"), (1, 2), (), (), ())
        self.run = SuiteRun(
            Outcome.TEST_FAILURES, "native failure", RunCounts(1, 0, 1, 0, 0),
            (TestOutcome(test, "failed", 0.003, "native assertion"),), (), 0.01,
            "F\n1 failed in 0.01s", self.coverage, self.identity, returncode=1,
            runner_counts=RunCounts(1, 0, 1, 0, 0))
        self.evidence = PerTestEvidence(
            Outcome.COMPLETED, "", (TestEvidence(test, frozenset({LineItem("target", 1)}), None),),
            (test,), (), KILLS_NOT_REQUESTED, (), 3.0, self.coverage, self.identity)

    def _record(self, name, request):
        request.validate()
        self.requests.append((name, request))

    def run_suite(self, request):
        self._record("run_suite", request)
        return self.run

    def collect_suite(self, request):
        self._record("collect_suite", request)
        return CollectionRun(Outcome.COLLECTION_FAILED, "cannot collect", self.identity,
                             "native collection output", "native collection diagnostic")

    def per_test_evidence(self, request):
        self._record("per_test_evidence", request)
        return self.evidence

    def benchmark(self, request):
        self._record("benchmark", request)
        if request.progress is not None:
            request.progress(BenchmarkProgress("rep", 1, request.runs, 2.0))
        return Benchmark(Outcome.COMPLETED, "", (2.0, 4.0), request.warmup_runs,
                         request.cpus, WALL_TIME_SCOPE, self.identity, allowed_cpus=(2, 3), returncode=0)


class DeliveringTests(RecordingTests):
    """A test group that runs a suite as delivered."""

    def run_as_delivered(self, request, *, sources=None):
        self._record("run_as_delivered", request)
        return self.run


def _delivering(group):
    """``group``'s adapter with a test group that runs as delivered."""
    delivering = DeliveringTests(group.adapter.language, group.run.tests[0].test)
    delivering.adapter = group.adapter
    # No public call sets an adapter's test group, so the double replaces the private one.
    group.adapter._test_operations = delivering
    return delivering


@contextlib.contextmanager
def pass_through_measurement(backend, suite, *targets):
    """Stands in for the owned measurement copy: yields its inputs unchanged."""
    yield backend, suite, targets


@pytest.fixture(params=[lambda: GenericBackend("python", (".py",)), lambda: GenericBackend("java", (".java",))], ids=["python", "java"])
def adapter(request, tmp_path, monkeypatch):
    language = request.param()
    filename, qualname = (("test_example.py", "test_bad") if language.name == "python"
                          else ("ExampleTest.java", "ExampleTest.bad"))
    test = tmp_path / filename
    test.write_text("fixture source", encoding="utf-8")
    group = RecordingTests(language.name, TestId(filename, qualname))
    # The language's own adapter, its test group the double's.
    adapter = PythonAdapter if language.name == "python" else JavaAdapter
    backend = adapter(ProjectContext(str(tmp_path), Interpreter(path=sys.executable)))
    backend._test_operations = group
    group.adapter = backend
    monkeypatch.setattr(language, "project_backend", lambda root=None: backend)
    module = __import__(type(language).__module__, fromlist=["config_module"])
    monkeypatch.setattr(module.config_module, "project_exec_mode", lambda: "local")
    monkeypatch.setattr(module.benchmark_lock, "acquire", contextlib.nullcontext)
    monkeypatch.setattr(module, "prepared_measurement", pass_through_measurement)
    return language, group, test


def test_failure_listing_uses_group_and_retains_native_ids(adapter, tmp_path):
    language, group, test = adapter
    assert language.failing_test_ids([str(test)], str(tmp_path), timeout=7) == [
        "test_example.test_bad" if language.name == "python" else "ExampleTest.bad"]
    name, request = group.requests[0]
    assert name == "run_suite" and request.timeout_s == 7
    assert request.coverage_target is None


@pytest.mark.parametrize("outcome", [Outcome.TIMED_OUT, Outcome.MISSING_SETUP])
def test_undecided_listing_is_never_green(adapter, tmp_path, outcome):
    language, group, test = adapter
    group.run = dataclasses.replace(group.run, outcome=outcome, coverage=None)
    assert language.failing_test_ids([str(test)], str(tmp_path), timeout=7) is None


@pytest.mark.parametrize("collects", [False, True], ids=["no collection step", "collection step"])
def test_a_run_that_does_not_compile_reports_its_reason_and_compiler_lines(
        adapter, tmp_path, monkeypatch, collects):
    """Without a collection step the compiler's lines are the run's
    reason_detail, as many as collection_errors allows. The compile_error
    diagnostic only marks the cause. A language that collects reports no
    reason."""
    language, group, test = adapter
    offered = group.supports
    monkeypatch.setattr(group, "supports",
                        lambda name: (collects or name != "collect_suite") and offered(name))
    lines = [f"test_cbor.c:{n}: error: unknown type name 'cbor_item'" for n in range(1, 7)]
    group.run = dataclasses.replace(
        group.run, outcome=Outcome.COLLECTION_FAILED, reason="the test file does not compile",
        reason_detail=tuple(lines),
        diagnostics=(RunDiagnostic("compile_error", "the test file does not compile"),),
        coverage=None)
    report = {}
    assert language.failing_test_ids([str(test)], str(tmp_path), timeout=7, report=report) is None
    expected = {"outcome": "collection_failed"}
    if not collects:
        expected["reason"] = "the test file does not compile: " + " | ".join(
            [*lines[:5], "[... 1 further error omitted ...]"])
    assert report == expected


def test_a_language_without_a_collection_step_describes_no_collection_failure(
        adapter, tmp_path, monkeypatch):
    """A test group without ``collect_suite`` leaves the reason to the run."""
    language, group, test = adapter
    offered = group.supports
    monkeypatch.setattr(group, "supports", lambda name: name != "collect_suite" and offered(name))
    assert language.describe_collection_failure([str(test)], str(tmp_path), timeout=11) is None
    assert group.requests == []


def test_python_and_java_state_the_journal_and_the_seed_as_their_classes_do(adapter, tmp_path):
    """Read on the project's adapter, Python's and Java's facts are their
    classes': both take the journal, only Python seeds its runs as
    delivered."""
    language, group, _test = adapter
    adapter_class = type(group.adapter)
    assert language.offers_timeout_journal(str(tmp_path)) is adapter_class.provides_timeout_journal
    assert language.seeds_delivered_runs(str(tmp_path)) is adapter_class.seeds_delivered_runs
    assert (language.offers_timeout_journal(str(tmp_path)),
            language.seeds_delivered_runs(str(tmp_path))) == (
        (True, True) if language.name == "python" else (True, False))


def test_a_run_result_holds_the_id_objects_of_its_failing_tests(adapter, tmp_path):
    """``stopping_tests`` holds the test id objects of the run's failed and
    errored tests, beside ``failures``' texts, so a removal needs no text
    matching."""
    language, group, test = adapter
    result = language.run_test_files([str(test)], str(tmp_path), timeout=7)
    failing = group.run.tests[0].test
    assert result.stopping_tests == (failing,)
    assert isinstance(result.stopping_tests[0], TestId)
    assert [failure.outcome for failure in result.failures] == ["failed"]


def test_a_failed_runner_still_lists_the_reports_failures(adapter, tmp_path):
    """A failed Python runner still lists the report's failures for merging
    to strip. Java keeps its own rule."""
    language, group, test = adapter
    group.run = dataclasses.replace(group.run, outcome=Outcome.INFRASTRUCTURE_FAILURE,
                                    coverage=None)
    expected = ["test_example.test_bad"] if language.name == "python" else None
    assert language.failing_test_ids([str(test)], str(tmp_path), timeout=7) == expected


def test_run_retains_genuine_zero_coverage_and_native_failure(adapter, tmp_path):
    language, group, test = adapter
    result = language.run_test_files([str(test)], str(tmp_path), timeout=8,
                                     measure_coverage_of="target", fail_fast=True, record_leaks=True)
    assert not result.ok and result.returncode == 1 and result.failed == 1
    assert result.line_coverage == 0.0 and result.branch_coverage is None
    assert result.output == "F\n1 failed in 0.01s" and not result.timed_out
    name, request = group.requests[0]
    assert name == "run_suite" and request.coverage_target == "target"
    assert request.timeout_s == 8
    if language.name == "python":
        assert request.isolation_config and request.remove_created_files
        assert request.record_leaks
        assert request.runner_options.stop_at_first_failure


def test_missing_measurement_is_not_zero_coverage(adapter, tmp_path):
    language, group, test = adapter
    group.run = dataclasses.replace(group.run, outcome=Outcome.MISSING_SETUP,
                                    coverage=None, counts=RunCounts(), returncode=None)
    result = language.run_test_files([str(test)], str(tmp_path), timeout=8,
                                     measure_coverage_of="target")
    assert not result.ok and result.line_coverage is None and result.branch_coverage is None


def test_evidence_retains_items_and_missing_test_time(adapter, tmp_path):
    language, group, test = adapter
    result = language.collect_eviction_data([str(test)], str(tmp_path),
                                            measure_coverage_of="target", timeout=9, warmup=False)
    assert result.per_test_time_ms == {}
    assert list(result.per_test_tokens.values()) == [frozenset({"L:target:1"})]
    assert not result.includes_kills and result.suite_time_ms == 3.0
    name, request = group.requests[0]
    assert name == "per_test_evidence" and not request.warmup and request.timeout_s == 9


def test_failed_evidence_hands_on_the_tests_it_names(adapter, tmp_path):
    """An evidence run that fails naming its tests (``stopping_tests``) hands
    their id objects to ``failure_fn`` beside its kind and reason."""
    from types import SimpleNamespace

    language, group, test = adapter
    named = group.run.tests[0].test
    group.evidence = SimpleNamespace(outcome=Outcome.TEST_FAILURES, reason="t failed",
                                     stopping_tests=(named,))
    calls = []
    result = language.collect_eviction_data(
        [str(test)], str(tmp_path), measure_coverage_of="target", timeout=9,
        failure_fn=lambda *args, **kwargs: calls.append((args, kwargs)))
    assert result is None
    assert calls == [(("test_failure", "t failed"), {"stopping_tests": (named,)})]


def test_timed_out_evidence_hands_on_no_test(adapter, tmp_path):
    """An evidence run the timeout ended hands on none of the tests its
    result names, as a timed-out mutation check names none."""
    from types import SimpleNamespace

    language, group, test = adapter
    group.evidence = SimpleNamespace(outcome=Outcome.TIMED_OUT, reason="t timed out",
                                     stopping_tests=(group.run.tests[0].test,))
    calls = []
    result = language.collect_eviction_data(
        [str(test)], str(tmp_path), measure_coverage_of="target", timeout=9,
        failure_fn=lambda *args, **kwargs: calls.append((args, kwargs)))
    assert result is None
    assert calls == [(("infrastructure", "t timed out"), {})]


def test_failed_evidence_has_no_usable_reduction(adapter, tmp_path):
    language, group, test = adapter
    group.evidence = dataclasses.replace(group.evidence, outcome=Outcome.INFRASTRUCTURE_FAILURE,
                                         reason="native evidence failed")
    failures = []
    result = language.collect_eviction_data([str(test)], str(tmp_path),
                                            measure_coverage_of="target", timeout=9,
                                            failure_fn=lambda *args: failures.append(args))
    assert result is None and failures == [("infrastructure", "native evidence failed")]


def test_benchmark_retains_requested_repetitions_affinity_and_progress(adapter, tmp_path):
    language, group, test = adapter
    events = []
    result = language.benchmark_tests(source_dir=str(tmp_path), test_file_paths=[str(test)],
                                      warmup_runs=1, benchmark_runs=2, timeout=10, progress_fn=events.append)
    assert result["ok"] and result["times_ms"] == [2.0, 4.0]
    assert (result["min_ms"], result["max_ms"], result["median_ms"]) == (2.0, 4.0, 3.0)
    assert result["requested_cpus"] is None and result["allowed_cpus"] == [2, 3]
    assert events == ["[benchmark:delivered] rep 1/2 (2ms)"]
    name, request = group.requests[0]
    assert name == "benchmark" and request.runs == 2 and request.warmup_runs == 1
    assert request.timeout_s == 10 and not request.serialize


@pytest.mark.parametrize("adapter", [lambda: GenericBackend("python", (".py",))],
                         indirect=True, ids=["python"])
@pytest.mark.parametrize("delivers", [True, False], ids=["delivered", "no-delivered"])
def test_the_normal_run_is_a_run_as_delivered_only_where_the_language_has_one(
        adapter, tmp_path, delivers):
    """Python's normal run is its run as delivered where it has runs as
    delivered, else run_test_files."""
    language, group, test = adapter
    if delivers:
        group = _delivering(group)
    else:
        group.adapter.run_as_delivered = None
    result = normal_run.run(language, [str(test)], str(tmp_path), timeout=8)
    assert [name for name, _ in group.requests] == (
        ["run_as_delivered"] if delivers else ["run_suite"])
    assert not result.ok and result.failed == 1


@pytest.mark.parametrize("adapter", [lambda: GenericBackend("python", (".py",))],
                         indirect=True, ids=["python"])
def test_the_run_as_delivered_hands_langadapter_the_normal_runs_wishes(adapter, tmp_path):
    """The normal run as delivered asks for the first-failure stop, the
    traceback, the leak record, the timeout journal where the language
    offers it, the session seed and the module under test in the copy."""
    language, group, test = adapter
    group = _delivering(group)
    normal_run.run(language, [str(test)], str(tmp_path), timeout=8, target="target",
                   fail_fast=True, tb="line", record_leaks=True, timeout_journal=True)
    (name, request), = group.requests
    assert name == "run_as_delivered" and request.timeout_s == 8
    assert request.runner_options.stop_at_first_failure
    assert request.runner_options.traceback == "line"
    assert request.record_leaks
    assert request.timeout_journal is language.offers_timeout_journal()
    assert request.session_seed == normal_run.SESSION_SEED
    assert request.module_under_test == "target"


@pytest.mark.parametrize("adapter", [lambda: GenericBackend("python", (".py",))],
                         indirect=True, ids=["python"])
def test_python_collection_diagnostic_uses_group(adapter, tmp_path):
    language, group, test = adapter
    assert language.describe_collection_failure([str(test)], str(tmp_path), timeout=11) == "native collection diagnostic"
    name, request = group.requests[0]
    assert name == "collect_suite" and request.timeout_s == 11
