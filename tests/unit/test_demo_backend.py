"""The demonstration's backend and engine: every operation that starts a
process is answered from the scenario, and the answers reach the phases in
the backend's own types."""
from __future__ import annotations

import ast
import hashlib
import inspect
import json

import pytest

from testuary.demo import backend as demo_backend
from testuary.demo import recorded
from testuary.demo import scenario as sc
from testuary.generation_evidence import source_sha256
from testuary import languages
from testuary import project
from testuary.languages import mutation
from testuary.languages.generic import GenericBackend

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


class _EveryDir(dict):
    """A recording's language facts, stated for every project directory."""

    def __init__(self, record):
        super().__init__()
        self._record = record

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return self._record


@pytest.fixture(autouse=True)
def _recorded_facts(monkeypatch):
    """The backends here play outside a playback, so the shipped Python
    recording's language facts stand for every project directory."""
    from testuary.demo import backend as scenario_backend
    from testuary.demo import cli as demo_cli

    record = next(iter(demo_cli.load_data("python")[0]["language_facts"].values()))
    monkeypatch.setattr(scenario_backend, "RECORDED_LANGUAGE", _EveryDir(record))


def test_a_recorded_run_keeps_the_counts_it_did_not_observe():
    """A run without its record, such as a crash, observes no count: the
    recording keeps None for each, never 0."""
    from testuary.languages import TestRunResult
    run = TestRunResult(False, -11, False, 0.1, None, None, None, None, "Segmentation fault", None, None)
    data = json.loads(json.dumps(recorded.run_result_json(run)))
    assert recorded.test_run_result(data) == run


def test_a_recorded_run_keeps_the_target_totals_it_reported():
    """A scenario's run keeps the target's statement and branch totals its
    coverage report gave, and a run without them is stored without the
    keys, as the shipped recordings hold it."""
    from testuary.languages import TestRunResult
    run = TestRunResult(True, 0, False, 0.1, 1, 1, 0, 0, "out", 100.0, 100.0,
                        total_statements=7, total_branches=2)
    data = json.loads(json.dumps(recorded.run_result_json(run)))
    assert recorded.test_run_result(data) == run
    bare = recorded.run_result_json(run._replace(total_statements=None, total_branches=None))
    assert "total_statements" not in bare and "total_branches" not in bare


def test_a_recorded_run_keeps_its_failures():
    """A scenario's run keeps the failing tests of its report. A run
    recorded before the field has none."""
    from testuary.languages import TestFailure, TestRunResult
    run = TestRunResult(False, 1, False, 0.1, 2, 1, 1, 0, "out", None, None,
                        failures=(TestFailure("demo.T.a", "failed", "boom", "at x"),))
    data = json.loads(json.dumps(recorded.run_result_json(run)))
    assert recorded.test_run_result(data) == run
    del data["failures"]
    assert recorded.test_run_result(data).failures == ()


def test_a_recorded_run_keeps_its_unsound_entries():
    """A scenario's run keeps the unsound-test entries of its report, and a
    run without any is stored without the key, as the shipped recordings
    hold it."""
    from testuary.languages import DeliveredRun, TestRunResult
    from testuary.languages.generic import DeliveredCase
    unsound = (("demo.T.a", "coroutine was never awaited"), (None, "Task was destroyed"))
    run = TestRunResult(True, 0, False, 0.1, 1, 1, 0, 0, "out", None, None, unsound=unsound)
    data = json.loads(json.dumps(recorded.run_result_json(run)))
    assert recorded.test_run_result(data) == run
    assert "unsound" not in recorded.run_result_json(run._replace(unsound=()))
    delivered = DeliveredRun((DeliveredCase("demo.T.a", "a", "passed"),), (), None, "out",
                             unsound=unsound)
    data = json.loads(json.dumps(recorded.delivered_run_json(delivered)))
    assert recorded.delivered_run(data) == delivered
    assert "unsound" not in recorded.delivered_run_json(delivered._replace(unsound=()))


def test_a_recorded_run_keeps_its_diagnostics_and_reason_detail():
    """A scenario's run keeps each diagnostic's kind and message and its
    reason detail, and a run without either is stored without its key, as
    the shipped recordings hold it."""
    from langadapter.runs import RunDiagnostic
    from testuary.languages import TestRunResult
    diagnostics = (RunDiagnostic("compile_error", "the test file test_calc.c does not compile"),)
    run = TestRunResult(False, -1, False, 0.1, 0, 0, 0, 0, "out", None, None,
                        diagnostics=diagnostics,
                        reason_detail=("test_calc.c:4:5: error: expected ';'",))
    data = json.loads(json.dumps(recorded.run_result_json(run)))
    assert recorded.test_run_result(data) == run
    bare = recorded.run_result_json(run._replace(diagnostics=(), reason_detail=()))
    assert "diagnostics" not in bare and "reason_detail" not in bare


def _method_nodes(cls, within=None) -> dict:
    """Each method of ``cls`` as its most-derived definition's AST node,
    parsed from the defining modules (a method's own source cannot always be
    parsed alone: a string in it may start at column 0), only the classes of
    the module ``within`` when given."""
    nodes = {}
    for klass in reversed(cls.__mro__):
        if klass is object or (within is not None and inspect.getmodule(klass) is not within):
            continue
        tree = ast.parse(inspect.getsource(inspect.getmodule(klass)))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == klass.__name__:
                for item in node.body:
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        nodes[item.name] = item
    return nodes


def _executing_methods(cls) -> dict:
    """The methods of ``cls`` that start a process, directly through the
    executor or ``subprocess``, or through another such method of the class,
    with whether each is public."""
    methods = _method_nodes(cls)
    direct, calls = set(), {}
    for name, tree in methods.items():
        called = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute):
                if node.attr in ("run", "run_timed", "validate_source", "benchmark",
                                 "run_suite", "run_as_delivered", "per_test_evidence",
                                 "collect_suite"):
                    direct.add(name)
                if (isinstance(node.value, ast.Name)
                        and node.value.id in ("self", "cls", cls.__name__)):
                    called.add(node.attr)
            if (isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name)
                    and node.value.id == "subprocess" and node.attr in ("run", "Popen")):
                direct.add(name)
        calls[name] = called & set(methods)
    executing = set(direct)
    changed = True
    while changed:
        changed = False
        for name, called in calls.items():
            if name not in executing and called & executing:
                executing.add(name)
                changed = True
    return {name: not name.startswith("_") for name in executing}


class _ExecutionScanFixture:
    def executor(self):
        return project.get_executor()

    def pure(self):
        return self.executor()

    def direct(self):
        return self.executor().run(["python", "--version"])

    def delegated(self):
        return self.direct()

    def validation(self):
        return self.executor().validate_source("pass")

    def timing(self):
        return self.executor().benchmark(None)

    def evidence(self):
        return self.executor().per_test_evidence(None)


def test_execution_scan_distinguishes_a_factory_from_running_a_command():
    assert _executing_methods(_ExecutionScanFixture) == {
        "direct": True, "delegated": True, "validation": True, "timing": True,
        "evidence": True,
    }


def test_pure_source_operations_start_no_process(monkeypatch):
    import subprocess

    def refuse(*args, **kwargs):
        pytest.fail("pure source operations must start no process")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    backend = demo_backend.ScenarioBackend(_Oracle({}))
    source = "def test_kept():\n    pass\n\ndef test_dropped():\n    pass\n"
    assert backend.check_structure(source).ok
    assert backend.structural_parser_identity()
    pruned = backend.prune_tests(source, ["test_dropped"])
    assert "test_kept" in pruned and "test_dropped" not in pruned
    assert "test_kept" in backend.finalize_test_source(pruned)
    assert backend.normalize_shadowed_definitions(pruned).parse_ok
    assert not backend.resolve_test_collisions([pruned, pruned]).ejected


def test_every_executing_backend_operation_is_answered_from_the_scenario():
    executing = _executing_methods(GenericBackend)
    public = sorted(name for name, is_public in executing.items() if is_public)
    # The contract is only meaningful if the scan finds the known operations.
    assert {"run_test_files", "failing_test_ids", "collect_eviction_data",
            "validate_test_source", "benchmark_tests"} <= set(public)
    own = {**demo_backend.ScenarioOperations.__dict__,
           **demo_backend.ScenarioBackend.__dict__}
    missing = [name for name in public
               if name not in own and name != "salvage_syntax"]
    assert missing == []
    # salvage_syntax executes only through validate_test_source, which the
    # scenario answers.
    source = inspect.getsource(demo_backend.ScenarioBackend.salvage_syntax)
    assert "get_executor" not in source and "validate_test_source" in source
    # The recorded calls' methods ask the oracle only.
    for entry in (*recorded.BACKEND_CALLS, recorded.MERGE_TEST_FILES):
        assert getattr(demo_backend.ScenarioBackend, entry.method).__module__ == recorded.__name__
    # No scenario backend operation written out calls an unanswered
    # executing one.
    unanswered = {name for name in executing if name not in own}
    for name, node in _method_nodes(demo_backend.ScenarioBackend, demo_backend).items():
        if name not in own:
            continue
        called = {
            sub.attr for sub in ast.walk(node)
            if isinstance(sub, ast.Attribute) and isinstance(sub.value, ast.Name)
            and sub.value.id in ("self", "super")
        }
        assert not called & unanswered, (name, called & unanswered)
        assert not any(
            isinstance(sub, ast.Attribute) and sub.attr == "get_executor"
            for sub in ast.walk(node)), name


def test_every_mutation_operation_is_answered_from_the_scenario():
    """The runner's sessions, runs, catalogue, survivors' text and kill
    matrices come from the scenario. The configured catalogue follows the
    real rule, which executes nothing."""
    for cls, base, names in (
        (demo_backend.ScenarioRunner, mutation.MutationRunner,
         ("open", "full_operators")),
        (demo_backend.ScenarioMutationSession, mutation.MutationSession,
         ("run", "close")),
        (demo_backend.ScenarioMeasurement, mutation.MutationMeasurement,
         ("survivors_text", "kill_matrix")),
    ):
        for name in names:
            assert name in cls.__dict__, (cls.__name__, name)
            assert hasattr(base, name)


class _Oracle:
    def __init__(self, answers):
        self.answers = answers
        self.calls = []

    def answer(self, call):
        self.calls.append(call)
        return self.answers[call.kind]

    def enter_segment(self, name):
        pass


@pytest.mark.parametrize("options", [
    {"early_stop_baseline_ms": 12},
    {"early_stop_factor": 3},
])
def test_demo_rejects_unknown_benchmark_options_before_replay(options):
    oracle = _Oracle({})
    backend = demo_backend.ScenarioBackend(oracle)
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        backend.benchmark_tests(
            source_dir="absent", test_file_paths=[], warmup_runs=1,
            benchmark_runs=2, timeout=60, **options,
        )
    assert oracle.calls == []


def test_the_dependency_check_probes_no_tool(monkeypatch):
    """The Python backend's check starts CBC to read its version, but the
    demonstration uses no tool and its guard refuses every process."""
    import subprocess

    def refuse(*args, **kwargs):
        pytest.fail("the dependency check must start no process")

    monkeypatch.setattr(subprocess, "run", refuse)
    monkeypatch.setattr(subprocess, "Popen", refuse)
    backend = demo_backend.ScenarioBackend(_Oracle({}))
    assert backend.check_dependencies() == []


@pytest.mark.parametrize("backend_class",
                         [demo_backend.ScenarioBackend, demo_backend.ScenarioJavaBackend])
def test_python_and_java_state_the_recorded_capability_report(backend_class):
    """The run decides its fallbacks from the capability report, so a
    playback states the one the recorded run found, for every language."""
    from langadapter.capabilities import Availability, Capability, CapabilityReport
    from langadapter.outcomes import Outcome

    name = backend_class(_Oracle({})).name
    report = CapabilityReport(
        language=name, interpreter="", outcome=Outcome.COMPLETED, reason="", runtime=None,
        capabilities=(Capability("mutation", Availability.MISSING_SETUP, "no engine"),),
        tools=(), branch_coverage="none", mutation_engines=())
    oracle = _Oracle({"backend.capabilities": {"report": recorded.report_json(report)}})
    assert backend_class(oracle).capabilities(source_dir="anywhere") == report
    (call,) = oracle.calls
    assert call.kind == "backend.capabilities" and call.key == {"language": name}


def test_the_runs_behind_the_notes_are_answered_from_the_scenario(tmp_path):
    """A run as delivered and a test run that records leaks, the notes'
    observations, are answered like every execution, keyed by what they
    run, and come back as the backend's own result types."""
    test_file = tmp_path / "test_x.py"
    test_file.write_text("def test_a():\n    pass\n", encoding="utf-8")
    leak = {"test": "test_x.test_a", "nodeid": "test_x.py::test_a", "added": ["helper"],
            "rebound": [], "removed": [], "xarray_options": [], "numpy_errors": [],
            "numpy_print": []}
    oracle = _Oracle({
        "backend.run_as_delivered": {"run": {
            "cases": [{"test": "test_x.test_a", "name": "test_a", "outcome": "passed",
                       "message": ""}],
            "failing": [], "reason": None, "output": "1 passed", "reverse": True, "seed": 3,
            "timed_out": False, "elapsed_s": 0.4}},
        "backend.run_test_files": {"result": _result(leaks=[leak])},
    })
    backend = demo_backend.ScenarioBackend(oracle)
    delivered = backend.run_as_delivered([str(test_file)], str(tmp_path), timeout=60,
                                         reverse=True, seed=3)
    assert delivered.ok and delivered.cases[0].name == "test_a"
    result = backend.run_test_files([str(test_file)], str(tmp_path), timeout=60,
                                    record_leaks=True)
    assert result.leaks[0].added == ("helper",)
    keys = [call.key for call in oracle.calls]
    assert keys[0]["reverse"] is True and keys[0]["seed"] == 3
    assert keys[1]["record_leaks"] is True
    # A run without the record keeps the key the scenarios hold.
    backend.run_test_files([str(test_file)], str(tmp_path), timeout=60)
    assert "record_leaks" not in oracle.calls[-1].key


def test_a_run_as_delivered_keys_its_target_and_is_test_time(tmp_path):
    """The target decides which file a run as delivered imports, so it is
    keyed whenever given, with or without the leak record, and the run is
    test time, as a test run is."""
    from testuary import phase_state

    test_file = tmp_path / "test_x.py"
    test_file.write_text("def test_a():\n    pass\n", encoding="utf-8")
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    oracle = _Oracle({"backend.run_as_delivered": {"run": {
        "cases": [], "failing": [], "reason": None, "output": "", "reverse": False,
        "seed": 1, "timed_out": False, "elapsed_s": None}}})
    backend = demo_backend.ScenarioBackend(oracle)
    target_dir = tmp_path / "target_number"
    target_dir.mkdir()
    with phase_state.timed_segment(target_dir, "coverage"):
        backend.run_as_delivered([str(test_file)], str(tmp_path), timeout=60,
                                 target=str(target))
    key = oracle.calls[0].key
    assert key["target"] == ["mod.py", sc.content_sha256(target)]
    assert "record_leaks" not in key
    assert "test" in phase_state.output_time_parts(target_dir, "coverage")


def _result(**overrides):
    data = {
        "ok": True, "returncode": 0, "timed_out": False, "elapsed_s": 1.5,
        "collected": 3, "passed": 3, "failed": 0, "errors": 0, "output": "",
        "line_coverage": 90.0, "branch_coverage": 80.0,
        "missing_lines": [4], "missing_branches": ["2->exit"],
        "covered_lines": [1, 2, 3], "covered_branches": ["1->2"],
    }
    data.update(overrides)
    return data


def test_a_test_run_is_answered_and_its_sidecar_written(tmp_path):
    suite = tmp_path / "suite"
    suite.mkdir()
    test_file = suite / "test_x.py"
    test_file.write_text("def test_a():\n    pass\n", encoding="utf-8")
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    matrix = {"lines": [{"line": 1, "status": "covered"}], "branches": [],
              "summary": {"total_lines": 1}}
    oracle = _Oracle({"backend.run_test_files": {
        "result": _result(), "coverage_matrix": matrix}})
    backend = demo_backend.ScenarioBackend(oracle)
    sidecar = tmp_path / "matrix.json"
    result = backend.run_test_files(
        [str(test_file)], str(tmp_path), timeout=10,
        measure_coverage_of=str(target), coverage_matrix_path=str(sidecar))
    assert isinstance(result, languages.TestRunResult)
    assert result.missing_lines == (4,) and result.covered_lines == (1, 2, 3)
    written = json.loads(sidecar.read_text(encoding="utf-8"))
    assert written["target_file"] == str(target)
    assert written["lines"] == matrix["lines"]
    # No setup file is placed beside the suite.
    assert not (suite / "conftest.py").is_file()
    key = oracle.calls[0].key
    assert key["files"] == [["test_x.py", sc.content_sha256(test_file)]]
    assert key["coverage_of"] == ["mod.py", sc.content_sha256(target)]


def test_the_coverage_sidecar_is_written_back_as_the_backend_wrote_it(tmp_path):
    """The scenario stores the sidecar's payload as the real backend wrote it,
    without the recording's target path, and writes it back with this run's."""
    real = tmp_path / "real.json"
    payload = {"schema_version": 3, "lines": [], "branches": [], "summary": {},
               "tool": "coverage.py", "target_file": "recording/mod.py"}
    real.write_text(json.dumps(payload), encoding="utf-8")
    out = tmp_path / "out.json"
    recorded.write_coverage_matrix("playback/mod.py", recorded.read_coverage_matrix(real), out)
    assert json.loads(out.read_text(encoding="utf-8")) == {**payload, "target_file": "playback/mod.py"}


def test_a_listing_without_placement_writes_no_setup_file(tmp_path):
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def test_a():\n    assert True\n", encoding="utf-8")
    backend = demo_backend.ScenarioBackend(_Oracle({"backend.failing_test_ids": {"ids": []}}))
    backend.failing_test_ids([str(candidate)], str(tmp_path), timeout=10)
    assert not (tmp_path / "conftest.py").exists()


def test_eviction_data_without_a_suite_time_round_trips():
    from testuary.languages.generic import EvictionData

    data = EvictionData(per_test_tokens={"t.test_a": frozenset({"L:mod.py:1"})},
                        per_test_time_ms={}, suite_time_ms=None, includes_kills=False)
    rebuilt = recorded.eviction_data(recorded.eviction_data_json(data))
    assert rebuilt.suite_time_ms is None
    assert rebuilt.per_test_tokens == data.per_test_tokens


def test_eviction_data_is_rebuilt_in_the_backend_type(tmp_path):
    test_file = tmp_path / "test_candidate.py"
    test_file.write_text("def test_a():\n    pass\n", encoding="utf-8")
    data = {
        "per_test_tokens": {"t.test_a": ["L:mod.py:1"]},
        "per_test_time_ms": {"t.test_a": 1.25}, "suite_time_ms": 1.25,
        "includes_kills": False, "target_total_lines": 2,
        "target_covered_lines": 1, "target_total_branches": 0,
        "target_covered_branches": 0, "suite_covered_lines": [1],
        "suite_covered_branches": [],
    }
    logged = []
    oracle = _Oracle({"backend.collect_eviction_data": {
        "data": data, "log": ["[evict] note"], "failure": None}})
    evidence = demo_backend.ScenarioBackend(oracle).collect_eviction_data(
        [str(test_file)], str(tmp_path), measure_coverage_of=str(test_file),
        timeout=10, log=logged.append)
    assert evidence.per_test_tokens == {"t.test_a": frozenset({"L:mod.py:1"})}
    assert evidence.suite_covered_lines == frozenset({1})
    assert logged == ["[evict] note"]


def test_a_validation_verdict_names_the_source_it_judged():
    oracle = _Oracle({"backend.validate_test_source": {"validation": {
        "ok": True, "status": "valid", "message": "", "line": 0, "col": 0,
        "validator_identity": "CPython 3.10.15 (x)"}}})
    verdict = demo_backend.ScenarioBackend(oracle).validate_test_source(
        "x = 1\n", ".")
    assert verdict.ok and verdict.validator_identity == "CPython 3.10.15 (x)"
    assert verdict.source_sha256 == source_sha256("x = 1\n")


def _measured(**overrides):
    """A mutation run's measurement as the scenario stores it."""
    data = {"score": 50.0, "killed": 1, "survived": 1, "timeout": 0, "non_viable": 0,
            "untested": 0, "total": 2, "truncated": False, "reason": "",
            "suite_failed": False, "stopping_tests": [], "lower_bound": False,
            "timeout_rule": None}
    data.update(overrides)
    return data


def test_a_delivered_run_keeps_its_timeout_and_time():
    """A stored run as delivered keeps whether the timeout ended it and its
    wall time, which the mutation limits and the limit record read."""
    from testuary.languages.generic import DeliveredRun

    run = DeliveredRun(cases=(), failing=None, reason="timeout", output="",
                       timed_out=True, elapsed_s=12.5)
    assert recorded.delivered_run(recorded.delivered_run_json(run)) == run


def test_a_delivered_run_keeps_its_named_tests():
    """A stored run as delivered keeps the id objects of the tests it
    names, which a measurement's removal reads, and a run that names none
    is stored without them."""
    from langadapter.suites import TestId
    from testuary.languages.generic import DeliveredRun

    failing = TestId("test_calc.py", "test_add", text="test_calc.py::test_add")
    run = DeliveredRun(cases=(), failing=("test_calc.test_add",), reason=None, output="",
                       stopping_tests=(failing,))
    stored = json.loads(json.dumps(recorded.delivered_run_json(run)))
    assert recorded.delivered_run(stored) == run
    assert "stopping_tests" not in recorded.delivered_run_json(run._replace(stopping_tests=()))


def test_a_mutation_measurement_keeps_its_reason_stop_and_timeout(tmp_path):
    """A stored mutation run keeps its reason, its stop, its lower bound and
    the per-mutant timeout the engine applied, which choose and fill the
    replies the model reads."""
    from langadapter.suites import TestId

    data = _measured(killed=0, survived=0, untested=2, score=0.0,
                     reason="the suite failed", suite_failed=True,
                     stopping_tests=[{"file": "test_a.py", "qualname": "test_a", "params": None,
                                      "text": "test_a.py::test_a"}],
                     lower_bound=True,
                     timeout_rule={"value": 10, "source": "request", "inputs": {}})
    measured = demo_backend.ScenarioMeasurement(None, "key", data)
    assert (measured.reason, measured.suite_failed, measured.stopping_tests,
            measured.lower_bound, measured.timeout_rule()) == (
        "the suite failed", True, (TestId("test_a.py", "test_a"),), True, data["timeout_rule"])
    assert recorded.MUTATION_RUN.write(measured, None, {}) == {"results": data}


def test_the_failing_test_listing_keeps_its_report(tmp_path):
    """The listing's report (the run's outcome) reaches the caller's dict on
    playback, as the real listing fills it."""
    candidate = tmp_path / "candidate.py"
    candidate.write_text("def test_a():\n    assert True\n", encoding="utf-8")
    backend = demo_backend.ScenarioBackend(_Oracle({"backend.failing_test_ids": {
        "ids": None, "report": {"outcome": "collection_error"}}}))
    report = {}
    assert backend.failing_test_ids([str(candidate)], str(tmp_path), timeout=10,
                                    report=report) is None
    assert report == {"outcome": "collection_error"}


def test_the_runner_answers_a_run_and_captures_its_matrix(tmp_path):
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    matrix = {"schema_version": 3, "target_file": "mod.py", "mutants": [],
              "summary": {"total": 0, "detected": 0, "survived": 0}}
    oracle = _Oracle({
        "mutation.run": {"results": _measured()},
        "mutation.capture_kill_matrix": {"kill_matrix": matrix},
    })
    runner = demo_backend.ScenarioRunner(
        oracle, {"cosmic_ray_distributor": "http", "cosmic_ray_operators": ""})
    session = runner.open(15)
    assert session.workers == 15 and session.use_http
    results = session.run(target=target, test_dir=tests,
                          test_files=[str(tests / "test_x.py")],
                          per_mutant_timeout=30, wall_clock_timeout=3600)
    assert (results.killed, results.survived) == (1, 1)
    run_key = oracle.calls[-1].key
    assert run_key["operators"] is None and run_key["fail_fast"] is True
    # A runner that records every killing test is keyed as the run without -x.
    runner.attribution = "all"
    session.run(target=target, test_dir=tests, test_files=[str(tests / "test_x.py")],
                per_mutant_timeout=30, wall_clock_timeout=3600)
    assert oracle.calls[-1].key["fail_fast"] is False
    out = tmp_path / "km.json"
    assert mutation.write_kill_matrix(results, out)
    captured = json.loads(out.read_text(encoding="utf-8"))
    assert captured["summary"] == matrix["summary"]
    assert {"t_mono_s", "clock_domain_id"} <= set(captured)
    capture_key = oracle.calls[-1].key
    assert capture_key == {
        "run": hashlib.sha256(sc.canonical(run_key).encode()).hexdigest()}


def test_a_scenario_survivor_sample_takes_the_measurement_parameters(tmp_path):
    """The mutation phase asks for the sample with the limits' size and change
    length, and the scenario answers the text recorded for both."""
    import inspect

    assert (list(inspect.signature(demo_backend.ScenarioMeasurement.survivors_text).parameters)
            == list(inspect.signature(mutation.MutationMeasurement.survivors_text).parameters))
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    oracle = _Oracle({
        "mutation.run": {"results": _measured()},
        "mutation.render_surviving_diffs": {"text": "--- mutant m1 (line 1) ---\n-x = 1\n+x = 2"},
    })
    runner = demo_backend.ScenarioRunner(
        oracle, {"cosmic_ray_distributor": "http", "cosmic_ray_operators": ""})
    results = runner.open(15).run(target=target, test_dir=tests,
                                  test_files=[str(tests / "test_x.py")],
                                  per_mutant_timeout=30, wall_clock_timeout=3600)
    assert results.survivors_text(max_mutants=10, max_chars=240).startswith("--- mutant m1")
    assert oracle.calls[-1].key["max_mutants"] == 10 and oracle.calls[-1].key["max_chars"] == 240


def test_the_configured_operators_follow_the_engine_rule():
    oracle = _Oracle({})
    runner = demo_backend.ScenarioRunner(oracle, {"cosmic_ray_operators": ""})
    assert runner.configured_operators() is None
    runner = demo_backend.ScenarioRunner(
        oracle, {"cosmic_ray_operators": "a, b"})
    assert runner.configured_operators() == ["a", "b"]
    assert oracle.calls == []


def test_the_python_merge_is_answered_by_the_scenario_not_run():
    """Python's merge is checked in the project's interpreter, a process
    the demonstration does not start: the scenario answers it, keyed by the
    two suites' digests, as Java's merge is."""
    calls = []

    class _Answers:
        def answer(self, call):
            calls.append(call)
            assert call.kind == "backend.merge_test_files"
            return {"source": "merged"}

    def forbidden(*args, **kwargs):
        pytest.fail("the merge ran natively during playback")

    original = GenericBackend.merge_test_files
    GenericBackend.merge_test_files = forbidden
    try:
        backend = demo_backend.ScenarioBackend(_Answers())
        assert backend.merge_test_files("def test_a(): pass\n", "def test_b(): pass\n") == "merged"
    finally:
        GenericBackend.merge_test_files = original
    (call,) = calls
    assert call.key == {"first": source_sha256("def test_a(): pass\n"),
                        "second": source_sha256("def test_b(): pass\n")}


def test_a_project_dir_the_recording_did_not_state_stops_the_playback(tmp_path, monkeypatch):
    monkeypatch.setattr(GenericBackend, "project_backend",
                        lambda self, source_dir=None: object())
    monkeypatch.setattr(demo_backend, "RECORDED_LANGUAGE", {"<elsewhere>": {}})
    with pytest.raises(sc.ScenarioStop, match="holds no language facts"):
        demo_backend.ScenarioBackend(_Oracle({})).project_backend(tmp_path).language_facts()


@pytest.mark.parametrize("module, name", [("backend", "ScenarioBackend"), ("backend", "ScenarioJavaBackend"),
                                          ("backend", "ScenarioGenericBackend")])
def test_the_demonstrations_take_every_option_of_a_test_run(module, name):
    """A playback takes every run option the phases pass (such as the timeout
    journal), or every run_tests would be a tool error."""
    import importlib

    from testuary.languages.generic import GenericBackend

    cls = getattr(importlib.import_module(f"testuary.demo.{module}"), name)
    wanted = set(inspect.signature(GenericBackend.run_test_files).parameters)
    assert wanted <= set(inspect.signature(cls.run_test_files).parameters)


def test_the_timeout_journal_is_asked_in_the_key_only_when_set(tmp_path):
    test = tmp_path / "test_a.py"
    test.write_text("def test_a():\n    assert True\n", encoding="utf-8")
    result = {"ok": True, "returncode": 0, "timed_out": False, "elapsed_s": 0.1, "collected": 1,
              "passed": 1, "failed": 0, "errors": 0, "output": "", "line_coverage": None,
              "branch_coverage": None, "active_at_end": "test_a.test_a"}
    oracle = _Oracle({"backend.run_test_files": {"result": result}})
    backend = demo_backend.ScenarioBackend(oracle)
    plain = backend.run_test_files([str(test)], str(tmp_path), timeout=10)
    journal = backend.run_test_files([str(test)], str(tmp_path), timeout=10, timeout_journal=True)
    first, second = oracle.calls
    assert "timeout_journal" not in first.key and second.key["timeout_journal"] is True
    assert plain.active_at_end == journal.active_at_end == "test_a.test_a"


def test_each_run_is_keyed_by_its_timeout(tmp_path):
    """A run's outcome depends on its time limit, so a test run, a listing
    and a run as delivered each hold it in their key."""
    test = tmp_path / "test_a.py"
    test.write_text("def test_a():\n    assert True\n", encoding="utf-8")
    oracle = _Oracle({
        "backend.run_test_files": {"result": _result()},
        "backend.failing_test_ids": {"ids": [], "report": {}},
        "backend.run_as_delivered": {"run": {
            "cases": [], "failing": [], "reason": None, "output": "", "reverse": False,
            "seed": 1, "timed_out": False, "elapsed_s": None}}})
    backend = demo_backend.ScenarioBackend(oracle)
    backend.run_test_files([str(test)], str(tmp_path), timeout=10)
    backend.failing_test_ids([str(test)], str(tmp_path), timeout=20)
    backend.run_as_delivered([str(test)], str(tmp_path), timeout=30)
    assert [call.key["timeout"] for call in oracle.calls] == [10, 20, 30]

