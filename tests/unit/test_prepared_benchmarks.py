"""Benchmark adapter requests use one public owned project for all samples."""
from contextlib import contextmanager
from pathlib import Path
import subprocess
import sys

import langadapter as lb
from langadapter.errors import InvalidRequest
from langadapter.outcomes import ExecutionIdentity, ObservationIdentity, ToolchainIdentity
from langadapter.java.adapter import JavaAdapter as ProjectJavaBackend
from langadapter.python.adapter import PythonAdapter as ProjectPythonBackend
from langadapter.runs import Benchmark, BenchmarkProgress, WALL_TIME_SCOPE
import pytest

from testuary import config
from testuary import project
from testuary.languages import preparation
from testuary.languages.generic import GenericBackend

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture(autouse=True)
def no_native_tools(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Controlled benchmark staging must not execute tools")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    # The bytecode store's helper is a native command; its own tests run it.
    monkeypatch.setattr(preparation, "bytecode_store", lambda backend: {})
    monkeypatch.setattr(config, "project_exec_mode", lambda: "local")
    monkeypatch.setattr(project, "project_work_dir", lambda: None)
    # Each language's runtime is this interpreter, selected in the settings,
    # and the timing lock is on, its default.
    monkeypatch.setattr(config, "load_config", lambda section: {
        "pytest_config": "none", "frozen_module_strategy": "skip",
        "python_executable": sys.executable, "java_home": sys.executable,
        "serialize": "true"})


def _project(tmp_path, language, inside):
    root = tmp_path / "project"
    root.mkdir()
    name, source = (("subject.py", b"value = 1\n") if language == "python"
                    else ("Subject.java", b"public class Subject {}\n"))
    target = root / name
    target.write_bytes(source)
    delivered = root if inside else tmp_path / "delivered"
    delivered.mkdir(exist_ok=True)
    test = delivered / ("test_subject.py" if language == "python" else "SubjectTest.java")
    test.write_bytes(b"# controlled fixture\n" if language == "python"
                     else b"public class SubjectTest {}\n")
    if language == "python":
        (delivered / "conftest.py").write_bytes(b"# declared setup\n")
    return root, target, test


def _identity(language, request):
    return ObservationIdentity(None, request.suite.identity(),
                               ToolchainIdentity(language, "fixture"), None,
                               ExecutionIdentity("local"))


@pytest.mark.parametrize("language", ["python", "java"])
@pytest.mark.parametrize("inside", [False, True])
@pytest.mark.parametrize("outcome", [lb.Outcome.COMPLETED, lb.Outcome.TIMED_OUT,
                                    lb.Outcome.TEST_FAILURES, lb.Outcome.INFRASTRUCTURE_FAILURE])
def test_benchmark_preserves_request_samples_progress_and_original_inputs(
        tmp_path, monkeypatch, language, inside, outcome):
    root, target, test = _project(tmp_path, language, inside)
    original_target, original_test = target.read_bytes(), test.read_bytes()
    requests, owners, contexts, messages = [], [], [], []
    prepare = preparation.prepare_project

    def record_owner(context, **options):
        owner = prepare(context, **options)
        owners.append(owner)
        return owner

    def observed(self, request):
        request.validate()
        copy = Path(self.context.root)
        assert copy != root and copy.is_dir()
        assert Path(owners[0].root).is_dir()
        assert (copy / target.name).read_bytes() == original_target
        if inside:
            assert request.suite.root == str(copy)
            assert request.suite.test_files == (str(copy / test.name),)
        elif language == "python":
            # An outside suite is copied beside the project's prepared copy, below their shared directory.
            suite = copy.parent / test.parent.name
            assert request.suite.root == str(suite)
            assert request.suite.test_files == (str(suite / test.name),)
        else:
            assert request.suite.root == str(test.parent)
            assert request.suite.test_files == (str(test),)
        assert request.runs == 3 and request.warmup_runs == 2 and request.timeout_s == 19
        assert request.serialize is False and request.cpus is None
        if language == "python":
            assert request.module_under_test == str(copy / target.name)
            assert request.isolation_config
            assert request.suite.setup_files == (str(Path(request.suite.root) / "conftest.py"),)
        else:
            assert request.module_under_test is None
        requests.append(request)
        contexts.append(self.context)
        request.progress(BenchmarkProgress("warmup", 1, 2, 7))
        request.progress(BenchmarkProgress("measured", 1, 3, 11))
        (copy / target.name).write_bytes(b"changed only in owned copy\n")
        return Benchmark(outcome, "native diagnostic", (11.0, 13.0), 2, None,
                         WALL_TIME_SCOPE, _identity(language, request),
                         allowed_cpus=(1, 3), returncode=0 if outcome == lb.Outcome.COMPLETED else 7)

    monkeypatch.setattr(preparation, "prepare_project", record_owner)
    project_backend = ProjectPythonBackend if language == "python" else ProjectJavaBackend
    monkeypatch.setattr(project_backend, "benchmark", observed)
    adapter = GenericBackend("python", (".py",)) if language == "python" else GenericBackend("java", (".java",))
    result = adapter.benchmark_tests(source_dir=str(root), test_file_paths=[str(test)],
                                    warmup_runs=2, benchmark_runs=3, timeout=19,
                                    suite_label="candidate", progress_fn=messages.append,
                                    instrument_target=str(target))
    assert len(owners) == len(requests) == 1
    assert not Path(owners[0].root).exists()
    assert target.read_bytes() == original_target and test.read_bytes() == original_test
    assert result["ok"] is (outcome == lb.Outcome.COMPLETED)
    assert result["times_ms"] == [11.0, 13.0]
    assert result["requested_cpus"] is None and result["allowed_cpus"] == [1, 3]
    assert messages == ["[benchmark:candidate] warmup 1/2 (7ms)",
                        "[benchmark:candidate] rep 1/3 (11ms)"]
    if outcome == lb.Outcome.COMPLETED:
        assert result["median_ms"] == 12.0 and result["min_ms"] == 11.0
    else:
        assert "median_ms" not in result
    with pytest.raises(InvalidRequest, match="root: no directory"):
        lb.create(language, contexts[0]).tests.benchmark(requests[0])


@pytest.mark.parametrize("language", ["python", "java"])
def test_benchmark_error_retains_primary_and_uncertain_owner(tmp_path, monkeypatch, language):
    root, target, test = _project(tmp_path, language, False)
    native_error = KeyboardInterrupt("native benchmark interrupted")
    cleanup_error = OSError("owner disposal unresolved")
    owner = lb.prepare_project(lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable)))
    monkeypatch.setattr(preparation, "prepare_project", lambda context, **options: owner)
    original_close = owner.close
    project_backend = ProjectPythonBackend if language == "python" else ProjectJavaBackend
    monkeypatch.setattr(project_backend, "benchmark", lambda *args: (_ for _ in ()).throw(native_error))
    adapter = GenericBackend("python", (".py",)) if language == "python" else GenericBackend("java", (".java",))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(owner, "close", lambda: (_ for _ in ()).throw(cleanup_error))
            with pytest.raises(KeyboardInterrupt) as caught:
                adapter.benchmark_tests(source_dir=str(root), test_file_paths=[str(test)],
                                        warmup_runs=1, benchmark_runs=1, timeout=2)
        assert caught.value is native_error
        assert owner in lb.recovery_owners(native_error)
        assert native_error.preparation_cleanup_error is cleanup_error
        assert Path(owner.root).is_dir() and target.is_file()
    finally:
        original_close()


def test_java_guest_benchmark_forwards_selected_cpu_set_without_native_execution(tmp_path, monkeypatch):
    root, _, test = _project(tmp_path, "java", False)
    contexts, requests = [], []
    selected = lb.create("java", lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable)))

    @contextmanager
    def staged(backend, suite, *targets):
        contexts.append(backend.context)
        yield selected, suite, ()

    def observed(self, request):
        requests.append(request)
        return Benchmark(lb.Outcome.COMPLETED, "", (1,), 0, request.cpus, WALL_TIME_SCOPE,
                         _identity("java", request), allowed_cpus=(2, 3))

    # Only the adapter's guest CPU request is checked here, not guest copying or native placement.
    monkeypatch.setattr(config, "project_exec_mode", lambda: "container")
    monkeypatch.setattr(config, "load_config", lambda section: {"cpuset": "2-3", "serialize": "true"})
    monkeypatch.setattr("testuary.languages.generic.prepared_measurement", staged)
    monkeypatch.setattr(ProjectJavaBackend, "benchmark", observed)
    adapter = GenericBackend("java", (".java",))
    monkeypatch.setattr(adapter, "project_backend", lambda root: selected)
    result = adapter.benchmark_tests(source_dir=str(root), test_file_paths=[str(test)],
                                    warmup_runs=0, benchmark_runs=1, timeout=1)
    assert len(contexts) == len(requests) == 1 and requests[0].cpus == "2-3"
    assert result["requested_cpus"] == "2-3" and result["allowed_cpus"] == [2, 3]
