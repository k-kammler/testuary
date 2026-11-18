"""Routing of ``GenericBackend.benchmark_tests``: local mode times through the
backend's timing-process launcher, container mode runs every run through the
project executor in the source directory, where the suite is scored. Both build
the anchored pytest command of the backend's other measurement runs."""
from __future__ import annotations

from contextlib import contextmanager
import json
import subprocess
from pathlib import Path

import pytest

from langadapter import placement
from langadapter.python.adapter import PythonAdapter as ProjectPythonBackend

from testuary import config
from testuary import project as project_module
from testuary.languages import get_backend, language_project_context

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _suite(tmp_path: Path) -> tuple[str, list]:
    suite_dir = tmp_path / "gen"
    suite_dir.mkdir()
    test = suite_dir / "test_a.py"
    test.write_text("def test_a():\n    assert True\n", encoding="utf-8")
    project = tmp_path / "repo"
    project.mkdir()
    return str(project), [str(test)]


@pytest.fixture(autouse=True)
def container_context(monkeypatch, tmp_path):
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER", "benchmark-container")
    monkeypatch.setattr(project_module, "project_work_dir", lambda: str(tmp_path))
    monkeypatch.setattr(project_module, "project_repo_dir", lambda: "/testbed")
    original_affinity = ProjectPythonBackend._allowed_cpus

    def affinity(self, prefix, cwd, env):
        if self.context.execution.kind == "local":
            return (0, 1, 2, 3)
        return original_affinity(self, prefix, cwd, env)

    monkeypatch.setattr(ProjectPythonBackend, "_allowed_cpus", affinity)
    original_probe = ProjectPythonBackend._probe

    def toolchain_probe(self):
        if self.context.execution.kind == "local":
            return {"implementation": "CPython", "version": "3.12.0", "tools": {}}, ""
        return original_probe(self)

    monkeypatch.setattr(ProjectPythonBackend, "_probe", toolchain_probe)

    # Container mode runs in place here, without the prepared project copy,
    # which other tests cover. Local mode still uses a copy.
    from testuary.languages.preparation import prepared_measurement

    @contextmanager
    def staging(backend, suite, *paths):
        if backend.context.execution.kind == "container":
            yield backend, suite, paths
        else:
            with prepared_measurement(backend, suite, *paths) as selected:
                yield selected

    monkeypatch.setattr("testuary.languages.generic.prepared_measurement", staging)


def _ok_proc() -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=["pytest"], returncode=0, stdout="1 passed", stderr="")


def test_local_mode_times_through_the_timing_launcher(tmp_path, monkeypatch, timed_run):
    source_dir, files = _suite(tmp_path)
    calls = []

    def fake_timed(command, *, env=None, cwd=None, timeout=None):
        calls.append({"command": list(command), "cwd": cwd, "env": dict(env or {})})
        return _ok_proc()

    monkeypatch.setattr(placement, "run_timed", timed_run(fake_timed))
    monkeypatch.delenv("PYTHONDONTWRITEBYTECODE", raising=False)
    backend = get_backend("python")
    stats = backend.benchmark_tests(
        source_dir=source_dir, test_file_paths=files,
        warmup_runs=1, benchmark_runs=2, timeout=60)

    assert stats["ok"] is True
    assert len(calls) == 3  # 1 warmup and 2 runs
    assert len({c["cwd"] for c in calls}) == 1
    copied_root = Path(calls[0]["cwd"])
    assert copied_root != Path(source_dir) and not copied_root.exists()
    assert Path(source_dir).is_dir()
    cmd = calls[0]["command"]
    # Anchored like every other measurement run.
    assert any(str(a).startswith("--rootdir=") for a in cmd)
    assert "no:cacheprovider" in cmd
    # The timed command disables the logfire and anyio plugins, runs quietly
    # without tracebacks and writes bytecode as usual.
    pairs = list(zip(cmd, cmd[1:]))
    assert ("-p", "no:logfire") in pairs and ("-p", "no:anyio") in pairs
    assert "-q" in cmd and "--tb=no" in cmd
    assert "-B" not in cmd and not any("pycache_prefix" in str(a) for a in cmd)
    assert all("PYTHONDONTWRITEBYTECODE" not in call["env"] for call in calls)


def _answer_the_probe(argv):
    """Answer setup probes and scratch cleanup, not timed runs."""
    if list(argv)[-1:] == ["allowed"]:
        return subprocess.CompletedProcess(
            args=list(argv), returncode=0, stdout="[0, 1, 2, 3]\n",
            stderr="")
    if list(argv)[:1] == ["cat"]:
        marker = Path(argv[1])
        if marker.name == ".langadapter-owner":
            assert marker.parent.name.startswith("langadapter_bench_")
            return subprocess.CompletedProcess(list(argv), 0, marker.read_text(encoding="ascii"), "")
        return subprocess.CompletedProcess(list(argv), 1, "", "no config file")
    if list(argv)[-2:-1] == ["-c"]:
        return subprocess.CompletedProcess(
            list(argv), 0, json.dumps({"implementation": "CPython",
                                      "version": "3.12.0", "tools": {}}), "")
    if list(argv)[-3:-2] == ["-c"]:
        # A helper script on a project path, the check for pytest-benchmark's
        # storage before the runs and its removal after them: none is there.
        return subprocess.CompletedProcess(list(argv), 1, "", "")
    if list(argv)[:2] == ["rm", "-rf"]:
        assert list(argv)[:3] == ["rm", "-rf", "--"]
        assert Path(argv[3]).name.startswith("langadapter_bench_")
        return subprocess.CompletedProcess(list(argv), 0, "", "")
    return None


def test_container_mode_routes_reps_through_executor(tmp_path, monkeypatch, in_run):
    source_dir, files = _suite(tmp_path)
    runs = []

    class _FakeExecutor:
        def run(self, argv, *, cwd=None, env=None, timeout=None, input=None):
            probe = _answer_the_probe(argv)
            if probe is not None:
                return probe
            runs.append({"argv": list(argv), "cwd": cwd})
            return _ok_proc()

    def fail_timed(*a, **k):
        raise AssertionError(
            "container mode must not launch the suite on the host")

    monkeypatch.setattr(config, "project_exec_mode", lambda: "container")
    monkeypatch.setattr("langadapter.python.adapter.executor_for", lambda *args, **kwargs: _FakeExecutor())
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", "/container/python")
    # The container and work directory a container run names.
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER", "fixture-container")
    monkeypatch.setenv("TESTUARY_PROJECT_WORK_DIR", str(tmp_path))
    in_run()
    monkeypatch.setattr(placement, "run_timed", fail_timed)

    backend = get_backend("python")
    stats = backend.benchmark_tests(
        source_dir="/testbed", test_file_paths=files,
        warmup_runs=1, benchmark_runs=2, timeout=60)

    assert stats["ok"] is True
    assert stats["timing_run_count"] == 2
    assert len(runs) == 3  # 1 warmup and 2 runs, all in the container
    assert all(r["cwd"] == "/testbed" for r in runs)
    assert all(r["argv"][0] == "/container/python" for r in runs)
    # No processor set requested. The probe read the allowed one.
    assert stats["requested_cpus"] is None
    assert stats["allowed_cpus"] == [0, 1, 2, 3]


def test_container_mode_pins_reps_to_the_configured_cpuset(tmp_path, monkeypatch, in_run):
    source_dir, files = _suite(tmp_path)
    runs = []

    probes = []

    class _FakeExecutor:
        def run(self, argv, *, cwd=None, env=None, timeout=None, input=None):
            probe = _answer_the_probe(argv)
            if probe is not None:
                if list(argv)[-1:] == ["allowed"]:
                    probes.append((list(argv), input))
                return probe
            runs.append(list(argv))
            return _ok_proc()

    monkeypatch.setattr(config, "project_exec_mode", lambda: "container")
    monkeypatch.setattr("langadapter.python.adapter.executor_for", lambda *args, **kwargs: _FakeExecutor())
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", "/container/python")
    # The container and work directory a container run names.
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER", "fixture-container")
    monkeypatch.setenv("TESTUARY_PROJECT_WORK_DIR", str(tmp_path))
    monkeypatch.setenv("TESTUARY_TIMING_CPUSET", "0-3")
    in_run()

    backend = get_backend("python")
    stats = backend.benchmark_tests(
        source_dir="/testbed", test_file_paths=files,
        warmup_runs=1, benchmark_runs=1, timeout=60)

    assert stats["ok"] is True
    assert len(runs) == 2  # 1 warmup and 1 run, the probe untimed
    assert all(r[:3] == ["taskset", "-c", "0-3"] for r in runs)
    assert all(r[3] == "/container/python" for r in runs)
    # The requested set and the set a process started like the timed runs may run
    # on, read once by a probe started the same way (the backend's resource
    # module, sent to the project interpreter).
    (argv, source), = probes
    # Between the interpreter and its isolation options: the context's
    # interpreter flags, or none from an adapter that starts the probe
    # without them.
    flags = list(language_project_context("python", "/testbed").interpreter_flags)
    assert argv in [["taskset", "-c", "0-3", "/container/python", *started,
                     "-I", "-S", "-", "allowed"] for started in ([], flags)]
    assert "def allowed_cpus" in source
    assert stats["requested_cpus"] == "0-3"
    assert stats["allowed_cpus"] == [0, 1, 2, 3]


def test_container_mode_red_warmup_fails_cleanly(tmp_path, monkeypatch, in_run):
    source_dir, files = _suite(tmp_path)

    class _RedExecutor:
        def run(self, argv, *, cwd=None, env=None, timeout=None, input=None):
            probe = _answer_the_probe(argv)
            if probe is not None:
                return probe
            return subprocess.CompletedProcess(
                args=list(argv), returncode=1, stdout="==== 1 failed in 0.01s ====", stderr="")

    monkeypatch.setattr(config, "project_exec_mode", lambda: "container")
    monkeypatch.setattr("langadapter.python.adapter.executor_for", lambda *args, **kwargs: _RedExecutor())
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", "/container/python")
    # The container and work directory a container run names.
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER", "fixture-container")
    monkeypatch.setenv("TESTUARY_PROJECT_WORK_DIR", str(tmp_path))
    in_run()

    backend = get_backend("python")
    stats = backend.benchmark_tests(
        source_dir="/testbed", test_file_paths=files,
        warmup_runs=1, benchmark_runs=2, timeout=60)

    assert stats["ok"] is False
    assert stats["returncode"] == 1
    assert stats["failure_kind"] == "test_failure"


def test_container_mode_wrapper_failure_after_green_pytest_is_infrastructure(
    tmp_path, monkeypatch, in_run):
    source_dir, files = _suite(tmp_path)

    class _BrokenWrapperExecutor:
        def run(self, argv, *, cwd=None, env=None, timeout=None, input=None):
            probe = _answer_the_probe(argv)
            if probe is not None:
                return probe
            return subprocess.CompletedProcess(
                args=list(argv), returncode=1, stdout="14 passed",
                stderr="TypeError: coverage shutdown failed",
            )

    monkeypatch.setattr(config, "project_exec_mode", lambda: "container")
    monkeypatch.setattr("langadapter.python.adapter.executor_for", lambda *args, **kwargs: _BrokenWrapperExecutor())
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", "/container/python")
    # The container and work directory a container run names.
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER", "fixture-container")
    monkeypatch.setenv("TESTUARY_PROJECT_WORK_DIR", str(tmp_path))
    in_run()

    stats = get_backend("python").benchmark_tests(
        source_dir="/testbed", test_file_paths=files,
        warmup_runs=1, benchmark_runs=2, timeout=60,
    )

    assert stats["ok"] is False
    assert stats["returncode"] == 1
    assert stats["failure_kind"] == "infrastructure"


def test_benchmark_uses_the_public_operation_and_preserves_progress(
        tmp_path, monkeypatch, timed_run):
    source_dir, files = _suite(tmp_path)
    (Path(files[0]).parent / "conftest.py").write_text("# setup\n", encoding="utf-8")
    requests = []
    messages = []
    original = ProjectPythonBackend.benchmark

    def record(self, request):
        requests.append(request)
        return original(self, request)

    monkeypatch.setattr(ProjectPythonBackend, "benchmark", record)
    monkeypatch.setattr(placement, "run_timed", timed_run(_ok_proc()))
    stats = get_backend("python").benchmark_tests(
        source_dir=source_dir, test_file_paths=files, warmup_runs=1,
        benchmark_runs=2, timeout=60, suite_label="candidate", progress_fn=messages.append)
    assert stats["ok"] and len(stats["times_ms"]) == 2
    (request,) = requests
    assert request.runs == 2 and request.warmup_runs == 1 and request.timeout_s == 60
    assert request.cpus is None and request.serialize is False
    # A suite outside the project is copied beside the project's copy, with
    # its setup.
    setup, = request.suite.setup_files
    assert Path(setup).name == "conftest.py" and Path(setup).parent.name == "gen"
    assert Path(setup).parent.parent.name.startswith("langadapter_project_")
    assert messages[0].startswith("[benchmark:candidate] warmup 1/1")
    assert messages[-1].startswith("[benchmark:candidate] rep 2/2")
