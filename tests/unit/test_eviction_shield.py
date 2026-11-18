"""The eviction's instrumented run loads a plugin that binds coverage.py's
modules to the real built-in functions. These tests pin how the plugin
reaches that run and no other. The language backend's tests pin what the
plugin may do, and ``tests/integration/test_collect_eviction_data_real.py``
runs it with the real tools."""
import json
import os
import subprocess
from pathlib import Path

import pytest
from langadapter.execution import LocalExecutor
# The container double knows LangAdapter's import probe only by its private script.
from langadapter.python._runs import _WHERE_IMPORTED

from testuary import config
from testuary import project as project_module
from testuary.languages import GenericBackend

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

#: The plugin module the eviction's measured run loads (``-p``).
_MODULE = "_langadapter_eviction_shield"
#: The shield directory's files: the plugin and LangAdapter's per-test times plugin.
_SHIELD_FILES = [f"{_MODULE}.py", f"{_MODULE}_times.py"]


class _Recorder:
    """Records each command, its environment and the plugin files in the
    work directory while it runs. The shielded measured run exits 1, which stops
    an eviction, and every other command runs."""

    def __init__(self, work_dir: Path):
        self.work_dir = work_dir
        self.calls = []

    def record(self, argv, env):
        argv = [str(a) for a in argv]
        plugins = sorted(self.work_dir.glob(f"**/shield/{_MODULE}.py"))
        self.calls.append({
            "argv": argv, "env": dict(env or {}), "plugins": plugins,
            "texts": [p.read_text(encoding="utf-8") for p in plugins],
            "listings": [sorted(q.name for q in p.parent.iterdir()) for p in plugins],
        })
        return argv


def _completed(argv, returncode, stdout=""):
    return subprocess.CompletedProcess(argv, returncode, stdout, "")


def _recorder(tmp_path, monkeypatch):
    work_dir = tmp_path / "wz"
    work_dir.mkdir()
    monkeypatch.setenv("TESTUARY_PROJECT_WORK_DIR", str(work_dir))
    recorder = _Recorder(work_dir)
    suite = work_dir / "suite"
    suite.mkdir()
    test_file = suite / "test_x.py"
    test_file.write_text("def test_x():\n    assert True\n", encoding="utf-8")
    target = tmp_path / "project" / "mod.py"
    target.parent.mkdir()
    target.write_text("x = 1\n", encoding="utf-8")
    native = LocalExecutor.run

    def run(self, argv, **kwargs):
        if _MODULE in [str(a) for a in argv]:
            return _completed(recorder.record(argv, kwargs.get("env")), 1)
        recorder.record(argv, kwargs.get("env"))
        return native(self, argv, **kwargs)

    monkeypatch.setattr(LocalExecutor, "run", run)
    return recorder, test_file, target


def _evict(tmp_path, monkeypatch, in_run):
    recorder, test_file, target = _recorder(tmp_path, monkeypatch)
    in_run()
    result = GenericBackend("python", (".py",)).collect_eviction_data(
        [str(test_file)], str(target.parent), measure_coverage_of=str(target),
        timeout=30, warmup=False,
    )
    assert result is None
    runs = [c for c in recorder.calls
            if "coverage" in c["argv"] and "pytest" in c["argv"]]
    assert len(runs) == 1
    return runs[0]


def _assert_loads_plugin(call):
    argv = call["argv"]
    at = argv.index(_MODULE)
    assert argv[at - 1] == "-p" and at > argv.index("pytest")
    assert len(call["plugins"]) == 1
    (text,) = call["texts"]
    assert "def pytest_configure(" in text
    assert call["listings"] == [_SHIELD_FILES]
    return str(call["plugins"][0].parent)


def test_eviction_loads_the_plugin_from_its_own_directory(tmp_path, monkeypatch, in_run):
    monkeypatch.delenv("TESTUARY_MAIN_PROJECT_EXEC", raising=False)
    call = _evict(tmp_path, monkeypatch, in_run)
    shield_dir = _assert_loads_plugin(call)
    assert call["env"]["PYTHONPATH"].startswith(shield_dir + os.pathsep)


def test_container_eviction_joins_the_import_path_for_linux(tmp_path, monkeypatch, in_run):
    """Inside the container the list is split with ':' whatever the host,
    and names the plugin's directory at its container work-dir path."""
    from guest_fixtures import container_work_dir, controlled_guest

    recorder, test_file, target = _recorder(tmp_path, monkeypatch)
    project = tmp_path / "project"

    def guest(args, options, host):
        recorder.record(args, options.get("env"))
        if _WHERE_IMPORTED in args:
            # The import probe of a run in a copy: the module comes from it.
            copy, module = args[-2:]
            return _completed(args, 0, json.dumps({
                "name": "mod", "top": module, "root": copy, "verdict": "copy",
                "candidates": []}))
        return _completed(args, 1 if _MODULE in args else 0)

    controlled_guest(monkeypatch, container="fixture-container", root="/fixture-project",
                     source=project, work_dir=recorder.work_dir, handle=guest)
    monkeypatch.setattr(project_module, "project_repo_dir", lambda: "/fixture-project")
    monkeypatch.setattr(config, "project_exec_mode", lambda: "container")
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", "/venv/bin/python")
    in_run()
    result = GenericBackend("python", (".py",)).collect_eviction_data(
        [str(test_file)], "/fixture-project", measure_coverage_of="/fixture-project/mod.py",
        timeout=30, warmup=False,
    )
    assert result is None
    (call,) = [c for c in recorder.calls if "coverage" in c["argv"] and "pytest" in c["argv"]]
    argv = call["argv"]
    at = argv.index(_MODULE)
    assert argv[at - 1] == "-p" and at > argv.index("pytest")
    assert len(call["plugins"]) == 1 and call["listings"] == [_SHIELD_FILES]
    shield = call["plugins"][0].parent.relative_to(recorder.work_dir).as_posix()
    assert call["env"]["PYTHONPATH"].startswith(f"{container_work_dir(recorder.work_dir)}/{shield}:")
    assert ";" not in call["env"]["PYTHONPATH"]


def test_other_suite_executions_do_not_load_the_plugin(tmp_path, monkeypatch, in_run):
    monkeypatch.delenv("TESTUARY_MAIN_PROJECT_EXEC", raising=False)
    recorder, test_file, target = _recorder(tmp_path, monkeypatch)
    in_run()
    GenericBackend("python", (".py",)).failing_test_ids([str(test_file)], str(target.parent), timeout=5)
    assert recorder.calls
    for call in recorder.calls:
        assert _MODULE not in call["argv"]
        assert _MODULE not in call["env"].get("PYTHONPATH", "")
        entries = call["env"].get("PYTHONPATH", "").split(os.pathsep)
        assert not any(Path(e).name == "shield" for e in entries)


