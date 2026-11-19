"""The project execution interpreter: resolution order, on-demand venv
provisioning, and the runtime-version surface."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
from langadapter.context import ProvisioningRequest

from testuary import config
from testuary import project as project_module
from testuary import prompts
from testuary.languages.generic import GenericBackend
from python_runtime import project_python

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _project_cfg(monkeypatch, **kw):
    cfg = {"python_executable": "", "python_version": "",
           "requirements": "", "project_name": "demo"}
    cfg.update(kw)
    monkeypatch.setattr(config, "get_project_config", lambda: cfg)


# Resolution order

def test_without_a_selection_the_discovered_environment(monkeypatch):
    """This session's interpreter stands in for the discovered environment."""
    _project_cfg(monkeypatch)
    assert project_python() == sys.executable


def test_env_var_missing_path_is_an_error(monkeypatch, tmp_path, in_run):
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", str(tmp_path / "nope"))
    in_run()
    with pytest.raises(RuntimeError, match="TESTUARY_PROJECT_PYTHON_EXECUTABLE"):
        project_python()


def test_explicit_executable_wins(monkeypatch, tmp_path):
    fake = tmp_path / "python"
    fake.write_text("", encoding="utf-8")
    _project_cfg(monkeypatch, python_executable=str(fake),
                 python_version="3.10")
    assert project_python() == str(fake.resolve())


def test_explicit_executable_missing_is_a_hard_error(monkeypatch, tmp_path):
    _project_cfg(monkeypatch,
                 python_executable=str(tmp_path / "nope" / "python"))
    with pytest.raises(RuntimeError, match="does not exist"):
        project_python()


def test_version_triggers_provisioning(monkeypatch, tmp_path):
    _project_cfg(monkeypatch, python_version="3.10",
                 requirements="")
    calls = {}

    def fake_provision(language, request, **kw):
        calls["args"] = (language, request.version, request.requirements)
        return str(tmp_path / "venv" / "python")

    monkeypatch.setattr(project_module, "provision_runtime", fake_provision)
    out = project_python()
    assert calls["args"] == ("python", "3.10", None)
    assert out.endswith("python")
    # Cached: a second call does not re-provision.
    calls.clear()
    assert project_python() == out
    assert calls == {}


def test_a_configured_version_reaches_the_resolver(monkeypatch, tmp_path, in_run):
    """Through the real project settings, not replaced ones, the
    configured ``python_version`` and ``requirements`` provision a venv."""
    monkeypatch.delenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", raising=False)
    requirements = tmp_path / "requirements.txt"
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_VERSION", "3.10")
    monkeypatch.setenv("TESTUARY_PROJECT_REQUIREMENTS", str(requirements))
    in_run()
    calls = {}

    def fake_provision(language, request, **kw):
        calls["args"] = (language, request.version, request.requirements)
        return str(tmp_path / "venv" / "python")

    monkeypatch.setattr(project_module, "provision_runtime", fake_provision)
    project_python()
    assert calls["args"] == ("python", "3.10", str(requirements))


# Provisioning: LangAdapter's Python adapter makes the environment
# (tests/python/test_provision.py there), and Testuary names and keeps it.

def _record_provisioning(monkeypatch, recorded, *, fails=None):
    from langadapter.python.adapter import PythonAdapter

    def provision(request, *, store, name, environment=None, log=None):
        recorded.append({"request": request, "store": store, "name": name,
                         "environment": environment})
        if fails is not None:
            raise fails
        log("environment ready: here")
        return str(store / name / "python")

    monkeypatch.setattr(PythonAdapter, "provision_runtime", staticmethod(provision))


def test_the_environment_is_langadapters_kept_under_testuarys_cache_by_project_name(
    monkeypatch, tmp_path, in_run, capsys,
):
    """Through the real project settings: the adapter provisions, keyed on
    the configured project's name, in Python's directory of Testuary's cache,
    with a relative requirements file under the configuration directory."""
    monkeypatch.setenv("TESTUARY_PROJECT_NAME", "venv-key-probe")
    in_run()
    monkeypatch.setattr(project_module, "_venv_home", lambda: tmp_path / ".venvs")
    recorded: list = []
    _record_provisioning(monkeypatch, recorded)
    capsys.readouterr()
    python = project_module.provision_runtime("python", ProvisioningRequest("3.10", "reqs.txt"))
    assert python == str(tmp_path / ".venvs" / "python" / "venv-key-probe" / "python")
    (call,) = recorded
    assert (call["store"], call["name"]) == (tmp_path / ".venvs" / "python", "venv-key-probe")
    assert call["request"].version == "3.10"
    assert call["request"].requirements == str(
        (config.configuration_base_dir() / "reqs.txt").resolve())
    assert capsys.readouterr().out == "[env] environment ready: here\n"


def test_a_refused_or_failed_provisioning_names_its_setting_or_says_why(monkeypatch, tmp_path):
    from langadapter.errors import InvalidRequest

    _project_cfg(monkeypatch, project_name="demo")
    monkeypatch.setattr(project_module, "_venv_home", lambda: tmp_path / ".venvs")
    _record_provisioning(monkeypatch, [], fails=InvalidRequest(
        "python_version", "provisioning an environment requires uv, which is not on PATH: "
        "install it with `pip install uv`"))
    with pytest.raises(RuntimeError, match=r"\[project\] python_version: .*pip install uv"):
        project_module.provision_runtime("python", ProvisioningRequest("3.10", None))
    _record_provisioning(monkeypatch, [], fails=RuntimeError(
        "the dependency install into the environment failed: package conflict"))
    with pytest.raises(RuntimeError, match=r"\[env\] the dependency install .*package conflict"):
        project_module.provision_runtime("python", ProvisioningRequest("3.10", None))


# The backend uses the resolved interpreter

def test_backend_subprocesses_use_the_resolved_interpreter(
    monkeypatch, tmp_path,
):
    import subprocess

    from langadapter.execution import LocalExecutor
    from testuary import languages

    backend = languages.get_backend("python")
    fake_py = str(tmp_path / "project-python")
    Path(fake_py).touch()
    _project_cfg(monkeypatch, python_executable=fake_py)
    seen: list = []

    def record_run(self, cmd, **kw):
        seen.append([str(c) for c in cmd])
        raise OSError("stop here")

    def refuse_launch(*args, **kwargs):
        pytest.fail("the interpreter recorder must not launch a subprocess")

    monkeypatch.setattr(LocalExecutor, "run", record_run)
    monkeypatch.setattr(subprocess, "Popen", refuse_launch)
    suite = tmp_path / "test_x.py"
    suite.write_text("def test_a():\n    assert True\n", encoding="utf-8")
    try:
        backend.failing_test_ids([str(suite)], str(tmp_path), timeout=10)
    except OSError:
        pass
    assert any(s and s[0] == fake_py for s in seen)


# Runtime version

def test_runtime_version_string_and_fragment(monkeypatch):
    v = f"Python {sys.version.split()[0]}"
    from testuary.languages import get_backend
    frags = get_backend("python").prompt_fragments()
    assert frags["lang_runtime_version"] == v
    out = prompts.resolve_agent_prompts("test_coverage", "python")
    # The prompt names the language level the adapter reports, the
    # interpreter's version without its patch, or the runtime line where
    # the adapter reports none.
    level = prompts._language_level(get_backend("python"), None) or v
    assert v.startswith(level)
    assert f"**Write the tests in:** {level}" in out["user_prompt_template"].splitlines()


# Pytest anchoring of a measurement directory

def _rootdirs_of_run(backend, paths, project):
    """``run_test_files``' result and the rootdirs its pytest runs used."""
    from langadapter.execution import LocalExecutor

    commands = []
    native = LocalExecutor.run

    def run(self, argv, **kwargs):
        if "pytest" in [str(item) for item in argv]:
            commands.append([str(item) for item in argv])
        return native(self, argv, **kwargs)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(LocalExecutor, "run", run)
        result = backend.run_test_files(paths, str(project), timeout=120)
    return result, [item.split("=", 1)[1] for command in commands for item in command
                    if item.startswith("--rootdir=")]


def test_directory_argument_anchors_at_the_suite_not_its_parent(tmp_path):
    """A caller may name the suite directory rather than a file inside it:
    the consolidation's green check runs pytest against the directory. The
    anchor is the suite, not the path's parent, and no run places a setup
    file beside the suite or its parent."""
    backend = GenericBackend("python", (".py",))
    project = tmp_path / "proj"
    project.mkdir()
    (project / "mod.py").write_text(
        "def f(x):\n    return x + 1\n", encoding="utf-8")
    staging = project / "testuary_phaseend_probe"
    suite = staging / "merged"
    suite.mkdir(parents=True)
    (suite / "test_candidate.py").write_text(
        "from mod import f\n\ndef test_a():\n    assert f(1) == 2\n",
        encoding="utf-8")

    result, rootdirs = _rootdirs_of_run(backend, [str(suite)], project)

    assert result.ok, result.output
    assert not (suite / "conftest.py").exists()
    assert not (staging / "conftest.py").exists()
    assert not (staging / "pytest.ini").exists()
    # The run happens in a prepared copy whose rootdir is the copied suite itself.
    assert rootdirs and all(Path(path).name == "merged" for path in rootdirs)
    assert all(Path(path).parent.name == "testuary_phaseend_probe" for path in rootdirs)


def test_file_argument_still_anchors_at_its_own_directory(tmp_path):
    backend = GenericBackend("python", (".py",))
    project = tmp_path / "proj"
    project.mkdir()
    (project / "mod.py").write_text(
        "def f(x):\n    return x + 1\n", encoding="utf-8")
    suite = project / "suite"
    suite.mkdir()
    path = suite / "test_candidate.py"
    path.write_text(
        "from mod import f\n\ndef test_a():\n    assert f(1) == 2\n",
        encoding="utf-8")

    result, rootdirs = _rootdirs_of_run(backend, [str(path)], project)

    assert result.ok, result.output
    assert not (suite / "conftest.py").exists()
    assert not (project / "conftest.py").exists()
    assert rootdirs and all(Path(path).name == "suite" for path in rootdirs)
