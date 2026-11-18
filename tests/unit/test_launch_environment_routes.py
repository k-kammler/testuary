"""Testuary's own project commands use the captured launch environment.

Under run settings, executor routes that bypass a backend context (version
probe, container text input/output, work-dir cleanup) launch commands with
the captured environment, without model-credential variables and without
values changed after capture. An unselected project interpreter is discovered
or refused, never Testuary's own by default. Credential values are synthetic.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import sys

import pytest

from testuary import project as execution
from testuary import config
from testuary import project as project_module
from testuary.settings import RunSettings, resolve_settings
from python_runtime import project_python

#: These tests exercise interpreter discovery itself (see tests/conftest.py).
REAL_INTERPRETER_DISCOVERY = True

_SHOW = ("import json, os; print(json.dumps({k: os.environ.get(k) for k in "
         "('SUBJECT_TOKEN', 'TESTUARY_OPENAI_API_KEY', 'MODEL_KEY', 'AMBIENT_AFTER')}))")


def _settings(tmp_path, extra=None, **sections):
    environment = {**os.environ, "SUBJECT_TOKEN": "subject",
                   "TESTUARY_OPENAI_API_KEY": "synthetic-provider-key",
                   "MODEL_KEY": "synthetic-alias-key", **(extra or {})}
    environment.pop("AMBIENT_AFTER", None)
    return RunSettings(
        base_dir=tmp_path, environment=environment,
        sections={"api": {"model": "chosen"}, **sections},
        models={"chosen": {"provider": "openai", "api_key_env": "MODEL_KEY"}},
    )


def test_local_commands_get_the_captured_environment_without_credentials(tmp_path, monkeypatch):
    settings = _settings(tmp_path)
    with config.use_run_settings(settings):
        # Changed after capture: an ambient value must not reach the child.
        monkeypatch.setenv("AMBIENT_AFTER", "late")
        result = project_module.get_executor().run([sys.executable, "-c", _SHOW], timeout=60)
    assert result.returncode == 0, result.stderr
    assert json.loads(result.stdout) == {
        "SUBJECT_TOKEN": "subject", "TESTUARY_OPENAI_API_KEY": None,
        "MODEL_KEY": None, "AMBIENT_AFTER": None,
    }


def test_explicit_command_environment_is_kept(tmp_path):
    with config.use_run_settings(_settings(tmp_path)):
        result = project_module.get_executor().run(
            [sys.executable, "-c", _SHOW], env={**os.environ, "SUBJECT_TOKEN": "explicit"}, timeout=60)
    assert json.loads(result.stdout)["SUBJECT_TOKEN"] == "explicit"


def test_container_runner_uses_the_launch_environment_as_its_control_environment(tmp_path):
    settings = _settings(tmp_path, {"TESTUARY_PROJECT_CONTAINER": "subject-container"},
                         main={"project_exec": "container"})
    with config.use_run_settings(settings):
        executor = project_module.get_executor()
    assert isinstance(executor, execution.ContainerExecutor)
    control = dict(executor.control_environment)
    assert control["SUBJECT_TOKEN"] == "subject"
    assert "TESTUARY_OPENAI_API_KEY" not in control and "MODEL_KEY" not in control


def test_settings_without_a_captured_environment_refuse_native_commands(tmp_path):
    with config.use_run_settings(RunSettings(base_dir=tmp_path)):
        with pytest.raises(RuntimeError, match=r"environment=dict\(os\.environ\)"):
            project_module.get_executor()


def _python_in(prefix: Path) -> Path:
    return prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def _discovery_settings(tmp_path, **environment):
    base = {key: value for key, value in os.environ.items()
            if key not in ("VIRTUAL_ENV", "CONDA_PREFIX")}
    return RunSettings(base_dir=tmp_path, sections={"project": {"source_dir": str(tmp_path)}},
                       environment={**base, **environment})


def test_unselected_interpreter_without_candidates_refuses(tmp_path):
    with config.use_run_settings(_discovery_settings(tmp_path)):
        with pytest.raises(RuntimeError, match="python_executable"):
            project_python()


def test_unselected_interpreter_uses_the_single_project_venv(tmp_path):
    interpreter = _python_in(tmp_path / ".venv")
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")
    with config.use_run_settings(_discovery_settings(tmp_path)):
        assert Path(project_python()).resolve() == interpreter.resolve()


def test_unselected_interpreter_uses_an_activated_conda_environment(tmp_path):
    prefix = tmp_path / "conda-env"
    # Conda keeps python.exe at the environment root on Windows.
    interpreter = prefix / ("python.exe" if os.name == "nt" else "bin/python")
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("", encoding="utf-8")
    with config.use_run_settings(_discovery_settings(tmp_path, CONDA_PREFIX=str(prefix))):
        assert Path(project_python()).resolve() == interpreter.resolve()


def test_the_same_environment_activated_and_in_the_project_is_one_candidate(tmp_path):
    interpreter = _python_in(tmp_path / ".venv")
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")
    settings = _discovery_settings(tmp_path, VIRTUAL_ENV=str(tmp_path / ".venv"))
    with config.use_run_settings(settings):
        assert Path(project_python()).resolve() == interpreter.resolve()


def test_two_distinct_candidates_refuse_instead_of_guessing(tmp_path):
    for prefix in (tmp_path / ".venv", tmp_path / "activated"):
        _python_in(prefix).parent.mkdir(parents=True)
        _python_in(prefix).write_text("", encoding="utf-8")
    settings = _discovery_settings(tmp_path, VIRTUAL_ENV=str(tmp_path / "activated"))
    with config.use_run_settings(settings):
        with pytest.raises(RuntimeError, match="several interpreters could run the project"):
            project_python()


def test_an_interpreter_selected_through_the_environment_is_kept(tmp_path, monkeypatch):
    """A launcher selects the project's runtime through TESTUARY_PROJECT_PYTHON_EXECUTABLE."""
    interpreter = tmp_path / "runner-python.exe"
    interpreter.write_text("", encoding="utf-8")
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", str(interpreter))
    settings = resolve_settings(base_dir=tmp_path, config_files=[], models={})
    with config.use_run_settings(settings):
        assert project_python() == str(interpreter)


def _ambient_credentials(monkeypatch):
    monkeypatch.setenv("TESTUARY_OPENAI_API_KEY", "synthetic-provider-key")
    monkeypatch.setenv("MODEL_KEY", "synthetic-alias-key")


def test_testuarys_own_activated_environment_is_not_chosen_silently(tmp_path):
    """With Testuary's own environment activated and no project .venv, that
    environment is refused, not selected."""
    with config.use_run_settings(_discovery_settings(tmp_path, VIRTUAL_ENV=sys.prefix)):
        with pytest.raises(RuntimeError, match="the one the caller runs in") as refused:
            project_python()
    assert "python_executable" in str(refused.value)


def test_testuarys_own_activated_environment_leaves_the_projects_venv_to_run_it(tmp_path):
    """With Testuary's own environment activated and a project .venv, the
    .venv runs the project: Testuary's environment is no second candidate."""
    interpreter = _python_in(tmp_path / ".venv")
    interpreter.parent.mkdir(parents=True)
    interpreter.write_text("", encoding="utf-8")
    with config.use_run_settings(_discovery_settings(tmp_path, VIRTUAL_ENV=sys.prefix)):
        assert Path(project_python()).resolve() == interpreter.resolve()


@pytest.mark.skipif(Path(sys.prefix).name != ".venv", reason="Testuary does not run from a project .venv here")
def test_testuary_installed_in_the_projects_own_venv_is_used(tmp_path):
    project = Path(sys.prefix).parent
    settings = RunSettings(base_dir=tmp_path, sections={"project": {"source_dir": str(project)}},
                           environment={key: value for key, value in os.environ.items()
                                        if key not in ("VIRTUAL_ENV", "CONDA_PREFIX")})
    with config.use_run_settings(settings):
        selected = Path(project_python())
    assert project_module._is_testuary_environment(str(selected))


@pytest.mark.parametrize("env_dir,inside", [(".venv", True), ("venv", True), ("envs/conda", True)])
def test_an_environment_inside_the_project_is_the_projects(tmp_path, env_dir, inside):
    """An environment inside the project counts as the project's under any
    directory name. One outside the project does not."""
    interpreter = tmp_path / "project" / env_dir / "Scripts" / "python.exe"
    assert project_module._is_project_venv(str(interpreter), str(tmp_path / "project")) is inside
    assert project_module._is_project_venv(str(tmp_path / "elsewhere" / "Scripts" / "python.exe"),
                                  str(tmp_path / "project")) is False
