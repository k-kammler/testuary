"""The project runtime a run resolves (``project.project_runtime``) on real
directories: the result for each selection and the text of each refusal, on this
machine and in a container, with and without run settings. The adapter's
``discover_runtime`` chooses it, and a refusal gives the adapter's reason
with the setting to change."""
from __future__ import annotations

import os
from pathlib import Path
import sys

import pytest

from testuary import config as config_module
from testuary import project as project_module
from testuary.settings import RunSettings
from python_runtime import project_python

#: These tests exercise interpreter discovery itself (see tests/conftest.py).
REAL_INTERPRETER_DISCOVERY = True

_JAVA = "java.exe" if os.name == "nt" else "java"
_PYTHON_SETTING = " (set in the project's settings, or with TESTUARY_PROJECT_PYTHON_EXECUTABLE)"
_JAVA_SETTING = " (set in the project's settings, or with TESTUARY_PROJECT_JAVA_HOME)"


@pytest.fixture(autouse=True)
def _fresh_caches(monkeypatch):
    for name in ("TESTUARY_PROJECT_PYTHON_EXECUTABLE", "TESTUARY_MAIN_PROJECT_EXEC", "JAVA_HOME",
                 "TESTUARY_PROJECT_JAVA_HOME", "TESTUARY_PROJECT_PYTHON_EXECUTABLE",
                 "TESTUARY_PROJECT_PYTHON_VERSION", "TESTUARY_PROJECT_REQUIREMENTS"):
        monkeypatch.delenv(name, raising=False)


def _python_in(prefix: Path) -> Path:
    interpreter = prefix / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_text("", encoding="utf-8")
    return interpreter


def _base_environment(**extra):
    base = {key: value for key, value in os.environ.items()
            if key not in ("VIRTUAL_ENV", "CONDA_PREFIX", "JAVA_HOME")}
    return {**base, **extra}


def _host(tmp_path, project=None, **environment):
    return RunSettings(base_dir=tmp_path,
                       sections={"project": {"source_dir": str(tmp_path), **(project or {})}},
                       environment=_base_environment(**environment))


def _guest(tmp_path, project=None, **environment):
    return RunSettings(base_dir=tmp_path,
                       sections={"main": {"project_exec": "container"},
                                 "project": {"source_dir": "/guest/project", **(project or {})}},
                       environment=_base_environment(**environment))


def _jdk(jdk_dir: Path) -> Path:
    java = jdk_dir / "bin" / _JAVA
    java.parent.mkdir(parents=True, exist_ok=True)
    java.write_text("", encoding="utf-8")
    return java


def _java() -> str:
    """The java executable a run chooses for the project."""
    from testuary.languages import language_setting_values

    project = config_module.get_project_config()
    values = language_setting_values("java", project, config_module.active_settings())
    return project_module.project_runtime(
        "java", values, root=str(project.get("source_dir") or "")).interpreter.path


def _refusal(call):
    with pytest.raises(RuntimeError) as refused:
        call()
    return str(refused.value)


# Python on this machine, under run settings

def test_the_projects_only_venv(tmp_path):
    interpreter = _python_in(tmp_path / ".venv")
    with config_module.use_run_settings(_host(tmp_path)):
        assert project_python() == str(interpreter)


def test_an_activated_environment_alone(tmp_path):
    interpreter = _python_in(tmp_path / "activated")
    with config_module.use_run_settings(_host(tmp_path, VIRTUAL_ENV=str(tmp_path / "activated"))):
        assert project_python() == str(interpreter)


def test_no_environment_is_refused(tmp_path):
    with config_module.use_run_settings(_host(tmp_path)):
        text = _refusal(project_python)
    looked = str(tmp_path / ".venv" / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
    assert text == (
        "[project] python_executable: no interpreter is selected, and no project .venv or "
        f"activated environment was found (looked at: {looked}): select one, or set "
        f"python_version{_PYTHON_SETTING}")


def test_several_environments_are_refused(tmp_path):
    first = _python_in(tmp_path / ".venv")
    second = _python_in(tmp_path / "activated")
    with config_module.use_run_settings(_host(tmp_path, VIRTUAL_ENV=str(tmp_path / "activated"))):
        text = _refusal(project_python)
    assert text == ("[project] python_executable: several interpreters could run the project "
                    f"({first}; {second}): select one{_PYTHON_SETTING}")


def test_the_callers_own_environment_outside_the_project_is_refused(tmp_path):
    with config_module.use_run_settings(_host(tmp_path, VIRTUAL_ENV=sys.prefix)):
        text = _refusal(project_python)
    assert text == (
        "[project] python_executable: the only environment found is the one the caller runs in "
        f"({sys.prefix}), activated in the shell, and a project runs with it only when it is "
        "selected: select the project's interpreter, or this one on purpose, or create the "
        f"project's .venv{_PYTHON_SETTING}")


def test_an_absolute_executable_is_kept(tmp_path):
    interpreter = _python_in(tmp_path / "elsewhere")
    with config_module.use_run_settings(_host(tmp_path, {"python_executable": str(interpreter)})):
        assert project_python() == str(interpreter)


def test_a_relative_executable_joins_the_settings_base(tmp_path):
    interpreter = _python_in(tmp_path / "env")
    relative = str(interpreter.relative_to(tmp_path))
    with config_module.use_run_settings(_host(tmp_path, {"python_executable": relative})):
        assert project_python() == str(interpreter)


def test_a_home_relative_executable_expands_under_settings(tmp_path):
    home = tmp_path / "home"
    interpreter = _python_in(home / "env")
    settings = _host(tmp_path, {"python_executable": "~/" + str(interpreter.relative_to(home))},
                     USERPROFILE=str(home), HOME=str(home))
    with config_module.use_run_settings(settings):
        assert project_python() == str(interpreter)


def test_a_variable_in_the_executable_expands_under_settings(tmp_path):
    interpreter = _python_in(tmp_path / "env")
    settings = _host(tmp_path, {"python_executable": "${ENV_HOME}/" + str(
        interpreter.relative_to(tmp_path / "env"))}, ENV_HOME=str(tmp_path / "env"))
    with config_module.use_run_settings(settings):
        assert project_python() == str(interpreter)


def test_a_missing_executable_is_refused(tmp_path):
    missing = tmp_path / "nope" / "python"
    with config_module.use_run_settings(_host(tmp_path, {"python_executable": str(missing)})):
        text = _refusal(project_python)
    assert text == (f"[project] python_executable: points at {missing}, which does not exist"
                    f"{_PYTHON_SETTING}")


def test_a_version_is_provisioned_once(tmp_path, monkeypatch):
    calls = []
    provisioned = str(tmp_path / "venvs" / "python")

    def provision(language, request, **_):
        calls.append((language, request.version, request.requirements))
        return provisioned

    monkeypatch.setattr(project_module, "provision_runtime", provision)
    requirements = tmp_path / "requirements.txt"
    settings = _host(tmp_path, {"python_version": "3.10", "requirements": str(requirements)})
    with config_module.use_run_settings(settings):
        assert project_python() == provisioned
        assert project_python() == provisioned
    assert calls == [("python", "3.10", str(requirements))]


def test_a_version_without_requirements(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(project_module, "provision_runtime",
                        lambda language, request, **_: calls.append(
                            (language, request.version, request.requirements)) or "provisioned")
    with config_module.use_run_settings(_host(tmp_path, {"python_version": "3.12"})):
        assert project_python() == "provisioned"
    assert calls == [("python", "3.12", None)]


def test_an_executable_wins_over_a_version(tmp_path, monkeypatch):
    interpreter = _python_in(tmp_path / "env")
    monkeypatch.setattr(project_module, "provision_runtime", lambda *a, **k: pytest.fail("provisioned"))
    settings = _host(tmp_path, {"python_executable": str(interpreter), "python_version": "3.10"})
    with config_module.use_run_settings(settings):
        assert project_python() == str(interpreter)


def test_a_discovered_environment_is_cached_per_selection(tmp_path):
    interpreter = _python_in(tmp_path / ".venv")
    with config_module.use_run_settings(_host(tmp_path)):
        assert project_python() == str(interpreter)
        interpreter.unlink()
        assert project_python() == str(interpreter)


# Python in a container

@pytest.mark.parametrize("selected", ["/guest/venv/bin/python", "venv/bin/python"])
def test_a_container_executable_is_taken_as_given(tmp_path, selected):
    with config_module.use_run_settings(_guest(tmp_path, {"python_executable": selected})):
        assert project_python() == selected


def test_no_container_interpreter_is_refused(tmp_path):
    with config_module.use_run_settings(_guest(tmp_path)):
        text = _refusal(project_python)
    assert text == ("[project] python_executable: no interpreter is selected for the project in "
                    f"the container: select one{_PYTHON_SETTING}")


def test_a_container_version_is_provisioned(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(project_module, "provision_runtime",
                        lambda language, request, **_: calls.append(
                            (language, request.version, request.requirements)) or "provisioned")
    with config_module.use_run_settings(_guest(tmp_path, {"python_version": "3.11"})):
        assert project_python() == "provisioned"
    assert calls == [("python", "3.11", None)]


# Python: the interpreter variable

def test_the_runner_variable_wins_over_the_projects_file(monkeypatch, tmp_path, in_run):
    """TESTUARY_PROJECT_PYTHON_EXECUTABLE is the environment's layer of the
    setting, over the project's file and its python_version."""
    interpreter = _python_in(tmp_path / "runner")
    (tmp_path / ".testuary").mkdir()
    (tmp_path / ".testuary" / "project.ini").write_text(
        "[project]\npython_executable = ignored\npython_version = 3.10\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", str(interpreter))
    in_run(project_settings=True)
    assert project_python() == str(interpreter)


# Java

def test_java_home_names_the_jdk_dir(tmp_path):
    java = _jdk(tmp_path / "jdk")
    with config_module.use_run_settings(_host(tmp_path, {"java_home": str(tmp_path / "jdk")})):
        assert _java() == str(java)


def test_java_home_names_the_java_executable(tmp_path):
    java = _jdk(tmp_path / "jdk")
    with config_module.use_run_settings(_host(tmp_path, {"java_home": str(java)})):
        assert _java() == str(java)


def test_a_relative_java_home_joins_the_settings_base(tmp_path):
    java = _jdk(tmp_path / "jdk")
    with config_module.use_run_settings(_host(tmp_path, {"java_home": "jdk"})):
        assert _java() == str(java)


def test_java_home_wins_over_java_home_variable(tmp_path):
    java = _jdk(tmp_path / "jdk")
    _jdk(tmp_path / "other")
    settings = _host(tmp_path, {"java_home": str(tmp_path / "jdk")},
                     JAVA_HOME=str(tmp_path / "other"))
    with config_module.use_run_settings(settings):
        assert _java() == str(java)


def test_the_java_home_variable(tmp_path):
    java = _jdk(tmp_path / "jdk")
    with config_module.use_run_settings(_host(tmp_path, JAVA_HOME=str(tmp_path / "jdk"))):
        assert _java() == str(java)


@pytest.mark.parametrize("spelling", ["jdk", "~/jdk", "${JDKS}/jdk"])
def test_a_relative_java_home_variable_joins_the_settings_base_unexpanded(tmp_path, spelling):
    java = _jdk(tmp_path / spelling)
    settings = _host(tmp_path, JAVA_HOME=spelling, JDKS=str(tmp_path / "jdks"),
                     USERPROFILE=str(tmp_path / "home"), HOME=str(tmp_path / "home"))
    with config_module.use_run_settings(settings):
        assert _java() == str(java)


def test_no_jdk_is_refused(tmp_path):
    with config_module.use_run_settings(_host(tmp_path)):
        text = _refusal(_java)
    assert text == ("[project] java_home: no JDK is selected for the project: select one, or "
                    f"set JAVA_HOME{_JAVA_SETTING}")


def test_a_missing_jdk_is_refused(tmp_path):
    with config_module.use_run_settings(_host(tmp_path, {"java_home": str(tmp_path / "jdk")})):
        text = _refusal(_java)
    assert text == (f"[project] java_home: {tmp_path / 'jdk'} names no JDK: "
                    f"{tmp_path / 'jdk' / 'bin' / _JAVA} does not exist{_JAVA_SETTING}")


@pytest.mark.parametrize("selected,expected", [
    ("/guest/jdk", "/guest/jdk/bin/java"),
    ("/guest/jdk/bin/java", "/guest/jdk/bin/java"),
    ("//guest/jdk", "/guest/jdk/bin/java"),
    ("/guest/./jdk/", "/guest/jdk/bin/java"),
])
def test_a_container_jdk_is_a_posix_path(tmp_path, selected, expected):
    with config_module.use_run_settings(_guest(tmp_path, {"java_home": selected})):
        assert _java() == expected


@pytest.mark.parametrize("selected", ["guest/jdk", "C:\\jdk", "/guest/../jdk", "/guest\\jdk"])
def test_a_container_jdk_that_is_no_posix_path_is_refused(tmp_path, selected):
    with config_module.use_run_settings(_guest(tmp_path, {"java_home": selected})):
        text = _refusal(_java)
    assert text == ("[project] java_home: a container JDK is an absolute POSIX path without "
                    f"backslashes or parent traversal{_JAVA_SETTING}")


@pytest.mark.parametrize("variable", [{}, {"JAVA_HOME": "/captured/jdk"}])
def test_a_container_without_java_home_is_refused(tmp_path, variable):
    """The host's JAVA_HOME names no JDK in the container."""
    with config_module.use_run_settings(_guest(tmp_path, **variable)):
        text = _refusal(_java)
    assert text == ("[project] java_home: no JDK is selected for the project in the container: "
                    f"select one{_JAVA_SETTING}")

