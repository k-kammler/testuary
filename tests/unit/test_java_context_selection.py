"""Captured Java selections cross the public factory without native launches."""
import os
import subprocess

import pytest

from langadapter.context import ContainerExecution, LocalExecution
from langadapter.execution import ContainerExecutor, executor_for
from testuary import config
from testuary import project
from testuary.languages import language_project_context
from testuary.languages.generic import GenericBackend
from testuary.settings import RunSettings


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Java context selection must not execute target or control tools")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)


@pytest.mark.parametrize("selected_root", [None, "/captured/project/module"])
@pytest.mark.parametrize("control", [None, "/captured/control/python"])
def test_guest_java_context_keeps_captured_paths_tools_and_explicit_module(
        tmp_path, monkeypatch, selected_root, control):
    environment = {"TESTUARY_PROJECT_CONTAINER": "captured-java",
                   "TESTUARY_PROJECT_CONTAINER_RUNNER": "podman",
                   "TESTUARY_PROJECT_WORK_DIR": str(tmp_path),
                   "TESTUARY_PROJECT_SOURCE_DIR": "/captured/project",
                   "JAVA_HOME": "/other/jdk", "PATH": "host-only-path"}
    if control is not None:
        environment["LANGADAPTER_HELPER_PYTHON"] = control
    settings = RunSettings(
        sections={"main": {"project_exec": "container"}, "project": {
            "java_home": "/captured/jdk", "build_java_home": "/captured/build-jdk",
            "maven": "/captured/tools/mvn", "gradle": "/captured/tools/gradle"}},
        environment=environment, base_dir=tmp_path)
    for name, value in {"TESTUARY_PROJECT_CONTAINER": "later-java",
                        "TESTUARY_PROJECT_SOURCE_DIR": "/later/project",
                        "LANGADAPTER_HELPER_PYTHON": "/later/control",
                        "LANGADAPTER_BUILD_JAVA_HOME": "/later/build",
                        "JAVA_HOME": "/later/runtime", "PATH": "later-path"}.items():
        monkeypatch.setenv(name, value)

    resolve = project.resolve_host_path

    def host_path_forbidden(value, *args, **kwargs):
        if value == str(tmp_path):
            # The captured work directory is the one host path of a guest selection.
            return resolve(value, *args, **kwargs)
        pytest.fail("A guest JDK/project/tool selector must not be resolved on the host")

    with config.use_run_settings(settings):
        monkeypatch.setattr(project, "resolve_host_path", host_path_forbidden)
        backend = GenericBackend("java", (".java",)).project_backend(selected_root)
    context = backend.context
    context.validate()
    assert context.root == (selected_root or "/captured/project")
    assert context.interpreter.path == "/captured/jdk/bin/java"
    assert isinstance(context.execution, ContainerExecution)
    assert context.execution.repository_dir == context.root
    assert context.execution.work_dir == str(tmp_path)
    assert context.execution.container == "captured-java"
    assert context.execution.runner == "podman"
    assert context.execution.helper_python == control
    executor = executor_for(context)
    assert isinstance(executor, ContainerExecutor)
    assert executor.container == "captured-java"
    assert executor.runner == "podman"
    assert executor.helper_python == control
    assert dict(context.environment) == {
        "LANGADAPTER_BUILD_JAVA_HOME": "/captured/build-jdk",
        "LANGADAPTER_MAVEN": "/captured/tools/mvn",
        "LANGADAPTER_GRADLE": "/captured/tools/gradle"}
    # The host runner receives exactly the captured values, not later ambient ones.
    assert dict(context.execution.control_environment) == environment
    assert not context.execution.control_environment.inherit


@pytest.mark.parametrize("path", [None, "captured-path"])
def test_local_java_path_is_captured_even_when_explicitly_absent(tmp_path, monkeypatch, path):
    jdk = tmp_path / "jdk"
    (jdk / "bin").mkdir(parents=True)
    java = jdk / "bin" / ("java.exe" if os.name == "nt" else "java")
    java.touch()
    settings = RunSettings(
        sections={"project": {"java_home": str(jdk)}},
        environment={} if path is None else {"PATH": path}, base_dir=tmp_path)
    monkeypatch.setenv("PATH", "later-path")
    monkeypatch.setenv("JAVA_HOME", "later-jdk")
    with config.use_run_settings(settings):
        context = language_project_context("java", str(tmp_path))
    assert isinstance(context.execution, LocalExecution)
    assert context.interpreter.path == str(java)
    # Captured, not an overlay: an absent PATH stays absent for every launch.
    assert not context.environment.inherit
    assert context.environment.get("PATH") == path
    assert "JAVA_HOME" not in context.environment


def test_java_guest_missing_selection_cannot_be_filled_from_later_environment(tmp_path, monkeypatch):
    settings = RunSettings(
        sections={"main": {"project_exec": "container"},
                  "project": {"java_home": "/captured/jdk"}},
        environment={"TESTUARY_PROJECT_WORK_DIR": str(tmp_path)}, base_dir=tmp_path)
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER", "later-container")
    with config.use_run_settings(settings), pytest.raises(ValueError, match="requires a container"):
        language_project_context("java", "/captured/project")
