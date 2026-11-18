"""Explicit guest roots do not replace captured runtime/container selections."""
import subprocess

import pytest

from langadapter.context import ContainerExecution
from langadapter.errors import InvalidRequest
from testuary import config
from testuary import project
from testuary.languages.generic import GenericBackend
from testuary.settings import RunSettings


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Context selection must not execute target or control tools")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)


def _settings(tmp_path, *, repository=True):
    environment = {
        "TESTUARY_PROJECT_CONTAINER": "captured-python",
        "TESTUARY_PROJECT_CONTAINER_RUNNER": "podman",
        "TESTUARY_PROJECT_WORK_DIR": str(tmp_path),
        "LANGADAPTER_HELPER_PYTHON": "/captured/control/python",
    }
    if repository:
        environment["TESTUARY_PROJECT_SOURCE_DIR"] = "/captured/project"
    return RunSettings(
        sections={"main": {"project_exec": "container", "pytest_config": "none"},
                  "project": {"python_executable": "/captured/env/bin/python"}},
        environment=environment, base_dir=tmp_path)


@pytest.mark.parametrize("selected_root", [None, "/captured/project/module", "/explicit/project"])
def test_public_python_factory_honors_explicit_guest_root(
        tmp_path, monkeypatch, selected_root):
    settings = _settings(tmp_path)
    for key, value in {
        "TESTUARY_PROJECT_SOURCE_DIR": "/later/project",
        "TESTUARY_PROJECT_CONTAINER": "later-container",
        "TESTUARY_PROJECT_PYTHON_EXECUTABLE": "/later/python",
        "LANGADAPTER_HELPER_PYTHON": "/later/control/python",
    }.items():
        monkeypatch.setenv(key, value)

    resolve = project.resolve_host_path

    def host_resolution_forbidden(value, *args, **kwargs):
        if value == str(tmp_path):
            # The captured work directory is the one host path of a guest selection.
            return resolve(value, *args, **kwargs)
        pytest.fail("Guest project/runtime selection must not be resolved on the host")

    with config.use_run_settings(settings):
        monkeypatch.setattr(project, "resolve_host_path", host_resolution_forbidden)
        backend = GenericBackend("python", (".py",)).project_backend(selected_root)

    context = backend.context
    context.validate()
    assert context.root == (selected_root or "/captured/project")
    assert context.interpreter.path == "/captured/env/bin/python"
    assert context.test_framework.source == "isolated"
    assert isinstance(context.execution, ContainerExecution)
    assert context.execution.repository_dir == context.root
    assert context.execution.work_dir == str(tmp_path)
    assert context.execution.container == "captured-python"
    assert context.execution.runner == "podman"
    assert context.execution.helper_python == "/captured/control/python"
    assert backend.executor.container == "captured-python"
    assert backend.executor.helper_python == "/captured/control/python"
    assert dict(context.execution.control_environment) == dict(settings.environment)
    assert not context.execution.control_environment.inherit
    assert not dict(context.environment)


def test_explicit_python_guest_root_does_not_require_a_profile_root(tmp_path):
    with config.use_run_settings(_settings(tmp_path, repository=False)):
        context = GenericBackend("python", (".py",)).project_backend("/explicit/project").context
    assert context.root == context.execution.repository_dir == "/explicit/project"


@pytest.mark.parametrize("selected_root", ["relative/project", "C:/host/project"])
def test_invalid_explicit_python_guest_root_is_not_hidden_by_profile(tmp_path, selected_root):
    with config.use_run_settings(_settings(tmp_path)), pytest.raises(InvalidRequest):
        GenericBackend("python", (".py",)).project_backend(selected_root)


def test_public_python_instances_keep_explicit_guest_modules_separate(tmp_path):
    with config.use_run_settings(_settings(tmp_path)):
        adapter = GenericBackend("python", (".py",))
        first = adapter.project_backend("/captured/project/first")
        assert adapter.project_backend("/captured/project/first") is first
        second = adapter.project_backend("/captured/project/second")
        configured = adapter.project_backend()
    assert second is not first and configured is not first
    assert first.context.root == "/captured/project/first"
    assert second.context.root == "/captured/project/second"
    assert configured.context.root == "/captured/project"
