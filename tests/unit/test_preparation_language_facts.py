"""Measurement copies get the run's bytecode store and take an external suite
with them only for a language whose adapter states ``reuses_bytecode`` and
``copies_external_suite`` (Python, a plugin stating them), for a backend and
for a description of one."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

import langadapter as lb
from testuary.languages import preparation


class _Plugin(lb.LangAdapter):
    language = "facts-fixture"


class _StatingPlugin(lb.LangAdapter):
    language = "stating-fixture"
    reuses_bytecode = True
    copies_external_suite = True


@pytest.fixture
def plugin():
    lb.register(_Plugin.language, (".facts-fixture",), _Plugin)
    yield _Plugin.language
    lb.unregister(_Plugin.language)


@pytest.fixture
def stating():
    lb.register(_StatingPlugin.language, (".stating-fixture",), _StatingPlugin)
    yield _StatingPlugin.language
    lb.unregister(_StatingPlugin.language)


@pytest.fixture
def stores(monkeypatch):
    monkeypatch.setattr(preparation, "_STORES", {})
    monkeypatch.setattr(preparation.atexit, "register", lambda function: None)
    yield
    preparation._close_stores()


def _layout(tmp_path):
    """A project and a suite directory beside it, below the directory they share."""
    project, suite = tmp_path / "project", tmp_path / "suite"
    project.mkdir()
    suite.mkdir()
    return project, suite


def _backends(language, project):
    context = lb.ProjectContext(str(project), lb.Interpreter(path=sys.executable))
    described = SimpleNamespace(language=language, context=context)
    return (described, lb.create(language, context))


@pytest.mark.parametrize("language,kept", [("python", True), ("java", False)])
def test_the_bytecode_store(tmp_path, stores, language, kept):
    project, _ = _layout(tmp_path)
    for backend in _backends(language, project):
        option = preparation.bytecode_store(backend)
        assert ("bytecode" in option) is kept
        assert option == {} or set(option) == {"bytecode"}


def test_no_bytecode_store_for_a_plugin_without_the_fact(tmp_path, stores, plugin):
    project, _ = _layout(tmp_path)
    for backend in _backends(plugin, project):
        assert preparation.bytecode_store(backend) == {}


def test_no_bytecode_store_for_an_unregistered_description(tmp_path, stores):
    context = lb.ProjectContext(str(tmp_path), lb.Interpreter(path=sys.executable))
    assert preparation.bytecode_store(SimpleNamespace(language="unregistered",
                                                      context=context)) == {}


def test_a_store_whose_creation_is_refused_is_a_setup_error(
        tmp_path, stores, monkeypatch):
    """A store whose creation LangAdapter refuses may have left its directory
    behind. The prepared measurement's failure is then this layer's setup
    error, and its cleanup is pending."""
    import time

    from testuary.languages import mutation

    def unfinished(cls, **options):
        raise lb.OwnershipRefused("the store's creation receipt was not finished")

    monkeypatch.setattr(preparation.BytecodeStore, "create", classmethod(unfinished))
    project, _ = _layout(tmp_path)
    backend = SimpleNamespace(language="python", context=lb.ProjectContext(
        str(project), lb.Interpreter(path=sys.executable)))
    with pytest.raises(lb.OwnershipRefused) as failed:
        preparation._prepared_owner(backend, None, time.perf_counter())
    assert failed.value.preparation_setup_error is failed.value
    assert mutation.mutation_cleanup_pending(failed.value)


def test_a_store_that_cannot_be_made_is_named_in_the_run_log(tmp_path, stores, monkeypatch):
    """The measurement runs without the store, as before, and the run log says
    so once, with the error's text, without the path that failed. A later
    measurement tries the store again without another line."""
    from testuary import _progress

    monkeypatch.setattr(preparation, "_UNSTORED", set())
    lines = []
    attempts = []
    monkeypatch.setattr(_progress, "write_plain_log", lambda message, level=0: lines.append(message))
    failed = str(tmp_path / "private" / "langadapter_bytecode_store")

    def refused(cls, **options):
        attempts.append(options)
        raise PermissionError(13, "Permission denied", failed)

    monkeypatch.setattr(preparation.BytecodeStore, "create", classmethod(refused))
    project, _ = _layout(tmp_path)
    backend = SimpleNamespace(language="python", context=lb.ProjectContext(
        str(project), lb.Interpreter(path=sys.executable)))
    assert preparation.bytecode_store(backend) == {}
    assert preparation.bytecode_store(backend) == {}
    assert len(attempts) == 2
    (line,) = lines
    assert line.startswith("[bytecode] a python measurement runs without the compiled-module store")
    assert "PermissionError (Permission denied)" in line
    assert "private" not in line and str(tmp_path) not in line


@pytest.mark.parametrize("language,kept", [("python", True), ("java", False)])
def test_an_external_suite_copied_beside_the_project(tmp_path, language, kept):
    project, suite = _layout(tmp_path)
    for backend in _backends(language, project):
        expected = {"companions": (str(suite.resolve()),)} if kept else {}
        assert preparation._companions(backend, SimpleNamespace(root=str(suite))) == expected
        assert preparation.session_companions(backend, [suite, suite / "nested"]) == expected
        assert preparation._companions(backend, None) == {}


def test_a_plugin_without_the_fact_keeps_its_suite_external(tmp_path, plugin):
    project, suite = _layout(tmp_path)
    for backend in _backends(plugin, project):
        assert preparation._companions(backend, SimpleNamespace(root=str(suite))) == {}
        assert preparation.session_companions(backend, [suite]) == {}


def test_an_unregistered_description_keeps_its_suite_external(tmp_path):
    project, suite = _layout(tmp_path)
    context = lb.ProjectContext(str(project), lb.Interpreter(path=sys.executable))
    described = SimpleNamespace(language="unregistered", context=context)
    assert preparation._companions(described, SimpleNamespace(root=str(suite))) == {}
    assert preparation.session_companions(described, [Path(suite)]) == {}


def test_a_plugin_stating_the_facts_gets_both(tmp_path, stores, stating):
    project, suite = _layout(tmp_path)
    for backend in _backends(stating, project):
        assert set(preparation.bytecode_store(backend)) == {"bytecode"}
        expected = {"companions": (str(suite.resolve()),)}
        assert preparation._companions(backend, SimpleNamespace(root=str(suite))) == expected
        assert preparation.session_companions(backend, [suite]) == expected


# The run's bytecode store: one per work directory for Python measurements, none
# for other languages, removed at the run's end.
def test_python_measurements_share_one_store_per_work_dir(tmp_path, monkeypatch):
    monkeypatch.setattr(preparation, "_STORES", {})
    registered = []
    monkeypatch.setattr(preparation.atexit, "register", registered.append)
    local = lb.ProjectContext(str(tmp_path), lb.Interpreter(path="python"))
    zone = tmp_path / "zone"
    zone.mkdir()
    guest = lb.ProjectContext("/repo", lb.Interpreter(path="/venv/bin/python"),
                              execution=lb.ContainerExecution("c1", str(zone)))
    first = preparation.bytecode_store(SimpleNamespace(language="python", context=local))["bytecode"]
    assert preparation.bytecode_store(
        SimpleNamespace(language="python", context=local))["bytecode"] is first
    # A container's commands reach only a store in its work directory.
    inside = preparation.bytecode_store(SimpleNamespace(language="python", context=guest))["bytecode"]
    assert inside is not first and Path(inside.root).parent == zone
    assert preparation.bytecode_store(SimpleNamespace(language="java", context=local)) == {}
    assert registered == [preparation._close_stores]
    preparation._close_stores()
    assert not Path(first.root).exists() and not Path(inside.root).exists()
    assert preparation._STORES == {}
