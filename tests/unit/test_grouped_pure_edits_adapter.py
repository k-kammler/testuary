"""Pruning and finalization consume public source results without a runtime."""
from types import SimpleNamespace

import pytest

from langadapter.sources import SourceResult
from testuary.languages.generic import GenericBackend
from testuary.consolidation import ValidationUnresolved

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture(params=[lambda: GenericBackend("python", (".py",)), lambda: GenericBackend("java", (".java",))], ids=["python", "java"])
def edits(request, monkeypatch):
    language = request.param()
    group = SimpleNamespace(result=None, calls=[], supports=lambda name: True)

    def prune(source, tests):
        group.calls.append(("prune", source, tuple(tests)))
        return group.result

    def finalize(source):
        group.calls.append(("finalize", source))
        return group.result

    group.prune_tests = prune
    group.finalize_source = finalize
    backend = SimpleNamespace(sources=group)
    # Pruning and finalization read the source as the project's runtime does.
    monkeypatch.setattr(language, "project_backend", lambda root=None: backend)
    return language, group


@pytest.mark.parametrize("operation", ["prune_tests", "finalize_test_source"])
@pytest.mark.parametrize("status", ["applied", "unchanged", "parse_error"])
def test_observed_native_source_is_preserved(edits, operation, status):
    language, group = edits
    group.result = SourceResult(status, "native result", "native parser")
    args = ("original", ["Example.test_one"]) if operation == "prune_tests" else ("original",)
    assert getattr(language, operation)(*args) == "native result"
    expected = ("prune", "original", ("Example.test_one",)) if operation == "prune_tests" else (
        "finalize", "original")
    assert group.calls == [expected]


@pytest.mark.parametrize("operation", ["prune_tests", "finalize_test_source"])
@pytest.mark.parametrize("status", ["unsupported", "missing_setup", "timeout", "infrastructure_failure"])
def test_unavailable_edit_is_not_successful_unchanged_source(edits, operation, status):
    language, group = edits
    group.result = SourceResult(status, "original", "", detail="edit unavailable")
    args = ("original", []) if operation == "prune_tests" else ("original",)
    with pytest.raises(ValidationUnresolved, match="edit unavailable"):
        getattr(language, operation)(*args)


def test_actual_java_edits_do_not_resolve_a_project_or_jdk(monkeypatch):
    from langadapter.java.adapter import JavaAdapter as NativeJavaBackend

    def forbidden(*args, **kwargs):
        raise AssertionError("pure edit attempted execution setup")

    monkeypatch.setattr(NativeJavaBackend, "_jdk", forbidden)
    monkeypatch.setattr(NativeJavaBackend, "project", forbidden)
    java = GenericBackend("java", (".java",))
    source = "class Example { @org.junit.jupiter.api.Test void testOne() {} }"
    assert "void testOne" not in java.prune_tests(source, ["Example.testOne"])
    assert java.finalize_test_source(source) == source
