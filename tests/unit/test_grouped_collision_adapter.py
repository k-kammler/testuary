"""Java collision callers distinguish judged source from unavailable validation."""
import dataclasses
import sys

import pytest

import langadapter as lb
from langadapter.sources import CollisionResolution, SourceObservation, SourceStatus

from testuary.languages.generic import GenericBackend
from testuary.consolidation import ValidationUnresolved, _merge_sources

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


SOURCES = ("class First { @Test void first() { assert true; } }",
           "class Second { @Test void second() { assert true; } }")
RESOLUTION = CollisionResolution(
    SOURCES, renames=({}, {"old": "new"}), deduped=((), ("shared",)),
    dropped_defs=((), ("dropped",)), localized_autouse=((), ("setup",)),
    choices=(("binding", 1, (0,)),), conflicts=(("conflict", (0, 1)),))


#: The start of the refusal of an answer of another form.
REFUSED_ANSWER = "infrastructure_failure: TypeError: native resolution must return a SourceObservation"


def completed(**fields):
    """The hook's completed answer: RESOLUTION with ``fields`` replaced."""
    return SourceObservation(SourceStatus.COMPLETED, dataclasses.replace(RESOLUTION, **fields),
                             "fixture-parser")


class NativeCollisions(lb.LangAdapter):
    language = "java"

    def __init__(self, root, runtime=True):
        super().__init__(lb.ProjectContext(str(root),
                                          lb.Interpreter(path=sys.executable) if runtime else None))
        self.requests = []
        self.result = completed()
        self.selected = None

    def resolve_collisions(self, sources, *, preference=None, localize_autouse=False):
        self.requests.append((sources, localize_autouse))
        if preference is not None:
            self.selected = preference("binding", (0, 1))
        return self.result


@pytest.fixture
def adapter(tmp_path, monkeypatch):
    native = NativeCollisions(tmp_path)
    java = GenericBackend("java", (".java",))
    monkeypatch.setattr(java, "project_backend", lambda root=None:
                        native)
    return java, native


def test_completed_collision_retains_every_native_resolution_field(adapter):
    java, native = adapter
    result = java.resolve_test_collisions(SOURCES, localize_autouse=True)
    assert result == RESOLUTION
    assert native.requests == [(SOURCES, True)] and native.selected == 0


def test_judged_syntax_rejection_stays_a_rejection(adapter):
    java, native = adapter
    native.result = completed(ejected=(1,), eject_reasons=("compiler rejected source",))
    result = java.resolve_test_collisions(SOURCES)
    assert result.ejected == (1,) and result.eject_reasons == ("compiler rejected source",)
    assert result.sources == SOURCES and len(native.requests) == 1


@pytest.mark.parametrize("status", ["timeout", "missing_setup", "infrastructure_failure"])
def test_native_validation_failure_never_ejects_source_as_invalid(adapter, status):
    java, native = adapter
    native.result = SourceObservation(SourceStatus(status), reason="source 1: validator did not judge")
    with pytest.raises(ValidationUnresolved, match=f"{status}: source 1: validator did not judge"):
        java.resolve_test_collisions(SOURCES)
    assert len(native.requests) == 1


@pytest.mark.parametrize("status, message", [
    (SourceStatus.INFRASTRUCTURE_FAILURE, "infrastructure_failure: no collision verdict"),
    (SourceStatus.UNSUPPORTED, REFUSED_ANSWER),
    (SourceStatus.REFUSED, REFUSED_ANSWER),
])
def test_unavailable_group_has_no_completed_resolution(adapter, status, message):
    """No unfinished answer gives a resolution. One whose status no unrun
    validation gives is refused as an answer of another form."""
    java, native = adapter
    native.result = SourceObservation(status, reason="no collision verdict")
    with pytest.raises(ValidationUnresolved, match=message):
        java.resolve_test_collisions(SOURCES)
    assert len(native.requests) == 1


def test_missing_runtime_refuses_before_native_validation(tmp_path, monkeypatch):
    native = NativeCollisions(tmp_path, runtime=False)
    java = GenericBackend("java", (".java",))
    monkeypatch.setattr(java, "project_backend", lambda root=None:
                        native)
    with pytest.raises(ValidationUnresolved, match="missing_setup: no project runtime"):
        java.resolve_test_collisions(SOURCES)
    assert native.requests == []


@pytest.mark.parametrize("evidence,selected", [
    (({"a": {"L:target:1"}}, {"b": {"K:mutant"}}), 1),
    (({"a": {"L:target:1"}}, {"b": {"B:target:1:2"}}), 1),
    (({"a": {"K:mutant"}}, {"b": {"K:mutant"}}), 0),
])
def test_a_colliding_name_goes_to_the_owner_with_more_kills_then_branches_then_lines(
        adapter, evidence, selected):
    java, native = adapter
    java.resolve_test_collisions(SOURCES, evidence=evidence)
    assert native.selected == selected and len(native.requests) == 1


def test_actual_merge_propagates_unresolved_validation_without_ejecting(adapter):
    java, native = adapter
    native.result = SourceObservation(SourceStatus.TIMEOUT, reason="source 1: validator did not judge")
    record, log = {}, []
    with pytest.raises(ValidationUnresolved, match="timeout"):
        _merge_sources(java, SOURCES, log.append, outcome=record,
                       evidence=[None] * len(SOURCES), source_dir=".")
    assert "ejected_sources" not in record and len(native.requests) == 1
    assert not any("unavailable" in message or "ejected" in message for message in log)
