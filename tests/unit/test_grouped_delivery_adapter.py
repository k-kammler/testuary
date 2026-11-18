"""Delivery adapters preserve native settings through the public test group."""
import contextlib
import dataclasses
import sys

import pytest

from langadapter import LangAdapter, Interpreter, Outcome, ProjectContext
from langadapter.errors import InvalidRequest
from langadapter.java.adapter import JavaAdapter
from langadapter.python.adapter import PythonAdapter
from langadapter.outcomes import ExecutionIdentity, ObservationIdentity, ToolchainIdentity
from langadapter.runs import RunCounts, SuiteRun, TestOutcome
from langadapter.suites import TestId

from testuary.languages.generic import GenericBackend


class NativeDelivery(LangAdapter):
    def __init__(self, root, language, filename):
        super().__init__(ProjectContext(str(root), Interpreter(path=sys.executable)))
        self.language = language
        self.requests = []
        self.identity = ObservationIdentity(None, (), ToolchainIdentity(language, "fixture"),
                                            None, ExecutionIdentity("local"))
        qualname = "test_value" if language == "python" else "ExampleTest.value"
        self.result = SuiteRun(
            Outcome.TEST_FAILURES, "native assertion", RunCounts(1, 0, 1, 0, 0),
            (TestOutcome(TestId(f"a/{filename}", qualname), "failed", 0.1, "assertion"),),
            (), 0.1, "native output", None, self.identity)

    def run_as_delivered(self, request, *, sources=None):
        self.requests.append((request, sources))
        return self.result


@contextlib.contextmanager
def pass_through_measurement(backend, suite, *targets):
    """Stands in for the owned measurement copy: yields its inputs unchanged."""
    yield backend, suite, targets


@pytest.fixture(params=[lambda: GenericBackend("python", (".py",)), lambda: GenericBackend("java", (".java",))], ids=["python", "java"])
def delivery(request, tmp_path, monkeypatch):
    language = request.param()
    filename = "test_example.py" if language.name == "python" else "ExampleTest.java"
    paths = [tmp_path / part / filename for part in ("a", "b")]
    for path in paths:
        path.parent.mkdir()
        path.write_text("fixture source", encoding="utf-8")
    native = NativeDelivery(tmp_path, language.name, filename)
    # The language's own adapter, its test group the double's.
    adapter = PythonAdapter if language.name == "python" else JavaAdapter
    project = adapter(ProjectContext(str(tmp_path), Interpreter(path=sys.executable)))
    project._test_operations = native.tests
    monkeypatch.setattr(language, "project_backend", lambda root: project)
    module = __import__(type(language).__module__, fromlist=["prepared_measurement"])
    monkeypatch.setattr(module, "prepared_measurement", pass_through_measurement)
    return language, native, paths


@pytest.mark.parametrize("key_form", ["absolute", "relative"])
def test_replacement_paths_cross_the_relative_public_boundary(delivery, tmp_path, key_form):
    language, native, paths = delivery
    key = str(paths[0]) if key_form == "absolute" else paths[0].relative_to(tmp_path).as_posix()
    result = language.run_as_delivered(
        [str(path) for path in paths], str(tmp_path), timeout=17, seed=7,
        sources={key: "replacement source"})
    assert result.output == "native output"
    assert result.reason is None and len(result.failing) == 1
    assert result.cases[0].message == "assertion"
    request, sources = native.requests[0]
    assert request.timeout_s == 17
    assert sources == {paths[0].relative_to(tmp_path).as_posix(): "replacement source"}
    assert request.session_seed == (7 if language.name == "python" else None)
    assert request.reverse_order is False
    assert request.isolation_config is (language.name == "python")
    assert request.remove_created_files
    assert paths[0].read_text(encoding="utf-8") == "fixture source"


@pytest.mark.parametrize("outcome", [Outcome.TIMED_OUT, Outcome.MISSING_SETUP,
                                    Outcome.INFRASTRUCTURE_FAILURE, Outcome.NO_TESTS])
def test_unavailable_delivery_never_reports_green(delivery, tmp_path, outcome):
    language, native, paths = delivery
    native.result = dataclasses.replace(native.result, outcome=outcome, reason="native unavailable")
    result = language.run_as_delivered([str(path) for path in paths], str(tmp_path), timeout=17)
    assert result.failing is None and result.reason == "native unavailable"
    assert result.output == "native output" and len(result.cases) == 1


@pytest.mark.parametrize("replacement", ["outside", "unlisted", "not_text", "not_mapping"])
def test_invalid_replacements_are_refused_before_native_execution(delivery, tmp_path, replacement):
    language, native, paths = delivery
    sources = {
        "outside": {str(tmp_path.parent / "outside.py"): "source"},
        "unlisted": {"helper.py": "source"},
        "not_text": {str(paths[0]): None},
        "not_mapping": [(str(paths[0]), "source")],
    }[replacement]
    with pytest.raises(InvalidRequest, match="sources"):
        language.run_as_delivered([str(path) for path in paths], str(tmp_path),
                                  timeout=17, sources=sources)
    assert not native.requests


def test_no_replacement_stays_none(delivery, tmp_path):
    language, native, paths = delivery
    language.run_as_delivered([str(path) for path in paths], str(tmp_path), timeout=17)
    assert native.requests[0][1] is None


def test_reverse_order_is_native_or_explicitly_unsupported(delivery, tmp_path):
    language, native, paths = delivery
    result = language.run_as_delivered([str(path) for path in paths], str(tmp_path),
                                      timeout=17, reverse=True)
    if language.name == "python":
        assert native.requests[0][0].reverse_order
    else:
        assert result.failing is None and "reverse-order" in result.reason
        assert not native.requests
