"""Structural observations and native construction use the public registry."""
from types import SimpleNamespace
import subprocess

import pytest

import langadapter as lb
from langadapter.sources import SourceObservation, SourceStatus, TestMetrics

from testuary import languages as context_module
from testuary.languages.generic import GenericBackend
from testuary.languages.generic import TEST_COUNTERS

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("structural adapter checks must not launch a native tool")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)


@pytest.fixture(params=[lambda: GenericBackend("python", (".py",)),
                        lambda: GenericBackend("java", (".java",))],
                ids=["python", "java"])
def selected(request):
    adapter = request.param()
    previous = lb.registration(adapter.name)
    group = SimpleNamespace(result=None, calls=[])

    class Registered(lb.LangAdapter):
        language = adapter.name

        def test_metrics(self, source):
            group.calls.append(source)
            return group.result

    lb.register(adapter.name, previous.file_extensions, Registered, replace=True)
    try:
        yield adapter, group, Registered
    finally:
        lb.register(previous.name, previous.file_extensions, previous.factory, replace=True)


@pytest.mark.parametrize("status", [SourceStatus.PARSE_ERROR, SourceStatus.UNSUPPORTED,
                                   SourceStatus.MISSING_SETUP, SourceStatus.INFRASTRUCTURE_FAILURE])
def test_unavailable_structural_counts_never_become_empty_or_free(selected, status):
    adapter, group, _ = selected
    group.result = SourceObservation(status, parser="selected parser", reason="counts unavailable")
    assert adapter.test_metrics("source") == dict.fromkeys(TEST_COUNTERS)
    observed = adapter.test_metrics_observation("source")
    assert observed.status is status and observed.value is None
    assert observed.reason == "counts unavailable" and observed.parser == "selected parser"
    assert group.calls == ["source", "source"]
    assert adapter.source_backend().context.interpreter is None


@pytest.mark.parametrize("values", [(0, 0, 0, 0, 0), (9, 2, None, 0, None)])
def test_measured_zero_and_partial_optional_counts_keep_their_native_values(selected, values):
    adapter, group, _ = selected
    group.result = SourceObservation(SourceStatus.COMPLETED,
        TestMetrics(True, "selected parser", *values), parser="selected parser")
    assert adapter.test_metrics("source") == dict(zip(TEST_COUNTERS, values))


def test_unparsed_provider_record_does_not_supply_zero_counts(selected):
    adapter, group, _ = selected
    group.result = SourceObservation(SourceStatus.COMPLETED,
                                    TestMetrics(False, "selected parser", tests=0))
    assert adapter.test_metrics("source") == dict.fromkeys(TEST_COUNTERS)


def test_same_adapter_refreshes_its_source_factory_without_selecting_a_runtime(selected):
    adapter, _, first_factory = selected
    first = adapter.source_backend()
    assert isinstance(first, first_factory) and adapter.source_backend() is first

    class Replacement(lb.LangAdapter):
        language = adapter.name

    previous = lb.registration(adapter.name)
    lb.register(adapter.name, previous.file_extensions, Replacement, replace=True)
    second = adapter.source_backend()
    assert isinstance(second, Replacement) and second is not first
    assert second.context == first.context and second.context.interpreter is None


def test_project_factory_replacement_is_seen_by_the_existing_adapter(selected, monkeypatch, tmp_path):
    adapter, _, factory = selected
    runtime = tmp_path / "selected-runtime"
    runtime.touch()
    context = lb.ProjectContext(str(tmp_path), lb.Interpreter(path=str(runtime)))
    module = context_module
    monkeypatch.setattr(module, "language_project_context", lambda language, root=None: context)
    first = adapter.project_backend(str(tmp_path))
    assert isinstance(first, factory) and first.context is context

    class Replacement(lb.LangAdapter):
        language = adapter.name

    previous = lb.registration(adapter.name)
    lb.register(adapter.name, previous.file_extensions, Replacement, replace=True)
    second = adapter.project_backend(str(tmp_path))
    assert isinstance(second, Replacement) and second.context is context and second is not first


@pytest.mark.parametrize("adapter,source,count", [
    (lambda: GenericBackend("python", (".py",)), "def test_value():\n    assert True\n", 1),
    (lambda: GenericBackend("java", (".java",)),
     "/* header only */\nclass ExampleTest {\n@Test void first() {}\n@Test void second() {}\n}\n", 2),
])
def test_real_source_observations_preserve_language_counts_and_empty_zero(adapter, source, count):
    language = adapter()
    assert language.test_metrics(source)["test_count"] == count
    assert language.test_metrics("") == dict.fromkeys(TEST_COUNTERS, 0)
    observed = language.test_metrics_observation("broken (")
    assert observed.value is None and observed.reason and observed.parser
