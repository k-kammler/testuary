"""Public registration, grouped source translation and explicit run refusal."""
from pathlib import Path
import subprocess

import pytest

import langadapter as lb
from langadapter.capabilities import Availability, Capability, CapabilityReport
from langadapter.outcomes import Outcome
from langadapter.sources import StructureCheck, TestMetrics

from testuary import languages
from testuary.languages import assess
from testuary.prompts import SLOTS

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture(autouse=True)
def forbid_processes(monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("registered source fixtures cannot start processes")
    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(subprocess, "run", refused)


@pytest.fixture
def registered():
    calls = []

    class Registered(lb.LangAdapter):
        language = "fixture_registered"
        file_extensions = (".fixture",)

        def __init__(self, context):
            super().__init__(context)
            calls.append(self)

        def check_structure(self, source):
            return StructureCheck(True, "", 0, 0, "fixture-parser")

        def test_metrics(self, source):
            return TestMetrics(True, "fixture-parser", 5, 2, 1, None, None)

        def skeleton(self, source):
            return ("Suite.two",)

        def function_fingerprints(self, source):
            return {"Suite.two": "fingerprint"}

        def imports(self, source):
            return ("fixture/library",)

        def is_test_file(self, path):
            return Path(path).name.startswith("test_")

    lb.register(Registered.language, Registered.file_extensions, Registered)
    try:
        yield Registered, calls
    finally:
        lb.unregister(Registered.language)


def test_registration_after_import_reaches_detection_and_cli_discovery(registered):
    factory, calls = registered
    assert factory.language in languages.available_language_names()
    assert languages.get_backend(factory.language) is not None
    assert ".fixture" in languages.supported_file_extensions()
    assert languages.detect_language("source.FIXTURE") == factory.language
    assert calls == []  # Discovery is not readiness or a tool probe.


def test_group_only_backend_reaches_common_source_translation(registered):
    factory, calls = registered
    adapter = languages.get_backend(factory.language)
    assert type(adapter) is languages.GenericBackend
    assert adapter.check_structure("source").ok
    assert adapter.structural_parser_identity() == "fixture-parser"
    assert adapter.skeleton("source") == ("Suite.two",)
    assert adapter.function_fingerprints("source") == {"Suite.two": "fingerprint"}
    assert adapter.imports("source") == ("fixture/library",)
    assert adapter.is_test_file(Path("test_subject.fixture"))
    assert adapter.test_metrics("source") == {
        "loc": 5, "test_count": 2, "assertion_count": 1,
        "fixture_count": None, "parametrize_case_count": None}
    # The project's adapter is asked once for the tests' language (a language
    # other than Python and Java may write them in another), and the source-only
    # context's reads them.
    assert len(calls) == 2
    assert calls[0].context != calls[1].context
    assert calls[1].context.interpreter is None
    assert adapter.test_metrics_observation("source").parser == "fixture-parser"


def test_reads_a_language_does_not_offer_are_empty():
    """A language whose adapter offers no skeleton, function fingerprints
    or imports reads as having none, without a read of the source. Its one
    hook answers, so the reads reach the registered adapter."""
    checked = []

    class Bare(lb.LangAdapter):
        language = "fixture_bare"
        file_extensions = (".bare",)

        def check_structure(self, source):
            checked.append(source)
            return StructureCheck(True, "", 0, 0, "fixture-parser")

    lb.register(Bare.language, Bare.file_extensions, Bare)
    try:
        adapter = languages.get_backend(Bare.language)
        assert adapter.skeleton("source") == ()
        assert adapter.function_fingerprints("source") == {}
        assert adapter.imports("source") == ()
        assert checked == []
        assert adapter.check_structure("source").ok and checked == ["source"]
    finally:
        lb.unregister(Bare.language)


def test_unregister_and_replacement_do_not_leave_import_time_discovery(registered):
    factory, calls = registered
    adapter = languages.get_backend(factory.language)
    adapter.test_metrics("source")
    previous = calls[-1]
    lb.register(factory.language, (".replaced",), factory, replace=True)
    assert languages.get_backend(factory.language) is adapter
    assert adapter.file_extensions == (".replaced",)
    assert languages.detect_language("subject.fixture") == "unknown"
    assert languages.detect_language("subject.replaced") == factory.language
    adapter.test_metrics("source")
    assert calls[-1] is not previous
    lb.unregister(factory.language)
    assert languages.get_backend(factory.language) is None
    assert factory.language not in languages.available_language_names()
    assert ".replaced" not in languages.supported_file_extensions()
    # Restore the fixture registration so its finalizer has an owned entry.
    lb.register(factory.language, factory.file_extensions, factory)


def test_registered_source_backend_has_explicit_execution_refusal(registered, tmp_path):
    factory, calls = registered
    adapter = languages.get_backend(factory.language)
    report = adapter.capabilities(str(tmp_path))
    assert report.outcome is Outcome.MISSING_SETUP
    assert "no project runtime" in report.reason
    assert assess(report).refusals
    # The generic backend: the solver check as for every language, and the
    # prompt words from the language's facts.
    assert set(adapter.prompt_fragments(str(tmp_path))) == set(SLOTS)
    assert len(calls) == 1
    assert calls[0].context.interpreter is None


@pytest.mark.parametrize("required_missing", [True, False])
def test_registered_report_preserves_required_and_optional_assessment(registered, monkeypatch, tmp_path,
                                                                     required_missing):
    factory, _ = registered
    runtime = tmp_path / "runtime"
    runtime.touch()
    backend = lb.create(factory.language, lb.ProjectContext(str(tmp_path), lb.Interpreter(str(runtime))))
    capabilities = tuple(Capability(name, Availability.UNSUPPORTED if required_missing and name == "mutation"
                                    else Availability.AVAILABLE)
                         for name in ("suite_runs", "coverage", "mutation"))
    report = CapabilityReport(factory.language, str(runtime), Outcome.COMPLETED, "", None,
                              capabilities, (), "none", ())
    monkeypatch.setattr(backend, "capabilities", lambda: report, raising=False)
    adapter = languages.get_backend(factory.language)
    # The adapter is a cached singleton: patched on the instance, the method
    # would stay shadowed after teardown (tests/conftest.py's guard).
    monkeypatch.setattr(type(adapter), "project_backend", lambda self, root=None: backend)
    # The report as the backend gives it, without the probe's operation list.
    monkeypatch.setattr(type(adapter), "capabilities", lambda self, root=None: report)
    result = assess(adapter.capabilities(str(tmp_path)))
    assert bool(result.refusals) is required_missing
    if required_missing:
        assert "mutation" in result.refusals[0]
    assert {f.family for f in result.applied} == {
        "kill attribution", "merging", "per-test evidence"}
