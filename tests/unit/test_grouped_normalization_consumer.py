"""The complete public source report reaches native and registered consumers."""
import hashlib
import subprocess

import pytest

import langadapter as lb
from langadapter.sources import RemovedBinding, SourceObservation, SourceStatus, SuiteNormalization
from agent_prompts import python_prompts

from testuary import languages
from testuary.agent import coverage_agent, mutation_agent

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("normalization consumers must not launch tools")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)


@pytest.mark.parametrize("language, source", [
    ("python", "def test_one():\n    assert False\n\ndef test_one():\n    assert True\n"),
    ("java", "class Example {\n void testOne() {}\n void testOne() { assert true; }\n}\n"),
    ("python", "def broken(:"), ("java", "class Example {"),
])
def test_builtin_consumer_reads_the_complete_grouped_report(language, source, monkeypatch):
    adapter = languages.get_backend(language)
    backend = adapter.source_backend()
    expected = backend.sources.normalize_bindings(source)
    calls = []
    native = backend.sources.normalize_bindings
    def recorded(value):
        calls.append(value)
        return native(value)
    monkeypatch.setattr(backend.sources, "normalize_bindings", recorded)
    result = adapter.normalize_shadowed_definitions(source)
    assert result == expected.value and calls == [source]
    assert result.rule_version and result.before_sha256 and result.after_sha256
    if result.parse_ok:
        assert result.removed_bindings
    else:
        assert not result.changed and result.source == source


@pytest.fixture
def registered():
    source = "first binding\nlast binding\n"
    retained = "last binding\n"
    report = SuiteNormalization(
        retained, True, True, None,
        (RemovedBinding(("Fixture",), "one", "fixture-method", 1, 1, 2, 2),),
        hashlib.sha256(source.encode()).hexdigest(),
        hashlib.sha256(retained.encode()).hexdigest(), "fixture-bindings-v1")
    observations = [SourceObservation(SourceStatus.COMPLETED, report, "fixture-parser")]
    calls = []

    class Registered(lb.LangAdapter):
        language = "normalization-consumer-fixture"
        file_extensions = (".normalization-consumer",)

        def normalize_bindings(self, source):
            calls.append(source)
            return observations[0]

    lb.register(Registered.language, Registered.file_extensions, Registered)
    try:
        yield languages.get_backend(Registered.language), source, report, observations, calls
    finally:
        lb.unregister(Registered.language)


def test_group_only_registration_needs_no_testuary_native_override(registered):
    adapter, source, report, _, calls = registered
    assert type(adapter) is languages.GenericBackend
    assert adapter.normalize_shadowed_definitions(source) == report
    assert calls == [source] and adapter.source_backend().context.interpreter is None


@pytest.mark.parametrize("status", [SourceStatus.MISSING_SETUP, SourceStatus.UNSUPPORTED,
                                    SourceStatus.INFRASTRUCTURE_FAILURE, SourceStatus.TIMEOUT])
def test_unavailable_report_is_not_unchanged_success(registered, status):
    adapter, source, _, observations, calls = registered
    observations[0] = SourceObservation(status, reason="analysis unavailable")
    with pytest.raises(RuntimeError, match=f"normalize_bindings: {status.value}: analysis unavailable"):
        adapter.normalize_shadowed_definitions(source)
    assert calls == [source]


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
def test_actual_agent_binding_gate_reads_registered_full_report(tmp_path, monkeypatch, registered, phase):
    adapter, source, report, _, calls = registered
    target = tmp_path / "target.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    shared = dict(target_file_path=str(target), target_file_content=target.read_text(),
                  source_dir=str(tmp_path), test_output_dir=str(tmp_path), language="python")
    if phase == "coverage":
        agent = coverage_agent.CoverageAgent(
            prompts=python_prompts("test_coverage"),
            deps=coverage_agent.CoverageTestDeps(**shared, key_functions=["f"]),
            session=coverage_agent.TestCoverageSession())
    else:
        agent = mutation_agent.MutationAgent(
            prompts=python_prompts("test_mutation"),
            deps=mutation_agent.MutationTestDeps(**shared, mutation_test_file_name="test_mutation.py"),
            session=mutation_agent.TestMutationSession())
    monkeypatch.setattr(agent, "test_backend", lambda: adapter)
    assert "Fixture.one" in agent._shadowed_definition_reject(source)
    if phase == "coverage":
        # The file the agent's backend names for the target.
        adapter.target_test_path(str(tmp_path), str(target)).write_text(source, encoding="utf-8")
    delivered = agent.target_path()
    delivered.write_text(source, encoding="utf-8")
    with pytest.raises(RuntimeError, match="shadowed definitions.*Fixture.one"):
        agent.assert_target_has_no_shadowed_definitions(phase + " final")
    assert delivered.read_text() == source
    assert calls == [source, source] and report.rule_version == "fixture-bindings-v1"
