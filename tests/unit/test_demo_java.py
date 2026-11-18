"""The recorded Java demo shares replay machinery without executing Java tools."""
from __future__ import annotations

import json
from dataclasses import asdict

import pytest
from langadapter.sources import CollisionResolution

from testuary.demo import cli, scenario
from testuary.demo.backend import ScenarioJavaBackend, ScenarioPitRunner
from testuary.languages.generic import GenericBackend

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

# The starter suite the Java scenario's authoring command writes for "zero".
SEED = ("package example;\n\nimport org.junit.jupiter.api.Test;\n"
        "import static org.junit.jupiter.api.Assertions.*;\n\n"
        "class CalculatorTest {\n    @Test\n"
        "    void zero() { assertEquals(0, Calculator.sign(0)); }\n}\n")


class _EveryDir(dict):
    """A recording's language facts, stated for every project directory."""

    def __init__(self, record):
        super().__init__()
        self._record = record

    def __contains__(self, key):
        return True

    def __getitem__(self, key):
        return self._record


@pytest.fixture(autouse=True)
def _recorded_facts(monkeypatch):
    """The backends here play outside a playback, so the shipped Java
    recording's language facts stand for every project directory."""
    from testuary.demo import backend as scenario_backend
    from testuary.demo import cli as demo_cli

    record = next(iter(demo_cli.load_data("java")[0]["language_facts"].values()))
    monkeypatch.setattr(scenario_backend, "RECORDED_LANGUAGE", _EveryDir(record))


def test_java_resources_are_a_recorded_run(tmp_path):
    manifest, files = cli.load_data("java")
    path = tmp_path / "scenario.json.gz"
    path.write_bytes(files["scenario.json.gz"])
    recorded = scenario.Scenario.load(path)
    assert manifest["language"] == "java"
    assert len(recorded.entries) == manifest["entries"]
    assert manifest["classes"] == {"real": manifest["entries"]}
    assert {entry.value_class for entry in recorded.entries} == {"real"}
    assert {entry.rule.split(":", 1)[0] for entry in recorded.entries} == {"recorded-v1"}
    # A loop that met its goal without a model request asks nothing in its
    # own segment.
    assert {"starter"} <= {entry.segment for entry in recorded.entries} <= {
        "starter", "coverage", "mutation"}
    assert all(entry.source.startswith("recorded run of Testuary") for entry in recorded.entries)
    assert "LICENSE-jsonassert.txt" in files
    assert "recorded in one real run" in manifest["statement"]
    assert manifest["language_facts"], "the recording states the Java language facts"


@pytest.mark.parametrize("language", ["jvm", "", "Java"])
def test_unknown_scenario_language_is_not_silently_python(language):
    with pytest.raises(ValueError, match="unsupported demo language"):
        cli.load_data(language)


def test_java_tree_keys_include_java_only(tmp_path):
    (tmp_path / "CalculatorTest.java").write_text(SEED, encoding="utf-8")
    (tmp_path / "test_ignored.py").write_text("def test_a(): pass\n", encoding="utf-8")
    (tmp_path / "conftest.py").write_text("raise RuntimeError()\n", encoding="utf-8")
    files = scenario.tree_files(tmp_path, language="java")
    assert [name for name, digest in files] == ["CalculatorTest.java"]
    assert scenario.suite_files([tmp_path], language="java") == {
        "files": [["dir", files]], "setup": []}


class _Answers:
    def __init__(self):
        self.calls = []

    def answer(self, call):
        self.calls.append(call)
        if call.kind == "backend.strip_module_level_skips":
            return {"source": SEED, "removed": 0}
        if call.kind == "backend.merge_test_files":
            return {"source": SEED}
        if call.kind == "backend.resolve_test_collisions":
            result = CollisionResolution(
                sources=(SEED,), ejected=(False,), eject_reasons=("",),
                renames=({},), deduped=((),), dropped_defs=((),), localized_autouse=((),))
            return {"resolution": asdict(result)}
        if call.kind == "backend.validate_test_source":
            return {"validation": {"ok": True, "status": "valid", "message": "synthetic fixture",
                                   "line": 0, "col": 0, "validator_identity": "synthetic fixture"}}
        raise AssertionError(call.kind)


def test_java_validated_rewrites_are_scripted_not_executed(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("native JDK-dependent source validation ran during playback")

    for name in ("strip_module_level_skips", "merge_test_files", "resolve_test_collisions"):
        monkeypatch.setattr(GenericBackend, name, forbidden)
    oracle = _Answers()
    backend = ScenarioJavaBackend(oracle)
    source = SEED
    assert backend.strip_module_level_skips(source) == (source, 0)
    assert backend.merge_test_files(source, source) == source
    result = backend.resolve_test_collisions([source], evidence=[{"CalculatorTest.zero": {"L:2", "L:1"}}])
    assert result.sources == (source,)
    assert [call.kind for call in oracle.calls] == [
        "backend.strip_module_level_skips", "backend.merge_test_files", "backend.resolve_test_collisions"]
    assert oracle.calls[-1].key["evidence"] == [{"CalculatorTest.zero": ["L:1", "L:2"]}]
    json.dumps(oracle.calls[-1].key)


def test_native_java_parsing_and_pruning_remain_real():
    backend = ScenarioJavaBackend(_Answers())
    source = SEED
    assert backend.check_structure(source).ok
    assert set(backend.function_fingerprints(source)) == {"CalculatorTest.zero"}
    assert "void zero" not in backend.prune_tests(source, ["CalculatorTest.zero"])


def test_java_facts_do_not_probe_a_jdk_and_no_python_setup_is_created(tmp_path, monkeypatch):
    """The playback states the recording's facts for the project directory."""
    from testuary import project
    from testuary.demo import backend as scenario_backend

    monkeypatch.delenv("JAVA_HOME", raising=False)
    monkeypatch.setattr(project, "project_runtime",
                        lambda *a, **k: pytest.fail("offline facts must not resolve a JDK"))
    record = scenario_backend.recorded_language(str(tmp_path))
    backend = ScenarioJavaBackend(_Answers())
    facts = backend.project_backend(tmp_path).language_facts()
    assert facts == scenario_backend.facts_from_record(record["facts"])
    assert facts.language == "Java"
    tests = tmp_path / "tests"
    backend.setup_test_env(str(tests), str(tmp_path))
    assert list(tests.iterdir()) == []
    assert isinstance(backend.mutation_runner({}), ScenarioPitRunner)


def test_java_syntax_salvage_uses_scripted_validator_without_source_execution(tmp_path):
    oracle = _Answers()
    report = {}
    source = SEED
    assert ScenarioJavaBackend(oracle).salvage_syntax(source, tmp_path, report=report) == source
    assert report["status"] == "valid"
    assert oracle.calls[-1].kind == "backend.validate_test_source"


@pytest.mark.parametrize("recorded", [None, {}])
def test_a_project_dir_without_recorded_facts_stops_the_playback(tmp_path, monkeypatch, recorded):
    """The playback's Java adapter has no JDK, so its own facts would name
    another JUnit generation than the recording's. A project directory the
    recording states no facts for stops the playback instead."""
    from testuary.demo import backend as scenario_backend

    monkeypatch.setattr(scenario_backend, "RECORDED_LANGUAGE", recorded)
    with pytest.raises(scenario.ScenarioStop, match="no language facts"):
        ScenarioJavaBackend(_Answers()).project_backend(tmp_path)
