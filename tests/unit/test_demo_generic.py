"""The demonstration of a language Testuary has no code for: the adapter its
manifest names and its recorded capability report."""
from __future__ import annotations

import langadapter as la
import pytest
from langadapter.capabilities import (
    Availability, Capability, CapabilityReport, OperationCapability, ToolVersion,
)
from langadapter.outcomes import Outcome
from langadapter.versions import Version, capability_runtime

from testuary.demo import backend as generic
from testuary.demo import cli as demo_cli
from testuary.demo import recorded

TOY = {"package": "langadapter.tutorial.example", "path": "steps/06_versions.py",
       "class": "ToyAdapter"}


def test_a_capability_report_comes_back_as_it_was_recorded():
    version = Version("toy", (1, 2), "1.2", "probe")
    report = CapabilityReport(
        language="toy", interpreter="/usr/bin/toy", outcome=Outcome.COMPLETED,
        reason="", runtime=capability_runtime(version),
        capabilities=(Capability("suite_runs", Availability.AVAILABLE),
                      Capability("mutation", Availability.MISSING_SETUP, "no engine")),
        tools=(ToolVersion("toyrun", "1.0", None),),
        branch_coverage="none", mutation_engines=("toy",),
        operations=(OperationCapability("tests.run_suite", True,
                                        readiness=Availability.AVAILABLE,
                                        execution_backed=True,
                                        execution_kinds=("run",)),
                    OperationCapability("tests.benchmark", False, "not implemented")),
        version_record=version)
    assert recorded.report_from_json(recorded.report_json(report)) == report


def test_an_adapter_is_loaded_from_a_file_of_an_installed_package_or_a_module():
    toy = generic.load_adapter(TOY)
    assert toy.__name__ == "ToyAdapter"
    assert generic.load_adapter({"module": "langadapter.python.adapter",
                                 "class": "PythonAdapter"}).__name__ == "PythonAdapter"


def test_an_adapter_file_is_found_one_path_part_at_a_time(monkeypatch):
    """Python 3.10's package files take one path part per join, so a path
    of several parts is joined part by part."""
    real = generic.resources.files

    class OnePart:
        def __init__(self, inner):
            self.inner = inner

        def joinpath(self, child):
            return OnePart(self.inner.joinpath(child))

        def __str__(self):
            return str(self.inner)

    monkeypatch.setattr(generic.resources, "files", lambda package: OnePart(real(package)))
    assert generic.load_adapter(TOY).__name__ == "ToyAdapter"


def test_registering_is_undone_and_leaves_a_registered_language_alone():
    assert "toy" not in la.names()
    undo = generic.register("toy", [".toy"], TOY)
    try:
        assert "toy" in la.names()
        assert generic.register("toy", [".toy"], TOY)() is None
        assert "toy" in la.names()
    finally:
        undo()
    assert "toy" not in la.names()


def test_the_toy_data_name_their_adapter_and_load():
    manifest, files = demo_cli.load_data("toy")
    assert manifest["language"] == "toy"
    assert manifest["adapter"] == TOY
    assert manifest["file_extensions"] == [".toy"]
    assert "scenario.json.gz" in files


def test_the_generic_scenario_runner_takes_the_real_runners_rules():
    """The playback's generic runner selects the catalogue, its named
    operators and the engine's own per-mutant timeout rule as the generic
    runner does, from the backend."""
    from types import SimpleNamespace

    from langadapter.mutation import Catalogue

    backend = SimpleNamespace(mutation_catalogue=lambda config: Catalogue(name="core"),
                              named_operators=lambda name: ("op/a", "op/b"),
                              own_mutant_timeout_rule=True)
    runner = generic.ScenarioGenericRunner(None, {"per_mutant_timeout_seconds": "auto"},
                                           engine="toy", language="toy", backend=backend)
    assert runner.catalogue() == Catalogue(name="core")
    assert runner.configured_operators() == ["op/a", "op/b"]
    assert runner.engine_sets_timeout() is True


@pytest.mark.usefixtures("default_run")
def test_a_toy_playback_runner_answers_the_rules_as_the_real_runner_does(tmp_path):
    """The playback backend builds its generic runner over the language's
    adapter, so its catalogue, named operators and timeout rule are the real
    runner's."""
    from testuary import languages
    from testuary.settings import _defaults, phase_section

    config = dict(_defaults()[phase_section("mutation")])
    undo = generic.register("toy", [".toy"], TOY)
    try:
        real = languages.get_backend("toy").mutation_runner(config, source_dir=str(tmp_path))
        played = generic.ScenarioGenericBackend(None, "toy", [".toy"]).mutation_runner(
            config, source_dir=str(tmp_path))
        assert played.catalogue() == real.catalogue()
        assert played.configured_operators() == real.configured_operators()
        assert played.engine_sets_timeout() == real.engine_sets_timeout()
    finally:
        undo()


def test_a_toy_run_is_keyed_by_its_own_test_files(tmp_path):
    """A language other than Python and Java keys its runs by the test files
    its backend's rule names, so a changed Toy suite is a new question."""
    from pathlib import Path

    from testuary import languages
    from testuary.demo.scenario import tree_files

    undo = generic.register("toy", [".toy"], TOY)
    try:
        toy = languages.get_backend("toy")
        name = next(candidate for candidate in ("test_calc.toy", "calc_test.toy")
                    if toy.is_test_file(Path(candidate)))
        (tmp_path / name).write_text("check 1\n", encoding="utf-8")
        (tmp_path / "test_other.py").write_text("", encoding="utf-8")
        assert [path for path, _digest in tree_files(tmp_path, language="toy")] == [name]
    finally:
        undo()
