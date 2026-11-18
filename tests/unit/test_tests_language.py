"""A language whose tests are written in another registered language, as a C
adapter's Python tests of an extension: Testuary names, places, prompts and
records them in the tests' language. Python and Java stay their own, and
Testuary asks their facts nothing to know it."""
from __future__ import annotations

import langadapter as lb
import pytest
from langadapter.adapter import ProjectSetting

from testuary import config, evidence, prompts
from testuary.languages import generic, get_backend, language_setting_values
from testuary.settings import resolve_settings


class Hosted(lb.LangAdapter):
    """A target language whose tests are Python's."""

    language = "hosted"
    extensions = (".hst",)
    display_name = "Hosted"
    code_fence = "hosted"
    test_file_name = "test_{stem}.hst"
    project_settings = (
        ProjectSetting("pytest_config", "The Python tests' pytest configuration.", "path",
                       default="auto"),
    )

    def language_facts(self):
        return lb.LanguageFacts(
            language="Hosted", test_framework="pytest", code_fence="python",
            runtime_version=None, test_file_patterns=("test_*.py",),
            test_function_pattern="test_*", test_class_pattern="Test*", timeout_marker=None,
            test_language="python", target_code_fence="hosted")


@pytest.fixture
def hosted(tmp_path, monkeypatch):
    lb.register("hosted", [".hst"], Hosted)
    monkeypatch.chdir(tmp_path)
    try:
        yield get_backend("hosted")
    finally:
        lb.unregister("hosted")


def test_the_tests_are_named_and_placed_in_their_own_language(hosted, tmp_path):
    assert hosted.test_language() == "python"
    assert hosted.test_file_extensions() == (".py",)
    assert hosted._test_file_name("calc") == "test_calc.py"
    assert hosted.scratch_test_path(str(tmp_path), "def test_a():\n    pass\n", "calc.hst",
                                    stage_as_delivered=False).name == "test_candidate.py"
    assert hosted.is_test_file(tmp_path / "test_calc.py")
    assert not hosted.is_test_file(tmp_path / "test_calc.hst")
    assert hosted.test_file_extensions() == (".py",)


def test_the_prompts_name_the_tests_language_and_fence_the_target_in_its_own(hosted, tmp_path,
                                                                            default_run):
    slots = prompts.fragments(hosted, str(tmp_path))
    assert (slots["lang_name"], slots["lang_code_fence"], slots["target_code_fence"],
            slots["lang_test_framework"]) == ("Python", "python", "hosted", "pytest")
    # The runtime line names the tests' language, also for a version the
    # adapter could not read.
    assert slots["lang_runtime_version"].startswith("Python")


def test_the_records_name_the_tests_language_beside_the_target_s(hosted, tmp_path, default_run):
    assert evidence._test_language("hosted", str(tmp_path)) == "python"
    condition = evidence.GenerationCondition(
        "starter", "hosted", {}, {}, "digest", True, True, test_language="python")
    assert condition.export()["test_language"] == "python"
    own = evidence.GenerationCondition("starter", "python", {}, {}, "digest", True, True)
    assert "test_language" not in own.export()


PYTHON_TESTS = "def test_one():\n    assert 1\n\n\ndef test_two():\n    assert 2\n"


def test_the_tests_sources_are_read_in_their_own_language(hosted):
    """The structural reads of the tests (their counts, skeleton and pool)
    take the tests' language's adapter, not the target's, whose source-only
    context knows no project and so reads its own language."""
    from langadapter.python.adapter import PythonAdapter

    assert isinstance(hosted.source_backend(), PythonAdapter)
    assert hosted.test_metrics(PYTHON_TESTS)["test_count"] == 2
    assert hosted.skeleton(PYTHON_TESTS) == ("test_one", "test_two")


def test_python_and_java_are_their_own_tests_language_without_asking(monkeypatch):
    def forbidden(self, source_dir=None):
        pytest.fail("Python's and Java's facts are not asked for the tests' language")

    monkeypatch.setattr(generic.GenericBackend, "language_facts", forbidden)
    for name in ("python", "java"):
        backend = get_backend(name)
        assert backend.test_language() == name
        assert backend.test_file_extensions() == tuple(backend.file_extensions)
        assert evidence._test_language(name, ".") is None


def test_a_declared_setting_testuary_owns_is_read_from_its_section(hosted, tmp_path):
    settings = resolve_settings(config_files=[], base_dir=tmp_path, models={}, environment={},
                                overrides={"main": {"pytest_config": "none"}})
    with config.use_run_settings(settings):
        assert language_setting_values("hosted") == {"pytest_config": "none"}


def test_a_suite_scored_on_its_own_run_is_named_in_the_tests_language(hosted):
    """A starter suite scored on its own coverage run, with no name staged for
    it, takes the tests' language's extension, as a suite measured from its
    slot does: Python tests of a hosted target are a .py file."""
    from types import SimpleNamespace

    from testuary import starters

    run = SimpleNamespace(elapsed_s=0.5, line_coverage=100.0, branch_coverage=None,
                          covered_lines=[1], missing_lines=[], covered_branches=None,
                          missing_branches=None)
    phase = SimpleNamespace(green={0: PYTHON_TESTS}, own_kms={}, starter_runs={0: run},
                            target_file_path="calc.hst", staged_names={}, backend=hosted,
                            suite_scoreds=[])
    starters._StarterPhase.scored_by_run(phase, 0, 0)
    (scored,) = phase.suite_scoreds
    assert scored.execution_name == f"test_{starters.slot_name(0)}.py"


def test_a_delivered_test_goes_to_the_projects_own_tests_in_the_tests_language(hosted, tmp_path):
    """The results view's default directory for a delivered Python test of a
    hosted target is the directory of the project's own Python tests, not the
    project's directory."""
    from testuary.web import _results

    project = tmp_path / "hostedproject"
    own = project / "tests"
    own.mkdir(parents=True)
    (project / "calc.hst").write_text("calc\n", encoding="utf-8")
    (own / "test_one.py").write_text(PYTHON_TESTS, encoding="utf-8")
    (own / "test_two.py").write_text(PYTHON_TESTS, encoding="utf-8")
    delivered = tmp_path / "test_calc.py"
    delivered.write_text(PYTHON_TESTS, encoding="utf-8")
    assert _results._tests_dir(project, delivered, hosted) == own
