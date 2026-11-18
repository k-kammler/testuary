"""check, verify and generate admit one selection with one requirement projection."""
from dataclasses import replace
from types import SimpleNamespace
import subprocess

import pytest

from langadapter import ContainerExecution
from langadapter.capabilities import Availability

import test_generation_reuse_interruptions as harness
from test_capability_fallbacks import LimitedBackend, _report
from test_native_readiness_producer import providers as providers, settings_for, sources
from testuary import languages, orchestrator, readiness
from testuary import config
from testuary.orchestrator import RunRequest, _invocation
from testuary.languages import setup_requirements
from testuary.readiness import _native_report
from testuary.settings import ConfigurationError

COVERAGE_FACET = (("tests.run_suite", ("coverage",)),)


class AdmissionReached(Exception):
    pass


@pytest.fixture(autouse=True)
def no_native_launch(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("native launch before admission"))
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kw: pytest.fail("output cleanup"))


def admitted(monkeypatch):
    """Record what generation's preflight receives, and stop there."""
    seen = []

    def reached(names, root, requirements=None, groups=None):
        seen.append(SimpleNamespace(names=names, root=root, requirements=requirements, groups=groups))
        raise AdmissionReached

    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", reached)
    return seen


@pytest.mark.parametrize("stop,reuse,coverage", [
    (None, (), True),
    ("starter", (), True),
    (None, ("starter", "coverage", "mutation"), True),
    (None, ("starter", "coverage", "mutation", "finishing"), False),
])
def test_check_requires_coverage_whenever_the_workflow_measures_it(tmp_path, providers, stop, reuse, coverage):
    sources(tmp_path, "module.native-a")
    _native_report(RunRequest(("module.native-a",), stop_after_phase=stop, reuse=reuse),
                   settings_for(tmp_path, providers))
    request = providers[1][0][1]
    assert request.required_facets == (COVERAGE_FACET if coverage else ())


def test_check_refuses_a_language_filter_that_excludes_a_selected_target(tmp_path, providers):
    names, calls, _, _, _ = providers
    sources(tmp_path, "a.native-a", "b.native-b")
    with pytest.raises(ConfigurationError, match="excludes targets: b.native-b"):
        _native_report(RunRequest(("a.native-a", "b.native-b"), languages=(names[0],)),
                       settings_for(tmp_path, providers))
    assert calls == []


def test_generation_admits_only_the_selected_targets_languages(tmp_path, monkeypatch):
    project, _, _, settings = harness.setup_case(tmp_path)
    (project / "Unselected.java").write_text("class Unselected {}\n", encoding="utf-8")
    seen = admitted(monkeypatch)
    request = RunRequest(("module.py",))
    with pytest.raises(AdmissionReached):
        orchestrator.run_pipeline(request, settings=settings)
    assert len(seen) == 1 and seen[0].names == {"python"}
    assert [(group.language, len(group.sources)) for group in seen[0].groups] == [("python", 1)]
    with config.use_run_settings(settings):
        expected = readiness.workflow_requirements(_invocation(request))
    assert seen[0].requirements == expected and expected.facets == COVERAGE_FACET


def test_generation_refuses_a_language_filter_that_excludes_a_selected_target(tmp_path, monkeypatch):
    _, _, _, settings = harness.setup_case(tmp_path)
    seen = admitted(monkeypatch)
    with pytest.raises(ConfigurationError, match="excludes targets"):
        orchestrator.run_pipeline(RunRequest(("module.py",), languages=("java",)), settings=settings)
    assert seen == []


def test_container_generation_is_admitted_through_its_guest_route(tmp_path, monkeypatch):
    _, _, _, settings = harness.setup_case(tmp_path)
    work_dir = tmp_path / "workzone"
    work_dir.mkdir()
    settings = replace(settings, sections={
        **settings.sections, "main": {**settings.sections["main"], "project_exec": "container"},
        "project": {"source_dir": "/guest/repo", "python_executable": "/guest/python",
                    "output_dir": str(tmp_path / "output")},
    }, environment={"TESTUARY_PROJECT_CONTAINER": "fixture", "TESTUARY_PROJECT_WORK_DIR": str(work_dir)})
    seen = admitted(monkeypatch)
    with pytest.raises(AdmissionReached):
        orchestrator.run_pipeline(RunRequest(("/guest/repo/module.py",)), settings=settings)
    assert len(seen) == 1 and seen[0].names == {"python"}
    [group] = seen[0].groups
    assert group.context.root == "/guest/repo"
    assert isinstance(group.context.execution, ContainerExecution)
    assert group.context.execution.container == "fixture"


def test_preflight_assesses_each_group_at_its_root_against_the_workflow(tmp_path, monkeypatch, default_run):
    roots, messages = [], []

    class Recording(LimitedBackend):
        def capabilities(self, source_dir=None):
            roots.append(source_dir)
            return self.report

    backend = Recording(_report(mutation=(Availability.MISSING_SETUP, "mutation tool absent")))
    monkeypatch.setattr(languages, "get_backend", lambda language: backend)
    monkeypatch.setattr(orchestrator, "log_print", lambda message, **kw: messages.append(message))
    groups = tuple(SimpleNamespace(language="python", context=SimpleNamespace(root=str(tmp_path / name)))
                   for name in ("first", "second"))
    orchestrator._preflight_run_dependencies(
        {"python"}, tmp_path / "unused", setup_requirements(coverage=True, mutation=False), groups)
    assert roots == [str(tmp_path / "first"), str(tmp_path / "second")]
    with pytest.raises(ConfigurationError) as refused:
        orchestrator._preflight_run_dependencies(
            {"python"}, tmp_path / "unused", setup_requirements(coverage=True, mutation=True), groups)
    assert ("The language adapter lacks mutation, which Testuary needs: mutation tool absent"
            in str(refused.value))
