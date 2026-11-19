"""Actual read-only setup composition, without native tools or model clients."""
from dataclasses import replace
import json
import os
import subprocess

import pytest

from langadapter import (
    Availability, LangAdapter, Environment, InvalidRequest, OperationCapability, Outcome, ProjectContext, ProjectSetupItem,
    ProjectSetupReport, register, unregister,
)
from testuary import orchestrator, readiness
from testuary import config
from testuary import project as project_module
from testuary.orchestrator import RunRequest
from testuary.readiness import _native_report
from testuary.settings import ConfigurationError, RunSettings
from test_output_admission import directory_alias as directory_alias


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("inspection launched execution, provisioning, cleanup or model construction")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(project_module, "project_runtime", forbidden)
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", forbidden)
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", forbidden)


@pytest.fixture
def providers(tmp_path_factory, monkeypatch):
    calls, answers, contexts = [], {}, []
    names = ("testuary-native-a", "testuary-native-b")
    # Each language's runtime fact names the program that runs its tests.
    runtime = tmp_path_factory.mktemp("runtime") / "selected-runtime"
    runtime.write_bytes(b"")

    def factory(language):
        class Native(LangAdapter):
            def inspect_project(self, request):
                return self._assess(request, "inspection")

            def verify_project(self, request):
                return self._assess(request, "verification")

            def _assess(self, request, scope):
                calls.append((language, request, scope, config.active_settings()))
                if language in answers:
                    return answers[language](request, scope)
                return ProjectSetupReport(
                    scope, Outcome.COMPLETED, "metadata observed", request.context, request.inputs,
                    (ProjectSetupItem("runtime", True, Availability.AVAILABLE, Outcome.COMPLETED,
                                      selected=request.context.interpreter.path, version="fixture version"),),
                    tuple(OperationCapability(name, True) for name in request.required_operations),
                )

        Native.language = language
        Native.runtime = str(runtime)
        return Native

    for name, suffix in zip(names, (".native-a", ".native-b")):
        register(name, (suffix,), factory(name))

    from testuary import languages as context_module
    built = context_module.setup_context

    def setup_context(language, settings):
        selected = built(language, settings)
        if language in names:
            contexts.append((settings, selected.root, {}))
        return selected

    monkeypatch.setattr(context_module, "setup_context", setup_context)
    try:
        yield names, calls, answers, contexts, str(runtime)
    finally:
        for name in names:
            unregister(name)


def settings_for(tmp_path, providers, *, sections=None, environment=None):
    sections = sections or {}
    return RunSettings(base_dir=tmp_path, sections={
        "api": {"model": "selected"},
        "project": {"source_dir": "original", "output_dir": "output"},
        **sections,
    }, models={"selected": {"provider": "openai", "model": "fixture",
                            "api_url": "https://example.test/v1", "api_key": "model-fixture-secret"}},
        environment=environment or {})


def sources(tmp_path, *names):
    root = tmp_path / "original"
    root.mkdir(exist_ok=True)
    for name in names:
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture source\n", encoding="utf-8")
    return root


@pytest.mark.parametrize("verify", [False, True])
def test_actual_composition_uses_original_captured_context_and_honest_scope(tmp_path, providers, verify, capsys):
    names, calls, _, contexts, _ = providers
    root = sources(tmp_path, "module.native-a")
    before = (root / "module.native-a").read_bytes()
    settings = settings_for(tmp_path, providers, environment={"PATH": "captured tools"})
    report = _native_report(RunRequest(("module.native-a",)), settings, verify=verify)
    assert report.scope == ("verification" if verify else "inspection")
    assert report.status == "unverified" and report.exit_code == (1 if verify else 0)
    assert report.source_dir == str(root) and report.credentials_configured is True
    assert len(calls) == len(contexts) == 1
    language, request, scope, active = calls[0]
    assert language == names[0] and scope == report.scope and active is settings
    assert request.context.root == str(root) and contexts[0][1] == str(root)
    assert request.context.environment.inherit is False
    assert dict(request.context.environment) == {"PATH": "captured tools"}
    # The tool check downloads what a run would. The quick check never does.
    assert request.source_files == ("module.native-a",) and request.allow_acquisition is verify
    assert request.inputs.required == ("module.native-a",)
    assert 0 < request.probe_timeout_s <= 10 and 0 < request.total_timeout_s <= 120
    assert next(item for item in report.items if item.key.endswith("item.runtime")).status == "verified"
    targets = next(item for item in report.items if item.key == "source.targets")
    assert (targets.status, targets.reason, targets.action, targets.tried_by) == (
        "verified", "Each target is in the project directory.", None, None)
    assert next(item for item in report.items if item.key.endswith("operation.tests.run_suite")).status == "unverified"
    assert config.active_settings() is None and not (tmp_path / "output").exists()
    assert (root / "module.native-a").read_bytes() == before
    assert capsys.readouterr() == ("", "")


def test_mixed_target_selection_has_distinct_language_namespaces(tmp_path, providers):
    names, calls, _, contexts, _ = providers
    sources(tmp_path, "a.native-a", "b.native-b", "c.native-a")
    report = _native_report(RunRequest(("unused.unknown", "b.native-b", "a.native-a", "c.native-a"), targets="2-4"),
                            settings_for(tmp_path, providers))
    assert report.source_files == ("b.native-b", "a.native-a", "c.native-a")
    assert [call[0] for call in calls] == [names[1], names[0]] and len(contexts) == 2
    assert [call[1].source_files for call in calls] == [("b.native-b",), ("a.native-a", "c.native-a")]
    assert len({item.key for item in report.items}) == len(report.items)
    assert not any("warmed" in operation or "derive" in operation or "per_test" in operation
                   for _, request, _, _ in calls for operation in request.required_operations)


@pytest.mark.parametrize("stop,reuse,overrides,mutation,delivered,timing", [
    (None, (), {}, True, False, False),
    ("starter", (), {}, False, False, False),
    ("coverage", (), {}, False, False, False),
    ("mutation", (), {}, True, False, False),
    ("finishing", ("mutation",), {}, False, False, False),
    (None, ("starter", "coverage", "mutation", "finishing"), {}, False, False, False),
    (None, ("mutation", "finishing"), {}, False, False, False),
    ("starter", (), {"main": {"phase_end_mutation": "true"}}, True, False, False),
    ("starter", (), {"coverage": {"report_reverse_order_failures": "true"}}, False, True, False),
    (None, ("mutation",), {"coverage": {"report_reverse_order_failures": "true"}}, False, True, False),
    (None, ("mutation",), {"mutation": {"report_reverse_order_failures": "true"}}, False, False, False),
    (None, ("starter", "coverage", "mutation"), {"finishing": {"measure_mutation": "true"}}, True, False, False),
    (None, ("mutation",), {"timing": {"measure": "true"}}, False, False, True),
    ("mutation", (), {"timing": {"measure": "true"}}, True, False, False),
    (None, ("starter", "coverage", "mutation", "finishing"), {"timing": {"measure": "true"}}, False, False, False),
])
def test_actual_workflow_projects_only_enabled_operation_flags(
        tmp_path, providers, stop, reuse, overrides, mutation, delivered, timing):
    sources(tmp_path, "module.native-a")
    sections = {
        "main": {"phase_end_mutation": "false"},
        "coverage": {"report_reverse_order_failures": "false"},
        "mutation": {"report_reverse_order_failures": "false"},
        "finishing": {"measure_mutation": "false"},
        "timing": {"measure": "false"},
    }
    for section, values in overrides.items():
        sections[section].update(values)
    _native_report(RunRequest(("module.native-a",), stop_after_phase=stop, reuse=reuse),
                   settings_for(tmp_path, providers, sections=sections))
    operations = providers[1][0][1].required_operations
    assert ("mutations.open_session" in operations) is mutation
    assert ("mutations.operators" in operations) is mutation
    # Delivered runs and timing are optional families even when their check
    # is configured: a language without them takes their fallback.
    assert "tests.run_as_delivered" not in operations
    assert "tests.benchmark" not in operations
    del delivered, timing


@pytest.mark.parametrize("source", ["./nested/module.native-a", "nested\\module.native-a", "absolute"])
def test_host_target_serialization_is_relative_and_does_not_resolve_links(tmp_path, providers, source):
    root = sources(tmp_path, "nested/module.native-a")
    if source == "absolute":
        source = str(root / "nested" / "module.native-a")
    if "\\" in source and os.name != "nt":
        pytest.skip("Windows path spelling is host-specific")
    _native_report(RunRequest((source,)), settings_for(tmp_path, providers))
    assert providers[1][0][1].source_files == ("nested/module.native-a",)


@pytest.mark.parametrize("source", ["../outside.native-a", "nested/../module.native-a", "outside", "unknown"])
def test_invalid_later_selection_prevents_all_native_assessments(tmp_path, providers, source):
    sources(tmp_path, "good.native-a")
    source = str(tmp_path / "foreign.native-a") if source == "outside" else "later.unknown" if source == "unknown" else source
    with pytest.raises((ConfigurationError, InvalidRequest)):
        _native_report(RunRequest(("good.native-a", source)), settings_for(tmp_path, providers))
    assert not providers[1] and config.active_settings() is None


def test_output_refusal_precedes_contexts_and_native_assessment(tmp_path, providers):
    sources(tmp_path, "module.native-a")
    output = tmp_path / "output"
    output.mkdir()
    retained = output / "foreign.txt"
    retained.write_text("keep", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="not empty"):
        _native_report(RunRequest(("module.native-a",)), settings_for(tmp_path, providers))
    assert not providers[1] and not providers[3] and retained.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize("absolute", [False, True])
def test_actual_source_alias_is_refused_by_native_admission_without_adoption(
        tmp_path, providers, directory_alias, absolute):
    """The language's check refuses a selected input through a link before
    its own part runs, and the check's targets item names the link and what
    to select instead."""
    root = sources(tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    saved = foreign / "module.native-a"
    saved.write_bytes(b"foreign source remains unchanged")
    alias = root / "alias"
    directory_alias(alias, foreign)
    source = str(alias / saved.name) if absolute else "alias/module.native-a"
    report = _native_report(RunRequest((source,)), settings_for(tmp_path, providers))
    targets = next(item for item in report.items if item.key == "source.targets")
    assert (targets.status, targets.reason, targets.tried_by) == (
        "missing_setup", f"selected project input cannot be a link: {root.resolve() / 'alias'}", None)
    assert targets.action == ("select the directory the link points to, or put the file or directory itself "
                              "in place of the link, then run the check again")
    assert report.status == "missing_setup" and report.exit_code == 1
    assert not providers[1] and saved.read_bytes() == b"foreign source remains unchanged"
    assert not (tmp_path / "output").exists() and config.active_settings() is None


@pytest.mark.parametrize("verify", [False, True])
def test_a_missing_target_is_missing_setup_with_the_language_checks_reason(tmp_path, providers, verify):
    """The language's check looks up the targets before its own part runs,
    and the check's targets item reports what it found."""
    sources(tmp_path)
    report = _native_report(RunRequest(("module.native-a",)), settings_for(tmp_path, providers), verify=verify)
    targets = next(item for item in report.items if item.key == "source.targets")
    assert (targets.status, targets.reason, targets.tried_by) == (
        "missing_setup", "a selected project input is missing", None)
    assert targets.action == "check that the selected project directory and files exist, then run the check again"
    assert not providers[1] and report.status == "missing_setup" and report.exit_code == 1


def test_same_settings_helpers_reuse_outer_scope_and_other_settings_are_refused(tmp_path, providers):
    from testuary.readiness import _configuration_report, _output_report, _setup_request

    root = sources(tmp_path, "module.native-a")
    selected = settings_for(tmp_path, providers)
    other = settings_for(tmp_path, providers)
    with config.use_run_settings(selected):
        report = _configuration_report(RunRequest(("module.native-a",)), selected)
        report = _output_report(report, RunRequest(("module.native-a",)), selected)
        context = ProjectContext(str(root), environment=Environment({}, inherit=False))
        request = _setup_request(RunRequest(("module.native-a",)), selected, context=context,
                                 required_operations=("tests.run_suite",))
        assert request.source_files == ("module.native-a",)
        assert config.active_settings() is selected
        with pytest.raises(ConfigurationError, match="conflicts"):
            _configuration_report(RunRequest(("module.native-a",)), other)
        assert config.active_settings() is selected
    assert config.active_settings() is None


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(7)])
def test_actual_provider_failure_or_cancellation_keeps_exact_cause_and_restores_scope(tmp_path, providers, error):
    sources(tmp_path, "module.native-a")

    def fail(request, scope):
        raise error

    providers[2][providers[0][0]] = fail
    with pytest.raises(type(error)) as caught:
        _native_report(RunRequest(("module.native-a",)), settings_for(tmp_path, providers))
    assert caught.value is error and config.active_settings() is None
    providers[2].clear()
    assert _native_report(RunRequest(("module.native-a",)), settings_for(tmp_path, providers)).exit_code == 0


def test_ordinary_native_io_failure_is_a_typed_failed_observation(tmp_path, providers):
    sources(tmp_path, "module.native-a")

    def fail(request, scope):
        raise OSError("observer unavailable")

    providers[2][providers[0][0]] = fail
    report = _native_report(RunRequest(("module.native-a",)), settings_for(tmp_path, providers))
    assert report.status == "failed" and report.exit_code == 1
    assert config.active_settings() is None


def test_one_total_deadline_caps_later_assessments_without_new_selection(tmp_path, providers, monkeypatch):
    sources(tmp_path, "a.native-a", "b.native-b")
    ticks = iter((100.0, 103.0, 108.0))
    monkeypatch.setattr(readiness, "time", type("Clock", (), {"monotonic": staticmethod(lambda: next(ticks))}))
    settings = settings_for(tmp_path, providers, sections={
        "project": {"source_dir": "original", "output_dir": "output", "setup_timeout_seconds": "7",
                    "setup_probe_timeout_seconds": "6"}})
    report = _native_report(RunRequest(("a.native-a", "b.native-b")), settings)
    assert len(providers[1]) == 1
    request = providers[1][0][1]
    assert request.total_timeout_s == request.probe_timeout_s == 4
    expired = next(item for item in report.items if item.key == "native.testuary-native-b.assessment")
    assert expired.status == "failed" and "setup_timeout_seconds" in expired.action
    assert report.status == "failed" and report.exit_code == 1
    # A language whose check never started has not confirmed its targets.
    targets = next(item for item in report.items if item.key == "source.targets")
    assert (targets.status, targets.reason, targets.tried_by) == (
        "unverified", "The check stopped before it confirmed the targets.", "run")


def test_unsupported_verify_is_not_successful_admission(tmp_path, providers):
    sources(tmp_path, "module.native-a")
    providers[2][providers[0][0]] = lambda request, scope: ProjectSetupReport(
        scope, Outcome.UNSUPPORTED, "verification not implemented", request.context, request.inputs, (),
        tuple(OperationCapability(name, False, readiness=Availability.UNSUPPORTED)
              for name in request.required_operations))
    report = _native_report(RunRequest(("module.native-a",)), settings_for(tmp_path, providers), verify=True)
    assert report.scope == "verification" and report.status == "unsupported" and report.exit_code == 1


def test_sequential_captured_inputs_do_not_drift(tmp_path, providers, monkeypatch):
    sources(tmp_path, "module.native-a")
    first = settings_for(tmp_path, providers, environment={"PATH": "first captured path"})
    second = settings_for(tmp_path, providers, environment={"PATH": "second captured path"})
    monkeypatch.setenv("PATH", "ambient later")
    for settings in (first, second):
        _native_report(RunRequest(("module.native-a",), resume=True), settings)
    assert [dict(call[1].context.environment) for call in providers[1]] == [dict(first.environment), dict(second.environment)]
    assert all(call[1].context.root == str(tmp_path / "original") for call in providers[1])
    assert config.active_settings() is None


@pytest.mark.parametrize("language,target,project,environment,expected", [
    ("python", "module.py", {"python_executable": "tools/python"}, {}, "explicit"),
    ("python", "module.py", {"python_version": "3.10.9", "python_executable": ""}, {}, "provisioning"),
    ("python", "module.py", {"python_version": "", "python_executable": ""}, {}, "absent"),
    ("java", "Example.java", {"java_home": "selected-jdk"}, {}, "explicit"),
    ("java", "Example.java", {"java_home": ""}, {}, "absent"),
])
def test_builtin_setup_context_enters_actual_producer_without_execution(
        tmp_path, providers, monkeypatch, language, target, project, environment, expected):
    import langadapter

    sources(tmp_path, target)
    captured = []

    def assess(selected_language, request):
        captured.append((selected_language, request))
        return ProjectSetupReport("inspection", Outcome.UNSUPPORTED, "controlled boundary", request.context,
                                  request.inputs, (), ())

    monkeypatch.setattr(langadapter, "inspect_project", assess)
    settings = settings_for(tmp_path, providers, environment=environment, sections={
        "project": {"source_dir": "original", "output_dir": "output", **project}})
    _native_report(RunRequest((target,)), settings)
    selected_language, request = captured[0]
    assert selected_language == language and request.context.root == str(tmp_path / "original")
    assert request.context.environment.inherit is False and config.active_settings() is None
    runtime = request.context.interpreter
    if expected == "absent":
        assert runtime is None
    elif expected == "provisioning":
        assert runtime.path is None and runtime.provisioning.version == "3.10.9"
    else:
        assert runtime.path.startswith(str(tmp_path)) and runtime.provisioning is None
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("verify", [0, 1, "true", None])
def test_verify_requires_explicit_boolean_before_side_effects(tmp_path, providers, verify):
    with pytest.raises(ConfigurationError):
        _native_report(RunRequest(("module.native-a",)), settings_for(tmp_path, providers), verify=verify)
    assert not providers[1] and not providers[3] and not list(tmp_path.iterdir())


def test_conflicting_active_run_is_refused_without_changing_it(tmp_path, providers):
    first = settings_for(tmp_path, providers)
    with config.use_run_settings(first):
        with pytest.raises(RuntimeError, match="already active"):
            _native_report(RunRequest(("module.native-a",)), settings_for(tmp_path, providers))
        assert config.active_settings() is first
    assert config.active_settings() is None


def test_composed_safe_export_and_real_configuration_warning_do_not_print(tmp_path, providers, capsys):
    sources(tmp_path, "module.native-a")

    def answer(request, scope):
        return ProjectSetupReport(scope, Outcome.COMPLETED, "model-fixture-secret native-fixture-secret",
            request.context, request.inputs,
            (ProjectSetupItem("runtime", True, None, None, reason="model-fixture-secret native-fixture-secret"),),
            tuple(OperationCapability(name, True) for name in request.required_operations))

    providers[2][providers[0][0]] = answer
    settings = replace(settings_for(tmp_path, providers, environment={"AUTH_TOKEN": "native-fixture-secret"},
                            sections={"project": {"name": "unknown-model-fixture-secret",
                                                  "source_dir": "original", "output_dir": "output"}}))
    report = _native_report(RunRequest(("module.native-a",)), settings)
    serialized = json.dumps(report.as_dict())
    assert "model-fixture-secret" not in serialized and "native-fixture-secret" not in serialized
    assert capsys.readouterr() == ("", "")


def test_guest_context_keeps_posix_targets_and_host_control_separate(tmp_path, providers, monkeypatch):
    """A container run: the language's context comes from the settings'
    container route, the target stays a guest path and the host's control
    environment stays out of the project's environment."""
    import langadapter

    seen = []
    settings = settings_for(tmp_path, providers, environment={
        "PATH": "host-only", "JAVA_HOME": "host-only-jdk",
        "TESTUARY_PROJECT_CONTAINER": "fixture-image",
        "TESTUARY_PROJECT_WORK_DIR": str(tmp_path / "work")},
        sections={"main": {"project_exec": "container"},
                  "project": {"source_dir": "/guest/project", "output_dir": "output"}})

    def assess(language, request):
        seen.append(request)
        return ProjectSetupReport("inspection", Outcome.UNSUPPORTED, "controlled guest observation", request.context,
                                  request.inputs, (), ())

    monkeypatch.setattr(langadapter, "inspect_project", assess)
    report = _native_report(RunRequest(("/guest/project/Module.native-a",)), settings)
    assert seen[0].source_files == ("Module.native-a",)
    assert not dict(seen[0].context.environment) and dict(seen[0].context.execution.control_environment) == dict(settings.environment)
    assert report.source_dir == "/guest/project" and report.execution == "container"
    assert next(item for item in report.items if item.key == "output.original_separation").status == "unverified"
    assert not list(tmp_path.iterdir())
