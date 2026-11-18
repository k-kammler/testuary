"""Typed native observations do not activate setup or generation."""
from dataclasses import replace
import json
import subprocess

import pytest

from langadapter import (
    Availability, LangAdapter, ContainerExecution, Environment, Interpreter,
    OperationCapability, Outcome, ProjectContext, ProjectInputSelection,
    ProjectSetupItem, ProjectSetupReport, ProjectSetupRequest, inspect_project,
    register, unregister, verify_project,
)
from testuary import config
from testuary import project as project_module
from testuary.orchestrator import RunRequest
from testuary.readiness import _compose_setup_report, _configuration_report, _setup_request
from testuary.settings import ConfigurationError, RunSettings


@pytest.fixture(autouse=True)
def no_native_launch(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("native launch"))
    monkeypatch.setattr(project_module, "project_runtime", lambda *a, **k: pytest.fail("runtime probe"))
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)


@pytest.fixture
def provider():
    calls, answers = [], []

    class Native(LangAdapter):
        language = "testuary-readiness-fixture"

        def inspect_project(self, request):
            return self._assess(request, "inspection")

        def verify_project(self, request):
            return self._assess(request, "verification")

        def _assess(self, request, scope):
            calls.append((request, scope))
            return answers[0](request, scope) if answers else self._unsupported_setup(request, scope)

    register(Native.language, (".testuary-readiness-fixture",), Native)
    try:
        yield Native.language, calls, answers
    finally:
        unregister(Native.language)


def facts(tmp_path):
    settings = RunSettings(base_dir=tmp_path, sections={
        "api": {"model": "selected"}, "project": {"source_dir": "original", "output_dir": "output"},
    }, models={"selected": {"provider": "openai", "model": "fixture", "api_url": "https://example.test/v1",
                            "api_key": "model-fixture-secret"}})
    configuration = _configuration_report(RunRequest(["module.py"]), settings)
    context = ProjectContext(str(tmp_path / "original"), Interpreter("selected-python"),
                             environment=Environment({"SENSITIVE_TOKEN": "native-fixture-secret"}, inherit=False))
    request = ProjectSetupRequest(context, ("module.py",), ("tests.run_suite",),
                                  inputs=ProjectInputSelection(required=("module.py",)))
    return settings, configuration, request


def observation(request, *, item=None, scope="verification", outcome=Outcome.COMPLETED,
                operation=None, diagnostic_paths=()):
    operation = operation or OperationCapability("tests.run_suite", True, readiness=Availability.AVAILABLE)
    context = None if item is not None and item.required and item.ambiguous else request.context
    return ProjectSetupReport(scope, outcome, "assessment facts", context, request.inputs,
                              () if item is None else (item,), (operation,), diagnostic_paths)


@pytest.mark.parametrize("scope", ["inspection", "verification"])
@pytest.mark.parametrize("readiness,outcome,ambiguous,expected", [
    (None, None, False, "unverified"),
    (None, Outcome.COMPLETED, False, "unverified"),
    (Availability.AVAILABLE, Outcome.COMPLETED, False, "verified"),
    (Availability.MISSING_SETUP, None, False, "missing_setup"),
    (None, Outcome.MISSING_SETUP, False, "missing_setup"),
    (Availability.UNSUPPORTED, None, False, "unsupported"),
    (None, Outcome.UNSUPPORTED, False, "unsupported"),
    (None, None, True, "ambiguous"),
    (None, Outcome.COMPLETED, True, "ambiguous"),
    (None, Outcome.TIMED_OUT, False, "failed"),
    (Availability.MISSING_SETUP, Outcome.TIMED_OUT, False, "failed"),
    (None, Outcome.INFRASTRUCTURE_FAILURE, False, "failed"),
    (None, Outcome.TEST_FAILURES, False, "failed"),
    (None, Outcome.NO_TESTS, False, "failed"),
    (None, Outcome.COLLECTION_FAILED, False, "failed"),
])
def test_item_uses_typed_outcome_readiness_and_ambiguity(
        scope, readiness, outcome, ambiguous, expected, tmp_path):
    settings, configuration, request = facts(tmp_path)
    item = ProjectSetupItem("runtime", True, readiness, outcome, ambiguous=ambiguous,
                            reason="Words missing setup / unsupported / completed must not decide status.")
    report = observation(request, item=item, scope=scope)
    result = _compose_setup_report(configuration, report, request=request, settings=settings)
    mapped = next(value for value in result.items if value.key == "native.item.runtime")
    assert mapped.status == expected
    assert result.scope == scope
    assert result.status == ("unverified" if expected == "verified" else expected)
    assert result.exit_code == (2 if expected == "ambiguous" else 0 if scope == "inspection" and expected in {"verified", "unverified"} else 1)
    assert not list(tmp_path.iterdir()) and config.active_settings() is None


@pytest.mark.parametrize("ambiguous,label", [(True, "Candidates: "), (False, "Looked in: ")])
def test_an_items_candidates_are_the_choices_found_or_the_places_looked_in(ambiguous, label, tmp_path):
    """An ambiguous item's candidates are the choices the check found, and
    another item's are the places it looked in and found none: the report
    labels them so."""
    settings, configuration, request = facts(tmp_path)
    item = ProjectSetupItem("runtime", True, Availability.MISSING_SETUP, None, ambiguous=ambiguous,
                            reason="No JDK was chosen.", candidates=("/opt/jdk-17", "/opt/jdk-21"))
    result = _compose_setup_report(configuration, observation(request, item=item), request=request, settings=settings)
    mapped = next(value for value in result.items if value.key == "native.item.runtime")
    assert mapped.reason == "No JDK was chosen.\n" + label + "/opt/jdk-17, /opt/jdk-21"

@pytest.mark.parametrize("scope", ["inspection", "verification"])
@pytest.mark.parametrize("outcome,status,code", [
    (Outcome.COMPLETED, "unverified", None), (Outcome.MISSING_SETUP, "missing_setup", 1),
    (Outcome.UNSUPPORTED, "unsupported", 1), (Outcome.TIMED_OUT, "failed", 1),
    (Outcome.INFRASTRUCTURE_FAILURE, "failed", 1), (Outcome.TEST_FAILURES, "failed", 1),
    (Outcome.NO_TESTS, "failed", 1), (Outcome.COLLECTION_FAILED, "failed", 1),
])
def test_assessment_completion_is_not_unverified_input_or_operation_success(scope, outcome, status, code, tmp_path):
    settings, configuration, request = facts(tmp_path)
    report = observation(request, scope=scope, outcome=outcome)
    result = _compose_setup_report(configuration, report, request=request, settings=settings)
    assert result.status == status
    assert result.exit_code == (code if code is not None else 0 if scope == "inspection" else 1)
    assert {value.status for value in result.items if value.key in {"source.targets", "output.admission"}} == {"unverified"}


@pytest.mark.parametrize("scope", ["inspection", "verification"])
@pytest.mark.parametrize("supported,readiness,status", [
    (True, None, "unverified"), (True, Availability.AVAILABLE, "verified"),
    (True, Availability.MISSING_SETUP, "missing_setup"),
    (True, Availability.UNSUPPORTED, "unsupported"), (False, None, "unsupported"),
])
def test_required_operations_are_explicit_not_static_support_or_global_completion(scope, supported, readiness, status, tmp_path):
    settings, configuration, request = facts(tmp_path)
    operation = OperationCapability("tests.run_suite", supported, readiness=readiness)
    report = observation(request, scope=scope, operation=operation)
    result = _compose_setup_report(configuration, report, request=request, settings=settings)
    mapped = next(item for item in result.items if item.key == "native.operation.tests.run_suite")
    assert mapped.required and mapped.status == status


def test_missing_operation_is_unknown_not_success(tmp_path):
    settings, configuration, request = facts(tmp_path)
    report = replace(observation(request), operations=())
    result = _compose_setup_report(configuration, report, request=request, settings=settings)
    assert result.status == "unverified" and result.exit_code == 1
    missing = next(item for item in result.items if item.key == "native.operation.tests.run_suite")
    assert missing.status == "unverified" and "Update the language adapter" in missing.action


def test_operations_after_an_early_setup_stop_are_not_tried_and_keep_the_setup_action(tmp_path):
    # A refused container helper ends the assessment before any operation is
    # tried. The operations say so, and the helper's action is the only one.
    settings, configuration, request = facts(tmp_path)
    helper = ProjectSetupItem("guest_control", True, Availability.MISSING_SETUP, Outcome.MISSING_SETUP,
                              reason="no control helper", action="Select the container's Python.")
    report = replace(observation(request, item=helper, outcome=Outcome.MISSING_SETUP), operations=())
    result = _compose_setup_report(configuration, report, request=request, settings=settings)
    operation = next(item for item in result.items if item.key == "native.operation.tests.run_suite")
    assert operation.status == "unverified" and operation.action is None
    assert operation.reason == "Not tried: this language's check stopped at an earlier problem."
    actions = [item.action for item in result.items
               if item.key.startswith("native.item.") or item.key.startswith("native.operation.")]
    assert [action for action in actions if action] == ["Select the container's Python."]


@pytest.mark.parametrize("outcome", [Outcome.TIMED_OUT, Outcome.MISSING_SETUP, Outcome.UNSUPPORTED])
def test_optional_failed_prerequisite_does_not_erase_required_facts(outcome, tmp_path):
    settings, configuration, request = facts(tmp_path)
    # This converter never supplies these source and output admission outcomes itself.
    configuration = replace(configuration, items=tuple(
        replace(item, status="verified") for item in configuration.items if item.key != "native.execution"))
    report = observation(request, item=ProjectSetupItem("optional", False, None, outcome))
    result = _compose_setup_report(configuration, report, request=request, settings=settings)
    assert result.status == "verified" and result.exit_code == 0
    assert next(item for item in result.items if item.key == "native.item.optional").status != "verified"


def test_safe_export_then_captured_model_redaction_preserves_unknowns_and_spaces(tmp_path):
    settings, configuration, request = facts(tmp_path)
    execution = ContainerExecution("target", str(tmp_path / "work"), helper_python="/control/python",
                                   control_environment={"AUTH_TOKEN": "control-fixture-secret", "PATH": "host path"})
    context = replace(request.context, execution=execution)
    request = replace(request, context=context)
    item = ProjectSetupItem("runtime", True, None, None,
        reason="native-fixture-secret model-fixture-secret control-fixture-secret",
        action="select /jdk with spaces/bin/java", candidates=("/jdk with spaces/bin/java", "native-fixture-secret"))
    report = observation(request, item=item, diagnostic_paths=("/real diagnostic with spaces.txt",))
    result = _compose_setup_report(configuration, report, request=request, settings=settings)
    exported = json.dumps(result.as_dict())
    assert all(secret not in exported for secret in ("native-fixture-secret", "model-fixture-secret", "control-fixture-secret"))
    assert "environment" not in exported and "host path" not in exported
    mapped = next(value for value in result.items if value.key == "native.item.runtime")
    assert mapped.selected is mapped.version is None and mapped.status == "unverified"
    assert "/jdk with spaces/bin/java" in mapped.reason and mapped.action == "select /jdk with spaces/bin/java"
    assert next(value for value in result.items if value.diagnostic_path).diagnostic_path == "/real diagnostic with spaces.txt"
    assert result.source_dir == configuration.source_dir and result.output_dir == configuration.output_dir


@pytest.mark.parametrize("scope", ["inspection", "verification"])
@pytest.mark.parametrize("target", ["module.py", "Example.java"])
def test_actual_registered_facade_keeps_unsupported_unknown_and_captured_inputs(scope, target, tmp_path, monkeypatch, provider):
    language, calls, _ = provider
    settings, configuration, request = facts(tmp_path)
    project = tmp_path / "original"
    project.mkdir()
    (project / target).write_text("class Example {}" if target.endswith(".java") else "value = 1\n", encoding="utf-8")
    monkeypatch.setenv("PATH", "later-ambient-path")
    monkeypatch.setenv("JAVA_HOME", "later-ambient-java")
    request = _setup_request(RunRequest((target,)), settings, context=request.context,
                             required_operations=("tests.run_suite",),
                             inputs=ProjectInputSelection(required=(target,)))
    before = {path: path.read_bytes() for path in project.iterdir()}
    report = (inspect_project if scope == "inspection" else verify_project)(language, request)
    result = _compose_setup_report(configuration, report, request=request, settings=settings)
    assert report.outcome is Outcome.UNSUPPORTED
    assert result.scope == scope and result.status == "unsupported" and result.exit_code == 1
    assert request.context.environment.inherit is False and "PATH" not in request.context.environment
    assert len(calls) == 1 and calls[0][0].context is request.context and calls[0][1] == scope
    assert calls[0][0].allow_acquisition is False
    assert calls[0][0].total_timeout_s == request.total_timeout_s
    assert {path: path.read_bytes() for path in project.iterdir()} == before
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("scope", ["inspection", "verification"])
def test_actual_facade_ambiguous_report_retains_redaction_without_selected_context(scope, tmp_path, provider):
    language, _, answers = provider
    settings, configuration, request = facts(tmp_path)
    project = tmp_path / "original"
    project.mkdir()
    (project / "module.py").write_text("value = 1\n", encoding="utf-8")
    request = _setup_request(RunRequest(("module.py",)), settings, context=request.context,
                             required_operations=("tests.run_suite",))

    def assess(selected, actual_scope):
        item = ProjectSetupItem("runtime", True, None, None, ambiguous=True,
            reason="native-fixture-secret model-fixture-secret",
            action="select /runtime with spaces", candidates=("/runtime with spaces", "native-fixture-secret"))
        return observation(selected, item=item, scope=actual_scope)

    answers.append(assess)
    report = (inspect_project if scope == "inspection" else verify_project)(language, request)
    assert report.context is None
    result = _compose_setup_report(configuration, report, request=request, settings=settings)
    exported = json.dumps(result.as_dict())
    assert "native-fixture-secret" not in exported and "model-fixture-secret" not in exported
    assert "/runtime with spaces" in exported and result.status == "ambiguous" and result.exit_code == 2


@pytest.mark.parametrize("scope", ["inspection", "verification"])
def test_composed_request_actual_facade_observations_do_not_admit_source_or_output(scope, tmp_path, provider):
    language, calls, answers = provider
    settings, configuration, original = facts(tmp_path)
    project = tmp_path / "original"
    project.mkdir()
    source = project / "module.py"
    source.write_text("value = 1\n", encoding="utf-8")
    requested = _setup_request(RunRequest(("module.py",)), settings, context=original.context,
                               required_operations=("tests.run_suite",))

    def assess(selected, actual_scope):
        return observation(selected, scope=actual_scope, item=ProjectSetupItem(
            "runtime", True, Availability.AVAILABLE, Outcome.COMPLETED,
            selected="/selected runtime/python", version="observed-fixture-version"))

    answers.append(assess)
    report = (inspect_project if scope == "inspection" else verify_project)(language, requested)
    result = _compose_setup_report(configuration, report, request=requested, settings=settings)
    runtime = next(item for item in result.items if item.key == "native.item.runtime")
    assert runtime.status == "verified" and runtime.selected == "/selected runtime/python"
    assert runtime.version == "observed-fixture-version"
    assert result.status == "unverified" and result.exit_code == (0 if scope == "inspection" else 1)
    assert calls[0][0].inputs.required == ("module.py",)
    assert source.read_text(encoding="utf-8") == "value = 1\n" and not (tmp_path / "output").exists()


@pytest.mark.parametrize("field", ["configuration", "report", "request", "settings"])
def test_converter_refuses_untyped_inputs_without_execution(field, tmp_path):
    settings, configuration, request = facts(tmp_path)
    report = observation(request)
    values = dict(configuration=configuration, report=report, request=request, settings=settings)
    values[field] = object()
    with pytest.raises(ConfigurationError, match="typed configuration"):
        _compose_setup_report(values["configuration"], values["report"], request=values["request"], settings=values["settings"])


@pytest.mark.parametrize("targets,sources", [(None, ("a.py", "path with,comma.py", "third.py")),
                                            ("2", ("path with,comma.py",)), ("2-3", ("path with,comma.py", "third.py"))])
def test_setup_request_uses_selected_targets_deadlines_and_no_acquisition(targets, sources, tmp_path):
    settings, _, original = facts(tmp_path)
    settings = replace(settings, sections={
        "project": {"setup_timeout_seconds": "13.5", "setup_probe_timeout_seconds": "2.25"}})
    invocation = RunRequest(("a.py", "path with,comma.py", "third.py"), targets=targets)
    inputs = ProjectInputSelection(exclude=(".venv",), required=("framework.ini",))
    requested = _setup_request(invocation, settings, context=original.context,
                               required_operations=("tests.run_suite",), inputs=inputs)
    assert requested.source_files == sources
    assert requested.context is original.context and requested.inputs is inputs
    assert requested.probe_timeout_s == 2.25 and requested.total_timeout_s == 13.5
    assert requested.allow_acquisition is False and requested.required_operations == ("tests.run_suite",)
    assert config.active_settings() is None and not list(tmp_path.iterdir())


def test_request_deadlines_come_from_the_project_settings(tmp_path):
    settings, _, original = facts(tmp_path)
    settings = replace(settings, sections={"project": {
        "name": "fixture", "setup_timeout_seconds": "27", "setup_probe_timeout_seconds": "3"}})
    requested = _setup_request(RunRequest(("module.py",)), settings, context=original.context,
                               required_operations=("tests.run_suite",))
    assert requested.total_timeout_s == 27 and requested.probe_timeout_s == 3
    assert requested.inputs is None and requested.context is original.context


@pytest.mark.parametrize("mode", ["local", "container"])
def test_request_preserves_captured_absence_and_host_guest_separation(mode, tmp_path, monkeypatch):
    settings, _, original = facts(tmp_path)
    values = {"EXPLICIT_TARGET": "native-fixture-secret"}
    context = replace(original.context, environment=Environment(values, inherit=False))
    if mode == "container":
        context = replace(context, root="/guest/source", execution=ContainerExecution(
            "selected-container", str(tmp_path / "host-work"), repository_dir="/guest/source",
            container_work_dir="/guest/work", helper_python="/guest/control/python",
            control_environment={"PATH": "captured-host-path", "JAVA_HOME": "captured-host-jdk"}))
    monkeypatch.setenv("PATH", "later-host-path")
    monkeypatch.setenv("JAVA_HOME", "later-host-jdk")
    requested = _setup_request(RunRequest(("module.py",)), settings, context=context,
                               required_operations=("tests.run_suite",))
    values["PATH"] = "mutated-input"
    assert requested.context is context and not context.environment.inherit
    assert "PATH" not in context.environment and "JAVA_HOME" not in context.environment
    assert "native-fixture-secret" not in repr(requested)
    if mode == "container":
        assert context.execution.control_environment["PATH"] == "captured-host-path"
        assert context.execution.control_environment["JAVA_HOME"] == "captured-host-jdk"
        assert not context.execution.control_environment.inherit
        assert context.execution.helper_python == "/guest/control/python"


@pytest.mark.parametrize("failure", ["local_overlay", "guest_overlay", "missing_host_control"])
def test_request_refuses_ambient_inheriting_context_before_setup(failure, tmp_path):
    settings, _, original = facts(tmp_path)
    context = original.context
    if failure != "local_overlay":
        context = replace(context, root="/guest/source", execution=ContainerExecution(
            "selected-container", str(tmp_path / "host-work"),
            control_environment={} if failure == "guest_overlay" else None))
    if failure != "missing_host_control":
        context = replace(context, environment={})
    with pytest.raises(ConfigurationError, match="captured"):
        _setup_request(RunRequest(("module.py",)), settings, context=context,
                       required_operations=("tests.run_suite",))
    assert config.active_settings() is None and not list(tmp_path.iterdir())


@pytest.mark.parametrize("field", ["invocation", "settings", "context"])
def test_request_requires_typed_inputs_without_execution(field, tmp_path):
    settings, _, original = facts(tmp_path)
    values = dict(invocation=RunRequest(("module.py",)), settings=settings, context=original.context)
    values[field] = object()
    with pytest.raises(ConfigurationError, match="selected ProjectContext"):
        _setup_request(values["invocation"], values["settings"], context=values["context"],
                       required_operations=("tests.run_suite",))
