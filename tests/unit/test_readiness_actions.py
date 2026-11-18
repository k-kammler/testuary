"""Every required native operation that is not confirmed names a next step.

The adapter gives each operation an action and refuses a report that leaves
out a requested operation, so Testuary's own action for an unreported one is
a fallback for other providers.
"""
from __future__ import annotations

import pytest

from langadapter import (
    Availability, OperationCapability, Outcome, ProjectSetupItem, ProjectSetupReport,
)

from test_native_readiness_producer import providers as providers, settings_for, sources
from testuary.orchestrator import RunRequest
from testuary.readiness import _native_report


def _report_with(operations):
    def answer(request, scope):
        return ProjectSetupReport(
            scope, Outcome.COMPLETED, "metadata observed", request.context, request.inputs,
            (ProjectSetupItem("runtime", True, Availability.AVAILABLE, Outcome.COMPLETED,
                              selected="selected-runtime", version="fixture version"),),
            operations(request))
    return answer


def test_operation_actions_pass_through_and_a_missing_action_stays_absent(tmp_path, providers):
    names, _, answers, _, _ = providers
    sources(tmp_path, "module.native-a")

    def operations(request):
        listed = list(request.required_operations)
        return tuple(
            OperationCapability(name, True, reason="installed, not tried",
                                readiness=Availability.MISSING_SETUP if index else None,
                                action="run the verification check to try it" if index else None)
            for index, name in enumerate(listed))

    answers[names[0]] = _report_with(operations)
    report = _native_report(RunRequest(("module.native-a",)), settings_for(tmp_path, providers))
    items = [item for item in report.items if ".operation." in item.key]
    assert len(items) > 1
    assert items[0].action is None
    assert {item.action for item in items[1:]} == {"run the verification check to try it"}
    assert all(item.status == "missing_setup" for item in items[1:])


@pytest.mark.parametrize("ceiling", [-1, 0])
def test_a_request_limit_that_allows_no_request_is_refused_with_its_fix(tmp_path, ceiling):
    """Zero or a negative limit would send no request. The settings refuse
    it before any report, naming the values to use instead."""
    from testuary.settings import RunSettings

    with pytest.raises(ValueError, match="max_requests_per_phase: expected a positive integer, or unlimited"):
        RunSettings(base_dir=tmp_path, environment={}, sections={
            "api": {"model": "selected"}, "coverage": {"max_requests_per_phase": ceiling}})


@pytest.mark.parametrize("model_fields,status", [
    ({"api_key": "key-fixture"}, "unverified"),
    ({}, "unverified"),
])
def test_the_model_key_item_says_where_the_key_goes(tmp_path, model_fields, status):
    from testuary.readiness import _configuration_report
    from testuary.settings import RunSettings

    settings = RunSettings(base_dir=tmp_path, environment={}, sections={
        "api": {"model": "selected"}, "project": {"source_dir": "original", "output_dir": "output"},
    }, models={"selected": {"provider": "openai", "model": "fixture",
                            "api_url": "https://example.test/v1", **model_fields}})
    report = _configuration_report(RunRequest(("module.py",)), settings)
    [item] = [item for item in report.items if item.key == "configuration.credentials"]
    assert item.status == status and item.action
    if not model_fields:
        assert "TESTUARY_OPENAI_API_KEY" in item.action and "[selected]" in item.action
        assert "service without API keys works" in item.reason
    assert "key-fixture" not in repr(report)


@pytest.mark.parametrize("explicit", [False, True])
def test_check_names_testuarys_own_environment_unless_it_was_selected(tmp_path, providers, explicit):
    import sys

    names, _, answers, _, _ = providers
    sources(tmp_path, "module.native-a")

    def own_runtime(request, scope):
        return ProjectSetupReport(
            scope, Outcome.COMPLETED, "metadata observed", request.context, request.inputs,
            (ProjectSetupItem("runtime", True, Availability.AVAILABLE, Outcome.COMPLETED,
                              selected=sys.executable, version="fixture version"),),
            tuple(OperationCapability(name, True) for name in request.required_operations))

    answers[names[0]] = own_runtime
    sections = {"project": {"source_dir": "original", "output_dir": "output",
                            **({"python_executable": sys.executable} if explicit else {})}}
    report = _native_report(RunRequest(("module.native-a",)), settings_for(tmp_path, providers,
                                                                          sections=sections))
    flagged = [item for item in report.items if item.key.endswith("runtime_selection")]
    if explicit:
        assert flagged == []
    else:
        [item] = flagged
        assert item.status == "missing_setup" and "python_executable" in item.action


def test_verification_lists_run_only_items_without_failing_on_them():
    from testuary.readiness import ReadinessItem, _readiness_summary

    run_only = ReadinessItem("native.python.operation.tests.run_suite", True, "unverified", tried_by="run")
    pending = ReadinessItem("native.python.item.pytest", True, "unverified", tried_by="verification")
    verified = ReadinessItem("native.python.item.runtime", True, "verified")
    assert _readiness_summary((verified, run_only), scope="verification") == ("verified", 0)
    assert _readiness_summary((verified, run_only, pending), scope="verification") == ("unverified", 1)
    assert _readiness_summary((verified, run_only), scope="inspection") == ("unverified", 0)
    with pytest.raises(ValueError, match="tried_by"):
        ReadinessItem("x", True, "unverified", tried_by="later")


def test_the_backend_marker_passes_through_to_operation_items(tmp_path, providers):
    names, _, answers, _, _ = providers
    sources(tmp_path, "module.native-a")

    def operations(request):
        return tuple(OperationCapability(name, True, reason="not tried", tried_by="run")
                     for name in request.required_operations)

    answers[names[0]] = _report_with(operations)
    report = _native_report(RunRequest(("module.native-a",)), settings_for(tmp_path, providers), verify=True)
    items = [item for item in report.items if ".operation." in item.key]
    assert items and {item.tried_by for item in items} == {"run"}


def _guest_settings(tmp_path, providers):
    return settings_for(tmp_path, providers, environment={
        "TESTUARY_PROJECT_CONTAINER": "fixture-image",
        "TESTUARY_PROJECT_WORK_DIR": str(tmp_path / "work")}, sections={
        "main": {"project_exec": "container"},
        "project": {"source_dir": "/guest/project", "output_dir": "output"}})


def _completed(request, scope):
    return ProjectSetupReport(
        scope, Outcome.COMPLETED, "controlled observation", request.context, request.inputs,
        (ProjectSetupItem("runtime", True, Availability.AVAILABLE, Outcome.COMPLETED,
                          selected="selected-runtime", version="fixture version"),),
        tuple(OperationCapability(name, True, readiness=Availability.AVAILABLE)
              for name in request.required_operations))


@pytest.mark.parametrize("route", ["container", "resume", "reuse"])
def test_completed_verification_passes_and_every_open_item_names_a_next_step(
        tmp_path, providers, monkeypatch, route):
    # Testuary's own output items in the resume, reuse and container routes
    # pass verification or name an action.
    import langadapter

    monkeypatch.setattr(langadapter, "verify_project", lambda language, request: _completed(request, "verification"))
    if route == "container":
        settings = _guest_settings(tmp_path, providers)
        request = RunRequest(("/guest/project/Module.native-a",))
    else:
        sources(tmp_path, "module.native-a")
        (tmp_path / "output").mkdir()
        (tmp_path / "output" / "saved.txt").write_text("saved work\n", encoding="utf-8")
        settings = settings_for(tmp_path, providers)
        request = RunRequest(("module.native-a",), resume=route == "resume",
                             reuse=("starter",) if route == "reuse" else ())
    report = _native_report(request, settings, verify=True)
    assert (report.status, report.exit_code) == ("verified", 0)
    items = {item.key: item for item in report.items}
    if route == "container":
        separation = items["output.original_separation"]
        assert not separation.required and separation.action
    else:
        saved = items["output.saved_artifacts"]
        assert saved.required and saved.tried_by == "run" and saved.action
    assert [key for key, item in items.items()
            if item.required and item.status != "verified" and not item.action] == []


@pytest.mark.parametrize("outcome", [Outcome.MISSING_SETUP, Outcome.UNSUPPORTED, Outcome.INFRASTRUCTURE_FAILURE])
def test_an_unfinished_native_assessment_names_a_next_step(tmp_path, providers, outcome):
    names, _, answers, _, _ = providers
    sources(tmp_path, "module.native-a")
    answers[names[0]] = lambda request, scope: ProjectSetupReport(
        scope, outcome, "controlled observation", request.context, request.inputs, (),
        tuple(OperationCapability(name, True) for name in request.required_operations))
    report = _native_report(RunRequest(("module.native-a",)), settings_for(tmp_path, providers))
    [assessment] = [item for item in report.items if item.key.endswith(".assessment")]
    assert assessment.status != "verified" and assessment.action
