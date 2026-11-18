"""Configuration observations are not native or provider readiness claims."""
from dataclasses import replace
import json
import subprocess
import sys

import pytest

from test_model_preflight import added_provider  # noqa: F401 - shared controlled factory
from testuary import orchestrator
from testuary import models as _model_protocols
from testuary import config
from testuary import project as project_module
from testuary.orchestrator import RunRequest, _invocation
from testuary.readiness import _configuration_error_report, _configuration_report
from testuary.settings import ConfigurationError, RunSettings


def inputs(tmp_path, *, sections=None, fields=None, environment=None):
    return RunSettings(base_dir=tmp_path, sections={
        "api": {"model": "selected"}, "project": {"source_dir": "source not probed", "output_dir": "outside"},
        **(sections or {}),
    }, models={"selected": {"provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
                            **(fields or {})}}, environment=environment or {})


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native process"))
    monkeypatch.setattr(project_module, "project_runtime", lambda *a, **k: pytest.fail("runtime probe"))
    monkeypatch.setattr(project_module, "resolve_project_source_path", lambda *a: pytest.fail("source inspection"))
    monkeypatch.setattr(orchestrator, "_admit_output", lambda *a, **k: pytest.fail("output admission"))
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)


@pytest.mark.parametrize("reuse,stop,expected", [
    ((), None, ("starter", "coverage", "mutation")),
    ((), "starter", ("starter",)), ((), "coverage", ("starter", "coverage")),
    ((), "mutation", ("starter", "coverage", "mutation")),
    ((), "finishing", ("starter", "coverage", "mutation")),
    (("starter", "coverage", "mutation", "finishing"), None, ()), (("starter",), None, ("coverage", "mutation")),
    (("coverage",), None, ("mutation",)), (("mutation",), None, ()),
    (("finishing",), None, ("starter", "coverage", "mutation")),
    (("starter", "coverage", "mutation"), None, ()),
])
def test_possible_model_phases_follow_the_actual_existing_preflight(reuse, stop, expected, tmp_path, monkeypatch):
    settings = inputs(tmp_path)
    request = RunRequest(("module.py",), reuse=reuse, stop_after_phase=stop)
    args = _invocation(request)
    monkeypatch.setattr(_model_protocols, "build_resolved_chat_model", lambda *a, **k: pytest.fail("client construction"))
    models, problems, credential = orchestrator._resolve_generation_models(args, settings)
    assert tuple(models) == expected and problems == ()
    assert credential == ("" if expected else None)
    assert not hasattr(args, "_prepared_models")
    with config.use_run_settings(settings):
        assert orchestrator._preflight_model_configuration(args) == problems
        assert args._prepared_models is None
    assert not list(tmp_path.iterdir()) and config.active_settings() is None


@pytest.mark.parametrize("provider", ["openai", "ollama", "vllm", "custom"])
def test_private_resolution_uses_captured_credentials_not_changed_environment(provider, tmp_path, monkeypatch):
    name = f"TESTUARY_{provider.upper()}_API_KEY"
    settings = inputs(tmp_path, fields={"provider": provider}, environment={name: "captured"})
    monkeypatch.setenv(name, "later")
    models, problems, credential = orchestrator._resolve_generation_models(_invocation(RunRequest(("a.py",))), settings)
    assert problems == () and credential == "captured"
    assert {model.identity.provider for model in models.values()} == {provider}
    assert config.active_settings() is None


def test_added_factory_is_imported_but_not_called_by_configuration_and_retained_by_execution(
        tmp_path, added_provider):  # noqa: F811
    settings = inputs(tmp_path, fields={"protocol": "added", "chat_model": "_fixture_preflight_provider:factory"})
    request = RunRequest(("module.py",), stop_after_phase="coverage")
    report = _configuration_report(request, settings)
    assert added_provider.models == []
    assert all("factory imports" in item.reason
               for item in report.items if item.key.startswith("configuration.model."))
    assert report.status == "unverified" and report.exit_code == 0
    assert all(item.selected == "added" for item in report.items if item.key.startswith("configuration.model."))
    with config.use_run_settings(settings):
        args = _invocation(request)
        assert orchestrator._preflight_model_configuration(args) == ()
        assert args._prepared_models is not None
        assert len(added_provider.models) == 11  # Ten slots, plus the editing client.
        assert all(model.calls == 0 for model in added_provider.models)
    assert config.active_settings() is None


@pytest.mark.parametrize("fields,expected", [
    ({"protocol": "added", "chat_model": "missing:factory", "request_options": {"timeout": 1}}, "controlled field"),
    ({"protocol": "ollama-native"}, "supported only for the starter phase"),
    ({"api_url": "relative"}, "absolute HTTP"),
    ({"api_key_env": "UNSET"}, "UNSET, which is not set"),
])
def test_configuration_errors_agree_with_execution_before_any_factory(fields, expected, tmp_path, monkeypatch):
    settings = inputs(tmp_path, fields=fields)
    args = _invocation(RunRequest(("module.py",)))
    monkeypatch.setattr(_model_protocols, "build_resolved_chat_model", lambda *a, **k: pytest.fail("client constructed"))
    _, problems, _ = orchestrator._resolve_generation_models(args, settings)
    assert any(expected in problem for problem in problems)
    with config.use_run_settings(settings):
        assert orchestrator._preflight_model_configuration(args) == problems
    with pytest.raises(ConfigurationError, match=expected):
        _configuration_report(RunRequest(("module.py",)), settings)
    assert config.active_settings() is None and not list(tmp_path.iterdir())


def test_report_captures_paths_policy_and_limits_without_certifying_them(tmp_path, monkeypatch):
    settings = inputs(tmp_path, sections={"api": {"model": "selected"},
        "starter": {"attempts_per_suite": 2},
        "coverage": {"max_requests_per_phase": 12, "max_input_tokens_per_phase": 100, "max_wall_clock_seconds_per_phase": 2.5,
                          "max_output_tokens": 32},
        "mutation": {"max_output_tokens": 16, "max_requests_per_phase": "unlimited"},
    }, fields={"max_output_tokens": 64})
    request = RunRequest(("a,b.py", "second.py"), targets="2", runs=3, resume=True)
    monkeypatch.chdir(tmp_path.parent)
    report = _configuration_report(request, settings)
    assert report.scope == "inspection" and report.status == "unverified" and report.exit_code == 0
    assert report.source_files == ("second.py",) and report.phases == ("starter", "coverage", "mutation", "finishing")
    assert report.source_dir == str(tmp_path / "source not probed")
    assert report.output_dir == str(tmp_path / "outside") and report.output_mode == "resume"
    assert report.credentials_configured is False
    limits = {(limit.phase, limit.name): limit for limit in report.limits}
    assert [limits[(p, "max_output_tokens")].value for p in ("starter", "coverage", "mutation")] == [64, 32, 16]
    assert limits[("starter", "attempts_per_suite")].value == 2
    assert ("starter", "max_requests_per_phase") not in limits
    assert limits[("coverage", "max_input_tokens_per_phase")].value == 100
    assert limits[("coverage", "max_wall_clock_seconds_per_phase")].value == 2.5
    assert limits[("mutation", "max_requests_per_phase")].value is None
    assert limits[("mutation", "max_requests_per_phase")].state == "unlimited"
    assert limits[("mutation", "max_input_tokens_per_phase")].value is None
    assert limits[("mutation", "max_input_tokens_per_phase")].state == "unlimited"
    assert limits[("setup", "setup_timeout_seconds")].value == 120
    credentials = next(item for item in report.items if item.key == "configuration.credentials")
    assert (credentials.status, credentials.action.startswith("If the service needs an API key")) == ("unverified", True)
    assert all(item.status == "unverified" for item in report.items
               if not item.key.startswith("configuration.model.") and item is not credentials)
    assert all(item.diagnostic_path is None for item in report.items)
    assert not list(tmp_path.iterdir()) and config.active_settings() is None


@pytest.mark.parametrize("credential,configured", [(None, False), ("", False), ("   ", False), ("private", True)])
def test_credentials_are_only_configuration_facts(credential, configured, tmp_path):
    settings = inputs(tmp_path, environment={} if credential is None else {"TESTUARY_OPENAI_API_KEY": credential})
    report = _configuration_report(RunRequest(("module.py",)), settings)
    assert report.credentials_configured is configured
    assert "authenticated" not in str(report.as_dict())
    assert all(item.key != "authentication" for item in report.items)


@pytest.mark.parametrize("reuse,resume,reuse_starter,mode", [
    ((), False, False, "fresh"), ((), True, False, "resume"),
    (("starter",), False, False, "reuse"), (("finishing",), False, False, "fresh"),
    ((), False, True, "reuse"), (("starter", "coverage", "mutation", "finishing"), False, False, "reuse"),
])
def test_output_modes_reuse_the_actual_preservation_policy(reuse, resume, reuse_starter, mode, tmp_path):
    settings = inputs(tmp_path, sections={"api": {"model": "selected"}, "starter": {"reuse_suites": reuse_starter}})
    report = _configuration_report(RunRequest(("module.py",), reuse=reuse, resume=resume), settings)
    assert report.output_mode == mode
    if reuse == ("starter", "coverage", "mutation", "finishing"):
        assert report.model is report.provider is report.model_alias is report.credentials_configured is None


def test_report_and_failure_exports_remove_secrets_without_inventing_paths(tmp_path):
    secret = "synthetic-private-marker"
    settings = inputs(tmp_path, fields={"api_key": secret, "model": secret,
        "api_url": f"https://user:{secret}@example.test/v1?token={secret}",
    })
    report = _configuration_report(RunRequest((f"{secret}.py",)), settings)
    assert secret not in repr(report) and secret not in json.dumps(report.as_dict())
    assert report.model == "[redacted]" and report.source_files == ("[redacted].py",)
    error = _configuration_error_report(ConfigurationError(f"actual failure {secret}"), settings=settings)
    assert error.status == "refused" and error.exit_code == 2
    assert [item.status for item in error.items] == ["refused"]
    assert error.source_dir is error.output_dir is error.credentials_configured is None
    assert not error.source_files and not error.phases and not error.limits
    assert error.items[0].reason == "actual failure [redacted]" and error.items[0].diagnostic_path is None


@pytest.mark.parametrize("ceiling", ["unlimited", 1])
def test_request_limit_uses_the_actual_tracker_semantics(ceiling, tmp_path):
    from testuary.agent._langgraph import UsageLimitExceeded
    from testuary.agent.coverage_agent import CoverageAgent

    settings = inputs(tmp_path, sections={"api": {"model": "selected"}, "coverage": {"max_requests_per_phase": ceiling}})
    report = _configuration_report(RunRequest(("module.py",), stop_after_phase="coverage"), settings)
    limit = next(item for item in report.limits if item.phase == "coverage" and item.name == "max_requests_per_phase")
    assert (limit.value, limit.state) == ((None, "unlimited") if ceiling == "unlimited" else (1, "configured"))
    with config.use_run_settings(settings):
        agent = CoverageAgent(prompts={}, deps=None, session=None)
        agent.log = lambda *a, **k: None
        tracker = agent.make_usage_tracker()
        tracker.note_request()
        assert tracker.logical_requests_used == 1
        if ceiling == 1:
            with pytest.raises(UsageLimitExceeded):
                tracker.note_request()
        else:
            for _ in range(5):
                tracker.note_request()


@pytest.mark.parametrize("scope", ["inspection", "verification"])
def test_configuration_refusal_retains_the_requested_scope_without_native_success(scope):
    report = _configuration_error_report(ConfigurationError("invalid selection"), scope=scope)
    assert report.scope == scope and report.status == "refused" and report.exit_code == 2
    assert report.execution is report.source_dir is report.model is report.output_dir is None
    assert report.items[0].diagnostic_path is None


def test_interrupted_resolution_restores_settings_and_does_not_fabricate_report(tmp_path, monkeypatch):
    settings = inputs(tmp_path)
    from testuary import models as _model_configuration

    monkeypatch.setattr(_model_configuration, "resolve_phase_model", lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))
    with pytest.raises(KeyboardInterrupt):
        _configuration_report(RunRequest(("module.py",)), settings)
    assert config.active_settings() is None
    with config.use_run_settings(replace(settings, base_dir=tmp_path.parent)):
        assert config.active_settings() is not None


def test_configuration_report_prints_nothing(tmp_path, capsys):
    settings = inputs(tmp_path)
    with config.use_run_settings(settings):
        config.get_project_config()
        project_module.output_dir()
    _configuration_report(RunRequest(("module.py",)), settings)
    assert capsys.readouterr().out == "" and config.active_settings() is None


@pytest.mark.parametrize("project", [
    {}, {"build_java_home": "", "maven": None, "gradle": "  "},
    {"build_java_home": "/guest/build jdk", "maven": "/guest/mvn", "gradle": " /guest/gradle "},
])
def test_java_tool_variables_use_only_the_supplied_project(project, tmp_path, monkeypatch, capsys):
    from types import MappingProxyType

    from langadapter.java.adapter import JavaAdapter

    from testuary.languages import language_setting_values

    captured = MappingProxyType(dict(project))
    expected = {
        variable: str(project[key]).strip()
        for key, variable in (("build_java_home", "LANGADAPTER_BUILD_JAVA_HOME"),
                              ("maven", "LANGADAPTER_MAVEN"), ("gradle", "LANGADAPTER_GRADLE"))
        if project.get(key) and str(project[key]).strip()
    }
    monkeypatch.setattr(config, "get_project_config", lambda **kwargs: pytest.fail("project resolved again"))
    for variable in ("LANGADAPTER_BUILD_JAVA_HOME", "LANGADAPTER_MAVEN", "LANGADAPTER_GRADLE"):
        monkeypatch.setenv(variable, "changed ambient selection")
    settings = inputs(tmp_path)
    with config.use_run_settings(settings):
        first = JavaAdapter.context_environment(language_setting_values("java", captured, settings))
        first["LANGADAPTER_MAVEN"] = "caller changed its detached view"
        assert JavaAdapter.context_environment(
            language_setting_values("java", captured, settings)) == expected
    assert dict(captured) == project and capsys.readouterr().out == ""
    assert config.active_settings() is None and not list(tmp_path.iterdir())


@pytest.mark.parametrize("error_type", [OSError, RuntimeError, KeyboardInterrupt, SystemExit])
def test_a_java_context_preserves_a_configuration_failure(error_type, monkeypatch, default_run):
    from testuary.languages import language_project_context

    error = error_type("project resolution failed")
    calls = []

    def fail():
        calls.append(True)
        raise error

    monkeypatch.setattr(config, "get_project_config", fail)
    with pytest.raises(error_type) as raised:
        language_project_context("java")
    assert raised.value is error and calls == [True]


def _check_refusal(tmp_path, chat_model):
    settings = inputs(tmp_path, fields={"protocol": "added", "chat_model": chat_model})
    with pytest.raises(ConfigurationError) as refused:
        _configuration_report(RunRequest(("module.py",)), settings)
    return str(refused.value)


def test_check_reports_a_chat_model_module_that_is_not_installed(tmp_path):
    message = _check_refusal(tmp_path, "no_such_module_for_check:factory")
    assert ("starter, coverage, mutation: chat_model no_such_module_for_check:factory: "
            "no_such_module_for_check: no module no_such_module_for_check is found in this "
            "environment") in message


def test_check_reports_an_import_error_inside_the_chat_model_module(tmp_path, monkeypatch):
    (tmp_path / "broken_provider_for_check.py").write_bytes(b"import no_such_dependency_for_check\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "broken_provider_for_check", raising=False)
    message = _check_refusal(tmp_path, "broken_provider_for_check:factory")
    assert ("chat_model broken_provider_for_check:factory: broken_provider_for_check: "
            "ModuleNotFoundError: No module named 'no_such_dependency_for_check'") in message


def test_check_reports_a_missing_factory_with_the_module_path(tmp_path, monkeypatch):
    (tmp_path / "factoryless_provider_for_check.py").write_bytes(b"VALUE = 1\n")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "factoryless_provider_for_check", raising=False)
    message = _check_refusal(tmp_path, "factoryless_provider_for_check:factory")
    assert "has no attribute 'factory'" in message
    assert "factoryless_provider_for_check.py" in message
