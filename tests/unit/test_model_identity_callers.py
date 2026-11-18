"""Model and endpoint selection at actual request and public-run boundaries."""
import json

import httpx
import pytest

import test_model_precedence_callers as caller_harness
from testuary import __main__ as cli, orchestrator
from testuary import models as _model_protocols
from testuary import config
from testuary.settings import ConfigurationError, resolve_settings

streams = caller_harness.streams
transport = caller_harness.transport
PHASES = caller_harness.PHASES


def configured(tmp_path, models, *, text="[api]\nmodel=explicit\n", environment=None, options=()):
    path = tmp_path / "identity.ini"
    path.write_text(text, encoding="utf-8")
    args = cli._build_parser().parse_args(list(options))
    return resolve_settings(
        base_dir=tmp_path, config_files=[path], models=models,
        environment={} if environment is None else environment,
        overrides={
            "main": {"hints_file": "", "seed": 37},
            "starter": {"suite_count": 1, "attempts_per_suite": 1,
                             "concurrent_requests": 1, "suite_run_timeout_seconds": 2},
            "coverage": {"max_requests_per_phase": 1, "max_loop_attempts": 0},
            "mutation": {"max_requests_per_phase": 1, "max_loop_attempts": 0},
            **orchestrator._cli_settings_overrides(args),
        },
    )


def invoke(phase, settings, tmp_path, monkeypatch, request):
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    record = caller_harness.invoke_phase(phase, tmp_path, settings, monkeypatch)
    assert len(sent) == 1
    return sent[0][:2], record["sent_settings"]


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("selected,alias,provider_fields,expected_model", [
    ("ollama_local", {"provider": "openai", "model": "chosen"}, "", "chosen"),
    ("openai_example", {"provider": "openai", "model": "chosen"}, "", "chosen"),
    ("openai_example", {"provider": "openai"}, "", "openai_example"),
    ("custom:model", None, "provider=openai\n", "custom:model"),
])
def test_provider_and_model_identity_reach_all_real_callers(
    phase, selected, alias, provider_fields, expected_model, tmp_path, monkeypatch, request,
):
    headers = []
    real_transport = httpx.MockTransport

    def capture(handle):
        def watched(request):
            headers.append(request.headers.get("Authorization"))
            return handle(request)
        return real_transport(watched)

    monkeypatch.setattr(httpx, "MockTransport", capture)
    models = {
        "openai": {"api_url": "https://openai.example/v1", "api_key": "openai-fixture"},
        "ollama": {"api_url": "https://wrong.example/v1", "api_key": "wrong-fixture"},
    }
    if alias is not None:
        models[selected] = alias
    settings = configured(tmp_path, models, text=f"[api]\nmodel={selected}\n{provider_fields}")
    (url, body), record = invoke(phase, settings, tmp_path, monkeypatch, request)
    assert str(url) == "https://openai.example/v1/chat/completions"
    assert body["model"] == expected_model
    assert headers == ["Bearer openai-fixture"]
    assert record["provider"] == "openai"
    assert record["model_alias"] == selected
    assert record["model"] == expected_model
    assert record["api_url"] == "https://openai.example/v1"
    assert "fixture" not in json.dumps(record)


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("environment,options,selected", [
    ({}, (), "explicit"),
    ({"TESTUARY_API_MODEL": "environment_model"}, (), "environment_model"),
    ({"TESTUARY_API_MODEL": "environment_model"}, ("--model", "cli_model"), "cli_model"),
])
def test_file_environment_and_cli_model_order_at_all_real_callers(
    phase, environment, options, selected, tmp_path, monkeypatch, request,
):
    models = {"openai": {"api_url": "https://example.test/v1", "api_key": "fixture"}}
    models.update({name: {"provider": "openai", "model": f"model/{name}"}
                   for name in ("explicit", "environment_model", "cli_model")})
    settings = configured(tmp_path, models, environment=environment, options=options)
    (_, body), record = invoke(phase, settings, tmp_path, monkeypatch, request)
    assert body["model"] == record["model"] == f"model/{selected}"
    assert record["model_alias"] == selected


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("catalogue_endpoint,alias_endpoint,options,expected", [
    (None, None, (), "https://configuration.example/v1"),
    ("https://catalogue.example/v1", None, (), "https://catalogue.example/v1"),
    ("https://catalogue.example/v1", "https://alias.example/v1", (), "https://alias.example/v1"),
    ("https://catalogue.example/v1", "https://alias.example/v1", ("--proxy-port", "8123"),
     "http://localhost:8123/v1"),
    ("https://catalogue.example/v1", "https://alias.example/v1", ("--api-url", "https://cli.example/v1"),
     "https://cli.example/v1"),
    ("https://catalogue.example/v1", "https://alias.example/v1",
     ("--proxy-port", "8123", "--api-url", "https://cli.example/v1"), "https://cli.example/v1"),
])
def test_endpoint_order_is_preserved_by_all_real_callers(
    phase, catalogue_endpoint, alias_endpoint, options, expected, tmp_path, monkeypatch, request,
):
    models = {"openai": {"api_key": "fixture"}, "explicit": {"provider": "openai", "model": "chosen"}}
    if catalogue_endpoint is not None:
        models["openai"]["api_url"] = catalogue_endpoint
    if alias_endpoint is not None:
        models["explicit"]["api_url"] = alias_endpoint
    settings = configured(tmp_path, models, options=options,
                          text="[api]\nmodel=explicit\n[openai]\napi_url=https://configuration.example/v1\n")
    (url, body), record = invoke(phase, settings, tmp_path, monkeypatch, request)
    assert str(url) == expected + "/chat/completions"
    assert body["model"] == "chosen"
    assert record["provider"] == "openai" and record["api_url"] == expected


@pytest.mark.parametrize("selected,alias,options,message", [
    ("gpt-4o", None, (), "is not an alias in"),
    ("openai_not_in_catalogue", None, (), "is not an alias in"),
    ("example", {"model": "chosen"}, (), "needs provider"),
    ("openai_example", {"provider": ""}, (), "provider"),
    ("openai_example", {"provider": 1}, (), "provider"),
    ("openai_example", {"provider": False}, (), "provider"),
    ("openai_example", {"provider": None}, (), "provider"),
    ("openai_example", {"provider": "openai"}, ("--api-url", ""), "absolute HTTP"),
    ("openai_example", {"provider": "openai"}, ("--api-url", "file:///private-sentinel"), "absolute HTTP"),
])
def test_public_run_refuses_invalid_identity_before_project_or_request_work(
    selected, alias, options, message, tmp_path, monkeypatch,
):
    models = {"openai": {"api_url": "https://example.test/v1", "api_key": "fixture"}}
    if alias is not None:
        models[selected] = alias
    settings = configured(tmp_path, models, text=f"[api]\nmodel={selected}\n", options=options)
    previous = tmp_path / "results" / "previous.txt"
    previous.parent.mkdir()
    previous.write_text("preserve", encoding="utf-8")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(config, "get_project_config", lambda: pytest.fail("project configuration reached"))
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kw: pytest.fail("cleanup reached"))
    monkeypatch.setattr(_model_protocols, "build_resolved_chat_model", lambda *a, **kw: pytest.fail("client built"))
    with pytest.raises(ConfigurationError, match=message) as caught:
        orchestrator.run_pipeline(cli._build_parser().parse_args([]), settings=settings)
    assert "private-sentinel" not in str(caught.value)
    after = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    assert after == before
    assert config.active_settings() is None
