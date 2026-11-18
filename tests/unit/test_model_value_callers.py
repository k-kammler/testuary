"""Empty, zero and false model controls at request and public-run boundaries."""
import json

import pytest

import test_model_precedence_callers as caller_harness
from testuary import __main__ as cli, orchestrator, settings as settings_module
from testuary import models as _model_protocols
from testuary import config
from testuary.settings import ConfigurationError

streams = caller_harness.streams
transport = caller_harness.transport
PHASES = caller_harness.PHASES


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("alias,fields,absent,expected", [
    ({"temperature": .6, "top_p": .9}, {"temperature": "", "top_p": ""},
     ("temperature", "top_p"), {}),
    ({}, {"temperature": 0, "top_p": 0, "seed": 0}, (), {"temperature": 0, "top_p": 0}),
    ({"max_output_tokens": 0}, {"max_output_tokens": 0},
     ("max_tokens", "max_completion_tokens", "max_output_tokens"), {}),
    ({"max_output_tokens": ""}, {"max_output_tokens": ""},
     ("max_tokens", "max_completion_tokens", "max_output_tokens"), {}),
    ({"max_output_tokens": 100}, {"max_output_tokens": 0}, (), {"max_completion_tokens": 100}),
    ({"reasoning_dialect": "deepseek", "reasoning": "on"}, {"reasoning": ""}, ("thinking",), {}),
    ({"reasoning_dialect": "openai", "reasoning_effort": "high"}, {"reasoning_effort": ""},
     ("reasoning_effort", "reasoning"), {}),
    ({"reasoning_dialect": "vllm", "reasoning_budget_tokens": 13}, {"reasoning_budget_tokens": ""},
     ("thinking_token_budget",), {}),
    ({"protocol": "openai-responses"}, {"protocol": ""}, (), {}),
    ({"request_options": {"top_k": 20}}, {"request_options": ""}, ("top_k",), {}),
    ({"request_options": {"top_k": 20}}, {"request_options": {}}, (), {"top_k": 20}),
    ({}, {"request_options": {"top_k": 0, "custom_enabled": False}}, (),
     {"top_k": 0, "custom_enabled": False}),
], ids=["empty-sampling", "real-zero-sampling-and-seed", "zero-both-caps", "empty-both-caps",
        "zero-phase-cap-keeps-model-cap", "empty-reasoning", "empty-effort", "empty-budget",
        "empty-protocol-defaults", "empty-options-string-clears", "empty-options-object-contributes-no-keys",
        "json-option-zero-and-false"])
def test_value_contracts_reach_all_actual_phase_requests(
    phase, alias, fields, absent, expected, tmp_path, monkeypatch, request,
):
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    models = {"explicit": {"provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
                           "api_key": "fixture", **alias}}
    api = {key: json.dumps(value) if isinstance(value, dict) else value for key, value in fields.items()}
    settings = caller_harness.settings_for(tmp_path, (), {}, models=models, api_overrides=api)
    record = caller_harness.invoke_phase(phase, tmp_path, settings, monkeypatch)
    assert len(sent) == 1 and sent[0][0].path == "/v1/chat/completions"
    body = sent[0][1]
    assert body["model"] == "chosen"
    assert record["sent_settings"]["protocol"] == "openai-chat"
    assert not set(absent).intersection(body)
    assert {key: body[key] for key in expected} == expected
    if "seed" in fields:
        assert body["seed"] == (38 if phase == "starter" else 0)
        assert record["sent_settings"]["values"]["seed"] == body["seed"]
    for key in ("temperature", "top_p", "reasoning", "reasoning_effort", "reasoning_budget_tokens"):
        if fields.get(key) == "":
            assert record["sent_settings"]["values"][key] is None
    if "max_output_tokens" in fields:
        assert record["sent_settings"]["values"]["phase_max_output_tokens"] is None
        assert record["sent_settings"]["values"]["max_output_tokens"] == (
            100 if alias["max_output_tokens"] == 100 else None)


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("key", ("reasoning_dialect", "chat_model", "api_key_env"))
def test_empty_alias_service_fields_clear_provider_values_without_adding_wire_fields(
    phase, key, tmp_path, monkeypatch, request,
):
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    lower = {"reasoning_dialect": "vllm", "chat_model": "unavailable_fixture:factory",
             "api_key_env": "UNSET_FIXTURE_KEY"}[key]
    settings = caller_harness.settings_for(tmp_path, (), {}, models={
        "openai": {"api_url": "https://example.test/v1", "api_key": "fixture", key: lower},
        "explicit": {"provider": "openai", "model": "chosen", key: ""},
    })
    record = caller_harness.invoke_phase(phase, tmp_path, settings, monkeypatch)
    assert len(sent) == 1
    assert key not in sent[0][1]
    if key != "api_key_env":
        assert record["sent_settings"][key] is None
    else:
        assert "api_key_env" not in record["sent_settings"]


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("field,alias,absent", [
    ("temperature", {"temperature": .6}, ("temperature",)),
    ("top_p", {"top_p": .9}, ("top_p",)),
    ("seed", {}, ()),
    ("max_output_tokens", {"max_output_tokens": 100}, ()),
    ("reasoning", {"reasoning_dialect": "deepseek", "reasoning": "on"}, ("thinking",)),
    ("reasoning_effort", {"reasoning_dialect": "openai", "reasoning_effort": "high"}, ("reasoning_effort",)),
    ("reasoning_budget_tokens", {"reasoning_dialect": "vllm", "reasoning_budget_tokens": 13},
     ("thinking_token_budget",)),
    ("protocol", {"protocol": "openai-responses"}, ()),
    ("request_options", {"request_options": {"top_k": 20}}, ("top_k",)),
])
def test_empty_environment_values_are_overrides_at_actual_requests(
    phase, field, alias, absent, tmp_path, monkeypatch, request,
):
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    settings = caller_harness.settings_for(
        tmp_path, ("[api]\nseed=12\nmax_output_tokens=50\n",),
        {f"TESTUARY_API_{field.upper()}": ""}, models={"explicit": {
            "provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
            "api_key": "fixture", **alias,
        }},
    )
    record = caller_harness.invoke_phase(phase, tmp_path, settings, monkeypatch)
    body = sent[0][1]
    assert sent[0][0].path == "/v1/chat/completions"
    assert not set(absent).intersection(body)
    if field == "seed":
        if phase == "starter":
            assert body["seed"] == 38
        else:
            assert "seed" not in body and record["sent_settings"]["values"]["seed"] is None
    elif field == "max_output_tokens":
        assert body["max_completion_tokens"] == 100
        assert record["sent_settings"]["values"]["phase_max_output_tokens"] is None
    elif field == "protocol":
        assert record["sent_settings"]["protocol"] == "openai-chat"
        assert record["sent_settings"]["sources"]["protocol"] == "default:phase/endpoint"
    elif field == "request_options":
        assert record["sent_settings"]["request_options"] == {}
    else:
        assert record["sent_settings"]["values"][field] is None
    source_key = "phase_max_output_tokens" if field == "max_output_tokens" else field
    if field not in {"seed", "request_options", "protocol"}:
        assert record["sent_settings"]["sources"][source_key] == "environment:[api]"


@pytest.mark.parametrize("field", [
    "temperature", "top_p", "seed", "max_output_tokens", "request_timeout_seconds",
    "reasoning", "reasoning_effort", "reasoning_budget_tokens", "protocol", "request_options",
    "reasoning_dialect", "chat_model", "api_key_env",
])
def test_public_model_preflight_refuses_false_outside_boolean_contracts_before_work(
    field, tmp_path, monkeypatch,
):
    alias = {}
    api = {}
    (alias if field in {"reasoning_dialect", "chat_model", "api_key_env"} else api)[field] = False
    assert_refused_before_work(tmp_path, monkeypatch, alias, api, field)


@pytest.mark.parametrize("field", [
    "request_timeout_seconds", "reasoning", "reasoning_effort", "reasoning_budget_tokens", "protocol",
    "request_options", "reasoning_dialect", "chat_model", "api_key_env",
])
def test_public_model_preflight_refuses_zero_outside_zero_contracts_before_work(field, tmp_path, monkeypatch):
    alias = {}
    api = {}
    (alias if field in {"reasoning_dialect", "chat_model", "api_key_env"} else api)[field] = 0
    assert_refused_before_work(tmp_path, monkeypatch, alias, api, field)


def test_public_model_preflight_refuses_empty_request_timeout_before_work(tmp_path, monkeypatch):
    assert_refused_before_work(tmp_path, monkeypatch, {}, {"request_timeout_seconds": ""}, "request_timeout_seconds")


def test_public_model_preflight_refuses_empty_environment_timeout_before_work(tmp_path, monkeypatch):
    assert_refused_before_work(tmp_path, monkeypatch, {}, {}, "request_timeout_seconds",
                               environment={"TESTUARY_API_REQUEST_TIMEOUT_SECONDS": ""})


def assert_refused_before_work(tmp_path, monkeypatch, alias, api, field, *, environment=None):
    previous = tmp_path / "previous.txt"
    previous.write_text("preserve", encoding="utf-8")
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(config, "get_project_config", lambda: pytest.fail("project accessed"))
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kw: pytest.fail("cleanup reached"))
    monkeypatch.setattr(_model_protocols, "build_resolved_chat_model", lambda *a, **kw: pytest.fail("client built"))
    resolution_calls = []

    def resolve_for_public_run(*, config_files, overrides, home, project_settings, load_extensions):
        assert config_files is None and not overrides and home is None and project_settings
        assert load_extensions
        resolution_calls.append(True)
        return caller_harness.settings_for(tmp_path, (), environment or {}, models={"explicit": {
            "provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
            "api_key": "fixture", **alias,
        }}, api_overrides=api)

    monkeypatch.setattr(settings_module, "resolve_settings", resolve_for_public_run)
    with pytest.raises(ConfigurationError, match=field):
        orchestrator.run_pipeline(cli._build_parser().parse_args([]))
    assert resolution_calls == [True]
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert config.active_settings() is None


@pytest.mark.parametrize("phase", PHASES)
def test_informational_prices_are_recorded_not_sent_at_each_actual_boundary(phase, tmp_path, monkeypatch, request):
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    prices = {"price_input_cache_hit": ".1", "price_input_cache_miss": ".2", "price_output": ".3"}
    settings = caller_harness.settings_for(tmp_path, (), {}, models={"explicit": {
        "provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
        "api_key": "fixture", **prices,
    }})
    record = caller_harness.invoke_phase(phase, tmp_path, settings, monkeypatch)
    assert record["sent_settings"]["information"] == prices
    assert not set(prices).intersection(sent[0][1])
    assert "fixture" not in json.dumps(record["sent_settings"])
