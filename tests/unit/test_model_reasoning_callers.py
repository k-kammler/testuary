"""Reasoning dialects and extra options at actual phase request boundaries."""
import json

import pytest

import test_model_precedence_callers as caller_harness
from test_model_value_callers import assert_refused_before_work

streams = caller_harness.streams
transport = caller_harness.transport
PHASES = caller_harness.PHASES


def invoke(phase, fields, tmp_path, monkeypatch, request):
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    settings = caller_harness.settings_for(tmp_path, (), {}, models={"explicit": {
        "provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
        "api_key": "fixture", **fields,
    }})
    record = caller_harness.invoke_phase(phase, tmp_path, settings, monkeypatch)
    assert len(sent) == 1
    return sent[0][0], sent[0][1], record


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("endpoint", ["https://ollama.com/v1", "https://ollama.com.evil/v1",
                                    "http://localhost:11434/gateway/v1"])
def test_endpoint_protocol_default_reaches_each_actual_phase_request(
    phase, endpoint, tmp_path, monkeypatch, request,
):
    native = phase == "starter" and endpoint == "https://ollama.com/v1"
    url, body, record = invoke(phase, {"api_url": endpoint}, tmp_path, monkeypatch, request)
    expected_path = "/api/chat" if native else "/gateway/v1/chat/completions" if "gateway" in endpoint else "/v1/chat/completions"
    assert url.path == expected_path and body["model"] == "chosen"
    assert record["sent_settings"]["protocol"] == ("ollama-native" if native else "openai-chat")
    assert record["sent_settings"]["sources"]["protocol"] == "default:phase/endpoint"


def test_public_agent_native_refusal_precedes_initial_client_or_project_work(tmp_path, monkeypatch):
    assert_refused_before_work(tmp_path, monkeypatch, {"protocol": "ollama-native"}, {}, "only for the starter phase")


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("dialect,fields,expected", [
    ("deepseek", {"reasoning": "on"}, {"thinking": {"type": "enabled"}}),
    ("deepseek", {"reasoning": "off"}, {"thinking": {"type": "disabled"}}),
    ("vllm", {"reasoning_template": "qwen", "reasoning": "on", "reasoning_budget_tokens": 13},
     {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 13}),
    ("vllm", {"reasoning_template": "qwen", "reasoning": "off"},
     {"chat_template_kwargs": {"enable_thinking": False}}),
    ("vllm", {"reasoning_template": "deepseek", "reasoning": "on", "reasoning_effort": "high"},
     {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}}),
    ("openrouter", {"reasoning": "on", "reasoning_effort": "high"},
     {"reasoning": {"enabled": True, "effort": "high"}}),
    ("openrouter", {"reasoning": "off"}, {"reasoning": {"enabled": False}}),
    ("openrouter", {"reasoning_budget_tokens": 13}, {"reasoning": {"max_tokens": 13}}),
    ("openai", {"reasoning_effort": "high"}, {"reasoning_effort": "high"}),
])
def test_every_chat_dialect_reaches_all_actual_phase_requests(
    phase, dialect, fields, expected, tmp_path, monkeypatch, request,
):
    url, body, record = invoke(phase, {"reasoning_dialect": dialect, **fields}, tmp_path, monkeypatch, request)
    assert url.path == "/v1/chat/completions"
    assert {key: body[key] for key in expected} == expected
    assert record["sent_settings"]["reasoning_dialect"] == dialect
    assert record["sent_settings"]["protocol"] == "openai-chat"
    sent = record["sent_settings"]
    assert sent["reasoning_template"] == fields.get("reasoning_template")
    assert {key: sent["values"][key] for key in fields if key != "reasoning_template"} == {
        key: value for key, value in fields.items() if key != "reasoning_template"}


@pytest.mark.parametrize("phase", PHASES)
def test_responses_reasoning_effort_reaches_each_actual_phase_request(phase, tmp_path, monkeypatch, request):
    url, body, record = invoke(phase, {"protocol": "openai-responses", "reasoning_effort": "high"},
                               tmp_path, monkeypatch, request)
    assert url.path == "/v1/responses" and body["reasoning"] == {"effort": "high"}
    assert "reasoning_effort" not in body
    assert record["sent_settings"]["values"]["reasoning_effort"] == "high"


@pytest.mark.parametrize("fields,expected", [
    ({"reasoning": "on"}, True), ({"reasoning": "off"}, False), ({"reasoning_effort": "high"}, "high"),
])
def test_native_starter_reasoning_reaches_the_actual_ndjson_request(fields, expected, tmp_path, monkeypatch, request):
    url, body, record = invoke("starter", {"protocol": "ollama-native", **fields}, tmp_path, monkeypatch, request)
    assert url.path == "/api/chat" and body["think"] == expected
    assert record["sent_settings"]["protocol"] == "ollama-native"
    assert body["options"]["seed"] == 38


@pytest.mark.parametrize("phase", PHASES)
def test_request_options_merge_null_delete_and_value_types_reach_actual_requests(
    phase, tmp_path, monkeypatch, request,
):
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    settings = caller_harness.settings_for(tmp_path, (
        '[api]\nrequest_options={"top_k": null, "custom": {"b": 2}}\n',
    ), {"TESTUARY_API_REQUEST_OPTIONS": '{"min_p": 0}'}, models={
        "openai": {"api_url": "https://example.test/v1", "api_key": "fixture",
                   "request_options": {"top_k": 20, "custom": {"a": 1}}},
        "explicit": {"provider": "openai", "model": "chosen", "request_options": {"other": True}},
    }, api_overrides={"request_options": '{"extra_enabled": false}'})
    record = caller_harness.invoke_phase(phase, tmp_path, settings, monkeypatch)
    expected = {"custom": {"b": 2}, "other": True, "min_p": 0, "extra_enabled": False}
    body = sent[0][1]
    assert {key: body[key] for key in expected} == expected and "top_k" not in body
    assert record["sent_settings"]["request_options"] == expected
    assert record["sent_settings"]["sources"]["request_options"] == {
        "custom": "config[0]:[api]", "other": "alias:[explicit]", "min_p": "environment:[api]",
        "extra_enabled": "overrides:[api]",
    }


@pytest.mark.parametrize("fields,match", [
    ({"reasoning": "yes"}, "reasoning must be"),
    ({"reasoning": "on"}, "need a reasoning_dialect"),
    ({"reasoning_dialect": "unknown"}, "reasoning_dialect must be"),
    ({"reasoning": "off", "reasoning_effort": "high"}, "cannot be combined"),
    ({"reasoning": "off", "reasoning_budget_tokens": 13}, "cannot be combined"),
    ({"reasoning_dialect": "deepseek", "reasoning_effort": "high"}, "enablement only"),
    ({"reasoning_dialect": "vllm", "reasoning_template": "qwen", "reasoning_effort": "high"},
     "no reasoning_effort"),
    ({"reasoning_dialect": "vllm", "reasoning": "on"}, "needs reasoning_template"),
    # The DeepSeek V3.1 and V3.2-Exp templates read no effort.
    ({"reasoning_dialect": "vllm", "reasoning_template": "deepseek_v3", "reasoning_effort": "low"},
     "no reasoning_effort"),
    ({"reasoning_dialect": "vllm", "reasoning_template": "deepseek_v4_1", "reasoning_effort": "extreme"},
     "takes reasoning_effort"),
    ({"reasoning_dialect": "openai", "reasoning": "on"}, "effort only"),
    ({"reasoning_dialect": "openrouter", "reasoning_effort": "high", "reasoning_budget_tokens": 13},
     "cannot combine"),
    ({"protocol": "ollama-native", "reasoning_budget_tokens": 13}, "cannot express"),
    ({"protocol": "openai-responses", "reasoning": "on"}, "effort only"),
    ({"protocol": "openai-responses", "reasoning_budget_tokens": 13}, "effort only"),
    ({"chat_model": "fixture:factory", "reasoning_effort": "high"}, "use request_options"),
])
def test_public_preflight_refuses_unexpressible_reasoning_before_work(fields, match, tmp_path, monkeypatch):
    assert_refused_before_work(tmp_path, monkeypatch, fields, {}, match)


@pytest.mark.parametrize("fields,option", [
    ({"protocol": "openai-chat"}, "messages"),
    ({"protocol": "openai-chat"}, "tools"),
    ({"protocol": "openai-responses"}, "max_output_tokens"),
    ({"protocol": "ollama-native"}, "options"),
    ({"chat_model": "fixture:factory"}, "max_retries"),
    ({"reasoning_dialect": "deepseek"}, "thinking"),
    ({"reasoning_dialect": "vllm"}, "chat_template_kwargs"),
    ({"reasoning_dialect": "vllm"}, "thinking_token_budget"),
    ({"reasoning_dialect": "openrouter"}, "reasoning"),
    ({}, "authorization"),
    ({}, "vendor"),
])
def test_public_preflight_refuses_reserved_and_nested_credential_options_before_work(
    fields, option, tmp_path, monkeypatch,
):
    value = {"Authorization": "private-fixture"} if option == "vendor" else None
    fields = {**fields, "request_options": {option: value}}
    match = "credential" if option in {"authorization", "vendor"} else "controlled field"
    assert_refused_before_work(tmp_path, monkeypatch, fields, {}, match)
    assert "private-fixture" not in json.dumps({path.name: path.read_text("utf-8")
                                               for path in tmp_path.rglob("*") if path.is_file()})
