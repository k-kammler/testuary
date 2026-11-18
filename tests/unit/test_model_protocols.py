"""Model request construction per protocol: native endpoints, the real SDK
clients and added chat-model factories."""
import asyncio
from dataclasses import replace
import json
from types import SimpleNamespace
import traceback

import httpx
from langchain_core.language_models import BaseChatModel
from langchain_core.language_models.fake_chat_models import FakeListChatModel
import pytest

from testuary import _chat_models
from testuary.models import resolve_phase_model
from testuary.models import build_resolved_chat_model, native_endpoint, native_request_body
from testuary.settings import ConfigurationError, RunSettings


def resolved(phase="starter", **fields):
    settings = RunSettings(
        sections={"api": {"model": "openai_example"}},
        models={"openai_example": {"provider": "openai", "model": "chosen", "api_url": "https://example.test/v1", **fields}}, environment={},
    )
    return resolve_phase_model(settings, phase)


@pytest.fixture
def transport(monkeypatch):
    sent, clients, async_clients = [], [], []

    def handle(request):
        sent.append((request.url, json.loads(request.content)))
        if request.url.path.endswith("/responses"):
            body = {
                "id": "resp_test", "object": "response", "created_at": 1,
                "model": "reported", "status": "completed", "error": None,
                "incomplete_details": None,
                "output": [{
                    "type": "message", "id": "msg_test", "role": "assistant", "status": "completed",
                    "content": [{"type": "output_text", "text": "answer", "annotations": []}],
                }],
                "parallel_tool_calls": False, "tools": [], "tool_choice": "auto",
                "usage": {
                    "input_tokens": 2, "output_tokens": 1, "total_tokens": 3,
                    "input_tokens_details": {"cached_tokens": 0},
                    "output_tokens_details": {"reasoning_tokens": 0},
                },
            }
        else:
            body = {
                "id": "chat_test", "object": "chat.completion", "created": 1, "model": "reported",
                "choices": [{"index": 0, "finish_reason": "stop", "message": {
                    "role": "assistant", "content": "answer",
                }}],
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            }
        return httpx.Response(200, json=body)

    def client(**kwargs):
        result = httpx.Client(transport=httpx.MockTransport(handle), **kwargs)
        clients.append(result)
        return result

    def async_client(**kwargs):
        result = httpx.AsyncClient(transport=httpx.MockTransport(handle), **kwargs)
        async_clients.append(result)
        return result

    monkeypatch.setattr(_chat_models, "httpx", SimpleNamespace(
        **{**vars(httpx), "Client": client, "AsyncClient": async_client},
    ))
    yield sent
    for value in clients:
        value.close()
    for value in async_clients:
        asyncio.run(value.aclose())


@pytest.mark.parametrize("endpoint, expected", [
    ("http://localhost:11434/v1", "http://localhost:11434/api/chat"),
    ("https://gateway.test/tenant/v1/", "https://gateway.test/tenant/api/chat"),
    ("https://gateway.test/tenant/api/chat", "https://gateway.test/tenant/api/chat"),
    ("https://user:secret@gateway.test/v1?deployment=x#ignored",
     "https://user:secret@gateway.test/api/chat?deployment=x"),
])
def test_native_endpoint_preserves_execution_route(endpoint, expected):
    assert native_endpoint(endpoint) == expected


@pytest.mark.parametrize("endpoint", [
    "relative/secret", "ftp://secret.example/v1", "https://example.test:secret/v1", "https://[secret/v1", None,
])
def test_invalid_native_endpoint_is_refused_without_echo(endpoint):
    with pytest.raises(ConfigurationError, match="absolute HTTP") as error:
        native_endpoint(endpoint)
    assert "secret" not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("fields, expected", [
    ({}, {"seed": 7}),
    ({"temperature": "", "top_p": 0}, {"seed": 7, "top_p": 0}),
    ({"temperature": 0, "max_output_tokens": 23}, {"seed": 7, "temperature": 0, "num_predict": 23}),
    ({"max_output_tokens": 0}, {"seed": 7}),
])
def test_native_body_uses_slot_seed_and_only_set_options(fields, expected):
    body = native_request_body(resolved(protocol="ollama-native", **fields), [{"role": "user", "content": "prompt"}], slot_seed=7)
    assert body == {"model": "chosen", "stream": True, "messages": [{"role": "user", "content": "prompt"}], "options": expected}


@pytest.mark.parametrize("fields, think", [
    ({"reasoning": "on"}, True), ({"reasoning": "off"}, False), ({"reasoning_effort": "high"}, "high"),
])
def test_native_body_sends_reasoning_and_detached_request_options(fields, think):
    settings = resolved(protocol="ollama-native", request_options={"format": {"type": "object"}}, **fields)
    messages = [{"role": "user", "content": "prompt"}]
    body = native_request_body(settings, messages, slot_seed=8)
    assert body["think"] == think
    body["format"]["type"] = "changed"
    body["messages"][0]["content"] = "changed"
    assert settings.request_options["format"]["type"] == "object"
    assert messages[0]["content"] == "prompt"


@pytest.mark.parametrize("slot_seed", [True, 1.5, "3", None])
def test_native_seed_requires_an_integer(slot_seed):
    with pytest.raises(ConfigurationError, match="slot_seed"):
        native_request_body(resolved(protocol="ollama-native"), [], slot_seed=slot_seed)


def test_native_cannot_use_the_chat_model_constructor():
    with pytest.raises(ConfigurationError, match="native stream"):
        build_resolved_chat_model(resolved(protocol="ollama-native"), api_key="secret")


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
@pytest.mark.parametrize("protocol", ["openai-chat", "openai-responses"])
def test_real_sdk_request_keeps_phase_sampling_caps_and_extra_fields(phase, protocol, transport):
    settings = resolved(phase, protocol=protocol, temperature=0, top_p=0, max_output_tokens=23,
                        request_options={"top_k": 12, "vendor_options": {"mode": ["a", "b"]}})
    if phase != "starter":
        settings = replace(settings, values={**settings.values, "seed": 0})
    model = build_resolved_chat_model(settings, api_key="credential")
    assert isinstance(model, BaseChatModel)
    assert model.max_retries == 0
    assert model.http_client.timeout.read == settings.values["request_timeout_seconds"]
    model.invoke("hello")
    assert len(transport) == 1
    url, body = transport[0]
    assert url.path == "/v1/" + ("responses" if protocol == "openai-responses" else "chat/completions")
    assert body["model"] == "chosen"
    assert body["temperature"] == body["top_p"] == 0
    assert body["top_k"] == 12
    assert body["vendor_options"] == {"mode": ["a", "b"]}
    if protocol == "openai-responses":
        assert body["max_output_tokens"] == 23
        assert "max_tokens" not in body
        assert "max_completion_tokens" not in body
    else:
        assert body["max_tokens"] == body["max_completion_tokens"] == 23
    assert body.get("seed") == (0 if phase != "starter" else None)
    assert "credential" not in json.dumps(body)


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
@pytest.mark.parametrize("protocol", ["openai-chat", "openai-responses"])
def test_unset_fields_are_not_sent_by_real_sdk(phase, protocol, transport):
    model = build_resolved_chat_model(resolved(phase, protocol=protocol, temperature=""), api_key="")
    model.invoke("hello")
    body = transport[0][1]
    assert not {"temperature", "top_p", "seed", "max_tokens", "max_completion_tokens", "max_output_tokens", "reasoning", "reasoning_effort"}.intersection(body)


@pytest.mark.parametrize("dialect, fields, expected", [
    ("deepseek", {"reasoning": "off"}, {"thinking": {"type": "disabled"}}),
    ("vllm", {"reasoning_template": "qwen", "reasoning": "on", "reasoning_budget_tokens": 4096},
     {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096}),
    ("vllm", {"reasoning_template": "deepseek", "reasoning": "on", "reasoning_effort": "high"},
     {"chat_template_kwargs": {"thinking": True, "reasoning_effort": "high"}}),
    ("openrouter", {"reasoning": "on", "reasoning_effort": "high"},
     {"reasoning": {"enabled": True, "effort": "high"}}),
    ("openai", {"reasoning_effort": "high"}, {"reasoning_effort": "high"}),
])
def test_real_chat_request_carries_dialect_fields(dialect, fields, expected, transport):
    model = build_resolved_chat_model(resolved("coverage", reasoning_dialect=dialect, max_output_tokens=23, **fields), api_key="")
    model.invoke("hello")
    body = transport[0][1]
    for key, value in expected.items():
        assert body[key] == value
    assert body["max_completion_tokens"] == 23
    assert ("max_tokens" in body) == (dialect != "openai")


def test_real_responses_request_uses_native_reasoning_and_output_cap(transport):
    model = build_resolved_chat_model(resolved("mutation", protocol="openai-responses", reasoning_effort="high", max_output_tokens=23), api_key="")
    model.invoke("hello")
    assert transport[0][1]["reasoning"] == {"effort": "high"}
    assert transport[0][1]["max_output_tokens"] == 23
    assert "max_tokens" not in transport[0][1]


@pytest.fixture
def factory_module(monkeypatch, provider_module):
    module = provider_module("testuary_protocol_fixture")
    calls = []

    def factory(**kwargs):
        calls.append(kwargs)
        return FakeListChatModel(responses=["answer"])

    module.factory = factory
    return module, calls


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_added_factory_receives_standard_settings_and_detached_options_once(phase, factory_module):
    module, calls = factory_module
    settings = resolved(phase, chat_model=f"{module.__name__}:factory", temperature=0, top_p=0,
                        max_output_tokens=23, request_timeout_seconds=18,
                        request_options={"vendor_options": {"mode": ["a", "b"]}})
    model = build_resolved_chat_model(settings, api_key="credential")
    assert isinstance(model, BaseChatModel)
    assert calls == [{"model": "chosen", "base_url": "https://example.test/v1", "api_key": "credential",
                      "temperature": 0, "top_p": 0, "timeout": 18, "max_retries": 0, "max_tokens": 23,
                      "vendor_options": {"mode": ["a", "b"]}}]
    calls[0]["vendor_options"]["mode"].append("changed")
    assert settings.request_options["vendor_options"]["mode"] == ("a", "b")
    assert "credential" not in repr(settings)


def test_added_factory_omits_unset_sampling_and_cap(factory_module):
    module, calls = factory_module
    build_resolved_chat_model(resolved("coverage", chat_model=f"{module.__name__}:factory"), api_key="")
    assert not {"temperature", "top_p", "seed", "max_tokens"}.intersection(calls[0])


@pytest.mark.parametrize("result", [None, object(), {"content": "answer"}])
def test_added_factory_must_return_a_chat_model(result, factory_module):
    module, _ = factory_module
    module.factory = lambda **kwargs: result
    with pytest.raises(ConfigurationError, match="BaseChatModel"):
        build_resolved_chat_model(resolved(chat_model=f"{module.__name__}:factory"), api_key="secret")


@pytest.mark.parametrize("failure_at", ["import", "construction"])
def test_factory_failure_is_sanitized_including_exception_context(failure_at, factory_module, tmp_path):
    module, _ = factory_module
    credential = "credential-private-sentinel"

    def fail(*args, **kwargs):
        raise RuntimeError("credential-private-sentinel")

    chat_model = f"{module.__name__}:factory"
    if failure_at == "import":
        provider = tmp_path / "failing_provider.py"
        provider.write_text("raise RuntimeError('credential-private-sentinel')\n", encoding="utf-8")
        chat_model = f"{provider}:factory"
    else:
        module.factory = fail
    with pytest.raises(ConfigurationError) as error:
        build_resolved_chat_model(resolved(chat_model=chat_model), api_key=credential)
    assert credential not in "".join(traceback.format_exception(error.value))


@pytest.mark.parametrize("factory", ["missing_fixture:factory", "testuary_protocol_fixture:missing"])
def test_missing_factory_is_a_configuration_error(factory, factory_module):
    with pytest.raises(ConfigurationError, match="is found in this environment|has no attribute"):
        build_resolved_chat_model(resolved(chat_model=factory), api_key="secret")


def test_factory_cancellation_is_not_reclassified(factory_module):
    module, _ = factory_module

    def cancel(**kwargs):
        raise KeyboardInterrupt

    module.factory = cancel
    with pytest.raises(KeyboardInterrupt):
        build_resolved_chat_model(resolved(chat_model=f"{module.__name__}:factory"), api_key="secret")
