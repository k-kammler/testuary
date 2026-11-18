"""langchain-openai changes temperature by model name: it removes an explicit
value other than 1 for gpt-5 models and sends 1 for o1 models when none was
configured. The request still carries exactly the configured sampling values,
so the run record states what was sent: explicit values reach the request,
unset values send nothing."""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage

import test_model_precedence_callers as caller_harness
from test_model_precedence_callers import streams, transport  # noqa: F401 (fixtures)
from test_protocol_loopback_callers import ROUTES
from testuary import _chat_models
from testuary.settings import PHASE_SECTIONS, resolve_settings

OPENAI_ROUTES = [route for route in ROUTES if route[1] != "ollama-native"]


def _settings(tmp_path, protocol, model, contents):
    path = tmp_path / "override.ini"
    path.write_text(contents, encoding="utf-8")
    return resolve_settings(base_dir=tmp_path, config_files=[path], environment={},
        models={"openai": {"api_url": "https://example.test/v1", "api_key": "fixture"},
                "explicit": {"provider": "openai", "model": model, "protocol": protocol}},
        overrides={
            "api": {"model": "explicit"}, "main": {"hints_file": "", "seed": 37},
            "starter": {"suite_count": 1, "attempts_per_suite": 1, "concurrent_requests": 1,
                             "suite_run_timeout_seconds": 2},
            "coverage": {"max_requests_per_phase": 1, "max_loop_attempts": 0},
            "mutation": {"max_requests_per_phase": 1, "max_loop_attempts": 0},
        })


@pytest.mark.parametrize("phase,protocol", OPENAI_ROUTES)
@pytest.mark.parametrize("model,configured,expected", [
    ("gpt-5-mini", "0.3", 0.3),
    ("o1-mini", "", None),
])
def test_model_name_does_not_change_the_sent_temperature(
    phase, protocol, model, configured, expected, tmp_path, monkeypatch, request,
):
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    settings = _settings(tmp_path, protocol, model, f"[{PHASE_SECTIONS[phase]}]\ntemperature={configured}\n")
    record = caller_harness.invoke_phase(phase, tmp_path, settings, monkeypatch)
    body = sent[0][1]
    assert record["sent_settings"]["values"]["temperature"] == expected
    assert body.get("temperature") == expected, body


def _payload(model, *, responses, reasoning, call=None, **sampling):
    client = _chat_models.build_chat_model(
        model=model, api_url="http://127.0.0.1:9/v1", api_key="fixture", timeout_seconds=1.0,
        extra_kwargs=sampling, use_responses_api=responses, preserve_reasoning_content=reasoning,
    )
    return client._get_request_payload([HumanMessage("task")], **(call or {}))


@pytest.mark.parametrize("responses", [False, True])
@pytest.mark.parametrize("reasoning", [False, True])
@pytest.mark.parametrize("model", ["gpt-5-mini", "gpt-5", "o1-mini", "o1", "gpt-5-chat-latest", "qwen3"])
def test_the_built_client_sends_the_configured_sampling_for_any_model_name(model, responses, reasoning):
    explicit = _payload(model, responses=responses, reasoning=reasoning, temperature=0.3, top_p=0.9)
    assert (explicit.get("temperature"), explicit.get("top_p")) == (0.3, 0.9)
    unset = _payload(model, responses=responses, reasoning=reasoning)
    assert "temperature" not in unset and "top_p" not in unset


@pytest.mark.parametrize("responses", [False, True])
def test_a_value_passed_with_the_call_is_kept(responses):
    payload = _payload("o1-mini", responses=responses, reasoning=False, temperature=0.3,
                       call={"temperature": 0.5})
    assert payload["temperature"] == 0.5


def test_the_private_method_the_client_overrides_still_exists():
    """The client class overrides langchain-openai's private
    _get_request_payload. A version that drops or renames it would stop
    the configured sampling from reaching the request without an error, so
    this check fails first, naming the method."""
    import inspect

    from langchain_openai.chat_models.base import BaseChatOpenAI

    assert "_get_request_payload" in vars(BaseChatOpenAI), (
        "langchain-openai no longer defines BaseChatOpenAI._get_request_payload, which "
        "testuary._chat_models._ChatOpenAIExactSampling overrides")
    parameters = inspect.signature(BaseChatOpenAI._get_request_payload).parameters
    assert list(parameters)[:2] == ["self", "input_"] and "stop" in parameters, parameters
