"""Optional sampling temperature in configuration and serialized requests."""

import json
from contextlib import closing

import httpx
import pytest

from starter_models import generate

from testuary import config

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.mark.parametrize("value,expected", [("", None), ("  ", None),
                                          ("0", 0.0), ("0.7", 0.7)])
def test_explicit_temperature_override(monkeypatch, value, expected, in_run):
    monkeypatch.setenv("TESTUARY_STARTER_TEMPERATURE", value)
    in_run()
    assert config.get_agent_config("test_initial")["temperature"] == expected


def test_invalid_temperature_is_not_silently_unset(monkeypatch, in_run):
    monkeypatch.setenv("TESTUARY_STARTER_TEMPERATURE", "invalid")
    with pytest.raises(ValueError, match=r"\[starter\] temperature: expected a number"):
        in_run()


@pytest.mark.parametrize("endpoint", ["https://api.deepseek.com", "https://model.example/v1"])
def test_chat_temperature_is_optional_on_the_wire(monkeypatch, endpoint):
    requests = []
    clients = []
    client_type = httpx.Client
    answer = "```python\ndef test_ok():\n    assert True\n```"

    def respond(request):
        requests.append(json.loads(request.content))
        chunk = {
            "id": "chatcmpl-test", "object": "chat.completion.chunk",
            "created": 1, "model": "deepseek-v4-pro",
            "choices": [{"index": 0, "delta": {"role": "assistant", "content": answer},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        return httpx.Response(
            200, headers={"content-type": "text/event-stream"},
            text="data: " + json.dumps(chunk) + "\n\ndata: [DONE]\n\n",
        )

    def client(**kwargs):
        instance = client_type(transport=httpx.MockTransport(respond), **kwargs)
        clients.append(instance)
        return instance

    # Patch only the factory's client constructor, not httpx.Client's type.
    from testuary import _chat_models
    from types import SimpleNamespace
    monkeypatch.setattr(_chat_models, "httpx", SimpleNamespace(
        Client=client, AsyncClient=httpx.AsyncClient, Timeout=httpx.Timeout,
    ))
    api = {"model": "deepseek-v4-pro", "api_url": endpoint, "api_key": "test-key"}
    messages = [{"role": "system", "content": "Generate tests."},
                {"role": "user", "content": "Source code."}]
    outcomes = []
    try:
        for temperature in (None, 0.0, 0.7):
            outcomes.append(generate(
                api, messages, temperature=temperature, seed=42, timeout=60,
                max_output_tokens=384000,
            ))
    finally:
        for instance in clients:
            instance.close()
    assert len(requests) == 3
    absent, zero, explicit = requests
    assert "temperature" not in absent
    assert zero.pop("temperature") == 0.0
    assert explicit.pop("temperature") == 0.7
    assert absent == zero == explicit
    assert absent["messages"] == messages
    assert absent["model"] == "deepseek-v4-pro"
    assert absent["seed"] == 42
    assert absent["max_tokens"] == 384000
    assert absent["stream"] is True
    assert outcomes[0] == outcomes[1] == outcomes[2]
    assert outcomes[0][0] == answer


def test_native_temperature_is_optional(monkeypatch):
    requests = []

    def stream(method, url, **kwargs):
        assert method == "POST"
        assert url == "https://ollama.com/api/chat"
        requests.append(kwargs["json"])
        return closing(httpx.Response(
            200, request=httpx.Request(method, url),
            text=json.dumps({"message": {"content": "answer"}, "done": True,
                             "prompt_eval_count": 10, "eval_count": 20}) + "\n",
        ))

    monkeypatch.setattr(httpx, "stream", stream)
    api = {"model": "example:cloud", "api_url": "https://ollama.com", "api_key": "test-key"}
    outcomes = [generate(
        api, "prompt", temperature=value, seed=42, timeout=60,
        max_output_tokens=4096,
    ) for value in (None, 0.0, 0.7)]
    absent, zero, explicit = requests
    assert "temperature" not in absent["options"]
    assert zero["options"].pop("temperature") == 0.0
    assert explicit["options"].pop("temperature") == 0.7
    assert absent == zero == explicit
    assert absent["options"] == {"seed": 42, "num_predict": 4096}
    assert outcomes[0] == outcomes[1] == outcomes[2]
