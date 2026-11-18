"""Starter generation routes by the configured host, not a URL substring."""
from contextlib import contextmanager
import json

import httpx
import pytest
from langchain_core.messages import AIMessageChunk

from starter_models import generate, starter_settings

from testuary import _chat_models
from testuary import config as config_module
from testuary import models
from testuary import starters
from phase_values import serve_starter_settings


@pytest.mark.parametrize("endpoint", [
    "https://not-ollama.com/v1",
    "https://ollama.com.example/v1",
    "https://example.test/ollama.com/v1",
    "https://example.test/v1?destination=ollama.com",
    "https://ollama.com@example.test/v1",
])
def test_similar_host_path_query_or_user_info_does_not_select_native(endpoint, monkeypatch):
    seen = []

    class Model:
        def stream(self, messages):
            yield AIMessageChunk(content="answer", usage_metadata={
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
            })

    def build(**kwargs):
        seen.append(kwargs)
        return Model()

    monkeypatch.setattr(_chat_models, "build_chat_model", build)
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: pytest.fail("wrong native route"))
    text, _ = generate(
        {"model": "chosen", "api_url": endpoint, "api_key": "key"},
        "prompt", temperature=0.7, seed=2, timeout=60,
    )
    assert text == "answer"
    assert len(seen) == 1
    assert seen[0]["api_url"] == endpoint


@pytest.mark.parametrize("endpoint, expected", [
    ("https://ollama.com/v1", "https://ollama.com/api/chat"),
    ("https://OLLAMA.com/v1", "https://OLLAMA.com/api/chat"),
    ("https://ollama.com:8443/v1/", "https://ollama.com:8443/api/chat"),
    ("https://ollama.com/", "https://ollama.com/api/chat"),
    ("https://ollama.com/api/chat", "https://ollama.com/api/chat"),
    ("https://ollama.com/gateway/v1", "https://ollama.com/gateway/api/chat"),
    ("https://user:key@ollama.com/v1?deployment=chosen#ignored",
     "https://user:key@ollama.com/api/chat?deployment=chosen"),
])
def test_native_route_preserves_the_configured_service_base(endpoint, expected, monkeypatch):
    seen = []

    class Response:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield '{"message":{"content":"answer"}}'
            yield '{"done":true,"prompt_eval_count":1,"eval_count":1}'

    @contextmanager
    def stream(method, url, **kwargs):
        seen.append((method, url, kwargs))
        yield Response()

    monkeypatch.setattr(httpx, "stream", stream)
    monkeypatch.setattr(_chat_models, "build_chat_model", lambda **k: pytest.fail("wrong compatible route"))
    text, _ = generate(
        {"model": "chosen", "api_url": endpoint, "api_key": "key"},
        "prompt", temperature=0.7, seed=2, timeout=60,
    )
    assert text == "answer"
    assert len(seen) == 1
    assert seen[0][0:2] == ("POST", expected)
    assert seen[0][2]["json"]["options"] == {"seed": 2, "temperature": 0.7}
    assert seen[0][2]["headers"] == {"Authorization": "Bearer key"}
    assert seen[0][2]["timeout"] == 60


@pytest.mark.parametrize("endpoint, expected", [
    ("https://OLLAMA.com/v1", "ollama_native"),
    ("https://example.test/ollama.com/v1", "openai_compatible"),
    ("https://ollama.com.example/v1", "openai_compatible"),
])
def test_actual_initial_worker_persists_the_route_used(endpoint, expected, tmp_path, monkeypatch):
    root = tmp_path / "subject"
    root.mkdir()
    target = root / "module.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    api = {"model": "chosen", "api_url": endpoint, "api_key": "key"}
    with config_module.use_run_settings(starter_settings(api)):
        config = config_module.get_agent_config("test_initial")
    config.update({"suite_count": 1, "attempts_per_suite": 1})
    serve_starter_settings(monkeypatch, config)
    monkeypatch.setattr(models, "get_api_config", lambda: api)
    sent = []

    class Model:
        def stream(self, messages):
            sent.append("openai_compatible")
            yield AIMessageChunk(content="no code fence", usage_metadata={
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
            })

    class Response:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield '{"message":{"content":"no code fence"}}'
            yield '{"done":true,"prompt_eval_count":1,"eval_count":1}'

    @contextmanager
    def stream(*args, **kwargs):
        sent.append("ollama_native")
        yield Response()

    monkeypatch.setattr(httpx, "stream", stream)
    monkeypatch.setattr(_chat_models, "build_chat_model", lambda **k: Model())
    suites = tmp_path / "suites"
    with config_module.use_run_settings(starter_settings(api)):
        result = starters.run_starter_phase(
            language="python", target_file_path=str(target),
            source_dir=str(root), test_output_dir=str(tmp_path / "output"),
            suites_dir=str(suites), log=lambda *a, **k: None,
        )
    records = [json.loads(line) for line in (suites / "prompt_transport.jsonl").read_text("utf-8").splitlines()]
    assert sent == [expected]
    assert [entry["provider_transport"] for entry in records] == [expected]
    assert result.status == "cancelled_zero_passing"
    assert result.requests_used == result.logical_requests == 1
