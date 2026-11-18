"""Added factories at actual phase boundaries, using loopback SDK requests."""
import asyncio
from dataclasses import replace
from importlib import metadata
import json
from types import SimpleNamespace

from langchain_openai import ChatOpenAI
import pytest

import test_model_precedence_callers as caller_harness
import test_protocol_loopback_callers as http_harness
import test_resolved_starter_requests as starter_harness
from testuary import model_requests as _model_requests
from testuary import evidence as validity
from testuary import config
from testuary import starters

loopback = http_harness.loopback
PHASES = ("starter", "coverage", "mutation")


@pytest.fixture
def added_factory(monkeypatch, provider_module):
    state = SimpleNamespace(parameters=[], clients=[])
    module = provider_module("_fixture_loopback_added_provider")

    def factory(**parameters):
        state.parameters.append(parameters.copy())
        custom = parameters.pop("custom_option")
        client = ChatOpenAI(**parameters, use_responses_api=False, stream_usage=True,
                            extra_body={"custom_option": custom})
        state.clients.append(client)
        return client

    module.factory = factory
    state.factory = module.__name__ + ":factory"
    installed_version = metadata.version
    monkeypatch.setattr(metadata, "packages_distributions", lambda: {module.__name__: ["fixture-added-provider"]})
    monkeypatch.setattr(metadata, "version", lambda name: "0.1" if name == "fixture-added-provider" else installed_version(name))
    yield state
    for client in state.clients:
        client.root_client.close()
        asyncio.run(client.root_async_client.close())


def settings_for(tmp_path, loopback, added_factory):
    settings = http_harness.configured(tmp_path, "added", loopback)
    return replace(settings, sections={**settings.sections,
        "starter": {**settings.sections["starter"], "temperature": 0, "request_timeout_seconds": 5},
        "coverage": {**settings.sections["coverage"], "seed": 0, "request_timeout_seconds": 5},
        "mutation": {**settings.sections["mutation"], "seed": 0, "request_timeout_seconds": 5},
    }, models={"explicit": {
        **settings.models["explicit"], "chat_model": added_factory.factory,
        "temperature": 0, "top_p": 0, "max_output_tokens": 64,
        "request_options": {"custom_option": {"items": [1, 2]}},
    }})


def records(tmp_path, phase):
    path = (tmp_path / "suites/request_records.jsonl" if phase == "starter" else
            tmp_path / "logs" / f"{phase}_usage.jsonl")
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("status", (200, 503, 401), ids=("delivered", "one-503", "terminal-401"))
def test_added_factory_streams_binds_and_accounts_at_actual_phase_entry(
    phase, status, tmp_path, monkeypatch, loopback, added_factory, capsys,
):
    if status != 200:
        loopback.failures.append(status)
    terminal = status == 401
    logical = 1 if phase == "starter" or terminal else 2
    expected = logical + int(status == 503)
    usage = {}
    configured = settings_for(tmp_path, loopback, added_factory)
    record = caller_harness.invoke_phase(phase, tmp_path, configured, monkeypatch,
                                        expected_requests=expected, usage_snapshot=usage)
    assert len(added_factory.clients) == len(added_factory.parameters) == 1
    assert added_factory.parameters[0] == {
        "model": "chosen", "base_url": loopback.endpoint, "api_key": http_harness.TOKEN,
        "timeout": 5, "max_retries": 0, "temperature": 0, "top_p": 0,
        "seed": 38 if phase == "starter" else 0, "max_tokens": 64,
        "custom_option": {"items": [1, 2]},
    }
    assert len(loopback.requests) == expected
    assert all(path == "/v1/chat/completions" and header == f"Bearer {http_harness.TOKEN}"
               for path, _, header in loopback.requests)
    assert all(body["model"] == "chosen" and body["custom_option"] == {"items": [1, 2]}
               and body["temperature"] == body["top_p"] == 0
               and body["max_completion_tokens"] == 64 for _, body, _ in loopback.requests)
    assert record["sent_settings"]["protocol"] == "added"
    assert record["sent_settings"]["chat_model"] == added_factory.factory
    assert usage == {"physical": expected, "logical": logical, "unknown": int(status != 200),
                     "input_tokens": 0 if terminal else logical * 10,
                     "output_tokens": 0 if terminal else logical * 3}
    journal = records(tmp_path, phase)
    assert [row["outcome"] for row in journal] == (
        [_model_requests.TERMINAL] if terminal else
        ([_model_requests.TRANSIENT] if status == 503 else []) + [_model_requests.DELIVERED] * logical
    )
    if status == 503:
        assert loopback.requests[0] == loopback.requests[1]
    if terminal:
        assert loopback.successes == 0 and journal[0]["usage_known"] is False
    else:
        returned_key = "response_model" if phase == "starter" else "model_name"
        assert all(row[returned_key] == "reported" for row in journal if row["outcome"] == _model_requests.DELIVERED)
    if phase == "starter":
        assert all(body["stream"] is True and "tools" not in body for _, body, _ in loopback.requests)
    else:
        names = {tool["function"]["name"] for tool in loopback.requests[0][1]["tools"]}
        assert "read_tests" in names
        if not terminal:
            outputs = [item for item in loopback.requests[-1][1]["messages"] if item["role"] == "tool"]
            assert len(outputs) == 1 and outputs[0]["tool_call_id"] == "read-1"
            assert "test_ok" in outputs[0]["content"]
    assert all(http_harness.TOKEN.encode() not in path.read_bytes()
               for path in tmp_path.rglob("*") if path.is_file())
    output = capsys.readouterr()
    assert http_harness.TOKEN not in output.out + output.err
    assert http_harness.TOKEN not in repr(configured) + json.dumps(configured.export())


def test_added_paid_initial_delivery_resumes_without_another_factory_or_request(
    tmp_path, monkeypatch, loopback, added_factory,
):
    configured = settings_for(tmp_path, loopback, added_factory)
    append = validity.append_jsonl

    def unavailable(path, record):
        if path.name == "request_records.jsonl":
            raise OSError("fixture request journal unavailable")
        return append(path, record)

    monkeypatch.setattr(validity, "append_jsonl", unavailable)
    with pytest.raises(_model_requests.RecordNotWritten):
        starter_harness.run_worker(tmp_path, configured)
    suites = tmp_path / "suites"
    delivery = suites / "suite_001" / starters._GENERATION_DELIVERY
    assert json.loads(delivery.read_text())["unrecorded"] is True
    assert len(loopback.requests) == len(added_factory.clients) == 1
    preserved = {name: (suites / name).read_bytes() for name in (
        "generation_origin.json", "prompt_transport.jsonl",
    )}
    monkeypatch.setattr(validity, "append_jsonl", append)
    resumed = replace(configured, sections={**configured.sections,
        "starter": {**configured.sections["starter"], "reuse_suites": True}})
    with config.use_run_settings(resumed):
        result = starters.run_starter_phase(
            language="python", target_file_path=str(tmp_path / "subject/module.py"),
            source_dir=str(tmp_path / "subject"), test_output_dir=str(tmp_path / "output"),
            suites_dir=str(suites), log=lambda *a, **kw: None,
        )
    assert len(loopback.requests) == len(added_factory.clients) == result.requests_used == result.logical_requests == 1
    assert result.input_tokens == 10 and result.output_tokens == 3
    assert result.unknown_usage_requests == 0 and result.suite_generation_attempts == {1: 1}
    assert not delivery.exists()
    assert all((suites / name).read_bytes() == content for name, content in preserved.items())
