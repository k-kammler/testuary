"""Actual SDK streaming, tool turns and recovery against a loopback-only server."""
import os
from collections import deque
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import threading
from types import SimpleNamespace
from urllib.parse import urlsplit

import pytest

import test_model_precedence_callers as caller_harness
import test_resolved_starter_requests as starter_harness
from testuary import model_requests as _model_requests
from testuary import evidence as validity
from testuary import config
from testuary import starters
from testuary.settings import RunSettings

ROUTES = [
    (phase, protocol) for phase in ("starter", "coverage", "mutation")
    for protocol in ("openai-chat", "openai-responses")
] + [("starter", "ollama-native")]
TOKEN = "synthetic-loopback-credential-not-for-output"


def chat_reply(*, stream, tool, model="reported"):
    message = {"role": "assistant", "content": "no code fence"}
    if tool:
        message = {"role": "assistant", "content": None, "tool_calls": [{
            "id": "read-1", "type": "function", "function": {"name": "read_tests", "arguments": "{}"},
        }]}
    result = {"id": "chat_fixture", "created": 1,
              "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13,
                        "prompt_tokens_details": {"cached_tokens": 2}}}
    if model is not None:
        result["model"] = model
    if stream:
        result.update(object="chat.completion.chunk", choices=[{
            "index": 0, "finish_reason": "stop", "delta": message,
        }])
        return "text/event-stream", "data: " + json.dumps(result) + "\n\ndata: [DONE]\n\n"
    result.update(object="chat.completion", choices=[{
        "index": 0, "finish_reason": "tool_calls" if tool else "stop", "message": message,
    }])
    return "application/json", json.dumps(result)


def responses_reply(*, stream, tool, model="reported"):
    output = [{"type": "message", "id": "msg_fixture", "role": "assistant", "status": "completed",
               "content": [{"type": "output_text", "text": "no code fence", "annotations": []}]}]
    if tool:
        output = [{"type": "function_call", "id": "fc_read", "call_id": "read-1",
                   "name": "read_tests", "arguments": "{}", "status": "completed"}]
    result = {
        "id": "resp_fixture", "object": "response", "created_at": 1,
        "status": "completed", "error": None, "incomplete_details": None,
        "output": output, "parallel_tool_calls": False, "tools": [], "tool_choice": "auto",
        "usage": {"input_tokens": 10, "output_tokens": 3, "total_tokens": 13,
                  "input_tokens_details": {"cached_tokens": 2}, "output_tokens_details": {"reasoning_tokens": 0}},
    }
    if model is not None:
        result["model"] = model
    if not stream:
        return "application/json", json.dumps(result)
    events = [{"type": "response.output_text.delta", "delta": "no code fence", "output_index": 0,
               "content_index": 0, "item_id": "msg_fixture", "sequence_number": 0},
              {"type": "response.completed", "sequence_number": 1, "response": result}]
    return "text/event-stream", "".join("data: " + json.dumps(event) + "\n\n" for event in events)


@pytest.fixture
def loopback(monkeypatch):
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("NO_PROXY", "127.0.0.1,localhost")
    state = SimpleNamespace(requests=[], failures=deque(), successes=0)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            state.requests.append((self.path, body, self.headers.get("Authorization")))
            status = state.failures.popleft() if state.failures else 200
            if status != 200:
                content_type, text = "application/json", json.dumps({"error": {
                    "message": getattr(state, "error_message", "fixture temporarily unavailable"), "type": "server_error",
                }})
            else:
                state.successes += 1
                tool = bool(body.get("tools")) and state.successes == 1
                model = getattr(state, "response_model", "reported")
                path = urlsplit(self.path).path
                if path.endswith("/api/chat"):
                    result = {
                        "message": {"content": "no code fence"}, "done": True,
                        "prompt_eval_count": 10, "eval_count": 3,
                    }
                    if model is not None:
                        result["model"] = model
                    content_type, text = "application/x-ndjson", json.dumps(result) + "\n"
                elif "input" in body:
                    content_type, text = responses_reply(stream=bool(body.get("stream")), tool=tool, model=model)
                else:
                    content_type, text = chat_reply(stream=bool(body.get("stream")), tool=tool, model=model)
            data = text.encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server.daemon_threads = True
    worker = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": .01}, daemon=True)
    state.endpoint = f"http://127.0.0.1:{server.server_port}/v1"
    worker.start()
    try:
        yield state
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)
        assert not worker.is_alive(), "loopback fixture thread did not stop"


def configured(tmp_path, protocol, loopback):
    retry = {"transient_first_wait_seconds": 0, "max_transient_wait_seconds": 0}
    return RunSettings(environment=dict(os.environ), base_dir=tmp_path, sections={
        "api": {"model": "explicit"}, "main": {"hints_file": "", "seed": 37},
        "starter": {"suite_count": 1, "attempts_per_suite": 1,
                         "concurrent_requests": 1, "suite_run_timeout_seconds": 2, **retry},
        "coverage": {"max_requests_per_phase": 2, "max_loop_attempts": 0, **retry},
        "mutation": {"max_requests_per_phase": 2, "max_loop_attempts": 0, **retry},
    }, models={"explicit": {
        "provider": "openai", "model": "chosen", "api_url": loopback.endpoint,
        "api_key": TOKEN, "protocol": protocol, "request_timeout_seconds": 5,
    }})


@pytest.mark.parametrize("phase,protocol", ROUTES)
@pytest.mark.parametrize("retry", (False, True), ids=["no-retry", "one-503"])
def test_real_loopback_requests_preserve_streams_tools_protocol_and_retry_accounting(
    phase, protocol, retry, tmp_path, monkeypatch, loopback,
):
    if retry:
        loopback.failures.append(503)
    expected = (1 if phase == "starter" else 2) + int(retry)
    usage = {}
    record = caller_harness.invoke_phase(phase, tmp_path, configured(tmp_path, protocol, loopback),
                                        monkeypatch, expected_requests=expected, usage_snapshot=usage)
    assert len(loopback.requests) == expected
    path = {"openai-chat": "/v1/chat/completions", "openai-responses": "/v1/responses",
            "ollama-native": "/api/chat"}[protocol]
    assert all(url == path and body["model"] == "chosen" and header == f"Bearer {TOKEN}"
               for url, body, header in loopback.requests)
    assert record["sent_settings"]["protocol"] == protocol
    logical = 1 if phase == "starter" else 2
    assert usage == {"physical": expected, "logical": logical, "unknown": int(retry),
                     "input_tokens": logical * 10, "output_tokens": logical * 3}
    if retry:
        assert loopback.requests[0] == loopback.requests[1]
    if phase == "starter":
        rows = [json.loads(line) for line in (tmp_path / "suites/request_records.jsonl").read_text().splitlines()]
        assert [row["outcome"] for row in rows] == (
            [_model_requests.TRANSIENT, _model_requests.DELIVERED] if retry else [_model_requests.DELIVERED])
        assert {row["logical_request_idx"] for row in rows} == {1}
        assert rows[-1]["response_model"] == "reported"
        assert all(body["stream"] is True for _, body, _ in loopback.requests)
    else:
        body = loopback.requests[-1][1]
        if protocol == "openai-chat":
            outputs = [item for item in body["messages"] if item["role"] == "tool"]
            assert len(outputs) == 1 and outputs[0]["tool_call_id"] == "read-1"
        else:
            outputs = [item for item in body["input"] if item["type"] == "function_call_output"]
            assert len(outputs) == 1 and outputs[0]["call_id"] == "read-1"
        assert "test_ok" in outputs[0].get("content", outputs[0].get("output", ""))
        assert loopback.successes == 2
    assert all(TOKEN.encode() not in path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())


@pytest.mark.parametrize("protocol", ("openai-chat", "openai-responses", "ollama-native"))
def test_loopback_paid_starter_response_is_processed_after_resume_without_another_request(
    protocol, tmp_path, monkeypatch, loopback,
):
    settings = configured(tmp_path, protocol, loopback)
    append = validity.append_jsonl

    def unavailable(path, record):
        if path.name == "request_records.jsonl":
            raise OSError("fixture request journal unavailable")
        return append(path, record)

    monkeypatch.setattr(validity, "append_jsonl", unavailable)
    with pytest.raises(_model_requests.RecordNotWritten):
        starter_harness.run_worker(tmp_path, settings)
    starters_dir = tmp_path / "suites"
    delivery = starters_dir / "suite_001" / starters._GENERATION_DELIVERY
    assert json.loads(delivery.read_text())["unrecorded"] is True
    assert len(loopback.requests) == 1
    origin = (starters_dir / "generation_origin.json").read_bytes()
    outbound = (starters_dir / "prompt_transport.jsonl").read_bytes()
    monkeypatch.setattr(validity, "append_jsonl", append)
    resumed = replace(settings, sections={**settings.sections,
        "starter": {**settings.sections["starter"], "reuse_suites": True}})
    with config.use_run_settings(resumed):
        result = starters.run_starter_phase(
            language="python", target_file_path=str(tmp_path / "subject/module.py"),
            source_dir=str(tmp_path / "subject"), test_output_dir=str(tmp_path / "output"),
            suites_dir=str(starters_dir), log=lambda *a, **kw: None,
        )
    assert len(loopback.requests) == result.requests_used == result.logical_requests == 1
    assert result.input_tokens == 10 and result.output_tokens == 3
    assert result.unknown_usage_requests == 0 and result.suite_generation_attempts == {1: 1}
    assert not delivery.exists()
    assert (starters_dir / "generation_origin.json").read_bytes() == origin
    assert (starters_dir / "prompt_transport.jsonl").read_bytes() == outbound
