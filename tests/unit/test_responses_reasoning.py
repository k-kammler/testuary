"""Native Responses reasoning survives real SDK and graph tool-call turns."""
import asyncio
from copy import deepcopy
import json
from types import SimpleNamespace
from typing import Annotated, TypedDict

import httpx
from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
import pytest

import test_coverage_agent as coverage_harness
import test_mutation_agent as mutation_harness
from testuary.agent import _context, _langgraph as lg
from testuary import _chat_models
from testuary import config
from testuary.models import resolve_phase_model
from testuary.models import build_resolved_chat_model
from testuary.settings import RunSettings


REASONING = {"type": "reasoning", "id": "rs_first", "status": "completed",
             "summary": [{"type": "summary_text", "text": "A summary"}],
             "encrypted_content": "opaque-content"}


def _no_classify(state):
    """The graph's classifier, for a test whose loop classifies nothing."""
    return None


@pytest.fixture
def response_transport(monkeypatch):
    sent, clients, async_clients = [], [], []

    def handle(request):
        sent.append(json.loads(request.content))
        assert request.url.path == "/v1/responses"
        output = [deepcopy(REASONING), {
            "type": "function_call", "id": "fc_first", "call_id": "call_first",
            "name": "observe", "arguments": "{}", "status": "completed",
        }] if len(sent) == 1 else [{
            "type": "message", "id": "msg_second", "role": "assistant", "status": "completed",
            "content": [{"type": "output_text", "text": "done", "annotations": []}],
        }]
        return httpx.Response(200, json={
            "id": f"resp_{len(sent)}", "object": "response", "created_at": 1,
            "model": "reported", "status": "completed", "error": None,
            "incomplete_details": None, "output": output, "parallel_tool_calls": False,
            "tools": [], "tool_choice": "auto", "usage": {
                "input_tokens": 2, "output_tokens": 1, "total_tokens": 3,
                "input_tokens_details": {"cached_tokens": 0},
                "output_tokens_details": {"reasoning_tokens": 1},
            },
        })

    def client(**kwargs):
        result = httpx.Client(transport=httpx.MockTransport(handle), **kwargs)
        clients.append(result)
        return result

    def async_client(**kwargs):
        result = httpx.AsyncClient(transport=httpx.MockTransport(handle), **kwargs)
        async_clients.append(result)
        return result

    monkeypatch.setattr(_chat_models, "httpx", SimpleNamespace(**{**vars(httpx), "Client": client, "AsyncClient": async_client}))
    yield sent
    for value in clients:
        value.close()
    for value in async_clients:
        asyncio.run(value.aclose())


def _model(phase, route, tmp_path=None):
    settings = RunSettings(sections={
        "api": {"model": "openai_example"},
        "coverage": {"preserve_reasoning_content": True, "protocol": "openai-responses"},
    }, models={
        "openai_example": {"provider": "openai", 
            "api_url": "https://example.test/v1", "model": "chosen",
            "protocol": "openai-responses", "api_key": "fixture",
        },
    })
    if route == "resolved":
        return build_resolved_chat_model(resolve_phase_model(settings, phase), api_key="fixture")
    harness = coverage_harness if phase == "coverage" else mutation_harness
    with config.use_run_settings(settings):
        return harness._make_agent(tmp_path).build_chat_model()


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
@pytest.mark.parametrize("route", ["agent", "resolved"])
@pytest.mark.parametrize("mode", ["per_mode"])
def test_real_graph_returns_native_reasoning_unchanged_with_tool_result(phase, route, mode, response_transport, tmp_path):
    calls = []

    @tool
    def observe() -> str:
        """Return one observation."""
        calls.append("observed")
        return "observation"

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    options = {"agent_mode_tools": {"edit": [observe]}, "resolve_agent_mode": lambda _: "edit"}
    graph = lg.build_react_graph(classify_node=_no_classify, model=_model(phase, route, tmp_path), state_type=State, **options)
    result = graph.invoke({"messages": [HumanMessage("task")]})
    assert calls == ["observed"]
    assert len(response_transport) == 2
    assert REASONING in response_transport[1]["input"]
    assert any(item.get("type") == "function_call_output" and item.get("output") == "observation"
               for item in response_transport[1]["input"])
    assistants = [message for message in result["messages"] if isinstance(message, AIMessage)]
    assert REASONING in assistants[0].content


def test_old_native_reasoning_is_stripped_only_when_context_rebuilds():
    text = {"type": "text", "text": "visible", "id": "msg_first"}
    call = {"type": "function_call", "name": "observe", "arguments": "{}", "call_id": "call_first"}
    old = AIMessage(content=[deepcopy(REASONING), text, call])
    latest = AIMessage(content=[deepcopy(REASONING), {"type": "text", "text": "latest"}])
    original = deepcopy(old.content)
    stripped = _context.strip_old_thinking([old, latest], reasoning_field_names=())
    assert stripped[0].content == [text, call]
    assert stripped[1] is latest
    assert old.content == original


def test_native_reasoning_cannot_be_silently_dropped_for_stateless_input(response_transport):
    model = _chat_models.build_chat_model(
        model="chosen", api_url="https://example.test/v1", api_key="fixture", timeout_seconds=60,
        preserve_reasoning_content=True, use_responses_api=True, extra_kwargs={"store": False},
    )
    block = {key: value for key, value in REASONING.items() if key != "encrypted_content"}
    with pytest.raises(ValueError, match="cannot preserve.*Responses reasoning"):
        model.invoke([HumanMessage("task"), AIMessage(content=[block])])
    assert response_transport == []


def test_native_encrypted_reasoning_can_be_replayed_statelessly(response_transport):
    model = _chat_models.build_chat_model(
        model="chosen", api_url="https://example.test/v1", api_key="fixture", timeout_seconds=60,
        preserve_reasoning_content=True, use_responses_api=True, extra_kwargs={"store": False},
    )
    block = {**deepcopy(REASONING), "index": 0}
    block["summary"][0]["index"] = 0
    original = deepcopy(block)
    model.invoke([HumanMessage("task"), AIMessage(content=[block])])
    assert REASONING in response_transport[0]["input"]
    assert block == original


@pytest.mark.parametrize("outbound", [None, [], [{**REASONING, "id": "changed"}], [REASONING]])
def test_native_replay_refuses_missing_changed_or_reused_items(monkeypatch, response_transport, outbound):
    model = _model("coverage", "resolved")
    monkeypatch.setattr(ChatOpenAI, "_get_request_payload", lambda *a, **k: {"input": outbound})
    with pytest.raises(ValueError, match="cannot preserve.*Responses reasoning"):
        model.invoke([AIMessage(content=[deepcopy(REASONING), deepcopy(REASONING)])])
    assert response_transport == []


def test_responses_output_format_is_not_changed_by_ambient_sdk_setting(monkeypatch, response_transport):
    monkeypatch.setenv("LC_OUTPUT_VERSION", "v0")
    model = _model("coverage", "resolved")
    assert model.output_version == "responses/v1"
    answer = model.invoke([HumanMessage("task")])
    assert REASONING in answer.content


def test_chat_sidecars_are_not_silently_dropped_on_a_native_route(response_transport):
    model = _model("coverage", "resolved")
    with pytest.raises(ValueError, match="cannot attach configured reasoning sidecars"):
        model.invoke([AIMessage("answer", additional_kwargs={"reasoning_content": "private"})])
    assert response_transport == []
