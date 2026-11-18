"""Configured reasoning fields through capture, replay, context and evidence."""
from __future__ import annotations

import json

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompt_values import ChatPromptValue
from langchain_openai import ChatOpenAI
from typing_extensions import Annotated, TypedDict
from langgraph.graph.message import add_messages

from testuary.agent import _context as context, _langgraph as lg
from testuary import _chat_log
from testuary.agent._helpers import AgentLogger
from testuary import _chat_models
from testuary import config as config_module
from testuary.models import DEFAULT_FIELDS, field_names
from testuary.settings import RunSettings
import test_coverage_agent as coverage_harness
import test_mutation_agent as mutation_harness


FIELDS = ("reasoning_details", "custom_reasoning")


def _no_classify(state):
    """The graph's classifier, for a test whose loop classifies nothing."""
    return None


def _model(fields=FIELDS):
    return _chat_models.build_chat_model(
        model="fixture", api_url="http://127.0.0.1:1/v1", api_key="EMPTY", timeout_seconds=3600,
        preserve_reasoning_content=True, reasoning_field_names=fields,
    )


def _response(**sidecars):
    return {
        "id": "fixture", "object": "chat.completion", "created": 0, "model": "fixture",
        "choices": [{"index": 0, "message": {"role": "assistant", "content": "answer", **sidecars},
                     "finish_reason": "stop"}],
    }


@pytest.mark.parametrize("prompt_value", [False, True])
def test_configured_fields_survive_capture_replay_and_strip(prompt_value):
    model = _model()
    old = model._create_chat_result(_response(
        reasoning_details=[{"text": "details"}], custom_reasoning="custom",
        reasoning="not configured",
    )).generations[0].message
    latest = AIMessage("next", additional_kwargs={"custom_reasoning": "latest"})
    messages = [HumanMessage("task"), old, latest]
    input_value = ChatPromptValue(messages=messages) if prompt_value else messages
    payload = model._get_request_payload(input_value)
    assert payload["messages"][1]["reasoning_details"] == [{"text": "details"}]
    assert payload["messages"][1]["custom_reasoning"] == "custom"
    assert "reasoning" not in old.additional_kwargs
    stripped = context.strip_old_thinking(messages, reasoning_field_names=FIELDS)
    assert stripped[1].additional_kwargs == {}
    assert stripped[2].additional_kwargs["custom_reasoning"] == "latest"
    assert old.additional_kwargs["custom_reasoning"] == "custom"


@pytest.mark.parametrize("outbound", [None, [], [{"role": "user"}], [None]])
def test_reasoning_replay_refuses_a_mismatched_payload(monkeypatch, outbound):
    model = _model()
    monkeypatch.setattr(ChatOpenAI, "_get_request_payload", lambda *a, **k: {"messages": outbound})
    with pytest.raises(ValueError, match="cannot attach configured reasoning"):
        model._get_request_payload([AIMessage("answer", additional_kwargs={"custom_reasoning": "keep"})])


def test_reasoning_replay_does_not_swallow_an_attachment_error(monkeypatch):
    class BrokenMessage(dict):
        def update(self, values):
            raise RuntimeError("fixture attachment failure")

    model = _model()
    monkeypatch.setattr(ChatOpenAI, "_get_request_payload", lambda *a, **k: {
        "messages": [BrokenMessage(role="assistant")],
    })
    with pytest.raises(RuntimeError, match="fixture attachment failure"):
        model._get_request_payload([AIMessage("answer", additional_kwargs={"custom_reasoning": "keep"})])


def test_empty_list_is_not_replaced_with_a_default():
    model = _model(())
    message = model._create_chat_result(_response(reasoning="not selected")).generations[0].message
    assert model.reasoning_field_names == ()
    assert message.additional_kwargs == {}
    assert field_names("") == ()
    assert field_names(None) == DEFAULT_FIELDS


@pytest.mark.parametrize("value", ["content", "role", "tool_calls", "custom,custom", "bad-field"])
def test_invalid_or_colliding_fields_are_refused_before_execution(value):
    with pytest.raises(ValueError, match="reasoning_field_names"):
        RunSettings(sections={"coverage": {"reasoning_field_names": value}})


def test_context_size_and_rebuild_use_the_same_configured_fields():
    old = AIMessage("", additional_kwargs={"custom_reasoning": "x" * 100, "reasoning": "ignored"})
    latest = AIMessage("next")
    assert context.estimated_tokens([old], reasoning_field_names=FIELDS) == 25
    hook = context.make_context_hook(lambda _: "", reasoning_field_names=FIELDS, rebuild="every_turn")
    rebuilt = hook({"messages": [old, latest]})["messages"]
    assert "custom_reasoning" not in rebuilt[0].additional_kwargs
    assert rebuilt[0].additional_kwargs["reasoning"] == "ignored"


def test_transport_and_durable_logs_follow_the_configured_fields(tmp_path):
    message = AIMessage("answer", additional_kwargs={"custom_reasoning": "keep", "reasoning": "ignore"})
    transport = tmp_path / "transport.jsonl"
    logger = lg.UsageLogger(None, prompt_transport_path=transport, reasoning_field_names=FIELDS)
    logger.log_prompt_transport(request_idx=1, messages=[message])
    expected = lg._message_transport_payload(message, reasoning_field_names=FIELDS)
    assert expected["reasoning"] == {"custom_reasoning": "keep"}
    assert json.loads(transport.read_text())["request_idx"] == 1
    flush = _chat_log.make_chat_log_flusher(output_dir=str(tmp_path), filename="chat.json", reasoning_field_names=FIELDS)
    flush([message])
    snapshot = _chat_log.make_retry_chat_snapshotter(output_dir=str(tmp_path), filename_prefix="retry", reasoning_field_names=FIELDS)
    snapshot(1, {"messages": [message]}, RuntimeError("fixture"))
    for filename in ("chat.json", "retry_pre_retry_001.json"):
        data = json.loads((tmp_path / filename).read_text())
        assert data["messages"][0]["additional_kwargs"] == {"custom_reasoning": "keep"}
    logger = AgentLogger("fixture")
    logger.reasoning_field_names = FIELDS
    assert logger.extract_messages([message])[0]["additional_kwargs"] == {"custom_reasoning": "keep"}


@pytest.mark.parametrize("mode", ["per_mode"])
def test_real_graph_trace_uses_configured_reasoning_fields(tmp_path, mode):
    class Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    class State(TypedDict):
        messages: Annotated[list, add_messages]

    trace_path = tmp_path / "trace.jsonl"
    options = {"agent_mode_tools": {"edit": []}, "resolve_agent_mode": lambda _: "edit"}
    graph = lg.build_react_graph(
        classify_node=_no_classify,
        model=Model(responses=[AIMessage("done", additional_kwargs={"custom_reasoning": "seven!!", "reasoning": "ignore"})]),
        state_type=State, trace_logger=lg.TraceLogger(trace_path), reasoning_field_names=FIELDS, **options,
    )
    graph.invoke({"messages": [HumanMessage("task")]})
    events = [json.loads(line) for line in trace_path.read_text().splitlines()]
    end = next(event for event in events if event["event"] == "model_invoke_end")
    assert end["reasoning_content_len"] == 7


@pytest.mark.parametrize("harness", [coverage_harness, mutation_harness], ids=["coverage", "mutation"])
def test_actual_agent_builder_passes_fields_into_context_and_graph(monkeypatch, tmp_path, harness, in_run):
    phase = "coverage" if harness is coverage_harness else "mutation"
    in_run(overrides={"api": {"provider": "ollama", "model": "test-model"},
                      phase: {"reasoning_field_names": ",".join(FIELDS)}})
    agent = harness._make_agent(tmp_path)
    captured = {}
    monkeypatch.setattr(lg, "build_react_graph", lambda **kwargs: captured.update(kwargs) or "graph")
    assert agent.build_graph(model=_model()) == "graph"
    assert captured["reasoning_field_names"] == FIELDS
    old = AIMessage("old", additional_kwargs={"custom_reasoning": "remove"})
    latest = AIMessage("latest", additional_kwargs={"custom_reasoning": "keep"})
    messages = captured["pre_model_hook"]({"messages": [old, latest]})["messages"]
    assistants = [message for message in messages if isinstance(message, AIMessage)]
    assert "custom_reasoning" not in assistants[0].additional_kwargs
    assert assistants[1].additional_kwargs["custom_reasoning"] == "keep"


def test_resolver_keeps_the_explicit_settings_field_list():
    settings = RunSettings(sections={"coverage": {"reasoning_field_names": "reasoning_content,reasoning_details,thinking"}})
    with config_module.use_run_settings(settings):
        assert config_module.get_agent_config("test_coverage")["reasoning_field_names"] == (
            "reasoning_content", "reasoning_details", "thinking",
        )
