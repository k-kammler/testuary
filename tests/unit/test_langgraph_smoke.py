"""Exercise chat-model request and response shapes without a live provider."""
from __future__ import annotations

from typing import Annotated, List

from typing import Any, Sequence

from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from typing_extensions import TypedDict


class _ToolAwareFakeModel(FakeMessagesListChatModel):
    """Fake chat model whose ``bind_tools`` is a pass-through, because the
    scripted responses already contain the tool calls."""

    def bind_tools(  # type: ignore[override]
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable:
        return self


@tool
def read_value(key: str) -> str:
    """Return the value associated with ``key``. Stub tool for the smoke test."""
    return f"value-for-{key}"


@tool
def mark_done(reason: str) -> str:
    """Signal that the agent is finished. Stub tool for the smoke test."""
    return f"ack: {reason}"


class _SmokeState(TypedDict):
    """Minimum LangGraph state for the smoke test."""

    messages: Annotated[List[BaseMessage], add_messages]


def _build_smoke_graph(model):
    """Build a two-node ReAct-style graph: model_decide -> tool_dispatch."""
    tools = [read_value, mark_done]
    model_with_tools = model.bind_tools(tools)

    def model_decide(state: _SmokeState) -> dict:
        response = model_with_tools.invoke(state["messages"])
        return {"messages": [response]}

    tool_node = ToolNode(tools)

    def route_after_model(state: _SmokeState) -> str:
        last = state["messages"][-1]
        if isinstance(last, AIMessage) and last.tool_calls:
            return "tool_dispatch"
        return END

    graph = StateGraph(_SmokeState)
    graph.add_node("model_decide", model_decide)
    graph.add_node("tool_dispatch", tool_node)
    graph.add_edge(START, "model_decide")
    graph.add_conditional_edges("model_decide", route_after_model)
    graph.add_edge("tool_dispatch", "model_decide")
    return graph.compile()


def test_langgraph_tool_call_round_trip_terminates():
    """The scripted model emits one tool call, then a final text answer."""

    # Step 1: model picks read_value with key="alpha".
    # Step 2: after seeing the ToolMessage, model emits final answer.
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_value",
                    "args": {"key": "alpha"},
                    "id": "call_1",
                }
            ],
        ),
        AIMessage(content="done"),
    ]
    model = _ToolAwareFakeModel(responses=scripted)
    graph = _build_smoke_graph(model)

    result = graph.invoke({"messages": [HumanMessage(content="go")]})
    msgs = result["messages"]

    assert len(msgs) == 4, [type(m).__name__ for m in msgs]
    assert isinstance(msgs[0], HumanMessage)
    assert isinstance(msgs[1], AIMessage) and msgs[1].tool_calls
    assert isinstance(msgs[2], ToolMessage)
    assert msgs[2].tool_call_id == "call_1"
    assert msgs[2].content == "value-for-alpha"
    assert isinstance(msgs[3], AIMessage)
    assert msgs[3].content == "done"


def test_langgraph_two_tool_calls_then_terminate():
    """Two consecutive tool calls before the model emits a final answer."""

    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "read_value", "args": {"key": "k1"}, "id": "c1"},
            ],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {"name": "mark_done", "args": {"reason": "ready"}, "id": "c2"},
            ],
        ),
        AIMessage(content="finishing"),
    ]
    graph = _build_smoke_graph(_ToolAwareFakeModel(responses=scripted))
    result = graph.invoke({"messages": [HumanMessage(content="go")]})
    msgs = result["messages"]
    assert len(msgs) == 6
    assert isinstance(msgs[2], ToolMessage) and msgs[2].content == "value-for-k1"
    assert isinstance(msgs[4], ToolMessage) and msgs[4].content == "ack: ready"
    assert msgs[5].content == "finishing"


def test_langgraph_system_prompt_and_history_preserved():
    """The system prompt and the accumulated history reach the model."""

    seen_message_lists: list[list[BaseMessage]] = []

    class _Recorder(_ToolAwareFakeModel):
        def invoke(self, input, config=None, **kwargs):  # type: ignore[override]
            seen_message_lists.append(list(input))
            return super().invoke(input, config=config, **kwargs)

    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {"name": "read_value", "args": {"key": "z"}, "id": "c1"},
            ],
        ),
        AIMessage(content="all good"),
    ]
    model = _Recorder(responses=scripted)
    graph = _build_smoke_graph(model)
    result = graph.invoke(
        {
            "messages": [
                SystemMessage(content="you are helpful"),
                HumanMessage(content="please act"),
            ]
        }
    )
    msgs = result["messages"]
    assert isinstance(msgs[0], SystemMessage)
    assert isinstance(msgs[1], HumanMessage)
    assert msgs[0].content == "you are helpful"

    assert len(seen_message_lists) == 2
    # First invocation: system and human.
    assert len(seen_message_lists[0]) == 2
    # Second invocation: system, human, AI tool call and ToolMessage.
    assert len(seen_message_lists[1]) == 4
    # System prompt survives every turn.
    for seen in seen_message_lists:
        assert isinstance(seen[0], SystemMessage)
        assert seen[0].content == "you are helpful"
