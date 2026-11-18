"""Prefix stability of the assembled request across consecutive turns.

The provider's prompt cache matches from the first token, so the tests
hash each message and assert on the first index where a request stops
matching its predecessor. The conversation is rewritten only on the turn
the agent's mode changes, where the prefix is lost anyway.
"""
import hashlib
import json
from typing import List

import pytest
from langchain_core.messages import (AIMessage, HumanMessage, SystemMessage,
                                     ToolMessage)
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from testuary.agent import _context as context


def _fingerprint(message) -> str:
    """A message's serialized identity, over every field the request carries."""
    payload = json.dumps(
        {
            "type": type(message).__name__,
            "content": message.content,
            "tool_calls": [
                {"name": c.get("name"), "args": c.get("args"),
                 "id": c.get("id")}
                for c in (getattr(message, "tool_calls", None) or [])
            ],
            "additional_kwargs": {
                k: v for k, v in sorted(message.additional_kwargs.items())
            },
            "tool_call_id": getattr(message, "tool_call_id", None),
        },
        sort_keys=True, default=str,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _first_divergence(previous: List[str], current: List[str]) -> int:
    """The index where the served prefix stops matching, or the shared
    length when one is a prefix of the other."""
    for i, (a, b) in enumerate(zip(previous, current)):
        if a != b:
            return i
    return min(len(previous), len(current))


class _Loop:
    """A tool-calling conversation, one turn at a time: an ``AIMessage``
    with a tool call and a reasoning sidecar, then its ``ToolMessage``."""

    def __init__(self, mode: str = "needs_editing"):
        self.messages = [HumanMessage(content="Write tests for calc.py.")]
        self.mode = mode
        self.turn = 0

    def state(self):
        return {"messages": list(self.messages), "agent_mode": self.mode}

    def turn_of(self, tool: str, args: dict, result: str):
        self.turn += 1
        call_id = f"call_{self.turn:04d}"
        self.messages.append(AIMessage(
            content=f"step {self.turn}",
            tool_calls=[{"name": tool, "args": args, "id": call_id}],
            additional_kwargs={
                "reasoning_content": f"thinking about step {self.turn}"},
        ))
        self.messages.append(
            ToolMessage(content=result, tool_call_id=call_id))


def _hook(**overrides):
    """The hook as ``CodingAgent.build_graph`` builds it: same window, dedup
    vocabulary and marker."""
    kwargs = dict(
        system_prompt="SYSTEM PROMPT",
        keep_last=30,
        dedup={
            "read_tools": {"read_source", "read_tests"},
            "file_key": lambda name, args: (
                "source" if name == "read_source" else "test"),
            "stub": context.dedup_stub,
        },
        strip_thinking=True,
    )
    kwargs.update(overrides)
    return context.make_context_hook(
        lambda state: "[state | file=calc.py | tests=green | coverage=80%]",
        **kwargs,
    )


def _run(loop: _Loop, hook, turns, *, read_every=7):
    """Drive ``turns`` turns, returning the per-turn (messages, divergence).

    A recurring read of the same file gives the dedup superseded payloads to
    stub, the rewrite whose deferral the design depends on.
    """
    out = []
    previous = None
    for i in range(turns):
        if i % read_every == 0:
            loop.turn_of("read_source", {}, "SOURCE " + "x" * 200)
        else:
            loop.turn_of("edit_tests", {"old": "a", "new": "b"}, "edit ok")
        sent = hook(loop.state())["messages"]
        current = [_fingerprint(m) for m in sent]
        out.append((sent, None if previous is None
                    else _first_divergence(previous, current)))
        previous = current
    return out


def test_same_mode_turns_only_append():
    """Nothing older than the newly appended pair changes while the mode
    holds, so the request extends its predecessor."""
    loop = _Loop()
    hook = _hook()
    results = _run(loop, hook, 40)

    for turn, (sent, divergence) in enumerate(results):
        if divergence is None:
            continue
        # The rebuilt marker is last, so two appended messages plus the
        # marker may differ.
        assert divergence >= len(sent) - 3, (
            f"turn {turn}: the request diverged from its predecessor at "
            f"index {divergence} of {len(sent)}, so everything from there "
            f"on is re-sent at miss price")


def test_a_mode_change_is_the_only_turn_that_rebuilds():
    """Every rebuild falls on a mode change."""
    loop = _Loop()
    hook = _hook()
    modes = ["needs_editing", "needs_testing", "needs_measurement"]
    rebuilt_on: List[int] = []
    changed_on: List[int] = []
    previous = None
    previous_mode = loop.mode
    for i in range(45):
        if i and i % 5 == 0:
            loop.mode = modes[(i // 5) % len(modes)]
        loop.turn_of("edit_tests", {"old": f"a{i}", "new": f"b{i}"}, "ok")
        sent = hook(loop.state())["messages"]
        current = [_fingerprint(m) for m in sent]
        if previous is not None:
            if _first_divergence(previous, current) < len(sent) - 3:
                rebuilt_on.append(i)
            if loop.mode != previous_mode:
                changed_on.append(i)
        previous = current
        previous_mode = loop.mode

    assert changed_on, "the loop never changed mode"
    unexplained = [i for i in rebuilt_on if i not in changed_on]
    assert not unexplained, (
        f"turn(s) {unexplained} rewrote the conversation without a mode "
        f"change; mode changes were on {changed_on}")


def test_the_marker_is_last_and_always_current():
    """The marker follows the newest message and carries this turn's value."""
    seen = []

    def marker(state):
        seen.append(len(state["messages"]))
        return f"[state | turn={len(state['messages'])}]"

    loop = _Loop()
    hook = context.make_context_hook(marker, system_prompt="SYSTEM",
                                     keep_last=30)
    for _ in range(6):
        loop.turn_of("edit_tests", {}, "ok")
        sent = hook(loop.state())["messages"]
        assert isinstance(sent[-1], SystemMessage)
        assert sent[-1].content == f"[state | turn={len(loop.messages)}]"
    assert len(seen) == 6


def test_the_held_view_is_per_hook_so_parallel_targets_do_not_share_it():
    """Two agents built side by side keep separate conversations."""
    a, b = _Loop(), _Loop()
    hook_a, hook_b = _hook(), _hook()
    a.turn_of("edit_tests", {}, "a-only")
    b.turn_of("edit_tests", {}, "b-only")
    sent_a = hook_a(a.state())["messages"]
    sent_b = hook_b(b.state())["messages"]

    assert any("a-only" == m.content for m in sent_a)
    assert not any("a-only" == m.content for m in sent_b)


def test_the_token_trigger_forces_a_rebuild_without_a_mode_change():
    """The token trigger rebuilds a conversation that grows within one mode."""
    loop = _Loop()
    hook = _hook(rebuild_trigger_tokens=1)
    results = _run(loop, hook, 6)

    diverged_early = [d for _, d in results[1:] if d is not None]
    assert diverged_early, "no turns were compared"
    assert min(diverged_early) < 3, (
        "the trigger never fired: every turn still extended its predecessor")


def test_a_retry_history_rewrite_invalidates_the_held_prefix():
    """A retry that replaces history without a mode change or a shorter list
    rebuilds the held view, so no pre-retry message is sent."""
    hook = _hook()
    original = _Loop()
    original.turn_of("read_source", {}, "ORIGINAL SOURCE")
    hook(original.state())

    replacement = [
        HumanMessage(content="retry prompt"),
        AIMessage(
            content="",
            tool_calls=[{"name": "edit_tests", "args": {}, "id": "retry_1"}],
        ),
        ToolMessage(content="RETRY RESULT", tool_call_id="retry_1"),
    ]
    sent = hook({"messages": replacement, "agent_mode": original.mode})[
        "messages"
    ]

    contents = [str(message.content) for message in sent]
    assert "retry prompt" in contents
    assert "RETRY RESULT" in contents
    assert "ORIGINAL SOURCE" not in contents


def test_a_threshold_rebuild_is_held_on_the_following_turn():
    """After a threshold rebuild compacts below the limit, the next request
    appends to that rebuilt view instead of rebuilding it a second time."""
    loop = _Loop()
    hook = _hook(keep_last=2, rebuild_trigger_tokens=700)
    previous = None
    rebuild_seen = False
    previous_length = 0

    for i in range(8):
        loop.turn_of("edit_tests", {}, f"large result {i} " + "x" * 900)
        sent = hook(loop.state())["messages"]
        current = [_fingerprint(message) for message in sent]
        if previous is not None and len(sent) < previous_length:
            rebuild_seen = True
            previous = current
            break
        previous = current
        previous_length = len(sent)

    assert rebuild_seen, "the synthetic conversation never crossed the trigger"
    loop.turn_of("edit_tests", {}, "small result")
    sent = hook(loop.state())["messages"]
    current = [_fingerprint(message) for message in sent]
    assert _first_divergence(previous, current) >= len(sent) - 3


def test_every_turn_rewrites_every_turn():
    """``every_turn`` rebuilds on every turn, for runs that must stay
    comparable with that assembly."""
    loop = _Loop()
    hook = _hook(rebuild="every_turn")
    results = _run(loop, hook, 40)

    divergences = [d for _, d in results[1:] if d is not None]
    # The system prompt and the kept task prompt lead every request.
    assert min(divergences) <= 2, (
        "with the rebuild running every turn the request should stop "
        "matching its predecessor at the front")


def test_an_unknown_rebuild_policy_is_refused():
    with pytest.raises(ValueError, match="mode_change, every_turn"):
        _hook(rebuild="on_change")


# render_read's last line of a read that showed the whole file.
_WHOLE_READ = "\n[lines 1-1 of 1]"


def test_a_deferred_dedup_still_stubs_at_the_rebuild():
    """A superseded read is stubbed at the next rebuild."""
    loop = _Loop()
    hook = _hook()
    loop.turn_of("read_source", {}, "FIRST COPY" + _WHOLE_READ)
    hook(loop.state())
    for _ in range(3):
        loop.turn_of("edit_tests", {}, "ok")
        hook(loop.state())
    loop.turn_of("read_source", {}, "SECOND COPY" + _WHOLE_READ)
    hook(loop.state())

    loop.mode = "needs_testing"
    loop.turn_of("run_tests", {}, "green")
    sent = hook(loop.state())["messages"]

    bodies = [m.content for m in sent if isinstance(m, ToolMessage)]
    assert "SECOND COPY" + _WHOLE_READ in bodies
    assert "FIRST COPY" + _WHOLE_READ not in bodies
    assert any(context.ELIDED_SENTINEL in b for b in bodies)


def test_an_unanswered_tool_call_at_the_tail_forces_a_rebuild():
    """An unanswered tool call at the tail, which the provider rejects,
    forces a rebuild instead of an append."""
    loop = _Loop()
    hook = _hook()
    loop.turn_of("edit_tests", {}, "ok")
    hook(loop.state())

    loop.messages.append(AIMessage(
        content="dangling",
        tool_calls=[{"name": "edit_tests", "args": {}, "id": "call_dangle"}]))
    sent = hook(loop.state())["messages"]

    assert not context.carries_tool_calls(sent[-1])
    conversation = [m for m in sent if not isinstance(m, SystemMessage)]
    assert not conversation or not context.carries_tool_calls(conversation[-1])


@pytest.mark.parametrize("keep_last", [4, 30])
def test_the_window_still_bounds_the_rebuilt_conversation(keep_last):
    """The rebuild applies the keep-last window."""
    loop = _Loop()
    hook = _hook(keep_last=keep_last)
    for i in range(20):
        loop.mode = f"mode_{i}"
        loop.turn_of("edit_tests", {}, "ok")
        sent = hook(loop.state())["messages"]

    conversation = [m for m in sent if not isinstance(m, SystemMessage)]
    assert len(conversation) <= keep_last


@tool
def _read_source_for_payload() -> str:
    """Read the target source."""
    return "source"


@tool
def _edit_tests_for_payload(content: str) -> str:
    """Replace the test file."""
    return content


def _request_payload(messages, tools):
    """Serialize through the exact ChatOpenAI request-body implementation."""
    from testuary import _chat_models

    model = _chat_models.build_chat_model(
        model="payload-test",
        api_url="http://127.0.0.1:1/v1",
        api_key="unused",
        timeout_seconds=1,
        preserve_reasoning_content=True,
    )
    return model._get_request_payload(  # contract under test
        messages,
        tools=[convert_to_openai_tool(item) for item in tools],
    )


def test_final_serialized_request_keeps_messages_and_ordered_tools_stable():
    """The serialized request body, including tool schema order, stays
    stable, which message hashes alone do not cover."""
    loop = _Loop()
    hook = _hook()
    tools = [_read_source_for_payload, _edit_tests_for_payload]

    loop.turn_of("read_source", {}, "source v1")
    first = _request_payload(hook(loop.state())["messages"], tools)
    loop.turn_of("edit_tests", {"content": "tests v2"}, "ok")
    second = _request_payload(hook(loop.state())["messages"], tools)

    assert [item["function"]["name"] for item in first["tools"]] == [
        "_read_source_for_payload",
        "_edit_tests_for_payload",
    ]
    assert first["tools"] == second["tools"]
    # The previous marker is the first changed message.
    first_messages = first["messages"]
    second_messages = second["messages"]
    divergence = next(
        index
        for index, pair in enumerate(zip(first_messages, second_messages))
        if pair[0] != pair[1]
    )
    assert divergence == len(first_messages) - 1
