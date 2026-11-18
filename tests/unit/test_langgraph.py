"""The LangGraph scaffolding in ``testuary.agent._langgraph``: graph execution,
model configuration and response handling, without live model calls."""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace
from typing import Annotated, Any, List, Sequence

import pytest
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
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field
from typing_extensions import TypedDict

import httpx

from testuary.agent import _langgraph as lg
from testuary import _chat_log
from testuary import _chat_models
from testuary.agent import _context as context
from testuary import model_requests as mr
import subprocess
from pydantic import ValidationError


# Test scaffolding: a tool-aware fake chat model


def _no_classify(state):
    """The graph's classifier, for a test whose loop classifies nothing."""
    return None


class _ToolAwareFakeModel(FakeMessagesListChatModel):
    """Fake chat model that accepts ``bind_tools`` as a no-op."""

    def bind_tools(  # type: ignore[override]
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable:
        return self


# UsageTracker


def test_usage_tracker_no_limit_never_raises():
    tracker = lg.UsageTracker()
    for _ in range(10):
        tracker.note_request()


def test_usage_tracker_request_limit_raises_on_overflow():
    tracker = lg.UsageTracker(request_limit=2)
    tracker.note_request()
    tracker.note_request()
    with pytest.raises(lg.UsageLimitExceeded):
        tracker.note_request()
    assert tracker.requests_used == 2


def test_limit_warning_is_stable_through_final_window():
    tracker = lg.UsageTracker(request_limit=10)
    messages = [HumanMessage(content="task")]

    for _ in range(8):
        tracker.note_request()
    first = lg._append_limit_warning(messages, tracker)
    tracker.note_request()
    second = lg._append_limit_warning(messages, tracker)
    tracker.note_request()
    final = lg._append_limit_warning(messages, tracker)

    assert len(first) == len(messages) + 1
    assert first[-1] == second[-1] == final[-1]
    assert "request(s) left" not in str(first[-1].content)


def test_limit_warning_absent_before_final_window():
    tracker = lg.UsageTracker(request_limit=10)
    tracker.note_request()
    messages = [HumanMessage(content="task")]

    assert lg._append_limit_warning(messages, tracker) is messages


def test_usage_tracker_heartbeat_fires_every_n_requests():
    """The progress heartbeat fires every ``heartbeat_every`` requests with the
    running count and the limit, so a long detached run reports its progress."""
    beats = []
    tracker = lg.UsageTracker(
        request_limit=100, heartbeat_every=3,
        heartbeat_fn=lambda n, lim, el: beats.append((n, lim)),
    )
    for _ in range(7):
        tracker.note_request()
    assert beats == [(3, 100), (6, 100)]


def test_usage_tracker_heartbeat_off_when_zero():
    beats = []
    tracker = lg.UsageTracker(
        request_limit=100, heartbeat_every=0,
        heartbeat_fn=lambda *a: beats.append(a),
    )
    for _ in range(10):
        tracker.note_request()
    assert beats == []


def test_usage_tracker_heartbeat_never_breaks_a_request():
    """A heartbeat is diagnostic: a raising callback aborts neither the
    request nor the run."""
    def boom(*_a):
        raise RuntimeError("heartbeat blew up")
    tracker = lg.UsageTracker(request_limit=100, heartbeat_every=1, heartbeat_fn=boom)
    tracker.note_request()  # must not raise
    assert tracker.requests_used == 1


def test_usage_tracker_note_tokens_accumulates():
    """note_tokens sums both counters across calls. None and 0 caps never
    raise."""
    tracker = lg.UsageTracker()
    tracker.note_tokens(100, 50)
    tracker.note_tokens(40, 10)
    assert tracker.input_tokens_used == 140
    assert tracker.output_tokens_used == 60


def test_usage_tracker_input_token_limit_raises_on_overflow():
    """Cap on prompt tokens fires when cumulative usage hits the cap."""
    tracker = lg.UsageTracker(input_token_limit=200)
    tracker.note_tokens(150, 5)
    with pytest.raises(lg.UsageLimitExceeded, match=r"max_input_tokens_per_phase \(\d+\) exceeded"):
        tracker.note_tokens(60, 5)


def test_usage_tracker_output_token_limit_raises_on_overflow():
    """Cap on completion tokens fires independently of the input cap."""
    tracker = lg.UsageTracker(output_token_limit=100)
    tracker.note_tokens(5_000, 80)
    with pytest.raises(lg.UsageLimitExceeded, match=r"max_output_tokens_per_phase \(100\) exceeded"):
        tracker.note_tokens(0, 30)


def test_usage_tracker_token_caps_of_zero_are_unlimited():
    """A cap of 0 (the settings default) is no cap."""
    tracker = lg.UsageTracker(
        input_token_limit=0,
        output_token_limit=0,
        combined_token_limit=0,
        wall_clock_limit_seconds=0,
    )
    for _ in range(1000):
        tracker.note_tokens(10_000, 5_000)
    # More than 10M tokens, no exception.
    assert tracker.input_tokens_used >= 10_000_000


def test_usage_tracker_combined_token_limit_raises_on_overflow():
    """The combined cap fires on the sum of input and output, not on either
    alone, and raises the same UsageLimitExceeded stop."""
    tracker = lg.UsageTracker(combined_token_limit=100)
    tracker.note_tokens(40, 30)   # 70 combined
    tracker.note_tokens(20, 5)    # 95 combined, still under
    assert tracker.input_tokens_used + tracker.output_tokens_used == 95
    with pytest.raises(lg.UsageLimitExceeded, match=r"max_total_tokens_per_phase \(100\) exceeded"):
        tracker.note_tokens(10, 0)  # 105 combined


def test_usage_tracker_wall_clock_limit_raises_on_overflow():
    """The wall-clock deadline fires from note_request (before the call),
    measured from the first request, as the same limit stop."""
    import time as _time
    tracker = lg.UsageTracker(wall_clock_limit_seconds=10)
    # Simulate the clock having started well in the past.
    tracker._started_monotonic = _time.monotonic() - 100.0
    with pytest.raises(lg.UsageLimitExceeded, match=r"max_wall_clock_seconds_per_phase \(10s\) exceeded"):
        tracker.note_request()


def test_usage_tracker_checks_the_deadline_before_a_retry():
    """A retry charges no model request but may not start after the
    deadline either."""
    import time as _time
    tracker = lg.UsageTracker(wall_clock_limit_seconds=10)
    tracker.note_request()
    tracker._started_monotonic = _time.monotonic() - 100.0
    with pytest.raises(lg.UsageLimitExceeded, match=r"max_wall_clock_seconds_per_phase \(10s\) exceeded"):
        tracker.note_sent_again()
    assert tracker.requests_used == 1


def test_usage_tracker_refuses_a_wait_that_ends_past_the_deadline():
    tracker = lg.UsageTracker(wall_clock_limit_seconds=30)
    tracker.note_request()
    tracker.check_deadline(10)
    with pytest.raises(lg.UsageLimitExceeded, match=r"_per_phase \(30s\) would be exceeded by the retry wait of 40s"):
        tracker.check_deadline(40)
    assert lg.UsageTracker().check_deadline(10_000) is None


def test_usage_tracker_wall_clock_within_limit_does_not_raise():
    """A fresh tracker with a generous deadline counts the request. The
    clock starts on the first request so setup time is not charged."""
    tracker = lg.UsageTracker(wall_clock_limit_seconds=3600)
    tracker.note_request()
    assert tracker.requests_used == 1
    assert tracker._started_monotonic is not None


def test_extract_token_counts_langchain_canonical_shape():
    """LangChain canonical: ``token_usage.prompt_tokens / completion_tokens``."""
    msg = SimpleNamespace(response_metadata={
        "token_usage": {"prompt_tokens": 123, "completion_tokens": 45},
    })
    assert lg._extract_token_counts(msg) == (123, 45)


def test_extract_token_counts_anthropic_shape():
    """Anthropic/Ollama-style: ``input_tokens / output_tokens``."""
    msg = SimpleNamespace(response_metadata={
        "token_usage": {"input_tokens": 200, "output_tokens": 30},
    })
    assert lg._extract_token_counts(msg) == (200, 30)


def test_extract_token_counts_usage_key_variant():
    """Some Ollama versions use ``usage`` instead of ``token_usage`` as the
    outer key, and the extractor reads it too."""
    msg = SimpleNamespace(response_metadata={
        "usage": {"prompt_tokens": 80, "completion_tokens": 12},
    })
    assert lg._extract_token_counts(msg) == (80, 12)


def test_extract_token_counts_langchain_usage_metadata_shape():
    msg = SimpleNamespace(
        response_metadata={},
        usage_metadata={"input_tokens": 90, "output_tokens": 12},
    )
    assert lg._extract_token_counts(msg) == (90, 12)


def test_extract_token_counts_returns_zero_when_metadata_absent():
    """No response_metadata or no token_usage block gives (0, 0), so a
    caller can add the result to the tracker unchecked."""
    assert lg._extract_token_counts(SimpleNamespace()) == (0, 0)
    assert lg._extract_token_counts(
        SimpleNamespace(response_metadata={}),
    ) == (0, 0)
    assert lg._extract_token_counts(
        SimpleNamespace(response_metadata={"token_usage": {}}),
    ) == (0, 0)


def test_extract_cache_hit_tokens_deepseek_field():
    """DeepSeek reports the cached-prefix part as prompt_cache_hit_tokens."""
    msg = SimpleNamespace(response_metadata={"token_usage": {
        "prompt_tokens": 618, "completion_tokens": 5,
        "prompt_cache_hit_tokens": 512, "prompt_cache_miss_tokens": 106}})
    assert lg._extract_cache_hit_tokens(msg) == 512


def test_extract_cache_hit_tokens_openai_cached_tokens():
    """OpenAI-standard nested field prompt_tokens_details.cached_tokens, used
    when the DeepSeek top-level field is absent."""
    msg = SimpleNamespace(response_metadata={"token_usage": {
        "prompt_tokens": 618, "completion_tokens": 5,
        "prompt_tokens_details": {"cached_tokens": 512}}})
    assert lg._extract_cache_hit_tokens(msg) == 512


def test_extract_cache_hit_tokens_langchain_usage_metadata_shape():
    msg = SimpleNamespace(
        response_metadata={},
        usage_metadata={
            "input_tokens": 618,
            "output_tokens": 5,
            "input_token_details": {"cache_read": 512},
        },
    )
    assert lg._extract_cache_hit_tokens(msg) == 512


def test_extract_cache_hit_tokens_unknown_when_provider_omits_it():
    """A provider that reports no cache field (Ollama's compatible endpoint,
    vLLM) leaves the cache split unknown, not zero hits. Malformed shapes
    never raise."""
    assert lg._extract_cache_hit_tokens(SimpleNamespace(response_metadata={
        "token_usage": {"prompt_tokens": 80, "completion_tokens": 12}})) is None
    assert lg._extract_cache_hit_tokens(SimpleNamespace()) is None
    assert lg._extract_cache_hit_tokens(
        SimpleNamespace(response_metadata={"token_usage": None})) is None
    assert lg._extract_cache_hit_tokens(SimpleNamespace(response_metadata={
        "token_usage": {"prompt_tokens": 80, "prompt_cache_hit_tokens": "12"}})) is None


def test_note_tokens_accumulates_cache_hits_separately():
    """note_tokens charges cache-hit tokens into their own running total,
    leaving the input total (their superset) intact. The default is 0."""
    t = lg.UsageTracker()
    t.note_tokens(618, 5, cache_hit_tokens=512)
    t.note_tokens(618, 5)
    assert t.input_tokens_used == 1236
    assert t.output_tokens_used == 10
    assert t.cache_hit_tokens_used == 512


def test_cache_hit_accounting_is_clamped_to_input_tokens():
    t = lg.UsageTracker()
    t.note_tokens(100, 5, cache_hit_tokens=140)
    assert t.input_tokens_used == 100
    assert t.cache_hit_tokens_used == 100


def test_usage_logger_appends_jsonl_line(tmp_path):
    """UsageLogger writes one JSON line per request, append-only."""
    path = tmp_path / "usage.jsonl"
    logger = lg.UsageLogger(path)
    logger.log(
        request_idx=1, input_tokens=10, output_tokens=5, cache_hit_tokens=None,
        model_name="test-model", finish_reason="stop",
    )
    logger.log(request_idx=2, input_tokens=20, output_tokens=8, cache_hit_tokens=None)
    text = path.read_text(encoding="utf-8")
    lines = [ln for ln in text.splitlines() if ln.strip()]
    assert len(lines) == 2
    import json
    first = json.loads(lines[0])
    assert first["request_idx"] == 1
    assert first["input_tokens"] == 10
    assert first["output_tokens"] == 5
    assert first["model_name"] == "test-model"
    assert first["finish_reason"] == "stop"
    assert "ts" not in first
    assert isinstance(first["t_mono_s"], float)
    assert first["clock_domain_id"]
    # Optional fields omitted on the second line stay absent.
    second = json.loads(lines[1])
    assert "model_name" not in second


def test_usage_logger_cache_partition_is_exact_and_clamped(tmp_path):
    import json

    path = tmp_path / "usage.jsonl"
    logger = lg.UsageLogger(path)
    logger.log(
        request_idx=1,
        input_tokens=100,
        output_tokens=5,
        cache_hit_tokens=140,
    )
    row = json.loads(path.read_text(encoding="utf-8"))
    assert row["cache_hit_tokens"] == 100
    assert row["cache_miss_tokens"] == 0
    assert row["cache_hit_tokens"] + row["cache_miss_tokens"] == row["input_tokens"]


def test_usage_logger_disabled_when_path_is_none(tmp_path):
    """``UsageLogger(None)`` is a silent no-op. ``enabled`` is False."""
    logger = lg.UsageLogger(None)
    assert logger.enabled is False
    logger.log(request_idx=1, input_tokens=1, output_tokens=1, cache_hit_tokens=None)


def test_usage_logger_records_redacted_provider_bound_messages(tmp_path):
    import json

    usage_path = tmp_path / "usage.jsonl"
    transport_path = tmp_path / "prompt_transport.jsonl"
    logger = lg.UsageLogger(
        usage_path,
        prompt_transport_path=transport_path,
        prompt_transport_context={
            "phase": "coverage",
            "prompt_template_version": "prompt-v1",
            "provider_transport": "openai_compatible",
        },
    )
    messages = [
        SystemMessage(content="stable system instruction"),
        HumanMessage(content="target-specific input"),
    ]
    logger.log_prompt_transport(request_idx=1, messages=messages)

    raw = transport_path.read_text(encoding="utf-8")
    record = json.loads(raw)
    assert record["roles"] == ["system", "user"]
    assert record["phase"] == "coverage"
    assert record["request_idx"] == 1
    assert len(record["message_sha256"]) == 2
    assert "stable system instruction" not in raw
    assert "target-specific input" not in raw


def test_transport_hash_includes_assistant_tool_call_arguments(tmp_path):
    import json

    path = tmp_path / "prompt_transport.jsonl"
    logger = lg.UsageLogger(
        tmp_path / "usage.jsonl",
        prompt_transport_path=path,
        prompt_transport_context={"phase": "mutation"},
    )
    first = AIMessage(
        content="",
        tool_calls=[{"name": "read_tests", "args": {"page": 1}, "id": "a"}],
    )
    second = AIMessage(
        content="",
        tool_calls=[{"name": "read_tests", "args": {"page": 2}, "id": "a"}],
    )
    logger.log_prompt_transport(request_idx=1, messages=[first])
    logger.log_prompt_transport(request_idx=2, messages=[second])
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert records[0]["roles"] == ["assistant"]
    assert records[0]["message_sha256"] != records[1]["message_sha256"]


# Token usage inside build_react_graph: model_decide charges tokens from the
# response metadata and writes a usage.jsonl line per invocation, and a token
# cap raises UsageLimitExceeded as a max_requests_per_phase overrun does.


def _make_token_wiring_graph(*, scripted, tracker, usage_logger=None):
    """A minimal per-phase graph with one tool that runs ``scripted`` against
    a fake model, charges ``tracker`` and optionally writes per-request lines
    through ``usage_logger``. The model emits ``emit_final_answer`` at once,
    so the graph ends after one invocation."""
    from langchain_core.tools import tool as _tool
    from pydantic import BaseModel as _BaseModel, Field as _Field

    @_tool
    def _noop() -> str:
        """no-op tool."""
        return "ok"

    class _Result(_BaseModel):
        answer: str = _Field(description="answer")

    class _State(TypedDict, total=False):
        messages: Annotated[List[BaseMessage], add_messages]
        result: Any
        phase: str

    return lg.build_react_graph(
        classify_node=_no_classify,
        model=_ToolAwareFakeModel(responses=scripted),
        agent_mode_tools={"p1": [_noop]},
        resolve_agent_mode=lambda _s: "p1",
        state_type=_State,
        structured_output=_Result,
        usage_tracker=tracker,
        usage_logger=usage_logger,
    )


def test_model_decide_charges_tokens_from_response_metadata():
    """After each model invocation, model_decide reads
    ``response_metadata.token_usage`` and passes it to
    ``tracker.note_tokens``. Otherwise every Result.input_tokens the entry
    points write would be 0."""
    scripted = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "emit_final_answer",
                "args": {"answer": "done"},
                "id": "f1",
            }],
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 250,
                    "completion_tokens": 40,
                },
            },
        ),
    ]
    tracker = lg.UsageTracker(request_limit=10)
    graph = _make_token_wiring_graph(scripted=scripted, tracker=tracker)
    graph.invoke({"messages": [HumanMessage(content="go")]})
    assert tracker.requests_used == 1
    assert tracker.input_tokens_used == 250
    assert tracker.output_tokens_used == 40


def test_an_empty_provider_reply_is_retried_and_is_no_model_turn(monkeypatch):
    """An Ollama reply that ended while loading the model holds no text, no
    tool call and 0 of 0 tokens. It is a transient failure: the request is
    sent again, it costs no model request (count_retries is
    off), and the conversation never holds it as a model turn."""
    from testuary import model_requests

    monkeypatch.setattr(model_requests.RequestAccount, "wait", lambda self, seconds: None)
    empty = AIMessage(content="", response_metadata={
        "finish_reason": "load", "token_usage": {"prompt_tokens": 0, "completion_tokens": 0}})
    emit = AIMessage(content="", tool_calls=[{
        "name": "emit_final_answer", "args": {"answer": "done"}, "id": "f1"}],
        response_metadata={"token_usage": {"prompt_tokens": 250, "completion_tokens": 40}})
    tracker = lg.UsageTracker(request_limit=10)
    graph = _make_token_wiring_graph(scripted=[empty, emit], tracker=tracker)
    state = graph.invoke({"messages": [HumanMessage(content="go")]})
    # Two requests sent, one model request.
    assert (tracker.requests_used, tracker.logical_requests_used) == (2, 1)
    assert tracker.output_tokens_used == 40
    # The load reply's reported 0 of 0 tokens are known usage.
    assert tracker.unknown_usage_requests == 0
    assert not any(isinstance(message, AIMessage) and not message.tool_calls
                   and not message.content for message in state["messages"])


def test_an_empty_reply_that_read_the_prompt_is_a_model_turn(monkeypatch):
    """A reply that ended with "stop" after reading the prompt, without text
    or a tool call, is the model's answer. It is not sent again (ten times
    over minutes, then a failed phase) but taken as the turn it is: the turn
    ends without a final answer, which the agent's recovery answers."""
    from testuary import model_requests

    waits = []
    monkeypatch.setattr(model_requests.RequestAccount, "wait",
                        lambda self, seconds: waits.append(seconds))
    empty = AIMessage(content="", response_metadata={
        "finish_reason": "stop", "token_usage": {"prompt_tokens": 812, "completion_tokens": 0}})
    emit = AIMessage(content="", tool_calls=[{
        "name": "emit_final_answer", "args": {"answer": "done"}, "id": "f1"}],
        response_metadata={"token_usage": {"prompt_tokens": 900, "completion_tokens": 40}})
    tracker = lg.UsageTracker(request_limit=10)
    graph = _make_token_wiring_graph(scripted=[empty, emit], tracker=tracker)
    with pytest.raises(lg.OutputValidationError):
        graph.invoke({"messages": [HumanMessage(content="go")]})
    assert waits == []
    assert (tracker.requests_used, tracker.logical_requests_used) == (1, 1)
    assert (tracker.input_tokens_used, tracker.unknown_usage_requests) == (812, 0)


def test_a_reply_with_output_or_a_tool_call_is_not_empty():
    assert not lg._empty_reply(AIMessage(content="ok"))
    assert not lg._empty_reply(AIMessage(content="", tool_calls=[
        {"name": "t", "args": {}, "id": "c1"}]))
    assert not lg._empty_reply(AIMessage(content="", response_metadata={
        "token_usage": {"prompt_tokens": 9, "completion_tokens": 3}}))
    # A reply that read the prompt and stopped is an answer.
    assert not lg._empty_reply(AIMessage(content="  ", response_metadata={
        "finish_reason": "stop", "token_usage": {"prompt_tokens": 9, "completion_tokens": 0}}))
    assert not lg._empty_reply(AIMessage(content=""))
    assert lg._empty_reply(AIMessage(content="  ", response_metadata={
        "token_usage": {"prompt_tokens": 0, "completion_tokens": 0}}))
    assert lg._empty_reply(AIMessage(content="", response_metadata={"done_reason": "load"}))


def test_only_the_latest_reply_s_final_answer_is_judged():
    """An earlier emit was judged when it was made. A later reply without
    one is no new attempt: the model is not told an attempt it did not just
    make was rejected."""
    emit = AIMessage(content="", tool_calls=[
        {"name": "emit_final_answer", "args": {"answer": "old"}, "id": "f1"}])
    rejected = ToolMessage(content="REJECTED", tool_call_id="f1")
    later = AIMessage(content="I will keep editing.")
    assert lg._find_final_answer_call(
        [HumanMessage(content="go"), emit, rejected, later], "emit_final_answer") is None
    again = AIMessage(content="", tool_calls=[
        {"name": "emit_final_answer", "args": {"answer": "new"}, "id": "f2"}])
    assert lg._find_final_answer_call(
        [HumanMessage(content="go"), emit, rejected, later, again],
        "emit_final_answer") == {"answer": "new"}


def test_graph_raises_usage_limit_when_input_token_cap_exceeded():
    """A token cap raises the same ``UsageLimitExceeded`` stop as
    ``max_requests_per_phase``, so the entry points finish the run the same way,
    without another model call."""
    scripted = [
        AIMessage(
            content="please continue",
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 500,
                    "completion_tokens": 50,
                },
            },
        ),
        AIMessage(content="..."),
    ]
    tracker = lg.UsageTracker(request_limit=10, input_token_limit=400)
    graph = _make_token_wiring_graph(scripted=scripted, tracker=tracker)
    with pytest.raises(lg.UsageLimitExceeded, match=r"max_input_tokens_per_phase \(\d+\) exceeded"):
        graph.invoke({"messages": [HumanMessage(content="go")]})


def test_model_decide_writes_usage_jsonl_line_per_invocation(tmp_path):
    """The UsageLogger sidecar gets one line per invocation: request
    index, both token counts, and the model name when present."""
    import json as _json
    log_path = tmp_path / "test_usage.jsonl"
    usage_logger = lg.UsageLogger(log_path)
    scripted = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "emit_final_answer",
                "args": {"answer": "done"},
                "id": "f1",
            }],
            response_metadata={
                "token_usage": {
                    "prompt_tokens": 99,
                    "completion_tokens": 11,
                },
                "model_name": "fake-model-1",
                "finish_reason": "tool_calls",
            },
        ),
    ]
    tracker = lg.UsageTracker(request_limit=10)
    graph = _make_token_wiring_graph(
        scripted=scripted, tracker=tracker, usage_logger=usage_logger,
    )
    graph.invoke({"messages": [HumanMessage(content="go")]})
    lines = [
        _json.loads(line)
        for line in log_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 1
    entry = lines[0]
    assert entry["input_tokens"] == 99
    assert entry["output_tokens"] == 11
    assert entry["model_name"] == "fake-model-1"
    assert entry["finish_reason"] == "tool_calls"


# Error classification


def test_is_context_window_error_substring_match():
    assert lg.is_context_window_error(Exception("token limit exceeded"))
    assert lg.is_context_window_error(Exception("This model's maximum context"))
    assert not lg.is_context_window_error(Exception("nothing relevant"))


def test_is_transient_llm_error_by_type_and_status_only():
    """Transient is decided by exception type and status code. The wording
    of a message is not evidence: a terminal provider error can mention a
    timeout, and a rule that read the wording would retry it."""
    class _With429(Exception):
        status_code = 429

    assert lg.model_requests.is_transient_failure(_With429("Rate limited"))
    assert lg.model_requests.is_transient_failure(httpx.ReadTimeout("operation timed out"))
    assert lg.model_requests.is_transient_failure(ConnectionResetError("connection reset"))
    assert not lg.model_requests.is_transient_failure(Exception("operation timed out"))
    assert not lg.model_requests.is_transient_failure(Exception("connection reset"))
    assert not lg.model_requests.is_transient_failure(Exception("AttributeError"))


def test_instrument_fault_is_never_transient():
    """A MutationInstrumentFault (a broken per-mutant run environment) is
    never retried, even when its message mentions a timeout: a rerun of the
    phase reruns the same broken instrument."""
    from testuary.languages.mutation import MutationInstrumentFault
    assert not lg.model_requests.is_transient_failure(
        MutationInstrumentFault("per-mutant timeout runs died"))


def test_is_transient_llm_error_openai_transport_wrappers():
    """The openai SDK wraps transport failures in its own exception types, not
    httpx subclasses. A dropped connection (APIConnectionError "Connection
    error.") and a request timeout are retryable, also when the wrapper only
    carries the underlying httpx error as its cause."""
    import httpx
    import openai

    req = httpx.Request("POST", "http://x/v1/chat/completions")
    assert lg.model_requests.is_transient_failure(
        openai.APIConnectionError(message="Connection error.", request=req)
    )
    assert lg.model_requests.is_transient_failure(openai.APITimeoutError(request=req))
    # The SDK raises its wrapper `from` the httpx error: caught via the cause.
    wrapped = openai.APIConnectionError(message="Connection error.", request=req)
    wrapped.__cause__ = httpx.ConnectError("refused")
    assert lg.model_requests.is_transient_failure(wrapped)
    # A non-transient error stays non-retryable.
    assert not lg.model_requests.is_transient_failure(ValueError("bad schema"))


def test_is_retryable_agent_error_retries_validation_and_transient_failures():
    """The recovery loop re-invokes the graph for output-validation
    rejections and for any other transient-looking failure escaping the
    graph, such as a tool raising. Model request failures arrive here
    terminal (the model node retried them), and a context-window failure is
    never retried."""
    assert not lg._is_retryable_agent_error(Exception("token limit"))
    assert not lg._is_retryable_agent_error(
        Exception("context window request timed out")
    )
    assert lg._is_retryable_agent_error(httpx.ConnectError("refused"))
    assert lg._is_retryable_agent_error(Exception("worker timed out"))
    assert lg._is_retryable_agent_error(lg.OutputValidationError("bad shape"))
    # A rejection is retried even when its text names the context.
    assert lg._is_retryable_agent_error(lg.OutputValidationError("x context window"))
    assert not lg._is_retryable_agent_error(KeyError("missing field"))
    assert not lg._is_retryable_agent_error(
        mr.TerminalRequestError(httpx.ConnectError("refused")))
    assert not lg._is_retryable_agent_error(
        mr.TransportExhausted("10 consecutive", httpx.ConnectError("x"), 10))
    # A delivered response whose record write failed is not sent again, even
    # when the write failure's text looks transient.
    assert not lg._is_retryable_agent_error(
        mr.RecordNotWritten(None, OSError(110, "Connection timed out")))

    class MutationInstrumentFault(Exception):
        pass

    assert not lg._is_retryable_agent_error(MutationInstrumentFault("run timed out"))


def test_an_output_cap_error_is_no_context_window_error():
    """A provider's refusal of the output cap names max_tokens, not a
    prompt too big for the context, which still counts when named."""
    assert not lg.is_context_window_text(
        "max_tokens is too large: 10000. This model supports at most 4096 "
        "completion tokens")
    assert lg.is_context_window_text(
        "'max_tokens' or 'max_completion_tokens' is too large: 4000. This "
        "model's maximum context length is 4096 tokens and your request has "
        "500 input tokens")




def _three_call_graph(tool, *, serial):
    from pydantic import BaseModel as _BaseModel

    class _Result(_BaseModel):
        answer: str = ""

    class _State(TypedDict, total=False):
        messages: Annotated[List[BaseMessage], add_messages]
        result: Any

    calls = [{"name": tool.name, "args": {"index": i}, "id": f"c{i}"} for i in range(3)]
    tool_mode = {"agent_mode_tools": {"p1": [tool]}, "resolve_agent_mode": lambda _s: "p1"}
    return lg.build_react_graph(
        classify_node=_no_classify,
        model=_ToolAwareFakeModel(responses=[AIMessage(content="", tool_calls=calls)]),
        **tool_mode,
        state_type=_State,
        structured_output=_Result,
        usage_tracker=lg.UsageTracker(request_limit=1),
        serial_tool_calls=serial,
    )


def _tool_results(graph):
    results = []
    with pytest.raises(lg.UsageLimitExceeded):
        for update in graph.stream({"messages": [HumanMessage(content="x")]},
                                   stream_mode="updates"):
            for node in update.values():
                results += [m for m in (node or {}).get("messages", [])
                            if isinstance(m, ToolMessage)]
    return results


def test_serial_tool_calls_run_one_after_another_in_call_order():
    import threading
    import time

    busy = threading.Lock()
    order = []

    @tool
    def exclusive(index: int) -> str:
        """Refuses a call that overlaps another."""
        if not busy.acquire(blocking=False):
            raise RuntimeError("overlapping call")
        try:
            time.sleep(0.05)
            order.append(index)
            return f"done {index}"
        finally:
            busy.release()

    results = _tool_results(_three_call_graph(exclusive, serial=True))
    assert order == [0, 1, 2]
    assert [(m.tool_call_id, m.content) for m in results] == [
        ("c0", "done 0"), ("c1", "done 1"), ("c2", "done 2")]


def test_tool_calls_run_in_parallel_unless_serial():
    import threading

    barrier = threading.Barrier(3, timeout=5)

    @tool
    def together(index: int) -> str:
        """Passes only when all three calls run at once."""
        barrier.wait()
        return f"done {index}"

    results = _tool_results(_three_call_graph(together, serial=False))
    assert [(m.tool_call_id, m.content) for m in results] == [
        ("c0", "done 0"), ("c1", "done 1"), ("c2", "done 2")]


# The retry hint


class _OutageThenScriptModel(_ToolAwareFakeModel):
    """Raises the scripted failures, one per request, then answers from the
    script. Records the messages of every request."""

    failures: List[Any] = []
    requests_seen: int = 0
    seen_messages: List[Any] = []

    def _generate(self, messages, *args: Any, **kwargs: Any):  # type: ignore[override]
        self.requests_seen += 1
        self.seen_messages = list(self.seen_messages) + [list(messages)]
        if self.requests_seen <= len(self.failures):
            raise self.failures[self.requests_seen - 1]
        return super()._generate(messages, *args, **kwargs)


class _PolicyResult(BaseModel):
    answer: str = Field(description="answer")


class _PolicyState(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    result: Any
    phase: str


@tool
def _policy_noop() -> str:
    """no-op tool."""
    return "ok"


def _final_answer_message() -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "emit_final_answer", "args": {"answer": "done"}, "id": "f1",
        }],
        response_metadata={
            "token_usage": {"prompt_tokens": 250, "completion_tokens": 40},
        },
    )


def _make_policy_graph(model, account, *, tracker):
    """A minimal per-phase graph whose model node runs under ``account``."""
    return lg.build_react_graph(
        classify_node=_no_classify,
        model=model,
        agent_mode_tools={"p1": [_policy_noop]},
        resolve_agent_mode=lambda _s: "p1",
        state_type=_PolicyState,
        structured_output=_PolicyResult,
        usage_tracker=tracker,
        request_account=account,
    )


class _FakeResponse:
    """Stand-in for the httpx ``Response`` of an openai-python
    ``APIStatusError``-like exception: headers and a body that ``json()``
    returns."""

    def __init__(self, headers=None, body=None, text=None):
        self.headers = headers or {}
        self._body = body
        self.text = text or ""

    def json(self):
        if isinstance(self._body, Exception):
            raise self._body
        return self._body


def _exc_with_response(headers=None, body=None, text=None) -> Exception:
    exc = Exception("rate limited")
    exc.response = _FakeResponse(headers=headers, body=body, text=text)
    return exc


def test_the_retry_hint_reads_top_level_keys_and_the_response_text():
    """``model_requests.retry_hint_seconds`` gives the retry rule its wait
    (its own tests cover headers, dates, bounds and malformed values). A
    top-level camelCase key of the body counts, an error body without a
    hint gives None, and the response text is read when ``response.json()``
    fails."""
    assert mr.retry_hint_seconds(_exc_with_response(body={"retryAfter": 11})) == 11.0
    assert mr.retry_hint_seconds(_exc_with_response(body={"error": "rate limit"})) is None
    exc = Exception("rate limited")
    exc.response = _FakeResponse(body=ValueError("not parsed"), text='{"retry_after": 3}')
    assert mr.retry_hint_seconds(exc) == 3.0


# run_graph_with_retries: a server's wait hint before the backoff


def test_a_transient_failure_escaping_the_graph_reruns_it_from_its_state(monkeypatch):
    """A transient-looking failure that escapes the graph (not a model
    request) re-runs the graph from its captured state after the failure's
    hint."""
    seq = {"n": 0}

    class _Graph:
        def stream(self, state, config=None, stream_mode=None):
            seq["n"] += 1
            if seq["n"] == 1:
                exc = Exception("rate limited")
                exc.status_code = 429
                exc.response = _FakeResponse(headers={"retry-after": "17"})
                raise exc
            yield {"messages": list(state["messages"]), "result": "done"}

    sleeps: list[float] = []
    monkeypatch.setattr(lg.time, "sleep", sleeps.append)
    res, err = lg.run_graph_with_retries(
        _Graph(),
        {"messages": [HumanMessage(content="x")]},
        max_retries=3,
    )
    assert err is None and res["result"] == "done"
    assert seq["n"] == 2
    assert sleeps == [17.0]


def test_a_rerun_without_a_hint_waits_by_the_doubling_backoff(monkeypatch):
    class _Graph:
        def stream(self, state, config=None, stream_mode=None):
            raise RuntimeError("tool subprocess timed out")
            yield  # pragma: no cover

    sleeps: list[float] = []
    monkeypatch.setattr(lg.time, "sleep", sleeps.append)
    res, err = lg.run_graph_with_retries(
        _Graph(), {"messages": [HumanMessage(content="x")]},
        max_retries=4,
    )
    assert res is None and err.startswith("max_retries (4) reached")
    assert sleeps == [10, 20, 40]


def _tool_failure_graph(flaky_error):
    """Two tools in one response: ``fine`` answers, ``flaky`` raises
    ``flaky_error`` once. The model records each request's messages."""
    from pydantic import BaseModel as _BaseModel

    class _Result(_BaseModel):
        answer: str = ""

    class _State(TypedDict, total=False):
        messages: Annotated[List[BaseMessage], add_messages]
        result: Any

    calls = {"fine": 0, "flaky": 0}
    requests: list = []

    @tool
    def fine() -> str:
        """Answers."""
        calls["fine"] += 1
        return "fine result"

    @tool
    def flaky() -> str:
        """Fails once."""
        calls["flaky"] += 1
        if calls["flaky"] == 1:
            raise flaky_error
        return "flaky result"

    class _Recording(_ToolAwareFakeModel):
        def invoke(self, input, config=None, **kwargs):
            requests.append(list(input))
            return super().invoke(input, config, **kwargs)

    graph = lg.build_react_graph(
        classify_node=_no_classify,
        model=_Recording(responses=[
            AIMessage(content="", tool_calls=[
                {"name": "fine", "args": {}, "id": "c0"},
                {"name": "flaky", "args": {}, "id": "c1"}]),
            AIMessage(content="", tool_calls=[
                {"name": "emit_final_answer", "args": {"answer": "done"}, "id": "f1"}]),
        ]),
        agent_mode_tools={"p1": [fine, flaky]},
        resolve_agent_mode=lambda _s: "p1",
        state_type=_State,
        structured_output=_Result,
        serial_tool_calls=True,
    )
    return graph, calls, requests


def test_a_tool_failure_rerun_resends_every_call_answered(monkeypatch):
    """A tool raising a temporary error reruns the graph after the backoff,
    and the resent history answers every call of the failed
    response: the failed call with the failure result, the others with their
    own results. Strict providers refuse a resent history with unanswered
    tool calls."""
    graph, calls, requests = _tool_failure_graph(httpx.ConnectError("refused"))
    sleeps: list[float] = []
    monkeypatch.setattr(lg.time, "sleep", sleeps.append)
    res, err = lg.run_graph_with_retries(
        graph, {"messages": [HumanMessage(content="x")]},
        max_retries=3,
    )
    assert err is None and res is not None
    assert sleeps == [10]
    assert calls == {"fine": 1, "flaky": 1}
    resent = requests[1]
    answers = {m.tool_call_id: m.content for m in resent if isinstance(m, ToolMessage)}
    assert answers == {
        "c0": "fine result",
        "c1": "The tool failed with a temporary error (ConnectError) and may not have "
              "completed. Check whether it took effect before calling it again.",
    }
    calling = next(i for i, m in enumerate(resent)
                   if isinstance(m, AIMessage) and m.tool_calls)
    assert [m.tool_call_id for m in resent[calling + 1:calling + 3]] == ["c0", "c1"]


def test_a_tool_failure_rerun_waits_for_the_tools_retry_hint(monkeypatch):
    """The wait after a temporary tool failure honors the Retry-After hint of
    the tool's own exception, not only the doubling backoff."""
    limited = Exception("rate limited")
    limited.status_code = 429
    limited.response = _FakeResponse(headers={"retry-after": "3"})
    graph, _calls, _requests = _tool_failure_graph(limited)
    sleeps: list[float] = []
    monkeypatch.setattr(lg.time, "sleep", sleeps.append)
    res, err = lg.run_graph_with_retries(
        graph, {"messages": [HumanMessage(content="x")]},
        max_retries=3,
    )
    assert err is None and res is not None
    assert sleeps == [3.0]


def test_a_tool_failure_that_is_not_temporary_still_ends_the_run(monkeypatch):
    graph, calls, requests = _tool_failure_graph(KeyError("missing field"))
    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    res, err = lg.run_graph_with_retries(
        graph, {"messages": [HumanMessage(content="x")]},
        max_retries=3,
    )
    assert res is None and err.startswith("non-retryable error: KeyError")
    assert len(requests) == 1


def test_model_node_waits_by_the_server_hint_then_by_backoff():
    """Inside the model node, a transient failure with a ``Retry-After``
    header waits exactly that long. One without a hint follows the doubling
    schedule of the phase's settings, indexed by consecutive failures, so a
    hinted first failure still counts as the first step."""
    class _Hinted(Exception):
        status_code = 429

        def __init__(self):
            super().__init__("rate limited")
            self.response = _FakeResponse(headers={"retry-after": "17"})

    waits: list[float] = []
    account = mr.RequestAccount(mr.RetrySettings(), wait_fn=waits.append)
    model = _OutageThenScriptModel(
        responses=[_final_answer_message()],
        failures=[_Hinted(), httpx.ConnectError("refused"),
                  httpx.ConnectError("refused")],
    )
    graph = _make_policy_graph(model, account, tracker=None)

    graph.invoke({"messages": [HumanMessage(content="go")]})

    assert waits == [17.0, 20.0, 40.0]
    assert model.requests_seen == 4


def test_run_graph_with_retries_no_wait_on_validation_rejection(monkeypatch):
    """A validator rejection (``OutputValidationError``) is a deterministic
    signal the framework raised itself, not a rate limit or transport fault.
    The recovery loop re-invokes at once, without the backoff sleep of the
    transient-error path."""
    sleeps: list[float] = []
    seq = {"n": 0}

    class _Graph:
        def stream(self, state, config=None, stream_mode=None):
            seq["n"] += 1
            if seq["n"] == 1:
                raise lg.OutputValidationError("premature CoverageTestResult")
            yield {"messages": [HumanMessage(content="ok")]}

    monkeypatch.setattr(lg.time, "sleep", sleeps.append)
    res, err = lg.run_graph_with_retries(
        _Graph(),
        {"messages": [HumanMessage(content="x")]},
        max_retries=3,
    )
    assert err is None
    assert res == {"messages": [HumanMessage(content="ok")]}
    # The validator rejection retried without any backoff sleep.
    assert sleeps == []


# build_react_graph end-to-end


@tool
def _stub_read() -> str:
    """Stub read tool."""
    return "source-bytes-here"


@tool
def _stub_write(content: str) -> str:
    """Stub write tool."""
    return f"wrote {len(content)} bytes"


class _StubGraphState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    result: Any


class _StubResult(BaseModel):
    score: int = Field(ge=0, le=100)
    note: str


def test_build_react_graph_with_structured_output_terminates():
    scripted = [
        AIMessage(
            content="",
            tool_calls=[{"name": "_stub_read", "args": {}, "id": "r1"}],
        ),
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "emit_final_answer",
                    "args": {"score": 42, "note": "ok"},
                    "id": "f1",
                }
            ],
        ),
    ]
    model = _ToolAwareFakeModel(responses=scripted)
    graph = lg.build_react_graph(
        classify_node=_no_classify,
        model=model,
        agent_mode_tools={"p": [_stub_read, _stub_write]},
        resolve_agent_mode=lambda s: "p",
        structured_output=_StubResult,
        state_type=_StubGraphState,
    )
    final_state = graph.invoke(
        {"messages": [HumanMessage(content="please score")]}
    )
    assert "result" in final_state
    assert isinstance(final_state["result"], _StubResult)
    assert final_state["result"].score == 42
    assert final_state["result"].note == "ok"


def test_make_context_hook_drives_a_multi_turn_loop_to_completion():
    """The live context hook (keep-N window, deduplication, marker) in a real
    graph loop: many turns beyond keep_last, with repeated reads of one file,
    still reach a structured result."""
    # Six reads of one file (more than keep_last=3), then the final answer.
    reads = [
        AIMessage(
            content="",
            tool_calls=[{"name": "_stub_read", "args": {}, "id": f"r{i}"}],
        )
        for i in range(6)
    ]
    final = AIMessage(
        content="",
        tool_calls=[
            {
                "name": "emit_final_answer",
                "args": {"score": 50, "note": "done"},
                "id": "f1",
            }
        ],
    )
    model = _ToolAwareFakeModel(responses=reads + [final])
    slots = {"_stub_read": "source"}
    hook = context.make_context_hook(
        lambda s: "[state | file=foo.py | focus=parse]",
        keep_last=3,
        dedup={
            "read_tools": {"_stub_read"},
            "file_key": lambda name, args: slots.get(name, name),
            "stub": context.dedup_stub,
        },
        strip_thinking=True,
    )
    graph = lg.build_react_graph(
        classify_node=_no_classify,
        model=model,
        agent_mode_tools={"p": [_stub_read, _stub_write]},
        resolve_agent_mode=lambda s: "p",
        structured_output=_StubResult,
        state_type=_StubGraphState,
        pre_model_hook=hook,
    )
    final_state = graph.invoke({"messages": [HumanMessage(content="go")]})
    assert isinstance(final_state["result"], _StubResult)
    assert final_state["result"].score == 50
    # The full trajectory is preserved in state (the hook windows only the
    # model's input, not the persisted messages).
    assert len(final_state["messages"]) > 6


def test_build_react_graph_converts_pydantic_validation_error_to_retryable():
    """A final-answer call that does not match the schema raises the
    retryable OutputValidationError, not Pydantic's ValidationError, so the
    model can fix its output instead of failing the run."""
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "emit_final_answer",
                    # score is required (1-100), missing here.
                    "args": {"note": "no score"},
                    "id": "f1",
                }
            ],
        ),
    ]
    model = _ToolAwareFakeModel(responses=scripted)
    graph = lg.build_react_graph(
        classify_node=_no_classify,
        model=model,
        agent_mode_tools={"p": [_stub_read]},
        resolve_agent_mode=lambda s: "p",
        structured_output=_StubResult,
        state_type=_StubGraphState,
    )
    with pytest.raises(lg.OutputValidationError) as ei:
        graph.invoke({"messages": [HumanMessage(content="x")]})
    # The Pydantic error message survives, so the recovery prompt can show
    # the model what was wrong.
    assert "rejected" in str(ei.value).lower() or "score" in str(ei.value).lower()


def test_build_react_graph_calls_output_validator_and_raises_on_reject():
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "emit_final_answer",
                    "args": {"score": 1, "note": "thin"},
                    "id": "f1",
                }
            ],
        ),
    ]
    model = _ToolAwareFakeModel(responses=scripted)

    def reject_thin_notes(state: Any, parsed: _StubResult) -> str | None:
        if len(parsed.note) < 5:
            return "note too short"
        return None

    graph = lg.build_react_graph(
        classify_node=_no_classify,
        model=model,
        agent_mode_tools={"p": [_stub_read]},
        resolve_agent_mode=lambda s: "p",
        structured_output=_StubResult,
        output_validator=reject_thin_notes,
        state_type=_StubGraphState,
    )
    with pytest.raises(lg.OutputValidationError):
        graph.invoke({"messages": [HumanMessage(content="x")]})


def test_build_react_graph_writes_trace_events_through_a_full_run(tmp_path):
    """A scripted run with the trace logger attached emits the expected
    events: what each model call returned (model_invoke events), the
    reasoning_content length (model_invoke_end.reasoning_content_len), and
    what the validator did (validator_check events)."""
    import json as _json

    trace_path = tmp_path / "trace.jsonl"
    trace = lg.TraceLogger(trace_path)

    scripted = [
        AIMessage(
            content="",
            tool_calls=[{"name": "_stub_read", "args": {}, "id": "c1"}],
            response_metadata={"finish_reason": "tool_calls"},
            additional_kwargs={"reasoning_content": "thought about the problem"},
        ),
        AIMessage(
            content="",
            tool_calls=[{
                "name": "emit_final_answer",
                "args": {"score": 50, "note": "ok seven"},
                "id": "f1",
            }],
            response_metadata={"finish_reason": "tool_calls"},
        ),
    ]
    model = _ToolAwareFakeModel(responses=scripted)
    graph = lg.build_react_graph(
        classify_node=_no_classify,
        model=model,
        agent_mode_tools={"p": [_stub_read, _stub_write]},
        resolve_agent_mode=lambda s: "p",
        structured_output=_StubResult,
        state_type=_StubGraphState,
        trace_logger=trace,
    )
    graph.invoke({"messages": [HumanMessage(content="go")]})

    rows = [
        _json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
    ]
    events = [r["event"] for r in rows]

    # At minimum: 2 model invocations, 1 validate (accept).
    assert events.count("model_invoke_start") == 2
    assert events.count("model_invoke_end") == 2
    assert events.count("validator_check") == 1

    # The first model_invoke_end captures reasoning_content length.
    first_end = next(r for r in rows if r["event"] == "model_invoke_end")
    assert first_end["reasoning_content_len"] > 0
    assert "_stub_read" in first_end["tool_calls"]
    assert first_end["finish_reason"] == "tool_calls"

    # Validator accepted at the end.
    val = next(r for r in rows if r["event"] == "validator_check")
    assert val["accepted"] is True


def test_build_react_graph_calls_pre_model_hook_exactly_once_per_turn():
    """The pre_model_hook (state markers, dynamic prompts, counters) runs
    once per model_decide turn, so its side effects are not doubled."""
    call_count = [0]

    def hook(state):
        call_count[0] += 1
        return None  # No update needed

    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "emit_final_answer",
                    "args": {"score": 1, "note": "hello"},
                    "id": "f1",
                }
            ],
        ),
    ]
    model = _ToolAwareFakeModel(responses=scripted)
    graph = lg.build_react_graph(
        classify_node=_no_classify,
        model=model,
        agent_mode_tools={"p": [_stub_read]},
        resolve_agent_mode=lambda s: "p",
        pre_model_hook=hook,
        structured_output=_StubResult,
        state_type=_StubGraphState,
    )
    graph.invoke({"messages": [HumanMessage(content="x")]})
    # One model_decide turn before validate: exactly one hook call.
    assert call_count[0] == 1


def test_build_react_graph_pre_hook_state_update_propagates_to_graph():
    """Keys other than messages that the pre_model_hook returns reach the
    graph state, so a hook can count or record a stage."""

    class _CounterState(TypedDict, total=False):
        messages: Annotated[List[BaseMessage], add_messages]
        turn_count: int
        result: Any

    def hook(state):
        return {"turn_count": (state.get("turn_count") or 0) + 1}

    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "emit_final_answer",
                    "args": {"score": 1, "note": "ok123"},
                    "id": "f1",
                }
            ],
        ),
    ]
    model = _ToolAwareFakeModel(responses=scripted)
    graph = lg.build_react_graph(
        classify_node=_no_classify,
        model=model,
        agent_mode_tools={"p": [_stub_read]},
        resolve_agent_mode=lambda s: "p",
        pre_model_hook=hook,
        structured_output=_StubResult,
        state_type=_CounterState,
    )
    final = graph.invoke({"messages": [HumanMessage(content="go")]})
    # Counter propagated through the model_decide return.
    assert final.get("turn_count") == 1


def test_run_graph_with_retries_returns_success_first_try():
    class _Graph:
        def stream(self, state, config=None, stream_mode=None):
            yield {"result": "ok", "messages": []}

    res, err = lg.run_graph_with_retries(
        _Graph(), {"messages": []}, max_retries=3,
    )
    assert res == {"result": "ok", "messages": []}
    assert err is None


def test_run_graph_with_retries_passes_through_usage_limit_exceeded():
    class _Graph:
        def stream(self, state, config=None, stream_mode=None):
            raise lg.UsageLimitExceeded("hit it")
            yield  # pragma: no cover -- never reached

    with pytest.raises(lg.UsageLimitExceeded):
        lg.run_graph_with_retries(_Graph(), {"messages": []}, max_retries=3)


def test_transient_retries_inside_the_node_do_not_consume_the_request_limit(monkeypatch):
    """A dropped attempt is a request sent only. With a limit of two
    model requests, three dropped attempts before delivery leave one
    model request spent and the run continuing."""
    tracker = lg.UsageTracker(request_limit=2)
    account = mr.RequestAccount(mr.RetrySettings(), wait_fn=lambda _s: None)
    model = _OutageThenScriptModel(
        responses=[_final_answer_message()],
        failures=[httpx.ConnectError("a")] * 3,
    )
    graph = _make_policy_graph(model, account, tracker=tracker)

    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    final_state, err = lg.run_graph_with_retries(
        graph, {"messages": [HumanMessage(content="go")]},
        max_retries=3,
    )

    assert err is None
    assert final_state["result"].answer == "done"
    assert model.requests_seen == 4
    assert tracker.requests_used == 4
    assert tracker.logical_requests_used == 1
    assert tracker.unknown_usage_requests == 3


def test_charging_every_attempt_consumes_the_request_limit(monkeypatch):
    """With charge_every_attempt every attempt is charged, so an outage
    can spend the whole request limit without a delivered response."""
    tracker = lg.UsageTracker(request_limit=2)
    account = mr.RequestAccount(
        mr.RetrySettings(charge_every_attempt=True), wait_fn=lambda _s: None)
    model = _OutageThenScriptModel(
        responses=[_final_answer_message()],
        failures=[httpx.ConnectError("a")] * 5,
    )
    graph = _make_policy_graph(model, account, tracker=tracker)

    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    with pytest.raises(lg.UsageLimitExceeded, match=r"max_requests_per_phase \(2\)"):
        lg.run_graph_with_retries(
            graph, {"messages": [HumanMessage(content="go")]},
            max_retries=3,
        )
    assert model.requests_seen == 2
    assert tracker.requests_used == 2
    assert tracker.logical_requests_used == 2


def test_run_graph_with_retries_returns_error_on_non_retryable():
    class _Graph:
        def stream(self, state, config=None, stream_mode=None):
            raise KeyError("not retryable")
            yield  # pragma: no cover

    res, err = lg.run_graph_with_retries(_Graph(), {"messages": []}, max_retries=5)
    assert res is None
    assert err is not None
    assert "non-retryable" in err


def test_run_graph_with_retries_does_not_reinvoke_on_a_failed_model_request(monkeypatch):
    class _Graph:
        attempts = 0

        def stream(self, state, config=None, stream_mode=None):
            type(self).attempts += 1
            raise mr.TerminalRequestError(httpx.ConnectError("refused"))
            yield  # pragma: no cover

    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    res, err = lg.run_graph_with_retries(
        _Graph(),
        {"messages": [HumanMessage(content="x")]},
        max_retries=5,
    )
    assert res is None
    assert "non-retryable error: TerminalRequestError" in err
    assert _Graph.attempts == 1


def test_run_graph_with_retries_captures_intermediate_state_via_callback(monkeypatch):
    """The on_intermediate_state callback fires per super-step, also on a
    hard failure, so the chat log is captured on LangGraph's error paths."""
    captured: list[dict] = []

    class _Graph:
        def stream(self, state, config=None, stream_mode=None):
            yield {"messages": ["first"]}
            yield {"messages": ["first", "second"]}
            raise Exception("operation timed out")
            yield  # pragma: no cover

    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    lg.run_graph_with_retries(
        _Graph(),
        {"messages": []},
        max_retries=1,
        on_intermediate_state=lambda s: captured.append(s),
    )
    # Two yielded states observed before the exception.
    assert len(captured) == 2
    assert captured[-1] == {"messages": ["first", "second"]}


def test_run_graph_with_retries_snapshots_full_state_before_retry(monkeypatch):
    snapshots = []

    class _Graph:
        attempts = 0

        def stream(self, state, config=None, stream_mode=None):
            type(self).attempts += 1
            if type(self).attempts == 1:
                messages = list(state["messages"]) + [
                    AIMessage(
                        content="",
                        tool_calls=[{
                            "name": "tool_a", "args": {"payload": "full"},
                            "id": "c1",
                        }],
                    ),
                    ToolMessage(
                        content="complete tool return",
                        tool_call_id="c1",
                        name="tool_a",
                    ),
                ]
                yield {"messages": messages}
                raise lg.OutputValidationError("rejected final answer")
            yield {"result": "recovered", "messages": list(state["messages"])}

    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    result, err = lg.run_graph_with_retries(
        _Graph(),
        {"messages": [HumanMessage(content="task")]},
        max_retries=3,
        on_before_retry=lambda attempt, state, exc: snapshots.append(
            (attempt, state, exc)
        ),
    )

    assert err is None
    assert result is not None and result["result"] == "recovered"
    assert len(snapshots) == 1
    attempt, state, exc = snapshots[0]
    assert attempt == 1
    assert isinstance(exc, lg.OutputValidationError)
    assert len(state["messages"]) == 3
    assert state["messages"][-1].content == "complete tool return"


def test_retry_chat_snapshotter_writes_full_failed_attempt(tmp_path):
    snapshot = _chat_log.make_retry_chat_snapshotter(
        output_dir=str(tmp_path),
        filename_prefix="mutation_chat",
        project_name="proj",
        extra_metadata_factory=lambda _stage: {"target": "module.py"},
    )
    messages = [
        HumanMessage(content="task"),
        AIMessage(
            content="analysis result",
            tool_calls=[{
                "name": "measure", "args": {"suite": "all tests"},
                "id": "m1",
            }],
        ),
        ToolMessage(
            content="full measurement payload",
            tool_call_id="m1",
            name="measure",
        ),
    ]

    snapshot(
        2,
        {"messages": messages},
        lg.OutputValidationError("invalid result"),
    )

    # Files are numbered in the order written, so a second snapshot of the
    # same attempt (sent again) does not overwrite the first.
    snapshot(2, {"messages": messages}, RuntimeError("transport"))
    assert json.loads((tmp_path / "mutation_chat_pre_retry_002.json").read_text(
        encoding="utf-8"))["metadata"]["error_type"] == "RuntimeError"
    path = tmp_path / "mutation_chat_pre_retry_001.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["metadata"]["stage"] == "pre_retry"
    assert payload["metadata"]["failed_attempt"] == 2
    assert payload["metadata"]["next_attempt"] == 3
    assert payload["metadata"]["error_type"] == "OutputValidationError"
    assert payload["metadata"]["target"] == "module.py"
    assert payload["messages"][1]["tool_calls"][0]["args"] == {
        "suite": "all tests"
    }
    assert payload["messages"][2]["content"] == "full measurement payload"


def test_transient_retry_in_the_node_resends_the_same_messages_without_compaction(monkeypatch):
    """A request sent again repeats the same request. The node does not leave
    the graph, so nothing captured is lost, and recovery invokes neither
    context compaction primitive."""
    from testuary.agent import _context as ctx

    real_drop = ctx.drop_oldest_pair_safe
    real_strip = ctx.strip_old_thinking

    def _unexpected(*_args, **_kwargs):
        raise AssertionError("recovery invoked the context compactor")

    ctx.drop_oldest_pair_safe = _unexpected
    ctx.strip_old_thinking = _unexpected
    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    try:
        account = mr.RequestAccount(mr.RetrySettings(), wait_fn=lambda _s: None)
        model = _OutageThenScriptModel(
            responses=[_final_answer_message()],
            failures=[httpx.ConnectError("refused")],
        )
        graph = _make_policy_graph(model, account, tracker=None)
        result, err = lg.run_graph_with_retries(
            graph, {"messages": [HumanMessage(content="task")]},
            max_retries=3,
        )
        assert err is None
        assert result["result"].answer == "done"
        assert len(model.seen_messages) == 2
        assert [m.content for m in model.seen_messages[0]] == ["task"]
        assert model.seen_messages[1] == model.seen_messages[0]
    finally:
        ctx.drop_oldest_pair_safe = real_drop
        ctx.strip_old_thinking = real_strip


def test_validation_retry_preserves_history_and_answers_rejected_tool_call(monkeypatch):
    seen_retry_states: list[dict] = []

    class _Graph:
        attempts = 0

        def stream(self, state, config=None, stream_mode=None):
            type(self).attempts += 1
            if type(self).attempts == 1:
                yield {
                    "messages": list(state["messages"]) + [AIMessage(
                        content="final reasoning",
                        tool_calls=[{
                            "name": "emit_mutation_test_result",
                            "args": {"status": "ok"},
                            "id": "final-1",
                        }],
                    )],
                    "agent_mode": "needs_editing",
                    "framework_counter": 11,
                }
                raise lg.OutputValidationError("mutation goal not reached")
            seen_retry_states.append(dict(state))
            yield {**state, "result": "done"}

    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    result, err = lg.run_graph_with_retries(
        _Graph(),
        {"messages": [HumanMessage(content="task")]},
        max_retries=3,
        retry_prompt_builder=lambda exc, _history: f"REJECTED: {exc}",
    )

    assert err is None
    assert result is not None and result["result"] == "done"
    assert len(seen_retry_states) == 1
    retried = seen_retry_states[0]
    assert retried["agent_mode"] == "needs_editing"
    assert retried["framework_counter"] == 11
    messages = retried["messages"]
    assert len(messages) == 4
    assert isinstance(messages[1], AIMessage)
    assert messages[1].content == "final reasoning"
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].tool_call_id == "final-1"
    assert isinstance(messages[3], HumanMessage)
    assert messages[3].content == "REJECTED: mutation goal not reached"


def test_context_window_error_stops_without_retry_or_state_rewrite(monkeypatch):
    calls = 0
    builder_calls = 0

    class _Graph:
        def stream(self, state, config=None, stream_mode=None):
            nonlocal calls
            calls += 1
            raise Exception("maximum context length exceeded")
            yield  # pragma: no cover

    def builder(_exc, _history):
        nonlocal builder_calls
        builder_calls += 1
        return "must not be called"

    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    result, err = lg.run_graph_with_retries(
        _Graph(),
        {"messages": [HumanMessage(content="task")]},
        max_retries=5,
        retry_prompt_builder=builder,
    )

    assert result is None
    assert err is not None and "context length" in err
    assert calls == 1
    assert builder_calls == 0


def test_validation_retry_fails_closed_when_tool_call_has_no_identifier(monkeypatch):
    class _Graph:
        attempts = 0

        def stream(self, state, config=None, stream_mode=None):
            type(self).attempts += 1
            yield {"messages": list(state["messages"]) + [AIMessage(
                content="",
                tool_calls=[{
                    "name": "emit_final_answer",
                    "args": {"status": "ok"},
                    "id": None,
                }],
            )]}
            raise lg.OutputValidationError("rejected")

    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    result, err = lg.run_graph_with_retries(
        _Graph(),
        {"messages": [HumanMessage(content="task")]},
        max_retries=3,
    )

    assert result is None
    assert err is not None and "has no id" in err
    assert _Graph.attempts == 1


# Chat-log export


def test_build_chat_model_default_has_no_reasoning_sidecar():
    from langchain_openai import ChatOpenAI

    m = _chat_models.build_chat_model(
        model="test-model",
        api_url="http://localhost:9999/v1",
        api_key="dummy",
        timeout_seconds=1.0,
    )
    # Without preserve_reasoning_content, build_chat_model builds no
    # reasoning sidecar. The agents turn it on through their own builder.
    assert type(m) is _chat_models._ChatOpenAIExactSampling
    assert isinstance(m, ChatOpenAI) and not isinstance(m, _chat_models._ChatOpenAIWithReasoning)


def test_build_chat_model_empty_api_key_falls_back_to_empty_literal():
    """An empty or false api_key becomes the literal "EMPTY", so the
    Authorization header is ``Bearer EMPTY``, which vLLM and LiteLLM proxies
    accept, not ``Bearer ``, which they answer with 401."""
    for empty_value in ("", None):
        m = _chat_models.build_chat_model(
            model="test-model",
            api_url="http://localhost:9999/v1",
            api_key=empty_value,  # type: ignore[arg-type]
            timeout_seconds=1.0,
        )
        # get_secret_value() reveals the string: not empty for an empty input.
        rendered = m.openai_api_key.get_secret_value() if hasattr(
            m.openai_api_key, "get_secret_value"
        ) else str(m.openai_api_key)
        assert rendered == "EMPTY", (
            f"empty api_key did not fall back to 'EMPTY' "
            f"(got {rendered!r})"
        )


def test_build_chat_model_use_responses_api_propagates():
    """The configured Responses API flag reaches the chat model."""
    m = _chat_models.build_chat_model(
        model="test-model",
        api_url="http://localhost:9999/v1",
        api_key="dummy",
        timeout_seconds=1.0,
        use_responses_api=True,
    )
    # langchain-openai exposes the flag as ``use_responses_api`` on
    # the ChatOpenAI instance.
    assert m.use_responses_api is True


def test_build_chat_model_use_responses_api_default_is_off():
    """The default keeps the Chat Completions path, used for vLLM, LiteLLM
    and DeepSeek (Responses API support on vLLM is per model and
    experimental). Unset, ChatOpenAI holds ``None`` rather than ``False``,
    and the Responses API is used only when set."""
    m = _chat_models.build_chat_model(
        model="test-model",
        api_url="http://localhost:9999/v1",
        api_key="dummy",
        timeout_seconds=1.0,
    )
    # None / falsy == default Chat Completions path.
    assert not m.use_responses_api


def test_build_chat_model_real_api_key_passed_through():
    """A non-empty key is passed through unchanged."""
    m = _chat_models.build_chat_model(
        model="test-model",
        api_url="http://localhost:9999/v1",
        api_key="sk-actual-key",
        timeout_seconds=1.0,
    )
    rendered = m.openai_api_key.get_secret_value() if hasattr(
        m.openai_api_key, "get_secret_value"
    ) else str(m.openai_api_key)
    assert rendered == "sk-actual-key"


def test_reasoning_subclass_is_noop_when_response_has_no_reasoning_fields():
    """When the response carries none of the configured fields, the subclass
    produces the same AIMessage as plain ChatOpenAI. Listing extra field
    names is harmless."""
    m = _chat_models._ChatOpenAIWithReasoning(
        model="test-model",
        base_url="http://localhost:9999/v1",
        api_key="dummy",
        reasoning_field_names=(
            "reasoning_content",
            "reasoning_details",
            "thinking",
        ),
    )
    response = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": "plain answer"},
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    result = m._create_chat_result(response)
    kw = result.generations[0].message.additional_kwargs
    # No reasoning fields captured because none are present.
    assert "reasoning_content" not in kw
    assert "reasoning_details" not in kw
    assert "thinking" not in kw


def test_reasoning_subclass_outbound_noop_when_no_reasoning_kwargs():
    """Without reasoning kwargs on the AIMessages, the outbound payload is
    byte-identical to plain ChatOpenAI's, with no empty fields added."""
    from langchain_core.messages import AIMessage, HumanMessage
    from langchain_openai import ChatOpenAI

    base = ChatOpenAI(
        model="test-model",
        base_url="http://localhost:9999/v1",
        api_key="dummy",
    )
    sub = _chat_models._ChatOpenAIWithReasoning(
        model="test-model",
        base_url="http://localhost:9999/v1",
        api_key="dummy",
        reasoning_field_names=(
            "reasoning_content",
            "reasoning_details",
            "thinking",
        ),
    )
    input_msgs = [HumanMessage(content="hi"), AIMessage(content="reply")]
    base_payload = base._get_request_payload(input_msgs)
    sub_payload = sub._get_request_payload(input_msgs)
    assert base_payload["messages"] == sub_payload["messages"]


def test_build_chat_model_with_preserve_reasoning_returns_subclass():
    from langchain_openai import ChatOpenAI

    m = _chat_models.build_chat_model(
        model="test-model",
        api_url="http://localhost:9999/v1",
        api_key="dummy",
        timeout_seconds=1.0,
        preserve_reasoning_content=True,
    )
    assert type(m).__name__ == "_ChatOpenAIWithReasoning"
    assert isinstance(m, ChatOpenAI)


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
def test_all_agents_default_to_reasoning_sidecar_when_config_omits_it(phase, in_run):
    """The run settings default ``preserve_reasoning_content`` to True
    (sidecar on) when the user's settings omit it, so thinking models keep
    their reasoning across turns. The editing agents build their model
    from them."""
    from testuary import config, models

    in_run(overrides={"api": {"model": "fixture", "provider": "openai"}})
    settings = models.resolve_phase_model(config.run_settings(), phase)
    m = models.build_resolved_chat_model(settings, api_key="EMPTY")
    assert type(m).__name__ == "_ChatOpenAIWithReasoning", (
        "the run settings default the reasoning sidecar OFF"
    )


def test_reasoning_subclass_outbound_payload_includes_reasoning_content():
    """The subclass re-emits ``reasoning_content`` from
    ``AIMessage.additional_kwargs`` in the outbound payload, so the backend
    sees the model's earlier reasoning on the next turn."""
    from langchain_core.messages import AIMessage, HumanMessage

    m = _chat_models._ChatOpenAIWithReasoning(
        model="test-model",
        base_url="http://localhost:9999/v1",
        api_key="dummy",
    )

    input_msgs = [
        HumanMessage(content="hi"),
        AIMessage(
            content="hello",
            additional_kwargs={"reasoning_content": "I am thinking..."},
        ),
        HumanMessage(content="next"),
    ]
    payload = m._get_request_payload(input_msgs)
    payload_msgs = payload["messages"]
    # The AIMessage at index 1 must carry reasoning_content.
    assert payload_msgs[1].get("reasoning_content") == "I am thinking..."
    # Other messages carry no reasoning_content.
    assert "reasoning_content" not in payload_msgs[0]
    assert "reasoning_content" not in payload_msgs[2]


def test_reasoning_subclass_outbound_payload_omits_when_absent():
    """Without reasoning_content on the AIMessage, the payload has no such
    field."""
    from langchain_core.messages import AIMessage, HumanMessage

    m = _chat_models._ChatOpenAIWithReasoning(
        model="test-model",
        base_url="http://localhost:9999/v1",
        api_key="dummy",
    )

    input_msgs = [
        HumanMessage(content="hi"),
        AIMessage(content="hello"),  # No reasoning_content
    ]
    payload = m._get_request_payload(input_msgs)
    for msg in payload["messages"]:
        assert "reasoning_content" not in msg


def test_reasoning_subclass_supports_custom_field_names():
    """The subclass round-trips any configured field names, for thinking
    models served with another sidecar field spelling."""
    from langchain_core.messages import AIMessage, HumanMessage

    m = _chat_models._ChatOpenAIWithReasoning(
        model="test-model",
        base_url="http://localhost:9999/v1",
        api_key="dummy",
        reasoning_field_names=("reasoning_content", "reasoning_details", "thinking"),
    )

    # Inbound: the response carries reasoning_details, not reasoning_content.
    response = {
        "id": "x",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "answer",
                    "reasoning_details": "{some json blob}",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
    }
    result = m._create_chat_result(response)
    kw = result.generations[0].message.additional_kwargs
    assert kw.get("reasoning_details") == "{some json blob}"
    # reasoning_content not present in response -> not synthesized.
    assert "reasoning_content" not in kw

    # Outbound: payload echoes reasoning_details on the AIMessage.
    input_msgs = [
        HumanMessage(content="hi"),
        AIMessage(
            content="prior",
            additional_kwargs={"reasoning_details": "old thought"},
        ),
    ]
    payload = m._get_request_payload(input_msgs)
    assert payload["messages"][1].get("reasoning_details") == "old thought"


def test_reasoning_subclass_inbound_captures_into_additional_kwargs():
    """The subclass captures ``reasoning_content`` from the raw response
    into ``AIMessage.additional_kwargs["reasoning_content"]``."""
    m = _chat_models._ChatOpenAIWithReasoning(
        model="test-model",
        base_url="http://localhost:9999/v1",
        api_key="dummy",
    )

    # Synthesize a response with reasoning_content present.
    response = {
        "id": "chatcmpl-test",
        "object": "chat.completion",
        "created": 0,
        "model": "test-model",
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "the answer",
                    "reasoning_content": "I considered all options",
                },
                "finish_reason": "stop",
            }
        ],
        "usage": {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
        },
    }
    result = m._create_chat_result(response)
    captured = result.generations[0].message.additional_kwargs.get(
        "reasoning_content"
    )
    assert captured == "I considered all options"


def test_reasoning_subclass_default_captures_ollama_reasoning_field():
    """The default field names capture Ollama Cloud's ``reasoning`` field,
    not only ``reasoning_content``, or Ollama thinking models lose their
    reasoning between turns. It is captured inbound and re-emitted
    outbound."""
    m = _chat_models._ChatOpenAIWithReasoning(
        model="test-model",
        base_url="http://localhost:9999/v1",
        api_key="dummy",
    )
    assert "reasoning" in m.reasoning_field_names  # default covers Ollama

    # Inbound: an Ollama-shaped response (message.reasoning, not _content).
    response = {
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": "391",
                    "reasoning": "17*23 = 340+51 = 391",
                },
                "finish_reason": "stop",
            }
        ],
    }
    result = m._create_chat_result(response)
    msg = result.generations[0].message
    assert msg.additional_kwargs.get("reasoning") == "17*23 = 340+51 = 391"
    assert "reasoning_content" not in msg.additional_kwargs  # not synthesized

    # Outbound: that captured reasoning is re-emitted on the AIMessage.
    payload = m._get_request_payload([msg])
    assert payload["messages"][0].get("reasoning") == "17*23 = 340+51 = 391"


def test_trace_logger_disabled_is_noop(tmp_path):
    """TraceLogger(None) is a no-op: events do nothing, no files written."""
    logger = lg.TraceLogger(None)
    assert logger.enabled is False
    logger.event("agent_start", foo=1, bar="baz")
    logger.model_invoke_start(["t1", "t2"], 10, "HumanMessage")
    # No file created.
    assert list(tmp_path.iterdir()) == []


def test_trace_logger_writes_one_jsonl_per_event(tmp_path):
    """When enabled, each ``event`` call appends one JSONL line."""
    import json as _json

    path = tmp_path / "trace.jsonl"
    logger = lg.TraceLogger(path)
    assert logger.enabled is True

    logger.event("agent_start", framework="langgraph", target_file="foo.py")
    logger.event("select_tools_decision", decision="HIDE", bound=["t1"])
    logger.event("agent_end", duration_s=12.3)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert len(lines) == 3
    rows = [_json.loads(line) for line in lines]
    assert rows[0]["event"] == "agent_start"
    assert rows[0]["framework"] == "langgraph"
    assert rows[1]["event"] == "select_tools_decision"
    assert rows[1]["bound"] == ["t1"]
    assert rows[2]["event"] == "agent_end"
    assert rows[2]["duration_s"] == 12.3
    # The append order is sufficient and no calendar metadata is persisted.
    for r in rows:
        assert "ts" not in r
        assert isinstance(r["t_mono_s"], float)
        assert r["clock_domain_id"]


def test_trace_logger_helpers_emit_expected_shapes(tmp_path):
    """The named helper methods emit the documented shapes."""
    import json as _json

    path = tmp_path / "trace.jsonl"
    logger = lg.TraceLogger(path)

    logger.model_invoke_start(
        bound_tool_names=["read", "write"],
        message_count=4,
        last_message_type="HumanMessage",
    )
    logger.model_invoke_end(
        duration_s=12.345,
        tool_calls=["read"],
        content_preview="thinking...",
        content_len=120,
        reasoning_content_len=5000,
        finish_reason="tool_calls",
    )
    logger.validator_check(accepted=False, rejection_reason="not green")
    logger.recovery_attempt(
        attempt=2, max_attempts=100,
        kind="context_window",
        exception_type="ContextOverflow",
        exception_message="overflow",
    )

    rows = [_json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    by_event = {r["event"]: r for r in rows}
    assert by_event["model_invoke_start"]["bound_tools"] == ["read", "write"]
    assert by_event["model_invoke_end"]["reasoning_content_len"] == 5000
    assert by_event["model_invoke_end"]["finish_reason"] == "tool_calls"
    assert by_event["validator_check"]["accepted"] is False
    assert by_event["recovery_attempt"]["kind"] == "context_window"


def test_trace_logger_swallows_io_errors(tmp_path):
    """A trace logger that cannot write never breaks the agent run."""
    bad_path = tmp_path / "no_such_subdir" / "trace.jsonl"
    logger = lg.TraceLogger(bad_path)
    # Construction creates the missing parent directory.
    assert logger.enabled is True
    # Then the parent directory is removed.
    import shutil as _shutil
    _shutil.rmtree(bad_path.parent)
    # Later events fail silently (ENOENT).
    logger.event("agent_start")  # must not raise


def test_trace_logger_lines_are_pinned_byte_for_byte(tmp_path, monkeypatch):
    """Each line is the event, the timing fields, then the payload without
    its None values, as json.dumps writes it with ensure_ascii off and str
    for a value JSON has no type for, ended by the platform's line end."""
    import json as _json
    import os as _os
    from types import SimpleNamespace

    path = tmp_path / "trace.jsonl"
    logger = lg.TraceLogger(path)
    timing = {"t_mono_s": 12.5, "clock_domain_id": "process-x"}
    monkeypatch.setattr(lg, "timing_fields", lambda: dict(timing))
    logger.event("agent_start", target_file="\u00e4pfel.py", skipped=None,
                 where=tmp_path / "x", shape=SimpleNamespace(a=1), counts=[1, 2.5])

    expected = _json.dumps({"event": "agent_start", **timing, "target_file": "\u00e4pfel.py",
                            "where": tmp_path / "x", "shape": SimpleNamespace(a=1), "counts": [1, 2.5]},
                           ensure_ascii=False, default=str)
    assert path.read_bytes() == (expected + _os.linesep).encode("utf-8")


def test_export_chat_log_writes_json_with_metadata_and_messages(tmp_path):
    msgs = [
        SystemMessage(content="sys"),
        HumanMessage(content="hi"),
        AIMessage(
            content="thinking...",
            tool_calls=[{"name": "t", "args": {"x": 1}, "id": "c1"}],
        ),
        ToolMessage(content="result", tool_call_id="c1", name="t"),
        AIMessage(content="done"),
    ]
    out_path = _chat_log.export_chat_log(
        msgs,
        str(tmp_path),
        "chat.json",
        project_name="demo",
        extra_metadata={"agent": "smoke"},
    )
    assert out_path is not None
    import json
    data = json.loads((tmp_path / "chat.json").read_text(encoding="utf-8"))
    assert data["metadata"]["project"] == "demo"
    assert data["metadata"]["agent"] == "smoke"
    assert data["metadata"]["message_count"] == 5
    assert len(data["messages"]) == 5
    # Tool-call structure preserved.
    ai = data["messages"][2]
    assert ai["type"] == "response"
    assert ai["tool_calls"][0]["name"] == "t"
    # Tool return structure preserved.
    tr = data["messages"][3]
    assert tr["type"] == "tool_return"
    assert tr["tool_name"] == "t"
    assert tr["content"] == "result"


def test_export_chat_log_includes_response_metadata_on_ai_message(tmp_path):
    """The response_metadata of an AIMessage (finish_reason and the like)
    appears in the chat log, so truncation, length cutoffs and the model
    identity show without a rerun."""
    ai = AIMessage(
        content="full response",
        response_metadata={
            "finish_reason": "length",
            "model_name": "minimax-m2",
            "token_usage": {"prompt_tokens": 100, "completion_tokens": 80000},
            "noise_field_should_be_dropped": "blah",
        },
    )
    out_path = _chat_log.export_chat_log([ai], str(tmp_path), "chat.json")
    assert out_path is not None
    import json
    data = json.loads((tmp_path / "chat.json").read_text(encoding="utf-8"))
    entry = data["messages"][0]
    assert entry["response_metadata"]["finish_reason"] == "length"
    assert entry["response_metadata"]["model_name"] == "minimax-m2"
    assert entry["response_metadata"]["token_usage"]["completion_tokens"] == 80000
    # Other fields are left out.
    assert "noise_field_should_be_dropped" not in entry["response_metadata"]


def test_export_chat_log_records_invalid_tool_calls(tmp_path):
    """A malformed call (args not valid JSON) is serialized on the wire
    and answered at dispatch, so the chat log must show it: an assistant
    entry whose completion tokens produced neither content nor parsed
    calls is unreadable in the log otherwise."""
    ai = AIMessage(content="", invalid_tool_calls=[{
        "name": "edit_tests", "args": "{not json",
        "id": "bad9", "error": "Unterminated string",
        "type": "invalid_tool_call",
    }])
    out_path = _chat_log.export_chat_log([ai], str(tmp_path), "chat.json")
    assert out_path is not None
    import json
    entry = json.loads(
        (tmp_path / "chat.json").read_text(encoding="utf-8"))["messages"][0]
    rec = entry["invalid_tool_calls"][0]
    assert rec["name"] == "edit_tests"
    assert rec["id"] == "bad9"
    assert "Unterminated" in rec["error"]
    assert rec["args_head"].startswith("{not json")


def test_a_lone_surrogate_keeps_every_later_chat_log_flush(tmp_path):
    """A model can send a lone surrogate in a tool call's arguments (the edit
    tool refuses it as bad characters), and the call stays in the history:
    every later flush still writes the whole log, and the character reads
    back."""
    flush = _chat_log.make_chat_log_flusher(output_dir=str(tmp_path), filename="chat.json")
    call = AIMessage("", tool_calls=[{"id": "e1", "name": "edit_tests",
                                      "args": {"new_text": "x\ud800y"}}])
    history = [HumanMessage("Write tests"), call]
    flush(history)
    history.append(ToolMessage("Refused: bad characters", tool_call_id="e1"))
    flush(history, stage="completed")
    log = json.loads((tmp_path / "chat.json").read_text(encoding="utf-8"))
    assert log["metadata"]["stage"] == "completed" and log["metadata"]["message_count"] == 3
    assert "x\ud800y" in json.dumps(log, ensure_ascii=False)


def test_a_lone_surrogate_in_a_request_keeps_its_usage_and_transport_records(tmp_path):
    """A request whose history holds a lone surrogate (a model's tool
    arguments) gets its message hashes recorded, and a usage line holding
    one is written and reads back. A text without one hashes as before."""
    from testuary.generation_evidence import source_sha256

    logger = lg.UsageLogger(tmp_path / "usage.jsonl", prompt_transport_path=tmp_path / "transport.jsonl")
    call = AIMessage("", tool_calls=[{"id": "e1", "name": "edit_tests",
                                      "args": {"new_text": "x\ud800y"}}])
    logger.log_prompt_transport(request_idx=1, messages=[HumanMessage("Write tests"), call])
    logger.log(request_idx=1, input_tokens=3, output_tokens=2, cache_hit_tokens=None,
               finish_reason="stop",
               extra={"note": "x\ud800y"})
    transport = (tmp_path / "transport.jsonl").read_text(encoding="utf-8").splitlines()
    assert [len(json.loads(line)["message_sha256"]) for line in transport] == [2]
    (usage,) = (tmp_path / "usage.jsonl").read_text(encoding="utf-8").splitlines()
    assert json.loads(usage)["note"] == "x\ud800y"
    text = "plain text, caf\u00e9"
    assert source_sha256(text) == hashlib.sha256(text.encode("utf-8")).hexdigest()
    assert source_sha256("x\ud800y") != source_sha256("x\\ud800y")


def test_export_chat_log_includes_reasoning_content_when_captured(tmp_path):
    """A reasoning sidecar field captured by the reasoning-preservation
    subclass appears in the chat log."""
    ai = AIMessage(
        content="final answer",
        additional_kwargs={
            "reasoning_content": "thinking carefully...",
            "some_other_kwarg": "should be dropped",
        },
    )
    out_path = _chat_log.export_chat_log([ai], str(tmp_path), "chat.json")
    assert out_path is not None
    import json
    data = json.loads((tmp_path / "chat.json").read_text(encoding="utf-8"))
    entry = data["messages"][0]
    assert entry["additional_kwargs"]["reasoning_content"] == "thinking carefully..."
    # Other kwargs are left out: the chat log shows reasoning kwargs only.
    assert "some_other_kwarg" not in entry["additional_kwargs"]


# Tool gating: the model can call only the tools its current phase binds.
# bind_tools alone is a hint, and ToolNode would dispatch any tool by name.

def test_build_react_graph_per_phase_mode_terminates():
    """Per-phase mode with two phases: the resolver returns the second and
    the model emits the final-answer tool. The graph ends without touching
    the other phase's tools."""
    scripted = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "emit_final_answer",
                "args": {"score": 9, "note": "ok"},
                "id": "f1",
            }],
        ),
    ]
    model = _ToolAwareFakeModel(responses=scripted)
    graph = lg.build_react_graph(
        classify_node=_no_classify,
        model=model,
        agent_mode_tools={
            "phase_a": [_stub_read],
            "phase_b": [_stub_write],
        },
        resolve_agent_mode=lambda s: "phase_b",
        structured_output=_StubResult,
        state_type=_StubGraphState,
    )
    final_state = graph.invoke(
        {"messages": [HumanMessage(content="please score")]}
    )
    assert final_state.get("result") is not None
    assert final_state["result"].score == 9


def test_build_react_graph_per_phase_mode_rejects_unbound_tool_call():
    """Per-phase dispatch does not execute a tool the agent mode does not bind.

    The provider may emit a tool name the prompt mentioned but the current
    agent mode does not bind. ToolNode returns an error ToolMessage, and the
    model can retry with a bound tool.
    """
    write_invocations: list[str] = []

    @tool
    def _phase_write_tracker(content: str) -> str:
        """Tracked write that is unavailable in the active phase."""
        write_invocations.append(content)
        return f"wrote {len(content)} bytes"

    scripted = [
        AIMessage(content="", tool_calls=[{
            "name": "_phase_write_tracker",
            "args": {"content": "x"},
            "id": "unbound",
        }]),
        AIMessage(content="", tool_calls=[{
            "name": "_stub_read",
            "args": {},
            "id": "bound",
        }]),
        AIMessage(content="", tool_calls=[{
            "name": "emit_final_answer",
            "args": {"score": 3, "note": "done"},
            "id": "finishing",
        }]),
    ]
    graph = lg.build_react_graph(
        classify_node=_no_classify,
        model=_ToolAwareFakeModel(responses=scripted),
        agent_mode_tools={
            "read_phase": [_stub_read],
            "write_phase": [_phase_write_tracker],
        },
        resolve_agent_mode=lambda _state: "read_phase",
        structured_output=_StubResult,
        state_type=_StubGraphState,
    )

    final_state = graph.invoke(
        {"messages": [HumanMessage(content="please score")]}
    )

    assert final_state.get("result") is not None
    assert write_invocations == []
    refused = [
        message for message in final_state["messages"]
        if (
            isinstance(message, ToolMessage)
            and message.name == "_phase_write_tracker"
        )
    ]
    assert len(refused) == 1
    assert refused[0].status == "error"
    assert "not a valid tool" in refused[0].content


def test_per_phase_unbound_call_gets_the_callers_answer_in_call_order():
    """With ``unbound_tool_reply`` a call of a tool the phase does not bind
    is answered with the caller's text, not ToolNode's generic error. The
    bound call of the same response runs, the answers keep the call order,
    and a name the caller does not know keeps ToolNode's error."""
    write_invocations: list[str] = []

    @tool
    def _gated_write(content: str) -> str:
        """Write that the active phase does not bind."""
        write_invocations.append(content)
        return "wrote"

    scripted = [
        AIMessage(content="", tool_calls=[
            {"name": "_gated_write", "args": {"content": "x"}, "id": "g"},
            {"name": "_stub_read", "args": {}, "id": "r"},
            {"name": "_made_up", "args": {}, "id": "u"},
        ]),
        AIMessage(content="", tool_calls=[{
            "name": "emit_final_answer",
            "args": {"score": 3, "note": "done"},
            "id": "finishing",
        }]),
    ]
    graph = lg.build_react_graph(
        classify_node=_no_classify,
        model=_ToolAwareFakeModel(responses=scripted),
        agent_mode_tools={"read_phase": [_stub_read], "write_phase": [_gated_write]},
        resolve_agent_mode=lambda _state: "read_phase",
        structured_output=_StubResult,
        state_type=_StubGraphState,
        unbound_tool_reply=lambda name: (
            f"{name} is offered after a read." if name == "_gated_write" else None),
    )

    final_state = graph.invoke({"messages": [HumanMessage(content="please score")]})

    assert write_invocations == []
    answers = [m for m in final_state["messages"]
               if isinstance(m, ToolMessage) and m.tool_call_id in ("g", "r", "u")]
    assert [m.tool_call_id for m in answers] == ["g", "r", "u"]
    assert answers[0].content == "_gated_write is offered after a read."
    assert answers[0].status == "error"
    assert answers[1].status != "error"
    assert "not a valid tool" in answers[2].content


def test_build_react_graph_per_phase_mode_routes_between_phases():
    """Per-phase mode: classify_node updates state['phase'] between turns,
    so the router moves from phase_a (read) to phase_b (write). At each turn
    the model sees only the current phase's tools."""
    seen_bindings: list[list[str]] = []

    class _Recorder(_ToolAwareFakeModel):
        def bind_tools(self, tools, *, tool_choice=None, **kw):
            seen_bindings.append([t.name for t in tools])
            return self

    # Sequence: read (phase_a) -> write (phase_b after classify flips) -> final.
    scripted = [
        AIMessage(content="", tool_calls=[
            {"name": "_stub_read", "args": {}, "id": "r1"},
        ]),
        AIMessage(content="", tool_calls=[
            {"name": "_stub_write", "args": {"content": "x"}, "id": "w1"},
        ]),
        AIMessage(content="", tool_calls=[{
            "name": "emit_final_answer",
            "args": {"score": 5, "note": "done"},
            "id": "f1",
        }]),
    ]
    model = _Recorder(responses=scripted)

    # State carries the phase as a simple slot. classify flips it
    # from phase_a -> phase_b after each dispatch.
    class _PhasedState(TypedDict):
        messages: Annotated[List[BaseMessage], add_messages]
        result: Any
        phase: str

    def _classify(state):
        # First call: flip to phase_b. Subsequent calls: stay there.
        return {"agent_mode": "phase_b"}

    graph = lg.build_react_graph(
        model=model,
        agent_mode_tools={
            "phase_a": [_stub_read],
            "phase_b": [_stub_write],
        },
        resolve_agent_mode=lambda s: s.get("agent_mode", "phase_a"),
        classify_node=_classify,
        structured_output=_StubResult,
        state_type=_PhasedState,
    )
    final_state = graph.invoke(
        {"messages": [HumanMessage(content="go")], "phase": "phase_a"}
    )
    assert final_state.get("result") is not None
    # First binding: phase_a (_stub_read), second binding: phase_b (_stub_write).
    # The final-answer tool is appended to every phase's bind so always present.
    assert "_stub_read" in seen_bindings[0]
    assert "_stub_write" not in seen_bindings[0]
    assert "_stub_write" in seen_bindings[1]
    assert "_stub_read" not in seen_bindings[1]


def test_build_react_graph_binds_each_modes_tools_in_order_then_the_final_tool():
    """The bind order is model input: each agent mode binds its tools in the
    given order, then the final-answer tool when there is a result schema."""
    seen_bindings: list[list[str]] = []

    class _Recorder(_ToolAwareFakeModel):
        def bind_tools(self, tools, *, tool_choice=None, **kw):
            seen_bindings.append([t.name for t in tools])
            return self

    modes = {"a": [_stub_write, _stub_read], "b": [_stub_read]}
    for structured_output in (_StubResult, None):
        lg.build_react_graph(
            classify_node=_no_classify,
            model=_Recorder(responses=[]),
            agent_mode_tools=modes,
            resolve_agent_mode=lambda s: "a",
            structured_output=structured_output,
            state_type=_StubGraphState,
        )
    assert seen_bindings == [
        ["_stub_write", "_stub_read", "emit_final_answer"],
        ["_stub_read", "emit_final_answer"],
        ["_stub_write", "_stub_read"],
        ["_stub_read"],
    ]


def test_build_react_graph_rejects_agent_mode_tools_without_resolve_agent_mode():
    """agent_mode_tools requires resolve_agent_mode. Pairing one without the
    other is a programming error and raises at build time."""
    with pytest.raises(ValueError, match="resolve_agent_mode"):
        lg.build_react_graph(
            classify_node=_no_classify,
            model=_ToolAwareFakeModel(responses=[]),
            agent_mode_tools={"only": [_stub_read]},
            state_type=_StubGraphState,
        )


def test_build_react_graph_allows_final_tool_in_any_phase():
    """The final-answer tool is always allowed, whatever a phase binds, so
    a model can end from a phase that binds only reads."""
    scripted = [
        AIMessage(
            content="",
            tool_calls=[{
                "name": "emit_final_answer",
                "args": {"score": 7, "note": "early"},
                "id": "f1",
            }],
        ),
    ]
    model = _ToolAwareFakeModel(responses=scripted)
    graph = lg.build_react_graph(
        classify_node=_no_classify,
        model=model,
        # No regular tools bound in this phase, only the final-answer.
        agent_mode_tools={"p": []},
        resolve_agent_mode=lambda s: "p",
        structured_output=_StubResult,
        state_type=_StubGraphState,
    )
    final_state = graph.invoke(
        {"messages": [HumanMessage(content="please score")]}
    )
    # The final-answer call was accepted (not refused) and the
    # graph terminated cleanly.
    assert final_state.get("result") is not None
    assert final_state["result"].score == 7


def test_build_chat_model_sends_max_tokens_in_extra_body():
    """langchain-openai serializes the output cap as ``max_completion_tokens``,
    which OpenAI-compatible servers that predate that field (DeepSeek's API)
    ignore, falling back to a small default. build_chat_model also puts the
    classic ``max_tokens`` in the request body so the cap is actually applied:
    it merges with a caller-supplied extra_body and is omitted when uncapped."""
    plain = _chat_models.build_chat_model(
        model="m", api_url="http://x", api_key="k", timeout_seconds=60,
        max_tokens=384000, extra_kwargs={"temperature": 0.7})
    assert plain.extra_body == {"max_tokens": 384000}

    merged = _chat_models.build_chat_model(
        model="m", api_url="http://x", api_key="k", timeout_seconds=60,
        max_tokens=384000,
        extra_kwargs={"extra_body": {"thinking_token_budget": 4096}})
    assert merged.extra_body == {
        "thinking_token_budget": 4096, "max_tokens": 384000}

    uncapped = _chat_models.build_chat_model(
        model="m", api_url="http://x", api_key="k", timeout_seconds=60,
        max_tokens=None, extra_kwargs={"temperature": 0.7})
    assert getattr(uncapped, "extra_body", None) is None


# Malformed tool calls (invalid_tool_calls) are answered at dispatch


def test_malformed_tool_call_is_answered_and_loop_recovers():
    """A tool call whose args are not valid JSON parses into
    ``invalid_tool_calls`` (empty ``tool_calls``), yet the API serializer
    still sends it as an assistant tool_calls message. The graph routes it
    to dispatch and answers the call id with corrective feedback, since the
    provider rejects a history with an unanswered call."""
    scripted = [
        AIMessage(content="", invalid_tool_calls=[{
            "name": "_noop", "args": "{not json", "id": "bad1",
            "error": "Unterminated string starting at char 1",
            "type": "invalid_tool_call",
        }]),
        AIMessage(content="", tool_calls=[{
            "name": "emit_final_answer",
            "args": {"answer": "done"},
            "id": "f1",
        }]),
    ]
    tracker = lg.UsageTracker(request_limit=10)
    graph = _make_token_wiring_graph(scripted=scripted, tracker=tracker)
    out = graph.invoke({"messages": [HumanMessage(content="go")]})
    tool_msgs = [m for m in out["messages"]
                 if isinstance(m, ToolMessage) and m.tool_call_id == "bad1"]
    assert len(tool_msgs) == 1
    assert "MALFORMED" in tool_msgs[0].content
    assert out["result"].answer == "done"


def test_mixed_valid_and_malformed_calls_all_answered():
    """One message with a dispatchable call and a malformed one: both call
    ids receive tool responses (the serializer sends both), and the loop
    ends normally."""
    scripted = [
        AIMessage(
            content="",
            tool_calls=[{"name": "_noop", "args": {}, "id": "ok1"}],
            invalid_tool_calls=[{
                "name": "_noop", "args": "{", "id": "bad2",
                "error": "Expecting value", "type": "invalid_tool_call",
            }],
        ),
        AIMessage(content="", tool_calls=[{
            "name": "emit_final_answer",
            "args": {"answer": "done"},
            "id": "f2",
        }]),
    ]
    tracker = lg.UsageTracker(request_limit=10)
    graph = _make_token_wiring_graph(scripted=scripted, tracker=tracker)
    out = graph.invoke({"messages": [HumanMessage(content="go")]})
    answered = {m.tool_call_id for m in out["messages"]
                if isinstance(m, ToolMessage)}
    assert {"ok1", "bad2"} <= answered
    assert out["result"].answer == "done"


def test_build_react_graph_has_one_graph_variant():
    """The graph binds each agent mode's tools at construction. It takes no
    per-turn ``select_tools`` and needs at least one agent mode."""
    with pytest.raises(TypeError):
        lg.build_react_graph(
            classify_node=_no_classify,
            model=_ToolAwareFakeModel(responses=[]),
            tools=[_stub_read],
            select_tools=lambda s: [_stub_read],
            state_type=_StubGraphState,
        )
    with pytest.raises(ValueError, match="at least one agent_mode"):
        lg.build_react_graph(
            classify_node=_no_classify,
            model=_ToolAwareFakeModel(responses=[]),
            resolve_agent_mode=lambda s: "p",
            state_type=_StubGraphState,
        )


# Final-answer validation through both compiled graph variants.
class _Model(FakeMessagesListChatModel):
    def bind_tools(self, tools, **kwargs):
        return self


class _State(TypedDict, total=False):
    messages: Annotated[list[BaseMessage], add_messages]
    result: Any


class _Result(BaseModel):
    score: int = Field(ge=0, le=100)


class _EmptyResult(BaseModel):
    score: int = 0


@tool
def _read() -> str:
    """Return controlled source content."""
    return "source"


@pytest.fixture
def _no_native_process(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("graph validation must not launch a native process")

    monkeypatch.setattr(subprocess, "Popen", refused)


def _answer(args, *, name="finish", call_id="finishing"):
    return AIMessage(content="", tool_calls=[{
        "name": name, "args": args, "id": call_id,
    }])


def _graph(mode, responses, trace, schema=_Result, validator=None):
    selection = {
        "agent_mode_tools": {"editing": [_read]},
        "resolve_agent_mode": lambda state: "editing",
    }
    return lg.build_react_graph(
        classify_node=_no_classify,
        model=_Model(responses=responses), state_type=_State,
        structured_output=schema, output_validator=validator,
        final_tool_name="finish", trace_logger=trace, **selection,
    )


def _checks(path):
    return [row for line in path.read_text(encoding="utf-8").splitlines()
            if (row := json.loads(line))["event"] == "validator_check"]


@pytest.mark.usefixtures("_no_native_process")
@pytest.mark.parametrize("mode", ["per_mode"])
@pytest.mark.parametrize("case", [
    "valid", "zero", "missing_call", "invalid_schema", "empty_args",
    "no_schema", "reject", "validator_error", "interrupt",
    "latest_message", "first_matching_call", "no_validator",
    "empty_rejection", "constructor_error", "constructor_interrupt",
])
def test_graph_final_validation_contract(mode, case, tmp_path):
    path = tmp_path / "trace.jsonl"
    trace = lg.TraceLogger(path)
    observed = []
    error = RuntimeError("validator did not execute")
    interrupted = KeyboardInterrupt("validation cancelled")

    def validator(state, result):
        observed.append((state, result))
        if case == "reject":
            return "measured result rejected"
        if case == "validator_error":
            raise error
        if case == "interrupt":
            raise interrupted
        if case == "empty_rejection":
            return ""
        return None

    schema = _EmptyResult if case == "empty_args" else _Result
    if case == "no_schema":
        schema = None
    elif case in {"constructor_error", "constructor_interrupt"}:
        class _FailingResult(_Result):
            def __init__(self, **data):
                raise error if case == "constructor_error" else interrupted

        schema = _FailingResult
    response = _answer({"score": 0 if case == "zero" else 37})
    if case in {"missing_call", "no_schema"}:
        response = AIMessage(content="finished without a tool")
    elif case == "invalid_schema":
        response = _answer({"score": 101})
    elif case == "empty_args":
        response = _answer({})
    elif case == "first_matching_call":
        response = AIMessage(content="", tool_calls=[
            {"name": "_read", "args": {}, "id": "ignored"},
            {"name": "finish", "args": {"score": 37}, "id": "first"},
            {"name": "finish", "args": {"score": 99}, "id": "second"},
        ])
    messages = [HumanMessage(content="generate a final result")]
    if case == "latest_message":
        messages = [_answer({"score": 9}, call_id="previous"), *messages]
    graph = _graph(mode, [response], trace, schema,
                   None if case == "no_validator" else validator)

    if case in {"missing_call", "invalid_schema", "reject", "constructor_error"}:
        with pytest.raises(lg.OutputValidationError) as caught:
            graph.invoke({"messages": messages})
        rejection = str(caught.value)
        if case == "missing_call":
            assert rejection == "model did not emit the final-answer tool call before exit"
            assert caught.value.__cause__ is None
        elif case == "invalid_schema":
            assert rejection.startswith("structured output rejected by Pydantic: ValidationError:")
            assert isinstance(caught.value.__cause__, ValidationError)
        elif case == "constructor_error":
            assert rejection == "structured output rejected by Pydantic: RuntimeError: validator did not execute"
            assert caught.value.__cause__ is error
        else:
            assert rejection == "measured result rejected"
        assert len(_checks(path)) == 1
        assert _checks(path)[0]["accepted"] is False
        assert _checks(path)[0]["reason"] == rejection
        assert len(observed) == (1 if case == "reject" else 0)
    elif case in {"validator_error", "interrupt", "constructor_interrupt"}:
        expected = error if case == "validator_error" else interrupted
        with pytest.raises(type(expected)) as caught:
            graph.invoke({"messages": messages})
        assert caught.value is expected
        assert len(observed) == (0 if case == "constructor_interrupt" else 1)
        assert _checks(path) == []
    else:
        result = graph.invoke({"messages": messages})["result"]
        if case == "no_schema":
            assert result is None
            assert observed == []
        else:
            assert result.score == (0 if case in {"zero", "empty_args"} else 37)
            assert len(observed) == (0 if case == "no_validator" else 1)
            if observed:
                assert observed[0][1] is result
        assert len(_checks(path)) == 1
        assert _checks(path)[0]["accepted"] is True
