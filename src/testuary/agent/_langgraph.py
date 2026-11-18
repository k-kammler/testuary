"""LangGraph + LangChain scaffolding for the agents.

Framework-independent pieces (``LoopBreakerState``,
``BestGreenState``, ``source_structure_error``, ``tool_msg``,
``contains_history_marker`` and others) are in ``_helpers.py``, the chat
model clients in ``_chat_models.py``, the chat log and its logger adapter
in ``_chat_log.py`` and the test output parsers in ``_test_output.py``.
This module owns:

  1. ``TraceLogger`` and ``UsageLogger``: the optional per-target trace
     and the per-request usage log.
  2. Recovery on LangChain messages. The context policy (dedup + keep-N +
     latest thinking) lives only in ``testuary.agent._context`` and runs in the
     pre-model hook. The model node retries a model request's transport
     failures through ``testuary.model_requests``, so a model request is
     charged once whatever its attempts sent. ``run_graph_with_retries``
     keeps the complete captured state across output-validation rejections
     and adds protocol-valid feedback to it. A context-window failure stops
     visibly. There is no second compaction policy. When retries are
     exhausted the entry point still emits a result.
  3. Graph factory: ``build_react_graph`` returns a compiled
     LangGraph instance for the standard agent loop:
     ``START -> model_decide -> tool_dispatch -> classify -> model_decide``
     while the model calls tools, ``model_decide -> validate -> END`` for
     its final answer,
     with one model and dispatch node pair per agent mode.
  4. ``UsageTracker``: counts model requests and tokens and raises
     ``UsageLimitExceeded`` when a run limit is exceeded.

The agent module supplies the state schema (a ``TypedDict``), the tools
of each agent mode with the function that resolves the mode, the classify
node, the pre-model hook and the structured-output Pydantic model with its
validator.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type
import httpx
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import BaseTool
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode
from langgraph.prebuilt.tool_node import ToolInvocationError
from pydantic import BaseModel
from typing_extensions import TypedDict
from testuary import phase_state
from testuary.agent import _limits
from testuary.agent._helpers import json_lines_logger
from testuary import evidence as validity
from testuary import model_requests
from testuary.phase_state import timing_fields
from testuary.models import DEFAULT_FIELDS, field_names
import threading


# Trace logger: an optional per-target JSONL log of an agent run's events
# (agent start and end, agent-mode changes, model requests with their bound
# tools, timing, response shape and reasoning length, validator outcomes,
# recovery attempts). Off by default, since the console and chat logs carry
# what a run needs. ``enable_trace_log = true`` in the phase's section
# (``[coverage]``, which the mutation phase inherits) turns it on for
# questions such as why the model called a tool, whether reasoning
# round-trips, or which tools a turn offered.


class TraceLogger:
    """Append-only JSONL trace of agent-loop events: one JSON object per
    line, in event order, with the event kind, the timing fields and the
    event's payload. Safe to follow while it grows. ``target_path`` None
    gives a no-op instance, so callers need no gating.
    """

    def __init__(self, target_path: Optional[Path]) -> None:
        self._log = None
        if target_path is not None:
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                # Each run starts from an empty trace.
                target_path.write_text("", encoding="utf-8")
                self._log = json_lines_logger(target_path)
            except Exception:
                # An unwritable target_path gives a silent no-op: logging
                # never breaks the run.
                self._log = None

    @property
    def enabled(self) -> bool:
        return self._log is not None

    def event(self, event_type: str, **payload: Any) -> None:
        """Append one event line. No-op when the logger is disabled. A
        line that cannot be written, on a full disk or a revoked
        permission, is dropped: tracing is never a hard dependency."""
        if self._log is None:
            return
        entry: Dict[str, Any] = {"event": event_type, **timing_fields()}
        # Drop None values to keep lines compact.
        for k, v in payload.items():
            if v is None:
                continue
            entry[k] = v
        self._log.info(entry)

    # Helpers for the common events. ``event`` takes any other.

    def model_invoke_start(
        self,
        bound_tool_names: List[str],
        message_count: int,
        last_message_type: Optional[str],
    ) -> None:
        self.event(
            "model_invoke_start",
            bound_tools=bound_tool_names,
            message_count=message_count,
            last_message_type=last_message_type,
        )

    def model_invoke_end(
        self,
        duration_s: float,
        tool_calls: List[str],
        content_preview: str,
        content_len: int,
        reasoning_content_len: int,
        finish_reason: Optional[str],
    ) -> None:
        self.event(
            "model_invoke_end",
            duration_s=round(duration_s, 3),
            tool_calls=tool_calls,
            content_preview=content_preview[:300] if content_preview else "",
            content_len=content_len,
            reasoning_content_len=reasoning_content_len,
            finish_reason=finish_reason,
        )

    def validator_check(
        self,
        accepted: bool,
        rejection_reason: Optional[str] = None,
    ) -> None:
        self.event(
            "validator_check",
            accepted=accepted,
            reason=rejection_reason,
        )

    def recovery_attempt(
        self,
        attempt: int,
        max_attempts: int,
        kind: str,
        exception_type: Optional[str],
        exception_message: Optional[str],
    ) -> None:
        self.event(
            "recovery_attempt",
            attempt=attempt,
            max_attempts=max_attempts,
            kind=kind,
            exception_type=exception_type,
            exception_message=model_requests.failure_text(exception_message)[:500] if exception_message else None,
        )


_NULL_TRACE = TraceLogger(None)


# Usage tracking (the per-run limits of model requests).


class UsageLimitExceeded(Exception):
    """Raised when ``UsageTracker`` sees a limit overrun.

    The agent entry point treats this as the designed limit stop and
    finalizes from retained framework state without another model call.
    ``limit`` names the limit for the display: the request, time or token
    limit.
    """

    def __init__(self, message: str = "", limit: str = "limit") -> None:
        super().__init__(message)
        self.limit = limit


def _limit_words(limit: str) -> str:
    """The display's words for the phase limit ``limit`` reached, such as
    ``Request limit reached``."""
    return f"{limit[:1].upper()}{limit[1:]} reached"


def _limit_text(exc: UsageLimitExceeded, usage_tracker: Optional["UsageTracker"]) -> str:
    """The words of the limit ``exc`` reached, with the request limit's
    count."""
    words = _limit_words(exc.limit)
    if exc.limit == "request limit" and usage_tracker is not None and usage_tracker.request_limit:
        return f"{words} ({usage_tracker.logical_requests_used} of {usage_tracker.request_limit})"
    return words


_LIMIT_WARNING = (
    "[BUDGET WARNING] The request budget is nearly exhausted. "
    "Emit your final structured output now if the framework can accept it. "
    "Otherwise, make one focused tool call rather than exploring."
)


def _append_limit_warning(
    messages: List[BaseMessage],
    usage_tracker: Optional["UsageTracker"],
) -> List[BaseMessage]:
    """Append the stable limit warning while the model requests left are
    within the last fifth of the limit (at least one). The text never names
    the changing number left: some OpenAI-compatible providers fold trailing
    system messages into the cached system prefix, so a text changing every
    turn would invalidate the otherwise reusable conversation prefix.
    """
    if usage_tracker is None or usage_tracker.request_limit is None:
        return messages
    remaining = usage_tracker.request_limit - usage_tracker.logical_requests_used
    threshold = max(1, usage_tracker.request_limit // 5)
    if 0 <= remaining <= threshold:
        return messages + [SystemMessage(content=_LIMIT_WARNING)]
    return messages


@dataclasses.dataclass
class UsageTracker:
    """The per-run limits of model requests, tokens and wall-clock time.

    The model node calls ``note_request`` before each request. Whichever
    limit is hit first raises ``UsageLimitExceeded``, which the entry
    point treats as the designed limit stop: ``finalize_result`` restores
    the best passing suite and the phase end proceeds. The limits:
      * ``request_limit`` (``max_requests_per_phase``): model requests.
      * ``input_token_limit``: cumulative prompt tokens.
      * ``output_token_limit``: cumulative completion tokens.
      * ``combined_token_limit``: cumulative prompt + completion tokens.
      * ``wall_clock_limit_seconds`` (``max_wall_clock_seconds_per_phase``):
        elapsed seconds since the first
        model invocation (per-phase deadline. Setup before the first call
        is not counted).

    A limit of ``None`` or ``0`` is unbounded. Only ``max_requests_per_phase``
    is set by default. The request and token caps are deterministic (the
    same counts on any machine). The wall-clock cap depends on the hardware
    and is meant for an operator who needs a hard time bound. ``note_tokens``
    adds the usage a response reports right after it arrives. A delivered
    response without reported usage is counted in ``unknown_usage_requests``
    and leaves the token counters unchanged.

    ``requests_used`` counts requests sent, every attempt.
    ``logical_requests_used`` counts the turns charged against
    ``max_requests_per_phase``: one per model turn, or one per attempt with
    count_retries on. They are equal in a run without transport
    failures.
    """

    request_limit: Optional[int] = None
    input_token_limit: Optional[int] = None
    output_token_limit: Optional[int] = None
    combined_token_limit: Optional[int] = None
    wall_clock_limit_seconds: Optional[float] = None
    requests_used: int = 0
    logical_requests_used: int = 0
    unknown_usage_requests: int = 0
    input_tokens_used: int = 0
    output_tokens_used: int = 0
    #: Cached-prefix input tokens (DeepSeek ``prompt_cache_hit_tokens`` /
    #: OpenAI ``prompt_tokens_details.cached_tokens``): a subset of
    #: ``input_tokens_used`` billed at the lower cache-hit rate, summed over
    #: the responses that report the cache split.
    cache_hit_tokens_used: int = 0
    #: Delivered responses with reported token usage but no reported cache
    #: split (for example Ollama or vLLM): unknown, not zero cache hits.
    unknown_cache_requests: int = 0
    _started_monotonic: Optional[float] = None
    #: Progress heartbeat: every ``heartbeat_every`` requests sent (0 is
    #: off) the agent's ``heartbeat_fn(logical_requests_used,
    #: request_limit_or_0, elapsed_s)`` reports where a long detached run is,
    #: with the phase and target the tracker does not hold.
    heartbeat_every: int = 0
    heartbeat_fn: Optional[Callable[[int, int, float], None]] = None

    def note_request(self) -> None:
        """Charge one model request and count one request sent, before the
        first attempt of a model turn.

        Raises ``UsageLimitExceeded`` when the request limit is reached or
        the wall-clock deadline has passed, before the request starts, so the
        deadline is overshot by at most one operation in flight.
        """
        self.check_deadline()
        if (
            self.request_limit is not None
            and self.logical_requests_used >= self.request_limit
        ):
            raise UsageLimitExceeded(
                f"max_requests_per_phase ({self.request_limit}) exceeded",
                limit="request limit",
            )
        self.logical_requests_used += 1
        self._count_request_sent()

    def note_sent_again(self) -> None:
        """Count one request sent again, which charges no model request:
        a turn sent again after a transient failure. Raises
        ``UsageLimitExceeded`` past the wall-clock deadline, as
        ``note_request`` does, so no retry starts after it."""
        self.check_deadline()
        self._count_request_sent()

    def check_deadline(self, wait_seconds: float = 0.0) -> None:
        """Raise ``UsageLimitExceeded`` when the wall-clock deadline has
        passed, or would pass during a wait of ``wait_seconds`` before the
        next request: the request after such a wait could not start, so the
        phase ends at once instead of waiting for nothing. The first call
        starts the clock."""
        if self._started_monotonic is None:
            self._started_monotonic = time.monotonic()
        if not self.wall_clock_limit_seconds:
            return
        elapsed = time.monotonic() - self._started_monotonic
        if wait_seconds > 0 and (
            elapsed + wait_seconds >= self.wall_clock_limit_seconds
        ):
            raise UsageLimitExceeded(
                f"max_wall_clock_seconds_per_phase ({self.wall_clock_limit_seconds:.0f}s) "
                f"would be exceeded by the retry wait of {wait_seconds:.0f}s "
                f"(elapsed {elapsed:.0f}s)",
                limit="time limit",
            )
        if elapsed >= self.wall_clock_limit_seconds:
            raise UsageLimitExceeded(
                f"max_wall_clock_seconds_per_phase ({self.wall_clock_limit_seconds:.0f}s) "
                f"exceeded (elapsed {elapsed:.0f}s)",
                limit="time limit",
            )

    def _count_request_sent(self) -> None:
        if self._started_monotonic is None:
            self._started_monotonic = time.monotonic()
        self.requests_used += 1
        if (
            self.heartbeat_every > 0
            and self.heartbeat_fn is not None
            and self.requests_used % self.heartbeat_every == 0
        ):
            elapsed = time.monotonic() - (self._started_monotonic or time.monotonic())
            try:
                self.heartbeat_fn(
                    self.logical_requests_used, self.request_limit or 0, elapsed,
                )
            except Exception:
                # A heartbeat is diagnostic. Never let it break a request.
                pass

    def note_unknown_usage(self) -> None:
        """Count a request sent whose token usage the provider did not
        report: a failed attempt, or a delivered response without usage."""
        self.unknown_usage_requests += 1

    def charge_tokens(self, input_tokens: int, output_tokens: int,
                      cache_hit_tokens: Optional[int] = 0) -> None:
        """Add a model invocation's token usage. ``cache_hit_tokens`` is the
        cached-prefix part of the input (capped at the input), summed for
        the actual-cost figure, or None when the provider did not report the
        split, which ``unknown_cache_requests`` counts."""
        input_count = max(0, int(input_tokens))
        self.input_tokens_used += input_count
        self.output_tokens_used += max(0, int(output_tokens))
        if cache_hit_tokens is None:
            self.unknown_cache_requests += 1
        else:
            self.cache_hit_tokens_used += min(
                input_count, max(0, int(cache_hit_tokens)))

    def note_tokens(self, input_tokens: int, output_tokens: int,
                    cache_hit_tokens: Optional[int] = 0) -> None:
        """Charge token usage after a model invocation (``charge_tokens``).
        Raises ``UsageLimitExceeded`` when any cap (input, output, or their
        combined sum) is reached. ``None`` or ``0`` is unbounded. The cache
        split does not affect the limit checks."""
        self.charge_tokens(input_tokens, output_tokens, cache_hit_tokens)
        if (
            self.input_token_limit
            and self.input_tokens_used >= self.input_token_limit
        ):
            raise UsageLimitExceeded(
                f"max_input_tokens_per_phase ({self.input_token_limit}) exceeded "
                f"(used {self.input_tokens_used})",
                limit="token limit",
            )
        if (
            self.output_token_limit
            and self.output_tokens_used >= self.output_token_limit
        ):
            raise UsageLimitExceeded(
                f"max_output_tokens_per_phase ({self.output_token_limit}) exceeded "
                f"(used {self.output_tokens_used})",
                limit="token limit",
            )
        if (
            self.combined_token_limit
            and (self.input_tokens_used + self.output_tokens_used)
            >= self.combined_token_limit
        ):
            raise UsageLimitExceeded(
                f"max_total_tokens_per_phase ({self.combined_token_limit}) "
                f"exceeded (used "
                f"{self.input_tokens_used + self.output_tokens_used})",
                limit="token limit",
            )


def _extract_token_counts(response: Any) -> Tuple[int, int]:
    """The ``(input, output)`` token counts of an AIMessage's
    ``response_metadata`` (``token_usage`` or ``usage``) or its
    ``usage_metadata``: ``prompt_tokens`` and ``completion_tokens``
    (LangChain, OpenAI) or ``input_tokens`` and ``output_tokens``
    (Anthropic, Ollama). ``(0, 0)`` when the provider supplied none.
    """
    meta = getattr(response, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    if not usage:
        usage = getattr(response, "usage_metadata", None) or {}
    if not isinstance(usage, dict):
        return 0, 0
    input_tokens = (
        usage.get("prompt_tokens")
        or usage.get("input_tokens")
        or 0
    )
    output_tokens = (
        usage.get("completion_tokens")
        or usage.get("output_tokens")
        or 0
    )
    try:
        return int(input_tokens), int(output_tokens)
    except (TypeError, ValueError):
        return 0, 0


def _extract_cache_hit_tokens(response: Any) -> Optional[int]:
    """Cached-prefix input tokens a response reports, or None when it reports
    no cache split. Reads DeepSeek's ``prompt_cache_hit_tokens``, the
    OpenAI-standard ``prompt_tokens_details.cached_tokens`` and the normalized
    ``usage_metadata.input_token_details.cache_read``. They agree where more
    than one appears. Ollama's compatible endpoint and vLLM may report none of
    them, which is unknown, not zero hits. An invalid value is unknown as well.
    Never raises."""

    def nested(block: Dict[str, Any], key: str, field: str) -> Any:
        value = block.get(key)
        return value.get(field) if isinstance(value, dict) else None

    meta = getattr(response, "response_metadata", None) or {}
    for block in (meta.get("token_usage"), meta.get("usage"),
                  getattr(response, "usage_metadata", None)):
        if not isinstance(block, dict):
            continue
        for value in (block.get("prompt_cache_hit_tokens"),
                      nested(block, "prompt_tokens_details", "cached_tokens"),
                      nested(block, "input_token_details", "cache_read")):
            if value is not None:
                return value if type(value) is int and value >= 0 else None
    return None


def _message_transport_payload(
    message: BaseMessage, *, reasoning_field_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Canonicalize the fields that determine one outbound message.

    Only the resulting hashes are persisted. Runtime identifiers and response
    metadata are omitted because the provider does not receive them.
    """
    if isinstance(message, SystemMessage):
        role = "system"
    elif isinstance(message, HumanMessage):
        role = "user"
    elif isinstance(message, AIMessage):
        role = "assistant"
    elif isinstance(message, ToolMessage):
        role = "tool"
    else:
        role = type(message).__name__
    payload: Dict[str, Any] = {"role": role, "content": message.content}
    if isinstance(message, AIMessage):
        if message.tool_calls:
            payload["tool_calls"] = message.tool_calls
        invalid = getattr(message, "invalid_tool_calls", None) or []
        if invalid:
            payload["invalid_tool_calls"] = invalid
        reasoning = {
            key: value
            for key, value in (message.additional_kwargs or {}).items()
            if key in field_names(reasoning_field_names)
        }
        if reasoning:
            payload["reasoning"] = reasoning
    elif isinstance(message, ToolMessage):
        payload["tool_call_id"] = message.tool_call_id
        if getattr(message, "name", None):
            payload["name"] = message.name
    return payload


class UsageLogger:
    """Append-only JSONL log of per-request token usage.

    One JSON line per request sent, failed attempts included::

        {"t_mono_s": 12.5, "clock_domain_id": "c1", "request_idx": N,
         "input_tokens": X,
         "output_tokens": Y, "cache_hit_tokens": H,
         "cache_miss_tokens": X - H, "model_name": "model",
         "finish_reason": "stop"}

    ``target_path`` None gives a no-op instance. The log lives next to
    ``<phase>_chat.json`` and ``<phase>_trace.jsonl``. The optional
    ``prompt_transport_path`` gets each request's message shape.
    """

    def __init__(
        self,
        target_path: Optional[Path],
        *,
        prompt_transport_path: Optional[Path] = None,
        prompt_transport_context: Optional[Dict[str, Any]] = None,
        reasoning_field_names: Optional[Sequence[str]] = None,
        observer: Optional[Callable[..., None]] = None,
    ) -> None:
        """The log starts fresh. Each line is one ``open("a")`` write."""
        self._path = target_path
        self.observer = observer
        self._prompt_transport_path = prompt_transport_path
        self._prompt_transport_context = dict(prompt_transport_context or {})
        self.reasoning_field_names = field_names(reasoning_field_names)
        self._lock = threading.Lock() if target_path is not None else None
        if target_path is not None:
            try:
                target_path.parent.mkdir(parents=True, exist_ok=True)
                target_path.write_text("", encoding="utf-8")
            except Exception:
                self._path = None
                self._lock = None
        if prompt_transport_path is not None:
            try:
                prompt_transport_path.parent.mkdir(parents=True, exist_ok=True)
                prompt_transport_path.write_text("", encoding="utf-8")
            except Exception:
                self._prompt_transport_path = None

    def log_prompt_transport(
        self, *, request_idx: int, messages: Sequence[BaseMessage],
    ) -> None:
        """Record the exact outbound message shape without its contents."""
        if self._prompt_transport_path is None:
            return
        rendered = [_message_transport_payload(message, reasoning_field_names=self.reasoning_field_names)
                    for message in messages]
        context = self._prompt_transport_context
        record = validity.prompt_transport_record(
                phase=context.get("phase", ""),
                request_idx=request_idx,
                messages=rendered,
                prompt_template_version=context.get("prompt_template_version"),
                provider_transport=context.get("provider_transport"),
            )
        for key in ("prompt_composition", "sent_settings"):
            if context.get(key) is not None:
                record[key] = context[key]
        validity.append_jsonl(self._prompt_transport_path, record)

    @property
    def enabled(self) -> bool:
        return self._path is not None

    def log(
        self,
        *,
        request_idx: int,
        input_tokens: int,
        output_tokens: int,
        cache_hit_tokens: Optional[int],
        model_name: Optional[str] = None,
        finish_reason: Optional[str] = None,
        extra: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Append one usage line (nothing when disabled). ``extra`` carries the
        request policy's fields: the outcome class, whether usage is known,
        the model request index and the attempt number.
        ``cache_hit_tokens`` is the prefix the provider charged at its cached
        rate, ``cache_miss_tokens`` the rest, both None when the provider
        reported no split or no usage. They belong on the per-request
        line: only it shows which turns re-sent a prefix, which is how a
        conversation with a stable prefix is recognized."""
        if self._path is None or self._lock is None:
            return
        input_count = max(0, int(input_tokens))
        hit = (None if cache_hit_tokens is None
               else min(input_count, max(0, int(cache_hit_tokens))))
        entry: Dict[str, Any] = {}
        entry.update(timing_fields())
        entry.update({
            "request_idx": int(request_idx),
            "input_tokens": input_count,
            "output_tokens": int(output_tokens),
            "cache_hit_tokens": hit,
            "cache_miss_tokens": None if hit is None else input_count - hit,
        })
        if hit is None:
            # Usage was reported without its cache split.
            entry["cache_usage_known"] = False
        if model_name:
            entry["model_name"] = model_name
        if finish_reason:
            entry["finish_reason"] = finish_reason
        if extra:
            entry.update(extra)
        try:
            line = json.dumps(entry, ensure_ascii=False, default=str)
        except Exception:
            return
        with self._lock:
            try:
                # A lone surrogate, which a model can send, is written as the
                # JSON escape json.loads reads back.
                with open(self._path, "a", encoding="utf-8", errors="backslashreplace") as f:
                    f.write(line + "\n")
            except Exception:
                pass


# Error classification.


_CONTEXT_WINDOW_PHRASES = (
    "token limit",
    "context length",
    "context_length",
    "context window",
    "context_window",
    "exceeds the model's context",
    "this model's maximum context",
    # llama.cpp: "the request exceeds the available context size".
    "context size",
)


def is_context_window_text(text: str) -> bool:
    """True when an error text names a prompt too big for the model's
    context. The one recognizer for the retry rule, a run's status and its
    failure kind, so they cannot disagree."""
    msg = (text or "").lower()
    return any(phrase in msg for phrase in _CONTEXT_WINDOW_PHRASES)


def is_context_window_error(exc: BaseException) -> bool:
    """True when ``exc`` looks like a prompt-too-big failure."""
    return is_context_window_text(str(exc))


#: Transport error types the rerun rule recognizes.
_RERUN_TRANSPORT_ERRORS = (
    httpx.TimeoutException,
    httpx.ConnectError,
    httpx.RemoteProtocolError,
    httpx.ReadError,
    httpx.NetworkError,
)


def _rerun_transient_single(exc: BaseException) -> bool:
    """The transient test for one exception, without its cause
    chain. An instrument fault re-runs the same broken instrument, so it is
    never transient, whatever its text says."""
    if type(exc).__name__ == "MutationInstrumentFault":
        return False
    msg = str(exc).lower()
    if "timed out" in msg or "timeout" in msg or "connection reset" in msg:
        return True
    if (getattr(exc, "status_code", 0) or 0) in (408, 429, 500, 502, 503, 504):
        return True
    if isinstance(exc, _RERUN_TRANSPORT_ERRORS):
        return True
    module = (type(exc).__module__ or "").split(".", 1)[0]
    return module == "openai" and type(exc).__name__ in (
        "APIConnectionError", "APITimeoutError")


def _recoverable_by_rerun(exc: BaseException) -> bool:
    """Whether re-running the graph from its captured state can resolve
    ``exc``: a transient-looking failure anywhere
    in the cause chain."""
    seen: set = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if _rerun_transient_single(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def _is_retryable_agent_error(exc: BaseException) -> bool:
    """True for failures the recovery loop resolves by re-invoking the graph:
    an output-validation rejection (retried with protocol-valid feedback
    appended to the captured state), or another failure escaping the graph,
    such as a tool raising, that looks transient (rerun from the captured
    state). Never retried here: a model request failure, since the model
    node retried its transport failures and one reaching this loop ended in
    a ceiling, a terminal failure or a cancellation. Nor a request whose
    record could not be written (its response was delivered and paid for,
    and a rerun would pay again), a context-window failure (the same
    oversized request again) or a mutation session whose cleanup failed.
    """
    # A rejection is the framework's own verdict on an answer, never a
    # provider's context refusal, so it is retried even when its text names
    # the context.
    if isinstance(exc, OutputValidationError):
        return True
    if is_context_window_error(exc):
        return False
    from testuary.languages.mutation import mutation_cleanup_pending

    # A tool whose mutation session failed its cleanup: that session admits
    # no further work, so a rerun only fails again.
    if mutation_cleanup_pending(exc):
        return False
    if isinstance(exc, (model_requests.TerminalRequestError,
                        model_requests.TransportExhausted,
                        model_requests.RequestCancelled,
                        model_requests.RecordNotWritten)):
        return False
    return _recoverable_by_rerun(exc)


#: The result a tool call gets when its tool raised a temporary error. The
#: recovery loop waits, then sends the answered history to the model again.
#: The error can come after the tool's effect (an edit written, then its
#: record failing), so the text does not claim that nothing happened.
TOOL_FAILURE_RESULT = (
    "The tool failed with a temporary error ({kind}) and may not have "
    "completed. Check whether it took effect before calling it again."
)


class ToolCallsFailed(Exception):
    """A tool of the last response raised a temporary error. Carries the
    answers to every call of that response, a failed call answered with
    ``TOOL_FAILURE_RESULT``, so the recovery loop resends a complete history
    instead of one ending in unanswered tool calls. The first failure is the
    cause."""

    def __init__(self, messages: Sequence[BaseMessage], cause: BaseException):
        super().__init__(f"{type(cause).__name__}: {cause}")
        self.messages = list(messages)
        self.__cause__ = cause


def _failure_recording_tool_node(tools: Sequence[BaseTool]) -> Tuple[ToolNode, List[BaseException]]:
    """A ``ToolNode`` that answers every call, with the list its failures are
    recorded in. Invalid arguments keep ToolNode's own corrective message.
    Any other exception is recorded and its call answered with
    ``TOOL_FAILURE_RESULT``. ``_dispatch_result`` then decides the outcome."""
    failures: List[BaseException] = []

    def handle(e):
        if isinstance(e, ToolInvocationError):
            return e.message
        failures.append(e)
        return TOOL_FAILURE_RESULT.format(kind=type(e).__name__)

    return ToolNode(list(tools), handle_tool_errors=handle), failures


def _dispatch_result(messages: List[BaseMessage], failures: List[BaseException]) -> Dict[str, Any]:
    """The dispatch node's update, or the failure that ends it. A failure the
    recovery loop does not rerun is raised as it is. Temporary failures
    raise ``ToolCallsFailed`` with every call answered."""
    found = list(failures)
    failures.clear()
    if not found:
        return {"messages": messages}
    for exc in found:
        if not _is_retryable_agent_error(exc):
            raise exc
    raise ToolCallsFailed(messages, found[0])


class OutputValidationError(Exception):
    """Raised by the validate node to force a retry: the final-answer call
    is missing, its arguments fail the Pydantic model, or the agent's
    output validator rejects the result. The outer recovery loop re-invokes
    the graph with a corrective prompt.
    """


# Recovery loop: the per-attempt policy, then the loop.


def _prepare_retry_attempt(
    attempt: int,
    state_to_use: Dict[str, Any],
    last_captured: Dict[str, Any],
    last_exc: Optional[BaseException],
    retry_prompt_builder: Optional[
        Callable[[BaseException, List[BaseMessage]], str]
    ],
) -> Tuple[Dict[str, Any], float, Optional[float], str]:
    """Prepare one retry without compacting or otherwise rewriting history.

    Returns ``(state_to_use, wait, wait_hint, state_action)``. Attempt 1 is
    the first invocation, not a recovery: no change, no wait. An
    output-validation failure keeps the complete captured state, answers
    every trailing tool call so the next request satisfies the provider
    protocol, appends the corrective user message and retries at once.
    Another retryable failure keeps the captured state (with a tool
    failure's answers appended) and waits for its hint or the doubling
    backoff. A model request's transport failures never reach this loop.
    The model node retries them. A non-retryable error raises
    ``ValueError``."""
    wait = 0.0
    wait_hint: Optional[float] = None
    if attempt == 1:
        return dict(state_to_use), wait, wait_hint, "initial"

    # Resume from the complete state captured immediately before the failing
    # node. This preserves messages and every non-message graph field.
    next_state = dict(last_captured)
    history = list(next_state.get("messages", []))

    if isinstance(last_exc, OutputValidationError):
        prompt = str(last_exc)
        if retry_prompt_builder is not None:
            built = retry_prompt_builder(last_exc, history)
            if built:
                prompt = built

        # Validation intercepts the synthetic result call before ToolNode can
        # answer it. Answer all calls in the trailing assistant message before
        # appending user feedback, otherwise OpenAI-compatible APIs reject the
        # conversation as an assistant tool call without matching results.
        trailing = history[-1] if history else None
        if isinstance(trailing, AIMessage) and trailing.tool_calls:
            tool_results: List[BaseMessage] = []
            for index, call in enumerate(trailing.tool_calls, start=1):
                call_id = (
                    call.get("id")
                    if isinstance(call, dict)
                    else getattr(call, "id", None)
                )
                name = (
                    call.get("name")
                    if isinstance(call, dict)
                    else getattr(call, "name", None)
                )
                if not call_id:
                    # Provider-produced tool calls have identifiers. Refuse an
                    # invalid captured shape instead of deleting or rewriting
                    # the assistant message to manufacture one.
                    raise ValueError(
                        "cannot recover rejected output: trailing tool call "
                        f"{index} has no id"
                    )
                tool_results.append(ToolMessage(
                    content=(
                        "The framework did not execute this call because the "
                        "submitted final result failed validation."
                    ),
                    tool_call_id=str(call_id),
                    name=str(name) if name else None,
                ))
            history.extend(tool_results)
        history.append(HumanMessage(content=prompt))
        next_state["messages"] = history
        return next_state, wait, wait_hint, "validation_feedback_appended"

    if last_exc is not None and _is_retryable_agent_error(last_exc):
        # The captured state is kept. After a tool failure it ends in the
        # response whose calls failed, so their answers are appended: a
        # history with unanswered tool calls is refused by strict providers.
        # The wait prefers the failure's hint over the doubling backoff,
        # which is capped at 60 s.
        if isinstance(last_exc, ToolCallsFailed):
            next_state["messages"] = history + list(last_exc.messages)
        # A tool failure's hint is on the tool's own exception.
        wait_hint = model_requests.retry_hint_seconds(
            last_exc.__cause__ if isinstance(last_exc, ToolCallsFailed) else last_exc)
        wait = wait_hint if wait_hint is not None else min(
            10 * (2 ** (attempt - 2)), 60,
        )
        return next_state, wait, wait_hint, (
            "tool_failures_answered" if isinstance(last_exc, ToolCallsFailed)
            else "state_preserved")

    raise ValueError(
        "retry preparation received a non-retryable error: "
        f"{type(last_exc).__name__ if last_exc else 'None'}"
    )


def _log_retry_attempt(
    *,
    log: Optional[Callable[[str], None]],
    trace_logger: Optional["TraceLogger"],
    attempt: int,
    max_retries: int,
    last_exc: Optional[BaseException],
    wait: float,
    wait_hint: Optional[float],
    state_action: str,
) -> None:
    """The log line and trace event of a recovery attempt (attempt > 1). The
    log names the wait, ``(server hint)``, ``(backoff)`` or ``(no wait)``."""
    kind = (
        "OutputValidationError"
        if isinstance(last_exc, OutputValidationError)
        else "context_window"
        if last_exc is not None and is_context_window_error(last_exc)
        else (
            "transient"
            if last_exc is not None and _recoverable_by_rerun(last_exc)
            else type(last_exc).__name__ if last_exc else "unknown"
        )
    )
    if log:
        wait_label = (
            f", waited={wait:.1f}s "
            + (
                "(server hint)" if wait_hint is not None
                else "(no wait)" if wait == 0
                else "(backoff)"
            )
        )
        log(
            f"Recovery attempt {attempt}/{max_retries} "
            f"(kind={kind}, state={state_action}{wait_label})"
        )
    if trace_logger is not None:
        trace_logger.recovery_attempt(
            attempt=attempt,
            max_attempts=max_retries,
            kind=kind,
            exception_type=type(last_exc).__name__ if last_exc else None,
            exception_message=str(last_exc) if last_exc else None,
        )


def _emit_intermediate(
    callback: Optional[Callable[[Dict[str, Any]], None]],
    event: Dict[str, Any],
) -> None:
    """Call the per-super-step callback, ignoring its errors: an observing
    hook never breaks the run."""
    if callback is not None:
        try:
            callback(event)
        except Exception:
            pass


def _max_retries_error_msg(
    max_retries: int, last_exc: Optional[BaseException],
) -> str:
    return (
        f"max_retries ({max_retries}) reached. "
        f"Final error: {type(last_exc).__name__}: "
        f"{str(last_exc)[:1000] if last_exc else 'unknown'}"
    )


def run_graph_with_retries(
    graph: Any,
    initial_state: Dict[str, Any],
    *,
    config: Optional[Dict[str, Any]] = None,
    max_retries: int,
    log: Optional[Callable[[str], None]] = None,
    retry_prompt_builder: Optional[
        Callable[[BaseException, List[BaseMessage]], str]
    ] = None,
    on_intermediate_state: Optional[
        Callable[[Dict[str, Any]], None]
    ] = None,
    on_before_retry: Optional[
        Callable[[int, Dict[str, Any], BaseException], None]
    ] = None,
    trace_logger: Optional["TraceLogger"] = None,
    failure_out: Optional[Dict[str, BaseException]] = None,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    """Synchronous unified recovery loop.

    ``failure_out``, when given, receives a non-retryable exception under
    ``"exception"``, so the caller can judge its kind.

    Wraps ``graph.stream(initial_state, config)`` with retry classification:

    * ``UsageLimitExceeded`` (the designed limit stop) and a failed
      mutation-session cleanup propagate.
    * Returns ``(result, None)`` on success.
    * Returns ``(None, error_msg)`` once ``max_retries`` graph attempts are
      spent or on any non-retryable error.

    An output-validation failure re-invokes the graph from the complete
    state captured up to the failing node, with matched tool results and a
    corrective user prompt appended. Nothing here compacts messages, strips
    reasoning, changes workflow state or selects tools. The model node
    retries a model request's transport failures, so one reaching this loop
    is terminal, as is a context-window failure: rebuilding the context
    belongs to the graph's pre-model hook alone.

    ``on_intermediate_state`` is an optional callback the caller can
    use to see the captured state of every super-step, so the chat log can
    be flushed even on a hard failure.

    ``on_before_retry`` receives the failed attempt number, its last captured
    state and the exception of each attempt that is retried. Errors of both
    callbacks are ignored: logging never changes the run.

    Backoff: exponential, capped at 60 s.
    It starts at 10 s. A failure's retry hint replaces it, and a validation
    rejection retries at once.
    """
    last_exc: Optional[BaseException] = None
    state_to_use = dict(initial_state)
    # Most recent intermediate state observed during the current attempt.
    # A retry resumes from this complete state.
    last_captured: Dict[str, Any] = dict(initial_state)

    for attempt in range(1, max_retries + 1):
        try:
            state_to_use, wait, wait_hint, state_action = _prepare_retry_attempt(
                attempt, state_to_use, last_captured, last_exc,
                retry_prompt_builder,
            )
        except Exception as exc:
            return None, (
                f"recovery-state error: {type(exc).__name__}: "
                f"{str(exc)[:1000]}"
            )
        if attempt > 1:
            if wait > 0:
                time.sleep(wait)
            _log_retry_attempt(
                log=log, trace_logger=trace_logger, attempt=attempt,
                max_retries=max_retries, last_exc=last_exc, wait=wait,
                wait_hint=wait_hint, state_action=state_action,
            )

        # Reset capture for this attempt. The stream below updates it per event.
        last_captured = dict(state_to_use)
        try:
            for event in graph.stream(
                state_to_use, config=config, stream_mode="values",
            ):
                last_captured = event
                _emit_intermediate(on_intermediate_state, event)
            return last_captured, None
        except UsageLimitExceeded:
            raise
        except Exception as exc:
            from testuary.languages.mutation import mutation_cleanup_pending

            # A failed session cleanup ends the agent's work, and its phase
            # propagates it, rather than an agent failure the run goes on
            # after.
            if mutation_cleanup_pending(exc):
                raise
            last_exc = exc
            if not _is_retryable_agent_error(exc):
                if failure_out is not None:
                    failure_out["exception"] = exc
                return None, (
                    f"non-retryable error: {type(exc).__name__}: "
                    f"{str(exc)[:1000]}"
                )
            if attempt < max_retries and on_before_retry is not None:
                try:
                    on_before_retry(attempt, dict(last_captured), exc)
                except Exception:
                    pass
            continue

    return None, _max_retries_error_msg(max_retries, last_exc)


# ReAct graph factory.


def _malformed_call_responses(
    last_ai: Optional[AIMessage],
) -> List[ToolMessage]:
    """Synthetic answers to the malformed calls of the last AIMessage. A
    call whose arguments are not valid JSON parses into
    ``invalid_tool_calls``, never into ``tool_calls``, so no ToolNode
    dispatches it, but the API serializer still sends it as an assistant
    ``tool_calls`` entry, and an unanswered one makes the provider reject
    the next request.
    """
    if last_ai is None:
        return []
    out: List[ToolMessage] = []
    for tc in getattr(last_ai, "invalid_tool_calls", None) or []:
        err = tc.get("error") if isinstance(tc, dict) else None
        out.append(ToolMessage(
            content=(
                "MALFORMED tool call: the arguments were not valid "
                "JSON, so nothing was executed"
                + (" (" + _limits.cut_text(str(err), _limits.value("malformed_args_chars"),
                                           what="the error", keep="head", sep=" ") + ")"
                   if err else "")
                + ". Re-emit the call with valid JSON arguments."
            ),
            tool_call_id=(
                (tc.get("id") if isinstance(tc, dict) else None) or ""
            ),
            name=(
                (tc.get("name") if isinstance(tc, dict) else None)
                or "unknown"
            ),
        ))
    return out


def _finish_reason(response: Any) -> Optional[str]:
    """The finish reason a reply reports: OpenAI-compatible
    ``finish_reason``, Ollama's ``done_reason``, or None."""
    meta = getattr(response, "response_metadata", None) or {}
    reason = meta.get("finish_reason") or meta.get("done_reason")
    return reason if isinstance(reason, str) and reason else None


def _empty_reply(response: Any) -> bool:
    """Whether a delivered reply is no answer and is sent again
    (``model_requests.unanswered_reply``): no text and no tool call (valid
    or not), and the provider ended it while it loaded the model or
    reported 0 input and 0 output tokens."""
    content = getattr(response, "content", None)
    if isinstance(content, list):
        text = "".join(part.get("text", "") if isinstance(part, dict) else str(part)
                       for part in content)
    else:
        text = str(content or "")
    return model_requests.unanswered_reply(
        text=text,
        tool_call=bool(getattr(response, "tool_calls", None)
                       or getattr(response, "invalid_tool_calls", None)),
        finish_reason=_finish_reason(response), usage=_usage_from_response(response))


def _usage_from_response(response: Any) -> Optional[model_requests.Usage]:
    """The usage a delivered response reports, or ``None`` when the provider
    reported none. A missing report is unknown usage, not zero."""
    meta = getattr(response, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    if not usage:
        usage = getattr(response, "usage_metadata", None) or {}
    if not isinstance(usage, dict) or not usage:
        return None
    if not any(k in usage for k in (
        "prompt_tokens", "input_tokens", "completion_tokens", "output_tokens",
    )):
        return None
    in_tok, out_tok = _extract_token_counts(response)
    return model_requests.reported_usage(in_tok, out_tok, _extract_cache_hit_tokens(response))


def _run_model_turn(
    state: Any,
    *,
    bind: Callable[[Any], Tuple[Any, List[str]]],
    pre_model_hook: Optional[Callable[[Any], Optional[Dict[str, Any]]]],
    usage_tracker: Optional["UsageTracker"],
    usage_logger: Optional["UsageLogger"],
    trace: "TraceLogger",
    request_account: Optional[model_requests.RequestAccount],
    on_send_again: Optional[
        Callable[[int, BaseException, float, List[BaseMessage]], None]
    ],
    reasoning_field_names: Sequence[str] = DEFAULT_FIELDS,
) -> Dict[str, Any]:
    """One model turn of an agent mode's model node: the pre-hook, the
    charge, the request, sent again after connection errors, the usage accounting
    and the trace events. ``bind(state)`` returns the tool-bound model and
    the bound tool names. The request policy retries a transient failure in
    place, without leaving the graph, and charges the usage tracker once per
    turn (once per attempt with count_retries on). A terminal
    failure or a retry ceiling raises out of the node with the provider's
    error, which the recovery loop reports as non-retryable.
    """
    pre_update: Dict[str, Any] = {}
    if pre_model_hook is not None:
        pre_update = pre_model_hook(state) or {}
    pass_state = {**state, **pre_update} if pre_update else state
    account = (
        request_account if request_account is not None
        else model_requests.RequestAccount(model_requests.RetrySettings())
    )
    charge_every_attempt = account.settings.charge_every_attempt
    bound, bound_names = bind(pass_state)
    timing: Dict[str, float] = {}

    def observe(request_state: str, message: str, label: str = "") -> None:
        observer = getattr(usage_logger, "observer", None)
        if observer is None:
            return
        from testuary.presentation import Usage

        # These are the phase's existing counters, not a display limit.
        known = usage_tracker is not None and usage_tracker.unknown_usage_requests == 0
        observer(operation=f"request/{account.logical_requests}", state=request_state,
                 message=message, label=label, usage=Usage(
                     logical_requests=(usage_tracker.logical_requests_used if usage_tracker is not None else account.logical_requests),
                     physical_requests=(usage_tracker.requests_used if usage_tracker is not None else account.physical_requests),
                     input_tokens=usage_tracker.input_tokens_used if known else None,
                     output_tokens=usage_tracker.output_tokens_used if known else None,
                     # The phase's configured limit. Zero or None is unbounded.
                     logical_request_limit=(usage_tracker.request_limit or None
                                            if usage_tracker is not None else None),
                 ))

    def before_attempt(attempt: int) -> None:
        # Charged before the call so a request over the limit is never
        # started. ``UsageLimitExceeded`` propagates through the policy
        # untouched, up to the entry point as the designed limit stop.
        if usage_tracker is None:
            return
        if attempt == 1 or charge_every_attempt:
            usage_tracker.note_request()
        else:
            usage_tracker.note_sent_again()

    def send(attempt: int) -> Any:
        # The soft limit warning is stable text. Whether it is appended
        # depends on the requests left under the limit, which the charge above may
        # have changed, so the outbound messages are built per attempt.
        msgs_in = _append_limit_warning(
            list(pass_state["messages"]), usage_tracker,
        )
        request_idx = usage_tracker.requests_used if usage_tracker is not None else 0
        if usage_logger is not None:
            usage_logger.log_prompt_transport(
                request_idx=request_idx, messages=msgs_in,
            )
        trace.model_invoke_start(
            bound_tool_names=bound_names,
            message_count=len(msgs_in),
            last_message_type=type(msgs_in[-1]).__name__ if msgs_in else None,
        )
        observe("running", "Sending model request")
        invoke_start = time.perf_counter()
        try:
            with phase_state.timed_part(phase_state.TIME_MODEL):
                response = bound.invoke(msgs_in)
        finally:
            timing["invoke"] = time.perf_counter() - invoke_start
        if _empty_reply(response):
            raise model_requests.empty_reply_error(
                _finish_reason(response), _usage_from_response(response))
        return response

    def failed_usage(exc: Optional[BaseException]) -> Optional[model_requests.Usage]:
        # The reported usage of an unanswered reply, else unknown.
        return exc.usage if isinstance(exc, model_requests.EmptyReply) else None

    def on_retry(attempt: int, exc: BaseException, wait: float) -> None:
        # A failed attempt: one line in the usage log so the physical
        # request is visible, then its usage (unknown, or an unanswered
        # reply's reported usage), whose charge can reach a token limit, a
        # trace event, and the caller's hook (log line and chat snapshot)
        # before the wait.
        reported = failed_usage(exc)
        if usage_logger is not None:
            usage_logger.log(
                request_idx=(
                    usage_tracker.requests_used if usage_tracker is not None else 0
                ),
                input_tokens=reported.input_tokens if reported is not None else 0,
                output_tokens=reported.output_tokens if reported is not None else 0,
                cache_hit_tokens=(reported.cache_hit_tokens if reported is not None else None),
                extra={
                    "outcome": model_requests.TRANSIENT,
                    "usage_known": reported is not None,
                    "logical_request_idx": (
                        usage_tracker.logical_requests_used
                        if usage_tracker is not None else 0
                    ),
                    "attempt": attempt,
                    "error_type": type(exc).__name__,
                    "wait_s": round(wait, 3),
                },
            )
        if usage_tracker is not None:
            if reported is None:
                usage_tracker.note_unknown_usage()
            else:
                usage_tracker.note_tokens(reported.input_tokens, reported.output_tokens,
                                          reported.cache_hit_tokens)
        trace.recovery_attempt(
            attempt=attempt, max_attempts=account.settings.consecutive_limit,
            kind="transient", exception_type=type(exc).__name__,
            exception_message=str(exc),
        )
        # A wait that would end past the wall-clock deadline is not started,
        # since the retry after it could not be sent: the phase ends here as
        # a designed limit stop, with the failed attempt recorded above.
        if usage_tracker is not None:
            usage_tracker.check_deadline(wait)
        if on_send_again is not None:
            try:
                on_send_again(attempt, exc, wait, list(pass_state["messages"]))
            except Exception:
                pass

        observe("retrying", "Temporary failure, sending the request again unchanged")
        if wait > 0:
            observe("waiting", f"Waiting {wait:.1f}s before sending it again")

    try:
        outcome = account.execute(
            send, usage_of=_usage_from_response,
            before_attempt=before_attempt, on_retry=on_retry,
            terminal=is_context_window_error,
        )
    except model_requests.TerminalRequestError as exc:
        if usage_tracker is not None:
            usage_tracker.note_unknown_usage()
        if usage_logger is not None:
            usage_logger.log(
                request_idx=(
                    usage_tracker.requests_used if usage_tracker is not None else 0
                ),
                input_tokens=0, output_tokens=0, cache_hit_tokens=None,
                extra={
                    "outcome": model_requests.TERMINAL,
                    "usage_known": False,
                    "error_type": type(exc.__cause__).__name__
                    if exc.__cause__ is not None else "unknown",
                },
            )
        observe("failed", "Model request failed")
        raise
    except model_requests.TransportExhausted as exc:
        # The attempt that reached a ceiling failed like the retried ones
        # before it, whose records ``on_retry`` writes: its usage is unknown,
        # or an unanswered reply's reported usage, and it has its line in
        # the usage log.
        reported = failed_usage(exc.__cause__)
        if usage_tracker is not None:
            if reported is None:
                usage_tracker.note_unknown_usage()
            else:
                usage_tracker.charge_tokens(reported.input_tokens, reported.output_tokens,
                                            reported.cache_hit_tokens)
        if usage_logger is not None:
            usage_logger.log(
                request_idx=(
                    usage_tracker.requests_used if usage_tracker is not None else 0
                ),
                input_tokens=reported.input_tokens if reported is not None else 0,
                output_tokens=reported.output_tokens if reported is not None else 0,
                cache_hit_tokens=(reported.cache_hit_tokens if reported is not None else None),
                extra={
                    "outcome": model_requests.TRANSIENT,
                    "usage_known": reported is not None,
                    "logical_request_idx": (
                        usage_tracker.logical_requests_used
                        if usage_tracker is not None else 0
                    ),
                    "attempt": exc.attempt,
                    "error_type": type(exc.__cause__).__name__
                    if exc.__cause__ is not None else "unknown",
                    "ends_phase": exc.reason,
                },
            )
        observe("failed", model_requests.exhausted_text(exc, account.service))
        raise
    except model_requests.RequestCancelled:
        observe("cancelled", "Model request cancelled")
        raise
    except KeyboardInterrupt:
        observe("cancelled", "Stopped by you")
        raise
    except UsageLimitExceeded as exc:
        # The line names the limit that ended the phase, not a request number.
        observe("limit_reached", _limit_text(exc, usage_tracker), label="Model requests")
        raise
    except BaseException:
        observe("failed", "Model request ended with an error")
        raise
    response = outcome.value
    invoke_dur = timing.get("invoke", 0.0)

    tool_call_names: List[str] = []
    content_str = ""
    reasoning_len = 0
    finish_reason = None
    model_name = None
    if isinstance(response, AIMessage):
        tool_call_names = [
            (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "?"))
            for tc in (response.tool_calls or [])
        ]
        content_str = (
            response.content if isinstance(response.content, str)
            else str(response.content)
        )
        additional = getattr(response, "additional_kwargs", None) or {}
        for k in reasoning_field_names:
            v = additional.get(k)
            if v:
                reasoning_len += len(str(v))
        metadata = getattr(response, "response_metadata", None) or {}
        finish_reason = metadata.get("finish_reason")
        model_name = metadata.get("model_name") or metadata.get("model")

    # Tokens are charged before the trace event, so a token-cap overrun
    # raises ``UsageLimitExceeded`` without a model_invoke_end event that
    # reports a success.
    usage = outcome.usage
    in_tok = usage.input_tokens if usage is not None else 0
    out_tok = usage.output_tokens if usage is not None else 0
    cache_hit = usage.cache_hit_tokens if usage is not None else None
    if usage_logger is not None:
        usage_logger.log(
            request_idx=(
                usage_tracker.requests_used if usage_tracker is not None else 0
            ),
            input_tokens=in_tok,
            output_tokens=out_tok,
            cache_hit_tokens=cache_hit,
            model_name=model_name,
            finish_reason=finish_reason,
            extra={
                "outcome": model_requests.DELIVERED,
                "usage_known": usage is not None,
                "logical_request_idx": (
                    usage_tracker.logical_requests_used
                    if usage_tracker is not None else 0
                ),
                "attempt": outcome.attempts,
            },
        )
    try:
        if usage_tracker is not None:
            if usage is not None:
                usage_tracker.note_tokens(in_tok, out_tok, cache_hit)
            else:
                usage_tracker.note_unknown_usage()
    except UsageLimitExceeded as exc:
        observe("limit_reached", f"Reply received. {_limit_text(exc, usage_tracker)}")
        raise

    trace.model_invoke_end(
        duration_s=invoke_dur,
        tool_calls=tool_call_names,
        content_preview=content_str[:300],
        content_len=len(content_str),
        reasoning_content_len=reasoning_len,
        finish_reason=finish_reason,
    )

    observe("succeeded", "Reply received")

    # The model's response goes through the add_messages reducer.
    # The pre-hook's other fields go into the state too, so a caller can
    # publish state updates (counters, telemetry) without a post-hook.
    out: Dict[str, Any] = {"messages": [response]}
    for k, v in pre_update.items():
        if k != "messages":
            out[k] = v
    return out


def _find_final_answer_call(
    messages: List[BaseMessage], final_tool_name: str,
) -> Optional[Dict[str, Any]]:
    """The arguments of the final-answer call in the model's latest reply,
    or None when that reply makes none. An earlier reply's call was judged
    when it was made, so it is never judged again."""
    latest = next((message for message in reversed(messages)
                   if isinstance(message, AIMessage)), None)
    for call in (latest.tool_calls or ()) if latest is not None else ():
        name = call.get("name") if isinstance(call, dict) else getattr(call, "name", None)
        if name == final_tool_name:
            args = call.get("args") if isinstance(call, dict) else getattr(call, "args", None)
            return args or {}
    return None


def _make_validate_node(
    *,
    structured_output: Optional[Type[BaseModel]],
    output_validator: Optional[Callable[[Any, BaseModel], Optional[str]]],
    final_tool_name: str,
    trace: TraceLogger,
) -> Callable[[Any], Dict[str, Any]]:
    def validate_node(state: Any) -> Dict[str, Any]:
        messages = state["messages"]
        result_obj: Optional[BaseModel] = None
        if structured_output is not None:
            args = _find_final_answer_call(messages, final_tool_name)
            if args is None:
                trace.validator_check(
                    accepted=False,
                    rejection_reason="model did not emit the final-answer tool call before exit",
                )
                raise OutputValidationError(
                    "model did not emit the final-answer tool call before exit"
                )
            try:
                result_obj = structured_output(**args)
            except Exception as exc:
                # Constructor failures remain retryable output rejections.
                reason = (
                    f"structured output rejected by Pydantic: "
                    f"{type(exc).__name__}: "
                    + _limits.cut_text(str(exc), _limits.value("rejection_chars"),
                                       what="the error", keep="head", sep=" ")
                )
                trace.validator_check(accepted=False, rejection_reason=reason)
                raise OutputValidationError(reason) from exc
        if output_validator is not None and result_obj is not None:
            err = output_validator(state, result_obj)
            if err:
                trace.validator_check(accepted=False, rejection_reason=err)
                raise OutputValidationError(err)
        trace.validator_check(accepted=True)
        return {"result": result_obj}

    return validate_node


def _tool_call_config(serial: bool) -> Optional[Dict[str, Any]]:
    """The run configuration of a tool dispatch, called inside the dispatch
    node. ``ToolNode`` maps a response's calls over an executor sized by
    ``max_concurrency``. One worker runs them in their order. The node's own
    configuration, which carries the graph runtime, is kept."""
    if not serial:
        return None
    from langchain_core.runnables.config import ensure_config

    return {**ensure_config(), "max_concurrency": 1}


def build_react_graph(
    *,
    model: BaseChatModel,
    agent_mode_tools: Optional[Dict[str, Sequence[BaseTool]]] = None,
    resolve_agent_mode: Optional[Callable[[Any], str]] = None,
    pre_model_hook: Optional[Callable[[Any], Optional[Dict[str, Any]]]] = None,
    structured_output: Optional[Type[BaseModel]] = None,
    output_validator: Optional[Callable[[Any, BaseModel], Optional[str]]] = None,
    state_type: Type[TypedDict],
    final_tool_name: str = "emit_final_answer",
    usage_tracker: Optional[UsageTracker] = None,
    trace_logger: Optional[TraceLogger] = None,
    usage_logger: Optional[UsageLogger] = None,
    classify_node: Callable[[Any], Optional[Dict[str, Any]]],
    request_account: Optional[model_requests.RequestAccount] = None,
    on_send_again: Optional[
        Callable[[int, BaseException, float, List[BaseMessage]], None]
    ] = None,
    reasoning_field_names: Optional[Sequence[str]] = None,
    serial_tool_calls: bool = False,
    unbound_tool_reply: Optional[Callable[[str], Optional[str]]] = None,
) -> Any:
    """Compile a ReAct-style LangGraph for one agent, with one model and
    dispatch node pair per agent mode::

        START -> route_by_agent_mode -> model_decide__X -> tool_dispatch__X
                                               |                   |
                                               v                   v
                                     validate -> END     classify -> route_by_agent_mode

    One ``model_decide`` per agent mode, each with its tools bound at
    construction. A call of a tool outside the current mode's subset runs
    nothing: ``ToolNode`` knows only the bound tools and answers an
    unknown name with an error message (or ``unbound_tool_reply``'s
    answer). The graph's topology enforces the gating.

    Components the caller supplies:

      * ``model``: a ``ChatOpenAI`` (or compatible) instance.
      * ``agent_mode_tools`` maps each agent mode's name to its tools, and
        ``resolve_agent_mode(state)`` returns the current mode's name
        (usually the classify node's function). Without either the build
        raises ``ValueError``. An unknown name or a raising resolver routes
        to the first mode.
      * ``pre_model_hook(state)``: optional, runs right before the model
        request. Its returned partial state can replace the messages sent
        this turn (for example to add a state-marker
        SystemMessage). Its other fields are written to the graph state.
      * ``structured_output``: optional Pydantic class. When given, a
        final-answer tool named ``final_tool_name`` with its schema is bound
        in every agent mode, so the model can finish from anywhere. The
        call's arguments become the structured result.
      * ``output_validator(state, parsed_result)``: optional. Returns
        ``None`` on success or an error message, which raises
        ``OutputValidationError`` for the outer recovery loop.
      * ``classify_node(state)``: the node after every dispatch, whose
        update the next agent mode is resolved from.
      * ``state_type``: the TypedDict subclass used for the graph's
        state schema.
      * ``final_tool_name``: name of the synthetic final-answer tool.
      * ``serial_tool_calls``: run the tool calls of one response one at a
        time, in their order. Otherwise ``ToolNode`` runs them in parallel
        threads.
      * ``unbound_tool_reply(name)``: optional. The answer
        to a call of a tool the current agent mode does not bind, or None to
        keep ToolNode's own error.
      * ``usage_tracker``, ``usage_logger``, ``trace_logger``,
        ``request_account``, ``on_send_again`` and
        ``reasoning_field_names`` go to every model turn
        (``_run_model_turn``).

    Returns the compiled graph, which ``run_graph_with_retries`` runs.
    """
    reasoning_field_names = field_names(reasoning_field_names)
    if not agent_mode_tools:
        raise ValueError("agent_mode_tools must declare at least one agent_mode.")
    if resolve_agent_mode is None:
        raise ValueError(
            "agent_mode_tools requires resolve_agent_mode to route between "
            "phase nodes."
        )
    agent_mode_tools = {k: list(v) for k, v in agent_mode_tools.items()}
    trace = trace_logger if trace_logger is not None else _NULL_TRACE
    agent_mode_names = list(agent_mode_tools.keys())

    # The final-answer tool, built once and bound in every agent mode so the
    # model can finish from anywhere.
    final_tool: Optional[BaseTool] = None
    if structured_output is not None:
        final_tool = _make_final_answer_tool(
            structured_output, final_tool_name
        )

    def _agent_mode_node_names(agent_mode: str) -> Tuple[str, str]:
        return f"model_decide__{agent_mode}", f"tool_dispatch__{agent_mode}"

    def _make_model_decide(full_bound: List[BaseTool]):
        """Closure capturing one agent mode's bound tools around the
        shared model turn (pre-hook, limit enforcement, telemetry, etc.)."""
        bound_model = model.bind_tools(full_bound)
        bound_names_static = [getattr(t, "name", "?") for t in full_bound]

        def model_decide(state: Any) -> Dict[str, Any]:
            return _run_model_turn(
                state,
                bind=lambda _s: (bound_model, bound_names_static),
                pre_model_hook=pre_model_hook,
                usage_tracker=usage_tracker,
                usage_logger=usage_logger,
                trace=trace,
                request_account=request_account,
                on_send_again=on_send_again,
                reasoning_field_names=reasoning_field_names,
            )

        return model_decide

    def _make_tool_dispatch(full: List[BaseTool]):
        """The dispatch node with the mode's bound tools and the final-answer
        tool (whose calls route to validate first). A call of a tool the mode
        does not bind gets ``unbound_tool_reply``'s answer, if any, else
        ToolNode's error, so the loop continues. Tool failures go through
        ``_dispatch_result``. Malformed calls (``invalid_tool_calls``)
        never reach the ToolNode. Each one is answered here with
        corrective feedback so no serialized call goes unanswered."""
        node, failures = _failure_recording_tool_node(full)
        full_names = {t.name for t in full}

        def dispatch(state: Any) -> Dict[str, Any]:
            msgs = state.get("messages", [])
            last_ai: Optional[AIMessage] = None
            for m in reversed(msgs):
                if isinstance(m, AIMessage):
                    last_ai = m
                    break
            malformed = _malformed_call_responses(last_ai)
            if last_ai is not None and last_ai.tool_calls:
                # A call of a tool this agent mode does not bind is answered with
                # what makes it available, when the caller says so.
                kept, refused = [], []
                for tc in last_ai.tool_calls:
                    reply = (None if unbound_tool_reply is None or tc["name"] in full_names
                             else unbound_tool_reply(tc["name"]))
                    if reply is None:
                        kept.append(tc)
                    else:
                        refused.append(ToolMessage(
                            content=reply, tool_call_id=tc.get("id") or "",
                            name=tc["name"], status="error"))
                if not kept:
                    return {"messages": refused + malformed}
                if refused:
                    filtered = last_ai.model_copy(update={"tool_calls": kept})
                    state = {**state, "messages": [
                        filtered if m is last_ai else m for m in msgs]}
                failures.clear()
                result = node.invoke(state, _tool_call_config(serial_tool_calls))
                executed = (
                    result.get("messages", [])
                    if isinstance(result, dict) else []
                )
                answered = list(executed)
                if refused:
                    # The answers in the order of the calls.
                    order = [tc.get("id") for tc in last_ai.tool_calls]
                    answered = sorted(
                        answered + refused,
                        key=lambda m: order.index(m.tool_call_id)
                        if m.tool_call_id in order else len(order))
                return _dispatch_result(answered + malformed, failures)
            return {"messages": malformed}

        return dispatch

    validate_node = _make_validate_node(
        structured_output=structured_output, output_validator=output_validator,
        final_tool_name=final_tool_name, trace=trace,
    )

    # Route after each model_decide: validate when the model emitted
    # the final-answer tool (or any non-tool response), else dispatch
    # to the current agent_mode's tool node.
    def make_route_after_model(agent_mode: str):
        _, dispatch_name = _agent_mode_node_names(agent_mode)

        def route(state: Any) -> str:
            msgs = state["messages"]
            last = msgs[-1] if msgs else None
            if not isinstance(last, AIMessage):
                return "validate"
            if not last.tool_calls:
                # A message carrying only malformed calls goes to
                # dispatch, where each one is answered with corrective
                # feedback. Routing it to validate would leave the
                # serialized calls unanswered.
                if getattr(last, "invalid_tool_calls", None):
                    return dispatch_name
                return "validate"
            for tc in last.tool_calls:
                name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None)
                if name == final_tool_name:
                    return "validate"
            return dispatch_name

        return route

    # From START and from classify to the resolved agent mode's
    # model_decide. An unknown name or a raising resolver falls back to the
    # first mode.
    def route_by_agent_mode(state: Any) -> str:
        try:
            agent_mode = resolve_agent_mode(state)
        except Exception:
            agent_mode = agent_mode_names[0]
        if agent_mode not in agent_mode_tools:
            agent_mode = agent_mode_names[0]
        return _agent_mode_node_names(agent_mode)[0]

    graph = StateGraph(state_type)
    graph.add_node("validate", validate_node)

    graph.add_node("classify", classify_node)

    # Build per-agent_mode nodes + their routing.
    for agent_mode, bound in agent_mode_tools.items():
        model_name, dispatch_name = _agent_mode_node_names(agent_mode)
        # The mode's tools, then the final-answer tool: the bind order.
        full = list(bound)
        if final_tool is not None and final_tool not in full:
            full.append(final_tool)
        graph.add_node(model_name, _make_model_decide(list(full)))
        graph.add_node(dispatch_name, _make_tool_dispatch(list(full)))
        # Per-agent_mode model -> dispatch or validate
        graph.add_conditional_edges(
            model_name,
            make_route_after_model(agent_mode),
            {
                dispatch_name: dispatch_name,
                "validate": "validate",
            },
        )
        # Per-agent_mode dispatch -> classify -> agent_mode router
        graph.add_edge(dispatch_name, "classify")

    # classify -> agent_mode router (which picks the next agent_mode's
    # model_decide based on the just-classified state).
    graph.add_conditional_edges(
        "classify",
        route_by_agent_mode,
        {
            _agent_mode_node_names(p)[0]: _agent_mode_node_names(p)[0]
            for p in agent_mode_tools
        },
    )

    # Entry: route from START to the initial agent_mode's model_decide.
    graph.add_conditional_edges(
        START,
        route_by_agent_mode,
        {
            _agent_mode_node_names(p)[0]: _agent_mode_node_names(p)[0]
            for p in agent_mode_tools
        },
    )
    graph.add_edge("validate", END)

    return graph.compile()


def _make_final_answer_tool(model_cls: Type[BaseModel], tool_name: str) -> BaseTool:
    """A stub tool whose schema is ``model_cls``: its body never runs (the
    graph routes its calls to the validate node), but binding it gives the
    model a typed schema for its structured result, LangChain's usual
    final-answer pattern inside a tool-calling loop.
    """
    from langchain_core.tools import StructuredTool

    def _emit(**kwargs: Any) -> str:
        # Should not run: the graph short-circuits final_answer to the
        # validate node before reaching tool_dispatch.
        return ""

    return StructuredTool.from_function(
        func=_emit,
        name=tool_name,
        description=(
            "Emit the final structured result for this run. Call this "
            "tool ONLY when you have everything you need to return; "
            "the args become the run's output."
        ),
        args_schema=model_cls,
    )
