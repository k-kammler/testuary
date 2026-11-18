"""Outcome classes, retries, waits and accounting for model requests.

One policy for every phase. A *model request* is one opportunity the
phase grants: a starter suite's generation attempt, or a turn of the coverage or
mutation phase. Each *request sent* is one attempt to obtain the response
for it, and ends in one of three classes:

* ``transient``: no usable response was delivered because of a connection
  error, a timeout, or a status of 408, 409, 429 or 500 and above. The same
  model request is retried after a wait.
* ``terminal``: any other failed request, including exhausted credits or
  quota even when a provider reports them with a rate-limit status. The phase
  ends with the provider's error.
* ``delivered``: the provider completed a response, usable or not. Whether the
  content is usable is the caller's business. It never turns a delivered
  response back into a transport failure.

By default a transient failure costs no model request. Charging is the
caller's job (the loops charge a usage tracker, the starter suites a slot
allowance). This module supplies the classification, the two ceilings, the
wait schedule and the per-request record with its usage status.

Every value of the policy comes from configuration. See ``RetrySettings``.
"""
from __future__ import annotations

import dataclasses
import email.utils
import json
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Mapping, Optional

import tenacity

from testuary import _words

TRANSIENT = "transient"
TERMINAL = "terminal"
DELIVERED = "delivered"
#: A request the provider refused for context overflow while a smaller output
#: token limit remains. Neither transient nor terminal: the caller adapts the limit
#: and sends again, and each refused request is recorded as sent.
CONTEXT_OVERFLOW = "context_overflow"

_TRANSIENT_STATUSES = frozenset({408, 409, 429})

# Transport exception types recognized by module and class name, so the
# classification does not import every client library it may meet.
_TRANSPORT_MODULES = frozenset({"httpx", "httpx2", "httpcore", "httpcore2"})
_TRANSPORT_NAMES = frozenset({
    "TimeoutException", "ConnectTimeout", "ReadTimeout", "WriteTimeout",
    "PoolTimeout", "ConnectError", "ReadError", "WriteError",
    "RemoteProtocolError", "NetworkError", "ProxyError",
})
# A SOCKS proxy's malformed reply, which httpcore passes on unwrapped. The
# client library wraps this and a proxy's refusal as a connection error, so
# a request sent without it is retried the same way.
_SOCKS_MODULES = frozenset({"socksio"})
_SOCKS_NAMES = frozenset({"ProtocolError", "SOCKSError"})
_CLIENT_TRANSPORT_NAMES = frozenset({"APIConnectionError", "APITimeoutError"})

# An instrument fault raised by the mutation engine is never a transport
# failure: repeating the request repeats the broken instrument.
_NEVER_TRANSIENT = frozenset({"MutationInstrumentFault"})

#: Provider error codes meaning the account cannot pay for more requests.
#: Some providers send them with a rate-limit status (OpenAI: HTTP 429 with
#: code ``insufficient_quota``). Retrying cannot succeed, so the request is
#: terminal. Add a provider's code once its response is known.
EXHAUSTED_QUOTA_CODES = frozenset({"insufficient_quota"})


def _error_code(exc: BaseException) -> Optional[str]:
    """The provider's structured error code from the client exception or its
    response body. Never derived from message wording."""
    code = getattr(exc, "code", None)
    if isinstance(code, str) and code:
        return code
    bodies = [getattr(exc, "body", None)]
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            bodies.append(response.json())
        except Exception:  # an unreadable body has no code
            pass
    for body in bodies:
        if not isinstance(body, dict):
            continue
        error = body.get("error")
        for holder in (error, body):
            if isinstance(holder, dict) and isinstance(holder.get("code"), str):
                return holder["code"]
    return None


def _status_code(exc: BaseException) -> Optional[int]:
    code = getattr(exc, "status_code", None)
    if code is None:
        response = getattr(exc, "response", None)
        code = getattr(response, "status_code", None)
    try:
        return int(code) if code is not None else None
    except (TypeError, ValueError):
        return None


class EmptyReply(Exception):
    """A provider reply that is no answer (``unanswered_reply``), such as
    an Ollama reply that ended while it loaded the model (finish reason
    "load", 0 of 0 tokens). It is no model turn: the request failed
    transiently and is sent again. ``usage`` is the usage the provider
    reported for it, which its request record keeps as known usage, or
    None when it reported none."""

    def __init__(self, message: str, usage: Optional["Usage"] = None):
        super().__init__(message)
        self.usage = usage


def unanswered_reply(*, text: str, tool_call: bool, finish_reason: Optional[str],
                     usage: Optional["Usage"]) -> bool:
    """Whether a delivered reply without text and without a tool call is
    no answer and its request is sent again: the provider ended it while
    it loaded the model (finish reason "load"), or reported 0 input and 0
    output tokens, so the model never processed the request. Any other
    empty reply, such as one that ended with "stop" after reading the
    prompt, is the model's answer and is handled as one."""
    if (text or "").strip() or tool_call:
        return False
    if finish_reason == "load":
        return True
    return usage is not None and usage.input_tokens == 0 and usage.output_tokens == 0


def empty_reply_error(finish_reason: Optional[str], usage: Optional["Usage"]) -> EmptyReply:
    """The ``EmptyReply`` of an unanswered reply, with its reported usage.
    No input means no cache-hit tokens: the cache split of a reply that read
    nothing is known to be zero."""
    if usage is not None and usage.input_tokens == 0 and usage.cache_hit_tokens is None:
        usage = Usage(0, usage.output_tokens, 0)
    return EmptyReply(
        "the provider's reply held no answer: no text, no tool call"
        + (f", finish reason {finish_reason!r}" if finish_reason else "")
        + (f", {usage.input_tokens} input and {usage.output_tokens} output tokens"
           if usage is not None else ", no reported usage"),
        usage)


def _transient_single(exc: BaseException) -> bool:
    if isinstance(exc, EmptyReply):
        return True
    name = type(exc).__name__
    if name in _NEVER_TRANSIENT:
        return False
    if isinstance(exc, (ConnectionError, TimeoutError)):
        return True
    module = (type(exc).__module__ or "").split(".", 1)[0]
    if module in _TRANSPORT_MODULES and name in _TRANSPORT_NAMES:
        return True
    if module in _SOCKS_MODULES and name in _SOCKS_NAMES:
        return True
    if module == "openai" and name in _CLIENT_TRANSPORT_NAMES:
        return True
    code = _status_code(exc)
    return code is not None and (code in _TRANSIENT_STATUSES or code >= 500)


def is_transient_failure(exc: BaseException) -> bool:
    """Whether ``exc`` is a transient transport failure.

    Decided by exception type and status code only. The wording of an error
    message is not evidence: a terminal provider error can mention a timeout.
    The cause chain is walked because client libraries raise their wrapper
    from the underlying transport error.
    """
    seen: set = set()
    current: Optional[BaseException] = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if (type(current).__name__ in _NEVER_TRANSIENT
                or _error_code(current) in EXHAUSTED_QUOTA_CODES):
            return False
        if _transient_single(current):
            return True
        current = current.__cause__ or current.__context__
    return False


def classify_failure(exc: BaseException) -> str:
    """``TRANSIENT`` or ``TERMINAL`` for a failed request."""
    return TRANSIENT if is_transient_failure(exc) else TERMINAL


# Retry hint


def _parse_retry_after_header(value: Any) -> Optional[float]:
    """``Retry-After`` per RFC 7231: integer seconds or an HTTP date. A date
    already past means no wait, not a negative one."""
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when is None:
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=timezone.utc)
    try:
        delta = (when - datetime.now(timezone.utc)).total_seconds()
    except (OverflowError, OSError, ValueError):
        return None
    return delta if delta > 0 else 0.0


def _retry_after_from_body(body: Any) -> Optional[float]:
    """Some providers put the hint into the JSON error body, possibly nested
    under ``error``."""
    if isinstance(body, (int, float)) and not isinstance(body, bool):
        return float(body)
    if isinstance(body, str):
        return _parse_retry_after_header(body)
    if not isinstance(body, dict):
        return None
    for key in ("retry_after", "retry-after", "retryAfter"):
        if key in body:
            return _retry_after_from_body(body[key])
    inner = body.get("error")
    if isinstance(inner, dict):
        return _retry_after_from_body(inner)
    return None


def retry_hint_seconds(
    exc: BaseException, *, max_seconds: float = 3600.0,
) -> Optional[float]:
    """The wait a failed response asks for, in seconds, or ``None``.

    Reads the ``Retry-After`` header of the response attached to ``exc``, then
    a retry field of the JSON error body. A negative or unreadable value is
    ``None``. A value above ``max_seconds`` is ``max_seconds``.
    """
    hint: Optional[float] = None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers is not None:
        try:
            raw = headers.get("retry-after") or headers.get("Retry-After")
        except Exception:
            raw = None
        hint = _parse_retry_after_header(raw)
    if hint is None:
        body = getattr(exc, "body", None)
        if body is None and response is not None:
            try:
                body = response.json()
            except Exception:
                try:
                    body = json.loads(response.text)
                except Exception:
                    body = None
        hint = _retry_after_from_body(body) if body is not None else None
    if hint is None or hint < 0:
        return None
    return min(float(hint), float(max_seconds))


# Settings, records and errors


@dataclasses.dataclass(frozen=True)
class RetrySettings:
    """The configured retry policy of one phase. Every value is a setting."""

    #: Consecutive transient failures of one model request before the phase
    #: ends with a transport error. Reset by a delivered response.
    consecutive_limit: int = 10
    #: Transient failures in the whole phase, shared by all its logical
    #: requests, as a bound on request cost. 0 disables the bound.
    phase_total: int = 100
    #: Wait before the second attempt without a hint. Doubles per failure.
    first_wait_s: float = 10.0
    #: Cap of the doubling wait.
    max_wait_s: float = 60.0
    #: Largest wait accepted from a provider's retry hint.
    hint_max_s: float = 3600.0
    #: Every attempt sent consumes a model request. Off by
    #: default.
    charge_every_attempt: bool = False

    def __post_init__(self) -> None:
        # A ceiling of 0 would be reached before the first request.
        if self.consecutive_limit < 1:
            raise ValueError(
                "max_consecutive_transient_failures must be at least 1, got "
                f"{self.consecutive_limit}")
        if self.phase_total < 0:
            raise ValueError(
                "max_transient_failures_per_phase must be 0 (no bound) or more, "
                f"got {self.phase_total}")

    @classmethod
    def from_config(cls, cfg: Mapping[str, Any]) -> "RetrySettings":
        return cls(
            consecutive_limit=int(cfg["max_consecutive_transient_failures"]),
            phase_total=int(cfg["max_transient_failures_per_phase"]),
            first_wait_s=float(cfg["transient_first_wait_seconds"]),
            max_wait_s=float(cfg["max_transient_wait_seconds"]),
            hint_max_s=float(cfg["max_transient_hint_seconds"]),
            charge_every_attempt=bool(cfg["count_retries"]),
        )

    def backoff(self) -> tenacity.wait.wait_base:
        """The wait after a failure without a provider's hint: the first wait,
        doubled per failure in a row, up to the cap."""
        return tenacity.wait_exponential(multiplier=self.first_wait_s, max=self.max_wait_s)


@dataclasses.dataclass
class RequestRecord:
    """One request sent."""

    logical_index: int
    attempt: int
    outcome: str
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    wait_s: float = 0.0
    usage_known: bool = False
    input_tokens: int = 0
    output_tokens: int = 0
    #: None when the provider did not report the cache split of the input.
    cache_hit_tokens: Optional[int] = None

    def as_dict(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {
            "logical_request_idx": self.logical_index,
            "attempt": self.attempt,
            "outcome": self.outcome,
            "usage_known": self.usage_known,
        }
        if self.error_type:
            out["error_type"] = self.error_type
            out["error_message"] = (self.error_message or "")[:500]
        if self.wait_s:
            out["wait_s"] = round(self.wait_s, 3)
        if self.usage_known:
            out["input_tokens"] = self.input_tokens
            out["output_tokens"] = self.output_tokens
            # The flag tells an unreported cache split from a reported count. A
            # record without it may carry a default zero.
            out["cache_usage_known"] = self.cache_hit_tokens is not None
            if self.cache_hit_tokens is not None:
                out["cache_hit_tokens"] = self.cache_hit_tokens
        return out


@dataclasses.dataclass
class Usage:
    """Token usage of a delivered response, when the provider reported it.

    ``cache_hit_tokens`` is the cached part of the input, or None when the
    provider did not report the split. An unreported split is not zero hits.
    """

    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: Optional[int] = None


def reported_usage(input_tokens: Any, output_tokens: Any,
                   cache_hit_tokens: Any = None) -> Optional[Usage]:
    """Usage from provider-reported counts, or None when either token count is
    absent or not a non-negative integer. An absent cache count, or one outside
    ``0..input_tokens``, leaves the cache split unknown."""

    def count(value: Any) -> Optional[int]:
        return value if type(value) is int and value >= 0 else None

    input_count, output_count = count(input_tokens), count(output_tokens)
    if input_count is None or output_count is None:
        return None
    cache = count(cache_hit_tokens)
    if cache is not None and cache > input_count:
        cache = None
    return Usage(input_count, output_count, cache)


@dataclasses.dataclass
class LogicalOutcome:
    """What one model request came to."""

    value: Any
    attempts: int
    unknown_usage_requests: int
    usage: Optional[Usage]


def failure_text(value: str | BaseException) -> str:
    """Diagnostic text of a failure, with captured credentials and URL user
    and query data redacted. The original failure and its classification are
    unchanged. LangChain's words for a response that streamed nothing get the
    reason a user can act on."""
    from testuary import config
    from testuary.settings import _redact_diagnostic

    text = _redact_diagnostic(str(value), config.active_settings())
    if "No generation chunks were returned" in text:
        text += (" (the model service sent no streamed answer: Testuary streams its "
                 "requests, so the service must support streaming)")
    return text


class TerminalRequestError(Exception):
    """A request failed in a way that retrying cannot fix. The provider's
    error is the cause."""

    def __init__(self, cause: BaseException):
        super().__init__(f"{type(cause).__name__}: {failure_text(cause)}")
        self.__cause__ = cause


class ContextOverflow(Exception):
    """The provider refused a request because the prompt does not fit the
    model's context. It is not retried, since the caller owns its prompt and
    its output limit. Its message names the provider's error type and text
    without the model's key, and it carries no cause, since a provider's
    error can hold the request and its key."""

    def __init__(self, cause: BaseException):
        super().__init__(f"{type(cause).__name__}: {failure_text(cause)}")
        self.__cause__ = cause


class TransportExhausted(Exception):
    """A retry ceiling was reached. The last transport failure is the cause.
    ``attempt`` is the attempt of the model request that reached it."""

    def __init__(self, reason: str, last: Optional[BaseException],
                 attempt: Optional[int] = None):
        detail = f"{reason}"
        if last is not None:
            detail += f"; last failure {type(last).__name__}: {failure_text(last)}"
        super().__init__(detail)
        self.reason = reason
        self.attempt = attempt
        self.__cause__ = last


def service_address(api_url: Optional[str]) -> str:
    """A model service's address for a display: without credentials and
    without a query, as a run's recorded model settings give it, or "" when
    unknown."""
    if not api_url:
        return ""
    from testuary.settings import _safe

    return str(_safe({"api_url": api_url}).get("api_url") or "")


def exhausted_text(exc: TransportExhausted, service: str = "") -> str:
    """A display's sentence for a request given up after its retries: what
    the service did, after how many attempts, and what to check. ``service``
    is its address without credentials (``service_address``). The
    exception's own text, which may hold an address with credentials, is
    never shown, and no key is."""
    chain, current = [], exc.__cause__
    while current is not None and current not in chain:
        chain.append(current)
        current = current.__cause__ or current.__context__
    status = next((code for code in map(_status_code, chain) if code), None)
    if any(isinstance(item, TimeoutError) or "Timeout" in type(item).__name__ for item in chain):
        what, check = "did not answer in time", "that it is running and not overloaded"
    elif status:
        what, check = f"answered with an error (HTTP {status})", "its status and the model's settings"
    else:
        what, check = "could not be reached", "that it is running and that the model's address is right"
    count = exc.attempt or 0
    attempts = f" after {count} attempt{'s' if count != 1 else ''}" if count else ""
    at = f" at {service}" if service else ""
    return f"{_words.SERVICE_FAILURE_START}{at} {what}{attempts}: check {check}, then continue the run."


class RequestCancelled(Exception):
    """The run was cancelled while waiting to retry."""


class RecordNotWritten(Exception):
    """A response was delivered but its record could not be written. Carries
    the delivered outcome so that the caller can keep the paid-for response
    before the phase stops. The write failure is the cause."""

    def __init__(self, outcome: "LogicalOutcome", cause: BaseException):
        super().__init__(
            "the record of a delivered response could not be written: "
            f"{type(cause).__name__}: {failure_text(cause)}")
        self.outcome = outcome
        self.__cause__ = cause


def failure_traceback(exc: BaseException) -> str:
    """The traceback text of ``exc`` with its chain, as Python prints it. A
    request error of this module ends the chain wherever it sits in it,
    since its message is already redacted while the raw provider error
    behind it may not be."""
    import traceback

    shown = traceback.TracebackException(type(exc), exc, exc.__traceback__, compact=True)
    link, error = shown, exc
    # Each step follows the link Python prints next: the cause, else the context.
    while link is not None and not isinstance(
            error, (TerminalRequestError, TransportExhausted, RecordNotWritten)):
        if link.__cause__ is not None:
            link, error = link.__cause__, error.__cause__
        elif not link.__suppress_context__:
            link, error = link.__context__, error.__context__
        else:
            link = None
    if link is not None:
        link.__cause__ = link.__context__ = None
    return "".join(shown.format())


# The per-phase account


def request_failure_kind(exc: Optional[BaseException]) -> Optional[str]:
    """``infrastructure`` when ``exc`` (or its cause chain) is a terminal model
    request failure or the ceiling on requests sent again. Otherwise None (unjudged)."""
    from testuary.phase_state import INFRASTRUCTURE_FAILURE

    seen: set = set()
    current = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if isinstance(current, (TerminalRequestError, TransportExhausted)):
            return INFRASTRUCTURE_FAILURE
        current = current.__cause__ or current.__context__
    return None


#: The longest single blocking wait of a request account or of the starter
#: suites' wait on their parallel generations, so an interruption is received
#: within this time on every platform.
WAIT_SLICE_SECONDS = 0.5


class RequestAccount:
    """Retry state and request records of one phase.

    Shared by every model request of the phase, from any thread. Counters
    are protected by a lock. The cancellation event ends a wait promptly.
    """

    def __init__(
        self,
        settings: RetrySettings,
        *,
        wait_fn: Optional[Callable[[float], None]] = None,
        cancel_event: Optional[threading.Event] = None,
        on_record: Optional[Callable[[RequestRecord], None]] = None,
        service: str = "",
    ) -> None:
        self.settings = settings
        #: The model service's address without credentials, for a display.
        self.service = service
        self.cancel = cancel_event if cancel_event is not None else threading.Event()
        self._wait_fn = wait_fn
        self._on_record = on_record
        self._lock = threading.Lock()
        self.physical_requests = 0
        self.logical_requests = 0
        self.transient_failures = 0
        self.unknown_usage_requests = 0
        self.records: List[RequestRecord] = []

    # Accounting

    def _record(self, record: RequestRecord) -> None:
        if record.error_message:
            record.error_message = failure_text(record.error_message)
        with self._lock:
            self.physical_requests += 1
            if not record.usage_known:
                self.unknown_usage_requests += 1
            self.records.append(record)
        if self._on_record is not None:
            # The consumer keeps the phase's request journal, from which a
            # resumed run recovers what was spent: a failure to record stops
            # the phase instead of letting the accounting drift.
            self._on_record(record)

    def _count_transient(self) -> int:
        with self._lock:
            self.transient_failures += 1
            return self.transient_failures

    def _next_logical_index(self) -> int:
        with self._lock:
            self.logical_requests += 1
            return self.logical_requests

    def wait(self, seconds: float) -> None:
        """Wait, or return at once when the run is cancelled.

        The wait runs in short slices: on Windows a thread blocked in one long
        timed wait does not receive Ctrl+C until it returns, and a Retry-After
        hint can ask for an hour.
        """
        if seconds <= 0:
            return
        if self._wait_fn is not None:
            self._wait_fn(seconds)
            return
        deadline = time.monotonic() + seconds
        while not self.cancel.is_set():
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                return
            self.cancel.wait(min(remaining, WAIT_SLICE_SECONDS))

    # One model request

    def execute(
        self,
        send: Callable[[int], Any],
        *,
        usage_of: Optional[Callable[[Any], Optional[Usage]]] = None,
        before_attempt: Optional[Callable[[int], None]] = None,
        on_retry: Optional[Callable[[int, BaseException, float], None]] = None,
        logical_index: Optional[int] = None,
        terminal: Optional[Callable[[BaseException], bool]] = None,
    ) -> LogicalOutcome:
        """Obtain the response for one model request.

        ``send(attempt)`` sends one request and returns the
        delivered value or raises. ``usage_of(value)`` returns the reported
        usage or ``None`` when the provider did not report it.
        ``before_attempt(attempt)`` runs before each attempt is sent, outside
        the classification, so a limit check raising there propagates as it
        is. ``on_retry(attempt, exc, wait)`` runs after a transient failure and
        before the wait. Attempts count from 1. ``terminal(exc)``
        marks a failure terminal whatever its status, such as a request too
        large for the model's context, which a resend cannot fix.

        Raises ``TerminalRequestError`` on a terminal failure,
        ``TransportExhausted`` when a ceiling is reached,
        ``RequestCancelled`` when the run is cancelled before an attempt or
        during a wait, and ``RecordNotWritten`` when the record of a delivered
        response cannot be written.
        """
        index = (
            logical_index if logical_index is not None
            else self._next_logical_index()
        )
        settings = self.settings
        # Every failure of a request that tenacity sends again is transient,
        # so the attempt number is also the count of failures in a row.
        attempt = 0
        unknown = 0
        last_wait = 0.0
        planned = 0.0
        # The transient failure being retried, and the phase's transient
        # failures counted with it.
        failure: Optional[BaseException] = None
        total = 0

        def begin(retry_state: tenacity.RetryCallState) -> None:
            nonlocal attempt
            if self.cancel.is_set():
                raise RequestCancelled("the phase's requests were cancelled")
            attempt = retry_state.attempt_number
            if before_attempt is not None:
                before_attempt(attempt)

        def send_once() -> Any:
            nonlocal unknown, failure, total
            try:
                return send(attempt)
            except Exception as raised:
                exc = raised
            # Classified, recorded and raised after the handler, so that a
            # record that cannot be written does not carry the provider's
            # error, which can hold the request and its key, as its context.
            if classify_failure(exc) == TERMINAL or (
                    terminal is not None and terminal(exc)):
                self._record(RequestRecord(
                    index, attempt, TERMINAL, type(exc).__name__,
                    str(exc), last_wait,
                ))
                # A terminal failure ends the phase at once: the other
                # model requests of the phase stop before their next
                # attempt.
                self.cancel.set()
                raise TerminalRequestError(exc) from exc
            failed = RequestRecord(
                index, attempt, TRANSIENT, type(exc).__name__, str(exc),
                last_wait,
            )
            # An unanswered reply's reported usage is known, such as
            # an Ollama load reply's 0 of 0 tokens.
            reported = exc.usage if isinstance(exc, EmptyReply) else None
            if reported is not None:
                failed.usage_known = True
                failed.input_tokens = int(reported.input_tokens)
                failed.output_tokens = int(reported.output_tokens)
                failed.cache_hit_tokens = (None if reported.cache_hit_tokens is None
                                           else int(reported.cache_hit_tokens))
            else:
                unknown += 1
            # A record that cannot be written is no transient failure: it
            # propagates as it is.
            self._record(failed)
            total = self._count_transient()
            failure = exc
            raise exc

        def ceiling(retry_state: tenacity.RetryCallState) -> bool:
            return (retry_state.attempt_number >= settings.consecutive_limit
                    or bool(settings.phase_total) and total >= settings.phase_total)

        def exhausted(retry_state: tenacity.RetryCallState) -> None:
            # A ceiling ends the phase: every other model request of the
            # phase stops before its next attempt.
            self.cancel.set()
            if retry_state.attempt_number >= settings.consecutive_limit:
                reason = (f"{retry_state.attempt_number} consecutive transient failures "
                          f"of one request (limit {settings.consecutive_limit})")
            else:
                reason = (f"{total} transient failures in this phase "
                          f"(limit {settings.phase_total})")
            raise TransportExhausted(reason, failure, attempt) from failure

        def delay(retry_state: tenacity.RetryCallState) -> float:
            nonlocal planned
            hint = retry_hint_seconds(failure, max_seconds=settings.hint_max_s)
            planned = hint if hint is not None else settings.backoff()(retry_state)
            return planned

        def before_sleep(retry_state: tenacity.RetryCallState) -> None:
            if on_retry is not None:
                on_retry(attempt, failure, planned)
            if self.cancel.is_set():
                raise RequestCancelled("cancelled before retry") from failure

        def sleep(seconds: float) -> None:
            # Sleeps the wait delay() planned, as the record and the retry
            # callback name it.
            nonlocal last_wait
            self.wait(planned)
            if self.cancel.is_set():
                raise RequestCancelled("cancelled while waiting to retry") from failure
            last_wait = planned

        value = tenacity.Retrying(
            retry=tenacity.retry_if_exception(lambda exc: exc is failure),
            stop=ceiling, wait=delay, sleep=sleep, before=begin,
            before_sleep=before_sleep, retry_error_callback=exhausted,
        )(send_once)
        usage = usage_of(value) if usage_of is not None else None
        record = RequestRecord(index, attempt, DELIVERED, wait_s=last_wait)
        if usage is not None:
            record.usage_known = True
            record.input_tokens = int(usage.input_tokens)
            record.output_tokens = int(usage.output_tokens)
            record.cache_hit_tokens = (None if usage.cache_hit_tokens is None
                                       else int(usage.cache_hit_tokens))
        else:
            unknown += 1
        outcome = LogicalOutcome(value, attempt, unknown, usage)
        try:
            self._record(record)
        except Exception as exc:
            raise RecordNotWritten(outcome, exc) from exc
        return outcome

    def next_logical_index(self) -> int:
        """Reserve the index of the next model request. ``execute`` does
        this itself unless the caller passes ``logical_index``."""
        return self._next_logical_index()

    def note_extra_request(
        self, logical_index: int, attempt: int, outcome: str,
        exc: Optional[BaseException] = None,
    ) -> None:
        """Record a request the caller sent itself within attempt
        ``attempt`` of a model request, such as one refused for context
        overflow before the caller sent a smaller one."""
        self._record(RequestRecord(
            logical_index, attempt, outcome,
            type(exc).__name__ if exc is not None else None,
            str(exc) if exc is not None else None,
        ))
