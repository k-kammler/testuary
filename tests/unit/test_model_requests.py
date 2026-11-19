"""The shared model-request policy: classification, ceilings, waits and
records, in the cases that concern the policy itself. The phase callers have
their own tests."""
from __future__ import annotations

import threading
from datetime import datetime, timedelta, timezone
from email.utils import format_datetime
from types import SimpleNamespace

import socket

import httpx
import openai
import pytest
import socksio

from testuary import model_requests as mr


# Classification


class _Status(Exception):
    def __init__(self, code, message="provider error", headers=None, body=None):
        super().__init__(message)
        self.status_code = code
        self.response = SimpleNamespace(
            status_code=code, headers=headers or {},
            json=lambda: body if body is not None else (_ for _ in ()).throw(ValueError()),
            text="",
        )
        self.body = body


class MutationInstrumentFault(Exception):
    """Same name as the engine's fault, matched by name."""


@pytest.mark.parametrize("exc", [
    httpx.ConnectError("refused"),
    httpx.ReadTimeout("read"),
    httpx.ConnectTimeout("connect"),
    httpx.RemoteProtocolError("closed"),
    httpx.ReadError("reset"),
    ConnectionResetError("reset by peer"),
    TimeoutError("socket"),
    openai.APIConnectionError(request=httpx.Request("POST", "http://x")),
    openai.APITimeoutError(request=httpx.Request("POST", "http://x")),
    httpx.ProxyError("Proxy Server could not connect: CONNECTION_REFUSED."),
    socksio.ProtocolError("Malformed reply"),
    _Status(408), _Status(409), _Status(429), _Status(500), _Status(502),
    _Status(503), _Status(529),
])
def test_transport_types_and_statuses_are_transient(exc):
    assert mr.classify_failure(exc) == mr.TRANSIENT


@pytest.mark.parametrize("exc", [
    _Status(400, "bad request"),
    _Status(401, "invalid api key"),
    _Status(403, "forbidden"),
    _Status(404, "model not found"),
    _Status(422, "unprocessable"),
    ValueError("unsupported role"),
    RuntimeError("empty stream"),
])
def test_other_failures_are_terminal(exc):
    assert mr.classify_failure(exc) == mr.TERMINAL


def test_the_word_timeout_in_a_message_is_not_evidence():
    """A message is not retried because it contains the word. A 400
    whose text mentions a timeout is terminal."""
    assert mr.classify_failure(_Status(400, "timeout parameter invalid")) == mr.TERMINAL
    assert mr.classify_failure(RuntimeError("operation timed out")) == mr.TERMINAL


def test_instrument_fault_is_never_transient_even_when_wrapped():
    fault = MutationInstrumentFault("worker timed out")
    assert mr.classify_failure(fault) == mr.TERMINAL
    wrapped = RuntimeError("phase failed")
    wrapped.__cause__ = fault
    assert mr.classify_failure(wrapped) == mr.TERMINAL


def test_cause_chain_is_walked():
    wrapper = RuntimeError("Connection error.")
    wrapper.__cause__ = httpx.ConnectError("refused")
    assert mr.classify_failure(wrapper) == mr.TRANSIENT


# Retry hint


def test_hint_from_header_seconds():
    assert mr.retry_hint_seconds(_Status(429, headers={"retry-after": "30"})) == 30.0
    assert mr.retry_hint_seconds(_Status(429, headers={"Retry-After": "7"})) == 7.0


def test_hint_from_http_date():
    when = datetime.now(timezone.utc) + timedelta(seconds=90)
    hint = mr.retry_hint_seconds(
        _Status(503, headers={"retry-after": format_datetime(when)}))
    assert 85.0 <= hint <= 90.0


def test_hint_from_a_past_date_is_no_wait():
    when = datetime.now(timezone.utc) - timedelta(seconds=90)
    assert mr.retry_hint_seconds(
        _Status(503, headers={"retry-after": format_datetime(when)})) == 0.0


def test_hint_from_body_and_nested_body():
    assert mr.retry_hint_seconds(_Status(429, body={"retry_after": 12})) == 12.0
    assert mr.retry_hint_seconds(
        _Status(429, body={"error": {"retryAfter": "4.5"}})) == 4.5


def test_hint_absent_negative_or_malformed_is_none():
    assert mr.retry_hint_seconds(_Status(429)) is None
    assert mr.retry_hint_seconds(_Status(429, headers={"retry-after": "-5"})) is None
    assert mr.retry_hint_seconds(_Status(429, headers={"retry-after": "soon"})) is None
    assert mr.retry_hint_seconds(httpx.ConnectError("refused")) is None


def test_hint_is_bounded_by_the_configured_maximum():
    exc = _Status(429, headers={"retry-after": "7200"})
    assert mr.retry_hint_seconds(exc) == 3600.0
    assert mr.retry_hint_seconds(exc, max_seconds=120) == 120.0


# Settings


def _settings(**over):
    base = dict(consecutive_limit=10, phase_total=100, first_wait_s=10.0,
                max_wait_s=60.0, hint_max_s=3600.0, charge_every_attempt=False)
    base.update(over)
    return mr.RetrySettings(**base)


def test_settings_come_from_configuration_keys():
    settings = mr.RetrySettings.from_config({
        "max_consecutive_transient_failures": 3,
        "max_transient_failures_per_phase": 50,
        "transient_first_wait_seconds": 1.5,
        "max_transient_wait_seconds": 9,
        "max_transient_hint_seconds": 120,
        "count_retries": True,
    })
    assert settings == mr.RetrySettings(3, 50, 1.5, 9.0, 120.0, True)


def test_backoff_doubles_from_the_first_wait_to_the_cap():
    backoff = _settings().backoff()
    assert [backoff(SimpleNamespace(attempt_number=n)) for n in range(1, 6)] == [10, 20, 40, 60, 60]


# Executing one model request


def _account(**over):
    waits = []
    account = mr.RequestAccount(_settings(**over), wait_fn=waits.append)
    return account, waits


def _script(*outcomes):
    """``send`` that raises the scripted exceptions in order, then returns the
    remaining values."""
    calls = []

    def send(attempt):
        calls.append(attempt)
        item = outcomes[len(calls) - 1]
        if isinstance(item, BaseException):
            raise item
        return item

    send.calls = calls
    return send


def test_delivered_first_time():
    account, waits = _account()

    outcome = account.execute(
        _script("ok"), usage_of=lambda v: mr.Usage(250, 40, 10))

    assert outcome.value == "ok"
    assert outcome.attempts == 1
    assert outcome.unknown_usage_requests == 0
    assert waits == []
    assert (account.physical_requests, account.logical_requests) == (1, 1)
    assert account.unknown_usage_requests == 0
    [record] = account.records
    assert record.as_dict() == {
        "logical_request_idx": 1, "attempt": 1, "outcome": "delivered",
        "usage_known": True, "input_tokens": 250, "output_tokens": 40,
        "cache_hit_tokens": 10, "cache_usage_known": True,
    }


def test_transient_failures_are_retried_with_doubling_waits():
    account, waits = _account()
    send = _script(httpx.ConnectError("a"), httpx.ConnectError("b"),
                   httpx.ConnectError("c"), "ok")

    outcome = account.execute(send, usage_of=lambda v: mr.Usage(1, 1))

    assert outcome.value == "ok"
    assert outcome.attempts == 4
    assert outcome.unknown_usage_requests == 3
    assert waits == [10, 20, 40]
    assert account.physical_requests == 4
    assert account.logical_requests == 1
    assert account.transient_failures == 3
    assert account.unknown_usage_requests == 3
    assert [r.outcome for r in account.records] == [
        "transient", "transient", "transient", "delivered"]
    assert [r.wait_s for r in account.records] == [0, 10, 20, 40]
    assert account.records[0].error_type == "ConnectError"


def test_a_failure_marked_terminal_is_not_retried_whatever_its_status():
    # A request too large for the context, reported with a transient status:
    # a resend cannot fix it, so the caller marks it terminal.
    account, waits = _account()
    send = _script(_Status(503), "ok")

    with pytest.raises(mr.TerminalRequestError):
        account.execute(send, terminal=lambda exc: True)

    assert send.calls == [1]
    assert waits == []


def test_server_hint_wins_over_backoff():
    account, waits = _account()
    send = _script(_Status(429, headers={"retry-after": "30"}), "ok")

    account.execute(send)

    assert waits == [30.0]


def test_consecutive_ceiling_ends_the_request_without_a_final_wait():
    account, waits = _account(consecutive_limit=3)
    send = _script(*[httpx.ConnectError(f"{i}") for i in range(5)])

    with pytest.raises(mr.TransportExhausted) as info:
        account.execute(send)

    assert "3 consecutive" in str(info.value)
    assert "ConnectError: 2" in str(info.value)
    assert isinstance(info.value.__cause__, httpx.ConnectError)
    assert send.calls == [1, 2, 3]
    assert waits == [10, 20]


def test_a_delivered_response_resets_the_consecutive_count():
    account, _waits = _account(consecutive_limit=3)

    account.execute(_script(httpx.ConnectError("1"), httpx.ConnectError("2"), "ok"))
    account.execute(_script(httpx.ConnectError("3"), httpx.ConnectError("4"), "ok"))

    assert account.transient_failures == 4
    assert account.logical_requests == 2


def test_phase_total_bounds_intermittent_failures():
    account, _waits = _account(consecutive_limit=10, phase_total=5)

    account.execute(_script(httpx.ConnectError("1"), httpx.ConnectError("2"), "ok"))
    account.execute(_script(httpx.ConnectError("3"), httpx.ConnectError("4"), "ok"))
    with pytest.raises(mr.TransportExhausted, match=r"5 transient failures in this phase"):
        account.execute(_script(httpx.ConnectError("5"), "ok"))


def test_phase_total_of_zero_means_unbounded():
    account, _waits = _account(consecutive_limit=2, phase_total=0)
    for _ in range(50):
        account.execute(_script(httpx.ConnectError("x"), "ok"))
    assert account.transient_failures == 50


def test_terminal_failure_raises_with_the_provider_error_and_no_wait():
    account, waits = _account()
    send = _script(_Status(401, "invalid api key"))

    with pytest.raises(mr.TerminalRequestError, match="invalid api key") as info:
        account.execute(send)

    assert isinstance(info.value.__cause__, _Status)
    assert send.calls == [1]
    assert waits == []
    [record] = account.records
    assert (record.outcome, record.error_type) == ("terminal", "_Status")


def test_terminal_after_transient_keeps_the_transient_records():
    account, waits = _account()
    send = _script(httpx.ConnectError("x"), _Status(401))

    with pytest.raises(mr.TerminalRequestError):
        account.execute(send)

    assert [r.outcome for r in account.records] == ["transient", "terminal"]
    assert waits == [10]


def test_delivered_without_reported_usage_is_unknown_not_zero():
    account, _waits = _account()

    outcome = account.execute(_script("ok"), usage_of=lambda v: None)

    assert outcome.usage is None
    assert outcome.unknown_usage_requests == 1
    assert account.unknown_usage_requests == 1
    [record] = account.records
    assert record.usage_known is False
    assert "input_tokens" not in record.as_dict()


def test_before_attempt_runs_outside_classification():
    """A limit check raising before the request propagates as it is."""
    class LimitCheck(Exception):
        pass

    account, _waits = _account()
    seen = []

    def before(attempt):
        seen.append(attempt)
        if attempt == 2:
            raise LimitCheck("limit")

    with pytest.raises(LimitCheck):
        account.execute(_script(httpx.ConnectError("x"), "ok"), before_attempt=before)
    assert seen == [1, 2]
    assert account.records[-1].outcome == "transient"


def test_on_retry_sees_attempt_error_and_wait():
    account, _waits = _account()
    seen = []
    send = _script(_Status(503, headers={"retry-after": "5"}), "ok")

    account.execute(send, on_retry=lambda a, e, w: seen.append((a, type(e).__name__, w)))

    assert seen == [(1, "_Status", 5.0)]


def test_the_retry_schedule_is_pinned():
    """The retry control as the request journal and a replay see it, for a
    mix of failures: each attempt's record with the wait before it, the
    waits slept, what the retry callback and the limit check saw, and the
    attempts and unknown usage counted."""
    account, waits = _account()
    before, retries = [], []
    send = _script(
        httpx.ConnectError("refused"),
        _Status(429, "slow down", headers={"retry-after": "7"}),
        mr.EmptyReply("loading", usage=mr.Usage(0, 0, 0)),
        _Status(503, "unavailable"),
        openai.APITimeoutError(request=httpx.Request("POST", "http://x")),
        "ok")

    outcome = account.execute(
        send, usage_of=lambda v: mr.Usage(5, 2, 1), before_attempt=before.append,
        on_retry=lambda a, e, w: retries.append((a, type(e).__name__, w)))

    assert (outcome.value, outcome.attempts, outcome.unknown_usage_requests) == ("ok", 6, 4)
    assert before == send.calls == [1, 2, 3, 4, 5, 6]
    assert waits == [10, 7.0, 40, 60, 60]
    assert retries == [(1, "ConnectError", 10), (2, "_Status", 7.0), (3, "EmptyReply", 40),
                       (4, "_Status", 60), (5, "APITimeoutError", 60)]
    assert [record.as_dict() for record in account.records] == [
        {"logical_request_idx": 1, "attempt": 1, "outcome": "transient", "usage_known": False,
         "error_type": "ConnectError", "error_message": "refused"},
        {"logical_request_idx": 1, "attempt": 2, "outcome": "transient", "usage_known": False,
         "error_type": "_Status", "error_message": "slow down", "wait_s": 10},
        {"logical_request_idx": 1, "attempt": 3, "outcome": "transient", "usage_known": True,
         "error_type": "EmptyReply", "error_message": "loading", "wait_s": 7.0,
         "input_tokens": 0, "output_tokens": 0, "cache_usage_known": True, "cache_hit_tokens": 0},
        {"logical_request_idx": 1, "attempt": 4, "outcome": "transient", "usage_known": False,
         "error_type": "_Status", "error_message": "unavailable", "wait_s": 40},
        {"logical_request_idx": 1, "attempt": 5, "outcome": "transient", "usage_known": False,
         "error_type": "APITimeoutError", "error_message": "Request timed out.", "wait_s": 60},
        {"logical_request_idx": 1, "attempt": 6, "outcome": "delivered", "usage_known": True,
         "wait_s": 60, "input_tokens": 5, "output_tokens": 2, "cache_usage_known": True,
         "cache_hit_tokens": 1}]
    assert (account.physical_requests, account.transient_failures, account.unknown_usage_requests) == (
        6, 5, 4)


@pytest.mark.parametrize("exc, retried", [
    (httpx.ReadError("reset"), True), (ConnectionResetError("reset"), True),
    (_Status(408), True), (_Status(409), True), (_Status(500), True),
    (_Status(400, "bad request"), False), (_Status(404, "no model"), False),
    (ValueError("unsupported role"), False),
])
def test_which_failures_are_sent_again(exc, retried):
    """A transient failure is sent again after its wait. Any other failure
    ends the request at once with the provider's error."""
    account, waits = _account()
    send = _script(exc, "ok")

    if retried:
        assert account.execute(send).attempts == 2
        assert (send.calls, waits) == ([1, 2], [10])
    else:
        with pytest.raises(mr.TerminalRequestError) as info:
            account.execute(send)
        assert info.value.__cause__ is exc
        assert (send.calls, waits, account.cancel.is_set()) == ([1], [], True)


def test_cancellation_ends_the_wait_and_the_request():
    cancel = threading.Event()
    account = mr.RequestAccount(_settings(), cancel_event=cancel)
    send = _script(httpx.ConnectError("x"), "ok")
    account.wait = lambda seconds: cancel.set()  # the wait is interrupted

    with pytest.raises(mr.RequestCancelled):
        account.execute(send)

    assert send.calls == [1]


def test_real_wait_returns_promptly_when_cancelled():
    cancel = threading.Event()
    account = mr.RequestAccount(_settings(), cancel_event=cancel)
    threading.Timer(0.05, cancel.set).start()
    started = datetime.now()
    account.wait(30)
    assert (datetime.now() - started).total_seconds() < 5


def test_extra_requests_of_one_attempt_are_recorded():
    account, _waits = _account()
    index = account.next_logical_index()

    def send(attempt):
        account.note_extra_request(index, attempt, mr.CONTEXT_OVERFLOW,
                                   RuntimeError("context length is only 8000"))
        account.note_extra_request(index, attempt, mr.CONTEXT_OVERFLOW,
                                   RuntimeError("context length is only 8000"))
        return "ok"

    account.execute(send, usage_of=lambda v: mr.Usage(1, 1), logical_index=index)

    assert account.physical_requests == 3
    assert account.logical_requests == 1
    assert [r.outcome for r in account.records] == [
        "context_overflow", "context_overflow", "delivered"]
    assert all(r.logical_index == index for r in account.records)


def test_phase_total_is_shared_across_threads_and_ends_every_request():
    """Reaching the total in one thread ends the phase: the thread that hit
    it gets the transport error, every other thread stops before its next
    attempt. Requests already in flight when the total is reached may still
    fail, so the count can pass the total by at most one per thread."""
    account = mr.RequestAccount(_settings(consecutive_limit=100, phase_total=40),
                                wait_fn=lambda s: None)
    ends = []

    def worker():
        # More failures per thread than the total, so no thread can finish
        # before the total is reached.
        try:
            for _ in range(100):
                account.execute(_script(httpx.ConnectError("x"), "ok"))
        except mr.TransportExhausted:
            ends.append("exhausted")
        except mr.RequestCancelled:
            ends.append("cancelled")
        else:
            ends.append("finished")

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert 40 <= account.transient_failures <= 40 + 7
    assert account.cancel.is_set()
    assert ends.count("exhausted") >= 1
    assert "finished" not in ends


def test_a_terminal_failure_cancels_the_other_requests_of_the_phase():
    account, _waits = _account()
    with pytest.raises(mr.TerminalRequestError):
        account.execute(_script(_Status(401)))

    with pytest.raises(mr.RequestCancelled):
        account.execute(_script("ok"))


def test_a_ceiling_cancels_the_other_requests_of_the_phase():
    account, _waits = _account(consecutive_limit=2)
    with pytest.raises(mr.TransportExhausted):
        account.execute(_script(httpx.ConnectError("1"), httpx.ConnectError("2")))

    with pytest.raises(mr.RequestCancelled):
        account.execute(_script("ok"))


# Limits and the request journal


def test_a_new_account_counts_only_its_own_failures():
    """Each session's account starts at zero: a ceiling an earlier session
    reached does not refuse the next one."""
    first, _ = _account(consecutive_limit=2)
    with pytest.raises(mr.TransportExhausted):
        first.execute(_script(httpx.ConnectError("1"), httpx.ConnectError("2")))
    second, waits = _account(consecutive_limit=2)
    outcome = second.execute(_script(httpx.ConnectError("3"), "ok"))
    assert outcome.value == "ok" and second.transient_failures == 1 and waits == [10]


@pytest.mark.parametrize("over", [
    {"consecutive_limit": 0}, {"phase_total": -1},
])
def test_impossible_limits_are_refused(over):
    with pytest.raises(ValueError):
        _settings(**over)


def test_a_record_that_cannot_be_written_stops_with_the_delivered_response():
    """A resumed run recovers its accounting from the journal, so a journal
    record that fails is an error, not a diagnostic. The delivered response
    comes back with the error, so that the caller can keep it."""
    def failing(record):
        raise OSError("disk full")

    account = mr.RequestAccount(_settings(), on_record=failing)

    with pytest.raises(mr.RecordNotWritten) as caught:
        account.execute(_script("ok"))

    assert caught.value.outcome.value == "ok"
    assert isinstance(caught.value.__cause__, OSError)


def test_a_failure_record_that_cannot_be_written_stops_the_request():
    written = []

    def failing_on_transient(record):
        if record.outcome == mr.TRANSIENT:
            raise OSError("disk full")
        written.append(record)

    account = mr.RequestAccount(
        _settings(), on_record=failing_on_transient, wait_fn=lambda _s: None)
    send = _script(httpx.ConnectError("refused"), "ok")

    with pytest.raises(OSError, match="disk full"):
        account.execute(send)

    assert send.calls == [1]
    assert written == []


@pytest.mark.parametrize("provider_error", [
    _Status(400, "bad request, key sk-run-sentinel"),
    httpx.ConnectError("refused, key sk-run-sentinel"),
])
def test_a_failure_record_that_cannot_be_written_carries_no_provider_error(provider_error):
    """A provider's error can hold the request and its key. The write error
    of its record is raised without it, so a crash log's traceback shows
    none of it."""
    def failing(record):
        raise OSError("disk full")

    account = mr.RequestAccount(_settings(), on_record=failing, wait_fn=lambda _s: None)

    with pytest.raises(OSError, match="disk full") as caught:
        account.execute(_script(provider_error))

    assert caught.value.__cause__ is None and caught.value.__context__ is None
    assert "sk-run-sentinel" not in mr.failure_traceback(caught.value)



class _FakeSocksProxy:
    """A local SOCKS5 proxy that fails each connection in turn: it refuses
    the CONNECT (httpx raises ProxyError), or it answers the greeting with a
    reply that is no SOCKS5 (socksio raises ProtocolError)."""

    def __init__(self, modes):
        self.modes = list(modes)
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(len(self.modes))
        self.port = self.server.getsockname()[1]
        self.thread = threading.Thread(target=self._serve, daemon=True)
        self.thread.start()

    def _serve(self):
        for mode in self.modes:
            connection, _ = self.server.accept()
            with connection:
                connection.recv(16)                          # greeting
                if mode == "refuse":
                    connection.sendall(b"\x05\x00")           # no authentication
                    connection.recv(512)                     # CONNECT request
                    connection.sendall(b"\x05\x05\x00\x01" + b"\x00" * 6)  # refused
                else:
                    connection.sendall(b"HTTP/1.1 400 Bad Request\r\n\r\n")
                # End the reply gracefully, then wait briefly for the client:
                # closing at once could reset the connection before the reply
                # is read.
                connection.shutdown(socket.SHUT_WR)
                connection.settimeout(1)
                try:
                    while connection.recv(512):
                        pass
                except OSError:
                    pass

    def close(self):
        self.thread.join(5)
        self.server.close()


def test_a_failing_socks_proxy_is_retried_like_a_lost_connection(monkeypatch):
    """The starter suites' native requests reach the model through the SOCKS
    proxy the environment names. A proxy that refuses the connection or
    answers garbage is a transport failure: retried with the same waits and
    recorded as transient, as the agent loops' client does."""
    proxy = _FakeSocksProxy(["refuse", "garbage"])
    # Cleared first: on Windows a lower-case name is the same variable.
    for name in ("NO_PROXY", "no_proxy", "HTTP_PROXY", "HTTPS_PROXY", "http_proxy",
                 "https_proxy", "all_proxy"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("ALL_PROXY", f"socks5://127.0.0.1:{proxy.port}")

    def send(attempt):
        if attempt == 3:
            return "ok"
        with httpx.stream("POST", "http://model.example/api/chat", json={}, timeout=5) as r:
            r.raise_for_status()
        return "unexpected"

    account, waits = _account()
    try:
        outcome = account.execute(send, usage_of=lambda v: mr.Usage(1, 1))
    finally:
        proxy.close()
    assert outcome.value == "ok" and outcome.attempts == 3
    assert waits == [10, 20]
    assert [(r.outcome, r.error_type) for r in account.records] == [
        ("transient", "ProxyError"), ("transient", "ProtocolError"), ("delivered", None)]
    assert account.transient_failures == 2 and account.logical_requests == 1


def test_an_unanswered_reply_is_only_a_load_or_a_reply_that_read_nothing():
    """Only a load reply or a reply of 0 of 0 tokens is no answer.
    An empty reply that read the prompt and stopped is the model's answer."""
    def unanswered(text="", tool_call=False, finish_reason=None, usage=None):
        return mr.unanswered_reply(text=text, tool_call=tool_call,
                                   finish_reason=finish_reason, usage=usage)

    assert unanswered(finish_reason="load")
    assert unanswered(finish_reason="load", usage=mr.Usage(0, 0))
    assert unanswered(finish_reason="stop", usage=mr.Usage(0, 0))
    assert unanswered(usage=mr.Usage(0, 0))
    assert not unanswered(finish_reason="stop", usage=mr.Usage(812, 0))
    assert not unanswered(finish_reason="stop")
    assert not unanswered()
    assert not unanswered(usage=mr.Usage(0, 3))
    assert not unanswered(text="ok", finish_reason="load", usage=mr.Usage(0, 0))
    assert not unanswered(tool_call=True, usage=mr.Usage(0, 0))


def test_an_unanswered_reply_is_retried_with_its_usage_known():
    """The load reply's 0 of 0 tokens are reported usage, recorded as known
    zero rather than unknown."""
    account, _waits = _account()
    empty = mr.empty_reply_error("load", mr.Usage(0, 0))

    outcome = account.execute(_script(empty, "ok"), usage_of=lambda v: mr.Usage(250, 40, 10))

    assert outcome.value == "ok" and outcome.attempts == 2
    assert outcome.unknown_usage_requests == 0
    assert account.unknown_usage_requests == 0
    assert account.transient_failures == 1
    failed = account.records[0].as_dict()
    assert failed["outcome"] == "transient" and failed["error_type"] == "EmptyReply"
    assert (failed["usage_known"], failed["input_tokens"], failed["output_tokens"],
            failed["cache_hit_tokens"]) == (True, 0, 0, 0)


def test_an_unanswered_reply_without_reported_usage_stays_unknown():
    account, _waits = _account()

    outcome = account.execute(_script(mr.empty_reply_error("load", None), "ok"),
                              usage_of=lambda v: mr.Usage(250, 40, 10))

    assert outcome.unknown_usage_requests == 1
    assert account.records[0].usage_known is False


class _StatusError(Exception):
    status_code = 503


@pytest.mark.parametrize("cause, sentence", [
    (httpx.ConnectError("refused at http://10.0.0.5:11434/v1"),
     "The model service could not be reached after 3 attempts: check that it is running "
     "and that the model's address is right, then continue the run."),
    (httpx.ReadTimeout("no answer"),
     "The model service did not answer in time after 3 attempts: check that it is running "
     "and not overloaded, then continue the run."),
    (_StatusError("down"),
     "The model service answered with an error (HTTP 503) after 3 attempts: check its status "
     "and the model's settings, then continue the run."),
])
def test_a_request_given_up_reads_as_one_plain_sentence(cause, sentence):
    """What a display shows for a request given up after its retries: what
    the service did, how often it was tried and what to check, without the
    service's address."""
    exhausted = mr.TransportExhausted("3 consecutive transient failures of one request (limit 3)",
                                      cause, 3)
    assert mr.exhausted_text(exhausted) == sentence
    assert "10.0.0.5" not in mr.exhausted_text(exhausted)


def test_a_request_given_up_names_the_service_without_its_credentials():
    """With the address the phase's model settings give, the sentence names
    the service, without credentials or query. The exception's text is
    never shown."""
    service = mr.service_address("http://user:secret@10.0.0.5:11434/v1?key=abc")
    assert service == "http://10.0.0.5:11434/v1"
    account = mr.RequestAccount(mr.RetrySettings(), service=service)
    exhausted = mr.TransportExhausted("3 consecutive transient failures of one request (limit 3)",
                                      httpx.ConnectError("refused at http://user:secret@10.0.0.5"), 3)
    assert mr.exhausted_text(exhausted, account.service) == (
        "The model service at http://10.0.0.5:11434/v1 could not be reached after 3 attempts: "
        "check that it is running and that the model's address is right, then continue the run.")
    assert mr.service_address("") == mr.service_address(None) == ""


# Exhausted credits or quota end a request at once, even with status 429.
#
# A provider that reports an exhausted account (OpenAI: HTTP 429 with error
# code insufficient_quota) stops the phase instead of being retried. The
# structured error code decides, never the message wording.
class _QuotaStatusError(Exception):
    def __init__(self, status_code, *, code=None, body=None, message="request failed"):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.body = body


class _Response:
    def __init__(self, status_code, payload):
        self.status_code = status_code
        self._payload = payload

    def json(self):
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


class _HTTPStatusError(Exception):
    def __init__(self, response):
        super().__init__("HTTP error")
        self.response = response


@pytest.mark.parametrize("exc", [
    _QuotaStatusError(429, code="insufficient_quota"),
    _QuotaStatusError(429, body={"error": {"code": "insufficient_quota", "type": "insufficient_quota"}}),
    _HTTPStatusError(_Response(429, {"error": {"code": "insufficient_quota"}})),
])
def test_exhausted_quota_is_terminal_even_with_a_rate_limit_status(exc):
    assert mr.classify_failure(exc) == mr.TERMINAL


@pytest.mark.parametrize("exc", [
    _QuotaStatusError(429),
    _QuotaStatusError(429, code="rate_limit_exceeded"),
    # Wording alone is never evidence of an exhausted account.
    _QuotaStatusError(429, message="insufficient_quota: please wait"),
    _HTTPStatusError(_Response(429, ValueError("body not read"))),
])
def test_ordinary_rate_limits_stay_retryable(exc):
    assert mr.classify_failure(exc) == mr.TRANSIENT


def test_a_wrapped_quota_error_is_found_in_the_cause_chain():
    try:
        try:
            raise _QuotaStatusError(429, code="insufficient_quota")
        except _QuotaStatusError as inner:
            raise RuntimeError("client wrapper") from inner
    except RuntimeError as outer:
        assert mr.classify_failure(outer) == mr.TERMINAL


def test_request_account_stops_after_one_quota_refusal_without_waiting():
    waits, calls = [], []
    account = mr.RequestAccount(mr.RetrySettings(), wait_fn=waits.append)

    def send(attempt):
        calls.append(attempt)
        raise _QuotaStatusError(429, code="insufficient_quota")

    with pytest.raises(mr.TerminalRequestError):
        account.execute(send)
    assert calls == [1] and waits == []
    assert [record.outcome for record in account.records] == [mr.TERMINAL]
