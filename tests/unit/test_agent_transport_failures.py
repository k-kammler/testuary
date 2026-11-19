"""How dropped attempts, outages and the transport ceiling count against
the request limit, through the real model node of ``build_react_graph``, the
real recovery loop and the coverage agent's real ``run_phase``.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any, List, Sequence, TypedDict

import httpx
import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.runnables import Runnable
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from pydantic import BaseModel, Field

import test_coverage_agent as coverage_harness
from testuary.agent import _langgraph as lg
from testuary import model_requests as mr
from testuary import models as model_source
from testuary import models
from testuary.agent import coverage_agent as tc

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _no_classify(state):
    """The graph's classifier, for a test whose loop classifies nothing."""
    return None


class _FailThenScript(FakeMessagesListChatModel):
    """Raises the scripted failures, one per request, then answers from the
    script."""

    failures: List[Any] = []
    requests_seen: int = 0

    def bind_tools(  # type: ignore[override]
        self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any,
    ) -> Runnable:
        return self

    def _generate(self, *args: Any, **kwargs: Any):  # type: ignore[override]
        self.requests_seen += 1
        if self.requests_seen <= len(self.failures):
            raise self.failures[self.requests_seen - 1]
        return super()._generate(*args, **kwargs)


class _Result(BaseModel):
    answer: str = Field(description="answer")


class _State(TypedDict, total=False):
    messages: Annotated[List[BaseMessage], add_messages]
    result: Any
    phase: str


@tool
def _noop() -> str:
    """no-op tool."""
    return "ok"


class _Rejected(Exception):
    """A provider's 401: retrying cannot succeed."""

    status_code = 401


def _final_answer(args=None, *, usage=True) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[{
            "name": "emit_final_answer",
            "args": args or {"answer": "done"}, "id": "f1",
        }],
        response_metadata=(
            {"token_usage": {"prompt_tokens": 250, "completion_tokens": 40}}
            if usage else {}
        ),
    )


def _graph(model, account, tracker, *, structured_output=_Result,
           validator=None, on_retry=None, usage_logger=None):
    return lg.build_react_graph(
        classify_node=_no_classify,
        model=model,
        agent_mode_tools={"p1": [_noop]},
        resolve_agent_mode=lambda _s: "p1",
        state_type=_State,
        structured_output=structured_output,
        output_validator=validator,
        usage_tracker=tracker,
        usage_logger=usage_logger,
        request_account=account,
        on_send_again=on_retry,
    )


def _run(model, tracker, *, settings=None, max_retries=101, validator=None,
         on_retry=None):
    waits: List[float] = []
    account = mr.RequestAccount(settings or mr.RetrySettings(), wait_fn=waits.append)
    graph = _graph(model, account, tracker, validator=validator, on_retry=on_retry)
    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(lg.time, "sleep", lambda _s: None)
        state, error = lg.run_graph_with_retries(
            graph, {"messages": [HumanMessage(content="go")]},
            max_retries=max_retries,
        )
    return state, error, waits, account


# ---------------------------------------------------------------------------
# Charging
# ---------------------------------------------------------------------------


def test_one_dropped_attempt_costs_one_logical_request():
    """Scenario: one dropped attempt in an agent phase."""
    model = _FailThenScript(responses=[_final_answer()],
                            failures=[httpx.ConnectError("refused")])
    tracker = lg.UsageTracker(request_limit=10)

    state, error, waits, _ = _run(model, tracker)

    assert error is None
    assert state["result"].answer == "done"
    assert model.requests_seen == 2
    assert tracker.requests_used == 2
    assert tracker.logical_requests_used == 1
    assert tracker.unknown_usage_requests == 1
    assert (tracker.input_tokens_used, tracker.output_tokens_used) == (250, 40)
    assert waits == [10]


def test_an_outage_with_a_small_limit_does_not_end_the_phase():
    """Scenario: outage with a small limit. Three dropped attempts under a
    limit of two model requests leave one model request spent."""
    model = _FailThenScript(responses=[_final_answer()],
                            failures=[httpx.ConnectError("x")] * 3)
    tracker = lg.UsageTracker(request_limit=2)

    state, error, waits, _ = _run(model, tracker)

    assert error is None
    assert state["result"].answer == "done"
    assert tracker.logical_requests_used == 1
    assert tracker.requests_used == 4
    assert waits == [10, 20, 40]


def test_charging_every_attempt_spends_a_logical_request_per_attempt():
    """Scenario: charge_every_attempt selected."""
    model = _FailThenScript(responses=[_final_answer()],
                            failures=[httpx.ConnectError("x")])
    tracker = lg.UsageTracker(request_limit=10)

    state, error, _, _ = _run(
        model, tracker, settings=mr.RetrySettings(charge_every_attempt=True))

    assert error is None
    assert tracker.logical_requests_used == 2
    assert tracker.requests_used == 2


# ---------------------------------------------------------------------------
# Ceilings
# ---------------------------------------------------------------------------


def test_ceiling_reached_ends_the_run_with_a_transport_error():
    """Scenario: ceiling reached, here with a configured limit of three."""
    model = _FailThenScript(responses=[_final_answer()],
                            failures=[httpx.ConnectError("x")] * 10)
    tracker = lg.UsageTracker(request_limit=100)

    state, error, waits, _ = _run(
        model, tracker, settings=mr.RetrySettings(consecutive_limit=3))

    assert state is None
    assert "TransportExhausted" in error
    assert "3 consecutive transient failures" in error
    assert model.requests_seen == 3
    assert tracker.logical_requests_used == 1
    assert waits == [10, 20]


def test_a_delivered_response_resets_the_consecutive_count():
    """Scenario: count resets. Two turns, each failing twice under a limit of
    three, reach no ceiling."""
    turn_one = AIMessage(content="thinking", response_metadata={
        "token_usage": {"prompt_tokens": 10, "completion_tokens": 1}})
    model = _FailThenScript(
        responses=[turn_one, _final_answer()],
        failures=[httpx.ConnectError("1"), httpx.ConnectError("2")],
    )
    # The second turn's failures are injected after the first delivery.
    real_generate = model._generate
    second_turn_failures = [httpx.ConnectError("3"), httpx.ConnectError("4")]

    def generate(*args, **kwargs):
        if model.requests_seen >= 3 and second_turn_failures:
            model.requests_seen += 1
            raise second_turn_failures.pop(0)
        return real_generate(*args, **kwargs)

    model._generate = generate  # type: ignore[method-assign]
    tracker = lg.UsageTracker(request_limit=100)

    state, error, _, account = _run(
        model, tracker, settings=mr.RetrySettings(consecutive_limit=3))

    assert error is None
    assert state["result"].answer == "done"
    assert account.transient_failures == 4
    assert tracker.logical_requests_used == 2


def test_validation_rejections_and_transport_failures_have_separate_ceilings():
    """Scenario: validation rejections do not use the transport ceiling.
    Nine rejections then nine transient failures reach neither ceiling of
    ten."""
    rejections = {"n": 0}

    def validator(_state, _result):
        if rejections["n"] < 9:
            rejections["n"] += 1
            return "not yet"
        return None

    # Nine delivered rejected answers, then nine drops before the accepted one.
    model = _FailThenScript(responses=[_final_answer()] * 10, failures=[])
    real_generate = model._generate

    def generate(*args, **kwargs):
        if 9 <= model.requests_seen < 18:
            model.requests_seen += 1
            raise httpx.ConnectError("x")
        return real_generate(*args, **kwargs)

    model._generate = generate  # type: ignore[method-assign]
    tracker = lg.UsageTracker(request_limit=100)

    state, error, _, account = _run(
        model, tracker, settings=mr.RetrySettings(consecutive_limit=10),
        max_retries=10, validator=validator)

    assert error is None
    assert state["result"].answer == "done"
    assert rejections["n"] == 9
    assert account.transient_failures == 9
    # Ten delivered responses, each a model request, and nine more sent.
    assert tracker.logical_requests_used == 10
    assert tracker.requests_used == 19


def test_terminal_failure_ends_the_run_with_the_provider_error():
    model = _FailThenScript(responses=[_final_answer()],
                            failures=[_Rejected("invalid api key")])
    tracker = lg.UsageTracker(request_limit=10)

    state, error, waits, _ = _run(model, tracker)

    assert state is None
    assert "TerminalRequestError" in error and "invalid api key" in error
    assert model.requests_seen == 1
    assert waits == []
    assert tracker.logical_requests_used == 1
    assert tracker.unknown_usage_requests == 1


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def test_a_delivered_response_without_usage_is_unknown_not_zero():
    model = _FailThenScript(responses=[_final_answer(usage=False)], failures=[])
    tracker = lg.UsageTracker(request_limit=10)

    _run(model, tracker)

    assert tracker.unknown_usage_requests == 1
    assert (tracker.input_tokens_used, tracker.output_tokens_used) == (0, 0)


def test_a_delivered_response_without_usage_records_an_unknown_cache_split(
        tmp_path, monkeypatch):
    """A reply without usage has no cache split either: its usage line holds
    None for both, not 0 hits and 0 misses."""
    model = _FailThenScript(responses=[_final_answer(usage=False)], failures=[])
    tracker = lg.UsageTracker(request_limit=10)
    account = mr.RequestAccount(mr.RetrySettings(), wait_fn=lambda _s: None)
    usage_path = tmp_path / "usage.jsonl"
    graph = _graph(model, account, tracker, usage_logger=lg.UsageLogger(usage_path))
    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)

    lg.run_graph_with_retries(
        graph, {"messages": [HumanMessage(content="go")]}, max_retries=101)

    line = _usage_lines(usage_path)[0]
    assert (line["usage_known"], line["cache_hit_tokens"], line["cache_miss_tokens"]) == (
        False, None, None)


def test_the_retry_hook_sees_attempt_error_wait_and_messages():
    model = _FailThenScript(responses=[_final_answer()],
                            failures=[httpx.ConnectError("refused")])
    seen = []

    _run(model, lg.UsageTracker(),
         on_retry=lambda a, e, w, m: seen.append((a, type(e).__name__, w, len(m))))

    assert seen == [(1, "ConnectError", 10.0, 1)]


def _usage_lines(path: Path) -> List[dict]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines()
            if line.strip()]


@pytest.mark.parametrize(("settings", "failures", "reason"), [
    (mr.RetrySettings(consecutive_limit=3), 3, "3 consecutive transient failures"),
    (mr.RetrySettings(phase_total=2), 2, "2 transient failures in this phase"),
])
def test_the_attempt_that_reaches_a_ceiling_is_recorded(
    tmp_path, settings, failures, reason, monkeypatch,
):
    """The attempt that reaches a ceiling, for which the retry callback does
    not run, still has its unknown usage and its usage log line."""
    model = _FailThenScript(responses=[_final_answer()],
                            failures=[httpx.ConnectError("x")] * failures)
    tracker = lg.UsageTracker(request_limit=10)
    account = mr.RequestAccount(settings, wait_fn=lambda _s: None)
    usage_path = tmp_path / "usage.jsonl"
    graph = _graph(model, account, tracker,
                   usage_logger=lg.UsageLogger(usage_path))

    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    state, error = lg.run_graph_with_retries(
        graph, {"messages": [HumanMessage(content="go")]},
        max_retries=101,
    )

    assert state is None and "TransportExhausted" in error
    assert model.requests_seen == tracker.requests_used == failures
    assert tracker.unknown_usage_requests == failures
    assert account.unknown_usage_requests == failures
    lines = _usage_lines(usage_path)
    assert [(line["outcome"], line["usage_known"], line["attempt"])
            for line in lines] == [
        ("transient", False, attempt) for attempt in range(1, failures + 1)]
    assert "ends_phase" not in lines[0]
    assert reason in lines[-1]["ends_phase"]
    assert lines[-1]["error_type"] == "ConnectError"


def test_a_retried_attempt_that_reaches_a_token_limit_keeps_its_usage_line(
        tmp_path, monkeypatch):
    """An unanswered reply whose reported usage reaches the input token
    limit has its usage log line before the limit ends the phase."""
    model = _FailThenScript(responses=[_final_answer()], failures=[
        mr.EmptyReply("loading", usage=mr.Usage(600, 0, 0))])
    tracker = lg.UsageTracker(request_limit=10, input_token_limit=500)
    account = mr.RequestAccount(mr.RetrySettings(), wait_fn=lambda _s: None)
    usage_path = tmp_path / "usage.jsonl"
    graph = _graph(model, account, tracker, usage_logger=lg.UsageLogger(usage_path))
    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)

    with pytest.raises(lg.UsageLimitExceeded):
        lg.run_graph_with_retries(
            graph, {"messages": [HumanMessage(content="go")]}, max_retries=101)

    assert [(line["outcome"], line["usage_known"]) for line in _usage_lines(usage_path)] == [
        ("transient", True)]


def test_a_failed_attempt_without_a_reported_cache_split_records_none(
        tmp_path, monkeypatch):
    """An unanswered reply's usage without the provider's cache split is an
    unknown split on its usage line, not zero hits and the whole input
    missed."""
    model = _FailThenScript(responses=[_final_answer()], failures=[
        mr.EmptyReply("loading", usage=mr.Usage(600, 0, None))])
    tracker = lg.UsageTracker(request_limit=10)
    account = mr.RequestAccount(mr.RetrySettings(), wait_fn=lambda _s: None)
    usage_path = tmp_path / "usage.jsonl"
    graph = _graph(model, account, tracker, usage_logger=lg.UsageLogger(usage_path))
    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)

    lg.run_graph_with_retries(
        graph, {"messages": [HumanMessage(content="go")]}, max_retries=101)

    first = _usage_lines(usage_path)[0]
    assert (first["outcome"], first["input_tokens"], first["cache_hit_tokens"],
            first["cache_miss_tokens"]) == ("transient", 600, None, None)


def test_the_attempt_that_reaches_a_ceiling_counts_cache_hits_within_its_input():
    """The ceiling attempt's reported usage is charged as every other
    attempt's: its cache hits count at most its input."""
    model = _FailThenScript(responses=[_final_answer()], failures=[
        mr.EmptyReply("loading", usage=mr.Usage(100, 0, 140))])
    tracker = lg.UsageTracker(request_limit=10)

    state, error, _waits, _account = _run(
        model, tracker, settings=mr.RetrySettings(consecutive_limit=1))

    assert state is None and "TransportExhausted" in error
    assert tracker.input_tokens_used == 100 and tracker.cache_hit_tokens_used == 100


def test_a_retry_wait_that_would_pass_the_deadline_is_not_started(monkeypatch):
    """With a 5 s deadline and a first wait of 10 s, the phase ends as a
    limit stop after the recorded failed attempt, without the wait or a
    second request."""
    model = _FailThenScript(responses=[_final_answer()],
                            failures=[httpx.ConnectError("refused")])
    tracker = lg.UsageTracker(request_limit=10, wall_clock_limit_seconds=5)
    waits: List[float] = []
    account = mr.RequestAccount(mr.RetrySettings(), wait_fn=waits.append)
    graph = _graph(model, account, tracker)

    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    with pytest.raises(lg.UsageLimitExceeded, match="retry wait of 10s"):
        lg.run_graph_with_retries(
            graph, {"messages": [HumanMessage(content="go")]},
            max_retries=101,
        )

    assert waits == []
    assert model.requests_seen == tracker.requests_used == 1
    assert tracker.unknown_usage_requests == 1


def test_no_retry_is_sent_after_the_deadline(monkeypatch):
    """The deadline is also checked before every retry: a wait that overran
    (the clock advances 35 s during a 10 s wait) leaves a 30 s deadline
    behind, and the retry is not sent."""
    model = _FailThenScript(responses=[_final_answer()],
                            failures=[httpx.ConnectError("refused")])
    tracker = lg.UsageTracker(request_limit=10, wall_clock_limit_seconds=30)
    waits: List[float] = []

    def overrunning_wait(seconds: float) -> None:
        waits.append(seconds)
        tracker._started_monotonic -= 35.0

    account = mr.RequestAccount(mr.RetrySettings(), wait_fn=overrunning_wait)
    graph = _graph(model, account, tracker)

    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    with pytest.raises(lg.UsageLimitExceeded, match="exceeded"):
        lg.run_graph_with_retries(
            graph, {"messages": [HumanMessage(content="go")]},
            max_retries=101,
        )

    assert waits == [10]
    assert model.requests_seen == tracker.requests_used == 1
    assert tracker.logical_requests_used == 1


def test_a_retry_within_the_deadline_proceeds():
    model = _FailThenScript(responses=[_final_answer()],
                            failures=[httpx.ConnectError("refused")])
    tracker = lg.UsageTracker(request_limit=10, wall_clock_limit_seconds=3600)

    state, error, waits, _ = _run(model, tracker)

    assert error is None and state["result"].answer == "done"
    assert waits == [10]
    assert model.requests_seen == tracker.requests_used == 2


# ---------------------------------------------------------------------------
# Through run_phase
# ---------------------------------------------------------------------------


def _coverage_answer() -> AIMessage:
    return _final_answer({
        "test_file_name": "test_target.py",
        "explanation": "done",
        "targeted_functions": ["f"],
    })


def _drive_run_phase(tmp_path, monkeypatch, model, *, config_over=None):
    """Run the real coverage ``run_phase`` over a minimal graph that uses the
    account and hook ``run_phase`` builds. The callers check accounting
    fields, not the status policy. A suite on disk makes an error status
    reachable."""
    monkeypatch.chdir(tmp_path)
    agent = coverage_harness._make_agent(tmp_path)
    (tmp_path / "test_target.py").write_text(
        "from target import f\n\ndef test_f():\n    assert f(1) == 2\n",
        encoding="utf-8")
    cfg = dict(tc._AGENT_CONFIG)
    cfg.update(config_over or {})
    monkeypatch.setattr(agent, "agent_config", lambda: cfg)
    captured = {}

    def build_graph(**kwargs):
        captured["account"] = kwargs["request_account"]
        # Waits are recorded, not spent.
        captured["waits"] = []
        kwargs["request_account"]._wait_fn = captured["waits"].append
        return _graph(
            model, kwargs["request_account"], kwargs["usage_tracker"],
            structured_output=agent.result_schema,
            on_retry=kwargs["on_send_again"],
            usage_logger=kwargs["usage_logger"],
        )

    monkeypatch.setattr(agent, "build_graph", build_graph)
    lines: List[str] = []
    monkeypatch.setattr(agent, "log", lambda m, level=0: lines.append(m))
    result = agent.run_phase(project_name="p", user_prompt="go")
    return result, captured, lines, agent.resolve_log_dir()


def test_run_phase_reports_logical_physical_and_unknown_usage_counts(
    tmp_path, monkeypatch,
):
    """Scenario: dropped attempt, through run_phase. The result, the usage
    log and the chat snapshot all show the attempt sent."""
    model = _FailThenScript(responses=[_coverage_answer()],
                            failures=[httpx.ConnectError("refused")])

    result, captured, lines, log_dir = _drive_run_phase(tmp_path, monkeypatch, model)

    assert result.error in (None, "")
    assert result.model_requests == 2
    assert result.logical_requests == 1
    assert result.unknown_usage_requests == 1
    assert (result.input_tokens, result.output_tokens) == (250, 40)
    assert captured["waits"] == [10.0]
    assert any("Sending again after attempt 1/10" in m for m in lines)

    usage_lines = [
        json.loads(line) for line in
        (log_dir / "coverage_usage.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert [(u["outcome"], u["usage_known"], u["logical_request_idx"], u["attempt"])
            for u in usage_lines] == [("transient", False, 1, 1), ("delivered", True, 1, 2)]
    assert "input_tokens" not in usage_lines[0] or usage_lines[0]["input_tokens"] == 0
    assert usage_lines[1]["input_tokens"] == 250
    assert (log_dir / "coverage_chat_pre_retry_001.json").exists()


def test_a_loop_session_after_one_ended_at_the_ceiling_starts_afresh(
    tmp_path, monkeypatch,
):
    """The coverage phase's account is built for each session, so a session
    that ended at the consecutive limit leaves the next one its full
    retries."""
    ended, first, _lines, _ = _drive_run_phase(
        tmp_path, monkeypatch,
        _FailThenScript(responses=[_coverage_answer()],
                        failures=[httpx.ConnectError("x")] * 3),
        config_over={"max_consecutive_transient_failures": 3},
    )
    assert "TransportExhausted" in (ended.error or "")
    resumed, second, _lines, _ = _drive_run_phase(
        tmp_path, monkeypatch,
        _FailThenScript(responses=[_coverage_answer()],
                        failures=[httpx.ConnectError("y")] * 2),
        config_over={"max_consecutive_transient_failures": 3},
    )
    assert resumed.error in (None, "")
    assert second["account"] is not first["account"]
    assert second["account"].transient_failures == 2


def test_run_phase_takes_its_settings_from_the_agent_configuration(
    tmp_path, monkeypatch,
):
    model = _FailThenScript(responses=[_coverage_answer()],
                            failures=[httpx.ConnectError("x")] * 3)

    result, captured, _lines, _ = _drive_run_phase(
        tmp_path, monkeypatch, model,
        config_over={"transient_first_wait_seconds": 1,
                     "max_transient_wait_seconds": 2,
                     "max_consecutive_transient_failures": 5},
    )

    assert result.error in (None, "")
    assert captured["waits"] == [1.0, 2.0, 2.0]
    assert captured["account"].settings.consecutive_limit == 5


def test_run_phase_ends_on_a_terminal_failure_with_the_provider_error(
    tmp_path, monkeypatch,
):
    model = _FailThenScript(responses=[_coverage_answer()],
                            failures=[_Rejected("invalid api key")])

    result, _captured, _lines, log_dir = _drive_run_phase(tmp_path, monkeypatch, model)

    assert result.status == "agent_error"
    assert "invalid api key" in result.error
    assert result.model_requests == 1
    assert result.logical_requests == 1
    assert result.unknown_usage_requests == 1
    usage_lines = [
        json.loads(line) for line in
        (log_dir / "coverage_usage.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]
    assert [u["outcome"] for u in usage_lines] == ["terminal"]
    assert usage_lines[0]["error_type"] == "_Rejected"


def test_run_phase_without_failures_reports_equal_counts(tmp_path, monkeypatch):
    """No-error parity: physical and logical counts agree and nothing is
    unknown."""
    model = _FailThenScript(responses=[_coverage_answer()], failures=[])

    result, captured, lines, _ = _drive_run_phase(tmp_path, monkeypatch, model)

    assert result.error in (None, "")
    assert result.model_requests == result.logical_requests == 1
    assert result.unknown_usage_requests == 0
    assert captured["waits"] == []
    assert not any("Sending again" in m for m in lines)


def _transport_labels(log_dir):
    return {
        json.loads(line)["provider_transport"] for line in
        (log_dir / "coverage_prompt_transport.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    }


def test_run_phase_names_the_compatible_client_for_an_ollama_endpoint(
    tmp_path, monkeypatch, in_run,
):
    """Agent requests to Ollama Cloud go through the OpenAI-compatible
    client, and the transport record names that client, not the host."""
    in_run(overrides={"api": {"model": "fixture", "provider": "ollama"}})
    api = dict(models.get_api_config())
    api["api_url"] = "https://ollama.com/v1"
    monkeypatch.setattr(models, "get_api_config", lambda: dict(api))
    model = _FailThenScript(responses=[_coverage_answer()], failures=[])

    _result, _captured, _lines, log_dir = _drive_run_phase(
        tmp_path, monkeypatch, model)

    assert _transport_labels(log_dir) == {"openai_compatible"}


def test_run_phase_names_the_responses_client_when_configured(
    tmp_path, monkeypatch, in_run,
):
    """A model whose catalogue entry selects the Responses API."""
    in_run(overrides={"api": {"model": "responses"}}, models={"responses": {
        "provider": "openai", "model": "fixture", "api_url": "https://example.test/v1",
        "protocol": "openai-responses"}})
    model = _FailThenScript(responses=[_coverage_answer()], failures=[])

    _result, _captured, _lines, log_dir = _drive_run_phase(tmp_path, monkeypatch, model)

    assert _transport_labels(log_dir) == {"openai_responses"}


def test_run_phase_names_an_installed_model_source(tmp_path, monkeypatch):
    """The demonstration's replaying model is no OpenAI client."""
    model = _FailThenScript(responses=[_coverage_answer()], failures=[])

    with model_source.installed(object()):
        _result, _captured, _lines, log_dir = _drive_run_phase(
            tmp_path, monkeypatch, model)

    assert _transport_labels(log_dir) == {"model_source"}
