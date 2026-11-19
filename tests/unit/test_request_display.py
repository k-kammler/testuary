"""Request projections use recorded facts, never limits or model content."""
import json
from asyncio import CancelledError as AsyncCancelledError
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, START, MessagesState, StateGraph

import test_starter_archives as archive_harness
import test_coverage_agent as coverage_harness
import test_mutation_agent as mutation_harness
import test_resolved_starter_requests as starter_harness
from testuary.agent import _langgraph as lg
from testuary import _chat_log
from testuary import model_requests as requests
from testuary._progress import RunProgress, ToolObserver, observe_target, phase_observer

streams = starter_harness.streams


@pytest.mark.parametrize("protocol", ["openai-chat", "openai-responses", "ollama-native"])
def test_actual_starter_worker_reports_only_after_transport_and_delivery_persistence(protocol, tmp_path, streams):
    events = []
    starter_dir = tmp_path / "suites"

    def observe(event):
        if event.operation.startswith("request/"):
            assert (starter_dir / "prompt_transport.jsonl").is_file()
            if event.state == "succeeded":
                assert (starter_dir / "suite_001" / archive_harness.starters._GENERATION_DELIVERY).is_file()
        events.append(event)

    with observe_target(RunProgress(observe), "module.py"):
        result, _ = starter_harness.run_worker(tmp_path, starter_harness.configured(protocol=protocol))
    projected = [event for event in events if event.operation.startswith("request/")]
    assert [event.state for event in projected] == ["running", "succeeded"]
    assert len({event.identity for event in projected}) == 1
    assert result.requests_used == result.logical_requests == len(streams[0]) == 1
    # Request events carry no limit as a total. Starter-suite progress counts
    # the accepted suites of those requested.
    assert all(event.total is None for event in events if event.operation != "suite-processing")
    assert {event.total for event in events if event.operation == "suite-processing"} == {1}
    assert all("no code fence" not in event.message and "fixture" not in event.message for event in events)
    saved = [event for event in events if event.operation == "suite-processing"][-1]
    assert saved.usage.input_tokens == result.input_tokens == 10
    assert saved.usage.output_tokens == result.output_tokens == 3
    assert saved.usage.cached_input_tokens is None and saved.usage.cost_usd is None
    record = json.loads((starter_dir / "request_records.jsonl").read_text().splitlines()[-1])
    outbound = json.loads((starter_dir / "prompt_transport.jsonl").read_text().splitlines()[-1])
    assert outbound["run"] == record["run"]
    assert outbound["slot"] == record["slot"] == 1
    assert outbound["logical_request_idx"] == record["logical_request_idx"] == 1
    assert outbound["request_idx"] == record["request_idx"] == 1
    assert outbound["attempt"] == record["attempt"] == 1
    if protocol == "ollama-native":
        assert "response_model" not in record  # This response reports no model identity.
    else:
        assert record["response_model"] == "reported"
        assert json.loads((starter_dir / "suite_001/original.json").read_text())["response_model"] == "reported"


def test_a_starter_request_sent_again_keeps_one_identity_and_unknown_tokens(tmp_path, streams):
    events, calls = [], []

    def generate(*args, **kwargs):
        calls.append(kwargs)
        if len(calls) == 1:
            raise httpx.ConnectError("private endpoint credential")
        return "no code fence", {"input_tokens": 10, "output_tokens": 3}

    cfg = starter_harness.configured(phase_fields={
        "transient_first_wait_seconds": .001, "max_transient_wait_seconds": .001})
    with observe_target(RunProgress(events.append), "module.py"):
        result, starter_dir = starter_harness.run_worker(tmp_path, cfg, generate=generate)
    projected = [event for event in events if event.operation.startswith("request/")]
    assert [event.state for event in projected] == ["running", "retrying", "waiting", "running", "succeeded"]
    assert len({event.identity for event in projected}) == 1
    assert (result.requests_used, result.logical_requests, result.unknown_usage_requests) == (2, 1, 1)
    records = [json.loads(line) for line in (starter_dir / "request_records.jsonl").read_text().splitlines()]
    assert [record["outcome"] for record in records] == [requests.TRANSIENT, requests.DELIVERED]
    assert [record["request_idx"] for record in records] == [1, 2]
    assert events[-1].usage.input_tokens is None and events[-1].usage.output_tokens is None
    assert all("private endpoint" not in event.message for event in events)


@pytest.mark.parametrize("unknown", [False, True])
def test_model_turn_projects_existing_counts_without_guessing_cache_or_cost(tmp_path, unknown):
    events, calls = [], []
    progress = RunProgress(events.append)
    tracker = lg.UsageTracker(request_limit=10)
    account = requests.RequestAccount(requests.RetrySettings(first_wait_s=.1), wait_fn=lambda wait: None)
    usage = None if unknown else {"input_tokens": 2, "output_tokens": 1, "total_tokens": 3}

    def invoke(messages):
        calls.append(messages)
        if len(calls) == 1:
            raise httpx.ConnectError("private endpoint credential")
        return AIMessage(content="private response", usage_metadata=usage)

    with observe_target(progress, "module.py"):
        logger = lg.UsageLogger(tmp_path / "usage.jsonl", prompt_transport_path=tmp_path / "transport.jsonl",
                                observer=phase_observer("coverage"))
    result = lg._run_model_turn({"messages": [HumanMessage(content="private prompt")]},
        bind=lambda state: (SimpleNamespace(invoke=invoke), []), pre_model_hook=None,
        usage_tracker=tracker, usage_logger=logger, trace=lg.TraceLogger(None),
        request_account=account, on_send_again=None)
    assert result["messages"][0].content == "private response"
    assert [event.state for event in events] == ["running", "retrying", "waiting", "running", "succeeded"]
    assert len({event.identity for event in events}) == 1
    assert events[-1].usage.logical_requests == tracker.logical_requests_used == 1
    assert events[-1].usage.physical_requests == tracker.requests_used == 2
    assert events[-1].usage.input_tokens is None  # Failed attempt has unknown usage.
    assert events[-1].usage.cached_input_tokens is events[-1].usage.cost_usd is None
    assert all("private" not in event.message and event.total is None for event in events)
    assert len((tmp_path / "usage.jsonl").read_text().splitlines()) == 2


@pytest.mark.parametrize("failure", ["terminal", "exhausted", "cancelled", "token-limit"])
def test_request_failure_is_terminal_without_a_false_success(tmp_path, failure):
    events = []
    account = requests.RequestAccount(requests.RetrySettings(consecutive_limit=10 if failure == "cancelled" else 1),
                                     wait_fn=lambda wait: account.cancel.set() if failure == "cancelled" else None)
    tracker = lg.UsageTracker(output_token_limit=1 if failure == "token-limit" else None)

    def invoke(messages):
        if failure == "terminal":
            raise ValueError("private invalid request")
        if failure == "exhausted":
            raise httpx.ConnectError("private connection")
        if failure == "cancelled":
            raise httpx.ConnectError("private cancelled request")
        return AIMessage(content="private response", usage_metadata={"input_tokens": 1, "output_tokens": 2, "total_tokens": 3})

    with observe_target(RunProgress(events.append), "module.py"):
        logger = lg.UsageLogger(tmp_path / "usage.jsonl", observer=phase_observer("mutation"))
    with pytest.raises((requests.TerminalRequestError, requests.TransportExhausted,
                        requests.RequestCancelled, lg.UsageLimitExceeded)):
        lg._run_model_turn({"messages": [HumanMessage(content="private prompt")]},
            bind=lambda state: (SimpleNamespace(invoke=invoke), []), pre_model_hook=None,
            usage_tracker=tracker, usage_logger=logger, trace=lg.TraceLogger(None),
            request_account=account, on_send_again=None)
    # A token limit the settings set ends the request as limit_reached.
    assert events[-1].state == {"cancelled": "cancelled", "token-limit": "limit_reached"}.get(
        failure, "failed")
    assert all(event.state != "succeeded" and "private" not in event.message for event in events)


def test_a_playback_without_an_answer_ends_the_turn_failed_and_passes_the_stop_on(tmp_path):
    """A demonstration whose scenario holds no answer stops with ScenarioStop,
    which no request policy classifies: the turn reports the request as
    failed and lets the same stop through, so the playback ends as stale."""
    from testuary.demo.scenario import ScenarioStop

    events = []
    stop = ScenarioStop(SimpleNamespace(describe=lambda: "model call"), "the scenario holds no answer")

    def invoke(messages):
        raise stop

    with observe_target(RunProgress(events.append), "module.py"):
        logger = lg.UsageLogger(tmp_path / "usage.jsonl", observer=phase_observer("coverage"))
    with pytest.raises(ScenarioStop) as raised:
        lg._run_model_turn({"messages": [HumanMessage(content="private prompt")]},
            bind=lambda state: (SimpleNamespace(invoke=invoke), []), pre_model_hook=None,
            usage_tracker=None, usage_logger=logger, trace=lg.TraceLogger(None),
            request_account=None, on_send_again=None)
    assert raised.value is stop
    assert (events[-1].state, events[-1].message) == ("failed", "Model request ended with an error")
    assert all(event.state != "succeeded" and "private" not in event.message for event in events)


@pytest.mark.parametrize("known", [False, True])
def test_a_zero_token_response_is_not_confused_with_unavailable_usage(tmp_path, known):
    events = []
    # A reply with text: one with no text, no tool call and no output tokens
    # is an empty reply, which is retried.
    response = AIMessage(content="ok", usage_metadata=(
        {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0} if known else None))
    tracker = lg.UsageTracker()
    with observe_target(RunProgress(events.append), "module.py"):
        logger = lg.UsageLogger(tmp_path / "usage.jsonl", observer=phase_observer("coverage"))
    lg._run_model_turn({"messages": []}, bind=lambda state: (SimpleNamespace(invoke=lambda messages: response), []),
        pre_model_hook=None, usage_tracker=tracker, usage_logger=logger, trace=lg.TraceLogger(None),
        request_account=None, on_send_again=None)
    assert events[-1].state == "succeeded"
    assert events[-1].usage.input_tokens == (0 if known else None)
    assert events[-1].usage.output_tokens == (0 if known else None)


@pytest.mark.parametrize("fail", [False, True, "cancelled"])
def test_actual_tool_node_reports_runs_without_arguments_outputs_or_errors(fail):
    @tool
    def probe(value: str) -> str:
        """Probe the fixture."""
        if fail == "cancelled":
            raise KeyboardInterrupt("private tool interruption")
        if fail:
            raise ValueError("private tool error")
        return "private output"

    events = []
    with observe_target(RunProgress(events.append), "module.py"):
        callback = ToolObserver(phase_observer("mutation"))
    message = AIMessage(content="", tool_calls=[{"name": "probe", "args": {"value": "private argument"}, "id": "call-1"}])
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode([probe], handle_tool_errors=False))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    node = builder.compile()
    with ThreadPoolExecutor(max_workers=1) as pool:
        if fail:
            with pytest.raises(KeyboardInterrupt if fail == "cancelled" else ValueError,
                               match="private tool interruption" if fail == "cancelled" else "private tool error"):
                pool.submit(node.invoke, {"messages": [message]}, {"callbacks": [callback]}).result()
        else:
            assert pool.submit(node.invoke, {"messages": [message]}, {"callbacks": [callback]}).result()["messages"][-1].content == "private output"
    terminal = "cancelled" if fail == "cancelled" else "failed" if fail else "succeeded"
    assert [event.state for event in events] == ["running", terminal]
    assert len({event.identity for event in events}) == 1
    assert all(event.usage is None and event.total is None and "private" not in event.message for event in events)
    assert callback._names == {}


@pytest.mark.parametrize("harness", [coverage_harness, mutation_harness], ids=["coverage", "mutation"])
@pytest.mark.parametrize("stage", ["graph", "finalization", "chat"])
@pytest.mark.parametrize("cancelled", [False, True])
def test_actual_phase_closes_output_recovery_when_execution_raises(harness, stage, cancelled, tmp_path, monkeypatch, default_run):
    agent = harness._make_agent(tmp_path)
    agent._generation_condition = None
    cls = type(agent)
    events = []
    error = AsyncCancelledError("private interruption") if cancelled else OSError("private finalization failure")
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(cls, "resolve_log_dir", lambda self: log_dir)
    monkeypatch.setattr(cls, "build_graph", lambda self, **kwargs: object())

    def graph(graph, state, **kwargs):
        kwargs["on_before_retry"](1, state, lg.OutputValidationError("private rejected output"))
        assert list(log_dir.glob("*retry*"))
        if stage == "graph":
            raise error
        return {"result": "fixture", "messages": []}, None

    def finalized(self, output, **kwargs):
        assert kwargs["usage_tracker"].requests_used == 0
        if stage == "finalization":
            raise error
        return output

    original = _chat_log.make_chat_log_flusher

    def flusher(**kwargs):
        write = original(**kwargs)

        def flush(messages, **options):
            if options.get("stage") == "completed" and stage == "chat":
                raise error
            return write(messages, **options)

        return flush

    monkeypatch.setattr(lg, "run_graph_with_retries", graph)
    monkeypatch.setattr(cls, "finalize_result", finalized)
    monkeypatch.setattr(_chat_log, "make_chat_log_flusher", flusher)
    with observe_target(RunProgress(events.append), "module.py"), pytest.raises(type(error)) as raised:
        agent.run_phase(project_name="fixture", user_prompt="private prompt")
    assert raised.value is error
    projected = [event for event in events if event.operation == "output-validation"]
    assert [event.state for event in projected] == ["retrying", "cancelled" if cancelled else "failed"]
    assert len({event.identity for event in projected}) == 1
    assert all("private" not in event.message for event in events)


@pytest.mark.parametrize("harness", [coverage_harness, mutation_harness], ids=["coverage", "mutation"])
def test_a_final_answer_a_limit_ends_names_the_limit_first(harness, tmp_path, monkeypatch, default_run):
    """After a rejected final answer, the limit that ends the loop is named
    first on the final-answer line, as on the step it ends."""
    agent = harness._make_agent(tmp_path)
    agent._generation_condition = None
    cls = type(agent)
    events = []
    monkeypatch.setattr(cls, "resolve_log_dir", lambda self: tmp_path / "logs")
    monkeypatch.setattr(cls, "build_graph", lambda self, **kwargs: object())

    def graph(graph, state, **kwargs):
        kwargs["on_before_retry"](1, state, lg.OutputValidationError("rejected output"))
        raise lg.UsageLimitExceeded("the request limit was reached", limit="request limit")

    monkeypatch.setattr(lg, "run_graph_with_retries", graph)
    monkeypatch.setattr(cls, "finalize_result", lambda self, output, **kwargs: output)
    with observe_target(RunProgress(events.append), "module.py"):
        agent.run_phase(project_name="fixture", user_prompt="prompt")
    final = [(event.state, event.message) for event in events if event.operation == "output-validation"]
    assert final[-1] == ("limit_reached", "Request limit reached before a usable final answer")


@pytest.mark.parametrize("harness", [coverage_harness, mutation_harness], ids=["coverage", "mutation"])
def test_a_rerun_after_a_tool_failure_is_not_an_output_validation_retry(harness, tmp_path, monkeypatch, default_run):
    agent = harness._make_agent(tmp_path)
    agent._generation_condition = None
    cls = type(agent)
    events = []
    log_dir = tmp_path / "logs"
    monkeypatch.setattr(cls, "resolve_log_dir", lambda self: log_dir)
    monkeypatch.setattr(cls, "build_graph", lambda self, **kwargs: object())

    def graph(graph, state, **kwargs):
        kwargs["on_before_retry"](1, state, RuntimeError("tool subprocess timed out"))
        assert list(log_dir.glob("*retry*"))
        return {"result": "fixture", "messages": []}, None

    monkeypatch.setattr(lg, "run_graph_with_retries", graph)
    monkeypatch.setattr(cls, "finalize_result", lambda self, output, **kwargs: output)
    with observe_target(RunProgress(events.append), "module.py"):
        agent.run_phase(project_name="fixture", user_prompt="private prompt")
    assert [event for event in events if event.operation == "output-validation"] == []


def test_captured_worker_observer_does_not_leak_into_another_target():
    first, second = [], []
    with observe_target(RunProgress(first.append), "first"):
        observer = phase_observer("starter")
    with observe_target(RunProgress(second.append), "second"), ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(observer, operation="fixture", state="running").result()
    assert first[0].target == "first" and second == []
    assert phase_observer("starter") is None


def test_broken_starter_observer_cannot_lose_a_response_or_trigger_a_new_request(tmp_path, streams):
    calls = []

    def broken(event):
        calls.append(event)
        raise RuntimeError("private observer error")

    with observe_target(RunProgress(broken), "module.py"):
        result, starter_dir = starter_harness.run_worker(tmp_path, starter_harness.configured())
    assert len(calls) == len(streams[0]) == result.requests_used == 1
    assert json.loads((starter_dir / "suite_001/original.json").read_text())["response_model"] == "reported"
    assert json.loads((starter_dir / "request_records.jsonl").read_text())["outcome"] == requests.DELIVERED
