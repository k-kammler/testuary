"""Actual request/tool observers render without changing persisted facts."""
from __future__ import annotations

import io
import json
from dataclasses import asdict
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import AIMessage, HumanMessage, messages_to_dict
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.prebuilt import ToolNode
from rich.console import Console

from testuary.agent import _langgraph as graph
from testuary import model_requests as requests
from testuary._progress import RunProgress, ToolObserver, observe_target, phase_observer
from testuary.presentation import TerminalPresenter
import test_starter_archives as archive_harness


drive = archive_harness.drive


@pytest.fixture(autouse=True)
def terminal_environment(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")


def _console():
    stream = io.StringIO()
    return Console(file=stream, width=240, height=35, force_terminal=True,
                   color_system=None, highlight=False, markup=False, record=True)


def _recorded_facts(path):
    # Only the clock point differs between the sequential control/display runs.
    return [{key: value for key, value in json.loads(line).items() if key != "t_mono_s"}
            for line in path.read_text(encoding="utf-8").splitlines()]


def _request(directory, phase, sink=None):
    calls, waits = [], []
    tracker = graph.UsageTracker(request_limit=10)
    account = requests.RequestAccount(requests.RetrySettings(first_wait_s=.1),
                                      wait_fn=waits.append)

    def invoke(messages):
        calls.append(messages_to_dict(messages))
        if len(calls) == 1:
            raise httpx.ConnectError("private endpoint credential")
        return AIMessage(content="private response", usage_metadata={
            "input_tokens": 2, "output_tokens": 1, "total_tokens": 3})

    with observe_target(RunProgress(sink) if sink is not None else None, "module.py"):
        logger = graph.UsageLogger(directory / "usage.jsonl",
                                  prompt_transport_path=directory / "transport.jsonl",
                                  observer=phase_observer(phase))
        result = graph._run_model_turn(
            {"messages": [HumanMessage(content="private prompt")]},
            bind=lambda state: (SimpleNamespace(invoke=invoke), []),
            pre_model_hook=None, usage_tracker=tracker, usage_logger=logger,
            trace=graph.TraceLogger(None), request_account=account,
            on_send_again=None)
    return {
        "calls": calls, "waits": waits, "response": result["messages"][0].content,
        "counts": (tracker.logical_requests_used, tracker.requests_used,
                   tracker.unknown_usage_requests, tracker.input_tokens_used,
                   tracker.output_tokens_used),
        "usage": _recorded_facts(directory / "usage.jsonl"),
        "transport": _recorded_facts(directory / "transport.jsonl"),
    }


@pytest.mark.parametrize("mode", ["plain", "detailed", "compact"])
@pytest.mark.parametrize("phase", ["coverage", "mutation"])
def test_actual_request_retries_render_without_changing_requests_or_records(tmp_path, mode, phase):
    control = _request(tmp_path / "control", phase)
    console = _console()
    observations, snapshots = [], []
    with TerminalPresenter(mode, console=console) as presenter:
        assert presenter.mode == mode
        def sink(event):
            observations.append(event)
            snapshots.append(asdict(event))
            presenter(event)

        displayed = _request(tmp_path / "displayed", phase, sink)
    assert displayed == control
    assert displayed["counts"] == (1, 2, 1, 2, 1)
    assert [event.state for event in observations] == [
        "running", "retrying", "waiting", "running", "succeeded"]
    assert len({event.identity for event in observations}) == 1
    assert all(event.phase == phase and event.target == "module.py" and event.total is None
               for event in observations)
    assert [asdict(event) for event in observations] == snapshots
    assert observations[-1].usage.input_tokens is None
    assert observations[-1].usage.output_tokens is None
    assert observations[-1].usage.cached_input_tokens is None
    assert observations[-1].usage.cost_usd is None
    text = console.file.getvalue() if mode == "plain" else console.export_text()
    assert all(state in text for state in ("RETRYING", "WAITING", "SUCCEEDED"))
    assert "model requests 1 of 10" in text
    assert "(2 sent, 1 sent again after connection errors)" in text
    assert all(value not in text for value in (
        "private prompt", "private response", "private endpoint credential",
        "cache-hit tokens", "cost $", "100%"))


@pytest.mark.parametrize("mode", ["plain", "detailed", "compact"])
@pytest.mark.parametrize("fail", [False, True])
def test_actual_tool_outcomes_render_without_exposing_content_or_scores(mode, fail):
    @tool
    def probe(value: str) -> str:
        """Return the fixture observation."""
        if fail:
            raise ValueError("private tool error")
        return "private tool output"

    console = _console()
    observations, snapshots = [], []
    with TerminalPresenter(mode, console=console) as presenter:
        assert presenter.mode == mode
        def sink(event):
            observations.append(event)
            snapshots.append(asdict(event))
            presenter(event)

        with observe_target(RunProgress(sink), "module.java"):
            callback = ToolObserver(phase_observer("mutation"))
        builder = StateGraph(MessagesState)
        builder.add_node("tools", ToolNode([probe], handle_tool_errors=False))
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        node = builder.compile()
        message = AIMessage(content="", tool_calls=[{
            "name": "probe", "args": {"value": "private tool argument"}, "id": "call-1"}])
        if fail:
            with pytest.raises(ValueError, match="private tool error"):
                node.invoke({"messages": [message]}, {"callbacks": [callback]})
        else:
            result = node.invoke({"messages": [message]}, {"callbacks": [callback]})
            assert result["messages"][-1].content == "private tool output"
    assert [event.state for event in observations] == ["running", "failed" if fail else "succeeded"]
    assert len({event.identity for event in observations}) == 1
    assert all(event.target == "module.java" and event.operation.startswith("tool/")
               and not event.metrics and event.usage is None and event.total is None
               for event in observations)
    assert [asdict(event) for event in observations] == snapshots
    assert callback._names == {}
    text = console.file.getvalue() if mode == "plain" else console.export_text()
    # The row names the tool in plain words. The model's own name for it is
    # in the detailed view only.
    assert "Tool" in text
    if mode == "detailed":
        assert "probe failed" in text if fail else "probe done" in text
    else:
        assert "probe" not in text
    # No content and no score: the phase label may name the mutation phase.
    assert all(value not in text for value in (
        "private tool argument", "private tool output", "private tool error",
        "Mutation score", "mutation score", "%"))


@pytest.mark.parametrize("mode", ["plain", "detailed", "compact"])
@pytest.mark.parametrize("language", ["python", "java"])
def test_paid_response_resume_renders_recorded_work_without_a_new_request(mode, language, drive):
    starter = drive.suites / "suite_001"
    starter.mkdir(parents=True)
    # The generation's manifest, written before its first request.
    (drive.suites / archive_harness.starters._GENERATION_MANIFEST).write_text(
        json.dumps({"generation_complete": False}), encoding="utf-8")
    delivery = starter / archive_harness.starters._GENERATION_DELIVERY
    delivery.write_text(json.dumps({
        "slot": 1, "slot_attempt": 1, "seed": 1, "id": "previous:1",
        "text": f"```{language}\n{archive_harness.SOURCES[language]}```",
        "usage": {"input_tokens": 10, "output_tokens": 3}, "physical_requests": 1,
    }), encoding="utf-8")
    console = _console()
    observations = []
    with TerminalPresenter(mode, console=console) as presenter:
        assert presenter.mode == mode

        def sink(event):
            if event.operation.startswith("saved-response/") and event.state == "succeeded":
                assert (starter / f"cleaned{archive_harness.SUFFIXES[language]}").is_file()
            observations.append(event)
            presenter(event)

        with observe_target(RunProgress(sink), "fixture"), pytest.raises(archive_harness._ReachedConsolidation):
            drive.run(language, reuse=True)
    replay = [event for event in observations if event.operation.startswith("saved-response/")]
    assert [event.state for event in replay] == ["running", "succeeded"]
    assert len({event.identity for event in replay}) == 1
    assert all(event.provenance == "recorded" and event.total is None for event in replay)
    assert not any(event.operation.startswith("request/") for event in observations)
    assert drive.model_calls == []
    assert not delivery.exists()
    suffix = archive_harness.SUFFIXES[language]
    assert (starter / f"original{suffix}").read_text("utf-8") == archive_harness.SOURCES[language]
    assert (starter / f"cleaned{suffix}").read_text("utf-8") == archive_harness.SOURCES[language]
    manifest = json.loads((drive.suites / archive_harness.starters._GENERATION_MANIFEST).read_text("utf-8"))
    assert manifest["logical_requests"] == manifest["requests_used"] == 1
    assert manifest["generation_attempts_per_suite"] == {"1": 1}
    text = console.file.getvalue() if mode == "plain" else console.export_text()
    assert "recorded" in text and "SUCCEEDED" in text
    assert "1 model request" in text
    assert archive_harness.SOURCES[language].strip() not in text
