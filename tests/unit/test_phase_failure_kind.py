"""A failed phase record says whether its cause was infrastructure or the method.

Reason text alone would mix method outcomes (no usable suite within the
limits) with infrastructure ones (the ceiling on requests sent again, terminal
request errors, tool failures), so a method failure could be rerun as missing
data. An unknown cause leaves the field absent.
"""
from __future__ import annotations

from types import SimpleNamespace

import pytest

import test_starter_phase as harness
from testuary.agent import _langgraph as lg
from testuary import model_requests as mr
from testuary import phase_state
from testuary import starters
from testuary import target_phases as phases

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def test_the_record_keeps_a_known_kind_and_leaves_an_unknown_one_absent(tmp_path):
    phase_state.mark_failed(tmp_path, "coverage", reason="request limit ended", failure_kind="method")
    assert phase_state.load(tmp_path)["coverage"]["0"]["failure_kind"] == "method"
    phase_state.mark_failed(tmp_path, "coverage", reason="unjudged end")
    block = phase_state.load(tmp_path)["coverage"]["0"]
    assert "failure_kind" not in block and block["reason"] == "unjudged end"
    with pytest.raises(ValueError, match="failure kind"):
        phase_state.mark_failed(tmp_path, "coverage", reason="x", failure_kind="other")


def test_a_failed_record_is_still_redone_on_resume(tmp_path):
    phase_state.mark_failed(tmp_path, "mutation", reason="sent again until no attempt was left",
                            failure_kind="infrastructure")
    state = phase_state.load(tmp_path)
    assert phase_state.validate(state, "mutation", target_dir=tmp_path, target_file_path="m.py",
                                canonical_source_path=tmp_path / "m.py") is False


def test_request_failures_are_infrastructure_and_other_errors_unjudged():
    exhausted = mr.TransportExhausted("10 consecutive transient failures", None, 10)
    assert mr.request_failure_kind(exhausted) == "infrastructure"
    assert mr.request_failure_kind(mr.TerminalRequestError(ValueError("401"))) == "infrastructure"
    try:
        try:
            raise exhausted
        except mr.TransportExhausted as inner:
            raise RuntimeError("graph wrapper") from inner
    except RuntimeError as wrapped:
        assert mr.request_failure_kind(wrapped) == "infrastructure"
    assert mr.request_failure_kind(ValueError("tool crashed")) is None
    assert mr.request_failure_kind(None) is None


def test_a_context_window_refusal_is_unjudged_as_its_status_is():
    from testuary.agent._base import CodingAgent

    refusal = mr.TerminalRequestError(ValueError("maximum context length is 65536 tokens"))
    message = f"non-retryable error: TerminalRequestError: {refusal}"
    assert CodingAgent.classify_error_status(message) == "context_overflow"
    assert CodingAgent.failure_kind_of(refusal) is None
    assert CodingAgent.failure_kind_of(mr.TerminalRequestError(ValueError("401"))
    ) == "infrastructure"
    # Retries that ran out after a gateway's context deadline are an
    # infrastructure failure, although the text mentions a context.
    deadline = mr.TransportExhausted("10 consecutive transient failures",
                                     ValueError("context deadline exceeded"), 10)
    assert CodingAgent.failure_kind_of(deadline) == "infrastructure"
    assert CodingAgent.classify_error_status(
        f"non-retryable error: TransportExhausted: {deadline}") == "agent_error"


def test_status_and_failure_kind_use_one_context_recognizer():
    # llama.cpp's refusal is a context-window failure for both.
    from testuary.agent._base import CodingAgent

    refusal = mr.TerminalRequestError(
        ValueError("the request exceeds the available context size"))
    message = f"non-retryable error: TerminalRequestError: {refusal}"
    assert CodingAgent.classify_error_status(message) == "context_overflow"
    assert CodingAgent.failure_kind_of(refusal) is None


def test_the_graph_attempts_are_bounded_at_one_hundred():
    # 100 graph attempts in all, the first included.
    from testuary import config as config_module
    from testuary.agent._base import CodingAgent

    assert config_module.get_agent_config("test_coverage")["max_loop_attempts"] == 100
    assert CodingAgent.graph_attempts({"max_loop_attempts": 100}) == 100
    assert CodingAgent.graph_attempts({"max_loop_attempts": 0}) == 1


def test_the_recovery_loop_hands_back_the_exception_that_ended_it():
    class Graph:
        def stream(self, state, config=None, stream_mode=None):
            raise mr.TransportExhausted("10 consecutive transient failures", None, 10)
            yield  # pragma: no cover

    failure = {}
    state, error = lg.run_graph_with_retries(Graph(), {"messages": []}, max_retries=2,
                                             failure_out=failure)
    assert state is None and "TransportExhausted" in error
    assert mr.request_failure_kind(failure["exception"]) == "infrastructure"


@pytest.mark.parametrize("status,kind", [
    ("failed_model_request", "infrastructure"),
    ("cancelled_zero_passing", "method"),
    ("validation_unresolved", None),
    ("failed_phase_selection", None),
])
def test_the_starter_phase_judges_only_the_causes_it_knows(tmp_path, monkeypatch, status, kind):
    root, target, _, _ = harness._project(tmp_path)
    monkeypatch.setattr(starters, "run_starter_phase", lambda **kw: starters.StarterResult(
        status=status, detail=f"{status} detail"))
    ctx = SimpleNamespace(top_file={"file_path": str(target)}, project_name="p",
                          work_tests=tmp_path / "tests", results_dir=tmp_path / "out",
                          workspace="rank1", log_print=lambda *a, **k: None, run_idx=0,
                          prepared_models=None, resume=False)
    result = phases.TestPipelineResult()
    phases._phase_starter(ctx, result)
    assert result.phase_outcomes["starter"] == "failed"
    assert result.phase_outcomes.get("starter_failure_kind") == kind


def test_the_checkpoint_writes_the_kind_the_phase_recorded(tmp_path):
    result = phases.TestPipelineResult()
    phases._record_outcome(result, "coverage", "failed", "the request limit ended before a measurement",
                           failure_kind=phase_state.METHOD_FAILURE)
    ctx = SimpleNamespace(results_dir=tmp_path, workspace="rank1_m", run_idx=0, progress=None,
                          log_print=lambda *a, **k: None)
    phases._checkpoint_phase(ctx, result, "coverage")
    block = phase_state.load(tmp_path / "rank1_m")["coverage"]["0"]
    assert (block["status"], block["failure_kind"]) == ("failed", "method")
    assert "the request limit ended before a measurement" in block["reason"]


def test_a_phases_own_lines_go_to_the_runs_logger(monkeypatch, tmp_path):
    """A phase's end line goes to the logger the run gives the target's
    phases, its main log, never to the coverage agent's, which writes into
    the coverage log of another target."""
    from testuary.agent import coverage_agent

    monkeypatch.setattr(coverage_agent, "log_print",
                        lambda *a, **k: pytest.fail(f"the coverage agent's logger got {a}"))
    lines = []
    result = phases.TestPipelineResult()
    phases._record_outcome(result, "coverage", "failed", "the request limit ended before a measurement")
    ctx = SimpleNamespace(results_dir=tmp_path, workspace="rank1_m", run_idx=0, progress=None,
                          log_print=lambda text, **kw: lines.append(text))
    phases._checkpoint_phase(ctx, result, "coverage")
    assert any("coverage" in line for line in lines), lines


def test_a_finishing_measurement_without_its_score_stays_unjudged(monkeypatch, tmp_path):
    """A score missing because the delivered suite is no longer green, often
    a flaky generated suite, is the method's own outcome and must not invite
    an infrastructure rerun."""
    import test_orchestrator_phases as pipeline

    ctx = pipeline._make_ctx(tmp_path)
    pipeline._green_pipeline(monkeypatch, ctx)
    monkeypatch.setattr(phases.measurement, "run_finishing", lambda *a, **k:
                        pipeline.FinishingMetrics(mutation_score_full=None))
    monkeypatch.setattr(phases, "_verify_restored_phase_validity", lambda *a: (True, ""))
    result = phases.run_target_phases(ctx)
    assert result.phase_outcomes["finishing"] == "failed"
    assert "finishing_failure_kind" not in result.phase_outcomes
