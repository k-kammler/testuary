"""Unit tests for the coverage agent (graph + validator surface)."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, Sequence

import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from agent_fixtures import logs_in, target_agent
from agent_prompts import python_prompts

from offered_tools import offered_tools
from testuary.agent import _langgraph as lg
from testuary.agent import coverage_agent as tc
from phase_values import set_phase_value
import functools
import os
from test_agent_normal_runs import TARGET
from testuary import config
from testuary.settings import RunSettings

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


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


def _make_agent(tmp_path, stem: str = "target") -> "tc.CoverageAgent":
    """The coverage harness's agent, which the request tests build beside
    test_mutation_agent's ``_make_agent``."""
    return target_agent(tmp_path, stem=stem)


@pytest.fixture
def agent(tmp_path):
    # Past the forced initial read, so the classifier tests see the working loop.
    a = target_agent(tmp_path)
    a.session.has_read = True
    a.session.initial_measured = True  # past the start's measurement
    return a


def test_several_edits_in_one_response_all_land(agent):
    """Two edit_tests calls in one response, on different lines, both reach
    the file (parallel threads would lose one to the last write)."""
    test_file = Path(agent.deps.test_output_dir) / "test_target.py"
    test_file.write_text(
        "from target import f\n\n\ndef test_a():\n    assert f(1) == 2\n\n\n"
        "def test_b():\n    assert f(2) == 3\n", encoding="utf-8")
    calls = [
        {"name": "edit_tests", "id": "e0",
         "args": {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"}},
        {"name": "edit_tests", "id": "e1",
         "args": {"old_text": "    assert f(2) == 3\n", "new_text": "    assert f(2) == 3\n    assert f(-1) == 0\n"}},
    ]
    graph = agent.build_graph(
        model=_ToolAwareFakeModel(responses=[AIMessage(content="", tool_calls=calls)]),
        usage_tracker=lg.UsageTracker(request_limit=1),
    )
    results = []
    with pytest.raises(lg.UsageLimitExceeded):
        for update in graph.stream({"messages": [HumanMessage(content="x")]},
                                   stream_mode="updates"):
            for node in update.values():
                results += [m for m in (node or {}).get("messages", [])
                            if getattr(m, "type", "") == "tool"]
    assert [m.tool_call_id for m in results] == ["e0", "e1"]
    text = test_file.read_text(encoding="utf-8")
    assert "assert f(0) == 1" in text and "assert f(-1) == 0" in text


def test_a_temporary_failure_after_an_edit_landed_asks_to_check_it(agent, monkeypatch):
    """A temporary error after the edit was written reruns the loop with an
    answer that asks to check the edit before repeating it, not one that
    claims the call did nothing."""
    import httpx

    test_file = Path(agent.deps.test_output_dir) / "test_target.py"
    test_file.write_text("from target import f\n\n\ndef test_a():\n    assert f(1) == 2\n",
                         encoding="utf-8")
    original = agent.after_write
    failures = []

    def after_write():
        original()
        if not failures:
            failures.append(1)
            raise httpx.ConnectError("record store unreachable")

    monkeypatch.setattr(agent, "after_write", after_write)
    requests = []

    class _Recording(_ToolAwareFakeModel):
        def invoke(self, input, config=None, **kwargs):
            requests.append(list(input))
            return super().invoke(input, config, **kwargs)

    edit = {"name": "edit_tests", "id": "e0", "args": {
        "old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"}}
    graph = agent.build_graph(
        model=_Recording(responses=[AIMessage(content="", tool_calls=[edit]),
                                    AIMessage(content="", tool_calls=[edit])]),
        usage_tracker=lg.UsageTracker(request_limit=2))
    monkeypatch.setattr(lg.time, "sleep", lambda _s: None)
    with pytest.raises(lg.UsageLimitExceeded):
        lg.run_graph_with_retries(graph, {"messages": [HumanMessage(content="x")]},
                                  max_retries=3)
    assert "assert f(0) == 1" in test_file.read_text(encoding="utf-8")
    answers = [m.content for m in requests[1] if getattr(m, "type", "") == "tool"]
    assert answers == ["The tool failed with a temporary error (ConnectError) and may not have "
                       "completed. Check whether it took effect before calling it again."]


def test_inherited_at_goal_stops_before_model_and_preserves_source(
        tmp_path, monkeypatch):
    target = tmp_path / "target.py"
    target.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    delivered = test_dir / "test_target.py"
    source = "from target import f\n\ndef test_f():\n    assert f(1) == 2\n"
    delivered.write_text(source, encoding="utf-8")
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "both")
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal", 90)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "extra_measurements_after_goal", 0)

    def forbidden(*args, **kwargs):
        raise AssertionError("the at-goal path must not enter the model loop")

    monkeypatch.setattr(tc.CoverageAgent, "run_phase", forbidden)

    logs_in(monkeypatch, tmp_path / "logs")
    result = tc.generate_coverage_tests(
        "project",
        top_file={"file_path": str(target), "key_functions": ["f"]},
        test_output_dir=str(test_dir),
        baseline={
            "content": source,
            "line_coverage": 100.0,
            "branch_coverage": 100.0,
            "suite_seconds": 0.01,
            "test_count": 1,
        },
    )

    assert result.status == "success"
    assert result.agent_assess_cov == 0
    assert result.model_requests == 0
    assert result.input_tokens == result.output_tokens == 0
    assert result.cache_hit_tokens == 0
    assert delivered.read_text(encoding="utf-8") == source


def test_inherited_shadowing_fails_the_phase_handoff(tmp_path):
    target = tmp_path / "target.py"
    target.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    delivered = test_dir / "test_target.py"
    source = (
        "class TestF:\n"
        "    def test_f(self):\n        assert False\n\n"
        "    def test_f(self):\n        assert True\n"
    )
    delivered.write_text(source, encoding="utf-8")

    with pytest.raises(RuntimeError, match="shadowed definitions"):
        tc.generate_coverage_tests(
            "project",
            top_file={"file_path": str(target), "key_functions": ["f"]},
            test_output_dir=str(test_dir),
            baseline={
                "content": source,
                "line_coverage": 100.0,
                "branch_coverage": 100.0,
                "suite_seconds": 0.01,
                "test_count": 1,
            },
        )

    assert delivered.read_text(encoding="utf-8") == source


def test_edit_gate_uses_backend_normalization_for_helpers_and_methods(agent):
    helper_duplicate = (
        "def helper():\n    return 1\n\n"
        "def helper():\n    return 2\n"
    )
    method_duplicate = (
        "class TestX:\n"
        "    def test_value(self):\n        assert 1\n\n"
        "    def test_value(self):\n        assert 2\n"
    )

    assert "helper" in agent._shadowed_definition_reject(helper_duplicate)
    assert "TestX.test_value" in agent._shadowed_definition_reject(
        method_duplicate
    )


def test_final_invariant_rejects_a_shadowed_executable_artifact(agent):
    path = Path(agent.deps.test_output_dir) / "test_target.py"
    path.write_text(
        "def test_x():\n    assert 1\n\n"
        "def test_x():\n    assert 2\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="shadowed definitions"):
        agent.assert_target_has_no_shadowed_definitions("coverage final")


@pytest.mark.parametrize(
    "reason,measured,expected",
    [
        ("coverage_goal_met", False, "success"),
        ("mutation_goal_met", False, "success"),
        ("coverage_stagnation", False, "stagnation"),
        ("mutation_stagnation", False, "stagnation"),
        ("final_answers_without_measurement", False, "stagnation"),
        ("repeated_test_timeout", True, "stagnation"),
        ("repeated_test_timeout", False, "tests_failed"),
        ("inherited_suite_too_slow", False, "skipped_inherited_slow"),
    ],
)
def test_one_stop_status_rule_serves_both_phases(reason, measured, expected):
    """Each framework stop's status, the same in the coverage and the
    mutation phase."""
    assert tc.stop_status(reason, measured=measured) == expected


# ---------------------------------------------------------------------------
# Phase classifier
# ---------------------------------------------------------------------------


def test_offered_tools_entry_is_needs_editing(agent):
    """After the initial read with nothing pending, the mode is needs_editing
    and measure_coverage and run_tests are hidden."""
    names = [t.__name__ for t in offered_tools(agent)]
    assert "measure_coverage" not in names
    assert "run_tests" not in names
    assert {"read_source", "read_tests",
            "edit_file"} <= set(names)


def test_bound_edit_tool_is_named_edit_tests_and_routes_to_edit_file(agent):
    """The bound editor is named ``edit_tests``, backed by the one
    ``edit_file`` body, and invoking it as the ToolNode does edits the file."""
    _reads, writes, _run, _measure = agent.tool_methods()
    assert [m.__name__ for m in writes] == ["edit_file"]      # one generic body
    bound = {agent._bind_edit_tool(m).name: agent._bind_edit_tool(m) for m in writes}
    assert set(bound) == {"edit_tests"}                       # what the model sees

    # The target's test file resolves once it is on disk.
    from pathlib import Path
    stem = Path(agent.deps.target_file_path).stem
    path = Path(agent.deps.test_output_dir) / f"test_{stem}.py"
    path.write_text("def test_a():\n    assert f(1) == 2\n", encoding="utf-8")
    assert agent.target_path() == path
    out = bound["edit_tests"].invoke(
        {"old_text": "    assert f(1) == 2\n",
         "new_text": "    assert f(1) == 2\n    assert f(2) == 3\n"}
    )
    assert "f(2) == 3" in path.read_text(encoding="utf-8")
    assert isinstance(out, str) and out  # the tool returns its echo string


def test_model_is_bound_with_edit_tests_via_real_graph(agent):
    """The real graph binds the editor as ``edit_tests`` in the editing mode,
    never the internal body name or the whole-file tool."""
    captured: set = set()

    class _Recording(_ToolAwareFakeModel):
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            captured.update(
                getattr(t, "name", getattr(t, "__name__", "")) for t in tools
            )
            return self

    # The first decision is needs_editing, and a request limit of 1 stops the
    # run after the first bind.
    model = _Recording(responses=[
        AIMessage(content="", tool_calls=[
            {"name": "read_source", "args": {}, "id": "c1"}]),
    ])
    graph = agent.build_graph(model=model, usage_tracker=lg.UsageTracker(request_limit=1))
    with pytest.raises(lg.UsageLimitExceeded):
        graph.invoke({"messages": [HumanMessage(content="x")]})

    assert "edit_tests" in captured
    assert "edit_file" not in captured
    assert "write_current_tests" not in captured


# ---------------------------------------------------------------------------
# Forced initial read: the source enters as a dedup-able read, not prompt text.
# ---------------------------------------------------------------------------


def test_forces_initial_read_attrs_set():
    assert tc.CoverageAgent.forces_initial_read is True
    assert tc.CoverageAgent.initial_agent_mode == "needs_initial_read"


def test_entry_is_read_only_until_first_read(tmp_path):
    """Before the first read, only the inherited-suite read binds."""
    agent = target_agent(tmp_path)
    assert agent.session.has_read is False
    assert agent.effective_agent_mode() == agent.INITIAL_READ_MODE
    names = {t.__name__ for t in offered_tools(agent)}
    assert names == {"read_tests"}


def test_first_read_flips_has_read_and_unlocks_writes(tmp_path):
    agent = target_agent(tmp_path)
    test_file = Path(agent.deps.test_output_dir) / "test_target.py"
    test_file.write_text("def test_f():\n    assert True\n", encoding="utf-8")
    agent.read_tests()
    assert agent.session.has_read is True
    # The start's second step binds only the measurement.
    assert agent.effective_agent_mode() == "needs_initial_measurement"
    assert {t.__name__ for t in offered_tools(agent)} == {"measure_coverage"}
    agent.session.initial_measured = True
    assert agent.effective_agent_mode() == "needs_editing"
    names = {t.__name__ for t in offered_tools(agent)}
    assert {"edit_file"} <= names


def test_classify_routes_to_initial_read_until_first_read(tmp_path):
    """classify_update lands in the read-only mode until the first read, even
    with edits pending."""
    agent = target_agent(tmp_path)
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = True
    assert agent.classify_update("needs_editing") == {
        "agent_mode": "needs_initial_read"
    }


def test_offered_tools_needs_testing_after_edit(agent):
    """After a write, edits pending: bind run_tests, hide measure_coverage."""
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = True
    agent.session.last_tests_passed = False
    names = set(t.__name__ for t in offered_tools(agent))
    assert "run_tests" in names
    assert "measure_coverage" not in names
    assert {"edit_file"} <= names


def test_offered_tools_needs_measurement_after_green_run(agent):
    """A green run with unmeasured edits binds measure_coverage, not writes or
    run_tests."""
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = False
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_measure = {"coverage": True}
    names = set(t.__name__ for t in offered_tools(agent))
    assert "measure_coverage" in names
    assert "run_tests" not in names
    assert "edit_file" not in names


def test_offered_tools_needs_editing_after_measurement(agent):
    """No edits since measure -> only writes/reads bind."""
    agent.session.tests_changed_since_run_tests = False
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_measure = {"coverage": False}
    names = set(t.__name__ for t in offered_tools(agent))
    assert "measure_coverage" not in names
    assert "run_tests" not in names
    assert {"edit_file"} <= names


def test_offered_tools_needs_editing_after_red_run(agent):
    """Red run with no edit since -> writes only."""
    agent.session.tests_changed_since_run_tests = False
    agent.session.last_tests_passed = False
    names = set(t.__name__ for t in offered_tools(agent))
    assert "run_tests" not in names
    assert "measure_coverage" not in names
    assert {"edit_file"} <= names


def test_classify_run_advances_to_needs_testing_on_edit(agent):
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = True
    assert agent.classify_update("needs_editing") == {"agent_mode": "needs_testing"}


def test_classify_run_advances_to_needs_measurement_on_green_with_edits(agent):
    agent.session.writes_done = 1
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_run_tests = False
    agent.session.tests_changed_since_measure = {"coverage": True}
    assert agent.classify_update("needs_testing") == {"agent_mode": "needs_measurement"}


def test_classify_run_back_to_needs_editing_after_measurement(agent):
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_run_tests = False
    agent.session.tests_changed_since_measure = {"coverage": False}
    assert agent.classify_update("needs_measurement") == {"agent_mode": "needs_editing"}


def test_classify_run_regresses_to_needs_editing_on_red(agent):
    agent.session.last_tests_passed = False
    agent.session.tests_changed_since_run_tests = False
    assert agent.classify_update("needs_testing") == {"agent_mode": "needs_editing"}


def test_classify_run_idempotent_when_phase_unchanged(agent):
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = True
    assert agent.classify_update("needs_testing") == {}


# ---------------------------------------------------------------------------
# Output validator
# ---------------------------------------------------------------------------


def test_validate_output_rejects_placeholder_explanation(agent):
    parsed = tc.CoverageTestResult(
        test_file_name="test_x.py",
        explanation="Brief explanation of test strategy and what is covered",
        targeted_functions=["f"],
    )
    err = agent.validate_output(state={}, parsed=parsed)
    assert err is not None
    assert "placeholder" in err.lower()


def test_validate_output_rejects_empty_targeted_functions(agent):
    parsed = tc.CoverageTestResult(
        test_file_name="test_x.py",
        explanation="Some real-looking explanation about strategy",
        targeted_functions=[],
    )
    err = agent.validate_output(state={}, parsed=parsed)
    assert err is not None
    assert "targeted_functions" in err or "empty" in err.lower()


def test_validate_output_rejects_when_no_coverage_measurement(agent):
    agent.session.coverage_measurements = 0
    parsed = tc.CoverageTestResult(
        test_file_name="test_x.py",
        explanation="Real explanation about strategy",
        targeted_functions=["f"],
    )
    err = agent.validate_output(state={}, parsed=parsed)
    assert err is not None
    assert "measurement" in err.lower() or "coverage" in err.lower()


def test_the_start_measurement_counts_for_a_final_answer(agent):
    """The start's measurement counts as the loop's first: a final answer
    right after it meets the goal and stagnation
    rules instead of a refusal for lack of a measurement, so the
    re-submission cap can end the loop at its entry. After an edit the
    start's measurement no longer describes the suite, and the answer is
    refused for lack of a measurement as before."""
    session = agent.session
    session.coverage_measurements = 0
    session.initial_measured = True
    session.prev_coverage_pct = 40.0
    session.last_tests_passed = True
    session.tests_changed_since_measure["coverage"] = False
    parsed = tc.CoverageTestResult(
        test_file_name="test_x.py",
        explanation="Real explanation about strategy",
        targeted_functions=["f"],
    )
    assert agent.validate_output(state={}, parsed=parsed) == agent.early_stop_reject_message()
    cap = int(agent.agent_config().get("max_final_answers_without_measurement", 10))
    for _ in range(cap - 2):
        assert agent.validate_output(state={}, parsed=parsed) is not None
    assert agent.validate_output(state={}, parsed=parsed) is None
    session.early_stop_result = None
    session.early_submission_rejections = 0
    session.tests_changed_since_measure["coverage"] = True
    assert agent.validate_output(state={}, parsed=parsed) == agent.tool_msg(
        "validator_no_coverage_measurement")


def test_validate_output_rejects_when_tests_not_green(agent):
    agent.session.coverage_measurements = 1
    agent.session.last_tests_passed = False
    parsed = tc.CoverageTestResult(
        test_file_name="test_x.py",
        explanation="Real explanation about strategy",
        targeted_functions=["f"],
    )
    err = agent.validate_output(state={}, parsed=parsed)
    assert err is not None
    assert "run_tests" in err or "pass" in err.lower()


def test_retry_prompt_builder_renders_validation_feedback(agent):
    """The builder reports the rejection without claiming a history change."""
    from testuary.agent import _langgraph as lg
    builder = agent.retry_prompt_builder()
    block = agent.recovery_state_block()
    exc = lg.OutputValidationError("reason text")
    rejected = builder(exc, [])
    assert rejected == (
        block
        + "Your previous attempt to return CoverageTestResult "
        "was rejected by the framework's output validator. "
        "Specific reason:\n\n"
        f"{exc}\n\n"
        "Fix the issue and try again."
    )


def test_validate_output_rejects_stale_measurement_at_acceptance(
    agent, monkeypatch,
):
    """At acceptance, an edit after the last coverage measurement is rejected
    until re-measured."""
    agent.session.coverage_measurements = 1
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_measure["coverage"] = True
    monkeypatch.setattr(agent, "goal_met", lambda: True)
    parsed = tc.CoverageTestResult(
        test_file_name="test_x.py",
        explanation="Real explanation about strategy",
        targeted_functions=["f"],
    )
    err = agent.validate_output(state={}, parsed=parsed)
    assert err is not None and "predates" in err
    agent.session.tests_changed_since_measure["coverage"] = False
    assert agent.validate_output(state={}, parsed=parsed) is None


def test_validate_output_accepts_when_all_gates_pass(agent):
    agent.session.coverage_measurements = 1
    agent.session.last_tests_passed = True
    agent.session.early_stop_result = tc.CoverageTestResult(
        test_file_name="test_x.py",
        explanation="synthesized by framework",
        targeted_functions=["f"],
    )
    parsed = tc.CoverageTestResult(
        test_file_name="test_x.py",
        explanation="Wrote a comprehensive suite covering every branch of f",
        targeted_functions=["f"],
    )
    err = agent.validate_output(state={}, parsed=parsed)
    assert err is None


def test_finalize_result_preserves_exact_coverage(agent, monkeypatch):
    """Strict consolidation floors need instrument precision, not display values."""
    exact_line = 96.24494511842865
    exact_branch = 95.48872180451127
    output = tc.CoverageTestResult(
        test_file_name="test_target.py",
        explanation="Exercises the target function across representative inputs",
        targeted_functions=["f"],
    )
    (Path(agent.deps.test_output_dir) / output.test_file_name).write_text(
        "def test_f():\n    assert True\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(agent, "_maybe_restore_last_green", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_log_run_summary", lambda: None)
    monkeypatch.setattr(
        agent,
        "get_run_metrics",
        lambda: {
            "tool_calls": 1,
            "writes": 1,
            "runs": 1,
            "assess_cov": 1,
            "best_line_coverage": exact_line,
            "final_line_coverage": exact_line,
            "final_branch_coverage": exact_branch,
        },
    )
    monkeypatch.setattr(tc.measurement, "compute_file_complexity", lambda _path: 1)
    result = agent.finalize_result(
        output,
        error_msg=None,
        duration_ms=1000.0,
        usage_tracker=SimpleNamespace(
            requests_used=7,
            input_tokens_used=0,
            output_tokens_used=0,
            cache_hit_tokens_used=0,
        ),
        final_messages=[],
    )
    assert result.line_coverage == exact_line
    assert result.branch_coverage == exact_branch
    assert result.model_requests == 7
    assert result.model_dump()["model_requests"] == 7


def test_a_submission_cap_stop_records_its_green_measured_suite_as_stagnation(agent, monkeypatch):
    """The cap stops only a green, measured suite, so the phase ends as a
    stop before its goal, as the mutation phase records the same stop, not
    as a failed phase."""
    output = tc.CoverageTestResult(test_file_name="test_target.py", explanation="Covers f",
                                   targeted_functions=["f"])
    (Path(agent.deps.test_output_dir) / output.test_file_name).write_text(
        "def test_f():\n    assert True\n", encoding="utf-8")
    agent.session.early_stop_reason = "final_answers_without_measurement"
    monkeypatch.setattr(agent, "_maybe_restore_last_green", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_log_run_summary", lambda: None)
    monkeypatch.setattr(agent, "get_run_metrics", lambda: {
        "tool_calls": 12, "writes": 1, "runs": 2, "assess_cov": 1, "best_line_coverage": 86.0,
        "final_line_coverage": 86.0, "final_branch_coverage": 80.0})
    monkeypatch.setattr(tc.measurement, "compute_file_complexity", lambda _path: 1)
    result = agent.finalize_result(
        output, error_msg=None, duration_ms=1000.0,
        usage_tracker=SimpleNamespace(requests_used=12, input_tokens_used=0, output_tokens_used=0,
                                      cache_hit_tokens_used=0),
        final_messages=[])
    assert (result.status, result.line_coverage) == ("stagnation", 86.0)


def test_finalize_owns_the_test_file_name(agent, monkeypatch):
    """A wrong echoed file name is replaced by the target's test file on disk,
    and without that file the result keeps the sentinel with tests_failed."""
    real = Path(agent.deps.test_output_dir) / "test_foo.py"
    real.write_text(
        "def test_a():\n    assert True\n\n"
        "def test_b():\n    assert True\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(agent, "_maybe_restore_last_green", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_log_run_summary", lambda: None)
    monkeypatch.setattr(
        agent, "get_run_metrics",
        lambda: {
            "tool_calls": 1, "writes": 1, "runs": 1, "assess_cov": 1,
            "best_line_coverage": 90.0,
            "final_line_coverage": 90.0, "final_branch_coverage": 80.0,
        },
    )
    monkeypatch.setattr(tc.measurement, "compute_file_complexity", lambda _p: 1)
    tracker = SimpleNamespace(
        input_tokens_used=0, output_tokens_used=0, cache_hit_tokens_used=0,
    )
    lying = tc.CoverageTestResult(
        test_file_name="test_wrong_name.py",
        explanation="claims a different artifact",
        targeted_functions=["f"],
    )
    out = agent.finalize_result(
        lying, error_msg=None, duration_ms=100.0,
        usage_tracker=tracker, final_messages=[],
    )
    assert out.test_file_name == "test_foo.py"
    assert out.test_count_standard == 2
    assert out.status == "success"

    real.unlink()
    out2 = agent.finalize_result(
        lying, error_msg=None, duration_ms=100.0,
        usage_tracker=tracker, final_messages=[],
    )
    assert out2.test_file_name == "test_failed.py"
    assert out2.status == "tests_failed"


def test_finalize_result_stamps_the_phase_clock_duration(agent, monkeypatch):
    """An armed phase clock stamps the duration from its arm point, an unarmed
    one keeps the passed model-loop span."""
    import time as _time

    output = tc.CoverageTestResult(
        test_file_name="test_target.py",
        explanation="Exercises the target function across inputs",
        targeted_functions=["f"],
    )
    (Path(agent.deps.test_output_dir) / output.test_file_name).write_text(
        "def test_f():\n    assert True\n", encoding="utf-8",
    )
    monkeypatch.setattr(agent, "_maybe_restore_last_green", lambda *a, **k: None)
    monkeypatch.setattr(agent, "_log_run_summary", lambda: None)
    monkeypatch.setattr(
        agent, "get_run_metrics",
        lambda: {
            "tool_calls": 1, "writes": 1, "runs": 1, "assess_cov": 1,
            "best_line_coverage": 90.0,
            "final_line_coverage": 90.0, "final_branch_coverage": 80.0,
        },
    )
    monkeypatch.setattr(tc.measurement, "compute_file_complexity", lambda _path: 1)
    tracker = SimpleNamespace(
        input_tokens_used=0, output_tokens_used=0, cache_hit_tokens_used=0,
    )

    unarmed = agent.finalize_result(
        output, error_msg=None, duration_ms=1500.0,
        usage_tracker=tracker, final_messages=[],
    )
    assert unarmed.agent_duration_s == pytest.approx(1.5)

    agent.session.phase_clock_start = _time.perf_counter() - 10.0
    armed = agent.finalize_result(
        output, error_msg=None, duration_ms=1500.0,
        usage_tracker=tracker, final_messages=[],
    )
    assert armed.agent_duration_s >= 9.9


# ---------------------------------------------------------------------------
# Validator-loop submission-spam cap
# ---------------------------------------------------------------------------


def _validatable_coverage_result() -> "tc.CoverageTestResult":
    return tc.CoverageTestResult(
        test_file_name="test_x.py",
        explanation="Wrote a comprehensive suite covering every branch of f",
        targeted_functions=["f"],
    )


def test_validator_loop_counter_increments_without_new_measurement(agent, monkeypatch):
    """Each rejection without a new measurement bumps the counter, the first
    one included."""
    agent.session.coverage_measurements = 1
    agent.session.last_tests_passed = True
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_final_answers_without_measurement", 10)
    parsed = _validatable_coverage_result()
    err = agent.validate_output(state={}, parsed=parsed)
    assert err is not None
    assert agent.session.early_submission_rejections == 1
    err = agent.validate_output(state={}, parsed=parsed)
    assert err is not None
    assert agent.session.early_submission_rejections == 2


def test_validator_loop_synthesizes_stop_after_submission_cap_exhausted(agent, monkeypatch):
    """The cap-th rejection without progress synthesizes the early-stop
    result and accepts."""
    agent.session.coverage_measurements = 1
    agent.session.last_tests_passed = True
    agent.session.best_coverage_pct = 86.0
    cap = 3
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_final_answers_without_measurement", cap)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "allow_early_stop", True)
    parsed = _validatable_coverage_result()
    for _ in range(cap - 1):
        err = agent.validate_output(state={}, parsed=parsed)
        assert err is not None
    err = agent.validate_output(state={}, parsed=parsed)
    assert err is None
    assert agent.session.early_stop_result is not None
    assert agent.session.early_stop_reason == "final_answers_without_measurement"


def test_validator_give_up_never_touches_the_measurements_without_improvement(agent, monkeypatch):
    """Bare re-submits advance only the submission counter, not the
    stagnation streak, so the stop is reported as submission spam."""
    agent.session.coverage_measurements = 1
    agent.session.last_tests_passed = True
    agent.session.best_coverage_pct = 86.0
    agent.session.measurements_without_improvement = 2
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_measurements_without_improvement", 3)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_final_answers_without_measurement", 4)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "allow_early_stop", True)
    parsed = _validatable_coverage_result()
    for expected_rejections in (1, 2, 3):
        err = agent.validate_output(state={}, parsed=parsed)
        assert err is not None
        assert agent.session.measurements_without_improvement == 2
        assert agent.session.early_submission_rejections == expected_rejections
    # The cap-th re-submit stops, with the streak unchanged.
    err = agent.validate_output(state={}, parsed=parsed)
    assert err is None
    assert agent.session.measurements_without_improvement == 2
    assert agent.session.early_stop_result is not None
    assert agent.session.early_stop_reason == "final_answers_without_measurement"


def test_validator_give_up_counter_resets_on_real_progress(agent, monkeypatch):
    """A real measurement between re-submissions resets the spam counter."""
    agent.session.coverage_measurements = 1
    agent.session.last_tests_passed = True
    agent.session.measurements_without_improvement = 0
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_measurements_without_improvement", 3)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_final_answers_without_measurement", 10)
    parsed = _validatable_coverage_result()
    agent.validate_output(state={}, parsed=parsed)
    assert agent.session.early_submission_rejections == 1
    assert agent.session.measurements_without_improvement == 0
    agent.session.coverage_measurements = 2
    agent.validate_output(state={}, parsed=parsed)
    assert agent.session.early_submission_rejections == 0
    assert agent.session.measurements_without_improvement == 0


def test_a_measurement_after_a_submission_on_the_start_measurement_is_progress(
        agent, monkeypatch):
    """A submission while the start's measurement still describes the suite,
    before any measurement of the model's own, then the model's first
    measurement: the next submission is progress and resets the counter."""
    agent.session.coverage_measurements = 0
    agent.session.prev_coverage_pct = 40.0
    agent.set_tests_changed(False)
    agent.session.last_tests_passed = True
    agent.session.measurements_without_improvement = 0
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_measurements_without_improvement", 3)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_final_answers_without_measurement", 10)
    parsed = _validatable_coverage_result()
    agent.validate_output(state={}, parsed=parsed)
    assert agent.session.early_submission_rejections == 1
    agent.session.coverage_measurements = 1
    agent.validate_output(state={}, parsed=parsed)
    assert agent.session.early_submission_rejections == 0


# ---------------------------------------------------------------------------
# State-marker injector
# ---------------------------------------------------------------------------


def test_state_marker_includes_tests_field_green_after_passing_run(agent):
    agent.session.last_tests_passed = True
    assert "tests=green" in agent.build_state_marker()


def test_state_marker_includes_tests_field_red_after_failing_run(agent):
    agent.session.last_tests_passed = False
    agent.session.last_run_tests_result = "FAILED test_x.py::test_y"
    assert "tests=red" in agent.build_state_marker()


def test_state_marker_includes_tests_field_untested_when_never_run(agent):
    assert "tests=untested" in agent.build_state_marker()


# ---------------------------------------------------------------------------
# Retry-prompt builder
# ---------------------------------------------------------------------------


def test_retry_prompt_includes_validator_message_on_validation_failure(agent):
    """The retry prompt names the validator's specific rejection."""
    builder = agent.retry_prompt_builder()
    exc = lg.OutputValidationError(
        "Your CoverageTestResult.explanation looks like placeholder text"
    )
    prompt = builder(exc, history=[])
    assert "placeholder text" in prompt
    assert "Continue where you left off" not in prompt


def test_retry_prompt_rejects_non_validation_errors(agent):
    """Transport failures cannot acquire model-visible recovery text."""
    builder = agent.retry_prompt_builder()
    with pytest.raises(TypeError, match="only for output validation"):
        builder(Exception("operation timed out"), history=[])


# ---------------------------------------------------------------------------
# Caller surface
# ---------------------------------------------------------------------------


def test_coverage_module_exposes_caller_surface():
    """The module exposes the surface the orchestrator calls."""
    for name in ("init_logging", "log_print", "save_logs",
                 "generate_coverage_tests"):
        assert callable(getattr(tc, name)), f"tc missing {name}"


# ---------------------------------------------------------------------------
# Graph topology + end-to-end with a scripted fake model
# ---------------------------------------------------------------------------


def test_graph_binds_instance_tools_and_isolates(tmp_path):
    """The bound tools belong to their own instance, and two agents do not
    interfere."""
    a = target_agent(tmp_path, stem="aa")
    b = target_agent(tmp_path, stem="bb")
    assert all(getattr(t, "__self__", None) is a for t in offered_tools(a))
    # A read through a's tool bumps a's read cache only.
    a.read_source()
    assert a.consecutive_reads("read_source") == 1
    assert b.consecutive_reads("read_source") == 0


def test_graph_raises_validation_error_when_validator_rejects(agent):
    """A final answer without the required state is rejected."""
    scripted = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "emit_coverage_test_result",
                    "args": {
                        "test_file_name": "test_target.py",
                        "explanation": (
                            "Wrote tests covering many cases including "
                            "the obvious branches."
                        ),
                        "targeted_functions": ["f"],
                    },
                    "id": "f1",
                },
            ],
        ),
    ]
    model = _ToolAwareFakeModel(responses=scripted)
    graph = agent.build_graph(model=model)
    with pytest.raises(lg.OutputValidationError):
        graph.invoke({"messages": [HumanMessage(content="x")]})


def test_graph_with_usage_tracker_raises_usage_limit_on_overflow(agent):
    """Past the request limit, the next model_decide raises UsageLimitExceeded
    without another model call."""
    scripted = [
        AIMessage(
            content="",
            tool_calls=[{"name": "read_source", "args": {}, "id": "c1"}],
        ),
        AIMessage(
            content="",
            tool_calls=[{"name": "read_source", "args": {}, "id": "c2"}],
        ),
    ]
    model = _ToolAwareFakeModel(responses=scripted)
    tracker = lg.UsageTracker(request_limit=1)
    graph = agent.build_graph(model=model, usage_tracker=tracker)
    with pytest.raises(lg.UsageLimitExceeded):
        graph.invoke({"messages": [HumanMessage(content="x")]})
    assert tracker.requests_used == 1


def test_log_run_summary_reports_branch_and_goal_metric(tmp_path, monkeypatch):
    # The summary shows line and branch coverage and names the goal metric.
    agent = target_agent(tmp_path)
    s = agent.session
    s.best_coverage_pct = 91.0
    s.best_branch_coverage_pct = 86.0
    s.prev_coverage_pct = 90.0
    s.prev_branch_coverage_pct = 85.0
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "branch")
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal", 90)
    lines: list = []
    monkeypatch.setattr(tc, "log_print", lambda msg, *a, **k: lines.append(str(msg)))
    agent._log_run_summary()
    blob = "\n".join(lines)
    assert "Best coverage seen: line 91.0% / branch 86.0%" in blob
    assert "Goal metric (branch): 86.0% vs goal 90% -> not met" in blob
    assert "Final coverage measurement: line 90.0% / branch 85.0%" in blob


def _answers_to(agent, calls):
    graph = agent.build_graph(
        model=_ToolAwareFakeModel(responses=[AIMessage(content="", tool_calls=calls)]),
        usage_tracker=lg.UsageTracker(request_limit=1),
    )
    results = []
    with pytest.raises(lg.UsageLimitExceeded):
        for update in graph.stream({"messages": [HumanMessage(content="x")]},
                                   stream_mode="updates"):
            for node in update.values():
                results += [m for m in (node or {}).get("messages", [])
                            if getattr(m, "type", "") == "tool"]
    return [m.content for m in results]


def test_a_gated_tool_call_is_told_what_makes_the_tool_available(agent):
    """An unbound edit_tests call is answered with the step that offers it
    again, not LangGraph's generic "not a valid tool"."""
    test_file = Path(agent.deps.test_output_dir) / "test_target.py"
    test_file.write_text("def test_a():\n    assert True\n", encoding="utf-8")
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = False
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_measure["coverage"] = True
    assert agent.resolve_agent_mode() == "needs_measurement"

    answers = _answers_to(agent, [{"name": "edit_tests", "id": "e0", "args": {
        "old_text": "    assert True\n", "new_text": "    assert 1\n"}}])

    assert answers == [
        "edit_tests is not offered now: the working file passed run_tests with "
        "changes not yet measured. Call measure_coverage first. edit_tests is "
        "offered again after the measurement, and run_tests after an "
        "edit_tests change."
    ]
    assert test_file.read_text(encoding="utf-8") == "def test_a():\n    assert True\n"


def test_a_gated_call_before_the_initial_read_names_the_read(tmp_path):
    answers = _answers_to(target_agent(tmp_path), [
        {"name": "read_source", "id": "r0", "args": {}}])

    assert answers == [
        "read_source is not offered yet: call read_tests first, then "
        "measure_coverage. The editing tools are offered after them."
    ]


@pytest.mark.parametrize("mode_state, tool, key", [
    ({"tests_changed_since_run_tests": False, "last_tests_passed": False},
     "run_tests", "tool_not_offered_run_tests"),
    ({"writes_done": 1, "tests_changed_since_run_tests": True},
     "measure_coverage", "tool_not_offered_measure_untested"),
    ({"tests_changed_since_run_tests": False, "last_tests_passed": False},
     "measure_coverage", "tool_not_offered_measure_red"),
    ({"tests_changed_since_run_tests": False, "last_tests_passed": True,
      "tests_changed_since_measure": {"coverage": False}},
     "measure_coverage", "tool_not_offered_measure_measured"),
])
def test_each_unbound_tool_answer_names_its_step(agent, mode_state, tool, key):
    for name, value in mode_state.items():
        setattr(agent.session, name, value)
    assert tool not in {t.__name__ for t in offered_tools(agent)}
    assert agent.unbound_tool_reply(tool) == agent.tool_msg(
        key, tool=tool, read="read_tests", edit="edit_tests",
        measure="measure_coverage")
    assert agent.unbound_tool_reply("no_such_tool") is None


def test_a_cached_re_read_does_not_supersede_the_read_it_points_to(agent):
    """The context rebuild keeps the read a cached re-read points to, since
    the cached header holds no content."""
    from langchain_core.messages import ToolMessage
    from testuary.agent import _context

    first = agent.read_source()
    cached = agent.read_source()
    assert cached.startswith("[Cached read #")
    msgs = [
        HumanMessage(content="task"),
        AIMessage(content="", tool_calls=[{"name": "read_source", "args": {}, "id": "r1"}]),
        ToolMessage(content=first, tool_call_id="r1", name="read_source"),
        AIMessage(content="", tool_calls=[{"name": "read_source", "args": {}, "id": "r2"}]),
        ToolMessage(content=cached, tool_call_id="r2", name="read_source"),
    ]
    slots = agent.dedup_slots()
    out = _context.dedup_file_payloads(
        msgs, read_tools={"read_source", "read_tests"},
        file_key=lambda name, args: slots.get(name, name), stub=_context.dedup_stub)

    assert out[2].content == first
    assert out[4].content == cached


def test_a_cached_re_read_behind_the_no_progress_banner_keeps_the_read_it_points_to(agent):
    """From the threshold of calls since the last edit on, an answer starts
    with the no-progress banner. A cached re-read behind it still holds no
    content, so the read it points to stays in the model's context."""
    from langchain_core.messages import ToolMessage
    from testuary.agent import _context

    first = agent.read_source()
    breaker = agent.session.loop_breaker
    breaker.calls_since_last_edit = breaker.no_progress_threshold - 1
    cached = agent.read_source()
    assert cached.startswith("[STUCK: ") and "[Cached read #" in cached
    msgs = [
        HumanMessage(content="task"),
        AIMessage(content="", tool_calls=[{"name": "read_source", "args": {}, "id": "r1"}]),
        ToolMessage(content=first, tool_call_id="r1", name="read_source"),
        AIMessage(content="", tool_calls=[{"name": "read_source", "args": {}, "id": "r2"}]),
        ToolMessage(content=cached, tool_call_id="r2", name="read_source"),
    ]
    slots = agent.dedup_slots()
    out = _context.dedup_file_payloads(
        msgs, read_tools={"read_source", "read_tests"},
        file_key=lambda name, args: slots.get(name, name), stub=_context.dedup_stub)

    assert out[2].content == first


def test_the_cached_read_texts_start_as_the_dedup_recognizes_them():
    from testuary.agent import _context

    for prompts in (python_prompts("test_coverage"), python_prompts("test_mutation")):
        cached = prompts["tool_messages"]["read_source_cached"]
        breaker = prompts["tool_messages"]["read_source_loop_breaker_prefix"]
        assert cached.startswith(_context.CACHED_READ_PREFIXES)
        assert breaker.startswith(_context.CACHED_READ_PREFIXES)


def test_a_cached_source_read_states_the_read_views_line_count(agent):
    """The cached header of a re-read counts lines as the read view does:
    a form feed stays inside its line."""
    agent.deps.target_file_content = "a = 1\n\fb = 2\n"
    assert agent._source_read(1, None).startswith("# target.py (2 lines)\n")
    assert "(2 lines, 13" in agent._source_read(1, None)


def test_the_tool_surface_record_fingerprints_the_final_answer_tool(agent, monkeypatch):
    """The chat log's tool fingerprint covers the final-answer tool, so a
    change of its name, description or schema changes it."""
    before = agent.tool_surface_record()["tool_descriptions_sha256"]
    monkeypatch.setattr(agent, "final_tool_name", "another_final_answer")
    assert agent.tool_surface_record()["tool_descriptions_sha256"] != before


def test_the_stop_explanation_states_the_best_coverage_as_the_loop_shows_it(agent):
    """The explanation of a designed stop states the best suite's coverage
    floored, as every other text of the loop does."""
    agent.session.best_coverage_pct = 88.96
    result = agent._synthesize_early_stop_result("measurements_without_improvement")
    assert "Final coverage: line 88.9%, no branches;" in result.explanation


def test_a_run_that_raises_leaves_the_edit_unrun(agent, monkeypatch):
    """A run that raises before it finished leaves the edit marked as not
    run, so the next run_tests runs it in place of the earlier result."""
    from testuary import normal_run

    def crashed(*args, **kwargs):
        raise RuntimeError("the runner crashed")

    agent.test_backend().target_test_path(
        agent.deps.test_output_dir, agent.deps.target_file_path,
    ).write_text("def test_a():\n    assert True\n", encoding="utf-8")
    monkeypatch.setattr(normal_run, "run", crashed)
    agent.session.tests_changed_since_run_tests = True
    agent.session.last_run_tests_result = "3 passed"
    with pytest.raises(RuntimeError):
        agent.run_tests()
    assert agent.session.tests_changed_since_run_tests is True


def test_the_framework_stop_asks_for_the_result_call_and_nothing_else(agent, monkeypatch):
    """The stop directive asks for the result call and no other tool."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "line")
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal", 90)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "extra_measurements_after_goal", 0)
    agent.session.best_coverage_pct = 95.0

    stop = agent._maybe_set_early_stop(source_tool="measure_coverage")

    assert stop == (
        "[FRAMEWORK STOP] Designed end condition reached. Call "
        "``emit_coverage_test_result`` now and no other tool. The framework "
        "has synthesised the canonical result from state and will override "
        "whatever you emit.")


def test_the_timeout_stop_text_asks_for_the_result_call():
    text = " ".join(python_prompts("test_coverage")["tool_messages"]["tests_timeout_abort"].split())
    assert "Call ``emit_coverage_test_result`` now and no other tool." in text
    assert "any more" not in text


def test_the_early_stop_rejection_states_the_goal_metric_with_both_values(agent, monkeypatch):
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "both")
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal", 90)
    agent.session.best_coverage_pct = 95.0
    agent.session.best_branch_coverage_pct = 70.0

    text = " ".join(agent.early_stop_reject_message().split())

    assert ("Best so far: line 95.0%, branch 70.0%; goal metric (lower of line "
            "and branch) 70.0%, goal 90%.") in text
    assert "uncovered branches" not in text


def test_the_regression_revert_floors_its_percentages(agent, monkeypatch):
    """Shown percentages are floored, so 89.96 below a best of 90.0 reads 89.9."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "restore_count", 1)
    monkeypatch.setattr(agent, "_maybe_restore_last_green", lambda *a, **k: True)
    agent.session.best_green.snapshot = {"content": "x", "test_count": 3}

    note = agent._maybe_revert_on_regression(
        agent.deps.target_file_path, True, current_goal=89.96, best_goal=90.0)

    assert "lowered the goal coverage to 89.9 (below the best 90.0)" in note


def test_a_file_back_at_the_snapshots_bytes_reports_the_snapshots_coverage(agent):
    """The model measured another suite, then edited the file back to the
    best-green snapshot's exact bytes and ran it green. At the phase's end
    the file stays, and the result's coverage is the snapshot's, not the
    other suite's."""
    import hashlib

    held = "def test_f():\n    assert f(1) == 2\n"
    other = held + "\ndef test_g():\n    assert f(2) == 3\n"
    target = Path(agent.deps.test_output_dir) / "test_target.py"
    target.write_text(held, encoding="utf-8")
    session = agent.session
    session.best_green.snapshot = {
        "content": held, "coverage": 90.0, "branch_coverage": 80.0, "test_count": 1,
        "loop_state": {"measured": True, "prev_coverage_pct": 90.0, "prev_branch_coverage_pct": 80.0,
                       "measured_suite_hash": hashlib.sha256(held.encode("utf-8")).hexdigest()}}
    # The measurement of the other suite, then the edit back and a green run.
    session.prev_coverage_pct, session.prev_branch_coverage_pct = 85.0, 70.0
    session.measured_suite_hash = hashlib.sha256(other.encode("utf-8")).hexdigest()
    agent.set_tests_changed(True)
    session.last_tests_passed, session.tests_changed_since_run_tests = True, False

    assert agent._settle_best_green(target) is False
    assert target.read_text(encoding="utf-8") == held
    assert (session.prev_coverage_pct, session.prev_branch_coverage_pct) == (90.0, 80.0)


def test_a_regression_revert_that_restores_nothing_sends_no_note(agent, monkeypatch):
    """The note tells the model its file went back to the best-green
    snapshot, so a revert that finds the file already holding the
    snapshot's bytes sends none, as the mutation phase does."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "restore_count", 1)
    held = "def test_f():\n    assert True\n"
    target = Path(agent.deps.test_output_dir) / "test_target.py"
    target.write_text(held, encoding="utf-8")
    agent.session.best_green.snapshot = {"content": held, "test_count": 1, "coverage": 90.0}

    assert agent._maybe_revert_on_regression(
        agent.deps.target_file_path, True, current_goal=80.0, best_goal=90.0) is None
    assert target.read_text(encoding="utf-8") == held


# The coverage phase opens on editing, as the mutation phase does: edit_tests
# is bound on the first request. An unmeasured baseline opens on editing too,
# recorded as unmeasured, and is measured after the first green edit.
def test_a_measured_baseline_opens_the_loop_on_editing(tmp_path):
    agent = target_agent(tmp_path)
    tc._seed_baseline(agent.session, {"content": "def test_f():\n    pass\n",
                                      "line_coverage": 50.0, "branch_coverage": 25.0})
    assert agent.resolve_agent_mode() == "needs_editing"


def test_a_baseline_without_measured_coverage_opens_on_editing_unmeasured(tmp_path):
    agent = target_agent(tmp_path)
    tc._seed_baseline(agent.session, {"content": "def test_f():\n    pass\n"})
    session = agent.session
    session.has_read = True
    session.initial_measured = True
    assert agent.resolve_agent_mode() == "needs_editing"
    # Green, never measured, with no measured bytes.
    assert session.last_tests_passed is True
    assert session.tests_changed_since_measure["coverage"] is True
    assert session.measured_suite_hash is None
    assert session.best_green.snapshot["loop_state"] == {"measured": False}
    assert agent.unbound_tool_reply("measure_coverage") == agent.tool_msg(
        "tool_not_offered_measure_entry", tool="measure_coverage",
        read="read_tests", edit="edit_tests", measure="measure_coverage")
    # The first edit, once green, is measured.
    session.writes_done = 1
    session.tests_changed_since_run_tests = True
    assert agent.resolve_agent_mode() == "needs_testing"
    session.tests_changed_since_run_tests = False
    assert agent.resolve_agent_mode() == "needs_measurement"


# A coverage phase whose limit ends it after a submission that passes under
# the instrument but fails as delivered. The run counts as red, so the phase
# restores the last suite that passed as delivered and reports the limit's
# end, not a success.
BASELINE = "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
#: Red in the normal run.
RED = "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 4\n"


def test_a_limit_stop_after_a_red_submission_delivers_the_last_green_suite(
    tmp_path, monkeypatch, in_run,
):
    project = tmp_path / "subject"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    target = project / "calc.py"
    target.write_text(TARGET, encoding="utf-8")
    output = tmp_path / "tests"
    output.mkdir()
    delivered = output / "test_calc.py"
    delivered.write_text(BASELINE, encoding="utf-8")
    replies = [
        AIMessage(content="", tool_calls=[{"name": "read_tests", "id": "r0", "args": {}}]),
        AIMessage(content="", tool_calls=[{"name": "measure_coverage", "id": "m0", "args": {}}]),
        AIMessage(content="", tool_calls=[{"name": "edit_tests", "id": "e0", "args": {
            "old_text": BASELINE, "new_text": RED}}]),
        AIMessage(content="", tool_calls=[{"name": "run_tests", "id": "t0", "args": {}}]),
    ]
    model = _ToolAwareFakeModel(responses=replies)
    monkeypatch.setattr(tc.CoverageAgent, "build_chat_model", lambda agent: model)
    seen = []
    run_tests = tc.CoverageAgent.run_tests

    @functools.wraps(run_tests)
    def observed_run_tests(agent, *args, **kwargs):
        out = run_tests(agent, *args, **kwargs)
        seen.append((agent.session.last_tests_passed, delivered.read_text(encoding="utf-8"), out))
        return out

    monkeypatch.setattr(tc.CoverageAgent, "run_tests", observed_run_tests)
    settings = RunSettings(base_dir=tmp_path, environment=dict(os.environ), sections={
        "api": {"model": "explicit"}, "main": {"hints_file": "", "seed": 37},
        "coverage": {"max_requests_per_phase": 4, "max_loop_attempts": 0},
    }, models={"explicit": {"provider": "openai", "model": "fixture",
                            "api_url": "https://example.test/v1", "api_key": "fixture"}})
    in_run.leave()
    with config.use_run_settings(settings):
        logs_in(monkeypatch, tmp_path / "logs")
        result = tc.generate_coverage_tests(
            "subject", {"file_path": str(target)}, str(output),
            baseline={"content": BASELINE, "line_coverage": 50.0, "branch_coverage": 50.0,
                      "test_count": 1, "suite_seconds": 0.01})
    assert len(seen) == 1
    passed, ran, out = seen[0]
    assert passed is False and ran == RED
    assert "test_add" in out
    # The last green suite is delivered.
    assert delivered.read_text(encoding="utf-8") == BASELINE
    assert result.status == "budget_exhausted"
    assert result.test_file_name == "test_calc.py"


IMPROVED = ("from calc import Box, add, make_box\n\n\ndef test_add():\n    assert add(1, 2) == 3\n\n\n"
            "def test_box():\n    assert isinstance(make_box(), Box)\n")


def _call(name, cid, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "id": cid, "args": args}])


def test_a_limit_stop_after_a_green_improvement_then_a_red_one_keeps_the_improvement(
    tmp_path, monkeypatch, in_run,
):
    """The last green suite is restored with its own coverage."""
    project = tmp_path / "subject"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    target = project / "calc.py"
    target.write_text(TARGET, encoding="utf-8")
    output = tmp_path / "tests"
    output.mkdir()
    delivered = output / "test_calc.py"
    delivered.write_text(BASELINE, encoding="utf-8")
    model = _ToolAwareFakeModel(responses=[
        # The read and the forced first measurement, then editing.
        _call("read_tests", "r0"), _call("measure_coverage", "m0"),
        _call("edit_tests", "e0", old_text=BASELINE, new_text=IMPROVED), _call("run_tests", "t0"),
        _call("measure_coverage", "m1"),
        _call("edit_tests", "e1", old_text=IMPROVED, new_text=RED), _call("run_tests", "t1"),
    ])
    monkeypatch.setattr(tc.CoverageAgent, "build_chat_model", lambda agent: model)
    seen = []
    run_tests, measure = tc.CoverageAgent.run_tests, tc.CoverageAgent.measure_coverage

    @functools.wraps(run_tests)
    def observed_run_tests(agent, *args, **kwargs):
        out = run_tests(agent, *args, **kwargs)
        seen.append(("run", agent.session.last_tests_passed, delivered.read_text(encoding="utf-8")))
        return out

    @functools.wraps(measure)
    def observed_measure(agent, *args, **kwargs):
        out = measure(agent, *args, **kwargs)
        seen.append(("measure", agent.session.prev_coverage_pct, delivered.read_text(encoding="utf-8")))
        return out

    monkeypatch.setattr(tc.CoverageAgent, "run_tests", observed_run_tests)
    monkeypatch.setattr(tc.CoverageAgent, "measure_coverage", observed_measure)
    settings = RunSettings(base_dir=tmp_path, environment=dict(os.environ), sections={
        "api": {"model": "explicit"}, "main": {"hints_file": "", "seed": 37},
        "coverage": {"max_requests_per_phase": 7, "max_loop_attempts": 0,
                          "goal": 100, "goal_metric": "line"},
    }, models={"explicit": {"provider": "openai", "model": "fixture",
                            "api_url": "https://example.test/v1", "api_key": "fixture"}})
    in_run.leave()
    with config.use_run_settings(settings):
        logs_in(monkeypatch, tmp_path / "logs")
        result = tc.generate_coverage_tests(
            "subject", {"file_path": str(target)}, str(output),
            baseline={"content": BASELINE, "line_coverage": 50.0, "branch_coverage": 50.0,
                      "test_count": 1, "suite_seconds": 0.01})
    assert [passed for kind, passed, _ in seen if kind == "run"] == [True, False]
    improved = [value for kind, value, text in seen if kind == "measure" and text == IMPROVED]
    assert delivered.read_text(encoding="utf-8") == IMPROVED
    assert result.status == "budget_exhausted"
    # 50.0 is the baseline's line coverage.
    assert improved and improved[-1] != 50.0
    assert result.line_coverage == improved[-1]
