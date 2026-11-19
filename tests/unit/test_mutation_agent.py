"""The LangGraph mutation agent: tool gating, validation, finalization, recovery."""
from __future__ import annotations

from typing import Any, Sequence

import pydantic
import pytest
from langchain_core.language_models.fake_chat_models import (
    FakeMessagesListChatModel,
)
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from agent_fixtures import coverage_agent, mutation_agent
from agent_prompts import python_prompts

from offered_tools import offered_tools
from testuary.agent import mutation_agent as tm
from phase_values import set_phase_value

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


class _ToolAwareFakeModel(FakeMessagesListChatModel):
    def bind_tools(  # type: ignore[override]
        self,
        tools: Sequence[Any],
        *,
        tool_choice: Any = None,
        **kwargs: Any,
    ) -> Runnable:
        return self


def test_measurement_identity_uses_executor_aware_target_hash(monkeypatch):
    seen = []
    monkeypatch.setattr(
        tm.project, "sha256_project_file",
        lambda path: seen.append(path) or "container-source-hash",
    )

    identity = tm.mutation_measurement_identity(
        target_file_path="/container/project/src.py",
        language="python",
        execution_name="test_src.py",
        config={"max_requests_per_phase": 250},
    )

    assert seen == ["/container/project/src.py"]
    assert identity["target_source_sha256"] == "container-source-hash"


def test_a_first_killer_matrix_never_stands_for_the_phase_end_measurement(monkeypatch):
    """The loop's engine records the first killing test and the phase end's
    records every killer. The identity carries this attribution, so a loop
    matrix is not reused as phase-end evidence."""
    from testuary import measurement

    monkeypatch.setattr(tm.project, "sha256_project_file", lambda _path: "h")
    config = {"max_requests_per_phase": 250}

    def identity(**kw):
        return tm.mutation_measurement_identity(
            target_file_path="src.py", language="python",
            execution_name="test_src.py", **kw)

    loop = identity(config=config)
    phase_end = identity(config=config, attribution=measurement.union_attribution(config))
    assert (loop["attribution"], phase_end["attribution"]) == ("first", "all")
    assert loop != phase_end
    first = {**config, "cover_attribution": "first"}
    assert identity(config=first, attribution=measurement.union_attribution(first)) == identity(
        config=first)


def test_measurement_identity_fails_closed_when_target_is_unreadable(monkeypatch):
    monkeypatch.setattr(tm.project, "sha256_project_file", lambda _path: None)

    with pytest.raises(OSError, match="cannot read target source"):
        tm.mutation_measurement_identity(
            target_file_path="/container/project/missing.py",
            language="python",
            execution_name="test_missing.py",
            config={"max_requests_per_phase": 250},
        )


def test_mutation_rejects_shadowed_preseed_before_framework_measurement(
        tmp_path, monkeypatch):
    target = tmp_path / "src.py"
    target.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    source = (
        "class TestF:\n"
        "    def test_f(self):\n        assert False\n\n"
        "    def test_f(self):\n        assert True\n"
    )

    monkeypatch.setattr(
        tm, "_seed_mutation_baseline_from_matrix", lambda _agent: False,
    )

    measured = []
    monkeypatch.setattr(
        tm.MutationAgent, "framework_baseline_measure",
        lambda self: measured.append(True),
    )
    with pytest.raises(RuntimeError, match="shadowed definitions"):
        tm.generate_mutation_tests(
            "project", str(target), str(test_dir),
            preseed={"content": source, "file_stem": "src"},
            coverage_baseline_pct=100.0,
        )
    assert measured == []


def _make_agent(tmp_path, stem: str = "src") -> "tm.MutationAgent":
    source = "def f(x): return x + 1\n"
    target = tmp_path / f"{stem}.py"
    target.write_text(source, encoding="utf-8")
    return mutation_agent(tmp_path, target_file_path=str(target), target_file_content=source,
                          mutation_test_file_name=f"test_{stem}_mutation.py",
                          coverage_baseline_pct=80.0)


@pytest.fixture
def agent(tmp_path):
    # Past the forced initial read, so the classifier tests see the working loop.
    a = _make_agent(tmp_path)
    a.session.has_read = True
    a.session.initial_measured = True  # past the start's measurement
    return a


def test_offered_tools_entry_is_needs_editing(agent):
    """After the initial read and the framework's baseline measurement the
    model starts editing: read and write tools bind, run_tests and
    measure_mutation do not."""
    agent.session.mutation_measurements = 1
    agent.session.tests_changed_since_mutation = False
    names = sorted(t.__name__ for t in offered_tools(agent))
    assert names == [
        "edit_file", "read_source", "read_tests",
    ]


def test_offered_tools_green_unmeasured_edits_is_needs_measurement(agent):
    """A green suite with edits since the last measurement binds the read
    tools and the measurement only."""
    agent.session.mutation_measurements = 1
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = False
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_mutation = True
    names = sorted(t.__name__ for t in offered_tools(agent))
    assert names == [
        "measure_mutation", "read_source", "read_tests",
    ]


# Forced initial read: the graph starts read-only so the source enters as a
# dedup-able read instead of prompt text.


def test_forces_initial_read_attrs_set():
    assert tm.MutationAgent.forces_initial_read is True
    assert tm.MutationAgent.initial_agent_mode == "needs_initial_read"


def test_entry_is_read_only_until_first_read(tmp_path):
    """Before the first read only the read tools bind, and only a read of the
    preceding phase's test suite satisfies the entry gate."""
    agent = _make_agent(tmp_path)
    agent.target_path().write_text(
        "def test_f():\n    assert True\n", encoding="utf-8",
    )
    assert agent.session.has_read is False
    assert agent.effective_agent_mode() == agent.INITIAL_READ_MODE
    names = {t.__name__ for t in offered_tools(agent)}
    assert names == {"read_tests"}


def test_first_read_flips_has_read_and_unlocks_measurement(tmp_path):
    agent = _make_agent(tmp_path)
    agent.target_path().write_text(
        "def test_f():\n    assert True\n", encoding="utf-8",
    )
    agent.read_tests()
    assert agent.session.has_read is True
    # The start's second step binds only the measurement. After it the loop
    # edits, and measurement binds after a green run with unmeasured changes.
    assert agent.effective_agent_mode() == "needs_initial_measurement"
    assert {t.__name__ for t in offered_tools(agent)} == {"measure_mutation"}
    agent.session.initial_measured = True
    assert agent.effective_agent_mode() == "needs_editing"
    names = {t.__name__ for t in offered_tools(agent)}
    assert "measure_mutation" not in names
    assert "edit_file" in names


def test_classify_routes_to_initial_read_until_first_read(tmp_path):
    agent = _make_agent(tmp_path)
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = True
    assert agent.classify_update("needs_editing") == {
        "agent_mode": "needs_initial_read"
    }


def test_offered_tools_needs_testing_after_edit(agent):
    """With edits pending after a write, read, write and run_tests bind and
    measure_mutation does not."""
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = True
    names = set(t.__name__ for t in offered_tools(agent))
    assert "run_tests" in names
    assert "measure_mutation" not in names
    assert {"edit_file",
            "read_source",
            "read_tests"} <= names


def test_offered_tools_needs_editing_after_measurement(agent):
    """Measured with no edits since: only the edit tools bind."""
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = False
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_mutation = False
    agent.session.mutation_measurements = 1
    names = set(t.__name__ for t in offered_tools(agent))
    assert "measure_mutation" not in names
    assert "run_tests" not in names
    assert {"edit_file"} <= names


def test_offered_tools_needs_editing_after_red_run(agent):
    """A red run with no edit since binds only the edit tools."""
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = False
    agent.session.last_tests_passed = False
    agent.session.mutation_measurements = 1
    names = set(t.__name__ for t in offered_tools(agent))
    assert "run_tests" not in names
    assert "measure_mutation" not in names
    assert {"edit_file"} <= names


def test_classify_run_advances_to_needs_testing_on_edit(agent):
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = True
    assert agent.classify_update("needs_measurement") == {"agent_mode": "needs_testing"}


def test_classify_run_advances_to_needs_measurement_on_green_with_edits(agent):
    agent.session.writes_done = 1
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_run_tests = False
    agent.session.tests_changed_since_mutation = True
    agent.session.mutation_measurements = 1
    assert agent.classify_update("needs_testing") == {"agent_mode": "needs_measurement"}


def test_classify_run_back_to_needs_editing_after_measurement(agent):
    agent.session.writes_done = 1
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_run_tests = False
    agent.session.tests_changed_since_mutation = False
    agent.session.mutation_measurements = 1
    assert agent.classify_update("needs_measurement") == {"agent_mode": "needs_editing"}


def test_classify_run_regresses_to_needs_editing_on_red(agent):
    agent.session.writes_done = 1
    agent.session.last_tests_passed = False
    agent.session.tests_changed_since_run_tests = False
    agent.session.mutation_measurements = 1
    assert agent.classify_update("needs_testing") == {"agent_mode": "needs_editing"}


def test_classify_run_idempotent_when_phase_unchanged(agent):
    agent.session.writes_done = 1
    agent.session.tests_changed_since_run_tests = True
    assert agent.classify_update("needs_testing") == {}


def test_classify_advances_out_of_entry_after_first_assess(agent):
    """A successful first measurement at entry increments
    mutation_measurements and moves from needs_measurement to needs_editing."""
    agent.session.writes_done = 0
    agent.session.measure_mutation_calls = 1
    agent.session.mutation_measurements = 1
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_mutation = False
    assert agent.classify_update("needs_measurement") == {"agent_mode": "needs_editing"}


def test_graph_terminates_with_structured_output(agent):
    # The validator accepts a voluntary return only after an end condition
    # such as mutation_goal_met.
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_mutation = False
    agent.session.mutation_measurements = 1
    agent.session.best_mutation_score = float(
        tm._AGENT_CONFIG.get("goal", 80)
    ) + 1.0
    scripted = [
        AIMessage(content="", tool_calls=[{
            "name": "emit_mutation_test_result",
            "args": {
                "mutation_test_file_name": "test_src_mutation.py",
                "explanation": "Comprehensive mutation suite covering all branches",
                "targeted_mutants": ["arithmetic operator AOR_1"],
            },
            "id": "f1",
        }]),
    ]
    model = _ToolAwareFakeModel(responses=scripted)
    graph = agent.build_graph(model=model)
    final = graph.invoke({"messages": [HumanMessage(content="test")]})
    assert final["result"] is not None
    assert final["result"].mutation_test_file_name == "test_src_mutation.py"


def test_graph_binds_instance_tools_and_isolates(tmp_path):
    """The bound-method tools belong to their own instance, and two agents
    with separate sessions and files do not interfere."""
    a = _make_agent(tmp_path, stem="aa")
    b = _make_agent(tmp_path, stem="bb")
    assert all(getattr(t, "__self__", None) is a for t in offered_tools(a))
    a.target_path().write_text(
        "def test_x():\n    assert True\n", encoding="utf-8",
    )
    a.edit_file("assert True", "assert 1 == 1")
    assert "assert 1 == 1" in a.target_path().read_text(encoding="utf-8")
    assert not b.target_path().exists()
    assert a.session.writes_done == 1
    assert b.session.writes_done == 0


def test_mutation_module_exposes_caller_surface():
    """The module exposes the surface the orchestrator calls."""
    for name in ("init_logging", "log_print", "save_logs",
                 "generate_mutation_tests", "set_mutation_session"):
        assert callable(getattr(tm, name)), f"tm missing {name}"


def test_validator_breaks_rejection_loop_on_early_stop(agent, monkeypatch):
    """After ``max_final_answers_without_measurement`` rejections without a
    new measurement, the validator synthesizes the ``final_answers_without_measurement``
    result and accepts, so a repeated MutationTestResult cannot loop until the
    recovery loop's retry cap."""
    set_phase_value(monkeypatch,
        tm._AGENT_CONFIG, "max_final_answers_without_measurement", 3,
    )
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 80)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "allow_early_stop", True)
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_mutation = False
    agent.session.best_mutation_score = 72.49
    agent.session.mutation_measurements = 19
    parsed = tm.MutationTestResult(
        mutation_test_file_name="test_src_mutation.py",
        explanation="model thinks it's done",
        targeted_mutants=["src.py:12:ReplaceBinaryOperator_Add_Sub"],
        mutation_score=72.49,
        status="success",
    )
    err1 = agent.validate_output(state={}, parsed=parsed)
    assert err1 is not None and "cannot return" in err1
    assert agent.session.early_submission_rejections == 1
    err2 = agent.validate_output(state={}, parsed=parsed)
    assert err2 is not None
    assert agent.session.early_submission_rejections == 2
    err3 = agent.validate_output(state={}, parsed=parsed)
    assert err3 is None, (
        f"validator should accept on the cap-th stuck rejection, got {err3!r}"
    )
    assert agent.session.early_stop_result is not None
    assert agent.session.early_stop_reason == "final_answers_without_measurement"
    assert agent.session.early_stop_result.status == "stagnation"
    assert agent.session.early_stop_result.mutation_score == pytest.approx(72.49)


def test_validator_rejection_counter_resets_on_real_progress(agent, monkeypatch):
    """A measure_mutation call between premature returns resets the
    rejection counter."""
    set_phase_value(monkeypatch,
        tm._AGENT_CONFIG, "max_final_answers_without_measurement", 3,
    )
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 80)
    agent.session.last_tests_passed = True
    agent.session.tests_changed_since_mutation = False
    agent.session.best_mutation_score = 60.0
    agent.session.mutation_measurements = 5
    parsed = tm.MutationTestResult(
        mutation_test_file_name="test_src_mutation.py",
        explanation="Targeted the surviving arithmetic-operator and boundary mutants",
        targeted_mutants=["src.py:12:ReplaceBinaryOperator_Add_Sub"],
        mutation_score=60.0,
        status="success",
    )
    agent.validate_output(state={}, parsed=parsed)
    agent.validate_output(state={}, parsed=parsed)
    assert agent.session.early_submission_rejections == 2
    agent.session.mutation_measurements = 6
    agent.validate_output(state={}, parsed=parsed)
    assert agent.session.early_submission_rejections == 0


def test_validate_result_fields_rejects_placeholder_explanation(agent):
    """Mutation rejects placeholder explanations through the shared content gate."""
    placeholder = tm.MutationTestResult(
        mutation_test_file_name="test_src_mutation.py",
        explanation="Brief explanation of which surviving mutants you targeted and how.",
        targeted_mutants=["src.py:7:ReplaceComparisonOperator_Lt_Gt"],
        mutation_score=60.0,
        status="success",
    )
    assert agent.validate_result_fields(placeholder) is not None
    empty = placeholder.model_copy(update={"explanation": "x"})
    assert agent.validate_result_fields(empty) is not None
    real = placeholder.model_copy(update={
        "explanation": "Targeted the surviving arithmetic and boundary mutants",
    })
    assert agent.validate_result_fields(real) is None


def test_goal_met_respects_a_pending_selective_switch(agent, monkeypatch):
    """A selective-proxy score at the goal switches to the full catalogue and
    does not satisfy the validator's goal predicate while the switch is
    still possible."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 80)
    agent.session.best_mutation_score = 95.0
    agent.session.selective_switch_possible = True
    assert agent.goal_met() is False
    agent.session.selective_switch_possible = False
    assert agent.goal_met() is True


def test_retry_prompt_builder_carries_the_recovery_state_block(agent):
    """The shared retry builder renders the mutation recovery block (working
    file, last run, best score, next steps) and the schema-specific
    rejection text, as for the coverage agent."""
    from testuary.agent import _langgraph as lg
    agent.session.best_mutation_score = 72.5
    agent.session.last_tests_passed = True
    builder = agent.retry_prompt_builder()
    rejected = builder(lg.OutputValidationError("targeted_mutants empty"), [])
    assert "[Recovery state snapshot" in rejected
    assert "Best mutation score:  72.5%" in rejected
    assert "return MutationTestResult" in rejected
    assert "targeted_mutants empty" in rejected
    assert "STALE VIEW" not in rejected
    assert "source is unchanged and already in your context" not in rejected
    assert "tool offered in the current state" in rejected
    with pytest.raises(TypeError, match="only for output validation"):
        builder(RuntimeError("boom"), [])


def test_after_run_tests_restores_best_green_after_repeated_red(
    agent, monkeypatch,
):
    """As in the coverage agent, after max_failed_test_attempts consecutive
    red runs the best-green snapshot returns to disk and the streak resets.
    The loop is then green, unchanged since the run and unmeasured, since
    the snapshot carries no measurement."""
    from types import SimpleNamespace
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "max_failed_test_attempts", 3)
    monkeypatch.setattr(agent, "_canonical_view", lambda res, w, rc=None: res.output)
    green = "def test_green():\n    assert True\n"
    agent.session.best_green.snapshot = {"content": green, "mutation": 70.0}
    agent.target_path().write_text(
        "def test_red():\n    assert False\n", encoding="utf-8",
    )
    red = SimpleNamespace(output="1 failed", returncode=1)
    out1 = agent.after_run_tests(red, 1, None)
    out2 = agent.after_run_tests(red, 1, None)
    assert "reverted" not in out1 and "reverted" not in out2
    out3 = agent.after_run_tests(red, 1, None)
    assert "reverted" in out3
    assert agent.target_path().read_text(encoding="utf-8") == green
    assert agent.session.last_tests_passed is True
    assert agent.session.tests_changed_since_run_tests is False
    assert agent.session.tests_changed_since_mutation is True
    assert agent.session.best_green.consecutive_failed_runs == 0


def test_after_run_tests_green_resets_the_red_streak(agent, monkeypatch):
    from types import SimpleNamespace
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "max_failed_test_attempts", 3)
    monkeypatch.setattr(agent, "_canonical_view", lambda res, w, rc=None: res.output)
    agent.session.best_green.snapshot = {"content": "x", "mutation": 70.0}
    red = SimpleNamespace(output="1 failed", returncode=1)
    ok = SimpleNamespace(output="3 passed", returncode=0)
    agent.after_run_tests(red, 1, None)
    agent.after_run_tests(red, 1, None)
    agent.after_run_tests(ok, 0, None)
    assert agent.session.best_green.consecutive_failed_runs == 0
    out = agent.after_run_tests(red, 1, None)
    assert "reverted" not in out


def test_stale_measurement_rejected_at_acceptance_not_before(
    agent, monkeypatch,
):
    """The staleness gate sits at voluntary acceptance in both agents. Below
    the goal a stale submission gets the ordinary stagnation rejection, at the
    goal the stale rejection until re-measured."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 80)
    agent.session.last_tests_passed = True
    agent.session.mutation_measurements = 3
    agent.session.tests_changed_since_mutation = True
    parsed = tm.MutationTestResult(
        mutation_test_file_name="test_src_mutation.py",
        explanation="Targeted the surviving arithmetic and boundary mutants",
        targeted_mutants=["src.py:12:ReplaceBinaryOperator_Add_Sub"],
        mutation_score=50.0,
        status="success",
    )
    agent.session.best_mutation_score = 50.0
    err_below = agent.validate_output(state={}, parsed=parsed)
    assert err_below is not None and "predates" not in err_below
    agent.session.best_mutation_score = 90.0
    err_at_goal = agent.validate_output(state={}, parsed=parsed)
    assert err_at_goal is not None and "predates" in err_at_goal
    agent.session.tests_changed_since_mutation = False
    assert agent.validate_output(state={}, parsed=parsed) is None


def test_targeted_mutants_is_a_required_field():
    """Like coverage's targeted_functions, a result without the key does not
    parse. Synthesized results pass an empty list and bypass the validator."""
    with pytest.raises(pydantic.ValidationError):
        tm.MutationTestResult(
            mutation_test_file_name="test_src_mutation.py",
            explanation="Targeted the surviving mutants",
            mutation_score=60.0,
        )


def test_validate_result_fields_rejects_empty_targeted_mutants(agent):
    """A voluntary result must name the mutants it targeted, as coverage
    requires for targeted_functions."""
    empty = tm.MutationTestResult(
        mutation_test_file_name="test_src_mutation.py",
        explanation="Targeted the surviving arithmetic and boundary mutants",
        targeted_mutants=[],
        mutation_score=60.0,
        status="success",
    )
    err = agent.validate_result_fields(empty)
    assert err is not None and "targeted_mutants" in err


def test_validate_result_fields_rejects_placeholder_mutant_entries(agent):
    """An entry echoing the schema description is a placeholder, not a
    mutant identifier."""
    echoed = tm.MutationTestResult(
        mutation_test_file_name="test_src_mutation.py",
        explanation="Targeted the surviving arithmetic and boundary mutants",
        targeted_mutants=["file:line:operator"],
        mutation_score=60.0,
        status="success",
    )
    err = agent.validate_result_fields(echoed)
    assert err is not None and "placeholder" in err
    real = echoed.model_copy(update={
        "targeted_mutants": ["combine.py:214:ReplaceBinaryOperator_Add_Sub"],
    })
    assert agent.validate_result_fields(real) is None


def test_validator_requires_a_mutation_measurement(agent, monkeypatch):
    """Without a mutation measurement on record the return is refused however
    often it is submitted. The submission cap does not release a run that
    never produced a score."""
    set_phase_value(monkeypatch,
        tm._AGENT_CONFIG, "max_final_answers_without_measurement", 3,
    )
    agent.session.last_tests_passed = True
    agent.session.mutation_measurements = 0
    parsed = tm.MutationTestResult(
        mutation_test_file_name="test_src_mutation.py",
        explanation="no measurement done yet",
        targeted_mutants=[],
        mutation_score=0.0,
        status="success",
    )
    for _ in range(10):
        assert agent.validate_output(state={}, parsed=parsed) is not None
    assert agent.session.early_stop_result is None
    assert agent.session.early_stop_result is None


def test_synthesize_mutation_result_signature():
    """The entry point calls it with a single positional ``reason``. The
    dependencies come from ``self.deps``."""
    import inspect
    sig = inspect.signature(tm.MutationAgent._synthesize_mutation_result)
    params = list(sig.parameters)
    assert params[:2] == ["self", "reason"]
    assert "deps" not in params
    assert "test_output_dir" not in params
    assert "target_file_path" not in params


# The message the recovery loop returns for an out-of-credit 402: non-retryable,
# so it comes back as an error string rather than being retried.
_BALANCE_402 = (
    "non-retryable error: PermissionDeniedError: Error code: 402 - "
    "{'error': {'message': 'Insufficient Balance'}}"
)


class _ZeroUsage:
    requests_used = 0
    input_tokens_used = 0
    output_tokens_used = 0
    cache_hit_tokens_used = 0


class _NonzeroRequestUsage(_ZeroUsage):
    requests_used = 9


def test_build_failure_result_labels_402_agent_error_not_stagnation(agent):
    """A 402 reaches build_failure_result (what run_phase calls on the graph's
    (None, error) return) as a non-retryable error. It reads agent_error, as in
    coverage, not the stagnation status of a plateau, and carries the message
    in the error field."""
    out = agent.build_failure_result(_BALANCE_402)
    assert out.status == "agent_error"
    assert out.error == _BALANCE_402


def test_build_failure_result_preserves_the_limit_and_context(agent):
    """Only the stagnation catch-all is relabeled. A run that ended at a
    limit reads budget_exhausted, a context overflow context_overflow, and a
    failure that only quotes a limit's words is an agent error."""
    message = "Budget exhausted: UsageLimitExceeded: max_requests_per_phase (250) exceeded"
    assert agent.build_failure_result(message).status == "agent_error"
    agent._limit_reached = True  # as run_phase sets it at a limit's end
    failed = agent.build_failure_result(message)
    assert failed.status == "budget_exhausted"
    agent._limit_reached = False
    ctx = agent.build_failure_result(
        "non-retryable error: BadRequestError: maximum context length exceeded")
    assert ctx.status == "context_overflow"


def test_synthesize_designed_stops_keep_their_labels(agent):
    """Only the error path is relabeled: a stagnation stop reads stagnation
    and a goal-met stop reads success."""
    assert agent._synthesize_mutation_result(
        "mutation_stagnation").status == "stagnation"
    assert agent._synthesize_mutation_result(
        "mutation_goal_met").status == "success"


def test_finalize_result_preserves_the_agent_error_status(agent, monkeypatch):
    """run_phase passes build_failure_result's output through finalize_result,
    which keeps the agent_error status."""
    monkeypatch.setattr(agent, "_settle_best_green", lambda *_args, **_kwargs: False)
    failed = agent.build_failure_result(_BALANCE_402)
    final = agent.finalize_result(
        failed, error_msg=_BALANCE_402, duration_ms=0.0,
        usage_tracker=_ZeroUsage(), final_messages=[],
    )
    assert final.status == "agent_error"


def test_finalize_records_a_designed_stop_reason_as_the_error(agent, monkeypatch):
    """A stagnation or limit stop carries its reason in the error field, as
    in the coverage phase, so tests.csv's error column names the mutation
    phase's endings. The goal-met stop is a success and has none."""
    monkeypatch.setattr(agent, "_settle_best_green", lambda *_args, **_kwargs: False)

    def _final(reason):
        agent.session.early_stop_result = agent._synthesize_mutation_result(reason)
        agent.session.early_stop_reason = reason
        return agent.finalize_result(
            agent.session.early_stop_result, error_msg=None, duration_ms=0.0,
            usage_tracker=_ZeroUsage(), final_messages=[],
        )

    stagnation = _final("mutation_stagnation")
    assert (stagnation.status, stagnation.error) == ("stagnation", "mutation_stagnation")
    assert not _final("mutation_goal_met").error


def test_finalize_derives_status_on_a_null_status_result(agent, monkeypatch):
    """A voluntary result never keeps status=None: with a measurement on
    record it reads success, without one tests_failed, and an error path
    reads its classified status. A synthesized stop status is kept."""
    monkeypatch.setattr(agent, "_settle_best_green", lambda *_args, **_kwargs: False)

    def _final(output, error=None):
        return agent.finalize_result(
            output, error_msg=error, duration_ms=100.0,
            usage_tracker=_ZeroUsage(), final_messages=[],
        )

    voluntary = tm.MutationTestResult(
        mutation_test_file_name="test_x_mutation.py",
        explanation="killed the boundary mutants",
        targeted_mutants=["m.py:1:AOR"],
    )
    agent.session.best_mutation_score = 81.0
    assert _final(voluntary).status == "success"
    agent.session.best_mutation_score = None
    assert _final(voluntary).status == "tests_failed"
    assert _final(voluntary, error="a provider error").status == "agent_error"
    agent._limit_reached = True  # as run_phase sets it at a limit's end
    assert _final(voluntary, error="budget end").status == "budget_exhausted"
    agent._limit_reached = False
    # run_phase substitutes a synthesized stop and sets the early-stop signal.
    synthesized = agent._synthesize_mutation_result("mutation_stagnation")
    agent.session.early_stop_result = synthesized
    assert _final(synthesized).status == "stagnation"
    agent.session.early_stop_result = None


def test_finalize_stamps_the_delivered_file_counts(agent, monkeypatch):
    """test_count_mutation and test_loc_mutation come from the delivered
    on-disk file at finalization."""
    monkeypatch.setattr(agent, "_settle_best_green", lambda *_args, **_kwargs: False)
    agent.target_path().write_text(
        "def test_a():\n    assert True\n\n"
        "def test_b():\n    assert True\n",
        encoding="utf-8",
    )
    out = agent.finalize_result(
        tm.MutationTestResult(
            mutation_test_file_name="test_x_mutation.py",
            explanation="x", targeted_mutants=["m.py:1:AOR"],
        ),
        error_msg=None, duration_ms=100.0,
        usage_tracker=_ZeroUsage(), final_messages=[],
    )
    assert out.test_count_mutation == 2
    assert out.test_loc_mutation == 4


def test_finalize_overwrites_model_filled_auto_fields(agent, monkeypatch):
    """Every [auto] field is overwritten from framework state, so a value the
    model filled (a wrong filename, an invented score, a status) never reaches
    the CSV row or the consolidation's file lookup."""
    monkeypatch.setattr(agent, "_settle_best_green", lambda *_args, **_kwargs: False)
    agent.session.best_mutation_score = 62.5
    lying = tm.MutationTestResult(
        mutation_test_file_name="test_wrong_name.py",
        explanation="claims a perfect run",
        targeted_mutants=["m.py:1:AOR"],
        mutation_score=99.9, status="success",
        agent_iterations=1, test_count_mutation=999,
    )
    out = agent.finalize_result(
        lying, error_msg=None, duration_ms=100.0,
        usage_tracker=_NonzeroRequestUsage(), final_messages=[],
    )
    assert out.mutation_score == 62.5
    assert out.mutation_test_file_name == agent.deps.mutation_test_file_name
    assert out.test_count_mutation is None  # no delivered file on disk
    assert out.agent_iterations == 0
    assert out.model_requests == 9
    assert out.model_dump()["model_requests"] == 9
    assert out.status == "success"


def test_finalize_result_stamps_the_phase_clock_duration(agent, monkeypatch):
    """With the phase clock armed (the entry point arms it before baseline
    preparation), the stamped duration spans from the arm point. An unarmed
    run keeps the passed model-loop span."""
    import time as _time

    monkeypatch.setattr(agent, "_settle_best_green", lambda *_args, **_kwargs: False)
    out = tm.MutationTestResult(
        mutation_test_file_name="test_x_mutation.py",
        explanation="x", targeted_mutants=["m:1:AOR"],
    )
    unarmed = agent.finalize_result(
        out, error_msg=None, duration_ms=1500.0,
        usage_tracker=_ZeroUsage(), final_messages=[],
    )
    assert unarmed.agent_duration_s == pytest.approx(1.5)
    agent.session.phase_clock_start = _time.perf_counter() - 10.0
    armed = agent.finalize_result(
        out, error_msg=None, duration_ms=1500.0,
        usage_tracker=_ZeroUsage(), final_messages=[],
    )
    assert armed.agent_duration_s >= 9.9


def test_non_retryable_402_recovery_returns_error_then_agent_error(agent):
    """run_graph_with_retries returns a 402 as an error without retrying, and
    build_failure_result, which run_phase calls on that (None, error) return,
    labels it agent_error rather than stagnation."""
    from testuary.agent import _langgraph as lg

    calls = {"n": 0}

    class _Balance402(Exception):
        def __init__(self):
            super().__init__("Error code: 402 - Insufficient Balance")
            self.status_code = 402

    class _RaisingGraph:
        def stream(self, *a, **k):
            calls["n"] += 1
            raise _Balance402()

    final_state, err = lg.run_graph_with_retries(
        _RaisingGraph(), {"messages": [HumanMessage(content="go")]},
        max_retries=5, log=lambda *_: None,
    )
    assert final_state is None
    assert calls["n"] == 1                     # non-retryable: not retried to 5
    assert err is not None and "402" in err
    assert agent.build_failure_result(err).status == "agent_error"


def test_classify_error_status_is_the_shared_single_source():
    """Coverage and mutation classify a terminal error through one
    staticmethod on the base, so both phases label errors alike."""
    from testuary.agent.coverage_agent import CoverageAgent
    assert (
        CoverageAgent.classify_error_status
        is tm.MutationAgent.classify_error_status
    )
    cls = tm.MutationAgent.classify_error_status
    assert cls("Error code: 402 - Insufficient Balance") == "agent_error"
    assert cls("maximum context length exceeded") == "context_overflow"
    # The run's end decides a limit, not the text: a failure that quotes the
    # limit's setting or exception is an agent error.
    assert cls("a tool failed: max_requests_per_phase, UsageLimitExceeded") == "agent_error"
    assert cls("Budget exhausted: UsageLimitExceeded", limit_reached=True) == "budget_exhausted"


def test_recovery_snapshot_counts_lines_as_read_tests_and_shows_the_run_summary(agent):
    """The mutation recovery block counts the working file's lines as
    read_tests does and shows a noted run result by its TESTS summary."""
    agent.target_path().write_text("def test_a():\n    assert True\n", encoding="utf-8")
    agent.session.last_run_tests_result = (
        "[STANDS ALONE, NOTE] The file passes on its own, but some tests leave\n"
        "shared state changed:\n\nTESTS: PASSED | collected 2 | passed 2 | failed 0 | "
        "errors 0 | elapsed 0.10s")

    block = agent.recovery_state_block()

    assert f"  Working file on disk: {agent.target_path().name} (2 lines, " in block
    assert ("  Last run_tests:    TESTS: PASSED | collected 2 | passed 2 | failed 0 | "
            "errors 0 | elapsed 0.10s\n") in block


def test_recovery_snapshot_before_any_run_tests_names_the_framework_run(agent):
    """Before the model's first run_tests the block says so and names the
    framework's run at the loop's start, whose status it shows."""
    block = agent.recovery_state_block()

    assert ("  Last run_tests:    (not called yet; the framework ran the suite at the "
            "loop's start)\n") in block
    assert "  Last test status:  GREEN at the loop's start (the framework's measurement)\n" in block
    # After a write the suite on disk is not yet run.
    agent.session.writes_done = 1
    assert ("  Last test status:  RED or unknown (tests failing or not yet run)\n"
            in agent.recovery_state_block())


def test_the_framework_stop_asks_for_the_result_call_and_nothing_else(agent, monkeypatch):
    """The stop directive names the result tool in the coverage phase's
    words."""
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 80)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "extra_measurements_after_goal", 0)
    agent.session.best_mutation_score = 90.0
    agent.session.selective_switch_possible = False

    stop = agent._maybe_set_mutation_stop(source_tool="measure_mutation")

    assert stop == (
        "[FRAMEWORK STOP] Designed end condition reached. Call "
        "``emit_mutation_test_result`` now and no other tool. The framework "
        "has synthesised the canonical result from state and will override "
        "whatever you emit.")


@pytest.mark.parametrize("key", [
    "tests_timeout_abort", "measure_mutation_inconclusive", "tests_timeout_inherited"])
def test_every_stop_text_asks_for_the_result_call(key):
    text = python_prompts("test_mutation")["tool_messages"][key]
    text = text if isinstance(text, str) else "\n".join(text)
    assert "Call ``emit_mutation_test_result`` now and no other tool." in " ".join(text.split())
    assert "any more" not in text


def test_the_early_stop_rejection_states_the_score_and_the_goal(agent, monkeypatch):
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 80)
    agent.session.best_mutation_score = 50.06

    text = " ".join(agent.early_stop_reject_message().split())

    assert "Best so far: mutation score 50.0%, goal 80%." in text


_FAILING_RUN = (
    "FAILURES\n"
    "FAILED tests/t.py::test_x - AssertionError: x\n"
    "============= 1 failed in 0.5s ============="
)


def test_a_red_run_ends_with_the_cycle_footer_in_both_loops(agent, tmp_path):
    """A red run_tests ends with the same footer in both loops: the count,
    the fix with edit_tests, and the next steps of the cycle (an edit, then
    run_tests, the measurement only after a passing run). No text claims
    the measurement runs the tests first."""
    coverage = tc_agent(tmp_path).compact_output(_FAILING_RUN, 1)
    mutation = agent.compact_output(_FAILING_RUN, 1)
    tail = ("The next step is an ``edit_tests`` change, then ``run_tests``. "
            "``{m}`` is offered only after a passing run.")
    assert coverage.endswith(tail.format(m="measure_coverage"))
    assert mutation.endswith(tail.format(m="measure_mutation"))
    assert "-> tests are RED. 1 failing test." in mutation
    for text in (coverage, mutation):
        assert "prelude" not in text and "auto-runs" not in text
    assert coverage.replace("measure_coverage", "measure_mutation") == mutation


_UTEST_RED_RUN = (
    "[==========] Running 1 test cases.\n"
    "[ RUN      ] sign.negative\n"
    "[       OK ] sign.negative (361ns)\n"
    "[==========] 1 test cases ran.\n"
    "[  PASSED  ] 1 tests.\n"
    "[==========] Running 1 test cases.\n"
    "[ RUN      ] sign.wrong_on_purpose\n"
    "suite/test_calc.c:9: Failure\n"
    "  Expected : (1) == (sign(0))\n"
    "    Actual : 1 vs 0\n"
    "[  FAILED  ] sign.wrong_on_purpose (5599ns)\n"
    "[==========] 1 test cases ran.\n"
    "[  PASSED  ] 0 tests.\n"
    "[  FAILED  ] 1 tests, listed below:\n"
    "[  FAILED  ] sign.wrong_on_purpose\n"
    "[==========] Running 1 test cases.\n"
    "[ RUN      ] sign.also_wrong\n"
    "suite/test_calc.c:13: Failure\n"
    "  Expected : (5) == (sign(4))\n"
    "    Actual : 5 vs 1\n"
    "[  FAILED  ] sign.also_wrong (6131ns)\n"
    "[==========] 1 test cases ran.\n"
    "[  PASSED  ] 0 tests.\n"
    "[  FAILED  ] 1 tests, listed below:\n"
    "[  FAILED  ] sign.also_wrong\n"
)


def test_a_red_run_without_a_pytest_or_junit_count_counts_its_reported_failures(agent):
    """A runner whose output has no pytest or JUnit count, such as C's utest,
    gets the failing tests of the run's own report counted in the footer, as
    the failure block above lists them, never "No test failed"."""
    from types import SimpleNamespace

    failures = [SimpleNamespace(test=name, outcome="failed", message="Failure", trace="")
                for name in ("sign.wrong_on_purpose", "sign.also_wrong")]
    text = agent.compact_output(_UTEST_RED_RUN, 1, failures=failures)
    assert "Failing tests (2):" in text
    assert "-> tests are RED. 2 failing tests." in text
    assert "No test failed" not in text


def tc_agent(tmp_path):
    target = tmp_path / "cov_target.py"
    target.write_text("def f(x):\n    return x\n", encoding="utf-8")
    return coverage_agent(tmp_path, target_file_path=str(target),
                          target_file_content="def f(x):\n    return x\n")


def test_the_mutation_loop_opens_on_editing_without_a_baseline_measurement(tmp_path):
    target = tmp_path / "target.py"
    target.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    agent = mutation_agent(tmp_path, target_file_path=str(target),
                           target_file_content=target.read_text(encoding="utf-8"),
                           mutation_test_file_name="test_target.py")
    session = agent.session
    session.has_read = True
    session.initial_measured = True
    session.last_tests_passed = True
    session.tests_changed_since_run_tests = False
    assert session.tests_changed_since_mutation is True
    assert agent.resolve_agent_mode() == "needs_editing"
    assert agent.unbound_tool_reply("measure_mutation") == agent.tool_msg(
        "tool_not_offered_measure_entry", tool="measure_mutation",
        read="read_tests", edit="edit_tests", measure="measure_mutation")
