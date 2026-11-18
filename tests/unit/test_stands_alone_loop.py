"""A green run that carries a STANDS ALONE note: the loops offer the edit
tool beside the measurement, and a goal end waits until a later green run
carries no note, within the loops' existing caps. Through the coverage
agent with the real Python backend, and the mutation agent's stop rule."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from test_agent_normal_runs import (
    HARMLESS_LEAK, ORDER_DEPENDENT, PLAIN, _Stop, _coverage_agent, _mutation_agent, _write)
from phase_values import set_phase_value
from testuary.agent import coverage_agent as tc
from testuary.agent import mutation_agent as tm

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _tool_names(agent, mode):
    return {tool.name for tool in agent.tool_mode_sets(*agent.phase_tools())[mode]}


@pytest.mark.parametrize("suite, noted", [(HARMLESS_LEAK, True), (PLAIN, False)])
def test_a_noted_green_run_offers_the_edit_tool_beside_the_measurement(
        tmp_path, suite, noted):
    """After the model's edit passes run_tests with a note, the next step
    offers the measurement and the edit tool; without a note only the
    measurement, as before. Both are the Assess step."""
    agent = _coverage_agent(tmp_path, PLAIN)
    agent.session.writes_done = 1
    _write(agent, suite)
    agent.run_tests()
    assert agent.session.last_tests_passed is True
    assert agent.session.stands_alone_noted is noted
    mode = agent.resolve_agent_mode()
    assert mode == ("needs_measurement_noted" if noted else "needs_measurement")
    names = _tool_names(agent, mode)
    assert "measure_coverage" in names and "run_tests" not in names
    assert (agent.edit_tool_name in names) is noted
    assert agent.MODE_STEPS[mode] == ("assess", "Measure")


def test_a_later_green_run_without_a_note_clears_it(tmp_path):
    agent = _coverage_agent(tmp_path, HARMLESS_LEAK)
    agent.run_tests()
    assert agent.session.stands_alone_noted is True
    _write(agent, PLAIN)
    agent.run_tests()
    assert agent.session.stands_alone_noted is False


def _no_stop_result(agent, monkeypatch, synthesize):
    monkeypatch.setattr(agent, "designed_stop", lambda result, reason: f"STOP {reason}")
    monkeypatch.setattr(agent, synthesize, lambda *args, **kwargs: None)


@pytest.mark.parametrize("noted", [True, False])
def test_the_coverage_goal_end_waits_while_the_latest_green_run_carries_a_note(
        tmp_path, monkeypatch, noted):
    agent = _coverage_agent(tmp_path, PLAIN)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal", 90)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "extra_measurements_after_goal", 0)
    monkeypatch.setattr(agent, "_best_goal_metric_value", lambda: 95.0)
    _no_stop_result(agent, monkeypatch, "_synthesize_early_stop_result")
    agent.session.stands_alone_noted = noted
    reply = agent._maybe_set_early_stop(source_tool="measure_coverage")
    if noted:
        assert reply.startswith("[GOAL REACHED, NOTE]")
        for name in ("edit_tests", "run_tests", "measure_coverage"):
            assert name in reply
        # The wait spends no post-goal measurement: the goal is not yet seen.
        assert agent.session.seen_coverage_goal is False
    else:
        assert reply == "STOP coverage_goal_met"


@pytest.mark.parametrize("noted", [True, False])
def test_the_mutation_goal_end_waits_while_the_latest_green_run_carries_a_note(
        tmp_path, monkeypatch, noted):
    agent = _mutation_agent(tmp_path, monkeypatch, [])
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "goal", 90)
    set_phase_value(monkeypatch, tm._AGENT_CONFIG, "extra_measurements_after_goal", 0)
    agent.session.best_mutation_score = 95.0
    agent.session.selective_switch_possible = False
    _no_stop_result(agent, monkeypatch, "_synthesize_mutation_result")
    agent.session.stands_alone_noted = noted
    reply = agent._maybe_set_mutation_stop(source_tool="measure_mutation")
    if noted:
        assert reply.startswith("[GOAL REACHED, NOTE]")
        for name in ("edit_tests", "run_tests", "measure_mutation"):
            assert name in reply
        assert agent.session.seen_mutation_goal is False
    else:
        assert reply == "STOP mutation_goal_met"


def test_a_reverse_run_that_could_not_run_names_no_test_and_holds_no_goal(
        tmp_path, monkeypatch):
    """The reverse-order note of a file that could not run in reverse order
    names no test the model could change: it is shown, and it does not hold
    the goal end."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "report_reverse_order_failures", True)
    agent = _coverage_agent(tmp_path, PLAIN)
    monkeypatch.setattr(agent, "_reverse_run", lambda backend, log_prefix: SimpleNamespace(
        reason="the runner could not reverse the file", failing=()))
    out = agent.run_tests()
    assert agent.session.last_tests_passed is True
    assert "could not run in reverse order" in out and "The suite still passes." in out
    assert agent.session.stands_alone_noted is False


def test_a_language_that_refuses_the_reverse_order_gets_no_reverse_run_and_no_note(
        tmp_path, monkeypatch):
    """A language whose runs as delivered refuse the reverse order, as
    Java's do, gets no reverse run and no order note, as one without runs
    as delivered: the note would come with every passing run and name
    nothing the model could change."""
    from langadapter.python.adapter import PythonAdapter

    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "report_reverse_order_failures", True)
    monkeypatch.setattr(PythonAdapter, "delivered_order_refusal",
                        lambda self, reverse: "the runner orders the tests itself" if reverse else None)
    agent = _coverage_agent(tmp_path, ORDER_DEPENDENT)
    backend_type = type(agent.test_backend())
    real = backend_type.run_as_delivered
    reverse_runs = []

    def counting(self, *args, **kwargs):
        reverse_runs.append(bool(kwargs.get("reverse")))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(backend_type, "run_as_delivered", counting)
    out = agent.run_tests()
    assert agent.session.last_tests_passed is True
    assert not any(reverse_runs)
    assert "[STANDS ALONE, ORDER]" not in out and "could not run in reverse order" not in out
    assert agent.session.stands_alone_noted is False


def test_the_goal_wait_reply_shows_the_note_it_waits_on(tmp_path, monkeypatch):
    """The measurement's prelude can run the edited file and keep only the
    measurement, so the reply that holds the goal end repeats the note."""
    agent = _coverage_agent(tmp_path, HARMLESS_LEAK)
    agent.run_tests()
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal", 90)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "extra_measurements_after_goal", 0)
    monkeypatch.setattr(agent, "_best_goal_metric_value", lambda: 95.0)
    _no_stop_result(agent, monkeypatch, "_synthesize_early_stop_result")
    reply = agent._maybe_set_early_stop(source_tool="measure_coverage")
    assert reply.startswith("[GOAL REACHED, NOTE]")
    assert "[STANDS ALONE, NOTE]" in reply and "test_reload" in reply


def test_the_mutation_measurement_of_a_changed_file_takes_its_own_note(
        tmp_path, monkeypatch):
    """An edit beside the measurement after a noted run: the measurement's
    normal run of the changed file gives the file's own notes, as run_tests
    does, so the goal end does not keep the earlier file's note."""
    agent = _mutation_agent(tmp_path, monkeypatch, [])
    agent.session.stands_alone_noted = True
    agent.session.stands_alone_note = "[STANDS ALONE, NOTE] the earlier file's note"
    agent.session.tests_changed_since_run_tests = True
    with pytest.raises(_Stop):
        agent._measure_mutation_engine()
    assert agent.session.stands_alone_noted is False
    assert agent.session.stands_alone_note == ""


@pytest.mark.parametrize("loop", ["coverage", "mutation"])
def test_a_restore_brings_back_the_snapshots_own_note(tmp_path, monkeypatch, loop):
    """A restored snapshot's bytes had no note: the restore clears the note
    of the discarded file, which would otherwise hold the goal end."""
    agent = (_coverage_agent(tmp_path, PLAIN) if loop == "coverage"
             else _mutation_agent(tmp_path, monkeypatch, []))
    state = agent._snapshot_loop_state(PLAIN)
    assert state["stands_alone_note"] == ""
    agent.session.stands_alone_note = "[STANDS ALONE, NOTE] the discarded file's note"
    agent.session.stands_alone_noted = True
    agent._apply_snapshot_state({"loop_state": state, "coverage": None,
                                 "branch_coverage": None, "mutation": None})
    assert agent.session.stands_alone_noted is False
    assert agent.session.stands_alone_note == ""
