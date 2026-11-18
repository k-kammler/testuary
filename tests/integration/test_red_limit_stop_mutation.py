"""The mutation phase at its limit's end after a red submission.

A real Cosmic Ray run: the loop improves the suite, measures it, then
replaces it with one whose test fails its normal run and reaches its
request limit. The phase delivers the last green suite with that suite's
own measured score, which differs from the starting suite's, and the red
suite is never measured.

Marked ``slow``, like the other real Cosmic Ray runs.
"""
from __future__ import annotations

import functools
import os

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage

from tool_fixtures import cosmic_ray_installed

from testuary import config
from testuary.agent import mutation_agent as tm
from testuary.settings import RunSettings

pytestmark = [pytest.mark.slow, pytest.mark.mutation_tool]

TARGET = ('class Box:\n    pass\n\n\n'
          'def make_box():\n    return Box()\n\n\n'
          'def add(a, b):\n    return a + b\n\n\n'
          'def unused(x):\n    if x:\n        return 1\n    return 2\n')
BASELINE = "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"
IMPROVED = ("from calc import Box, add, make_box, unused\n\n\n"
            "def test_add():\n    assert add(1, 2) == 3\n\n\n"
            "def test_box():\n    assert isinstance(make_box(), Box)\n\n\n"
            "def test_unused():\n    assert unused(1) == 1\n    assert unused(0) == 2\n")
#: Red in the normal run.
RED = "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 4\n"


class _ToolAwareFakeModel(FakeMessagesListChatModel):
    """A scripted model that accepts ``bind_tools``."""

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


def _call(name, cid, **args):
    return AIMessage(content="", tool_calls=[{"name": name, "id": cid, "args": args}])


def _name(text):
    return {BASELINE: "BASELINE", IMPROVED: "IMPROVED", RED: "RED"}.get(text, "?")


def test_a_mutation_limit_stop_after_a_red_submission_keeps_the_last_green_suite(
    tmp_path, monkeypatch,
):
    if not cosmic_ray_installed():
        pytest.skip("cosmic-ray not installed")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setenv("PYTHONUTF8", "1")
    project = tmp_path / "subject"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    target = project / "calc.py"
    target.write_text(TARGET, encoding="utf-8")
    output = tmp_path / "tests"
    output.mkdir()
    measured = []
    real_measure = tm.MutationAgent._measure_mutation_engine

    def observed_measure(agent, framework_prep=False):
        out = real_measure(agent, framework_prep=framework_prep)
        measured.append(("framework" if framework_prep else "agent", agent.session.prev_mutation_score,
                         _name(agent.target_path().read_text(encoding="utf-8"))))
        return out

    monkeypatch.setattr(tm.MutationAgent, "_measure_mutation_engine", observed_measure)
    monkeypatch.setattr(tm, "_seed_mutation_baseline_from_matrix", lambda agent: False)
    model = _ToolAwareFakeModel(responses=[
        _call("read_tests", "r0"),
        # The loop's start measures the inherited suite before any edit.
        _call("measure_mutation", "m0"),
        _call("edit_tests", "e0", old_text=BASELINE, new_text=IMPROVED),
        _call("run_tests", "t0"),
        _call("measure_mutation", "m1"),
        _call("edit_tests", "e1", old_text=IMPROVED, new_text=RED),
        _call("run_tests", "t1"),
    ])
    monkeypatch.setattr(tm.MutationAgent, "build_chat_model", lambda agent: model)
    runs = []
    run_tests = tm.MutationAgent.run_tests

    @functools.wraps(run_tests)
    def observed_run_tests(agent, *args, **kwargs):
        out = run_tests(agent, *args, **kwargs)
        runs.append((agent.session.last_tests_passed, _name(agent.target_path().read_text(encoding="utf-8"))))
        return out

    monkeypatch.setattr(tm.MutationAgent, "run_tests", observed_run_tests)
    agents = []
    real_init = tm.MutationAgent.__init__

    def init(agent, *a, **k):
        real_init(agent, *a, **k)
        agents.append(agent)

    monkeypatch.setattr(tm.MutationAgent, "__init__", init)
    settings = RunSettings(base_dir=tmp_path, environment=dict(os.environ), sections={
        "api": {"model": "explicit"}, "main": {"hints_file": "", "seed": 37},
        "mutation": {"max_requests_per_phase": 7, "max_loop_attempts": 0,
                          "goal": 100, "cosmic_ray_clone_sweep_hours": 0},
    }, models={"explicit": {"provider": "openai", "model": "fixture",
                            "api_url": "https://example.test/v1", "api_key": "fixture"}})
    with config.use_run_settings(settings):
        tm.init_logging("subject", str(tmp_path / "logs"), "mutation.jsonl")
        result = tm.generate_mutation_tests(
            "subject", str(target), str(output),
            preseed={"content": BASELINE, "file_stem": "calc"}, coverage_baseline_pct=80.0)
    assert runs == [(True, "IMPROVED"), (False, "RED")]
    assert agents[-1].target_path().read_text(encoding="utf-8") == IMPROVED
    assert result.status == "budget_exhausted"
    assert not any(name == "RED" for _, _, name in measured)
    baseline = [score for kind, score, name in measured if kind == "framework" and name == "BASELINE"]
    improved = [score for kind, score, name in measured if kind == "agent" and name == "IMPROVED"]
    assert baseline and improved and baseline[-1] != improved[-1]
    assert result.mutation_score == improved[-1]
