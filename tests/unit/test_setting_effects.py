"""Settings that each do what their config.ini comment says."""
from __future__ import annotations

import argparse
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage

from testuary.agent import _context
from testuary.agent import _langgraph as lg
from testuary import settings as settings_module
from testuary import config
from langadapter.outcomes import Outcome
from langadapter.python.adapter import PythonAdapter
from testuary.languages import mutation
from testuary import measurement
from testuary.settings import RunSettings


def _turns(start, count):
    return [HumanMessage(content=f"question {i}") if i % 2 == 0 else AIMessage(content=f"answer {i}")
            for i in range(start, start + count)]


@pytest.mark.parametrize("trigger, expected", [(0, 6), (1, 2)])
def test_context_rebuild_trigger_tokens_rebuilds_within_one_tool_set(trigger, expected):
    """[main] context_rebuild_trigger_tokens: without it, a turn with the
    same tool set extends the kept conversation. Reached, it rebuilds,
    which keeps only the last history_messages."""
    hook = _context.make_context_hook(lambda state: "", keep_last=2, rebuild="mode_change",
                                      rebuild_trigger_tokens=trigger)
    first = _turns(0, 4)
    assert len(hook({"messages": first, "agent_mode": "editing"})["messages"]) == 2
    second = first + _turns(4, 4)
    assert len(hook({"messages": second, "agent_mode": "editing"})["messages"]) == expected


@pytest.mark.parametrize("every, calls", [(2, 2), (0, 0)])
def test_progress_heartbeat_requests_reports_every_n_requests(every, calls):
    """[coverage] progress_heartbeat_requests: every N model requests a
    progress line, none with zero."""
    seen = []
    tracker = lg.UsageTracker(request_limit=10, heartbeat_every=every,
                              heartbeat_fn=lambda used, limit, elapsed: seen.append(used))
    for _ in range(4):
        tracker.note_request()
    assert len(seen) == calls


def test_progress_heartbeat_requests_reaches_the_agents(tmp_path):
    settings = RunSettings(base_dir=tmp_path, sections={"coverage": {"progress_heartbeat_requests": "7"}})
    with config.use_run_settings(settings):
        assert config.get_agent_config("test_coverage")["progress_heartbeat_requests"] == 7


class _OpenedSessions:
    """A language with Python's session attempts that records the options
    of each mutation session the runner opens."""

    mutation_session_attempts = PythonAdapter.mutation_session_attempts

    def __init__(self):
        self.opened = []
        self.mutations = SimpleNamespace(open_session=self.open_session)

    def open_session(self, engine, **options):
        self.opened.append(options)
        return SimpleNamespace(outcome=Outcome.COMPLETED, reason="", close=lambda: None)


def _opened_settings(tmp_path, section):
    """The Cosmic Ray settings the mutation runner opens a session with
    in a run whose [mutation] section sets ``section``."""
    language = _OpenedSessions()
    with config.use_run_settings(RunSettings(base_dir=tmp_path, sections={"mutation": section})):
        mutation.MutationRunner(language, config.get_agent_config(
            "test_mutation", parent="test_coverage")).open(1, suite_dirs=())
    return language.opened[0]["settings"]


@pytest.mark.parametrize("section, expected", [
    ({"cosmic_ray_cache_max_entries": "3", "cosmic_ray_cache_max_age_days": "7"}, (3, 7)),
    ({}, (200, 0)),
])
def test_the_catalogue_cache_limits_reach_cosmic_ray(tmp_path, section, expected):
    """[mutation] cosmic_ray_cache_max_entries and _max_age_days bound
    the engine's catalogue cache."""
    chosen = _opened_settings(tmp_path, section)
    assert (chosen.cache_max_entries, chosen.cache_max_age_days) == expected


def test_the_empty_session_directories_stay_unset(tmp_path):
    """[mutation] cosmic_ray_session_dir and cosmic_ray_local_session_dir:
    their empty defaults leave both session directories unset."""
    with config.use_run_settings(RunSettings(base_dir=tmp_path, sections={})):
        configured = config.get_agent_config("test_mutation", parent="test_coverage")
    assert (configured["cosmic_ray_session_dir"], configured["cosmic_ray_local_session_dir"]) == ("", "")
    chosen = _opened_settings(tmp_path, {})
    assert (chosen.session_dir, chosen.local_session_dir) == (None, None)


@pytest.mark.parametrize("value, expected", [("true", True), ("false", False)])
def test_measure_lint_switches_the_lint_count(tmp_path, value, expected):
    """[finishing] measure_lint: finishing counts Ruff findings only when
    it is on."""
    settings = RunSettings(base_dir=tmp_path, sections={"finishing": {"measure_lint": value}})
    with config.use_run_settings(settings):
        assert measurement._bool_flag(measurement._RAW_METRICS_CONFIG, "measure_lint") is expected


def test_demo_playback_seconds_sets_the_default_length(monkeypatch):
    """[demo] playback_seconds: the playback's length at speed 1, read from
    the packaged configuration, unless --demo-playback-seconds names one."""
    from testuary.demo import cli as demo_cli

    assert demo_cli._playback_seconds(argparse.Namespace(demo_duration=None)) == float(
        settings_module._defaults()["demo"]["playback_seconds"])
    monkeypatch.setattr(settings_module, "_defaults", lambda: {"demo": {"playback_seconds": "30"}})
    assert demo_cli._playback_seconds(argparse.Namespace(demo_duration=None)) == 30.0
    assert demo_cli._playback_seconds(argparse.Namespace(demo_duration="5")) == 5.0
    monkeypatch.setattr(settings_module, "_defaults", lambda: {"demo": {"playback_seconds": "0"}})
    with pytest.raises(ValueError, match="playback_seconds must be a positive number"):
        demo_cli._playback_seconds(argparse.Namespace(demo_duration=None))
