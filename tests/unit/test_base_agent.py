"""Tests for the ``Agent`` and ``CodingAgent`` base classes.

Per-run state is plain instance attributes, so two instances are isolated,
and a tool is a bound method whose schema omits ``self``.
"""
from __future__ import annotations

import types

import pytest
from langchain_core.tools import tool as lc_tool

from agent_prompts import python_prompts

from testuary.agent._base import CodingAgent
from testuary.agent._helpers import CodingAgentSession, ELIDED_SENTINEL
import ast
import functools
import hashlib
import inspect
import json
import platform
import textwrap
from pathlib import Path
import test_model_surface_snapshot as surface

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


class _Demo(CodingAgent):
    """Minimal concrete editing agent for the foundation tests."""

    read_tool_names = ("read_source", "read_tests")

    def target_path(self):
        return "/work/foo.py"

    def sandbox_root(self):
        return "/work"

    def echo(self, text: str) -> str:
        """Echo the text, prefixed by this instance's deps."""
        return f"{self.deps}:{text}"


def _make(prompts=None, deps="D"):
    return _Demo(
        prompts=prompts or {"tool_messages": {}},
        deps=deps,
        session=CodingAgentSession(),
    )


def test_tool_msg_resolves_from_prompts() -> None:
    a = _make(prompts={"tool_messages": {"hi": "hello $x"}})
    assert a.tool_msg("hi", x="world") == "hello world"


def test_loop_breaker_uses_instance_session() -> None:
    a = _make()
    a.session.loop_breaker.no_progress_threshold = 2
    a.session.current_tool_name = "read"
    assert "STUCK" not in a.track_output("x")   # below threshold
    assert "STUCK" in a.track_output("x")       # threshold reached
    a.progress_made()
    assert a.session.loop_breaker.calls_since_last_edit == 0


def test_loop_breaker_check_escalates() -> None:
    a = _make(prompts={"tool_messages": {"run_tests_cached_rejected": "$last_run_tests_result"}})
    a.session.tests_changed_since_run_tests = False
    a.session.last_run_tests_result = "cached body"
    out = a.run_tests_cached_skip()
    assert "cached body" in out and "run_tests" in out
    assert a.session.loop_breaker.consecutive_cached_calls["run_tests"] == 1


def test_two_instances_have_isolated_state() -> None:
    a, b = _make(), _make()
    a.session.current_tool_name = "read"
    a.track_output("x")
    assert a.session.loop_breaker.calls_since_last_edit == 1
    assert b.session.loop_breaker.calls_since_last_edit == 0


def test_bound_method_binds_as_tool_without_self() -> None:
    a = _make(deps="inst")
    t = lc_tool(a.echo)
    assert "self" not in t.args
    assert set(t.args) == {"text"}
    assert t.invoke({"text": "hi"}) == "inst:hi"


def test_hooks_default_and_override() -> None:
    a = _make()
    assert a.target_path() == "/work/foo.py"
    assert a.sandbox_root() == "/work"


@pytest.fixture(params=["coverage", "mutation"])
def editing_finalizer(request, tmp_path, monkeypatch):
    from testuary.agent import coverage_agent as coverage
    from testuary.agent import mutation_agent as mutation

    target = tmp_path / "source.py"
    target.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    deps = {
        "target_file_path": str(target),
        "target_file_content": target.read_text(encoding="utf-8"),
        "source_dir": str(tmp_path),
        "test_output_dir": str(tmp_path),
        "language": "python",
    }
    common_fields = {
        "agent_duration_s": 999.0, "model_requests": 999,
        "logical_requests": 999, "unknown_usage_requests": 999,
        "input_tokens": 999, "output_tokens": 999, "cache_hit_tokens": 999,
    }
    if request.param == "coverage":
        agent = coverage.CoverageAgent(
            prompts=python_prompts("test_coverage"),
            deps=coverage.CoverageTestDeps(**deps, key_functions=["f"]),
            session=coverage.TestCoverageSession(),
        )
        output = coverage.CoverageTestResult(
            test_file_name="wrong.py", explanation="Exercise f's boundary",
            targeted_functions=["f"], **common_fields,
        )
        delivered = tmp_path / "test_source.py"
        monkeypatch.setattr(agent, "_log_run_summary", lambda: None)
        monkeypatch.setattr(coverage.measurement, "compute_file_complexity", lambda _p: 1)
        restore_name = "_maybe_restore_last_green"
    else:
        agent = mutation.MutationAgent(
            prompts=python_prompts("test_mutation"),
            deps=mutation.MutationTestDeps(
                **deps, mutation_test_file_name="test_source_mutation.py",
            ),
            session=mutation.TestMutationSession(),
        )
        output = mutation.MutationTestResult(
            mutation_test_file_name="wrong.py", explanation="Exercise f's boundary",
            targeted_mutants=["source.py:2:AOR"], **common_fields,
        )
        delivered = agent.target_path()
        restore_name = "_settle_best_green"

    clock = [100.0]
    events = []
    source = "def test_a():\n    assert True\n\ndef test_b():\n    assert True\n"

    def restore(*args, **kwargs):
        delivered.write_text(source, encoding="utf-8")
        if request.param == "coverage":
            agent.session.prev_coverage_pct = 96.24494511842865
            agent.session.prev_branch_coverage_pct = 95.48872180451127
        events.append("restore")
        clock[0] += 3.456

    monkeypatch.setattr(agent, restore_name, restore)
    return types.SimpleNamespace(
        agent=agent, output=output, phase=request.param, delivered=delivered,
        source=source, clock=clock, events=events,
    )


@pytest.mark.parametrize("armed", [False, True])
@pytest.mark.parametrize("has_usage", [False, True])
@pytest.mark.parametrize("error_msg", [None, "limit stop"])
def test_editing_finalizers_share_terminal_accounting(
    editing_finalizer, monkeypatch, armed, has_usage, error_msg,
):
    import testuary.agent._base as common

    case = editing_finalizer
    case.agent.session.phase_clock_start = 90.0 if armed else None
    monkeypatch.setattr(common.time, "perf_counter", lambda: case.clock[0])
    measured_score = 87.123456789
    if case.phase == "mutation":
        case.agent.session.best_mutation_score = measured_score
    usage = types.SimpleNamespace(
        requests_used=12 if has_usage else 0,
        logical_requests_used=9 if has_usage else 0,
        unknown_usage_requests=3 if has_usage else 0,
        input_tokens_used=700 if has_usage else 0,
        output_tokens_used=100 if has_usage else 0,
        cache_hit_tokens_used=222 if has_usage else 0,
    )
    messages = [object()]
    original = CodingAgent.finalize_result
    stamped = []

    def shared_stamp(self, output, **kwargs):
        assert case.events == ["restore"]
        assert output.status == ("budget_exhausted" if error_msg else "success")
        assert output.error == (error_msg or "")
        assert kwargs["duration_ms"] == 1234.0
        assert kwargs["usage_tracker"] is usage
        assert kwargs["final_messages"] is messages
        stamped.append(output)
        return original(self, output, **kwargs)

    monkeypatch.setattr(CodingAgent, "finalize_result", shared_stamp)
    # An error here is a limit's end: run_phase marks the run so.
    case.agent._limit_reached = error_msg is not None
    result = case.agent.finalize_result(
        case.output, error_msg=error_msg, duration_ms=1234.0,
        usage_tracker=usage, final_messages=messages,
    )

    assert len(stamped) == 1
    assert case.delivered.read_text(encoding="utf-8") == case.source
    expected = {
        "agent_duration_s": 13.5 if armed else 1.2,
        "model_requests": usage.requests_used,
        "logical_requests": usage.logical_requests_used,
        "unknown_usage_requests": usage.unknown_usage_requests,
        "input_tokens": usage.input_tokens_used,
        "output_tokens": usage.output_tokens_used,
        "cache_hit_tokens": usage.cache_hit_tokens_used,
    }
    assert {key: getattr(result, key) for key in expected} == expected
    assert case.output.agent_duration_s == 999.0
    assert case.output.model_requests == 999
    if case.phase == "coverage":
        assert result.test_file_name == case.delivered.name
        assert result.line_coverage == 96.24494511842865
        assert result.branch_coverage == 95.48872180451127
        assert result.test_count_standard == 2
    else:
        assert result.mutation_test_file_name == case.delivered.name
        assert result.mutation_score == measured_score
        assert result.test_count_mutation == 2


def test_editing_finalizers_preserve_absent_output(editing_finalizer):
    case = editing_finalizer
    assert case.agent.finalize_result(
        None, error_msg=None, duration_ms=1234.0,
        usage_tracker=None, final_messages=[],
    ) is None
    assert case.events == ["restore"]


def test_editing_finalizers_do_not_stamp_after_restore_failure(
    editing_finalizer, monkeypatch,
):
    case = editing_finalizer
    failure = OSError("restore failed")

    def refuse_restore(*args, **kwargs):
        raise failure

    def refuse_stamp(*args, **kwargs):
        pytest.fail("failed terminal work must not reach the accounting stamp")

    restore_name = (
        "_maybe_restore_last_green" if case.phase == "coverage"
        else "_settle_best_green"
    )
    monkeypatch.setattr(case.agent, restore_name, refuse_restore)
    monkeypatch.setattr(CodingAgent, "finalize_result", refuse_stamp)
    with pytest.raises(OSError, match="restore failed") as caught:
        case.agent.finalize_result(
            case.output, error_msg=None, duration_ms=1234.0,
            usage_tracker=None, final_messages=[],
        )
    assert caught.value is failure
    assert case.output.agent_duration_s == 999.0


def test_log_thinking_sets_current_tool_and_read_cache() -> None:
    a = _make()
    a.log_thinking("read_source")
    assert a.session.current_tool_name == "read_source"
    assert a.consecutive_reads("read_source") == 1
    a.log_thinking("read_source")                  # immediate re-read
    assert a.consecutive_reads("read_source") == 2
    a.log_thinking("read_tests")           # a different read tool
    assert a.consecutive_reads("read_tests") == 1
    assert a.consecutive_reads("read_source") == 0  # other read cleared
    a.log_thinking("edit_file")                 # a non-read tool
    assert a.consecutive_reads("read_tests") == 0  # all cleared


def test_track_output_counts_and_truncates() -> None:
    a = _make()
    a.session.current_tool_name = "run_tests"
    out = a.track_output("hello")
    assert out.endswith("hello")                   # no banner below threshold
    assert a.session.session_tool_calls == 1


def test_track_output_prepends_no_progress_banner() -> None:
    a = _make()
    a.session.loop_breaker.no_progress_threshold = 1
    a.session.current_tool_name = "run_tests"
    out = a.track_output("body")
    assert "STUCK" in out and out.endswith("body")


def test_progress_made_clears_cached_call_counters() -> None:
    a = _make()
    a.session.loop_breaker.consecutive_cached_calls["run_tests"] = 4
    a.session.loop_breaker.calls_since_last_edit = 7
    a.progress_made()
    assert a.session.loop_breaker.consecutive_cached_calls == {}
    assert a.session.loop_breaker.calls_since_last_edit == 0


# -- the shared editing tool: edit_file over a real temp file --

_RT_MSGS = {
    "edit_no_file": "NO FILE",
    "edit_read_failed": "READ FAILED $name: $exc",
    "edit_empty_old": "EMPTY OLD",
    "edit_history_marker": "MARKER IN ARGS",
    "edit_not_found": "NOT FOUND $name ($line_count lines)$hint",
    "edit_multi_match": "MULTI $occurrences $name",
    "edit_unchanged": "UNCHANGED $name",
    "edit_post_edit_syntax_error": "SYNTAX $parse_err",
    "edit_ok": (
        "OK $replaced_msg $name$loose_note lines=$n_lines "
        "region=$lo-$hi window=$context_window\n$snippet"
    ),
}


class _EditAgent(CodingAgent):
    """Concrete editing agent over a real temp file, for edit_file."""

    read_tool_names = ("read_source",)

    def __init__(self, path):
        super().__init__(
            prompts={"tool_messages": _RT_MSGS},
            deps=types.SimpleNamespace(
                language="python", source_dir=str(path.parent),
            ),
            session=CodingAgentSession(),
        )
        self._path = path
        self._sandbox = str(path.parent)
        self.wrote = 0

    def target_path(self):
        return self._path

    def sandbox_root(self):
        return self._sandbox

    def after_write(self):
        self.wrote += 1


def _edit_agent(tmp_path, body="def f():\n    return 1\n"):
    p = tmp_path / "target_test.py"
    p.write_text(body, encoding="utf-8")
    return _EditAgent(p)


def test_edit_success_writes_and_echoes(tmp_path) -> None:
    a = _edit_agent(tmp_path)
    out = a.edit_file("return 1", "return 2")
    assert a._path.read_text() == "def f():\n    return 2\n"
    assert out.startswith("OK 1 occurrence")
    assert a.wrote == 1
    assert a.session.last_tests_passed is False


def test_edit_not_found_leaves_file(tmp_path) -> None:
    a = _edit_agent(tmp_path)
    out = a.edit_file("return 999", "return 2")
    assert out.startswith("NOT FOUND")
    assert a._path.read_text() == "def f():\n    return 1\n"
    assert a.wrote == 0


def test_edit_multi_match_rejected(tmp_path) -> None:
    a = _edit_agent(tmp_path, body="a = 1\nb = 1\n")
    out = a.edit_file("= 1", "= 2")
    assert out == "MULTI 2 target_test.py"
    assert a.wrote == 0


def test_edit_replace_all(tmp_path) -> None:
    a = _edit_agent(tmp_path, body="a = 1\nb = 1\n")
    out = a.edit_file("= 1", "= 2", replace_all=True)
    assert a._path.read_text() == "a = 2\nb = 2\n"
    assert out.startswith("OK all 2 occurrences")


def test_edit_insert_via_anchor_grows_file(tmp_path) -> None:
    """An insert keeps the anchor line in new_text, so the file gains a line
    and loses nothing."""
    a = _edit_agent(tmp_path)  # body: "def f():\n    return 1\n"
    a.edit_file("    return 1\n", "    y = 0\n    return 1\n")
    assert a._path.read_text() == "def f():\n    y = 0\n    return 1\n"
    assert a.wrote == 1


def test_edit_delete_with_empty_new_text(tmp_path) -> None:
    """An empty new_text removes the matched block."""
    a = _edit_agent(tmp_path, body="a = 1\nb = 2\n")
    a.edit_file("a = 1\n", "")
    assert a._path.read_text() == "b = 2\n"
    assert a.wrote == 1


def test_edit_empty_old_rejected(tmp_path) -> None:
    a = _edit_agent(tmp_path)
    assert a.edit_file("", "x") == "EMPTY OLD"
    assert a.wrote == 0


def test_edit_unchanged_when_old_equals_new(tmp_path) -> None:
    a = _edit_agent(tmp_path)
    assert a.edit_file("return 1", "return 1") == "UNCHANGED target_test.py"
    assert a.wrote == 0


def test_edit_post_edit_syntax_error_rejected(tmp_path) -> None:
    a = _edit_agent(tmp_path)
    out = a.edit_file("    return 1", "    return (")  # unbalanced bracket
    assert out.startswith("SYNTAX")
    assert a._path.read_text() == "def f():\n    return 1\n"
    assert a.wrote == 0


def test_edit_no_file_when_target_missing(tmp_path) -> None:
    a = _EditAgent(tmp_path / "missing.py")
    assert a.edit_file("x", "y") == "NO FILE"


def test_edit_rejects_history_marker(tmp_path) -> None:
    a = _edit_agent(tmp_path)
    out = a.edit_file("return 1" + ELIDED_SENTINEL, "return 2")
    assert out == "MARKER IN ARGS"
    assert a.wrote == 0


def test_edit_sandbox_escape_blocked(tmp_path) -> None:
    a = _edit_agent(tmp_path)
    a._sandbox = str(tmp_path / "elsewhere")  # the file lives outside this
    out = a.edit_file("return 1", "return 2")
    assert "BLOCKED by sandbox gate" in out
    assert a.wrote == 0


def test_sandbox_guard_allows_path_in_sandbox_root(tmp_path) -> None:
    a = _edit_agent(tmp_path)
    a.session.current_tool_name = "edit_tests"
    assert a.sandbox_guard(a._path) is None


def test_sandbox_guard_blocks_escape(tmp_path) -> None:
    a = _edit_agent(tmp_path)
    a._sandbox = str(tmp_path / "elsewhere")  # the target lives outside this
    a.session.current_tool_name = "edit_tests"
    out = a.sandbox_guard(a._path)
    assert out is not None and "BLOCKED by sandbox gate" in out


# -- shared state-marker building blocks --

def test_stalled_metric_none_when_unmeasured() -> None:
    a = _make()
    assert a.stalled_metric(None, lambda v: f"{v}%") == "none"


def test_stalled_metric_bare_value_without_streak() -> None:
    a = _make()
    a.session.measurements_without_improvement = 0
    assert a.stalled_metric(56, lambda v: f"{v}%") == "56%"


def test_stalled_metric_appends_stall_suffix() -> None:
    a = _make()
    a.session.measurements_without_improvement = 3
    assert a.stalled_metric(56, lambda v: f"{v}%") == "56% (stalled 3 measurements)"


class _RecordingLogger:
    def __init__(self) -> None:
        self.messages: list = []

    def log(self, message: str, level: int = 0) -> None:
        self.messages.append(message)


class _ProgressAgent(CodingAgent):
    def phase_label(self) -> str:
        return "coverage"


def _progress_agent() -> _ProgressAgent:
    return _ProgressAgent(
        prompts={"tool_messages": {}}, deps="D",
        session=CodingAgentSession(), logger=_RecordingLogger(),
    )


def test_log_progress_formats_the_uniform_line() -> None:
    a = _progress_agent()
    a.log_progress(3, "line 56% branch 43%", "56%",
                   target="goal 80%", delta="+4 vs prev")
    assert a.logger.messages == [
        "[coverage #3] line 56% branch 43% | best 56% | goal 80% (+4 vs prev)"
    ]


def test_log_progress_omits_empty_target_and_delta() -> None:
    a = _progress_agent()
    a.log_progress(1, "score 71.4%", "71.4%")
    assert a.logger.messages == ["[coverage #1] score 71.4% | best 71.4%"]


def test_edit_whitespace_loose_match(tmp_path) -> None:
    a = _edit_agent(tmp_path)  # default body indents return with 4 spaces
    # The 8-space old/new only match loosely, and new_text is re-indented to
    # the file's 4 spaces.
    out = a.edit_file("        return 1", "        return 2")
    assert a._path.read_text() == "def f():\n    return 2\n"
    assert "whitespace-loose match" in out


def test_edit_isolation_between_instances(tmp_path) -> None:
    a = _edit_agent(tmp_path, body="x = 1\n")
    pb = tmp_path / "b.py"
    pb.write_text("y = 1\n", encoding="utf-8")
    b = _EditAgent(pb)
    a.edit_file("x = 1", "x = 2")
    assert a.wrote == 1 and b.wrote == 0
    assert pb.read_text() == "y = 1\n"  # b untouched by a's edit


# -- shared read helpers: cached_read / file_read --

_READ_MSGS = {
    "cached": "CACHED #$n $name",
    "nudge": "NUDGE #$n",
    "missing": "NO FILE",
    "rerr": "READ FAIL $name: $exc",
}


def _read_agent():
    return _Demo(
        prompts={"tool_messages": _READ_MSGS}, deps="D",
        session=CodingAgentSession(),
    )


def test_cached_read_renders_then_caches_then_nudges() -> None:
    a = _read_agent()
    content = "a\nb\nc\n"
    # 1st whole read -> full line-numbered render.
    a.log_thinking("read_source")
    out = a.cached_read("read_source", content=content, name="f.py",
                        cached_key="cached", cached_subs={"name": "f.py"})
    assert "1\ta" in out
    # 2nd immediate whole re-read -> cached header, not the body.
    a.log_thinking("read_source")
    out = a.cached_read("read_source", content=content, name="f.py",
                        cached_key="cached", cached_subs={"name": "f.py"})
    assert out == "CACHED #2 f.py"
    # 4th -> the loop-breaker nudge is prepended when a key is supplied.
    a.log_thinking("read_source")
    a.log_thinking("read_source")
    out = a.cached_read("read_source", content=content, name="f.py",
                        cached_key="cached", cached_subs={"name": "f.py"},
                        loop_breaker_key="nudge")
    assert out.startswith("NUDGE #4")
    assert "CACHED #4 f.py" in out


def test_cached_read_first_paged_request_renders() -> None:
    a = _read_agent()
    a.log_thinking("read_source")
    a.log_thinking("read_source")  # n == 2, but a first-time paged read renders
    out = a.cached_read("read_source", content="a\nb\nc\n", name="f.py",
                        cached_key="cached", start_line=2, end_line=3)
    assert "2\tb" in out and "CACHED" not in out


def test_cached_read_repeated_paged_range_is_cached() -> None:
    """Rereading the same paged range without an edit returns the cached
    header, while a different range still renders."""
    a = _read_agent()
    content = "a\nb\nc\nd\ne\n"
    a.log_thinking("read_source")
    out = a.cached_read("read_source", content=content, name="f.py",
                        cached_key="cached", cached_subs={"name": "f.py"},
                        start_line=2, end_line=3)
    assert "2\tb" in out                              # first sight: full render
    a.log_thinking("read_source")
    out = a.cached_read("read_source", content=content, name="f.py",
                        cached_key="cached", cached_subs={"name": "f.py"},
                        start_line=2, end_line=3)
    assert out == "CACHED #2 f.py"                    # same range again: cached
    a.log_thinking("read_source")
    out = a.cached_read("read_source", content=content, name="f.py",
                        cached_key="cached", cached_subs={"name": "f.py"},
                        start_line=4, end_line=5)
    assert "4\td" in out and "CACHED" not in out


def test_file_read_missing_returns_message(tmp_path) -> None:
    a = _read_agent()
    a.log_thinking("read_tests")
    out = a.file_read("read_tests", path=tmp_path / "none.py",
                      name="t.py", missing_msg="NO FILE")
    assert out == "NO FILE"


def test_file_read_renders_existing_file(tmp_path) -> None:
    a = _read_agent()
    p = tmp_path / "t.py"
    p.write_text("x = 1\ny = 2\n", encoding="utf-8")
    a.log_thinking("read_tests")
    out = a.file_read("read_tests", path=p, name="t.py",
                      missing_msg="NO FILE")
    assert "1\tx = 1" in out and "2\ty = 2" in out


def test_file_read_error_uses_key_then_fallback(tmp_path, monkeypatch) -> None:
    a = _read_agent()
    p = tmp_path / "t.py"
    p.write_text("ok\n", encoding="utf-8")

    def _boom(*_a, **_k):
        raise OSError("disk gone")
    monkeypatch.setattr(type(p), "read_text", _boom)
    a.log_thinking("read_tests")
    out = a.file_read("read_tests", path=p, name="t.py",
                      missing_msg="NO FILE", read_error_key="rerr")
    assert out == "READ FAIL t.py: disk gone"
    # Without a key, a plain fallback message is used.
    a.log_thinking("read_tests")
    out = a.file_read("read_tests", path=p, name="t.py",
                      missing_msg="NO FILE")
    assert "ERROR reading t.py" in out and "disk gone" in out


def test_the_configured_run_limits_reach_the_usage_tracker(
    tmp_path, monkeypatch, in_run,
) -> None:
    """``max_total_tokens_per_phase`` and ``max_wall_clock_seconds_per_phase`` from
    the coverage and mutation settings sections reach the usage tracker
    through the resolver and ``make_usage_tracker``."""
    from testuary import config

    cfg = tmp_path / "limits.ini"
    cfg.write_text(
        "[coverage]\nmax_total_tokens_per_phase = 5000\n"
        "max_wall_clock_seconds_per_phase = 90\n"
        "[mutation]\nmax_total_tokens_per_phase = 7000\n"
        "max_wall_clock_seconds_per_phase = 30\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("TESTUARY_CONFIG_FILE", str(cfg))
    in_run()
    coverage = config.get_agent_config("test_coverage")
    mutation = config.get_agent_config(
        "test_mutation", parent="test_coverage")

    limits = []
    for resolved in (coverage, mutation):
        agent = _make()
        agent.agent_config = lambda resolved=resolved: resolved
        agent.phase_label = lambda: "limits"
        agent.log = lambda *args, **kwargs: None
        tracker = agent.make_usage_tracker()
        limits.append(
            (tracker.combined_token_limit, tracker.wall_clock_limit_seconds))
    assert limits == [(5000, 90.0), (7000, 30.0)]


def _past_the_deadline(tracker) -> None:
    import time

    tracker.note_request()
    tracker._started_monotonic = time.monotonic() - 100.0
    tracker.note_request()


@pytest.mark.parametrize("setting, value, reach, kind", [
    ("max_requests_per_phase", "1",
     lambda tracker: (tracker.note_request(), tracker.note_request()), "request"),
    ("max_input_tokens_per_phase", "10",
     lambda tracker: tracker.note_tokens(10, 0), "token"),
    ("max_output_tokens_per_phase", "10",
     lambda tracker: tracker.note_tokens(0, 10), "token"),
    ("max_total_tokens_per_phase", "10",
     lambda tracker: tracker.note_tokens(5, 5), "token"),
    ("max_wall_clock_seconds_per_phase", "10", _past_the_deadline, "time"),
])
def test_a_limit_stop_names_the_setting_the_user_set(
    tmp_path, monkeypatch, in_run, setting, value, reach, kind,
) -> None:
    """Each phase limit's stop, which tests.csv's error column records,
    names the setting that set the limit, with its value; the stop is a
    ``budget_exhausted`` outcome, whose description names its kind."""
    from testuary import config
    from testuary.agent import _langgraph as lg
    from testuary import results

    cfg = tmp_path / "limits.ini"
    cfg.write_text(f"[coverage]\n{setting} = {value}\n", encoding="utf-8")
    monkeypatch.setenv("TESTUARY_CONFIG_FILE", str(cfg))
    in_run()
    resolved = config.get_agent_config("test_coverage")
    agent = _make()
    agent.agent_config = lambda: resolved
    agent.phase_label = lambda: "limits"
    agent.log = lambda *args, **kwargs: None
    tracker = agent.make_usage_tracker()
    with pytest.raises(lg.UsageLimitExceeded) as stopped:
        reach(tracker)
    assert str(stopped.value).startswith(f"{setting} ({value}"), stopped.value
    error = f"Budget exhausted: UsageLimitExceeded: {stopped.value}"
    # The run that ended at this limit reads budget_exhausted.
    assert CodingAgent.classify_error_status(error, limit_reached=True) == "budget_exhausted"
    assert kind in results.outcome_text("budget_exhausted")


def _with_docstring(method, docstring):
    def stand_in(*args, **kwargs):
        return method(*args, **kwargs)

    functools.update_wrapper(stand_in, method)
    stand_in.__doc__ = docstring
    return stand_in


# Python 3.10 to 3.12 keep a docstring's indentation; 3.13 and later remove
# the common indentation of the lines after the first when compiling. The
# model receives the same tool descriptions from both, and from the docstring
# this interpreter compiled.
@pytest.mark.parametrize("builder", ["test_coverage", "test_mutation"])
def test_the_tool_descriptions_are_the_same_on_every_python_version(
    builder, tmp_path: Path,
):
    agent = surface._BUILDERS[builder](tmp_path)
    reads, writes, run_m, measure_m = agent.tool_methods()
    indented = 0
    for method in [*reads, *writes, run_m, measure_m]:
        source = ast.parse(textwrap.dedent(inspect.getsource(method))).body[0]
        written = ast.get_docstring(source, clean=False)
        first, _, rest = written.partition("\n")
        compiled = f"{first}\n{textwrap.dedent(rest)}"
        indented += written != compiled
        described = {
            agent._bind_tool(_with_docstring(method, docstring)).description
            for docstring in (written, compiled, method.__doc__)
        }
        assert len(described) == 1, method.__name__
    assert indented


@pytest.mark.parametrize("builder", ["test_coverage", "test_mutation"])
def test_the_chat_log_records_the_interpreter_and_the_descriptions(
    builder, tmp_path: Path,
):
    from testuary.agent import _langgraph as lg

    agent = surface._BUILDERS[builder](tmp_path)
    reads, writes, run_t, measure_t = agent.phase_tools()
    final_tool = lg._make_final_answer_tool(agent.result_schema, agent.final_tool_name)
    described = sorted(
        ((tool.name, tool.description, tool.args_schema.model_json_schema())
         for tool in reads + writes + [run_t, measure_t, final_tool]),
        key=lambda entry: entry[0],
    )
    expected = hashlib.sha256(
        json.dumps(described, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()

    metadata = agent.chat_log_metadata()

    assert metadata["python_version"] == platform.python_version()
    assert metadata["tool_descriptions_sha256"] == expected
    # The agent's own fields are still there.
    assert metadata["target_file"] == agent.deps.target_file_path


class _FrameAgent(_Demo):
    def __init__(self, failure):
        super().__init__(prompts={"tool_messages": {}},
                         deps=types.SimpleNamespace(target_file_path="/work/calc.py"),
                         session=CodingAgentSession())
        self.failure = failure

    def test_paths(self):
        raise self.failure


def test_frame_files_name_the_target_alone_without_test_paths() -> None:
    agent = _FrameAgent(NotImplementedError())
    assert agent.frame_files() == ("calc.py",)


def test_frame_files_let_a_failing_test_path_lookup_through() -> None:
    agent = _FrameAgent(OSError("the test directory is unreadable"))
    with pytest.raises(OSError, match="unreadable"):
        agent.frame_files()
