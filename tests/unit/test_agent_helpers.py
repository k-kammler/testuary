"""Tests for the pure helpers and dataclass primitives of
``testuary.agent._helpers``: tool-return helpers, pytest output processing,
loop-breaker state, best-green snapshots, the write gate of the test output directory
and advisory per-test reports.

History reduction and the recovery loop need message objects and a model
client and are covered by ``test_graph.py``.
"""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_fixtures import run_result
from testuary.agent import _helpers as ah
from testuary.agent import _edit_tool as edit_tool
from testuary import _test_output
from testuary.agent._helpers import tool_msg
from testuary.prompts import load_agent_prompts


# ---------------------------------------------------------------------------
# contains_history_marker
# ---------------------------------------------------------------------------

def test_contains_history_marker_finds_the_sentinel() -> None:
    """The elision sentinel is detected, so write-tool validators keep the
    placeholder from reaching disk as test content."""
    payload = "some code\n<<<TESTUARY_ELIDED>>>\nmore code"
    assert ah.contains_history_marker(payload) is True


def test_contains_history_marker_ignores_a_prose_shaped_marker() -> None:
    """Only the exact history sentinel counts, not a prose-shaped marker."""
    prose = "x = 1\n[trimmed_history: 12 lines elided]\ny = 2"
    assert ah.contains_history_marker(prose) is False


def test_contains_history_marker_clean_payload() -> None:
    """A real-looking Python payload without markers passes."""
    assert ah.contains_history_marker("def foo(): return 1") is False


def test_contains_history_marker_empty_and_none() -> None:
    """Empty string and None give False, so cleared arguments do not
    trigger the validator."""
    assert ah.contains_history_marker("") is False
    assert ah.contains_history_marker(None) is False


# ---------------------------------------------------------------------------
# source_structure_error (language-agnostic, via the LanguageBackend)
# ---------------------------------------------------------------------------

def test_source_structure_error_clean_source_returns_none() -> None:
    """Valid source passes without a message."""
    assert ah.source_structure_error("def f(): return 1\n", "python") is None


def test_source_structure_error_includes_line_and_context() -> None:
    """A syntax error message names the line and shows a snippet window with
    an arrow on the offending line."""
    bad = "def f():\n    return 1\n  )))\n"
    err = ah.source_structure_error(bad, "python")
    assert err is not None
    assert "syntax error" in err.lower()
    assert "line" in err
    assert "→" in err


def test_source_structure_error_quotes_the_offending_source_as_written() -> None:
    """The message quotes the offending source as written, not as a Python
    literal: line breaks and indentation become single spaces, cut to 80
    characters with the omission marker after the quote."""
    spanning = ah.source_structure_error("x = (1,\n     2,\n", "python")
    assert spanning.split("\n\n")[0] == "syntax error at line 1:1: 'x = (1, 2,'"
    long_line = "x = f(" + "alpha, " * 20
    cut = ah.source_structure_error(long_line + "\n", "python").split("\n\n")[0]
    assert cut.startswith(f"syntax error at line 1:1: '{long_line[:80]}' ")
    assert cut.endswith(f"{len(long_line.strip()) - 80} chars of the source omitted ...]")


def test_source_structure_error_empty_input_is_ok() -> None:
    """An empty payload passes."""
    assert ah.source_structure_error("", "python") is None


def test_source_structure_error_unknown_language_does_not_block() -> None:
    """A language with no backend cannot be validated, so the gate returns
    None rather than rejecting the source."""
    assert ah.source_structure_error("@@@ not python @@@", "brainfuck") is None


# ---------------------------------------------------------------------------
# strip_markdown_code_fences
# ---------------------------------------------------------------------------

def test_strip_markdown_code_fences_both_fences_present() -> None:
    """Both fences are stripped and inner backticks survive."""
    fenced = "```python\nx = 1\nprint(`hi`)\n```\n"
    assert edit_tool.strip_markdown_code_fences(fenced) == "x = 1\nprint(`hi`)"


def test_strip_markdown_code_fences_opener_only_strips_opener() -> None:
    """An unclosed fence loses its opener and keeps the body."""
    fenced = "```python\nx = 1\n"
    assert edit_tool.strip_markdown_code_fences(fenced) == "x = 1"


def test_strip_markdown_code_fences_no_fence_is_passthrough() -> None:
    """Plain source is returned unchanged."""
    plain = "def f():\n    return 1\n"
    assert edit_tool.strip_markdown_code_fences(plain) == plain


def test_strip_markdown_code_fences_blank_input() -> None:
    assert edit_tool.strip_markdown_code_fences("") == ""
    assert edit_tool.strip_markdown_code_fences("   ") == "   "


# ---------------------------------------------------------------------------
# whitespace_loose_match + apply_indent_diff (used together by edit_file)
# ---------------------------------------------------------------------------

def test_whitespace_loose_match_unique_match_returns_block_and_diff() -> None:
    """A 4-space ``old_text`` matches the unique 8-space region and reports
    the +4 indent delta for re-indenting ``new_text``."""
    haystack = (
        "def outer():\n"
        "        x = 1\n"          # 8 spaces in the file
        "        return x\n"
    )
    needle = "    x = 1\n    return x"   # 4 spaces pasted
    result = edit_tool.whitespace_loose_match(needle, haystack)
    assert result is not None
    actual, diff, offset = result
    assert "x = 1" in actual
    assert haystack[offset:].startswith(actual) and offset == len("def outer():\n")
    assert diff == 4   # haystack has 4 MORE leading spaces than needle


def test_whitespace_loose_match_zero_matches_returns_none() -> None:
    """No match gives None, so the caller rejects rather than guesses."""
    assert edit_tool.whitespace_loose_match("def nothing(): pass", "x = 1\ny = 2") is None


def test_whitespace_loose_match_multiple_matches_returns_none() -> None:
    """Two equally good loose matches give None."""
    haystack = "    x = 1\n    return x\n    x = 1\n    return x\n"
    needle = "x = 1\nreturn x"
    assert edit_tool.whitespace_loose_match(needle, haystack) is None


def test_whitespace_loose_match_empty_inputs() -> None:
    assert edit_tool.whitespace_loose_match("", "stuff") is None
    assert edit_tool.whitespace_loose_match("stuff", "") is None


def test_whitespace_loose_match_keeps_a_form_feed_inside_its_line() -> None:
    """Lines end at a newline only, as reads number them."""
    haystack = "def f():\n        x = 1  # a\x0cb\n"
    result = edit_tool.whitespace_loose_match("    x = 1  # a\x0cb", haystack)
    assert result is not None
    assert result[0] == "        x = 1  # a\x0cb"


def test_whitespace_loose_match_needle_ending_in_a_blank_line() -> None:
    """Only the last line's newline is dropped, not the blank line's."""
    haystack = "def f():\n        x = 1\n\n        return x\n"
    result = edit_tool.whitespace_loose_match("    x = 1\n    ", haystack)
    assert result is not None
    assert result[0] == "        x = 1\n"


def test_strip_line_number_prefixes_keeps_a_form_feed_inside_its_line() -> None:
    text = "5\t    # page\x0cbreak\n6\t    assert f(1) == 2\n"
    assert edit_tool.strip_line_number_prefixes(text) == "    # page\x0cbreak\n    assert f(1) == 2\n"


def test_apply_indent_diff_positive_prepends_spaces() -> None:
    """+4 indent: every non-blank line gets 4 leading spaces."""
    text = "x = 1\nreturn x"
    assert edit_tool.apply_indent_diff(text, 4) == "    x = 1\n    return x"


def test_apply_indent_diff_negative_strips_only_when_present() -> None:
    """-4 indent strips 4 leading spaces where present and leaves lines with
    fewer alone."""
    text = "    x = 1\nreturn x\n        deeper"
    out = edit_tool.apply_indent_diff(text, -4)
    assert out == "x = 1\nreturn x\n    deeper"


def test_apply_indent_diff_zero_is_passthrough() -> None:
    """diff=0 leaves the text unchanged."""
    assert edit_tool.apply_indent_diff("    indented", 0) == "    indented"


def test_apply_indent_diff_preserves_blank_lines() -> None:
    """Blank lines are not indented."""
    text = "x = 1\n\nreturn x"
    out = edit_tool.apply_indent_diff(text, 4)
    assert out == "    x = 1\n\n    return x"


# ---------------------------------------------------------------------------
# format_thinking_elapsed
# ---------------------------------------------------------------------------

def test_format_thinking_elapsed_minutes_format() -> None:
    """60s+ is rendered as ``MmSSs`` with zero-padded seconds."""
    assert ah.format_thinking_elapsed(165.0) == "2m45s"
    assert ah.format_thinking_elapsed(60.0) == "1m00s"


def test_format_thinking_elapsed_seconds_format() -> None:
    """1s to 60s renders as ``N.Ns`` with one decimal."""
    assert ah.format_thinking_elapsed(3.4) == "3.4s"
    assert ah.format_thinking_elapsed(1.0) == "1.0s"


def test_format_thinking_elapsed_below_one_second_returns_none() -> None:
    """Sub-second times give None."""
    assert ah.format_thinking_elapsed(0.5) is None
    assert ah.format_thinking_elapsed(0.0) is None


# ---------------------------------------------------------------------------
# parse_pytest_elapsed
# ---------------------------------------------------------------------------

def test_parse_pytest_elapsed_finds_footer_seconds() -> None:
    """The seconds come from pytest's ``in N.NNs`` footer."""
    out = "...\n=========== 5 passed in 1.42s ==========="
    assert _test_output.parse_pytest_elapsed(out) == 1.42


def test_parse_pytest_elapsed_missing_marker_returns_none() -> None:
    """Output without the footer (collection error, timeout, kill) gives
    None."""
    assert _test_output.parse_pytest_elapsed("INTERNALERROR> Traceback ...") is None


# ---------------------------------------------------------------------------
# group_pytest_failures
# ---------------------------------------------------------------------------

def test_group_pytest_failures_empty_input_returns_empty_string() -> None:
    """Output without FAILED or ERROR summary lines gives no header."""
    assert _test_output.group_pytest_failures("hello world") == ""


def test_group_pytest_failures_single_failure_returns_empty() -> None:
    """A single failure gives no grouping, which would only repeat pytest's
    short summary."""
    out = "FAILED tests/test_foo.py::test_a - AssertionError: x"
    assert _test_output.group_pytest_failures(out) == ""


def test_group_pytest_failures_multiple_share_root_cause() -> None:
    """Failures with the same error class and 30-character message head form
    one group, and the largest group gets an ACTION line."""
    out = (
        "FAILED tests/test_a.py::t1 - AssertionError: x != y\n"
        "FAILED tests/test_a.py::t2 - AssertionError: x != y\n"
        "FAILED tests/test_a.py::t3 - AssertionError: x != y\n"
        "FAILED tests/test_b.py::t4 - TypeError: cannot int + str\n"
    )
    grouped = _test_output.group_pytest_failures(out)
    assert "Failure groups" in grouped
    assert "[3x] AssertionError" in grouped
    assert "[1x] TypeError" in grouped
    assert "ACTION" in grouped


# ---------------------------------------------------------------------------
# LoopBreakerState + helpers
# ---------------------------------------------------------------------------

def test_loop_breaker_no_progress_below_threshold_returns_empty() -> None:
    """Below the threshold the counter increments without a banner."""
    state = ah.LoopBreakerState(no_progress_threshold=3)
    assert ah.bump_no_progress(state) == ""
    assert ah.bump_no_progress(state) == ""
    assert state.calls_since_last_edit == 2


def test_loop_breaker_no_progress_at_threshold_returns_banner() -> None:
    """At the threshold the STUCK banner carries the configured wording."""
    state = ah.LoopBreakerState(
        no_progress_threshold=2,
        file_kind="test file",
        edit_action_text="edit it",
        return_action_text="return result",
    )
    ah.bump_no_progress(state)
    banner = ah.bump_no_progress(state)
    assert "STUCK" in banner
    assert "test file" in banner
    assert "edit it" in banner
    assert "Repeating a previous read, run, or measurement" in banner
    assert "different currently offered tool" in banner
    assert "no read / measurement / re-call" not in banner


def test_loop_breaker_check_tiered_escalation() -> None:
    """The per-tool message escalates from a mild note to REPEATED CALL and
    then to STUCK and STOP."""
    state = ah.LoopBreakerState()
    msg1 = ah.loop_breaker_check(state, "read_x", "cached payload")
    assert "no new result" in msg1
    for _ in range(2):
        ah.loop_breaker_check(state, "read_x", "cached payload")
    msg3 = ah.loop_breaker_check(state, "read_x", "cached payload")
    assert "REPEATED CALL #4" in msg3
    for _ in range(2):
        ah.loop_breaker_check(state, "read_x", "cached payload")
    msg6 = ah.loop_breaker_check(state, "read_x", "cached payload")
    assert "STUCK" in msg6
    assert "STOP" in msg6


def test_loop_breaker_check_counts_per_tool_independently() -> None:
    """Each ``tool_key`` has its own counter."""
    state = ah.LoopBreakerState()
    for _ in range(5):
        ah.loop_breaker_check(state, "tool_a", "x")
    msg_b = ah.loop_breaker_check(state, "tool_b", "x")
    assert "no new result" in msg_b   # tier 1, not REPEATED CALL


# ---------------------------------------------------------------------------
# BestGreenState + green_snapshot_key + update_best_green
# ---------------------------------------------------------------------------

def test_green_snapshot_key_orders_unmeasured_below_measured() -> None:
    """An unmeasured mutation score sorts below any measured one."""
    none_key = ah.green_snapshot_key(
        {"mutation": None, "coverage": 50.0, "test_count": 10,
         "snapshot_clock_s": 1.0},
    )
    real_key = ah.green_snapshot_key(
        {"mutation": 0.0, "coverage": 50.0, "test_count": 10,
         "snapshot_clock_s": 1.0},
    )
    assert real_key > none_key


def test_green_snapshot_key_orders_by_mutation_then_coverage_only() -> None:
    """Same mutation, higher coverage wins. Equal scores give equal keys
    whatever the test count or capture time, so ``update_best_green`` keeps
    the earlier snapshot."""
    base = {"mutation": 75.0, "coverage": 80.0, "test_count": 10,
            "snapshot_clock_s": 5.0}
    higher_cov = {**base, "coverage": 85.0}
    fewer_tests = {**base, "test_count": 5}
    earlier_time = {**base, "snapshot_clock_s": 1.0}
    assert ah.green_snapshot_key(higher_cov) > ah.green_snapshot_key(base)
    assert ah.green_snapshot_key(fewer_tests) == ah.green_snapshot_key(base)
    assert ah.green_snapshot_key(earlier_time) == ah.green_snapshot_key(base)
    assert ah.green_snapshot_key({**base, "mutation": 76.0, "coverage": 1.0}) > (
        ah.green_snapshot_key(higher_cov))


def test_green_snapshot_key_does_not_rank_by_one_runs_time() -> None:
    """At equal mutation and coverage, a recorded ``exec_time_ms`` (one
    run's elapsed time) and the test count order nothing."""
    slow_few = {"mutation": 75.0, "coverage": 80.0,
                "exec_time_ms": 900.0, "test_count": 3,
                "snapshot_clock_s": 1.0}
    fast_many = {"mutation": 75.0, "coverage": 80.0,
                 "exec_time_ms": 100.0, "test_count": 30,
                 "snapshot_clock_s": 1.0}
    assert ah.green_snapshot_key(slow_few) == ah.green_snapshot_key(fast_many)
    measured = {"mutation": 75.0, "coverage": 80.0,
                "exec_time_ms": 5000.0, "test_count": 1,
                "snapshot_clock_s": 1.0}
    unmeasured = {"mutation": 75.0, "coverage": 80.0,
                  "exec_time_ms": None, "test_count": 1,
                  "snapshot_clock_s": 1.0}
    assert ah.green_snapshot_key(measured) == ah.green_snapshot_key(unmeasured)


def test_green_snapshot_key_none_snapshot_sorts_below_realistic_scores() -> None:
    """A None snapshot sorts below any non-negative measurement, so
    ``update_best_green`` accepts the first real snapshot."""
    realistic = {"mutation": 0.0, "coverage": 0.0, "test_count": 0,
                 "snapshot_clock_s": 0.0}
    assert ah.green_snapshot_key(realistic) > ah.green_snapshot_key(None)


def test_update_best_green_first_call_takes_snapshot(tmp_path: Path) -> None:
    """A fresh state takes the snapshot and writes the copy
    ``.best_passing_<name>.bak`` next to the test file."""
    test_file = tmp_path / "test_foo.py"
    test_file.write_text("def test_x(): assert True\n", encoding="utf-8")
    state = ah.BestGreenState()
    took = ah.update_best_green(
        state, test_file, coverage=80.0, mutation=70.0, test_count=1,
    )
    assert took is True
    assert state.snapshot is not None
    assert state.snapshot["mutation"] == 70.0
    backup = tmp_path / ".best_passing_test_foo.py.bak"
    assert backup.exists()
    assert backup.read_text(encoding="utf-8") == test_file.read_text(encoding="utf-8")


def test_update_best_green_java_sidecar_does_not_end_in_java(tmp_path: Path) -> None:
    """A Java sidecar ends in ``.bak``, not ``.java``, since surefire's
    ``*Test.java`` include would compile it as a duplicate class."""
    test_file = tmp_path / "H264EncoderMutationTest.java"
    test_file.write_text(
        "public class H264EncoderMutationTest {}\n", encoding="utf-8",
    )
    state = ah.BestGreenState()
    took = ah.update_best_green(state, test_file, mutation=70.0, test_count=1)
    assert took is True
    sidecar = tmp_path / ".best_passing_H264EncoderMutationTest.java.bak"
    assert sidecar.exists()
    assert not sidecar.name.endswith(".java")
    java_files = [p.name for p in tmp_path.iterdir() if p.name.endswith(".java")]
    assert java_files == ["H264EncoderMutationTest.java"]


def test_update_best_green_worse_score_does_not_replace(tmp_path: Path) -> None:
    """A worse measurement returns False and keeps the snapshot."""
    test_file = tmp_path / "test_foo.py"
    test_file.write_text("def test_x(): assert True\n", encoding="utf-8")
    state = ah.BestGreenState()
    ah.update_best_green(state, test_file, mutation=80.0)
    first_snap = state.snapshot
    test_file.write_text("def test_y(): assert True\n", encoding="utf-8")
    took = ah.update_best_green(state, test_file, mutation=60.0)
    assert took is False
    assert state.snapshot is first_snap


def test_update_best_green_missing_file_returns_false(tmp_path: Path) -> None:
    """A missing file returns False rather than raising."""
    state = ah.BestGreenState()
    took = ah.update_best_green(
        state, tmp_path / "missing.py", coverage=80.0, mutation=70.0,
    )
    assert took is False
    assert state.snapshot is None


def test_update_best_green_persist_disabled(tmp_path: Path) -> None:
    """With ``persist_to_disk=False`` only the in-memory snapshot updates."""
    test_file = tmp_path / "test_foo.py"
    test_file.write_text("ok\n", encoding="utf-8")
    state = ah.BestGreenState()
    ah.update_best_green(state, test_file, mutation=70.0, persist_to_disk=False)
    assert list(tmp_path.glob(".best_passing_*")) == []
    assert state.snapshot is not None


# ---------------------------------------------------------------------------
# assert_write_under -- the write gate of the test output directory
# ---------------------------------------------------------------------------


def test_assert_write_under_accepts_in_sandbox(tmp_path):
    """Writes inside the test output directory are allowed."""
    output_dir = tmp_path / "outdir" / "_work" / "rank1" / "tests"
    target = output_dir / "pkg" / "test_mod.py"
    (output_dir / "pkg").mkdir(parents=True)
    target.write_text("x = 1\n")
    ah.assert_write_under(str(target), str(output_dir))


def test_assert_write_under_refuses_escape(tmp_path):
    """Any target outside ``test_output_dir`` is refused."""
    output_dir = tmp_path / "outdir" / "_work" / "rank1" / "tests"
    output_dir.mkdir(parents=True)
    canonical = tmp_path / "projects" / "myproj" / "src" / "mod.py"
    canonical.parent.mkdir(parents=True)
    canonical.write_text("x = 1\n")
    with pytest.raises(ah.SandboxEscapeError):
        ah.assert_write_under(str(canonical), str(output_dir))


def test_assert_write_under_refuses_parent_escape(tmp_path):
    """A ``..`` path is refused when its resolved target leaves the test
    output directory."""
    output_dir = tmp_path / "outdir" / "_work" / "rank1" / "tests"
    sibling = tmp_path / "outdir" / "_work" / "rank2" / "x.py"
    sibling.parent.mkdir(parents=True)
    sibling.write_text("")
    sneaky = output_dir / ".." / ".." / "rank2" / "x.py"
    output_dir.mkdir(parents=True)
    with pytest.raises(ah.SandboxEscapeError):
        ah.assert_write_under(str(sneaky), str(output_dir))


def test_assert_write_under_refuses_empty_test_output_dir(tmp_path):
    """An empty ``test_output_dir`` raises rather than allowing the write."""
    with pytest.raises(ah.SandboxEscapeError):
        ah.assert_write_under(str(tmp_path / "f.py"), "")


def test_the_sandbox_refusals_name_the_test_output_dir(tmp_path):
    """The refusal the loop returns to the model names the field the check
    receives, ``test_output_dir`` (the agent's sandbox root), and the test
    output directory. An escape names what it protects as the guard's reply
    does, the canonical source."""
    output_dir = tmp_path / "tests"
    output_dir.mkdir()
    with pytest.raises(ah.SandboxEscapeError) as empty:
        ah.assert_write_under(str(tmp_path / "f.py"), "")
    with pytest.raises(ah.SandboxEscapeError) as escape:
        ah.assert_write_under(str(tmp_path / "f.py"), str(output_dir))
    for refusal in (str(empty.value), str(escape.value)):
        assert "test_output_dir" in refusal and "test output folder" in refusal
        for retired in ("source_dir", "project_root", "work zone"):
            assert retired not in refusal
    assert "canonical source" in str(escape.value)
    assert "project's source" not in str(escape.value)


def test_assert_write_under_accepts_production_test_layout(
    tmp_path,
):
    """In the real layout the tests directory is a sibling of the project
    copy, so test writes are accepted against ``test_output_dir`` and
    refused against ``source_dir``."""
    work_dir = tmp_path / "_work" / "rank1"
    work_src = work_dir / "myproj"  # the agent's "source_dir"
    work_tests = work_dir / "tests"  # the agent's "test_output_dir"
    work_src.mkdir(parents=True)
    work_tests.mkdir(parents=True)

    test_file = work_tests / "test_foo.py"
    ah.assert_write_under(str(test_file), str(work_tests))
    with pytest.raises(ah.SandboxEscapeError):
        ah.assert_write_under(str(test_file), str(work_src))


# ---------------------------------------------------------------------------
# unsound_async_rejection -- the strict-green guard, shared by every agent
# ---------------------------------------------------------------------------

def _run(returncode, unsound):
    """A run result with ``returncode`` and the ``unsound`` entries its
    diagnostics gave: (the test as printed or None, the matched text)."""
    return SimpleNamespace(returncode=returncode, unsound=tuple(unsound))


def test_unsound_async_rejection_none_when_returncode_nonzero():
    """A red run is left alone: the guard only turns green into red."""
    run = _run(1, [("t", "w")])
    assert ah.unsound_async_rejection(run, lambda k, **kw: "MSG") is None


def test_unsound_async_rejection_none_when_green_and_clean():
    """A clean green run (no warnings) is not reclassified."""
    assert ah.unsound_async_rejection(_run(0, []), lambda k, **kw: "MSG") is None


def test_unsound_async_rejection_names_tests_when_green_with_warnings():
    """A green run with warnings gets the red message naming the offending
    tests."""
    run = _run(0, [("test_foo", "coroutine never awaited")])
    captured = {}

    def tool_msg(key, **kw):
        captured["key"] = key
        return "RED:\n" + kw["sources_block"]

    out = ah.unsound_async_rejection(run, tool_msg)
    assert out is not None
    assert captured["key"] == "tests_strict_green_unsound"
    assert "test_foo" in out and "coroutine never awaited" in out


def test_unsound_async_rejection_fallback_when_no_sources():
    """Without test names the message lists the matched warnings."""
    run = _run(0, [(None, "leaked task A"), (None, "leaked task B")])
    out = ah.unsound_async_rejection(run, lambda k, **kw: kw["sources_block"])
    assert out is not None
    assert "leaked task A" in out and "test names not parsed" in out


# ---------------------------------------------------------------------------
# log_test_summary: shared per-run console summary.
# ---------------------------------------------------------------------------

def _capture_log():
    """A (messages, log_fn) pair: log_fn records each message for assertions."""
    msgs: list = []

    def log_fn(message, level=0):
        msgs.append(message)

    return msgs, log_fn


def test_log_test_summary_structured_res_beats_text():
    """Structured counts win over parsing Maven-shaped text, so a failing
    run is not logged as having no results."""
    res = run_result(
        ok=False, returncode=1, elapsed_s=0.4, collected=12, passed=10, failed=2,
        output="[INFO] Tests run: 12, Failures: 2\n[INFO] BUILD FAILURE\n",
    )
    msgs, log_fn = _capture_log()
    _test_output.log_test_summary(res.output, res.returncode, log_fn, res=res)
    combined = "\n".join(msgs)
    assert "no test results" not in combined.lower()
    assert "FAILED" in combined


def test_log_test_summary_maven_exit_zero_with_failures_is_red():
    """Exit 0 with structured failures (Maven's failure.ignore) is FAILED."""
    res = run_result(ok=False, returncode=0, elapsed_s=1.0, collected=47, passed=42,
                     failed=4, errors=1)
    msgs, log_fn = _capture_log()
    _test_output.log_test_summary(res.output, 0, log_fn, res=res)
    assert "FAILED" in "\n".join(msgs)


def test_log_test_summary_collected_zero_with_outcomes_uses_derived_total():
    """With collected=0 but passed>0 (pytest -q), the total comes from the
    outcome counts, so a green run is not logged as collecting no tests."""
    res = run_result(elapsed_s=1.0, collected=0, passed=5,
                     output="..... [100%]\n5 passed in 1.23s")
    msgs, log_fn = _capture_log()
    _test_output.log_test_summary(res.output, 0, log_fn, res=res)
    combined = "\n".join(msgs)
    assert "No tests were collected" not in combined
    assert "PASSED" in combined


def test_log_test_summary_truly_empty_warns():
    """collected=0 and no outcomes with a non-zero exit -> a no-results warning."""
    res = run_result(ok=False, returncode=5, elapsed_s=0.1, output="no tests ran in 0.10s")
    msgs, log_fn = _capture_log()
    _test_output.log_test_summary(res.output, 5, log_fn, res=res)
    combined = "\n".join(msgs).lower()
    assert "no test results" in combined or "no tests were collected" in combined


# ---------------------------------------------------------------------------
# Metric-first best-green comparator
# ---------------------------------------------------------------------------

def test_green_snapshot_key_stays_metric_first():
    """Mutation ranks before coverage, and None sorts lowest."""
    higher_mutation = {"mutation": 80.0, "coverage": 90.0, "test_count": 10}
    lower_mutation = {"mutation": 70.0, "coverage": 95.0, "test_count": 5}
    assert ah.green_snapshot_key(higher_mutation) > ah.green_snapshot_key(lower_mutation)
    assert ah.green_snapshot_key(None) < ah.green_snapshot_key(lower_mutation)


# ``testuary.agent_helpers.tool_msg``, which looks up a key in
# ``prompts['tool_messages']`` and substitutes runtime values with
# ``string.Template``. Every agent calls it through a per-module ``_tool_msg``
# wrapper.
def _every_loaded_agent_prompts():
    """Yield ``(agent_name, prompts_dict)`` for every agent that ships
    a prompts JSON."""
    for name in ("test_coverage", "test_mutation", "test_initial"):
        yield name, load_agent_prompts(name)


def test_tool_msg_with_substitution() -> None:
    """``string.Template`` substitution with ``$var``."""
    prompts = {"tool_messages": {"k": "hello $name"}}
    assert tool_msg(prompts, "k", name="world") == "hello world"


def test_tool_msg_with_braced_substitution() -> None:
    """The ``${var}`` form serves names that touch the surrounding text."""
    prompts = {"tool_messages": {"k": "${count}items"}}
    assert tool_msg(prompts, "k", count=42) == "42items"


def test_tool_msg_missing_key_raises_keyerror() -> None:
    """A wrong key raises ``KeyError`` instead of returning ``""``."""
    prompts = {"tool_messages": {"k": "x"}}
    with pytest.raises(KeyError):
        tool_msg(prompts, "missing_key")


def test_tool_msg_no_tool_messages_section_raises_keyerror() -> None:
    """Without a ``tool_messages`` key, the lookup in the empty default
    raises ``KeyError`` too."""
    prompts = {"system_prompt": "..."}
    with pytest.raises(KeyError):
        tool_msg(prompts, "k")


def test_tool_msg_without_values_writes_one_dollar_and_refuses_a_placeholder() -> None:
    """Without keyword arguments the text is substituted too, so ``$$``
    writes one dollar sign as with values, and a ``$name`` left in the text
    is loud (``KeyError``) instead of reaching the model."""
    assert tool_msg({"tool_messages": {"k": "costs $$5"}}, "k") == "costs $5"
    with pytest.raises(KeyError):
        tool_msg({"tool_messages": {"k": "hello $name"}}, "k")


def test_tool_msg_subs_with_missing_placeholder_raises() -> None:
    """With keyword arguments, ``Template.substitute`` runs and raises
    ``KeyError`` for a ``$var`` missing from them."""
    prompts = {"tool_messages": {"k": "hello $name, age $age"}}
    with pytest.raises(KeyError):
        tool_msg(prompts, "k", name="world")  # missing age=


@pytest.mark.parametrize("agent_name,prompts", list(_every_loaded_agent_prompts()))
def test_every_shipped_agent_tool_message_resolves(
    agent_name: str, prompts: dict
) -> None:
    """Every ``tool_messages`` value in every shipped prompts JSON is a
    non-empty string."""
    tm = prompts.get("tool_messages")
    if not tm:
        pytest.skip(f"{agent_name} has no tool_messages section")
    for key, value in tm.items():
        assert isinstance(value, str)
        assert value, f"{agent_name}:tool_messages.{key} is empty"


#: The two error windows' full replies, as the code before the shared
#: window function gave them: an error line in the middle, on the first
#: and last lines, past the end, without a column and in an empty file.
_WINDOW_CASES = [
    ('middle', 'a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\n', 3, 2,
     'bad\n\n     1: a = 1\n     2: b = 2\n\u2192    3: c = 3\n        ^\n     4: d = 4\n     5: e = 5',
     'invalid: bad\n\n     1: a = 1\n     2: b = 2\n>    3: c = 3\n        ^\n     4: d = 4\n     5: e = 5'),
    ('first', 'a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\n', 1, 1,
     'bad\n\n\u2192    1: a = 1\n       ^\n     2: b = 2\n     3: c = 3',
     'invalid: bad\n\n>    1: a = 1\n       ^\n     2: b = 2\n     3: c = 3'),
    ('last', 'a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\n', 6, 6,
     'bad\n\n     4: d = 4\n     5: e = 5\n\u2192    6: f = 6\n            ^',
     'invalid: bad\n\n     4: d = 4\n     5: e = 5\n>    6: f = 6\n            ^'),
    ('past_end', 'a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\n', 8, 3,
     'bad\n\n     6: f = 6',
     'invalid: bad\n\n     6: f = 6'),
    ('no_column', 'a = 1\nb = 2\nc = 3\nd = 4\ne = 5\nf = 6\n', 4, 0,
     'bad\n\n     2: b = 2\n     3: c = 3\n\u2192    4: d = 4\n     5: e = 5\n     6: f = 6',
     'invalid: bad\n\n     2: b = 2\n     3: c = 3\n>    4: d = 4\n     5: e = 5\n     6: f = 6'),
    ('empty', '', 1, 1,
     'bad\n\n(empty file)',
     'invalid: bad\n\n(empty file)'),
]


@pytest.mark.parametrize("name, content, line, col, structure, validation", _WINDOW_CASES)
def test_the_error_windows_reply_byte_for_byte(monkeypatch, name, content, line, col,
                                                structure, validation):
    """The structural parser's and the target validator's error windows
    reach the model, so their text is pinned for each place of the error
    line, each with its own marker."""
    from types import SimpleNamespace

    from testuary import languages

    result = SimpleNamespace(ok=False, line=line, col=col, message="bad", snippet=None)
    monkeypatch.setattr(languages, "get_backend", lambda _language: SimpleNamespace(
        check_structure=lambda _content: result))
    assert ah.source_structure_error(content, "python") == structure
    assert ah.source_validation_error(content, SimpleNamespace(
        ok=False, line=line, col=col, status="invalid", message="bad")) == validation
