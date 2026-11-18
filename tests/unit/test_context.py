"""Tests for the context-reduction policy in ``testuary.agent._context``: the
pair-safe drop boundary at both ends, latest-thinking-only, read rendering
and paging, read deduplication and the context hook.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import (
    AIMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from testuary.agent import _context as context


def _ai_call(call_id, name="run_tests"):
    return AIMessage(content="", tool_calls=[{"name": name, "args": {}, "id": call_id}])


def _tool(call_id, content="result", name="run_tests"):
    return ToolMessage(content=content, tool_call_id=call_id, name=name)


# ---------------------------------------------------------------------------
# drop_oldest_pair_safe
# ---------------------------------------------------------------------------


def test_drop_keeps_all_when_under_limit():
    msgs = [HumanMessage(content="go"), _ai_call("c1"), _tool("c1")]
    out = context.drop_oldest_pair_safe(msgs, keep_last=10)
    assert out == msgs


def test_drop_keeps_system_messages_always():
    sys = SystemMessage(content="instructions")
    msgs = [sys] + [HumanMessage(content=f"m{i}") for i in range(20)]
    out = context.drop_oldest_pair_safe(msgs, keep_last=5)
    # The system message stays in front of the last 5 humans.
    assert out[0] is sys
    assert len(out) == 6
    assert out[-1].content == "m19"


def test_drop_does_not_start_with_orphaned_tool_result():
    # A cut that leaves a tool result without its call drops that result.
    msgs = [
        HumanMessage(content="go"),
        _ai_call("c1"),
        _tool("c1"),       # this pair would be split by keep_last=2
        _ai_call("c2"),
        _tool("c2"),
    ]
    context.drop_oldest_pair_safe(msgs, keep_last=2)
    out3 = context.drop_oldest_pair_safe(msgs, keep_last=4)
    # The task prompt plus the last 3 would start with _tool("c1"), whose
    # call is gone, so it is dropped.
    assert out3[0].content == "go"
    assert not isinstance(out3[1], ToolMessage)
    assert out3[1].tool_calls[0]["id"] == "c2"
    assert isinstance(out3[-1], ToolMessage)
    assert out3[-1].tool_call_id == "c2"


def test_drop_handles_multi_call_group_sliced_at_start():
    # The cut strands the second result of a two-call message.
    msgs = [
        HumanMessage(content="go"),
        AIMessage(content="", tool_calls=[
            {"name": "a", "args": {}, "id": "x"},
            {"name": "b", "args": {}, "id": "y"},
        ]),
        _tool("x", name="a"),
        _tool("y", name="b"),
        _ai_call("z"),
        _tool("z"),
    ]
    out = context.drop_oldest_pair_safe(msgs, keep_last=4)
    # The leading orphan _tool("y") is dropped.
    assert out[0].content == "go"
    assert not isinstance(out[1], ToolMessage)


def test_drop_does_not_end_with_unanswered_call():
    msgs = [
        HumanMessage(content="go"),
        _ai_call("c1"),
        _tool("c1"),
        _ai_call("c2"),  # unanswered, so dropped
    ]
    out = context.drop_oldest_pair_safe(msgs, keep_last=10)
    assert not (isinstance(out[-1], AIMessage) and out[-1].tool_calls)
    assert isinstance(out[-1], ToolMessage)


@pytest.mark.parametrize("keep_last", [1, 2, 3, 4, 5, 6, 30])
def test_drop_keeps_the_task_prompt_as_the_first_user_turn(keep_last):
    # System prompt, task prompt, then tool rounds. Without the task prompt
    # the window would have no user turn and no task.
    sys = SystemMessage(content="instructions")
    task = HumanMessage(content="Write tests for retry.py.")
    msgs = [sys, task]
    for i in range(40):
        msgs += [_ai_call(f"c{i}"), _tool(f"c{i}")]
    out = context.drop_oldest_pair_safe(msgs, keep_last=keep_last)
    assert out[0] is sys
    assert out[1] is task
    convo = out[1:]
    assert len(convo) <= keep_last
    assert [m for m in convo if isinstance(m, HumanMessage)] == [task]
    calls = {tc["id"] for m in convo if isinstance(m, AIMessage) for tc in m.tool_calls}
    results = {m.tool_call_id for m in convo if isinstance(m, ToolMessage)}
    assert calls == results
    assert not context.carries_tool_calls(out[-1])


def test_drop_zero_keeps_only_systems():
    sys = SystemMessage(content="s")
    msgs = [sys, HumanMessage(content="go"), _ai_call("c1"), _tool("c1")]
    out = context.drop_oldest_pair_safe(msgs, keep_last=0)
    assert out == [sys]


def test_drop_pops_trailing_malformed_call():
    # A call with invalid JSON args lands in ``invalid_tool_calls`` but is
    # still serialized as a tool call, so left at the end it would be an
    # unanswered call that the provider rejects.
    malformed = AIMessage(content="", invalid_tool_calls=[
        {"name": "run_tests", "args": "{not json", "id": "bad1",
         "error": "Unterminated string", "type": "invalid_tool_call"},
    ])
    msgs = [HumanMessage(content="go"), _ai_call("c1"), _tool("c1"), malformed]
    out = context.drop_oldest_pair_safe(msgs, keep_last=10)
    assert isinstance(out[-1], ToolMessage)
    assert out[-1].tool_call_id == "c1"


def test_drop_pops_trailing_raw_sidecar_call():
    # Same rule for the raw ``additional_kwargs`` sidecar the serializer
    # falls back to when no parsed calls exist.
    raw = AIMessage(content="", additional_kwargs={"tool_calls": [
        {"id": "raw1", "type": "function",
         "function": {"name": "run_tests", "arguments": "{}"}},
    ]})
    msgs = [HumanMessage(content="go"), _ai_call("c1"), _tool("c1"), raw]
    out = context.drop_oldest_pair_safe(msgs, keep_last=10)
    assert isinstance(out[-1], ToolMessage)
    assert out[-1].tool_call_id == "c1"


def test_carries_tool_calls_predicate():
    assert context.carries_tool_calls(_ai_call("c1"))
    assert context.carries_tool_calls(
        AIMessage(content="", invalid_tool_calls=[
            {"name": "t", "args": "{", "id": "i", "error": None,
             "type": "invalid_tool_call"}]))
    assert context.carries_tool_calls(
        AIMessage(content="", additional_kwargs={"tool_calls": [{}]}))
    assert not context.carries_tool_calls(AIMessage(content="plain"))
    assert not context.carries_tool_calls(HumanMessage(content="x"))
    assert not context.carries_tool_calls(_tool("c1"))


# ---------------------------------------------------------------------------
# strip_old_thinking
# ---------------------------------------------------------------------------


def test_strip_clears_old_reasoning_sidecar_keeps_latest():
    old = AIMessage(content="old", additional_kwargs={"reasoning_content": "long old think"})
    latest = AIMessage(content="new", additional_kwargs={"reasoning_content": "current think"})
    out = context.strip_old_thinking([HumanMessage(content="go"), old, latest])
    assert out[1].additional_kwargs.get("reasoning_content") in (None, "")
    assert out[2].additional_kwargs.get("reasoning_content") == "current think"


def test_strip_removes_inline_think_from_old_ai():
    old = AIMessage(content="<think>" + "z" * 5000 + "</think>answer")
    latest = AIMessage(content="finishing")
    out = context.strip_old_thinking([old, latest])
    assert "<think>" not in out[0].content
    assert out[0].content == "answer"
    assert out[1].content == "finishing"


def test_strip_noop_when_no_ai_messages():
    msgs = [HumanMessage(content="go"), ToolMessage(content="r", tool_call_id="c1")]
    out = context.strip_old_thinking(msgs)
    assert out == msgs


# ---------------------------------------------------------------------------
# render_read
# ---------------------------------------------------------------------------


def test_render_small_file_numbered_whole():
    content = "import os\n\ndef foo():\n    return 1\n"
    out = context.render_read(content, name="foo.py", max_chars=10000)
    assert "1\timport os" in out
    assert "3\tdef foo():" in out
    assert "lines 1-" in out


def test_a_final_newline_does_not_count_as_a_line():
    out = context.render_read("a\nb\n", name="two.py", max_chars=None)
    assert out == "# two.py (2 lines)\n1\ta\n2\tb\n[lines 1-2 of 2]"
    assert context.render_read("a\nb", name="two.py", max_chars=None) == out


def test_render_paging_marker_when_over_cap():
    content = "\n".join(f"line {i}" for i in range(1, 1001))
    out = context.render_read(content, name="big.py", max_chars=200)
    assert "[lines 1-19 of 1000]\n[... 981 lines of big.py omitted, call this read tool " \
        "with start_line=20 for the next part ...]" in out
    assert "981 lines" not in context.render_read(content, name="big.py", max_chars=None)


def test_render_start_line_slice():
    content = "\n".join(f"line {i}" for i in range(1, 21))
    out = context.render_read(content, name="f.py", max_chars=10000, start_line=10)
    assert "10\tline 10" in out
    assert "9\tline 9" not in out


def test_render_end_line_slice():
    content = "\n".join(f"line {i}" for i in range(1, 21))
    out = context.render_read(content, name="f.py", max_chars=10000, start_line=5, end_line=8)
    assert "5\tline 5" in out
    assert "8\tline 8" in out
    assert "9\tline 9" not in out
    assert "lines 5-8 of 20" in out


def test_render_start_past_eof_self_corrects():
    content = "a\nb\nc"
    out = context.render_read(content, name="f.py", max_chars=10000, start_line=99)
    assert "past the end" in out
    assert "3 lines" in out


def test_render_clamps_nonpositive_start():
    content = "a\nb\nc"
    out = context.render_read(content, name="f.py", max_chars=10000, start_line=0)
    assert "1\ta" in out


def test_render_single_huge_line_still_returns_progress():
    content = "x" * 5000
    out = context.render_read(content, name="f.py", max_chars=200, start_line=1)
    # A line larger than the cap is still returned, never an empty body.
    assert "1\t" in out


# ---------------------------------------------------------------------------
# dedup_file_payloads
# ---------------------------------------------------------------------------

_READ = {"read_source"}


def _fkey(name, args):
    return "source" if "source" in name else "test"


def _stub(fk):
    return f"[Superseded earlier read of {fk}. <<<TESTUARY_ELIDED>>>]"


def _read_call(cid, name="read_source", **args):
    return AIMessage(content="", tool_calls=[{"name": name, "args": args, "id": cid}])


def _read_ret(cid, content, name="read_source", shown="1-100 of 100"):
    """A read's answer with render_read's last line, ``shown`` its lines."""
    return ToolMessage(content=f"{content}\n[lines {shown}]", tool_call_id=cid, name=name)


def _dedup(msgs):
    return context.dedup_file_payloads(
        msgs, read_tools=_READ, file_key=_fkey, stub=_stub
    )


def test_dedup_whole_read_superseded_by_later_whole_read():
    msgs = [
        _read_call("c1"),
        _read_ret("c1", "FULL CONTENT V1 " * 100),
        _read_call("c2"),
        _read_ret("c2", "FULL CONTENT V2 " * 100),
    ]
    out = _dedup(msgs)
    assert "<<<TESTUARY_ELIDED>>>" in out[1].content
    assert "FULL CONTENT V2" in out[3].content


def test_a_read_cut_at_its_size_limit_covers_only_its_shown_lines():
    # A whole-file call whose answer stopped at line 19 leaves an earlier
    # page of lines 60-80 in place.
    msgs = [
        _read_call("c1", start_line=60, end_line=80),
        _read_ret("c1", "PAGE " * 20, shown="60-80 of 1000"),
        _read_call("c2"),
        _read_ret("c2", "HEAD " * 50, shown="1-19 of 1000"),
    ]
    out = _dedup(msgs)
    assert "PAGE" in out[1].content


def test_a_read_covers_the_lines_its_answer_shows_whatever_its_arguments():
    # An end_line before start_line shows the file from start_line to its
    # end, so the read covers an earlier page inside that part.
    msgs = [
        _read_call("c1", start_line=60, end_line=80),
        _read_ret("c1", "PAGE " * 20, shown="60-80 of 100"),
        _read_call("c2", start_line=50, end_line=10),
        _read_ret("c2", "TAIL " * 50, shown="50-100 of 100"),
    ]
    out = _dedup(msgs)
    assert "<<<TESTUARY_ELIDED>>>" in out[1].content


def test_dedup_page_superseded_by_covering_page():
    # A later 5-40 read covers an earlier 10-20 read.
    msgs = [
        _read_call("c1", start_line=10, end_line=20),
        _read_ret("c1", "INNER " * 50, shown="10-20 of 300"),
        _read_call("c2", start_line=5, end_line=40),
        _read_ret("c2", "OUTER " * 50, shown="5-40 of 300"),
    ]
    out = _dedup(msgs)
    assert "<<<TESTUARY_ELIDED>>>" in out[1].content
    assert "OUTER" in out[3].content


def test_dedup_distinct_pages_coexist():
    msgs = [
        _read_call("c1", start_line=1, end_line=100),
        _read_ret("c1", "PAGE ONE " * 50, shown="1-100 of 300"),
        _read_call("c2", start_line=101, end_line=200),
        _read_ret("c2", "PAGE TWO " * 50, shown="101-200 of 300"),
    ]
    out = _dedup(msgs)
    assert "PAGE ONE" in out[1].content
    assert "PAGE TWO" in out[3].content


def test_dedup_whole_read_supersedes_prior_page():
    msgs = [
        _read_call("c1", start_line=1, end_line=100),
        _read_ret("c1", "OLD PAGE " * 50, shown="1-100 of 300"),
        _read_call("c2"),  # whole read
        _read_ret("c2", "WHOLE FILE " * 100, shown="1-300 of 300"),
    ]
    out = _dedup(msgs)
    assert "<<<TESTUARY_ELIDED>>>" in out[1].content
    assert "WHOLE FILE" in out[3].content


def test_dedup_same_region_reread_supersedes():
    msgs = [
        _read_call("c1", start_line=50, end_line=80),
        _read_ret("c1", "OLD REGION " * 20, shown="50-80 of 300"),
        _read_call("c2", start_line=50, end_line=80),
        _read_ret("c2", "NEW REGION " * 20, shown="50-80 of 300"),
    ]
    out = _dedup(msgs)
    assert "<<<TESTUARY_ELIDED>>>" in out[1].content
    assert "NEW REGION" in out[3].content


def test_dedup_different_files_independent():
    msgs = [
        _read_call("c1", name="read_source"),
        _read_ret("c1", "SOURCE " * 100, name="read_source"),
        _read_call("c2", name="read_test_file"),
        _read_ret("c2", "TEST " * 100, name="read_test_file"),
    ]
    out = context.dedup_file_payloads(
        msgs,
        read_tools={"read_source", "read_test_file"},
        file_key=_fkey,
        stub=_stub,
    )
    assert "SOURCE" in out[1].content
    assert "TEST" in out[3].content


def test_dedup_preserves_message_count_and_pairing():
    msgs = [
        _read_call("c1"),
        _read_ret("c1", "V1 " * 100),
        _read_call("c2"),
        _read_ret("c2", "V2 " * 100),
    ]
    out = _dedup(msgs)
    assert len(out) == len(msgs)
    assert out[1].tool_call_id == "c1"
    assert out[3].tool_call_id == "c2"


# ---------------------------------------------------------------------------
# make_context_hook
# ---------------------------------------------------------------------------


def test_dedup_stub_carries_sentinel_and_filekey():
    out = context.dedup_stub("source")
    assert "<<<TESTUARY_ELIDED>>>" in out
    assert "source" in out
    # The write guards recognize it as a marker.
    from testuary.agent import _helpers as _h
    assert _h.contains_history_marker(out)


def test_task_clause_renders_file_only():
    # The clause names only the target file.
    out = context.task_clause("/abs/path/to/foo.py")
    assert out == "file=foo.py"


def test_task_clause_basename_from_windows_path():
    assert context.task_clause("c:\\\\win\\\\bar.py") == "file=bar.py"


def test_task_clause_missing_target():
    assert context.task_clause(None) == "file=?"


def _marker(_session):
    return "[state | phase=coverage best=80%]"


def test_hook_appends_marker_last():
    hook = context.make_context_hook(_marker)
    state = {"messages": [SystemMessage(content="sys"), HumanMessage(content="task")]}
    out = hook(state)["messages"]
    # The marker is rebuilt every turn, so it goes last and leaves the
    # prefix byte-identical to the previous request.
    assert isinstance(out[-1], SystemMessage)
    assert out[-1].content.startswith("[state |")
    assert [m.content for m in out[:-1]] == ["sys", "task"]


def test_hook_rotates_stale_marker():
    hook = context.make_context_hook(_marker)
    stale = SystemMessage(content="[state | phase=coverage best=10%]")
    state = {"messages": [SystemMessage(content="sys"), stale, HumanMessage(content="task")]}
    out = hook(state)["messages"]
    markers = [m for m in out if isinstance(m, SystemMessage) and m.content.startswith("[state |")]
    assert len(markers) == 1
    assert "best=80%" in markers[0].content


def test_hook_injects_system_prompt_when_missing():
    hook = context.make_context_hook(lambda s: "", system_prompt="SYSTEM")
    state = {"messages": [HumanMessage(content="task")]}
    out = hook(state)["messages"]
    assert isinstance(out[0], SystemMessage) and out[0].content == "SYSTEM"


def test_hook_keep_n_windows_and_keeps_system_and_marker():
    hook = context.make_context_hook(_marker, keep_last=4)
    msgs = [SystemMessage(content="sys")]
    for i in range(20):
        msgs.append(HumanMessage(content=f"m{i}"))
    out = hook({"messages": msgs})["messages"]
    # The system message, the marker, the task prompt and the last 3 humans
    # remain.
    assert any(isinstance(m, SystemMessage) and m.content == "sys" for m in out)
    assert any(m.content.startswith("[state |") for m in out if isinstance(m, SystemMessage))
    humans = [m for m in out if isinstance(m, HumanMessage)]
    assert [h.content for h in humans] == ["m0", "m17", "m18", "m19"]


def test_hook_keep_last_none_does_not_window():
    hook = context.make_context_hook(_marker, keep_last=None)
    msgs = [HumanMessage(content=f"m{i}") for i in range(50)]
    out = hook({"messages": msgs})["messages"]
    humans = [m for m in out if isinstance(m, HumanMessage)]
    assert len(humans) == 50  # nothing dropped (profiling pattern)


def test_hook_applies_dedup_and_strip_thinking():
    msgs = [
        SystemMessage(content="sys"),
        _read_call("c1"),
        _read_ret("c1", "OLD " * 100),
        AIMessage(content="<think>scratch</think>", tool_calls=[]),
        _read_call("c2"),
        _read_ret("c2", "NEW " * 100),
        AIMessage(content="latest"),
    ]
    hook = context.make_context_hook(
        lambda s: "",
        dedup={"read_tools": _READ, "file_key": _fkey, "stub": _stub},
    )
    out = hook({"messages": msgs})["messages"]
    # The first read is stubbed and the old <think> cleared, while the latest
    # message is untouched.
    read_rets = [m for m in out if isinstance(m, ToolMessage)]
    assert "<<<TESTUARY_ELIDED>>>" in read_rets[0].content
    assert all(
        "<think>" not in str(m.content) and "scratch" not in str(m.content)
        for m in out
        if isinstance(m, AIMessage)
    )
    assert any(isinstance(m, AIMessage) and m.content == "latest" for m in out)
