"""The edit tool through the real coverage and mutation graphs: what its
results state about the file, and several edits in one response."""
from __future__ import annotations

import re
from typing import Any, Sequence

import pytest
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import Runnable

from agent_fixtures import mutation_agent, target_agent

from testuary.agent import _langgraph as lg
from testuary.agent import coverage_agent as tc

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

BASE = ("from target import f\n\n\ndef test_a():\n    assert f(1) == 2\n\n\n"
        "def test_b():\n    assert f(2) == 3\n")


class _FakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools: Sequence[Any], *, tool_choice: Any = None, **kwargs: Any) -> Runnable:
        return self


def _coverage(tmp_path):
    return target_agent(tmp_path), tmp_path / "test_target.py"


def _mutation(tmp_path):
    source = "def f(x):\n    return x + 1\n"
    (tmp_path / "target.py").write_text(source, encoding="utf-8")
    agent = mutation_agent(tmp_path, target_file_path=str(tmp_path / "target.py"),
                           target_file_content=source, mutation_test_file_name="test_target_mutation.py",
                           coverage_baseline_pct=80.0)
    return agent, tmp_path / "test_target_mutation.py"


@pytest.fixture(params=["coverage", "mutation"])
def edit_setup(request, tmp_path):
    agent, path = {"coverage": _coverage, "mutation": _mutation}[request.param](tmp_path)
    agent.session.has_read = True
    agent.session.initial_measured = True  # past the start's measurement
    path.write_text(BASE, encoding="utf-8")
    return agent, path


def _respond(agent, calls):
    """One model response holding ``calls``; the tool results in order."""
    calls = [{"name": "edit_tests", "id": f"e{i}", "args": args} for i, args in enumerate(calls)]
    # As run() does, the session holds the graph's usage tracker.
    tracker = agent.session.usage_tracker_ref = lg.UsageTracker(request_limit=1)
    graph = agent.build_graph(model=_FakeModel(responses=[AIMessage(content="", tool_calls=calls)]),
                              usage_tracker=tracker)
    results = []
    with pytest.raises(lg.UsageLimitExceeded):
        for update in graph.stream({"messages": [HumanMessage(content="x")]}, stream_mode="updates"):
            for node in update.values():
                results += [m for m in (node or {}).get("messages", []) if getattr(m, "type", "") == "tool"]
    assert [m.tool_call_id for m in results] == [c["id"] for c in calls]
    return [str(m.content) for m in results]


def _landed(text):
    match = re.search(r"File is now\s+(\d+) lines; edit landed at lines\s+(\d+)-(\d+)", text)
    assert match, text
    return tuple(int(group) for group in match.groups())


def test_a_deletion_reports_where_it_landed(edit_setup):
    agent, path = edit_setup
    (result,) = _respond(agent, [{"old_text": "\n\ndef test_b():\n    assert f(2) == 3\n", "new_text": ""}])
    assert "def test_b" not in path.read_text(encoding="utf-8")
    # The deleted block ended the file: it lands at the last line, not past it.
    assert _landed(result) == (5, 5, 5) and "5\t    assert f(1) == 2" in result


def test_a_deletion_inside_the_file_reports_the_line_after_it(edit_setup):
    agent, path = edit_setup
    (result,) = _respond(agent, [{"old_text": "def test_a():\n    assert f(1) == 2\n\n\n", "new_text": ""}])
    assert _landed(result) == (5, 4, 4) and "4\tdef test_b():" in result


FORM_FEED = ("from target import f\n\x0c\n\ndef test_a():\n    # page\x0cbreak\n    assert f(1) == 2\n\n\n"
             "def test_b():\n    assert f(2) == 3\n")


@pytest.mark.parametrize("form", ["single", "list"])
def test_edit_views_number_lines_as_reads_do(edit_setup, form):
    """A form feed stays inside its line: views, counts and reads agree."""
    agent, path = edit_setup
    path.write_text(FORM_FEED, encoding="utf-8")
    edit = {"old_text": "    assert f(2) == 3\n", "new_text": "    assert f(2) == 3\n    assert f(3) == 4\n"}
    (result,) = _respond(agent, [edit if form == "single" else {"edits": [edit]}])
    assert "9\tdef test_b():" in result and "11\t    assert f(3) == 4" in result, result
    assert "5\t    # page\x0cbreak" in result and "File is now\n11 lines" in result
    if form == "single":
        assert _landed(result) == (11, 10, 11)
    read = agent.read_tests()
    assert "9\tdef test_b():" in read and "11\t    assert f(3) == 4" in read


def test_a_numbered_old_text_with_a_form_feed_lands(edit_setup):
    """Line numbers copied from a view are removed per line."""
    agent, path = edit_setup
    path.write_text(FORM_FEED, encoding="utf-8")
    (result,) = _respond(agent, [{
        "old_text": "5\t    # page\x0cbreak\n6\t    assert f(1) == 2\n",
        "new_text": "5\t    # page\x0cbreak\n6\t    assert f(1) == 2\n7\t    assert f(0) == 1\n"}])
    assert result.startswith("Replaced"), result
    assert "    # page\x0cbreak\n    assert f(1) == 2\n    assert f(0) == 1\n" in path.read_text(encoding="utf-8")


def test_a_loose_match_replaces_the_block_it_matched(edit_setup):
    """The matched block's text also occurs earlier, inside another line."""
    agent, path = edit_setup
    path.write_text("from target import f\n\n\ndef test_a():\n    x = 10\n    assert f(x) == 11\n\n\n"
                    "def test_b():\n    x = 1\n    assert f(x) == 2\n", encoding="utf-8")
    single, = _respond(agent, [{"old_text": "        x = 1", "new_text": "        x = 3"}])
    text = path.read_text(encoding="utf-8")
    assert "    x = 10\n" in text and "def test_b():\n    x = 3\n" in text
    assert _landed(single)[1:] == (10, 10)
    # In a list: the third item's block "    x = 2" also starts "    x = 20".
    _respond(agent, [{"edits": [{"old_text": "        x = 10", "new_text": "        x = 20"},
                                {"old_text": "        x = 3", "new_text": "        x = 2"},
                                {"old_text": "        x = 2", "new_text": "        x = 5"}]}])
    text = path.read_text(encoding="utf-8")
    assert "    x = 20\n" in text and "def test_b():\n    x = 5\n" in text


@pytest.mark.parametrize("old_text", ["```\n```", "6\t"])
@pytest.mark.parametrize("form", ["single", "list"])
def test_an_old_text_empty_after_cleaning_is_refused(edit_setup, old_text, form):
    """An old_text that cleaning empties would match everywhere."""
    agent, path = edit_setup
    edit = {"old_text": old_text, "new_text": "    assert f(0) == 1\n", "replace_all": True}
    (result,) = _respond(agent, [edit if form == "single" else {"edits": [edit]}])
    assert path.read_text(encoding="utf-8") == BASE
    assert "``old_text`` holds only line-number prefixes or markdown" in result, result


def test_an_omitted_new_text_is_refused_not_a_deletion(edit_setup):
    agent, path = edit_setup
    block = "\n\ndef test_b():\n    assert f(2) == 3\n"
    missing, empty = _respond(agent, [{"old_text": block}, {}])
    assert path.read_text(encoding="utf-8") == BASE
    assert missing.startswith("ERROR: ``new_text`` is missing."), missing
    assert empty.startswith("ERROR: no edit given. Pass ``old_text`` and ``new_text`` for one"), empty
    # Deletion stays an explicit "".
    (deleted,) = _respond(agent, [{"old_text": block, "new_text": ""}])
    assert deleted.startswith("Replaced") and "test_b" not in path.read_text(encoding="utf-8")


def test_a_lone_surrogate_in_new_text_is_refused(edit_setup):
    agent, path = edit_setup
    edit = {"old_text": "    assert f(2) == 3\n", "new_text": "    assert f(2) == 3\n    s = '\ud800'\n"}
    single = agent.edit_file(**edit)
    listed = agent.edit_file(edits=[edit])
    assert path.read_text(encoding="utf-8") == BASE
    assert single.startswith("ERROR: ``new_text`` holds a character that cannot be written as")
    assert "at position 30" in single and "a lone surrogate" in listed


def test_a_replacement_whose_text_occurs_earlier_reports_where_it_landed(edit_setup):
    agent, path = edit_setup
    (result,) = _respond(agent, [{"old_text": "    assert f(2) == 3\n", "new_text": "    assert f(1) == 2\n"}])
    assert _landed(result)[1:] == (9, 9)


def test_not_found_and_success_state_the_same_line_count(edit_setup):
    agent, path = edit_setup
    ok, missing = _respond(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"},
        {"old_text": "    assert f(9) == 10\n", "new_text": "    assert f(9) == 11\n"},
    ])
    stated = int(re.search(r"\((\d+) lines on\s+disk\)", missing).group(1))
    assert stated == _landed(ok)[0] == path.read_text(encoding="utf-8").count("\n") == 10


def test_a_chain_anchored_on_an_earlier_insert_lands(edit_setup):
    agent, path = edit_setup
    first, second = _respond(agent, [
        {"old_text": "    assert f(2) == 3\n",
         "new_text": "    assert f(2) == 3\n\n\ndef test_c():\n    assert f(5) == 6\n"},
        {"old_text": "    assert f(5) == 6\n", "new_text": "    assert f(5) == 6\n    assert f(6) == 7\n"},
    ])
    assert "def test_c():\n    assert f(5) == 6\n    assert f(6) == 7\n" in path.read_text(encoding="utf-8")
    assert _landed(first)[0] == 13 and _landed(second)[0] == 14


# Several edits as one list (``edits``).


def _respond_list(agent, edits, **single):
    (result,) = _respond(agent, [dict(single, edits=edits)])
    return result


def test_a_list_lands_a_new_version_before_the_old_one_is_deleted(edit_setup):
    """One at a time, the new test_a is rejected as a duplicate and the
    deletion then removes the only test_a."""
    agent, path = edit_setup
    result = _respond_list(agent, [
        {"old_text": "    assert f(2) == 3\n",
         "new_text": "    assert f(2) == 3\n\n\ndef test_a():\n    assert f(1) == 2\n    assert f(0) == 1\n"},
        {"old_text": "def test_a():\n    assert f(1) == 2\n\n\n", "new_text": ""},
    ])
    text = path.read_text(encoding="utf-8")
    assert result.startswith("Applied all 2 edits"), result
    assert text.count("def test_a") == 1 and "assert f(0) == 1" in text


def test_a_list_lands_a_header_and_its_body(edit_setup):
    agent, path = edit_setup
    _respond_list(agent, [
        {"old_text": "    assert f(2) == 3\n",
         "new_text": "    assert f(2) == 3\n\n\ndef test_c():\n    with open(__file__):\n"},
        {"old_text": "    with open(__file__):\n", "new_text": "    with open(__file__):\n        f(None)\n"},
    ])
    assert "def test_c():\n    with open(__file__):\n        f(None)\n" in path.read_text(encoding="utf-8")


def test_a_list_swaps_two_names(edit_setup):
    agent, path = edit_setup
    _respond_list(agent, [
        {"old_text": "def test_a():", "new_text": "def test_b():"},
        {"old_text": "def test_b():\n    assert f(2) == 3", "new_text": "def test_a():\n    assert f(2) == 3"},
    ])
    text = path.read_text(encoding="utf-8")
    assert "def test_b():\n    assert f(1) == 2" in text and "def test_a():\n    assert f(2) == 3" in text


def test_a_list_applies_all_or_nothing_and_names_the_failing_edit(edit_setup):
    agent, path = edit_setup
    result = _respond_list(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"},
        {"old_text": "    assert f(9) == 10\n", "new_text": "    assert f(9) == 11\n"},
    ])
    assert path.read_text(encoding="utf-8") == BASE
    assert result.startswith("REJECTED: edit 2 of 2 failed, so none of the 2"), result
    assert "ERROR: old_text not found" in result


def test_a_list_runs_the_checks_once_on_the_final_result(edit_setup):
    agent, path = edit_setup
    result = _respond_list(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"},
        {"old_text": "    assert f(2) == 3\n", "new_text": "    assert f(2) == (3\n"},
    ])
    assert path.read_text(encoding="utf-8") == BASE
    assert result.startswith("REJECTED: the result of all 2 edits fails a check"), result
    assert "syntactically invalid" in result


def test_a_syntax_error_reads_as_the_compiler_states_it(edit_setup):
    """A rejected edit names the language's own diagnostic at its place and
    the language's usual causes, not Python advice for every language."""
    agent, path = edit_setup
    result, = _respond(agent, [{"old_text": "    assert f(2) == 3\n",
                                "new_text": "    assert f(2) == (3\n"}])
    assert path.read_text(encoding="utf-8") == BASE
    assert "syntax_error: '(' was never closed" in result
    assert ">    9:     assert f(2) == (3" in result
    assert "Common causes: missing colon at end of def/if/for" in result
    assert "orphan" not in result


def test_both_forms_in_one_call_are_refused(edit_setup):
    agent, path = edit_setup
    result = _respond_list(agent, [{"old_text": "def test_a():", "new_text": "def test_x():"}],
                           old_text="def test_b():", new_text="def test_y():")
    assert path.read_text(encoding="utf-8") == BASE
    assert result.startswith("ERROR: pass either ``old_text`` and ``new_text`` for one edit"), result


def test_a_list_with_a_top_level_replace_all_is_refused(edit_setup):
    agent, path = edit_setup
    result = _respond_list(agent, [{"old_text": "def test_b():", "new_text": "def test_c():"}],
                           replace_all=True)
    assert path.read_text(encoding="utf-8") == BASE
    assert "in a list, ``replace_all`` goes on\nthe item" in result, result


def test_a_list_that_cancels_out_writes_nothing(edit_setup):
    agent, path = edit_setup
    agent.session.last_tests_passed = True
    # Called directly: after a green run the graph offers no edit tool.
    result = agent.edit_file(edits=[
        {"old_text": "def test_b():", "new_text": "def test_c():"},
        {"old_text": "def test_c():", "new_text": "def test_b():"},
    ])
    assert result.startswith("Working file unchanged: ") and "Together the 2 edits leave it" in result
    assert agent.session.writes_done == 0 and agent.session.last_tests_passed is True


def test_a_one_item_list_reads_in_the_singular(edit_setup):
    agent, path = edit_setup
    ok = _respond_list(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"}])
    assert ok.startswith("Applied 1 edit to ") and "View of the changed region (line numbers after\nthe edit," in ok
    assert "the edit landed where" in ok
    failed = _respond_list(agent, [{"old_text": "    assert f(9) == 10\n", "new_text": ""}])
    assert failed.startswith("REJECTED: edit 1 of 1 failed, so it was not applied\n"), failed
    gate = _respond_list(agent, [{"old_text": "    assert f(2) == 3\n", "new_text": "    assert f(2) == (3\n"}])
    assert gate.startswith("REJECTED: the result of the 1 edit fails a check, so it was not applied\n"), gate


def test_a_multi_item_list_reads_in_the_plural(edit_setup):
    """The header's fill-ins are built in code, so the whole text is checked."""
    agent, path = edit_setup
    body = "".join(f"\n\ndef test_x{i}():\n    assert f({i + 10}) == {i + 11}\n" for i in range(6))
    path.write_text(BASE + body, encoding="utf-8")
    result = _respond_list(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"},
        {"old_text": "    assert f(15) == 16\n", "new_text": "    assert f(15) == 16\n    assert f(-5) == -4\n"},
    ])
    n_lines = path.read_text(encoding="utf-8").count("\n")
    assert result.startswith(
        f"Applied all 2 edits in order to {path.name}. File is now\n"
        f"{n_lines} lines. Views of the changed regions (line numbers after\n"
        "all edits, 5 lines of surrounding context):\nEdit 1, lines "), result
    assert "\n\nIf the surrounding context looks right, the edits landed where\n" in result


def test_a_later_list_item_that_matches_several_times_counts_the_working_copy(edit_setup):
    """Edit 1 makes the second copy, which is not on disk."""
    agent, path = edit_setup
    result = _respond_list(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(2) == 3\n"},
        {"old_text": "    assert f(2) == 3\n", "new_text": "    assert f(2) == 4\n"},
    ])
    assert path.read_text(encoding="utf-8") == BASE
    assert f"ERROR: old_text appears 2 times in {path.name} as edit 1\nof this list left it." in result
    assert "set ``replace_all`` on the item" in result and "re-call edit_tests" not in result


def test_a_list_item_that_matches_several_times_is_told_where_replace_all_goes(edit_setup):
    agent, path = edit_setup
    result = _respond_list(agent, [{"old_text": "def test_", "new_text": "def test_x"}])
    assert "appears 2 times" in result and "set ``replace_all`` on the item itself" in result


def test_a_failing_list_item_is_logged_with_its_text(edit_setup, monkeypatch):
    agent, path = edit_setup
    logged = []
    monkeypatch.setattr(agent, "log", lambda message, *args, **kwargs: logged.append(str(message)))
    _respond_list(agent, [{"old_text": "    assert f(9) == 10\n", "new_text": "    assert f(9) == 11\n"}])
    (line,) = [entry for entry in logged if "REJECTED (edits item 1: not found)" in entry]
    assert "old[:120]='    assert f(9) == 10 \\\\n '" in line and "f(9) == 11" in line


def test_an_empty_list_is_refused(edit_setup):
    agent, path = edit_setup
    result = _respond_list(agent, [])
    assert path.read_text(encoding="utf-8") == BASE
    assert result.startswith("ERROR: ``edits`` is empty."), result


def test_an_unchanged_list_item_rejects_the_list(edit_setup):
    agent, path = edit_setup
    result = _respond_list(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"},
        {"old_text": "def test_b():", "new_text": "def test_b():"},
    ])
    assert path.read_text(encoding="utf-8") == BASE
    assert result.startswith("REJECTED: edit 2 of 2 failed"), result


def test_a_list_shows_each_changed_region_in_final_line_numbers(edit_setup):
    agent, path = edit_setup
    body = "".join(f"\n\ndef test_x{i}():\n    assert f({i + 10}) == {i + 11}\n" for i in range(6))
    path.write_text(BASE + body, encoding="utf-8")
    result = _respond_list(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"},
        {"old_text": "    assert f(15) == 16\n", "new_text": "    assert f(15) == 16\n    assert f(-5) == -4\n"},
    ])
    text = path.read_text(encoding="utf-8").splitlines()
    first = text.index("    assert f(0) == 1") + 1
    second = text.index("    assert f(-5) == -4") + 1
    assert f"Edit 1, lines {first - 1}-{first}:" in result, result
    assert f"Edit 2, lines {second - 1}-{second}:" in result, result
    assert f"{second}	    assert f(-5) == -4" in result


def test_chained_list_edits_share_one_region(edit_setup):
    agent, path = edit_setup
    result = _respond_list(agent, [
        {"old_text": "    assert f(2) == 3\n",
         "new_text": "    assert f(2) == 3\n\n\ndef test_c():\n    assert f(5) == 6\n"},
        {"old_text": "    assert f(5) == 6\n", "new_text": "    assert f(5) == 6\n    assert f(6) == 7\n"},
    ])
    assert "Edits 1, 2, lines 9-14:" in result, result


def test_a_list_item_matched_loosely_is_noted(edit_setup):
    agent, path = edit_setup
    result = _respond_list(agent, [
        {"old_text": "      assert f(1) == 2\n", "new_text": "      assert f(1) == 2\n      assert f(0) == 1\n"},
        {"old_text": "    assert f(2) == 3\n", "new_text": "    assert f(2) == 3\n    assert f(3) == 4\n"},
    ])
    assert "    assert f(0) == 1\n" in path.read_text(encoding="utf-8")
    assert "edit 1: whitespace-loose match, re-indented its new_text by -2 chars" in result, result


def test_a_list_item_replacing_every_occurrence_shows_each(edit_setup):
    agent, path = edit_setup
    result = _respond_list(agent, [
        {"old_text": "def test_", "new_text": "def test_renamed_", "replace_all": True},
        {"old_text": "from target import f\n", "new_text": "from target import f  # noqa\n"},
    ])
    text = path.read_text(encoding="utf-8")
    assert text.count("def test_renamed_") == 2
    # Views within eleven lines of each other meet, so they merge.
    assert "Edits 1, 2, lines 1-8:" in result, result


def test_a_list_is_one_write(edit_setup):
    agent, path = edit_setup
    before = agent.session.writes_done
    _respond_list(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"},
        {"old_text": "    assert f(2) == 3\n", "new_text": "    assert f(2) == 3\n    assert f(3) == 4\n"},
    ])
    assert agent.session.writes_done == before + 1
    assert agent.session.last_tests_passed is False


def test_the_tool_offers_the_list_and_keeps_the_single_edit_text(edit_setup):
    from langchain_core.utils.function_calling import convert_to_openai_tool

    agent, path = edit_setup
    (tool,) = agent.phase_tools()[1]
    function = convert_to_openai_tool(tool)["function"]
    # The whole schema reaches the model, the item's docstring included,
    # which neither surface snapshot holds.
    assert function["parameters"] == {
        "properties": {
            "old_text": {"default": "", "type": "string"},
            "new_text": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None},
            "replace_all": {"default": False, "type": "boolean"},
            "edits": {"anyOf": [{"items": {
                "description": "One edit of an ``edits`` list: ``old_text``, ``new_text`` and an\n"
                               "optional ``replace_all``, as for a single edit.",
                "properties": {"old_text": {"title": "Old Text", "type": "string"},
                               "new_text": {"title": "New Text", "type": "string"},
                               "replace_all": {"title": "Replace All", "type": "boolean"}},
                "required": ["old_text", "new_text"],
                "title": "EditItem",
                "type": "object"}, "type": "array"}, {"type": "null"}], "default": None},
        },
        "type": "object",
    }
    description = " ".join(function["description"].split())
    assert "Several edits: pass ``edits``, a list of ``{old_text, new_text}`` items, each with an " \
           "optional ``replace_all``, instead of ``old_text`` and ``new_text``." in description
    assert "Use the list for changes that depend on each other" in description


AFTER_EDIT = "An earlier edit in this reply already changed this file"


def test_a_lone_edit_that_misses_keeps_the_not_found_text(edit_setup):
    agent, path = edit_setup
    (result,) = _respond(agent, [{"old_text": "    assert f(9) == 10\n", "new_text": "    assert f(9) == 11\n"}])
    assert result == agent.tool_msg("edit_not_found", name=path.name, line_count=9, hint="")


def test_a_miss_after_an_earlier_edit_of_the_same_reply_says_so(edit_setup):
    agent, path = edit_setup
    ok, missing = _respond(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"},
        {"old_text": "    assert f(9) == 10\n", "new_text": "    assert f(9) == 11\n"},
    ])
    assert ok.startswith("Replaced") and AFTER_EDIT in missing


def test_a_miss_in_a_later_reply_does_not_say_so(edit_setup):
    agent, path = edit_setup
    first = AIMessage(content="", tool_calls=[{"name": "edit_tests", "id": "e0", "args": {
        "old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"}}])
    second = AIMessage(content="", tool_calls=[{"name": "edit_tests", "id": "e1", "args": {
        "old_text": "    assert f(9) == 10\n", "new_text": "    assert f(9) == 11\n"}}])
    tracker = agent.session.usage_tracker_ref = lg.UsageTracker(request_limit=2)
    graph = agent.build_graph(model=_FakeModel(responses=[first, second]),
                              usage_tracker=tracker)
    results = []
    with pytest.raises(lg.UsageLimitExceeded):
        for update in graph.stream({"messages": [HumanMessage(content="x")]}, stream_mode="updates"):
            for node in update.values():
                results += [m for m in (node or {}).get("messages", []) if getattr(m, "type", "") == "tool"]
    texts = {m.tool_call_id: str(m.content) for m in results}
    assert texts["e0"].startswith("Replaced") and "ERROR: old_text not found" in texts["e1"]
    assert AFTER_EDIT not in texts["e1"]


def test_a_later_list_item_that_misses_says_so(edit_setup):
    agent, path = edit_setup
    result = _respond_list(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"},
        {"old_text": "    assert f(9) == 10\n", "new_text": "    assert f(9) == 11\n"},
    ])
    assert result.startswith("REJECTED: edit 2 of 2") and AFTER_EDIT not in result
    assert "as edit 1 of this list\nleft it (10 lines at that point). Nothing was written." in result


def test_a_later_list_item_whose_anchor_an_earlier_item_deleted(edit_setup):
    """The count is the list's working copy, labeled so; nothing claims a
    write, and read_tests is not offered as the list's state."""
    agent, path = edit_setup
    result = _respond_list(agent, [
        {"old_text": "\n\ndef test_b():\n    assert f(2) == 3\n", "new_text": ""},
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"},
        {"old_text": "    assert f(2) == 3\n", "new_text": "    assert f(2) == 3\n    assert f(5) == 6\n"},
    ])
    assert path.read_text(encoding="utf-8") == BASE
    assert "as edits 1 to 2 of this list\nleft it (6 lines at that point)" in result, result
    assert "lines on\ndisk" not in result and AFTER_EDIT not in result
    assert "read_tests shows the file on\ndisk, without the edits of this list." in result


def _limit(monkeypatch, **values):
    from testuary.agent import _limits

    real = _limits.value
    monkeypatch.setattr(_limits, "value", lambda name: values[name] if name in values else real(name))


def test_a_list_over_the_item_limit_is_refused(edit_setup, monkeypatch):
    agent, path = edit_setup
    _limit(monkeypatch, edit_list_items=1)
    result = _respond_list(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"},
        {"old_text": "    assert f(2) == 3\n", "new_text": "    assert f(2) == 3\n    assert f(3) == 4\n"},
    ])
    assert path.read_text(encoding="utf-8") == BASE
    assert result.startswith("ERROR: ``edits`` has 2 items, at most 1 apply in one"), result


def test_region_views_over_the_limit_are_marked(edit_setup, monkeypatch):
    agent, path = edit_setup
    _limit(monkeypatch, edit_list_views=1)
    body = "".join(f"\n\ndef test_x{i}():\n    assert f({i + 10}) == {i + 11}\n" for i in range(6))
    path.write_text(BASE + body, encoding="utf-8")
    result = _respond_list(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n"},
        {"old_text": "    assert f(15) == 16\n", "new_text": "    assert f(15) == 16\n    assert f(-5) == -4\n"},
    ])
    assert "Edit 1, lines" in result and "Edit 2, lines" not in result
    assert "[... 1 further changed region omitted, read_tests shows it ...]" in result


def _asserts(numbers):
    return "".join(f"    assert f({i}) == {i + 1}\n" for i in numbers)


def test_a_long_list_region_is_cut_inside_its_new_text(edit_setup):
    agent, path = edit_setup
    result = _respond_list(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n" + _asserts(range(20, 60))}])
    assert "  [... 17 lines of the changed region omitted, they are the new_text you just submitted ...]" \
        in result, result


def test_a_merged_view_does_not_call_unchanged_lines_submitted_text(edit_setup):
    """Two inserts within eleven lines share one view; its cut omits lines
    of neither new_text."""
    agent, path = edit_setup
    body = "".join(f"\n\ndef test_x{i}():\n    assert f({i + 10}) == {i + 11}\n" for i in range(3))
    path.write_text(BASE + body, encoding="utf-8")
    result = _respond_list(agent, [
        {"old_text": "    assert f(1) == 2\n", "new_text": "    assert f(1) == 2\n" + _asserts(range(20, 32))},
        {"old_text": "    assert f(10) == 11\n", "new_text": "    assert f(10) == 11\n" + _asserts(range(40, 52))},
    ])
    assert "Edits 1, 2, lines 5-37:" in result, result
    assert "  [... 9 lines of the changed region omitted, read_tests shows them ...]" in result
    assert "new_text you just submitted" not in result


def _remap_one_at_a_time(regions, at, old_len, new_len, index):
    """Reference: the mapping applied one replacement at a time."""
    delta = new_len - old_len
    landed = [at, at + new_len, {index}]
    kept = []
    for start, end, items in regions:
        if end < at or (end == at and start < at):
            kept.append([start, end, items])
        elif start > at + old_len or (start == at + old_len and end > start):
            kept.append([start + delta, end + delta, items])
        else:
            landed = [min(landed[0], start), max(landed[1], end + delta), landed[2] | items]
    kept.append(landed)
    return sorted(kept, key=lambda region: (region[0], region[1]))


def test_the_one_pass_region_mapping_equals_one_replacement_at_a_time():
    import random

    remap = tc.CoverageAgent._remap_regions
    for seed in range(3000):
        rnd = random.Random(seed)
        length, reference, regions = rnd.randint(0, 60), [], []
        for index in range(1, rnd.randint(2, 6)):
            old_len, new_len = rnd.randint(1, 4), rnd.randint(0, 6)
            befores, at = [], rnd.randint(0, 3)
            while at + old_len <= length and len(befores) < rnd.randint(1, 8):
                befores.append(at)
                # Adjacent replacements too, as replace_all finds them.
                at += old_len + rnd.choice([0, 0, 1, 2, 5])
            if not befores:
                continue
            for at in reversed(befores):
                reference = _remap_one_at_a_time(reference, at, old_len, new_len, index)
            regions = remap(regions, befores, old_len, new_len, index)
            assert regions == reference, (seed, index)
            length += len(befores) * (new_len - old_len)


def test_one_omitted_line_reads_in_the_singular(edit_setup, monkeypatch):
    agent, path = edit_setup
    _limit(monkeypatch, edit_echo_lines=1)
    edit = {"old_text": "    assert f(1) == 2\n",
            "new_text": "    assert f(1) == 2\n    assert f(0) == 1\n    assert f(3) == 4\n"}
    (single,) = _respond(agent, [edit])
    assert ("  [... 1 line of the inserted content omitted, it is from the new_text you just "
            "submitted ...]") in single, single
    path.write_text(BASE, encoding="utf-8")
    listed = _respond_list(agent, [edit])
    assert ("  [... 1 line of the changed region omitted, it is from the new_text you just "
            "submitted ...]") in listed, listed
    path.write_text("from target import f\n\n\ndef test_a():\n    x = 1\n    y = 2\n    assert f(x) == y\n",
                    encoding="utf-8")
    merged = _respond_list(agent, [{"old_text": "    x = 1\n", "new_text": "    x = 10\n"},
                                   {"old_text": "    assert f(x) == y\n", "new_text": "    assert f(x) == y + 9\n"}])
    assert "Edits 1, 2, lines 5-7:" in merged
    assert "  [... 1 line of the changed region omitted, read_tests shows it ...]" in merged, merged


def test_a_view_pasted_into_old_and_new_text_lands_without_numbers(edit_setup):
    """Both the read view and the edit views number lines N<TAB>text."""
    agent, path = edit_setup
    read = agent.read_tests()
    lines = [line for line in read.splitlines() if line.startswith(("8\t", "9\t"))]
    assert lines == ["8\tdef test_b():", "9\t    assert f(2) == 3"]
    pasted = "\n".join(lines) + "\n"
    (result,) = _respond(agent, [{"old_text": pasted, "new_text": pasted + "10\t    assert f(3) == 4\n"}])
    assert result.startswith("Replaced"), result
    assert path.read_text(encoding="utf-8").endswith(
        "def test_b():\n    assert f(2) == 3\n    assert f(3) == 4\n")


def test_an_old_text_found_as_sent_is_not_cleaned(edit_setup):
    """Text the file holds as sent is content, numbers or not, and its
    new_text keeps its numbers too."""
    agent, path = edit_setup
    path.write_text(BASE + '\n\nDATA = """\n1\tone\n2\ttwo\n"""\n', encoding="utf-8")
    (result,) = _respond(agent, [{"old_text": "1\tone\n2\ttwo\n", "new_text": "1\tone\n2\ttwo\n3\tthree\n"}])
    assert result.startswith("Replaced"), result
    assert '"""\n1\tone\n2\ttwo\n3\tthree\n"""' in path.read_text(encoding="utf-8")


def test_a_dict_literal_with_integer_keys_is_edited_as_written(edit_setup):
    agent, path = edit_setup
    path.write_text(BASE + "\n\nNAMES = {\n    1: 'one',\n    2: 'two',\n}\n", encoding="utf-8")
    (result,) = _respond(agent, [{"old_text": "    1: 'one',\n    2: 'two',\n",
                                  "new_text": "    1: 'one',\n    2: 'two',\n    3: 'three',\n"}])
    assert result.startswith("Replaced"), result
    assert "    1: 'one',\n    2: 'two',\n    3: 'three',\n}" in path.read_text(encoding="utf-8")


def test_a_pasted_view_with_a_bare_number_for_an_empty_line_lands(edit_setup):
    """The read view shows an empty line as "N<TAB>". A paste that drops that
    tab still matches the file, and nothing numbered is written."""
    agent, path = edit_setup
    (result,) = _respond(agent, [{
        "old_text": "5\t    assert f(1) == 2\n6\n7\n8\tdef test_b():\n",
        "new_text": "5\t    assert f(1) == 2\n6\n7\n8\tdef test_b():\n9\t    assert f(0) == 1\n"}])
    assert result.startswith("Replaced"), result
    assert "    assert f(1) == 2\n\n\ndef test_b():\n    assert f(0) == 1\n" in path.read_text(encoding="utf-8")
