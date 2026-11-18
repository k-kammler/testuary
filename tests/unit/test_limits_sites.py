"""Every limit cuts its own output, and ``unlimited`` passes it in full.

Each case sets one limit small, checks that the output is cut with the shared
marker, then sets it to unlimited (None) and checks that nothing is left
out. The last test checks that every limit of the registry is read by the
package.
"""
from __future__ import annotations

import re
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage

from agent_fixtures import target_agent

import testuary
from testuary.agent import _context as context
from testuary.agent import _helpers as ah
from testuary import _test_output
from testuary.agent import _limits
from testuary.agent import _langgraph as lg
from testuary.languages import mutation
from testuary.agent import coverage_agent as tc
from testuary import loadable

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

#: The package's own source files, in the repository or installed.
SOURCE = Path(testuary.__file__).resolve().parent


@pytest.fixture
def limit(monkeypatch):
    """Set limits by name: ``limit["read_chars"] = 50``; others keep their
    defaults."""
    values = {}
    defaults = _limits.defaults()
    monkeypatch.setattr(_limits, "value", lambda name: values.get(name, defaults[name]))
    return values


def _agent(tmp_path, **options):
    agent = target_agent(tmp_path, **options)
    agent.session.has_read = True
    agent.session.initial_measured = True  # past the start's measurement
    return agent


def _both(limit, name, small, render):
    limit[name] = small
    cut = render()
    limit[name] = None
    full = render()
    return cut, full


def test_tool_result_chars(limit, tmp_path):
    """A tool result keeps a third from its start, where tools put their
    banners, and the rest from its end, the test summary."""
    agent = _agent(tmp_path)
    text = "[banner]\n" + "x" * 500 + "\nsummary"
    cut, full = _both(limit, "tool_result_chars", 100, lambda: agent.track_output(text))
    marker = "[... 417 chars of the tool result omitted ...]"
    assert cut == "[banner]\n" + "x" * 24 + f"\n{marker}\n" + "x" * 59 + "\nsummary"
    assert full == text


def test_read_chars(limit, tmp_path):
    # A fresh agent per read: a repeated read is answered from the cache.
    source = "".join(f"x{i} = {i}\n" for i in range(200))
    cut, full = _both(limit, "read_chars", 60, lambda: _agent(tmp_path, source=source).read_source())
    assert "lines of target.py omitted, call this read tool with start_line=" in cut
    assert "omitted" not in full and "x199 = 199" in full


def _failures(n):
    blocks = "".join(f"_____ test_{i} _____\n\n    def test_{i}():\n>       assert False\nE       assert False\n\n"
                     for i in range(n))
    summary = "".join(f"FAILED t.py::test_{i} - AssertionError: boom\n" for i in range(n))
    return (f"===== FAILURES =====\n{blocks}===== short test summary info =====\n{summary}"
            f"===== {n} failed in 0.1s =====\n")


def test_test_tracebacks(limit):
    cut, full = _both(limit, "test_tracebacks", 1, lambda: _test_output.compact_test_output(_failures(3), 1))
    assert "[... 2 further failure tracebacks omitted, the short summary below names every " \
           "failing test ...]\nTheir tests: test_1, test_2" in cut
    assert "further failure tracebacks" not in full and "_____ test_2 _____" in full


def test_test_failure_names(limit):
    limit["test_tracebacks"] = 1
    cut, full = _both(limit, "test_failure_names", 1, lambda: _test_output.compact_test_output(_failures(3), 1))
    assert "Their tests: test_1 [... 1 further name omitted ...]" in cut
    assert "Their tests: test_1, test_2\n" in full


def test_test_warning_lines(limit):
    text = ("=== warnings summary ===\n"
            + "".join(f"test_x.py::test_{i}\n  RuntimeWarning: w{i}\n" for i in range(3))
            + "-- Docs: https://docs.pytest.org/\n3 passed, 3 warnings in 0.1s\n")
    cut, full = _both(limit, "test_warning_lines", 2, lambda: _test_output.compact_test_output(text, 0))
    assert "[... 4 further warning lines omitted ...]" in cut and "w0" in cut and "w2" not in cut
    assert "omitted" not in full and "RuntimeWarning: w2" in full


def test_test_cause_lines(limit):
    text = "".join(f"E   ModuleNotFoundError: No module named 'm{i}'\n" for i in range(3))
    cut, full = _both(limit, "test_cause_lines", 1, lambda: _test_output.compact_test_output(text, 2))
    assert "[... 2 further import and collection error lines omitted ...]" in cut
    assert "omitted" not in full.split("--- end import")[0]


def test_test_fallback_chars(limit):
    text = "start " + "y" * 3000 + " end"
    cut, full = _both(limit, "test_fallback_chars", 300, lambda: _test_output.compact_test_output(text, 1))
    assert "chars of the test output omitted" in cut and "start" in cut and "end" in cut
    assert "omitted" not in full and "y" * 3000 in full


def test_maven_error_lines(limit):
    text = "".join(f"[ERROR] problem {i}\n" for i in range(5))
    cut, full = _both(limit, "maven_error_lines", 2, lambda: _test_output.compact_test_output(text, 1))
    assert "[... 3 further [ERROR] and [WARNING] lines omitted ...]" in cut
    assert "further [ERROR]" not in full


def test_failure_group_names(limit):
    text = "".join(f"FAILED t.py::test_{i} - AssertionError: boom\n" for i in range(5))
    cut, full = _both(limit, "failure_group_names", 2, lambda: _test_output.group_pytest_failures(text))
    assert "[... 3 further test names omitted ...]" in cut
    assert "omitted" not in full and "test_4" in full


def test_collection_errors(limit):
    text = "".join(f"E   ImportError: cannot import name x{i}\n" for i in range(3))
    cut, full = _both(limit, "collection_errors", 1, lambda: _test_output.collection_error_summary(text, 2))
    assert "[... 2 further errors omitted ...]" in cut
    assert "omitted" not in full and "x2" in full


@pytest.mark.parametrize("name,entries,marker", [
    ("unsound_async_items", [(f"test_{i}", "w") for i in range(12)], "[... 10 further tests omitted ...]"),
    ("unsound_async_warnings", [(None, f"warning {i}") for i in range(9)],
     "[... 7 further warnings omitted ...]"),
])
def test_unsound_async_rejection_lists(limit, name, entries, marker):
    def render():
        run = SimpleNamespace(returncode=0, unsound=tuple(entries))
        return ah.unsound_async_rejection(run, lambda key, **kw: kw["sources_block"])
    cut, full = _both(limit, name, 2, render)
    assert marker in cut and "omitted" not in full


def test_timeout_suspects(limit):
    names = [f"test_{i}" for i in range(5)]
    cut, full = _both(limit, "timeout_suspects", 2, lambda: ah.suspect_names(names))
    assert cut == "test_0, test_1 [... 3 further tests omitted ...]"
    assert full == ", ".join(names)


def test_leak_names(limit):
    entry = {"rebound": [f"name_{i}" for i in range(4)], "xarray_options": ["a", "b", "c"]}
    cut, full = _both(limit, "leak_names", 1, lambda: loadable.describe_leak(entry))
    assert "rebound name_0 [... 3 further names omitted ...] of the module under test" in cut
    assert "changed xarray options a [... 2 further names omitted ...]" in cut
    assert "omitted" not in full


def test_survivor_limits():
    survivors = [{"id": f"m{i}", "line": i, "diff": "@@ -1 +1 @@\n-" + "a" * 400} for i in range(3)]
    cut = mutation.render_survivors(survivors, max_mutants=1, max_chars=50)
    assert "[... 2 further mutant changes omitted" in cut
    assert "chars of the change omitted ...]" in cut
    full = mutation.render_survivors(survivors, max_mutants=None, max_chars=None)
    assert "omitted" not in full and full.count("--- mutant ") == 3


def test_uncovered_and_branch_limits(limit, tmp_path):
    src = tmp_path / "big.py"
    src.write_text("".join(f"def f{i}(x):\n    if x: return {i}\n    return 0\n\n\n\n" for i in range(30)),
                   encoding="utf-8")
    missing = [6 * i + 2 for i in range(30)]
    cut, full = _both(limit, "uncovered_chars", 300, lambda: tc._render_uncovered_code(str(src), missing))
    assert "further uncovered regions omitted, read them with read_source" in cut
    assert "\nComplete uncovered ranges: " in cut
    assert "omitted" not in full
    limit["uncovered_chars"] = 300
    few, many = _both(limit, "uncovered_suggestions", 1,
                      lambda: tc._render_uncovered_code(str(src), missing))
    assert few.count("read_source(start_line=") == 1 and many.count("read_source(start_line=") > 20
    assert "further suggestions omitted ...]" in few and "further suggestions" not in many
    arcs = [f"{6 * i + 2}->{6 * i + 3}" for i in range(30)]
    cut, full = _both(limit, "branch_arcs", 2, lambda: tc._render_uncovered_branches(str(src), arcs))
    assert "[... 28 further entries omitted ...]" in cut and "further entries" not in full
    limit["uncovered_chars"] = None
    cut, full = _both(limit, "branch_snippet_chars", 5,
                      lambda: tc._render_uncovered_branches(str(src), ["2->3"]))
    assert "chars of the line omitted ...]" in cut and "omitted" not in full


def test_recovery_result_chars(limit, tmp_path):
    agent = _agent(tmp_path)
    agent.session.last_run_tests_result = "TESTS: " + "x" * 400 + "\nsecond line"
    cut, full = _both(limit, "recovery_result_chars", 50, agent.recovery_state_block)
    assert "chars of the result omitted ...]" in cut and "omitted" not in full
    assert "second line" not in full


class _StructureBackend:
    def __init__(self, snippet):
        self.snippet = snippet

    def check_structure(self, content):
        return type("Result", (), {"ok": False, "message": "syntax error at line 2:5",
                                   "line": 2, "col": 5, "snippet": self.snippet})()


def test_syntax_snippet_chars(limit, monkeypatch):
    from testuary import languages

    node = "call(a,\n b,\n c\n other = 1\n more = 2\n" + "z = 3\n" * 20
    monkeypatch.setattr(languages, "get_backend", lambda language: _StructureBackend(node.strip()))
    cut, full = _both(limit, "syntax_snippet_chars", 80,
                      lambda: ah.source_structure_error("a\nb\n", "python"))
    # The node quoted as written, its line breaks read as spaces: its first
    # 80 characters, then the marker.
    flat = " ".join(node.split())
    assert cut.startswith("syntax error at line 2:5: '" + flat[:80] + "' [... ")
    assert full.startswith("syntax error at line 2:5: '" + flat + "'\n")
    # A missing token quotes an empty source. A language without a source
    # gives the bare message.
    missing = type("Result", (), {"message": "missing token at line 1:3", "snippet": ""})()
    assert ah.structure_message(missing) == "missing token at line 1:3: ''"
    unknown = type("Result", (), {"message": "syntax error at line 1:3", "snippet": None})()
    assert ah.structure_message(unknown) == "syntax error at line 1:3"


def test_a_short_insertion_is_echoed_whole_under_a_tiny_limit(limit, tmp_path):
    def render():
        agent = _edit_agent(tmp_path)
        return agent.edit_file("    assert f(1) == 2\n",
                               "    assert f(1) == 2\n    assert f(0) == 1\n")

    # A two-line insertion keeps one line from each end, so nothing is left
    # out and no "0 lines omitted" marker appears.
    cut, full = _both(limit, "edit_echo_lines", 1, render)
    assert "omitted" not in cut and cut == full


def test_state_marker_chars(limit, tmp_path):
    agent = _agent(tmp_path)
    agent.session.last_uncovered_ranges = ", ".join(str(i) for i in range(400))
    cut, full = _both(limit, "state_marker_chars", 20, agent.build_state_marker)
    assert "chars of uncovered ranges omitted ...]" in cut and "omitted" not in full


def _edit_agent(tmp_path):
    agent = _agent(tmp_path)
    (tmp_path / "test_target.py").write_text(
        "from target import f\n\n\ndef test_a():\n    assert f(1) == 2\n", encoding="utf-8")
    return agent


def test_edit_echo_lines(limit, tmp_path):
    inserted = "".join(f"    assert f({i}) == {i + 1}\n" for i in range(40))

    def render():
        agent = _edit_agent(tmp_path)
        return agent.edit_file("    assert f(1) == 2\n", "    assert f(1) == 2\n" + inserted)

    cut, full = _both(limit, "edit_echo_lines", 10, render)
    assert "lines of the inserted content omitted, they are the new_text you just submitted" in cut
    assert "inserted content omitted" not in full and "assert f(39) == 40" in full


@pytest.mark.parametrize("name,small,marker", [
    ("edit_evidence_lines", 1, None),
    ("edit_evidence_chars", 10, "chars of the line omitted ...]"),
])
def test_edit_evidence(limit, tmp_path, name, small, marker):
    old = "from target import f\nimport os  # absent\ndef test_a():\n"

    def render():
        agent = _edit_agent(tmp_path)
        return agent.edit_file(old, "x")

    cut, full = _both(limit, name, small, render)
    if marker:
        assert marker in cut and marker not in full
    else:
        assert "def test_a():" not in cut and "def test_a():" in full


def test_rejection_and_malformed_limits(limit):
    call = AIMessage(content="", invalid_tool_calls=[
        {"id": "c1", "name": "t", "args": "{", "error": "E" * 500, "type": "invalid_tool_call"}])
    cut, full = _both(limit, "malformed_args_chars", 20,
                      lambda: lg._malformed_call_responses(call)[0].content)
    assert "chars of the error omitted ...]" in cut and "omitted" not in full
    assert "rejection_chars" in (SOURCE / "agent" / "_langgraph.py").read_text(encoding="utf-8")


def test_history_messages_unlimited_keeps_every_message():
    messages = [SystemMessage(content="s")] + [
        HumanMessage(content=str(i)) if i % 2 else AIMessage(content=str(i)) for i in range(50)]
    assert len(context.drop_oldest_pair_safe(messages, None)) == 51
    assert len(context.drop_oldest_pair_safe(messages, 30)) <= 31
    assert not isinstance(context.drop_oldest_pair_safe(messages, 30)[1], ToolMessage)


#: Limits whose reader is outside this package, with the reason.
_READ_ELSEWHERE = {
}


@pytest.mark.parametrize("name", list(_limits.defaults()))
def test_every_limit_is_read_by_the_package(name):
    if name in _READ_ELSEWHERE:
        pytest.skip(_READ_ELSEWHERE[name])
    text = "\n".join(path.read_text(encoding="utf-8") for path in SOURCE.rglob("*.py"))
    assert re.search(rf'_limits\.value\("{name}"\)', text), name
