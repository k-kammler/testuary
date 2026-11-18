"""Coverage phase tools read_source, read_tests, edit_file and run_tests.

Each test builds a ``CoverageAgent`` with synthetic deps and a fresh session,
fakes the backend where the tool runs a command, and checks the returned
string and the session. ``measure_coverage`` is tested in
test_test_coverage_measure*.py.
"""
from __future__ import annotations

from pathlib import Path

from agent_fixtures import install_backend, project_agent, run_result
from agent_prompts import c_prompts, python_prompts
from langadapter.runs import RunDiagnostic

import pytest

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


# read_source: the source is fixed during generation, so reads are cached.

def test_read_source_first_call_returns_full_content(tmp_path: Path) -> None:
    """The first call returns a header and the full source."""
    a = project_agent(tmp_path, source="def f():\n    return 42\n")
    out = a.read_source()
    assert "def f():" in out
    assert "return 42" in out
    assert "foo.py" in out


def test_read_source_immediate_re_read_returns_cached_header(tmp_path: Path) -> None:
    """A repeated read_source with no other tool between returns a short
    cached header."""
    a = project_agent(tmp_path)
    a.read_source()
    out = a.read_source()
    assert "ached" in out.lower() or "Cached" in out


def test_read_source_loop_breaker_after_4_reads(tmp_path: Path) -> None:
    """From the fourth consecutive read without an edit, the cached reply
    starts with a loop breaker that names concrete next tools."""
    a = project_agent(tmp_path)
    for _ in range(4):
        a.read_source()
    out = a.read_source()
    assert a.consecutive_reads("read_source") >= 5
    assert "edit_tests" in out


# read_tests: the test file can change between calls, so reads are not cached.

def test_read_tests_no_file_returns_clear_message(tmp_path: Path) -> None:
    """Before the baseline suite is on disk, ``read_tests`` returns the
    no-file tool message."""
    a = project_agent(tmp_path)
    out = a.read_tests()
    assert "[FRAMEWORK ERROR]" in out
    assert "required suite from the preceding phase" in out
    assert "``edit_tests`` remains unavailable" in out
    assert "Retry ``read_tests`` once" in out


def test_read_tests_returns_current_file_bytes(tmp_path: Path) -> None:
    """``read_tests`` returns the file's content verbatim after a header
    line."""
    a = project_agent(tmp_path)
    test_path = Path(a.deps.test_output_dir) / "test_foo.py"
    test_path.write_text("def test_x():\n    assert True\n", encoding="utf-8")
    out = a.read_tests()
    assert "def test_x():" in out
    assert "assert True" in out


# edit_file: the shared CodingAgent edit gate.

def test_edit_unique_match_updates_file_and_counter(tmp_path: Path) -> None:
    """A unique match updates the file on disk, increments ``writes_done``
    and invalidates the measurement cache."""
    a = project_agent(tmp_path)
    test_path = Path(a.deps.test_output_dir) / "test_foo.py"
    test_path.write_text("def test_a():\n    assert 1 == 1\n", encoding="utf-8")
    a.session.tests_changed_since_measure["coverage"] = False
    a.edit_file("1 == 1", "2 == 2")
    assert a.session.writes_done == 1
    assert "2 == 2" in test_path.read_text(encoding="utf-8")
    assert a.session.tests_changed_since_measure["coverage"] is True
    records = (
        a.resolve_log_dir() / "generated_test_validity_writes.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(records) == 1


def test_invalid_implementation_observation_is_rejected_before_write(
    tmp_path: Path,
) -> None:
    a = project_agent(tmp_path)
    test_path = Path(a.deps.test_output_dir) / "test_foo.py"
    original = "import src.foo as foo\n\ndef test_a():\n    assert foo.f() == 1\n"
    test_path.write_text(original, encoding="utf-8")
    out = a.edit_file(
        "    assert foo.f() == 1",
        "    assert open(foo.__file__).read()",
    )
    assert "REJECTED" in out
    assert "was NOT applied" in out
    assert test_path.read_text(encoding="utf-8") == original
    assert a.session.writes_done == 0
    assert not a.session.last_tests_passed
    rejection = (
        a.resolve_log_dir() / "generated_test_validity_rejections.jsonl"
    ).read_text(encoding="utf-8")
    assert '"write_applied":false' in rejection


def test_invalid_route_switches_are_rejected_before_behavioral_edit(
    tmp_path: Path,
) -> None:
    a = project_agent(tmp_path)
    test_path = Path(a.deps.test_output_dir) / "test_foo.py"
    original = "import src.foo as foo\n\ndef test_a():\n    assert foo.f() == 1\n"
    test_path.write_text(original, encoding="utf-8")

    first = a.edit_file(
        "    assert foo.f() == 1",
        "    assert open(foo.__file__).read()",
    )
    second = a.edit_file(
        original,
        "import dis\nimport src.foo as foo\n\n"
        "def test_a():\n    assert list(dis.get_instructions(foo.f))\n",
    )
    accepted = a.edit_file(
        "    assert foo.f() == 1",
        "    assert foo.f() == 1\n    assert callable(foo.f)",
    )

    assert "REJECTED" in first
    assert "REJECTED" in second
    assert "replaced 1 occurrence" in accepted.lower()
    assert a.session.writes_done == 1
    assert "callable(foo.f)" in test_path.read_text(encoding="utf-8")
    rejection_lines = (
        a.resolve_log_dir() / "generated_test_validity_rejections.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert len(rejection_lines) == 2


def test_a_validity_analysis_without_an_answer_stops_the_edit(
    tmp_path: Path, monkeypatch,
) -> None:
    """The edit gate refuses an edit the validity analysis could not judge.
    The error leaves the tool, and the file stays as it was."""
    import pytest
    from langadapter.python.adapter import PythonAdapter
    from testuary.consolidation import ValidityUnresolved

    def broken(self, source, target):
        raise RuntimeError("the analysis failed")

    monkeypatch.setattr(PythonAdapter, "analyze_validity", broken)
    a = project_agent(tmp_path)
    test_path = Path(a.deps.test_output_dir) / "test_foo.py"
    original = "def test_a():\n    assert 1 == 1\n"
    test_path.write_text(original, encoding="utf-8")
    with pytest.raises(ValidityUnresolved, match="analyze_validity"):
        a.edit_file("1 == 1", "2 == 2")
    assert test_path.read_text(encoding="utf-8") == original
    assert a.session.writes_done == 0


def test_edit_no_file_returns_directive(tmp_path: Path) -> None:
    """Without a test file on disk edit_file refuses, since edits apply to
    the baseline suite."""
    a = project_agent(tmp_path)
    a.edit_file("anything", "else")
    assert a.session.writes_done == 0


# run_tests: the backend's test run is faked.

def test_run_tests_passing_run_updates_state(monkeypatch, tmp_path: Path) -> None:
    """A green run sets last_tests_passed, increments run_tests_calls and
    clears tests_changed_since_run_tests, so a re-call without an edit is
    answered from the cache."""
    install_backend(monkeypatch, run_result(ok=True, returncode=0, output="5 passed in 0.4s"))
    a = project_agent(tmp_path)
    test_path = Path(a.deps.test_output_dir) / "test_foo.py"
    test_path.write_text("def test_x(): assert True\n", encoding="utf-8")
    out = a.run_tests()
    assert a.session.last_tests_passed is True
    assert a.session.run_tests_calls == 1
    assert a.session.tests_changed_since_run_tests is False
    assert "passed" in out.lower() or "PASS" in out


def test_run_tests_failing_run_records_failure(monkeypatch, tmp_path: Path) -> None:
    """A non-zero exit clears last_tests_passed, and the compact output
    includes the failed-test summary."""
    failed_out = (
        "============= FAILURES =============\n"
        "___ test_x ___\n"
        "AssertionError: expected 1 got 2\n"
        "============= short test summary info =============\n"
        "FAILED tests/test_foo.py::test_x - AssertionError: expected 1 got 2\n"
        "============= 1 failed in 0.5s ============="
    )
    install_backend(monkeypatch, run_result(ok=False, returncode=1, output=failed_out))
    a = project_agent(tmp_path)
    test_path = Path(a.deps.test_output_dir) / "test_foo.py"
    test_path.write_text("def test_x(): assert False\n", encoding="utf-8")
    out = a.run_tests()
    assert a.session.last_tests_passed is False
    assert a.session.run_tests_calls == 1
    assert "FAILED" in out or "failed" in out.lower()


def test_run_tests_cached_re_call_skips_subprocess(monkeypatch, tmp_path: Path) -> None:
    """A re-call without an edit between gets the cached reply without a
    second run."""
    sub_calls = {"n": 0}
    install_backend(monkeypatch, run_result(ok=True, returncode=0, output="3 passed in 0.1s"), counter=sub_calls)
    a = project_agent(tmp_path)
    test_path = Path(a.deps.test_output_dir) / "test_foo.py"
    test_path.write_text("def test_x(): assert True\n", encoding="utf-8")
    a.run_tests()
    a.run_tests()
    assert sub_calls["n"] == 1  # second call served from the cache


def test_run_tests_collection_error_promotes_specific_errors(monkeypatch, tmp_path: Path) -> None:
    """A collection error (exit 2, no test results) leads with the error
    lines extracted by the shared summary helper."""
    out_text = (
        "E   ImportError: cannot import name 'foo' from 'bar'\n"
        "collected 0 items / 1 error\n"
    )
    install_backend(monkeypatch, run_result(ok=False, returncode=2, output=out_text))
    a = project_agent(tmp_path)
    test_path = Path(a.deps.test_output_dir) / "test_foo.py"
    test_path.write_text("import missing_module\n", encoding="utf-8")
    out = a.run_tests()
    assert "[Discovery / compile error.]" in out
    assert "ImportError: cannot import name 'foo' from 'bar'" in out


#: A C test file's compile failure as the language reports it: the step,
#: and the compiler's lines, one of them twice.
_COMPILER_LINES = ("test_calc.c:4:5: error: implicit declaration of function 'f'",
                   "test_calc.c:4:5: error: implicit declaration of function 'f'",
                   "test_calc.c:9:1: error: expected ';' before '}' token")


def _run_tests_after(monkeypatch, tmp_path: Path, run, prompts) -> str:
    """The run_tests reply to ``run`` with the agent's ``prompts``."""
    install_backend(monkeypatch, run)
    a = project_agent(tmp_path)
    a.prompts = prompts
    (Path(a.deps.test_output_dir) / "test_foo.py").write_text("x\n", encoding="utf-8")
    return a.run_tests()


def test_run_tests_compile_error_leads_with_the_compilers_lines(monkeypatch, tmp_path: Path) -> None:
    """A test file that does not compile, which the language marks with a
    compile_error diagnostic, leads with the compiler's lines from the run's
    reason detail, each once, and the language's load-error help."""
    failed = run_result(ok=False, returncode=1, output="the compiler's output")._replace(
        diagnostics=(RunDiagnostic("compile_error", "the test file test_calc.c does not compile"),),
        reason_detail=_COMPILER_LINES)
    out = _run_tests_after(monkeypatch, tmp_path, failed, c_prompts("test_coverage"))
    assert out.startswith(
        "[Discovery / compile error.] No tests could be\n"
        "collected, so nothing was run. This is ALWAYS a problem in the\n"
        "working file itself (or in code it imports), not in test logic.\n"
        "\n"
        "Specific error(s) blocking collection:\n"
        "  - test_calc.c:4:5: error: implicit declaration of function 'f'\n"
        "  - test_calc.c:9:1: error: expected ';' before '}' token\n"
        "\n"
        "Fix: edit the working file with ``edit_tests`` to address the\n"
        "specific error(s) above -- anchor on the broken line(s) and\n"
        "replace them. Common patterns:\n"
        "  - implicit declaration of ASSERT_NOT_NULL or another check -> utest.h\n"
        "    has only the checks the toolkit lists: use ASSERT_TRUE(p != NULL).\n"
        "  - 'utest_result' undeclared -> a check stands outside a test: make the\n"
        "    helper return a value and check it in the test.\n"
        "  - implicit declaration / undeclared identifier -> include the target's\n"
        "    header, or declare the function as the header does.\n"
        "  - undefined reference -> the function is static or not in the target:\n"
        "    test it through the functions the target exports.\n"
        "  - conflicting types -> do not redeclare the target's functions.\n"
        "Re-running run_tests without an actual edit will produce the\n"
        "same error.\n"
        "\n"
        "--- Full utest.h output (for reference, stack trace below) ---\n")


def test_run_tests_without_a_compile_error_diagnostic_has_no_compile_block(
        monkeypatch, tmp_path: Path) -> None:
    """A failed run whose language marks no compile error, such as Java's,
    gets no collection-error block from its reason detail."""
    failed = run_result(ok=False, returncode=1, output="1 failed")._replace(
        reason_detail=_COMPILER_LINES)
    out = _run_tests_after(monkeypatch, tmp_path, failed, python_prompts("test_coverage"))
    assert "[Discovery / compile error.]" not in out


def test_run_tests_missing_test_file_returns_clear_error(monkeypatch, tmp_path: Path) -> None:
    """Without a test file on disk run_tests returns a directive without
    running the tests."""
    runs = {"n": 0}
    install_backend(monkeypatch, run_result(), counter=runs)
    a = project_agent(tmp_path)
    a.run_tests()
    assert runs["n"] == 0   # find_target_test_file returns None, so nothing runs
    assert a.session.last_tests_passed is False
