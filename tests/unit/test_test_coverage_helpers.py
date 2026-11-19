"""The coverage agent's helpers: the shared renderers of ``_helpers``, the
RED-suite footer and pure helpers of ``coverage_agent``, and the
``CoverageAgent`` methods that use session state, on an instance."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_fixtures import coverage_agent
from agent_prompts import python_prompts

from offered_tools import offered_tools
from testuary import _test_output
from testuary.agent import coverage_agent as tc
from phase_values import set_phase_value

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _agent(root: Path) -> "tc.CoverageAgent":
    return coverage_agent(root, target_file_path=str(root / "src" / "foo.py"),
                          target_file_content="x = 1\n", test_output_dir=str(root / "tests"),
                          key_functions=["f"])


# _compact_test_output

def test_compact_test_output_passing_keeps_summary_line() -> None:
    out = "============= 5 passed in 0.42s ============="
    compacted = _test_output.compact_test_output(out, 0)
    assert "passed" in compacted


def test_compact_test_output_failed_includes_red_footer() -> None:
    out = (
        "============================= FAILURES =============================\n"
        "___ test_alpha ___\n"
        "Traceback ...\n"
        "AssertionError: expected 1 got 2\n"
        "============= short test summary info =============\n"
        "FAILED tests/test_x.py::test_alpha - AssertionError: expected 1 got 2\n"
        "============= 1 failed in 0.5s ============="
    )
    compacted = _test_output.red_suite_footer(out, 1)
    assert "tests are RED" in compacted
    assert "edit_tests" in compacted


def test_compact_test_output_many_failures_recommend_edit_file() -> None:
    failed_lines = "\n".join(
        f"FAILED tests/t.py::test_{i} - AssertionError: x" for i in range(6)
    )
    out = "FAILURES\n" + failed_lines + "\n============= 6 failed in 0.5s ============="
    compacted = _test_output.red_suite_footer(out, 1)
    assert "edit_tests" in compacted
    assert "root cause" in compacted.lower()


def test_compact_test_output_1_failure_recommends_edit_file() -> None:
    out = (
        "FAILURES\n"
        "FAILED tests/t.py::test_x - AssertionError: x\n"
        "============= 1 failed in 0.5s ============="
    )
    compacted = _test_output.red_suite_footer(out, 1)
    assert "edit_tests" in compacted


def test_compact_test_output_promotes_maven_error_lines() -> None:
    """Maven ``[ERROR]`` lines move to the top of the compacted body, so the
    tool result truncation cannot cut them off."""
    out = (
        "[INFO] Scanning for projects...\n"
        "[INFO] Building jcodec 1.0\n"
        "[ERROR] COMPILATION ERROR :\n"
        "[ERROR] Source option 6 is no longer supported. Use 7 or later.\n"
        "[ERROR] Target option 6 is no longer supported. Use 7 or later.\n"
        "[ERROR] Failed to execute goal org.apache.maven.plugins:maven-compiler-plugin\n"
        "[INFO] BUILD FAILURE\n"
        "[ERROR] [Help 1] http://example.com/MojoFailureException\n"
    )
    compacted = _test_output.compact_test_output(out, 1)
    header_pos = compacted.find("--- Maven [ERROR] block")
    assert header_pos != -1, f"missing Maven [ERROR] block header: {compacted[:300]!r}"
    assert "Source option 6 is no longer supported" in compacted
    err_block = compacted[header_pos:compacted.find("--- end Maven", header_pos)]
    assert err_block.count("\n") <= 14  # 12 ERROR lines + 2 frame lines


def test_compact_test_output_promotes_import_error_to_top() -> None:
    """An import or collection error, such as an invented import, moves to a
    framed block at the top, so the model sees the cause (cannot import name
    X from Y) first and the head-slice truncation cannot cut it off. The
    lines stay in place below too."""
    out = (
        "============================= test session starts ===========================\n"
        "collected 0 items / 1 error\n"
        "================================= ERRORS ====================================\n"
        "________________ ERROR collecting tests/test_gruntz.py ______________________\n"
        "tests/test_gruntz.py:3: in <module>\n"
        "    from sympy.series.gruntz import mrv_max\n"
        "E   ImportError: cannot import name 'mrv_max' from 'sympy.series.gruntz'\n"
        "=========================== short test summary info =========================\n"
        "ERROR tests/test_gruntz.py - ImportError: cannot import name 'mrv_max'\n"
        "============================== 1 error in 0.42s =============================\n"
    )
    compacted = _test_output.compact_test_output(out, 2)
    assert compacted.startswith("--- import / collection error (fix this first) ---")
    head = compacted.split("--- end import / collection error ---")[0]
    assert "cannot import name 'mrv_max'" in head  # the cause is in the top block
    # The detail (the original collecting line) is still present below.
    assert "ERROR collecting tests/test_gruntz.py" in compacted


def test_compact_test_output_no_import_error_no_banner() -> None:
    """A plain failing run (no import/collection error) gets no banner."""
    out = (
        "================================ FAILURES ===================================\n"
        "____________________________ test_add _______________________________________\n"
        "    assert add(2, 2) == 5\n"
        "E   assert 4 == 5\n"
        "============================== 1 failed in 0.1s =============================\n"
    )
    compacted = _test_output.compact_test_output(out, 1)
    assert "import / collection error" not in compacted


_JAVAC_OUTPUT = (
    "/w/src/test/java/p/FooTest.java:12: error: cannot find symbol\n"
    "        foo.bar();\n"
    "           ^\n"
    "  symbol:   method bar()\n"
    "  location: variable foo of type Foo\n"
    "/w/src/test/java/p/FooTest.java:20: error: incompatible types: "
    "String cannot be converted to int\n"
    "        int x = \"a\";\n"
    "                ^\n"
    "2 errors\n"
)


def test_compact_test_output_keeps_the_warnings_and_the_short_summary() -> None:
    """pytest's warnings summary (here the unawaited coroutine behind a
    failure) is kept, and the short test summary after it keeps its
    header."""
    out = (
        "....F..\n"
        "=================================== FAILURES ===================================\n"
        "__________________________________ test_meta ___________________________________\n"
        "tests/test_u.py:834: in test_meta\n"
        "    assert isinstance(instance, Meta)\n"
        "E   assert False\n"
        "=============================== warnings summary ===============================\n"
        "tests/test_u.py::test_meta\n"
        "  /x/_pytest/python.py:1: RuntimeWarning: coroutine 'Meta.__call__' was never awaited\n"
        "    Enable tracemalloc to get traceback where the object was allocated.\n"
        "\n"
        "-- Docs: https://docs.pytest.org/en/stable/how-to/capture-warnings.html\n"
        "=========================== short test summary info ============================\n"
        "FAILED tests/test_u.py::test_meta - assert False\n"
        "1 failed, 6 passed, 1 warning in 0.39s\n"
    )
    compact = _test_output.compact_test_output(out, 1)
    assert "RuntimeWarning: coroutine 'Meta.__call__' was never awaited" in compact
    assert "tests/test_u.py::test_meta\n" in compact
    assert "short test summary info" in compact
    assert "FAILED tests/test_u.py::test_meta - assert False" in compact
    assert "-- Docs:" not in compact
    green = _test_output.compact_test_output(
        "=== warnings summary ===\n"
        "tests/test_u.py:9\n"
        "  PytestUnknownMarkWarning: Unknown pytest.mark.timeout\n"
        "7 passed, 1 warning in 0.2s\n", 0)
    assert "PytestUnknownMarkWarning" in green


def test_compact_test_output_gives_no_import_banner_for_a_collected_test() -> None:
    """An ImportError that a collected test raises or chains is that test's
    failure, not a collection error: no import banner leads the output."""
    out = (
        "=================================== FAILURES ===================================\n"
        "____________________________________ test_x ____________________________________\n"
        "    def test_x():\n"
        ">       load_plugin()\n"
        "E   ImportError: cannot import name 'y' from 'z'\n"
        "\n"
        "During handling of the above exception, another exception occurred:\n"
        "E   RuntimeError: plugin unavailable\n"
        "=========================== short test summary info ============================\n"
        "FAILED test_a.py::test_x - RuntimeError: plugin unavailable\n"
        "1 failed, 2 passed in 0.10s\n"
    )
    compact = _test_output.compact_test_output(out, 1)
    assert "import / collection error" not in compact
    assert "E   ImportError: cannot import name 'y' from 'z'" in compact
    collection = _test_output.compact_test_output(
        "E   ModuleNotFoundError: No module named 'm'\n"
        "!!! Interrupted: 1 error during collection !!!\n"
        "1 error in 0.05s\n", 2)
    assert collection.startswith("--- import / collection error (fix this first) ---")


def test_compact_test_output_keeps_javac_error_lines() -> None:
    """A suite that does not compile shows javac's errors, not only its
    closing count ("2 errors"), which the results-line pattern keeps."""
    compacted = _test_output.compact_test_output(_JAVAC_OUTPUT, -1)
    assert "FooTest.java:12: error: cannot find symbol" in compacted
    assert "symbol:   method bar()" in compacted
    assert "location: variable foo of type Foo" in compacted
    assert ("FooTest.java:20: error: incompatible types: "
            "String cannot be converted to int") in compacted
    assert "2 errors" in compacted
    # The source line javac quotes and the caret under the place.
    assert "        foo.bar();\n           ^\n  symbol:   method bar()" in compacted
    assert '        int x = "a";\n                ^\n' in compacted


def test_compact_test_output_keeps_javac_errors_in_any_language() -> None:
    """A JDK in another locale writes its diagnostics in that language
    (here German, as javac on a German Windows does): they are kept with
    their source, caret, details and closing count, and the footer reads
    a compile error."""
    out = (
        "C:\\w\\src\\test\\java\\demo\\CalcTest.java:6: Fehler: Symbol nicht gefunden\r\n"
        "  @Test public void uses() { Missing m = new Missing(); }\r\n"
        "                             ^\r\n"
        "  Symbol: Klasse Missing\r\n"
        "  Ort: Klasse CalcTest\r\n"
        "1 Fehler\r\n"
    )
    compacted = _test_output.compact_test_output(out, -1)
    assert "CalcTest.java:6: Fehler: Symbol nicht gefunden" in compacted
    assert "  @Test public void uses() { Missing m = new Missing(); }" in compacted
    assert "  Symbol: Klasse Missing" in compacted and "  Ort: Klasse CalcTest" in compacted
    assert "1 Fehler" in compacted
    assert not compacted.startswith("Tests FAILED (exit code -1):")
    assert "Discovery / compile error" in _test_output.red_suite_footer(out, -1)


#: The JUnit launcher's console output for two failing tests, as a real
#: run printed it (JUnit 4 under the Vintage engine).
_JUNIT_OUTPUT = (
    "\n"
    "Failures (2):\n"
    "  JUnit Vintage:CalcTest:throwsInside\n"
    "    MethodSource [className = 'demo.CalcTest', methodName = 'throwsInside', "
    "methodParameterTypes = '']\n"
    "    => java.lang.NumberFormatException: For input string: \"x\"\n"
    "       java.base/java.lang.NumberFormatException.forInputString"
    "(NumberFormatException.java:67)\n"
    "       java.base/java.lang.Integer.parseInt(Integer.java:662)\n"
    "       demo.CalcTest.throwsInside(CalcTest.java:8)\n"
    "       java.base/java.lang.reflect.Method.invoke(Method.java:580)\n"
    "  JUnit Vintage:CalcTest:addsLarger\n"
    "    MethodSource [className = 'demo.CalcTest', methodName = 'addsLarger', "
    "methodParameterTypes = '']\n"
    "    => java.lang.AssertionError: expected:<9> but was:<5>\n"
    "       org.junit.Assert.fail(Assert.java:89)\n"
    "       demo.Calc.add(Calc.java:4)\n"
    "       demo.CalcTest.addsLarger(CalcTest.java:6)\n"
    "\n"
    "Test run finished after 99 ms\n"
    "[         2 containers found      ]\n"
    "[         0 containers failed     ]\n"
    "[         4 tests found           ]\n"
    "[         2 tests successful      ]\n"
    "[         2 tests failed          ]\n"
    "\n"
    "NOTE: Picked up JDK_JAVA_OPTIONS: -Xmx1g\n"
)


def _junit_failures():
    from testuary.languages import TestFailure
    return (TestFailure("demo.CalcTest.throwsInside", "error", 'For input string: "x"'),
            TestFailure("demo.CalcTest.addsLarger", "failed", "expected:<9> but was:<5>"))


def test_compact_test_output_reads_junit_failures_from_the_report() -> None:
    """JUnit's failures come from the structured report: every failing test
    with its exception and its frames in the target and the test, not a
    1,200-character cut of the launcher's console with its summary box."""
    compacted = _test_output.compact_test_output(
        _JUNIT_OUTPUT, 1, failures=_junit_failures(),
        frame_files=("Calc.java", "CalcTest.java"))
    assert compacted.startswith(
        "Failing tests (2):\n"
        "  - demo.CalcTest.throwsInside (error): java.lang.NumberFormatException: "
        'For input string: "x"\n'
        "      at demo.CalcTest.throwsInside(CalcTest.java:8)\n"
        "      [... 3 other frames omitted ...]\n"
        "  - demo.CalcTest.addsLarger: java.lang.AssertionError: expected:<9> but was:<5>\n"
        "      at demo.Calc.add(Calc.java:4)\n"
        "      at demo.CalcTest.addsLarger(CalcTest.java:6)\n"
        "      [... 1 other frame omitted ...]\n")
    assert "containers" not in compacted and "JDK_JAVA_OPTIONS" not in compacted


def test_a_failure_report_takes_frames_from_the_reported_trace() -> None:
    """A report that carries the trace gives the exception and the frames,
    and the console is not read."""
    from testuary.languages import TestFailure
    trace = ("java.lang.IllegalStateException\n"
             "\tat org.junit.Assert.fail(Assert.java:89)\n"
             "\tat demo.CalcTest.load(CalcTest.java:6)\n")
    report = _test_output.failure_report(
        (TestFailure("demo.CalcTest.adds", "error", "", trace),), ("CalcTest.java",))
    assert report == ("Failing tests (1):\n"
                      "  - demo.CalcTest.adds (error): java.lang.IllegalStateException\n"
                      "      at demo.CalcTest.load(CalcTest.java:6)\n"
                      "      [... 1 other frame omitted ...]")


def test_a_trace_with_a_blank_head_keeps_the_message() -> None:
    from testuary.languages import TestFailure
    trace = "\n\tat demo.CalcTest.load(CalcTest.java:6)\n"
    report = _test_output.failure_report(
        (TestFailure("demo.CalcTest.adds", "failed", "boom", trace),), ("CalcTest.java",))
    assert report == ("Failing tests (1):\n"
                      "  - demo.CalcTest.adds: boom\n"
                      "      at demo.CalcTest.load(CalcTest.java:6)")


def test_a_reported_failure_names_its_exception() -> None:
    """A JUnit report's message lacks the exception type. The trace's head
    gives it, with the whole message."""
    from testuary.languages import TestFailure
    trace = ("org.opentest4j.AssertionFailedError: expected: <9> but was: <5>\n"
             "second line\n"
             "\tat org.junit.jupiter.api.AssertionUtils.fail(AssertionUtils.java:151)\n"
             "\tat demo.CalcTest.adds(CalcTest.java:9)")
    failures = (TestFailure("demo.CalcTest.adds", "failed",
                            "expected: <9> but was: <5>\nsecond line", trace),)
    assert _test_output.failure_report(failures, ("CalcTest.java",)) == (
        "Failing tests (1):\n"
        "  - demo.CalcTest.adds: org.opentest4j.AssertionFailedError: "
        "expected: <9> but was: <5>\n"
        "      second line\n"
        "      at demo.CalcTest.adds(CalcTest.java:9)\n"
        "      [... 1 other frame omitted ...]")


def test_an_error_without_a_message_shows_its_stack_once() -> None:
    """A failure whose message is its whole stack, as a JUnit report without
    a message attribute gives it, shows the exception and its own frames
    once, not the stack as message lines."""
    from testuary.languages import TestFailure
    stack = ("java.lang.NullPointerException\n"
             "\tat demo.Calculator.load(Calculator.java:12)\n"
             "\tat demo.CalcTest.loads(CalcTest.java:15)\n"
             "\tat java.base/java.lang.reflect.Method.invoke(Method.java:580)")
    failures = (TestFailure("demo.CalcTest.loads", "error", stack, stack),)
    report = _test_output.failure_report(failures, ("Calculator.java", "CalcTest.java"))
    assert report == (
        "Failing tests (1):\n"
        "  - demo.CalcTest.loads (error): java.lang.NullPointerException\n"
        "      at demo.Calculator.load(Calculator.java:12)\n"
        "      at demo.CalcTest.loads(CalcTest.java:15)\n"
        "      [... 1 other frame omitted ...]")


def test_pytest_output_keeps_its_own_failure_section() -> None:
    """pytest's FAILURES section is the failure detail: the report's
    failures add nothing to it."""
    from testuary.languages import TestFailure
    out = ("=================================== FAILURES ===================================\n"
           "____________________________________ test_x ____________________________________\n"
           "E   assert 1 == 2\n"
           "1 failed in 0.10s\n")
    compacted = _test_output.compact_test_output(
        out, 1, failures=(TestFailure("test_a.test_x", "failed", "assert 1 == 2"),))
    assert compacted == _test_output.compact_test_output(out, 1)
    assert "Failing tests" not in compacted


def test_compact_test_output_collection_error_footer() -> None:
    out = (
        "==================================== ERRORS ====================================\n"
        "_______________________ ERROR collecting tests/test_x.py _______________________\n"
        "ImportError: cannot import name 'foo'\n"
        "!!!!!!!!!!!!!!!!!!!! Interrupted: 1 error during collection !!!!!!!!!!!!!!!!!!!!\n"
        "=============================== 1 error in 0.30s ==============================="
    )
    compacted = _test_output.red_suite_footer(out, 2)
    assert "Discovery / compile error" in compacted
    assert "edit_tests" in compacted


def test_red_footer_counts_junit_failed_tests() -> None:
    """JUnit's launcher summary ("1 tests failed") is counted. The
    containers line is not."""
    out = (
        "Test run finished after 83 ms\n"
        "[         2 containers found      ]\n"
        "[         0 containers failed     ]\n"
        "[        22 tests found           ]\n"
        "[        19 tests successful      ]\n"
        "[         3 tests failed          ]\n"
    )
    footer = _test_output.red_suite_footer(out, 1)
    assert "3 failing tests." in footer
    assert "Discovery / compile error" not in footer


def test_red_footer_counts_fixture_errors() -> None:
    """A fixture that raised (ERROR at setup of a test) is counted and named,
    not reported as "0 failing tests"."""
    out = (
        "==================================== ERRORS ====================================\n"
        "_________________________ ERROR at setup of test_read _________________________\n"
        "E   FileNotFoundError: data.json\n"
        "=========================== short test summary info ============================\n"
        "ERROR tests/test_x.py::test_read - FileNotFoundError: data.json\n"
        "========================= 3 passed, 1 error in 0.40s =========================\n"
    )
    footer = _test_output.red_suite_footer(out, 1)
    assert "1 test error (a fixture raised in setup or teardown" in footer
    assert "0 failing tests" not in footer
    both = _test_output.red_suite_footer(out.replace("3 passed, 1 error", "1 failed, 2 passed, 2 errors"), 1)
    assert "1 failing test, 2 test errors (a fixture" in both


def test_red_footer_tells_a_setup_error_from_a_collection_error() -> None:
    """Only pytest's collection-error markers make a collection error: a run
    whose only errors are in setup gets the setup footer, also when a path
    in its output holds the word "collections"."""
    out = (
        "==================================== ERRORS ====================================\n"
        "_________________________ ERROR at setup of test_read _________________________\n"
        "E   FileNotFoundError: /data/collections/data.json\n"
        "=========================== short test summary info ============================\n"
        "ERROR tests/test_x.py::test_read - FileNotFoundError: /data/collections/data.json\n"
        "========================= 3 passed, 1 error in 0.40s =========================\n"
    )
    footer = _test_output.red_suite_footer(out, 1)
    assert "1 test error (a fixture raised in setup or teardown" in footer
    assert "Discovery / compile error" not in footer
    collected = out.replace("ERROR at setup of test_read", "ERROR collecting tests/test_x.py")
    assert "Discovery / compile error" in _test_output.red_suite_footer(
        collected.replace("3 passed, 1 error", "1 error"), 2)


def test_red_footer_counts_junit_classes_failing_outside_their_tests() -> None:
    """A @BeforeClass that raised fails its class's container, not a test:
    the footer names the failing class, not "0 failing tests"."""
    out = (
        "Test run finished after 61 ms\n"
        "[         2 containers found      ]\n"
        "[         1 containers failed     ]\n"
        "[         0 tests found           ]\n"
        "[         0 tests failed          ]\n"
    )
    footer = _test_output.red_suite_footer(out, 1)
    assert "1 test class failing outside its tests (a @BeforeClass" in footer
    assert "0 failing tests" not in footer
    assert "Discovery / compile error" not in footer


def test_red_footer_keeps_pytest_failed_count() -> None:
    out = "============= 2 failed, 3 passed in 0.5s ============="
    assert "2 failing tests." in _test_output.red_suite_footer(out, 1)


def test_red_footer_gives_compile_advice_for_javac_errors() -> None:
    footer = _test_output.red_suite_footer(_JAVAC_OUTPUT, -1)
    assert "Discovery / compile error" in footer
    assert "0 failing tests" not in footer


# _compress_line_ranges

def test_compress_line_ranges_combines_consecutive_runs() -> None:
    assert tc._compress_line_ranges([1, 2, 3, 5, 7, 8, 9]) == "1-3, 5, 7-9"
    assert tc._compress_line_ranges([9, 8, 7, 5, 3, 2, 1]) == "1-3, 5, 7-9"


def test_compress_line_ranges_dedupes_and_handles_singleton() -> None:
    assert tc._compress_line_ranges([42]) == "42"
    assert tc._compress_line_ranges([5, 5, 5]) == "5"


def test_compress_line_ranges_empty_input() -> None:
    assert tc._compress_line_ranges([]) == ""


# _green_snapshot_key

def test_green_snapshot_key_orders_by_the_goal_metric_only() -> None:
    """The goal-metric coverage value is the whole key. Equal values give
    equal keys whatever the test count or capture time, so the earlier
    snapshot keeps its place. Under the default ``both`` mode both line and
    branch values are needed."""
    higher_cov = {"coverage": 90.0, "branch_coverage": 90.0, "test_count": 99, "snapshot_clock_s": 1.0}
    lower_cov = {"coverage": 70.0, "branch_coverage": 70.0, "test_count": 5, "snapshot_clock_s": 1.0}
    assert tc._green_snapshot_key(higher_cov) > tc._green_snapshot_key(lower_cov)
    same_cov_more_tests = {"coverage": 80.0, "branch_coverage": 80.0, "test_count": 20, "snapshot_clock_s": 100.0}
    same_cov_fewer_tests = {"coverage": 80.0, "branch_coverage": 80.0, "test_count": 5, "snapshot_clock_s": 1.0}
    assert tc._green_snapshot_key(same_cov_fewer_tests) == tc._green_snapshot_key(same_cov_more_tests)


def test_green_snapshot_key_does_not_rank_by_one_runs_time() -> None:
    """At an equal goal-metric value, a recorded ``exec_time_ms`` (one
    run's elapsed time) and the test count order nothing."""
    slow_few = {"coverage": 80.0, "branch_coverage": 80.0,
                "exec_time_ms": 900.0, "test_count": 3,
                "snapshot_clock_s": 1.0}
    fast_many = {"coverage": 80.0, "branch_coverage": 80.0,
                 "exec_time_ms": 100.0, "test_count": 30,
                 "snapshot_clock_s": 1.0}
    assert tc._green_snapshot_key(slow_few) == tc._green_snapshot_key(fast_many)
    measured = {"coverage": 80.0, "branch_coverage": 80.0,
                "exec_time_ms": 5000.0, "test_count": 1,
                "snapshot_clock_s": 1.0}
    unmeasured = {"coverage": 80.0, "branch_coverage": 80.0,
                  "exec_time_ms": None, "test_count": 1,
                  "snapshot_clock_s": 1.0}
    assert tc._green_snapshot_key(measured) == tc._green_snapshot_key(unmeasured)


def test_green_snapshot_key_unmeasured_coverage_sorts_low() -> None:
    unmeasured = {"coverage": None, "branch_coverage": None,
                  "test_count": 50, "snapshot_clock_s": 1.0}
    measured = {"coverage": 10.0, "branch_coverage": 10.0,
                "test_count": 5, "snapshot_clock_s": 0.5}
    assert tc._green_snapshot_key(measured) > tc._green_snapshot_key(unmeasured)


def test_green_snapshot_key_none_input() -> None:
    real = {"coverage": 0.0, "branch_coverage": 0.0,
            "test_count": 0, "snapshot_clock_s": 0.0}
    assert tc._green_snapshot_key(real) > tc._green_snapshot_key(None)


# Coverage-goal metric (line, branch, either, both): _goal_metric_value

def test_goal_metric_value_line_returns_line_only(monkeypatch) -> None:
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "line")
    assert tc._goal_metric_value(85.0, 70.0) == 85.0
    assert tc._goal_metric_value(85.0, None) == 85.0


def test_goal_metric_value_branch_returns_branch_only(monkeypatch) -> None:
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "branch")
    assert tc._goal_metric_value(85.0, 70.0) == 70.0
    assert tc._goal_metric_value(None, 70.0) == 70.0


def test_goal_metric_value_max_returns_higher(monkeypatch) -> None:
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "either")
    assert tc._goal_metric_value(85.0, 70.0) == 85.0
    assert tc._goal_metric_value(40.0, 90.0) == 90.0
    assert tc._goal_metric_value(50.0, None) == 50.0
    assert tc._goal_metric_value(None, 60.0) == 60.0


def test_goal_metric_value_min_requires_both(monkeypatch) -> None:
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "both")
    assert tc._goal_metric_value(85.0, 70.0) == 70.0
    assert tc._goal_metric_value(40.0, 90.0) == 40.0
    assert tc._goal_metric_value(None, 70.0) is None


@pytest.mark.parametrize("mode", ["both", "branch", "either", "line"])
def test_a_target_without_branches_is_judged_by_line_coverage(monkeypatch, mode) -> None:
    """Line coverage measured and no branch coverage: the target has no
    branches, so its goal is line coverage under every metric (under both
    and branch it could otherwise never be met)."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", mode)
    assert tc._goal_metric_value(85.0, None) == 85.0
    assert tc._goal_metric_value(None, None) is None


@pytest.mark.parametrize("mode", ["both", "branch"])
def test_a_target_without_branches_at_its_line_goal_meets_the_goal(monkeypatch, mode, tmp_path) -> None:
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", mode)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal", 100)
    a = _agent(tmp_path)
    a.session.best_coverage_pct = 100.0
    a.session.best_branch_coverage_pct = None
    assert a._coverage_goal_met() is True


def test_goal_metric_value_unknown_mode_falls_back_to_min(monkeypatch) -> None:
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "typo")
    assert tc._goal_metric_value(85.0, 70.0) == 70.0


def test_coverage_goal_met_min_mode_requires_both_metrics(monkeypatch, tmp_path) -> None:
    """Under the default both mode, 95% line and 70% branch coverage do not
    meet a 90% goal: branch lowers the minimum below it."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "both")
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal", 90)
    a = _agent(tmp_path)
    a.session.best_coverage_pct = 95.0
    a.session.best_branch_coverage_pct = 70.0
    assert a._coverage_goal_met() is False
    a.session.best_branch_coverage_pct = 95.0
    assert a._coverage_goal_met() is True


def test_coverage_stalled_below_goal_disabled_when_maximum_le_zero(monkeypatch, tmp_path) -> None:
    """``max_measurements_without_improvement <= 0`` disables the stagnation gate entirely."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "both")
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal", 90)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_measurements_without_improvement", 0)
    a = _agent(tmp_path)
    a.session.best_coverage_pct = 50.0
    a.session.best_branch_coverage_pct = 50.0
    a.session.measurements_without_improvement = 1_000_000  # way past any cap
    assert a._coverage_stalled_below_goal() is False, "patience=0 must disable the gate"
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_measurements_without_improvement", -1)
    assert a._coverage_stalled_below_goal() is False, "patience=-1 must disable the gate"
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "max_measurements_without_improvement", 5)
    assert a._coverage_stalled_below_goal() is True


def test_maybe_set_early_stop_min_mode_does_NOT_fire_on_line_only(monkeypatch, tmp_path) -> None:
    """line=91, branch=86, goal=90, metric=both gives no ``FRAMEWORK STOP``:
    the goal check goes through the goal metric, not best_coverage_pct, so
    with branch below the goal the early-stop helper returns None."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "both")
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal", 90)
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "extra_measurements_after_goal", 0)
    src = tmp_path / "src" / "foo.py"
    src.parent.mkdir()
    src.write_text("x = 1\n")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    a = tc.CoverageAgent(
        prompts=python_prompts("test_coverage"),
        deps=tc.CoverageTestDeps(
            target_file_path=str(src), target_file_content="x = 1\n",
            source_dir=str(tmp_path), test_output_dir=str(test_dir),
            language="python", key_functions=[],
        ),
        session=tc.TestCoverageSession(
            best_coverage_pct=91.0, best_branch_coverage_pct=86.0, last_tests_passed=True,
        ),
    )
    result = a._maybe_set_early_stop(source_tool="measure_coverage")
    assert result is None, (
        f"early-stop fired under both mode despite branch 86% < 90%: got {result!r}"
    )
    assert a.session.early_stop_result is None
    assert a.session.early_stop_reason is None
    # With branch above the goal both metrics reach 90, so the early stop fires.
    a.session.best_branch_coverage_pct = 92.0
    result = a._maybe_set_early_stop(source_tool="measure_coverage")
    assert result is not None
    assert a.session.early_stop_reason == "coverage_goal_met"


def test_coverage_goal_met_max_mode_either_clears(monkeypatch, tmp_path) -> None:
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal_metric", "either")
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "goal", 90)
    a = _agent(tmp_path)
    a.session.best_coverage_pct = 95.0
    a.session.best_branch_coverage_pct = 60.0
    assert a._coverage_goal_met() is True


# _target_test_file

def test_target_test_file_prefers_expected_stem(tmp_path: Path) -> None:
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    expected = test_dir / "test_foo.py"
    expected.write_text("# stub")
    (test_dir / "test_other.py").write_text("# other rank's file")
    out = _agent(tmp_path)._target_test_file(str(test_dir), "src/foo.py")
    assert out == expected


def test_target_test_file_falls_back_to_sole_sibling(tmp_path: Path) -> None:
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    only = test_dir / "test_alpha.py"
    only.write_text("# stub")
    out = _agent(tmp_path)._target_test_file(str(test_dir), "src/foo.py")
    assert out == only


def test_target_test_file_returns_none_on_ambiguous_dir(tmp_path: Path) -> None:
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_a.py").write_text("# a")
    (test_dir / "test_b.py").write_text("# b")
    out = _agent(tmp_path)._target_test_file(str(test_dir), "src/foo.py")
    assert out is None


def test_target_test_file_ignores_the_failure_marker(tmp_path: Path) -> None:
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_failed.py").write_text("# failure marker")
    out = _agent(tmp_path)._target_test_file(str(test_dir), "src/foo.py")
    assert out is None


# _looks_placeholder_explanation / _looks_placeholder_name

def test_looks_placeholder_explanation_catches_short_or_empty() -> None:
    assert tc._looks_placeholder_explanation("") is True
    assert tc._looks_placeholder_explanation("   ") is True
    assert tc._looks_placeholder_explanation("ok") is True


def test_looks_placeholder_explanation_catches_field_description_echoes() -> None:
    s = "Brief explanation of test strategy and what is covered."
    assert tc._looks_placeholder_explanation(s) is True


def test_looks_placeholder_explanation_passes_real_explanation() -> None:
    real = (
        "Tests target the parse_url function, focusing on schemes "
        "(http, https, ftp), userinfo, and trailing-slash edge cases."
    )
    assert tc._looks_placeholder_explanation(real) is False


def test_placeholder_fragments_are_the_current_field_descriptions() -> None:
    """Each phrase is the opening of a result field's description, so an
    echoed description is caught after the descriptions change."""
    descriptions = [field.description.lower()
                    for field in tc.CoverageTestResult.model_fields.values()
                    if field.description]
    assert all(any(text.startswith(fragment) for text in descriptions)
               for fragment in tc._PLACEHOLDER_FRAGMENTS)
    assert tc._looks_placeholder_explanation(
        "Filename of the on-disk working file (e.g. test_encoder.py).") is True


def test_looks_placeholder_name_loose_match_on_names_list() -> None:
    assert tc._looks_placeholder_name("parse_url") is False
    assert tc._looks_placeholder_name("foo") is False
    assert tc._looks_placeholder_name("") is True
    assert tc._looks_placeholder_name("list of functions/methods explicitly tested") is True


# build_state_marker

def test_state_marker_untested_shows_no_measurement(tmp_path) -> None:
    marker = _agent(tmp_path).build_state_marker()
    assert "tests=untested" in marker
    assert "coverage=none" in marker
    assert "next_valid" not in marker


def test_state_marker_red_state_facts(tmp_path) -> None:
    a = _agent(tmp_path)
    a.session.last_tests_passed = False
    a.session.last_run_tests_result = "FAILED test_x"
    a.session.tests_changed_since_run_tests = False
    marker = a.build_state_marker()
    assert "tests=red" in marker
    assert "edits_since_run=0" in marker
    assert "next_valid" not in marker


def test_state_marker_red_with_edit_state_facts(tmp_path) -> None:
    a = _agent(tmp_path)
    a.session.last_tests_passed = False
    a.session.last_run_tests_result = "FAILED test_x"
    a.session.tests_changed_since_run_tests = True
    marker = a.build_state_marker()
    assert "tests=red" in marker
    assert "edits_since_run=1+" in marker


def test_state_marker_green_with_coverage(tmp_path) -> None:
    a = _agent(tmp_path)
    a.session.last_tests_passed = True
    a.session.last_run_tests_result = "150 passed"
    a.session.best_coverage_pct = 78.5
    marker = a.build_state_marker()
    assert "tests=green" in marker
    assert "coverage=line 78%, no branches; goal metric (line) 78%, goal " in marker


def test_state_marker_includes_stagnation_counter_when_stalled(tmp_path) -> None:
    a = _agent(tmp_path)
    a.session.last_tests_passed = True
    a.session.best_coverage_pct = 84.3
    a.session.measurements_without_improvement = 4
    assert "stalled 4 measurements" in a.build_state_marker()


def test_state_marker_includes_uncovered_ranges_when_present(tmp_path) -> None:
    a = _agent(tmp_path)
    a.session.last_tests_passed = True
    a.session.best_coverage_pct = 84.3
    a.session.last_uncovered_ranges = "142, 156-160, 245-251"
    marker = a.build_state_marker()
    assert "uncovered:" in marker
    assert "142" in marker
    assert "156-160" in marker


def test_state_marker_omits_uncovered_line_when_empty(tmp_path) -> None:
    a = _agent(tmp_path)
    a.session.last_tests_passed = True
    a.session.best_coverage_pct = 95.0
    a.session.last_uncovered_ranges = ""
    assert "uncovered:" not in a.build_state_marker()


# The offered tools: measure_coverage hidden outside the needs_measurement phase


def test_offered_tools_hides_assess_when_red_no_edit(tmp_path) -> None:
    a = _agent(tmp_path)
    a.session.has_read = True  # past the read-only entry gate
    a.session.initial_measured = True  # past the start's measurement
    a.session.last_tests_passed = False
    a.session.tests_changed_since_run_tests = False
    a.session.last_run_tests_result = "FAILED test_x"
    names = {t.__name__ for t in offered_tools(a)}
    assert "measure_coverage" not in names


def test_offered_tools_exposes_assess_when_green_with_unmeasured_edit(tmp_path) -> None:
    a = _agent(tmp_path)
    a.session.has_read = True  # past the read-only entry gate
    a.session.initial_measured = True  # past the start's measurement
    a.session.writes_done = 1
    a.session.tests_changed_since_run_tests = False
    a.session.last_tests_passed = True
    a.session.tests_changed_since_measure["coverage"] = True
    names = {t.__name__ for t in offered_tools(a)}
    assert "measure_coverage" in names


# _log_test_summary

def test_compact_test_output_keeps_maven_error_lines() -> None:
    maven_red = (
        "[INFO] Scanning for projects...\n"
        "[INFO] -------< org.example:foo >--------\n"
        "[INFO] Compiling test sources\n"
        "[ERROR] /work/tests/UtilTest.java:[42,15] cannot find symbol\n"
        "[ERROR]   symbol:   class Foo\n"
        "[ERROR]   location: class UtilTest\n"
        "[INFO] BUILD FAILURE\n"
    )
    out = _test_output.compact_test_output(maven_red, returncode=1)
    assert "[ERROR]" in out
    assert "cannot find symbol" in out
    assert "BUILD FAILURE" in out


# _canonical_test_summary_line

def test_canonical_test_summary_line_python_passed() -> None:
    from testuary.languages import TestRunResult
    res = TestRunResult(
        ok=True, returncode=0, timed_out=False, elapsed_s=1.23,
        collected=5, passed=5, failed=0, errors=0,
        output="", line_coverage=None, branch_coverage=None,
    )
    assert _test_output.canonical_test_summary_line(res) == (
        "TESTS: PASSED | collected 5 | passed 5 | failed 0 | errors 0 | elapsed 1.23s"
    )


def test_canonical_test_summary_line_java_failed() -> None:
    from testuary.languages import TestRunResult
    res = TestRunResult(
        ok=False, returncode=1, timed_out=False, elapsed_s=0.45,
        collected=12, passed=10, failed=2, errors=0,
        output="", line_coverage=None, branch_coverage=None,
    )
    assert _test_output.canonical_test_summary_line(res) == (
        "TESTS: FAILED | collected 12 | passed 10 | failed 2 | errors 0 | elapsed 0.45s"
    )


def test_canonical_test_summary_line_no_results_distinguished_from_failed() -> None:
    from testuary.languages import TestRunResult
    res = TestRunResult(
        ok=False, returncode=1, timed_out=False, elapsed_s=0.3,
        collected=0, passed=0, failed=0, errors=0,
        output="[ERROR] cannot find symbol Foo\n",
        line_coverage=None, branch_coverage=None,
    )
    assert "NO_RESULTS" in _test_output.canonical_test_summary_line(res)


def test_canonical_test_summary_line_counts_no_test_for_a_collection_error() -> None:
    """A module that fails to load is an error but not a collected test:
    the run shows no results, collected 0 with the error."""
    from testuary.languages import TestRunResult
    out = ("_____ ERROR collecting test_broken.py _____\n"
           + "E   ModuleNotFoundError: No module named 'm'\n"
           + "!!!!! Interrupted: 1 error during collection !!!!!\n"
           + "1 error in 0.55s\n")
    res = TestRunResult(
        ok=False, returncode=2, timed_out=False, elapsed_s=0.55,
        collected=1, passed=0, failed=0, errors=1, output=out,
        line_coverage=None, branch_coverage=None,
    )
    assert _test_output.canonical_test_summary_line(res) == (
        "TESTS: NO_RESULTS | collected 0 | passed 0 | failed 0 | errors 1 | elapsed 0.55s"
    )
    res = res._replace(collected=3, passed=2, output="collected 2 items / 1 error\n" + out)
    assert _test_output.canonical_test_summary_line(res).startswith(
        "TESTS: FAILED | collected 2 | passed 2 | failed 0 | errors 1 |")


def test_canonical_test_summary_line_timeout() -> None:
    from testuary.languages import TestRunResult
    res = TestRunResult(
        ok=False, returncode=-1, timed_out=True, elapsed_s=None,
        collected=0, passed=0, failed=0, errors=0,
        output="", line_coverage=None, branch_coverage=None,
    )
    assert "TIMEOUT" in _test_output.canonical_test_summary_line(res)


def test_compact_test_output_keeps_maven_tests_run_summary() -> None:
    maven_green = (
        "[INFO] Scanning for projects...\n"
        "[INFO] T E S T S\n"
        "[INFO] Running UtilTest\n"
        "[INFO] Tests run: 12, Failures: 0, Errors: 0, Skipped: 0\n"
        "[INFO] BUILD SUCCESS\n"
    )
    out = _test_output.compact_test_output(maven_green, returncode=0)
    assert "Tests run: 12" in out
    assert "BUILD SUCCESS" in out
