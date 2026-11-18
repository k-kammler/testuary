"""A test run's output, read for the agents and the logs: the compacted
output, failure listings and summary lines."""
from __future__ import annotations

import re as _re
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
from testuary.agent import _limits


# Test output helpers: pure functions (no agent state) that parse and
# compact test output, so the coverage and mutation agents' tools give the
# model the same feedback. The failing-suite NEXT-ACTION footer names a
# phase's own tools, so the agent appends it.

# pytest's "in N.NNs" footer line.
_PYTEST_ELAPSED_RE = _re.compile(r"\bin\s+(\d+(?:\.\d+)?)\s*s\b")

# pytest's short-summary "FAILED test_x.py::test_y - ErrCls: msg" lines.
_PYTEST_SHORT_SUMMARY_RE = _re.compile(
    r"^(FAILED|ERROR)\s+(\S+)\s+-\s+(\S+):\s*(.*)$"
)


def parse_pytest_elapsed(full_output: str) -> Optional[float]:
    """The elapsed seconds of pytest's "in N.NNs" footer, or None without it
    (a collection error, a timeout)."""
    m = _PYTEST_ELAPSED_RE.search(full_output)
    return float(m.group(1)) if m else None


def group_pytest_failures(full_output: str) -> str:
    """Group pytest's short-summary FAILED and ERROR lines by error class and
    the first 30 characters of the message: the groups, largest first, with
    up to ``failure_group_names`` test names each, and an ACTION line when
    the largest group has 3 or more failures. "" with fewer than two parsed
    failures. A preamble for models that miss a root cause many failures
    share.
    """
    groups: Dict[tuple, List[str]] = {}
    for line in full_output.splitlines():
        m = _PYTEST_SHORT_SUMMARY_RE.match(line.strip())
        if m is None:
            continue
        _kind, test_name, err_class, msg = m.groups()
        key = (err_class, msg.strip()[:30])
        groups.setdefault(key, []).append(test_name)
    if not groups:
        return ""
    sorted_groups = sorted(groups.items(), key=lambda kv: -len(kv[1]))
    n_total = sum(len(v) for v in groups.values())
    if n_total < 2:
        return ""
    lines = [
        f"=== Failure groups ({n_total} failure(s) across "
        f"{len(groups)} root-cause class(es)) ===",
    ]
    for (err_class, msg_head), tests in sorted_groups:
        lines.append(f"  [{len(tests)}x] {err_class}: {msg_head}")
        shown, omitted = _limits.cut_items(
            tests, _limits.value("failure_group_names"), what="further test names")
        for t in shown:
            lines.append(f"      - {t}")
        if omitted:
            lines.append(f"      {omitted}")
    if sorted_groups and len(sorted_groups[0][1]) >= 3:
        biggest = sorted_groups[0]
        lines.append(
            f"  ACTION: the largest group ({len(biggest[1])} tests) "
            f"likely shares ONE root cause; fix that single issue first "
            f"and those {len(biggest[1])} tests should pass together. "
            f"If a group's failures look intractable for the current "
            f"test approach, REMOVE those tests rather than rewriting "
            f"the entire file -- the goal is a green baseline first, "
            f"broader coverage second."
        )
    lines.append("")
    return "\n".join(lines)


_FAILURE_BLOCK_HEADER_RE = _re.compile(r'^_{3,}\s+(\S+?)\s+_{3,}$')

#: A javac diagnostic line, ``Foo.java:12: error: cannot find symbol``.
_JAVAC_ERROR_RE = _re.compile(r'\.java:\d+: [^:\n]+: ')

#: The detail lines javac prints under an error.
_JAVAC_DETAIL_RE = _re.compile(r'^\s+\S')

#: javac's caret line under the source line it quotes.
_JAVAC_CARET_RE = _re.compile(r'^\s*\^\s*$')

#: javac's closing count in any language, "2 errors", "3 Fehler".
_JAVAC_COUNT_RE = _re.compile(r'^\d+ \w+\s*$')


def _omitted_tracebacks(count: int, names: Sequence[str] = ()) -> str:
    """The marker for the failure tracebacks left out, then their tests.
    Failures usually share root causes, and the grouped and the short
    summary (kept in full) name every failing test, so a few full tracebacks
    suffice."""
    text = _limits.marker(count, "items", "further failure tracebacks",
                          "the short summary below names every failing test")
    shown, omitted = _limits.cut_items(names, _limits.value("test_failure_names"),
                                       what="further names")
    if shown:
        text += "\nTheir tests: " + ", ".join(shown) + (f" {omitted}" if omitted else "")
    return text


#: pytest's count of the tests it collected, "collected 3 items".
_COLLECTED_ITEMS_RE = _re.compile(r'^collected (\d+) items?\b', _re.MULTILINE)

#: pytest's count of the modules it could not collect.
_COLLECTION_ERRORS_RE = _re.compile(r'\b(\d+) errors? during collection\b')


def canonical_test_summary_line(res: Any, returncode: Optional[int] = None) -> str:
    """The summary line of a ``TestRunResult``, the same in every language::

        TESTS: <status> | collected N | passed P | failed F | errors E | elapsed Ts

    The language's own failure detail follows it unchanged, since the model
    needs that format to write a fix. ``<status>`` is ``PASSED``,
    ``FAILED``, ``TIMEOUT`` or ``NO_RESULTS`` (no
    test was reached: a compile or configuration error or a missing engine).
    ``returncode``, when given, replaces the run's own, so a run
    ``_classify_strict_green`` turned failing reads ``FAILED``.
    """
    passed = int(getattr(res, "passed", 0) or 0)
    failed = int(getattr(res, "failed", 0) or 0)
    errors = int(getattr(res, "errors", 0) or 0)
    collected = getattr(res, "collected", None)
    elapsed = getattr(res, "elapsed_s", None)
    if returncode is None:
        returncode = getattr(res, "returncode", 0)
    timed_out = bool(getattr(res, "timed_out", False))

    derived_total = passed + failed + errors
    total = (
        collected if (collected is not None and collected > 0)
        else derived_total
    )
    # A module that fails to load is an error but no test: pytest's own
    # "collected N items" leaves it out, and a count without that line
    # leaves out the errors pytest reports "during collection".
    output = str(getattr(res, "output", "") or "")
    m_collected = _COLLECTED_ITEMS_RE.search(output)
    if m_collected is not None:
        total = int(m_collected.group(1))
    else:
        m_errors = _COLLECTION_ERRORS_RE.search(output)
        if m_errors is not None:
            total = max(total - int(m_errors.group(1)), 0)
    if timed_out:
        status = "TIMEOUT"
    elif total == 0 and returncode != 0:
        status = "NO_RESULTS"
    elif failed == 0 and errors == 0 and returncode == 0:
        status = "PASSED"
    else:
        status = "FAILED"

    elapsed_str = f"{elapsed:.2f}s" if elapsed is not None else "n/a"
    return (
        f"TESTS: {status} | "
        f"collected {total} | passed {passed} | "
        f"failed {failed} | errors {errors} | "
        f"elapsed {elapsed_str}"
    )


def collection_error_summary(full_out: str, returncode: int) -> Optional[str]:
    """The error lines of a collection-error run (exit code 2 and no test
    passed or failed: an import, syntax or indentation defect in the test
    file), which the run_tests view leads with. The block, possibly empty,
    for such a run, else None."""
    if returncode != 2:
        return None
    if _re.search(r"(\d+)\s+passed", full_out) or _re.search(
        r"(\d+)\s+failed", full_out
    ):
        return None
    err_lines: List[str] = []
    for ln in full_out.splitlines():
        stripped = ln.strip()
        if stripped.startswith("E   ") and ":" in stripped:
            err_lines.append(stripped[4:])
        if "while importing test module" in stripped:
            err_lines.append(stripped)
    if not err_lines:
        return ""
    file_line = ""
    m_floc = _re.search(r"^([^\s:]+\.py):(\d+):", full_out, _re.MULTILINE)
    if m_floc:
        file_line = f"{m_floc.group(1)}:{m_floc.group(2)}"
    return _error_block(err_lines, file_line)


def compile_error_summary(run: Any) -> Optional[str]:
    """The error lines of a run whose test file does not compile or link,
    which the run_tests view leads with: the run's reason detail, where its
    language marks the run with a ``compile_error`` diagnostic. The block,
    possibly empty, for such a run, else None."""
    if not any(entry.kind == "compile_error" for entry in getattr(run, "diagnostics", ())):
        return None
    return _error_block(run.reason_detail, "") if run.reason_detail else ""


def _error_block(err_lines: Sequence[str], file_line: str) -> str:
    """The block of a run's error lines, each once in the place it first
    appears, and the place in the test file they point to when known."""
    unique = list(dict.fromkeys(err_lines))
    return (
        "Specific error(s) blocking collection:\n"
        + "\n".join(f"  - {e}" for e in _limits.cut_lines(
            unique, _limits.value("collection_errors"), what="further errors"))
        + (f"\n  (in {file_line})\n" if file_line else "\n")
    )


def _loading_failed(output: str) -> bool:
    """Whether a run's tests could not be loaded: pytest reports errors
    during collection or an ERROR collecting a module, or no test passed or
    failed at all."""
    return bool(
        _COLLECTION_ERRORS_RE.search(output)
        or "ERROR collecting" in output
        or not _re.search(r'\d+\s+(passed|failed)', output)
    )


#: pytest's FAILURES or ERRORS section header.
_FAILURE_SECTION_RE = _re.compile(r'^=+ (FAILURES|ERRORS) =+', _re.MULTILINE)

#: The test a JUnit launcher failure block names: the className and
#: methodName of its MethodSource line, or the className of its ClassSource.
_JUNIT_SOURCE_RE = _re.compile(
    r"(?:Method|Class)Source \[className = '([^']*)'(?:, methodName = '([^']*)')?")

#: A stack frame as the launcher prints it, ``demo.CalcTest.adds(CalcTest.java:6)``.
_JUNIT_FRAME_RE = _re.compile(r'^[\w$./]+\(.*\)$')


def _launcher_failures(output: str) -> Dict[str, Tuple[str, List[str]]]:
    """The exception line and the stack frames of each test in the JUnit
    launcher's "Failures" section, by the test's dotted id (the class's,
    for a class that failed outside its tests). Read from the console
    while the structured report carries no trace."""
    details: Dict[str, Tuple[str, List[str]]] = {}
    current: Optional[str] = None
    for line in output.splitlines():
        found = _JUNIT_SOURCE_RE.search(line)
        if found:
            cls = found.group(1).replace("$", ".")
            current = f"{cls}.{found.group(2)}" if found.group(2) else cls
            details[current] = ("", [])
            continue
        text = line.strip()
        if current is None or not line.startswith("    ") or not text:
            current = None
            continue
        exception, frames = details[current]
        if text.startswith("=> "):
            details[current] = (text[3:], frames)
        elif _JUNIT_FRAME_RE.match(text):
            frames.append(f"at {text}")
        elif not frames:
            details[current] = (f"{exception} {text}".strip(), frames)
    return details


def failure_report(failures: Sequence[Any], frame_files: Sequence[str] = (),
                   output: str = "") -> str:
    """The view of a runner whose console output has no failure section
    (JUnit's launcher): the failed and errored tests of its structured
    report, each with its exception and message and the frames of its trace
    in ``frame_files`` (the target's and the tests' file names), the others
    counted. A trace's lines before its first frame are the exception with
    its whole message, as the runner wrote them. A failure without a trace
    takes its exception line and frames from the launcher's console
    ``output``."""
    launcher = _launcher_failures(output) if output else {}
    lines = [f"Failing tests ({len(failures)}):"]
    for failure in failures:
        kind = "" if failure.outcome == "failed" else " (error)"
        trace = (failure.trace or "").splitlines()
        exception, console = "", []
        if not trace:
            key = max((key for key in launcher if failure.test == key
                       or failure.test.startswith(key + ".")
                       or failure.test.startswith(key + "[")), key=len, default=None)
            if key is not None:
                exception, console = launcher[key]
        message = (failure.message or "").strip().splitlines() or ["(no message)"]
        if exception and (not (failure.message or "").strip() or message[0] in exception):
            message = [exception]
        first_frame = next((index for index, line in enumerate(trace)
                            if line.strip().startswith("at ")), None)
        if first_frame:
            message = "\n".join(trace[:first_frame]).strip().splitlines() or message
        lines.append(f"  - {failure.test}{kind}: {message[0]}")
        lines.extend(f"      {line}" for line in message[1:])
        frames = [line.strip() for line in trace
                  if line.strip().startswith("at ")] or console
        own = [frame for frame in frames
               if any(_re.search(r'[(/\\]' + _re.escape(name) + ':', frame)
                      for name in frame_files)]
        lines.extend(f"      {frame}" for frame in own)
        if len(frames) > len(own):
            lines.append("      " + _limits.marker(
                len(frames) - len(own), "items",
                "other frames"))
    return "\n".join(lines)


def compact_test_output(full_output: str, returncode: int,
                        failures: Sequence[Any] = (),
                        frame_files: Sequence[str] = ()) -> str:
    """Compact pytest or Maven output to its essential lines.

    One pass keeps the results line, the FAILURES and ERRORS sections (the
    first ``test_tracebacks`` full tracebacks, the rest one marker naming
    their tests), pytest's warnings summary (capped), the short test summary
    with its FAILED and ERROR lines, collection and import errors, javac's
    error lines with their details, and Maven's ``[ERROR]``, ``[WARNING]``,
    test-count and build-result lines. Then go on top: the import or
    collection cause when loading failed, the structured ``failures`` report
    when the output has no failure section, the failure groups, and the
    first Maven ``[ERROR]`` and ``[WARNING]`` lines, so the head slice of the
    tool result truncation keeps them. When nothing was kept, a passing
    run reads "All tests PASSED" and a failing one gets a head and tail cut
    of its output. Kept lines of a failing run that never say FAILED get its
    exit code appended. The caller appends the failing-suite NEXT-ACTION
    footer.
    """
    lines = full_output.split("\n")
    tracebacks = _limits.value("test_tracebacks")
    keep: list[str] = []
    in_failure_section = False
    full_blocks_kept = 0
    skipping_block = False
    skipped_block_count = 0
    skipped_names: list[str] = []
    javac_errors = False
    # Lines read since the last javac diagnostic, None outside its block.
    javac_block: Optional[int] = None
    # pytest's warnings summary, kept within ``test_warning_lines``.
    in_warnings = False
    warning_lines: list[str] = []

    def end_warnings() -> None:
        shown = [ln for ln in warning_lines if ln.strip()]
        keep.extend(_limits.cut_lines(shown, _limits.value("test_warning_lines"),
                                      what="further warning lines"))
        warning_lines.clear()

    for line in lines:
        # javac's diagnostics, in the JDK locale's language ("error:",
        # "Fehler:"): each with its quoted source line, the caret and its
        # indented details (symbol, location), then javac's closing count.
        if _JAVAC_ERROR_RE.search(line):
            javac_errors = True
            javac_block = 0
            keep.append(line)
            continue
        if javac_block is not None:
            javac_block += 1
            if (javac_block == 1 or _JAVAC_CARET_RE.match(line)
                    or _JAVAC_DETAIL_RE.match(line)):
                keep.append(line)
                continue
            javac_block = None
        if javac_errors and _JAVAC_COUNT_RE.match(line):
            keep.append(line)
            continue
        if in_warnings:
            # The section ends at pytest's documentation link or the next
            # ===== separator, which is then read as any other line.
            if line.startswith("-- Docs:"):
                in_warnings = False
                end_warnings()
                continue
            if line.startswith("=") and len(line) > 3:
                in_warnings = False
                end_warnings()
            else:
                warning_lines.append(line)
                continue
        # The results line ("X passed, Y failed in Z.Zs") and the section
        # headers, which show the model where the structured part begins.
        if _re.search(r'\d+\s+(passed|failed|error)', line):
            keep.append(line)
            continue
        if "FAILURES" in line and line.startswith("="):
            in_failure_section = True
            keep.append(line)
            continue
        if "ERRORS" in line and line.startswith("="):
            in_failure_section = True
            keep.append(line)
            continue

        if in_failure_section:
            # End of the FAILURES section: the short summary starts.
            if line.startswith("=") and "short test summary" in line.lower():
                in_failure_section = False
                if skipping_block and skipped_block_count > 0:
                    keep.append(_omitted_tracebacks(skipped_block_count, skipped_names))
                    skipped_block_count = 0
                    skipped_names = []
                keep.append(line)
                continue
            # Another ===== separator ends the section before a different
            # one, such as the warnings summary: flush the marker and read the
            # separator as that section's first line.
            if line.startswith("=") and len(line) > 3:
                in_failure_section = False
                if skipping_block and skipped_block_count > 0:
                    keep.append(_omitted_tracebacks(skipped_block_count, skipped_names))
                    skipped_block_count = 0
                    skipped_names = []
            else:
                # Per-failure block header: ___ test_name ___
                m = _FAILURE_BLOCK_HEADER_RE.match(line.strip())
                if m:
                    if tracebacks is None or full_blocks_kept < tracebacks:
                        full_blocks_kept += 1
                        skipping_block = False
                        keep.append(line)
                    else:
                        skipping_block = True
                        skipped_block_count += 1
                        skipped_names.append(m.group(1))
                    continue
                if not skipping_block:
                    keep.append(line)
                continue

        # pytest's warnings summary (an unawaited coroutine, a deprecation)
        # and its short test summary, wherever they follow.
        if line.startswith("=") and "warnings summary" in line.lower():
            in_warnings = True
            keep.append(line)
            continue
        if line.startswith("=") and "short test summary" in line.lower():
            keep.append(line)
            continue

        # Outside the failure section: the short summary's FAILED and ERROR
        # lines, collection errors and import errors.
        if line.startswith("FAILED ") or line.startswith("ERROR "):
            keep.append(line)
            continue
        if "collection" in line.lower() and "error" in line.lower():
            keep.append(line)
            continue
        if (
            "ModuleNotFoundError" in line
            or "ImportError" in line
            or "SyntaxError" in line
        ):
            keep.append(line)
            continue
        # Maven and Surefire lines, which pytest output never contains. Java
        # compile errors sit at the top of Maven output, where a tail slice
        # misses them.
        if (
            line.startswith("[ERROR]")
            or line.startswith("[WARNING]")
            or "Tests run:" in line
            or "BUILD FAILURE" in line
            or "BUILD SUCCESS" in line
            or line.startswith("Failed tests:")
            or "Tests in error:" in line
        ):
            keep.append(line)
            continue
    if in_warnings:
        end_warnings()

    # A framed copy of the import or collection cause goes on top, so the
    # truncation's head slice keeps it and the model sees the reason
    # (``cannot import name X from Y``) before the tracebacks. The lines also
    # stay in place. Only when loading failed: an ImportError a collected
    # test raises or chains is that test's failure, shown with its traceback.
    cause_lines: list[str] = []
    seen_cause: set[str] = set()
    for line in (lines if _loading_failed(full_output) else ()):
        s = line.strip()
        if not s or s in seen_cause:
            continue
        if (
            "cannot import name" in s
            or s.startswith("ImportError")
            or s.startswith("E   ImportError")
            or "ModuleNotFoundError" in s
            or (s.startswith("ERROR ") and "collecting" in s)
            or ("collection error" in s.lower())
        ):
            cause_lines.append(s)
            seen_cause.add(s)
    cause_lines = _limits.cut_lines(
        cause_lines, _limits.value("test_cause_lines"),
        what="further import and collection error lines")
    output = "\n".join(keep).strip()
    if cause_lines:
        banner = (
            "--- import / collection error (fix this first) ---\n"
            + "\n".join(cause_lines)
            + "\n--- end import / collection error ---"
        )
        output = banner + ("\n\n" + output if output else "")
    # A runner whose console output has no failure section (JUnit's
    # launcher) is read from its structured report ``failures``: every
    # failing test with its message, never a cut of the raw console text.
    if failures and not _FAILURE_SECTION_RE.search(full_output):
        report = failure_report(failures, frame_files, full_output)
        output = report + ("\n\n" + output if output else "")
    if not output:
        if returncode == 0:
            output = "All tests PASSED"
        else:
            # The head, where Java compile errors usually are, and the tail,
            # where the BUILD FAILURE or the final stack trace usually is.
            output = (
                f"Tests FAILED (exit code {returncode}):\n"
                + _limits.cut_text(full_output.strip(), _limits.value("test_fallback_chars"),
                                   what="the test output")
            )
    elif returncode != 0 and "FAILED" not in output.upper():
        output += f"\n\nTests FAILED (exit code {returncode})"

    grouped = group_pytest_failures(full_output)
    if grouped:
        output = grouped + output

    # The first Maven ``[ERROR]`` and ``[WARNING]`` lines (a compile or
    # plugin failure) go on top: the tool result truncation keeps a third
    # from the head and the rest from the tail, which would cut a build error
    # in the middle. pytest output has no such lines.
    err_lines = [
        ln for ln in full_output.splitlines()
        if ln.startswith("[ERROR]") or ln.startswith("[WARNING]")
    ]
    if err_lines:
        # Capped, so hundreds of ERROR lines of a multi-module build do not
        # fill the tool result limit themselves.
        head_block = "\n".join(_limits.cut_lines(
            err_lines, _limits.value("maven_error_lines"),
            what="further [ERROR] and [WARNING] lines"))
        output = (
            f"--- Maven [ERROR] block ---\n"
            f"{head_block}\n"
            f"--- end Maven [ERROR] block ---\n\n"
            f"{output}"
        )

    return output


def log_test_summary(
    full_output: str,
    returncode: int,
    log_fn: Callable[..., None],
    res: Any,
) -> None:
    """Log a test run's summary line (passed, failed, errors, elapsed) with
    the caller's ``log_fn(message, level)``. The counts come from the
    structured ``TestRunResult`` ``res``, which the backend parsed from its
    native format, so the line is right for Java too (Maven prints no
    pytest-style summary).
    """
    label = "run_tests"
    passed = int(getattr(res, "passed", 0) or 0)
    failed = int(getattr(res, "failed", 0) or 0)
    errors = int(getattr(res, "errors", 0) or 0)
    collected = getattr(res, "collected", None)
    elapsed_s = getattr(res, "elapsed_s", None)
    if elapsed_s is None:
        elapsed_s = parse_pytest_elapsed(full_output)

    derived_total = passed + failed + errors
    if derived_total > 0:
        total = derived_total
    elif collected is not None and collected > 0:
        total = collected
    else:
        total = 0
    is_collection_error = errors > 0 and passed + failed == 0
    elapsed_suffix = f" in {elapsed_s:.2f}s" if elapsed_s is not None else ""

    if total == 0 and returncode != 0:
        log_fn(
            f"[{label}] FAILED (exit code {returncode}): no test results "
            "recorded (build, compile, or discovery failure: inspect the "
            "run_tests output for the first error line)",
            level=1,
        )
    elif total == 0:
        log_fn(f"[{label}] No tests were collected.", level=1)
    elif is_collection_error:
        # The ImportError or SyntaxError is at the bottom of pytest's ERRORS
        # section, after the call chain: log the section (head and tail when
        # long) so the reason shows.
        section_lines: List[str] = []
        capture = False
        for line in full_output.splitlines():
            if "= ERRORS =" in line or "===== ERRORS =====" in line:
                capture = True
                continue
            if capture:
                if line.startswith("=") and " " in line and not line.startswith("___"):
                    break
                section_lines.append(line)
        cleaned = [ln for ln in section_lines if ln.strip()]
        if len(cleaned) <= 30:
            kept = cleaned
        else:
            kept = cleaned[:15] + [
                f"    ... [{len(cleaned) - 30} lines elided] ...",
            ] + cleaned[-15:]
        snippet = "\n    ".join(kept).strip()
        log_fn(
            f"[{label}] FAILED: {errors} collection error"
            f"{'s' if errors != 1 else ''} prevented test discovery "
            "(likely import or syntax issue in test or source code)",
            level=1,
        )
        if snippet:
            log_fn(f"  ERRORS section:\n    {snippet}", level=2)
    elif passed + failed + errors == 0 and total > 0 and returncode != 0:
        log_fn(
            f"[{label}] FAILED (exit code {returncode}): the test runner "
            f"discovered {total} test{'s' if total != 1 else ''} but none ran "
            f"to a pass/fail/error verdict. Likely causes: the subprocess was "
            f"killed mid-run (timeout / SIGKILL / OOM), a top-level import / "
            f"syntax error not classified as an 'error', or a fixture-level "
            f"exception that aborted before any test executed.",
            level=1,
        )
    else:
        # Java's Maven runs with -Dmaven.test.failure.ignore=true, so a suite
        # with failures still exits 0. The structured counts are the truth.
        any_failure = (failed > 0) or (errors > 0)
        status = "PASSED" if (returncode == 0 and not any_failure) else "FAILED"
        log_fn(
            f"[{label}] {status}: {passed} passed, {failed} failed, "
            f"{errors} errors / {total} total{elapsed_suffix}",
            level=1,
        )


#: pytest's closing summary line, "1 failed, 2 passed, 1 error in 0.40s".
_PYTEST_SUMMARY_RE = _re.compile(
    r'\d+\s+(passed|failed|errors?|skipped)\b.*\bin\s+\d+(\.\d+)?s\b')
#: pytest's own collection-error markers: a module's section head and the
#: interruption that follows.
_PYTEST_COLLECTION_ERROR_RE = _re.compile(
    r'ERROR collecting |\b\d+ errors? during collection\b')


def red_suite_footer(full_output: str, returncode: int,
                     measure_tool: str = "measure_coverage", failures: Sequence[Any] = ()) -> str:
    """The NEXT-ACTION footer both loops append to a red run_tests: the failure
    count (or a note on a discovery error), how to fix the failures with
    ``edit_tests``, and the next steps (an edit, then run_tests, with
    ``measure_tool`` offered only after a passing run). "" on a green run.
    ``failures`` is the run's structured report, counted when the output
    gives no count of its own (C's utest)."""
    if returncode == 0:
        return ""
    # JUnit's launcher summary ("[   1 tests failed   ]"), else pytest's "1 failed".
    m_failed = (_re.search(r'\[\s*(\d+)\s+tests\s+failed\s*\]', full_output)
                or _re.search(r'(\d+)\s+failed', full_output))
    m_collected = _re.search(r'collected\s+(\d+)\s+items?', full_output)
    n_failed = int(m_failed.group(1)) if m_failed else 0
    n_collected = int(m_collected.group(1)) if m_collected else 0
    # pytest's errors of collected tests (a fixture that raised in setup or
    # teardown), from its closing summary line, and JUnit's test classes
    # that failed outside their tests (a @BeforeClass / @BeforeAll method).
    summaries = [ln for ln in full_output.splitlines() if _PYTEST_SUMMARY_RE.search(ln)]
    m_errors = _re.search(r'(\d+)\s+errors?\b', summaries[-1]) if summaries else None
    n_errors = int(m_errors.group(1)) if m_errors else 0
    m_containers = _re.search(r'\[\s*(\d+)\s+containers\s+failed\s*\]', full_output)
    n_containers = int(m_containers.group(1)) if m_containers else 0
    is_collection_error = (
        n_failed == 0 and n_containers == 0 and (
            _PYTEST_COLLECTION_ERROR_RE.search(full_output) is not None
            or _JAVAC_ERROR_RE.search(full_output) is not None
        )
    )
    if not (n_failed or n_errors or n_containers or is_collection_error):
        n_failed = len(failures)

    if not (n_failed or n_errors or n_containers or is_collection_error):
        # A run the strict-green rules turned RED: the block above says why.
        data_phrase = (
            "No test failed, but the run counts as failed for the reason "
            "stated above. Fix that with ``edit_tests`` first."
        )
    elif is_collection_error:
        data_phrase = (
            "Discovery / compile error -- the test framework "
            "could not load the file. Fix the bad import, syntax "
            "error, or wrong module / package path with "
            "``edit_tests``: anchor old_text on the broken line and "
            "put the corrected line in new_text."
        )
    else:
        parts = []
        if n_failed or not (n_errors or n_containers):
            parts.append(
                f"{n_failed} failing test"
                f"{'s' if n_failed != 1 else ''}"
                + (f" of {n_collected} collected" if n_collected else ""))
        if n_errors:
            parts.append(
                f"{n_errors} test error{'s' if n_errors != 1 else ''} (a fixture "
                "raised in setup or teardown: ERROR at setup / teardown of the test)")
        if n_containers:
            parts.append(
                f"{n_containers} test class{'es' if n_containers != 1 else ''} "
                "failing outside its tests (a @BeforeClass / @BeforeAll or "
                "@AfterClass / @AfterAll method or the class's setup raised)")
        count_phrase = ", ".join(parts)
        # The grouped summary is shown only when failures could be grouped.
        grouped = (" (see grouped summary above for root-cause clusters)"
                   if group_pytest_failures(full_output) else "")
        data_phrase = (
            f"{count_phrase}{grouped}.\n"
            "Fix each failing block with ``edit_tests`` (old_text = "
            "the failing block, new_text = the corrected block). "
            "Failures that share one root cause are often a single "
            "edit to the common helper, fixture, or import."
        )
    return (
        "\n\n-> tests are RED. "
        + data_phrase
        + "\n\nThe next step is an ``edit_tests`` change, then ``run_tests``. "
        f"``{measure_tool}`` is offered only after a passing run."
    )
