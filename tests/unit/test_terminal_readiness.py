"""Readiness views from hand-written events, and the command line's setup refusal.

The events use the presentation's event fields. They verify no tools,
credentials, dependencies or version support. The command-line cases run the
real setup observers and the typed refusal report with empty isolated
settings, without native or provider work.
"""
from __future__ import annotations

from dataclasses import asdict
from io import StringIO
import socket
import subprocess
import sys

import pytest
from rich.console import Console

from testuary.presentation import DisplayEvent, Metric, TerminalPresenter
from testuary.presentation.terminal import safe_text


_PROJECT = "/work/Example Project"
_OUTPUT = "/work/generated suites"
_PYTHON = (
    f"Project: {_PROJECT}; targets: src/alpha.py, src/beta.py; context: local Linux; "
    "framework Python: 3.12.3; subject Python: /work/Example Project/.venv/bin/python "
    "3.10.14, selected explicitly; supervisor Python: not required"
)
_JAVA = (
    "Project: /workspace/example; targets: src/main/java/example/Alpha.java; "
    "context: Linux Docker guest, container subject-java; host: Windows; "
    "framework Python: host 3.12.3; measurement JDK: guest /opt/jdk-17 17.0.12; "
    "build JDK: guest /opt/jdk-21 21.0.4; release target: 8; "
    "build: guest Maven wrapper 3.9.9; JUnit Jupiter: 5.10.3; "
    "JaCoCo: 0.8.12; PIT: 1.16.1; guest control Python: /opt/control/bin/python "
    "3.12.3, separate from Java execution"
)
_RUN = (
    "Model: local-example; credentials: configured, not authenticated; "
    "provider quota: unknown; provider context limit: unknown; provider cost: unknown; "
    "effective limits: per-phase model requests 200, per-response output tokens 4096, "
    "per-phase wall clock unlimited; "
    f"output: {_OUTPUT}; resume: disabled, existing output is refused"
)
_SCOPE = (
    "Inspection only: selected configuration and candidate versions inspected; "
    "native build/discovery unverified; no model request, build, tests or install"
)
_VERIFY_JAVA = (
    "testuary check --verify --source-dir /workspace/example "
    "--source-files src/main/java/example/Alpha.java --model local-example"
)
_CASES = {
    "ambiguous_python": (
        "ambiguous", "Python runtime", "local .venv and captured activated environment",
        "Two distinct compatible subject interpreters; no runtime selected",
        'Set --python-executable "/work/Example Project/.venv/bin/python"',
    ),
    "missing_python": (
        "missing setup", "Python runtime", "/work/Example Project/.venv/bin/python",
        "Selected subject interpreter does not exist; no framework-Python fallback",
        "Create the subject environment using the project's documented setup, then "
        "select its interpreter with --python-executable",
    ),
    "conda_activation": (
        "missing setup", "Conda invocation", "prepared env subject with activation requirements",
        "Interpreter path alone does not supply the required activation",
        "Run conda run -n subject python --version and select the prepared subject "
        "environment using the documented Conda invocation route",
    ),
    "ambiguous_build": (
        "ambiguous", "Java build", "/workspace/example/pom.xml and build.gradle",
        "Maven and Gradle routes both exist; no build selected",
        "Select the intended build descriptor in project configuration before generation",
    ),
    "missing_wrapper": (
        "missing setup", "Java wrapper", "/workspace/example/.mvn/wrapper",
        "Maven wrapper distribution is not cached; no global Maven fallback or download",
        "In the selected project environment, run ./mvnw --version after explicitly "
        "authorizing the wrapper download, then repeat inspection",
    ),
    "missing_guest_helper": (
        "missing setup", "guest control", "subject-java:/opt/control/bin/python",
        "Selected guest helper is absent; host Python cannot verify guest execution",
        "Select a prepared guest control interpreter in the container profile; "
        "do not substitute the host interpreter",
    ),
    "unsupported_jdk": (
        "unsupported", "mutation execution", "guest /opt/jdk-8 with selected PIT plugin",
        "Selected measurement-JDK/plugin tuple is incompatible; release target 8 "
        "does not require running the measurement engine on JDK 8",
        "Set --java-home /opt/jdk-17 to the prepared compatible guest measurement JDK",
    ),
    "unverified_tuple": (
        "unverified", "version tuple", "measurement JDK 23 and selected PIT plugin",
        "No verification evidence for this tuple; not automatically supported or incompatible",
        f"Run {_VERIFY_JAVA} for these exact selected inputs before generation",
    ),
    "failed_discovery": (
        "failed", "native test discovery", "/workspace/example selected Maven module",
        "Native discovery failed; required cached dependency is absent; no model request made",
        "In the selected project environment, run ./mvnw -pl core test-compile "
        "after preparing its native dependencies, then repeat verification",
    ),
    "partial_verification": (
        "unverified", "native verification", "/workspace/example selected Maven module",
        "Build verified; test discovery unverified after interruption; no overall-ready claim",
        f"Repeat {_VERIFY_JAVA} with the same selected inputs",
    ),
    "occupied_output": (
        "failed", "output policy", _OUTPUT,
        "Destination already contains results; existing files are unchanged",
        'Select a new --output-dir, or use --resume only for the matching saved run',
    ),
    "resume_mismatch": (
        "failed", "resume validation", _OUTPUT,
        "Saved run targets differ from the requested targets; no result overwritten",
        "Use the saved run's targets for resume, or select a new --output-dir",
    ),
}


def _event(message, *, operation="inspection", state="succeeded", metrics=()):
    return DisplayEvent(
        run_id="first-run-offline-view", target="explicit selected targets",
        phase="setup", operation=operation, state=state, elapsed_s=0,
        message=message, metrics=metrics, provenance="synthetic",
    )


def _render(events, mode, width):
    stream = StringIO()
    console = Console(file=stream, force_terminal=True, color_system=None,
                      width=width, height=60, highlight=False, markup=False)
    with TerminalPresenter(mode, console=console) as presenter:
        assert presenter.mode == mode
        for event in events:
            presenter(event)
        if mode == "compact":
            frame_stream = StringIO()
            Console(file=frame_stream, color_system=None, width=width,
                    highlight=False, markup=False).print(presenter.frame())
            return frame_stream.getvalue()
    return stream.getvalue()


def _words(text):
    return " ".join(safe_text(text).split())


@pytest.fixture(autouse=True)
def no_external_work(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")

    def forbidden(*args, **kwargs):
        raise AssertionError("A synthetic readiness view must not execute or connect")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.mark.parametrize("mode", ("plain", "detailed", "compact"))
@pytest.mark.parametrize("width", (40, 100))
@pytest.mark.parametrize("context", (_PYTHON, _JAVA))
def test_first_run_identifies_selected_inputs_without_native_or_provider_readiness(mode, width, context):
    event = _event(f"{context}; {_RUN}; {_SCOPE}")
    before = asdict(event)
    output = _render((event,), mode, width)
    normalized = _words(output)

    for text in ("demo", "Project:", "targets:", "context:", "framework Python:",
                 "Model: local-example", "configured, not authenticated", "quota: unknown",
                 "context limit: unknown", "cost: unknown", "model requests 200",
                 "output tokens 4096", "wall clock unlimited", "resume: disabled",
                 "Inspection only", "native build/discovery unverified", "no model request"):
        assert text in normalized
    assert _words(context) in normalized
    assert _OUTPUT in normalized
    assert "READY" not in output
    assert "100%" not in output
    assert "Saved diagnostics:" not in output
    assert "cost $0" not in output
    assert asdict(event) == before
    if mode in ("plain", "detailed"):
        assert len(output.splitlines()) == 1  # one unsplit record per observation
    else:
        assert all(len(line) <= width for line in output.splitlines())


@pytest.mark.parametrize("mode", ("plain", "detailed", "compact"))
@pytest.mark.parametrize("width", (40, 100))
@pytest.mark.parametrize("case", tuple(_CASES))
def test_setup_issue_preserves_status_checked_location_and_next_action(mode, width, case):
    status, operation, checked, reason, action = _CASES[case]
    event = _event(
        f"Status: {status}; required operation: {operation}; checked: {checked}; "
        f"reason: {reason}; Next: {action}; diagnostics: none saved",
        operation=operation, state="failed" if status == "failed" else "waiting",
    )
    output = _render((event,), mode, width)
    normalized = _words(output)

    for text in ("demo", f"Status: {status}", f"checked: {checked}", reason, f"Next: {action}"):
        assert _words(text) in normalized
    assert "diagnostics: none saved" in normalized
    assert "Saved diagnostics:" not in normalized
    assert "READY" not in output
    assert "100%" not in output
    if mode in ("plain", "detailed"):
        assert len(output.splitlines()) == 1  # one unsplit record per observation
    else:
        assert all(len(line) <= width for line in output.splitlines())


@pytest.mark.parametrize("mode", ("plain", "detailed", "compact"))
@pytest.mark.parametrize("width", (40, 100))
def test_native_verified_is_scoped_not_provider_authentication(mode, width):
    event = _event(
        "Status: verified; scope: selected native build and test discovery; "
        "no tests or mutants measured; model credentials configured, not authenticated; "
        "provider quota unknown; later generation and measurement success not guaranteed",
        operation="native verification",
    )
    output = _words(_render((event,), mode, width))
    assert "demo" in output
    assert "Status: verified; scope: selected native build and test discovery" in output
    assert "no tests or mutants measured" in output
    assert "not authenticated" in output
    assert "quota unknown" in output
    assert "success not guaranteed" in output
    assert "READY" not in output


@pytest.mark.parametrize("mode", ("plain", "detailed", "compact"))
@pytest.mark.parametrize("width", (40, 100))
def test_guest_inputs_and_host_output_are_separate_selected_locations(mode, width):
    event = _event(
        f"{_JAVA}; output: host C:/runs/Example Project; "
        "resume: explicit matching saved run only; guest project is not the host output; "
        "versions describe selected guest candidates, not host PATH or verified execution; "
        "inspection only; native readiness unverified; diagnostics: none saved",
    )
    output = _words(_render((event,), mode, width))
    for text in ("demo", "Project: /workspace/example", "host: Windows",
                 "measurement JDK: guest /opt/jdk-17 17.0.12",
                 "build JDK: guest /opt/jdk-21 21.0.4", "release target: 8",
                 "output: host C:/runs/Example Project", "matching saved run only",
                 "not host PATH or verified execution", "native readiness unverified"):
        assert text in output
    assert "Saved diagnostics:" not in output


@pytest.mark.parametrize("mode", ("plain", "detailed", "compact"))
@pytest.mark.parametrize("width", (40, 100))
@pytest.mark.parametrize("complete", (True, False))
def test_completion_lists_only_supplied_existing_artifacts(mode, width, complete, tmp_path):
    delivered = tmp_path / "test_alpha.py"
    delivered.write_text("# Authored fixture, not generated or executed.\n", encoding="utf-8")
    log = tmp_path / "run.log"
    log.write_text("Synthetic diagnostic.\n", encoding="utf-8")
    absent_report = tmp_path / "measurements.json"
    event = _event(
        f"Selected coverage result: {'complete' if complete else 'partial'}; "
        f"whole run completion: unverified; delivered tests: {delivered}; "
        f"Saved diagnostics: {log}; measurements: "
        + ("measured line coverage zero; no mutation measurement requested in this fixture"
           if complete else "incomplete; no headline mutation score; independent tests retained")
        + "; output/resume policy unchanged",
        operation="result", state="succeeded" if complete else "failed",
        metrics=(Metric("Line coverage", 0 if complete else None, "%",
                        "measured" if complete else "unavailable"),
                 Metric("Mutation score", basis="unavailable" if complete else "incomplete")),
    )
    output = _render((event,), mode, width)
    # Long temporary paths may wrap inside words. Compare their nonspace form.
    compact_text = "".join(safe_text(output).split())
    assert "".join(str(delivered).split()) in compact_text
    assert "".join(str(log).split()) in compact_text
    assert str(absent_report) not in output
    assert "demo" in output
    assert "wholeruncompletion:unverified" in compact_text
    if complete:
        # Synthetic observations: the value reads as synthetic, not measured.
        assert "Linecoverage:0%[demo]" in compact_text
        assert "Mutationscore:unavailable" in compact_text
    else:
        assert "Linecoverage:unavailable" in compact_text
        assert "Mutationscore:incomplete" in compact_text
        assert "noheadlinemutationscore" in compact_text
        assert "0%" not in output
    assert not absent_report.exists()


@pytest.mark.parametrize("mode", ("plain", "detailed", "compact"))
@pytest.mark.parametrize("width", (40, 100))
def test_unavailable_versions_and_limits_are_unknown_not_zero_or_supported(mode, width):
    event = _event(
        "Status: unverified; inspection only; project: /workspace/example; "
        "targets: src/alpha.py; context: Linux guest; subject Python version: unknown; "
        "guest helper version: unknown; lockfile exists, dependencies not verified; "
        "model: not selected; credentials: not configured; authentication: not attempted; "
        "model request limit: unresolved; token limit: unresolved; provider quota: unknown; "
        "output: not selected; resume: not requested; checked: project configuration; "
        "Next: select --model local-example and the prepared guest runtime before verification; "
        "diagnostics: none saved", state="waiting",
    )
    output = _words(_render((event,), mode, width))
    for text in ("demo", "subject Python version: unknown", "guest helper version: unknown",
                 "dependencies not verified", "not configured", "not attempted",
                 "limit: unresolved", "quota: unknown", "output: not selected",
                 "resume: not requested", "Next: select --model local-example"):
        assert text in output
    assert "0%" not in output
    assert "100%" not in output
    assert "READY" not in output
    assert "Saved diagnostics:" not in output


@pytest.mark.parametrize("mode", ("auto", "compact", "detailed", "plain"))
def test_redirected_setup_is_plain_and_does_not_claim_or_create_a_diagnostic(mode, tmp_path):
    stream = StringIO()
    event = _event(f"{_PYTHON}; {_RUN}; {_SCOPE}; diagnostics: none saved")
    with TerminalPresenter(mode, stream=stream, width=100) as presenter:
        assert presenter.mode == "plain"
        presenter(event)
    assert "\x1b" not in stream.getvalue()
    assert "diagnostics: none saved" in _words(stream.getvalue())
    assert "Saved diagnostics:" not in stream.getvalue()
    assert not tuple(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ("plain", "detailed", "compact"))
@pytest.mark.parametrize("width", (40, 100))
@pytest.mark.parametrize("prefix", ((), ("generate",)))
def test_actual_cli_setup_refusal_is_not_failed_generation_or_a_saved_log(
        mode, width, prefix, tmp_path, monkeypatch, capsys):
    from testuary import __main__ as cli
    from testuary import presentation, settings
    from testuary import config
    from testuary import project

    stream = StringIO()
    events = []
    displays = []
    reports = []
    original_report_writer = presentation.present_readiness

    class RecordingPresenter(TerminalPresenter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.diagnostic_writes = 0
            self.close_calls = 0

        def __call__(self, event):
            events.append(event)
            super().__call__(event)

        def write_diagnostic(self, write):
            self.diagnostic_writes += 1
            return super().write_diagnostic(write)

        def close(self):
            self.close_calls += 1
            return super().close()

    def present_report(report, *, presenter, before_run=False):
        assert displays == [presenter]
        assert not presenter._closed
        if mode == "compact":
            assert presenter._live is not None
        observations = tuple(presenter.view.latest().items())
        event_count = len(events)
        writes = presenter.diagnostic_writes
        # A refused generate ran no check: its report is the pre-run setup summary.
        assert before_run is True
        original_report_writer(report, presenter=presenter, before_run=before_run)
        assert presenter.diagnostic_writes == writes + 1
        assert tuple(presenter.view.latest().items()) == observations
        assert len(events) == event_count
        if mode == "compact":
            # The run already ended as refused: the pinned view is not started
            # again below the report.
            assert presenter._live is None
        reports.append(report)

    def presenter(selected_mode, *, no_color):
        console = Console(file=stream, force_terminal=True, color_system=None,
                          no_color=no_color, width=width, height=60,
                          highlight=False, markup=False)
        display = RecordingPresenter(selected_mode, console=console)
        assert display.mode == selected_mode
        displays.append(display)
        return display

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["testuary", *prefix, "--display", mode, "--no-color"])
    monkeypatch.setattr(settings, "resolve_settings",
                        lambda **kwargs: settings.RunSettings(base_dir=tmp_path))
    monkeypatch.setattr(project, "project_runtime",
                        lambda *args, **kwargs: pytest.fail("Setup refusal must precede provisioning"))
    monkeypatch.setattr(presentation, "TerminalPresenter", presenter)
    monkeypatch.setattr(presentation, "present_readiness", present_report)
    with pytest.raises(SystemExit) as stopped:
        cli._main()

    captured = capsys.readouterr()
    text = _words(stream.getvalue() + captured.out + captured.err)
    assert stopped.value.code == 2
    assert "model" in text
    assert "Checking run inputs" in text
    assert "Configuration refused" in text
    assert "Run started" not in text
    assert "Run has failed targets" not in text
    if mode == "detailed":
        # Generate ran no check: the detailed report is the run's own setup.
        assert text.count("Setup before the run") == 1 and "Testuary quick check" not in text
        assert "A check does not promise" not in text and "exit status" not in text
        assert "Source directory: not reported" in text and "Where it runs: not reported" in text
        assert "API key: unknown" in text and "Provider quota" not in text
        assert "Limits: not reported" in text
        assert "Fix: Correct it, then run the command again." in text
    else:
        # The summary names the problem in words and leaves unknown facts out.
        assert text.count("Setup before the run: 1 problem found.") == 1
        assert "quick check" not in text
        assert "Missing or blocking: Settings:" in text and "Testuary |" not in text
        assert "Source directory:" not in text and "API key" not in text
    assert text.count("Next step: Correct it, then run the command again.") == 1
    assert "check again" not in text
    assert "Diagnostic:" not in text
    assert "saved diagnostics" not in text.lower()
    assert "run.log" not in text and "main.jsonl" not in text
    assert [event.message for event in events] == ["Checking run inputs", "Configuration refused"]
    assert [event.state for event in events] == ["running", "refused"]
    assert all(event.phase == "pipeline" for event in events)
    assert all(event.usage is None and event.metrics == () for event in events)
    assert len(reports) == 1
    assert (reports[0].scope, reports[0].status, reports[0].exit_code) == ("inspection", "refused", 2)
    assert all(item.diagnostic_path is None for item in reports[0].items)
    assert len(displays) == 1 and displays[0].close_calls == 1
    assert displays[0]._closed and displays[0]._live is None
    assert config.active_settings() is None
    assert not tuple(tmp_path.iterdir())


def test_a_language_is_named_as_its_adapter_declares(monkeypatch):
    """The adapter's display_name, not its key with a capital: JavaScript,
    never Javascript. Python and Java declare their keys' own spelling."""
    import langadapter as la

    from testuary import _words

    class Adapter:
        display_name = "JavaScript"

    monkeypatch.setattr(la, "registration", lambda name: type("R", (), {"factory": Adapter}))
    assert _words.language_name("native.javascript.assessment") == "JavaScript"
    monkeypatch.undo()
    assert _words.language_name("native.python.assessment") == "Python"
    assert _words.language_name("native.java.item.jvm") == "Java"
    assert _words.language_name("configuration.model.coverage") == ""


@pytest.mark.parametrize("tried_by, expected", [
    ("run", "no check of its own (the run tries its operations)"),
    (None, "the Python check"),
])
def test_a_language_without_a_check_of_its_own_says_so_once(tried_by, expected):
    from testuary import _words
    from testuary.readiness import ReadinessItem

    item = ReadinessItem("native.python.assessment", True, "unverified", tried_by=tried_by)
    assert _words.plain_name(item) == expected

