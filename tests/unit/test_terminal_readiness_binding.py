"""Controlled typed-report rendering; no native readiness or provider evidence."""
from __future__ import annotations

from dataclasses import replace
from io import StringIO
from pathlib import Path
import builtins
import os
import socket
import subprocess

import pytest
from rich.console import Console

from testuary.presentation import DisplayEvent, TerminalPresenter, present_readiness
from testuary import _words
from testuary.presentation import readiness as shown_report
from testuary.readiness import ReadinessItem, ReadinessLimit, RunReadiness


_STATUSES = ("verified", "missing_setup", "unsupported", "ambiguous", "unverified", "failed")


def _report(**changes):
    report = RunReadiness(
        scope="inspection", status="verified", exit_code=0,
        source_dir="/work/Example  Project", execution="container",
        source_files=("src/alpha,one.py", "src/second target.py"),
        phases=("starter", "coverage", "mutation"), model_alias="example",
        provider="local", model="example-model", credentials_configured=True,
        output_dir="C:/runs/generated suites", output_mode="fresh",
        limits=(ReadinessLimit("starter", "model requests", 0, "requests", "configured"),
                ReadinessLimit("coverage", "wall clock", None, "seconds", "unlimited"),
                ReadinessLimit("mutation", "tokens", None, "tokens", "unknown")),
        items=(ReadinessItem("framework Python", False, "verified", "host Python", "3.12.3"),
               ReadinessItem("measurement JDK", True, "unverified", "guest /opt/jdk-17", "17.0.12",
                             "Only the selected candidate version is inspected",
                             "Run bounded verification with the selected guest inputs"),
               ReadinessItem("build JDK", True, "verified", "guest /opt/jdk-21", "21.0.4"),
               ReadinessItem("release target", True, "verified", "8"),
               ReadinessItem("wrapper", True, "verified", "guest Maven", "3.9.9")),
    )
    return replace(report, **changes)


def _display(mode, width, **kwargs):
    stream = StringIO()
    console = Console(file=stream, force_terminal=True, color_system=None,
                      width=width, height=60, highlight=False, markup=False)
    presenter = TerminalPresenter(mode, console=console, **kwargs)
    assert presenter.mode == mode
    return stream, presenter


def _synthetic(report, presenter):
    # The report has no provenance field, so the fixture label is written beside it.
    presenter.write_diagnostic(lambda: presenter._write_plain("Synthetic readiness fixture"))
    return present_readiness(report, presenter=presenter)


def _flat(output):
    return "".join(output.split())


@pytest.fixture(autouse=True)
def no_native_or_provider_work(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")

    def forbidden(*args, **kwargs):
        raise AssertionError("Readiness presentation must not execute or connect")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(socket.socket, "connect", forbidden)


@pytest.mark.parametrize("mode", ("plain", "detailed", "compact"))
@pytest.mark.parametrize("width", (40, 100))
@pytest.mark.parametrize("scope", ("inspection", "verification"))
@pytest.mark.parametrize("status", _STATUSES)
def test_actual_type_preserves_scope_and_reported_status_without_generation(mode, width, scope, status):
    report = _report(scope=scope, status=status, exit_code=2)
    before = report.as_dict()
    stream, presenter = _display(mode, width)
    with presenter:
        assert _synthetic(report, presenter) is None
        assert not presenter.view.latest() and presenter._live is None
    output = stream.getvalue()
    flat = _flat(output)
    assert "Syntheticreadinessfixture" in flat
    if mode == "detailed":
        check = "tool check" if scope == "verification" else "quick check"
        assert _flat(f"Testuary {check}: {shown_report._OVERALL[status]} (exit status 2)") in flat
        assert "Target:src/alpha,one.py" in flat
        assert "Target:src/secondtarget.py" in flat
        assert "Whereitruns:inacontainer" in flat
        assert "Model:example|provider:local|modelnameattheservice:example-model" in flat
        assert "APIkey:set,nottried" in flat
        assert "Providerquota" not in flat
        assert "RunDirectory:C:/runs/generatedsuites|anewrun".lower() in flat.lower()
        assert "Phases:Starterphase,Coveragephase,Mutationphase" in flat
        assert "Limit|Starterphase|modelrequests:0requests" in flat
        assert "wallclock:unlimited" in flat and "tokens:unknown" in flat
        assert "[configured]" not in flat and "[unlimited]" not in flat
        expected = ("framework Python (optional)", "measurement JDK (required)",
                    "guest /opt/jdk-17", "version: 17.0.12", "guest /opt/jdk-21",
                    "version: 21.0.4", "release target", "Selected: 8", "guest Maven",
                    "version: 3.9.9", "Fix:")
    else:
        # The summary states facts in words; the overall status label stays detailed.
        reach = "tool check" if scope == "verification" else "quick check"
        assert output.splitlines()[1].startswith(f"No problems found by a {reach}")
        expected = ("Target" + "s:", "  src/alpha,one.py", "  src/second target.py",
                    "Model: example-model via local",
                    "API key: found, not tried by this check.",
                    "Run directory: C:/runs/generated suites (a new run)",
                    "Limits: starter phase model requests: 0 requests | coverage phase no wall "
                    "clock limit | mutation phase tokens limit unknown",
                    "  framework Python host Python (3.12.3) (optional)",
                    "  build JDK guest /opt/jdk-21 (21.0.4)", "  release target 8",
                    "  wrapper guest Maven (3.9.9)",
                    "Installed but not tried:", "  measurement JDK guest /opt/jdk-17 (17.0.12)",
                    "    Run bounded verification with the selected guest inputs")
        for line in expected:
            assert line in output.splitlines()
        assert "Source directory: /work/Example  Project (in a container)" in output.splitlines()
        assert "exit code" not in output and status.replace("_", " ") + " |" not in output
    if mode == "detailed":
        for text in expected:
            assert _flat(text) in flat
    assert "Run started" not in output and "Run has failed targets" not in output
    assert "READY" not in output and "Diagnostic:" not in output
    assert "100%" not in output and "00:00:" not in output
    assert report.as_dict() == before
    # Records are not split for the terminal width; paths stay whole.
    if mode == "detailed":
        assert "Source directory: /work/Example  Project | Where it runs: in a container" in output.splitlines()


@pytest.mark.parametrize("mode", ("plain", "detailed", "compact"))
@pytest.mark.parametrize("configured", (True, False, None))
@pytest.mark.parametrize("policy", ("fresh", "resume", "reuse"))
def test_unknown_selection_credentials_and_limits_are_not_inferred(mode, configured, policy):
    report = _report(source_dir=None, execution=None, source_files=(), phases=(),
                     model_alias=None, provider=None, model=None,
                     credentials_configured=configured, output_dir=None,
                     output_mode=policy, limits=(), items=())
    stream, presenter = _display(mode, 100)
    with presenter:
        _synthetic(report, presenter)
    output = stream.getvalue()
    flat = _flat(output)
    if mode == "detailed":
        mode_words = {"fresh": "a new run", "resume": "continues an earlier run in this directory",
                      "reuse": "reuses results of an earlier run"}[policy]
        for text in ("Source directory: not reported", "Where it runs: not reported",
                     "Targets: none reported", "Phases: none reported",
                     "Model: not reported", "provider: not reported", "Run directory: not reported",
                     mode_words, "Limits: not reported", "Items: none reported"):
            assert _flat(text) in flat
        label = {True: "set, not tried", False: "not set", None: "unknown"}[configured]
        assert _flat(f"API key: {label}") in flat
    else:
        # Unknown facts are left out of the summary rather than guessed.
        sentence = {True: "API key: found, not tried by this check.",
                    False: "API key: none found.", None: None}[configured]
        for absent in ("Source directory:", "Target", "Model:", "Run directory:", "Phases:", "Limits:", "Found"):
            assert absent not in output
        if sentence:
            assert any(line.startswith(sentence) for line in output.splitlines())
        else:
            assert "API key" not in output
        assert "--display detailed" not in output
    assert "Diagnostic:" not in output
    assert "0%" not in output and "unlimited" not in output


@pytest.mark.parametrize("mode", ("plain", "detailed"))
@pytest.mark.parametrize("status", _STATUSES)
def test_item_reason_and_action_are_used_without_recomputing_overall_status(mode, status):
    action = 'Set --python-executable "/work/Example  Project/.venv/bin/python"'
    reason = "Checked .venv and activated environment; selected version not known"
    report = _report(status="unverified", exit_code=0, items=(
        ReadinessItem("subject Python", True, status, "/work/Example  Project/.venv/bin/python", None,
                      reason, action),))
    stream, presenter = _display(mode, 100)
    with presenter:
        _synthetic(report, presenter)
    output = stream.getvalue()
    label = status.replace("_", " ")
    if mode == "detailed":
        assert "Testuary quick check: some required parts not tried yet (exit status 0)" in output
        assert "Source directory: /work/Example  Project" in output
        assert f"subject Python (required): {_words.status_word(status)}" in output
        assert "version: not reported" in output
        assert reason in output and action in output
        return
    lines = output.splitlines()
    assert "Source directory: /work/Example  Project (in a container)" in lines
    if status == "verified":
        # A verified item shows its selection; its reason stays in detailed.
        assert "  subject Python /work/Example  Project/.venv/bin/python" in lines
        assert reason not in output and action not in output
    elif status == "unverified":
        # Not tried yet: listed with the producer's action, not counted as a problem.
        assert lines[1].startswith("No problems found")
        start = lines.index("Not tried yet:")
        assert lines[start + 1:start + 3] == [
            f"  subject Python /work/Example  Project/.venv/bin/python: {reason}", f"    {action}"]
    else:
        assert lines[1].startswith("1 problem found")
        assert f"  Subject Python (/work/Example  Project/.venv/bin/python): {reason}" in lines
        # A single problem's action is the closing next step.
        assert lines[-1] == f"Next step: {action}"
    assert "Testuary quick check:" not in output and label + " |" not in output


@pytest.mark.parametrize("mode", ("plain", "detailed"))
def test_missing_action_is_reported_not_invented(mode):
    report = _report(items=(ReadinessItem("native discovery", True, "failed"),))
    stream, presenter = _display(mode, 100)
    with presenter:
        _synthetic(report, presenter)
    output = stream.getvalue()
    # Without a supplied fix the next step only points to the full observations.
    assert output.splitlines()[-1] == (
        "Next step: read every observation with the same command and --display detailed, "
        "then fix the problem and check again.")
    assert "Fix:" not in output
    assert "Diagnostic:" not in output


@pytest.mark.parametrize("diagnostic", (None, "", "provided"))
def test_only_producer_supplied_diagnostic_is_displayed_without_path_inspection(diagnostic, tmp_path, monkeypatch):
    actual = tmp_path / "native-discovery.txt"
    actual.write_text("Authored fixture diagnostic, not a native result.\n", encoding="utf-8")
    path = str(actual) if diagnostic == "provided" else diagnostic
    report = _report(items=(ReadinessItem("native discovery", True, "failed",
                                         reason="Discovery did not complete", diagnostic_path=path),))
    stream, presenter = _display("plain", 100)

    def forbidden(*args, **kwargs):
        raise AssertionError("Diagnostic availability belongs to the producer")

    with presenter, monkeypatch.context() as guarded:
        guarded.setattr(builtins, "open", forbidden)
        guarded.setattr(os, "stat", forbidden)
        for method in ("exists", "is_file", "stat", "open"):
            guarded.setattr(Path, method, forbidden)
        _synthetic(report, presenter)
    if diagnostic == "provided":
        assert f"    More detail: {actual}" in stream.getvalue().splitlines()
    else:
        assert "More detail" not in stream.getvalue()
    assert actual.read_text(encoding="utf-8").startswith("Authored fixture")


def test_report_uses_one_existing_diagnostic_boundary_and_no_events_or_second_live(monkeypatch):
    stream, presenter = _display("compact", 100)
    calls = []
    with presenter:
        presenter(DisplayEvent("control", "selected target", "setup", "control", "running", 1,
                               provenance="synthetic"))
        previous = tuple(presenter.view.latest().items())
        original = presenter.write_diagnostic

        def boundary(write):
            calls.append(write)
            return original(write)

        monkeypatch.setattr(presenter, "write_diagnostic", boundary)
        present_readiness(_report(), presenter=presenter)
        assert len(calls) == 1
        assert tuple(presenter.view.latest().items()) == previous
        assert presenter._live is not None
    assert presenter._closed and presenter._live is None
    assert "No problems found by a quick check" in stream.getvalue()


def test_readiness_retains_console_redaction_controls_and_exact_path_spaces():
    secret = "fixture-private-value"
    report = _report(items=(ReadinessItem(
        "native discovery", True, "failed", "\x1b[31m/work/Example  Project\x1b[0m",
        reason=f"api_key={secret}; Bearer fixture-bearer; https://user:pass@example.invalid/?token=value",
        action=f"Use {secret} only in private configuration"),))
    stream, presenter = _display("plain", 100, secrets=(secret,))
    with presenter:
        _synthetic(report, presenter)
    output = stream.getvalue()
    # The selection is shown with its exact spaces; the console removes the color codes.
    assert any(line.startswith("  Native discovery (/work/Example  Project): ")
               for line in output.splitlines())
    assert secret not in output and "fixture-bearer" not in output
    assert "user:pass" not in output and "token=value" not in output
    assert "[redacted]" in output and "\x1b" not in output


def test_large_integer_limit_is_not_rounded_into_a_float():
    value = 2 ** 80 + 1
    report = _report(limits=(ReadinessLimit("starter", "tokens", value, "tokens", "configured"),))
    stream, presenter = _display("plain", 100)
    with presenter:
        _synthetic(report, presenter)
    assert str(value) in stream.getvalue()


def test_invalid_report_refuses_before_any_output():
    stream, presenter = _display("plain", 100)
    with presenter, pytest.raises(TypeError, match="RunReadiness"):
        present_readiness({}, presenter=presenter)
    assert stream.getvalue() == ""
