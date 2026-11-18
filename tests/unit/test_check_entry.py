"""The public check and actual CLI consume one report without generation."""
import json
import runpy
import sys

import pytest

from agent_log import console_records
from langadapter import Availability, OperationCapability, Outcome, ProjectSetupItem, ProjectSetupReport
from testuary import __main__ as cli, orchestrator, presentation
from testuary.agent._helpers import AgentLogger
from testuary.orchestrator import RunRequest
from testuary.settings import ConfigurationError
from test_native_readiness_producer import (
    no_execution as no_execution, providers as providers, settings_for, sources,
)


def invoke(monkeypatch, settings, *flags):
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        return settings

    monkeypatch.setattr("testuary.settings.resolve_settings", resolve)
    monkeypatch.setattr(sys, "argv", ["testuary", "check", *flags])
    return calls


@pytest.mark.parametrize("verify", [False, True])
def test_public_api_returns_same_scope_without_terminal_or_pipeline_side_effects(
        tmp_path, providers, monkeypatch, capsys, verify):
    sources(tmp_path, "module.native-a")
    monkeypatch.setattr(orchestrator, "run_pipeline", lambda *a, **kw: pytest.fail("generation"))
    settings = settings_for(tmp_path, providers)
    report = orchestrator.check(RunRequest(("module.native-a",)), settings=settings, verify=verify)
    assert report.scope == ("verification" if verify else "inspection")
    assert report.status == "unverified" and report.exit_code == (1 if verify else 0)
    assert capsys.readouterr() == ("", "") and not (tmp_path / "output").exists()


@pytest.mark.parametrize("verify", [False, True])
def test_actual_json_cli_matches_public_api_and_has_no_presenter_or_generation(
        tmp_path, providers, monkeypatch, capsys, verify):
    sources(tmp_path, "module.native-a")
    settings = settings_for(tmp_path, providers)
    expected = orchestrator.check(RunRequest(("module.native-a",)), settings=settings, verify=verify)
    calls = invoke(monkeypatch, settings, "--source-files", "module.native-a", "--json",
                   *("--verify",) if verify else ())
    monkeypatch.setattr(presentation, "TerminalPresenter", lambda *a, **kw: pytest.fail("JSON presenter"))
    monkeypatch.setattr(orchestrator, "run_pipeline", lambda *a, **kw: pytest.fail("generation"))
    assert cli._main() == expected.exit_code
    captured = capsys.readouterr()
    assert json.loads(captured.out) == expected.as_dict() and captured.err == ""
    assert len(calls) == 1 and not (tmp_path / "output").exists()


@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("json_output", [False, True])
def test_actual_success_uses_one_c_consumer_or_only_json(tmp_path, providers, monkeypatch, capsys, verify, json_output):
    sources(tmp_path, "module.native-a")
    settings = settings_for(tmp_path, providers)
    flags = ["--source-files", "module.native-a", "--display", "plain", "--no-color"]
    if verify:
        flags.append("--verify")
    if json_output:
        flags.append("--json")
    calls = invoke(monkeypatch, settings, *flags)
    delivered = []
    original = presentation.present_readiness

    def present(report, *, presenter):
        delivered.append((report, presenter))
        original(report, presenter=presenter)

    monkeypatch.setattr(presentation, "present_readiness", present)
    assert cli._main() == (1 if verify else 0)
    captured = capsys.readouterr()
    assert len(calls) == 1 and len(delivered) == (0 if json_output else 1)
    if json_output:
        assert json.loads(captured.out)["scope"] == ("verification" if verify else "inspection")
        assert captured.err == ""
    else:
        assert captured.out == "" and captured.err and "\x1b" not in captured.err
        assert delivered[0][0].scope == ("verification" if verify else "inspection")


@pytest.mark.parametrize("flags", [
    (), ("--source-files", "module.native-a", "--runs", "0"),
    ("--source-files", "module.native-a", "--runs", "2", "--run", "1"),
    ("--source-files", "module.native-a", "--languages", "unknown"),
    ("--source-files", "module.native-a", "--resume", "--reuse", "coverage"),
    ("--source-files", "module.native-a", "--reuse", "bogus"),
    ("--source-files", "module.native-a", "--stop-after-phase", "bogus"),
    ("--source-files", "../outside.native-a"),
    ("--source-files", "module.unknown"),
])
@pytest.mark.parametrize("verify", [False, True])
def test_actual_json_input_refusal_is_one_unknown_safe_report_before_native(
        tmp_path, providers, monkeypatch, capsys, flags, verify):
    settings = settings_for(tmp_path, providers)
    invoke(monkeypatch, settings, "--json", *flags, *("--verify",) if verify else ())
    with pytest.raises(SystemExit) as caught:
        cli._main()
    assert caught.value.code == 2 and isinstance(caught.value.__cause__, ConfigurationError)
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["scope"] == ("verification" if verify else "inspection")
    assert report["status"] == "refused" and report["exit_code"] == 2
    assert report["source_dir"] is report["model"] is report["output_dir"] is None
    assert captured.err == "" and not providers[1] and not list(tmp_path.iterdir())


def test_resolution_refusal_uses_same_json_boundary_without_inventing_selection(tmp_path, providers, monkeypatch, capsys):
    invoke(monkeypatch, settings_for(tmp_path, providers), "--source-files", "module.native-a", "--verify", "--json")
    error = ValueError("unknown configuration field")

    def fail(**kwargs):
        raise error

    monkeypatch.setattr("testuary.settings.resolve_settings", fail)
    with pytest.raises(SystemExit) as caught:
        cli._main()
    report = json.loads(capsys.readouterr().out)
    assert report["scope"] == "verification" and report["source_files"] == [] and report["limits"] == []
    assert caught.value.__cause__.__cause__ is error and report["exit_code"] == 2


@pytest.mark.parametrize("flags, named", [
    (("--source-files", "module.unknown"), "module.unknown"),
    (("--source-files", "module.native-a", "--reuse", "bogus"), "bogus"),
    (("--source-files", "module.native-a", "--stop-after-phase", "bogus"), "--stop-after-phase"),
])
def test_human_refusal_calls_same_presenter_once_and_does_not_execute(
        tmp_path, providers, monkeypatch, capsys, flags, named):
    settings = settings_for(tmp_path, providers)
    invoke(monkeypatch, settings, *flags, "--display", "plain", "--no-color")
    calls = []
    original = presentation.present_readiness

    def present(report, *, presenter):
        calls.append(report)
        original(report, presenter=presenter)

    monkeypatch.setattr(presentation, "present_readiness", present)
    with pytest.raises(SystemExit) as caught:
        cli._main()
    captured = capsys.readouterr()
    assert caught.value.code == 2 and len(calls) == 1
    assert captured.out == "" and named in captured.err and "\x1b" not in captured.err
    assert not providers[1] and not list(tmp_path.iterdir())


@pytest.mark.parametrize("scope", ["inspection", "verification"])
@pytest.mark.parametrize("ambiguous,status,code", [(True, "ambiguous", 2), (False, "unsupported", 1)])
def test_native_scope_status_and_exit_are_not_replaced_by_cli(tmp_path, providers, monkeypatch, capsys, scope, ambiguous, status, code):
    sources(tmp_path, "module.native-a")

    def observe(request, actual_scope):
        return ProjectSetupReport(actual_scope, Outcome.COMPLETED if ambiguous else Outcome.UNSUPPORTED,
            "unselected runtime", None if ambiguous else request.context, request.inputs,
            (ProjectSetupItem("runtime", True, None, None, ambiguous=ambiguous),),
            tuple(OperationCapability(name, False, readiness=Availability.UNSUPPORTED)
                  for name in request.required_operations))

    providers[2][providers[0][0]] = observe
    invoke(monkeypatch, settings_for(tmp_path, providers), "--source-files", "module.native-a", "--json",
           *("--verify",) if scope == "verification" else ())
    assert cli._main() == code
    report = json.loads(capsys.readouterr().out)
    assert report["scope"] == scope and report["status"] == status and report["exit_code"] == code


@pytest.mark.parametrize("error", [KeyboardInterrupt(), SystemExit(9), RuntimeError("unexpected observer failure")])
def test_api_and_actual_cli_do_not_turn_unexpected_causes_into_setup_success(tmp_path, providers, monkeypatch, capsys, error):
    sources(tmp_path, "module.native-a")

    def fail(request, scope):
        raise error

    providers[2][providers[0][0]] = fail
    settings = settings_for(tmp_path, providers)
    for api in (True, False):
        if not api:
            invoke(monkeypatch, settings, "--source-files", "module.native-a", "--json")
        with pytest.raises(type(error)) as caught:
            (orchestrator.check(RunRequest(("module.native-a",)), settings=settings) if api else cli._main())
        assert caught.value is error
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("argv", [["check", "--help"], ["check", "--demo"], ["generate", "--json"], ["--verify"]])
def test_parsed_help_and_check_only_controls_do_not_access_configuration(monkeypatch, capsys, argv):
    monkeypatch.setattr(sys, "argv", ["testuary", *argv])
    monkeypatch.setattr("testuary.settings.resolve_settings", lambda **kw: pytest.fail("help configuration"))
    with pytest.raises(SystemExit) as caught:
        cli._main()
    assert caught.value.code == (0 if "--help" in argv else 2)
    captured = capsys.readouterr()
    if "--help" in argv:
        assert "--verify" in captured.out and "--json" in captured.out


def test_check_cli_preserves_targets_repeat_and_explicit_root_overrides(tmp_path, providers, monkeypatch, capsys):
    sources(tmp_path, "a.native-a", "b.native-b")
    settings = settings_for(tmp_path, providers)
    calls = invoke(monkeypatch, settings, "--source-files", "a.native-a,b.native-b", "--targets", "2",
                   "--runs", "3", "--resume", "--source-dir", str(tmp_path / "original"),
                   "--output-dir", str(tmp_path / "output"), "--json")
    assert cli._main() == 0
    report = json.loads(capsys.readouterr().out)
    assert report["source_files"] == ["b.native-b"] and report["output_mode"] == "resume"
    assert len(providers[1]) == 1 and providers[1][0][0] == providers[0][1]
    assert calls[0]["overrides"]["project"]["source_dir"] == str(tmp_path / "original")
    assert calls[0]["overrides"]["project"]["output_dir"] == str(tmp_path / "output")


def console_entry(entry):
    if entry == "console":
        return cli.main()
    return runpy.run_path(cli.__file__, run_name="__main__")


@pytest.mark.parametrize("entry", ["console", "module"])
@pytest.mark.parametrize("json_output", [False, True])
@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("prior_logger", [False, True])
def test_real_entry_check_failure_preserves_previous_logs_stdout_and_owner(
        tmp_path, providers, monkeypatch, capsys, entry, json_output, verify, prior_logger):
    root = sources(tmp_path, "module.native-a")
    original = (root / "module.native-a").read_bytes()
    settings = settings_for(tmp_path, providers)
    flags = ["--source-files", "module.native-a", "--display", "plain", "--no-color"]
    if json_output:
        flags.append("--json")
    if verify:
        flags.append("--verify")
    invoke(monkeypatch, settings, *flags)
    error = RuntimeError("unresolved native inspection failure")
    owner = object()
    error.cleanup_handle = owner

    def fail(request, scope):
        raise error

    providers[2][providers[0][0]] = fail
    logger = None
    before = None
    if prior_logger:
        logger = AgentLogger("previous-run")
        log_path = tmp_path / "previous" / "main.jsonl"
        logger.enable_streaming(str(log_path.parent), log_path.name)
        logger.log("Previously completed work")
        before = log_path.read_bytes()
    monkeypatch.setattr(orchestrator._log, "logger", logger)
    capsys.readouterr()
    with pytest.raises(RuntimeError) as caught:
        console_entry(entry)
    assert caught.value is error and caught.value.cleanup_handle is owner
    assert orchestrator.get_main_logger() is logger
    captured = capsys.readouterr()
    assert captured.out == ""
    if json_output:
        assert captured.err == ""
    else:
        assert "FAILED | unresolved native inspection failure" in captured.err
        assert "\x1b" not in captured.err
    if prior_logger:
        assert log_path.read_bytes() == before
        assert len(logger._console_logs) == 1
    assert (root / "module.native-a").read_bytes() == original
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("entry", ["console", "module"])
@pytest.mark.parametrize("refused", [False, True])
def test_real_entry_json_success_and_refusal_stay_single_read_only_reports(
        tmp_path, providers, monkeypatch, capsys, entry, refused):
    sources(tmp_path, "module.native-a")
    invoke(monkeypatch, settings_for(tmp_path, providers), "--json", "--source-files",
           "module.unknown" if refused else "module.native-a")
    monkeypatch.setattr(orchestrator, "log_print", lambda *a, **kw: pytest.fail("generation fatal log"))
    monkeypatch.setattr(cli, "log_print", orchestrator.log_print)
    if refused or entry == "module":
        with pytest.raises(SystemExit) as caught:
            console_entry(entry)
        assert caught.value.code == (2 if refused else 0)
    else:
        assert console_entry(entry) == 0
    captured = capsys.readouterr()
    report = json.loads(captured.out)
    assert report["exit_code"] == (2 if refused else 0)
    assert captured.err == "" and not (tmp_path / "output").exists()


@pytest.mark.parametrize("entry", ["console", "module"])
@pytest.mark.parametrize("explicit", [False, True])
def test_generation_fatal_logging_remains_at_real_entry(tmp_path, providers, monkeypatch, capsys, entry, explicit):
    settings = settings_for(tmp_path, providers)
    invoke(monkeypatch, settings, "--source-files", "module.native-a", "--display", "plain", "--no-color")
    monkeypatch.setattr(sys, "argv", ["testuary", *(["generate"] if explicit else []),
                                     "--source-files", "module.native-a", "--display", "plain", "--no-color"])
    error = RuntimeError("generation failure control")
    logger = AgentLogger("current-run")
    path = tmp_path / "current" / "main.jsonl"
    logger.enable_streaming(str(path.parent), path.name)
    monkeypatch.setattr(orchestrator._log, "logger", logger)

    def fail(*args, **kwargs):
        raise error

    monkeypatch.setattr(orchestrator, "run_pipeline", fail)
    with pytest.raises(RuntimeError) as caught:
        console_entry(entry)
    assert caught.value is error
    messages = [row["message"] for row in console_records(path)]
    # One entry whichever way the command started (``python -m testuary``
    # or the installed script): run_command()'s.
    assert len(messages) == 1
    assert messages[0].startswith("FATAL: uncaught exception ended the run:")
    assert "generation failure control" in messages[0]
    assert "FATAL:" in capsys.readouterr().out
