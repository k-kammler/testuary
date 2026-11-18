"""Explicit run diagnostics coexist with presentation without stream replacement."""
import builtins
import io
import sys
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace

import pytest

from agent_log import console_records
from testuary import __main__ as cli, orchestrator, settings
from testuary import _chat_log
from testuary import _progress
from testuary.agent._base import CodingAgent
from testuary.agent import _helpers as helpers
from testuary.agent._helpers import AgentLogger
from testuary._progress import console_preferences, diagnostic_print
from testuary import measurement as metrics
from testuary.agent import coverage_agent
from testuary.presentation import TerminalPresenter
from testuary.settings import RunSettings


def test_scoped_writer_preserves_exact_print_arguments_and_global_streams():
    destination = io.StringIO()
    observed = []
    original = sys.stdout, sys.stderr, builtins.print

    def writer(write):
        observed.append("write")
        write()

    with console_preferences(diagnostic_writer=writer):
        diagnostic_print("first", "second", sep="/", end="!", file=destination, flush=True)
        with console_preferences():
            diagnostic_print("nested plain", file=destination)
        diagnostic_print("last", file=destination)
    diagnostic_print("outside", file=destination)
    assert destination.getvalue() == "first/second!nested plain\nlast\noutside\n"
    assert observed == ["write", "write"]
    assert (sys.stdout, sys.stderr, builtins.print) == original


@pytest.mark.parametrize("exception", [OSError("write failed"), RuntimeError("write failed"),
                                       KeyboardInterrupt(), SystemExit(2)])
def test_real_presenter_does_not_retry_or_hide_an_original_diagnostic_error(exception):
    attempts = []

    class Broken:
        def write(self, text):
            attempts.append(text)
            raise exception

    with TerminalPresenter("plain", stream=io.StringIO()) as presenter:
        with console_preferences(diagnostic_writer=presenter.write_diagnostic):
            with pytest.raises(type(exception)) as caught:
                diagnostic_print("one diagnostic", file=Broken())
    assert caught.value is exception
    assert attempts == ["one diagnostic"]


@pytest.mark.parametrize("exception", [None, RuntimeError("run failed"), KeyboardInterrupt()])
def test_actual_run_scopes_writer_across_workers_and_restores_it_on_every_exit(exception, monkeypatch, capsys):
    calls, events = [], []
    before = sys.stdout, sys.stderr, builtins.print

    def writer(write):
        calls.append("callback")
        write()

    def execute(args, languages):
        with ThreadPoolExecutor(max_workers=1) as pool:
            pool.submit(diagnostic_print, "worker diagnostic").result()
        orchestrator.log_print("main fallback")
        if exception is not None:
            raise exception
        return 0

    monkeypatch.setattr(orchestrator, "_run_pipeline", execute)
    args = SimpleNamespace(_diagnostic_writer=writer)
    if exception is None:
        assert orchestrator.run_pipeline(args, settings=RunSettings(), event_sink=events.append) == 0
    else:
        with pytest.raises(type(exception)) as caught:
            orchestrator.run_pipeline(args, settings=RunSettings(), event_sink=events.append)
        assert caught.value is exception
    diagnostic_print("outside")
    assert calls == ["callback", "callback"]
    assert capsys.readouterr().out == "worker diagnostic\nmain fallback\noutside\n"
    assert (sys.stdout, sys.stderr, builtins.print) == before
    assert events[-1].state == ("succeeded" if exception is None else
                                "cancelled" if isinstance(exception, KeyboardInterrupt) else "failed")
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda *a: 0)
    assert orchestrator.run_pipeline(SimpleNamespace(), settings=RunSettings()) == 0


def test_actual_logger_routing_preserves_saved_debug_and_failure_diagnostics(tmp_path, monkeypatch, capsys):
    logger = AgentLogger("subject", "coverage")
    saved = tmp_path / "coverage.jsonl"
    logger.enable_streaming(str(tmp_path), filename_template=saved.name)
    calls = []

    def writer(write):
        calls.append("callback")
        write()

    with console_preferences(plain=True, diagnostic_writer=writer):
        logger.log("visible diagnostic", level=1)
        logger.debug("saved detail")
        assert [row["message"] for row in console_records(saved)] == ["visible diagnostic", "saved detail"]

        def fail(path, *args, **kwargs):
            raise OSError("offline saved-log failure")

        # The log's file cannot be opened any more.
        monkeypatch.setattr(helpers, "open", fail, raising=False)
        logger.log("diagnostic after write failure")
    assert len(calls) == 3
    output = capsys.readouterr().out
    assert "visible diagnostic" in output and "saved detail" not in output and "\x1b" not in output
    assert "agent log" in output and "could not be written" in output
    assert "diagnostic after write failure" in output
    assert logger._console_logs[-1]["message"] == "diagnostic after write failure"


def test_explicit_fallback_and_chat_failure_writers_use_one_boundary(tmp_path, monkeypatch, capsys):
    calls = []

    def writer(write):
        calls.append("callback")
        write()

    monkeypatch.setattr(metrics._log, "logger", None)
    agent = CodingAgent({}, None, None)
    blocked = tmp_path / "not-a-directory"
    blocked.write_text("file", encoding="utf-8")
    with console_preferences(diagnostic_writer=writer):
        agent.log("agent fallback")
        metrics.log_print("metrics fallback")
        assert _chat_log.export_chat_log([], str(blocked), "chat.json") is None
    assert len(calls) == 3
    output = capsys.readouterr().out
    assert output.startswith("agent fallback\nmetrics fallback\n")
    assert "WARNING: the chat log chat.json could not be written" in output


def test_real_cli_attaches_presenter_writer_without_configuration_override(monkeypatch, capsys):
    received = []
    before = sys.stdout, sys.stderr, builtins.print
    monkeypatch.setattr(sys, "argv", ["testuary", "--display", "plain", "--no-color"])
    monkeypatch.setattr(settings, "resolve_settings", lambda **kw: RunSettings())
    original = TerminalPresenter.write_diagnostic

    def observed(presenter, write):
        received.append(presenter)
        original(presenter, write)

    monkeypatch.setattr(TerminalPresenter, "write_diagnostic", observed)

    def execute(args, languages):
        assert isinstance(args._diagnostic_writer.__self__, TerminalPresenter)
        assert orchestrator._cli_settings_overrides(args) == {}
        orchestrator.log_print("CLI diagnostic")
        return 0

    monkeypatch.setattr(orchestrator, "_run_pipeline", execute)
    assert cli._main() == 0
    assert len(received) == 1
    captured = capsys.readouterr()
    assert captured.out == "CLI diagnostic\n"
    assert "Checking run inputs" in captured.err and "\x1b" not in captured.err
    assert (sys.stdout, sys.stderr, builtins.print) == before


def test_progressive_and_retry_chat_failures_route_existing_diagnostics(tmp_path, capsys):
    writes = []

    def writer(write):
        writes.append("callback")
        write()

    # A file where the log directory should be: each write fails as a full or
    # read-only disk would make it fail.
    blocked = tmp_path / "not-a-dir"
    blocked.write_text("file", encoding="utf-8")
    flush = _chat_log.make_chat_log_flusher(output_dir=str(blocked), filename="unused.json")
    snapshot = _chat_log.make_retry_chat_snapshotter(output_dir=str(blocked), filename_prefix="unused")
    with console_preferences(diagnostic_writer=writer):
        flush([], stage="in_progress")
        snapshot(1, {"messages": []}, RuntimeError("rejected output"))
    assert len(writes) == 2
    output = capsys.readouterr().out
    assert "WARNING: the chat log unused.json could not be written" in output
    assert "WARNING: the chat log unused_pre_retry_001.json could not be written" in output


@pytest.mark.parametrize("restore_fails", [False, True])
def test_source_integrity_failure_diagnostics_preserve_existing_repair(restore_fails, tmp_path, monkeypatch, capsys):
    from pathlib import Path
    from testuary import target_phases as phases

    canonical, work = tmp_path / "canonical.py", tmp_path / "work.py"
    canonical.write_bytes(b"pass\n")
    work.write_bytes(b"divergent\n")
    if restore_fails:
        original = Path.write_bytes

        def failed(path, data):
            if path == work:
                raise OSError("offline repair failure")
            return original(path, data)

        monkeypatch.setattr(Path, "write_bytes", failed)
    writes = []

    def writer(write):
        writes.append("callback")
        write()

    ctx = SimpleNamespace(top_file={"file_path": str(work)}, canonical_path=canonical)
    with console_preferences(diagnostic_writer=writer), phases._source_integrity_guard(ctx, "coverage"):
        pass
    assert len(writes) == (2 if restore_fails else 1)
    output = capsys.readouterr().out
    assert "[CRITICAL] coverage entry" in output
    assert ("copying the original back failed" in output) is restore_fails
    assert work.read_bytes() == (b"divergent\n" if restore_fails else canonical.read_bytes())


def test_each_module_log_keeps_its_console_policy_before_its_logger(monkeypatch):
    """Before a module's log starts, a line follows that module's console
    policy: an agent module prints only a line the console shows anyway,
    finishing prints every line as a diagnostic, and the run's main log
    writes the line with its level to the plain log and prints it. With a
    logger, ``echo`` reaches it only when set."""
    printed, plain, logged = [], [], []
    monkeypatch.setattr(_progress, "diagnostic_print",
                        lambda message, logged=False: printed.append((message, logged)))
    monkeypatch.setattr(_progress, "write_plain_log",
                        lambda message, level=0: plain.append((message, level)))
    monkeypatch.setattr(_progress, "_DETAIL_ECHO", False)
    for module_log in (coverage_agent._log, metrics._log, orchestrator._log):
        monkeypatch.setattr(module_log, "logger", None)

    coverage_agent.log_print("detail", 1)
    coverage_agent.log_print("WARNING: shown", 1)
    metrics.log_print("finishing detail", 2)
    orchestrator.log_print("banner", 1)
    assert printed == [("WARNING: shown", False), ("finishing detail", False), ("banner", True)]
    assert plain == [("banner", 1)]

    class _Logger:
        def log(self, message, **kwargs):
            logged.append((message, kwargs))

    monkeypatch.setattr(orchestrator._log, "logger", _Logger())
    monkeypatch.setattr(coverage_agent._log, "logger", _Logger())
    orchestrator.log_print("summary", echo=True)
    coverage_agent.log_print("line", 2)
    assert logged == [("summary", {"level": 0, "echo": True}), ("line", {"level": 2})]
