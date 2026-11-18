"""An interrupted command ends with a short report and exit status 130, with
no traceback to obscure what was kept and how to continue."""
from __future__ import annotations

import os
import sys

import pytest

from testuary import __main__ as cli
from testuary import exits
from testuary import orchestrator
from testuary import settings as settings_module
from testuary.settings import RunSettings


def _interrupt(*args, **kwargs):
    raise KeyboardInterrupt


def _configured(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    configured = RunSettings(base_dir=tmp_path, environment=dict(os.environ))
    monkeypatch.setattr(settings_module, "resolve_settings", lambda **kwargs: configured)


def test_interrupted_generation_reports_briefly_and_exits_130(tmp_path, monkeypatch, capsys):
    _configured(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "_run_pipeline", _interrupt)
    monkeypatch.setattr(sys, "argv", ["testuary", "generate", "--source-files", "module.py",
                                      "--display", "plain"])
    assert cli.main() == exits.INTERRUPTED == 130
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "--resume" in err


def test_interrupted_check_reports_briefly_and_exits_130(tmp_path, monkeypatch, capsys):
    _configured(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "check", _interrupt)
    monkeypatch.setattr(sys, "argv", ["testuary", "check", "--source-files", "module.py", "--json"])
    assert cli.main() == 130
    err = capsys.readouterr().err
    assert "Traceback" not in err
    assert "No setup was changed." in err


@pytest.mark.parametrize("argv,advice", [
    (["generate", "--source-files", "m.py"], "Continue with the same command and --resume."),
    (["generate", "--source-files", "m.py", "--reuse", "starter"], "same command and --resume."),
    (["generate", "--source-files", "m.py", "--reuse", "coverage"],
     "(--resume does not combine with this --reuse selection)."),
    (["--source-files", "m.py", "--reuse=starter,coverage,mutation,finishing"], "--resume does not combine"),
    (["--source-files", "m.py", "--reuse", "coverage", "--reuse", "starter"], "same command and --resume."),
    (["--demo", "--output-dir", "out"], "new or empty --output-dir. A demonstration cannot be resumed."),
    (["init"], "Run the command again when ready."),
    (["provision"], "Run the command again when ready."),
    (["new-language", "go"], "Run the command again when ready."),
    (["tutorial"], "Start the tutorial again with a new or empty --output-dir."),
    (["check", "--source-files", "m.py"], "No setup was changed."),
    (["setup"], "Run the command again when ready."),
    (["setup", "project", "demo"], "Run the command again when ready."),
    (["setup", "--demo", "--output-dir", "out"], "new or empty --output-dir. A demonstration cannot be resumed."),
])
def test_the_advice_fits_the_interrupted_command(argv, advice):
    text = cli._interrupt_advice(argv)
    assert text.endswith(advice) or advice in text
    if argv[:1] in (["init"], ["provision"], ["new-language"], ["tutorial"], ["setup"]) or "--demo" in argv:
        assert "--resume" not in text


def test_an_interrupted_setup_is_not_told_to_resume(monkeypatch, capsys):
    """setup has no --resume: Ctrl+C at one of its questions says to run it again."""
    from testuary import setup

    monkeypatch.setattr(setup, "run_setup", _interrupt)
    monkeypatch.setattr(sys, "argv", ["testuary", "setup", "model"])
    assert cli.main() == exits.INTERRUPTED
    err = capsys.readouterr().err
    assert "Stopped by you. Run the command again when ready." in err and "--resume" not in err


def test_an_interrupted_demonstration_is_not_told_to_resume(monkeypatch, capsys):
    monkeypatch.setattr(cli, "run_command", _interrupt)
    monkeypatch.setattr(sys, "argv", ["testuary", "--demo", "--output-dir", "out"])
    assert cli.main() == 130
    err = capsys.readouterr().err
    assert "new or empty --output-dir" in err and "--resume" not in err


def test_a_phase_stop_with_repetitions_is_refused_before_work():
    parse = cli._build_parser().parse_args
    with pytest.raises(ValueError, match="one repetition"):
        cli._check_run_options(parse(["--runs", "3", "--stop-after-phase", "coverage"]))
    for arguments in (["--stop-after-phase", "coverage"], ["--run", "2", "--stop-after-phase", "coverage"],
                      ["--runs", "3"]):
        cli._check_run_options(parse(arguments))


def test_in_process_callers_receive_the_interruption(tmp_path, monkeypatch, capsys):
    """The demonstration runs commands in its own process and records an
    interruption as a cancellation, so only the process entry reports it."""
    _configured(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "_run_pipeline", _interrupt)
    monkeypatch.setattr(sys, "argv", ["testuary", "generate", "--source-files", "module.py",
                                      "--display", "plain"])
    with pytest.raises(KeyboardInterrupt):
        cli.run_command()
    assert "Interrupted." not in capsys.readouterr().err


class _Session:
    def __init__(self, failure=None):
        self.failure = failure
        self.closes = 0

    def close(self):
        self.closes += 1
        if self.failure is not None:
            raise self.failure


def test_an_interrupt_during_cleanup_closes_its_session_once_more(tmp_path, monkeypatch, capsys):
    """A second Ctrl+C during the first one's cleanup leaves the session
    as the interrupt's cleanup handle: the entry closes it once more."""
    session = _Session()

    def interrupted(*args, **kwargs):
        stop = KeyboardInterrupt()
        stop.mutation_cleanup_handle = session
        raise stop

    _configured(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "_run_pipeline", interrupted)
    monkeypatch.setattr(sys, "argv", ["testuary", "generate", "--source-files", "module.py",
                                      "--display", "plain"])
    assert cli.main() == exits.INTERRUPTED
    assert session.closes == 1
    assert "cleanup remains" not in capsys.readouterr().err


@pytest.mark.parametrize("failure", [OSError("the clone directory is busy"), KeyboardInterrupt()])
def test_a_failed_retry_keeps_the_interrupted_exit_and_says_what_remains(
    tmp_path, monkeypatch, capsys, failure,
):
    session = _Session(failure)

    def interrupted(*args, **kwargs):
        stop = KeyboardInterrupt()
        stop.mutation_cleanup_handle = session
        raise stop

    _configured(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "_run_pipeline", interrupted)
    monkeypatch.setattr(sys, "argv", ["testuary", "generate", "--source-files", "module.py",
                                      "--display", "plain"])
    assert cli.main() == exits.INTERRUPTED
    err = capsys.readouterr().err
    assert "The mutation tool may still be running after the interruption" in err
    assert type(failure).__name__ in err


def test_a_requested_stop_that_was_not_reached_ends_without_a_traceback(tmp_path, monkeypatch, capsys):
    def stop_not_reached(*args, **kwargs):
        raise orchestrator._PhaseStopNotReached(
            "requested phase boundary was not completed successfully: initial")

    _configured(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "_run_pipeline", stop_not_reached)
    monkeypatch.setattr(sys, "argv", ["testuary", "generate", "--source-files", "module.py",
                                      "--stop-after-phase", "starter", "--display", "plain"])
    assert cli.main() == 1
    out, err = capsys.readouterr()
    for stream in (out, err):
        assert "Traceback" not in stream and "FATAL" not in stream
    assert "Stopped: requested phase boundary was not completed successfully: initial." in err


def test_a_response_that_streamed_nothing_names_streaming():
    from testuary.model_requests import failure_text

    assert "must support streaming" in failure_text(ValueError("No generation chunks were returned"))
    assert failure_text(ValueError("other")) == "other"
