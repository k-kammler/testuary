"""Scaled generation is not part of Testuary: no phase, flag, setting or
pipeline field names it."""
from __future__ import annotations

import sys

import pytest

from testuary import __main__ as cli
from testuary import orchestrator, phase_state
from testuary import orchestrator as invocation
from testuary import config as config_module
from testuary import target_phases as phases

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def test_the_phases_have_no_scaled_unit():
    assert phase_state.PHASES == ("starter", "coverage", "mutation", "finishing")
    assert orchestrator.available_phase_names() == phase_state.PHASES
    assert "scaled" not in phases.ReuseFlags.__dataclass_fields__
    assert not [name for name in phases.TestPipelineResult.__dataclass_fields__
                if "scaled" in name]


def test_the_cli_registers_no_scaled_option():
    parser = cli._build_parser()
    options = {option for action in parser._actions for option in action.option_strings}
    assert "--repeat-scaled" not in options
    assert "--use-scaled-run" not in options
    assert "scaled" not in parser.format_help()
    assert not [name for name in vars(parser.parse_args([])) if "scaled" in name]


def test_a_typed_request_carries_no_scaled_flag():
    request = invocation.RunRequest(source_files=("module.py",))
    assert not [name for name in vars(invocation._invocation(request)) if "scaled" in name]


@pytest.mark.parametrize("argv", [
    ["--repeat-scaled", "1"], ["--use-scaled-run", "1"], ["--stop-after-phase", "scaled"],
])
def test_the_cli_rejects_scaled_options(argv, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["testuary", *argv])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2


@pytest.mark.parametrize("command", [[], ["--demo"]])
def test_the_cli_rejects_scaled_reuse(command, monkeypatch, capsys):
    """A run and a demonstration refuse an unknown phase before anything
    runs. check refuses it in its own report (test_check_entry)."""
    monkeypatch.setattr(sys, "argv", ["testuary", *command, "--reuse", "scaled"])
    with pytest.raises(SystemExit) as error:
        cli.main()
    assert error.value.code == 2
    assert "--reuse: unknown value scaled. Choose from" in capsys.readouterr().err


def test_no_configuration_names_the_scaled_phase():
    """Scaled settings are absent from package defaults and resolved settings."""
    from configparser import ConfigParser
    from pathlib import Path

    for path in (Path(config_module.__file__).with_name("config.ini"),):
        config = ConfigParser(interpolation=None)
        config.read(path, encoding="utf-8")
        assert "test_scaled" not in config, path
        for section in config.sections():
            assert not [k for k in config[section] if "scaled" in k], (
                path, section)
    record = config_module.resolved_test_preset()["sections"]
    assert not [name for name in record if "scaled" in name]
    assert not [key for values in record.values() for key in values if "scaled" in key]
