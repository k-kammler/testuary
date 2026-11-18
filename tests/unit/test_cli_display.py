"""Display preferences are separate from generation configuration."""
from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from testuary import __main__ as cli, orchestrator
from testuary.settings import RunSettings


def test_display_defaults_do_not_override_generation_settings():
    args = cli._build_parser().parse_args([])
    assert (args.display, args.no_color, args.demo_language) == ("auto", False, "python")
    assert orchestrator._cli_settings_overrides(args) == {}


@pytest.mark.parametrize("mode", ["auto", "compact", "detailed", "plain"])
def test_display_preferences_are_honored_without_config_overrides(mode):
    args = cli._build_parser().parse_args(["--display", mode, "--no-color"])
    assert (args.display, args.no_color) == (mode, True)
    assert orchestrator._cli_settings_overrides(args) == {}


def test_java_demo_delegates_before_loading_configuration(monkeypatch):
    from testuary.demo import cli as demo_cli

    captured = {}
    monkeypatch.setattr(sys, "argv", ["testuary", "--demo", "--demo-language", "java",
                                     "--display", "detailed", "--no-color"])
    monkeypatch.setattr(demo_cli, "main", lambda args, parser: captured.update(vars(args)) or 0)
    monkeypatch.setattr(orchestrator, "run_pipeline", lambda *a, **k: pytest.fail("normal pipeline"))
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 0
    assert captured["demo_language"] == "java"
    assert captured["display"] == "detailed"
    assert captured["no_color"] is True


def test_demo_language_is_not_a_normal_language_filter(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["testuary", "--demo-language", "java"])
    monkeypatch.setattr(orchestrator, "run_pipeline", lambda *a, **k: pytest.fail("normal pipeline"))
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 2
    assert "--demo-language requires --demo" in capsys.readouterr().err


@pytest.mark.parametrize("language", ["python", "java"])
@pytest.mark.parametrize("exception", [None, RuntimeError("fixture failure"), KeyboardInterrupt()])
def test_demo_delegation_labels_nested_pipeline_and_restores_normal_run(monkeypatch, language, exception):
    from testuary.demo import cli as demo_cli

    events = []

    def execute(args, languages):
        args._progress.emit(target="fixture", phase="finishing", operation="benchmark",
                            state="succeeded", provenance="recorded")
        if exception is not None:
            raise exception
        return 0

    def replay(args, parser):
        # The replay invokes the ordinary pipeline without the outer --demo flag.
        return orchestrator.run_pipeline(SimpleNamespace(), settings=RunSettings(),
                                         event_sink=events.append)

    monkeypatch.setattr(sys, "argv", ["testuary", "--demo", "--demo-language", language])
    monkeypatch.setattr(demo_cli, "main", replay)
    monkeypatch.setattr(orchestrator, "_run_pipeline", execute)
    with pytest.raises(SystemExit if exception is None else type(exception)):
        cli._main()
    assert len(events) == 3
    assert all(event.provenance == "synthetic" for event in events)
    assert events[-1].state == ("succeeded" if exception is None else
                                "cancelled" if isinstance(exception, KeyboardInterrupt) else "failed")

    events.clear()
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda *a: 0)
    assert orchestrator.run_pipeline(SimpleNamespace(), settings=RunSettings(),
                                     event_sink=events.append) == 0
    assert all(event.provenance == "observed" for event in events)
