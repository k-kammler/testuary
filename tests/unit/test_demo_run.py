"""Replay sources are scoped even when setup or the pipeline fails."""
from __future__ import annotations

import sys

import pytest

from testuary import __main__ as cli
from testuary import reduction as cover_source
from testuary import models as model_source
from testuary import languages
from testuary.demo import cli as demo_cli
from testuary.demo.scenario import Oracle
from testuary.languages import LanguageBackend


def _stand_in(oracle):
    # The registry keeps file_extensions on its adapter. A stand-in must accept it.
    return LanguageBackend()


@pytest.fixture
def empty_sources():
    assert model_source.current() is None
    assert cover_source.current() is None
    yield
    assert model_source.current() is None
    assert cover_source.current() is None


def test_backend_constructor_failure_installs_nothing(empty_sources):
    def broken(oracle):
        raise RuntimeError("backend fixture constructor failed")

    with pytest.raises(RuntimeError, match="constructor failed"):
        demo_cli.play_oracle(Oracle(), ["testuary"], backend_class=broken)


def test_existing_cover_source_is_preserved_when_setup_fails(empty_sources):
    existing = object()
    with cover_source.installed(existing):
        with pytest.raises(RuntimeError, match="already installed"):
            demo_cli.play_oracle(Oracle(), ["testuary"], backend_class=_stand_in)
        assert cover_source.current() is existing
        assert model_source.current() is None


def test_override_failure_removes_just_its_sources(monkeypatch, empty_sources):
    def broken(*args):
        raise RuntimeError("override fixture failed")

    monkeypatch.setattr(languages, "override_backend", broken)
    with pytest.raises(RuntimeError, match="override fixture failed"):
        demo_cli.play_oracle(Oracle(), ["testuary"], backend_class=_stand_in)


@pytest.mark.parametrize("failure", [RuntimeError("run fixture failed"), KeyboardInterrupt()])
def test_pipeline_exception_restores_backend_argv_and_sources(
        monkeypatch, empty_sources, failure):
    original_backend = languages.get_backend("python")
    original_argv = list(sys.argv)

    def broken():
        assert sys.argv == ["testuary", "fixture"]
        assert model_source.current() is not None
        assert cover_source.current() is not None
        raise failure

    monkeypatch.setattr(cli, "run_command", broken)
    with pytest.raises(type(failure)):
        demo_cli.play_oracle(Oracle(), ["testuary", "fixture"], backend_class=_stand_in)
    assert languages.get_backend("python") is original_backend
    assert sys.argv == original_argv


def test_success_closes_oracle_then_restores_sources(monkeypatch, empty_sources):
    finished = []
    oracle = Oracle()
    monkeypatch.setattr(oracle, "finish", lambda: finished.append(True))
    monkeypatch.setattr(cli, "run_command", lambda: 0)
    assert demo_cli.play_oracle(oracle, ["testuary"], backend_class=_stand_in) is None
    assert finished == [True]


def test_a_refused_recorded_configuration_stops_the_playback_as_stale(
        monkeypatch, empty_sources):
    """Settings, aliases or options of the recording that this version
    refuses mean the data no longer fit it: a stop, not a usage error."""
    from testuary.demo.scenario import ScenarioStop
    from testuary.settings import ConfigurationError

    def refused():
        try:
            raise ConfigurationError("[coverage] old_setting is not a setting")
        except ConfigurationError as refusal:
            raise SystemExit(2) from refusal

    monkeypatch.setattr(cli, "run_command", refused)
    stop = demo_cli.play_oracle(Oracle(), ["testuary", "fixture"], backend_class=_stand_in)
    assert isinstance(stop, ScenarioStop)
    assert stop.call.kind == "configuration"
    assert "old_setting is not a setting" in stop.reason


def test_another_exit_of_the_run_is_not_taken_for_stale_data(monkeypatch, empty_sources):
    def exits():
        raise SystemExit(2)

    monkeypatch.setattr(cli, "run_command", exits)
    with pytest.raises(SystemExit):
        demo_cli.play_oracle(Oracle(), ["testuary"], backend_class=_stand_in)
