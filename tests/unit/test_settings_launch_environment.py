"""Which environment native commands are launched with.

An explicit ``environment`` is authoritative. ``resolve_settings`` always
captures a launch environment. ``use_environment`` decides only whether
``TESTUARY_`` variables are inputs. Direct construction without an environment
captures nothing and says so, so native execution can refuse instead of
launching tools in an empty environment.
"""
from __future__ import annotations

import pytest

from testuary.settings import RunSettings, resolve_settings


@pytest.fixture
def process_environment(monkeypatch):
    # The suite-wide timing override is itself a TESTUARY_ input, so it is unset.
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    monkeypatch.setenv("SUBJECT_SETTING", "subject")
    monkeypatch.setenv("TESTUARY_COVERAGE_MAX_REQUESTS_PER_PHASE", "7")
    monkeypatch.setenv("TESTUARY_OPENAI_API_KEY", "synthetic-key")


def _resolve(tmp_path, **kwargs):
    return resolve_settings(base_dir=tmp_path, config_files=[], models={}, **kwargs)


def test_direct_construction_without_an_environment_captures_nothing(process_environment):
    settings = RunSettings()
    assert settings.environment_captured is False
    assert dict(settings.environment) == {}


def test_replacing_fields_keeps_the_capture_state(tmp_path):
    from dataclasses import replace

    assert replace(RunSettings(), base_dir=tmp_path).environment_captured is False
    captured = replace(RunSettings(environment={}), base_dir=tmp_path)
    assert captured.environment_captured is True and dict(captured.environment) == {}


def test_direct_construction_keeps_an_explicit_environment_even_when_empty():
    assert RunSettings(environment={}).environment_captured is True
    settings = RunSettings(environment={"PATH": "chosen"})
    assert settings.environment_captured is True
    assert dict(settings.environment) == {"PATH": "chosen"}


def test_enabled_environment_captures_the_process_and_applies_overrides(tmp_path, process_environment):
    settings = _resolve(tmp_path)
    assert settings.environment_captured is True
    assert settings.environment["SUBJECT_SETTING"] == "subject"
    assert settings.environment["TESTUARY_OPENAI_API_KEY"] == "synthetic-key"
    assert settings.sections["coverage"]["max_requests_per_phase"] == "7"


def test_disabled_environment_still_captures_a_launch_environment(tmp_path, process_environment):
    settings = _resolve(tmp_path, use_environment=False)
    assert settings.environment_captured is True
    assert settings.environment["SUBJECT_SETTING"] == "subject"
    assert not any(name.startswith("TESTUARY_") for name in settings.environment)
    assert settings.sections["coverage"]["max_requests_per_phase"] != "7"


def test_disabled_environment_keeps_an_explicit_mapping_without_testuary_inputs(tmp_path):
    supplied = {"PATH": "chosen", "SYSTEMROOT": "C:/Windows",
                "TESTUARY_COVERAGE_MAX_REQUESTS_PER_PHASE": "7", "TESTUARY_OPENAI_API_KEY": "synthetic-key",
                "LANGADAPTER_HELPER_PYTHON": "/control/python"}
    settings = _resolve(tmp_path, use_environment=False, environment=supplied)
    assert dict(settings.environment) == {"PATH": "chosen", "SYSTEMROOT": "C:/Windows"}
    assert settings.sections["coverage"]["max_requests_per_phase"] != "7"


def test_enabled_environment_uses_an_explicit_mapping_as_given(tmp_path, process_environment):
    supplied = {"PATH": "chosen", "TESTUARY_COVERAGE_MAX_REQUESTS_PER_PHASE": "9"}
    settings = _resolve(tmp_path, environment=supplied)
    assert dict(settings.environment) == supplied
    assert settings.sections["coverage"]["max_requests_per_phase"] == "9"
