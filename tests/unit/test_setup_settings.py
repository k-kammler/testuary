"""Setup deadlines and the run directory: defaults, validation and precedence
between settings files, environment and command line."""
from pathlib import Path
import subprocess

import pytest

from testuary import config as config_module
from testuary import project as project_module
from testuary.settings import RunSettings, resolve_settings


FIELDS = ("setup_timeout_seconds", "setup_probe_timeout_seconds")


@pytest.fixture(autouse=True)
def no_setup_or_provider(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native probe"))
    monkeypatch.setattr(project_module, "project_runtime", lambda *args, **kwargs: pytest.fail("provisioning"))
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)


def test_default_setup_deadlines_are_distinct_from_phase_limits():
    settings = RunSettings()
    assert {key: float(settings.sections["project"][key]) for key in FIELDS} == {
        "setup_timeout_seconds": 120, "setup_probe_timeout_seconds": 10,
    }
    assert "setup_timeout_seconds" not in settings.sections["starter"]
    assert "setup_timeout_seconds" not in settings.sections["mutation"]


@pytest.mark.parametrize("key", FIELDS)
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf"), True, "", "unknown", None])
@pytest.mark.parametrize("entry", ["constructed", "resolved"])
def test_invalid_setup_deadline_refuses_before_work(key, value, entry, tmp_path):
    fields = {"project": {key: value}}
    with pytest.raises(ValueError, match=key):
        if entry == "constructed":
            RunSettings(sections=fields, base_dir=tmp_path)
        else:
            resolve_settings(config_files=[], overrides=fields, models={},
                             base_dir=tmp_path, use_environment=False)
    assert not list(tmp_path.iterdir())


def test_files_environment_and_explicit_deadlines_share_existing_precedence(tmp_path, monkeypatch):
    path = tmp_path / "settings.ini"
    path.write_text("[project]\nsetup_timeout_seconds = 80\nsetup_probe_timeout_seconds = 8\n", encoding="utf-8")
    settings = resolve_settings(config_files=[path], models={}, base_dir=tmp_path,
                                environment={"TESTUARY_PROJECT_SETUP_TIMEOUT_SECONDS": "60"},
                                overrides={"project": {"setup_probe_timeout_seconds": 2.5}})
    monkeypatch.setenv("TESTUARY_PROJECT_SETUP_TIMEOUT_SECONDS", "1")
    with config_module.use_run_settings(settings):
        config = config_module.get_project_config()
        assert float(config["setup_timeout_seconds"]) == 60
        assert float(config["setup_probe_timeout_seconds"]) == 2.5
    exported = settings.export()
    exported["sections"]["project"]["setup_timeout_seconds"] = "changed"
    assert settings.sections["project"]["setup_timeout_seconds"] == "60"


def test_configured_deadlines_and_smaller_total_are_retained(tmp_path):
    # Native admission caps probes by the remaining total and does not extend
    # the deadline when a configured probe is longer.
    settings = RunSettings(base_dir=tmp_path, sections={"project": {
        "name": "subject", "setup_timeout_seconds": "0.5", "setup_probe_timeout_seconds": "2"}})
    with config_module.use_run_settings(settings):
        config = config_module.get_project_config()
        assert float(config["setup_timeout_seconds"]) == 0.5
        assert float(config["setup_probe_timeout_seconds"]) == 2


def test_partial_overlay_defaults_do_not_replace_captured_deadline_inputs(tmp_path):
    path = tmp_path / "settings.ini"
    path.write_text("[project]\nsetup_timeout_seconds = 15\n", encoding="utf-8")
    settings = resolve_settings(config_files=[path], models={}, use_environment=False, base_dir=tmp_path)
    path.write_text("[project]\nsetup_timeout_seconds = 500\n", encoding="utf-8")
    with config_module.use_run_settings(settings):
        assert float(config_module.get_project_config()["setup_timeout_seconds"]) == 15
        assert float(config_module.get_project_config()["setup_probe_timeout_seconds"]) == 10
    assert Path(settings.base_dir) == tmp_path


def _output_dir(tmp_path, *, project=None, config=None):
    """The run directory with ``project`` in the project's settings file and
    ``config`` on the command line."""
    (tmp_path / ".testuary").mkdir(exist_ok=True)
    (tmp_path / ".testuary" / "project.ini").write_text(
        "[project]\n" + "".join(f"{key} = {value}\n" for key, value in (project or {}).items()),
        encoding="utf-8")
    overrides = {"project": {"name": "p", **(config or {})}, "api": {"model": "m", "provider": "openai"}}
    settings = resolve_settings(config_files=[], overrides=overrides, models={},
                                base_dir=tmp_path, use_environment=False)
    with config_module.use_run_settings(settings):
        return project_module.output_dir()


def test_the_project_settings_files_output_dir_is_used(tmp_path):
    """The bundled [project] default does not override a value the project's
    settings file sets, and the command line wins over both."""
    assert _output_dir(tmp_path, project={"output_dir": "own/${project}"}) == tmp_path / "own" / "p"
    assert _output_dir(tmp_path, project={"output_dir": "own/${project}"},
                       config={"output_dir": "chosen"}) == tmp_path / "chosen"
    assert _output_dir(tmp_path) == tmp_path / ".testuary" / "results" / "m"
