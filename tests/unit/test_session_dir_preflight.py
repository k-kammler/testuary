"""An unusable mutation session directory is refused before any model request.

The Python mutation engine refuses a configured cosmic_ray_local_session_dir
it cannot create or write at its first measurement, which comes after the
starter requests. The readiness check and the run start apply the engine's
own rule and words first."""
from __future__ import annotations

import os

import pytest

import test_resolved_starter_requests as starter_harness
from testuary import orchestrator
from testuary import config
from testuary.languages import setup_requirements
from testuary.readiness import _session_dir_item, session_dir_refusal
from testuary.settings import ConfigurationError, RunSettings

streams = starter_harness.streams
MUTATION = setup_requirements(coverage=True, mutation=True)


def _settings(tmp_path, session_dir, *, exec_mode="local", **sections):
    return RunSettings(base_dir=tmp_path, environment=dict(os.environ),
                       models=starter_harness.configured().models, sections={
        "api": {"model": "explicit"}, "main": {"hints_file": "", "seed": 37, "project_exec": exec_mode},
        "mutation": {"cosmic_ray_local_session_dir": str(session_dir)}, **sections})


def test_the_readiness_item_names_the_engines_refusal(tmp_path):
    blocked = tmp_path / "a-file"
    blocked.write_text("not a directory", encoding="utf-8")
    settings = _settings(tmp_path, blocked / "sessions")
    with config.use_run_settings(settings):
        item = _session_dir_item({"python"}, MUTATION, settings, verify=True)
        refusal = session_dir_refusal({"python"}, create=True)
    assert item.status == "failed" and item.reason == refusal
    assert "(cosmic_ray_local_session_dir, CosmicRaySettings.local_session_dir) cannot be used" in refusal
    assert "The run is refused rather than moved to another directory" in refusal


def test_a_dir_below_a_file_fails_without_verification(tmp_path):
    """A directory that cannot be created reads failed even without --verify,
    in the engine's words, and nothing is created."""
    blocked = tmp_path / "a-file"
    blocked.write_text("not a directory", encoding="utf-8")
    settings = _settings(tmp_path, blocked / "deeper" / "sessions")
    with config.use_run_settings(settings):
        item = _session_dir_item({"python"}, MUTATION, settings, verify=False)
    assert item.status == "failed"
    assert "(cosmic_ray_local_session_dir, CosmicRaySettings.local_session_dir) cannot be used" in item.reason
    assert blocked.is_file()


def test_a_dir_on_a_drive_that_is_not_present_fails_without_verification(tmp_path):
    """No ancestor of the directory exists: it reads failed, not a crash."""
    absent = [letter for letter in "QRSTUVWXYZ" if not os.path.exists(f"{letter}:/")]
    if os.name != "nt" or not absent:
        pytest.skip("needs a Windows drive letter that is not present")
    settings = _settings(tmp_path, f"{absent[0]}:/node-local/sessions")
    with config.use_run_settings(settings):
        item = _session_dir_item({"python"}, MUTATION, settings, verify=False)
    assert item.status == "failed"
    assert "(cosmic_ray_local_session_dir, CosmicRaySettings.local_session_dir) cannot be used" in item.reason


def test_a_missing_dir_is_left_to_verification_which_creates_it(tmp_path):
    session_dir = tmp_path / "shm" / "sessions"
    settings = _settings(tmp_path, session_dir)
    with config.use_run_settings(settings):
        inspected = _session_dir_item({"python"}, MUTATION, settings, verify=False)
        assert not session_dir.exists()
        verified = _session_dir_item({"python"}, MUTATION, settings, verify=True)
    assert (inspected.status, inspected.tried_by) == ("unverified", "verification")
    assert verified.status == "verified" and session_dir.is_dir()


@pytest.mark.parametrize("languages, requirements, exec_mode, expected", [
    ({"java"}, MUTATION, "local", None),
    ({"python"}, setup_requirements(coverage=True, mutation=False), "local", None),
    ({"python"}, MUTATION, "container", "unverified"),
])
def test_the_item_appears_only_where_the_python_engine_uses_the_dir(
    languages, requirements, exec_mode, expected, tmp_path,
):
    settings = _settings(tmp_path, tmp_path / "unused", exec_mode=exec_mode)
    with config.use_run_settings(settings):
        item = _session_dir_item(languages, requirements, settings, verify=True)
    assert (item.status if item is not None else None) == expected
    assert not (tmp_path / "unused").exists()


def test_a_run_refuses_an_unusable_dir_before_any_request(tmp_path, monkeypatch, streams):
    from testuary.__main__ import _build_parser

    project = tmp_path / "projects/subject"
    project.mkdir(parents=True)
    (project / "module.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    blocked = tmp_path / "a-file"
    blocked.write_text("not a directory", encoding="utf-8")
    settings = _settings(tmp_path, blocked / "sessions", project={
        "name": "subject", "source_dir": str(project),
        "output_dir": str(tmp_path / "output")})
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kw: pytest.fail("cleanup preceded refusal"))
    args = _build_parser().parse_args(["--source-files", "module.py"])
    with pytest.raises(ConfigurationError, match=r"cosmic_ray_local_session_dir, CosmicRaySettings\.local_session_dir\) cannot be used.*No phase ran"):
        orchestrator.run_pipeline(args, settings=settings)
    assert streams[0] == []
    assert not (tmp_path / "output").exists()


def test_a_run_refuses_a_local_distributor_with_several_workers_before_any_request(
    tmp_path, monkeypatch, streams,
):
    """Cosmic Ray's local distributor runs one mutant at a time, so a Python
    run that configures it with four workers is refused before any request,
    naming both settings."""
    from testuary.__main__ import _build_parser

    project = tmp_path / "projects/subject"
    project.mkdir(parents=True)
    (project / "module.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    settings = _settings(tmp_path, tmp_path / "sessions", project={
        "name": "subject", "source_dir": str(project),
        "output_dir": str(tmp_path / "output")})
    settings = RunSettings(base_dir=tmp_path, environment=dict(os.environ), models=settings.models,
                           sections={**settings.sections, "mutation": {
                               "cosmic_ray_distributor": "local", "engine_workers": 4}})
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kw: pytest.fail("cleanup preceded refusal"))
    args = _build_parser().parse_args(["--source-files", "module.py"])
    with pytest.raises(ConfigurationError, match=(
            r"cosmic_ray_distributor = local runs one mutant at a time, but "
            r"engine_workers = 4.*No phase ran")):
        orchestrator.run_pipeline(args, settings=settings)
    assert streams[0] == []
