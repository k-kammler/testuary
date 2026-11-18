"""Actual invocations preserve original modes and repair only private source."""
import atexit
from dataclasses import replace
from pathlib import Path
import stat
import subprocess

import pytest

import test_generation_reuse_interruptions as harness
from testuary import orchestrator
from testuary import config
from testuary import target_phases as phases


def snapshot(root):
    return {str(path.relative_to(root)): (
        path.read_bytes() if path.is_file() else None, path.stat().st_mode,
    ) for path in [root, *root.rglob("*")]}


@pytest.mark.parametrize("readonly", [True, False])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_sequential_public_runs_never_chmod_originals_or_register_unlock(
        readonly, outcome, tmp_path, monkeypatch):
    project, target, _, settings = harness.setup_case(tmp_path)
    resource = project / "resource.txt"
    resource.write_bytes(b"declared project resource\r\n")
    chmod = Path.chmod
    modes = {path: path.stat().st_mode for path in [project, *project.rglob("*")]}
    if readonly:
        for path, mode in modes.items():
            chmod(path, mode & ~(stat.S_IWUSR | stat.S_IWGRP | stat.S_IWOTH))
    before = snapshot(project)
    registrations, permission_attempts, calls = [], [], []
    failure = RuntimeError("controlled phase failure")
    cancel = KeyboardInterrupt()

    def private_chmod(path, mode, *args, **kwargs):
        if path.absolute().is_relative_to(project):
            permission_attempts.append((path, mode))
            raise AssertionError("original permission mutation")
        return chmod(path, mode, *args, **kwargs)

    def run_phase(ctx):
        assert snapshot(project) == before
        assert not permission_attempts
        work_target = Path(ctx.top_file["file_path"])
        assert work_target != target and work_target.is_relative_to(ctx.work_tests.parent)
        assert Path(ctx.top_file["canonical_path"]) == target
        assert work_target.read_bytes() == target.read_bytes()
        assert work_target.stat().st_mode & stat.S_IWUSR
        assert work_target.parent.stat().st_mode & stat.S_IWUSR
        copied_resource = work_target.parent / resource.name
        assert copied_resource.read_bytes() == resource.read_bytes()
        assert copied_resource.stat().st_mode & stat.S_IWUSR
        with phases._source_integrity_guard(ctx, "coverage"):
            work_target.write_bytes(b"temporary work mutation")
        assert work_target.read_bytes() == target.read_bytes()
        assert snapshot(project) == before
        calls.append(config.active_settings())
        if outcome == "failure":
            raise failure
        if outcome == "cancelled":
            raise cancel
        return phases.TestPipelineResult(stopped_after_phase="coverage")

    monkeypatch.setattr(Path, "chmod", private_chmod)
    monkeypatch.setattr(atexit, "register", lambda *args, **kwargs: registrations.append((args, kwargs)))
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native process"))
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_check_canonical_against_git", lambda *a: [])
    monkeypatch.setattr(phases, "run_target_phases", run_phase)
    try:
        for index in (1, 2):
            selected = replace(settings, sections={**settings.sections,
                "project": {**settings.sections["project"], "output_dir": str(tmp_path / f"output_{index}")},
            })
            if outcome == "cancelled":
                with pytest.raises(KeyboardInterrupt) as caught:
                    orchestrator.run_pipeline(harness.public_args("--stop-after-phase", "coverage"), settings=selected)
                assert caught.value is cancel
            else:
                result = orchestrator.run_pipeline(harness.public_args("--stop-after-phase", "coverage"), settings=selected)
                assert result == (1 if outcome == "failure" else 0)
            assert snapshot(project) == before
            assert config.active_settings() is None and orchestrator.get_main_logger() is None
        assert len(calls) == 2 and not permission_attempts
        assert calls[0] is not calls[1]
        assert not registrations
    finally:
        # Fixture cleanup restores only the modes set by this test.
        for path, mode in modes.items():
            chmod(path, mode)
