"""Neutral deployment inputs preserve execution and recovery controls."""
from __future__ import annotations

import pytest

from testuary import project as _benchmark_lock
from testuary import project as execution
from testuary import config
from testuary import project

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.mark.parametrize("mode", ["auto", "always", "never"])
def test_namespace_precedence_and_stable_lookup(monkeypatch, mode, in_run):
    monkeypatch.setattr(config, "load_config", lambda name: {"work_dir_job_namespace": mode})
    monkeypatch.setenv("TESTUARY_EXECUTION_ID", "execution/one")
    in_run()
    expected = "" if mode == "never" else "job_execution_one"
    assert project.resolve_work_dir_namespace() == expected
    assert project.resolve_work_dir_namespace() == expected
    monkeypatch.setenv("TESTUARY_MAIN_WORK_DIR_NAMESPACE", "override.1")
    in_run()
    assert project.resolve_work_dir_namespace() == ("" if mode == "never" else "job_explicit_override.1")
    monkeypatch.setenv("TESTUARY_MAIN_WORK_DIR_NAMESPACE", "../invalid")
    in_run()
    if mode == "never":
        assert project.resolve_work_dir_namespace() == ""
    else:
        with pytest.raises(ValueError):
            project.resolve_work_dir_namespace()


def test_namespace_cleanup_preserves_other_execution(monkeypatch, tmp_path, in_run):
    from testuary.orchestrator import _cleanup_outdir
    monkeypatch.setattr(config, "load_config", lambda name: {"work_dir_job_namespace": "auto"})
    monkeypatch.setenv("TESTUARY_EXECUTION_ID", "one")
    in_run()
    for name in ("_work/job_one/session.db", "_work/job_two/session.db", "results/suite.py", "logs/run.json"):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
    _cleanup_outdir(outdir=tmp_path, keep_tests=True)
    assert not (tmp_path / "_work/job_one").exists()
    for name in ("_work/job_two/session.db", "results/suite.py", "logs/run.json"):
        assert (tmp_path / name).read_bytes() == name.encode()


@pytest.mark.parametrize("configured", ["true", "false"])
def test_serialization_resolution(monkeypatch, configured):
    calls = []
    def load(name):
        calls.append(name)
        return {"serialize": configured}
    monkeypatch.setattr(config, "load_config", load)
    assert _benchmark_lock._serialize_enabled() is (configured == "true")
    assert calls == ["timing"]


def test_serialization_is_not_cached(in_run):
    in_run(overrides={"timing": {"serialize": "false"}})
    assert _benchmark_lock._serialize_enabled() is False
    in_run(overrides={"timing": {"serialize": "true"}})
    assert _benchmark_lock._serialize_enabled() is True


def test_operational_inputs_do_not_change_resolved_method_settings(monkeypatch, in_run):
    in_run()
    before = config.resolved_test_preset()
    monkeypatch.setenv("TESTUARY_EXECUTION_ID", "execution-1")
    monkeypatch.setenv("TESTUARY_TIMING_SERIALIZE", "false")
    in_run()
    after = config.resolved_test_preset()
    # The variable sets [timing] serialize, which the record shows as set.
    assert after["sections"]["timing"].pop("serialize") == "false"
    before["sections"]["timing"].pop("serialize")
    assert after == before


@pytest.mark.parametrize("runner", ["docker", "podman", "apptainer", "singularity"])
def test_neutral_controls_cross_container_boundary(runner):
    values = {"TESTUARY_EXECUTION_ID": "execution-1", "TESTUARY_TIMING_SERIALIZE": "false",
              "PYTHONPATH": "/project", "PYTHONHASHSEED": "0", "UNRELATED_KEY": "private"}
    tokens = execution.ContainerExecutor("fixture", runner=runner)._forward_env(values)
    flag = "--env" if runner in ("apptainer", "singularity") else "-e"
    assert tokens == [part for key, value in values.items() if key != "UNRELATED_KEY"
                      for part in (flag, key + "=" + value)]


# Portable package execution inputs.
def test_explicit_namespace_preserves_modes(monkeypatch, in_run):
    monkeypatch.setenv("TESTUARY_MAIN_WORK_DIR_NAMESPACE", "allocation-42")
    in_run()
    monkeypatch.setattr(config, "load_config", lambda name: {"work_dir_job_namespace": "auto"})
    assert project.resolve_work_dir_namespace() == "job_explicit_allocation-42"
    monkeypatch.setenv("TESTUARY_MAIN_WORK_DIR_NAMESPACE", "../escape")
    in_run()
    with pytest.raises(ValueError):
        project.resolve_work_dir_namespace()
    monkeypatch.setattr(config, "load_config", lambda name: {"work_dir_job_namespace": "never"})
    assert project.resolve_work_dir_namespace() == ""
