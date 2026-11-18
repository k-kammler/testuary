"""``testuary.phase_state``, the resumable-run checkpoint.

A phase is trusted on resume only when its status is complete, the target
path and the canonical-source hash match, every artifact exists with a
matching hash, and the recorded execution context is complete and unchanged.
Each assertion of ``False`` marks partial or stale work that must not be
reused.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from testuary import phase_state as ps
from testuary import config
from testuary.evidence import finishing_settings
from testuary.settings import RunSettings


def _write(p: Path, text: str) -> Path:
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(text, encoding="utf-8")
    return p


def test_sha256_file_stable_and_content_sensitive(tmp_path):
    a = _write(tmp_path / "a.txt", "hello")
    b = _write(tmp_path / "b.txt", "hello")
    c = _write(tmp_path / "c.txt", "world")
    assert ps.sha256_file(a) == ps.sha256_file(b)
    assert ps.sha256_file(a) != ps.sha256_file(c)


def test_sha256_file_missing_returns_none(tmp_path):
    """None, not a raise: a missing artifact must fail validation
    closed, never compare equal to a recorded hash."""
    assert ps.sha256_file(tmp_path / "nope.txt") is None


def test_load_missing_returns_empty(tmp_path):
    assert ps.load(tmp_path) == {}


def test_save_then_load_roundtrips(tmp_path):
    ps.save(tmp_path, {"target_number": 1, "coverage": {"status": "complete"}})
    assert ps.load(tmp_path)["coverage"]["status"] == "complete"


def test_load_corrupt_json_returns_empty(tmp_path):
    """A truncated or garbage marker reads as nothing complete, so the work
    is redone, and never raises. Atomic writes make such a marker unlikely."""
    ps.state_path(tmp_path).write_text("{ not json", encoding="utf-8")
    assert ps.load(tmp_path) == {}


def test_save_leaves_no_tempfile(tmp_path):
    ps.save(tmp_path, {"target_number": 2})
    leftovers = [p.name for p in tmp_path.iterdir()
                 if p.name.startswith(".phase_state.")]
    assert leftovers == []


def test_save_overwrites_atomically_preserving_prior_on_noop(tmp_path):
    ps.save(tmp_path, {"target_number": 1, "coverage": {"status": "complete"}})
    ps.save(tmp_path, {"target_number": 1, "coverage": {"status": "in_progress"}})
    assert ps.load(tmp_path)["coverage"]["status"] == "in_progress"


def _src(tmp_path) -> Path:
    return _write(tmp_path / "proj" / "foo.py", "def f():\n    return 1\n")


def test_started_then_not_validated(tmp_path, default_run):
    """in_progress is not complete, so validate is False and the phase
    is redone after a crash."""
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    ps.mark_started(
        target_dir, "coverage", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src,
    )
    state = ps.load(target_dir)
    # The per-run shape: state[phase][str(run_idx)] holds the
    # status block. Default run_idx=0 lives under "0".
    assert state["coverage"]["0"]["status"] == ps.IN_PROGRESS
    assert isinstance(state["coverage"]["0"]["started_t_mono_s"], float)
    assert state["coverage"]["0"]["started_clock_domain_id"]
    assert ps.validate(
        state, "coverage", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
    ) is False


def test_complete_with_matching_artifact_validates(tmp_path, default_run):
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    artifact = _write(target_dir / "tests" / "test_foo.py", "def test_x(): pass\n")
    ps.mark_started(
        target_dir, "coverage", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src,
    )
    ps.mark_complete(
        target_dir, "coverage",
        artifacts={"tests/test_foo.py": artifact},
        metrics={"coverage": 91.0},
    )
    state = ps.load(target_dir)
    assert isinstance(state["coverage"]["0"]["completed_t_mono_s"], float)
    assert state["coverage"]["0"]["completed_clock_domain_id"]
    assert ps.validate(
        state, "coverage", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
    ) is True


def _context(name):
    """A complete schema 1 resume context, told apart by ``name``."""
    return {"schema_version": 1, "project_exec": "local", "context": name}


def test_changed_resume_context_invalidates_complete_phase(
    tmp_path, monkeypatch,
):
    """A checkpoint from a different framework, settings or container
    context is redone even when its source and artifact bytes still match."""
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    artifact = _write(
        target_dir / "tests" / "test_foo.py", "def test_x(): pass\n"
    )
    monkeypatch.setattr(
        ps, "_current_resume_context", lambda _target=None: _context("A")
    )
    ps.mark_started(
        target_dir, "coverage", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src,
    )
    ps.mark_complete(
        target_dir, "coverage",
        artifacts={"tests/test_foo.py": artifact},
    )
    state = ps.load(target_dir)
    assert state["coverage"]["0"]["resume_context"] == _context("A")

    monkeypatch.setattr(
        ps, "_current_resume_context", lambda _target=None: _context("B")
    )
    assert ps.validate(
        state, "coverage", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
    ) is False


def test_incomplete_current_schema_resume_context_fails_closed(
    tmp_path, monkeypatch,
):
    """A new marker with an unavailable framework or settings identity is never
    reusable merely because the same None value is observed twice."""
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    artifact = _write(
        target_dir / "tests" / "test_foo.py", "def test_x(): pass\n"
    )
    # A container run whose testbed identity was never recorded: the run
    # cannot say which image produced the artifact, so the marker is not
    # trusted.
    incomplete = {
        "schema_version": 1,
        "project_exec": "container",
        "environment": None,
    }
    monkeypatch.setattr(
        ps, "_current_resume_context", lambda _target=None: incomplete
    )
    ps.mark_started(
        target_dir, "coverage", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src,
    )
    ps.mark_complete(
        target_dir, "coverage",
        artifacts={"tests/test_foo.py": artifact},
    )

    assert ps.validate(
        ps.load(target_dir), "coverage", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
    ) is False


def test_a_completed_unit_without_resume_context_is_refused(tmp_path, default_run):
    """Every unit is started with its execution context, so a completed
    unit without one does not validate: its recorded context is
    incomplete."""
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    artifact = _write(
        target_dir / "tests" / "test_foo.py", "def test_x(): pass\n"
    )
    ps.mark_complete(
        target_dir, "coverage",
        artifacts={"tests/test_foo.py": artifact},
    )
    state = ps.load(target_dir)
    state["coverage"]["0"]["target_file_path"] = "proj/foo.py"
    state["coverage"]["0"]["target_source_sha256"] = ps.sha256_file(src)
    ps.save(target_dir, state)

    assert ps.validate(
        ps.load(target_dir), "coverage", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
    ) is False
    assert ps.rejection_reason(
        ps.load(target_dir), "coverage", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
    ) == "the recorded execution context is incomplete"


def _container_settings(*, container="job-17", python="/opt/venv/bin/python"):
    """Resolved settings of a container run, as the cluster guide sets them."""
    from testuary.settings import resolve_settings
    return resolve_settings(models={}, config_files=[], environment={
        "TESTUARY_MAIN_PROJECT_EXEC": "container", "TESTUARY_PROJECT_CONTAINER": container,
        "TESTUARY_PROJECT_CONTAINER_RUNNER": "podman", "TESTUARY_PROJECT_PYTHON_EXECUTABLE": python,
    })


def test_a_container_run_resumes_under_its_documented_settings(tmp_path):
    """A container run records the container command and the project's
    interpreter, which the documented settings name, so it resumes under
    the same settings without any file of a harness. A new container of
    the same setup (another name) is the same context. Another
    interpreter is not, and the reason says so."""
    from testuary import config
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    artifact = _write(target_dir / "tests" / "test_foo.py", "def test_x(): pass\n")
    with config.use_run_settings(_container_settings()):
        ps.mark_started(
            target_dir, "coverage", target_number=1, stem="foo",
            target_file_path="proj/foo.py", canonical_source_path=src,
        )
        ps.mark_complete(target_dir, "coverage", artifacts={"tests/test_foo.py": artifact})
    with config.use_run_settings(_container_settings(container="job-18")):
        assert ps.validate(
            ps.load(target_dir), "coverage", target_dir=target_dir,
            target_file_path="proj/foo.py", canonical_source_path=src,
        ) is True
    with config.use_run_settings(_container_settings(python="/opt/other/bin/python")):
        assert ps.validate(
            ps.load(target_dir), "coverage", target_dir=target_dir,
            target_file_path="proj/foo.py", canonical_source_path=src,
        ) is False
        assert "runtime" in ps.rejection_reason(
            ps.load(target_dir), "coverage", target_dir=target_dir,
            target_file_path="proj/foo.py", canonical_source_path=src,
        )


def test_container_mode_source_hash_records_and_validates(
        tmp_path, monkeypatch):
    """In container mode the canonical source lives inside the container:
    the direct host read yields None, and the hash must route through
    the executor-aware reader so the checkpoint records a real hash and
    resume validation can pass. A run whose source hash stayed None
    would redo every completed phase on resume."""
    import hashlib
    from testuary import config
    from testuary import project
    content = b"def target():\n    return 1\n"
    monkeypatch.setattr(
        project, "sha256_project_file",
        lambda path: hashlib.sha256(content).hexdigest())
    src = "/container/only/xarray/core/groupby.py"
    target_dir = tmp_path / "results" / "rank1_foo"
    artifact = _write(target_dir / "tests" / "test_foo.py",
                      "def test_x(): pass\n")
    with config.use_run_settings(_container_settings()):
        ps.mark_started(
            target_dir, "coverage", target_number=1, stem="foo",
            target_file_path="xarray/core/groupby.py", canonical_source_path=src,
        )
        state = ps.load(target_dir)
        assert state["coverage"]["0"]["target_source_sha256"] == hashlib.sha256(
            content).hexdigest()
        ps.mark_complete(
            target_dir, "coverage",
            artifacts={"tests/test_foo.py": artifact},
            metrics={"coverage": 91.0},
        )
        state = ps.load(target_dir)
        assert ps.validate(
            state, "coverage", target_dir=target_dir,
            target_file_path="xarray/core/groupby.py", canonical_source_path=src,
        ) is True


def test_container_mode_unreadable_source_still_fails_closed(
        tmp_path, monkeypatch, default_run):
    """When the executor cannot read the source either (a dead container),
    the hash stays None and validation refuses."""
    from testuary import config
    from testuary import project
    monkeypatch.setattr(config, "project_exec_mode", lambda: "container")
    monkeypatch.setattr(project, "sha256_project_file", lambda path: None)
    src = "/container/only/mod.py"
    target_dir = tmp_path / "results" / "rank1_foo"
    artifact = _write(target_dir / "tests" / "test_foo.py",
                      "def test_x(): pass\n")
    ps.mark_started(
        target_dir, "coverage", target_number=1, stem="foo",
        target_file_path="mod.py", canonical_source_path=src,
    )
    ps.mark_complete(
        target_dir, "coverage",
        artifacts={"tests/test_foo.py": artifact},
        metrics={},
    )
    state = ps.load(target_dir)
    assert state["coverage"]["0"]["target_source_sha256"] is None
    assert ps.rejection_reason(
        state, "coverage", target_dir=target_dir, target_file_path="mod.py",
        canonical_source_path=src) == (
        "the canonical source's hash is not recorded: the source could not be read "
        "when the phase started, or an earlier version recorded the phase")
    assert ps.validate(
        state, "coverage", target_dir=target_dir,
        target_file_path="mod.py", canonical_source_path=src,
    ) is False


def test_tampered_artifact_fails_validation(tmp_path, default_run):
    """An artifact changed after the marker was written fails the hash
    check, so altered work is never reused as success."""
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    artifact = _write(target_dir / "tests" / "test_foo.py", "def test_x(): pass\n")
    ps.mark_started(
        target_dir, "coverage", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src,
    )
    ps.mark_complete(
        target_dir, "coverage", artifacts={"tests/test_foo.py": artifact},
    )
    artifact.write_text("def test_x(): assert False\n", encoding="utf-8")
    state = ps.load(target_dir)
    assert ps.validate(
        state, "coverage", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
    ) is False


def test_missing_artifact_fails_validation(tmp_path, default_run):
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    artifact = _write(target_dir / "tests" / "test_foo.py", "x\n")
    ps.mark_complete(
        target_dir, "coverage", artifacts={"tests/test_foo.py": artifact},
    )
    # Pin the source hash so only the missing artifact can fail it.
    st = ps.load(target_dir)
    st["coverage"]["0"]["target_file_path"] = "proj/foo.py"
    st["coverage"]["0"]["target_source_sha256"] = ps.sha256_file(src)
    ps.save(target_dir, st)
    artifact.unlink()
    assert ps.validate(
        ps.load(target_dir), "coverage", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
    ) is False


def test_a_runs_own_source_hash_decides_its_reuse(tmp_path, default_run):
    """r0 and r1 are made from one source. After the source changed, a
    redone r0 records the new source, and r1, made from the old one, stays
    not reusable."""
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    artifact = _write(target_dir / "tests" / "test_foo.py", "def test_x(): pass\n")
    started = {"target_number": 1, "stem": "foo", "target_file_path": "proj/foo.py",
               "canonical_source_path": src}
    for run in (0, 1):
        ps.mark_started(target_dir, "coverage", run_idx=run, **started)
        ps.mark_complete(target_dir, "coverage", artifacts={"tests/test_foo.py": artifact}, run_idx=run)
    src.write_text("def f():\n    return 999\n", encoding="utf-8")
    ps.mark_started(target_dir, "coverage", run_idx=0, **started)
    where = {"target_dir": target_dir, "target_file_path": "proj/foo.py",
             "canonical_source_path": src, "run_idx": 1}
    state = ps.load(target_dir)
    assert ps.validate(state, "coverage", **where) is False
    assert ps.rejection_reason(state, "coverage", **where) == (
        "the canonical source changed since the phase ran")


def test_changed_source_invalidates_complete_phase(tmp_path, default_run):
    """Tests generated against an older source are stale even when green.
    Editing the original projects/<file> invalidates the completed
    phase."""
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    artifact = _write(target_dir / "tests" / "test_foo.py", "def test_x(): pass\n")
    ps.mark_started(
        target_dir, "coverage", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src,
    )
    ps.mark_complete(
        target_dir, "coverage", artifacts={"tests/test_foo.py": artifact},
    )
    src.write_text("def f():\n    return 999\n", encoding="utf-8")
    assert ps.validate(
        ps.load(target_dir), "coverage", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
    ) is False


def test_target_path_mismatch_invalidates(tmp_path, default_run):
    """When a re-profile moves target N to a different file, the marker is
    not trusted, since it is keyed by path."""
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    artifact = _write(target_dir / "tests" / "test_foo.py", "x\n")
    ps.mark_started(
        target_dir, "coverage", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src,
    )
    ps.mark_complete(
        target_dir, "coverage", artifacts={"tests/test_foo.py": artifact},
    )
    assert ps.validate(
        ps.load(target_dir), "coverage", target_dir=target_dir,
        target_file_path="proj/OTHER.py", canonical_source_path=src,
    ) is False


def test_finishing_phase_validates_without_artifact(tmp_path, default_run):
    """Finishing produces no file. Success is recorded by its metrics. It
    validates with an empty artifact set as long as path and
    source still match."""
    from testuary.evidence import finishing_settings

    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    ps.mark_started(
        target_dir, "finishing", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src,
    )
    ps.mark_complete(target_dir, "finishing", metrics={
        "coverage": 91.0, "mutation": 78.0, "finishing_settings": finishing_settings()})
    assert ps.validate(
        ps.load(target_dir), "finishing", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
    ) is True


@pytest.mark.parametrize("key, value", [
    ("measure_lint", "true"), ("coverage_timeout_seconds", "900"),
    ("suite_run_timeout_seconds", "300"),
])
def test_finishing_measured_under_another_finishing_setting_is_measured_again(
        tmp_path, in_run, key, value):
    """A resume after measure_lint, coverage_timeout_seconds or
    suite_run_timeout_seconds changed measures Finishing again, since its
    numbers depend on each."""
    in_run()
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    ps.mark_started(
        target_dir, "finishing", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src,
    )
    ps.mark_complete(target_dir, "finishing", metrics={
        "coverage": 91.0, "mutation": 78.0, "finishing_settings": finishing_settings()})
    in_run(overrides={"finishing": {key: value}})
    assert ps.validate(
        ps.load(target_dir), "finishing", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
    ) is False


def test_repoint_artifact_revalidates_against_preserved_original(tmp_path, default_run):
    """The mutation phase's consolidation overwrites the coverage file, which
    would make the coverage marker stale. Re-pointing it at the preserved
    archive/mutation/pre_merge original re-validates it, so a consolidated
    target still resumes."""
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    cov = _write(target_dir / "tests" / "test_foo.py", "cov\n")
    ps.mark_started(
        target_dir, "starter", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src,
    )
    ps.mark_complete(target_dir, "starter", metrics={"ok": True})
    ps.mark_started(
        target_dir, "coverage", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src,
    )
    ps.mark_complete(target_dir, "coverage", artifacts={"tests/test_foo.py": cov})
    # The consolidation overwrites the coverage file, so the marker is stale.
    Path(cov).write_text("merged\n", encoding="utf-8")
    assert ps.validate(
        ps.load(target_dir), "coverage", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
    ) is False
    # Preserve the original and re-point the coverage marker at it.
    pre = _write(
        target_dir / "archive" / "mutation" / "pre_merge" / "test_foo.py",
        "cov\n")
    ps.repoint_artifact(
        target_dir, "coverage",
        old_rel="tests/test_foo.py",
        new_rel="archive/mutation/pre_merge/test_foo.py",
        new_abspath=pre,
    )
    # coverage validates again.
    assert ps.validate(
        ps.load(target_dir), "coverage", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
    ) is True


def test_unknown_phase_rejected(tmp_path):
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    import pytest
    with pytest.raises(ValueError):
        ps.mark_started(
            target_dir, "9.9", target_number=1, stem="foo",
            target_file_path="proj/foo.py", canonical_source_path=src,
        )


def test_per_run_marker_blocks_are_independent(tmp_path, default_run):
    """``mark_complete(..., run_idx=1)`` does not overwrite the run_idx=0
    block. Each run keeps its own status and artifacts, so a partial sweep
    resumes from the first incomplete run."""
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    a1 = _write(target_dir / "tests" / "test_foo.py", "run1\n")
    a2 = _write(target_dir / "tests_run_2" / "test_foo.py", "run2\n")
    ps.mark_started(
        target_dir, "coverage", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src,
        run_idx=0,
    )
    ps.mark_complete(
        target_dir, "coverage", artifacts={"tests/test_foo.py": a1},
        run_idx=0,
    )
    ps.mark_started(
        target_dir, "coverage", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src,
        run_idx=1,
    )
    ps.mark_complete(
        target_dir, "coverage",
        artifacts={"tests_run_2/test_foo.py": a2},
        run_idx=1,
    )
    state = ps.load(target_dir)
    # Both run blocks present with their own artifacts.
    assert state["coverage"]["0"]["status"] == ps.COMPLETE
    assert state["coverage"]["1"]["status"] == ps.COMPLETE
    assert "tests/test_foo.py" in state["coverage"]["0"]["artifacts"]
    assert "tests_run_2/test_foo.py" in state["coverage"]["1"]["artifacts"]
    # Each run validates against its own artifact.
    assert ps.validate(
        state, "coverage", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
        run_idx=0,
    ) is True
    assert ps.validate(
        state, "coverage", target_dir=target_dir,
        target_file_path="proj/foo.py", canonical_source_path=src,
        run_idx=1,
    ) is True


def test_phase_metrics_reads_per_run_block(tmp_path, default_run):
    """``phase_metrics`` returns the ``metrics`` dict from the
    per-run block ``state[phase][str(run_idx)]["metrics"]``. A flat read
    (``state[phase]["metrics"]``) would return ``{}`` under the per-run
    shape and lose the coverage baseline in the restore path's result."""
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    art = _write(target_dir / "tests" / "t.py", "p\n")
    ps.mark_started(
        target_dir, "coverage", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src,
    )
    ps.mark_complete(
        target_dir, "coverage", artifacts={"tests/t.py": art},
        metrics={"coverage": 91.5},
    )
    state = ps.load(target_dir)
    assert ps.phase_metrics(state, "coverage", run_idx=0) == {
        "coverage": 91.5,
    }
    # An absent phase, an absent run and an empty state all yield {}.
    assert ps.phase_metrics(state, "mutation", run_idx=0) == {}
    assert ps.phase_metrics(state, "coverage", run_idx=4) == {}
    assert ps.phase_metrics({}, "coverage", run_idx=0) == {}


def test_phase_metrics_isolates_per_run(tmp_path, default_run):
    """Two runs of the same phase keep distinct metrics. The
    accessor returns the requested run's, not a merge."""
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    a1 = _write(target_dir / "tests" / "t.py", "r1\n")
    a2 = _write(target_dir / "tests_run_2" / "t.py", "r2\n")
    ps.mark_started(
        target_dir, "coverage", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src, run_idx=0,
    )
    ps.mark_complete(
        target_dir, "coverage", artifacts={"tests/t.py": a1},
        metrics={"coverage": 80.0}, run_idx=0,
    )
    ps.mark_started(
        target_dir, "coverage", target_number=1, stem="foo",
        target_file_path="proj/foo.py", canonical_source_path=src, run_idx=1,
    )
    ps.mark_complete(
        target_dir, "coverage", artifacts={"tests_run_2/t.py": a2},
        metrics={"coverage": 95.0}, run_idx=1,
    )
    state = ps.load(target_dir)
    assert ps.phase_metrics(state, "coverage", run_idx=0)["coverage"] == 80.0
    assert ps.phase_metrics(state, "coverage", run_idx=1)["coverage"] == 95.0


def test_a_changed_final_measurement_setting_measures_again_on_resume(tmp_path):
    """The metrics unit records the settings of finishing. Under
    other values a resume measures the delivered suites again, and says
    which setting changed. Generation is untouched."""
    from testuary import config
    from testuary.evidence import finishing_settings
    from testuary.settings import RunSettings

    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"

    def settings(runs):
        return RunSettings(base_dir=tmp_path, sections={"timing": {"run_count": runs}})

    with config.use_run_settings(settings(5)):
        ps.mark_started(target_dir, "finishing", target_number=1, stem="foo",
                        target_file_path="proj/foo.py", canonical_source_path=src)
        ps.mark_complete(target_dir, "finishing", metrics={"finishing_settings": finishing_settings()})
        assert ps.validate(ps.load(target_dir), "finishing", target_dir=target_dir,
                           target_file_path="proj/foo.py", canonical_source_path=src) is True
    with config.use_run_settings(settings(30)):
        assert ps.validate(ps.load(target_dir), "finishing", target_dir=target_dir,
                           target_file_path="proj/foo.py", canonical_source_path=src) is False
        assert ps.rejection_reason(ps.load(target_dir), "finishing", target_dir=target_dir,
                                   target_file_path="proj/foo.py", canonical_source_path=src) == (
            "Finishing's settings changed: [timing] run_count")
        # A unit that recorded no settings is measured again too.
        state = ps.load(target_dir)
        del ps._read_run_block(state, "finishing", 0)["metrics"]["finishing_settings"]
        assert ps.validate(state, "finishing", target_dir=target_dir,
                           target_file_path="proj/foo.py", canonical_source_path=src) is False


# The finishing unit records whether Finishing measured
# mutation (``[finishing] measure_mutation``), and a resume
# under the other value measures the delivered suites again.
def _settings(tmp_path, value):
    return RunSettings(base_dir=tmp_path, sections={"finishing": {"measure_mutation": value}})


def test_the_recorded_setting_is_the_one_the_final_measurement_reads(tmp_path):
    for value in ("true", "false"):
        with config.use_run_settings(_settings(tmp_path, value)):
            assert finishing_settings()["measure_mutation"] == f'"{value}"'


def test_switching_the_final_mutation_measurement_measures_again_on_resume(tmp_path):
    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_foo"
    where = {"target_dir": target_dir, "target_file_path": "proj/foo.py", "canonical_source_path": src}
    with config.use_run_settings(_settings(tmp_path, "true")):
        ps.mark_started(target_dir, "finishing", target_number=1, stem="foo",
                        target_file_path="proj/foo.py", canonical_source_path=src)
        ps.mark_complete(target_dir, "finishing", metrics={"finishing_settings": finishing_settings()})
        assert ps.validate(ps.load(target_dir), "finishing", **where) is True
    with config.use_run_settings(_settings(tmp_path, "false")):
        assert ps.validate(ps.load(target_dir), "finishing", **where) is False
        assert ps.rejection_reason(ps.load(target_dir), "finishing", **where) == (
            "Finishing's settings changed: measure_mutation")


def test_a_java_container_run_records_its_jdk(tmp_path):
    """A container run of another language records the runtime its adapter
    describes, so a resume notices a changed JDK, and is complete without a
    Python interpreter; a Python run's record stays as it was."""
    from testuary.settings import resolve_settings

    def java_settings(jdk):
        return resolve_settings(models={}, config_files=[], environment={
            "TESTUARY_MAIN_PROJECT_EXEC": "container", "TESTUARY_PROJECT_CONTAINER": "job-17",
            "TESTUARY_PROJECT_CONTAINER_RUNNER": "podman", "TESTUARY_PROJECT_JAVA_HOME": jdk})

    src = _src(tmp_path)
    target_dir = tmp_path / "results" / "rank1_Foo"
    artifact = _write(target_dir / "tests" / "FooTest.java", "class FooTest {}\n")
    where = dict(target_dir=target_dir, target_file_path="src/Foo.java", canonical_source_path=src)
    with config.use_run_settings(java_settings("/opt/jdk-21")):
        assert ps._environment_identity("src/Foo.java") == {
            "runner": "podman", "python_executable": "", "runtime": "/opt/jdk-21/bin/java"}
        ps.mark_started(target_dir, "coverage", target_number=1, stem="Foo",
                        target_file_path="src/Foo.java", canonical_source_path=src)
        ps.mark_complete(target_dir, "coverage", artifacts={"tests/FooTest.java": artifact})
        assert ps.validate(ps.load(target_dir), "coverage", **where) is True
    with config.use_run_settings(java_settings("/opt/jdk-17")):
        assert ps.validate(ps.load(target_dir), "coverage", **where) is False
        assert ps.rejection_reason(ps.load(target_dir), "coverage", **where) == (
            "the container command or the project's runtime changed")
    with config.use_run_settings(_container_settings()):
        assert ps._environment_identity("proj/foo.py") == {
            "runner": "podman", "python_executable": "/opt/venv/bin/python"}


def test_the_recorded_source_hash_is_the_target_files_raw_bytes_hash(tmp_path, default_run):
    """The records the run's own writer makes give back the SHA-256 of the
    target file's bytes as read in binary, a CRLF file's included, for the
    target's path as the completion names it, and None for a target
    without records."""
    import hashlib

    from testuary import project, results

    source = tmp_path / "project" / "pkg" / "mod.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"def f():\r\n    return 1\r\n")
    output = tmp_path / "output"
    target_dir = output / "results" / project.workspace_name("pkg/mod.py")
    for phase in ("starter", "coverage"):
        ps.mark_started(target_dir, phase, target_number=1, stem="mod",
                        target_file_path=str(source), canonical_source_path=source)
    recorded = results.recorded_source_sha256(output, "pkg/mod.py")
    assert recorded == hashlib.sha256(source.read_bytes()).hexdigest()
    assert recorded != hashlib.sha256(source.read_text(encoding="utf-8").encode()).hexdigest()
    assert results.recorded_source_sha256(output, "pkg/other.py") is None


@pytest.mark.parametrize("other", ["reused", "damaged"])
def test_the_recorded_source_hash_skips_a_phase_without_one(other, tmp_path, default_run):
    """A phase the run reused records no hash of its own, and a damaged
    phase entry holds no block: the hash the other phase recorded is the
    result."""
    import hashlib

    from testuary import project, results

    source = tmp_path / "project" / "pkg" / "mod.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"def f():\n    return 1\n")
    output = tmp_path / "output"
    target_dir = output / "results" / project.workspace_name("pkg/mod.py")
    ps.mark_started(target_dir, "coverage", target_number=1, stem="mod",
                    target_file_path=str(source), canonical_source_path=source)
    if other == "reused":
        ps.record_restored(target_dir, "starter")
    else:
        state = ps.load(target_dir)
        state["mutation"] = ["not a block"]
        ps.save(target_dir, state)
    assert results.recorded_source_sha256(output, "pkg/mod.py") == (
        hashlib.sha256(source.read_bytes()).hexdigest())
