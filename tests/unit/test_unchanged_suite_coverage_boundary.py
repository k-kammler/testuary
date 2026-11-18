"""The coverage boundary on an unchanged suite: evidence reuse.

A goal met at entry, or an agent that ran and changed nothing, hands the
boundary a suite byte-identical to the one the phase inherited. Such a phase
end takes over the starter suites' evidence instead of measuring, reducing and
selecting the suite again. The tests run the real ``_consolidate_coverage`` on
the harness of ``test_coverage_boundary_identity``: the starter suites' boundary runs
for real, and everything below the mutation engine is production code. Every
shortfall of the eligibility rule takes the normal path: one engine run, two
suite executions, two evidence collections and one cover solve.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import shutil
from contextlib import contextmanager

import pytest

import test_coverage_boundary_identity as boundary
from testuary import languages, reduction
from testuary import evidence as validity
from testuary import target_phases as orchestrator

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _drive_unchanged(tmp_path, monkeypatch, *, agent_suite=None, prepare=None,
                     **drive):
    """Run the production coverage boundary over ``agent_suite`` (the
    inherited suite by default) and count the work done inside it.
    ``prepare(ctx, result)`` runs after the starter suites' boundary and before the
    coverage boundary, so a test can alter the inherited archive. The
    counters start at the coverage boundary. ``drive`` goes to the harness
    (a hook before the coverage boundary)."""
    work = {"engine_runs": 0, "suite_runs": 0, "evidence_collections": 0,
            "cover_solves": 0}
    counting = {"on": False}
    captured = {}

    def count(name):
        if counting["on"]:
            work[name] += 1

    def counting_runner(directory):
        count("engine_runs")
        return boundary._fixed_catalogue_runner(directory)

    @contextmanager
    def counting_session(**_kw):
        yield counting_runner

    backend_type = type(languages.get_backend("python"))
    real_run = backend_type.run_test_files
    real_delivered = backend_type.run_as_delivered
    real_collect = backend_type.collect_eviction_data
    real_solve = reduction.solve_cover
    real_boundary = boundary.orchestrator._consolidate_coverage

    def run_test_files(self, *args, **kwargs):
        count("suite_runs")
        return real_run(self, *args, **kwargs)

    # The normal run of Python is its run as delivered.
    def run_as_delivered(self, *args, **kwargs):
        count("suite_runs")
        return real_delivered(self, *args, **kwargs)

    def collect_eviction_data(self, *args, **kwargs):
        count("evidence_collections")
        return real_collect(self, *args, **kwargs)

    def solve_cover(*args, **kwargs):
        count("cover_solves")
        return real_solve(*args, **kwargs)

    def consolidate_coverage(ctx, result, baseline_content):
        captured["ctx"] = ctx
        captured["result"] = result
        captured["starter_before"] = _tree_bytes(
            orchestrator._archive_dir(ctx, "starter"))
        if prepare is not None:
            prepare(ctx, result)
        counting["on"] = True
        try:
            return real_boundary(ctx, result, baseline_content)
        finally:
            counting["on"] = False

    # The backend is a cached instance, so the class is patched, not the object.
    monkeypatch.setattr(backend_type, "run_test_files", run_test_files)
    monkeypatch.setattr(backend_type, "run_as_delivered", run_as_delivered)
    monkeypatch.setattr(
        backend_type, "collect_eviction_data", collect_eviction_data)
    monkeypatch.setattr(reduction, "solve_cover", solve_cover)
    monkeypatch.setattr(
        boundary.orchestrator, "_consolidate_coverage", consolidate_coverage)
    monkeypatch.setattr(boundary, "_fixed_catalogue_session", counting_session)

    try:
        run = boundary._drive_coverage_boundary(
            tmp_path, monkeypatch,
            boundary.BASELINE_SUITE if agent_suite is None else agent_suite,
            **drive)
    except BaseException as exc:
        exc.captured = captured
        raise
    run.starter_archive = orchestrator._archive_dir(run.ctx, "starter")
    run.starter_before = captured["starter_before"]
    return run, work


def _tree_bytes(root):
    """Every file under ``root`` with its bytes, keyed by relative path."""
    return {
        str(p.relative_to(root)).replace(os.sep, "/"): p.read_bytes()
        for p in root.rglob("*") if p.is_file()
    }


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


NO_WORK = {"engine_runs": 0, "suite_runs": 0, "evidence_collections": 0,
           "cover_solves": 0}

# The work on an unchanged suite without evidence reuse. Every eligibility
# shortfall falls back to it. Each suite check (the phase's own suite, the
# merge) is two runs: the normal run, then the coverage tool's.
NORMAL_WORK = {"engine_runs": 1, "suite_runs": 4, "evidence_collections": 2,
               "cover_solves": 1}

# Every file a normal coverage phase end leaves in its archive. A reusing
# phase end leaves the same set, because resume and the score derivation read
# them by name.
COVERAGE_ARCHIVE = [
    "candidate2_coverage_matrix.json",
    "candidate2_kill_matrix.json",
    "candidate_agent.py",
    "consolidation_record.json",
    "derived_reduced_kill_matrix.json",
    "measured_reduced_kill_matrix.json",
    "merged_coverage_matrix.json",
    "merged_kill_matrix.json",
    "reduced_coverage_matrix.json",
    "reduced_kill_matrix.json",
    "selected_coverage_matrix.json",
    "selected_kill_matrix.json",
    "validity_delivered.json",
    "validity_derivations.json",
    "validity_merged.json",
]


# The reuse

# Every shortfall of the eligibility rule takes the normal path

def _record_is_not_inherited(run):
    record = _json(run.archive / "consolidation_record.json")
    assert record["delivered_from"] != "inherited"
    assert "inherited_from" not in record
    return record


def test_a_changed_suite_is_consolidated(tmp_path, monkeypatch):
    """One added comment line, the tests unchanged: the bytes differ, so the
    boundary measures, reduces and selects."""
    run, work = _drive_unchanged(
        tmp_path, monkeypatch,
        agent_suite=boundary.BASELINE_SUITE + "# touched by the agent\n")

    assert work == NORMAL_WORK
    _record_is_not_inherited(run)


@pytest.mark.parametrize("missing", [
    ("validity_delivered.json",),
    ("selected_coverage_matrix.json", "reduced_coverage_matrix.json"),
    ("selected_kill_matrix.json", "reduced_kill_matrix.json"),
])
def test_a_missing_inherited_artifact_is_consolidated(
    tmp_path, monkeypatch, missing,
):
    """Without one of the files the reuse needs, the boundary takes the
    normal path, at its full cost. Each kind is removed under every name the
    archive offers it, canonical and candidate alike, since either would
    serve. Even a missing inherited kill matrix costs no additional engine
    run: the merged suite is byte-identical to the agent's candidate, whose
    standalone measurement is the one engine run, and
    ``consolidate_phase_end`` reuses a matrix for byte-identical bytes rather
    than measuring them twice."""
    def remove(ctx, result):
        archive = orchestrator._archive_dir(ctx, "starter")
        for name in missing:
            path = archive / name
            if path.exists():
                path.unlink()

    run, work = _drive_unchanged(tmp_path, monkeypatch, prepare=remove)

    assert work == NORMAL_WORK
    _record_is_not_inherited(run)


@pytest.mark.parametrize("change", [
    {"source_sha256": "0" * 64}, {"source_sha256": None},
    {"per_test_tokens": None},
])
def test_coverage_evidence_that_is_not_bound_to_the_bytes_is_consolidated(
    tmp_path, monkeypatch, change,
):
    """The reuse copies only coverage evidence bound to the delivered bytes.
    Unbinding every coverage sidecar the archive offers, canonical and
    candidate alike, leaves nothing it may copy, so the boundary
    consolidates. An unstamped payload and one without per-test tokens are
    the same case as a payload stamped with other bytes."""
    def rebind(ctx, result):
        archive = orchestrator._archive_dir(ctx, "starter")
        for name in ("selected_coverage_matrix.json",
                     "reduced_coverage_matrix.json"):
            path = archive / name
            if not path.exists():
                continue
            payload = _json(path)
            payload.update(change)
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    run, work = _drive_unchanged(tmp_path, monkeypatch, prepare=rebind)

    assert work == NORMAL_WORK
    _record_is_not_inherited(run)


def test_the_canonical_coverage_is_what_the_reuse_binds_against(
    tmp_path, monkeypatch,
):
    """The reduced sidecar describes the reduction, which is the delivered
    suite only for some winners. The canonical sidecar describes what
    shipped, for every winner, so the reuse reads it and an unbound reduced
    sidecar beside it does not prevent the reuse, also when the reduction
    did not win."""
    def unbind_the_candidate_sidecar(ctx, result):
        path = (orchestrator._archive_dir(ctx, "starter")
                / "reduced_coverage_matrix.json")
        payload = _json(path)
        payload["source_sha256"] = "0" * 64
        path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    run, work = _drive_unchanged(
        tmp_path, monkeypatch, prepare=unbind_the_candidate_sidecar)

    assert work == NO_WORK
    record = _json(run.archive / "consolidation_record.json")
    assert record["delivered_from"] == "inherited"
    assert record["inherited_artifacts"][
        "selected_coverage_matrix.json"] == "selected_coverage_matrix.json"


def test_an_unbound_inherited_validity_record_is_consolidated(
    tmp_path, monkeypatch,
):
    def unbind(ctx, result):
        path = orchestrator._archive_dir(ctx, "starter") / "validity_delivered.json"
        record = _json(path)
        record["final_clean_source_sha256"] = "0" * 64
        validity.write_json_atomic(path, record)

    run, work = _drive_unchanged(tmp_path, monkeypatch, prepare=unbind)

    assert work == NORMAL_WORK
    _record_is_not_inherited(run)


def test_the_switch_off_restores_the_path_without_reuse(tmp_path, monkeypatch, in_run):
    """``phase_end_evidence_reuse = false``, here through the configuration
    environment layer the loader applies: the unchanged suite is measured,
    reduced and selected, and the reduction wins."""
    monkeypatch.setenv("TESTUARY_COVERAGE_PHASE_END_EVIDENCE_REUSE", "false")
    in_run()

    run, work = _drive_unchanged(tmp_path, monkeypatch)

    assert work == NORMAL_WORK
    record = _record_is_not_inherited(run)
    assert record["delivered_from"] == "reduced"
    assert run.suite_path.read_text(encoding="utf-8") == boundary.BASELINE_SUITE
    assert sorted(p.name for p in run.archive.iterdir()) == COVERAGE_ARCHIVE


# Durability

def test_an_interrupted_copy_leaves_the_phase_incomplete_and_the_source_intact(
    tmp_path, monkeypatch,
):
    """The process ends after the first destination file. No record exists,
    so a resume finds the phase's archive unverifiable and redoes it. The
    inherited archive is untouched, and the redo completes the phase end."""
    real_copy = shutil.copyfile
    calls = {"n": 0}

    def failing_copy(src, dst, *args, **kwargs):
        # Only the phase end's copies into its archive count: on Linux the
        # backend's project copy also uses shutil.copyfile.
        parts = Path(dst).parts
        if "archive" in parts and "coverage" in parts[parts.index("archive"):]:
            calls["n"] += 1
            if calls["n"] == 2:
                raise OSError("simulated interruption")
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", failing_copy)
    with pytest.raises(OSError, match="simulated interruption") as info:
        _drive_unchanged(tmp_path, monkeypatch)

    ctx = info.value.captured["ctx"]
    result = info.value.captured["result"]
    coverage_archive = orchestrator._archive_dir(ctx, "coverage")
    starter_archive = orchestrator._archive_dir(ctx, "starter")
    suite_path = ctx.work_tests / "test_calc.py"
    assert not (coverage_archive / "consolidation_record.json").exists()
    assert not (ctx.work_tests / ".mutation_baseline_matrix.json").exists()
    ok, reason = orchestrator._verify_restored_phase_validity(
        ctx, "coverage", suite_path)
    assert ok is False and "missing" in reason
    assert _tree_bytes(starter_archive) == info.value.captured["starter_before"]

    monkeypatch.setattr(shutil, "copyfile", real_copy)
    orchestrator._consolidate_coverage(ctx, result, boundary.BASELINE_SUITE)

    assert sorted(p.name for p in coverage_archive.iterdir()) == COVERAGE_ARCHIVE
    assert _json(coverage_archive / "consolidation_record.json")["delivered_from"] == "inherited"
    assert orchestrator._verify_restored_phase_validity(
        ctx, "coverage", suite_path) == (True, "")
