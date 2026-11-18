"""The mutation boundary on an unchanged suite reuses the coverage phase's evidence.

The mutation agent edits the coverage-named suite in place, so an unchanged
suite still holds the bytes the coverage boundary delivered: after a goal met at
entry, or after an agent that changed nothing. The boundary then takes the
coverage evidence over, whether or not the agent's result carries its own
matrix. The tests run the real ``_consolidate_mutation`` on the harness of
``test_coverage_boundary_identity``, with everything below the mutation engine
real, and exercise the archive's consumers (the mutation agent's entry and
resume). The external score reader is not checked here.
"""
from __future__ import annotations

import json
import os
import shutil
from contextlib import contextmanager

import pytest
from langadapter.mutation import upgrade_kill_matrix

import test_coverage_boundary_identity as boundary
import test_orchestrator_phases as phases

from testuary import languages, reduction
from testuary import target_phases as orchestrator

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _drive_unchanged(tmp_path, monkeypatch, *, agent_matrix, agent_suite=None,
                     prepare=None):
    """Run the starter-suite and coverage boundaries, then the mutation
    boundary over the delivered suite (unchanged by default), counting the
    mutation boundary's work alone. ``agent_matrix`` selects whether the agent's result
    carries a matrix bound to the same bytes, as after an entry stop that
    derived its baseline from the sidecar. ``prepare(ctx, result)`` runs
    before the mutation boundary."""
    run = boundary._drive_coverage_boundary(
        tmp_path, monkeypatch, boundary.BASELINE_SUITE)
    delivered = run.suite_path.read_text(encoding="utf-8")
    sidecar = json.loads(
        (run.ctx.work_tests / ".mutation_baseline_matrix.json").read_text("utf-8"))
    run.coverage_before = _tree_bytes(run.archive)

    work = {"engine_runs": 0, "suite_runs": 0, "evidence_collections": 0,
            "cover_solves": 0}

    def counting_runner(directory):
        work["engine_runs"] += 1
        return boundary._fixed_catalogue_runner(directory)

    @contextmanager
    def counting_session(**_kw):
        yield counting_runner

    backend_type = type(languages.get_backend("python"))
    real_run = backend_type.run_test_files
    real_delivered = backend_type.run_as_delivered
    real_collect = backend_type.collect_eviction_data
    real_solve = reduction.solve_cover

    def run_test_files(self, *args, **kwargs):
        work["suite_runs"] += 1
        return real_run(self, *args, **kwargs)

    # The normal run of Python is its run as delivered.
    def run_as_delivered(self, *args, **kwargs):
        work["suite_runs"] += 1
        return real_delivered(self, *args, **kwargs)

    def collect_eviction_data(self, *args, **kwargs):
        work["evidence_collections"] += 1
        return real_collect(self, *args, **kwargs)

    def solve_cover(*args, **kwargs):
        work["cover_solves"] += 1
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(backend_type, "run_test_files", run_test_files)
    monkeypatch.setattr(backend_type, "run_as_delivered", run_as_delivered)
    monkeypatch.setattr(
        backend_type, "collect_eviction_data", collect_eviction_data)
    monkeypatch.setattr(reduction, "solve_cover", solve_cover)
    monkeypatch.setattr(
        orchestrator.measurement, "union_mutation_runner", counting_session)

    if agent_suite is not None:
        run.suite_path.write_text(agent_suite, encoding="utf-8")
    result = run.result
    result.mutation_inherited_source = delivered
    result.mutation_result = phases._mutation_result(
        run.suite_path.name, status="success",
        measurement_identity=sidecar["measurement_identity"],
        mutation_matrix=sidecar["matrix"] if agent_matrix else None,
        measured_suite_sha256=sidecar["suite_sha256"] if agent_matrix else None,
        measured_execution_name=run.suite_path.name if agent_matrix else None,
    )
    if prepare is not None:
        prepare(run.ctx, result)

    merge = orchestrator._consolidate_mutation(run.ctx, result)
    # run_target_phases stores the consolidation result on the result. The
    # checkpoint and finishing read it from there.
    result.mutation_merge_result = merge
    run.consolidated = merge
    run.starter_archive = orchestrator._archive_dir(run.ctx, "starter")
    run.mutation_archive = orchestrator._archive_dir(run.ctx, "mutation")
    run.delivered_before = delivered
    run.sidecar = sidecar
    return run, work


def _tree_bytes(root):
    return {
        str(p.relative_to(root)).replace(os.sep, "/"): p.read_bytes()
        for p in root.rglob("*") if p.is_file()
    }


def _json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def _archive_files(run):
    return sorted(
        str(p.relative_to(run.mutation_archive)).replace(os.sep, "/")
        for p in run.mutation_archive.rglob("*") if p.is_file()
    )


NO_WORK = {"engine_runs": 0, "suite_runs": 0, "evidence_collections": 0,
           "cover_solves": 0}

# Every file a normal mutation phase end leaves in its archive. A reusing
# phase end leaves the same set, because resume and the score derivation read
# them by name.
MUTATION_ARCHIVE = [
    "candidate2_coverage_matrix.json",
    "candidate2_kill_matrix.json",
    "consolidation_record.json",
    "derived_reduced_kill_matrix.json",
    "measured_reduced_kill_matrix.json",
    "merged_coverage_matrix.json",
    "merged_kill_matrix.json",
    "pre_merge/candidate_agent.py",
    "pre_merge/candidate_inherited.py",
    "reduced_coverage_matrix.json",
    "reduced_kill_matrix.json",
    "selected_coverage_matrix.json",
    "selected_kill_matrix.json",
    "validity_delivered.json",
    "validity_derivations.json",
    "validity_merged.json",
]


# The reuse

@pytest.mark.parametrize("agent_matrix", [True, False])
def test_a_reusing_phase_end_makes_no_measurement_reduction_or_selection_call(
    tmp_path, monkeypatch, agent_matrix,
):
    """With or without a matrix bound to the agent's result, the engine, the
    suite executor, the evidence collector and the cover solver are not called."""
    _run, work = _drive_unchanged(tmp_path, monkeypatch, agent_matrix=agent_matrix)

    assert work == NO_WORK


# Every shortfall of the eligibility rule takes the normal path

def _record_is_not_inherited(run):
    record = _json(run.mutation_archive / "consolidation_record.json")
    assert record["delivered_from"] != "inherited"
    assert "inherited_from" not in record
    return record


def test_a_changed_suite_is_consolidated(tmp_path, monkeypatch):
    """One added comment line: the bytes differ from the inherited suite, so
    the boundary measures the agent's candidate, merges, reduces and selects."""
    run, work = _drive_unchanged(
        tmp_path, monkeypatch, agent_matrix=False,
        agent_suite=boundary.BASELINE_SUITE + "# touched by the agent\n")

    assert work["engine_runs"] >= 1
    assert work["suite_runs"] >= 1
    assert work["cover_solves"] == 1
    assert run.consolidated.winner != "inherited"
    _record_is_not_inherited(run)


def test_a_different_inherited_measurement_identity_is_consolidated(
    tmp_path, monkeypatch,
):
    """The coverage record says its evidence was measured on a selective
    catalogue. The mutation phase requires the full one, so the inherited
    candidate is measured again and the normal path selects."""
    def selective(ctx, result):
        path = orchestrator._archive_dir(ctx, "coverage") / "consolidation_record.json"
        record = _json(path)
        record["measurement_identity"]["catalogue"] = "selective"
        path.write_text(json.dumps(record, indent=2), encoding="utf-8")

    run, work = _drive_unchanged(
        tmp_path, monkeypatch, agent_matrix=True, prepare=selective)

    assert work["engine_runs"] >= 1
    assert work["cover_solves"] == 1
    assert run.consolidated.winner != "inherited"
    _record_is_not_inherited(run)


def test_coverage_evidence_that_is_not_bound_to_the_bytes_is_consolidated(
    tmp_path, monkeypatch,
):
    """As at the coverage boundary: the inherited per-test coverage must be
    stamped with the delivered bytes' hash, or the boundary consolidates
    rather than publish another suite's evidence as the delivered suite's."""
    def rebind(ctx, result):
        archive = orchestrator._archive_dir(ctx, "coverage")
        for name in ("selected_coverage_matrix.json",
                     "reduced_coverage_matrix.json"):
            path = archive / name
            if not path.exists():
                continue
            payload = _json(path)
            payload["source_sha256"] = "0" * 64
            path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    run, work = _drive_unchanged(
        tmp_path, monkeypatch, agent_matrix=True, prepare=rebind)

    assert work["cover_solves"] == 1
    assert run.consolidated.winner != "inherited"
    _record_is_not_inherited(run)


def test_the_switch_off_restores_the_path_without_reuse(tmp_path, monkeypatch, in_run):
    """``phase_end_evidence_reuse = false`` for the mutation section only: the
    coverage boundary still reuses, and the mutation boundary checks the
    suite twice (each check the normal run, then the coverage tool's run),
    collects evidence twice and solves the cover, using the agent's bound
    matrix for the merge."""
    monkeypatch.setenv("TESTUARY_MUTATION_PHASE_END_EVIDENCE_REUSE", "false")
    in_run()

    run, work = _drive_unchanged(tmp_path, monkeypatch, agent_matrix=True)

    assert _json(run.archive / "consolidation_record.json")["delivered_from"] == "inherited"
    assert work == {
        "engine_runs": 0, "suite_runs": 4, "evidence_collections": 2,
        "cover_solves": 1,
    }
    record = _record_is_not_inherited(run)
    assert record["delivered_from"] == "reduced"
    assert run.consolidated.winner == "reduced"
    assert run.suite_path.read_text(encoding="utf-8") == run.delivered_before
    assert _archive_files(run) == MUTATION_ARCHIVE


# Consumers of the reused archives

def _drive_without_source_archives(
    tmp_path, monkeypatch,
):
    """Provenance is a field, not a link. With the starter suites' archive
    deleted the coverage phase verifies and the mutation boundary still
    reuses. With the coverage archive deleted too, resume reads the same
    evidence."""
    run = boundary._drive_coverage_boundary(
        tmp_path, monkeypatch, boundary.BASELINE_SUITE)
    starter_archive = orchestrator._archive_dir(run.ctx, "starter")
    # The archived matrix as every reader takes it, in this version's form.
    starter_km = upgrade_kill_matrix(_json(starter_archive / "selected_kill_matrix.json"))
    coverage_before = _tree_bytes(run.archive)
    shutil.rmtree(starter_archive)

    assert orchestrator._verify_restored_phase_validity(
        run.ctx, "coverage", run.suite_path) == (True, "")
    assert _tree_bytes(run.archive) == coverage_before

    def forbidden(**_kw):
        raise AssertionError("the mutation boundary opened the engine")

    monkeypatch.setattr(
        orchestrator.measurement, "union_mutation_runner", forbidden)
    sidecar = _json(run.ctx.work_tests / ".mutation_baseline_matrix.json")
    run.result.mutation_inherited_source = run.suite_path.read_text("utf-8")
    run.result.mutation_result = phases._mutation_result(
        run.suite_path.name, status="success",
        measurement_identity=sidecar["measurement_identity"],
        mutation_matrix=None,
    )
    merge = orchestrator._consolidate_mutation(run.ctx, run.result)
    assert merge.winner == "inherited"
    assert merge.kill_matrix == starter_km

    mutation_before = _tree_bytes(orchestrator._archive_dir(run.ctx, "mutation"))
    shutil.rmtree(run.archive)
    run.result.mutation_merge_result = None
    assert orchestrator._resume_standard_km(run.ctx, run.result) == starter_km
    assert orchestrator._verify_restored_phase_validity(
        run.ctx, "mutation", run.suite_path) == (True, "")
    assert _tree_bytes(orchestrator._archive_dir(run.ctx, "mutation")) == mutation_before
    return run, starter_km


def test_the_result_does_not_depend_on_the_source_archives(tmp_path, monkeypatch):
    _drive_without_source_archives(tmp_path, monkeypatch)
