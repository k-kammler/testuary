"""Finishing reads the archived phase outputs and runs no mutation engine.

Per-phase columns come from each phase's persisted matrices
(``_attach_phase_scores``) and finishing reuses the matrix persisted at the
mutation phase's end (``_resume_standard_km`` into ``measure_finishing``).
The target here measured the starter phase's end on an unchanged suite and
reused that evidence at the ends of the coverage and mutation phases, so every
per-phase and final number equals the starter measurement and the records
name the source phases.
"""
from __future__ import annotations

import json

import pytest

import test_orchestrator_phases as phases
import test_unchanged_suite_mutation_boundary as mutation_boundary
from testuary import languages
from testuary.languages import mutation as _kill_matrix
from testuary import measurement
from testuary import target_phases as orchestrator

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture
def target(tmp_path, monkeypatch):
    """A target after all three phase ends ran on an unchanged suite."""
    run, _work = mutation_boundary._drive_unchanged(
        tmp_path, monkeypatch, agent_matrix=True)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("Finishing touched the mutation engine")

    monkeypatch.setattr(
        orchestrator.measurement, "union_mutation_runner", forbidden)
    monkeypatch.setattr(
        type(languages.get_backend("python")), "mutation_runner", forbidden)
    return run


def _archived(run, phase, name):
    return json.loads(
        (run.ctx.results_dir / run.ctx.workspace / "r0" / "archive" / phase / name)
        .read_text("utf-8"))


def test_per_phase_columns_are_read_from_the_archived_matrices(target):
    """Each phase's columns are its delivered suite's persisted matrix and
    coverage. A reusing coverage phase end copies the starter files, so the
    coverage columns equal the starter ones and name the starter phase."""
    combined = phases._metrics_result(target_file_path=target.target)

    orchestrator._attach_phase_scores(target.ctx, combined)

    for phase in ("starter", "coverage"):
        km = _archived(target, phase, "selected_kill_matrix.json")
        cov = _archived(target, phase, "selected_coverage_matrix.json")
        got = combined.phase_finishing[phase]
        assert got.mutation_score_full == _kill_matrix.matrix_score_percent(km)
        assert got.line_coverage == cov["line"]
        assert got.branch_coverage == cov["branch"]
    starter = combined.phase_finishing["starter"]
    coverage = combined.phase_finishing["coverage"]
    assert (coverage.mutation_score_full, coverage.line_coverage,
            coverage.branch_coverage) == (
        starter.mutation_score_full, starter.line_coverage,
        starter.branch_coverage)
    assert _archived(target, "coverage", "consolidation_record.json")[
        "inherited_from"] == "starter"
    assert _archived(target, "mutation", "consolidation_record.json")[
        "inherited_from"] == "coverage"


def test_the_finishing_mutation_score_after_two_reuses_is_the_starter_measurement(
    target, tmp_path,
):
    """Both later phase ends inherited, so finishing reuses the starter
    matrix and the final score equals the starter one."""
    starter_km = _archived(target, "starter", "selected_kill_matrix.json")
    runs = []

    finishing = measurement.measure_finishing(
        scratch_root=tmp_path / "finishing_scratch",
        merged_test_path=target.suite_path,
        shared_setup_paths=[],
        run_mutation_on=lambda directory: runs.append(directory),
        standard_coverage=(100.0, 100.0),
        inherited_standard_km=orchestrator._resume_standard_km(
            target.ctx, target.result),
    )

    assert runs == []
    assert finishing.mutation_score_full == (
        _kill_matrix.matrix_score_percent(starter_km))
    combined = phases._metrics_result(target_file_path=target.target)
    combined.finishing = finishing
    orchestrator._attach_phase_scores(target.ctx, combined)
    scores = {
        phase: combined.phase_finishing[phase].mutation_score_full
        for phase in ("starter", "coverage", "mutation")
    }
    assert scores == {
        phase: _kill_matrix.matrix_score_percent(starter_km)
        for phase in ("starter", "coverage", "mutation")
    }


def test_phase_scores_off_reads_nothing(target, monkeypatch, in_run):
    monkeypatch.setenv("TESTUARY_FINISHING_PHASE_SCORES", "off")
    in_run()
    combined = phases._metrics_result(target_file_path=target.target)

    orchestrator._attach_phase_scores(target.ctx, combined)

    assert combined.phase_finishing == {}


def test_finishing_takes_the_mutation_boundarys_persisted_matrix(target):
    """The standard suite's final matrix is the one persisted at the mutation
    phase's end, from the merge result on a fresh run and from the archive on
    a resume, read in this version's form."""
    from langadapter.mutation import upgrade_kill_matrix

    persisted = _archived(target, "mutation", "selected_kill_matrix.json")

    fresh = orchestrator._resume_standard_km(target.ctx, target.result)
    assert fresh == target.consolidated.kill_matrix

    target.result.mutation_merge_result = None
    resumed = orchestrator._resume_standard_km(target.ctx, target.result)
    assert resumed == upgrade_kill_matrix(persisted)
    assert {m["id"]: m["status"] for m in resumed["mutants"]} == {
        m["id"]: m["status"] for m in fresh["mutants"]}


def test_measure_finishing_runs_no_mutation_with_an_inherited_matrix(
    target, tmp_path,
):
    """Given that matrix, finishing scores it without calling the mutation
    runner."""
    km = _archived(target, "mutation", "selected_kill_matrix.json")
    runs = []

    def run_mutation_on(directory):
        runs.append(directory)
        return None

    finishing = measurement.measure_finishing(
        scratch_root=tmp_path / "finishing_scratch",
        merged_test_path=target.suite_path,
        shared_setup_paths=[],
        run_mutation_on=run_mutation_on,
        standard_coverage=(100.0, 100.0),
        inherited_standard_km=km,
    )

    assert runs == []
    assert finishing.mutation_score_full == _kill_matrix.matrix_score_percent(km)
    assert finishing.mutation_measurement == "complete"
