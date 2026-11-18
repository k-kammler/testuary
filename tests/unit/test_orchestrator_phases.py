"""The per-phase orchestrator helpers and ``run_target_phases`` with mocked agents.

Each ``_phase_*`` helper invokes its agent, records its outcome and passes
results through the pipeline. The agents are mocked, so no model is called.
The pure helpers are covered in ``test_orchestrator.py``.
"""
from __future__ import annotations

import hashlib
import json
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from testuary import target_phases as orchestrator
from testuary.agent.coverage_agent import CoverageTestResult
from testuary.agent.mutation_agent import MutationTestResult
from testuary.measurement import CombinedTestMetrics, FinishingMetrics
from phase_values import set_phase_value

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture(autouse=True)
def _stub_finishing(monkeypatch):
    """Stub run_finishing so no test opens a Cosmic Ray session, whose
    per-worker prepared copies and their cleanup take about a minute per test
    on a slow disk.

    The stub carries a standard-suite full-catalogue mutation score, as a green
    finishing does, because finishing records ``ok`` only when that score is
    present. Tests of the absent-score or per-phase paths override
    run_finishing themselves."""
    monkeypatch.setattr(
        orchestrator.measurement, "run_finishing",
        lambda *a, **kw: FinishingMetrics(mutation_score_full=75.0),
    )
    monkeypatch.setattr(
        orchestrator,
        "_verify_restored_phase_validity",
        lambda *_args, **_kwargs: (True, ""),
    )


def _coverage_result(
    test_file_name: str = "test_foo.py",
    *, status: str = "ok", line_coverage: float = 80.0, **kw,
) -> CoverageTestResult:
    """Build a CoverageTestResult with every required field set to a
    deterministic default."""
    defaults: dict[str, Any] = dict(
        test_file_name=test_file_name,
        explanation="",
        targeted_functions=[],
        line_coverage=line_coverage,
        # Above the shipped minimum, so the below-minimum skip does
        # not fire. test_below_minimum.py covers that skip.
        branch_coverage=85.0,
        mutation_score=None,
        test_count_standard=0,
        test_loc_standard=0,
        target_loc=0,
        target_complexity=0,
        agent_iterations=0,
        agent_writes=0,
        agent_runs=0,
        agent_assess_cov=0,
        agent_assess_mut=0,
        agent_duration_s=0.0,
        status=status,
        error=None,
    )
    defaults.update(kw)
    return CoverageTestResult(**defaults)


def _mutation_result(
    mutation_test_file_name: str = "test_foo_mutation.py",
    *, mutation_score: float = 70.0, status: str = "ok", **kw,
) -> MutationTestResult:
    defaults: dict[str, Any] = dict(
        mutation_test_file_name=mutation_test_file_name,
        explanation="",
        targeted_mutants=[],
        mutation_score=mutation_score,
        test_count_mutation=0,
        test_loc_mutation=0,
        agent_iterations=0,
        agent_duration_s=0.0,
        status=status,
        error=None,
    )
    defaults.update(kw)
    return MutationTestResult(**defaults)


def _metrics_result(
    *, coverage_final_line: float | None = 90.0,
    coverage_final_branch: float | None = 80.0,
    **kw,
) -> CombinedTestMetrics:
    defaults: dict[str, Any] = dict(
        target_file_path="",
        target_complexity_sum=0,
        target_loc=0,
        coverage_final_line=coverage_final_line,
        coverage_final_branch=coverage_final_branch,
        lint_smells_target=0,
        lint_smells_tests=0,
        test_files=[],
        duration_s=0.0,
        status="ok",
        error=None,
    )
    defaults.update(kw)
    return CombinedTestMetrics(**defaults)


def _make_ctx(tmp_path: Path, **overrides) -> orchestrator.TargetPhasesContext:
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    work_tests = work_dir / "tests"
    work_tests.mkdir()
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src_file = src_dir / "foo.py"
    src_file.write_text("def f(): return 1\n", encoding="utf-8")
    # The marker the source-dir walk stops at. Without one it climbs out of
    # the temporary tree and resolves to whatever repository encloses it, so
    # the phases would measure and clean against that tree instead of this one.
    (src_dir / "pyproject.toml").write_text("", encoding="utf-8")
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    defaults = dict(
        project_name="proj",
        workspace="rank1_foo",
        top_file={"file_path": str(src_file), "key_functions": ["f"], "target_number": 1},
        canonical_path=src_file,
        work_tests=work_tests,
        results_dir=results_dir,
        target_logs_dir=log_dir,
        reuse=orchestrator.ReuseFlags(),
        log_print=lambda *a, **k: None,
    )
    defaults.update(overrides)
    return orchestrator.TargetPhasesContext(**defaults)


def _observe_checkpoints(monkeypatch, start=None, done=None) -> None:
    """Observe the phases' checkpoints (``_record_start`` and
    ``_record_done``) in place of their phase_state writes."""
    if start is not None:
        monkeypatch.setattr(orchestrator, "_record_start", lambda _ctx, phase: start(phase))
    if done is not None:
        monkeypatch.setattr(orchestrator, "_record_done",
                            lambda _ctx, phase, *outcome: done(phase, *outcome))


def _patch_starter_handoff(monkeypatch, ctx) -> None:
    baseline = ctx.work_tests.parent / "phase1_baseline.py"
    baseline.write_text(
        "from foo import f\n\ndef test_phase1():\n    assert f() == 1\n",
        encoding="utf-8",
    )
    def fake_starter(_ctx, result):
        orchestrator._record_outcome(result, "starter", "ok", None)
        return orchestrator.starters.StarterResult(
            status="ok",
            baseline_file=str(baseline),
            selected_tests=1,
            baseline_line_coverage=50.0,
            baseline_branch_coverage=50.0,
        )

    monkeypatch.setattr(orchestrator, "_phase_starter", fake_starter)
    monkeypatch.setattr(
        orchestrator, "_consolidate_coverage", lambda *args, **kwargs: None,
    )


def test_consolidation_stats_keep_the_actual_fallback_winner_and_count():
    phase = SimpleNamespace(pool_tests=500)
    winner = SimpleNamespace(label="agent", test_count=468)

    coverage = orchestrator._coverage_consolidation_stats(phase, winner)
    merge = orchestrator.measurement.MergeResult(
        merged_path="test_x.py",
        pre_merge_coverage_path="coverage.py",
        pre_merge_mutation_path="mutation.py",
        union_tests=176,
        delivered_tests=133,
        winner="single_suite",
    )
    mutation = orchestrator._mutation_consolidation_stats(merge)

    assert coverage == {
        "merged_tests": 500,
        "delivered_tests": 468,
        "delivered_from": "agent",
    }
    assert mutation == {
        "merged_tests": 176,
        "delivered_tests": 133,
        "delivered_from": "single_suite",
    }


def _version_2_archive(ctx, phase):
    """An archive of ``phase`` whose delivered kill matrix is of schema
    version 2, in that version's words."""
    adir = ctx.results_dir / ctx.workspace / "r0" / "archive" / phase
    adir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha256(b"the delivered suite").hexdigest()
    (adir / "selected_kill_matrix.json").write_text(json.dumps({
        "schema_version": 2, "source_sha256": digest, "summary": {"killed": 1, "suspicious": 1},
        "mutants": [{"id": "m1", "status": "ok_killed"}, {"id": "m2", "status": "bad_survived"},
                    {"id": "m3", "status": "ok_suspicious"}]}), encoding="utf-8")
    (adir / "consolidation_record.json").write_text(
        json.dumps({"selected_source_sha256": digest}), encoding="utf-8")
    return adir


def test_a_delivered_version_2_matrix_reads_in_this_version_s_words(tmp_path):
    """The loader of a phase's delivered kill matrix, which the starter and
    coverage reuse read, gives a version-2 matrix in schema 3's words."""
    ctx = _make_ctx(tmp_path)
    km = orchestrator._delivered_kill_matrix(_version_2_archive(ctx, "coverage"), ctx.log_print)
    assert km["schema_version"] == 3
    assert [m["status"] for m in km["mutants"]] == ["killed", "survived", "non_viable"]
    assert km["summary"] == {"detected": 1, "non_viable": 1}


def test_a_resume_reads_a_stored_version_2_matrix_in_this_version_s_words(tmp_path):
    """A resumed target's stored mutation matrix of schema version 2 reads in
    schema 3's words."""
    ctx = _make_ctx(tmp_path)
    _version_2_archive(ctx, "mutation")
    km = orchestrator._resume_standard_km(ctx, SimpleNamespace(mutation_merge_result=None))
    assert km["schema_version"] == 3
    assert [m["status"] for m in km["mutants"]] == ["killed", "survived", "non_viable"]


def test_a_delivered_matrix_of_a_later_schema_is_not_used(tmp_path):
    """A delivered kill matrix of a schema version this version cannot read
    is absent for its readers, with the reason logged, as evidence of other
    bytes is."""
    ctx = _make_ctx(tmp_path)
    adir = _version_2_archive(ctx, "coverage")
    stored = json.loads((adir / "selected_kill_matrix.json").read_text(encoding="utf-8"))
    (adir / "selected_kill_matrix.json").write_text(
        json.dumps({**stored, "schema_version": 99}), encoding="utf-8")
    logged = []
    assert orchestrator._delivered_kill_matrix(adir, lambda line, **kw: logged.append(line)) is None
    assert len(logged) == 1 and "coverage/selected_kill_matrix.json" in logged[0]


def test_resume_standard_km_refuses_a_matrix_of_other_bytes(tmp_path):
    """A resume does not report a matrix whose own hash names bytes other
    than the ones the mutation record says were delivered: it has none to
    reuse, as when the archive holds no matrix."""
    ctx = _make_ctx(tmp_path)
    mdir = ctx.results_dir / ctx.workspace / "r0" / "archive" / "mutation"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "selected_kill_matrix.json").write_text(
        json.dumps({"mutants": [{"id": "m1", "status": "killed"}],
                    "source_sha256": hashlib.sha256(
                        b"another suite").hexdigest()}),
        encoding="utf-8")
    (mdir / "consolidation_record.json").write_text(
        json.dumps({"selected_source_sha256": hashlib.sha256(
            b"the delivered suite").hexdigest()}),
        encoding="utf-8")
    result = SimpleNamespace(mutation_merge_result=None)
    assert orchestrator._resume_standard_km(ctx, result) is None


def test_resume_standard_km_prefers_in_memory_fresh_run(tmp_path):
    # A fresh run uses the merged matrix in memory, not its persisted copy.
    ctx = _make_ctx(tmp_path)
    mdir = ctx.results_dir / ctx.workspace / "r0" / "archive" / "mutation"
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "merged_kill_matrix.json").write_text(
        json.dumps({"mutants": [{"id": "disk"}]}), encoding="utf-8")
    fresh = {"mutants": [{"id": "mem", "status": "killed"}]}
    result = SimpleNamespace(mutation_merge_result=SimpleNamespace(kill_matrix=fresh))
    assert orchestrator._resume_standard_km(ctx, result) == fresh


def test_resume_standard_km_none_when_no_matrix(tmp_path):
    # No in-memory result and no persisted matrix -> None, so the green-checked
    # re-run path of finishing decides (it must refuse a non-green
    # baseline rather than emit a false score).
    ctx = _make_ctx(tmp_path)
    assert orchestrator._resume_standard_km(
        ctx, SimpleNamespace(mutation_merge_result=None)) is None


def test_snapshot_phase_original_preserves_source_bound_bytes(tmp_path):
    """The archive snapshot is byte-identical to the selected source so its
    recorded measurements remain bound to the same artifact."""
    ctx = _make_ctx(tmp_path)
    src = ctx.work_tests / "test_foo.py"
    live = (
        "# === banner of a pruned section ===\n"
        "\n"
        "# attached rationale\n"
        "def test_x():\n"
        "    assert True\n"
    )
    src.write_text(live, encoding="utf-8")
    artifacts: dict[str, str] = {}
    orchestrator._snapshot_phase_original(ctx, artifacts, "coverage", src)
    dest = ctx.work_tests.parent / "originals" / "test_foo_coverage.py"
    snapshot = dest.read_text(encoding="utf-8")
    assert src.read_text(encoding="utf-8") == live
    assert snapshot == live


def test_phase_snapshots_dir_follows_work_tests_not_work_dir(tmp_path):
    """The originals snapshot dir is a sibling of the test dir. In container
    mode work_tests is the bind-mounted work directory (not work_dir), so deriving
    the dir from work_tests keeps the snapshot in the tree the in-container
    measurement can reach."""
    zone = tmp_path / "workzone"
    (zone / "tests").mkdir(parents=True)
    ctx = _make_ctx(tmp_path, work_tests=zone / "tests")
    assert orchestrator._phase_snapshots_dir(ctx) == zone / "originals"
    assert orchestrator._phase_snapshots_dir(ctx) != tmp_path / "work" / "originals"


def test_snapshot_phase_original_missing_src_is_noop(tmp_path):
    ctx = _make_ctx(tmp_path)
    artifacts: dict[str, str] = {}
    orchestrator._snapshot_phase_original(
        ctx, artifacts, "starter", ctx.work_tests / "absent.py",
    )
    assert artifacts == {}
    assert not (ctx.work_tests.parent / "originals").exists()


def test_attach_phase_scores_off_reuses_only_mutation(tmp_path, monkeypatch, in_run):
    """mode=off attaches nothing but the reused mutation result of finishing."""
    monkeypatch.setenv("TESTUARY_FINISHING_PHASE_SCORES", "off")
    in_run()
    ctx = _make_ctx(tmp_path)
    combined = _metrics_result()
    combined.finishing = FinishingMetrics(line_coverage=88.0)
    orchestrator._attach_phase_scores(
        ctx, combined,
    )
    assert combined.phase_finishing["mutation"] is combined.finishing
    assert "starter" not in combined.phase_finishing


def test_attach_phase_scores_absent_matrices_leave_empty(tmp_path, monkeypatch, in_run):
    """A phase whose matrices are absent leaves its per-phase columns empty
    rather than failing the target. The mutation reuse still stands."""
    monkeypatch.setenv("TESTUARY_FINISHING_PHASE_SCORES", "full")
    in_run()
    ctx = _make_ctx(tmp_path)
    combined = _metrics_result()
    combined.finishing = FinishingMetrics(line_coverage=88.0)
    orchestrator._attach_phase_scores(
        ctx, combined,
    )
    assert "starter" not in combined.phase_finishing
    assert combined.phase_finishing["mutation"] is combined.finishing


def test_attach_phase_scores_reads_the_delivered_suite_when_agent_stood(
    tmp_path, monkeypatch, in_run,
):
    """When the coverage agent's own suite stands (winner ``agent``), the
    coverage columns come from that delivered suite's canonical files, not
    the losing reduction's files with their lower line and branch coverage.
    The reader does not branch on the winner."""
    monkeypatch.setenv("TESTUARY_FINISHING_PHASE_SCORES", "full")
    in_run()
    ctx = _make_ctx(tmp_path)
    adir = orchestrator._archive_dir(ctx, "coverage")
    digest = hashlib.sha256(b"the agent suite").hexdigest()
    (adir / "consolidation_record.json").write_text(
        json.dumps({"delivered_from": "agent", "selected_source_sha256": digest}),
        encoding="utf-8")
    # The delivered agent suite's own measurement, which must be read.
    (adir / "selected_coverage_matrix.json").write_text(
        json.dumps({"line": 90.0, "branch": 85.0, "source_sha256": digest}),
        encoding="utf-8")
    (adir / "selected_kill_matrix.json").write_text(
        json.dumps({"mutants": [
            {"id": "a", "status": "killed"},
            {"id": "b", "status": "killed"},
            {"id": "c", "status": "survived"},
        ], "source_sha256": digest}), encoding="utf-8")
    # The losing reduction describes itself. None of it may be read.
    reduction = hashlib.sha256(b"the reduction").hexdigest()
    (adir / "reduced_coverage_matrix.json").write_text(
        json.dumps({"line": 70.0, "branch": 60.0,
                    "source_sha256": reduction}), encoding="utf-8")
    (adir / "reduced_kill_matrix.json").write_text(
        json.dumps({"mutants": [
            {"id": "a", "status": "survived"},
            {"id": "b", "status": "survived"},
            {"id": "c", "status": "survived"},
        ], "source_sha256": reduction}), encoding="utf-8")
    combined = _metrics_result()
    combined.finishing = FinishingMetrics(line_coverage=88.0)
    orchestrator._attach_phase_scores(
        ctx, combined,
    )
    cov = combined.phase_finishing["coverage"]
    assert cov.line_coverage == 90.0
    assert cov.branch_coverage == 85.0
    # The delivered suite's 2 killed of 3, never blank when the agent stood
    # with a measured matrix.
    assert cov.mutation_score_full == pytest.approx(2 / 3 * 100.0)


def test_attach_phase_scores_refuses_evidence_of_other_bytes(
    tmp_path, monkeypatch, in_run,
):
    """A canonical file whose own hash names bytes other than the ones the
    phase record says were delivered is refused, not reported as the
    delivered suite's numbers: the phase's columns stay empty, as for a phase
    that persisted no evidence."""
    monkeypatch.setenv("TESTUARY_FINISHING_PHASE_SCORES", "full")
    in_run()
    ctx = _make_ctx(tmp_path)
    adir = orchestrator._archive_dir(ctx, "coverage")
    (adir / "consolidation_record.json").write_text(
        json.dumps({"selected_source_sha256": hashlib.sha256(
            b"the delivered suite").hexdigest()}),
        encoding="utf-8")
    (adir / "selected_coverage_matrix.json").write_text(
        json.dumps({"line": 90.0, "branch": 85.0,
                    "source_sha256": hashlib.sha256(
                        b"another suite").hexdigest()}),
        encoding="utf-8")
    combined = _metrics_result()
    combined.finishing = FinishingMetrics(line_coverage=88.0)
    orchestrator._attach_phase_scores(
        ctx, combined,
    )
    assert "coverage" not in combined.phase_finishing


def _patch_agent(monkeypatch, mod, **calls: Any) -> dict:
    """Mock ``init_logging``, ``save_logs`` and the methods passed by name on
    an agent module. Returns a call counter with one slot per ``calls`` key,
    so a test can assert ``counts['<name>'] == 0`` for a phase not invoked."""
    counts: dict = {"init": 0, "save": 0}

    def fake_init(*a, **kw):
        counts["init"] += 1

    def fake_save(*a, **kw):
        counts["save"] += 1

    monkeypatch.setattr(mod, "init_logging", fake_init)
    monkeypatch.setattr(mod, "save_logs", fake_save)
    for name, value in calls.items():
        counts[name] = 0
        # Wrap each supplied callable to count its invocations.
        def _wrap(orig, _name=name):
            def _wrapped(*a, **kw):
                counts[_name] += 1
                return orig(*a, **kw)
            return _wrapped
        monkeypatch.setattr(mod, name, _wrap(value))
    return counts


def test_phase_coverage_records_failed_when_file_missing(monkeypatch, tmp_path):
    """A returned test file name without the file on disk records 'failed'
    with reason 'file missing on disk'."""
    ctx = _make_ctx(tmp_path)
    fake_result = _coverage_result("test_foo.py", status="ok", line_coverage=0.0)
    _patch_agent(
        monkeypatch, orchestrator.coverage_agent,
        generate_coverage_tests=lambda *a, **kw: fake_result,
    )

    result = orchestrator.TestPipelineResult()
    orchestrator._phase_coverage(ctx, result)
    assert result.phase_outcomes["coverage"] == "failed"
    assert "missing" in result.phase_outcomes["coverage_detail"]


def test_phase_coverage_records_failed_when_test_failed_artifact(
    monkeypatch, tmp_path,
):
    """A synthesized ``test_failed.py`` artifact records 'failed' with a
    detail naming the artifact and the status."""
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_failed.py").write_text("# fail\n")
    fake_result = _coverage_result(
        "test_failed.py", status="agent_exhausted_retries", line_coverage=0.0,
    )
    _patch_agent(
        monkeypatch, orchestrator.coverage_agent,
        generate_coverage_tests=lambda *a, **kw: fake_result,
    )
    result = orchestrator.TestPipelineResult()
    orchestrator._phase_coverage(ctx, result)
    assert result.phase_outcomes["coverage"] == "failed"
    assert "test_failed.py" in result.phase_outcomes["coverage_detail"]
    assert "agent_exhausted_retries" in result.phase_outcomes["coverage_detail"]


def test_phase_coverage_records_failed_on_agent_error(monkeypatch, tmp_path):
    """A hard terminal status fails the phase even with a file on disk, as in
    the mutation phase. The artifact is partial, and reading it as ok would
    hide the failure from the target's record."""
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text("def test_x(): pass\n")
    fake_result = _coverage_result(
        "test_foo.py", status="agent_error", line_coverage=61.0,
        error="non-retryable error: Error code: 402 - Insufficient Balance",
    )
    _patch_agent(
        monkeypatch, orchestrator.coverage_agent,
        generate_coverage_tests=lambda *a, **kw: fake_result,
    )
    result = orchestrator.TestPipelineResult()
    orchestrator._phase_coverage(ctx, result)
    assert result.phase_outcomes["coverage"] == "failed"
    detail = result.phase_outcomes["coverage_detail"]
    assert "agent_error" in detail and "402" in detail


def test_phase_coverage_limit_stop_without_measurement_fails(
    monkeypatch, tmp_path,
):
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text("def test_x(): pass\n")
    fake_result = _coverage_result(
        "test_foo.py", status="budget_exhausted", line_coverage=None,
    )
    _patch_agent(
        monkeypatch, orchestrator.coverage_agent,
        generate_coverage_tests=lambda *a, **kw: fake_result,
    )
    result = orchestrator.TestPipelineResult()
    orchestrator._phase_coverage(ctx, result)
    assert result.phase_outcomes["coverage"] == "failed"
    assert "unmeasured" in result.phase_outcomes["coverage_detail"]


def test_phase_coverage_unknown_status_fails_closed(monkeypatch, tmp_path):
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text("def test_x(): pass\n")
    fake_result = _coverage_result(
        "test_foo.py", status=None, line_coverage=80.0,
    )
    _patch_agent(
        monkeypatch, orchestrator.coverage_agent,
        generate_coverage_tests=lambda *a, **kw: fake_result,
    )
    result = orchestrator.TestPipelineResult()
    orchestrator._phase_coverage(ctx, result)
    assert result.phase_outcomes["coverage"] == "failed"
    assert "no status" in result.phase_outcomes["coverage_detail"]


def test_phase_coverage_save_logs_called_even_on_exception(
    monkeypatch, tmp_path,
):
    """save_logs runs in a ``finally`` even when the agent raises, so the log
    survives a crash."""
    ctx = _make_ctx(tmp_path)
    counts = _patch_agent(
        monkeypatch, orchestrator.coverage_agent,
        generate_coverage_tests=lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("boom"),
        ),
    )
    with pytest.raises(RuntimeError):
        orchestrator._phase_coverage(ctx, orchestrator.TestPipelineResult())
    assert counts["save"] == 1


def test_restore_coverage_picks_preferred_filename(tmp_path, monkeypatch):
    """The preferred filename ``test_<stem>.py`` wins over any other
    test_*.py. The collect-only check is patched because the fixture file
    has no test functions. A test below covers the real collect path."""
    monkeypatch.setattr(
        orchestrator, "_restored_test_file_collects",
        lambda _ctx, _name: (True, ""),
    )
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text("# preferred")
    (ctx.work_tests / "test_other.py").write_text("# fallback")
    result = orchestrator.TestPipelineResult()
    name = orchestrator._restore_coverage_test_file_from_disk(ctx, result)
    assert name == "test_foo.py"
    assert result.phase_outcomes["coverage"] == "skipped_reused"


def test_restore_coverage_falls_back_to_any_test_file(tmp_path, monkeypatch):
    """Without test_<stem>.py a generic test_*.py other than test_failed.py
    is picked."""
    monkeypatch.setattr(
        orchestrator, "_restored_test_file_collects",
        lambda _ctx, _name: (True, ""),
    )
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_helpers.py").write_text("# alt")
    # Decoys that must not be picked.
    (ctx.work_tests / "test_failed.py").write_text("# failed")
    result = orchestrator.TestPipelineResult()
    name = orchestrator._restore_coverage_test_file_from_disk(ctx, result)
    assert name == "test_helpers.py"
    assert result.phase_outcomes["coverage"] == "skipped_reused"


def test_restore_coverage_fails_when_restored_file_doesnt_collect(
    tmp_path, monkeypatch,
):
    """A restored test file that no longer collects against the current
    source (for example one naming a function the source no longer has)
    fails coverage at restore time, so mutation and metrics are skipped.
    This guards ``--reuse coverage`` against a corrupted test file."""
    monkeypatch.setattr(
        orchestrator, "_restored_test_file_collects",
        lambda _ctx, _name: (False, "NameError: name 'lo' is not defined"),
    )
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text("# stale")
    result = orchestrator.TestPipelineResult()
    name = orchestrator._restore_coverage_test_file_from_disk(ctx, result)
    assert name is None
    assert result.phase_outcomes["coverage"] == "failed"
    err = result.phase_outcomes.get("coverage_detail", "")
    assert "test_foo.py" in err
    assert "does not collect" in err
    assert "NameError" in err


@pytest.mark.parametrize("mutation_reused", [True, False])
def test_restore_with_mutation_reused_checks_the_suite_as_delivered(
        tmp_path, monkeypatch, mutation_reused):
    """With the mutation phase reused as well, no phase checks the restored
    delivered suite as delivered: the restore does, and a suite that
    does not pass as delivered fails the target instead of being delivered
    unchecked. With the mutation phase run, its entry check does it."""
    monkeypatch.setattr(
        orchestrator, "_restored_test_file_collects",
        lambda _ctx, _name: (True, ""),
    )
    checked = []

    def check(_ctx, path):
        checked.append(path.name)
        return False, ("the restored suite does not pass as delivered "
                       "(failing: test_foo.test_seed)")

    monkeypatch.setattr(orchestrator, "_check_restored_delivered_suite",
                        check)
    ctx = _make_ctx(tmp_path, reuse=orchestrator.ReuseFlags(
        coverage=True, mutation=mutation_reused))
    (ctx.work_tests / "test_foo.py").write_text("# restored")
    result = orchestrator.TestPipelineResult()
    name = orchestrator._restore_coverage_test_file_from_disk(ctx, result)
    if not mutation_reused:
        assert name == "test_foo.py" and checked == []
        return
    assert name is None and checked == ["test_foo.py"]
    assert result.phase_outcomes["coverage"] == "failed"
    assert "does not pass as delivered" in result.phase_outcomes[
        "coverage_detail"]


def test_restore_coverage_no_file_hard_fails_the_target(tmp_path):
    """Reject coverage reuse when its required suite is absent, before mutation runs."""
    ctx = _make_ctx(tmp_path)
    result = orchestrator.TestPipelineResult()
    name = orchestrator._restore_coverage_test_file_from_disk(ctx, result)
    assert name is None
    assert result.phase_outcomes["coverage"] == "failed"
    assert result.coverage_phase_failed is True


def test_reused_target_test_present_detects_the_delivered_file(tmp_path):
    """The reuse check sees the delivered test file only once it exists on
    disk, so --reuse mutation can tell 'reuse this' from the
    hard-fail 'nothing to reuse' case."""
    from testuary import languages
    ctx = _make_ctx(tmp_path)
    assert orchestrator._reused_target_test_present(ctx) is False
    backend = languages.get_backend("python")
    p = backend.target_test_path(
        str(ctx.work_tests), str(ctx.canonical_path),
    )
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("def test_x(): pass\n", encoding="utf-8")
    assert orchestrator._reused_target_test_present(ctx) is True


def test_a_restored_test_file_whose_run_observed_no_count_did_not_load(tmp_path, monkeypatch):
    """A run without its record, such as a crash, observes no count, which
    the language layer reports as None: the load check reads it as nothing
    collected and refuses the file."""
    from types import SimpleNamespace

    class _Crashed:
        def setup_test_env(self, test_output_dir, source_dir, target_file_path):
            pass

        def run_test_files(self, paths, source_dir, **kw):
            return SimpleNamespace(ok=False, returncode=-11, timed_out=False, collected=None, passed=None,
                                   failed=None, errors=None, output="Segmentation fault",
                                   line_coverage=None, branch_coverage=None, elapsed_s=0.1)

    monkeypatch.setattr(orchestrator.languages, "detect_language", lambda _p: "python")
    monkeypatch.setattr(orchestrator.languages, "get_backend", lambda _l: _Crashed())
    monkeypatch.setattr(orchestrator.project, "resolve_source_dir", lambda _p: "/proj/root")
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text("def test_x(): pass\n")

    ok, reason = orchestrator._restored_test_file_collects(ctx, "test_foo.py")

    assert ok is False and reason.startswith("the test file did not load (exit=-11)")


def test_restored_test_file_collects_writes_conftest_via_setup_test_env(
    tmp_path, monkeypatch,
):
    """The ``--reuse coverage`` path calls ``backend.setup_test_env`` once
    with (work_tests, source_dir, target_file_path) before the collect-only
    check. Without it work_tests has no conftest.py, pytest cannot put the
    target's package on sys.path, and ``from <project>.X import Y`` raises
    ModuleNotFoundError at collection.
    """
    from types import SimpleNamespace
    calls: list[tuple[str, tuple, dict]] = []

    class _FakeBackend:
        def setup_test_env(self, test_output_dir, source_dir, target_file_path):
            calls.append((
                "setup_test_env",
                (test_output_dir, source_dir, target_file_path),
                {},
            ))

        def run_test_files(self, paths, source_dir, **kw):
            calls.append(("run_test_files", (paths, source_dir), kw))
            return SimpleNamespace(
                ok=True, returncode=0, timed_out=False,
                collected=3, passed=3, failed=0, errors=0,
                output="", line_coverage=None, branch_coverage=None,
                elapsed_s=0.1,
            )

    monkeypatch.setattr(
        orchestrator.languages, "detect_language", lambda _p: "python",
    )
    monkeypatch.setattr(
        orchestrator.languages, "get_backend", lambda _l: _FakeBackend(),
    )
    monkeypatch.setattr(
        orchestrator.project, "resolve_source_dir",
        lambda _p: "/proj/root",
    )

    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text("def test_x(): pass\n")
    ok, reason = orchestrator._restored_test_file_collects(ctx, "test_foo.py")

    assert ok is True
    assert reason == ""
    # setup_test_env runs first, or the conftest is missing when pytest collects.
    assert [c[0] for c in calls] == ["setup_test_env", "run_test_files"]
    setup_args = calls[0][1]
    assert setup_args == (
        str(ctx.work_tests), "/proj/root", str(ctx.top_file["file_path"]),
    )
    # Both backend calls use the source_dir from ``resolve_source_dir``,
    # not ``ctx.work_tests.parent``, which lies outside the project tree of
    # a Java multi-module project, where the pom would not be found.
    run_args = calls[1][1]
    assert run_args[1] == "/proj/root", (
        f"run_test_files got source_dir={run_args[1]!r}; "
        f"must be the resolved root, not ctx.work_tests.parent"
    )


def test_restored_test_file_collects_fails_when_setup_test_env_raises_not_implemented(
    tmp_path, monkeypatch,
):
    """A backend whose setup_test_env raises NotImplementedError yields a
    clear collect-check failure instead of skipping the call, since restored
    tests cannot be validated without a test environment."""
    from types import SimpleNamespace

    class _StubBackend:
        def setup_test_env(self, *a, **kw):
            raise NotImplementedError("c backend not implemented")

        def run_test_files(self, *a, **kw):  # never reached
            return SimpleNamespace(
                ok=False, returncode=1, timed_out=False,
                collected=0, passed=0, failed=0, errors=0,
                output="", line_coverage=None, branch_coverage=None,
                elapsed_s=0.0,
            )

    monkeypatch.setattr(
        orchestrator.languages, "detect_language", lambda _p: "c",
    )
    monkeypatch.setattr(
        orchestrator.languages, "get_backend", lambda _l: _StubBackend(),
    )
    monkeypatch.setattr(
        orchestrator.project, "resolve_source_dir", lambda _p: "/proj",
    )

    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text("# stub")
    ok, reason = orchestrator._restored_test_file_collects(ctx, "test_foo.py")
    assert ok is False
    assert "setup_test_env not implemented" in reason


def test_phase_mutation_skips_when_no_coverage_file(monkeypatch, tmp_path):
    """Skip mutation and record its reason when no usable coverage file exists."""
    ctx = _make_ctx(tmp_path)
    counts = _patch_agent(
        monkeypatch, orchestrator.mutation_agent,
        set_mutation_session=lambda *a, **kw: None,
        generate_mutation_tests=lambda *a, **kw: None,
    )
    result = orchestrator.TestPipelineResult()  # no test_file_name
    out = orchestrator._phase_mutation(ctx, result, session=None)
    assert out is None
    assert result.phase_outcomes["mutation"] == "skipped_no_coverage_file"
    assert counts["init"] == 0
    assert counts["generate_mutation_tests"] == 0


def test_phase_mutation_records_failed_on_agent_error(monkeypatch, tmp_path):
    """A non-retryable provider error (an out-of-credit 402) ends the loop with
    status agent_error. As for the inconclusive branch and the coverage phase,
    the phase records the target 'failed', so the degraded score is not read
    as a completed phase and an automatic rerun does not skip the target."""
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text(
        "def test_a():\n    assert True\n", encoding="utf-8"
    )
    fake_result = _mutation_result(
        "test_foo.py", mutation_score=61.0, status="agent_error",
        error=(
            "non-retryable error: PermissionDeniedError: "
            "Error code: 402 - Insufficient Balance"
        ),
    )
    _patch_agent(
        monkeypatch, orchestrator.mutation_agent,
        set_mutation_session=lambda *a, **kw: None,
        generate_mutation_tests=lambda *a, **kw: fake_result,
    )
    result = orchestrator.TestPipelineResult(test_file_name="test_foo.py")
    out = orchestrator._phase_mutation(ctx, result, session=None)
    assert out is fake_result
    assert result.phase_outcomes["mutation"] == "failed"
    detail = result.phase_outcomes["mutation_detail"]
    assert "agent_error" in detail
    assert "402" in detail


def test_coverage_boundary_fails_closed_without_an_engine(
    monkeypatch, tmp_path,
):
    """With consolidation mutation on, a boundary whose engine cannot start
    is a failed target, never a silent degrade to a coverage-only cover."""
    from contextlib import contextmanager
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text(
        "def test_a():\n    assert True\n", encoding="utf-8")
    result = orchestrator.TestPipelineResult()
    result.test_file_name = "test_foo.py"
    inherited = "def test_z():\n    assert True\n"
    starter_archive = orchestrator._archive_dir(ctx, "starter")
    starter_archive.mkdir(parents=True, exist_ok=True)
    (starter_archive / "consolidation_record.json").write_text(
        json.dumps({
            "selected_source_sha256": hashlib.sha256(
                inherited.encode("utf-8")
            ).hexdigest(),
            "execution_name": "test_foo.py",
        }),
        encoding="utf-8",
    )

    @contextmanager
    def _dead_runner_cm(**kw):
        yield None

    monkeypatch.setattr(
        orchestrator.config_module, "consolidation_measure_mutation", lambda: True)
    monkeypatch.setattr(
        orchestrator.measurement, "union_mutation_runner", _dead_runner_cm)
    monkeypatch.setattr(
        orchestrator.languages, "get_backend", lambda lang: SimpleNamespace())
    with pytest.raises(RuntimeError, match=(
            "the coverage phase end needs kill evidence .*engine could not "
            "start, so this target fails rather than being reduced")):
        orchestrator._consolidate_coverage(
            ctx, result, baseline_content=inherited)


def test_mutation_boundary_fails_closed_without_an_engine(
    monkeypatch, tmp_path,
):
    from contextlib import contextmanager
    ctx = _make_ctx(tmp_path)
    inherited = "def test_a():\n    assert True\n"
    (ctx.work_tests / "test_foo.py").write_text(
        inherited + "\ndef test_m():\n    assert True\n", encoding="utf-8")
    result = orchestrator.TestPipelineResult(test_file_name="test_foo.py")
    result.mutation_inherited_source = inherited
    result.mutation_result = _mutation_result("test_foo.py")
    coverage_archive = orchestrator._archive_dir(ctx, "coverage")
    coverage_archive.mkdir(parents=True, exist_ok=True)
    (coverage_archive / "consolidation_record.json").write_text(
        json.dumps({
            "selected_source_sha256": hashlib.sha256(
                inherited.encode("utf-8")
            ).hexdigest(),
            "execution_name": "test_foo.py",
            "measurement_identity": None,
        }),
        encoding="utf-8",
    )

    @contextmanager
    def _dead_runner_cm(**kw):
        yield None

    monkeypatch.setattr(
        orchestrator.config_module, "consolidation_measure_mutation", lambda: True)
    monkeypatch.setattr(
        orchestrator.measurement, "union_mutation_runner", _dead_runner_cm)
    with pytest.raises(RuntimeError, match="engine could not start"):
        orchestrator._consolidate_mutation(ctx, result)

    durable = orchestrator._archive_dir(ctx, "mutation") / "pre_merge"
    assert (
        durable / "candidate_inherited.py"
    ).read_text(encoding="utf-8") == inherited
    assert (
        durable / "candidate_agent.py"
    ).read_text(encoding="utf-8") == (
        inherited + "\ndef test_m():\n    assert True\n"
    )


def _fake_agent_config(monkeypatch, *, coverage_goal=100, metric="both",
                       mutation_goal=100):
    sections = {
        "test_coverage": {
            "goal": coverage_goal,
            "goal_metric": metric,
        },
        "test_mutation": {"goal": mutation_goal},
    }

    def _get(section, parent=None):
        return dict(sections[section])

    monkeypatch.setattr(orchestrator.config_module, "get_agent_config", _get)


def test_phase_mutation_records_failed_on_inherited_red(monkeypatch, tmp_path):
    """An inherited-red early exit means the phase never ran: the suite that
    was green at the coverage handoff fails in this environment. The phase
    records the target 'failed' so it is flagged for rerun, never read as a
    clean stop."""
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text(
        "def test_a():\n    assert True\n", encoding="utf-8"
    )
    fake_result = _mutation_result(
        "test_foo.py", mutation_score=None,
        status="skipped_inherited_red",
    )
    _patch_agent(
        monkeypatch, orchestrator.mutation_agent,
        set_mutation_session=lambda *a, **kw: None,
        generate_mutation_tests=lambda *a, **kw: fake_result,
    )
    result = orchestrator.TestPipelineResult(test_file_name="test_foo.py")
    out = orchestrator._phase_mutation(ctx, result, session=None)
    assert out is fake_result
    assert result.phase_outcomes["mutation"] == "failed"
    detail = result.phase_outcomes["mutation_detail"]
    assert "inherited suite arrived red" in detail
    assert "a test is flaky or the test environment changed" in detail


def test_phase_mutation_records_failed_on_an_inherited_suite_too_slow(monkeypatch, tmp_path):
    """A suite that ran longer than its time limit before any edit leaves a
    phase that never measured anything: recorded failed with its reason,
    never read as a stop before the goal."""
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text("def test_a():\n    assert True\n", encoding="utf-8")
    fake_result = _mutation_result("test_foo.py", mutation_score=None, status="skipped_inherited_slow")
    _patch_agent(
        monkeypatch, orchestrator.mutation_agent,
        set_mutation_session=lambda *a, **kw: None,
        generate_mutation_tests=lambda *a, **kw: fake_result,
    )
    result = orchestrator.TestPipelineResult(test_file_name="test_foo.py")
    assert orchestrator._phase_mutation(ctx, result, session=None) is fake_result
    assert result.phase_outcomes["mutation"] == "failed"
    assert "ran longer than its time limit" in result.phase_outcomes["mutation_detail"]


def test_phase_mutation_restores_coverage_file_on_agent_exception(
    monkeypatch, tmp_path,
):
    """The restore runs on the failure path too: an agent crash must not
    leave the working directory without its coverage file, which the
    mandatory measurement in finishing needs."""
    ctx = _make_ctx(tmp_path)
    cov_path = ctx.work_tests / "test_foo.py"
    cov_path.parent.mkdir(parents=True, exist_ok=True)
    cov_bytes = "def test_cov():\n    assert True\n"
    cov_path.write_text(cov_bytes, encoding="utf-8")
    _patch_agent(
        monkeypatch, orchestrator.mutation_agent,
        set_mutation_session=lambda *a, **kw: None,
        generate_mutation_tests=lambda *a, **kw: (_ for _ in ()).throw(
            ValueError("boom"),
        ),
    )
    result = orchestrator.TestPipelineResult(test_file_name="test_foo.py")
    out = orchestrator._phase_mutation(ctx, result, session=None)
    assert out is None
    assert result.phase_outcomes["mutation"] == "failed"
    assert cov_path.exists()
    assert cov_path.read_text(encoding="utf-8") == cov_bytes


def test_phase_mutation_catches_exception_and_records_failed(
    monkeypatch, tmp_path,
):
    """An agent exception records 'failed' with the exception class and
    message in the detail. _phase_mutation does not raise it."""
    ctx = _make_ctx(tmp_path)
    _patch_agent(
        monkeypatch, orchestrator.mutation_agent,
        set_mutation_session=lambda *a, **kw: None,
        generate_mutation_tests=lambda *a, **kw: (_ for _ in ()).throw(
            ValueError("bad config"),
        ),
    )
    result = orchestrator.TestPipelineResult(test_file_name="test_foo.py")
    out = orchestrator._phase_mutation(ctx, result, session=None)
    assert out is None
    assert result.phase_outcomes["mutation"] == "failed"
    assert "ValueError" in result.phase_outcomes["mutation_detail"]
    assert "bad config" in result.phase_outcomes["mutation_detail"]


def test_phase_finishing_skips_without_coverage_file(monkeypatch, tmp_path):
    """Finishing skips when no usable coverage file exists."""
    ctx = _make_ctx(tmp_path)
    counts = _patch_agent(
        monkeypatch, orchestrator.measurement,
        measure_combined=lambda *a, **kw: None,
    )
    result = orchestrator.TestPipelineResult()
    out = orchestrator._phase_finishing(ctx, result)
    assert out is None
    assert result.phase_outcomes["finishing"] == "skipped_no_coverage_file"
    assert counts["init"] == 0
    assert counts["measure_combined"] == 0


def test_phase_finishing_records_failed_on_exception(monkeypatch, tmp_path):
    """An exception in measure_combined records 'failed' with a detail and
    is not raised."""
    ctx = _make_ctx(tmp_path)
    _patch_agent(
        monkeypatch, orchestrator.measurement,
        measure_combined=lambda *a, **kw: (_ for _ in ()).throw(
            RuntimeError("subprocess died"),
        ),
    )
    result = orchestrator.TestPipelineResult(test_file_name="test_foo.py")
    out = orchestrator._phase_finishing(ctx, result)
    assert out is None
    assert result.phase_outcomes["finishing"] == "failed"
    assert "RuntimeError" in result.phase_outcomes["finishing_detail"]


def test_open_mutation_session_yields_none_when_mutation_skipped(tmp_path):
    """A skipped mutation phase opens no session. The optional re-measurement
    in finishing, off by default, opens its own pool."""
    ctx = _make_ctx(
        tmp_path,
        reuse=orchestrator.ReuseFlags(starter=True, mutation=True),
    )
    with orchestrator._open_mutation_session(ctx) as session:
        assert session is None


def _install_mutation_runner(monkeypatch, open_session):
    """Route the orchestrator's mutation runner to ``open_session``
    (``workers -> session``). The suite directories each opening declares are
    kept in ``runner.declared``."""
    runner = SimpleNamespace(declared=[])

    def opened(workers, *, suite_dirs):
        runner.declared.append(tuple(suite_dirs))
        return open_session(workers)

    runner.open = opened
    monkeypatch.setattr(
        orchestrator.languages, "get_backend",
        lambda lang: SimpleNamespace(
            mutation_runner=lambda config, **kw: runner),
    )
    return runner


def test_open_mutation_session_yields_session_when_setup_ok(
    monkeypatch, tmp_path,
):
    """An opened session is yielded and closed on exit."""
    ctx = _make_ctx(tmp_path)
    cm_log: list[str] = []

    class FakeSession:
        label = "real-session"

        def close(self):
            cm_log.append("exit")

    def open_session(workers):
        cm_log.append("enter")
        return FakeSession()

    runner = _install_mutation_runner(monkeypatch, open_session)

    with orchestrator._open_mutation_session(ctx) as session:
        assert session is not None
        assert session.label == "real-session"
    assert cm_log == ["enter", "exit"]
    # The agents' suites lie in the work tree's tests.
    assert runner.declared == [(ctx.work_tests,)]


def test_open_mutation_session_exit_propagates_teardown_errors(
    monkeypatch, tmp_path,
):
    """A successful phase body cannot turn unresolved teardown into success."""
    ctx = _make_ctx(tmp_path)
    error = RuntimeError("teardown failed")

    class FakeSession:
        label = "x"

        def close(self):
            raise error

    pooled = FakeSession()
    _install_mutation_runner(monkeypatch, lambda workers: pooled)
    with pytest.raises(RuntimeError) as caught:
        with orchestrator._open_mutation_session(ctx) as session:
            assert session is pooled
    assert caught.value is error
    assert error.mutation_cleanup_handle is pooled
    assert error.mutation_cleanup_error is error


def _retained_mutation_error(kind):
    owner = SimpleNamespace(close=lambda: None)
    if kind == "native":
        from langadapter.mutation import MutationCleanupError

        return MutationCleanupError(owner, "native cleanup unresolved"), owner
    error = OSError("cleanup owner retained")
    markers = {
        "adapter": "mutation_cleanup_handle",
        "prepared_setup": "preparation_setup_error",
        "prepared_close": "preparation_cleanup_error",
    }
    setattr(error, markers[kind], owner)
    return error, owner


@pytest.mark.parametrize("kind", ["native", "adapter", "prepared_setup", "prepared_close"])
def test_open_mutation_session_refuses_retained_setup_before_fallback(
    monkeypatch, tmp_path, kind,
):
    ctx = _make_ctx(tmp_path)
    error, owner = _retained_mutation_error(kind)
    cause = ValueError("original setup cause")
    error.__cause__ = cause
    events = []

    def open_session(workers):
        events.append(("open", workers))
        raise error

    _install_mutation_runner(monkeypatch, open_session)
    monkeypatch.setattr(orchestrator.measurement, "mutation_config", lambda: {"engine_workers": 3})
    ctx = replace(ctx, log_print=lambda *a, **kw: events.append("fallback"))
    before = ctx.canonical_path.read_bytes()
    with pytest.raises(type(error)) as caught:
        with orchestrator._open_mutation_session(ctx):
            pytest.fail("Retained setup must not enter per-call fallback")
    assert caught.value is error
    assert caught.value.__cause__ is cause
    assert events == [("open", 3)]
    assert ctx.canonical_path.read_bytes() == before
    if kind == "native":
        assert error.cleanup_handle is owner
    else:
        assert owner in vars(error).values()


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
def test_open_mutation_session_setup_cancellation_and_clean_fallback(
    monkeypatch, tmp_path, error_type,
):
    ctx = _make_ctx(tmp_path)
    error = error_type("unowned setup failure")

    def open_session(workers):
        raise error

    _install_mutation_runner(monkeypatch, open_session)
    if isinstance(error, Exception):
        with orchestrator._open_mutation_session(ctx) as session:
            assert session is None
    else:
        with pytest.raises(error_type) as caught:
            with orchestrator._open_mutation_session(ctx):
                pytest.fail("Setup cancellation must propagate")
        assert caught.value is error


@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, SystemExit])
@pytest.mark.parametrize("close_fails", [False, True])
def test_open_mutation_session_body_keeps_primary_and_retryable_owner(
    monkeypatch, tmp_path, error_type, close_fails,
):
    ctx = _make_ctx(tmp_path)
    primary = error_type("phase failure")
    cause = ValueError("original cause")
    primary.__cause__ = cause
    cleanup = OSError("close failed")

    class ReplaySession:
        fail = close_fails
        closes = 0

        def close(self):
            self.closes += 1
            if self.fail:
                raise cleanup

    pooled = ReplaySession()
    _install_mutation_runner(monkeypatch, lambda workers: pooled)
    with pytest.raises(error_type) as caught:
        with orchestrator._open_mutation_session(ctx) as session:
            assert session is pooled
            raise primary
    assert caught.value is primary
    assert primary.__cause__ is cause
    assert pooled.closes == 1
    if close_fails:
        assert primary.mutation_cleanup_handle is pooled
        assert primary.mutation_cleanup_error is cleanup
        pooled.fail = False
        primary.mutation_cleanup_handle.close()
        assert pooled.closes == 2
    else:
        assert not hasattr(primary, "mutation_cleanup_handle")


@pytest.mark.parametrize("kind", ["native", "prepared_setup", "prepared_close"])
def test_phase_mutation_retained_failure_never_restores_inherited_source(
    monkeypatch, tmp_path, kind,
):
    ctx = _make_ctx(tmp_path)
    target = ctx.work_tests / "test_foo.py"
    inherited = "def test_cov():\n    assert True\n"
    pending = "def test_partial():\n    assert False\n"
    target.write_text(inherited, encoding="utf-8")
    error, owner = _retained_mutation_error(kind)
    sessions = []

    def generate(*args, **kwargs):
        target.write_text(pending, encoding="utf-8")
        raise error

    _patch_agent(
        monkeypatch, orchestrator.mutation_agent,
        set_mutation_session=sessions.append, generate_mutation_tests=generate,
    )
    result = orchestrator.TestPipelineResult(test_file_name="test_foo.py")
    with pytest.raises(type(error)) as caught:
        orchestrator._phase_mutation(ctx, result, session=owner)
    assert caught.value is error
    assert sessions == [owner, None]
    assert result.mutation_inherited_source == inherited
    assert target.read_text(encoding="utf-8") == pending
    assert "mutation" not in result.phase_outcomes


@pytest.mark.parametrize("retained", [False, True])
def test_source_integrity_exit_defers_only_retained_cleanup(
    monkeypatch, tmp_path, retained,
):
    ctx = _make_ctx(tmp_path)
    canonical = tmp_path / "canonical.py"
    canonical.write_bytes(ctx.canonical_path.read_bytes())
    ctx = replace(ctx, canonical_path=canonical,
                  top_file={**ctx.top_file, "canonical_path": str(canonical)})
    work = Path(ctx.top_file["file_path"])
    error = _retained_mutation_error("adapter")[0] if retained else ValueError("method failure")
    messages = []
    monkeypatch.setattr("testuary._progress.diagnostic_print", lambda *a, **kw: messages.append(a))
    with pytest.raises(type(error)) as caught:
        with orchestrator._source_integrity_guard(ctx, "mutation"):
            work.write_text("pending source", encoding="utf-8")
            raise error
    assert caught.value is error
    if retained:
        assert work.read_text(encoding="utf-8") == "pending source"
        assert not messages
    else:
        assert work.read_bytes() == canonical.read_bytes()
        assert len(messages) == 1


@pytest.mark.parametrize("scenario", [
    "setup_retained", "body_retained", "body_cancel_close_failed",
    "close_failed", "clean", "clean_fallback",
])
def test_run_target_phases_pooled_cleanup_precedes_later_work(
    monkeypatch, tmp_path, scenario,
):
    ctx = _make_ctx(tmp_path)
    _patch_starter_handoff(monkeypatch, ctx)
    delivered = ctx.work_tests / "test_foo.py"
    inherited = "def test_cov():\n    assert True\n"
    revised = inherited + "\ndef test_mut():\n    assert True\n"
    delivered.write_text(inherited, encoding="utf-8")
    canonical = tmp_path / "canonical.py"
    canonical_bytes = ctx.canonical_path.read_bytes()
    canonical.write_bytes(canonical_bytes)
    ctx = replace(ctx, canonical_path=canonical,
                  top_file={**ctx.top_file, "canonical_path": str(canonical)})
    work = Path(ctx.top_file["file_path"])
    monkeypatch.setattr(orchestrator, "_dump_resolved_preset", lambda ctx: None)
    _patch_agent(
        monkeypatch, orchestrator.coverage_agent,
        generate_coverage_tests=lambda *a, **kw: _coverage_result(),
    )
    error, owner = _retained_mutation_error("native")
    cleanup = OSError("pool close incomplete")
    if scenario == "body_cancel_close_failed":
        error = KeyboardInterrupt("phase cancelled")
        error.cleanup_handle = owner
    elif scenario == "close_failed":
        error = cleanup
    cause = ValueError("original failure cause")
    error.__cause__ = cause
    events = []
    bound = []
    checkpoints = []

    class ReplaySession:
        fail = scenario in {"body_cancel_close_failed", "close_failed"}

        def close(self):
            events.append("close")
            if self.fail:
                raise cleanup

    pooled = ReplaySession()

    def open_session(workers):
        events.append("open")
        if scenario == "setup_retained":
            raise error
        if scenario == "clean_fallback":
            raise orchestrator.mutation.MutationSetupError("setup fully cleaned")
        return pooled

    # Patch the cached adapter's class: an instance patch outlives the test.
    backend_type = type(orchestrator.languages.get_backend("python"))
    monkeypatch.setattr(backend_type, "mutation_runner",
                        lambda self, *a, **kw: SimpleNamespace(
                            open=lambda workers, *, suite_dirs: open_session(workers)))

    def generate(**kwargs):
        events.append("generate")
        delivered.write_text(revised, encoding="utf-8")
        if scenario in {"body_retained", "body_cancel_close_failed"}:
            work.write_text("pending source", encoding="utf-8")
            raise error
        return _mutation_result("test_foo.py")

    _patch_agent(
        monkeypatch, orchestrator.mutation_agent,
        set_mutation_session=bound.append, generate_mutation_tests=generate,
    )

    def consolidate(*args):
        events.append("consolidate")
        return SimpleNamespace(winner="agent")

    def measure_finishing(*args, **kwargs):
        events.append("finishing")
        return _metrics_result()

    def attach_finishing(ctx, result, combined):
        events.append("finalize")
        combined.finishing = FinishingMetrics(mutation_score_full=75)

    monkeypatch.setattr(orchestrator, "_consolidate_mutation", consolidate)
    _patch_agent(monkeypatch, orchestrator.measurement, measure_combined=measure_finishing)
    monkeypatch.setattr(orchestrator, "_attach_finishing", attach_finishing)
    monkeypatch.setattr(orchestrator, "_checkpoint_phase", lambda c, r, phase: checkpoints.append(phase))
    if scenario.startswith("clean"):
        result = orchestrator.run_target_phases(ctx)
        assert result.mutation_result is not None
        assert events == (["open", "generate", "close"] if scenario == "clean" else ["open", "generate"]) + ["consolidate", "finishing", "finalize"]
        assert "mutation" in checkpoints
    else:
        with pytest.raises(type(error)) as caught:
            orchestrator.run_target_phases(ctx)
        assert caught.value is error
        assert error.__cause__ is cause
        assert not {"consolidate", "finishing", "finalize"}.intersection(events)
        assert "mutation" not in checkpoints
        if scenario == "setup_retained":
            assert events == ["open"]
            assert not bound
            assert delivered.read_text(encoding="utf-8") == inherited
        else:
            assert events == ["open", "generate", "close"]
            assert bound == [pooled, None]
            assert delivered.read_text(encoding="utf-8") == revised
        if scenario in {"setup_retained", "body_retained"}:
            assert error.cleanup_handle is owner
        else:
            assert error.mutation_cleanup_handle is pooled
            assert error.mutation_cleanup_error is cleanup
            pooled.fail = False
            error.mutation_cleanup_handle.close()
            assert events[-1] == "close"
        if scenario in {"body_retained", "body_cancel_close_failed"}:
            assert work.read_text(encoding="utf-8") == "pending source"
    assert canonical.read_bytes() == canonical_bytes


def test_mutation_config_merges_parent_test_coverage_section(monkeypatch):
    """``_mutation_config`` reads [mutation] and fills missing keys
    from [coverage]."""
    def fake_load(section):
        if section == "test_mutation":
            return {"a": "1", "max_per_mutant_timeout_factor": "1.5"}
        return {"a": "ignored", "b": "from_parent",
                "suite_run_timeout_seconds": "600"}
    monkeypatch.setattr(orchestrator.config_module, "load_config", fake_load)
    cfg = orchestrator.measurement.mutation_config()
    assert cfg["a"] == "1"          # child wins
    assert cfg["b"] == "from_parent"  # filled from parent
    # The per-mutant ceiling derives from the merged settings (child factor,
    # parent suite timeout). A partial section fails instead.
    assert cfg["mutation_max_timeout"] == 900


def test_run_target_phases_bails_when_coverage_phase_fails(monkeypatch, tmp_path):
    """When the coverage phase returns a file name without a file on disk,
    the later phases are not invoked."""
    ctx = _make_ctx(tmp_path)
    _patch_starter_handoff(monkeypatch, ctx)
    # No test file is written, so coverage fails with "file missing on disk".
    _patch_agent(
        monkeypatch, orchestrator.coverage_agent,
        generate_coverage_tests=lambda *a, **kw: _coverage_result(
            "test_foo.py", status="ok", line_coverage=0.0,
        ),
    )
    mut_counts = _patch_agent(
        monkeypatch, orchestrator.mutation_agent,
        set_mutation_session=lambda *a, **kw: None,
        generate_mutation_tests=lambda *a, **kw: None,
    )
    finishing_counts = _patch_agent(
        monkeypatch, orchestrator.measurement,
        measure_combined=lambda *a, **kw: None,
    )
    result = orchestrator.run_target_phases(ctx)
    assert result.coverage_phase_failed is True
    # Downstream phases were never invoked after coverage failed.
    assert mut_counts["generate_mutation_tests"] == 0
    assert finishing_counts["measure_combined"] == 0


def _green_pipeline(monkeypatch, ctx):
    """Stub the agents green, suppress the mutation session and write the
    on-disk artifacts the success path expects."""
    _patch_starter_handoff(monkeypatch, ctx)
    # Real test_* functions so the merge's green check (a real pytest run on the
    # merged suite) collects and passes them: a green pipeline must merge green.
    (ctx.work_tests / "test_foo.py").write_text("def test_t(): pass\n")
    _patch_agent(
        monkeypatch, orchestrator.coverage_agent,
        generate_coverage_tests=lambda *a, **kw: _coverage_result(
            "test_foo.py", status="ok", line_coverage=80.0),
    )
    def _generate_mutation(*args, **kwargs):
        delivered = ctx.work_tests / "test_foo.py"
        delivered.write_text(
            delivered.read_text(encoding="utf-8")
            + "\ndef test_tm(): pass\n",
            encoding="utf-8",
        )
        return _mutation_result("test_foo.py", mutation_score=75.0)

    _patch_agent(
        monkeypatch, orchestrator.mutation_agent,
        set_mutation_session=lambda *a, **kw: None,
        generate_mutation_tests=_generate_mutation,
    )
    _patch_agent(
        monkeypatch, orchestrator.measurement,
        measure_combined=lambda *a, **kw: _metrics_result(
            coverage_final_line=92.0, coverage_final_branch=85.0),
    )

    def fake_consolidate_mutation(_ctx, result):
        coverage_path = _ctx.work_tests / result.test_file_name
        inherited_src = result.mutation_inherited_source
        mutation_src = coverage_path.read_text(encoding="utf-8")
        pre_dir = _ctx.work_tests.parent / "_pre_merge"
        pre_dir.mkdir(parents=True, exist_ok=True)
        pre_coverage = pre_dir / "candidate_inherited.py"
        pre_mutation = pre_dir / "candidate_agent.py"
        pre_coverage.write_text(inherited_src, encoding="utf-8")
        pre_mutation.write_text(mutation_src, encoding="utf-8")
        return orchestrator.measurement.MergeResult(
            merged_path=str(coverage_path),
            pre_merge_coverage_path=str(pre_coverage),
            pre_merge_mutation_path=str(pre_mutation),
            union_tests=2,
            delivered_tests=2,
        )

    monkeypatch.setattr(
        orchestrator, "_consolidate_mutation", fake_consolidate_mutation
    )

    @contextmanager
    def fake_session(_ctx):
        yield None
    monkeypatch.setattr(orchestrator, "_open_mutation_session", fake_session)


def test_failed_requested_phase_is_not_reported_as_stopped(
    monkeypatch, tmp_path,
):
    ctx = _make_ctx(tmp_path, stop_after_phase="mutation")
    _green_pipeline(monkeypatch, ctx)
    _patch_agent(
        monkeypatch,
        orchestrator.mutation_agent,
        generate_mutation_tests=lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("boom")
        ),
    )

    result = orchestrator.run_target_phases(ctx)

    assert result.phase_outcomes["mutation"] == "failed"
    assert result.stopped_after_phase is None


def test_restored_coverage_fails_before_collection_on_stale_validity(
    monkeypatch, tmp_path,
):
    ctx = _make_ctx(tmp_path)
    result = orchestrator.TestPipelineResult()
    (ctx.work_tests / "test_foo.py").write_text(
        "def test_ok():\n    assert True\n", encoding="utf-8",
    )
    monkeypatch.setattr(
        orchestrator,
        "_verify_restored_phase_validity",
        lambda *_args, **_kwargs: (False, "stale policy"),
    )
    monkeypatch.setattr(
        orchestrator,
        "_restored_test_file_collects",
        lambda *_args, **_kwargs: pytest.fail(
            "collection must not run before validity succeeds"
        ),
    )

    restored = orchestrator._restore_coverage_test_file_from_disk(ctx, result)

    assert restored is None
    assert result.phase_outcomes["coverage"] == "failed"
    assert "stale policy" in result.phase_outcomes["coverage_detail"]


def _ctx_with_canonical(tmp_path: Path) -> orchestrator.TargetPhasesContext:
    """Context whose ``file_path`` is in the target's copy and whose
    ``canonical_path`` is the canonical source, as __main__.py sets them."""
    ctx = _make_ctx(tmp_path)
    # The target's copy lies in work_dir, the canonical source in a sibling.
    canonical = tmp_path / "canonical" / "src.py"
    canonical.parent.mkdir(parents=True, exist_ok=True)
    canonical.write_text("def f(): return 1\n", encoding="utf-8")
    work = ctx.work_tests.parent / "src.py"
    work.write_bytes(canonical.read_bytes())
    # The copy in top_file and the canonical source in canonical_path, as
    # the orchestrator sets them.
    ctx.top_file["file_path"] = str(work)
    ctx.top_file["canonical_path"] = str(canonical)
    return replace(ctx, canonical_path=canonical)


def test_integrity_guard_restores_on_entry_mismatch(tmp_path, capsys):
    """Work-dir source that diverged from canonical before the phase (for
    example left broken by the previous phase) is repaired at entry, so the
    body runs against the clean version."""
    ctx = _ctx_with_canonical(tmp_path)
    canon = Path(ctx.top_file["canonical_path"])
    work = Path(ctx.top_file["file_path"])
    work.write_bytes(b"corrupted = lo\n")
    assert work.read_bytes() != canon.read_bytes()
    with orchestrator._source_integrity_guard(ctx, "mutation"):
        # Inside the body the file is already restored.
        assert work.read_bytes() == canon.read_bytes()
    cap = capsys.readouterr()
    assert "[CRITICAL]" in cap.out
    assert "mutation entry" in cap.out
    assert f"the work copy {work} differs from the original" in cap.out


def test_integrity_guard_restores_on_exit_mismatch(tmp_path, capsys):
    """When the phase body modifies the copy's source (a Cosmic Ray
    escape, an orphan worker), the guard restores it at exit and reports
    it as critical."""
    ctx = _ctx_with_canonical(tmp_path)
    canon = Path(ctx.top_file["canonical_path"])
    work = Path(ctx.top_file["file_path"])
    with orchestrator._source_integrity_guard(ctx, "finishing"):
        # Simulate a worker that escaped its prepared copy.
        work.write_bytes(b"def f(): return 999  # mutated\n")
    # On exit the guard restored from canonical.
    assert work.read_bytes() == canon.read_bytes()
    cap = capsys.readouterr()
    assert "[CRITICAL]" in cap.out
    assert "finishing exit" in cap.out


def test_integrity_guard_is_noop_when_canonical_missing(tmp_path, capsys):
    """Without the canonical source on this host, as for a container
    run's original, the guard skips: a change in the body stays, and
    nothing is reported."""
    ctx = _make_ctx(tmp_path, canonical_path=tmp_path / "absent.py")
    work = Path(ctx.top_file["file_path"])
    with orchestrator._source_integrity_guard(ctx, "coverage"):
        work.write_bytes(b"changed\n")
    assert work.read_bytes() == b"changed\n"
    assert "[CRITICAL]" not in capsys.readouterr().out


def _consolidate_cfg(section, **kw):
    if section == "test_coverage":
        return {
            "suite_run_timeout_seconds": 300,
            "phase_end_evidence_reuse": True,
            "reduction_solver": "exact",
            "reduction_solver_time_limit_seconds": 120.0,
            "reduction_min_test_time_ms": 0.001,
            "reduction_solver_random_seed": 1,
            "reduction_tie_break": "name_order",
        }
    return {}


def _drive_consolidate(
    tmp_path,
    monkeypatch,
    *,
    remeasured_line,
    remeasured_branch,
    reduced_km=None,
    agent_line=100.0,
    agent_branch=90.0,
    candidate2_line=None,
    candidate2_branch=None,
    candidate2_km=None,
    baseline_km=None,
    baseline_line=None,
    baseline_branch=None,
    baseline_stamp=None,
    phase_end=None,
):
    """Run _consolidate_coverage with a mocked consolidate_phase_end whose merged
    reduced suite measures at the given line/branch. The real
    pick_phase_result / scored_from_phase then run, so this exercises the
    keep-best boundary decision. Baseline metrics are unknown unless a test
    supplies them explicitly."""
    from testuary.consolidation import PhaseConsolidation
    default_km = {
        "mutants": [{"id": "m0", "status": "survived"}]
    }
    if reduced_km is None:
        reduced_km = default_km
    if candidate2_km is None:
        candidate2_km = default_km
    if baseline_km is None:
        baseline_km = default_km
    # Persist the identity using the same effective configuration that the
    # mocked measurement and later phase boundary use.
    monkeypatch.setattr(orchestrator.config_module, "get_agent_config", _consolidate_cfg)
    ctx = _make_ctx(tmp_path)
    cov_src = "def test_a():\n    assert True\ndef test_b():\n    assert True\n"
    (ctx.work_tests / "test_foo.py").write_text(cov_src, encoding="utf-8")
    result = orchestrator.TestPipelineResult()
    result.test_file_name = "test_foo.py"
    result.test_result = SimpleNamespace(
        line_coverage=agent_line,
        branch_coverage=agent_branch,
    )
    result.starter_result = SimpleNamespace(
        baseline_line_coverage=baseline_line,
        baseline_branch_coverage=baseline_branch,
        baseline_suite_seconds=1.0,
        selected_tests=1,
    )
    if baseline_km is not None:
        starter_archive = (
            ctx.results_dir / ctx.workspace / "r0" / "archive" / "starter"
        )
        starter_archive.mkdir(parents=True, exist_ok=True)
        baseline_source = "def test_z():\n    assert True\n"
        identity = orchestrator.mutation_agent.mutation_measurement_identity(
            target_file_path=str(ctx.top_file["file_path"]),
            language="python",
            execution_name="test_foo.py",
            config={},
            attribution="all",
        )
        baseline_sha256 = hashlib.sha256(
            baseline_source.encode("utf-8")
        ).hexdigest()
        # What the starter phase persisted: the delivered suite's matrix under the
        # canonical name, stamped with its bytes, beside a reduction that did
        # was not delivered and killed nothing. Reading the reduction as the delivered
        # suite would score the baseline at zero.
        (starter_archive / "selected_kill_matrix.json").write_text(
            json.dumps({**baseline_km,
                        "source_sha256": baseline_stamp or baseline_sha256,
                        "execution_name": "test_foo.py"}),
            encoding="utf-8",
        )
        (starter_archive / "reduced_kill_matrix.json").write_text(
            json.dumps({
                "mutants": [
                    {**mutant, "status": "survived"}
                    for mutant in baseline_km["mutants"]
                ],
                "source_sha256": hashlib.sha256(
                    b"the reduction").hexdigest(),
                "execution_name": "test_foo.py",
            }),
            encoding="utf-8",
        )
        (starter_archive / "consolidation_record.json").write_text(
            json.dumps({
                "selected_source_sha256": baseline_sha256,
                "execution_name": "test_foo.py",
                "measurement_identity": identity,
            }),
            encoding="utf-8",
        )

    # The merged and the reduced suite are separate candidates at a phase end,
    # so the merged one carries its own measurement here as a real
    # consolidation leaves it: the coverage recorded when the merged file was
    # measured, and the matrix the reduced one was derived from.
    phase = PhaseConsolidation(
        merged_source="def test_consolidated():\n    assert True\n",
        reduced_source="def test_consolidated():\n    assert True\n",
        reduction_valid=True,
        line=remeasured_line, branch=remeasured_branch, runtime_ms=10.0,
        union_line=remeasured_line, union_branch=remeasured_branch,
        merged_km=reduced_km,
        reduced_km=reduced_km, selected_tests=1, pool_tests=2)
    phase.candidate2_source = cov_src
    phase.candidate2_execution_name = "test_foo.py"
    phase.candidate2_green = True
    phase.candidate2_coverage = {
        "line": (
            candidate2_line if candidate2_line is not None else agent_line
        ),
        "branch": (
            candidate2_branch
            if candidate2_branch is not None else agent_branch
        ),
        "covered_statement_lines": [1],
        "covered_branch_arcs": [],
    }
    phase.candidate2_km = candidate2_km

    monkeypatch.setattr(orchestrator.config_module, "get_agent_config", _consolidate_cfg)
    # These are mutation-mode boundaries with kill matrices as evidence, so
    # the mode stays on and only the engine start is faked. The runner is
    # never called because consolidate_phase_end is stubbed.
    from contextlib import contextmanager

    @contextmanager
    def _fake_runner_cm(**kw):
        yield lambda d: None

    monkeypatch.setattr(
        orchestrator.config_module, "consolidation_measure_mutation", lambda: True)
    monkeypatch.setattr(
        orchestrator.measurement, "union_mutation_runner", _fake_runner_cm)
    monkeypatch.setattr(
        orchestrator.config_module, "consolidation_mutation_method", lambda: "both")
    monkeypatch.setattr(
        "testuary.consolidation.consolidate_phase_end",
        phase_end or (lambda *a, **k: phase))
    monkeypatch.setattr(
        "testuary.consolidation.benchmark_tied_candidates",
        lambda *args, **kwargs: (
            lambda candidate: {
                "agent": 10.0,
                "baseline": 20.0,
                "merged": 30.0,
            }.get(candidate.label, 5.0)
        ),
    )
    backend = orchestrator.languages.get_backend("python")
    monkeypatch.setattr(
        orchestrator.languages, "get_backend", lambda lang: backend)

    orchestrator._consolidate_coverage(
        ctx, result, baseline_content="def test_z():\n    assert True\n"
    )
    return ctx, result, cov_src, phase


def test_consolidate_coverage_fails_on_an_unimplemented_operation(
        tmp_path, monkeypatch):
    """A merge the language does not implement raises at the coverage
    boundary instead of returning without a consolidation record."""
    def unimplemented(*_args, **_kwargs):
        raise NotImplementedError("merging is not implemented for toy")

    with pytest.raises(RuntimeError, match="coverage phase end cannot merge"):
        _drive_consolidate(tmp_path, monkeypatch, remeasured_line=80.0,
                           remeasured_branch=70.0, phase_end=unimplemented)


def test_consolidate_coverage_reverts_when_it_covers_less(tmp_path, monkeypatch):
    """A higher-scoring agent suite wins over a lower-coverage reduction."""
    ctx, result, cov_src, phase = _drive_consolidate(
        tmp_path, monkeypatch, remeasured_line=80.0, remeasured_branch=70.0)
    assert (ctx.work_tests / "test_foo.py").read_text(encoding="utf-8") == cov_src
    assert result.test_result.line_coverage == 100.0  # not overwritten downward
    assert (ctx.work_tests / ".mutation_baseline_matrix.json").exists()
    # The agent's pre-consolidation suite is preserved as a per-target candidate.
    assert (
        ctx.results_dir / ctx.workspace / "r0" / "archive" / "coverage"
        / "candidate_agent.py"
    ).read_text(encoding="utf-8") == cov_src


def test_consolidate_coverage_uses_candidate2_mutation_floor(
    tmp_path, monkeypatch,
):
    """A green coverage merge that masks an agent kill must leave the agent
    standing and persist that standalone matrix for the mutation phase."""
    merged = {
        "mutants": [
            {"id": "m1", "status": "survived"},
            {"id": "m2", "status": "survived"},
        ]
    }
    agent = {
        "mutants": [
            {"id": "m1", "status": "killed"},
            {"id": "m2", "status": "survived"},
        ]
    }
    baseline = {
        "mutants": [
            {"id": "m1", "status": "survived"},
            {"id": "m2", "status": "survived"},
        ]
    }
    ctx, _result, cov_src, _phase = _drive_consolidate(
        tmp_path,
        monkeypatch,
        remeasured_line=100.0,
        remeasured_branch=90.0,
        reduced_km=merged,
        candidate2_line=100.0,
        candidate2_branch=90.0,
        candidate2_km=agent,
        baseline_km=baseline,
    )

    assert (ctx.work_tests / "test_foo.py").read_text(
        encoding="utf-8"
    ) == cov_src
    sidecar = json.loads(
        (ctx.work_tests / ".mutation_baseline_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert sidecar["matrix"] == agent
    # The sidecar is stamped with the delivered bytes' hash, binding the
    # matrix to the exact suite the mutation phase inherits.
    assert sidecar["suite_sha256"] == hashlib.sha256(
        cov_src.encode("utf-8")
    ).hexdigest()
    archive = ctx.results_dir / ctx.workspace / "r0" / "archive" / "coverage"
    persisted = json.loads(
        (archive / "selected_kill_matrix.json").read_text(encoding="utf-8")
    )
    assert persisted["mutants"] == agent["mutants"]
    assert persisted["source_sha256"] == hashlib.sha256(
        cov_src.encode("utf-8")
    ).hexdigest()
    # The masked merge keeps its own matrix under its own name.
    reduction = json.loads(
        (archive / "reduced_kill_matrix.json").read_text(encoding="utf-8")
    )
    assert reduction["mutants"] == merged["mutants"]


def test_consolidate_coverage_selects_previous_phase_on_mutation_score(
    tmp_path, monkeypatch,
):
    """The previous phase wins when it has the highest mutation score."""
    previous = {
        "mutants": [
            {"id": "m1", "status": "killed"},
            {"id": "m2", "status": "survived"},
        ]
    }
    regressed = {
        "mutants": [
            {"id": "m1", "status": "survived"},
            {"id": "m2", "status": "survived"},
        ]
    }
    ctx, result, _cov_src, _phase = _drive_consolidate(
        tmp_path,
        monkeypatch,
        remeasured_line=100.0,
        remeasured_branch=90.0,
        reduced_km=regressed,
        candidate2_line=100.0,
        candidate2_branch=90.0,
        candidate2_km=regressed,
        baseline_km=previous,
        baseline_line=100.0,
        baseline_branch=90.0,
    )

    assert (ctx.work_tests / "test_foo.py").read_text(
        encoding="utf-8"
    ) == "def test_z():\n    assert True\n"
    assert result.consolidation_stats["coverage"]["delivered_from"] == "baseline"
    persisted = json.loads(
        (
            ctx.results_dir
            / ctx.workspace
            / "r0" / "archive"
            / "coverage"
            / "selected_kill_matrix.json"
        ).read_text(encoding="utf-8")
    )
    assert persisted["mutants"] == previous["mutants"]
    assert persisted["source_sha256"] == hashlib.sha256(
        b"def test_z():\n    assert True\n"
    ).hexdigest()


def test_consolidate_coverage_refuses_inherited_evidence_of_other_bytes(
    tmp_path, monkeypatch,
):
    """The starter phase's record names the inherited suite's bytes, but the matrix
    under the canonical name describes other bytes. The boundary does not
    score the inherited candidate with it: that candidate has no mutation
    measurement, so the mutation-first selector cannot choose it, although
    the refused matrix would have made it the winner (compare
    ``test_consolidate_coverage_selects_previous_phase_on_mutation_score``)."""
    previous = {
        "mutants": [
            {"id": "m1", "status": "killed"},
            {"id": "m2", "status": "survived"},
        ]
    }
    regressed = {
        "mutants": [
            {"id": "m1", "status": "survived"},
            {"id": "m2", "status": "survived"},
        ]
    }
    ctx, result, _cov_src, _phase = _drive_consolidate(
        tmp_path,
        monkeypatch,
        remeasured_line=100.0,
        remeasured_branch=90.0,
        reduced_km=regressed,
        candidate2_line=100.0,
        candidate2_branch=90.0,
        candidate2_km=regressed,
        baseline_km=previous,
        baseline_line=100.0,
        baseline_branch=90.0,
        baseline_stamp=hashlib.sha256(b"another suite").hexdigest(),
    )

    assert result.consolidation_stats["coverage"]["delivered_from"] != "baseline"
    record = json.loads(
        (
            ctx.results_dir / ctx.workspace / "r0" / "archive" / "coverage"
            / "consolidation_record.json"
        ).read_text(encoding="utf-8")
    )
    inherited = next(
        candidate
        for candidate in record["selection_audit"]["candidates"]
        if candidate["label"] == "baseline"
    )
    assert inherited["mutation"] is None


def test_consolidate_coverage_audits_identity_loss_without_gating(
    tmp_path, monkeypatch,
):
    """Equal scores select the reduction even when kill identities differ."""
    previous = {
        "mutants": [
            {"id": "m1", "status": "killed"},
            {"id": "m2", "status": "survived"},
        ]
    }
    agent = {
        "mutants": [
            {"id": "m1", "status": "survived"},
            {"id": "m2", "status": "killed"},
        ]
    }

    _drive_consolidate(
        tmp_path,
        monkeypatch,
        remeasured_line=100.0,
        remeasured_branch=90.0,
        reduced_km=previous,
        candidate2_line=100.0,
        candidate2_branch=90.0,
        candidate2_km=agent,
        baseline_km=previous,
        baseline_line=100.0,
        baseline_branch=90.0,
    )

    record = json.loads(
        (
            tmp_path
            / "results"
            / "rank1_foo"
            / "r0" / "archive"
            / "coverage"
            / "consolidation_record.json"
        ).read_text(encoding="utf-8")
    )
    assert record["delivered_from"] == "reduced"
    assert record["shipped_has_matrix"] is True
    assert record["selection_audit"]["delivered_from_missing_from_merged_input"][
        "kills"
    ] == 1


def test_consolidate_coverage_uses_full_precision_candidate_scores(
    tmp_path, monkeypatch,
):
    """Display-rounded metrics do not replace full-precision measurements."""
    exact_line = 96.24494511842865
    exact_branch = 95.48872180451127
    ctx, _result, _cov_src, phase = _drive_consolidate(
        tmp_path,
        monkeypatch,
        remeasured_line=exact_line,
        remeasured_branch=exact_branch,
        agent_line=96.2,
        agent_branch=95.5,
        candidate2_line=exact_line,
        candidate2_branch=exact_branch,
    )
    assert (ctx.work_tests / "test_foo.py").read_text(
        encoding="utf-8"
    ) == phase.reduced_source


def test_the_load_check_messages_name_no_test_runner(monkeypatch, tmp_path):
    """The restored file's load check runs every language's tests, so its
    messages name the check and the output, not a Python runner's pass."""
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text("def test_x(): pass\n", encoding="utf-8")
    runs = []
    backend = SimpleNamespace(
        setup_test_env=lambda *a: None,
        run_test_files=lambda *a, **kw: runs.pop(0))
    monkeypatch.setattr(orchestrator.languages, "get_backend", lambda language: backend)
    runs[:] = [SimpleNamespace(timed_out=True)]
    loads, timed_out = orchestrator._restored_test_file_collects(ctx, "test_foo.py")
    assert not loads
    assert "collect-only" not in timed_out and "timed out" in timed_out
    runs[:] = [SimpleNamespace(timed_out=False, ok=False, collected=0,
                               returncode=2, output="ImportError: no module")]
    loads, failed = orchestrator._restored_test_file_collects(ctx, "test_foo.py")
    assert not loads
    assert "pytest" not in failed and "ImportError: no module" in failed


def test_a_failed_phase_without_a_reason_names_its_log(monkeypatch, tmp_path):
    """A phase that ended on an error without a message points at its log
    by path."""
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text("def test_x(): pass\n")
    fake_result = _coverage_result(
        "test_foo.py", status="agent_error", line_coverage=61.0, error="",
    )
    _patch_agent(
        monkeypatch, orchestrator.coverage_agent,
        generate_coverage_tests=lambda *a, **kw: fake_result,
    )
    result = orchestrator.TestPipelineResult()
    orchestrator._phase_coverage(ctx, result)
    detail = result.phase_outcomes["coverage_detail"]
    assert f"see {ctx.target_logs_dir / 'coverage.jsonl'}." in detail


def test_the_load_check_without_language_support_names_the_file(monkeypatch, tmp_path):
    """Without language support for the selected file, the load check says
    the file cannot be tested."""
    ctx = _make_ctx(tmp_path)
    monkeypatch.setattr(orchestrator.languages, "get_backend", lambda language: None)
    loads, message = orchestrator._restored_test_file_collects(ctx, "test_foo.py")
    assert not loads
    assert message.endswith("this target cannot be tested.")


def test_a_score_just_below_its_minimum_reads_below_it(monkeypatch):
    """The below-minimum reasons cut a score to one decimal, as every shown
    percentage: 74.96% reads 74.9%, never "75.0% is below minimum
    75%"."""
    from testuary.agent import coverage_agent

    monkeypatch.setattr(orchestrator.config_module, "get_agent_config",
                        lambda name, parent=None: {"minimum": 75 if name == "test_coverage" else 80})
    set_phase_value(monkeypatch, coverage_agent._AGENT_CONFIG, "goal_metric", "line")
    result = SimpleNamespace(
        test_result=SimpleNamespace(line_coverage=74.96, branch_coverage=None),
        finishing_result=SimpleNamespace(finishing=FinishingMetrics(mutation_score_full=79.97)))
    assert orchestrator._below_coverage_minimum(result).startswith(
        "coverage 74.9% is below minimum 75%")
    assert orchestrator._below_mutation_minimum(result) == (
        "mutation score 79.9% is below minimum 80%")


@pytest.mark.parametrize("loop_score, final_score, expected", [
    (50.0, 80.0, None),
    (90.0, 40.0, "mutation score 40.0% is below minimum 65%"),
])
def test_the_mutation_minimum_judges_the_delivered_suite(
    monkeypatch, tmp_path, loop_score, final_score, expected,
):
    """The mutation minimum judges the suite the phase end delivered, by
    Finishing's full-catalogue score, as the coverage minimum judges the
    coverage phase end's choice: the inherited suite can win with the
    loop's own best below the minimum, and the reverse."""
    ctx = _make_ctx(tmp_path)
    _patch_starter_handoff(monkeypatch, ctx)
    (ctx.work_tests / "test_foo.py").write_text("def test_cov():\n    assert True\n", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "_dump_resolved_preset", lambda ctx: None)
    _patch_agent(monkeypatch, orchestrator.coverage_agent,
                 generate_coverage_tests=lambda *a, **kw: _coverage_result())
    backend_type = type(orchestrator.languages.get_backend("python"))
    monkeypatch.setattr(backend_type, "mutation_runner", lambda self, *a, **kw: SimpleNamespace(
        open=lambda workers, *, suite_dirs: SimpleNamespace(close=lambda: None)))
    _patch_agent(monkeypatch, orchestrator.mutation_agent, set_mutation_session=lambda session: None,
                 generate_mutation_tests=lambda **kw: _mutation_result("test_foo.py", mutation_score=loop_score))
    monkeypatch.setattr(orchestrator, "_consolidate_mutation",
                        lambda ctx, result: SimpleNamespace(winner="inherited"))
    _patch_agent(monkeypatch, orchestrator.measurement, measure_combined=lambda *a, **kw: _metrics_result())

    def attach_finishing(ctx, result, combined):
        combined.finishing = FinishingMetrics(mutation_score_full=final_score)

    monkeypatch.setattr(orchestrator, "_attach_finishing", attach_finishing)
    monkeypatch.setattr(orchestrator, "_checkpoint_phase", lambda c, r, phase: None)
    result = orchestrator.run_target_phases(ctx)
    assert result.mutation_result.mutation_score == loop_score
    assert result.below_minimum_reason == expected


def test_reading_the_phases_records_makes_no_dir_for_a_phase_that_did_not_run(tmp_path):
    """A resume reads each phase's consolidation record. A phase that did not
    run, such as a skipped mutation phase, keeps no archive directory, so a
    later reuse still tells it from a phase that ran."""
    ctx = SimpleNamespace(results_dir=tmp_path, workspace="ws", run_idx=0)
    coverage = orchestrator._archive_dir(ctx, "coverage")
    record = {"merged_tests": 3, "delivered_tests": 2, "delivered_from": "reduced"}
    (coverage / "consolidation_record.json").write_text(json.dumps(record), encoding="utf-8")
    result = orchestrator.TestPipelineResult()
    orchestrator._restore_consolidation_stats(ctx, result)
    assert result.consolidation_stats == {"coverage": record}
    assert sorted(path.name for path in coverage.parent.iterdir()) == ["coverage"]


def test_the_reuse_load_check_runs_with_the_suite_run_timeout(tmp_path, monkeypatch):
    """The load check of a restored test file runs the suite with the
    coverage phase's suite_run_timeout_seconds, as the suite's other runs
    do, and a timeout names that limit."""
    limit = int(orchestrator.config_module.get_agent_config("test_coverage")["suite_run_timeout_seconds"])
    timeouts = []

    class _Backend:
        timed_out = False

        def setup_test_env(self, *args):
            pass

        def run_test_files(self, paths, source_dir, **kw):
            timeouts.append(kw["timeout"])
            return SimpleNamespace(ok=not self.timed_out, returncode=0, timed_out=self.timed_out,
                                   collected=1, passed=1, failed=0, errors=0, output="",
                                   line_coverage=None, branch_coverage=None, elapsed_s=0.1)

    backend = _Backend()
    monkeypatch.setattr(orchestrator.languages, "detect_language", lambda _p: "python")
    monkeypatch.setattr(orchestrator.languages, "get_backend", lambda _l: backend)
    monkeypatch.setattr(orchestrator.project, "resolve_source_dir", lambda _p: str(tmp_path))
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo.py").write_text("def test_x(): pass\n", encoding="utf-8")
    assert orchestrator._restored_test_file_collects(ctx, "test_foo.py") == (True, "")
    backend.timed_out = True
    assert orchestrator._restored_test_file_collects(ctx, "test_foo.py") == (
        False, f"the test file's load check timed out after {limit} s")
    assert timeouts == [limit, limit]
