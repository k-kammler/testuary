"""Tests for the helpers of ``testuary.orchestrator``, which drives the test
phases of one target: result shapes, phase decisions and outcome
bookkeeping, with isolated fixtures and mocked dependencies."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from testuary import results as csv_writer
from testuary import target_phases as orchestrator

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture(autouse=True)
def _restored_suites_pass_as_delivered(monkeypatch):
    """Accept the check of a restored delivered suite. These
    tests restore suites under --reuse, and the check would run them against
    whatever project tree the fake target resolves to.
    test_orchestrator_phases.py and test_phase_end_normal_run.py cover it."""
    monkeypatch.setattr(orchestrator, "_check_restored_delivered_suite",
                        lambda _ctx, _path: (True, ""))


# ReuseFlags / TestPipelineResult defaults

def test_reuse_flags_default_all_phases_run() -> None:
    sf = orchestrator.ReuseFlags()
    assert sf.coverage is False
    assert sf.mutation is False
    assert sf.finishing is False


def test_test_pipeline_result_default_is_empty() -> None:
    """The orchestrator fills the fields as each phase finishes. The list
    and dict defaults are not shared between instances."""
    r1 = orchestrator.TestPipelineResult()
    r2 = orchestrator.TestPipelineResult()
    r1.phase_outcomes["coverage"] = "ok"
    assert r2.phase_outcomes == {}


# coverage_phase_failed property

def test_a_coverage_outcome_the_phase_never_records_is_not_a_failure() -> None:
    """Only ``failed`` fails the coverage phase, the one failing outcome it
    records."""
    r = orchestrator.TestPipelineResult()
    r.phase_outcomes["coverage"] = "skipped_no_file"
    assert r.coverage_phase_failed is False


def test_coverage_phase_failed_default_no_outcome_recorded() -> None:
    """Before any phase recorded an outcome, the property is False."""
    r = orchestrator.TestPipelineResult()
    assert r.coverage_phase_failed is False


# _record_outcome

# _coverage_failure_reason

def test_coverage_failure_reason_for_missing_file_on_disk() -> None:
    """Another file name than test_failed.py, missing on disk, gives 'file
    missing on disk', which points to a lost write or an early return
    before the flush."""
    result_obj = SimpleNamespace(
        status="ok",
        test_file_name="test_collections_abc.py",
    )
    reason = orchestrator._coverage_failure_reason(result_obj)
    assert "file missing on disk" in reason


# _combined_test_files: which test files form the delivered suite

def _make_ctx(tmp_path: Path, **overrides) -> orchestrator.TargetPhasesContext:
    """Build a minimal TargetPhasesContext for the file-discovery helpers
    that do not need the full agent stack."""
    work_dir = tmp_path / "work"
    work_dir.mkdir()
    work_tests = work_dir / "tests"
    work_tests.mkdir()
    canonical_path = tmp_path / "src" / "foo.py"
    canonical_path.parent.mkdir()
    canonical_path.write_text("def f():\n    return 1\n", encoding="utf-8")
    (canonical_path.parent / "pyproject.toml").write_text("", encoding="utf-8")
    defaults = dict(
        project_name="proj",
        workspace="rank1_test",
        top_file={
            "file_path": str(canonical_path),
            "canonical_path": str(canonical_path),
            "target_number": 1,
        },
        canonical_path=canonical_path,
        work_tests=work_tests,
        results_dir=tmp_path / "results",
        target_logs_dir=tmp_path / "logs",
        reuse=orchestrator.ReuseFlags(starter=True),
        log_print=lambda *a, **k: None,
    )
    defaults.update(overrides)
    return orchestrator.TargetPhasesContext(**defaults)


def _persist_restored_validity(
    ctx: orchestrator.TargetPhasesContext, phase: str, source_path: Path,
) -> None:
    from testuary import languages
    from testuary import evidence as validity
    from testuary import project

    archive = ctx.results_dir / ctx.workspace / "r0" / "archive" / phase
    archive.mkdir(parents=True, exist_ok=True)
    source = source_path.read_text(encoding="utf-8")
    backend = languages.get_backend("python")
    identity = backend.generated_test_target_identity(
        str(ctx.canonical_path), project.resolve_source_dir(str(ctx.canonical_path)),
    )
    validity.write_json_atomic(
        archive / "consolidation_record.json", {"delivered_from": "merged"},
    )
    validity.write_json_atomic(
        archive / "validity_delivered.json",
        validity.direct_record(
            identity=identity,
            analyzed_source=source,
            stage=f"test.{phase}.delivered",
        ),
    )


@pytest.mark.parametrize("outcome, duration", [("ran", 20.0), ("crashed", None)])
@pytest.mark.parametrize("unmeasured", [None, (2, 1)])
def test_restore_starter_baseline_from_snapshot_and_marker(tmp_path: Path, unmeasured, outcome,
                                                          duration) -> None:
    """Restore the starter suite and measurements from its durable snapshot and marker.
    The suites that left without complete numbers come from the generation
    manifest. One written before suites could leave leaves them unknown.
    The phase's time is its ran segments', none without one."""
    ctx = _make_ctx(tmp_path)
    target_dir = ctx.results_dir / ctx.workspace
    arch = target_dir / "r0" / "archive" / "starter"
    arch.mkdir(parents=True)
    snap = arch / "result.py"
    snap.write_text("def test_a():\n    assert True\n", encoding="utf-8")
    _persist_restored_validity(ctx, "starter", snap)
    (arch / "consolidation_record.json").write_text(json.dumps({
        "delivered_from": "merged",
        "merged_tests": 9,
        "delivered_tests": 6,
    }), encoding="utf-8")
    starter_dir = arch / "suites"
    starter_dir.mkdir()
    (starter_dir / "_generation.json").write_text(json.dumps({
        "requests_used": 25,
        "input_tokens": 1000,
        "cache_hit_tokens": 500,
        "output_tokens": 200,
        "generation_seconds": 12.5,
        "n_generated": 25,
        "generation_attempts_per_suite": {"1": 1},
        **({} if unmeasured is None else {
            "n_unmeasured_coverage": unmeasured[0], "n_unmeasured_mutation": unmeasured[1]}),
    }), encoding="utf-8")
    target_dir.joinpath("phase_state.json").write_text(json.dumps({
        "starter": {"0": {"segments": [{
            "seconds": 20.0, "outcome": outcome,
        }], "metrics": {
            "baseline_line": 75.0, "baseline_branch": 60.0,
            "suites_passing": 20, "baseline_suite_seconds": 1.5,
            "selected_tests": 6,
            "suite_line_stats": {
                "mean": 71.0, "median": 72.0, "std": 3.0,
                "min": 60.0, "max": 80.0,
            },
            "suite_branch_stats": {
                "mean": 61.0, "median": 62.0, "std": 4.0,
                "min": 50.0, "max": 70.0,
            },
            "suite_mutation_stats": {
                "mean": 51.0, "median": 52.0, "std": 5.0,
                "min": 40.0, "max": 60.0,
            },
            "suite_mutation_method": "derive",
            "baseline_missing_lines": [3, 4],
            "baseline_missing_branches": ["2->4"],
        }}},
    }), encoding="utf-8")

    ir = orchestrator._restore_starter_baseline(ctx)
    assert ir is not None
    assert ir.baseline_file == str(snap)
    assert ir.baseline_line_coverage == 75.0
    assert ir.baseline_branch_coverage == 60.0
    assert ir.baseline_suite_seconds == 1.5
    assert ir.selected_tests == 6
    assert ir.pool_tests == 9
    assert ir.n_green == 20
    assert ir.requests_used == 25
    assert ir.input_tokens == 1000
    assert ir.duration_s == duration
    assert ir.suite_line_stats["mean"] == 71.0
    assert ir.suite_branch_stats["median"] == 62.0
    assert ir.suite_mutation_stats["max"] == 60.0
    assert ir.suite_mutation_method == "derive"
    expected = unmeasured or (None, None)
    assert (ir.n_unmeasured_coverage, ir.n_unmeasured_mutation) == expected

    result = orchestrator.TestPipelineResult()
    result.starter_result = ir
    handoff = orchestrator._baseline_handoff(result)
    assert handoff is not None
    assert handoff["content"].startswith("def test_a")
    assert handoff["line_coverage"] == 75.0
    assert handoff["test_count"] == 6
    assert handoff["missing_lines"] == (3, 4)
    assert handoff["missing_branches"] == ("2->4",)

    orchestrator._restore_consolidation_stats(ctx, result)
    csv_dir = tmp_path / "fresh_csv"
    csv_dir.mkdir()
    csv_writer.write_tests_row(
        csv_dir, workspace="rank1_csv",
        target_number=1,
        file_path="pkg/demo.py",
        starter_result=ir,
        consolidation_stats=result.consolidation_stats,
    )
    with (csv_dir / "tests.csv").open(
        encoding="utf-8-sig", newline=""
    ) as handle:
        row = next(csv.DictReader(handle))
    assert row["starter_requests"] == "25"
    assert row["starter_merged_tests"] == "9"
    assert row["starter_delivered_tests"] == "6"
    for metric in ("line", "branch", "mutation"):
        for statistic in ("mean", "median", "std", "min", "max"):
            assert row[f"starter_suite_{metric}_{statistic}"] != ""
    assert row["starter_suite_mutation_method"] == "derive"
    assert (row["starter_suites_unmeasured_coverage"], row["starter_suites_unmeasured_mutation"]) == tuple(
        "" if count is None else str(count) for count in expected)


def test_restore_starter_baseline_without_snapshot_fails(tmp_path: Path) -> None:
    captured: list = []
    ctx = _make_ctx(tmp_path, log_print=lambda m, level=0: captured.append(m))
    with pytest.raises(RuntimeError, match="no starter snapshot"):
        orchestrator._restore_starter_baseline(ctx)


def test_the_steps_between_two_phases_are_timed_in_the_log(tmp_path: Path, monkeypatch) -> None:
    """The checkpoint writes and the display after a phase's END marker, and
    the origin, checkpoint and display around the next START marker, each log
    their time, so minutes between two phases never go unattributed. A
    checkpoint that takes 300 s reads so."""
    clock = [0.0]
    monkeypatch.setattr(orchestrator.time, "perf_counter", lambda: clock[0])
    captured: list = []

    def slow_checkpoint(*_args):
        clock[0] += 300.0

    monkeypatch.setattr(orchestrator, "_record_start", slow_checkpoint)
    monkeypatch.setattr(orchestrator, "_record_done", slow_checkpoint)
    ctx = _make_ctx(tmp_path, log_print=lambda m, level=0: captured.append(m))
    result = orchestrator.TestPipelineResult()
    orchestrator._record_outcome(result, "finishing", "ok", None)
    orchestrator._checkpoint_phase(ctx, result, "finishing")
    assert captured[-1] == "[pipeline] finishing recorded: checkpoint 300.0s, display 0.0s"
    orchestrator._start_phase(ctx, "finishing")
    assert captured[-1] == "[pipeline] finishing started: origin 0.0s, checkpoint 300.0s, display 0.0s"


def test_combined_test_files_ignores_another_file_the_mutation_result_names(
        tmp_path: Path) -> None:
    """Mutation revises the delivered file, so a different name in its
    result does not add a second file to the delivered suite."""
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_foo_mutation.py").write_text("# stub\n")
    r = orchestrator.TestPipelineResult(
        test_file_name="test_foo.py",
        mutation_result=SimpleNamespace(
            mutation_test_file_name="test_foo_mutation.py",
        ),
    )
    files = orchestrator._combined_test_files(r)
    assert files == ["test_foo.py"]


def test_combined_test_files_empty_when_no_coverage(tmp_path: Path) -> None:
    """Without a coverage file the list is empty. The caller refuses to
    run on an empty suite."""
    r = orchestrator.TestPipelineResult()
    assert orchestrator._combined_test_files(r) == []


# _final_line_coverage / _final_mutation_score: precedence of the sources

def test_final_line_coverage_falls_back_to_coverage_phase() -> None:
    """Without a finishing result (skipped or failed) and line value, the
    coverage phase's value is used."""
    r = orchestrator.TestPipelineResult(
        test_result=SimpleNamespace(line_coverage=70.0, mutation_score=None),
    )
    assert orchestrator._final_line_coverage(r) == 70.0


def test_final_line_coverage_is_none_when_the_final_run_gave_no_value() -> None:
    """Finishing measured the delivered suite without a value (its coverage
    run timed out or ran red): the coverage phase's value describes another
    suite and does not stand for it."""
    r = orchestrator.TestPipelineResult(
        test_result=SimpleNamespace(line_coverage=70.0, mutation_score=None),
        finishing_result=SimpleNamespace(coverage_final_line=None),
    )
    assert orchestrator._final_line_coverage(r) is None


def test_final_line_coverage_none_when_nothing_measured() -> None:
    """Nothing measured gives None, which the CSV writes blank. A measured
    zero stays 0.0."""
    r = orchestrator.TestPipelineResult()
    assert orchestrator._final_line_coverage(r) is None


def test_final_mutation_score_none_when_no_mutation_phase() -> None:
    """A coverage test_result alone never yields a mutation score. Without
    a mutation or finishing result the score is None."""
    r = orchestrator.TestPipelineResult(
        test_result=SimpleNamespace(line_coverage=80.0),
    )
    assert orchestrator._final_mutation_score(r) is None


def test_final_mutation_score_none_when_nothing_measured() -> None:
    r = orchestrator.TestPipelineResult()
    assert orchestrator._final_mutation_score(r) is None


# _has_usable_test_file

def test_has_usable_test_file_false_when_test_failed_marker(tmp_path: Path) -> None:
    """The synthesized ``test_failed.py`` records the failure and contains
    no passing tests."""
    ctx = _make_ctx(tmp_path)
    (ctx.work_tests / "test_failed.py").write_text("# fail", encoding="utf-8")
    r = orchestrator.TestPipelineResult(test_file_name="test_failed.py")
    assert orchestrator._has_usable_test_file(ctx, r) is False


def test_has_usable_test_file_false_when_missing_on_disk(tmp_path: Path) -> None:
    """A name without a file on disk, typically after an incomplete write,
    keeps later phases from using a nonexistent suite."""
    ctx = _make_ctx(tmp_path)
    r = orchestrator.TestPipelineResult(test_file_name="test_foo.py")
    assert orchestrator._has_usable_test_file(ctx, r) is False


def test_has_usable_test_file_false_when_no_name(tmp_path: Path) -> None:
    """No name at all, as when the agent crashed before writing its first
    file."""
    ctx = _make_ctx(tmp_path)
    r = orchestrator.TestPipelineResult()
    assert orchestrator._has_usable_test_file(ctx, r) is False


# Coverage restore: CoverageTestResult from the metrics the per-run
# phase_state marker holds under its phase and run number.

def test_consolidated_resume_uses_mutation_validity_for_delivered_suite(
    tmp_path: Path, monkeypatch,
) -> None:
    """A completed mutation phase replaces the coverage-named canonical file.

    A resume of finishing alone restores that final file, so its bytes are
    bound by the mutation phase's delivered validity record. The coverage phase's
    record describes the unchanged pre-mutation source and does not reject
    the delivered suite.
    """
    ctx = _make_ctx(
        tmp_path,
        reuse=orchestrator.ReuseFlags(
            starter=True, coverage=True, mutation=True,
            finishing=False,
        ),
    )
    delivered_suite = ctx.work_tests / "test_foo.py"
    delivered_suite.write_text(
        "def test_coverage():\n    assert True\n\n"
        "def test_mutation():\n    assert True\n",
        encoding="utf-8",
    )
    coverage_source = tmp_path / "coverage_source.py"
    coverage_source.write_text(
        "def test_coverage():\n    assert True\n",
        encoding="utf-8",
    )
    _persist_restored_validity(ctx, "coverage", coverage_source)
    _persist_restored_validity(ctx, "mutation", delivered_suite)
    monkeypatch.setattr(
        orchestrator, "_restored_test_file_collects",
        lambda _ctx, _name: (True, ""),
    )

    result = orchestrator.TestPipelineResult()
    restored = orchestrator._restore_coverage_test_file_from_disk(ctx, result)

    assert restored == "test_foo.py"
    assert result.phase_outcomes["coverage"] == "skipped_reused"


def test_a_restored_mutation_result_keeps_the_suites_source_observation(
    tmp_path: Path, monkeypatch,
) -> None:
    """A resume that rebuilds a target's tests.csv row reads the mutation
    phase's result from its checkpoint. The checkpoint keeps the mutation
    suite's structural observation, so the rebuilt row has the
    mutation_*_count cells a fresh run writes."""
    from testuary.agent.mutation_agent import MutationTestResult

    observation = {"mutation": {
        "status": "completed", "reason": "", "parser": "fixture parser",
        "tests": 6, "code_lines": 15, "assertions": 6, "fixtures": 0, "parametrized_cases": 0}}
    result = orchestrator.TestPipelineResult()
    result.phase_outcomes["mutation"] = "ok"
    result.mutation_result = MutationTestResult(
        mutation_test_file_name="test_foo.py", explanation="ran", targeted_mutants=[],
        mutation_score=80.0, source_observations=observation)
    recorded: dict = {}
    monkeypatch.setattr(orchestrator, "_record_done",
                        lambda _ctx, _phase, _success, _artifacts, metrics: recorded.update(metrics))
    ctx = _make_ctx(tmp_path)

    orchestrator._checkpoint_phase(ctx, result, "mutation")
    monkeypatch.setattr(orchestrator, "_recorded_metrics", lambda _ctx, _phase: dict(recorded))
    restored = orchestrator._restored_mutation_result(ctx)

    assert restored.source_observations == observation
    csv_writer.write_tests_row(ctx.results_dir, workspace=ctx.workspace, target_number=1,
                               file_path=str(ctx.canonical_path), mutation_result=restored,
                               preserve_existing=True)
    with open(ctx.results_dir / "tests.csv", newline="", encoding="utf-8-sig") as handle:
        [row] = list(csv.DictReader(handle))
    assert (row["mutation_assertion_count"], row["mutation_fixture_count"],
            row["mutation_parametrize_count"]) == ("6", "0", "0")


def test_standard_mutation_finishing_missing_respects_opt_out(monkeypatch):
    """measure_mutation gates the check for a missing standard
    mutation score. Opted out, an absent score is intended (False). Enabled,
    it is a failure (True)."""
    monkeypatch.setattr("testuary.config.finishing_measures_mutation", lambda: False)
    assert orchestrator._standard_mutation_finishing_missing(
        SimpleNamespace(finishing=None)
    ) is False

    monkeypatch.setattr("testuary.config.finishing_measures_mutation", lambda: True)
    assert orchestrator._standard_mutation_finishing_missing(
        SimpleNamespace(finishing=None)
    ) is True


def _metrics_phase_ctx(tmp_path, monkeypatch):
    # The phase saves its log into the run directory too: keep it out of
    # the working tree.
    from testuary import measurement

    monkeypatch.setattr(measurement.project, "output_dir", lambda **kw: tmp_path / "out")
    work_tests = tmp_path / "work" / "tests"
    work_tests.mkdir(parents=True)
    (work_tests / "test_foo.py").write_text(
        "def test_a():\n    assert True\n", encoding="utf-8",
    )
    src = tmp_path / "foo.py"
    src.write_text("def f():\n    return 1\n", encoding="utf-8")
    ctx = orchestrator.TargetPhasesContext(
        log_print=lambda *a, **k: None,
        project_name="proj", workspace="rank1_foo",
        top_file={"target_number": 1, "file_path": str(src), "canonical_path": str(src)},
        canonical_path=src, work_tests=work_tests,
        results_dir=tmp_path / "results", target_logs_dir=tmp_path / "logs",
        reuse=orchestrator.ReuseFlags(),
    )
    result = orchestrator.TestPipelineResult(test_file_name="test_foo.py")
    return ctx, result


def test_phase_finishing_fails_when_its_mandatory_mutation_run_raises(
    tmp_path, monkeypatch,
):
    """The standard suite's full-catalogue mutation score is required. When
    run_finishing raises (the engine cannot start) and Finishing's
    mutation measurement is not opted out, Finishing records 'failed', not
    'ok'. Otherwise the checkpoint would be written complete, with a blank
    mutation column and the run marked done."""
    from testuary import measurement

    ctx, result = _metrics_phase_ctx(tmp_path, monkeypatch)
    monkeypatch.setattr(
        measurement, "measure_combined",
        lambda **kw: SimpleNamespace(
            finishing=None, status="success", error=None))

    def _raise(_ctx, _res, _combined):
        raise RuntimeError("[finishing] mutation engine could not be started")

    monkeypatch.setattr(orchestrator, "_attach_finishing", _raise)
    monkeypatch.setattr("testuary.config.finishing_measures_mutation", lambda: True)

    combined = orchestrator._phase_finishing(ctx, result)

    assert result.phase_outcomes.get("finishing") == "failed", (
        "Finishing recorded ok despite the mandatory standard-suite "
        "mutation measurement failing"
    )
    # finishing_status in the CSV is metrics.status. It must track the recorded
    # outcome so a failed Finishing is not shipped as success.
    assert combined.status == "failed"
    assert combined.error


def test_phase_finishing_takes_the_coverage_its_phase_end_measured(
    tmp_path, monkeypatch,
):
    """The delivered suite's coverage is the one the phase end that delivered
    its source measured, so finishing judges the suite no second time. The
    record hashes the source text, and the file on disk may carry the
    platform's line ends (CRLF on Windows). A phase end of another source
    gives none, and finishing measures."""
    import hashlib

    from testuary import measurement

    ctx, result = _metrics_phase_ctx(tmp_path, monkeypatch)
    delivered = Path(ctx.work_tests) / result.test_file_name
    source = "def test_a():\n    assert True\n"
    delivered.write_text(source, encoding="utf-8", newline="\r\n")
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()
    # The phase ends of the mutation phase and of the coverage phase name
    # the same source. Only the coverage phase's has coverage evidence.
    (orchestrator._archive_dir(ctx, "mutation") / "consolidation_record.json").write_text(
        json.dumps({"selected_source_sha256": digest}), encoding="utf-8")
    archive = orchestrator._archive_dir(ctx, "coverage")
    (archive / "consolidation_record.json").write_text(
        json.dumps({"selected_source_sha256": digest}), encoding="utf-8")
    (archive / "selected_coverage_matrix.json").write_text(
        json.dumps({"source_sha256": digest, "line": 92.5, "branch": 80.0}), encoding="utf-8")
    recorded = []

    def combined(**kw):
        recorded.append(kw.get("recorded_coverage"))
        return SimpleNamespace(finishing=None, status="success", error=None,
                               coverage_final_line=90.0)

    monkeypatch.setattr(measurement, "measure_combined", combined)
    monkeypatch.setattr(orchestrator, "_attach_finishing", lambda *args: None)
    orchestrator._phase_finishing(ctx, result)
    delivered.write_text("def test_b():\n    assert True\n", encoding="utf-8")
    orchestrator._phase_finishing(ctx, result)
    assert recorded == [(92.5, 80.0), None]
    # The phases that did not run get no archive directory.
    assert not (orchestrator._run_dir(ctx) / "archive" / "starter").exists()


def test_phase_finishing_fails_when_the_final_coverage_gave_no_value(
    tmp_path, monkeypatch,
):
    """A final coverage run without a value (timed out or red) leaves the
    delivered suite's coverage unmeasured: finishing records it
    failed with the reason, as for a missing mutation score, so tests.csv
    and a resume see it."""
    from testuary import measurement

    ctx, result = _metrics_phase_ctx(tmp_path, monkeypatch)
    monkeypatch.setattr(
        measurement, "measure_combined",
        lambda **kw: SimpleNamespace(
            finishing=None, status="success", error=None,
            coverage_final_line=None))

    def _attach_ok(_ctx, _res, combined):
        combined.finishing = SimpleNamespace(
            mutation_score_full=88.0,
        )

    monkeypatch.setattr(orchestrator, "_attach_finishing", _attach_ok)
    monkeypatch.setattr("testuary.config.finishing_measures_mutation", lambda: True)

    combined = orchestrator._phase_finishing(ctx, result)

    assert result.phase_outcomes.get("finishing") == "failed"
    assert "final coverage" in result.phase_outcomes["finishing_detail"]
    assert combined.status == "failed"
    assert "final coverage" in combined.error


# _attach_finishing: finishing opens the mutation session, and
# always closes it, when mutation is enabled. test_metrics covers
# measure_finishing itself.


def _finishing_ctx(tmp_path, *, mutation_skipped: bool):
    work_tests = tmp_path / "work_tests"
    work_tests.mkdir()
    (work_tests / "test_mod.py").write_text("def test_a(): pass\n", encoding="utf-8")
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    ctx = SimpleNamespace(
        top_file={"target_number": 1, "file_path": str(target)},
        work_tests=str(work_tests),
        work_dir=str(tmp_path),
        target_logs_dir=tmp_path / "logs",
        results_dir=tmp_path / "results",
        workspace="ws",
        reuse=SimpleNamespace(mutation=mutation_skipped),
        log_print=lambda *a, **k: None,
    )
    # Without a merge result or a persisted reduced matrix under
    # archive/mutation/, Finishing runs its own green-checked rerun.
    result = SimpleNamespace(test_file_name="test_mod.py", mutation_merge_result=None)
    return ctx, result, target


class _CountingSession:
    def __init__(self, seen):
        self.seen = seen

    def close(self):
        self.seen["exit"] += 1


class _CountingRunner:
    """A mutation runner that counts its sessions' opening and closing."""

    def __init__(self, seen):
        self.seen = seen

    def open(self, workers, *, suite_dirs):
        self.seen["enter"] += 1
        return _CountingSession(self.seen)

    def full_operators(self):
        return None

    def configured_operators(self):
        return None


def _install_mutation_runner(monkeypatch, factory):
    """Route the Python backend's mutation runner to ``factory(config)``."""
    from testuary import languages
    monkeypatch.setattr(type(languages.get_backend("python")),
                        "mutation_runner",
                        lambda self, config, **kw: factory(config))


def test_attach_finishing_opens_and_closes_session_when_mutation_enabled(
    monkeypatch, tmp_path: Path,
) -> None:
    from testuary import measurement
    from testuary import project

    ctx, result, target = _finishing_ctx(tmp_path, mutation_skipped=False)
    monkeypatch.setattr(project, "resolve_source_dir", lambda p: str(tmp_path))

    seen = {"enter": 0, "exit": 0}
    _install_mutation_runner(monkeypatch, lambda config: _CountingRunner(seen))

    captured = {}

    def fake_measure(**kw):
        captured.update(kw)
        return measurement.FinishingMetrics(
            mutation_score_full=77.0, mutation_measurement="complete")

    monkeypatch.setattr(measurement, "measure_finishing", fake_measure)

    combined = measurement.CombinedTestMetrics(
        target_file_path=str(target),
        coverage_final_line=80.0,
        coverage_final_branch=70.0,
    )
    orchestrator._attach_finishing(ctx, result, combined)

    assert combined.finishing is not None
    assert combined.finishing.mutation_score_full == 77.0
    assert seen == {"enter": 1, "exit": 1}
    assert callable(captured["run_mutation_on"])
    assert Path(captured["merged_test_path"]).name == "test_mod.py"
    assert Path(captured["scratch_root"]) == tmp_path / ".finishing_scratch"


def test_attach_finishing_measures_mutation_when_generation_skipped(
    monkeypatch, tmp_path: Path,
) -> None:
    """--reuse mutation reuses the existing mutation tests, so the
    standard suite still carries them and the mandatory mutation score is
    still measured: the engine opens even though generation was skipped."""
    from testuary import measurement
    from testuary import project

    ctx, result, target = _finishing_ctx(tmp_path, mutation_skipped=True)
    monkeypatch.setattr(project, "resolve_source_dir", lambda p: str(tmp_path))

    seen = {"enter": 0, "exit": 0}
    _install_mutation_runner(monkeypatch, lambda config: _CountingRunner(seen))

    def fake_measure(**kw):
        return measurement.FinishingMetrics()

    monkeypatch.setattr(measurement, "measure_finishing", fake_measure)

    combined = measurement.CombinedTestMetrics(
        target_file_path=str(target), coverage_final_line=100.0,
        coverage_final_branch=100.0)
    orchestrator._attach_finishing(ctx, result, combined)

    # The engine opened (and closed) despite --reuse mutation.
    assert seen == {"enter": 1, "exit": 1}
    assert combined.finishing is not None


def test_attach_finishing_hard_stops_when_engine_unavailable(
    monkeypatch, tmp_path: Path,
) -> None:
    """The standard suite's mutation score is mandatory: an unavailable
    engine ends the run with a message, with no coverage-only fallback."""
    from testuary import measurement
    from testuary import project

    ctx, result, target = _finishing_ctx(tmp_path, mutation_skipped=False)
    monkeypatch.setattr(project, "resolve_source_dir", lambda p: str(tmp_path))

    def boom(*a, **k):
        raise RuntimeError("cosmic-ray not installed")

    _install_mutation_runner(monkeypatch, boom)

    combined = measurement.CombinedTestMetrics(
        target_file_path=str(target), coverage_final_line=100.0,
        coverage_final_branch=100.0)
    with pytest.raises(RuntimeError, match="mandatory"):
        orchestrator._attach_finishing(ctx, result, combined)


def test_attach_finishing_opt_out_skips_mutation(
    monkeypatch, tmp_path: Path,
) -> None:
    """measure_mutation = false is the explicit opt-out: no engine
    opens, finishing measures coverage only, and the run does not stop."""
    from testuary import measurement
    from testuary import project

    ctx, result, target = _finishing_ctx(tmp_path, mutation_skipped=False)
    monkeypatch.setattr(project, "resolve_source_dir", lambda p: str(tmp_path))
    monkeypatch.setattr("testuary.config.finishing_measures_mutation", lambda: False)

    def boom(*a, **k):
        raise AssertionError("engine must not open under the opt-out")

    _install_mutation_runner(monkeypatch, boom)

    captured = {}

    def fake_measure(**kw):
        captured.update(kw)
        return measurement.FinishingMetrics()

    monkeypatch.setattr(measurement, "measure_finishing", fake_measure)

    combined = measurement.CombinedTestMetrics(
        target_file_path=str(target), coverage_final_line=100.0,
        coverage_final_branch=100.0)
    orchestrator._attach_finishing(ctx, result, combined)

    assert combined.finishing is not None
    # No mutation measurement is attempted under the opt-out.
    assert captured["run_mutation_on"] is None


def test_attach_finishing_opt_out_takes_no_inherited_mutation_score(
    monkeypatch, tmp_path: Path,
) -> None:
    """Under measure_mutation = false Finishing records no mutation score, no
    measurement status and no kill matrix, and no mutation minimum is
    judged, also when the mutation phase measured the delivered suite
    completely."""
    from testuary import measurement
    from testuary import project

    ctx, result, target = _finishing_ctx(tmp_path, mutation_skipped=False)
    monkeypatch.setattr(project, "resolve_source_dir", lambda p: str(tmp_path))
    monkeypatch.setattr("testuary.config.finishing_measures_mutation", lambda: False)
    result.mutation_merge_result = SimpleNamespace(kill_matrix={"mutants": [
        {"id": "m1", "status": "killed", "killers": ["test_mod.py::test_a"]}]})
    assert orchestrator._resume_standard_km(ctx, result) is not None

    combined = measurement.CombinedTestMetrics(target_file_path=str(target))
    combined.coverage_final_line, combined.coverage_final_branch = 90.0, 80.0
    orchestrator._attach_finishing(ctx, result, combined)

    finishing = combined.finishing
    assert finishing.mutation_score_full is None
    assert finishing.mutation_measurement is None
    assert not (Path(ctx.work_tests) / "kill_matrix.json").exists()
    result.finishing_result = SimpleNamespace(finishing=finishing)
    assert orchestrator._below_mutation_minimum(result) is None


def test_finishing_with_the_mutation_phase_s_matrix_starts_no_engine(
    monkeypatch, tmp_path: Path,
) -> None:
    """Finishing takes the mutation phase end's complete matrix without an
    engine session of its own, so an engine that cannot start does not stop
    it."""
    from testuary import measurement
    from testuary import project

    ctx, result, target = _finishing_ctx(tmp_path, mutation_skipped=False)
    monkeypatch.setattr(project, "resolve_source_dir", lambda p: str(tmp_path))
    seen = {"enter": 0, "exit": 0}

    class Unstartable(_CountingRunner):
        def open(self, workers, *, suite_dirs):
            raise RuntimeError("the engine cannot start")

    _install_mutation_runner(monkeypatch, lambda config: Unstartable(seen))
    result.mutation_merge_result = SimpleNamespace(kill_matrix={"mutants": [
        {"id": "m1", "status": "killed", "killers": ["test_mod.py::test_a"]}]})

    combined = measurement.CombinedTestMetrics(target_file_path=str(target))
    combined.coverage_final_line, combined.coverage_final_branch = 90.0, 80.0
    orchestrator._attach_finishing(ctx, result, combined)

    assert combined.finishing.mutation_score_full == 100.0
    assert combined.finishing.mutation_measurement == "complete"
    assert seen == {"enter": 0, "exit": 0}


def test_finishing_notes_no_time_limit_for_an_inherited_incomplete_matrix(
    monkeypatch, tmp_path: Path,
) -> None:
    """The mutation phase end's matrix that left mutants untested noted its
    own cut in its own phase. Finishing takes it without a note of its own,
    whatever left the mutants untested."""
    from testuary import measurement
    from testuary import project
    from testuary import results as time_limits
    from testuary.languages import mutation as kill_matrix

    ctx, result, target = _finishing_ctx(tmp_path, mutation_skipped=False)
    monkeypatch.setattr(project, "resolve_source_dir", lambda p: str(tmp_path))
    seen = {"enter": 0, "exit": 0}
    _install_mutation_runner(monkeypatch, lambda config: _CountingRunner(seen))
    result.mutation_merge_result = SimpleNamespace(kill_matrix={
        "mutants": [{"id": "m1", "status": "killed", "killers": ["test_mod.py::test_a"]},
                    {"id": "m2", "status": "untested"}],
        kill_matrix.INCOMPLETE_KEY: {"reason": "untested_mutants_in_catalogue"}})

    combined = measurement.CombinedTestMetrics(target_file_path=str(target))
    combined.coverage_final_line, combined.coverage_final_branch = 90.0, 80.0
    with time_limits.collecting(tmp_path / "r0"), time_limits.in_phase("finishing"):
        orchestrator._attach_finishing(ctx, result, combined)

    assert combined.finishing.mutation_measurement == "incomplete"
    assert time_limits.warnings(tmp_path / "r0") == []


def test_attach_finishing_hard_stops_when_standard_suite_missing(
    monkeypatch, tmp_path: Path,
) -> None:
    """A missing standard suite at finishing is a missing required input:
    the run stops with a message instead of skipping it."""
    from testuary import measurement
    from testuary import project

    ctx, result, target = _finishing_ctx(tmp_path, mutation_skipped=False)
    monkeypatch.setattr(project, "resolve_source_dir", lambda p: str(tmp_path))
    result.test_file_name = "does_not_exist.py"

    combined = measurement.CombinedTestMetrics(target_file_path=str(target))
    with pytest.raises(RuntimeError, match="standard test suite is mandatory"):
        orchestrator._attach_finishing(ctx, result, combined)


# _dump_resolved_preset: the run's record of the settings it ran under

def test_dump_resolved_preset_never_raises_when_results_dir_unwritable(
    tmp_path: Path, monkeypatch,
) -> None:
    """A dump write failure is logged and swallowed: the dump is a record, not
    a gate on the run."""
    from testuary import config as config_module

    monkeypatch.setattr(
        config_module, "resolved_test_preset",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    logs: list[str] = []
    ctx = _make_ctx(tmp_path, log_print=lambda *a, **k: logs.append(a[0] if a else ""))
    orchestrator._dump_resolved_preset(ctx)  # must not raise
    assert any("dump skipped" in m for m in logs)
    assert any("RuntimeError: boom" in m for m in logs)


def test_restore_agent_usage_noop_without_data(tmp_path) -> None:
    """Without checkpoint tokens nothing is set, not zeros."""
    obj = SimpleNamespace(
        input_tokens=None, output_tokens=None, agent_iterations=None)
    orchestrator._restore_agent_usage(obj, {})
    assert obj.input_tokens is None
    assert obj.output_tokens is None
    assert obj.agent_iterations is None



def test_restore_agent_usage_leaves_request_accounting_unset_without_it(
    tmp_path,
) -> None:
    """A checkpoint without request counts: the counts stay unset, not
    zero."""
    obj = SimpleNamespace(
        input_tokens=None, output_tokens=None, agent_iterations=None,
        cache_hit_tokens=None, model_requests=None, logical_requests=None,
        unknown_usage_requests=None)
    orchestrator._restore_agent_usage(
        obj, {"input_tokens": 100, "output_tokens": 50})
    assert obj.input_tokens == 100
    assert obj.model_requests is None
    assert obj.logical_requests is None
    assert obj.unknown_usage_requests is None
