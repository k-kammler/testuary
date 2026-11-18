"""The target phases: runs the four phases of one target in order.

The orchestration is a top-level function (``run_target_phases``) plus
a handful of phase-helper functions, not a class. A sequential
pipeline with shared resource lifecycle is the natural fit for
functions threaded through a context object.

Phases run in order: ``starter`` -> ``coverage`` -> ``mutation`` ->
``finishing``. The coverage and mutation phases each end by consolidating
the agent's suite with the suite it grew from, so Finishing reads one
canonical file. Measurement is last so a single pass covers the suite
the target delivered.

What lives here:

  - starter phase. Delegates to
    ``starters.run_starter_phase``, which hands the coverage agent its
    baseline suite.
  - coverage tests. Delegates to
    ``coverage_agent.generate_coverage_tests``. With
    ``--reuse coverage`` it restores the file from
    disk.
  - mutation tests. Delegates to
    ``mutation_agent.generate_mutation_tests``. The mutation
    phase's ``measure_mutation`` calls share a single
    ``MutationSession`` so the engine's worker pool (the cosmic-ray HTTP
    workers for Python) is spawned once per target instead of once per call
    (amortizes the pool setup cost per target).
  - Finishing: the delivered suite's measurement. Delegates to
    ``measurement.measure_combined``, a measurement helper rather
    than an LLM agent.

What lives in ``testuary.orchestrator``: source-tree discovery, work-dir
setup, resume planning, and the per-target loop that calls
``run_target_phases(ctx)``. ``__main__`` is the CLI that parses arguments and
calls into it.
"""
from __future__ import annotations

import contextlib
import dataclasses
import json
import shutil
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, TYPE_CHECKING
from testuary import generation_evidence, languages, normal_run, phase_state, project
from testuary import evidence as validity
from testuary.generation_evidence import source_sha256
from testuary import measurement
from testuary.languages import mutation
from testuary.languages.generic import FAILED_MARKER
from testuary import starters
from testuary.agent import coverage_agent, mutation_agent
from testuary import config as config_module
from testuary.agent.coverage_agent import CoverageTestResult
from testuary.agent.mutation_agent import MutationTestResult
from testuary.measurement import CombinedTestMetrics
from testuary.agent._helpers import log_phase_finished, log_phase_started


if TYPE_CHECKING:
    from testuary.settings import RunSettings

# =============================================================================
# CONTEXT + RESULT TYPES
# =============================================================================

@dataclass(frozen=True)
class ReuseFlags:
    """Per-phase reuse flags, typed wrapper around argparse booleans.

    ``starter``  -> --reuse starter (also skipped whenever the coverage
                    suite itself is reused: the baseline it produces IS
                    the coverage file being restored)
    ``coverage`` -> --reuse coverage
    ``mutation`` -> --reuse mutation
    ``finishing`` -> --reuse finishing (the delivered-suite measurement, last)

    A resume sets those whose phase is verifiably complete.
    """
    starter: bool = False
    coverage: bool = False
    mutation: bool = False
    finishing: bool = False


@dataclass(frozen=True)
class TargetPhasesContext:
    """Everything ``run_target_phases`` needs to drive the test
    phases for one target. Constructed by ``testuary.orchestrator`` from
    argparse + the per-target workspace setup."""

    project_name: str                   # e.g. "Python-3.12.13" (source directory)
    workspace: str                      # e.g. "xarray_core_groupby". The
                                        # per-target label the run uses for the
                                        # work / results / logs subdir naming
    top_file: Dict[str, Any]            # the target's descriptor
    canonical_path: Path                # the target's canonical source
    work_tests: Path                    # <outdir>/_work/[<namespace>/]<workspace>/tests
    results_dir: Path                   # <outdir>/results
    target_logs_dir: Path                  # <outdir>/logs/test/<workspace>/
    reuse: ReuseFlags
    # The logger of the phase log lines and the phases' own lines: the
    # orchestrator's main logger, so a crash mid-pipeline is localized in
    # the always-flushed main.jsonl (the per-agent logs stream to the target
    # directory, which a crash before the first flush leaves empty).
    log_print: Callable[..., None]
    # Which run of the test pipeline this is (0 for r0), under ``--runs N``
    # or ``--run K``. The checkpoint block is per (phase, run_idx), so a
    # segment written without it would land on the first run's record.
    run_idx: int = 0
    # The phases this run's resume found complete in this run's own
    # records. They restore from this run, every other skipped phase
    # from the first run (a reuse).
    resumed: frozenset = frozenset()
    # Explicit diagnostic boundary. None preserves the complete pipeline.
    # A stop is honored only after the selected logical phase has finished
    # its existing consolidation and durable checkpoint work.
    stop_after_phase: Optional[str] = None
    settings: Optional[RunSettings] = None
    # Display only: no limits, scientific results or resume state live here.
    progress: Optional[Any] = None
    # The selected source path: the display names the target by it, and the
    # checkpoints record it as the target's path, which a resume compares.
    # Empty falls back to the workspace name on the display.
    display_target: str = ""
    # Ephemeral prepared clients, never part of result or restart data.
    prepared_models: Optional[Any] = None
    # A resumed run (--resume): an unfinished starter phase continues from
    # its saved generations (see _reuses_suites).
    resume: bool = False
    # Construction provenance only, never a request identity or limit counter.
    generation_invocation_id: str = field(default_factory=lambda: uuid.uuid4().hex)


@dataclass
class TestPipelineResult:
    """Aggregated result of a target's phases: the single object ``__main__`` reads
    after ``run_target_phases`` returns.

    ``final_line_coverage`` is Finishing's measurement, else the
    coverage phase's. ``final_mutation_score`` is the mutation phase's
    score: the coverage phase measures no score, and Finishing's
    final mutation measurement is recorded separately.
    """

    test_file_name: Optional[str] = None
    starter_result: Optional[Any] = None
    test_result: Optional[CoverageTestResult] = None
    mutation_result: Optional[MutationTestResult] = None
    # Transient, source-bound Phase 3 inputs. They are populated before the
    # Mutation Agent edits the delivered path and consumed by the boundary.
    # Resume reconstructs the inherited source from the coverage archive.
    mutation_inherited_source: Optional[str] = None
    mutation_merge_result: Optional[Any] = None
    finishing_result: Optional[CombinedTestMetrics] = None
    phase_outcomes: Dict[str, str] = field(default_factory=dict)
    # Wall-clock duration per phase, captured even when the phase
    # fails / returns None. The CSV writer fills a phase's
    # ``<phase>_duration_s`` from this when the row has none, so a failed
    # phase's time is reported too. The phase's own record wins where it
    # holds a time, and ``agent_duration_s`` has its own column.
    phase_durations: Dict[str, float] = field(default_factory=dict)
    # Phase-end consolidation stages, keyed by phase ("coverage" / "mutation"):
    # merged and delivered test counts, the keep-best winner, and the strip flag.
    # The CSV writer flattens each into <phase>_<key> columns.
    consolidation_stats: Dict[str, Dict[str, Any]] = field(
        default_factory=dict)
    # Cached "best available" derived values.
    final_line_coverage: Optional[float] = None
    final_mutation_score: Optional[float] = None
    # Set when the target file's best coverage / mutation score ended below the
    # configured ``minimum`` / ``minimum`` floor.
    # ``complete_tests_below_minimum`` decides whether the remaining test
    # phases still run (Finishing records the numbers either way).
    # The reason string names the metric, the value, and the floor it missed.
    below_minimum_reason: Optional[str] = None
    # Internal control signal consumed by the top-level orchestrator. It is not
    # serialized into run outputs and is None on every normal run.
    stopped_after_phase: Optional[str] = None

    @property
    def coverage_phase_failed(self) -> bool:
        """True when the target has no usable on-disk test file: the
        coverage phase failed, or the starter phase cancelled the
        target (zero green starter tests) before coverage ran.
        ``testuary.orchestrator`` uses this to skip every later phase."""
        return (self.phase_outcomes.get("coverage") == "failed"
                or self.phase_outcomes.get("starter") == "failed")


# =============================================================================
# PUBLIC ENTRY POINT
# =============================================================================

def _run_dir(ctx: "TargetPhasesContext") -> Path:
    """This run's directory below the target's directory, such as
    ``results/<workspace>/r0``: its suite, archive and resolved preset."""
    return project.run_dir(ctx.results_dir / ctx.workspace, getattr(ctx, "run_idx", 0))


def _restore_run_idx(ctx: "TargetPhasesContext", phase: str) -> int:
    """The run a skipped ``phase`` restores its suite and measurements from:
    this run for a phase its resume found complete, else the first run, since
    a reuse takes the first run's results as the input of every run."""
    return ctx.run_idx if phase in getattr(ctx, "resumed", ()) else 0


def _recorded_metrics(ctx: "TargetPhasesContext", phase: str) -> Dict[str, Any]:
    """The metrics the checkpoint of ``phase`` recorded, from the run a
    restore takes the phase from."""
    state = phase_state.load(ctx.results_dir / ctx.workspace)
    return phase_state.phase_metrics(state, phase, run_idx=_restore_run_idx(ctx, phase))


def _restore_run_dir(ctx: "TargetPhasesContext", phase: str) -> Path:
    """The directory of the run a skipped ``phase`` restores from."""
    return project.run_dir(ctx.results_dir / ctx.workspace, _restore_run_idx(ctx, phase))


def _requested_phase_stop(
    ctx: TargetPhasesContext,
    result: TestPipelineResult,
    phase: str,
) -> bool:
    """Mark an explicitly requested, successfully reached phase boundary."""
    if ctx.stop_after_phase != phase:
        return False
    if result.phase_outcomes.get(phase) == "failed":
        return False
    result.stopped_after_phase = phase
    ctx.log_print(
        f"[pipeline] requested stop reached after {phase}. "
        "Later phases and final result derivation are skipped.",
        level=1,
    )
    return True


def _below_coverage_minimum(result: TestPipelineResult) -> Optional[str]:
    """Reason string when the coverage phase's goal-metric value ended
    below ``minimum``, else None.

    Judged on the same combined goal metric the goal / measurements without improvement checks
    use, so the floor and the goal read one scale. A target with no
    measurable value is not judged here: the coverage-failed path already
    handles a target that produced no usable suite, and a missing branch
    measurement under a branch-based metric cannot prove the floor was
    missed."""
    cov_min = float(
        config_module.get_agent_config("test_coverage")["minimum"]
    )
    if cov_min <= 0:
        return None
    tr = result.test_result
    value = coverage_agent._goal_metric_value(
        getattr(tr, "line_coverage", None),
        getattr(tr, "branch_coverage", None),
    )
    if value is None or value >= cov_min:
        return None
    metric = coverage_agent._goal_metric()
    return (
        f"coverage {measurement.floor_pct(value):.1f}% is below minimum {cov_min:.0f}% "
        f"(goal metric: {metric})"
    )


def _below_mutation_minimum(result: TestPipelineResult) -> Optional[str]:
    """Reason string when the delivered suite's full-catalogue mutation
    score, Finishing's, ended below ``minimum``, else None. Same
    judged-only-when-measurable policy as ``_below_coverage_minimum``."""
    mut_min = float(
        config_module.get_agent_config(
            "test_mutation", parent="test_coverage",
        )["minimum"]
    )
    if mut_min <= 0:
        return None
    score = _final_mutation_shown(result)
    if score is None or score >= mut_min:
        return None
    return (
        f"mutation score {measurement.floor_pct(score):.1f}% is below "
        f"minimum {mut_min:.0f}%"
    )


def _dump_resolved_preset(ctx: TargetPhasesContext) -> None:
    """Record the resolved test preset to ``<results_dir>/resolved_settings.json``
    and log a one-line summary, so every run carries its own statement of the
    goals / limits / switch knobs it executed under. A resume keeps the
    statement its run began with, and any other run writes its own, also
    over an earlier run's whose phases it reuses. A dump failure is logged
    and swallowed: it is a record, never a gate on the run."""
    log = ctx.log_print
    try:
        phases = ("starter", "coverage", "mutation")
        if ctx.stop_after_phase in phases:
            phases = phases[:phases.index(ctx.stop_after_phase) + 1]
        phases = tuple(phase for phase in phases if not (
            getattr(ctx.reuse, phase) or (phase == "starter" and ctx.reuse.coverage)
        ))
        language = languages.detect_language(str(ctx.canonical_path))
        preset = config_module.resolved_test_preset(model_phases=phases, language=language)
        # The fallbacks the run takes for this file's language, recorded only
        # when one applies (``languages.profile``).
        fallbacks = languages.records(language)
        if fallbacks is not None:
            preset["fallbacks"] = fallbacks
    except Exception as exc:
        log(f"[preset] resolved-config dump skipped: {type(exc).__name__}: {exc}")
        return
    try:
        _run_dir(ctx).mkdir(parents=True, exist_ok=True)
        preset_path = _run_dir(ctx) / "resolved_settings.json"
        if not (ctx.resume and preset_path.exists()):
            preset_path.write_text(json.dumps(preset, indent=2, sort_keys=True), encoding="utf-8")
    except OSError as exc:
        log(f"[preset] resolved-config dump write failed: {exc}")
    cov = config_module.get_agent_config("test_coverage")
    mut = config_module.get_agent_config("test_mutation", parent="test_coverage")
    log(
        "[preset] coverage goal={cg}/{cm} requests={cr} without_improvement={cp} "
        "| mutation goal={mg} requests={mr} "
        "without_improvement={mp} full_after_selective={mf}@{ms} per_mutant_timeout={mt}s".format(
            cg=cov["goal"], cm=cov["goal_metric"],
            cr=cov["max_requests_per_phase"] or "unlimited", cp=cov["max_measurements_without_improvement"],
            mg=mut["goal"], mr=mut["max_requests_per_phase"] or "unlimited",
            mp=mut["max_measurements_without_improvement"],
            mf=mut["full_after_selective_goal"],
            ms=mut["selective_switch_threshold"],
            mt=mut["per_mutant_timeout_seconds"],
        ))


# The coverage and mutation results' outcome, duration and suite counts a
# checkpoint records, so a restored phase reports them like the run that
# produced it and tests.csv keeps them when a resume measures again.
_RESTORED_FIELDS = {
    "coverage": ("status", "error", "agent_duration_s", "test_count_standard",
                 "test_loc_standard", "target_loc", "target_complexity"),
    "mutation": ("status", "error", "agent_duration_s", "test_count_mutation",
                 "test_loc_mutation", "mutation_switched", "mutation_selective_final",
                 "mutation_full_at_switch", "source_observations"),
}


#: The model use a coverage or mutation checkpoint records, so a reuse
#: reports the cost of the run that produced the suite.
_USAGE_FIELDS = ("input_tokens", "cache_hit_tokens", "output_tokens", "agent_iterations",
                 "model_requests", "logical_requests", "unknown_usage_requests",
                 "unknown_cache_requests")


#: A result field the checkpoint records under the glossary's name. The
#: field keeps its result-schema name.
_RECORD_KEYS = {"test_count_mutation": "mutation_test_count",
                "test_loc_mutation": "mutation_test_loc"}


def _restored_fields(result_obj: Any, phase: str) -> Dict[str, Any]:
    """The ``_RESTORED_FIELDS`` of a phase's result, for its checkpoint."""
    return {_RECORD_KEYS.get(name, name): getattr(result_obj, name, None)
            for name in _RESTORED_FIELDS[phase]}


def _usage_fields(result_obj: Any) -> Dict[str, Any]:
    """The ``_USAGE_FIELDS`` of a phase's result, for its checkpoint."""
    return {name: getattr(result_obj, name, None) for name in _USAGE_FIELDS}


def _restore_agent_usage(result_obj: Any, metrics: Dict[str, Any], phase: Optional[str] = None) -> None:
    """Add the prior run's model use back into a restored coverage or
    mutation result, so a ``--reuse`` run reports it like the run that
    generated the suite. With ``phase``, also its ``_RESTORED_FIELDS``. A
    value the checkpoint lacks stays unset: absent is not zero."""
    for name in _RESTORED_FIELDS.get(phase, ()):
        value = metrics.get(_RECORD_KEYS.get(name, name))
        if value is not None:
            setattr(result_obj, name, value)
    for name in _USAGE_FIELDS:
        if metrics.get(name) is not None:
            setattr(result_obj, name, int(metrics[name]))


def _restored_coverage_result(ctx: "TargetPhasesContext", test_file_name: str) -> Optional[CoverageTestResult]:
    """The coverage phase's result rebuilt from its checkpoint, None when
    the checkpoint records no coverage."""
    metrics = _recorded_metrics(ctx, "coverage")
    line, branch = metrics.get("coverage"), metrics.get("branch_coverage")
    if line is None and branch is None:
        return None
    restored = CoverageTestResult(
        test_file_name=test_file_name,
        explanation="Restored from a prior run's checkpoint.",
        targeted_functions=[],
        line_coverage=float(line) if line is not None else None,
        branch_coverage=float(branch) if branch is not None else None,
    )
    _restore_agent_usage(restored, metrics, "coverage")
    return restored


def _restored_mutation_result(ctx: "TargetPhasesContext") -> Optional[MutationTestResult]:
    """The mutation phase's result rebuilt from its checkpoint, None when
    the checkpoint records no score."""
    metrics = _recorded_metrics(ctx, "mutation")
    score = metrics.get("mutation_score")
    if score is None:
        return None
    restored = MutationTestResult(
        mutation_test_file_name=str(metrics.get("mutation_file", "")),
        explanation="Restored from a prior run's checkpoint.",
        targeted_mutants=[],
        mutation_score=float(score),
    )
    _restore_agent_usage(restored, metrics, "mutation")
    return restored


def completed_phase_results(ctx: "TargetPhasesContext") -> Dict[str, Any]:
    """The results of the phases this run completed, rebuilt from their
    checkpoints as a resume restores them, for the tests.csv row of a target
    whose pipeline crashed: ``starter_result``, ``test_result``,
    ``mutation_result`` and ``consolidation_stats`` as
    ``results.write_tests_row`` takes them. A phase without its
    checkpoint, or whose records do not rebuild, is left out."""
    state = phase_state.load(ctx.results_dir / ctx.workspace)
    done = frozenset(phase for phase in ("starter", "coverage", "mutation")
                     if (phase_state._read_run_block(state, phase, ctx.run_idx) or {}).get("status")
                     == phase_state.COMPLETE)
    # A restore reads the records of this run for the phases it completed.
    ctx = dataclasses.replace(ctx, resumed=done)
    rebuilt: Dict[str, Any] = {}
    for key, phase, rebuild in (
            ("starter_result", "starter", lambda: _restore_starter_baseline(ctx)),
            ("test_result", "coverage",
             lambda: _restored_coverage_result(ctx, "".join(_canonical_suite_name(ctx)))),
            ("mutation_result", "mutation", lambda: _restored_mutation_result(ctx))):
        if phase in done:
            try:
                rebuilt[key] = rebuild()
            except Exception:
                pass
    stats = {phase: _recorded_consolidation(ctx, phase) for phase in sorted(done)}
    stats = {phase: record for phase, record in stats.items() if record}
    if stats:
        rebuilt["consolidation_stats"] = stats
    return rebuilt


def run_target_phases(ctx: TargetPhasesContext) -> TestPipelineResult:
    """Run a target under the run's active settings, which a context's
    ``settings`` must be when it names any."""
    from testuary._progress import observe_target

    from testuary import results

    run_dir = project.run_dir(Path(ctx.results_dir) / ctx.workspace, ctx.run_idx)
    with observe_target(ctx.progress, _display_target(ctx)), results.collecting(run_dir):
        if ctx.settings is not None and config_module.active_settings() is not ctx.settings:
            raise ValueError("phase settings differ from the active run settings")
        return _run_target_phases(ctx)


def _run_target_phases(ctx: TargetPhasesContext) -> TestPipelineResult:
    """Run the phases for one target and return aggregated results.

    Idempotent at the phase level: each phase reads its own --reuse flag
    and either runs or restores-from-disk. The mutation engine session
    is opened once around the mutation phase so the cosmic-ray worker
    pool is reused across every ``measure_mutation`` call.

    When ``ctx`` carries the resumable-run hooks, each phase that
    actually runs (not the restore-from-disk path) is bracketed by its
    started record and, on typed success, a durable extract and its
    checkpoint. A restored phase fires no hooks: the resume layer that
    set its skip flag already validated and restored it.
    """
    from testuary._progress import observed_step

    result = TestPipelineResult()
    _guard_generation_reuse(ctx)
    _dump_resolved_preset(ctx)

    # ---- starter phase ----
    # Runs first and writes the coverage agent's NON-frozen baseline to
    # the canonical coverage test path. Skipped whenever the coverage
    # suite itself is reused (the baseline IS that file). A pool that
    # ends with zero green tests cancels the target: nothing downstream
    # can stand on an empty baseline claim.
    if not ctx.reuse.starter and not ctx.reuse.coverage:
        _start_phase(ctx, "starter")
        _t0 = time.perf_counter()
        with _phase_segment(ctx, "starter", suites_reused=_reuses_starter_pool(ctx)):
            result.starter_result = _phase_starter(ctx, result)
        result.phase_durations["starter"] = time.perf_counter() - _t0
        if (
            result.starter_result is not None
            and result.starter_result.status != "ok"
        ):
            _checkpoint_phase(ctx, result, "starter")
            return result
        if result.starter_result is None:
            _record_outcome(
                result,
                "starter",
                "failed",
                "the starter phase produced no suite to hand to the coverage phase",
            )
            _checkpoint_phase(ctx, result, "starter")
            return result
        _checkpoint_phase(ctx, result, "starter")
    else:
        _record_outcome(result, "starter", "skipped_reused", None)
        _record_phase_restored(ctx, "starter")
        if not ctx.reuse.coverage or not ctx.reuse.finishing:
            # Rebuild the authoritative starter result for both downstream handoff
            # and final CSV reconstruction. A metrics-only resume reuses
            # coverage, but still needs the starter manifest, consolidation
            # counts, tokens, and baseline scores. Leaving ``starter_result``
            # unset made Finishing write blank starter columns.
            result.starter_result = ir = _restore_starter_baseline(ctx)
            # Logged here, not in the restore, which a crash row's rebuild uses too.
            ctx.log_print(f"[restore] starter baseline restored for the coverage handoff: "
                          f"{Path(ir.baseline_file).name} (line {ir.baseline_line_coverage} "
                          f"branch {ir.baseline_branch_coverage})")
            if not ctx.reuse.coverage:
                _place_restored_starter(ctx, result.starter_result)
        _observe_restored_phase(ctx, "starter", result)

    if _requested_phase_stop(ctx, result, "starter"):
        return result

    # ---- coverage tests ----
    baseline = None if ctx.reuse.coverage else _baseline_handoff(result)
    if not ctx.reuse.coverage:
        _start_phase(ctx, "coverage")
        _t0 = time.perf_counter()
        with _phase_segment(ctx, "coverage"):
            _refuse_foreign_starter_measurement(ctx)
            with _source_integrity_guard(ctx, "coverage"):
                result.test_result = _phase_coverage(ctx, result, baseline)
            result.test_file_name = (
                result.test_result.test_file_name
                if result.test_result else None
            )
            if baseline is not None and result.test_file_name:
                with observed_step("coverage", "consolidation", "Consolidation"):
                    _consolidate_coverage(ctx, result, baseline["content"])
        result.phase_durations["coverage"] = time.perf_counter() - _t0
        _checkpoint_phase(ctx, result, "coverage")
    else:
        result.test_file_name = _restore_coverage_test_file_from_disk(ctx, result)
        _record_phase_restored(ctx, "coverage")
        # On the restore path the coverage agent did not run this
        # session, so rebuild a minimal ``CoverageTestResult`` from
        # the checkpoint metrics. Downstream phases read
        # ``test_result.line_coverage`` to surface the coverage-phase
        # baseline in their prompts.
        if result.test_file_name and result.test_result is None:
            try:
                result.test_result = _restored_coverage_result(ctx, result.test_file_name)
            except Exception:
                pass

    if not _has_usable_test_file(ctx, result):
        # The coverage phase produced no usable file. Every later phase
        # is skipped. The caller reads ``result.coverage_phase_failed``.
        return result

    if ctx.reuse.coverage:
        _observe_restored_phase(ctx, "coverage", result)

    if _requested_phase_stop(ctx, result, "coverage"):
        return result

    # ---- minimum-coverage floor ----
    # A suite below ``minimum`` is recorded as below the floor.
    # ``complete_tests_below_minimum`` decides what happens to the rest
    # of the target's phases: true keeps generating (a complete suite is still
    # an evaluation deliverable), false ends generation here. The metrics
    # phase runs either way: the numbers that triggered the skip belong
    # in the CSV.
    complete_below_min = bool(
        config_module.get_agent_config("test_coverage").get(
            "complete_tests_below_minimum", True,
        )
    )
    skip_generation = False
    cov_floor_reason = _below_coverage_minimum(result)
    if cov_floor_reason:
        result.below_minimum_reason = cov_floor_reason
        log = ctx.log_print
        if complete_below_min:
            log(f"[minimum] {cov_floor_reason}. The mutation phase runs "
                f"anyway (complete_tests_below_minimum = true).", level=1)
        else:
            skip_generation = True
            log(f"[minimum] {cov_floor_reason}. The mutation phase is skipped "
                f"for this target (complete_tests_below_minimum = false).", level=1)
            _record_outcome(
                result, "mutation", "skipped_below_minimum",
                cov_floor_reason,
            )

    # ---- mutation tests (own cosmic-ray session) ----
    # Reuse one worker pool across mutation measurements in this phase.
    # Finishing opens its own session when another measurement is needed.
    if not skip_generation:
        if not ctx.reuse.mutation:
            _start_phase(ctx, "mutation")
            _t0 = time.perf_counter()
            with _phase_segment(ctx, "mutation"):
                with _open_mutation_session(ctx) as session:
                    with _source_integrity_guard(ctx, "mutation"):
                        result.mutation_result = _phase_mutation(
                            ctx, result, session,
                        )
                # Consolidation runs after the agent's worker session closes so
                # its own measurements never overlap that fleet.
                if result.mutation_result is not None:
                    with observed_step("mutation", "consolidation", "Consolidation"):
                        result.mutation_merge_result = _consolidate_mutation(
                            ctx, result,
                        )
            result.phase_durations["mutation"] = time.perf_counter() - _t0
            _checkpoint_phase(ctx, result, "mutation")
        elif not _reused_target_test_present(ctx):
            # Mutation revises the delivered coverage-named suite, so reusing
            # the mutation phase needs that file on disk. The score is
            # restored from the checkpoint below.
            _record_outcome(
                result, "mutation", "failed",
                "no delivered test file on disk to reuse under "
                "--reuse mutation",
            )
            return result
        elif result.mutation_result is None:
            # Restored, not run this session (--reuse mutation /
            # --resume): rebuild a minimal MutationTestResult from
            # the checkpoint so _final_mutation_score has the score. The
            # Finishing's mutation measurement does not feed that
            # score, so this is the only place the resumed score is
            # recovered. Mirrors the coverage restore above.
            backend = languages.get_backend(
                languages.detect_language(str(ctx.top_file["file_path"]))
            )
            restored_suite = (
                backend.find_target_test_file(
                    str(ctx.work_tests), str(ctx.top_file["file_path"])
                )
                if backend is not None else None
            )
            if restored_suite is None:
                _record_outcome(
                    result,
                    "mutation",
                    "failed",
                    "no restored suite is available for mutation validity "
                    "verification",
                )
                return result
            valid, validity_reason = _verify_restored_phase_validity(
                ctx, "mutation", restored_suite,
            )
            if not valid:
                _record_outcome(
                    result,
                    "mutation",
                    "failed",
                    "the restored mutation suite does not match its validation "
                    f"record: {validity_reason}",
                )
                return result
            try:
                result.mutation_result = _restored_mutation_result(ctx)
            except Exception:
                pass

    if not skip_generation and ctx.reuse.mutation:
        _observe_restored_phase(ctx, "mutation", result)

    if _requested_phase_stop(ctx, result, "mutation"):
        return result

    # Interim coverage and mutation values are recomputed after metrics.
    result.final_line_coverage = _final_line_coverage(result)
    result.final_mutation_score = _final_mutation_score(result)

    # Measure the delivered suite after generation and consolidation.
    if not ctx.reuse.finishing:
        _start_phase(ctx, "finishing")
        _t0 = time.perf_counter()
        with _phase_segment(ctx, "finishing"):
            with _source_integrity_guard(ctx, "finishing"), observed_step(
                    "finishing", "delivered-measurement", "Checking the delivered suite"):
                result.finishing_result = _phase_finishing(ctx, result)
        result.phase_durations["finishing"] = time.perf_counter() - _t0
        _checkpoint_phase(ctx, result, "finishing")

    # Recompute now that metrics may have produced better numbers
    # (the delivered-suite coverage supersedes the coverage phase's
    # numbers).
    result.final_line_coverage = _final_line_coverage(result)
    result.final_mutation_score = _final_mutation_score(result)
    _restore_consolidation_stats(ctx, result)
    if ctx.reuse.finishing:
        _observe_restored_phase(ctx, "finishing", result)

    # ---- minimum-mutation floor ----
    # Same contract as the coverage floor: a kill rate below
    # ``minimum`` cannot back the correctness claim. It judges the suite
    # the mutation phase end delivered, by Finishing's full-catalogue
    # score (the score tests.csv reports), as the coverage floor judges
    # the coverage phase end's choice. No phase follows, so the floor only
    # records the reason.
    if result.below_minimum_reason is None:
        mut_floor_reason = _below_mutation_minimum(result)
        if mut_floor_reason:
            result.below_minimum_reason = mut_floor_reason
            ctx.log_print(f"[minimum] {mut_floor_reason}.", level=1)

    _requested_phase_stop(ctx, result, "finishing")

    return result


# =============================================================================
# RESUMABLE-RUN CHECKPOINT HOOKS
# =============================================================================

@contextlib.contextmanager
def _source_integrity_guard(ctx: TargetPhasesContext, phase_label: str):
    """Pin the copy's source against the canonical for a
    phase.

    Single invariant: ``copy/<file>`` must be byte-identical to
    ``canonical/<file>`` for the duration of coverage / mutation /
    metrics. Verified at phase entry and exit by direct
    byte comparison against canonical. On mismatch (whenever, by
    whatever cause), the canonical source is copied over the copy's
    file and a ``[CRITICAL]`` line is emitted so the operator knows
    something broke the contract. Exceptional exit with a retained cleanup
    owner defers restoration rather than writing under uncertain execution.

    The selected original is read-only input to this guard. Only the private
    copy's file is repaired. Original bytes and permissions are not changed.
    """
    work_path = Path(ctx.top_file["file_path"])
    canonical_path = Path(ctx.canonical_path)
    if not canonical_path.exists() or not work_path.exists():
        yield
        return

    def _verify_or_restore(phase_end: str) -> None:
        try:
            work_bytes = work_path.read_bytes()
            canon_bytes = canonical_path.read_bytes()
        except OSError:
            return
        if work_bytes == canon_bytes:
            return
        try:
            work_path.write_bytes(canon_bytes)
        except OSError as exc:
            from testuary._progress import diagnostic_print

            diagnostic_print(
                f"[CRITICAL] {phase_label} {phase_end}: "
                f"the work copy {work_path} differs from the original "
                f"{canonical_path}, and copying the original back failed ({exc!r}). "
                f"The work copy is now in an unknown state.",
                flush=True,
            )
            return
        from testuary._progress import diagnostic_print

        diagnostic_print(
            f"[CRITICAL] {phase_label} {phase_end}: "
            f"the work copy {work_path} differs from the original and was "
            f"restored from it. Treat results from this phase as suspect.",
            flush=True,
        )

    _verify_or_restore("entry")
    try:
        yield
    except BaseException as exc:
        if not mutation.mutation_cleanup_pending(exc):
            _verify_or_restore("exit")
        raise
    else:
        _verify_or_restore("exit")


def _phase_segment(
    ctx: TargetPhasesContext, phase: str, *, suites_reused: bool = False,
):
    """Time one attempt at ``phase``, durably.

    The elapsed time lands in the phase's own checkpoint block, so it is
    discarded by whatever discards the artifact it describes and a resumed
    run reports the work that produced what it holds rather than the tail of
    it. The worker count travels with the number because a mutation phase's
    wall-clock is only comparable against another measured with the same
    parallelism."""
    workers = None
    if phase == "mutation":
        try:
            workers = int(
                config_module.load_config("test_mutation")["engine_workers"])
        except (KeyError, TypeError, ValueError):
            workers = None
    return _named_segment(phase, phase_state.timed_segment(
        ctx.results_dir / ctx.workspace, phase,
        run_idx=ctx.run_idx,
        engine_workers=workers, suites_reused=suites_reused,
    ))


@contextlib.contextmanager
def _named_segment(phase: str, segment):
    """``segment`` with ``phase`` named as the phase a limit reached in a
    shared measurement belongs to."""
    from testuary import results

    with results.in_phase(phase), segment as value:
        yield value


def _record_phase_restored(ctx: TargetPhasesContext, phase: str) -> None:
    """Record that this run reused ``phase`` and did no work, so the cost
    it carries is visibly not this run's."""
    try:
        phase_state.record_restored(
            ctx.results_dir / ctx.workspace, phase, run_idx=ctx.run_idx)
    except (OSError, ValueError):
        # Provenance is worth having and never worth failing a reuse for.
        pass


def _observe_restored_phase(ctx, phase, result):
    if ctx.progress is not None:
        from testuary.presentation import Metric

        scores = _phase_result_scores(result, phase)
        names = {
            "starter": (("line coverage", "baseline_line"), ("branch coverage", "baseline_branch")),
            "coverage": (("line coverage", "coverage"), ("branch coverage", "branch_coverage")),
            "mutation": (("mutation score", "mutation_score"),),
            "finishing": (("line coverage", "final_line_coverage"),
                          ("branch coverage", "final_branch_coverage"),
                          ("mutation score", "final_mutation_score")),
        }
        metrics = []
        for name, key in names[phase]:
            value = scores.get(key)
            try:
                metric = Metric(name, value, "%", "reused" if value is not None else "unavailable")
            except (TypeError, ValueError):
                metric = Metric(name, unit="%", basis="failed")
            metrics.append(metric)
        target = _display_target(ctx)
        if phase == "finishing":
            # Finishing's events are at the step level (_display_phase).
            from testuary.presentation.view import FINISHING, phase_label

            shape = dict(operation=FINISHING, level="step", label=phase_label(phase))
        else:
            shape = dict(operation=f"restore/run-{ctx.run_idx}", level="phase")
        ctx.progress.emit(target=target, phase=phase, state="succeeded",
                          message="Reused recorded phase evidence",
                          metrics=tuple(metrics), provenance="recorded", **shape,
                          **ctx.progress.phase_counts(target, ctx.run_idx, finished=True))
        _report_run(ctx.progress)


def _reuses_suites(ctx: TargetPhasesContext) -> bool:
    """Whether the starter phase reuses the saved starter suites: when
    reuse_suites asks for it, and on every resume, so a resumed starter phase
    reuses what it generated and a slot keeps the attempts it spent. A
    resume never buys a starter again."""
    return config_module.get_agent_config("test_initial")["reuse_suites"] or ctx.resume


def _reuses_starter_pool(ctx: TargetPhasesContext) -> bool:
    """Whether the starter phase reuses a saved starter pool: it reuses
    saved starters (``_reuses_suites``) and a pool was saved."""
    if not _reuses_suites(ctx):
        return False
    return starters._has_persisted_pool(_run_dir(ctx) / "archive" / "starter" / "suites")


def _context_generation_condition(ctx: TargetPhasesContext, phase: str):
    from testuary.evidence import current_condition

    target = str(ctx.top_file["file_path"])
    return current_condition(phase, languages.detect_language(target), project.resolve_source_dir(target))


def _guard_generation_reuse(ctx: TargetPhasesContext) -> None:
    from testuary.evidence import generation_only, phase_origin_path, require_origin

    target_dir = ctx.results_dir / ctx.workspace
    run_dir = _run_dir(ctx)
    if _reuses_suites(ctx):
        suites = run_dir / "archive" / "starter" / "suites"
        if starters._has_persisted_pool(suites):
            require_origin(suites / "generation_origin.json",
                           generation_only(_context_generation_condition(ctx, "starter")))
    for phase in ("starter", "coverage", "mutation"):
        if not getattr(ctx.reuse, phase):
            continue
        # Mutation may be disabled on a fresh run, without reusing evidence.
        if phase == "mutation" and not (_restore_run_dir(ctx, phase) / "archive" / phase).exists():
            continue
        # A reuse restores the first run's suites and checkpoint
        # observations, a resume this run's.
        require_origin(phase_origin_path(target_dir, phase, _restore_run_idx(ctx, phase)),
                       _context_generation_condition(ctx, phase))


def _generation_metadata(ctx: TargetPhasesContext, phase: str) -> Dict[str, Any]:
    """The generation condition a phase's checkpoint records, when there is one."""
    if phase not in {"starter", "coverage", "mutation"}:
        return {}
    from testuary.evidence import phase_origin_path, recorded_condition

    condition = recorded_condition(
        phase_origin_path(ctx.results_dir / ctx.workspace, phase, ctx.run_idx))
    return {"generation_condition": condition} if condition is not None else {}


def _record_start(ctx: TargetPhasesContext, phase: str) -> None:
    """Mark ``phase`` begun in the target's phase_state.json, so a crash
    in it is redone, never trusted. The record names the target by its
    project-relative path (``display_target``), so renumbered targets keep
    their checkpoints, and hashes its source, so tests of an older source are
    stale. The run's own block (``run_idx``) holds it."""
    phase_state.mark_started(
        ctx.results_dir / ctx.workspace, phase,
        target_number=ctx.top_file["target_number"], stem=ctx.canonical_path.stem,
        target_file_path=ctx.display_target, canonical_source_path=ctx.canonical_path,
        run_idx=ctx.run_idx, **_generation_metadata(ctx, phase))


def _record_done(ctx: TargetPhasesContext, phase: str, success: bool,
                 artifacts: Dict[str, str], metrics: Dict[str, Any]) -> None:
    """Record ``phase``'s end in phase_state.json: a failure with its
    reason, else each artifact (a path in the copy under its dir-relative
    key) copied into the run's results directory first and the
    checkpoint, which hashes the copies, last. ``metrics["repoint"]`` moves an
    earlier phase's checkpoint onto the original the merge preserved."""
    target_results_dir = ctx.results_dir / ctx.workspace
    run_idx = ctx.run_idx
    if not success:
        reason = (metrics or {}).get("reason") or "no usable artifact"
        phase_state.mark_failed(
            target_results_dir, phase, reason=str(reason),
            run_idx=run_idx, failure_kind=(metrics or {}).get("failure_kind"),
        )
        return
    repoint = (metrics or {}).pop("repoint", None) or []
    durable: Dict[str, Path] = {}
    for rel, work_path in artifacts.items():
        durable_rel = _artifact_rel_for_run(rel, run_idx)
        dest = target_results_dir / durable_rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(work_path, dest)
        durable[durable_rel] = dest
    phase_state.mark_complete(
        target_results_dir, phase,
        artifacts=durable, metrics=metrics,
        run_idx=run_idx,
        **_generation_metadata(ctx, phase),
    )
    for sp_name, old_rel, new_rel in repoint:
        new_abs = target_results_dir / _artifact_rel_for_run(new_rel, run_idx)
        if new_abs.exists():
            phase_state.repoint_artifact(
                target_results_dir, sp_name,
                old_rel=_artifact_rel_for_run(old_rel, run_idx),
                new_rel=_artifact_rel_for_run(new_rel, run_idx),
                new_abspath=new_abs, run_idx=run_idx,
            )


def _artifact_rel_for_run(rel: str, run_idx: int) -> str:
    """A phase's relative artifact key inside its run's directory:
    ``tests/test_foo.py`` -> ``r0/tests/test_foo.py`` for the first run,
    ``r1/tests/test_foo.py`` for the second. A key already inside a run
    directory stays as it is."""
    if not rel:
        return rel
    head = rel.split("/", 1)[0]
    if len(head) > 1 and head[0] == "r" and head[1:].isdigit():
        return rel
    return f"{project.run_name(run_idx)}/{rel}"


def _start_phase(ctx: TargetPhasesContext, phase: str) -> None:
    """Record the generation origin of a starter, coverage or mutation
    attempt, mark the phase begun (``_record_start``) and log an explicit
    phase log line, so the operator log shows every phase start."""
    started = time.perf_counter()
    if phase in ("starter", "coverage", "mutation"):
        from testuary.evidence import copy_origin, phase_origin_path, require_origin, write_origin

        origin = phase_origin_path(ctx.results_dir / ctx.workspace, phase, ctx.run_idx)
        condition = _context_generation_condition(ctx, phase)
        suites = _run_dir(ctx) / "archive" / "starter" / "suites"
        if phase == "starter" and _reuses_starter_pool(ctx):
            if origin.exists():
                require_origin(origin, condition)
            else:
                copy_origin(suites / "generation_origin.json", origin, condition)
        else:
            write_origin(origin, condition)
    log_print = ctx.log_print
    origin_seconds = time.perf_counter() - started
    log_phase_started(log_print, phase)
    marked = time.perf_counter()
    _record_start(ctx, phase)
    shown = time.perf_counter()
    _display_phase(ctx, phase, "running", message="Phase started")
    _log_boundary_steps(log_print, phase, "started", (
        ("origin", origin_seconds), ("checkpoint", shown - marked),
        ("display", time.perf_counter() - shown)))


def _log_boundary_steps(log_fn, phase: str, event: str, steps) -> None:
    """Log the time of each step that runs between two phases (the
    checkpoint writes, the generation origin, the display), so no time
    between a phase's END and the next phase's START goes unattributed in
    the log."""
    parts = ", ".join(f"{name} {seconds:.1f}s" for name, seconds in steps)
    log_fn(f"[pipeline] {phase} {event}: {parts}")


def _display_target(ctx) -> str:
    return getattr(ctx, "display_target", "") or ctx.workspace


def _report_run(progress) -> None:
    report = getattr(progress, "report_run", None)
    if report is not None:
        report()


def _metric_goal(phase: str, name: str) -> Optional[float]:
    """The configured goal of a metric the editing phase works towards."""
    try:
        if phase == "coverage" and name in ("line coverage", "branch coverage"):
            config = config_module.get_agent_config("test_coverage")
            selected = str(config["goal_metric"]).strip().lower()
            if selected in (name.split()[0], "both", "either"):
                return float(config["goal"])
        if phase == "mutation" and name == "mutation score":
            return float(config_module.get_agent_config(
                "test_mutation", parent="test_coverage")["goal"])
    except (KeyError, TypeError, ValueError):
        return None
    return None


def _display_phase(ctx, phase, state, *, message="", metrics=None) -> None:
    progress = ctx.progress
    if progress is None:
        return
    from testuary.presentation import Metric
    from testuary.presentation.view import phase_label

    observations = []
    values = metrics or {}
    names = {
        "starter": (("line coverage", "baseline_line"), ("branch coverage", "baseline_branch")),
        "coverage": (("line coverage", "coverage"), ("branch coverage", "branch_coverage")),
        "mutation": (("mutation score", "mutation_score"),),
        "finishing": (("line coverage", "final_line_coverage"), ("branch coverage", "final_branch_coverage"),
                    ("mutation score", "final_mutation_score")),
    }
    for name, key in names.get(phase, ()):
        value = values.get(key) if state == "succeeded" else None
        goal = _metric_goal(phase, name)
        try:
            metric = Metric(name, value=value, unit="%",
                            basis="measured" if value is not None else "unavailable", goal=goal)
        except (TypeError, ValueError):
            metric = Metric(name, unit="%", basis="failed", goal=goal)
        observations.append(metric)
    target = _display_target(ctx)
    if phase == "finishing":
        # The fourth phase, in its step-level events, counted with the
        # phases.
        from testuary.presentation.view import FINISHING

        progress.emit(target=target, phase=phase, operation=FINISHING, state=state,
                      message=message, metrics=tuple(observations), level="step",
                      label=phase_label(phase),
                      **progress.phase_counts(target, ctx.run_idx,
                                      finished=state == "succeeded"))
    else:
        progress.emit(target=target, phase=phase,
                      operation=f"phase/run-{ctx.run_idx}", state=state,
                      message=message, metrics=tuple(observations), level="phase",
                      **progress.phase_counts(target, ctx.run_idx,
                                      finished=state == "succeeded"))
    if state == "succeeded":
        _report_run(progress)


def _phase_snapshots_dir(ctx: TargetPhasesContext) -> Path:
    """The directory holding each phase's immutable output snapshot, a sibling
    of ``tests/`` so test collection never sees it. Deriving it from
    ``work_tests`` keeps it in the same tree as the suite under both local mode
    (the work dir) and container mode (the bind-mounted work directory), so the
    in-container measurement can reach it at the matching container path."""
    return Path(ctx.work_tests).parent / "originals"


def _archive_path(ctx: TargetPhasesContext, phase: str) -> Path:
    """The durable per-target archive subdirectory for a phase, under
    ``results/<ws>/r<k>/archive/<phase>/`` for run k. The matrices grid, the keep-best loss
    dumps, and the starter pool live here rather than under ``logs/test/<ws>/``
    so nothing load-bearing lives under the delete-on-rerun logs semantics. ``logs/``
    then holds only the chat / usage / trace logs. The directory names the phase,
    so filenames inside do not repeat it (``merged_kill_matrix.json``, not
    ``<phase>_union_kill_matrix.json``). A read takes this path alone, so a
    phase that did not run gets no directory."""
    return _run_dir(ctx) / "archive" / phase


def _archive_dir(ctx: TargetPhasesContext, phase: str) -> Path:
    """A phase's archive directory (``_archive_path``), made for a writer."""
    d = _archive_path(ctx, phase)
    d.mkdir(parents=True, exist_ok=True)
    return d


def _verify_restored_phase_validity(
    ctx: TargetPhasesContext, phase: str, source_path: Path,
) -> tuple[bool, str]:
    """Require a current source-bound validity record before phase reuse."""
    target = str(ctx.top_file["file_path"])
    backend = languages.get_backend(languages.detect_language(target))
    if backend is None:
        return False, "no backend can verify restored generated-test validity"
    try:
        source = source_path.read_text(encoding="utf-8")
        identity = backend.generated_test_target_identity(
            target, project.resolve_source_dir(target),
        )
        return validity.verify_delivered_archive(
            archive_dir=_archive_dir(ctx, phase),
            source=source,
            identity=identity,
        )
    except Exception as exc:
        return False, f"validity verification raised {type(exc).__name__}: {exc}"


def _canonical_suite_name(ctx: TargetPhasesContext) -> Tuple[str, str]:
    """The canonical coverage suite's stem and suffix, off which every
    per-phase snapshot is named so the set is uniform and language-agnostic."""
    backend = languages.get_backend(
        languages.detect_language(str(ctx.canonical_path))
    )
    cov = backend.target_test_path(
        str(ctx.work_tests), str(ctx.canonical_path),
    )
    return cov.stem, cov.suffix


def _snapshot_phase_original(
    ctx: TargetPhasesContext,
    artifacts: Dict[str, str],
    phase: str,
    src_path: Path,
) -> None:
    """Copy a phase's just-finished output suite to an immutable
    ``originals/<stem>.<phase><suffix>`` snapshot and register it for durable
    extraction. Fires only on the run path (the checkpoint skips it on a
    resumed phase), so the snapshot captures exactly what this phase produced
    before a later phase overwrites the canonical file."""
    src = Path(src_path)
    if not src.exists():
        return
    stem, suffix = _canonical_suite_name(ctx)
    dest = measurement.phase_original_path(
        _phase_snapshots_dir(ctx), stem, suffix, phase,
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(str(src), str(dest))
    # Preserve the exact selected bytes. The consolidation record binds its
    # measurements to this source hash, so archive cleanup must not rewrite the
    # snapshot after selection.
    # Two-tree: the staging copy stays in the target's copy, in ``originals/``, so the
    # in-container measurement reaches it at the matching container
    # path. The durable home is the phase's archive directory, ``archive/<phase>/
    # result<suffix>`` (dir-clean: the phase is the directory).
    artifacts[f"archive/{phase}/result{suffix}"] = str(dest)


def _phase_result_scores(result: TestPipelineResult, phase: str) -> Dict[str, Any]:
    """Read completed observations without checkpointing or measuring."""
    fields = {
        "starter": ("starter_result", (
            ("baseline_line", "baseline_line_coverage"),
            ("baseline_branch", "baseline_branch_coverage"))),
        "coverage": ("test_result", (
            ("coverage", "line_coverage"),
            ("branch_coverage", "branch_coverage"))),
        "mutation": ("mutation_result", (("mutation_score", "mutation_score"),)),
        "finishing": ("finishing_result", (
            ("final_line_coverage", "coverage_final_line"),
            ("final_branch_coverage", "coverage_final_branch"))),
    }
    source, attributes = fields.get(phase, ("", ()))
    measured = getattr(result, source, None)
    scores = {key: getattr(measured, attribute, None) for key, attribute in attributes}
    if phase == "finishing":
        scores["final_mutation_score"] = _final_mutation_shown(result)
    return scores


def _final_mutation_shown(result: TestPipelineResult) -> Optional[float]:
    """The delivered suite's full-catalogue mutation score from the final
    measurement: the run log's "final standard suite" score, which the
    mutation minimum judges. None when the measurement did not produce one."""
    finishing = getattr(result.finishing_result, "finishing", None)
    return getattr(finishing, "mutation_score_full", None)


def _checkpoint_phase(
    ctx: TargetPhasesContext, result: TestPipelineResult, phase: str,
) -> None:
    """Record the phase's end (``_record_done``) with its success and artifacts.

    Success is read from ``phase_outcomes``, the same signal the run
    uses to decide skip-downstream. Only the literal "ok"
    outcome (the phase ran AND produced a usable artifact) counts
    as success. "failed" / "skipped_*" do not, so a failed phase
    writes no checkpoint and is redone on the next resume.
    The artifact path is the file in the target's copy, copied into the results
    directory before the checkpoint is written.
    """
    _lp = ctx.log_print
    outcome = result.phase_outcomes.get(phase, "unknown")
    log_phase_finished(_lp, phase, outcome=outcome)
    success = outcome == "ok"
    artifacts: Dict[str, str] = {}
    metrics: Dict[str, Any] = {}
    # Failed-path reason is forwarded via metrics["reason"]. The
    # callback uses it as the ``failed`` record's reason string so
    # phase_state.json shows ``status=failed`` with diagnostic
    # context instead of an indefinite ``in_progress``.
    if not success:
        outcome_reason = result.phase_outcomes.get(phase, "unknown")
        detail = result.phase_outcomes.get(f"{phase}_detail")
        metrics["reason"] = (
            f"{outcome_reason}: {detail}" if detail else str(outcome_reason)
        )
        failure_kind = result.phase_outcomes.get(f"{phase}_failure_kind")
        if failure_kind is not None:
            metrics["failure_kind"] = failure_kind
    if success:
        if phase == "starter" and result.starter_result is not None:
            ir = result.starter_result
            if ir.baseline_file and Path(ir.baseline_file).exists():
                artifacts[f"tests/{Path(ir.baseline_file).name}"] = (
                    ir.baseline_file
                )
                _snapshot_phase_original(
                    ctx, artifacts, "starter", Path(ir.baseline_file),
                )
            metrics = {
                "baseline_line": ir.baseline_line_coverage,
                "baseline_branch": ir.baseline_branch_coverage,
                "suites_passing": ir.n_green,
                "suite_line_stats": dict(ir.suite_line_stats),
                "suite_branch_stats": dict(ir.suite_branch_stats),
                "suite_mutation_stats": dict(ir.suite_mutation_stats),
                "suite_mutation_method": ir.suite_mutation_method,
                # The baseline handoff a resumed run rebuilds from this
                # checkpoint: the coverage agent seeds best-green with these and
                # the phase-end consolidation merges against the snapshot.
                "baseline_suite_seconds": ir.baseline_suite_seconds,
                "baseline_missing_lines": (
                    None if ir.baseline_missing_lines is None
                    else list(ir.baseline_missing_lines)),
                "baseline_missing_branches": (
                    None if ir.baseline_missing_branches is None
                    else list(ir.baseline_missing_branches)),
                "selected_tests": ir.selected_tests,
            }
        elif phase == "coverage" and result.test_file_name:
            p = ctx.work_tests / result.test_file_name
            if p.exists():
                artifacts[f"tests/{result.test_file_name}"] = str(p)
                _snapshot_phase_original(ctx, artifacts, "coverage", p)
            metrics = {
                "coverage": getattr(
                    result.test_result, "line_coverage", None,
                ),
                "branch_coverage": getattr(
                    result.test_result, "branch_coverage", None,
                ),
                **_usage_fields(result.test_result),
                **_restored_fields(result.test_result, "coverage"),
                # The coverage agent extends the starter baseline in
                # place, so the starter phase's checkpoint points at an overwritten
                # file. Re-point it at the coverage artifact the
                # baseline was merged into, re-hashed, so a resumed run
                # validates both.
                "repoint": [
                    ("starter", f"tests/{result.test_file_name}",
                     f"tests/{result.test_file_name}"),
                ],
            }
        elif phase == "mutation":
            mr = result.mutation_result
            merge_result = result.mutation_merge_result
            if merge_result is not None:
                # Both agents revise one delivered file. Consolidation retains
                # the two logical input sources outside the collectable test
                # directory and writes the selected source back to that file.
                merged = Path(merge_result.merged_path)
                if merged.exists():
                    artifacts[f"tests/{merged.name}"] = str(merged)
                    _snapshot_phase_original(ctx, artifacts, "mutation", merged)
                inherited_pre = Path(merge_result.pre_merge_coverage_path)
                for pre in (
                    merge_result.pre_merge_coverage_path,
                    merge_result.pre_merge_mutation_path,
                ):
                    pp = Path(pre)
                    if pp.exists():
                        artifacts[
                            f"archive/mutation/pre_merge/{pp.name}"
                        ] = str(pp)
                # The starter and coverage checkpoints point at the tests/ file the
                # mutation phase revised in place. Re-point them at the exact
                # inherited source retained outside test collection.
                metrics["repoint"] = [
                    ("starter", f"tests/{merged.name}",
                     f"archive/mutation/pre_merge/{inherited_pre.name}"),
                    ("coverage", f"tests/{merged.name}",
                     f"archive/mutation/pre_merge/{inherited_pre.name}"),
                ]
            else:
                # A framework stop may have no merge, but the mutation agent
                # still uses the one delivered suite path.
                p = ctx.work_tests / (result.test_file_name or "")
                if p.exists():
                    artifacts[f"tests/{p.name}"] = str(p)
                else:
                    # A framework stop can leave no standalone mutation file.
                    # In that case the phase archive is the materialized
                    # boundary artifact.
                    _, suffix = _canonical_suite_name(ctx)
                    phase_end = (
                        _archive_path(ctx, "mutation") / f"result{suffix}"
                    )
                    if phase_end.exists():
                        artifacts[f"archive/mutation/result{suffix}"] = str(
                            phase_end
                        )
            # Persist the score regardless of whether the suite was
            # consolidated: a resumed run RESTORES this phase rather than
            # running it, and _final_mutation_score reads only
            # result.mutation_result.mutation_score (Finishing's mutation
            # measurement does not feed it), so without this a
            # resumed target records a null mutation score.
            if mr is not None and mr.mutation_score is not None:
                metrics["mutation_score"] = mr.mutation_score
                metrics["mutation_file"] = mr.mutation_test_file_name
                metrics.update(_usage_fields(mr))
                metrics.update(_restored_fields(mr, "mutation"))
        elif phase == "finishing":
            # Finishing has run by the time this hook fires
            # (it is synchronous and the last phase). Read its object.
            m = result.finishing_result
            from testuary.evidence import finishing_settings

            metrics = {
                "final_line_coverage": getattr(m, "coverage_final_line", None),
                "final_branch_coverage": getattr(m, "coverage_final_branch", None),
                # A resume measures again when these settings changed.
                "finishing_settings": finishing_settings(),
            }
    recorded = time.perf_counter()
    _record_done(ctx, phase, success, artifacts, metrics)
    checkpoint_seconds = time.perf_counter() - recorded
    # The display also shows the final mutation score, which the recorded
    # metrics above do not carry.
    shown = ({**metrics, "final_mutation_score": _final_mutation_shown(result)}
             if phase == "finishing" else metrics)
    displayed = time.perf_counter()
    message = "Phase finished" if success else "Phase failed"
    loop_result = {"coverage": result.test_result, "mutation": result.mutation_result}.get(phase)
    if success and getattr(loop_result, "logical_requests", None) == 0:
        # The loop stopped at its first measurement: the suite it received
        # already met its goal.
        message = "The suite it received already met the goal, so no model request was needed"
    _display_phase(ctx, phase, "succeeded" if success else "failed",
                      message=message, metrics=shown)
    _log_boundary_steps(_lp, phase, "recorded", (
        ("checkpoint", checkpoint_seconds),
        ("display", time.perf_counter() - displayed)))


# =============================================================================
# STARTER AND COVERAGE PHASES
# =============================================================================

def _phase_starter(
    ctx: TargetPhasesContext, result: TestPipelineResult,
):
    """Run the starter phase. Records the phase outcome.
    A cancelled pool (zero green tests) is recorded as failed and the
    caller aborts the target."""
    target = str(ctx.top_file["file_path"])
    language = languages.detect_language(target)
    try:
        from testuary._progress import run_with_steps

        # The phase's steps are shown as it begins each, the last one ended
        # by how the phase ends.
        starter_result = run_with_steps(
            "starter", lambda done: done.status == "ok", starters.run_starter_phase,
            language=language,
            target_file_path=target,
            source_dir=project.resolve_source_dir(target),
            test_output_dir=str(ctx.work_tests),
            suites_dir=str(_archive_dir(ctx, "starter") / "suites"),
            matrices_dir=str(_archive_dir(ctx, "starter")),
            log=ctx.log_print or print,
            run_idx=ctx.run_idx,
            reuse_suites=_reuses_suites(ctx),
            **({"prepared_models": ctx.prepared_models} if ctx.prepared_models is not None else {}),
        )
    except NotImplementedError as exc:
        _record_outcome(result, "starter", "failed", str(exc),
                        failure_kind=phase_state.INFRASTRUCTURE_FAILURE)
        return None
    if starter_result.status != "ok":
        _record_outcome(result, "starter", "failed", starter_result.detail,
                        failure_kind=_STARTER_FAILURE_KINDS.get(starter_result.status))
    else:
        _record_outcome(result, "starter", "ok", None)
    return starter_result


def _place_restored_starter(
    ctx: TargetPhasesContext, ir: "starters.StarterResult",
) -> None:
    """Put the restored starter suite where the coverage phase reads, edits
    and prunes it: the copy's suite file, written atomically with the
    archive snapshot's bytes. The archive stays as the starter phase
    delivered it, and ``ir.baseline_file`` names the work file, as after a
    starter phase that ran."""
    backend = languages.get_backend(languages.detect_language(str(ctx.canonical_path)))
    work = backend.target_test_path(str(ctx.work_tests), str(ctx.canonical_path))
    work.parent.mkdir(parents=True, exist_ok=True)
    temporary = work.with_name(f".{work.name}.{uuid.uuid4().hex}.tmp")
    try:
        shutil.copyfile(ir.baseline_file, temporary)
        temporary.replace(work)
    finally:
        temporary.unlink(missing_ok=True)
    ir.baseline_file = str(work)


def _restore_starter_baseline(
    ctx: TargetPhasesContext,
) -> Optional["starters.StarterResult"]:
    """Restore a reusable starter suite from its durable result evidence.

    Read source from the archived snapshot, measurements from its checkpoint,
    generation accounting from the suite manifest, and counts and selection from
    the consolidation record. Reject absent or mismatched evidence without relying
    on an earlier CSV row."""
    _, suffix = _canonical_suite_name(ctx)
    snap = _restore_run_dir(ctx, "starter") / "archive" / "starter" / f"result{suffix}"
    if not snap.exists():
        raise RuntimeError(
            f"no starter snapshot is available for the coverage handoff: {snap}"
        )
    valid, validity_reason = _verify_restored_phase_validity(
        ctx, "starter", snap,
    )
    if not valid:
        raise RuntimeError(
            "the restored starter suite does not match its validation record: "
            f"{validity_reason}"
        )
    metrics: Dict[str, Any] = {}
    state: Dict[str, Any] = {}
    try:
        state = phase_state.load(ctx.results_dir / ctx.workspace)
        metrics = phase_state.phase_metrics(
            state, "starter", run_idx=_restore_run_idx(ctx, "starter"))
    except Exception:
        pass
    phase_dir = snap.parent
    record = _read_json(phase_dir / "consolidation_record.json")
    if not record:
        raise RuntimeError(
            "restored starter suite has no consolidation record"
        )
    manifest = _read_json(phase_dir / "suites" / "_generation.json")
    if not manifest:
        raise RuntimeError(
            "restored starter suite has no generation manifest"
        )

    def required_int(mapping: Dict[str, Any], key: str, source: str) -> int:
        value = mapping.get(key)
        if isinstance(value, bool):
            value = None
        try:
            parsed = int(value)
        except (TypeError, ValueError):
            parsed = -1
        if parsed < 0:
            raise RuntimeError(
                f"restored starter {source} has no non-negative {key}"
            )
        return parsed

    ir = starters.StarterResult(status="ok")
    ir.baseline_file = str(snap)
    ir.baseline_line_coverage = metrics.get("baseline_line")
    ir.baseline_branch_coverage = metrics.get("baseline_branch")
    ir.baseline_suite_seconds = metrics.get("baseline_suite_seconds")
    for key in ("baseline_missing_lines", "baseline_missing_branches"):
        if metrics.get(key) is not None:
            setattr(ir, key, tuple(metrics[key]))
    ir.suite_line_stats = dict(metrics.get("suite_line_stats") or {})
    ir.suite_branch_stats = dict(metrics.get("suite_branch_stats") or {})
    ir.suite_mutation_stats = dict(metrics.get("suite_mutation_stats") or {})
    if not isinstance(metrics.get("suite_mutation_method"), str):
        raise RuntimeError("restored starter checkpoint has no suite_mutation_method")
    ir.suite_mutation_method = metrics["suite_mutation_method"]
    # A phase end that counted no merged suite records null.
    ir.pool_tests = (None if record.get("merged_tests") is None
                     else required_int(record, "merged_tests", "consolidation record"))
    ir.selected_tests = required_int(
        record, "delivered_tests", "consolidation record"
    )
    checkpoint_selected = metrics.get("selected_tests")
    if checkpoint_selected is not None and int(checkpoint_selected) != ir.selected_tests:
        raise RuntimeError(
            "restored starter selected-test count disagrees between the "
            "checkpoint and consolidation record"
        )
    ir.winner = str(record.get("delivered_from") or "")
    if not ir.winner:
        raise RuntimeError("restored starter consolidation record has no winner")
    ir.baseline_from_single_suite = ir.winner == "single_suite"
    ir.n_green = required_int(metrics, "suites_passing", "checkpoint")
    ir.n_generated = required_int(manifest, "n_generated", "generation manifest")
    ir.requests_used = required_int(
        manifest, "requests_used", "generation manifest"
    )
    ir.input_tokens = required_int(
        manifest, "input_tokens", "generation manifest"
    )
    ir.cache_hit_tokens = required_int(
        manifest, "cache_hit_tokens", "generation manifest"
    )
    ir.output_tokens = required_int(
        manifest, "output_tokens", "generation manifest"
    )
    ir.generation_seconds = float(manifest.get("generation_seconds") or 0.0)
    # Request accounting: a manifest without it leaves it unknown rather
    # than zero.
    for key in ("logical_requests", "unknown_usage_requests",
                "unknown_cache_requests"):
        value = manifest.get(key)
        setattr(ir, key, int(value) if value is not None else None)
    ir.suite_generation_attempts = {
        int(key): int(value)
        for key, value in (
            manifest.get("generation_attempts_per_suite") or {}
        ).items()
    }
    # The passing suites that left without complete numbers: a manifest
    # written before suites could leave leaves them unknown, not zero.
    for key in ("n_unmeasured_coverage", "n_unmeasured_mutation"):
        value = manifest.get(key)
        setattr(ir, key, int(value) if value is not None else None)
    ir.duration_s = phase_state.output_seconds(
        ctx.results_dir / ctx.workspace, "starter", run_idx=_restore_run_idx(ctx, "starter"))
    return ir


def _baseline_handoff(result: TestPipelineResult) -> dict:
    """Return the required starter suite as the coverage handoff."""
    ir = result.starter_result
    if ir is None:
        raise RuntimeError("the coverage phase requires the suite handed over by the starter phase")
    if not getattr(ir, "baseline_file", None):
        raise RuntimeError(
            "the starter phase produced no suite for the coverage phase"
        )
    try:
        return {
            "content": Path(ir.baseline_file).read_text(encoding="utf-8"),
            "line_coverage": ir.baseline_line_coverage,
            "branch_coverage": ir.baseline_branch_coverage,
            "suite_seconds": ir.baseline_suite_seconds,
            "test_count": ir.selected_tests,
            "missing_lines": getattr(ir, "baseline_missing_lines", None),
            "missing_branches": getattr(ir, "baseline_missing_branches", None),
        }
    except OSError as exc:
        raise RuntimeError(
            f"the starter phase's suite at {ir.baseline_file} could not be "
            f"read: {type(exc).__name__}: {exc}"
        ) from exc


def _coverage_measurement_identity(target: str, execution_name: str, mut_cfg) -> Dict[str, Any]:
    """The measurement identity of the coverage phase end's mutation results
    for the suite ``execution_name`` under the [mutation] settings ``mut_cfg``."""
    return mutation_agent.mutation_measurement_identity(
        target_file_path=target, language=languages.detect_language(target),
        execution_name=execution_name, config=dict(mut_cfg),
        attribution=measurement.union_attribution(mut_cfg))


def _refuse_foreign_starter_measurement(ctx: TargetPhasesContext) -> None:
    """Refuse, before the coverage loop sends a request, a starter phase whose
    delivered mutation results the coverage phase end could not inherit:
    measured for another target source, file name, catalogue or settings,
    such as by another Testuary version. The phase end logs an unusable
    matrix, so this read logs nothing."""
    archive = _archive_path(ctx, "starter")
    record = _read_json(archive / "consolidation_record.json")
    if record is None or _delivered_kill_matrix(archive, lambda *a, **k: None) is None:
        return
    mut_cfg = config_module.get_agent_config("test_mutation", parent="test_coverage")
    identity = _coverage_measurement_identity(
        str(ctx.top_file["file_path"]), record.get("execution_name"), mut_cfg)
    if record.get("measurement_identity") != identity:
        raise RuntimeError("the mutation results the coverage phase inherited were measured "
                           "for another target source, file name, catalogue or settings")


def _consolidate_coverage(
    ctx: TargetPhasesContext,
    result: TestPipelineResult,
    baseline_content: str,
) -> None:
    """The post-coverage consolidation: union the agent's edited suite
    with the baseline it started from, apply set cover, and select among the
    reduced suite, merged suite, and both inputs by the shared score-first rule.
    Evidence-identity differences are recorded for audit, not used as a gate."""
    from testuary.consolidation import (
        InheritedMatrix, PhaseEnd, PhaseEndOptions, ScoredSuite, _coverage_sets, run_phase_end,
    )
    log_print = ctx.log_print

    cov_path = ctx.work_tests / result.test_file_name
    if not cov_path.exists():
        return
    cov_src = cov_path.read_text(encoding="utf-8")
    # Preserve the agent's suite before consolidation can replace it. The phase
    # snapshot is taken only after this step. Per-target under the coverage
    # archive, so two targets with the same file name cannot overwrite each
    # other.
    try:
        (_archive_dir(ctx, "coverage")
         / f"candidate_agent{cov_path.suffix}").write_text(
            cov_src, encoding="utf-8")
    except OSError:
        pass
    target = str(ctx.top_file["file_path"])
    language = languages.detect_language(target)
    backend = languages.get_backend(language)
    source_dir = project.resolve_source_dir(target)
    cfg = config_module.get_agent_config("test_coverage")
    timeout = int(cfg["suite_run_timeout_seconds"])

    # The mutation phase reuses this consolidation's matrix as its baseline, but
    # only when it is exactly the delivered suite's. Clear any stale file up front.
    # It is rewritten below when the winning candidate, whichever it is, has
    # a standalone matrix.
    baseline_matrix_path = ctx.work_tests / ".mutation_baseline_matrix.json"
    try:
        baseline_matrix_path.unlink()
    except OSError:
        pass

    # Measure the merged suite (the agent's edited suite merged with the baseline
    # it started from): full-catalogue mutation, when on, supplies kill tokens
    # to the set cover. The session stays open for the optional sanity
    # re-measure. Nullcontext yields a None runner when consolidation mutation is
    # off, keeping one code path.
    mut_cfg = config_module.get_agent_config("test_mutation", parent="test_coverage")
    phase_measurement_identity = _coverage_measurement_identity(
        target, result.test_file_name, mut_cfg)
    runner_cm = measurement.phase_end_runner(
        "coverage", language=language, target_file=target,
        source_dir=source_dir, config=dict(mut_cfg),
        log_fn=lambda m: log_print(m, level=1),
    )
    # The baseline's per-test tokens ride into the merge from its own
    # boundary's persisted sidecar: the collision victim choice among
    # un-renamable owners reads measured evidence for BOTH candidates
    # (candidate 2's standalone measurement runs before the merge inside
    # consolidate_phase_end).
    baseline_tokens = None
    starter_record = None
    starter_reusable_km = None
    starter_coverage = None
    try:
        starter_archive = _archive_path(ctx, "starter")
        starter_record = _read_json(
            starter_archive / "consolidation_record.json"
        )
        starter_coverage = _delivered_coverage(starter_archive, ctx.log_print)
        starter_reusable_km = _delivered_kill_matrix(starter_archive, ctx.log_print)
        if starter_coverage:
            baseline_tokens = starter_coverage.get("per_test_tokens") or None
    except OSError:
        pass
    baseline_hash = source_sha256(baseline_content)
    if (
        starter_record is None
        or starter_record.get("selected_source_sha256") != baseline_hash
        or starter_record.get("execution_name") != result.test_file_name
    ):
        raise RuntimeError(
            "the coverage phase's inherited suite does not match the "
            "starter phase's consolidation record"
        )
    inherited_identity = starter_record.get("measurement_identity")
    from testuary.agent import _helpers as agent_helpers

    generation_scope, generation_inputs = _editing_generation_inputs(
        ctx, "coverage", predecessor_record=starter_record,
        inherited_source=baseline_content, agent_source=cov_src,
        agent_result=result.test_result,
    )

    # Evidence reuse. The agent's suite is byte-identical to the inherited
    # one, the starter phase delivered those bytes under this execution name
    # with this measurement identity, and its evidence is complete and
    # bound: the phase end takes that evidence over and makes no execution,
    # measurement, reduction or selection call. A goal met at entry and an
    # agent that changed nothing are the same case. Any shortfall takes the
    # normal path below, with the reason logged.
    if cfg["phase_end_evidence_reuse"] and cov_src == baseline_content:
        from testuary.consolidation import inherited_phase_end_eligibility
        evidence, reason = inherited_phase_end_eligibility(
            source_phase_dir=_archive_dir(ctx, "starter"),
            delivered_source=cov_src, execution_name=result.test_file_name,
            measurement_identity=phase_measurement_identity,
            require_mutation=config_module.consolidation_measure_mutation(),
            target_file=target, source_dir=source_dir,
        )
        if evidence is not None:
            _reuse_coverage_phase_end(
                ctx, result, evidence=evidence, delivered=cov_src,
                target=target, source_dir=source_dir, language=language,
                measurement_identity=phase_measurement_identity,
                mut_cfg=mut_cfg, baseline_matrix_path=baseline_matrix_path,
                cov_path=cov_path, log=lambda m: log_print(m, level=1),
            )
            return
        log_print(
            f"[coverage] the suite is unchanged, but the inherited evidence "
            f"is not reusable ({reason}). Consolidating.", level=1)

    def _input_candidates(phase):
        """The agent's suite and the inherited baseline as candidates."""
        pre = result.test_result
        changed = _loop_suite_changed(phase)
        candidate2_coverage = (
            phase.candidate2_coverage
            if isinstance(phase.candidate2_coverage, dict)
            else {}
        )
        pre_line = candidate2_coverage.get("line")
        if pre_line is None and pre is not None:
            pre_line = getattr(pre, "line_coverage", None)
        pre_branch = candidate2_coverage.get("branch")
        if pre_branch is None and pre is not None:
            pre_branch = getattr(pre, "branch_coverage", None)
        pre_mutation = (
            mutation.matrix_score_percent(phase.candidate2_km)
            if phase.candidate2_km is not None else None
        )
        agent_lines, agent_branches = _coverage_sets(candidate2_coverage)
        agent_source = phase.candidate2_source or cov_src
        agent_scored = ScoredSuite(
            label="agent", source=agent_source,
            line=pre_line, branch=pre_branch,
            mutation=pre_mutation, runtime_ms=None,
            green=phase.candidate2_green,
            status=("passing" if phase.candidate2_green else "incomplete"),
            kill_matrix=phase.candidate2_km,
            coverage_evidence=(candidate2_coverage or None),
            per_test_tokens=(
                {
                    str(test_id): frozenset(map(str, tokens))
                    for test_id, tokens in (
                        candidate2_coverage.get("per_test_tokens") or {}
                    ).items()
                }
                if candidate2_coverage.get("per_test_tokens") is not None
                else None
            ),
            covered_lines=agent_lines,
            covered_branches=agent_branches,
            test_count=(
                phase.candidate2_candidate.test_count if changed
                else getattr(pre, "test_count_standard", None)
                if pre is not None else None
            ),
            execution_name=(
                phase.candidate2_execution_name
                or result.test_file_name or ""
            ),
            status_reason=phase.candidate2_status_reason,
            target_revision=phase.target_revision,
            measurement_identity=phase.measurement_identity,
        )
        # The baseline is the suite the starter phase delivered, scored by the evidence
        # bound to it above.
        baseline_km = starter_reusable_km
        baseline_coverage = starter_coverage or {}
        starter_result = result.starter_result
        baseline_line = baseline_coverage.get("line")
        if baseline_line is None and starter_result is not None:
            baseline_line = getattr(
                starter_result, "baseline_line_coverage", None
            )
        baseline_branch = baseline_coverage.get("branch")
        if baseline_branch is None and starter_result is not None:
            baseline_branch = getattr(
                starter_result, "baseline_branch_coverage", None
            )
        baseline_seconds = (
            getattr(starter_result, "baseline_suite_seconds", None)
            if starter_result is not None else None
        )
        baseline_lines, baseline_branches = _coverage_sets(baseline_coverage)
        baseline_scored = ScoredSuite(
            label="baseline",
            source=baseline_content,
            line=baseline_line,
            branch=baseline_branch,
            mutation=(
                mutation.matrix_score_percent(baseline_km)
                if baseline_km is not None else None
            ),
            runtime_ms=(
                float(baseline_seconds) * 1000.0
                if baseline_seconds is not None else None
            ),
            green=True,
            status="passing",
            kill_matrix=baseline_km,
            coverage_evidence=(baseline_coverage or None),
            per_test_tokens=(
                {
                    str(test_id): frozenset(map(str, tokens))
                    for test_id, tokens in (
                        baseline_coverage.get("per_test_tokens") or {}
                    ).items()
                }
                if baseline_coverage.get("per_test_tokens") is not None
                else None
            ),
            covered_lines=baseline_lines,
            covered_branches=baseline_branches,
            test_count=(
                getattr(starter_result, "selected_tests", None)
                if starter_result is not None else None
            ),
            execution_name=result.test_file_name or "",
            target_revision=phase.target_revision,
            measurement_identity=phase.measurement_identity,
        )
        return agent_scored, baseline_scored

    # Every outcome persists in run_phase_end: the union measurement, the
    # delivered suite's matrix when the merge is delivered, the losing probe's matrix
    # when the agent stands, the coverage sidecars, and the decision record.
    outcome = run_phase_end(
        PhaseEnd(
            name="coverage", backend=backend, sources=[baseline_content, cov_src],
            target_file_path=target, source_dir=source_dir, timeout=timeout,
            matrices_dir=_archive_dir(ctx, "coverage"),
            execution_name=result.test_file_name,
            log=lambda m: log_print(m, level=1),
            options=PhaseEndOptions(
                mutation_method=config_module.consolidation_mutation_method(),
                reduction=config_module.solver_settings(cfg),
                candidate_evidence=[baseline_tokens, None],
                inherited1=InheritedMatrix(
                    starter_reusable_km, baseline_hash, result.test_file_name,
                    inherited_identity),
                measurement_identity=phase_measurement_identity,
                candidate2_label="agent",
            ),
            generation_scope=generation_scope,
            generation_inputs=generation_inputs,
        ),
        runner_cm,
        own_candidates=lambda phase, consolidated: [*consolidated, *_input_candidates(phase)],
        no_winner=lambda names: (
            "the coverage phase end produced no passing candidate with "
            f"complete {names} measurements"),
    )
    phase, winner = outcome.phase, outcome.winner
    if winner is None:
        raise RuntimeError(f"[coverage] {outcome.detail}")
    delivered_tests = outcome.delivered_tests
    result.consolidation_stats["coverage"] = _coverage_consolidation_stats(
        phase, winner)

    # The mutation phase reuses the selected coverage suite's matrix as its
    # baseline, regardless of which candidate won. The sidecar binds the
    # matrix to the exact delivered bytes: the derive verifies the hash and
    # refuses a mismatched or unhashed sidecar with a named reason.
    if winner.kill_matrix is not None:
        _write_baseline_matrix(
            baseline_matrix_path,
            source_sha256(winner.source),
            mutation_agent.mutation_measurement_identity(
                target_file_path=target,
                language=language,
                execution_name=result.test_file_name,
                config=dict(mut_cfg),
                attribution=measurement.union_attribution(mut_cfg),
            ),
            winner.kill_matrix)
        log_print(
            "[coverage] persisted the delivered suite's matrix as the mutation "
            "baseline (hash-stamped)", level=1)

    # Always write the exact selected source, including when the agent
    # candidate wins, so the delivered bytes match the persisted source hash
    # and measurements.
    measurement._atomic_write_text(cov_path, winner.source)

    agent_changed = winner.label == "agent" and _loop_suite_changed(phase)
    if winner.label == "agent" and not agent_changed:
        # The agent suite already stands on disk. Capture the alternatives for
        # replay when consolidation produced a different measured candidate.
        if phase.reduced_source is not None:
            dump = _archive_dir(ctx, "coverage") / "consolidation_loss"
            try:
                dump.mkdir(parents=True, exist_ok=True)
                suffix = cov_path.suffix
                (dump / f"agent_suite{suffix}").write_text(
                    cov_src, encoding="utf-8")
                (dump / f"baseline_suite{suffix}").write_text(
                    baseline_content, encoding="utf-8")
                (dump / f"consolidated_suite{suffix}").write_text(
                    phase.reduced_source, encoding="utf-8")
            except OSError:
                pass
        log_print(
            f"[coverage] the agent's suite stands "
            f"({phase.detail or 'no improvement over the agent'})", level=1)
        return

    # Another candidate won (reduced, merged or baseline), or the agent's
    # suite without the tests a measurement named: report its measured
    # metrics, and for the agent's suite its test count and test lines.
    if result.test_result is not None:
        if winner.line is not None:
            result.test_result.line_coverage = winner.line
        if winner.branch is not None:
            result.test_result.branch_coverage = winner.branch
        if agent_changed:
            delivered = measurement.record(
                measurement.observe_file(cov_path, backend=backend))
            # By name, as the results table reads them.
            for name, key in (("test_count_standard", "tests"),
                              ("test_loc_standard", "code_lines")):
                setattr(result.test_result, name, delivered[key])
    agent_helpers.log_phase_metrics(
        lambda m: log_print(m, level=1), "coverage",
        line=winner.line, branch=winner.branch,
        mutation=winner.mutation, tests=delivered_tests,
        note=(f"consolidated (pool={phase.pool_tests}) |" if phase.pool_tests is not None
              else "consolidated (pool not counted) |"),
    )


def _loop_suite_changed(phase) -> bool:
    """Whether the phase end's removal fallback took tests from the loop's
    suite, its candidate."""
    return any(entry["place"] == "phase_end.candidate" and "removed" in entry
               for entry in phase.measurement_removals)


def _reuse_coverage_phase_end(
    ctx: TargetPhasesContext, result: TestPipelineResult, *, evidence,
    delivered: str, target: str, source_dir: str, language: str,
    measurement_identity: dict, mut_cfg: Dict[str, Any],
    baseline_matrix_path: Path, cov_path: Path, log,
) -> None:
    """The coverage phase end from the starter phase's evidence: the archive
    is copied and rebound, the mutation baseline sidecar is written from the
    inherited matrix and bound to the delivered bytes, the delivered bytes
    are written as they are, and the phase reports the inherited scores with
    the winner ``inherited``."""
    from testuary.agent import _helpers as agent_helpers
    from testuary.consolidation import reuse_inherited_phase_end
    inherited = reuse_inherited_phase_end(
        source_phase_dir=_archive_dir(ctx, "starter"), source_phase="starter",
        destination_dir=_archive_dir(ctx, "coverage"),
        evidence=evidence, target_file=target,
        source_dir=source_dir, delivered_source=delivered,
        execution_name=result.test_file_name,
        measurement_identity=measurement_identity,
        generation_scope=_generation_scope(ctx, "coverage"),
    )
    if inherited.kill_matrix is not None:
        _write_baseline_matrix(
            baseline_matrix_path,
            source_sha256(delivered),
            mutation_agent.mutation_measurement_identity(
                target_file_path=target, language=language,
                execution_name=result.test_file_name,
                config=dict(mut_cfg),
                attribution=measurement.union_attribution(mut_cfg),
            ),
            inherited.kill_matrix, inherited_from="starter")
    measurement._atomic_write_text(cov_path, delivered)
    result.consolidation_stats["coverage"] = {
        "merged_tests": inherited.delivered_tests,
        "delivered_tests": inherited.delivered_tests,
        "delivered_from": "inherited",
    }
    line = (inherited.coverage or {}).get("line")
    branch = (inherited.coverage or {}).get("branch")
    if result.test_result is not None:
        if line is not None:
            result.test_result.line_coverage = line
        if branch is not None:
            result.test_result.branch_coverage = branch
    agent_helpers.log_phase_metrics(
        log, "coverage", line=line, branch=branch,
        mutation=(
            mutation.matrix_score_percent(inherited.kill_matrix)
            if inherited.kill_matrix is not None else None
        ),
        tests=inherited.delivered_tests,
        note="inherited from the starter phase; no measurement, reduction "
             "or selection |",
    )


def _coverage_consolidation_stats(phase, winner) -> dict:
    return {
        "merged_tests": phase.pool_tests,
        "delivered_tests": (
            winner.test_count if winner is not None else None),
        "delivered_from": winner.label if winner is not None else "agent",
    }


def _phase_coverage(
    ctx: TargetPhasesContext, result: TestPipelineResult,
    baseline: Optional[dict] = None,
) -> Optional[CoverageTestResult]:
    """Run the coverage agent. Records phase outcome on result."""
    driver = coverage_agent
    driver.init_logging(
        ctx.workspace,
        log_dir=str(ctx.target_logs_dir),
        log_filename="coverage.jsonl",
    )
    try:
        test_result = driver.generate_coverage_tests(
            ctx.project_name,
            top_file=ctx.top_file,
            test_output_dir=str(ctx.work_tests),
            baseline=baseline,
            **({"prepared_models": ctx.prepared_models} if ctx.prepared_models is not None else {}),
        )
    finally:
        driver.save_logs()

    # Outcome bookkeeping for the caller's skip-downstream logic. A file on
    # disk is necessary but not sufficient: the phase fails closed on a
    # hard terminal status, on an unknown status, and on a limit stop that
    # never produced a measurement, so a degraded artifact is flagged
    # rather than read as a completed phase, exactly as the mutation
    # phase maps its own statuses.
    test_file_on_disk = (
        test_result.test_file_name != FAILED_MARKER
        and (ctx.work_tests / test_result.test_file_name).exists()
    )
    status = getattr(test_result, "status", None)
    # A model request failure is infrastructure, a limit stop without a
    # usable suite is the method's outcome, and other ends stay unjudged.
    request_kind = getattr(test_result, "failure_kind", None)
    if not test_file_on_disk:
        _record_outcome(result, "coverage", "failed", _coverage_failure_reason(test_result),
                        failure_kind=request_kind or (
                            phase_state.METHOD_FAILURE if status == "budget_exhausted" else None))
    elif status in ("agent_error", "context_overflow", "tests_failed"):
        reason = (getattr(test_result, "error", "") or "").strip()
        _record_outcome(
            result, "coverage", "failed",
            f"coverage phase ended on a non-retryable error ({status}): "
            f"{reason[:300] or 'see ' + str(ctx.target_logs_dir / 'coverage.jsonl')}. The suite "
            f"on disk is a partial artifact, not a completed phase.",
            failure_kind=request_kind,
        )
    elif status == "budget_exhausted" and getattr(
        test_result, "line_coverage", None,
    ) is None:
        _record_outcome(
            result, "coverage", "failed",
            "the request limit ended the coverage phase before any "
            "coverage measurement; the suite on disk is unmeasured.",
            failure_kind=phase_state.METHOD_FAILURE,
        )
    elif status is None:
        _record_outcome(
            result, "coverage", "failed",
            "the coverage phase returned no status; an unclassified end "
            "fails closed.",
        )
    else:
        _record_outcome(result, "coverage", "ok", None)
    return test_result


def _check_restored_delivered_suite(
    ctx: TargetPhasesContext, path: Path,
) -> tuple[bool, str]:
    """The normal run of the delivered suite restored with the mutation
    phase reused, since no phase runs after it. Nothing is pruned."""
    from testuary.consolidation import check_restored_suite
    log_print = ctx.log_print
    mut_cfg = config_module.get_agent_config("test_mutation", parent="test_coverage")
    target = str(ctx.top_file["file_path"])
    backend = languages.get_backend(languages.detect_language(target))
    return check_restored_suite(
        backend, path.read_text(encoding="utf-8"),
        target_file_path=target,
        source_dir=project.resolve_source_dir(target),
        timeout=int(mut_cfg["suite_run_timeout_seconds"]),
        log=lambda m: log_print(m, level=1),
    )


def _restored_test_file_collects(
    ctx: TargetPhasesContext, file_name: str,
) -> tuple[bool, str]:
    """Verify the on-disk test file at least loads cleanly, with one
    fail-fast normal run (:mod:`testuary.normal_run`).

    Catches the case where a stale, restored test file references a
    name that no longer exists in the (now-canonical) source, for example
    when a previous run wrote tests against a corrupted source that had a
    spurious global, the source got restored, and the tests now
    ``NameError`` / ``ImportError`` at collection. Catching it here
    avoids surfacing the failure deep inside the mutation phase's
    normal run, after 20 to 30 seconds of cosmic-ray session setup. This
    check catches it at once with a clear message.
    """
    language = languages.detect_language(str(ctx.canonical_path))
    backend = languages.get_backend(language)
    if backend is None:
        # No backend means the target cannot be tested at all. A guard that
        # cannot run must fail closed (report a problem), not wave the target
        # through. The caller treats a non-empty message as a hard failure.
        return False, (
            f"no language support for {Path(ctx.canonical_path).name} "
            f"(language={language!r}), so this target cannot be tested."
        )
    # Set up the test environment before validating. On the resume
    # path (--reuse coverage) the orchestrator has not yet
    # called any agent's setup_test_env, so there is no conftest.py
    # in work_tests and pytest cannot put the target's package on
    # sys.path. This adds it so ``from sympy.ntheory.X import Y``
    # resolves at collection, since the source is right there in the
    # target's copy. A language that prepares a suite directory (Python's conftest)
    # writes it there. Stub backends
    # raise NotImplementedError, which reaches the operator
    # instead of failing the collect-check silently.
    target_file_path = str(ctx.top_file["file_path"])
    source_dir = project.resolve_source_dir(target_file_path)
    try:
        backend.setup_test_env(
            str(ctx.work_tests), source_dir, target_file_path,
        )
    except NotImplementedError as exc:
        return False, f"backend.setup_test_env not implemented: {exc!r}"
    except Exception as exc:
        return False, f"backend.setup_test_env raised: {exc!r}"
    path = ctx.work_tests / file_name
    # The suite's other runs take this limit, and a slow build needs it.
    timeout = int(config_module.get_agent_config("test_coverage")["suite_run_timeout_seconds"])
    try:
        # ``source_dir`` resolved through the per-target override so
        # the run lands inside the build-system module that owns the
        # target (Java: ``<work_src>`` or a sibling like ``javase/``,
        # never ``work_dir``, which sits outside the project tree and
        # has no pom).
        res = normal_run.run(
            backend, [str(path)], source_dir,
            timeout=timeout, target=target_file_path, fail_fast=True, tb="line",
        )
    except Exception as exc:
        return False, f"the normal run raised: {exc!r}"
    if res.timed_out:
        return False, f"the test file's load check timed out after {timeout} s"
    # ``passed == 0 and failed == 0 and errors > 0`` is the
    # collection-error signature. The file imported but no test ran.
    # A real failed test has ``failed > 0`` and is accepted (preflight
    # catches it one phase later with full output). A count the run did
    # not observe (None, a run without its record such as a crash) counts
    # as nothing collected.
    if res.ok or res.collected:
        return True, ""
    return False, (
        f"the test file did not load (exit={res.returncode}); output tail:\n"
        + (res.output or "")[-2000:]
    )


def _reused_target_test_present(ctx: TargetPhasesContext) -> bool:
    """True when the target's delivered test file exists on disk to reuse.
    Used to decide whether the delivered suite can actually be reused, or
    whether its absence is a hard target failure."""
    backend = languages.get_backend(
        languages.detect_language(str(ctx.canonical_path))
    )
    if backend is None:
        return False
    return backend.target_test_path(
        str(ctx.work_tests), str(ctx.canonical_path),
    ).exists()


def _restore_coverage_test_file_from_disk(
    ctx: TargetPhasesContext, result: TestPipelineResult,
) -> Optional[str]:
    """Skip-coverage path: locate an existing target test file via the
    language backend so Java's ``<Stem>Test.java`` resolves as well as
    Python's ``test_<stem>.py``.
    Delegates the per-language preferred-then-fallback search +
    decoy exclusion (``test_failed.py`` for Python, etc.) to
    ``backend.find_target_test_file``."""
    target_file_path = str(ctx.top_file["file_path"])
    backend = languages.get_backend(
        languages.detect_language(target_file_path)
    )

    if backend is not None:
        found = backend.find_target_test_file(
            str(ctx.work_tests), target_file_path,
        )
        if found is not None and found.exists():
            # Mutation consolidation overwrites the coverage-named canonical
            # suite with the selected delivered suite. When mutation is also being
            # reused, those bytes are therefore bound by mutation's delivered
            # validity record, not coverage's earlier record (whose immutable
            # source lives in archive/mutation/pre_merge).
            validity_phase = "mutation" if ctx.reuse.mutation else "coverage"
            if not ctx.reuse.mutation:
                # The mutation phase runs again, from the coverage phase's own
                # suite: the phase's snapshot, which the record binds.
                _, suffix = _canonical_suite_name(ctx)
                own = (_restore_run_dir(ctx, "coverage") / "archive" / "coverage"
                       / f"result{suffix}")
                if own.exists():
                    shutil.copyfile(own, found)
            valid, validity_reason = _verify_restored_phase_validity(
                ctx, validity_phase, found,
            )
            if not valid:
                _record_outcome(
                    result,
                    "coverage",
                    "failed",
                    "the restored coverage suite does not match its validation "
                    f"record: {validity_reason}",
                )
                return None
            ok, reason = _restored_test_file_collects(ctx, found.name)
            if ok and ctx.reuse.mutation:
                # No phase runs after the restore to check the delivered
                # suite, so its normal run is checked here.
                accepted, why = _check_restored_delivered_suite(ctx, found)
                if not accepted:
                    _record_outcome(result, "coverage", "failed", why)
                    return None
            if ok:
                _record_outcome(result, "coverage", "skipped_reused", None)
                return found.name
            # File exists but is stale / corrupted vs the current
            # source. Recording as "failed" downstream-skips mutation
            # and Finishing, so no hour goes to a suite known to be broken.
            _record_outcome(
                result, "coverage", "failed",
                f"restored test file {found.name!r} does not collect: "
                f"{reason}",
            )
            return None

    # --reuse coverage reuses the coverage suite from disk. It is
    # the mandatory correctness core, so a missing file is a hard target
    # failure, not a benign skip. The target ends and the run moves on.
    _record_outcome(
        result, "coverage", "failed",
        "no coverage test file on disk to reuse under "
        "--reuse coverage; the standard suite is mandatory",
    )
    return None


def _coverage_failure_reason(test_result: CoverageTestResult) -> str:
    bits = [getattr(test_result, "status", "unknown")]
    if test_result.test_file_name == FAILED_MARKER:
        bits.append(FAILED_MARKER)
    else:
        bits.append("file missing on disk")
    return " / ".join(bits)


# =============================================================================
# MUTATION PHASE
# =============================================================================

def _phase_mutation(
    ctx: TargetPhasesContext,
    result: TestPipelineResult,
    session: Optional[mutation.MutationSession],
) -> Optional[MutationTestResult]:
    """Run the mutation agent. ``session`` is the pooled mutation-engine
    session for this phase. When None, mutation engine setup failed
    and the agent's tool falls back to per-call engine spawning."""
    if not result.test_file_name or result.test_file_name == FAILED_MARKER:
        _record_outcome(result, "mutation", "skipped_no_coverage_file", None)
        return None

    driver = mutation_agent
    driver.init_logging(
        ctx.workspace,
        log_dir=str(ctx.target_logs_dir),
        log_filename="mutation.jsonl",
    )
    # Inject the shared mutation session for the agent's measure_mutation tool.
    driver.set_mutation_session(session)
    try:
        # Surface the coverage phase's coverage so the mutation prompt
        # knows the baseline.
        coverage_phase_cov = (
            getattr(result.test_result, "line_coverage", None)
            if result.test_result is not None else None
        )
        # One active file: preserve the inherited bytes in memory, then let the
        # Mutation Agent revise that same delivered path. This prevents pytest
        # from collecting a coverage/mutation sibling pair while retaining two
        # logical, source-bound inputs for the phase end.
        preseed = None
        cov_path = ctx.work_tests / result.test_file_name
        cov_bytes: Optional[str] = None
        if cov_path.exists():
            cov_bytes = cov_path.read_text(encoding="utf-8")
            preseed = {"content": cov_bytes, "file_stem": cov_path.stem}
            result.mutation_inherited_source = cov_bytes
        agent_completed = False
        cleanup_pending = False
        try:
            mutation_result = driver.generate_mutation_tests(
                project_name=ctx.project_name,
                target_file_path=str(ctx.top_file["file_path"]),
                test_output_dir=str(ctx.work_tests),
                preseed=preseed,
                coverage_baseline_pct=coverage_phase_cov,
                **({"prepared_models": ctx.prepared_models} if ctx.prepared_models is not None else {}),
            )
            agent_completed = True
        except BaseException as exc:
            cleanup_pending = mutation.mutation_cleanup_pending(exc)
            if cleanup_pending or not isinstance(exc, Exception):
                raise
            # An error that ends the phase without its outcome is not the
            # method's result.
            _record_outcome(
                result, "mutation", "failed",
                f"{type(exc).__name__}: {exc}",
                failure_kind=phase_state.INFRASTRUCTURE_FAILURE,
            )
            return None
        finally:
            # A failed agent may leave a partial write at the delivered path.
            # Restore after ordinary failure, but not while cleanup retains
            # an uncertain writer. A completed agent leaves revised bytes.
            if cov_bytes is not None and not agent_completed and not cleanup_pending:
                try:
                    cov_path.write_text(cov_bytes, encoding="utf-8")
                except OSError:
                    pass
        if not cov_path.exists():
            raise RuntimeError(
                f"the mutation phase produced no delivered suite at "
                f"{cov_path}, which consolidation and Finishing require"
            )
        # A returned result with ``status == 'inconclusive'`` means
        # the engine ran but produced 0 useful outcomes (every
        # mutant was INCOMPETENT or untested). The mutation_score
        # is a placeholder, not a real measurement. Do not let it
        # propagate to the CSV or downstream phases as success.
        if getattr(mutation_result, "status", None) == "inconclusive":
            _record_outcome(
                result, "mutation", "failed",
                "inconclusive: engine returned with 0 useful outcomes "
                "(every mutant INCOMPETENT or untested). 0% score is "
                "not a real measurement; see the [CRITICAL] line in "
                f"{ctx.target_logs_dir / 'mutation.jsonl'} for the breakdown and the "
                "likely root causes (worker import error, broken "
                "test_command, etc.).",
                failure_kind=phase_state.INFRASTRUCTURE_FAILURE,
            )
            return mutation_result
        # An inherited-red early exit means the phase never ran: the suite
        # that was green at the coverage handoff fails in this environment,
        # so there is no measured work to stand behind. Record the phase
        # failed so the target is flagged rather than read as a clean stop.
        if getattr(mutation_result, "status", None) == "skipped_inherited_red":
            _record_outcome(
                result, "mutation", "failed",
                "the inherited suite arrived red at the mutation phase entry "
                "(it was green at the coverage handoff, so a test is flaky or "
                "the test environment changed); the phase never ran.",
                # Unjudged: a generated suite that turns red later is often
                # flaky, the method's own outcome, not only the environment.
            )
            return mutation_result
        # The suite it received ran longer than its time limit before any
        # edit, so the phase never measured anything.
        if getattr(mutation_result, "status", None) == "skipped_inherited_slow":
            _record_outcome(
                result, "mutation", "failed",
                "the inherited suite ran longer than its time limit at the mutation "
                "phase entry, before any edit, so the phase never ran.",
            )
            return mutation_result
        # A non-retryable provider error (an out-of-credit 402 among them) or
        # a context overflow ended the loop before it could complete. The
        # score on record is a partial measurement of an under-optimized
        # suite, not a finished phase. Record the phase failed so the target is
        # flagged rather than read as a clean stop.
        if getattr(mutation_result, "status", None) in (
            "agent_error", "context_overflow",
        ):
            reason = (getattr(mutation_result, "error", "") or "").strip()
            _record_outcome(
                result, "mutation", "failed",
                f"mutation phase ended on a non-retryable error "
                f"({mutation_result.status}): "
                f"{reason[:300] or 'see ' + str(ctx.target_logs_dir / 'mutation.jsonl')}. The score "
                f"on record is a partial measurement, not a completed phase.",
                failure_kind=getattr(mutation_result, "failure_kind", None),
            )
            return mutation_result
        _record_outcome(result, "mutation", "ok", None)
        return mutation_result
    finally:
        driver.set_mutation_session(None)
        driver.save_logs()


# =============================================================================
# MERGE: the mutation suite merges into the coverage suite (not an "agent")
# =============================================================================

def _configured_operators(language: str, cfg) -> Optional[List[str]]:
    """The configured (selective) operator catalogue, for the derived
    kill-matrix view. None when the engine has no selective catalogue."""
    try:
        runner = languages.get_backend(language).mutation_runner(dict(cfg))
        return runner.configured_operators()
    except Exception:
        return None


def _consolidate_mutation(
    ctx: TargetPhasesContext,
    result: TestPipelineResult,
) -> Optional[Any]:
    """The mutation phase's consolidation: merge the mutation agent's suite
    with the coverage baseline it grew from so the target keeps one canonical
    test file. Framework-only. Mirrors ``_consolidate_coverage``: it never
    records a phase outcome, so it cannot downgrade the mutation agent's
    result. Returns the ``MergeResult`` on success, or ``None`` when there is
    nothing to merge (no inherited source or delivered suite) or the
    backend kept the files separate (parse error / unsupported language)."""
    log_print = ctx.log_print
    if not result.test_file_name:
        return None
    delivered_name = result.test_file_name
    delivered_path = ctx.work_tests / delivered_name
    inherited_source = result.mutation_inherited_source
    if inherited_source is None or not delivered_path.exists():
        return None
    agent_source = delivered_path.read_text(encoding="utf-8")
    mutation_result = result.mutation_result
    language = languages.detect_language(str(ctx.canonical_path))
    source_dir = project.resolve_source_dir(str(ctx.top_file["file_path"]))
    cfg = config_module.get_agent_config("test_mutation", parent="test_coverage")
    coverage_archive = _archive_path(ctx, "coverage")
    coverage_record = _read_json(
        coverage_archive / "consolidation_record.json"
    )
    if coverage_record is None:
        raise RuntimeError(
            "the mutation phase cannot check its inherited suite: "
            "the coverage phase's consolidation record is missing"
        )
    inherited_hash = source_sha256(inherited_source)
    recorded_hash = coverage_record.get("selected_source_sha256")
    recorded_name = coverage_record.get("execution_name")
    if recorded_hash != inherited_hash or recorded_name != delivered_name:
        raise RuntimeError(
            "the mutation phase's inherited suite does not match the "
            "coverage phase's consolidation record: "
            f"source_sha256={inherited_hash} recorded={recorded_hash}, "
            f"execution_name={delivered_name!r} recorded={recorded_name!r}"
        )
    generation_scope, generation_inputs = _editing_generation_inputs(
        ctx, "mutation", predecessor_record=coverage_record,
        inherited_source=inherited_source, agent_source=agent_source,
        agent_result=mutation_result,
    )
    # The phase end's identity: a loop matrix of first killers is reused
    # only when the phase end attributes kills the same way.
    expected_agent_identity = mutation_agent.mutation_measurement_identity(
        target_file_path=str(ctx.top_file["file_path"]),
        language=language,
        execution_name=delivered_name,
        config=dict(cfg),
        attribution=measurement.union_attribution(cfg),
    )
    # The inherited candidate is the suite the coverage phase delivered, so
    # both come from that phase's canonical artifacts, whatever won there.
    candidate1_km = _delivered_kill_matrix(coverage_archive, ctx.log_print)
    candidate1_coverage = _delivered_coverage(coverage_archive, ctx.log_print)
    inherited_identity = coverage_record.get("measurement_identity")
    reusable_agent_matrix = None
    if (
        mutation_result is not None
        and getattr(mutation_result, "measurement_identity", None)
        == expected_agent_identity
    ):
        reusable_agent_matrix = getattr(
            mutation_result, "mutation_matrix", None
        )

    # Preserve the two exact logical inputs before the first mutation-engine
    # call. Successful targets already write these same paths and bytes through
    # the normal checkpoint below. Writing them now changes no completed
    # output, while an allocation-level OOM retains enough evidence for a
    # no-model boundary replay.
    durable_pre_merge = _archive_dir(ctx, "mutation") / "pre_merge"
    durable_pre_merge.mkdir(parents=True, exist_ok=True)
    measurement._atomic_write_text(
        durable_pre_merge / f"candidate_inherited{delivered_path.suffix}",
        inherited_source,
    )
    measurement._atomic_write_text(
        durable_pre_merge / f"candidate_agent{delivered_path.suffix}",
        agent_source,
    )

    # Evidence reuse, as at the coverage boundary: an unchanged suite whose
    # inherited evidence is complete and bound stands as it is.
    if cfg["phase_end_evidence_reuse"] and agent_source == inherited_source:
        from testuary.consolidation import (
            inherited_phase_end_eligibility, reuse_inherited_phase_end,
        )
        evidence, reason = inherited_phase_end_eligibility(
            source_phase_dir=coverage_archive, delivered_source=agent_source,
            execution_name=delivered_name,
            measurement_identity=expected_agent_identity,
            require_mutation=config_module.consolidation_measure_mutation(),
            target_file=str(ctx.top_file["file_path"]),
            source_dir=source_dir,
        )
        if evidence is not None:
            inherited = reuse_inherited_phase_end(
                source_phase_dir=coverage_archive, source_phase="coverage",
                destination_dir=_archive_dir(ctx, "mutation"),
                evidence=evidence,
                target_file=str(ctx.top_file["file_path"]),
                source_dir=source_dir, delivered_source=agent_source,
                execution_name=delivered_name,
                measurement_identity=expected_agent_identity,
                generation_scope=generation_scope,
            )
            # The checkpoint copies a merge result's pre-merge paths from the
            # target's copy into the phase archive. They must therefore be the
            # copies the measuring path produces, not the durable
            # ones written above: copying a file onto itself raises and the
            # target would be recorded crashed.
            work_pre_merge = Path(ctx.work_tests).parent / "_pre_merge"
            work_pre_merge.mkdir(parents=True, exist_ok=True)
            work_inherited = (
                work_pre_merge / f"candidate_inherited{delivered_path.suffix}")
            work_agent = (
                work_pre_merge / f"candidate_agent{delivered_path.suffix}")
            measurement._atomic_write_text(work_inherited, inherited_source)
            measurement._atomic_write_text(work_agent, agent_source)
            merge_result = measurement.MergeResult(
                merged_path=str(delivered_path),
                pre_merge_coverage_path=str(work_inherited),
                pre_merge_mutation_path=str(work_agent),
                kill_matrix=inherited.kill_matrix,
                union_tests=inherited.delivered_tests,
                delivered_tests=inherited.delivered_tests,
                winner="inherited",
                winner_source_label="coverage",
            )
            result.consolidation_stats["mutation"] = (
                _mutation_consolidation_stats(merge_result))
            log_print(
                "[mutation] phase end inherited the coverage phase's "
                "evidence: the suite is byte-identical under the same "
                "measurement identity, so no measurement, reduction or "
                "selection", level=1)
            return merge_result
        log_print(
            f"[mutation] the suite is unchanged, but the inherited evidence "
            f"is not reusable ({reason}). Consolidating.", level=1)

    # The one full-catalogue mutation run: measured on the union here,
    # reused by Finishing.
    with measurement.phase_end_runner(
        "mutation", always=True,
        language=language,
        target_file=str(ctx.top_file["file_path"]),
        source_dir=source_dir,
        config=dict(cfg),
        log_fn=lambda m: log_print(m, level=1),
    ) as run_mutation_on:
        merge_result = measurement.consolidate_test_suite(
            test_output_dir=str(ctx.work_tests),
            coverage_file_name=delivered_name,
            language=language,
            target_file_path=str(ctx.top_file["file_path"]),
            source_dir=source_dir,
            run_mutation_on=run_mutation_on,
            candidate1_km=candidate1_km,
            candidate1_coverage=candidate1_coverage,
            candidate1_source=inherited_source,
            candidate1_measurement_identity=inherited_identity,
            candidate2_source=agent_source,
            candidate2_km=reusable_agent_matrix,
            candidate2_source_sha256=getattr(
                mutation_result, "measured_suite_sha256", None
            ),
            candidate2_execution_name=getattr(
                mutation_result, "measured_execution_name", None
            ),
            candidate2_measurement_identity=getattr(
                mutation_result, "measurement_identity", None
            ),
            measurement_identity=expected_agent_identity,
            timeout=int(cfg["suite_run_timeout_seconds"]),
            reduction=config_module.solver_settings(cfg),
            matrices_dir=str(_archive_dir(ctx, "mutation")),
            selective_operators=_configured_operators(language, cfg),
            log_fn=lambda m: log_print(m, level=1),
            generation_scope=generation_scope,
            generation_inputs={"standalone_0": generation_inputs["baseline"],
                               "standalone_1": generation_inputs["agent"]},
        )
    if merge_result is None:
        return None
    # The delivered suite's matrix is already persisted as
    # archive/mutation/selected_kill_matrix.json. A resumed target reloads that
    # same file (see _resume_standard_km), so the headline matches a fresh
    # run's.
    result.consolidation_stats["mutation"] = (
        _mutation_consolidation_stats(merge_result))
    return merge_result


def _generation_scope(ctx, phase):
    return generation_evidence.construction_scope(phase, ctx.run_idx, ctx.generation_invocation_id)


def _editing_generation_inputs(ctx, phase, *, predecessor_record, inherited_source,
                               agent_source, agent_result):
    """Capture actual inherited/edited inputs before any selected-record write."""
    from testuary import models as _model_source
    from testuary.evidence import phase_origin_path

    scope = _generation_scope(ctx, phase)
    parent_phase = "starter" if phase == "coverage" else "coverage"
    baseline = generation_evidence.predecessor(
        predecessor_record, phase=parent_phase, source=inherited_source)
    if agent_source == inherited_source:
        agent = baseline
    else:
        origin = None
        requests = getattr(agent_result, "model_requests", None)
        if (type(requests) is int and requests > 0 and config_module.active_settings() is not None
                and _model_source.current() is None):
            origin = generation_evidence.capture_origin(
                phase_origin_path(ctx.results_dir / ctx.workspace, phase, ctx.run_idx))
        agent = generation_evidence.construction(
            agent_source, **scope, operation="edited", parents=[baseline], origin=origin,
            reason="editing generation inputs unavailable" if origin is None else "",
        )
    return scope, {"baseline": baseline, "agent": agent}


def _write_baseline_matrix(path: Path, suite_sha256: str, measurement_identity: Any,
                           matrix: Any, inherited_from: Optional[str] = None) -> None:
    """Write the mutation phase's baseline matrix sidecar, bound to the
    delivered suite's bytes by their hash and to its measurement, with the
    phase or check a reused matrix came from."""
    record = {"suite_sha256": suite_sha256, "measurement_identity": measurement_identity,
              "matrix": matrix}
    if inherited_from is not None:
        record["inherited_from"] = inherited_from
    path.write_text(json.dumps(record, indent=2), encoding="utf-8")


def _mutation_consolidation_stats(merge_result) -> dict:
    return {
        "merged_tests": getattr(merge_result, "union_tests", None),
        "delivered_tests": getattr(merge_result, "delivered_tests", None),
        "delivered_from": getattr(merge_result, "winner", "merged"),
    }


# =============================================================================
# Final delivered-suite measurement
# =============================================================================

def _phase_finishing(
    ctx: TargetPhasesContext,
    result: TestPipelineResult,
) -> Optional[CombinedTestMetrics]:
    """Measure the delivered suite after coverage and mutation.
    
    Attach Finishing's measurement and the configured phase-original
    measurements. Finishing uses its own mutation session when a reusable
    measurement is unavailable."""
    if not result.test_file_name or result.test_file_name == FAILED_MARKER:
        _record_outcome(result, "finishing", "skipped_no_coverage_file", None)
        return None

    files = _combined_test_files(result)
    measurement.init_logging(
        ctx.workspace,
        log_dir=str(ctx.target_logs_dir),
        log_filename="finishing.jsonl",
    )
    try:
        try:
            # coverage.py's ``--include`` filter matches the EXACT path
            # that gets traced during the run. The tests execute the
            # target's copy of the source (Python imports resolve
            # against work_dir/<project_name>), so coverage is tracked
            # against the copy's path. The canonical
            # projects/ path would produce zero matches.
            #
            # ``source_dir`` flows through the per-target routing
            # override (Java: the owning Maven module's directory).
            # Passing ``work_dir`` directly would point Maven at the
            # workspace root which holds no pom and would fail-fast
            # with "no pom.xml at" that root.
            finishing_source_dir = project.resolve_source_dir(
                str(ctx.top_file["file_path"])
            )
            metrics = measurement.measure_combined(
                target_file_path=str(ctx.top_file["file_path"]),
                test_output_dir=str(ctx.work_tests),
                source_dir=finishing_source_dir,
                test_file_names=files,
                recorded_coverage=_phase_end_coverage(ctx, result),
            )
        except Exception as exc:
            _record_outcome(
                result, "finishing", "failed",
                f"{type(exc).__name__}: {exc}",
            )
            return None
        finishing_error: Optional[str] = None
        try:
            _attach_finishing(ctx, result, metrics)
        except Exception as exc:
            finishing_error = f"{type(exc).__name__}: {exc}"
            measurement.log_print(
                f"[finishing] standalone metrics failed ({finishing_error})",
                level=1,
            )
        # The standard suite's full-catalogue mutation score is the phase's
        # primary result, not informational: run_finishing raises
        # rather than degrade when that score is mandatory and the engine
        # cannot start. When it is absent and measure_mutation has not
        # been opted out, Finishing did not produce its canonical
        # number, so record it failed. The phase is then not recorded
        # complete and the target re-runs on resume instead of silently emitting
        # a blank mutation column and being marked done. The per-phase
        # snapshots stay best-effort (they swallow their own errors). Only the
        # mandatory standard score gates the outcome.
        if _standard_mutation_finishing_missing(metrics):
            measured = getattr(getattr(metrics, "finishing", None),
                               "mutation_measurement", None)
            detail = finishing_error or (
                "the final mutation measurement did not resolve its "
                "catalogue; no standard-suite score is reported"
                if measured == "incomplete" else
                "Finishing produced no standard-suite full-catalogue "
                "mutation score"
            )
            # Unjudged: no standard-suite score can come from the engine or
            # from a delivered suite that is no longer green (often a flaky
            # generated suite, the method's own outcome).
            _record_outcome(result, "finishing", "failed", detail)
            # The CSV's finishing_status reads metrics.status. Keep it in step
            # with the recorded outcome so a finalize that measured no
            # standard-suite mutation score is not reported as success.
            if metrics is not None:
                metrics.status = "failed"
                metrics.error = getattr(metrics, "error", None) or detail
            return metrics
        if metrics.coverage_final_line is None:
            # The delivered suite's final coverage run gave no value (it
            # timed out or ran red): its coverage is unmeasured, and the
            # coverage phase's value describes another suite.
            detail = ("the final coverage measurement of the delivered suite "
                      "gave no value")
            _record_outcome(result, "finishing", "failed", detail)
            metrics.status = "failed"
            metrics.error = getattr(metrics, "error", None) or detail
            return metrics
        _record_outcome(result, "finishing", "ok", None)
        return metrics
    finally:
        measurement.save_logs()


def _phase_end_coverage(
    ctx: "TargetPhasesContext", result: "TestPipelineResult",
) -> Optional[Tuple[Optional[float], Optional[float]]]:
    """The line and branch coverage of the delivered suite as the phase end
    that delivered it measured: the latest phase whose record names the
    delivered source, with coverage evidence bound to it. A record hashes
    the source text, so the file is read as text: on disk it carries the
    platform's line ends. Finishing takes these numbers instead of judging
    the suite a second time. None when no phase end measured this source
    (a result directory whose records lack it), and finishing measures."""
    try:
        source = (Path(ctx.work_tests) / result.test_file_name).read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    digest = source_sha256(source)
    for phase in ("mutation", "coverage", "starter"):
        archive = _archive_path(ctx, phase)
        record = _read_json(archive / "consolidation_record.json") or {}
        if record.get("selected_source_sha256") != digest:
            continue
        coverage = _delivered_coverage(archive, ctx.log_print) or {}
        if coverage.get("line") is not None:
            return coverage["line"], coverage.get("branch")
    return None


def _resume_standard_km(
    ctx: "TargetPhasesContext", result: "TestPipelineResult",
) -> Optional[dict]:
    """The standard (merged-suite) kill matrix to hand Finishing.

    On a fresh run this is ``mutation_merge_result.kill_matrix``: the selected
    suite's matrix from mutation-phase consolidation. On a resume the mutation phase is restored from
    disk, so ``mutation_merge_result`` is None. Reload
    ``archive/mutation/selected_kill_matrix.json``, the file the consolidation
    persists with exactly that matrix, so a resumed target reports the same
    headline a fresh one computes. Reusing the already-measured matrix also
    avoids a fresh scratch re-run whose un-mutated baseline could fail to
    collect and be scored by cosmic-ray as every mutant killed (a false
    100%). Only a measurement that records every killing test is
    reused: for a derived matrix, or a phase end's first killers, it is
    None and Finishing measures the suite."""
    mr = result.mutation_merge_result
    if mr is not None and getattr(mr, "kill_matrix", None) is not None:
        km = mr.kill_matrix
    else:
        km = _delivered_kill_matrix(_archive_path(ctx, "mutation"), ctx.log_print)
    if km is None or "derived" in km or measurement.union_attribution(measurement.mutation_config()) != "all":
        return None
    return km


def _restore_consolidation_stats(
    ctx: "TargetPhasesContext", result: "TestPipelineResult",
) -> None:
    """Restore phase-boundary metadata omitted from an in-memory resume.

    A resumed phase reloads its delivered matrix from disk while its
    ``MergeResult`` object stays behind, so the winner and the delivered test
    count come from ``consolidation_record.json``, which holds the
    authoritative decision, rather than from whatever ``tests.csv`` carried
    into the resume.
    """
    for phase in ("starter", "coverage", "mutation"):
        if phase in result.consolidation_stats:
            continue
        record = _recorded_consolidation(ctx, phase)
        if record:
            result.consolidation_stats[phase] = record


def _recorded_consolidation(ctx: "TargetPhasesContext", phase: str) -> Optional[Dict[str, Any]]:
    """A phase's consolidation counts and winner from its
    ``consolidation_record.json``, None without the record."""
    record = _read_json(_archive_path(ctx, phase) / "consolidation_record.json")
    if not record:
        return None
    return {key: record.get(key)
            for key in ("merged_tests", "delivered_tests", "delivered_from")}


def _attach_finishing(
    ctx: TargetPhasesContext,
    result: TestPipelineResult,
    combined: CombinedTestMetrics,
) -> None:
    """Compute the standalone test-evaluation metrics and attach them to
    ``combined`` so they ride into tests.csv.

    The standard (merged) suite gets one full-catalogue mutation run in a
    clean directory. Its coverage is reused from ``combined`` (already
    measured).

    The standard suite (coverage + mutation) is the correctness core, so its
    mutation score is mandatory: the measurement runs regardless of
    ``--reuse mutation`` (which only means "reuse the mutation
    tests already on disk"), and a missing standard suite or an unavailable
    engine ends the run with a message rather than emitting empty columns.
    ``measure_mutation = false`` is the one explicit opt-out.
    """
    target_file = str(ctx.top_file["file_path"])
    language = languages.detect_language(target_file)
    source_dir = project.resolve_source_dir(target_file)

    merged_path = Path(ctx.work_tests) / result.test_file_name

    shared: List[Path] = []
    setup_backend = languages.get_backend(language)
    if setup_backend is not None:
        try:
            shared = [Path(p) for p in setup_backend.shared_setup_files(
                str(ctx.work_tests),
            )]
        except NotImplementedError:
            shared = []

    # Finishing reuses the delivered suite's coverage, which the phases
    # already measured (standard_coverage).
    combined.finishing = measurement.run_finishing(
        target_file=target_file,
        source_dir=source_dir,
        language=language,
        merged_test_path=merged_path,
        shared_setup_paths=shared,
        scratch_root=Path(ctx.work_tests).parent / ".finishing_scratch",
        config=measurement.mutation_config(),
        inherited_standard_km=_resume_standard_km(ctx, result),
        standard_coverage=(
            combined.coverage_final_line, combined.coverage_final_branch,
        ),
        kill_matrix_out=Path(ctx.work_tests) / "kill_matrix.json",
        log_fn=measurement.log_print,
    )
    _attach_phase_scores(ctx, combined)


def _read_json(path: Path) -> Optional[dict]:
    """Load a persisted json artifact, or None when it is absent or unreadable."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _delivered_evidence(archive_dir: Path, name: str, log: Callable[..., None]) -> Optional[dict]:
    """The canonical evidence ``name`` of the suite a phase DELIVERED,
    whatever won, or None when there is none to use.

    The file names the bytes it describes and the phase's record names the
    bytes the phase delivered. Evidence of other bytes is refused, with the
    reason logged, and treated as absent: every reader has a defined path for
    absent evidence, and none may report another suite's numbers as the
    delivered suite's."""
    payload = _read_json(archive_dir / name)
    if payload is None:
        return None
    record = _read_json(archive_dir / "consolidation_record.json") or {}
    delivered = record.get("selected_source_sha256")
    described = payload.get("source_sha256") if isinstance(payload, dict) else None
    if delivered is None or described != delivered:
        log(
            f"[evidence] {archive_dir.name}/{name} is not bound to the suite "
            f"the phase delivered (it describes {described}, the record names "
            f"{delivered}), so it is not used", level=1)
        return None
    return payload


def _delivered_kill_matrix(archive_dir: Path, log: Callable[..., None]) -> Optional[dict]:
    """The kill matrix of the suite a phase DELIVERED, whatever won, in this
    version's words: a matrix of an earlier schema reads through
    ``upgrade_kill_matrix``. One of a schema version this version cannot
    read is absent, with the reason logged, as evidence of other bytes is
    (:func:`_delivered_evidence`)."""
    from langadapter.mutation import upgrade_kill_matrix

    km = _delivered_evidence(archive_dir, "selected_kill_matrix.json", log)
    if km is None:
        return None
    try:
        return upgrade_kill_matrix(km)
    except ValueError as exc:
        log(f"[evidence] {archive_dir.name}/selected_kill_matrix.json: {exc}, "
            "so it is not used", level=1)
        return None


def _delivered_coverage(archive_dir: Path, log: Callable[..., None]) -> Optional[dict]:
    """The per-test coverage of the suite a phase DELIVERED, whatever won."""
    return _delivered_evidence(archive_dir, "selected_coverage_matrix.json", log)


def _attach_phase_scores(
    ctx: TargetPhasesContext,
    combined: CombinedTestMetrics,
) -> None:
    """Attach each phase's own metrics by reading what the phase already
    persisted, not by re-measuring the snapshot.

    Each phase's consolidation writes ``archive/<phase>/selected_kill_matrix.json``
    and ``selected_coverage_matrix.json`` for the suite it delivered, so the
    starter and coverage columns are that matrix's mutation score and
    that sidecar's recorded line / branch, with no second mutation run. The final (mutation)
    suite is Finishing's measurement above, reused. These columns are informational: a
    phase whose matrices are absent leaves them empty rather than failing the
    target. ``phase_scores`` selects full (coverage + mutation),
    coverage-only, or off."""
    if combined.finishing is not None:
        combined.phase_finishing["mutation"] = combined.finishing

    mode = measurement.phase_scores_mode()
    if mode == "off":
        return

    language = languages.detect_language(str(ctx.top_file["file_path"]))
    selective = _configured_operators(language, measurement.mutation_config())
    sel_set = set(selective) if selective else None

    for phase in ("starter", "coverage"):
        adir = _archive_path(ctx, phase)
        # The suite the phase delivered, whatever won.
        cov = _delivered_coverage(adir, ctx.log_print)
        km = _delivered_kill_matrix(adir, ctx.log_print) if mode != "coverage" else None
        if cov is None and km is None:
            continue
        combined.phase_finishing[phase] = measurement.FinishingMetrics(
            line_coverage=(cov or {}).get("line"),
            branch_coverage=(cov or {}).get("branch"),
            mutation_score_full=(
                mutation.matrix_score_percent(km) if km else None
            ),
            mutation_score_selective=(
                mutation.matrix_score_percent(km, operators=sel_set)
                if (km and sel_set) else None
            ),
        )


# =============================================================================
# MUTATION SESSION LIFECYCLE
# =============================================================================

@contextlib.contextmanager
def _open_mutation_session(ctx: TargetPhasesContext):
    """Reuse one mutation-engine session across measurements within the phase.
    
    Yield None when skipped or setup fails without a retained cleanup owner.
    The agent can then open an engine session per call. Unresolved setup/close
    failures propagate. Body failures keep their original exception and owner.
    Finishing uses a separate session after successful pooled cleanup."""
    if ctx.reuse.mutation:
        yield None
        return

    config = measurement.mutation_config()
    workers = int(config["engine_workers"])

    # Acquire the session by hand to tell a setup failure
    # (yield None, fall back) from in-with exceptions (propagate up).
    # Using ``with engine.session(): yield`` inside a try/except would
    # catch the user's body exceptions and try to yield twice, which is
    # a generator protocol violation.
    target_language = languages.detect_language(str(ctx.canonical_path))
    # Mutation session lives at the build-system module root, same as
    # the test invocations. For Java the workspace directory holds no pom.
    # ``resolve_source_dir`` resolves the actual module root.
    mutation_source_dir = project.resolve_source_dir(
        str(ctx.top_file["file_path"])
    )
    try:
        # Opening the pooled session spawns the worker clones (~20s at high
        # worker counts) and closing it tears them down. Wire the logger
        # into the runner now so that setup/teardown is not silent. The
        # agent re-wires its own logger for the per-call measurements later.
        log_print = ctx.log_print
        runner = languages.get_backend(target_language).mutation_runner(
            dict(config), source_dir=mutation_source_dir,
            log=measurement.level_tolerant_log(log_print),
        )
        # The agents' suites lie in the work tree's tests directory, which the
        # session declares as its suites' directory.
        session = runner.open(workers, suite_dirs=(ctx.work_tests,))
    except Exception as exc:
        if mutation.mutation_cleanup_pending(exc):
            raise
        # Setup failed (engine selection, the fleet's start, port
        # allocation and so on). Agents fall back to a session per call,
        # still correct, just without the amortization win. Surface it: a
        # silent fallback hides both the lost speed-up and the reason the
        # pooled session is absent.
        log_print = ctx.log_print
        log_print(
            f"[mutation-session] pooled session setup failed "
            f"({type(exc).__name__}: {exc}). Each measurement starts its own "
            f"engine instead (slower).",
            level=1,
        )
        yield None
        return

    try:
        yield session
    except BaseException as exc:
        mutation.close_mutation_session(session, primary=exc)
        raise
    else:
        mutation.close_mutation_session(session)


def _standard_mutation_finishing_missing(combined) -> bool:
    """True when the standard suite's full-catalogue mutation score, the
    phase's primary result, is mandatory but absent.

    ``measure_mutation = false`` is the one explicit opt-out: in that
    mode the score is intentionally not measured and this returns False. With
    Finishing's mutation measurement on (the default), an absent score means
    ``run_finishing`` raised or produced no value, so Finishing did
    not produce its canonical number and the caller records it failed rather
    than ok."""
    if not config_module.finishing_measures_mutation():
        return False
    fin = getattr(combined, "finishing", None)
    return fin is None or getattr(
        fin, "mutation_score_full", None,
    ) is None


# =============================================================================
# HELPERS
# =============================================================================

def _has_usable_test_file(
    ctx: TargetPhasesContext, result: TestPipelineResult,
) -> bool:
    """True when a test file of the coverage phase lies on disk that
    subsequent phases can build on."""
    name = result.test_file_name
    if not name or name == FAILED_MARKER:
        return False
    return (ctx.work_tests / name).exists()


def _combined_test_files(result: TestPipelineResult) -> List[str]:
    """Return the delivered standard suite as a file list.

    Coverage and mutation revise the same file, so the list holds at most
    that one name."""
    return [result.test_file_name] if result.test_file_name else []


def _final_line_coverage(result: TestPipelineResult) -> Optional[float]:
    """Preference: Finishing > coverage phase (the mutation phase
    does not measure coverage, only mutation). When Finishing's
    coverage run gave no value, the result is None: the coverage phase measured
    another suite."""
    if result.finishing_result is not None:
        return result.finishing_result.coverage_final_line
    if result.test_result is not None:
        return result.test_result.line_coverage
    return None


def _final_mutation_score(result: TestPipelineResult) -> Optional[float]:
    """The mutation phase is the only source of a mutation score. The
    coverage phase is coverage-only, never a score."""
    if result.mutation_result is not None:
        score = result.mutation_result.mutation_score
        if score is not None:
            return score
    return None


#: Starter statuses whose cause is known. A pool not generated because a
#: model request failed is infrastructure, and generated suites of which none
#: is green are the method's outcome. Unresolved validation and a phase-end
#: selection without a complete candidate stay unjudged.
_STARTER_FAILURE_KINDS = {
    "failed_model_request": phase_state.INFRASTRUCTURE_FAILURE,
    "cancelled_zero_passing": phase_state.METHOD_FAILURE,
}


def _record_outcome(
    result: TestPipelineResult,
    phase: str,
    outcome: str,
    detail: Optional[str],
    *,
    failure_kind: Optional[str] = None,
) -> None:
    """Stamp the phase outcome onto the in-flight result. A failed phase
    carries its ``failure_kind`` (phase_state.FAILURE_KINDS) when it knows
    its cause. Without one the failure is unjudged."""
    result.phase_outcomes[phase] = outcome
    # A failed phase's artifact is replaced by the next attempt, so its time
    # counts for nothing kept. The starter keeps its generations across
    # attempts, which count once, in the attempt that produced them.
    if outcome == "failed" and phase != "starter":
        phase_state.discard_segment(phase)
    if detail:
        result.phase_outcomes[f"{phase}_detail"] = detail
    if failure_kind is not None:
        result.phase_outcomes[f"{phase}_failure_kind"] = failure_kind
