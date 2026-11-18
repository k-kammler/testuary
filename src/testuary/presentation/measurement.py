"""Display observations of LangAdapter's measurement runs: mutation
measurement progress (``MutationProgress``) and warmed suite timing
(``tests.benchmark_warmed``).

Mutation progress: the backend reports how many mutants are measured so far,
from a separate thread, where the engine can observe them, and then how many
kills its verification has checked. Each kind is a row of its own. Progress
is not evidence: a finished mutant's verdict can still change until the run
ends, so these observations never carry a score. The producer reports the
run's outcome with its own observation.

Warmed timing: values come from the backend's records. Nothing is
recomputed, estimated or filled in here. Headline summaries appear only for
a successful, complete result. Other results report their validity instead
of a number.
"""
from __future__ import annotations

from langadapter import Outcome
from langadapter.mutation import MutationProgress
from langadapter.runs import WarmedBenchmarkProgress, WarmedSuiteBenchmark

from .events import DisplayEvent, Metric, Provenance
from .view import KILL_VERIFICATION, MUTATION_MEASUREMENT

_MUTATION_OPERATION = MUTATION_MEASUREMENT
_MUTATION_LABEL = "Mutation testing"
_MUTATION_MESSAGE = "Mutant results can still change until mutation testing ends"
#: The progress kinds with a row of their own: ``operation``, ``label``,
#: ``unit`` and ``message``. Any other kind counts in the measurement's row,
#: in its own unit.
#: Each counts work done, not mutants killed: mutants read as mutants run.
_ROWS = {
    "kill verifications": (
        KILL_VERIFICATION, "Re-checking caught mutants", "re-checked",
        "Caught mutants without a named test that caught them, or cut by the time limit, are run again"),
}
_UNITS = {"mutants": "mutants run"}
#: How a row ends, by its label and the run's end state.
_ENDINGS = {"succeeded": "done", "failed": "failed", "cancelled": "stopped"}


def mutation_progress_ending(kind: str, state: str) -> str:
    """The message of the row of the progress ``kind`` when its mutation run
    ends in ``state``: succeeded, failed or cancelled."""
    label = _ROWS[kind][1] if kind in _ROWS else _MUTATION_LABEL
    return f"{label} {_ENDINGS[state]}"


def mutation_progress_event(progress: MutationProgress, *, run_id: str, target: str,
                            elapsed_s: float, phase: str = "mutation",
                            operation: str | None = None,
                            provenance: Provenance = "observed") -> DisplayEvent:
    """One progress record as a running operation observation.

    ``elapsed_s`` is the run clock the producer uses for its other
    observations. The record's own ``elapsed_s`` counts from the start of the
    mutation run and is not substituted for it. An unknown total stays
    unknown.
    """
    if not isinstance(progress, MutationProgress):
        raise TypeError("progress must be a MutationProgress")
    own, label, unit, message = _ROWS.get(progress.kind, (
        _MUTATION_OPERATION, _MUTATION_LABEL, _UNITS.get(progress.kind, progress.kind or "items"),
        _MUTATION_MESSAGE))
    return DisplayEvent(run_id=run_id, target=target, phase=phase,
                        operation=operation or own, state="running", elapsed_s=elapsed_s,
                        message=message, completed=progress.finished, total=progress.total,
                        work_unit=unit, provenance=provenance, level="operation",
                        label=label)


_WARMED_OPERATION = "warmed-timing"
_WARMED_LABEL = "Warmed suite timing"
_HEADLINES = (("mean_of_fork_means_ms", "Mean of fork means"),
              ("median_of_fork_medians_ms", "Median of fork medians"),
              ("sample_min_ms", "Sample minimum"),
              ("sample_max_ms", "Sample maximum"),
              ("between_fork_mean_sd_ms", "Between-fork sample SD"))
_VALIDITY = {"complete": "complete", "incomplete": "incomplete",
             "non_repeatable": "not repeatable", "no_samples": "no samples"}


def _warmed_event(*, run_id: str, target: str, phase: str, state: str, elapsed_s: float,
                  message: str, completed: int, total: int, work_unit: str,
                  metrics: tuple[Metric, ...] = (), provenance: Provenance) -> DisplayEvent:
    return DisplayEvent(run_id=run_id, target=target, phase=phase, operation=_WARMED_OPERATION,
                        state=state, elapsed_s=elapsed_s, message=message,
                        completed=completed, total=total, work_unit=work_unit,
                        metrics=metrics, provenance=provenance, level="operation",
                        label=_WARMED_LABEL)


def _warmed_state(outcome: Outcome | None, cancelled: bool = False) -> str:
    # Cancellation is the result's own flag. Outcome describes the native run.
    if cancelled:
        return "cancelled"
    return "succeeded" if outcome is Outcome.COMPLETED else "failed"


def warmed_progress_event(progress: WarmedBenchmarkProgress, *, run_id: str, target: str,
                          phase: str = "suite timing",
                          provenance: Provenance = "observed") -> DisplayEvent:
    """One progress record as a display observation. An entered iteration is not a sample."""
    if not isinstance(progress, WarmedBenchmarkProgress):
        raise TypeError("progress must be a WarmedBenchmarkProgress")
    fork = None if progress.fork is None else f"Fork {progress.fork + 1} of {progress.requested_forks}"
    if progress.stage == "setup":
        message = "Preparing the suite. Build and discovery are outside the timed interval"
    elif progress.stage == "fork_start":
        message = f"{fork} started in a new JVM"
    elif progress.stage in ("iteration_start", "iteration_end"):
        runs = progress.requested_warmup_runs if progress.kind == "warmup" else progress.requested_runs
        message = f"{fork}: {progress.kind} {progress.repetition + 1} of {runs}"
        if progress.stage == "iteration_start":
            message += " running"
        else:
            message += " finished"
            if progress.elapsed_ns is not None:
                message += f" in {progress.elapsed_ns / 1e6:.3f} ms (raw)"
            if progress.outcome is not None and progress.outcome is not Outcome.COMPLETED:
                message += f", outcome {progress.outcome.value.replace('_', ' ')}"
    elif progress.stage == "fork_end":
        message = f"{fork} finished"
    else:
        message = "Timing finished"
    if progress.reason:
        message += f", {progress.reason}"
    state = _warmed_state(progress.outcome) if progress.stage == "complete" else "running"
    return _warmed_event(run_id=run_id, target=target, phase=phase, state=state,
                         elapsed_s=progress.operation_elapsed_s or 0.0, message=message,
                         completed=progress.warmup_completed + progress.measurements_completed,
                         total=progress.requested_forks * (progress.requested_warmup_runs
                                                           + progress.requested_runs),
                         work_unit="iterations", provenance=provenance)


def warmed_result_event(result: WarmedSuiteBenchmark, *, run_id: str, target: str,
                        phase: str = "suite timing",
                        provenance: Provenance = "observed") -> DisplayEvent:
    """The final result: outcome, validity, counts and, only when usable, summaries."""
    if not isinstance(result, WarmedSuiteBenchmark):
        raise TypeError("result must be a WarmedSuiteBenchmark")
    usable = result.outcome is Outcome.COMPLETED and result.validity == "complete"
    boundary = ("around JUnit Launcher.execute"
                if result.timing_boundary == "junit_launcher_execute" else result.timing_boundary)
    mode = "Warmed JVM timing" if result.mode == "warmed_jvm" else f"{result.mode} timing"
    parts = [f"{mode} {boundary} ({result.clock}, startup and discovery excluded)",
             f"validity {_VALIDITY.get(result.validity, result.validity)}",
             f"{result.forks_completed} of {result.requested_forks} forks",
             f"{result.warmup_completed} warmups and {result.measurements_completed} measurements"]
    if result.outcome is not Outcome.COMPLETED:
        parts.insert(1, f"outcome {result.outcome.value.replace('_', ' ')}")
    if result.comparison == "not_assessed":
        parts.append("comparison not assessed")
    if any(sample == 0 for fork in result.forks for sample in fork.samples_ns):
        parts.append("zero-duration samples are limited by timer resolution")
    if not result.disposal_confirmed:
        parts.append("process disposal not confirmed")
    if not usable:
        parts.append("no timing summary")
    if result.reason:
        parts.append(result.reason)
    metrics = tuple(
        Metric(name, getattr(result, attribute), " ms",
               "measured" if getattr(result, attribute) is not None else "unavailable")
        for attribute, name in _HEADLINES) if usable else ()
    return _warmed_event(run_id=run_id, target=target, phase=phase,
                         state=_warmed_state(result.outcome, result.cancelled),
                         elapsed_s=result.operation_elapsed_s or 0.0, message=", ".join(parts),
                         completed=result.measurements_completed,
                         total=result.requested_forks * result.requested_runs,
                         work_unit="measurements", metrics=metrics, provenance=provenance)
