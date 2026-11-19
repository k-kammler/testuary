"""Post-phase suite consolidation: merge, measure, and set-cover.

At each phase boundary, candidate suites are merged into one file. A
phase with no inherited artifact merges its own pool. A later phase merges its
agent output with the suite inherited from the previous phase. The merged file
is measured per test, and a runtime-weighted set cover selects tests that carry
the measured line, branch, and mutation tokens.

This module builds and measures source-bound input, merged, and reduced
candidates. The boundary selector consumes those records without running the
mutation engine again. Evidence identities are retained for the audit but do
not decide eligibility.
"""
from __future__ import annotations

import math
import statistics
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, NamedTuple, Optional, Sequence, Tuple
from testuary import languages, phase_state, config, project
from testuary import evidence as validity
from testuary.generation_evidence import source_sha256
from testuary.phase_state import timing_fields
from testuary import loadable
from testuary import normal_run


#: Headroom factor for the per-test eviction measurement timeout over the plain
#: suite timeout. The instrumented run (branch coverage + a per-test dynamic
#: context, under isolation, plus a discarded warmup pass) is several times
#: heavier than a green check, so a slow-but-valid suite would otherwise be
#: dropped over the measurement-only cost.
_EVICTION_TIMEOUT_MULTIPLIER = 3

#: Bounded rounds of evidence repair: each round re-adds omitted carriers for
#: statement, branch, or killed-mutant identities that fresh measurement lost
#: against the merged matrix, then remeasures once. The bound caps work on a
#: reduction that is already an optimization. Any remaining identity
#: difference is recorded for audit, and independent candidate measurements
#: determine which suite is selected.
_EVIDENCE_REPAIR_ROUNDS = 3

_PHASE_END_SIDECARS = frozenset({
    "selected_kill_matrix.json",
    "merged_kill_matrix.json",
    "merged_kill_matrix_selective.json",
    "reduced_kill_matrix.json",
    "derived_reduced_kill_matrix.json",
    "measured_reduced_kill_matrix.json",
    "candidate2_kill_matrix.json",
    "candidate2_coverage_matrix.json",
    "selected_coverage_matrix.json",
    "starter_pool_kill_matrix.json",
    "starter_pool_attribution.json",
    "starter_pool_derivation.json",
    "starter_pool_derived_kill_matrices.json",
    "starter_pool_pooled_kill_matrix.json",
    "merged_coverage_matrix.json",
    "reduced_coverage_matrix.json",
    "consolidation_record.json",
    "consolidation_failure.json",
    "validity_merged.json",
    "validity_derivations.json",
    "validity_delivered.json",
})


@dataclass
class PhaseConsolidation:
    """Outcome of :func:`consolidate_phase_end`.

    The record binds the merged and reduced sources to the merged measurement,
    the reduced matrix derived as a cross-check, and the independently measured
    reduced matrix. ``reduced_source`` is absent when no reduced candidate can
    be constructed and validated.
    """

    merged_source: Optional[str] = None
    #: Whether the merged suite passes its normal run.
    merged_green: bool = True
    #: Why a tool could not measure the merged suite after its passing normal
    #: run, which leaves it without numbers. "" when it was measured.
    merged_incomplete: str = ""
    reduced_source: Optional[str] = None
    execution_name: str = ""
    target_revision: Optional[str] = None
    measurement_identity: Dict[str, Any] = field(default_factory=dict)
    structural_parser_identity: str = ""
    target_validator_identity: str = ""
    merged_source_sha256: str = ""
    reduced_source_sha256: str = ""
    #: True when set cover produced a passing reduced-suite candidate. This is
    #: independent of whether pruning changed the bytes: a valid cover may keep
    #: every test. Operational paths that can only offer the merged suite leave
    #: this false.
    reduction_valid: bool = False
    merged_km: Optional[dict] = None
    reduced_km: Optional[dict] = None
    derived_reduced_km: Optional[dict] = None
    measured_reduced_km: Optional[dict] = None
    line: Optional[float] = None
    branch: Optional[float] = None
    runtime_ms: Optional[float] = None
    #: The merged suite's test count, None while it was not counted.
    pool_tests: Optional[int] = None
    selected_tests: int = 0
    #: derived minus measured reduced-suite mutation score when both ran (the
    #: sanity check). 0.0 means the restriction-derive matched a fresh
    #: re-measure.
    derive_measure_delta: Optional[float] = None
    derive_measure_differences: List[Dict[str, Any]] = field(
        default_factory=list
    )
    #: The merged suite's per-test coverage/kill tokens (test id -> sorted
    #: token list) and the cover's kept-test ids, JSON-ready, so the caller
    #: can persist the union and reduced coverage sidecars without a re-run.
    per_test_tokens: Optional[dict] = None
    per_test_cost_ms: Optional[dict] = None
    effective_cost_us: Optional[dict] = None
    kept_tests: Optional[list] = None
    #: The kept-test set the run-time derive restricted the merged matrix
    #: with, recorded at the moment of restriction. Later fallbacks can
    #: replace ``kept_tests`` (which describes the delivered suite), so an
    #: offline replay of the derived matrix joins on this set, not that one.
    derive_kept_tests: Optional[list] = None
    #: Suite-level coverage from the same coverage.py run, including
    #: collection/import-time execution that has no per-test dynamic context.
    #: These exact sets are the phase-end comparison source. Per-test tokens
    #: remain the set-cover source.
    suite_covered_lines: Optional[List[int]] = None
    suite_covered_branches: Optional[List[str]] = None
    target_total_lines: Optional[int] = None
    target_total_branches: Optional[int] = None
    #: Aggregate coverage of the green merged union before set cover. The
    #: fields above describe the suite that can be delivered, which may differ
    #: after pruning because collection/import-time execution has no per-test
    #: token carrier.
    union_line: Optional[float] = None
    union_branch: Optional[float] = None
    union_covered_lines: Optional[List[int]] = None
    union_covered_branches: Optional[List[str]] = None
    #: The ordinary merged-suite coverage run completed successfully. This is
    #: independent of the later per-test eviction run: eviction can fail after
    #: aggregate line/branch evidence and the runnable test count are known.
    aggregate_coverage_measured: bool = False
    #: The starter phase's separately measured starter pool. The starter phase attaches
    #: it after consolidation so the native pool evidence is persisted beside
    #: the merged and reduced matrices.
    starter_pool_km: Optional[dict] = None
    #: Per-slot attributed mutation counts of the starter pool, with the run's
    #: attribution facts (``metrics.STARTER_POOL_ATTRIBUTION_SCHEMA``).
    starter_pool_attribution: Optional[dict] = None
    #: The starter suites' derivation record (``starter_selection.SCHEMA``),
    #: under ``suite_mutation`` derive or both.
    starter_pool_derivation: Optional[dict] = None
    #: Each starter suite's matrix derived from the pooled run, by suite.
    starter_pool_derived_kms: Optional[dict] = None
    #: The pooled run's matrix when the per-suite measurement is the starter
    #: pool matrix (both). Under derive the starter pool matrix is the pooled
    #: run's.
    starter_pool_pooled_km: Optional[dict] = None
    #: The phase's own new suite (candidate 2) measured standalone before the
    #: merge: its kill matrix (None without a mutation runner) and a
    #: coverage-matrix dict in the union-coverage-sidecar shape. A later audit
    #: ORs candidate 1 (the previous phase's delivered suite) with these to
    #: check the merge kept every kill and token either candidate had alone,
    #: and the phase end's selection scores candidate 2 on this measurement
    #: (candidate2_candidate). Set only at a two-candidate phase end. None
    #: otherwise or when the standalone measurement is red or fails. The
    #: merged suite takes this matrix only when it has candidate 2's bytes
    #: and execution name.
    candidate2_km: Optional[dict] = None
    candidate2_coverage: Optional[dict] = None
    candidate2_source: Optional[str] = None
    candidate2_execution_name: str = ""
    candidate2_green: bool = False
    candidate2_status_reason: str = ""
    candidate2_candidate: Optional["ScoredSuite"] = None
    #: The merged pool's summed measured per-test times, the runtime of the
    #: unreduced union when it is a keep-best candidate, None when a test
    #: has no measured time.
    pool_runtime_ms: Optional[float] = None
    #: The merge's collision-resolution outcome (renamed / deduped /
    #: autouse-localized counts, the renamed names when the merge renamed
    #: any, dropped "source_i:name" entries, ejected source count),
    #: persisted into the consolidation record so a cross-source drop is
    #: machine-auditable.
    collision_resolution: Optional[Dict[str, object]] = None
    #: Same-suite definitions removed from phase-end inputs before any
    #: standalone measurement or merge. Empty on every fresh normalized path.
    input_normalizations: List[Dict[str, object]] = field(default_factory=list)
    #: Tests the evidence-repair pass re-added to the cover, so a
    #: union-vs-delivered comparison can name every repair from the record.
    repaired_tests: List[str] = field(default_factory=list)
    evidence_repairs: List[Dict[str, Any]] = field(default_factory=list)
    #: Why the builder could not provide a valid reduced candidate. This is an
    #: operational record, not a separate phase-end selection rule.
    fallback_reason: Optional[str] = None
    fallback_candidates: List[str] = field(default_factory=list)
    #: Each run the mutation tool's origin check stopped at this phase end:
    #: the suite, the reason and every stopping test's name (the runner's
    #: ``stops``, the same list, so a later stop is recorded too).
    mutation_origin_stops: List[Dict[str, Any]] = field(default_factory=list)
    # When only the unreduced merge can be built, its size, rename-twin count,
    # attempted-reduction size, and redundancy ratio are recorded for review.
    fallback_union_stats: Optional[Dict[str, object]] = None
    #: Score-complete candidate census and evidence-identity differences for
    #: audit. These observations never participate in phase-end selection.
    selection_audit: Dict[str, object] = field(default_factory=dict)
    #: Provenance of the cover and mutation-score policy used at this boundary.
    solver: str = ""
    solver_metadata: Optional[Dict[str, Any]] = None
    mutation_method: str = ""
    #: Backend-owned generated-test validity records. The merged source is
    #: analyzed once after collision resolution and before execution. Sources
    #: produced only by test removal or restoration carry derivation records.
    validity_source_dir: str = ""
    merged_validity_record: Optional[Dict[str, Any]] = None
    validity_derivations: List[Dict[str, Any]] = field(default_factory=list)
    #: The removal fallback: each removal of the tests a failing
    #: measurement named and each failure that removed none, in order
    #: (``measurement_removals``), the validity derivation of each suite a
    #: removal changed, and each changed candidate's source by the name it is
    #: archived under unless it is delivered.
    measurement_removals: List[Dict[str, Any]] = field(default_factory=list)
    removal_derivations: List[Dict[str, Any]] = field(default_factory=list)
    removed_sources: Dict[str, str] = field(default_factory=dict)
    detail: str = ""


@dataclass(frozen=True)
class PhaseEndContext:
    """What every phase-end step runs with: the target's language backend,
    the target, the source directory, the test run's timeout, whether a suite
    is staged under its delivered name, and the log."""
    backend: Any
    target_file_path: str
    source_dir: str
    timeout: int
    stage_as_delivered: bool = False
    log: Callable[[str], None] = print


@dataclass(frozen=True)
class InheritedMatrix:
    """A mutation matrix measured before this phase end, with what it was
    measured on: the source's SHA-256, the execution name and the
    measurement identity."""
    km: Optional[dict] = None
    sha256: Optional[str] = None
    execution_name: Optional[str] = None
    measurement_identity: Optional[Dict[str, Any]] = None

    def bound_to(self, source_hash: Optional[str], execution_name: Optional[str],
                 identity: Optional[Dict[str, Any]]) -> bool:
        """Whether the matrix was measured on the source of ``source_hash``
        under ``execution_name`` and ``identity``."""
        return (self.sha256 == source_hash
                and self.execution_name == execution_name
                and self.measurement_identity == identity)


@dataclass(frozen=True)
class ReductionSettings:
    """A phase's suite reduction settings (its reduction_* settings): the
    solver, its time limit, the smallest run time counted for one test, the
    solver's random seed and the tie-break among optimal sets of tests."""
    solver: str
    solver_time_limit_seconds: float
    minimum_test_runtime_ms: float
    solver_random_seed: int
    tie_break: str


@dataclass(frozen=True)
class PhaseEndOptions:
    """How a phase end measures and reduces: how the reduced suite's
    mutation score is obtained, the suite reduction settings, the inherited
    matrices of candidate 1 and candidate 2, the measurement identity, each
    candidate's per-test evidence for the merge, the directory for a
    failed solver's record, and the label the driver's selection gives
    candidate 2, which names its changed source in the archive."""
    mutation_method: str
    reduction: ReductionSettings
    inherited1: InheritedMatrix = field(default_factory=InheritedMatrix)
    inherited2: InheritedMatrix = field(default_factory=InheritedMatrix)
    measurement_identity: Optional[Dict[str, Any]] = None
    candidate_evidence: Optional[Sequence[Optional[Dict[str, Sequence[str]]]]] = None
    solver_failure_dir: Optional[str] = None
    candidate2_label: str = "standalone_1"


def _diff_body(diff) -> Optional[str]:
    """The discriminating content of a mutation diff: its +/- lines only.
    Header lines (the a/b file names) and @@ hunk ranges render differently
    across capture eras, so an identity keyed on the raw string reads every
    kill as missing when two matrices from different eras are compared. The
    changed lines themselves are era-stable."""
    if diff is None:
        return None
    text = str(diff).split("--- mutation diff ---")[-1]
    return "\n".join(
        line for line in text.splitlines()
        if line.startswith(("+", "-"))
        and not line.startswith(("+++", "---")))


def _mutant_identity(mutant: dict) -> tuple:
    """Cross-capture mutant identity. A row the language backend keyed
    (``key``: the mutated file, the operator and its occurrence in the file)
    is identified by that key, which holds across runs on the same source
    and catalogue. Two mutants of one operator on one line with the same
    change differ only by occurrence. A row without a key falls back to
    (operator, line, diff body), since each capture may assign fresh ids.
    Minimal synthetic matrices may carry only an id. Use it only when no
    stable catalogue field is available."""
    key = mutant.get("key")
    if (isinstance(key, dict) and key.get("operator") is not None
            and key.get("occurrence") is not None):
        return ("key", key.get("file"), key["operator"], key["occurrence"])
    stable = (mutant.get("operator"), mutant.get("line"),
              _diff_body(mutant.get("diff")))
    if any(value is not None for value in stable):
        return ("catalogue", *stable)
    return ("id", mutant.get("id"))


def _killed_by_identity(km: Optional[dict]) -> Dict[Any, dict]:
    """The killed mutants of a matrix by their cross-capture identities."""
    from testuary.languages import mutation
    return {_mutant_identity(m): m for m in (km or {}).get("mutants", [])
            if m.get("status") in mutation.DETECTED_STATUSES}


def _coverage_sets(
    matrix: Optional[dict],
) -> tuple[Optional[FrozenSet[int]], Optional[FrozenSet[str]]]:
    """Exact suite-level statement lines and branch arcs from a sidecar:
    the coverage.py aggregate sets, which include collection and
    import-time execution.

    A sidecar without those sets returns ``None``, meaning the suite's
    coverage in this basis is unknown. The per-test ``L:``/``B:`` tokens
    are a different vocabulary, recording traced lines and line
    transitions, so their union answers a different question and is
    never substituted here: a caller comparing one against the other
    finds evidence missing from a suite that covers every line.
    """
    if not isinstance(matrix, dict):
        return None, None
    raw_lines = matrix.get("covered_statement_lines")
    raw_branches = matrix.get("covered_branch_arcs")
    if not isinstance(raw_lines, list) or not isinstance(raw_branches, list):
        return None, None
    return (
        frozenset(int(line) for line in raw_lines),
        frozenset(str(arc) for arc in raw_branches),
    )


def _stage_in(backend, directory, source: str, target_file_path: str, *,
              stage_as_delivered: bool) -> Path:
    """Write ``source`` into ``directory``, created when absent, under the
    scratch name the backend gives it. Returns the written path."""
    Path(directory).mkdir(parents=True, exist_ok=True)
    path = backend.scratch_test_path(
        str(directory), source, target_file_path,
        stage_as_delivered=stage_as_delivered)
    path.write_text(source, encoding="utf-8")
    return path


def _missing_evidence(res: "PhaseConsolidation") -> Tuple[set, set, set]:
    """The merged suite's covered lines, covered branches (as ``from->to``
    text) and killed mutants (by identity) that the reduced suite lacks."""
    lines = set(res.union_covered_lines or []) - set(res.suite_covered_lines or [])
    branches = ({str(x) for x in (res.union_covered_branches or [])}
                - {str(x) for x in (res.suite_covered_branches or [])})
    kills = (_killed_by_identity(res.merged_km).keys()
             - _killed_by_identity(res.reduced_km).keys())
    return lines, branches, kills


class _EvidenceRepair:
    """The bounded repair of the evidence a reduced suite lost against the
    merged suite: up to ``_EVIDENCE_REPAIR_ROUNDS`` rounds, each adding to a
    trial suite the cheapest unused carrier of every missing line, branch
    and kill that no test added before it carries, the trial adopted when it
    passes, is measured completely and scores no lower."""

    def __init__(self, context: PhaseEndContext, mutation_runner, *, merged: str,
                 res: "PhaseConsolidation", per_test_tokens, solver_cost_ms):
        self.backend = context.backend
        self.merged = merged
        self.target_file_path = context.target_file_path
        self.source_dir = context.source_dir
        self.timeout = context.timeout
        self.res = res
        self.per_test_tokens = per_test_tokens
        # The solver's costs: measured times, and imputed ones for untimed tests and the first test.
        self.solver_cost_ms = solver_cost_ms
        self.mutation_runner = mutation_runner
        self.stage_as_delivered = context.stage_as_delivered
        self.log = context.log
        self.merged_killed_by_identity = _killed_by_identity(res.merged_km)

    def carriers(self, prefix: str, identity: object, kept: set) -> List[str]:
        """The tests not kept whose tokens carry ``identity`` (a line ``L``, a
        branch ``B``, a named killer ``K`` or the test a crash ended ``C``),
        cheapest first."""
        if prefix == "L":
            suffix = f":{int(identity)}"
        elif prefix == "B":
            suffix = f":{identity}"
        else:
            suffix = ""
        candidates = []
        for test_id, tokens in self.per_test_tokens.items():
            if test_id in kept:
                continue
            text_tokens = {str(token) for token in tokens}
            if prefix in ("K", "C"):
                match = f"{prefix}:{identity}" in text_tokens
            else:
                match = any(
                    token.startswith(prefix + ":") and token.endswith(suffix)
                    for token in text_tokens
                )
            if match:
                candidates.append(test_id)
        candidates.sort(
            key=lambda test_id: (
                float(self.solver_cost_ms.get(test_id, 0.0)), test_id
            )
        )
        return candidates

    def kill_carriers(self, identity: object, kept: set) -> List[str]:
        """The carriers of a missing kill: its named killers, else the test
        its crash ended, else the tests reaching its line."""
        mutant = self.merged_killed_by_identity[identity]
        carriers = (self.carriers("K", mutant.get("id"), kept)
                    or self.carriers("C", mutant.get("id"), kept))
        if not carriers and mutant.get("line") is not None:
            # Every recorded killer is kept and the kill still does not
            # land: a test reaching the mutated line changes what runs
            # beside the killer.
            carriers = self.carriers("L", mutant["line"], kept)
        return carriers

    def choose(self, missing: Tuple[set, set, set], kept: set) -> Tuple[set, List[str]]:
        """The cheapest carrier of each missing line, branch and kill, lines
        first, unless a test already added carries it, and the items no
        carrier has."""
        missing_lines, missing_branches, missing_kills = missing
        wanted = [(f"L:{line}", self.carriers("L", line, kept))
                  for line in sorted(missing_lines)]
        wanted += [(f"B:{branch}", self.carriers("B", branch, kept))
                   for branch in sorted(missing_branches)]
        wanted += [(f"K:{identity}", self.kill_carriers(identity, kept))
                   for identity in sorted(missing_kills, key=str)]
        additions: set = set()
        without_carrier: List[str] = []
        for item, carriers in wanted:
            if not carriers:
                without_carrier.append(item)
            elif not additions.intersection(carriers):
                additions.add(carriers[0])
        return additions, without_carrier

    def trial(self, trial_kept: set, attempt: Dict[str, Any]):
        """The trial suite of ``trial_kept``: pruned from the merged suite,
        validated, staged, run and measured. Its (source, validation,
        coverage run, kill matrix), or None with the attempt's outcome set.
        The mutation run happens in the trial's scratch directory, which exists
        while it runs. Under derive the trial's matrix is the merged matrix
        restricted to the trial's tests, with no engine run."""
        remove = [t for t in self.per_test_tokens if t not in trial_kept]
        trial_source = self.backend.prune_tests(self.merged, remove) if remove else self.merged
        validated = self._validated(trial_source, attempt)
        if validated is None:
            return None
        trial_source, validation = validated
        with project.scratch_dir(prefix="testuary_phaseend_repair_") as td:
            rdir = Path(td) / "repaired"
            rpath = _stage_in(self.backend, rdir, trial_source, self.target_file_path,
                              stage_as_delivered=self.stage_as_delivered)
            # The normal run decides whether the trial passes. The coverage
            # tool then measures it.
            coverage_run, passed = _check_then_measure(
                self.backend, [rpath], self.source_dir, timeout=self.timeout,
                measure_coverage_of=self.target_file_path, log=self.log)
            if not passed:
                loaded = self._loadable(trial_source, rpath, attempt)
                if loaded is None:
                    return None
                trial_source, validation, coverage_run = loaded
            if not getattr(coverage_run, "ok", False):
                # It passes, but the coverage tool could not measure it: a
                # trial without numbers is not kept.
                attempt["outcome"] = "coverage_measurement_incomplete"
                return None
            trial_km = (_derive_kill_matrix_for_tests(
                self.res.merged_km, self.per_test_tokens, trial_kept)
                if self.res.mutation_method == "derive" else self.mutation_runner(rdir))
        if trial_km is None:
            attempt["outcome"] = "mutation_measurement_incomplete"
            return None
        return trial_source, validation, coverage_run, trial_km

    def _validated(self, source: str, attempt: Dict[str, Any]):
        """``source`` finalized, with its validation, or None with the
        attempt's outcome set when it fails validation."""
        source, validation, error = _finalize_and_validate_source(
            self.backend, source, self.source_dir)
        if error is not None:
            attempt["outcome"] = f"invalid: {error}"
            return None
        return source, validation

    def _loadable(self, trial_source: str, rpath: Path, attempt: Dict[str, Any]):
        """A trial whose normal run failed, after the backend's load repairs:
        its (source, validation, coverage run) when a repair that removes no
        test lets it pass, else None with the attempt's outcome set. The
        repaired bytes are finalized and validated again, as the reduced
        suite's are. A trial that does not load usually has a carrier that
        calls a member the reduction removed."""
        loaded, repairs = loadable.keep_if_loadable(
            self.backend, trial_source, self.source_dir, timeout=self.timeout,
            target_file_path=self.target_file_path,
            validate=lambda text: validate_judged(self.backend, text, self.source_dir),
            log=self.log, label="consolidate", stage_as_delivered=self.stage_as_delivered)
        if repairs:
            attempt["load_repairs"] = repairs
        if not repairs or loadable.fewer_tests(self.backend, trial_source, loaded):
            attempt["outcome"] = "does_not_load" if repairs else "not_passing"
            return None
        validated = self._validated(loaded, attempt)
        if validated is None:
            return None
        loaded, validation = validated
        rpath.write_text(loaded, encoding="utf-8")
        coverage_run, passed = _check_then_measure(
            self.backend, [rpath], self.source_dir, timeout=self.timeout,
            measure_coverage_of=self.target_file_path, log=self.log)
        if not passed:
            attempt["outcome"] = "not_passing"
            return None
        return loaded, validation, coverage_run

    def regresses(self, trial, attempt: Dict[str, Any]) -> bool:
        """Whether the trial ``trial`` (trial()'s result) is not kept: its
        mutation score or line coverage is unmeasured, or its (mutation
        score, branch, line) is lower than the reduced suite's. Sets the
        attempt's outcome when it is not kept."""
        from testuary.languages import mutation

        _source, _validation, coverage_run, trial_km = trial
        before_score = (
            mutation.matrix_score_percent(self.res.reduced_km), self.res.branch, self.res.line,
        )
        after_score = (
            mutation.matrix_score_percent(trial_km),
            getattr(coverage_run, "branch_coverage", None),
            getattr(coverage_run, "line_coverage", None),
        )
        if after_score[0] is None or after_score[2] is None:
            attempt["outcome"] = "aggregate_measurement_incomplete"
            return True
        before_cmp = tuple(
            float("-inf") if value is None else float(value)
            for value in before_score
        )
        after_cmp = tuple(
            float("-inf") if value is None else float(value)
            for value in after_score
        )
        if after_cmp < before_cmp:
            attempt["outcome"] = "aggregate_regression"
            return True
        return False

    def adopt(self, kept: set, additions: set, trial) -> None:
        """The reduced suite becomes the trial ``trial`` (its source,
        validation, coverage run and kill matrix) with the tests ``kept``,
        of which ``additions`` are the round's repairs."""
        trial_source, validation, coverage_run, trial_km = trial
        self.res.reduced_source = trial_source
        self.res.reduced_source_sha256 = validation.source_sha256
        self.res.target_validator_identity = validation.validator_identity
        self.res.reduced_km = trial_km
        if self.res.mutation_method == "derive":
            self.res.derived_reduced_km = trial_km
            self.res.derive_kept_tests = sorted(kept)
        else:
            self.res.measured_reduced_km = trial_km
        self.res.line = getattr(coverage_run, "line_coverage", None)
        self.res.branch = getattr(coverage_run, "branch_coverage", None)
        self.res.suite_covered_lines = sorted(
            int(line) for line in (coverage_run.covered_lines or ())
        )
        self.res.suite_covered_branches = sorted(
            str(branch) for branch in (coverage_run.covered_branches or ())
        )
        self.res.selected_tests = len(kept)
        self.res.kept_tests = sorted(kept)
        self.res.runtime_ms = _measured_time_ms(self.res.per_test_cost_ms, kept)
        self.res.repaired_tests = sorted(set(self.res.repaired_tests) | additions)
        if self.res.mutation_method == "both":
            _refresh_derive_cross_check(
                self.res,
                per_test_tokens=self.per_test_tokens,
                kept=kept,
                measured_km=trial_km,
                log=self.log,
            )

    def run(self, kept: set) -> None:
        """The rounds over the kept tests ``kept``, then the exhausted record
        when evidence is still missing after the last. An adopted trial's
        tests are on ``res.kept_tests``, every round on
        ``res.evidence_repairs``."""
        for round_no in range(1, _EVIDENCE_REPAIR_ROUNDS + 1):
            missing = _missing_evidence(self.res)
            missing_lines, missing_branches, missing_kills = missing
            if not (missing_lines or missing_branches or missing_kills):
                return

            additions, without_carrier = self.choose(missing, kept)
            attempt = {
                "round": round_no,
                "missing_lines_before": sorted(missing_lines),
                "missing_branches_before": sorted(missing_branches),
                "missing_kills_before": sorted(map(str, missing_kills)),
                "added_tests": sorted(additions),
                "without_carrier": sorted(without_carrier),
                "accepted": False,
            }
            self.res.evidence_repairs.append(attempt)
            if not additions:
                attempt["outcome"] = "no_carrier"
                self.log("[consolidate] evidence repair stopped: no unused carrier remains")
                return
            self.log(
                f"[consolidate] evidence-repair round {round_no}: adding "
                f"{len(additions)} carrier(s) for {len(missing_lines)} line, "
                f"{len(missing_branches)} branch, and {len(missing_kills)} "
                "mutant difference(s)"
            )
            trial_kept = set(kept) | additions
            trial = self.trial(trial_kept, attempt)
            if trial is None or self.regresses(trial, attempt):
                return
            kept = trial_kept
            self.adopt(kept, additions, trial)
            attempt["accepted"] = True
            attempt["outcome"] = "accepted"
            self.log(
                f"[consolidate] evidence-repair round {round_no}: adopted "
                f"{len(kept)} test(s)"
            )
        missing_lines, missing_branches, missing_kills = _missing_evidence(self.res)
        if missing_lines or missing_branches or missing_kills:
            self.res.evidence_repairs.append({
                "round": _EVIDENCE_REPAIR_ROUNDS,
                "missing_lines_after": sorted(missing_lines),
                "missing_branches_after": sorted(missing_branches),
                "missing_kills_after": sorted(map(str, missing_kills)),
                "added_tests": [],
                "without_carrier": [],
                "accepted": False,
                "outcome": "repair_budget_exhausted",
            })
            self.log("[consolidate] evidence repair stopped: no repair round left")


#: How a run's verdict names a run its diagnostics mark as ended without a
#: test result, by the diagnostic's kind, before the diagnostic's message.
_ENDED_BY = {
    "compile_error": "It ended as a collection failure",
    "coverage_counts_inconsistent": "The coverage tool could not read its coverage counts",
}


def red_verdict(run) -> str:
    """Why a suite run was not green, in one clause, with the failing tests
    the runner's report names, and how the run ended where its diagnostics
    say so (``_ENDED_BY``).

    A run that exceeded its timeout and a run whose tests failed both arrive
    as ``ok=False`` and lead to the same decision, so the record has to
    distinguish them: the first is a statement about the machine and the
    second about the suite."""
    if getattr(run, "timed_out", False):
        return "Verdict: timed out."
    counts = ", ".join(
        f"{name}={getattr(run, name, 0)}"
        for name in ("collected", "passed", "failed", "errors")
    )
    failing = [failure.test for failure in getattr(run, "failures", ()) or ()]
    named = f" Failing: {', '.join(failing)}." if failing else ""
    ended = "".join(f" {_ENDED_BY[entry.kind]}: {entry.message}."
                    for entry in getattr(run, "diagnostics", ()) if entry.kind in _ENDED_BY)
    return f"Verdict: exit {getattr(run, 'returncode', '?')} ({counts}).{named}{ended}"


#: The target validator's verdicts that judge no code: it timed out, or it
#: did not start or answer.
VALIDATOR_UNRESOLVED = ("timeout", "validator_error")
#: Validations of one source before it counts as unresolved.
VALIDATION_ATTEMPTS = 3


class ValidationUnresolved(Exception):
    """The target validator judged a source in none of its attempts."""


class ValidityUnresolved(ValidationUnresolved):
    """The generated-test validity gate the language offers gave no answer
    for a source: its target identity, analysis or screening did not
    complete. The source stays unjudged, as one the target validator did
    not judge."""


def validate_judged(backend, source: str, source_dir: str):
    """The target validator's verdict on ``source``, asked again while it
    judges nothing, or ``ValidationUnresolved`` when no attempt judged it. The
    starter phase keeps a suite it raises for. At a phase end it ends the
    phase, since an unjudged suite's place in the selection is unknown."""
    for _ in range(VALIDATION_ATTEMPTS):
        validation = backend.validate_test_source(source, source_dir)
        if validation.status not in VALIDATOR_UNRESOLVED:
            return validation
    raise ValidationUnresolved(f"{validation.status}: {validation.message}")


def _finalize_and_validate_source(backend, source: str, source_dir: str):
    """Return final bytes, target verdict, and an optional failure detail.
    A source the structural parser rejects is not validated (its verdict is
    None). One the validator does not judge raises ``ValidationUnresolved``."""
    finalized = backend.finalize_test_source(source)
    structure = backend.check_structure(finalized)
    if not structure.ok:
        from testuary.agent._helpers import structure_message
        return finalized, None, f"structural parser: {structure_message(structure)}"
    validation = validate_judged(backend, finalized, source_dir)
    if not validation.ok:
        return (
            finalized,
            validation,
            f"target validator {validation.status}: {validation.message}",
        )
    return finalized, validation, None


def _merge_sources(
    backend,
    sources: Sequence[str],
    log: Callable[[str], None],
    *,
    outcome: Dict[str, object],
    evidence: Sequence[Optional[Dict[str, Sequence[str]]]],
    source_dir: str,
) -> Optional[str]:
    """Merge sources into one merged suite, resolving name collisions so they
    share one namespace. Returns the merged source, or None when the
    complete merge is unavailable or nothing survives. ``outcome`` receives
    what the collision resolution did, and why a merge is unavailable.

    Reconciliation is structural first, ejection last, at three levels:

    - ``resolve_test_collisions`` dedups byte-identical colliding definitions
      and renames differing ones by identifier rewrite. A single binding that
      cannot be alias-renamed is dropped with its dependents. A source that
      will not parse, or whose rewrite will not re-parse, is ejected, which
      makes the complete merge unavailable.
    - each resolved source is merged pairwise. A failed rewrite or merge makes
      the complete merge unavailable rather than dropping one input.

    The third level is at the caller: after this returns, the merged suite's
    normal run decides (``_check_then_measure``), and one whose tests pass
    alone yet fail in the merge, a conflict that exists only once merged, is
    no candidate."""
    with phase_state.timed_part(phase_state.TIME_MERGE):
        return _merge_sources_inner(
            backend, sources, log,
            outcome=outcome, evidence=evidence, source_dir=source_dir)


def _merge_sources_inner(
    backend,
    sources: Sequence[str],
    log: Callable[[str], None],
    *,
    outcome: Dict[str, object],
    evidence: Sequence[Optional[Dict[str, Sequence[str]]]],
    source_dir: str,
) -> Optional[str]:
    # Preserve byte identity when every input is the same delivered suite.
    # Running structurally identical sources through the generic pairwise
    # merger can reformat imports or whitespace even though no merge work is
    # needed. That cosmetic rewrite would invalidate a source-bound matrix and
    # force an otherwise redundant mutation measurement.
    if (len(sources) > 1
            and all(source == sources[0] for source in sources[1:])):
        outcome.update({
            "renamed": 0,
            "deduped": 0,
            "autouse_localized": 0,
            "dropped": [],
            "ejected_sources": 0,
            "identical_sources": len(sources) - 1,
        })
        log(
            "[consolidate] merge reused one byte-identical source for "
            f"{len(sources)} identical candidates"
        )
        return sources[0]

    # A module-level autouse fixture applies to every test in its module.
    # Merging independent suites into one module would therefore extend each
    # such fixture to every other input's tests: one suite that monkeypatches
    # the module under test would stub it for the whole merge, and the other
    # inputs' tests would fail and be stripped. The merge keeps every fixture
    # on the tests of the source it came from. A single source has nothing to
    # widen, so it is left byte for byte as written: rewriting it would change
    # the hash its matrices are bound to for no effect.
    resolution = backend.resolve_test_collisions(
        list(sources), localize_autouse=len(sources) > 1, evidence=evidence)
    # Every definition-level outcome is named: a rename keeps both bodies, a
    # dedup shares one byte-identical copy, and a DROP removes a body (a
    # binding whose rename cannot be proven safe: the candidates come
    # without shadowed duplicates, _finalize_candidates removed them). A drop
    # is the one outcome that can cost tests, so it is never silent.
    n_ren = sum(len(r) for r in resolution.renames) if resolution.renames else 0
    n_ded = sum(len(d) for d in resolution.deduped) if resolution.deduped else 0
    n_autouse = (
        sum(len(names) for names in resolution.localized_autouse)
        if getattr(resolution, "localized_autouse", None) else 0
    )
    dropped = [(i, name) for i, names in enumerate(resolution.dropped_defs)
               for name in names] if resolution.dropped_defs else []
    # Every boundary's record carries what the resolution did, so a
    # cross-source drop is found in the record, not by searching the log.
    outcome["renamed"] = n_ren
    if n_ren:
        # The names the rename gave, which tell a renamed copy from a test
        # that carries such a suffix of its own.
        outcome["renamed_names"] = sorted(
            {new for pairs in resolution.renames for new in pairs.values()})
    outcome["deduped"] = n_ded
    outcome["autouse_localized"] = n_autouse
    outcome["dropped"] = [f"source_{i}:{name}" for i, name in dropped]
    outcome["ejected_sources"] = len(resolution.ejected)
    if n_ren or n_ded or n_autouse or dropped:
        log(f"[consolidate] merge collision resolution: {n_ren} renamed, "
            f"{n_ded} deduped, {n_autouse} autouse fixture(s) localized, "
            f"{len(dropped)} definition(s) dropped")
    for i, name in dropped:
        log(f"[consolidate] merge dropped definition '{name}' from source {i} "
            f"(shadowed duplicate or unrenamable binding)")
    if resolution.ejected:
        reasons = list(getattr(resolution, "eject_reasons", ()) or ())
        why = ", ".join(
            f"source {index}: {reasons[n] if n < len(reasons) else 'ejected'}"
            for n, index in enumerate(resolution.ejected))
        log(
            "[consolidate] collision rewriting could not preserve every "
            f"input, so the complete merge is unavailable ({why})"
        )
        outcome["unavailable"] = f"collision rewriting ejected {why}"
        return None
    kept = list(resolution.sources)
    if not kept:
        outcome["unavailable"] = "no source survived collision rewriting"
        return None
    for i, rewritten in enumerate(kept):
        structure = backend.check_structure(rewritten)
        if not structure.ok or not validate_judged(
                backend, rewritten, source_dir).ok:
            log(
                "[consolidate] collision rewrite failed validation for "
                f"source {i}, so the complete merge is unavailable"
            )
            outcome["unavailable"] = (
                f"the collision rewrite of source {i} failed validation")
            return None
    merged = kept[0]
    for s in kept[1:]:
        combined = backend.merge_test_files(merged, s)
        if combined is None:
            log(
                "[consolidate] a merge step failed, so the complete merge is "
                "unavailable"
            )
            outcome["unavailable"] = "the language could not merge two sources"
            return None
        structure = backend.check_structure(combined)
        if not structure.ok or not validate_judged(
                backend, combined, source_dir).ok:
            log(
                "[consolidate] a merged source failed validation, so the "
                "complete merge is unavailable"
            )
            outcome["unavailable"] = "a merged source failed validation"
            return None
        merged = combined
    return merged


def _removal_rounds() -> int:
    """The removal fallback's bound of rounds per suite and place,
    ``[starter] failing_test_cleanup_rounds``."""
    return int(config.get_agent_config("test_initial")["failing_test_cleanup_rounds"])


def _mutation_failure(runner: Any, directory: Path, km: Optional[dict]) -> Optional["_Failed"]:
    """A mutation measurement of ``directory`` as the removal fallback reads
    it: without a matrix, the tests the runner's checks named there
    (``named_of``, none from a runner that names none), with a matrix
    stamped incomplete (a time limit left mutants untested), a timed-out
    failure, else None."""
    from testuary.languages.mutation import INCOMPLETE_KEY

    if km is not None:
        return _Failed("mutation", timed_out=True) if km.get(INCOMPLETE_KEY) else None
    named_of = getattr(runner, "named_of", None)
    return _Failed("mutation", tuple(named_of(directory)) if named_of is not None else ())


def _measured_removing(measure: Callable[[str, List["_Failed"], bool], Any], source: str, *,
                       backend, place: str, log: Callable[[str], None],
                       ) -> Tuple[str, Any, List[Dict[str, Any]]]:
    """``source`` measured at a place of the removal fallback by
    ``measure(source, failed_out, first)``, which appends to ``failed_out``
    the failure of a measurement after the passing normal run, or the cause
    of a failing normal run: the tests a failure names are removed round by
    round (:func:`_remove_named_failures`), and each changed source is
    measured again (``first`` false). A failing normal run, a failure
    without a measurement, before any removal is no measurement failure.
    Returns the source, its last measurement and the record's entries."""
    failed: List[_Failed] = []
    last = [measure(source, failed, True)]
    if not failed or not failed[0].measurement:
        return source, last[0], []
    prior = [failed[0]]

    def again(changed: str) -> Optional[_Failed]:
        out: List[_Failed] = []
        last[0] = measure(changed, out, False)
        now = out[0] if out else None
        if now is not None and not now.measurement:
            now = now._replace(measurement=prior[0].measurement)
        prior[0] = now or prior[0]
        return now

    changed, _failed, entries = _remove_named_failures(
        backend, source, place=place, failed=failed[0], measure=again,
        rounds=_removal_rounds(), log=log)
    return changed, last[0], entries


def _note_removals(context: "PhaseEndContext", res: "PhaseConsolidation", place: str,
                   entries: List[Dict[str, Any]], parent: str, child: str,
                   archive_name: Optional[str] = None) -> bool:
    """A place's removal ``entries`` on ``res``, in the order the removals
    ran (the inherited suite's, measured before the phase end, before the
    places already noted), and when they changed the suite from ``parent``
    to ``child`` its validity derivation and, under ``archive_name``, its
    changed source. Returns whether they changed it."""
    at = 0 if place == "phase_end.inherited" else len(res.measurement_removals)
    res.measurement_removals[at:at] = entries
    removed = [item["test"] for entry in entries for item in entry.get("removed", ())]
    if not removed:
        return False
    res.removal_derivations.append(validity.derivation_record(
        identity=context.backend.generated_test_target_identity(
            context.target_file_path, context.source_dir),
        parent_source=parent, child_source=child, operation="measurement_remove",
        removed_tests=removed, stage=place))
    if archive_name is not None:
        res.removed_sources[archive_name] = child
    return True


class _Failed(NamedTuple):
    """A measurement that failed a suite whose normal run passed, as a place
    of the removal fallback reads it: which measurement, the id
    objects of the tests it names with each one's per-test outcome (None
    where the name carries none), whether it timed out, a cause that ends
    the removal (such as a failing normal run after one), and the place's own
    result for the place's own branch."""

    measurement: str
    tests: Tuple[Any, ...] = ()
    outcomes: Tuple[Optional[str], ...] = ()
    timed_out: bool = False
    cause: str = ""
    result: Any = None


def _without_tests(backend, source: str, test_ids: Sequence[str]) -> Tuple[str, Any]:
    """``source`` without the tests ``test_ids`` names, in the listing form
    the starter cleanup prunes, finalized, with the language's account of
    the removal (``PruneOutcome``): the one removal step the cleanup and the
    removal fallback share."""
    from testuary.languages.generic import prune_report

    outcome = prune_report(backend, source, test_ids)
    return backend.finalize_test_source(outcome.source), outcome


def _removed_entry(place: str, failed: _Failed, round_: int, **fields: Any) -> Dict[str, Any]:
    return {"place": place, "measurement": failed.measurement, "round": round_, **fields}


def _end_cause(failed: _Failed, round_: int, rounds: int) -> str:
    """Why the removal ends at ``failed`` after ``round_`` of its ``rounds``
    rounds, or "" when it removes the tests ``failed`` names."""
    return (failed.cause or ("the measurement timed out" if failed.timed_out else "")
            or ("the measurement named no test" if not failed.tests else "")
            or ("the rounds are spent" if round_ == rounds else ""))


def _named_parts(backend, source: str, listing: List[str], failed: _Failed,
                 detail: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """A round's removed tests, each with its outcome where the run gave one,
    and its other fields: the named tests ``source`` does not hold
    (``unmatched``) and why a named case's test went whole (``detail``). A
    named test is held when its removal alone changes ``source``."""
    from testuary.languages.generic import prune_report

    held = {text for text in listing if prune_report(backend, source, [text]).source != source}
    removed = [{"test": text, **({"outcome": out} if out else {})}
               for text, out in zip(listing, failed.outcomes or (None,) * len(listing))
               if text in held]
    unmatched = [text for text in listing if text not in held]
    return removed, {**({"unmatched": unmatched} if unmatched else {}),
                     **({"detail": detail} if detail else {})}


def _remove_named_failures(backend, source: str, *, place: str,
                           failed: _Failed, measure: Callable[[str], Optional[_Failed]],
                           rounds: int, log: Callable[[str], None],
                           ) -> Tuple[str, Optional[_Failed], List[Dict[str, Any]]]:
    """``source`` without the tests its failing measurement names, round by
    round, from the place's first measurement failure ``failed``:
    ``measure(source)`` runs the place's normal run and measurements again
    and gives None when they pass, else the failure. A failure that timed
    out, names no test, names only tests the suite does not hold, or comes
    after ``rounds`` removals ends it with an ``unnamed`` entry. Returns the
    source, the last failure (None when the measurements pass) and the
    record's entries."""
    from testuary.languages.generic import case_id

    # The removal's own project context, as the prune's.
    adapter = backend.project_backend()
    entries: List[Dict[str, Any]] = []
    round_ = 0
    while failed is not None:
        cause = _end_cause(failed, round_, rounds)
        if cause:
            entries.append(_removed_entry(place, failed, round_ + 1, unnamed=cause))
            break
        listing = [case_id(adapter, test) for test in failed.tests]
        changed, outcome = _without_tests(backend, source, listing)
        removed, fields = _named_parts(backend, source, listing, failed, outcome.detail)
        if not removed or changed == source:
            entries.append(_removed_entry(place, failed, round_ + 1, **fields,
                                          unnamed="the suite holds no named test"))
            break
        round_ += 1
        entries.append(_removed_entry(place, failed, round_, removed=removed, **fields))
        log(f"[consolidate] {place}: removed {', '.join(item['test'] for item in removed)}, "
            f"which its {failed.measurement} measurement failed")
        source = changed
        failed = measure(source)
    return source, failed, entries


def _failed_run(measurement: str, run: Any) -> _Failed:
    """A measurement's failing run (a coverage run's ``TestRunResult``) as the
    removal fallback reads it: the id objects of the tests it failed, then
    those its ``coverage_counts_inconsistent`` diagnostics name and it did
    not fail (the places' own reader of them), whether it timed out, and the
    per-test outcomes, the failures' and none for a diagnostic's test, ()
    where the failures do not pair with the failed tests."""
    tests = tuple(getattr(run, "stopping_tests", ()) or ())
    failures = tuple(getattr(run, "failures", ()) or ())
    counted = tuple(dict.fromkeys(
        entry.test for entry in getattr(run, "diagnostics", ()) or ()
        if entry.kind == "coverage_counts_inconsistent" and entry.test is not None
        and entry.test not in tests))
    outcomes = (tuple(item.outcome for item in failures) + (None,) * len(counted)
                if len(failures) == len(tests) else ())
    return _Failed(measurement, tests + counted, outcomes,
                   getattr(run, "timed_out", None) is True, result=run)


def _removal_cause(backend, source: str, run: Any) -> str:
    """Why ``source``, a suite after a removal, cannot stay, from its normal
    run ``run``: the removal left no test in it by the backend's structural
    count, or the run fails. "" when it passes."""
    if run.ok and not run.timed_out and run.passed:
        return ""
    if not backend.test_metrics(source).get("test_count"):
        return "the removal left no test"
    return "the normal run failed after a removal"


def _check_then_measure(
    backend,
    file_paths: Sequence[Path],
    source_dir: str,
    *,
    timeout: int,
    measure_coverage_of: str,
    log: Callable[[str], None],
    coverage_matrix_path: Optional[str] = None,
) -> Tuple[Any, bool]:
    """Whether a staged suite passes, without removing a test: the normal run
    (:mod:`testuary.normal_run`) decides, and only a suite that passes is
    measured, by the coverage tool's run of the target
    ``measure_coverage_of``. A phase end chooses only among suites that pass
    with complete numbers. One that does not is left out with its reason
    (:func:`red_verdict`). Returns the normal run when it fails, else the
    coverage tool's run, red when the tool could not measure the suite (its
    failing tests are logged), and whether the normal run passed."""
    paths = [str(p) for p in file_paths]
    verdict = normal_run.run(backend, paths, source_dir, timeout=timeout,
                             target=measure_coverage_of)
    if not getattr(verdict, "ok", False):
        return verdict, False
    measured = backend.run_test_files(
        paths, source_dir, timeout=timeout,
        measure_coverage_of=measure_coverage_of,
        coverage_matrix_path=coverage_matrix_path,
    )
    if not getattr(measured, "ok", False):
        log("[consolidate] the suite passes its normal run, but the coverage "
            f"tool could not measure it: {red_verdict(measured)}")
    return measured, True


def _covered_lines_of(tokens) -> set:
    """The line numbers a token set covers, parsed from the shared
    ``L:<name>:<line>`` coverage-token schema every backend emits."""
    out: set = set()
    for tok in tokens:
        if tok.startswith("L:"):
            try:
                out.add(int(tok.rsplit(":", 1)[1]))
            except (ValueError, IndexError):
                pass
    return out


def _killed_ids_of(tokens) -> set:
    """The mutant ids a token set kills by assertion, from the shared
    ``K:<mutant_id>`` kill-token schema."""
    return {tok[2:] for tok in tokens if tok.startswith("K:")}


def _log_catalogue_diff(derived_km, measured_km,
                        log: Callable[[str], None]) -> List[Dict[str, Any]]:
    """Name the mutants where a derived matrix and a fresh re-measure disagree
    (id -> status), so a derive-vs-measure delta is never left as variance.
    The two are matched by cross-capture identity. A mutant is named by its
    shown id (the stable ``mutant_id``) in the re-measure, else in the
    derived matrix."""
    from testuary.languages.mutation import shown_mutant_id

    names: Dict[object, object] = {}

    def _status(km) -> Dict[object, object]:
        statuses = {}
        for m in km.get("mutants") or []:
            identity = _mutant_identity(m)
            names[identity] = shown_mutant_id(m)
            statuses[identity] = m.get("status")
        return statuses
    d, me = _status(derived_km), _status(measured_km)
    diffs = [(names[identity], d.get(identity), me.get(identity))
             for identity in set(d) | set(me)
             if d.get(identity) != me.get(identity)]
    for mid, ds, ms in sorted(diffs, key=lambda x: str(x[0]))[:20]:
        log(f"[consolidate]   mutant {mid}: derived={ds} measured={ms}")
    if len(diffs) > 20:
        log(f"[consolidate]   ... and {len(diffs) - 20} more disagreeing mutants")
    return [
        {
            "mutant_id": mid,
            "derived_status": derived_status,
            "measured_status": measured_status,
        }
        for mid, derived_status, measured_status in sorted(
            diffs, key=lambda item: str(item[0])
        )
    ]


def _measured_time_ms(per_test_time_ms, tests) -> Optional[float]:
    """The summed measured times of ``tests``, exactly rounded, so that their
    order never changes the sum, None when one of them has no measured time:
    a missing measurement is never zero."""
    times = [(per_test_time_ms or {}).get(test) for test in tests]
    return None if any(time is None for time in times) else math.fsum(times)


def _derive_kill_matrix_for_tests(merged_km, per_test_tokens, kept):
    """Restrict one merged matrix to the evidence carried by ``kept``. The
    result describes the kept tests, not the merged bytes, so a binding the
    merged matrix carries (a reused stored matrix's) is cleared."""
    from testuary.languages import mutation

    killed_ids: set = set()
    covered_lines: set = set()
    for test_id in kept:
        tokens = {str(token) for token in per_test_tokens.get(test_id, ())}
        killed_ids.update(_killed_ids_of(tokens))
        covered_lines.update(_covered_lines_of(tokens))
    derived = mutation.restrict_kill_matrix(
        merged_km,
        kept_killed_ids=killed_ids,
        kept_lines=covered_lines,
    )
    if "source_sha256" in derived:
        # Cleared in its place, so the stamp of the suite it describes keeps
        # the stored matrix's key order.
        derived["source_sha256"] = None
    return derived


def _refresh_derive_cross_check(
    res: "PhaseConsolidation",
    *,
    per_test_tokens,
    kept,
    measured_km,
    log: Callable[[str], None],
) -> None:
    """Bind the derive cross-check to the final measured reduced source."""
    from testuary.languages import mutation

    derived_km = _derive_kill_matrix_for_tests(
        res.merged_km, per_test_tokens, kept,
    )
    res.derive_kept_tests = sorted(kept)
    res.derived_reduced_km = derived_km
    derived = mutation.matrix_score_percent(derived_km)
    measured = mutation.matrix_score_percent(measured_km)
    if derived is None or measured is None:
        res.derive_measure_delta = None
        res.derive_measure_differences = []
        return
    res.derive_measure_delta = derived - measured
    res.derive_measure_differences = _log_catalogue_diff(
        derived_km, measured_km, log,
    )
    if abs(res.derive_measure_delta) > 1e-9:
        log(
            f"[consolidate] derive-vs-measure on the reduced suite: "
            f"derived {derived:.4f} vs measured {measured:.4f} "
            f"(diff {res.derive_measure_delta:+.4f}). The measured matrix is kept"
        )
    else:
        log(
            f"[consolidate] derive==measure on the reduced suite "
            f"({measured:.4f}). The measured matrix is kept"
        )


def _collect_eviction(
    backend,
    path: Path,
    km: Optional[dict],
    km_path: Path,
    *,
    source_dir: str,
    target_file_path: str,
    timeout: int,
    log: Callable[[str], None],
) -> Tuple[Any, Dict[str, str]]:
    """The per-test eviction data of the suite at ``path``, with the kill
    matrix ``km`` written to ``km_path`` first when there is one. Returns
    the data (None when the eviction failed) and the failure the backend
    reported."""
    import json
    kill_matrix_path = None
    if km:
        km_path.write_text(json.dumps(km), encoding="utf-8")
        kill_matrix_path = str(km_path)
    evict_timeout = int(timeout * _EVICTION_TIMEOUT_MULTIPLIER)
    eviction_failure: Dict[str, str] = {}
    data = backend.collect_eviction_data(
        [str(path)], source_dir,
        measure_coverage_of=target_file_path, timeout=evict_timeout,
        kill_matrix_path=kill_matrix_path, log=log,
        # A phase end's evidence failure removes no test, so the names
        # the failure gives are not kept.
        failure_fn=lambda kind, detail, stopping_tests=(): eviction_failure.update(
            kind=kind, detail=detail,
        ),
    )
    return data, eviction_failure


def measure_candidate_standalone(
    backend,
    candidate_src: str,
    *,
    target_file_path: str,
    source_dir: str,
    timeout: int,
    mutation_runner: Optional[Callable[[Path], Optional[dict]]],
    stage_as_delivered: bool,
    log: Callable[[str], None],
    inherited: InheritedMatrix = InheritedMatrix(),
    expected_measurement_identity: Optional[Dict[str, Any]] = None,
    candidate_label: str = "standalone_1",
    candidate_name: str = "candidate-2",
    place: Optional[str] = None,
    removals: Optional[List[Dict[str, Any]]] = None,
    failed_out: Optional[List["_Failed"]] = None,
) -> "ScoredSuite":
    """Measure a candidate suite standalone, the phase's own new suite
    (candidate 2) at a phase end, the delivered suite in Finishing or a
    starter suite: its kill matrix and per-test coverage tokens, mirroring
    the merged-suite measurement.

    The scratch lives under its own ``scratch_dir`` so it never disturbs
    the merged measurement's directory. Returns a ``ScoredSuite`` whose
    coverage evidence has the same
    schema_version/target_file/line/branch/per_test_tokens shape as the union
    coverage sidecar. An inherited mutation matrix is reused only when its
    source hash, execution name and measurement identity match this staged
    candidate. The
    candidate is measured as it is, never stripped: one whose normal run
    fails is failing, one a tool could not measure is incomplete, each with
    its reason (:func:`_check_then_measure`).

    At a place of the removal fallback (``place``) a coverage run or
    mutation measurement that fails the candidate after its passing normal
    run removes the tests it names (:func:`_measured_removing`), the
    candidate without them is measured again without the inherited matrix,
    and ``removals`` receives the record's entries. ``failed_out`` receives
    such a failure, or a failing normal run's cause."""
    if place is not None:
        def measure(source: str, out: List[_Failed], first: bool) -> "ScoredSuite":
            return measure_candidate_standalone(
                backend, source, target_file_path=target_file_path, source_dir=source_dir,
                timeout=timeout, mutation_runner=mutation_runner,
                stage_as_delivered=stage_as_delivered, log=log,
                inherited=inherited if first else InheritedMatrix(),
                expected_measurement_identity=expected_measurement_identity,
                candidate_label=candidate_label, candidate_name=candidate_name,
                failed_out=out)

        _source, scored, entries = _measured_removing(
            measure, candidate_src, backend=backend, place=place, log=log)
        removals.extend(entries)
        return scored
    with project.scratch_dir(prefix="testuary_phaseend_cand2_") as td:
        cdir = Path(td) / "candidate2"
        cpath = _stage_in(backend, cdir, candidate_src, target_file_path,
                          stage_as_delivered=stage_as_delivered)
        run, passed = _check_then_measure(
            backend, [cpath], source_dir, timeout=timeout,
            measure_coverage_of=target_file_path, log=log,
        )
        measured_source = cpath.read_text(encoding="utf-8")
        if not getattr(run, "ok", False):
            if failed_out is not None:
                failed_out.append(_failed_run("coverage", run) if passed else _Failed(
                    "", cause=_removal_cause(backend, measured_source, run)))
            # A suite that passes its normal run but that the coverage tool
            # could not measure has no numbers: incomplete, not failing.
            reason = (f"the coverage tool could not measure it: {red_verdict(run)}"
                      if passed else red_verdict(run))
            log(f"[consolidate] {candidate_name} standalone suite is "
                + ("not measured" if passed else "not green")
                + f", so the {candidate_name} sidecars stay absent. " + reason)
            return ScoredSuite(
                label=candidate_label,
                source=measured_source,
                green=passed,
                status="incomplete" if passed else "failing",
                status_reason=reason,
                execution_name=cpath.name,
                target_revision=(expected_measurement_identity or {}).get(
                    "target_source_sha256"
                ),
                measurement_identity=dict(
                    expected_measurement_identity or {}
                ),
            )
        measured_hash = source_sha256(measured_source)
        reusable = bool(
            inherited.km is not None
            and inherited.bound_to(measured_hash, cpath.name,
                                   expected_measurement_identity)
        )
        if inherited.km is not None and not reusable:
            raise ValueError(
                f"{candidate_name} inherited mutation matrix is not bound to "
                "the exact source, execution name, and measurement identity"
            )
        if reusable:
            km = inherited.km
            log(
                f"[consolidate] {candidate_name} mutation matrix reused for the "
                "same source hash and delivered execution name"
            )
        else:
            km = mutation_runner(cdir) if mutation_runner is not None else None
            failure = (_mutation_failure(mutation_runner, cdir, km)
                       if mutation_runner is not None else None)
            if failure is not None and failed_out is not None:
                failed_out.append(failure)
        data, eviction_failure = _collect_eviction(
            backend, cpath, km, Path(td) / "candidate2_km.json",
            source_dir=source_dir, target_file_path=target_file_path,
            timeout=timeout, log=log,
        )
        reason = ""
        if eviction_failure:
            reason = ("the per-test coverage run could not measure it: "
                      + eviction_failure["detail"])
        elif data is not None:
            # Known limit: only the percentages are compared, so evidence that
            # covers other lines at equal percentages goes unrecorded, and its
            # covered sets stand.
            disagreement = _evidence_disagreement(
                getattr(run, "line_coverage", None), getattr(run, "branch_coverage", None),
                data, "in the coverage run")
            if disagreement is not None:
                reason = ("its per-test evidence disagrees with its coverage run: "
                          + disagreement)
                data = None
        if data is None:
            # Without per-test evidence (the language gives none, its per-test
            # coverage run, the coverage tool's, could not measure the suite
            # after its passing normal run, or that run's evidence disagrees
            # with the coverage run) the candidate is scored on its green
            # coverage run, without per-test tokens: the reductions are
            # skipped, the phase end can still select it, and the reason is
            # recorded.
            if reason:
                log(f"[consolidate] {candidate_name}: {reason}. It is scored on "
                    "its coverage run, without per-test evidence")
            return scored_from_measurement(
                label=candidate_label, source=measured_source, kill_matrix=km,
                coverage=coverage_evidence_of_run(run, target_file_path),
                execution_name=cpath.name, green=True, status="passing",
                test_count=source_test_count(backend, measured_source),
                status_reason=reason,
                target_revision=(expected_measurement_identity or {}).get(
                    "target_source_sha256"
                ),
                measurement_identity=expected_measurement_identity,
            )
        coverage = {
            "schema_version": 1,
            "target_file": target_file_path,
            "line": getattr(run, "line_coverage", None),
            "branch": getattr(run, "branch_coverage", None),
            "target_total_statements": data.target_total_lines,
            "covered_statement_lines": (
                sorted(data.suite_covered_lines)
                if data.suite_covered_lines is not None else None
            ),
            "target_total_branches": data.target_total_branches,
            "covered_branch_arcs": (
                sorted(data.suite_covered_branches)
                if data.suite_covered_branches is not None else None
            ),
            "per_test_tokens": {
                t: sorted(map(str, toks))
                for t, toks in data.per_test_tokens.items()
            },
        }
    return scored_from_measurement(
        label=candidate_label,
        source=measured_source,
        kill_matrix=km,
        coverage=coverage,
        execution_name=cpath.name,
        green=True,
        status="passing",
        test_count=len(data.per_test_tokens),
        target_revision=(expected_measurement_identity or {}).get(
            "target_source_sha256"
        ),
        measurement_identity=expected_measurement_identity,
    )


def coverage_evidence_of_run(run: Any, target_file_path: str) -> Dict[str, Any]:
    """A candidate's coverage evidence from one green coverage run: its
    covered lines and branches and the target's totals, without per-test
    tokens."""
    covered = getattr(run, "covered_lines", None)
    taken = getattr(run, "covered_branches", None)
    return {
        "schema_version": 1,
        "target_file": target_file_path,
        "line": getattr(run, "line_coverage", None),
        "branch": getattr(run, "branch_coverage", None),
        "target_total_statements": getattr(run, "total_statements", None),
        "covered_statement_lines": sorted(covered) if covered is not None else None,
        "target_total_branches": getattr(run, "total_branches", None),
        "covered_branch_arcs": sorted(taken) if taken is not None else None,
        "per_test_tokens": None,
    }


def _exclude_reduced(res: "PhaseConsolidation", issue: str) -> None:
    """Keep the reduced candidate out of the phase end's selection, with
    ``issue`` in the selection audit."""
    res.reduction_valid = False
    res.reduced_km = None
    res.line = None
    res.branch = None
    res.suite_covered_lines = None
    res.suite_covered_branches = None
    res.selection_audit["reduction_measurement_issue"] = issue


def _reject_reduced(res: "PhaseConsolidation", detail: str,
                    issue: str) -> "PhaseConsolidation":
    """Drop the reduced suite before it is measured: ``detail`` says why
    and ``issue`` goes in the selection audit."""
    res.detail = detail
    res.reduced_source = None
    res.kept_tests = None
    res.selected_tests = 0
    res.reduction_valid = False
    res.selection_audit["reduction_measurement_issue"] = issue
    return res


def _fall_back(res: "PhaseConsolidation", detail: str, reason: str,
               candidates: List[str],
               union_stats: Dict[str, object]) -> "PhaseConsolidation":
    """End the phase end without a reduction: ``detail`` says why, and
    ``reason``, the ``candidates`` left in its place and the merge's
    ``union_stats`` are recorded."""
    res.detail = detail
    res.fallback_reason = reason
    res.fallback_candidates = candidates
    res.fallback_union_stats = union_stats
    return res


def _check_reduced(context: PhaseEndContext, res: "PhaseConsolidation", *,
                   merged: str) -> None:
    """The normal run on the reduced suite's current bytes. A reduction that
    keeps every test is the merged suite, which passed. A reduced suite that
    does not pass leaves the selection. Nothing is removed from it."""
    if (res.reduced_source is None or not res.reduction_valid
            or res.reduced_source == merged):
        return
    with project.scratch_dir(prefix="testuary_phaseend_reduced_") as td:
        rdir = Path(td) / "reduced"
        rpath = _stage_in(context.backend, rdir, res.reduced_source,
                          context.target_file_path,
                          stage_as_delivered=context.stage_as_delivered)
        run = normal_run.run(context.backend, [rpath], context.source_dir,
                             timeout=context.timeout,
                             target=context.target_file_path)
    if not getattr(run, "ok", False):
        context.log(f"[consolidate] the reduced suite does not pass its normal run "
                    f"({red_verdict(run)}), so phase-end selection excludes it")
        _exclude_reduced(res, "reduced_failing")


def _refresh_reduced_coverage(context: PhaseEndContext, *, source: str,
                              result: PhaseConsolidation) -> bool:
    """Measure exact aggregate coverage for a changed reduced suite."""
    with project.scratch_dir(prefix="testuary_phaseend_cov_") as td:
        path = _stage_in(context.backend, td, source, context.target_file_path,
                         stage_as_delivered=context.stage_as_delivered)
        run = context.backend.run_test_files(
            [str(path)],
            context.source_dir,
            timeout=context.timeout,
            measure_coverage_of=context.target_file_path,
        )
    lines = getattr(run, "covered_lines", None)
    branches = getattr(run, "covered_branches", None)
    if not getattr(run, "ok", False) or lines is None or branches is None:
        # The reduced suite passed its normal run: a red run here is the
        # coverage tool's, named in the log.
        why = ""
        if not getattr(run, "ok", False):
            result.reduction_valid = False
            why = f" ({red_verdict(run)})"
        context.log(
            "[consolidate] the coverage tool could not measure the reduced "
            f"suite{why}, so the phase-end selector will exclude the incomplete "
            "reduction"
        )
        return False
    result.line = getattr(run, "line_coverage", None)
    result.branch = getattr(run, "branch_coverage", None)
    result.suite_covered_lines = sorted(int(line) for line in lines)
    result.suite_covered_branches = sorted(str(branch) for branch in branches)
    return True


def _partition(items: List[Dict[str, Any]], covered: str,
               key: Callable[[Dict[str, Any]], Any]) -> Tuple[List[Any], set]:
    """The items of status ``covered`` by ``key``, sorted with duplicates
    kept, and the missing items as a set. The covered pass runs first."""
    hits = sorted(key(item) for item in items if item["status"] == covered)
    return hits, {key(item) for item in items if item["status"] == "missing"}


def _duplicated(hits: List[Any], missing: set, items: List[Any]) -> bool:
    """Whether a matrix's covered items repeat, overlap the missing ones or
    leave items uncounted."""
    seen = set(hits)
    return bool(len(seen) != len(hits) or seen & missing
                or len(seen) + len(missing) != len(items))


def _check_summary(raw_matrix: Dict[str, Any], lines: List[int], missing_lines: set,
                   branches: List[str], missing_branches: set) -> None:
    """The matrix's summary against the counts of its own lines and branches."""
    expected = {
        "total_lines": len(lines) + len(missing_lines),
        "covered_lines": len(lines),
        "missing_lines": len(missing_lines),
        "covered_branches": len(branches),
        "missing_branches": len(missing_branches),
    }
    summary = raw_matrix.get("summary")
    if not isinstance(summary, dict) or any(
        summary.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("raw aggregate coverage summary disagrees")


def coverage_percents(covered_lines: Optional[int], total_lines: Optional[int],
                      covered_branches: Optional[int], total_branches: Optional[int],
                      ) -> Tuple[Optional[float], Optional[float]]:
    """A suite's line and branch coverage of its target in percent, covered
    over total, as LangAdapter reports them: None where a count is
    unknown, and for a target without lines or without branches, since the
    ratio then has nothing to count."""
    line = (100.0 * covered_lines / total_lines
            if covered_lines is not None and total_lines else None)
    branch = (100.0 * covered_branches / total_branches
              if covered_branches is not None and total_branches else None)
    return line, branch


def coverage_disagreement(line: Optional[float], branch: Optional[float],
                          other_line: Optional[float], other_branch: Optional[float],
                          ) -> bool:
    """Whether two measurements of one suite's coverage differ, in line or
    in branch coverage: by more than 1e-9 percent, or where one has a value
    and the other none. Two None values agree, as for a target without
    lines or branches."""
    return any((a is None) != (b is None)
               or (a is not None and abs(float(a) - float(b)) > 1e-9)
               for a, b in ((line, other_line), (branch, other_branch)))


def _check_scalars(run: Any, lines: List[int], missing_lines: set, branches: List[str],
                   missing_branches: set) -> None:
    """The run's measured line and branch coverage against the matrix's."""
    derived_line, derived_branch = coverage_percents(
        len(lines), len(lines) + len(missing_lines),
        len(branches), len(branches) + len(missing_branches))
    # A run without line coverage agrees only for a target without lines.
    if coverage_disagreement(getattr(run, "line_coverage", None),
                             getattr(run, "branch_coverage", None), derived_line, derived_branch):
        raise ValueError("raw and scalar aggregate coverage disagree")


def _matrix_coverage(raw_matrix: Dict[str, Any], run: Any, target_file: Optional[str]):
    """An exact matrix's covered and missing lines and branches, after its
    checks in this order, each with its own ValueError: the schema, the
    target, unknown statuses (lines and branches together), duplicates
    (both together), the summary and the run's scalar coverage."""
    matrix_lines, matrix_branches = raw_matrix["lines"], raw_matrix["branches"]
    if raw_matrix.get("schema_version") != 1:
        raise ValueError("raw aggregate coverage schema is unsupported")
    if target_file is not None and raw_matrix.get("target_file") != target_file:
        raise ValueError("raw aggregate coverage target disagrees")
    if any(not isinstance(item, dict) or item.get("status") not in {"covered", "missing"}
           for item in matrix_lines) or any(
            not isinstance(item, dict) or item.get("status") not in {"taken", "missing"}
            for item in matrix_branches):
        raise ValueError("raw aggregate coverage matrix is malformed")
    lines, missing_lines = _partition(matrix_lines, "covered", lambda item: int(item["line"]))
    branches, missing_branches = _partition(
        matrix_branches, "taken", lambda item: f"{int(item['from'])}->{int(item['to'])}")
    if (_duplicated(lines, missing_lines, matrix_lines)
            or _duplicated(branches, missing_branches, matrix_branches)):
        raise ValueError("raw aggregate coverage matrix has duplicates")
    _check_summary(raw_matrix, lines, missing_lines, branches, missing_branches)
    _check_scalars(run, lines, missing_lines, branches, missing_branches)
    return lines, missing_lines, branches, missing_branches


def aggregate_coverage_evidence(
    run: Any, raw_matrix: Dict[str, Any], *, target_file: Optional[str] = None,
) -> Dict[str, Any]:
    """One suite run's aggregate coverage from the exact matrix
    ``raw_matrix`` it wrote, checked against the run's own numbers. Per-test
    eviction data is not read."""
    lines, missing_lines, branches, missing_branches = _matrix_coverage(
        raw_matrix, run, target_file)
    return {
        "line": getattr(run, "line_coverage", None),
        "branch": getattr(run, "branch_coverage", None),
        "target_total_statements": len(set(lines) | missing_lines),
        "covered_statement_lines": lines,
        "target_total_branches": len(set(branches) | missing_branches),
        "covered_branch_arcs": branches,
    }


def _comparable(line: Optional[float], data: Any) -> bool:
    """Whether a run's line coverage ``line`` and per-test evidence ``data``
    can be compared: the run measured line coverage and the evidence holds
    its covered lines and the target's line total."""
    return (line is not None and data.suite_covered_lines is not None
            and data.target_total_lines is not None)


def _evidence_disagreement(line: Optional[float], branch: Optional[float],
                           data: Any, measured: str) -> Optional[str]:
    """How per-test evidence ``data`` disagrees with a run's line and branch
    coverage percentages, that run named by ``measured``, or None when they
    agree or are not comparable (:func:`_comparable`)."""
    # Unmeasured evidence or a run without line coverage only skips the check.
    if not _comparable(line, data):
        return None
    evidence_line, evidence_branch = _evidence_percents(data)
    if coverage_disagreement(line, branch, evidence_line, evidence_branch):
        return (f"line coverage {line} {measured}, {evidence_line} in the "
                f"evidence, and branch coverage {branch} {measured}, {evidence_branch} "
                "in the evidence")
    return None


def _evidence_percents(data: Any) -> Tuple[Optional[float], Optional[float]]:
    """The line and branch coverage of comparable per-test evidence ``data``
    (:func:`_comparable`), from its covered sets and the target's totals."""
    branches = data.suite_covered_branches
    return coverage_percents(
        len(data.suite_covered_lines), data.target_total_lines,
        len(branches) if branches is not None else None, data.target_total_branches)


def _only_in(values: Any, other: Any, kind: Callable[[Any], Any],
             ) -> Tuple[Optional[List[Any]], Optional[List[Any]]]:
    """The entries only ``values`` has and those only ``other`` has, each
    sorted as ``kind`` gives them. (None, None) when either is None."""
    if values is None or other is None:
        return None, None
    mine, theirs = set(map(kind, values)), set(map(kind, other))
    return sorted(mine - theirs), sorted(theirs - mine)


def coverage_sides(first: str, first_values: Tuple[Any, ...], second: str,
                   second_values: Tuple[Any, ...]) -> Dict[str, Any]:
    """Two measurements of one suite's coverage side by side, each given as
    its line and branch coverage, its covered lines and its taken arcs: for
    each name, its percentages and the covered lines and taken arcs only it
    has. A set that either side lacks is None on both."""
    (line, branch, lines, arcs), (other_line, other_branch, other_lines, other_arcs) = (
        first_values, second_values)
    lines_only = _only_in(lines, other_lines, int)
    arcs_only = _only_in(arcs, other_arcs, str)
    return {
        first: {"line": line, "branch": branch, "covered_lines_only": lines_only[0],
                "taken_arcs_only": arcs_only[0]},
        second: {"line": other_line, "branch": other_branch,
                 "covered_lines_only": lines_only[1], "taken_arcs_only": arcs_only[1]},
    }


def evidence_coverage_difference(res: "PhaseConsolidation", data: Any,
                                 ) -> Optional[Dict[str, Any]]:
    """How the merged suite's per-test evidence ``data`` differs from the
    coverage its run measured on ``res``, in percentages, covered lines or
    taken arcs (:func:`coverage_sides`, the sides ``measured`` and
    ``per_test_evidence``). None when they agree, or when the run measured
    no coverage or the evidence holds none."""
    if not _comparable(res.line, data):
        return None
    evidence_line, evidence_branch = _evidence_percents(data)
    sides = coverage_sides(
        "measured", (res.line, res.branch, res.suite_covered_lines, res.suite_covered_branches),
        "per_test_evidence", (evidence_line, evidence_branch, data.suite_covered_lines,
                              data.suite_covered_branches))
    differ = coverage_disagreement(res.line, res.branch, evidence_line, evidence_branch) or any(
        side[key] for side in sides.values() for key in ("covered_lines_only", "taken_arcs_only"))
    return sides if differ else None


def _adopt_evidence_coverage(res: "PhaseConsolidation", data: Any) -> None:
    """The merged suite's covered sets and target totals on ``res`` from its
    per-test evidence ``data``, which agrees with its measured coverage."""
    res.target_total_lines = data.target_total_lines
    res.target_total_branches = data.target_total_branches
    res.suite_covered_lines = (
        sorted(data.suite_covered_lines)
        if data.suite_covered_lines is not None else None
    )
    res.suite_covered_branches = (
        sorted(data.suite_covered_branches)
        if data.suite_covered_branches is not None else None
    )
    res.union_covered_lines = (
        list(res.suite_covered_lines)
        if res.suite_covered_lines is not None else None
    )
    res.union_covered_branches = (
        list(res.suite_covered_branches)
        if res.suite_covered_branches is not None else None
    )


def source_test_count(backend: Any, source: str) -> int:
    """Count source-level tests with the backend's structural parser."""
    value = backend.test_metrics(source).get("test_count")
    if isinstance(value, bool):
        raise RuntimeError("structural test count is not a positive integer")
    try:
        count = int(value)
    except (TypeError, ValueError) as exc:
        raise RuntimeError(
            "structural test count is not a positive integer"
        ) from exc
    if count <= 0 or value != count:
        raise RuntimeError("structural test count is not a positive integer")
    return count


def _finalize_candidates(
    backend, candidates: Sequence[str], *, source_dir: str,
    res: PhaseConsolidation, log: Callable[[str], None],
) -> Tuple[Tuple[str, ...], set[int]]:
    """The candidates as the phase end measures them: shadowed definitions
    removed (each removal in ``res.input_normalizations``), finalized,
    structurally sound and valid for the target, else ValueError
    (ValidationUnresolved when the validator does not judge one). Also
    returns the indices of the candidates this changed, whose inherited
    evidence is stale."""
    normalized_candidates: List[str] = []
    changed_inputs: set[int] = set()
    for i, source in enumerate(candidates):
        normalized = backend.normalize_shadowed_definitions(source)
        if not normalized.parse_ok:
            raise ValueError(
                f"phase-end candidate {i} does not parse under "
                f"{normalized.rule_version}: {normalized.parse_error}"
            )
        finalized = backend.finalize_test_source(normalized.source)
        structure = backend.check_structure(finalized)
        if not structure.ok:
            from testuary.agent._helpers import structure_message
            raise ValueError(
                f"phase-end candidate {i} failed the structural parser: "
                f"{structure_message(structure)}"
            )
        validation = validate_judged(backend, finalized, source_dir)
        if not validation.ok:
            raise ValueError(
                f"phase-end candidate {i} failed target validation: "
                f"{validation.status}: {validation.message}"
            )
        normalized_candidates.append(finalized)
        if normalized.changed or finalized != source:
            changed_inputs.add(i)
            log(
                f"[consolidate] candidate {i} finalized before phase-end "
                "measurement, so stale inherited evidence is ignored"
            )
        for binding in normalized.removed_bindings:
            res.input_normalizations.append({
                "candidate_index": i,
                "scope": list(binding.scope),
                "name": binding.name,
                "kind": binding.kind,
                "removed_span": [
                    binding.removed_start_line,
                    binding.removed_end_line,
                ],
                "retained_span": [
                    binding.retained_start_line,
                    binding.retained_end_line,
                ],
                "rule_version": normalized.rule_version,
                "before_sha256": normalized.before_sha256,
                "after_sha256": normalized.after_sha256,
            })
    return tuple(normalized_candidates), changed_inputs


def _record_aggregate_coverage(
    backend, res: PhaseConsolidation, run, aggregate_matrix_path: Path, *,
    merged: str, target_file_path: str,
) -> bool:
    """Record the merged suite's aggregate coverage from its passing run
    ``run`` and the matrix the run wrote at ``aggregate_matrix_path``.
    False when that matrix is not exact: the merged suite is then
    incomplete, for the reason in ``res.detail``."""
    import json

    raw_matrix = None
    try:
        raw_matrix = json.loads(
            aggregate_matrix_path.read_text(encoding="utf-8")
        )
    except (OSError, ValueError):
        pass
    if not (isinstance(raw_matrix, dict) and isinstance(raw_matrix.get("lines"), list)
            and isinstance(raw_matrix.get("branches"), list)):
        res.detail = (
            "the merged suite's aggregate coverage matrix could not be "
            "persisted"
        )
        res.merged_incomplete = res.detail
        return False
    aggregate = aggregate_coverage_evidence(
        run, raw_matrix, target_file=target_file_path,
    )
    res.line = aggregate["line"]
    res.branch = aggregate["branch"]
    res.union_line = res.line
    res.union_branch = res.branch
    # A target with no branch opportunities legitimately has no branch
    # percentage. Successful line evidence still proves that the aggregate
    # coverage run completed and must be persisted.
    res.aggregate_coverage_measured = res.union_line is not None
    res.pool_tests = source_test_count(backend, merged)
    res.target_total_lines = aggregate["target_total_statements"]
    res.target_total_branches = aggregate["target_total_branches"]
    res.suite_covered_lines = list(aggregate["covered_statement_lines"])
    res.union_covered_lines = list(res.suite_covered_lines)
    res.suite_covered_branches = list(aggregate["covered_branch_arcs"])
    res.union_covered_branches = list(res.suite_covered_branches)
    return True


def _record_per_test_evidence(
    res: PhaseConsolidation, data, eviction_failure: Dict[str, str],
    mutation_runner: Optional[Callable[[Path], Optional[dict]]],
) -> Optional[PhaseConsolidation]:
    """Record the merged suite's per-test evidence ``data`` in ``res``.
    Without evidence a cover can use, the phase end ends without a
    reduction and ``res`` is returned."""
    if data is None:
        if eviction_failure.get("kind") == "test_failure":
            # The per-test coverage run is the coverage tool's: a test that
            # fails there, after the merged suite's passing normal run, leaves
            # it without per-test evidence, as below, not failing.
            reason = eviction_failure.get(
                "detail", "per-test eviction run failed",
            )
            return _fall_back(
                res,
                "the merged suite's per-test coverage run could not measure it, so "
                f"the reduced candidate is unavailable: {reason}",
                "eviction_test_failure", ["merged_unreduced"],
                _union_bloat_stats(None, union_tests=res.pool_tests))
        # The per-test eviction (the instrumented coverage run) failed, so the
        # cover cannot run. Preserve the passing, measured merge as its own
        # phase-end candidate, but do not present it as a valid reduction. The
        # common selector will compare it with the measured input candidates.
        # Aggregate union coverage still persists: it belongs to the earlier
        # passing merged-suite run, not to the failed per-test instrument.
        return _fall_back(
            res, "the merged suite's per-test eviction failed, so "
            "the reduced candidate is unavailable",
            "eviction_data_unavailable", ["merged_unreduced"],
            _union_bloat_stats(None, union_tests=res.pool_tests))

    res.pool_tests = len(data.per_test_tokens)
    res.pool_runtime_ms = _measured_time_ms(data.per_test_time_ms, data.per_test_tokens)
    difference = evidence_coverage_difference(res, data)
    if difference is not None:
        # The coverage the merged suite's run measured stays its coverage.
        # The reduction works on the per-test evidence, and a reduced suite
        # it changes is measured again in its own run.
        res.selection_audit["merged_evidence_difference"] = difference
    else:
        _adopt_evidence_coverage(res, data)
    res.per_test_tokens = {
        t: sorted(map(str, toks)) for t, toks in data.per_test_tokens.items()
    }
    res.per_test_cost_ms = {
        str(t): float(cost) for t, cost in data.per_test_time_ms.items()
    }
    if mutation_runner is not None and not res.merged_km:
        # The merged suite's mutation was asked for but not measured (the
        # runner returned no kill matrix). A reduction could not be
        # re-measured for mutation either, so the selector, which needs every
        # axis, would never choose it, and a cover on coverage tokens alone
        # would read as a valid reduction. Record why it is absent instead.
        return _fall_back(res, "the merged suite's mutation was not measured, so the "
                          "reduced candidate is unavailable",
                          "merged_mutation_unmeasured", [],
                          _union_bloat_stats(data.per_test_tokens,
                                             renamed_names=_renamed_names(res)))
    merged_kill_count = len(_killed_by_identity(res.merged_km))
    if merged_kill_count and not data.includes_kills:
        # Mutation was measured but its killers could not be attributed to tests
        # (no K: tokens: a tool that records no killer, or an attribution gap).
        # A mutation-aware cover cannot be constructed from coverage tokens
        # alone. Preserve the merge as a measured candidate and leave
        # ``reduction_valid`` false. A run with no mutation
        # runner can still reduce on coverage tokens.
        return _fall_back(res, "the merged suite's kills could not be attributed to "
                          "its tests, so the reduced candidate is unavailable",
                          "kills_unattributed", ["merged_unreduced"],
                          _union_bloat_stats(data.per_test_tokens,
                                             renamed_names=_renamed_names(res)))
    token_universe = set().union(
        *(set(tokens) for tokens in data.per_test_tokens.values())
    ) if data.per_test_tokens else set()
    if not token_universe:
        return _fall_back(
            res, "the merged suite produced an empty per-test token universe, so "
            "the reduced candidate is unavailable",
            "empty_cover", ["merged_unreduced"],
            _union_bloat_stats(data.per_test_tokens,
                               renamed_names=_renamed_names(res)))
    return None


def _solve_reduction(
    res: PhaseConsolidation, data, *, reduction: ReductionSettings,
    solver_failure_dir: Optional[str], target_file_path: str, log: Callable[[str], None],
) -> Tuple[Any, Dict[str, float]]:
    """The cover of the per-test tokens in ``data`` the solver chose (the
    least time under exact, an approximation under greedy) under the
    suite reduction settings ``reduction``, and each test's cost, with the
    solver's metadata in ``res``. A solver
    that fails records why there and, with ``solver_failure_dir``, in its
    consolidation_failure.json, then raises."""
    import json
    from testuary.reduction import (
        ExactCoverError, first_test_setup, reduction_costs, solve_cover,
    )

    # A test without a measured time is not free: it costs the slowest
    # measured test's time, imputed and recorded apart. Without any
    # measured time every test costs the same, so the reduction
    # minimizes the number of tests. The measured times stay as measured.
    # The run's first test also paid the run's one-time setup, which every
    # reduced suite pays again in its own first test. It is charged to no
    # test (reduction.first_test_setup).
    first_test = data.first_test
    first_setup = first_test_setup(
        data.per_test_tokens, data.per_test_time_ms, first_test)
    measured_ms, imputed_ms = reduction_costs(
        data.per_test_tokens, data.per_test_time_ms, first_test=first_test)
    cost_ms = {**measured_ms, **imputed_ms}
    if first_setup is not None:
        log(f"[consolidate] the first test {first_setup['test']} measured "
            f"{first_setup['measured_ms']:.1f}ms with the run's setup and costs "
            f"{first_setup['cost_ms']:.1f}ms, the rest is charged to no test")
    untimed_ms = {test: cost for test, cost in imputed_ms.items()
                  if first_setup is None or test != first_setup["test"]}
    if untimed_ms:
        log(f"[consolidate] {len(untimed_ms)} of {len(cost_ms)} test(s) have "
            "no measured time. "
            + ("Each costs the slowest measured test's time, an imputed cost"
               if measured_ms else
               "Every test costs the same, so the reduction minimizes the "
               "number of tests"))
    try:
        cover = solve_cover(
            dict(data.per_test_tokens),
            measured_ms,
            imputed_cost_ms=imputed_ms,
            solver=reduction.solver,
            time_limit_s=reduction.solver_time_limit_seconds,
            minimum_test_runtime_ms=reduction.minimum_test_runtime_ms,
            random_seed=reduction.solver_random_seed,
            tie_break=reduction.tie_break,
        )
    except (ExactCoverError, ValueError) as exc:
        metadata = dict(getattr(exc, "metadata", {}) or {})
        metadata.setdefault("solver", str(reduction.solver))
        metadata.setdefault("failure", "invalid_solver_input")
        metadata["error_type"] = type(exc).__name__
        metadata["error"] = str(exc)
        res.solver_metadata = metadata
        if solver_failure_dir is not None:
            failure_dir = Path(solver_failure_dir)
            failure_dir.mkdir(parents=True, exist_ok=True)
            failure_record = {
                "schema_version": 1,
                **timing_fields(),
                "target_file": target_file_path,
                "target_revision": res.target_revision,
                "merged_source_sha256": res.merged_source_sha256,
                "structural_parser_identity": res.structural_parser_identity,
                "target_validator_identity": res.target_validator_identity,
                "reduction_solver": str(reduction.solver),
                "reduction_solver_metadata": metadata,
            }
            (failure_dir / "consolidation_failure.json").write_text(
                json.dumps(failure_record, indent=2, allow_nan=False),
                encoding="utf-8",
            )
        raise
    res.effective_cost_us = dict(cover.effective_cost_us)
    res.solver_metadata = dict(cover.solver_metadata or {})
    if first_setup is not None:
        res.solver_metadata["first_test_setup"] = first_setup
    return cover, cost_ms


def _build_reduced(
    context: PhaseEndContext, res: PhaseConsolidation, *, merged: str, remove: List[str],
) -> Optional[PhaseConsolidation]:
    """Put the reduced suite, ``merged`` without the tests in ``remove``,
    in ``res`` once it is valid and loads with all its tests. Otherwise
    ``res`` is returned with the reduced suite rejected."""
    backend, source_dir = context.backend, context.source_dir
    res.reduced_source = backend.prune_tests(merged, remove) if remove else merged
    reduced, reduced_validation, reduced_error = _finalize_and_validate_source(
        backend, res.reduced_source, source_dir
    )
    if reduced_error is not None:
        return _reject_reduced(
            res, f"the reduced suite failed validation: {reduced_error}",
            "reduced_source_invalid")
    # A reduction can leave a suite that does not load (a kept test calls a
    # removed helper). Its recovery is kept only when it removes no test, so
    # the reduced matrix derived from the kept tests stays exact.
    loaded, repairs = loadable.keep_if_loadable(
        backend, reduced, source_dir, timeout=context.timeout,
        target_file_path=context.target_file_path,
        validate=lambda text: validate_judged(backend, text, source_dir),
        log=context.log, label="consolidate",
        stage_as_delivered=context.stage_as_delivered)
    if repairs:
        res.selection_audit.setdefault("load_repairs", []).append(
            {"suite": "reduced", "removed": repairs})
        if loadable.fewer_tests(backend, reduced, loaded):
            return _reject_reduced(
                res, "the reduced suite loads only without some of its tests",
                "reduced_source_does_not_load")
        reduced, reduced_validation, reduced_error = _finalize_and_validate_source(
            backend, loaded, source_dir)
        if reduced_error is not None:
            return _reject_reduced(
                res,
                f"the recovered reduced suite failed validation: {reduced_error}",
                "reduced_source_invalid")
    res.reduced_source = reduced
    res.reduced_source_sha256 = reduced_validation.source_sha256
    res.target_validator_identity = reduced_validation.validator_identity
    res.reduction_valid = True
    return None


def _bind_inputs(context: PhaseEndContext, res: PhaseConsolidation,
                 candidates: Sequence[str], inherited1: InheritedMatrix,
                 inherited2: InheritedMatrix,
                 measurement_identity: Optional[Dict[str, Any]], mutating: bool,
                 ) -> Tuple[Tuple[str, ...], set, InheritedMatrix, InheritedMatrix]:
    """The phase end's inputs bound to the target: the finalized
    candidates and the indexes of those finalizing changed, the two
    inherited matrices (empty for a changed input), the target revision and
    the measurement identity on ``res``. ``mutating`` says whether a
    mutation runner measures. Raises OSError for an unreadable target and
    ValueError for an identity of another revision or a candidate 1 matrix
    not bound to its source."""
    backend = context.backend
    res.structural_parser_identity = backend.structural_parser_identity()
    candidates, changed_inputs = _finalize_candidates(
        backend, candidates, source_dir=context.source_dir, res=res,
        log=context.log)
    if 0 in changed_inputs:
        inherited1 = InheritedMatrix()
    if len(candidates) - 1 in changed_inputs:
        inherited2 = InheritedMatrix()
    res.target_revision = project.sha256_project_file(context.target_file_path)
    if res.target_revision is None:
        raise OSError(
            f"cannot read target source for phase consolidation: "
            f"{context.target_file_path}"
        )
    res.measurement_identity = dict(measurement_identity or {
        "target_source_sha256": res.target_revision,
        "backend": type(backend).__name__,
        "catalogue": "full" if mutating else "none",
    })
    if (
        res.measurement_identity.get("target_source_sha256")
        != res.target_revision
    ):
        raise ValueError(
            "phase measurement identity does not match the target revision"
        )
    if inherited1.km is not None:
        if not inherited1.bound_to(
            source_sha256(candidates[0]), res.measurement_identity.get("execution_name"),
            res.measurement_identity,
        ):
            raise ValueError(
                "candidate-1 inherited mutation matrix is not bound to "
                "the exact source and measurement identity"
            )
    return candidates, changed_inputs, inherited1, inherited2


def _measure_candidate2_side(context: PhaseEndContext, mutation_runner,
                             res: PhaseConsolidation, candidates: Sequence[str],
                             inherited2: InheritedMatrix, label: str) -> str:
    """Candidate 2, the phase's own new suite, measured standalone into
    ``res`` before the merge. A later audit can then compare the merge with
    the union of both candidates and detect a merge that lost a kill or a
    token either candidate held alone, and the merge's collision resolution
    sees both candidates' per-test evidence: the victim choice among owners
    that cannot be renamed reads measured tokens, and a choice made before
    any measurement would be positional by default. Without a mutation
    runner it measures the coverage and the per-test evidence, so the phase
    end chooses the suite on its own numbers. Except for a ValueError (such
    as an inherited matrix not bound to the candidate) or a mutation
    session whose cleanup failed, it never raises into the merged
    measurement, and it never changes that measurement. It is a place of
    the removal fallback, and returns the suite the merge takes:
    the candidate without the tests a removal took, else candidates[-1],
    which the archive keeps under the selection's ``label``."""
    entries: List[Dict[str, Any]] = []
    try:
        candidate2_record = measure_candidate_standalone(
            context.backend, candidates[-1],
            target_file_path=context.target_file_path,
            source_dir=context.source_dir, timeout=context.timeout,
            mutation_runner=mutation_runner,
            stage_as_delivered=context.stage_as_delivered,
            inherited=inherited2,
            expected_measurement_identity=res.measurement_identity,
            log=context.log, place="phase_end.candidate", removals=entries,
        )
        changed = _note_removals(
            context, res, "phase_end.candidate", entries, candidates[-1],
            candidate2_record.source, archive_name=(
                f"candidate_{label}_removed{Path(candidate2_record.execution_name).suffix}"))
        res.candidate2_candidate = candidate2_record
        res.candidate2_km = candidate2_record.kill_matrix
        res.candidate2_coverage = candidate2_record.coverage_evidence
        res.candidate2_source = candidate2_record.source
        res.candidate2_green = candidate2_record.green
        res.candidate2_status_reason = candidate2_record.status_reason
        res.candidate2_execution_name = candidate2_record.execution_name
        return candidate2_record.source if changed else candidates[-1]
    except ValueError:
        raise
    except Exception as exc:  # the side channel never fails the phase
        from testuary.languages.mutation import mutation_cleanup_pending

        # Unless its session's cleanup failed: that session admits no
        # further measurement.
        if mutation_cleanup_pending(exc):
            raise
        res.candidate2_km = None
        res.candidate2_coverage = None
        res.candidate2_source = None
        res.candidate2_green = False
        res.candidate2_status_reason = (
            f"standalone measurement raised {type(exc).__name__}: {exc}"
        )
        res.candidate2_execution_name = ""
        context.log(f"[consolidate] candidate-2 standalone measurement raised "
                    f"({exc}), so the candidate-2 sidecars stay absent")
        return candidates[-1]


def _without_removed_tests(backend, evidence: Optional[Sequence[Optional[Dict[str, Sequence[str]]]]],
                           before: str, after: str):
    """``evidence`` without the second candidate's entries of the tests a
    removal took from it, so that ``before`` became ``after``: the entries
    whose test ``before`` holds and ``after`` does not. Both candidates are
    then weighed by their pool evidence of the tests the merge receives."""
    if after == before or not evidence or len(evidence) != 2 or not evidence[-1]:
        return evidence
    from testuary.languages.generic import prune_report

    def held(source: str, test: str) -> bool:
        return prune_report(backend, source, [test]).source != source

    kept = {test: tokens for test, tokens in evidence[-1].items()
            if held(after, test) or not held(before, test)}
    return [*evidence[:-1], kept]


def _fold_validated(context: PhaseEndContext, res: PhaseConsolidation,
                    candidates: Sequence[str],
                    evidence: Optional[Sequence[Optional[Dict[str, Sequence[str]]]]],
                    changed: set) -> Optional[Tuple[str, Any]]:
    """The candidates merged into one validated suite, with the merge's
    record, the merged source and its validity record on ``res``. Each
    candidate's per-test evidence guides the merge, except a changed
    input's, and candidate 2's measured tokens stand in for missing ones.
    Returns the merged suite and the target's identity, or None with
    ``res.detail`` saying why no merge is available. A merged suite that
    violates the generated-test validity policy raises RuntimeError."""
    backend, source_dir = context.backend, context.source_dir
    merged_evidence: List[Optional[Dict[str, Sequence[str]]]] = list(
        evidence or ([None] * len(candidates)))
    while len(merged_evidence) < len(candidates):
        merged_evidence.append(None)
    for index in changed:
        if index < len(merged_evidence):
            merged_evidence[index] = None
    if res.candidate2_coverage and merged_evidence[-1] is None:
        tokens = res.candidate2_coverage.get("per_test_tokens")
        if tokens:
            merged_evidence[-1] = {k: list(v) for k, v in tokens.items()}

    merge_outcome: Dict[str, object] = {}
    merged = _merge_sources(
        backend,
        candidates,
        context.log,
        outcome=merge_outcome,
        evidence=merged_evidence,
        source_dir=source_dir,
    )
    # The record keeps what the merge did, also when the merge is
    # unavailable or its suite is refused.
    res.collision_resolution = merge_outcome
    if merged is None:
        res.detail = ("the complete merge is unavailable: "
                      + str(merge_outcome.get("unavailable") or "no source survived"))
        return None
    merged, merged_validation, merged_error = _finalize_and_validate_source(
        backend, merged, source_dir
    )
    if merged_error is not None:
        res.detail = f"the complete merged suite failed validation: {merged_error}"
        return None
    res.merged_source = merged
    res.target_validator_identity = merged_validation.validator_identity
    res.merged_source_sha256 = merged_validation.source_sha256

    target_identity = backend.generated_test_target_identity(
        context.target_file_path, source_dir,
    )
    merged_findings = backend.analyze_generated_test_validity(
        merged, target_identity,
    )
    if merged_findings:
        names = sorted({
            finding.enclosing_test or "<module>"
            for finding in merged_findings
        })
        raise RuntimeError(
            "the synthesized merged suite violates generated-test validity "
            f"policy {target_identity.policy_version}: {', '.join(names)}"
        )
    res.merged_validity_record = validity.direct_record(
        identity=target_identity,
        analyzed_source=merged,
        audit_flags=backend.screen_generated_test_source(
            merged, target_identity,
        ),
        stage="phase_end.merged",
    )
    return merged, target_identity


def _merged_matrix(context: PhaseEndContext, mutation_runner, res: PhaseConsolidation,
                   merged: str, mpath: Path, mdir: Path,
                   inherited1: InheritedMatrix) -> None:
    """The merged suite's mutation matrix on ``res``: candidate 1's inherited
    matrix when its bytes, execution name and measurement identity are the
    merged suite's, else the phase's own suite's when its bytes and
    execution name are, else the runner's measurement of the staged suite
    in ``mdir``."""
    merged_hash = source_sha256(merged)
    if (
        inherited1.km is not None
        and inherited1.bound_to(merged_hash, mpath.name, res.measurement_identity)
    ):
        res.merged_km = inherited1.km
        context.log(
            "[consolidate] merged mutation matrix reused from the "
            "byte-identical inherited candidate"
        )
    elif (
        res.candidate2_km is not None
        and res.candidate2_source is not None
        and res.candidate2_execution_name == mpath.name
        and source_sha256(res.candidate2_source) == merged_hash
    ):
        res.merged_km = res.candidate2_km
        context.log(
            "[consolidate] merged mutation matrix reused from the "
            "byte-identical agent candidate"
        )
    else:
        res.merged_km = mutation_runner(mdir)


def _measure_merged(context: PhaseEndContext, mutation_runner, res: PhaseConsolidation,
                    merged: str, inherited1: InheritedMatrix,
                    failed_out: Optional[List[_Failed]] = None):
    """The merged suite staged under its execution name and measured: its
    normal run, then the coverage tool's run with the aggregate record, its
    mutation matrix and the per-test evidence, all in one scratch directory
    that stays open while the runner and the eviction use it. Returns the
    eviction's data and failure, or None with ``res`` saying why the merged
    suite is no candidate. An identity naming another execution name raises
    ValueError. ``failed_out`` receives the failure of a coverage run or
    mutation measurement after the passing normal run, or a failing normal
    run's cause, for the removal fallback."""
    with project.scratch_dir(prefix="testuary_phaseend_") as td:
        mdir = Path(td) / "merged"
        mdir.mkdir(parents=True, exist_ok=True)
        aggregate_matrix_path = mdir / "aggregate_coverage_matrix.json"
        mpath = context.backend.scratch_test_path(
            str(mdir), merged, context.target_file_path,
            stage_as_delivered=context.stage_as_delivered)
        res.execution_name = mpath.name
        identity_execution_name = res.measurement_identity.get(
            "execution_name"
        )
        if (
            identity_execution_name is not None
            and identity_execution_name != mpath.name
        ):
            raise ValueError(
                "phase measurement identity does not match the delivered "
                "execution name"
            )
        res.measurement_identity["execution_name"] = mpath.name
        mpath.write_text(merged, encoding="utf-8")

        # The merged file must pass to be measured. A test that passes in its
        # own suite can fail in the merge. Such a merge is no candidate, and
        # nothing is removed from it.
        run, passed = _check_then_measure(
            context.backend, [mpath], context.source_dir, timeout=context.timeout,
            measure_coverage_of=context.target_file_path,
            coverage_matrix_path=str(aggregate_matrix_path),
            log=context.log,
        )
        if not getattr(run, "ok", False):
            if failed_out is not None:
                failed_out.append(_failed_run("coverage", run) if passed else _Failed(
                    "", cause=_removal_cause(context.backend, merged, run)))
            res.merged_green = passed
            res.detail = (
                "the coverage tool could not measure the merged suite, so it "
                f"is no candidate: {red_verdict(run)}" if passed else
                "the merged suite does not pass, so it is no candidate: "
                f"{red_verdict(run)}")
            res.merged_incomplete = res.detail if passed else ""
            return None
        if not _record_aggregate_coverage(
                context.backend, res, run, aggregate_matrix_path, merged=merged,
                target_file_path=context.target_file_path):
            if failed_out is not None:
                failed_out.append(_Failed("coverage", cause="the coverage matrix could not be read"))
            return None

        # Measure the merged file's mutation, then its per-test coverage with
        # those kills merged in: one file, one namespace, killer reconciliation
        # done inside the backend (no killer parse in the orchestration).
        if mutation_runner is not None:
            _merged_matrix(context, mutation_runner, res, merged, mpath, mdir,
                           inherited1)
            failure = _mutation_failure(mutation_runner, mdir, res.merged_km)
            if failure is not None and failed_out is not None:
                failed_out.append(failure)

        data, eviction_failure = _collect_eviction(
            context.backend, mpath, res.merged_km, Path(td) / "merged_km.json",
            source_dir=context.source_dir, target_file_path=context.target_file_path,
            timeout=context.timeout, log=context.log,
        )
    return data, eviction_failure


def _measured_merged(context: PhaseEndContext, mutation_runner, res: PhaseConsolidation,
                     merged: str, inherited1: InheritedMatrix) -> Tuple[str, Any]:
    """The merged suite measured (:func:`_measure_merged`) at its place of
    the removal fallback: a coverage run or mutation measurement
    that fails it after its passing normal run removes the tests it names,
    and the merged suite without them, measured again from the record as it
    was before the first measurement and without the inherited matrix, is
    the merged suite. ``validity_merged.json`` keeps the suite as built, and
    the removal is the first record of its chain. Returns the merged suite
    and its measurement."""
    start = dict(vars(res))

    def measure(source: str, out: List[_Failed], first: bool):
        if not first:
            vars(res).update(start)
        return _measure_merged(context, mutation_runner, res, source,
                               inherited1 if first else InheritedMatrix(), failed_out=out)

    changed, measured, entries = _measured_removing(
        measure, merged, backend=context.backend, place="phase_end.merged", log=context.log)
    if _note_removals(context, res, "phase_end.merged", entries, merged, changed):
        res.merged_source, res.merged_source_sha256 = changed, source_sha256(changed)
    return changed, measured


def _reduced_matrices(context: PhaseEndContext, mutation_runner,
                      res: PhaseConsolidation, merged: str, data: Any,
                      kept: set) -> Optional[str]:
    """The reduced suite's matrices on ``res``: derived from the merged
    matrix, checked by the reduced suite's normal run, then re-measured, or
    reused when the reduction keeps every test, and the derive cross-check
    under ``both``. Returns the reduced source as its normal run checked
    it."""
    # Derive the reduced matrix: restrict the merged matrix to the kept tests'
    # kills (K: tokens) and covered lines (L: tokens), no engine re-run, no
    # killer-string parse. Assertion, incompetent, and collection-error kills
    # derive exactly. A timeout kill carries no killer, so it is derived via the
    # line-coverage proxy (a kept test covering the mutated line), which is
    # necessary but not sufficient, so the derived score can exceed a fresh
    # re-measure when the hanging test is dropped. ``both`` re-measures and logs
    # the disagreeing mutants for exactly this reason. Skipped in measure mode
    # (the engine re-measures the reduced suite instead).
    if res.merged_km and data.includes_kills and res.mutation_method != "measure":
        res.derive_kept_tests = sorted(kept)
        res.reduced_km = _derive_kill_matrix_for_tests(
            res.merged_km, data.per_test_tokens, kept,
        )
        res.derived_reduced_km = res.reduced_km

    # The normal run decides before anything measures the reduced suite. It
    # can fail although the merged suite passed: a removed test may have set
    # up state a kept test relies on, and pruning changes module-level
    # bindings, so a test shadowed in the merged file by a later same-named
    # definition can become collectable once its twin is pruned.
    _check_reduced(context, res, merged=merged)
    checked_reduced = res.reduced_source

    # Re-measure a changed reduced suite under ``both`` and ``measure``. A cover
    # that keeps every test is byte-identical to the already measured merge and
    # reuses that exact matrix instead of running the same suite again. The reported
    # reduced-candidate score is the engine's own measurement rather than the
    # timeout-proxy-derived restriction. Under ``both``, retain the restriction
    # as a cross-check and log concrete catalogue differences. ``derive`` is the
    # explicit no-engine probe mode.
    if (res.reduction_valid and mutation_runner is not None and res.merged_km
            and res.mutation_method in ("measure", "both")
            and res.reduced_source != merged):
        with project.scratch_dir(prefix="testuary_phaseend_red_") as rtd:
            rdir = Path(rtd) / "reduced"
            _stage_in(
                context.backend, rdir, res.reduced_source, context.target_file_path,
                stage_as_delivered=context.stage_as_delivered)
            res.measured_reduced_km = mutation_runner(rdir)
        # Retain the fresh re-measure as the reduced candidate. A reduction
        # whose re-measure produced no matrix is excluded while the
        # independently measured merge and inputs remain available to the
        # common phase-end selector.
        if res.measured_reduced_km is not None:
            res.reduced_km = res.measured_reduced_km
        else:
            # Keep the passing reduction visible but incomplete. Mutation-mode
            # selection excludes it because its required matrix is absent.
            context.log("[consolidate] the reduced suite's re-measure produced no "
                        "matrix, so the phase-end selector will exclude that incomplete "
                        "candidate in mutation mode")
            res.reduced_km = None
            res.selection_audit[
                "reduction_measurement_issue"
            ] = "reduced_remeasure_no_matrix"
    elif res.reduced_source == merged:
        res.reduced_km = res.merged_km
        if res.mutation_method in ("measure", "both"):
            # The reduced artifact is byte-identical to the already measured
            # merge. Reuse that exact measurement, but bind it explicitly to
            # the measured-reduced role so persistence and derive auditing do
            # not lose a sidecar merely because the cover was a no-op.
            res.measured_reduced_km = res.merged_km
    if res.mutation_method == "both" and res.measured_reduced_km is not None:
        _refresh_derive_cross_check(
            res, per_test_tokens=data.per_test_tokens, kept=kept,
            measured_km=res.measured_reduced_km, log=context.log)
    return checked_reduced


def _refresh_and_repair(context: PhaseEndContext, mutation_runner,
                        res: PhaseConsolidation, merged: str, data: Any, kept: set,
                        cost_ms: Dict[str, float]) -> None:
    """A changed valid reduction's coverage measured again (excluded when
    the coverage tool cannot), then, with a mutation runner and complete
    evidence, the bounded repair of the evidence the reduction lost against
    the merged suite, and the counts of what it still lacks in the
    selection audit."""
    if (res.reduction_valid and res.reduced_source is not None
            and res.reduced_source != merged):
        if not _refresh_reduced_coverage(
            context,
            source=res.reduced_source,
            result=res,
        ):
            _exclude_reduced(res, "reduced_coverage_refresh_failed")

    if (
        res.reduction_valid
        and res.reduced_source is not None
        and res.reduced_km is not None
        and res.suite_covered_lines is not None
        and res.suite_covered_branches is not None
        and mutation_runner is not None
    ):
        _EvidenceRepair(
            context,
            mutation_runner,
            merged=merged,
            res=res,
            per_test_tokens=data.per_test_tokens,
            solver_cost_ms=cost_ms,
        ).run(kept)
        missing_lines, missing_branches, missing_kills = _missing_evidence(res)
        res.selection_audit.update({
            "reduction_missing_merged_lines": len(missing_lines),
            "reduction_missing_merged_branches": len(missing_branches),
            "reduction_missing_merged_kills": len(missing_kills),
        })
        if missing_lines or missing_branches or missing_kills:
            context.log(
                "[consolidate] bounded evidence repair left "
                f"{len(missing_lines)} line, {len(missing_branches)} branch, "
                f"and {len(missing_kills)} mutant identity difference(s). "
                "The common score selector remains authoritative"
            )


def _record_reduction_derivation(res: PhaseConsolidation, merged: str, data: Any,
                                 target_identity: Any) -> None:
    """The validity derivation of the reduced suite from the merged one on
    ``res``: the tests the reduced suite lacks and the tests the repair
    restored."""
    if res.reduced_source is None:
        return
    removed_from_merge = sorted(
        set(data.per_test_tokens) - set(res.kept_tests or ())
    )
    res.validity_derivations.append(validity.derivation_record(
        identity=target_identity,
        parent_source=merged,
        child_source=res.reduced_source,
        operation="set_cover_remove_restore",
        removed_tests=removed_from_merge,
        restored_tests=res.repaired_tests,
        stage="phase_end.reduced",
    ))


def consolidate_phase_end(
    context: PhaseEndContext,
    candidates: Sequence[str],
    mutation_runner: Optional[Callable[[Path], Optional[dict]]],
    options: PhaseEndOptions,
) -> PhaseConsolidation:
    """Merge the candidate suites into one merged suite, measure that merged
    suite (per-test coverage + full-attribution mutation, killers reconciled
    inside the backend), set-cover the measurement for a reduced suite of least
    time (approximated under greedy), and derive the reduced suite's matrix
    by restricting the merged matrix to the kept tests' kills. ``context``
    holds what every step runs with and ``options`` how the phase end
    measures and reduces.
    Language-independent: the orchestration reads only the shared
    ``L:``/``B:``/``K:`` token schema and opaque mutant ids, never an engine
    killer string.

    ``mutation_runner`` measures the merged file's mutation. None gives a
    coverage-only cover and no reduced matrix. With
    ``options.mutation_method`` derive, the reduced candidate uses the
    restricted score without an engine run. ``measure`` re-measures it.
    ``both`` (the packaged default) uses the fresh measurement and keeps the
    restriction only as a logged cross-check, including the derive-vs-measure
    delta and concrete catalogue difference. The reported reduced-candidate
    score is therefore the engine's measurement, not a timeout proxy. A red
    reduction (reduced_failing) or a passing reduction with an incomplete
    measurement remains visible as an ineligible candidate, its tests kept.
    The common phase-end selector chooses among every complete passing
    candidate.

    The context's ``stage_as_delivered`` sets which name every measurement in this call
    uses and is passed to each staging site unchanged. Product phase ends
    pass True, so every selectable suite is judged under the name it can be
    delivered with."""
    res = PhaseConsolidation(
        solver=str(options.reduction.solver),
        mutation_method=str(options.mutation_method),
        validity_source_dir=str(context.source_dir),
    )
    res.mutation_origin_stops = getattr(mutation_runner, "stops", res.mutation_origin_stops)
    candidates, changed_inputs, inherited1, inherited2 = _bind_inputs(
        context, res, candidates, options.inherited1, options.inherited2,
        options.measurement_identity, mutation_runner is not None,
    )
    # Advisory only, never a strip or a gate: name the tests whose verdict
    # hinges on the positional consumption of a stubbed return sequence, so
    # a later kill-reproducibility adjudication on such a test reads as a
    # flagged class, not a surprise.
    for candidate in candidates:
        for name in context.backend.order_dependent_mock_warnings(candidate):
            context.log(f"[consolidate] advisory: {name} asserts the positional "
                        f"consumption of a mock side_effect sequence. Under a "
                        f"scheduler that reorders independent calls this verdict "
                        f"is not a property of the code under test")
    # Every phase end with exactly two candidates measures the second alone,
    # the starter phase's end with exactly two passing starter suites too.
    # The starter phase passes its suites' pool evidence in, whatever their
    # number.
    evidence = options.candidate_evidence
    if len(candidates) == 2:
        second = _measure_candidate2_side(
            context, mutation_runner, res, candidates, inherited2, options.candidate2_label)
        evidence = _without_removed_tests(context.backend, evidence, candidates[-1], second)
        candidates = [*candidates[:-1], second]

    folded = _fold_validated(context, res, candidates, evidence, changed_inputs)
    if folded is None:
        return res
    merged, target_identity = folded

    merged, measured = _measured_merged(context, mutation_runner, res, merged, inherited1)
    if measured is None:
        return res
    data, eviction_failure = measured

    ended = _record_per_test_evidence(res, data, eviction_failure, mutation_runner)
    if ended is not None:
        return ended
    cover, cost_ms = _solve_reduction(
        res, data, reduction=options.reduction,
        solver_failure_dir=options.solver_failure_dir,
        target_file_path=context.target_file_path, log=context.log)
    kept = set(cover.keep)
    res.selected_tests = len(kept)
    res.kept_tests = sorted(kept)
    # The reduced suite's time is its kept tests' measured per-test sum, as
    # the merged suite's is. The solver's cost sum, with its imputed and
    # minimum costs, stays in its own record (solver_metadata). Selection
    # benchmarks tied non-reduced candidates separately.
    res.runtime_ms = _measured_time_ms(data.per_test_time_ms, kept)
    ended = _build_reduced(
        context, res, merged=merged,
        remove=[t for t in data.per_test_tokens if t not in kept])
    if ended is not None:
        return ended

    checked_reduced = _reduced_matrices(context, mutation_runner, res, merged, data, kept)

    _refresh_and_repair(context, mutation_runner, res, merged, data, kept, cost_ms)

    if res.reduced_source != checked_reduced:
        # The evidence repair put tests back: the final bytes get their own
        # normal run.
        _check_reduced(context, res, merged=merged)

    _record_reduction_derivation(res, merged, data, target_identity)

    return res


def aggregate_coverage_matrix(
    *,
    target_file: str,
    source_sha256: Optional[str],
    structural_parser_identity: Optional[str],
    target_validator_identity: Optional[str],
    line: Optional[float],
    branch: Optional[float],
    target_total_statements: Optional[int],
    covered_statement_lines: Optional[List[int]],
    target_total_branches: Optional[int],
    covered_branch_arcs: Optional[List[str]],
    per_test_tokens: Optional[dict],
    per_test_runtime_ms: Optional[dict],
    effective_cost_us: Optional[dict],
) -> Dict[str, Any]:
    """Build the canonical aggregate-coverage sidecar payload."""
    return {
        "schema_version": 1,
        "target_file": target_file,
        "source_sha256": source_sha256,
        "structural_parser_identity": structural_parser_identity,
        "target_validator_identity": target_validator_identity,
        "line": line,
        "branch": branch,
        "target_total_statements": target_total_statements,
        "covered_statement_lines": covered_statement_lines,
        "target_total_branches": target_total_branches,
        "covered_branch_arcs": covered_branch_arcs,
        "per_test_tokens": per_test_tokens,
        "per_test_runtime_ms": per_test_runtime_ms,
        "effective_cost_us": effective_cost_us,
    }


def _restrict(mapping: Optional[dict], ids) -> Optional[dict]:
    """``mapping``'s entries for ``ids``, None when either is None."""
    if mapping is None or ids is None:
        return None
    return {t: mapping[t] for t in ids if t in mapping}


def persist_coverage_sidecars(
    phase: PhaseConsolidation,
    matrices_dir,
    target_file: str,
) -> List[str]:
    """Write the phase end's per-test coverage sidecars beside the kill
    matrices in the phase's archive directory: ``merged_coverage_matrix.json``
    (every merged test's L:/B:/K: tokens) and ``reduced_coverage_matrix.json``
    (the kept tests in the same layout, then their ids). Aggregate union coverage comes from
    the original merged-suite run. Aggregate reduced coverage comes from a
    fresh run whenever pruning changed the source, so collection/import-time
    execution is not inferred from per-test tokens. The directory names the
    phase, so the filenames do not.
    A write failure raises: measurement evidence that cannot be persisted
    fails the boundary loudly instead of surfacing in a later audit. Returns
    the names written, in order."""
    import json
    if matrices_dir is None or (
        not phase.aggregate_coverage_measured
        and phase.per_test_tokens is None
    ):
        return []
    mdir = Path(matrices_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    (mdir / "merged_coverage_matrix.json").write_text(
        json.dumps(aggregate_coverage_matrix(
            target_file=target_file,
            source_sha256=phase.merged_source_sha256 or None,
            structural_parser_identity=phase.structural_parser_identity,
            target_validator_identity=phase.target_validator_identity,
            line=phase.union_line,
            branch=phase.union_branch,
            target_total_statements=phase.target_total_lines,
            covered_statement_lines=phase.union_covered_lines,
            target_total_branches=phase.target_total_branches,
            covered_branch_arcs=phase.union_covered_branches,
            per_test_tokens=phase.per_test_tokens,
            per_test_runtime_ms=phase.per_test_cost_ms,
            effective_cost_us=phase.effective_cost_us,
        ), indent=2), encoding="utf-8")
    written = ["merged_coverage_matrix.json"]
    if phase.kept_tests is not None and phase.per_test_tokens is not None:
        kept = phase.kept_tests
        reduced = aggregate_coverage_matrix(
            target_file=target_file,
            source_sha256=phase.reduced_source_sha256 or None,
            structural_parser_identity=phase.structural_parser_identity,
            target_validator_identity=phase.target_validator_identity,
            line=phase.line,
            branch=phase.branch,
            target_total_statements=phase.target_total_lines,
            covered_statement_lines=phase.suite_covered_lines,
            target_total_branches=phase.target_total_branches,
            covered_branch_arcs=phase.suite_covered_branches,
            per_test_tokens=_restrict(phase.per_test_tokens, kept),
            per_test_runtime_ms=_restrict(phase.per_test_cost_ms, kept),
            effective_cost_us=_restrict(phase.effective_cost_us, kept),
        )
        reduced["kept_tests"] = kept
        reduced["derive_kept_tests"] = phase.derive_kept_tests
        (mdir / "reduced_coverage_matrix.json").write_text(
            json.dumps(reduced, indent=2), encoding="utf-8")
        written.append("reduced_coverage_matrix.json")
    return written


@dataclass
class InheritedPhaseEnd:
    """What a phase end took over when its suite is the inherited suite:
    the inherited evidence rebound to the destination, and the delivered
    suite's test count, which is also its merged count."""

    kill_matrix: Optional[dict]
    coverage: Optional[dict]
    delivered_tests: Optional[int]


@dataclass
class InheritedEvidence:
    """The inherited phase's evidence that may stand for the delivered bytes.

    Built by ``inherited_phase_end_eligibility``, which admits only payloads
    that are bound to exactly those bytes. ``reuse_inherited_phase_end``
    copies the files it names and writes the validity records again from
    the delivered bytes."""

    record: dict
    coverage: dict
    coverage_name: str
    kill_matrix: Optional[dict]
    kill_matrix_name: Optional[str]


#: The destination names a normal phase end writes, by the role of the
#: inherited payload that supplies them. On an unchanged suite the union, the
#: second candidate and the reduction are the same bytes, and no derivation or
#: re-measurement happened, so every kill-matrix name carries the inherited
#: delivered matrix and every coverage name the inherited delivered coverage. The
#: record maps each destination file to the inherited file it came from.
_INHERITED_COPIES = {
    "kill_matrix": (
        "selected_kill_matrix.json",
        "reduced_kill_matrix.json", "merged_kill_matrix.json",
        "candidate2_kill_matrix.json", "derived_reduced_kill_matrix.json",
        "measured_reduced_kill_matrix.json",
    ),
    "coverage": (
        "selected_coverage_matrix.json",
        "reduced_coverage_matrix.json", "merged_coverage_matrix.json",
        "candidate2_coverage_matrix.json",
    ),
}


def _bound_coverage_payload(path: Path, digest: str) -> Optional[dict]:
    """The per-test coverage payload at ``path`` when it is bound to the
    source whose hash is ``digest``, else None.

    The reuse trusts the file's own binding rather than its name, and
    requires the per-test tokens its consumers read. Anything else is not
    evidence of the delivered bytes and takes the normal path."""
    import json
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if payload.get("source_sha256") != digest:
        return None
    if payload.get("per_test_tokens") is None:
        return None
    return payload


def inherited_phase_end_eligibility(
    *,
    source_phase_dir,
    delivered_source: str,
    execution_name: str,
    measurement_identity: Optional[dict],
    require_mutation: bool,
    target_file: str,
    source_dir: str,
) -> Tuple[Optional["InheritedEvidence"], str]:
    """Whether the inherited phase's evidence may stand for ``delivered_source``.

    Returns ``(evidence, "")`` when every condition holds, otherwise
    ``(None, reason)``. The conditions: the inherited phase delivered exactly
    these bytes under this execution name, and its measurement identity is
    the destination's. Its validity record
    binds the bytes under the
    current policy, and every
    payload the destination would copy is itself bound to those bytes and
    complete. Any shortfall is a reason to take the normal path, never an
    error."""
    import json
    sdir = Path(source_phase_dir)
    try:
        record = json.loads(
            (sdir / "consolidation_record.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None, "inherited consolidation record missing or unreadable"
    if not isinstance(record, dict):
        return None, "inherited consolidation record is not an object"
    digest = source_sha256(delivered_source)
    if record.get("selected_source_sha256") != digest:
        return None, "inherited suite bytes differ"
    if record.get("execution_name") != execution_name:
        return None, "inherited execution name differs"
    if record.get("measurement_identity") != measurement_identity:
        return None, "inherited measurement identity differs"
    if not (sdir / "validity_delivered.json").is_file():
        return None, (
            "inherited evidence incomplete: missing "
            "['validity_delivered.json']")
    backend = languages.get_backend(languages.detect_language(target_file))
    if backend is None:
        return None, "no backend can verify the inherited validity record"
    try:
        delivered = validity.load_record(sdir / "validity_delivered.json")
        identity = backend.generated_test_target_identity(
            target_file, source_dir)
    except Exception as exc:  # a reason, never a failure
        return None, f"inherited validity record unreadable: {exc}"
    if not validity.record_binds_source(delivered, delivered_source, identity):
        return None, "inherited validity record does not bind the suite bytes"

    # The canonical coverage evidence describes the suite the inherited phase
    # delivered, whatever candidate won, and carries that suite's hash.
    coverage_name = "selected_coverage_matrix.json"
    coverage = _bound_coverage_payload(sdir / coverage_name, digest)
    if coverage is None:
        return None, (
            "no inherited per-test coverage evidence is bound to the suite "
            "bytes"
        )

    kill_matrix = None
    kill_matrix_name = None
    if require_mutation:
        if not record.get("shipped_has_matrix", False):
            return None, "inherited suite delivered without a mutation matrix"
        # The canonical matrix states the source it describes, which must be
        # these bytes.
        kill_matrix_name = "selected_kill_matrix.json"
        try:
            kill_matrix = json.loads(
                (sdir / kill_matrix_name).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return None, "inherited mutation matrix missing or unreadable"
        if not isinstance(kill_matrix, dict):
            return None, "inherited mutation matrix is not an object"
        # A matrix of an earlier schema reads in this version's words.
        from langadapter.mutation import upgrade_kill_matrix
        try:
            kill_matrix = upgrade_kill_matrix(kill_matrix)
        except ValueError:
            return None, ("inherited mutation matrix has a schema version this "
                          "version cannot read")
        if kill_matrix.get("source_sha256") != digest:
            return None, (
                "inherited mutation matrix is not bound to the suite bytes"
            )
    return InheritedEvidence(
        record=record, coverage=coverage, coverage_name=coverage_name,
        kill_matrix=kill_matrix, kill_matrix_name=kill_matrix_name,
    ), ""


def check_restored_suite(
    backend,
    source: str,
    *,
    target_file_path: str,
    source_dir: str,
    timeout: int,
    log: Callable[[str], None],
) -> Tuple[bool, str]:
    """The normal run of a suite a run restores as delivered without
    running the phase that delivered it (``--reuse coverage`` with
    ``--reuse mutation``), under its delivered name. Returns ``(accepted,
    reason)``. Nothing is pruned: the restored measurements describe these
    bytes, so a suite that does not pass refuses the reuse."""
    with project.scratch_dir(prefix="testuary_restored_") as td:
        path = backend.scratch_test_path(
            td, source, target_file_path, stage_as_delivered=True)
        path.write_text(source, encoding="utf-8")
        run = normal_run.run(backend, [str(path)], source_dir,
                             timeout=timeout, target=target_file_path)
    if not getattr(run, "ok", False):
        return False, (
            f"the restored suite does not pass its normal run ({red_verdict(run)}). "
            "Run the mutation phase again instead of reusing it")
    log("[consolidate] the restored suite passes its normal run")
    return True, ""


def _phase_end_dir(matrices_dir) -> Path:
    """The phase archive directory ``matrices_dir``, created when absent and
    without the files a phase end owns."""
    mdir = Path(matrices_dir)
    mdir.mkdir(parents=True, exist_ok=True)
    for stale in [*(mdir / name for name in _PHASE_END_SIDECARS),
                  *mdir.glob("candidate_*_removed*")]:
        if stale.is_file() or stale.is_symlink():
            stale.unlink()
    return mdir


def reuse_inherited_phase_end(
    *,
    source_phase_dir,
    source_phase: str,
    destination_dir,
    evidence: "InheritedEvidence",
    target_file: str,
    source_dir: str,
    delivered_source: str,
    execution_name: str,
    measurement_identity: Optional[dict],
    generation_scope: dict,
) -> InheritedPhaseEnd:
    """Materialize a phase end from the inherited phase's evidence.

    Called only with the ``InheritedEvidence`` that
    ``inherited_phase_end_eligibility`` admitted. Copies those payloads byte
    for byte under every name a normal phase end writes, writes the
    destination's validity records from the delivered bytes, and writes a
    consolidation record whose winner is ``inherited``. Nothing is executed,
    measured, reduced or selected. ``generation_scope`` is the destination
    phase's construction scope. Scientific
    files come first and the record last, so an interruption leaves an
    archive without a record, which a resume treats as incomplete. The result
    references no file of the source directory."""
    import json
    import shutil
    source_record = evidence.record
    from testuary import generation_evidence
    ancestry = generation_evidence.construction(
        delivered_source, **generation_scope, operation="inherited",
        parents=[generation_evidence.predecessor(source_record, phase=source_phase, source=delivered_source)],
    )
    sdir = Path(source_phase_dir)
    mdir = _phase_end_dir(destination_dir)

    expected: List[str] = []
    inherited_artifacts: Dict[str, str] = {}
    for role, source_name in (
        ("kill_matrix", evidence.kill_matrix_name),
        ("coverage", evidence.coverage_name),
    ):
        if source_name is None:
            continue
        src = sdir / source_name
        for destination_name in _INHERITED_COPIES[role]:
            shutil.copyfile(src, mdir / destination_name)
            expected.append(destination_name)
            inherited_artifacts[destination_name] = source_name

    backend = languages.get_backend(languages.detect_language(target_file))
    if backend is None:
        raise RuntimeError(
            f"no backend can validate the inherited suite for "
            f"{target_file}")
    identity = backend.generated_test_target_identity(target_file, source_dir)
    audit_flags = backend.screen_generated_test_source(
        delivered_source, identity)
    merged_record = validity.direct_record(
        identity=identity, analyzed_source=delivered_source,
        audit_flags=audit_flags, stage="phase_end.merged",
    )
    merged_record["inherited_from"] = source_phase
    validity.write_json_atomic(mdir / "validity_merged.json", merged_record)
    expected.append("validity_merged.json")
    validity.write_json_atomic(
        mdir / "validity_derivations.json",
        {
            "schema_version": validity.RECORD_SCHEMA_VERSION,
            "records": [],
            "inherited_from": source_phase,
            **timing_fields(),
        },
    )
    expected.append("validity_derivations.json")
    delivered_record = validity.direct_record(
        identity=identity, analyzed_source=delivered_source,
        audit_flags=audit_flags, stage="phase_end.delivered_assertion",
    )
    delivered_record["inherited_from"] = source_phase
    validity.write_json_atomic(mdir / "validity_delivered.json", delivered_record)
    expected.append("validity_delivered.json")

    kill_matrix = evidence.kill_matrix
    coverage = evidence.coverage
    delivered_tests = source_record.get("delivered_tests")
    digest = source_sha256(delivered_source)
    # A measured phase end's record layout, from the inherited values. The
    # inherited matrix is the delivered suite's mutation measurement, so
    # mutation_measured reads it as a measured phase end reads its own.
    record = _consolidation_record(
        target_file,
        PhaseConsolidation(
            target_revision=source_record.get("target_revision"),
            measurement_identity=measurement_identity,
            structural_parser_identity=source_record.get("structural_parser_identity"),
            target_validator_identity=source_record.get("target_validator_identity"),
            merged_source_sha256=digest, reduced_source_sha256=digest,
            pool_tests=delivered_tests,
        ),
        PhaseEndDelivery(
            winner_label="inherited", delivered_km=kill_matrix,
            winner_source_label=source_phase, delivered_tests=delivered_tests,
            delivered_source=delivered_source, execution_name=execution_name,
        ),
        ancestry,
        f"inherited from the {source_phase} phase: the suite is "
        "byte-identical to the inherited suite under the same "
        "measurement identity with complete bound evidence. Nothing "
        "was executed, measured, reduced or selected",
        expected, delivered_record,
    )
    # Nothing was reduced, selected or measured here, and "measured" says so.
    record.update({
        "reduction_valid": None,
        "selection_audit": None,
        "derive_measure_differences": None,
        "reduction_solver": None,
        "consolidation_mutation_method": None,
        "inherited_from": source_phase,
        "inherited_artifacts": inherited_artifacts,
        "measured": False,
        "measurement_calls": 0,
    })
    (mdir / "consolidation_record.json").write_text(
        json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")
    expected.append("consolidation_record.json")
    missing = [name for name in expected if not (mdir / name).exists()]
    if missing:
        raise RuntimeError(
            f"inherited phase-end persistence incomplete under {mdir}: "
            f"missing {missing}")
    return InheritedPhaseEnd(
        kill_matrix=kill_matrix, coverage=coverage,
        delivered_tests=delivered_tests,
    )


def _bound_to_source(
    evidence: dict, source_sha256: Optional[str],
    execution_name: Optional[str],
) -> dict:
    """``evidence`` stamped with the source it describes, as a new dict.

    A kill matrix carries no binding of its own, so without a stamp a reader
    cannot tell from the file which suite it belongs to. The stamp adds the
    source's hash and the delivered suite's execution name, each when it is
    known. A payload that is already stamped with other bytes is a caller
    error, never overwritten. An execution name the payload carries is
    replaced."""
    stamped = dict(evidence)
    if source_sha256 is not None:
        existing = stamped.get("source_sha256")
        if existing not in (None, source_sha256):
            raise RuntimeError(
                "evidence is already bound to different source bytes"
            )
        stamped["source_sha256"] = source_sha256
    if execution_name is not None:
        stamped["execution_name"] = execution_name
    return stamped


#: The candidates the merged run measured: their tests carry its per-test
#: times by their ids in the merged suite.
_MERGED_RUN_LABELS = frozenset({"merged", "reduced"})


def _known_or(value: Any, fallback: Any) -> Any:
    """``value``, or ``fallback`` when ``value`` is None."""
    return fallback if value is None else value


def selected_coverage_payload(
    phase: "PhaseConsolidation", candidate: "ScoredSuite", target_file: str,
) -> Optional[dict]:
    """The coverage evidence of the suite a phase END DELIVERED, built from
    the winning candidate itself.

    The per-candidate sidecars describe the candidate they are named for, so
    none of them is the delivered suite's evidence except by coincidence of
    which candidate won. This builds the delivered suite's own payload, which
    ``persist_phase_end`` stamps with the delivered bytes' hash. Returns None
    when the candidate carries no coverage at all, so a phase that measured no
    coverage writes no canonical sidecar rather than an empty one. The
    per-test times come from the merged run, so only a merged or reduced
    winner carries them."""
    evidence = (
        candidate.coverage_evidence
        if isinstance(candidate.coverage_evidence, dict) else {}
    )
    tokens: Optional[Dict[str, List[str]]] = None
    raw_tokens = _known_or(candidate.per_test_tokens, evidence.get("per_test_tokens"))
    if raw_tokens is not None:
        tokens = {
            str(test_id): sorted(str(token) for token in token_set)
            for test_id, token_set in raw_tokens.items()
        }
    if (
        candidate.line is None and candidate.branch is None and tokens is None
    ):
        return None

    # Another winner's test ids can name renamed collisions in the merge.
    times, costs = ((phase.per_test_cost_ms, phase.effective_cost_us)
                    if candidate.label in _MERGED_RUN_LABELS else (None, None))

    covered_lines = (
        sorted(candidate.covered_lines)
        if candidate.covered_lines is not None
        else evidence.get("covered_statement_lines")
    )
    covered_branches = (
        sorted(candidate.covered_branches)
        if candidate.covered_branches is not None
        else evidence.get("covered_branch_arcs")
    )
    # The winner's own totals, the phase end's where its evidence has none.
    payload = aggregate_coverage_matrix(
        target_file=target_file,
        source_sha256=None,
        structural_parser_identity=phase.structural_parser_identity,
        target_validator_identity=phase.target_validator_identity,
        line=candidate.line,
        branch=candidate.branch,
        target_total_statements=_known_or(
            evidence.get("target_total_statements"), phase.target_total_lines),
        covered_statement_lines=covered_lines,
        target_total_branches=_known_or(
            evidence.get("target_total_branches"), phase.target_total_branches),
        covered_branch_arcs=covered_branches,
        per_test_tokens=tokens,
        per_test_runtime_ms=_restrict(times, tokens),
        effective_cost_us=_restrict(costs, tokens),
    )
    payload["kept_tests"] = sorted(tokens) if tokens is not None else None
    payload["candidate_label"] = candidate.label
    return payload


def _delivered_validity_record(phase: PhaseConsolidation, target_file: str,
                               source: str) -> Dict[str, Any]:
    """The delivered validity record of the delivered source ``source``,
    after its checks: a backend for the target's language exists, the source
    parses, it has no shadowed definitions and the generated-test validity
    policy finds nothing. A failed check raises RuntimeError."""
    backend = languages.get_backend(
        languages.detect_language(target_file)
    )
    if backend is None:
        raise RuntimeError(
            f"no backend can validate the delivered suite for "
            f"{target_file}"
        )
    normalized = backend.normalize_shadowed_definitions(source)
    if not normalized.parse_ok:
        raise RuntimeError(
            "the delivered suite does not parse under "
            f"{normalized.rule_version}: {normalized.parse_error}"
        )
    if normalized.changed:
        names = ", ".join(dict.fromkeys(
            ".".join((*binding.scope, binding.name))
            for binding in normalized.removed_bindings
        ))
        raise RuntimeError(
            "the delivered suite contains shadowed definitions "
            f"({names}), so persistence is refused"
        )
    recorded_identity = (
        (phase.merged_validity_record or {}).get("target_identity") or {}
    )
    identity_root = str(
        recorded_identity.get("project_root")
        or phase.validity_source_dir
        or Path(target_file).parent
    )
    target_identity = backend.generated_test_target_identity(
        target_file, identity_root,
    )
    delivered_findings = backend.analyze_generated_test_validity(
        source, target_identity,
    )
    if delivered_findings:
        names = sorted({
            finding.enclosing_test or "<module>"
            for finding in delivered_findings
        })
        raise RuntimeError(
            "the delivered suite violates generated-test validity "
            f"policy {target_identity.policy_version}: {', '.join(names)}"
        )
    return validity.direct_record(
        identity=target_identity,
        analyzed_source=source,
        audit_flags=backend.screen_generated_test_source(
            source, target_identity,
        ),
        stage="phase_end.delivered_assertion",
    )


@dataclass(frozen=True)
class PhaseEndDelivery:
    """What a phase end delivers: the winner's label (with the input suite's
    label when a single suite won), and the delivered suite's kill matrix,
    test count, source, coverage, candidate and execution name."""
    winner_label: str
    delivered_km: Optional[dict]
    winner_source_label: Optional[str] = None
    delivered_tests: Optional[int] = None
    delivered_source: Optional[str] = None
    delivered_coverage: Optional[dict] = None
    delivered_candidate: Optional["ScoredSuite"] = None
    execution_name: Optional[str] = None

    @property
    def delivered_hash(self) -> Optional[str]:
        """The delivered source's SHA-256, None without a delivered source."""
        return (source_sha256(self.delivered_source)
                if self.delivered_source is not None else None)


def _delivered_coverage(phase: PhaseConsolidation, target_file: str,
                        delivery: PhaseEndDelivery) -> Optional[dict]:
    """The delivered suite's coverage: the delivery's own, else the named
    candidate's, else None."""
    if delivery.delivered_coverage is not None or delivery.delivered_candidate is None:
        return delivery.delivered_coverage
    return selected_coverage_payload(phase, delivery.delivered_candidate, target_file)


def _write_matrices(mdir: Path, phase: PhaseConsolidation, target_file: str,
                    delivery: PhaseEndDelivery, coverage: Optional[dict]) -> List[str]:
    """Writes into ``mdir`` the kill matrices of the merged, delivered and
    reduced suites (the reduced one also as measured and as derived) and of
    the phase's own suite, with that suite's coverage matrix, each bound to
    the suite it describes, and the delivered suite's selected coverage
    ``coverage``. Returns the names written, in order. Selected coverage
    without the delivered source, or bound to other bytes, raises
    RuntimeError."""
    import json
    expected: List[str] = []
    # Every file names the suite it describes and carries that suite's hash,
    # whatever won: the merge under ``merged_*``, the reduction under
    # ``reduced_*``, the phase's own suite under ``candidate2_*`` and the
    # delivered suite under ``selected_*``. No file changes meaning with the
    # outcome, so no reader needs the winner to know what it is reading.
    merged_hash = phase.merged_source_sha256 or None
    reduced_hash = phase.reduced_source_sha256 or None
    candidate2_hash = (source_sha256(phase.candidate2_source)
                       if phase.candidate2_source is not None else None)
    for name, matrix, digest in (
        ("merged_kill_matrix.json", phase.merged_km, merged_hash),
        ("selected_kill_matrix.json", delivery.delivered_km, delivery.delivered_hash),
        ("reduced_kill_matrix.json", phase.reduced_km, reduced_hash),
        ("candidate2_kill_matrix.json", phase.candidate2_km, candidate2_hash),
        ("derived_reduced_kill_matrix.json", phase.derived_reduced_km, reduced_hash),
        ("measured_reduced_kill_matrix.json", phase.measured_reduced_km, reduced_hash),
        ("candidate2_coverage_matrix.json", phase.candidate2_coverage, candidate2_hash),
    ):
        if matrix is not None:
            (mdir / name).write_text(
                json.dumps(_bound_to_source(matrix, digest, delivery.execution_name),
                           indent=2),
                encoding="utf-8",
            )
            expected.append(name)
    if coverage is not None:
        if delivery.delivered_source is None:
            raise RuntimeError(
                "selected coverage evidence requires the selected source"
            )
        existing_hash = coverage.get("source_sha256")
        if existing_hash not in (None, delivery.delivered_hash):
            raise RuntimeError(
                "selected coverage evidence is bound to different source bytes"
            )
        selected_coverage = dict(coverage)
        selected_coverage.update({
            "schema_version": selected_coverage.get("schema_version", 1),
            "target_file": target_file,
            "source_sha256": delivery.delivered_hash,
            "execution_name": delivery.execution_name,
            "structural_parser_identity": phase.structural_parser_identity,
            "target_validator_identity": phase.target_validator_identity,
        })
        validity.write_json_atomic(
            mdir / "selected_coverage_matrix.json", selected_coverage,
        )
        expected.append("selected_coverage_matrix.json")
    return expected


def _ancestry_removals(phase: PhaseConsolidation,
                       delivery: "PhaseEndDelivery") -> List[Dict[str, Any]]:
    """The removals' validity derivations in the delivered suite's own
    ancestry: the merged suite's for a merged or reduced suite delivered,
    else the delivered candidate's, whose changed source it delivers."""
    if delivery.winner_label in ("merged", "reduced"):
        return [record for record in phase.removal_derivations
                if record["stage"] == "phase_end.merged"]
    return [record for record in phase.removal_derivations
            if record["stage"] != "phase_end.merged"
            and record["child_source_sha256"] == delivery.delivered_hash]


def _write_payloads(mdir: Path, phase: PhaseConsolidation,
                    delivered_validity_record: Optional[dict],
                    delivery: "PhaseEndDelivery") -> List[str]:
    """Writes the starter pool's matrices and records, the validity records
    and each changed candidate source that is not delivered into ``mdir``.
    Returns the names written, in order."""
    import json
    expected: List[str] = []
    derivations = [*_ancestry_removals(phase, delivery), *phase.validity_derivations]
    if phase.starter_pool_km is not None:
        (mdir / "starter_pool_kill_matrix.json").write_text(
            json.dumps(phase.starter_pool_km, indent=2), encoding="utf-8")
        expected.append("starter_pool_kill_matrix.json")
    for name, payload in (
        ("starter_pool_attribution.json", phase.starter_pool_attribution),
        ("starter_pool_derivation.json", phase.starter_pool_derivation),
        ("starter_pool_derived_kill_matrices.json",
         {"schema": phase.starter_pool_derivation["schema"] if phase.starter_pool_derivation else None,
          "suites": phase.starter_pool_derived_kms} if phase.starter_pool_derived_kms else None),
        ("starter_pool_pooled_kill_matrix.json", phase.starter_pool_pooled_km),
        ("validity_merged.json", phase.merged_validity_record),
        ("validity_derivations.json",
         {"schema_version": validity.RECORD_SCHEMA_VERSION,
          "records": derivations, **timing_fields()}
         if derivations else None),
        ("validity_delivered.json", delivered_validity_record),
    ):
        if payload is not None:
            validity.write_json_atomic(mdir / name, payload)
            expected.append(name)
    for name, source in phase.removed_sources.items():
        if source_sha256(source) != delivery.delivered_hash:
            (mdir / name).write_text(source, encoding="utf-8")
            expected.append(name)
    return expected


def _consolidation_record(target_file: str, phase: PhaseConsolidation,
                          delivery: PhaseEndDelivery, ancestry: Optional[dict],
                          detail: str, expected: List[str],
                          delivered_validity_record: Optional[dict]) -> Dict[str, Any]:
    """consolidation_record.json's content: the decision (the winner, the
    exact source when a single input suite wins), the counts, the files
    written (``expected``) and why the absent ones are absent. Its keys are
    in the order the file holds them."""
    return {
        "schema_version": 2,
        **timing_fields(),
        "target_file": target_file,
        "delivered_from": delivery.winner_label,
        "winner_source_label": delivery.winner_source_label,
        "generation_ancestry": ancestry,
        "selected_source_sha256": delivery.delivered_hash,
        "execution_name": delivery.execution_name,
        "target_revision": phase.target_revision,
        "measurement_identity": phase.measurement_identity,
        "structural_parser_identity": phase.structural_parser_identity,
        "target_validator_identity": phase.target_validator_identity,
        "merged_source_sha256": phase.merged_source_sha256 or None,
        "reduced_source_sha256": phase.reduced_source_sha256 or None,
        "merged_tests": phase.pool_tests,
        "delivered_tests": delivery.delivered_tests,
        "reduction_valid": bool(phase.reduction_valid),
        "repaired_tests": sorted(phase.repaired_tests),
        "evidence_repairs": phase.evidence_repairs,
        "unreduced_fallback": "merged_unreduced" in phase.fallback_candidates,
        "fallback_reason": phase.fallback_reason,
        "fallback_candidates": sorted(phase.fallback_candidates),
        "fallback_merged_stats": phase.fallback_union_stats,
        "selection_audit": phase.selection_audit,
        "detail": detail or phase.detail,
        "derive_measure_delta": phase.derive_measure_delta,
        "derive_measure_differences": phase.derive_measure_differences,
        "reduction_solver": phase.solver,
        "reduction_solver_metadata": phase.solver_metadata,
        "consolidation_mutation_method": phase.mutation_method,
        "artifacts": sorted(expected),
        "collision_resolution": phase.collision_resolution,
        "input_normalizations": phase.input_normalizations,
        # Measured when the merge or the delivered suite has a kill matrix: a
        # single suite kept without a merge (the tie-break when the merge was
        # unavailable) was measured on its own.
        "mutation_measured": phase.merged_km is not None or delivery.delivered_km is not None,
        "shipped_has_matrix": delivery.delivered_km is not None,
        # Only a phase end whose runner the origin check stopped holds it.
        **({"mutation_origin_stops": phase.mutation_origin_stops}
           if phase.mutation_origin_stops else {}),
        **({"measurement_removals": phase.measurement_removals}
           if phase.measurement_removals else {}),
        "validity_policy_version": (
            (delivered_validity_record or {}).get("policy_version")
        ),
        "validity_artifacts": sorted(
            name for name in expected if name.startswith("validity_")
        ),
    }


def _construction_ancestry(delivery: PhaseEndDelivery, generation_scope: Optional[dict],
                           generation_inputs: Optional[dict]) -> Optional[dict]:
    """The delivered suite's generation ancestry: its construction within
    ``generation_scope`` by the winner, or, without a scope, a construction
    that claims no inputs."""
    from testuary import generation_evidence
    if generation_scope:
        return generation_evidence.select_construction(
            delivery.delivered_source, scope=generation_scope,
            winner=delivery.winner_source_label or delivery.winner_label,
            inputs=generation_inputs or {},
        )
    # Without a generation scope the output belongs to no phase or
    # invocation, so it cannot claim construction inputs.
    return generation_evidence.construction(
        delivery.delivered_source, phase=None, operation="unavailable",
        reason="no generation scope",
    )


def persist_phase_end(
    *,
    matrices_dir,
    target_file: str,
    phase: PhaseConsolidation,
    delivery: PhaseEndDelivery,
    detail: str = "",
    generation_scope: Optional[dict] = None,
    generation_inputs: Optional[dict] = None,
) -> None:
    """The one persistence point for a phase end, called on every outcome
    branch (merged wins, a single suite wins, the merge fails, the files
    stay separate), with what the phase end delivers (``delivery``). Writes
    everything the consolidation produced:

    - ``merged_kill_matrix.json``: the merged suite's full measurement,
      whenever mutation ran.
    - ``reduced_kill_matrix.json``: the reduction's matrix, whenever the
      reduction has one, whatever won.
    - ``selected_kill_matrix.json`` and ``selected_coverage_matrix.json``:
      the delivered suite's evidence, whatever won, which every later reader
      takes.
    - ``candidate2_*``: the phase's own suite, measured standalone.
    - the per-test coverage sidecars (``persist_coverage_sidecars``).
    - ``consolidation_record.json``: always. The decision itself (the
      winner, the exact source when a single input suite wins, the counts,
      which artifacts exist and why the absent ones are absent), so a reader
      never infers an outcome from a missing file.

    A write failure raises, and the written set is verified before
    returning: a boundary that cannot persist its evidence fails at the
    boundary."""
    import json
    if matrices_dir is None:
        return
    ancestry = _construction_ancestry(delivery, generation_scope, generation_inputs)
    delivered_validity_record = (
        _delivered_validity_record(phase, target_file, delivery.delivered_source)
        if delivery.delivered_source is not None else None
    )

    coverage = _delivered_coverage(phase, target_file, delivery)

    # An incomplete attempt can leave sidecars in the phase archive before a
    # verified-prefix resume reruns this boundary. Remove only files owned by
    # this persistence point so an unresolved or differently shaped retry
    # cannot inherit a stale matrix. Immutable snapshots, suites, candidate
    # sources, and pre_merge/ are outside this set and remain untouched.
    mdir = _phase_end_dir(matrices_dir)

    expected = _write_matrices(mdir, phase, target_file, delivery, coverage)
    expected += _write_payloads(mdir, phase, delivered_validity_record, delivery)
    expected += persist_coverage_sidecars(phase, mdir, target_file)
    # The record goes last, so an incomplete set leaves no record, which a
    # resume treats as incomplete.
    missing = [name for name in expected if not (mdir / name).exists()]
    if missing:
        raise RuntimeError(
            f"phase-boundary persistence incomplete under {mdir}: "
            f"missing {missing}")

    record = _consolidation_record(
        target_file, phase, delivery, ancestry, detail, expected,
        delivered_validity_record,
    )
    (mdir / "consolidation_record.json").write_text(
        json.dumps(record, indent=2, allow_nan=False), encoding="utf-8")


@dataclass
class ScoredSuite:
    """A candidate single-file suite with its measured metrics, its source, and
    its kill matrix, the unit the phase-end keep-best-of-N selects among. A
    non-green suite (``green=False``) is out of contention. The merged-reduced
    candidate is non-green when the merge could not be measured."""

    label: str
    source: str
    line: Optional[float] = None
    branch: Optional[float] = None
    mutation: Optional[float] = None
    runtime_ms: Optional[float] = None
    green: bool = True
    status: str = "passing"
    kill_matrix: Optional[dict] = None
    coverage_evidence: Optional[dict] = None
    per_test_tokens: Optional[Dict[str, FrozenSet[str]]] = None
    test_count: Optional[int] = None
    covered_lines: Optional[FrozenSet[int]] = None
    covered_branches: Optional[FrozenSet[str]] = None
    execution_name: str = ""
    source_sha256: str = ""
    status_reason: str = ""
    removed_tests: Tuple[str, ...] = ()
    repaired_tests: Tuple[str, ...] = ()
    target_revision: Optional[str] = None
    measurement_identity: Dict[str, Any] = field(default_factory=dict)
    suite_timing: Optional[Dict[str, Any]] = None
    #: Its mutation score is derived from another suite's kill matrix,
    #: not measured (:func:`scored_from_phase`).
    mutation_derived: bool = False

    def __post_init__(self) -> None:
        computed = source_sha256(self.source)
        if self.source_sha256 and self.source_sha256 != computed:
            raise ValueError(
                f"candidate {self.label!r} source hash does not match its "
                "source bytes"
            )
        self.source_sha256 = computed


#: Why a built reduction is no choice, by its audit issue, in plain words.
_REDUCTION_ISSUES = {
    "reduced_failing": "the reduced suite does not pass",
    "reduced_coverage_refresh_failed": "the coverage tool could not measure the reduced suite",
    "reduced_remeasure_no_matrix": "the mutation tool gave no result for the reduced suite",
}
#: The reduced suite's mutation score under reduced_suite_mutation = derive.
_DERIVED_REDUCED_MUTATION = (
    "its mutation score is derived from the merged suite's kill matrix for its "
    "tests, not measured")


def scored_from_phase(phase: "PhaseConsolidation") -> ScoredSuite:
    """The reduced suite from :func:`consolidate_phase_end` as a phase-end
    candidate. A successful cover can keep every test and therefore have the
    same bytes as the merged suite. ``reduction_valid`` distinguishes that
    valid no-op reduction from a builder path that produced only the merge.
    A reduction never built is ``absent``, with the phase end's detail, and
    a built one the phase end excluded is ``failing`` or ``incomplete``,
    with its issue. Removed tests and the test count belong to a built
    reduction only."""
    from testuary.languages.mutation import matrix_score_percent
    mutation = (matrix_score_percent(phase.reduced_km)
                if phase.reduced_km else None)
    built = phase.reduced_source is not None
    issue = phase.selection_audit.get("reduction_measurement_issue")
    if not built:
        status, reason = "absent", phase.detail
    elif phase.reduction_valid:
        status, reason = "passing", _REDUCTION_ISSUES.get(issue, "")
    else:
        status = "failing" if issue == "reduced_failing" else "incomplete"
        reason = _REDUCTION_ISSUES.get(issue, "")
    derived = phase.reduced_km is not None and phase.reduced_km is phase.derived_reduced_km
    if derived:
        # A derived mutation score is named as derived, never as measured.
        reason = ", and ".join(filter(None, (reason, _DERIVED_REDUCED_MUTATION)))
    # A reduction never built has no numbers of its own: the phase's line
    # and branch are then the merged suite's, its time the kept tests'.
    line, branch, runtime_ms = ((phase.line, phase.branch, phase.runtime_ms)
                                if built else (None, None, None))
    return ScoredSuite(
        label="reduced", source=phase.reduced_source or "",
        line=line, branch=branch, mutation=mutation,
        runtime_ms=runtime_ms,
        green=built and phase.reduction_valid,
        status=status,
        status_reason=reason,
        mutation_derived=derived,
        kill_matrix=phase.reduced_km,
        coverage_evidence={
            "line": line,
            "branch": branch,
            "covered_statement_lines": phase.suite_covered_lines,
            "covered_branch_arcs": phase.suite_covered_branches,
            "per_test_tokens": (
                {
                    test_id: tokens
                    for test_id, tokens in phase.per_test_tokens.items()
                    if phase.kept_tests is None or test_id in phase.kept_tests
                }
                if phase.per_test_tokens is not None else None
            ),
        },
        per_test_tokens=(
            {
                str(test_id): frozenset(map(str, tokens))
                for test_id, tokens in phase.per_test_tokens.items()
                if phase.kept_tests is None or test_id in phase.kept_tests
            }
            if phase.per_test_tokens is not None else None
        ),
        test_count=phase.selected_tests if built else None,
        covered_lines=(
            frozenset(phase.suite_covered_lines)
            if phase.suite_covered_lines is not None else None
        ),
        covered_branches=(
            frozenset(phase.suite_covered_branches)
            if phase.suite_covered_branches is not None else None
        ),
        execution_name=phase.execution_name,
        removed_tests=tuple(sorted(
            set(phase.per_test_tokens or {}) - set(phase.kept_tests or ())
        )) if built else (),
        repaired_tests=tuple(sorted(phase.repaired_tests)),
        target_revision=phase.target_revision,
        measurement_identity=phase.measurement_identity,
    )


def scored_from_measurement(
    *,
    label: str,
    source: str,
    kill_matrix: Optional[dict],
    coverage: Optional[dict],
    execution_name: str,
    green: bool,
    status: str,
    runtime_ms: Optional[float] = None,
    test_count: Optional[int] = None,
    status_reason: str = "",
    removed_tests: Sequence[str] = (),
    repaired_tests: Sequence[str] = (),
    target_revision: Optional[str] = None,
    measurement_identity: Optional[Dict[str, Any]] = None,
) -> ScoredSuite:
    """Build one selectable candidate from measurements bound to its bytes.

    Coverage identities and aggregate values come from the same sidecar. A
    missing sidecar leaves those axes unmeasured, so the common selector
    excludes the candidate instead of interpreting missing evidence as zero.
    """
    from testuary.languages import mutation

    cov = coverage if isinstance(coverage, dict) else {}
    lines, branches = _coverage_sets(cov)
    return ScoredSuite(
        label=label,
        source=source,
        line=cov.get("line"),
        branch=cov.get("branch"),
        mutation=(
            mutation.matrix_score_percent(kill_matrix)
            if kill_matrix is not None else None
        ),
        runtime_ms=runtime_ms,
        green=green,
        status=status,
        kill_matrix=kill_matrix,
        coverage_evidence=dict(cov) if cov else None,
        per_test_tokens=(
            {
                str(test_id): frozenset(map(str, tokens))
                for test_id, tokens in (cov.get("per_test_tokens") or {}).items()
            }
            if cov.get("per_test_tokens") is not None else None
        ),
        test_count=test_count,
        covered_lines=lines,
        covered_branches=branches,
        execution_name=execution_name,
        status_reason=status_reason,
        removed_tests=tuple(sorted(str(item) for item in removed_tests)),
        repaired_tests=tuple(sorted(str(item) for item in repaired_tests)),
        target_revision=target_revision,
        measurement_identity=dict(measurement_identity or {}),
    )



def merged_scored_from_phase(phase: "PhaseConsolidation") -> ScoredSuite:
    """The merged suite as a phase-end candidate: the union coverage and kill
    matrix measured before the cover ran, and the pool's warmed per-test
    runtime. Naming the merged suite apart from the reduced suite is what lets the
    selection see a cover that lost evidence, rather than delivering the merged suite
    under the reduced suite's name."""
    from testuary.languages.mutation import matrix_score_percent
    mutation = (matrix_score_percent(phase.merged_km)
                if phase.merged_km else None)
    return ScoredSuite(
        label="merged", source=phase.merged_source or "",
        line=phase.union_line, branch=phase.union_branch, mutation=mutation,
        runtime_ms=phase.pool_runtime_ms,
        green=bool(phase.merged_source) and phase.merged_green,
        status=(
            "failing" if phase.merged_source and not phase.merged_green
            else "passing" if phase.merged_source and not phase.merged_incomplete
            else "incomplete"
        ),
        kill_matrix=phase.merged_km,
        coverage_evidence={
            "line": phase.union_line,
            "branch": phase.union_branch,
            "covered_statement_lines": phase.union_covered_lines,
            "covered_branch_arcs": phase.union_covered_branches,
            "per_test_tokens": phase.per_test_tokens,
        },
        per_test_tokens=(
            {
                str(test_id): frozenset(map(str, tokens))
                for test_id, tokens in phase.per_test_tokens.items()
            }
            if phase.per_test_tokens is not None else None
        ),
        test_count=phase.pool_tests,
        covered_lines=(
            frozenset(phase.union_covered_lines)
            if phase.union_covered_lines is not None else None
        ),
        covered_branches=(
            frozenset(phase.union_covered_branches)
            if phase.union_covered_branches is not None else None
        ),
        execution_name=phase.execution_name,
        status_reason=("" if phase.merged_green and not phase.merged_incomplete
                       else phase.detail),
        target_revision=phase.target_revision,
        measurement_identity=phase.measurement_identity,
    )


def phase_end_candidates(
    phase: "PhaseConsolidation",
) -> List[ScoredSuite]:
    """The two suites a consolidation produces, named apart: the reduced and
    the merged suite. A tie on every score goes to the reduced suite, the
    leaner of the two, by the selection's preferred label, whatever their
    order."""
    return [scored_from_phase(phase), merged_scored_from_phase(phase)]


def measurement_names(required_axes: Sequence[str]) -> str:
    """The measurements a phase end requires, named in prose: "branch and
    line", or "mutation, branch and line" with phase_end_mutation on."""
    names = list(required_axes)
    if len(names) < 2:
        return "".join(names)
    return ", ".join(names[:-1]) + " and " + names[-1]


def _tie_break_time(candidate: Any) -> Optional[float]:
    """The time the runtime tie-break gave ``candidate``, None when it could
    not time it (no timing benchmark, a failed timing run)."""
    if (candidate.suite_timing or {}).get("status") in ("unavailable", "failed"):
        return None
    return candidate.runtime_ms


def record_selection_audit(
    phase: "PhaseConsolidation",
    candidates: Sequence[ScoredSuite],
    winner: Optional[ScoredSuite],
    *,
    required_axes: Sequence[str],
) -> None:
    """Record candidate completeness and evidence differences without gating.

    The phase-end decision is score based. This record separately answers
    whether the selected suite retained each measured input's killed-mutant,
    covered-line, and covered-branch identities. Missing measurements remain
    explicit rather than being interpreted as empty evidence.
    """
    inputs = [
        candidate for candidate in candidates
        if candidate.label not in _MERGED_RUN_LABELS
    ]
    active_axes = tuple(required_axes)

    census = [
        {
            "label": candidate.label,
            "passing": bool(candidate.green),
            "status": candidate.status,
            "fully_measured": selectable(candidate, active_axes),
            "mutation": candidate.mutation,
            "branch": candidate.branch,
            "line": candidate.line,
            # A tied candidate the tie-break could not time records null,
            # not its provisional time from another basis.
            "runtime_ms": _tie_break_time(candidate),
            "tests": candidate.test_count,
            "source_sha256": (candidate.source_sha256
                              if candidate.status != "absent" else None),
            "execution_name": candidate.execution_name,
            "status_reason": candidate.status_reason,
            "removed_tests": list(candidate.removed_tests),
            "repaired_tests": list(candidate.repaired_tests),
            "target_revision": candidate.target_revision,
            "measurement_identity": candidate.measurement_identity,
            "suite_timing": candidate.suite_timing,
            # Only a candidate whose mutation score is derived holds it.
            **({"mutation_derived": True} if candidate.mutation_derived else {}),
        }
        for candidate in candidates
    ]
    complete_candidates = [candidate for candidate in candidates
                           if selectable(candidate, active_axes)]
    top_labels: List[str] = []
    selection_basis: Optional[str] = None
    runtime_tie_break_basis: Optional[str] = None
    if complete_candidates:
        top_labels = [candidate.label
                      for candidate in top_tied(complete_candidates, active_axes)]
        if len(top_labels) == 1:
            selection_basis = "score"
        elif winner is not None and winner.label == "reduced":
            selection_basis = "reduced_preference"
        else:
            selection_basis = "timing_median"
            medians = [_tie_break_time(candidate) for candidate in complete_candidates
                       if candidate.label in top_labels]
            runtime_tie_break_basis = (
                "input_order" if None in medians or medians.count(min(medians)) > 1
                else "median")
    missing_measurements = {
        "kills": [c.label for c in inputs if c.kill_matrix is None],
        "lines": [c.label for c in inputs if c.covered_lines is None],
        "branches": [c.label for c in inputs if c.covered_branches is None],
    }

    def _missing_count(kind: str) -> Optional[int]:
        if winner is None:
            return None
        if kind == "kills":
            if winner.kill_matrix is None or missing_measurements[kind]:
                return None
            required = set().union(*(
                _killed_by_identity(candidate.kill_matrix)
                for candidate in inputs
            )) if inputs else set()
            return len(required - _killed_by_identity(winner.kill_matrix).keys())
        attribute = (
            "covered_lines" if kind == "lines" else "covered_branches"
        )
        winner_values = getattr(winner, attribute)
        if winner_values is None or missing_measurements[kind]:
            return None
        required = set().union(*(
            set(getattr(candidate, attribute) or ()) for candidate in inputs
        )) if inputs else set()
        return len(required - set(winner_values))

    tie_break_rounds: List[Dict[str, Any]] = []
    for candidate in candidates:
        record = candidate.suite_timing or {}
        if record.get("order") != "interleaved":
            continue
        batch = record.get("batch")
        if any(item["batch"] == batch for item in tie_break_rounds):
            continue
        tie_break_rounds.append({
            "batch": batch,
            "candidates": sorted(
                other.label for other in candidates
                if (other.suite_timing or {}).get("batch") == batch
                and (other.suite_timing or {}).get("order") == "interleaved"
            ),
            "rounds": record.get("rounds"),
        })
    phase.selection_audit = {
        **phase.selection_audit,
        "selector": "_then_".join(active_axes),
        "active_axes": list(active_axes),
        "delivered_from": winner.label if winner is not None else None,
        "selection_basis": selection_basis,
        "runtime_tie_break_basis": runtime_tie_break_basis,
        # The order the timing tie-break took, null when none ran.
        "runtime_tie_break_candidate_order": (
            top_labels if runtime_tie_break_basis is not None else None),
        "runtime_tie_break_rounds": tie_break_rounds,
        "top_score_labels": top_labels,
        "candidates": census,
        "input_evidence_missing_measurements": missing_measurements,
        "delivered_from_missing_from_merged_input": {
            kind: _missing_count(kind)
            for kind in ("kills", "lines", "branches")
        },
    }
    # A merged suite selected over a distinct reduction is an ordinary selection
    # outcome, recorded here at every boundary. The reduction keeps its own
    # evidence under its own names. Nothing is rewritten to make one
    # candidate stand for another.
    if (
        winner is not None
        and winner.label == "merged"
        and phase.merged_source is not None
        and phase.reduced_source is not None
        and phase.reduced_source != phase.merged_source
    ):
        phase.selection_audit["reduced_not_selected"] = True
        phase.selection_audit["merged_stats"] = _union_bloat_stats(
            phase.per_test_tokens,
            attempted_kept=len(phase.kept_tests or ()),
            renamed_names=_renamed_names(phase),
        )



def measure_interleaved(
    backend,
    sources: Sequence[Tuple[str, str]],
    *,
    source_dir: str,
    target_file_path: str,
    timeout: int,
    warmup_runs: int,
    benchmark_runs: int,
    stage_as_delivered: bool = False,
    log: Callable[[str], None] = print,
) -> Dict[str, Any]:
    """Time the suites of ``sources`` (``(label, source)`` pairs) together,
    interleaved round by round through the backend's
    ``benchmark_interleaved``: ``warmup_runs`` discarded rounds, then
    ``benchmark_runs`` timed rounds, each executing every suite once with a
    rotating first suite, so that a drift in the machine's load falls on all
    of them alike rather than on the one whose block it overlaps.

    Returns ``{"status": "unavailable", "error"}`` when the language has no
    timing benchmark, else ``{"status": "measured", "rounds", "results"}``,
    each result either ``{"status": "measured", "samples_ms", "median_ms",
    "warmup_samples_ms"}`` or, for a suite whose tests failed (it left the
    rounds), ``{"status": "failed", "failure_kind": "test_failure",
    "returncode", "error"}``. A timeout, malformed result or tool failure
    raises: it is an infrastructure failure, not a measurement."""
    benchmark = getattr(backend, "benchmark_interleaved", None)
    if benchmark is None:
        raise RuntimeError(
            "the phase-end runtime tie-break requires a backend benchmark_interleaved "
            "implementation"
        )
    if benchmark_runs <= 0:
        raise RuntimeError(
            "the phase-end runtime tie-break requires at least one measured repetition, but the resolved "
            f"[{config.phase_section('timing')}] run_count={benchmark_runs!r}"
        )
    for label, source in sources:
        if not source:
            raise RuntimeError(
                f"the candidate {label!r} of the phase-end runtime tie-break has no source to "
                "benchmark"
            )
    from contextlib import ExitStack

    with ExitStack() as stack:
        staged = []
        for label, source in sources:
            td = stack.enter_context(
                project.scratch_dir(prefix="testuary_tiebreak_"))
            path = _stage_in(backend, td, source, target_file_path,
                             stage_as_delivered=stage_as_delivered)
            staged.append((label, [str(path)]))
        stats = benchmark(
            source_dir=source_dir,
            suites=staged,
            warmup_runs=warmup_runs,
            benchmark_runs=benchmark_runs,
            timeout=timeout,
            progress_fn=lambda m: log(f"[consolidate] {m}"),
            instrument_target=target_file_path,
        )
    stats = stats or {}
    if stats.get("unavailable"):
        return {"status": "unavailable", "error": stats.get("error")}
    rounds = [
        {"round": int(item["round"]), "warmup": bool(item["warmup"]),
         "order": [str(label) for label in item["order"]]}
        for item in stats.get("rounds") or ()
    ]
    records = stats.get("suites") or {}
    # The suite whose failure ended the rounds is named, not the suites the
    # end left incomplete.
    for label, _source in sources:
        record = records.get(label) or {}
        if (not record.get("ok", False) and not record.get("incomplete")
                and record.get("failure_kind") != "test_failure"):
            raise RuntimeError(
                f"the benchmark of the phase-end runtime tie-break for {label!r} failed: "
                f"{record.get('error') or 'unknown error'}"
            )
    results: Dict[str, Dict[str, Any]] = {}
    for label, _source in sources:
        record = records.get(label)
        if record is None:
            raise RuntimeError(
                f"the benchmark of the phase-end runtime tie-break returned no result for {label!r}"
            )
        if not record.get("ok", False):
            if record.get("failure_kind") == "test_failure":
                results[label] = {
                    "status": "failed",
                    "failure_kind": "test_failure",
                    "returncode": record.get("returncode"),
                    "error": record.get("error"),
                }
                continue
            raise RuntimeError(
                f"the benchmark of the phase-end runtime tie-break for {label!r} failed: "
                f"{record.get('error') or 'unknown error'}"
            )
        median = record.get("median_ms")
        if median is None:
            raise RuntimeError(
                f"the benchmark of the phase-end runtime tie-break for {label!r} returned no median"
            )
        samples = record.get("times_ms")
        warmups = record.get("warmup_times_ms")
        if not all(
            isinstance(found, list)
            and len(found) == expected
            and all(
                isinstance(sample, (int, float))
                and math.isfinite(float(sample))
                and float(sample) >= 0
                for sample in found
            )
            for found, expected in ((samples, benchmark_runs),
                                    (warmups, warmup_runs))
        ):
            raise RuntimeError(
                f"the benchmark of the phase-end runtime tie-break for {label!r} did not return "
                "every finite timing sample"
            )
        samples = [float(sample) for sample in samples]
        value = float(median)
        if not math.isclose(value, float(statistics.median(samples)),
                            rel_tol=0.0, abs_tol=1e-9):
            raise RuntimeError(
                f"the benchmark of the phase-end runtime tie-break for {label!r} returned a median "
                "that does not replay from its samples"
            )
        results[label] = {
            "status": "measured",
            "samples_ms": samples,
            "median_ms": value,
            "warmup_samples_ms": [float(sample) for sample in warmups],
        }
    return {"status": "measured", "rounds": rounds, "results": results}


class TiedCandidateTimer:
    """The phase-end runtime tie-break: ``measure_tied`` times the tied
    candidates together (:func:`measure_interleaved`), and a call returns
    one candidate's median from that measurement.

    The tied set is measured as one interleaved batch. A later tied set that
    lies within the last batch (the same candidates, or its survivors after
    an exclusion) reuses that batch, whose candidates shared every round. Any
    other set is measured anew as a whole, so the compared medians always
    come from the same rounds. A call for a candidate never measured
    measures it alone.

    Timing only adds a time. When the language has no timing benchmark, or
    a tied candidate's timing run fails (which gives it no time and a
    warning, never a verdict on the suite), no candidate of the batch has
    a time (None), so the input order decides."""

    def __init__(self, backend, *, source_dir: str, target_file_path: str,
                 timeout: int, warmup_runs: int, benchmark_runs: int,
                 stage_as_delivered: bool,
                 log: Callable[[str], None]) -> None:
        self._backend = backend
        self._source_dir = source_dir
        self._target_file_path = target_file_path
        self._timeout = timeout
        self.warmup_runs = int(warmup_runs)
        self.benchmark_runs = int(benchmark_runs)
        self._stage_as_delivered = stage_as_delivered
        self._log = log
        self._measured: Dict[str, Optional[float]] = {}
        self._batch: FrozenSet[str] = frozenset()
        self.batches = 0

    def measure_tied(self, candidates: Sequence[Any]) -> None:
        labels = [candidate.label for candidate in candidates]
        if set(labels) <= self._batch:
            return
        outcome = measure_interleaved(
            self._backend,
            [(candidate.label, candidate.source) for candidate in candidates],
            source_dir=self._source_dir,
            target_file_path=self._target_file_path,
            timeout=self._timeout,
            warmup_runs=self.warmup_runs,
            benchmark_runs=self.benchmark_runs,
            stage_as_delivered=self._stage_as_delivered,
            log=self._log,
        )
        self.batches += 1
        self._batch = frozenset(labels)
        if outcome["status"] == "unavailable":
            # The language has no timing benchmark (the timing fallback):
            # no tied candidate has a time, so the input order decides.
            for candidate in candidates:
                candidate.suite_timing = {
                    "status": "unavailable", "error": outcome.get("error")}
                self._measured[candidate.label] = None
            self._log("[consolidate] tie-break: no timing benchmark for "
                      + ", ".join(repr(label) for label in labels)
                      + ", so the input order decides")
            return
        failed = []
        for candidate in candidates:
            result = outcome["results"][candidate.label]
            common = {"order": "interleaved", "batch": self.batches,
                      "rounds": outcome["rounds"]}
            if result["status"] == "failed":
                candidate.suite_timing = {
                    "status": "failed",
                    "failure_kind": "test_failure",
                    "returncode": result.get("returncode"),
                    "error": result.get("error"),
                    **common,
                }
                failed.append(candidate.label)
                continue
            candidate.suite_timing = {
                "warmup_runs": self.warmup_runs,
                "timing_run_count": self.benchmark_runs,
                "samples_ms": result["samples_ms"],
                "median_ms": result["median_ms"],
                **common,
            }
            self._measured[candidate.label] = result["median_ms"]
            self._log(f"[consolidate] tie-break: '{candidate.label}' "
                      f"median {result['median_ms']:.0f}ms over "
                      f"{self.benchmark_runs} interleaved rounds")
        if failed:
            # A timing run that failed gives no time, so the batch's times
            # do not compare: the input order decides, as without a timing
            # benchmark.
            for candidate in candidates:
                self._measured[candidate.label] = None
            self._log(f"WARNING: {'suite' if len(failed) == 1 else 'suites'} "
                      + ", ".join(repr(label) for label in failed)
                      + " could not be timed, so equally good suites keep their order")

    def __call__(self, candidate) -> Optional[float]:
        """The candidate's median time, None when it was not timed."""
        if candidate.label not in self._measured:
            self.measure_tied([candidate])
        return self._measured[candidate.label]


def benchmark_tied_candidates(
    backend,
    *,
    source_dir: str,
    target_file_path: str,
    timeout: int,
    stage_as_delivered: bool = False,
    log: Callable[[str], None] = print,
) -> TiedCandidateTimer:
    """The phase-end runtime tie-break with the configured repetitions:
    ``warmup_runs`` discarded rounds, then ``benchmark_runs`` timed rounds of
    the tied candidates interleaved (:class:`TiedCandidateTimer`), each
    candidate's runtime the median of its timed executions.

    Reached only when the scores tie and no reduced suite is among the tied,
    so the cost is paid only where runtime decides the delivered suite. A test
    failure in a timing run gives that candidate no time and a warning, and
    the input order decides the tie. A timeout, malformed result, or
    measurement-tool failure aborts selection as infrastructure failure. A
    recorded elapsed value from a different execution context is not a
    comparable replacement."""
    # Resolved per call rather than from a constant bound at import: the rep
    # count is a runtime setting, and a caller that imports this module before
    # the setting is in place would otherwise carry a stale one for the process.
    timing = config.load_config("timing")
    return TiedCandidateTimer(
        backend, source_dir=source_dir,
        target_file_path=target_file_path, timeout=timeout,
        warmup_runs=int(timing["warmup_count"]),
        benchmark_runs=int(timing["run_count"]),
        stage_as_delivered=stage_as_delivered, log=log,
    )


def _renamed_names(phase: PhaseConsolidation) -> Sequence[str]:
    """The names the merge's collision rename gave, from its record."""
    return (phase.collision_resolution or {}).get("renamed_names", ())


def _union_bloat_stats(
    per_test_tokens, attempted_kept: Optional[int] = None,
    *, union_tests: Optional[int] = None, renamed_names: Sequence[str] = (),
) -> Dict[str, object]:
    """Size, rename-twin count, and reduction attempt of the merged union,
    recorded when the boundary falls back from the reduction or selects the
    merged suite over a distinct reduction. A twin is the copy of a test
    that the merge's collision rename kept beside the original, in a renamed
    class too: tests whose ids agree once each dotted part the rename gave
    (``renamed_names``) is cut back to its name before ``_<k>`` form a
    group, and each group counts its tests beyond the first.
    ``attempted_kept`` is the size of the
    reduction the boundary attempted before falling back (None when no
    cover ran). The redundancy ratio is the union share that attempt
    judged droppable. ``per_test_tokens=None`` leaves token-derived fields
    null while preserving an independently measured aggregate
    ``merged_tests`` count when supplied. The record's ``fallback_reason``
    names the instrument gap that left the per-test universe unmeasured."""
    names = list(per_test_tokens) if per_test_tokens is not None else None
    if names is None:
        return {
            "merged_tests": union_tests, "rename_twins": None,
            "attempted_reduced_tests": attempted_kept,
            "redundancy_ratio": None,
        }
    renamed = set(renamed_names)
    groups: Dict[str, int] = {}
    for name in names:
        key = ".".join(part.rsplit("_", 1)[0] if part in renamed else part
                       for part in name.split("."))
        groups[key] = groups.get(key, 0) + 1
    twins = sum(n - 1 for n in groups.values())
    ratio = None
    if attempted_kept and names:
        ratio = round(1.0 - (attempted_kept / len(names)), 4)
    return {
        "merged_tests": len(names), "rename_twins": twins,
        "attempted_reduced_tests": attempted_kept,
        "redundancy_ratio": ratio,
    }


def _axis_value(value: Optional[float]) -> float:
    """A metric for ranking: the measured number, or ``-inf`` when unknown
    (None), so an unmeasured axis always loses to a measured one."""
    return float("-inf") if value is None else float(value)


def _candidate_axis_measured(candidate: Any, axis: str) -> bool:
    """Whether an axis was measured, including an empty denominator.

    Coverage reports no percentage when a target has no statements or no
    branches, and mutation reports no percentage when a catalogue has no valid
    mutants. Their attached identity set or matrix distinguishes that valid
    result from a measurement that never ran. A kill matrix the wall clock
    cut, stamped incomplete, measures no mutation axis: its score is not
    known.
    """
    if getattr(candidate, axis) is not None:
        return True
    evidence_attribute = {
        "mutation": "kill_matrix",
        "branch": "covered_branches",
        "line": "covered_lines",
    }[axis]
    evidence = getattr(candidate, evidence_attribute, None)
    if axis == "mutation" and isinstance(evidence, dict):
        from testuary.languages import mutation

        return not evidence.get(mutation.INCOMPLETE_KEY)
    return evidence is not None


def selectable(candidate: Any, required_axes: Sequence[str]) -> bool:
    """Whether the phase-end selection can choose ``candidate``: it passes,
    no tool left its numbers incomplete, and each of ``required_axes`` was
    measured."""
    return bool(candidate.green and getattr(candidate, "status", None) != "incomplete"
                and all(_candidate_axis_measured(candidate, axis) for axis in required_axes))


def top_tied(candidates: Sequence[Any], required_axes: Sequence[str]) -> List[Any]:
    """The ``candidates`` measured on every one of ``required_axes`` that tie
    on the best score, in their order. Empty when none is measured."""
    measured = [c for c in candidates
                if all(_candidate_axis_measured(c, axis) for axis in required_axes)]
    if not measured:
        return []

    def score(c: Any) -> Tuple[float, ...]:
        return tuple(_axis_value(getattr(c, axis)) for axis in required_axes)

    best = max(score(c) for c in measured)
    return [c for c in measured if score(c) == best]


def choose_best(
    candidates: Sequence[ScoredSuite],
    *,
    preferred_label: Optional[str] = None,
    runtime_of: Optional[Callable[[ScoredSuite], float]] = None,
    required_axes: Sequence[str],
    measure_tied: Optional[Callable[[Sequence[ScoredSuite]], None]] = None,
) -> Optional[ScoredSuite]:
    """Select one passing, fully measured phase-end candidate
    (:func:`selectable`).

    Candidates compare lexicographically on ``required_axes``, the scores
    :func:`phase_end_axes` names. None when no passing
    candidate is measured on every axis. If the best-scoring set contains
    ``preferred_label``, that candidate wins untimed. Otherwise every tied
    candidate is timed and the lowest median wins. A tie without
    ``runtime_of``, or an invalid median, raises ``RuntimeError`` instead of
    falling back to an elapsed time from another context. ``measure_tied``,
    when given, receives the whole tied set before any ``runtime_of`` call,
    so the tied candidates can be timed together. Exact evidence identities
    are not part of the decision. The audit records and checks them.

    ``required_axes`` is explicit, so a failed measurement cannot silently
    drop a score (an unknown axis raises ``ValueError``). An attached
    evidence matrix or identity set counts as the measurement of an axis
    whose denominator is empty, so its percentage is undefined. A caller
    that deliberately runs without mutation analysis requests the
    coverage-only pair.
    """
    active_axes = tuple(required_axes)
    unknown = set(active_axes) - {"mutation", "branch", "line"}
    if unknown:
        raise ValueError(f"unknown phase-end score axes: {sorted(unknown)}")
    tied = top_tied([c for c in candidates if selectable(c, active_axes)], active_axes)
    if not tied:
        return None
    if len(tied) == 1:
        return tied[0]
    # Equal scores: the phase-end policy names the reduced suite the
    # preferred result of the reduction, so no second, noisy runtime decision.
    if preferred_label is not None:
        preferred = next(
            (c for c in tied if c.label == preferred_label), None
        )
        if preferred is not None:
            return preferred
    # Otherwise their run cost separates them. One recorded elapsed time is a
    # single sample under the phase's load, so the winner is timed with the
    # warmup-and-repetitions protocol of every reported suite runtime.
    if runtime_of is None:
        raise RuntimeError(
            f"phase-end candidates tie on {measurement_names(active_axes)}, "
            "but no suite timing was provided"
        )
    if measure_tied is not None:
        measure_tied(tied)
    measured: Dict[int, float] = {}
    for candidate in tied:
        value = float(runtime_of(candidate))
        if not math.isfinite(value) or value < 0:
            raise RuntimeError(
                f"phase-end suite timing for {candidate.label!r} "
                f"returned an invalid median: {value!r}"
            )
        measured[id(candidate)] = value
    return min(tied, key=lambda c: (measured[id(c)], tied.index(c)))


def pick_phase_result(
    scored: Sequence[ScoredSuite],
    *,
    runtime_of: Optional[Callable[[Any], float]] = None,
    required_axes: Sequence[str],
    log: Callable[[str], None] = print,
) -> Optional[ScoredSuite]:
    """Select the delivered suite by mutation, branch, and line score.

    Only a passing suite with complete numbers is chosen: an incomplete one,
    which a tool could not measure, is no candidate. A tied reduced suite is
    preferred. If the reduced suite is not in the highest-scoring tie,
    ``runtime_of`` benchmarks only the tied candidates. Evidence identities
    remain attached to the returned suite for persistence and audit, but do
    not filter the candidate set.
    """
    def _measure(candidate) -> float:
        measured = runtime_of(candidate)
        if measured is None:
            # Untimed: no time is recorded, and the tie value 0.0, equal for
            # every untimed candidate, keeps the input order.
            return 0.0
        # A repeated tie-break measurement supersedes the provisional elapsed
        # time for selection records and logs. ``choose_best`` validates the
        # value before it can determine a winner.
        candidate.runtime_ms = float(measured)
        return candidate.runtime_ms

    chosen = choose_best(
        scored,
        preferred_label="reduced",
        runtime_of=_measure if runtime_of is not None else None,
        measure_tied=getattr(runtime_of, "measure_tied", None),
        required_axes=required_axes,
    )
    if chosen is None:
        return None
    log(f"[consolidate] phase result: adopting '{chosen.label}' "
        f"(mutation={chosen.mutation}, branch={chosen.branch}, "
        f"line={chosen.line}, runtime="
        + (f"{chosen.runtime_ms:.0f}ms)" if chosen.runtime_ms is not None else "unmeasured)"))
    return chosen


#: The scores a phase end ranks by, in order: the mutation score, then
#: branch and line coverage.
_PHASE_END_AXES = ("mutation", "branch", "line")


def phase_end_axes(runner) -> Tuple[str, ...]:
    """The scores a phase end requires: the mutation score besides branch and
    line coverage when it measures kills (a ``runner``)."""
    return _PHASE_END_AXES if runner is not None else _PHASE_END_AXES[1:]


def select_phase_result(
    phase: "PhaseConsolidation", candidates: Sequence[ScoredSuite], *,
    required_axes: Sequence[str], backend, source_dir: str,
    target_file_path: str, timeout: int, log: Callable[[str], None],
) -> Optional[ScoredSuite]:
    """A phase end's result among ``candidates`` (``pick_phase_result``, a
    tie timed by benchmark runs as delivered), with the selection recorded
    on ``phase``. None when no candidate has every required score."""
    winner = pick_phase_result(
        candidates,
        runtime_of=benchmark_tied_candidates(
            backend, source_dir=source_dir, target_file_path=target_file_path,
            timeout=timeout, stage_as_delivered=True, log=log),
        required_axes=required_axes, log=log)
    record_selection_audit(phase, candidates, winner, required_axes=required_axes)
    return winner


@dataclass(frozen=True)
class PhaseEnd:
    """One phase end's inputs: the suites it merges (``sources``, in merge
    order), where its records go (``matrices_dir``, also the solver's failure
    directory) and the name its suite is delivered under. ``options`` holds how
    the phase end measures and reduces, as the driver sets them, with each
    candidate's per-test evidence, which chooses the victims of cross-source
    name collisions in the merge. The solver's failure directory is
    ``matrices_dir``. Every candidate is measured under the delivered
    name."""

    name: str
    backend: Any
    sources: Sequence[str]
    target_file_path: str
    source_dir: str
    timeout: int
    matrices_dir: Any
    execution_name: Optional[str]
    log: Callable[[str], None]
    options: PhaseEndOptions
    generation_scope: Optional[dict] = None
    generation_inputs: Optional[dict] = None
    #: The unresolved record names the execution name and the generation
    #: inputs, as the mutation phase's does.
    unresolved_bound: bool = False


@dataclass
class PhaseEndOutcome:
    """A phase end's result: its consolidation, the selected suite (None when
    unresolved), the candidates in their selection order and the delivered
    test count. ``label`` is the label the suite is delivered under
    (single_suite for an input suite, whose own label is ``source_label``)."""

    phase: PhaseConsolidation
    winner: Optional[ScoredSuite]
    candidates: List[ScoredSuite]
    required_axes: Tuple[str, ...]
    label: str = "unresolved"
    source_label: Optional[str] = None
    delivered_tests: Optional[int] = None
    detail: str = ""


def run_phase_end(
    end: PhaseEnd,
    runner_scope,
    *,
    own_candidates: Callable[[PhaseConsolidation, List[ScoredSuite]], List[ScoredSuite]],
    no_winner: Callable[[str], str],
    single_suite: Optional[str] = None,
    delivered_tests: Optional[Callable[[ScoredSuite], Optional[int]]] = None,
) -> PhaseEndOutcome:
    """The phase end the coverage phase (target_phases._consolidate_coverage)
    and the mutation phase (measurement.consolidate_test_suite) share:
    consolidate ``end``'s sources with the runner ``runner_scope`` yields
    (closed once consolidation ends, so a driver that holds its runner longer
    passes a nullcontext of it), put the driver's candidates around the
    consolidation's (``own_candidates``), select (``select_phase_result``)
    and persist the outcome once. A winner whose label starts with ``single_suite`` is
    delivered as the single suite it is, its label kept as the source label.
    An unresolved end is persisted with ``no_winner``'s detail and returned.
    Delivery and failing are the driver's."""
    with runner_scope as runner:
        try:
            phase = consolidate_phase_end(
                PhaseEndContext(
                    backend=end.backend, target_file_path=end.target_file_path,
                    source_dir=end.source_dir, timeout=end.timeout,
                    stage_as_delivered=True, log=end.log),
                end.sources, runner,
                replace(end.options,
                        solver_failure_dir=(str(end.matrices_dir)
                                            if end.matrices_dir is not None else None)),
            )
        except NotImplementedError as exc:
            raise RuntimeError(
                f"the {end.name} phase end cannot merge and select the suites: the "
                f"language backend does not implement an operation it needs "
                f"({exc})") from exc
    required_axes = phase_end_axes(runner)
    candidates = own_candidates(phase, phase_end_candidates(phase))
    winner = select_phase_result(
        phase, candidates, required_axes=required_axes, backend=end.backend,
        source_dir=end.source_dir, target_file_path=end.target_file_path,
        timeout=end.timeout, log=end.log)
    if winner is None:
        detail = no_winner(measurement_names(required_axes))
        bound = end.unresolved_bound
        persist_phase_end(
            matrices_dir=end.matrices_dir, target_file=end.target_file_path,
            phase=phase,
            delivery=PhaseEndDelivery(
                winner_label="unresolved", delivered_km=None,
                execution_name=end.execution_name if bound else None),
            detail=detail, generation_scope=end.generation_scope,
            generation_inputs=end.generation_inputs if bound else None)
        return PhaseEndOutcome(phase, None, candidates, required_axes, detail=detail)
    label, source_label = winner.label, None
    if single_suite is not None and label.startswith(single_suite):
        label, source_label = "single_suite", winner.label
    tests = delivered_tests(winner) if delivered_tests is not None else winner.test_count
    persist_phase_end(
        matrices_dir=end.matrices_dir, target_file=end.target_file_path, phase=phase,
        delivery=PhaseEndDelivery(
            winner_label=label, winner_source_label=source_label,
            delivered_km=winner.kill_matrix, delivered_tests=tests,
            delivered_source=winner.source,
            delivered_candidate=winner, execution_name=end.execution_name),
        generation_scope=end.generation_scope,
        generation_inputs=end.generation_inputs)
    return PhaseEndOutcome(phase, winner, candidates, required_axes, label=label,
                           source_label=source_label, delivered_tests=tests)
