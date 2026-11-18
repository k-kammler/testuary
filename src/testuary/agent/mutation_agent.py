"""The mutation agent: after the coverage phase delivers a green test file,
it revises that file in place (``test_<stem>.py``, seeded with the coverage
suite) to kill surviving mutants, while the phase end keeps the inherited
bytes as a separate candidate.
Its agent mode (``needs_editing``, ``needs_testing``, ``needs_measurement``,
``needs_measurement_noted``)
gates the tools bound on each turn. A ``MutationAgent`` keeps its per-target
state on ``self.session`` and ``self.deps``, binds its tools per instance and
edits through the shared ``CodingAgent.edit_file`` gate. The entry point is
``generate_mutation_tests``, which returns a ``MutationTestResult``.
"""
from __future__ import annotations

import hashlib
import json
import time
from contextlib import nullcontext
from dataclasses import dataclass, field as _field
from pathlib import Path
from string import Template
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic.json_schema import SkipJsonSchema
from typing_extensions import Annotated, TypedDict

from testuary import languages, normal_run
from testuary.agent import _limits
from testuary.generation_evidence import source_sha256
from testuary import measurement
from testuary import config as config_module
from testuary import hints
from testuary import project
from testuary import prompts as prompts_module
from testuary.languages import mutation
from testuary.agent import _helpers as agent_helpers
from testuary.agent._base import CodingAgent, refuse_rewritten_inherited_suite, stop_status
from testuary.agent._helpers import (
    update_best_green,
    regression_revert_decision,
)


# Configuration

# ``test_mutation`` inherits keys absent from its own section from the
# ``test_coverage`` parent (max_loop_attempts, suite_run_timeout_seconds
# and others).
_API_CONFIG = config_module.PhaseConfiguration("api", api=True)
_AGENT_CONFIG = config_module.PhaseConfiguration("test_mutation", parent="test_coverage")
_AGENT_NAME = "test_mutation"


def mutation_measurement_identity(
    *, target_file_path: str, language: str, execution_name: str,
    config: Optional[Dict[str, Any]] = None,
    attribution: str = "first",
) -> Dict[str, Any]:
    """The identity of a reusable full-catalogue mutation matrix.
    ``attribution`` is how the run recorded killing tests, "first" (the loop's
    engine stops at the first killer) or "all" (the phase end records every
    killer), so the two never match. The configuration hashed is the
    settings as configured, the shipped engine default included, while a
    run's records name the engine that ran. Raises OSError when the target
    cannot be read."""
    cfg = dict(config or _AGENT_CONFIG)
    cfg.pop("log_fn", None)
    encoded = json.dumps(
        cfg, sort_keys=True, separators=(",", ":"), default=str,
    ).encode("utf-8")
    target_hash = project.sha256_project_file(target_file_path)
    if target_hash is None:
        raise OSError(
            f"cannot read target source for measurement identity: "
            f"{target_file_path}"
        )
    return {
        "target_source_sha256": target_hash,
        "language": language,
        "execution_name": execution_name,
        "config_sha256": hashlib.sha256(encoded).hexdigest(),
        "catalogue": "full",
        "attribution": attribution,
    }


# Logging

_log = agent_helpers.PhaseLogger("test_mutation")


def init_logging(project_name, log_dir, log_filename):
    """Start the mutation phase's streamed log ``log_filename`` in ``log_dir``
    for ``project_name``. Its configuration names the phase's model and its
    loop attempt limit."""
    _log._config = {"model": _API_CONFIG["model"], "max_loop_attempts": _AGENT_CONFIG["max_loop_attempts"]}
    _log.init(project_name, log_dir, log_filename)


log_print = _log.log
log_debug = _log.debug
save_logs = _log.save

#: The fraction of scored mutants hitting the per-mutant timeout from which
#: the score counts as timeout-driven (load-dependent, not reproducible) and
#: an alarm is logged: timeouts count as kills.
_MUTATION_TIMEOUT_ALARM_FRAC = 0.2


def _timeout_alarm_message(
    killed: int, survived: int, timeout: int, non_viable: int,
    rule: str, applied: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """The alarm when the timeout fraction of the scored mutants reaches
    ``_MUTATION_TIMEOUT_ALARM_FRAC`` (the score then follows the clock under
    load, not the assertions), else None. ``rule`` is the applied timeout as
    text, ``applied`` the engine's statement of it, which selects the advice."""
    scored = killed + survived + timeout + non_viable
    if scored <= 0 or timeout / scored < _MUTATION_TIMEOUT_ALARM_FRAC:
        return None
    return (
        f"[measure_mutation] [CRITICAL] {timeout}/{scored} mutants "
        f"({100.0 * timeout / scored:.0f}%) hit the per-mutant timeout "
        f"({rule}): the score is timeout-driven, not "
        f"assertion-driven, and is not reproducible. To change it, "
        f"{mutation.timeout_advice(applied)} Or reduce "
        f"engine_workers (contention)."
    )


def _requested_timeout_text(runner: Any, per_mutant_timeout: int) -> str:
    """What the request asks for as the per-mutant timeout, for the line
    logged before the run. A runner that does not say applies the derived timeout."""
    describe = getattr(runner, "requested_timeout_text", None)
    return describe(per_mutant_timeout) if callable(describe) else f"{per_mutant_timeout}s"


@dataclass(frozen=True)
class _EngineArgs:
    """The arguments of a measurement's engine runs: the workers of a run of
    its own, the per-mutant timeout, the wall clock and the suite's test
    files."""
    workers: int
    per_mutant_timeout: int
    wall_clock_timeout: int
    test_files: List[str]


# Result schema

class MutationTestResult(BaseModel):
    """Result returned by the mutation agent. The ``[auto]`` fields are
    filled when the phase ends (``finalize_result``, or
    ``generate_mutation_tests`` itself on a baseline stop), and the model
    leaves them null."""

    mutation_test_file_name: str = Field(
        description="Filename of the on-disk working file (e.g. test_encoder.py)."
    )
    explanation: str = Field(
        description="Brief explanation of which surviving mutants you targeted and how."
    )
    targeted_mutants: List[str] = Field(
        description="Ids of the surviving mutants the new tests target, as the survivor list shows them (the id after '--- mutant').",
    )
    mutation_score: Optional[float] = Field(default=None, description="[auto] Final mutation score %.")
    mutation_switched: Optional[bool] = Field(default=None, description="[auto] Whether the loop switched from the selective proxy to the full catalogue.")
    mutation_selective_final: Optional[float] = Field(default=None, description="[auto] Selective-proxy score at the switch (null if no switch).")
    mutation_full_at_switch: Optional[float] = Field(default=None, description="[auto] Full-catalogue score the moment the loop switched (null if no switch).")
    test_count_mutation: Optional[int] = Field(default=None, description="[auto] Test functions in the revised delivered suite.")
    test_loc_mutation: Optional[int] = Field(default=None, description="[auto] LOC in the revised delivered suite.")
    source_observations: SkipJsonSchema[Dict[str, Any]] = Field(default_factory=dict)
    agent_iterations: Optional[int] = Field(default=None, description="[auto] Total tool calls in this run.")
    agent_duration_s: Optional[float] = Field(default=None, description="[auto] Wall-clock seconds.")
    model_requests: SkipJsonSchema[Optional[int]] = Field(
        default=None,
        description="[auto] Total model requests made during this run.",
    )
    logical_requests: SkipJsonSchema[Optional[int]] = Field(
        default=None,
        description="[auto] Logical model requests: turns charged against the request limit. Equal to model_requests in a run without transport failures.",
    )
    unknown_usage_requests: SkipJsonSchema[Optional[int]] = Field(
        default=None,
        description="[auto] Physical requests whose token usage the provider did not report; the token totals undercount by their usage.",
    )
    input_tokens: Optional[int] = Field(default=None, description="[auto] Total prompt tokens consumed across model invocations.")
    output_tokens: Optional[int] = Field(default=None, description="[auto] Total completion tokens emitted across model invocations.")
    cache_hit_tokens: Optional[int] = Field(default=None, description="[auto] Cached-prefix prompt tokens (provider prompt-cache hits), a subset of input_tokens.")
    failure_kind: SkipJsonSchema[Optional[str]] = Field(
        default=None,
        description="[auto] infrastructure when a model request failure ended the run; otherwise unjudged.",
    )
    unknown_cache_requests: SkipJsonSchema[Optional[int]] = Field(
        default=None,
        description="[auto] Requests with reported token usage whose cache split the provider did not report; cache_hit_tokens sums the others.",
    )
    status: Optional[str] = Field(default=None, description="[auto] success | stagnation | budget_exhausted | agent_error | context_overflow | inconclusive | tests_failed | skipped.")
    error: Optional[str] = Field(default=None, description="[auto] Error message when status != success.")
    measured_suite_sha256: Optional[str] = Field(
        default=None, description="[auto] SHA-256 of the suite backing mutation_matrix."
    )
    measured_execution_name: Optional[str] = Field(
        default=None, description="[auto] Execution filename backing mutation_matrix."
    )
    mutation_matrix: Optional[dict] = Field(
        default=None, exclude=True, repr=False,
        description="[auto] Latest complete full-catalogue matrix for phase-end reuse.",
    )
    measurement_identity: Optional[dict] = Field(
        default=None, exclude=True, repr=False,
        description="[auto] Target and instrument identity backing mutation_matrix.",
    )


# Dependencies

class MutationTestDeps(BaseModel):
    """Per-target inputs the tools need at runtime."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_file_path: str = Field(description="Absolute path to the source file under test.")
    target_file_content: str = Field(description="Source code of the file under test.")
    source_dir: str = Field(description="Absolute path to the source directory.")
    test_output_dir: str = Field(description="Directory holding the rank's test files, including the writable mutation file.")
    language: str = Field(description="Programming language (python, java, ...).")
    mutation_test_file_name: str = Field(description="Name of the writable mutation test file.")
    coverage_baseline_pct: Optional[float] = Field(default=None, description="Coverage % from the coverage phase (prompt context).")

    _coerce_paths = model_validator(mode="before")(
        agent_helpers.coerce_paths_to_str
    )


# Prompts

# The model sees the language only in the prompt text, which the entry point
# resolves for the run's language.


# Session state

@dataclass
class TestMutationSession(agent_helpers.CodingAgentSession):
    """The mutation agent's per-target session state."""
    # Best-so-far.
    best_mutation_score: Optional[float] = None
    prev_mutation_score: Optional[float] = None
    mutation_post_goal_count: int = 0
    seen_mutation_goal: bool = False
    # Two catalogues (full_after_selective_goal): the loop runs on the fast
    # selective operator set and switches to the full catalogue when the
    # selective score reaches the switch threshold or plateaus. ``mutation_mode``
    # is "selective" until then and "full" after. The switch resets
    # best_mutation_score, so comparisons stay full against full, and while a
    # switch is possible a selective plateau switches instead of ending the phase.
    mutation_mode: str = "selective"
    selective_final_score: Optional[float] = None
    full_score_at_switch: Optional[float] = None
    switch_iteration: Optional[int] = None
    # None until the first selective measurement. True while a switch to the
    # full catalogue can happen. False once switched, with the switch off, or
    # when the engine's full catalogue cannot be read.
    selective_switch_possible: Optional[bool] = None
    #: Structured copies of the latest measurement, for the entry
    #: point's baseline handoff into the user prompt.
    last_mutation_counts: Optional[Tuple[int, int]] = None
    last_surviving_sample: str = ""
    # Consecutive green regressions (score strictly below best) since the
    # last new best or the last in-loop revert. Drives restore_count.
    mutation_regression_count: int = 0
    # ``measure_mutation_calls``: every call the model made, refused and cached
    # included. ``mutation_measurements``: completed cycles the model started
    # (the validator's measurement gate). ``engine_runs``: completed engine
    # runs, framework preparation and the switch's run included.
    # ``framework_prep_runs``: preparation cycles (0 for a derived baseline).
    # ``cached_measure_returns``: calls answered from the cache.
    mutation_measurements: int = 0
    measure_mutation_calls: int = 0
    engine_runs: int = 0
    framework_prep_runs: int = 0
    cached_measure_returns: int = 0
    # Cache flags and the last results.
    tests_changed_since_mutation: bool = True
    last_mutation_result: str = ""
    last_complete_matrix: Optional[dict] = None
    last_complete_matrix_hash: Optional[str] = None
    last_complete_execution_name: Optional[str] = None
    last_complete_measurement_identity: Optional[dict] = None
    # Engine returned no useful outcomes (every mutant INCOMPETENT /
    # untested). The 0% score is a failure mode, not a measurement.
    mutation_inconclusive: bool = False
    # Loop and no-progress guard, with the mutation wording.
    loop_breaker: agent_helpers.LoopBreakerState = _field(default_factory=lambda: agent_helpers.LoopBreakerState(
        return_action_text="return MutationTestResult if the mutation-phase exit conditions are met"))


# The mutation engine session the orchestrator installs: ``measure_mutation``
# reuses its pooled workers instead of starting them per call (about 20 s at
# 48 workers). Module-level, so it outlives each target's ``MutationAgent``.
_mutation_session: Optional["mutation.MutationSession"] = None


def set_mutation_session(
    session: Optional["mutation.MutationSession"],
) -> None:
    """Install or clear the shared mutation engine session. Called by
    the orchestrator at the start and end of each mutation phase."""
    global _mutation_session
    _mutation_session = session


# Helpers without session state

def _canonical_mutation_summary_line(
    results: "mutation.MutationMeasurement",
    elapsed_s: float,
) -> str:
    """The canonical one-line mutation result, in the shape of the tests'
    ``TESTS: ...`` line::

        MUTATION: <STATUS> | mutants T | killed K | survived S | timeout TO |
                  suspicious SP | untested U | score X.X% | elapsed Es

    with ``<STATUS>`` one of SUCCESS, INCONCLUSIVE, TRUNCATED, NO_MUTANTS.
    """
    total_attempted = (
        results.killed + results.survived + results.timeout
        + results.non_viable + results.untested
    )
    useful = results.killed + results.survived
    if total_attempted == 0:
        status = "NO_MUTANTS"
    elif results.truncated:
        status = "TRUNCATED"
    elif useful == 0:
        status = "INCONCLUSIVE"
    else:
        status = "SUCCESS"
    score_str = (
        f"{measurement.floor_pct(results.score):.1f}%"
        if results.score is not None else "n/a"
    )
    return (
        f"MUTATION: {status} | "
        f"mutants {total_attempted} | "
        f"killed {results.killed} | survived {results.survived} | "
        f"timeout {results.timeout} | "
        f"suspicious {results.non_viable} | "
        f"untested {results.untested} | "
        f"score {score_str} | elapsed {elapsed_s:.1f}s"
    )


def _killed_count(results: "mutation.MutationMeasurement") -> int:
    """The mutants the score counts as killed: those a failing test or a
    crash killed, those the per-mutant clock ended and the invalid ones,
    as the engine's score (cr-rate) counts them."""
    return results.killed + results.timeout + results.non_viable


def _mutation_summary_line(results: "mutation.MutationMeasurement") -> str:
    """One-line summary of a mutation run for the model's tool response.
    The killed count is the score's numerator and the scored mutants its
    denominator, so the two give the score shown. A partial run's
    denominator also counts its untested mutants."""
    scored = results.total + (results.untested if getattr(results, "lower_bound", False) else 0)
    killed = f"killed {_killed_count(results)}/{scored}"
    if results.timeout:
        killed += f", {results.timeout} of them by timeout"
    if results.non_viable:
        killed += f", {results.non_viable} of them invalid"
    bits = [killed, f"survived {results.survived}"]
    if results.untested:
        bits.append(f"untested {results.untested}")
    # A cut run with untested mutants says so in its own sentence.
    suffix = (" [TRUNCATED -- partial results]"
              if results.truncated and not results.untested else "")
    return (f"Mutation score: {measurement.floor_pct(results.score):.1f}% "
            f"({'; '.join(bits)}){suffix}")


def _matrix_measurement(matrix: Dict[str, Any], operators: Optional[set],
                        score: float) -> SimpleNamespace:
    """A stored matrix's counts in the shape ``_mutation_summary_line``
    reads, counted by the live reading's own ``summary_counts``, so a
    derived baseline is summarized in the tool's own words."""
    counts = mutation.summary_counts(
        mutant for mutant in matrix.get("mutants") or []
        if operators is None or mutant.get("operator") in operators)
    return SimpleNamespace(
        truncated=False, score=score, lower_bound=bool(counts["untested"]),
        total=(counts["killed"] + counts["survived"] + counts["timeout"]
               + counts["non_viable"]),
        **counts)


def _shown_delta(score: float, prev_best: float) -> float:
    """The change from the best to ``score`` as the two are shown (floored
    to one decimal), so the shown best plus the shown change gives the
    shown score."""
    return round(measurement.floor_pct(score) - measurement.floor_pct(prev_best), 1)


def _catalogue_switch_banner(selective_final: float) -> str:
    """The model-facing note before the first full-catalogue score after the
    switch from the selective set, for a measured and a derived baseline."""
    return (
        "[CATALOGUE SWITCH] The selective operator set is maxed "
        f"({measurement.floor_pct(selective_final):.1f}%) and "
        "locked in. Scoring has switched to the FULL catalogue: that is "
        "now the reported metric and the goal. The score and surviving "
        "mutants below are full-catalogue -- keep adding tests to kill "
        "them.\n\n"
    )


def _selective_approximation_banner(threshold: float) -> str:
    """The model-facing note before a selective score while a switch to the
    full catalogue is pending, for a measured and a derived baseline."""
    return (
        "[SELECTIVE APPROXIMATION] This score is the fast selective "
        "operator subset, an approximation of the full catalogue. Once "
        f"it reaches {threshold:.0f}% (or plateaus) scoring "
        "switches to the full catalogue and that becomes the goal, so "
        "keep killing these survivors to push toward the switch.\n\n"
    )


# Graph state

class MutationGraphState(TypedDict, total=False):
    """The mutation loop's graph state: the conversation's messages, the
    phase's result once it ends and the agent's mode."""
    messages: Annotated[List[BaseMessage], add_messages]
    result: Optional[MutationTestResult]
    agent_mode: str  # "needs_editing" | "needs_testing" | "needs_measurement[_noted]"


# Agent

#: The placeholder phrase a lazy model echoes back instead of writing a real
#: explanation (the ``MutationTestResult.explanation`` field description).
_MUTATION_PLACEHOLDER_FRAGMENTS = (
    "brief explanation of which surviving mutants you targeted and how",
)

_MUTANT_ID_PLACEHOLDER_FRAGMENTS = (
    "ids of the surviving mutants the new tests target",
    "as the survivor list shows them",
    "file:line:operator",
)


def _looks_placeholder_mutant(text: str) -> bool:
    """Whether a ``targeted_mutants`` item echoes the schema. An id may be
    short, so no length check applies."""
    return agent_helpers.looks_placeholder_text(
        text, _MUTANT_ID_PLACEHOLDER_FRAGMENTS, min_len=0,
    )


class MutationAgent(CodingAgent):
    """The mutation agent, which adds tests to the working file, seeded with
    the inherited suite, to kill surviving mutants. Its tools: the two reads,
    ``edit_file`` (offered as ``edit_tests``), ``run_tests`` and
    ``measure_mutation``."""

    read_tool_names = ("read_source", "read_tests")
    guards_duplicate_test_names = True
    guards_generated_test_validity = True
    edit_tool_name = "edit_tests"

    result_schema = MutationTestResult
    graph_state_type = MutationGraphState
    final_tool_name = "emit_mutation_test_result"
    forces_initial_read = True
    forces_initial_measurement = True
    designated_read_name = "read_tests"
    measure_tool_name = "measure_mutation"
    no_measurement_key = "validator_no_mutation_measurement"
    stale_measurement_key = "validator_stale_mutation_measurement"
    initial_agent_mode = "needs_initial_read"

    def __init__(
        self,
        prompts: Dict[str, Any],
        deps: MutationTestDeps,
        session: TestMutationSession,
        logger: Any = None,
        mutation_session: Optional["mutation.MutationSession"] = None,
    ) -> None:
        super().__init__(prompts, deps, session, logger)
        self.mutation_session = mutation_session

    def _tool_execution_context(self):
        worker = getattr(self.mutation_session, "worker", None)
        return worker() if callable(worker) else nullcontext()

    # Hooks for the shared edit gate.

    def target_path(self) -> Path:
        return Path(self.deps.test_output_dir) / self.deps.mutation_test_file_name

    def _capture_complete_matrix(self, measurement: Any,
                                 source_hash: str) -> None:
        """Keep the full-catalogue matrix of this measurement of these bytes
        without another run. The phase end reuses it only while the final bytes
        hash and execution name still match. A run without a matrix leaves
        the fields empty.
        """
        if self.session.mutation_mode != "full":
            return
        out = Path(self.deps.test_output_dir) / ".mutation_latest_matrix.json"
        try:
            out.unlink(missing_ok=True)
            mutation.write_kill_matrix(measurement, out, self.log)
            matrix = json.loads(out.read_text(encoding="utf-8"))
            if not isinstance(matrix, dict) or not matrix.get("mutants"):
                return
            self.session.last_complete_matrix = matrix
            self.session.last_complete_matrix_hash = source_hash
            self.session.last_complete_execution_name = self.target_path().name
            self.session.last_complete_measurement_identity = (
                mutation_measurement_identity(
                    target_file_path=self.deps.target_file_path,
                    language=self.deps.language,
                    execution_name=self.target_path().name,
                )
            )
        except (OSError, ValueError, NotImplementedError, AttributeError) as exc:
            self.log(
                f"[mutation] latest matrix capture unavailable "
                f"({type(exc).__name__}: {exc}). The phase end will measure the "
                "candidate once.",
                level=1,
            )
        finally:
            out.unlink(missing_ok=True)

    loop_state_fields = (("mode", "mutation_mode"),) + tuple((name, name) for name in (
        "measured_suite_hash", "prev_mutation_score", "last_mutation_result",
        "last_surviving_sample", "last_mutation_counts", "last_suite_exec_ms",
        "last_complete_matrix",
        "last_complete_matrix_hash", "last_complete_execution_name",
        "last_complete_measurement_identity"))
    loop_text_fields = ("last_mutation_result", "last_surviving_sample")
    swallows_restore_errors = True

    def live_scores(self, measured: bool) -> Dict[str, Any]:
        return {
            "mutation": self.session.prev_mutation_score if measured else None,
            "coverage": getattr(self.deps, "coverage_baseline_pct", None),
        }

    def _snapshot_current(self, state: Dict[str, Any]) -> bool:
        # A measurement on the selective catalogue before the switch does not hold.
        return state.get("mode") == self.session.mutation_mode

    def _restore_unmeasured(self, snap: Dict[str, Any], state: Dict[str, Any]) -> None:
        # The score recorded with the snapshot's bytes on the current
        # catalogue, None otherwise.
        self.session.prev_mutation_score = (
            snap.get("mutation")
            if state.get("mode", self.session.mutation_mode) == self.session.mutation_mode
            else None)

    def _synthesize_mutation_result(self, reason: str) -> MutationTestResult:
        """The result from session state when a designed end fires."""
        deps = self.deps
        session = self.session
        parts = [f"Stopped early ({reason})."]
        if session.best_mutation_score is not None:
            parts.append(f"Final mutation score: {measurement.floor_pct(session.best_mutation_score):.1f}%.")
        parts.append(
            f"{session.writes_done} writes, {session.mutation_measurements} mutation measurements."
        )
        if session.mutation_inconclusive:
            status = "inconclusive"
            parts.append(
                "Inconclusive: engine returned with 0 useful outcomes. "
                "The mutation_score is a placeholder, not a measurement."
            )
        else:
            status = stop_status(reason, measured=self._best_green_measured("mutation"))
        return MutationTestResult(
            mutation_test_file_name=deps.mutation_test_file_name,
            explanation=" ".join(parts),
            targeted_mutants=[],
            mutation_score=session.best_mutation_score,
            mutation_switched=session.switch_iteration is not None,
            mutation_selective_final=session.selective_final_score,
            mutation_full_at_switch=session.full_score_at_switch,
            status=status,
            agent_iterations=session.session_tool_calls,
        )

    def _maybe_set_mutation_stop(self, *, source_tool: str) -> Optional[str]:
        """The directive of a designed end, with a result on
        ``session.early_stop_result``, or None. In priority order, each only
        while no switch to the full catalogue is pending (the goal is judged on
        the full catalogue, and a selective plateau switches instead of ending
        the phase): the best score at the goal with ``extra_measurements_after_goal`` used
        up. The streak at ``max_measurements_without_improvement`` below the goal with
        ``allow_early_stop``. At the goal while the latest green run
        carries a STANDS ALONE note, the end waits: the reply says so, and the
        wait spends no post-goal measurement.
        """
        session = self.session
        mut_goal = float(_AGENT_CONFIG["goal"])
        extra_measurements_after_goal = int(_AGENT_CONFIG["extra_measurements_after_goal"])
        max_measurements_without_improvement = int(_AGENT_CONFIG["max_measurements_without_improvement"])
        early_stop = bool(_AGENT_CONFIG["allow_early_stop"])
        switch_pending = bool(session.selective_switch_possible)

        def _set_stop(reason: str) -> str:
            return self.designed_stop(self._synthesize_mutation_result(reason), reason)

        if (not switch_pending
                and session.best_mutation_score is not None
                and session.best_mutation_score >= mut_goal):
            if session.stands_alone_noted:
                self.log(
                    f"[{source_tool}] goal reached "
                    f"({measurement.floor_pct(session.best_mutation_score):.1f}% >= "
                    f"{mut_goal:g}%), but the latest green run carries a STANDS ALONE "
                    "note: the end waits.",
                    level=1,
                )
                return self.goal_waits_reply()
            if not session.seen_mutation_goal:
                session.seen_mutation_goal = True
                if extra_measurements_after_goal == 0:
                    self.log(
                        f"[{source_tool}] FRAMEWORK STOP: goal reached "
                        f"({measurement.floor_pct(session.best_mutation_score):.1f}% >= "
                        f"{mut_goal:g}%) and extra_measurements_after_goal=0.",
                        level=1,
                    )
                    return _set_stop("mutation_goal_met")
                self.log(
                    f"[{source_tool}] goal reached "
                    f"({measurement.floor_pct(session.best_mutation_score):.1f}% >= {mut_goal:g}%). "
                    f"Polish limit: {extra_measurements_after_goal} more measurements.",
                    level=1,
                )
            else:
                session.mutation_post_goal_count += 1
                if session.mutation_post_goal_count >= extra_measurements_after_goal:
                    self.log(
                        f"[{source_tool}] FRAMEWORK STOP: post-goal polish "
                        f"({session.mutation_post_goal_count}/{extra_measurements_after_goal}) exhausted.",
                        level=1,
                    )
                    return _set_stop("mutation_goal_met")

        below_goal = (
            session.best_mutation_score is None or session.best_mutation_score < mut_goal
        )
        if (
            early_stop
            and max_measurements_without_improvement > 0
            and below_goal
            and session.measurements_without_improvement >= max_measurements_without_improvement
            and not switch_pending
        ):
            self.log(
                f"[{source_tool}] FRAMEWORK STOP: mutation stagnation "
                f"({session.measurements_without_improvement}/{max_measurements_without_improvement}) "
                f"below goal -> handing off.",
                level=1,
            )
            return _set_stop("mutation_stagnation")

        return None

    def _maybe_revert_on_regression(
        self, score: float, prev_best: Optional[float],
    ) -> Optional[str]:
        """The in-loop regression revert: a green ``score`` strictly below the
        best is a regression (an equal one is lateral), and after
        ``restore_count`` of them the working file returns to the best-green
        snapshot (``restore_count <= 0`` turns this off, while the terminal
        restore still runs). Returns the model-facing note of a revert, else None.
        """
        session = self.session
        is_regression = prev_best is not None and score < prev_best
        should, session.mutation_regression_count = regression_revert_decision(
            regression_count=session.mutation_regression_count,
            is_regression=is_regression,
            n_restore=int(_AGENT_CONFIG["restore_count"]),
        )
        if not should:
            return None
        snap = session.best_green.snapshot
        if not self._restore_snapshot(self.target_path()):
            return None
        self.log(
            f"[restore] mutation file reverted to best-green snapshot "
            f"(score {snap.get('mutation')}) after a regression below best "
            f"{measurement.floor_pct(prev_best):.1f}%.",
            level=1,
        )
        return self.tool_msg(
            "revert_on_regression",
            current=f"{measurement.floor_pct(score):.1f}",
            best=f"{measurement.floor_pct(prev_best):.1f}",
            test_count=str(snap.get("test_count")),
        )

    def _timeout_before_edits(self, timeout: int) -> Optional[str]:
        # With no edit yet the inherited suite is too slow as it arrived: the
        # phase ends.
        if self.session.writes_done:
            return None
        self.session.early_stop_result = self._synthesize_mutation_result(
            "inherited_suite_too_slow",
        )
        self.session.early_stop_reason = "inherited_suite_too_slow"
        return self.tool_msg("tests_timeout_inherited", timeout=timeout)

    def synthesize_stop_result(self, reason: str) -> Any:
        return self._synthesize_mutation_result(reason)

    # Tools: reads.

    def read_source(self, start_line: int = 1, end_line: Optional[int] = None) -> str:
        """Read the source of the file under test (line-numbered).

        This tool becomes available after the mandatory initial ``read_tests``
        call returns. Use it when survivor feedback does not contain enough
        source context for a targeted edit, or to refresh after a context
        trim. Pass ``start_line`` / ``end_line`` to page a file past the size
        limit. An immediate consecutive whole re-read (no other tool call
        between) returns a short cached header; calling any other tool resets
        the counter.
        """
        return self._source_read(start_line, end_line)

    # Tools: writes.

    # ``edit_file`` is the shared CodingAgent gate, driven by the hooks above.

    # Tools: measurement.

    # run_tests hooks: the run names the working file, which holds the
    # inherited suite edited in place.

    def test_paths(self) -> List[str]:
        return [str(self.target_path())]

    restored_file_phrase = "the mutation file"

    def snapshot_score_text(self, snap: Dict[str, Any]) -> str:
        return f"score {snap.get('mutation')}"

    def _record_red_normal_run(self, result: str) -> None:
        """A refused normal run leaves the suite red, as a red run_tests does:
        the run of the current file is recorded, so the loop returns to
        editing and a measurement of the unchanged file is refused."""
        session = self.session
        session.last_tests_passed = False
        session.last_run_tests_result = result
        session.tests_changed_since_run_tests = False

    def measure_mutation(self) -> str:
        """Run the mutation engine against the target source. The working
        file is the one collectable suite, so its tests are the
        kill-witnesses.

        Returns mutation_score % plus killed / survived / timeout /
        suspicious counts, plus a sample of surviving-mutant diffs when
        the engine exposes them.

        Stop semantics:
          - goal reached + post-goal patience exhausted -> framework
            synthesises MutationTestResult and ends the mutation phase
          - stagnation below goal (streak hits patience) +
            early_stop_on_stagnation -> same
        """
        self.log_thinking("measure_mutation")
        session = self.session
        session.measure_mutation_calls += 1
        if self.in_start_measurement():
            return self.track_output(self._initial_measurement())

        # Refused while the tests are known red and the file is unchanged.
        if (
            not session.last_tests_passed
            and not session.tests_changed_since_run_tests
            and session.last_run_tests_result
        ):
            self.log(
                "[measure_mutation] REFUSED: tests known red, mutation file "
                "unchanged since last run_tests. Edit required.",
                level=1,
            )
            return self.track_output(self.tool_msg(
                ("measure_mutation_refused_known_refusal" if self.refused_by()
                 else "measure_mutation_refused_known_red"),
                last_run_tests_result=session.last_run_tests_result,
                refused_by=self.refused_by_phrase(),
            ))

        if not session.tests_changed_since_mutation and session.last_mutation_result:
            session.cached_measure_returns += 1
            self.log(
                "[measure_mutation] Skipped: tests unchanged since last call.",
                level=1,
            )
            # A cached call can optionally count as a no-improvement event.
            self.count_cached_measurement(session.best_mutation_score is not None, self.log)
            # The stop check still runs after the counter advanced: the last
            # real measurement may have reached an end.
            stop_msg = self._maybe_set_mutation_stop(
                source_tool="measure_mutation[cached]",
            )
            cached_key = (
                "measure_mutation_cached_rejected"
                if session.best_mutation_score is not None
                else "measure_mutation_cached_red"
            )
            cached = agent_helpers.loop_breaker_check(
                session.loop_breaker, "measure_mutation",
                self.tool_msg(
                    cached_key,
                    last_result=session.last_mutation_result,
                ),
            )
            return self.track_output(
                cached + "\n\n" + stop_msg if stop_msg is not None else cached
            )

        return self.track_output(self._measure_mutation_engine())

    def _initial_measurement(self) -> str:
        """The start's measurement, the model's first measure_mutation call
        after the designated read: the stored measurement of the exact suite
        (the entry's baseline measurement or its derive, matched by the
        bytes' hash) when one exists, else a measurement of the suite now,
        as framework preparation. Either answer is the summary the tool
        gives for that suite. The loop then enters Revise."""
        session = self.session
        session.initial_measured = True
        path = self.target_path()
        digest = (source_sha256(path.read_text(encoding="utf-8"))
                  if path.exists() else None)
        if (digest is not None and session.measured_suite_hash == digest
                and session.last_mutation_result):
            self.log("[measure_mutation] start: the stored measurement of the "
                     "suite answers (no engine run)", level=1)
            return session.last_mutation_result
        self.log("[measure_mutation] start: no stored measurement of these "
                 "bytes, measuring the suite", level=1)
        session.framework_prep_runs += 1
        return self._measure_mutation_engine(framework_prep=True)

    def framework_baseline_measure(self) -> None:
        """Measure the inherited suite as framework preparation: the tool's
        measurement core, counted as phase cost (``framework_prep_runs`` and the
        preparation clock), never as model activity. The entry point calls it
        when no coverage matrix certifies the inherited suite."""
        self.session.framework_prep_runs += 1
        self._measure_mutation_engine(framework_prep=True)

    def _measure_mutation_engine(self, framework_prep: bool = False) -> str:
        """Run the suite normally, then the engine, update the session and return
        the summary. The tool passes it through ``track_output``, framework
        preparation calls it directly (``mutation_measurements`` counts only
        the model's cycles)."""
        deps = self.deps
        session = self.session
        log_prefix = "baseline_measure" if framework_prep else "measure_mutation"

        # The normal run decides whether the suite is green.
        # Mutating against a red suite reports every mutant as "killed" by
        # something already broken.
        suite_run_timeout_seconds = int(_AGENT_CONFIG["suite_run_timeout_seconds"])
        backend = languages.get_backend(deps.language)
        normal, reply = self._normal_run(backend, suite_run_timeout_seconds, log_prefix)
        if reply is not None:
            return reply
        reply = self._coverage_run_reply(backend, suite_run_timeout_seconds, log_prefix)
        if reply is not None:
            return reply

        engine_args = self._engine_args(normal.elapsed_s, backend)
        engine_start = time.perf_counter()
        try:
            runner = self._engine_runner(engine_args, log_prefix)
            results = self._run_engine(
                runner, engine_args,
                runner.full_operators() if session.mutation_mode == "full" else None)
        except NotImplementedError as exc:
            return self.tool_msg("measure_mutation_engine_not_implemented", exc=str(exc))

        engine_elapsed = time.perf_counter() - engine_start
        session.engine_runs += 1
        self.log(_canonical_mutation_summary_line(results, engine_elapsed), level=1)
        # The raw outcomes and run parameters, in the saved log only. The console
        # keeps the summary line.
        applied = results.timeout_rule() if hasattr(results, "timeout_rule") else None
        applied_text = (mutation.timeout_rule_text(applied)
                        or _requested_timeout_text(runner, engine_args.per_mutant_timeout))
        self.debug(
            f"[{log_prefix}] raw outcomes: killed={results.killed} "
            f"survived={results.survived} timeout={results.timeout} "
            f"suspicious={results.non_viable} untested={results.untested}; "
            f"mode={session.mutation_mode} workers={engine_args.workers} "
            f"per_mutant_timeout={applied_text} "
            f"elapsed={engine_elapsed:.1f}s", level=2,
        )
        # Timeouts count as kills but depend on load: a large share makes the
        # score the clock's, so it is logged as an alarm.
        alarm = _timeout_alarm_message(
            results.killed, results.survived, results.timeout,
            results.non_viable, applied_text, applied,
        )
        if alarm:
            self.log(alarm, level=1)

        unusable = self._unusable_outcome(results, log_prefix, applied_text=applied_text)
        if unusable is not None:
            return unusable

        session.last_mutation_counts = (_killed_count(results), results.survived)
        switch_threshold = float(
            _AGENT_CONFIG["selective_switch_threshold"])
        results, just_switched, reply = self._maybe_switch_catalogue(
            runner, engine_args, results, switch_threshold, log_prefix, framework_prep)
        if reply is not None:
            return reply
        score = results.score
        prev_best = self._record_mutation_score(score, framework_prep)

        session.tests_changed_since_mutation = False

        summary = self._mutation_reply(results, score, prev_best, engine_args, just_switched,
                                       switch_threshold)

        self._snapshot_measured_suite(results, score, just_switched)

        # The in-loop regression revert, after _snapshot_measured_suite so the
        # best-green snapshot is current. Its note tells the model the file
        # changed.
        revert_note = self._maybe_revert_on_regression(score, prev_best)
        if revert_note:
            summary += "\n\n" + revert_note

        stop_msg = self._maybe_set_mutation_stop(source_tool=log_prefix)
        if stop_msg is not None:
            return summary + "\n\n" + stop_msg
        return summary

    def _red_when_measured_message(self, results: Any, log_prefix: str) -> str:
        """The message for a suite the normal run passed (the measurement
        runs only then) but that failed in the measuring run, so no mutant
        was tested: the mutation tool's refusal (``_tool_refusal``)."""
        stopping = mutation.test_texts(getattr(results, "stopping_tests", ()) or ())
        msg = (self.tool_msg("measure_mutation_red_when_measured_named",
                             tests="\n".join(f"- {test}" for test in stopping))
               if stopping else self.tool_msg("measure_mutation_red_when_measured"))
        return self._tool_refusal(
            "mutation", msg, list(stopping), f"[{log_prefix}] 0 mutants tested: the suite "
            f"failed while measured after a green normal run: {getattr(results, 'reason', '')}")

    def _mutation_reply(self, results: Any, score: float, prev_best: Optional[float],
                        engine_args: _EngineArgs, just_switched: bool,
                        switch_threshold: float) -> str:
        """The measurement's reply, kept as the last result: the summary with
        the change against the previous best, the time-limit note, the goal
        and the streak, the switch's or the selective approximation's
        banner, a sample of surviving mutants and the stall hint. Logs the
        time-limit warning and the progress line."""
        session = self.session
        delta_str = ""
        if prev_best is not None:
            delta_str = (
                f" (prev best: {measurement.floor_pct(prev_best):.1f}%, "
                f"delta: {_shown_delta(score, prev_best):+.1f}pp)"
            )

        mut_goal = float(_AGENT_CONFIG["goal"])
        max_measurements_without_improvement = int(_AGENT_CONFIG["max_measurements_without_improvement"])
        summary_head = _mutation_summary_line(results)
        if results.truncated:
            from testuary.results import note

            self.log("WARNING: " + note(
                "mutation", "mutation.measurement_timeout_seconds",
                engine_args.wall_clock_timeout))
        if results.truncated and results.untested:
            delta_str += "\n" + self.tool_msg(
                "measure_mutation_time_limit", limit=engine_args.wall_clock_timeout,
                untested=results.untested)
        summary = (
            f"{summary_head}{delta_str}\n"
            f"Goal: {mut_goal:g}%. "
            f"Stagnation: {session.measurements_without_improvement}/{max_measurements_without_improvement}."
        )
        if just_switched:
            summary = (
                _catalogue_switch_banner(session.selective_final_score)
                + summary
            )
        elif session.mutation_mode == "selective" and session.selective_switch_possible:
            summary = _selective_approximation_banner(switch_threshold) + summary

        self.log_progress(
            session.mutation_measurements,
            f"score {measurement.floor_pct(score):.1f}%",
            f"{measurement.floor_pct(session.best_mutation_score):.1f}%",
            target=f"goal {mut_goal:g}%",
            delta=(
                f"{_shown_delta(score, prev_best):+.1f}pp vs prev"
                if prev_best is not None else ""
            ),
        )

        surviving_diffs = results.survivors_text(
            max_mutants=_limits.value("survivor_diffs"),
            max_chars=_limits.value("survivor_diff_chars"),
        )
        if surviving_diffs:
            summary = f"{summary}\n\n--- Surviving mutants (sample) ---\n{surviving_diffs}"
        session.last_surviving_sample = surviving_diffs or ""
        hint = self._mutation_stall_hint()
        if hint:
            summary = f"{summary}\n\n{hint}"
        session.last_mutation_result = summary
        return summary

    def _snapshot_measured_suite(self, results: Any, score: float, just_switched: bool) -> None:
        """Record the measured suite's hash, keep the run's complete matrix and
        update the best-green snapshot with this score. A failure is logged
        and does not end the measurement."""
        session = self.session
        # The best-green snapshot of the working file, with the coverage
        # phase's coverage for the log.
        try:
            mut_path = self.target_path()
            if mut_path.exists():
                mtext = mut_path.read_text(encoding="utf-8")
                session.measured_suite_hash = source_sha256(mtext)
                self._capture_complete_matrix(
                    results, session.measured_suite_hash,
                )
                structural = measurement.record(measurement.observe_file(
                    mut_path, backend=self.test_backend(),
                ))
                inherited_cov = getattr(self.deps, "coverage_baseline_pct", None)
                if just_switched:
                    # Best-green restarts on the full catalogue too: a selective
                    # score never outranks a full one.
                    session.best_green.snapshot = None
                update_best_green(
                    session.best_green, mut_path,
                    coverage=inherited_cov, mutation=score, test_count=structural["tests"],
                    test_source_observation=structural,
                    exec_time_ms=session.last_suite_exec_ms,
                    persist_to_disk=True, log_fn=self.log,
                    loop_state=self._snapshot_loop_state(mtext),
                )
        except Exception as exc:
            self.log(
                f"[snapshot] update_best_green failed silently: "
                f"{type(exc).__name__}: {exc}",
                level=2,
            )

    def _maybe_switch_catalogue(self, runner: Any, engine_args: _EngineArgs, results: Any,
                                switch_threshold: float, log_prefix: str,
                                framework_prep: bool) -> Tuple[Any, bool, Optional[str]]:
        """The catalogue switch: once the selective score reaches
        ``switch_threshold`` or plateaus, with full_after_selective_goal on,
        the same suite is measured again on the full catalogue, which is kept
        from here, and the best is cleared, so the score's record sets it to
        the full score with the streak reset (a new metric, not a
        regression). The model is told. Returns the results to record,
        whether the switch happened, and the reply that ends the call when
        the full catalogue's run is unusable."""
        session = self.session
        score = results.score
        selective_active = (
            bool(_AGENT_CONFIG["full_after_selective_goal"])
            and session.mutation_mode == "selective"
            and session.switch_iteration is None
        )
        # A switch needs a readable full catalogue (full_operators() not None),
        # resolved once, so the stop check knows a selective plateau switches.
        if selective_active and session.selective_switch_possible is None:
            session.selective_switch_possible = runner.full_operators() is not None
        if not selective_active:
            session.selective_switch_possible = False

        # The plateau counts the measurement just taken, as its record will.
        switch_after_measurements = int(_AGENT_CONFIG["max_measurements_without_improvement"])
        if not (selective_active and session.selective_switch_possible and (
            score >= switch_threshold
            or (switch_after_measurements > 0 and self._streak_after(score) >= switch_after_measurements)
        )):
            return results, False, None
        switch_reason = (
            "selective goal reached" if score >= switch_threshold
            else "selective proxy plateaued"
        )
        results = self._run_engine(runner, engine_args, runner.full_operators())
        session.engine_runs += 1
        # The full-catalogue run answers to the same outcome gates as
        # the selective one: no switch state is written from a run
        # that produced nothing usable.
        unusable = self._unusable_outcome(results, log_prefix, after_switch=True)
        if unusable is not None:
            return results, False, unusable
        session.selective_final_score = score
        score = results.score
        session.full_score_at_switch = score
        session.last_mutation_counts = (_killed_count(results), results.survived)
        session.mutation_mode = "full"
        # The switch's measurement index: 0 for the framework baseline,
        # else the model-triggered cycle that is completing now.
        session.switch_iteration = (
            0 if framework_prep else session.mutation_measurements + 1
        )
        session.best_mutation_score = None
        session.seen_mutation_goal = False
        session.mutation_post_goal_count = 0
        session.mutation_regression_count = 0
        session.selective_switch_possible = False
        self.log(
            f"[{log_prefix}] {switch_reason} "
            f"({measurement.floor_pct(session.selective_final_score):.1f}%). "
            f"Switched to the full catalogue, full score "
            f"{measurement.floor_pct(score):.1f}%.", level=1)
        return results, True, None

    def _record_mutation_score(self, score: float, framework_prep: bool) -> Optional[float]:
        """Count the measurement, keep ``score`` as the previous one and as
        the best when it is the first or higher, and reset or grow the
        streak of measurements without improvement (``_resets_streak``),
        before the stop check. Returns the best before this measurement. The
        framework baseline is preparation: the validator's gate stays closed
        until the model measures."""
        session = self.session
        if not framework_prep:
            session.mutation_measurements += 1
        session.prev_mutation_score = score
        prev_best = session.best_mutation_score
        session.measurements_without_improvement = self._streak_after(score)
        if prev_best is None or score > prev_best:
            session.best_mutation_score = score
        return prev_best

    def _streak_after(self, score: float) -> int:
        """The streak of measurements without improvement that a measurement
        scoring ``score`` leaves (``_resets_streak``), the best kept so far
        being the higher of the two."""
        session = self.session
        best = session.best_mutation_score
        if not self._resets_streak(best, score if best is None or score > best else best,
                                   min_points=float(_AGENT_CONFIG["min_improvement_points"]),
                                   minimum=float(_AGENT_CONFIG["minimum"])):
            return session.measurements_without_improvement + 1
        return 0

    def _normal_run(self, backend: Any, suite_run_timeout_seconds: int,
                   log_prefix: str) -> Tuple[Any, Optional[str]]:
        """The fail-fast normal run of the working file and the reply that
        ends the call, None when the run is green: its timeout, a red run
        (before any write the inherited suite is red and the phase ends), or
        a green run that leaked unsound asynchronous warnings. A green run
        clears the timeout streak and, after an edit since the last
        run_tests, gives the file its notes."""
        deps = self.deps
        session = self.session
        source_dir = deps.source_dir
        target_file = deps.target_file_path
        # A file changed since its last run_tests (an edit beside the
        # measurement after a noted run) gets its notes from this run, as
        # from run_tests, so the caller's goal end sees the file's own note.
        notes_due = session.tests_changed_since_run_tests
        normal = normal_run.run(
            backend, self.test_paths(), source_dir,
            timeout=suite_run_timeout_seconds, target=target_file, fail_fast=True, tb="line",
            record_leaks=notes_due,
            timeout_journal=bool(self.journal_request(backend, source_dir)),
        )
        if normal.timed_out:
            return normal, self.on_timeout(
                suite_run_timeout_seconds, log_prefix,
                getattr(normal, "active_at_end", None),
            )
        # A green fail-fast normal run ran the whole suite: its time is the suite
        # time the best-green snapshot records.
        session.last_suite_exec_ms = (
            normal.elapsed_s * 1000.0 if normal.elapsed_s is not None else None
        )
        if not normal.ok:
            msg = self.tool_msg(
                "measure_mutation_red_suite",
                tests_output=_limits.cut_text(
                    normal.output, _limits.value("failing_run_output_chars"),
                    what="the test output", keep="tail"),
            )
            session.tests_changed_since_mutation = False
            session.last_mutation_result = msg
            self._record_red_normal_run(msg)
            self.log(
                f"[{log_prefix}] normal run FAILED "
                f"(returncode={normal.returncode}). Output tail:\n"
                + normal.output[-3000:],
                level=1,
            )
            # Red before any write: the inherited suite, green at handoff, is
            # flaky or its environment changed, so the phase ends.
            if session.writes_done == 0:
                self.log(
                    f"[{log_prefix}] INHERITED-RED EARLY EXIT: "
                    f"the normal run failed before any edit. Ending phase with "
                    f"status=skipped_inherited_red.",
                    level=1,
                )
                session.early_stop_result = MutationTestResult(
                    mutation_test_file_name=deps.mutation_test_file_name,
                    explanation=(
                        "Skipped: the inherited suite is red before any "
                        "edit, mutation cannot be measured. Re-run the "
                        "coverage phase to regenerate a green suite."
                    ),
                    targeted_mutants=[],
                    mutation_score=None,
                    status="skipped_inherited_red",
                )
                session.early_stop_reason = "inherited_red"
                return normal, (
                    "[FRAMEWORK STOP] Coverage tests are red. "
                    + msg
                    + "\nReturn ``MutationTestResult`` now."
                )
            return normal, msg

        # A green normal run ran in time: the timeout streak clears, as after a
        # green run_tests.
        session.consecutive_timeouts = 0
        # The strict-green rule of run_tests: a green run that leaked
        # unsound asynchronous warnings has tests that pass without running
        # their code, so it kills no mutant and is refused like a red one.
        if self.agent_config()["reject_unsound_async_green"]:
            unsound = agent_helpers.unsound_async_rejection(normal, self.tool_msg)
            if unsound is not None:
                msg = self.tool_msg("measure_mutation_red_suite",
                                    tests_output=unsound)
                session.tests_changed_since_mutation = False
                session.last_mutation_result = msg
                self._record_red_normal_run(msg)
                self.log(f"[{log_prefix}] the normal run emitted unsound-async "
                         "warnings, so the measurement is refused", level=1)
                return normal, msg
        if notes_due:
            self._run_notes(normal, backend, log_prefix)
            session.stands_alone_noted = bool(session.stands_alone_note)
        return normal, None

    def _coverage_run_reply(self, backend: Any, suite_run_timeout_seconds: int,
                            log_prefix: str) -> Optional[str]:
        """The coverage tool's run of the suite after the green normal run,
        and the coverage tool's refusal when it does not finish or is not
        green, else None. Its numbers belong to the suite as much as the
        mutation score, and a test that fails only under the coverage tool
        is named here, before the mutation tool starts."""
        covered = backend.run_test_files(
            self.test_paths(), self.deps.source_dir,
            timeout=suite_run_timeout_seconds, measure_coverage_of=self.deps.target_file_path,
            **self.journal_request(backend, self.deps.source_dir),
        )
        if covered.timed_out:
            # The normal run finished in time: only the coverage tool's
            # run did not, so the tool could not measure the suite. The suite
            # is not too slow (the inherited one included) and no timeout
            # counts. The running test, when the journal names it, is the test
            # that stopped the tool.
            from testuary import results

            self.log("WARNING: " + results.note(
                "mutation", "mutation.suite_run_timeout_seconds", suite_run_timeout_seconds))
            active = getattr(covered, "active_at_end", None)
            msg = (self.tool_msg("measure_mutation_timeout_under_coverage_named",
                                 timeout=suite_run_timeout_seconds, test=active) if active
                   else self.tool_msg("measure_mutation_timeout_under_coverage",
                                      timeout=suite_run_timeout_seconds))
            return self._tool_refusal(
                "coverage", msg, [active] if active else [],
                f"[{log_prefix}] the suite did not finish under the coverage tool "
                f"within {suite_run_timeout_seconds}s after a green normal run; the mutation "
                "tool does not start")
        if not covered.ok:
            return self._red_under_coverage_message(covered, log_prefix)
        return None

    def _unusable_outcome(self, results: Any, log_prefix: str, *, after_switch: bool = False,
                          applied_text: str = "") -> Optional[str]:
        """The reply that ends the call when an engine run is unusable, else
        None. No mutant attempted (an empty catalogue or an early engine
        death): the suite's failure under the tool, or no mutant tested. None
        killed or survived (inconclusive: the score means nothing): the phase
        ends. The catalogue switch's run (``after_switch``) has no suite failure
        branch, and its log line names no cause. The first run's line names
        the per-mutant timeout ``applied_text`` as a likely one."""
        session = self.session
        attempted = (
            results.killed + results.survived + results.timeout
            + results.non_viable + results.untested
        )
        if attempted == 0:
            session.tests_changed_since_mutation = False
            if not after_switch and getattr(results, "suite_failed", False):
                return self._red_when_measured_message(results, log_prefix)
            return self._zero_mutants_message(results, log_prefix)
        if (results.killed + results.survived) != 0:
            return None
        session.mutation_inconclusive = True
        what = "full-catalogue switch measurement" if after_switch else "mutation measurement"
        cause = "" if after_switch else (
            " Likely: worker import error, broken test_command, or suite too slow "
            f"for the per-mutant timeout ({applied_text}).")
        self.log(
            f"[CRITICAL] {what} INCONCLUSIVE: 0 useful outcomes from {attempted} "
            f"attempted (suspicious={results.non_viable}, timeout={results.timeout}, "
            f"untested={results.untested}).{cause}",
            level=1,
        )
        session.tests_changed_since_mutation = False
        session.early_stop_result = self._synthesize_mutation_result("engine_inconclusive")
        session.early_stop_reason = "engine_inconclusive"
        return self.tool_msg("measure_mutation_inconclusive")

    def _engine_args(self, suite_s: Optional[float], backend: Any) -> _EngineArgs:
        """The engine runs' arguments: the configured workers, the per-mutant
        timeout derived from the suite's time ``suite_s``, the wall clock and
        the test files of the working tests directory."""
        return _EngineArgs(
            workers=int(_AGENT_CONFIG["engine_workers"]),
            per_mutant_timeout=config_module.per_mutant_timeout(
                _AGENT_CONFIG["per_mutant_timeout_seconds"],
                suite_s,
                cap_s=int(_AGENT_CONFIG["mutation_max_timeout"]),
                floor_s=int(_AGENT_CONFIG["min_per_mutant_timeout_seconds"]),
                multiplier=float(_AGENT_CONFIG["per_mutant_timeout_factor"]),
            ),
            wall_clock_timeout=int(_AGENT_CONFIG["measurement_timeout_seconds"]),
            test_files=mutation.suite_test_files(backend, Path(self.deps.test_output_dir)),
        )

    def _engine_runner(self, engine_args: _EngineArgs, log_prefix: str) -> Any:
        """The runner of a measurement's first engine run, with the run's
        start line logged: the installed pooled session's, which logs to this
        target from now on, or a runner of its own per call."""
        if self.mutation_session is not None:
            sess_workers = getattr(self.mutation_session, "workers", "?")
            sess_use_http = getattr(self.mutation_session, "use_http", False)
            mode = "parallel-http" if sess_use_http else "sequential-local"
            poller_s = _AGENT_CONFIG["cosmic_ray_progress_interval_s"]
            runner = self.mutation_session.runner
            self.log(
                f"[{log_prefix}] reusing pooled session "
                f"(workers={sess_workers}, mode={mode}, "
                f"per_mutant={_requested_timeout_text(runner, engine_args.per_mutant_timeout)}, "
                f"wall_clock={engine_args.wall_clock_timeout}s, "
                f"poller={poller_s}s)",
                level=1,
            )
            # The session opened before this target's logger was bound:
            # its lines come here, set per call, as it serves several targets.
            runner.log.target = self.log
            return runner
        runner = languages.get_backend(self.deps.language).mutation_runner(
            dict(_AGENT_CONFIG), source_dir=self.deps.source_dir,
            log=self.log,
        )
        self.log(
            f"[{log_prefix}] {runner.name} run "
            f"(workers={engine_args.workers}, "
            f"per_mutant={_requested_timeout_text(runner, engine_args.per_mutant_timeout)}, "
            f"wall_clock={engine_args.wall_clock_timeout}s)",
            level=1,
        )
        return runner

    def _run_engine(self, runner: Any, engine_args: _EngineArgs,
                    operators: Optional[List[str]]) -> Any:
        """One engine run of the working tests against the target with
        ``operators``, None for the configured catalogue: in the installed
        pooled session, else by ``runner``. An error of the run is the
        caller's to answer."""
        target = Path(self.deps.target_file_path)
        test_dir = Path(self.deps.test_output_dir)
        if self.mutation_session is not None:
            return self.mutation_session.run(
                target=target, test_dir=test_dir, test_files=engine_args.test_files,
                per_mutant_timeout=engine_args.per_mutant_timeout,
                wall_clock_timeout=engine_args.wall_clock_timeout, operators=operators,
            )
        return runner.run_once(
            workers=engine_args.workers, target=target, test_dir=test_dir,
            test_files=engine_args.test_files, per_mutant_timeout=engine_args.per_mutant_timeout,
            wall_clock_timeout=engine_args.wall_clock_timeout, operators=operators,
        )

    def _red_under_coverage_message(self, res: Any, log_prefix: str) -> str:
        """The message for a suite the normal run passed but whose run under
        the coverage tool failed, collected nothing, ended without results or
        left coverage counts the tool could not read, so the suite has no
        coverage and no mutant is tested: the coverage tool's refusal
        (``_tool_refusal``), naming the failing tests when the run names
        them."""
        stopping, red = agent_helpers.stopping_tests(res)
        if red:
            msg = self.tool_msg("measure_mutation_red_under_coverage",
                                tests_output=self._canonical_view(res, None, res.returncode or 1))
            what = "the suite failed under the coverage tool"
        elif agent_helpers.coverage_counts_inconsistent(res):
            msg = self.tool_msg("measure_mutation_counts_inconsistent")
            what = "the coverage tool could not read the coverage counts of its run of the suite"
        else:
            msg = self.tool_msg("measure_mutation_unmeasured_under_coverage",
                                output=_limits.cut_text(
                                    res.output, _limits.value("empty_run_output_chars"),
                                    what="the test output", keep="tail"))
            what = "the coverage tool's run of the suite ended without a test result"
        return self._tool_refusal(
            "coverage", msg, stopping, f"[{log_prefix}] {what} after a green normal run; "
            "the mutation tool does not start")

    def _tool_refusal(self, tool: str, msg: str, stopping: List[str], line: str) -> str:
        """A measurement ``tool`` refused after the green normal run: the log
        ``line``'s record names the tests that stopped the tool, which the
        results page reads. The suite reads red as after a red run_tests, with
        the refusal as the tool's, and it counts as a failed run: after
        max_failed_test_attempts of them the best-green snapshot returns."""
        session = self.session
        self.log(line, level=1, tool=tool, stopping_tests=list(stopping))
        session.tests_changed_since_mutation = False
        session.last_mutation_result = msg
        self.record_refusal(tool, msg)
        # A restored snapshot brings back the state of its own bytes.
        max_attempts, should_restore = self.failed_run_due()
        if should_restore and self._restore_snapshot(self.target_path()):
            self.log(
                f"[restore] {max_attempts} consecutive failed runs - "
                "reverted the mutation file to the best-green snapshot.",
                level=1,
            )
            msg = self.tool_msg("tests_restore_note", max_attempts=max_attempts, output=msg)
        return msg

    def _zero_mutants_message(self, results: Any, log_prefix: str) -> str:
        """The message for a run that tested no mutant, with the engine's
        reason when it gave one (it failed, or its clock ran out), so the
        model is not left guessing between an uncovered target and a
        broken engine."""
        msg = self.tool_msg("measure_mutation_zero_mutants")
        reason = getattr(results, "reason", "") or ""
        if reason:
            self.log(f"[{log_prefix}] 0 mutants tested: {reason}", level=1)
            msg += "\n\n" + self.tool_msg("measure_mutation_engine_reason", reason=reason)
        return msg

    def _mutation_stall_hint(self) -> str:
        """The hint appended to measure_mutation's answer when the score stalled
        below the goal with the early stop off: no designed stop fires, so
        the model keeps iterating until the request limit ends the loop."""
        if bool(_AGENT_CONFIG["allow_early_stop"]):
            return ""
        max_measurements_without_improvement = int(_AGENT_CONFIG["max_measurements_without_improvement"])
        if max_measurements_without_improvement <= 0:
            return ""
        best = self.session.best_mutation_score
        mut_goal = float(_AGENT_CONFIG["goal"])
        below = best is None or best < mut_goal
        if below and self.session.measurements_without_improvement >= max_measurements_without_improvement:
            return self.tool_msg("stall_hint_mutation_keep_iterating")
        return ""

    # Workflow phase classifier (gates which tools the model sees).

    def tests_changed_since_measurement(self) -> bool:
        return self.session.tests_changed_since_mutation

    def set_tests_changed(self, changed: bool) -> None:
        self.session.tests_changed_since_mutation = changed

    def last_measured_value(self) -> Optional[float]:
        return self.session.prev_mutation_score

    def classify_log_fields(self) -> Dict[str, Any]:
        s = self.session
        return {
            "edits_since_run": s.tests_changed_since_run_tests,
            "edits_since_mutation": s.tests_changed_since_mutation,
            "writes_done": s.writes_done,
            "mutation_measurements": s.mutation_measurements,
        }

    # run_phase hooks.

    def phase_label(self) -> str:
        return "mutation"

    def agent_end_fields(self) -> Dict[str, Any]:
        s = self.session
        return {
            "mutation_measurements": s.mutation_measurements,
            "measure_mutation_calls": s.measure_mutation_calls,
            "engine_runs": s.engine_runs,
            "framework_prep_runs": s.framework_prep_runs,
            "cached_measure_returns": s.cached_measure_returns,
            "writes_done": s.writes_done,
            "run_tests_calls": s.run_tests_calls,
            "best_mutation_score": s.best_mutation_score,
            "last_tests_passed": s.last_tests_passed,
            "restored_from_green": s.best_green.restored_from_green,
            "mutation_mode": s.mutation_mode,
            "switch_iteration": s.switch_iteration,
        }

    def recovery_state_block(self) -> str:
        score = self.session.best_mutation_score
        best = None if score is None else (
            f"  Best mutation score:  {measurement.floor_pct(score):.1f}%  "
            f"(target: {_AGENT_CONFIG['goal']}%)")
        return self.working_state_block(best, "mutation")

    def build_failure_result(self, error_msg: str) -> "MutationTestResult":
        # An error path: the run's end and the message name the status
        # (budget_exhausted, context_overflow, agent_error), with the message
        # as the error. inconclusive stands.
        result = self._synthesize_mutation_result(error_msg)
        if result.status != "inconclusive":
            result = result.model_copy(update={
                "status": self.classify_error_status(error_msg, self._limit_reached),
            })
        return result.model_copy(update={"error": error_msg})

    def finalize_result(
        self, output: Any, *, error_msg: Optional[str], duration_ms: float,
        usage_tracker: Any, final_messages: List[Any],
    ) -> Any:
        # No useful engine outcome: any score is a placeholder, and the status
        # inconclusive records the phase as failed.
        if self.session.mutation_inconclusive and output is not None:
            try:
                if getattr(output, "status", None) != "inconclusive":
                    output = output.model_copy(update={"status": "inconclusive"})
            except Exception:
                pass
        # The [auto] fields come from framework state, so no value the model
        # filled reaches the CSV row or steers the consolidation's file lookup.
        if output is not None:
            try:
                output = output.model_copy(update={
                    "mutation_score": self.session.best_mutation_score,
                    "mutation_test_file_name": (
                        self.deps.mutation_test_file_name
                    ),
                })
            except Exception:
                pass
        # The catalogue switch fields, from session state as in a synthesized
        # result. The model leaves them null.
        if output is not None:
            try:
                output = output.model_copy(update={
                    "mutation_switched": self.session.switch_iteration is not None,
                    "mutation_selective_final": self.session.selective_final_score,
                    "mutation_full_at_switch": self.session.full_score_at_switch,
                })
            except Exception:
                pass
        # The merge takes the file on disk, so a red or unverified last edit
        # must not survive.
        self._settle_best_green(self.target_path(), log_fn=self.log)
        self.assert_target_has_no_shadowed_definitions("mutation final")
        # As in the coverage result: the delivered file's counts, a derived
        # status (a synthesized one stands) and the error.
        if output is not None:
            structural = measurement.record(measurement.observe_file(
                self.target_path(), backend=self.test_backend(),
            ))
            updates: Dict[str, Any] = {
                "test_count_mutation": structural["tests"],
                "test_loc_mutation": structural["code_lines"],
                "source_observations": {"mutation": structural},
                "agent_iterations": self.session.session_tool_calls,
            }
            try:
                current_hash = source_sha256(
                    self.target_path().read_text(encoding="utf-8"))
            except OSError:
                current_hash = None
            if (
                current_hash is not None
                and current_hash == self.session.last_complete_matrix_hash
                and self.target_path().name
                == self.session.last_complete_execution_name
                and self.session.last_complete_matrix is not None
            ):
                updates.update({
                    "measured_suite_sha256": current_hash,
                    "measured_execution_name": self.target_path().name,
                    "mutation_matrix": self.session.last_complete_matrix,
                    "measurement_identity": (
                        self.session.last_complete_measurement_identity
                    ),
                })
            # A synthesized result keeps its status and error. Otherwise the
            # framework derives them over anything the model filled.
            if self.session.early_stop_result is None:
                if error_msg:
                    updates["status"] = self.classify_error_status(error_msg, self._limit_reached)
                elif self.session.best_mutation_score is None:
                    updates["status"] = "tests_failed"
                else:
                    updates["status"] = "success"
                updates["error"] = error_msg or ""
            elif (not getattr(output, "error", "")
                    and self.session.early_stop_reason
                    and getattr(output, "status", None) != "success"):
                # A designed stop short of the goal names its reason in the error
                # field, as the coverage result does.
                updates["error"] = self.session.early_stop_reason
            try:
                output = output.model_copy(update=updates)
            except Exception:
                pass
        return super().finalize_result(
            output, error_msg=error_msg, duration_ms=duration_ms,
            usage_tracker=usage_tracker, final_messages=final_messages,
        )

    def build_state_marker(self, _s: Any) -> str:
        """The state marker: the task and the best mutation score, so a history
        trim loses neither."""
        session = self.session
        score = self.stalled_metric(
            session.best_mutation_score, lambda v: f"{measurement.floor_pct(v):.1f}%",
        )
        field = f"mutation={score}"
        head = self.task_head()
        return f"[state | {head}{field}]"

    # Output validator hooks. The shared base owns the acceptance flow.

    def agent_config(self) -> Dict[str, Any]:
        return _AGENT_CONFIG

    def validate_result_fields(self, parsed: MutationTestResult) -> Optional[str]:
        """The rejection of a result whose explanation is the placeholder phrase
        or whose ``targeted_mutants`` list is empty or echoes the schema, else
        None: the content gates coverage applies to its result."""
        if agent_helpers.looks_placeholder_text(
            parsed.explanation, _MUTATION_PLACEHOLDER_FRAGMENTS,
        ):
            return self.tool_msg("validator_placeholder_explanation")
        if not parsed.targeted_mutants:
            self.log(
                "[validator] Rejected: targeted_mutants is empty.", level=1,
            )
            return self.tool_msg("validator_empty_targeted_mutants")
        placeholder_ids = [
            m for m in parsed.targeted_mutants if _looks_placeholder_mutant(m)
        ]
        if placeholder_ids:
            self.log(
                f"[validator] Rejected targeted_mutants placeholder entries: "
                f"{placeholder_ids!r}",
                level=1,
            )
            return self.tool_msg(
                "validator_placeholder_targeted_mutants",
                names=repr(placeholder_ids),
            )
        return None

    def goal_met(self) -> bool:
        """Whether the best score reached the goal on the full catalogue. While a
        switch is possible, a selective score at the goal switches instead."""
        if self.session.selective_switch_possible:
            return False
        best = self.session.best_mutation_score
        mut_goal = float(_AGENT_CONFIG["goal"])
        return best is not None and best >= mut_goal

    def early_stop_allowed(self) -> bool:
        """Whether an early stop may end the phase: not while a switch is
        pending, since a selective plateau switches instead."""
        return not self.session.selective_switch_possible

    def measurement_count(self) -> int:
        return self.session.mutation_measurements

    def early_stop_reject_message(self) -> str:
        score = self.session.best_mutation_score
        goal = float(_AGENT_CONFIG["goal"])
        best = ("no measurement yet" if score is None else
                f"mutation score {measurement.floor_pct(score):.1f}%, goal {goal:g}%")
        return self.tool_msg("validator_stagnation_not_exhausted", best=best)


def _seed_mutation_baseline_from_matrix(agent: "MutationAgent") -> bool:
    """Seed the mutation baseline from the coverage phase's full-catalogue
    matrix instead of an engine run. True when it seeded (the caller then
    skips the engine baseline). The post-coverage consolidation writes
    ``.mutation_baseline_matrix.json`` for the selected suite only, bound to
    its bytes, so the scores and survivor sample derive from the inherited
    suite's own matrix. The session is seeded as a measured baseline would
    be (best, previous, counts, survivors, the cached result), without a
    ``mutation_measurements`` count. As the start's measurement it satisfies
    the validator's measurement precondition while the suite is unchanged.
    The at-goal stop of ``_maybe_set_mutation_stop`` can end the
    phase. With two catalogues, a derived selective score at the switch
    threshold switches at once on the same matrix's full score.
    """
    deps = agent.deps
    session = agent.session
    matrix_path = Path(deps.test_output_dir) / ".mutation_baseline_matrix.json"
    if not matrix_path.exists():
        return False
    try:
        sidecar = json.loads(matrix_path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return False
    # Without a matching suite hash the matrix may not belong to the suite on
    # disk: the derive refuses and the baseline is measured instead.
    matrix = sidecar.get("matrix") if isinstance(sidecar, dict) else None
    stamped = (
        sidecar.get("suite_sha256") if isinstance(sidecar, dict) else None
    )
    if not isinstance(matrix, dict) or not stamped:
        log_print(
            "[mutation] baseline sidecar refused: no suite hash binds the "
            "matrix to the inherited bytes. Measuring instead.", level=1,
        )
        return False
    # A matrix of an earlier kill-matrix version reads in this version's words.
    try:
        matrix = mutation.upgrade_kill_matrix(matrix)
    except ValueError as refusal:
        log_print(f"[mutation] baseline sidecar refused: {refusal}. Measuring "
                  f"instead.", level=1)
        return False
    expected_identity = mutation_measurement_identity(
        target_file_path=deps.target_file_path,
        language=deps.language,
        execution_name=agent.target_path().name,
    )
    # The baseline reads the matrix's statuses only, which do not depend on
    # how killing tests were attributed: the sidecar holds the phase end's
    # all-killers matrix, the loop's own engine stops at the first killer.
    sidecar_identity = sidecar.get("measurement_identity")
    if (not isinstance(sidecar_identity, dict)
            or {k: v for k, v in sidecar_identity.items() if k != "attribution"}
            != {k: v for k, v in expected_identity.items() if k != "attribution"}):
        log_print(
            "[mutation] baseline sidecar refused: target or mutation "
            "configuration identity mismatch. Measuring instead.", level=1,
        )
        return False
    try:
        inherited = agent.target_path().read_text(encoding="utf-8")
    except OSError:
        log_print(
            "[mutation] baseline sidecar refused: no inherited suite on "
            "disk to verify against. Measuring instead.", level=1,
        )
        return False
    digest = source_sha256(inherited)
    if digest != stamped:
        log_print(
            f"[mutation] baseline sidecar refused: suite hash mismatch "
            f"(sidecar {stamped[:12]}, inherited {digest[:12]}). Measuring "
            f"instead.", level=1,
        )
        return False
    # A selective derive needs operator labels on the mutants. Without them only
    # a full derive is possible.
    loop_full = (
        str(_AGENT_CONFIG["loop_catalogue"]) == "full"
    )
    backend = languages.get_backend(deps.language)
    try:
        runner = backend.mutation_runner(dict(_AGENT_CONFIG),
                                         source_dir=deps.source_dir)
    except Exception:
        return False
    sel_ops: Optional[set] = None
    if not loop_full:
        configured = runner.configured_operators()
        if not configured:
            return False
        if not mutation.has_operators(matrix):
            return False
        sel_ops = set(configured)

    total_mutants = len(matrix.get("mutants") or [])
    log_debug(
        f"[mutation] baseline derive: matrix={matrix_path.name}, "
        f"mutants={total_mutants}, "
        f"catalogue={'full' if loop_full else 'selective'}, "
        f"selective_ops={'all' if sel_ops is None else len(sel_ops)}",
        level=1,
    )
    score = mutation.matrix_score_percent(matrix, operators=sel_ops)
    if score is None:
        return False
    killed = len(mutation.killed_ids(matrix, operators=sel_ops))
    valid = len(mutation.valid_ids(matrix, operators=sel_ops))
    max_diffs = _limits.value("survivor_diffs")
    sample = mutation.render_survivors_from_matrix(
        matrix, operators=sel_ops, max_mutants=max_diffs,
        max_chars=_limits.value("survivor_diff_chars"),
    )

    session.prev_mutation_score = score
    session.last_mutation_counts = (killed, valid - killed)
    session.last_surviving_sample = sample
    session.best_mutation_score = score
    session.measurements_without_improvement = 0
    if loop_full:
        session.mutation_mode = "full"
    log_print(
        f"[mutation] baseline derived from the coverage full-catalogue matrix: "
        f"{measurement.floor_pct(score):.1f}% ({killed} killed / {valid - killed} "
        f"surviving), with no engine re-measure", level=1,
    )

    # Two-phase immediate switch: a derived selective already at the switch
    # threshold switches now, reusing the same matrix's full score.
    selective_active = (
        bool(_AGENT_CONFIG["full_after_selective_goal"])
        and session.mutation_mode == "selective"
    )
    full_available = runner.full_operators() is not None
    if selective_active and full_available:
        session.selective_switch_possible = True
        threshold = float(
            _AGENT_CONFIG["selective_switch_threshold"])
        if score >= threshold:
            full_score = mutation.matrix_score_percent(matrix, operators=None)
            if full_score is not None:
                full_killed = len(mutation.killed_ids(matrix))
                full_valid = len(mutation.valid_ids(matrix))
                session.selective_final_score = score
                session.full_score_at_switch = full_score
                session.best_mutation_score = full_score
                session.prev_mutation_score = full_score
                session.last_mutation_counts = (
                    full_killed, full_valid - full_killed)
                session.last_surviving_sample = (
                    mutation.render_survivors_from_matrix(
                        matrix, operators=None, max_mutants=max_diffs,
                        max_chars=_limits.value("survivor_diff_chars"))
                )
                session.mutation_mode = "full"
                session.switch_iteration = 0
                session.selective_switch_possible = False
                log_print(
                    f"[mutation] derived selective "
                    f"{measurement.floor_pct(score):.1f}% >= switch threshold "
                    f"{threshold:.0f}%. Switched to the full catalogue reusing "
                    f"the coverage matrix, full score "
                    f"{measurement.floor_pct(full_score):.1f}%", level=1,
                )
    else:
        # No switch is possible (the feature is off, the loop is full-catalogue,
        # or the engine's full catalogue cannot be read), so a derived score at
        # the goal is the real metric and the at-goal stop below may fire on it.
        session.selective_switch_possible = False

    # Either way, the derived baseline answers later calls from the cache, as
    # a measured one would, until an edit (``tests_changed_since_mutation``).
    goal = float(_AGENT_CONFIG["goal"])
    max_measurements_without_improvement = int(_AGENT_CONFIG["max_measurements_without_improvement"])
    shown_ops = None if session.mutation_mode == "full" else sel_ops
    summary = (
        f"{_mutation_summary_line(_matrix_measurement(matrix, shown_ops, session.best_mutation_score))}\n"
        f"Goal: {goal:g}%. Stagnation: 0/{max_measurements_without_improvement}."
    )
    if (
        session.switch_iteration == 0
        and session.selective_final_score is not None
    ):
        summary = (
            _catalogue_switch_banner(session.selective_final_score) + summary
        )
    elif session.selective_switch_possible:
        summary = _selective_approximation_banner(float(
            _AGENT_CONFIG["selective_switch_threshold"]
        )) + summary
    if session.last_surviving_sample:
        summary = (
            f"{summary}\n\n--- Surviving mutants (sample) ---\n"
            f"{session.last_surviving_sample}"
        )
    session.last_mutation_result = summary
    session.tests_changed_since_mutation = False
    session.measured_suite_hash = digest
    if loop_full:
        session.last_complete_matrix = matrix
        session.last_complete_matrix_hash = digest
        session.last_complete_execution_name = agent.target_path().name
        # The matrix is the sidecar's, measured with its attribution.
        session.last_complete_measurement_identity = sidecar_identity

    # The derived baseline stamps best-green the same way the first
    # measurement would, so restore ranking and the operator log carry the
    # starting score on either preparation path.
    try:
        mut_path = agent.target_path()
        if mut_path.exists():
            structural = measurement.record(measurement.observe_file(
                mut_path, backend=backend,
            ))
            update_best_green(
                session.best_green, mut_path,
                coverage=getattr(deps, "coverage_baseline_pct", None),
                mutation=session.best_mutation_score, test_count=structural["tests"],
                test_source_observation=structural,
                exec_time_ms=session.last_suite_exec_ms,
                persist_to_disk=True, log_fn=agent.log,
                loop_state=agent._snapshot_loop_state(inherited),
            )
    except Exception as exc:
        log_debug(
            f"[mutation] baseline derive best-green stamp failed: "
            f"{type(exc).__name__}: {exc}", level=1,
        )

    # A baseline at the goal ends the phase without a model call (the stop
    # bypasses the validator), except while a switch is pending: a selective
    # score at the goal switches first.
    agent._maybe_set_mutation_stop(source_tool="baseline_derive")
    log_debug(
        f"[mutation] baseline derive seeded: mode={session.mutation_mode}, "
        f"best={session.best_mutation_score}, "
        f"switch_iteration={session.switch_iteration}, "
        f"early_stop={'yes' if session.early_stop_result is not None else 'no'}",
        level=1,
    )
    return True


# Entry point


@_limits.in_phase("mutation")
def generate_mutation_tests(
    project_name: str,
    target_file_path: str,
    test_output_dir: str,
    preseed: Optional[dict] = None,
    coverage_baseline_pct: Optional[float] = None,
    *, prepared_models=None,
) -> MutationTestResult:
    """Generate the mutation test file for one target and return its
    ``MutationTestResult``.
    ``preseed`` is the coverage phase's suite (``content``, and the source's
    ``file_stem`` for the class-language rename), written into the working
    file before the agent starts, so the model edits the inherited suite in
    place. The phase end keeps that suite as a separate measured candidate.
    Raises RuntimeError without a non-empty ``preseed``."""
    if not preseed or not str(preseed.get("content") or "").strip():
        raise RuntimeError(
            "mutation requires a nonempty validated suite from the coverage "
            "phase and stopped before prompt construction or a model call"
        )
    file_content = project.read_file_content(target_file_path)
    source_dir = project.resolve_source_dir(target_file_path)
    language = languages.detect_language(target_file_path)
    Path(test_output_dir).mkdir(parents=True, exist_ok=True)
    backend = languages.get_backend(language)
    mutation_file_name = backend.target_test_path(
        test_output_dir, target_file_path).name
    backend.setup_test_env(test_output_dir, source_dir, target_file_path)

    resolved_prompts = prompts_module.resolve_agent_prompts(
        _AGENT_NAME, language, source_dir=source_dir,
        hint_profile=hints.hint_profile(),
    )

    log_print(f"[mutation] target: {target_file_path}")
    log_print(f"[mutation] mutation file: {mutation_file_name}")
    if coverage_baseline_pct is not None:
        log_print(f"[mutation] coverage from coverage phase: {coverage_baseline_pct:.1f}%")

    deps = MutationTestDeps(
        target_file_path=target_file_path,
        target_file_content=file_content,
        source_dir=source_dir,
        test_output_dir=test_output_dir,
        language=language,
        mutation_test_file_name=mutation_file_name,
        coverage_baseline_pct=coverage_baseline_pct,
    )

    session = TestMutationSession()
    # Phase clock: everything from here (the pre-seed, framework baseline
    # preparation, the model loop, terminal finalization) is phase cost.
    session.phase_clock_start = time.perf_counter()
    # ``full`` runs the full catalogue from the first measurement. The agent
    # sees the full score throughout and the selective-proxy switch never fires.
    if str(_AGENT_CONFIG["loop_catalogue"]) == "full":
        session.mutation_mode = "full"
    mut_path = Path(test_output_dir) / mutation_file_name
    seed = preseed["content"]
    refuse_rewritten_inherited_suite(backend, seed, "mutation", "coverage")
    mut_path.write_text(seed, encoding="utf-8")
    update_best_green(
        session.best_green, mut_path,
        coverage=coverage_baseline_pct,
        log_fn=lambda m, level=1: log_print(m, level=level),
        loop_state={"measured": False},
    )
    log_print(
        f"[mutation] pre-seeded with the coverage suite "
        f"({len(seed.splitlines())} lines)"
    )

    agent = MutationAgent(
        prompts=resolved_prompts,
        deps=deps,
        session=session,
        logger=_log.logger,
        mutation_session=_mutation_session,
    )
    agent._prepared_models = prepared_models
    agent.record_generation_origin()

    # The framework baseline, the loop's measurement run once before the
    # model: the prompt carries the starting score and survivors, and a suite
    # at the goal ends the phase without a model call. An exact full-catalogue
    # matrix from the coverage phase derives it instead.
    prep_start = time.perf_counter()
    if not _seed_mutation_baseline_from_matrix(agent):
        log_print("[mutation] framework baseline measurement")
        agent.framework_baseline_measure()
    session.framework_prep_seconds = time.perf_counter() - prep_start
    if session.early_stop_result is not None:
        log_print(
            "[mutation] phase ended at the baseline "
            f"({session.early_stop_reason})"
        )
        # The baseline stop never reaches run_phase: the phase cost (framework
        # preparation is the whole phase, no tokens) and the inherited file's
        # counts are stamped here.
        result = session.early_stop_result
        structural = measurement.record(measurement.observe_file(
            Path(test_output_dir) / mutation_file_name, backend=backend,
        ))
        try:
            result = result.model_copy(update={
                "agent_duration_s": round(
                    time.perf_counter() - session.phase_clock_start, 1,
                ),
                "model_requests": 0,
                "logical_requests": 0,
                "unknown_usage_requests": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cache_hit_tokens": 0,
                "unknown_cache_requests": 0,
                "test_count_mutation": structural["tests"],
                "test_loc_mutation": structural["code_lines"],
                "source_observations": {"mutation": structural},
                "measured_suite_sha256": (
                    session.last_complete_matrix_hash
                    if session.last_complete_matrix is not None else None
                ),
                "measured_execution_name": (
                    session.last_complete_execution_name
                    if session.last_complete_matrix is not None else None
                ),
                "mutation_matrix": session.last_complete_matrix,
                "measurement_identity": (
                    session.last_complete_measurement_identity
                ),
            })
        except Exception:
            pass
        agent.assert_target_has_no_shadowed_definitions("mutation final")
        return result

    # The inherited suite is green, as the coverage phase records its own at
    # entry: the framework's normal run ran it green (a red or timed-out one
    # ends the phase above), or its matrix was derived from the coverage
    # phase's green measurement of the same bytes. A refusal of the coverage
    # tool at the baseline stands, so the loop starts by editing.
    if not agent.refused_by():
        session.last_tests_passed = True
        session.tests_changed_since_run_tests = False

    baseline_score = session.prev_mutation_score
    killed, survived = session.last_mutation_counts or (None, None)
    if baseline_score is not None:
        baseline_mutation_clause = (
            "The framework already measured the baseline: mutation score "
            f"{measurement.floor_pct(baseline_score):.1f}% "
            f"({killed} killed / {survived} surviving)."
        )
    else:
        baseline_mutation_clause = (
            "The baseline was not measured before the loop; the start's "
            "measurement measures it."
        )

    prompt = Template(resolved_prompts["user_prompt_template"]).substitute(
        file_path=project.display_path(target_file_path, source_dir),
        mutation_test_file_name=mutation_file_name,
        coverage_baseline_clause=(
            f" It measured {measurement.floor_pct(coverage_baseline_pct):.1f}% line coverage."
            if coverage_baseline_pct is not None
            else " Its line coverage was not measured."
        ),
        mutation_goal_clause=f"{float(_AGENT_CONFIG['goal']):g}%",
        mutation_patience=int(_AGENT_CONFIG["max_measurements_without_improvement"]),
        stagnation_min_delta=f"{float(_AGENT_CONFIG['min_improvement_points']):g}",
        mutation_minimum=f"{float(_AGENT_CONFIG['minimum']):g}",
        baseline_mutation_clause=baseline_mutation_clause,
    )

    return agent.run_phase(project_name=project_name, user_prompt=prompt)
