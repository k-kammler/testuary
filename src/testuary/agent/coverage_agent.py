"""The coverage agent: it extends the starter phase's suite for a target with
coverage feedback, and the mutation agent later starts from this phase's test
file. A ``CoverageAgent`` keeps its per-target state on ``self.session`` and
``self.deps``, binds its tools per instance and edits through the shared
``CodingAgent.edit_file`` gate.
"""

import re
import time
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

from testuary import languages
from testuary.agent import _limits
from testuary.generation_evidence import source_sha256
from testuary import config
from testuary import hints
from testuary import project
from testuary import prompts as prompts_module
from testuary import measurement
from testuary.agent import _helpers as agent_helpers
from testuary.languages.generic import FAILED_MARKER
from testuary.agent._base import CodingAgent, refuse_rewritten_inherited_suite, stop_status
from testuary.agent._helpers import (
    LoopBreakerState,
    regression_revert_decision,
    update_best_green,
)


# Configuration

_API_CONFIG = config.PhaseConfiguration("api", api=True)
_AGENT_CONFIG = config.PhaseConfiguration("test_coverage")
_AGENT_NAME = "test_coverage"

# Logging

_log = agent_helpers.PhaseLogger("test_coverage")


def init_logging(project_name, log_dir, log_filename):
    """Start the coverage phase's streamed log ``log_filename`` in ``log_dir``
    for ``project_name``. Its configuration names the phase's model and its
    loop attempt limit."""
    _log._config = {"model": _API_CONFIG["model"], "max_loop_attempts": _AGENT_CONFIG["max_loop_attempts"]}
    _log.init(project_name, log_dir, log_filename)


log_print = _log.log
save_logs = _log.save


# Helpers without session state: test output and coverage rendering


def _compress_line_ranges(lines: list) -> str:
    """Compress [1,2,3,5,7,8,9] into '1-3, 5, 7-9'."""
    if not lines:
        return ""
    sorted_lines = sorted(set(lines))
    ranges = []
    start = prev = sorted_lines[0]
    for n in sorted_lines[1:]:
        if n == prev + 1:
            prev = n
        else:
            ranges.append(f"{start}-{prev}" if start != prev else str(start))
            start = prev = n
    ranges.append(f"{start}-{prev}" if start != prev else str(start))
    return ", ".join(ranges)


def _jacoco_miss_line(output: Optional[str]) -> str:
    """The JaCoCo diagnosis line of a run's ``output``, stripped, which says
    why the run measured no coverage, else an empty string."""
    for line in (output or "").splitlines():
        if "[testuary][jacoco-miss]" in line:
            return line.strip()
    return ""


def _format_trend(label: str, prev: Optional[float], curr: float, unit: str = "%") -> str:
    """One-line trend summary for a metric measurement."""
    if prev is None:
        return f"{label}: {measurement.floor_pct(curr):.1f}{unit} (first measurement)"
    delta = curr - prev
    sign = "+" if delta >= 0 else ""
    return f"{label}: {measurement.floor_pct(prev):.1f}{unit} -> {measurement.floor_pct(curr):.1f}{unit} ({sign}{delta:.1f}{unit})"


def _render_uncovered_code(target_file: str, missing_lines) -> str:
    """The uncovered lines of ``target_file`` for the model to test, from the
    backend's ``missing_lines`` (1-based): every region line-numbered with two
    lines of context and ``>>>`` markers, or, past the ``uncovered_chars``
    limit, what fits, marked as a subset, with the complete range map and
    read_source calls for the rest. ``_render_uncovered_branches`` renders the
    branches.
    """
    missing = sorted(set(missing_lines or ()))
    if not missing:
        return ""
    try:
        source_lines = project.read_file_content(target_file).split("\n")
    except Exception:
        return ""

    missing_set = set(missing)
    blocks = _group_into_blocks(missing, context=2)
    header = f"=== UNCOVERED ({len(missing)} lines, {len(blocks)} regions) ===\n\n"

    def _render_block(block_start, block_end, context_start, context_end) -> str:
        out = f"--- Lines {block_start}-{block_end} ---\n"
        for i in range(context_start - 1, min(context_end, len(source_lines))):
            line_num = i + 1
            marker = ">>>" if line_num in missing_set else "   "
            out += f"{marker} {line_num:4d} | {source_lines[i]}\n"
        return out + "\n"

    rendered = [_render_block(*b) for b in blocks]
    limit = _limits.value("uncovered_chars")

    # Common case: it all fits, so inline every region, line-numbered.
    if limit is None or len(header) + sum(len(r) for r in rendered) <= limit:
        return header + "".join(rendered)

    # Big-file case: inline what fits, then a clearly-signposted read-fallback.
    parts: List[str] = []
    used = len(header)
    shown = 0
    for r in rendered:
        if shown > 0 and used + len(r) > limit:
            break
        parts.append(r)
        used += len(r)
        shown += 1
    rest = blocks[shown:]
    # The suggestions are examples beside the complete range list, so a
    # shorter list needs no marker of its own.
    tail = (
        _limits.marker(len(rest), "items", "further uncovered regions",
                       "read them with read_source and add tests for each")
        + f"\nComplete uncovered ranges: {_compress_line_ranges(missing)}\n"
        + _read_suggestions(rest))
    return header + "".join(parts) + tail


def _read_suggestions(rest) -> str:
    """Example read_source calls for regions not shown, within the
    ``uncovered_suggestions`` limit."""
    examples, omitted = _limits.cut_items(rest, _limits.value("uncovered_suggestions"),
                                          what="further suggestions")
    return ("For example: " + "; ".join(
        f"read_source(start_line={cs}, end_line={ce})" for _, _, cs, ce in examples)
        + (f" {omitted}" if omitted else ""))


#: Java's untaken-branch count for a line, the part after its ``->``.
_UNTAKEN_COUNT_RE = re.compile(r'(\d+) of \d+ branches untaken')


def _untaken_branches(missing_branches) -> Tuple[int, int]:
    """The untaken branches and their decision lines in ``missing_branches``:
    a ``"src->dst"`` arc is one branch, Java's ``"line->k of n branches
    untaken"`` entry is ``k``."""
    count, lines = 0, set()
    for entry in missing_branches or ():
        src, _, dst = entry.partition("->")
        m = _UNTAKEN_COUNT_RE.fullmatch(dst)
        count += int(m.group(1)) if m else 1
        lines.add(src)
    return count, len(lines)


def _coverage_summary(line_pct, branch_pct, missing_lines, missing_branches,
                      raw_output: str = "") -> Tuple[str, Optional[str]]:
    """The measurement's summary block and the compressed uncovered line
    ranges for the state marker (None when the measurement leaves them as
    they were). The live measurement and the stored one render through it."""
    ranges: Optional[str] = None
    output = "--- Coverage Summary ---"
    if line_pct is not None:
        output += f"\nLine coverage: {measurement.floor_pct(line_pct):.1f}%"
    if branch_pct is not None:
        output += f"\nBranch coverage: {measurement.floor_pct(branch_pct):.1f}%"
    if missing_lines:
        ranges = _compress_line_ranges(list(missing_lines))
        output += f"\nUncovered lines: {ranges}"
        output += "\nThe uncovered source is appended below."
    elif line_pct is not None:
        output += "\nAll target lines covered."
        ranges = ""
    if missing_branches:
        untaken, decisions = _untaken_branches(missing_branches)
        output += (f"\nUncovered branches: {untaken} untaken on {decisions}"
                   f" decision line{'s' if decisions != 1 else ''} -- the"
                   " decision lines and untaken directions are appended"
                   " below.")
    elif branch_pct is not None:
        output += "\nAll target branches covered."
    elif line_pct is not None:
        # Line coverage without branch coverage: the target has no
        # branches, which is a measurement, not a failure to measure.
        output += "\nThe target has no branches."
    else:
        output += "\n(coverage was not measurable this run)"
        # The backend's diagnosis, so the model sees why coverage was
        # unmeasurable.
        diagnosis = _jacoco_miss_line(raw_output)
        if diagnosis:
            output += "\n" + diagnosis
    return output, ranges


def _coverage_report(target_file: str, summary: str, missing_lines,
                     missing_branches) -> str:
    """The measurement's answer before its loop notes: the summary, then
    the uncovered lines and the uncovered branches in source context."""
    report = summary if summary.strip() else (
        "Coverage report is empty - check that tests exist and pass.")
    uncovered_section = _render_uncovered_code(target_file, missing_lines)
    if uncovered_section:
        report += "\n\n" + uncovered_section
    # Empty once branch coverage is complete.
    branch_section = _render_uncovered_branches(target_file, missing_branches)
    if branch_section:
        report += "\n\n" + branch_section
    return report


def _render_uncovered_branches(target_file: str, missing_branches) -> str:
    """The uncovered branches in the detail of the uncovered lines: each
    decision line in source context, marked, with its untaken destinations
    inline. ``missing_branches`` are ``"src->dst"`` arcs (``dst`` is ``exit``
    for a jump out of the function) or Java's one entry per line, ``"line->k
    of n branches untaken"``, which names no destination. Empty when branch
    coverage is complete. The limit and read fallback are the lines'."""
    arcs = [b for b in (missing_branches or ()) if "->" in b]
    if not arcs:
        return ""
    try:
        source_lines = project.read_file_content(target_file).split("\n")
    except Exception:
        return ""

    # The untaken destinations per decision (source) line, stable order.
    dests: Dict[int, List[str]] = {}
    for a in arcs:
        src_s, _, dst = a.partition("->")
        try:
            src = int(src_s)
        except ValueError:
            continue
        slot = dests.setdefault(src, [])
        if dst not in slot:
            slot.append(dst)
    if not dests:
        return ""

    src_lines = sorted(dests)
    src_set = set(src_lines)
    blocks = _group_into_blocks(src_lines, context=2)
    header = (f"=== UNCOVERED BRANCHES ({_untaken_branches(arcs)[0]} untaken "
              f"branches, {len(src_lines)} decision lines) ===\n\n")

    def _describe_dest(dst: str) -> str:
        """The untaken destination with the code it leads to, not a bare line
        number: ``exit`` leaves the function or loop without entering, and
        ``exit:N`` names the code object it leaves by its first line N (its def
        or first decorator), shown as the def line."""
        if dst == "exit":
            return "exit (control leaves the function / loop here)"
        if dst.startswith("exit:") and dst[5:].isdigit():
            first = int(dst[5:])
            if 0 < first <= len(source_lines):
                header = next(
                    (text.strip() for text in source_lines[first - 1:first + 20]
                     if text.lstrip().startswith(("def ", "async def ", "class "))),
                    source_lines[first - 1].strip())
                return "exit from " + _limits.cut_text(
                    header, _limits.value("branch_snippet_chars"), what="the line",
                    keep="head", sep=" ") + " (control leaves the function here)"
            return "exit (control leaves the function / loop here)"
        try:
            dl = int(dst)
        except ValueError:
            return f"line {dst}"
        if 0 < dl <= len(source_lines):
            snippet = source_lines[dl - 1].strip()
            if snippet:
                return f"line {dst} (" + _limits.cut_text(
                    snippet, _limits.value("branch_snippet_chars"), what="the line",
                    keep="head", sep=" ") + ")"
        return f"line {dst}"

    def _render_block(block_start, block_end, context_start, context_end) -> str:
        out = f"--- Lines {block_start}-{block_end} ---\n"
        for i in range(context_start - 1, min(context_end, len(source_lines))):
            line_num = i + 1
            if line_num in src_set:
                if all(_UNTAKEN_COUNT_RE.fullmatch(d) for d in dests[line_num]):
                    note = "    <- " + "; ".join(dests[line_num])
                else:
                    targets = "; ".join(_describe_dest(d) for d in dests[line_num])
                    note = f"    <- branch to {targets} not taken"
                out += f">>> {line_num:4d} | {source_lines[i]}{note}\n"
            else:
                out += f"    {line_num:4d} | {source_lines[i]}\n"
        return out + "\n"

    rendered = [_render_block(*b) for b in blocks]
    limit = _limits.value("uncovered_chars")
    if limit is None or len(header) + sum(len(r) for r in rendered) <= limit:
        return header + "".join(rendered)

    parts: List[str] = []
    used = len(header)
    shown = 0
    for r in rendered:
        if shown > 0 and used + len(r) > limit:
            break
        parts.append(r)
        used += len(r)
        shown += 1
    rest = blocks[shown:]
    listed, arcs_omitted = _limits.cut_items(arcs, _limits.value("branch_arcs"),
                                             what="further entries")
    arc_map = ", ".join(listed) + (f" {arcs_omitted}" if arcs_omitted else "")
    tail = (
        _limits.marker(len(rest), "items", "further branch regions",
                       "read them with read_source and add tests")
        + f"\nUncovered branches: {arc_map}\n" + _read_suggestions(rest))
    return header + "".join(parts) + tail


def _group_into_blocks(line_numbers: list, context: int = 2) -> list:
    """The sorted ``line_numbers`` grouped into blocks (at most 3 apart join), as
    ``(block_start, block_end, context_start, context_end)``."""
    if not line_numbers:
        return []

    blocks = []
    block_start = block_end = line_numbers[0]

    for num in line_numbers[1:]:
        if num <= block_end + 3:  # merge blocks that are close together
            block_end = num
        else:
            blocks.append((
                block_start, block_end,
                max(1, block_start - context),
                block_end + context
            ))
            block_start = block_end = num

    blocks.append((
        block_start, block_end,
        max(1, block_start - context),
        block_end + context
    ))
    return blocks


def _cleanup_temp_files(test_output_dir: str) -> None:
    """Remove temporary files created during test generation."""
    for name in (".coverage", "coverage.json"):
        p = Path(test_output_dir) / name
        if p.exists():
            p.unlink(missing_ok=True)
    # The best-green sidecars: the end-of-target restore writes the snapshot
    # from memory. The glob covers every language and several targets.
    try:
        for p in Path(test_output_dir).glob(".best_passing_*"):
            p.unlink(missing_ok=True)
    except Exception:
        pass
    log_print("[cleanup] Removed temporary coverage artifacts", level=1)


def _green_snapshot_key(
    snapshot: Optional[Dict[str, Any]],
) -> tuple:
    """The ranking key of a green snapshot: its goal metric value, higher being
    better (0.0 when unmeasured, None sorts below every snapshot). A later
    suite replaces the snapshot only with a strictly higher value, as in the
    mutation phase. Runtime and test count decide nothing."""
    if snapshot is None:
        return (-1.0,)
    goal_value = _goal_metric_value(
        snapshot.get("coverage"), snapshot.get("branch_coverage"),
    )
    return (goal_value if goal_value is not None else 0.0,)


def _goal_metric() -> str:
    """The coverage metric ``goal`` is checked against: ``line``, ``branch``,
    ``either`` or ``both`` (the strictest, also for an unknown value)."""
    val = str(_AGENT_CONFIG["goal_metric"]).lower().strip()
    if val not in ("line", "branch", "either", "both"):
        return "both"
    return val


def _goal_metric_value(
    line_pct: Optional[float], branch_pct: Optional[float],
) -> Optional[float]:
    """The value of a (line, branch) pair compared against ``goal`` under
    ``goal_metric``: line coverage for a measurement without branch coverage
    (a target without branches), whatever the setting. None when the value
    the setting needs was not measured."""
    mode = _goal_metric()
    if line_pct is not None and branch_pct is None:
        return line_pct
    if mode == "line":
        return line_pct
    if mode == "branch":
        return branch_pct
    if mode == "either":
        present = [v for v in (line_pct, branch_pct) if v is not None]
        return max(present) if present else None
    # both: both numbers must be available.
    if line_pct is None or branch_pct is None:
        return None
    return min(line_pct, branch_pct)


# Result schema

class CoverageTestResult(BaseModel):
    """Result returned by the test generation agent."""

    test_file_name: str = Field(
        description="Filename of the on-disk working file (e.g. test_encoder.py)."
    )
    explanation: str = Field(
        description="Brief explanation of test strategy and what is covered."
    )
    targeted_functions: List[str] = Field(
        description="List of functions/methods explicitly tested."
    )
    # Filled by ``finalize_result`` from the files on disk and the counters.
    # The model leaves them null.
    line_coverage: Optional[float] = Field(default=None, description="[auto] Final line-coverage % from the last measure_coverage measurement.")
    branch_coverage: Optional[float] = Field(default=None, description="[auto] Final branch-coverage % from the last measure_coverage measurement.")
    test_count_standard: Optional[int] = Field(default=None, description="[auto] Number of test functions in the on-disk standard test file.")
    test_loc_standard: Optional[int] = Field(default=None, description="[auto] Lines of code in the on-disk standard test file.")
    target_loc: Optional[int] = Field(default=None, description="[auto] Lines of code in the target source file.")
    target_complexity: Optional[int] = Field(default=None, description="[auto] Cyclomatic-style complexity proxy for the target source file.")
    source_observations: SkipJsonSchema[Dict[str, Any]] = Field(default_factory=dict)
    agent_iterations: Optional[int] = Field(default=None, description="[auto] Total tool calls made during this run.")
    agent_writes: Optional[int] = Field(default=None, description="[auto] edit_tests call count.")
    agent_runs: Optional[int] = Field(default=None, description="[auto] run_tests call count.")
    agent_assess_cov: Optional[int] = Field(default=None, description="[auto] measure_coverage call count.")
    agent_duration_s: Optional[float] = Field(default=None, description="[auto] Wall-clock seconds spent in this generate_coverage_tests call.")
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
        description="[auto] Physical requests whose token usage the provider did not report. The token totals undercount by their usage.",
    )
    input_tokens: Optional[int] = Field(default=None, description="[auto] Total prompt tokens consumed across all model invocations in this run.")
    output_tokens: Optional[int] = Field(default=None, description="[auto] Total completion tokens emitted across all model invocations in this run.")
    cache_hit_tokens: Optional[int] = Field(default=None, description="[auto] Cached-prefix prompt tokens (provider prompt-cache hits), a subset of input_tokens.")
    failure_kind: SkipJsonSchema[Optional[str]] = Field(
        default=None,
        description="[auto] infrastructure when a model request failure ended the run, otherwise unjudged.",
    )
    unknown_cache_requests: SkipJsonSchema[Optional[int]] = Field(
        default=None,
        description="[auto] Requests with reported token usage whose cache split the provider did not report. cache_hit_tokens sums the others.",
    )
    status: Optional[str] = Field(default=None, description="[auto] Outcome category: success | stagnation | tests_failed | context_overflow | budget_exhausted | agent_error.")
    error: Optional[str] = Field(default=None, description="[auto] Error message when status is not success.")


# Agent dependencies

class CoverageTestDeps(BaseModel):
    """Dependencies injected into every tool call."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    target_file_path: str = Field(description="Absolute path to the file under test.")
    target_file_content: str = Field(description="Source code of the file under test.")
    source_dir: str = Field(description="Absolute path to the source directory.")
    test_output_dir: str = Field(description="Directory where test files are written.")
    language: str = Field(description="Programming language (python, java, ...).")
    key_functions: List[str] = Field(default_factory=list)

    _coerce_paths = model_validator(mode="before")(
        agent_helpers.coerce_paths_to_str
    )


# Prompts

# The model sees the language only in the prompt text, which the entry point
# resolves for ``deps.language``.


# The result schema's field descriptions, which a model that skips the
# structured output often echoes.
_PLACEHOLDER_FRAGMENTS = (
    "brief explanation of test strategy and what is covered",
    "list of functions/methods explicitly tested",
    "filename of the on-disk working file",
)


def _looks_placeholder_explanation(text: str) -> bool:
    return agent_helpers.looks_placeholder_text(text, _PLACEHOLDER_FRAGMENTS)


def _looks_placeholder_name(text: str) -> bool:
    """Whether a ``targeted_functions`` item is a placeholder phrase. A name
    may be short, so no length check applies."""
    return agent_helpers.looks_placeholder_text(
        text, _PLACEHOLDER_FRAGMENTS, min_len=0,
    )


# Session state


@dataclass
class TestCoverageSession(agent_helpers.CodingAgentSession):
    """The coverage agent's per-target session state."""
    # The handed-over suite's bytes hash and measure_coverage report, which
    # answer the start's measurement while the file has those bytes.
    stored_measurement: Optional[Tuple[str, str]] = None
    # Measurement caches
    tests_changed_since_measure: Dict[str, bool] = _field(
        default_factory=lambda: {"coverage": True}
    )
    last_coverage_result: str = ""
    # Trend tracking
    prev_coverage_pct: Optional[float] = None
    prev_branch_coverage_pct: Optional[float] = None
    coverage_measurements: int = 0
    # The best values, for the measurements without improvement and the
    # goal: the line and branch values of the measurement with the best
    # goal metric, one suite's pair.
    best_coverage_pct: Optional[float] = None
    best_branch_coverage_pct: Optional[float] = None
    coverage_regression_count: int = 0
    # The latest uncovered line ranges, shown in every turn's state marker.
    last_uncovered_ranges: str = ""
    seen_coverage_goal: bool = False
    coverage_post_goal_count: int = 0
    # The backend of ``deps.language``, resolved at the run's start, through
    # which ``_target_test_file`` finds the target's test file.
    active_backend: Optional[Any] = None
    # Loop / no-progress guard.
    loop_breaker: LoopBreakerState = _field(default_factory=lambda: LoopBreakerState(
        return_action_text="return CoverageTestResult if the coverage-phase exit conditions are met"))


# Graph state


class CoverageGraphState(TypedDict, total=False):
    """The coverage loop's graph state: the conversation's messages, the
    phase's result once it ends and the agent's mode."""
    messages: Annotated[List[BaseMessage], add_messages]
    result: Optional[CoverageTestResult]
    agent_mode: str


# Agent


class CoverageAgent(CodingAgent):
    """The coverage agent, which extends the target's test file for coverage.
    Its tools: the two reads, ``edit_file`` (offered as ``edit_tests``),
    ``run_tests`` (strict green) and ``measure_coverage`` (coverage and the
    uncovered code)."""

    read_tool_names = ("read_source", "read_tests")
    guards_duplicate_test_names = True
    guards_generated_test_validity = True
    edit_tool_name = "edit_tests"
    result_schema = CoverageTestResult
    graph_state_type = CoverageGraphState
    final_tool_name = "emit_coverage_test_result"
    forces_initial_read = True
    forces_initial_measurement = True
    designated_read_name = "read_tests"
    measure_tool_name = "measure_coverage"
    no_measurement_key = "validator_no_coverage_measurement"
    stale_measurement_key = "validator_stale_coverage_measurement"
    initial_agent_mode = "needs_initial_read"

    # Hooks for the shared edit gate.

    def target_path(self) -> Optional[Path]:
        return self._target_test_file(self.deps.test_output_dir, self.deps.target_file_path)

    # ``source_language`` keeps the CodingAgent default.

    # Per-agent file and state helpers.

    def _target_test_file(self, test_dir: str, target_file_path: str) -> Optional[Path]:
        """The target's test file in its test directory: the backend's
        ``find_target_test_file``."""
        return self.test_backend().find_target_test_file(test_dir, target_file_path)

    def _best_goal_metric_value(self) -> Optional[float]:
        """The run's best goal metric value, for the goal, the early stop and recovery."""
        return _goal_metric_value(
            self.session.best_coverage_pct, self.session.best_branch_coverage_pct,
        )

    def _best_coverage_text(self, decimals: int) -> Optional[str]:
        """The best suite's line and branch values, the goal metric and the
        goal, for the state marker and the recovery snapshot. None before
        a measurement."""
        line = self.session.best_coverage_pct
        branch = self.session.best_branch_coverage_pct
        if line is None and branch is None:
            return None

        def pct(v: Optional[float]) -> str:
            return "n/a" if v is None else f"{measurement.floor_pct(v, decimals):.{decimals}f}%"

        if line is not None and branch is None:
            values, metric = f"line {pct(line)}, no branches", "line"
        else:
            values = f"line {pct(line)}, branch {pct(branch)}"
            metric = {
                "line": "line", "branch": "branch",
                "both": "lower of line and branch",
                "either": "higher of line and branch",
            }[_goal_metric()]
        goal = float(_AGENT_CONFIG["goal"])
        return (f"{values}; goal metric ({metric}) "
                f"{pct(self._best_goal_metric_value())}, goal {goal:g}%")

    def _coverage_goal_met(self) -> bool:
        cov_goal = float(_AGENT_CONFIG["goal"])
        best = self._best_goal_metric_value()
        return best is not None and best >= cov_goal

    def _coverage_stalled_below_goal(self) -> bool:
        max_measurements_without_improvement = int(_AGENT_CONFIG["max_measurements_without_improvement"])
        # A maximum of 0 or less disables the stall gate: the phase runs until
        # the max_requests_per_phase limit is used up, never on a lack of improvement.
        if max_measurements_without_improvement <= 0:
            return False
        cov_goal = float(_AGENT_CONFIG["goal"])
        cov_stalled = self.session.measurements_without_improvement >= max_measurements_without_improvement
        best = self._best_goal_metric_value()
        cov_below = best is None or best < cov_goal
        return cov_stalled and cov_below

    def _coverage_stall_hint(self) -> str:
        """Hint appended to measure_coverage output when coverage is stuck."""
        early_stop = bool(_AGENT_CONFIG["allow_early_stop"])
        if self._coverage_goal_met():
            return self.tool_msg("stall_hint_both_goals_met")
        if self._coverage_stalled_below_goal():
            if early_stop:
                return self.tool_msg("stall_hint_both_stalled_return")
            return self.tool_msg("stall_hint_coverage_keep_iterating")
        return ""

    loop_state_fields = tuple((name, name) for name in (
        "measured_suite_hash", "prev_coverage_pct", "prev_branch_coverage_pct",
        "last_coverage_result", "last_uncovered_ranges", "last_suite_exec_ms"))
    loop_text_fields = ("last_coverage_result", "last_uncovered_ranges")
    snapshot_rank_key = staticmethod(_green_snapshot_key)

    def _restore_unmeasured(self, snap: Dict[str, Any], state: Dict[str, Any]) -> None:
        self.session.prev_coverage_pct = snap.get("coverage")
        self.session.prev_branch_coverage_pct = snap.get("branch_coverage")

    def live_scores(self, measured: bool) -> Dict[str, Any]:
        session = self.session
        return {
            "coverage": session.prev_coverage_pct if measured else None,
            "branch_coverage": session.prev_branch_coverage_pct if measured else None,
        }

    def _maybe_update_best_green(self, test_dir: str, target_file_path: Optional[str]) -> None:
        """Snapshot the test file when its key beats the best-green snapshot. A
        measurement of other bytes counts as none, so a snapshot is never
        credited with values it did not earn."""
        if not target_file_path:
            return
        target_test = self._target_test_file(test_dir, target_file_path)
        if target_test is None or not target_test.exists():
            return
        try:
            content = target_test.read_text(encoding="utf-8")
        except OSError:
            return

        measured_live = self.measured_bytes(content)
        cov_fresh = (
            self.session.prev_coverage_pct if measured_live else None
        )
        branch_fresh = (
            self.session.prev_branch_coverage_pct if measured_live else None
        )
        observation = measurement.observe_file(target_test, backend=self.test_backend())
        structural = measurement.record(observation)
        test_count = structural["tests"]

        # The helper also writes the ``.bak`` sidecar, which survives a killed
        # process for diagnosis.
        took = update_best_green(
            self.session.best_green, target_test,
            coverage=cov_fresh, branch_coverage=branch_fresh,
            test_count=test_count,
            test_source_observation=structural,
            exec_time_ms=(
                self.session.last_suite_exec_ms if measured_live else None
            ),
            rank_key=_green_snapshot_key,
            persist_to_disk=True,
            log_fn=lambda msg, level=2: log_print(msg, level=level),
            loop_state=self._snapshot_loop_state(content),
        )
        if took:
            goal_value = _goal_metric_value(cov_fresh, branch_fresh)
            metric = _goal_metric()

            def _pct(v: Optional[float]) -> str:
                return f"{v:.2f}%" if v is not None else "n/a"

            log_print(
                f"[snapshot] New best-green: line={_pct(cov_fresh)} "
                f"branch={_pct(branch_fresh)} "
                f"{metric}={_pct(goal_value)} tests={test_count}",
                level=2,
            )

    def _log_run_summary(self) -> None:
        """End-of-run block summarizing what happened during this run."""
        session = self.session
        log_print("=== Test generation summary ===")
        log_print(f"Test-file writes: {session.writes_done}", level=1)
        log_print(f"Coverage measurements: {session.coverage_measurements}", level=1)
        if (
            session.best_coverage_pct is not None
            or session.best_branch_coverage_pct is not None
        ):
            line_s = (
                f"{session.best_coverage_pct:.1f}%"
                if session.best_coverage_pct is not None else "n/a"
            )
            branch_s = (
                f"{session.best_branch_coverage_pct:.1f}%"
                if session.best_branch_coverage_pct is not None else "n/a"
            )
            log_print(f"Best coverage seen: line {line_s} / branch {branch_s}", level=1)
        best_goal = self._best_goal_metric_value()
        if best_goal is not None:
            goal = float(_AGENT_CONFIG["goal"])
            outcome = "met" if best_goal >= goal else "not met"
            log_print(
                f"Goal metric ({_goal_metric()}): {best_goal:.1f}% vs "
                f"goal {goal:g}% -> {outcome}",
                level=1,
            )
        if (
            session.prev_coverage_pct is not None
            or session.prev_branch_coverage_pct is not None
        ):
            line_f = (
                f"{session.prev_coverage_pct:.1f}%"
                if session.prev_coverage_pct is not None else "n/a"
            )
            branch_f = (
                f"{session.prev_branch_coverage_pct:.1f}%"
                if session.prev_branch_coverage_pct is not None else "n/a"
            )
            log_print(
                f"Final coverage measurement: line {line_f} / branch {branch_f}",
                level=1,
            )
        if session.best_green.restored_from_green:
            log_print("Restored test file from last-green snapshot.", level=1)

    def get_run_metrics(self) -> Dict[str, Any]:
        """The run's counters and latest measurements, for the tests.csv row."""
        session = self.session
        return {
            "tool_calls": session.session_tool_calls,
            "writes": session.writes_done,
            "runs": session.run_tests_calls,
            "assess_cov": session.coverage_measurements,
            "best_line_coverage": session.best_coverage_pct,
            "final_line_coverage": session.prev_coverage_pct,
            "final_branch_coverage": session.prev_branch_coverage_pct,
        }

    def build_failure_result(self, error_msg: str) -> CoverageTestResult:
        """The result of a run that produced none: after restoring the
        best-green snapshot, the target's non-empty test file on disk, which
        later phases can run, else ``test_failed.py`` (the coverage phase
        failed for this target).
        """
        test_output_dir = self.deps.test_output_dir
        target_file_path = self.deps.target_file_path
        if target_file_path:
            try:
                self._maybe_restore_last_green(test_output_dir, target_file_path)
                target_test = self._target_test_file(test_output_dir, target_file_path)
                if target_test is not None and target_test.exists() and target_test.stat().st_size > 0:
                    return CoverageTestResult(
                        test_file_name=target_test.name,
                        explanation=(
                            f"Test generation interrupted ({error_msg}). The "
                            "on-disk test file is the last-green snapshot, and "
                            "the downstream phases can run against it."
                        ),
                        targeted_functions=[],
                    )
            except Exception:
                pass
        return CoverageTestResult(
            test_file_name=FAILED_MARKER,
            explanation=f"Generation failed with no recoverable green snapshot: {error_msg}",
            targeted_functions=[],
        )

    def _synthesize_early_stop_result(
        self, reason: str, target_file_path: Optional[str] = None,
    ) -> CoverageTestResult:
        """The result from session state when a designed stop fires, after
        restoring the best-green snapshot, so the file on disk is the best
        green run."""
        test_output_dir = self.deps.test_output_dir
        session = self.session
        target_test_name: Optional[str] = None
        if target_file_path:
            try:
                self._maybe_restore_last_green(test_output_dir, target_file_path)
                target_test = self._target_test_file(test_output_dir, target_file_path)
                if target_test is not None and target_test.exists() and target_test.stat().st_size > 0:
                    target_test_name = target_test.name
            except Exception:
                pass

        parts: List[str] = [f"Stopped early ({reason})."]
        best = self._best_coverage_text(1)
        if best is not None:
            parts.append(f"Final coverage: {best}.")
        parts.append(
            f"{session.writes_done} writes, "
            f"{session.coverage_measurements} coverage measurements."
        )
        explanation = " ".join(parts)

        return CoverageTestResult(
            test_file_name=target_test_name or FAILED_MARKER,
            explanation=explanation,
            targeted_functions=[],
        )

    def _maybe_restore_last_green(
        self, test_output_dir: str, target_file_path: Optional[str] = None,
        *, force: bool = False,
    ) -> bool:
        """Revert the test file to the best green snapshot by
        ``_green_snapshot_key`` (not the latest) when the live file is red or
        green without a strictly higher goal value. Equal scores keep the
        snapshot. ``force`` skips the comparison: the in-loop regression revert,
        whose fresh measurement found the regression, passes it, since the
        cached ``prev_coverage_pct`` may be stale. The terminal restore does not.
        Returns whether it wrote the snapshot."""
        if self.session.best_green.snapshot is None:
            return False  # no green snapshot was ever captured this run
        target = (
            self._target_test_file(test_output_dir, target_file_path)
            if target_file_path else None
        )
        if not (self._restore_snapshot(target) if force
                else self._settle_best_green(target)):
            return False
        snap = self.session.best_green.snapshot
        log_print(
            f"[restore] Reverted test file to best-green snapshot. "
            f"Snapshot: coverage="
            f"{snap.get('coverage') if snap.get('coverage') is not None else 'n/a'}, "
            f"tests={snap.get('test_count')}.",
            level=1,
        )
        return True

    def _maybe_revert_on_regression(
        self,
        target_file_path: str,
        is_regression: bool,
        current_goal: Optional[float] = None,
        best_goal: Optional[float] = None,
    ) -> Optional[str]:
        """The in-loop regression revert: a green measurement strictly below the
        best goal metric is a regression (an equal one is lateral), and after
        ``restore_count`` of them the file returns to the best-green snapshot
        (``restore_count <= 0`` turns this off, while the terminal restore
        still runs). Returns the model-facing note of a revert, else None."""
        session = self.session
        should, session.coverage_regression_count = regression_revert_decision(
            regression_count=session.coverage_regression_count,
            is_regression=is_regression,
            n_restore=int(_AGENT_CONFIG["restore_count"]),
        )
        if not should or session.best_green.snapshot is None:
            return None
        # Forced: the fresh measurement found the regression, and the model must
        # never read "[REVERTED]" while the file keeps the regressed content,
        # nor when nothing was restored.
        if not self._maybe_restore_last_green(
            self.deps.test_output_dir, target_file_path, force=True,
        ):
            return None
        snap = session.best_green.snapshot
        cur = f"{measurement.floor_pct(current_goal):.1f}" if current_goal is not None else "n/a"
        best = f"{measurement.floor_pct(best_goal):.1f}" if best_goal is not None else "n/a"
        return self.tool_msg(
            "revert_on_regression",
            current=cur, best=best, test_count=str(snap.get("test_count")),
        )

    def build_state_marker(self, _s: Any = None) -> str:
        """The state marker of every model request: the task, the test status,
        coverage with its no-improvement count, edits since the run and the
        uncovered ranges (the tool list is the graph's gating, not repeated)."""
        session = self.session
        if session.last_tests_passed:
            tests_field = "green"
        elif session.last_run_tests_result:
            tests_field = "red"
        else:
            tests_field = "untested"

        cov_field = self.stalled_metric(self._best_coverage_text(0), str)

        edits_field = "1+" if session.tests_changed_since_run_tests else "0"

        head = self.task_head()
        line1 = (
            f"[state | {head}tests={tests_field} | coverage={cov_field} | "
            f"edits_since_run={edits_field}]"
        )

        if session.last_uncovered_ranges:
            ranges = session.last_uncovered_ranges
            ranges = _limits.cut_text(ranges, _limits.value("state_marker_chars"),
                                      what="uncovered ranges", keep="head", sep=" ")
            return f"{line1}\n[uncovered: {ranges}]"
        return line1

    # Tools: reads.

    def read_source(self, start_line: int = 1, end_line: Optional[int] = None) -> str:
        """Read the source of the file under test (line-numbered).

        This tool becomes available after the mandatory initial ``read_tests``
        call returns. Use it when coverage feedback does not contain enough
        source context for a targeted edit, or to refresh after a context
        trim. Pass ``start_line`` / ``end_line`` to page a file past the size
        limit. An immediate consecutive whole re-read (no other tool call
        between) returns a short cached header; calling any other tool resets
        the counter.
        """
        return self._source_read(start_line, end_line)

    # Tools: writes.

    # ``edit_file`` is the shared CodingAgent gate, driven by the hooks above.

    # run_tests and its strict-green machinery.

    def timeout_suspect_path(self) -> Optional[Path]:
        return self.test_backend().find_target_test_file(
            self.deps.test_output_dir, self.deps.target_file_path,
        )

    def synthesize_stop_result(self, reason: str) -> Any:
        return self._synthesize_early_stop_result(reason, self.deps.target_file_path)

    # run_tests hooks. The skeleton and classify core are on CodingAgent.

    def before_run_tests(self) -> Optional[str]:
        deps = self.deps
        session = self.session
        if not session.last_run_tests_result:
            session.last_run_tests_result = (
                "(run_tests was called but ended before it produced "
                "output. Edit the working file and call run_tests again.)"
            )
        backend = self.test_backend()
        if backend is None:
            msg = self.tool_msg(
                "tests_unsupported_language", language=deps.language,
            )
            session.last_tests_passed = False
            session.last_run_tests_result = msg
            return msg
        target_test = backend.find_target_test_file(
            deps.test_output_dir, deps.target_file_path,
        )
        if target_test is None:
            msg = self.tool_msg(
                "tests_no_test_file",
                target_name=Path(deps.target_file_path).name,
                test_dir=deps.test_output_dir,
            )
            session.last_tests_passed = False
            session.last_run_tests_result = msg
            return msg
        return None

    def test_paths(self) -> List[str]:
        deps = self.deps
        target_test = self.test_backend().find_target_test_file(
            deps.test_output_dir, deps.target_file_path,
        )
        return [str(target_test)]

    restored_file_phrase = "test file"

    def phase_log(self, message: str, level: int = 0) -> None:
        log_print(message, level=level)

    def after_green_run(self) -> None:
        self._maybe_update_best_green(self.deps.test_output_dir, self.deps.target_file_path)

    def failed_run_target(self) -> Optional[Path]:
        return self._target_test_file(self.deps.test_output_dir, self.deps.target_file_path)

    def snapshot_score_text(self, snap: Dict[str, Any]) -> str:
        coverage = snap.get("coverage")
        return (f"coverage={coverage if coverage is not None else 'n/a'}, "
                f"tests={snap.get('test_count')}")

    def _coverage_reply(self, final_output: str, line_pct: Optional[float],
                        branch_pct: Optional[float], prev_best_goal: Optional[float]) -> str:
        """The loop's reply to a measurement, ``final_output`` with the stall
        hint, kept as the cached result. The fresh value may make the file the
        best-green snapshot, a fall below the best goal metric
        ``prev_best_goal`` reverts it, and the stop check runs last."""
        deps = self.deps
        session = self.session
        hint = self._coverage_stall_hint()
        if hint:
            final_output += "\n\n" + hint

        session.last_coverage_result = final_output
        session.tests_changed_since_measure["coverage"] = False

        # A fresh coverage value may make the file the best-green snapshot.
        self._maybe_update_best_green(deps.test_output_dir, deps.target_file_path)

        # In-loop regression revert.
        current_goal = _goal_metric_value(line_pct, branch_pct)
        is_regression = (
            current_goal is not None
            and prev_best_goal is not None
            and current_goal < prev_best_goal
        )
        revert_note = self._maybe_revert_on_regression(
            deps.target_file_path, is_regression,
            current_goal=current_goal, best_goal=prev_best_goal,
        )
        if revert_note:
            final_output += "\n\n" + revert_note

        stop_msg = self._maybe_set_early_stop(source_tool="measure_coverage")
        if stop_msg is not None:
            return self.track_output(final_output + "\n\n" + stop_msg)
        return self.track_output(final_output)

    def _record_coverage(self, line_pct: Optional[float], branch_pct: Optional[float],
                         target_test: Path, initial: bool) -> Tuple[Optional[float], ...]:
        """Record a measurement's values: the previous line and branch
        coverage, the best pair, the count and the measured suite's hash, and
        the streak of measurements without improvement (not for the start's
        measurement). Returns the best goal metric and the line and branch
        coverage from before this measurement."""
        session = self.session
        # The previous values, for the trend log and the no-improvement tracker.
        prev_best_goal = self._best_goal_metric_value()
        prev_line_pct = session.prev_coverage_pct
        prev_branch_pct = session.prev_branch_coverage_pct

        if branch_pct is not None:
            session.prev_branch_coverage_pct = branch_pct

        # The best values are the pair of the suite with the best goal
        # metric: the best line and the best branch value of two suites
        # would credit the goal with a pair no suite reached.
        new_goal = _goal_metric_value(line_pct, branch_pct)
        if new_goal is not None and (prev_best_goal is None or new_goal > prev_best_goal):
            session.best_coverage_pct = line_pct
            session.best_branch_coverage_pct = branch_pct

        if line_pct is not None:
            session.prev_coverage_pct = line_pct
            if not initial:
                session.coverage_measurements += 1
            # The bytes this measurement scored. prev_* count for a file only
            # while its bytes match.
            try:
                session.measured_suite_hash = source_sha256(
                    target_test.read_text(encoding="utf-8"))
            except OSError:
                session.measured_suite_hash = None

        if (line_pct is not None or branch_pct is not None) and not initial:
            if self._resets_streak(
                    prev_best_goal, self._best_goal_metric_value(),
                    min_points=float(_AGENT_CONFIG["min_improvement_points"]),
                    minimum=float(_AGENT_CONFIG["minimum"])):
                session.measurements_without_improvement = 0
            else:
                session.measurements_without_improvement += 1
        return prev_best_goal, prev_line_pct, prev_branch_pct

    def _log_coverage_progress(self, line_pct: Optional[float], branch_pct: Optional[float],
                               prev_line_pct: Optional[float],
                               prev_branch_pct: Optional[float]) -> None:
        """Log a measurement's progress line against the goal and its line and
        branch trend lines against the previous measurement."""
        session = self.session
        prev_goal_pct = _goal_metric_value(prev_line_pct, prev_branch_pct)
        if line_pct is not None or branch_pct is not None:
            metric_parts = []
            if line_pct is not None:
                metric_parts.append(f"line {measurement.floor_pct(line_pct):.0f}%")
            if branch_pct is not None:
                metric_parts.append(f"branch {measurement.floor_pct(branch_pct):.0f}%")
            best_goal = self._best_goal_metric_value()
            cur_goal = _goal_metric_value(line_pct, branch_pct)
            cov_goal = float(_AGENT_CONFIG["goal"])
            # The best and the goal are on the goal metric, named so the line does
            # not read as line coverage against the goal.
            metric = _goal_metric()
            self.log_progress(
                session.coverage_measurements,
                " ".join(metric_parts),
                (f"{measurement.floor_pct(best_goal):.0f}% ({metric})"
                 if best_goal is not None else "none"),
                target=f"goal {cov_goal:g}% ({metric})",
                delta=(
                    f"{cur_goal - prev_goal_pct:+.0f}pp vs prev"
                    if cur_goal is not None and prev_goal_pct is not None else ""
                ),
            )
        if line_pct is not None:
            log_print(_format_trend("Coverage (line)", prev_line_pct, line_pct), level=1)
        if branch_pct is not None:
            log_print(_format_trend("Coverage (branch)", prev_branch_pct, branch_pct), level=1)

    def _normal_run_reply(self) -> Optional[str]:
        """The prelude's reply that ends the measurement, else None. Coverage
        of a red suite is wrong, since the lines behind a failure look
        uncovered, so a red suite answers with its failure. A prelude that
        reverted the file to the best-green snapshot answers with the
        revert."""
        session = self.session
        restores_before = session.best_green.restores
        tests_passed, tests_output = self._ensure_tests_evaluated(
            log_prefix="measure_coverage",
        )
        if session.best_green.restores != restores_before:
            # The prelude reverted the file to the best-green snapshot and
            # restored the state its bytes left: report the revert and
            # measure nothing the snapshot did not already measure.
            log_print(
                "[measure_coverage] auto-prelude reverted to the best-green "
                "snapshot: returning the revert (no coverage measurement).",
                level=1,
            )
            return self.track_output(tests_output)
        if not tests_passed:
            msg = self.tool_msg("measure_coverage_red_suite", tests_output=tests_output)
            log_print(
                "[measure_coverage] auto-prelude RED: returning failure "
                "detail (no coverage measurement).",
                level=1,
            )
            session.last_coverage_result = msg
            return self.track_output(msg)
        return None

    def _measured_run(self) -> Tuple[Any, Optional[Path], Optional[str]]:
        """The working file's run under the coverage tool, as ``(run, working
        file, None)``, or ``(None, None, reply)`` when the working file is
        missing or the coverage tool refuses the run (its timeout, a red run,
        one whose coverage counts the tool could not read, or one without a
        test result after a green normal run)."""
        deps = self.deps
        test_dir = deps.test_output_dir
        source_dir = deps.source_dir
        target_file = deps.target_file_path
        backend = languages.get_backend(deps.language)

        target_test = backend.find_target_test_file(test_dir, target_file)
        if target_test is None:
            return None, None, self.track_output(self.tool_msg(
                "measure_coverage_no_test_file",
                name=Path(target_file).name,
                test_dir=test_dir,
            ))

        cov_timeout = int(_AGENT_CONFIG["suite_run_timeout_seconds"])
        res = backend.run_test_files(
            [str(target_test)], source_dir,
            timeout=cov_timeout, measure_coverage_of=target_file,
            **self.journal_request(backend, source_dir),
        )
        if res.timed_out:
            # The normal run passed in time: only the coverage tool's run did
            # not finish, so the tool could not measure the suite. The suite
            # is not too slow and no timeout counts. The running test, when
            # the journal names it, is the test that stopped the tool.
            from testuary import results

            log_print("WARNING: " + results.note(
                "coverage", "coverage.suite_run_timeout_seconds", cov_timeout))
            active = getattr(res, "active_at_end", None)
            msg = (self.tool_msg("measure_coverage_timeout_when_measured_named",
                                 timeout=cov_timeout, test=active) if active
                   else self.tool_msg("measure_coverage_timeout_when_measured",
                                      timeout=cov_timeout))
            return None, None, self.track_output(self._coverage_refusal(
                msg, [active] if active else [], target_test,
                f"the measured run did not finish within {cov_timeout}s after a "
                "green normal run"))
        if not res.ok:
            # The measured run failed after a green normal run (a test failing
            # only under the coverage tool, a run that collected nothing or
            # ended without results there, coverage counts the tool could not
            # read, or flaky): the tool could not measure the suite.
            stopping, red = agent_helpers.stopping_tests(res)
            if red:
                msg = self.tool_msg(
                    "measure_coverage_red_when_measured",
                    tests_output=self._canonical_view(res, None, res.returncode or 1))
                what = "the measured run is RED after a green normal run"
            elif agent_helpers.coverage_counts_inconsistent(res):
                msg = self.tool_msg("measure_coverage_counts_inconsistent")
                what = ("the coverage tool could not read the measured run's coverage "
                        "counts after a green normal run")
            else:
                msg = self.tool_msg("measure_coverage_unmeasured", output=_limits.cut_text(
                    res.output, _limits.value("empty_run_output_chars"),
                    what="the test output", keep="tail"))
                what = ("the measured run ended without a test result after a green "
                        "normal run")
            return None, None, self.track_output(self._coverage_refusal(msg, stopping, target_test, what))
        return res, target_test, None

    def _coverage_refusal(self, msg: str, stopping: List[str], target_test: Path,
                          what: str) -> str:
        """A measurement the coverage tool refused after a green normal run
        (``what``, for the log): the line's record names the tests that
        stopped the tool, the loop returns to editing with the refusal as the
        tool's, and it counts as a failed run, so after
        max_failed_test_attempts of them the best-green snapshot returns."""
        # The tests that stopped the coverage tool go into the line's record,
        # which the results page reads.
        log_print(f"[measure_coverage] {what}: no coverage recorded.",
                  level=1, tool="coverage", stopping_tests=list(stopping))
        self.session.last_coverage_result = msg
        self.record_refusal("coverage", msg)
        # A restored snapshot brings back the state of its own bytes.
        max_attempts, should_restore = self.failed_run_due()
        if should_restore and self._restore_snapshot(target_test):
            log_print(
                f"[restore] {max_attempts} consecutive failed runs - reverted "
                "test file to the best-green snapshot.",
                level=1,
            )
            msg = self.tool_msg("tests_restore_note", max_attempts=max_attempts, output=msg)
        return msg

    def _ensure_tests_evaluated(self, *, log_prefix: str) -> tuple[bool, str]:
        """``(passed, output)`` of the test file on disk for ``measure_coverage``'s
        prelude: a run now when it changed since the last run_tests, else the
        cached state."""
        if self.session.tests_changed_since_run_tests:
            return self._run_tests_uncached(log_prefix)
        cached_output = self.session.last_run_tests_result or "(no earlier run_tests result is recorded)"
        return self.session.last_tests_passed, cached_output

    def measure_coverage(self) -> str:
        """Measure line/branch coverage of the target file AND show the source
        code of any uncovered lines/branches in one call.

        The first part of the return is the numeric summary (line %, branch %,
        missing-line count). The second part is the actual source code of the
        uncovered regions with context, so you can write targeted tests
        without needing a separate tool."""
        session = self.session
        self.log_thinking("measure_coverage")
        if self.in_start_measurement():
            return self._initial_measurement()
        # Refused while the tests are known red and the file is unchanged: the
        # prelude would report the same failures.
        if (
            not session.last_tests_passed
            and not session.tests_changed_since_run_tests
            and session.last_run_tests_result
        ):
            log_print(
                "[measure_coverage] REFUSED: tests known red, file unchanged "
                "since last run_tests. Edit required before measurement.",
                level=1,
            )
            msg = self.tool_msg(
                ("measure_coverage_refused_known_refusal" if self.refused_by()
                 else "measure_coverage_refused_known_red"),
                last_run_tests_result=session.last_run_tests_result,
                refused_by=self.refused_by_phrase(),
            )
            return self.track_output(msg)
        # Skip-unchanged gate (same shape as run_tests).
        if not session.tests_changed_since_measure["coverage"] and session.last_coverage_result:
            # A cached call can optionally count as a no-improvement event.
            self.count_cached_measurement(session.best_coverage_pct is not None, log_print)
            log_print(
                "[measure_coverage] Skipped: test file unchanged since the "
                "last measurement. Returning cached result.",
                level=1,
            )
            # Deterministic stop check after advancing the counter.
            stop_msg = self._maybe_set_early_stop(source_tool="measure_coverage[cached]")
            cached_msg_key = (
                "measure_coverage_cached_rejected"
                if session.best_coverage_pct is not None
                else "measure_coverage_cached_red"
            )
            cached_msg = agent_helpers.loop_breaker_check(
                session.loop_breaker, "coverage",
                self.tool_msg(cached_msg_key, last_result=session.last_coverage_result),
            )
            if stop_msg is not None:
                return self.track_output(cached_msg + "\n\n" + stop_msg)
            return self.track_output(cached_msg)
        return self._measure_coverage_run()

    def _initial_measurement(self) -> str:
        """The start's measurement, the model's first measure_coverage call
        after the designated read: the stored measurement of the exact suite
        (the handed-over baseline's, matched by its bytes' hash) when one
        exists, else a measurement of the suite now. Either answer is the
        report the tool gives for that suite. The loop then enters Revise."""
        session = self.session
        session.initial_measured = True
        target_test = self._target_test_file(
            self.deps.test_output_dir, self.deps.target_file_path)
        digest = None
        if target_test is not None and target_test.exists():
            digest = source_sha256(target_test.read_text(encoding="utf-8"))
        stored = session.stored_measurement
        if stored is not None and digest is not None and stored[0] == digest:
            log_print("[measure_coverage] start: the stored measurement of the "
                      "suite answers (no run)", level=1)
            return self.track_output(stored[1])
        log_print("[measure_coverage] start: no stored measurement of these "
                  "bytes, measuring the suite", level=1)
        return self._measure_coverage_run(initial=True)

    def _measure_coverage_run(self, initial: bool = False) -> str:
        """The measurement after the gates: run the suite under coverage,
        update the session and answer. ``initial`` is the start's
        measurement of a suite with no stored measurement: the report alone,
        without the loop's counters, notes and stop checks."""
        deps = self.deps
        session = self.session
        # Set first, with a cached result, so after an early return below the
        # next call still answers from the cache.
        session.tests_changed_since_measure["coverage"] = False
        if not session.last_coverage_result:
            session.last_coverage_result = self.tool_msg("measure_coverage_no_prior_result")

        reply = self._normal_run_reply()
        if reply is not None:
            return reply

        res, target_test, reply = self._measured_run()
        if reply is not None:
            return reply
        session.last_suite_exec_ms = (
            res.elapsed_s * 1000.0 if res.elapsed_s is not None else None
        )

        line_pct: Optional[float] = res.line_coverage
        branch_pct: Optional[float] = res.branch_coverage
        output, ranges = _coverage_summary(
            line_pct, branch_pct, res.missing_lines, res.missing_branches,
            res.output)
        if ranges is not None:
            session.last_uncovered_ranges = ranges

        if line_pct is None and branch_pct is None:
            diagnosis = _jacoco_miss_line(res.output) or "the run measured no coverage"
            log_print(f"[measure_coverage] {diagnosis}", level=1)

        prev_best_goal, prev_line_pct, prev_branch_pct = self._record_coverage(
            line_pct, branch_pct, target_test, initial)
        self._log_coverage_progress(line_pct, branch_pct, prev_line_pct, prev_branch_pct)

        final_output = _coverage_report(
            deps.target_file_path, output, res.missing_lines, res.missing_branches)
        if initial:
            # The start's measurement: the report alone. The loop's counters,
            # notes and stop checks begin with the model's own edits. Its
            # value credits the handed-over suite as the best-green snapshot.
            session.last_coverage_result = final_output
            session.tests_changed_since_measure["coverage"] = False
            self._maybe_update_best_green(deps.test_output_dir, deps.target_file_path)
            return self.track_output(final_output)

        return self._coverage_reply(final_output, line_pct, branch_pct, prev_best_goal)

    def _maybe_set_early_stop(self, *, source_tool: str) -> Optional[str]:
        """The directive of a designed end, with the canonical result on
        ``session.early_stop_result``, or None. In priority order: the coverage
        goal reached and ``extra_measurements_after_goal`` used up. Coverage did not improve
        below the goal with ``allow_early_stop``. At the goal while the
        latest green run carries a STANDS ALONE note, the end waits: the reply
        says so, and the wait spends no post-goal measurement.
        """
        deps = self.deps
        session = self.session
        cov_goal = float(_AGENT_CONFIG["goal"])
        extra_measurements_after_goal = int(_AGENT_CONFIG["extra_measurements_after_goal"])
        max_measurements_without_improvement = int(_AGENT_CONFIG["max_measurements_without_improvement"])
        early_stop = bool(_AGENT_CONFIG["allow_early_stop"])

        def _set_stop(reason: str) -> str:
            return self.designed_stop(self._synthesize_early_stop_result(
                reason, deps.target_file_path,
            ), reason)

        # Priority 1: goal reached + polish exhausted. Routed through
        # ``_best_goal_metric_value`` so it respects goal_metric.
        best_for_goal = self._best_goal_metric_value()
        if best_for_goal is not None and best_for_goal >= cov_goal:
            if session.stands_alone_noted:
                log_print(
                    f"[{source_tool}] goal reached ({best_for_goal:.1f}% >= "
                    f"{cov_goal:g}%), but the latest green run carries a "
                    "STANDS ALONE note: the end waits.",
                    level=1,
                )
                return self.goal_waits_reply()
            if not session.seen_coverage_goal:
                session.seen_coverage_goal = True
                if extra_measurements_after_goal == 0:
                    log_print(
                        f"[{source_tool}] FRAMEWORK STOP: goal reached "
                        f"({best_for_goal:.1f}% >= {cov_goal:g}%) and "
                        f"extra_measurements_after_goal=0 -> handing off to "
                        f"the mutation phase.",
                        level=1,
                    )
                    return _set_stop("coverage_goal_met")
                log_print(
                    f"[{source_tool}] goal reached "
                    f"({best_for_goal:.1f}% >= {cov_goal:g}%). "
                    f"Polish limit: {extra_measurements_after_goal} more measurements.",
                    level=1,
                )
            else:
                session.coverage_post_goal_count += 1
                if session.coverage_post_goal_count >= extra_measurements_after_goal:
                    log_print(
                        f"[{source_tool}] FRAMEWORK STOP: post-goal polish "
                        f"({session.coverage_post_goal_count}/{extra_measurements_after_goal}) "
                        f"exhausted -> handing off to the mutation phase.",
                        level=1,
                    )
                    return _set_stop("coverage_goal_met")

        # Priority 2: measurements without improvement below goal.
        below_goal = best_for_goal is None or best_for_goal < cov_goal
        if (
            early_stop
            and max_measurements_without_improvement > 0
            and below_goal
            and session.measurements_without_improvement >= max_measurements_without_improvement
        ):
            log_print(
                f"[{source_tool}] FRAMEWORK STOP: coverage stagnation "
                f"({session.measurements_without_improvement}/{max_measurements_without_improvement}) below goal -> "
                "handing off to the mutation phase.",
                level=1,
            )
            return _set_stop("coverage_stagnation")

        return None

    # Workflow phase classifier.

    def tests_changed_since_measurement(self) -> bool:
        return self.session.tests_changed_since_measure.get("coverage", True)

    def set_tests_changed(self, changed: bool) -> None:
        self.session.tests_changed_since_measure["coverage"] = changed

    def last_measured_value(self) -> Optional[float]:
        return self.session.prev_coverage_pct

    def classify_log_fields(self) -> Dict[str, Any]:
        s = self.session
        return {
            "edits_since_run": s.tests_changed_since_run_tests,
            "edits_since_assess": s.tests_changed_since_measure.get("coverage", True),
        }

    # run_phase hooks.

    def phase_label(self) -> str:
        return "coverage"

    def chat_metadata(self) -> Dict[str, Any]:
        return {
            "target_file": self.deps.target_file_path,
            "test_output_dir": self.deps.test_output_dir,
        }

    def agent_start_fields(self) -> Dict[str, Any]:
        cfg = self.agent_config()
        model_settings = self.phase_model_settings()
        model_values = model_settings.values if model_settings is not None else cfg
        return {
            "target_file": self.deps.target_file_path,
            "language": self.deps.language,
            "key_functions": self.deps.key_functions,
            "max_requests_per_phase": cfg["max_requests_per_phase"],
            "preserve_reasoning_content": bool(
                model_values.get("preserve_reasoning_content", True)
            ),
            "reasoning_field_names": list(self.reasoning_fields()),
        }

    def agent_end_fields(self) -> Dict[str, Any]:
        s = self.session
        return {
            "coverage_measurements": s.coverage_measurements,
            "writes_done": s.writes_done,
            "run_tests_calls": s.run_tests_calls,
            "best_coverage_pct": s.best_coverage_pct,
            "last_tests_passed": s.last_tests_passed,
            "restored_from_green": s.best_green.restored_from_green,
        }

    def finalize_result(
        self, output: Any, *, error_msg: Optional[str], duration_ms: float,
        usage_tracker: Any, final_messages: List[Any],
    ) -> Any:
        deps = self.deps
        _cleanup_temp_files(deps.test_output_dir)
        self._maybe_restore_last_green(
            deps.test_output_dir, target_file_path=deps.target_file_path,
        )
        self.assert_target_has_no_shadowed_definitions("coverage final")
        self._log_run_summary()
        if output is None:
            return output
        metrics = self.get_run_metrics()
        # The name is the test file's on disk (the sentinel when none is
        # non-empty), never the model's, which could steer the scans.
        target_test = self._target_test_file(
            deps.test_output_dir, deps.target_file_path,
        )
        artifact_exists = (
            target_test is not None and target_test.exists()
            and target_test.stat().st_size > 0
        )
        real_name = target_test.name if artifact_exists else FAILED_MARKER
        test_path_on_disk = (
            target_test if artifact_exists
            else Path(deps.test_output_dir) / real_name
        )
        test_structure = measurement.record(
            measurement.observe_file(test_path_on_disk, backend=self.test_backend()),
        )
        target_structure = measurement.record(measurement.observe_file(
            deps.target_file_path, backend=self.test_backend(),
            read_text=project.read_project_text,
        ))
        test_count = test_structure["tests"]
        test_loc = test_structure["code_lines"]
        target_loc = target_structure["code_lines"]
        target_complexity = (
            measurement.compute_file_complexity(deps.target_file_path)
            if target_structure["status"] == "completed" else None
        )
        has_coverage = metrics["final_line_coverage"] is not None
        stop_reason = self.session.early_stop_reason
        if not artifact_exists:
            derived_status = "tests_failed"
        elif error_msg:
            # agent_error, budget_exhausted (the suite is valid, the limit was
            # reached) or context_overflow.
            derived_status = self.classify_error_status(error_msg, self._limit_reached)
        elif stop_reason:
            derived_status = stop_status(stop_reason, measured=self._best_green_measured("coverage"))
        elif not has_coverage:
            derived_status = "tests_failed"
        else:
            derived_status = "success"
        output = output.model_copy(update={
            "test_file_name": real_name,
            # At instrument precision: the consolidation uses them as strict
            # floors, where display rounding would make a false regression.
            "line_coverage": metrics["final_line_coverage"],
            "branch_coverage": metrics["final_branch_coverage"],
            "test_count_standard": test_count,
            "source_observations": {"delivered": test_structure, "target": target_structure},
            "test_loc_standard": test_loc,
            "target_loc": target_loc,
            "target_complexity": target_complexity,
            "agent_iterations": metrics["tool_calls"],
            "agent_writes": metrics["writes"],
            "agent_runs": metrics["runs"],
            "agent_assess_cov": metrics["assess_cov"],
            "status": derived_status,
            "error": error_msg or (
                stop_reason if stop_reason and derived_status != "success"
                else ""
            ),
        })
        return super().finalize_result(
            output, error_msg=error_msg, duration_ms=duration_ms,
            usage_tracker=usage_tracker, final_messages=final_messages,
        )

    # Output validator hooks. The shared base owns the acceptance flow.

    def agent_config(self) -> Dict[str, Any]:
        return _AGENT_CONFIG

    def validate_result_fields(self, parsed: CoverageTestResult) -> Optional[str]:
        if _looks_placeholder_explanation(parsed.explanation):
            log_print(
                f"[validator] Rejected explanation as placeholder: "
                f"{parsed.explanation!r}",
                level=1,
            )
            return self.tool_msg("validator_placeholder_explanation")
        if not parsed.targeted_functions:
            log_print("[validator] Rejected: targeted_functions is empty.", level=1)
            return self.tool_msg("validator_empty_targeted_functions")
        placeholder_names = [
            n for n in parsed.targeted_functions if _looks_placeholder_name(n)
        ]
        if placeholder_names:
            log_print(
                f"[validator] Rejected targeted_functions placeholder entries: "
                f"{placeholder_names!r}",
                level=1,
            )
            return self.tool_msg(
                "validator_placeholder_targeted_functions",
                names=repr(placeholder_names),
            )
        return None

    def goal_met(self) -> bool:
        return self._coverage_goal_met()

    def measurement_count(self) -> int:
        return self.session.coverage_measurements

    def early_stop_reject_message(self) -> str:
        return self.tool_msg(
            "validator_stagnation_not_exhausted",
            best=self._best_coverage_text(1) or "no measurement yet")

    # The recovery prompt's state block. The base builds the retry prompt.

    def recovery_state_block(self) -> str:
        best = self._best_coverage_text(1)
        return self.working_state_block(
            None if best is None else f"  Best coverage:     {best}", "coverage")

# Entry point


def _seed_baseline(session: "TestCoverageSession", baseline: dict,
                   target_file: Optional[str] = None) -> None:
    """Seed the best-green snapshot with the starter phase's baseline, so a
    regression restores it: the handed-over suite is editable but never lost
    to a bad first edit. A baseline whose measurement names its uncovered
    lines and branches is stored as measure_coverage's report for those
    bytes, which answers the start's measurement."""
    session.best_green.snapshot = {
        "content": baseline["content"],
        "coverage": baseline.get("line_coverage"),
        "branch_coverage": baseline.get("branch_coverage"),
        "test_count": baseline.get("test_count"),
        "exec_time_ms": (
            baseline["suite_seconds"] * 1000.0
            if baseline.get("suite_seconds") is not None else None
        ),
        "snapshot_clock_s": time.monotonic(),
    }
    measured = baseline.get("line_coverage") is not None
    suite_ms = session.best_green.snapshot["exec_time_ms"]
    baseline_hash = source_sha256(baseline["content"])
    report, ranges = "", ""
    if (measured and target_file is not None
            and baseline.get("missing_lines") is not None
            and baseline.get("missing_branches") is not None):
        summary, ranges = _coverage_summary(
            baseline["line_coverage"], baseline.get("branch_coverage"),
            baseline["missing_lines"], baseline["missing_branches"])
        ranges = ranges or ""
        report = _coverage_report(target_file, summary, baseline["missing_lines"],
                                  baseline["missing_branches"])
        session.stored_measurement = (baseline_hash, report)
        session.last_coverage_result = report
        session.last_uncovered_ranges = ranges
    session.best_green.snapshot["loop_state"] = (
        {
            "measured": True,
            "measured_suite_hash": baseline_hash,
            "prev_coverage_pct": float(baseline["line_coverage"]),
            "prev_branch_coverage_pct": (
                float(baseline["branch_coverage"])
                if baseline.get("branch_coverage") is not None else None
            ),
            "last_coverage_result": report,
            "last_uncovered_ranges": ranges,
            "last_suite_exec_ms": suite_ms,
        }
        if measured else {"measured": False}
    )
    if baseline.get("line_coverage") is not None:
        line = float(baseline["line_coverage"])
        session.best_coverage_pct = line
        session.prev_coverage_pct = line
    if baseline.get("branch_coverage") is not None:
        branch = float(baseline["branch_coverage"])
        session.best_branch_coverage_pct = branch
        session.prev_branch_coverage_pct = branch
    session.last_tests_passed = True
    session.tests_changed_since_run_tests = False
    # The loop opens in Revise either way. A baseline the framework measured
    # is recorded as measured, one it could not measure stays unmeasured.
    if measured:
        session.measured_suite_hash = baseline_hash
        session.tests_changed_since_measure["coverage"] = False


@_limits.in_phase("coverage")
def generate_coverage_tests(
    project_name: str,
    top_file: dict,
    test_output_dir: str,
    baseline: Optional[dict] = None,
    *, prepared_models=None,
) -> CoverageTestResult:
    """Extend the target's starter suite with coverage feedback. ``baseline``
    (the suite, its measured coverage, suite time and test count) seeds the
    best-green state, so a regression restores that suite. Raises
    RuntimeError without a non-empty baseline."""
    if not baseline or not str(baseline.get("content") or "").strip():
        raise RuntimeError(
            "coverage requires a nonempty validated suite from the starter "
            "phase and stopped before prompt construction or a model call"
        )

    file_path = top_file["file_path"]
    language = languages.detect_language(file_path)
    key_functions = top_file.get("key_functions", [])
    file_content = project.read_file_content(file_path)
    source_dir = project.resolve_source_dir(file_path)

    backend = languages.get_backend(language)
    resolved_prompts = prompts_module.resolve_agent_prompts(
        _AGENT_NAME, language, source_dir=source_dir,
        hint_profile=hints.hint_profile(),
    )

    Path(test_output_dir).mkdir(parents=True, exist_ok=True)

    backend.setup_test_env(test_output_dir, source_dir, file_path)
    refuse_rewritten_inherited_suite(backend, baseline["content"], "coverage", "starter")

    log_print(f"[coverage] target: {file_path}")
    log_print(f"[coverage] language: {language}")
    log_print(f"[coverage] test output dir: {test_output_dir}")

    session = TestCoverageSession()
    # Phase clock: everything from here (the baseline seed, the model loop,
    # terminal finalization) is phase cost.
    session.phase_clock_start = time.perf_counter()
    session.active_backend = backend
    _seed_baseline(session, baseline, file_path)
    log_print(
        f"[coverage] baseline seeded from the starter phase: "
        f"line {baseline.get('line_coverage')} "
        f"branch {baseline.get('branch_coverage')} "
        f"({baseline.get('test_count')} tests)"
    )

    deps = CoverageTestDeps(
        target_file_path=file_path,
        target_file_content=file_content,
        source_dir=source_dir,
        test_output_dir=test_output_dir,
        language=language,
        key_functions=key_functions,
    )

    agent = CoverageAgent(
        prompts=resolved_prompts, deps=deps, session=session, logger=_log.logger,
    )
    agent._prepared_models = prepared_models
    agent.record_generation_origin()
    agent._maybe_set_early_stop(source_tool="baseline")
    if session.early_stop_result is not None:
        session.framework_prep_seconds = (
            time.perf_counter() - session.phase_clock_start
        )
        log_print(
            "[coverage] phase ended at the inherited baseline "
            f"({session.early_stop_reason}). No model call was made."
        )
        return agent.finalize_result(
            session.early_stop_result,
            error_msg=None,
            duration_ms=session.framework_prep_seconds * 1000.0,
            usage_tracker=SimpleNamespace(
                requests_used=0,
                logical_requests_used=0,
                unknown_usage_requests=0,
                input_tokens_used=0,
                output_tokens_used=0,
                cache_hit_tokens_used=0,
                unknown_cache_requests=0,
            ),
            final_messages=[],
        )


    goal_value = f"{float(_AGENT_CONFIG['goal']):g}"
    goal_metric = _goal_metric()
    goal_clauses = {
        "line": f"{goal_value}% line coverage",
        "branch": f"{goal_value}% branch coverage",
        "either": f"{goal_value}% on EITHER line or branch (whichever is higher)",
        "both": (
            f"{goal_value}% on BOTH line and branch "
            f"(whichever is lower must clear the bar)"
        ),
    }
    coverage_goal_clause = goal_clauses[goal_metric]
    # A measured target without branches is judged by line coverage.
    no_branches = (baseline.get("line_coverage") is not None
                   and baseline.get("branch_coverage") is None)
    if no_branches and goal_metric != "line":
        coverage_goal_clause = f"{goal_value}% line coverage (the target has no branches)"

    best = agent._best_coverage_text(1)
    baseline_clause = f" It measured {best}." if best is not None else ""
    # The working file's name, as the backend delivers it for the target.
    working_file = backend.target_test_path(test_output_dir, file_path)
    user_prompt = Template(resolved_prompts["user_prompt_template"]).substitute(
        file_path=project.display_path(file_path, source_dir),
        language=language,
        test_output_dir=test_output_dir,
        coverage_goal=goal_value,
        coverage_goal_clause=coverage_goal_clause,
        coverage_patience=int(_AGENT_CONFIG["max_measurements_without_improvement"]),
        stagnation_min_delta=f"{float(_AGENT_CONFIG['min_improvement_points']):g}",
        coverage_minimum=f"{float(_AGENT_CONFIG['minimum']):g}",
        baseline_clause=baseline_clause,
        working_file_name=working_file.name if working_file is not None else "",
    )

    return agent.run_phase(project_name=project_name, user_prompt=user_prompt)
