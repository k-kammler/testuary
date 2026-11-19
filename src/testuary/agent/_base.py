"""The base class of the coverage and mutation agents: ``CodingAgent`` holds
the resolved prompts, logging, usage tracking and output validation, the
per-target state, the shared tools and the graph (the edit tool is the
``_edit_tool.EditTool`` mixin). Phase subclasses supply hooks for target
paths, validation and measurement. Bound tool methods become LangChain
tools without the ``self`` argument."""
from __future__ import annotations

import hashlib
import inspect
import json
import platform
import sys
import time
from asyncio import CancelledError as AsyncCancelledError
from concurrent.futures import CancelledError as FutureCancelledError
from contextlib import nullcontext
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool as lc_tool

from testuary import languages
from testuary.agent import _limits
from testuary import evidence as validity
from testuary.generation_evidence import source_sha256
from testuary.phase_state import timing_fields
from testuary import config as config_module
from testuary import models
from testuary import model_requests
from testuary.agent import _context as context
from testuary.agent import _langgraph as lg
from testuary import _chat_log
from testuary.agent import _helpers as agent_helpers
from testuary.agent._edit_tool import EditTool
from testuary import _test_output
from testuary import models as model_source
from testuary import normal_run
from testuary import loadable


def refuse_rewritten_inherited_suite(backend: Any, content: str, phase: str, source: str) -> None:
    """Raise RuntimeError when the suite the ``source`` phase handed over does
    not parse under the backend's shadowing rule or contains shadowed
    definitions, since the ``phase`` does not rewrite a measured suite."""
    normalized = backend.normalize_shadowed_definitions(content)
    if not normalized.parse_ok:
        raise RuntimeError(
            f"the inherited {phase} suite does not parse under "
            f"{normalized.rule_version}: {normalized.parse_error}"
        )
    if normalized.changed:
        raise RuntimeError(
            f"the suite handed over by the {source} phase contains shadowed "
            f"definitions, and {phase} refuses to rewrite a measured suite"
        )


def stop_status(reason: str, *, measured: bool) -> str:
    """The status of a phase that a framework stop ``reason`` ended, in
    either loop, never from the model's fields. ``measured`` says whether
    the best-green snapshot, the suite the phase delivers, is measured."""
    if reason.endswith("_goal_met"):
        return "success"
    if reason == "inherited_suite_too_slow":
        return "skipped_inherited_slow"
    if reason == "repeated_test_timeout":
        return "stagnation" if measured else "tests_failed"
    # The submission cap stops only a green, measured suite (the validator
    # accepts its stop on no other), a stop before the goal as stagnation's.
    if "stagnation" in reason or reason == "final_answers_without_measurement":
        return "stagnation"
    return "tests_failed"


class CodingAgent(EditTool):
    """Shared prompts, logging, dependencies, state, editing tools,
    validation and phase hooks for the coverage and mutation agents."""

    def __init__(
        self,
        prompts: Dict[str, Any],
        deps: Any,
        session: Any,
        logger: Any = None,
    ) -> None:
        self.prompts = prompts
        self.deps = deps
        self.session = session
        self.logger = logger
        self._configured_chat_model = None
        self._prepared_models = None

    def log(self, message: str, level: int = 0, **fields: Any) -> None:
        """Log a line. ``fields`` go into its saved record."""
        agent_helpers.log_line(self.logger, message, level, **fields)

    def debug(self, message: str, level: int = 0) -> None:
        """Record a detail entry in the saved log only (the logger's streamed
        JSON timeline, tagged ``debug``), never on the console. Without a
        logger it is dropped."""
        if self.logger is not None and hasattr(self.logger, "debug"):
            self.logger.debug(message, level=level)

    def tool_msg(self, key: str, **subs: Any) -> str:
        """Resolve a tool-facing message from this agent's prompts."""
        return agent_helpers.tool_msg(self.prompts, key, **subs)

    def agent_config(self) -> Dict[str, Any]:
        """The agent's resolved settings (its ``_AGENT_CONFIG``), from the subclass."""
        raise NotImplementedError

    def make_usage_tracker(self) -> lg.UsageTracker:
        """The run's usage tracker, sized from this agent's settings: the
        request limit and the optional per-run input, output and combined
        token caps and wall-clock deadline (0 is unlimited). Whichever is hit
        first fires the designed limit stop through ``finalize_result``.
        Logs the effective limits once."""
        cfg = self.agent_config()
        request_limit = cfg["max_requests_per_phase"]
        in_cap = int(cfg["max_input_tokens_per_phase"]) or None
        out_cap = int(cfg["max_output_tokens_per_phase"]) or None
        combined_cap = int(cfg["max_total_tokens_per_phase"]) or None
        wall_cap = float(cfg["max_wall_clock_seconds_per_phase"]) or None

        def _show(v, suffix=""):
            return f"{v}{suffix}" if v else "unlimited"

        self.log(
            f"[run-limits] max_requests_per_phase={_show(request_limit)} "
            f"input_tokens={_show(in_cap)} output_tokens={_show(out_cap)} "
            f"combined_tokens={_show(combined_cap)} "
            f"wall_clock={_show(wall_cap, 's')}",
            level=1,
        )
        return lg.UsageTracker(
            request_limit=request_limit,
            input_token_limit=in_cap,
            output_token_limit=out_cap,
            combined_token_limit=combined_cap,
            wall_clock_limit_seconds=wall_cap,
            heartbeat_every=int(cfg["progress_heartbeat_requests"]),
            heartbeat_fn=self.build_heartbeat_fn(),
        )

    def build_heartbeat_fn(self):
        """The usage tracker's progress-heartbeat callback: every N requests it
        logs the phase, the target, the elapsed time and the request count
        against the limit, so a long detached run shows it is alive."""
        target = getattr(self.deps, "target_file_path", None)
        name = Path(target).name if target else "?"
        label = self.phase_label()

        def _beat(requests_used: int, request_limit: int, elapsed_s: float) -> None:
            used = f"{requests_used}/{request_limit}" if request_limit else str(requests_used)
            self.log(
                f"[progress] phase={label} file={name} "
                f"elapsed={elapsed_s / 60.0:.1f}m requests={used}",
                level=1,
            )

        return _beat

    def resolve_log_dir(self) -> Path:
        """The directory of the run's logs, traces and chat snapshots: the
        streaming logger's directory."""
        return Path(self.logger._streaming_path).parent

    # The read gate and the content-check hook every editing agent shares. The
    # loops enforce the gate through mode binding and add their own gates in
    # ``CodingAgent.validate_output``.

    #: The read that opens the read gate (``session.has_read``), so a secondary
    #: read cannot: ``read_tests`` for the loops, which revise the inherited
    #: suite. Empty accepts any read.
    designated_read_name: str = ""

    def _note_read(self, tool_name: str) -> None:
        """Open the read gate (``session.has_read``) on the designated read, or
        on any read when the agent names none."""
        if not self.designated_read_name or tool_name == self.designated_read_name:
            self.session.has_read = True

    def validate_result_fields(self, parsed: Any) -> Optional[str]:
        """The model-facing rejection of a well-formed but empty or placeholder
        result (beyond the schema's types, presence and ranges), or None to
        accept. The default accepts. Each agent adds its own field checks."""
        return None

    #: The read tools' names, for the read cache of consecutive calls. A
    #: non-read tool clears every read counter.
    read_tool_names: tuple = ()

    #: Whether an edit that shadows a definition in the same scope (a duplicate
    #: test name) is rejected. On for the test-writing agents.
    guards_duplicate_test_names: bool = False

    #: Test-writing agents enable the backend-owned generated-test policy.
    #: Other coding agents do not edit a test suite and leave it disabled.
    guards_generated_test_validity: bool = False

    #: The model-facing name of the shared ``edit_file`` body (``edit_tests``
    #: for the loops), also used in the log and the read-counter reset.
    edit_tool_name: str = "edit_file"

    #: Run one response's tool calls one at a time, in order: in parallel the
    #: edits each read the same file and the last write won, while every call
    #: reported success. The mutation tools also share one worker scope.
    serial_tool_calls: bool = True

    #: The model-facing name of the measurement tool (``measure_coverage``,
    #: ``measure_mutation``), for shared text that names the next step.
    measure_tool_name: str = ""

    #: Graph wiring the subclass fills: the Pydantic result the model emits,
    #: the LangGraph state TypedDict, the synthetic final-answer tool name,
    #: and the agent_mode the graph starts in before the first classify.
    result_schema: Any = None
    graph_state_type: Any = None
    final_tool_name: str = ""
    initial_agent_mode: str = "needs_editing"

    #: Whether the graph starts in ``INITIAL_READ_MODE``, binding only the
    #: designated read, so the file enters the history as a read the dedup pass
    #: can reclaim rather than as prompt text. Agents that opt in leave it out
    #: of their user prompt and start in ``INITIAL_READ_MODE``.
    forces_initial_read: bool = False

    #: No secondary read can satisfy the gate in ``INITIAL_READ_MODE``.

    #: The read-only mode name used when ``forces_initial_read`` is on.
    INITIAL_READ_MODE: str = "needs_initial_read"

    #: Whether the designated read is followed by ``INITIAL_MEASURE_MODE``: the
    #: first measurement call, the start's second step, is answered from the
    #: suite's stored measurement when one exists. Then the loop enters Revise.
    forces_initial_measurement: bool = False

    #: The measure-only mode name used when ``forces_initial_measurement`` is on.
    INITIAL_MEASURE_MODE: str = "needs_initial_measurement"

    #: Each graph mode's step, as a phase's plan and its step observations
    #: name it: the start's read and measurement, each run once, then the
    #: cycle's revise, validate and assess, one for every mode.
    MODE_STEPS: Dict[str, Tuple[str, str]] = {
        INITIAL_READ_MODE: ("read", "Read tests"),
        INITIAL_MEASURE_MODE: ("measure", "First measurement"),
        "needs_editing": ("revise", "Edit tests"),
        "needs_testing": ("validate", "Run tests"),
        "needs_measurement": ("assess", "Measure"),
        "needs_measurement_noted": ("assess", "Measure"),
    }

    #: Whether this agent's loop ended at a request, token or time limit: the
    #: run's status follows it, not its error text.
    _limit_reached: bool = False

    def observe_step(self, mode: Optional[str], state: str, message: str = "") -> None:
        """Show graph mode ``mode``'s step in ``state`` on the run's display, if
        one observes it, with ``message`` in place of the state's own words
        (the step a limit ends names the limit). Nothing reaches the model,
        the logs or the records, and an observer's error never reaches the
        loop."""
        observer = getattr(self, "_display_observer", None)
        step = self.MODE_STEPS.get(mode) if mode is not None else None
        if observer is None or step is None:
            return
        if state == "running":
            self._shown_mode = mode
        try:
            observer(operation=f"step/{step[0]}", level="step", label=step[1], state=state,
                     message=message or {"running": "Step started",
                                         "failed": "Step failed"}.get(state, "Step done"))
        except Exception:
            pass

    def phase_model_settings(self):
        """This phase's model settings from the run settings, not from the
        current environment. None under a model source."""
        if model_source.current() is not None:
            return None
        from testuary.models import resolve_phase_model

        return resolve_phase_model(config_module.run_settings(), self.phase_label())

    def reasoning_fields(self) -> tuple[str, ...]:
        from testuary.models import field_names

        settings = self.phase_model_settings()
        if settings is not None:
            return settings.values["reasoning_field_names"]
        return field_names(self.agent_config().get("reasoning_field_names"))

    def build_chat_model(self) -> Any:
        """The chat model this agent's graph runs on: an installed model
        source's, else one built from the run settings (kept per run scope)."""
        source = model_source.current()
        if source is not None:
            self._configured_chat_model = None
            return source.agent_chat_model(self.phase_label())
        from testuary.models import build_resolved_chat_model

        settings = self.phase_model_settings()
        scope = config_module.active_run_scope()
        cached = self._configured_chat_model
        if cached is not None and cached[0] is scope and cached[1] == settings:
            return cached[2]
        self._configured_chat_model = None
        model = (self._prepared_models.take(settings, scope)
                 if self._prepared_models is not None else None)
        if model is None:
            model = build_resolved_chat_model(settings, api_key=models.get_api_config()["api_key"])
        self._configured_chat_model = (scope, settings, model)
        return model

    def provider_transport(self) -> str:
        """The client this agent's requests go through, as its transport
        records name it: an installed model source, else the protocol of its
        settings (the OpenAI-compatible chat client or the Responses API). The
        endpoint does not change it: unlike the starter phase, the agents
        reach Ollama Cloud through that client too."""
        if model_source.current() is not None:
            return "model_source"
        return {"openai-chat": "openai_compatible", "openai-responses": "openai_responses",
                "added": "added"}[self.phase_model_settings().protocol]

    # Graph construction. Agents differ by the hooks and class attributes above

    def tests_changed_since_measurement(self) -> bool:
        """Whether the working suite changed since its last completed
        measurement."""
        raise NotImplementedError

    # The best-green snapshot's loop state: what a measured snapshot carries
    # and what a restore brings back.

    #: The loop state a measured snapshot records, as (key, session
    #: attribute) pairs in record order.
    loop_state_fields: Tuple[Tuple[str, str], ...] = ()
    #: The text fields among them, restored as "" when the snapshot holds none.
    loop_text_fields: Tuple[str, ...] = ()
    #: Whether a snapshot restore that raises is logged and counts as none
    #: (the mutation phase) instead of propagating.
    swallows_restore_errors: bool = False

    def _snapshot_loop_state(self, content: str) -> Dict[str, Any]:
        """The loop state the working file ``content`` leaves, for its
        best-green snapshot: measured only while the last completed
        measurement scored exactly these bytes, and then everything that
        measurement left (``loop_state_fields``)."""
        session = self.session
        # The note of the green run of these bytes, which a restore brings back.
        note = {"stands_alone_note": session.stands_alone_note}
        if not self.measured_bytes(content):
            return {"measured": False, **note}
        return {
            "measured": True,
            **note,
            **{key: getattr(session, name) for key, name in self.loop_state_fields},
        }

    def _snapshot_current(self, state: Dict[str, Any]) -> bool:
        """Whether a measured snapshot's measurement holds for the loop now."""
        return True

    def _restore_unmeasured(self, snap: Dict[str, Any], state: Dict[str, Any]) -> None:
        """Put back the scores recorded with an unmeasured snapshot's bytes,
        None where they were not measured, never the discarded suite's."""
        raise NotImplementedError

    def _apply_snapshot_state(self, snap: Dict[str, Any]) -> None:
        """Put the loop in the state the restored snapshot's bytes left: green
        and unchanged since its run. A measured snapshot whose measurement
        holds now (``_snapshot_current``) brings back its measurement, so
        those bytes are not measured again. Any other stays unmeasured with
        the values recorded with its bytes (``_restore_unmeasured``)."""
        session = self.session
        session.last_tests_passed = True
        session.tests_changed_since_run_tests = False
        session.best_green.consecutive_failed_runs = 0
        state = snap.get("loop_state") or {}
        session.stands_alone_note = state.get("stands_alone_note") or ""
        session.stands_alone_noted = bool(session.stands_alone_note)
        if state.get("measured") and self._snapshot_current(state):
            self.set_tests_changed(False)
            for key, name in self.loop_state_fields:
                value = state.get(key)
                setattr(session, name, (value or "") if key in self.loop_text_fields else value)
        else:
            self.set_tests_changed(True)
            session.measured_suite_hash = None
            self._restore_unmeasured(snap, state)

    def _best_green_measured(self, score: str) -> bool:
        """Whether the best-green snapshot holds the bytes last measured, with
        their ``score`` ("coverage" or "mutation")."""
        snapshot = self.session.best_green.snapshot or {}
        content = snapshot.get("content")
        return (isinstance(content, str) and snapshot.get(score) is not None
                and source_sha256(content)
                == self.session.measured_suite_hash)

    def _restore_snapshot(self, target: Optional[Path]) -> bool:
        """Write the best-green snapshot over ``target`` and restore the
        loop state its bytes left. False, with nothing changed, when there
        is no snapshot or target, or the file already holds the snapshot."""
        snap = self.session.best_green.snapshot
        if snap is None or target is None:
            return False
        try:
            restored = agent_helpers.restore_best_green(
                self.session.best_green, target, current_is_green=False, force=True,
            )
        except Exception as exc:
            if not self.swallows_restore_errors:
                raise
            self.log(
                f"[restore] restoring the best-green snapshot failed "
                f"({type(exc).__name__}: {exc}).",
                level=1,
            )
            return False
        if not restored:
            return False
        self._apply_snapshot_state(snap)
        return True

    #: The key the terminal restore ranks the live file and the snapshot by,
    #: None for agent_helpers.green_snapshot_key.
    snapshot_rank_key: Optional[Callable[[Optional[Dict[str, Any]]], tuple]] = None

    def live_scores(self, measured: bool) -> Dict[str, Any]:
        """The known-green live file's scores for the terminal restore's
        ranking, its measured ones only while ``measured`` (the last completed
        measurement scored its bytes), else None."""
        raise NotImplementedError

    def _settle_best_green(self, target: Optional[Path], *,
                           log_fn: Optional[Callable[..., None]] = None) -> bool:
        """At the phase's end, leave the best measured green file on ``target``,
        since the merge takes the file as it is: the live file stays only while
        known green (its last run passed and nothing changed since) and ranked
        strictly above the snapshot by ``snapshot_rank_key``, so a red,
        unverified, unmeasured, lower or equal last edit returns to the
        snapshot. A file that holds the snapshot's own bytes while the last
        measurement scored other bytes takes the snapshot's loop state too.
        True when the snapshot was written."""
        session = self.session
        snap = session.best_green.snapshot
        if snap is None or target is None:
            return False
        current_is_green = (
            session.last_tests_passed and not session.tests_changed_since_run_tests
        )
        live = None
        if current_is_green:
            try:
                measured = self.measured_bytes(path=target)
            except OSError:
                measured = False
            live = self.live_scores(measured)
        try:
            restored = agent_helpers.restore_best_green(
                session.best_green, target, current_is_green=current_is_green,
                live_candidate=live, rank_key=self.snapshot_rank_key, log_fn=log_fn,
            )
            if not restored and not (target.exists()
                                     and target.read_text(encoding="utf-8") == snap["content"]
                                     and not self.measured_bytes(path=target)):
                return False
            # The delivered file's loop state, so the result reports what that
            # file left.
            self._apply_snapshot_state(snap)
        except Exception as exc:
            if not self.swallows_restore_errors:
                raise
            self.log(
                f"[restore] best-green restore failed silently: "
                f"{type(exc).__name__}: {exc}",
                level=2,
            )
            return False
        return restored

    def measured_bytes(self, content: Optional[str] = None, *,
                       path: Optional[Path] = None) -> bool:
        """Whether the last completed measurement scored exactly ``content``,
        or the text of ``path``, which is read only when the changed flag and
        the measured hash leave the answer open."""
        session = self.session
        if self.tests_changed_since_measurement() or session.measured_suite_hash is None:
            return False
        if path is not None:
            content = path.read_text(encoding="utf-8")
        return (content is not None
                and source_sha256(content)
                == session.measured_suite_hash)

    def resolve_agent_mode(self) -> str:
        """The agent mode the session state calls for now, which gates the tools
        and drives the classify transition: ``needs_editing`` (nothing pending:
        reads and writes), ``needs_testing`` (edits not yet run: reads, writes
        and run_tests), ``needs_measurement`` (green and unmeasured since the
        model's own edits: reads and the measurement tool) or
        ``needs_measurement_noted`` (the same after a green run that carried a
        STANDS ALONE note: the edit tool too, so the model can fix the tests
        the note names first). The loop starts in Revise: before the model's
        first edit the handed-over suite is never routed to a measurement, and
        an unmeasured baseline stays unmeasured."""
        session = self.session
        edits_since_run = bool(session.tests_changed_since_run_tests)
        edits_since_measurement = bool(self.tests_changed_since_measurement())
        has_writes = session.writes_done > 0
        if has_writes and edits_since_run:
            return "needs_testing"
        if has_writes and session.last_tests_passed and edits_since_measurement:
            return ("needs_measurement_noted" if session.stands_alone_noted
                    else "needs_measurement")
        return "needs_editing"

    def effective_agent_mode(self) -> str:
        """The agent mode the graph routes on:
        ``INITIAL_READ_MODE`` before a forced first read,
        ``INITIAL_MEASURE_MODE`` before a forced first measurement, else
        ``resolve_agent_mode``'s."""
        if self.forces_initial_read and not self.session.has_read:
            return self.INITIAL_READ_MODE
        if self.forces_initial_measurement and not self.session.initial_measured:
            return self.INITIAL_MEASURE_MODE
        return self.resolve_agent_mode()

    def in_start_measurement(self) -> bool:
        """Whether a measurement call now is the start's: the designated read
        returned and the measurement tool has not answered yet."""
        return (self.forces_initial_measurement and self.session.has_read
                and not self.session.initial_measured)

    def tool_methods(self) -> tuple:
        """This agent's tool methods, unwrapped, as ``(reads, writes, run_tests,
        measure)``: two lists and two methods, which ``tool_mode_sets`` gates by
        agent mode and ``phase_tools`` wraps. The measure tool is the method
        named ``measure_tool_name``."""
        return (
            [self.read_source, self.read_tests],
            [self.edit_file],
            self.run_tests,
            getattr(self, self.measure_tool_name),
        )

    def dedup_slots(self) -> Dict[str, str]:
        """The context-dedup slot of each read tool, so two reads of one file
        collapse: ``read_source`` the source, ``read_tests`` the working file."""
        return {
            "read_source": "source",
            "read_tests": "test",
        }

    def classify_log_fields(self) -> Dict[str, Any]:
        """Extra fields of the mode-transition log line. None by default."""
        return {}

    def classify_update(self, current_agent_mode: str) -> Dict[str, Any]:
        """The mode-transition delta: empty when the mode is unchanged, else
        ``{"agent_mode": <new>}``, which ``build_graph`` logs and traces."""
        new_agent_mode = self.effective_agent_mode()
        if new_agent_mode == current_agent_mode:
            return {}
        return {"agent_mode": new_agent_mode}

    def initial_read_tools(self, reads: List[Any]) -> List[Any]:
        """The reads bound in ``INITIAL_READ_MODE``: the designated read when one
        is named, matched by name in ``reads`` (wrapped or not, RuntimeError
        when absent), else all reads."""
        if not self.designated_read_name:
            return list(reads)
        def _name(t: Any) -> str:
            return getattr(t, "name", None) or getattr(t, "__name__", "")
        designated = [t for t in reads if _name(t) == self.designated_read_name]
        if not designated:
            raise RuntimeError(
                "designated initial read tool is not available: "
                f"{self.designated_read_name}"
            )
        return designated

    def unbound_tool_reply(self, name: str) -> Optional[str]:
        """The answer to a call of one of this agent's tools that the
        current agent mode does not bind: the tool and the step that offers it.
        None for a name that is not one of its tools."""
        reads, _writes, run_m, measure_m = self.tool_methods()
        run_name, measure_name = run_m.__name__, measure_m.__name__
        if name not in {m.__name__ for m in reads} | {
                self.edit_tool_name, run_name, measure_name}:
            return None
        agent_mode = self.effective_agent_mode()
        if agent_mode == self.INITIAL_READ_MODE:
            key = "tool_not_offered_initial_read"
        elif agent_mode == self.INITIAL_MEASURE_MODE:
            key = "tool_not_offered_initial_measure"
        elif agent_mode == "needs_measurement":
            key = "tool_not_offered_measure_first"
        elif name == run_name:
            key = "tool_not_offered_run_tests"
        elif agent_mode == "needs_testing":
            key = "tool_not_offered_measure_untested"
        elif self.session.last_tests_passed and self.measurement_staleness() is not None:
            # Revise at entry, on a green suite not measured.
            key = "tool_not_offered_measure_entry"
        elif self.session.last_tests_passed:
            key = "tool_not_offered_measure_measured"
        elif self.refused_by():
            key = "tool_not_offered_measure_refused"
        else:
            key = "tool_not_offered_measure_red"
        return self.tool_msg(key, tool=name, read=self.designated_read_name,
                             edit=self.edit_tool_name, measure=measure_name,
                             refused_by=self.refused_by_phrase())

    def _tool_execution_context(self):
        """The execution scope entered on the dispatched tool's thread."""
        return nullcontext()

    def _bind_tool(self, method: Any) -> Any:
        @wraps(method)
        def scoped(*args, **kwargs):
            from testuary.languages.mutation import tool_scope

            # A Ctrl+C stops what the call started, and the call records nothing.
            with tool_scope(), self._tool_execution_context():
                return method(*args, **kwargs)

        # The tool's description is its docstring as Python 3.13 and later
        # compile it, without the indentation, on every Python version.
        return lc_tool(scoped, description=inspect.cleandoc(method.__doc__))

    def _bind_edit_tool(self, method: Any) -> Any:
        """The shared ``edit_file`` body as a tool named ``edit_tool_name``
        (``edit_tests`` for the loops). LangChain routes a call of that name
        back to ``edit_file``."""
        tool = self._bind_tool(method)
        tool.name = self.edit_tool_name
        return tool

    def phase_tools(self) -> tuple:
        """The tools ``build_graph`` binds, grouped as ``tool_methods``
        groups them: every method wrapped for the model, the edit tool under
        this agent's name."""
        reads_m, writes_m, run_m, measure_m = self.tool_methods()
        return (
            [self._bind_tool(m) for m in reads_m],
            [self._bind_edit_tool(m) for m in writes_m],
            self._bind_tool(run_m),
            self._bind_tool(measure_m),
        )

    def tool_mode_sets(self, reads, writes, run_tests_t, measure_t) -> dict:
        modes = {
            "needs_editing": reads + writes,
            "needs_testing": reads + writes + [run_tests_t],
            "needs_measurement": reads + [measure_t],
            "needs_measurement_noted": reads + writes + [measure_t],
        }
        if self.forces_initial_read:
            modes[self.INITIAL_READ_MODE] = self.initial_read_tools(reads)
        if self.forces_initial_measurement:
            modes[self.INITIAL_MEASURE_MODE] = [measure_t]
        return modes

    def preflight_tool_binding(self, model) -> None:
        """Validate the real tool schemas without running a graph or tool."""
        from langchain_core.runnables import Runnable
        from testuary.settings import ConfigurationError

        final_tool = lg._make_final_answer_tool(self.result_schema, self.final_tool_name)
        try:
            for tools in self.tool_mode_sets(*self.phase_tools()).values():
                bound = model.bind_tools([*tools, final_tool])
                if not isinstance(bound, Runnable):
                    raise TypeError("bound model must be runnable")
        except Exception:
            raise ConfigurationError("chat_model tool binding failed. Check its tool support") from None

    def tool_surface_record(self) -> Dict[str, Any]:
        """The interpreter version and a fingerprint of every tool this agent
        binds, the final-answer tool included: its name, description and
        argument schema, so every chat log states which tools the model
        received."""
        reads, writes, run_t, measure_t = self.phase_tools()
        final_tool = lg._make_final_answer_tool(self.result_schema, self.final_tool_name)
        described = sorted(
            ((tool.name, tool.description, tool.args_schema.model_json_schema())
             for tool in reads + writes + [run_t, measure_t, final_tool]),
            key=lambda entry: entry[0],
        )
        return {
            "python_version": platform.python_version(),
            "tool_descriptions_sha256": source_sha256(
                json.dumps(described, ensure_ascii=False, sort_keys=True)),
        }

    def chat_log_metadata(self) -> Dict[str, Any]:
        """The chat log's metadata: the tool surface record, the agent's own
        fields and the generation condition when one is recorded."""
        metadata = {**self.tool_surface_record(), **self.chat_metadata()}
        condition = getattr(self, "_generation_condition", None)
        if condition is not None:
            metadata["generation_condition"] = condition.export()
        return metadata

    def record_generation_origin(self) -> None:
        from testuary.evidence import current_condition, write_origin

        self._generation_condition = current_condition(
            self.phase_label(), self.deps.language, self.deps.source_dir, templates=self.prompts,
        )
        write_origin(self.resolve_log_dir() / f"{self.phase_label()}_generation_origin.json",
                     self._generation_condition)

    def build_graph(
        self,
        *,
        model: Optional[Any] = None,
        usage_tracker: Optional[Any] = None,
        trace_logger: Optional[Any] = None,
        usage_logger: Optional[Any] = None,
        request_account: Optional[Any] = None,
        on_send_again: Optional[Any] = None,
    ) -> Any:
        """This agent's LangGraph react graph: the shared topology with the
        agent's tools, dedup slots, result, state and final-tool wiring and log
        fields. ``request_account`` is the phase's state for requests sent again (the
        bundled defaults without one)."""
        chat_model = model if model is not None else self.build_chat_model()
        system_prompt = self.prompts["system_prompt"]

        reads, writes, run_tests_t, measure_t = self.phase_tools()

        slots = self.dedup_slots()
        main_cfg = config_module.load_config("main")
        reasoning_fields = self.reasoning_fields()
        pre_hook = context.make_context_hook(
            self.build_state_marker,
            system_prompt=system_prompt,
            keep_last=_limits.value("history_messages"),
            dedup={
                "read_tools": {t.name for t in reads},
                "file_key": lambda name, args: slots.get(name, name),
                "stub": context.dedup_stub,
            },
            strip_thinking=True,
            rebuild=config_module.context_rebuild(),
            rebuild_trigger_tokens=int(
                main_cfg.get("context_rebuild_trigger_tokens", 0) or 0),
            reasoning_field_names=reasoning_fields,
        )

        def classify(state: Any) -> Dict[str, Any]:
            current_agent_mode = state.get("agent_mode", self.initial_agent_mode)
            update = self.classify_update(current_agent_mode)
            if not update:
                return {}
            new_agent_mode = update["agent_mode"]
            self.observe_step(current_agent_mode, "succeeded")
            self.observe_step(new_agent_mode, "running")
            try:
                extra = "".join(
                    f", {k}={v}" for k, v in self.classify_log_fields().items()
                )
                self.log(
                    f"[classify_agent_mode] {current_agent_mode} -> "
                    f"{new_agent_mode} (last_tests_passed="
                    f"{self.session.last_tests_passed}{extra})",
                    level=2,
                )
            except Exception:
                pass
            if trace_logger is not None and trace_logger.enabled:
                trace_logger.event(
                    "phase_transition",
                    from_agent_mode=current_agent_mode,
                    to_agent_mode=new_agent_mode,
                    last_tests_passed=self.session.last_tests_passed,
                )
            return update

        agent_mode_tools = self.tool_mode_sets(reads, writes, run_tests_t, measure_t)

        return lg.build_react_graph(
            model=chat_model,
            agent_mode_tools=agent_mode_tools,
            resolve_agent_mode=lambda _s: self.effective_agent_mode(),
            pre_model_hook=pre_hook,
            structured_output=self.result_schema,
            output_validator=self.validate_output,
            state_type=self.graph_state_type,
            final_tool_name=self.final_tool_name,
            usage_tracker=usage_tracker,
            trace_logger=trace_logger,
            usage_logger=usage_logger,
            classify_node=classify,
            request_account=request_account,
            on_send_again=on_send_again,
            reasoning_field_names=reasoning_fields,
            serial_tool_calls=self.serial_tool_calls,
            unbound_tool_reply=self.unbound_tool_reply,
        )

    # The run every editing agent shares

    def run_phase(self, *, project_name: str, user_prompt: str) -> Any:
        """Run one target's agent loop and return its result, the flow every
        ``generate_*`` entry point shares: the chat, trace and usage logs, the
        graph under the phase's request policy, output-validation retries in
        one usage limit, the framework's early-stop override, and
        ``finalize_result`` for the agent's terminal policy."""
        label = self.phase_label()
        if not hasattr(self, "_generation_condition"):
            self.record_generation_origin()
        target = self.deps.target_file_path
        reasoning_fields = self.reasoning_fields()
        self.logger.reasoning_field_names = reasoning_fields

        log_dir = self.resolve_log_dir()
        flush_chat_log = _chat_log.make_chat_log_flusher(
            output_dir=str(log_dir),
            filename=f"{label}_chat.json",
            project_name=project_name,
            extra_metadata_factory=lambda _stage: self.chat_log_metadata(),
            reasoning_field_names=reasoning_fields,
        )
        snapshot_retry_chat = _chat_log.make_retry_chat_snapshotter(
            output_dir=str(log_dir),
            filename_prefix=f"{label}_chat",
            project_name=project_name,
            extra_metadata_factory=lambda _stage: self.chat_log_metadata(),
            reasoning_field_names=reasoning_fields,
        )
        flush_chat_log([], stage="starting")

        usage_tracker = self.make_usage_tracker()
        # Stash on the session so a measure tool can read cumulative token
        # totals for its per-iteration history entries.
        self.session.usage_tracker_ref = usage_tracker
        cfg = self.agent_config()
        from testuary._progress import ToolObserver, phase_observer

        observer = phase_observer(label)
        self._display_observer = observer
        trace_path = (
            log_dir / f"{label}_trace.jsonl"
            if bool(cfg["enable_trace_log"]) else None
        )
        trace_logger = lg.TraceLogger(trace_path)
        usage_logger = lg.UsageLogger(
            log_dir / f"{label}_usage.jsonl",
            prompt_transport_path=(
                log_dir / f"{label}_prompt_transport.jsonl"
            ),
            prompt_transport_context={
                "phase": label,
                "prompt_template_version": validity.PROMPT_TEMPLATE_VERSION,
                "provider_transport": self.provider_transport(),
                "prompt_composition": (getattr(self, "_generation_condition", None).prompt_composition
                                       if getattr(self, "_generation_condition", None) is not None else None),
                "sent_settings": (self.phase_model_settings().export()
                                  if self.phase_model_settings() is not None else None),
            },
            reasoning_field_names=reasoning_fields,
            observer=observer,
        )
        trace_logger.event(
            "agent_start", framework="langgraph", **self.agent_start_fields()
        )

        # A transient request failure is retried in the model node. A terminal
        # one or the retry ceiling ends the run with the provider's error.
        model_settings = self.phase_model_settings()
        request_account = model_requests.RequestAccount(
            model_requests.RetrySettings.from_config(cfg),
            service=model_requests.service_address(
                model_settings.identity.api_url if model_settings is not None else None),
        )

        def _on_send_again(
            attempt: int, exc: BaseException, wait: float, msgs: List[Any],
        ) -> None:
            hint = model_requests.retry_hint_seconds(
                exc, max_seconds=request_account.settings.hint_max_s,
            )
            self.log(
                f"Sending again after attempt {attempt}/"
                f"{request_account.settings.consecutive_limit} "
                f"({type(exc).__name__}), waiting {wait:.1f}s "
                + ("(server hint)" if hint is not None else "(backoff)"),
                level=1,
            )
            snapshot_retry_chat(attempt, {"messages": msgs}, exc)

        graph = self.build_graph(
            usage_tracker=usage_tracker,
            trace_logger=trace_logger,
            usage_logger=usage_logger,
            request_account=request_account,
            on_send_again=_on_send_again,
        )
        initial_state: Dict[str, Any] = {
            "messages": [HumanMessage(content=user_prompt)]
        }
        captured_messages: List[Any] = list(initial_state["messages"])

        def _capture(state: Dict[str, Any]) -> None:
            msgs = state.get("messages")
            if not msgs:
                return
            captured_messages.clear()
            captured_messages.extend(msgs)
            flush_chat_log(list(captured_messages), stage="in_progress")

        request_limit = cfg["max_requests_per_phase"]
        config: Dict[str, Any] = {
            # Without a request limit the graph's own step bound is lifted too.
            "recursion_limit": sys.maxsize if request_limit is None else max(50, request_limit * 4),
            "configurable": {
                "thread_id": f"{project_name}-{label}-{Path(target).stem}",
            },
        }
        if observer is not None:
            config["callbacks"] = [ToolObserver(observer)]

        observed_output_retry = False

        def observe_output_retry(attempt, state, exc):
            nonlocal observed_output_retry
            snapshot_retry_chat(attempt, state, exc)
            # Only a rejected output is an output-validation retry. A rerun
            # after a transient failure (a tool raising) is not.
            if observer is not None and isinstance(exc, lg.OutputValidationError):
                observed_output_retry = True
                observer(operation="output-validation", label="Final answer", state="retrying",
                         message="The model's final answer was not usable. It was told why "
                                 "and continues.")

        # The recovery loop runs the graph again after a rejected output or an
        # escaped transient failure, at most ``max_loop_attempts`` attempts.
        max_retries = self.graph_attempts(cfg)

        start_perf = time.perf_counter()
        error_msg: Optional[str] = None
        final_messages: List[Any] = []
        output: Any = None
        failure: Dict[str, BaseException] = {}
        self._failure_kind = None
        self._limit_reached = False
        # The loop's end as its steps show it: a limit its settings set is no
        # failure.
        ended = "succeeded"
        # The words of the limit that ended the loop, and whether its result
        # is the model's own final answer: the final-answer line names both.
        limit_words = ""
        answered = False

        self.observe_step(self.initial_agent_mode, "running")
        try:
            try:
                from testuary.languages.mutation import cancel_on_interrupt

                # A Ctrl+C cancels a measurement a tool runs on another
                # thread, instead of waiting for it to end.
                with cancel_on_interrupt():
                    final_state, retry_err = lg.run_graph_with_retries(
                        graph, initial_state, config=config, max_retries=max_retries,
                        failure_out=failure,
                        log=lambda m: self.log(m, level=1),
                        on_intermediate_state=_capture,
                        on_before_retry=observe_output_retry,
                        trace_logger=trace_logger,
                        retry_prompt_builder=self.retry_prompt_builder(),
                    )
            except lg.UsageLimitExceeded as exceeded:
                ended = "limit_reached"
                self._limit_reached = True
                limit_words = lg._limit_words(exceeded.limit)
                final_messages = list(captured_messages)
                # Exhaustion finalizes the existing tracker, never a new graph.
                error_msg = (
                    f"Budget exhausted: {type(exceeded).__name__}: "
                    f"{str(exceeded)[:1000]}"
                )
                output = self.build_failure_result(error_msg)
            else:
                if final_state is not None and final_state.get("result") is not None:
                    output = final_state["result"]
                    answered = True
                    final_messages = list(final_state.get("messages", []))
                else:
                    err_text = retry_err or "graph terminated without a result"
                    self._failure_kind = self.failure_kind_of(failure.get("exception"))
                    self.log(f"Agent error: {err_text}")
                    output = self.build_failure_result(err_text)
                    error_msg = err_text
                    final_messages = list(captured_messages)

            if self.session.early_stop_result is not None:
                self.log(
                    f"Framework stopped agent early "
                    f"({self.session.early_stop_reason}). Using the synthesized result.",
                    level=1,
                )
                output = self.session.early_stop_result
                answered = False

            model_loop_ms = (time.perf_counter() - start_perf) * 1000
            output = self.finalize_result(
                output, error_msg=error_msg, duration_ms=model_loop_ms,
                usage_tracker=usage_tracker, final_messages=final_messages,
            )

            # The phase clock includes preparation and finalization. The model
            # loop closes before finalization starts.
            if self.session.phase_clock_start is not None:
                duration_ms = (
                    time.perf_counter() - self.session.phase_clock_start
                ) * 1000
            else:
                duration_ms = model_loop_ms
            trace_logger.event(
                "agent_end", duration_s=round(duration_ms / 1000.0, 2),
                model_loop_s=round(model_loop_ms / 1000.0, 2),
                framework_prep_s=round(self.session.framework_prep_seconds, 2),
                error=error_msg, **self.agent_end_fields(),
            )
            flush_chat_log(final_messages, error=error_msg, stage="completed")
            if error_msg and ended != "limit_reached":
                ended = "failed"
            self.observe_step(getattr(self, "_shown_mode", None), ended, limit_words)
            if observed_output_retry:
                observer(operation="output-validation", label="Final answer", state=ended,
                         message=("Usable final answer received" if ended == "succeeded" and answered
                                  else f"{limit_words} before a usable final answer"
                                  if ended == "limit_reached" else "No usable final answer"))
                observed_output_retry = False
            self.logger.log_analysis(target, duration_ms, final_messages, error_msg)
            self.log(
                f"Duration: {duration_ms / 1000:.1f}s "
                f"(model loop {model_loop_ms / 1000:.1f}s, framework prep "
                f"{self.session.framework_prep_seconds:.1f}s)"
            )
            return output
        except BaseException as exc:
            if observed_output_retry:
                cancelled = isinstance(exc, (KeyboardInterrupt, AsyncCancelledError,
                                            FutureCancelledError, model_requests.RequestCancelled))
                observer(operation="output-validation", label="Final answer",
                         state="cancelled" if cancelled else "failed", message="No usable final answer")
            raise

    # run_phase hooks (per-agent variation)

    def phase_label(self) -> str:
        """Phase name used in chat, trace, usage logs and graph thread identifiers."""
        raise NotImplementedError

    def chat_metadata(self) -> Dict[str, Any]:
        """The agent's chat-log metadata fields. The target file by default."""
        return {"target_file": self.deps.target_file_path}

    def agent_start_fields(self) -> Dict[str, Any]:
        """Extra ``agent_start`` trace event fields. The target file by default."""
        return {"target_file": self.deps.target_file_path}

    def agent_end_fields(self) -> Dict[str, Any]:
        """Extra fields of the ``agent_end`` trace event, read at the run's end.
        None by default."""
        return {}

    def recovery_state_block(self) -> str:
        """The agent's progress for validation feedback: the framework's current
        measurements, so the model need not infer them from older turns."""
        return ""

    def working_state_block(self, best_line: Optional[str], measurement: str) -> str:
        """The recovery state block of a coding loop: the working file on
        disk, the last test run, ``best_line`` when given, the test status
        and the next steps, which name the loop's ``measurement``."""
        session = self.session
        lines = ["[Recovery state snapshot - your in-memory progress so far]"]

        path = self.target_path()
        if path is not None and path.exists():
            try:
                n_lines = agent_helpers.line_count(path.read_text(encoding="utf-8"))
                lines.append(f"  Working file on disk: {path.name} ({n_lines} lines, {session.writes_done} write(s) so far)")
            except Exception:
                lines.append(f"  Working file on disk: {path.name} ({session.writes_done} write(s) so far)")
        else:
            lines.append(f"  Working file on disk: NONE YET ({session.writes_done} write(s) so far)")

        if self.refused_by():
            lines.append(f"  Last run_tests:    passed; {self.refused_by_phrase()} then "
                         "could not measure the file")
        elif session.last_run_tests_result:
            head = agent_helpers.run_summary_line(session.last_run_tests_result)
            lines.append("  Last run_tests:    " + _limits.cut_text(
                head, _limits.value("recovery_result_chars"), what="the result",
                keep="head", sep=" "))
        else:
            lines.append("  Last run_tests:    (not called yet; the framework ran the suite at the loop's start)")

        if best_line is not None:
            lines.append(best_line)

        # Before the model's first run_tests and first write, the status is
        # the framework's run at the loop's start, which the line above names.
        if not session.last_run_tests_result and session.writes_done == 0:
            lines.append("  Last test status:  GREEN at the loop's start (the framework's measurement)")
        elif session.last_tests_passed:
            lines.append("  Last test status:  GREEN (all on-disk tests passing)")
        elif self.refused_by():
            lines.append("  Last test status:  NOT MEASURABLE (edit the tests "
                         f"that stop {self.refused_by_phrase()})")
        else:
            lines.append("  Last test status:  RED or unknown (tests failing or not yet run)")

        lines.append("")
        lines.append(
            "Continue with a tool offered in the current state. Call "
            "read_tests if your view of the working suite is stale. Use "
            "read_source only when you need source context and that tool is "
            "offered. Otherwise make the next valid edit, test run, or "
            f"{measurement} measurement."
        )
        return "\n".join(lines) + "\n\n"

    def retry_prompt_builder(self) -> Any:
        """The builder of feedback after an output-validation rejection, the only
        failure it serves (the model node retries transport failures)."""
        def builder(exc: BaseException, history: List[Any]) -> str:
            if not isinstance(exc, lg.OutputValidationError):
                raise TypeError(
                    "retry feedback is defined only for output validation"
                )
            result_name = getattr(
                self.result_schema, "__name__", "the final result",
            )
            prompt = (
                self.recovery_state_block()
                + f"Your previous attempt to return {result_name} "
                "was rejected by the framework's output validator. "
                "Specific reason:\n\n"
                f"{exc}\n\n"
                "Fix the issue and try again."
            )
            for m in reversed(history):
                if isinstance(m, AIMessage):
                    metadata = getattr(m, "response_metadata", None) or {}
                    if metadata.get("finish_reason") == "length":
                        prompt += agent_helpers.TRUNCATION_HINT
                    break
            return prompt

        return builder

    @staticmethod
    def graph_attempts(cfg: Dict[str, Any]) -> int:
        """The recovery loop's bound on graph attempts, the first one
        included: ``max_loop_attempts``, at least one."""
        return max(1, int(cfg["max_loop_attempts"]))

    @staticmethod
    def failure_kind_of(exc: Optional[BaseException]) -> Optional[str]:
        """The failure kind of a run that ended on ``exc``. A request failure
        is infrastructure, except a request too large for the model's
        context, anywhere in the cause chain, which is unjudged, as its
        ``context_overflow`` status is. A message that only mentions a
        context, such as a gateway's context deadline, does not count."""
        seen: set = set()
        current = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if lg.is_context_window_error(current):
                return None
            current = current.__cause__ or current.__context__
        return model_requests.request_failure_kind(exc)

    @staticmethod
    def classify_error_status(error_msg: str, limit_reached: bool = False) -> str:
        """The failure status of a run that ended on ``error_msg``, the same for
        every editing agent: ``context_overflow``, ``budget_exhausted`` when the
        run ended at a request, token or time limit (``limit_reached``), else
        ``agent_error`` (a provider error, an out-of-credit 402 among them,
        never an early stop)."""
        # The retry rule's context-window recognizer: a mere mention of a
        # context (a gateway's context deadline) does not count.
        if lg.is_context_window_text(error_msg):
            return "context_overflow"
        return "budget_exhausted" if limit_reached else "agent_error"

    def build_failure_result(self, error_msg: str) -> Any:
        """The canonical result for a run that produced none, including a
        graph error or a reached limit. Per-agent: each builds its own type."""
        raise NotImplementedError

    def finalize_result(
        self, output: Any, *, error_msg: Optional[str], duration_ms: float,
        usage_tracker: Any, final_messages: List[Any],
    ) -> Any:
        """The final result: the run's raw ``output`` (past the early-stop
        override) stamped with duration and token usage. An agent overrides it
        for its restore, status and CSV columns, then calls this."""
        if output is None:
            return output
        # With the phase clock armed, the duration spans the preparation, the
        # model loop and the override's terminal work.
        if self.session.phase_clock_start is not None:
            duration_ms = (
                time.perf_counter() - self.session.phase_clock_start
            ) * 1000
        try:
            return output.model_copy(update={
                "agent_duration_s": round(duration_ms / 1000.0, 1),
                "model_requests": int(getattr(usage_tracker, "requests_used", 0)),
                "logical_requests": int(
                    getattr(usage_tracker, "logical_requests_used", 0)),
                "unknown_usage_requests": int(
                    getattr(usage_tracker, "unknown_usage_requests", 0)),
                "input_tokens": int(usage_tracker.input_tokens_used),
                "output_tokens": int(usage_tracker.output_tokens_used),
                "cache_hit_tokens": int(usage_tracker.cache_hit_tokens_used),
                "unknown_cache_requests": int(
                    getattr(usage_tracker, "unknown_cache_requests", 0)),
                "failure_kind": getattr(self, "_failure_kind", None),
            })
        except Exception:
            return output

    # Building blocks of each agent's per-turn state marker

    def task_head(self) -> str:
        """The marker's task clause and its `` | `` separator, or "" without a
        clause: the target file, so a history trim never loses which file is
        under work."""
        task = context.task_clause(self.deps.target_file_path)
        return f"{task} | " if task else ""

    def stalled_metric(self, value: Any, fmt: Callable[[Any], str]) -> str:
        """Format a best-metric ``value`` for the state marker: ``fmt(value)``
        with the ``(stalled N measurements)`` suffix when the no-improvement count
        is positive, or the literal ``none`` when no measurement has landed."""
        if value is None:
            return "none"
        out = fmt(value)
        stall = self.session.measurements_without_improvement
        if stall > 0:
            out += f" (stalled {stall} measurements)"
        return out

    def log_progress(
        self, step: int, metric: str, best: str,
        *, target: str = "", delta: str = "",
    ) -> None:
        """Log the per-measurement progress line every phase shares:
        ``[<phase> #<step>] <metric> | best <best> | <target> (<delta>)``, with
        ``target`` the distance to the goal and ``delta`` the change since the
        previous measurement."""
        line = f"[{self.phase_label()} #{step}] {metric} | best {best}"
        if target:
            line += f" | {target}"
        if delta:
            line += f" ({delta})"
        self.log(line, level=1)

    def count_cached_measurement(self, has_best: bool, log: Callable[..., None]) -> None:
        """Count a cached measurement call (no file change) as a no-improvement
        event when a best value exists and count_repeated_measurements is
        on (bundled off: no new suite was measured), and say so through ``log``."""
        count_cached = bool(self.agent_config()["count_repeated_measurements"])
        if has_best and count_cached:
            self.session.measurements_without_improvement += 1
            max_measurements_without_improvement = int(self.agent_config()["max_measurements_without_improvement"])
            log(
                f"[{self.measure_tool_name}] Cached call (no file change) "
                f"advanced the no-improvement counter to "
                f"{self.session.measurements_without_improvement}/{max_measurements_without_improvement}.",
                level=1,
            )
        elif has_best and not count_cached:
            log(
                f"[{self.measure_tool_name}] Cached call (no file change) "
                "did not advance the counter (count_repeated_measurements=false). "
                "Edit the file to take a real measurement.",
                level=1,
            )

    @staticmethod
    def _resets_streak(prev_best: Optional[float], new_best: Optional[float], *,
                       min_points: float, minimum: float) -> bool:
        """Whether a measurement that moves the best value from ``prev_best``
        to ``new_best`` resets the streak of measurements without
        improvement: a first best does, and so does a rise by at least
        ``min_points`` points, or any rise while ``prev_best`` is below
        ``minimum``, where the suite cannot yet back a correctness claim. A
        measurement without a best value counts as no improvement."""
        if new_best is None:
            return False
        if prev_best is None:
            return True
        min_delta = 0.0 if prev_best < minimum else min_points
        return new_best > prev_best and new_best - prev_best >= min_delta

    # Loop breaker (shared logic, per-target counters on self.session)

    def progress_made(self) -> None:
        """Reset the no-progress and cached-call counters after a successful
        edit, so the cached-call escalation starts over."""
        self.session.loop_breaker.calls_since_last_edit = 0
        self.session.loop_breaker.consecutive_cached_calls.clear()

    # Per-step logging and output tracking (shared)

    def log_thinking(self, tool_name: str) -> None:
        """Log the ``chat <elapsed>`` and ``running <tool>`` lines (no ``chat``
        line for a run's first tool), record ``current_tool_name`` for
        ``track_output``, and update the read cache: a read tool counts its own
        consecutive calls and clears the others', any other tool clears all."""
        s = self.session
        s.current_tool_name = tool_name
        now = time.monotonic()
        if s.last_tool_event_at is not None:
            pretty = agent_helpers.format_thinking_elapsed(
                now - s.last_tool_event_at
            )
            if pretty:
                self.log(f"chat {pretty}")
        self.log(f"running {tool_name}", level=1)
        s.last_tool_event_at = now
        if tool_name in self.read_tool_names:
            s.consecutive_reads_by_tool[tool_name] = (
                s.consecutive_reads_by_tool.get(tool_name, 0) + 1
            )
            for other in self.read_tool_names:
                if other != tool_name:
                    s.consecutive_reads_by_tool[other] = 0
                    s.read_ranges_seen_by_tool[other] = set()
        else:
            for name in self.read_tool_names:
                s.consecutive_reads_by_tool[name] = 0
                s.read_ranges_seen_by_tool[name] = set()

    def consecutive_reads(self, tool_name: str) -> int:
        """Times ``tool_name`` was called in immediate succession (no other
        tool between), for the cached short-read header."""
        return self.session.consecutive_reads_by_tool.get(tool_name, 0)

    def track_output(self, output: str) -> str:
        """Count the tool call, truncate the output to the tool result limit,
        and prepend the no-progress banner when it fires. The tool name is
        ``current_tool_name`` (set by ``log_thinking``)."""
        s = self.session
        s.session_tool_calls += 1
        # Tools put banners first ([STRICT GREEN], [NOTE] revert,
        # [validator]) that say why a passing-looking run fails, and the test
        # summary comes last: the cut keeps a third from the start and the
        # rest from the end.
        output = _limits.cut_text(output, _limits.value("tool_result_chars"),
                                  what="the tool result")
        if s.current_tool_name:
            return agent_helpers.bump_no_progress(s.loop_breaker) + output
        return output

    # Shared read rendering (the read tools are thin wrappers over these)

    def cached_read(
        self,
        tool_name: str,
        *,
        content: str,
        name: str,
        cached_key: str,
        cached_subs: Optional[Dict[str, Any]] = None,
        loop_breaker_key: Optional[str] = None,
        start_line: int = 1,
        end_line: Optional[int] = None,
    ) -> str:
        """Render a read of fixed in-memory ``content`` (the source): a re-read
        of a range already seen in the current streak (no edit or run between)
        returns the short ``cached_key`` header instead, after the
        ``loop_breaker_key`` message from the fourth call on. The caller calls
        ``log_thinking(tool_name)`` first. The output is tracked."""
        self._note_read(tool_name)
        n = self.consecutive_reads(tool_name)
        seen = self.session.read_ranges_seen_by_tool.setdefault(tool_name, set())
        this_range = (start_line, end_line)
        repeat = this_range in seen
        seen.add(this_range)
        if repeat and n >= 2:
            self.log(f"[{tool_name}] Cached: re-read of an already-read "
                     f"range #{n}.", level=1)
            msg = self.tool_msg(cached_key, n=n, **(cached_subs or {}))
            if loop_breaker_key is not None and n >= 4:
                msg = self.tool_msg(loop_breaker_key, n=n) + "\n\n" + msg
            return self.track_output(msg)
        return self.track_output(context.render_read(
            content, name=name, max_chars=_limits.value("read_chars"),
            start_line=start_line, end_line=end_line,
        ))

    def file_read(
        self,
        tool_name: str,
        *,
        path: Any,
        name: str,
        missing_msg: str,
        read_error_key: Optional[str] = None,
        start_line: int = 1,
        end_line: Optional[int] = None,
    ) -> str:
        """Render a read of an on-disk file that can change (the working file),
        uncached: ``missing_msg`` when ``path`` is absent, the ``read_error_key``
        message (or a plain one) on a read failure. The caller calls
        ``log_thinking(tool_name)`` first. The output is tracked."""
        if path is None or not path.exists():
            return self.track_output(missing_msg)
        try:
            content = path.read_text(encoding="utf-8")
        except Exception as exc:
            if read_error_key is not None:
                msg = self.tool_msg(
                    read_error_key, name=path.name,
                    exc_type=type(exc).__name__, exc=str(exc),
                )
            else:
                msg = f"ERROR reading {path.name}: {exc}"
            return self.track_output(msg)
        self._note_read(tool_name)
        return self.track_output(context.render_read(
            content, name=name, max_chars=_limits.value("read_chars"),
            start_line=start_line, end_line=end_line,
        ))

    def _source_read(self, start_line: int, end_line: Optional[int]) -> str:
        """The ``read_source`` tool's answer: the source under test from the
        dependencies, with the cached header on a re-read of a range seen."""
        self.log_thinking("read_source")
        name = Path(self.deps.target_file_path).name
        content = self.deps.target_file_content or ""
        return self.cached_read(
            "read_source", content=content, name=name,
            cached_key="read_source_cached",
            cached_subs={
                "name": name,
                "line_count": agent_helpers.line_count(content),
                "char_count": len(content),
            },
            loop_breaker_key="read_source_loop_breaker_prefix",
            start_line=start_line, end_line=end_line,
        )

    def read_tests(self, start_line: int = 1, end_line: Optional[int] = None) -> str:
        """Read your working file (line-numbered),
        not the source under test (that is ``read_source``). Pass
        ``start_line`` / ``end_line`` to page a file past the size limit.
        """
        return self._working_file_read(start_line, end_line)

    def _working_file_read(self, start_line: int, end_line: Optional[int]) -> str:
        """The ``read_tests`` tool's answer: the working file as it is on disk."""
        self.log_thinking("read_tests")
        path = self.target_path()
        name = path.name if path is not None else "working file"
        return self.file_read(
            "read_tests", path=path, name=name,
            missing_msg=self.tool_msg("read_tests_no_file"),
            read_error_key="read_tests_read_error",
            start_line=start_line, end_line=end_line,
        )

    # Shared run_tests: one skeleton, one classify core, per-agent hooks

    def run_tests(self) -> str:
        """Run the tests for the file you are editing and report green or red.

        Returns a compact summary: pass / fail counts, the first failing
        tracebacks, and any collection errors. A green run that emits
        unsound-async warnings (unawaited coroutines, leaked tasks,
        unraisable exceptions) is reported as RED -- a falsely-passing test
        is not green. A byte-identical re-run with no edit since the last
        call returns the cached result; edit the file first to make progress.
        """
        self.log_thinking("run_tests")
        cached = self.run_tests_cached_skip()
        if cached is not None:
            return self.track_output(cached)
        _passed, output = self._run_tests_uncached("run_tests")
        return self.track_output(output)

    def _run_tests_uncached(self, log_prefix: str) -> "tuple[bool, str]":
        """Run the tests, classify the outcome and apply the agent's post-run
        policy, returning ``(passed, output)``. ``before_run_tests`` may return
        a guard message instead. A timeout goes to ``on_timeout``.
        ``_classify_strict_green`` gives the effective return code.
        ``after_run_tests`` builds the model-facing text and runs the phase's
        policy. The caller decides on the cache: the ``run_tests`` tool checks
        ``run_tests_cached_skip`` first."""
        guard = self.before_run_tests()
        if guard is not None:
            return self.session.last_tests_passed, guard
        session = self.session
        session.run_tests_calls += 1
        backend = self.test_backend()
        timeout = self.run_tests_timeout()
        res = normal_run.run(
            backend, self.test_paths(), self.deps.source_dir,
            timeout=timeout, target=self.deps.target_file_path, tb="short",
            record_leaks=True,
            timeout_journal=bool(self.journal_request(backend, self.deps.source_dir)),
        )
        if res.timed_out:
            return False, self.on_timeout(
                timeout, log_prefix,
                active_test=getattr(res, "active_at_end", None))
        _test_output.log_test_summary(
            res.output or "", res.returncode, self.log, res=res,
        )
        effective_returncode, unsound_warning = self._classify_strict_green(
            res, backend, log_prefix,
        )
        session.last_tests_passed = effective_returncode == 0
        if effective_returncode == 0:
            session.consecutive_timeouts = 0
            # A green run's notes are the STANDS ALONE notes (a run the
            # strict-green rules turned red has none). Those that name tests
            # hold the goal end (``_run_notes``).
            session.stands_alone_noted = bool(session.stands_alone_note)
        output = self.after_run_tests(res, effective_returncode, unsound_warning)
        # ``after_run_tests`` can change the outcome (a revert restores a green
        # snapshot), so ``last_tests_passed`` decides, not the exit code.
        return session.last_tests_passed, output

    def _classify_strict_green(
        self, res: Any, backend: Any, log_prefix: str,
    ) -> "tuple[int, Optional[str]]":
        """``(effective_returncode, unsound_warning)`` of a completed normal run
        under the strict-green rules. Red: failures or errors despite exit 0
        (Maven's failure.ignore). A green run with unsound async warnings
        (unawaited coroutines, leaked tasks, unraisable exceptions), which
        would poison a snapshot or a measurement. The warning is the
        strict-green block of a run those rules turned red, or the notes on a
        green run (``_run_notes``), else None. A run red by its counts gets
        none."""
        effective_returncode = res.returncode
        failed = int(getattr(res, "failed", 0) or 0)
        errors = int(getattr(res, "errors", 0) or 0)
        if effective_returncode == 0 and (failed + errors) > 0:
            effective_returncode = 1
            self.log(
                f"[{log_prefix}] Reclassifying as RED: runner exit was 0 "
                f"(Maven failure-ignore) but suite has {failed} failure(s) "
                f"and {errors} error(s).",
                level=1,
            )
        unsound_warning: Optional[str] = None
        if (
            self.agent_config()["reject_unsound_async_green"]
            and effective_returncode == 0
        ):
            unsound_warning = agent_helpers.unsound_async_rejection(res, self.tool_msg)
            if unsound_warning is not None:
                effective_returncode = 1
                self.log(
                    f"[{log_prefix}] Strict-green override: unsound-async "
                    "warnings in a green run, treating as RED.",
                    level=1,
                )
        if effective_returncode == 0 and unsound_warning is None:
            unsound_warning = self._run_notes(res, backend, log_prefix)
        return effective_returncode, unsound_warning

    def goal_waits_reply(self) -> str:
        """The reply of a measurement at the goal while the latest green run
        carries a STANDS ALONE note: the goal end waits for a green run
        without one, within the loop's caps."""
        _reads, _writes, run_m, measure_m = self.tool_methods()
        return self.tool_msg("goal_waits_for_note", edit=self.edit_tool_name,
                             run=run_m.__name__, measure=measure_m.__name__,
                             note=self.session.stands_alone_note)

    def _reverse_order_note(self) -> bool:
        """Whether a passing run gets the reverse-order note: the
        ``report_reverse_order_failures`` setting, for a language that runs
        a suite as delivered in reverse order. A language that refuses that
        order gets no reverse run and no note, as one without runs as
        delivered: the note would come with every passing run."""
        if not self.agent_config().get("report_reverse_order_failures"):
            return False
        offers = getattr(self.test_backend(), "offers_reverse_order", None)
        return callable(offers) and offers(self.deps.source_dir)

    @staticmethod
    def journal_request(backend: Any, source_dir: str) -> Dict[str, Any]:
        """The per-test timeout journal for a loop's run, from a backend
        whose project's adapter takes it, so a run a timeout ends names
        the test that was running."""
        offers = getattr(backend, "offers_timeout_journal", None)
        return ({"timeout_journal": True} if callable(offers) and offers(source_dir)
                else {})

    def _run_notes(self, res: Any, backend: Any, log_prefix: str) -> Optional[str]:
        """The notes on a passing normal run, which decide nothing: the tests
        that left the module under test or a library setting changed (the
        normal run's leak record) and, with the reverse-order diagnostic, the
        tests that fail in the file's reverse order. The notes that name
        tests become the session's ``stands_alone_note``, which holds the
        goal end. A reverse run that could not run names none."""
        leaks: Dict[str, Dict[str, Any]] = {}
        for record in getattr(res, "leaks", None) or ():
            leaks.setdefault(record.test, loadable.leak_entry(record))
        parts = []
        naming = []
        if leaks:
            lines = "".join(f"  - {test}: {loadable.describe_leak(leaks[test])}\n"
                            for test in sorted(leaks))
            parts.append(self.tool_msg(
                "tests_stands_alone_note",
                leaks_block=("Tests that leave the module under test or a library "
                             "setting changed:\n" + lines)))
            naming.append(parts[-1])
        reverse = self._reverse_run(backend, log_prefix) if self._reverse_order_note() else None
        if reverse is not None and (reverse.reason is not None
                                    or reverse.failing):
            if reverse.reason is not None:
                order = ("The file passed in file order but could not run "
                         f"in reverse order: {reverse.reason}\n")
            else:
                order = ("These tests passed in file order but failed in "
                         "reverse order:\n"
                         + "".join(f"  - {test}\n"
                                   for test in reverse.failing))
            parts.append(self.tool_msg("tests_stands_alone_order",
                                       order_block=order))
            if reverse.reason is None:
                naming.append(parts[-1])
        self.session.stands_alone_note = "".join(naming)
        return "".join(parts) or None

    def _reverse_run(self, backend: Any, log_prefix: str) -> Any:
        """The file's run as delivered in reverse order, kept per file
        content, so an unchanged file is not run again."""
        paths = self.test_paths()
        digest = hashlib.sha256()
        for path in paths:
            digest.update(str(path).encode("utf-8") + b"\0")
            try:
                digest.update(Path(path).read_bytes())
            except OSError:
                pass
        key = digest.hexdigest()
        if key in self.session.delivered_verdicts:
            self.log(f"[{log_prefix}] reverse order: the file is unchanged, "
                     "so its run is reused", level=1)
            return self.session.delivered_verdicts[key]
        reverse = backend.run_as_delivered(
            paths, self.deps.source_dir, timeout=self.run_tests_timeout(), reverse=True)
        self.session.delivered_verdicts[key] = reverse
        return reverse

    def _canonical_view(self, res: Any, unsound_warning: Optional[str],
                        effective_returncode: Optional[int] = None) -> str:
        """Render the shared test summary, compact failure detail and strict-green warning.
        ``effective_returncode`` is the verdict after the strict-green rules:
        a run they turned red reads as failed and gets the red detail."""
        red = None if effective_returncode is None else effective_returncode != 0
        summary = _test_output.canonical_test_summary_line(res, returncode=effective_returncode)
        body = summary + "\n\n" + self.compact_output(
            res.output, res.returncode, red=red,
            failures=getattr(res, "failures", ()) or (),
        )
        return unsound_warning + body if unsound_warning else body

    # Per-agent hooks (the only variation in the shared tool bodies)

    def before_run_tests(self) -> Optional[str]:
        """The pre-run hook: invalidate per-run caches and, when the agent cannot
        run yet (no test file, an unsupported language), set
        ``last_tests_passed`` and return a guard message instead. None by default."""
        return None

    def run_tests_cached_skip(self) -> Optional[str]:
        """Return cached test output when suite bytes are unchanged, otherwise None."""
        session = self.session
        if session.tests_changed_since_run_tests or not session.last_run_tests_result:
            return None
        self.log(
            "[run_tests] Skipped: tests unchanged since the last run. "
            "Returning the cached result.",
            level=1,
        )
        return agent_helpers.loop_breaker_check(
            session.loop_breaker, "run_tests",
            self.tool_msg(
                "run_tests_cached_refused" if self.refused_by() else "run_tests_cached_rejected",
                last_run_tests_result=session.last_run_tests_result,
                refused_by=self.refused_by_phrase(),
            ),
        )

    def record_refusal(self, tool: str, result: str) -> None:
        """A measurement ``tool`` ("coverage" or "mutation") refused: the
        suite needs an edit as after a red run_tests, so the loop returns to
        editing and a measurement of the unchanged file is refused, but the
        refusal is remembered as the tool's (``refused_by``)."""
        session = self.session
        session.last_tests_passed = False
        session.last_run_tests_result = result
        session.tests_changed_since_run_tests = False
        session.refusal = (tool, result)

    def refused_by(self) -> str:
        """The tool whose refusal is the latest result of the red suite, or
        "" when none is: a run or a restore since replaces it."""
        session = self.session
        tool, result = session.refusal
        if tool and not session.last_tests_passed and session.last_run_tests_result == result:
            return tool
        return ""

    def refused_by_phrase(self) -> str:
        """``refused_by`` as the replies name it."""
        return {"coverage": "the coverage tool",
                "mutation": "the mutation tool"}.get(self.refused_by(), "")

    def test_paths(self) -> List[str]:
        """Test paths passed to the language backend for this target."""
        raise NotImplementedError

    def test_backend(self) -> Any:
        """Resolve the test execution backend from source_language."""
        return languages.get_backend(self.source_language())

    def run_tests_timeout(self) -> int:
        """Test execution timeout from the agent's suite_run_timeout_seconds setting."""
        return int(self.agent_config()["suite_run_timeout_seconds"])

    def on_timeout(self, timeout: int, log_prefix: str,
                   active_test: Optional[str] = None) -> str:
        """Handle a run that exceeded ``timeout``: set state and return the
        model-facing message, a run not passed. ``active_test`` is the test
        the timeout journal shows running when the run was cut off, None
        when the journal names none. A loop can end the phase first
        (``_timeout_before_edits``). Otherwise ``_timeout_message`` decides.
        The bytes count as run, so without a revert the loop returns to
        editing."""
        session = self.session
        self.log(
            f"[{log_prefix}] TIMEOUT: tests exceeded {timeout}s limit"
            + (f" (running: {active_test})" if active_test else ""),
            level=1,
        )
        from testuary import results

        label = self.phase_label()
        self.phase_log("WARNING: " + results.note(
            label, f"{label}.suite_run_timeout_seconds", timeout))
        session.last_tests_passed = False
        session.consecutive_timeouts += 1
        msg = self._timeout_before_edits(timeout)
        if msg is None:
            msg = self._timeout_message(timeout, active_test)
        session.last_run_tests_result = msg
        # A revert leaves the snapshot's bytes, unchanged since their own run.
        session.tests_changed_since_run_tests = False
        return msg

    def _timeout_before_edits(self, timeout: int) -> Optional[str]:
        """The message when a timeout before any edit ends the phase, else
        None (the default)."""
        return None

    def timeout_suspect_path(self) -> Optional[Path]:
        """The file whose tests added since the best green a timeout names as
        suspects."""
        return self.target_path()

    def _timeout_message(self, timeout: int, active_test: Optional[str]) -> str:
        """When the journal names the running test (``active_test``), the file
        stays, the model is told the test and the timeout counts as a failed
        run (the restore after max_failed_test_attempts applies). Otherwise
        the tests added since the last green in time are named as suspects
        and the file returns to that green. The phase ends after
        suite_timeout_stop_after timeouts when a best-green exists."""
        session = self.session
        snap = session.best_green.snapshot
        stop_after = int(self.agent_config()["suite_timeout_stop_after"])
        stop_due = (snap is not None and stop_after > 0
                    and session.consecutive_timeouts >= stop_after)
        if active_test:
            msg = self.tool_msg("tests_timeout_named", timeout=timeout, test=active_test)
            max_attempts, should_restore = self.failed_run_due()
            if should_restore and self._restore_snapshot(self.failed_run_target()):
                self.phase_log(
                    f"[restore] {max_attempts} consecutive failed runs - "
                    f"reverted {self.restored_file_phrase} to the best-green snapshot.",
                    level=1,
                )
                msg = self.tool_msg("tests_restore_note", max_attempts=max_attempts,
                                    output=msg)
            return self._timeout_stop(timeout) if stop_due else msg

        # Without a best-green there is nothing to revert to or end with: the
        # model is told and edits.
        if snap is None:
            return self.tool_msg("tests_timeout", timeout=timeout)

        suspects: List[str] = []
        try:
            path = self.timeout_suspect_path()
            current = (
                path.read_text(encoding="utf-8")
                if path is not None and path.exists() else ""
            )
            suspects = agent_helpers.tests_added_since(
                self.test_backend(), current, snap.get("content", ""),
            )
        except Exception:
            suspects = []

        # Revert to the best-green snapshot (the best-scoring suite that ran
        # in time), back in the state its bytes left.
        restored = self._restore_snapshot(self.failed_run_target())
        if stop_due:
            return self._timeout_stop(timeout)
        if restored:
            return self.tool_msg(
                "tests_timeout_restored",
                timeout=timeout,
                test_count=snap.get("test_count"),
                suspects=agent_helpers.suspect_names(suspects),
            )
        # The file already holds the snapshot: nothing to revert to.
        return self.tool_msg("tests_timeout", timeout=timeout)

    def _timeout_stop(self, timeout: int) -> str:
        """End the phase on repeated timeouts with the best-green result."""
        session = self.session
        session.early_stop_result = self.synthesize_stop_result("repeated_test_timeout")
        session.early_stop_reason = "repeated_test_timeout"
        return self.tool_msg(
            "tests_timeout_abort",
            timeout=timeout, count=session.consecutive_timeouts,
        )

    def failed_run_due(self) -> Tuple[int, bool]:
        """Count one failed run. Return the max_failed_test_attempts setting
        and whether the failed runs in a row make the best-green snapshot
        due back now."""
        max_attempts = int(self.agent_config()["max_failed_test_attempts"])
        due, self.session.best_green.consecutive_failed_runs = (
            agent_helpers.regression_revert_decision(
                regression_count=self.session.best_green.consecutive_failed_runs,
                is_regression=True,
                n_restore=max_attempts,
            )
        )
        return max_attempts, due

    def compact_output(self, full_out: str, returncode: int,
                       red: Optional[bool] = None, failures: Any = ()) -> str:
        """The run output as the compact failure view the model sees. ``red`` is
        the outcome the strict-green rules decided, else the exit code decides.
        ``failures`` are the failing tests of the structured report. The
        shared compactor, and for an agent with a measure tool the red-suite
        footer naming it, also on a run the strict-green rules turned red."""
        compact = _test_output.compact_test_output(
            full_out, returncode, failures=failures, frame_files=self.frame_files())
        if not self.measure_tool_name:
            return compact
        return compact + _test_output.red_suite_footer(
            full_out, returncode if red is None else int(red),
            measure_tool=self.measure_tool_name, failures=failures)

    def frame_files(self) -> Tuple[str, ...]:
        """The file names whose stack frames a failure report shows: the
        target's and the tests'."""
        names = [Path(str(self.deps.target_file_path)).name]
        try:
            names += [Path(str(path)).name for path in self.test_paths()]
        except NotImplementedError:
            pass
        return tuple(dict.fromkeys(names))

    #: How the restore log names the file a failed-run restore reverted.
    restored_file_phrase: str = ""

    def phase_log(self, message: str, level: int = 0) -> None:
        """Log one of the loop's own lines. Through ``log`` by default."""
        self.log(message, level=level)

    def after_green_run(self) -> None:
        """React to a green run, after the red streak is reset. Nothing by
        default."""

    def failed_run_target(self) -> Optional[Path]:
        """The file a restore after repeated failed runs reverts."""
        return self.target_path()

    def snapshot_score_text(self, snap: Dict[str, Any]) -> str:
        """The restored snapshot's scores, as the restore log names them."""
        raise NotImplementedError

    def after_run_tests(
        self, res: Any, effective_returncode: int, unsound_warning: Optional[str],
    ) -> str:
        """Apply post-run policy, update cached results and return model-facing
        output: a collection error's lines first, and after repeated failed
        runs (``failed_run_due``) the best-green snapshot restored."""
        session = self.session
        output = self._canonical_view(res, unsound_warning, effective_returncode)

        # A test file that does not compile or link, or a collection error
        # (exit code 2 without results), puts its error lines first.
        err_summary = _test_output.compile_error_summary(res)
        if err_summary is None:
            err_summary = _test_output.collection_error_summary(
                res.output or "", res.returncode,
            )
        if err_summary is not None:
            output = self.tool_msg(
                "tests_collection_error", err_summary=err_summary,
            ) + "\n" + output

        # The run evaluated the bytes on disk. A restore below leaves the
        # snapshot's bytes, which are unchanged since their own run.
        session.tests_changed_since_run_tests = False
        if effective_returncode == 0:
            session.last_tests_passed = True
            session.best_green.consecutive_failed_runs = 0
            self.after_green_run()
        else:
            session.last_tests_passed = False
            max_attempts, should_restore = self.failed_run_due()
            snap = session.best_green.snapshot
            if should_restore and self._restore_snapshot(self.failed_run_target()):
                self.phase_log(
                    f"[restore] {max_attempts} consecutive failed "
                    f"run_tests calls - reverted {self.restored_file_phrase} to the "
                    f"best-green snapshot ({self.snapshot_score_text(snap)}).",
                    level=1,
                )
                output = self.tool_msg(
                    "tests_restore_note",
                    max_attempts=max_attempts,
                    output=output,
                )

        session.last_run_tests_result = output
        return output

    def target_path(self) -> Any:
        """The file this agent edits, which the subclass resolves from ``self.deps``."""
        raise NotImplementedError

    def sandbox_root(self) -> str:
        """The directory writes must stay under: the test output directory."""
        return self.deps.test_output_dir

    def source_language(self) -> str:
        """Language used for the post-edit parse check, from the agent dependencies."""
        return self.deps.language

    def _shadowed_definition_reject(self, new_content: str) -> Optional[str]:
        """The rejection of a write that leaves a same-scope definition
        shadowed, by the backend's rule (the one every normalization uses), or None."""
        if not self.guards_duplicate_test_names:
            return None
        result = self.test_backend().normalize_shadowed_definitions(
            new_content
        )
        if not result.changed:
            return None
        dupes = list(dict.fromkeys(
            ".".join((*binding.scope, binding.name))
            for binding in result.removed_bindings
        ))
        return self.tool_msg("edit_duplicate_test_name",
                             names=", ".join(dupes))

    def assert_target_has_no_shadowed_definitions(self, stage: str) -> None:
        """Raise RuntimeError when the working suite does not parse or has
        shadowed definitions: entry normalization and the edit gate prevent
        both, and rewriting a measured suite would leave its scores stale."""
        if not self.guards_duplicate_test_names:
            return
        path = self.target_path()
        if path is None or not path.exists():
            return
        source = path.read_text(encoding="utf-8")
        result = self.test_backend().normalize_shadowed_definitions(source)
        if not result.parse_ok:
            raise RuntimeError(
                f"{stage} suite does not parse under "
                f"{result.rule_version}: {result.parse_error}"
            )
        if result.changed:
            names = ", ".join(dict.fromkeys(
                ".".join((*binding.scope, binding.name))
                for binding in result.removed_bindings
            ))
            raise RuntimeError(
                f"{stage} suite contains shadowed definitions ({names}). "
                "The suite was not rewritten because its measurements would "
                "be stale"
            )

    def generated_test_identity(self):
        identity = getattr(self, "_generated_test_identity", None)
        if identity is None:
            identity = self.test_backend().generated_test_target_identity(
                self.deps.target_file_path, self.deps.source_dir,
            )
            self._generated_test_identity = identity
        return identity

    def _generated_test_validity_reject(
        self, existing: str, proposed: str,
    ) -> Optional[str]:
        if not self.guards_generated_test_validity:
            return None
        identity = self.generated_test_identity()
        findings = self.test_backend().analyze_generated_test_validity(
            proposed, identity,
        )
        if not findings:
            self._pending_generated_test_validity = validity.direct_record(
                identity=identity,
                analyzed_source=proposed,
                audit_flags=self.test_backend().screen_generated_test_source(
                    proposed, identity,
                ),
                stage=f"{self.phase_label()}.accepted_edit",
            )
            return None

        validity.append_jsonl(
            self.resolve_log_dir() / "generated_test_validity_rejections.jsonl",
            {
                "schema_version": validity.RECORD_SCHEMA_VERSION,
                "record_type": "edit_rejection",
                "phase": self.phase_label(),
                "request_idx": int(getattr(
                    self.session.usage_tracker_ref, "requests_used", 0,
                )),
                "policy_version": identity.policy_version,
                "target_identity": identity.record(),
                "working_source_sha256": source_sha256(existing),
                "proposed_source_sha256": source_sha256(proposed),
                "findings": [asdict(finding) for finding in findings],
                "write_applied": False,
                **timing_fields(),
            },
        )
        first = findings[0]
        return self.tool_msg(
            "edit_generated_test_validity",
            test=first.enclosing_test or "shared test setup",
            operation=first.operation,
        )

    def _persist_generated_test_validity_write(self) -> None:
        record = getattr(self, "_pending_generated_test_validity", None)
        if record is None:
            return
        validity.append_jsonl(
            self.resolve_log_dir() / "generated_test_validity_writes.jsonl",
            record,
        )
        self._pending_generated_test_validity = None

    def set_tests_changed(self, changed: bool) -> None:
        """Set whether the working suite changed since its last measurement."""
        raise NotImplementedError

    def after_write(self) -> None:
        """Invalidate this agent's measurement caches after a successful write
        and count it: the file is unverified until the next ``run_tests``."""
        self.set_tests_changed(True)
        self.session.tests_changed_since_run_tests = True
        self.session.writes_done += 1

    # Shared output validator: a voluntary return waits for a designed end

    # ``validate_result_fields`` above is the output-content hook.

    #: The validator's refusals of a loop's measurement, as prompt keys: none
    #: yet (``measurement_precondition``) and one older than the latest edit
    #: (``measurement_staleness``).
    no_measurement_key: str = ""
    stale_measurement_key: str = ""

    def last_measured_value(self) -> Optional[float]:
        """The loop's goal metric from its last measurement, None before one."""
        raise NotImplementedError

    def measurement_precondition(self) -> Optional[str]:
        """Return a rejection message if no meaningful phase measurement exists, otherwise None."""
        if not self.no_measurement_key:
            raise NotImplementedError
        # The start's measurement counts as the loop's first while it still
        # describes the suite. After an edit only a measurement of
        # the model's own does.
        start_counts = (self.session.initial_measured and self.last_measured_value() is not None
                        and not self.tests_changed_since_measurement())
        if self.measurement_count() == 0 and not start_counts:
            return self.tool_msg(self.no_measurement_key)
        return None

    def measurement_staleness(self) -> Optional[str]:
        """The rejection when the measurement on record predates the latest edit,
        else None. Checked last at a voluntary acceptance, which takes the bytes
        its measurement describes. None without a ``stale_measurement_key``."""
        if self.stale_measurement_key and self.tests_changed_since_measurement():
            return self.tool_msg(self.stale_measurement_key)
        return None

    def goal_met(self) -> bool:
        """Whether this run reached its phase goal (coverage / mutation
        threshold). Default False: an agent that signals its goal through
        ``early_stop_result`` leaves this False."""
        return False

    def early_stop_allowed(self) -> bool:
        """Whether a count that reached max_measurements_without_improvement may end the phase now.
        True by default. A pending switch of criterion (the selective to the
        full catalogue) defers, so a plateau switches instead of ending it."""
        return True

    def measurement_count(self) -> int:
        """The count of real measurements this run, used to tell a new
        measurement (progress) from a bare re-submission in the spam cap."""
        raise NotImplementedError

    def synthesize_stop_result(self, reason: str) -> Any:
        """The canonical result a designed stop for ``reason`` records on
        ``early_stop_result``. Default None: the agent's entry point
        finalizes from session state instead."""
        return None

    def designed_stop(self, result: Any, reason: str) -> str:
        """Record ``result`` as the phase's result for the designed end
        ``reason``, and return the directive that asks for the result call."""
        self.session.early_stop_result = result
        self.session.early_stop_reason = reason
        return (
            "[FRAMEWORK STOP] Designed end condition reached. Call "
            f"``{self.final_tool_name}`` now and no other tool. The "
            "framework has synthesised the canonical result from state "
            "and will override whatever you emit."
        )

    def early_stop_reject_message(self) -> str:
        """The phase-specific message returned when a voluntary return is
        refused for not yet reaching the goal / limit of measurements without improvement."""
        raise NotImplementedError

    def validate_output(self, state: Any, parsed: Any) -> Optional[str]:
        """The rejection of a voluntary model return before a designed end, or
        None to accept (the entry point then uses ``early_stop_result`` when a
        tool synthesized one). The checks' order and the submission cap are here."""
        session = self.session
        # A designed stop a tool synthesized is accepted first: the entry point
        # uses early_stop_result, so checking the parsed fields would waste requests.
        if session.early_stop_result is not None:
            return None
        head = self.validate_result_fields(parsed)
        if head is not None:
            return head
        # A voluntary stop needs a green suite and a measurement a later phase
        # can read. Only the framework's own stop (above) and the request limit
        # end a run without them.
        if not session.last_tests_passed:
            self.log(
                "[validator] Rejected: test suite not green at end of run.",
                level=1,
            )
            return self.tool_msg("validator_not_green")
        measurement_gate = self.measurement_precondition()
        if measurement_gate is not None:
            return measurement_gate

        cfg = self.agent_config()
        early_stop = bool(cfg["allow_early_stop"])
        max_measurements_without_improvement = int(cfg["max_measurements_without_improvement"])
        goal = self.goal_met()
        early_stop_due = (
            early_stop and max_measurements_without_improvement > 0
            and session.measurements_without_improvement >= max_measurements_without_improvement
            and self.early_stop_allowed()
        )
        if goal or early_stop_due:
            stale = self.measurement_staleness()
            if stale is not None:
                self.log(
                    "[validator] Rejected: the criterion measurement on "
                    "record predates the latest edit.",
                    level=1,
                )
                return stale
            return None

        # A re-submission without a new measurement counts on its own counter,
        # apart from the no-improvement count the measure tools advance, so the
        # recorded stop reason (no improvement or submission cap) names what happened.
        submit_cap = int(cfg["max_final_answers_without_measurement"])
        count = self.measurement_count()
        progress = (
            session.measures_at_last_submission is not None
            and count > session.measures_at_last_submission
        )
        if progress:
            session.early_submission_rejections = 0
        else:
            session.early_submission_rejections += 1
        session.measures_at_last_submission = count
        if (
            early_stop
            and submit_cap > 0
            and session.early_submission_rejections >= submit_cap
        ):
            synth = self.synthesize_stop_result("final_answers_without_measurement")
            if synth is not None:
                session.early_stop_result = synth
                session.early_stop_reason = "final_answers_without_measurement"
            self.log(
                f"[validator] FRAMEWORK STOP: result re-submitted "
                f"{session.early_submission_rejections}/{submit_cap} times with "
                f"no new measurement between -> accepting.",
                level=1,
            )
            return None

        self.log(
            f"[validator] Rejected premature {type(parsed).__name__}: "
            f"goal_met={goal}, "
            f"without_improvement={session.measurements_without_improvement}/{max_measurements_without_improvement}, "
            f"submissions={session.early_submission_rejections}/{submit_cap}. "
            f"Keep working until one of these conditions fires.",
            level=1,
        )
        return self.early_stop_reject_message()

    # Shared editing tool: surgical content replace

    def sandbox_guard(self, path: Any) -> Optional[str]:
        """None when ``path`` lies under ``sandbox_root()``, else the refusal every
        write tool returns (an escape is a wiring bug, not a model action),
        logged under the current tool's name."""
        try:
            agent_helpers.assert_write_under(
                str(path), self.sandbox_root(),
            )
            return None
        except agent_helpers.SandboxEscapeError as exc:
            tool = self.session.current_tool_name or "write"
            self.log(f"[{tool}] BLOCKED: {exc}", level=1)
            return (
                f"BLOCKED by sandbox gate: {exc}. Refused to protect canonical "
                "source. Report this as a bug."
            )
