"""Immutable observations for terminal clients, not execution or accounting
state, and the run's plan as displays show it."""
from __future__ import annotations

import dataclasses
import functools
import math
import typing
from dataclasses import dataclass
from typing import Any, Literal, Mapping

#: ``limit_reached``: the work stopped at a limit its settings set (a loop's
#: requests, tokens or time), an ordinary end and no failure.
State = Literal["running", "waiting", "retrying", "succeeded", "failed", "cancelled", "refused",
                "limit_reached"]
Basis = Literal["measured", "progress", "reused", "unavailable", "incomplete",
                "unsupported", "failed"]
Provenance = Literal["observed", "recorded", "derived", "synthetic"]
Level = Literal["run", "target", "phase", "step", "operation"]
STATES = ("running", "waiting", "retrying", "succeeded", "failed", "cancelled", "refused",
          "limit_reached")
BASES = ("measured", "progress", "reused", "unavailable", "incomplete",
         "unsupported", "failed")
PROVENANCE = ("observed", "recorded", "derived", "synthetic")
LEVELS = ("run", "target", "phase", "step", "operation")
Failure = Literal["execution", "outcome"]
FAILURES = ("execution", "outcome")


def record_from_dict(cls: type, data: Mapping[str, Any]) -> Any:
    """A record (``cls``, one of these frozen dataclasses) from its mapping
    as ``dataclasses.asdict`` gives it and an event file or
    ``completion.json`` holds it, checked as every record is. A field that
    holds a record, an optional one or a tuple of them is built the same
    way. A missing or null tuple of records reads as empty."""
    fields = dict(data)
    for name, record, many in _record_fields(cls):
        if many:
            fields[name] = tuple(record_from_dict(record, item) for item in fields.get(name) or ())
        elif fields.get(name) is not None:
            fields[name] = record_from_dict(record, fields[name])
    return cls(**fields)


@functools.lru_cache(maxsize=None)
def _record_fields(cls: type) -> tuple:
    """(name, record type, whether a tuple of them) of each field of
    ``cls`` that holds records."""
    found = []
    for name, kind in typing.get_type_hints(cls).items():
        args = typing.get_args(kind)
        if dataclasses.is_dataclass(kind):
            found.append((name, kind, False))
        elif typing.get_origin(kind) is tuple and args and dataclasses.is_dataclass(args[0]):
            found.append((name, args[0], True))
        else:
            found += [(name, arg, False) for arg in args if dataclasses.is_dataclass(arg)][:1]
    return tuple(found)


def _count(name: str, value: int | None) -> None:
    if value is not None and (type(value) is not int or value < 0):
        raise ValueError(f"{name} must be a nonnegative integer or None")


@dataclass(frozen=True)
class Metric:
    """A named value in the producer's units, with its measurement status.

    Percentage values use 0 to 100, not a fraction. A progress value is never
    described as a final score. None is unavailable, not an observed zero.
    goal is the configured target in the same unit (for example a coverage goal).
    It is shown beside the value, never as progress toward completion.
    """

    name: str
    value: float | None = None
    unit: str = ""
    basis: Basis = "measured"
    goal: float | None = None

    def __post_init__(self) -> None:
        if not self.name or self.basis not in BASES:
            raise ValueError("a metric needs a name and a recognized basis")
        if self.value is not None and not math.isfinite(self.value):
            raise ValueError("metric values must be finite or None")
        if self.goal is not None and (isinstance(self.goal, bool) or not math.isfinite(self.goal)):
            raise ValueError("a metric goal must be finite or None")


@dataclass(frozen=True)
class Usage:
    """Already-accounted usage. Omitted provider observations remain unknown.

    logical_request_limit is the configured ceiling for the phase, a limit and
    not a prediction of how many requests the phase will need.
    """

    logical_requests: int | None = None
    physical_requests: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    cached_input_tokens: int | None = None
    cost_usd: float | None = None
    logical_request_limit: int | None = None

    def __post_init__(self) -> None:
        for name in ("logical_requests", "physical_requests", "input_tokens",
                     "output_tokens", "cached_input_tokens", "logical_request_limit"):
            _count(name, getattr(self, name))
        if self.cost_usd is not None and (
            not math.isfinite(self.cost_usd) or self.cost_usd < 0
        ):
            raise ValueError("cost_usd must be nonnegative and finite or None")


@dataclass(frozen=True)
class DisplayEvent:
    """One observation of an operation, emitted after the producer recorded
    its facts. It cannot control execution, retries, reduction or what is
    saved.

    The identity tells concurrent work apart and shows only at the debug
    level. ``elapsed_s`` is the run's clock, ``completed`` and ``total`` count
    actual work, not a request limit, and ``message`` is a short diagnostic,
    never a prompt or a model reply. ``level`` places it in the layered view
    (run, target, phase, step or operation), or None to infer it from the
    identity. ``label`` is its name in plain words. ``log_path`` names the
    run's detailed logs on a run-level observation. ``failure``, on a failed
    one, tells an execution failure ("execution") from a result of the method
    ("outcome", such as a target without a suite that meets the
    minimums), or None.

    ``state`` is running, waiting, retrying, succeeded, failed, cancelled,
    refused or limit_reached. limit_reached is work that stopped at a limit
    its settings set (a loop's requests, tokens or time), an ordinary end
    and no failure.
    """

    run_id: str
    target: str
    phase: str
    operation: str
    state: State
    elapsed_s: float
    message: str = ""
    completed: int | None = None
    total: int | None = None
    work_unit: str = "items"
    metrics: tuple[Metric, ...] = ()
    usage: Usage | None = None
    provenance: Provenance = "observed"
    level: Level | None = None
    label: str = ""
    log_path: str = ""
    failure: Failure | None = None

    def __post_init__(self) -> None:
        if any(not value for value in (self.run_id, self.target, self.phase,
                                       self.operation, self.work_unit)):
            raise ValueError("display identities and work_unit must not be empty")
        if self.state not in STATES or self.provenance not in PROVENANCE:
            raise ValueError("unrecognized display state or provenance")
        if self.level is not None and self.level not in LEVELS:
            raise ValueError("unrecognized display level")
        if not isinstance(self.label, str) or not isinstance(self.log_path, str):
            raise ValueError("label and log_path must be text")
        if self.failure is not None and (self.failure not in FAILURES or self.state != "failed"):
            raise ValueError("failure is execution or outcome, on a failed observation only")
        if not math.isfinite(self.elapsed_s) or self.elapsed_s < 0:
            raise ValueError("elapsed_s must be nonnegative and finite")
        _count("completed", self.completed)
        _count("total", self.total)
        if (self.completed is not None and self.total is not None
                and self.completed > self.total):
            raise ValueError("completed work must not exceed its known total")
        object.__setattr__(self, "metrics", tuple(self.metrics))

    @property
    def identity(self) -> tuple[str, str, str, str]:
        return self.run_id, self.target, self.phase, self.operation

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DisplayEvent":
        """The observation a mapping of its fields describes
        (``record_from_dict``)."""
        return record_from_dict(cls, data)


# What a run will do, as displays show it before and while it runs.
#
# A ``RunPlan`` names the run's files and runs, the first three of its four
# phases in order, each with its state, its steps and its limits, and
# Finishing, the fourth. The orchestrator builds it once from the settings the run
# resolved and the run's options (``testuary.orchestrator.run_plan``). A
# display draws it, so no display derives the plan on its own.

#: A planned phase or step: it runs, it was switched off for this run, or a
#: recorded result is reused in its place.
PLANNED_STATES = ("planned", "off", "reused")


@dataclass(frozen=True)
class PlannedStep:
    """One step of a phase, or Finishing: its name, its label
    in plain words and its state: planned, off when the run stops before
    it (``--stop-after-phase``), or reused when the run takes its results
    from an earlier one (``--reuse``)."""

    step: str
    label: str
    state: str = "planned"

    def __post_init__(self) -> None:
        if not self.step or not self.label or self.state not in PLANNED_STATES:
            raise ValueError("a planned step needs a name, a label and a known state")


@dataclass(frozen=True)
class PlannedLimit:
    """A configured limit or goal of a phase, in its unit. None is no limit."""

    name: str
    value: float | None
    unit: str = ""


@dataclass(frozen=True)
class PlannedPhase:
    """One of the run's first three phases: its name and label, its state
    (planned, off or reused, as a ``PlannedStep``'s), the steps it goes
    through and its limits.

    The steps form three groups: a start that runs once (``steps[:cycle]``),
    a cycle that repeats (``steps[cycle:end]``, empty in the starter phase)
    and the phase end that runs once after it (``steps[end:]``, the
    consolidation of every phase). ``end`` None is the number of steps: no
    phase-end group."""

    phase: str
    label: str
    state: str = "planned"
    steps: tuple[PlannedStep, ...] = ()
    limits: tuple[PlannedLimit, ...] = ()
    cycle: int = 0
    end: int | None = None

    def __post_init__(self) -> None:
        if not self.phase or not self.label or self.state not in PLANNED_STATES:
            raise ValueError("a planned phase needs a name, a label and a known state")
        object.__setattr__(self, "steps", tuple(self.steps))
        object.__setattr__(self, "limits", tuple(self.limits))
        if self.end is None:
            object.__setattr__(self, "end", len(self.steps))
        if not 0 <= self.cycle <= self.end <= len(self.steps):
            raise ValueError("a planned phase's cycle and end must lie within its steps, in order")


@dataclass(frozen=True)
class RunPlan:
    """The run's files (relative to the project's source directory), its runs
    (r0, r1 and so on), its first three phases in order and Finishing, the
    fourth."""

    files: tuple[str, ...]
    runs: tuple[str, ...]
    phases: tuple[PlannedPhase, ...]
    finishing: PlannedStep

    def __post_init__(self) -> None:
        for name in ("files", "runs", "phases"):
            object.__setattr__(self, name, tuple(getattr(self, name)))

    @classmethod
    def from_dict(cls, value: Mapping[str, Any]) -> "RunPlan":
        """The plan from its dictionary (``record_from_dict``)."""
        return record_from_dict(cls, value)

    def summary(self) -> str:
        """The plan in one line, as a display shows it before the run: the
        phases in order with their states, then Finishing."""
        def shown(label: str, state: str) -> str:
            return label if state == "planned" else f"{label} ({state})"

        phases = ", ".join(shown(phase.label, phase.state) for phase in self.phases)
        return f"{phases}, then {shown(self.finishing.label, self.finishing.state)}"
