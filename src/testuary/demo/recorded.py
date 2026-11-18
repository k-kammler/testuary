"""Each call a demonstration records, declared once.

A recorded call is a real method of the run: its kind names the method
(``backend.run_test_files`` is the language backend's ``run_test_files``).
An entry declares the object that answers it (its ``role``), the key the
playback matches it by, and the answer in both directions: what the run
receives from a stored answer (``read``, side effects included, such as a
coverage file the call writes or the lines it logs), and how a real result
is stored (``write``). The playback's methods are made from the entries
(``playback``): they bind the run's arguments by the real method's
signature and ask the oracle. A recording calls the real method of that name
with the run's own arguments (``stored_answer``), so each recorded call has
one declaration, which the playback and the recorder both read.
"""
from __future__ import annotations

import contextlib
import dataclasses
import inspect
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

from langadapter.capabilities import (
    Availability,
    Capability,
    CapabilityReport,
    OperationCapability,
    ToolVersion,
)
from langadapter.outcomes import Outcome
from langadapter.runs import RunDiagnostic
from langadapter.sources import CollisionResolution
from langadapter.suites import TestId
from langadapter.versions import Version
from langchain_core.messages import message_to_dict, messages_from_dict
from langchain_core.outputs import ChatGeneration, ChatResult

from testuary import phase_state
from testuary.generation_evidence import source_sha256
from testuary.demo.scenario import (
    Call,
    ScenarioStop,
    agent_message_hashes,
    canonical,
    file_entry,
    kill_matrix_entry,
    message_hashes,
    sha256_bytes,
    suite_files,
    tree_files,
)
from testuary.languages import DeliveredRun, SourceValidation, TestFailure, TestRunResult
from testuary.languages.generic import DeliveredCase, EvictionData, GenericBackend, LeakRecord
from testuary.phase_state import timing_fields
from testuary.reduction import LosslessReduction, _effective_costs_us


class Arguments:
    """A recorded call's arguments, bound by the real method's signature
    with its defaults: ``owner`` is the object the run called, ``values``
    the arguments by name."""

    def __init__(self, signature: inspect.Signature, owner: Any, args: tuple,
                 kwargs: Dict[str, Any]) -> None:
        bound = signature.bind(owner, *args, **kwargs)
        bound.apply_defaults()
        self.owner = owner
        #: The key the playback asked by, once it is built.
        self.key: Optional[Dict[str, Any]] = None
        self.values: Dict[str, Any] = {name: bound.arguments[name]
                                       for name in list(signature.parameters)[1:]}

    def __getitem__(self, name: str) -> Any:
        return self.values[name]


@dataclass(frozen=True)
class RecordedCall:
    """One recorded call. ``role`` names the object whose method of the
    kind's name answers it in a recording: the language backend, the
    mutation runner, session or measurement, the model or the exact cover's
    solver. ``key`` and ``context`` come
    from the arguments, ``read`` converts a stored answer to what the run
    receives, ``write`` converts the real result to the stored answer.
    ``callbacks`` are the arguments a recording collects instead of passing
    the run's (their calls reach ``write`` and are replayed by ``read``),
    ``omit`` those it leaves out, ``timed`` the run's timed part the call
    counts toward, ``prepare`` normalizes the arguments before both, and
    ``name`` is the real method's name where the kind names it otherwise."""

    kind: str
    role: str
    key: Callable[[Arguments], Dict[str, Any]]
    read: Callable[[Dict[str, Any], Arguments], Any]
    write: Callable[[Any, Arguments, Dict[str, List[tuple]]], Dict[str, Any]]
    context: Callable[[Arguments], Dict[str, Any]] = lambda arguments: {}
    callbacks: Tuple[str, ...] = ()
    omit: Tuple[str, ...] = ()
    timed: Optional[Callable[[Arguments], Optional[str]]] = None
    prepare: Optional[Callable[[Arguments], None]] = field(default=None)
    name: Optional[str] = None

    @property
    def method(self) -> str:
        """The real method's name."""
        return self.name or self.kind.split(".", 1)[1]


def playback(entry: RecordedCall, real: Callable[..., Any]):
    """The method that asks the owner's oracle for ``entry``, bound by the
    signature of ``real``, the real method it stands for, or a function,
    which it stands for as a method of the owner. It takes the named
    arguments only: an argument the key does not hold could change the real
    answer."""
    parameters = [parameter for parameter in inspect.signature(real).parameters.values()
                  if parameter.kind is not inspect.Parameter.VAR_KEYWORD]
    if not parameters or parameters[0].name != "self":
        parameters.insert(0, inspect.Parameter("self", inspect.Parameter.POSITIONAL_ONLY))
    signature = inspect.Signature(parameters)

    def method(owner, *args, **kwargs):
        arguments = Arguments(signature, owner, args, kwargs)
        part = entry.timed(arguments) if entry.timed is not None else None
        with phase_state.timed_part(part) if part else contextlib.nullcontext():
            if entry.prepare is not None:
                entry.prepare(arguments)
            arguments.key = entry.key(arguments)
            call = Call(entry.kind, arguments.key,
                        {**entry.context(arguments), "recorded": entry, "arguments": arguments})
            return entry.read(owner.ask(call), arguments)

    method.__name__ = entry.method
    method.__qualname__ = entry.method
    method.__doc__ = real.__doc__
    method.__signature__ = signature
    return method


def real_result(entry: RecordedCall, real: Any,
                arguments: Arguments) -> Tuple[Any, Dict[str, List[tuple]]]:
    """The result of the method of ``real`` of the entry's name, called with
    the run's arguments, and the calls of its callbacks, which a recording
    collects."""
    collected: Dict[str, List[tuple]] = {name: [] for name in entry.callbacks}
    values = {name: value for name, value in arguments.values.items() if name not in entry.omit}
    for name in entry.callbacks:
        # A call's keyword arguments follow its positional ones, when it has any.
        values[name] = (lambda *args, _name=name, **named: collected[_name].append(
            args + ((named,) if named else ())))
    return getattr(real, entry.method)(**values), collected


def stored_answer(entry: RecordedCall, result: Any, arguments: Arguments,
                  collected: Dict[str, List[tuple]]) -> Dict[str, Any]:
    """A real result as the scenario stores it: JSON, as a reader gets it."""
    return json.loads(json.dumps(entry.write(result, arguments, collected)))


# The stored forms, both ways.

def target_entry(target: Any) -> Optional[List[str]]:
    """A file under test as a key names it: its name and bytes, or its name
    alone when it is no file."""
    if target is None:
        return None
    path = Path(target)
    return file_entry(path) if path.is_file() else [path.name, None]


def _leaks(data: Dict[str, Any]):
    """A recorded run's leak records, None when it recorded none."""
    if data.get("leaks") is None:
        return None
    return tuple(LeakRecord(**{name: tuple(value) if isinstance(value, list) else value
                               for name, value in record.items()})
                 for record in data["leaks"])


def _unsound(data: Dict[str, Any]):
    """A recorded run's unsound-test entries, () when it recorded none."""
    return tuple((test, message) for test, message in data.get("unsound") or ())


def test_run_result(data: Dict[str, Any]) -> TestRunResult:
    """The ``TestRunResult`` ``run_result_json`` stored."""
    def _tuple(value: Any) -> Any:
        return tuple(value) if value is not None else None

    def _count(value: Any) -> Optional[int]:
        # A count the run did not observe stays None.
        return int(value) if value is not None else None

    return TestRunResult(
        ok=bool(data["ok"]),
        returncode=int(data["returncode"]),
        timed_out=bool(data["timed_out"]),
        elapsed_s=(float(data["elapsed_s"])
                   if data["elapsed_s"] is not None else None),
        collected=_count(data["collected"]),
        passed=_count(data["passed"]),
        failed=_count(data["failed"]),
        errors=_count(data["errors"]),
        output=str(data["output"]),
        line_coverage=data["line_coverage"],
        branch_coverage=data["branch_coverage"],
        missing_lines=tuple(data.get("missing_lines") or ()),
        missing_branches=tuple(data.get("missing_branches") or ()),
        covered_lines=_tuple(data.get("covered_lines")),
        covered_branches=_tuple(data.get("covered_branches")),
        total_statements=_count(data.get("total_statements")),
        total_branches=_count(data.get("total_branches")),
        leaks=_leaks(data),
        failures=tuple(TestFailure(**record) for record in data.get("failures") or ()),
        active_at_end=data.get("active_at_end"),
        unsound=_unsound(data),
        diagnostics=tuple(RunDiagnostic(kind, message)
                          for kind, message in data.get("diagnostics") or ()),
        reason_detail=tuple(data.get("reason_detail") or ()),
        stopping_tests=test_ids(data.get("stopping_tests")),
    )


def test_ids(data: Any) -> Tuple[TestId, ...]:
    """The named tests' id objects that ``asdict`` stored."""
    return tuple(TestId(**fields) for fields in data or ())


#: The parts of a recorded run stored only when the run has any, as the
#: shipped recordings hold them, each entry in its stored form: an
#: unsound-test entry and a diagnostic's kind and message as lists, a line of
#: reason detail as text, a named test's id object as its fields.
_STORED_IF_ANY: Dict[str, Callable[[Any], Any]] = {
    "unsound": list,
    "diagnostics": lambda entry: [entry.kind, entry.message],
    "reason_detail": str,
    "stopping_tests": asdict,
}


def run_result_json(result: TestRunResult) -> Dict[str, Any]:
    """A ``TestRunResult`` as scenario JSON. A total the run did not report
    is stored without its key, as the shipped recordings hold it."""
    data = result._asdict()
    for name in ("total_statements", "total_branches"):
        if data[name] is None:
            del data[name]
    for name in ("missing_lines", "missing_branches", "covered_lines",
                 "covered_branches"):
        if data[name] is not None:
            data[name] = list(data[name])
    if data.get("leaks") is not None:
        data["leaks"] = [{name: list(value) if isinstance(value, tuple) else value
                          for name, value in record._asdict().items()}
                         for record in data["leaks"]]
    data["failures"] = [record._asdict() for record in data.get("failures") or ()]
    for name, stored in _STORED_IF_ANY.items():
        entries = data.pop(name)
        if entries:
            data[name] = [stored(entry) for entry in entries]
    return data


def delivered_run_json(run: DeliveredRun) -> Dict[str, Any]:
    """A ``DeliveredRun`` as scenario JSON."""
    data = {"cases": [case._asdict() for case in run.cases],
            "failing": list(run.failing) if run.failing is not None else None,
            "reason": run.reason, "output": run.output,
            "reverse": bool(run.reverse), "seed": int(run.seed),
            "timed_out": bool(run.timed_out), "elapsed_s": run.elapsed_s,
            "leaks": None if run.leaks is None else [
                {name: list(value) if isinstance(value, tuple) else value
                 for name, value in record._asdict().items()} for record in run.leaks],
            "active_at_end": run.active_at_end}
    if run.unsound:
        data["unsound"] = [list(entry) for entry in run.unsound]
    if run.stopping_tests:
        data["stopping_tests"] = [asdict(test) for test in run.stopping_tests]
    return data


def delivered_run(data: Dict[str, Any]) -> DeliveredRun:
    """The ``DeliveredRun`` ``delivered_run_json`` stored."""
    return DeliveredRun(
        cases=tuple(DeliveredCase(**case) for case in data["cases"]),
        failing=tuple(data["failing"]) if data["failing"] is not None else None,
        reason=data["reason"], output=str(data["output"]),
        reverse=bool(data["reverse"]), seed=int(data["seed"]),
        timed_out=bool(data["timed_out"]), elapsed_s=data["elapsed_s"],
        leaks=_leaks(data),
        active_at_end=data.get("active_at_end"),
        unsound=_unsound(data),
        stopping_tests=test_ids(data.get("stopping_tests")))


def eviction_data_json(data: EvictionData) -> Dict[str, Any]:
    """An ``EvictionData`` as scenario JSON, sets as sorted lists."""
    return {
        "per_test_tokens": {
            test_id: sorted(tokens)
            for test_id, tokens in sorted(data.per_test_tokens.items())
        },
        "per_test_time_ms": dict(sorted(data.per_test_time_ms.items())),
        "suite_time_ms": data.suite_time_ms,
        "includes_kills": data.includes_kills,
        "target_total_lines": data.target_total_lines,
        "target_covered_lines": data.target_covered_lines,
        "target_total_branches": data.target_total_branches,
        "target_covered_branches": data.target_covered_branches,
        "suite_covered_lines": (
            sorted(data.suite_covered_lines)
            if data.suite_covered_lines is not None else None
        ),
        "suite_covered_branches": (
            sorted(data.suite_covered_branches)
            if data.suite_covered_branches is not None else None
        ),
        "first_test": data.first_test,
    }


def eviction_data(data: Dict[str, Any]) -> EvictionData:
    """The ``EvictionData`` ``eviction_data_json`` stored."""
    def _set(value: Any) -> Any:
        return frozenset(value) if value is not None else None

    return EvictionData(
        per_test_tokens={
            str(test_id): frozenset(tokens)
            for test_id, tokens in data["per_test_tokens"].items()
        },
        per_test_time_ms={
            str(test_id): float(ms)
            for test_id, ms in data["per_test_time_ms"].items()
        },
        # A report that timed no test has no suite time.
        suite_time_ms=(float(data["suite_time_ms"])
                       if data["suite_time_ms"] is not None else None),
        includes_kills=bool(data["includes_kills"]),
        target_total_lines=data.get("target_total_lines"),
        target_covered_lines=data.get("target_covered_lines"),
        target_total_branches=data.get("target_total_branches"),
        target_covered_branches=data.get("target_covered_branches"),
        suite_covered_lines=_set(data.get("suite_covered_lines")),
        suite_covered_branches=_set(data.get("suite_covered_branches")),
        first_test=data.get("first_test"),
    )


def read_coverage_matrix(path: Any) -> Optional[Dict[str, Any]]:
    """The coverage sidecar a real run wrote, as the scenario holds it: the
    backend's payload without its target path, which is each run's own, or
    None when it wrote none."""
    if not path or not Path(path).is_file():
        return None
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    data.pop("target_file", None)
    return data


def write_coverage_matrix(target: Any, matrix: Dict[str, Any], out_path: Any) -> None:
    """Write a coverage sidecar as the backend does: the payload the
    scenario holds with this run's target path."""
    payload = {**matrix, "target_file": str(target)}
    Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def validation_json(validation: SourceValidation) -> Dict[str, Any]:
    """A ``SourceValidation`` as scenario JSON, without the source's hash,
    which the key holds."""
    return {"ok": bool(validation.ok), "status": str(validation.status),
            "message": str(validation.message), "line": int(validation.line),
            "col": int(validation.col),
            "validator_identity": str(validation.validator_identity)}


def replayed_stats(answer: Dict[str, Any], progress_fn) -> Dict[str, Any]:
    """A recorded timing's statistics, its progress lines shown again."""
    for line in answer.get("progress") or ():
        if progress_fn is not None:
            progress_fn(str(line))
    return dict(answer["stats"])


def report_json(report: CapabilityReport) -> Dict[str, Any]:
    """A capability report as the scenario records an answer."""
    record = dataclasses.asdict(report)
    record["outcome"] = report.outcome.value
    for item in record["capabilities"] + record["operations"]:
        for name in ("availability", "readiness"):
            if item.get(name) is not None:
                item[name] = Availability(item[name]).value
    return json.loads(json.dumps(record))


def report_from_json(record: Dict[str, Any]) -> CapabilityReport:
    """The capability report ``report_json`` recorded."""
    version = record.get("version_record")
    return CapabilityReport(
        language=record["language"], interpreter=record["interpreter"],
        outcome=Outcome(record["outcome"]), reason=record["reason"],
        runtime=record["runtime"],
        capabilities=tuple(Capability(item["name"], Availability(item["availability"]),
                                      item["reason"])
                           for item in record["capabilities"]),
        tools=tuple(ToolVersion(**item) for item in record["tools"]),
        branch_coverage=record["branch_coverage"],
        mutation_engines=tuple(record["mutation_engines"]),
        operations=tuple(OperationCapability(**item) for item in record["operations"]),
        version_record=Version(**version) if version is not None else None)


def _resolution(result: Dict[str, Any]) -> CollisionResolution:
    return CollisionResolution(
        sources=tuple(result["sources"]), ejected=tuple(result["ejected"]),
        eject_reasons=tuple(result["eject_reasons"]),
        renames=tuple(dict(pairs) for pairs in result["renames"]),
        deduped=tuple(tuple(names) for names in result["deduped"]),
        dropped_defs=tuple(tuple(names) for names in result["dropped_defs"]),
        localized_autouse=tuple(tuple(names) for names in result["localized_autouse"]),
        choices=tuple((name, selected, tuple(owners))
                      for name, selected, owners in result["choices"]),
        conflicts=tuple((name, tuple(owners)) for name, owners in result["conflicts"]))


# The language backend's recorded calls.

def _files(arguments: Arguments, name: str = "test_paths") -> Dict[str, Any]:
    return suite_files(arguments[name], language=arguments.owner.name)


def _paths(arguments: Arguments, name: str = "test_paths") -> Dict[str, Any]:
    return {"test_paths": list(arguments[name])}


def _validated(answer: Dict[str, Any], arguments: Arguments) -> SourceValidation:
    validation = answer["validation"]
    return SourceValidation(
        bool(validation["ok"]), str(validation["status"]),
        str(validation["message"]), int(validation["line"]),
        int(validation["col"]), str(validation["validator_identity"]),
        source_sha256(arguments["source"]),
    )


VALIDATE_TEST_SOURCE = RecordedCall(
    "backend.validate_test_source", "backend",
    key=lambda a: {"source": source_sha256(a["source"])},
    context=lambda a: {"source": a["source"]},
    read=_validated,
    write=lambda result, a, collected: {"validation": validation_json(result)})


def _failing_key(a: Arguments) -> Dict[str, Any]:
    covered = {"measures_coverage": True} if a["measure_coverage_of"] else {}
    delivered = ({"delivered_seed": int(a["delivered_seed"])}
                 if a["delivered_seed"] is not None else {})
    return {**_files(a), "instrumented": a["instrument_target"] is not None,
            "timeout": int(a["timeout"]), **covered, **delivered}


def _failing_ids(answer: Dict[str, Any], a: Arguments) -> Optional[List[str]]:
    # The run's report (its outcome, and the test running at a timeout)
    # goes to the caller's dict, as the real listing fills it.
    if a["report"] is not None:
        a["report"].update(answer["report"])
    return None if answer["ids"] is None else [str(test_id) for test_id in answer["ids"]]


FAILING_TEST_IDS = RecordedCall(
    "backend.failing_test_ids", "backend", key=_failing_key, context=_paths,
    read=_failing_ids,
    write=lambda result, a, collected: {"ids": result, "report": dict(a["report"] or {})})


def _delivered_key(a: Arguments) -> Dict[str, Any]:
    key = {**_files(a), "reverse": bool(a["reverse"]), "seed": int(a["seed"]),
           "timeout": int(a["timeout"])}
    if a["sources"] is not None:
        key["sources"] = sorted([name, source_sha256(text)] for name, text in a["sources"].items())
    # The normal run's wishes: the output options when either is set, the
    # leak record when asked, and the target whenever given, since it
    # decides which file the run imports.
    wishes = ({"fail_fast": bool(a["fail_fast"]), "tb": a["tb"]}
              if (a["fail_fast"] or a["tb"]) else {})
    if a["record_leaks"]:
        wishes["record_leaks"] = True
    if a["target"]:
        wishes["target"] = target_entry(a["target"])
    if a["timeout_journal"]:
        wishes["timeout_journal"] = True
    key.update(wishes)
    return key


RUN_AS_DELIVERED = RecordedCall(
    "backend.run_as_delivered", "backend", key=_delivered_key, context=_paths,
    read=lambda answer, a: delivered_run(answer["run"]),
    write=lambda result, a, collected: {"run": delivered_run_json(result)},
    timed=lambda a: phase_state.TIME_TEST)


def _test_run_key(a: Arguments) -> Dict[str, Any]:
    key = {
        **_files(a),
        "coverage_of": target_entry(a["measure_coverage_of"]),
        "coverage_matrix": a["coverage_matrix_path"] is not None,
        "instrumented": bool(a["measure_coverage_of"] or a["instrument_target"]),
        "fail_fast": bool(a["fail_fast"]),
        "tb": a["tb"],
        "timeout": int(a["timeout"]),
    }
    if a["record_leaks"]:
        key["record_leaks"] = True
    # A run that asks the timeout journal for the test still running at a
    # timeout.
    if a["timeout_journal"]:
        key["timeout_journal"] = True
    return key


def _test_run(answer: Dict[str, Any], a: Arguments) -> TestRunResult:
    if a["coverage_matrix_path"] and answer.get("coverage_matrix"):
        write_coverage_matrix(a["measure_coverage_of"], answer["coverage_matrix"],
                              a["coverage_matrix_path"])
    return test_run_result(answer["result"])


RUN_TEST_FILES = RecordedCall(
    "backend.run_test_files", "backend", key=_test_run_key,
    context=lambda a: {**_paths(a), "measure_coverage_of": a["measure_coverage_of"]},
    read=_test_run,
    write=lambda result, a, collected: {
        "result": run_result_json(result),
        "coverage_matrix": read_coverage_matrix(a["coverage_matrix_path"])},
    timed=lambda a: (phase_state.TIME_COVERAGE if a["measure_coverage_of"]
                     else phase_state.TIME_TEST))


def _eviction_key(a: Arguments) -> Dict[str, Any]:
    matrix = a["kill_matrix_path"]
    return {
        **_files(a),
        "coverage_of": target_entry(a["measure_coverage_of"]),
        "warmup": bool(a["warmup"]),
        "kill_matrix": (kill_matrix_entry(Path(matrix))
                        if matrix and Path(matrix).is_file() else None),
    }


def _failure_entry(calls: List[tuple]) -> Optional[List[Any]]:
    """The evidence run's recorded failure: its kind and detail, then the
    named tests' id objects as their fields when the failure names any."""
    if not calls:
        return None
    kind, detail, *named = calls[0]
    return [kind, detail, *([[asdict(test) for test in named[0]["stopping_tests"]]]
                            if named else [])]


def _evicted(answer: Dict[str, Any], a: Arguments) -> Optional[EvictionData]:
    for line in answer.get("log") or ():
        if a["log"] is not None:
            a["log"](str(line))
    failure = answer.get("failure")
    if failure and a["failure_fn"] is not None:
        a["failure_fn"](str(failure[0]), str(failure[1]),
                        **({"stopping_tests": test_ids(failure[2])} if len(failure) > 2 else {}))
    data = answer.get("data")
    return eviction_data(data) if data is not None else None


COLLECT_EVICTION_DATA = RecordedCall(
    "backend.collect_eviction_data", "backend", key=_eviction_key,
    context=lambda a: {**_paths(a), "kill_matrix_path": a["kill_matrix_path"]},
    read=_evicted,
    write=lambda result, a, collected: {
        "data": eviction_data_json(result) if result is not None else None,
        "log": [args[0] for args in collected["log"]],
        "failure": _failure_entry(collected["failure_fn"])},
    callbacks=("log", "failure_fn"))


def _captured(answer: Dict[str, Any], a: Arguments) -> None:
    if answer.get("coverage_matrix"):
        write_coverage_matrix(a["target"], answer["coverage_matrix"], a["out_path"])


CAPTURE_COVERAGE_MATRIX = RecordedCall(
    "backend.capture_coverage_matrix", "backend",
    key=lambda a: {**suite_files([str(p) for p in a["test_paths"]],
                                 language=a.owner.name),
                   "coverage_of": target_entry(a["target"])},
    context=lambda a: {"test_paths": [str(p) for p in a["test_paths"]]},
    read=_captured,
    write=lambda result, a, collected: {"coverage_matrix": read_coverage_matrix(a["out_path"])})


BENCHMARK_TESTS = RecordedCall(
    "backend.benchmark_tests", "backend",
    key=lambda a: {
        **_files(a, "test_file_paths"),
        "warmup_runs": int(a["warmup_runs"]),
        "timing_run_count": int(a["benchmark_runs"]),
        "suite_label": a["suite_label"],
        "instrumented": a["instrument_target"] is not None,
    },
    context=lambda a: _paths(a, "test_file_paths"),
    read=lambda answer, a: replayed_stats(answer, a["progress_fn"]),
    write=lambda result, a, collected: {"stats": dict(result),
                                        "progress": [args[0] for args in collected["progress_fn"]]},
    callbacks=("progress_fn",))


def _suites(a: Arguments) -> None:
    a.values["suites"] = [(str(label), list(paths)) for label, paths in a["suites"]]


BENCHMARK_INTERLEAVED = RecordedCall(
    "backend.benchmark_interleaved", "backend",
    key=lambda a: {
        "suites": [{"label": label, **suite_files(paths, language=a.owner.name)}
                   for label, paths in a["suites"]],
        "warmup_runs": int(a["warmup_runs"]),
        "timing_run_count": int(a["benchmark_runs"]),
        "instrumented": a["instrument_target"] is not None,
    },
    context=lambda a: {"suites": {label: list(paths) for label, paths in a["suites"]}},
    read=lambda answer, a: replayed_stats(answer, a["progress_fn"]),
    write=lambda result, a, collected: {"stats": dict(result),
                                        "progress": [args[0] for args in collected["progress_fn"]]},
    callbacks=("progress_fn",), prepare=_suites)


DESCRIBE_COLLECTION_FAILURE = RecordedCall(
    "backend.describe_collection_failure", "backend", key=_files, context=_paths,
    read=lambda answer, a: str(answer["text"]),
    write=lambda result, a, collected: {"text": result})


def _pruned(answer: Dict[str, Any], a: Arguments) -> Tuple[str, List[str]]:
    return str(answer["source"]), [str(module) for module in answer["dropped"]]


def _prune_written(result: Tuple[str, List[str]], a: Arguments, collected) -> Dict[str, Any]:
    pruned, dropped = result
    return {"source": pruned, "dropped": list(dropped)}


PRUNE_UNRESOLVABLE_IMPORTS = RecordedCall(
    "backend.prune_unresolvable_imports", "backend",
    key=lambda a: {"source": source_sha256(a["source"])},
    context=lambda a: {"source": a["source"]}, read=_pruned, write=_prune_written)

PRUNE_COLLECTION_BREAKERS = RecordedCall(
    "backend.prune_collection_breakers", "backend",
    key=lambda a: {"source": source_sha256(a["source"])},
    context=lambda a: {"source": a["source"]}, read=_pruned, write=_prune_written)

# Python checks a merge in the project's interpreter, and the demonstration
# starts no process.
MERGE_TEST_FILES = RecordedCall(
    "backend.merge_test_files", "backend",
    key=lambda a: {"first": source_sha256(a["coverage_src"]),
                   "second": source_sha256(a["mutation_src"])},
    read=lambda answer, a: answer["source"],
    write=lambda result, a, collected: {"source": result})

STRIP_MODULE_LEVEL_SKIPS = RecordedCall(
    "backend.strip_module_level_skips", "backend",
    key=lambda a: {"source": source_sha256(a["source"])},
    read=lambda answer, a: (answer["source"], int(answer["removed"])),
    write=lambda result, a, collected: dict(zip(("source", "removed"), result)))


def _collisions_key(a: Arguments) -> Dict[str, Any]:
    evidence = a["evidence"]
    return {"sources": [source_sha256(source) for source in a["sources"]],
            "localize_autouse": bool(a["localize_autouse"]),
            "evidence": [{name: sorted(tokens) for name, tokens in item.items()}
                         for item in evidence] if evidence is not None else None}


RESOLVE_TEST_COLLISIONS = RecordedCall(
    "backend.resolve_test_collisions", "backend", key=_collisions_key,
    read=lambda answer, a: _resolution(answer["resolution"]),
    write=lambda result, a, collected: {"resolution": asdict(result)})

# The operations a language lacks decide the run's fallbacks, so the
# playback states them as the recorded run found them.
CAPABILITIES = RecordedCall(
    "backend.capabilities", "backend",
    key=lambda a: {"language": a.owner.name},
    read=lambda answer, a: report_from_json(answer["report"]),
    write=lambda result, a, collected: {"report": report_json(result)})

#: The recorded calls every language's scenario backend answers.
BACKEND_CALLS = (VALIDATE_TEST_SOURCE, FAILING_TEST_IDS, RUN_AS_DELIVERED, RUN_TEST_FILES,
                 COLLECT_EVICTION_DATA, CAPTURE_COVERAGE_MATRIX, BENCHMARK_TESTS,
                 BENCHMARK_INTERLEAVED, DESCRIBE_COLLECTION_FAILURE,
                 PRUNE_UNRESOLVABLE_IMPORTS, PRUNE_COLLECTION_BREAKERS, CAPABILITIES)


def backend_method(entry: RecordedCall):
    """The playback method of a language backend's recorded call, bound by
    the generic backend's method of that name."""
    return playback(entry, getattr(GenericBackend, entry.method))


# The mutation engine's recorded calls: the runner's, a session's and a
# measurement's.

def _run_options(runner: Any, operators: Optional[List[str]]) -> Dict[str, Any]:
    """The key parts a mutation run and a per-suite run share: the
    operators' hash and whether the run stops at its first kill."""
    return {
        "operators": (source_sha256(canonical(list(operators)))
                      if operators is not None else None),
        "fail_fast": runner.attribution != "all",
    }


def run_key(key: Dict[str, Any]) -> str:
    """The name a mutation run's key gives its measurement: its survivors'
    text and kill matrix are asked for by it."""
    return source_sha256(canonical(key))


def _mutation_run_key(a: Arguments) -> Dict[str, Any]:
    runner = a.owner.runner
    key = {
        "tests": tree_files(Path(a["test_dir"]), language=runner.demo_language),
        "target": target_entry(a["target"]),
        "per_mutant_timeout": int(a["per_mutant_timeout"]),
        "wall_clock_timeout": int(a["wall_clock_timeout"]),
        **_run_options(runner, a["operators"]),
    }
    if a["attribution"] is not None:
        key["attribution"] = a["attribution"]
    return key


def _measured(answer: Dict[str, Any], a: Arguments) -> Any:
    from testuary.demo.backend import ScenarioMeasurement

    return ScenarioMeasurement(a.owner.runner, run_key(a.key), answer["results"])


MUTATION_RUN = RecordedCall(
    "mutation.run", "session", key=_mutation_run_key,
    context=lambda a: {"test_dir": str(a["test_dir"])},
    read=_measured,
    write=lambda result, a, collected: {"results": {
        "killed": result.killed, "survived": result.survived,
        "timeout": result.timeout, "non_viable": result.non_viable,
        "untested": result.untested, "total": result.total,
        "score": result.score, "truncated": result.truncated,
        "reason": result.reason, "suite_failed": result.suite_failed,
        "stopping_tests": [asdict(test) for test in result.stopping_tests],
        "lower_bound": result.lower_bound, "timeout_rule": result.timeout_rule()}})


def _per_suite_key(a: Arguments) -> Dict[str, Any]:
    runner = a.owner.runner
    entries = [{"name": str(name),
                "tests": tree_files(Path(test_dir), language=runner.demo_language),
                "test_files": sorted(Path(path).name for path in test_files),
                "per_mutant_timeout": int(per_mutant_timeout)}
               for name, test_dir, test_files, per_mutant_timeout in a["suites"]]
    if not entries:
        raise ValueError("per_suite needs at least one suite")
    return {"suites": entries, "target": target_entry(a["target"]),
            "wall_clock_timeout": int(a["wall_clock_timeout"]),
            **_run_options(runner, a["operators"])}


def _per_suite_run(answer: Dict[str, Any], a: Arguments) -> Any:
    from testuary.demo.backend import ScenarioPerSuiteRun

    return ScenarioPerSuiteRun(answer)


# Each suite's verdicts, keyed like a standalone run of each suite: its
# measured files and limits.
MUTATION_PER_SUITE = RecordedCall(
    "mutation.per_suite", "session", key=_per_suite_key,
    context=lambda a: {"test_dirs": {str(name): str(test_dir)
                                     for name, test_dir, _files, _timeout in a["suites"]}},
    read=_per_suite_run,
    write=lambda result, a, collected: {
        "implementation": str(result.implementation),
        "outcome": result.outcome.value, "reason": str(result.reason or ""),
        "runs": {str(name): {"outcome": run.outcome.value, "reason": str(run.reason or ""),
                             "measured": bool(run.measured)}
                 for name, run in result.runs.items()},
        "kill_matrices": {str(name): result.kill_matrix(name) for name in result.runs},
        "union": result.kill_matrix()})

FULL_OPERATORS = RecordedCall(
    "mutation.full_operators", "runner", key=lambda a: {},
    read=lambda answer, a: (list(answer["operators"]) if answer["operators"] is not None
                            else None),
    write=lambda result, a, collected: {"operators": list(result) if result is not None
                                        else None})

# The survivor sample as the recording took it for the run: its size and each
# change's length decide the text.
RENDER_SURVIVING_DIFFS = RecordedCall(
    "mutation.render_surviving_diffs", "measurement", name="survivors_text",
    key=lambda a: {"run": a.owner.run_key,
                   "max_mutants": None if a["max_mutants"] is None else int(a["max_mutants"]),
                   "max_chars": None if a["max_chars"] is None else int(a["max_chars"])},
    read=lambda answer, a: str(answer["text"]),
    write=lambda result, a, collected: {"text": result})


def _kill_matrix(answer: Dict[str, Any], a: Arguments) -> Optional[Dict[str, Any]]:
    payload = answer.get("kill_matrix")
    if payload is None:
        return None
    return {**payload, **timing_fields()}


def _matrix_written(result: Any, a: Arguments, collected) -> Dict[str, Any]:
    # The clock fields belong to the run that reads the matrix.
    if isinstance(result, dict):
        result = {name: value for name, value in result.items()
                  if name not in ("t_mono_s", "clock_domain_id")}
    return {"kill_matrix": result}


CAPTURE_KILL_MATRIX = RecordedCall(
    "mutation.capture_kill_matrix", "measurement", name="kill_matrix",
    key=lambda a: {"run": a.owner.run_key}, read=_kill_matrix, write=_matrix_written)


# Model requests and the exact cover.

MODEL_SUITE = RecordedCall(
    "model.seed", "model", name="generate_suite",
    key=lambda a: {"seed": int(a["seed"]), "temperature": a["temperature"],
                   "messages": message_hashes(a["messages"])},
    context=lambda a: {"messages": a["messages"]},
    read=lambda answer, a: (str(answer["text"]), dict(answer["usage"])),
    write=lambda result, a, collected: {"text": result[0], "usage": dict(result[1])})


def _turn(a: Arguments) -> None:
    a.values["turn"] = a.owner.next_turn()


# One request of an agent, counted across the agent modes. A recording's
# chat model sends it with the tools of the agent mode that sent it.
MODEL_AGENT = RecordedCall(
    "model.agent", "model", name="_generate",
    key=lambda a: {"phase": a.owner.phase, "turn": a["turn"],
                   "messages": agent_message_hashes(a["messages"])},
    context=lambda a: {"messages": a["messages"]},
    read=lambda answer, a: ChatResult(generations=[ChatGeneration(
        message=messages_from_dict([answer["message"]])[0])]),
    write=lambda result, a, collected: {"message": message_to_dict(result)},
    omit=("run_manager", "turn"), prepare=_turn)


KIND = "cover.exact"
NOTE = ("answered from the scenario: the selection the package's exact "
        "reduction returned with the bundled CBC when the scenario was built")


def problem_key(
    per_test_tokens: Mapping[str, FrozenSet[str]],
    per_test_cost_ms: Mapping[str, float],
    *,
    forced_ids: FrozenSet[str],
    time_limit_s: float,
    minimum_test_runtime_ms: float,
    random_seed: int,
    tie_break: str,
) -> Dict[str, Any]:
    """What identifies a cover problem: its tests with their tokens and
    costs, the forced tests and the solver settings."""
    tests = sorted(per_test_tokens)
    tokens = {t: sorted(per_test_tokens[t]) for t in tests}
    costs = {t: float(per_test_cost_ms[t]) for t in tests}
    return {
        "tests": len(tests),
        "tokens_sha256": sha256_bytes(canonical(tokens).encode("utf-8")),
        "costs_sha256": sha256_bytes(canonical(costs).encode("utf-8")),
        "forced": sorted(t for t in forced_ids if t in per_test_tokens),
        "time_limit_s": float(time_limit_s),
        "min_test_runtime_ms": float(minimum_test_runtime_ms),
        "random_seed": int(random_seed),
        "tie_break": str(tie_break),
    }


def solver_description(metadata: Mapping[str, Any]) -> Dict[str, Any]:
    """A solve's metadata without what the problem determines (the costs)
    and what describes the solving machine (the times and the time left for
    each tier)."""
    meta = {k: v for k, v in metadata.items()
            if k not in ("raw_cost_ms", "effective_cost_us", "elapsed_s")}
    meta["tiers"] = [
        {k: v for k, v in tier.items() if k not in ("elapsed_s", "time_limit_s")}
        for tier in metadata.get("tiers") or []
    ]
    return meta


def answer_of(reduction: LosslessReduction) -> Dict[str, Any]:
    """The scenario answer for a solved problem: the selection, the optima
    and the solver's description."""
    return {
        "keep": list(reduction.keep),
        "runtime_objective_us": int(reduction.runtime_objective_us),
        "test_count_objective": int(reduction.test_count_objective),
        "solver_metadata": solver_description(reduction.solver_metadata or {}),
    }


def check_answer(
    answer: Mapping[str, Any],
    per_test_tokens: Mapping[str, FrozenSet[str]],
    per_test_cost_ms: Mapping[str, float],
    *,
    forced_ids: FrozenSet[str],
    minimum_test_runtime_ms: float,
) -> Optional[str]:
    """Why ``answer`` does not solve the problem, or None when it does."""
    tests = sorted(per_test_tokens)
    keep: List[str] = list(answer["keep"])
    if keep != sorted(set(keep)) or not set(keep) <= set(tests):
        return "names tests the problem does not hold"
    if not {t for t in forced_ids if t in per_test_tokens} <= set(keep):
        return "leaves out a forced test"
    universe = set().union(*per_test_tokens.values()) if tests else set()
    covered = set().union(*(per_test_tokens[t] for t in keep)) if keep else set()
    if covered != universe:
        return "does not cover every token"
    _, effective = _effective_costs_us(
        tests, per_test_cost_ms, minimum_ms=minimum_test_runtime_ms)
    if sum(effective[t] for t in keep) != int(answer["runtime_objective_us"]):
        return "does not meet the recorded runtime optimum"
    if len(keep) != int(answer["test_count_objective"]):
        return "does not meet the recorded test-count optimum"
    return None


def reduction_of(
    answer: Mapping[str, Any],
    per_test_tokens: Mapping[str, FrozenSet[str]],
    per_test_cost_ms: Mapping[str, float],
    *,
    minimum_test_runtime_ms: float,
) -> LosslessReduction:
    """The reduction ``exact_cover`` would have returned for this answer."""
    tests = sorted(per_test_tokens)
    raw, effective = _effective_costs_us(
        tests, per_test_cost_ms, minimum_ms=minimum_test_runtime_ms)
    keep = tuple(answer["keep"])
    meta = dict(answer.get("solver_metadata") or {})
    # As the solver's: the measured times apart from the imputed costs.
    imputed = meta.get("imputed_cost_ms") or {}
    meta.update({
        "raw_cost_ms": {t: raw[t] for t in tests if t not in imputed},
        "effective_cost_us": {t: effective[t] for t in tests},
        "selected_ids": list(keep),
    })
    meta.setdefault("demonstration", NOTE)
    return LosslessReduction(
        keep=keep,
        remove=tuple(t for t in tests if t not in set(keep)),
        solver="exact",
        effective_cost_us=tuple((t, effective[t]) for t in tests),
        runtime_objective_us=int(answer["runtime_objective_us"]),
        test_count_objective=int(answer["test_count_objective"]),
        solver_metadata=meta,
    )


def _weighed(a: Arguments) -> Dict[str, float]:
    """The costs the solver weighs: the measured times and the imputed
    costs."""
    return {**a["per_test_cost_ms"], **(a["imputed_cost_ms"] or {})}


def _cover_key(a: Arguments) -> Dict[str, Any]:
    return problem_key(a["per_test_tokens"], _weighed(a), forced_ids=a["forced_ids"],
                       time_limit_s=a["time_limit_s"],
                       minimum_test_runtime_ms=a["minimum_test_runtime_ms"],
                       random_seed=a["random_seed"], tie_break=a["tie_break"])


def _covered(answer: Dict[str, Any], a: Arguments) -> LosslessReduction:
    problem = check_answer(answer, a["per_test_tokens"], _weighed(a),
                           forced_ids=a["forced_ids"],
                           minimum_test_runtime_ms=a["minimum_test_runtime_ms"])
    if problem is not None:
        raise ScenarioStop(Call(KIND, a.key), f"the scenario's selection {problem}. It must be "
                                              f"regenerated for this package version")
    return reduction_of(answer, a["per_test_tokens"], _weighed(a),
                        minimum_test_runtime_ms=a["minimum_test_runtime_ms"])


# A recording solves the problem with the bundled CBC.
EXACT_COVER = RecordedCall(KIND, "cover", name="exact_cover", key=_cover_key, read=_covered,
                           write=lambda result, a, collected: answer_of(result))


#: Every recorded call, by kind.
CALLS = {entry.kind: entry for entry in (
    *BACKEND_CALLS, MERGE_TEST_FILES, STRIP_MODULE_LEVEL_SKIPS, RESOLVE_TEST_COLLISIONS,
    MUTATION_RUN, MUTATION_PER_SUITE, FULL_OPERATORS, RENDER_SURVIVING_DIFFS,
    CAPTURE_KILL_MATRIX, MODEL_SUITE, MODEL_AGENT, EXACT_COVER)}
