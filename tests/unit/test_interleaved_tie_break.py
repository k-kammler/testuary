"""The phase-end timing tie-break times the tied candidates interleaved.

Each candidate keeps one prepared copy for all its executions. Warm-up rounds
come first, then timed rounds that execute every candidate once with a
rotating first candidate, so a drift in machine load falls on all candidates
alike.
"""
from __future__ import annotations

import contextlib
from pathlib import Path
import subprocess
import sys

from langadapter import Outcome
from langadapter.outcomes import ExecutionIdentity, ObservationIdentity, ToolchainIdentity
from langadapter.java.adapter import JavaAdapter as ProjectJavaBackend
from langadapter.python.adapter import PythonAdapter as ProjectPythonBackend
from langadapter.runs import Benchmark, WALL_TIME_SCOPE
import pytest

from test_prepared_benchmarks import _project
from testuary import project as _benchmark_lock
from testuary import config
from testuary import project
from testuary.languages import generic, preparation
from testuary.languages.generic import GenericBackend
from testuary import consolidation as consolidate

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


class _Machine:
    """Controlled native timing: each execution takes the suite's own time
    plus a load that grows by ``drift`` ms with every execution."""

    def __init__(self, own, drift=0.0, fail=None):
        self.own = own
        self.drift = drift
        self.fail = dict(fail or {})
        self.executions = []
        self.requests = []
        self.locked = []

    def label(self, request):
        text = Path(request.suite.test_files[0]).read_text(encoding="utf-8")
        return next(label for label in self.own if f"# {label}\n" in text)

    def benchmark(self, language, request):
        request.validate()
        self.requests.append(request)
        label = self.label(request)
        times = []
        for repetition in range(request.warmup_runs + request.runs):
            self.locked.append(_HELD[0])
            self.executions.append(label)
            outcome = self.fail.get((label, len(self.executions)))
            if outcome is not None:
                return Benchmark(outcome, "1 failed", tuple(times), request.warmup_runs,
                                 None, WALL_TIME_SCOPE, _identity(language, request),
                                 returncode=1)
            sample = self.own[label] + self.drift * len(self.executions)
            if repetition >= request.warmup_runs:
                times.append(sample)
        return Benchmark(Outcome.COMPLETED, "", tuple(times), request.warmup_runs, None,
                         WALL_TIME_SCOPE, _identity(language, request),
                         allowed_cpus=(1, 3), returncode=0)


_HELD = [False]


def _identity(language, request):
    return ObservationIdentity(None, request.suite.identity(),
                               ToolchainIdentity(language, "controlled"), None,
                               ExecutionIdentity("local"))


@pytest.fixture
def controlled(tmp_path, monkeypatch, request):
    language = request.param
    root, target, _test = _project(tmp_path, language, False)

    def native_forbidden(*args, **kwargs):
        pytest.fail("the controlled timing must not execute native tools")

    monkeypatch.setattr(subprocess, "Popen", native_forbidden)
    monkeypatch.setattr(subprocess, "run", native_forbidden)
    monkeypatch.setattr(preparation, "bytecode_store", lambda backend: {})
    monkeypatch.setattr(config, "project_exec_mode", lambda: "local")
    monkeypatch.setattr(project, "project_work_dir", lambda: None)
    # Each language's runtime is this interpreter, selected in the settings.
    monkeypatch.setattr(config, "load_config", lambda section: {
        "pytest_config": "none", "frozen_module_strategy": "skip",
        "python_executable": sys.executable, "java_home": sys.executable})
    if language == "java":
        monkeypatch.setattr(ProjectJavaBackend, "file_rule",
                            lambda self, source: "CandidateTest")

    @contextlib.contextmanager
    def held():
        assert not _HELD[0], "the benchmark lock is not reentrant here"
        _HELD[0] = True
        try:
            yield
        finally:
            _HELD[0] = False

    monkeypatch.setattr(_benchmark_lock, "acquire", held)
    owners = []
    prepare = preparation.prepare_project

    def counted(context, **options):
        owners.append(context)
        return prepare(context, **options)

    monkeypatch.setattr(preparation, "prepare_project", counted)
    native = ProjectPythonBackend if language == "python" else ProjectJavaBackend

    def install(machine):
        monkeypatch.setattr(native, "benchmark",
                            lambda self, req: machine.benchmark(language, req))
        return machine

    adapter = (GenericBackend("python", (".py",)) if language == "python"
               else GenericBackend("java", (".java",)))
    return adapter, root, target, install, owners


def _candidates(language, labels, timing=1.0):
    return [consolidate.ScoredSuite(
        label=label,
        source=(f"// x\n# {label}\npublic class CandidateTest {{}}\n"
                if language == "java" else f"# {label}\ndef test_ok(): pass\n"),
        line=50.0, branch=50.0, mutation=50.0, green=True, test_count=1,
        runtime_ms=timing,
    ) for label in labels]


def _timer(adapter, root, target, warmup_runs=1, benchmark_runs=5):
    return consolidate.TiedCandidateTimer(
        adapter, source_dir=str(root), target_file_path=str(target), timeout=19,
        warmup_runs=warmup_runs, benchmark_runs=benchmark_runs, stage_as_delivered=False, log=lambda message: None,
    )


@pytest.mark.parametrize("controlled", ["python", "java"], indirect=True)
def test_a_load_drift_does_not_decide_a_tie(controlled):
    """The faster suite wins although the load grows during the tie-break.
    Timed in blocks, the first suite's block would see less load and win."""
    adapter, root, target, install, _owners = controlled
    machine = install(_Machine({"baseline": 100.0, "agent": 90.0}, drift=30.0))
    candidates = _candidates(adapter.name, ["baseline", "agent"])

    winner = consolidate.pick_phase_result(candidates, runtime_of=_timer(adapter, root, target), required_axes=consolidate._PHASE_END_AXES)

    assert winner.label == "agent"
    assert machine.executions == ["baseline", "agent", "agent", "baseline",
                                  "baseline", "agent", "agent", "baseline",
                                  "baseline", "agent", "agent", "baseline"]


@pytest.mark.parametrize("controlled", ["python", "java"], indirect=True)
@pytest.mark.parametrize("warmup_runs", [0, 1, 2])
def test_each_candidate_keeps_one_copy_and_each_execution_holds_the_lock(
        controlled, warmup_runs):
    adapter, root, target, install, owners = controlled
    machine = install(_Machine({"baseline": 200.0, "agent": 100.0, "suite_000": 150.0}))
    candidates = _candidates(adapter.name, ["baseline", "agent", "suite_000"])

    winner = consolidate.pick_phase_result(
        candidates, runtime_of=_timer(adapter, root, target, warmup_runs, 3), required_axes=consolidate._PHASE_END_AXES)

    assert winner.label == "agent"
    assert len(owners) == 3
    assert len(machine.requests) == 3 * (warmup_runs + 3)
    assert all((request.runs, request.warmup_runs, request.timeout_s, request.serialize)
               == (1, 0, 19, False) for request in machine.requests)
    assert all(machine.locked)
    rounds = winner.suite_timing["rounds"]
    assert [item["round"] for item in rounds] == list(range(1, warmup_runs + 4))
    assert [item["warmup"] for item in rounds] == [True] * warmup_runs + [False] * 3
    labels = ["baseline", "agent", "suite_000"]
    assert [item["order"] for item in rounds] == [
        labels[number % 3:] + labels[:number % 3] for number in range(warmup_runs + 3)]
    assert winner.suite_timing["samples_ms"] == [100.0, 100.0, 100.0]
    assert winner.suite_timing["median_ms"] == 100.0
    assert (winner.suite_timing["warmup_runs"],
            winner.suite_timing["timing_run_count"]) == (warmup_runs, 3)

    phase = consolidate.PhaseConsolidation()
    consolidate.record_selection_audit(phase, candidates, winner, required_axes=consolidate._PHASE_END_AXES)
    assert phase.selection_audit["runtime_tie_break_rounds"] == [{
        "batch": 1, "candidates": ["agent", "baseline", "suite_000"], "rounds": rounds,
    }]


@pytest.mark.parametrize("controlled", ["python"], indirect=True)
def test_a_red_candidate_leaves_the_rounds_and_gives_no_time(controlled):
    adapter, root, target, install, _owners = controlled
    # The third execution is the agent's, the first of round two.
    machine = install(_Machine({"baseline": 200.0, "agent": 100.0},
                               fail={("agent", 3): Outcome.TEST_FAILURES}))
    candidates = _candidates(adapter.name, ["baseline", "agent"])

    winner = consolidate.pick_phase_result(candidates, runtime_of=_timer(adapter, root, target), required_axes=consolidate._PHASE_END_AXES)

    # Its failed timing run gives the agent no time, so the input order
    # decides; it is not rejected.
    assert winner.label == "baseline"
    assert candidates[1].green is True
    assert candidates[1].suite_timing["status"] == "failed"
    assert machine.executions == ["baseline", "agent", "agent"] + ["baseline"] * 5
    assert winner.suite_timing["samples_ms"] == [200.0] * 5


@pytest.mark.parametrize("controlled", ["python"], indirect=True)
@pytest.mark.parametrize("outcome", [Outcome.TIMED_OUT, Outcome.INFRASTRUCTURE_FAILURE])
def test_an_infrastructure_failure_stops_the_rounds_and_aborts(controlled, outcome):
    adapter, root, target, install, _owners = controlled
    machine = install(_Machine({"baseline": 200.0, "agent": 100.0},
                               fail={("agent", 2): outcome}))

    with pytest.raises(RuntimeError, match="'agent' failed"):
        consolidate.pick_phase_result(_candidates(adapter.name, ["baseline", "agent"]),
                                      runtime_of=_timer(adapter, root, target), required_axes=consolidate._PHASE_END_AXES)
    assert machine.executions == ["baseline", "agent"]


@pytest.mark.parametrize("controlled", ["python"], indirect=True)
def test_the_interleaved_result_keeps_benchmark_tests_stats(controlled):
    adapter, root, target, install, _owners = controlled
    install(_Machine({"baseline": 200.0, "agent": 100.0}))
    staged = []
    for candidate in _candidates(adapter.name, ["baseline", "agent"]):
        path = Path(root).parent / f"{candidate.label}_test_subject.py"
        path.write_text(candidate.source, encoding="utf-8")
        staged.append((candidate.label, [str(path)]))
    progress = []

    stats = adapter.benchmark_interleaved(
        source_dir=str(root), suites=staged, warmup_runs=1, benchmark_runs=2,
        timeout=19, progress_fn=progress.append, instrument_target=str(target))

    assert stats["ok"] is True
    assert stats["suites"]["agent"] == {
        "ok": True, "times_ms": [100.0, 100.0], "warmup_times_ms": [100.0],
        "requested_cpus": None, "allowed_cpus": [1, 3], "warmup_runs": 1,
        "timing_run_count": 2, "median_ms": 100.0, "mean_ms": 100.0, "min_ms": 100.0,
        "max_ms": 100.0, "early_stopped": False, "early_stop_reason": None,
    }
    assert progress == ["[benchmark:interleaved] warmup 1/1: baseline, agent",
                        "[benchmark:interleaved] round 1/2: agent, baseline",
                        "[benchmark:interleaved] round 2/2: baseline, agent"]
    with pytest.raises(ValueError, match="labels repeat"):
        adapter.benchmark_interleaved(source_dir=str(root), suites=staged * 2,
                                      warmup_runs=0, benchmark_runs=1, timeout=19)


def test_a_language_without_a_timing_benchmark_keeps_the_input_order(monkeypatch):
    backend = GenericBackend("python", (".py",))
    monkeypatch.setattr(generic.GenericBackend, "_supports", lambda self, op, root=None: False)
    assert backend.benchmark_interleaved(
        source_dir=".", suites=[("a", ["x.py"])], warmup_runs=1, benchmark_runs=1,
        timeout=1) == {"ok": False, "unavailable": True, "rounds": [], "suites": {},
                       "error": "python has no timing benchmark",
                       "failure_kind": "infrastructure"}


class _Rounds:
    """A backend whose interleaved timing answers from a table."""

    def __init__(self, medians, failing=()):
        self.medians = medians
        self.failing = set(failing)
        self.batches = []

    def scratch_test_path(self, directory, source, target, **kw):
        return Path(directory) / "test_probe.py"

    def benchmark_interleaved(self, *, suites, warmup_runs, benchmark_runs, **kw):
        labels = [label for label, _paths in suites]
        self.batches.append(labels)
        results = {}
        for label in labels:
            if label in self.failing:
                results[label] = {"ok": False, "failure_kind": "test_failure",
                                  "returncode": 1, "error": "1 failed"}
                continue
            value = self.medians[label]
            results[label] = {"ok": True, "times_ms": [value] * benchmark_runs,
                              "warmup_times_ms": [value] * warmup_runs, "median_ms": value}
        return {"ok": not (self.failing & set(labels)), "suites": results, "rounds": [
            {"round": 1, "warmup": False, "order": labels}]}


def _scored(label, mutation):
    return consolidate.ScoredSuite(label=label, source=f"# {label}\n", line=50.0,
                                   branch=50.0, mutation=mutation, green=True,
                                   test_count=1, runtime_ms=1.0)


def test_a_tie_whose_timing_runs_fail_is_decided_by_the_input_order():
    """Failed timing runs give no time and reject nothing: the tie at the
    top is decided by the input order, and no lower-scoring candidate is
    timed."""
    backend = _Rounds({"a": 1.0, "b": 1.0, "c": 30.0, "d": 20.0}, failing={"a", "b"})
    candidates = [_scored("a", 60.0), _scored("b", 60.0), _scored("c", 50.0),
                  _scored("d", 50.0)]
    timer = consolidate.TiedCandidateTimer(
        backend, source_dir=".", target_file_path="mod.py", timeout=1,
        warmup_runs=1, benchmark_runs=3, stage_as_delivered=False, log=lambda message: None)

    winner = consolidate.pick_phase_result(candidates, runtime_of=timer, required_axes=consolidate._PHASE_END_AXES)

    assert winner.label == "a"
    assert [candidate.green for candidate in candidates] == [True] * 4
    assert backend.batches == [["a", "b"]]


def test_the_survivors_of_a_batch_reuse_its_rounds():
    """A selection repeated after its winner was excluded (the starters'
    check under the delivered filename) reuses the batch the survivors were
    timed in."""
    backend = _Rounds({"a": 1.0, "b": 9.0, "c": 5.0})
    candidates = [_scored("a", 60.0), _scored("b", 60.0), _scored("c", 60.0)]
    timer = consolidate.TiedCandidateTimer(
        backend, source_dir=".", target_file_path="mod.py", timeout=1,
        warmup_runs=0, benchmark_runs=1, stage_as_delivered=False, log=lambda message: None)

    assert consolidate.pick_phase_result(candidates, runtime_of=timer, required_axes=consolidate._PHASE_END_AXES).label == "a"
    candidates[0].green = False
    winner = consolidate.pick_phase_result(candidates, runtime_of=timer, required_axes=consolidate._PHASE_END_AXES)

    assert winner.label == "c"
    assert backend.batches == [["a", "b", "c"]]


def test_a_sample_count_that_does_not_match_the_setting_aborts():
    class _Short(_Rounds):
        def benchmark_interleaved(self, **kw):
            stats = super().benchmark_interleaved(**kw)
            stats["suites"]["a"]["times_ms"] = [1.0]
            return stats

    timer = consolidate.TiedCandidateTimer(
        _Short({"a": 1.0, "b": 2.0}), source_dir=".", target_file_path="mod.py",
        timeout=1, warmup_runs=1, benchmark_runs=3, stage_as_delivered=False, log=lambda message: None)
    with pytest.raises(RuntimeError, match="every finite timing sample"):
        consolidate.pick_phase_result([_scored("a", 60.0), _scored("b", 60.0)],
                                      runtime_of=timer, required_axes=consolidate._PHASE_END_AXES)
