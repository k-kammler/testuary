"""The recorded calls' declarations: each shipped answer reads back into
what the run receives, and writing that again gives the stored answer."""
from __future__ import annotations

import inspect
import json
from importlib import resources
from pathlib import Path
from types import SimpleNamespace

import pytest

from testuary.demo import recorded
from testuary.demo.scenario import Scenario, canonical

DATA = resources.files("testuary.demo") / "data"
RECORDINGS = {"python": DATA, "java": DATA / "java", "toy": DATA / "toy",
              "tutorial": DATA / "tutorial"}


def _entries():
    for name, data_dir in RECORDINGS.items():
        for index, entry in enumerate(Scenario.load(Path(str(data_dir / "scenario.json.gz"))).entries):
            yield pytest.param(entry, id=f"{name}-{index}-{entry.kind}")


class _Values(dict):
    """A call's arguments for reading and writing one answer: the files it
    writes and reads in one directory, its callbacks collected, an empty
    report dict as each caller of the failing-test listing passes, any other
    argument absent."""

    def __init__(self, scratch_dir: Path, collected):
        super().__init__(coverage_matrix_path=str(scratch_dir / "coverage.json"),
                         out_path=str(scratch_dir / "coverage.json"), source="",
                         measure_coverage_of="target", target="target", report={})
        for name in ("log", "failure_fn", "progress_fn"):
            self[name] = lambda *args, _name=name: collected.setdefault(_name, []).append(args)

    def __missing__(self, name):
        return None


def _arguments(entry, scratch_dir: Path, collected) -> recorded.Arguments:
    owner = SimpleNamespace(runner=None, name="python", run_key="", phase="")
    arguments = recorded.Arguments(
        inspect.Signature([inspect.Parameter("self", inspect.Parameter.POSITIONAL_ONLY)]),
        owner, (), {})
    arguments.values = _Values(scratch_dir, collected)
    arguments.key = entry.key
    return arguments


@pytest.mark.parametrize("entry", list(_entries()))
def test_each_recorded_answer_reads_back_and_is_written_again_as_stored(entry, tmp_path):
    declared = recorded.CALLS[entry.kind]
    if declared is recorded.EXACT_COVER:
        # Read against the cover problem, which the key holds as hashes:
        # each playback checks the answer solves the problem it builds.
        pytest.skip("an exact cover is read against its problem")
    collected: dict = {}
    arguments = _arguments(entry, tmp_path, collected)
    result = declared.read(entry.answer, arguments)
    if declared is recorded.MODEL_AGENT:
        # The run receives the reply as a chat result, the chat model's own.
        result = result.generations[0].message
    callbacks = {name: collected.get(name, []) for name in declared.callbacks}
    stored = recorded.stored_answer(declared, result, arguments, callbacks)
    assert canonical(stored) == canonical(entry.answer)


def test_every_kind_the_recordings_hold_is_declared():
    kinds = {entry.kind for data_dir in RECORDINGS.values()
             for entry in Scenario.load(Path(str(data_dir / "scenario.json.gz"))).entries}
    assert kinds <= set(recorded.CALLS)
    assert all(entry.kind == kind for kind, entry in recorded.CALLS.items())


#: A named test's id object as a recording stores it.
NAMED_FIELDS = {"file": "test_calc.py", "qualname": "test_add", "params": None,
                "text": "test_calc.py::test_add"}


def test_a_recorded_run_keeps_its_named_tests_and_a_replay_rebuilds_them():
    """A run that names failing tests is recorded with their id objects'
    fields, and a replay gives the id objects back. A run that names none
    is recorded without the field."""
    from langadapter.suites import TestId
    from testuary.languages import TestFailure, TestRunResult

    failing = TestId("test_calc.py", "test_add", text="test_calc.py::test_add")
    result = TestRunResult(
        ok=False, returncode=1, timed_out=False, elapsed_s=0.1, collected=1,
        passed=0, failed=1, errors=0, output="", line_coverage=None,
        branch_coverage=None, failures=(TestFailure("test_calc.test_add", "failed", "", ""),),
        stopping_tests=(failing,))
    data = json.loads(json.dumps(recorded.run_result_json(result)))
    assert data["stopping_tests"] == [NAMED_FIELDS]
    replayed = recorded.test_run_result(data).stopping_tests
    assert replayed == (failing,) and replayed[0].text == failing.text
    assert "stopping_tests" not in recorded.run_result_json(result._replace(stopping_tests=()))


def test_an_evidence_failure_that_names_tests_is_recorded_and_replayed_with_them(tmp_path):
    """The evidence run's failure callback gets the named tests by keyword:
    a recording stores their fields after the failure's kind and detail,
    and a replay hands the rebuilt id objects to the callback."""
    from langadapter.suites import TestId

    failing = TestId("test_calc.py", "test_add", text="test_calc.py::test_add")

    class Backend:
        def collect_eviction_data(self, *, failure_fn, **_arguments):
            failure_fn("infrastructure", "the counts cannot be read", stopping_tests=(failing,))

    entry = recorded.COLLECT_EVICTION_DATA
    arguments = _arguments(entry, tmp_path, {})
    result, calls = recorded.real_result(entry, Backend(), arguments)
    stored = json.loads(json.dumps(recorded.stored_answer(entry, result, arguments, calls)))
    assert stored["failure"] == ["infrastructure", "the counts cannot be read", [NAMED_FIELDS]]
    replayed = []
    arguments.values["failure_fn"] = lambda kind, detail, stopping_tests=(): replayed.append(
        (kind, detail, stopping_tests))
    assert entry.read(stored, arguments) is None
    assert replayed == [("infrastructure", "the counts cannot be read", (failing,))]
