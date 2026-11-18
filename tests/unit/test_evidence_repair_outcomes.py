"""``testuary.consolidation._EvidenceRepair``: for each outcome a
repair round can end in, the attempt record's keys in the order
consolidation_record.json writes them, and the repair's own log lines."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from testuary import consolidation
from testuary.consolidation import (
    _EVIDENCE_REPAIR_ROUNDS,
    PhaseConsolidation,
    _EvidenceRepair,
    PhaseEndContext,
    _missing_evidence,
)
from testuary.languages import SourceValidation
from testuary.languages.generic import StructureResult

pytestmark = pytest.mark.usefixtures("default_run")

_ROUND_KEYS = ["round", "missing_lines_before", "missing_branches_before",
               "missing_kills_before", "added_tests", "without_carrier", "accepted"]
_KILLED = {"id": "m1", "operator": "op", "line": 20, "status": "killed",
           "killer": "test_target.py::test_kill"}
_SURVIVED = {"id": "m1", "operator": "op", "line": 20, "status": "survived", "killer": None}


def _reduced():
    """A reduced suite that lacks line 10."""
    return PhaseConsolidation(
        mutation_method="measure", merged_km={"mutants": [dict(_KILLED)]},
        reduced_km={"mutants": [dict(_KILLED)]}, union_covered_lines=[10],
        union_covered_branches=[], suite_covered_lines=[], suite_covered_branches=[],
        line=0.0, branch=None, kept_tests=["base"])


def _repair(monkeypatch, *, tokens=None, valid=True, normal_ok=True, coverage_ok=True,
            load_repairs=None, fewer=False, mutants=(_KILLED,), line_coverage=100.0,
            covered_lines=(10,), repaired_passes=False, repaired_valid=True, result=None,
            named=()):
    """One repair over ``result``, by default _reduced(), with a fake
    backend shaped by the arguments. ``repaired_passes`` lets the normal
    run pass a load-repaired suite, ``repaired_valid`` False makes the
    validator reject one. Returns the kept tests, the attempt records and
    the repair's log lines."""

    class Backend:
        @staticmethod
        def prune_tests(_source, remove):
            return "".join(f"# {t}\n" for t in sorted(set(tokens_of) - set(remove)))

        @staticmethod
        def finalize_test_source(source):
            return source

        @staticmethod
        def check_structure(_source):
            return StructureResult(True, "", 0, 0)

        @staticmethod
        def validate_test_source(source, _source_dir):
            ok = valid and (repaired_valid or "# repaired" not in source)
            return SourceValidation(
                ok, "valid" if ok else "invalid", "" if ok else "bad test", 0, 0,
                "fake-target", hashlib.sha256(source.encode("utf-8")).hexdigest())

        @staticmethod
        def scratch_test_path(directory, _source, _target, *, stage_as_delivered):
            return Path(directory) / "test_target.py"

        @staticmethod
        def run_test_files(paths, _source_dir, *, measure_coverage_of=None, **_kwargs):
            repaired = "# repaired" in Path(paths[0]).read_text(encoding="utf-8")
            ok = (coverage_ok if measure_coverage_of is not None
                  else normal_ok or (repaired_passes and repaired))
            return SimpleNamespace(
                ok=ok, timed_out=False, returncode=0 if ok else 1, collected=2,
                passed=2 if ok else 1, failed=0 if ok else 1, errors=0, failures=(),
                line_coverage=line_coverage, branch_coverage=None,
                covered_lines=covered_lines, covered_branches=(),
                stopping_tests=() if ok else named)

    tokens_of = tokens if tokens is not None else {
        "base": frozenset(), "line": frozenset({"L:target.py:10"})}
    monkeypatch.setattr(consolidation.loadable, "keep_if_loadable",
                        lambda _backend, source, *a, **kw: (
                            (source + "# repaired\n", load_repairs) if load_repairs else (source, {})))
    monkeypatch.setattr(consolidation.loadable, "fewer_tests", lambda *_a: fewer)
    result = _reduced() if result is None else result
    logs = []
    _EvidenceRepair(
        PhaseEndContext(backend=Backend(), target_file_path="target.py",
                        source_dir=".", timeout=10, stage_as_delivered=False,
                        log=logs.append),
        lambda _directory: None if mutants is None else {
            "mutants": [dict(m) for m in mutants]},
        merged="".join(f"# {t}\n" for t in tokens_of), res=result,
        per_test_tokens=tokens_of, solver_cost_ms={t: 1.0 for t in tokens_of}).run({"base"})
    kept = set(result.kept_tests)
    return kept, result.evidence_repairs, [line for line in logs if "evidence" in line]


_ADDING = ("[consolidate] evidence-repair round 1: adding 1 carrier(s) for 1 line, "
           "0 branch, and 0 mutant difference(s)")


def test_a_round_without_a_carrier_stops_and_says_so(monkeypatch):
    kept, attempts, logs = _repair(monkeypatch, tokens={"base": frozenset()})
    assert kept == {"base"}
    assert [list(a) for a in attempts] == [_ROUND_KEYS + ["outcome"]]
    assert attempts[0]["outcome"] == "no_carrier"
    assert logs == ["[consolidate] evidence repair stopped: no unused carrier remains"]


def test_a_test_that_carries_a_missing_line_and_branch_is_added_alone():
    """A test added for a line that also carries a missing branch is the
    branch's carrier too: no second test is added for it."""
    tokens = {"base": frozenset(), "both": frozenset({"L:target.py:10", "B:target.py:10->12"}),
              "branch": frozenset({"B:target.py:10->12"})}
    repair = _EvidenceRepair(
        PhaseEndContext(backend=None, target_file_path="target.py", source_dir=".",
                        timeout=10, stage_as_delivered=False, log=print),
        None, merged="", res=_reduced(), per_test_tokens=tokens,
        solver_cost_ms={"base": 1.0, "both": 2.0, "branch": 1.0})
    assert repair.choose(({10}, {"10->12"}, set()), {"base"}) == ({"both"}, [])


def test_a_kill_a_test_added_for_a_line_carries_adds_no_test():
    """A test added for a missing line that is also a missing kill's named
    killer carries that kill: no second test is added for it."""
    tokens = {"base": frozenset(), "both": frozenset({"L:target.py:10", "K:m1"}),
              "killer": frozenset({"K:m1"})}
    repair = _EvidenceRepair(
        PhaseEndContext(backend=None, target_file_path="target.py", source_dir=".",
                        timeout=10, stage_as_delivered=False, log=print),
        None, merged="", res=_reduced(), per_test_tokens=tokens,
        solver_cost_ms={"base": 1.0, "both": 2.0, "killer": 1.0})
    (identity,) = repair.merged_killed_by_identity
    assert repair.choose(({10}, set(), {identity}), {"base"}) == ({"both"}, [])


def test_a_trial_the_target_validator_rejects_is_invalid(monkeypatch):
    kept, attempts, logs = _repair(monkeypatch, valid=False)
    assert kept == {"base"}
    assert [list(a) for a in attempts] == [_ROUND_KEYS + ["outcome"]]
    assert attempts[0]["outcome"] == "invalid: target validator invalid: bad test"
    assert logs == [_ADDING]


def test_a_trial_whose_load_repair_removes_tests_does_not_load(monkeypatch):
    kept, attempts, logs = _repair(monkeypatch, normal_ok=False,
                                   load_repairs={"removed": ["test_x"]}, fewer=True)
    assert kept == {"base"}
    assert [list(a) for a in attempts] == [_ROUND_KEYS + ["load_repairs", "outcome"]]
    assert attempts[0]["outcome"] == "does_not_load"
    assert attempts[0]["load_repairs"] == {"removed": ["test_x"]}
    assert logs == [_ADDING]


def test_a_trial_whose_load_repaired_bytes_the_validator_rejects_is_invalid(monkeypatch):
    kept, attempts, logs = _repair(monkeypatch, normal_ok=False, repaired_passes=True,
                                   repaired_valid=False, load_repairs={"removed": ["helper"]})
    assert kept == {"base"}
    assert [list(a) for a in attempts] == [_ROUND_KEYS + ["load_repairs", "outcome"]]
    assert attempts[0]["outcome"] == "invalid: target validator invalid: bad test"
    assert logs == [_ADDING]


@pytest.mark.parametrize("load_repairs, keys", [
    (None, _ROUND_KEYS + ["outcome"]),
    ({"removed": ["test_x"]}, _ROUND_KEYS + ["load_repairs", "outcome"]),
])
def test_a_trial_that_fails_its_normal_run_is_not_passing(monkeypatch, load_repairs, keys):
    """Without a load repair, or with one that keeps every test but still
    fails."""
    kept, attempts, logs = _repair(monkeypatch, normal_ok=False, load_repairs=load_repairs)
    assert kept == {"base"}
    assert [list(a) for a in attempts] == [keys]
    assert attempts[0]["outcome"] == "not_passing"
    assert logs == [_ADDING]


@pytest.mark.parametrize("changes, outcome", [
    ({"coverage_ok": False}, "coverage_measurement_incomplete"),
    ({"mutants": None}, "mutation_measurement_incomplete"),
    ({"line_coverage": None}, "aggregate_measurement_incomplete"),
    ({"mutants": (_SURVIVED,)}, "aggregate_regression"),
])
def test_each_measured_trial_that_is_not_kept_names_its_outcome(monkeypatch, changes, outcome):
    kept, attempts, logs = _repair(monkeypatch, **changes)
    assert kept == {"base"}
    assert [list(a) for a in attempts] == [_ROUND_KEYS + ["outcome"]]
    assert attempts[0]["outcome"] == outcome
    assert attempts[0]["accepted"] is False
    assert logs == [_ADDING]


def test_a_trial_whose_coverage_run_names_a_test_removes_nothing(monkeypatch):
    """A repair trial whose coverage run fails and names a test is not
    passing as today: the repair, whose aim is to put tests back, removes
    nothing and writes no removal entry."""
    from langadapter.suites import TestId

    result = _reduced()
    kept, attempts, _logs = _repair(
        monkeypatch, coverage_ok=False, result=result,
        named=(TestId("test_target.py", "line", text="test_target.py::line"),))
    assert kept == {"base"}
    assert attempts[0]["outcome"] == "coverage_measurement_incomplete"
    assert result.measurement_removals == []


def test_an_adopted_trial_ends_the_repair_once_nothing_is_missing(monkeypatch):
    kept, attempts, logs = _repair(monkeypatch)
    assert kept == {"base", "line"}
    assert [list(a) for a in attempts] == [_ROUND_KEYS + ["outcome"]]
    assert attempts[0]["outcome"] == "accepted" and attempts[0]["accepted"] is True
    assert logs == [_ADDING, "[consolidate] evidence-repair round 1: adopted 2 test(s)"]


def test_a_trial_adopted_after_a_load_repair_records_the_bytes_it_adopts(monkeypatch, tmp_path):
    """The load-repaired bytes are validated again, so the record, the
    reduced kill matrices and the delivered reduction's coverage name the
    hash of the suite adopted, not of the trial before its repair."""
    result = _reduced()
    kept, attempts, _logs = _repair(monkeypatch, normal_ok=False, repaired_passes=True,
                                    load_repairs={"removed": ["helper"]}, result=result)
    assert kept == {"base", "line"} and attempts[0]["outcome"] == "accepted"
    assert result.reduced_source.endswith("# repaired\n")
    adopted_hash = hashlib.sha256(result.reduced_source.encode("utf-8")).hexdigest()
    assert result.reduced_source_sha256 == adopted_hash
    result.per_test_tokens = {"base": frozenset(), "line": frozenset({"L:target.py:10"})}
    delivery = consolidation.PhaseEndDelivery(
        winner_label="reduced", delivered_km=result.reduced_km,
        delivered_source=result.reduced_source,
        delivered_candidate=consolidation.scored_from_phase(result),
        execution_name="test_target.py")
    coverage = consolidation._delivered_coverage(result, "target.py", delivery)
    written = consolidation._write_matrices(tmp_path, result, "target.py", delivery, coverage)
    for name in ("reduced_kill_matrix.json", "measured_reduced_kill_matrix.json"):
        assert name in written
        recorded = json.loads((tmp_path / name).read_text(encoding="utf-8"))
        assert recorded["source_sha256"] == adopted_hash
    assert coverage is not None and "selected_coverage_matrix.json" in written


def test_a_derive_repair_derives_each_trial_s_matrix_without_an_engine(monkeypatch):
    """Under derive a trial's matrix is the merged matrix restricted to its
    tests, as derive defines the reduced suite's matrix. The engine, which
    here would give no matrix, never runs."""
    result = _reduced()
    result.mutation_method = "derive"
    tokens = {"base": frozenset({"K:m1"}), "line": frozenset({"L:target.py:10"})}
    kept, attempts, _logs = _repair(monkeypatch, tokens=tokens, mutants=None, result=result)
    assert kept == {"base", "line"} and attempts[0]["outcome"] == "accepted"
    assert result.reduced_km is result.derived_reduced_km
    assert result.measured_reduced_km is None
    assert [m["status"] for m in result.reduced_km["mutants"]] == ["killed"]


def test_an_adopted_trial_s_time_is_its_tests_measured_sum(monkeypatch):
    """The reduced suite's time after an adopted repair is its tests'
    measured per-test sum, and None while one of them has no measured
    time, never a zero for the missing one."""
    timed = _reduced()
    timed.per_test_cost_ms = {"base": 2.0, "line": 3.0}
    _repair(monkeypatch, result=timed)
    assert timed.kept_tests == ["base", "line"] and timed.runtime_ms == 5.0
    untimed = _reduced()
    untimed.per_test_cost_ms = {"base": 2.0}
    _repair(monkeypatch, result=untimed)
    assert untimed.kept_tests == ["base", "line"] and untimed.runtime_ms is None


def test_adopted_rounds_that_leave_evidence_missing_end_with_no_round_left(monkeypatch):
    """Every round adopts a carrier whose run still misses line 10, so the
    last round is followed by the exhausted record."""
    tokens = {"base": frozenset(), **{f"c{i}": frozenset({"L:target.py:10"})
                                     for i in range(1, _EVIDENCE_REPAIR_ROUNDS + 2)}}
    kept, attempts, logs = _repair(monkeypatch, tokens=tokens, covered_lines=())
    assert kept == {"base"} | {f"c{i}" for i in range(1, _EVIDENCE_REPAIR_ROUNDS + 1)}
    assert [list(a) for a in attempts] == [_ROUND_KEYS + ["outcome"]] * _EVIDENCE_REPAIR_ROUNDS + [
        ["round", "missing_lines_after", "missing_branches_after", "missing_kills_after",
         "added_tests", "without_carrier", "accepted", "outcome"]]
    assert [a["outcome"] for a in attempts] == ["accepted"] * _EVIDENCE_REPAIR_ROUNDS + [
        "repair_budget_exhausted"]
    assert logs[-1] == "[consolidate] evidence repair stopped: no repair round left"
    assert logs[:-1] == [line for n in range(1, _EVIDENCE_REPAIR_ROUNDS + 1) for line in (
        f"[consolidate] evidence-repair round {n}: adding 1 carrier(s) for 1 line, "
        "0 branch, and 0 mutant difference(s)",
        f"[consolidate] evidence-repair round {n}: adopted {n + 1} test(s)")]


@pytest.mark.parametrize(
    "merged, reduced, union_lines, suite_lines, union_branches, suite_branches, missing", [
        ([_KILLED], [_SURVIVED], [10, 11], [10], ["1->2", "2->3"], ["1->2"],
         ({11}, {"2->3"}, {("catalogue", "op", 20, None)})),
        ([_KILLED, {**_SURVIVED, "id": "m2", "status": "timeout"},
          {"id": "m3", "key": {"file": "t.py", "operator": "op", "occurrence": 2},
           "status": "killed"}],
         [_KILLED], [], [], [], [], (set(), set(), {("key", "t.py", "op", 2)})),
        ([], [_KILLED], None, None, None, None, (set(), set(), set())),
    ])
def test_the_missing_evidence_is_what_the_merged_suite_has_and_the_reduced_lacks(
        merged, reduced, union_lines, suite_lines, union_branches, suite_branches,
        missing):
    """The merged suite's covered lines, covered branches as text and killed
    mutants by identity that the reduced suite lacks. A timed-out mutant
    counts as killed, and two mutants of one identity count once."""
    res = PhaseConsolidation(
        merged_km={"mutants": merged}, reduced_km={"mutants": reduced},
        union_covered_lines=union_lines, suite_covered_lines=suite_lines,
        union_covered_branches=union_branches, suite_covered_branches=suite_branches)
    assert _missing_evidence(res) == missing
