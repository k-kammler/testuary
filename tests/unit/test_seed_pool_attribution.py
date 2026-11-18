"""Each starter suite's mutation result, measured alone, kept per slot.

The starter suites are measured per suite: each suite's verdicts equal a
standalone run of it. The record keeps each slot's counts from its own
matrix, the verdicts behind them and how the measurement ran, and its reader
refuses malformed or contradictory records.
"""
from __future__ import annotations

import json

from testuary import measurement as metrics
from testuary.consolidation import (
    persist_phase_end,
    PhaseConsolidation,
    PhaseEndDelivery,
)


def _own(*statuses):
    return {"mutants": [{"id": f"m{i}", "status": status, **(
        {"killer": "test_calc.py::test_a"} if status == "killed" else {})}
        for i, status in enumerate(statuses, 1)]}


def _measured():
    # Slot 1 kills m1 and times out on m2. Slot 3 kills only m3, and its run
    # has no result for m4. Slot 2 has no accepted suite.
    rows = [
        {"id": "m1", "status": "killed", "suite_verdicts": {
            "suite_001": {"verdict": "killed", "killers": ["test_calc.py::test_a"]},
            "suite_003": {"verdict": "survived", "killers": []}}},
        {"id": "m2", "status": "timeout", "suite_verdicts": {
            "suite_001": {"verdict": "timeout", "killers": []},
            "suite_003": {"verdict": "survived", "killers": []}}},
        {"id": "m3", "status": "killed", "suite_verdicts": {
            "suite_001": {"verdict": "survived", "killers": []},
            "suite_003": {"verdict": "killed", "killers": ["test_calc.py::test_a"]}}},
        {"id": "m4", "status": "untested", "suite_verdicts": {
            "suite_001": {"verdict": "survived", "killers": []},
            "suite_003": {"verdict": "untested", "killers": []}}},
    ]
    return {"union": {"mutants": rows},
            "suites": {"suite_001": _own("killed", "timeout", "survived", "survived"),
                       "suite_003": _own("survived", "survived", "killed", "untested")},
            "implementation": "per_suite_runs", "unmeasured": []}


def _record(**overrides):
    record = metrics.per_suite_attribution_record(
        _measured(), slots_by_name={"suite_001": 1, "suite_003": 3}, requested_slots=3, mode="full")
    record.update(overrides)
    return record


def test_each_slot_counts_its_own_measurement_and_its_verdicts():
    record = _record()
    assert (record["schema"], record["basis"], record["implementation"]) == (
        metrics.STARTER_POOL_ATTRIBUTION_SCHEMA, "per_suite_verdicts", "per_suite_runs")
    first, second, third = record["slots"]
    assert (first["valid"], first["killed"]) == (4, 2)
    assert first["verdicts"] == {"killed": 1, "collection_error": 0, "non_viable": 0,
                                 "timeout": 1, "survived": 2, "untested": 0}
    # A slot without an accepted suite has no counts, never zeros.
    assert second == {"slot": 2, "accepted": False, "measured": False, "killed": None}
    # A mutant the suite's run did not reach is not measured, not survived,
    # and counts as not killed.
    assert (third["valid"], third["killed"], third["verdicts"]["untested"]) == (4, 1, 1)


def test_the_verdict_keys_are_langadapter_s_verdicts():
    """Each measured slot counts its verdicts under LangAdapter's verdict
    words, in their order."""
    from langadapter.mutation import VERDICTS

    first = _record()["slots"][0]
    assert list(first["verdicts"]) == list(VERDICTS) == [
        "killed", "collection_error", "non_viable", "timeout", "survived", "untested"]


def test_an_accepted_suite_without_a_result_carries_no_counts():
    measured = _measured()
    del measured["suites"]["suite_003"]
    record = metrics.per_suite_attribution_record(
        measured, slots_by_name={"suite_001": 1, "suite_003": 3}, requested_slots=3, mode="first")
    assert record["slots"][2] == {"slot": 3, "accepted": True, "measured": False, "killed": None}
    assert metrics.per_suite_attribution_record(
        None, slots_by_name={}, requested_slots=1, mode="first") is None


def test_a_slot_carries_its_suites_run_outcome():
    # Slot 1's run was cut by the wall clock and keeps its counts. Slot 3's
    # run measured nothing: no counts, never zeros, and its outcome says why.
    measured = _measured()
    del measured["suites"]["suite_003"]
    measured["outcomes"] = {"suite_001": "timed_out", "suite_003": "infrastructure_failure"}
    record = metrics.per_suite_attribution_record(
        measured, slots_by_name={"suite_001": 1, "suite_003": 3}, requested_slots=3, mode="full")
    first, second, third = record["slots"]
    assert (first["outcome"], first["killed"]) == ("timed_out", 2)
    assert "outcome" not in second
    assert third == {"slot": 3, "accepted": True, "measured": False,
                     "outcome": "infrastructure_failure", "killed": None}


def test_scores_are_each_suites_standalone_score():
    assert metrics.per_suite_scores(_measured()) == {"suite_001": 50.0, "suite_003": 25.0}
    assert metrics.per_suite_scores(None) == {}


def test_the_record_is_written_beside_the_merged_matrix_and_read_back(tmp_path):
    phase = PhaseConsolidation()
    phase.starter_pool_km = _measured()["union"]
    phase.starter_pool_attribution = _record()
    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="merged", delivered_km=None))
    assert (tmp_path / "starter_pool_kill_matrix.json").is_file()
    assert json.loads((tmp_path / "starter_pool_attribution.json").read_text("utf-8")) == _record()
    record = json.loads((tmp_path / "consolidation_record.json").read_text("utf-8"))
    assert "starter_pool_attribution.json" in record["artifacts"]
