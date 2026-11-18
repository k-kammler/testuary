"""The starter suites' results derived from one pooled run."""
from __future__ import annotations

from testuary.languages import get_backend
from testuary import starter_selection
import pytest

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

ALPHA = "from calc import add\n\n\ndef test_alpha():\n    assert add(1, 1) == 2\n"
BETA = "from calc import add\n\n\ndef test_beta():\n    assert add(2, 2) == 4\n"
JAVA = ("package demo;\n\nimport org.junit.jupiter.api.Test;\n\n"
        "public class CalcTest {\n    @Test void adds() {}\n}\n")


def _row(mid, status, line, killers=()):
    row = {"id": mid, "operator": "core/Op", "line": line, "diff": f"- {mid}\n+ x", "status": status}
    if killers:
        row["killers"] = list(killers)
    return row


POOLED = {"mutants": [
    _row("m1", "killed", 2, ["test_candidate_suite_001.py::test_alpha"]),
    _row("m2", "killed", 3),
    _row("m3", "timeout", 4),
    _row("m4", "survived", 5),
    _row("m6", "killed", 6, ["test_candidate_suite_002.py::test_beta"]),
]}


def _python_pool(tmp_path):
    pool, reason = starter_selection.stage_pool(
        get_backend("python"),
        {"suite_001": ("test_candidate.py", ALPHA), "suite_002": ("test_candidate.py", BETA)},
        tmp_path)
    assert reason is None
    return pool


def test_python_suites_are_staged_by_the_backend_under_distinct_names(tmp_path):
    """Two starters staged alone as the same file get the slot in their
    pooled names, their texts unchanged, and the pool is written."""
    pool = _python_pool(tmp_path)
    assert [(f.slot, f.original, f.path) for f in pool.files] == [
        ("suite_001", "test_candidate.py", "test_candidate_suite_001.py"),
        ("suite_002", "test_candidate.py", "test_candidate_suite_002.py")]
    assert (tmp_path / "test_candidate_suite_002.py").read_text(encoding="utf-8") == BETA


def test_java_suites_are_pooled_under_renamed_classes(tmp_path):
    """Two Java suites declaring the same class run together under
    renamed classes, and a pooled killer maps back to the suite's own
    class."""
    backend = get_backend("java")
    suites = {"suite_001": ("demo/CalcTest.java", JAVA), "suite_002": ("demo/CalcTest.java", JAVA)}
    pool, reason = starter_selection.stage_pool(backend, suites, tmp_path)
    assert reason is None
    assert [(f.path, f.renamed) for f in pool.files] == [
        ("demo/CalcTest_suite_001.java", (("CalcTest", "CalcTest_suite_001"),)),
        ("demo/CalcTest_suite_002.java", (("CalcTest", "CalcTest_suite_002"),))]
    assert (tmp_path / "demo" / "CalcTest_suite_002.java").is_file()
    pooled = {"mutants": [_row("j1", "killed", 7, ["demo.CalcTest_suite_002.adds()"])]}
    derived = starter_selection.derive_suite_matrix(pool, pooled, "suite_002", {"demo.CalcTest.adds": {"L:Calc.java:7"}})
    assert derived["mutants"][0]["killers"] == ["demo.CalcTest.adds()"]
    assert starter_selection.derive_suite_matrix(
        pool, pooled, "suite_001", {"demo.CalcTest.adds": {"L:Calc.java:9"}})["mutants"][0]["status"] == "survived"


def test_the_derivation_restricts_the_pool_to_the_suites_tests_that_ran(tmp_path):
    """The suite's killers, a kill no test is named for, and a timeout on a
    line its tests cover are credited. Another suite's kill is not, and
    neither is a timeout on a line only another suite's test covers."""
    pool = _python_pool(tmp_path)
    alpha = {"test_candidate.test_alpha": {"L:calc.py:2"}}
    tests = {**alpha, "test_candidate.test_beta": {"L:calc.py:4"}}
    ran = starter_selection.suite_tokens(alpha)
    assert ran == alpha
    derived = starter_selection.derive_suite_matrix(pool, POOLED, "suite_001", ran)
    assert [row["status"] for row in derived["mutants"]] == [
        "killed", "killed", "survived", "survived", "survived"]
    assert derived["mutants"][0]["killers"] == ["test_candidate.py::test_alpha"]
    covering = starter_selection.suite_tokens(tests)
    assert starter_selection.derive_suite_matrix(pool, POOLED, "suite_001", covering)["mutants"][2]["status"] == "timeout"
    assert starter_selection.derive_suite_matrix(pool, POOLED, "suite_001", {}) is None
    assert starter_selection.counts(derived) == {"valid": 5, "killed": 2, "score": 40.0}
    assert starter_selection.covered_lines(tests) == {2, 4}


def test_differences_name_the_rule_and_refuse_to_match_by_position(tmp_path):
    pool = _python_pool(tmp_path)
    derived = starter_selection.derive_suite_matrix(pool, POOLED, "suite_001", {"t": {"L:calc.py:4"}})
    measured = {"mutants": [
        _row("m1", "killed", 2), _row("m2", "survived", 3),
        {**_row("m3", "timeout", 4), "diff": "- other\n+ y"}, _row("m5", "survived", 6),
    ]}
    found = starter_selection.differences(POOLED, derived, measured, killed={"m1"}, lines={4})
    raw = {"pooled_killers": [], "measured_killers": []}
    assert found == [
        {"mutant_id": "m2", "derived_status": "killed", "measured_status": "survived",
         "derivation_rule": "without_named_test", "pooled_status": "killed", **raw},
        {"mutant_id": "m3", "derived_status": "timeout", "measured_status": "timeout",
         "cause": "different_mutants_under_one_id", "pooled_status": "timeout", **raw},
        {"mutant_id": "m4", "derived_status": "survived", "measured_status": None,
         "cause": "mutant_in_one_matrix", "pooled_status": "survived", **raw},
        {"mutant_id": "m5", "derived_status": None, "measured_status": "survived",
         "cause": "mutant_in_one_matrix", "pooled_status": None, **raw},
        {"mutant_id": "m6", "derived_status": "survived", "measured_status": None,
         "cause": "mutant_in_one_matrix", "pooled_status": "killed",
         "pooled_killers": ["test_candidate_suite_002.py::test_beta"], "measured_killers": []},
    ]


def test_rows_with_a_catalogue_key_match_across_runs_with_other_ids(tmp_path):
    """The engine's catalogue key matches rows whose ids differ between the
    pooled and the per-suite runs (catalogue cache off). The record keeps
    both ids and each side's raw status and killers."""
    pool = _python_pool(tmp_path)
    key = {"file": "calc.py", "operator": "core/Op", "occurrence": 0, "line": 2}
    pooled = {"mutants": [{**_row("p1", "killed", 2, ["test_candidate_suite_001.py::test_alpha"]),
                           "key": key}]}
    measured = {"mutants": [{**_row("s9", "survived", 2), "key": key, "diff": "other text"}]}
    derived = starter_selection.derive_suite_matrix(pool, pooled, "suite_001", {"t": {"L:calc.py:2"}})
    assert starter_selection.differences(pooled, derived, measured, killed={"p1"}, lines={2}) == [{
        "mutant_id": "p1", "measured_mutant_id": "s9", "derived_status": "killed",
        "measured_status": "survived", "derivation_rule": "kept_killer",
        "pooled_status": "killed", "pooled_killers": ["test_candidate_suite_001.py::test_alpha"],
        "measured_killers": []}]


def test_the_record_carries_none_for_a_missing_result_never_zeros(tmp_path):
    pool = _python_pool(tmp_path)
    one = starter_selection.derive_suite_matrix(pool, POOLED, "suite_001", {"t": {"L:calc.py:2"}})
    record = starter_selection.record(
        requested="both", used="both", reason=None, attribution={"pooled": "all", "measured": "first"},
        slots_by_label={"suite_001": 1, "suite_002": 3}, requested_slots=3,
        pool_names={"suite_001": "test_candidate_suite_001.py", "suite_002": "test_candidate_suite_002.py"},
        pooled_km=POOLED,
        derived={"suite_001": one, "suite_002": None}, measured={"suite_001": one},
        kept={"suite_001": ({"m1"}, {2})}, seconds={"pooled": 1.5, "per_suite": 9.0}, pool=pool)
    first, second, third = record["slots"]
    assert first["differences"] == [] and first["derived"] == first["measured"]
    assert first["measured_identity"]["mutants"] == 5 and first["measured_budget"] is None
    assert first["staged_files"][0]["path"] == "test_candidate_suite_001.py"
    assert second == {"slot": 2, "accepted": False}
    assert (third["derived"], third["measured"]) == (None, None)
    assert third["derivation_unavailable"] == "no per-test evidence for this suite"
    assert "differences" not in third
    assert record["attribution"] == {"pooled": "all", "measured": "first"}
    assert record["mean_score"] == {"derived": 40.0, "measured": 40.0}
    assert record["seconds"] == {"pooled": 1.5, "per_suite": 9.0}
    assert record["pooled"]["files"] == ["test_candidate_suite_001.py", "test_candidate_suite_002.py"]
    assert record["pooled"]["identity"]["mutants"] == 5 and record["pooled"]["budget"] is None


UNSTAGEABLE = "the starter suites cannot be pooled (unsupported: no pooled staging)"


def test_an_unstageable_pool_gives_every_slot_its_reason():
    record = starter_selection.record(
        requested="derive", used="measure", reason=UNSTAGEABLE,
        attribution={"pooled": None, "measured": "first"}, slots_by_label={"suite_001": 1},
        requested_slots=1, pool_names={}, pooled_km=None, derived={}, measured=None,
        kept={}, seconds={"pooled": None, "per_suite": 3.0})
    assert record["pooled"] is None
    assert record["slots"][0]["derivation_unavailable"] == UNSTAGEABLE
    assert record["slots"][0]["staged_files"] is None


def _measured_record(matrix, outcomes=None):
    return starter_selection.record(
        requested="both", used="both", reason=None, attribution={"pooled": None, "measured": "full"},
        slots_by_label={"suite_001": 1}, requested_slots=1, pool_names={},
        pooled_km=None, derived={"suite_001": None}, measured={"suite_001": matrix},
        kept={}, seconds={"pooled": None, "per_suite": 1.0}, measured_outcomes=outcomes)


def test_a_measurement_its_wall_clock_ended_gives_no_usable_counts():
    """The resolved mutants of a run the wall clock ended are the fast ones,
    so their counts are no score: the slot records the stamp and the run's
    outcome, and neither a killed nor a valid count to divide."""
    stamp = {"reason": "untested_mutants_in_catalogue", "untested": 81, "total": 85}
    cut = {"mutants": [_row("m1", "killed", 2), _row("m2", "survived", 3),
                       _row("m3", "untested", 4)], "incomplete": stamp}
    measured = _measured_record(cut, {"suite_001": "timed_out"})["slots"][0]["measured"]
    assert measured == {"valid": None, "killed": None, "score": None,
                        "incomplete": stamp, "outcome": "timed_out"}


def test_a_complete_measurement_keeps_its_counts_and_record_shape():
    complete = {"mutants": [_row("m1", "killed", 2), _row("m2", "survived", 3)]}
    record = _measured_record(complete, {"suite_001": "completed"})
    assert record["slots"][0]["measured"] == {"valid": 2, "killed": 1, "score": 50.0}
    assert record == _measured_record(complete)
