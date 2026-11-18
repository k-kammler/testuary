"""Tests for ``testuary.results``.

The writer combines optional coverage, mutation and Finishing results,
on-disk test counts and durable phase timing records. These tests cover row
composition, missing values, score derivation and replacement on resume.
"""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from testuary import results as csv_writer

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _read_one_row(results_dir: Path) -> dict:
    """Read the single row of ``results_dir/tests.csv`` as a dict."""
    csv_path = results_dir / "tests.csv"
    assert csv_path.exists(), f"tests.csv not written at {csv_path}"
    with csv_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1, f"expected exactly 1 row, got {len(rows)}"
    return rows[0]


def _read_csv_path(csv_path: Path) -> list:
    """Return every row of a CSV as a list of dicts."""
    assert csv_path.exists(), f"CSV not written at {csv_path}"
    with csv_path.open(encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))


class _FakeResult:
    """Stand-in for a pydantic Result with the ``to_csv_dict`` contract
    that ``csv_writer._to_dict`` expects."""
    def __init__(self, **kw: Any) -> None:
        self._data = kw
    def to_csv_dict(self) -> dict:
        return dict(self._data)


# ---------------------------------------------------------------------------
# Schema invariants: every Result whose tokens land in tests.csv carries
# input_tokens and output_tokens, or the writer would emit 0 silently.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "import_path,class_name",
    [
        ("testuary.agent.coverage_agent",  "CoverageTestResult"),
        ("testuary.agent.mutation_agent",  "MutationTestResult"),
    ],
)
def test_per_phase_result_carries_usage_fields(import_path, class_name):
    """Each per-phase result declares request and token telemetry fields."""
    import importlib
    mod = importlib.import_module(import_path)
    cls = getattr(mod, class_name)
    fields = cls.model_fields
    assert "model_requests" in fields, (
        f"{class_name} regressed: missing model_requests field"
    )
    assert "input_tokens" in fields, (
        f"{class_name} regressed: missing input_tokens field"
    )
    assert "output_tokens" in fields, (
        f"{class_name} regressed: missing output_tokens field"
    )
    assert fields["input_tokens"].default is None
    assert fields["output_tokens"].default is None
    assert fields["model_requests"].default is None
    assert "model_requests" not in cls.model_json_schema()["properties"]


# ---------------------------------------------------------------------------
# write_tests_row
# ---------------------------------------------------------------------------

def test_write_tests_row_minimal(tmp_path: Path) -> None:
    """All optional sources are None. Row still writes with the
    identification columns + a sensible default score."""
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo",
        target_number=1,
        file_path="src/foo.py",
    )
    row = _read_one_row(tmp_path)
    assert row["target_number"] == "1"
    assert "foo.py" in row["file_path"]
    # No coverage result and no reuse, so no coverage status.
    assert row.get("coverage_status") == ""


def test_write_tests_row_with_test_result(tmp_path: Path) -> None:
    """Write coverage-result fields into the canonical CSV row."""
    test_result = SimpleNamespace(
        line_coverage=85.0,
        branch_coverage=78.0,
        mutation_score=None,
        test_count_standard=42,
        test_loc_standard=120,
        target_loc=300,
        target_complexity=15,
        agent_iterations=8,
        model_requests=11,
        agent_duration_s=120.5,
        status="ok",
        error="",
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo",
        target_number=1,
        file_path="src/foo.py",
        test_result=test_result,
    )
    row = _read_one_row(tmp_path)
    assert row["line_coverage"] == "85.0"
    assert row["branch_coverage"] == "78.0"
    assert row["test_count_delivered"] == "42"
    assert row["target_loc"] == "300"
    assert row["coverage_requests"] == "11"
    assert row["coverage_status"] == "ok"


def test_write_tests_row_mutation_result_overrides_mutation_score(tmp_path: Path) -> None:
    """Prefer the mutation result's score when composing the CSV row."""
    test_result = SimpleNamespace(
        line_coverage=85.0,
        branch_coverage=78.0,
        mutation_score=50.0,         # coverage measurement
        test_count_standard=42,
        test_loc_standard=120,
        target_loc=300,
        target_complexity=15,
        agent_iterations=8,
        agent_duration_s=120.5,
        status="ok",
        error="",
    )
    mutation_result = SimpleNamespace(
        mutation_score=72.5,         # mutation measurement
        test_count_mutation=15,
        test_loc_mutation=40,
        agent_iterations=3,
        model_requests=17,
        agent_duration_s=300.0,
        status="ok",
        error="",
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo",
        target_number=1,
        file_path="src/foo.py",
        test_result=test_result,
        mutation_result=mutation_result,
    )
    row = _read_one_row(tmp_path)
    assert row["mutation_score"] == "72.5"  # mutation result takes precedence
    assert row["mutation_requests"] == "17"
    assert row["mutation_test_count"] == "15"


def test_write_tests_row_finishing_canonicalizes_mutation_score(tmp_path: Path) -> None:
    """The headline mutation_score is the measured score of the delivered suite,
    not the mutation phase's in-loop best, and ``score`` follows it."""
    mutation_result = SimpleNamespace(
        mutation_score=100.0,        # in-loop best (a lucky high draw)
        test_count_mutation=15, test_loc_mutation=40,
        agent_iterations=3, agent_duration_s=300.0, status="ok", error="",
    )
    finishing_result = SimpleNamespace(
        to_csv_dict=lambda: {
            "finishing_mutation_score_full": 92.73,
            "finishing_mutation_measurement": "complete",
            "finishing_status": "success",
        },
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=_switch_test_result(), mutation_result=mutation_result,
        finishing_result=finishing_result,
    )
    row = _read_one_row(tmp_path)
    assert row["mutation_score"] == "92.73"   # consolidated, not the 100 in-loop
    assert row["finishing_mutation_score_full"] == "92.73"
    assert row["score"] == "92.73"            # headline tracks the canonical value


def _incomplete_final(**more) -> "_FakeResult":
    """A Finishing result whose mutation measurement ran and did not resolve its
    catalogue: no score, the resolved mutants' counts as evidence."""
    return _FakeResult(
        finishing_mutation_measurement="incomplete",
        finishing_mutation_score_full=None,
        finishing_killed=5, finishing_survived=3,
        finishing_total=8, finishing_status="failed", **more)


def test_an_unresolved_final_measurement_leaves_the_headline_empty(
        tmp_path: Path) -> None:
    """Neither the in-loop score, a partial ratio nor a placeholder stands for
    a Finishing measurement that did not resolve its catalogue. The counts stay
    as evidence."""
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=_switch_test_result(),
        mutation_result=SimpleNamespace(
            mutation_score=100.0, test_count_mutation=15, test_loc_mutation=40,
            agent_iterations=3, agent_duration_s=300.0, status="ok",
            error=""),
        finishing_result=_incomplete_final(),
    )
    row = _read_one_row(tmp_path)
    assert row["mutation_score"] == ""
    assert row["score"] == ""
    assert row["finishing_mutation_score_full"] == ""
    assert row["finishing_mutation_measurement"] == "incomplete"
    assert (row["finishing_killed"], row["finishing_total"]) == (
        "5", "8")


def test_a_resumed_unresolved_measurement_does_not_keep_the_earlier_score(
        tmp_path: Path) -> None:
    """A resumed run whose Finishing measurement did not resolve writes over a
    row that had a score: the preserved row must not refill the headline,
    the final scores, the mutation phase's snapshot or the score key."""
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        finishing_result=_FakeResult(
            finishing_mutation_measurement="complete",
            finishing_mutation_score_full=90.0,
            finishing_mutation_score_selective=85.0,
            mutation_original_mutation_full=90.0,
            mutation_original_mutation_selective=85.0,
            coverage_final_line=91.0, finishing_status="ok"),
    )
    assert _read_one_row(tmp_path)["mutation_score"] == "90.0"
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        finishing_result=_incomplete_final(coverage_final_line=91.0),
        preserve_existing=True,
    )
    row = _read_one_row(tmp_path)
    for key in ("mutation_score", "score", "finishing_mutation_score_full",
                "finishing_mutation_score_selective",
                "mutation_original_mutation_full",
                "mutation_original_mutation_selective"):
        assert row[key] == "", key
    assert row["finishing_mutation_measurement"] == "incomplete"
    # Columns this write does not measure keep their values.
    assert row["line_coverage"] == "91.0"


def test_a_resumed_complete_measurement_keeps_no_bound_of_the_earlier_one(
        tmp_path: Path) -> None:
    """A resumed run whose Finishing measurement resolved its catalogue writes
    over a row from one that did not: the lower bound and the untested count,
    which only an incomplete measurement has, stay in neither the row nor its
    summary."""
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        finishing_result=_incomplete_final(
            finishing_mutation_score_lower_bound=62.5, finishing_untested=3),
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        finishing_result=_FakeResult(
            finishing_mutation_measurement="complete",
            finishing_mutation_score_full=90.0, finishing_killed=9,
            finishing_survived=1, finishing_total=10, finishing_status="ok"),
        preserve_existing=True,
    )
    row = _read_one_row(tmp_path)
    assert (row["finishing_mutation_score_lower_bound"], row["finishing_untested"]) == ("", "")
    finishing = csv_writer.run_summary(row)["finishing"]
    assert (finishing["mutation_score"], finishing["mutation_score_lower_bound"],
            finishing["mutants_untested"]) == (90.0, None, None)


def test_a_resumed_measurement_without_a_matrix_keeps_no_earlier_count(
        tmp_path: Path) -> None:
    """A resumed run whose Finishing measurement gave no kill matrix writes
    over a row from an incomplete one: none of the earlier measurement's
    counts, its lower bound or its untested count stays."""
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        finishing_result=_incomplete_final(
            finishing_timeout_kills=2, finishing_mutation_score_lower_bound=62.5,
            finishing_untested=3),
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        finishing_result=_FakeResult(
            finishing_mutation_measurement="failed", finishing_status="failed",
            finishing_error="the mutation engine gave no kill matrix"),
        preserve_existing=True,
    )
    row = _read_one_row(tmp_path)
    for key in ("finishing_killed", "finishing_survived", "finishing_total",
                "finishing_timeout_kills", "finishing_mutation_score_lower_bound",
                "finishing_untested"):
        assert row[key] == "", key
    assert row["finishing_mutation_measurement"] == "failed"
    finishing = csv_writer.run_summary(row)["finishing"]
    assert (finishing["mutation_score_lower_bound"], finishing["mutants_untested"]) == (None, None)


def test_a_resumed_clean_result_does_not_bring_back_the_earlier_error(
        tmp_path: Path) -> None:
    """A crashed attempt's row holds the crash's error. A later write whose
    phases ended without an error leaves the error cell empty beside their
    statuses."""
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        crashed=("coverage", "the coverage phase crashed: RuntimeError: boom"),
    )
    assert _read_one_row(tmp_path)["error"] != ""
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=_switch_test_result(),
        finishing_result=_FakeResult(
            finishing_mutation_measurement="complete",
            finishing_mutation_score_full=90.0, finishing_status="ok"),
        preserve_existing=True,
    )
    row = _read_one_row(tmp_path)
    assert (row["coverage_status"], row["finishing_status"], row["error"]) == ("ok", "ok", "")


def test_tests_csv_has_no_column_no_measurement_fills(tmp_path: Path) -> None:
    """The mutation phase revises the one delivered suite, so no measurement
    gives a complexity of its own, and the header names none."""
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py")
    header = (tmp_path / "tests.csv").read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    assert "test_complexity_sum_delivered" in header
    assert not {"mutation_test_complexity_sum", "mutation_test_complexity_max"} & set(header)


def test_a_complete_zero_score_is_a_score(tmp_path: Path) -> None:
    """A Finishing measurement that resolved its catalogue and killed nothing
    reports 0.0, not an empty cell."""
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        finishing_result=_FakeResult(
            finishing_mutation_measurement="complete",
            finishing_mutation_score_full=0.0,
            finishing_killed=0, finishing_total=4,
            finishing_status="ok"),
    )
    row = _read_one_row(tmp_path)
    assert row["mutation_score"] == "0.0"
    assert row["finishing_mutation_score_full"] == "0.0"
    assert row["score"] == "0.0"


def test_a_write_without_a_final_measurement_keeps_the_score(
        tmp_path: Path) -> None:
    """A later write that measured no mutation (a metrics re-run under the
    opt-out, a timing pass, a crash row) keeps a valid score."""
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        finishing_result=_FakeResult(
            finishing_mutation_measurement="complete",
            finishing_mutation_score_full=90.0, finishing_status="ok"),
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        finishing_result=_FakeResult(
            suite_timing_median_ms=140.0, finishing_status="ok"),
        preserve_existing=True,
    )
    row = _read_one_row(tmp_path)
    assert row["mutation_score"] == "90.0"
    assert row["finishing_mutation_score_full"] == "90.0"
    assert row["score"] == "90.0"


def _switch_test_result() -> SimpleNamespace:
    return SimpleNamespace(
        line_coverage=85.0, branch_coverage=78.0, mutation_score=None,
        test_count_standard=42, test_loc_standard=120,
        target_loc=300, target_complexity=15,
        agent_iterations=8, agent_duration_s=120.5, status="ok", error="",
    )


def test_write_tests_row_records_catalogue_switch_columns(tmp_path: Path) -> None:
    """The mutation catalogue-switch analysis columns flow from the mutation
    result into tests.csv so a run can be analyzed for where the selective
    proxy converged and how far the full score was at the switch."""
    mutation_result = SimpleNamespace(
        mutation_score=91.0,
        mutation_switched=True,
        mutation_selective_final=100.0,
        mutation_full_at_switch=88.0,
        test_count_mutation=15, test_loc_mutation=40,
        agent_iterations=3, agent_duration_s=300.0, status="ok", error="",
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=_switch_test_result(), mutation_result=mutation_result,
    )
    row = _read_one_row(tmp_path)
    assert row["mutation_switched"] == "True"
    assert row["mutation_selective_final"] == "100.0"
    assert row["mutation_full_at_switch"] == "88.0"


def test_write_tests_row_switch_columns_blank_when_no_switch(tmp_path: Path) -> None:
    """A run that never switched leaves the analysis columns blank (the
    mutation result carries None), not a misleading 0."""
    mutation_result = SimpleNamespace(
        mutation_score=72.5,
        mutation_switched=False,
        mutation_selective_final=None,
        mutation_full_at_switch=None,
        test_count_mutation=15, test_loc_mutation=40,
        agent_iterations=3, agent_duration_s=300.0, status="ok", error="",
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=_switch_test_result(), mutation_result=mutation_result,
    )
    row = _read_one_row(tmp_path)
    assert row["mutation_switched"] == "False"
    assert row["mutation_selective_final"] == ""
    assert row["mutation_full_at_switch"] == ""


def test_write_tests_row_mutation_error_appends_to_existing_error(tmp_path: Path) -> None:
    """Append the mutation error without discarding the coverage error."""
    test_result = SimpleNamespace(
        line_coverage=None, branch_coverage=None, mutation_score=None,
        test_count_standard=0, test_loc_standard=0, target_loc=0,
        target_complexity=0, agent_iterations=0, agent_duration_s=0,
        status="failed", error="phase-2.1 fail",
    )
    mutation_result = SimpleNamespace(
        mutation_score=None, test_count_mutation=0, test_loc_mutation=0,
        agent_iterations=0, agent_duration_s=0, status="failed",
        error="phase-2.2 fail",
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo",
        target_number=1,
        file_path="src/foo.py",
        test_result=test_result,
        mutation_result=mutation_result,
    )
    row = _read_one_row(tmp_path)
    assert "phase-2.1 fail" in row["error"]
    assert "phase-2.2 fail" in row["error"]
    assert " / " in row["error"]


@pytest.mark.parametrize("coverage, mutation, expected", [
    (("budget_exhausted", "Budget exhausted: UsageLimitExceeded: max_requests_per_phase (20) exceeded"),
     ("agent_error", "the provider refused the request"), "the provider refused the request"),
    (("stagnation", "coverage_stagnation"),
     ("budget_exhausted", "Budget exhausted: UsageLimitExceeded: max_wall_clock_seconds_per_phase (600s) exceeded"),
     "coverage_stagnation"),
])
def test_a_loop_a_limit_ended_leaves_the_error_cell(tmp_path: Path, coverage, mutation, expected) -> None:
    """A request, token or time limit the settings set is a designed ending:
    the loop's status names it, and the ``error`` cell holds only the other
    loop's note."""
    test_result = SimpleNamespace(
        line_coverage=80.0, branch_coverage=70.0, mutation_score=None,
        test_count_standard=3, test_loc_standard=30, target_loc=50,
        target_complexity=5, agent_iterations=4, agent_duration_s=10,
        status=coverage[0], error=coverage[1],
    )
    mutation_result = SimpleNamespace(
        mutation_score=None, test_count_mutation=0, test_loc_mutation=0,
        agent_iterations=0, agent_duration_s=0, status=mutation[0], error=mutation[1],
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=test_result, mutation_result=mutation_result,
    )
    row = _read_one_row(tmp_path)
    assert (row["coverage_status"], row["mutation_status"]) == (coverage[0], mutation[0])
    assert row["error"] == expected


def test_write_tests_row_finishing_overrides_coverage_phase(tmp_path: Path) -> None:
    """The final ``coverage_final_line`` overrides the coverage phase's
    ``line_coverage``: the delivered suite's measurement is canonical."""
    test_result = SimpleNamespace(
        line_coverage=85.0,
        branch_coverage=78.0,
        mutation_score=None,
        test_count_standard=42,
        test_loc_standard=120,
        target_loc=300,
        target_complexity=15,
        agent_iterations=8,
        agent_duration_s=120.5,
        status="ok",
        error="",
    )
    # ``_to_dict`` calls ``to_csv_dict()`` or ``model_dump()`` and turns a
    # plain dict into ``{}``, so ``_FakeResult`` provides the contract.
    finishing_result = _FakeResult(
        coverage_final_line=92.5,
        coverage_final_branch=86.0,
        target_complexity_sum=22,
        target_loc=305,
        finishing_duration_s=12.3,
        finishing_status="ok",
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo",
        target_number=1,
        file_path="src/foo.py",
        test_result=test_result,
        finishing_result=finishing_result,
    )
    row = _read_one_row(tmp_path)
    assert row["line_coverage"] == "92.5"        # metrics overrode coverage's 85
    assert row["branch_coverage"] == "86.0"
    assert "scaled_line_coverage" not in row
    assert row["target_complexity"] == "22"      # metrics overrode coverage's 15
    assert row["target_loc"] == "305"            # metrics overrode coverage's 300
    assert row["finishing_status"] == "ok"


def test_a_final_coverage_run_without_a_value_leaves_the_coverage_cells_empty(
        tmp_path: Path) -> None:
    """The coverage phase's values describe another suite than the delivered
    one, so they never fill in for a final coverage run that gave no value,
    not even through a preserved row, and the run summary is not complete."""
    from testuary import results as run_summary

    final = dict(finishing_mutation_measurement="complete",
                 finishing_mutation_score_full=88.0, finishing_status="failed",
                 finishing_error="the final coverage measurement gave no value")
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=SimpleNamespace(
            line_coverage=85.0, branch_coverage=78.0, mutation_score=None,
            test_count_standard=42, test_loc_standard=120, target_loc=300,
            target_complexity=15, agent_iterations=8, agent_duration_s=120.5,
            input_tokens=111, output_tokens=222, status="ok", error=""),
        finishing_result=_FakeResult(coverage_final_line=None,
                                     coverage_final_branch=None, **final),
    )
    row = _read_one_row(tmp_path)
    assert (row["line_coverage"], row["branch_coverage"]) == ("", "")
    assert row["mutation_score"] == "88.0"
    assert run_summary.run_summary(row)["fully_measured"] is False
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        finishing_result=_FakeResult(coverage_final_line=91.0,
                                     coverage_final_branch=80.0, **final),
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        finishing_result=_FakeResult(coverage_final_line=None,
                                     coverage_final_branch=None, **final),
        preserve_existing=True,
    )
    row = _read_one_row(tmp_path)
    assert (row["line_coverage"], row["branch_coverage"]) == ("", "")


def test_write_tests_row_preserve_existing_backfills_metrics(tmp_path: Path) -> None:
    """A re-measure over a reused suite (preserve_existing=True) updates the
    metric columns in place, keeps the reused phases' coverage and mutation
    columns, and ``score`` follows the canonical mutation score."""
    # A full prior row: coverage, mutation and a first metrics pass.
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=SimpleNamespace(
            line_coverage=85.0, branch_coverage=78.0, mutation_score=None,
            test_count_standard=42, test_loc_standard=120,
            target_loc=300, target_complexity=15,
            agent_iterations=8, agent_duration_s=120.5,
            input_tokens=111, output_tokens=222, status="ok", error="",
        ),
        mutation_result=SimpleNamespace(
            mutation_score=72.5, test_count_mutation=15, test_loc_mutation=40,
            agent_iterations=3, agent_duration_s=300.0,
            input_tokens=333, output_tokens=444, status="ok", error="",
        ),
        finishing_result=_FakeResult(
            coverage_final_line=90.0, finishing_mutation_score_full=88.0,
            finishing_mutation_measurement="complete", finishing_status="ok",
        ),
    )

    # Re-measure: coverage and mutation are reused (None), only the metrics
    # are recomputed, with higher coverage and a fresh full mutation score.
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=None,
        mutation_result=None,
        finishing_result=_FakeResult(
            coverage_final_line=97.0, finishing_mutation_score_full=91.0,
            finishing_mutation_measurement="complete", finishing_status="ok",
        ),
        preserve_existing=True,
    )

    with (tmp_path / "tests.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1            # in-place update, not a second row
    row = rows[0]
    # Metric columns recomputed by the re-measure.
    assert row["line_coverage"] == "97.0"
    assert row["finishing_mutation_score_full"] == "91.0"
    # mutation_score follows the re-measure, not the in-loop 72.5.
    assert row["mutation_score"] == "91.0"
    # Reused columns without a fresh result stay.
    assert row["mutation_test_count"] == "15"
    assert row["test_count_delivered"] == "42"
    assert row["coverage_input_tokens"] == "111"
    assert row["mutation_input_tokens"] == "333"
    # Headline score tracks the canonical mutation score.
    assert row["score"] == "91.0"


def test_write_tests_row_emits_finishing_metrics(tmp_path: Path) -> None:
    """Finishing's standalone test-evaluation metrics (full-catalogue
    mutation, the combined counts and coverage) flow into the tests.csv row
    under their ``final_`` names."""
    finishing_result = _FakeResult(
        finishing_mutation_score_full=80.0,
        finishing_mutation_score_selective=72.0,
        finishing_mutation_measurement="complete",
        finishing_killed=9,
        finishing_survived=1,
        finishing_total=10,
        finishing_line_coverage=95.0,
        finishing_branch_coverage=88.0,
        finishing_status="ok",
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo",
        target_number=1,
        file_path="src/foo.py",
        finishing_result=finishing_result,
    )
    row = _read_one_row(tmp_path)
    assert row["finishing_mutation_score_full"] == "80.0"
    assert row["finishing_mutation_score_selective"] == "72.0"
    assert row["finishing_mutation_measurement"] == "complete"
    assert row["mutation_score"] == "80.0"
    assert row["finishing_total"] == "10"
    assert row["finishing_line_coverage"] == "95.0"


def test_a_row_written_into_an_older_file_rewrites_it_with_the_current_columns(
        tmp_path: Path) -> None:
    """A resumed run writes into an earlier version's tests.csv with dropped
    columns. The file is rewritten with the current columns: the other
    target's row keeps its values and the dropped columns go."""
    old_header = csv_writer.TESTS_FIELDNAMES + ["scaled_line_coverage",
                                                "final_kill_jaccard"]
    with open(tmp_path / "tests.csv", "w", newline="",
              encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=old_header,
                                lineterminator="\r\n")
        writer.writeheader()
        writer.writerow({"target_number": 2, "run": 0, "file_path": "src/bar.py",
                         "line_coverage": 70.0, "scaled_line_coverage": "",
                         "final_kill_jaccard": 0.0})
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        finishing_result=_FakeResult(coverage_final_line=92.5),
    )
    with open(tmp_path / "tests.csv", newline="", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)
        header, rows = reader.fieldnames, list(reader)
    assert "scaled_line_coverage" not in header
    assert "final_kill_jaccard" not in header
    assert [(r["target_number"], r["line_coverage"]) for r in rows] == [
        ("2", "70.0"), ("1", "92.5")]


def test_tests_fieldnames_carry_suite_runtime_columns() -> None:
    """The suite timing columns are in the tests.csv header, so
    append_csv_row's extrasaction='ignore' does not drop a written value."""
    for col in (
        "suite_timing_median_ms", "suite_timing_samples_ms",
        "suite_timing_run_count", "suite_timing_warmup_count",
        "suite_timing_requested_cpus", "suite_timing_allowed_cpus",
    ):
        assert col in csv_writer.TESTS_FIELDNAMES


def test_write_tests_row_emits_suite_runtime_columns(tmp_path: Path) -> None:
    """A finalize result carrying the suite_timing_* fields flows them into
    the tests.csv row under those names."""
    finishing_result = _FakeResult(
        suite_timing_median_ms=142.0,
        suite_timing_samples_ms="140.0;142.0;150.0",
        suite_timing_run_count=3,
        suite_timing_warmup_count=1,
        suite_timing_requested_cpus="0-3",
        suite_timing_allowed_cpus="0,1,2,3",
        finishing_status="ok",
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        finishing_result=finishing_result,
    )
    row = _read_one_row(tmp_path)
    assert row["suite_timing_median_ms"] == "142.0"
    assert row["suite_timing_samples_ms"] == "140.0;142.0;150.0"
    assert row["suite_timing_run_count"] == "3"
    assert row["suite_timing_warmup_count"] == "1"
    assert row["suite_timing_requested_cpus"] == "0-3"
    assert row["suite_timing_allowed_cpus"] == "0,1,2,3"


def test_write_tests_row_suite_runtime_default_off_leaves_columns_empty(
    tmp_path: Path,
) -> None:
    """Without a timing measurement the suite_timing_* columns are in the
    header but empty."""
    finishing_result = _FakeResult(
        coverage_final_line=90.0, finishing_status="ok",
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        finishing_result=finishing_result,
    )
    row = _read_one_row(tmp_path)
    assert row["suite_timing_median_ms"] == ""
    assert row["suite_timing_samples_ms"] == ""
    assert row["suite_timing_run_count"] == ""


def test_write_tests_row_suite_runtime_does_not_overwrite_existing_cells(
    tmp_path: Path,
) -> None:
    """A preserve_existing write without timing values keeps the earlier
    suite_timing_* cells and every other column: a None value never blanks
    an existing cell."""
    # A full row including a timing measurement.
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=SimpleNamespace(
            line_coverage=85.0, branch_coverage=78.0, mutation_score=None,
            test_count_standard=42, test_loc_standard=120,
            target_loc=300, target_complexity=15,
            agent_iterations=8, agent_duration_s=120.5,
            input_tokens=111, output_tokens=222, status="ok", error="",
        ),
        mutation_result=SimpleNamespace(
            mutation_score=72.5, test_count_mutation=15, test_loc_mutation=40,
            agent_iterations=3, agent_duration_s=300.0,
            input_tokens=333, output_tokens=444, status="ok", error="",
        ),
        finishing_result=_FakeResult(
            coverage_final_line=90.0, finishing_mutation_score_full=88.0,
            finishing_mutation_measurement="complete",
            suite_timing_median_ms=142.0,
            suite_timing_samples_ms="140.0;142.0",
            suite_timing_run_count=2, suite_timing_warmup_count=1,
            finishing_status="ok",
        ),
    )
    # A later write that recomputes some metrics without timing values.
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=None, mutation_result=None,
        finishing_result=_FakeResult(
            coverage_final_line=97.0, finishing_status="ok",
        ),
        preserve_existing=True,
    )
    row = _read_one_row(tmp_path)
    # Timing preserved.
    assert row["suite_timing_median_ms"] == "142.0"
    assert row["suite_timing_samples_ms"] == "140.0;142.0"
    assert row["suite_timing_run_count"] == "2"
    # Other cells intact. The recomputed coverage still wins.
    assert row["line_coverage"] == "97.0"
    assert row["mutation_test_count"] == "15"
    assert row["mutation_score"] == "88.0"


def test_write_tests_row_emits_per_phase_orig_columns(tmp_path: Path) -> None:
    """A real ``CombinedTestMetrics`` with per-phase snapshots emits the
    ablation columns, with coverage rising from phase to phase, and the
    mutation snapshot reuses Finishing's measurement."""
    from testuary.measurement import (
        CombinedTestMetrics, FinishingMetrics,
    )
    c = CombinedTestMetrics(target_file_path="src/foo.py")
    c.finishing = FinishingMetrics(
        line_coverage=88.0, mutation_score_full=61.0,
    )
    c.phase_finishing["mutation"] = c.finishing
    c.phase_finishing["starter"] = FinishingMetrics(
        line_coverage=55.0, mutation_score_full=33.0,
    )
    c.phase_finishing["coverage"] = FinishingMetrics(
        line_coverage=80.0, mutation_score_full=50.0,
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo",
        target_number=1,
        file_path="src/foo.py",
        finishing_result=c,
    )
    row = _read_one_row(tmp_path)
    assert row["starter_original_line"] == "55.0"
    assert row["starter_original_mutation_full"] == "33.0"
    assert row["coverage_original_line"] == "80.0"
    assert row["mutation_original_line"] == "88.0"
    assert row["mutation_original_mutation_full"] == "61.0"


def test_phase_finishing_columns_disjoint_from_finishing() -> None:
    """The per-phase ablation columns and Finishing's columns both feed
    tests.csv, so their names must not collide."""
    from testuary.measurement import (
        finishing_csv_columns, phase_finishing_csv_columns,
    )
    fin = set(finishing_csv_columns())
    phase = set(phase_finishing_csv_columns())
    assert fin.isdisjoint(phase)


def test_finishing_columns_single_sourced_in_tests_csv(tmp_path: Path) -> None:
    """tests.csv takes its ``final_*`` columns from
    ``finishing_csv_columns()``, not from the fixed ``TESTS_FIELDNAMES``,
    so a new metric needs no schema edit."""
    from testuary.measurement import finishing_csv_columns
    fin_cols = finishing_csv_columns()
    assert fin_cols                                     # the source is non-empty
    # Not in the fixed base,
    assert not any(c in csv_writer.TESTS_FIELDNAMES for c in fin_cols)
    # but every one is in a written row's header.
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
    )
    with (tmp_path / "tests.csv").open(encoding="utf-8-sig") as f:
        header = next(csv.reader(f))
    assert all(c in header for c in fin_cols)


def test_write_tests_row_reused_path_scans_disk(tmp_path: Path) -> None:
    """With test_result None (reused phase), the standard test file on disk
    is scanned, so the count columns are not blank."""
    work_tests = tmp_path / "work_tests"
    work_tests.mkdir()
    standard_file = work_tests / "test_foo.py"
    standard_file.write_text(
        "def test_alpha():\n    assert True\n",
        encoding="utf-8",
    )
    # Source file for target_loc / complexity scan
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src_file = src_dir / "foo.py"
    src_file.write_text("def f():\n    return 42\n", encoding="utf-8")
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo",
        target_number=1,
        file_path=str(src_file),
        # No test_result: the coverage_reused flag, not the missing result,
        # makes the status "reused" (a failed starter phase has none either).
        work_tests=work_tests,
        coverage_reused=True,
    )
    row = _read_one_row(tmp_path)
    assert row["coverage_status"] == "reused"
    assert row["test_count_delivered"] == "1"
    assert int(row["test_loc_delivered"]) >= 1


def test_test_count_recounts_from_disk_when_finishing_reports_zero(
    tmp_path: Path,
) -> None:
    """Finishing in a resumed run can report test_count_delivered=0 (not None),
    which would otherwise stick. The writer recounts from the on-disk suite
    so a suite that exists never records a spurious zero."""
    work_tests = tmp_path / "work_tests"
    work_tests.mkdir()
    (work_tests / "test_foo.py").write_text(
        "def test_a():\n    assert True\n\n\ndef test_b():\n    assert True\n",
        encoding="utf-8")
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    src_file = src_dir / "foo.py"
    src_file.write_text("def f():\n    return 42\n", encoding="utf-8")
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo",
        target_number=1,
        file_path=str(src_file),
        finishing_result=_FakeResult(test_count_delivered=0,
                                     finishing_status="success"),
        work_tests=work_tests,
    )
    row = _read_one_row(tmp_path)
    assert row["test_count_delivered"] == "2"


def test_reported_test_count_not_clobbered_by_disk_backfill(
    tmp_path: Path,
) -> None:
    """The disk recount fires only when the count is falsy: a phase that
    reported a real count keeps it, even if the on-disk file differs."""
    work_tests = tmp_path / "work_tests"
    work_tests.mkdir()
    (work_tests / "test_foo.py").write_text(
        "def test_a():\n    assert True\n", encoding="utf-8")
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    (src_dir / "foo.py").write_text("def f():\n    return 42\n", encoding="utf-8")
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo",
        target_number=1,
        file_path=str(src_dir / "foo.py"),
        test_result=SimpleNamespace(test_count_standard=7, status="ok"),
        work_tests=work_tests,
    )
    row = _read_one_row(tmp_path)
    assert row["test_count_delivered"] == "7"


def test_write_tests_row_score_from_mutation_score(tmp_path: Path) -> None:
    """Use a numeric mutation score for the cross-model score column."""
    test_result = SimpleNamespace(
        line_coverage=85.0, branch_coverage=78.0,
        test_count_standard=42, test_loc_standard=120, target_loc=300,
        target_complexity=15, agent_iterations=8, agent_duration_s=120.5,
        status="ok", error="",
    )
    mutation_result = SimpleNamespace(
        mutation_score=72.5, test_count_mutation=10, test_loc_mutation=40,
        agent_iterations=4, agent_duration_s=60.0, status="ok", error="",
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo",
        target_number=1,
        file_path="src/foo.py",
        test_result=test_result,
        mutation_result=mutation_result,
    )
    row = _read_one_row(tmp_path)
    assert row["score"] == "72.5"


def test_write_tests_row_score_fallback_on_tests_failed(tmp_path: Path) -> None:
    """Write a zero headline score when the coverage result reports tests_failed."""
    test_result = SimpleNamespace(
        line_coverage=None, branch_coverage=None, mutation_score=None,
        test_count_standard=0, test_loc_standard=0, target_loc=0,
        target_complexity=0, agent_iterations=5, agent_duration_s=300,
        status="tests_failed", error="couldn't make tests green",
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo",
        target_number=1,
        file_path="src/foo.py",
        test_result=test_result,
    )
    row = _read_one_row(tmp_path)
    assert row["score"] == "0"


def test_write_tests_row_appends_not_overwrites(tmp_path: Path) -> None:
    """Two ``write_tests_row`` calls for different targets give two rows."""
    for target_number in (1, 2):
        csv_writer.write_tests_row(
            results_dir=tmp_path, workspace="rank1_foo",
            target_number=target_number,
            file_path=f"src/file_{target_number}.py",
        )
    csv_path = tmp_path / "tests.csv"
    with csv_path.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2
    assert rows[0]["target_number"] == "1"
    assert rows[1]["target_number"] == "2"


# ---------------------------------------------------------------------------
# Per-phase token columns
# ---------------------------------------------------------------------------


def test_write_tests_row_populates_phase_token_columns(tmp_path: Path) -> None:
    """Each phase result's ``input_tokens`` and ``output_tokens`` land in the
    matching tests.csv columns."""
    coverage = SimpleNamespace(
        line_coverage=92.5, branch_coverage=88.0, mutation_score=None,
        test_count_standard=10, test_loc_standard=120,
        target_loc=80, target_complexity=12,
        agent_iterations=14, agent_duration_s=45.2,
        input_tokens=12345, output_tokens=2345,
        status="success", error="",
    )
    mutation = SimpleNamespace(
        mutation_score=78.5, test_count_mutation=4, test_loc_mutation=50,
        agent_iterations=8, agent_duration_s=22.1,
        input_tokens=9876, output_tokens=1234,
        status="success", error="",
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=coverage, mutation_result=mutation,
    )
    row = _read_one_row(tmp_path)
    assert row["coverage_input_tokens"] == "12345"
    assert row["coverage_output_tokens"] == "2345"
    assert row["mutation_input_tokens"] == "9876"
    assert row["mutation_output_tokens"] == "1234"
    assert "input_tokens_scaled" not in row


def test_write_tests_row_token_columns_blank_when_result_omits_tokens(
    tmp_path: Path,
) -> None:
    """A skipped phase or a result without token fields leaves the columns
    blank, not 0, so a skipped phase and zero tokens stay distinguishable."""
    coverage = SimpleNamespace(
        line_coverage=90.0, branch_coverage=85.0, mutation_score=None,
        test_count_standard=5, test_loc_standard=60,
        target_loc=40, target_complexity=8,
        agent_iterations=2, agent_duration_s=10.0,
        status="success", error="",
        # NOT setting input_tokens / output_tokens
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=coverage,
    )
    row = _read_one_row(tmp_path)
    # A count the result does not carry is not measured: empty, never 0.
    assert row["coverage_input_tokens"] == ""
    assert row["coverage_output_tokens"] == ""
    # Skipped mutation columns stay empty.
    assert row["mutation_input_tokens"] == ""


def test_a_count_the_result_carries_as_zero_is_written_zero(tmp_path: Path) -> None:
    """A phase that made no model request (it stopped at its start) carries
    0, which is a measurement and is written."""
    coverage = SimpleNamespace(
        line_coverage=90.0, branch_coverage=85.0, mutation_score=None,
        status="success", error="", input_tokens=0, output_tokens=0,
        model_requests=0, logical_requests=0)
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=coverage,
    )
    row = _read_one_row(tmp_path)
    assert (row["coverage_input_tokens"], row["coverage_output_tokens"]) == ("0", "0")
    assert row["coverage_logical_requests"] == "0"


# ---------------------------------------------------------------------------
# Idempotent-by-rank writes (resume safety)
# ---------------------------------------------------------------------------

def test_write_tests_row_same_target_is_idempotent(tmp_path: Path) -> None:
    """A resumed target that rewrites its row replaces it: last write wins."""
    for cov in (70.0, 91.5):
        csv_writer.write_tests_row(
            results_dir=tmp_path, workspace="rank1_foo",
            target_number=1,
            file_path="src/foo.py",
            test_result=SimpleNamespace(
                line_coverage=cov, branch_coverage=0.0,
                mutation_score=None, test_count_standard=1,
                test_loc_standard=1, target_loc=1, target_complexity=1,
                agent_iterations=1, agent_duration_s=1.0,
                status="ok", error="",
            ),
        )
    with (tmp_path / "tests.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["target_number"] == "1"
    # Last write wins.
    assert float(rows[0]["line_coverage"]) == 91.5


def test_write_tests_row_idempotent_preserves_other_targets(tmp_path: Path) -> None:
    """Rewriting rank 2 must leave ranks 1 and 3 intact and ordered."""
    for target_number in (1, 2, 3):
        csv_writer.write_tests_row(
            results_dir=tmp_path, workspace="rank1_foo", target_number=target_number,
            file_path=f"src/f{target_number}.py",
        )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=2,
        file_path="src/f2.py",
    )
    with (tmp_path / "tests.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert [r["target_number"] for r in rows] == ["1", "3", "2"]
    assert len(rows) == 3


def test_append_csv_row_no_dedup_key_still_plain_appends(tmp_path: Path) -> None:
    """Without dedup_key, append_csv_row appends plainly (two rows, one key)."""
    p = tmp_path / "x.csv"
    for i in (1, 1):
        csv_writer.append_csv_row(p, ["target_number", "v"], {"target_number": i, "v": i})
    with p.open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 2


# ---------------------------------------------------------------------------
# Repeated-phase run_idx column
# ---------------------------------------------------------------------------


def test_tests_csv_carries_the_run_column_from_0(tmp_path: Path) -> None:
    """Every row has ``run``, the run as its directory names it (0 for r0, the
    default), placed right after ``rank``."""
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="x.py",
    )
    rows = _read_csv_path(tmp_path / "tests.csv")
    assert rows[0]["run"] == "0" and "run_idx" not in rows[0]
    assert csv_writer.TESTS_FIELDNAMES[:2] == ["target_number", "run"]


def test_tests_csv_dedup_key_is_target_number_plus_run(
    tmp_path: Path,
) -> None:
    """Replace an existing rank/run-index row without removing other repetitions."""
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="x.py",
        run_idx=0,
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="x.py",
        run_idx=1,
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="x.py",
        run_idx=0,
    )
    rows = _read_csv_path(tmp_path / "tests.csv")
    # One row per (rank, run): rewriting the first run replaced its row only.
    assert len(rows) == 2
    pairs = sorted((r["target_number"], r["run"]) for r in rows)
    assert pairs == [("1", "0"), ("1", "1")]


def test_append_csv_row_composite_dedup_key(tmp_path: Path) -> None:
    """Match all composite-key fields while retaining single-key deduplication."""
    csv_path = tmp_path / "x.csv"
    fields = ["target_number", "run_idx", "value"]
    csv_writer.append_csv_row(
        csv_path, fields,
        {"target_number": 1, "run_idx": 1, "value": "a"},
        dedup_key=("target_number", "run_idx"),
    )
    csv_writer.append_csv_row(
        csv_path, fields,
        {"target_number": 1, "run_idx": 2, "value": "b"},
        dedup_key=("target_number", "run_idx"),
    )
    csv_writer.append_csv_row(
        csv_path, fields,
        {"target_number": 1, "run_idx": 1, "value": "c"},  # replaces row 1
        dedup_key=("target_number", "run_idx"),
    )
    rows = _read_csv_path(csv_path)
    assert len(rows) == 2
    by_key = {(r["target_number"], r["run_idx"]): r["value"] for r in rows}
    assert by_key[("1", "1")] == "c"  # replaced
    assert by_key[("1", "2")] == "b"  # preserved


# ---------------------------------------------------------------------------
# Durations come from the target's own durable record
# ---------------------------------------------------------------------------

def _target_with_segments(tmp_path: Path) -> str:
    """A results dir holding one target whose phases recorded their time."""
    from testuary import phase_state

    workspace = "proj__mod_calc"
    target_dir = tmp_path / workspace
    target_dir.mkdir(parents=True)
    with phase_state.timed_segment(target_dir, phase_state.SEGMENT_PIPELINE):
        with phase_state.timed_segment(target_dir, "coverage"):
            phase_state.record_time(phase_state.TIME_MODEL, 120.0)
            phase_state.record_time(phase_state.TIME_COVERAGE, 30.0)
        with phase_state.timed_segment(target_dir, "mutation"):
            phase_state.record_time(phase_state.TIME_MUTATION, 200.0)
    return workspace


def test_phase_duration_is_the_recorded_phase_not_the_agent_loop(
        tmp_path: Path) -> None:
    """The column says what the phase cost. The agent loop's own time is a
    smaller number and gets a column that says so."""
    workspace = _target_with_segments(tmp_path)

    csv_writer.write_tests_row(
        results_dir=tmp_path,
        workspace=workspace,
        target_number=1,
        file_path="proj/mod/calc.py",
        test_result=SimpleNamespace(agent_duration_s=5.0, status="ok"),
        mutation_result=SimpleNamespace(agent_duration_s=7.0, status="ok"),
    )

    row = _read_one_row(tmp_path)
    assert row["coverage_agent_duration_s"] == "5.0"
    assert row["mutation_agent_duration_s"] == "7.0"
    assert float(row["coverage_duration_s"]) >= 0.0
    assert row["coverage_duration_s"] != "5.0"
    assert row["mutation_duration_s"] != "7.0"


def test_the_row_carries_the_target_number_total_and_where_it_went(
        tmp_path: Path) -> None:
    workspace = _target_with_segments(tmp_path)

    csv_writer.write_tests_row(
        results_dir=tmp_path,
        workspace=workspace,
        target_number=1,
        file_path="proj/mod/calc.py",
    )

    row = _read_one_row(tmp_path)
    assert float(row["duration_s_target"]) > 0.0
    assert row["time_model_s"] == "120.0"
    assert row["time_coverage_s"] == "30.0"
    assert row["time_mutation_s"] == "200.0"
    # Nothing else was instrumented, so those cells are empty, not zero.
    assert row["time_merge_s"] == ""
    assert row["time_other_s"] != ""


def test_the_measurement_copies_have_their_own_column(tmp_path: Path) -> None:
    """Copying the project for a measurement is its own part, not test,
    coverage or mutation time."""
    from testuary import phase_state

    workspace = "proj__mod_calc"
    target_dir = tmp_path / workspace
    target_dir.mkdir(parents=True)
    with phase_state.timed_segment(target_dir, phase_state.SEGMENT_PIPELINE):
        with phase_state.timed_segment(target_dir, "coverage"):
            phase_state.record_time(phase_state.TIME_TEST, 4.0)
            phase_state.record_time(phase_state.TIME_PREPARATION, 1.5)
    csv_writer.write_tests_row(results_dir=tmp_path, workspace=workspace, target_number=1,
                               file_path="proj/mod/calc.py")
    row = _read_one_row(tmp_path)
    assert (row["time_test_s"], row["time_preparation_s"]) == ("4.0", "1.5")


def test_target_duration_is_refreshed_after_the_pipeline_segment_closes(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import sys
    import time

    from testuary import phase_state

    readings = iter((1000.0, 1000.01))
    original_clock = time.perf_counter
    segment_methods = (
        phase_state._TimedSegment.__enter__.__code__,
        phase_state._TimedSegment.__exit__.__code__,
    )

    def clock():
        if sys._getframe(1).f_code in segment_methods:
            return next(readings)
        return original_clock()

    monkeypatch.setattr(time, "perf_counter", clock)
    workspace = "proj__mod_calc"
    target_dir = tmp_path / workspace
    target_dir.mkdir(parents=True)
    with phase_state.timed_segment(target_dir, phase_state.SEGMENT_PIPELINE):
        csv_writer.write_tests_row(
            results_dir=tmp_path,
            workspace=workspace,
            target_number=1,
            file_path="proj/mod/calc.py",
        )
        assert _read_one_row(tmp_path)["duration_s_target"] == ""

    csv_writer.refresh_tests_row_durations(
        tmp_path, workspace=workspace, target_number=1,
    )

    row = _read_one_row(tmp_path)
    assert float(row["duration_s_target"]) > 0.0
    assert row["time_other_s"] != ""
    assert next(readings, None) is None


def test_a_reused_phase_reports_what_producing_it_cost(tmp_path: Path) -> None:
    """The reuse path passes no result for the phase it skipped. Without the
    record the cell would be empty, which reads as a phase that took no time."""
    from testuary import phase_state

    workspace = _target_with_segments(tmp_path)
    produced = phase_state.output_seconds(tmp_path / workspace, "coverage")
    phase_state.record_restored(tmp_path / workspace, "coverage")

    csv_writer.write_tests_row(
        results_dir=tmp_path,
        workspace=workspace,
        target_number=1,
        file_path="proj/mod/calc.py",
        test_result=None,
    )

    row = _read_one_row(tmp_path)
    assert float(row["coverage_duration_s"]) == pytest.approx(produced)


def test_without_a_record_the_in_process_wall_clock_still_lands(
        tmp_path: Path) -> None:
    """A crash before any phase opened a segment still reports what it burned."""
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo",
        target_number=1,
        file_path="proj/mod/calc.py",
        phase_durations={"coverage": 12.5},
    )

    row = _read_one_row(tmp_path)
    assert row["coverage_duration_s"] == "12.5"
    assert row["duration_s_target"] == ""


# ---------------------------------------------------------------------------
# Request accounting columns
# ---------------------------------------------------------------------------

_ACCOUNTING_COLUMNS = [
    "starter_logical_requests", "starter_unknown_usage_requests",
    "coverage_logical_requests", "coverage_unknown_usage_requests",
    "mutation_logical_requests", "mutation_unknown_usage_requests",
]


def test_request_accounting_columns_are_appended_after_the_existing_ones():
    """The accounting columns follow the earlier columns, which keep their
    names and order, so old and new tables agree on shared columns."""
    assert csv_writer.TESTS_FIELDNAMES[-11:-5] == _ACCOUNTING_COLUMNS
    assert csv_writer.TESTS_FIELDNAMES[-5] == "source_observations"
    assert csv_writer.TESTS_FIELDNAMES[-4:-1] == [
        "starter_unknown_cache_requests", "coverage_unknown_cache_requests",
        "mutation_unknown_cache_requests"]
    assert csv_writer.TESTS_FIELDNAMES[-1] == "below_minimum_reason"
    assert csv_writer.TESTS_FIELDNAMES[-12] == "error"
    assert len(csv_writer.TESTS_FIELDNAMES) == len(set(csv_writer.TESTS_FIELDNAMES))


def test_request_accounting_columns_are_filled_from_results(tmp_path: Path) -> None:
    """The request columns count requests sent, the accounting columns
    model requests and requests with unknown usage per phase."""
    starter_result = SimpleNamespace(
        requests_used=27, logical_requests=25, unknown_usage_requests=2,
        input_tokens=1000, output_tokens=500, cache_hit_tokens=0,
        duration_s=3.0, suite_generation_attempts={},
    )
    test_result = SimpleNamespace(
        test_file_name="test_foo.py", model_requests=41, logical_requests=40,
        unknown_usage_requests=1, status="success", error="",
    )
    mutation_result = SimpleNamespace(
        mutation_score=72.5, model_requests=17, logical_requests=17,
        unknown_usage_requests=0, status="success", error="",
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        starter_result=starter_result, test_result=test_result,
        mutation_result=mutation_result,
    )
    row = _read_one_row(tmp_path)
    assert row["starter_requests"] == "27"
    assert row["starter_logical_requests"] == "25"
    assert row["starter_unknown_usage_requests"] == "2"
    assert row["coverage_requests"] == "41"
    assert row["coverage_logical_requests"] == "40"
    assert row["coverage_unknown_usage_requests"] == "1"
    assert row["mutation_requests"] == "17"
    assert row["mutation_logical_requests"] == "17"
    assert row["mutation_unknown_usage_requests"] == "0"


def test_request_accounting_columns_are_blank_when_results_lack_them(
    tmp_path: Path,
) -> None:
    """A result restored from an older record carries no counts, so the
    columns stay empty rather than reading zero."""
    starter_result = SimpleNamespace(
        requests_used=27, logical_requests=None, unknown_usage_requests=None,
        input_tokens=1000, output_tokens=500, cache_hit_tokens=0,
        duration_s=3.0, suite_generation_attempts={},
    )
    test_result = SimpleNamespace(
        test_file_name="test_foo.py", model_requests=41, status="success",
        error="",
    )
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        starter_result=starter_result, test_result=test_result,
    )
    row = _read_one_row(tmp_path)
    assert row["starter_requests"] == "27"
    assert row["starter_logical_requests"] == ""
    assert row["starter_unknown_usage_requests"] == ""
    assert row["coverage_requests"] == "41"
    assert row["coverage_logical_requests"] == ""
    assert row["coverage_unknown_usage_requests"] == ""
    assert row["mutation_logical_requests"] == ""


def test_a_failed_starter_phase_row_claims_no_coverage_reuse(
    tmp_path: Path,
) -> None:
    """A starter phase without a green suite ends the target before the
    coverage phase. Its row has no coverage result, which is not a reuse, so
    the coverage status stays empty."""
    from testuary.starters import StarterResult

    starter_result = StarterResult(
        status="cancelled_zero_passing", n_sets_requested=10, n_generated=10,
        requests_used=10, input_tokens=9000, output_tokens=7000)
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=None, starter_result=starter_result, work_tests=None,
        phase_durations={}, run_idx=0, preserve_existing=False,
    )
    row = _read_one_row(tmp_path)
    assert row["starter_status"] == "cancelled_zero_passing"
    assert row["coverage_status"] == ""


def test_a_row_without_a_measured_suite_has_no_score(tmp_path: Path) -> None:
    """A failed starter phase measured no suite, so the score stays empty."""
    from testuary.starters import StarterResult

    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_foo", target_number=1, file_path="src/foo.py",
        test_result=None, starter_result=StarterResult(status="cancelled_zero_passing"),
        work_tests=None, phase_durations={}, run_idx=0, preserve_existing=False,
    )
    assert _read_one_row(tmp_path)["score"] == ""

