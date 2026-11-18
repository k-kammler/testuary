"""The summaries of a file's runs: each run's final values and, across the
runs, the fully measured count and each value's mean, minimum and maximum over the
runs that measured it."""
from __future__ import annotations

import csv
import json

from testuary import results as run_summary


def _row(run, line, branch, mutation, status="success"):
    return {"target_number": "3", "run": str(run), "file_path": "pkg/mod.py", "line_coverage": line,
            "branch_coverage": branch, "mutation_score": mutation,
            "starter_status": "success", "coverage_status": status, "mutation_status": status}


def test_a_run_summary_holds_the_rows_final_values_and_outcomes():
    summary = run_summary.run_summary(_row(2, "87.5", "", "60"))
    assert {key: summary[key] for key in (
        "run", "file", "fully_measured", *run_summary.VALUES, *run_summary.STATUSES)} == {
        "run": "r2", "file": "pkg/mod.py", "fully_measured": True,
        "line_coverage": 87.5, "branch_coverage": None, "mutation_score": 60.0,
        "starter_status": "success", "coverage_status": "success", "mutation_status": "success",
    }
    # A row without the detail columns has them unknown, never zero.
    assert summary["phases"]["coverage"]["logical_requests"] is None
    assert summary["starter"]["line_coverage"] == dict.fromkeys(run_summary.STATISTICS)
    assert run_summary.run_summary(_row(0, "40", "", "", "failed"))["fully_measured"] is False


def test_the_summary_across_runs_counts_only_measured_values():
    runs = [run_summary.run_summary(row) for row in (
        _row(0, "80", "60", "50"), _row(1, "90", "", "70"), _row(2, "", "", "", "crashed"))]
    across = run_summary.file_summary("pkg/mod.py", runs)
    assert across["runs"] == ["r0", "r1", "r2"] and across["fully_measured_runs"] == 2
    assert across["line_coverage"] == {"runs": 2, "mean": 85.0, "minimum": 80.0, "maximum": 90.0}
    assert across["branch_coverage"] == {"runs": 1, "mean": 60.0, "minimum": 60.0, "maximum": 60.0}
    assert across["mutation_score"] == {"runs": 2, "mean": 60.0, "minimum": 50.0, "maximum": 70.0}


def test_the_summaries_are_written_beside_the_results(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    rows = [_row(1, "90", "", "70"), _row(0, "80", "", "50"),
            {**_row(0, "10", "", "10"), "target_number": "4"}]
    with open(results / "tests.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    across = run_summary.write_file_summaries(results, "mod", 3)
    assert across["runs"] == ["r0", "r1"]
    assert json.loads((results / "mod" / "summary.json").read_text(encoding="utf-8")) == across
    for k, line in ((0, 80.0), (1, 90.0)):
        written = json.loads((results / "mod" / f"r{k}" / "summary.json").read_text(encoding="utf-8"))
        assert written["line_coverage"] == line
    assert run_summary.write_file_summaries(results, "other", 9) is None
    run_summary.write_project_summary(results)
    project = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    assert project["files"] == [across]
    assert project["values"]["line_coverage"]["mean"] == across["line_coverage"]["mean"]
    assert not list(results.rglob(".summary.*"))


def test_a_run_summary_lists_each_loop_a_limit_stopped_before_its_goal(tmp_path):
    """Beside the time limits, a run's warnings name each loop whose phase
    record ends at one of its limits, with the setting the stop names, so
    the web shows what the end-of-run summary prints. The time limits come
    first. A stop the record does not name is still a warning."""
    from testuary import phase_state

    results = tmp_path / "results"
    (results / "mod").mkdir(parents=True)
    rows = [_row(run, "80", "", "50", status="budget_exhausted") for run in (0, 1, 2)]
    with open(results / "tests.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    ended = "Budget exhausted: UsageLimitExceeded: "
    metrics = {
        ("coverage", 0): "max_input_tokens_per_phase (5000) exceeded (used 5100)",
        ("mutation", 0): "max_total_tokens_per_phase (1000000) exceeded (used 1000400)",
        ("mutation", 1): "max_wall_clock_seconds_per_phase (120s) would be exceeded by the retry "
                         "wait of 30s (elapsed 100s)",
        ("coverage", 2): None,
    }
    state = {}
    for (phase, run), stop in metrics.items():
        recorded = {"status": "budget_exhausted"}
        if stop is not None:
            recorded["error"] = ended + stop
        state.setdefault(phase, {})[str(run)] = {"metrics": recorded}
    state["coverage"]["1"] = {"metrics": {"status": "success"}}
    phase_state.state_path(results / "mod").write_text(json.dumps(state), encoding="utf-8")
    with run_summary.collecting(results / "mod" / "r1"):
        run_summary.note("mutation", "mutation.measurement_timeout_seconds", 600)
    across = run_summary.write_file_summaries(results, "mod", 3)
    first, second, third = (json.loads((results / "mod" / f"r{k}" / "summary.json").read_text(
        encoding="utf-8"))["warnings"] for k in (0, 1, 2))

    def stopped(phase, setting, value, unit, text):
        return {"kind": "loop_limit", "phase": phase, "setting": setting, "value": value,
                "unit": unit, "count": 1, "text": text}

    assert first == [
        stopped("coverage", "coverage.max_input_tokens_per_phase", 5000, None,
                "The coverage phase stopped at its input token limit before reaching its goal "
                "(coverage.max_input_tokens_per_phase = 5000)."),
        stopped("mutation", "mutation.max_total_tokens_per_phase", 1000000, None,
                "The mutation phase stopped at its token limit before reaching its goal "
                "(mutation.max_total_tokens_per_phase = 1000000).")]
    assert [warning["kind"] for warning in second] == ["time_limit", "loop_limit"]
    assert second[1] == stopped(
        "mutation", "mutation.max_wall_clock_seconds_per_phase", 120, "s",
        "The mutation phase stopped at its time limit before reaching its goal "
        "(mutation.max_wall_clock_seconds_per_phase = 120 s).")
    assert third == [stopped("coverage", None, None, None,
                             "The coverage phase stopped at one of its limits before reaching its goal.")]
    assert across["warning_runs"] == 3
    assert run_summary.warnings(results / "mod" / "r1") == second
    assert run_summary.LABELS["loop_limit"] == "Limit reached before the goal"


def test_a_run_summary_carries_each_phases_values_use_starters_and_time():
    row = {**_row(0, "90", "80", "70"), "coverage_original_line": "88.5", "coverage_logical_requests": "12",
           "coverage_input_tokens": "4000", "coverage_turns": "5", "coverage_duration_s": "61.5",
           "finishing_line_coverage": "90", "starter_suite_line_median": "55.5",
           "starter_suite_line_std": "3.25", "duration_s_target": "120", "time_model_s": "70"}
    summary = run_summary.run_summary(row)
    coverage = summary["phases"]["coverage"]
    assert (coverage["line_coverage"], coverage["logical_requests"], coverage["input_tokens"],
            coverage["turns"], coverage["seconds"]) == (88.5, 12.0, 4000.0, 5.0, 61.5)
    assert summary["finishing"]["line_coverage"] == 90.0
    assert summary["starter"]["line_coverage"]["median"] == 55.5
    assert summary["starter"]["line_coverage"]["std"] == 3.25
    assert (summary["time"]["total"], summary["time"]["model_requests"]) == (120.0, 70.0)
    # Every name a summary uses has a label in plain words.
    named = {*run_summary.VALUES, *run_summary.STATISTICS, *run_summary.TIME, *run_summary.PHASE_VALUES,
             "finishing", *(name for columns in run_summary.PHASE_VALUES.values() for name in columns)}
    assert named <= set(run_summary.LABELS)


class _Row(dict):
    """A tests.csv row that records each column read from it."""

    def __init__(self):
        super().__init__()
        self.read = set()

    def __getitem__(self, column):
        self.read.add(column)
        return "0"

    def get(self, column, default=None):
        self.read.add(column)
        return "1"


def test_every_column_a_run_summary_reads_is_a_tests_csv_column():
    """The summary reads tests.csv's row by column name: each column it
    reads is one the writer writes, the generated final columns included."""
    from testuary import results as csv_writer

    row = _Row()
    run_summary.run_summary(row)
    written = set(csv_writer.TESTS_FIELDNAMES + csv_writer._finishing_columns()
                  + csv_writer._phase_finishing_columns())
    assert row.read and not row.read - written, sorted(row.read - written)


def test_a_cut_mutation_measurement_carries_its_lower_bound_up_to_the_project():
    """A run whose final mutation measurement left mutants untested has no
    score but a lower bound and its untested mutants. The file's summary
    states the lower bound across such runs, and the project's the lower
    bounds of the files without a score."""
    cut = run_summary.run_summary({
        "run": "0", "file_path": "calc.py", "line_coverage": "80",
        "finishing_mutation_score_lower_bound": "40.5", "finishing_untested": "22"})
    assert cut["mutation_score"] is None
    assert (cut["finishing"]["mutation_score_lower_bound"], cut["finishing"]["mutants_untested"]) == (40.5, 22.0)
    measured = run_summary.run_summary({"run": "1", "file_path": "calc.py", "mutation_score": "70",
                                        "finishing_mutation_score_lower_bound": "70"})
    summary = run_summary.file_summary("calc.py", [cut, measured])
    assert summary["mutation_score_lower_bound"] == {"runs": 1, "mean": 40.5, "minimum": 40.5,
                                                     "maximum": 40.5}
    only_cut = run_summary.file_summary("io.py", [cut])
    project = run_summary.project_summary([summary, only_cut])
    assert project["mutation_score_lower_bound"]["values"] == [40.5]
    assert project["values"]["mutation_score"]["count"] == 1


def test_each_phase_outcome_has_plain_words():
    """The outcomes the phases report and the phase records' states read
    as words, an unknown outcome as it is."""
    from testuary import phase_state
    from testuary.agent import mutation_agent

    described = mutation_agent.MutationTestResult.model_fields["status"].description
    listed = described.split("]", 1)[1].strip(" .").split(" | ")
    states = (phase_state.IN_PROGRESS, phase_state.COMPLETE, phase_state.FAILED)
    # The statuses a tests.csv row can hold besides the phases' own.
    written = ("skipped_below_minimum", "reuse_refused", "crashed")
    for status in (*listed, *states, *written, "ok"):
        assert status in run_summary.OUTCOMES, status
    assert run_summary.outcome_text("budget_exhausted") == (
        "request, token or time limit reached")
    # Passing suites the tools could not measure leave too, so a starter
    # phase without a suite says what was missing.
    assert run_summary.outcome_text("cancelled_zero_passing") == (
        "no starter suite passed with complete measurements")
    assert run_summary.outcome_text("something_new") == "something_new"
    assert run_summary.outcome_text("skipped_inherited_slow") == (
        "skipped: the suite it received runs longer than its time limit")
    # The outcomes tests.csv records besides the phases' own.
    for status in ("crashed", "inherited_not_portable", "reused", "reuse_refused", "unknown"):
        assert status in run_summary.OUTCOMES, status
    for kind in phase_state.FAILURE_KINDS:
        assert run_summary.failure_kind_text(kind) != kind


def test_the_project_summary_has_statistics_across_files_beside_the_starters():
    files = []
    for target_index, (line, starter) in enumerate((("80", "50"), ("90", "60"), ("", "40"))):
        run = run_summary.run_summary({**_row(0, line, "", "70"), "target_number": str(target_index),
                                       "starter_suite_line_median": starter})
        files.append(run_summary.file_summary(f"f{target_index}.py", [run]))
    project = run_summary.project_summary(files)
    line = project["values"]["line_coverage"]
    assert (line["count"], line["mean"], line["median"], line["minimum"], line["maximum"]) == (
        2, 85.0, 85.0, 80.0, 90.0)
    assert line["values"] == [80.0, 90.0] and line["std"] is not None
    assert project["starter_median"]["line_coverage"]["values"] == [50.0, 60.0, 40.0]
    assert project["values"]["branch_coverage"]["count"] == 0


def test_the_display_and_the_summary_name_each_phase_alike():
    from testuary.presentation.view import phase_label

    for phase in ("starter", "coverage", "mutation", "finishing"):
        assert run_summary.LABELS[phase] == phase_label(phase)

