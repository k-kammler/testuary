"""The time limits a run reached: noted in its run directory when reached,
shown on the console as a sentence that names the setting, and listed as
warnings in the run's summary."""
from __future__ import annotations

import csv
import json
from pathlib import Path

from testuary import results as _time_limits
from testuary import results as run_summary

import test_test_coverage_measure as coverage_tools
import test_test_mutation_tools as mutation_tools
import pytest

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def test_a_noted_limit_is_one_warning_per_phase_and_setting(tmp_path):
    with _time_limits.collecting(tmp_path / "r0"):
        _time_limits.note("mutation", "mutation.measurement_timeout_seconds", 600)
        _time_limits.note("coverage", "coverage.suite_run_timeout_seconds", 300)
        _time_limits.note("mutation", "mutation.measurement_timeout_seconds", 600)
    assert _time_limits.warnings(tmp_path / "r0") == [
        {"kind": "time_limit", "phase": "mutation",
         "setting": "mutation.measurement_timeout_seconds", "value": 600, "unit": "s",
         "count": 2, "text": "The mutation measurement reached its time limit 2 times "
                             "(mutation.measurement_timeout_seconds = 600 s). "
                             "Untested mutants count as not killed."},
        {"kind": "time_limit", "phase": "coverage", "setting": "coverage.suite_run_timeout_seconds",
         "value": 300, "unit": "s", "count": 1,
         "text": "A test run reached its time limit (coverage.suite_run_timeout_seconds = 300 s)."},
    ]


def test_a_run_that_reached_no_limit_has_no_file_and_no_warning(tmp_path):
    with _time_limits.collecting(tmp_path / "r0"):
        pass
    assert not (tmp_path / "r0" / _time_limits.FILE).exists()
    assert _time_limits.warnings(tmp_path / "r0") == []
    # Outside a run the sentence is still given, and nothing is written.
    assert _time_limits.note("finishing", "finishing.coverage_timeout_seconds", 1800) == (
        "The final coverage run reached its time limit "
        "(finishing.coverage_timeout_seconds = 1800 s).")
    assert not list(tmp_path.rglob(_time_limits.FILE))


def test_the_run_summary_lists_the_runs_warnings(tmp_path):
    results = tmp_path / "results"
    results.mkdir()
    rows = [{"target_number": "3", "run": str(run), "file_path": "pkg/mod.py", "line_coverage": "80",
             "branch_coverage": "", "mutation_score": "50"} for run in (0, 1)]
    with open(results / "tests.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with _time_limits.collecting(results / "mod" / "r1"):
        _time_limits.note("finishing", "mutation.measurement_timeout_seconds", 900)
    across = run_summary.write_file_summaries(results, "mod", 3)
    first, second = (json.loads((results / "mod" / f"r{k}" / "summary.json").read_text(encoding="utf-8"))
                     for k in (0, 1))
    assert first["warnings"] == []
    [warning] = second["warnings"]
    assert (warning["phase"], warning["setting"], warning["value"], warning["count"]) == (
        "finishing", "mutation.measurement_timeout_seconds", 900, 1)
    assert across["warning_runs"] == 1
    assert run_summary.LABELS["time_limit"] == "Time limit reached"


def test_the_mutation_loop_notes_a_measurement_its_clock_cut(monkeypatch, tmp_path: Path):
    mutation_tools._install_backend(monkeypatch, mutation_tools._trr(0, "3 passed in 0.1s"))
    results = mutation_tools.tm.mutation.MutationMeasurement(
        score=100.0 * 8 / 15, killed=8, survived=2, timeout=0, non_viable=0,
        untested=5, total=10, truncated=True, lower_bound=True)
    mutation_tools._install_fake_engine(monkeypatch, results)
    agent = mutation_tools._make_agent(tmp_path)
    with _time_limits.collecting(tmp_path / "r0"):
        agent.measure_mutation()
    [warning] = _time_limits.warnings(tmp_path / "r0")
    assert (warning["phase"], warning["setting"]) == (
        "mutation", "mutation.measurement_timeout_seconds")
    assert warning["value"] == int(mutation_tools.tm._AGENT_CONFIG["measurement_timeout_seconds"])


def test_the_coverage_loop_notes_a_test_run_that_timed_out(monkeypatch, tmp_path: Path):
    coverage_tools._install_backend(
        monkeypatch, cov=coverage_tools._trr(ok=False, returncode=-1, timed_out=True))
    agent = coverage_tools._make_agent(tmp_path)
    monkeypatch.setattr(agent, "_ensure_tests_evaluated", lambda *, log_prefix: (True, "all passed"))
    with _time_limits.collecting(tmp_path / "r0"):
        agent.measure_coverage()
    [warning] = _time_limits.warnings(tmp_path / "r0")
    assert (warning["phase"], warning["setting"], warning["count"]) == (
        "coverage", "coverage.suite_run_timeout_seconds", 1)
