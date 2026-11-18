"""Testuary's pipeline on a Java project end to end, with a scripted model
and real measurements (``java_workflow.py``), each run in a fresh process:
the starter suites merge into one suite, both agents edit, run and measure
through the Java language, and the delivered suite measures as recorded.
Needs a JDK 21 and Maven (``LANGADAPTER_TEST_JAVA_HOMES``,
``LANGADAPTER_TEST_MAVEN``)."""
from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import pytest

from java_fixtures import jdk_and_maven

pytestmark = [pytest.mark.slow, pytest.mark.mutation_tool]

DRIVER = Path(__file__).with_name("java_workflow.py")


def _run(tmp_path, variant):
    jdk, maven = jdk_and_maven()
    work = tmp_path / variant
    work.mkdir()
    done = subprocess.run(
        [sys.executable, str(DRIVER), str(work), str(jdk), maven, variant],
        capture_output=True, text=True, encoding="utf-8", errors="replace",
        timeout=3600)
    assert done.returncode == 0, done.stdout[-4000:] + done.stderr[-4000:]
    assert "STATUS 0" in done.stdout, done.stdout[-4000:]
    rows = list(csv.DictReader(open(work / "out" / "results" / "tests.csv",
                                    encoding="utf-8")))
    assert len(rows) == 1
    suite = next((work / "out" / "results").glob("*/r0/tests/CalcTest.java"))
    return rows[0], suite.read_text(encoding="utf-8"), done.stdout


def test_three_starters_merge_into_a_reduced_suite_that_kills_every_mutant(tmp_path):
    row, suite, _log = _run(tmp_path, "full")
    # The merge renamed one colliding test and localized one autouse fixture.
    # The record states it, and the console prints no line for the merge.
    record = json.loads(next((tmp_path / "full" / "out" / "results").glob(
        "*/r0/archive/starter/consolidation_record.json")).read_text(encoding="utf-8"))
    resolution = record["collision_resolution"]
    assert (resolution["renamed"], resolution["autouse_localized"]) == (1, 1)
    assert float(row["finishing_line_coverage"]) == 100.0
    assert float(row["finishing_branch_coverage"]) == 100.0
    assert float(row["finishing_mutation_score_full"]) == 100.0
    assert row["finishing_mutation_measurement"] == "complete"
    assert int(row["mutation_delivered_tests"]) == 4
    # Of two tests that kill the same mutants (``positive`` and
    # ``positive_2``, ``clampsHigh`` and ``clampsInside``) the phase end
    # credits both, and the cover keeps the one its measured timing makes
    # cheaper: either one, never both.
    for test in ("void negative()", "void zero()"):
        assert test in suite
    assert ("void positive()" in suite) != ("void positive_2()" in suite)
    assert ("void clampsHigh()" in suite) != ("void clampsInside()" in suite)
    assert "zeroIsPositive" not in suite


def test_both_agents_edit_run_and_measure_java_tests(tmp_path):
    row, suite, _log = _run(tmp_path, "weak")
    assert float(row["starter_original_mutation_full"]) == pytest.approx(400 / 7)
    assert float(row["coverage_original_line"]) == pytest.approx(500 / 6)
    assert float(row["finishing_line_coverage"]) == 100.0
    assert float(row["finishing_mutation_score_full"]) == 100.0
    # Each loop: read, the start's measurement, edit, run, measure, finish.
    assert (int(row["coverage_logical_requests"]),
            int(row["mutation_logical_requests"])) == (6, 6)
    assert "void clamps()" in suite and "void zero()" in suite


def test_a_ctrl_c_cancels_the_pit_measurement_and_the_run_resumes(tmp_path):
    """A Ctrl+C two seconds into the mutation agent's PIT measurement
    cancels it: the command ends as interrupted within seconds, no
    model request follows the measuring call, and a resume of the same
    workspace completes the run."""
    import json

    jdk, maven = jdk_and_maven()
    work = tmp_path / "cancel"
    work.mkdir()
    stopped = subprocess.run(
        [sys.executable, str(DRIVER), str(work), str(jdk), maven, "cancel"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    assert stopped.returncode == 130, stopped.stdout[-4000:] + stopped.stderr[-4000:]
    events = json.loads((work / "requests-cancel.json").read_text(encoding="utf-8"))
    kinds = [event["kind"] for event in events]
    assert "stop" in kinds and "cancel" in kinds, kinds
    stop = next(event["at"] for event in events if event["kind"] == "stop")
    end = next(event["at"] for event in events if event["kind"] == "end")
    assert end - stop < 15.0, end - stop
    mutation_turns = [event["turn"] for event in events
                      if event["kind"] == "agent" and event.get("phase") == "mutation"]
    # Read, the start's measurement (from the stored one), edit, run,
    # measure: the stop came during the measurement.
    assert mutation_turns == [1, 2, 3, 4, 5], mutation_turns
    resumed = subprocess.run(
        [sys.executable, str(DRIVER), str(work), str(jdk), maven, "resume"],
        capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=3600)
    assert resumed.returncode == 0, resumed.stdout[-4000:] + resumed.stderr[-4000:]
    assert "STATUS 0" in resumed.stdout, resumed.stdout[-4000:]
    rows = list(csv.DictReader(open(work / "out" / "results" / "tests.csv", encoding="utf-8")))
    assert len(rows) == 1 and rows[0]["finishing_mutation_measurement"] == "complete"
