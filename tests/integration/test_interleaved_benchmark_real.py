"""Real-subprocess check of the interleaved timing of tied suites.

Two copies of the tiny fixture's suite are timed round by round through
pytest itself, and a red suite leaves the rounds while the other completes
them. Marked ``slow``.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from testuary.languages import GenericBackend
from testuary import consolidation as consolidate


pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("default_run")]


def _stage(tmp_path: Path, fixtures_dir: Path):
    work = tmp_path / "project"
    shutil.copytree(fixtures_dir / "tiny_project", work)
    return work, work / "src" / "arith.py", work / "tests" / "test_arith.py"


def test_tied_suites_are_timed_round_by_round(tmp_path, fixtures_dir):
    work, target, test = _stage(tmp_path, fixtures_dir)
    # A staged candidate runs outside the fixture's tests directory, so it
    # finds the target itself, as a generated suite does.
    future = "from __future__ import annotations\n"
    source = test.read_text(encoding="utf-8").replace(
        future, future + f"import sys\nsys.path.insert(0, {str(target.parent)!r})\n", 1)
    red = source + "\n\ndef test_red():\n    assert False\n"
    candidates = [
        consolidate.ScoredSuite(label=label, source=text, line=50.0, branch=50.0,
                                mutation=50.0, green=True, test_count=1, runtime_ms=1.0)
        for label, text in (("baseline", source), ("agent", source), ("red", red))
    ]
    timer = consolidate.TiedCandidateTimer(
        GenericBackend("python", (".py",)), source_dir=str(work),
        target_file_path=str(target), timeout=120, warmup_runs=1, benchmark_runs=3,
        stage_as_delivered=False, log=lambda message: None)

    winner = consolidate.pick_phase_result(candidates, runtime_of=timer, required_axes=consolidate._PHASE_END_AXES)

    assert winner.label == "baseline"
    assert candidates[2].green is True
    assert candidates[2].suite_timing["status"] == "failed"
    for candidate in candidates[:2]:
        record = candidate.suite_timing
        assert len(record["samples_ms"]) == 3
        assert all(sample > 0 for sample in record["samples_ms"])
        assert [item["order"][0] for item in record["rounds"]] == [
            "baseline", "agent", "baseline", "agent"]
    # The red suite ran once, in the warm-up round, and left the rounds.
    assert candidates[0].suite_timing["rounds"][0]["order"] == [
        "baseline", "agent", "red"]
    assert all("red" not in item["order"]
               for item in candidates[0].suite_timing["rounds"][1:])
