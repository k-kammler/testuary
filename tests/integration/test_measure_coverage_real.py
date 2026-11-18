"""The coverage agent's ``measure_coverage`` tool on a real coverage run of
``tests/fixtures/tiny_project``.

The tool body runs ``coverage run`` and ``coverage json`` as subprocesses,
without mocks. Only the agent loop is bypassed. Unlike the unit tests, this
exercises coverage.py's command-line flags, the PYTHONPATH and working
directory, the coverage.json schema and the real line count of a small file.

Marked ``slow``, 5 to 15 seconds.
"""
from __future__ import annotations

import shutil
from pathlib import Path
import pytest

from agent_prompts import python_prompts

from testuary.agent import coverage_agent as tc


pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("default_run")]


def _stage_fixture_project(tmp_path: Path, fixtures_dir: Path) -> dict:
    """Copy the tiny fixture project into ``tmp_path`` and return the
    paths the agent expects. Each test gets its own copy, so coverage files
    of one test do not reach the next."""
    src_root = fixtures_dir / "tiny_project"
    work = tmp_path / "project"
    shutil.copytree(src_root, work)
    target = work / "src" / "arith.py"
    return {
        "source_dir": str(work),
        "target_file_path": str(target),
        "target_file_content": target.read_text(encoding="utf-8"),
        "test_output_dir": str(work / "tests"),
        "language": "python",
        "key_functions": ["add", "is_positive", "both_truthy"],
    }


def test_measure_coverage_produces_nonzero_score_on_real_subprocess(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """The fixture suite covers every line of ``arith.py``, so a real
    coverage run reports about 100 %, the session's measurement counter
    advances and the value lands in the recorded best.

    When the first measurement reaches goal, the helper sets
    ``_session.early_stop_result`` and returns a directive in the tool
    output, which ``run_target_phases`` picks up after ``agent.run_sync``
    returns.
    """
    paths = _stage_fixture_project(tmp_path, fixtures_dir)
    deps = tc.CoverageTestDeps(**paths)
    session = tc.TestCoverageSession()
    agent = tc.CoverageAgent(prompts=python_prompts("test_coverage"), deps=deps, session=session)

    output = agent.measure_coverage()
    # When the goal is met, the tool returns a directive containing
    # FRAMEWORK STOP and stashes the synthesized result on state.
    if session.early_stop_result is not None:
        assert session.early_stop_reason == "coverage_goal_met"
        assert "FRAMEWORK STOP" in output

    assert session.coverage_measurements == 1
    assert session.best_coverage_pct is not None
    assert session.best_coverage_pct >= 90.0, (
        f"fixture suite should fully cover arith.py; got "
        f"{session.best_coverage_pct}"
    )
