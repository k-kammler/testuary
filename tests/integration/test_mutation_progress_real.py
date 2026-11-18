"""A real local Cosmic Ray run reports its progress to the display's operation layer.

The backend polls its session database about once a second,
MutationSession.run passes the observed target's callback, and RunProgress
turns each record into the display's "Mutation testing" row, which ends
with the run. Marked ``slow``; skipped without ``cosmic-ray``.
"""
from __future__ import annotations

import pytest

from test_mutation_seam_real import CONFIG, OPERATORS, _project
from tool_fixtures import cosmic_ray_installed

from testuary import languages
from testuary._progress import RunProgress, observe_target
from testuary.languages import mutation

pytestmark = [pytest.mark.slow, pytest.mark.mutation_tool]


def test_a_real_run_shows_its_mutants_and_ends_its_row(tmp_path, default_run):
    if not cosmic_ray_installed():
        pytest.skip("cosmic-ray not installed")
    root = _project(tmp_path)
    backend = languages.get_backend("python")
    runner = backend.mutation_runner(dict(CONFIG), source_dir=str(root))
    events = []
    progress = RunProgress(events.append)
    progress.emit(target="calc.py", phase="mutation", operation="phase/mutation", state="running",
                  level="phase", label="Mutation phase")
    with observe_target(progress, "calc.py"):
        measured = runner.run_once(
            workers=1, target=root / "calc.py", test_dir=root / "tests",
            test_files=mutation.suite_test_files(backend, root / "tests"),
            per_mutant_timeout=30, wall_clock_timeout=600, operators=list(OPERATORS))
    rows = [event for event in events if event.level == "operation"]
    assert measured.total == 4
    assert rows, "the run reported no progress"
    assert all(row.label == "Mutation testing" and row.phase == "mutation" for row in rows)
    assert {row.total for row in rows} <= {4, None}
    assert all(0 <= row.completed <= 4 for row in rows)
    assert [row.state for row in rows][-1] == "succeeded"
    assert all(row.state == "running" for row in rows[:-1])
