"""The kill matrix of a real Cosmic Ray run through Testuary's mutation
runner.

Runs an actual mutation pass against ``tests/fixtures/tiny_project`` and
asserts that the measurement's kill matrix records a killing test per killed
mutant: the attribution the suite-time reducer needs, read from the real
Cosmic Ray session schema.

Marked ``slow``: a real cosmic-ray run on three tiny functions is a few seconds.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from tool_fixtures import cosmic_ray_installed

from testuary import languages
from testuary.languages import mutation

# Slow, and the code under test reads the run's settings: each test runs in one.
pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("default_run"), pytest.mark.mutation_tool]


def test_cosmic_ray_capture_writes_killers_against_real_schema(
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    if not cosmic_ray_installed():
        pytest.skip("cosmic-ray not installed; integration test requires it")
    work = tmp_path / "project"
    shutil.copytree(fixtures_dir / "tiny_project", work)
    target = work / "src" / "arith.py"
    test_dir = work / "tests"

    backend = languages.get_backend("python")
    runner = backend.mutation_runner(
        {"engine": "cosmic_ray", "cosmic_ray_operators": "",
         "cosmic_ray_clone_sweep_hours": 0},
        source_dir=str(work))
    results = runner.run_once(
        workers=1, target=target, test_dir=test_dir,
        test_files=mutation.suite_test_files(backend, test_dir),
        per_mutant_timeout=30, wall_clock_timeout=300)
    out = test_dir / ".evict_kill_matrix.json"
    assert mutation.write_kill_matrix(results, out)

    assert results.killed > 0, "fixture suite should kill several mutants"
    data = results.kill_matrix()
    killed = [m for m in data["mutants"] if m["status"] == "killed"]
    assert killed, "captured matrix must record killed mutants"
    # Every killed mutant names a pytest node id as its killer. Complete
    # attribution sets includes_kills, which the mutation set cover requires.
    attributed = [m for m in killed if m.get("killer")]
    assert len(attributed) == len(killed), (
        f"all killed mutants must be attributed to a test; "
        f"{len(killed) - len(attributed)} unattributed"
    )
    assert all("::" in m["killer"] for m in attributed), (
        "killer must be a pytest node id (path::test)"
    )
