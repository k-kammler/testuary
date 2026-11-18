"""Real Cosmic Ray runs through Testuary's mutation runner.

The tiny fixture project with one local worker, and a sibling layout (the
tests outside the source directory) with the ``http`` distributor and two
workers, which clones the source per worker.

Marked ``slow``, 15 to 60 seconds each, and skipped without ``cosmic-ray``.
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


def _stage_fixture(tmp_path: Path, fixtures_dir: Path) -> Path:
    """Copy the tiny fixture project into ``tmp_path`` and return the
    source_dir path."""
    work = tmp_path / "project"
    shutil.copytree(fixtures_dir / "tiny_project", work)
    return work


def _run(config, source_dir, target, test_dir, workers, *, log):
    backend = languages.get_backend("python")
    # The sweep of abandoned clone directories stays off in tests.
    runner = backend.mutation_runner({**config, "cosmic_ray_clone_sweep_hours": 0},
                                     source_dir=str(source_dir), log=log)
    return runner.run_once(
        workers=workers, target=target, test_dir=test_dir,
        test_files=mutation.suite_test_files(backend, test_dir),
        per_mutant_timeout=30, wall_clock_timeout=120)


def test_cosmic_ray_single_worker_run_produces_score(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path, fixtures_dir: Path,
) -> None:
    """One local worker on the fixture project with the Wong-Mathur set:
    real mutants measured, the fixture suite (one assertion per mutable
    operator) kills at least 70 %, and the measurement holds the run its
    survivors' text and kill matrix are read from."""
    if not cosmic_ray_installed():
        pytest.skip("cosmic-ray not installed; integration test requires it")

    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setenv("PYTHONUTF8", "1")

    source_dir = _stage_fixture(tmp_path, fixtures_dir)
    log_messages: list[str] = []
    result = _run(
        {"cosmic_ray_operators": "wong_mathur_selective"}, source_dir,
        source_dir / "src" / "arith.py", source_dir / "tests", 1,
        log=lambda msg, level=0: log_messages.append(msg))

    assert result.killed + result.survived + result.timeout > 0, (
        f"cosmic-ray produced no measurable mutants. Result: {result}\n"
        "Log messages:\n" + "\n".join(log_messages)
    )
    assert result.score >= 70.0, (
        f"fixture suite should kill >= 70 % of mutants; got "
        f"score={result.score:.1f} (killed={result.killed}, "
        f"survived={result.survived}, total={result.total})"
    )
    assert result.run is not None and result.run.mutants
    assert result.kill_matrix()["mutants"]


def test_cosmic_ray_http_workers_kill_mutants_in_sibling_layout(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    """In the sibling layout (tests outside the source directory, as for
    sympy, simplejson or CPython) the HTTP workers must import their own
    clone's mutated source, not the live unmutated source. Otherwise every
    mutant survives although the suite pins the result.

    Two workers with the ``http`` distributor take the source-clone path
    (one worker is the local distributor and never clones). The suite is
    prepared with the production conftest, so imports resolve as in a real
    run."""
    if not cosmic_ray_installed():
        pytest.skip("cosmic-ray not installed; integration test requires it")

    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setenv("PYTHONUTF8", "1")

    work = tmp_path / "work"
    proj = work / "proj"
    test_dir = work / "tests"
    proj.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (proj / "arith.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8",
    )
    (test_dir / "test_arith.py").write_text(
        "from arith import add\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n    assert add(0, 0) == 0\n",
        encoding="utf-8",
    )
    languages.get_backend("python").setup_test_env(
        test_output_dir=str(test_dir), source_dir=str(proj),
        target_file_path=str(proj / "arith.py"),
    )

    log_messages: list[str] = []
    result = _run(
        {"cosmic_ray_operators": "wong_mathur_selective",
         "cosmic_ray_distributor": "http"},
        proj, proj / "arith.py", test_dir, 2,
        log=lambda msg, level=0: log_messages.append(msg))

    assert result.killed > 0, (
        f"sibling-layout HTTP workers killed no mutants (score "
        f"{result.score:.1f}); they are importing the live unmutated "
        f"source instead of their clone.\nLogs:\n"
        + "\n".join(log_messages)
    )
    assert result.score >= 70.0, (
        f"expected most mutants killed; got score={result.score:.1f} "
        f"(killed={result.killed}, survived={result.survived})"
    )
