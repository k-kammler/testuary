"""A reduced suite's mutation kills cannot be derived from the cover: a real
Cosmic Ray counterexample.

The cover keeps a recorded killer for every killed mutant, but not every test
a kill depends on. Here a preparatory test changes a setting of another
project module that the killing test needs (the instrument restores the
module under test before each test, but no other project module), and covers
nothing the killing test does not, so the cover removes it. The reduced
suite stays green on the original program and no longer kills the mutant.
Changed phase-end candidates are therefore measured, not credited by the
cover.

Marked slow, 20 to 60 seconds.
"""
from __future__ import annotations

import json

import pytest

from tool_fixtures import cosmic_ray_installed

from testuary import languages
from testuary.languages import mutation
from testuary.reduction import solve_cover

# Slow, and the code under test reads the run's settings: each test runs in one.
pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("default_run"), pytest.mark.mutation_tool]

#: ``x * settings.FACTOR`` made ``x ** settings.FACTOR`` gives 9 for 3 once
#: the first test has set FACTOR to 2, and 3 without it; the second test
#: accepts 3 or 6, so it passes on the original program either way and kills
#: the mutant only after the first test. FACTOR lives in another module,
#: which the instrument's per-test restore of the target leaves as it is.
SETTINGS = "FACTOR = 1\n"
TARGET = "import settings\n\n\ndef scale(x):\n    return x * settings.FACTOR\n"
SUITE = ("import settings\nimport stateful\n\n\n"
         "def test_a_prepares():\n"
         "    settings.FACTOR = 2\n"
         "    assert stateful.scale(0) == 0\n\n\n"
         "def test_b_checks():\n"
         "    assert stateful.scale(3) in (3, 6)\n")
MUL_POW = ["core/ReplaceBinaryOperator_Mul_Pow"]


def _measure(target, root, tests, matrix):
    """The engine's kills of the suite on disk, and the matrix's rows."""
    backend = languages.get_backend("python")
    # The sweep of abandoned clone directories stays off in tests.
    runner = backend.mutation_runner({"cosmic_ray_clone_sweep_hours": 0}, source_dir=str(root))
    with runner.session(1, suite_dirs=()) as session:
        measured = session.run(
            target=target, test_dir=tests,
            test_files=mutation.suite_test_files(backend, tests),
            per_mutant_timeout=30, wall_clock_timeout=300, operators=MUL_POW)
    assert mutation.write_kill_matrix(measured, matrix)
    rows = json.loads(matrix.read_text(encoding="utf-8"))["mutants"]
    return {row["id"] for row in rows if row["status"] == "killed"}, rows


def test_a_lossless_reduction_loses_a_kill_that_needs_a_removed_test(
        tmp_path, monkeypatch):
    if not cosmic_ray_installed():
        pytest.skip("cosmic-ray not installed; the test runs it")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setenv("PYTHONUTF8", "1")
    root = tmp_path / "proj"
    tests = root / "tests"
    tests.mkdir(parents=True)
    (root / "settings.py").write_text(SETTINGS, encoding="utf-8")
    target = root / "stateful.py"
    target.write_text(TARGET, encoding="utf-8")
    suite = tests / "test_stateful.py"
    suite.write_text(SUITE, encoding="utf-8")
    backend = languages.get_backend("python")
    backend.setup_test_env(str(tests), str(root),
                           target_file_path=str(target))

    # The full suite kills the mutant; the second test is its killer.
    full_matrix = tmp_path / "full_matrix.json"
    full, rows = _measure(target, root, tests, full_matrix)
    assert len(rows) == 1 and len(full) == 1, rows
    assert rows[0]["killer"].endswith("test_b_checks"), rows

    # Select a cover explicitly: token preservation does not establish kills.
    evidence = backend.collect_eviction_data(
        [str(suite)], str(root), measure_coverage_of=str(target), timeout=120,
        kill_matrix_path=str(full_matrix),
    )
    assert evidence is not None and evidence.includes_kills
    cover = solve_cover(evidence.per_test_tokens, {t: 1.0 for t in evidence.per_test_tokens})
    assert len(cover.remove) == 1
    reduced = backend.finalize_test_source(backend.prune_tests(SUITE, cover.remove))
    assert backend.validate_test_source(reduced, str(root)).ok
    suite.write_text(reduced, encoding="utf-8")
    assert backend.run_test_files(
        [str(suite)], str(root), timeout=120, instrument_target=str(target),
    ).ok
    reduced_source = suite.read_text(encoding="utf-8")
    assert "def test_a_prepares" not in reduced_source
    assert "def test_b_checks" in reduced_source

    # The reduced suite's actual measurement loses the kill despite the cover.
    kept, _ = _measure(target, root, tests, tmp_path / "reduced_matrix.json")
    assert kept == set()
