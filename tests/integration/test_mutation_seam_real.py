"""Testuary's mutation measurement through the backend's public session on
one real project, its suites without the instrument as Testuary runs them:
the counts, the in-loop score, both summary lines, the survivors' text the
model reads and the kill matrix. Mutant ids are run-local.

Marked ``slow``, and skipped without ``cosmic-ray``.
"""
from __future__ import annotations

import re
from pathlib import Path

import pytest

from tool_fixtures import cosmic_ray_installed

from testuary import languages
from testuary.languages import mutation
from testuary.agent import mutation_agent as tm

# Slow, and the code under test reads the run's settings: each test runs in one.
pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("default_run"), pytest.mark.mutation_tool]

TARGET = (
    "def add(a, b):\n"
    "    return a + b\n"
    "\n"
    "\n"
    "def is_small(n):\n"
    "    return n < 10\n"
    "\n"
    "\n"
    "def scale(n):\n"
    "    return n * 2 + 1\n"
)
TESTS = (
    "from calc import add, is_small, scale\n"
    "\n"
    "\n"
    "def test_add():\n"
    "    assert add(2, 3) == 5\n"
    "\n"
    "\n"
    "def test_calls():\n"
    "    is_small(3)\n"
    "    scale(1)\n"
)
OPERATORS = ["core/ReplaceBinaryOperator_Add_Sub",
             "core/ReplaceBinaryOperator_Mul_Add",
             "core/ReplaceComparisonOperator_Lt_LtE"]
#: The sweep of abandoned clone directories stays off: on a shared machine it
#: would remove other runs' directories from the system temporary directory.
CONFIG = {"cosmic_ray_operators": "", "cosmic_ray_catalogue_cache_dir": "",
          "cosmic_ray_distributor": "local", "engine": "cosmic_ray",
          "cosmic_ray_clone_sweep_hours": 0}
_ID = re.compile(r"\b[0-9a-f]{32}\b")

# One change shown for survivors on lines 6 and 10: the sample takes the
# middle of the lines spread over the places (spread_sample), line 10, and
# there the operator first by name.
SURVIVORS_TEXT = (
    "All 3 surviving mutants (id@line): <id>@L6, <id>@L10, <id>@L10\n\n"
    "[... 2 further mutant changes omitted, read_source around the lines "
    "listed above and strengthen those tests ...]"
    "\n\n"
    "--- mutant <id> (line 10) ---\n"
    "@@ -7,5 +7,5 @@\n"
    "-    return n * 2 + 1\n"
    "+    return n * 2 - 1"
)


def _project(tmp_path: Path) -> Path:
    root = tmp_path / "proj"
    (root / "tests").mkdir(parents=True)
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    test_file = root / "tests" / "test_calc.py"
    test_file.write_text(TESTS, encoding="utf-8")
    return root


def _rows(matrix):
    return sorted(
        (row.get("operator"), row.get("line"), row.get("status"),
         tuple(k.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
               for k in row.get("killers") or ()))
        for row in matrix["mutants"])


def test_the_seam_reads_a_real_run(tmp_path):
    if not cosmic_ray_installed():
        pytest.skip("cosmic-ray not installed")
    root = _project(tmp_path)
    backend = languages.get_backend("python")
    runner = backend.mutation_runner(dict(CONFIG), source_dir=str(root))
    measured = runner.run_once(
        workers=1, target=root / "calc.py", test_dir=root / "tests",
        test_files=mutation.suite_test_files(backend, root / "tests"),
        per_mutant_timeout=30, wall_clock_timeout=600,
        operators=list(OPERATORS))

    assert [measured.killed, measured.survived, measured.timeout,
            measured.non_viable, measured.untested, measured.total,
            measured.truncated] == [1, 3, 0, 0, 0, 4, False]
    assert measured.score == pytest.approx(25.0)
    assert tm._mutation_summary_line(measured) == (
        "Mutation score: 25.0% (killed 1/4; survived 3)")
    assert tm._canonical_mutation_summary_line(measured, 1.0) == (
        "MUTATION: SUCCESS | mutants 4 | killed 1 | survived 3 | "
        "timeout 0 | suspicious 0 | untested 0 | score 25.0% | "
        "elapsed 1.0s")
    assert _ID.sub("<id>", measured.survivors_text(max_mutants=1)) == \
        SURVIVORS_TEXT

    matrix = measured.kill_matrix()
    assert _rows(matrix) == [
        ("core/ReplaceBinaryOperator_Add_Sub", 2, "killed",
         ("test_calc.py::test_add",)),
        ("core/ReplaceBinaryOperator_Add_Sub", 10, "survived", ()),
        ("core/ReplaceBinaryOperator_Mul_Add", 10, "survived", ()),
        ("core/ReplaceComparisonOperator_Lt_LtE", 6, "survived", ()),
    ]
    assert matrix["summary"] == {
        "total": 4, "detected": 1, "survived": 3, "timeout": 0,
        "non_viable": 0, "untested": 0, "unresolved": 0}
    provenance = matrix["catalogue_provenance"]
    assert provenance["requested_catalogue"] == "filtered"
    assert provenance["requested_operators"] == sorted(OPERATORS)
    assert {"t_mono_s", "clock_domain_id"} <= set(matrix)


OTHER = (
    "from calc import is_small\n"
    "\n"
    "\n"
    "def test_small():\n"
    "    assert is_small(9)\n"
    "    assert not is_small(10)\n"
)


def test_per_suite_measures_each_suite_as_run_does(tmp_path):
    """Two suites, one in the project and one staged outside it as a
    starter suite is: each suite's matrix equals the run of that suite
    alone in the same session."""
    if not cosmic_ray_installed():
        pytest.skip("cosmic-ray not installed")
    root = _project(tmp_path)
    other = tmp_path / "staged"
    other.mkdir()
    (other / "test_other.py").write_text(OTHER, encoding="utf-8")
    backend = languages.get_backend("python")
    runner = backend.mutation_runner(dict(CONFIG), source_dir=str(root))
    suites = [("calc", root / "tests", ["test_calc.py"], 30),
              ("other", other, ["test_other.py"], 30)]
    measure = dict(target=root / "calc.py", wall_clock_timeout=600,
                   operators=list(OPERATORS))
    # The staged suite lies outside the project, so the session declares it.
    with runner.session(1, suite_dirs=(other,)) as session:
        result = session.per_suite(suites=suites, **measure)
        alone = {name: session.run(test_dir=test_dir, test_files=files,
                                   per_mutant_timeout=timeout, **measure)
                 for name, test_dir, files, timeout in suites}
        with pytest.raises(ValueError, match="same place in the project copy"):
            session.per_suite(suites=[suites[0], ("again", *suites[0][1:])],
                              **measure)

    assert result.outcome.value == "completed", result.reason
    assert result.implementation == "per_suite_runs"
    assert result.target == str(root / "calc.py")
    for name, measured in alone.items():
        assert _rows(result.kill_matrix(name)) == _rows(measured.kill_matrix())
    merged = {(row["operator"], row["line"]): row["status"]
             for row in result.kill_matrix()["mutants"]}
    assert merged == {
        ("core/ReplaceBinaryOperator_Add_Sub", 2): "killed",
        ("core/ReplaceBinaryOperator_Add_Sub", 10): "survived",
        ("core/ReplaceBinaryOperator_Mul_Add", 10): "survived",
        ("core/ReplaceComparisonOperator_Lt_LtE", 6): "killed",
    }
