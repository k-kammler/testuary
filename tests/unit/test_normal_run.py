"""The normal test run (normal_run): a language with the measurement
instrument runs as delivered without it, with the fixed session seed, and
one without the instrument through run_test_files. No run places the
instrument."""
from __future__ import annotations

import random
from pathlib import Path
from types import SimpleNamespace

import pytest

from testuary import languages, normal_run
from testuary.consolidation import _check_then_measure, red_verdict
from testuary.languages import DeliveredRun
from testuary.languages.generic import DeliveredCase

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

TARGET = "def add(a, b):\n    return a + b\n"

#: Passes when something seeds random with 0 before the test, as the
#: instrument did, and fails under the run as delivered's session seed (1).
SEEDED = (
    "import random\n"
    "from calc import add\n\n"
    "def test_the_instrument_seeds_random():\n"
    f"    assert random.random() == {random.Random(0).random()!r}\n"
    "    assert add(1, 1) == 2\n"
)


def _project(tmp_path: Path, test_source: str):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    test = tests / "test_calc.py"
    test.write_text(test_source, encoding="utf-8")
    return languages.get_backend("python"), root, test


def test_the_normal_runs_session_seed_is_1():
    """The normal run gives every run the session seed 1."""
    assert normal_run.SESSION_SEED == 1


class _Backend:
    def __init__(self, delivers: bool, seeds: bool, instrument: bool = False):
        self.calls = []
        self._delivers, self._seeds, self._instrument = delivers, seeds, instrument

    def runs_as_delivered(self, source_dir=None):
        return self._delivers

    def seeds_delivered_runs(self, source_dir=None):
        return self._seeds

    def has_measurement_instrument(self, source_dir=None):
        return self._instrument

    def failing_test_ids(self, paths, source_dir, **kwargs):
        self.calls.append(("failing_test_ids", tuple(paths), kwargs))
        return []

    def run_test_files(self, paths, source_dir, **kwargs):
        self.calls.append(("run_test_files", tuple(paths), kwargs))
        return SimpleNamespace(ok=True)

    def run_as_delivered(self, paths, source_dir, **kwargs):
        self.calls.append(("run_as_delivered", tuple(paths), kwargs))
        return DeliveredRun(cases=(DeliveredCase("t.test_a", "test_a", "passed"),
                                   DeliveredCase("t.test_b", "test_b", "failed", "boom")),
                            failing=("t.test_b",), reason=None, output="1 failed",
                            timed_out=False, elapsed_s=0.4)


#: Whether the language runs suites as delivered, whether those runs take a
#: session seed, whether it has the determinism instrument, and its normal run.
LANGUAGES = [
    (True, True, False, "run_as_delivered"),
    (True, True, True, "run_as_delivered"),
    (True, False, True, "run_test_files"),
    (False, True, True, "run_test_files"),
    (False, False, False, "run_test_files"),
]


@pytest.mark.parametrize("delivers, seeds, instrument, method", LANGUAGES)
def test_the_normal_run_follows_the_language(delivers, seeds, instrument, method):
    """A language whose runs as delivered take a session seed runs as
    delivered with the fixed seed. Any other runs the backend's
    run_test_files with the module under test. The instrument decides
    nothing."""
    backend = _Backend(delivers, seeds, instrument)
    run = normal_run.run(backend, ["t.py"], "root", timeout=7, target="calc.py", fail_fast=True)
    (name, paths, kwargs), = backend.calls
    assert (name, paths) == (method, ("t.py",))
    if method == "run_test_files":
        assert kwargs["instrument_target"] == "calc.py" and kwargs["fail_fast"] is True
    else:
        assert kwargs["seed"] == 1 and kwargs["timeout"] == 7
        # The module under test makes a prepared copy's tests import the copy.
        assert kwargs["target"] == "calc.py"
        assert (run.ok, run.returncode, run.collected, run.passed, run.failed) == (False, 1, 2, 1, 1)
        assert [(f.test, f.message) for f in run.failures] == [("t.test_b", "boom")]
        assert run.elapsed_s == 0.4 and run.line_coverage is None


@pytest.mark.parametrize("language, delivered", [("python", True), ("java", False), ("c", False)])
def test_each_languages_normal_run_stays_as_it_was(language, delivered, tmp_path):
    """Python's runs as delivered take the session seed, so its normal run
    is the run as delivered. Java's and C's take none, so theirs is the
    language's own test run."""
    backend = languages.get_backend(language)
    assert backend is not None
    assert normal_run._through_run_test_files(backend, str(tmp_path)) is not delivered


def test_a_delivered_run_without_test_outcomes_reads_as_no_outcome():
    run = normal_run._result_of(DeliveredRun(cases=(), failing=None, reason="no tests collected",
                                             output="", timed_out=True))
    assert (run.ok, run.returncode, run.timed_out, run.output) == (False, -1, True, "no tests collected")


#: A src-layout project's own pytest configuration: a registered marker,
#: and warnings as errors.
SRC_LAYOUT_CONFIG = (
    "[tool.pytest.ini_options]\n"
    'markers = ["unit: a unit test"]\n'
    'filterwarnings = ["error"]\n'
)
#: Passes only where the project's configuration applies: its marker is
#: registered and its warnings filter is in force.
SRC_LAYOUT_TEST = (
    "import pytest\n"
    "from calc import add\n\n"
    "@pytest.mark.unit\n"
    "def test_the_project_configuration_applies(pytestconfig):\n"
    '    assert any(line.startswith("unit:") for line in pytestconfig.getini("markers"))\n'
    '    assert "error" in pytestconfig.getini("filterwarnings")\n'
    "    assert add(1, 1) == 2\n"
)


def test_a_src_layout_projects_normal_run_applies_its_configuration(tmp_path, in_run):
    """A pure-Python project in a src layout ([project] python_import_root =
    src): its normal run runs in the project directory, where its pytest
    configuration lies, and imports the package from src."""
    from testuary import project

    root = tmp_path / "proj"
    (root / "src" / "calc").mkdir(parents=True)
    (root / "src" / "calc" / "__init__.py").write_text(TARGET, encoding="utf-8")
    (root / "pyproject.toml").write_text(SRC_LAYOUT_CONFIG, encoding="utf-8")
    (root / "tests").mkdir()
    test = root / "tests" / "test_calc.py"
    test.write_text(SRC_LAYOUT_TEST, encoding="utf-8")
    in_run(overrides={"api": {"provider": "ollama", "model": "test-model"},
                      "project": {"python_import_root": "src"}})
    target = str(root / "src" / "calc" / "__init__.py")
    source_dir = project.resolve_source_dir(target)
    assert Path(source_dir) == root.resolve()
    assert project.display_path(target, source_dir) == "calc/__init__.py"
    run = normal_run.run(languages.get_backend("python"), [test], source_dir, timeout=120,
                         target=target)
    assert run.ok, run.output


def test_a_test_that_needs_the_instrument_fails_its_normal_run(tmp_path):
    """Through Python's real backend: the run as delivered seeds random with
    the session seed, so a test that needs the instrument's seed fails, and
    no instrument is placed beside the suite."""
    backend, root, test = _project(tmp_path, SEEDED)
    run = normal_run.run(backend, [test], str(root), timeout=120, target=str(root / "calc.py"))
    assert run.ok is False
    assert not (test.parent / "conftest.py").is_file()


#: A passing test, a failing test and a parametrized test with one
#: failing case.
NAMED = (
    "import pytest\n"
    "from calc import add\n\n"
    "def test_passes():\n"
    "    assert add(1, 1) == 2\n\n"
    "def test_fails():\n"
    "    assert add(1, 1) == 3\n\n"
    "@pytest.mark.parametrize('a', [1, 2])\n"
    "def test_case(a):\n"
    "    assert add(a, 0) == 1\n"
)


def test_pythons_normal_run_names_its_failing_tests_by_their_id_objects(tmp_path):
    """Through Python's real backend: the normal run's result names the
    failed tests as the run's id objects, a parametrized case alone, as a
    measurement's removal reads them."""
    backend, root, test = _project(tmp_path, NAMED)
    run = normal_run.run(backend, [test], str(root), timeout=120, target=str(root / "calc.py"))
    assert run.ok is False
    assert sorted((named.qualname, named.params) for named in run.stopping_tests) == [
        ("test_case", "2"), ("test_fails", None)]


#: Rebinds a name of the module under test without restoring it.
REBINDS = (
    "import calc\n\n"
    "def test_rebinds():\n"
    "    calc.add = lambda a, b: 0\n"
    "    assert calc.add(1, 1) == 0\n\n"
    "def test_needs_the_module():\n"
    "    assert calc.add(1, 1) == 2\n"
)


def test_no_run_restores_the_module_under_test(tmp_path):
    """Without the instrument nothing restores the module under test
    between tests: a test that needs it as imported, after one that rebinds
    a name of it, fails the normal run and the coverage tool's run alike."""
    backend, root, test = _project(tmp_path, REBINDS)
    target = str(root / "calc.py")
    normal = normal_run.run(backend, [test], str(root), timeout=120, target=target)
    measured = backend.run_test_files([str(test)], str(root), timeout=120,
                                      measure_coverage_of=target)
    for run in (normal, measured):
        assert [f.test.rsplit(".", 1)[-1] for f in run.failures] == [
            "test_needs_the_module"], run.output


def test_a_prepared_suite_dir_carries_no_instrument(tmp_path):
    backend, root, _test = _project(tmp_path, "def test_x():\n    assert True\n")
    out = tmp_path / "out"
    backend.setup_test_env(str(out), str(root), str(root / "calc.py"))
    # Only LangAdapter's private marker tells its instrument from other conftest text.
    from langadapter.python.project import _INSTRUMENT_MARKER

    conftest = out / "conftest.py"
    assert not conftest.is_file() or _INSTRUMENT_MARKER not in conftest.read_text(encoding="utf-8")


class _MeasuredBackend(_Backend):
    """Passes its normal run, and test_traced fails under the coverage tool."""

    def run_test_files(self, paths, source_dir, **kwargs):
        self.calls.append(("run_test_files", tuple(paths), kwargs))
        if kwargs.get("measure_coverage_of") is None:
            return SimpleNamespace(ok=True)
        return SimpleNamespace(ok=False, returncode=1,
                               failures=(SimpleNamespace(test="t.test_traced"),))

    def run_as_delivered(self, paths, source_dir, **kwargs):
        self.calls.append(("run_as_delivered", tuple(paths), kwargs))
        return DeliveredRun(cases=(DeliveredCase("t.test_traced", "test_traced", "passed"),),
                            failing=(), reason=None, output="1 passed")


def test_a_suite_failing_only_under_the_coverage_tool_is_left_without_numbers():
    """A phase end's check: the normal run decides, and only a suite it
    passes is measured, by the coverage tool's run. A test that fails only
    under the tool leaves the suite without numbers, named."""
    backend = _MeasuredBackend(delivers=True, seeds=True)
    logged = []
    run, passed = _check_then_measure(backend, ["t.py"], "root", timeout=7,
                                      measure_coverage_of="calc.py", log=logged.append)
    assert passed
    assert [name for name, _, _ in backend.calls] == ["run_as_delivered", "run_test_files"]
    assert backend.calls[-1][2]["measure_coverage_of"] == "calc.py"
    assert not run.ok and "t.test_traced" in red_verdict(run)
    assert any("coverage tool" in line and "t.test_traced" in line for line in logged)


def test_a_suite_failing_its_normal_run_is_not_measured():
    backend = _Backend(delivers=True, seeds=True)
    run, passed = _check_then_measure(backend, ["t.py"], "root", timeout=7,
                                      measure_coverage_of="calc.py", log=print)
    assert not passed
    assert [name for name, _, _ in backend.calls] == ["run_as_delivered"]
    assert not run.ok


@pytest.mark.parametrize("delivers, seeds, instrument, method", LANGUAGES)
def test_the_failing_listing_is_the_normal_runs(delivers, seeds, instrument, method):
    """The cleanup's listing of failing tests is the normal run's: as
    delivered with the session seed for a language whose runs as delivered
    take it, else run_test_files', which places nothing."""
    backend = _Backend(delivers, seeds, instrument)
    report = {}
    assert normal_run.failing_ids(backend, ["t.py"], "root", timeout=7,
                                  target="calc.py", report=report) == []
    (name, paths, kwargs), = backend.calls
    assert (name, paths) == ("failing_test_ids", ("t.py",))
    if method == "run_as_delivered":
        assert kwargs == {"timeout": 7, "instrument_target": "calc.py", "report": report,
                          "delivered_seed": 1}
    else:
        assert kwargs == {"timeout": 7, "instrument_target": "calc.py", "report": report}


def test_the_failing_listing_is_the_normal_runs_through_the_backend(tmp_path):
    """Through Python's real backend: a test that needs the instrument's
    seed is listed failing."""
    backend, root, test = _project(tmp_path, SEEDED)
    report = {}
    ids = normal_run.failing_ids(backend, [test], str(root), timeout=120,
                                 target=str(root / "calc.py"), report=report)
    assert ids == ["test_calc.test_the_instrument_seeds_random"], report


def test_the_mutation_green_check_is_the_normal_run():
    """The un-mutated suite's green check before a mutation measurement is
    the normal run, whose time sizes the per-mutant timeout."""
    from testuary import measurement

    backend = _Backend(delivers=True, seeds=True)
    measurement._green_check_with_retry(backend, "d", "root", 10, lambda m: None,
                                        test_paths=["t.py"], instrument_target="calc.py")
    assert [name for name, _, _ in backend.calls] == ["run_as_delivered"]


def test_a_several_file_mutation_green_check_runs_them_as_the_tool_does():
    """The mutation tool runs a directory of several files in one session, so
    its green check, whose time sizes the per-mutant timeout, runs them in
    one session too. A one-file suite's check stays the normal run."""
    from testuary import measurement

    backend = _Backend(delivers=True, seeds=True)
    measurement._green_check_with_retry(backend, "d", "root", 10, lambda m: None,
                                        test_paths=["a.py", "b.py"], instrument_target="calc.py")
    (name, paths, kwargs), = backend.calls
    assert (name, paths) == ("run_test_files", ("a.py", "b.py"))
    assert kwargs["instrument_target"] == "calc.py" and kwargs["timeout"] == 10


def test_the_load_probe_lists_the_normal_runs_failing_tests(tmp_path):
    """The load recoveries probe a candidate alone in a fresh directory: the
    listing is the normal run's."""
    from testuary import loadable

    backend, root, _test = _project(tmp_path, SEEDED)
    probe = loadable.scratch_probe(backend, str(root), str(root / "calc.py"), timeout=120)
    ids, timed_out, outcome = probe(SEEDED)
    assert timed_out is False
    # The probe stages the candidate under its own scratch name.
    assert [test.rsplit(".", 1)[-1] for test in ids] == [
        "test_the_instrument_seeds_random"], (ids, outcome)
