"""With the real Python backend, whose normal run is the run as delivered
without the instrument: a phase end's normal run, which removes nothing and
leaves out a suite that fails it, and the starter-suite cleanup, which
prunes the tests its normal run fails and looks for no culprit."""
from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from testuary import languages
from testuary import config as config_module
from testuary import project as project_module
from testuary import starter_salvage
from testuary import normal_run
from testuary.consolidation import (
    check_restored_suite, merged_scored_from_phase,
    phase_end_candidates, pick_phase_result, _check_then_measure, _PHASE_END_AXES,
)
from per_suite_double import killing_tests
from phase_end_calls import phase_end

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

TARGET = ('class Box:\n    pass\n\n\n'
          'def make_box():\n    return Box()\n\n\n'
          'def add(a, b):\n    return a + b\n')

RELOAD = '''import importlib

import calc


def test_reload():
    importlib.reload(calc)
    assert calc.add(1, 2) == 3
'''
VICTIM = '''import calc
from calc import Box


def test_victim():
    assert isinstance(calc.make_box(), Box)
'''
PLAIN = '''from calc import add


def test_add():
    assert add(2, 3) == 5
'''
SEED_ZERO = '''import random


def test_seed_zero():
    assert random.random() == 0.8444218515250481
'''
#: Collected under the instrument's hash seed only.
HASH_SEED = '''import os

assert os.environ["PYTHONHASHSEED"] == "0"


def test_hash_seed():
    assert True
'''
CULPRIT_AND_VICTIM = '''import importlib

import calc
from calc import Box


def test_reload():
    importlib.reload(calc)
    assert calc.add(1, 2) == 3


def test_victim():
    assert isinstance(calc.make_box(), Box)
'''
#: Two reloads, either of which breaks the victim as delivered.
TWO_RELOADS = '''import importlib

import calc
from calc import Box


def test_reload_first():
    importlib.reload(calc)
    assert calc.add(1, 2) == 3


def test_reload_second():
    importlib.reload(calc)
    assert calc.add(2, 2) == 4


def test_victim():
    assert isinstance(calc.make_box(), Box)
'''
#: A reload before a test that fails as delivered for another reason.
RELOAD_AND_SEED = '''import importlib
import random

import calc


def test_reload():
    importlib.reload(calc)
    assert calc.add(1, 2) == 3


def test_seed_zero():
    assert random.random() == 0.8444218515250481
'''


def _project(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    return languages.get_backend("python"), str(root / "calc.py"), str(root)


def _staged(tmp_path: Path, text: str) -> Path:
    tests = tmp_path / "tests"
    tests.mkdir(exist_ok=True)
    path = tests / "test_suite.py"
    path.write_text(text, encoding="utf-8")
    return path


def _cleaned(tmp_path, source):
    """The starter-suite cleanup of ``source``: its class and cleaned source."""
    backend, target, root = _project(tmp_path)
    return starter_salvage._clean_suite(
        backend, source, target, root, tmp_path / "scratch", timeout=120,
        log=lambda m: None, cleanup_out={})


def _phase_end(backend, candidates, target, root):
    return phase_end(
        backend, candidates, target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=None, log=lambda m: None,
    )


# The consolidation.

def test_a_merge_is_decided_by_the_normal_run_and_keeps_its_tests(tmp_path):
    """A merge whose test changes the module for a later one fails its
    normal run: it is no candidate, its failing test is named, and nothing
    is removed."""
    backend, target, root = _project(tmp_path)
    res = _phase_end(backend, [RELOAD, VICTIM], target, root)
    assert "def test_reload" in res.merged_source
    assert "def test_victim" in res.merged_source
    assert res.merged_green is False, res.detail
    assert "does not pass" in res.detail and "test_victim" in res.detail


def test_a_leak_that_breaks_nothing_stays(tmp_path):
    backend, target, root = _project(tmp_path)
    res = _phase_end(backend, [RELOAD, PLAIN], target, root)
    assert res.merged_green is True, res.detail
    assert "def test_reload" in res.merged_source


def test_no_second_run_judges_a_phase_end_suite(tmp_path):
    """The normal run, the run as delivered, is the only verdict. No run
    records leaks."""
    backend, target, root = _project(tmp_path)
    calls = {"delivered": 0, "record_leaks": 0}

    class Spy(type(backend)):
        def run_as_delivered(self, *args, **kwargs):
            calls["delivered"] += 1
            return super().run_as_delivered(*args, **kwargs)

        def run_test_files(self, *args, **kwargs):
            if kwargs.get("record_leaks"):
                calls["record_leaks"] += 1
            return super().run_test_files(*args, **kwargs)

    res = _phase_end(Spy(backend.name, backend.file_extensions), [RELOAD, PLAIN], target, root)
    assert calls["record_leaks"] == 0
    assert calls["delivered"] > 0
    assert "def test_reload" in res.merged_source


def test_a_merge_failing_its_normal_run_is_not_recorded_green(tmp_path):
    """The merge's only test needs the instrument's random seed, so it fails
    the normal run with the session seed. The phase keeps its source for the
    record, but neither it nor its scored suite reads as green."""
    backend, target, root = _project(tmp_path)
    res = _phase_end(backend, [SEED_ZERO], target, root)
    assert res.merged_source and res.merged_green is False, res.detail
    assert "does not pass" in res.detail
    scored = merged_scored_from_phase(res)
    # Its normal run fails, so it is failing, with the reason.
    assert not scored.green and scored.status == "failing"
    assert scored.status_reason == res.detail


# A suite restored without its phase.

def _restored(tmp_path, source):
    tmp_path.mkdir(parents=True)
    backend, target, root = _project(tmp_path)
    return check_restored_suite(
        backend, source, target_file_path=target, source_dir=root,
        timeout=120, log=lambda m: None)


def test_a_restored_suite_is_decided_by_its_normal_run(tmp_path):
    """The restored suite runs once, its normal run. Nothing is pruned, and
    one that does not pass refuses the reuse with its failing test named."""
    assert _restored(tmp_path / "plain", PLAIN) == (True, "")
    accepted, reason = _restored(tmp_path / "seed", SEED_ZERO)
    assert accepted is False, reason
    assert "does not pass its normal run" in reason
    assert "test_seed_zero" in reason


# _check_then_measure on its own.

@pytest.mark.parametrize("source", [HASH_SEED, SEED_ZERO], ids=["hash-seed", "random-seed"])
def test_a_suite_needing_the_instrument_fails_its_normal_run(tmp_path, source):
    """A file collected only under the instrument's hash seed, and a test
    that needs its random seed, fail their normal run. Nothing is removed."""
    backend, target, root = _project(tmp_path)
    path = _staged(tmp_path, source)
    run, passed = _check_then_measure(backend, [path], root, timeout=120,
                                      measure_coverage_of=target, log=lambda m: None)
    assert run.ok is False and passed is False
    assert path.read_text(encoding="utf-8") == source


# The reduced suite.

#: A module setting that test A changes, test B happens to reset and test C
#: needs at its default.
STATE_TARGET = ('FACTOR = 1\n\n\n'
                'def scale(x):\n    return x * FACTOR\n\n\n'
                'def other(x):\n    return x + 1\n')
STATE_SUITE = '''import calc


def test_a_other():
    assert calc.other(1) == 2
    calc.FACTOR = 3


def test_b_reset():
    calc.FACTOR = 1


def test_c_scale():
    assert calc.scale(2) == 2
'''
#: A fixed catalogue: A kills one mutant, C the other, B none, so the
#: reduction keeps A and C and drops B.
_STATE_KILLERS = {"m_other": "test_a_other", "m_scale": "test_c_scale"}


def _state_runner(directory):
    pyfile = next(Path(directory).glob("test_*.py"))
    present = set(re.findall(r"def (test_\w+)",
                             pyfile.read_text(encoding="utf-8")))
    mutants = []
    for mutant, killer in _STATE_KILLERS.items():
        if killer in present:
            name = f"{pyfile.name}::{killer}"
            mutants.append({"id": mutant, "status": "killed",
                            "killer": name, "killers": [name],
                            "killing_tests": killing_tests([name]), "line": None})
        else:
            mutants.append({"id": mutant, "status": "survived",
                            "killer": None, "killers": [], "line": None})
    return {"mutants": mutants, "summary": {}}


def _state_phase_end(tmp_path):
    root = tmp_path / "proj"
    root.mkdir(parents=True)
    (root / "calc.py").write_text(STATE_TARGET, encoding="utf-8")
    backend = languages.get_backend("python")
    res = phase_end(
        backend, [STATE_SUITE], target_file_path=str(root / "calc.py"),
        source_dir=str(root), timeout=120, mutation_runner=_state_runner,
        mutation_method="both", stage_as_delivered=True,
        log=lambda m: None,
    )
    winner = pick_phase_result(phase_end_candidates(res), log=lambda m: None, required_axes=_PHASE_END_AXES)
    return backend, str(root), res, winner


def _delivered_run(backend, directory: Path, source: str, root: str):
    directory.mkdir(parents=True)
    path = directory / "test_delivered.py"
    path.write_text(source, encoding="utf-8")
    return backend.run_as_delivered([str(path)], root, timeout=120)


def test_a_reduced_suite_is_decided_by_its_normal_run_before_selection(tmp_path):
    """The reduction drops the test that resets the module's setting, so the
    reduced suite fails its normal run: it is no candidate, nothing is
    removed from it, and the phase end delivers the merged suite, which
    passes."""
    backend, root, res, winner = _state_phase_end(tmp_path)
    assert "def test_b_reset" not in res.reduced_source
    assert res.reduction_valid is False
    assert res.selection_audit["reduction_measurement_issue"] == "reduced_failing"
    assert winner.label == "merged" and "def test_b_reset" in winner.source
    assert _delivered_run(backend, tmp_path / "delivered", winner.source, root).ok


def test_a_passing_reduced_suite_is_run_once_and_keeps_its_measurement(
        tmp_path, monkeypatch):
    """A reduced suite that passes its normal run on its final bytes keeps
    its evidence. A reduction that keeps every test is the merged suite,
    which passed, and is not run again."""
    backend, target, root = _project(tmp_path)
    reduced_runs = []
    real = normal_run.run

    def counting(backend, paths, *args, **kwargs):
        if Path(paths[0]).parent.parent.name.startswith("testuary_phaseend_reduced_"):
            reduced_runs.append(paths)
        return real(backend, paths, *args, **kwargs)

    monkeypatch.setattr(normal_run, "run", counting)

    def runner(directory):
        return {"mutants": [], "summary": {}}

    res = phase_end(
        backend, [PLAIN, PLAIN.replace("test_add", "test_add_again")],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=runner, mutation_method="measure", log=lambda m: None)
    assert res.reduced_source != res.merged_source
    assert len(reduced_runs) == 1
    assert res.reduction_valid

    reduced_runs.clear()
    single = phase_end(
        backend, [PLAIN], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=runner, mutation_method="measure",
        log=lambda m: None)
    assert single.reduced_source == single.merged_source
    assert reduced_runs == []


# The starter-suite cleanup.

@pytest.mark.parametrize("source, failing", [
    (CULPRIT_AND_VICTIM, "test_victim"), (TWO_RELOADS, "test_victim"),
    (RELOAD_AND_SEED, "test_seed_zero")], ids=["reload", "two-reloads", "seed"])
def test_the_cleanup_prunes_only_the_tests_its_normal_run_fails(tmp_path, source, failing):
    """Each fixture's failing test fails without the instrument: the cleanup
    prunes it and keeps the reloads before it, so no culprit is looked
    for."""
    cls, green = _cleaned(tmp_path, source)
    assert "importlib.reload" in green
    assert f"def {failing}" not in green
    assert cls == "pruned"


def test_a_leak_that_breaks_nothing_stays_in_the_cleanup(tmp_path):
    cls, green = _cleaned(tmp_path, RELOAD)
    assert cls == "all_passing" and "def test_reload" in green


#: Passes under the hash seed the instrument sets, fails as delivered.
HASH_SEED_TEST = ('import os\n\nfrom calc import add\n\n\n'
                  'def test_hash_seed():\n'
                  '    assert os.environ["PYTHONHASHSEED"] == "0"\n\n\n'
                  'def test_add():\n    assert add(2, 3) == 5\n')


def _without_consolidation_mutation(monkeypatch) -> None:
    """Turn the phase end's mutation off, as the starter-suite tests
    do, so no engine starts against these fixtures."""
    from testuary import config as config_module

    monkeypatch.setattr(config_module, "consolidation_measure_mutation",
                        lambda: False)


def _starter_record(starters_dir: Path) -> dict:
    return json.loads((starters_dir / "suite_001" / "original.json").read_text(
        encoding="utf-8"))


def test_the_starter_phase_cleanup_is_the_normal_run(tmp_path, monkeypatch):
    """Through the starter phase: the test that needs the instrument's hash
    seed is pruned by the normal run. The cleanup record holds no
    acceptance as delivered."""
    import test_starter_phase as harness

    _without_consolidation_mutation(monkeypatch)
    root, target, tests_dir, starters_dir = harness._project(tmp_path)
    harness._run(root, target, tests_dir, starters_dir, [HASH_SEED_TEST],
                 monkeypatch, suite_count=1)
    meta = _starter_record(starters_dir)
    assert meta["classification"] == "pruned"
    assert "portability" not in meta["cleanup"]
    cleaned = (starters_dir / "suite_001" / "cleaned.py").read_text(
        encoding="utf-8")
    assert "def test_add" in cleaned
    assert "test_hash_seed" not in cleaned


# The settings.

def test_the_package_has_no_reject_non_portable_tests_and_the_reverse_note_off():
    from testuary import config as config_module

    for phase in ("test_coverage", "test_mutation"):
        config = config_module.get_agent_config(
            phase, parent="test_coverage" if phase == "test_mutation"
            else None)
        assert "reject_non_portable_tests" not in config, phase
        assert config["report_reverse_order_failures"] is False, phase


def test_the_run_as_delivered_in_a_container_through_the_adapter(
        tmp_path, monkeypatch, in_run, import_record):
    """Testuary's adapter describes the project through its configuration
    and public context, and the run goes through the backend's prepared copy
    in the guest. The setup file keeps its execution-side location, not the host path
    or the temporary staged location: the suite in the work directory is copied
    beside the project's copy, below the directory they share."""
    import posixpath
    import subprocess
    import sys

    from guest_fixtures import container_work_dir, controlled_guest
    # Only LangAdapter's private marker makes the setup file read as its instrument.
    from langadapter.python.project import _INSTRUMENT_MARKER

    zone = tmp_path / "wz"
    suite = zone / "tests"
    suite.mkdir(parents=True)
    test = suite / "test_c.py"
    test.write_text("def test_a():\n    assert True\n", encoding="utf-8")
    (suite / "conftest.py").write_text(
        "SETUP_FILE = __file__\n" + _INSTRUMENT_MARKER + "\n",
        encoding="utf-8")
    project = tmp_path / "testbed"
    project.mkdir()
    (project / "subject.py").write_text("value = 1\n", encoding="utf-8")
    heads, roots = [], []

    def pytest_run(args, options, host):
        if "pytest" not in args:
            # The adapter's helper scripts around the run, such as its probe
            # for pytest-benchmark's storage, run on their path's host side.
            script, path = args[args.index("-c") + 1:]
            return subprocess.run([sys.executable, "-c", script, str(host(path))],
                                  capture_output=True, text=True)
        report = next(a.split("=", 1)[1] for a in args
                      if a.startswith("--junit-xml="))
        stage = host(args[args.index("pytest") + 1]).parent
        heads.append((stage / "conftest.py").read_text(encoding="utf-8"))
        roots.append(options.get("cwd"))
        host(report).write_text(
            '<testsuites><testsuite><testcase file="test_c.py" classname="test_c" '
            'name="test_a"/></testsuite></testsuites>', encoding="utf-8")
        import_record(options.get("env"), host)
        return subprocess.CompletedProcess(args, 0, "1 passed\n", "")

    calls = controlled_guest(monkeypatch, container="delivery-container", root="/testbed",
                             source=project, work_dir=zone, handle=pytest_run)
    monkeypatch.setattr(project_module, "project_repo_dir", lambda: "/testbed")
    monkeypatch.setattr(config_module, "project_exec_mode", lambda: "container")
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", "/venv/bin/python")
    in_run()
    run = languages.get_backend("python").run_as_delivered(
        [str(test)], "/testbed", timeout=60)
    assert run.ok, run.output
    (head,) = heads
    staged = {"__name__": "conftest", "__file__": "staged"}
    exec(head, staged)
    # The run used a guest copy of the project, removed afterwards.
    (copy,) = [args[-1] for args in calls if args[0] == "cp"]
    assert staged["SETUP_FILE"] == posixpath.join(
        posixpath.dirname(copy), container_work_dir(zone).lstrip("/"), "tests", "conftest.py")
    assert ["cp", "-R", "-P", "--", "/testbed/.", copy] in calls
    assert ["rm", "-rf", "--", copy] in calls and roots == [copy]

