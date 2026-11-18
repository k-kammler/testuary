"""A mutation run whose launch left nothing running gives no result, and the
mutation session measures the next suite, with real Cosmic Ray.

The first Popen of the cosmic-ray exec launch fails: the command's own on
Windows and other POSIX hosts, the supervisor helper's on Linux. Kept as a
command to dispose of, such a launch would make the session's cleanup fail
and end the work on the target."""
from __future__ import annotations

import errno
import json
import subprocess
import sys

import pytest

from testuary import config as config_module
from testuary import measurement
from testuary.settings import resolve_settings

pytestmark = [pytest.mark.slow, pytest.mark.mutation_tool]

CALC = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
GOOD = ("from calc import add, sub\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n\n\n"
        "def test_sub():\n    assert sub(3, 2) == 1\n")


def test_a_failed_launch_gives_no_result_and_the_session_continues(tmp_path, monkeypatch):
    # Only LangAdapter's private launcher shows which start is the cosmic-ray exec launch.
    from langadapter import _process_tree

    project, suite = tmp_path / "proj", tmp_path / "good"
    for new_dir in (project, suite):
        new_dir.mkdir()
    (project / "calc.py").write_text(CALC, encoding="utf-8")
    (suite / "test_good.py").write_text(GOOD, encoding="utf-8")
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", sys.executable)
    # tests/conftest.py sets zero for the whole suite, which a resolved run
    # rejects. These runs time nothing.
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    settings = resolve_settings(base_dir=tmp_path, models={})
    with config_module.use_run_settings(settings):
        config = dict(config_module.get_agent_config("test_mutation", parent="test_coverage"))
        config.update({"cosmic_ray_operators": "", "cosmic_ray_catalogue_cache_dir": "",
                       "cosmic_ray_distributor": "local", "engine_workers": 1,
                       "cosmic_ray_clone_sweep_hours": 0})
        state = {"armed": False, "fail": False, "failed": 0}
        start, popen = _process_tree.start, subprocess.Popen

        def exec_start(args, **options):
            state["armed"] = state["fail"] and "exec" in args and any(
                "cosmic" in str(arg) for arg in args)
            try:
                return start(args, **options)
            finally:
                state["armed"] = False

        def refused(*args, **options):
            if state["armed"]:
                state["armed"], state["fail"] = False, False
                state["failed"] += 1
                raise OSError(errno.EAGAIN, "Resource temporarily unavailable (test)")
            return popen(*args, **options)

        monkeypatch.setattr(_process_tree, "start", exec_start)
        monkeypatch.setattr(subprocess, "Popen", refused)
        lines = []
        with measurement.union_mutation_runner(
                language="python", target_file=str(project / "calc.py"),
                source_dir=str(project), config=config,
                log_fn=lambda message: lines.append(str(message))) as run:
            assert run is not None
            first = run(suite)
            state["fail"] = True
            second = run(suite)
            third = run(suite)
    assert state["failed"] == 1
    assert first is not None and first["mutants"], "\n".join(lines[-20:])
    assert "incomplete" not in first
    # No result and no exception that ends the work: the engine fails the run
    # whose exec did not start, as for a missing cosmic-ray executable.
    assert second is None
    assert any("cosmic-ray exec did not start" in line for line in lines)
    assert third is not None and "incomplete" not in third

    def verdicts(matrix):
        return {json.dumps(row["key"], sort_keys=True): row["status"]
                for row in matrix["mutants"]}
    assert verdicts(third) == verdicts(first)
