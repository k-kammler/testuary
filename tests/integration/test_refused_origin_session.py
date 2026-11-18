"""A measurement refused at the origin check leaves Testuary's mutation
session usable, with real Cosmic Ray.

Under the starter mode both or derive, the starter suites are measured by
the pooled run and then the per-suite run in one mutation session. Here the
pooled run is refused at the origin check (its tests import the module under
test from a vendored copy outside their directory: they pass, and their run's
import record refuses them), and the session still measures the per-suite
run that gives the starter scores."""
from __future__ import annotations

import os
import sys

import pytest

from testuary import languages
from testuary import config as config_module
from testuary.languages.mutation import MutationInstrumentFault
from testuary import measurement
from testuary.settings import resolve_settings

pytestmark = [pytest.mark.slow, pytest.mark.mutation_tool]

CALC = "def add(a, b):\n    return a + b\n"
ELSEWHERE = ("import sys\n\n"
             "sys.path.insert(0, {vendored!r})\n"
             "from calc import add\n\n\n"
             "def test_add():\n    assert add(2, 3) == 5\n")
GREEN = "from calc import add\n\n\ndef test_add():\n    assert add(2, 3) == 5\n"


@pytest.fixture
def layout(tmp_path, monkeypatch):
    project, refused, green, vendored = (
        tmp_path / "proj", tmp_path / "pool", tmp_path / "good", tmp_path / "vendored")
    for new_dir in (project, refused, green, vendored):
        new_dir.mkdir()
    (project / "calc.py").write_text(CALC, encoding="utf-8")
    # Outside the suite's directory: the normal run runs each test file alone,
    # as a user who copied it would, so a copy beside it would not be there.
    (vendored / "calc.py").write_text(CALC, encoding="utf-8")
    (refused / "test_vendored.py").write_text(ELSEWHERE.format(vendored=str(vendored)),
                                              encoding="utf-8")
    (green / "test_good.py").write_text(GREEN, encoding="utf-8")
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
        yield project, refused, green, config


def test_a_refused_run_leaves_the_session_to_the_next_run(layout):
    project, refused, green, config = layout
    runner = languages.get_backend("python").mutation_runner(config, source_dir=str(project))
    session = runner.open(1, suite_dirs=(refused.parent,))
    try:
        common = dict(target=project / "calc.py", per_mutant_timeout=30, wall_clock_timeout=600)
        with pytest.raises(MutationInstrumentFault):
            session.run(test_dir=refused, test_files=[str(refused / "test_vendored.py")],
                        **common)
        measured = session.run(test_dir=green, test_files=[str(green / "test_good.py")], **common)
        assert measured.run.score() == 100.0
        per_suite = session.per_suite(target=project / "calc.py",
                                      suites=[("suite_000", green, [str(green / "test_good.py")], 30)],
                                      wall_clock_timeout=600)
        assert per_suite.runs["suite_000"].score() == 100.0
    finally:
        session.close()


def test_the_union_runner_measures_the_starters_after_a_refused_pool(layout):
    project, refused, green, config = layout
    lines = []
    with measurement.union_mutation_runner(language="python", target_file=str(project / "calc.py"),
                                       source_dir=str(project), config=config,
                                       log_fn=lambda message: lines.append(str(message))) as run:
        assert run is not None
        assert run(refused, attribution="all") is None
        measured = run.per_suite({"suite_000": green})
    assert measured is not None, "\n".join(lines[-20:])
    assert measured["outcomes"] == {"suite_000": "completed"}
    assert measured["unmeasured"] == []
    assert measured["union"]["mutants"]
    assert os.path.exists(project / "calc.py")
