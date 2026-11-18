"""The agents' test runs are the normal run, and the notes on a passing run
decide nothing: through the coverage agent's tools with the real Python
backend, and the mutation agent's normal run with a fake one."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from agent_fixtures import coverage_agent, mutation_agent, run_result

from testuary.languages import DeliveredRun
from testuary.languages.generic import DeliveredCase
from testuary.agent import coverage_agent as tc
from testuary.agent import mutation_agent as tm
from phase_values import set_phase_value

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

TARGET = ('class Box:\n    pass\n\n\n'
          'def make_box():\n    return Box()\n\n\n'
          'def add(a, b):\n    return a + b\n')

PLAIN = '''from calc import add


def test_add():
    assert add(2, 3) == 5
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
HARMLESS_LEAK = '''import importlib

import calc
from calc import add


def test_reload():
    importlib.reload(calc)
    assert calc.add(1, 2) == 3


def test_add():
    assert add(2, 3) == 5
'''
#: Passes in the file's order only.
ORDER_DEPENDENT = '''STATE = []


def test_first():
    STATE.append(1)


def test_second():
    assert STATE == [1]
'''
#: The reload removed; one test more than the earlier best green suite.
REPAIRED = '''import calc
from calc import Box, add


def test_victim():
    assert isinstance(calc.make_box(), Box)


def test_add():
    assert add(2, 3) == 5
'''


def _coverage_agent(tmp_path: Path, suite: str) -> "tc.CoverageAgent":
    project = tmp_path / "proj"
    project.mkdir()
    (project / "calc.py").write_text(TARGET, encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_calc.py").write_text(suite, encoding="utf-8")
    return coverage_agent(project, target_file_path=str(project / "calc.py"),
                          target_file_content=TARGET, test_output_dir=str(tests))


def _write(agent, suite: str) -> None:
    (Path(agent.deps.test_output_dir) / "test_calc.py").write_text(
        suite, encoding="utf-8")
    agent.session.tests_changed_since_run_tests = True


def test_the_reverse_order_diagnostic_tells_what_it_observed(
        tmp_path, monkeypatch):
    """The diagnostic reports an observed disagreement, not a cause, and
    the run stays green."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "report_reverse_order_failures",
                        True)
    agent = _coverage_agent(tmp_path, ORDER_DEPENDENT)
    out = agent.run_tests()
    assert agent.session.last_tests_passed is True
    assert ("[STANDS ALONE, ORDER] These tests passed in file order but "
            "failed in reverse order:\n  - test_calc.test_second\n") in out
    assert "This may indicate order-dependent state. The suite still passes." in out


def test_the_loops_normal_run_decides_alone(tmp_path, monkeypatch):
    """run_tests is the normal run, the run as delivered, made once: a
    reload that breaks a later test is an ordinary red run naming the
    failing test."""
    agent = _coverage_agent(tmp_path, CULPRIT_AND_VICTIM)
    backend_type = type(agent.test_backend())
    real = backend_type.run_as_delivered
    calls = []

    def counting(self, *args, **kwargs):
        calls.append(kwargs.get("reverse", False))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(backend_type, "run_as_delivered", counting)
    out = agent.run_tests()
    assert agent.session.last_tests_passed is False
    assert "[STANDS ALONE]" not in out
    assert "test_victim" in out
    assert calls == [False]


RELOAD_THEN_THREE_FAILURES = '''import importlib

import calc
from calc import Box


def test_reload():
    importlib.reload(calc)
    assert calc.add(1, 2) == 3


def test_victim_one():
    assert isinstance(calc.make_box(), Box)


def test_victim_two():
    assert isinstance(calc.make_box(), Box)


def test_victim_three():
    assert type(calc.make_box()) is Box
'''


def test_with_the_run_as_delivered_as_normal_run_the_failures_read_as_red(
        tmp_path, monkeypatch):
    """The three tests a reload breaks fail as in any red run, and the final
    answer is refused as not green."""
    agent = _coverage_agent(tmp_path, RELOAD_THEN_THREE_FAILURES)
    out = agent.run_tests()
    assert agent.session.last_tests_passed is False
    assert "[STANDS ALONE]" not in out
    for name in ("test_victim_one", "test_victim_two", "test_victim_three"):
        assert name in out
    assert "-> tests are RED. 3 failing tests.\n" in out
    refusal = agent.measure_coverage()
    assert refusal.startswith("REFUSED")
    rejected = agent.validate_output(state={}, parsed=tc.CoverageTestResult(
        test_file_name="test_calc.py", explanation="Covers make_box and add on their own",
        targeted_functions=["make_box", "add"]))
    assert rejected == agent.tool_msg("validator_not_green")


def _leak(test, **changed):
    fields = {name: () for name in ("added", "rebound", "removed", "xarray_options",
                                    "numpy_errors", "numpy_print")}
    fields.update(changed)
    return SimpleNamespace(test=test, **fields)


def _notes(tmp_path, leaks=(), reverse=None):
    tmp_path.mkdir()
    agent = _coverage_agent(tmp_path, PLAIN)
    backend = SimpleNamespace(run_as_delivered=lambda *a, **kw: reverse)
    return agent._run_notes(SimpleNamespace(leaks=list(leaks)), backend, "run_tests")


#: The rendered notes word for word: templates plus the text the agent builds
#: around them (leak lines, reverse order).
_NOTE = (
        '[STANDS ALONE, NOTE] Some tests leave shared state changed, so a test\n'
        'run alone or in another order can fail because of them:\n'
        '\n'
        'Tests that leave the module under test or a library setting changed:\n'
        '  - test_calc.test_np: changed numpy error settings all\n'
        '\n'
        'What to do:\n'
        '  1. A test must leave the module under test as it found it. Change\n'
        '     it only through monkeypatch, which restores what it set, or\n'
        '     restore it in a fixture; reloading the module in place replaces\n'
        '     every function and class the later tests use. To run import-time\n'
        '     code (an ``if TYPE_CHECKING:`` block, an ``except ImportError:``\n'
        '     fallback), reload it in a fixture that first saves\n'
        '     ``dict(module.__dict__)`` as ``saved`` and afterwards restores it\n'
        '     with ``module.__dict__.clear()`` and ``module.__dict__.update(saved)``.\n'
        '  2. A test must restore a library setting it changes: use the\n'
        "     library's context manager (np.errstate, np.printoptions,\n"
        '     xr.set_options) or save and restore the setting in a fixture.\n'
        "  3. A test must not rely on the measurement's fixed random seed or on\n"
        "     dask's default scheduler: seed or configure what it uses.\n"
    )
_ORDER = (
        '[STANDS ALONE, ORDER] These tests passed in file order but failed in reverse order:\n'
        '  - test_calc.test_second\n'
        '\n'
        'This may indicate order-dependent state. The suite still passes.\n'
    )


def test_the_rendered_notes_are_snapshotted(tmp_path, monkeypatch):
    # Only a library setting changed: the note does not claim the module did,
    # and it says how to restore the setting.
    note = _notes(tmp_path / "note", [_leak("test_calc.test_np", numpy_errors=("all",))])
    assert note == _NOTE
    assert "module under test changed" not in note.split("\n")[0]
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "report_reverse_order_failures", True)
    reverse = DeliveredRun(cases=(DeliveredCase("test_calc.test_second", "test_second", "failed"),),
                           failing=("test_calc.test_second",), reason=None, output="", reverse=True)
    assert _notes(tmp_path / "order", (), reverse) == _ORDER


def test_a_reload_that_breaks_nothing_stays_green_with_a_note(tmp_path):
    agent = _coverage_agent(tmp_path, HARMLESS_LEAK)
    out = agent.run_tests()
    assert agent.session.last_tests_passed is True
    assert "[STANDS ALONE, NOTE]" in out
    assert "test_calc.test_reload" in out and "rebound" in out
    assert "possible cause" not in out


def test_the_reverse_order_run_is_kept_per_file_content(tmp_path, monkeypatch):
    """No run as delivered decides; the reverse-order diagnostic's run is
    made once per file content."""
    set_phase_value(monkeypatch, tc._AGENT_CONFIG, "report_reverse_order_failures", True)
    agent = _coverage_agent(tmp_path, PLAIN)
    backend_type = type(agent.test_backend())
    real = backend_type.run_as_delivered
    calls = {"n": 0}

    def counting(self, *args, **kwargs):
        # The normal run is a run as delivered too; the reverse runs count.
        calls["n"] += bool(kwargs.get("reverse"))
        return real(self, *args, **kwargs)

    monkeypatch.setattr(backend_type, "run_as_delivered", counting)
    agent._run_tests_uncached("run_tests")
    agent._run_tests_uncached("run_tests")
    assert calls["n"] == 1
    _write(agent, HARMLESS_LEAK)
    agent._run_tests_uncached("run_tests")
    assert calls["n"] == 2


def test_the_coverage_prelude_decides_by_the_normal_run(tmp_path):
    agent = _coverage_agent(tmp_path, CULPRIT_AND_VICTIM)
    out = agent.measure_coverage()
    assert agent.session.last_tests_passed is False
    assert "[STANDS ALONE]" not in out


def test_without_the_reverse_order_note_no_reverse_run_is_made(tmp_path, monkeypatch):
    """A passing run gets its leak note from its normal run, observed
    without restoring anything, and no run in reverse order unless the
    reverse-order note is on."""
    agent = _coverage_agent(tmp_path, HARMLESS_LEAK)
    backend_type = type(agent.test_backend())
    real = backend_type.run_as_delivered

    def forward_only(self, *args, **kwargs):
        assert not kwargs.get("reverse"), "no reverse run without the reverse-order note"
        return real(self, *args, **kwargs)

    monkeypatch.setattr(backend_type, "run_as_delivered", forward_only)
    out = agent.run_tests()
    assert agent.session.last_tests_passed is True
    assert "[STANDS ALONE, NOTE]" in out and "[STANDS ALONE]" not in out


# The mutation agent's normal run

class _Stop(Exception):
    """The engine was reached."""


def _mutation_agent(tmp_path, monkeypatch, delivered_calls):
    tests = tmp_path / "tests"
    tests.mkdir()
    source = tmp_path / "src" / "foo.py"
    source.parent.mkdir()
    source.write_text("def f():\n    return 1\n", encoding="utf-8")
    failing = DeliveredRun(
        cases=(DeliveredCase("test_foo.test_x", "test_x", "failed", "boom"),),
        failing=("test_foo.test_x",), reason=None,
        output="FAILED test_foo.py::test_x - boom\n1 failed in 0.1s")

    def run_as_delivered(paths, source_dir, *, timeout, reverse=False,
                         seed=1, sources=None):
        delivered_calls.append(list(paths))
        return failing

    def engine(config):
        raise _Stop("the engine was reached")

    monkeypatch.setattr(tm.languages, "get_backend", lambda lang: SimpleNamespace(
        run_test_files=lambda *a, **kw: run_result(collected=1, passed=1, output="1 passed in 0.1s"),
        run_as_delivered=run_as_delivered,
        shared_setup_files=lambda test_dir: (),
        file_extensions=(".py",), test_file_extensions=lambda: (".py",),
        mutation_runner=lambda config, **kw: engine(config),
    ))
    agent = mutation_agent(tmp_path, target_file_content="def f():\n    return 1\n",
                           test_output_dir=str(tests), mutation_test_file_name="test_foo.py",
                           coverage_baseline_pct=85.0)
    agent.target_path().write_text("def test_x():\n    assert True\n",
                                   encoding="utf-8")
    return agent


def test_the_mutation_measurement_starts_with_the_normal_run(tmp_path, monkeypatch):
    """A green normal run reaches the engine: no run as delivered judges the
    file first."""
    calls: list = []
    agent = _mutation_agent(tmp_path, monkeypatch, calls)
    with pytest.raises(_Stop):
        agent._measure_mutation_engine()
    assert calls == []


def test_the_framework_baseline_is_not_checked(tmp_path, monkeypatch):
    calls: list = []
    agent = _mutation_agent(tmp_path, monkeypatch, calls)
    with pytest.raises(_Stop):
        agent._measure_mutation_engine(framework_prep=True)
    assert calls == []
