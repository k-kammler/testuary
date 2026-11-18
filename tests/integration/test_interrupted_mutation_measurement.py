"""A stop or a failed cleanup inside a mutation measurement of the starter
suites, through ``testuary generate`` with real Cosmic Ray.

The cluster's stop is a SIGINT, a KeyboardInterrupt in the main thread
wherever the run is. Inside any mutation run of the starter suites (the
pooled run, each per-suite run and the merge's runs) it ends the command at
once as interrupted: status 130, no further mutation run, green check or
coverage measurement, no result row, and the phase left resumable. A
cleanup failure of the measuring session (LangAdapter's
MutationCleanupError) ends the work on the target as an execution failure
naming it, with no further work in that session.

Marked ``slow``, like the other real Cosmic Ray runs.
"""
from __future__ import annotations

import csv
import functools
import os
import sys

import pytest

from tool_fixtures import cosmic_ray_installed

from langadapter.mutation import MutationCleanupError
from langadapter.python.adapter import PythonAdapter
from testuary import __main__ as cli
from testuary import exits
from testuary import phase_state
from testuary import settings as settings_module
from testuary.languages import mutation as mutation_runs
from testuary.languages import generic as python_language
from testuary import starters
from testuary import measurement
from testuary.settings import RunSettings

pytestmark = [pytest.mark.slow, pytest.mark.mutation_tool]

TARGET = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
SUITES = {
    1: "from module import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
    2: "from module import sub\n\n\ndef test_sub():\n    assert sub(3, 2) == 1\n",
}
#: Two mutants keep each real run short. The measurement path is unchanged.
OPERATORS = ["core/ReplaceBinaryOperator_Add_Sub", "core/ReplaceBinaryOperator_Sub_Add"]


@pytest.fixture
def case(tmp_path, monkeypatch):
    if not cosmic_ray_installed():
        pytest.skip("cosmic-ray not installed")
    project = tmp_path / "projects" / "subject"
    project.mkdir(parents=True)
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    (project / "module.py").write_text(TARGET, encoding="utf-8")
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", sys.executable)
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(mutation_runs.MutationRunner, "full_operators", lambda self: list(OPERATORS))

    def generate(api_cfg, messages, **kwargs):
        if kwargs.get("on_request") is not None:
            kwargs["on_request"](kwargs["seed"], kwargs.get("max_output_tokens"))
        suite = SUITES[int(kwargs["seed"]) - 37]
        return (f"Here is the suite.\n\n```python\n{suite}```\n",
                {"input_tokens": 10, "output_tokens": 3})

    monkeypatch.setattr(starters, "_default_generate", generate)
    events = []
    real_green = measurement._green_check_with_retry
    real_coverage = python_language.GenericBackend.collect_eviction_data

    def green(*args, **kwargs):
        events.append("green check")
        return real_green(*args, **kwargs)

    def coverage(self, *args, **kwargs):
        events.append("coverage")
        return real_coverage(self, *args, **kwargs)

    monkeypatch.setattr(measurement, "_green_check_with_retry", green)
    monkeypatch.setattr(python_language.GenericBackend, "collect_eviction_data", coverage)

    def run(mode, name):
        output = tmp_path / f"output_{name}"
        settings = RunSettings(base_dir=tmp_path, environment=dict(os.environ), sections={
            "api": {"model": "explicit"}, "main": {"hints_file": "", "seed": 37},
            "starter": {"suite_count": 2, "attempts_per_suite": 1, "concurrent_requests": 1,
                             "suite_run_timeout_seconds": 120, "max_output_tokens": 40,
                             "suite_mutation": mode},
            "mutation": {"cosmic_ray_catalogue_cache_dir": "",
                              "cosmic_ray_distributor": "local", "engine_workers": 1,
                              "cosmic_ray_clone_sweep_hours": 0},
            "project": {"name": "subject", "source_dir": str(project), "output_dir": str(output)},
        }, models={"explicit": {"provider": "openai", "model": "fixture",
                                "api_url": "https://example.test/v1", "api_key": "fixture"}})
        monkeypatch.setattr(settings_module, "resolve_settings", lambda **kwargs: settings)
        monkeypatch.setattr(sys, "argv", [
            "testuary", "generate", "--source-files", "module.py",
            "--stop-after-phase", "starter", "--display", "plain"])
        events.clear()
        return cli.main(), output

    return events, run, monkeypatch


#: The Python adapter's own opening of a mutation session, which the tests wrap.
_OPEN = PythonAdapter.open_mutation_session


def _engine(case, *, at=0, failure="interrupt"):
    """Count the measurements Testuary asks of each mutation session the
    Python adapter opens (``run`` and ``per_suite``), and end the ``at``-th
    with ``failure``: a stop, or the session's cleanup error."""
    events, _run, monkeypatch = case

    def run(session, measure, request):
        events.append("mutation")
        if events.count("mutation") == at:
            events.append("ended")
            if failure == "interrupt":
                raise KeyboardInterrupt
            raise MutationCleanupError(session, "OSError: the session directory could not be removed")
        return measure(request)

    def opened(adapter, *args, **options):
        session = _OPEN(adapter, *args, **options)
        for name in ("run", "per_suite"):
            setattr(session, name, functools.partial(run, session, getattr(session, name)))
        return session

    monkeypatch.setattr(PythonAdapter, "open_mutation_session", opened)


def _starter_status(output):
    (workspace,) = [path for path in (output / "results").iterdir() if path.is_dir()]
    return ((phase_state.load(workspace).get("starter") or {}).get("0") or {}).get("status")


def _rows(output):
    path = output / "results" / "tests.csv"
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


@pytest.mark.parametrize("mode", ["both", "derive", "measure"])
def test_a_stop_or_a_failed_cleanup_in_any_starter_measurement_ends_the_work(mode, case, capsys):
    events, run, _monkeypatch = case
    _engine(case)
    status, _output = run(mode, "complete")
    assert status == 0
    total = events.count("mutation")
    assert total >= 2, events
    capsys.readouterr()
    for at in range(1, total + 1):
        _engine(case, at=at)
        status, output = run(mode, f"stop_{at}")
        assert status == exits.INTERRUPTED, (mode, at)
        # Nothing ran after the stop.
        assert events[-1] == "ended", (mode, at, events)
        assert _starter_status(output) == "in_progress", (mode, at)
        assert _rows(output) == [], (mode, at)
        err = capsys.readouterr().err
        assert "Traceback" not in err and "--resume" in err, (mode, at, err[-2000:])

        _engine(case, at=at, failure="cleanup")
        status, output = run(mode, f"cleanup_{at}")
        assert status == 1, (mode, at)
        assert events[-1] == "ended", (mode, at, events)
        # A crash is recorded as the phase's infrastructure failure, which a
        # resume redoes, as it redoes a phase left in progress.
        assert _starter_status(output) == "failed", (mode, at)
        (row,) = _rows(output)
        assert "crash" in str(row).lower(), (mode, at, row)
        log = (output / "logs" / "run.log").read_text(encoding="utf-8")
        assert "MutationCleanupError" in log, (mode, at, log[-3000:])
        capsys.readouterr()


#: The starting suite kills one of the two mutants, so the loop starts.
MUTATION_TARGET = TARGET
MUTATION_BASELINE = "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def test_a_stop_during_an_agent_measurement_ends_the_mutation_phase(tmp_path, monkeypatch):
    """The mutation agent's measuring tool runs on a worker thread, so a
    stop raises in the main thread while it waits for the tool. The phase
    then ends as interrupted: the measurement in flight is cancelled
    and gives no result, and no model request and no tool call follows."""
    import _thread
    import threading

    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from testuary import config as config_module
    from testuary.agent import mutation_agent as tm

    if not cosmic_ray_installed():
        pytest.skip("cosmic-ray not installed")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setattr(mutation_runs.MutationRunner, "full_operators", lambda self: list(OPERATORS))
    project = tmp_path / "subject"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    target = project / "calc.py"
    target.write_text(MUTATION_TARGET, encoding="utf-8")
    output = tmp_path / "tests"
    output.mkdir()
    events = []

    class _Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

        def invoke(self, input, config=None, **kwargs):
            events.append("model request")
            return super().invoke(input, config, **kwargs)

    def call(name, cid, **args):
        return AIMessage(content="", tool_calls=[{"name": name, "id": cid, "args": args}])

    improved = MUTATION_BASELINE.replace("import add", "import add, sub") + (
        "\n\ndef test_sub():\n    assert sub(3, 2) == 1\n")
    model = _Model(responses=[
        call("read_tests", "r0"),
        # The forced first measurement, served from the stored baseline.
        call("measure_mutation", "m0"),
        call("edit_tests", "e0", old_text=MUTATION_BASELINE, new_text=improved),
        call("run_tests", "t0"),
        call("measure_mutation", "m1"),
        call("run_tests", "t1"),
        call("measure_mutation", "m2"),
    ])
    monkeypatch.setattr(tm.MutationAgent, "build_chat_model", lambda agent: model)
    monkeypatch.setattr(tm, "_seed_mutation_baseline_from_matrix", lambda agent: False)

    def run(measure, request):
        on_worker = threading.current_thread() is not threading.main_thread()
        events.append("tool measurement" if on_worker else "baseline measurement")
        if on_worker:
            _thread.interrupt_main()
        result = measure(request)
        events.append("measured")
        return result

    def opened(adapter, *args, **options):
        session = _OPEN(adapter, *args, **options)
        for name in ("run", "per_suite"):
            setattr(session, name, functools.partial(run, getattr(session, name)))
        return session

    monkeypatch.setattr(PythonAdapter, "open_mutation_session", opened)
    settings = RunSettings(base_dir=tmp_path, environment=dict(os.environ), sections={
        "api": {"model": "explicit"}, "main": {"hints_file": "", "seed": 37},
        "mutation": {"max_requests_per_phase": 5, "max_loop_attempts": 0, "goal": 100,
                          "cosmic_ray_catalogue_cache_dir": "", "cosmic_ray_distributor": "local",
                          "engine_workers": 1, "cosmic_ray_clone_sweep_hours": 0},
    }, models={"explicit": {"provider": "openai", "model": "fixture",
                            "api_url": "https://example.test/v1", "api_key": "fixture"}})
    with config_module.use_run_settings(settings), pytest.raises(KeyboardInterrupt):
        tm.init_logging("subject", str(tmp_path / "logs"), "mutation.jsonl")
        tm.generate_mutation_tests(
            "subject", str(target), str(output),
            preseed={"content": MUTATION_BASELINE, "file_stem": "calc"}, coverage_baseline_pct=80.0)
    # Four model requests led to the measuring call, which the stop
    # cancelled. Nothing followed it.
    assert events == ["baseline measurement", "measured", *["model request"] * 5,
                      "tool measurement"], events
    # The cancelled measurement's session closed itself: none is left for
    # the exit to close.
    assert not mutation_runs._CANCELLED


def test_a_ctrl_c_cancels_an_agent_measurement_within_seconds(tmp_path, monkeypatch):
    """With the engine's cancel, a Ctrl+C during the mutation
    agent's measuring tool stops the measurement in flight: the run ends as
    interrupted within seconds, far below the measurement's length, the
    cancelled measurement is no result, and no model request follows."""
    import _thread
    import threading
    import time

    session_module = pytest.importorskip("langadapter.mutation")
    if not hasattr(session_module, "MeasurementCancelled"):
        pytest.skip("the installed LangAdapter has no measurement cancel")
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from testuary import config as config_module
    from testuary.agent import mutation_agent as tm

    if not cosmic_ray_installed():
        pytest.skip("cosmic-ray not installed")
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setattr(mutation_runs.MutationRunner, "full_operators", lambda self: list(OPERATORS))
    project = tmp_path / "subject"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    (project / "calc.py").write_text(MUTATION_TARGET, encoding="utf-8")
    output = tmp_path / "tests"
    output.mkdir()
    events = []

    class _Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

        def invoke(self, input, config=None, **kwargs):
            events.append("model request")
            return super().invoke(input, config, **kwargs)

    def call(name, cid, **args):
        return AIMessage(content="", tool_calls=[{"name": name, "id": cid, "args": args}])

    # Every test sleeps, so each mutant's run takes seconds.
    slow = ("import time\nfrom calc import add, sub\n\n\n"
            "def test_add():\n    time.sleep(3)\n    assert add(1, 2) == 3\n\n\n"
            "def test_sub():\n    time.sleep(3)\n    assert sub(3, 2) == 1\n")
    model = _Model(responses=[
        call("read_tests", "r0"),
        # The forced first measurement, served from the stored baseline.
        call("measure_mutation", "m0"),
        call("edit_tests", "e0", old_text=MUTATION_BASELINE, new_text=slow),
        call("run_tests", "t0"),
        call("measure_mutation", "m1"),
        call("run_tests", "t1"),
    ])
    monkeypatch.setattr(tm.MutationAgent, "build_chat_model", lambda agent: model)
    monkeypatch.setattr(tm, "_seed_mutation_baseline_from_matrix", lambda agent: False)
    stopped = {}

    def run(measure, request):
        if threading.current_thread() is not threading.main_thread():
            events.append("tool measurement")

            def stop():
                stopped["at"] = time.monotonic()
                _thread.interrupt_main()

            threading.Timer(1.0, stop).start()
        result = measure(request)
        events.append("measured")
        return result

    def opened(adapter, *args, **options):
        session = _OPEN(adapter, *args, **options)
        for name in ("run", "per_suite"):
            setattr(session, name, functools.partial(run, getattr(session, name)))
        return session

    monkeypatch.setattr(PythonAdapter, "open_mutation_session", opened)
    settings = RunSettings(base_dir=tmp_path, environment=dict(os.environ), sections={
        "api": {"model": "explicit"}, "main": {"hints_file": "", "seed": 37},
        "mutation": {"max_requests_per_phase": 5, "max_loop_attempts": 0, "goal": 100,
                          "cosmic_ray_catalogue_cache_dir": "", "cosmic_ray_distributor": "local",
                          "engine_workers": 1, "cosmic_ray_clone_sweep_hours": 0},
    }, models={"explicit": {"provider": "openai", "model": "fixture",
                            "api_url": "https://example.test/v1", "api_key": "fixture"}})
    with config_module.use_run_settings(settings), pytest.raises(KeyboardInterrupt):
        tm.init_logging("subject", str(tmp_path / "logs"), "mutation.jsonl")
        tm.generate_mutation_tests(
            "subject", str(project / "calc.py"), str(output),
            preseed={"content": MUTATION_BASELINE, "file_stem": "calc"}, coverage_baseline_pct=80.0)
    ended = time.monotonic()
    assert ended - stopped["at"] < 5.0, ended - stopped["at"]
    # The baseline was measured, then four requests led to the measuring
    # call, which the stop cancelled: no result and no request after it.
    assert events == ["measured", *["model request"] * 5, "tool measurement"], events
    assert not mutation_runs._CANCELLED
