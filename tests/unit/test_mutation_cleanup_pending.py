"""A mutation session whose cleanup failed admits no further work.

Its error (LangAdapter's MutationCleanupError, or any error that keeps the
session's cleanup handle) ends the work at once. The agents neither retry it
nor turn it into an agent failure the run continues after. The starter and
finishing sessions close like the loops' own sessions: they keep the error
that ended their work and report a cleanup that fails after it."""
from __future__ import annotations

from types import SimpleNamespace

import httpx
import pytest
from langchain_core.messages import HumanMessage

import test_langgraph as graph_harness
from testuary.agent import _langgraph as lg
from testuary.languages import mutation as mutation_runs
from testuary import measurement

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _pending(message="mutation session cleanup failed: TimeoutExpired: timed out"):
    error = RuntimeError(message)
    error.mutation_cleanup_handle = object()
    assert mutation_runs.mutation_cleanup_pending(error)
    return error


def test_a_pending_cleanup_is_never_retried_even_when_its_text_reads_temporary():
    pending = _pending()
    pending.__cause__ = httpx.ConnectError("refused")
    assert not lg._is_retryable_agent_error(pending)
    assert lg._is_retryable_agent_error(RuntimeError("tool subprocess timed out"))


def test_a_pending_cleanup_escaping_the_graph_ends_the_loop_with_that_error(monkeypatch):
    streams = []

    class _Graph:
        def stream(self, state, config=None, stream_mode=None):
            streams.append(1)
            raise pending
            yield  # pragma: no cover

    pending = _pending()
    sleeps = []
    monkeypatch.setattr(lg.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError) as raised:
        lg.run_graph_with_retries(_Graph(), {"messages": [HumanMessage(content="x")]},
                                  max_retries=3)
    assert raised.value is pending
    assert streams == [1] and sleeps == []


def test_a_tool_whose_session_cleanup_failed_ends_the_agent_without_a_rerun(monkeypatch):
    """Through the real tool dispatch the failed call is not answered and
    retried. Its error leaves the recovery loop for the phase to handle."""
    pending = _pending()
    graph, calls, requests = graph_harness._tool_failure_graph(pending)
    sleeps = []
    monkeypatch.setattr(lg.time, "sleep", sleeps.append)
    with pytest.raises(RuntimeError) as raised:
        lg.run_graph_with_retries(graph, {"messages": [HumanMessage(content="x")]},
                                  max_retries=3)
    assert raised.value is pending
    assert calls["flaky"] == 1 and len(requests) == 1 and sleeps == []


class _Session:
    def __init__(self, close_error=None):
        self.close_error = close_error
        self.closes = 0

    def close(self):
        self.closes += 1
        if self.close_error is not None:
            raise self.close_error


def _engine(monkeypatch, session):
    runner = SimpleNamespace(open=lambda workers, **kwargs: session,
                             full_operators=lambda: ["core/Op"],
                             configured_operators=lambda: None)
    monkeypatch.setattr(measurement.languages, "get_backend",
                        lambda language: SimpleNamespace(mutation_runner=lambda config, **kw: runner))
    return runner


CONFIG = {"engine_workers": 1, "per_mutant_timeout_seconds": 0,
          "measurement_timeout_seconds": 60, "mutation_max_timeout": 30,
          "min_per_mutant_timeout_seconds": 1, "per_mutant_timeout_factor": 10.0}


def _merged(tmp_path, body_error=None):
    with measurement.union_mutation_runner(language="python", target_file=str(tmp_path / "m.py"),
                                       source_dir=str(tmp_path), config=dict(CONFIG)) as run:
        assert run is not None
        if body_error is not None:
            raise body_error


def test_the_merged_session_reports_a_cleanup_that_fails_after_its_work(tmp_path, monkeypatch):
    failure = _pending()
    session = _Session(close_error=failure)
    _engine(monkeypatch, session)
    with pytest.raises(RuntimeError) as raised:
        _merged(tmp_path)
    assert raised.value is failure and session.closes == 1


def test_the_merged_session_keeps_the_error_that_ended_its_work(tmp_path, monkeypatch):
    session = _Session(close_error=_pending())
    _engine(monkeypatch, session)
    with pytest.raises(KeyboardInterrupt) as raised:
        _merged(tmp_path, KeyboardInterrupt())
    assert raised.value.mutation_cleanup_error is session.close_error


def _finishing(tmp_path, monkeypatch, outcome):
    suite = tmp_path / "test_m.py"
    suite.write_text("def test_x():\n    pass\n", encoding="utf-8")

    def measure(**kwargs):
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(measurement, "measure_finishing", measure)
    return measurement.run_finishing(
        target_file=str(tmp_path / "m.py"), source_dir=str(tmp_path), language="python",
        merged_test_path=suite, shared_setup_paths=[], scratch_root=tmp_path / "scratch",
        config=dict(CONFIG), standard_coverage=(90.0, 80.0))


def test_the_finishing_measurement_reports_a_cleanup_that_fails_after_it(tmp_path, monkeypatch):
    failure = _pending()
    _engine(monkeypatch, _Session(close_error=failure))
    with pytest.raises(RuntimeError) as raised:
        _finishing(tmp_path, monkeypatch, "measured")
    assert raised.value is failure


def test_the_finishing_measurement_keeps_the_error_that_ended_it(tmp_path, monkeypatch):
    session = _Session(close_error=_pending())
    _engine(monkeypatch, session)
    with pytest.raises(KeyboardInterrupt) as raised:
        _finishing(tmp_path, monkeypatch, KeyboardInterrupt())
    assert raised.value.mutation_cleanup_error is session.close_error


def test_the_finishing_measurement_closes_normally_after_its_work(tmp_path, monkeypatch):
    session = _Session()
    _engine(monkeypatch, session)
    assert _finishing(tmp_path, monkeypatch, "measured") == "measured"
    assert session.closes == 1


def test_the_finishing_measurement_records_every_killing_test(tmp_path, monkeypatch):
    """Finishing's own run of the delivered suite records every killing
    test, so its kill_matrix.json is complete."""
    runner = _engine(monkeypatch, _Session())
    _finishing(tmp_path, monkeypatch, "measured")
    assert runner.attribution == "all"


def test_an_interruption_that_keeps_a_native_cleanup_owner_is_pending():
    """LangAdapter keeps an interruption during a native session cleanup an
    interruption and attaches the session as its cleanup handle."""
    interrupted = KeyboardInterrupt()
    assert not mutation_runs.mutation_cleanup_pending(interrupted)
    interrupted.cleanup_handle = object()
    assert mutation_runs.mutation_cleanup_pending(interrupted)


def test_a_session_cleanup_still_owed_stays_pending():
    """A session cleanup still owed ends the work at once."""
    error = OSError("launch failed")
    error.mutation_cleanup_handle = object()
    assert mutation_runs.mutation_cleanup_pending(error)
