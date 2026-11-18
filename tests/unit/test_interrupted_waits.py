"""An interruption (SIGINT, Ctrl+C) ends a long Retry-After wait at once.

A KeyboardInterrupt is not an Exception, yet it must cancel the other
generation workers so the executor does not join them in their wait. On
Windows a thread blocked in one long timed wait does not receive Ctrl+C until
the wait returns, so waits proceed in short slices.
"""
from __future__ import annotations

import _thread
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import httpx
import pytest

import test_starter_phase as harness
from testuary import model_requests as mr
from testuary import starters
from phase_values import serve_starter_settings

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


class _RateLimited(Exception):
    """A provider's 429 that asks for a retry after 2000 s."""

    status_code = 429

    def __init__(self):
        super().__init__("rate limited")
        self.response = httpx.Response(429, headers={"Retry-After": "2000"})


@pytest.fixture(autouse=True)
def _consolidation_mutation_off(monkeypatch):
    from testuary import config
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: False)


def test_cancellation_ends_a_long_worker_wait_promptly():
    account = mr.RequestAccount(mr.RetrySettings())
    ended = []
    worker = threading.Thread(target=lambda: (account.wait(2000), ended.append(time.monotonic())))
    started = time.monotonic()
    worker.start()
    time.sleep(0.2)
    account.cancel.set()
    worker.join(5)
    assert ended and ended[0] - started < 3


def test_an_interruption_reaches_a_main_thread_wait_within_a_slice():
    account = mr.RequestAccount(mr.RetrySettings())
    timer = threading.Timer(0.3, _thread.interrupt_main)
    started = time.monotonic()
    timer.start()
    try:
        with pytest.raises(KeyboardInterrupt):
            account.wait(20)
    finally:
        timer.cancel()
    assert time.monotonic() - started < 3


def test_an_interruption_during_a_retry_after_wait_ends_generation_with_its_record(
    tmp_path, monkeypatch,
):
    root, target, tests_dir, starters_dir = harness._project(tmp_path)
    from testuary import config
    from testuary import models

    base_cfg = config.get_agent_config("test_initial")
    base_cfg.update({"suite_count": 1, "suite_run_timeout_seconds": 120})
    serve_starter_settings(monkeypatch, base_cfg)
    monkeypatch.setattr(models, "get_api_config",
                        lambda: {"model": "fake", "api_url": "", "api_key": "x"})
    requests = []

    def gen(prompt, *, temperature, seed, timeout):
        requests.append(seed)
        if len(requests) == 1:
            # Interrupt the main thread once the worker is in its wait.
            threading.Timer(0.5, _thread.interrupt_main).start()
        raise _RateLimited()

    started = time.monotonic()
    with pytest.raises(KeyboardInterrupt), models.installed(SimpleNamespace(generate_suite=gen)):
        starters.run_starter_phase(
            language="python", target_file_path=str(target),
            source_dir=str(root), test_output_dir=str(tests_dir),
            suites_dir=str(starters_dir), log=lambda *a, **k: None,
        )
    assert time.monotonic() - started < 10
    assert requests == [requests[0]]
    manifest = json.loads((Path(starters_dir) / starters._GENERATION_MANIFEST).read_text("utf-8"))
    assert manifest["generation_complete"] is False
    assert manifest["transient_failures"] == 1
    [record] = [json.loads(line) for line in
                (Path(starters_dir) / "request_records.jsonl").read_text("utf-8").splitlines() if line]
    assert record["outcome"] == mr.TRANSIENT
