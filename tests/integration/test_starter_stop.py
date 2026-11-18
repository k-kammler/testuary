"""A Ctrl+C, or two, during the green check of the starter suites, through
``testuary generate`` with real suite runs.

The starter suites are checked on the main thread, each response's tests
under a cancel scope. A stop ends them and removes their work directory and its
ownership receipt, so the command ends as interrupted with nothing left in
the temporary directory.

Marked ``slow``, like the other real-subprocess runs.
"""
from __future__ import annotations

import _thread
import tempfile
import threading
import time

import pytest

from test_interrupted_mutation_measurement import case  # noqa: F401  (the fixture)

from testuary.languages import mutation as mutation_runs
from testuary.languages import preparation
from testuary.languages import generic as python_language
from testuary import starters
from testuary import measurement

pytestmark = pytest.mark.slow

SLOW = ("import time\nfrom module import add\n\n\n"
        "def test_add():\n    time.sleep(5)\n    assert add(1, 2) == 3\n")


@pytest.mark.parametrize("presses, gap_s", [(1, 0.0), (2, 0.005), (2, 0.150)])
@pytest.mark.parametrize("during", ["response check", "measurement green check"])
def test_a_ctrl_c_during_the_starter_green_check_leaves_nothing(case, tmp_path, during, presses, gap_s):  # noqa: F811
    """``response check``: the check of a generated suite, which runs it to
    find its failing tests. ``measurement green check``: the un-mutated run
    before the starter suites' mutation measurement."""
    _events, run, monkeypatch = case
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    for name in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(name, str(temporary))
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    # The run's bytecode store lives as long as the process: this test's one
    # lies in its own temporary directory.
    preparation._close_stores()

    def generate(api_cfg, messages, **kwargs):
        if kwargs.get("on_request") is not None:
            kwargs["on_request"](kwargs["seed"], kwargs.get("max_output_tokens"))
        return (f"Here is the suite.\n\n```python\n{SLOW}```\n",
                {"input_tokens": 10, "output_tokens": 3})

    monkeypatch.setattr(starters, "_default_generate", generate)
    timers = []
    pressed = []

    def press():
        pressed.append(time.monotonic())
        _thread.interrupt_main()

    def first_press():
        if not timers:
            # The suite's test is running when the stop comes.
            timers.append(threading.Timer(1.5, press))
            timers[-1].start()

    if during == "response check":
        real_ids = python_language.GenericBackend.failing_test_ids

        def failing_test_ids(self, *args, **kwargs):
            first_press()
            return real_ids(self, *args, **kwargs)

        monkeypatch.setattr(python_language.GenericBackend, "failing_test_ids", failing_test_ids)
    else:
        real_green = measurement._green_check_with_retry

        def green(*args, **kwargs):
            first_press()
            return real_green(*args, **kwargs)

        monkeypatch.setattr(measurement, "_green_check_with_retry", green)
    real_cancel = mutation_runs.cancel_in_flight

    def cancel_in_flight():
        if presses == 2 and len(timers) == 1:
            timers.append(threading.Timer(gap_s, press))
            timers[-1].start()
        return real_cancel()

    monkeypatch.setattr(mutation_runs, "cancel_in_flight", cancel_in_flight)
    status, _output = run("derive", "stopped")
    ended = time.monotonic()
    for timer in timers:
        timer.join()
    assert status == 130
    assert len(timers) == presses
    # The 5 s test was stopped, not waited for.
    assert ended - pressed[0] < 4.0, ended - pressed[0]
    preparation._close_stores()
    assert sorted(path.name for path in temporary.iterdir()) == []
