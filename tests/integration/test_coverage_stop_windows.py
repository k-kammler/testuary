"""Two Ctrl+C during a coverage agent's tool, on Windows, with a real suite
run.

A terminal's Ctrl+C reaches the process as an interrupt of its main
thread, here raised in the process itself. The second one follows the
first one reaching the tool by 5 ms or 150 ms. The agent ends as
interrupted within seconds, no process of the run and nothing in the
temporary directory remain, and a later run in the same directories completes.
Linux runs the same through process-group signals (test_coverage_stop.py).

Marked ``slow``, like the other real-subprocess runs.
"""
from __future__ import annotations

import os
import sys
import tempfile
import threading
import time

import pytest

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform != "win32", reason="Linux uses process-group signals"),
]


def _processes_in(base_dir):
    psutil = pytest.importorskip("psutil")
    found = []
    for process in psutil.process_iter(["pid", "cmdline", "cwd"]):
        try:
            text = " ".join(process.info["cmdline"] or []) + " " + (process.info["cwd"] or "")
        except (psutil.Error, TypeError):
            continue
        if str(base_dir) in text and process.pid != os.getpid():
            found.append(f"{process.pid}: {text[:200]}")
    return found


@pytest.mark.parametrize("gap_s", [0.005, 0.150])
@pytest.mark.parametrize("tool", ["measure_coverage", "run_tests"])
def test_two_ctrl_c_during_a_coverage_tool_leave_nothing(tmp_path, monkeypatch, tool, gap_s):
    import _thread

    pytest.importorskip("langadapter").CancelScope
    sys.path.insert(0, os.path.dirname(__file__))
    import _coverage_stop_child as child

    from testuary import __main__ as cli
    from testuary.languages import mutation as mutation_runs
    from testuary.languages import preparation

    temporary = tmp_path / "tmp"
    temporary.mkdir()
    for name in ("TEMP", "TMP", "TMPDIR"):
        monkeypatch.setenv(name, str(temporary))
    monkeypatch.setattr(tempfile, "tempdir", str(temporary))
    # The run's bytecode store lives as long as the process: this test's
    # one lies in its own temporary directory.
    preparation._close_stores()
    cancelled_before = set(mutation_runs._CANCELLED)
    work = tmp_path / "work"
    work.mkdir()
    events = {}
    written = []
    real_write = os.write

    def write(fd, data):
        # The child module marks the tool's start and the first stop's
        # cancel on stdout; here they drive the interrupts.
        if fd == 1 and data in (b"RUNNING\n", b"CANCELLING\n"):
            written.append(data.strip().decode())
            if data == b"RUNNING\n" and "first" not in events:
                events["first"] = threading.Timer(2.0, _thread.interrupt_main)
                events["first"].start()
                events["at"] = time.monotonic() + 2.0
            if data == b"CANCELLING\n" and "second" not in events:
                events["second"] = threading.Timer(gap_s, _thread.interrupt_main)
                events["second"].start()
            return len(data)
        return real_write(fd, data)

    monkeypatch.setattr(os, "write", write)
    monkeypatch.setattr(cli, "_retry_interrupted_cleanup", lambda stop: None)
    status = child.main(work, tool, monkeypatch.setattr)
    stopped_in = time.monotonic() - events["at"]
    for timer in ("first", "second"):
        events[timer].join()
    assert status == 130 and written[:2] == ["RUNNING", "CANCELLING"], written
    # The 10 s test was stopped, not waited for.
    assert stopped_in < 8.0, stopped_in
    assert not mutation_runs._CANCELLED - cancelled_before
    time.sleep(1.0)
    assert _processes_in(tmp_path) == []
    # As the process's exit does.
    preparation._close_stores()
    assert sorted(path.name for path in temporary.iterdir()) == []
    monkeypatch.setattr(os, "write", real_write)
    assert child.main(work, "finish", monkeypatch.setattr) == 0
