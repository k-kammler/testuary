"""Two Ctrl+C during a coverage agent's tool, on Linux, with a real suite
run.

Both interrupts go to the process group, as a terminal sends them, the
second one 5 ms or 150 ms after the first reaches the tool. The run ends
as interrupted, with no process of the run and nothing in TMPDIR left, and
a later run in the same directories completes.

Marked ``slow``, like the other real-subprocess runs.
"""
from __future__ import annotations

import os
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

import pytest

from test_second_stop import MARK, _processes_of, _wait_for

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform != "linux", reason="process groups and /proc are Linux's"),
]

CHILD = Path(__file__).resolve().parent / "_coverage_stop_child.py"


def _run(tmp_path, work, temporary, mark, mode, out, err):
    env = dict(os.environ, TMPDIR=str(temporary), PYTHONIOENCODING="utf-8", **{MARK: mark})
    with out.open("w", encoding="utf-8") as stdout, err.open("w", encoding="utf-8") as stderr:
        return subprocess.Popen([sys.executable, str(CHILD), str(work), mode], stdout=stdout,
                                stderr=stderr, env=env, start_new_session=True)


@pytest.mark.parametrize("gap_s", [0.005, 0.150])
@pytest.mark.parametrize("tool", ["measure_coverage", "run_tests"])
def test_two_ctrl_c_during_a_coverage_tool_leave_nothing(tmp_path, tool, gap_s):
    pytest.importorskip("langadapter").CancelScope
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    mark = uuid.uuid4().hex
    out, err = tmp_path / "out.txt", tmp_path / "err.txt"
    child = _run(tmp_path, work, temporary, mark, tool, out, err)
    try:
        assert _wait_for(out, "RUNNING", child, 300), err.read_text(encoding="utf-8")[-3000:]
        # The suite's test is running.
        time.sleep(2.0)
        started = time.monotonic()
        os.killpg(child.pid, signal.SIGINT)
        assert _wait_for(out, "CANCELLING", child, 30), err.read_text(encoding="utf-8")[-3000:]
        time.sleep(gap_s)
        os.killpg(child.pid, signal.SIGINT)
        status = child.wait(timeout=300)
        stopped_in = time.monotonic() - started
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
    errors = err.read_text(encoding="utf-8")
    assert status == 130, errors[-3000:]
    assert "remains unfinished" not in errors, errors[-3000:]
    assert "FINISHED" not in out.read_text(encoding="utf-8")
    # The 10 s test was stopped, not waited for.
    assert stopped_in < 8.0, stopped_in
    time.sleep(1.0)
    assert _processes_of(mark, tmp_path, set()) == []
    assert sorted(path.name for path in temporary.iterdir()) == []
    # A later run in the same directories completes.
    out2, err2 = tmp_path / "out2.txt", tmp_path / "err2.txt"
    again = _run(tmp_path, work, temporary, mark, "finish", out2, err2)
    assert again.wait(timeout=300) == 0, err2.read_text(encoding="utf-8")[-3000:]
    assert "FINISHED" in out2.read_text(encoding="utf-8")
