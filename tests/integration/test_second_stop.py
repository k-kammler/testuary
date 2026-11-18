"""A second Ctrl+C during the first one's cancel, on Linux, with real
Cosmic Ray http workers.

Both interrupts go to the process group, as a terminal sends them. The
second must not cut the cancelled measurement's cleanup short: the run
ends as interrupted, and no engine process, no session file in TMPDIR and
no worker clone of the run remains.

Marked ``slow``, like the other real Cosmic Ray runs.
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

from tool_fixtures import cosmic_ray_installed

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(sys.platform != "linux", reason="process groups and /proc are Linux's"),
    pytest.mark.mutation_tool,
]

CHILD = Path(__file__).resolve().parent / "_second_stop_child.py"
CLONES = Path("/dev/shm/cr_worker_clones")
MARK = "TESTUARY_SECOND_STOP_MARK"


def _clones():
    return set(CLONES.iterdir()) if CLONES.is_dir() else set()


def _processes_of(mark, base_dir, clones):
    """The processes that inherited the mark, or run in the test's directory
    or in one of the run's clones."""
    found = []
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            environ = (entry / "environ").read_bytes()
            cmdline = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
            cwd = os.readlink(entry / "cwd")
        except OSError:
            continue
        if (f"{MARK}={mark}".encode() in environ or str(base_dir) in cmdline
                or str(base_dir) in cwd or any(cwd.startswith(str(clone)) for clone in clones)):
            found.append(f"{entry.name}: {cmdline[:200]}")
    return found


def _wait_for(path, text, child, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if text in path.read_text(encoding="utf-8", errors="replace"):
            return True
        if child.poll() is not None:
            return False
        time.sleep(0.1)
    return False


@pytest.mark.parametrize("gap_s", [0.005, 0.150])
def test_a_second_ctrl_c_leaves_no_process_and_no_file(tmp_path, gap_s):
    """The second stop follows the first one's cancel by ``gap_s``, so it
    lands during the cancelled measurement's cleanup on any machine: timed
    from the first signal, two signals 5 ms apart may reach the process as
    one. The engine's workers carry the test's mark in their environment,
    so an orphaned worker or supervisor is found."""
    session_module = pytest.importorskip("langadapter.mutation")
    if not hasattr(session_module, "MeasurementCancelled"):
        pytest.skip("the installed LangAdapter has no measurement cancel")
    if not cosmic_ray_installed():
        pytest.skip("cosmic-ray not installed")
    temporary = tmp_path / "tmp"
    temporary.mkdir()
    work = tmp_path / "work"
    work.mkdir()
    mark = uuid.uuid4().hex
    before = _clones()
    out, err = tmp_path / "out.txt", tmp_path / "err.txt"
    env = dict(os.environ, TMPDIR=str(temporary), PYTHONIOENCODING="utf-8", **{MARK: mark})
    with out.open("w", encoding="utf-8") as stdout, err.open("w", encoding="utf-8") as stderr:
        child = subprocess.Popen([sys.executable, str(CHILD), str(work)], stdout=stdout,
                                 stderr=stderr, env=env, start_new_session=True)
        try:
            assert _wait_for(out, "MEASURING", child, 300), err.read_text(encoding="utf-8")[-3000:]
            # The engine's workers are measuring.
            time.sleep(2.0)
            os.killpg(child.pid, signal.SIGINT)
            assert _wait_for(out, "CANCELLING", child, 30), err.read_text(encoding="utf-8")[-3000:]
            time.sleep(gap_s)
            os.killpg(child.pid, signal.SIGINT)
            status = child.wait(timeout=300)
        finally:
            if child.poll() is None:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
    errors = err.read_text(encoding="utf-8")
    assert status == 130, errors[-3000:]
    assert "remains unfinished" not in errors, errors[-3000:]
    assert "NOT STOPPED" not in out.read_text(encoding="utf-8")
    run_clones = _clones() - before
    time.sleep(1.0)
    assert _processes_of(mark, tmp_path, run_clones) == []
    assert sorted(path.name for path in temporary.iterdir()) == []
    assert sorted(str(path) for path in _clones() - before) == []
