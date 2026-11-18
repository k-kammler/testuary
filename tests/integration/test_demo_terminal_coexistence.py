"""Public offline playback with scoped diagnostics on a real Linux terminal."""
from __future__ import annotations

import errno
import json
import os
from pathlib import Path
import select
import subprocess
import sys
import time

import pytest

from test_demo_java_end_to_end import _scientific_projection
from tool_fixtures import same_code_environment


_CHILD = r'''
import json
import os
from pathlib import Path
import sys
from dataclasses import asdict
from io import StringIO

from rich.console import Console
from testuary import __main__ as entry, presentation, _progress

output = Path(sys.argv[1])
mode = sys.argv[2]
original_presenter = presentation.TerminalPresenter
original_streams = sys.stdout, sys.stderr
original_writer = _progress._DIAGNOSTIC_WRITER
os.environ.pop("JAVA_HOME", None)
os.environ["PATH"] = ""
original_environment = dict(os.environ)
original_directory = Path.cwd()
frames, writes, markers, presenters = [], [], [], []


class CapturePresenter(original_presenter):
    def __init__(self, display, **kwargs):
        super().__init__(display, **kwargs)
        assert self.mode == mode
        if mode == "compact":
            assert sys.stdout.isatty() and sys.stderr.isatty()
            assert self.console.is_terminal and not self.console.is_dumb_terminal
            assert (self.console.width, self.console.height) == (100, 35)
        presenters.append(self)
        self._checked_phases = set()

    def write_diagnostic(self, write):
        record = {"index": len(writes), "live_before": self._live is not None,
                  "mode_before": self.mode, "completed": False}
        writes.append(record)
        try:
            super().write_diagnostic(write)
            record["completed"] = True
        finally:
            record["live_after"] = self._live is not None
            record["mode_after"] = self.mode

    def __call__(self, event):
        super().__call__(event)
        if (event.phase in {"starter", "coverage", "mutation"}
                and event.state == "running"
                and event.operation.startswith("phase/run-")
                and event.phase not in self._checked_phases):
            from testuary import orchestrator

            self._checked_phases.add(event.phase)
            assert _progress._DIAGNOSTIC_WRITER == self.write_diagnostic
            # A display shows warnings and errors, not ordinary log lines.
            marker = f"WARNING: COEXISTENCE/{event.phase}/DIAGNOSTIC"
            before = len(writes)
            previous_live = self._live
            orchestrator.log_print(marker)
            assert len(writes) == before + 1
            if mode == "compact":
                assert previous_live is not None and self._live is not previous_live
                assert self._live is not None
            markers.append({"text": marker, "write_index": before})
        mirror = Console(file=StringIO(), width=100, height=35, record=True,
                         color_system=None, highlight=False, markup=False)
        mirror.print(self.frame())
        frames.append({"event": asdict(event), "projection": mirror.export_text(clear=False),
                       "html": mirror.export_html(clear=False)})


presentation.TerminalPresenter = CapturePresenter
sys.argv = ["testuary", "--demo", "--demo-language", "java",
            "--display", mode, "--no-color", "--output-dir", str(output)]
try:
    entry._main()
except SystemExit as exc:
    status = exc.code
else:
    raise AssertionError("public CLI did not exit")
finally:
    presentation.TerminalPresenter = original_presenter

report = {"status": status, "mode": mode, "stdout_isatty": sys.stdout.isatty(),
          "stderr_isatty": sys.stderr.isatty(), "writes": writes, "markers": markers,
          "callback_restored": _progress._DIAGNOSTIC_WRITER is original_writer,
          "streams_preserved": (sys.stdout, sys.stderr) == original_streams,
          "environment_restored": dict(os.environ) == original_environment,
          "directory_restored": Path.cwd() == original_directory,
          "presenters_closed": all(p._closed and p._live is None for p in presenters),
          "frames": frames,
          "origins": {name: str(Path(module.__file__).resolve())
                      for name, module in sorted(sys.modules.items())
                      if name.startswith(("testuary", "langadapter"))
                      and getattr(module, "__file__", None)}}
output.with_suffix(".terminal.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
output.with_suffix(".frame.html").write_text(frames[-1]["html"], encoding="utf-8")
assert status == 0
assert len(presenters) == 1
assert len(markers) == 3
assert all(report[name] for name in ("callback_restored", "streams_preserved",
                                   "environment_restored", "directory_restored",
                                   "presenters_closed"))
raise SystemExit(status)
'''


def _command(output: Path, mode: str) -> list[str]:
    return [sys.executable, "-B", "-c", _CHILD, str(output), mode]


def _pty_demo(output: Path) -> bytes:
    import fcntl
    import pty
    import struct
    import termios

    master, slave = pty.openpty()
    process = None
    chunks = []
    try:
        fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", 35, 100, 0, 0))
        environment = same_code_environment()
        environment["TERM"] = "xterm-256color"
        environment["COLUMNS"] = "100"
        environment["LINES"] = "35"
        environment.pop("FORCE_COLOR", None)
        process = subprocess.Popen(_command(output, "compact"), stdin=slave,
                                   stdout=slave, stderr=slave, env=environment)
        os.close(slave)
        slave = None
        deadline = time.monotonic() + 120
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("offline terminal check exceeded 120 seconds")
            ready, _, _ = select.select([master], [], [], min(remaining, 1))
            if not ready:
                continue
            try:
                data = os.read(master, 65536)
            except OSError as exc:
                if exc.errno != errno.EIO:
                    raise
                break
            if not data:
                break
            chunks.append(data)
        status = process.wait(timeout=5)
        assert status == 0, b"".join(chunks).decode("utf-8", errors="replace")
    finally:
        if process is not None and process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        os.close(master)
        if slave is not None:
            os.close(slave)
        output.with_suffix(".pty").write_bytes(b"".join(chunks))
    return b"".join(chunks)


def _check_output(output: Path) -> dict:
    report = json.loads(output.with_suffix(".terminal.json").read_text(encoding="utf-8"))
    result = json.loads((output / "DEMO.json").read_text(encoding="utf-8"))["result"]
    from testuary.demo import cli as demo_cli

    manifest = demo_cli.load_data("java")[0]
    assert result["outcome"] == "completed" and result["answers"] == manifest["entries"]
    assert result["classes"] == manifest["classes"]
    assert result["processes"] == [] and result["network_connections"] == 0
    assert (output / "demo" / "audit.jsonl").read_bytes() == b""
    assert report["status"] == 0 and len(report["markers"]) == 3
    assert all(report[name] for name in ("callback_restored", "streams_preserved",
                                       "environment_restored", "directory_restored",
                                       "presenters_closed"))
    assert report["writes"] and all(write["completed"] for write in report["writes"])
    frames = report["frames"]
    assert frames and frames[-1]["event"]["state"] == "succeeded"
    assert all(frame["event"]["provenance"] == "synthetic" for frame in frames)
    assert all("demo playback" in " ".join(frame["projection"].split())
               for frame in frames)
    return report


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "linux", reason="requires Linux PTY semantics")
def test_public_demo_keeps_scoped_diagnostics_outside_compact_refresh(tmp_path):
    plain = tmp_path / "plain"
    control = subprocess.run(_command(plain, "plain"), capture_output=True,
                             env=same_code_environment(),
                             text=True, timeout=120)
    plain.with_suffix(".stdout").write_text(control.stdout, encoding="utf-8")
    plain.with_suffix(".stderr").write_text(control.stderr, encoding="utf-8")
    assert control.returncode == 0, control.stdout + control.stderr
    baseline = _check_output(plain)
    assert not baseline["stdout_isatty"] and not baseline["stderr_isatty"]
    assert "\x1b" not in control.stdout + control.stderr

    compact = tmp_path / "compact"
    raw = _pty_demo(compact)
    observed = _check_output(compact)
    assert observed["stdout_isatty"] and observed["stderr_isatty"]
    assert _scientific_projection(compact) == _scientific_projection(plain)
    for marker in observed["markers"]:
        payload = marker["text"].encode()
        assert raw.count(payload) == control.stdout.count(marker["text"]) == 1
        write = observed["writes"][marker["write_index"]]
        assert write["live_before"] and write["live_after"]
        assert write["mode_before"] == write["mode_after"] == "compact"
        position = raw.index(payload)
        assert raw.rfind(b"\x1b[?25h", 0, position) > raw.rfind(b"\x1b[?25l", 0, position)
        assert raw.find(b"\x1b[?25l", position) > position
