"""Layered progress from the actual producers, on the offline Java demonstration.

The demonstration replays recorded observations through the real pipeline, so
this checks the producers' layer events and the presenter together, without a model, JDK
or build tool. Only the events' structure is asserted.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys

import pytest

from tool_fixtures import same_code_environment

pytestmark = pytest.mark.slow


def _exemplar() -> dict:
    from testuary.demo import cli

    return cli.load_data("java")[0]["exemplar"]


_LOG_TIMESTAMP = re.compile(r"^\S*\d:\d{2}:\d{2}\.\d{3}\s")


def _run(tmp_path, display):
    environment = same_code_environment()
    environment.pop("JAVA_HOME", None)
    output = tmp_path / display
    return subprocess.run(
        [sys.executable, "-B", "-m", "testuary", "--demo", "--demo-language", "java",
         "--display", display, "--no-color", "--output-dir", str(output)],
        cwd=tmp_path, env=environment, capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=120)


def _events(stream: str) -> list[str]:
    return [line for line in stream.splitlines()
            if re.match(r"^\d{2}:\d{2}:\d{2} (RUNNING|WAITING|RETRYING|SUCCEEDED|FAILED|STOPPED)\b", line)]


def test_plain_log_reports_layers_with_totals_and_source_paths(tmp_path):
    # Loading the data here, not at import, makes a mismatch fail the tests
    # rather than the collection.
    exemplar = _exemplar()
    source = exemplar["target"]
    result = _run(tmp_path, "plain")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    events = _events(result.stderr)
    run = [line for line in events if "| Run |" in line]
    assert run and all("phases done" in line for line in run[1:])
    assert re.search(r"\| Run \| \d+ of \d+ phases done \(\d+%\)", run[-1])
    files = [line for line in events if f"| {source} | " in line]
    assert files, "events name the selected source path"
    # Progress names the source path, not the internal per-file directory name. The
    # final summary may list delivered files inside that directory.
    assert not any(exemplar["workspace"] in line for line in events)
    for phase in ("Starter phase", "Coverage phase", "Mutation phase"):
        assert any(f"| {source} | {phase}" in line and "phases (" in line for line in events), phase
    # Finishing, the fourth phase, in its step-level events, counted.
    for state, count in (("Phase started", "3 of 4 phases"), ("Phase finished", "4 of 4 phases")):
        assert any(f"| {source} | Finishing | {count}" in line and f"| {state}" in line
                   for line in events), state
    assert any("| Consolidation |" in line for line in events)
    assert any("| Checking the delivered suite |" in line for line in events)
    assert not any("total unknown" in line and "| Starter phase |" in line for line in events)
    assert any("| Starter phase |" in line and "starter suites accepted (" in line for line in events)
    assert not any("evidence saved" in line for line in events)
    assert any("| Model request 1 | Reply received. Processing it next." in line for line in events)


@pytest.mark.parametrize("display", ["plain", "detailed"])
def test_timestamped_log_lines_appear_only_at_the_debug_level(tmp_path, display):
    result = _run(tmp_path, display)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    combined = (result.stdout + result.stderr).splitlines()
    # Outside debug only the log's errors and warnings are echoed.
    assert not any(_LOG_TIMESTAMP.match(line)
                   and not any(prefix in line for prefix in ("ERROR", "WARNING", "FATAL"))
                   for line in combined)
    assert "\x1b[" not in result.stdout + result.stderr
    assert not any("Main orchestration log" in line for line in combined)
    (tmp_path / "debug-run").mkdir()
    debug = _run(tmp_path / "debug-run", "debug")
    assert debug.returncode == 0, debug.stdout[-2000:] + debug.stderr[-2000:]
    assert any(_LOG_TIMESTAMP.match(line) for line in (debug.stdout + debug.stderr).splitlines())
    assert "Main orchestration log: " in debug.stdout + debug.stderr


_FRAMES = """
import io, json, runpy, sys
from rich.console import Console
from testuary import presentation

frames = []
original = presentation.TerminalPresenter


class Recording(original):
    def __call__(self, event):
        super().__call__(event)
        snapshot = Console(file=io.StringIO(), width=120, color_system=None,
                           force_terminal=True, highlight=False, markup=False)
        with snapshot.capture() as captured:
            snapshot.print(self.frame())
        frames.append({"level": event.level, "message": event.message, "frame": captured.get()})


def factory(mode, *, no_color=False):
    console = Console(file=io.StringIO(), force_terminal=True, width=120, height=40,
                      highlight=False, markup=False)
    return Recording(mode, console=console, no_color=no_color)


presentation.TerminalPresenter = factory
sys.argv = ["testuary", "--demo", "--demo-language", "java", "--output-dir", sys.argv[1],
            "--display", "compact", "--no-color"]
try:
    runpy.run_module("testuary", run_name="__main__", alter_sys=True)
except SystemExit as stop:
    code = stop.code
else:
    code = 0
print(json.dumps({"exit": code, "frames": frames}))
"""


def test_log_location_reaches_the_display_before_the_first_file_starts(tmp_path):
    # Plain log of the real pipeline: a run line names the run log before any file starts.
    result = _run(tmp_path, "plain")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    events = _events(result.stderr)
    first_file = next(index for index, line in enumerate(events) if "| Target started" in line)
    logged = [line for line in events[:first_file] if "| Run |" in line and "| Log: " in line]
    assert logged, events[:first_file]
    log = logged[0].rsplit("| Log: ", 1)[1].strip()
    assert log.startswith(str(tmp_path / "plain")) and log.endswith("run.log")
    assert os.path.isfile(log)
    # Compact view on the same pipeline: the Log row is pinned before the first file event.
    environment = same_code_environment()
    environment.pop("JAVA_HOME", None)
    output = tmp_path / "compact"
    recorded = subprocess.run([sys.executable, "-B", "-c", _FRAMES, str(output)], cwd=tmp_path,
                              env=environment, capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=120)
    assert recorded.returncode == 0, recorded.stderr[-2000:]
    data = __import__("json").loads(recorded.stdout.splitlines()[-1])
    assert data["exit"] == 0
    frames = data["frames"]
    first_target = next(index for index, item in enumerate(frames) if item["level"] == "target")
    # The Log row names this run's log: in full when it fits the terminal,
    # else its end behind "...", which keeps the run directory's name.
    flat = ["".join(item["frame"].split()) for item in frames[:first_target]]
    log_row = "".join(f"Log{output}".split())
    end = "".join(os.path.join(output.name, "run", "logs", "run.log").split())
    assert any(("Log..." in frame or log_row in frame)
               and end + "(everystep:--displaydetailed)" in frame for frame in flat)
