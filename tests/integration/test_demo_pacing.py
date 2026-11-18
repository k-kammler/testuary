"""Paced demonstration playback changes the presentation only.

The recorded Java demonstration runs twice through the public entry point,
instantly and paced, into the same run directory one after the other, since a
record can hold the hash of a text that names the directory. Its summary, its
ledger and every record must agree once what any two runs differ in is set
aside: random identifiers, clock readings and durations, and the run's own
timing diagnostics (a model request that takes more than a second is logged
with its duration, in the run log and as a console record).
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tool_fixtures import same_code_environment

# A run summary's "time" holds where the time went, all durations.
_TIME_KEY = re.compile(r"(t_mono|clock_domain|elapsed|duration|_ms$|seconds|timestamp|_s$|^time$)",
                       re.I)
_ID = re.compile(r"\b[0-9a-f]{32}\b|\b[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}\b")
_TEMP = re.compile(r"\btmp[a-z0-9_]{8}")
_DURATION = re.compile(r"\b\d+(?:\.\d+)?s\b")
_CLOCK = re.compile(r"^\d+:\d\d:\d\d\.\d+ ")
_SLOW_REQUEST = "chat <S>"


def _testuary(*arguments: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", "-m", "testuary", *arguments],
                          cwd=cwd, env=same_code_environment(), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=180,
                          stdin=subprocess.DEVNULL)


def _text(value: str, root: str) -> str:
    for form in (root, root.replace("\\", "/"), root.replace("\\", "\\\\")):
        value = value.replace(form, "<OUT>")
    return _DURATION.sub("<S>", _TEMP.sub("<TMP>", _ID.sub("<ID>", value)))


def _untimed(value):
    if isinstance(value, dict):
        return {k: _untimed(v) for k, v in value.items() if not isinstance(v, (int, float))}
    return value


def _normal(value, root: str):
    if isinstance(value, dict) and any("clock_domain" in key for key in value):
        value = _untimed(value)  # a timing record: its numbers are durations
    if isinstance(value, dict):
        return {k: _normal(v, root) for k, v in value.items() if not _TIME_KEY.search(k)}
    if isinstance(value, list):
        items = [_normal(v, root) for v in value]
        return [item for item in items
                if not (isinstance(item, dict) and item.get("message") == _SLOW_REQUEST)]
    return _text(value, root) if isinstance(value, str) else value


def _record(path: Path, root: str):
    raw = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        return _normal(json.loads(raw), root)
    if path.suffix == ".jsonl":
        lines = [json.loads(line) for line in raw.splitlines() if line.strip()]
        return sorted(json.dumps(line, sort_keys=True) for line in _normal(lines, root))
    if path.suffix == ".csv":
        rows = list(csv.reader(io.StringIO(raw)))
        keep = [i for i, name in enumerate(rows[0]) if not _TIME_KEY.search(name)] if rows else []
        return [[_text(row[i], root) for i in keep if i < len(row)] for row in rows]
    lines = [_CLOCK.sub("", _text(line, root)) for line in raw.splitlines()]
    return [line for line in lines if line.strip() != _SLOW_REQUEST]


def _records(root: Path, played_in: Path) -> dict:
    """The records under ``root``, of a playback into ``played_in``."""
    return {_ID.sub("<ID>", path.relative_to(root).as_posix()): _record(path, str(played_in))
            for path in sorted(root.rglob("*")) if path.is_file()}


def _without_playback(records: dict) -> dict:
    """The records without the one value that names the playback itself."""
    records = dict(records)
    if isinstance(records.get("DEMO.json"), dict):
        record = dict(records["DEMO.json"])
        record["result"] = {k: v for k, v in record["result"].items() if k != "playback"}
        records["DEMO.json"] = record
    if "DEMO.md" in records:
        records["DEMO.md"] = [line for line in records["DEMO.md"]
                              if not line.startswith("- playback:")]
    return records


def _summary(result: subprocess.CompletedProcess, root: str) -> list:
    """The end-of-run summary and the demonstration's closing lines."""
    lines = result.stderr.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("Run finished"))
    summary = [line for line in lines[start:]
               if line.strip() and not re.match(r"^\d\d:\d\d:\d\d ", line)]
    closing = result.stdout.splitlines()
    end = next(i for i, line in enumerate(closing) if line.startswith("Demo complete"))
    return [_text(line, root) for line in summary + closing[end:]]


@pytest.mark.slow
def test_paced_and_instant_playback_write_the_same_summary_and_records(tmp_path):
    out, instant, paced = tmp_path / "out", tmp_path / "instant", tmp_path / "paced"
    plain = _testuary("--demo", "--demo-language", "java", "--output-dir", str(out), cwd=tmp_path)
    assert plain.returncode == 0, plain.stdout[-2000:] + plain.stderr[-2000:]
    out.rename(instant)
    began = time.monotonic()
    # Redirected output plays instantly unless a speed is named: 50 seconds
    # at 25 times their speed is about two seconds of pauses.
    slow = _testuary("--demo", "--demo-language", "java", "--demo-speed", "25",
                     "--demo-playback-seconds", "50", "--output-dir", str(out), cwd=tmp_path)
    paced_seconds = time.monotonic() - began
    assert slow.returncode == 0, slow.stdout[-2000:] + slow.stderr[-2000:]
    out.rename(paced)
    assert "Playback: paced, at least about 2 seconds." in slow.stdout
    assert "Playback:" not in plain.stdout
    assert "- playback: instant" in (instant / "DEMO.md").read_text(encoding="utf-8")
    assert "- playback: paced, at least about 2 seconds (50 s at speed 25)" in (
        paced / "DEMO.md").read_text(encoding="utf-8")
    # The playback lasts at least its target, about two seconds here. The run's
    # own time counts toward it, so a slow machine adds no pauses beyond it.
    assert paced_seconds >= 0.9 * 50 / 25
    summary = _summary(plain, str(out))
    assert summary == _summary(slow, str(out))
    assert any(line.startswith("  Line coverage: ") and "[demo]" in line for line in summary)
    assert _without_playback(_records(paced, out)) == _without_playback(_records(instant, out))


def test_the_pacing_options_require_the_demonstration(tmp_path):
    for arguments in (["--demo-speed", "0.5"], ["--demo-playback-seconds", "60"], ["--demo-step"]):
        result = _testuary(*arguments, "--source-files", "x.py", cwd=tmp_path)
        assert result.returncode == 2 and "require --demo" in result.stderr, result.stderr
    result = _testuary("check", "--demo-playback-seconds", "60", cwd=tmp_path)
    assert result.returncode == 2 and "check does not run a demonstration" in result.stderr


def test_an_unusable_speed_or_a_step_without_a_terminal_is_a_usage_error(tmp_path):
    result = _testuary("--demo", "--demo-speed", "fast", "--output-dir", str(tmp_path / "a"),
                       cwd=tmp_path)
    assert result.returncode == 2 and "unknown demonstration speed" in result.stderr
    result = _testuary("--demo", "--demo-playback-seconds", "0", "--output-dir", str(tmp_path / "a"),
                       cwd=tmp_path)
    assert result.returncode == 2 and "--demo-playback-seconds must be a positive number" in result.stderr
    result = _testuary("--demo", "--demo-step", "--output-dir", str(tmp_path / "b"), cwd=tmp_path)
    assert result.returncode == 2 and "--demo-step needs an interactive terminal" in result.stderr
    assert not (tmp_path / "a").exists() and not (tmp_path / "b").exists()


@pytest.mark.slow
@pytest.mark.skipif(os.name == "nt", reason="sends SIGINT to a child process")
def test_ctrl_c_ends_a_paced_playback_at_once(tmp_path):
    output = tmp_path / "demo"
    process = subprocess.Popen(
        [sys.executable, "-B", "-m", "testuary", "--demo", "--demo-language", "java",
         "--demo-speed", "1", "--output-dir", str(output)],
        cwd=tmp_path, env=same_code_environment(), stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    # Into the paced playback, whose whole length is at least 120 seconds.
    time.sleep(4)
    began = time.monotonic()
    process.send_signal(signal.SIGINT)
    stdout, stderr = process.communicate(timeout=30)
    assert time.monotonic() - began < 3, stdout[-2000:] + stderr[-2000:]
    assert process.returncode == 130
    record = json.loads((output / "DEMO.json").read_text(encoding="utf-8"))
    assert record["result"]["outcome"] == "cancelled"


# Runs the playback with the pacer's waits counted and summed, printed at exit.
_COUNTING = """
import atexit, runpy, sys
from testuary.demo import pacing
seen = []
wait = pacing.Pacer.on_event
def on_event(self, event):
    seen.append(self.share)
    return wait(self, event)
pacing.Pacer.on_event = on_event
atexit.register(lambda: print("OBSERVED", len(seen), sum(seen), file=sys.__stderr__))
sys.argv = ["testuary", *sys.argv[1:]]
runpy.run_module("testuary", run_name="__main__")
"""


def test_every_installed_data_set_has_its_observation_count():
    from testuary.demo import cli, pacing

    assert set(pacing.OBSERVATIONS) == set(cli.datasets())


@pytest.mark.slow
@pytest.mark.parametrize("data", ["python", "java", "toy", "tutorial"])
def test_a_paced_playback_splits_its_length_evenly_over_its_observations(tmp_path, data):
    from testuary.demo import pacing

    chosen = ([] if data == "python" else
              ["--demo-data", data] if data == "tutorial" else ["--demo-language", data])
    result = subprocess.run(
        [sys.executable, "-B", "-c", _COUNTING, "--demo", *chosen, "--demo-speed", "1000",
         "--demo-playback-seconds", "2", "--display", "plain", "--output-dir", str(tmp_path / "demo")],
        cwd=tmp_path, env=same_code_environment(), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=180, stdin=subprocess.DEVNULL)
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    observed = re.search(r"OBSERVED (\d+) (\S+)", result.stderr)
    # The playback shows as many observations as its share was made for, and
    # the shares, all equal, add up to its length at this speed.
    assert int(observed.group(1)) == pacing.OBSERVATIONS[data]
    assert float(observed.group(2)) == pytest.approx(2 / 1000)
