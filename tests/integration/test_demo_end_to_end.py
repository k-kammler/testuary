"""``testuary --demo`` end to end, each run in a fresh process from an empty
working directory with the package's variables cleared: it consumes the
shipped scenario exactly and reproduces what the recorded run delivered, as
its manifest records it, and a changed input stops it naming the call."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from tool_fixtures import same_code_environment

pytestmark = pytest.mark.slow


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()


def _sources(files: dict) -> dict:
    """The delivered suite's test and setup sources. The matrices and the
    copy of the project's pytest settings are by-products of real test runs,
    which a playback does not make."""
    return {name: digest for name, digest in files.items()
            if name.endswith((".py", ".java", ".ini"))}


def _delivered(results: Path) -> dict:
    """The suite the first run delivered, from its run directory, where the
    recorder reads it."""
    from testuary import project

    tests = project.phase_artifact_path(results, "tests", 0)
    return _sources({path.name: _sha(path) for path in sorted(tests.iterdir())
                     if path.is_file() and path.name != "DEMO_NOTICE.txt"})


def _headline(run: Path, columns) -> dict:
    """The results table's ``columns``, as the recording keeps them."""
    import csv

    with open(run / "results" / "tests.csv", encoding="utf-8-sig", newline="") as handle:
        row = next(csv.DictReader(handle))
    return {name: row.get(name, "") for name in columns}


def _demo(tmp_path: Path, name: str):
    cwd = tmp_path / f"cwd_{name}"
    cwd.mkdir()
    out = tmp_path / f"out_{name}"
    proc = subprocess.run(
        [sys.executable, "-m", "testuary", "--demo", "--output-dir", str(out)],
        cwd=cwd, env=same_code_environment(), capture_output=True, text=True,
        timeout=3600,
    )
    return proc, out


def test_the_demonstration_reproduces_the_recorded_run(tmp_path):
    from testuary.demo import cli as demo_cli

    manifest, _files = demo_cli.load_data()
    proc, out = _demo(tmp_path, "a")
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    # The report before the run names no API key: the playback is free.
    shown = proc.stdout + proc.stderr
    assert "The playback starts now. It sends no model request and costs nothing." in shown
    assert "api key" not in shown.lower() and "cost money" not in shown
    # A playback writes into a new directory: nothing saved is continued.
    assert "(a new run)" in shown and "continues an earlier run" not in shown
    result = json.loads((out / "DEMO.json").read_text(encoding="utf-8"))["result"]
    assert result["outcome"] == "completed"
    assert result["answers"] == manifest["entries"]
    assert result["classes"] == manifest["classes"]
    # The set-cover selections come from the scenario: nothing is started.
    assert result["processes"] == []
    assert result["network_connections"] == 0

    workspace = manifest["exemplar"]["workspace"]
    results = out / "run" / "results" / workspace
    recorded = manifest["recorded"]
    assert _delivered(results) == _sources(recorded["delivered_suite"])
    assert _headline(out / "run", recorded["results"]) == recorded["results"]
    for path in (out / "run").rglob("*.json"):
        assert json.loads(path.read_text(encoding="utf-8"))["testuary_demo"], path
    ledger = (out / "demo" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(ledger) == manifest["entries"]

    # A second run answers the same calls and delivers the same suite; only
    # the order of concurrent calls may differ.
    proc2, out2 = _demo(tmp_path, "b")
    assert proc2.returncode == 0, proc2.stdout[-3000:] + proc2.stderr[-3000:]
    assert _delivered(out2 / "run" / "results" / workspace) == _sources(recorded["delivered_suite"])
    ledger2 = (out2 / "demo" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    assert sorted(ledger2) == sorted(ledger)


_STALE_PROBE = r'''
import json, os, sys
from pathlib import Path
from testuary.demo import cli as demo_cli
from testuary.demo.scenario import Scenario

work = Path(sys.argv[1])
manifest, files = demo_cli.load_data()
preset = demo_cli._write_workspace(work / "workspace", manifest, files)
(work / "scenario.json.gz").write_bytes(files["scenario.json.gz"])
scenario = Scenario.load(work / "scenario.json.gz")
first = min(e.key["seed"] for e in scenario.entries if e.kind == "model.seed")
changed = 0
for entry in scenario.entries:
    if entry.kind == "model.seed" and entry.key["seed"] == first:
        text = entry.answer["text"]
        # One changed byte in a merge input: a test of the first seed.
        entry.answer["text"] = text.replace("def test_", "def tesT_", 1)
        changed += entry.answer["text"] != text
assert changed == 1
os.chdir(work / "workspace")
os.environ["TESTUARY_CONFIG_FILE"] = str(preset)
os.environ["TESTUARY_HOME"] = str(work / "workspace")
inv = manifest["invocation"]
project = work / "workspace" / "projects" / manifest["exemplar"]["instance"]
stop = demo_cli.play(scenario, [
    "testuary", "--config", str(preset), "--model", inv["model"],
    "--project-name", inv["project"], "--source-dir", str(project),
    "--source-files", inv["source_files"], "--output-dir", str(work / "run"), "--resume",
])
print(json.dumps(None if stop is None else
                 {"kind": stop.call.kind, "reason": stop.reason}))
'''


def test_a_changed_input_stops_the_demonstration_naming_the_call(tmp_path):
    probe = tmp_path / "stale_probe.py"
    probe.write_text(_STALE_PROBE, encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    proc = subprocess.run(
        [sys.executable, str(probe), str(work)], env=same_code_environment(),
        capture_output=True, text=True, timeout=1800,
    )
    assert proc.returncode == 0, proc.stdout[-3000:] + proc.stderr[-3000:]
    stop = json.loads(proc.stdout.strip().splitlines()[-1])
    assert stop is not None
    assert stop["kind"].startswith(("backend.", "model.", "mutation."))
    assert "regenerated" in stop["reason"]
