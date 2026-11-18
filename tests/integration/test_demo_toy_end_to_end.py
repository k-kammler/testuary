"""The demonstration of a language Testuary has no code for: the Toy of
``langadapter tutorial``, played through the public command. Its adapter
comes from the manifest, its executions and model answers from the
recording, and the playback starts no process and contacts no service."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

import testuary
from testuary import project
from tool_fixtures import same_code_environment


def _testuary(*arguments: str, cwd: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", "-m", "testuary", *arguments],
                          cwd=cwd, env=same_code_environment(), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=180,
                          stdin=subprocess.DEVNULL)


@pytest.mark.slow
def test_the_toy_demo_plays_every_phase_from_its_recording(tmp_path):
    out = tmp_path / "out"
    result = _testuary("--demo", "--demo-language", "toy", "--demo-speed", "instant",
                       "--display", "plain", "--output-dir", str(out), cwd=tmp_path)
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    assert "Demo complete" in result.stdout
    # The plain display writes the run's progress and summary to stderr.
    assert "calc.toy: a suite that meets the minimums" in result.stderr
    assert "Mutation score: 100% [demo]" in result.stderr
    suite = project.run_dir(out / "run" / "results" / "calc", 0) / "tests" / "test_calc.toy"
    # The suite the recorded run delivered, as the recording's manifest names it.
    manifest = json.loads((Path(testuary.__file__).parent / "demo" / "data" / "toy"
                           / "manifest.json").read_text(encoding="utf-8"))
    digest = hashlib.sha256(suite.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
    assert digest == manifest["recorded"]["delivered_suite"]["test_calc.toy"]
    # The Toy lacks the operations behind these fallbacks, as in the recorded run.
    assert "Fallback: validity screening: such tests are not rejected" in result.stderr
    played = json.loads((out / "DEMO.json").read_text(encoding="utf-8"))["result"]
    assert played["processes"] == [] and played["network_connections"] == 0
    assert (out / "demo" / "audit.jsonl").read_bytes() == b""


def test_a_language_without_demo_data_is_a_usage_error(tmp_path):
    result = _testuary("--demo", "--demo-language", "cobol", "--output-dir",
                       str(tmp_path / "out"), cwd=tmp_path)
    assert result.returncode == 2
    assert "there is no demonstration of 'cobol'. Testuary has: python, java, toy" in result.stderr
