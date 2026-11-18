"""``testuary setup --demo`` end to end in its own process: the wizard on
the demo's project, then the playback of the recorded run."""
from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import pytest

from tool_fixtures import same_code_environment


def _setup_demo(*arguments: str, cwd: Path, timeout: float = 300) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-B", "-m", "testuary", "setup", "--demo", *arguments],
                          cwd=cwd, env=same_code_environment(), capture_output=True, text=True,
                          encoding="utf-8", errors="replace", timeout=timeout,
                          stdin=subprocess.DEVNULL)


def _played(out: Path, result: subprocess.CompletedProcess, name: str) -> None:
    assert result.returncode == 0, result.stdout[-3000:] + result.stderr[-3000:]
    project = (out / "workspace" / "projects" / name).resolve()
    assert f"Run it in {project} with: testuary" in result.stdout
    assert "Demo complete:" in result.stdout
    record = json.loads((out / "demo" / "DEMO.json").read_text(encoding="utf-8"))
    assert record["result"]["outcome"] == "completed"
    assert record["result"]["processes"] == [] and record["result"]["network_connections"] == 0
    assert "[project]" in (project / ".testuary" / "project.ini").read_text(encoding="utf-8")
    # The answer ledger is for the detailed views: the setup demo shows none.
    assert not any(line.startswith("[demo] ") for line in result.stdout.splitlines())


@pytest.mark.slow
def test_the_java_setup_demo_sets_up_the_project_and_plays_its_run(tmp_path):
    out = tmp_path / "out"
    _played(out, _setup_demo("--demo-language", "java", "--output-dir", str(out), cwd=tmp_path),
            "jsonassert")


@pytest.mark.slow
def test_named_speed_types_the_answers_visibly_and_paces_the_run(tmp_path):
    out = tmp_path / "out"
    began = time.monotonic()
    result = _setup_demo("--demo-language", "java", "--demo-speed", "25", "--demo-playback-seconds", "50",
                         "--output-dir", str(out), cwd=tmp_path)
    _played(out, result, "jsonassert")
    assert "Playback: paced, at least about 2 seconds." in result.stdout
    assert time.monotonic() - began >= 2


@pytest.mark.slow
def test_the_python_setup_demo_sets_up_the_project_and_plays_its_run(tmp_path):
    out = tmp_path / "out"
    _played(out, _setup_demo("--output-dir", str(out), cwd=tmp_path, timeout=600),
            "tenacity")
