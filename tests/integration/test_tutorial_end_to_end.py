"""testuary tutorial plays every step: each extension point is applied with
Testuary's setup, testuary check passes on it as a command, a broken profile
stops that check, and nothing contacts a model."""
from __future__ import annotations

import io
import os
import subprocess
import sys

import pytest

from langadapter.tutorial import INSTANT, play

from testuary import tutorial
from tool_fixtures import same_code_environment

pytestmark = pytest.mark.slow


def test_the_tutorial_plays_every_step_and_its_checks_pass(tmp_path, monkeypatch, capfd):
    # The commands import the Testuary and LangAdapter under test, without the
    # variables of Testuary, LangAdapter and the tracing services.
    environment = same_code_environment()
    for name in set(os.environ) - set(environment):
        monkeypatch.delenv(name)
    monkeypatch.setenv("PYTHONPATH", environment["PYTHONPATH"])
    tutorial_dir = tmp_path / "tutorial"
    out = io.StringIO()
    assert play(tutorial.tutorial(), tutorial_dir, mode=INSTANT, out=out) == 0, out.getvalue()
    captured = capfd.readouterr()
    commands = captured.out + captured.err
    assert commands.count("No problems found by a quick check") == 3, commands
    text = out.getvalue()
    assert "Step 4 of 6 (optional): Hints from an extension" in text
    assert "ended with status" not in text
    assert f"Model: {tutorial.MODEL_ANSWERS['model.name']} via Ollama" in commands
    # The recorded run with the same alias, profile and hints plays back in
    # full, and the last step shows what it used.
    assert "Step 6 of 6 (optional): What the recorded run used" in text
    assert "prompt profile: prompt_profile.json (carried with the data)" in text
    assert "hints file: hints.json (carried with the data)" in text
    assert "The playback completed with all" in text
    assert sys.executable in text
    # A broken profile: the same check stops with the place named.
    profile = tutorial_dir / tutorial.PROFILE
    profile.write_text(profile.read_text(encoding="utf-8").replace(
        '"system_prompt": {"append": "Cover each branch with one focused test."}',
        '"no_such_part": {"append": "x"}'), encoding="utf-8")
    completed = subprocess.run(tutorial.tutorial().steps[2].command, cwd=tutorial_dir,
                               env={**os.environ, **tutorial.KEY_SETTING},
                               capture_output=True, text=True, timeout=120)
    assert completed.returncode == 2
    assert "unknown component 'no_such_part'" in completed.stdout + completed.stderr
