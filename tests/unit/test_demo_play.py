"""testuary.demo.play: the offline demonstration from Python, each call in a
process of its own."""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import testuary
from testuary import demo


class _Process:
    def __init__(self, statuses):
        self.statuses = list(statuses)

    def wait(self):
        status = self.statuses.pop(0)
        if isinstance(status, BaseException):
            raise status
        return status


def _started(monkeypatch, *statuses):
    calls = []

    def popen(command):
        calls.append(command)
        return _Process(statuses or [0])

    monkeypatch.setattr(demo.subprocess, "Popen", popen)
    return calls


def test_play_runs_testuary_demo_in_a_process_of_its_own(monkeypatch, tmp_path):
    calls = _started(monkeypatch, 0)
    assert demo.play("toy", output_dir=tmp_path / "out", speed="instant", duration=30) == 0
    command, = calls
    # The playback imports the Testuary that play came from.
    root = str(Path(testuary.__file__).resolve().parents[1])
    assert command[:2] == [sys.executable, "-c"] and command[3] == root
    assert command[4:] == ["--demo", "--output-dir", str(tmp_path / "out"), "--demo-speed",
                           "instant", "--demo-language", "toy", "--demo-playback-seconds", "30"]


def _module_path(tmp_path, pythonpath):
    """The module path a playback process starts its command line with,
    for a Testuary in ``tmp_path / "root"`` whose command prints it."""
    package = tmp_path / "root" / "testuary"
    package.mkdir(parents=True, exist_ok=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "__main__.py").write_text(
        "import json, sys\n\n\ndef main():\n    print(json.dumps(sys.path))\n    return 0\n",
        encoding="utf-8")
    environment = dict(os.environ, PYTHONPATH=os.pathsep.join(pythonpath))
    result = subprocess.run([sys.executable, "-c", demo._COMMAND, str(package.parent)],
                            env=environment, capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def test_the_playback_puts_its_dir_first_only_when_python_does_not_search_it(tmp_path):
    root = str(tmp_path / "root")
    # A source tree: first, ahead of every other directory.
    assert _module_path(tmp_path, [])[0] == root
    # A directory Python searches already, as it does an installed package's
    # site-packages after the standard library, stays where it is.
    other = str(tmp_path / "other")
    assert _module_path(tmp_path, [other, root])[:2] == [other, root]


def test_play_names_a_data_set_in_place_of_a_language(monkeypatch, tmp_path):
    calls = _started(monkeypatch, 3)
    assert demo.play(data="jsonassert", output_dir=tmp_path, speed=2) == 3
    command, = calls
    assert command[4:] == ["--demo", "--output-dir", str(tmp_path), "--demo-speed", "2",
                           "--demo-data", "jsonassert"]
    with pytest.raises(ValueError):
        demo.play("java", data="jsonassert", output_dir=tmp_path)


def test_ctrl_c_waits_for_the_playbacks_cleanup(monkeypatch, tmp_path):
    """The playback gets the same Ctrl+C and ends with its own status."""
    _started(monkeypatch, KeyboardInterrupt(), 130)
    assert demo.play(output_dir=tmp_path) == 130
