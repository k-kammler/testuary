"""The demonstration's process guard watches the playback, not the imports
of the libraries it runs: a library the playback imports first may start a
process when it is imported (aiohttp, which openai's client imports, calls
platform.system(), and on Windows before Python 3.12 that runs "cmd /c ver"),
and the demonstration still completes. The run is a fresh process, so
nothing is imported before it."""
from __future__ import annotations

import subprocess
import sys

import pytest

from tool_fixtures import same_code_environment

pytestmark = pytest.mark.slow

# The child: the chat model library starts one process at its first import,
# a stand-in for any library that does, and then the Python demonstration.
_PROCESS_AT_IMPORT = """
import subprocess
import sys
from importlib.abc import MetaPathFinder


class ProcessAtImport(MetaPathFinder):
    started = False

    def find_spec(self, name, path=None, target=None):
        if name == "langchain_openai" and not ProcessAtImport.started:
            ProcessAtImport.started = True
            subprocess.run([sys.executable, "-c", "pass"], check=True)
        return None


sys.meta_path.insert(0, ProcessAtImport())
sys.argv = ["testuary", "--demo", "--demo-speed", "instant", "--output-dir", sys.argv[1]]
from testuary.__main__ import main

sys.exit(main())
"""


def test_a_library_that_starts_a_process_when_imported_does_not_stop_the_demonstration(tmp_path):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    environment = same_code_environment()
    proc = subprocess.run(
        [sys.executable, "-c", _PROCESS_AT_IMPORT, str(tmp_path / "out")],
        cwd=cwd, env=environment, capture_output=True, text=True, timeout=3600,
    )
    shown = proc.stdout + proc.stderr
    assert "starts no process" not in shown, shown[-3000:]
    assert proc.returncode == 0, shown[-3000:]
    assert "Demo complete" in shown
