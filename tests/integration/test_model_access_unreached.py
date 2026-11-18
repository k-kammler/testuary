"""A Testuary run never reaches the model access for other programs: the
recorded demonstrations play through a fresh process in which
testuary._model_access cannot be imported and models.catalogue_chat_model
fails when called."""
from __future__ import annotations

import subprocess
import sys

import pytest

from tool_fixtures import same_code_environment

pytestmark = pytest.mark.slow

_UNREACHED = """
import sys
from importlib.abc import MetaPathFinder


class Unreached(MetaPathFinder):
    def find_spec(self, name, path=None, target=None):
        if name == "testuary._model_access":
            raise ImportError("a run reached the model access for other programs")
        return None


sys.meta_path.insert(0, Unreached())
from testuary import models


def unreached(*args, **kwargs):
    raise AssertionError("a run called catalogue_chat_model")


models.catalogue_chat_model = unreached
sys.argv = ["testuary", "--demo", "--demo-language", sys.argv[2], "--demo-speed", "instant",
            "--output-dir", sys.argv[1]]
from testuary.__main__ import main

sys.exit(main())
"""


@pytest.mark.parametrize("language", ["python", "java"])
def test_a_recorded_run_never_reaches_the_model_access(tmp_path, language):
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    environment = same_code_environment()
    proc = subprocess.run(
        [sys.executable, "-c", _UNREACHED, str(tmp_path / "out"), language],
        cwd=cwd, env=environment, capture_output=True, text=True, timeout=3600,
    )
    shown = proc.stdout + proc.stderr
    assert proc.returncode == 0, shown[-3000:]
    assert "Demo complete" in shown and "model access" not in shown
