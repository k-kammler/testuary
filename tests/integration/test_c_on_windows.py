"""C on native Windows: a run is refused before any phase, and the refusal
names Linux and WSL, not a missing compiler."""
import subprocess
import sys

import pytest

from tool_fixtures import same_code_environment

pytestmark = pytest.mark.skipif(sys.platform != "win32",
                                reason="the C adapter's refusal names WSL on native Windows only")


def test_generate_for_c_on_native_windows_names_linux_and_wsl(tmp_path):
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    # A model at a closed port: the run is refused before any model request.
    (home / "models.ini").write_text(
        "[closed]\nprovider = ollama\nmodel = closed-model\napi_url = http://127.0.0.1:9/v1\n",
        encoding="utf-8")
    (project / "calc.c").write_text("int add(int a, int b) { return a + b; }\n", encoding="utf-8")
    environment = {**same_code_environment(), "TESTUARY_HOME": str(home)}
    result = subprocess.run(
        [sys.executable, "-m", "testuary", "generate", "--model", "closed", "--source-files", "calc.c",
         "--stop-after-phase", "starter", "--no-color"],
        cwd=project, env=environment, capture_output=True, text=True, timeout=120)
    output = result.stdout + result.stderr
    assert result.returncode == 2, output
    assert "runs on Linux only" in output and "WSL" in output, output
    assert "no project runtime is available" not in output
    assert "No phase ran." in output
