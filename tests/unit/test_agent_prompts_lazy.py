"""The agents' prompts are resolved by the phase that runs them, not at import.

Resolving them reads the project's Python version, so importing the agents
must not need a Python interpreter for the project, which a Java run lacks."""
from __future__ import annotations

import os
import subprocess
import sys


def test_the_agents_import_without_a_python_interpreter_for_the_project(tmp_path):
    script = (
        "import testuary.agent.coverage_agent, testuary.agent.mutation_agent\n"
        "print('imported')\n"
        "from testuary import config, prompts, settings\n"
        "run = settings.resolve_settings(models={}, project_settings=False)\n"
        "with config.use_run_settings(run):\n"
        "    try:\n"
        "        prompts.resolve_agent_prompts('test_coverage', 'python', hints={})\n"
        "    except RuntimeError:\n"
        "        print('needed on first use')\n"
    )
    env = {**os.environ, "TESTUARY_PROJECT_PYTHON_EXECUTABLE": str(tmp_path / "absent" / "python")}
    done = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True,
                          env=env, cwd=tmp_path, timeout=120)
    assert done.returncode == 0, done.stderr
    assert done.stdout.split() == ["imported", "needed", "on", "first", "use"]
