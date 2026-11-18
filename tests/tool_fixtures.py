"""Whether a real tool the tests run is installed, and the environment
that makes a child Python run the code under test."""
from __future__ import annotations

import os
import shutil
from pathlib import Path

import langadapter
import testuary
from testuary import config
from testuary.settings import resolve_settings
from python_runtime import project_python


def cosmic_ray_installed() -> bool:
    """Whether the ``cosmic-ray`` the engine would run is installed, resolved
    the way the engine resolves it, beside the project interpreter, so an
    environment that is installed but not activated is not mistaken for a
    missing tool. Presence only, not a start: a tool that starts slowly on a
    loaded machine is installed, and one that does not start fails the test
    that runs it instead of turning it into a skip. The interpreter is the
    run's, else the one the bundled defaults and this process's variables
    select."""
    if config.active_settings() is None:
        with config.use_run_settings(resolve_settings(models={}, project_settings=False)):
            return cosmic_ray_installed()
    bindir = str(Path(project_python()).parent)
    return (shutil.which("cosmic-ray", path=bindir)
            or shutil.which("cosmic-ray")) is not None


def same_code_environment() -> dict:
    """This process's environment for a child Python, without the variables of
    Testuary, LangAdapter and the tracing services (LangChain, LangSmith,
    Langfuse), which would configure the child or send its data elsewhere.
    Its PYTHONPATH makes the child import the Testuary and LangAdapter code
    under test, not another checkout an editable install points at."""
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("TESTUARY_", "LANGADAPTER_", "LANGCHAIN_",
                                           "LANGSMITH_", "LANGFUSE_"))}
    roots = [str(Path(module.__file__).resolve().parents[1]) for module in (testuary, langadapter)]
    if environment.get("PYTHONPATH"):
        roots.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(roots)
    return environment
