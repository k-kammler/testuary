"""The agents' prompts resolved for Python, as tests build agents with them."""
from __future__ import annotations

import tempfile
from functools import cache
from pathlib import Path

from testuary import config, prompts
from testuary.settings import resolve_settings


@cache
def python_prompts(agent: str) -> dict:
    """The prompts of ``agent`` (test_coverage or test_mutation) resolved for
    Python, once per test session, in the active run or in a run of the
    bundled defaults for an empty project."""
    if config.active_settings() is not None:
        return prompts.resolve_agent_prompts(agent, "python", hints={})
    with tempfile.TemporaryDirectory(prefix="testuary_prompts_") as project_dir:
        settings = resolve_settings(base_dir=project_dir, models={}, project_settings=False)
        with config.use_run_settings(settings):
            return prompts.resolve_agent_prompts(agent, "python", hints={})


@cache
def c_prompts(agent: str) -> dict:
    """The prompts of ``agent`` resolved for C, for a make project with one
    source file, in the active run or in a run of the bundled defaults for
    that project."""
    with tempfile.TemporaryDirectory(prefix="testuary_prompts_") as project_dir:
        root = Path(project_dir)
        (root / "Makefile").write_text("app: src/calc.o\n\t$(CC) -o $@ $^\n", encoding="utf-8")
        (root / "src").mkdir()
        (root / "src" / "calc.c").write_text("int f(void) { return 1; }\n", encoding="utf-8")
        if config.active_settings() is not None:
            return prompts.resolve_agent_prompts(agent, "c", source_dir=project_dir, hints={})
        settings = resolve_settings(base_dir=project_dir, models={}, project_settings=False)
        with config.use_run_settings(settings):
            return prompts.resolve_agent_prompts(agent, "c", source_dir=project_dir, hints={})
