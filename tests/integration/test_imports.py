"""Every listed submodule of testuary imports.

A failure usually means a stale import path, a missing dependency in
``[project.dependencies]`` or a data file missing from the wheel. Behavior
is tested under ``tests/unit/``.
"""
from __future__ import annotations

import importlib

import pytest

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


# The modules whose import this test checks, listed explicitly. The public
# interface is the one docs/reference/api.md documents.
IMPORTED_MODULES = [
    "testuary",
    "testuary.config",
    "testuary.models",
    "testuary.project",
    "testuary.results",
    "testuary.phase_state",
    "testuary.agent._helpers",
    "testuary.agent.coverage_agent",
    "testuary.agent.mutation_agent",
    "testuary.target_phases",
    "testuary.measurement",
    "testuary.languages.mutation",
    "testuary.prompts",
    "testuary._templates",
]


@pytest.mark.parametrize("module_name", IMPORTED_MODULES)
def test_the_module_imports(module_name: str) -> None:
    importlib.import_module(module_name)


def test_target_phases_defines_what_the_orchestrator_imports() -> None:
    """testuary.target_phases defines run_target_phases, TargetPhasesContext
    and ReuseFlags, which the orchestrator imports, and TestPipelineResult."""
    from testuary.target_phases import (
        run_target_phases,
        TargetPhasesContext,
        ReuseFlags,
        TestPipelineResult,
    )
    assert callable(run_target_phases)
    assert callable(TargetPhasesContext)
    assert callable(TestPipelineResult)
    # ReuseFlags has a default for every field.
    flags = ReuseFlags()
    assert hasattr(flags, "coverage")
    assert hasattr(flags, "mutation")
    assert hasattr(flags, "finishing")


def test_mutation_runs_public_api() -> None:
    from testuary import languages
    from testuary.languages.mutation import (
        MutationInstrumentFault,
        MutationMeasurement,
        MutationRunner,
        MutationSession,
    )
    assert issubclass(MutationInstrumentFault, RuntimeError)
    assert callable(MutationMeasurement.of)
    assert callable(MutationSession)
    runner = languages.get_backend("python").mutation_runner(
        {"engine": "cosmic_ray"})
    assert isinstance(runner, MutationRunner)


def test_py_typed_marker_ships() -> None:
    """The PEP 561 marker ships beside the code, so a consumer's type checker
    uses testuary's hints instead of treating the package as untyped. The wheel
    bundles the whole package directory (``packages = ["src/testuary"]``), so a
    marker present here ships."""
    from importlib.resources import files
    assert files("testuary").joinpath("py.typed").is_file()
