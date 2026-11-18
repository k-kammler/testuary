"""Model input outside the tool-description and Python prompt snapshots, byte for byte.

test_model_surface_snapshot covers tool and result descriptions, and
test_prompts_loader the Python prompts with built-in replacement guidance off.
This file covers the Python additions under the package defaults, the prompts
resolved for Java and for a language without a backend, the filled import
hints, and the bound tools' schemas without descriptions (the surface
snapshot holds them). REGEN_MODEL_SURFACE=1 regenerates the fixture.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
from langadapter.context import ProjectContext
from langadapter.java.adapter import JavaAdapter
from langchain_core.utils.function_calling import convert_to_openai_tool

import test_model_surface_snapshot as surface
from testuary.agent import _langgraph as lg
from testuary import prompts
from testuary.evidence import composition_digest
from testuary import languages
from testuary.languages import get_backend
from testuary.languages.generic import GenericBackend

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "model_input_snapshot.json"
AGENTS = ("test_initial", "test_coverage", "test_mutation")


def _without_descriptions(value):
    if isinstance(value, dict):
        return {key: _without_descriptions(item) for key, item in value.items() if key != "description"}
    if isinstance(value, list):
        return [_without_descriptions(item) for item in value]
    return value


def _tool_schemas(tmp_path):
    schemas = {}
    for name, build in surface._BUILDERS.items():
        (tmp_path / name).mkdir()
        agent = build(tmp_path / name)
        reads, writes, run_tool, measure_tool = agent.phase_tools()
        tools = [*reads, *writes, run_tool, measure_tool,
                 lg._make_final_answer_tool(agent.result_schema, agent.final_tool_name)]
        schemas[name] = {tool.name: _without_descriptions(convert_to_openai_tool(tool)["function"])
                         for tool in tools}
    return schemas


def _resolved(language):
    return {agent: prompts.resolve_agent_prompts(agent, language, runtime_version="fixture")
            for agent in AGENTS}


def _import_hints(tmp_path):
    """The starter-suite prompt's filled ``$import_hint`` for a Python module in a
    package, one named like a standard-library module, and a Java class in a
    package and in the default package. The template holds only the slot."""
    from testuary.starters import _import_hint

    root = tmp_path / "hints"
    module = root / "pkg" / "mod.py"
    module.parent.mkdir(parents=True)
    (module.parent / "__init__.py").write_text("", encoding="utf-8")
    module.write_text("def f():\n    return 1\n", encoding="utf-8")
    shadowing = module.parent / "types.py"
    shadowing.write_text("def g():\n    return 2\n", encoding="utf-8")
    packaged = root / "src" / "main" / "java" / "com" / "example" / "Calc.java"
    packaged.parent.mkdir(parents=True)
    packaged.write_text("package com.example;\n\nclass Calc {}\n", encoding="utf-8")
    plain = root / "Plain.java"
    plain.write_text("class Plain {}\n", encoding="utf-8")
    return {
        "python": _import_hint("python", str(module), str(root)),
        "python_standard_library_name": _import_hint("python", str(shadowing), str(root)),
        "java_package": _import_hint("java", str(packaged), str(root)),
        "java_default_package": _import_hint("java", str(plain), str(root)),
    }


def _fixed_java(tmp_path, monkeypatch):
    """Java's backend with an adapter that reports fixed facts instead of a
    JDK's, for the rest of the test."""
    facts = SimpleNamespace(test_framework="JUnit 5", timeout_marker="", runtime_version="21")

    class FixedFacts(JavaAdapter):
        """Java's adapter reporting fixed facts instead of a JDK's."""

        def language_facts(self):
            return facts

    project = FixedFacts(ProjectContext(str(tmp_path)))

    class FixedJava(GenericBackend):
        def project_backend(self, source_dir=None):
            return project

    monkeypatch.setitem(languages._ADAPTERS, "java", FixedJava("java", (".java",)))


def _render(tmp_path, monkeypatch):
    _fixed_java(tmp_path, monkeypatch)
    python = _resolved("python")
    return {
        "python_defaults": {
            "builtin_replacement_rule": get_backend("python").prompt_fragments()["lang_builtin_replacement_rule"],
            "stands_alone_messages": {
                agent: {key: text for key, text in python[agent].get("tool_messages", {}).items()
                        if key.startswith("tests_stands_alone_")}
                for agent in AGENTS},
        },
        "java_prompts": _resolved("java"),
        "tool_schemas": _tool_schemas(tmp_path),
        "import_hints": _import_hints(tmp_path),
    }


#: The prompt_composition digests of Python's and Java's coverage and
#: mutation prompts as this file resolves them. A run's generation origin
#: records the digest and a resume compares it, so another digest refuses
#: the resume of an earlier run's folder at these phases.
RECORDED_COMPOSITIONS = {
    ("python", "test_coverage"): "3656db0e0942d9717e4c84e758f44fbcaaab05619df85019d6ab657917dfeb44",
    ("python", "test_mutation"): "a8a34d58e579652ba5b619228eca2165a53df5e56da1983e7bb634134b26fc12",
    ("java", "test_coverage"): "a3d2136ad61b4e2783d700bd70961dd95925b240c7894eefd2ae6e08665610e9",
    ("java", "test_mutation"): "2afe711f9cea9ca3f32e41757890da7716e62948781154960e2e982f13334be7",
}


def test_python_and_java_prompt_compositions_keep_their_recorded_digests(tmp_path, monkeypatch):
    """The coverage and mutation phases of a Python or Java folder written
    before resume with the same prompt composition."""
    _fixed_java(tmp_path, monkeypatch)
    digests = {}
    for language, agent in RECORDED_COMPOSITIONS:
        resolved = prompts.resolve_agent_prompts(agent, language, runtime_version="fixture")
        digests[language, agent] = composition_digest({key: resolved[key] for key in (
            "system_prompt", "user_prompt_template", "tool_messages")})
    assert digests == RECORDED_COMPOSITIONS


def test_model_input_is_byte_identical_to_snapshot(tmp_path, monkeypatch):
    rendered = json.loads(json.dumps(_render(tmp_path, monkeypatch)))
    if os.environ.get(surface.REGEN_ENV) == "1":
        FIXTURE.write_text(json.dumps(rendered, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        pytest.skip(f"regenerated {FIXTURE.name}")
    expected = json.loads(FIXTURE.read_text(encoding="utf-8"))
    for part in expected:
        assert rendered[part] == expected[part], (
            f"{part}: model input changed. A deliberate change of model input "
            f"regenerates with {surface.REGEN_ENV}=1 and commits the fixture in the same change.")
    assert set(rendered) == set(expected)
