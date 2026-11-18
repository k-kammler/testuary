"""Compare rendered agent metadata against its fixed model-surface snapshot.

The snapshot records tool names, descriptions and argument names, final-answer
metadata, result-field descriptions, loop-breaker text and framework texts in
tool results. The separate prompt snapshot checks resolved prompt text. A
fixture is regenerated only with REGEN_MODEL_SURFACE=1, since it records the
model input."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict

import pytest
from langchain_core.utils.function_calling import convert_to_openai_tool

from agent_fixtures import mutation_agent, target_agent

from testuary.agent import _langgraph as lg
from testuary.agent import _limits
from testuary.agent import mutation_agent as tm

REGEN_ENV = "REGEN_MODEL_SURFACE"
FIXTURE_NAME = "model_surface_snapshot.json"
AGENT_NAMES = ["test_coverage", "test_mutation"]


def _mutation_agent(tmp_path: Path) -> "tm.MutationAgent":
    return mutation_agent(tmp_path, target_file_path=str(tmp_path / "target.py"),
                          target_file_content="def f(x):\n    return x + 1\n",
                          mutation_test_file_name="test_target.py", coverage_baseline_pct=80.0)


_BUILDERS = {"test_coverage": target_agent, "test_mutation": _mutation_agent}


def _write_refusals(agent: Any) -> tuple[str, str]:
    """The sandbox gate's two refusals as every write tool returns them, its
    paths as placeholders: a target outside the test output directory, and
    no test output directory."""
    root = Path(agent.sandbox_root()).resolve()
    outside = root.parent / "outside.py"
    escaped = agent.sandbox_guard(outside).replace(str(outside), "<target>")
    deps = agent.deps
    agent.deps = deps.model_copy(update={"test_output_dir": ""})
    try:
        unset = agent.sandbox_guard("<target>")
    finally:
        agent.deps = deps
    return escaped.replace(str(root), "<test output dir>"), unset


def render_model_surface(agent: Any) -> Dict[str, Any]:
    """Everything the model sees from this agent besides the resolved
    prompts, from the tools ``build_graph`` binds (``phase_tools``), the
    synthetic final-answer tool, the loop-breaker action texts and the
    framework's own texts in tool results: the omission marker of every cut
    output, the answer to a failed tool call and the sandbox gate's two
    refusals of a write."""
    reads, writes, run_t, measure_t = agent.phase_tools()
    tools = reads + writes + [run_t, measure_t]
    write_refused_outside, write_refused_unset = _write_refusals(agent)

    final = lg._make_final_answer_tool(agent.result_schema, agent.final_tool_name)
    result_properties = final.args_schema.model_json_schema()["properties"]
    breaker = agent.session.loop_breaker
    return {
        "tools": {
            t.name: {"description": t.description, "args": sorted(t.args)}
            for t in tools
        },
        "final_answer": {
            "name": agent.final_tool_name,
            "description": final.description,
            "result_fields": {
                name: (field.get("description") or "")
                for name, field in result_properties.items()
            },
        },
        "loop_breaker": {
            "file_kind": breaker.file_kind,
            "edit_action_text": breaker.edit_action_text,
            "return_action_text": breaker.return_action_text,
        },
        "framework_text": {
            "tool_failure_result": lg.TOOL_FAILURE_RESULT,
            "omission_marker": _limits.marker(7, "chars", "<what>"),
            "omission_marker_counted": _limits.marker(7, "items", "<what>"),
            "omission_marker_with_rest": _limits.marker(7, "lines", "<what>", "<how to see the rest>"),
            "write_refused_outside": write_refused_outside,
            "write_refused_unset": write_refused_unset,
        },
    }


@pytest.mark.parametrize("agent_name", AGENT_NAMES)
def test_rendered_result_fields_match_provider_tool_schema(
    agent_name: str, tmp_path: Path,
) -> None:
    agent = _BUILDERS[agent_name](tmp_path)
    final = lg._make_final_answer_tool(agent.result_schema, agent.final_tool_name)
    properties = convert_to_openai_tool(final)["function"]["parameters"]["properties"]

    assert render_model_surface(agent)["final_answer"]["result_fields"] == {
        name: (field.get("description") or "") for name, field in properties.items()
    }
    assert "explanation" in properties
    assert "status" in properties


@pytest.mark.parametrize("agent_name", AGENT_NAMES)
@pytest.mark.parametrize("count", [None, 0, 3])
def test_runtime_evidence_is_serialized_but_not_model_visible(
    agent_name: str, count: int | None, tmp_path: Path,
) -> None:
    agent = _BUILDERS[agent_name](tmp_path)
    values = {
        "explanation": "Tests the target.",
        "source_observations": {"target": {"status": "unavailable", "value": None}},
        "model_requests": count,
        "logical_requests": count,
        "unknown_usage_requests": count,
    }
    if agent_name == "test_coverage":
        values.update(test_file_name="test_target.py", targeted_functions=["f"])
    else:
        values.update(mutation_test_file_name="test_target.py", targeted_mutants=[])
    result = agent.result_schema.model_validate(values)
    serialized = result.model_dump(mode="json")
    final = lg._make_final_answer_tool(agent.result_schema, agent.final_tool_name)
    properties = convert_to_openai_tool(final)["function"]["parameters"]["properties"]
    snapshot_fields = render_model_surface(agent)["final_answer"]["result_fields"]

    for name in (
        "source_observations", "model_requests", "logical_requests", "unknown_usage_requests",
    ):
        assert serialized[name] == values[name]
        assert name not in properties
        assert name not in snapshot_fields


@pytest.mark.parametrize("agent_name", AGENT_NAMES)
def test_model_surface_is_byte_identical_to_snapshot(
    agent_name: str, tmp_path: Path, fixtures_dir: Path,
) -> None:
    fixture = fixtures_dir / FIXTURE_NAME
    surface = render_model_surface(_BUILDERS[agent_name](tmp_path))

    if os.environ.get(REGEN_ENV) == "1":
        snap_all = (
            json.loads(fixture.read_text("utf-8")) if fixture.exists() else {}
        )
        snap_all[agent_name] = surface
        fixture.write_text(
            json.dumps(snap_all, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )

    snap = json.loads(fixture.read_text("utf-8"))[agent_name]
    assert surface == snap, (
        f"{agent_name}: the model-visible surface changed. If this is a "
        f"deliberate prompt-engineering decision, regenerate with "
        f"{REGEN_ENV}=1 pytest {Path(__file__).name} and commit the fixture "
        f"in the same change."
    )
