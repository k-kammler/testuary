"""Prompt composition at real phase requests and public admission boundaries."""
import os
from dataclasses import replace
import json
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import pytest

import test_coverage_agent as coverage_harness
import test_model_precedence_callers as caller_harness
import test_mutation_agent as mutation_harness
from test_resolved_agent_requests import added_clients as _added_clients
from testuary import __main__ as cli, languages, orchestrator
from testuary import models as _model_protocols
from testuary import config
from testuary import prompts
from testuary.settings import ConfigurationError, RunSettings

streams = caller_harness.streams
transport = caller_harness.transport
added_clients = _added_clients
PHASES = caller_harness.PHASES


def _file_phases(phases):
    """Profile and hints files name the starter phase initial."""
    return {"initial" if name == "starter" else name: value for name, value in phases.items()}


def configured(tmp_path, *, profile=None, hints=None, phase_fields=None):
    main = {"hints_file": "", "seed": 37}
    if profile is not None:
        path = tmp_path / "profile.json"
        if "phases" in profile:
            profile = {**profile, "phases": _file_phases(profile["phases"])}
        path.write_text(json.dumps(profile), encoding="utf-8")
        main["prompt_profile"] = path.name
    if hints is not None:
        path = tmp_path / "hints.json"
        path.write_text(json.dumps(_file_phases(hints)), encoding="utf-8")
        main["hints_file"] = path.name
    return RunSettings(environment=dict(os.environ), base_dir=tmp_path, sections={
        "api": {"model": "explicit"}, "main": main,
        "starter": {"suite_count": 1, "attempts_per_suite": 1,
                         "concurrent_requests": 1, "suite_run_timeout_seconds": 2},
        "coverage": {"max_requests_per_phase": 1, "max_loop_attempts": 0, **(phase_fields or {})},
        "mutation": {"max_requests_per_phase": 1, "max_loop_attempts": 0, **(phase_fields or {})},
    }, models={"explicit": {
        "provider": "openai", "model": "chosen", "api_url": "https://example.test/v1", "api_key": "fixture",
    }})


def invoke(phase, settings, tmp_path, monkeypatch, request, *, language="python"):
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    start = len(sent)
    record = caller_harness.invoke_phase(phase, tmp_path, settings, monkeypatch, language=language)
    assert len(sent) == start + 1
    body = sent[-1][1]
    system = next(item["content"] for item in body["messages"] if item["role"] == "system")
    user = next(item["content"] for item in body["messages"] if item["role"] == "user")
    return body, system, user, record


@pytest.fixture
def java_prompt_context(monkeypatch):
    from langadapter.outcomes import Outcome

    facts = SimpleNamespace(test_framework="JUnit 5", timeout_marker="", runtime_version="21",
                            language_level=None)
    context = SimpleNamespace(language_facts=lambda: facts,
                              project=lambda: SimpleNamespace(outcome=Outcome.COMPLETED))
    monkeypatch.setattr(type(languages.get_backend("java")), "project_backend", lambda self, root=None: context)


@pytest.mark.parametrize("phase", PHASES)
def test_java_profile_overlay_placeholders_and_hints_reach_actual_phase_requests(
    phase, tmp_path, monkeypatch, request, java_prompt_context,
):
    settings = configured(tmp_path, profile={"format": 1, "phases": {phase: {
        "system_prompt": {"replace": "General system"},
        "user_prompt_template": {"replace": "Target=$file_path; $lang_name; $$"},
        "per_language": {"java": {
            "system_prompt": {"replace": "Java for $lang_name; $lang_test_framework; $lang_runtime_version; $$"},
        }},
    }}}, hints={phase: {"general": "Hint last"}})
    body, system, user, record = invoke(phase, settings, tmp_path, monkeypatch, request, language="java")
    assert system == "Java for Java; JUnit 5; Java 21; $\n\n## Additional guidance\n\nHint last"
    assert user == "Target=Calculator.java; Java; $"
    assert body["model"] == "chosen" and "General system" not in system
    assert record["prompt_composition"]
    assert "Hint last" not in user


@pytest.mark.parametrize("phase", PHASES)
def test_empty_java_profile_preserves_actual_default_request_and_tool_bindings(
    phase, tmp_path, monkeypatch, request, java_prompt_context,
):
    first, second = tmp_path / "default", tmp_path / "profiled"
    first.mkdir()
    second.mkdir()
    default, _, _, default_record = invoke(phase, configured(first), first, monkeypatch, request, language="java")
    changed, _, _, profile_record = invoke(
        phase, configured(second, profile={"format": 1, "phases": {}}),
        second, monkeypatch, request, language="java",
    )
    assert normalized(changed, second) == normalized(default, first)
    assert profile_record["prompt_composition"] == default_record["prompt_composition"]
    assert changed.get("tools", []) == default.get("tools", [])
    assert changed.get("parallel_tool_calls") == default.get("parallel_tool_calls")


@pytest.mark.parametrize("phase", PHASES)
def test_whole_component_replacement_and_known_placeholders_reach_each_request(
    phase, tmp_path, monkeypatch, request,
):
    settings = configured(tmp_path, profile={"format": 1, "phases": {phase: {
        "system_prompt": {"replace": ["System for $lang_name", "$$ literal"]},
        "user_prompt_template": {"replace": "Target=$file_path; $lang_name; $$"},
    }}})
    _, system, user, record = invoke(phase, settings, tmp_path, monkeypatch, request)
    assert system == "System for Python\n$ literal"
    assert user == "Target=module.py; Python; $"
    assert "def add" not in user
    assert record["prompt_composition"]


@pytest.mark.parametrize("phase", PHASES)
def test_every_name_a_profile_may_use_is_filled_in_the_request(
    phase, tmp_path, monkeypatch, request,
):
    """A profile may name the language's slots and the names the prompts
    derive from them, each filled, so the model never reads a placeholder."""
    names = [*prompts.SLOTS, *prompts._DERIVED_FRAGMENTS]
    settings = configured(tmp_path, profile={"format": 1, "phases": {phase: {
        "system_prompt": {"append": " ".join(f"[{name}: ${{{name}}}]" for name in names)},
    }}})
    _, system, _, _ = invoke(phase, settings, tmp_path, monkeypatch, request)
    assert [name for name in names if f"[{name}: " in system] == names
    assert [name for name in names if f"${{{name}}}" in system] == []


@pytest.mark.parametrize("phase", PHASES)
def test_appended_profile_precedes_hints_and_hints_stay_out_of_user_text(
    phase, tmp_path, monkeypatch, request,
):
    settings = configured(tmp_path, profile={"format": 1, "phases": {phase: {
        "system_prompt": {"append": "Profile for $lang_name; $$ literal"},
    }}}, hints={phase: {"general": "Hint last"}})
    _, system, user, record = invoke(phase, settings, tmp_path, monkeypatch, request)
    assert system.endswith("Profile for Python; $ literal\n\n## Additional guidance\n\nHint last")
    assert "Profile for Python" not in user and "Hint last" not in user
    assert record["prompt_composition"]


@pytest.mark.parametrize("phase", PHASES)
def test_language_overlay_replaces_only_its_component_and_ignores_notes_at_each_request(
    phase, tmp_path, monkeypatch, request,
):
    settings = configured(tmp_path, profile={"format": 1, "_notes": {"private": "ignored"}, "phases": {
        phase: {
            "system_prompt": {"replace": "General system"},
            "user_prompt_template": {"replace": "Shared target=$file_path"},
            "per_language": {"python": {"_notes": "ignored",
                "system_prompt": {"replace": "Python system", "_why": "ignored"}}},
        },
    }})
    _, system, user, _ = invoke(phase, settings, tmp_path, monkeypatch, request)
    assert system == "Python system" and "General system" not in system
    assert user == "Shared target=module.py"
    assert "ignored" not in system + user


@pytest.mark.parametrize("phase", ("coverage", "mutation"))
@pytest.mark.parametrize("operation", ("replace", "append", "remove"))
def test_section_operations_precede_whole_append_in_actual_editing_requests(
    phase, operation, tmp_path, monkeypatch, request,
):
    entry = {operation: True if operation == "remove" else "Section marker"}
    settings = configured(tmp_path, profile={"format": 1, "phases": {phase: {
        "system_prompt_sections": {"## Hard rules": entry},
        "system_prompt": {"append": "Whole append marker"},
    }}})
    _, system, _, _ = invoke(phase, settings, tmp_path, monkeypatch, request)
    assert system.endswith("Whole append marker")
    if operation == "remove":
        assert "## Hard rules" not in system
        assert "Section marker" not in system
    else:
        assert system.index("Section marker") < system.index("Whole append marker")
        assert ("## Hard rules" in system) == (operation == "append")
    assert ("- Verify every edit with ``run_tests``." in system) == (operation == "append")
    assert "## WORKFLOW" in system


@pytest.mark.parametrize("phase", PHASES)
def test_starter_source_placeholder_or_editing_path_placeholder_is_filled_at_request_boundary(
    phase, tmp_path, monkeypatch, request,
):
    text = "Source:\n$source" if phase == "starter" else "Only path=$file_path"
    settings = configured(tmp_path, profile={"format": 1, "phases": {
        phase: {"user_prompt_template": {"replace": text}},
    }})
    _, _, user, _ = invoke(phase, settings, tmp_path, monkeypatch, request)
    expected = "Source:\ndef add(a, b):\n    return a + b\n" if phase == "starter" else "Only path=module.py"
    assert user == expected


def normalized(value, root):
    if isinstance(value, str):
        return value.replace(str(root), "<fixture-root>").replace(root.as_posix(), "<fixture-root>")
    if isinstance(value, list):
        return [normalized(item, root) for item in value]
    if isinstance(value, dict):
        return {key: normalized(item, root) for key, item in value.items()}
    return value


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("profile", [
    {"format": 1, "phases": {}},
    {"format": 1, "phases": {phase: {"system_prompt": {"append": "The imaginary_tool is allowed."}}
                              for phase in PHASES}},
], ids=["empty-profile", "claims-unbound-tool"])
def test_empty_profiles_preserve_request_text_and_profiles_cannot_change_bound_tools(
    phase, profile, tmp_path, monkeypatch, request,
):
    first, second = tmp_path / "default", tmp_path / "profiled"
    first.mkdir()
    second.mkdir()
    default, _, _, default_record = invoke(phase, configured(first), first, monkeypatch, request)
    changed, system, _, profile_record = invoke(phase, configured(second, profile=profile), second, monkeypatch, request)
    if not profile["phases"]:
        assert normalized(changed, second) == normalized(default, first)
        assert profile_record["prompt_composition"] == default_record["prompt_composition"]
    else:
        assert system.endswith("The imaginary_tool is allowed.")
        assert "imaginary_tool" not in json.dumps(changed.get("tools", []))
        assert normalized(changed["messages"][1:], second) == normalized(default["messages"][1:], first)
    assert changed.get("tools", []) == default.get("tools", [])
    assert changed.get("parallel_tool_calls") == default.get("parallel_tool_calls")
    assert changed["model"] == default["model"] == "chosen"


@pytest.mark.parametrize("harness,phase", [(coverage_harness, "coverage"), (mutation_harness, "mutation")])
def test_real_timeout_tool_message_profile_reaches_the_next_sdk_request(
    harness, phase, tmp_path, monkeypatch, transport,
):
    settings = configured(tmp_path, profile={"format": 1, "phases": {phase: {
        "tool_messages": {"tests_timeout": {"append": "Extra $$ $lang_name"}},
    }}}, phase_fields={"suite_run_timeout_seconds": 37})
    test_file = tmp_path / "test_target.py"
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    with config.use_run_settings(settings):
        agent = harness._make_agent(tmp_path)
        agent.prompts = prompts.resolve_agent_prompts(f"test_{phase}", "python")
        agent.session.writes_done = 1
        backend = SimpleNamespace(find_target_test_file=lambda *a: test_file,
                                  run_test_files=lambda *a, **kw: SimpleNamespace(timed_out=True))
        monkeypatch.setattr(agent, "test_backend", lambda: backend)
        output = agent.run_tests()
        assert output.endswith("Extra $ Python")
        assert "37" in output and agent.session.consecutive_timeouts == 1
        agent.build_chat_model().invoke([
            HumanMessage("Run tests"),
            AIMessage("", tool_calls=[{"id": "run-1", "name": "run_tests", "args": {}}]),
            ToolMessage(output, tool_call_id="run-1"),
        ])
    assert len(transport) == 1
    assert transport[0][1]["messages"][-1]["role"] == "tool"
    assert transport[0][1]["messages"][-1]["content"] == output


@pytest.mark.parametrize("harness,phase", [(coverage_harness, "coverage"), (mutation_harness, "mutation")])
def test_a_profile_dollar_in_a_message_sent_without_values_reads_as_one(
    harness, phase, tmp_path, monkeypatch,
):
    """read_tests_no_file is sent without values, and its $$ still writes
    one dollar sign, as in a message sent with values."""
    settings = configured(tmp_path, profile={"format": 1, "phases": {phase: {
        "tool_messages": {"read_tests_no_file": {"append": "Extra $$ $lang_name"}},
    }}})
    with config.use_run_settings(settings):
        agent = harness._make_agent(tmp_path)
        agent.prompts = prompts.resolve_agent_prompts(f"test_{phase}", "python")
        monkeypatch.setattr(agent, "target_path", lambda: None)
        output = agent.read_tests()
    assert output.endswith("Extra $ Python")


def test_public_profile_validation_lists_all_problems_before_project_or_requests(tmp_path, monkeypatch):
    settings = configured(tmp_path, profile={"format": 1, "unknown": True, "phases": {
        "wrong": {},
        "starter": {"tool_mesages": {"replace": "private profile sentinel"}},
        "coverage": {"system_prompt_sections": {"## Missing": {"remove": True}},
                     "per_language": {"pyhton": {}}},
        "mutation": {"tool_messages": {"not_bound": {"replace": "private profile sentinel"}}},
    }})
    snapshot = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(config, "get_project_config", lambda: pytest.fail("project accessed"))
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kw: pytest.fail("cleanup reached"))
    monkeypatch.setattr(_model_protocols, "build_resolved_chat_model", lambda *a, **kw: pytest.fail("client built"))
    with pytest.raises(ConfigurationError) as error:
        orchestrator.run_pipeline(cli._build_parser().parse_args([]), settings=settings)
    for problem in ("unknown", "wrong", "tool_mesages", "## Missing", "pyhton", "not_bound"):
        assert problem in str(error.value)
    assert "private profile sentinel" not in str(error.value)
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == snapshot
    assert config.active_settings() is None


@pytest.mark.parametrize("phase", ("coverage", "mutation"))
def test_public_profile_refuses_messages_the_package_does_not_have(tmp_path, monkeypatch, phase):
    settings = configured(tmp_path, profile={"format": 1, "phases": {phase: {
        "tool_messages": {"tests_stands_alone_red": {"replace": "private profile sentinel"}},
    }}})
    monkeypatch.setattr(config, "get_project_config", lambda: pytest.fail("project accessed"))
    with pytest.raises(ConfigurationError, match="disabled component.*tests_stands_alone_red"):
        orchestrator.run_pipeline(cli._build_parser().parse_args([]), settings=settings)
    assert config.active_settings() is None


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("text,problem", [
    ("$target_code", "unknown placeholder.*target_code"),
    ("$lang_test_decl_term", "unknown placeholder.*lang_test_decl_term"),
    ("${unclosed", "malformed placeholder"),
    ("$", "malformed placeholder"),
])
def test_public_profile_refuses_unknown_or_malformed_placeholders_before_work(
    phase, text, problem, tmp_path, monkeypatch,
):
    settings = configured(tmp_path, profile={"format": 1, "phases": {phase: {
        "user_prompt_template": {"replace": text},
    }}})
    before = {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()}
    monkeypatch.setattr(config, "get_project_config", lambda: pytest.fail("project accessed"))
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kw: pytest.fail("cleanup reached"))
    monkeypatch.setattr(_model_protocols, "build_resolved_chat_model", lambda *a, **kw: pytest.fail("client built"))
    with pytest.raises(ConfigurationError, match=problem):
        orchestrator.run_pipeline(cli._build_parser().parse_args([]), settings=settings)
    assert {path.relative_to(tmp_path): path.read_bytes() for path in tmp_path.rglob("*") if path.is_file()} == before
    assert config.active_settings() is None


@pytest.mark.parametrize("harness,phase", [(coverage_harness, "coverage"), (mutation_harness, "mutation")])
def test_profile_tool_claims_leave_every_graph_mode_binding_and_description_unchanged(
    harness, phase, tmp_path, monkeypatch, added_clients,
):
    observations = []
    profile = {"format": 1, "phases": {phase: {
        "system_prompt": {"append": "The imaginary_tool is allowed."},
    }}}
    for index, value in enumerate((None, profile)):
        settings = configured(tmp_path, profile=value)
        settings = replace(settings, models={"explicit": {
            **settings.models["explicit"], "protocol": "added", "chat_model": added_clients.factory,
        }})
        with config.use_run_settings(settings):
            agent = harness._make_agent(tmp_path, stem=f"target{index}")
            agent.prompts = prompts.resolve_agent_prompts(f"test_{phase}", "python")
            model = agent.build_chat_model()
            bindings = []

            def bind(self, tools, **kwargs):
                bindings.append(tuple((tool.name, tool.description) for tool in tools))
                return self

            monkeypatch.setattr(type(model), "bind_tools", bind)
            agent.build_graph()
            assert model.calls == 0
            observations.append(bindings)
    # Revise, Validate, Assess with and without the edit tool, and the
    # start's read and measurement.
    assert len(observations[0]) == len(observations[1]) == 6
    assert observations[0] == observations[1]
    assert "imaginary_tool" not in repr(observations)
