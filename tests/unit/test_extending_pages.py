"""The examples on the extending pages work as shown.

Included examples are the shipped files test_extension_examples resolves.
The JSON, settings and Python written on the pages are resolved here through
the settings and prompt composition a run uses. Nothing here sends a model
request.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from testuary import config
from testuary import hints as hints_module
from testuary import prompts as prompts_module
from testuary.settings import resolve_settings
from source_tree import requires_source_tree
import ast
import os
import langadapter as la

ROOT = Path(__file__).resolve().parents[2]
PAGES = ROOT / "docs" / "extending"
EXAMPLES = "src/testuary/_templates/examples/"


def _blocks(page: str, language: str) -> list[str]:
    text = (PAGES / page).read_text(encoding="utf-8")
    return [block for block in re.findall(rf"```{language}\n(.*?)```", text, re.S)
            if not block.startswith("--8<--")]


def _prompts(tmp_path, *, profile=None, hints=None):
    if profile is not None:
        (tmp_path / "profile.json").write_text(profile, encoding="utf-8")
    if hints is not None:
        (tmp_path / "hints.json").write_text(hints, encoding="utf-8")
    settings = resolve_settings(
        base_dir=tmp_path, config_files=[], models={}, environment={},
        overrides={"main": {"prompt_profile": "profile.json" if profile is not None else "",
                            "hints_file": "hints.json" if hints is not None else ""}})
    with config.use_run_settings(settings):
        return {phase: prompts_module.resolve_agent_prompts(f"test_{phase}", "python", runtime_version="fixture")
                for phase in ("initial", "coverage", "mutation")}


def _profile(name: str) -> str:
    [profile] = [block for block in _blocks("prompt-profile.md", "json") if json.loads(block)["name"] == name]
    return profile


def test_the_profile_on_the_page_changes_each_phase_as_described(tmp_path):
    prompts = _prompts(tmp_path, profile=_profile("boundary-tests"))
    for phase, sentence in (("initial", "Exercise empty, boundary and error inputs."),
                            ("coverage", "Use explicit expected values."),
                            ("mutation", "Keep assertions tied to observable behavior.")):
        assert sentence in prompts[phase]["system_prompt"]


def test_the_section_and_tool_message_profile_on_the_page_works_as_described(tmp_path):
    prompts = _prompts(tmp_path, profile=_profile("independent-tests"))
    coverage = prompts["coverage"]
    rules = coverage["system_prompt"].split("## Hard rules", 1)[1]
    assert "Keep every test independent of the others." in rules
    assert "## Test libraries available" not in coverage["system_prompt"]
    assert "## Test libraries available" in prompts["mutation"]["system_prompt"]
    assert coverage["tool_messages"]["edit_not_found"].endswith("Read the test file before the next edit.")


def test_the_scripted_model_on_the_provider_page_walks_the_coverage_steps():
    from langchain_core.messages import HumanMessage

    from testuary import starters
    from testuary.agent import _langgraph
    from testuary.agent.coverage_agent import CoverageAgent, CoverageTestResult

    [script] = [block for block in _blocks("model-provider.md", "python") if "class ScriptedChat" in block]
    namespace: dict = {}
    exec(script, namespace)
    model = namespace["factory"](model="scripted", base_url="", api_key="", timeout=600.0, max_retries=0)
    answer = "".join(chunk.content for chunk in model.stream([HumanMessage("Write the tests.")]))
    assert "def test_positive():" in starters._extract_code(answer, "python")
    agent = CoverageAgent(prompts={}, deps=None, session=None)
    agent.preflight_tool_binding(model)
    modes = agent.tool_mode_sets(*agent.phase_tools())
    result_tool = _langgraph._make_final_answer_tool(agent.result_schema, agent.final_tool_name)
    steps = (agent.INITIAL_READ_MODE, agent.INITIAL_MEASURE_MODE, "needs_editing", "needs_testing",
             "needs_measurement", "needs_editing")
    calls = [model.bind_tools([*modes[step], result_tool]).invoke([HumanMessage("Go on.")]).tool_calls[0]
             for step in steps]
    assert [call["name"] for call in calls] == [
        "read_tests", "measure_coverage", "edit_tests", "run_tests", "measure_coverage",
        "emit_coverage_test_result"]
    CoverageTestResult(**calls[-1]["args"])


def test_the_trial_run_settings_on_the_provider_page_are_settings(tmp_path):
    [trial] = [block for block in _blocks("model-provider.md", "ini") if "[starter]" in block]
    (tmp_path / ".testuary").mkdir()
    (tmp_path / ".testuary" / "project.ini").write_text(trial, encoding="utf-8")
    settings = resolve_settings(base_dir=tmp_path, config_files=[], models={}, environment={})
    with config.use_run_settings(settings):
        assert config.consolidation_measure_mutation() is False
        assert int(config.get_agent_config("test_initial")["suite_count"]) == 1


def test_the_service_section_on_the_provider_page_selects_the_factory(tmp_path):
    from testuary.models import resolve_phase_model

    [catalogue] = [block for block in _blocks("model-provider.md", "ini") if "[my_service]" in block]
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "models.ini").write_text(catalogue, encoding="utf-8")
    (home / "my_chat.py").write_text("def factory(**parameters):\n    raise NotImplementedError\n",
                                     encoding="utf-8")
    settings = resolve_settings(base_dir=project, config_files=[], environment={}, home=home,
                                overrides={"api": {"model": "my_alias"}})
    model = resolve_phase_model(settings, "coverage")
    assert (model.protocol, model.chat_model) == ("added", "my_chat.py:factory")
    assert (model.identity.provider, model.identity.api_url) == ("my_service", "https://service.example/api")


def test_the_trial_alias_on_the_provider_page_builds_the_scripted_model_without_a_key(tmp_path):
    from testuary.models import build_resolved_chat_model, resolve_phase_model, resolve_provider_credential

    [catalogue] = [block for block in _blocks("model-provider.md", "ini") if "scripted_chat.py" in block]
    [script] = [block for block in _blocks("model-provider.md", "python") if "class ScriptedChat" in block]
    home, project = tmp_path / "home", tmp_path / "project"
    home.mkdir()
    project.mkdir()
    (home / "models.ini").write_text(catalogue, encoding="utf-8")
    (home / "scripted_chat.py").write_text(script, encoding="utf-8")
    settings = resolve_settings(base_dir=project, config_files=[], environment={}, home=home,
                                overrides={"api": {"model": "my_alias"}})
    model = resolve_phase_model(settings, "coverage")
    key = resolve_provider_credential(model.identity.provider, settings.models[model.identity.provider],
                                      settings.models["my_alias"], {})
    assert (model.protocol, key) == ("added", "")
    assert type(build_resolved_chat_model(model, api_key=key)).__name__ == "ScriptedChat"


def test_the_hints_on_the_page_are_valid_and_reach_their_phase(tmp_path, in_run):
    hints = _hints_json("Cover the public input boundary cases.")
    in_run()
    assert hints_module.validate_hints(json.loads(hints)) == []
    in_run.leave()
    prompts = _prompts(tmp_path, hints=hints)
    assert "Cover the public input boundary cases." in prompts["initial"]["system_prompt"]
    assert "Use explicit expected values." in prompts["coverage"]["system_prompt"]
    assert "Use explicit expected values." not in prompts["mutation"]["system_prompt"]


def _hints_json(marker: str) -> str:
    [block] = [block for block in _blocks("hints.md", "json") if marker in block]
    return block


def test_every_hints_file_on_the_page_is_valid(in_run):
    in_run()
    for block in _blocks("hints.md", "json"):
        assert hints_module.validate_hints(json.loads(block)) == [], block


def test_every_profile_on_the_page_resolves(tmp_path):
    for index, block in enumerate(_blocks("prompt-profile.md", "json")):
        directory = tmp_path / str(index)
        directory.mkdir()
        _prompts(directory, profile=block)


def _hints_python(marker: str) -> str:
    [block] = [block for block in _blocks("hints.md", "python") if marker in block]
    return block


def test_the_extension_hints_on_the_page_are_valid_hints():
    namespace: dict = {}
    exec(_hints_python("testuary_hints ="), namespace)
    assert hints_module.validate_hints(namespace["testuary_hints"], label="page") == []
    assert hints_module.request_text(namespace["testuary_hints"], "python", "coverage") == (
        "Name each test after the behavior it checks.")


def test_the_request_text_call_on_the_page_prints_the_guidance(tmp_path, monkeypatch, capsys):
    hints = _hints_json("Cover the public input boundary cases.")
    (tmp_path / "hints.json").write_text(hints, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    exec(_hints_python("request_text("), {"__name__": "guide_example"})
    assert capsys.readouterr().out == "Use explicit expected values.\n"


@requires_source_tree
def test_included_examples_are_the_shipped_examples():
    included = [name for page in PAGES.glob("*.md")
                for name in re.findall(r'^--8<-- "([^"]+)"', page.read_text(encoding="utf-8"), re.M)]
    assert included and all(name.startswith(EXAMPLES) and (ROOT / name).is_file() for name in included)


# The example of docs/extending/languages.md runs as written, without a model
# request: the Toy's adapter from LangAdapter's tutorial at Testuary's level 1,
# registered from outside, passes Testuary's check, which names the fallback of
# each optional family the Toy does not provide.
GUIDE = Path(__file__).parents[2] / "docs" / "extending" / "languages.md"


def _example() -> str:
    """The worked example: the page's one block that runs Testuary's check."""
    blocks = re.findall(r"^```python\n(.*?)^```", GUIDE.read_text(encoding="utf-8"),
                        re.MULTILINE | re.DOTALL)
    (block,) = [block for block in blocks if "orchestrator.check(" in block]
    return block


@requires_source_tree
def test_the_language_page_example_runs_as_written(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # As in a user's shell: no Testuary settings from the test environment.
    for name in [name for name in os.environ if name.startswith("TESTUARY_")]:
        monkeypatch.delenv(name)
    exec(compile(ast.parse(_example()), str(GUIDE), "exec"), {"__name__": "guide_example"})
    out = capsys.readouterr().out
    check = next(line for line in out.splitlines() if line.startswith("check:"))
    # Level 1 is enough: nothing required is unsupported.
    assert check != "check: unsupported", out
    assert "The language adapter lacks merging" in out
    assert "Without it, the better single candidate is kept" in out
    assert "The language adapter lacks load repair" in out
    # The example leaves no registration behind.
    assert "toy" not in la.names()
