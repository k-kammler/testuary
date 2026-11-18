"""Prompt profiles compose text only, with early aggregate validation."""
from dataclasses import FrozenInstanceError
from string import Template
from types import SimpleNamespace

import pytest

from testuary import config
from testuary import prompts
from testuary.prompts import validate_profile
from testuary.settings import ConfigurationError
from testuary.settings import RunSettings


@pytest.fixture(autouse=True)
def java_prompt_facts(monkeypatch):
    from testuary.languages import get_backend
    facts = SimpleNamespace(test_framework="JUnit 5", timeout_marker="@Timeout({seconds})", runtime_version="17",
                            language_level=None)
    monkeypatch.setattr(type(get_backend("java")), "project_backend",
                        lambda self, source_dir=None: SimpleNamespace(language_facts=lambda: facts))


@pytest.fixture
def templates():
    return {phase: {
        "system_prompt": "Preamble\n\n## Hard rules\nOriginal $lang_name\n\n## Other\nOther text\n",
        "user_prompt_template": "$file_path\n$source",
        "tool_messages": {"tests_timeout": "Timed out after $timeout", "tests_red": "Failed: $output"},
        "tool_descriptions": {"run_tests": "Unchanged description"},
    } for phase in ("initial", "coverage", "mutation")}


def _profile(templates, phase="coverage", **components):
    return validate_profile({"format": 1, "phases": {phase: components}}, templates,
                            languages=("python", "java"), fragments=("lang_name", "read_protocol"))


@pytest.mark.parametrize("phase", ["initial", "coverage", "mutation"])
@pytest.mark.parametrize("language", ["python", "java"])
def test_empty_profile_preserves_every_bundled_component(phase, language):
    bundled = {p: prompts.load_agent_prompts(f"test_{p}") for p in ("initial", "coverage", "mutation")}
    profile = validate_profile({"format": 1, "phases": {}}, bundled, languages=("python", "java"))
    assert profile.compose(phase, language, bundled[phase]) == bundled[phase]
    assert profile.changes(phase, language) == ()


@pytest.mark.parametrize("kind", ["system_prompt", "user_prompt_template"])
@pytest.mark.parametrize("operation", ["replace", "append"])
@pytest.mark.parametrize("content", ["Text $lang_name", ["Text", "$lang_name"]])
def test_whole_components_accept_text_and_lines(templates, kind, operation, content):
    profile = _profile(templates, **{kind: {operation: content}})
    text = content if isinstance(content, str) else "\n".join(content)
    output = profile.compose("coverage", "python", templates["coverage"])
    assert output[kind] == (text if operation == "replace" else f"{templates['coverage'][kind]}\n\n{text}")
    assert output["tool_messages"] == templates["coverage"]["tool_messages"]
    assert output["tool_descriptions"] == templates["coverage"]["tool_descriptions"]


def test_section_removal_leaves_other_text_identical(templates):
    profile = _profile(templates, system_prompt_sections={"## Hard rules": {"remove": True}})
    result = profile.compose("coverage", "python", templates["coverage"])
    assert result["system_prompt"] == "Preamble\n\n## Other\nOther text\n"
    assert profile.changes("coverage", "python") == (("system_prompt_sections", "## Hard rules", "remove"),)


@pytest.mark.parametrize("operation", ["replace", "append"])
def test_section_text_stays_separated_from_the_next_heading(templates, operation):
    profile = _profile(templates, system_prompt_sections={"## Hard rules": {operation: "New section text"}})
    output = profile.compose("coverage", "python", templates["coverage"])["system_prompt"]
    assert "New section text\n" in output
    assert output.endswith("## Other\nOther text\n")


def test_system_append_is_applied_after_sections_even_with_a_repeated_heading(templates):
    appended = "## Hard rules\nThis append must stay"
    profile = _profile(templates, system_prompt={"append": appended},
                       system_prompt_sections={"## Hard rules": {"remove": True}})
    output = profile.compose("coverage", "python", templates["coverage"])["system_prompt"]
    assert output == "Preamble\n\n## Other\nOther text\n\n\n" + appended


def test_language_overrides_the_same_entry_not_unrelated_components(templates):
    profile = _profile(templates,
        system_prompt_sections={"## Hard rules": {"replace": "## Hard rules\nGeneral\n\n"},
                                "## Other": {"remove": True}},
        tool_messages={"tests_timeout": {"append": "General extra"}},
        per_language={"python": {"system_prompt_sections": {"## Hard rules": {"remove": True}},
                                 "tool_messages": {"tests_red": {"replace": "Specific $output"}}}},
    )
    python = profile.compose("coverage", "python", templates["coverage"])
    java = profile.compose("coverage", "java", templates["coverage"])
    assert python["system_prompt"] == "Preamble\n\n"
    assert java["system_prompt"] == "Preamble\n\n## Hard rules\nGeneral\n\n"
    assert python["tool_messages"]["tests_timeout"].endswith("General extra")
    assert python["tool_messages"]["tests_red"] == "Specific $output"
    assert java["tool_messages"]["tests_red"] == templates["coverage"]["tool_messages"]["tests_red"]


def test_text_placeholders_are_component_scoped_but_may_be_omitted(templates):
    profile = _profile(templates, user_prompt_template={"replace": "${file_path} $$ $lang_name"})
    text = profile.compose("coverage", "python", templates["coverage"])["user_prompt_template"]
    assert Template(text).substitute(file_path="file.py", lang_name="Python") == "file.py $ Python"
    assert "$source" not in text
    with pytest.raises(ConfigurationError, match="unknown placeholder.*source"):
        _profile(templates, tool_messages={"tests_timeout": {"append": "$source"}})


@pytest.mark.parametrize("entry", [
    {}, {"replace": "a", "append": "b"}, {"subtract": "a"}, {"remove": True},
    {"replace": None}, {"replace": 3}, {"replace": ["a", 3]}, {"replace": "$"},
    {"replace": "${unclosed"}, {"replace": "$target_code"}, "plain text",
])
def test_invalid_component_operations_are_refused(templates, entry):
    with pytest.raises(ConfigurationError, match="profile.phases.coverage.system_prompt"):
        _profile(templates, system_prompt=entry)


@pytest.mark.parametrize("remove", [False, "true", 1, None])
def test_remove_is_a_boolean_true_only(templates, remove):
    with pytest.raises(ConfigurationError, match="remove must be true"):
        _profile(templates, system_prompt_sections={"## Hard rules": {"remove": remove}})


@pytest.mark.parametrize("value", [None, [], "text", 1])
def test_profile_is_an_object(templates, value):
    with pytest.raises(ConfigurationError, match="JSON object"):
        validate_profile(value, templates, languages=("python",))


@pytest.mark.parametrize("value", [{}, {"format": True, "phases": {}}, {"format": 2, "phases": {}},
                                   {"format": 1, "phases": []}, {"format": 1, "phases": {}, "name": 1}])
def test_header_fields_are_strict(templates, value):
    with pytest.raises(ConfigurationError):
        validate_profile(value, templates, languages=("python",))


def test_all_phases_are_checked_and_errors_do_not_echo_prompt_content(templates):
    value = {"format": 1, "unknown": True, "phases": {
        "wrong": {}, "initial": {"tool_mesages": {"replace": "private content"}},
        "coverage": {"system_prompt_sections": {"## Missing": {"remove": True}},
                     "per_language": {"pyhton": {}}},
        "mutation": {"tool_messages": {"not_bound": {"replace": "private content"}}},
    }}
    with pytest.raises(ConfigurationError) as caught:
        validate_profile(value, templates, languages=("python", "java"))
    message = str(caught.value)
    for key in ("unknown", "wrong", "tool_mesages", "## Missing", "pyhton", "not_bound"):
        assert key in message
    assert "private content" not in message


@pytest.mark.parametrize("localized", [False, True])
def test_whole_system_replacement_and_sections_conflict_including_language_overlay(templates, localized):
    components = {"system_prompt": {"replace": "Replacement"}}
    sections = {"system_prompt_sections": {"## Hard rules": {"remove": True}}}
    components.update({"per_language": {"python": sections}} if localized else sections)
    with pytest.raises(ConfigurationError, match="cannot be combined"):
        _profile(templates, **components)


def test_disabled_tool_messages_are_not_available_to_profiles(templates):
    with pytest.raises(ConfigurationError, match="disabled component.*tests_stands_alone_red"):
        _profile(templates, tool_messages={"tests_stands_alone_red": {"replace": "anything"}})


def test_notes_and_empty_language_blocks_are_ignored(templates):
    profile = validate_profile({"format": 1, "_notes": {"anything": True}, "phases": {
        "_notes": [3], "coverage": {"_notes": True, "system_prompt": {"append": "Text", "_why": "note"},
                                    "per_language": {"_notes": "note", "python": {"_notes": "note"}}},
    }}, templates, languages=("python",))
    assert profile.compose("coverage", "python", templates["coverage"])["system_prompt"].endswith("Text")


def test_profile_and_composed_views_are_detached(templates):
    value = {"format": 1, "name": "Ablation", "phases": {
        "coverage": {"tool_messages": {"tests_red": {"replace": "Changed"}}},
    }}
    profile = validate_profile(value, templates, languages=("python",))
    value["phases"]["coverage"]["tool_messages"]["tests_red"]["replace"] = "Mutated"
    with pytest.raises(FrozenInstanceError):
        profile.name = "other"
    with pytest.raises(TypeError):
        profile.operations("coverage", "python")["tool_messages"]["tests_red"] = ("append", "other")
    result = profile.compose("coverage", "python", templates["coverage"])
    result["tool_messages"]["tests_red"] = "Another mutation"
    assert profile.compose("coverage", "python", templates["coverage"])["tool_messages"]["tests_red"] == "Changed"
    assert templates["coverage"]["tool_messages"]["tests_red"] == "Failed: $output"


def test_multiple_section_edits_use_original_boundaries(templates):
    profile = _profile(templates, system_prompt_sections={"## Hard rules": {"remove": True},
                                                         "## Other": {"replace": "Last"}})
    assert profile.compose("coverage", "python", templates["coverage"])["system_prompt"] == "Preamble\n\nLast"


def test_ambiguous_bundled_heading_is_refused(templates):
    templates["coverage"]["system_prompt"] += "## Hard rules\nSecond section"
    with pytest.raises(ConfigurationError, match="ambiguous repeated heading"):
        _profile(templates, system_prompt_sections={"## Hard rules": {"remove": True}})


def test_error_order_matches_profile_input_order(templates):
    value = {"format": 1, "first_unknown": True, "second_unknown": True, "phases": {}}
    with pytest.raises(ConfigurationError) as caught:
        validate_profile(value, templates, languages=("python",))
    assert str(caught.value).index("first_unknown") < str(caught.value).index("second_unknown")


@pytest.mark.parametrize("components", [{"per_language": []}, {"per_language": {"python": []}},
                                      {"tool_messages": []}, {"system_prompt_sections": []}])
def test_nested_blocks_must_be_objects(templates, components):
    with pytest.raises(ConfigurationError):
        _profile(templates, **components)


@pytest.mark.parametrize("phase", ["initial", "coverage", "mutation"])
@pytest.mark.parametrize("language", ["python", "java"])
def test_real_renderer_preserves_defaults_for_empty_profiles(phase, language):
    bundled = {p: prompts.load_agent_prompts(f"test_{p}") for p in ("initial", "coverage", "mutation")}
    profile = validate_profile({"format": 1, "phases": {}}, bundled, languages=("python", "java"))
    with config.use_run_settings(RunSettings()):
        before = prompts.resolve_agent_prompts(f"test_{phase}", language, hints={})
        after = prompts.resolve_agent_prompts(f"test_{phase}", language, hints={}, prompt_profile=profile)
        assert after == before


@pytest.mark.parametrize("phase", ["initial", "coverage", "mutation"])
@pytest.mark.parametrize("language", ["python", "java"])
def test_real_renderer_composes_fragments_then_hints_and_keeps_literal_dollars(phase, language):
    with config.use_run_settings(RunSettings()):
        bundled = {p: prompts.load_agent_prompts(f"test_{p}") for p in ("initial", "coverage", "mutation")}
        components = {"system_prompt": {"append": "Profile for $lang_name; $$ literal"},
                      "user_prompt_template": {"replace": "$file_path: $$ price; $lang_name"}}
        profile = validate_profile({"format": 1, "phases": {phase: components}}, bundled,
                                   languages=("python", "java"), fragments=("lang_name",))
        output = prompts.resolve_agent_prompts(f"test_{phase}", language, hints={phase: {"general": "Hint last"}},
                                            prompt_profile=profile)
        assert f"Profile for {language.capitalize()}; $ literal\n\n## Additional guidance\n\nHint last" in output["system_prompt"]
        assert Template(output["user_prompt_template"]).substitute(file_path="path") == f"path: $ price; {language.capitalize()}"
        assert output["tool_messages"] == prompts.resolve_agent_prompts(f"test_{phase}", language, hints={})["tool_messages"]


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
def test_real_tool_message_profiles_preserve_target_substitution_and_escaping(phase):
    bundled = {p: prompts.load_agent_prompts(f"test_{p}") for p in ("initial", "coverage", "mutation")}
    profile = validate_profile({"format": 1, "phases": {
        phase: {"tool_messages": {"tests_timeout": {"append": "Extra $$ $lang_name"}}},
    }}, bundled, languages=("python",), fragments=("lang_name",))
    with config.use_run_settings(RunSettings()):
        output = prompts.resolve_agent_prompts(f"test_{phase}", "python", hints={}, prompt_profile=profile)
    rendered = Template(output["tool_messages"]["tests_timeout"]).safe_substitute()
    assert rendered.endswith("Extra $ Python")


def test_composition_rechecks_component_availability_under_current_settings():
    """A profile validated against messages the composed prompts do not
    have is refused when it is composed."""
    bundled = {p: prompts.load_agent_prompts(f"test_{p}") for p in ("initial", "coverage", "mutation")}
    bundled["coverage"] = {**bundled["coverage"], "tool_messages": {
        **bundled["coverage"]["tool_messages"], "tests_extra_note": "Extra note"}}
    profile = validate_profile({"format": 1, "phases": {
        "coverage": {"tool_messages": {"tests_extra_note": {"append": "Extra"}}},
    }}, bundled, languages=("python",))
    with pytest.raises(ConfigurationError, match="unavailable.*current settings"):
        prompts.resolve_agent_prompts("test_coverage", "python", hints={}, prompt_profile=profile)


def test_renderer_refuses_an_unvalidated_profile_value():
    with pytest.raises(TypeError, match="validated PromptProfile"):
        prompts.resolve_agent_prompts("test_coverage", "python", hints={}, prompt_profile={})


def test_placeholder_checks_need_no_python_3_11_template_methods(templates, monkeypatch):
    """Template.get_identifiers and is_valid exist only from Python 3.11 on,
    and Testuary supports 3.10. Without them, a profile still validates and
    its placeholder errors are still found."""
    import string

    from testuary.prompts import _placeholders

    for name in ("get_identifiers", "is_valid"):
        if hasattr(string.Template, name):
            monkeypatch.delattr(string.Template, name)
    profile = _profile(templates, user_prompt_template={"replace": "${file_path} $$ $lang_name"})
    assert "file_path" in profile.compose("coverage", "python", templates["coverage"])["user_prompt_template"]
    with pytest.raises(ConfigurationError, match="malformed placeholder syntax"):
        _profile(templates, system_prompt={"replace": "${unclosed"})
    with pytest.raises(ConfigurationError, match="unknown placeholder.*source"):
        _profile(templates, tool_messages={"tests_timeout": {"append": "$source"}})
    assert _placeholders("$a ${b} $$ $a") == (["a", "b"], True)
    assert _placeholders("$ cost") == ([], False)
