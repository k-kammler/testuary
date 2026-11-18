"""Generation evidence never gains equivalence from missing or redacted input."""
from dataclasses import FrozenInstanceError, replace
import json
from pathlib import Path

import pytest

from testuary.evidence import (
    capture_condition, composition_digest, require_matching_condition, reuse_problems,
    reuse_warnings,
)
from testuary.models import resolve_phase_model
from testuary.settings import PHASE_SECTIONS, ConfigurationError, RunSettings


TEMPLATES = {
    "system_prompt": "Use $lang_name\n## Hard rules\nUse real source.\n",
    "user_prompt_template": "$file_path\n$source",
    "tool_messages": {"tests_timeout": "Timed out after $timeout"},
}


def _model(phase="coverage", *, alias="chosen", api=None, fields=None, timeout=600):
    return resolve_phase_model(RunSettings(
        sections={
            "api": {"model": alias, **(api or {})},
            PHASE_SECTIONS[phase]: {"request_timeout_seconds": timeout},
        }, models={alias: {
            "provider": "openai", "model": "model-1", "api_url": "https://example.test/v1",
            "temperature": 0.2, "top_p": 0.9, "max_output_tokens": 80,
            "preserve_reasoning_content": False,
            **(fields or {}),
        }},
    ), phase)


def _capture(model, *, language="python", templates=None, extension_version=None):
    return capture_condition(model, language=language, templates=TEMPLATES if templates is None else templates,
                             extension_version=extension_version,
                             initial_generation={"seed_base": 0, "reasoning_grace_seconds": None}
                             if model.phase == "starter" else None)


def test_unknown_initial_slot_and_stream_controls_do_not_prove_equivalence():
    condition = capture_condition(_model("starter"), language="python", templates=TEMPLATES)
    assert "starter generation controls are unknown" in reuse_problems(condition.export(), condition)
    with pytest.raises(ConfigurationError, match="starter generation controls are unknown"):
        require_matching_condition(condition.export(), condition)


@pytest.mark.parametrize("field,value", [("seed_base", 17), ("reasoning_grace_seconds", 3.5)])
def test_initial_slot_base_and_stream_cutoff_are_generation_changes(field, value):
    model = _model("starter")
    before = capture_condition(model, language="python", templates=TEMPLATES,
                               initial_generation={"seed_base": 0, "reasoning_grace_seconds": None})
    after = capture_condition(model, language="python", templates=TEMPLATES,
                              initial_generation={"seed_base": 0, "reasoning_grace_seconds": None, field: value})
    assert reuse_problems(before.export(), after) == (f"request.initial_generation.{field} changed",)


@pytest.mark.parametrize("grace", [None, 0, 0.0])
def test_disabled_initial_grace_is_canonical_and_zero_seed_is_known(grace):
    condition = capture_condition(_model("starter"), language="python", templates=TEMPLATES,
                                  initial_generation={"seed_base": 0, "reasoning_grace_seconds": grace})
    assert condition.request["initial_generation"] == {"seed_base": 0, "reasoning_grace_seconds": None}
    require_matching_condition(_capture(_model("starter")).export(), condition)


def test_initial_controls_are_detached_and_frozen():
    inputs = {"seed_base": -3, "reasoning_grace_seconds": 2}
    condition = capture_condition(_model("starter"), language="java", templates=TEMPLATES,
                                  initial_generation=inputs)
    inputs["seed_base"] = 99
    assert condition.request["initial_generation"]["seed_base"] == -3
    with pytest.raises(TypeError):
        condition.request["initial_generation"]["seed_base"] = 9
    exported = condition.export()
    exported["request"]["initial_generation"]["seed_base"] = 11
    assert condition.request["initial_generation"]["seed_base"] == -3


@pytest.mark.parametrize("recorded_controls", [None, "absent"])
def test_old_initial_conditions_without_controls_cannot_match_known_inputs(recorded_controls):
    current = _capture(_model("starter"))
    assert current.request["values"]["seed"] is None
    recorded = current.export()
    if recorded_controls == "absent":
        del recorded["request"]["initial_generation"]
    else:
        recorded["request"]["initial_generation"] = recorded_controls
    with pytest.raises(ConfigurationError, match="request.initial_generation is unknown"):
        require_matching_condition(recorded, current)


@pytest.mark.parametrize("inputs", [
    {}, {"seed_base": 0}, {"reasoning_grace_seconds": None},
    {"seed_base": True, "reasoning_grace_seconds": None},
    {"seed_base": None, "reasoning_grace_seconds": None},
    {"seed_base": "secret-seed", "reasoning_grace_seconds": None},
    {"seed_base": 1.5, "reasoning_grace_seconds": None},
    {"seed_base": 0, "reasoning_grace_seconds": "secret-grace"},
    {"seed_base": 0, "reasoning_grace_seconds": True},
    {"seed_base": 0, "reasoning_grace_seconds": -1},
    {"seed_base": 0, "reasoning_grace_seconds": float("nan")},
    {"seed_base": 0, "reasoning_grace_seconds": float("inf")},
    {"seed_base": 0, "reasoning_grace_seconds": 10 ** 400},
    {"seed_base": 0, "reasoning_grace_seconds": None, "private": "secret-extra"},
    "secret-nonmapping",
])
def test_initial_condition_refuses_incomplete_or_invalid_controls_without_values(inputs):
    with pytest.raises(ConfigurationError) as caught:
        capture_condition(_model("starter"), language="python", templates=TEMPLATES,
                          initial_generation=inputs)
    assert "secret-" not in str(caught.value)


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
def test_initial_controls_cannot_be_attached_to_an_editing_phase(phase):
    with pytest.raises(ConfigurationError, match="only for the starter phase"):
        capture_condition(_model(phase), language="python", templates=TEMPLATES,
                          initial_generation={"seed_base": 0, "reasoning_grace_seconds": None})


def test_a_record_without_a_later_measurement_setting_names_it_as_lacking():
    """A record written before a measurement setting was recorded lacks it,
    which is reported as such, apart from settings that changed."""
    current = replace(_capture(_model()), measurement={
        "phase_end_mutation": "true", "suite_run_timeout_seconds": "120", "reduction_solver": "auto"})
    recorded = current.export()
    recorded["measurement"] = {"phase_end_mutation": "false", "reduction_solver": "auto"}
    assert reuse_problems(recorded, current) == (
        "recorded measurement settings lack: suite_run_timeout_seconds",
        "measurement settings changed: phase_end_mutation")


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_matching_conditions_roundtrip_without_ambient_reads(phase, tmp_path, monkeypatch):
    model = _model(phase)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TESTUARY_API_MODEL", "another")
    monkeypatch.setattr(Path, "read_bytes", lambda *a: pytest.fail("condition reread a file"))
    condition = _capture(model)
    recorded = json.loads(json.dumps(condition.export()))
    assert reuse_problems(recorded, condition) == ()
    require_matching_condition(recorded, condition)
    assert recorded["phase"] == phase
    assert recorded["request"]["values"]["temperature"] == 0.2
    assert recorded["request"]["values"]["max_output_tokens"] == 80


def test_condition_and_detached_exports_do_not_mutate_after_capture():
    condition = _capture(_model(fields={"request_options": {"custom": {"items": [1, 2]}}}))
    with pytest.raises(FrozenInstanceError):
        condition.phase = "mutation"
    with pytest.raises(TypeError):
        condition.request["request_options"]["custom"]["items"][0] = 9
    exported = condition.export()
    exported["request"]["request_options"]["custom"]["items"].append(3)
    exported["model"]["model"] = "changed"
    assert condition.export()["request"]["request_options"]["custom"]["items"] == [1, 2]
    assert condition.export()["model"]["model"] == "model-1"


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_transport_timeout_alias_layer_and_prices_are_not_generation_changes(phase):
    before = _capture(_model(phase, fields={"price_output": 1}, timeout=10))
    after = _capture(_model(phase, alias="renamed", fields={"price_output": 9}, timeout=1200))
    assert before.export() == after.export()
    require_matching_condition(before.export(), after)
    text = json.dumps(after.export())
    for field in ("model_alias", "sources", "price_output", "request_timeout_seconds"):
        assert field not in text


def test_only_the_effective_output_cap_enters_generation_condition():
    model = _model()
    condition = _capture(model)
    changed = replace(model, values={**model.values, "model_max_output_tokens": 800,
                                    "phase_max_output_tokens": 80})
    require_matching_condition(condition.export(), _capture(changed))
    changed = replace(changed, values={**changed.values, "max_output_tokens": 40})
    assert reuse_problems(condition.export(), _capture(changed)) == ("request.values.max_output_tokens changed",)


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_credential_rotation_alone_does_not_change_condition(phase):
    before = _capture(_model(phase, fields={
        "api_key": "private-first", "api_url": "https://user:first-password@example.test/v1",
    }))
    after = _capture(_model(phase, fields={
        "api_key": "private-second", "api_url": "https://next:second-password@example.test/v1",
    }))
    require_matching_condition(before.export(), after)
    for text in (repr(after), json.dumps(after.export())):
        for secret in ("private-first", "private-second", "first-password", "second-password", "user:", "next:"):
            assert secret not in text


@pytest.mark.parametrize("query", ["key=private-query", "deployment=different", "api-version=2026"])
def test_endpoint_query_routing_never_becomes_equivalent_through_redaction(query):
    before = _capture(_model(fields={"api_url": f"https://example.test/v1?{query}"}))
    after = _capture(_model(fields={"api_url": "https://example.test/v1?different=secret"}))
    assert before.model == after.model
    assert before.service_identity_known is False
    problems = reuse_problems(before.export(), after)
    assert any("service identity is unknown" in problem for problem in problems)
    with pytest.raises(ConfigurationError) as caught:
        require_matching_condition(before.export(), after)
    for text in (json.dumps(before.export()), json.dumps(after.export()), str(caught.value)):
        assert query not in text
        assert "different=secret" not in text


@pytest.mark.parametrize("options", [
    {"extra_headers": {"X-Api-Key": "private-header"}},
    {"custom": {"X-Api-Key": "private-header"}},
    {"custom": [{"X-Api-Key": "private-header"}]},
])
def test_redacted_option_fields_are_unknown_not_silently_equivalent(options):
    current = _capture(_model(fields={"request_options": options}))
    assert current.request_options_known is False
    assert "private-header" not in json.dumps(current.export())
    with pytest.raises(ConfigurationError, match="request options are unknown") as caught:
        require_matching_condition(current.export(), current)
    assert "private-header" not in str(caught.value)


@pytest.mark.parametrize("fields,path", [
    ({"deployment": "other"}, "model.deployment"),
    ({"provider": "gateway"}, "model.provider"),
    ({"model": "other-model"}, "model.model"),
    ({"protocol": "openai-responses"}, "model.protocol"),
    ({"temperature": 0}, "request.values.temperature"),
    ({"top_p": 0}, "request.values.top_p"),
    ({"max_output_tokens": 40}, "request.values.max_output_tokens"),
    ({"preserve_reasoning_content": True}, "request.values.preserve_reasoning_content"),
    ({"reasoning_field_names": "custom"}, "request.values.reasoning_field_names"),
    ({"request_options": {"custom": "private-option"}}, "request.request_options"),
])
def test_effective_generation_changes_name_fields_not_values(fields, path):
    before = _capture(_model())
    after = _capture(_model(fields=fields))
    assert any(path in problem for problem in reuse_problems(before.export(), after))
    with pytest.raises(ConfigurationError) as caught:
        require_matching_condition(before.export(), after)
    assert "new output/run" in str(caught.value)
    assert "explicitly compatible stage" in str(caught.value)
    assert "private-option" not in str(caught.value)


@pytest.mark.parametrize("api_url", [
    "https://example.test:8443/v1", "http://127.0.0.1:9090/v1", "https://other.test/v2",
])
def test_the_provider_and_model_name_decide_and_a_changed_address_is_a_warning(api_url):
    before = _capture(_model())
    after = _capture(_model(fields={"api_url": api_url}))
    assert reuse_problems(before.export(), after) == ()
    require_matching_condition(before.export(), after)
    warnings = reuse_warnings(before.export(), after)
    assert len(warnings) == 1 and "model address (api_url)" in warnings[0]
    assert api_url not in warnings[0] and "example.test" not in warnings[0]
    assert reuse_warnings(after.export(), after) == ()


@pytest.mark.parametrize("field,value", [("seed", 4), ("reasoning", "on"),
                                         ("reasoning_effort", "high"), ("reasoning_budget_tokens", 99)])
def test_each_resolved_reasoning_or_seed_field_enters_the_condition(field, value):
    model = _model()
    changed = replace(model, values={**model.values, field: value})
    assert f"request.values.{field} changed" in reuse_problems(_capture(model).export(), _capture(changed))


def test_reasoning_body_and_dialect_are_captured_separately():
    model = _model()
    changed = replace(model, reasoning_dialect="vllm", reasoning_body={"thinking_token_budget": 99})
    problems = reuse_problems(_capture(model).export(), _capture(changed))
    assert "model.reasoning_dialect changed" in problems
    assert "request.reasoning_body has unsupported fields" not in problems
    assert any("request.reasoning_body" in problem for problem in problems)


def test_phase_and_language_differences_are_not_ignored():
    original = _capture(_model("coverage"))
    assert "phase changed" in reuse_problems(original.export(), _capture(_model("mutation")))
    assert "language changed" in reuse_problems(original.export(), _capture(_model(), language="java"))


@pytest.mark.parametrize("recorded", [None, {}, [], "private-old-input", {"format": True}, {"format": 2}])
def test_missing_or_unsupported_records_never_gain_equivalence(recorded):
    with pytest.raises(ConfigurationError) as caught:
        require_matching_condition(recorded, _capture(_model()))
    assert "unknown" in str(caught.value)
    assert "private-old-input" not in str(caught.value)


def test_missing_unset_field_is_unknown_not_equivalent_to_unset():
    current = _capture(_model())
    recorded = current.export()
    assert recorded["request"]["values"]["seed"] is None
    del recorded["request"]["values"]["seed"]
    assert reuse_problems(recorded, current) == ("request.values.seed is unknown",)


def test_unsupported_and_malformed_record_fields_are_refused_without_values():
    current = _capture(_model())
    recorded = current.export()
    recorded["old_shape"] = "private-extra"
    recorded["request"] = "private-broken"
    problems = reuse_problems(recorded, current)
    assert "generation condition has unsupported fields" in problems
    assert "request is unknown or invalid" in problems
    assert "private" not in ";".join(problems)


@pytest.mark.parametrize("previous,current", [(None, None), (None, "1.0"), ("1.0", None), ("1.0", "2.0")])
def test_an_added_provider_needs_matching_known_distribution_version(previous, current):
    model = _model(fields={"chat_model": "extension.chat:Model"})
    before = _capture(model, extension_version=previous)
    after = _capture(model, extension_version=current)
    assert reuse_problems(before.export(), after)
    require_matching_condition(_capture(model, extension_version="1.0").export(),
                               _capture(model, extension_version="1.0"))


@pytest.mark.parametrize("part", ["system_prompt", "user_prompt_template", "tool_messages"])
def test_all_text_components_and_hints_affect_composition(part):
    changed = {**TEMPLATES, part: {"tests_timeout": "Changed"} if part == "tool_messages" else "Changed"}
    before = _capture(_model())
    after = _capture(_model(), templates=changed)
    assert reuse_problems(before.export(), after) == ("prompt_composition changed",)
    assert composition_digest({**TEMPLATES, "system_prompt": TEMPLATES["system_prompt"]
                               + "\n\n## Additional guidance\n\nHint"}) != before.prompt_composition


def test_digest_uses_text_not_input_key_order_or_file_identity():
    reordered = {key: TEMPLATES[key] for key in reversed(TEMPLATES)}
    assert composition_digest(reordered) == composition_digest(TEMPLATES)
    missing_messages = {key: value for key, value in TEMPLATES.items() if key != "tool_messages"}
    assert composition_digest(missing_messages) == composition_digest({**missing_messages, "tool_messages": {}})
    assert composition_digest({**TEMPLATES, "system_prompt": TEMPLATES["system_prompt"].replace("\n", "\r\n")}) != (
        composition_digest(TEMPLATES))


@pytest.mark.parametrize("templates", [
    None, {}, {"system_prompt": "only"}, {**TEMPLATES, "unexpected": "ignored?"},
    {**TEMPLATES, "system_prompt": None}, {**TEMPLATES, "user_prompt_template": 1},
    {**TEMPLATES, "tool_messages": []}, {**TEMPLATES, "tool_messages": {"name": None}},
    {**TEMPLATES, "tool_messages": {1: "text"}}, {**TEMPLATES, "system_prompt": "\ud800"},
])
def test_unavailable_or_invalid_text_cannot_create_a_composition(templates):
    with pytest.raises(ConfigurationError):
        composition_digest(templates)


@pytest.mark.parametrize("language", [None, "", "   ", 1])
def test_condition_requires_a_language(language):
    with pytest.raises(ConfigurationError):
        _capture(_model(), language=language)


@pytest.mark.parametrize("version", ["", "   ", 1])
def test_invalid_extension_version_is_not_treated_as_known(version):
    with pytest.raises(ConfigurationError):
        _capture(_model(), extension_version=version)


def test_guard_preserves_paid_response_and_exhausted_accounting_bytes(tmp_path):
    current = _capture(_model())
    original = _capture(_model(fields={"temperature": 0.8})).export()
    artifacts = {name: tmp_path / name for name in ("condition.json", "response.json", "requests.jsonl")}
    for name, content in (("condition.json", json.dumps(original)), ("response.json", '{"paid":"saved"}'),
                          ("requests.jsonl", '{"attempts":100,"exhausted":true}\n')):
        artifacts[name].write_text(content, encoding="utf-8")
    before = {name: path.read_bytes() for name, path in artifacts.items()}
    with pytest.raises(ConfigurationError):
        require_matching_condition(json.loads(artifacts["condition.json"].read_bytes()), current)
    assert {name: path.read_bytes() for name, path in artifacts.items()} == before


def test_results_without_a_known_condition_are_refused_with_the_reason():
    from testuary.evidence import reuse_problems

    for recorded, reason in ((None, "no generation records"), ({"format": 0}, "format is unknown")):
        [problem] = reuse_problems(recorded, None)
        assert reason in problem and "cannot be resumed or reused" in problem and "fresh run" in problem
