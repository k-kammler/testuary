"""Canonical public settings activate together."""
import json

import pytest

from testuary.models import native_request_body, resolve_phase_model
from testuary.settings import RunSettings


# The sections that take model controls. [finishing] sends no model request
# and refuses them.
SECTIONS = ("api", "openai", "ollama", "vllm", "starter", "coverage", "mutation")


@pytest.mark.parametrize("section", SECTIONS)
def test_canonical_controls_are_public_and_detached(section):
    controls = {"temperature": 0, "top_p": 0, "request_timeout_seconds": 12,
                "reasoning": "on", "reasoning_budget_tokens": 13,
                "reasoning_effort": "", "protocol": "openai-chat",
                "request_options": '{"top_k": 7}', "max_output_tokens": 0}
    settings = RunSettings(sections={section: controls})
    assert settings.sections[section]["temperature"] == "0"
    assert settings.sections[section]["reasoning_budget_tokens"] == "13"
    assert settings.export()["sections"][section]["reasoning_budget_tokens"] == "13"
    controls["temperature"] = 99
    assert settings.sections[section]["temperature"] == "0"


def test_reasoning_budget_safe_export_is_not_mistaken_for_a_credential():
    settings = RunSettings(models={"alias": {"reasoning_budget_tokens": 13, "api_key": "private-sentinel"}})
    assert settings.export()["models"]["alias"] == {"reasoning_budget_tokens": 13}
    assert "private-sentinel" not in json.dumps(settings.export())


@pytest.mark.parametrize("raw", [
    '{"api_key": "private-sentinel", "top_k": 7}',
    '{"nested": [{"authorization_token": "private-sentinel"}], "top_k": 7}',
    'invalid-private-sentinel', '["private-sentinel"]',
])
def test_json_text_request_options_are_secret_safe_before_preflight(raw):
    settings = RunSettings(sections={"api": {"request_options": raw}},
                           models={"alias": {"request_options": raw}})
    for exported in (settings.export(), repr(settings)):
        assert "private-sentinel" not in str(exported)
    assert settings.sections["api"]["request_options"] == raw
    assert settings.models["alias"]["request_options"] == raw


@pytest.mark.parametrize("controls,match", [
    ({"reasoning_budget_tokens": 0}, "reasoning = off"),
    ({"temperature": float("nan")}, "expected a number"),
    ({"top_p": 1.1}, "between zero and one"),
    ({"reasoning": "false"}, "on, off or empty"),
    ({"temperature": False}, "expected a number"),
    ({"request_timeout_seconds": ""}, "positive"),
    ({"request_timeout_seconds": 0}, "positive"),
    ({"reasoning_effort": False}, "effort text"),
    ({"reasoning_effort": 0}, "effort text"),
    ({"reasoning_effort": "0"}, "effort text"),
])
def test_invalid_public_controls_refuse(controls, match):
    with pytest.raises(ValueError, match=match):
        RunSettings(sections={"starter": controls})


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_invalid_catalogue_effort_is_refused_by_every_phase(phase):
    settings = RunSettings(sections={"api": {"model": "openai_example"}}, models={
        "openai_example": {"provider": "openai", "model": "chosen", "reasoning_dialect": "openai", "reasoning_effort": "0"},
    })
    with pytest.raises(ValueError, match="effort text"):
        resolve_phase_model(settings, phase)


def test_actual_packaged_defaults_do_not_send_new_controls():
    settings = RunSettings(sections={"api": {"model": "openai_example"}},
                           models={"openai_example": {"provider": "openai", "model": "chosen"}})
    for phase in ("starter", "coverage", "mutation"):
        model = resolve_phase_model(settings, phase)
        assert model.values["temperature"] is None
        assert model.values["request_timeout_seconds"] == (600 if phase == "starter" else 3600)
        assert model.values["reasoning"] is None and model.values["reasoning_budget_tokens"] is None
        assert model.values["top_p"] is None and model.values["seed"] is None
        assert model.protocol == "openai-chat"
        assert model.reasoning_body == model.request_options == {}
    cloud = resolve_phase_model(RunSettings(
        sections={"api": {"model": "cloud"}},
        models={"cloud": {"provider": "openai", "model": "chosen",
                          "api_url": "https://ollama.com/v1"}}), "starter")
    assert cloud.protocol == "ollama-native"
    assert native_request_body(cloud, [], slot_seed=5)["options"] == {"seed": 5}
