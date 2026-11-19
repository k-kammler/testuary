"""All generation callers share the smaller-positive-output-cap rule."""
import asyncio
import pytest
from langchain_core.messages import HumanMessage

import test_coverage_agent as coverage_harness
import test_mutation_agent as mutation_harness
from testuary import _chat_models
from testuary import config as config_module
from testuary import models
from testuary import starters
from testuary.settings import ConfigurationError, RunSettings


CAPS = [
    (None, None, None), (0, 0, None), ("", "0", None),
    (0, 4096, 4096), (4096, 0, 4096), (None, 4096, 4096),
    (4096, "", 4096), (4096, 1024, 1024), (512, 4096, 512),
    ("4096", "512", 512),
]


@pytest.mark.parametrize("model_cap, phase_cap, expected", CAPS)
@pytest.mark.parametrize("harness", [coverage_harness, mutation_harness], ids=["coverage", "mutation"])
def test_actual_editing_builder_uses_positive_caps_only(
    model_cap, phase_cap, expected, harness, tmp_path, monkeypatch, in_run,
):
    # The model's cap from its catalogue entry, the phase's from its
    # section. None leaves the key out.
    phase = "coverage" if harness is coverage_harness else "mutation"
    entry = {"provider": "openai", "model": "fixture", "api_url": "http://127.0.0.1:1/v1"}
    if model_cap is not None:
        entry["max_output_tokens"] = model_cap
    sections = {"api": {"model": "capped"}}
    if phase_cap is not None:
        sections[phase] = {"max_output_tokens": phase_cap}
    in_run(overrides=sections, models={"capped": entry})
    agent = harness._make_agent(tmp_path)
    seen = []
    monkeypatch.setattr(_chat_models, "build_chat_model", lambda **k: seen.append(k) or "model")
    assert agent.build_chat_model() == "model"
    assert len(seen) == 1
    assert seen[0]["max_tokens"] == expected


@pytest.mark.parametrize("model_cap, phase_cap, expected", CAPS)
def test_starter_resolution_uses_the_same_cap(model_cap, phase_cap, expected):
    assert starters._resolve_output_cap(
        {"max_output_tokens": phase_cap}, {"max_output_tokens": model_cap},
    ) == expected


@pytest.mark.parametrize("cap", [-1, "-1", 1.5, "1.5", True, "not-an-integer", {}, []])
@pytest.mark.parametrize("side", ["model", "phase"])
def test_invalid_caps_are_refused_before_constructing_the_editing_client(cap, side, monkeypatch):
    # A phase's cap is refused with its settings, a model's when the phase
    # resolves its model.
    monkeypatch.setattr(_chat_models, "build_chat_model", lambda **k: pytest.fail("client built"))
    entry = {"provider": "openai", "model": "fixture", "api_url": "http://127.0.0.1:1/v1"}
    phase = {}
    (entry if side == "model" else phase)["max_output_tokens"] = cap
    with pytest.raises(ValueError, match="max_output_tokens"):
        settings = RunSettings(sections={"api": {"model": "capped"}, "coverage": phase},
                               models={"capped": entry})
        models.build_resolved_chat_model(models.resolve_phase_model(settings, "coverage"), api_key="EMPTY")
    with pytest.raises(ConfigurationError, match="max_output_tokens"):
        starters._resolve_output_cap(phase, entry)


@pytest.mark.parametrize("model_cap, phase_cap, expected", [(0, 0, None), (4096, 0, 4096), (4096, 512, 512)])
def test_actual_chat_client_payload_keeps_the_existing_double_cap(model_cap, phase_cap, expected):
    model = _chat_models.build_chat_model(
        model="fixture", api_url="http://127.0.0.1:1/v1", api_key="EMPTY", timeout_seconds=3600,
        max_tokens=models.resolve_output_cap(phase_cap, model_cap),
    )
    try:
        payload = model._get_request_payload([HumanMessage("task")])
        if expected is None:
            assert "max_completion_tokens" not in payload
            assert "max_tokens" not in payload.get("extra_body", {})
        else:
            assert payload["max_completion_tokens"] == expected
            assert payload["extra_body"]["max_tokens"] == expected
    finally:
        model.http_client.close()
        asyncio.run(model.http_async_client.aclose())


@pytest.mark.parametrize("section", ["api", "openai", "starter", "coverage", "mutation"])
@pytest.mark.parametrize("blank", ["", "  "])
def test_blank_caps_resolve_through_authoritative_settings_and_real_readers(section, blank, monkeypatch):
    settings = RunSettings(sections={
        "api": {"model": "openai_example", **({"max_output_tokens": blank} if section == "api" else {})},
        **({section: {"max_output_tokens": blank}} if section != "api" else {}),
    }, models={"openai_example": {"provider": "openai", "model": "fixture", "max_output_tokens": 4096}})
    seen = []
    monkeypatch.setattr(_chat_models, "build_chat_model", lambda **k: seen.append(k) or "model")
    with config_module.use_run_settings(settings):
        api = models.get_api_config()
        for phase in ("coverage", "mutation"):
            resolved = models.resolve_phase_model(settings, phase)
            assert models.build_resolved_chat_model(resolved, api_key=api["api_key"]) == "model"
        assert starters._resolve_output_cap(config_module.get_agent_config("test_initial"), api) == 4096
    assert [request["max_tokens"] for request in seen] == [4096, 4096]
