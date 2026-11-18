"""Catalogue credentials resolve from the captured invocation, never records."""
import json

import pytest
from langchain_core.messages import AIMessageChunk

from starter_models import generate

from testuary import _chat_models
from testuary import config
from testuary import models as models_module
from testuary.settings import ConfigurationError, RunSettings


@pytest.mark.parametrize("provider_fields, alias_fields, env, expected", [
    ({}, {}, {"TESTUARY_OPENAI_API_KEY": "provider-env"}, "provider-env"),
    ({"api_key_env": "CUSTOM_KEY"}, {}, {"CUSTOM_KEY": "custom-env", "TESTUARY_OPENAI_API_KEY": "provider-env"}, "custom-env"),
    ({"api_key_env": "PROVIDER_KEY"}, {"api_key_env": "ALIAS_KEY"}, {"ALIAS_KEY": "alias-env"}, "alias-env"),
    ({"api_key_env": "CUSTOM_KEY", "api_key": "provider-catalogue"}, {}, {"CUSTOM_KEY": "custom-env"}, "provider-catalogue"),
    ({"api_key": "provider-catalogue"}, {"api_key": "alias-catalogue"}, {"TESTUARY_OPENAI_API_KEY": "provider-env"}, "alias-catalogue"),
    ({"api_key": "provider-catalogue"}, {"api_key": ""}, {"TESTUARY_OPENAI_API_KEY": "provider-env"}, ""),
    ({"api_key_env": "PROVIDER_KEY"}, {"api_key_env": ""}, {"TESTUARY_OPENAI_API_KEY": "provider-env"}, "provider-env"),
    ({}, {}, {}, ""),
])
def test_actual_alias_credential_precedence(provider_fields, alias_fields, env, expected):
    settings = RunSettings(sections={"api": {"model": "openai_example"}}, environment=env, models={
        "openai": provider_fields, "openai_example": {"provider": "openai", "model": "chosen", **alias_fields},
    })
    with config.use_run_settings(settings):
        assert models_module.get_api_config()["api_key"] == expected


@pytest.mark.parametrize("models, environment", [
    ({"openai": {"api_key_env": "MISSING"}}, {}),
    ({"openai": {"api_key_env": "MISSING"}}, {"TESTUARY_OPENAI_API_KEY": "fallback-key"}),
    ({"openai_example": {"provider": "openai", "api_key_env": "MISSING"}}, {}),
    ({"openai_example": {"provider": "openai", "api_key_env": "EMPTY"}}, {"EMPTY": ""}),
    ({"openai_example": {"provider": "openai", "api_key_env": "EMPTY"}}, {"EMPTY": "   "}),
    ({"openai_example": {"provider": "openai", "api_key_env": "MISSING", "api_key": "sentinel-credential"}}, {}),
])
def test_configured_missing_key_variable_refuses_before_any_request(models, environment):
    settings = RunSettings(sections={"api": {"model": "openai_example"}},
                           models={"openai_example": {"provider": "openai", "model": "chosen"}, **models}, environment=environment)
    with config.use_run_settings(settings):
        with pytest.raises(ConfigurationError, match="api_key_env") as caught:
            models_module.get_api_config()
    assert "sentinel-credential" not in str(caught.value)


@pytest.mark.parametrize("key, value", [
    ("api_key_env", None), ("api_key_env", False), ("api_key_env", 7),
    ("api_key", None), ("api_key", False), ("api_key", 7),
])
def test_bad_credential_fields_refuse_without_values(key, value):
    settings = RunSettings(sections={"api": {"model": "openai_example"}},
                           models={"openai_example": {"provider": "openai", key: value}})
    with config.use_run_settings(settings):
        with pytest.raises(ConfigurationError, match=key):
            models_module.get_api_config()


def test_api_key_env_for_raw_models_uses_the_same_reader():
    settings = RunSettings(sections={"api": {"model": "raw-model", "provider": "openai"}},
                           models={"openai": {"api_key_env": "CUSTOM_KEY"}},
                           environment={"CUSTOM_KEY": "custom-env"})
    with config.use_run_settings(settings):
        assert models_module.get_api_config()["api_key"] == "custom-env"


def test_an_explicit_empty_catalogue_key_clears_the_raw_model_environment_key():
    settings = RunSettings(sections={"api": {"model": "raw-model", "provider": "openai"}},
                           models={"openai": {"api_key": ""}},
                           environment={"TESTUARY_OPENAI_API_KEY": "provider-env"})
    with config.use_run_settings(settings):
        assert models_module.get_api_config()["api_key"] == ""


def test_credential_resolution_uses_the_captured_environment_not_ambient(monkeypatch):
    settings = RunSettings(sections={"api": {"model": "openai_example"}},
                           models={"openai_example": {"provider": "openai", "api_key_env": "CUSTOM_KEY"}},
                           environment={"CUSTOM_KEY": "captured-key"})
    monkeypatch.setenv("CUSTOM_KEY", "changed-key")
    monkeypatch.setenv("TESTUARY_OPENAI_API_KEY", "other-key")
    with config.use_run_settings(settings):
        assert models_module.get_api_config()["api_key"] == "captured-key"
    assert "captured-key" not in json.dumps(settings.export()) + repr(settings)


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_actual_request_builders_receive_the_captured_named_key(phase, monkeypatch):
    settings = RunSettings(sections={"api": {"model": "openai_example"}},
                           models={"openai_example": {"provider": "openai", "api_key_env": "CUSTOM_KEY"}},
                           environment={"CUSTOM_KEY": "captured-key"})
    monkeypatch.setenv("CUSTOM_KEY", "changed-key")
    observed = []

    class Model:
        def stream(self, messages):
            yield AIMessageChunk(content="answer", usage_metadata={
                "input_tokens": 1, "output_tokens": 1, "total_tokens": 2,
            })

    def build(**kwargs):
        observed.append(kwargs)
        return Model()

    monkeypatch.setattr(_chat_models, "build_chat_model", build)
    with config.use_run_settings(settings):
        api = models_module.get_api_config()
        if phase == "starter":
            answer, usage = generate(api, "prompt", temperature=.7, seed=3, timeout=60)
            assert answer == "answer"
            assert "captured-key" not in json.dumps(usage)
        else:
            resolved = models_module.resolve_phase_model(config.run_settings(), phase)
            assert isinstance(models_module.build_resolved_chat_model(resolved, api_key=api["api_key"]), Model)
    assert len(observed) == 1
    assert observed[0]["api_key"] == "captured-key"
