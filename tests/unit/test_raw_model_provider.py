"""Raw API models require explicit selection, without guessing a provider."""
import httpx
import pytest
from langchain_core.messages import AIMessageChunk

from starter_models import generate

from testuary import _chat_models
from testuary import config
from testuary import models
from testuary.settings import ConfigurationError, RunSettings


@pytest.mark.parametrize("model", ["gpt-4o", "openai_uncatalogued", "unprefixed"])
@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_uncatalogued_models_refuse_before_a_request_constructor(model, phase, monkeypatch):
    built = []
    monkeypatch.setattr(_chat_models, "build_chat_model", lambda **kwargs: built.append(kwargs))
    def refuse_http(*args, **kwargs):
        raise AssertionError("unexpected HTTP route for an unconfigured provider")
    monkeypatch.setattr(httpx, "stream", refuse_http)
    settings = RunSettings(sections={"api": {"model": model}}, models={}, environment={})
    with config.use_run_settings(settings):
        with pytest.raises(ConfigurationError, match=r"is not an alias in .*models\.ini.*\[api\] provider"):
            api = models.get_api_config()
            if phase == "starter":
                generate(api, "prompt", temperature=.7, seed=3, timeout=60)
            else:
                resolved = models.resolve_phase_model(config.run_settings(), phase)
                models.build_resolved_chat_model(resolved, api_key=api["api_key"])
    assert built == []


@pytest.mark.parametrize("model", ["", "   "])
def test_empty_model_does_not_select_the_providers_model(model):
    settings = RunSettings(sections={"api": {"model": model, "provider": "openai"}}, models={})
    with config.use_run_settings(settings):
        with pytest.raises(ConfigurationError, match=r"\[api\] model"):
            models.get_api_config()


@pytest.mark.parametrize("provider", ["", "   "])
def test_explicit_empty_provider_does_not_infer_a_model_prefix(provider):
    settings = RunSettings(sections={"api": {"model": "openai_uncatalogued", "provider": provider}}, models={})
    with config.use_run_settings(settings):
        with pytest.raises(ConfigurationError, match=r"\[api\] provider"):
            models.get_api_config()


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_raw_model_uses_provider_catalogue_without_substituting_the_model(phase, monkeypatch):
    settings = RunSettings(
        sections={"api": {"model": "custom:model", "provider": " OPENAI "},
                  "openai": {"api_url": "https://config.example/v1"}},
        models={"openai": {"api_url": "https://catalogue.example/v1", "model": "wrong-model",
                           "api_key_env": "CUSTOM_KEY", "max_output_tokens": "1024"}},
        environment={"CUSTOM_KEY": "captured-key"},
    )
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
        api = models.get_api_config()
        assert api["provider"] == "openai"
        assert api["model"] == api["model_alias"] == "custom:model"
        assert api["api_url"] == "https://catalogue.example/v1"
        assert api["max_output_tokens"] == 1024
        if phase == "starter":
            answer, _ = generate(api, "prompt", temperature=.7, seed=3, timeout=60)
            assert answer == "answer"
        else:
            resolved = models.resolve_phase_model(config.run_settings(), phase)
            assert isinstance(models.build_resolved_chat_model(resolved, api_key=api["api_key"]), Model)
    assert len(observed) == 1
    assert observed[0]["api_url"] == "https://catalogue.example/v1"
    assert observed[0]["model"] == "custom:model"
    assert observed[0]["api_key"] == "captured-key"


def test_explicit_endpoint_override_wins_over_raw_provider_catalogue():
    settings = RunSettings(
        sections={"api": {"model": "custom:model", "provider": "openai", "api_url": "https://override.example/v1"}},
        models={"openai": {"api_url": "https://catalogue.example/v1"}},
    )
    with config.use_run_settings(settings):
        assert models.get_api_config()["api_url"] == "https://override.example/v1"
