"""An explicit catalogue provider must reach every existing request caller."""
import pytest
from langchain_core.messages import AIMessageChunk

from starter_models import generate

from testuary import _chat_models
from testuary import config
from testuary import models
from testuary.settings import ConfigurationError, RunSettings


def _settings(alias, provider):
    return RunSettings(sections={"api": {"model": alias}}, models={
        "openai": {"api_url": "https://right.example/v1", "api_key": "right-key", "max_output_tokens": 4096},
        "ollama": {"api_url": "https://wrong.example/v1", "api_key": "wrong-key", "max_output_tokens": 8192},
        alias: {"provider": provider, "model": "chosen"},
    })


@pytest.mark.parametrize("alias, provider", [
    ("ollama_local", "openai"), ("ollama_local", " OpenAI "),
    ("unprefixed", "openai"), ("foreign_alias", "openai"),
])
def test_explicit_provider_drives_the_current_api_reader(alias, provider):
    settings = _settings(alias, provider)
    with config.use_run_settings(settings):
        api = models.get_api_config()
    assert api["provider"] == "openai"
    assert api["model"] == "chosen"
    assert api["api_url"] == "https://right.example/v1"
    assert api["api_key"] == "right-key"
    assert api["max_output_tokens"] == 4096
    assert settings.models[alias]["provider"] == provider


@pytest.mark.parametrize("provider", ["", None, False, 7])
def test_invalid_explicit_provider_cannot_fall_back_to_the_alias_prefix(provider):
    with config.use_run_settings(_settings("ollama_local", provider)):
        with pytest.raises(ConfigurationError, match="provider"):
            models.get_api_config()


def test_unprefixed_alias_without_a_provider_is_refused():
    settings = RunSettings(sections={"api": {"model": "unprefixed"}}, models={"unprefixed": {"model": "chosen"}})
    with config.use_run_settings(settings):
        with pytest.raises(ConfigurationError, match="provider"):
            models.get_api_config()


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_actual_generation_builders_use_the_declared_provider(phase, monkeypatch):
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
    with config.use_run_settings(_settings("ollama_local", "openai")):
        api = models.get_api_config()
        if phase == "starter":
            text, _ = generate(api, "prompt", temperature=.7, seed=3, timeout=60,
                                               max_output_tokens=api["max_output_tokens"])
            assert text == "answer"
        else:
            resolved = models.resolve_phase_model(config.run_settings(), phase)
            assert isinstance(models.build_resolved_chat_model(resolved, api_key=api["api_key"]), Model)
    assert len(observed) == 1
    assert observed[0]["model"] == "chosen"
    assert observed[0]["api_url"] == "https://right.example/v1"
    assert observed[0]["api_key"] == "right-key"
    assert observed[0]["max_tokens"] == 4096
