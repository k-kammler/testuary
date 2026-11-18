"""Model identity is explicit, invocation-local and safe to record."""
from dataclasses import FrozenInstanceError
import json

import pytest

from testuary import settings as settings_module
from testuary.models import resolve_model_identity, resolve_phase_model
from testuary.settings import ConfigurationError, RunSettings


def _settings(*, model="openai_example", api=None, models=None, environment=None):
    return RunSettings(
        sections={"api": {"model": model, **(api or {})}},
        models={} if models is None else models,
        environment={} if environment is None else environment,
    )


def test_an_alias_explicit_provider_wins_over_its_name_prefix():
    settings = _settings(model="ollama_local", models={
        "ollama_local": {"provider": "openai", "model": "chosen"},
        "openai": {"api_url": "https://openai.example/v1"},
        "ollama": {"api_url": "https://wrong.example/v1"},
    })
    identity = resolve_model_identity(settings)
    assert identity.provider == "openai"
    assert identity.model == "chosen"
    assert identity.model_alias == "ollama_local"
    assert identity.api_url == "https://openai.example/v1"


def test_an_alias_names_its_provider_and_defaults_its_model_to_its_name():
    identity = resolve_model_identity(_settings(models={"openai_example": {"provider": "openai"}}))
    assert identity.provider == "openai"
    assert identity.model == identity.model_alias == "openai_example"
    assert identity.api_url == "https://api.openai.com/v1"


def test_a_raw_api_model_requires_an_explicit_provider_even_if_it_has_a_prefix():
    for model in ("gpt-4o", "openai_not_in_catalogue"):
        with pytest.raises(ConfigurationError, match=r"is not an alias in .*models\.ini.*\[api\] provider"):
            resolve_model_identity(_settings(model=model))


def test_a_raw_model_uses_its_explicit_provider_without_model_substitution():
    identity = resolve_model_identity(_settings(model="custom:model", api={"provider": "openai"}))
    assert identity.model == identity.model_alias == "custom:model"
    assert identity.provider == "openai"


@pytest.mark.parametrize("provider", ["", 1, False, None])
def test_an_invalid_explicit_alias_provider_never_falls_back_to_its_prefix(provider):
    with pytest.raises(ConfigurationError, match="provider"):
        resolve_model_identity(_settings(models={"openai_example": {"provider": provider}}))


@pytest.mark.parametrize("alias", ["example", "openai_example", "my_model"])
def test_an_alias_without_a_provider_is_refused_whatever_its_name(alias):
    # The alias name is only a label: "my_model" does not become provider "my".
    with pytest.raises(ConfigurationError,
                       match=rf"model alias \[{alias}\] needs provider = ollama, openai or vllm"):
        resolve_model_identity(_settings(model=alias, models={alias: {"model": "chosen"}}))



@pytest.mark.parametrize("api, provider, alias, endpoint", [
    ({}, {}, {}, "https://api.openai.com/v1"),
    ({}, {"api_url": "https://provider.example/v1"}, {}, "https://provider.example/v1"),
    ({}, {"api_url": "https://provider.example/v1"}, {"api_url": "https://alias.example/v1"},
     "https://alias.example/v1"),
    ({"api_url": "http://localhost:8123/v1"}, {"api_url": "https://provider.example/v1"},
     {"api_url": "https://alias.example/v1"}, "http://localhost:8123/v1"),
])
def test_endpoint_precedence(api, provider, alias, endpoint):
    identity = resolve_model_identity(_settings(api=api, models={
        "openai": provider, "openai_example": {"provider": "openai", **alias},
    }))
    assert identity.api_url == endpoint


def test_an_explicit_empty_endpoint_is_not_replaced_with_the_alias_endpoint():
    with pytest.raises(ConfigurationError, match="absolute HTTP"):
        resolve_model_identity(_settings(api={"api_url": ""}, models={
            "openai_example": {"provider": "openai", "api_url": "https://alias.example/v1"},
        }))


@pytest.mark.parametrize("endpoint", ["relative/v1", "file:///tmp/model", "https://", "https://host:secret/v1"])
def test_invalid_endpoint_errors_do_not_repeat_the_endpoint(endpoint):
    with pytest.raises(ConfigurationError, match="absolute HTTP") as caught:
        resolve_model_identity(_settings(api={"provider": "openai", "api_url": endpoint}))
    assert endpoint not in str(caught.value)


def test_custom_provider_aliases_need_no_package_registration():
    identity = resolve_model_identity(_settings(model="gateway-model", models={
        "gateway-model": {"provider": "private-gateway", "model": "model/1"},
        "private-gateway": {"api_url": "http://localhost:9000/v1", "deployment": "deployment-1"},
    }))
    assert identity.provider == "private-gateway"
    assert identity.model == "model/1"
    assert identity.deployment == "deployment-1"


def test_deployment_precedence_and_explicit_empty_are_preserved():
    catalogues = {"openai": {"deployment": "provider"}, "openai_example": {"provider": "openai", "deployment": "alias"}}
    assert resolve_model_identity(_settings(models=catalogues)).deployment == "alias"
    assert resolve_model_identity(_settings(api={"deployment": ""}, models=catalogues)).deployment == ""


def test_identity_resolution_does_not_read_ambient_environment_or_files(monkeypatch):
    settings = _settings(models={"openai_example": {"provider": "openai", "model": "chosen"}})
    monkeypatch.setenv("TESTUARY_API_MODEL", "changed")
    monkeypatch.setenv("TESTUARY_OPENAI_API_URL", "https://changed.example/v1")
    monkeypatch.setattr(settings_module, "_read_ini_file", lambda _: pytest.fail("model identity reread a file"))
    identity = resolve_model_identity(settings)
    assert identity.model == "chosen"
    assert identity.api_url == "https://api.openai.com/v1"


def test_identity_exports_never_include_credentials_or_captured_environment():
    identity = resolve_model_identity(_settings(models={"openai_example": {"provider": "openai", 
        "api_url": "https://user:private-password@[::1]:8443/v1?key=private-query#fragment",
        "api_key": "private-key",
    }}, environment={"TESTUARY_OPENAI_API_KEY": "private-env"}))
    assert identity.export()["api_url"] == "https://[::1]:8443/v1"
    for text in (repr(identity), json.dumps(identity.export())):
        for credential in ("user", "private-password", "private-query", "private-key", "private-env", "fragment"):
            assert credential not in text
    with pytest.raises(FrozenInstanceError):
        identity.provider = "ollama"


def test_missing_model_is_refused_before_constructing_any_request():
    with pytest.raises(ConfigurationError, match="model must name an alias"):
        resolve_model_identity(RunSettings())


def test_identity_export_preserves_an_explicit_zero_port():
    identity = resolve_model_identity(_settings(models={"openai_example": {"provider": "openai", 
        "api_url": "https://user:private-password@example.test:0/v1?key=private-query",
    }}))
    assert identity.export()["api_url"] == "https://example.test:0/v1"
    assert "private" not in repr(identity)


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
@pytest.mark.parametrize("options", [
    {"extra_headers": {"X-Api-Key": "private-header"}},
    {"custom": {"X-Api-Key": "private-header", "allowed": 0}},
    {"custom": [{"X-Api-Key": "private-header", "allowed": False}]},
])
def test_phase_model_exports_redact_options_without_changing_request_inputs(phase, options):
    model = resolve_phase_model(_settings(models={"openai_example": {"provider": "openai", 
        "request_options": options,
    }}), phase)
    for text in (repr(model), json.dumps(model.export())):
        assert "private-header" not in text
    assert model.export()["request_options_known"] is False
    assert "private-header" in str(model.request_options)
    assert model.export()["request_options"] != options


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_phase_model_export_preserves_known_non_secret_options(phase):
    options = {"top_k": 0, "custom": [False, None, {"threshold": 0}]}
    model = resolve_phase_model(_settings(models={"openai_example": {"provider": "openai", 
        "request_options": options,
    }}), phase)
    exported = model.export()
    assert exported["request_options_known"] is True
    assert exported["request_options"] == options
    exported["request_options"]["custom"].append("changed")
    assert model.export()["request_options"] == options
