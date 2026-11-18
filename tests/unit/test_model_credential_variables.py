"""The run's model-credential variable names come from the same identification
as diagnostic redaction, so launch-environment minimization cannot drift from
what is treated as a credential. Values here are synthetic."""
from __future__ import annotations

import pytest

from testuary.settings import (
    ConfigurationError, RunSettings, _redact_diagnostic, model_credential_variables,
)


def _settings(**environment):
    return RunSettings(
        sections={"api": {"model": "coverage_model"},
                  "ollama": {"api_url": "https://ollama.com/v1"}},
        models={
            "coverage_model": {"provider": "ollama", "api_key_env": "COVERAGE_MODEL_KEY"},
            "mutation_model": {"provider": "openai", "api_key_env": " MUTATION_MODEL_KEY "},
            "plain_model": {"provider": "vllm", "api_key": "synthetic-catalogue-key"},
        },
        environment={
            "TESTUARY_OLLAMA_API_KEY": "synthetic-ollama-key",
            "TESTUARY_OPENAI_API_KEY": "",
            "COVERAGE_MODEL_KEY": "synthetic-coverage-key",
            "SUBJECT_SERVICE_TOKEN": "subject-value",
            "PATH": "/usr/bin",
            **environment,
        },
    )


def test_credential_variables_are_the_declared_and_supplied_model_keys():
    names = model_credential_variables(_settings())
    assert names == {"TESTUARY_OLLAMA_API_KEY", "COVERAGE_MODEL_KEY", "MUTATION_MODEL_KEY"}
    # Ordinary subject variables and empty provider defaults are not credentials.
    assert "SUBJECT_SERVICE_TOKEN" not in names and "PATH" not in names
    assert "TESTUARY_OPENAI_API_KEY" not in names


def test_redaction_and_variable_names_share_one_identification():
    settings = _settings(MUTATION_MODEL_KEY="synthetic-mutation-key")
    text = ("synthetic-ollama-key synthetic-coverage-key synthetic-mutation-key "
            "synthetic-catalogue-key subject-value")
    redacted = _redact_diagnostic(text, settings)
    assert redacted == "[redacted] [redacted] [redacted] [redacted] subject-value"
    for name in model_credential_variables(settings):
        assert settings.environment[name] not in redacted


def test_credential_variables_require_resolved_settings():
    with pytest.raises(ConfigurationError):
        model_credential_variables({"environment": {}})
