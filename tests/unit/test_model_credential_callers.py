"""Captured credential precedence and omission at the actual SDK boundaries."""
import json

import httpx
import pytest

import test_model_precedence_callers as caller_harness
import test_model_protocols as protocol_harness
import test_resolved_starter_requests as starter_harness

streams = starter_harness.streams
transport = protocol_harness.transport

SOURCES = [
    pytest.param({}, {}, "captured-default", id="provider-environment"),
    pytest.param({"api_key_env": "PROVIDER_KEY"}, {}, "captured-provider", id="provider-named-environment"),
    pytest.param({"api_key_env": "PROVIDER_KEY"}, {"api_key_env": "ALIAS_KEY"},
                 "captured-alias", id="alias-named-environment"),
    pytest.param({"api_key_env": "PROVIDER_KEY", "api_key": "catalogue-provider"}, {},
                 "catalogue-provider", id="provider-catalogue-over-environment"),
    pytest.param({"api_key_env": "PROVIDER_KEY", "api_key": "catalogue-provider"},
                 {"api_key_env": "ALIAS_KEY", "api_key": "catalogue-alias"},
                 "catalogue-alias", id="alias-catalogue-over-provider"),
    pytest.param({"api_key": "catalogue-provider"}, {"api_key": ""}, "EMPTY", id="empty-catalogue-clears"),
    pytest.param({"api_key_env": "PROVIDER_KEY"}, {"api_key_env": ""},
                 "captured-default", id="empty-named-selector-clears"),
]


def test_a_named_variable_must_be_set_even_when_the_file_has_a_key():
    """As the models.ini template says: an unset api_key_env variable is
    refused although the file sets api_key, and a set one yields to it."""
    from testuary.models import resolve_provider_credential
    from testuary.settings import ConfigurationError

    alias = {"api_key_env": "ALIAS_KEY", "api_key": "catalogue-alias"}
    with pytest.raises(ConfigurationError, match="api_key_env names .*, which is not set"):
        resolve_provider_credential("openai", {}, alias, {})
    assert resolve_provider_credential(
        "openai", {}, alias, {"ALIAS_KEY": "captured-alias"}) == "catalogue-alias"


@pytest.mark.parametrize("phase", ("starter", "coverage", "mutation"))
@pytest.mark.parametrize("provider_fields,alias_fields,expected", SOURCES)
def test_credentials_are_captured_before_the_request_and_never_recorded(
    phase, provider_fields, alias_fields, expected, tmp_path, monkeypatch, request, capsys,
):
    headers = []
    real_transport = httpx.MockTransport

    def watched_transport(handle):
        def watched(message):
            headers.append(message.headers.get("Authorization"))
            return handle(message)
        return real_transport(watched)

    monkeypatch.setattr(httpx, "MockTransport", watched_transport)
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    environment = {"TESTUARY_OPENAI_API_KEY": "captured-default",
                   "PROVIDER_KEY": "captured-provider", "ALIAS_KEY": "captured-alias"}
    configured = caller_harness.settings_for(tmp_path, (), environment, models={
        "openai": {"api_url": "https://example.test/v1", **provider_fields},
        "explicit": {"provider": "openai", "model": "chosen", **alias_fields},
    })
    for key in (*environment, "OPENAI_API_KEY"):
        monkeypatch.setenv(key, "changed-ambient-credential")
    record = caller_harness.invoke_phase(phase, tmp_path, configured, monkeypatch)
    assert len(sent) == 1 and headers == [f"Bearer {expected}"]
    exported = json.dumps(configured.export()) + repr(configured) + json.dumps(record)
    output = capsys.readouterr()
    secrets = {*environment.values(), "catalogue-provider", "catalogue-alias", "changed-ambient-credential"}
    for secret in secrets:
        assert secret not in exported + output.out + output.err
        assert all(secret.encode() not in path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())
