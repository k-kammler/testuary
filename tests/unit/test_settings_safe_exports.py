"""Public settings exports redact refused credentials before model preflight."""
import json

import pytest

from testuary.settings import RunSettings


@pytest.mark.parametrize("name", ["authorization", "Authorization", "AUTHORIZATION",
                                 "proxy_authorization", "custom_Authorization", "authorization_token"])
@pytest.mark.parametrize("location", ["sections", "model-text", "model-mapping"])
def test_safe_export_removes_authorization_at_every_nested_level(name, location):
    secret = "credential-sentinel-not-for-export"
    options = {name: secret, "allowed": [{"nested": {name: secret, "value": 0}}],
               "reasoning_budget_tokens": 17, "max_output_tokens": 40}
    text = json.dumps(options)
    if location == "sections":
        settings = RunSettings(sections={"starter": {"request_options": text}})
        raw = settings.sections["starter"]["request_options"]
        exported = json.loads(settings.export()["sections"]["starter"]["request_options"])
    else:
        field = "models"
        settings = RunSettings(**{field: {"example": {"request_options": text if location.endswith("text") else options}}})
        raw = getattr(settings, field)["example"]["request_options"]
        exported = settings.export()[field]["example"]["request_options"]
        if isinstance(exported, str):
            exported = json.loads(exported)
    assert secret not in json.dumps(settings.export()) and secret not in repr(settings)
    assert name not in exported and name not in exported["allowed"][0]["nested"]
    assert exported["allowed"][0]["nested"]["value"] == 0
    assert exported["reasoning_budget_tokens"] == 17 and exported["max_output_tokens"] == 40
    # Safe export is a detached view, not an execution-input rewrite.
    if isinstance(raw, str):
        assert raw == text
    else:
        assert raw[name] == secret and raw["allowed"][0]["nested"][name] == secret
    assert options[name] == secret


@pytest.mark.parametrize("invalid", ['{"authorization": "private"', '[{"authorization": "private"}]'])
def test_invalid_options_do_not_escape_via_the_safe_export(invalid):
    settings = RunSettings(sections={"mutation": {"request_options": invalid}})
    assert settings.export()["sections"]["mutation"]["request_options"] == "[invalid request options]"
    assert "private" not in repr(settings)
    assert settings.sections["mutation"]["request_options"] == invalid
