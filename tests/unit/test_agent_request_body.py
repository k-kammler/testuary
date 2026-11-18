"""What the agent phases send to the model.

A declared setting is not necessarily sent. The tests build the request
body through the model an agent builds in a run
(``models.build_resolved_chat_model``) and
``_ChatOpenAIWithReasoning._get_request_payload`` and pin the
provider-facing keys, so forwarding a new setting has to change a test.
The payload is built, never sent.
"""
from __future__ import annotations

import pytest
from langchain_core.messages import HumanMessage, SystemMessage

import test_coverage_agent as coverage_harness
from testuary import config
from testuary import models
from testuary.agent import coverage_agent as tc
from testuary.settings import ConfigurationError

# The keys an agent's request body carries. Each absent setting is also
# asserted by name below.
AGENT_PAYLOAD_KEYS = ["extra_body", "messages", "model", "stream", "tools"]

# Settings that exist in the configuration or the model template and do not
# reach an agent's request. ``max_completion_tokens`` is how langchain-openai
# spells a token cap for this endpoint.
NOT_SENT = [
    "temperature", "seed", "reasoning_effort", "thinking", "think",
    "thinking_token_budget", "max_tokens", "max_completion_tokens",
    "reasoning", "top_p", "stop",
]


@pytest.fixture(autouse=True)
def explicit_model_selection(in_run):
    in_run(overrides={"api": {"model": "fixture", "provider": "openai"}})


def _payload(phase="coverage", api_key="EMPTY"):
    """Serialize a request through the model ``phase``'s agent builds in
    the current run."""
    model = models.build_resolved_chat_model(
        models.resolve_phase_model(config.run_settings(), phase), api_key=api_key)
    return model, model._get_request_payload(  # under test
        [SystemMessage("system"), HumanMessage("user")], tools=[],
    )


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
def test_an_agent_request_carries_only_the_model_stream_messages_tools_and_extra_body(phase):
    """Both agent phases send the same five keys from their own settings,
    with ``parallel_tool_calls`` as the only provider extra."""
    _model, payload = _payload(phase)

    assert sorted(payload) == AGENT_PAYLOAD_KEYS
    assert payload["extra_body"] == {"parallel_tool_calls": False}
    assert payload["stream"] is False


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
@pytest.mark.parametrize("name", NOT_SENT)
def test_the_declared_settings_an_agent_never_sends(phase, name):
    """Each absent setting is named, so forwarding one fails a test that
    names it."""
    _model, payload = _payload(phase)

    assert name not in payload
    assert name not in payload.get("extra_body", {})


def test_reasoning_effort_without_a_dialect_stops_the_agent_before_a_request(
    monkeypatch, tmp_path, in_run,
):
    """An alias's ``reasoning_effort`` is not in the API settings, and the
    agent's model refuses it without a ``reasoning_dialect``, so it is never
    dropped silently."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TESTUARY_HOME", str(tmp_path))
    (tmp_path / "models.ini").write_text(
        "[myalias]\n"
        "provider = openai\n"
        "model = a-thinking-model\n"
        "api_url = http://127.0.0.1:1/v1\n"
        "reasoning_effort = high\n",
        encoding="utf-8",
    )
    (tmp_path / "config.ini").write_text(
        "[api]\nmodel = myalias\n", encoding="utf-8")
    # The run of this home directory's config.ini and models.ini.
    in_run(models=None)

    api_config = models.get_api_config()

    assert config.run_settings().models["myalias"]["reasoning_effort"] == "high"
    assert "reasoning_effort" not in api_config
    assert api_config["model"] == "a-thinking-model"
    with pytest.raises(ConfigurationError, match="need a reasoning_dialect"):
        _payload()


def test_an_explicit_zero_cap_resolves_to_no_cap():
    """Zero, empty and absent caps add no cap, and a cap that is no
    nonnegative integer is refused. The starter phase declares 0, and the
    agent sections omit the key."""
    resolve = models.resolve_output_cap
    assert resolve("0") is None and resolve("") is None and resolve(None) is None
    assert resolve() is None
    assert resolve("1024") == 1024
    for invalid in ("none", "-1"):
        with pytest.raises(ConfigurationError, match="max_output_tokens"):
            resolve(invalid)


def test_an_empty_credential_is_sent_as_the_literal_EMPTY():
    """An empty credential becomes the literal ``EMPTY``, so an endpoint that
    checks credentials fails at the provider, not locally."""
    model, _payload_ = _payload(api_key="")

    assert model.openai_api_key.get_secret_value() == "EMPTY"


def test_an_agent_builds_from_current_api_and_agent_configuration(
    tmp_path, monkeypatch, in_run,
):
    """The builder reads the current run's settings: in a run that selects
    another model and endpoint, the agent builds that model there."""
    agent = coverage_harness._make_agent(tmp_path)
    before = agent.build_chat_model()
    assert agent.agent_config() is tc._AGENT_CONFIG

    in_run(overrides={"api": {"model": "changed-model", "provider": "openai",
                              "api_url": "http://127.0.0.1:2/v1"}})
    after = agent.build_chat_model()

    assert before.model_name != "changed-model"
    assert after.model_name == "changed-model"
    assert str(after.openai_api_base) == "http://127.0.0.1:2/v1"
    assert agent.agent_config() is tc._AGENT_CONFIG
    # The timeout is the phase model's, resolved from the run's settings.
    expected_timeout = models.resolve_phase_model(
        config.run_settings(), "coverage").values["request_timeout_seconds"]
    for model in (before, after):
        assert model.http_client.timeout.read == expected_timeout
        assert model.reasoning_field_names == tuple(
            tc._AGENT_CONFIG["reasoning_field_names"])
        assert model.extra_body == {"parallel_tool_calls": False}
