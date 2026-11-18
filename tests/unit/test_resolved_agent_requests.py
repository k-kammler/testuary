"""Captured model settings reach actual editing-agent SDK requests."""
from langchain_core.messages import AIMessage, HumanMessage
from types import SimpleNamespace

import httpx
import pytest

import test_coverage_agent as coverage_harness
import test_mutation_agent as mutation_harness
import test_model_protocols as protocol_harness
from testuary.agent import _langgraph as lg
from testuary import models as _model_source
from testuary import config
from testuary.settings import RunSettings

transport = protocol_harness.transport


@pytest.mark.parametrize("harness, cap, timeout", [
    (coverage_harness, 40, 12), (mutation_harness, 20, 13),
], ids=["coverage", "mutation"])
@pytest.mark.parametrize("protocol", ["openai-chat", "openai-responses"])
def test_actual_agent_request_consumes_captured_catalogue_settings(harness, cap, timeout, protocol, tmp_path, transport):
    settings = RunSettings(sections={
        "api": {"model": "explicit"},
        "coverage": {"max_output_tokens": 40, "request_timeout_seconds": 12},
        "mutation": {"max_output_tokens": 20, "request_timeout_seconds": 13},
    }, models={"explicit": {
        "provider": "openai", "model": "chosen", "api_url": "https://example.test/v1", "api_key": "fixture",
        "protocol": protocol, "temperature": 0, "top_p": 0, "max_output_tokens": 100,
        "request_options": {"custom_option": {"items": [1, 2]}},
    }})
    with config.use_run_settings(settings):
        agent = harness._make_agent(tmp_path)
        model = agent.build_chat_model()
        model.invoke([HumanMessage("task")])
        assert agent.provider_transport() == ("openai_responses" if protocol == "openai-responses" else "openai_compatible")
    url, body = transport[0]
    assert url.path == ("/v1/responses" if protocol == "openai-responses" else "/v1/chat/completions")
    assert model.http_client.timeout.read == timeout
    assert model.max_retries == 0
    assert body["model"] == "chosen"
    assert body["temperature"] == body["top_p"] == 0
    assert body["custom_option"] == {"items": [1, 2]}
    assert body["parallel_tool_calls"] is False
    assert body["max_output_tokens" if protocol == "openai-responses" else "max_completion_tokens"] == cap
    assert ("max_tokens" in body) == (protocol == "openai-chat")


@pytest.mark.parametrize("harness", [coverage_harness, mutation_harness], ids=["coverage", "mutation"])
def test_actual_graph_uses_the_same_captured_reasoning_fields_as_the_client(harness, monkeypatch, tmp_path, transport):
    settings = RunSettings(sections={"api": {"model": "explicit"}}, models={"explicit": {
        "provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
        "reasoning_field_names": "custom_reasoning", "preserve_reasoning_content": True,
    }})
    observed = {}
    monkeypatch.setattr(lg, "build_react_graph", lambda **kwargs: observed.update(kwargs) or "graph")
    with config.use_run_settings(settings):
        agent = harness._make_agent(tmp_path)
        model = agent.build_chat_model()
        assert agent.build_graph(model=model) == "graph"
        assert model.reasoning_field_names == ("custom_reasoning",)
        assert observed["reasoning_field_names"] == model.reasoning_field_names
        if harness is coverage_harness:
            assert agent.agent_start_fields()["reasoning_field_names"] == ["custom_reasoning"]
        old = AIMessage("old", additional_kwargs={"custom_reasoning": "remove"})
        latest = AIMessage("latest", additional_kwargs={"custom_reasoning": "keep"})
        messages = observed["pre_model_hook"]({"messages": [old, latest]})["messages"]
        assistants = [message for message in messages if isinstance(message, AIMessage)]
        assert "custom_reasoning" not in assistants[0].additional_kwargs
        assert assistants[1].additional_kwargs["custom_reasoning"] == "keep"
    assert transport == []


@pytest.mark.parametrize("harness", [coverage_harness, mutation_harness], ids=["coverage", "mutation"])
@pytest.mark.parametrize("ambient_output", [None, "responses/v1"])
def test_declared_chat_protocol_cannot_be_changed_by_sdk_auto_routing(harness, ambient_output, monkeypatch, tmp_path, transport):
    if ambient_output is not None:
        monkeypatch.setenv("LC_OUTPUT_VERSION", ambient_output)
    settings = RunSettings(sections={"api": {"model": "explicit"}}, models={"explicit": {
        "provider": "openai", "model": "example-codex-compatible", "api_url": "https://example.test/v1",
        "protocol": "openai-chat",
    }})
    with config.use_run_settings(settings):
        harness._make_agent(tmp_path).build_chat_model().invoke([HumanMessage("task")])
    assert transport[0][0].path == "/v1/chat/completions"


@pytest.mark.parametrize("harness", [coverage_harness, mutation_harness], ids=["coverage", "mutation"])
def test_sequential_invocations_do_not_keep_the_previous_model_options(harness, tmp_path, transport):
    agent = harness._make_agent(tmp_path)
    for temperature in (0.2, 0.8):
        settings = RunSettings(sections={"api": {"model": "explicit"}}, models={"explicit": {
            "provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
            "temperature": temperature,
        }})
        with config.use_run_settings(settings):
            agent.build_chat_model().invoke([HumanMessage("task")])
    assert [body["temperature"] for _, body in transport] == [0.2, 0.8]


@pytest.mark.parametrize("harness", [coverage_harness, mutation_harness], ids=["coverage", "mutation"])
def test_installed_model_source_does_not_require_provider_configuration(harness, tmp_path, transport):
    model, called = object(), []
    source = SimpleNamespace(agent_chat_model=lambda phase: called.append(phase) or model)
    with config.use_run_settings(RunSettings()), _model_source.installed(source):
        agent = harness._make_agent(tmp_path)
        assert agent.build_chat_model() is model
        assert agent.provider_transport() == "model_source"
        assert agent.reasoning_fields() == tuple(agent.agent_config()["reasoning_field_names"])
    assert called == [agent.phase_label()]
    assert transport == []


@pytest.fixture
def added_clients(monkeypatch, provider_module):
    from langchain_core.language_models import BaseChatModel
    from langchain_core.outputs import ChatGeneration, ChatResult
    from testuary import model_requests as _model_requests

    state = SimpleNamespace(models=[], action=None)

    class ConstructorModel(BaseChatModel):
        parameters: dict
        bindings: list = []
        calls: int = 0

        @property
        def _llm_type(self):
            return "agent-constructor-fixture"

        def bind_tools(self, tools, **kwargs):
            names = tuple(tool.name for tool in tools)
            self.bindings.append(names)
            return self.bind(bound_names=names)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            assert set(kwargs) <= {"bound_names"}
            self.calls += 1
            if state.action is not None:
                state.action(self)
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="fixture reply",
                usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
            ))])

    module = provider_module("_fixture_agent_constructor")

    def factory(**parameters):
        model = ConstructorModel(parameters=parameters)
        state.models.append(model)
        return model

    module.factory = factory
    state.factory = module.__name__ + ":factory"
    monkeypatch.setattr(_model_requests.RequestAccount, "wait", lambda *a: None)
    return state


def added_settings(added_clients):
    return RunSettings(sections={"api": {"model": "explicit"}}, models={"explicit": {
        "provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
        "protocol": "added", "chat_model": added_clients.factory, "temperature": 0,
    }})


@pytest.mark.parametrize("harness", [coverage_harness, mutation_harness], ids=["coverage", "mutation"])
def test_rebuilt_graphs_keep_one_phase_client_and_the_existing_tool_gates(harness, tmp_path, added_clients):
    with config.use_run_settings(added_settings(added_clients)):
        agent = harness._make_agent(tmp_path)
        base = agent.build_chat_model()
        for _ in range(2):
            graph = agent.build_graph()
            with pytest.raises(lg.OutputValidationError):
                graph.invoke({"messages": [HumanMessage("task")]})
        assert agent.build_chat_model() is base
    assert added_clients.models == [base]
    assert base.calls == 2
    reads, writes, run_tests, measure = agent.phase_tools()
    final = {agent.final_tool_name}
    expected = [
        {tool.name for tool in reads + writes} | final,
        {tool.name for tool in reads + writes + [run_tests]} | final,
        {tool.name for tool in reads + [measure]} | final,
        # Assess after a run with a STANDS ALONE note adds the edit tool.
        {tool.name for tool in reads + writes + [measure]} | final,
        {agent.designated_read_name} | final,
        # The start's second step binds only the measurement.
        {measure.name} | final,
    ]
    assert [set(names) for names in base.bindings] == expected * 2


@pytest.mark.parametrize("harness", [coverage_harness, mutation_harness], ids=["coverage", "mutation"])
def test_an_agent_request_sent_again_and_graph_rebuild_keep_the_base_client(harness, tmp_path, added_clients):
    def action(model):
        if model.calls == 1:
            raise httpx.ConnectError("fixture interruption")

    added_clients.action = action
    with config.use_run_settings(added_settings(added_clients)):
        agent = harness._make_agent(tmp_path)
        graph = agent.build_graph()
        with pytest.raises(lg.OutputValidationError):
            graph.invoke({"messages": [HumanMessage("task")]})
        assert agent.build_chat_model() is added_clients.models[0]
        agent.build_graph()
    assert len(added_clients.models) == 1
    assert added_clients.models[0].calls == 2


@pytest.mark.parametrize("harness", [coverage_harness, mutation_harness], ids=["coverage", "mutation"])
def test_identical_settings_in_new_invocations_do_not_borrow_agent_clients(harness, tmp_path, added_clients):
    settings = added_settings(added_clients)
    agent = harness._make_agent(tmp_path)
    clients = []
    scopes = []
    for _ in range(2):
        with config.use_run_settings(settings):
            scopes.append(config.active_run_scope())
            clients.append(agent.build_chat_model())
            assert agent.build_chat_model() is clients[-1]
            assert config.active_run_scope() is scopes[-1]
        assert config.active_run_scope() is None
    assert len(added_clients.models) == 2
    assert clients[0] is not clients[1]
    assert scopes[0] is not scopes[1]


@pytest.mark.parametrize("harness", [coverage_harness, mutation_harness], ids=["coverage", "mutation"])
def test_cancelled_invocation_releases_agent_client_scope(harness, tmp_path, added_clients):
    settings = added_settings(added_clients)
    agent = harness._make_agent(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        with config.use_run_settings(settings):
            cancelled = agent.build_chat_model()
            raise KeyboardInterrupt("fixture cancellation")
    assert config.active_settings() is None
    assert config.active_run_scope() is None
    with config.use_run_settings(settings):
        assert agent.build_chat_model() is not cancelled
    assert len(added_clients.models) == 2


@pytest.mark.parametrize("harness", [coverage_harness, mutation_harness], ids=["coverage", "mutation"])
def test_distinct_targets_with_equal_settings_have_distinct_agent_clients(harness, tmp_path, added_clients):
    with config.use_run_settings(added_settings(added_clients)):
        first = harness._make_agent(tmp_path, "first").build_chat_model()
        second = harness._make_agent(tmp_path, "second").build_chat_model()
        assert first is not second
    assert len(added_clients.models) == 2
