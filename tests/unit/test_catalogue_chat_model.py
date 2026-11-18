"""catalogue_chat_model: a chat model for a catalogue alias, handed to
another program, through an added alias whose factory returns a scripted
model, so no request leaves the machine."""
import json
import re
from types import SimpleNamespace

import httpx
import pytest
from langchain_core.language_models import BaseChatModel
from langchain_core.load import dumps
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_tool

from testuary import model_requests
from testuary.models import catalogue_chat_model
from testuary.settings import ConfigurationError

SENTINEL = "sk-test-sentinel"


class Scripted(BaseChatModel):
    """Replies with the next scripted outcome: a message, or an error raised
    as a provider's failure. Keeps the key it was built with, as a client
    does."""

    state: object
    api_key: str = ""

    @property
    def _llm_type(self):
        return "scripted"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.state.seen.append(kwargs)
        outcome = self.state.outcomes.pop(0)
        if isinstance(outcome, BaseException):
            raise outcome
        return ChatResult(generations=[ChatGeneration(message=outcome)])

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[convert_to_openai_tool(item) for item in tools], **kwargs)


def reply(text="done", usage=True):
    meta = {"input_tokens": 3, "output_tokens": 2, "total_tokens": 5} if usage else None
    return AIMessage(content=text, usage_metadata=meta)


@pytest.fixture
def catalogue(tmp_path, monkeypatch, provider_module):
    """A user directory whose models.ini names the added alias ``scripted``, a
    project directory, the credential in the environment, and no waits."""
    state = SimpleNamespace(outcomes=[], seen=[], parameters=[])
    module = provider_module("_fixture_scripted_provider")

    def factory(**parameters):
        state.parameters.append(parameters)
        return Scripted(state=state, api_key=parameters["api_key"])

    module.factory = factory
    home = tmp_path / "home"
    home.mkdir()

    def write(models="", config=""):
        (home / "models.ini").write_text(
            "[scripted]\nprovider = openai\nmodel = scripted-model\n"
            "api_url = http://127.0.0.1:9/v1\nchat_model = _fixture_scripted_provider:factory\n"
            "max_output_tokens = 64\n" + models, encoding="utf-8")
        (home / "config.ini").write_text(config, encoding="utf-8")

    write()
    monkeypatch.setenv("TESTUARY_HOME", str(home))
    monkeypatch.setenv("TESTUARY_OPENAI_API_KEY", SENTINEL)
    monkeypatch.setattr(model_requests.RequestAccount, "wait", lambda self, seconds: None)
    project = tmp_path / "project"
    project.mkdir()
    records = tmp_path / "records"
    state.write = write

    def model(label="profile", **values):
        return catalogue_chat_model("scripted", base_dir=project, records_dir=records,
                                    label=label, request_values=values or None)

    state.model = model
    state.records = records
    return state


def lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_an_alias_resolves_as_a_run_does_and_records_each_request(catalogue):
    model = catalogue.model()
    catalogue.outcomes += [reply("first"), reply("second", usage=False)]
    assert model.invoke([HumanMessage("a")]).content == "first"
    assert model.invoke([HumanMessage("b")]).content == "second"
    built = catalogue.parameters[0]
    assert (built["model"], built["base_url"], built["max_tokens"]) == (
        "scripted-model", "http://127.0.0.1:9/v1", 64)
    records = lines(catalogue.records / "request_records.jsonl")
    assert [(r["logical_request_idx"], r["outcome"], r["usage_known"], r["label"])
            for r in records] == [(1, "delivered", True, "profile"), (2, "delivered", False, "profile")]
    assert (records[0]["input_tokens"], records[0]["output_tokens"]) == (3, 2)
    assert "input_tokens" not in records[1]


def test_the_callers_values_are_the_top_layer_and_no_phase_section_applies(catalogue):
    catalogue.write(config="[coverage]\ntemperature = 0.9\n[api]\ntop_p = 0.5\n")
    catalogue.model(temperature=0.3)
    built = catalogue.parameters[0]
    assert built["temperature"] == 0.3 and built["top_p"] == 0.5
    catalogue.model(label="improve")
    assert "temperature" not in catalogue.parameters[1]


@pytest.mark.parametrize("left_out", [{"config_files": []}, {"project_settings": False}],
                         ids=["config_files", "project_settings"])
def test_config_files_and_project_settings_leave_out_their_settings_file(
        catalogue, tmp_path, left_out):
    """An empty config_files reads no configuration file, and project_settings
    false no project.ini, as for resolve_settings."""
    project = tmp_path / "project"
    if "config_files" in left_out:
        catalogue.write(config="[api]\ntop_p = 0.5\n")
    else:
        (project / ".testuary").mkdir()
        (project / ".testuary" / "project.ini").write_text("[api]\ntop_p = 0.5\n", encoding="utf-8")
    catalogue.model()
    catalogue_chat_model("scripted", base_dir=project, records_dir=catalogue.records,
                         label="improve", **left_out)
    assert catalogue.parameters[0]["top_p"] == 0.5
    assert "top_p" not in catalogue.parameters[1]


def test_named_configuration_files_replace_the_global_one(catalogue, tmp_path):
    """A list of configuration files, a relative one from base_dir, is read
    in place of the global config.ini."""
    catalogue.write(config="[api]\ntop_p = 0.5\ntemperature = 0.9\n")
    (tmp_path / "project" / "named.ini").write_text("[api]\ntop_p = 0.25\n", encoding="utf-8")
    catalogue_chat_model("scripted", base_dir=tmp_path / "project", records_dir=catalogue.records,
                         label="profile", config_files=["named.ini"])
    built = catalogue.parameters[0]
    assert built["top_p"] == 0.25 and "temperature" not in built


def test_one_list_of_catalogue_files_serves_a_run_and_a_handed_out_model(catalogue, tmp_path):
    """A program passes the same models_files to resolve_settings for its
    run and to catalogue_chat_model: both resolve the alias from those
    files, not from the user directory's models.ini."""
    from testuary.settings import resolve_settings

    project = tmp_path / "project"
    (project / "own-models.ini").write_text(
        "[scripted]\nprovider = openai\nmodel = own-model\n"
        "chat_model = _fixture_scripted_provider:factory\n", encoding="utf-8")
    run = resolve_settings(base_dir=project, models_files=["own-models.ini"])
    assert run.models["scripted"]["model"] == "own-model"
    catalogue_chat_model("scripted", base_dir=project, records_dir=catalogue.records,
                         label="profile", models_files=["own-models.ini"])
    assert catalogue.parameters[0]["model"] == "own-model"


def test_an_absent_earlier_catalogue_file_and_an_alias_the_files_lack_are_refused_naming_them(
        catalogue, tmp_path):
    project = tmp_path / "project"
    (project / "own-models.ini").write_text("[other]\nprovider = openai\nmodel = m\n",
                                            encoding="utf-8")
    with pytest.raises(ConfigurationError, match="missing-models.ini"):
        catalogue_chat_model("scripted", base_dir=project, records_dir=catalogue.records,
                             label="profile", models_files=["missing-models.ini", "own-models.ini"])
    with pytest.raises(ConfigurationError, match=r"\[scripted\] is not defined in .*own-models\.ini"):
        catalogue_chat_model("scripted", base_dir=project, records_dir=catalogue.records,
                             label="profile", models_files=["own-models.ini"])
    assert not catalogue.parameters and not catalogue.records.exists()


def test_the_user_dirs_catalogue_in_a_list_keeps_its_key(catalogue, tmp_path):
    """The user directory's models.ini named in models_files keeps a literal
    api_key, the credential above the environment's, and no record holds
    it."""
    catalogue.write(models="[openai]\napi_key = sk-home-sentinel\n")
    model = catalogue_chat_model("scripted", base_dir=tmp_path / "project",
                                 records_dir=catalogue.records, label="profile",
                                 models_files=[str(tmp_path / "home" / "models.ini")])
    catalogue.outcomes += [reply("done")]
    model.invoke([HumanMessage("a")])
    assert catalogue.parameters[0]["api_key"] == "sk-home-sentinel"
    texts = [path.read_text(encoding="utf-8") for path in catalogue.records.iterdir()]
    assert texts and not [text for text in texts if "sk-home-sentinel" in text]


def test_a_literal_key_in_a_catalogue_file_in_a_project_dir_is_refused(catalogue, tmp_path):
    """A named catalogue file outside the user directory that holds a literal
    api_key is refused before any request, by a run's settings and by
    catalogue_chat_model alike, naming the file and api_key_env and not the
    key."""
    from testuary.settings import resolve_settings

    project = tmp_path / "project"
    (project / "own-models.ini").write_text(
        "[openai]\napi_key = sk-project-sentinel\n[scripted]\nprovider = openai\n"
        "model = own-model\napi_url = http://127.0.0.1:9/v1\n"
        "chat_model = _fixture_scripted_provider:factory\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match="own-models.ini") as refused:
        resolve_settings(base_dir=project, models_files=["own-models.ini"])
    assert "api_key_env" in str(refused.value) and "sk-project-sentinel" not in str(refused.value)
    with pytest.raises(ConfigurationError, match="api_key_env"):
        catalogue_chat_model("scripted", base_dir=project, records_dir=catalogue.records,
                             label="profile", models_files=["own-models.ini"])
    assert not catalogue.parameters and not catalogue.records.exists()


def test_a_key_variable_in_a_catalogue_file_in_a_project_dir_is_refused(catalogue, tmp_path,
                                                                          monkeypatch):
    """A program's model whose catalogue file in the project directory names a
    set key variable is refused before any request, naming the file."""
    monkeypatch.setenv("TESTUARY_FIXTURE_PROVIDER_KEY", "sk-provider-sentinel")
    project = tmp_path / "project"
    (project / "own-models.ini").write_text(
        "[openai]\napi_key_env = TESTUARY_FIXTURE_PROVIDER_KEY\n[scripted]\nprovider = openai\n"
        "model = own-model\nchat_model = _fixture_scripted_provider:factory\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"own-models\.ini: \[openai\] api_key_env decides"):
        catalogue_chat_model("scripted", base_dir=project, records_dir=catalogue.records,
                             label="profile", models_files=["own-models.ini"])
    assert not catalogue.parameters and not catalogue.records.exists()


def test_an_alias_a_later_file_redefines_keeps_its_providers_key(catalogue, tmp_path, monkeypatch):
    """Only the alias's section is replaced: the provider section of the
    first file still names the credential of the alias the later file
    defines anew."""
    monkeypatch.setenv("TESTUARY_FIXTURE_PROVIDER_KEY", "sk-provider-sentinel")
    project = tmp_path / "project"
    lists = tmp_path / "lists"
    lists.mkdir()
    (lists / "first.ini").write_text(
        "[openai]\napi_key_env = TESTUARY_FIXTURE_PROVIDER_KEY\n[scripted]\nprovider = openai\n"
        "model = first-model\n"
        "chat_model = _fixture_scripted_provider:factory\n", encoding="utf-8")
    (lists / "later.ini").write_text(
        "[scripted]\nprovider = openai\nmodel = later-model\n"
        "chat_model = _fixture_scripted_provider:factory\n", encoding="utf-8")
    catalogue_chat_model("scripted", base_dir=project, records_dir=catalogue.records,
                         label="profile", models_files=[lists / "first.ini", lists / "later.ini"])
    built = catalogue.parameters[0]
    assert (built["model"], built["api_key"]) == ("later-model", "sk-provider-sentinel")


def test_a_tool_binding_goes_through_the_retries_and_records(catalogue):
    @tool
    def measure(path: str) -> str:
        """Measure the suite at path."""
        return path

    catalogue.outcomes += [httpx.ConnectError("refused"), reply("called")]
    bound = catalogue.model().bind_tools([measure])
    assert bound.invoke([HumanMessage("go")]).content == "called"
    assert catalogue.seen[-1]["tools"][0]["function"]["name"] == "measure"
    assert [r["outcome"] for r in lines(catalogue.records / "request_records.jsonl")] == [
        "transient", "delivered"]


def test_the_bundled_retry_settings_apply_whatever_the_phases_set(catalogue):
    catalogue.write(config="[coverage]\nmax_consecutive_transient_failures = 2\n")
    catalogue.outcomes += [httpx.ConnectError("refused")] * 3 + [reply()]
    assert catalogue.model().invoke([HumanMessage("a")]).content == "done"
    assert len(lines(catalogue.records / "request_records.jsonl")) == 4


def test_a_ceiling_a_terminal_refusal_and_an_overflow_each_raise_their_class(catalogue):
    model = catalogue.model()
    catalogue.outcomes += [httpx.ConnectError("refused")] * 10
    with pytest.raises(model_requests.TransportExhausted):
        model.invoke([HumanMessage("a")])
    catalogue.outcomes += [ValueError("bad request")]
    with pytest.raises(model_requests.TerminalRequestError):
        model.invoke([HumanMessage("b")])
    overflow = ValueError("This model's maximum context length is 8192 tokens")
    catalogue.outcomes += [overflow]
    with pytest.raises(model_requests.ContextOverflow) as caught:
        model.invoke([HumanMessage("c")])
    assert caught.value.__cause__ is None and str(caught.value) == (
        "ValueError: This model's maximum context length is 8192 tokens")
    catalogue.outcomes += [reply("again")]
    assert model.invoke([HumanMessage("d")]).content == "again"
    outcomes = [r["outcome"] for r in lines(catalogue.records / "request_records.jsonl")]
    assert outcomes == ["transient"] * 10 + ["terminal", "terminal", "delivered"]


def test_each_model_appends_its_settings_and_a_resumed_caller_keeps_them(catalogue):
    catalogue.model(label="profile")
    catalogue.model(label="improve")
    catalogue.outcomes += [reply()]
    catalogue.model(label="profile").invoke([HumanMessage("a")])
    settings = lines(catalogue.records / "model_settings.jsonl")
    assert [s["label"] for s in settings] == ["profile", "improve", "profile"]
    assert settings[0]["model_alias"] == "scripted" and "phase" not in settings[0]
    assert settings[0]["values"]["model_max_output_tokens"] == 64


def test_no_credential_leaves_the_handed_out_model(catalogue):
    model = catalogue.model()
    catalogue.outcomes += [reply(), ValueError("bad request")]
    model.invoke([HumanMessage("a")])
    with pytest.raises(model_requests.TerminalRequestError) as caught:
        model.invoke([HumanMessage("b")])
    texts = [repr(model), str(model), model.model_dump_json(), dumps(model), str(caught.value),
             (catalogue.records / "request_records.jsonl").read_text(encoding="utf-8"),
             (catalogue.records / "model_settings.jsonl").read_text(encoding="utf-8")]
    assert catalogue.parameters[0]["api_key"] == SENTINEL
    assert not [text for text in texts if SENTINEL in text]


@pytest.mark.parametrize("values, named", [
    ({"model": "other"}, "'model'"), ({"protocol": "openai-chat"}, "'protocol'"),
    ({"api_key": "x"}, "'api_key'"), ({"api_key_env": "X"}, "'api_key_env'"),
    ({"temprature": 0.1}, "'temprature'"),
])
def test_a_key_the_caller_may_not_set_is_refused_before_any_request(catalogue, values, named):
    with pytest.raises(ConfigurationError, match=named) as caught:
        catalogue.model(**values)
    assert ("Accepted keys" in str(caught.value)) == (named == "'temprature'")
    assert not catalogue.parameters and not catalogue.records.exists()


def test_an_unusable_alias_or_label_is_refused_before_any_request(catalogue, tmp_path):
    with pytest.raises(ConfigurationError, match=r"\[unknown\] is not defined"):
        catalogue_chat_model("unknown", base_dir=tmp_path / "project",
                             records_dir=catalogue.records, label="profile")
    with pytest.raises(ConfigurationError, match="label"):
        catalogue.model(label=" ")
    catalogue.write(models="api_key_env = TESTUARY_FIXTURE_UNSET\n")
    with pytest.raises(ConfigurationError, match="TESTUARY_FIXTURE_UNSET"):
        catalogue.model()
    (tmp_path / "home" / "models.ini").write_text(
        "[native]\nprovider = ollama\nmodel = qwen\napi_url = https://ollama.com\n"
        "protocol = ollama-native\n", encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"\[native\] uses the protocol ollama-native"):
        catalogue_chat_model("native", base_dir=tmp_path / "project",
                             records_dir=catalogue.records, label="profile")
    assert not catalogue.parameters and not catalogue.records.exists()


def test_request_values_are_the_documented_keys_and_an_unknown_key_lists_them(catalogue):
    """REQUEST_VALUES holds exactly the keys catalogue_chat_model's reference
    names for request_values, and the refusal of another key lists them."""
    from testuary import models

    documented = re.search(r"Its keys are (.+?)\.\s", models.catalogue_chat_model.__doc__, re.S)
    assert isinstance(models.REQUEST_VALUES, frozenset)
    assert models.REQUEST_VALUES == set(re.findall(r"``([a-z_]+)``", documented.group(1)))
    with pytest.raises(ConfigurationError) as caught:
        catalogue.model(temprature=0.1)
    assert ", ".join(sorted(models.REQUEST_VALUES)) in str(caught.value)


def test_the_total_ceiling_counts_over_the_models_life_and_a_new_model_starts_over(catalogue):
    model = catalogue.model()
    catalogue.outcomes += [httpx.ConnectError("refused")] * 100
    for _ in range(10):
        with pytest.raises(model_requests.TransportExhausted):
            model.invoke([HumanMessage("a")])
    catalogue.outcomes += [httpx.ConnectError("refused"), reply("late")]
    with pytest.raises(model_requests.TransportExhausted, match=r"in this phase \(limit 100\)"):
        model.invoke([HumanMessage("b")])
    assert catalogue.model(label="again").invoke([HumanMessage("c")]).content == "late"


def test_a_client_whose_tool_binding_carries_no_arguments_is_refused(catalogue, monkeypatch):
    from langchain_core.runnables import RunnableLambda

    monkeypatch.setattr(Scripted, "bind_tools", lambda self, tools, **kwargs: RunnableLambda(str))
    with pytest.raises(ConfigurationError, match=r"\[scripted\].*bind_tools"):
        catalogue.model().bind_tools([])


class ProviderError(Exception):
    """A provider's refusal that holds the key, in its text and in the
    request headers it keeps, as an SDK's errors do."""

    def __init__(self, text):
        super().__init__(text)
        self.headers = {"Authorization": f"Bearer {SENTINEL}"}


def test_a_provider_error_holding_the_key_leaves_no_trace(catalogue):
    import traceback

    other = "sk-other-alias-secret"
    catalogue.write(models=f"\n[other]\nprovider = openai\nmodel = other-model\napi_key = {other}\n")
    model = catalogue.model()
    catalogue.outcomes += [httpx.ConnectError(f"refused for {SENTINEL} and {other}"), reply()]
    model.invoke([HumanMessage("a")])
    raised = []
    for outcomes, kind in (
            ([ProviderError(f"invalid key {SENTINEL}, catalogue key {other}")],
             model_requests.TerminalRequestError),
            ([ProviderError(f"maximum context length exceeded, key {SENTINEL}")],
             model_requests.ContextOverflow),
            ([httpx.ConnectError(f"refused for {SENTINEL}")] * 10, model_requests.TransportExhausted)):
        catalogue.outcomes += outcomes
        with pytest.raises(kind) as caught:
            model.invoke([HumanMessage("b")])
        raised.append(caught.value)
    for error in raised:
        assert error.__cause__ is None and error.__context__ is None
        shown = "".join(traceback.format_exception(error)) + repr(error)
        assert SENTINEL not in shown and other not in shown and "[redacted]" in str(error)
    assert "ProviderError" in str(raised[0]) and raised[2].attempt == 10
    records = (catalogue.records / "request_records.jsonl").read_text(encoding="utf-8")
    assert SENTINEL not in records and other not in records and "[redacted]" in records


def test_a_request_record_that_cannot_be_written_carries_no_provider_error(catalogue):
    import traceback

    model = catalogue.model()
    # A directory in the records file's place, so that appending a record fails.
    (catalogue.records / "request_records.jsonl").mkdir()
    for outcome in (ProviderError(f"invalid key {SENTINEL}"),
                    httpx.ConnectError(f"refused for {SENTINEL}")):
        catalogue.outcomes.append(outcome)
        with pytest.raises(OSError) as caught:
            model.invoke([HumanMessage("a")])
        assert caught.value.__cause__ is None and caught.value.__context__ is None
        assert SENTINEL not in "".join(traceback.format_exception(caught.value))


#: The provider section the tests' provider models go through: an address
#: of its own and the scripted client.
OLLAMA_SECTION = ("[ollama]\napi_url = http://127.0.0.1:9/v1\n"
                  "chat_model = _fixture_scripted_provider:factory\n")


def provider_model(tmp_path, model, provider):
    return catalogue_chat_model(model, provider=provider, base_dir=tmp_path / "project",
                                records_dir=tmp_path / "records", label="profile")


def test_a_providers_model_reaches_its_address_with_its_key_and_records_it_as_a_run(
        catalogue, tmp_path, monkeypatch):
    from testuary.models import resolve_model_identity
    from testuary.settings import resolve_settings

    catalogue.write(models=OLLAMA_SECTION)
    monkeypatch.setenv("TESTUARY_OLLAMA_API_KEY", "sk-ollama-test")
    model = provider_model(tmp_path, "qwen3-coder:30b", "ollama")
    catalogue.outcomes.append(reply())
    model.invoke([HumanMessage("a")])
    built = catalogue.parameters[0]
    assert (built["model"], built["base_url"], built["api_key"]) == (
        "qwen3-coder:30b", "http://127.0.0.1:9/v1", "sk-ollama-test")
    run = resolve_settings(base_dir=tmp_path / "project",
                           overrides={"api": {"model": "qwen3-coder:30b", "provider": "ollama"}})
    identity = resolve_model_identity(run).export()
    recorded = lines(catalogue.records / "model_settings.jsonl")[0]
    assert {key: recorded[key] for key in identity} == identity
    assert identity["provider"] == "ollama" and identity["model"] == "qwen3-coder:30b"


def test_an_alias_named_with_a_provider_resolves_with_its_own_provider(catalogue, tmp_path):
    provider_model(tmp_path, "scripted", "vllm")
    built = catalogue.parameters[0]
    assert (built["model"], built["base_url"], built["api_key"]) == (
        "scripted-model", "http://127.0.0.1:9/v1", SENTINEL)
    assert lines(catalogue.records / "model_settings.jsonl")[0]["provider"] == "openai"


def test_an_unknown_provider_and_a_name_without_one_are_refused_before_any_request(
        catalogue, tmp_path):
    with pytest.raises(ConfigurationError, match=r"provider 'olama' is neither .* of .*models\.ini"):
        provider_model(tmp_path, "qwen3-coder:30b", "olama")
    with pytest.raises(ConfigurationError,
                       match=r"\[qwen3-coder:30b\] is not defined in .*models\.ini.*pass provider"):
        catalogue_chat_model("qwen3-coder:30b", base_dir=tmp_path / "project",
                             records_dir=catalogue.records, label="profile")
    assert not catalogue.parameters and not catalogue.records.exists()


def test_an_alias_named_as_the_provider_is_refused_before_any_request(catalogue, tmp_path):
    """An alias section names a provider of its own, so it is no provider
    section: its address and key do not serve another model name."""
    with pytest.raises(ConfigurationError, match=r"provider 'scripted' is neither .* of .*models\.ini"):
        provider_model(tmp_path, "qwen3-coder:30b", "scripted")
    assert not catalogue.parameters and not catalogue.records.exists()
