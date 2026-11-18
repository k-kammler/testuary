"""Public entry points reject model configuration before project side effects."""
import os
from types import SimpleNamespace
import sys

import pytest

from testuary import __main__ as cli, orchestrator
from testuary import models as _model_source
from testuary import config
from testuary.settings import ConfigurationError, RunSettings, resolve_settings
from agent_fixtures import logs_in


def arguments(**options):
    args = cli._build_parser().parse_args([])
    for key, value in options.items():
        setattr(args, key, value)
    return args


def invalid_settings(tmp_path):
    return RunSettings(base_dir=tmp_path, sections={"api": {"model": "explicit"}}, models={"explicit": {
        "provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
        "protocol": "ollama-native", "reasoning_budget_tokens": 13,
    }})


def test_api_lists_each_selected_phase_problem_before_project_access(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "get_project_config", lambda: pytest.fail("project configuration reached"))
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kwargs: pytest.fail("cleanup reached"))
    with pytest.raises(ConfigurationError) as refused:
        orchestrator.run_pipeline(arguments(), settings=invalid_settings(tmp_path))
    message = str(refused.value)
    assert "starter: ollama-native cannot express reasoning_budget_tokens" in message
    # A problem common to several phases is listed once, with those phases.
    assert "coverage, mutation: ollama-native is supported only for the starter phase" in message
    assert config.active_settings() is None
    assert list(tmp_path.iterdir()) == []


def test_cli_model_preflight_exits_two_without_touching_existing_results(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TESTUARY_HOME", str(tmp_path))
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    (tmp_path / "config.ini").write_text("[api]\nmodel = explicit\n", encoding="utf-8")
    (tmp_path / "models.ini").write_text(
        "[explicit]\nprovider = openai\nmodel = chosen\napi_url = https://example.test/v1\n"
        "protocol = ollama-native\nreasoning_budget_tokens = 13\n", encoding="utf-8",
    )
    sentinel = tmp_path / "results" / "previous.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(config, "get_project_config", lambda: pytest.fail("project configuration reached"))
    monkeypatch.setattr("sys.argv", ["testuary", "--display", "plain", "--no-color"])
    with pytest.raises(SystemExit) as refused:
        cli._main()
    assert refused.value.code == 2
    message = capsys.readouterr().err
    assert "starter:" in message and "coverage, mutation:" in message
    assert "reasoning_budget_tokens" in message
    assert sentinel.read_text("utf-8") == "preserve"
    assert config.active_settings() is None


@pytest.mark.parametrize("options, phases", [
    ({"stop_after_phase": "starter"}, {"starter"}),
    ({"stop_after_phase": "coverage"}, {"starter", "coverage"}),
    ({"reuse_starter": True}, {"coverage", "mutation"}),
    ({"reuse_coverage": True}, {"mutation"}),
    ({"reuse_mutation": True}, set()),
    ({"reuse_starter_suites": True}, {"starter", "coverage", "mutation"}),
])
def test_model_preflight_checks_only_phases_that_can_generate(options, phases, tmp_path):
    with config.use_run_settings(invalid_settings(tmp_path)):
        problems = orchestrator._preflight_model_configuration(arguments(**options))
    assert {phase for problem in problems for phase in problem.split(":", 1)[0].split(", ")} == phases


@pytest.mark.parametrize("source", [False, True])
def test_no_provider_configuration_needed_when_generation_is_replaced_or_reused(source, tmp_path):
    with config.use_run_settings(RunSettings(base_dir=tmp_path)):
        if source:
            with _model_source.installed(SimpleNamespace()):
                assert orchestrator._preflight_model_configuration(arguments()) == ()
        else:
            assert orchestrator._preflight_model_configuration(arguments(reuse_starter=True, reuse_coverage=True, reuse_mutation=True, reuse_finishing=True)) == ()


def test_missing_selected_credential_is_a_preflight_problem(tmp_path):
    settings = RunSettings(base_dir=tmp_path, sections={"api": {"model": "explicit"}}, models={"explicit": {
        "provider": "openai", "model": "chosen", "api_url": "https://example.test/v1", "api_key_env": "UNSET_FIXTURE_KEY",
    }})
    with config.use_run_settings(settings):
        problems = orchestrator._preflight_model_configuration(arguments())
    assert len(problems) == 1
    assert problems[0].startswith("credentials: api_key_env")


def test_bad_reasoning_fields_are_configuration_not_execution_failure(tmp_path, monkeypatch):
    settings = RunSettings(base_dir=tmp_path, sections={"api": {"model": "explicit"}}, models={"explicit": {
        "provider": "openai", "model": "chosen", "api_url": "https://example.test/v1", "reasoning_field_names": "content",
    }})
    monkeypatch.setattr(config, "get_project_config", lambda: pytest.fail("project configuration reached"))
    with pytest.raises(ConfigurationError, match="reasoning_field_names cannot overwrite"):
        orchestrator.run_pipeline(arguments(), settings=settings)
    assert config.active_settings() is None


def test_valid_builtin_preflight_neither_constructs_a_client_nor_reads_ambient_credentials(tmp_path, monkeypatch):
    from testuary import models as _model_protocols

    settings = RunSettings(base_dir=tmp_path, sections={"api": {"model": "explicit"}}, models={"explicit": {
        "provider": "openai", "model": "chosen", "api_url": "https://example.test/v1", "api_key_env": "FIXTURE_KEY",
    }}, environment={"FIXTURE_KEY": "captured"})
    monkeypatch.setenv("FIXTURE_KEY", "")
    monkeypatch.setattr(_model_protocols, "build_resolved_chat_model", lambda *a, **k: pytest.fail("client constructed"))
    with config.use_run_settings(settings):
        assert orchestrator._preflight_model_configuration(arguments()) == ()
    assert list(tmp_path.iterdir()) == []


def test_preflight_cancellation_releases_the_invocation_guard(tmp_path, monkeypatch):
    from testuary import models as _model_configuration

    def interrupted(*args):
        raise KeyboardInterrupt

    monkeypatch.setattr(_model_configuration, "resolve_phase_model", interrupted)
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run_pipeline(arguments(), settings=invalid_settings(tmp_path))
    assert config.active_settings() is None
    with config.use_run_settings(RunSettings(base_dir=tmp_path)):
        assert config.active_settings() is not None


def test_a_replaced_model_resolution_does_not_outlive_its_patch_in_the_starters(monkeypatch):
    """The starter phase looks the phase model up at call time, as every
    other caller does, so a replacement that was active while
    testuary.starters was first imported (the cancellation test above, when
    it is the first test in a worker to reach the starters) does not stay
    there after the replacement ends."""
    import importlib

    import testuary
    from testuary import models as _model_configuration

    real = _model_configuration.resolve_phase_model
    importlib.import_module("testuary.starters")
    saved = sys.modules.pop("testuary.starters")
    try:
        with monkeypatch.context() as patched:
            patched.setattr(_model_configuration, "resolve_phase_model",
                            lambda *a: pytest.fail("the replacement outlived its patch"))
            fresh = importlib.import_module("testuary.starters")
        assert _model_configuration.resolve_phase_model is real
        assert getattr(fresh, "resolve_phase_model", real) is real
    finally:
        sys.modules["testuary.starters"] = saved
        testuary.starters = saved


@pytest.mark.parametrize("protocol, option", [
    ("openai-responses", "max_output_tokens"),
    ("added", "base_url"), ("added", "timeout"), ("added", "max_retries"),
])
def test_preflight_refuses_request_options_overriding_controlled_inputs(protocol, option, tmp_path):
    fields = {"provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
              "protocol": protocol, "request_options": {option: 5}}
    if protocol == "added":
        fields["chat_model"] = "fixture:factory"
    settings = RunSettings(base_dir=tmp_path, sections={"api": {"model": "explicit"}}, models={"explicit": fields})
    with config.use_run_settings(settings):
        problems = orchestrator._preflight_model_configuration(arguments())
    assert len(problems) == 1 and problems[0].startswith("starter, coverage, mutation: ")
    assert "cannot override controlled field" in problems[0] and option in problems[0]


def test_catalogue_typo_diagnostics_cover_all_sections_without_echoing_values(tmp_path, monkeypatch):
    settings = RunSettings(base_dir=tmp_path, sections={"api": {"model": "explicit"}}, models={
        "openai": {"reasoning_efort": "credential-sentinel"},
        "explicit": {"provider": "openai", "model": "chosen", "temperatue": "credential-sentinel"},
        "unused_alias": {"provider": "openai", "api_urll": "credential-sentinel"},
    })
    monkeypatch.setattr(config, "get_project_config", lambda: pytest.fail("project configuration reached"))
    with pytest.raises(ConfigurationError) as refused:
        orchestrator.run_pipeline(arguments(), settings=settings)
    message = str(refused.value)
    for section, key, replacement in (
        ("openai", "reasoning_efort", "reasoning_effort"),
        ("explicit", "temperatue", "temperature"),
        ("unused_alias", "api_urll", "api_url"),
    ):
        assert f"[{section}]" in message
        assert key in message and replacement in message
    assert "credential-sentinel" not in message
    assert list(tmp_path.iterdir()) == []


def test_catalogue_validation_accepts_request_information_and_canonical_fields(tmp_path):
    fields = {"provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
              "deployment": "v1", "api_key": "private", "api_key_env": "FIXTURE_KEY",
              "price_input_cache_hit": ".1", "price_input_cache_miss": ".2", "price_output": ".3",
              "protocol": "openai-chat", "reasoning_dialect": "vllm",
              "temperature": "", "reasoning": "", "reasoning_budget_tokens": "",
              "request_timeout_seconds": 20, "request_options": {"stop": ["end"]}}
    settings = RunSettings(base_dir=tmp_path, sections={"api": {"model": "explicit"}},
                           models={"explicit": fields}, environment={"FIXTURE_KEY": "captured"})
    with config.use_run_settings(settings):
        assert orchestrator._preflight_model_configuration(arguments()) == ()


@pytest.mark.parametrize("unknown,close", [("temprature", "temperature"), ("model_mode", None)])
def test_actual_pipeline_refuses_unknown_catalogue_fields_before_target_work(unknown, close, tmp_path, monkeypatch):
    settings = RunSettings(base_dir=tmp_path, sections={"api": {"model": "explicit"}}, models={
        "explicit": {"provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
                     "api_key": "fixture", unknown: "private-sentinel"},
    })
    monkeypatch.setattr(config, "get_project_config", lambda: pytest.fail("target work reached"))
    with pytest.raises(ConfigurationError) as refused:
        orchestrator.run_pipeline(arguments(), settings=settings)
    assert f"unknown setting {unknown!r}" in str(refused.value)
    assert close is None or close in str(refused.value)
    assert "private-sentinel" not in str(refused.value)
    assert list(tmp_path.iterdir()) == []


def test_nontext_catalogue_key_is_refused_by_settings_without_its_value(tmp_path):
    with pytest.raises(TypeError, match="mapping keys must be strings") as refused:
        RunSettings(base_dir=tmp_path, sections={"api": {"model": "explicit"}}, models={
            "explicit": {"provider": "openai", "model": "chosen", 1: "credential-sentinel"},
        })
    assert "credential-sentinel" not in str(refused.value)


@pytest.mark.parametrize("section", ["api", "openai", "ollama", "vllm"])
@pytest.mark.parametrize("key", ["", "credential-sentinel"])
def test_configuration_file_credentials_refuse_with_source_and_supported_routes(section, key, tmp_path):
    path = tmp_path / "private-config.ini"
    path.write_text(f"[{section}]\napi_key = {key}\n", encoding="utf-8")
    with pytest.raises(ValueError) as refused:
        resolve_settings(config_files=[path], base_dir=tmp_path, use_environment=False, models={})
    message = str(refused.value)
    assert str(path) in message
    assert f"[{section}] api_key" in message
    assert "catalogue" in message and "environment" in message
    assert "credential-sentinel" not in message


def test_explicit_configuration_credentials_also_refuse_without_echoing_the_key():
    with pytest.raises(ValueError, match="api_key.*catalogue.*environment") as refused:
        RunSettings(sections={"api": {"api_key": "credential-sentinel"}})
    assert "credential-sentinel" not in str(refused.value)


@pytest.mark.parametrize("provider", ["openai", "ollama", "vllm", "custom"])
def test_default_credential_environment_is_not_a_configuration_field(provider, tmp_path):
    from testuary.models import resolve_provider_credential

    settings = resolve_settings(base_dir=tmp_path, config_files=[],
                                overrides={"api": {"model": "explicit"}}, models={"explicit": {
        "provider": provider, "model": "chosen", "api_url": "https://example.test/v1",
    }}, environment={f"TESTUARY_{provider.upper()}_API_KEY": "captured"})
    assert "api_key" not in settings.sections.get(provider, {})
    assert resolve_provider_credential(provider, {}, {}, settings.environment) == "captured"
    assert "captured" not in str(settings.export())
    with config.use_run_settings(settings):
        assert orchestrator._preflight_model_configuration(arguments()) == ()


def test_cli_configuration_key_refusal_is_exit_two_and_keeps_the_key_private(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TESTUARY_HOME", str(tmp_path))
    (tmp_path / "config.ini").write_text("[openai]\napi_key = credential-sentinel\n", encoding="utf-8")
    monkeypatch.setattr(config, "get_project_config", lambda: pytest.fail("project configuration reached"))
    monkeypatch.setattr("sys.argv", ["testuary", "--display", "plain", "--no-color"])
    with pytest.raises(SystemExit) as refused:
        cli._main()
    assert refused.value.code == 2
    error = capsys.readouterr().err
    assert "[openai] api_key" in error
    assert "credential-sentinel" not in error
    assert config.active_settings() is None


@pytest.fixture
def added_provider(monkeypatch, provider_module):
    from langchain_core.language_models import BaseChatModel
    from langchain_core.messages import AIMessage
    from langchain_core.outputs import ChatGeneration, ChatResult

    state = SimpleNamespace(models=[], reject=None, reject_seed=None, allow_requests=False)

    class Model(BaseChatModel):
        parameters: dict
        bindings: list = []
        calls: int = 0

        @property
        def _llm_type(self):
            return "preflight-fixture"

        def bind_tools(self, tools, **kwargs):
            if state.reject == "binding":
                raise ValueError("credential-sentinel")
            if state.reject == "binding-result":
                return object()
            names = tuple(tool.name for tool in tools)
            self.bindings.append(names)
            return self.bind(bound_names=names)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            assert state.allow_requests, "preflight made a model request"
            self.calls += 1
            return ChatResult(generations=[ChatGeneration(message=AIMessage(
                content="no code fence", usage_metadata={
                    "input_tokens": 10, "output_tokens": 3, "total_tokens": 13,
                },
            ))])

    module = provider_module("_fixture_preflight_provider")

    def factory(**parameters):
        if state.reject == "construction" or (state.reject_seed is not None
                                               and parameters.get("seed") == state.reject_seed):
            raise ValueError("credential-sentinel")
        if state.reject == "type":
            return object()
        if state.reject == "cancelled":
            raise KeyboardInterrupt("fixture cancellation")
        model = Model(parameters=parameters)
        state.models.append(model)
        return model

    module.factory = factory
    state.factory = module.__name__ + ":factory"
    return state


def added_settings(tmp_path, added_provider, *, factory=None, phase_fields=None, starter_fields=None):
    return RunSettings(base_dir=tmp_path, sections={
        "api": {"model": "explicit"}, "main": {"seed": 37},
        # One attempt per slot: these tests count the requests of two slots.
        "starter": {"suite_count": 2, "refill_failed_suites": False, "max_output_tokens": 40,
                         **(starter_fields or {})},
        "coverage": {"max_output_tokens": 60, "max_loop_attempts": 0,
                          **(phase_fields or {})},
        "mutation": {"max_output_tokens": 80, "max_loop_attempts": 0},
    }, models={"explicit": {
        "provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
        "protocol": "added", "chat_model": factory or added_provider.factory,
        "api_key": "credential-sentinel", "temperature": 0,
    }}, environment=dict(os.environ))


@pytest.mark.parametrize("rejection, message", [
    ("import", "no module _missing_preflight_fixture is found"),
    ("construction", "chat_model construction failed"),
    ("type", "chat_model must construct"),
    ("binding", "tool binding failed"),
    ("binding-result", "tool binding failed"),
])
def test_added_provider_refusal_precedes_target_work_and_is_secret_safe(
        rejection, message, tmp_path, monkeypatch, added_provider):
    added_provider.reject = rejection
    factory = "_missing_preflight_fixture:factory" if rejection == "import" else None
    monkeypatch.setattr(config, "get_project_config", lambda: pytest.fail("target work reached"))
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kwargs: pytest.fail("cleanup reached"))
    with pytest.raises(ConfigurationError) as refused:
        orchestrator.run_pipeline(arguments(), settings=added_settings(tmp_path, added_provider, factory=factory))
    assert message in str(refused.value)
    assert "credential-sentinel" not in str(refused.value)
    assert all(model.calls == 0 for model in added_provider.models)
    assert config.active_settings() is None
    assert list(tmp_path.iterdir()) == []


def test_added_provider_preflight_checks_all_starting_slots_without_requests(tmp_path, added_provider):
    args = arguments()
    with config.use_run_settings(added_settings(tmp_path, added_provider)):
        assert orchestrator._preflight_model_configuration(args) == ()
    assert len(added_provider.models) == 4
    assert [(model.parameters.get("seed"), model.parameters["max_tokens"])
            for model in added_provider.models[:2]] == [(38, 40), (39, 40)]
    assert [model.parameters["max_tokens"] for model in added_provider.models[2:]] == [60, 80]
    # Each loop binds its six modes: Revise, Validate, Assess with and
    # without the edit tool, and the start's read and measurement.
    assert [len(model.bindings) for model in added_provider.models] == [0, 0, 6, 6]
    assert all(model.calls == 0 for model in added_provider.models)
    assert list(tmp_path.iterdir()) == []


def test_added_provider_cancellation_releases_the_invocation(tmp_path, added_provider):
    added_provider.reject = "cancelled"
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run_pipeline(arguments(), settings=added_settings(tmp_path, added_provider))
    assert config.active_settings() is None
    with config.use_run_settings(RunSettings()):
        assert config.active_settings() is not None


def test_prepared_starter_slots_run_on_the_same_clients_but_next_target_does_not(
        tmp_path, monkeypatch, added_provider):
    from testuary import starters

    args = arguments(stop_after_phase="starter")
    with config.use_run_settings(added_settings(tmp_path, added_provider)):
        assert orchestrator._preflight_model_configuration(args) == ()
        prepared = list(added_provider.models)
        added_provider.allow_requests = True
        for name in ("first", "second"):
            project = tmp_path / name
            project.mkdir()
            target = project / "module.py"
            target.write_text("def add(x):\n    return x + 1\n", encoding="utf-8")
            result = starters.run_starter_phase(
                language="python", target_file_path=str(target),
                source_dir=str(project), test_output_dir=str(project / "output"),
                suites_dir=str(project / "suites"), log=lambda *a, **k: None,
                prepared_models=args._prepared_models,
            )
            assert result.requests_used == result.logical_requests == 2
    assert len(added_provider.models) == 4
    assert all(model.calls == 1 for model in added_provider.models)
    assert added_provider.models[:2] == prepared
    assert all(added_provider.models[i] is not added_provider.models[i + 2] for i in (0, 1))


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
def test_prepared_editing_client_reaches_actual_entry_and_is_not_shared_between_targets(
        phase, tmp_path, monkeypatch, added_provider):
    from testuary.agent import coverage_agent, mutation_agent

    cls = coverage_agent.CoverageAgent if phase == "coverage" else mutation_agent.MutationAgent

    def finalized(agent, output, **kwargs):
        return SimpleNamespace(client=agent.build_chat_model(), requests=kwargs["usage_tracker"].requests_used)

    monkeypatch.setattr(cls, "finalize_result", finalized)
    monkeypatch.setattr(mutation_agent, "_seed_mutation_baseline_from_matrix", lambda agent: True)
    # Only the phase under test generates.
    args = arguments(**({"reuse_starter": True, "stop_after_phase": "coverage"} if phase == "coverage"
                        else {"reuse_coverage": True}))
    with config.use_run_settings(added_settings(tmp_path, added_provider)):
        assert orchestrator._preflight_model_configuration(args) == ()
        assert len(added_provider.models) == 1
        prepared = added_provider.models[0]
        added_provider.allow_requests = True
        observed = []
        for name in ("first", "second"):
            project = tmp_path / name
            project.mkdir()
            (project / "pyproject.toml").write_text("", encoding="utf-8")
            target = project / "module.py"
            target.write_text("def add(x):\n    return x + 1\n", encoding="utf-8")
            tests = project / "tests"
            tests.mkdir()
            source = "from module import add\n\ndef test_add():\n    assert add(1) == 2\n"
            (tests / "test_module.py").write_text(source, encoding="utf-8")
            logs_in(monkeypatch, project / "logs")
            if phase == "coverage":
                result = coverage_agent.generate_coverage_tests(
                    "fixture", top_file={"file_path": str(target)}, test_output_dir=str(tests),
                    baseline={"content": source, "line_coverage": 50.0, "branch_coverage": 50.0,
                              "test_count": 1, "suite_seconds": 0.01},
                    prepared_models=args._prepared_models,
                )
            else:
                result = mutation_agent.generate_mutation_tests(
                    "fixture", str(target), str(tests), preseed={"content": source},
                    prepared_models=args._prepared_models,
                )
            assert result.requests == 1
            observed.append(result.client)
    assert observed[0] is prepared
    assert observed[1] is not prepared
    assert len(added_provider.models) == 2
    assert [model.calls for model in added_provider.models] == [1, 1]


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_target_phase_handlers_forward_the_prepared_owner(phase, tmp_path, monkeypatch, default_run):
    import test_orchestrator_phases as harness
    from testuary import target_phases as phases

    owner = object()
    ctx = harness._make_ctx(tmp_path, prepared_models=owner)
    result = phases.TestPipelineResult(test_file_name="test_foo.py")
    (ctx.work_tests / "test_foo.py").write_text("def test_f():\n    assert True\n", encoding="utf-8")
    seen = []

    def called(*args, **kwargs):
        seen.append(kwargs["prepared_models"])
        raise RuntimeError("fixture reached entry")

    if phase == "starter":
        monkeypatch.setattr(phases.starters, "run_starter_phase", called)
        with pytest.raises(RuntimeError, match="fixture reached entry"):
            phases._phase_starter(ctx, result)
    else:
        driver = phases.coverage_agent if phase == "coverage" else phases.mutation_agent
        monkeypatch.setattr(driver, "init_logging", lambda *a, **k: None)
        monkeypatch.setattr(driver, "save_logs", lambda: None)
        monkeypatch.setattr(driver, "generate_coverage_tests" if phase == "coverage" else "generate_mutation_tests", called)
        if phase == "coverage":
            with pytest.raises(RuntimeError, match="fixture reached entry"):
                phases._phase_coverage(ctx, result)
        else:
            assert phases._phase_mutation(ctx, result, session=None) is None
    assert seen == [owner]


def test_paid_starter_recovery_preflights_starting_clients_without_requests(tmp_path, added_provider):
    args = arguments(stop_after_phase="starter", reuse_starter_suites=True)
    with config.use_run_settings(added_settings(tmp_path, added_provider)):
        assert orchestrator._preflight_model_configuration(args) == ()
    assert len(added_provider.models) == 2
    assert all(model.calls == 0 for model in added_provider.models)
    assert args._prepared_models is not None


def test_completed_paid_starter_slots_are_not_replayed_by_preflight(tmp_path, monkeypatch, added_provider):
    from importlib import metadata
    from testuary import starters

    distributions, version = metadata.packages_distributions, metadata.version
    monkeypatch.setattr(metadata, "packages_distributions", lambda: {
        **distributions(), "_fixture_preflight_provider": ["fixture-preflight-provider"],
    })
    monkeypatch.setattr(metadata, "version", lambda name: "1.0" if name == "fixture-preflight-provider" else version(name))
    project = tmp_path / "subject"
    project.mkdir()
    target = project / "module.py"
    target.write_text("def add(x):\n    return x + 1\n", encoding="utf-8")
    starter_dir = project / "suites"
    for reuse in (False, True):
        args = arguments(stop_after_phase="starter")
        settings = added_settings(tmp_path, added_provider, starter_fields={"reuse_suites": reuse})
        added_provider.allow_requests = not reuse
        with config.use_run_settings(settings):
            assert orchestrator._preflight_model_configuration(args) == ()
            result = starters.run_starter_phase(
                language="python", target_file_path=str(target),
                source_dir=str(project), test_output_dir=str(project / "output"),
                suites_dir=str(starter_dir), log=lambda *a, **k: None,
                prepared_models=args._prepared_models,
            )
        assert result.requests_used == result.logical_requests == 2
        records = (starter_dir / "request_records.jsonl").read_bytes()
        if not reuse:
            paid = records
        else:
            assert records == paid
    assert len(added_provider.models) == 4
    assert [model.calls for model in added_provider.models] == [1, 1, 0, 0]


def test_added_preflight_refuses_a_later_starting_slot_before_target_work(tmp_path, monkeypatch, added_provider):
    added_provider.reject_seed = 39
    monkeypatch.setattr(config, "get_project_config", lambda: pytest.fail("target work reached"))
    with pytest.raises(ConfigurationError, match="starter: chat_model construction failed"):
        orchestrator.run_pipeline(arguments(stop_after_phase="starter"),
                                  settings=added_settings(tmp_path, added_provider))
    assert len(added_provider.models) == 1
    assert added_provider.models[0].parameters["seed"] == 38
    assert added_provider.models[0].calls == 0


def test_added_preflight_skips_reused_phases_and_does_not_retain_stale_preparations(tmp_path, added_provider):
    args = arguments(stop_after_phase="starter")
    with config.use_run_settings(added_settings(tmp_path, added_provider)):
        assert orchestrator._preflight_model_configuration(args) == ()
        assert args._prepared_models is not None
        count = len(added_provider.models)
        args.reuse_starter = True
        added_provider.reject = "construction"
        assert orchestrator._preflight_model_configuration(args) == ()
        assert args._prepared_models is None
        assert len(added_provider.models) == count


def test_prepared_clients_cannot_be_borrowed_by_a_new_invocation(tmp_path, monkeypatch, added_provider):
    from testuary import starters

    settings = added_settings(tmp_path, added_provider)
    args = arguments(stop_after_phase="starter")
    with config.use_run_settings(settings):
        assert orchestrator._preflight_model_configuration(args) == ()
        prepared = list(added_provider.models)
    project = tmp_path / "subject"
    project.mkdir()
    target = project / "module.py"
    target.write_text("def add(x):\n    return x + 1\n", encoding="utf-8")
    added_provider.allow_requests = True
    with config.use_run_settings(settings):
        result = starters.run_starter_phase(
            language="python", target_file_path=str(target),
            source_dir=str(project), test_output_dir=str(project / "output"),
            suites_dir=str(project / "suites"), log=lambda *a, **k: None,
            prepared_models=args._prepared_models,
        )
    assert result.requests_used == result.logical_requests == 2
    assert all(model.calls == 0 for model in prepared)
    assert len(added_provider.models) == 4
    assert [model.calls for model in added_provider.models[2:]] == [1, 1]


def test_cli_added_provider_binding_refusal_preserves_results_and_reports_no_secret(
        tmp_path, monkeypatch, capsys, added_provider):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TESTUARY_HOME", str(tmp_path))
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    (tmp_path / "config.ini").write_text("[api]\nmodel = explicit\n", encoding="utf-8")
    (tmp_path / "models.ini").write_text(
        "[explicit]\nprovider = openai\nmodel = chosen\napi_url = https://example.test/v1\n"
        f"protocol = added\nchat_model = {added_provider.factory}\napi_key = credential-sentinel\n",
        encoding="utf-8",
    )
    sentinel = tmp_path / "results" / "previous.txt"
    sentinel.parent.mkdir()
    sentinel.write_text("preserve", encoding="utf-8")
    added_provider.reject = "binding"
    monkeypatch.setattr(config, "get_project_config", lambda: pytest.fail("target work reached"))
    monkeypatch.setattr("sys.argv", ["testuary", "--display", "plain", "--no-color"])
    with pytest.raises(SystemExit) as refused:
        cli._main()
    assert refused.value.code == 2
    error = capsys.readouterr().err
    diagnostic = " ".join(error.split())
    assert "coverage: chat_model tool binding failed" in diagnostic
    assert "mutation: chat_model tool binding failed" in diagnostic
    assert "credential-sentinel" not in error
    assert sentinel.read_text("utf-8") == "preserve"
    assert all(model.calls == 0 for model in added_provider.models)


LEAKED_KEY = "leaked-key-sentinel-7f3a"


@pytest.mark.parametrize("command", [["--display", "plain", "--no-color"],
                                     ["check", "--display", "plain", "--no-color"]])
def test_a_provider_module_whose_import_error_quotes_the_key_never_shows_it(command, tmp_path, monkeypatch, capsys):
    """An added provider's module may fail to import with a message that
    quotes the key. Neither generate nor check shows the key, in its output
    or in any file it writes."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("TESTUARY_HOME", str(tmp_path))
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    (tmp_path / "leaky_provider_for_privacy.py").write_text(
        f"raise RuntimeError('cannot authenticate with {LEAKED_KEY}')\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, "leaky_provider_for_privacy", raising=False)
    (tmp_path / "config.ini").write_text("[api]\nmodel = explicit\n", encoding="utf-8")
    (tmp_path / "models.ini").write_text(
        "[explicit]\nprovider = openai\nmodel = chosen\napi_url = https://example.test/v1\n"
        f"protocol = added\nchat_model = leaky_provider_for_privacy:factory\napi_key = {LEAKED_KEY}\n",
        encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["testuary", *command, "--source-files", "module.py"])
    with pytest.raises(SystemExit) as refused:
        cli._main()
    assert refused.value.code != 0
    output = capsys.readouterr()
    assert "leaky_provider_for_privacy" in output.out + output.err
    assert "RuntimeError" in output.out + output.err
    assert LEAKED_KEY not in output.out + output.err
    for path in tmp_path.rglob("*"):
        # The provider module, its bytecode and models.ini hold the key themselves.
        if (path.is_file() and "__pycache__" not in path.parts
                and path.name not in {"models.ini", "leaky_provider_for_privacy.py"}):
            assert LEAKED_KEY.encode() not in path.read_bytes(), path
