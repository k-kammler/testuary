"""The configuration accessors read the active run's settings: the bundled
defaults, the user directory's files and the ``TESTUARY_`` variables as
``resolve_settings`` resolved them, and nothing outside a run. Each test's
own directory (``tmp_cwd``) is its user directory.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from testuary import config as config_module
from testuary import models
from testuary.settings import ConfigurationError
from testuary.settings import _bundled_data_path, _bundled_template_path, _deep_merge_ini
from testuary.config import load_config
from testuary import config
from testuary.settings import PHASE_SECTIONS, resolve_settings


@pytest.fixture(autouse=True)
def _the_test_dir_is_the_user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A file a test writes into ``tmp_cwd`` is in the user directory."""
    from testuary import settings

    monkeypatch.setattr(settings, "platform_user_dir", lambda: tmp_path)


def resolved(**kwargs):
    """The run settings of the user directory's files and the test's variables,
    without a project settings file."""
    kwargs.setdefault("project_settings", False)
    return resolve_settings(**kwargs)


def agent(section: str, parent: str | None = None, **kwargs) -> dict:
    """``get_agent_config`` inside a run of ``resolved(**kwargs)``."""
    with config.use_run_settings(resolved(**kwargs)):
        return config_module.get_agent_config(section, parent=parent)


# ---------------------------------------------------------------------------
# Bundled-data lookup
# ---------------------------------------------------------------------------

def test_bundled_config_ini_exists() -> None:
    """The package ships ``testuary/config.ini`` as the default."""
    p = _bundled_data_path("config.ini")
    assert p is not None
    assert p.exists()
    assert p.is_file()


def test_bundled_models_template_exists() -> None:
    """The package ships ``templates/models.ini`` for ``testuary init``."""
    p = _bundled_template_path("models.ini")
    assert p is not None
    assert p.exists()


def test_repository_model_example_forwards_environment_credentials(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from testuary.agent import _langgraph
    from testuary import _chat_models

    example = Path(__file__).resolve().parents[2] / "models.example.ini"
    (tmp_cwd / "models.ini").write_bytes(example.read_bytes())
    monkeypatch.setenv("TESTUARY_OPENAI_API_KEY", "example-environment-key")
    monkeypatch.setenv("TESTUARY_API_MODEL", "example_model")
    with config.use_run_settings(resolved()):
        api = models.get_api_config()
    assert api["provider"] == "openai"
    assert api["api_key"] == "example-environment-key"
    assert api["model"] == "REPLACE_WITH_MODEL_IDENTIFIER"
    assert api["api_url"] == "https://provider.example/v1"
    assert api["max_output_tokens"] == 8192

    monkeypatch.setattr(_langgraph.httpx, "Client", lambda **kwargs: object())
    monkeypatch.setattr(_langgraph.httpx, "AsyncClient", lambda **kwargs: object())
    monkeypatch.setattr(_chat_models, "_ChatOpenAIExactSampling", lambda **kwargs: kwargs)
    client = _chat_models.build_chat_model(
        model=api["model"], api_url=api["api_url"], api_key=api["api_key"],
        timeout_seconds=1.0, max_tokens=api["max_output_tokens"],
    )
    assert client["api_key"] == api["api_key"]
    assert client["model"] == api["model"]
    assert client["base_url"] == api["api_url"]
    assert client["max_tokens"] == api["max_output_tokens"]


def test_bundled_data_returns_none_for_missing_name() -> None:
    assert _bundled_data_path("does_not_exist.ini") is None


# ---------------------------------------------------------------------------
# Deep merge
# ---------------------------------------------------------------------------

def test_deep_merge_override_wins_per_key() -> None:
    """``override`` wins per key, and keys only in ``base`` survive."""
    base = {"sec": {"a": "base_a", "b": "base_b"}}
    override = {"sec": {"a": "override_a"}}
    merged = _deep_merge_ini(base, override)
    assert merged == {"sec": {"a": "override_a", "b": "base_b"}}


def test_deep_merge_new_section_in_override_is_added() -> None:
    base = {"sec1": {"a": "1"}}
    override = {"sec2": {"b": "2"}}
    merged = _deep_merge_ini(base, override)
    assert merged == {"sec1": {"a": "1"}, "sec2": {"b": "2"}}


def test_deep_merge_does_not_mutate_inputs() -> None:
    """Both layers come out unchanged, so the bundled dict can be reused."""
    base = {"sec": {"a": "1"}}
    override = {"sec": {"a": "2"}}
    base_copy = {"sec": dict(base["sec"])}
    override_copy = {"sec": dict(override["sec"])}
    _deep_merge_ini(base, override)
    assert base == base_copy
    assert override == override_copy


# ---------------------------------------------------------------------------
# Outside a run
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("read", [
    lambda: config_module.load_config("main"),
    lambda: config_module.get_agent_config("test_coverage"),
    lambda: config_module.get_project_config(),
    lambda: config_module.project_exec_mode(),
    lambda: config_module.configured_environment(),
    lambda: config_module.configuration_base_dir(),
    lambda: config_module.run_settings().models,
])
def test_no_configuration_is_read_outside_a_run(read, tmp_cwd: Path,
                                                monkeypatch: pytest.MonkeyPatch) -> None:
    """Outside a run nothing reads configuration files or the environment:
    no second resolution beside ``resolve_settings``."""
    (tmp_cwd / "config.ini").write_text("[main]\nproject_exec = container\n", encoding="utf-8")
    monkeypatch.setenv("TESTUARY_MAIN_PROJECT_EXEC", "container")
    with pytest.raises(RuntimeError, match="no run settings are active"):
        read()


# ---------------------------------------------------------------------------
# What a run's settings give the accessors
# ---------------------------------------------------------------------------

def test_bundled_mutation_workers_default_is_sequential(tmp_cwd: Path) -> None:
    """The bundled mutation worker count is 1 (sequential cosmic-ray), so a
    mutant's verdict is reproducible by default."""
    with config.use_run_settings(resolved()):
        assert load_config("mutation")["engine_workers"] == "1"
        assert load_config("mutation")["cosmic_ray_distributor"] == "auto"


def test_get_agent_config_surfaces_cosmic_ray_clone_dir(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cosmic_ray_clone_dir surfaces in the agent settings and takes an
    environment override, so worker clones can move off a shared filesystem."""
    cfg = agent("test_mutation")
    assert "cosmic_ray_clone_dir" in cfg
    assert "cosmic_ray_local_session_dir" in cfg
    monkeypatch.setenv("TESTUARY_MUTATION_COSMIC_RAY_CLONE_DIR", "/dev/shm/cr_clones")
    assert agent("test_mutation")["cosmic_ray_clone_dir"] == "/dev/shm/cr_clones"


def test_get_agent_config_surfaces_cosmic_ray_keep_output(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cosmic_ray_keep_output surfaces as a parsed bool, False by default and
    overridable by the environment."""
    assert agent("test_mutation")["cosmic_ray_keep_output"] is False
    monkeypatch.setenv("TESTUARY_MUTATION_COSMIC_RAY_KEEP_OUTPUT", "true")
    assert agent("test_mutation")["cosmic_ray_keep_output"] is True


def test_the_user_dirs_config_overrides_specific_keys(tmp_cwd: Path) -> None:
    """A partial config.ini overrides only its own keys."""
    (tmp_cwd / "config.ini").write_text(
        "[openai]\napi_url = https://file.example/v1\n", encoding="utf-8",
    )
    with config.use_run_settings(resolved()):
        assert load_config("openai")["api_url"] == "https://file.example/v1"
        assert load_config("ollama")["api_url"] == "https://ollama.com/v1"


def test_load_config_returns_copies_and_empty_dict_for_missing_section(tmp_cwd: Path) -> None:
    """A missing section is ``{}``, not ``None``, and a caller's change to
    what it got changes no later read."""
    with config.use_run_settings(resolved()):
        assert load_config("section_that_does_not_exist") == {}
        load_config("api")["model"] = "changed"
        load_config()["api"]["model"] = "changed"
        assert load_config("api").get("model") != "changed"


def test_load_models_without_a_catalogue(tmp_cwd: Path) -> None:
    """Without a models.ini in the user directory the catalogue is empty."""
    with config.use_run_settings(resolved()):
        assert config_module.run_settings().models == {}


def test_load_models_reads_the_user_dir(tmp_cwd: Path) -> None:
    """The user directory's ``models.ini`` is the run's catalogue."""
    (tmp_cwd / "models.ini").write_text(
        "[my_model]\nprovider = ollama\nmodel = test-model\n", encoding="utf-8",
    )
    with config.use_run_settings(resolved()):
        assert config_module.run_settings().models["my_model"] == {"provider": "ollama", "model": "test-model"}


def test_api_key_comes_from_models_ini_provider_section_not_config(
    tmp_cwd: Path,
) -> None:
    """API keys come from models.ini's ``[<provider>]`` section; config.ini
    supplies the api_url and may hold no key."""
    (tmp_cwd / "config.ini").write_text(
        "[api]\nmodel = ollama_test\nprovider = ollama\n\n"
        "[ollama]\napi_url = https://ollama.example/v1\n",
        encoding="utf-8",
    )
    (tmp_cwd / "models.ini").write_text(
        "[ollama]\napi_key = sk-from-models\n\n"
        "[ollama_test]\nprovider = ollama\nmodel = test-model:cloud\n",
        encoding="utf-8",
    )
    with config.use_run_settings(resolved()):
        before = load_config("ollama")
        api = models.get_api_config()
        assert load_config("ollama") == before
    assert api["api_key"] == "sk-from-models"
    assert api["api_url"] == "https://ollama.example/v1"


def test_alias_api_url_in_models_ini_overrides_config_default(
    tmp_cwd: Path,
) -> None:
    """A per-alias ``api_url`` in models.ini overrides the provider default in
    config.ini."""
    (tmp_cwd / "config.ini").write_text(
        "[api]\nmodel = openai_gateway\nprovider = openai\n\n"
        "[openai]\napi_url = https://api.openai.com/v1\n",
        encoding="utf-8",
    )
    (tmp_cwd / "models.ini").write_text(
        "[openai]\napi_key = sk-from-models\n\n"
        "[openai_gateway]\nprovider = openai\nmodel = qwen3\n"
        "api_url = https://gateway.example.com/v1\n",
        encoding="utf-8",
    )
    with config.use_run_settings(resolved()):
        api = models.get_api_config()
    assert api["api_url"] == "https://gateway.example.com/v1"
    assert api["api_key"] == "sk-from-models"
    assert api["provider"] == "openai"


def test_raw_model_without_a_provider_is_refused(tmp_cwd: Path) -> None:
    (tmp_cwd / "config.ini").write_text(
        "[api]\nmodel = unprefixed\n", encoding="utf-8",
    )
    with config.use_run_settings(resolved()):
        with pytest.raises(ConfigurationError, match=r"is not an alias in .*models\.ini.*\[api\] provider"):
            models.get_api_config()


def test_reasoning_budget_tokens_surfaces_per_model(tmp_cwd: Path) -> None:
    """The reasoning cap is per model: a ``[<provider>]`` default in models.ini
    reaches the phase's model, and a per-alias value overrides it."""
    (tmp_cwd / "models.ini").write_text(
        "[vllm]\napi_key = EMPTY\nreasoning_dialect = vllm\nreasoning_budget_tokens = 16384\n\n"
        "[vllm_test]\nprovider = vllm\nmodel = test-model\n\n"
        "[vllm_over]\nprovider = vllm\nmodel = other\nreasoning_budget_tokens = 4096\n",
        encoding="utf-8",
    )
    for alias, expected in (("vllm_test", 16384), ("vllm_over", 4096)):
        (tmp_cwd / "config.ini").write_text(
            f"[api]\nmodel = {alias}\nprovider = vllm\n", encoding="utf-8",
        )
        with config.use_run_settings(resolved()):
            model = models.resolve_phase_model(config.run_settings(), "coverage")
            assert model.values["reasoning_budget_tokens"] == expected


def test_the_api_configuration_refuses_an_alias_cap_the_run_refuses(tmp_cwd: Path) -> None:
    """The agents' and the starter's API configuration is the run's model
    identity: an alias's output cap that is no nonnegative integer is refused,
    as the run's model resolution refuses it, not dropped."""
    (tmp_cwd / "models.ini").write_text(
        "[capped]\nprovider = vllm\nmodel = test-model\nmax_output_tokens = many\n", encoding="utf-8")
    (tmp_cwd / "config.ini").write_text("[api]\nmodel = capped\n", encoding="utf-8")
    with config.use_run_settings(resolved()):
        with pytest.raises(ConfigurationError, match="max_output_tokens"):
            models.get_api_config()


def test_api_key_env_fallback_when_models_ini_has_none(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Without a models.ini key, ``TESTUARY_<PROVIDER>_API_KEY`` is used."""
    monkeypatch.setenv("TESTUARY_OLLAMA_API_KEY", "sk-from-env")
    (tmp_cwd / "config.ini").write_text(
        "[api]\nmodel = ollama_test\nprovider = ollama\n\n"
        "[ollama]\napi_url = https://ollama.example/v1\n",
        encoding="utf-8",
    )
    (tmp_cwd / "models.ini").write_text(
        "[ollama_test]\nprovider = ollama\nmodel = test-model:cloud\n",
        encoding="utf-8",
    )
    with config.use_run_settings(resolved()):
        assert models.get_api_config()["api_key"] == "sk-from-env"


def test_the_working_dirs_files_are_no_layer(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A config.ini or models.ini in the working directory is not read when the
    user directory is another."""
    from testuary import settings

    elsewhere = tmp_cwd / "user"
    elsewhere.mkdir()
    monkeypatch.setattr(settings, "platform_user_dir", lambda: elsewhere)
    (tmp_cwd / "config.ini").write_text("[coverage]\nmax_requests_per_phase = 99\n", encoding="utf-8")
    (tmp_cwd / "models.ini").write_text("[my_model]\nmodel = m\n", encoding="utf-8")
    with config.use_run_settings(resolved()):
        assert load_config("coverage")["max_requests_per_phase"] != "99"
        assert config_module.run_settings().models == {}


# ---------------------------------------------------------------------------
# get_agent_config allow-list: a key it does not list is silently dropped,
# and the phase falls back to its code default.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "agent_name",
    ["test_coverage", "test_mutation"],
)
def test_get_agent_config_exposes_token_caps(agent_name: str, tmp_cwd: Path) -> None:
    """Both agent loops' sections surface the per-run token caps, 0
    (unlimited) by default. [starter], with no agent loop, takes none."""
    cfg = agent(agent_name)
    assert "max_input_tokens_per_phase" in cfg, (
        f"{agent_name}: get_agent_config leaves out max_input_tokens_per_phase."
    )
    assert "max_output_tokens_per_phase" in cfg
    assert cfg["max_input_tokens_per_phase"] == 0
    assert cfg["max_output_tokens_per_phase"] == 0


def test_get_agent_config_surfaces_min_improvement_points(tmp_cwd: Path) -> None:
    """min_improvement_points surfaces as 0.5 in both loops, so a gain below
    half a point counts as stagnation."""
    cov = agent("test_coverage")
    assert "min_improvement_points" in cov, (
        "get_agent_config leaves out min_improvement_points."
    )
    assert cov["min_improvement_points"] == 0.5
    mut = agent("test_mutation")
    assert "min_improvement_points" in mut
    assert mut["min_improvement_points"] == 0.5


def test_get_agent_config_surfaces_cosmic_ray_distributor(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cosmic_ray_distributor surfaces, ``auto`` by default, and takes an
    override."""
    assert agent("test_mutation").get("cosmic_ray_distributor") == "auto"
    monkeypatch.setenv("TESTUARY_MUTATION_COSMIC_RAY_DISTRIBUTOR", "http")
    assert agent("test_mutation").get("cosmic_ray_distributor") == "http"


def test_get_agent_config_surfaces_reasoning_grace(tmp_cwd: Path) -> None:
    """reasoning_grace_seconds surfaces as a float, 0 (off) by default. The
    reasoning-token cap is per model (models.ini), not a phase key."""
    cfg = agent("test_initial")
    assert "thinking_token_budget" not in cfg, (
        "thinking_token_budget is a setting of each model (models.ini), not a "
        "phase key."
    )
    assert cfg["reasoning_grace_seconds"] == 0.0
    assert isinstance(cfg["reasoning_grace_seconds"], float)


def test_get_agent_config_surfaces_attempts_per_suite(tmp_cwd: Path) -> None:
    """attempts_per_suite surfaces as an int, 4 by default: the first wave plus
    three refills per starter-suite slot."""
    cfg = agent("test_initial")
    assert "attempts_per_suite" in cfg, (
        "get_agent_config leaves out attempts_per_suite, the attempt limit of "
        "each starter-suite slot."
    )
    assert cfg["attempts_per_suite"] == 4
    assert isinstance(cfg["attempts_per_suite"], int)


def test_get_agent_config_attempts_per_suite_env_override(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("TESTUARY_STARTER_ATTEMPTS_PER_SUITE", "5")
    assert agent("test_initial")["attempts_per_suite"] == 5


def test_get_agent_config_surfaces_the_floors(tmp_cwd: Path) -> None:
    """The coverage and mutation floors surface with their shipped values."""
    cov = agent("test_coverage")
    assert cov["minimum"] == 75.0
    mut = agent("test_mutation", parent="test_coverage")
    assert mut["minimum"] == 65.0
    assert cov["minimum"] < cov["goal"]
    assert mut["minimum"] < mut["goal"]
    # A missed floor still completes the phase by default.
    assert cov["complete_tests_below_minimum"] is True


def test_get_agent_config_caps_ship_at_ten(tmp_cwd: Path) -> None:
    """Submission spam and consecutive suite timeouts both end a phase at 10."""
    cfg = agent("test_coverage")
    assert cfg["max_final_answers_without_measurement"] == 10
    assert cfg["suite_timeout_stop_after"] == 10
    mut = agent("test_mutation", parent="test_coverage")
    assert mut["max_final_answers_without_measurement"] == 10
    assert mut["suite_timeout_stop_after"] == 10


def test_get_agent_config_surfaces_reject_unsound_async_green(tmp_cwd: Path) -> None:
    """reject_unsound_async_green surfaces, True by default, in both loops."""
    for section in ("test_coverage", "test_mutation"):
        assert agent(section).get("reject_unsound_async_green") is True, section


def test_get_agent_config_surfaces_the_suite_reduction_solver_settings(tmp_cwd: Path) -> None:
    """The suite reduction's solver settings surface with their shipped
    values in both loops, and consolidation measures mutation by both
    methods."""
    for section in ("test_coverage", "test_mutation"):
        cfg = agent(section, parent="test_coverage" if section == "test_mutation" else None)
        assert cfg["reduction_solver"] == "exact"
        assert cfg["reduction_solver_time_limit_seconds"] == 120.0
        assert cfg["reduction_solver_random_seed"] == 1
    with config.use_run_settings(resolved()):
        assert config_module.consolidation_measure_mutation() is True
        assert config_module.consolidation_mutation_method() == "both"


def test_global_seed_is_the_single_default_for_every_seeded_op(tmp_cwd: Path) -> None:
    """``[main] seed`` defaults to 0 and is the generation base unless
    overridden. CBC's solver seed is separate."""
    with config.use_run_settings(resolved()):
        assert config_module.global_seed() == 0
        assert config_module.get_agent_config("test_initial")["seed_base"] == 0
        assert config_module.get_agent_config("test_coverage")["reduction_solver_random_seed"] == 1


def test_global_seed_env_override_propagates(tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The global seed moves every consumer that does not override it."""
    monkeypatch.setenv("TESTUARY_MAIN_SEED", "7")
    with config.use_run_settings(resolved()):
        assert config_module.global_seed() == 7
        assert config_module.get_agent_config("test_initial")["seed_base"] == 7
        assert config_module.get_agent_config("test_coverage")["reduction_solver_random_seed"] == 1


def test_get_agent_config_surfaces_the_keys_the_phases_read(tmp_cwd: Path) -> None:
    """Keys the phases read surface with their settings values, such as
    goal_metric ``branch`` over the code default ``both``."""
    cov = agent("test_coverage")
    assert cov.get("goal_metric") == "branch"
    # The uncovered-source limit is [limits] uncovered_chars.
    assert "coverage_max_uncovered_chars" not in cov
    assert "max_final_answers_without_measurement" in cov
    assert "cosmic_ray_progress_interval_s" in agent("test_mutation")


def test_get_agent_config_surfaces_mutation_full_after_selective(tmp_cwd: Path) -> None:
    """The full-after-selective switch keys surface, typed, with the bundled
    defaults."""
    mut = agent("test_mutation", parent="test_coverage")
    assert mut["full_after_selective_goal"] is False
    assert mut["selective_switch_threshold"] == 85.0


def test_get_agent_config_full_after_selective_honors_config(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An enabled switch reaches the phase, with the string coerced to a bool."""
    monkeypatch.setenv("TESTUARY_MUTATION_FULL_AFTER_SELECTIVE_GOAL", "true")
    monkeypatch.setenv("TESTUARY_MUTATION_SELECTIVE_SWITCH_THRESHOLD", "95")
    monkeypatch.setenv("TESTUARY_MUTATION_LOOP_CATALOGUE", "selective")
    mut = agent("test_mutation", parent="test_coverage")
    assert mut["full_after_selective_goal"] is True
    assert mut["selective_switch_threshold"] == 95.0


def test_get_agent_config_token_caps_honor_env_override(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Apply TESTUARY_<SECTION>_<KEY> overrides to the configured token caps."""
    monkeypatch.setenv("TESTUARY_COVERAGE_MAX_INPUT_TOKENS_PER_PHASE", "500000")
    monkeypatch.setenv("TESTUARY_COVERAGE_MAX_OUTPUT_TOKENS_PER_PHASE", "100000")
    cfg = agent("test_coverage")
    assert cfg["max_input_tokens_per_phase"] == 500000
    assert cfg["max_output_tokens_per_phase"] == 100000


def test_get_agent_config_exposes_and_coerces_reuse_suites(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """reuse_suites surfaces as a bool, False by default, and the string
    'false' coerces to False."""
    assert agent("test_initial")["reuse_suites"] is False
    monkeypatch.setenv("TESTUARY_STARTER_REUSE_SUITES", "1")
    assert agent("test_initial")["reuse_suites"] is True
    monkeypatch.setenv("TESTUARY_STARTER_REUSE_SUITES", "false")
    assert agent("test_initial")["reuse_suites"] is False


def test_get_agent_config_cover_attribution_coercion(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """cover_attribution is 'full' by default, and an override sets 'first'."""
    assert agent("test_mutation")["cover_attribution"] == "full"
    monkeypatch.setenv("TESTUARY_MUTATION_COVER_ATTRIBUTION", "first")
    assert agent("test_mutation")["cover_attribution"] == "first"


def test_get_agent_config_loop_catalogue_coercion(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """loop_catalogue resolves to 'full' (default) or 'selective'."""
    assert agent("test_mutation")["loop_catalogue"] == "full"
    monkeypatch.setenv("TESTUARY_MUTATION_LOOP_CATALOGUE", "full")
    assert agent("test_mutation")["loop_catalogue"] == "full"


def test_full_loop_catalogue_disables_the_two_phase_switch(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A full-catalogue loop disables the two-phase switch even when set."""
    monkeypatch.setenv("TESTUARY_MUTATION_FULL_AFTER_SELECTIVE_GOAL", "true")
    monkeypatch.setenv("TESTUARY_MUTATION_LOOP_CATALOGUE", "full")
    cfg = agent("test_mutation")
    assert cfg["loop_catalogue"] == "full"
    assert cfg["full_after_selective_goal"] is False
    # With a selective loop the configured value stands.
    monkeypatch.setenv("TESTUARY_MUTATION_LOOP_CATALOGUE", "selective")
    assert agent("test_mutation")["full_after_selective_goal"] is True


def test_phase_end_evidence_reuse_defaults_on_and_the_mutation_section_inherits(
    tmp_cwd: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Evidence reuse is on by default, the coverage section turns it off for
    both phases, and the mutation section alone for the mutation phase."""
    def reuse():
        return (agent("test_coverage")["phase_end_evidence_reuse"],
                agent("test_mutation", parent="test_coverage")["phase_end_evidence_reuse"])

    assert reuse() == (True, True)
    monkeypatch.setenv("TESTUARY_COVERAGE_PHASE_END_EVIDENCE_REUSE", "false")
    assert reuse() == (False, False)
    monkeypatch.delenv("TESTUARY_COVERAGE_PHASE_END_EVIDENCE_REUSE")
    monkeypatch.setenv("TESTUARY_MUTATION_PHASE_END_EVIDENCE_REUSE", "false")
    assert reuse() == (True, False)


def test_the_context_rebuild_policy_resolves_and_is_recorded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The policy is ``mode_change`` by default and the variable wins over the
    file. The settings refuse an unknown value, naming the choices. The
    resolved-preset record carries the policy."""
    with config.use_run_settings(resolved()):
        assert config_module.context_rebuild() == "mode_change"
        record = config_module.resolved_test_preset()["sections"]["main"]
    assert record["context_rebuild"] == "mode_change"
    assert "context_compact_on_tool_change" not in record
    rebuild = tmp_path / "rebuild.ini"
    rebuild.write_text("[main]\ncontext_rebuild = every_turn\n", encoding="utf-8")
    monkeypatch.setenv("TESTUARY_CONFIG_FILE", str(rebuild))
    with config.use_run_settings(resolved()):
        assert config_module.context_rebuild() == "every_turn"
    monkeypatch.setenv("TESTUARY_MAIN_CONTEXT_REBUILD", "MODE_CHANGE")
    with config.use_run_settings(resolved()):
        assert config_module.context_rebuild() == "mode_change"
    monkeypatch.setenv("TESTUARY_MAIN_CONTEXT_REBUILD", "never")
    with pytest.raises(ValueError, match=r"context_rebuild: choose one of mode_change, every_turn"):
        resolved()


def test_the_built_in_guidance_is_on_by_default(tmp_cwd: Path) -> None:
    """The package asks for scoped built-in replacement in both agent phases."""
    assert agent("test_coverage")["builtin_replacement_guidance"] is True
    assert agent("test_mutation", parent="test_coverage")["builtin_replacement_guidance"] is True


def test_the_tie_break_default_and_unknown_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every reduction phase resolves the default and an unknown value is
    refused."""
    for section, parent in (("test_initial", None), ("test_coverage", None),
                            ("test_mutation", "test_coverage")):
        assert agent(section, parent=parent)["reduction_tie_break"] == "name_order"
    unknown = tmp_path / "unknown.ini"
    unknown.write_text("[coverage]\nreduction_tie_break = first\n", encoding="utf-8")
    monkeypatch.setenv("TESTUARY_CONFIG_FILE", str(unknown))
    with pytest.raises(ValueError, match="reduction_tie_break"):
        agent("test_coverage")


def test_explicit_overlay_can_disable_phase_end_evidence_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    preset = tmp_path / "reuse.ini"
    preset.write_text("[coverage]\nphase_end_evidence_reuse = false\n", encoding="utf-8")
    monkeypatch.setenv("TESTUARY_CONFIG_FILE", str(preset))
    with config.use_run_settings(resolved()):
        assert config_module.get_agent_config("test_coverage")["phase_end_evidence_reuse"] is False
        assert config_module.get_agent_config(
            "test_mutation", parent="test_coverage")["phase_end_evidence_reuse"] is False
        assert config_module.resolved_test_preset()["sections"]["coverage"][
            "phase_end_evidence_reuse"] == "false"


def test_phase_ids_and_agent_names_name_their_sections(tmp_cwd: Path) -> None:
    """One mapping names each phase's config.ini section, another its agent;
    load_config takes the phase id or the agent name for it."""
    from testuary.settings import AGENT_NAMES, PHASE_SECTIONS, _defaults, phase_section

    defaults = _defaults()
    with config.use_run_settings(resolved()):
        for phase, section in PHASE_SECTIONS.items():
            assert section in defaults
            assert phase_section(phase) == section
            assert load_config(phase) == load_config(section)
        for phase, agent_name in AGENT_NAMES.items():
            assert phase_section(agent_name) == PHASE_SECTIONS[phase]
            assert load_config(agent_name) == load_config(phase)
    assert phase_section("main") == "main"


# max_measurements_without_improvement as a share of the phase's request limit.
#
# A share such as 10% resolves to ceil(share x max_requests_per_phase) measurements,
# computed exactly. It is refused while the phase has no request limit. A whole
# number keeps its meaning.
def _resolved_maximum(phase="coverage", **fields):
    settings = resolve_settings(config_files=[], models={}, environment={},
                                overrides={PHASE_SECTIONS[phase]: fields})
    with config.use_run_settings(settings):
        parent = "test_coverage" if phase == "mutation" else None
        return config.get_agent_config(f"test_{phase}", parent=parent)["max_measurements_without_improvement"]


@pytest.mark.parametrize("share,limit,expected", [
    ("10%", 1000, 100),
    ("10%", 30, 3),        # exact: 0.1 x 30 in floats rounds up to 4
    ("10%", 25, 3),        # 2.5 rounds up
    ("12.5%", 8, 1),
    ("100%", 7, 7),
    ("0%", 50, 0),         # zero disables the stop, as 0 does
])
def test_a_share_resolves_to_the_rounded_up_part_of_the_request_limit(share, limit, expected):
    assert _resolved_maximum(max_measurements_without_improvement=share, max_requests_per_phase=limit) == expected


def test_the_mutation_loop_takes_its_own_request_limit():
    assert _resolved_maximum("mutation", max_measurements_without_improvement="20%", max_requests_per_phase=40) == 8


def test_a_whole_number_keeps_its_meaning():
    assert _resolved_maximum(max_measurements_without_improvement=10, max_requests_per_phase=1000) == 10
    assert _resolved_maximum(max_measurements_without_improvement=10, max_requests_per_phase="unlimited") == 10


def test_a_share_without_a_request_limit_is_refused():
    with pytest.raises(ConfigurationError, match=r"\[coverage\] max_measurements_without_improvement = 10% "
                                                 r"is a share of max_requests_per_phase, but this phase "
                                                 r"has no request limit"):
        _resolved_maximum(max_measurements_without_improvement="10%", max_requests_per_phase="unlimited")


@pytest.mark.parametrize("text", ["150%", "-5%", "ten%", "%", "1e1%"])
def test_a_share_outside_zero_to_hundred_percent_is_refused(text):
    with pytest.raises(ValueError, match="max_measurements_without_improvement: expected a whole number, or a "
                                         "share of max_requests_per_phase from 0% to 100%"):
        _resolved_maximum(max_measurements_without_improvement=text, max_requests_per_phase=100)
