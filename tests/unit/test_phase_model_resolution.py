"""Public model controls, precedence and protocol resolution."""
from dataclasses import FrozenInstanceError
import json

import pytest

from testuary.models import resolve_phase_model
from testuary.settings import ConfigurationError, RunSettings, resolve_settings


@pytest.fixture
def make_settings():
    def make(*, sections=None, models=None, environment=None, files=(), root=None):
        # A test's own alias fields name no provider unless the test is about it.
        catalogue = {"openai_example": {"provider": "openai", "model": "chosen"},
                     **{name: ({"provider": "openai", **fields}
                               if isinstance(fields, dict) and "provider" not in fields else fields)
                        for name, fields in (models or {}).items()}}
        return resolve_settings(
            config_files=files, base_dir=root, models=catalogue,
            environment=environment or {}, overrides={"api": {"model": "openai_example"}, **(sections or {})},
        )

    return make


@pytest.mark.parametrize("phase, temperature, timeout", [
    ("starter", None, 600), ("coverage", None, 3600), ("mutation", None, 3600),
])
def test_unchanged_defaults_and_slot_seed_contract(make_settings, phase, temperature, timeout):
    result = resolve_phase_model(make_settings(), phase)
    assert result.values["temperature"] == temperature
    assert result.values["top_p"] is None
    assert result.values["seed"] is None
    assert result.values["request_timeout_seconds"] == timeout
    assert result.values["reasoning"] is None
    assert result.reasoning_body == result.request_options == {}
    assert result.protocol == "openai-chat"


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_alias_overrides_packaged_temperature_and_sampling(make_settings, phase):
    result = resolve_phase_model(make_settings(models={"openai_example": {"provider": "openai", 
        "temperature": .6, "top_p": 0, "max_output_tokens": 32768,
    }}), phase)
    assert result.values["temperature"] == .6
    assert result.values["top_p"] == 0
    assert result.values["max_output_tokens"] == 32768
    assert result.sources["temperature"] == "alias:[openai_example]"


def test_provider_configuration_then_catalogue_then_alias(make_settings):
    result = resolve_phase_model(make_settings(
        sections={"openai": {"reasoning_effort": "low"}},
        models={"openai": {"reasoning_effort": "medium", "reasoning_dialect": "openai"},
                "openai_example": {"provider": "openai", "reasoning_effort": "high"}},
    ), "coverage")
    assert result.values["reasoning_effort"] == "high"
    assert result.reasoning_body == {"reasoning_effort": "high"}


def test_layer_before_section_and_mutation_inheritance(make_settings, tmp_path):
    early = tmp_path / "early.ini"
    late = tmp_path / "late.ini"
    early.write_text("[mutation]\ntemperature = .1\n[coverage]\ntemperature = .2\n")
    late.write_text("[api]\ntemperature = .3\n")
    result = make_settings(files=(early, late), environment={"TESTUARY_API_TEMPERATURE": ".4"})
    for phase in ("starter", "coverage", "mutation"):
        request = resolve_phase_model(result, phase)
        assert request.values["temperature"] == .4
        assert request.sources["temperature"] == "environment:[api]"


def test_missing_api_sections_do_not_reorder_layers(make_settings, tmp_path):
    early = tmp_path / "early.ini"
    late = tmp_path / "late.ini"
    early.write_text("[mutation]\ntemperature = .1\n")
    late.write_text("[coverage]\ntemperature = .2\n")
    result = resolve_phase_model(make_settings(files=(early, late)), "mutation")
    assert result.values["temperature"] == .2
    assert result.sources["temperature"] == "config[1]:[coverage]"


def test_within_one_layer_api_then_coverage_then_mutation(make_settings):
    result = make_settings(sections={
        "api": {"model": "openai_example", "temperature": ".3"},
        "coverage": {"temperature": ".2"},
        "mutation": {"temperature": ".1"},
    })
    assert [resolve_phase_model(result, phase).values["temperature"]
            for phase in ("starter", "coverage", "mutation")] == [.3, .2, .1]


@pytest.mark.parametrize("raw, expected", [("", None), ("0", 0), (".2", .2)])
def test_present_empty_clears_while_zero_is_sent(make_settings, raw, expected):
    result = resolve_phase_model(make_settings(
        models={"openai_example": {"provider": "openai", "temperature": .6}},
        sections={"starter": {"temperature": raw}},
    ), "starter")
    assert result.values["temperature"] == expected
    assert result.sources["temperature"] == "overrides:[starter]"


@pytest.mark.parametrize("model_cap, phase_cap, expected", [
    (32768, 8192, 8192), (4096, 8192, 4096), (32768, 0, 32768),
    (0, 8192, 8192), ("", "", None),
])
def test_model_and_phase_caps_are_independent(make_settings, model_cap, phase_cap, expected):
    result = resolve_phase_model(make_settings(
        models={"openai_example": {"provider": "openai", "max_output_tokens": model_cap}},
        sections={"coverage": {"max_output_tokens": phase_cap}},
    ), "coverage")
    assert result.values["max_output_tokens"] == expected
    assert result.sources["model_max_output_tokens"] == "alias:[openai_example]"
    assert result.sources["phase_max_output_tokens"] == "overrides:[coverage]"


def test_later_no_cap_replaces_only_its_own_cap_group(make_settings, tmp_path):
    early = tmp_path / "early.ini"
    late = tmp_path / "late.ini"
    early.write_text("[coverage]\nmax_output_tokens = 100\n")
    late.write_text("[api]\nmax_output_tokens = 0\n")
    result = resolve_phase_model(make_settings(files=(early, late), models={
        "openai": {"max_output_tokens": 50},
        "openai_example": {"provider": "openai", "max_output_tokens": 200},
    }), "mutation")
    assert result.values["phase_max_output_tokens"] is None
    assert result.values["model_max_output_tokens"] == result.values["max_output_tokens"] == 200


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
def test_agent_seed_can_be_explicit_zero(make_settings, phase):
    result = resolve_phase_model(make_settings(sections={"coverage": {"seed": "0"}}), phase)
    assert result.values["seed"] == 0


def test_api_seed_is_agent_only_not_a_starter_slot_override(make_settings):
    settings = make_settings(sections={"api": {"model": "openai_example", "seed": "12"}})
    assert resolve_phase_model(settings, "starter").values["seed"] is None
    assert resolve_phase_model(settings, "coverage").values["seed"] == 12


@pytest.mark.parametrize("fields", [{"seed": 0}, {"seed": ""}])
def test_catalogue_does_not_define_agent_or_slot_seeds(make_settings, fields):
    with pytest.raises(ConfigurationError, match="a setting of the coverage and mutation phases"):
        resolve_phase_model(make_settings(models={"openai_example": fields}), "starter")


@pytest.mark.parametrize("endpoint, expected", [
    ("https://ollama.com/v1", "ollama-native"),
    ("https://OLLAMA.COM:8443/gateway/v1", "ollama-native"),
    ("https://ollama.com.evil/v1", "openai-chat"),
    ("https://other.example/ollama.com/v1", "openai-chat"),
    ("http://localhost:11434/v1", "openai-chat"),
])
def test_starter_protocol_uses_only_the_exact_host(make_settings, endpoint, expected):
    settings = make_settings(models={"openai_example": {"provider": "openai", "api_url": endpoint}})
    assert resolve_phase_model(settings, "starter").protocol == expected
    assert resolve_phase_model(settings, "coverage").protocol == "openai-chat"


def test_starter_native_can_be_explicit_on_local_ollama(make_settings):
    result = resolve_phase_model(make_settings(
        models={"openai_example": {"provider": "openai", "api_url": "http://localhost:11434/v1"}},
        sections={"starter": {"protocol": "ollama-native"}},
    ), "starter")
    assert result.protocol == "ollama-native"
    assert result.sources["protocol"] == "overrides:[starter]"


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
def test_agent_native_requests_are_refused(make_settings, phase):
    with pytest.raises(ConfigurationError, match="only for the starter phase"):
        resolve_phase_model(make_settings(sections={"coverage": {"protocol": "ollama-native"}}), phase)


@pytest.mark.parametrize("dialect, fields, expected", [
    ("deepseek", {"reasoning": "on"}, {"thinking": {"type": "enabled"}}),
    ("deepseek", {"reasoning": "off"}, {"thinking": {"type": "disabled"}}),
    ("vllm", {"reasoning_template": "qwen", "reasoning": "on", "reasoning_budget_tokens": 4096},
     {"chat_template_kwargs": {"enable_thinking": True}, "thinking_token_budget": 4096}),
    ("vllm", {"reasoning_template": "qwen", "reasoning": "off"},
     {"chat_template_kwargs": {"enable_thinking": False}}),
    ("vllm", {"reasoning_template": "deepseek", "reasoning": "off"},
     {"chat_template_kwargs": {"thinking": False}}),
    ("vllm", {"reasoning_template": "deepseek", "reasoning_effort": "high"},
     {"chat_template_kwargs": {"reasoning_effort": "high"}}),
    ("vllm", {"reasoning_budget_tokens": 4096}, {"thinking_token_budget": 4096}),
    ("openrouter", {"reasoning": "on", "reasoning_effort": "high"},
     {"reasoning": {"enabled": True, "effort": "high"}}),
    ("openrouter", {"reasoning_budget_tokens": 4096}, {"reasoning": {"max_tokens": 4096}}),
    ("openai", {"reasoning_effort": "high"}, {"reasoning_effort": "high"}),
])
@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_reasoning_dialect_mapping(make_settings, dialect, fields, expected, phase):
    result = resolve_phase_model(make_settings(models={"openai_example": {"provider": "openai", 
        "reasoning_dialect": dialect, **fields,
    }}), phase)
    assert result.reasoning_body == expected


@pytest.mark.parametrize("protocol, fields, expected", [
    ("ollama-native", {"reasoning": "on"}, {"think": True}),
    ("ollama-native", {"reasoning": "off"}, {"think": False}),
    ("ollama-native", {"reasoning_effort": "high"}, {"think": "high"}),
    ("openai-responses", {"reasoning_effort": "high"}, {"reasoning": {"effort": "high"}}),
])
def test_nonchat_reasoning_mapping(make_settings, protocol, fields, expected):
    result = resolve_phase_model(make_settings(models={"openai_example": {"provider": "openai", "protocol": protocol, **fields}}), "starter")
    assert result.reasoning_body == expected


@pytest.mark.parametrize("fields, message", [
    ({"reasoning": "yes"}, "reasoning must be"),
    ({"reasoning": "on"}, "need a reasoning_dialect"),
    ({"reasoning_dialect": "unknown"}, "reasoning_dialect must be"),
    ({"reasoning": "off", "reasoning_effort": "high"}, "cannot be combined"),
    ({"reasoning": "off", "reasoning_budget_tokens": 10}, "cannot be combined"),
    ({"reasoning_dialect": "deepseek", "reasoning_effort": "high"}, "enablement only"),
    ({"reasoning_dialect": "vllm", "reasoning_template": "qwen", "reasoning_effort": "high"},
     "qwen chat template has no reasoning_effort"),
    ({"reasoning_dialect": "vllm", "reasoning": "on"}, "needs reasoning_template on the alias, one of qwen, deepseek, deepseek_v3, deepseek_v4_1"),
    ({"reasoning_dialect": "vllm", "reasoning_template": "granite"}, "must be empty or one of qwen, deepseek, deepseek_v3, deepseek_v4_1"),
    ({"reasoning_dialect": "deepseek", "reasoning_template": "qwen"}, "vllm reasoning_dialect only"),
    ({"reasoning_dialect": "openai", "reasoning": "on"}, "effort only"),
    ({"reasoning_dialect": "openrouter", "reasoning_effort": "high", "reasoning_budget_tokens": 10}, "cannot combine"),
    ({"protocol": "ollama-native", "reasoning_budget_tokens": 10}, "cannot express"),
    ({"protocol": "openai-responses", "reasoning": "on"}, "effort only"),
    ({"protocol": "openai-responses", "reasoning_budget_tokens": 10}, "effort only"),
    ({"chat_model": "example:Chat", "reasoning_effort": "high"}, "use request_options"),
    ({"chat_model": "invalid"}, "item:attribute"),
    ({"chat_model": "example:Chat", "protocol": "openai-chat"}, "vice versa"),
    ({"protocol": "added"}, "vice versa"),
    ({"protocol": "invalid"}, "protocol must be"),
    ({"reasoning_budget_tokens": 0}, "reasoning = off"),
    ({"reasoning_budget_tokens": -1}, "positive"),
    ({"temperature": float("nan")}, "finite"),
    ({"temperature": -.1}, "nonnegative"),
    ({"top_p": 1.1}, "between zero and one"),
    ({"request_timeout_seconds": 0}, "positive"),
    ({"request_timeout_seconds": ""}, "positive"),
])
def test_unexpressible_or_invalid_fields_refuse_without_the_value(make_settings, fields, message):
    with pytest.raises(ConfigurationError, match=message):
        resolve_phase_model(make_settings(models={"openai_example": fields}), "starter")


def test_options_merge_and_null_delete_across_actual_captured_layers(make_settings, tmp_path):
    path = tmp_path / "sampling.ini"
    path.write_text('[coverage]\nrequest_options = {"top_k": null, "custom": {"b": 2}}\n')
    result = resolve_phase_model(make_settings(
        files=(path,), models={
            "openai": {"request_options": {"top_k": 20, "custom": {"a": 1}}},
            "openai_example": {"provider": "openai", "request_options": {"other": True}},
        }, environment={"TESTUARY_API_REQUEST_OPTIONS": '{"min_p": 0}'},
    ), "mutation")
    assert result.request_options == {"custom": {"b": 2}, "other": True, "min_p": 0}
    assert result.sources["request_options"] == {
        "custom": "config[0]:[coverage]", "other": "alias:[openai_example]", "min_p": "environment:[api]",
    }


@pytest.mark.parametrize("dialect, controlled", [
    ("deepseek", "thinking"), ("vllm", "chat_template_kwargs"),
    ("vllm", "thinking_token_budget"), ("openrouter", "reasoning"),
])
def test_options_cannot_override_dialect_controls_even_when_unset(make_settings, dialect, controlled):
    with pytest.raises(ConfigurationError, match="controlled field"):
        resolve_phase_model(make_settings(models={"openai_example": {"provider": "openai", 
            "reasoning_dialect": dialect, "request_options": {controlled: None},
        }}), "coverage")


def test_added_protocol_selects_factory_and_preserves_json_options(make_settings):
    result = resolve_phase_model(make_settings(models={"openai_example": {"provider": "openai", 
        "chat_model": "example:Chat", "request_options": {"reasoning": {"enabled": True}},
    }}), "coverage")
    assert result.protocol == "added"
    assert result.chat_model == "example:Chat"
    assert result.request_options == {"reasoning": {"enabled": True}}


def test_prices_are_information_not_request_fields(make_settings):
    result = resolve_phase_model(make_settings(models={"openai_example": {"provider": "openai", 
        "price_input_cache_hit": ".1", "price_input_cache_miss": ".2", "price_output": "1",
    }}), "coverage")
    assert result.information == {"price_input_cache_hit": ".1", "price_input_cache_miss": ".2", "price_output": "1"}
    assert not set(result.information).intersection(result.values)


def test_resolution_is_frozen_and_safe_to_record_without_rereads(make_settings, monkeypatch):
    settings = make_settings(models={"openai_example": {"provider": "openai", 
        "api_url": "https://user:secret@host.example/v1?key=private",
        "api_key": "credential", "request_options": {"custom": [1, {"nested": True}]},
    }})
    from testuary import settings as settings_module
    monkeypatch.setattr(settings_module, "_read_ini_file", lambda _: pytest.fail("reread"))
    monkeypatch.setenv("TESTUARY_API_MODEL", "changed")
    result = resolve_phase_model(settings, "coverage")
    text = json.dumps(result.export()) + repr(result)
    for secret in ("secret", "private", "credential", "user:"):
        assert secret not in text
    with pytest.raises(FrozenInstanceError):
        result.phase = "starter"
    with pytest.raises(TypeError):
        result.request_options["custom"][1]["nested"] = False
    exported = result.export()
    exported["request_options"]["custom"][1]["nested"] = False
    assert result.request_options["custom"][1]["nested"] is True


def test_canonical_phase_fields_are_accepted_by_the_real_public_schema():
    settings = RunSettings(sections={"coverage": {"temperature": .2}})
    assert settings.sections["coverage"]["temperature"] == "0.2"


def test_phase_names_are_not_silently_normalized(make_settings):
    with pytest.raises(ConfigurationError, match="model phase"):
        resolve_phase_model(make_settings(), "unknown")


@pytest.mark.parametrize("model, expected", [
    ("deepseek-ai/DeepSeek-V4.1-Flash", "deepseek_v4_1"),
    ("deepseek-ai/DeepSeek-V4-Flash", "deepseek"),
    ("deepseek-ai/DeepSeek-V4-Pro", "deepseek"),
    ("deepseek-ai/DeepSeek-V3.1-Terminus", "deepseek_v3"),
    ("deepseek-ai/DeepSeek-V3.2-Exp", "deepseek_v3"),
    ("Qwen/Qwen3-8B", "qwen"),
    ("Qwen/Qwen3-30B-A3B-FP8", "qwen"),
    # Templates that do not read the keys: they always reason, or never.
    ("Qwen/QwQ-32B", None),
    ("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B", None),
    ("deepseek-ai/DeepSeek-R1", None),
    ("deepseek-ai/DeepSeek-V3", None),
    ("Qwen/Qwen2.5-72B-Instruct", None),
    ("Qwen/Qwen3-235B-A22B-Instruct-2507", None),
    ("Qwen/Qwen3-Coder-30B-A3B-Instruct", None),
    ("meta-llama/Llama-3.3-70B-Instruct", None),
])
def test_a_vllm_template_is_inferred_only_where_the_published_template_reads_its_keys(
        make_settings, model, expected):
    """A served model whose published chat template reads a family's keys
    gets them, and the record names the template. Any other model's
    enablement is refused before a request, naming reasoning_template, since
    its template would ignore the value the record states."""
    fields = {"provider": "openai", "model": model, "reasoning_dialect": "vllm", "reasoning": "off"}
    settings = make_settings(models={"openai_example": fields})
    if expected is None:
        with pytest.raises(ConfigurationError, match="needs reasoning_template"):
            resolve_phase_model(settings, "coverage")
        return
    result = resolve_phase_model(settings, "coverage")
    key = {"qwen": "enable_thinking", "deepseek": "thinking", "deepseek_v3": "thinking",
           "deepseek_v4_1": "enable_thinking"}[expected]
    assert result.reasoning_body == {"chat_template_kwargs": {key: False}}
    assert result.export()["reasoning_template"] == expected


@pytest.mark.parametrize("model, effort, sent", [
    ("deepseek-ai/DeepSeek-V4.1-Flash", "low", "low"),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "max", "max"),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "40", 40),
    ("deepseek-ai/DeepSeek-V4-Pro", "max", "max"),
])
def test_an_effort_reaches_the_templates_that_read_it(make_settings, model, effort, sent):
    fields = {"provider": "openai", "model": model, "reasoning_dialect": "vllm",
              "reasoning": "on", "reasoning_effort": effort}
    result = resolve_phase_model(make_settings(models={"openai_example": fields}), "coverage")
    key = "enable_thinking" if "4.1" in model else "thinking"
    assert result.reasoning_body == {"chat_template_kwargs": {key: True, "reasoning_effort": sent}}


@pytest.mark.parametrize("model, effort, refusal", [
    # The V3.1 and V3.2-Exp templates read thinking only.
    ("deepseek-ai/DeepSeek-V3.1", "low", "deepseek_v3 chat template has no reasoning_effort"),
    ("deepseek-ai/DeepSeek-V3.2-Exp", "high", "deepseek_v3 chat template has no reasoning_effort"),
    # The V4.1 template rejects any other effort.
    ("deepseek-ai/DeepSeek-V4.1-Flash", "extreme", "takes reasoning_effort low, medium, high, max or 1 to 100"),
    ("deepseek-ai/DeepSeek-V4.1-Flash", "101", "takes reasoning_effort"),
    # V4's encoding accepts high or max only.
    ("deepseek-ai/DeepSeek-V4-Flash", "low", "deepseek chat template takes reasoning_effort high, max"),
    ("deepseek-ai/DeepSeek-V4-Pro", "medium", "deepseek chat template takes reasoning_effort high, max"),
])
def test_an_effort_the_template_does_not_read_is_refused_before_a_request(
        make_settings, model, effort, refusal):
    fields = {"provider": "openai", "model": model, "reasoning_dialect": "vllm",
              "reasoning": "on", "reasoning_effort": effort}
    with pytest.raises(ConfigurationError, match=refusal):
        resolve_phase_model(make_settings(models={"openai_example": fields}), "coverage")


def test_v3_templates_still_switch_reasoning_on_and_off(make_settings):
    for reasoning in ("on", "off"):
        fields = {"provider": "openai", "model": "deepseek-ai/DeepSeek-V3.2-Exp",
                  "reasoning_dialect": "vllm", "reasoning": reasoning}
        result = resolve_phase_model(make_settings(models={"openai_example": fields}), "coverage")
        assert result.reasoning_body == {"chat_template_kwargs": {"thinking": reasoning == "on"}}


def test_a_declared_template_serves_a_model_the_name_does_not_tell(make_settings):
    fields = {"provider": "openai", "model": "Qwen/QwQ-32B", "reasoning_dialect": "vllm",
              "reasoning": "on", "reasoning_template": "qwen"}
    result = resolve_phase_model(make_settings(models={"openai_example": fields}), "coverage")
    assert result.reasoning_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_the_template_may_be_set_for_a_whole_provider_section(make_settings):
    """As reasoning_dialect may: a [vllm] section in config.ini or other
    settings serves every alias of that provider."""
    result = resolve_phase_model(make_settings(
        sections={"api": {"model": "served"},
                  "vllm": {"reasoning_dialect": "vllm", "reasoning_template": "qwen"}},
        models={"served": {"provider": "vllm", "model": "Qwen/QwQ-32B", "reasoning": "off",
                           "api_url": "http://localhost:8000/v1"}}), "coverage")
    assert result.reasoning_body == {"chat_template_kwargs": {"enable_thinking": False}}
    assert result.export()["reasoning_template"] == "qwen"


def test_the_qwen_template_reads_no_effort(make_settings):
    fields = {"provider": "openai", "model": "Qwen/Qwen3-8B", "reasoning_dialect": "vllm",
              "reasoning": "on", "reasoning_effort": "low"}
    with pytest.raises(ConfigurationError, match="qwen chat template has no reasoning_effort"):
        resolve_phase_model(make_settings(models={"openai_example": fields}), "coverage")


def test_template_keys_in_request_options_without_the_vllm_dialect_still_reach_the_request(make_settings):
    """Template keys set through request_options reach the request: without
    the vllm dialect they are not reserved."""
    result = resolve_phase_model(make_settings(models={"openai_example": {
        "provider": "openai", "request_options": '{"chat_template_kwargs": {"thinking": true}}'}}), "coverage")
    assert result.request_options == {"chat_template_kwargs": {"thinking": True}}
    assert result.reasoning_body == {} and result.export()["reasoning_template"] is None
