"""Configured model provenance in the phase pipeline's writer of the
resolved settings."""
import configparser
from importlib import resources
from importlib import metadata
import json

import pytest

from test_orchestrator import _make_ctx
from testuary import __main__ as cli, orchestrator as public_run
from testuary import models as _model_protocols
from testuary import models as _model_source
from testuary import config
from testuary import target_phases as orchestrator
from testuary.settings import RunSettings, resolve_settings


def settings_for(tmp_path, **alias):
    return resolve_settings(base_dir=tmp_path, config_files=[],
        environment={"TESTUARY_COVERAGE_TEMPERATURE": ".25"}, overrides={
            "api": {"model": "explicit", "top_p": 0}, "main": {"hints_file": ""},
            "coverage": {"max_output_tokens": 64},
        }, models={
            "openai": {"reasoning_dialect": "vllm"},
            "explicit": {"provider": "openai", "model": "chosen",
                         "api_url": "https://user:private-url@example.test/v1?key=private-query",
                         "api_key": "private-key", "deployment": "service-a", "max_output_tokens": 128,
                         "reasoning": "on", "reasoning_budget_tokens": 13, "reasoning_template": "qwen",
                         "preserve_reasoning_content": True, "reasoning_field_names": "trace_blob",
                         "request_options": {"custom_enabled": False}, "price_output": ".3", **alias},
        })


def dump(tmp_path, settings, monkeypatch, **context):
    monkeypatch.setattr(_model_protocols, "build_resolved_chat_model", lambda *a, **kw: pytest.fail("client built"))
    ctx = _make_ctx(tmp_path, **{"reuse": orchestrator.ReuseFlags(),
                              "log_print": lambda *a, **kw: None, **context})
    with config.use_run_settings(settings):
        orchestrator._dump_resolved_preset(ctx)
    path = ctx.results_dir / ctx.workspace / "r0" / "resolved_settings.json"
    assert path.is_file()
    text = path.read_text("utf-8")
    assert all(secret not in text for secret in ("private-url", "private-query", "private-key", "user:"))
    return json.loads(text), path


def test_actual_settings_writer_records_all_phase_model_fields_and_sources(tmp_path, monkeypatch):
    data, _ = dump(tmp_path, settings_for(tmp_path), monkeypatch)
    assert set(data["model"]) == {"starter", "coverage", "mutation"}
    for phase, model in data["model"].items():
        assert {key: model[key] for key in ("source", "phase", "provider", "model_alias", "model", "protocol",
                                         "api_url", "deployment", "reasoning_dialect", "chat_model",
                                         "extension_version")} == {
            "source": "configuration", "phase": phase, "provider": "openai", "model_alias": "explicit",
            "model": "chosen", "protocol": "openai-chat", "api_url": "https://example.test/v1",
            "deployment": "service-a", "reasoning_dialect": "vllm", "chat_model": None, "extension_version": None,
        }
        assert model["sources"]["protocol"] == "default:phase/endpoint"
        assert not {"extension_file", "extension_sha256"} & set(model)
        assert model["values"]["temperature"] == (None if phase == "starter" else .25)
        if phase != "starter":
            assert model["sources"]["temperature"] == "environment:[coverage]"
        assert model["values"]["top_p"] == 0 and model["sources"]["top_p"] == "overrides:[api]"
        assert model["values"]["max_output_tokens"] == (128 if phase == "starter" else 64)
        assert model["values"]["preserve_reasoning_content"] is True
        assert model["values"]["reasoning_field_names"] == ["trace_blob"]
        assert model["values"]["reasoning"] == "on" and model["values"]["reasoning_budget_tokens"] == 13
        assert model["request_options"] == {"custom_enabled": False}
        assert model["sources"]["request_options"] == {"custom_enabled": "alias:[explicit]"}
        assert model["information"] == {"price_output": ".3"}
    assert config.active_settings() is None


@pytest.mark.parametrize("distributions,version,expected", [
    (["fixture-dist"], "1.2.3", "1.2.3"), ([], "1.2.3", None),
    (["one", "two"], "1.2.3", "1.2.3"), (["fixture-dist"], None, None),
])
def test_actual_settings_writer_records_added_factory_version_only_when_known(
    distributions, version, expected, tmp_path, monkeypatch, provider_module,
):
    """The version of the first distribution holding the module that has
    one, as LangAdapter's Extension gives it."""
    provider_module("fixture_client")
    monkeypatch.setattr(metadata, "packages_distributions", lambda: {"fixture_client": distributions})

    def installed_version(name):
        if version is None:
            raise metadata.PackageNotFoundError(name)
        return version

    monkeypatch.setattr(metadata, "version", installed_version)
    settings = settings_for(tmp_path, chat_model="fixture_client:factory", protocol="added",
                            reasoning="", reasoning_budget_tokens="", reasoning_dialect="",
                            reasoning_template="")
    data, _ = dump(tmp_path, settings, monkeypatch)
    for model in data["model"].values():
        assert model["chat_model"] == "fixture_client:factory" and model["protocol"] == "added"
        assert model["extension_version"] == expected


@pytest.mark.parametrize("reuse,stop,expected", [
    (orchestrator.ReuseFlags(), "starter", {"starter"}),
    (orchestrator.ReuseFlags(), "coverage", {"starter", "coverage"}),
    (orchestrator.ReuseFlags(starter=True), None, {"coverage", "mutation"}),
    (orchestrator.ReuseFlags(coverage=True), None, {"mutation"}),
    (orchestrator.ReuseFlags(mutation=True), None, {"starter", "coverage"}),
    (orchestrator.ReuseFlags(coverage=True, mutation=True), None, set()),
])
def test_settings_model_records_include_only_phases_that_can_generate(reuse, stop, expected, tmp_path, monkeypatch):
    data, _ = dump(tmp_path, settings_for(tmp_path), monkeypatch, reuse=reuse, stop_after_phase=stop)
    assert set(data.get("model", {})) == expected


def test_starter_only_settings_do_not_resolve_unsupported_unused_agent_protocol(tmp_path, monkeypatch):
    settings = settings_for(tmp_path, protocol="ollama-native", reasoning="", reasoning_budget_tokens="")
    data, _ = dump(tmp_path, settings, monkeypatch, stop_after_phase="starter")
    assert set(data["model"]) == {"starter"} and data["model"]["starter"]["protocol"] == "ollama-native"


def test_model_source_settings_record_does_not_claim_configured_settings_or_require_a_provider(tmp_path, monkeypatch):
    with _model_source.installed(object()):
        data, _ = dump(tmp_path, RunSettings(base_dir=tmp_path), monkeypatch)
    assert data["model"] == {phase: {"source": "model_source"} for phase in ("starter", "coverage", "mutation")}


def test_existing_settings_bytes_are_preserved_on_resume(tmp_path, monkeypatch):
    _, path = dump(tmp_path, settings_for(tmp_path), monkeypatch)
    previous = path.read_bytes()
    later = tmp_path / "later"
    later.mkdir()
    dump(later, settings_for(later, model="another"), monkeypatch, results_dir=path.parent)
    assert path.read_bytes() == previous


def test_the_test_preset_of_direct_settings_names_no_model(tmp_path):
    with config.use_run_settings(settings_for(tmp_path)):
        assert "model" not in config.resolved_test_preset()


@pytest.mark.parametrize("language", ("python", "java"))
def test_exact_shipped_demo_catalogue_passes_current_model_preflight_and_records_information(
    language, tmp_path, monkeypatch,
):
    directory = resources.files("testuary.demo").joinpath("data")
    if language == "java":
        directory = directory.joinpath("java")
    raw = directory.joinpath("models.ini").read_text(encoding="utf-8")
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(raw)
    models = {section: dict(parser[section]) for section in parser.sections()}
    selected = json.loads(directory.joinpath("manifest.json").read_text(encoding="utf-8"))["invocation"]["model"]
    assert selected in models
    # A recorded alias without a provider gets one when the demonstration
    # writes its workspace, as the run then sees it.
    models[selected].setdefault("provider", selected.partition("_")[0])
    settings = RunSettings(base_dir=tmp_path, sections={
        "api": {"model": selected}, "main": {"hints_file": ""},
    }, models=models, environment={"TESTUARY_OPENAI_API_KEY": "private-fixture"})
    monkeypatch.setattr(_model_protocols, "build_resolved_chat_model", lambda *a, **kw: pytest.fail("client built"))
    with config.use_run_settings(settings):
        assert public_run._preflight_model_configuration(cli._build_parser().parse_args([])) == ()
        resolved = config.resolved_test_preset(model_phases=("starter", "coverage", "mutation"))
    expected = {key: value for key, value in models[selected].items() if key.startswith("price_")}
    assert all(model["information"] == expected for model in resolved["model"].values())
    assert "private-fixture" not in json.dumps(resolved)
    assert directory.joinpath("models.ini").read_text(encoding="utf-8") == raw
    assert list(tmp_path.iterdir()) == []
