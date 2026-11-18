"""Resolved settings through public execution and actual phase consumers."""
from __future__ import annotations

import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import FrozenInstanceError, replace
from types import SimpleNamespace

import pytest

from testuary import orchestrator
from testuary import config
from testuary import hints
from testuary import models as models_module
from testuary import project as project_module
from testuary import settings as settings_module
from testuary.settings import ConfigurationError, RunSettings, resolve_settings
from phase_values import set_phase_value
from python_runtime import project_python


def test_ordered_partial_files_environment_and_explicit_fields(tmp_cwd):
    first = tmp_cwd / "first.ini"
    second = tmp_cwd / "second.ini"
    first.write_text("[main]\nseed=4\n[starter]\nsuite_count=2\n", encoding="utf-8")
    second.write_text("[main]\nseed=5\n[starter]\nsuite_count=3\n", encoding="utf-8")
    settings = resolve_settings(
        config_files=[first, second], models={},
        environment={"TESTUARY_MAIN_SEED": "6"},
        overrides={"main": {"seed": 0, "phase_end_mutation": False, "hints_file": ""}},
    )
    assert settings.main.seed == 0
    assert settings.main.phase_end_mutation is False
    assert settings.main.hints_file == ""
    assert settings.sections["starter"]["suite_count"] == "3"
    assert settings.sections["starter"]["suite_run_timeout_seconds"] == "600"


def test_disabled_environment_does_not_refuse_ambient_retired_settings(tmp_cwd, monkeypatch):
    monkeypatch.setenv("TESTUARY_MAIN_SEED", "invalid")
    monkeypatch.setenv("TESTUARY_STARTER_N_SETS", "invalid")
    settings = resolve_settings(use_environment=False, models={})
    assert settings.main.seed == 0
    # A launch environment is still captured, without Testuary's own inputs.
    assert settings.environment_captured
    assert not any(name.startswith("TESTUARY_") for name in settings.environment)


@pytest.mark.parametrize("sections, message", [
    ({"covreage": {}}, "coverage"),
    ({"coverage": {"max_requsets_per_phase": "2"}}, "max_requests_per_phase"),
    ({"starter": {"concurrent_requests": 0}}, "positive"),
    ({"coverage": {"phase_end_evidence_reuse": "tru"}}, "true or false"),
    ({"coverage": {"max_requests_per_phase": "1.5"}}, "positive integer, or unlimited"),
    ({"coverage": {"max_requests_per_phase": "0"}}, "positive integer, or unlimited"),
    ({"starter": {"temperature": "nan"}}, "expected a number"),
    ({"starter": {"suite_count": None}}, "Omit the field"),
    ({"main": {"reduced_suite_mutation": "typo"}}, "choose one"),
    ({"mutation": {"per_mutant_timeout_seconds": 0}}, "positive integer"),
])
def test_invalid_settings_fail_before_execution(sections, message):
    with pytest.raises(ValueError, match=message):
        RunSettings(sections=sections)


def test_mutation_takes_its_own_goal_keys_and_refuses_coverage_only_ones(tmp_cwd):
    """[mutation]'s goal, minimum, restore_count and extra_measurements_after_goal share
    [coverage]'s names but are its own; goal_metric stays coverage's."""
    path = tmp_cwd / "goals.ini"
    path.write_text("[coverage]\ngoal=91\n[mutation]\ngoal=70\nminimum=60\nrestore_count=2\n"
                    "extra_measurements_after_goal=1\n", encoding="utf-8")
    settings = resolve_settings(config_files=[path], models={}, environment={})
    assert float(settings.sections["coverage"]["goal"]) == 91
    mutation = settings.sections["mutation"]
    assert [float(mutation[k]) for k in ("goal", "minimum", "restore_count", "extra_measurements_after_goal")] == [70, 60, 2, 1]
    path.write_text("[mutation]\ngoal_metric=line\n", encoding="utf-8")
    with pytest.raises(ValueError, match="goal_metric"):
        resolve_settings(config_files=[path], models={}, environment={})


def test_explicit_missing_file_is_not_treated_as_default(tmp_cwd):
    with pytest.raises(ValueError, match="absent"):
        resolve_settings(config_files=["missing.ini"], models={})


def test_malformed_ini_diagnostic_does_not_disclose_its_line(tmp_cwd):
    path = tmp_cwd / "broken.ini"
    path.write_text("a-secret-in-a-malformed-line", encoding="utf-8")
    with pytest.raises(ValueError) as caught:
        resolve_settings(config_files=[path], models={})
    assert "a-secret" not in str(caught.value)


def test_settings_are_copied_frozen_and_secret_safe():
    source = {"main": {"seed": 12}}
    models = {"openai_example": {"provider": "openai", 
        "api_key": "do-not-export", "model": "example",
        "api_url": "https://username:password@example.test/v1?key=query-secret",
    }}
    settings = RunSettings(sections=source, models=models)
    source["main"]["seed"] = 99
    models["openai_example"]["api_key"] = "changed"
    assert settings.main.seed == 12
    with pytest.raises(TypeError):
        settings.sections["main"]["seed"] = "0"
    with pytest.raises(FrozenInstanceError):
        settings.main.seed = 0
    exported = json.dumps(settings.export())
    for secret in ("do-not-export", "password", "query-secret", "username"):
        assert secret not in exported
        assert secret not in repr(settings)


def test_resolved_object_does_not_read_ambient_sources_during_execution(tmp_cwd, monkeypatch):
    settings = RunSettings(
        sections={"api": {"model": "openai_example"}, "main": {"seed": 7}},
        models={"openai_example": {"provider": "openai", "model": "chosen", "api_key": "explicit-key"}},
    )
    (tmp_cwd / "config.ini").write_text("[main]\nseed=500\n", encoding="utf-8")
    monkeypatch.setenv("TESTUARY_MAIN_SEED", "999")
    monkeypatch.setenv("TESTUARY_OPENAI_API_KEY", "ambient-key")
    monkeypatch.setattr(settings_module, "_read_ini_file", lambda _: pytest.fail("execution reread a file"))
    with config.use_run_settings(settings):
        assert config.global_seed() == 7
        assert models_module.get_api_config()["model"] == "chosen"
        assert models_module.get_api_config()["api_key"] == "explicit-key"
        with ThreadPoolExecutor(max_workers=1) as pool:
            assert pool.submit(config.global_seed).result() == 7


def test_explicit_fields_win_over_environment_bypasses(tmp_cwd):
    from testuary import starters
    from testuary import measurement

    settings = resolve_settings(
        models={},
        environment={"TESTUARY_MAIN_PHASE_END_MUTATION": "true",
                     "TESTUARY_MAIN_PROJECT_EXEC": "container",
                     "TESTUARY_STARTER_CONCURRENT_REQUESTS": "8",
                     "TESTUARY_STARTER_CONCURRENT_REQUESTS_OLLAMA": "8",
                     "TESTUARY_FINISHING_PHASE_SCORES": "full"},
        overrides={"main": {"phase_end_mutation": False, "project_exec": "local"},
                   "api": {"model": "example", "provider": "openai"},
                   "starter": {"concurrent_requests": 2, "concurrent_requests_ollama": 2},
                   "finishing": {"phase_scores": "off"}},
    )
    with config.use_run_settings(settings):
        assert config.project_exec_mode() == "local"
        assert config.consolidation_measure_mutation() is False
        assert starters._resolve_concurrent_requests(config.get_agent_config("test_initial")) == 2
        assert measurement.phase_scores_mode() == "off"
        assert config.resolved_test_preset()["sections"]["finishing"]["phase_scores"] == "off"


def test_editing_agents_and_phase_end_read_the_current_settings():
    from testuary.agent import coverage_agent, mutation_agent
    from testuary import starters
    from testuary import measurement

    settings = RunSettings(sections={
        "coverage": {"max_requests_per_phase": 17},
        "mutation": {"max_requests_per_phase": 19, "cosmic_ray_worker_ready_timeout_s": 43,
                          "cosmic_ray_verbose": True},
    })
    with config.use_run_settings(settings):
        assert coverage_agent._AGENT_CONFIG["max_requests_per_phase"] == 17
        assert mutation_agent._AGENT_CONFIG["max_requests_per_phase"] == 19
        agent = dict(mutation_agent._AGENT_CONFIG)
        phase_end = measurement.mutation_config()
        starter_end = starters._starter_mutation_config()
        assert agent["suite_run_timeout_seconds"] == int(phase_end["suite_run_timeout_seconds"])
        assert agent == starter_end
        assert phase_end["cosmic_ray_worker_ready_timeout_s"] == "43"
        assert str(phase_end["cosmic_ray_verbose"]).lower() == "true"


def test_pipeline_sequential_runs_restore_state_even_after_failure(monkeypatch):
    calls = []

    def execute(args, languages):
        calls.append((config.global_seed(), models_module.get_api_config()["model"], args))
        config.set_invocation_source_dir("temporary-work-tree")
        args.targets = "mutated"
        if config.global_seed() == 1:
            raise RuntimeError("simulated execution failure")
        return 0

    monkeypatch.setattr(orchestrator, "_run_pipeline", execute)
    args = SimpleNamespace(targets="original")
    with pytest.raises(RuntimeError, match="simulated"):
        orchestrator.run_pipeline(args, settings=RunSettings(
            sections={"main": {"seed": 1}, "api": {"model": "first", "provider": "openai"}}))
    assert orchestrator.run_pipeline(args, settings=RunSettings(
        sections={"main": {"seed": 2}, "api": {"model": "second", "provider": "openai"}})) == 0
    assert [item[:2] for item in calls] == [(1, "first"), (2, "second")]
    assert args.targets == "original"
    assert config.active_run() is None


def test_pipeline_rejects_ambiguous_resolved_and_cli_settings():
    with pytest.raises(ValueError, match="cannot be combined"):
        orchestrator.run_pipeline(SimpleNamespace(model="other"), settings=RunSettings())


def test_runtime_value_error_is_not_misclassified_as_configuration(monkeypatch):
    def execute(args, languages):
        raise ValueError("execution evidence is invalid")
    monkeypatch.setattr(orchestrator, "_run_pipeline", execute)
    with pytest.raises(ValueError, match="execution evidence") as caught:
        orchestrator.run_pipeline(SimpleNamespace(), settings=RunSettings())
    assert not isinstance(caught.value, ConfigurationError)
    assert config.active_settings() is None


def test_run_guard_rejects_overlap_but_worker_reads_are_supported():
    settings = RunSettings(sections={"main": {"seed": 11}})
    with config.use_run_settings(settings):
        with ThreadPoolExecutor(max_workers=1) as pool:
            def overlap():
                with config.use_run_settings(settings):
                    return 0
            with pytest.raises(RuntimeError, match="already active"):
                pool.submit(overlap).result()
            assert pool.submit(config.global_seed).result() == 11
    with config.use_run_settings(settings):
        assert config.global_seed() == 11


def test_phase_context_settings_are_the_active_run_settings(monkeypatch, tmp_path):
    from testuary import target_phases as phases

    settings = RunSettings(sections={"main": {"seed": 18}})
    ctx = phases.TargetPhasesContext(
        log_print=lambda *a, **k: None,
        project_name="example", workspace="task", top_file={},
        canonical_path=tmp_path / "source.py", work_tests=tmp_path,
        results_dir=tmp_path, target_logs_dir=tmp_path, reuse=phases.ReuseFlags(), settings=settings,
    )
    def execute(context):
        assert context.settings is settings
        assert config.global_seed() == 18
        return phases.TestPipelineResult()
    monkeypatch.setattr(phases, "_run_target_phases", execute)
    with config.use_run_settings(settings):
        phases.run_target_phases(ctx)
        with pytest.raises(ValueError, match="differ"):
            phases.run_target_phases(replace(ctx, settings=RunSettings()))


def test_cli_config_is_repeatable_and_does_not_fill_absent_fields():
    from testuary.__main__ import _build_parser

    args = _build_parser().parse_args(["--config", "one.ini", "--config", "two.ini"])
    assert args.config == ["one.ini", "two.ini"]
    assert not orchestrator._cli_settings_overrides(args)
    assert args.reuse_starter_suites is None


def test_cli_failure_records_diagnostics_without_replacing_process_exception_hook(monkeypatch):
    from testuary import __main__ as cli

    recorded = []
    previous = sys.excepthook
    def fail():
        raise RuntimeError("execution failed")
    monkeypatch.setattr(cli, "_main", fail)
    monkeypatch.setattr(cli, "log_print", lambda message, **kwargs: recorded.append(message))
    with pytest.raises(RuntimeError, match="execution failed"):
        cli.main()
    assert "execution failed" in recorded[0]
    assert sys.excepthook is previous


def test_nested_catalogue_values_are_detached_and_deeply_frozen():
    catalogue = {"example": {"options": [{"nested": {"values": [1, 2]}}]}}
    settings = RunSettings(models=catalogue)
    catalogue["example"]["options"][0]["nested"]["values"].append(3)
    values = settings.models["example"]["options"][0]["nested"]["values"]
    assert values == (1, 2)
    with pytest.raises(TypeError):
        settings.models["example"]["options"][0]["nested"]["values"] = ()


def test_export_redacts_credentials_inside_nested_catalogue_sequences():
    settings = RunSettings(models={"example": {"options": [[{
        "api_key": "private", "api_url": "https://name:secret@example.test/v1?key=hidden",
        "headers": {"Authorization": "private"}, "allowed": [1, {"secret": "private", "value": 0}],
    }]]}})
    exported = settings.export()
    options = exported["models"]["example"]["options"][0][0]
    assert options == {"api_url": "https://example.test/v1", "allowed": [1, {"value": 0}]}
    assert "private" not in repr(settings)
    options["allowed"][1]["value"] = 5
    assert settings.models["example"]["options"][0][0]["allowed"][1]["value"] == 0


@pytest.mark.parametrize("catalogue", [{"example": 1}, {1: {}}, []])
def test_malformed_catalogue_structure_is_refused_early(catalogue):
    with pytest.raises(ValueError, match="models must map names"):
        RunSettings(models=catalogue)


def test_secret_safe_export_of_malformed_endpoint_does_not_raise():
    settings = RunSettings(models={"example": {
        "api_url": "https://name:private@example.test:secret/v1",
    }})
    assert settings.export()["models"]["example"]["api_url"] == "[invalid endpoint]"
    assert "private" not in repr(settings)
    assert "secret" not in repr(settings)


def test_timing_settings_are_live_raw_section_values(monkeypatch):
    from testuary import measurement

    with config.use_run_settings(RunSettings(sections={"timing": {
        "measure": True, "run_count": 3,
    }})):
        assert measurement._bool_flag(measurement._TIMING_CONFIG, "measure") is True
        assert int(measurement._TIMING_CONFIG["run_count"]) == 3
    with config.use_run_settings(RunSettings(sections={"timing": {
        "measure": False, "run_count": 2,
    }})):
        assert measurement._bool_flag(measurement._TIMING_CONFIG, "measure") is False
        assert int(measurement._TIMING_CONFIG["run_count"]) == 2


def test_the_resolved_settings_record_only_the_runs_provider_section():
    """resolved_settings.json records the provider section the run's model
    goes through and no other provider's, whose address can be private. A
    run whose model a model source supplies goes through none of them."""
    settings = RunSettings(environment={}, sections={
        "api": {"model": "explicit"},
        "ollama": {"api_url": "http://private-ollama.test:11434"},
        "vllm": {"api_url": "http://private-vllm.test:8000/v1"},
    }, models={"explicit": {"provider": "openai", "model": "chosen", "api_url": "https://example.test/v1"}})
    with config.use_run_settings(settings):
        record = config.resolved_test_preset(model_phases=("coverage",))
        with models_module.installed(object()):
            sourced = config.resolved_test_preset(model_phases=("coverage",))
    assert set(record["sections"]) & set(settings_module.PROVIDER_SECTIONS) == {"openai"}
    assert "private" not in json.dumps(record)
    assert not set(sourced["sections"]) & set(settings_module.PROVIDER_SECTIONS)
    # The record still holds every other section.
    assert {"api", "main", "starter", "coverage", "mutation"} <= set(record["sections"])


def test_setup_failure_restores_state_and_releases_the_run_guard(monkeypatch):
    from testuary.agent import coverage_agent

    class BrokenLogger:
        logger = "original"
        log_dir_override = "original-directory"

        @property
        def log_filename_override(self):
            raise RuntimeError("logger setup failed")

    logger = BrokenLogger()
    settings = RunSettings()
    with monkeypatch.context() as patch:
        patch.setattr(coverage_agent, "_log", logger)
        with pytest.raises(RuntimeError, match="logger setup failed"):
            with config.use_run_settings(settings):
                pytest.fail("setup unexpectedly completed")
        assert logger.logger == "original"
        assert logger.log_dir_override == "original-directory"
        assert config.active_settings() is None
    with config.use_run_settings(settings):
        assert config.active_settings() is settings


def test_keyboard_interrupt_restores_loggers_and_configuration():
    from testuary.agent import coverage_agent

    previous = coverage_agent._log._config
    settings = RunSettings(sections={"main": {"seed": 37}})
    with pytest.raises(KeyboardInterrupt):
        with config.use_run_settings(settings):
            coverage_agent._log._config = {"seed": 37}
            raise KeyboardInterrupt
    assert coverage_agent._log._config is previous
    assert config.active_settings() is None
    with config.use_run_settings(settings):
        assert config.global_seed() == 37


def test_captured_base_survives_a_different_working_directory(tmp_path, monkeypatch):
    base = tmp_path / "base"
    other = tmp_path / "other"
    (base / "projects" / "example").mkdir(parents=True)
    other.mkdir()
    (base / "hints.json").write_text('{"coverage": {"general": "from base"}}', encoding="utf-8")
    (other / "hints.json").write_text('{"coverage": {"general": "ambient"}}', encoding="utf-8")
    settings = RunSettings(base_dir=base, sections={"project": {"name": "example", "output_dir": "out"}})
    monkeypatch.chdir(other)
    with config.use_run_settings(settings):
        assert project_module.output_dir() == base / "out"
        assert hints.load_hints()["coverage"]["general"] == "from base"
        assert project_module.resolve_project_source_path("projects/example") == base / "projects" / "example"
        # The label names the project only. Its root is the project directory.
        assert project_module.resolve_source_dir("unknown.py") == str(base.resolve())


def test_explicit_empty_hints_disables_ambient_hint_file(tmp_cwd):
    (tmp_cwd / "hints.json").write_text('{"coverage": {"all": "ambient"}}', encoding="utf-8")
    with config.use_run_settings(RunSettings(sections={"main": {"hints_file": ""}})):
        assert hints.hints_filename() == ""
        assert hints.load_hints() == {}


def _java() -> str:
    """The java executable a run chooses for the project (``project_runtime``)."""
    from testuary.languages import language_setting_values

    project = config.get_project_config()
    values = language_setting_values("java", project, config.active_settings())
    return project_module.project_runtime(
        "java", values, root=str(project.get("source_dir") or "")).interpreter.path


def _java_tools() -> dict:
    """The Java tools' variables a run's context adds."""
    from langadapter.java.adapter import JavaAdapter

    from testuary.languages import language_setting_values

    return JavaAdapter.context_environment(language_setting_values("java"))


def test_project_settings_expose_java_toolchain_and_runtime_metadata(tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    base = tmp_path / "base"
    java = base / "jdk" / "bin" / ("java.exe" if os.name == "nt" else "java")
    java.parent.mkdir(parents=True)
    java.write_text("", encoding="utf-8")
    settings = RunSettings(base_dir=base, sections={"project": {
        "name": "sample", "java_home": "jdk", "build_java_home": "build-jdk",
        "maven": "tools/mvn", "runtime_version": "Java 21"}})
    monkeypatch.chdir(other)
    monkeypatch.setenv("JAVA_HOME", "ambient-jdk")
    with config.use_run_settings(settings):
        project = config.get_project_config()
        assert project["java_home"] == str(base / "jdk")
        assert project["runtime_version"] == "Java 21"
        assert _java() == str(java)
        assert _java_tools() == {
            "LANGADAPTER_BUILD_JAVA_HOME": str(base / "build-jdk"),
            "LANGADAPTER_MAVEN": str(base / "tools" / "mvn"),
        }


def test_container_paths_remain_posix_without_host_existence_checks(tmp_path):
    settings = RunSettings(base_dir=tmp_path, sections={
        "main": {"project_exec": "container"},
        "project": {"name": "sample", "java_home": "/opt/jdk", "python_executable": "/opt/venv/bin/python"},
    }, environment={"TESTUARY_PROJECT_SOURCE_DIR": "/repo", "TESTUARY_PROJECT_CONTAINER": "sample"})
    with config.use_run_settings(settings):
        project = config.get_project_config()
        assert project["python_executable"] == "/opt/venv/bin/python"
        assert project["java_home"] == "/opt/jdk"
        assert project_python() == "/opt/venv/bin/python"
        assert _java() == "/opt/jdk/bin/java"
        assert project_module.resolve_source_dir("/repo/src/module.py") == "/repo"


@pytest.mark.parametrize("source", ["project", "environment"])
@pytest.mark.parametrize("value, expected", [
    ("/opt/jdk", "/opt/jdk/bin/java"),
    ("/opt/jdk/bin/java", "/opt/jdk/bin/java"),
    ("//opt//jdk/./", "/opt/jdk/bin/java"),
    ("/opt/project jdk/bin/./java", "/opt/project jdk/bin/java"),
])
def test_container_java_uses_guest_paths_without_host_probes(
    tmp_path, monkeypatch, source, value, expected,
):
    from pathlib import Path

    settings = RunSettings(base_dir=tmp_path, sections={
        "main": {"project_exec": "container"},
        "project": {"java_home": value if source == "project" else ""},
    }, environment={"JAVA_HOME": value if source == "environment" else "/ignored/jdk"})
    monkeypatch.setenv("JAVA_HOME", "/ambient/changed/jdk")

    def refuse_probe(*args, **kwargs):
        pytest.fail("guest Java selection must not inspect or resolve a host path")

    with config.use_run_settings(settings):
        _java_tools()  # reads Testuary's own packaged configuration, not a host path
        with monkeypatch.context() as guard:
            guard.setattr(Path, "is_file", refuse_probe)
            guard.setattr(Path, "exists", refuse_probe)
            guard.setattr(project_module, "resolve_host_path", refuse_probe)
            if source == "project":
                assert _java() == expected
            else:
                # The host's JAVA_HOME names no JDK in the container.
                with pytest.raises(RuntimeError, match="no JDK is selected for the project in "
                                                       "the container"):
                    _java()


@pytest.mark.parametrize("value", [
    "jdk", "../jdk", "C:/jdk", r"C:\jdk", "/opt/../jdk", r"/opt\jdk",
])
def test_container_java_refuses_non_guest_paths(tmp_path, value):
    with config.use_run_settings(RunSettings(base_dir=tmp_path, sections={
        "main": {"project_exec": "container"},
        "project": {"java_home": value},
    })):
        with pytest.raises(RuntimeError, match="absolute POSIX.*parent traversal"):
            _java()


@pytest.mark.parametrize("explicit", [False, True])
def test_container_java_resolves_environment_and_explicit_override_order(
    tmp_path, monkeypatch, explicit,
):
    environment = {
        "TESTUARY_PROJECT_JAVA_HOME": "/captured/override/jdk",
        "JAVA_HOME": "/captured/fallback/jdk",
    }
    project = {"name": "sample"}
    if explicit:
        project["java_home"] = "/explicit/bin/java"
    settings = resolve_settings(
        config_files=(), base_dir=tmp_path, models={},
        environment=environment,
        overrides={"main": {"project_exec": "container"}, "project": project},
    )
    environment["TESTUARY_PROJECT_JAVA_HOME"] = "/changed/input/jdk"
    monkeypatch.setenv("TESTUARY_PROJECT_JAVA_HOME", "/ambient/override/jdk")
    monkeypatch.setenv("JAVA_HOME", "/ambient/fallback/jdk")
    with config.use_run_settings(settings):
        expected = "/explicit/bin/java" if explicit else "/captured/override/jdk/bin/java"
        assert _java() == expected


@pytest.mark.parametrize("use_environment", [False, True])
def test_container_java_does_not_adopt_late_ambient_home(
    tmp_path, monkeypatch, use_environment,
):
    settings = resolve_settings(
        config_files=(), base_dir=tmp_path, models={},
        environment={} if use_environment else {
            "JAVA_HOME": "/captured/jdk", "TESTUARY_PROJECT_JAVA_HOME": "/ignored/override/jdk"},
        use_environment=use_environment,
        overrides={"main": {"project_exec": "container"}},
    )
    monkeypatch.setenv("JAVA_HOME", "/ambient/jdk")
    monkeypatch.setenv("TESTUARY_PROJECT_JAVA_HOME", "/ambient/override/jdk")
    with config.use_run_settings(settings):
        # Neither the captured platform JAVA_HOME (a host path), nor a
        # disabled Testuary input, nor a value set after capture selects
        # the container's JDK.
        with pytest.raises(RuntimeError, match="no JDK is selected for the project in the "
                                               "container"):
            _java()


@pytest.mark.parametrize("explicit_executable", [False, True])
def test_local_java_still_resolves_existing_home_or_executable(
    tmp_path, explicit_executable,
):
    java = tmp_path / "jdk" / "bin" / ("java.exe" if os.name == "nt" else "java")
    java.parent.mkdir(parents=True)
    java.touch()
    selected = java if explicit_executable else java.parent.parent
    with config.use_run_settings(RunSettings(base_dir=tmp_path, sections={
        "project": {"java_home": str(selected)},
    })):
        assert _java() == str(java)


def test_local_java_still_refuses_missing_executable(tmp_path):
    with config.use_run_settings(RunSettings(base_dir=tmp_path, sections={
        "project": {"java_home": "missing-jdk"},
    })):
        with pytest.raises(RuntimeError, match="names no JDK.*does not exist"):
            _java()


def test_captured_execution_environment_is_not_reloaded(tmp_cwd, monkeypatch):
    settings = resolve_settings(models={}, environment={
        "PATH": "captured-tools", "LANGADAPTER_BUILD_JAVA_HOME": "captured-jdk",
        "TESTUARY_PROJECT_CONTAINER_RUNNER": "apptainer",
    })
    monkeypatch.setenv("PATH", "changed-tools")
    monkeypatch.setenv("LANGADAPTER_BUILD_JAVA_HOME", "changed-jdk")
    with config.use_run_settings(settings):
        assert config.environment_value("PATH") == "captured-tools"
        assert config.environment_value("LANGADAPTER_BUILD_JAVA_HOME") == "captured-jdk"
        assert config.environment_value("TESTUARY_PROJECT_CONTAINER_RUNNER") == "apptainer"
    assert "captured-jdk" not in repr(settings)


@pytest.mark.parametrize("use_environment", (True, False))
def test_supervisor_selector_uses_only_enabled_resolver_snapshot(tmp_path, monkeypatch, use_environment):
    name = "LANGADAPTER_HELPER_PYTHON"
    environment = {name: "/control/venv/bin/python"}
    settings = resolve_settings(
        config_files=(), base_dir=tmp_path, models={},
        environment=environment, use_environment=use_environment,
    )
    environment[name] = "/changed/input/python"
    monkeypatch.setenv(name, "/ambient/python")
    with config.use_run_settings(settings):
        expected = "/control/venv/bin/python" if use_environment else None
        assert config.environment_value(name) == expected
        assert config.configured_environment().get(name) == expected
    assert "/control/venv/bin/python" not in repr(settings)


def test_direct_settings_omit_supervisor_without_ambient_adoption(tmp_path, monkeypatch):
    name = "LANGADAPTER_HELPER_PYTHON"
    monkeypatch.setenv(name, "/ambient/python")
    with config.use_run_settings(RunSettings(base_dir=tmp_path)):
        assert config.environment_value(name) is None
        assert name not in config.configured_environment()


def test_host_paths_expand_only_captured_variables_and_work_dir(tmp_path, monkeypatch):
    base = tmp_path / "base"
    source = base / "source"
    source.mkdir(parents=True)
    other = tmp_path / "other"
    other.mkdir()
    settings = RunSettings(base_dir=base, sections={"project": {
        "requirements": "$INPUTS/deps.txt",
    }}, environment={"INPUTS": "source", "TESTUARY_PROJECT_WORK_DIR": "work"})
    monkeypatch.setenv("INPUTS", str(other))
    monkeypatch.chdir(other)
    with config.use_run_settings(settings):
        assert config.get_project_config()["requirements"] == str(source / "deps.txt")
        assert project_module.resolve_project_source_path("$INPUTS") == source
        assert project_module.project_work_dir() == str(base / "work")


def test_invocation_interpreter_cache_does_not_cross_bases(tmp_path):
    for name in ("first", "second"):
        base = tmp_path / name
        executable = base / "tools" / "python"
        executable.parent.mkdir(parents=True)
        executable.write_text("", encoding="utf-8")
        with config.use_run_settings(RunSettings(base_dir=base, sections={
            "project": {"python_executable": "tools/python"},
        })):
            assert project_python() == str(executable)


def test_selected_virtual_environment_keeps_its_executable_path(tmp_path, monkeypatch):
    import subprocess
    import venv
    from pathlib import Path

    base = tmp_path / "base"
    environment = base / "selected-venv"
    venv.EnvBuilder(with_pip=False, symlinks=os.name != "nt").create(environment)
    relative = "selected-venv/Scripts/python.exe" if os.name == "nt" else "selected-venv/bin/python"
    executable = base / relative
    later = tmp_path / "later"
    later.mkdir()
    settings = RunSettings(base_dir=base, sections={"project": {"python_executable": relative}})
    monkeypatch.chdir(later)
    with config.use_run_settings(settings):
        configured = config.get_project_config()["python_executable"]
        selected = project_python()
        assert selected == configured == str(executable)
        result = subprocess.run(
            [selected, "-c", "import json,sys; print(json.dumps([sys.prefix,sys.base_prefix]))"],
            capture_output=True, check=True, text=True, timeout=30,
        )
    prefix, base_prefix = json.loads(result.stdout)
    assert Path(prefix) == environment
    assert prefix != base_prefix


@pytest.mark.parametrize("key", ["python_executable", "java_home", "build_java_home", "maven", "gradle"])
def test_selected_tool_links_are_not_canonicalized_like_source_dirs(tmp_path, key):
    base = tmp_path / "base"
    base.mkdir()
    actual = base / "actual"
    actual.write_text("fixture", encoding="utf-8")
    selected = base / "selected"
    try:
        selected.symlink_to(actual)
    except OSError as exc:
        pytest.skip(f"host cannot create a symlink: {exc}")
    settings = RunSettings(base_dir=base, sections={"project": {key: "./selected"}})
    with config.use_run_settings(settings):
        assert config.get_project_config()[key] == str(selected)


def test_a_phase_mapping_is_read_only_and_resolved_settings_stay_authoritative(monkeypatch, in_run):
    from testuary.agent import coverage_agent

    with pytest.raises(TypeError):
        coverage_agent._AGENT_CONFIG["max_requests_per_phase"] = 1
    in_run()
    set_phase_value(monkeypatch, coverage_agent._AGENT_CONFIG, "max_requests_per_phase", 999)
    assert coverage_agent._AGENT_CONFIG["max_requests_per_phase"] == 999
    in_run.leave()
    with config.use_run_settings(RunSettings(sections={"coverage": {"max_requests_per_phase": 5}})):
        assert coverage_agent._AGENT_CONFIG["max_requests_per_phase"] == 5


def test_sequential_starter_phases_use_resolved_settings_in_real_workers(tmp_path):
    from testuary import starters

    observed = []
    for seed_base in (31, 51):
        base = tmp_path / str(seed_base)
        project = base / "project"
        project.mkdir(parents=True)
        (project / "pyproject.toml").write_text("", encoding="utf-8")
        target = project / "sample.py"
        target.write_text("def add(x):\n    return x + 1\n", encoding="utf-8")
        # Native workers run the selected interpreter in the captured environment.
        settings = RunSettings(base_dir=base, sections={
            "main": {"seed": seed_base, "phase_end_mutation": False},
            "api": {"model": f"openai_{seed_base}"},
            "project": {"python_executable": sys.executable},
            "starter": {"suite_count": 2, "attempts_per_suite": 1,
                             "cache_warmup": False, "concurrent_requests": 2,
                             "concurrent_requests_ollama": 2,
                             "reduction_solver": "greedy"},
        }, models={f"openai_{seed_base}": {"provider": "openai", "model": f"model-{seed_base}"}},
            environment=dict(os.environ))
        overlap = threading.Barrier(2, timeout=10)

        def generate(prompt, *, temperature, seed, timeout, **kwargs):
            overlap.wait()
            observed.append((seed_base, config.global_seed(), models_module.get_api_config()["model"], seed))
            return ("```python\nfrom sample import add\n\ndef test_add():\n    assert add(1) == 2\n```",
                    {"input_tokens": 7, "output_tokens": 9})

        source = SimpleNamespace(generate_suite=generate)
        with config.use_run_settings(settings), models_module.installed(source):
            result = starters.run_starter_phase(
                language="python", target_file_path=str(target),
                source_dir=str(project), test_output_dir=str(base / "tests"),
                suites_dir=str(base / "suites"), log=lambda *args, **kwargs: None,
            )
        assert result.status == "ok"
        assert result.n_generated == result.n_green == 2
        assert result.requests_used == 2
        assert result.input_tokens == 14
        assert config.active_settings() is None
    assert sorted(observed) == [
        (31, 31, "model-31", 32), (31, 31, "model-31", 33),
        (51, 51, "model-51", 52), (51, 51, "model-51", 53),
    ]


def test_interrupted_starter_workers_quiesce_before_settings_are_restored(tmp_path):
    from testuary import starters

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    target = project / "sample.py"
    target.write_text("def add(x):\n    return x + 1\n", encoding="utf-8")
    settings = RunSettings(environment=dict(os.environ), base_dir=tmp_path, sections={
        "main": {"seed": 90, "phase_end_mutation": False},
        "api": {"model": "openai_interruption"},
        "starter": {"suite_count": 2, "attempts_per_suite": 1,
                         "cache_warmup": False, "concurrent_requests": 2,
                         "concurrent_requests_ollama": 2},
    }, models={"openai_interruption": {"provider": "openai", "model": "offline-interruption-model"}})
    overlap = threading.Barrier(2, timeout=10)
    finished = threading.Event()

    def generate(prompt, *, temperature, seed, timeout, **kwargs):
        overlap.wait()
        if seed == 91:
            raise KeyboardInterrupt
        time.sleep(0.05)
        assert config.active_settings() is settings
        finished.set()
        return ("```python\nfrom sample import add\n\ndef test_add():\n    assert add(1) == 2\n```", {})

    with pytest.raises(KeyboardInterrupt):
        with config.use_run_settings(settings), models_module.installed(
                SimpleNamespace(generate_suite=generate)):
            starters.run_starter_phase(
                language="python", target_file_path=str(target),
                source_dir=str(project), test_output_dir=str(tmp_path / "tests"),
                suites_dir=str(tmp_path / "suites"), log=lambda *args, **kwargs: None,
            )
    assert finished.is_set()
    assert config.active_settings() is None
    with config.use_run_settings(RunSettings(sections={"main": {"seed": 5}})):
        assert config.global_seed() == 5


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
def test_sequential_editing_entries_and_real_graphs_keep_their_settings(tmp_path, monkeypatch, phase):
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage
    from testuary import _chat_models
    from testuary.agent import coverage_agent, mutation_agent

    class FakeModel(FakeMessagesListChatModel):
        def bind_tools(self, tools, **kwargs):
            return self

    captured = []

    def build_model(**kwargs):
        captured.append(kwargs)
        return FakeModel(responses=[AIMessage(content="", tool_calls=[{
            "name": "read_tests", "args": {}, "id": "read",
        }])])

    monkeypatch.setattr(_chat_models, "build_chat_model", build_model)
    cls = coverage_agent.CoverageAgent if phase == "coverage" else mutation_agent.MutationAgent

    def finalize(agent, output, **kwargs):
        return SimpleNamespace(requests=kwargs["usage_tracker"].requests_used,
                               config=dict(agent.agent_config()), root=agent.deps.source_dir)

    monkeypatch.setattr(cls, "finalize_result", finalize)
    # Mutation measurement is stubbed. The entry, graph, provider construction
    # and request accounting are real.
    monkeypatch.setattr(mutation_agent, "_seed_mutation_baseline_from_matrix", lambda agent: True)
    for limit in (1, 2):
        base = tmp_path / str(limit)
        project = base / "project"
        project.mkdir(parents=True)
        (project / "pyproject.toml").write_text("", encoding="utf-8")
        target = project / "sample.py"
        target.write_text("def add(x):\n    return x + 1\n", encoding="utf-8")
        tests = base / "tests"
        tests.mkdir()
        source = "from sample import add\n\ndef test_add():\n    assert add(1) == 2\n"
        (tests / "test_sample.py").write_text(source, encoding="utf-8")
        settings = RunSettings(base_dir=base, sections={
            "api": {"model": f"openai_{limit}"},
            "coverage": {"max_requests_per_phase": limit, "max_loop_attempts": 0,
                              "goal": 100},
            "mutation": {"max_requests_per_phase": limit},
        }, models={f"openai_{limit}": {"provider": "openai", "model": f"model-{limit}",
                                       "api_url": "https://example.test/v1"}},
            environment=dict(os.environ))
        with config.use_run_settings(settings):
            driver = coverage_agent if phase == "coverage" else mutation_agent
            driver.init_logging("sample", str(base / "logs"), f"{phase}.jsonl")
            if phase == "coverage":
                result = coverage_agent.generate_coverage_tests(
                    "sample", top_file={"file_path": str(target)}, test_output_dir=str(tests),
                    baseline={"content": source, "line_coverage": 50.0, "branch_coverage": 50.0,
                              "test_count": 1, "suite_seconds": 0.01},
                )
            else:
                result = mutation_agent.generate_mutation_tests(
                    "sample", str(target), str(tests), preseed={"content": source},
                )
        assert result.requests == limit
        assert result.config["max_requests_per_phase"] == limit
        assert result.root == str(project)
        assert (base / "logs" / f"{phase}_chat.json").exists()
        assert config.active_settings() is None
    assert [item["model"] for item in captured] == ["model-1", "model-2"]
