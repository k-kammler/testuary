"""Invocation-scoped hint input and pre-work configuration refusals."""
from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from testuary import orchestrator
from testuary import config as config_module
from testuary import hints as hints_module
from testuary import prompts
from testuary.settings import ConfigurationError, RunSettings, resolve_settings


@pytest.mark.parametrize("raw, diagnostic", [
    ("not-json", "valid JSON"), ("[]", "JSON object"), ("null", "JSON object"),
    ('{"covrage": {"general": "x"}}', "covrage"),
    ('{"coverage": {"genral": "x"}}', "genral"),
    ('{"coverage": {"general": ["x", 3]}}', "coverage.general"),
    ('{"coverage": {"general": null}}', "coverage.general"),
    ('{"coverage": {"per_profile": {"p": ["x", false]}}}', "per_profile"),
    ('{"coverage": {"per_language": {"python": ["x", {}]}}}', "per_language"),
    ('{"coverage": {"per_profile_per_language": {"p": {"python": ["x", null]}}}}', "per_profile_per_language"),
])
def test_invalid_hint_input_refuses_before_runtime_or_cleanup(raw, diagnostic, tmp_path, monkeypatch):
    (tmp_path / "hints.json").write_text(raw, encoding="utf-8")
    retained = tmp_path / "results" / "previous.txt"
    retained.parent.mkdir()
    retained.write_text("preserve", encoding="utf-8")
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda *a, **k: pytest.fail("phase entered"))
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda *a, **k: pytest.fail("cleanup entered"))
    with pytest.raises(ConfigurationError, match=diagnostic):
        orchestrator.run_pipeline(SimpleNamespace(), settings=RunSettings(environment=dict(os.environ), base_dir=tmp_path))
    assert retained.read_text("utf-8") == "preserve"
    assert config_module.active_settings() is None


@pytest.mark.parametrize("selection", ["object", "file", "environment"])
@pytest.mark.parametrize("filename", ["missing.json", "hints.json"])
def test_explicit_missing_hint_file_refuses_even_when_default_name_is_selected(selection, filename, tmp_path):
    if selection == "object":
        settings = RunSettings(environment=dict(os.environ), base_dir=tmp_path, sections={"main": {"hints_file": filename}})
    elif selection == "file":
        config = tmp_path / "config.ini"
        config.write_text(f"[main]\nhints_file = {filename}\n", encoding="utf-8")
        settings = resolve_settings(base_dir=tmp_path, config_files=(config,), environment={}, models={})
    else:
        settings = resolve_settings(base_dir=tmp_path, config_files=(), models={},
                                    environment={"TESTUARY_MAIN_HINTS_FILE": filename})
    with pytest.raises(ConfigurationError, match="selected hints file"):
        with config_module.use_run_settings(settings):
            pytest.fail("invocation entered")


def test_missing_optional_default_and_explicit_empty_keep_prompts_unchanged(tmp_path):
    with config_module.use_run_settings(RunSettings(environment=dict(os.environ), base_dir=tmp_path)):
        before = prompts.resolve_agent_prompts("test_coverage", "python")
        assert hints_module.load_hints() == {}
    (tmp_path / "hints.json").write_text("invalid", encoding="utf-8")
    with config_module.use_run_settings(RunSettings(environment=dict(os.environ), base_dir=tmp_path, sections={"main": {"hints_file": ""}})):
        assert hints_module.load_hints() == {}
        assert prompts.resolve_agent_prompts("test_coverage", "python") == before


@pytest.mark.parametrize("phase", ["initial", "coverage", "mutation"])
def test_changed_hint_file_and_worker_mutation_do_not_change_captured_input(phase, tmp_path):
    path = tmp_path / "hints.json"
    payload = {phase: {"general": ["captured", "guidance"]}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    with config_module.use_run_settings(RunSettings(environment=dict(os.environ), base_dir=tmp_path)):
        path.write_text(json.dumps({phase: {"general": "later"}}), encoding="utf-8")

        def read():
            hints = hints_module.load_hints()
            assert hints == payload
            hints[phase]["general"].append("not shared")
            return prompts.resolve_agent_prompts(f"test_{phase}", "python")["system_prompt"]

        with ThreadPoolExecutor(max_workers=2) as workers:
            rendered = list(workers.map(lambda _: read(), range(4)))
        assert all(value.endswith("captured\nguidance") for value in rendered)
        assert hints_module.load_hints() == payload
    with config_module.use_run_settings(RunSettings(environment=dict(os.environ), base_dir=tmp_path)):
        assert hints_module.load_hints()[phase]["general"] == "later"


@pytest.mark.parametrize("exit_kind", ["failure", "cancel"])
def test_hint_snapshot_is_restored_after_failed_or_cancelled_invocation(exit_kind, tmp_path):
    path = tmp_path / "hints.json"
    path.write_text('{"initial": {"general": "first"}}', encoding="utf-8")
    failure = RuntimeError if exit_kind == "failure" else KeyboardInterrupt
    with pytest.raises(failure):
        with config_module.use_run_settings(RunSettings(environment=dict(os.environ), base_dir=tmp_path)):
            raise failure
    path.write_text('{"initial": {"general": "second"}}', encoding="utf-8")
    with config_module.use_run_settings(RunSettings(environment=dict(os.environ), base_dir=tmp_path)):
        assert hints_module.load_hints()["initial"]["general"] == "second"


def test_hint_setup_failure_does_not_strand_invocation_guard(tmp_path):
    path = tmp_path / "hints.json"
    path.write_text("invalid", encoding="utf-8")
    with pytest.raises(ConfigurationError):
        with config_module.use_run_settings(RunSettings(environment=dict(os.environ), base_dir=tmp_path)):
            pytest.fail("invocation entered")
    path.write_text("{}", encoding="utf-8")
    with config_module.use_run_settings(RunSettings(environment=dict(os.environ), base_dir=tmp_path)):
        assert hints_module.load_hints() == {}


def test_notes_are_ignored_and_file_is_read_once_per_invocation(tmp_path, monkeypatch):
    path = tmp_path / "hints.json"
    payload = {"_note": ["about this file"], "coverage": {"_note": {"about": "this phase"}, "general": "guidance"}}
    path.write_text(json.dumps(payload), encoding="utf-8")
    reads = []
    original = Path.read_bytes

    def read(candidate):
        if candidate == path:
            reads.append(candidate)
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", read)
    with config_module.use_run_settings(RunSettings(environment=dict(os.environ), base_dir=tmp_path)):
        assert hints_module.load_hints() == payload
        assert hints_module.load_hints() == payload
        assert prompts.resolve_agent_prompts("test_coverage", "python")["system_prompt"].endswith("guidance")
    assert reads == [path]


def test_unreadable_hint_input_is_not_a_silent_fallback(tmp_path, monkeypatch):
    path = tmp_path / "hints.json"
    path.write_text("{}", encoding="utf-8")
    original = Path.read_bytes

    def read(candidate):
        if candidate == path:
            raise OSError("private-error-sentinel")
        return original(candidate)

    monkeypatch.setattr(Path, "read_bytes", read)
    with pytest.raises(ConfigurationError, match="cannot read hints file") as error:
        with config_module.use_run_settings(RunSettings(environment=dict(os.environ), base_dir=tmp_path)):
            pytest.fail("invocation entered")
    assert "private-error-sentinel" not in str(error.value)


def test_all_hint_problems_are_listed_without_cell_contents(tmp_path):
    (tmp_path / "hints.json").write_text(json.dumps({
        "covrage": {"general": "private-text-sentinel"},
        "coverage": {"genral": "private-text-sentinel", "general": ["private-text-sentinel", 3]},
        "mutation": {"per_language": False},
    }), encoding="utf-8")
    with pytest.raises(ConfigurationError) as error:
        with config_module.use_run_settings(RunSettings(environment=dict(os.environ), base_dir=tmp_path)):
            pytest.fail("invocation entered")
    for label in ("covrage", "genral", "coverage.general", "mutation.per_language"):
        assert label in str(error.value)
    assert "private-text-sentinel" not in str(error.value)


def test_cli_missing_selected_hint_file_exits_two_before_work(tmp_path, monkeypatch, capsys):
    from testuary import __main__ as cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    monkeypatch.setattr(sys, "argv", ["testuary", "--project-name", "subject", "--source-files", "module.py",
                                     "--hints-file", "missing.json", "--runs", "1", "--display", "plain", "--no-color"])
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda *a, **k: pytest.fail("phase entered"))
    with pytest.raises(SystemExit) as error:
        cli._main()
    assert error.value.code == 2
    assert "selected hints file" in capsys.readouterr().err
    assert not (tmp_path / "testuary").exists()


def test_cold_cli_import_with_bad_hints_still_reports_configuration_exit_two(tmp_path):
    (tmp_path / "hints.json").write_text("invalid", encoding="utf-8")
    environment = dict(os.environ)
    environment.pop("TESTUARY_TIMING_RUN_COUNT", None)
    result = subprocess.run(
        [sys.executable, "-B", "-m", "testuary", "--hints-file", "hints.json",
         "--source-files", "module.py", "--runs", "1", "--display", "plain", "--no-color"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2, result.stderr
    assert "valid JSON" in result.stderr
    assert "Traceback" not in result.stderr
    assert not (tmp_path / "testuary").exists()


def test_actual_initial_worker_uses_the_snapshot_after_file_changes(tmp_path, monkeypatch):
    from testuary import models
    from testuary import starters

    (tmp_path / "hints.json").write_text('{"initial": {"general": "captured-guidance"}}', encoding="utf-8")
    root = tmp_path / "subject"
    root.mkdir()
    target = root / "module.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    seen = []

    def generate(messages, **kwargs):
        seen.append(messages)
        return "no code fence", {"input_tokens": 1, "output_tokens": 1}

    settings = RunSettings(environment=dict(os.environ), base_dir=tmp_path, sections={
        "api": {"model": "openai_example"}, "project": {"name": "subject"},
        "starter": {"suite_count": 1, "attempts_per_suite": 1},
    }, models={"openai_example": {"provider": "openai", "model": "chosen"}})
    source = SimpleNamespace(generate_suite=generate)
    with config_module.use_run_settings(settings), models.installed(source):
        (tmp_path / "hints.json").write_text('{"initial": {"general": "later-guidance"}}', encoding="utf-8")
        result = starters.run_starter_phase(
            language="python", target_file_path=str(target), source_dir=str(root),
            test_output_dir=str(tmp_path / "output"), suites_dir=str(tmp_path / "suites"),
            log=lambda *a, **k: None,
        )
    assert result.status == "cancelled_zero_passing"
    assert len(seen) == 1
    prompt = json.dumps(seen)
    assert "captured-guidance" in prompt
    assert "later-guidance" not in prompt
