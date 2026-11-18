"""Real CLI refusals use one presenter and one captured configuration."""
import subprocess
import sys

import pytest

from testuary import __main__ as cli, orchestrator, presentation, settings as configuration
from testuary import config
from testuary import project
from testuary.settings import ConfigurationError, RunSettings


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native execution"))
    monkeypatch.setattr(project, "project_runtime", lambda *args, **kwargs: pytest.fail("provisioning"))
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)


@pytest.fixture
def presenters(monkeypatch):
    original = presentation.TerminalPresenter
    instances = []

    class Recorder(original):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.closes = 0
            self.diagnostic_writes = 0
            instances.append(self)

        def close(self):
            self.closes += 1
            return super().close()

        def write_diagnostic(self, callback):
            self.diagnostic_writes += 1
            return super().write_diagnostic(callback)

    monkeypatch.setattr(presentation, "TerminalPresenter", Recorder)
    return instances


def values(root, *, secret="private-fixture-key"):
    return RunSettings(base_dir=root, sections={"api": {"model": "fixture"},
        "project": {"source_dir": str(root / "project"), "output_dir": str(root / "output")}},
        models={"fixture": {"provider": "openai", "model": "test-model",
                            "api_url": "https://example.test/v1", "api_key": secret}})


@pytest.mark.parametrize("prefix", [[], ["generate"]])
@pytest.mark.parametrize("mode", ["plain", "compact", "detailed"])
@pytest.mark.parametrize("stage", ["resolution", "execution"])
def test_actual_cli_refusal_is_redacted_scoped_and_closes_once(prefix, mode, stage, tmp_path, monkeypatch, capsys, presenters):
    captured = values(tmp_path)
    failure = ConfigurationError("invalid request input private-fixture-key")
    calls = []

    def resolve(**kwargs):
        calls.append(kwargs)
        if stage == "resolution":
            # Fails before any credential, model or runtime is resolved.
            raise ConfigurationError("invalid configuration field")
        return captured

    monkeypatch.setattr(configuration, "resolve_settings", resolve)
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda *a: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(sys, "argv", ["testuary", *prefix, "--display", mode, "--no-color"])
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 2 and len(calls) == 1
    if stage == "execution":
        assert stopped.value.__cause__ is failure
    text = capsys.readouterr()
    assert text.out == "" and "\x1b[" not in text.err
    # The person ran generate, not check: the refusal does not tell them to check again.
    assert text.err.count("Next step: Correct it, then run the command again.") == 1
    assert "check again" not in text.err
    assert "private-fixture-key" not in text.err and "Diagnostic:" not in text.err
    if mode == "detailed":
        assert text.err.count("Setup before the run") == 1
        assert "Testuary quick check" not in text.err and text.err.count("Fix:") == 1
        assert "Source directory: not reported" in text.err and "API key: unknown" in text.err
    else:
        # generate ran no check: the refusal reads as the setup before the run.
        assert text.err.count("Setup before the run: 1 problem found.") == 1
        assert "quick check" not in text.err
        assert "Fix:" not in text.err and "Source directory:" not in text.err and "API key" not in text.err
    assert len(presenters) == 1 and presenters[0].closes == presenters[0].diagnostic_writes == 1
    assert config.active_settings() is None and not list(tmp_path.iterdir())


@pytest.mark.parametrize("mode", ["plain", "detailed"])
def test_a_refused_check_says_to_check_again(mode, monkeypatch, capsys, presenters):
    def refuse(**kwargs):
        raise ConfigurationError("invalid configuration field")

    monkeypatch.setattr(configuration, "resolve_settings", refuse)
    monkeypatch.setattr(sys, "argv", ["testuary", "check", "--display", mode, "--no-color"])
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 2
    assert capsys.readouterr().err.count("Correct it, then run the check again.") >= 1


@pytest.mark.parametrize("prefix", [[], ["generate"]])
def test_successful_cli_resolves_all_flags_once_and_preserves_invocation_policy(prefix, tmp_path, monkeypatch, presenters):
    calls, resolved, observed = [], [], []
    real_resolve = configuration.resolve_settings

    def resolve(**kwargs):
        calls.append(kwargs)
        result = real_resolve(**kwargs, base_dir=tmp_path, use_environment=False,
                              models={"fixture": {"provider": "openai", "api_url": "https://example.test/v1"}})
        resolved.append(result)
        return result

    def execute(args, selected_languages):
        assert config.active_settings() is resolved[0]
        assert orchestrator._cli_settings_overrides(args) == {} and args.config is None
        observed.append((args, selected_languages))
        return 0

    monkeypatch.setattr(configuration, "resolve_settings", resolve)
    monkeypatch.setattr(orchestrator, "_run_pipeline", execute)
    monkeypatch.setattr(sys, "argv", ["testuary", *prefix, "--config", str(tmp_path / "overlay.ini"),
        "--model", "fixture", "--project-name", "profile", "--source-dir", "original",
        "--output-dir", "output", "--python-executable", "runtime/python", "--java-home", "jdk",
        "--build-java-home", "build-jdk", "--python-import-root", "src", "--api-url", "https://selected.test/v1",
        "--proxy-port", "8002", "--concurrent-requests", "2", "--reuse-starter-suites", "--phase-scores", "coverage",
        "--hints-file", "", "--prompt-profile", "", "--source-files", "one.py,two.py", "--targets", "2",
        "--runs", "1", "--resume", "--reuse", "starter", "--stop-after-phase", "mutation",
        "--languages", "python", "--display", "plain", "--no-color", "--no-project-settings",
        "--home", str(tmp_path / "home")])
    (tmp_path / "overlay.ini").write_text("[main]\nseed = 8\n", encoding="utf-8")
    (tmp_path / "home").mkdir()
    # The command line's relative paths are taken from the working directory.
    monkeypatch.chdir(tmp_path)
    assert cli._main() == 0
    assert len(calls) == len(resolved) == len(observed) == 1
    assert calls[0]["project_settings"] is False and observed[0][0].no_project_settings is None
    # The run takes the directory through the settings, not a second time.
    assert calls[0]["home"] == str(tmp_path / "home") and observed[0][0].home is None
    settings = resolved[0]
    assert settings.main.seed == 8
    assert settings.sections["api"]["api_url"] == "https://selected.test/v1"
    assert settings.sections["project"]["source_dir"] == str(tmp_path / "original")
    assert settings.sections["project"]["output_dir"] == str(tmp_path / "output")
    assert settings.sections["starter"]["concurrent_requests"] == "2"
    assert settings.sections["starter"]["concurrent_requests_ollama"] == "2"
    assert settings.sections["starter"]["reuse_suites"] == "true"
    assert settings.sections["finishing"]["phase_scores"] == "coverage"
    args, selected = observed[0]
    assert (args.source_files, args.targets, args.runs, args.resume, args.reuse_starter,
            args.stop_after_phase, selected, args.display, args.no_color) == (
                ("one.py", "two.py"), "2", 1, True, True, "mutation", {"python"}, "plain", True)
    assert callable(args._diagnostic_writer)
    assert len(presenters) == 1 and presenters[0].closes == 1
    assert presenters[0].diagnostic_writes == 0
    assert config.active_settings() is None


def test_actual_occupied_output_refusal_preserves_evidence(tmp_path, monkeypatch, capsys, presenters):
    output = tmp_path / "output"
    output.mkdir()
    evidence = output / "unrelated.txt"
    evidence.write_bytes(b"existing evidence")
    captured = values(tmp_path)
    monkeypatch.setattr(configuration, "resolve_settings", lambda **kwargs: captured)
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: pytest.fail("native setup"))
    monkeypatch.setattr(sys, "argv", ["testuary", "generate", "--source-files", "module.py", "--display", "plain"])
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 2
    text = capsys.readouterr()
    assert "the run directory is not empty" in text.err and "Diagnostic:" not in text.err
    assert evidence.read_bytes() == b"existing evidence"
    assert list(output.iterdir()) == [evidence]
    assert len(presenters) == 1 and presenters[0].closes == 1
    assert config.active_settings() is None


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), RuntimeError("ordinary execution error"), ValueError("ordinary body error")])
def test_unexpected_body_failures_are_not_converted_to_configuration_reports(failure, tmp_path, monkeypatch, presenters):
    monkeypatch.setattr(configuration, "resolve_settings", lambda **kw: values(tmp_path))
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda *a: (_ for _ in ()).throw(failure))
    monkeypatch.setattr(sys, "argv", ["testuary", "generate", "--display", "plain"])
    with pytest.raises(type(failure)) as caught:
        cli._main()
    assert caught.value is failure
    assert len(presenters) == 1 and presenters[0].closes == 1 and presenters[0].diagnostic_writes == 0
    assert config.active_settings() is None


def test_readiness_renderer_failure_is_not_retried_or_swallowed(tmp_path, monkeypatch, presenters):
    failure = OSError("display output failed")
    calls = []

    def render(report, *, presenter, before_run=False):
        calls.append((report, presenter))
        raise failure

    monkeypatch.setattr(configuration, "resolve_settings", lambda **kw: values(tmp_path))
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda *a: (_ for _ in ()).throw(ConfigurationError("bad input")))
    monkeypatch.setattr(presentation, "present_readiness", render)
    monkeypatch.setattr(sys, "argv", ["testuary", "--display", "plain"])
    with pytest.raises(OSError) as caught:
        cli._main()
    assert caught.value is failure and len(calls) == 1
    assert len(presenters) == 1 and presenters[0].closes == 1
    assert config.active_settings() is None
