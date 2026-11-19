"""`testuary generate` and its flags without `generate` share one public CLI route."""
import sys

import pytest

from testuary import __main__ as cli, orchestrator, settings
from testuary import config
from testuary import project


@pytest.fixture(autouse=True)
def _no_native_or_provider(monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native process"))
    monkeypatch.setattr(project, "project_runtime", lambda *args, **kwargs: pytest.fail("provisioning"))
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration",
                        lambda *a, **k: pytest.fail("phase model preflight"))


def _files(root):
    return {str(path.relative_to(root)): path.read_bytes()
            for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("argv", [[], ["generate"], ["--help"], ["-h"], ["generate", "--help"]])
def test_help_and_empty_commands_never_resolve_or_write(argv, tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    # argparse wraps the help to the terminal's width.
    monkeypatch.setenv("COLUMNS", "80")
    monkeypatch.setattr(sys, "argv", ["testuary", *argv])
    monkeypatch.setattr(settings, "resolve_settings", lambda **kw: pytest.fail("settings resolution"))
    monkeypatch.setattr(orchestrator, "run_pipeline", lambda *a, **k: pytest.fail("generation"))
    monkeypatch.setattr(settings, "_bundled_template_path", lambda *a: pytest.fail("template access"))
    if "--help" in argv or "-h" in argv:
        with pytest.raises(SystemExit) as stopped:
            cli._main()
        assert stopped.value.code == 0
    else:
        assert cli._main() == 2
    text = capsys.readouterr().out
    assert "usage:" in text and "Run started" not in text
    if argv != ["generate", "--help"]:
        # The main help and the hint of a bare command stay short.
        assert len(text.splitlines()) < 40
    assert _files(tmp_path) == {}


@pytest.mark.parametrize("reuse", [[], ["--resume"], ["--reuse", "starter,mutation"]])
@pytest.mark.parametrize("outcome", [0, 1, RuntimeError("controlled body failure"), KeyboardInterrupt()])
def test_the_flags_with_and_without_generate_reach_identical_settings_and_invocation(
        reuse, outcome, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    overlay = tmp_path / "run.ini"
    overlay.write_text("[api]\nmodel = openai/test-model\n[starter]\nsuite_count = 3\n", encoding="utf-8")
    calls = []
    arguments = ["--config", str(overlay), "--source-files", "one.py,two.py", "--targets", "2",
                 "--runs", "1", "--languages", "python", "--output-dir", "destination",
                 "--stop-after-phase", "coverage", "--display", "plain", "--no-color", *reuse]

    def execute(args, selected_languages):
        captured = config.active_settings()
        calls.append((
            args.source_files, args.targets, args.runs, args.resume,
            args.reuse_starter, args.reuse_mutation, args.stop_after_phase,
            selected_languages, captured.export(), callable(args._diagnostic_writer),
        ))
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(orchestrator, "_run_pipeline", execute)
    before = _files(tmp_path)
    for prefix in ([], ["generate"]):
        monkeypatch.setattr(sys, "argv", ["testuary", *prefix, *arguments])
        original_argv = list(sys.argv)
        if isinstance(outcome, BaseException):
            with pytest.raises(type(outcome)) as caught:
                cli._main()
            assert caught.value is outcome
        else:
            assert cli._main() == outcome
        assert sys.argv == original_argv
        assert config.active_settings() is None
    assert len(calls) == 2 and calls[0] == calls[1]
    assert calls[0][7] == {"python"} and calls[0][-1] is True
    assert _files(tmp_path) == before


@pytest.mark.parametrize("prefix", [[], ["generate"]])
@pytest.mark.parametrize("arguments", [
    ["--runs", "0"], ["--run", "-1"], ["--runs", "2", "--run", "1"], ["--resume", "--reuse", "coverage"],
    ["--languages", "not_a_language"], ["--unknown-option"],
    ["--reuse", "bogus"], ["--reuse", ","], ["--reuse", "test"],
    ["--reuse", "finishing", "--runs", "2"], ["--stop-after-phase", "coverage", "--runs", "2"],
])
def test_generation_argument_refusal_precedes_resolution(prefix, arguments, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["testuary", *prefix, *arguments])
    monkeypatch.setattr(settings, "resolve_settings", lambda **kw: pytest.fail("settings resolution"))
    monkeypatch.setattr(orchestrator, "run_pipeline", lambda *a, **k: pytest.fail("generation"))
    try:
        code = cli._main()
    except SystemExit as stopped:
        code = stopped.code
    assert code == 2 and _files(tmp_path) == {}


def test_the_languages_help_says_what_the_list_does():
    """The list refuses a run whose targets are in another language
    (test_run_admission). The dependency check follows the targets'
    languages, and nothing is detected without the list."""
    help_text = next(action.help for action in cli._build_parser()._actions
                     if "--languages" in action.option_strings)
    assert help_text.startswith("Refuse the run if a target")
    assert "preflight" not in help_text and "detect" not in help_text


@pytest.mark.parametrize("language", ["python", "java"])
def test_generate_demo_uses_existing_single_delegation(language, tmp_path, monkeypatch):
    from testuary.demo import cli as demo_cli

    calls = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings, "resolve_settings", lambda **kw: pytest.fail("settings resolution"))
    monkeypatch.setattr(orchestrator, "run_pipeline", lambda *a, **k: pytest.fail("normal pipeline"))
    monkeypatch.setattr(demo_cli, "main", lambda args, parser: calls.append(vars(args)) or 0)
    for prefix in ([], ["generate"]):
        monkeypatch.setattr(sys, "argv", ["testuary", *prefix, "--demo", "--demo-language", language,
                                         "--display", "plain", "--no-color"])
        with pytest.raises(SystemExit) as stopped:
            cli._main()
        assert stopped.value.code == 0
    assert len(calls) == 2 and calls[0] == calls[1]
    assert _files(tmp_path) == {}
