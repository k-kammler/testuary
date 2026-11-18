"""Help and template argument refusal precede filesystem or execution work."""
import sys

import pytest

from testuary import __main__ as cli
from testuary import orchestrator
from testuary import project as project_module
from testuary import settings as settings_module
from testuary.settings import RunSettings


@pytest.fixture(autouse=True)
def no_native_work(monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native process"))
    monkeypatch.setattr(project_module, "project_runtime", lambda *args, **kwargs: pytest.fail("provisioning"))


def test_a_refused_provision_writes_its_error_to_standard_error(monkeypatch, capsys):
    def refuse(**kwargs):
        raise ValueError("the settings were refused")

    monkeypatch.setattr(settings_module, "resolve_settings", refuse)
    assert cli._cmd_provision([]) == 1
    out, err = capsys.readouterr()
    assert out == "" and "the settings were refused" in err


def files(root):
    return {str(path.relative_to(root)): path.read_bytes() for path in root.rglob("*") if path.is_file()}


@pytest.mark.parametrize("argv", [["init", "--help"], ["new-language", "--help"],
                                  ["new-language", "java", "--help"], ["provision", "--help"],
                                  ["tutorial", "--help"]])
def test_every_current_help_path_is_side_effect_free(argv, tmp_path, monkeypatch, capsys):
    from testuary import settings as settings_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["testuary", *argv])
    monkeypatch.setattr(orchestrator, "run_pipeline", lambda *a, **k: pytest.fail("generation"))
    monkeypatch.setattr(settings_module, "_bundled_template_path", lambda *a: pytest.fail("template access"))
    monkeypatch.setattr(settings_module, "resolve_settings", lambda **kw: pytest.fail("run settings resolution"))
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 0
    assert "usage:" in capsys.readouterr().out
    assert files(tmp_path) == {}


@pytest.mark.parametrize("argv", [["init", "--typo"], ["init", "extra"], ["new-language"],
                                  ["new-language", "java", "extra"], ["new-language", "--typo"],
                                  ["tutorial", "--typo"]])
def test_template_arguments_are_validated_before_access(argv, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["testuary", *argv])
    monkeypatch.setattr(settings_module, "_bundled_template_path", lambda *a: pytest.fail("template access"))
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 2
    assert files(tmp_path) == {}


def test_init_still_creates_templates_and_preserves_existing_file(tmp_path, monkeypatch):
    from testuary import settings

    monkeypatch.chdir(tmp_path)
    models = settings.platform_user_dir() / "models.ini"
    models.write_bytes(b"private existing configuration\r\n")
    monkeypatch.setattr(sys, "argv", ["testuary", "init"])
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 0
    # The model catalogue belongs to the user directory, the hints to the project.
    assert models.read_bytes() == b"private existing configuration\r\n"
    assert set(files(tmp_path)) == {"hints.json"}
    models.unlink()
    with pytest.raises(SystemExit):
        cli._main()
    assert models.read_text(encoding="utf-8").startswith("# Model catalogue")


def test_init_writes_its_example_aliases_as_comments(tmp_path, monkeypatch):
    """The setup's walk takes a usable alias as model access set, and the
    examples' local vLLM aliases would count as usable with no service
    running, so init writes every example commented out."""
    from testuary import settings, setup

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["testuary", "init"])
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 0
    home = settings.platform_user_dir()
    assert setup.models(tmp_path, home=home) == ()
    assert "\n# [vllm_qwen3_coder_next]\n# provider = vllm\n" in (home / "models.ini").read_text(
        encoding="utf-8")


def test_init_writes_the_hints_where_runs_read_them(tmp_path, monkeypatch):
    """From a project's subdirectory, init writes hints.json to the project
    directory, where a run's default hints_file points."""
    from testuary.settings import resolve_settings

    project = tmp_path / "proj"
    (project / ".testuary").mkdir(parents=True)
    (project / "src").mkdir()
    monkeypatch.chdir(project / "src")
    monkeypatch.setattr(sys, "argv", ["testuary", "init"])
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 0
    assert (project / "hints.json").is_file() and not (project / "src" / "hints.json").exists()
    settings = resolve_settings(base_dir=project / "src", environment={})
    assert str((project / "hints.json").resolve()) in settings.consumer_files


def test_new_language_names_itself_in_its_help(tmp_path, monkeypatch, capsys):
    """testuary new-language --help shows Testuary's command, not
    LangAdapter's, though LangAdapter's scaffold answers it."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["testuary", "new-language", "--help"])
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: testuary new-language") and "langadapter new-language" not in out


def test_new_language_forwards_to_langadapters_scaffold(tmp_path, monkeypatch):
    """testuary new-language runs LangAdapter's scaffold for Testuary."""
    from langadapter import cli as langadapter_cli

    seen = []
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(langadapter_cli, "main",
                        lambda argv, prog="langadapter": seen.append((argv, prog)) or 0)
    monkeypatch.setattr(sys, "argv", ["testuary", "new-language", "sample", "--file-extension", ".xyz"])
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 0
    assert seen == [(["new-language", "sample", "--file-extension", ".xyz", "--for", "testuary"],
                     "testuary")]
    assert not files(tmp_path)


@pytest.mark.parametrize("argv", [["--display", "plain", "--no-color"]])
def test_empty_invocation_has_honest_configuration_error_without_diagnostic_claim(
        argv, tmp_path, monkeypatch, capsys):
    from testuary import settings as settings_module

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(settings_module, "resolve_settings", lambda **kw: RunSettings(base_dir=tmp_path))
    monkeypatch.setattr(sys, "argv", ["testuary", *argv])
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 2
    captured = capsys.readouterr()
    text = captured.out + captured.err
    assert "model" in text and "Configuration refused" in text
    assert "Run started" not in text and "see saved diagnostics" not in text
    assert files(tmp_path) == {}


def test_late_execution_failure_does_not_invent_a_log_path(tmp_path, monkeypatch):
    events = []
    failure = RuntimeError("controlled execution failure")

    def fail(*args):
        raise failure

    monkeypatch.setattr(orchestrator, "_run_pipeline", fail)
    with pytest.raises(RuntimeError) as caught:
        orchestrator.run_pipeline(cli._build_parser().parse_args([]),
                                  settings=RunSettings(base_dir=tmp_path), event_sink=events.append)
    assert caught.value is failure
    assert events[0].message == "Checking run inputs"
    assert events[-1].state == "failed" and "saved" not in events[-1].message
    assert files(tmp_path) == {}


def test_a_project_file_that_sets_an_address_is_refused_before_any_request(tmp_path, monkeypatch, capsys):
    """A project's settings file cannot choose where the model's key goes: a
    run of a project whose project.ini sets [openai] api_url ends at its
    start with status 2, names the step, and connects to nothing."""
    import socket

    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen()
    listener.setblocking(False)
    port = listener.getsockname()[1]
    home = tmp_path / "home"
    home.mkdir()
    (home / "models.ini").write_text("[openai]\napi_key = sk-test\n\n[gpt]\nprovider = openai\n"
                                     "model = gpt-test\n", encoding="utf-8")
    project = tmp_path / "project"
    (project / ".testuary").mkdir(parents=True)
    (project / ".testuary" / "project.ini").write_text(
        f"[api]\nmodel = gpt\n\n[openai]\napi_url = http://127.0.0.1:{port}/v1\n", encoding="utf-8")
    (project / "a.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.setattr(sys, "argv", ["testuary", "--home", str(home), "a.py"])
    try:
        with pytest.raises(SystemExit) as stopped:
            cli._main()
        assert stopped.value.code == 2
        text = "".join(capsys.readouterr())
        assert "[openai] api_url decides where the model's key goes" in text
        assert "global configuration, models.ini, the environment or on the command line" in text
        with pytest.raises(BlockingIOError):
            listener.accept()
    finally:
        listener.close()
