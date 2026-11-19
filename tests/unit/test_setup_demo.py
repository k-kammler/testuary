"""The setup wizard's demonstration: answers by key, no process or
connection, the written project checked against the recorded run."""
from __future__ import annotations

import configparser
import subprocess

import pytest

from testuary import setup as _setup
from testuary.demo import setup_demo


@pytest.fixture
def played(monkeypatch):
    """Stand in for the playback, which a test process cannot start once
    phase modules are loaded. The integration test plays it for real."""
    calls = []

    def playback(output, language, args):
        calls.append((output, language))
        return 0

    monkeypatch.setattr(setup_demo, "_playback_run", playback)
    return calls


def _project(project_dir):
    """The project's settings file in ``project_dir``."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(project_dir / ".testuary" / "project.ini", encoding="utf-8")
    return parser


@pytest.mark.parametrize("language, project_name, alias", [
    ("python", "tenacity", "ollama_deepseek"),
    ("java", "jsonassert", "ollama_deepseek")])
def test_the_wizard_writes_the_recorded_runs_project(tmp_path, played, capsys, language, project_name, alias):
    out = tmp_path / "out"
    status = setup_demo.main(["--demo", "--demo-language", language, "--demo-speed", "instant",
                              "--output-dir", str(out)])
    assert status == 0 and played == [(out / "demo", language)]
    project = (out / "workspace" / "projects" / project_name).resolve()
    settings = _project(project)
    assert settings["api"]["model"] == alias and settings["project"]["language"] == language
    text = capsys.readouterr().out
    assert setup_demo.BANNER in text and f"Run it in {project} with: testuary" in text
    # The key is left to an environment variable the demo does not set.
    models = configparser.ConfigParser(interpolation=None)
    models.read(out / "models.ini", encoding="utf-8")
    assert not any(models.has_option(section, "api_key") for section in models.sections())
    assert "TESTUARY_OLLAMA_API_KEY is not set here" in text
    # The project is complete: the Java demo's project has a build file.
    assert "build file" not in text
    if language == "java":
        assert settings["project"]["maven"] == setup_demo._MAVEN


def test_a_testuary_variable_does_not_reach_the_demonstrated_setup(tmp_path, played, monkeypatch):
    """The setup demonstration sets the TESTUARY_ variables aside, so a
    directory variable of the user's does not move the project it writes, and
    the variable is back afterwards."""
    import os

    monkeypatch.setenv("TESTUARY_MAIN_TESTUARY_DIR", "elsewhere")
    status = setup_demo.main(["--demo", "--demo-speed", "instant", "--output-dir", str(tmp_path / "o")])
    assert status == 0
    assert os.environ["TESTUARY_MAIN_TESTUARY_DIR"] == "elsewhere"


def test_the_wizard_starts_no_process_and_opens_no_connection(tmp_path, played, monkeypatch):
    def refused(*args, **kwargs):
        raise AssertionError("the setup demo started a process or opened a connection")

    import httpx

    monkeypatch.setattr(subprocess, "run", refused)
    monkeypatch.setattr(subprocess, "Popen", refused)
    monkeypatch.setattr(httpx, "get", refused)
    status = setup_demo.main(["--demo", "--demo-speed", "instant", "--output-dir", str(tmp_path / "o")])
    assert status == 0


def test_the_probe_answers_what_the_python_setup_asks_of_this_interpreter(tmp_path):
    """Python's setup asks the interpreter its version, naming nothing, and
    which of the modules it names it cannot import. The probe answers as
    this interpreter by what is asked, whatever the script's words. Another
    program does not start."""
    import platform
    import sys

    from langadapter.python.adapter import PythonAdapter

    probe = setup_demo.DemoProbe()
    (interpreter,) = [question for question in PythonAdapter.setup_questions(tmp_path, probe)
                      if question.key == "python.interpreter"]
    assert interpreter.check(sys.executable) is None
    note = interpreter.after(sys.executable)
    assert note.missing == () and platform.python_version() in note.lines[0]
    reworded = "import platform; print(platform.python_version())"
    assert probe.run([sys.executable, "-c", reworded], 30) == (0, platform.python_version() + "\n")
    assert probe.run([sys.executable, "-c", "reworded", "pytest", "no_such_demo_module"], 60) == (
        0, "no_such_demo_module\n")
    assert probe.run(["python3", "-c", "import sys; print(sys.version_info)"], 30) is None


def test_an_unknown_question_stops_the_demo_as_outdated(tmp_path, played, monkeypatch, capsys):
    answers = setup_demo._answers

    def without_targets(language, manifest, models_text="", targets=""):
        return {key: value for key, value in answers(language, manifest, models_text, targets).items()
                if key != "targets"}

    monkeypatch.setattr(setup_demo, "_answers", without_targets)
    status = setup_demo.main(["--demo", "--demo-speed", "instant", "--output-dir", str(tmp_path / "o")])
    assert status == 3 and not played
    assert "'targets', which the demo cannot answer" in capsys.readouterr().err


def test_a_setup_that_is_not_the_recorded_run_is_not_played(tmp_path, played, monkeypatch, capsys):
    answers = setup_demo._answers
    monkeypatch.setattr(setup_demo, "_answers",
                        lambda language, manifest, models_text="", targets="": {
                            **answers(language, manifest, models_text, targets),
                                                    "model.alias": "another_alias",
                                                    "model": "another_alias"})
    status = setup_demo.main(["--demo", "--demo-speed", "instant", "--output-dir", str(tmp_path / "o")])
    assert status == 3 and not played
    assert "differs from the recorded run: model another_alias" in capsys.readouterr().err


def test_usage_errors_write_nothing(tmp_path, played, capsys):
    full = tmp_path / "full"
    full.mkdir()
    (full / "kept.txt").write_text("x", encoding="utf-8")
    assert setup_demo.main(["--demo", "--output-dir", str(full)]) == 2
    assert setup_demo.main(["--demo", "--demo-speed", "fast", "--output-dir", str(tmp_path / "a")]) == 2
    assert "unknown demonstration speed" in capsys.readouterr().err
    assert sorted(p.name for p in full.iterdir()) == ["kept.txt"] and not (tmp_path / "a").exists()
    assert not played


def test_the_setup_demonstration_parses_its_options_as_the_command_line_does():
    """No abbreviated option, as the command line's parsers refuse them, and
    each playback option with the command line's help."""
    parser = setup_demo._parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--demo", "--demo-lang", "java"])
    assert parser._option_string_actions["--demo-speed"].help


def test_a_file_as_the_run_dir_is_a_usage_error(tmp_path, played, capsys):
    target = tmp_path / "file.txt"
    target.write_text("x", encoding="utf-8")
    assert setup_demo.main(["--demo", "--output-dir", str(target)]) == 2
    assert "new or empty directory" in capsys.readouterr().err


def test_data_that_do_not_match_their_manifest_stop_as_outdated(tmp_path, played, monkeypatch, capsys):
    def mismatch(outdir, language):
        raise setup_demo.demo_cli.DataMismatch("the shipped data do not match their manifest")

    monkeypatch.setattr(setup_demo, "_prepare", mismatch)
    status = setup_demo.main(["--demo", "--demo-speed", "instant", "--output-dir", str(tmp_path / "o")])
    assert status == 3
    assert "no longer fit this package version" in capsys.readouterr().err


def test_the_typist_shows_a_question_as_the_wizard_does(monkeypatch):
    """The demonstrated wizard shows the reason, the numbered choices and the
    prompt as terminal_ask does, with no default on a secret question."""
    asked = []
    monkeypatch.setattr("builtins.input", lambda prompt: asked.append(prompt) or "2")
    question = _setup.Question("provider", "Provider", default="ollama",
                               choices=("ollama", "openai"), reason="Pick one")
    shown_by_wizard = []
    monkeypatch.setattr("builtins.print", lambda line: shown_by_wizard.append(line))
    assert _setup.terminal_ask(question) == "openai"
    monkeypatch.undo()
    written = []
    typist = setup_demo._Typist({"provider": "openai", "key": "secret"}, None, written.append)
    typist(question)
    assert "".join(written) == "".join(line + "\n" for line in shown_by_wizard) + asked[0] + "openai\n"
    written.clear()
    typist(_setup.Question("key", "API key", default="old", secret=True))
    assert "".join(written) == "API key: (hidden)\n"


def test_answers_are_typed_at_the_paced_speed(monkeypatch):
    pauses = []
    monkeypatch.setattr(setup_demo.time, "sleep", pauses.append)
    shown = []
    typist = setup_demo._Typist({"k": "abc", "s": "secret"}, (2.0, False, 120.0), shown.append)
    assert typist(_setup.Question("k", "Prompt", default="d")) == "abc"
    assert "".join(shown) == "Prompt [d]: abc\n"
    # Three characters and the pause after the answer, at twice the speed.
    assert pauses == [setup_demo._CHARACTER / 2] * 3 + [setup_demo._ANSWERED / 2]
    shown.clear()
    assert typist(_setup.Question("s", "Key", secret=True)) == "secret"
    assert "secret" not in "".join(shown) and "(hidden)" in "".join(shown)


def test_setup_demo_is_reached_through_testuary_setup(tmp_path, played):
    status = _setup.main(["--demo", "--demo-speed", "instant", "--output-dir", str(tmp_path / "o")])
    assert status == 0 and played


def test_stepping_types_an_answer_only_after_enter_pressed_while_it_waits():
    import queue
    import threading
    from types import SimpleNamespace

    typist = setup_demo._Typist({"k": "abc"}, None, lambda text: None)
    typist.step, typist.keys = True, SimpleNamespace(pressed=queue.Queue(), close=lambda: None)
    typist.keys.pressed.put("\r")  # pressed before the question: ignored
    done = threading.Event()
    threading.Thread(target=lambda: (typist(_setup.Question("k", "Prompt")), done.set()),
                     daemon=True).start()
    assert not done.wait(0.3)
    typist.keys.pressed.put("\r")
    assert done.wait(1)


def test_a_recorded_ollama_cloud_alias_is_set_up_as_recorded():
    """A recorded scenario's alias drives the model answers: an Ollama alias
    without an address is Ollama Cloud, with its key named as an unset
    environment variable."""
    manifest = {"exemplar": {"instance": "tenacity"},
                "invocation": {"model": "ollama_deepseek"}}
    models = "[ollama_deepseek]\nprovider = ollama\nmodel = deepseek-v4.1-flash\n"
    answers = setup_demo._answers("python", manifest, models, "")
    assert answers["model.provider"] == "ollama-cloud"
    assert answers["model.name"] == "deepseek-v4.1-flash"
    assert answers["model.alias"] == "ollama_deepseek"
    assert "model.address" not in answers
    assert answers["project.folder"] == "workspace/projects/tenacity"
    # An alias with an address is no recorded Ollama Cloud alias.
    with pytest.raises(setup_demo.demo_cli.DataMismatch, match="not an Ollama Cloud alias"):
        setup_demo._model_answers(manifest, "[ollama_deepseek]\nprovider = openai\n"
                                  "model = x\napi_url = http://h/v1\n")


def test_the_demonstration_lists_each_model_service_once(tmp_path, played, capsys):
    """Without a terminal too, the model services appear once, with their
    labels, before the question that takes a number or a name."""
    import re

    assert setup_demo.main(["--demo", "--demo-speed", "instant", "--output-dir", str(tmp_path / "o")]) == 0
    text = capsys.readouterr().out
    shown = [line.strip() for line in text[:text.index("Which model service?")].splitlines()
             if re.fullmatch(r"\s+\d+\. .+", line)]
    assert shown == [f"{index}. {provider.label} ({provider.kind})"
                     for index, provider in enumerate(_setup.PROVIDERS, 1)]
