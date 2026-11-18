"""The setup's questions offer what it finds: the installed languages, the
project's first, with how many files each has; the project's environments
and the Pythons on PATH; the catalogue's usable model aliases. A
console answer may be a choice's number."""
from __future__ import annotations

import os
import sys

import pytest

from testuary import setup
from testuary import setup as _setup


class _Probe(setup.Probe):
    """The real file system, a fixed PATH and no network."""

    def __init__(self, on_path=None, environment=None, versions=None):
        self.on_path = dict(on_path or {})
        self.env = dict(environment or {})
        self.versions = dict(versions or {})

    def which(self, command):
        return self.on_path.get(command, "")

    def environment(self, name):
        return self.env.get(name, "")

    def run(self, argv, timeout):
        """Each interpreter reports its version (3.12.0 unless named) to the
        script run alone, or does not run (None), and lacks none of the
        tools a script names after it."""
        if len(argv) > 3:
            return 0, "\n"
        version = self.versions.get(argv[0], "3.12.0")
        return (0, version + "\n") if version else None

    def local_ollama(self):
        return False


def _walk(base_dir, probe, answers):
    """Run the project part with ``answers`` and return every question it
    asks, in order."""
    asked = []

    def ask(question):
        asked.append(question)
        return answers.get(question.key, "")

    try:
        _setup.run_setup(ask, probe, lambda text: None, base_dir, part="project", home=base_dir)
    except setup.SetupError:
        pass
    return {question.key: question for question in asked}


def test_the_projects_language_comes_first_with_its_files(tmp_path):
    (tmp_path / "pkg").mkdir()
    for name in ("a.py", "b.py"):
        (tmp_path / "pkg" / name).write_text("x = 1\n", encoding="utf-8")
    language = _walk(tmp_path, _Probe(), {"project.confirm": "no"})["project.language"]
    assert language.choices[0] == "python" and language.default == "python"
    assert set(language.choices) == set(setup.languages()) and not language.open
    assert "python (2 files)" in language.reason


def test_the_interpreters_are_the_projects_environments_then_those_on_path(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    script = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    venv = tmp_path.joinpath(".venv", *script)
    venv.parent.mkdir(parents=True)
    venv.write_text("", encoding="utf-8")
    other = tmp_path / "elsewhere" / "python3"
    other.parent.mkdir()
    other.write_text("", encoding="utf-8")
    asked = _walk(tmp_path, _Probe(on_path={"python3": str(other)}), {"project.confirm": "no"})
    interpreter = asked["python.interpreter"]
    assert interpreter.open and interpreter.default == str(venv)
    assert interpreter.choices[:2] == (str(venv), str(other))
    assert interpreter.choices[-1] == sys.executable
    assert len(set(interpreter.choices)) == len(interpreter.choices)


def test_an_interpreter_that_does_not_run_or_is_unsupported_is_neither_offered_nor_taken(tmp_path):
    """A Python that does not start (such as a store stub) and one older than
    the release supports (such as Python 2) are left out of the choices,
    and naming one is refused with the reason."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    stub, old = str(tmp_path / "stub" / "python"), str(tmp_path / "old" / "python")
    probe = _Probe(on_path={"python3": stub, "python": old}, versions={stub: None, old: "2.7.18"})
    asked, refusals = [], []

    def ask(question):
        asked.append(question)
        if question.key == "python.interpreter":
            # First the unsupported Python, then the one Testuary runs in.
            first = sum(item.key == "python.interpreter" for item in asked) == 1
            return old if first else sys.executable
        return "no" if question.key == "project.confirm" else ""

    def say(text):
        refusals.append(text)

    try:
        _setup.run_setup(ask, probe, say, tmp_path, part="project", home=tmp_path)
    except setup.SetupError:
        pass
    interpreter = next(question for question in asked if question.key == "python.interpreter")
    assert interpreter.choices == (sys.executable,)
    assert any("Python 2.7.18 is not supported: the Python adapter runs project Python 3.9" in text
               for text in refusals), refusals


def test_the_usable_model_aliases_are_offered_first(tmp_path):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "models.ini").write_text(
        "[zeta]\nprovider = openai\nmodel = z\n\n[alpha]\nprovider = ollama\nmodel = a\n",
        encoding="utf-8")
    probe = _Probe(environment={"TESTUARY_OPENAI_API_KEY": "fixture-key"})
    model = _walk(tmp_path, probe, {"project.confirm": "no"})["model"]
    # zeta has its key, and alpha needs a local service the probe does not find.
    assert model.choices == ("zeta", "alpha") and not model.open


def test_a_console_answer_may_be_a_choices_number(monkeypatch, capsys):
    question = setup.Question("python.interpreter", "The interpreter", default="/a",
                              choices=("/a", "/b"), open=True)
    monkeypatch.setattr("builtins.input", lambda prompt: "2")
    assert setup.terminal_ask(question) == "/b"
    assert "  1. /a" in capsys.readouterr().out
    monkeypatch.setattr("builtins.input", lambda prompt: "/c")
    assert setup.terminal_ask(question) == "/c"
    monkeypatch.setattr("builtins.input", lambda prompt: "3")
    assert setup.terminal_ask(question) == "3"


@pytest.mark.parametrize("answer, expected", [("2", "java"), ("", ""), ("java", "java")])
def test_a_numbered_language_answer_is_the_language(answer, expected, monkeypatch):
    """Enter stays empty, so the walk takes the default."""
    question = setup.Question("project.language", "Its language", default="python",
                              choices=("python", "java"))
    monkeypatch.setattr("builtins.input", lambda prompt: answer)
    assert setup.terminal_ask(question) == expected


def test_an_environment_linked_to_another_python_is_offered_apart(tmp_path):
    """An environment's python is a link to the Python it was made from, but
    another environment: both are offered."""
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    system = tmp_path / "usr" / "python3"
    system.parent.mkdir()
    system.write_text("", encoding="utf-8")
    script = ("Scripts", "python.exe") if os.name == "nt" else ("bin", "python")
    venv = tmp_path.joinpath(".venv", *script)
    venv.parent.mkdir(parents=True)
    try:
        venv.symlink_to(system)
    except OSError:
        pytest.skip("this system makes no symbolic links for this user")
    asked = _walk(tmp_path, _Probe(on_path={"python3": str(system)}), {"project.confirm": "no"})
    choices = asked["python.interpreter"].choices
    assert str(venv) in choices and str(system) in choices
