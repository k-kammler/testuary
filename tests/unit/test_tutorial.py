"""testuary tutorial: each extension point is applied with Testuary's own
setup, shown and checked with testuary check, and nothing contacts a model."""
from __future__ import annotations

import ast
import configparser
import json
import sys
from importlib import resources
from pathlib import Path

from testuary import tutorial

EXAMPLES = resources.files("testuary").joinpath("_templates", "examples")


def _steps():
    return {step.title: step for step in tutorial.tutorial().steps}


def _ini(path):
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(path, encoding="utf-8")
    return parser


def test_the_tutorial_sets_up_what_its_recorded_run_used():
    """The last steps play a run recorded with the tutorial's alias, prompt
    profile and hints: the data carry the same ones."""
    from testuary.demo import cli as demo_cli

    manifest, files = demo_cli.load_data(tutorial.DEMO_DATA)
    models = configparser.ConfigParser(interpolation=None)
    models.read_string(files["models.ini"].decode("utf-8"))
    assert manifest["invocation"]["model"] == tutorial.ALIAS
    # An Ollama alias without an address is Ollama Cloud's.
    assert tutorial.MODEL_ANSWERS["model.provider"] == "ollama-cloud"
    assert dict(models[tutorial.ALIAS]) == {"provider": "ollama",
                                            "model": tutorial.MODEL_ANSWERS["model.name"]}
    steps = _steps()
    carried = manifest["settings_files"]
    profile = steps["A prompt profile"].files[tutorial.PROFILE]
    assert json.loads(files[carried["prompt_profile"]]) == json.loads(profile)
    assert json.loads(files[carried["hints_file"]]) == _extension_hints(
        steps["Hints from an extension"].files[tutorial.HINTS_EXTENSION_PATH])


def _extension_hints(text):
    """The value of testuary_hints in an extension file's text."""
    assignment = next(node for node in ast.parse(text).body if isinstance(node, ast.Assign)
                      and node.targets[0].id == "testuary_hints")
    return ast.literal_eval(assignment.value)


def test_the_steps_are_the_project_the_alias_the_profile_and_extension_hints():
    steps = tutorial.tutorial().steps
    assert [step.title for step in steps] == ["A small project", "A model alias",
                                               "A prompt profile",
                                               "Hints from an extension",
                                               "A recorded run with these extensions",
                                               "What the recorded run used"]
    assert not steps[0].optional
    for step in steps[1:4]:
        assert step.optional and step.check is not None
        assert step.command[1:4] == ("-m", "testuary", "check")
        assert step.command[step.command.index("--model") + 1] == tutorial.ALIAS
        assert step.env[tutorial.KEY_VARIABLE] == "tutorial-placeholder"


def test_the_hints_step_writes_one_file_and_installs_nothing():
    step = _steps()["Hints from an extension"]
    assert list(step.files) == [tutorial.HINTS_EXTENSION_PATH] and step.show == (tutorial.HINTS_EXTENSION_PATH,)
    assert "PYTHONPATH" not in step.env


def _write(step, tutorial_dir):
    for name, text in step.files.items():
        path = tutorial_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")


def test_the_extension_is_named_in_the_settings_and_its_hints_found_and_shown(tmp_path):
    step = _steps()["Hints from an extension"]
    _write(step, tmp_path)
    result = step.check(tmp_path)
    assert result.ok, result.lines
    assert tuple(result.lines[:3]) == ("It wrote in .testuary/project.ini:", "[main]",
                                       "extensions = tutorial_hints.py")
    assert result.lines[-1] == "Prefer plain assert statements over assertion helpers."
    assert _ini(tmp_path / ".testuary" / "project.ini")["main"]["extensions"] == "tutorial_hints.py"


def test_invalid_extension_hints_are_reported_with_the_item(tmp_path):
    step = _steps()["Hints from an extension"]
    tutorial_dir = tmp_path / "bad"
    _write(step, tutorial_dir)
    (tutorial_dir / tutorial.HINTS_EXTENSION_PATH).write_text(
        'testuary_hints = {"coverage": {"bogus_cell": "x"}}\n', encoding="utf-8")
    result = step.check(tutorial_dir)
    assert not result.ok
    assert any("tutorial_hints.py testuary_hints" in line and "bogus_cell" in line
               for line in result.lines), result.lines


def test_the_prepared_profile_is_the_example():
    step = _steps()["A prompt profile"]
    expected = EXAMPLES.joinpath(tutorial.PROFILE).read_text(encoding="utf-8")
    assert step.files == {tutorial.PROFILE: expected}


def test_the_alias_is_written_by_the_setup_and_shown(tmp_path):
    result = _steps()["A model alias"].check(tmp_path)
    assert result.ok, result.lines
    alias = _ini(tmp_path / "models.ini")[tutorial.ALIAS]
    # The alias holds the address the setup confirmed, Ollama Cloud's.
    assert dict(alias) == {"provider": "ollama", "model": "deepseek-v4.1-flash:1017",
                           "api_url": "https://ollama.com/v1"}
    # The setup's own questions with the prepared answers, then what it wrote.
    assert "Which model service? A number from 1 to 6, or its name: ollama-cloud" in result.lines
    assert (f"A short name for this model in models.ini, which projects use (its alias): "
            f"{tutorial.ALIAS}") in result.lines
    assert result.lines[-4:] == (f"[{tutorial.ALIAS}]", "provider = ollama",
                                 "model = deepseek-v4.1-flash:1017",
                                 "api_url = https://ollama.com/v1")


def test_the_setup_contacts_nothing_and_ignores_the_learners_key(monkeypatch):
    monkeypatch.setenv(tutorial.KEY_VARIABLE, "a-real-key")
    offline = tutorial._Offline()
    assert offline.environment(tutorial.KEY_VARIABLE) == ""
    assert offline.model_list(None, "https://ollama.com/v1", "a-real-key") is None
    assert offline.local_ollama() is False


def test_the_profile_is_selected_in_the_settings_and_other_settings_stay(tmp_path):
    project = tmp_path / ".testuary" / "project.ini"
    project.parent.mkdir()
    project.write_text("# mine\n[api]\nmodel = elsewhere\n", encoding="utf-8")
    result = _steps()["A prompt profile"].check(tmp_path)
    assert result.ok and result.lines[0] == "It wrote in .testuary/project.ini:"
    text = project.read_text(encoding="utf-8")
    assert text.startswith("# mine\n")
    settings = _ini(project)
    assert settings["main"]["prompt_profile"] == tutorial.PROFILE
    assert settings["api"]["model"] == "elsewhere"


def test_the_profile_goes_into_the_tutorials_own_project(tmp_path):
    """A tutorial directory inside another project becomes a project of its
    own, and the project above it keeps its settings file."""
    outer = tmp_path / ".testuary" / "project.ini"
    outer.parent.mkdir()
    outer.write_text("[api]\nmodel = elsewhere\n", encoding="utf-8")
    tutorial_dir = tmp_path / "tutorial"
    tutorial_dir.mkdir()
    assert _steps()["A prompt profile"].check(tutorial_dir).ok
    assert outer.read_text(encoding="utf-8") == "[api]\nmodel = elsewhere\n"
    from testuary import settings

    resolved = settings.resolve_settings(base_dir=tutorial_dir, models={}, use_environment=False,
                                         home=tutorial_dir)
    assert resolved.sections["main"]["prompt_profile"] == tutorial.PROFILE


def test_the_profile_step_takes_a_relative_dir(tmp_path, monkeypatch):
    """The tutorial command's directory defaults to a relative path, which the
    engine passes to each step's check as given."""
    monkeypatch.chdir(tmp_path)
    tutorial_dir = Path("testuary-tutorial")
    tutorial_dir.mkdir()
    result = _steps()["A prompt profile"].check(tutorial_dir)
    assert result.ok and result.lines[0] == "It wrote in .testuary/project.ini:", result.lines
    assert _ini(tmp_path / tutorial_dir / ".testuary" / "project.ini")["main"]["prompt_profile"] == tutorial.PROFILE


def test_the_tutorial_ends_with_the_playback_of_its_recorded_run():
    steps = tutorial.tutorial().steps
    playback, shown = steps[-2], steps[-1]
    assert playback.command == (sys.executable, "-m", "testuary", "--demo", "--demo-data",
                                tutorial.DEMO_DATA, "--output-dir", tutorial.RUN_DIR)
    assert playback.optional and shown.optional and shown.check is not None


def _demo_record(tutorial_dir, outcome="completed"):
    import json

    run = tutorial_dir / tutorial.RUN_DIR
    run.mkdir(parents=True)
    (run / "DEMO.json").write_text(json.dumps({
        "configuration": {"model": "tutorial_model (deepseek-v4.1-flash:1017, provider ollama, "
                                   "not contacted)",
                          "target": "tenacity tenacity/stop.py",
                          "prompt profile": "prompt_profile.json (carried with the data)",
                          "hints file": "hints.json (carried with the data)"},
        "result": {"outcome": outcome, "answers": 42}}), encoding="utf-8")


def test_the_last_step_shows_what_the_recorded_run_used(tmp_path):
    check = tutorial.tutorial().steps[-1].check
    assert not check(tmp_path).ok
    _demo_record(tmp_path)
    result = check(tmp_path)
    assert result.ok
    assert result.lines == (
        "model: tutorial_model (deepseek-v4.1-flash:1017, provider ollama, not contacted)",
        "target: tenacity tenacity/stop.py",
        "prompt profile: prompt_profile.json (carried with the data)",
        "hints file: hints.json (carried with the data)",
        "The playback completed with all 42 recorded answers.")


def test_a_playback_that_stopped_is_not_done(tmp_path):
    _demo_record(tmp_path, outcome="stale")
    result = tutorial.tutorial().steps[-1].check(tmp_path)
    assert not result.ok and result.lines[-1] == "The playback ended stale."


def _private(dotted):
    # testuary.config, models and project hold Testuary's internal
    # helpers: their names are not public, with or without an underscore.
    parts = dotted.split(".")
    return parts[0] == "testuary" and (
        any(part.startswith("_") for part in parts[1:]) or parts[1:2] in (["config"], ["models"], ["project"]))


def test_the_tutorial_uses_no_private_name_of_testuary():
    source = resources.files("testuary").joinpath("tutorial", "__init__.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    found = []
    # Each name a Testuary import binds, with the module it stands for.
    modules = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if _private(alias.name):
                    found.append(alias.name)
                if alias.name.startswith("testuary"):
                    modules[(alias.asname or alias.name).split(".")[0]] = alias.name
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("testuary"):
            for alias in node.names:
                name = f"{node.module}.{alias.name}"
                if _private(name):
                    found.append(name)
                modules[alias.asname or alias.name] = name
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name) \
                and node.value.id in modules:
            name = f"{modules[node.value.id]}.{node.attr}"
            if _private(name):
                found.append(name)
    assert not found, f"the tutorial uses private names: {sorted(set(found))}"


def test_the_private_name_check_catches_each_kind():
    assert _private("testuary.agent._helpers")
    assert _private("testuary.config")
    assert _private("testuary.config.load_config")
    assert _private("testuary.settings._packaged_entries")
    assert not _private("testuary.hints.extension_hints")
    assert not _private("testuary.settings.update_file")
    assert not _private("langadapter.tutorial._helper")


def test_each_step_takes_its_title_and_first_sentences_from_the_table():
    from testuary import tutorial as extension_points

    steps = tutorial.tutorial().steps
    for step, name in zip(steps[1:], ("model_alias", "prompt_profile", "extension_hints")):
        point = extension_points.get(name)
        assert step.title == point.title
        assert step.text[0] == f"{point.what} It makes possible {point.enables}."
        assert step.text[1] == f"Where: {point.declared}. Default: {point.default}."
