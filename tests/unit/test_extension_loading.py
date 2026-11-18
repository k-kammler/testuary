"""The extensions [main] extensions names, loaded at the start of a command
that uses them through LangAdapter's loader, each file relative to the
settings file that names it, and refused naming the item, the reason and
where it is named. A run's settings, which the quick check and
testuary-web's pages also enter, load nothing."""
import re
import uuid
from types import SimpleNamespace

import pytest
from langadapter import names, unregister

from testuary import _extensions, config, orchestrator
from testuary.settings import ConfigurationError, RunSettings, extension_items, resolve_settings


def _project(tmp_path, project_ini=""):
    """A user directory with no config.ini and a project whose settings file
    holds ``project_ini``."""
    home = tmp_path / "home"
    home.mkdir()
    project = tmp_path / "project"
    (project / ".testuary").mkdir(parents=True)
    (project / ".testuary" / "project.ini").write_text(project_ini, encoding="utf-8")
    return home, project


def _files(loaded):
    return [extension.file for _, _, extension, _ in loaded]


def _marking(path, marker):
    """An extension file that writes ``marker`` when it is imported."""
    path.write_text(f"from pathlib import Path\nPath({str(marker)!r}).write_text('ran')\n",
                    encoding="utf-8")


def test_a_file_item_is_relative_to_the_settings_file_that_names_it(tmp_path):
    home, project = _project(tmp_path, "[main]\nextensions = my_hints.py\n")
    hints = project / ".testuary" / "my_hints.py"
    hints.write_text("testuary_hints = {}\n", encoding="utf-8")
    settings = resolve_settings(base_dir=project, home=home, models={}, environment={})
    loaded = _extensions.load(extension_items(settings))
    assert _files(loaded) == [str(hints.resolve())]
    assert loaded[0][3].testuary_hints == {}


def test_an_item_of_the_variable_is_relative_to_the_project_directory(tmp_path):
    home, project = _project(tmp_path)
    (project / "tools").mkdir()
    (project / "tools" / "my_hints.py").write_text("", encoding="utf-8")
    settings = resolve_settings(base_dir=project / "tools", home=home, models={},
                                environment={"TESTUARY_MAIN_EXTENSIONS": "tools/my_hints.py"})
    assert _files(_extensions.load(extension_items(settings))) == [
        str((project / "tools" / "my_hints.py").resolve())]


def test_a_module_item_is_loaded_by_its_name(tmp_path, monkeypatch):
    name = f"notes_ext_{uuid.uuid4().hex}"
    (tmp_path / f"{name}.py").write_text("testuary_hints = {}\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    home, project = _project(tmp_path, f"[main]\nextensions = {name}\n")
    settings = resolve_settings(base_dir=project, home=home, models={}, environment={})
    loaded = _extensions.load(extension_items(settings))
    assert [(item, extension.kind) for item, _, extension, _ in loaded] == [(name, "module")]


def test_one_file_named_twice_is_loaded_once_at_its_first_place(tmp_path):
    """The user's config.ini and the project's settings file name one file,
    the second time by another spelling."""
    shared = tmp_path / "shared.py"
    shared.write_text("", encoding="utf-8")
    home, project = _project(tmp_path, f"[main]\nextensions = {shared.parent}/./{shared.name}\n")
    (home / "config.ini").write_text(f"[main]\nextensions = {shared}\n", encoding="utf-8")
    settings = resolve_settings(base_dir=project, home=home, models={}, environment={})
    loaded = _extensions.load(extension_items(settings))
    assert [(item, source) for item, source, _, _ in loaded] == [
        (str(shared), str((home / "config.ini").resolve()))]


def test_a_command_refuses_an_item_that_cannot_load_naming_where_it_is_named(tmp_path):
    home, project = _project(tmp_path)
    (home / "config.ini").write_text("[main]\nextensions = notes_wbe_xyz\n", encoding="utf-8")
    message = (r"\[main\] extensions: notes_wbe_xyz: no module notes_wbe_xyz is found in this "
               rf"environment. It is named in {re.escape(str((home / 'config.ini').resolve()))}$")
    with pytest.raises(ConfigurationError, match=message):
        resolve_settings(base_dir=project, home=home, models={}, environment={}, load_extensions=True)


@pytest.mark.parametrize("environment, place", [
    ({"TESTUARY_MAIN_EXTENSIONS": "broken.py"}, "the variable TESTUARY_MAIN_EXTENSIONS"),
    ({}, "the program's overrides"),
])
def test_an_item_that_fails_on_import_is_refused_with_the_reason(tmp_path, environment, place):
    home, project = _project(tmp_path)
    (project / "broken.py").write_text("raise RuntimeError('half written')\n", encoding="utf-8")
    overrides = {} if environment else {"main": {"extensions": "broken.py"}}
    settings = resolve_settings(base_dir=project, home=home, models={}, environment=environment,
                                overrides=overrides)
    with pytest.raises(ConfigurationError, match=rf"broken\.py: RuntimeError: half written\. "
                                                 rf"It is named in {re.escape(place)}$"):
        _extensions.load(extension_items(settings))


def test_resolving_and_entering_a_runs_settings_load_nothing(tmp_path):
    """The quick check and testuary-web's pages resolve settings and enter
    them: an extension's code does not run, and one that cannot load does
    not refuse."""
    home, project = _project(tmp_path, "[main]\nextensions = marking.py, missing.py\n")
    _marking(project / ".testuary" / "marking.py", tmp_path / "ran")
    settings = resolve_settings(base_dir=project, home=home, models={}, environment={})
    with config.use_run_settings(settings):
        pass
    assert not (tmp_path / "ran").exists()


def test_without_extensions_a_commands_resolution_loads_nothing(tmp_path, monkeypatch):
    home, project = _project(tmp_path)
    monkeypatch.setattr(_extensions, "load", lambda rows: pytest.fail("loaded"))
    resolve_settings(base_dir=project, home=home, models={}, environment={}, load_extensions=True)


def test_a_commands_resolution_takes_a_language_an_extension_adds(tmp_path):
    """[project] may name an extension's language and a setting it declares
    once the command's resolution has loaded the extension."""
    language = f"toy{uuid.uuid4().hex[:8]}"
    home, project = _project(tmp_path, "[main]\nextensions = toy.py\n"
                                       f"[project]\nlanguage = {language}\ntoy_level = 3\n")
    (project / ".testuary" / "toy.py").write_text(
        "from langadapter import LangAdapter\n"
        "from langadapter.adapter import ProjectSetting\n"
        "class Toy(LangAdapter):\n"
        f"    language = {language!r}\n"
        f"    file_extensions = ('.{language}',)\n"
        "    project_settings = (ProjectSetting('toy_level', 'How toy.', 'text', default='1'),)\n"
        "langadapter_language = Toy\n", encoding="utf-8")
    try:
        settings = resolve_settings(base_dir=project, home=home, models={}, environment={},
                                    load_extensions=True)
        assert settings.sections["project"]["language"] == language
        assert settings.sections["project"]["toy_level"] == "3"
    finally:
        if language in names():
            unregister(language)


def test_a_run_loads_the_extensions_of_settings_resolved_without_them(tmp_path, monkeypatch):
    """A program's settings may come from a resolution that loaded nothing:
    the run loads them before work starts, and refuses one that cannot
    load."""
    _marking(tmp_path / "marking.py", tmp_path / "ran")
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda args, languages: 0)
    settings = RunSettings(sections={"main": {"extensions": "marking.py"}}, base_dir=tmp_path,
                           environment={})
    assert orchestrator.run_pipeline(SimpleNamespace(), settings=settings) == 0
    assert (tmp_path / "ran").read_text() == "ran"
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda args, languages: pytest.fail("ran"))
    broken = RunSettings(sections={"main": {"extensions": "missing.py"}}, base_dir=tmp_path,
                         environment={})
    with pytest.raises(ConfigurationError, match=r"missing\.py: .* It is named in the "
                                                 r"program's overrides$"):
        orchestrator.run_pipeline(SimpleNamespace(), settings=broken)


def test_the_loaded_items_are_read_without_importing(tmp_path):
    """``loaded`` lists only what a load took, each file once."""
    home, project = _project(tmp_path, "[main]\nextensions = a.py, b.py\n")
    for name in ("a.py", "b.py"):
        (project / ".testuary" / name).write_text("", encoding="utf-8")
    rows = extension_items(resolve_settings(base_dir=project, home=home, models={},
                                            environment={}))
    assert _extensions.loaded(rows) == ()
    _extensions.load_item(*rows[1])
    assert [item for item, _, _, _ in _extensions.loaded(rows)] == ["b.py"]


_PROVIDER = ("from langchain_core.language_models.fake_chat_models import FakeListChatModel\n"
             "def GatewayChat(**parameters):\n"
             "    return FakeListChatModel(responses=['answer'])\n")


def _gateway(home, chat_model):
    """The user's models.ini with the alias gate, whose chat_model is
    ``chat_model``, or none when it is empty."""
    named = f"chat_model = {chat_model}\n" if chat_model else ""
    (home / "models.ini").write_text("[gate]\nprovider = openai\nmodel = chosen\n"
                                     f"api_url = http://127.0.0.1:9/v1\n{named}", encoding="utf-8")


def test_a_provider_in_one_file_beside_models_ini(tmp_path):
    from langchain_core.language_models import BaseChatModel

    from testuary.models import build_resolved_chat_model, resolve_phase_model

    home, project = _project(tmp_path, "[api]\nmodel = gate\n")
    (home / "my_chat.py").write_text(_PROVIDER, encoding="utf-8")
    _gateway(home, "my_chat.py:GatewayChat")
    model = resolve_phase_model(resolve_settings(base_dir=project, home=home, environment={}),
                                "coverage")
    assert (model.protocol, model.chat_model_dir) == ("added", str(home.resolve()))
    assert isinstance(build_resolved_chat_model(model, api_key="key"), BaseChatModel)


def test_a_provider_named_in_a_variable_is_relative_to_the_project_directory(tmp_path):
    from testuary.models import resolve_chat_model_factory, resolve_phase_model

    home, project = _project(tmp_path, "[api]\nmodel = gate\n")
    (project / "tools").mkdir()
    (project / "tools" / "my_chat.py").write_text(_PROVIDER, encoding="utf-8")
    _gateway(home, "")
    settings = resolve_settings(base_dir=project / "tools", home=home, environment={
        "TESTUARY_OPENAI_CHAT_MODEL": "tools/my_chat.py:GatewayChat"})
    model = resolve_phase_model(settings, "coverage")
    assert model.chat_model_dir == str(project.resolve())
    assert resolve_chat_model_factory(model).__name__ == "GatewayChat"


def test_an_absolute_provider_file_splits_at_the_last_colon(tmp_path):
    from testuary.models import resolve_chat_model_factory, resolve_phase_model

    provider = tmp_path / "elsewhere" / "my_chat.py"
    provider.parent.mkdir()
    provider.write_text(_PROVIDER, encoding="utf-8")
    home, project = _project(tmp_path, "[api]\nmodel = gate\n")
    _gateway(home, f"{provider}:GatewayChat")
    model = resolve_phase_model(resolve_settings(base_dir=project, home=home, environment={}),
                                "starter")
    assert resolve_chat_model_factory(model).__name__ == "GatewayChat"


def test_the_resolved_preset_records_the_provider_file_and_its_digest(tmp_path):
    import hashlib

    home, project = _project(tmp_path, "[api]\nmodel = gate\n")
    (home / "my_chat.py").write_text(_PROVIDER, encoding="utf-8")
    _gateway(home, "my_chat.py:GatewayChat")
    settings = resolve_settings(base_dir=project, home=home, environment={})
    with config.use_run_settings(settings):
        record = config.resolved_test_preset(model_phases=("coverage",))["model"]["coverage"]
    assert record["chat_model"] == "my_chat.py:GatewayChat"
    assert record["extension_file"] == str((home / "my_chat.py").resolve())
    assert record["extension_sha256"] == hashlib.sha256((home / "my_chat.py").read_bytes()).hexdigest()
    assert record["extension_version"] is None


def test_a_provider_that_cannot_load_is_refused_naming_it(tmp_path):
    from testuary.models import resolve_chat_model_factory, resolve_phase_model

    with pytest.raises(ConfigurationError, match=r"chat_model missing_chat_xyz:Chat: missing_chat_xyz: "
                                                 r"no module missing_chat_xyz is found"):
        resolve_chat_model_factory(resolve_phase_model(RunSettings(
            sections={"api": {"model": "gate"}}, base_dir=tmp_path, environment={},
            models={"gate": {"provider": "openai", "model": "x", "api_url": "http://127.0.0.1:9/v1",
                             "chat_model": "missing_chat_xyz:Chat"}}), "coverage"))


class _Reached(Exception):
    """The command reached its work, past the resolution of its settings."""


def _command(monkeypatch, project, argv):
    """``testuary`` with ``argv`` in ``project``, its check and run replaced
    by _Reached."""
    import sys

    from testuary import __main__ as cli

    def reach(*args, **kwargs):
        raise _Reached

    monkeypatch.chdir(project)
    monkeypatch.setattr(sys, "argv", ["testuary", *argv])
    monkeypatch.setattr(orchestrator, "check", reach)
    monkeypatch.setattr(orchestrator, "run_pipeline", reach)
    with pytest.raises(_Reached):
        cli._main()


@pytest.mark.parametrize("argv, loads", [
    (["check", "--json", "--source-files", "module.py"], False),
    (["check", "--json", "--verify", "--source-files", "module.py"], True),
    (["--source-files", "module.py"], True),
])
def test_the_tool_check_and_a_run_load_the_extensions_and_the_quick_check_none(
        tmp_path, monkeypatch, argv, loads):
    home, project = _project(tmp_path, "[main]\nextensions = marking.py\n")
    _marking(project / ".testuary" / "marking.py", tmp_path / "ran")
    (project / "module.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    _command(monkeypatch, project, argv)
    assert (tmp_path / "ran").exists() is loads


def _toy_project(tmp_path, language_line):
    """A project whose settings name toy.py, which adds a language with the
    setting toy_level, and set ``language_line`` and toy_level. Returns the
    project and the language's name."""
    language = f"toy{uuid.uuid4().hex[:8]}"
    home, project = _project(tmp_path, "[main]\nextensions = toy.py\n"
                                       f"[project]\n{language_line.format(language)}\ntoy_level = 3\n")
    (project / ".testuary" / "toy.py").write_text(
        "from langadapter import LangAdapter\n"
        "from langadapter.adapter import ProjectSetting\n"
        "class Toy(LangAdapter):\n"
        f"    language = {language!r}\n"
        f"    file_extensions = ('.{language}',)\n"
        "    project_settings = (ProjectSetting('toy_level', 'How toy.', 'text', default='1'),)\n"
        "langadapter_language = Toy\n", encoding="utf-8")
    return home, project, language


def test_provision_accepts_a_language_an_extension_adds(tmp_path, monkeypatch):
    from testuary import __main__ as cli
    from testuary import languages

    home, project, language = _toy_project(tmp_path, "language = {}")
    monkeypatch.chdir(project)
    monkeypatch.setattr(languages, "setup_context", lambda name, settings: SimpleNamespace(
        interpreter=SimpleNamespace(provisioning=None)))
    try:
        assert cli._cmd_provision([]) == 0
    finally:
        if language in names():
            unregister(language)


def test_settings_that_load_nothing_leave_an_extensions_language_unchecked(tmp_path):
    """The quick check and testuary-web's pages resolve without loading:
    the language and its setting are kept and named, and nothing runs."""
    home, project, language = _toy_project(tmp_path, "language = {}")
    settings = resolve_settings(base_dir=project, home=home, models={}, environment={})
    assert settings.sections["project"]["language"] == language
    assert settings.unchecked == ("[project] language", "[project] toy_level")
    assert language not in names()


def test_a_misspelled_language_is_refused_once_the_extensions_load(tmp_path, monkeypatch):
    """The tool check and a run load the extensions and check fully, the
    run also for settings a program resolved without loading."""
    home, project, language = _toy_project(tmp_path, "language = {}x")
    try:
        with pytest.raises(ValueError, match=rf"{language}x is not installed"):
            resolve_settings(base_dir=project, home=home, models={}, environment={},
                             load_extensions=True)
        settings = resolve_settings(base_dir=project, home=home, models={}, environment={})
        monkeypatch.setattr(orchestrator, "_run_pipeline", lambda args, languages: pytest.fail("ran"))
        with pytest.raises(ConfigurationError, match=rf"{language}x is not installed"):
            orchestrator.run_pipeline(SimpleNamespace(), settings=settings)
    finally:
        if language in names():
            unregister(language)


def test_settings_without_extensions_stay_strict(tmp_path):
    home, project = _project(tmp_path, "[project]\nlanguage = toynone\n")
    with pytest.raises(ValueError, match="toynone is not installed"):
        resolve_settings(base_dir=project, home=home, models={}, environment={})


def test_a_replaced_resolution_keeps_where_each_item_is_named(tmp_path, monkeypatch):
    """A program that replaces a field of resolved settings before a run
    keeps each item's source and directory. An item a replace adds counts
    as an override."""
    import dataclasses

    home, project = _project(tmp_path)
    (home / "config.ini").write_text("[main]\nextensions = user_ext.py\n", encoding="utf-8")
    _marking(home / "user_ext.py", tmp_path / "ran")
    settings = resolve_settings(base_dir=project, home=home, models={}, environment={})
    replaced = dataclasses.replace(settings, environment={"PATH": "x"})
    user = (home / "config.ini").resolve()
    assert extension_items(replaced) == (("user_ext.py", str(user), str(user.parent)),)
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda args, languages: 0)
    assert orchestrator.run_pipeline(SimpleNamespace(), settings=replaced) == 0
    assert (tmp_path / "ran").exists()
    added = dataclasses.replace(settings, sections={
        **settings.sections, "main": {**settings.sections["main"], "extensions": "user_ext.py, more.py"}})
    assert [(item, source) for item, source, _ in extension_items(added)] == [
        ("user_ext.py", "overrides"), ("more.py", "overrides")]


def test_a_run_s_settings_capture_no_hints_even_when_they_are_invalid(tmp_path):
    """Once a process has loaded an extension with invalid hints, entering
    a run's settings, as the quick check and testuary-web's pages do,
    neither captures nor refuses them: a run does at its start."""
    home, project = _project(tmp_path, "[main]\nextensions = bad.py\n")
    (project / ".testuary" / "bad.py").write_text("testuary_hints = {'initail': {}}\n",
                                                 encoding="utf-8")
    settings = resolve_settings(base_dir=project, home=home, models={}, environment={})
    _extensions.load(extension_items(settings))
    with config.use_run_settings(settings):
        assert config.current_run().extension_hints == ()


_TEMPLATES = {"system_prompt": "s", "user_prompt_template": "u"}


def test_a_resume_compares_a_provider_files_digest_and_names_it(tmp_path):
    """The generation condition records the SHA-256 of a provider in one
    file where an installed provider records its version, and no resolved
    path, so a moved project still resumes. An edited file refuses."""
    import dataclasses
    import json

    from testuary.evidence import current_condition, reuse_problems

    home, project = _project(tmp_path, "[api]\nmodel = gate\n")
    (home / "my_chat.py").write_text(_PROVIDER, encoding="utf-8")
    _gateway(home, "my_chat.py:GatewayChat")
    with config.use_run_settings(resolve_settings(base_dir=project, home=home, environment={})):
        condition = current_condition("coverage", "python", str(project), templates=_TEMPLATES)
    import hashlib
    assert condition.model["extension_sha256"] == hashlib.sha256((home / "my_chat.py").read_bytes()).hexdigest()
    assert str(home.resolve()) not in json.dumps(condition.export())
    assert reuse_problems(condition.export(), condition) == ()
    edited = dataclasses.replace(condition, model={**condition.model, "extension_sha256": "0" * 64})
    assert "the client factory's file my_chat.py changed" in reuse_problems(condition.export(), edited)


def test_a_resume_compares_a_language_files_digest_and_names_it(tmp_path):
    import dataclasses

    from testuary.evidence import language_package, reuse_problems
    from testuary.evidence import capture_condition
    from testuary.models import resolve_phase_model

    home, project, language = _toy_project(tmp_path, "language = {}")
    try:
        settings = resolve_settings(base_dir=project, home=home, environment={}, load_extensions=True, models={
            "m": {"provider": "openai", "model": "x", "api_url": "http://127.0.0.1:9/v1"}},
            overrides={"api": {"model": "m"}})
        package = language_package(language)
        assert package["item"] == "toy.py" and package["version"] is None and package["module"] is None
        assert len(package["sha256"]) == 64 and str(project.resolve()) not in str(package)
        model = resolve_phase_model(settings, "coverage")
        condition = capture_condition(model, language=language, templates=_TEMPLATES,
                                      language_package=package)
        assert reuse_problems(condition.export(), condition) == ()
        edited = dataclasses.replace(condition, language_package={**package, "sha256": "0" * 64})
        assert "the language file toy.py changed" in reuse_problems(condition.export(), edited)
    finally:
        if language in names():
            unregister(language)


def test_a_provider_files_load_error_keeps_no_credential(tmp_path, monkeypatch):
    """The resolved settings record and the generation condition load the
    provider too: a load error there leaves the model's key out, as the
    client's construction does."""
    from testuary.evidence import current_condition

    home, project = _project(tmp_path, "[api]\nmodel = gate\n")
    (home / "leaking.py").write_text("import os\nraise RuntimeError('leaked ' + os.environ['GATE_KEY'])\n",
                                     encoding="utf-8")
    (home / "models.ini").write_text("[gate]\nprovider = openai\nmodel = chosen\napi_url = "
                                     "http://127.0.0.1:9/v1\napi_key_env = GATE_KEY\n"
                                     "chat_model = leaking.py:Chat\n", encoding="utf-8")
    monkeypatch.setenv("GATE_KEY", "key-sentinel-7")
    settings = resolve_settings(base_dir=project, home=home, environment={"GATE_KEY": "key-sentinel-7"})
    with config.use_run_settings(settings):
        for load in (lambda: config.resolved_test_preset(model_phases=("coverage",)),
                     lambda: current_condition("coverage", "python", str(project), templates=_TEMPLATES)):
            with pytest.raises(ConfigurationError, match="RuntimeError: leaked") as refused:
                load()
            assert "key-sentinel-7" not in str(refused.value)
