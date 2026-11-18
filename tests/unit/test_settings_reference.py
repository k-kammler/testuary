"""One description of every setting (settings.reference), shared with the
settings a language declares, each resolved value with its source, and a
change of a settings file written only when the result validates."""
from __future__ import annotations

import os
import stat

import langadapter as lb
import pytest
from langadapter.adapter import ProjectSetting

from testuary import settings
from testuary.settings import ConfigurationError, reference, resolve_settings, update_file


class Settled(lb.LangAdapter):
    language = "settled"
    file_extensions = (".stl",)
    project_settings = (
        ProjectSetting("settled_mode", "How Settled runs its tests.", "choice", default="fast",
                       choices=("fast", "slow"), check="one of the choices"),
        ProjectSetting("python_executable", "Settled's own words for an interpreter.", "text"),
    )


@pytest.fixture
def settled():
    lb.register("settled", [".stl"], Settled)
    try:
        yield
    finally:
        lb.unregister("settled")


def _described(sections):
    return {(setting.section, setting.name): setting
            for section in sections for setting in section.settings}


def test_the_reference_is_the_packaged_configuration_with_its_comments():
    sections = reference(languages=("python", "java"))
    packaged = settings._packaged_entries()
    assert [section.name for section in sections] == [name for name, _, _ in packaged]
    assert [section.purpose for section in sections] == [purpose for _, purpose, _ in packaged]
    described = _described(sections)
    model = described[("api", "model")]
    assert model.purpose.startswith("A model alias of models.ini") and model.default == ""
    assert (model.kind, model.languages) == ("text", ())
    assert described[("main", "context_rebuild")].choices == ("mode_change", "every_turn")
    assert described[("main", "phase_end_mutation")].kind == "boolean"
    assert described[("coverage", "max_requests_per_phase")].check == "a number, or unlimited"
    assert described[("limits", "read_chars")].kind == "number"
    # A project setting a language declares takes the declaration's kind.
    interpreter = described[("project", "python_executable")]
    assert (interpreter.kind, interpreter.languages) == ("executable", ("python",))
    assert interpreter.purpose.startswith("The project's Python interpreter")


#: The declared settings whose config.ini text is Testuary's own. Testuary
#: hands LangAdapter its source directory as the project, so LangAdapter's
#: "below the project" would name a directory other than the one Testuary's
#: pages call the project directory. LangAdapter's "the caller" is
#: Testuary, which Testuary's text names.
_OWN_TEXTS = {"python_import_root", "python_executable"}


def test_a_declared_settings_comment_is_its_declaration():
    """config.ini's comment of a setting an installed language declares is
    the declaration's text, so the reference and the settings view show one
    text. Only the settings of _OWN_TEXTS differ."""
    from langadapter import names, registration

    comments = {key: comment for _, _, entries in settings._packaged_entries()
                for key, _, comment in entries}
    differing = {setting.name for language in names()
                 for setting in getattr(registration(language).factory, "project_settings", ())
                 if setting.name in comments and comments[setting.name] != setting.purpose}
    assert differing == _OWN_TEXTS


def test_a_declared_setting_joins_the_project_section(settled):
    described = _described(reference())
    mode = described[("project", "settled_mode")]
    assert (mode.purpose, mode.kind, mode.default, mode.choices, mode.check, mode.languages) == (
        "How Settled runs its tests.", "choice", "fast", ("fast", "slow"),
        "one of the choices", ("settled",))
    # Settled declares Python's interpreter too: the setting names both
    # languages, and the first declaration's kind and words stay.
    interpreter = described[("project", "python_executable")]
    assert {"python", "settled"} <= set(interpreter.languages)
    assert interpreter.kind == "executable"
    assert interpreter.purpose.startswith("The project's Python interpreter")
    assert ("project", "settled_mode") not in _described(reference(languages=("python",)))


def test_a_setting_several_languages_declare_is_used_by_each(settled, tmp_path):
    """A project setting two languages declare counts for a project in
    either language, and only another language's project leaves it unused."""
    for language, active in (("settled", True), ("python", True), ("java", False)):
        resolved = resolve_settings(base_dir=tmp_path, home=tmp_path, environment={}, models={},
                                    overrides={"project": {"language": language,
                                                           "python_executable": "py"}})
        values = {(item.setting.section, item.setting.name): item
                  for item in settings.setting_values(resolved)}
        assert values[("project", "python_executable")].active is active, language


def test_each_value_names_its_source(tmp_path):
    config = tmp_path / "config.ini"
    config.write_text("[main]\nseed = 7\n", encoding="utf-8")
    resolved = resolve_settings(base_dir=tmp_path, home=tmp_path, models={},
                                environment={"TESTUARY_MAIN_CONTEXT_REBUILD": "every_turn"},
                                overrides={"coverage": {"max_requests_per_phase": 5}})
    values = {(item.setting.section, item.setting.name): item
              for item in settings.setting_values(resolved, languages=("python", "java"))}
    assert (values[("main", "seed")].value, values[("main", "seed")].source) == (
        "7", str(config.resolve()))
    assert values[("main", "context_rebuild")].source == "environment"
    assert (values[("coverage", "max_requests_per_phase")].value,
            values[("coverage", "max_requests_per_phase")].source) == ("5", "overrides")
    assert values[("main", "phase_end_mutation")].source == "default"


def test_sections_and_settings_have_plain_names_and_unused_values_say_why(tmp_path):
    """A settings view's group titles and labels, and the settings the run
    does not use: another provider's service, another mutation engine's
    options, another language's settings."""
    sections = {section.name: section for section in reference(languages=("python", "java"))}
    assert (sections["coverage"].title, sections["openai"].title) == ("Coverage phase", "OpenAI service")
    assert _described(sections.values())[("main", "context_rebuild")].label == "Context rebuild"
    assert _described(sections.values())[("openai", "api_url")].label == "API URL"
    # A name's short form is written out: dir is directory.
    assert (_described(sections.values())[("project", "source_dir")].label,
            _described(sections.values())[("project", "output_dir")].label) == (
        "Source directory", "Output directory")
    # A phase's limits are per phase, as their names and the run plan say.
    assert {_described(sections.values())[(section, name)].label
            for section in ("coverage", "mutation")
            for name in ("max_requests_per_phase", "max_wall_clock_seconds_per_phase")} == {
        "Request limit (model requests per phase)", "Time limit per phase (seconds, 0 for none)"}
    resolved = resolve_settings(
        base_dir=tmp_path, home=tmp_path, environment={},
        models={"local": {"provider": "ollama", "model": "m"}},
        overrides={"api": {"model": "local"}, "mutation": {"engine": "pit"},
                   "project": {"language": "java"}})
    values = {(item.setting.section, item.setting.name): item
              for item in settings.setting_values(resolved, languages=("python", "java"))}
    assert values[("ollama", "api_url")].active
    assert (values[("openai", "api_url")].active, values[("openai", "api_url")].inactive) == (
        False, "the model's provider is ollama")
    cosmic = next(item for key, item in values.items() if key[1].startswith("cosmic_ray_"))
    assert (cosmic.active, cosmic.inactive) == (False, "the mutation engine is pit")
    assert values[("project", "java_home")].active
    assert values[("project", "python_executable")].active is False
    assert values[("project", "python_executable")].inactive == "the project's language is java"
    # Without a model or language set, nothing is grayed for them.
    plain = resolve_settings(base_dir=tmp_path, home=tmp_path, environment={}, models={})
    assert all(item.active for item in settings.setting_values(plain, languages=("python", "java"))
               if item.setting.section in ("openai", "project"))


def test_an_update_keeps_every_other_line(tmp_path):
    config = tmp_path / "config.ini"
    config.write_bytes(b"# Mine.\r\n[main]\r\n# the seed\r\nseed = 1\r\nprompt_profile = old\r\n")
    update_file(config, "main", {"seed": 3}, remove=["prompt_profile"])
    assert config.read_bytes() == b"# Mine.\r\n[main]\r\n# the seed\r\nseed = 3\r\n"
    update_file(config, "coverage", {"max_requests_per_phase": "unlimited"})
    assert config.read_bytes().endswith(b"\r\n[coverage]\r\nmax_requests_per_phase = unlimited\r\n")
    created = tmp_path / "new" / "config.ini"
    created.parent.mkdir()
    update_file(created, "main", {"seed": 2})
    assert created.read_text(encoding="utf-8") == "[main]\nseed = 2\n"


def test_an_update_creates_a_projects_settings_file_and_its_dir(tmp_path):
    """A setting of a project without a Testuary directory yet: the update
    creates the directory and the file, which runs there then read."""
    target = settings.file_locations(tmp_path).project_settings
    update_file(target, "main", {"prompt_profile": "profile.json"})
    assert target.read_text(encoding="utf-8") == "[main]\nprompt_profile = profile.json\n"
    resolved = resolve_settings(base_dir=tmp_path, environment={})
    assert resolved.main.prompt_profile == "profile.json"


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_a_created_file_takes_the_umask_without_changing_it(tmp_path, monkeypatch):
    """The web server's request threads write settings files while other
    threads create files and start runs, and the umask belongs to the whole
    process, so a write never sets it, not even for a moment. An existing
    file keeps its mode."""
    umask = os.umask
    changes = []
    monkeypatch.setattr(os, "umask", lambda mask: changes.append(mask) or umask(mask))
    target = tmp_path / "config.ini"
    previous = umask(0o027)
    try:
        update_file(target, "main", {"seed": 2})
        created = stat.S_IMODE(target.stat().st_mode)
        target.chmod(0o604)
        update_file(target, "main", {"seed": 3})
    finally:
        umask(previous)
    assert changes == []
    assert created == 0o640
    assert stat.S_IMODE(target.stat().st_mode) == 0o604


@pytest.mark.parametrize("section, values, message", [
    ("main", {"sed": "1"}, "unknown setting 'sed'"),
    ("mian", {"seed": "1"}, "unknown section 'mian'"),
    ("main", {"seed": "many"}, "expected a whole number"),
    ("main", {"seed": "1\n[api]"}, "one line"),
    ("main", {"seed": None}, "None is not a setting value"),
])
def test_an_invalid_result_is_refused_and_the_file_kept(tmp_path, section, values, message):
    config = tmp_path / "config.ini"
    original = b"# Mine.\n[main]\nseed = 1\n"
    config.write_bytes(original)
    with pytest.raises(ConfigurationError, match=message):
        update_file(config, section, values)
    assert config.read_bytes() == original
    assert [path.name for path in tmp_path.iterdir()] == ["config.ini"]


def test_a_file_that_is_already_invalid_is_not_written(tmp_path):
    config = tmp_path / "config.ini"
    config.write_bytes(b"[main]\nnot_a_setting = 1\n")
    with pytest.raises(ConfigurationError, match="not_a_setting"):
        update_file(config, "main", {"seed": 2})
    assert config.read_bytes() == b"[main]\nnot_a_setting = 1\n"
