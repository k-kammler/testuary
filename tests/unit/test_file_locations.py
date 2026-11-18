"""Testuary's two file locations: the per-user dir with the model
catalogue and global settings, and the project's ``.testuary/`` dir with
project.ini and the results, found from the current dir upward. One
function names them and the resolver reads through it, with each value's
source."""
from __future__ import annotations

import os
import re
from pathlib import Path

import pytest

from testuary import settings
from testuary import config
from testuary import project as project_module
from testuary.settings import file_locations, resolve_settings
from test_output_admission import directory_alias as directory_alias

#: The real function, which the suite's conftest replaces for every test.
PLATFORM_USER_DIR = settings.platform_user_dir


@pytest.fixture
def home(tmp_path):
    user_dir = tmp_path / "home"
    user_dir.mkdir()
    return user_dir


@pytest.fixture
def project(tmp_path):
    project_dir = tmp_path / "project"
    (project_dir / ".testuary").mkdir(parents=True)
    return project_dir


def test_the_user_dir_holds_the_catalogue_and_the_global_configuration(tmp_path, home):
    locations = file_locations(tmp_path, environment={"TESTUARY_HOME": str(home)})
    assert locations.user == home.resolve()
    assert (locations.config, locations.models) == (home.resolve() / "config.ini",
                                                    home.resolve() / "models.ini")
    assert file_locations(tmp_path, home=tmp_path / "other",
                          environment={"TESTUARY_HOME": str(home)}).user == (tmp_path / "other").resolve()


def test_named_catalogue_files_are_read_in_place_of_the_user_dirs(home, project):
    """models_files names the catalogue files read, a relative one from
    base_dir, and the user directory's models.ini is not read."""
    (home / "models.ini").write_text("[local_qwen]\nprovider = openai\nmodel = home-model\n",
                                     encoding="utf-8")
    (project / "own-models.ini").write_text(
        "[local_qwen]\nprovider = openai\nmodel = named-model\n", encoding="utf-8")
    resolved = resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)},
                                models_files=["own-models.ini"])
    assert resolved.models["local_qwen"]["model"] == "named-model"
    assert str(project.resolve() / "own-models.ini") in resolved.configuration_files
    assert str(home.resolve() / "models.ini") not in resolved.configuration_files
    with config.use_run_settings(resolved):
        assert config._ABSENT_MODELS_FILE_KEY not in config.resolved_test_preset()


def test_a_later_catalogue_files_section_replaces_an_earlier_one_whole(tmp_path, home, project):
    """An alias a later file defines anew holds only that file's keys, with
    no key of the earlier file's section, its api_key_env included. An alias
    the later file does not define comes from the earlier file, and a
    provider section is replaced whole too. Files outside the project
    directory set addresses and key variables."""
    lists = tmp_path / "lists"
    lists.mkdir()
    (lists / "imported.ini").write_text(
        "[local_qwen]\nprovider = openai\nmodel = first-model\napi_key_env = FIRST_KEY\n"
        "temperature = 0.3\n[cloud_coder]\nprovider = openai\nmodel = coder\n"
        "[openai]\napi_url = http://first.invalid/v1\nmax_output_tokens = 100\n",
        encoding="utf-8")
    (lists / "own-models.ini").write_text(
        "[local_qwen]\nprovider = openai\nmodel = second-model\n"
        "api_url = http://second.invalid/v1\n[openai]\napi_url = http://provider.invalid/v1\n",
        encoding="utf-8")
    resolved = resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)},
                                models_files=[lists / "imported.ini", "../lists/own-models.ini"])
    assert dict(resolved.models["local_qwen"]) == {
        "provider": "openai", "model": "second-model", "api_url": "http://second.invalid/v1"}
    assert resolved.models["cloud_coder"]["model"] == "coder"
    assert dict(resolved.models["openai"]) == {"api_url": "http://provider.invalid/v1"}


@pytest.mark.parametrize("text, named", [
    ("[local_qwen]\nprovider = openai\nmodel = m\napi_url = http://elsewhere.invalid/v1\n",
     "[local_qwen] api_url decides"),
    ("[openai]\napi_url = http://elsewhere.invalid/v1\n", "[openai] api_url decides"),
    ("[openai]\napi_key_env = HOME_SECRET\n", "[openai] api_key_env decides"),
    ("[local_qwen]\nprovider = openai\nmodel = m\napi_url = http://elsewhere.invalid/v1\n"
     "[openai]\napi_key_env = HOME_SECRET\n", "[local_qwen] api_url and [openai] api_key_env decide"),
], ids=["an-alias-address", "a-provider-address", "a-key-variable", "both"])
@pytest.mark.parametrize("place", ["project/inner/models.ini", "alias/models.ini"],
                         ids=["a-subdir", "a-link-into-the-project"])
def test_a_catalogue_file_in_the_project_dir_may_not_set_where_a_key_goes(
        tmp_path, home, project, directory_alias, text, named, place):
    """A catalogue file in the project directory, a subdirectory or a link into it
    included, that sets api_url or api_key_env is refused before any
    request, naming the file and each value, as the project's settings
    file may not set an address."""
    (project / "inner").mkdir()
    (project / "inner" / "models.ini").write_text(text, encoding="utf-8")
    directory_alias(tmp_path / "alias", project / "inner")
    with pytest.raises(settings.ConfigurationError, match=re.escape(named)) as refused:
        resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)},
                         models_files=[tmp_path / place])
    assert str(tmp_path / place) in str(refused.value)


def test_the_user_dirs_catalogue_in_the_project_dir_sets_where_a_key_goes(project):
    """The user directory's models.ini sets addresses and key variables also
    where the user directory lies in the project directory."""
    home = project / "own-home"
    home.mkdir()
    (home / "models.ini").write_text("[openai]\napi_url = http://own.invalid/v1\napi_key_env = OWN\n",
                                     encoding="utf-8")
    resolved = resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)})
    assert resolved.models["openai"]["api_url"] == "http://own.invalid/v1"
    named = resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)},
                             models_files=["own-home/models.ini"])
    assert named.models["openai"]["api_key_env"] == "OWN"


def test_the_home_dir_is_no_project_for_catalogue_files(tmp_path, home, monkeypatch):
    """A run started in the home directory, which is never a project, reads a
    catalogue file there that sets an address."""
    (tmp_path / "lists.ini").write_text("[openai]\napi_url = http://own.invalid/v1\n", encoding="utf-8")
    monkeypatch.setattr(settings, "_home_dir", lambda: tmp_path.resolve())
    resolved = resolve_settings(base_dir=tmp_path, environment={"TESTUARY_HOME": str(home)},
                                models_files=["lists.ini"])
    assert resolved.models["openai"]["api_url"] == "http://own.invalid/v1"


def test_a_settings_file_is_checked_for_the_addresses_a_project_may_not_set(tmp_path):
    """check_project_settings_file refuses a settings file that sets [api]
    api_url or a provider's api_url wherever it lies, naming the file and
    each address. A file without one and a missing file pass."""
    named = tmp_path / "elsewhere" / "testuary.ini"
    named.parent.mkdir()
    named.write_text("[api]\napi_url = http://a.invalid/v1\nmodel = m\n[openai]\napi_url = http://b.invalid/v1\n",
                     encoding="utf-8")
    with pytest.raises(settings.ConfigurationError, match=re.escape(
            f"{named}: [api] api_url and [openai] api_url decide where the model's key goes")):
        settings.check_project_settings_file(named)
    named.write_text("[api]\nmodel = m\n", encoding="utf-8")
    settings.check_project_settings_file(named)
    settings.check_project_settings_file(str(tmp_path / "missing.ini"))


def test_a_catalogue_file_is_checked_for_where_a_key_goes_wherever_it_lies(tmp_path):
    """check_project_catalogue_file refuses a catalogue file that sets
    api_url or api_key_env in any section, outside a project too, naming
    the file and each value. A file without one and a missing file pass."""
    named = tmp_path / "elsewhere" / "models.ini"
    named.parent.mkdir()
    named.write_text("[openai]\napi_key_env = SECRET\n[gw]\nprovider = openai\nmodel = m\n"
                     "api_url = http://gw.invalid/v1\n", encoding="utf-8")
    with pytest.raises(settings.ConfigurationError, match=re.escape(
            f"{named}: [openai] api_key_env and [gw] api_url decide where the model's key goes")):
        settings.check_project_catalogue_file(str(named))
    named.write_text("[gw]\nprovider = openai\nmodel = m\n", encoding="utf-8")
    settings.check_project_catalogue_file(named)
    settings.check_project_catalogue_file(tmp_path / "missing.ini")


def test_an_empty_list_of_catalogue_files_reads_no_catalogue(home, project):
    (home / "models.ini").write_text("[local_qwen]\nprovider = openai\nmodel = m\n",
                                     encoding="utf-8")
    resolved = resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)},
                                models_files=[])
    assert not resolved.models
    assert str(home.resolve() / "models.ini") not in resolved.configuration_files


def test_an_absent_last_catalogue_file_reads_as_empty_and_is_recorded(home, project):
    """The last file of models_files may not be written yet: it reads as no
    sections, and the run's settings record names it."""
    (project / "imported.ini").write_text("[local_qwen]\nprovider = openai\nmodel = m\n",
                                          encoding="utf-8")
    resolved = resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)},
                                models_files=["imported.ini", "own-models.ini"])
    assert resolved.models["local_qwen"]["model"] == "m"
    with config.use_run_settings(resolved):
        record = config.resolved_test_preset()
    assert record[config._ABSENT_MODELS_FILE_KEY] == str(project.resolve() / "own-models.ini")


def test_an_absent_earlier_catalogue_file_is_refused_naming_it(home, project):
    (project / "own-models.ini").write_text("", encoding="utf-8")
    with pytest.raises(settings.ConfigurationError, match="nowhere-models.ini"):
        resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)},
                         models_files=["nowhere-models.ini", "own-models.ini"])


def test_a_single_catalogue_path_is_refused(home, project):
    (project / "own-models.ini").write_text("", encoding="utf-8")
    with pytest.raises(TypeError, match="models_files must be an ordered sequence of paths"):
        resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)},
                         models_files="own-models.ini")


def test_a_parsed_catalogue_and_catalogue_files_together_are_refused(home, project):
    (project / "own-models.ini").write_text("", encoding="utf-8")
    with pytest.raises(TypeError, match="models and models_files each name the catalogue"):
        resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)},
                         models={}, models_files=["own-models.ini"])


def test_the_run_paths_refusal_without_catalogue_files_keeps_its_text(home, project):
    """A guard that passes before and after: without models_files a run's
    refusal of a model that is no alias names the user directory's models.ini
    in the words it had before models_files existed."""
    from testuary import models

    (home / "models.ini").write_text("", encoding="utf-8")
    resolved = resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)},
                                overrides={"api": {"model": "gpt-4o"}})
    with pytest.raises(settings.ConfigurationError) as caught:
        models.resolve_model_identity(resolved)
    assert str(caught.value) == (
        f"model 'gpt-4o' is not an alias in {home.resolve() / 'models.ini'}, and no [api] "
        "provider is set: add the alias to that file, or set [api] provider for a model name "
        "the provider serves")


def test_without_testuary_home_the_user_dir_is_the_one_platformdirs_names(tmp_path, monkeypatch):
    import platformdirs

    monkeypatch.setattr(settings, "platform_user_dir", PLATFORM_USER_DIR)
    expected = Path(platformdirs.user_config_dir("testuary", appauthor=False, roaming=True))
    assert settings.platform_user_dir() == expected
    assert file_locations(tmp_path, environment={}).user == expected.resolve()


def test_the_project_is_found_from_the_current_dir_upward(home, project):
    inner = project / "src" / "pkg"
    inner.mkdir(parents=True)
    locations = file_locations(inner, environment={"TESTUARY_HOME": str(home)})
    assert locations.project == project.resolve()
    assert locations.testuary == project.resolve() / ".testuary"
    assert locations.project_settings == project.resolve() / ".testuary" / "project.ini"
    assert locations.results == project.resolve() / ".testuary" / "results"


def test_without_a_testuary_dir_the_current_dir_is_the_project(tmp_path, home):
    plain = tmp_path / "plain"
    plain.mkdir()
    locations = file_locations(plain, environment={"TESTUARY_HOME": str(home)})
    assert locations.project == plain.resolve()
    assert locations.project_settings == plain.resolve() / ".testuary" / "project.ini"


def test_the_user_dir_is_never_a_projects_testuary_dir(tmp_path):
    user = tmp_path / ".testuary"
    user.mkdir()
    inner = tmp_path / "work"
    inner.mkdir()
    locations = file_locations(inner, environment={"TESTUARY_HOME": str(user)})
    assert locations.project == inner.resolve()


def test_the_testuary_dir_name_is_a_setting(tmp_path, home):
    project = tmp_path / "project"
    (project / "build" / "testuary").mkdir(parents=True)
    inner = project / "src"
    inner.mkdir()
    locations = file_locations(inner, environment={"TESTUARY_HOME": str(home)},
                               testuary_dir="build/testuary")
    assert locations.testuary == project.resolve() / "build" / "testuary"
    (home / "config.ini").write_text("[main]\ntestuary_dir = build/testuary\n", encoding="utf-8")
    resolved = resolve_settings(base_dir=inner, environment={"TESTUARY_HOME": str(home)})
    assert resolved.project_dir == project.resolve()
    assert resolved.results_dir == project.resolve() / "build" / "testuary" / "results"
    for refused in ("/elsewhere", "../outside", "a/../../b", "", ".", "./", "a/.", "a//b",
                    "~/x", "~", "C:x", "C:/x", "\\\\srv\\share", "\\x"):
        with pytest.raises(ValueError, match="testuary_dir"):
            resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)},
                             overrides={"main": {"testuary_dir": refused}})


def test_other_programs_find_the_configured_testuary_dir(tmp_path, home):
    """project_locations finds the Testuary directory by its configured name, as
    a run does, so the demonstration names the settings file a run reads."""
    from testuary.demo.cli import _normal_run_files
    from testuary.settings import project_locations

    project = tmp_path / "project"
    (project / "build" / "testuary").mkdir(parents=True)
    environment = {"TESTUARY_HOME": str(home)}
    assert project_locations(project, environment=environment).testuary == project.resolve() / ".testuary"
    (home / "config.ini").write_text("[main]\ntestuary_dir = build/testuary\n", encoding="utf-8")
    located = project_locations(project, environment=environment)
    assert located.project_settings == resolve_settings(
        base_dir=project, environment=environment).project_dir / "build" / "testuary" / "project.ini"
    assert located.results == project.resolve() / "build" / "testuary" / "results"
    assert project_locations(project, environment={
        **environment, "TESTUARY_MAIN_TESTUARY_DIR": "other"}).testuary == project.resolve() / "other"
    from argparse import Namespace

    assert (located.project_settings, True) in _normal_run_files(
        project, environment, False, Namespace())


def test_the_setup_names_the_settings_file_from_its_own_user_dir(tmp_path, home):
    """The Testuary directory name comes from the setup's own home, which the
    setup demonstration sets to its output directory."""
    from testuary import setup as _setup

    (home / "config.ini").write_text("[main]\ntestuary_dir = .other\n", encoding="utf-8")
    project_dir = tmp_path / "p"
    project_dir.mkdir()
    _, path = _setup._project_settings_file(project_dir, own=True, home=home)
    assert path == project_dir.resolve() / ".other" / "project.ini"


def test_command_line_paths_are_taken_from_the_working_dir(home, project, monkeypatch):
    """Command-line paths resolve from the working directory (sub/out, as the
    help says), the project's settings file from the project directory, and a
    container's POSIX root stays as given."""
    from testuary import orchestrator
    from testuary.__main__ import _build_parser

    sub = project / "sub"
    sub.mkdir()
    (project / ".testuary" / "project.ini").write_text("[main]\nhints_file = tips.json\n",
                                                       encoding="utf-8")
    monkeypatch.chdir(sub)
    args = _build_parser().parse_args(["--output-dir", "out", "--source-dir", "code",
                                       "--python-executable", "venv/bin/python",
                                       "--prompt-profile", "profile.json"])
    assert orchestrator._cli_settings_overrides(args)["project"]["output_dir"] == "out"
    overrides = orchestrator._from_working_dir(orchestrator._cli_settings_overrides(args))
    resolved = resolve_settings(overrides=overrides, environment={"TESTUARY_HOME": str(home)})
    assert resolved.project_dir == project.resolve()
    project_section = resolved.sections["project"]
    assert Path(project_section["output_dir"]) == sub.resolve() / "out"
    assert Path(project_section["source_dir"]) == sub.resolve() / "code"
    assert Path(project_section["python_executable"]) == sub.resolve() / "venv" / "bin" / "python"
    assert Path(resolved.sections["main"]["prompt_profile"]) == sub.resolve() / "profile.json"
    assert resolved.sections["main"]["hints_file"] == "tips.json"
    container = _build_parser().parse_args(["--source-dir", "/repo", "--output-dir", ""])
    assert orchestrator._from_working_dir(orchestrator._cli_settings_overrides(container))["project"] == {
        "source_dir": "/repo", "output_dir": ""}


def test_a_run_with_its_settings_given_can_read_no_stray_project_file(tmp_path, home):
    """A preset run below a project takes the project's settings file unless
    project_settings=False: then the preset decides and the start directory is
    the project."""
    repository = tmp_path / "repo"
    (repository / ".testuary").mkdir(parents=True)
    (repository / ".testuary" / "project.ini").write_text(
        "[coverage]\nmax_requests_per_phase = 7\n", encoding="utf-8")
    start = repository / "scripts"
    start.mkdir()
    preset = tmp_path / "preset.ini"
    preset.write_text("[coverage]\nmax_requests_per_phase = 100\n", encoding="utf-8")
    environment = {"TESTUARY_HOME": str(home)}
    found = resolve_settings(config_files=[preset], base_dir=start, environment=environment)
    assert (found.sections["coverage"]["max_requests_per_phase"], found.project_dir) == (
        "7", repository.resolve())
    given = resolve_settings(config_files=[preset], base_dir=start, environment=environment,
                             project_settings=False)
    assert (given.sections["coverage"]["max_requests_per_phase"], given.project_dir) == (
        "100", start.resolve())
    assert "project" not in given._layer_files


def test_the_home_dir_is_never_a_project(tmp_path, home, monkeypatch):
    """The project search stops below the home directory, whose ~/.testuary may
    hold older releases' environments, and reads no settings file there."""
    me = tmp_path / "me"
    (me / ".testuary").mkdir(parents=True)
    (me / ".testuary" / "project.ini").write_text("[coverage]\nmax_requests_per_phase = 7\n",
                                                   encoding="utf-8")
    work = me / "work"
    work.mkdir()
    monkeypatch.setattr(settings, "_home_dir", lambda: me.resolve())
    environment = {"TESTUARY_HOME": str(home)}
    for start in (work, me):
        resolved = resolve_settings(base_dir=start, environment=environment)
        assert resolved.project_dir == start.resolve()
        assert "project" not in resolved._layer_files
        assert resolved.sections["coverage"]["max_requests_per_phase"] != "7"


def test_the_layers_are_defaults_global_project_environment_command_line(home, project):
    (home / "config.ini").write_text("[main]\nseed = 1\ncontext_rebuild = every_turn\n"
                                     "context_rebuild_trigger_tokens = 5\n", encoding="utf-8")
    (project / ".testuary" / "project.ini").write_text(
        "[main]\nseed = 2\ncontext_rebuild_trigger_tokens = 6\n", encoding="utf-8")
    resolved = resolve_settings(base_dir=project, environment={
        "TESTUARY_HOME": str(home), "TESTUARY_MAIN_CONTEXT_REBUILD_TRIGGER_TOKENS": "7"},
        overrides={"main": {"seed": 3}})
    assert resolved.main.seed == 3
    assert resolved.main.context_rebuild == "every_turn"
    assert resolved.main.context_rebuild_trigger_tokens == 7
    labels = [label for label, _ in resolved.configuration_layers()]
    assert labels == ["defaults", "config[0]", "project", "environment", "overrides"]
    global_file = str((home / "config.ini").resolve())
    project_file = str((project / ".testuary" / "project.ini").resolve())
    assert resolved.configuration_files[:2] == (global_file, project_file)
    sources = {(item.setting.section, item.setting.name): item.source
               for item in settings.setting_values(resolved, languages=("python",))}
    assert sources[("main", "context_rebuild")] == global_file
    assert sources[("main", "seed")] == "overrides"
    assert sources[("main", "context_rebuild_trigger_tokens")] == "environment"
    without = resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)})
    sources = {(item.setting.section, item.setting.name): item.source
               for item in settings.setting_values(without, languages=("python",))}
    assert sources[("main", "seed")] == project_file
    assert without.project_dir == project.resolve()


def test_the_catalogue_and_global_configuration_come_from_the_user_dir_only(tmp_path, home):
    (home / "models.ini").write_text("[mine]\nprovider = ollama\nmodel = m\n", encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    # Files of these names in the working directory are not read.
    (work / "models.ini").write_text("[stale]\nprovider = ollama\nmodel = x\n", encoding="utf-8")
    (work / "config.ini").write_text("[main]\nseed = 9\n", encoding="utf-8")
    resolved = resolve_settings(base_dir=work, environment={"TESTUARY_HOME": str(home)})
    assert set(resolved.models) == {"mine"}
    assert resolved.main.seed == 0
    assert str((home / "models.ini").resolve()) in resolved.configuration_files
    assert not any(Path(path).parent == work.resolve() for path in resolved.configuration_files)


def test_an_explicit_configuration_replaces_the_global_one(tmp_path, home, project):
    (home / "config.ini").write_text("[main]\nseed = 1\n", encoding="utf-8")
    (project / ".testuary" / "project.ini").write_text("[main]\ncontext_rebuild_trigger_tokens = 6\n",
                                                       encoding="utf-8")
    explicit = tmp_path / "explicit.ini"
    explicit.write_text("[main]\ncontext_rebuild = every_turn\n", encoding="utf-8")
    resolved = resolve_settings(base_dir=project, config_files=[explicit],
                                environment={"TESTUARY_HOME": str(home)})
    assert (resolved.main.seed, resolved.main.context_rebuild,
            resolved.main.context_rebuild_trigger_tokens) == (0, "every_turn", 6)
    from_variable = resolve_settings(base_dir=project, environment={
        "TESTUARY_HOME": str(home), "TESTUARY_CONFIG_FILE": str(explicit)})
    assert (from_variable.main.seed, from_variable.main.context_rebuild) == (0, "every_turn")


def test_an_empty_list_of_configuration_files_reads_no_global_configuration(tmp_path, home, project):
    """Only config_files None reads the global configuration or the file
    TESTUARY_CONFIG_FILE names. An empty list adds no file layer, so a
    program whose settings file does not exist yet reads the defaults, the
    project's settings file and the environment, and with project_settings
    False only the defaults and the environment."""
    (home / "config.ini").write_text("[main]\nseed = 1\n", encoding="utf-8")
    (project / ".testuary" / "project.ini").write_text("[main]\ncontext_rebuild_trigger_tokens = 6\n",
                                                       encoding="utf-8")
    explicit = tmp_path / "explicit.ini"
    explicit.write_text("[main]\nseed = 2\n", encoding="utf-8")
    for variables in ({}, {"TESTUARY_CONFIG_FILE": str(explicit)}):
        environment = {"TESTUARY_HOME": str(home), **variables}
        resolved = resolve_settings(base_dir=project, config_files=[], environment=environment)
        assert (resolved.main.seed, resolved.main.context_rebuild_trigger_tokens) == (0, 6)
        assert [label for label, _ in resolved.configuration_layers()] == ["defaults", "project"]
        alone = resolve_settings(base_dir=project, config_files=[], project_settings=False,
                                 environment=environment)
        assert alone.main.seed == 0
        assert [label for label, _ in alone.configuration_layers()] == ["defaults"]


def test_home_points_the_user_dir_elsewhere(tmp_path, home):
    other = tmp_path / "other"
    other.mkdir()
    (other / "models.ini").write_text("[theirs]\nprovider = ollama\nmodel = m\n", encoding="utf-8")
    resolved = resolve_settings(base_dir=tmp_path, home=other,
                                environment={"TESTUARY_HOME": str(home)})
    assert set(resolved.models) == {"theirs"}


def test_testuary_home_is_an_input_only_with_the_environment(tmp_path, home, monkeypatch):
    (home / "models.ini").write_text("[mine]\nprovider = ollama\nmodel = m\n", encoding="utf-8")
    platform = tmp_path / "platform"
    monkeypatch.setattr(settings, "platform_user_dir", lambda: platform)
    assert set(resolve_settings(base_dir=tmp_path, environment={"TESTUARY_HOME": str(home)},
                                use_environment=False).models) == set()
    # The process environment, without the suite's own Testuary variables.
    for name in [name for name in os.environ if name.startswith("TESTUARY_")]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("TESTUARY_HOME", str(home))
    assert set(resolve_settings(base_dir=tmp_path).models) == {"mine"}


def test_the_project_settings_file_refuses_what_a_configuration_file_refuses(home, project):
    settings_file = project / ".testuary" / "project.ini"
    for text, refusal in (("[main]\nsed = 1\n", "sed"),
                          ("[main]\ntestuary_dir = other\n", "testuary_dir"),
                          # Whether starter suites can run together is the
                          # language's own, so no setting decides it.
                          ("[starter]\npool_java_starters = false\n",
                           r"\[starter\]: unknown setting 'pool_java_starters'")):
        settings_file.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=refusal):
            resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)})


def test_the_results_go_below_the_projects_testuary_dir(home, project):
    resolved = resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)},
                                overrides={"api": {"model": "my/model"}})
    assert resolved.sections["project"]["output_dir"] == ""
    assert resolved.results_dir == project.resolve() / ".testuary" / "results"
    with config.use_run_settings(resolved):
        assert project_module.output_dir() == project.resolve() / ".testuary" / "results" / "my_model"
    explicit = resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)},
                                overrides={"project": {"output_dir": "out/${model}"},
                                           "api": {"model": "m"}})
    with config.use_run_settings(explicit):
        assert project_module.output_dir() == project.resolve() / "out" / "m"


def test_a_direct_settings_object_names_its_project_and_results(tmp_path):
    direct = settings.RunSettings(base_dir=tmp_path)
    assert direct.project_dir == tmp_path.resolve()
    assert direct.results_dir == tmp_path.resolve() / ".testuary" / "results"
    elsewhere = settings.RunSettings(base_dir=tmp_path, project_dir=tmp_path / "p")
    assert elsewhere.results_dir == (tmp_path / "p").resolve() / ".testuary" / "results"


def test_the_results_may_be_in_the_projects_testuary_dir_but_nowhere_else_in_it(home, project):
    from testuary import orchestrator
    from testuary.settings import ConfigurationError

    resolved = resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)})
    with config.use_run_settings(resolved):
        orchestrator._check_output_outside_source(project.resolve() / ".testuary" / "results" / "m",
                                                  project.resolve())
        with pytest.raises(ConfigurationError, match="inside the source directory"):
            orchestrator._check_output_outside_source(project.resolve() / "results" / "m",
                                                      project.resolve())


def test_the_work_copy_leaves_out_the_projects_testuary_dir(tmp_path, home, project):
    from testuary import orchestrator

    (project / "calc.py").write_text("x = 1\n", encoding="utf-8")
    (project / ".testuary" / "project.ini").write_text("[main]\nseed = 2\n", encoding="utf-8")
    (project / ".testuary" / "results" / "m").mkdir(parents=True)
    (project / ".testuary" / "results" / "m" / "run.json").write_text("{}", encoding="utf-8")
    resolved = resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)})
    copy = tmp_path / "copy"
    with config.use_run_settings(resolved):
        orchestrator._copy_project_to_work(projects_root=project.resolve(), work_src=copy)
    assert sorted(path.name for path in copy.iterdir()) == ["calc.py"]
    assert ".testuary" in config.configuration_exclusions(project.resolve(), resolved)


def test_a_run_keeps_its_results_out_of_version_control(home, project, tmp_path):
    from testuary import orchestrator

    resolved = resolve_settings(base_dir=project, environment={"TESTUARY_HOME": str(home)})
    outdir = resolved.results_dir / "m"
    outdir.mkdir(parents=True)
    elsewhere = tmp_path / "out"
    elsewhere.mkdir()
    with config.use_run_settings(resolved):
        orchestrator._keep_results_out_of_version_control(elsewhere)
        assert not (resolved.results_dir / ".gitignore").exists()
        orchestrator._keep_results_out_of_version_control(outdir)
        assert (resolved.results_dir / ".gitignore").read_text(encoding="utf-8") == "*\n"
        (resolved.results_dir / ".gitignore").write_text("# mine\n", encoding="utf-8")
        orchestrator._keep_results_out_of_version_control(outdir)
        assert (resolved.results_dir / ".gitignore").read_text(encoding="utf-8") == "# mine\n"
