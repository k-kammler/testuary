"""Ordered settings inputs remain available to the shared request resolver."""
from dataclasses import FrozenInstanceError, replace
import json
import re

import pytest

from testuary import config
from testuary import settings as settings_module
from testuary.settings import RunSettings, resolve_settings


def _layers_of(settings, section):
    """One section's inputs in order, labeled by layer, from the settings'
    captured configuration layers."""
    return tuple((name, fields[section]) for name, fields in settings.configuration_layers()
                 if section in fields)


def test_direct_settings_keep_defaults_separate_from_explicit_fields():
    settings = RunSettings(sections={"starter": {"temperature": 0}})
    layers = _layers_of(settings, "starter")
    assert [name for name, _ in layers] == ["defaults", "overrides"]
    assert layers[0][1]["temperature"] == ""
    assert layers[1][1] == {"temperature": "0"}
    assert settings.sections["starter"]["temperature"] == "0"


@pytest.mark.parametrize("section", ["starter", "coverage", "mutation", "api"])
def test_a_temperature_in_any_model_section_is_a_number(tmp_path, section):
    """A blank default keeps temperature a number: text is refused and a
    fraction is kept, in each section that sets a model's values."""
    path = tmp_path / "settings.ini"
    path.write_text(f"[{section}]\ntemperature = invalid\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"\[{section}\] temperature: expected a number"):
        resolve_settings(base_dir=tmp_path, config_files=[path], models={}, environment={})
    path.write_text(f"[{section}]\ntemperature = 0.5\n", encoding="utf-8")
    settings = resolve_settings(base_dir=tmp_path, config_files=[path], models={}, environment={})
    assert settings.sections[section]["temperature"] == "0.5"


def test_resolved_layers_preserve_file_environment_and_explicit_order(tmp_path):
    first = tmp_path / "first.ini"
    second = tmp_path / "second.ini"
    first.write_text("[api]\nmodel=first\n[starter]\ntemperature=0.2\n", encoding="utf-8")
    second.write_text("[api]\nmodel=second\n[starter]\ntemperature=0.4\n", encoding="utf-8")
    settings = resolve_settings(
        base_dir=tmp_path, config_files=[first, second], models={},
        environment={"TESTUARY_STARTER_TEMPERATURE": "0.6"},
        overrides={"starter": {"temperature": ""}},
    )
    layers = _layers_of(settings, "starter")
    assert [(name, fields["temperature"]) for name, fields in layers] == [
        ("defaults", ""), ("config[0]", "0.2"), ("config[1]", "0.4"),
        ("environment", "0.6"), ("overrides", ""),
    ]
    assert _layers_of(settings, "api")[1:] == (
        ("config[0]", {"model": "first"}), ("config[1]", {"model": "second"}),
    )
    assert settings.sections["api"]["model"] == "second"
    assert settings.sections["starter"]["temperature"] == ""


def test_implicit_file_and_environment_selected_file_are_captured(tmp_path):
    (tmp_path / "config.ini").write_text("[main]\nseed=2\n", encoding="utf-8")
    (tmp_path / "chosen.ini").write_text("[main]\nseed=3\n", encoding="utf-8")
    implicit = resolve_settings(base_dir=tmp_path, home=tmp_path, models={},
                                environment={})
    selected = resolve_settings(base_dir=tmp_path, home=tmp_path, models={},
                                environment={"TESTUARY_CONFIG_FILE": "chosen.ini"})
    assert _layers_of(implicit, "main")[1:] == (("config[0]", {"seed": "2"}),)
    assert _layers_of(selected, "main")[1:] == (("config[0]", {"seed": "3"}),)


def test_layer_values_are_detached_and_frozen(tmp_path):
    fields = {"main": {"seed": 8}}
    settings = resolve_settings(base_dir=tmp_path, config_files=[], models={},
                                environment={}, overrides=fields)
    fields["main"]["seed"] = 99
    assert _layers_of(settings, "main")[-1] == ("overrides", {"seed": "8"})
    with pytest.raises(TypeError):
        _layers_of(settings, "main")[-1][1]["seed"] = "0"
    with pytest.raises(FrozenInstanceError):
        settings._layers = ()


def test_empty_false_and_zero_are_not_dropped_from_their_source():
    settings = RunSettings(sections={"main": {
        "seed": 0, "phase_end_mutation": False, "hints_file": "",
    }})
    assert _layers_of(settings, "main")[-1][1] == {
        "seed": "0", "phase_end_mutation": "false", "hints_file": "",
    }


def test_sections_absent_from_a_layer_are_not_backfilled():
    settings = RunSettings(sections={"coverage": {"max_requests_per_phase": 3}})
    assert [name for name, _ in _layers_of(settings, "starter")] == ["defaults"]
    assert _layers_of(settings, "undeclared") == ()


def test_authoritative_layer_access_does_not_reload_sources(tmp_path, monkeypatch):
    path = tmp_path / "settings.ini"
    path.write_text("[main]\nseed=5\n", encoding="utf-8")
    settings = resolve_settings(base_dir=tmp_path, config_files=[path], models={},
                                environment={"TESTUARY_MAIN_SEED": "6"})
    path.write_text("[main]\nseed=9\n", encoding="utf-8")
    monkeypatch.setenv("TESTUARY_MAIN_SEED", "10")
    monkeypatch.setattr(settings_module, "_read_ini_file", lambda _: pytest.fail("settings were reloaded"))
    with config.use_run_settings(settings):
        assert _layers_of(settings, "main")[1:] == (
            ("config[0]", {"seed": "5"}), ("environment", {"seed": "6"}),
        )
        assert config.global_seed() == 6


def test_provenance_does_not_change_existing_export_or_reveal_paths_and_secrets(tmp_path):
    path = tmp_path / "private-location.ini"
    path.write_text("[api]\napi_url=https://user:secret@example.test/v1?key=hidden\n",
                    encoding="utf-8")
    settings = resolve_settings(base_dir=tmp_path, config_files=[path],
                                models={"openai": {"api_key": "private-key"}},
                                environment={})
    assert _layers_of(settings, "api")[-1][0] == "config[0]"
    assert set(settings.export()) == {"sections", "models"}
    for text in (repr(settings), json.dumps(settings.export())):
        for private in ("private-key", "secret", "hidden", "private-location", str(tmp_path)):
            assert private not in text


def test_replacing_explicit_sections_rebuilds_provenance():
    first = RunSettings(sections={"main": {"seed": 1}})
    second = replace(first, sections={"main": {"seed": 2}})
    assert _layers_of(first, "main")[-1] == ("overrides", {"seed": "1"})
    assert _layers_of(second, "main")[-1] == ("overrides", {"seed": "2"})
    assert second.main.seed == 2


def test_merging_the_captured_inputs_reconstructs_the_effective_sections(tmp_path):
    path = tmp_path / "layer.ini"
    path.write_text("[coverage]\nmax_requests_per_phase=3\n[mutation]\nmax_requests_per_phase=4\n", encoding="utf-8")
    settings = resolve_settings(
        base_dir=tmp_path, config_files=[path], models={},
        environment={"TESTUARY_API_MODEL": "chosen", "TESTUARY_COVERAGE_MAX_REQUESTS_PER_PHASE": "5"},
        overrides={"main": {"seed": 0}, "mutation": {"max_requests_per_phase": 6}},
    )
    for section, effective in settings.sections.items():
        merged = {}
        for _, values in _layers_of(settings, section):
            merged.update(values)
        assert merged == effective


def test_disabling_environment_does_not_capture_its_section_layer(tmp_path):
    settings = resolve_settings(
        base_dir=tmp_path, config_files=[], models={}, use_environment=False,
        environment={"TESTUARY_MAIN_SEED": "invalid"},
    )
    assert [name for name, _ in _layers_of(settings, "main")] == ["defaults"]
    assert settings.main.seed == 0


CLUSTER_MUTATION = {
    "TESTUARY_MUTATION_ENGINE_WORKERS": "15",
    "TESTUARY_MUTATION_COSMIC_RAY_DISTRIBUTOR": "http",
    "TESTUARY_MUTATION_COSMIC_RAY_CLONE_DIR": "/dev/shm/testuary_1/runs/task",
    "TESTUARY_MUTATION_COSMIC_RAY_LOCAL_SESSION_DIR": "/dev/shm/testuary_1/runs/task/sessions",
}


def test_the_cluster_mutation_variables_reach_the_mutation_settings_every_phase_reads(tmp_path):
    """The mutation variables a cluster job sets are the current setting
    names. Every phase's mutation measurement reads the [mutation]
    section."""
    settings = resolve_settings(base_dir=tmp_path, config_files=[], models={},
                                environment=CLUSTER_MUTATION)
    with config.use_run_settings(settings):
        mutation = config.get_agent_config("test_mutation", parent="test_coverage")
    assert (mutation["engine_workers"], mutation["cosmic_ray_distributor"],
            mutation["cosmic_ray_clone_dir"], mutation["cosmic_ray_local_session_dir"]) == (
        15, "http", "/dev/shm/testuary_1/runs/task", "/dev/shm/testuary_1/runs/task/sessions")


@pytest.mark.parametrize("name,value,message", (
    ("TESTUARY_MUTATION_COSMIC_RAY_CLONEDIR", "/x", "unknown setting 'cosmic_ray_clonedir'"),
    # The removed serial recovery's activation sets nothing now: it is refused.
    ("TESTUARY_MUTATION_RECOVERY", "x", "unknown setting 'recovery'"),
    ("TESTUARY_MUTATON_ENGINE_WORKERS", "15", "unknown section 'mutaton'"),
    ("TESTUARY_STARTER_MUTATION_ENGINE_WORKERS", "15", r"\[starter\]: unknown setting"),
    ("TESTUARY_MUTATION_ENGINE_WORKERS", "0", "expected a positive value"),
    ("TESTUARY_MUTATION_COSMIC_RAY_DISTRIBUTOR", "htttp", "choose one of auto, local, http"),
))
def test_a_misspelled_mutation_variable_or_value_is_refused(tmp_path, name, value, message):
    with pytest.raises(ValueError, match=message):
        resolve_settings(base_dir=tmp_path, config_files=[], models={},
                         environment={**CLUSTER_MUTATION, name: value})


@pytest.mark.parametrize("name,message", (
    ("TESTUARY_COVREAGE_GOAL",
     "environment variable TESTUARY_COVREAGE_GOAL: unknown section 'covreage'. The nearest known section is 'coverage'"),
    ("TESTUARY_NOWHERE_GOAL", "environment variable TESTUARY_NOWHERE_GOAL: unknown section 'nowhere'"),
))
def test_a_variable_of_an_unknown_section_is_refused_naming_the_section(tmp_path, name, message):
    with pytest.raises(ValueError, match=re.escape(message)):
        resolve_settings(base_dir=tmp_path, config_files=[], models={}, environment={name: "1"})


def test_environment_only_inputs_and_credentials_set_no_setting(tmp_path):
    """Variables Testuary reads for itself, provider credentials and the
    variables the catalogue names are no settings, also where their name
    begins with a section's."""
    from testuary import phase_state

    assert phase_state._CLOCK_DOMAIN_ENV in settings_module.ENVIRONMENT_ONLY
    (tmp_path / "empty.ini").write_text("", encoding="utf-8")
    environment = {name: "x" for name in settings_module.ENVIRONMENT_ONLY}
    environment.update(TESTUARY_HOME=str(tmp_path), TESTUARY_CONFIG_FILE=str(tmp_path / "empty.ini"),
                       TESTUARY_MYPROVIDER_API_KEY="credential", TESTUARY_CHOSEN="credential")
    models = {"chosen": {"provider": "openai", "model": "m", "api_key_env": "TESTUARY_CHOSEN"}}
    settings = resolve_settings(base_dir=tmp_path, models=models, environment=environment)
    assert "work_dir_namespace" not in settings.sections["main"]


@pytest.mark.parametrize("section, key", [
    ("coverage", "goal"), ("coverage", "minimum"),
    ("mutation", "goal"), ("mutation", "minimum"),
    ("mutation", "selective_switch_threshold"),
])
def test_a_percent_setting_outside_0_to_100_is_refused(section, key):
    """A goal, minimum or threshold in percent above 100 or below 0 can never
    be met or is always met: the settings the console and the web
    interface read refuse it, by its name."""
    resolve = lambda value: resolve_settings(models={}, environment={}, config_files=[],  # noqa: E731
                                             overrides={section: {key: value}})
    for value in (0, 50, 100):
        resolve(value)
    for value in (150, 101, -1):
        with pytest.raises(ValueError, match=rf"\[{section}\] {key}: expected a percentage from 0 to 100"):
            resolve(value)


@pytest.mark.parametrize("key", ["measurement_timeout_seconds", "min_per_mutant_timeout_seconds"])
def test_a_mutation_timeout_of_zero_or_less_is_refused(key):
    """A mutation measurement with no time at all, or a per-mutant timeout
    with no minimum, would fail inside the language adapter, and the run
    would report no mutation score without saying why: the settings the
    console and the web interface read refuse it, by its name."""
    resolve = lambda value: resolve_settings(models={}, environment={}, config_files=[],  # noqa: E731
                                             overrides={"mutation": {key: value}})
    resolve(1)
    for value in (0, -5):
        with pytest.raises(ValueError, match=rf"\[mutation\] {key}: expected a positive value"):
            resolve(value)


def test_starter_cleanup_rounds_below_one_are_refused_by_the_settings_check():
    """The starter phase needs at least one cleanup round: the settings the
    console and the web interface read refuse fewer under [starter], before
    any phase starts."""
    resolve = lambda value: resolve_settings(  # noqa: E731
        models={}, environment={}, config_files=[],
        overrides={"starter": {"failing_test_cleanup_rounds": value}})
    resolve(1)
    for value in (0, -1):
        with pytest.raises(ValueError, match=r"\[starter\] failing_test_cleanup_rounds: expected a positive value"):
            resolve(value)


@pytest.mark.parametrize("section, key, accepted", [
    ("coverage", "goal", True),
    ("mutation", "max_requests_per_phase", True),
    ("starter", "goal", False),
    ("starter", "max_requests_per_phase", False),
    ("mutation", "goal_metric", False),
    ("finishing", "max_requests_per_phase", False),
])
def test_a_phase_section_takes_only_the_keys_its_phase_reads(section, key, accepted):
    """The starter phase runs no agent loop and only the coverage phase reads
    the coverage goal and minimum, so other sections refuse those keys instead
    of ignoring them."""
    resolve = lambda: resolve_settings(models={}, environment={}, config_files=[],  # noqa: E731
                                       overrides={section: {key: 5}})
    if accepted:
        resolve()
    else:
        with pytest.raises(ValueError, match=rf"\[{section}\]: unknown setting '{key}'"):
            resolve()


def test_a_settings_value_names_every_layer_that_sets_it(tmp_path):
    """``source`` is the layer whose value is in force; ``sources`` names
    every layer that sets the key, in input order, so a program can offer a
    reset for a level another one overrides. No value is added."""
    first = tmp_path / "first.ini"
    second = tmp_path / "second.ini"
    first.write_text("[starter]\ntemperature=0.2\n", encoding="utf-8")
    second.write_text("[starter]\ntemperature=0.4\n", encoding="utf-8")
    settings = resolve_settings(
        base_dir=tmp_path, config_files=[first, second], models={},
        environment={"TESTUARY_STARTER_TEMPERATURE": "0.6"})
    values = {(item.setting.section, item.setting.name): item
              for item in settings_module.setting_values(settings)}
    temperature = values[("starter", "temperature")]
    assert temperature.source == "environment"
    assert temperature.sources == (str(first), str(second), "environment")
    assert values[("api", "model")].sources == ()


@pytest.mark.parametrize("section, old, new", [
    ("project", "source_folder", "source_dir"),
    ("main", "testuary_folder", "testuary_dir"),
    ("main", "work_folder_job_namespace", "work_dir_job_namespace"),
])
def test_a_renamed_setting_is_refused_under_its_old_name_naming_the_new(tmp_path, section, old, new):
    """An old setting name is refused as any unknown setting is, naming the
    setting that reads its "folder" as "dir"."""
    (tmp_path / "home").mkdir()
    config_file = tmp_path / "config.ini"
    config_file.write_text(f"[{section}]\n{old} = x\n", encoding="utf-8")
    with pytest.raises(ValueError, match=rf"unknown setting '{old}'\. The nearest known setting is '{new}'$"):
        resolve_settings(base_dir=tmp_path, home=tmp_path / "home", config_files=[config_file],
                         project_settings=False)


@pytest.mark.parametrize("old, new", [
    ("TESTUARY_PROJECT_SOURCE_FOLDER", "source_dir"),
    ("TESTUARY_MAIN_TESTUARY_FOLDER", "testuary_dir"),
    ("TESTUARY_MAIN_WORK_FOLDER_NAMESPACE", "TESTUARY_MAIN_WORK_DIR_NAMESPACE"),
    ("TESTUARY_PROJECT_WORK_FOLDER", "TESTUARY_PROJECT_WORK_DIR"),
])
def test_a_renamed_variable_is_refused_naming_the_new(tmp_path, old, new):
    """An old variable is refused, naming the setting or the variable that
    reads its "folder" as "dir"."""
    (tmp_path / "home").mkdir()
    with pytest.raises(ValueError, match=rf"The nearest known \w+ is '{new}'$"):
        resolve_settings(base_dir=tmp_path, home=tmp_path / "home", environment={old: "x"},
                         project_settings=False)


def test_each_layer_adds_its_extensions_to_the_earlier_layers(tmp_path):
    """[main] extensions is the one setting whose layers combine: the
    global configuration's items, then the project's, the variable's and the
    overrides', each as written and in that order. Other settings replace.
    Each item names its layer and the directory a file item is relative to."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.ini").write_text("[main]\nextensions = my_tools.web\nseed = 1\n", encoding="utf-8")
    project = tmp_path / "project"
    (project / ".testuary").mkdir(parents=True)
    (project / ".testuary" / "project.ini").write_text(
        "[main]\nextensions = my_hints.py\nseed = 2\n", encoding="utf-8")
    settings = resolve_settings(base_dir=project, home=home, models={},
                                environment={"TESTUARY_MAIN_EXTENSIONS": "tools/more.py"},
                                overrides={"main": {"extensions": "notes_web"}})
    assert settings.sections["main"]["extensions"] == "my_tools.web, my_hints.py, tools/more.py, notes_web"
    assert settings.sections["main"]["seed"] == "2"
    values = {(item.setting.section, item.setting.name): item
              for item in settings_module.setting_values(settings)}
    user, own = (home / "config.ini").resolve(), (project / ".testuary" / "project.ini").resolve()
    assert values[("main", "extensions")].sources == (str(user), str(own), "environment", "overrides")
    project = project.resolve()
    assert settings_module.extension_items(settings) == (
        ("my_tools.web", str(user), str(user.parent)), ("my_hints.py", str(own), str(own.parent)),
        ("tools/more.py", "environment", str(project)), ("notes_web", "overrides", str(project)))


@pytest.mark.parametrize("text", ["a.py, , my_tools.web", "a.py,", "a.py\nmy_tools.web", "a.py\x0cb.py",
                                  "a.py b.py"])
def test_an_extensions_list_with_an_empty_item_or_a_line_break_is_refused(text):
    with pytest.raises(ValueError, match=r"\[main\] extensions: expected a comma list"):
        RunSettings(sections={"main": {"extensions": text}})


def test_the_extensions_check_loads_nothing(tmp_path):
    """The check is syntactic: an item no module or file answers is kept,
    and another setting is saved beside it."""
    settings = RunSettings(sections={"main": {"extensions": " no_such_module_xyz ,missing.py"}})
    assert settings.sections["main"]["extensions"] == "no_such_module_xyz, missing.py"
    path = tmp_path / "config.ini"
    path.write_text("[main]\nextensions = no_such_module_xyz\n", encoding="utf-8")
    settings_module.update_file(path, "starter", {"temperature": "0.3"})
    assert "temperature = 0.3" in path.read_text(encoding="utf-8")


@pytest.fixture
def user_config(tmp_path, monkeypatch):
    """The user's config.ini in a user directory of the test's own."""
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("TESTUARY_HOME", str(home))
    return home / "config.ini"


def test_add_extension_names_an_item_once_after_the_others(user_config):
    user_config.write_text("# Mine.\n[main]\nextensions = my_hints.py\nseed = 3\n", encoding="utf-8")
    assert settings_module.add_extension("my_tools.web") is True
    assert settings_module.add_extension(" my_tools.web ") is False
    text = user_config.read_text(encoding="utf-8")
    assert "extensions = my_hints.py, my_tools.web\n" in text
    assert text.startswith("# Mine.\n") and "seed = 3\n" in text


def test_add_extension_writes_nothing_for_a_named_item(user_config):
    user_config.write_text("[main]\nextensions = my_tools.web\n", encoding="utf-8")
    before = user_config.read_bytes()
    assert settings_module.add_extension("my_tools.web") is False
    assert user_config.read_bytes() == before


def test_remove_extension_keeps_the_other_items_and_lines(user_config):
    """The last item's removal removes the key."""
    user_config.write_text("[main]\nextensions = my_hints.py, my_tools.web\nseed = 3\n", encoding="utf-8")
    assert settings_module.remove_extension("my_tools.web") is True
    assert user_config.read_text(encoding="utf-8") == "[main]\nextensions = my_hints.py\nseed = 3\n"
    assert settings_module.remove_extension("my_hints.py") is True
    assert user_config.read_text(encoding="utf-8") == "[main]\nseed = 3\n"


def test_remove_extension_of_an_absent_item_writes_nothing(user_config):
    user_config.write_text("[main]\nseed = 3\n", encoding="utf-8")
    before = user_config.read_bytes()
    assert settings_module.remove_extension("my_tools.web") is False
    assert user_config.read_bytes() == before


@pytest.mark.parametrize("item", ["", "  ", "my_hints.py, my_tools.web", "my\nhints.py"])
def test_an_extension_item_that_is_not_one_name_is_refused(user_config, item):
    user_config.write_text("[main]\nseed = 3\n", encoding="utf-8")
    before = user_config.read_bytes()
    with pytest.raises(settings_module.ConfigurationError, match=re.escape(repr(item))):
        settings_module.add_extension(item)
    assert user_config.read_bytes() == before


def test_api_takes_no_chat_model_which_its_provider_section_takes(tmp_path):
    """No model resolution reads chat_model from [api]: the setting and its
    variable are refused as unknown, and the provider's section keeps it."""
    (tmp_path / "home").mkdir()
    config_file = tmp_path / "config.ini"
    config_file.write_text("[api]\nchat_model = pkg.mod:Chat\n", encoding="utf-8")
    with pytest.raises(ValueError, match=r"\[api\]: unknown setting 'chat_model'"):
        resolve_settings(base_dir=tmp_path, home=tmp_path / "home", config_files=[config_file],
                         project_settings=False)
    with pytest.raises(ValueError, match=r"\[api\]: unknown setting 'chat_model'"):
        resolve_settings(base_dir=tmp_path, home=tmp_path / "home", project_settings=False,
                         environment={"TESTUARY_API_CHAT_MODEL": "pkg.mod:Chat"})
    settings = resolve_settings(base_dir=tmp_path, home=tmp_path / "home", project_settings=False,
                                environment={"TESTUARY_OPENAI_CHAT_MODEL": "pkg.mod:Chat"})
    assert settings.sections["openai"]["chat_model"] == "pkg.mod:Chat"


@pytest.mark.parametrize("key, value", [("reasoning_dialect", "vllm"), ("reasoning_template", "qwen")])
def test_api_takes_no_reasoning_dialect_or_template(tmp_path, key, value):
    """No model resolution reads them from [api]: the settings and their
    variables are refused as unknown, and the provider's section keeps
    them."""
    (tmp_path / "home").mkdir()
    for environment in ({f"TESTUARY_API_{key.upper()}": value}, {}):
        config_file = tmp_path / "config.ini"
        config_file.write_text("" if environment else f"[api]\n{key} = {value}\n", encoding="utf-8")
        with pytest.raises(ValueError, match=rf"\[api\]: unknown setting '{key}'"):
            resolve_settings(base_dir=tmp_path, home=tmp_path / "home", config_files=[config_file],
                             project_settings=False, environment=environment)
    settings = resolve_settings(base_dir=tmp_path, home=tmp_path / "home", project_settings=False,
                                environment={f"TESTUARY_OPENAI_{key.upper()}": value})
    assert settings.sections["openai"][key] == value
