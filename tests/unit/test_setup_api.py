"""The setup for other programs (testuary.setup): the walk answered through
one function, and the model and project catalogues listed and changed as
the setup changes them, each change keeping every other line."""
from __future__ import annotations

import os
import re
import sys

import pytest

from testuary import settings, setup
from testuary import settings as _ini_text
from testuary import setup as _setup

COMMENTED = (
    "# The model catalogue.\n"
    "\n"
    "# The provider's key.\n"
    "[ollama]\n"
    "api_key = secret-in-file\n"
    "\n"
    "# A cloud model.\n"
    "[cloud]\n"
    "provider = ollama\n"
    "# its model\n"
    "model = m1\n"
    "\n"
    "# The last one.\n"
    "[local_vllm]\n"
    "provider = vllm\n"
    "model = m2\n"
    "api_url = http://localhost:8000/v1\n"
)


class _Probe(setup.Probe):
    """The real file system, with a fixed environment and no network."""

    def __init__(self, environment=None):
        self.env = dict(environment or {})
        self.restricted = []

    def environment(self, name):
        return self.env.get(name, "")

    def local_ollama(self):
        return False

    def model_list(self, provider, address, key, *, timeout=15):
        return None

    def restrict_to_user(self, path):
        self.restricted.append(path)
        return True


def test_removal_keeps_every_other_line():
    text = COMMENTED.replace("\n", "\r\n")
    removed = _ini_text.remove_section(text, "cloud")
    # The comment above [cloud] describes it and goes with it.
    assert removed == (
        "# The model catalogue.\r\n\r\n# The provider's key.\r\n[ollama]\r\n"
        "api_key = secret-in-file\r\n\r\n# The last one.\r\n"
        "[local_vllm]\r\nprovider = vllm\r\nmodel = m2\r\napi_url = http://localhost:8000/v1\r\n")
    assert _ini_text.remove_section(text, "missing") == text
    # A comment block that starts the file is the file's own. A section at
    # the end leaves no blank line behind.
    assert _ini_text.remove_section("# Header\n[a]\nx = 1\n\n[b]\ny = 2\n", "a") == (
        "# Header\n\n[b]\ny = 2\n")
    assert _ini_text.remove_section("# Header\n\n[a]\nx = 1\n\n# b\n[b]\ny = 2\n", "b") == (
        "# Header\n\n[a]\nx = 1\n")
    keys = _ini_text.remove_keys("[a]\nx = 1\ny = 2\n  continued\n# kept\nz = 3\n[b]\ny = 9\n",
                                 "a", ["Y"])
    assert keys == "[a]\nx = 1\n# kept\nz = 3\n[b]\ny = 9\n"


def test_an_alias_at_its_providers_local_endpoint_needs_no_key(tmp_path):
    """A vLLM alias without an address uses the vllm provider's configured
    endpoint, on this machine by default, so it needs no key, as the setup
    asks for it as optional. A configured endpoint elsewhere needs one."""
    (tmp_path / "models.ini").write_text("[mine]\nprovider = vllm\nmodel = m\n", encoding="utf-8")
    (entry,) = setup.models(tmp_path, home=tmp_path, probe=_Probe())
    assert (entry.key_source, entry.usable, entry.address) == ("not needed", True, "")
    (tmp_path / "config.ini").write_text("[vllm]\napi_url = https://gpu.example.org/v1\n",
                                         encoding="utf-8")
    (entry,) = setup.models(tmp_path, home=tmp_path, probe=_Probe())
    assert (entry.key_source, entry.usable) == ("missing", False)


@pytest.mark.parametrize("address, local", [
    ("http://localhost:8000/v1", True), ("http://127.0.0.1:8000/v1", True),
    ("http://[::1]:8000/v1", True), ("https://localhost.example.com/v1", False),
    ("https://gpu.example.org/v1?via=127.0.0.1", False)])
def test_an_address_is_on_this_machine_by_its_host(tmp_path, address, local):
    (tmp_path / "models.ini").write_text(
        f"[mine]\nprovider = vllm\nmodel = m\napi_url = {address}\n", encoding="utf-8")
    (entry,) = setup.models(tmp_path, home=tmp_path, probe=_Probe())
    assert (entry.key_source, entry.usable) == (("not needed", True) if local else ("missing", False))


#: The run's words for [main] testuary_dir in a project's settings file.
LOCATES = ("[main] testuary_dir locates this file. Set it in the global configuration, "
           "the environment or on the command line")


def test_the_setup_and_the_run_refuse_the_project_files_testuary_dir_alike(tmp_path):
    """The project's settings file cannot set the directory that locates it:
    set_project refuses a file that does, in the run's words."""
    project = tmp_path / ".testuary" / "project.ini"
    project.parent.mkdir()
    project.write_text("[main]\ntestuary_dir = elsewhere\n", encoding="utf-8")
    with pytest.raises(setup.SetupError) as refused:
        setup.set_project({"source_files": "a.py"}, base_dir=tmp_path, home=tmp_path)
    assert LOCATES in str(refused.value)
    with pytest.raises(settings.ConfigurationError, match=re.escape(LOCATES)):
        settings.resolve_settings(base_dir=tmp_path, home=tmp_path, models={}, use_environment=False)
    assert project.read_text(encoding="utf-8") == "[main]\ntestuary_dir = elsewhere\n"


def _run_finds_key(project_dir, alias):
    """Whether a run with ``alias`` gets a key, by the run's own resolution."""
    from testuary.models import resolve_model_identity, resolve_provider_credential

    resolved = settings.resolve_settings(base_dir=project_dir, home=project_dir, overrides={"api": {"model": alias}})
    identity = resolve_model_identity(resolved)
    try:
        return bool(resolve_provider_credential(
            identity.provider, resolved.models.get(identity.provider, {}),
            resolved.models.get(identity.model_alias, {}), resolved.environment))
    except settings.ConfigurationError:
        return False


def test_an_alias_is_usable_exactly_when_a_run_gets_its_key(tmp_path, monkeypatch):
    """A variable that an alias or its provider names refuses every run while
    it is unset, whatever else holds a key and wherever the service runs, so
    the setup lists such an alias as missing its key, naming that variable."""
    environment = {"TESTUARY_OPENAI_API_KEY": "default-key"}
    for name in [name for name in os.environ if name.startswith("TESTUARY_") and name != "TESTUARY_HOME"]:
        monkeypatch.delenv(name)
    monkeypatch.delenv("UNSET_KEY", raising=False)
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    (tmp_path / "models.ini").write_text(
        "[gateway]\napi_url = https://gateway.example.org/v1\napi_key = in-file\napi_key_env = UNSET_KEY\n"
        "\n[via_gateway]\nprovider = gateway\nmodel = m\n"
        "\n[named]\nprovider = openai\nmodel = gpt\napi_key_env = UNSET_KEY\n"
        "\n[local_named]\nprovider = vllm\nmodel = m\napi_url = http://localhost:8000/v1\n"
        "api_key_env = UNSET_KEY\n"
        "\n[plain]\nprovider = openai\nmodel = gpt\n", encoding="utf-8")
    entries = {entry.alias: entry for entry in setup.models(tmp_path, home=tmp_path,
                                                             probe=_Probe(environment))}
    assert {alias: (entry.key_source, entry.key_variable, entry.usable)
            for alias, entry in entries.items()} == {
        "via_gateway": ("missing", "UNSET_KEY", False),
        "named": ("missing", "UNSET_KEY", False),
        "local_named": ("missing", "UNSET_KEY", False),
        "plain": ("environment", "TESTUARY_OPENAI_API_KEY", True)}
    for alias, entry in entries.items():
        assert entry.usable == _run_finds_key(tmp_path, alias), alias


def test_a_service_key_variable_is_its_credential_never_a_setting(tmp_path, monkeypatch):
    """A service may be named like a settings section, main_x for one: its
    TESTUARY_<NAME>_API_KEY is that service's key, which a run reads, and
    never a setting the run would refuse. No _API_KEY variable reaches the
    resolved settings, TESTUARY_WEB_API_KEY included."""
    for name in [name for name in os.environ if name.startswith("TESTUARY_") and name != "TESTUARY_HOME"]:
        monkeypatch.delenv(name)
    monkeypatch.setenv("TESTUARY_MAIN_X_API_KEY", "key-of-main-x")
    monkeypatch.setenv("TESTUARY_WEB_API_KEY", "key-of-web")
    setup.set_model("mine", provider="main_x", model="m", address="https://gpu.example.org/v1",
                    base_dir=tmp_path, home=tmp_path, probe=_Probe({"TESTUARY_MAIN_X_API_KEY": "key-of-main-x"}))
    assert _run_finds_key(tmp_path, "mine")
    resolved = settings.resolve_settings(base_dir=tmp_path, home=tmp_path, overrides={"api": {"model": "mine"}})
    assert [(section, key) for section, values in resolved.sections.items() for key in values
            if key.endswith("api_key")] == []


def test_the_model_list_names_each_keys_source_and_holds_no_key(tmp_path):
    (tmp_path / "models.ini").write_text(
        COMMENTED + "\n[paid]\nprovider = openai\nmodel = gpt\n"
        "\n[deepseek]\napi_url = https://user:password@api.deepseek.com/v1\n"
        "[custom]\nprovider = deepseek\nmodel = chat\napi_key_env = MY_KEY\n", encoding="utf-8")
    probe = _Probe({"MY_KEY": "secret-in-environment"})
    entries = {entry.alias: entry for entry in setup.models(tmp_path, home=tmp_path, probe=probe)}
    assert set(entries) == {"cloud", "local_vllm", "paid", "custom"}
    assert (entries["cloud"].key_source, entries["cloud"].usable) == ("models.ini", True)
    assert (entries["local_vllm"].key_source, entries["local_vllm"].address) == (
        "not needed", "http://localhost:8000/v1")
    assert (entries["paid"].key_source, entries["paid"].key_variable,
            entries["paid"].usable) == ("missing", "TESTUARY_OPENAI_API_KEY", False)
    assert (entries["custom"].key_source, entries["custom"].key_variable,
            entries["custom"].address) == ("environment", "MY_KEY", "https://api.deepseek.com/v1")
    assert "secret" not in repr(entries) and "password" not in repr(entries)
    # One rule: the setup offers exactly the usable aliases.
    assert setup.usable_aliases(tmp_path, home=tmp_path, probe=probe) == ("cloud", "local_vllm", "custom")
    assert _setup._usable_aliases(_setup._Files(tmp_path, tmp_path), probe) == ["cloud", "local_vllm",
                                                                      "custom"]


def test_a_model_is_written_as_the_terminal_walk_writes_it(tmp_path):
    answers = {"model.provider": "openai", "model.key_storage": "file", "model.key": "sk-test",
               "model.name": "gpt-x", "model.alias": "gpt_x", "model.confirm": "yes"}
    walked, called = tmp_path / "walked", tmp_path / "called"
    walked.mkdir()
    called.mkdir()
    probe = _Probe()
    result = setup.run_setup(lambda question: answers.get(question.key, ""), probe,
                             lambda text: None, walked, part="model", home=walked)
    assert result.alias == "gpt_x"
    path = setup.set_model("gpt_x", provider="openai", model="gpt-x", key="sk-test",
                           address="https://api.openai.com/v1",
                           base_dir=called, home=called, probe=probe)
    assert path == settings.file_locations(called, home=called).models
    assert (called / "models.ini").read_bytes() == (walked / "models.ini").read_bytes()
    # The file holds a key, so both were restricted to their owner.
    assert len(probe.restricted) == 2


def test_a_fresh_machines_missing_user_dir_is_made(tmp_path):
    """On a fresh machine the user directory does not exist yet. The walk and
    set_model make it and write the catalogue there."""
    answers = {"model.provider": "openai", "model.key_storage": "file", "model.key": "sk-test",
               "model.name": "gpt-x", "model.alias": "gpt_x", "model.confirm": "yes"}
    walked, called = tmp_path / "walked" / "user", tmp_path / "called" / "user"
    result = setup.run_setup(lambda question: answers.get(question.key, ""), _Probe(),
                             lambda text: None, tmp_path, part="model", home=walked)
    assert result.alias == "gpt_x" and (walked / "models.ini").is_file()
    setup.set_model("gpt_x", provider="openai", model="gpt-x", key="sk-test",
                    address="https://api.openai.com/v1",
                    base_dir=tmp_path, home=called, probe=_Probe())
    assert (called / "models.ini").read_bytes() == (walked / "models.ini").read_bytes()


def test_an_alias_given_ollama_clouds_address_sends_there_over_a_configured_one(tmp_path):
    """The user's config.ini points [ollama] at an Ollama on this machine. An
    alias set with Ollama Cloud's address keeps it, so its run sends its
    requests and the cloud key to the cloud, not to the configured address."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.ini").write_text(
        "[ollama]\napi_url = http://localhost:11434/v1\n", encoding="utf-8")
    setup.set_model("cloud", provider="ollama", model="m", address="https://ollama.com/v1",
                    key="cloud-key", base_dir=tmp_path, home=home, probe=_Probe())
    resolved = settings.resolve_settings(
        base_dir=tmp_path, home=home, overrides={"api": {"model": "cloud"}})
    from testuary.models import resolve_model_identity

    assert resolve_model_identity(resolved).api_url == "https://ollama.com/v1"


def test_a_projects_settings_file_follows_the_given_user_dir(tmp_path):
    """project() and set_project() take the user directory as the other setup
    functions do: its settings name the Testuary directory."""
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.ini").write_text("[main]\ntestuary_dir = .other\n", encoding="utf-8")
    project_dir = tmp_path / "p"
    project_dir.mkdir()
    path = setup.set_project({"source_files": "a.py"}, base_dir=project_dir, home=home)
    assert path == project_dir.resolve() / ".other" / "project.ini"
    entry = setup.project(project_dir, home=home)
    assert (entry.settings_file, entry.fields) == (path, {"source_files": "a.py"})


def test_an_edit_keeps_the_other_entries_and_comments(tmp_path):
    (tmp_path / "models.ini").write_text(COMMENTED, encoding="utf-8")
    setup.set_model("cloud", provider="ollama", model="m3", base_dir=tmp_path, home=tmp_path,
                     probe=_Probe())
    edited = COMMENTED.replace("provider = ollama\n# its model\nmodel = m1\n",
                               "provider = ollama\nmodel = m3\n# its model\n")
    assert (tmp_path / "models.ini").read_text(encoding="utf-8") == edited
    setup.remove_model("local_vllm", base_dir=tmp_path, home=tmp_path, probe=_Probe())
    assert (tmp_path / "models.ini").read_text(encoding="utf-8") == (
        edited[:edited.index("\n# The last one.")])
    with pytest.raises(setup.SetupError, match="no model alias ollama"):
        setup.remove_model("ollama", base_dir=tmp_path, home=tmp_path)


@pytest.mark.parametrize("alias, values, field, message", [
    ("ollama", {"provider": "ollama", "model": "m"}, "alias", "built-in provider"),
    ("same", {"provider": "same", "model": "m", "address": "http://localhost:1/v1"},
     "alias", "need different names"),
    ("Bad-Name", {"provider": "ollama", "model": "m"}, "alias", "lowercase letters"),
    ("ok", {"provider": "service", "model": "m"}, "address", "holds no address for service"),
    ("ok", {"provider": "ollama", "model": ""}, "model", "a model name is needed"),
    ("ok", {"provider": "ollama", "model": "m", "address": "localhost"}, "address",
     "an address such as"),
    ("ok", {"provider": "ollama", "model": "m", "key_variable": "1BAD"}, "key_variable",
     "a variable name"),
    ("ok", {"provider": "ollama", "model": "two\nlines"}, "model", "must be one line"),
    ("ok", {"provider": "ollama", "model": "m", "key": "a\nkey"}, "key", "must be one line"),
])
def test_values_the_setup_refuses_are_refused_and_nothing_is_written(tmp_path, alias, values,
                                                                     field, message):
    (tmp_path / "models.ini").write_text(COMMENTED, encoding="utf-8")
    with pytest.raises(setup.SetupError, match=message) as refused:
        setup.set_model(alias, base_dir=tmp_path, home=tmp_path, probe=_Probe(), **values)
    assert any(name == field and message in problem for name, problem in refused.value.problems)
    # A refused key is never repeated in the message.
    assert "a\nkey" not in str(refused.value)
    assert (tmp_path / "models.ini").read_text(encoding="utf-8") == COMMENTED


def test_a_project_is_written_into_its_own_settings_file(tmp_path):
    (tmp_path / "src" / "calc").mkdir(parents=True)
    path = setup.set_project({"source_files": "src/calc/ops.py", "language": "python",
                              "python_import_root": "src", "model": "cloud"}, base_dir=tmp_path)
    assert path == tmp_path.resolve() / ".testuary" / "project.ini"
    assert path.read_text(encoding="utf-8") == (
        "[project]\nsource_files = src/calc/ops.py\nlanguage = python\npython_import_root = src\n\n"
        "[api]\nmodel = cloud\n")
    assert (tmp_path / ".testuary" / "results" / ".gitignore").read_bytes() == b"*\n"
    resolved = settings.resolve_settings(base_dir=tmp_path / "src", environment={})
    assert resolved.project_dir == tmp_path.resolve()
    assert resolved.sections["project"]["source_files"] == "src/calc/ops.py"
    assert resolved.sections["api"]["model"] == "cloud"
    entry = setup.project(tmp_path / "src")
    assert (entry.project_dir, entry.settings_file) == (tmp_path.resolve(), path)
    assert entry.fields == {"source_files": "src/calc/ops.py", "language": "python",
                            "python_import_root": "src", "model": "cloud"}


def test_a_change_keeps_the_files_other_lines_and_removes_what_is_asked(tmp_path):
    (tmp_path / ".testuary").mkdir()
    settings_file = tmp_path / ".testuary" / "project.ini"
    settings_file.write_text("# Mine.\n[main]\nseed = 3\n\n[project]\n# the files\nsource_files = a.py\n",
                             encoding="utf-8")
    setup.set_project({"source_files": "b.py"}, base_dir=tmp_path)
    assert settings_file.read_text(encoding="utf-8") == (
        "# Mine.\n[main]\nseed = 3\n\n[project]\n# the files\nsource_files = b.py\n")
    setup.set_project({}, remove=("source_files",), base_dir=tmp_path)
    assert settings_file.read_text(encoding="utf-8") == "# Mine.\n[main]\nseed = 3\n\n[project]\n# the files\n"


def test_a_project_value_a_run_refuses_is_refused_by_field_and_nothing_is_written(tmp_path):
    with pytest.raises(setup.SetupError) as refused:
        setup.set_project({"surce_files": "a.py", "language": "cobol", "model": "two\nlines"},
                          base_dir=tmp_path)
    assert sorted(field for field, _ in refused.value.problems) == ["language", "model", "surce_files"]
    with pytest.raises(setup.SetupError, match="setup_timeout_seconds") as refused:
        setup.set_project({"setup_timeout_seconds": "many"}, base_dir=tmp_path)
    # A value only the run's own check refuses is named by its key too.
    (field, problem), = refused.value.problems
    assert field == "setup_timeout_seconds" and "setup_timeout_seconds" in problem
    with pytest.raises(setup.SetupError) as refused:
        setup.set_project({"model": "x", "setup_timeout_seconds": "abc"}, base_dir=tmp_path)
    assert [field for field, _ in refused.value.problems] == ["setup_timeout_seconds"]
    assert not (tmp_path / ".testuary").exists()


def test_a_project_value_the_setup_refuses_is_refused_by_field(tmp_path):
    """The setup's project part refuses an import directory, interpreter, JDK
    or build tool that does not exist. set_project, which the web
    interface's project form writes through, refuses them the same way,
    each named, and writes nothing."""
    missing = tmp_path / "missing"
    with pytest.raises(setup.SetupError) as refused:
        setup.set_project({"python_import_root": "nope", "python_executable": str(missing / "python.exe"),
                           "java_home": str(missing), "maven": str(missing / "mvn")}, base_dir=tmp_path)
    problems = dict(refused.value.problems)
    # The oldest JDK is LangAdapter's support declaration's, which its own
    # tests pin.
    assert re.fullmatch(r"a JDK \d+ or later is needed", problems.pop("java_home"))
    assert problems == {
        "python_import_root": "no such directory", "python_executable": "this interpreter does not run",
        "maven": "no such file"}
    assert not (tmp_path / ".testuary").exists()
    (tmp_path / "src").mkdir()
    setup.set_project({"python_import_root": "src", "python_executable": sys.executable},
                      base_dir=tmp_path)
    assert setup.project(tmp_path).fields["python_executable"] == sys.executable


def test_the_files_are_where_the_resolver_names_them(tmp_path, monkeypatch):
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    moved = settings.FileLocations(
        user=elsewhere, config=elsewhere / "config.ini", models=elsewhere / "models.ini",
        project=tmp_path, testuary=tmp_path / ".testuary",
        project_settings=tmp_path / ".testuary" / "project.ini",
        results=tmp_path / ".testuary" / "results")
    monkeypatch.setattr(settings, "file_locations", lambda base_dir=None, **kwargs: moved)
    setup.set_model("cloud", provider="ollama", model="m", base_dir=tmp_path, probe=_Probe())
    assert (elsewhere / "models.ini").is_file() and not (tmp_path / "models.ini").exists()
    assert [entry.alias for entry in setup.models(tmp_path, probe=_Probe())] == ["cloud"]


def test_the_machine_part_writes_only_a_configuration_that_validates(tmp_path):
    answers = {"machine.mutation_workers": "4", "machine.starter_requests": "2",
               "machine.confirm": "yes"}

    def walk():
        return setup.run_setup(lambda question: answers.get(question.key, ""), _Probe(),
                               lambda text: None, tmp_path, part="machine", home=tmp_path)

    (tmp_path / "config.ini").write_text("# Mine.\n[main]\nseed = 1\n", encoding="utf-8")
    walk()
    assert (tmp_path / "config.ini").read_text(encoding="utf-8") == (
        "# Mine.\n[main]\nseed = 1\n\n[mutation]\nengine_workers = 4\n\n"
        "[starter]\nconcurrent_requests = 2\nconcurrent_requests_ollama = 2\n")
    (tmp_path / "config.ini").write_text("[main]\nnot_a_setting = 1\n", encoding="utf-8")
    with pytest.raises(setup.SetupError, match="not_a_setting"):
        walk()
    assert (tmp_path / "config.ini").read_text(encoding="utf-8") == "[main]\nnot_a_setting = 1\n"


def test_the_public_names_are_the_setups_own():
    for name in setup.__all__:
        assert getattr(setup, name) is getattr(_setup, name)


# A list of catalogue files (models_files)

SHARED = (
    "[gateway]\n"
    "api_url = https://gate.example/v1\n"
    "\n"
    "[local_qwen]\n"
    "provider = vllm\n"
    "model = qwen\n"
    "\n"
    "[cloud_coder]\n"
    "provider = gateway\n"
    "model = coder\n"
    "request_timeout_seconds = 60\n"
)
OWN = "# Mine.\n[local_qwen]\nprovider = vllm\nmodel = qwen-2\n"


def _list(tmp_path, own=OWN, shared=SHARED):
    """Runs in ``work`` reading lists/shared.ini, then lists/own.ini, both
    outside the project."""
    lists, work = tmp_path / "lists", tmp_path / "work"
    lists.mkdir()
    work.mkdir()
    (lists / "shared.ini").write_text(shared, encoding="utf-8")
    if own is not None:
        (lists / "own.ini").write_text(own, encoding="utf-8")
    return {"base_dir": work, "home": tmp_path / "home", "probe": _Probe(),
            "models_files": [lists / "shared.ini", lists / "own.ini"]}


def test_a_list_of_catalogue_files_is_read_joined_and_each_alias_names_its_file(tmp_path):
    where = _list(tmp_path)
    shared, own = where["models_files"]
    entries = {entry.alias: (entry.model, entry.file) for entry in setup.models(**where)}
    assert entries == {"local_qwen": ("qwen-2", str(own)), "cloud_coder": ("coder", str(shared))}


def test_a_change_through_a_list_overrides_an_earlier_section_whole_in_its_last_file(tmp_path):
    where = _list(tmp_path, own=None)
    shared, own = where["models_files"]
    assert setup.set_model("cloud_coder", provider="gateway", model="coder-2", **where) == own
    assert shared.read_text(encoding="utf-8") == SHARED
    assert settings._read_ini_file(own) == {"cloud_coder": {
        "provider": "gateway", "model": "coder-2", "request_timeout_seconds": "60"}}


def test_a_removal_through_a_list_takes_only_a_section_of_its_last_file(tmp_path):
    where = _list(tmp_path)
    shared, own = where["models_files"]
    with pytest.raises(setup.SetupError, match="cloud_coder comes from an earlier file"):
        setup.remove_model("cloud_coder", **where)
    assert own.read_text(encoding="utf-8") == OWN
    setup.remove_model("local_qwen", **where)
    assert "local_qwen" not in own.read_text(encoding="utf-8")
    assert {entry.alias: entry.file for entry in setup.models(**where)}["local_qwen"] == str(shared)


def test_no_key_is_written_through_a_list(tmp_path):
    """A key is refused. A new address copies the provider's section
    without the key the user directory's catalogue holds, so it needs a
    variable, and a last file holding a key is refused."""
    home, work, own = tmp_path / "home", tmp_path / "work", tmp_path / "ext" / "models.ini"
    home.mkdir()
    work.mkdir()
    catalogue = ("[gateway]\napi_url = https://gate.example/v1\napi_key = sk-home\n\n"
                 "[cloud_coder]\nprovider = gateway\nmodel = coder\n")
    (home / "models.ini").write_text(catalogue, encoding="utf-8")
    where = {"base_dir": work, "home": home, "probe": _Probe(), "models_files": [home / "models.ini", own]}
    model = {"provider": "gateway", "model": "coder", "address": "https://gate2.example/v1"}
    with pytest.raises(setup.SetupError) as refused:
        setup.set_model("cloud_coder", key="sk-new", **model, **where)
    assert refused.value.problems[0][0] == "key"
    with pytest.raises(setup.SetupError) as refused:
        setup.set_model("cloud_coder", **model, **where)
    assert refused.value.problems == (("key_variable", f"{home / 'models.ini'} holds the key of [gateway], "
                                                       f"which is not copied into {own}: name the variable "
                                                       "that holds it"),)
    assert not own.exists()
    setup.set_model("cloud_coder", key_variable="GATE_KEY", **model, **where)
    assert settings._read_ini_file(own) == {
        "gateway": {"api_url": "https://gate2.example/v1", "api_key_env": "GATE_KEY"},
        "cloud_coder": {"provider": "gateway", "model": "coder", "api_key_env": "GATE_KEY"}}
    assert (home / "models.ini").read_text(encoding="utf-8") == catalogue
    own.write_text("[gateway]\napi_url = https://gate2.example/v1\napi_key = sk-own\n", encoding="utf-8")
    with pytest.raises(setup.SetupError, match=r"\[gateway\] api_key"):
        setup.set_model("cloud_coder", key_variable="GATE_KEY", **model, **where)


def test_an_address_holding_a_credential_is_not_copied_through_a_list(tmp_path):
    shared = "[cloud_coder]\nprovider = openai\nmodel = coder\napi_url = https://gate.example/v1?api_token=secret\n"
    where = _list(tmp_path, own=None, shared=shared)
    (shown,) = setup.models(**where)
    with pytest.raises(setup.SetupError, match=r"\[cloud_coder\] in .*shared\.ini has an address that holds "
                                               "a credential, which is not copied"):
        setup.set_model("cloud_coder", provider="openai", model="coder-2", address=shown.address, **where)
    assert not where["models_files"][1].exists()


def test_a_change_that_breaks_another_alias_is_refused_and_an_earlier_broken_one_blocks_none(tmp_path):
    catalogue = ("[gateway]\nprovider = openai\nmodel = base\napi_url = https://gate.example/v1\n\n"
                 "[cloud_coder]\nprovider = gateway\nmodel = coder\n\n[broken]\nprovider = nowhere\nmodel = x\n")
    (tmp_path / "models.ini").write_text(catalogue, encoding="utf-8")
    with pytest.raises(setup.SetupError, match="leaves cloud_coder without a provider, model, address"):
        setup.remove_model("gateway", base_dir=tmp_path, home=tmp_path, probe=_Probe())
    assert (tmp_path / "models.ini").read_text(encoding="utf-8") == catalogue
    setup.set_model("fresh", provider="vllm", model="m", base_dir=tmp_path, home=tmp_path, probe=_Probe())
    assert "[fresh]" in (tmp_path / "models.ini").read_text(encoding="utf-8")


def test_a_model_only_the_last_file_holds_is_removed_through_a_list(tmp_path):
    where = _list(tmp_path, own=OWN + "\n[mine]\nprovider = vllm\nmodel = m\n")
    setup.remove_model("mine", **where)
    assert "mine" not in {entry.alias for entry in setup.models(**where)}


@pytest.mark.skipif(os.name != "posix", reason="file modes are POSIX's")
def test_the_catalogue_a_change_is_checked_with_is_readable_by_its_owner_only(tmp_path, monkeypatch):
    """The state after a change is resolved from a temporary file beside the
    last file, made owner-only, and removed afterwards."""
    import stat

    where = _list(tmp_path, own=None)
    modes, resolve = [], settings.resolve_settings

    def recording(**options):
        modes.extend(stat.S_IMODE(os.stat(path).st_mode) for path in options.get("models_files") or ()
                     if os.path.basename(path).startswith(".own.ini."))
        return resolve(**options)

    monkeypatch.setattr(settings, "resolve_settings", recording)
    setup.set_model("cloud_coder", provider="gateway", model="coder-2", **where)
    assert modes == [0o600]
    assert sorted(os.listdir(tmp_path / "lists")) == ["own.ini", "shared.ini"]
