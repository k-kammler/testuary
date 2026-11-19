"""The setup wizard's files and consent: credentials stay the user's,
provider sections change only with consent, a replacement writes exactly
what it shows, the printed commands work, and the files stay readable."""
from __future__ import annotations

import configparser
import os
import stat
import subprocess

import pytest

import test_setup_wizard as wizard
from testuary import settings
from testuary import settings as _ini_text
from testuary import setup as _setup



@pytest.fixture(autouse=True)
def _the_test_dir_is_the_user_dir(tmp_path, monkeypatch):
    """The setup writes models.ini and config.ini in the user directory. Here
    it is the test's own directory, where the project's files are too."""
    monkeypatch.setattr(settings, "platform_user_dir", lambda: tmp_path)


class _RealRestriction(wizard._Probe):
    """The fixed machine of the wizard tests, with the real restriction of
    a file to its owner. ``sizes`` and ``modes`` record each file's size and
    permission bits when it is restricted."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.sizes = []
        self.modes = []

    def restrict_to_user(self, path):
        self.sizes.append(path.stat().st_size)
        self.modes.append(stat.S_IMODE(path.stat().st_mode))
        return _setup.Probe.restrict_to_user(self, path)


@pytest.fixture
def open_umask():
    """A umask of 0 on POSIX, so a created file keeps the permissions it asks for."""
    if os.name != "posix":
        yield
        return
    previous = os.umask(0)
    try:
        yield
    finally:
        os.umask(previous)


#: The sample Python project of the wizard tests.
project = wizard.project

def _restricted_to_owner(path):
    if os.name == "posix":
        return stat.S_IMODE(path.stat().st_mode) == 0o600
    listing = subprocess.run(["icacls", str(path)], capture_output=True, text=True).stdout
    entries = [line.strip() for line in listing.splitlines()[:-1] if line.strip()]
    entries[0] = entries[0][len(str(path)):].strip()
    return len(entries) == 1 and "(I)" not in entries[0] and ":(F)" in entries[0]


def test_models_ini_with_a_key_stays_the_users_after_every_write(tmp_path, open_umask):
    probe = _RealRestriction(ollama=False, models=("deepseek-chat", "deepseek-coder"))
    wizard._run(tmp_path, {"model.provider": "deepseek", "model.key_storage": "file",
                           "model.key": "sk-first", "model.name": "deepseek-chat"},
                probe=probe, part="model")
    models = tmp_path / "models.ini"
    assert _restricted_to_owner(models)
    # A later alias-only edit with the key left to the environment: the file
    # still holds the first key and stays restricted.
    wizard._run(tmp_path, {"model.provider": "deepseek", "model.key_storage": "environment",
                           "model.name": "deepseek-coder"}, probe=probe, part="model")
    assert "api_key = sk-first" in models.read_text(encoding="utf-8")
    assert _restricted_to_owner(models)
    # Each temporary file was restricted while still empty, and none is left.
    assert probe.sizes == [0, 0]
    if os.name == "posix":
        # Owner-only already then: created with 0o600, whatever the umask.
        assert probe.modes == [0o600, 0o600]
    assert [path.name for path in tmp_path.iterdir() if path.name.startswith(".models.ini.")] == []


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_other_files_keep_their_mode(tmp_path):
    config = tmp_path / "config.ini"
    config.write_text("[mutation]\n", encoding="utf-8")
    os.chmod(config, 0o644)
    wizard._run(tmp_path, {"machine.mutation_workers": "4"}, part="machine")
    assert stat.S_IMODE(config.stat().st_mode) == 0o644


def test_a_second_local_service_gets_its_own_section(tmp_path):
    original = ("[local]\napi_url = http://localhost:8000/v1\n\n"
                "[local_m1]\nprovider = local\nmodel = m1\n")
    (tmp_path / "models.ini").write_text(original, encoding="utf-8")
    probe = wizard._Probe(ollama=False, models=("m2",))
    wizard._run(tmp_path, {"model.provider": "openai-compatible",
                           "model.address": "http://localhost:8001/v1", "model.key_storage": "environment",
                           "model.name": "m2"}, probe=probe, part="model")
    text = (tmp_path / "models.ini").read_text(encoding="utf-8")
    assert text.startswith(original)
    assert "[local2]\napi_url = http://localhost:8001/v1\n" in text
    assert "[local2_m2]\nprovider = local2\n" in text


def test_the_walk_refuses_an_alias_named_like_a_provider_section(tmp_path):
    """Writing such an alias would replace the provider's section whole and
    drop its stored key, so the walk refuses the name as set_model does."""
    original = "[gateway]\napi_url = https://gateway.example.org/v1\napi_key = sk-stored\n"
    (tmp_path / "models.ini").write_text(original, encoding="utf-8")
    with pytest.raises(_setup.SetupError, match="no valid answer to 'A short name for this model"):
        wizard._run(tmp_path, {"model.provider": "openai-compatible",
                               "model.address": "https://api.deepseek.com/v1",
                               "model.key_storage": "environment", "model.name": "chat",
                               "model.alias": ["gateway"] * 5, "model.confirm": "yes"},
                    probe=wizard._Probe(ollama=False, models=("chat",)), part="model")
    assert (tmp_path / "models.ini").read_text(encoding="utf-8") == original


def test_a_built_in_provider_is_no_service_name(tmp_path):
    with pytest.raises(_setup.SetupError, match="no valid answer to 'A short name for this service"):
        wizard._run(tmp_path, {"model.provider": "openai-compatible",
                               "model.address": "https://api.deepseek.com/v1",
                               "model.provider_name": ["openai"] * 5},
                    probe=wizard._Probe(ollama=False), part="model")


@pytest.mark.parametrize("consent", ["", "yes"])
def test_a_changed_provider_key_needs_its_own_consent(tmp_path, consent):
    original = ("[openai]\napi_key = sk-old\n\n"
                "[gpt_a]\nprovider = openai\nmodel = gpt-4o\n")
    models = tmp_path / "models.ini"
    models.write_bytes(original.encode("utf-8"))
    _result, ask, said = wizard._run(tmp_path, {
        "model.provider": "openai", "model.key_storage": "file", "model.key": "sk-new",
        "model.name": "gpt-4o", "model.alias": "gpt_b", "model.provider_confirm": consent},
        probe=wizard._Probe(ollama=False, models=("gpt-4o",)), part="model")
    assert "model.provider_confirm" in ask.asked
    if consent:
        assert "api_key = sk-new" in models.read_text(encoding="utf-8")
    else:
        assert models.read_bytes() == original.encode("utf-8")
        assert "  Nothing written to models.ini." in said


def test_the_provider_consent_names_its_aliases(tmp_path):
    (tmp_path / "models.ini").write_text(
        "[openai]\napi_key = sk-old\n\n[gpt_a]\nprovider = openai\nmodel = gpt-4o\n", encoding="utf-8")
    seen = []

    def ask(question):
        seen.append(question)
        return {"model.provider": "openai", "model.key_storage": "file", "model.key": "sk-new",
                "model.name": "gpt-4o", "model.alias": "gpt_b"}.get(question.key, "")

    _setup.run_setup(ask, wizard._Probe(ollama=False, models=("gpt-4o",)), lambda line: None,
                     tmp_path, part="model")
    consent = next(question for question in seen if question.key == "model.provider_confirm")
    assert consent.prompt == "Replace api_key of [openai] in models.ini? The aliases gpt_a use it"
    assert consent.default == "no"


def test_a_file_key_that_outranks_the_variable_is_named(tmp_path):
    (tmp_path / "models.ini").write_text(
        "[deepseek]\napi_url = https://api.deepseek.com/v1\napi_key = sk-file\n", encoding="utf-8")
    result, _ask, said = wizard._run(tmp_path, {
        "model.provider": "deepseek", "model.key_storage": "environment", "model.name": "deepseek-chat"},
        probe=wizard._Probe(ollama=False, models=("deepseek-chat",),
                            environment={"TESTUARY_DEEPSEEK_API_KEY": "sk-variable"}), part="model")
    assert any("takes precedence over TESTUARY_DEEPSEEK_API_KEY" in line for line in said)
    assert "remove api_key from [deepseek] in models.ini to use TESTUARY_DEEPSEEK_API_KEY" in result.missing


def test_a_confirmed_update_replaces_exactly_the_asked_keys(tmp_path):
    project = tmp_path / "proj"
    (project / "pkg").mkdir(parents=True)
    (project / "pkg" / "core.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (project / ".testuary").mkdir()
    settings_file = project / ".testuary" / "project.ini"
    settings_file.write_text(
        "[project]\n# my notes\nlanguage = java\npython_import_root = lib\nhint_profile = speed\n"
        "requirements = req.txt\n  continued\n\n[api]\nmodel = old_alias\n", encoding="utf-8")
    _result, _ask, said = wizard._run(tmp_path, {"project.folder": "proj", "model": "",
                                                 "python.import_root": ".",
                                                 "project.confirm": "yes"}, part="project")
    shown = next(text for text in said if text.startswith("[project]"))
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(settings_file, encoding="utf-8")
    written = {f"{key} = {value}" for key, value in parser["project"].items()}
    assert set(shown.splitlines()[1:]) <= written
    # The answered keys hold exactly the answers, and the others stay.
    assert "python_import_root" not in parser["project"] and not parser.has_option("api", "model")
    assert parser["project"]["hint_profile"] == "speed"
    assert parser["project"]["requirements"] == "req.txt\ncontinued"
    assert "# my notes" in settings_file.read_text(encoding="utf-8")


def test_the_closing_line_links_the_online_settings_reference_of_this_version(project):
    from testuary import __version__

    _result, _ask, said = wizard._run(project, {"model.name": "qwen3:8b", "project.folder": "proj"})
    url = f"https://github.com/k-kammler/testuary/blob/v{__version__}/docs/reference/settings.md"
    assert said[-1].endswith(f"see the settings reference ({url}).")


def test_the_printed_verify_command_parses():
    from testuary.__main__ import _build_parser

    args = _build_parser(checking=True).parse_args(["--verify"])
    assert args.verify and args.files == []


def test_a_header_comment_and_a_key_in_another_case_are_edited_in_place():
    text = "[app] ; my project\nSource_Files = a.py\n"
    edited = _ini_text.set_values(text, "app", [("source_files", "b.py"), ("model", "m")])
    assert edited == "[app] ; my project\nsource_files = b.py\nmodel = m\n"
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(edited)
    assert dict(parser["app"]) == {"source_files": "b.py", "model": "m"}


def test_an_unreadable_project_settings_file_is_one_error_line(tmp_path, monkeypatch, capsys):
    (tmp_path / "a.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / ".testuary").mkdir()
    (tmp_path / ".testuary" / "project.ini").write_text("[a]\nx = 1\n[a]\ny = 2\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    assert _setup.main(["project"], ask=wizard._Answers({"project.confirm": "yes"}),
                       probe=wizard._Probe()) == 1
    err = capsys.readouterr().err
    assert err.startswith("testuary setup: invalid INI configuration at ")
    assert "DuplicateSectionError" in err and "Traceback" not in err


# ---- declined parts and usable names -----------------------------------------

def test_a_declined_part_is_named_and_no_variable_for_an_unwritten_alias(project):
    result, _ask, _said = wizard._run(project, {
        "model.provider": "openai", "model.name": "gpt-4o", "model.confirm": "no",
        "machine.confirm": "no", "project.folder": "proj", "project.confirm": "no"},
        probe=wizard._Probe(ollama=False, models=("gpt-4o",)))
    assert result.missing == [
        "model access: run testuary setup model",
        "parallel work on this machine (defaults apply): run testuary setup machine",
        "a project: run testuary setup project"]


def test_names_are_ones_the_run_can_use(project):
    assert _setup._label("My-Project.v2") == "my_project_v2"
    # A name given with the project part is the project's label.
    wizard._run(project, {"project.folder": "proj", "model": ""}, part="project", name="model")
    assert "[project]\nname = model\n" in wizard._settings_file(project).read_text(encoding="utf-8")


def test_every_default_name_passes_its_own_check(tmp_path):
    assert _setup._free_service_name({}, "openai", "https://api.openai.com/v1") == "openai2"
    catalogue = {"local": {"provider": "ollama", "model": "m"}}
    assert _setup._free_service_name(catalogue, "local", "http://localhost:8000/v1") == "local2"


def test_only_a_section_with_a_provider_is_an_alias(project):
    (project / "models.ini").write_text(
        "[deepseek]\napi_url = https://api.deepseek.com/v1\n\n"
        "[ds]\nprovider = deepseek\nmodel = deepseek-chat\n", encoding="utf-8")
    with pytest.raises(_setup.SetupError, match="no valid answer to 'The model, an alias"):
        wizard._run(project, {"project.folder": "proj", "model": ["deepseek"] * 5}, part="project")
    wizard._run(project, {"project.folder": "proj", "model": "ds"}, part="project")
    assert "[api]\nmodel = ds\n" in wizard._settings_file(project).read_text(encoding="utf-8")


def test_the_project_itself_is_an_import_dir_when_src_exists(project):
    wizard._run(project, {"project.folder": "proj", "python.import_root": ".", "model": ""},
                part="project")
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(wizard._settings_file(project), encoding="utf-8")
    assert "python_import_root" not in parser["project"]


@pytest.mark.parametrize("config, expected", [
    ("", {"concurrent_requests", "concurrent_requests_ollama"}),
    ("[api]\nmodel = local_qwen\n", {"concurrent_requests_ollama"}),
])
def test_the_machine_part_alone_writes_the_starter_limit_the_run_reads(tmp_path, config, expected):
    (tmp_path / "models.ini").write_text(
        "[local_qwen]\nprovider = ollama\nmodel = qwen3\napi_url = http://localhost:11434/v1\n\n"
        "[other]\nprovider = openai\nmodel = gpt-4o\napi_url = http://localhost:9/v1\n", encoding="utf-8")
    if config:
        (tmp_path / "config.ini").write_text(config, encoding="utf-8")
    wizard._run(tmp_path, {}, part="machine")
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(tmp_path / "config.ini", encoding="utf-8")
    assert set(parser["starter"]) == expected


def test_each_unusable_named_file_is_refused_with_its_reason(tmp_path, monkeypatch, capsys):
    import sys

    from testuary import __main__ as cli

    project = tmp_path / "proj"
    (project / ".testuary").mkdir(parents=True)
    (project / "a.py").write_text("x = 1\n", encoding="utf-8")
    (project / "notes.txt").write_text("", encoding="utf-8")
    (project / "pkg").mkdir()
    (tmp_path / "outside.py").write_text("x = 1\n", encoding="utf-8")
    (project / ".testuary" / "project.ini").write_text(
        "[project]\nlanguage = python\nsource_files = a.py\n", encoding="utf-8")
    monkeypatch.chdir(project)
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    for file, reason in (("../outside.py", "outside the source directory"),
                         ("notes.txt", "not a python source file"),
                         ("missing.py", "missing.py does not exist"),
                         ("pkg", "pkg is a directory, not a source file")):
        monkeypatch.setattr(sys, "argv", ["testuary", file, "--display", "plain"])
        try:
            status = cli.main()
        except SystemExit as stop:
            status = stop.code
        assert status == 2
        assert reason in "".join(capsys.readouterr())
    assert not (project / ".testuary" / "results").exists()


@pytest.mark.parametrize("java_home, on_path, expected", [
    (False, "jdk21", ("jdk21", "the java on PATH, JDK 21")),
    (True, "", ("jdk21", "from JAVA_HOME")),
    (False, "", ("", "JAVA_HOME is not set and no java is on PATH")),
    (False, "jdk8", ("", "JAVA_HOME is not set and the java on PATH is JDK 8")),
])
def test_the_jdk_question_says_what_it_looked_at(tmp_path, java_home, on_path, expected):
    jdks = {}
    for name in ("jdk21", "jdk8"):
        jdks[name] = tmp_path / "opt" / name
        (jdks[name] / "bin").mkdir(parents=True)
        (jdks[name] / "bin" / "java").write_text("", encoding="utf-8")
    java = tmp_path / "jproj"
    (java / "src/main/java/demo").mkdir(parents=True)
    (java / "src/main/java/demo/Calc.java").write_text("class Calc {}\n", encoding="utf-8")
    # Maven's wrapper is mvnw.cmd on Windows and the shell script mvnw elsewhere.
    for wrapper in ("mvnw", "mvnw.cmd"):
        (java / wrapper).write_text("", encoding="utf-8")
    (java / "pom.xml").write_text("<project/>", encoding="utf-8")

    class _Path(wizard._Probe):
        def which(self, command):
            return str(jdks[on_path] / "bin" / "java") if command == "java" and on_path else ""

        def read_text(self, path):
            home = str(path.parent)
            version = "1.8.0" if "jdk8" in home else "21" if "jdk21" in home else None
            return f'JAVA_VERSION="{version}"\n' if version else None

    probe = _Path(environment={"JAVA_HOME": str(jdks["jdk21"])} if java_home else {})
    seen = []

    def ask(question):
        seen.append(question)
        if question.key == "java.jdk" and not question.default:
            return str(jdks["jdk21"])
        return {"project.folder": "jproj", "model": ""}.get(question.key, "")

    _setup.run_setup(ask, probe, lambda line: None, tmp_path, part="project")
    question = next(item for item in seen if item.key == "java.jdk")
    default = str(jdks[expected[0]].resolve()) if expected[0] and not java_home else (
        str(jdks[expected[0]]) if expected[0] else "")
    assert (question.default, question.reason) == (default, expected[1])
    assert f"java_home = {jdks['jdk21'].resolve() if not java_home else jdks['jdk21']}" in (
        tmp_path / "jproj" / ".testuary" / "project.ini").read_text(encoding="utf-8")


def test_the_help_names_the_demonstration_and_a_new_models_ini_no_examples(tmp_path):
    assert "testuary setup --demo" in _setup._parser().format_help()
    wizard._run(tmp_path, {"model.name": "qwen3:8b"}, part="model")
    assert "example" not in (tmp_path / "models.ini").read_text(encoding="utf-8").lower()


# ---- key variables, template notes and suggested names ------------------------

def test_a_custom_key_variable_belongs_to_the_new_alias_only(tmp_path):
    """A custom key variable goes to the new alias: in the provider's
    section it would change every alias of the service, without a question."""
    from testuary.models import resolve_provider_credential

    (tmp_path / "models.ini").write_text(
        "[deepseek]\napi_url = https://api.deepseek.com/v1\napi_key = sk-file\n\n"
        "[ds_old]\nprovider = deepseek\nmodel = deepseek-chat\n", encoding="utf-8")
    _result, ask, _said = wizard._run(tmp_path, {
        "model.provider": "deepseek", "model.key_storage": "environment",
        "model.key_variable": "MY_DS_KEY", "model.name": "deepseek-chat", "model.alias": "ds_new"},
        probe=wizard._Probe(ollama=False, models=("deepseek-chat",)), part="model")
    assert "model.provider_confirm" not in ask.asked
    models = wizard_models(tmp_path)
    assert "api_key_env" not in models["deepseek"]
    assert models["ds_new"]["api_key_env"] == "MY_DS_KEY"
    # ds_old keeps its file key while MY_DS_KEY is unset.
    assert resolve_provider_credential("deepseek", models["deepseek"], models["ds_old"], {}) == "sk-file"


def wizard_models(user_dir):
    parser = configparser.ConfigParser(interpolation=None)
    parser.read(user_dir / "models.ini", encoding="utf-8")
    return {section: dict(parser[section]) for section in parser.sections()}


def test_every_suggested_name_passes_the_names_check():
    """Each suggested name passes the check. OpenAI would otherwise
    suggest openai, a built-in provider's name."""
    check = _setup._service_name_check({"local": {"provider": "ollama", "model": "m"}})
    for answer, suggested in (("OpenAI", "openai2"), ("Local", "local2"), ("Setup", "setup")):
        problem = check(answer)
        assert problem.endswith(f"for example {suggested}") and check(suggested) is None
    assert _setup._label_check()("My-App").endswith("for example my_app")


def test_a_service_that_answers_without_a_key_needs_no_variable(tmp_path):
    """No contradictory lines, nothing asked for, and no variable that
    would refuse every run while unset."""
    result, _ask, said = wizard._run(tmp_path, {
        "model.provider": "openai-compatible", "model.address": "http://localhost:8001/v1",
        "model.key_storage": "environment", "model.name": "m2"},
        probe=wizard._Probe(ollama=False, models=("m2",)), part="model")
    assert not any("is not set here" in line or "Access confirmed" in line for line in said)
    assert any("without an API key, so none is needed" in line for line in said)
    assert not any("environment variable" in item for item in result.missing)
    assert all("api_key_env" not in fields for fields in wizard_models(tmp_path).values())


def test_a_service_whose_list_cannot_be_read_still_names_the_variable(tmp_path):
    class _Unreadable(wizard._Probe):
        def model_list(self, provider, address, key, *, timeout=15):
            return None

    result, _ask, said = wizard._run(tmp_path, {
        "model.provider": "openai-compatible", "model.address": "http://localhost:8001/v1",
        "model.key_storage": "environment", "model.name": "m2"},
        probe=_Unreadable(ollama=False), part="model")
    assert any("not set here: set it before a run if the service needs an API key" in line
               for line in said)
    assert "set the environment variable TESTUARY_LOCAL_API_KEY" in result.missing
