"""``testuary setup``: the three parts write the keys a user writes by
hand, in the same files, each only when not yet set, and a key never
appears anywhere but its chosen place."""
from __future__ import annotations

import re

import pytest

from testuary import settings
from testuary import settings as _ini_text
from testuary import setup as _setup


class _Probe(_setup.Probe):
    """The real file system, with the services, interpreters and JDKs of a
    fixed machine."""

    def __init__(self, *, ollama=True, models=("qwen3:8b",), environment=None, missing_tools=()):
        self.missing_tools = list(missing_tools)
        self.ollama = ollama
        self.models = list(models)
        self.env = dict(environment or {})
        self.restricted = []
        self.listed_with = []

    def cpu_count(self):
        return 16

    def environment(self, name):
        return self.env.get(name, "")

    def local_ollama(self):
        return self.ollama

    def model_list(self, provider, address, key, *, timeout=15):
        self.listed_with.append((provider.kind, address, bool(key)))
        return list(self.models)

    def run(self, argv, timeout):
        """Every interpreter is a Python 3.12.4, its version the answer to the
        script run alone, that lacks ``missing_tools`` among the tools a
        script names after it."""
        if len(argv) == 3:
            return 0, "3.12.4\n"
        return 0, " ".join(module for module in argv[3:]
                           if module.replace("_", "-") in self.missing_tools) + "\n"

    def read_text(self, path):
        """A JDK 21 wherever its home's name holds jdk."""
        return 'JAVA_VERSION="21"\n' if "jdk" in str(path.parent) else None

    def restrict_to_user(self, path):
        self.restricted.append(path)
        return True


class _Answers:
    def __init__(self, answers):
        self.answers = dict(answers)
        self.asked = []
        self.questions = {}

    def __call__(self, question):
        self.asked.append(question.key)
        self.questions[question.key] = question
        if question.key not in self.answers:
            return ""
        answer = self.answers[question.key]
        return answer.pop(0) if isinstance(answer, list) else answer


@pytest.fixture
def project(tmp_path):
    package = tmp_path / "proj" / "src" / "pkg"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "core.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (package / "util.py").write_text("def g():\n    return 2\n", encoding="utf-8")
    tests = tmp_path / "proj" / "tests"
    tests.mkdir()
    (tests / "test_core.py").write_text("def test_f():\n    pass\n", encoding="utf-8")
    return tmp_path


def _user(name):
    """A file of the user directory, which the suite gives each test of its own."""
    return settings.platform_user_dir() / name


def _settings_file(project, project_dir="proj"):
    """The project's settings file the setup writes in ``project_dir``."""
    return project / project_dir / ".testuary" / "project.ini"


def _run(base_dir, answers, probe=None, **kwargs):
    said = []
    ask = _Answers(answers)
    result = _setup.run_setup(ask, probe or _Probe(), said.append, base_dir, **kwargs)
    return result, ask, said


def test_the_walk_writes_model_machine_and_project_that_a_run_reads(project):
    result, ask, said = _run(project, {"model.name": "qwen3:8b", "project.folder": "proj"})
    assert ask.asked == [
        "model.run", "model.provider", "model.name", "model.alias", "model.confirm",
        "machine.run", "machine.mutation_workers", "machine.starter_requests", "machine.confirm",
        "project.run", "project.folder", "project.language", "python.import_root",
        "python.interpreter", "targets", "model", "project.confirm"]
    models = _user("models.ini").read_text(encoding="utf-8")
    assert "[ollama_qwen3_8b]\nprovider = ollama\nmodel = qwen3:8b\napi_url = http://localhost:11434/v1\n" in models
    config = _user("config.ini").read_text(encoding="utf-8")
    assert "[mutation]\nengine_workers = 8\n" in config
    assert "[starter]\nconcurrent_requests_ollama = 1\n" in config
    text = _settings_file(project).read_text(encoding="utf-8")
    # Test files and an empty __init__.py are no targets.
    assert "source_files = src/pkg/core.py, src/pkg/util.py\n" in text
    assert text.endswith("\n[api]\nmodel = ollama_qwen3_8b\n")
    assert (project / "proj" / ".testuary" / "results" / ".gitignore").read_bytes() == b"*\n"
    assert result.project == str((project / "proj").resolve())
    assert result.run == ("testuary",) and result.missing == []
    assert said[-2] == f"Run it in {(project / 'proj').resolve()} with: testuary"

    from testuary import config as config_module
    from testuary import models as models_module
    from testuary.settings import resolve_settings

    settings = resolve_settings(base_dir=project / "proj" / "src", environment={})
    with config_module.use_run_settings(settings):
        assert config_module.get_agent_config("test_mutation")["engine_workers"] == 8
        assert config_module.get_project_config()["python_import_root"] == "src"
        assert models_module.get_api_config()["api_url"] == "http://localhost:11434/v1"


def test_the_import_dir_question_says_what_the_dir_is(project):
    _, ask, _ = _run(project, {"model.name": "qwen3:8b", "project.folder": "proj"})
    question = ask.questions["python.import_root"]
    assert "Python packages are in" in question.prompt and "src" in question.prompt
    assert "calc.ops" in question.reason


def test_a_second_project_asks_only_the_project_questions(project):
    _run(project, {"model.name": "qwen3:8b", "project.folder": "proj"})
    (project / "other").mkdir()
    (project / "other" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    _result, ask, said = _run(project, {"project.folder": "other"})
    assert ask.asked[:2] == ["project.run", "project.folder"]
    assert "Model access: ollama_qwen3_8b, set." in said
    assert "This machine: parallel work set." in said
    assert "source_files = mod.py\n" in _settings_file(project, "other").read_text(encoding="utf-8")


def test_a_key_kept_in_the_file_is_hidden_everywhere_else(project):
    secret = "sk-not-shown-anywhere"
    probe = _Probe(ollama=False, models=("deepseek-chat",))
    result, _ask, said = _run(project, {
        "model.provider": "deepseek", "model.key_storage": "file", "model.key": secret,
        "model.name": "deepseek-chat"}, probe=probe, part="model")
    models = _user("models.ini").read_text(encoding="utf-8")
    assert f"[deepseek]\napi_url = https://api.deepseek.com/v1\napi_key = {secret}\n" in models
    assert "[deepseek_deepseek_chat]\nprovider = deepseek\nmodel = deepseek-chat\n" in models
    # The temporary file is restricted before the key is written into it,
    # and the rename keeps that access.
    (restricted,) = probe.restricted
    assert restricted.parent == settings.platform_user_dir() and restricted.name.startswith(".models.ini.")
    assert not restricted.exists() and _user("models.ini").is_file()
    assert probe.listed_with == [("deepseek", "https://api.deepseek.com/v1", True)]
    assert not any(secret in line for line in said)
    assert secret not in repr(result)
    assert any("api_key = (hidden)" in line for line in said)


def test_a_key_left_to_the_environment_is_not_written(project):
    probe = _Probe(ollama=False, models=("gpt-4o",))
    result, _ask, said = _run(project, {
        "model.provider": "openai", "model.name": "gpt-4o"}, probe=probe, part="model")
    models = _user("models.ini").read_text(encoding="utf-8")
    assert "api_key =" not in models.split("[openai_gpt_4o]")[1]
    # The template header describes api_key_env. No setting may carry it.
    assert not any(line.startswith("api_key_env") for line in models.splitlines())
    assert result.missing == ["set the environment variable TESTUARY_OPENAI_API_KEY"]
    assert probe.listed_with == []


def test_an_existing_file_keeps_its_comments_and_other_sections(project):
    original = "# My models.\n[mine]\n# the one I use\nprovider = openai\nmodel = gpt-4o\napi_key = k\n"
    _user("models.ini").write_bytes(original.replace("\n", "\r\n").encode("utf-8"))
    _run(project, {"model.provider": "ollama-local", "model.name": "qwen3:8b"}, part="model")
    text = _user("models.ini").read_bytes().decode("utf-8")
    assert text.startswith(original.replace("\n", "\r\n"))
    assert "\r\n[ollama_qwen3_8b]\r\nprovider = ollama\r\n" in text


def test_a_named_part_asks_even_when_set_and_an_existing_section_needs_consent(project):
    _run(project, {"model.name": "qwen3:8b", "project.folder": "proj"})
    before = _settings_file(project).read_text(encoding="utf-8")
    _result, ask, _said = _run(project, {"project.folder": "proj", "targets": "2"}, part="project")
    assert "project.run" not in ask.asked
    assert _settings_file(project).read_text(encoding="utf-8") == before
    _run(project, {"project.folder": "proj", "targets": "2", "project.confirm": "yes"}, part="project")
    assert "source_files = src/pkg/util.py\n" in _settings_file(project).read_text(encoding="utf-8")
    _result, ask, _said = _run(project, {"machine.mutation_workers": "4"}, part="machine")
    assert ask.asked == ["machine.mutation_workers", "machine.starter_requests", "machine.confirm"]
    assert "engine_workers = 4" in _user("config.ini").read_text(encoding="utf-8")


@pytest.mark.parametrize("variable", ["TESTUARY_STARTER_CONCURRENT_REQUESTS",
                                      "TESTUARY_STARTER_CONCURRENT_REQUESTS_OLLAMA"])
def test_a_variable_that_sets_the_starter_requests_keeps_them_out_of_the_walk(project, variable):
    probe = _Probe(environment={variable: "3"})
    _result, ask, _said = _run(project, {"model.name": "qwen3:8b", "project.folder": "proj"}, probe)
    assert "machine.mutation_workers" in ask.asked
    assert "machine.starter_requests" not in ask.asked


def test_a_skipped_part_is_named_in_the_closing_check(project):
    result, ask, said = _run(project, {"model.run": "no", "machine.run": "no", "project.run": "no"})
    assert result.missing == [
        "model access: run testuary setup model",
        "parallel work on this machine (defaults apply): run testuary setup machine",
        "a project: run testuary setup project"]
    assert not _user("models.ini").exists() and not (project / "proj" / ".testuary").exists()
    assert said[-2].startswith("Still missing:")


def test_an_invalid_answer_is_asked_again_and_then_stops(project):
    _result, ask, said = _run(project, {"machine.mutation_workers": ["zero", "-1", "6"]}, part="machine")
    assert ask.asked.count("machine.mutation_workers") == 3
    assert said.count("  a whole number above zero") == 2
    with pytest.raises(_setup.SetupError, match="no valid answer to 'How many mutation-testing"):
        _run(project, {"machine.mutation_workers": ["x"] * 5}, part="machine")


def test_a_java_project_names_its_jdk_and_build_tool(project):
    java = project / "jproj"
    (java / "src/main/java/demo").mkdir(parents=True)
    (java / "src/main/java/demo/Calc.java").write_text("class Calc {}\n", encoding="utf-8")
    (java / "src/test/java/demo").mkdir(parents=True)
    (java / "src/test/java/demo/CalcTest.java").write_text("class CalcTest {}\n", encoding="utf-8")
    (java / "pom.xml").write_text("<project/>", encoding="utf-8")
    maven = project / "mvn.cmd"
    maven.write_text("", encoding="utf-8")
    _run(project, {"project.folder": "jproj", "java.jdk": ["/no/java", "/opt/jdk21"],
                   "java.build": str(maven), "model": ""}, part="project")
    text = _settings_file(project, "jproj").read_text(encoding="utf-8")
    assert "language = java\njava_home = /opt/jdk21\n" in text
    assert f"maven = {maven}\n" in text
    assert "source_files = src/main/java/demo/Calc.java\n" in text


def _section_names(text):
    """The section names in ``text``, in order."""
    return re.findall(r"^\s*\[([^\]]+)\]", text, re.M)


@pytest.mark.parametrize("newline", ["\n", "\r\n"])
def test_the_text_edit_replaces_inserts_and_appends_without_touching_the_rest(newline):
    text = newline.join(["# top", "[a]", "x = 1", "", "# about b", "[b]", "y = 2", ""])
    edited = _ini_text.set_values(text, "a", [("x", "5"), ("z", "6")])
    assert edited == newline.join(["# top", "[a]", "x = 5", "z = 6", "", "# about b", "[b]", "y = 2", ""])
    appended = _ini_text.set_values(text, "c", [("w", "7")])
    assert appended == text + newline + newline.join(["[c]", "w = 7", ""])
    assert _section_names(appended) == ["a", "b", "c"]
    with pytest.raises(ValueError):
        _ini_text.set_values(text, "a", [("x", "two\nlines")])


def test_the_real_probe_restricts_a_file_to_its_owner(tmp_path):
    import os
    import stat

    path = tmp_path / "models.ini"
    path.write_text("[x]\n", encoding="utf-8")
    assert _setup.Probe().restrict_to_user(path) is (os.name in ("posix", "nt"))
    if os.name == "posix":
        assert stat.S_IMODE(path.stat().st_mode) == 0o600
    assert path.read_text(encoding="utf-8") == "[x]\n"


def test_the_real_probe_survives_undecodable_output_and_gives_no_stdin(monkeypatch):
    """A candidate interpreter that writes bytes the locale cannot decode
    answers, and one that reads its input does not wait for the timeout."""
    import subprocess
    import sys

    code, output = _setup.Probe().run(
        [sys.executable, "-c", "import sys; sys.stdout.buffer.write(bytes([0x81, 0xff]))"], 30)
    assert code == 0 and output
    calls = []
    monkeypatch.setattr(subprocess, "run", lambda argv, **options: calls.append(options)
                        or subprocess.CompletedProcess(argv, 0, "", ""))
    _setup.Probe().run(["python"], 30)
    assert calls[0]["stdin"] is subprocess.DEVNULL


def test_the_command_runs_one_part_and_checks_a_written_project(project, monkeypatch, capsys):
    monkeypatch.chdir(project)
    checked = []
    status = _setup.main(["project", "calc"], ask=_Answers({"project.folder": "proj", "model": ""}),
                         probe=_Probe(), check=lambda project_dir: checked.append(project_dir) or 0)
    project_dir = (project / "proj").resolve()
    assert status == 0 and checked == [str(project_dir)]
    # The name is the project's label in results and records.
    assert "[project]\nname = calc\n" in _settings_file(project).read_text(encoding="utf-8")
    assert f"Checking the project in {project_dir} (quick check):" in capsys.readouterr().out


def test_the_command_refuses_a_name_without_the_project_part(project, monkeypatch):
    monkeypatch.chdir(project)
    with pytest.raises(SystemExit) as refused:
        _setup.main(["machine", "proj"], ask=_Answers({}), probe=_Probe())
    assert refused.value.code == 2


def test_the_command_reports_a_stop_and_writes_nothing(project, monkeypatch, capsys):
    monkeypatch.chdir(project)
    status = _setup.main(["machine"], ask=_Answers({"machine.mutation_workers": ["x"] * 5}), probe=_Probe())
    assert status == 1 and not _user("config.ini").exists()
    assert ("testuary setup: no valid answer to 'How many mutation-testing processes may run "
            "at once?'") in capsys.readouterr().err


def test_the_terminal_hides_a_secret_and_needs_a_terminal(monkeypatch, capsys):
    import builtins
    import getpass

    prompts = []
    monkeypatch.setattr(builtins, "input", lambda prompt: prompts.append(("input", prompt)) or "")
    monkeypatch.setattr(getpass, "getpass", lambda prompt: prompts.append(("hidden", prompt)) or "k")
    _setup.terminal_ask(_setup.Question("machine.mutation_workers", "Workers?", default="8", reason="16 CPUs"))
    assert _setup.terminal_ask(_setup.Question("model.key", "The key", secret=True)) == "k"
    assert prompts == [("input", "Workers? [8]: "), ("hidden", "The key: ")]
    assert "(16 CPUs)" in capsys.readouterr().out

    def closed(prompt):
        raise EOFError

    monkeypatch.setattr(builtins, "input", closed)
    with pytest.raises(_setup.SetupError, match="needs answers from a terminal"):
        _setup.terminal_ask(_setup.Question("project.folder", "Directory?"))


def test_the_entry_runs_the_setup_command(project, monkeypatch):
    import builtins
    import sys

    from testuary import __main__ as cli

    monkeypatch.chdir(project)
    answers = iter(["4", "2", "yes"])
    monkeypatch.setattr(builtins, "input", lambda prompt: next(answers))
    monkeypatch.setattr(sys, "argv", ["testuary", "setup", "machine"])
    with pytest.raises(SystemExit) as done:
        cli.main()
    assert done.value.code == 0
    config = _user("config.ini").read_text(encoding="utf-8")
    assert "engine_workers = 4" in config and "concurrent_requests = 2" in config


def test_a_service_is_chosen_by_number_and_another_api_gets_a_meaningful_name(project):
    probe = _Probe(ollama=False, models=("big-model",))
    _result, ask, said = _run(project, {
        "model.provider": "6", "model.address": "https://api.together.xyz/v1",
        "model.key_storage": "environment", "model.name": "big-model"}, probe=probe, part="model")
    assert said[1].startswith("  1. Ollama on this machine (ollama-local)")
    text = _user("models.ini").read_text(encoding="utf-8")
    assert "[together]\napi_url = https://api.together.xyz/v1\n" in text
    assert "[together_big_model]\nprovider = together\nmodel = big-model\n" in text
    # A new models.ini holds the template's comments, not its example aliases.
    assert _section_names(text) == ["together", "together_big_model"]
    assert text.startswith("#")
    assert _setup._service_name("http://localhost:8000/v1") == "local"


def test_missing_test_tools_are_named_at_the_interpreter_question(project):
    result, _ask, said = _run(project, {"project.folder": "proj", "model": ""},
                              probe=_Probe(missing_tools=("pytest", "cosmic-ray")), part="project")
    assert any("Not installed in this interpreter: pytest, cosmic-ray" in line for line in said)
    # The advice names the versions LangAdapter pins for the interpreter.
    assert any(re.match(r"install pytest==\S+, cosmic-ray==\S+ in ", item) for item in result.missing)


def test_testuarys_own_interpreter_is_named_as_such(project):
    seen = []

    def ask(question):
        seen.append(question)
        return {"project.folder": "proj", "model": ""}.get(question.key, "")

    _setup.run_setup(ask, _Probe(), lambda line: None, project, part="project")
    interpreter = next(question for question in seen if question.key == "python.interpreter")
    assert "interpreter Testuary runs in" in interpreter.reason


def test_a_java_project_without_a_build_file_is_told_so(project):
    java = project / "jproj"
    (java / "src/main/java/demo").mkdir(parents=True)
    (java / "src/main/java/demo/Calc.java").write_text("class Calc {}\n", encoding="utf-8")
    result, ask, said = _run(project, {"project.folder": "jproj", "java.jdk": "/opt/jdk21", "model": ""},
                             part="project")
    assert "java.build" not in ask.asked
    assert any("no Maven or Gradle build file" in line for line in said)
    assert any(item.startswith("a pom.xml or build.gradle in ") for item in result.missing)


@pytest.mark.parametrize("status", [0, 1])
def test_the_run_command_follows_only_a_clean_check(project, monkeypatch, capsys, status):
    monkeypatch.chdir(project)
    code = _setup.main(["project"], ask=_Answers({"project.folder": "proj", "model": ""}),
                       probe=_Probe(), check=lambda project_dir: status)
    out = capsys.readouterr().out
    where = f" in {(project / 'proj').resolve()}"
    after = out.split("Checking the project")[1]
    assert code == status
    assert (f"Run it{where} with: testuary. To try the project's tools first: "
            "testuary check --verify" in after) == (status == 0)
    assert ("Run it" in out.split("Checking the project")[0]) is False
    assert (f"Fix the problems above, then check again{where} with: testuary check" in after) == (status == 1)


def test_an_interpreter_set_in_the_project_settings_is_an_explicit_selection(tmp_path):
    import sys

    from testuary.readiness import _testuary_environment_selected
    from testuary.settings import RunSettings

    chosen = RunSettings(base_dir=tmp_path, sections={"project": {"python_executable": sys.executable}})
    unset = RunSettings(base_dir=tmp_path, sections={"project": {}})
    assert not _testuary_environment_selected(sys.executable, tmp_path, chosen)
    assert _testuary_environment_selected(sys.executable, tmp_path, unset)


def _files(root, names):
    for name, text in names.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def test_the_source_list_follows_each_languages_own_rules(tmp_path):
    """Every language's sources are listed by its file extensions, the directories it
    ignores, the files it names as no sources and its test-file rule, with
    no language branch. An empty file holds nothing to test (AP4)."""
    probe = _setup.Probe()
    python = _files(tmp_path / "py", {
        "mod.py": "x = 1\n", "pkg/core.py": "y = 2\n", "pkg/__init__.py": "",
        "empty.py": "", "setup.py": "setup()\n", "tests.py": "t = 1\n", "Test_x.py": "t = 1\n",
        "test_mod.py": "t = 1\n", "conftest.py": "c = 1\n", ".venv/lib/a.py": "a = 1\n",
        "tests/test_b.py": "t = 1\n"})
    assert probe.source_files(python, "python") == ["mod.py", "pkg/core.py"]
    java = _files(tmp_path / "java", {
        "src/main/java/demo/Calc.java": "class Calc {}\n",
        "src/main/java/demo/CalcIT.java": "class CalcIT {}\n",
        "src/main/java/demo/CalcTestCase.java": "class CalcTestCase {}\n",
        "src/main/java/demo/CalcTest.java": "class CalcTest {}\n",
        "src/test/java/demo/CalcTests.java": "class CalcTests {}\n",
        "target/gen/Gen.java": "class Gen {}\n"})
    assert probe.source_files(java, "java") == ["src/main/java/demo/Calc.java"]


def test_a_link_that_leads_nowhere_is_no_source_file(tmp_path):
    """The project part lists sources through this probe, so a dangling link
    must not end it."""
    python = _files(tmp_path / "py", {"mod.py": "x = 1\n"})
    try:
        (python / "gone.py").symlink_to(python / "missing.py")
    except OSError:
        pytest.skip("this system lets the test create no links")
    assert _setup.Probe().source_files(python, "python") == ["mod.py"]


def test_the_terminal_shows_the_model_services_once_and_takes_a_number(project, monkeypatch, capsys):
    """The terminal lists each model service once, with its label and kind,
    and a number picks one."""
    import builtins

    monkeypatch.chdir(project)
    answers = {"Which model service": "6", "The service's API address": "https://api.together.xyz/v1",
               "Keep the API key": "environment", "The model's name": "big-model"}
    shown = []

    def typed(prompt):
        if prompt.startswith("Which model service"):
            shown.extend(line.strip() for line in capsys.readouterr().out.splitlines()
                         if re.fullmatch(r"\s+\d+\. .+", line))
        return next((answer for start, answer in answers.items() if prompt.startswith(start)), "")

    monkeypatch.setattr(builtins, "input", typed)
    _setup.run_setup(_setup.terminal_ask, _Probe(ollama=False, models=("big-model",)), print,
                     project, part="model")
    assert shown == [f"{index}. {provider.label} ({provider.kind})"
                     for index, provider in enumerate(_setup.PROVIDERS, 1)]
    assert "[together_big_model]" in _user("models.ini").read_text(encoding="utf-8")


def _cgroups(root, version, mount_root, member, quota):
    """A /proc and cgroup file system under ``root`` for a process in the
    cgroup ``member`` whose CPU quota is ``quota`` microseconds per 100000
    (None: no quota), under the version's usual mount, whose root is
    ``mount_root`` (a container's cgroup namespace mounts its own cgroup as
    the root)."""
    proc = root / "proc" / "self"
    proc.mkdir(parents=True)
    point = "/sys/fs/cgroup" if version == 2 else "/sys/fs/cgroup/cpu,cpuacct"
    kind = "cgroup2 cgroup2 rw" if version == 2 else "cgroup cgroup rw,cpu,cpuacct"
    (proc / "mountinfo").write_text(
        f"22 1 8:1 / / rw - ext4 /dev/sda1 rw\n30 22 0:26 {mount_root} {point} rw - {kind}\n",
        encoding="utf-8")
    (proc / "cgroup").write_text(
        f"0::{member}\n" if version == 2 else f"5:memory:{member}\n4:cpu,cpuacct:{member}\n",
        encoding="utf-8")
    top = root / point.lstrip("/")
    inside = top.joinpath(*[part for part in member[len(mount_root):].split("/") if part])
    inside.mkdir(parents=True, exist_ok=True)
    for directory, limit in ((top, None), (inside, quota)):
        if version == 2:
            (directory / "cpu.max").write_text(f"{limit or 'max'} 100000\n", encoding="utf-8")
        else:
            (directory / "cpu.cfs_quota_us").write_text(f"{limit or -1}\n", encoding="utf-8")
            (directory / "cpu.cfs_period_us").write_text("100000\n", encoding="utf-8")


class _Machine(_Probe):
    """``_Probe`` with the setup's own CPU count."""

    cpu_count = _setup.Probe.cpu_count


@pytest.mark.parametrize("version, mount_root, member", [
    (2, "/", "/user.slice/job"), (2, "/", "/"), (1, "/", "/job"), (1, "/docker/abc", "/docker/abc")])
def test_the_suggested_workers_follow_the_cpus_a_quota_leaves(project, monkeypatch, version, mount_root,
                                                             member):
    """On a machine of 16 CPUs, a CPU quota of 4 (a container started with
    --cpus 4) leaves 4 to the setup's count under cgroups v1 and v2, and
    the machine part suggests half of them."""
    _cgroups(project / "system", version, mount_root, member, 400000)
    monkeypatch.setattr(_setup.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(_setup, "allowed_cpus", lambda: list(range(16)), raising=False)
    monkeypatch.setattr(_setup, "_SYSTEM_ROOT", project / "system", raising=False)
    assert _setup.Probe().cpu_count() == 4
    _result, ask, _said = _run(project, {}, probe=_Machine(), part="machine")
    question = ask.questions["machine.mutation_workers"]
    assert question.default == "2" and question.reason.startswith("Testuary may use 4 ")


@pytest.mark.parametrize("allowed, quota, expected", [
    (list(range(3)), None, 3), (list(range(3)), 800000, 3), (list(range(8)), 250000, 3), (None, None, 16)])
def test_the_cpu_count_takes_the_affinity_and_the_quota(project, monkeypatch, allowed, quota, expected):
    """The count is the CPUs the affinity allows, the machine's where the
    platform has no affinity, and at most what a quota leaves, rounded up."""
    _cgroups(project / "system", 2, "/", "/job", quota)
    monkeypatch.setattr(_setup.os, "cpu_count", lambda: 16)
    monkeypatch.setattr(_setup, "allowed_cpus", lambda: allowed, raising=False)
    monkeypatch.setattr(_setup, "_SYSTEM_ROOT", project / "system", raising=False)
    assert _setup.Probe().cpu_count() == expected


def test_the_cpu_count_without_cgroups_is_the_affinity(project, monkeypatch):
    """Where no cgroup can be read, as on Windows and macOS, the count is
    the affinity's, else the machine's."""
    monkeypatch.setattr(_setup.os, "cpu_count", lambda: 6)
    monkeypatch.setattr(_setup, "allowed_cpus", lambda: None, raising=False)
    monkeypatch.setattr(_setup, "_SYSTEM_ROOT", project / "nothing", raising=False)
    assert _setup.Probe().cpu_count() == 6
