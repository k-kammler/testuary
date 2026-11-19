"""A language package's declared project settings reach Testuary.

Settled, registered from outside, declares a runtime and a choice with a
default. Testuary accepts both in ``[project]`` and builds the project
context from the adapter's runtime discovery and tool variables. Python's
and Java's settings keep their own sections.
"""
from __future__ import annotations

import sys

import langadapter as lb
import pytest
from langadapter.adapter import ProjectSetting, RuntimeChoice
from langadapter.context import Interpreter

from testuary import settings as settings_module
from testuary import config
from testuary.languages import language_project_context, setup_context
from testuary.settings import resolve_settings


class Settled(lb.LangAdapter):
    language = "settled"
    file_extensions = (".stl",)
    project_settings = (
        ProjectSetting("settled_runtime", "The program that runs Settled's tests.", "executable",
                       check="a program that starts"),
        ProjectSetting("settled_mode", "How Settled runs its tests.", "choice", default="fast",
                       choices=("fast", "slow"), check="one of the choices"),
    )

    @classmethod
    def discover_runtime(cls, root, values, *, container=False, environment=None, resolve_path=None):
        runtime = values.get("settled_runtime") or ""
        return RuntimeChoice(Interpreter(path=runtime) if runtime else None,
                             ("--mode", values["settled_mode"]))

    @classmethod
    def context_environment(cls, values, *, container=False):
        return {"SETTLED_MODE": values["settled_mode"]}


@pytest.fixture
def settled():
    lb.register("settled", [".stl"], Settled)
    try:
        yield
    finally:
        lb.unregister("settled")


def _settings(tmp_path, **project):
    return resolve_settings(config_files=[], base_dir=tmp_path, models={},
                            environment={}, overrides={"project": project} if project else None)


def test_a_languages_declared_settings_build_its_project_context(settled, tmp_path):
    settings = _settings(tmp_path, settled_runtime=sys.executable, settled_mode="slow")
    assert settings.sections["project"]["settled_mode"] == "slow"
    with config.use_run_settings(settings):
        context = language_project_context("settled", str(tmp_path))
    assert context.interpreter.path == sys.executable
    assert context.interpreter_flags == ("--mode", "slow")
    assert context.environment["SETTLED_MODE"] == "slow"


def test_an_unset_setting_takes_its_declared_default(settled, tmp_path):
    with config.use_run_settings(_settings(tmp_path)):
        context = language_project_context("settled", str(tmp_path))
    assert context.interpreter is None
    assert context.interpreter_flags == ("--mode", "fast")


def test_the_setup_context_describes_the_same_runtime(settled, tmp_path):
    settings = _settings(tmp_path, settled_runtime=sys.executable)
    context = setup_context("settled", settings)
    assert context.interpreter.path == sys.executable
    assert context.environment["SETTLED_MODE"] == "fast"


class Described(Settled):
    """Settled with a description of its own and a container supervisor."""

    language = "described"

    @classmethod
    def discover_runtime(cls, root, values, *, container=False, environment=None, resolve_path=None):
        if not values.get("settled_runtime"):
            raise lb.InvalidRequest("settled_runtime", "names no program: select one")
        return RuntimeChoice(Interpreter(path="discovered"), helper_python="/opt/py/bin/python")

    @classmethod
    def describe_runtime(cls, root, values, *, container=False, environment=None, resolve_path=None):
        return RuntimeChoice(Interpreter(path="described"), helper_python="/opt/py/bin/python")


@pytest.fixture
def described():
    lb.register("described", [".dsc"], Described)
    try:
        yield
    finally:
        lb.unregister("described")


def test_setup_describes_and_a_run_discovers_through_the_same_path(described, tmp_path):
    """Every language's context comes from its adapter: a setup from its
    description (describe_runtime), a run from its discovery, and a run's
    refused selection names the setting (AP4)."""
    assert setup_context("described", _settings(tmp_path)).interpreter.path == "described"
    with config.use_run_settings(_settings(tmp_path, settled_runtime="given")):
        assert language_project_context("described", str(tmp_path)).interpreter.path == "discovered"
    # A run's refusal names the setting and the variable that sets it, as
    # Python's and Java's did before they took this path.
    with config.use_run_settings(_settings(tmp_path)):
        with pytest.raises(RuntimeError, match=(
                r"\[project\] settled_runtime: names no program: select one \(set in the "
                r"project's settings, or with TESTUARY_PROJECT_SETTLED_RUNTIME\)")):
            language_project_context("described", str(tmp_path))


@pytest.mark.parametrize("language", ["python", "java"])
def test_python_and_java_runs_take_their_adapters_whole_runtime_choice(language, tmp_path,
                                                                        monkeypatch):
    """Python's and Java's run contexts come from the path every language
    takes: the adapter's discovery chooses the interpreter, its flags and the
    test framework, at the selected project rather than its working copy,
    and its context_environment adds the tools' variables (AP4 item 4)."""
    from langadapter.context import TestFrameworkConfig

    factory = lb.registration(language).factory
    found_at = []

    def discover(cls, root, values, *, container=False, environment=None, resolve_path=None):
        found_at.append(root)
        return RuntimeChoice(Interpreter(path=sys.executable), ("--chosen",),
                             TestFrameworkConfig(source="isolated"))

    monkeypatch.setattr(factory, "discover_runtime", classmethod(discover))
    monkeypatch.setattr(factory, "context_environment",
                        classmethod(lambda cls, values, *, container=False: {"CHOSEN_TOOL": "1"}))
    source = tmp_path / "source"
    source.mkdir()
    with config.use_run_settings(_settings(tmp_path, source_dir=str(source))):
        context = language_project_context(language, str(tmp_path / "copy"))
    assert context.interpreter.path == sys.executable
    assert context.interpreter_flags == ("--chosen",)
    assert context.test_framework.source == "isolated"
    assert context.environment["CHOSEN_TOOL"] == "1"
    assert set(found_at) == {str(source)}


def test_a_runs_provisioning_request_is_fulfilled_once_for_any_language(settled, tmp_path,
                                                                        monkeypatch):
    """A run fulfils the provisioning request a language's discovery makes,
    once while the run settings stay active, for every language (AP4 item 4)."""
    from langadapter.context import ProvisioningRequest

    from testuary import project as project_module

    def discover(cls, root, values, *, container=False, environment=None, resolve_path=None):
        return RuntimeChoice(Interpreter(provisioning=ProvisioningRequest("9.1", None)))

    monkeypatch.setattr(Settled, "discover_runtime", classmethod(discover))
    made = []
    # A path absolute on every system. A rooted path without a drive is not absolute on Windows.
    provided = str(tmp_path / "made" / "run")
    monkeypatch.setattr(project_module, "provision_runtime",
                        lambda language, request, **_: made.append((language, request.version))
                        or provided)
    with config.use_run_settings(_settings(tmp_path)):
        first = language_project_context("settled", str(tmp_path))
        second = language_project_context("settled", str(tmp_path))
    assert first.interpreter.path == second.interpreter.path == provided
    assert made == [("settled", "9.1")]


def test_a_container_takes_the_supervisor_the_runtime_names(described, tmp_path):
    from langadapter import ContainerExecution

    from testuary.settings import RunSettings

    settings = RunSettings(base_dir=tmp_path, sections={
        "main": {"project_exec": "container"}, "project": {"source_dir": "/selected/source"}},
        environment={"TESTUARY_PROJECT_CONTAINER": "box", "TESTUARY_PROJECT_WORK_DIR": "zone"})
    execution = setup_context("described", settings).execution
    assert isinstance(execution, ContainerExecution)
    assert execution.helper_python == "/opt/py/bin/python"


def test_a_stated_runtime_version_reaches_every_languages_context(settled, tmp_path):
    """[project] runtime_version states the runtime's version line. Every
    language's adapter reads it from its context instead of asking the
    runtime (AP4: the starter's Python-only runtime line goes)."""
    settings = _settings(tmp_path, runtime_version="Settled 9.1")
    assert setup_context("settled", settings).runtime_version == "Settled 9.1"
    with config.use_run_settings(settings):
        assert language_project_context("settled", str(tmp_path)).runtime_version == "Settled 9.1"
    with config.use_run_settings(_settings(tmp_path)):
        assert language_project_context("settled", str(tmp_path)).runtime_version is None


def test_the_starter_prompt_names_the_stated_python_as_the_zero_shot_prompt_does(tmp_path):
    """The starter's prompt names the language level the adapter reports, as
    every agent's does: for a stated Python 3.10.14, Python 3.10, which is
    also what the zero-shot prompt passes, so the two prompts are equal."""
    from string import Template

    from testuary import prompts
    from testuary.project import display_path
    from testuary.starters import _import_hint

    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    target = root / "pkg" / "mod.py"
    target.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    settings = _settings(root, runtime_version="Python 3.10.14", python_executable=sys.executable)

    def full(templates):
        user = Template(templates["user_prompt_template"]).substitute(
            file_path=display_path(str(target), str(root)),
            import_hint=_import_hint("python", str(target), str(root)), source="x = 1\n")
        return f"{templates['system_prompt']}\n\n{user}"

    with config.use_run_settings(settings):
        starter = full(prompts.resolve_agent_prompts("test_initial", "python",
                                                     source_dir=str(root)))
        zero_shot = full(prompts.resolve_agent_prompts("test_initial", "python",
                                                       runtime_version="Python 3.10"))
    assert "Write the tests in: Python 3.10\n" in starter
    assert starter == zero_shot


def test_a_declared_host_path_resolves_by_its_rule(tmp_path):
    """The project configuration resolves the host paths each installed
    language declares (ProjectSetting.host_path) under the configuration
    directory: a path always, a command only when it names a directory, a host
    file also for a container."""

    class Pathed(Settled):
        language = "pathed"
        project_settings = (
            ProjectSetting("pathed_tool", "A tool.", "path", host_path="path"),
            ProjectSetting("pathed_launcher", "A launcher.", "executable", host_path="command"),
            ProjectSetting("pathed_list", "A list.", "path", host_path="host file"),
            ProjectSetting("pathed_note", "A note.", "text"),
        )

    lb.register("pathed", [".pth"], Pathed)
    try:
        settings = _settings(tmp_path, pathed_tool="tools/t", pathed_launcher="run",
                             pathed_list="lists/l.txt", pathed_note="tools/n")
        with config.use_run_settings(settings):
            project = config.get_project_config()
        assert project["pathed_tool"] == str((tmp_path / "tools" / "t").absolute())
        assert project["pathed_launcher"] == "run"
        assert project["pathed_list"] == str((tmp_path / "lists" / "l.txt").resolve())
        assert project["pathed_note"] == "tools/n"
        with config.use_run_settings(_settings(tmp_path, pathed_launcher="bin/run")):
            assert config.get_project_config()["pathed_launcher"] == str(
                (tmp_path / "bin" / "run").absolute())
    finally:
        lb.unregister("pathed")


def test_a_setting_two_languages_declare_takes_the_first_ones_rule(tmp_path):
    """A setting name two installed languages declare is one [project]
    setting: the first language by name gives its host-path rule, as it
    gives its kind and default."""

    class First(Settled):
        language = "aaa-first"
        project_settings = (ProjectSetting("shared_tool", "A tool.", "path", host_path="path"),)

    class Second(Settled):
        language = "zzz-second"
        project_settings = (ProjectSetting("shared_tool", "A launcher.", "executable",
                                           host_path="command"),)

    lb.register("zzz-second", [".zzz"], Second)
    lb.register("aaa-first", [".aaa"], First)
    try:
        with config.use_run_settings(_settings(tmp_path, shared_tool="run")):
            assert config.get_project_config()["shared_tool"] == str((tmp_path / "run").absolute())
    finally:
        lb.unregister("aaa-first")
        lb.unregister("zzz-second")


def test_a_setting_no_language_declares_is_still_refused(settled, tmp_path):
    with pytest.raises(Exception, match="settled_speed"):
        _settings(tmp_path, settled_speed="1")


def test_testuarys_own_settings_keep_their_sections():
    """Python's and Java's declared settings are Testuary's own: none is
    added to [project], and pytest_config stays in [main]."""
    from langadapter import names, registration

    def declared(language):
        return {setting.name for setting in
                getattr(registration(language).factory, "project_settings", ()) or ()}

    bundled = settings_module._read_ini_file(settings_module._bundled_data_path("config.ini"))
    defaults = settings_module._defaults()
    # An installed language beyond the built-in ones (C) adds its own.
    others = set().union(*(declared(name) for name in names()
                           if name not in ("java", "python")))
    others -= declared("python") | declared("java")
    assert {key: value for key, value in defaults["project"].items()
            if key not in others} == bundled["project"]
    assert "pytest_config" in defaults["main"] and "pytest_config" not in defaults["project"]


def test_setup_asks_the_declared_settings_and_writes_them(settled, tmp_path):
    from test_setup_wizard import _run

    project = tmp_path / "proj"
    project.mkdir()
    (project / "calc.stl").write_text("double(x) = x * 2\n", encoding="utf-8")
    (project / "test_calc.stl").write_text("test doubles\n", encoding="utf-8")
    result, ask, said = _run(tmp_path, {
        "project.folder": "proj", "settled.settled_runtime": sys.executable,
        "settled.settled_mode": ["medium", "slow"], "model": ""}, part="project")
    assert ask.asked[ask.asked.index("project.language"):ask.asked.index("targets")] == [
        "project.language", "settled.settled_runtime", "settled.settled_mode", "settled.settled_mode"]
    assert "  The value must be one of fast, slow." in said
    written = (project / ".testuary" / "project.ini").read_text(encoding="utf-8")
    assert "language = settled" in written and "source_files = calc.stl" in written
    assert f"settled_runtime = {sys.executable}" in written and "settled_mode = slow" in written


def test_a_languages_own_questions_are_asked_through_the_same_path(tmp_path):
    """A language whose adapter asks its own questions (setup_questions) is
    asked them as Python and Java are: its check, the value it writes, the
    note after an answer and the notes between questions."""
    from langadapter.adapter import SetupNote, SetupQuestion

    from test_setup_wizard import _run

    class Asked(Settled):
        language = "asked"

        @classmethod
        def setup_questions(cls, project_dir, probe, *, consumer="the caller"):
            yield SetupNote(("  a note before",), ("a missing thing",))
            yield SetupQuestion(
                "asked.where", f"Where {consumer} finds it", setting="settled_mode",
                default="fast", check=lambda answer: None if answer in ("fast", "slow") else "no",
                written=lambda answer: answer.upper(),
                after=lambda answer: SetupNote((f"  took {answer}",)))

    lb.register("asked", [".ask"], Asked)
    try:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "calc.ask").write_text("double(x) = x * 2\n", encoding="utf-8")
        result, ask, said = _run(tmp_path, {"project.folder": "proj",
                                            "asked.where": ["medium", "slow"], "model": ""},
                                 part="project")
    finally:
        lb.unregister("asked")
    assert ask.asked[ask.asked.index("project.language"):ask.asked.index("targets")] == [
        "project.language", "asked.where", "asked.where"]
    assert "  a note before" in said and "  no" in said and "  took slow" in said
    assert "a missing thing" in result.missing
    written = (project / ".testuary" / "project.ini").read_text(encoding="utf-8")
    assert "settled_mode = SLOW" in written


def test_check_reports_each_declared_setting_by_its_kind(settled, tmp_path):
    from testuary.readiness import _declared_setting_items

    settings = _settings(tmp_path, settled_runtime=str(tmp_path / "missing"), settled_mode="slow")
    with config.use_run_settings(settings):
        items = {item.key: item for item in _declared_setting_items({"settled"}, settings)}
    runtime = items["native.settled.setting.settled_runtime"]
    assert (runtime.required, runtime.status) == (True, "missing_setup")
    assert runtime.action == "Set [project] settled_runtime to a program that starts."
    # The problem is a sentence of its own after the purpose.
    assert runtime.reason == "The program that runs Settled's tests. No such program."
    assert items["native.settled.setting.settled_mode"].status == "verified"
    with config.use_run_settings(_settings(tmp_path)):
        unset = _declared_setting_items({"settled"}, _settings(tmp_path))
    mode, runtime = sorted(unset, key=lambda item: item.key)
    assert (runtime.required, runtime.status, runtime.tried_by) == (False, "unverified", "run")


def test_an_unset_c_setting_reads_its_purpose_and_not_set(tmp_path):
    """An unset declared setting is its purpose and "Not set.": the purpose
    says what an empty value does, and no language finds c_address_space_limit."""
    from langadapter import registration

    from testuary.readiness import _declared_setting_items

    (limit,) = [setting for setting in registration("c").factory.project_settings
                if setting.name == "c_address_space_limit"]
    settings = _settings(tmp_path)
    with config.use_run_settings(settings):
        items = {item.key: item for item in _declared_setting_items({"c"}, settings)}
    assert items["native.c.setting.c_address_space_limit"].reason == f"{limit.purpose} Not set."


def test_check_reads_a_relative_setting_from_the_project_not_the_configuration(settled, tmp_path):
    """A relative path a language's setting names is read from the project
    directory, where the adapter reads it and the setup asks it, also when the
    configuration lies elsewhere, as a configuration directory above its
    project does."""
    from testuary.readiness import _declared_setting_items

    outer = tmp_path / "outer"
    project = outer / "project"
    (project / "tools").mkdir(parents=True)
    (project / "tools" / "run").write_text("", encoding="utf-8")
    settings = _settings(outer, source_dir=str(project), settled_runtime="tools/run")
    with config.use_run_settings(settings):
        items = {item.key: item for item in _declared_setting_items({"settled"}, settings)}
    runtime = items["native.settled.setting.settled_runtime"]
    assert (runtime.required, runtime.status) == (True, "verified"), runtime.reason


def test_check_leaves_a_containers_setting_path_to_the_run(settled, tmp_path):
    """Inside a container a setting's path is the container's, which this
    computer cannot look up: the check says so and the run tries it."""
    from testuary.readiness import _declared_setting_items

    settings = resolve_settings(
        config_files=[], base_dir=tmp_path, models={}, environment={},
        overrides={"main": {"project_exec": "container"},
                   "project": {"source_dir": "/work/project", "settled_runtime": "tools/run"}})
    with config.use_run_settings(settings):
        items = {item.key: item for item in _declared_setting_items({"settled"}, settings)}
    runtime = items["native.settled.setting.settled_runtime"]
    assert (runtime.status, runtime.tried_by) == ("unverified", "run"), runtime.reason
    assert "inside the container" in runtime.reason
    assert items["native.settled.setting.settled_mode"].status == "verified"


def test_a_languages_requested_runtime_is_provisioned_by_its_own_adapter(settled, tmp_path,
                                                                         monkeypatch):
    """Provisioning asks the requesting language's adapter, not Python's,
    and keeps each language's environments in a directory of their own under
    Testuary's cache, named after the project (AP4 item 5)."""
    from langadapter.context import ProvisioningRequest

    from testuary import project as project_module

    made = []

    def provision(cls, request, *, store, name, environment=None, log=None):
        made.append((store, name, request.version))
        return str(store / name / "run")

    monkeypatch.setattr(Settled, "provision_runtime", classmethod(provision), raising=False)
    monkeypatch.setattr(project_module, "_venv_home", lambda: tmp_path / "store")
    with config.use_run_settings(_settings(tmp_path, name="calc")):
        runtime = project_module.provision_runtime("settled", ProvisioningRequest("9.1", None))
    assert runtime == str(tmp_path / "store" / "settled" / "calc" / "run")
    assert made == [(tmp_path / "store" / "settled", "calc", "9.1")]


def test_the_provision_command_prepares_what_any_languages_settings_request(
        settled, tmp_path, monkeypatch, capsys):
    """testuary provision prepares the runtime each installed language's
    settings request, as its adapter describes them, and prints it (AP4)."""
    from langadapter.context import ProvisioningRequest

    from testuary import __main__ as cli
    from testuary import project as project_module

    def requested(cls, root, values, *, container=False, environment=None, resolve_path=None):
        return RuntimeChoice(Interpreter(provisioning=ProvisioningRequest(
            values["settled_mode"], None)))

    monkeypatch.setattr(Settled, "discover_runtime", classmethod(requested))
    asked = []
    monkeypatch.setattr(project_module, "provision_runtime",
                        lambda language, request, **_: asked.append((language, request.version))
                        or "/made/run")
    (tmp_path / ".testuary").mkdir()
    (tmp_path / ".testuary" / "project.ini").write_text("[project]\nsettled_mode = slow\n",
                                                        encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    assert cli._cmd_provision([]) == 0
    assert asked == [("settled", "slow")]
    assert "settled: /made/run (ready)" in capsys.readouterr().out


def test_the_setup_offers_the_installed_languages_as_langadapter_lists_them():
    """No language comes first by its name (AP4)."""
    from testuary import setup

    lb.register("aardvark", [".aard"], Settled)
    try:
        assert setup.languages() == tuple(lb.names())
    finally:
        lb.unregister("aardvark")
