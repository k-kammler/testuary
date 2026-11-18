"""Inspection describes captured original inputs without execution resolution."""
import os
import json
from pathlib import Path
import subprocess

import pytest

from langadapter import (
    ContainerExecution, Interpreter, LocalExecution, register, unregister,
    Outcome, verify_project,
)
from langadapter.capabilities import OPERATION_NAMES
from langadapter.errors import UnsupportedLanguage
from langadapter.execution import LocalExecutor
from langadapter.java.adapter import JavaAdapter
from testuary import config as config_module
from testuary import project as project_module
from testuary.orchestrator import check
from testuary.orchestrator import RunRequest
from testuary.languages import language_setting_values, setup_context, setup_requirements
from testuary.readiness import RunReadiness, _compose_setup_report, _setup_request
from testuary.settings import ConfigurationError, RunSettings


@pytest.fixture(autouse=True)
def no_execution_resolution(monkeypatch):
    def refused(*args, **kwargs):
        pytest.fail("setup called an execution/provisioning resolver")

    for name in ("project_runtime", "provision_runtime", "project_work_dir", "project_repo_dir"):
        monkeypatch.setattr(project_module, name, refused)


def _settings(root, language, *, project=None, environment=None, guest=False, **kwargs):
    return RunSettings(
        base_dir=root,
        sections={"main": {"project_exec": "container" if guest else "local"},
                  "project": {"source_dir": "/selected/source" if guest else "source",
                              **(project or {})}},
        environment=environment or {}, **kwargs,
    )


@pytest.mark.parametrize("language", ["python", "java"])
def test_original_root_is_not_the_runtime_work_copy(tmp_path, language):
    settings = _settings(tmp_path, language)
    with config_module.use_run_settings(settings):
        config_module.set_invocation_source_dir(str(tmp_path / "private-copy"))
        context = setup_context(language, settings)
        assert config_module.active_settings() is settings
    assert context.root == str(tmp_path / "source")
    assert context.interpreter is None
    assert isinstance(context.execution, LocalExecution)
    assert not context.environment.inherit


@pytest.mark.parametrize("language", ["python", "java"])
def test_no_ambient_runtime_environment_or_later_chdir(tmp_path, monkeypatch, language):
    selected = {"PATH": "captured", "VALUE": "before"}
    settings = _settings(tmp_path, language, environment=selected)
    selected["PATH"] = "mutated"
    monkeypatch.setenv("PATH", "later-path")
    monkeypatch.setenv("JAVA_HOME", "later-jdk")
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", "later-python")
    monkeypatch.chdir(tmp_path.parent)
    context = setup_context(language, settings)
    assert dict(context.environment) == {"PATH": "captured", "VALUE": "before"}
    assert context.interpreter is None
    assert context.root == str(tmp_path / "source")
    assert config_module.active_settings() is None


@pytest.mark.parametrize("runtime", ["env/bin/python", "env/bin/python3"])
def test_python_executable_keeps_the_selected_lexical_address(tmp_path, runtime):
    settings = _settings(tmp_path, "python", project={"python_executable": runtime})
    context = setup_context("python", settings)
    assert context.interpreter == Interpreter(path=str(tmp_path / runtime))
    assert not Path(context.interpreter.path).exists()


def test_python_version_is_an_unexecuted_provisioning_description(tmp_path):
    settings = _settings(tmp_path, "python", project={"python_version": "3.11",
                                                       "requirements": "requirements.txt"})
    context = setup_context("python", settings)
    assert context.interpreter.path is None
    assert context.interpreter.provisioning.version == "3.11"
    assert context.interpreter.provisioning.requirements == str(tmp_path / "requirements.txt")
    assert not (tmp_path / "requirements.txt").exists()


def test_python_explicit_executable_has_existing_precedence_over_version(tmp_path):
    context = setup_context("python", _settings(tmp_path, "python", project={
        "python_version": "3.11", "python_executable": "selected/python"}))
    assert context.interpreter.path == str(tmp_path / "selected" / "python")
    assert context.interpreter.provisioning is None


@pytest.mark.parametrize("config,source,path", [
    ("auto", "project", None), ("none", "isolated", None),
    ("pytest.ini", "project", "pytest.ini"),
])
@pytest.mark.parametrize("strategy,flags", [("skip", ()), ("unfreeze", ("-X", "frozen_modules=off"))])
def test_python_framework_and_flags_are_descriptions(tmp_path, config, source, path, strategy, flags):
    settings = RunSettings(base_dir=tmp_path, environment={}, sections={
        "main": {"pytest_config": config}, "project": {"frozen_module_strategy": strategy}})
    context = setup_context("python", settings)
    assert context.test_framework.source == source
    assert context.test_framework.path == path
    assert context.interpreter_flags == flags


@pytest.mark.parametrize("selected", ["jdk", "jdk/bin/java", "jdk/bin/java.exe"])
def test_java_selection_does_not_check_or_bootstrap_jdk(tmp_path, selected):
    settings = _settings(tmp_path, "java", project={
        "java_home": selected, "build_java_home": "build-jdk", "maven": "mvn", "gradle": "tools/gradle"})
    context = setup_context("java", settings)
    address = tmp_path / selected
    expected = address if address.name in {"java", "java.exe"} else address / "bin" / (
        "java.exe" if os.name == "nt" else "java")
    assert context.interpreter.path == str(expected)
    assert dict(context.environment) == {
        "LANGADAPTER_BUILD_JAVA_HOME": str(tmp_path / "build-jdk"),
        "LANGADAPTER_MAVEN": "mvn", "LANGADAPTER_GRADLE": str(tmp_path / "tools" / "gradle"),
    }


@pytest.mark.parametrize("language", ["python", "java"])
def test_guest_target_and_host_control_are_separate(tmp_path, monkeypatch, language):
    captured = {
        "PATH": "host-path", "JAVA_HOME": "C:/host-jdk", "PYTHONPATH": "C:/host/imports",
        "TESTUARY_PROJECT_CONTAINER": "borrowed", "TESTUARY_PROJECT_CONTAINER_RUNNER": "podman",
        "TESTUARY_PROJECT_WORK_DIR": "zone", "LANGADAPTER_HELPER_PYTHON": "/control/python",
    }
    settings = _settings(tmp_path, language, guest=True, environment=captured, project={
        "python_executable": "/target/bin/python", "java_home": "/target/jdk",
        "maven": "/tools/mvn", "build_java_home": "/build/jdk"})
    monkeypatch.setenv("LANGADAPTER_HELPER_PYTHON", "/later/helper")
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER", "later-container")
    context = setup_context(language, settings)
    assert context.root == "/selected/source"
    assert isinstance(context.execution, ContainerExecution)
    assert context.execution.container == "borrowed"
    assert context.execution.runner == "podman"
    assert context.execution.helper_python == "/control/python"
    assert context.execution.work_dir == str(tmp_path / "zone")
    assert dict(context.execution.control_environment) == captured
    assert not context.execution.control_environment.inherit
    assert not context.environment.inherit
    assert dict(context.environment) == ({} if language == "python" else {
        "LANGADAPTER_MAVEN": "/tools/mvn", "LANGADAPTER_BUILD_JAVA_HOME": "/build/jdk"})
    assert context.interpreter.path == ("/target/bin/python" if language == "python" else "/target/jdk/bin/java")


@pytest.mark.parametrize("host_home", [None, "/host/jdk", "C:/host/jdk"])
@pytest.mark.parametrize("guest_home", [None, "/target/jdk"])
def test_guest_java_runtime_selection_never_uses_host_control_home(tmp_path, host_home, guest_home):
    captured = {"TESTUARY_PROJECT_CONTAINER": "borrowed", "TESTUARY_PROJECT_WORK_DIR": "zone"}
    if host_home is not None:
        captured["JAVA_HOME"] = host_home
    project = {"java_home": guest_home} if guest_home is not None else {}
    settings = _settings(tmp_path, "java", guest=True, project=project, environment=captured)
    context = setup_context("java", settings)
    request = _setup_request(RunRequest(source_files=("Subject.java",)), settings,
                             context=context, required_operations=setup_requirements(coverage=False, mutation=True).operations)
    assert request.context is context
    assert request.context.interpreter == (
        None if guest_home is None else Interpreter(path="/target/jdk/bin/java"))
    assert dict(context.environment) == {}
    assert dict(context.execution.control_environment) == captured
    assert not request.allow_acquisition


@pytest.mark.parametrize("explicit", [False, True])
def test_local_java_keeps_captured_home_fallback_and_explicit_precedence(tmp_path, explicit):
    captured_home = str(tmp_path / "host-jdk")
    settings = _settings(tmp_path, "java", environment={"JAVA_HOME": captured_home},
                         project={"java_home": "selected-jdk"} if explicit else {})
    context = setup_context("java", settings)
    selected = tmp_path / ("selected-jdk" if explicit else "host-jdk")
    request = _setup_request(RunRequest(source_files=("Subject.java",)), settings,
                             context=context, required_operations=setup_requirements(coverage=False, mutation=False).operations)
    assert request.context.interpreter == Interpreter(path=str(
        selected / "bin" / ("java.exe" if os.name == "nt" else "java")))
    assert dict(context.environment) == {"JAVA_HOME": captured_home}


@pytest.mark.parametrize("guest", [False, True])
def test_java_setup_reuses_one_captured_project_without_catalogue_output(tmp_path, monkeypatch, capsys,
                                                                       guest):
    section = "subject-private-secret"
    captured = {"AUTH_TOKEN": "private-secret", "TESTUARY_PROJECT_CONTAINER": "borrowed",
                "TESTUARY_PROJECT_WORK_DIR": "zone"}
    settings = _settings(tmp_path, "java", guest=guest, environment=captured,
        project={"name": section, "maven": "/tools/mvn" if guest else "mvn"})
    resolve = config_module.get_project_config
    calls = []

    def project(**options):
        calls.append(options)
        return resolve(**options)

    monkeypatch.setattr(config_module, "get_project_config", project)
    context = setup_context("java", settings)
    assert len(calls) == 1
    request = _setup_request(RunRequest(source_files=("Subject.java",)), settings,
                             context=context, required_operations=setup_requirements(coverage=False, mutation=False).operations)
    assert request.context is context and request.context.interpreter is None
    assert dict(context.environment)["LANGADAPTER_MAVEN"] == ("/tools/mvn" if guest else "mvn")
    assert len(calls) == 2
    assert capsys.readouterr() == ("", "")
    with config_module.use_run_settings(settings):
        values = language_setting_values("java", settings=settings)
        assert JavaAdapter.context_environment(values, container=guest) == {
            "LANGADAPTER_MAVEN": "/tools/mvn" if guest else "mvn"}
    assert capsys.readouterr() == ("", "")


@pytest.mark.parametrize("guest", [False, True])
def test_actual_java_check_keeps_only_one_redacted_catalogue_fact(tmp_path, monkeypatch, capsys, guest):
    import langadapter

    section = "subject-private-secret"
    settings = RunSettings(base_dir=tmp_path, sections={
        "main": {"project_exec": "container" if guest else "local"},
        "project": {"name": section, "source_dir": "/selected/source" if guest else "source"},
        "api": {"model": "selected"}}, models={"selected": {
            "provider": "openai", "model": "controlled-model", "api_key": "private-secret",
            "api_url": "https://example.test/v1"}}, environment={
            "AUTH_TOKEN": "private-secret", "TESTUARY_PROJECT_CONTAINER": "borrowed",
            "TESTUARY_PROJECT_WORK_DIR": "zone"})
    selected = []

    def assess(language, request):
        selected.append((language, request))
        return langadapter.ProjectSetupReport("inspection", Outcome.UNSUPPORTED,
            "controlled native boundary only", request.context, request.inputs, (), ())

    monkeypatch.setattr(langadapter, "inspect_project", assess)
    report = check(RunRequest(source_files=("Subject.java",)), settings=settings)
    assert capsys.readouterr() == ("", "")
    assert len(selected) == 1 and selected[0][0] == "java"
    assert "private-secret" not in json.dumps(report.as_dict())
    assert selected[0][1].context.interpreter is None
    assert not selected[0][1].allow_acquisition


@pytest.mark.parametrize("value", ["jdk", "C:/jdk", "/jdk/../other", "/jdk\\bin"])
def test_guest_java_invalid_selection_is_not_host_repaired(tmp_path, value):
    settings = _settings(tmp_path, "java", guest=True, project={"java_home": value}, environment={
        "TESTUARY_PROJECT_CONTAINER": "selected", "TESTUARY_PROJECT_WORK_DIR": "zone"})
    with pytest.raises(ConfigurationError, match="absolute POSIX"):
        setup_context("java", settings)


@pytest.mark.parametrize("language", ["python", "java"])
def test_root_parent_alias_is_preserved_for_nonfollowing_native_admission(tmp_path, language):
    settings = _settings(tmp_path, language, project={"source_dir": "parent/../source"})
    context = setup_context(language, settings)
    assert context.root == str(tmp_path / "parent" / ".." / "source")


def test_actual_readiness_request_keeps_selection_and_disabled_acquisition(tmp_path):
    settings = _settings(tmp_path, "python", project={
        "python_version": "3.11", "setup_timeout_seconds": "7.5", "setup_probe_timeout_seconds": "2"})
    context = setup_context("python", settings)
    operations = setup_requirements(coverage=False, mutation=True).operations
    request = _setup_request(RunRequest(source_files=("first.py", "second.py"), targets="2"), settings,
                             context=context, required_operations=operations)
    assert request.context is context
    assert request.source_files == ("second.py",)
    assert request.required_operations == operations
    assert request.total_timeout_s == 7.5 and request.probe_timeout_s == 2
    assert request.allow_acquisition is False


@pytest.mark.parametrize("missing", [False, True])
def test_actual_tool_verification_leaves_only_run_only_items_open(tmp_path, monkeypatch,
                                                                 collection_answer, missing):
    """Under --verify, imported tools verify, and what only a run tries stays
    unverified and listed without failing. A missing tool still fails."""
    source = tmp_path / "source"
    source.mkdir()
    (source / "subject.py").write_bytes(b"value = 1\n")
    runtime = tmp_path / "runtime"
    runtime.write_bytes(b"controlled interpreter placeholder\n")
    settings = _settings(tmp_path, "python", project={"python_executable": str(runtime)},
                         environment={"TESTUARY_PROJECT_WORK_DIR": str(tmp_path)})
    with config_module.use_run_settings(settings):
        invocation = RunRequest(source_files=("subject.py",))
        context = setup_context("python", settings)
        request = _setup_request(invocation, settings, context=context,
                                 required_operations=setup_requirements(coverage=False, mutation=True).operations)
        calls = []

        def run(executor, argv, **options):
            # LangAdapter's executor holds its context only as a private member, so the stand-in reads it there.
            calls.append((executor._context, argv, options))
            if "--collect-only" in argv:
                assert not missing and executor._context.root != context.root
                return collection_answer(executor, argv, options)
            if argv[1:3] == ["-I", "-S"]:
                value = {"implementation": "CPython", "version": "3.10.19", "tools": {
                    "pytest": None if missing else "8.1.1", "coverage": "7.6.8", "cosmic-ray": "8.4.1"}}
            else:
                assert executor._context.root != context.root and options["cwd"] == executor._context.root
                assert "env" not in options and executor._context.environment is context.environment
                value = {"status": "imported"}
            return subprocess.CompletedProcess(argv, 0, json.dumps(value), "")

        monkeypatch.setattr(LocalExecutor, "run", run)
        native = verify_project("python", request)
        configuration = RunReadiness(
            scope="inspection", status="unverified", exit_code=0, source_dir=context.root,
            execution="local", source_files=invocation.source_files, phases=(), model_alias=None,
            provider=None, model=None, credentials_configured=None, output_dir=None,
            output_mode="fresh", limits=(), items=())
        report = _compose_setup_report(configuration, native, request=request, settings=settings)
        assert report.scope == "verification" and report.exit_code == (1 if missing else 0)
        assert report.status == ("missing_setup" if missing else "verified")
        assert native.outcome is (Outcome.MISSING_SETUP if missing else Outcome.COMPLETED)
        assert report.source_dir == context.root
        items = {item.key: item for item in report.items}
        assert items["native.item.native_execution"].status == "unverified"
        assert items["native.item.native_execution"].tried_by == "run"
        assert items["native.operation.tests.run_suite"].status == ("missing_setup" if missing else "unverified")
        assert items["native.operation.tests.run_suite"].tried_by == (None if missing else "run")
        assert all(not Path(selected.root).exists() for selected, _, _ in calls[1:])
        assert (source / "subject.py").read_bytes() == b"value = 1\n"
        assert not request.allow_acquisition


def test_other_active_settings_are_not_silently_replaced(tmp_path):
    settings = _settings(tmp_path, "python")
    with config_module.use_run_settings(settings):
        with pytest.raises(ConfigurationError, match="conflicts"):
            setup_context("python", _settings(tmp_path, "python"))
        assert config_module.active_settings() is settings


@pytest.fixture
def registered_language():
    def no_construction(context):
        pytest.fail("context description instantiated a native backend")

    register("setup-fixture", (".setup-fixture",), no_construction)
    try:
        yield "setup-fixture"
    finally:
        unregister("setup-fixture")


def test_an_unknown_language_is_refused_and_a_registered_one_needs_no_policy(
        tmp_path, registered_language):
    with pytest.raises(UnsupportedLanguage):
        setup_context("not-registered", _settings(tmp_path, "python"))
    # Any registered language gets a context from its facts: its runtime fact
    # names the program that runs its tests (none here).
    context = setup_context(registered_language, _settings(tmp_path, registered_language))
    assert context.root == str(tmp_path / "source") and context.interpreter is None


@pytest.mark.parametrize("mutation", [False, True])
def test_operation_projection_has_only_the_profiles_required_families(mutation):
    """Running the tests with their test units, and mutation when measured.
    Every optional family (delivered runs, timing, merging, load repair) is
    left to its fallback."""
    operations = setup_requirements(coverage=False, mutation=mutation).operations
    assert len(operations) == len(set(operations))
    assert set(operations) <= set(OPERATION_NAMES)
    assert ("mutations.open_session" in operations) is mutation
    assert {"tests.run_suite", "sources.prune_tests", "sources.test_metrics"} <= set(operations)
    for optional in ("tests.run_as_delivered", "tests.benchmark", "sources.merge_sources",
                     "sources.prune_collection_breakers", "tests.per_test_evidence"):
        assert optional not in operations


def test_invalid_scope_flag_is_not_truthy_admission():
    with pytest.raises(TypeError, match="booleans"):
        setup_requirements(coverage=False, mutation="false").operations


def test_selected_groups_keep_target_order_and_one_context_per_language(tmp_path):
    from testuary.languages import selected_groups

    settings = _settings(tmp_path, "python", project={"python_executable": "env/bin/python",
                                                      "java_home": "jdk"})
    sources = ("pkg/a.py", "src/main/java/demo/B.java", "pkg/c.py")
    groups = selected_groups(sources, settings)
    assert [(group.language, group.sources) for group in groups] == [
        ("python", ("pkg/a.py", "pkg/c.py")), ("java", ("src/main/java/demo/B.java",))]
    for group in groups:
        assert group.context == setup_context(group.language, settings)
        assert not group.context.environment.inherit


def test_python_only_selection_creates_no_java_group(tmp_path):
    from testuary.languages import selected_groups

    settings = _settings(tmp_path, "python")
    assert [group.language for group in selected_groups(("pkg/a.py",), settings)] == ["python"]


def test_unclaimed_target_refuses_before_any_context(tmp_path):
    from testuary.languages import selected_groups

    with pytest.raises(ConfigurationError, match="no installed language"):
        selected_groups(("pkg/a.py", "notes.unknown-suffix"), _settings(tmp_path, "python"))
    with pytest.raises(TypeError, match="sequence"):
        selected_groups("pkg/a.py", _settings(tmp_path, "python"))


@pytest.mark.parametrize("mutation", [False, True])
@pytest.mark.parametrize("coverage", [False, True])
def test_one_requirement_projection_serves_setup_and_capability_preflight(tmp_path, coverage, mutation):
    from langadapter import ProjectSetupRequest
    from testuary.languages import setup_requirements

    requirements = setup_requirements(coverage=coverage, mutation=mutation)
    assert requirements.operations == setup_requirements(coverage=False, mutation=mutation).operations
    assert (requirements.facets == (("tests.run_suite", ("coverage",)),)) is coverage
    assert ("coverage" in requirements.capabilities) is coverage
    assert ("mutation" in requirements.capabilities) is mutation
    request = ProjectSetupRequest(setup_context("python", _settings(tmp_path, "python")), ("a.py",),
                                  requirements.operations, required_facets=requirements.facets)
    assert request.facets("tests.run_suite") == (("coverage",) if coverage else ())


def test_capability_preflight_requires_only_the_projected_measurements(default_run):
    from langadapter.capabilities import (
        CAPABILITIES, Availability, Capability, CapabilityReport,
    )
    from testuary.languages import setup_requirements
    from testuary.languages import assess

    report = CapabilityReport("python", "python", Outcome.COMPLETED, "", "3.12", tuple(
        Capability(name, Availability.MISSING_SETUP, "coverage tool missing") if name == "coverage"
        else Capability(name, Availability.AVAILABLE) for name in CAPABILITIES), (), "arcs", ())
    assert assess(report, setup_requirements(coverage=False, mutation=True)).refusals == ()
    refused = assess(report, setup_requirements(coverage=True, mutation=True)).refusals
    assert refused == ("The language adapter lacks coverage, which Testuary needs: "
                       "coverage tool missing",)
    assert assess(report).refusals == refused
