"""Project contexts and source operations cross the public backend boundary."""
import hashlib
import os
import sys
from pathlib import Path
from subprocess import CompletedProcess

import pytest

from langadapter.context import ContainerExecution, Environment, LocalExecution
from langadapter.python.adapter import PythonAdapter as ProjectPythonBackend
from testuary import config as config_module
from testuary import project as project_module
from testuary.languages import language_project_context
from testuary.models import resolve_provider_credential
from testuary.languages.generic import GenericBackend
from testuary.settings import RunSettings, resolve_settings
from per_suite_double import killing_tests


@pytest.fixture
def configured(monkeypatch):
    monkeypatch.setattr(config_module, "project_exec_mode", lambda: "local")
    monkeypatch.setattr(project_module, "project_work_dir", lambda: None)
    monkeypatch.setattr(config_module, "load_config", lambda section: {"pytest_config": "none"})


def test_context_describes_local_project(configured, tmp_path, in_run):
    in_run(overrides={"main": {"pytest_config": "none"}})
    context = language_project_context("python", str(tmp_path))
    context.validate()
    assert context.root == str(tmp_path.resolve())
    assert isinstance(context.execution, LocalExecution)
    assert context.interpreter.path == sys.executable
    assert context.interpreter_flags == ("-X", "frozen_modules=off")
    assert context.test_framework.source == "isolated"


def test_delivery_crosses_the_public_operation(configured, tmp_path, monkeypatch, default_run):
    from langadapter.outcomes import ExecutionIdentity, ObservationIdentity, ToolchainIdentity
    from langadapter.outcomes import Outcome
    from langadapter.runs import RunCounts, SideEffects, SuiteRun, TestOutcome
    from langadapter.suites import TestId

    test = tmp_path / "test_a.py"
    test.write_text("def test_a():\n    pass\n", encoding="utf-8")
    calls = []

    def delivered(self, request, *, sources=None):
        calls.append((request, sources))
        return SuiteRun(
            outcome=Outcome.TEST_FAILURES, reason="red", counts=RunCounts(failed=1),
            tests=(TestOutcome(TestId("test_a.py", "TestX.test_a", "1"), "failed"),),
            unmatched=(), wall_time_s=1, output="failed", coverage=None,
            identity=ObservationIdentity(None, (), ToolchainIdentity("CPython", "3.12"),
                                         None, ExecutionIdentity("local")),
            side_effects=SideEffects(session_seed=3, reverse_order=True))

    monkeypatch.setattr(ProjectPythonBackend, "run_as_delivered", delivered)
    replacement = "def test_a():\n    assert False\n"
    run = GenericBackend("python", (".py",)).run_as_delivered(
        [str(test)], str(tmp_path), timeout=60, seed=3, reverse=True,
        sources={str(test): replacement})
    assert run.failing == ("test_a.TestX.test_a",) and run.reason is None
    assert run.cases[0].name == "test_a[1]"
    request, sources = calls[0]
    assert request.session_seed == 3 and request.reverse_order
    assert request.remove_created_files and request.isolation_config
    assert request.module_under_test is None
    assert sources == {"test_a.py": replacement}


def test_context_describes_container_paths(configured, monkeypatch, tmp_path, in_run):
    monkeypatch.setattr(config_module, "project_exec_mode", lambda: "container")
    monkeypatch.setattr(project_module, "project_work_dir", lambda: str(tmp_path))
    monkeypatch.setattr(project_module, "project_repo_dir", lambda: "/testbed")
    monkeypatch.setattr(config_module, "load_config", lambda section: {
        "pytest_config": "none", "python_executable": "/opt/env/bin/python"})
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER", "project-container")
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER_RUNNER", "podman")
    monkeypatch.setenv("TESTUARY_PROJECT_WORK_DIR", str(tmp_path))
    in_run(overrides={"main": {"pytest_config": "none"},
                      "project": {"python_executable": "/opt/env/bin/python"}})
    context = language_project_context("python")
    context.validate()
    assert context.root == "/testbed"
    assert isinstance(context.execution, ContainerExecution)
    assert context.execution.container == "project-container"
    assert context.execution.runner == "podman"
    assert context.execution.work_dir == str(tmp_path)
    assert not any(k.startswith("TESTUARY_") for k in context.environment)


def test_container_context_uses_only_the_resolved_execution_selection(monkeypatch, tmp_path):
    from langadapter.execution import ContainerExecutor

    settings = RunSettings(
        sections={"main": {"project_exec": "container"},
                  "project": {"python_executable": "/guest/bin/python"}},
        environment={"TESTUARY_PROJECT_CONTAINER": "captured-container",
                     "TESTUARY_PROJECT_CONTAINER_RUNNER": "podman",
                     "TESTUARY_PROJECT_WORK_DIR": str(tmp_path),
                     "TESTUARY_PROJECT_SOURCE_DIR": "/captured-project",
                     "LANGADAPTER_HELPER_PYTHON": "/captured/control-python"},
        base_dir=tmp_path)
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER", "later-container")
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER_RUNNER", "docker")
    monkeypatch.setenv("TESTUARY_PROJECT_SOURCE_DIR", "/later-project")
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", "/later/python")
    monkeypatch.setenv("LANGADAPTER_HELPER_PYTHON", "/later/control-python")

    with config_module.use_run_settings(settings):
        backend = GenericBackend("python", (".py",)).project_backend()

    context = backend.context
    assert context.root == "/captured-project"
    assert context.interpreter.path == "/guest/bin/python"
    assert context.execution.container == "captured-container"
    assert context.execution.runner == "podman"
    assert context.execution.work_dir == str(tmp_path)
    assert context.execution.helper_python == "/captured/control-python"
    assert isinstance(backend.executor, ContainerExecutor)
    assert backend.executor.container == "captured-container"
    assert backend.executor.runner == "podman"
    assert backend.executor.helper_python == "/captured/control-python"


@pytest.mark.parametrize("language", ["python", "java"])
def test_a_containers_own_python_supervises_without_a_selected_helper(monkeypatch, tmp_path,
                                                                     language):
    """Without pidfd in a container, the project's interpreter is the helper
    unless LANGADAPTER_HELPER_PYTHON selects another. Java selects one."""
    from testuary.languages import get_backend
    from testuary.languages import setup_context

    settings = RunSettings(
        sections={"main": {"project_exec": "container"},
                  "project": {"source_dir": "/captured-project",
                              "python_executable": "/guest/bin/python",
                              "java_home": "/guest/jdk"}},
        models=_MODELS,
        environment={"TESTUARY_PROJECT_CONTAINER": "captured-container",
                     "TESTUARY_PROJECT_WORK_DIR": str(tmp_path),
                     "TESTUARY_PROJECT_SOURCE_DIR": "/captured-project"},
        base_dir=tmp_path)
    monkeypatch.setenv("LANGADAPTER_HELPER_PYTHON", "/later/control-python")
    with config_module.use_run_settings(settings):
        inspected = setup_context(language, settings)
        executed = get_backend(language).project_backend()
    expected = "/guest/bin/python" if language == "python" else None
    assert inspected.execution.helper_python == expected
    assert executed.context.execution.helper_python == expected
    if language == "python":
        assert executed.executor.helper_python == expected


def test_missing_resolved_container_cannot_be_filled_from_ambient_environment(monkeypatch, tmp_path):
    settings = RunSettings(
        sections={"main": {"project_exec": "container"},
                  "project": {"python_executable": "/guest/bin/python"}},
        environment={"TESTUARY_PROJECT_WORK_DIR": str(tmp_path)}, base_dir=tmp_path)
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER", "unresolved-container")

    with config_module.use_run_settings(settings), pytest.raises(ValueError, match="requires a container"):
        language_project_context("python", "/captured-project")


@pytest.mark.parametrize("language", ["python", "java"])
@pytest.mark.parametrize("relative", [None, "project"])
def test_local_context_root_is_anchored_in_the_resolved_base(
        monkeypatch, tmp_path, language, relative):
    base, later = tmp_path / "base", tmp_path / "later"
    (base / "project").mkdir(parents=True)
    later.mkdir()
    jdk = base / "jdk"
    (jdk / "bin").mkdir(parents=True)
    java = jdk / "bin" / ("java.exe" if os.name == "nt" else "java")
    java.touch()
    settings = RunSettings(
        sections={"project": {"python_executable": sys.executable,
                              "java_home": str(jdk)}}, environment={}, base_dir=base)
    monkeypatch.chdir(later)
    with config_module.use_run_settings(settings):
        context = language_project_context(language, relative)

    assert context.root == str(base / "project" if relative else base)
    assert context.interpreter.path == (sys.executable if language == "python" else str(java))


def test_java_context_build_tools_are_selected_from_resolved_project_settings(monkeypatch, tmp_path):

    (tmp_path / "jdk" / "bin").mkdir(parents=True)
    java = tmp_path / "jdk" / "bin" / ("java.exe" if os.name == "nt" else "java")
    java.touch()
    settings = RunSettings(
        sections={"project": {"java_home": "jdk", "build_java_home": "build-jdk",
                              "maven": "tools/maven", "gradle": "gradle"}},
        environment={}, base_dir=tmp_path)
    monkeypatch.setenv("LANGADAPTER_BUILD_JAVA_HOME", "later-build-jdk")
    monkeypatch.setenv("TESTUARY_PROJECT_BUILD_JAVA_HOME", "later-consumer-jdk")
    monkeypatch.setenv("TESTUARY_PROJECT_MAVEN", "later-maven")
    monkeypatch.setenv("JAVA_HOME", "later-runtime")

    with config_module.use_run_settings(settings):
        context = GenericBackend("java", (".java",)).project_backend(str(tmp_path)).context

    assert context.interpreter.path == str(java)
    # Only the selected tools: absent PATH/JAVA_HOME do not revive host values.
    assert not context.environment.inherit
    assert dict(context.environment) == {
        "LANGADAPTER_BUILD_JAVA_HOME": str(tmp_path / "build-jdk"),
        "LANGADAPTER_MAVEN": str(tmp_path / "tools" / "maven"),
        "LANGADAPTER_GRADLE": "gradle",
    }


# Synthetic credentials: the default provider variable and an alias's api_key_env.
_CREDENTIALS = {"TESTUARY_OLLAMA_API_KEY": "synthetic-default-key",
                "AUDIT_ALIAS_KEY": "synthetic-alias-key"}
_MODELS = {"ollama_audit": {"provider": "ollama", "api_key_env": "AUDIT_ALIAS_KEY"}}


def _captured_settings(tmp_path, *, guest=False, captured=True):
    jdk = tmp_path / "jdk"
    (jdk / "bin").mkdir(parents=True, exist_ok=True)
    (jdk / "bin" / ("java.exe" if os.name == "nt" else "java")).touch()
    environment = {"AUDIT_EXECUTION_MARKER": "captured-before", "PATH": "captured-path",
                   "TESTUARY_PROJECT_WORK_DIR": str(tmp_path / "zone"), **_CREDENTIALS}
    project = {"source_dir": str(tmp_path), "python_executable": sys.executable,
               "java_home": str(jdk)}
    if guest:
        environment |= {"TESTUARY_PROJECT_CONTAINER": "captured-container",
                        "TESTUARY_PROJECT_SOURCE_DIR": "/captured/project",
                        "LANGADAPTER_HELPER_PYTHON": "/captured/control/python"}
        project |= {"source_dir": "/captured/project", "python_executable": "/guest/bin/python",
                    "java_home": "/guest/jdk"}
    return RunSettings(sections={"main": {"project_exec": "container" if guest else "local"},
                                 "project": project}, models=_MODELS,
                       environment=environment if captured else None, base_dir=tmp_path)


@pytest.mark.parametrize("language", ["python", "java"])
@pytest.mark.parametrize("guest", [False, True])
def test_execution_context_uses_the_setup_captured_route(monkeypatch, tmp_path, language, guest):
    """Check and execution share one captured route, work directory and environment."""
    from testuary.languages import get_backend
    from testuary.languages import setup_context

    settings = _captured_settings(tmp_path, guest=guest)
    (tmp_path / "zone").mkdir()
    for name, value in {"AUDIT_EXECUTION_MARKER": "ambient-after", "PATH": "later-path",
                        "TESTUARY_PROJECT_WORK_DIR": str(tmp_path / "later-zone"),
                        "TESTUARY_PROJECT_CONTAINER": "later-container",
                        "LANGADAPTER_HELPER_PYTHON": "/later/control"}.items():
        monkeypatch.setenv(name, value)

    with config_module.use_run_settings(settings):
        inspected = setup_context(language, settings)
        executed = get_backend(language).project_backend().context

    assert dict(executed.environment) == dict(inspected.environment)
    assert executed.execution == inspected.execution
    assert executed.execution.work_dir == str((tmp_path / "zone").resolve())
    ordinary = {name: value for name, value in settings.environment.items()
                if name not in _CREDENTIALS}
    if guest:
        # Guest launches start empty, and the host runner gets no model credentials.
        control = executed.execution.control_environment
        assert control == Environment(ordinary, inherit=False)
        assert executed.execution.helper_python == "/captured/control/python"
        assert "AUDIT_EXECUTION_MARKER" not in executed.environment
    else:
        assert not executed.environment.inherit
        assert executed.environment == inspected.environment
        assert executed.environment["AUDIT_EXECUTION_MARKER"] == "captured-before"
        assert executed.environment["PATH"] == "captured-path"
    assert not set(_CREDENTIALS) & set(executed.environment)
    # Model requests still read the captured credentials.
    assert settings.environment["AUDIT_ALIAS_KEY"] == "synthetic-alias-key"


@pytest.mark.parametrize("guest", [False, True])
def test_uncaptured_settings_refuse_native_contexts(tmp_path, guest):
    from testuary.languages import setup_context
    from testuary.settings import ConfigurationError

    settings = _captured_settings(tmp_path, guest=guest, captured=False)
    assert not settings.environment_captured
    message = r"environment=dict\(os.environ\).*resolve_settings\(\)"
    for language in ("python", "java"):
        with pytest.raises(ConfigurationError, match=message):
            setup_context(language, settings)
    with config_module.use_run_settings(settings):
        for language in ("python", "java"):
            with pytest.raises(ConfigurationError, match=message):
                language_project_context(language)


def test_actual_python_execution_measures_the_captured_environment(tmp_path, monkeypatch):
    """The actual adapter's public suite run sees the captured environment without credentials."""
    from langadapter.runs import SuiteRunRequest
    from langadapter.suites import Suite
    from testuary.languages import get_backend

    project = tmp_path / "project"
    project.mkdir()
    (project / "test_snapshot.py").write_text(
        "import os\n\n\ndef test_captured_environment():\n"
        "    assert os.environ.get('AUDIT_EXECUTION_MARKER') == 'captured-before'\n"
        f"    assert not {set(_CREDENTIALS)!r} & os.environ.keys()\n"
        f"    assert not {set(_CREDENTIALS.values())!r} & set(os.environ.values())\n",
        encoding="utf-8")
    environment = dict(os.environ, AUDIT_EXECUTION_MARKER="captured-before", **_CREDENTIALS)
    settings = RunSettings(sections={"project": {"source_dir": str(project),
                                                 "python_executable": sys.executable}},
                           models=_MODELS, environment=environment, base_dir=tmp_path)
    monkeypatch.setenv("AUDIT_EXECUTION_MARKER", "ambient-after")
    for name, value in _CREDENTIALS.items():
        monkeypatch.setenv(name, value)

    with config_module.use_run_settings(settings):
        backend = get_backend("python").project_backend(str(project))
        result = backend.tests.run_suite(SuiteRunRequest(
            suite=Suite(root=str(project), test_files=("test_snapshot.py",)), timeout_s=60))
        # The model request path still resolves the alias's credential.
        credential = resolve_provider_credential(
            "ollama", {}, _MODELS["ollama_audit"], config_module.configured_environment())

    assert result.counts.passed == 1 and result.counts.failed == 0, result.output
    assert credential == "synthetic-alias-key"


def test_public_instances_belong_to_contexts(configured, tmp_path, monkeypatch, default_run):
    backend = GenericBackend("python", (".py",))
    first = backend.project_backend(str(tmp_path))
    assert backend.project_backend(str(tmp_path)) is first
    other = tmp_path / "other"
    other.mkdir()
    assert backend.project_backend(str(other)) is not first
    monkeypatch.setattr(config_module, "load_config", lambda section: {
        "pytest_config": "none", "frozen_module_strategy": "skip"})
    assert backend.project_backend(str(tmp_path)) is not first


def test_source_validation_preserves_identity_and_status(configured, tmp_path, default_run):
    backend = GenericBackend("python", (".py",))
    source = "def test_a():\n    assert True\n"
    result = backend.validate_test_source(source, str(tmp_path))
    assert result.ok and result.status == "valid"
    assert result.source_sha256 == hashlib.sha256(source.encode()).hexdigest()
    assert "Python" in result.validator_identity
    rejected = backend.validate_test_source("return 1\n", str(tmp_path))
    assert not rejected.ok and rejected.status == "syntax_error"


def test_failed_validator_is_not_a_syntax_verdict(configured, monkeypatch, tmp_path, default_run):
    class FailedExecutor:
        def run(self, *args, **kwargs):
            return CompletedProcess([], 1, "", "interpreter unavailable")

    monkeypatch.setattr("langadapter.execution.LocalExecutor.run", FailedExecutor.run)
    result = GenericBackend("python", (".py",)).validate_test_source("assert True\n", str(tmp_path))
    assert not result.ok and result.status == "validator_error"
    assert "interpreter unavailable" in result.message


def test_pruning_and_finalization_use_public_operations(configured, monkeypatch, default_run):
    calls = []
    prune = ProjectPythonBackend.prune_tests
    finalize = ProjectPythonBackend.finalize_source

    def record_prune(self, source, ids):
        calls.append("prune")
        return prune(self, source, ids)

    def record_finalize(self, source):
        calls.append("finalize")
        return finalize(self, source)

    monkeypatch.setattr(ProjectPythonBackend, "prune_tests", record_prune)
    monkeypatch.setattr(ProjectPythonBackend, "finalize_source", record_finalize)
    source = ("def test_kept():\n    assert True\n\n"
              "# detached comment\n\n"
              "def test_dropped():\n    assert False\n")
    backend = GenericBackend("python", (".py",))
    result = backend.finalize_test_source(
        backend.prune_tests(source, ["pkg.test_s.test_dropped"]))
    assert "test_kept" in result and "test_dropped" not in result
    assert "detached comment" not in result
    assert calls == ["prune", "finalize"]


def test_validity_records_use_public_operations(configured, tmp_path, monkeypatch, default_run):
    calls = []
    for method in ("target_identity", "analyze_validity", "screen_source"):
        original = getattr(ProjectPythonBackend, method)

        def record(self, *args, _original=original, _method=method, **kwargs):
            calls.append(_method)
            return _original(self, *args, **kwargs)

        monkeypatch.setattr(ProjectPythonBackend, method, record)
    target = tmp_path / "mod.py"
    target.write_text("def sign(x):\n    return x > 0\n", encoding="utf-8")
    backend = GenericBackend("python", (".py",))
    identity = backend.generated_test_target_identity(str(target), str(tmp_path))
    source = "import mod\n\ndef test_path():\n    assert mod.__file__\n"
    findings = backend.analyze_generated_test_validity(source, identity)
    assert findings and findings[0].rule_id == "target_source_path"
    assert findings[0].policy_version == identity.policy_version
    # A review signal: the test lists the files of a directory.
    flags = backend.screen_generated_test_source(
        "import os\n\n"
        "def test_listing():\n    assert os.listdir('.')\n", identity)
    assert flags
    assert identity.record()["source_path"] == str(target.resolve())
    assert calls == ["target_identity", "analyze_validity", "screen_source"]


def test_failed_validity_parse_is_not_a_clean_report(configured, tmp_path, default_run):
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    backend = GenericBackend("python", (".py",))
    identity = backend.generated_test_target_identity(str(target), str(tmp_path))
    with pytest.raises(SyntaxError):
        backend.analyze_generated_test_validity("def broken(:", identity)
    (flag,) = backend.screen_generated_test_source("def broken(:", identity)
    assert flag.rule_id == "unparseable_source"
    assert flag.start_line == flag.end_line == 1
    assert flag.start_col == flag.end_col > 0


def _raise(message):
    def broken(self, *args, **kwargs):
        raise RuntimeError(message)
    return broken


def test_an_offered_target_identity_that_does_not_complete_is_an_error(
        configured, tmp_path, monkeypatch, default_run):
    """An offered validity policy whose identity fails is an error, never a
    stand-in that silently turns the gate off."""
    from testuary.consolidation import ValidityUnresolved
    monkeypatch.setattr(ProjectPythonBackend, "target_identity",
                        _raise("the identity probe failed"))
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(ValidityUnresolved,
                       match="target_identity did not complete: infrastructure_failure"):
        GenericBackend("python", (".py",)).generated_test_target_identity(
            str(target), str(tmp_path))


def test_a_validity_analysis_that_does_not_complete_is_an_error(
        configured, tmp_path, monkeypatch, default_run):
    """An analysis without an answer is an error, never an empty report."""
    from testuary.consolidation import ValidityUnresolved
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    backend = GenericBackend("python", (".py",))
    identity = backend.generated_test_target_identity(str(target), str(tmp_path))
    monkeypatch.setattr(ProjectPythonBackend, "analyze_validity",
                        _raise("the analysis failed"))
    with pytest.raises(ValidityUnresolved,
                       match="analyze_validity did not complete: infrastructure_failure"):
        backend.analyze_generated_test_validity("def test_a():\n    pass\n", identity)


def test_a_validity_screening_that_does_not_complete_is_an_error(
        configured, tmp_path, monkeypatch, default_run):
    from langadapter.sources import PolicyReport
    from testuary.consolidation import ValidityUnresolved
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    backend = GenericBackend("python", (".py",))
    identity = backend.generated_test_target_identity(str(target), str(tmp_path))
    monkeypatch.setattr(
        ProjectPythonBackend, "screen_source",
        lambda self, source, target: PolicyReport(
            "timeout", (), identity.policy_version, "host-ast", detail="too slow"))
    with pytest.raises(ValidityUnresolved, match="screen_source did not complete: timeout"):
        backend.screen_generated_test_source("def test_a():\n    pass\n", identity)


@pytest.mark.parametrize("status", ["infrastructure_failure", "timeout", "missing_setup"])
def test_a_merge_without_an_answer_is_an_error(configured, monkeypatch, status, default_run):
    """A merge whose tools did not run is an error, never "no merge"."""
    from langadapter.sources import SourceResult
    backend = GenericBackend("python", (".py",))
    monkeypatch.setattr(
        type(backend.project_backend().sources), "merge_sources",
        lambda self, first, second, *, separator: SourceResult(
            status, first, "", detail="the merge tool did not run"))
    with pytest.raises(RuntimeError, match=f"merge did not complete: {status}"):
        backend.merge_test_files("def test_a():\n    pass\n", "def test_b():\n    pass\n")


def test_a_refused_merge_is_no_merge(configured, monkeypatch, default_run):
    """A merge the language refuses for these suites keeps the fallback."""
    from langadapter.sources import SourceResult
    backend = GenericBackend("python", (".py",))
    monkeypatch.setattr(
        type(backend.project_backend().sources), "merge_sources",
        lambda self, first, second, *, separator: SourceResult(
            "refused", first, "host-ast", detail="the classes differ"))
    assert backend.merge_test_files("def test_a():\n    pass\n",
                                    "def test_b():\n    pass\n") is None


def test_failure_listing_uses_public_suite_run(configured, tmp_path, monkeypatch, default_run):
    from langadapter.outcomes import ExecutionIdentity, ObservationIdentity, ToolchainIdentity
    from langadapter.outcomes import Outcome
    from langadapter.runs import RunCounts, SideEffects, SuiteRun, TestOutcome
    from langadapter.suites import TestId

    child = tmp_path / "sub"
    child.mkdir()
    files = (tmp_path / "test_root.py", child / "test_child.py")
    for test in files:
        test.write_text("def test_a(): pass\n", encoding="utf-8")
    requests = []

    def run_suite(self, request):
        requests.append(request)
        return SuiteRun(
            Outcome.TEST_FAILURES, "red", RunCounts(failed=2),
            tuple(TestOutcome(TestId("sub/test_child.py", "TestX.test_a", value), "failed")
                  for value in ("1", "2")), (), 1.0, "failed", None,
            ObservationIdentity(None, (), ToolchainIdentity("CPython", "3.12"), None,
                                ExecutionIdentity("local")), SideEffects())

    monkeypatch.setattr(ProjectPythonBackend, "run_suite", run_suite)
    result = GenericBackend("python", (".py",)).failing_test_ids(
        [str(test) for test in files], str(tmp_path), timeout=60)
    # Each failing parametrized case is listed by the id prune_tests removes.
    assert result == ["sub.test_child.TestX.test_a[1]", "sub.test_child.TestX.test_a[2]"]
    request, = requests
    assert request.coverage_target is None
    assert request.remove_created_files and request.isolation_config
    # No instrument is placed beside the files.
    assert request.suite.setup_files == ()
    assert not any((test.parent / "conftest.py").is_file() for test in files)


@pytest.mark.parametrize("outcome", ["timed_out", "infrastructure_failure", "collection_failed"])
def test_undecided_failure_listing_is_not_green(configured, tmp_path, monkeypatch, outcome, default_run):
    from langadapter.outcomes import ExecutionIdentity, ObservationIdentity, ToolchainIdentity
    from langadapter.outcomes import Outcome
    from langadapter.runs import RunCounts, SuiteRun

    test = tmp_path / "test_a.py"
    test.write_text("def test_a(): pass\n", encoding="utf-8")
    monkeypatch.setattr(ProjectPythonBackend, "run_suite", lambda self, request:
                        SuiteRun(Outcome(outcome), "undecided", RunCounts(), (), (),
                                 None, "", None, ObservationIdentity(
                                     None, request.suite.identity(),
                                     ToolchainIdentity("python", "fixture"), None,
                                     ExecutionIdentity("local"))))
    assert GenericBackend("python", (".py",)).failing_test_ids([str(test)], str(tmp_path), timeout=30) is None


def test_failures_are_listed_whatever_the_runners_exit_status(configured, tmp_path, monkeypatch, default_run):
    """Failed and errored cases are listed even after an internal error (exit
    status 3), so the strip prunes them instead of the merge ending red."""
    from langadapter.outcomes import ExecutionIdentity, ObservationIdentity, ToolchainIdentity
    from langadapter.outcomes import Outcome
    from langadapter.runs import RunCounts, SuiteRun, TestOutcome
    from langadapter.suites import TestId

    test = tmp_path / "test_a.py"
    test.write_text("def test_a(): pass\n", encoding="utf-8")
    cases = (TestOutcome(TestId("test_a.py", "test_a"), "failed"),
             TestOutcome(TestId("test_a.py", "test_b"), "error"),
             TestOutcome(TestId("test_a.py", "test_c"), "passed"))
    monkeypatch.setattr(ProjectPythonBackend, "run_suite", lambda self, request:
                        SuiteRun(Outcome.INFRASTRUCTURE_FAILURE, "internal error",
                                 RunCounts(failed=1, errors=1, passed=1), cases, (),
                                 None, "INTERNALERROR", 3, ObservationIdentity(
                                     None, request.suite.identity(),
                                     ToolchainIdentity("python", "fixture"), None,
                                     ExecutionIdentity("local"))))
    assert GenericBackend("python", (".py",)).failing_test_ids([str(test)], str(tmp_path), timeout=30) == [
        "test_a.test_a", "test_a.test_b"]


def test_a_plugin_crash_lists_only_the_failures_that_ran_before_it(configured, tmp_path, default_run):
    """After a plugin crash (exit status 3), only the real failures that ran
    before it are listed, not pytest's synthetic internal-error record."""
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "conftest.py").write_text(
        "def pytest_runtest_logreport(report):\n"
        "    if report.when == 'call' and report.nodeid.endswith('test_crash'):\n"
        "        raise RuntimeError('plugin crash while recording')\n", encoding="utf-8")
    test = tests / "test_x.py"
    test.write_text(
        "def test_fail_one():\n    assert False\n\n\ndef test_pass():\n    pass\n\n\n"
        "def test_fail_two():\n    assert 1 == 2\n\n\ndef test_crash():\n    pass\n\n\n"
        "def test_after():\n    assert False\n", encoding="utf-8")
    assert GenericBackend("python", (".py",)).failing_test_ids([str(test)], str(tmp_path), timeout=60) == [
        "test_x.test_fail_one", "test_x.test_fail_two"]


def test_the_failure_listing_places_no_instrument(configured, tmp_path, default_run):
    test = tmp_path / "test_seed.py"
    test.write_text("import random\n"
                    "def test_seed_zero():\n"
                    "    assert random.random() == 0.8444218515250481\n",
                    encoding="utf-8")
    result = GenericBackend("python", (".py",)).failing_test_ids(
        [str(test)], str(tmp_path), timeout=60)
    assert result == ["test_seed.test_seed_zero"]
    assert not (tmp_path / "conftest.py").is_file()


def _plain(value):
    """A value as JSON holds it: tuples and sets as sorted or ordered lists."""
    if isinstance(value, (set, frozenset)):
        return sorted(value)
    if isinstance(value, tuple):
        return list(value)
    return value


#: Reference measurements of the suite below. The matrix omits its target path.
REFERENCE_SUITE = {"returncode": 0, "timed_out": False, "collected": 2, "passed": 2,
                "failed": 0, "errors": 0, "line_coverage": 100.0,
                "branch_coverage": None, "missing_lines": [], "missing_branches": [],
                "covered_lines": [1, 4], "covered_branches": []}
REFERENCE_MATRIX = {"schema_version": 1, "branches": [],
                 "lines": [{"line": 1, "status": "covered"},
                           {"line": 4, "status": "covered"}],
                 "summary": {"covered_branches": 0, "covered_lines": 2,
                             "missing_branches": 0, "missing_lines": 0,
                             "total_lines": 2}}


def test_suite_measurement_matches_the_reference_values(configured, tmp_path, monkeypatch, default_run):
    import json

    root = tmp_path / "project"
    tests = root / "tests"
    tests.mkdir(parents=True)
    target = root / "sign.py"
    target.write_text("def sign(x):\n"
                      "    if x > 0:  # pragma: no cover\n"
                      "        return 1\n"
                      "    return -1\n", encoding="utf-8")
    test = tests / "test_sign.py"
    test.write_text("from pathlib import Path\nimport sign as subject\nfrom sign import sign\n"
                    "def test_positive():\n"
                    "    assert sign(2) == 1\n"
                    "    assert Path(subject.__file__).resolve() == Path(__file__).resolve().parents[1] / 'sign.py'\n"
                    "def test_negative(): assert sign(-2) == -1\n", encoding="utf-8")
    backend = GenericBackend("python", (".py",))
    public_matrix = tmp_path / "public.json"
    calls = []
    original = ProjectPythonBackend.run_suite

    def recorded(self, request):
        calls.append((self.context.root, request))
        assert Path(self.context.root) != root
        assert Path(request.coverage_target).read_bytes() == target.read_bytes()
        return original(self, request)

    monkeypatch.setattr(ProjectPythonBackend, "run_suite", recorded)
    new = backend.run_test_files(
        [str(test)], str(root), timeout=60, measure_coverage_of=str(target),
        coverage_matrix_path=str(public_matrix))
    assert new.ok, new.output
    assert {field: _plain(getattr(new, field)) for field in REFERENCE_SUITE} == REFERENCE_SUITE
    matrix = json.loads(public_matrix.read_text())
    assert matrix.pop("target_file") == str(target)
    assert matrix == REFERENCE_MATRIX
    # The covered statements: lines 2 and 3 are excluded by the pragma.
    assert new.covered_lines == (1, 4)
    assert [item["line"] for item in json.loads(public_matrix.read_text())["lines"]] == [1, 4]
    (copied_root, request), = calls
    assert request.coverage_target == request.module_under_test == str(Path(copied_root) / "sign.py")
    assert not Path(copied_root).exists()
    assert request.remove_created_files and request.isolation_config


def test_failed_matrix_write_is_not_a_complete_measurement(configured, tmp_path, default_run):
    target = tmp_path / "sign.py"
    target.write_text("def sign(x): return x\n", encoding="utf-8")
    test = tmp_path / "test_sign.py"
    test.write_text("from sign import sign\n"
                    "def test_sign(): assert sign(2) == 2\n", encoding="utf-8")
    result = GenericBackend("python", (".py",)).run_test_files(
        [str(test)], str(tmp_path), timeout=60, measure_coverage_of=str(target),
        coverage_matrix_path=str(tmp_path / "absent" / "matrix.json"))
    assert not result.ok and result.passed == 1
    assert result.line_coverage is None
    assert "matrix could not be written" in result.output


def test_counts_are_pytests_own(configured, tmp_path, default_run):
    """Counts follow pytest's closing summary: an unimportable module is one
    error (NO_RESULTS), and a test's own text such as "3 passed" is no count."""
    from testuary._test_output import canonical_test_summary_line

    test = tmp_path / "test_broken.py"
    test.write_text("import not_a_module_anywhere\n\ndef test_a(): pass\n", encoding="utf-8")
    result = GenericBackend("python", (".py",)).run_test_files([str(test)], str(tmp_path), timeout=60)
    assert (result.collected, result.passed, result.failed, result.errors) == (0, 0, 0, 1)
    assert canonical_test_summary_line(result).startswith(
        "TESTS: NO_RESULTS | collected 0 | passed 0 | failed 0 | errors 1 | elapsed ")
    assert "generated xml file" not in result.output
    # A failing test whose teardown fails too (a failure and an error), a
    # non-strict xpass (neither), and an assertion naming "3 passed".
    mixed = tmp_path / "test_mixed.py"
    mixed.write_text(
        "import pytest\n\n\n@pytest.fixture\ndef broken():\n    yield 1\n"
        "    raise RuntimeError('teardown boom')\n\n\n"
        "def test_fails(broken):\n    assert False, 'expected 3 passed and 2 failed'\n\n\n"
        "@pytest.mark.xfail(reason='non-strict')\ndef test_xpass():\n    assert True\n\n\n"
        "def test_ok():\n    print('500 error from a fake server')\n", encoding="utf-8")
    result = GenericBackend("python", (".py",)).run_test_files([str(mixed)], str(tmp_path), timeout=60)
    assert "expected 3 passed" in result.output
    assert (result.collected, result.passed, result.failed, result.errors) == (3, 1, 1, 1)
    assert canonical_test_summary_line(result).startswith(
        "TESTS: FAILED | collected 3 | passed 1 | failed 1 | errors 1 | elapsed ")


@pytest.mark.parametrize("config", [
    ("pytest.ini", "[pytest]\npython_files = check_*.py\n"),
    ("pyproject.toml", '[tool.pytest.ini_options]\npython_files = ["check_*.py"]\n')])
def test_a_test_directory_expands_with_the_projects_python_files(
        configured, monkeypatch, tmp_path, config, default_run):
    """A test directory expands with the project's own ``python_files``
    (``pytest_config = auto``), as pytest collects in place: LangAdapter
    asks pytest with the run's configuration."""
    import subprocess

    monkeypatch.setattr(config_module, "load_config", lambda section: {"pytest_config": "auto"})

    def _test_files_of(paths, root):
        return ProjectPythonBackend(language_project_context("python", root)).test_files(paths)

    name, text = config
    (tmp_path / name).write_text(text, encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "check_a.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    (tests / "test_b.py").write_text("def test_b():\n    pass\n", encoding="utf-8")
    assert _test_files_of([str(tests)], str(tmp_path)) == [str(tests / "check_a.py")]
    in_place = subprocess.run(
        [sys.executable, "-m", "pytest", "--collect-only", "-q", "-p", "no:cacheprovider",
         "tests"], cwd=tmp_path, capture_output=True, text=True, timeout=120)
    assert "check_a.py::test_a" in in_place.stdout and "test_b" not in in_place.stdout
    result = GenericBackend("python", (".py",)).run_test_files([str(tests)], str(tmp_path), timeout=60)
    assert (result.collected, result.passed) == (1, 1)
    assert _test_files_of([str(tests)], str(tests)) == [str(tests / "test_b.py")]


def test_leak_recording_keeps_custom_setup(configured, tmp_path, default_run):
    target = tmp_path / "sign.py"
    target.write_text("def sign(x): return x\n", encoding="utf-8")
    setup = tmp_path / "conftest.py"
    setup_source = "import pytest\n@pytest.fixture\ndef value(): return 2\n"
    setup.write_text(setup_source, encoding="utf-8")
    test = tmp_path / "test_sign.py"
    test.write_text("import sign\n"
                    "def test_changes_module(value):\n"
                    "    sign.sign = lambda x: value\n"
                    "    assert sign.sign(0) == 2\n", encoding="utf-8")
    result = GenericBackend("python", (".py",)).run_test_files(
        [str(test)], str(tmp_path), timeout=60, measure_coverage_of=str(target),
        record_leaks=True)
    assert result.ok, result.output
    assert setup.read_text(encoding="utf-8") == setup_source
    assert result.leaks is not None
    assert tuple(leak.test for leak in result.leaks) == ("test_sign.test_changes_module",)


#: Reference evidence for the suite below, each test's tokens sorted.
REFERENCE_EVIDENCE = {
    "per_test_tokens": {
        "test_sign.TestNegative.test_negative": [
            "B:sign.py:-1->2", "B:sign.py:2->-1", "C:crash", "L:sign.py:2"],
        "test_sign.test_positive": [
            "B:sign.py:-1->2", "B:sign.py:2->-1", "K:named", "L:sign.py:2"]},
    "tests": ["test_sign.TestNegative.test_negative", "test_sign.test_positive"],
    "target_total_lines": 2, "target_covered_lines": 2,
    "target_total_branches": 0, "target_covered_branches": 0,
    "suite_covered_lines": [1, 2], "suite_covered_branches": []}


def test_per_test_evidence_matches_the_reference_tokens_and_stored_kills(
        configured, tmp_path, monkeypatch, default_run):
    import json

    target = tmp_path / "sign.py"
    target.write_text("def sign(x):\n    return x > 0\n", encoding="utf-8")
    test = tmp_path / "test_sign.py"
    test.write_text("import pytest\nfrom sign import sign\n"
                    "@pytest.mark.parametrize('x', [1, 2])\n"
                    "def test_positive(x): assert sign(x)\n"
                    "class TestNegative:\n"
                    "    def test_negative(self): assert not sign(-1)\n", encoding="utf-8")
    matrix = tmp_path / "kills.json"
    matrix.write_text(json.dumps({"mutants": [
        {"id": "named", "killers": ["test_sign.py::test_positive[1]"],
         "killing_tests": killing_tests(["test_sign.py::test_positive[1]"])},
        {"id": "crash", "active_at_end": "test_sign.py::TestNegative::test_negative"},
    ]}), encoding="utf-8")
    backend = GenericBackend("python", (".py",))
    options = dict(measure_coverage_of=str(target), timeout=60,
                   warmup=False, kill_matrix_path=str(matrix))
    calls = []
    original = ProjectPythonBackend.per_test_evidence

    def recorded(self, request):
        calls.append(request)
        return original(self, request)

    monkeypatch.setattr(ProjectPythonBackend, "per_test_evidence", recorded)
    new = backend.collect_eviction_data([str(test)], str(tmp_path), **options)
    assert new is not None
    assert {test: sorted(tokens) for test, tokens in new.per_test_tokens.items()} == (
        REFERENCE_EVIDENCE["per_test_tokens"])
    assert new.includes_kills
    assert sorted(new.per_test_time_ms) == REFERENCE_EVIDENCE["tests"]
    assert new.suite_time_ms == pytest.approx(sum(new.per_test_time_ms.values()))
    for field in ("target_total_lines", "target_covered_lines", "target_total_branches",
                  "target_covered_branches", "suite_covered_lines", "suite_covered_branches"):
        assert _plain(getattr(new, field)) == REFERENCE_EVIDENCE[field], field
    request, = calls
    assert request.warmup is False and request.mutation_run is None
    assert request.isolation_config and request.remove_created_files
    assert new.per_test_tokens["test_sign.TestNegative.test_negative"] & {"C:crash"}
    assert "K:crash" not in new.per_test_tokens["test_sign.TestNegative.test_negative"]


def test_a_run_reports_its_failing_tests_with_their_messages(configured, tmp_path, default_run):
    """A run carries its failed and errored tests with their messages."""
    test = tmp_path / "test_two.py"
    test.write_text("import pytest\n\n\n@pytest.fixture\ndef broken():\n"
                    "    raise RuntimeError('no data')\n\n\n"
                    "def test_fails():\n    assert 1 == 2, 'one is not two'\n\n\n"
                    "def test_errors(broken):\n    pass\n\n\n"
                    "def test_passes():\n    pass\n", encoding="utf-8")
    result = GenericBackend("python", (".py",)).run_test_files([str(test)], str(tmp_path), timeout=60)
    assert [(f.test, f.outcome) for f in result.failures] == [
        ("test_two.test_fails", "failed"), ("test_two.test_errors", "error")]
    assert "one is not two" in result.failures[0].message
    assert "no data" in result.failures[1].message


def test_unmapped_stored_killer_keeps_evidence_coverage_only(configured, tmp_path, default_run):
    import json

    target = tmp_path / "sign.py"
    target.write_text("def sign(x): return x\n", encoding="utf-8")
    test = tmp_path / "test_sign.py"
    test.write_text("from sign import sign\n"
                    "def test_positive(): assert sign(2) == 2\n", encoding="utf-8")
    matrix = tmp_path / "kills.json"
    matrix.write_text(json.dumps({"mutants": [{"id": "m", "killers": [
        "test_sign.py::test_positive", "elsewhere.py::test_other"]}]}), encoding="utf-8")
    result = GenericBackend("python", (".py",)).collect_eviction_data(
        [str(test)], str(tmp_path), measure_coverage_of=str(target), timeout=60,
        warmup=False, kill_matrix_path=str(matrix))
    assert result is not None and not result.includes_kills
    assert all(not token.startswith("K:") for items in result.per_test_tokens.values()
               for token in items)


def test_kills_without_a_named_killer_keep_the_kill_reduction(configured, tmp_path, default_run):
    """A kill naming no killer (the mutant breaks the import) is reproduced by
    every kept test, so it keeps the kill reduction. An unresolved one does not."""
    import json

    target = tmp_path / "sign.py"
    target.write_text("def sign(x): return x\n", encoding="utf-8")
    test = tmp_path / "test_sign.py"
    test.write_text("from sign import sign\n"
                    "def test_positive(): assert sign(2) == 2\n", encoding="utf-8")
    matrix = tmp_path / "kills.json"
    backend = GenericBackend("python", (".py",))
    options = dict(measure_coverage_of=str(target), timeout=60, warmup=False,
                   kill_matrix_path=str(matrix))
    import_kill = {"id": "m", "status": "killed", "killers": []}
    matrix.write_text(json.dumps({"mutants": [import_kill]}), encoding="utf-8")
    result = backend.collect_eviction_data([str(test)], str(tmp_path), **options)
    assert result is not None and result.includes_kills
    assert all(not token.startswith("K:") for items in result.per_test_tokens.values()
               for token in items)
    matrix.write_text(json.dumps({"mutants": [import_kill, {
        "id": "u", "status": "killed", "killers": [], "unresolved": True}]}),
        encoding="utf-8")
    result = backend.collect_eviction_data([str(test)], str(tmp_path), **options)
    assert result is not None and not result.includes_kills


def test_a_stored_kill_matrix_reaches_the_adapter_in_this_versions_words(
        configured, tmp_path, default_run, monkeypatch):
    """A stored kill matrix of version 2 reaches the adapter's kill_tokens
    as this version's, the words an adapter reads."""
    import json

    target = tmp_path / "sign.py"
    target.write_text("def sign(x): return x\n", encoding="utf-8")
    test = tmp_path / "test_sign.py"
    test.write_text("from sign import sign\n"
                    "def test_positive(): assert sign(2) == 2\n", encoding="utf-8")
    matrix = tmp_path / "kills.json"
    matrix.write_text(json.dumps({"schema_version": 2, "mutants": [
        {"id": "m", "status": "ok_killed", "killers": []}]}), encoding="utf-8")
    received = []
    original = ProjectPythonBackend.kill_tokens

    def recorded(self, kill_matrix, test_ids):
        received.append(kill_matrix)
        return original(self, kill_matrix, test_ids)

    monkeypatch.setattr(ProjectPythonBackend, "kill_tokens", recorded)
    result = GenericBackend("python", (".py",)).collect_eviction_data(
        [str(test)], str(tmp_path), measure_coverage_of=str(target), timeout=60,
        warmup=False, kill_matrix_path=str(matrix))
    assert result is not None and result.includes_kills
    (stored,) = received
    assert stored["schema_version"] == 3
    assert [row["status"] for row in stored["mutants"]] == ["killed"]


def _evidence_with_an_unmatched_context(configured, tmp_path, monkeypatch):
    """The per-test evidence of a one-test suite whose run also reports a
    coverage context that no test claims."""
    import dataclasses

    from langadapter.runs import ContextEvidence
    from langadapter.suites import LineItem

    target = tmp_path / "sign.py"
    target.write_text("def sign(x): return x\n", encoding="utf-8")
    test = tmp_path / "test_sign.py"
    test.write_text("from sign import sign\n"
                    "def test_positive(): assert sign(2) == 2\n", encoding="utf-8")
    original = ProjectPythonBackend.per_test_evidence

    def with_orphan(self, request):
        result = original(self, request)
        return dataclasses.replace(result, unmatched_items=(
            ContextEvidence("[engine:junit-vintage]/[runner:SignTest]/[test:odd%5B0%5D]",
                            frozenset({LineItem("sign.py", 1)})),))

    monkeypatch.setattr(ProjectPythonBackend, "per_test_evidence", with_orphan)
    logged = []
    result = GenericBackend("python", (".py",)).collect_eviction_data(
        [str(test)], str(tmp_path), measure_coverage_of=str(target), timeout=60,
        warmup=False, log=logged.append)
    return result, logged


def test_an_unmatched_coverage_context_stays_beside_the_tests_in_python(
        configured, tmp_path, monkeypatch, default_run):
    """Python keeps the coverage contexts that no reported test claims."""
    result, _logged = _evidence_with_an_unmatched_context(configured, tmp_path, monkeypatch)
    assert any(name.startswith("[engine:junit-vintage]") for name in result.per_test_tokens)


def test_an_unmatched_coverage_context_is_no_pseudo_test(configured, tmp_path, monkeypatch, default_run):
    """A language that drops them (Java) gives the reduction only the
    suite's tests, and the log names what was left out."""
    monkeypatch.setattr(ProjectPythonBackend, "unmatched_evidence_kept", False, raising=False)
    result, logged = _evidence_with_an_unmatched_context(configured, tmp_path, monkeypatch)
    assert list(result.per_test_tokens) == ["test_sign.test_positive"]
    assert any("1 coverage context(s) name no test" in line for line in logged)


def test_python_keeps_unmatched_contexts_and_java_does_not():
    assert GenericBackend("python", (".py",))._keeps_unmatched_contexts()
    assert not GenericBackend("java", (".java",))._keeps_unmatched_contexts()


def test_syntax_repair_keeps_the_project_validator_and_report(monkeypatch, tmp_path, default_run):
    """Python's repair asks the project interpreter's validator for each
    candidate, and the report keeps what the caller put in it."""
    from langadapter.sources import SourceValidation

    calls = []

    def validate(self, source):
        calls.append(source)
        if len(calls) == 1:
            return SourceValidation("syntax_error", "unmatched brace", 1, 0, "target compiler")
        return SourceValidation("valid", "", 0, 0, "target compiler")

    monkeypatch.setattr(ProjectPythonBackend, "validate_source", validate)
    report = {"previous": "preserved"}
    source = "}\ndef test_a():\n    pass\n"
    repaired = GenericBackend("python", (".py",)).salvage_syntax(source, str(tmp_path), report=report)
    assert calls == [source, repaired]
    assert repaired == "\ndef test_a():\n    pass\n"
    assert report == {"previous": "preserved", "status": "repaired",
                      "repairs": 1, "message": ""}


def test_syntax_repair_does_not_turn_an_unrun_validator_into_bad_code(monkeypatch, tmp_path, default_run):
    calls = []

    def validate(self, source):
        calls.append(source)
        raise RuntimeError("could not start")

    monkeypatch.setattr(ProjectPythonBackend, "validate_source", validate)
    source = "def test_a():\n    pass\n"
    report = {}
    assert GenericBackend("python", (".py",)).salvage_syntax(
        source, str(tmp_path), report=report) is None
    assert calls == [source]
    assert report["status"] == "validator_failed" and report["repairs"] == 0
    assert "could not start" in report["message"]


@pytest.mark.parametrize("method", ["prune_unresolvable_imports", "prune_collection_breakers"])
def test_source_recovery_preserves_public_execution_failure(
        configured, tmp_path, monkeypatch, method, default_run):
    from langadapter.sources import INFRASTRUCTURE_FAILURE, SourceResult
    from testuary.consolidation import ValidationUnresolved

    source = "def test_a(): pass\n"

    def failed(self, text, **options):
        return SourceResult(INFRASTRUCTURE_FAILURE, text, "target", detail="could not run")

    monkeypatch.setattr(ProjectPythonBackend, method, failed)
    options = {"timeout": 60} if method == "prune_collection_breakers" else {}
    with pytest.raises(ValidationUnresolved, match="could not run"):
        getattr(GenericBackend("python", (".py",)), method)(source, str(tmp_path), **options)


@pytest.mark.parametrize("missing_operation", ["coverage", "mutation", "suite_runs"])
def test_dependency_preflight_reads_the_project_capabilities(
        configured, tmp_path, monkeypatch, missing_operation, default_run):
    from langadapter.capabilities import Availability, Capability, CapabilityReport
    from langadapter.outcomes import Outcome

    contexts = []

    from langadapter.capabilities import CAPABILITIES
    from testuary.languages import assess
    family = {"suite_runs": "run tests"}.get(missing_operation, missing_operation)

    def capabilities(self):
        contexts.append(self.context)
        return CapabilityReport(
            "python", self.interpreter, Outcome.COMPLETED, "", "Python 3.12", tuple(
                Capability(operation, Availability.MISSING_SETUP,
                           "tool missing from the target interpreter")
                if operation == missing_operation else
                Capability(operation, Availability.AVAILABLE)
                for operation in CAPABILITIES), (), "arcs", ())

    monkeypatch.setattr(ProjectPythonBackend, "capabilities", capabilities)
    assessment = assess(GenericBackend("python", (".py",)).capabilities(source_dir=str(tmp_path)))
    refusal = (f"The language adapter lacks {family}, which Testuary needs: tool missing from "
               "the target interpreter")
    # Coverage runs the suite, so missing suite runs refuse coverage too.
    expected = [refusal] if family != "run tests" else [
        refusal, refusal.replace("run tests", "coverage")]
    assert list(assessment.refusals) == expected
    # Without suite runs, the families that run the suite list fallbacks too.
    assert (assessment.fallbacks == ()) is (family != "run tests")
    assert contexts[0].root == str(tmp_path.resolve())


def test_dependency_preflight_does_not_treat_an_unrun_probe_as_ready(
        configured, tmp_path, monkeypatch, default_run):
    from langadapter.capabilities import CapabilityReport
    from langadapter.outcomes import Outcome

    from testuary.languages import assess

    monkeypatch.setattr(ProjectPythonBackend, "capabilities", lambda self: CapabilityReport(
        "python", self.interpreter, Outcome.INFRASTRUCTURE_FAILURE,
        "interpreter could not start", None, (), (), "arcs", ()))
    assert assess(GenericBackend("python", (".py",)).capabilities(source_dir=str(tmp_path))).refusals == (
        "the project environment could not be probed: interpreter could not start",)


def test_dependency_preflight_keeps_the_host_solver_check(
        configured, tmp_path, monkeypatch):
    from testuary import project as _execution_tools

    monkeypatch.setattr(ProjectPythonBackend, "capabilities", lambda self: pytest.fail(
        "the solver check reads no capability report"))
    monkeypatch.setattr(config_module, "get_agent_config", lambda phase: {
        "reduction_solver": "exact", "engine": "cosmic_ray"})
    monkeypatch.setattr(_execution_tools, "host_solver_mismatches", lambda: ("CBC unavailable",))
    assert GenericBackend("python", (".py",)).check_dependencies() == ["CBC unavailable"]


def test_the_solver_check_reads_the_mutation_phase_over_the_coverage_phase(tmp_path):
    """A run whose phases all choose the greedy solver needs no exact
    solver's tools: [mutation] holds no reduction_solver of its own and takes
    [coverage]'s, as the mutation phase does. check_dependencies returns this
    check."""
    settings = resolve_settings(base_dir=tmp_path, home=tmp_path, project_settings=False, environment={},
                                models={}, overrides={"starter": {"reduction_solver": "greedy"},
                                                      "coverage": {"reduction_solver": "greedy"}})
    with config_module.use_run_settings(settings):
        assert GenericBackend("python", (".py",))._solver_dependencies() == []


def test_orchestrator_preflight_stops_before_phases_when_target_tools_are_missing(
        configured, tmp_path, monkeypatch, default_run):
    from langadapter.capabilities import Availability, Capability, CapabilityReport
    from langadapter.outcomes import Outcome
    from testuary import languages, orchestrator
    from testuary.settings import ConfigurationError

    from langadapter.capabilities import CAPABILITIES

    monkeypatch.setattr(ProjectPythonBackend, "capabilities", lambda self: CapabilityReport(
        "python", self.interpreter, Outcome.COMPLETED, "", "Python 3.12", tuple(
            Capability(operation, Availability.MISSING_SETUP, "coverage missing in target")
            if operation == "coverage" else Capability(operation, Availability.AVAILABLE)
            for operation in CAPABILITIES), (), "arcs", ()))
    monkeypatch.setattr(config_module, "get_agent_config", lambda phase: {
        "reduction_solver": "greedy", "engine": "cosmic_ray"})
    backend = GenericBackend("python", (".py",))
    monkeypatch.setattr(languages, "get_backend", lambda language: backend)
    with pytest.raises(ConfigurationError, match="coverage missing in target") as refused:
        orchestrator._preflight_run_dependencies({"python"}, tmp_path)
    assert "No phase ran" in str(refused.value)


def test_unmatched_contexts_follow_the_adapters_facts_not_its_language(tmp_path, default_run):
    """A language whose ``LanguageFacts`` keep unmatched contexts has them
    kept, whatever its name: only Python's were, by name."""
    import langadapter as lb
    from langadapter.capabilities import LanguageFacts

    class Facts(lb.LangAdapter):
        language = "factlang"

        def language_facts(self):
            return LanguageFacts(
                language="Facts", test_framework="none", code_fence="text",
                runtime_version=None, test_file_patterns=("test_*.fl",),
                test_function_pattern="test_*", test_class_pattern=None,
                timeout_marker=None, unmatched_evidence_kept=True)

    lb.register("factlang", (".fl",), Facts)
    try:
        assert GenericBackend("factlang", (".fl",))._keeps_unmatched_contexts(str(tmp_path))
    finally:
        lb.unregister("factlang")
