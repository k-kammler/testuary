"""Adapter staging uses public ownership without changing native request policy."""
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

import langadapter as lb
from langadapter.outcomes import ExecutionIdentity, FileIdentity, ObservationIdentity, ToolchainIdentity
from langadapter.preparation import original_text
from langadapter.python.adapter import PythonAdapter as ProjectPythonBackend
from langadapter.runs import CoverageObservation, PerTestEvidence, RunCounts, SuiteRun, TestOutcome
from langadapter.suites import TestId
from testuary import config
from testuary import project as project_module
from testuary.languages import preparation
from testuary.languages.generic import GenericBackend

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Controlled adapter checks must not execute native tools")
    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    # The bytecode store's helper is a native command, which its own tests run.
    monkeypatch.setattr(preparation, "bytecode_store", lambda backend: {})
    monkeypatch.setattr(config, "project_exec_mode", lambda: "local")
    monkeypatch.setattr(project_module, "project_work_dir", lambda: None)
    monkeypatch.setattr(config, "load_config", lambda section: {
        "pytest_config": "none", "frozen_module_strategy": "skip",
        "python_executable": sys.executable})


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    (root / "subject.py").write_bytes(b"value = 1\n")
    suite = tmp_path / "delivered"
    suite.mkdir()
    test = suite / "test_subject.py"
    test.write_bytes(b"def test_subject(): pass\n")
    return root, test


def identity(suite):
    return ObservationIdentity(None, suite.identity(), ToolchainIdentity("python", "fixture"),
                               None, ExecutionIdentity("local"))


@pytest.mark.parametrize("inside", [False, True])
def test_public_factory_maps_only_project_owned_suite_paths_and_typed_targets(project, inside):
    root, test = project
    if inside:
        test = root / "test_subject.py"
        test.write_bytes(b"def test_subject(): pass\n")
    context = lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable))
    original = lb.create("python", context)
    suite = lb.Suite(str(test.parent), (test.name,))
    with preparation.prepared_measurement(original, suite, str(root / "subject.py"),
                                         "subject.py", "/external/subject.py") as (backend, staged, targets):
        copy = Path(backend.context.root)
        assert copy != root and copy.is_dir()
        assert backend.context.interpreter is original.context.interpreter
        assert targets[:2] == (str(copy / "subject.py"),) * 2
        external = "/external/subject.py" if sys.platform != "win32" else str(Path("/external/subject.py").resolve())
        assert targets[2] == external
        # A suite outside the project is copied beside the project's copy, in
        # its place below their shared directory.
        assert Path(staged.root) == (copy if inside else copy.parent / test.parent.name)
        assert Path(staged.root) != test.parent
        assert Path(staged.test_files[0]).read_bytes() == test.read_bytes()
        (copy / "subject.py").write_bytes(b"changed\n")
    assert not copy.exists() and (root / "subject.py").read_bytes() == b"value = 1\n"
    assert test.read_bytes() == b"def test_subject(): pass\n"


def _project_venv(root, *, link=False):
    """A project .venv as python -m venv makes it, its interpreter selected."""
    venv = root / ".venv"
    (venv / "bin").mkdir(parents=True)
    (venv / "pyvenv.cfg").write_text("home = base\n", encoding="utf-8")
    interpreter = venv / "bin" / "python"
    if link:
        interpreter.symlink_to(sys.executable)
    else:
        interpreter.write_bytes(b"")
    return interpreter


def test_measurement_copies_leave_out_the_selected_project_environment(project):
    root, test = project
    interpreter = _project_venv(root)
    original = lb.create("python", lb.ProjectContext(str(root), lb.Interpreter(path=str(interpreter))))
    suite = lb.Suite(str(test.parent), (test.name,))
    with preparation.prepared_measurement(original, suite) as (backend, staged, targets):
        copy = Path(backend.context.root)
        assert (copy / "subject.py").read_bytes() == b"value = 1\n"
        assert not (copy / ".venv").exists()
        # The runtime stays the selected interpreter in its original place.
        assert backend.context.interpreter.path == str(interpreter)
    # Without an in-project environment the whole tree is copied.
    assert preparation.project_inputs(lb.ProjectContext(
        str(root), lb.Interpreter(path=sys.executable))) is None


def test_measurement_copies_leave_out_testuarys_own_files(project, monkeypatch, tmp_path):
    """Testuary's configuration and model catalogue inside the source directory
    are not project inputs, so generated tests never find them in the copy."""
    root, test = project
    (root / "config.ini").write_bytes(b"[main]\n")
    (root / "models.ini").write_bytes(b"[synthetic]\nkey = synthetic-value\n")
    outside = tmp_path / "elsewhere.ini"
    outside.write_bytes(b"[main]\n")
    run = _run_reading(root, root / "config.ini", root / "models.ini", outside)
    monkeypatch.setattr(config, "active_settings", lambda: run)
    original = lb.create("python", lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable)))
    suite = lb.Suite(str(test.parent), (test.name,))
    with preparation.prepared_measurement(original, suite) as (backend, staged, targets):
        copy = Path(backend.context.root)
        assert (copy / "subject.py").read_bytes() == b"value = 1\n"
        assert not (copy / "config.ini").exists() and not (copy / "models.ini").exists()
    assert (root / "models.ini").read_bytes() == b"[synthetic]\nkey = synthetic-value\n"
    # The files apply alongside a project environment, which comes first.
    # Files outside the project and a call outside a run change nothing.
    interpreter = _project_venv(root)
    plan = preparation.project_inputs(lb.ProjectContext(str(root), lb.Interpreter(path=str(interpreter))))
    assert plan.exclude == (".venv", "config.ini", "models.ini")
    selected = lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable))
    monkeypatch.setattr(config, "active_settings", lambda: _run_reading(root, outside))
    assert preparation.project_inputs(selected) is None
    monkeypatch.setattr(config, "active_settings", lambda: None)
    assert preparation.project_inputs(selected) is None
    # A guest project in the mapped work directory: the same host files are left out.
    monkeypatch.setattr(config, "active_settings", lambda: run)
    assert preparation.project_inputs(_guest(tmp_path)).exclude == ("config.ini", "models.ini")


def test_measurement_copies_leave_out_the_projects_testuary_dir(project, monkeypatch, tmp_path):
    """A copy of the project itself, as a container run measures it in the
    mapped work directory, leaves out the project's Testuary directory with its
    settings and results, as the run's work copy does."""
    root, _ = project
    (root / ".testuary" / "results").mkdir(parents=True)
    monkeypatch.setattr(config, "active_settings", lambda: _run_reading(root))
    host = lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable))
    assert preparation.project_inputs(host).exclude == (".testuary",)
    assert preparation.project_inputs(_guest(tmp_path)).exclude == (".testuary",)


def _run_reading(project_dir, *files):
    """The settings of a run in ``project_dir`` that read ``files``."""
    from testuary.settings import RunSettings

    return RunSettings(base_dir=project_dir, project_dir=project_dir, environment={},
                       configuration_files=tuple(str(path) for path in files))


def _guest(work_dir):
    """The ``project`` fixture's directory as a container run sees it, in the
    work directory ``work_dir`` mapped to /workzone."""
    return lb.ProjectContext("/workzone/project", lb.Interpreter(path="/usr/bin/python3"),
        execution=lb.ContainerExecution("fixture", str(work_dir), container_work_dir="/workzone"))


_POPEN = subprocess.Popen  # the isolation fixture forbids commands in the tests


def _directory_link(link, target):
    if os.name == "nt":
        command = Path(os.environ["SystemRoot"]) / "System32" / "cmd.exe"
        made = _POPEN([str(command), "/c", "mklink", "/J", str(link), str(target)],
                      stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = made.communicate(timeout=60)
        assert made.returncode == 0, (output + error).decode(errors="replace")
    else:
        link.symlink_to(target, target_is_directory=True)


def test_a_file_read_through_a_linked_dir_is_left_out_and_never_copied(project, tmp_path, monkeypatch):
    """Configuration read through project/conf, a link to a directory outside the
    project: the file is left out at its in-project path, and the copy, which
    never copies links, refuses rather than copying what lies behind it."""
    root, test = project
    shared = tmp_path / "shared-conf"
    shared.mkdir()
    (shared / "config.ini").write_bytes(b"[synthetic]\nkey = synthetic-value\n")
    _directory_link(root / "conf", shared)
    try:
        # RunSettings lists such a file as read and as resolved.
        monkeypatch.setattr(config, "active_settings", lambda: _run_reading(root, root / "conf" / "config.ini"))
        context = lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable))
        assert preparation.project_inputs(context).exclude == ("conf/config.ini",)
        original = lb.create("python", context)
        with pytest.raises(Exception, match="link"):
            with preparation.prepared_measurement(original, lb.Suite(str(test.parent), (test.name,))):
                pass
        assert (shared / "config.ini").read_bytes() == b"[synthetic]\nkey = synthetic-value\n"
    finally:
        if os.name == "nt":
            os.rmdir(root / "conf")
        else:
            (root / "conf").unlink()


def test_a_linked_model_catalogue_is_left_out_of_the_copy(project, tmp_path, monkeypatch):
    root, test = project
    shared = tmp_path / "shared-models.ini"
    shared.write_bytes(b"[synthetic]\nkey = synthetic-value\n")
    try:
        (root / "models.ini").symlink_to(shared)
    except OSError:
        pytest.skip("creating a file symbolic link needs a privilege here")
    monkeypatch.setattr(config, "active_settings", lambda: _run_reading(root, root / "models.ini"))
    context = lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable))
    assert preparation.project_inputs(context).exclude == ("models.ini",)
    with preparation.prepared_measurement(lb.create("python", context),
                                          lb.Suite(str(test.parent), (test.name,))) as (backend, _, _):
        copy = Path(backend.context.root)
        assert (copy / "subject.py").is_file() and not (copy / "models.ini").exists()
    assert (root / "models.ini").is_symlink() and shared.read_bytes() == b"[synthetic]\nkey = synthetic-value\n"


class _PosixNamed:
    """The os module named as on POSIX, where the backend's copies compare
    names case-sensitively."""

    name = "posix"

    def __getattr__(self, attribute):
        return getattr(os, attribute)


@pytest.mark.parametrize("written", ["as listed", "in another case"])
def test_configuration_read_in_another_letter_case_stays_out_of_the_copy(
        project, monkeypatch, written):
    """On a case-insensitive file system a file read as models.ini may be
    listed as Models.ini. A copy that compared names as listed would keep it,
    key included. The source directory may also be written in another case."""
    # Only LangAdapter's private copy module compares names, so the test makes it compare as on POSIX.
    from langadapter import _prepared_copy

    root, test = project
    key = b"sk-synthetic-letter-case-key"
    (root / "Models.ini").write_bytes(b"[explicit]\napi_key = " + key + b"\n")
    (root / "conf").mkdir()
    (root / "conf" / "config.ini").write_bytes(b"[main]\nseed = 3\n")
    (root / "conf" / "hints.json").write_bytes(b'{"_note": "' + key + b'"}\n')
    (root / "sub").mkdir()
    (root / "sub" / "models.ini").write_bytes(b"[unread]\nprovider = openai\n")
    # A case-sensitive file system has no Conf/Config.ini at all.
    if not (root / "Conf" / "Config.ini").exists():
        pytest.skip("the file system here is case-sensitive")
    # Read in another letter case than listed, as RunSettings may name them.
    run = _run_reading(root, *(root / name for name in ("models.ini", "Conf/Config.ini", "Conf/Hints.json")))
    monkeypatch.setattr(config, "active_settings", lambda: run)
    monkeypatch.setattr(_prepared_copy, "os", _PosixNamed())
    written_root = root if written == "as listed" else root.parent / root.name.upper()
    context = lb.ProjectContext(str(written_root), lb.Interpreter(path=sys.executable))
    with preparation.prepared_measurement(lb.create("python", context),
                                          lb.Suite(str(test.parent), (test.name,))) as (backend, _, _):
        copy = Path(backend.context.root)
        found = {path.relative_to(copy).as_posix() for path in copy.rglob("*") if path.is_file()}
        keyed = sorted(name for name in found if key in (copy / name).read_bytes())
    assert keyed == []
    assert not {"Models.ini", "conf/config.ini", "conf/hints.json"} & found
    assert {"sub/models.ini", "subject.py"} <= found


def test_a_hard_link_to_the_catalogue_stays_out_of_the_copy(project, monkeypatch):
    """A hard link inside the project to the catalogue has its own name and
    resolves to itself, so only its file identity shows that it is the
    catalogue."""
    root, test = project
    key = b"sk-synthetic-hard-link-key"
    (root / "models.ini").write_bytes(b"[explicit]\napi_key = " + key + b"\n")
    (root / "extra").mkdir()
    try:
        os.link(root / "models.ini", root / "extra" / "creds.ini")
    except OSError:
        pytest.skip("this file system cannot make a hard link")
    (root / "extra" / "notes.txt").write_bytes(b"kept\n")
    monkeypatch.setattr(config, "active_settings", lambda: _run_reading(root, root / "models.ini"))
    context = lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable))
    selection = preparation.project_inputs(context)
    assert selection.excludes("extra/creds.ini") and selection.excludes("models.ini")
    with preparation.prepared_measurement(lb.create("python", context),
                                          lb.Suite(str(test.parent), (test.name,))) as (backend, _, _):
        copy = Path(backend.context.root)
        found = {path.relative_to(copy).as_posix() for path in copy.rglob("*") if path.is_file()}
        keyed = sorted(name for name in found if key in (copy / name).read_bytes())
    assert keyed == [] and "extra/creds.ini" not in found
    assert {"extra/notes.txt", "subject.py"} <= found


@pytest.mark.skipif(os.name == "nt", reason="a POSIX virtual environment links its interpreter")
def test_a_linked_project_environment_no_longer_refuses_the_copy(project):
    root, test = project
    interpreter = _project_venv(root, link=True)
    original = lb.create("python", lb.ProjectContext(str(root), lb.Interpreter(path=str(interpreter))))
    with preparation.prepared_measurement(original, lb.Suite(str(test.parent), (test.name,))) as (
            backend, staged, targets):
        assert not (Path(backend.context.root) / ".venv").exists()


@pytest.mark.parametrize("relative", [False, True])
def test_suite_coverage_request_is_owned_and_genuine_zero_stays_zero(project, monkeypatch, tmp_path, relative):
    root, test = project
    calls = []

    def observed(self, request):
        request.validate()
        copy = Path(self.context.root)
        # The run's context is a prepared copy of the project.
        assert Path(original_text(self.context, str(copy))) == root
        assert request.coverage_target == request.module_under_test == str(copy / "subject.py")
        assert request.remove_created_files and request.record_leaks
        assert request.timeout_s == 17 and request.runner_options.stop_at_first_failure
        assert request.runner_options.traceback == "line"
        calls.append((copy, request))
        coverage = CoverageObservation(FileIdentity.of(copy / "subject.py", copy),
                                       (1,), (), (), (), traced_lines=())
        (copy / "subject.py").write_bytes(b"changed\n")
        return SuiteRun(lb.Outcome.COMPLETED, "", RunCounts(collected=1, passed=1, failed=0, errors=0),
                        (), (), 0.25, "native output", coverage, identity(request.suite), returncode=0, leaks=())

    monkeypatch.setattr(ProjectPythonBackend, "run_suite", observed)
    output = tmp_path / "coverage.json"
    selected = "subject.py" if relative else str(root / "subject.py")
    language = GenericBackend("python", (".py",))
    for _ in range(2):
        result = language.run_test_files([str(test)], str(root), timeout=17,
            measure_coverage_of=selected, coverage_matrix_path=str(output),
            record_leaks=True, fail_fast=True, tb="line")
        assert result.ok and result.returncode == 0 and result.line_coverage == 0.0
        assert result.output == "native output" and result.covered_lines == ()
        payload = json.loads(output.read_text())
        assert payload["summary"]["covered_lines"] == 0 and payload["target_file"] == selected
    assert calls[0][0] != calls[1][0] and all(not copy.exists() for copy, _ in calls)
    assert (root / "subject.py").read_bytes() == b"value = 1\n" and output.is_file()


@pytest.mark.parametrize("warmup", [False, True])
def test_per_test_evidence_keeps_native_warmup_and_missing_times(project, monkeypatch, warmup):
    root, test = project
    calls = []

    def evidence(self, request):
        request.validate()
        assert request.warmup is warmup and request.timeout_s == 19
        assert request.target == request.module_under_test == str(Path(self.context.root) / "subject.py")
        assert request.remove_created_files and request.isolation_config
        calls.append(Path(self.context.root))
        return PerTestEvidence(lb.Outcome.COMPLETED, "", (), (), (), "not_requested", (),
                               None, None, identity(request.suite))

    monkeypatch.setattr(ProjectPythonBackend, "per_test_evidence", evidence)
    result = GenericBackend("python", (".py",)).collect_eviction_data([str(test)], str(root), measure_coverage_of="subject.py",
                                                 timeout=19, warmup=warmup)
    assert result.per_test_tokens == {} and result.per_test_time_ms == {}
    assert result.suite_time_ms is None and not result.includes_kills
    assert len(calls) == 1 and not calls[0].exists()


def test_delivered_replacements_seed_and_order_survive_owned_staging(project, monkeypatch):
    root, test = project
    calls = []
    replacement = "def test_subject(): assert False\n"

    def delivered(self, request, *, sources):
        request.validate()
        assert request.session_seed == 7 and request.reverse_order
        assert request.module_under_test is None
        assert request.remove_created_files and request.isolation_config
        assert sources == {test.name: replacement}
        calls.append(Path(self.context.root))
        return SuiteRun(lb.Outcome.TEST_FAILURES, "native red", RunCounts(failed=1),
            (TestOutcome(TestId(test.name, "test_subject"), "failed"),), (), 0.2, "native diagnostic",
            None, identity(request.suite), returncode=1)

    monkeypatch.setattr(ProjectPythonBackend, "run_as_delivered", delivered)
    result = GenericBackend("python", (".py",)).run_as_delivered([str(test)], str(root), timeout=23, reverse=True, seed=7,
                                            sources={str(test): replacement})
    assert result.failing == ("test_subject.test_subject",) and result.reason is None
    assert result.output == "native diagnostic" and result.reverse and result.seed == 7
    assert len(calls) == 1 and not calls[0].exists()
    assert test.read_bytes() == b"def test_subject(): pass\n"


@pytest.mark.parametrize("lines, reason", [
    (3, "no verdict: two | three | four"),
    (1, "no verdict: four"),
    (None, "no verdict: one | two | three | four")])
def test_a_delivered_run_without_a_verdict_shows_its_detail_to_the_limit(
        project, monkeypatch, lines, reason):
    """The reason's detail lines follow the run_reason_lines limit (None for
    all of them)."""
    from testuary.agent import _limits

    root, test = project

    def delivered(self, request, *, sources):
        return SuiteRun(lb.Outcome.INFRASTRUCTURE_FAILURE, "no verdict", RunCounts(), (), (),
                        None, "output", None, identity(request.suite),
                        reason_detail=("one", "two", "three", "four"))

    monkeypatch.setattr(ProjectPythonBackend, "run_as_delivered", delivered)
    monkeypatch.setattr(_limits, "value", lambda name: {"run_reason_lines": lines}[name])
    result = GenericBackend("python", (".py",)).run_as_delivered([str(test)], str(root), timeout=23)
    assert result.failing is None and result.reason == reason


class WithoutNotes:
    def __getattribute__(self, name):
        if name == "add_note":
            raise AttributeError(name)
        return super().__getattribute__(name)


class OlderOSError(WithoutNotes, OSError):
    pass


class OlderInterrupt(WithoutNotes, KeyboardInterrupt):
    pass


@pytest.mark.parametrize("error_type", [OSError, KeyboardInterrupt, OlderOSError, OlderInterrupt])
def test_uncertain_close_keeps_the_original_measurement_error_and_owner(project, monkeypatch, error_type):
    root, test = project
    original = lb.create("python", lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable)))
    owner = lb.prepare_project(original.context)
    monkeypatch.setattr(preparation, "prepare_project", lambda context, **options: owner)
    real_close = owner.close
    error = error_type("native measurement failed")
    try:
        with monkeypatch.context() as patch:
            patch.setattr(owner, "close", lambda: (_ for _ in ()).throw(OSError("uncertain disposal")))
            with pytest.raises(error_type) as caught:
                with preparation.prepared_measurement(original, lb.Suite(str(test.parent), (test.name,))):
                    raise error
        assert caught.value is error and owner in lb.recovery_owners(error)
        assert isinstance(error.preparation_cleanup_error, OSError)
        assert error.preparation_cleanup_error.prepared_project is owner
        if hasattr(error, "add_note"):
            assert any("uncertain disposal" in note for note in error.__notes__)
        assert Path(owner.root).is_dir()
    finally:
        real_close()


@pytest.mark.parametrize("execution_error", [False, True])
def test_failure_listing_does_not_discard_an_uncertain_owner(project, monkeypatch, execution_error):
    root, test = project
    original = lb.create("python", lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable)))
    owner = lb.prepare_project(original.context)
    monkeypatch.setattr(preparation, "prepare_project", lambda context, **options: owner)
    native_error = OSError("native observation failed")
    cleanup_error = OSError("uncertain disposal")

    def observed(self, request):
        if execution_error:
            raise native_error
        return SuiteRun(lb.Outcome.COMPLETED, "", RunCounts(collected=1, passed=1),
                        (), (), 0.1, "native output", None, identity(request.suite))

    monkeypatch.setattr(ProjectPythonBackend, "run_suite", observed)
    real_close = owner.close
    try:
        with monkeypatch.context() as patch:
            patch.setattr(owner, "close", lambda: (_ for _ in ()).throw(cleanup_error))
            with pytest.raises(OSError) as caught:
                GenericBackend("python", (".py",)).failing_test_ids([str(test)], str(root), timeout=1)
        assert caught.value is (native_error if execution_error else cleanup_error)
        # A native error keeps LangAdapter's owner, a cleanup error this layer's marker.
        assert owner in (*lb.recovery_owners(caught.value), getattr(caught.value, "prepared_project", None))
        assert caught.value.preparation_cleanup_error is cleanup_error
        assert Path(owner.root).is_dir()
    finally:
        real_close()


def test_failure_listing_preserves_actual_post_allocation_factory_fault(project, monkeypatch):
    root, test = project
    error = OSError("context marker cannot be written")
    original_open = Path.open

    def fail_marker(path, *args, **kwargs):
        if path.name == ".langadapter-project-context":
            raise error
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(ProjectPythonBackend, "run_suite", lambda *a, **k: pytest.fail(
        "retained preparation failure must not start an observation"))
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_marker)
        with pytest.raises(OSError) as caught:
            GenericBackend("python", (".py",)).failing_test_ids([str(test)], str(root), timeout=1)
    assert caught.value is error and error.preparation_setup_error is error
    # The failed preparation's owner keeps the whole copy until it is closed.
    (owner,) = [item for item in lb.recovery_owners(error) if isinstance(item, lb.PreparedProject)]
    copied = Path(owner.root, "subject.py")
    assert Path(owner.root) != root and copied.read_bytes() == b"value = 1\n"
    owner.close()
    assert not copied.exists()
    assert (root / "subject.py").read_bytes() == b"value = 1\n"


def test_failure_listing_keeps_owner_when_only_first_directory_was_allocated(project, monkeypatch):
    # No public call fails between the copy's allocation and its snapshot's, so the fault patches the private one.
    from langadapter._owned_directory import OwnedDirectory

    root, test = project
    error = OSError("input snapshot allocation failed")
    created = OwnedDirectory.create.__func__
    allocated = []

    def allocate(cls, **kwargs):
        if kwargs.get("prefix") == "langadapter_inputs_":
            raise error
        allocated.append(created(cls, **kwargs))
        return allocated[-1]

    monkeypatch.setattr(ProjectPythonBackend, "run_suite", lambda *a, **k: pytest.fail(
        "partial preparation must not start an observation"))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(OwnedDirectory, "create", classmethod(allocate))
            with pytest.raises(OSError) as caught:
                GenericBackend("python", (".py",)).failing_test_ids([str(test)], str(root), timeout=1)
        assert caught.value is error and error.preparation_setup_error is error
        # One owner, the failed preparation's: no copy was completed.
        (recovery,) = lb.recovery_owners(error)
        assert not isinstance(recovery, lb.PreparedProject)
        (owned,) = allocated
        owned.verify()
        assert owned.receipt.is_file() and owned.root.is_dir()
        assert (root / "subject.py").read_bytes() == b"value = 1\n"
    finally:
        for recovery in lb.recovery_owners(error):
            recovery.close()
            recovery.close()
    assert not owned.root.exists() and not owned.receipt.exists()


def test_failure_listing_keeps_unowned_setup_fallback(project, monkeypatch):
    root, test = project
    error = OSError("unowned setup failure")
    monkeypatch.setattr(preparation, "prepare_project", lambda *a, **k: (_ for _ in ()).throw(error))
    assert GenericBackend("python", (".py",)).failing_test_ids([str(test)], str(root), timeout=1) is None
    assert not hasattr(error, "preparation_setup_error")


def test_failure_listing_keeps_cleaned_observation_fallback(project, monkeypatch):
    root, test = project
    error = OSError("native listing failed, disposal confirmed")
    owners = []
    prepare = preparation.prepare_project

    def prepared(context, **options):
        owners.append(prepare(context, **options))
        return owners[-1]

    monkeypatch.setattr(preparation, "prepare_project", prepared)
    monkeypatch.setattr(ProjectPythonBackend, "run_suite", lambda *a, **k: (_ for _ in ()).throw(error))
    assert GenericBackend("python", (".py",)).failing_test_ids([str(test)], str(root), timeout=1) is None
    # The copy that ran is removed.
    (owner,) = owners
    assert not Path(owner.root).exists()


def test_guest_host_suite_mapping_is_typed_and_does_not_rewrite_external_paths(tmp_path):
    zone = tmp_path / "zone"
    zone.mkdir()
    execution = lb.ContainerExecution("borrowed", str(zone), container_work_dir="/workzone",
                                     helper_python="/control/python3")
    original = lb.ProjectContext("/workzone/source", lb.Interpreter(path="/tools/python"), execution=execution)
    scoped = lb.ProjectContext("/workzone/copy/project", original.interpreter, execution=execution)
    suite = lb.Suite(str(zone / "source" / "tests"), ("test_example.py",), ("conftest.py",))
    mapped = preparation._suite(suite, original, scoped)
    assert Path(mapped.root) == zone / "copy" / "project" / "tests"
    assert mapped.test_files == (str(zone / "copy" / "project" / "tests" / "test_example.py"),)
    assert preparation._project_path("/workzone/source/target.py", original, scoped) == "/workzone/copy/project/target.py"
    assert preparation._project_path("/external/target.py", original, scoped) == "/external/target.py"


@pytest.mark.parametrize("name", ["subject.py", "Subject.java"])
def test_guest_typed_host_target_maps_through_only_the_declared_project_mount(tmp_path, name):
    zone = tmp_path / "zone"
    source = zone / "source"
    source.mkdir(parents=True)
    target = source / "nested" / name
    execution = lb.ContainerExecution("borrowed", str(zone), container_work_dir="/workzone",
                                     helper_python="/control/python3")
    original = lb.ProjectContext("/workzone/source", lb.Interpreter(path="/tools/runtime"), execution=execution)
    scoped = lb.ProjectContext("/workzone/copy/project", original.interpreter, execution=execution)
    assert preparation._project_path(str(target), original, scoped) == "/workzone/copy/project/nested/" + name
    external = tmp_path / "external" / name
    assert preparation._project_path(str(external), original, scoped) == str(external)
    assert preparation._project_path("/workzone/source/nested/" + name, original, scoped) == \
        "/workzone/copy/project/nested/" + name


@pytest.mark.parametrize("path", ["C:/external/Subject.java", r"C:\external\subject.py"])
def test_unmapped_windows_target_is_not_invented_as_a_relative_guest_file(tmp_path, path):
    execution = lb.ContainerExecution("borrowed", str(tmp_path), container_work_dir="/workzone",
                                     helper_python="/control/python3")
    original = lb.ProjectContext("/workzone/source", lb.Interpreter(path="/tools/runtime"), execution=execution)
    scoped = lb.ProjectContext("/workzone/copy/project", original.interpreter, execution=execution)
    assert preparation._project_path(path, original, scoped) == path


@pytest.mark.parametrize("project, suite_parts, copied", [
    ("/home/user/xarray", ("tests",), True),
    ("/workzone/proj", ("tests",), True),
    ("/workzone/proj", ("proj", "tests"), False),
    ("/home/user/xarray", None, False),
], ids=["project-outside-the-zone", "project-in-the-zone", "suite-in-the-project",
        "suite-outside-the-zone"])
def test_a_container_suite_in_the_work_dir_is_copied_beside_the_project(
        tmp_path, project, suite_parts, copied):
    """A suite in the work directory is a companion of a container project also
    when the project lies outside the zone (as in the xarray images), so the
    suite is copied and a run prints the same paths as an in-place run."""
    from types import SimpleNamespace

    zone = tmp_path / "zone"
    suite_dir = zone.joinpath(*suite_parts) if suite_parts else tmp_path / "elsewhere" / "tests"
    suite_dir.mkdir(parents=True)
    (suite_dir / "test_x.py").write_bytes(b"def test_x(): pass\n")
    context = lb.ProjectContext(project, lb.Interpreter(path="/venv/bin/python"),
                                execution=lb.ContainerExecution(
                                    "fixture", str(zone), container_work_dir="/workzone"))
    backend = SimpleNamespace(language="python", context=context)
    options = preparation._companions(backend, lb.Suite(str(suite_dir), ("test_x.py",)))
    assert options == ({"companions": (str(suite_dir.resolve()),)} if copied else {})


@pytest.mark.parametrize("coverage, matrix, wall, expected", [
    (False, False, 0.25, 0.25), (True, False, 0.25, 0.75), (True, True, 0.25, 0.875),
    (False, False, None, None)])
def test_a_run_is_charged_its_own_wall_time_and_the_copy_preparation(
        project, monkeypatch, tmp_path, coverage, matrix, wall, expected):
    """time_test_s and time_coverage_s keep what a run in place measures, and
    the copy, its bytecode and its removal are charged to a preparation part of
    their own. A run's time covers the runner, coverage.py's JSON report and,
    for a coverage matrix, the statements read from the data."""
    import time

    from testuary import phase_state

    root, test = project
    target_dir = tmp_path / "target_number"
    target_dir.mkdir()

    def observed(self, request):
        time.sleep(0.05)
        return SuiteRun(lb.Outcome.COMPLETED, "", RunCounts(collected=1, passed=1, failed=0, errors=0),
                        (), (), wall, "", None, identity(request.suite), returncode=0,
                        coverage_report_s=0.5 if coverage else None,
                        coverage_statements_s=0.125 if coverage else None)

    monkeypatch.setattr(ProjectPythonBackend, "run_suite", observed)
    with phase_state.timed_segment(target_dir, "coverage"):
        GenericBackend("python", (".py",)).run_test_files(
            [str(test)], str(root), timeout=17,
            measure_coverage_of="subject.py" if coverage else None,
            coverage_matrix_path=str(tmp_path / "matrix.json") if matrix else None)

    parts = phase_state.output_time_parts(target_dir, "coverage")
    charged = parts.pop("coverage" if coverage else "test")
    assert charged == expected if wall is not None else charged >= 0.05
    assert set(parts) == {"preparation"}


@pytest.mark.parametrize("wall, expected", [(0.25, 0.25), (None, None)])
def test_a_run_as_delivered_is_charged_its_own_wall_time(project, monkeypatch, tmp_path,
                                                          wall, expected):
    """The run as delivered, Python's normal run, is test time as a test
    run is: time_test_s keeps the runner's own wall time, and the copy is
    charged to the preparation part."""
    import time

    from testuary import phase_state

    root, test = project
    target_dir = tmp_path / "target_number"
    target_dir.mkdir()

    def observed(self, request, *, sources=None):
        time.sleep(0.05)
        return SuiteRun(lb.Outcome.COMPLETED, "", RunCounts(collected=1, passed=1, failed=0, errors=0),
                        (), (), wall, "", None, identity(request.suite), returncode=0)

    monkeypatch.setattr(ProjectPythonBackend, "run_as_delivered", observed)
    with phase_state.timed_segment(target_dir, "coverage"):
        GenericBackend("python", (".py",)).run_as_delivered(
            [str(test)], str(root), timeout=17, seed=1)

    parts = phase_state.output_time_parts(target_dir, "coverage")
    charged = parts.pop("test")
    assert charged == expected if wall is not None else charged >= 0.05
    assert set(parts) == {"preparation"}


def test_a_measurement_inside_an_open_span_charges_no_preparation(project, tmp_path):
    """The mutation sessions' span holds their whole time, the copy
    included, so the copy is not charged a second time."""
    from testuary import phase_state

    root, test = project
    target_dir = tmp_path / "target_number"
    target_dir.mkdir()
    with phase_state.timed_segment(target_dir, "mutation"):
        with phase_state.timed_part(phase_state.TIME_MUTATION):
            with preparation.prepared_measurement(
                    lb.create("python", lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable))),
                    lb.Suite(str(test.parent), (test.name,))):
                pass

    assert set(phase_state.output_time_parts(target_dir, "mutation")) == {"mutation"}
