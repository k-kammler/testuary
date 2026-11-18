"""Java observations preserve native requests through owned project copies."""
import json
from pathlib import Path
import subprocess
import sys

import langadapter as lb
from langadapter.outcomes import ExecutionIdentity, FileIdentity, ObservationIdentity, ToolchainIdentity
from langadapter.java.adapter import JavaAdapter as ProjectJavaBackend
from langadapter.runs import CoverageObservation, PerTestEvidence, RunCounts, SuiteRun, TestOutcome
from langadapter.suites import TestId
import pytest

from testuary import config
from testuary import project as project_module
from testuary.languages import preparation
from testuary.languages.generic import GenericBackend

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture(autouse=True)
def no_native_tools(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("Controlled Java staging must not run tools")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    monkeypatch.setattr(subprocess, "run", forbidden)
    # The bytecode store's helper is a native command, which its own tests run.
    monkeypatch.setattr(preparation, "bytecode_store", lambda backend: {})
    monkeypatch.setattr(config, "project_exec_mode", lambda: "local")
    monkeypatch.setattr(project_module, "project_work_dir", lambda: None)
    # The JDK is this interpreter, selected in the project's settings, with no build tools.
    configured = config.load_config
    monkeypatch.setattr(config, "load_config", lambda section=None: (
        {**configured(section), "java_home": sys.executable, "build_java_home": "",
         "maven": "", "gradle": ""} if section == "project" else configured(section)))


@pytest.fixture
def project(tmp_path):
    root = tmp_path / "project"
    root.mkdir()
    target = root / "Subject.java"
    target.write_bytes(b"public class Subject {}\n")
    delivered = tmp_path / "delivered"
    delivered.mkdir()
    test = delivered / "SubjectTest.java"
    test.write_bytes(b"public class SubjectTest {}\n")
    return root, target, test


def _identity(suite):
    return ObservationIdentity(None, suite.identity(), ToolchainIdentity("java", "fixture"),
                               None, ExecutionIdentity("local"))


@pytest.mark.parametrize("operation", ["run", "matrix", "listing", "delivered", "evidence"])
@pytest.mark.parametrize("inside", [False, True])
def test_java_observations_use_public_copy_and_keep_native_evidence(
        project, monkeypatch, tmp_path, operation, inside):
    root, target, test = project
    if inside:
        test = root / test.name
        test.write_bytes(b"public class SubjectTest {}\n")
    requests, copies = [], []
    coverage = None
    expected_source = target.read_bytes()

    def observed(self, request, **kwargs):
        nonlocal coverage
        request.validate()
        copy = Path(self.context.root)
        assert copy != root and copy.is_dir()
        assert (copy / target.name).read_bytes() == expected_source
        assert Path(request.suite.root) == (copy if inside else test.parent)
        assert Path(request.suite.test_files[0]).read_bytes() == test.read_bytes()
        assert request.timeout_s == 17
        # As for Python, every observation except the coverage matrix removes
        # files that earlier runs created in the copy.
        assert request.remove_created_files is (operation != "matrix")
        requests.append(request)
        copies.append(copy)
        coverage = CoverageObservation(FileIdentity.of(copy / target.name, copy),
                                       (1,), (), (), (), traced_lines=())
        if operation in ("run", "matrix"):
            assert request.coverage_target == str(copy / target.name)
        elif operation == "evidence":
            assert request.target == str(copy / target.name) and not request.warmup
        elif operation == "delivered":
            assert kwargs == {"sources": {test.name: "public class SubjectTest { /* replacement */ }"}}
            assert request.module_under_test is None
        (copy / target.name).write_bytes(b"changed in copy\n")
        if operation == "evidence":
            return PerTestEvidence(outcome=lb.Outcome.COMPLETED, reason="", tests=(),
                                   untimed=(), unmatched_contexts=(), kills="not_requested",
                                   unmapped_killers=(), suite_time_ms=None, coverage=coverage,
                                   identity=_identity(request.suite))
        return SuiteRun(lb.Outcome.TEST_FAILURES if operation in ("listing", "delivered")
                        else lb.Outcome.COMPLETED, "", RunCounts(collected=1, passed=0, failed=1),
                        (TestOutcome(TestId(test.name, "demo.SubjectTest.check"), "failed"),), (),
                        0.2, "native JUnit output", coverage, _identity(request.suite), returncode=1)

    method = {"delivered": "run_as_delivered", "evidence": "per_test_evidence"}.get(operation, "run_suite")
    monkeypatch.setattr(ProjectJavaBackend, method, observed)
    adapter = GenericBackend("java", (".java",))
    matrix = tmp_path / "coverage.json"
    if operation == "run":
        result = adapter.run_test_files([str(test)], str(root), timeout=17,
                                       measure_coverage_of="Subject.java", coverage_matrix_path=str(matrix))
        assert result.line_coverage == 0.0 and result.covered_lines == ()
        assert result.output == "native JUnit output" and result.leaks is None
        assert json.loads(matrix.read_text())["target_file"] == "Subject.java"
    elif operation == "matrix":
        adapter.capture_coverage_matrix(target=target, source_dir=root,
                                        test_paths=[str(test)], out_path=matrix, timeout=17)
        assert json.loads(matrix.read_text())["target_file"] == str(target)
    elif operation == "listing":
        assert adapter.failing_test_ids([str(test)], str(root), timeout=17) == ["demo.SubjectTest.check"]
    elif operation == "delivered":
        result = adapter.run_as_delivered([str(test)], str(root), timeout=17, seed=7,
                 sources={str(test): "public class SubjectTest { /* replacement */ }"})
        assert result.failing == ("demo.SubjectTest.check",) and result.seed == 7 and not result.reverse
        assert result.output == "native JUnit output"
    else:
        result = adapter.collect_eviction_data([str(test)], str(root), timeout=17,
                                              measure_coverage_of=str(target), warmup=False)
        assert result.per_test_time_ms == {} and result.suite_time_ms is None
        assert result.target_covered_lines == 0 and not result.includes_kills
    assert len(requests) == len(copies) == 1 and not copies[0].exists()
    assert target.read_bytes() == expected_source
    assert test.read_bytes() == b"public class SubjectTest {}\n"


@pytest.mark.parametrize("setup", [False, True])
def test_java_listing_keeps_fully_cleaned_failure_fallback(project, monkeypatch, setup):
    root, _, test = project
    error = OSError("ordinary failed listing")
    monkeypatch.setattr(ProjectJavaBackend, "run_suite", lambda *a: (_ for _ in ()).throw(error))
    if setup:
        monkeypatch.setattr(preparation, "prepare_project", lambda *a: (_ for _ in ()).throw(error))
    assert GenericBackend("java", (".java",)).failing_test_ids([str(test)], str(root), timeout=1) is None
    assert not hasattr(error, "preparation_cleanup_error")
    assert not hasattr(error, "preparation_setup_error")


def test_java_listing_preserves_actual_post_allocation_setup_failure(project, monkeypatch):
    root, target, test = project
    error = OSError("context marker cannot be written")
    original_open = Path.open

    def fail_marker(path, *args, **kwargs):
        if path.name == ".langadapter-project-context":
            raise error
        return original_open(path, *args, **kwargs)

    monkeypatch.setattr(ProjectJavaBackend, "run_suite", lambda *args: pytest.fail("no execution after setup failure"))
    with monkeypatch.context() as patch:
        patch.setattr(Path, "open", fail_marker)
        with pytest.raises(OSError) as caught:
            GenericBackend("java", (".java",)).failing_test_ids([str(test)], str(root), timeout=1)
    assert caught.value is error and error.preparation_setup_error is error
    # The failed preparation's owner keeps the whole copy until it is closed.
    (owner,) = [item for item in lb.recovery_owners(error) if isinstance(item, lb.PreparedProject)]
    copied = Path(owner.root, target.name)
    assert copied.read_bytes() == b"public class Subject {}\n"
    owner.close()
    assert not copied.exists()
    assert target.read_bytes() == b"public class Subject {}\n"


def test_java_listing_keeps_partial_factory_owner_without_full_workspace(project, monkeypatch):
    # No public call fails between the copy's allocation and its snapshot's, so the fault patches the private one.
    from langadapter._owned_directory import OwnedDirectory

    root, target, test = project
    error = OSError("input snapshot allocation failed")
    created = OwnedDirectory.create.__func__
    allocated = []

    def allocate(cls, **kwargs):
        if kwargs.get("prefix") == "langadapter_inputs_":
            raise error
        allocated.append(created(cls, **kwargs))
        return allocated[-1]

    monkeypatch.setattr(ProjectJavaBackend, "run_suite", lambda *a, **k: pytest.fail(
        "partial preparation must not start an observation"))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(OwnedDirectory, "create", classmethod(allocate))
            with pytest.raises(OSError) as caught:
                GenericBackend("java", (".java",)).failing_test_ids([str(test)], str(root), timeout=1)
        assert caught.value is error and error.preparation_setup_error is error
        # One owner, the failed preparation's: no workspace was completed.
        (recovery,) = lb.recovery_owners(error)
        assert not isinstance(recovery, lb.PreparedProject)
        (owned,) = allocated
        owned.verify()
        assert owned.receipt.is_file() and owned.root.is_dir()
        assert target.read_bytes() == b"public class Subject {}\n"
    finally:
        for recovery in lb.recovery_owners(error):
            recovery.close()
            recovery.close()
    assert not owned.root.exists() and not owned.receipt.exists()


def test_java_listing_keeps_uncertain_close_primary_and_owner(project, monkeypatch):
    root, _, test = project
    owner = lb.prepare_project(lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable)))
    original_close = owner.close
    native_error, cleanup_error = OSError("native listing failure"), OSError("unresolved disposal")
    monkeypatch.setattr(preparation, "prepare_project", lambda context: owner)
    monkeypatch.setattr(ProjectJavaBackend, "run_suite", lambda *args: (_ for _ in ()).throw(native_error))
    try:
        with monkeypatch.context() as patch:
            patch.setattr(owner, "close", lambda: (_ for _ in ()).throw(cleanup_error))
            with pytest.raises(OSError) as caught:
                GenericBackend("java", (".java",)).failing_test_ids([str(test)], str(root), timeout=1)
        assert caught.value is native_error
        assert owner in lb.recovery_owners(native_error)
        assert native_error.preparation_cleanup_error is cleanup_error
        assert Path(owner.root).is_dir()
    finally:
        original_close()
