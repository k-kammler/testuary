"""A prepared pooled mutation session on LangAdapter's adapters, for Python
with Cosmic Ray and Java with PIT (local and in a guest), stays open across
two worker calls, each with a green and coverage run in a separate prepared
copy."""
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import json
import os
from pathlib import Path, PurePosixPath
import sys
import threading
import time
import traceback
import urllib.request

import pytest

import langadapter as lb
from langadapter.runs import SuiteRunRequest
from testuary.languages import mutation, preparation
from testuary.languages.generic import GenericMutationRunner


def PitRunner(backend, config, **options):
    """The generic mutation runner on Java's engine, PIT."""
    return GenericMutationRunner(backend, config, engine="pit", **options)


pytestmark = [pytest.mark.usefixtures("default_run"), pytest.mark.slow, pytest.mark.skipif(
    os.environ.get("LANGADAPTER_TEST_PREPARED") != "1",
    reason="runs the native prepared sessions only with LANGADAPTER_TEST_PREPARED=1"),
    pytest.mark.mutation_tool]

_SUITE_TIMEOUT_SECONDS = 120
_MUTATION_TIMEOUT_SECONDS = 120
_PER_MUTANT_TIMEOUT_SECONDS = 30
_TOOL_WAIT_SECONDS = 240

_POM = """<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>demo</groupId><artifactId>pooled</artifactId><version>1.0</version>
  <properties><maven.compiler.release>11</maven.compiler.release></properties>
  <dependencies><dependency><groupId>org.junit.jupiter</groupId>
    <artifactId>junit-jupiter</artifactId><version>5.14.4</version>
    <scope>test</scope></dependency></dependencies>
  <build><plugins><plugin><groupId>org.apache.maven.plugins</groupId>
    <artifactId>maven-surefire-plugin</artifactId><version>3.6.0</version>
  </plugin></plugins></build>
</project>
"""
_JAVA = """package demo;
public final class Calc {
    public static int sign(int x) {
        if (x > 0) return 1;
        if (x < 0) return -1;
        return 0;
    }
}
"""
_JAVA_TEST = """package demo;
import static org.junit.jupiter.api.Assertions.assertEquals;
import org.junit.jupiter.api.Test;
class CalcTest {
    @Test void positive() { assertEquals(1, Calc.sign(5)); }
    @Test void negative() { assertEquals(-1, Calc.sign(-5)); }
}
"""


class _ToolJournal:
    """Retain fixture stages even when the outer future hides a worker error."""

    def __init__(self, path):
        self.path = path
        self.started = time.monotonic()
        self.lock = threading.Lock()

    def write(self, event, **fields):
        with self.lock:
            record = {"event": event, "elapsed_s": time.monotonic() - self.started, **fields}
            with self.path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(record, ensure_ascii=True) + "\n")

    @contextmanager
    def stage(self, name, *, request):
        self.write("started", stage=name, request=request)
        try:
            yield
        except BaseException as error:
            self.failure(error, stage=name, request=request)
            raise
        else:
            self.write("completed", stage=name, request=request)

    def failure(self, error, **fields):
        try:
            self.write("failed", error_type=type(error).__name__, error=str(error),
                       traceback="".join(traceback.format_exception(
                           type(error), error, error.__traceback__)),
                       notes=list(getattr(error, "__notes__", ())), **fields)
        except BaseException as secondary:
            error._fixture_journal_error = secondary
            if hasattr(error, "add_note"):
                try:
                    error.add_note(f"Fixture journal failed: {secondary}")
                except BaseException:
                    pass


def _wait_tool(worker, tool, selected_test, expected_passed, *, journal, request):
    journal.write("started", stage="future_wait", request=request,
                  timeout_s=_TOOL_WAIT_SECONDS)
    future = worker.submit(tool, selected_test, expected_passed, request)
    try:
        result = future.result(timeout=_TOOL_WAIT_SECONDS)
    except BaseException as error:
        journal.failure(error, stage="future_wait", request=request, worker_done=future.done())
        raise
    journal.write("completed", stage="future_wait", request=request)
    return result


def _write(root, relative, text):
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")
    return path


def _java_selection(guest):
    if guest:
        names = ("LANGADAPTER_TEST_CONTAINER", "LANGADAPTER_TEST_CONTAINER_JAVA",
                 "LANGADAPTER_TEST_WORK_DIR", "LANGADAPTER_TEST_CONTAINER_HELPER_PYTHON",
                 "LANGADAPTER_TEST_CONTAINER_MAVEN", "LANGADAPTER_TEST_CONTAINER_MAVEN_REPOSITORY",
                 "LANGADAPTER_TEST_TOOL_CACHE")
        values = [os.environ.get(name) for name in names]
        if not all(values):
            pytest.skip("requires scheduled offline guest, mapped work directory and cached JDK/Maven/PIT")
        container, java, zone, helper, maven, repository, cache = values
        execution = lb.ContainerExecution(container, work_dir=zone,
            container_work_dir="/workzone", helper_python=helper)
        return java, execution, {"LANGADAPTER_MAVEN": maven, "LANGADAPTER_JAVA_TOOLS": cache}, repository
    homes = os.environ.get("LANGADAPTER_TEST_JAVA_HOMES", "").split(os.pathsep)
    home = next((Path(item) for item in homes if item.strip()
                 and (Path(item) / "release").is_file()
                 and 'JAVA_VERSION="21' in (Path(item) / "release").read_text()), None)
    maven = os.environ.get("LANGADAPTER_TEST_MAVEN")
    if home is None or not maven:
        pytest.skip("requires scheduled cached JDK21 and Maven")
    java = home / "bin" / ("java.exe" if os.name == "nt" else "java")
    return str(java), lb.LocalExecution(), {"LANGADAPTER_MAVEN": maven}, None


@pytest.mark.parametrize("route", ["python", "java", "java-guest"])
def test_native_pool_survives_two_worker_calls_and_independent_checks(tmp_path, monkeypatch, route):
    def no_download(*args, **kwargs):
        pytest.fail("finite pooled fixture attempted a tool download")
    monkeypatch.setattr(urllib.request, "urlopen", no_download)
    source = tmp_path / "source"
    config = {"cosmic_ray_distributor": "http", "cosmic_ray_operators": "",
              "cosmic_ray_catalogue_cache_dir": "", "cosmic_ray_clone_sweep_hours": 0,
              "cosmic_ray_clone_dir": str(tmp_path / "clones"),
              "cosmic_ray_session_dir": str(tmp_path / "sessions"),
              "cosmic_ray_init_timeout_s": 60,
              "cosmic_ray_worker_ready_timeout_s": 60}
    if route == "python":
        target = _write(source, "arith.py", "def add(a, b):\n    return a + b\n")
        test = _write(source, "tests/test_arith.py",
                      "from arith import add\n\ndef test_add():\n    assert add(2, 3) != 0\n")
        context = lb.ProjectContext(str(source), lb.Interpreter(path=sys.executable))
        runner_type, language = mutation.MutationRunner, "python"
        operators = ["core/ReplaceBinaryOperator_Add_Sub"]
        expected_mutants = 1
    else:
        java, execution, environment, repository = _java_selection(route == "java-guest")
        target = _write(source, "src/main/java/demo/Calc.java", _JAVA)
        test = _write(source, "src/test/java/demo/CalcTest.java", _JAVA_TEST)
        _write(source, "pom.xml", _POM)
        _write(source, ".mvn/maven.config", "-o\n" + (
            f"-Dmaven.repo.local={repository}\n" if repository else ""))
        root = str(source)
        if route == "java-guest":
            root = str(PurePosixPath("/workzone") /
                       PurePosixPath(source.resolve().relative_to(Path(execution.work_dir).resolve()).as_posix()))
        context = lb.ProjectContext(root, lb.Interpreter(path=java),
                                    execution=execution, environment=environment)
        runner_type, language = PitRunner, "java"
        operators = ["org.pitest.mutationtest.engine.gregor.mutators.ConditionalsBoundaryMutator"]
        expected_mutants = 2
    native = lb.create(language, context)
    runner = runner_type(native, config)
    original = target.read_bytes()
    initial_test = test.read_bytes()
    measurements, child_roots = [], []
    journal = _ToolJournal(tmp_path / "tool-events.jsonl")
    journal.write("limits", suite_timeout_s=_SUITE_TIMEOUT_SECONDS,
                  mutation_timeout_s=_MUTATION_TIMEOUT_SECONDS,
                  per_mutant_timeout_s=_PER_MUTANT_TIMEOUT_SECONDS,
                  tool_wait_timeout_s=_TOOL_WAIT_SECONDS)
    with runner.session(2 if route == "python" else 1, suite_dirs=(test.parent,)) as session:
        owner = session._prepared.owner
        root = owner.root
        native_session = session._session

        def tool(selected_test, expected_passed, request):
            with journal.stage("whole_tool", request=request), session.worker():
                setup = tuple(str(path) for path in (
                    selected_test.parent / "conftest.py",) if route == "python" and path.is_file())
                suite = lb.Suite(str(selected_test.parent), (selected_test.name,), setup)
                with journal.stage("independent_check", request=request), preparation.prepared_measurement(
                        native, suite, str(target)) as (child, selected, mapped):
                    assert child.context.root != root
                    child_roots.append(child.context.root)
                    with journal.stage("green_coverage_run", request=request):
                        run = child.tests.run_suite(SuiteRunRequest(selected, _SUITE_TIMEOUT_SECONDS,
                            coverage_target=mapped[0], timeout_journal=True))
                    assert run.outcome is lb.Outcome.COMPLETED, (run.reason, run.output)
                    assert run.counts.passed == expected_passed
                    assert run.coverage is not None and run.coverage.covered
                with journal.stage("mutation_request", request=request):
                    result = session.run(target=target, test_dir=selected_test.parent,
                                         test_files=[selected_test.name],
                                         per_mutant_timeout=_PER_MUTANT_TIMEOUT_SECONDS,
                                         wall_clock_timeout=_MUTATION_TIMEOUT_SECONDS, operators=operators)
                assert result.run.outcome is lb.Outcome.COMPLETED, result.run.reason
                assert result.run.complete and result.run.score() == 100.0
                assert len(result.run.mutants) == expected_mutants
                if route != "python":
                    assert all(item.row["native_status"] == "KILLED" for item in result.run.mutants)
                assert result.run.identity is not None
                assert result.run.identity.observation.suite == tuple(
                    sorted(suite.identity(), key=lambda item: item.path))
                assert result.kill_matrix()["target_file"] == str(target)
                measurements.append(result)
                assert session._session is native_session and native_session.is_open
                with pytest.raises(lb.OwnershipRefused), owner.use():
                    pass

        with ThreadPoolExecutor(max_workers=1) as worker:
            if route == "python":
                test.write_text("from arith import add\n\ndef test_add():\n    assert add(2, 3) == 5\n",
                                encoding="utf-8", newline="\n")
            else:
                test.write_text(_JAVA_TEST.replace(
                    "    @Test void negative()",
                    "    @Test void zero() { assertEquals(0, Calc.sign(0)); }\n    @Test void negative()"),
                    encoding="utf-8", newline="\n")
            assert test.read_bytes() != initial_test
            _wait_tool(worker, tool, test, 1 if route == "python" else 3,
                       journal=journal, request=1)
            if route == "python":
                new = _write(source, "tests/test_new.py",
                    "from arith import add\n\ndef test_new(expected):\n    assert add(2, 3) == expected\n")
                _write(source, "tests/conftest.py",
                    "import pytest\n\n@pytest.fixture\ndef expected():\n    return 5\n")
            else:
                new = _write(source, "src/test/java/demo/NewCalcTest.java",
                    _JAVA_TEST.replace("CalcTest", "NewCalcTest").replace(
                        "assertEquals(1, Calc.sign(5))", "assertEquals(0, Calc.sign(0))").replace(
                        "assertEquals(-1, Calc.sign(-5))", "assertEquals(0, Calc.sign(0))"))
            _wait_tool(worker, tool, new, 1 if route == "python" else 2,
                       journal=journal, request=2)
        assert len(child_roots) == 2 and len(set(child_roots)) == 2
    assert session._closed and not native_session.is_open
    with pytest.raises(lb.OwnershipRefused, match="closed"), owner.use():
        pass
    assert target.read_bytes() == original and test.read_bytes() != initial_test and new.is_file()
    matrix = tmp_path / "durable-matrix.json"
    assert mutation.write_kill_matrix(measurements[-1], matrix)
    payload = json.loads(matrix.read_text())
    assert payload["summary"]["total"] == expected_mutants
    assert payload["target_file"] == str(target)
    assert len(measurements) == 2
    if route == "python":
        assert not list((tmp_path / "clones").iterdir())
