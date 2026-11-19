"""Prepared copies for native sessions and ToolNode borrowing, without processes."""
from pathlib import Path
import subprocess
import sys
import threading

import pytest

import langadapter as lb
from langadapter.java.adapter import JavaAdapter as NativeJava
from langadapter.mutation import OperatorCatalogue
from langadapter.mutation import MutationRun
from langadapter.mutation import MutationSession as NativeSession
from langadapter.python.adapter import PythonAdapter as NativePython
from langadapter.runs import RunCounts, SuiteRun, SuiteRunRequest
from testuary.languages import mutation, preparation
from testuary.languages.generic import GenericMutationRunner


def PitRunner(backend, config, **options):
    """The generic mutation runner on Java's engine, PIT."""
    return GenericMutationRunner(backend, config, engine="pit", **options)


class Native(NativeSession):
    def __init__(self, context, events, outcome=lb.Outcome.COMPLETED):
        super().__init__(outcome=outcome, reason="controlled setup refusal" if outcome != lb.Outcome.COMPLETED else "")
        self.context, self.events = context, events
        self.requests = []
        self.inputs = []
        self.identities = []
        self.closes = 0
        self.fail_close = False

    def measure(self, request):
        assert Path(request.target).is_relative_to(Path(self.context.root))
        assert Path(request.target).read_bytes() == b"original\n"
        self.requests.append(request)
        self.inputs.append(tuple(path.read_bytes() for path in request.suite.files()))
        self.identities.append(request.suite.identity())
        (Path(self.context.root) / "native-output.txt").write_text("kept across calls")
        return MutationRun.failed(request.target, lb.Outcome.TIMED_OUT, "controlled incomplete observation")

    def release(self):
        self.closes += 1
        if self.fail_close:
            raise OSError("native owner still held")
        self.events.append("close")


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("Prepared mutation fixtures start no native process")
    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    # The bytecode store's helper is a native command, run by its own tests.
    monkeypatch.setattr(preparation, "bytecode_store", lambda backend: {})


@pytest.fixture(params=["python", "java", "pooled-fixture"])
def project(request, tmp_path, monkeypatch):
    class Plugin(lb.LangAdapter):
        language = "pooled-fixture"
        native_mutation_engines = ("cosmic_ray",)
        mutation_session_attempts = NativePython.mutation_session_attempts

        def validate_mutation_session_options(self, engine, **options):
            assert engine == "cosmic_ray"

    language = request.param
    if language == Plugin.language:
        lb.register(language, (".pooled-fixture",), Plugin)
    backend_type = {"python": NativePython, "java": NativeJava, Plugin.language: Plugin}[language]
    root = tmp_path / "source"
    root.mkdir()
    target = root / ("Subject.java" if language == "java" else "subject.py")
    target.write_bytes(b"original\n")
    tests = tmp_path / "delivered"
    tests.mkdir()
    test = tests / ("SubjectTest.java" if language == "java" else "test_subject.py")
    test.write_bytes(b"delivered\n")
    engine = "pit" if language == "java" else "cosmic_ray"
    events, sessions, options, outcomes = [], [], [], []

    def opened(backend, selected, **choices):
        assert selected == engine
        session = Native(backend.context, events, outcomes.pop(0) if outcomes else lb.Outcome.COMPLETED)
        sessions.append(session)
        options.append(choices)
        events.append("open")
        return session

    monkeypatch.setattr(backend_type, "operation_context_issue", lambda *a: "")
    monkeypatch.setattr(backend_type, "execution_setup_issue", lambda *a: "")
    monkeypatch.setattr(backend_type, "open_mutation_session", opened, raising=False)
    monkeypatch.setattr(backend_type, "mutation_operators", lambda *a: OperatorCatalogue(engine, ("op/A",)),
                        raising=False)
    context = lb.ProjectContext(str(root), lb.Interpreter(path=sys.executable))
    backend = lb.create(language, context)
    config = {"cosmic_ray_operators": "", "pit_mutators": "", "cosmic_ray_distributor": "http",
              "per_mutant_timeout_seconds": "7"}
    runner = (PitRunner if language == "java" else mutation.MutationRunner)(backend, config)
    try:
        yield runner, target, test, sessions, options, events, outcomes, backend_type
    finally:
        if language == Plugin.language:
            lb.unregister(language)


def measure(session, target, test):
    return session.run(target=target, test_dir=test.parent, test_files=[test.name],
                       per_mutant_timeout=7, wall_clock_timeout=31, operators=["op/A"])


def test_native_observation_never_executes_against_the_original_shared_root(project, default_run):
    runner, target, test, natives, _, _, _, _ = project
    with runner.session(2, suite_dirs=(test.parent,)) as session:
        measured = measure(session, target, test)
        root = Path(natives[0].context.root)
        assert root != target.parent
        assert not (target.parent / "native-output.txt").exists()
    assert not root.exists() and target.read_bytes() == b"original\n"
    assert measured.kill_matrix()["target_file"] == str(target)


@pytest.mark.parametrize("inside", [False, True])
def test_one_copy_and_session_persist_across_calls_with_durable_results(project, inside, default_run):
    runner, target, test, natives, options, events, _, _ = project
    if inside:
        test = target.parent / test.name
        test.write_bytes(b"delivered\n")
    with runner.session(2, suite_dirs=(test.parent,)) as session:
        owner = session._prepared.owner
        root = Path(owner.root)
        results = [measure(session, target, test), measure(session, target, test)]
        assert len(natives) == 1 and natives[0].is_open
        assert (root / "native-output.txt").read_text() == "kept across calls"
        with pytest.raises(lb.OwnershipRefused), owner.use():
            pass
        for request, result in zip(natives[0].requests, results):
            assert request.target == str(root / target.name)
            # A Python suite outside the project runs from
            # its declared directory's copy beside the project's copy.
            outside = (root.parent / test.parent.name if runner.backend.language == "python"
                       else test.parent)
            assert Path(request.suite.root) == (root if inside else outside)
            assert request.catalogue.operators == ("op/A",) and request.attribution == "first"
            assert request.per_mutant_timeout == 7 and request.wall_clock_limit_s == 31
            assert request.workers is None and not request.fail_on_unresolved
            assert result.kill_matrix()["target_file"] == str(target)
            assert result.run.target_file == request.target
            assert result.run.resolution.method == "measured" and result.run.score() is None
    session.close()
    assert events == ["open", "close"] and natives[0].closes == 1 and session._closed
    assert not root.exists()
    assert target.read_bytes() == b"original\n" and test.read_bytes() == b"delivered\n"
    assert results[0].kill_matrix()["target_file"] == str(target)
    assert options[0]["workers"] == 2


@pytest.mark.parametrize("inside", [False, True])
def test_native_receiver_reads_current_edits_new_tests_and_setup_with_one_pool(project, inside, default_run):
    runner, target, test, natives, _, events, _, _ = project
    if inside:
        suite_dir = target.parent / "tests"
        suite_dir.mkdir()
        test = suite_dir / test.name
        test.write_bytes(b"at open")
    setup = test.parent / "conftest.py"
    with runner.session(2, suite_dirs=(test.parent,)) as session:
        root = Path(session._prepared.context.root)
        target.write_bytes(b"unrelated original source edit")
        for number in range(2):
            test.write_bytes(f"current {number}".encode())
            setup.write_bytes(f"setup {number}".encode())
            measure(session, target, test)
            expected = (f"current {number}".encode(),) + (
                (f"setup {number}".encode(),) if runner.name != "pit" else ())
            assert natives[0].inputs[-1] == expected
            request = natives[0].requests[-1]
            assert natives[0].identities[-1] == lb.Suite(
                str(test.parent), (test.name,), (setup.name,) if runner.name != "pit" else ()).identity()
            assert request.target == str(root / target.name)
            assert (root / target.name).read_bytes() == b"original\n"
            with pytest.raises(lb.OwnershipRefused), session._prepared.owner.use():
                pass
        new = test.with_name("AnotherTest.java" if runner.name == "pit" else "test_new.py")
        new.write_bytes(b"created after opening")
        measure(session, target, new)
        assert natives[0].inputs[-1][0] == b"created after opening"
        setup.unlink()
        measure(session, target, new)
        assert natives[0].inputs[-1] == (b"created after opening",)
        # The removed setup file leaves the staged suite, in the copy or in
        # its companion of a declared outside directory.
        staged = Path(natives[0].requests[-1].suite.root)
        assert not (staged / "conftest.py").exists()
        assert len(natives) == 1 and len(natives[0].requests) == 4
    session.close()
    assert events == ["open", "close"] and natives[0].closes == 1
    assert target.read_bytes() == b"unrelated original source edit"


def test_write_failure_cannot_fall_through_to_stale_native_measurement(project, monkeypatch, default_run):
    runner, target, test, natives, _, events, _, _ = project
    test = target.parent / test.name
    test.write_bytes(b"at open")
    session = runner.open(2, suite_dirs=(test.parent,))
    copied = Path(session._prepared.context.root) / test.name
    written = Path.write_bytes
    primary = OSError("fresh input write failed")
    def write(path, data):
        if path == copied:
            raise primary
        return written(path, data)
    with monkeypatch.context() as patch:
        patch.setattr(Path, "write_bytes", write)
        with pytest.raises(OSError) as caught:
            measure(session, target, test)
    assert caught.value is primary and primary.mutation_cleanup_handle is session
    assert natives[0].requests == [] and Path(session._prepared.owner.root).is_dir()
    with pytest.raises(RuntimeError, match="retried"):
        measure(session, target, test)
    session.close()
    session.close()
    assert session._closed and events == ["open", "close"]


@pytest.mark.parametrize("stage", ["native", "workspace"])
def test_failed_close_retains_original_owner_and_retries_without_reclosing_native(project, monkeypatch, stage, default_run):
    runner, target, test, natives, _, _, _, _ = project
    session = runner.open(2, suite_dirs=(test.parent,))
    owner = session._prepared.owner
    measure(session, target, test)
    if stage == "native":
        natives[0].fail_close = True
    close = lb.PreparedProject.close
    def closed(prepared):
        if stage == "workspace" and prepared is owner:
            raise OSError("copy disposal held")
        return close(prepared)
    with monkeypatch.context() as patch:
        patch.setattr(lb.PreparedProject, "close", closed)
        with pytest.raises(Exception) as caught:
            session.close()
    error = caught.value
    assert error.mutation_cleanup_handle is session and not session._closed
    assert Path(owner.root).is_dir()
    with pytest.raises(RuntimeError, match="retried"):
        measure(session, target, test)
    natives[0].fail_close = False
    error.mutation_cleanup_handle.close()
    assert session._closed
    with pytest.raises(lb.OwnershipRefused, match="closed"), owner.use():
        pass
    assert natives[0].closes == (2 if stage == "native" else 1)
    target.write_bytes(b"later source\n")
    session.close()
    assert target.read_bytes() == b"later source\n"


def test_body_interruption_preserves_primary_through_native_cleanup_failure(project, default_run):
    runner, target, test, natives, _, _, _, _ = project
    original = KeyboardInterrupt("consumer interrupted")
    with pytest.raises(KeyboardInterrupt) as caught:
        with runner.session(2, suite_dirs=(test.parent,)) as session:
            measure(session, target, test)
            natives[0].fail_close = True
            raise original
    assert caught.value is original and original.mutation_cleanup_handle is session
    assert mutation.mutation_cleanup_pending(original)
    natives[0].fail_close = False
    session.close()
    assert session._closed
    with pytest.raises(lb.OwnershipRefused, match="closed"), session._prepared.owner.use():
        pass


def test_clean_setup_refusal_closes_copy_and_keeps_ordinary_fallback(project, default_run):
    runner, _, _, natives, _, events, outcomes, _ = project
    outcomes.append(lb.Outcome.MISSING_SETUP)
    with pytest.raises(mutation.MutationSetupError) as caught:
        runner.open(2, suite_dirs=())
    assert not mutation.mutation_cleanup_pending(caught.value)
    assert events == ["open", "close"] and not Path(natives[0].context.root).exists()


def test_http_fallback_closes_failed_owner_and_reuses_the_same_prepared_copy(project, default_run):
    runner, target, test, natives, options, events, outcomes, _ = project
    if runner.name == "pit":
        pytest.skip("PIT has no HTTP/sequential fallback")
    runner.config["allow_sequential_fallback"] = True
    outcomes.extend((lb.Outcome.INFRASTRUCTURE_FAILURE, lb.Outcome.COMPLETED))
    with runner.session(2, suite_dirs=(test.parent,)) as session:
        assert events == ["open", "close", "open"] and natives[0].closes == 1
        assert natives[0].context is natives[1].context
        assert [selected["distributor"] for selected in options] == ["http", "local"]
        assert [selected["workers"] for selected in options] == [2, 1]
        assert not session.use_http
        measure(session, target, test)
    assert events == ["open", "close", "open", "close"]


def test_failed_setup_close_retains_both_native_and_prepared_recovery(project, monkeypatch, default_run):
    runner, target, _, natives, _, _, outcomes, _ = project
    runner.config["allow_sequential_fallback"] = True
    outcomes.append(lb.Outcome.INFRASTRUCTURE_FAILURE)
    closed = Native.release
    def refused(native):
        native.fail_close = True
        return closed(native)
    with monkeypatch.context() as patch:
        patch.setattr(Native, "release", refused)
        with pytest.raises(Exception) as caught:
            runner.open(2, suite_dirs=())
    error = caught.value
    recovery = error.mutation_cleanup_handle
    assert error.cleanup_handle is natives[0] and recovery._pending_session is natives[0]
    assert len(natives) == 1 and natives[0].closes == 1
    assert Path(recovery.owner.root).is_dir()
    assert Path(natives[0].context.root).is_dir() and target.read_bytes() == b"original\n"
    natives[0].fail_close = False
    recovery.close()
    recovery.close()
    assert natives[0].closes == 2
    with pytest.raises(lb.OwnershipRefused, match="closed"), recovery.owner.use():
        pass


def test_project_backend_creation_fault_retains_public_prepared_owner(project, monkeypatch):
    runner, target, _, natives, _, _, _, _ = project
    error = OSError("controlled backend creation fault")
    with monkeypatch.context() as patch:
        patch.setattr(mutation, "create", lambda *a: (_ for _ in ()).throw(error))
        with pytest.raises(OSError) as caught:
            runner.open(2, suite_dirs=())
    (owner,) = [item for item in lb.recovery_owners(error) if isinstance(item, lb.PreparedProject)]
    assert caught.value is error and mutation.mutation_cleanup_pending(error)
    assert natives == [] and Path(owner.root).is_dir()
    assert target.read_bytes() == b"original\n"
    owner.close()
    owner.close()


def test_unowned_native_open_error_is_cleaned_without_fallback_owner(project, monkeypatch, default_run):
    runner, target, _, natives, _, _, _, backend_type = project
    roots = []
    error = OSError("ordinary unowned native opening failure")
    def refused(backend, *args, **kwargs):
        roots.append(Path(backend.context.root))
        raise error
    monkeypatch.setattr(backend_type, "open_mutation_session", refused)
    with pytest.raises(OSError) as caught:
        runner.open(2, suite_dirs=())
    assert caught.value is error and not mutation.mutation_cleanup_pending(error)
    assert len(roots) == 1 and roots[0] != target.parent and not roots[0].exists()
    assert natives == [] and target.read_bytes() == b"original\n"


def test_close_refusal_inside_borrow_leaves_the_session_closable(project, default_run):
    runner, target, test, _, _, _, _, _ = project
    session = runner.open(2, suite_dirs=(test.parent,))
    with session.worker():
        with pytest.raises(RuntimeError, match="active"):
            session.close()
        with pytest.raises(RuntimeError, match="retried"):
            measure(session, target, test)
    session.close()
    assert session._closed


def test_actual_toolnode_uses_one_explicit_borrow_with_independent_green_child(project, monkeypatch, default_run):
    from langchain_core.messages import AIMessage
    from langchain_core.tools import StructuredTool
    from langgraph.prebuilt import ToolNode
    from langgraph.graph import END, START, MessagesState, StateGraph

    runner, target, test, natives, _, _, _, backend_type = project
    parent_thread = threading.get_ident()
    observed = []
    def green(backend, request):
        assert Path(backend.context.root) != target.parent
        identity = lb.ObservationIdentity(None, request.suite.identity(),
                                          lb.ToolchainIdentity(backend.language, "fixture"), None,
                                          lb.ExecutionIdentity("local"))
        return SuiteRun(lb.Outcome.COMPLETED, "", RunCounts(collected=1, passed=1),
                        (), (), 0.1, "green", None, identity, returncode=0)
    monkeypatch.setattr(backend_type, "run_suite", green, raising=False)
    with runner.session(2, suite_dirs=(test.parent,)) as session:
        root = session._prepared.context.root
        def tool():
            with session.worker():
                assert threading.get_ident() != parent_thread
                with preparation.prepared_measurement(runner.backend, lb.Suite(str(test.parent), (test.name,))) as (
                        child, suite, _):
                    assert child.context.root != root
                    assert child.tests.run_suite(SuiteRunRequest(suite, 1)).outcome == lb.Outcome.COMPLETED
                observed.append(measure(session, target, test))
                return "done"
        node = ToolNode([StructuredTool.from_function(tool, name="measure", description="controlled observation")],
                        handle_tool_errors=False)
        builder = StateGraph(MessagesState)
        builder.add_node("tools", node)
        builder.add_edge(START, "tools")
        builder.add_edge("tools", END)
        graph = builder.compile()
        for number in range(2):
            call = {"type": "tool_call", "name": "measure", "args": {}, "id": str(number)}
            reply = graph.invoke({"messages": [AIMessage(content="", tool_calls=[call])]},
                                 config={"max_concurrency": 1})
            assert reply["messages"][-1].content == "done"
        assert len(natives) == 1 and len(natives[0].requests) == 2
    assert len(observed) == 2 and session._closed


def test_competing_worker_and_wrong_thread_close_refuse_without_abandoning_owner(project, default_run):
    runner, target, test, natives, _, _, _, _ = project
    session = runner.open(2, suite_dirs=(test.parent,))
    entered, release = threading.Event(), threading.Event()
    errors = []
    def worker():
        try:
            with session.worker():
                entered.set()
                assert release.wait(3)
        except BaseException as error:
            errors.append(error)
    thread = threading.Thread(target=worker)
    thread.start()
    try:
        assert entered.wait(3)
        with pytest.raises(RuntimeError, match="active"):
            measure(session, target, test)
        with pytest.raises(lb.OwnershipRefused):
            lb.create("python", session._prepared.context)
    finally:
        release.set()
        thread.join(3)
    assert not thread.is_alive() and errors == [] and natives[0].is_open
    def close():
        try:
            session.close()
        except BaseException as error:
            errors.append(error)
    foreign = threading.Thread(target=close)
    foreign.start()
    foreign.join(3)
    assert not foreign.is_alive() and len(errors) == 1
    assert errors[0].mutation_cleanup_handle is session and natives[0].is_open
    session.close()
    assert session._closed


def test_the_session_copy_is_preparation_and_the_native_session_mutation(
        project, monkeypatch, tmp_path, default_run):
    """The prepared copy around a session (its creation, each run's staged
    suite, its close and removal) is preparation, and the native session's
    open, runs and close are mutation time, as for a session in the project
    itself."""
    import time

    from testuary import phase_state

    runner, target, test, _, _, _, _, _ = project
    target_dir = tmp_path / "target_number"
    target_dir.mkdir()
    create = mutation._PreparedMutationLifetime.__init__

    def slow_copy(self, backend, suite_dirs=()):
        time.sleep(0.3)
        create(self, backend, suite_dirs)

    monkeypatch.setattr(mutation._PreparedMutationLifetime, "__init__", slow_copy)
    with phase_state.timed_segment(target_dir, "mutation"):
        with runner.session(2, suite_dirs=(test.parent,)) as session:
            measure(session, target, test)
    parts = phase_state.output_time_parts(target_dir, "mutation")
    assert set(parts) == {"mutation", "preparation"}
    assert parts["preparation"] >= 0.3 > parts["mutation"]


def test_a_runs_origin_check_is_preparation_not_mutation(project, monkeypatch, tmp_path, default_run):
    """The adapter reports the run that checked where the tests import the
    target from (MutationRun.origin_run_s), and it counts as preparation."""
    import dataclasses
    import time

    from testuary import phase_state

    runner, target, test, _, _, _, _, _ = project
    target_dir = tmp_path / "target_number"
    target_dir.mkdir()
    measured = Native.measure

    def with_origin(self, request):
        time.sleep(0.3)
        return dataclasses.replace(measured(self, request), origin_run_s=0.3)

    monkeypatch.setattr(Native, "measure", with_origin)
    with phase_state.timed_segment(target_dir, "mutation"):
        with runner.session(2, suite_dirs=(test.parent,)) as session:
            measure(session, target, test)
    parts = phase_state.output_time_parts(target_dir, "mutation")
    assert parts["preparation"] >= 0.3 > parts["mutation"]


def test_a_declared_suite_dir_is_staged_beside_the_copy(project, default_run):
    """A Python session copies the suite directory its caller declared beside
    the project's copy and stages each run's suite there, so its runs write
    nothing into the declared directory."""
    runner, target, test, natives, _, _, _, _ = project
    with runner.session(2, suite_dirs=(test.parent,)) as session:
        measure(session, target, test)
        root = Path(natives[0].context.root)
        (request,) = natives[0].requests
        staged = Path(request.suite.root)
    if runner.backend.language == "python":
        assert staged == root.parent / test.parent.name
    else:
        assert staged == test.parent
    assert sorted(path.name for path in test.parent.iterdir()) == [test.name]


def test_an_undeclared_suite_dir_is_refused_with_both_named(project, default_run):
    runner, target, test, natives, _, _, _, _ = project
    other = test.parent.parent / "elsewhere"
    other.mkdir()
    with runner.session(2, suite_dirs=(other,)) as session:
        if runner.backend.language != "python":
            measure(session, target, test)
            return
        with pytest.raises(mutation.UndeclaredSuiteFolder) as refused:
            measure(session, target, test)
    assert str(test.parent.resolve()) in str(refused.value)
    assert str(other.resolve()) in str(refused.value)
    assert natives[0].requests == []


@pytest.mark.parametrize("copies", [True, False], ids=["copies", "in-place"])
def test_the_declaration_follows_the_adapters_fact_not_its_language(project, monkeypatch, copies, default_run):
    """An adapter that copies a suite directory outside the project
    (``copies_external_suite``) refuses one its caller did not declare. One
    that runs such a suite in place needs no declaration, whatever its
    language."""
    for adapter in (lb.LangAdapter, NativePython, NativeJava):
        monkeypatch.setattr(adapter, "copies_external_suite", copies)
    runner, target, test, natives, _, _, _, _ = project
    other = test.parent.parent / "elsewhere"
    other.mkdir()
    with runner.session(2, suite_dirs=(other,)) as session:
        if not copies:
            measure(session, target, test)
            return
        with pytest.raises(mutation.UndeclaredSuiteFolder):
            measure(session, target, test)
    assert natives[0].requests == []


def test_a_suite_inside_the_project_needs_no_declaration(project, default_run):
    runner, target, test, natives, _, _, _, _ = project
    inside = target.parent / test.name
    inside.write_bytes(b"delivered\n")
    with runner.session(2, suite_dirs=()) as session:
        measure(session, target, inside)
    (request,) = natives[0].requests
    assert Path(request.suite.root) == Path(natives[0].context.root)


def test_a_run_of_its_own_declares_its_suite_dir(project, default_run):
    runner, target, test, natives, _, _, _, _ = project
    runner.run_once(workers=2, target=target, test_dir=test.parent, test_files=[test.name],
                    per_mutant_timeout=7, wall_clock_timeout=31, operators=["op/A"])
    assert len(natives[0].requests) == 1


def test_every_session_opener_declares_its_suite_dirs():
    """Every production call that opens a mutation session names the directories
    its suites lie in. The list of call sites is fixed, so adding an opener
    means changing it."""
    import ast

    import testuary

    package = Path(testuary.__file__).parent
    openers, undeclared = set(), []
    for path in sorted(package.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        functions = [node for node in ast.walk(tree)
                     if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
        for function in functions:
            for node in ast.walk(function):
                if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                        and node.func.attr in ("open", "session") and node.args):
                    continue
                if "worker" not in ast.unparse(node.args[0]):
                    continue
                site = (path.relative_to(package).as_posix(), function.name)
                openers.add(site)
                if not any(keyword.arg == "suite_dirs" for keyword in node.keywords):
                    undeclared.append(site)
    assert undeclared == []
    assert openers == {
        ("target_phases.py", "_open_mutation_session"),
        ("measurement.py", "union_mutation_runner"),
        ("measurement.py", "run_finishing"),
        ("languages/mutation.py", "session"),
        ("languages/mutation.py", "run_once"),
        # The Java demonstration's runner, which opens a scenario session.
        ("demo/backend.py", "open"),
    }
