"""Actual mutation tools borrow the pooled execution scope on their thread."""
from asyncio import CancelledError as AsyncCancelledError
from concurrent.futures import CancelledError as FutureCancelledError, ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import inspect
import subprocess
import threading
from types import SimpleNamespace

import pytest
import langadapter as lb
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool as lc_tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.errors import NodeCancelledError
from langgraph.prebuilt import ToolNode
from langadapter.runs import RunCounts, SuiteRun

import test_prepared_mutation_sessions as prepared_sessions
import test_model_surface_snapshot as surface
from test_test_mutation_tools import _make_agent, _trr
from agent_fixtures import passing_delivery
from agent_prompts import python_prompts

from testuary.agent import _langgraph as lg
from testuary.languages import preparation
from testuary.agent import mutation_agent as tm

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

prepared_project = prepared_sessions.project


@pytest.fixture(autouse=True)
def no_processes(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("Mutation tool worker checks start no native process")

    monkeypatch.setattr(subprocess, "Popen", refuse)
    monkeypatch.setattr(subprocess, "run", refuse)
    # The bytecode store's helper is a native command, which its own tests run.
    monkeypatch.setattr(preparation, "bytecode_store", lambda backend: {})


def _tools(agent):
    reads, writes, run, measure = agent.phase_tools()
    return reads + writes + [run, measure]


def _graph(agent):
    builder = StateGraph(MessagesState)
    builder.add_node("tools", ToolNode(_tools(agent), handle_tool_errors=False))
    builder.add_edge(START, "tools")
    builder.add_edge("tools", END)
    return builder.compile()


def _dispatch(graph, name, arguments=None):
    result = graph.invoke(
        {"messages": [AIMessage(content="", tool_calls=[{
            "name": name, "args": arguments or {}, "id": name,
            "type": "tool_call",
        }])]},
        config={"max_concurrency": 1},
    )
    return result["messages"][-1].content


def _pooled_agent(project, session, monkeypatch, *, fake_green=True):
    runner, target, test, natives, _, _, _, _ = project
    agent = tm.MutationAgent(
        prompts=python_prompts("test_mutation"),
        deps=tm.MutationTestDeps(
            target_file_path=str(target), target_file_content=target.read_text(),
            source_dir=str(target.parent), test_output_dir=str(test.parent),
            language=runner.backend.language, mutation_test_file_name=test.name,
            coverage_baseline_pct=80.0,
        ),
        session=tm.TestMutationSession(), mutation_session=session,
    )
    checks = []

    def borrowed(kind):
        assert session._worker_thread == threading.get_ident()
        checks.append((kind, threading.get_ident()))

    thinking = agent.log_thinking

    def observed_thinking(name):
        borrowed(name)
        thinking(name)

    def green(*args, **kwargs):
        borrowed("green")
        return _trr(out="1 passed")

    def delivered(*args, **kwargs):
        borrowed("delivered")
        return passing_delivery(*args, **kwargs)

    # Patch the cached adapter's class: an instance patch outlives the test.
    backend_type = type(tm.languages.get_backend(agent.deps.language))
    if fake_green:
        monkeypatch.setattr(backend_type, "run_test_files", lambda self, *a, **kw: green(*a, **kw))
    monkeypatch.setattr(backend_type, "run_as_delivered", lambda self, *a, **kw: delivered(*a, **kw))
    monkeypatch.setattr(agent, "log_thinking", observed_thinking)
    return agent, checks


@pytest.mark.parametrize("prepared_project", ["python", "java"], indirect=True)
@pytest.mark.parametrize("name", ["read_source", "read_tests", "edit_tests", "run_tests", "measure_mutation"])
def test_real_tool_body_borrows_before_reads_or_validation(prepared_project, monkeypatch, name):
    runner, target, test, natives, _, events, _, _ = prepared_project
    opening_thread = threading.get_ident()
    with runner.session(2, suite_dirs=(test.parent,)) as session:
        agent, checks = _pooled_agent(prepared_project, session, monkeypatch)
        # The empty-match edit refusal exercises the real shared edit gate
        # without needing a parser for the controlled native fixture's bytes.
        arguments = {"old_text": "", "new_text": ""} if name == "edit_tests" else {}
        result = _dispatch(_graph(agent), name, arguments)
        assert isinstance(result, str) and result
        assert checks[0][0] == name
        assert all(thread != opening_thread for _, thread in checks)
        assert session._worker_thread is None and not session._closed
        assert len(natives) == 1 and natives[0].closes == 0
        if name in {"run_tests", "measure_mutation"}:
            kinds = [kind for kind, _ in checks]
            # The normal run decides, once: Python's is its run as
            # delivered, Java's run_test_files, and no run as delivered
            # judges the file again.
            if agent.deps.language == "python":
                assert kinds.count("delivered") == 1
            else:
                assert "green" in kinds and "delivered" not in kinds
        if name == "measure_mutation":
            assert len(natives[0].requests) == 1
            assert agent.session.engine_runs == 1
            assert agent.session.best_mutation_score is None
    assert events == ["open", "close"] and session._closed
    assert target.read_bytes() == b"original\n" and test.read_bytes() == b"delivered\n"


@pytest.mark.parametrize("prepared_project", ["python", "java"], indirect=True)
def test_repeated_measurements_reuse_the_same_owner_and_nested_run(prepared_project, monkeypatch):
    runner, _, _, natives, _, events, _, _ = prepared_project
    with runner.session(2, suite_dirs=(prepared_project[2].parent,)) as session:
        agent, checks = _pooled_agent(prepared_project, session, monkeypatch)
        owner = session._prepared.owner
        root = Path(owner.root)
        graph = _graph(agent)
        first = _dispatch(graph, "measure_mutation")
        agent.session.tests_changed_since_mutation = True
        second = _dispatch(graph, "measure_mutation")
        assert second == first
        assert session._prepared.owner is owner
        assert root.is_dir()
        with pytest.raises(lb.OwnershipRefused), owner.use():
            pass
        assert len(natives) == 1 and len(natives[0].requests) == 2
        assert agent.session.engine_runs == 2
        assert [kind for kind, _ in checks].count("measure_mutation") == 2
        assert session._worker_thread is None
    assert not root.exists() and events == ["open", "close"]


@pytest.mark.parametrize("prepared_project", ["python", "java"], indirect=True)
def test_real_measure_tool_nests_independent_prepared_green_observation(prepared_project, monkeypatch):
    runner, target, _, natives, _, _, _, native_type = prepared_project
    copies = []
    with runner.session(2, suite_dirs=(prepared_project[2].parent,)) as session:
        # The real green observation runs, and the coverage tool's run of the
        # suite after it. Only their native suite runs are observed: Python's
        # green observation is its run as delivered, faked here.
        agent, _ = _pooled_agent(prepared_project, session, monkeypatch, fake_green=False)
        adapter = tm.languages.get_backend(agent.deps.language)
        monkeypatch.setattr(type(adapter), "project_backend", lambda *args: runner.backend)

        def observed(native, request):
            request.validate()
            copy = Path(native.context.root)
            assert copy != Path(session._prepared.owner.root) and copy != target.parent
            assert session._worker_thread == threading.get_ident()
            copies.append(copy)
            return SuiteRun(
                lb.Outcome.COMPLETED, "", RunCounts(collected=1, passed=1, failed=0, errors=0),
                (), (), 0.1, "1 passed", None,
                lb.ObservationIdentity(None, request.suite.identity(),
                                       lb.ToolchainIdentity(native.language, "fixture"), None,
                                       lb.ExecutionIdentity("local")),
                returncode=0,
            )

        monkeypatch.setattr(native_type, "run_suite", observed)
        assert _dispatch(_graph(agent), "measure_mutation")
        assert len(copies) == (1 if agent.deps.language == "python" else 2)
        assert not any(copy.exists() for copy in copies)
        assert len(natives[0].requests) == 1 and session._worker_thread is None
        assert agent.session.engine_runs == 1
    assert session._closed and target.read_bytes() == b"original\n"


@pytest.mark.parametrize("prepared_project", ["python", "java"], indirect=True)
@pytest.mark.parametrize("error_type", [RuntimeError, KeyboardInterrupt, AsyncCancelledError, FutureCancelledError])
def test_actual_dispatch_releases_borrow_without_closing_pool_on_failure(
    prepared_project, monkeypatch, error_type,
):
    runner, _, _, natives, _, events, _, _ = prepared_project
    original = error_type("controlled tool interruption")
    with runner.session(2, suite_dirs=(prepared_project[2].parent,)) as session:
        agent, checks = _pooled_agent(prepared_project, session, monkeypatch)
        observed = agent.log_thinking

        def interrupted(name):
            observed(name)
            raise original

        with monkeypatch.context() as patch:
            patch.setattr(agent, "log_thinking", interrupted)
            expected = NodeCancelledError if error_type is AsyncCancelledError else error_type
            with pytest.raises(expected) as caught:
                _dispatch(_graph(agent), "read_tests")
        if error_type is AsyncCancelledError:
            assert caught.value.__cause__ is original
        else:
            assert caught.value is original
        assert session._worker_thread is None and natives[0].closes == 0
        assert checks[0][0] == "read_tests"
        assert _dispatch(_graph(agent), "read_tests")
    assert events == ["open", "close"] and natives[0].closes == 1


@pytest.mark.parametrize("prepared_project", ["python", "java"], indirect=True)
def test_competing_actual_tool_refuses_before_its_body(prepared_project, monkeypatch):
    runner, _, _, natives, _, _, _, _ = prepared_project
    entered, release = threading.Event(), threading.Event()
    with runner.session(2, suite_dirs=(prepared_project[2].parent,)) as session:
        agent, checks = _pooled_agent(prepared_project, session, monkeypatch)
        observed = agent.log_thinking

        def waiting(name):
            observed(name)
            entered.set()
            assert release.wait(5)

        monkeypatch.setattr(agent, "log_thinking", waiting)
        tool = next(tool for tool in _tools(agent) if tool.name == "read_tests")
        with ThreadPoolExecutor(max_workers=1) as executor:
            future = executor.submit(tool.invoke, {})
            try:
                assert entered.wait(5)
                with pytest.raises(RuntimeError, match="Another mutation session operation"):
                    tool.invoke({})
                assert len(checks) == 1 and natives[0].closes == 0
            finally:
                release.set()
            assert future.result(timeout=5)
        assert session._worker_thread is None
    assert session._closed


class _ScriptedModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@pytest.mark.parametrize("prepared_project", ["python", "java"], indirect=True)
def test_one_response_with_several_tool_calls_runs_them_in_call_order(prepared_project, monkeypatch):
    """A mutation-phase response with several tool calls runs every call, one
    after another in call order, each borrowing the session's worker. Calls
    in parallel threads would collide on the borrowed worker and end the
    phase."""
    runner, _, _, natives, _, _, _, _ = prepared_project
    calls = [
        {"name": "read_source", "args": {"start_line": 1, "end_line": 1}, "id": "c0"},
        {"name": "read_source", "args": {"start_line": 1, "end_line": 2}, "id": "c1"},
        {"name": "read_tests", "args": {}, "id": "c2"},
    ]
    with runner.session(2, suite_dirs=(prepared_project[2].parent,)) as session:
        agent, checks = _pooled_agent(prepared_project, session, monkeypatch)
        agent.session.has_read = True
        agent.session.initial_measured = True  # past the start's measurement
        graph = agent.build_graph(
            model=_ScriptedModel(responses=[AIMessage(content="", tool_calls=calls)]),
            usage_tracker=lg.UsageTracker(request_limit=1),
        )
        results = []
        # The request limit of one ends the run at the next model request.
        with pytest.raises(lg.UsageLimitExceeded):
            for update in graph.stream({"messages": [HumanMessage(content="x")]},
                                       stream_mode="updates"):
                for node in update.values():
                    results += [m for m in (node or {}).get("messages", [])
                                if isinstance(m, ToolMessage)]
        assert [m.tool_call_id for m in results] == ["c0", "c1", "c2"]
        assert all(m.status != "error" and "Another mutation session" not in m.content
                   for m in results)
        assert [kind for kind, _ in checks] == ["read_source", "read_source", "read_tests"]
        assert session._worker_thread is None and natives[0].closes == 0


@pytest.mark.parametrize("session_kind", ["absent", "no-worker", "noncallable-worker"])
def test_replay_and_custom_sessions_without_worker_keep_actual_read_behavior(
    tmp_path, session_kind,
):
    session = {
        "absent": None,
        "no-worker": SimpleNamespace(),
        "noncallable-worker": SimpleNamespace(worker=None),
    }[session_kind]
    agent = _make_agent(tmp_path, mutation_session=session)
    agent.target_path().write_text("def test_f(): pass\n", encoding="utf-8")
    assert "def test_f()" in _dispatch(_graph(agent), "read_tests")
    assert agent.mutation_session is session and agent.session.has_read


def test_binding_does_not_open_worker_or_change_tool_schemas(tmp_path):
    calls = []

    @contextmanager
    def worker():
        calls.append("enter")
        try:
            yield None
        finally:
            calls.append("exit")

    agent = _make_agent(tmp_path, mutation_session=SimpleNamespace(worker=worker))
    reads, writes, run, measure = agent.tool_methods()
    # Each description is the docstring as Python 3.13 and later compile it.
    expected = [lc_tool(method, description=inspect.cleandoc(method.__doc__))
                for method in reads + writes + [run, measure]]
    expected[len(reads)].name = agent.edit_tool_name
    bound = _tools(agent)
    assert [(tool.name, tool.description, tool.args_schema.model_json_schema()) for tool in bound] == [
        (tool.name, tool.description, tool.args_schema.model_json_schema()) for tool in expected
    ]
    assert calls == []
    next(tool for tool in bound if tool.name == "read_source").invoke({})
    assert calls == ["enter", "exit"]


@pytest.mark.parametrize("agent_name", ["test_coverage", "test_mutation"])
def test_dispatched_tool_surface_matches_the_saved_model_contract(agent_name, tmp_path, fixtures_dir):
    import json

    agent = surface._BUILDERS[agent_name](tmp_path)
    saved = json.loads((fixtures_dir / surface.FIXTURE_NAME).read_text(encoding="utf-8"))
    assert surface.render_model_surface(agent)["tools"] == saved[agent_name]["tools"]


def test_worker_admission_failure_preserves_exception_and_never_enters_tool(tmp_path, monkeypatch):
    original = RuntimeError("controlled ownership refusal")

    @contextmanager
    def refused():
        raise original
        yield

    agent = _make_agent(tmp_path, mutation_session=SimpleNamespace(worker=refused))
    monkeypatch.setattr(agent, "log_thinking", lambda *args: pytest.fail("No tool body after refused borrow"))
    with pytest.raises(RuntimeError) as caught:
        _dispatch(_graph(agent), "read_source")
    assert caught.value is original
