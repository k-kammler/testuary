"""Shared pytest fixtures for the testuary test suite.

Tests run against the installed (editable) package: ``pip install -e .``
makes ``import testuary`` work without touching ``sys.path`` here.
Pytest discovers tests under ``tests/`` automatically.
"""
from __future__ import annotations

import contextlib
import importlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path, PureWindowsPath
from typing import Iterator

import pytest

from testuary import config
from testuary import project
from testuary import settings


# The developer's files at the repo root: older working copies of
# config.ini and models.ini, and a project settings file. The suite asserts
# the SHIPPED (bundled) defaults plus each test's own overrides, so a
# .testuary/project.ini there, which every test resolving from the repo
# directory would find, must not leak in. The skip is installed here, at
# conftest import (before the agent modules below build their module-level
# _AGENT_CONFIG and before any test is collected), so every config read in
# the session sees the bundled defaults. A tmp_cwd file with the same name
# still reads, since its resolved path differs.
_REPO_WORKING_FILES = {
    (Path(__file__).resolve().parents[1] / _n).resolve()
    for _n in ("config.ini", "models.ini", ".testuary/project.ini")
}
_real_read_ini = settings._read_ini_file


def _read_ini_skip_repo_working(path):
    try:
        if Path(path).resolve() in _REPO_WORKING_FILES:
            return {}
    except OSError:
        pass
    return _real_read_ini(path)


settings._read_ini_file = _read_ini_skip_repo_working


# Load each agent module once, under the bundled config, at collection
# time. Their module-level setup reads required config keys (retries,
# max_requests_per_phase, ...) that a test's minimal monkeypatched
# get_agent_config would not carry. Importing here keeps that one-time import
# off the per-test path, where a test that monkeypatches get_agent_config and
# then transitively triggers a first import would otherwise hit a KeyError.
for _agent_mod in (
    "testuary.agent.coverage_agent",
    "testuary.agent.mutation_agent",
):
    try:
        importlib.import_module(_agent_mod)
    except Exception:
        pass


@pytest.fixture(autouse=True)
def _isolate_testuary_env(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Remove host Testuary overrides.

    Repository working configuration files are excluded separately by the
    loader patch above. Each test supplies its own execution condition. A
    test that sets or clears a runtime override (a project, an import root)
    would otherwise hand it to every later test.
    """
    stripped = [k for k in os.environ if k.startswith("TESTUARY_")]
    for key in stripped:
        monkeypatch.delenv(key, raising=False)
    yield


@pytest.fixture(autouse=True)
def _own_user_dir(tmp_path_factory, monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test gets its own empty user directory in place of the platform's,
    so the developer's own models.ini and global config.ini never reach a
    test, and a file one test writes there never reaches another."""
    from testuary import settings

    user_dir = tmp_path_factory.mktemp("user-dir")
    monkeypatch.setattr(settings, "platform_user_dir", lambda: user_dir)
    # Provisioned environments go to platformdirs' cache directory: never the
    # developer's own.
    import platformdirs

    cache = tmp_path_factory.mktemp("user-cache")
    monkeypatch.setattr(platformdirs, "user_cache_dir", lambda *args, **kwargs: str(cache))


@pytest.fixture(autouse=True)
def _discovered_project_interpreter(request, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that selects no project interpreter discovers one or refuses;
    this session's interpreter stands in for the discovered environment, so
    tests about other behavior need not select one. A module that tests the
    discovery itself sets ``REAL_INTERPRETER_DISCOVERY = True``."""
    if not getattr(request.module, "REAL_INTERPRETER_DISCOVERY", False):
        from langadapter.python import project as python_project

        monkeypatch.setattr(python_project, "_discovered_interpreter",
                            lambda root, environment: sys.executable)


#: pytest's configuration files in its order, each with the section that makes
#: it one, for the stand-in below.
_PYTEST_CONFIGURATIONS = (("pytest.ini", "[pytest]"), ("pyproject.toml", "[tool.pytest.ini_options]"),
                          ("tox.ini", "[pytest]"), ("setup.cfg", "[tool:pytest]"))


@pytest.fixture(autouse=True)
def _stand_in_pytest_configuration(monkeypatch: pytest.MonkeyPatch) -> None:
    """LangAdapter asks pytest in the project's interpreter which configuration
    file it reads; tests whose executors are stand-ins cannot run it, so in
    every test the first of pytest's files in the project's root that holds
    its section stands in. LangAdapter's own tests check the real lookup."""
    from langadapter.python import adapter as python_adapter

    def from_the_root(executor, python, project_dir, cache=None, *, flags=None):
        for name, section in _PYTEST_CONFIGURATIONS:
            path = Path(str(project_dir or "")) / name
            if project_dir and path.is_file() and section in path.read_text(
                    encoding="utf-8", errors="ignore"):
                return f"{str(project_dir).rstrip('/')}/{name}"
        return None

    monkeypatch.setattr(python_adapter, "project_pytest_config", from_the_root)


#: A variable this test process has and no captured environment holds.
_PROCESS_ONLY = "PROCESS_ONLY_VARIABLE"


@pytest.fixture
def collection_answer(monkeypatch: pytest.MonkeyPatch):
    """A tool check collects the copy's tests with pytest in the project's
    interpreter. A test whose executor is a stand-in hands that collection to
    the returned function: the project holds no test, so pytest collects
    none. The collection's environment is the captured one with LangAdapter's
    own variables, never this process's, so a variable only this process has
    does not reach it."""
    monkeypatch.setenv(_PROCESS_ONLY, "1")

    def answer(executor, argv, options):
        assert "--collect-only" in argv
        assert _PROCESS_ONLY not in options["env"]
        # LangAdapter's executor holds its context only as a private member, so the stand-in reads it there.
        assert set(executor._context.environment) <= set(options["env"])
        return subprocess.CompletedProcess(argv, 5, "no tests ran in 0.01s\n", "")

    return answer


# Language adapters are session-cached singletons (languages._ADAPTERS), so a
# test that patches a method on the INSTANCE (monkeypatch.setattr(backend, ...))
# leaks at teardown: the attribute existed only on the class, monkeypatch
# captures the bound method as the old value, and teardown writes that bound
# method back as a permanent instance attribute -- silently shadowing every
# later class-level patch in the same process. Patch the class
# (monkeypatch.setattr(type(backend), ...)) instead. This guard names each test
# whose teardown left such a shadow, removes the shadow so later tests see the
# class behavior again, and fails the session.
#
# The guard reads the adapter cache itself: a get_backend() call for every
# registered language would create an adapter for each that lacks one, so
# doing that after every test would add process state of its own. A renamed
# cache raises here rather than silently disabling the guard.
_BACKEND_SHADOWS: dict = {}
_BACKEND_SHADOW_LEAKS: list = []


def _cached_backends() -> dict:
    from testuary import languages as _languages
    return dict(_languages._ADAPTERS)


def _backend_method_shadows() -> dict:
    out: dict = {}
    for name, inst in _cached_backends().items():
        cls = type(inst)
        # An object without instance attributes cannot shadow a method.
        shadow = sorted(
            key for key in getattr(inst, "__dict__", {})
            if callable(getattr(cls, key, None)))
        if shadow:
            out[name] = shadow
    return out


def _remove_new_backend_shadows(nodeid: str) -> None:
    """Record and delete each shadow that was absent at session start."""
    cached = _cached_backends()
    for name, keys in _backend_method_shadows().items():
        for key in keys:
            if key not in _BACKEND_SHADOWS.get(name, []):
                _BACKEND_SHADOW_LEAKS.append((nodeid, f"{name}.{key}"))
                delattr(cached[name], key)


def pytest_sessionstart(session):
    global _BACKEND_SHADOWS
    _BACKEND_SHADOWS = _backend_method_shadows()
    _RESULTS_AT_START.update(_results_files(_LIVE_RESULTS))


@pytest.hookimpl(trylast=True)
def pytest_runtest_logfinish(nodeid, location):
    _remove_new_backend_shadows(nodeid)


def pytest_sessionfinish(session, exitstatus):
    if _BACKEND_SHADOW_LEAKS:
        lines = "\n".join(f"  {nodeid} left {attr}"
                          for nodeid, attr in _BACKEND_SHADOW_LEAKS)
        print(
            "\nERROR: instance-level method shadows leaked onto cached "
            "language backends (patch type(backend), never the instance):\n"
            + lines)
        session.exitstatus = 1
    now = _results_files(_LIVE_RESULTS)
    written = sorted(str(path) for path in now if _RESULTS_AT_START.get(path) != now[path])
    if written:
        print("\nERROR: the session wrote into the working tree's results directory:\n"
              + "\n".join(f"  {path}" for path in written))
        session.exitstatus = 1


@pytest.fixture(autouse=True)
def _ctrl_c_handler_restored():
    """An interrupted command line ignores Ctrl+C from its cleanup until it
    exits. In a test, the next test gets the handler back."""
    import signal

    handler = signal.getsignal(signal.SIGINT)
    yield
    if signal.getsignal(signal.SIGINT) is not handler:
        signal.signal(signal.SIGINT, handler)


_TEMPORARY_VARIABLES = ("TEMP", "TMP", "TMPDIR")


@pytest.fixture(scope="session", autouse=True)
def _library_temporary_dir(tmp_path_factory):
    """The language backend keeps the scratch directory of a failed run for
    diagnosis, and tests make runs fail on purpose. Its directories go below
    pytest's base temporary directory, which pytest rotates, instead of the
    system's temporary directory."""
    temp_dir = tmp_path_factory.mktemp("system-temp", numbered=False)
    saved = {name: os.environ.get(name) for name in _TEMPORARY_VARIABLES}
    saved_dir = tempfile.tempdir
    for name in _TEMPORARY_VARIABLES:
        os.environ[name] = str(temp_dir)
    tempfile.tempdir = str(temp_dir)
    try:
        yield temp_dir
    finally:
        tempfile.tempdir = saved_dir
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@pytest.fixture(scope="session", autouse=True)
def _agent_prompts_before_any_test(_library_temporary_dir):
    """Tests build agents with the Python prompts of agent_prompts. Many
    read them after installing a stand-in language backend, which cannot
    render them, so they are resolved once with the real backend before
    any test runs, whatever the order the tests run in, with this session's
    interpreter standing in for the discovered environment as in every test
    (``_discovered_project_interpreter``). That importing the agents
    resolves nothing is tested in a separate process
    (test_agent_prompts_lazy.py), which this fixture cannot reach."""
    from agent_prompts import python_prompts
    from langadapter.python import project as python_project

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(python_project, "_discovered_interpreter",
                      lambda root, environment: sys.executable)
        python_prompts("test_coverage")
        python_prompts("test_mutation")


@pytest.fixture
def tmp_cwd(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Change ``Path.cwd()`` to a fresh tmpdir for the duration of a
    test. Used by config-loader / file-discovery tests that want to
    control what's in CWD without polluting the real workspace."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


@pytest.fixture
def in_run(tmp_path: Path) -> Iterator:
    """``in_run(**kwargs)`` resolves run settings (``resolve_settings``, by
    default from the user directory's files, the test's TESTUARY_ variables, no
    catalogue and no project settings file, with ``tmp_path`` as the
    project) and enters them for the rest of the test, leaving the run an
    earlier call entered; it returns them. ``in_run.leave()`` leaves it, for
    a test that starts its own run. Configuration is read inside a run
    only."""
    stack = contextlib.ExitStack()

    def enter(**kwargs):
        stack.close()
        kwargs.setdefault("base_dir", tmp_path)
        kwargs.setdefault("models", {})
        kwargs.setdefault("project_settings", False)
        resolved = settings.resolve_settings(**kwargs)
        stack.enter_context(config.use_run_settings(resolved))
        return resolved

    enter.leave = stack.close
    with stack:
        yield enter


@pytest.fixture
def default_run(in_run):
    """A run of the bundled defaults with a model selected, as every real run
    has one, for modules whose tests read configuration; a test replaces it
    with ``in_run(...)``. No model is called."""
    return in_run(overrides={"api": {"provider": "ollama", "model": "test-model"}})


#: An import record that saw no module of the project, and the variable
#: that names its path.
_NO_PROJECT_IMPORTS = {"target": "", "names": [], "seen": {}, "missing": {}, "compiled": {}}
_ORIGIN_RECORD = "LANGADAPTER_IMPORT_ORIGIN_RECORD"


@pytest.fixture
def import_record():
    """``import_record(env, host)`` writes an import record that saw no
    module of the project where ``env`` names one, as LangAdapter's origin
    plugin does in a real run. ``host`` maps the named path to the file
    this process writes: a guest's path to its path on the host, by
    default the path itself. A run that wrote no record is refused, as it
    should be, so a double of a run calls it."""
    def write(env, host=Path):
        record = (env or {}).get(_ORIGIN_RECORD)
        if record:
            host(record).write_text(json.dumps(_NO_PROJECT_IMPORTS), encoding="utf-8")
    return write


@pytest.fixture
def timed_run(import_record):
    """``timed_run(answer)`` is a double of ``langadapter.placement.run_timed``.
    A call writes the import record its environment names (a benchmark's
    first warmup), then answers with ``answer``: a result, or a function
    called as ``run_timed`` is. A benchmark refuses a warmup that wrote no
    record, as it should."""
    def build(answer):
        def run(command, *, env=None, cwd=None, timeout=None):
            import_record(env)
            if callable(answer):
                return answer(command, env=env, cwd=cwd, timeout=timeout)
            return answer
        return run
    return build


@pytest.fixture
def fixtures_dir() -> Path:
    """Path to ``tests/fixtures/`` -- small canned inputs used by
    multiple tests (tiny target source files, sample config snippets,
    etc.)."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def persist_starter_resume():
    """Write complete handoff evidence for a one-test synthetic starter phase,
    with the generation origin of the starter phase and of each phase the
    context reuses, as the active run records it."""
    from testuary import languages, phase_state
    from testuary import evidence as validity
    from testuary.starters import StarterResult, _persist_generation_manifest

    def persist(ctx, source: str) -> Path:
        target_dir = ctx.results_dir / ctx.workspace
        archive = target_dir / "r0" / "archive" / "starter"
        archive.mkdir(parents=True, exist_ok=True)
        snapshot = archive / "result.py"
        snapshot.write_text(source, encoding="utf-8")
        identity = languages.get_backend("python").generated_test_target_identity(
            str(ctx.canonical_path),
            project.resolve_source_dir(str(ctx.canonical_path)),
        )
        validity.write_json_atomic(
            archive / "validity_delivered.json",
            validity.direct_record(
                identity=identity, analyzed_source=source,
                stage="test.initial.delivered",
            ),
        )
        validity.write_json_atomic(
            archive / "consolidation_record.json",
            {"delivered_from": "single_suite", "merged_tests": 1, "delivered_tests": 1},
        )
        _persist_generation_manifest(
            archive / "suites",
            StarterResult(
                status="ok", n_generated=1, n_green=1, requests_used=1,
                logical_requests=1, unknown_usage_requests=0,
                input_tokens=100, cache_hit_tokens=20, output_tokens=50,
                generation_seconds=2.0, suite_generation_attempts={1: 1},
            ),
            generation_seconds=2.0, complete=True,
            # No seed archives are written, so the one slot's evidence is
            # unavailable.
            suite_slots={"version": 1, "requested_slots": 1, "slots": [{
                "slot": 1, "state": "unavailable", "disposition_basis": "unavailable",
                "terminal": True, "attempts_used": 1, "attempts": []}]},
        )
        phase_state.mark_complete(
            target_dir, "starter",
            artifacts={"r0/archive/starter/result.py": snapshot},
            metrics={"suites_passing": 1, "selected_tests": 1,
                     "baseline_line": 50.0, "baseline_branch": 50.0,
                     "baseline_suite_seconds": 0.25,
                     # As every starter marker records them.
                     "suite_mutation_method": ""},
        )
        target = str(ctx.top_file["file_path"])
        for phase in ("starter", "coverage", "mutation"):
            if phase == "starter" or getattr(ctx.reuse, phase, False):
                validity.write_origin(
                    validity.phase_origin_path(target_dir, phase, 0),
                    validity.current_condition(phase, languages.detect_language(target),
                                               project.resolve_source_dir(target)))
        return snapshot

    return persist

@pytest.fixture(autouse=True)
def _no_wall_clock_tie_break(monkeypatch):
    """The phase-end tie-break times its candidates with real suite runs. Under
    a parallel test session those runs land on top of other workers' mutation
    subprocesses, and a mutation verdict judged on a wall clock is not
    measurable under that load. The selection order is covered by injecting a
    runtime function directly (``tests/unit/test_phase_end_selection.py``), so
    the suite takes no wall-clock measurement of its own: the configured
    timer gets no repetitions (the settings refuse a run_count of 0), so the
    phase end refuses with its own message. A test that sets its run's
    TESTUARY_TIMING_RUN_COUNT times for real."""
    from testuary import consolidation

    configured = consolidation.benchmark_tied_candidates

    def without_repetitions(*args, **kwargs):
        timer = configured(*args, **kwargs)
        if "TESTUARY_TIMING_RUN_COUNT" not in os.environ:
            timer.benchmark_runs = 0
        return timer

    monkeypatch.setattr(consolidation, "benchmark_tied_candidates", without_repetitions)


# A test that leaves the run directory unset writes into the working tree's
# .testuary/results/, where a real run's results live, and a test that names
# the working tree as its output directory writes into results/. Each test is
# checked on the directories a run writes to (results/<model>/<target>/), and
# the whole directories once at the end of the session.
_LIVE_RESULTS = (Path(__file__).resolve().parents[1] / "results",
                 Path(__file__).resolve().parents[1] / ".testuary" / "results")


def _results_dirs(roots, depth: int = 3) -> dict:
    """Each directory down to ``depth`` below each of ``roots``, with its
    modification time, which a file created or removed in it changes."""
    found = {}
    pending = [(root, 0) for root in roots]
    while pending:
        results_dir, level = pending.pop()
        try:
            found[results_dir] = results_dir.stat().st_mtime_ns
            children = list(os.scandir(results_dir)) if level < depth else []
        except OSError:
            continue
        pending.extend((Path(child.path), level + 1) for child in children
                       if child.is_dir(follow_symlinks=False))
    return found


def _results_files(roots) -> dict:
    files = {}
    for results_dir, _dirs, names in (entry for root in roots for entry in os.walk(root)):
        for name in names:
            path = Path(results_dir) / name
            try:
                files[path] = path.stat().st_mtime_ns
            except OSError:
                pass
    return files


_RESULTS_AT_START: dict = {}


@pytest.fixture(autouse=True)
def _live_results_untouched():
    before = _results_dirs(_LIVE_RESULTS)
    yield
    after = _results_dirs(_LIVE_RESULTS)
    changed = sorted(str(path) for path in after if before.get(path) != after[path])
    if changed:
        pytest.fail("the test wrote into the working tree's results directory: "
                    + ", ".join(changed) + ". Point the run directory at the test's tmp_path.")


#: The marker of a test that starts a real mutation tool (the suite split):
#: such tests run in a selection of their own.
MUTATION_TOOL = "mutation_tool"
#: The Cosmic Ray and PIT commands the current test started.
_TOOL_STARTS = []


def starts_mutation_tool(args):
    """Whether a command starts Cosmic Ray or PIT as LangAdapter launches
    them: Cosmic Ray's console script, its command line in a Python -c, an
    HTTP worker or the container's session runner, PIT through its argument
    file or report class. A probe that only names a tool does not count."""
    words = ([os.fsdecode(args)] if isinstance(args, (str, bytes))
             else [os.fsdecode(a) if isinstance(a, (bytes, os.PathLike)) else str(a) for a in args])
    return bool(words) and (PureWindowsPath(words[0]).stem.lower() == "cosmic-ray" or any(
        mark in word for word in words
        for mark in ("cosmic_ray.cli", "'http-worker'", "_cr_session_runner", "pit.args",
                     "org.pitest")))


def pytest_configure(config):
    """Record each command LangAdapter's process tree starts that starts a
    mutation tool."""
    # No public hook sees each command LangAdapter starts, so the guard wraps its private launcher.
    from langadapter import _process_tree

    original = _process_tree.start
    if getattr(original, "records_mutation_tools", False):
        return

    def start(args, **options):
        if starts_mutation_tool(args):
            _TOOL_STARTS.append(args)
        return original(args, **options)

    start.records_mutation_tools = True
    _process_tree.start = start


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_setup(item):
    _TOOL_STARTS.clear()


@pytest.fixture(autouse=True)
def mutation_tool_guard(request):
    """Fail a test that starts Cosmic Ray or PIT, in its fixtures' setup or
    itself, without the mutation_tool marker. It sees the pytest process
    only: a test whose child process starts a tool is marked by hand."""
    yield
    if _TOOL_STARTS and request.node.get_closest_marker(MUTATION_TOOL) is None:
        command = " ".join(map(str, _TOOL_STARTS[0]))[:200]
        pytest.fail(f"the test starts a mutation tool ({command}) without the "
                    f"{MUTATION_TOOL} marker", pytrace=False)


@pytest.fixture
def provider_module(tmp_path_factory, monkeypatch: pytest.MonkeyPatch):
    """A module of the given name from an empty source file on the import
    path, as an added provider's chat_model item names one: LangAdapter's
    loader takes a module with a source file. The test sets its
    attributes."""
    def make(name: str):
        directory = tmp_path_factory.mktemp("provider")
        (directory / f"{name}.py").write_text("", encoding="utf-8")
        monkeypatch.syspath_prepend(str(directory))
        monkeypatch.delitem(sys.modules, name, raising=False)
        module = importlib.import_module(name)
        monkeypatch.setitem(sys.modules, name, module)
        return module
    return make
