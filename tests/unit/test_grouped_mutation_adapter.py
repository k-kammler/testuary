"""Mutation adapters consume the public group without flat convenience methods."""
from types import SimpleNamespace

import pytest

from langadapter import LangAdapter, MutationOperations, Outcome
from langadapter.mutation import OperatorCatalogue
from langadapter.mutation import MutationRun
from langadapter.mutation import MutationCleanupError, MutationSession
from langadapter.python.adapter import PythonAdapter

from testuary.languages import mutation
from testuary.languages.generic import GenericMutationRunner

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def PitRunner(backend, config, **options):
    """The generic mutation runner on Java's engine, PIT."""
    return GenericMutationRunner(backend, config, engine="pit", **options)


class _GroupOnly(SimpleNamespace):
    """A backend with only the given groups: every hook keeps LangAdapter's
    default, or Python's for Cosmic Ray's group."""

    def __getattr__(self, name):
        engine = getattr(self.__dict__.get("mutations"), "engine", None)
        hooks = PythonAdapter if engine == "cosmic_ray" else LangAdapter
        value = getattr(hooks, name)
        return value.__get__(self) if hasattr(value, "__get__") else value


class RecordingSession(MutationSession):
    def __init__(self, events, outcome=Outcome.COMPLETED, *, fail_close=False):
        super().__init__(outcome=outcome, reason="native setup failure" if outcome is not Outcome.COMPLETED else "")
        self.events = events
        self.fail_close = fail_close
        self.requests = []
        self.raw = MutationRun.failed("target.py", Outcome.TIMED_OUT, "incomplete native run")

    def measure(self, request):
        self.requests.append(request)
        return self.raw

    def release(self):
        self.events.append("close")
        if self.fail_close:
            raise OSError("workspace still held")


class RecordingMutations(MutationOperations):
    def __init__(self, engine, sessions):
        self.engine = engine
        self.sessions = list(sessions)
        self.opened = []
        self.catalogue_reads = []

    def engines(self):
        return (self.engine,)

    def operators(self, engine):
        self.catalogue_reads.append(engine)
        return OperatorCatalogue(engine, ("op/A", "op/B"))

    def open_session(self, engine, **options):
        assert engine == self.engine
        self.opened.append(options)
        session = self.sessions.pop(0)
        session.events.append("open")
        return session


@pytest.mark.parametrize("runner_type,engine,config", [
    (mutation.MutationRunner, "cosmic_ray", {"cosmic_ray_operators": ""}),
    (PitRunner, "pit", {"pit_mutators": "", "per_mutant_timeout_seconds": "45"}),
])
def test_group_only_backend_preserves_catalogue_requests_and_run(
        tmp_path, runner_type, engine, config):
    (tmp_path / "tests.py").write_text("def test_value(): pass\n", encoding="utf-8")
    (tmp_path / "target.py").write_text("value = 1\n", encoding="utf-8")
    events = []
    native = RecordingSession(events)
    group = RecordingMutations(engine, [native])
    runner = runner_type(_GroupOnly(mutations=group), config)
    listed = runner.full_operators()
    listed.append("caller-only")
    assert runner.full_operators() == ["op/A", "op/B"]
    assert group.catalogue_reads == [engine]
    with runner.session(3, suite_dirs=()) as session:
        measured = session.run(
            target=tmp_path / "target.py", test_dir=tmp_path,
            test_files=["tests.py"], per_mutant_timeout=45,
            wall_clock_timeout=600, operators=["op/B"])
        request = native.requests[0]
        assert request.catalogue.operators == ("op/B",)
        assert request.attribution == "first"
        assert request.per_mutant_timeout == 45
        assert request.wall_clock_limit_s == 600
        assert request.workers is None and not request.fail_on_unresolved
        assert request.suite.test_files == ("tests.py",)
        assert measured.total == 0 and measured.truncated
        assert measured.run.outcome is Outcome.TIMED_OUT
        assert measured.run.score() is None
        assert measured.kill_matrix()["resolution"]["method"] == "measured"
    assert events == ["open", "close"]
    assert not native.is_open
    if engine == "pit":
        assert group.opened == [{"workers": 3}]
    else:
        # The default distributor follows the three workers.
        assert group.opened[0]["distributor"] == "http"
        assert group.opened[0]["workers"] == 3


def test_http_failure_is_closed_before_sequential_fallback():
    events = []
    failed = RecordingSession(events, Outcome.INFRASTRUCTURE_FAILURE)
    success = RecordingSession(events)
    group = RecordingMutations("cosmic_ray", [failed, success])
    runner = mutation.MutationRunner(_GroupOnly(mutations=group), {
        "cosmic_ray_distributor": "http", "allow_sequential_fallback": True})
    with runner.session(4, suite_dirs=()):
        assert events == ["open", "close", "open"]
        assert not failed.is_open
    assert events == ["open", "close", "open", "close"]
    assert [options["workers"] for options in group.opened] == [4, 1]


def test_failed_cleanup_prevents_fallback_and_retains_recovery_handle():
    events = []
    failed = RecordingSession(events, Outcome.INFRASTRUCTURE_FAILURE, fail_close=True)
    group = RecordingMutations("cosmic_ray", [failed, RecordingSession(events)])
    runner = mutation.MutationRunner(_GroupOnly(mutations=group), {
        "cosmic_ray_distributor": "http", "allow_sequential_fallback": True})
    with pytest.raises(MutationCleanupError) as error:
        runner.open(4, suite_dirs=())
    assert error.value.cleanup_handle is failed
    assert len(group.opened) == 1
    failed.fail_close = False
    error.value.cleanup_handle.close()
    assert not failed.is_open


@pytest.mark.parametrize("runner_type,engine,config", [
    (mutation.MutationRunner, "cosmic_ray", {}),
    (PitRunner, "pit", {}),
])
def test_failed_setup_without_fallback_is_closed(runner_type, engine, config):
    events = []
    failed = RecordingSession(events, Outcome.MISSING_SETUP)
    group = RecordingMutations(engine, [failed])
    runner = runner_type(_GroupOnly(mutations=group), config)
    with pytest.raises(mutation.MutationSetupError, match="native setup failure"):
        runner.open(1, suite_dirs=())
    assert events == ["open", "close"]


def test_sequential_failure_is_closed_as_well():
    events = []
    group = RecordingMutations("cosmic_ray", [
        RecordingSession(events, Outcome.INFRASTRUCTURE_FAILURE),
        RecordingSession(events, Outcome.MISSING_SETUP)])
    runner = mutation.MutationRunner(_GroupOnly(mutations=group), {
        "cosmic_ray_distributor": "http", "allow_sequential_fallback": True})
    with pytest.raises(mutation.MutationSetupError, match="native setup failure"):
        runner.open(4, suite_dirs=())
    assert events == ["open", "close", "open", "close"]
