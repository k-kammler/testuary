"""Testuary's side of the public mutation session: its readings of a run,
the settings, catalogues and requests it builds from its configuration,
and the sessions it opens."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from langadapter import mutation as native
from langadapter.mutation import Classification, MutationClass
from langadapter.mutation import Mutant, MutantKey, MutationRun
from langadapter.outcomes import Outcome
from langadapter.python.adapter import PythonAdapter

from testuary import model_requests as _model_requests
from testuary.languages import mutation
from testuary.languages.generic import GenericBackend
from langadapter.python.cosmic_ray import CosmicRaySettings
from testuary import config as config_module
from testuary import starters
from testuary import measurement
from testuary.settings import RunSettings


def _mutant(job, line, sub, *, diff=None, row=None):
    klass, status = {
        native.BY_FAILING_TEST: (MutationClass.DETECTED, native.KILLED_STATUS),
        native.WITHOUT_NAMED_TEST: (MutationClass.DETECTED, native.KILLED_STATUS),
        native.BY_TIMEOUT: (MutationClass.DETECTED, native.TIMEOUT_STATUS),
        native.NON_VIABLE: (MutationClass.DETECTED, native.NON_VIABLE_STATUS),
        native.SURVIVED: (MutationClass.UNDETECTED, native.SURVIVED_STATUS),
        native.UNTESTED: (MutationClass.NOT_RUN, native.UNTESTED_STATUS),
    }[sub]
    return Mutant(
        key=MutantKey("calc.py", "core/Op", 0, line, 0, None),
        engine_id=job, test_outcome=None, worker_outcome=None,
        classification=Classification(status, klass, sub),
        killers=(), diff=diff, row={"id": job, "status": status, **(row or {})})


def _run(mutants, outcome=Outcome.COMPLETED, incomplete=None):
    return MutationRun(target_file="calc.py", mutants=tuple(mutants),
                       summary={}, requested_catalogue="full",
                       requested_operators=(), outcome=outcome,
                       incomplete=incomplete)


def test_the_counts_are_the_sessions_reading_before_attribution():
    """A timeout the journal attributed to a failing test is a kill in the
    run's classification, and a timeout in the counts the agent's gates and
    summary lines use."""
    run = _run([
        _mutant("a", 1, native.BY_FAILING_TEST),
        _mutant("b", 2, native.WITHOUT_NAMED_TEST),
        _mutant("c", 3, native.BY_TIMEOUT),
        _mutant("e", 5, native.BY_FAILING_TEST,
                row={"pre_attribution_status": native.TIMEOUT_STATUS}),
        _mutant("f", 6, native.NON_VIABLE),
        _mutant("g", 7, native.SURVIVED),
        _mutant("h", 8, native.UNTESTED),
    ])
    measured = mutation.MutationMeasurement.of(run)
    assert (measured.killed, measured.timeout, measured.non_viable,
            measured.survived, measured.untested, measured.total) == (
        2, 2, 1, 1, 1, 6)
    assert measured.score == pytest.approx(run.progress_score())
    assert not measured.truncated and measured.run is run


def test_a_cut_run_counts_its_untested_mutants_as_not_killed():
    run = _run([_mutant("a", 1, native.BY_FAILING_TEST),
                _mutant("b", 2, native.UNTESTED)], Outcome.TIMED_OUT,
               {"reason": "untested_mutants_in_catalogue", "untested": 1,
                "total": 2})
    measured = mutation.MutationMeasurement.of(run)
    # One of two mutants killed: a lower bound, never the 100% of the
    # mutants that have a result.
    assert measured.truncated and measured.score == 50.0
    assert run.progress_score() == 50.0 and run.score() is None


def test_a_run_without_results_scores_zero_in_the_loop():
    measured = mutation.MutationMeasurement.of(
        _run([_mutant("a", 1, native.UNTESTED)]))
    assert measured.score == 0.0 and measured.total == 0


def test_a_run_refused_for_failing_tests_names_them():
    """A run the tool refused because tests failed without a mutant names
    those tests when the adapter gives them (MutationRun.stopping_tests of
    the run contract), so the loop's reply can name them."""
    refused = SimpleNamespace(
        measured=False, failure=None, reason="the suite fails without a mutant",
        outcome=Outcome.TEST_FAILURES,
        stopping_tests=(SimpleNamespace(text="test_a.py::test_loaded"),))
    measured = mutation.MutationMeasurement.of(refused)
    assert measured.suite_failed
    assert measured.stopping_tests == refused.stopping_tests
    assert mutation.test_texts(measured.stopping_tests) == ("test_a.py::test_loaded",)


def test_an_instrument_fault_carries_the_tests_its_run_names():
    """A run whose check before the first mutant fails as an instrument
    fault, such as Python's origin refusal, still raises, and the fault
    carries the run's stopping tests as id objects for a model-silent
    place to remove."""
    from langadapter.suites import TestId

    named = TestId("test_a.py", "test_loaded", text="test_a.py::test_loaded")
    fault_run = MutationRun.failed("calc.py", Outcome.INFRASTRUCTURE_FAILURE,
                                   "the origin check refused", failure="instrument_check",
                                   stopping_tests=(named,))
    with pytest.raises(mutation.MutationInstrumentFault, match="origin check") as raised:
        mutation.MutationMeasurement.of(fault_run)
    assert raised.value.stopping_tests == (named,)


def test_an_instrument_fault_raises_and_other_failures_are_empty():
    fault = MutationRun.failed("calc.py", Outcome.INFRASTRUCTURE_FAILURE,
                               "no tests ran", failure="instrument_check")
    with pytest.raises(mutation.MutationInstrumentFault, match="no tests"):
        mutation.MutationMeasurement.of(fault)
    # A worker that raised for every mutant is a broken run environment too,
    # not a run that tested nothing.
    workers = MutationRun.failed("calc.py", Outcome.INFRASTRUCTURE_FAILURE,
                                 "the worker raised an exception for every one of the 2 mutants",
                                 failure="workers")
    with pytest.raises(mutation.MutationInstrumentFault, match="every one of the 2"):
        mutation.MutationMeasurement.of(workers)
    for failure in ("initialization", "session_read", "session", None):
        measured = mutation.MutationMeasurement.of(MutationRun.failed(
            "calc.py", Outcome.INFRASTRUCTURE_FAILURE, "x", failure=failure))
        # Empty, with the adapter's reason kept for the model.
        assert measured == mutation.MutationMeasurement.empty(reason="x")
        assert measured.kill_matrix() is None
    # The request classifier never retries it as a transport failure.
    assert not _model_requests._transient_single(
        mutation.MutationInstrumentFault("x"))


def test_the_survivors_text_lists_every_survivor_over_the_cap():
    diff = ("--- mutation diff ---\n--- acalc.py\n+++ bcalc.py\n"
            "@@ -1,2 +1,2 @@\n-    return a + b\n+    return a - b\n")
    # The run lists mutants out of line order. The list is ordered by source
    # line, and the one change shown is the middle of the two.
    run = _run([_mutant("j3", 5, native.SURVIVED, diff=diff),
                _mutant("j1", 1, native.BY_FAILING_TEST),
                _mutant("j2", 2, native.SURVIVED, diff=diff)])
    measured = mutation.MutationMeasurement.of(run)
    one = measured.survivors_text(max_mutants=1)
    assert one.startswith(
        "All 2 surviving mutants (id@line): j2@L2, j3@L5\n\n"
        "[... 1 further mutant change omitted, read_source around the lines "
        "listed above and strengthen those tests ...]\n\n")
    assert measured.survivors_text(max_mutants=None).count("--- mutant ") == 2
    assert one.endswith("--- mutant j3 (line 5) ---\n@@ -1,2 +1,2 @@\n"
                        "-    return a + b\n+    return a - b")
    both = measured.survivors_text(max_mutants=10)
    assert both.count("--- mutant ") == 2 and "All 2" not in both
    assert mutation.MutationMeasurement.of(
        _run([_mutant("k", 1, native.BY_FAILING_TEST)])).survivors_text() == ""


def test_the_kill_matrix_carries_the_timing_fields():
    matrix = mutation.MutationMeasurement.of(
        _run([_mutant("a", 1, native.BY_FAILING_TEST)])).kill_matrix()
    assert matrix["mutants"][0]["id"] == "a"
    assert {"t_mono_s", "clock_domain_id"} <= set(matrix)


def test_the_configured_catalogue():
    assert mutation.configured_catalogue(
        {"cosmic_ray_operators": ""}).kind == "full"
    with pytest.raises(KeyError):
        mutation.configured_catalogue({})
    named = mutation.configured_catalogue(
        {"cosmic_ray_operators": "wong_mathur_selective"})
    assert named.name == "wong_mathur_selective"
    listed = mutation.configured_catalogue(
        {"cosmic_ray_operators": "core/A, core/B"})
    assert listed.operators == ("core/A", "core/B")
    runner = mutation.MutationRunner(
        None, {"cosmic_ray_operators": "wong_mathur_selective"})
    assert len(runner.configured_operators()) > 10


def test_a_request_carries_the_operators_and_the_attribution(tmp_path):
    runner = mutation.MutationRunner(None, {"cosmic_ray_operators": "core/A"})
    request = runner.request(
        target=tmp_path / "calc.py", test_dir=tmp_path,
        test_files=[str(tmp_path / "test_a.py")], per_mutant_timeout=12,
        wall_clock_timeout=600)
    assert request.catalogue.operators == ("core/A",)
    assert request.attribution == "first"
    assert (request.workers, request.fail_on_unresolved) == (None, False)
    assert (request.per_mutant_timeout, request.wall_clock_limit_s) == (12, 600)
    assert request.suite.test_files == (str(tmp_path / "test_a.py"),)
    every_killer = mutation.MutationRunner(None, {})
    every_killer.attribution = "all"
    full = every_killer.request(
        target=tmp_path / "calc.py", test_dir=tmp_path, test_files=["t.py"],
        per_mutant_timeout=1, wall_clock_timeout=1,
        operators=["core/A", "core/B"])
    assert full.catalogue.operators == ("core/A", "core/B")
    assert full.attribution == "all"


def test_one_run_can_record_every_killing_test(tmp_path):
    """The pooled starter run records every killing test, whatever
    the configured attribution, and only that run."""
    common = dict(target=tmp_path / "calc.py", test_dir=tmp_path, test_files=["t.py"],
                  per_mutant_timeout=1, wall_clock_timeout=1)
    first = mutation.MutationRunner(None, {"cosmic_ray_operators": ""})
    assert first.request(**common, attribution="all").attribution == "all"
    assert first.request(**common).attribution == "first"
    full = mutation.MutationRunner(None, {"cosmic_ray_operators": ""})
    full.attribution = "all"
    assert full.request(**common, attribution="first").attribution == "first"
    with pytest.raises(ValueError, match="first or all"):
        first.request(**common, attribution="full")
    # The session passes it on to the request it measures.
    seen = {}

    def request(**options):
        seen.update(options)
        raise RuntimeError("stop after the request")

    session = mutation.MutationSession(SimpleNamespace(request=request), None, 1, False)
    with pytest.raises(RuntimeError, match="stop after the request"):
        session.run(**common, attribution="all")
    assert seen["attribution"] == "all"


class _Backend:
    """Records each session it opens. ``outcomes`` are their outcomes in
    order. Its session attempts are Python's."""

    mutation_session_attempts = PythonAdapter.mutation_session_attempts

    def __init__(self, *outcomes):
        self.outcomes = list(outcomes)
        self.opened = []
        self.mutations = SimpleNamespace(open_session=self.open_mutation_session)

    def open_mutation_session(self, engine, **options):
        self.opened.append(options)
        outcome = self.outcomes.pop(0)
        return SimpleNamespace(outcome=outcome, reason="2 of 4 workers",
                               close=lambda: None)


@pytest.mark.parametrize("distributor, workers, expected", [
    ("http", 4, ("http", 4, True)),
    ("http", 1, ("http", 1, True)),
    ("auto", 4, ("http", 4, True)),
    ("auto", 1, ("local", 1, False)),
    (None, 4, ("http", 4, True)),
    ("local", 1, ("local", 1, False)),
])
def test_the_distributor_follows_the_worker_count_unless_configured(
        monkeypatch, distributor, workers, expected, default_run):
    """The default auto runs local for one worker and http for more. An
    explicit distributor is kept, also http with one worker."""
    backend = _Backend(Outcome.COMPLETED)
    config = {} if distributor is None else {"cosmic_ray_distributor": distributor}
    session = mutation.MutationRunner(backend, config).open(workers, suite_dirs=())
    options = backend.opened[0]
    assert (options["distributor"], options["workers"], session.use_http) == expected
    assert options["allow_fewer_workers"] is True
    assert session.workers == options["workers"]


def test_a_local_distributor_with_several_workers_is_refused(monkeypatch, default_run):
    """It would run one mutant at a time while the run records four
    workers: the session refuses it, naming both settings."""
    backend = _Backend(Outcome.COMPLETED)
    with pytest.raises(mutation.MutationSetupError) as refused:
        mutation.MutationRunner(backend, {"cosmic_ray_distributor": "local"}).open(4, suite_dirs=())
    assert backend.opened == []
    assert str(refused.value) == (
        "cosmic_ray_distributor = local runs one mutant at a time, but "
        "engine_workers = 4: set cosmic_ray_distributor to auto or http, "
        "or engine_workers to 1")


def test_a_fleet_that_does_not_start(monkeypatch, default_run):
    with pytest.raises(mutation.MutationSetupError, match="2 of 4"):
        mutation.MutationRunner(
            _Backend(Outcome.INFRASTRUCTURE_FAILURE),
            {"cosmic_ray_distributor": "http"}).open(4, suite_dirs=())
    backend = _Backend(Outcome.INFRASTRUCTURE_FAILURE, Outcome.COMPLETED)
    lines = []
    session = mutation.MutationRunner(
        backend, {"cosmic_ray_distributor": "http",
                  "allow_sequential_fallback": True},
        log=lambda message, level=1: lines.append(message)).open(4, suite_dirs=())
    assert [o["distributor"] for o in backend.opened] == ["http", "local"]
    assert not session.use_http and "sequential local distributor" in lines[0]
    assert session.workers == 1


def test_a_run_forwards_mutation_progress_only_for_an_observed_target(tmp_path, monkeypatch, default_run):
    """The "Mutation measurement" row: the native request carries a progress
    callback while a display observes the target, and none otherwise. The
    row ends with the run's stated outcome: a complete run at every mutant
    measured, whatever its last progress record said."""
    from langadapter.mutation import MutationProgress
    from testuary import _progress

    seen, rows = [], []
    results = [_run([]), _run([]), _run([], Outcome.INFRASTRUCTURE_FAILURE)]

    def native_run(request):
        seen.append(request.progress)
        if request.progress is not None:
            request.progress(MutationProgress("mutants", 1, 2, 0.5))
        return results.pop(0)

    session = mutation.MutationRunner(
        _Backend(Outcome.COMPLETED), {"cosmic_ray_distributor": "local", "cosmic_ray_operators": "core/A"}).open(1, suite_dirs=())
    session._session.run = native_run
    display = SimpleNamespace(
        current_phase=lambda target: "mutation",
        mutation_progress=lambda target, phase, record, state=None, message=None: rows.append(
            (target, phase, record.finished, state)))
    arguments = dict(target=tmp_path / "calc.py", test_dir=tmp_path, test_files=["test_a.py"],
                     per_mutant_timeout=1, wall_clock_timeout=10)
    session.run(**arguments)
    with _progress.observe_target(display, "calc.py"):
        session.run(**arguments)
        session.run(**arguments)
    assert seen[0] is None and callable(seen[1]) and callable(seen[2])
    assert rows == [("calc.py", "mutation", 1, None), ("calc.py", "mutation", 2, "succeeded"),
                    ("calc.py", "mutation", 1, None), ("calc.py", "mutation", 1, "failed")]


def test_the_test_files_of_a_suite_directory(tmp_path):
    for name in ("test_a.py", "conftest.py", "notes.txt", "helper.py"):
        (tmp_path / name).write_text("", encoding="utf-8")
    backend = SimpleNamespace(
        file_extensions=(".py",), test_file_extensions=lambda: (".py",),
        shared_setup_files=lambda d: (str(Path(d) / "conftest.py"),))
    assert mutation.suite_test_files(backend, tmp_path) == [
        str(tmp_path / "helper.py"), str(tmp_path / "test_a.py")]


def test_compact_diff_keeps_hunks_and_changed_lines():
    raw = ("--- a/foo.py\n+++ b/foo.py\n@@ -1,3 +1,3 @@\n"
           "-    return x + 1\n+    return x - 1\n")
    out = mutation.compact_diff(raw)
    assert out == "@@ -1,3 +1,3 @@\n-    return x + 1\n+    return x - 1"


def test_compact_diff_of_nothing_is_empty():
    assert mutation.compact_diff("") == ""


def test_compact_diff_falls_back_to_the_first_six_lines():
    """No hunk or changed line: the first six non-empty lines, so the model
    still sees something."""
    text = "\n".join(f"line {i}" for i in range(20))
    assert mutation.compact_diff(text) == "\n".join(
        f"line {i}" for i in range(6))


def test_compact_diff_cuts_with_the_shared_marker():
    text = "@@ -1,1 +1,1 @@\n+" + "x" * 1000
    out = mutation.compact_diff(text, max_chars=120)
    assert out == text[:120] + f"\n[... {len(text) - 120} chars of the change omitted ...]"
    assert mutation.compact_diff(text, max_chars=None) == text


def test_an_unknown_engine_is_refused(default_run):
    from testuary import languages
    with pytest.raises(ValueError, match="fake_backend"):
        languages.get_backend("python").mutation_runner(
            {"engine": "fake_backend"})


@pytest.mark.parametrize("configured, workers, effective", [
    ("auto", 1, "local"), ("auto", 6, "http"), ("http", 1, "http"), ("local", 1, "local"),
    ("local", 6, "refused"),
])
def test_the_effective_settings_record_the_distributor_used(tmp_path, configured, workers, effective):
    """The record keeps the configured value and names the one a Python run
    uses beside it."""
    import os

    from testuary import config as config_module
    from testuary.settings import RunSettings

    settings = RunSettings(base_dir=tmp_path, environment=dict(os.environ), sections={
        "mutation": {"cosmic_ray_distributor": configured, "engine_workers": workers}})
    with config_module.use_run_settings(settings):
        record = config_module.resolved_test_preset()
    assert (record["sections"]["mutation"]["cosmic_ray_distributor"],
            record["effective"]["cosmic_ray_distributor"]) == (configured, effective)


def test_a_run_the_engine_refused_as_failing_says_so():
    measured = mutation.MutationMeasurement.of(
        _run([], Outcome.TEST_FAILURES))
    assert measured.suite_failed and measured.total == 0
    other = mutation.MutationMeasurement.of(_run([_mutant("a", 1, native.UNTESTED)]))
    assert not other.suite_failed


def test_a_measurement_cut_by_its_clock_is_not_a_refused_suite():
    run = _run([_mutant("a", 1, native.BY_FAILING_TEST),
                _mutant("b", 2, native.UNTESTED)], Outcome.TIMED_OUT,
               {"reason": "untested_mutants_in_catalogue", "untested": 1, "total": 2})
    measured = mutation.MutationMeasurement.of(run)
    assert measured.truncated and not measured.suite_failed


# The ``engine`` setting names a language's own engine.
#
# Its shipped default (Cosmic Ray) is read for every language and stands for
# the language's own engine, so a Java run keeps PIT with it. Another
# language's engine, or an unknown one, is refused.
ENGINES = {"python": ("cosmic_ray",), "java": ("pit",)}


def _runner(language, engine, monkeypatch):
    backend = GenericBackend(language, (".py",) if language == "python" else (".java",))
    project = SimpleNamespace(native_mutation_engines=ENGINES[language])
    monkeypatch.setattr(backend, "project_backend", lambda source_dir=None: project)
    return backend.mutation_runner({"engine": engine})


@pytest.mark.parametrize("language, configured, engine", [
    ("python", "", "cosmic_ray"), ("python", "cosmic_ray", "cosmic_ray"),
    ("java", "", "pit"), ("java", "pit", "pit"), ("java", "cosmic_ray", "pit"),
])
def test_the_language_runs_its_own_engine(language, configured, engine, monkeypatch):
    assert _runner(language, configured, monkeypatch).name == engine


@pytest.mark.parametrize("language, configured, message", [
    ("python", "pit", "Unknown Python mutation engine: 'pit'. Valid: 'cosmic_ray' (default)."),
    ("java", "mutmut", "Unknown Java mutation engine: 'mutmut'. Valid: 'pit' (default)."),
])
def test_another_engine_is_refused(language, configured, message, monkeypatch):
    with pytest.raises(ValueError) as refused:
        _runner(language, configured, monkeypatch)
    assert str(refused.value) == message


# Every engine built from the mutation configuration gets its Cosmic Ray settings.
FIELDS = ("clone_sweep_hours", "worker_ready_timeout_s", "verbose", "per_file_adjudication")


def sessions():
    """The session settings of each configuration an engine is built from, as
    the mutation runner opens a session with it."""
    configurations = {
        "editing agent": config_module.get_agent_config("test_mutation", parent="test_coverage"),
        "starter-suite phase end": starters._starter_mutation_config(),
        "Finishing": measurement.mutation_config(),
    }
    settings = {}
    for route, config in configurations.items():
        backend = _Backend(Outcome.COMPLETED)
        mutation.MutationRunner(backend, config).open(1, suite_dirs=())
        settings[route] = backend.opened[0]["settings"]
    return settings


def values(session):
    return tuple(getattr(session, field) for field in FIELDS)


@pytest.mark.parametrize("hours", ["0", "1.5"])
def test_the_configured_stale_clone_age_reaches_every_engine(hours):
    with config_module.use_run_settings(RunSettings(sections={"mutation": {"cosmic_ray_clone_sweep_hours": hours}})):
        assert {route: session.clone_sweep_hours for route, session in sessions().items()} == dict.fromkeys(
            ("editing agent", "starter-suite phase end", "Finishing"), float(hours))


def test_every_configured_session_setting_reaches_every_engine():
    configured = {"cosmic_ray_clone_sweep_hours": "1.5",
                  "cosmic_ray_worker_ready_timeout_s": "43", "cosmic_ray_verbose": "true",
                  "cosmic_ray_per_file_adjudication": "false"}
    with config_module.use_run_settings(RunSettings(sections={"mutation": configured})):
        for route, session in sessions().items():
            assert values(session) == (1.5, 43, True, False), route


def test_the_bundled_values_equal_the_engine_defaults():
    with config_module.use_run_settings(RunSettings()):
        for route, session in sessions().items():
            assert values(session) == values(CosmicRaySettings()), route


def test_the_record_names_the_run_languages_validity_policy(tmp_path):
    """The resolved-settings record named the Python adapter's policy for a
    Java run too: it names the run language's, ``none`` without one."""
    import os

    from langadapter.java.adapter import JavaAdapter
    from langadapter.python.adapter import PythonAdapter
    from testuary import config as config_module
    from testuary.settings import RunSettings

    settings = RunSettings(base_dir=tmp_path, environment=dict(os.environ), sections={})
    with config_module.use_run_settings(settings):
        records = {language: config_module.resolved_test_preset(language=language)
                   for language in ("python", "java", None, "unregistered")}
    assert {language: record["generated_test_validity"]["policy_version"]
            for language, record in records.items()} == {
        "python": PythonAdapter.validity_policy_version,
        "java": JavaAdapter.validity_policy_version,
        None: "none", "unregistered": "none"}
    assert JavaAdapter.validity_policy_version.startswith("java-")
