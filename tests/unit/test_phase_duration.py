"""Phase duration across a crash, a kill, a reuse and a repeat. A phase's
recorded duration consists only of segments written by the run that produced
the artifact the phase holds.
"""
import contextlib
import json
import time

import pytest

from testuary import phase_state

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture
def target_dir(tmp_path):
    (tmp_path / "src.py").write_text("x = 1\n", encoding="utf-8")
    return tmp_path


def _started(target_dir):
    phase_state.mark_started(
        target_dir, "coverage", target_number=1, stem="src",
        target_file_path="src.py", canonical_source_path=target_dir / "src.py")


def _segments(target_dir, phase="coverage", run_idx=0):
    state = json.loads(
        phase_state.state_path(target_dir).read_text(encoding="utf-8"))
    return state[phase][str(run_idx)]["segments"]


def test_a_completed_attempt_reports_its_own_time(target_dir):
    with phase_state.timed_segment(target_dir, "coverage", engine_workers=16):
        phase_state.record_time(phase_state.TIME_MODEL, 0.25)

    recorded = _segments(target_dir)
    assert len(recorded) == 1
    assert recorded[0]["outcome"] == phase_state.SEGMENT_RAN
    assert recorded[0]["engine_workers"] == 16
    assert recorded[0]["model_seconds"] == 0.25
    assert phase_state.output_seconds(target_dir, "coverage") is not None


def test_a_crash_then_a_completion_reports_only_the_completion(target_dir):
    """A crashed attempt re-runs from scratch, so its time bought nothing."""
    with pytest.raises(RuntimeError):
        with phase_state.timed_segment(target_dir, "coverage"):
            time.sleep(0.05)
            raise RuntimeError("engine died")
    with phase_state.timed_segment(target_dir, "coverage"):
        time.sleep(0.01)

    outcomes = [s["outcome"] for s in _segments(target_dir)]
    assert outcomes == [phase_state.SEGMENT_CRASHED, phase_state.SEGMENT_RAN]
    ran = [s["seconds"] for s in _segments(target_dir)
           if s["outcome"] == phase_state.SEGMENT_RAN]
    assert phase_state.output_seconds(target_dir, "coverage") == pytest.approx(
        round(sum(ran), 3))
    assert _segments(target_dir)[0]["seconds"] > 0


def test_a_failed_attempt_is_not_the_time_of_what_its_block_holds(target_dir):
    """A coverage attempt that ends failed without raising leaves nothing
    the next attempt keeps, so its segment closes as crashed. The starter's
    failed attempt keeps its generations, so its time stays ran."""
    from testuary import target_phases as phases

    for phase, outcome in (("coverage", phase_state.SEGMENT_CRASHED), ("starter", phase_state.SEGMENT_RAN)):
        with phase_state.timed_segment(target_dir, phase):
            phases._record_outcome(phases.TestPipelineResult(), phase, "failed",
                                   "the request limit ended before a measurement")
        assert _segments(target_dir, phase)[-1]["outcome"] == outcome
    assert phase_state.output_seconds(target_dir, "coverage") is None


def test_a_run_holding_a_killed_pipelines_phase_has_no_target_time(target_dir):
    """A killed invocation's pipeline closes as abandoned, its time unknown.
    The resume restores the phase that pipeline produced, so the run's own
    time is unknown too: None, never the resume's part of it."""
    phase_state.open_segment(target_dir, phase_state.SEGMENT_PIPELINE)
    with phase_state.timed_segment(target_dir, "coverage"):
        time.sleep(0.01)
    # The kill leaves the pipeline open. The resume's pipeline abandons it,
    # restores the coverage phase and closes.
    with phase_state.timed_segment(target_dir, phase_state.SEGMENT_PIPELINE):
        phase_state.record_restored(target_dir, "coverage")

    assert [s["outcome"] for s in _segments(target_dir, phase_state.SEGMENT_PIPELINE)] == [
        phase_state.SEGMENT_ABANDONED, phase_state.SEGMENT_RAN]
    assert phase_state.target_seconds(target_dir) is None
    assert "other" not in phase_state.target_time_parts(target_dir)


@pytest.mark.parametrize("before, counted", [("kill", False), ("crash", False), (None, True)])
def test_a_starter_run_again_with_its_saved_suites_counts_only_beside_counted_pipelines(
        target_dir, before, counted):
    """A kill or a crash during the starter phase leaves its pipeline
    uncounted. The resume runs the starter again with the suites that
    pipeline saved, so part of the starter's work is of unknown time and
    the run has no total, never the resume's part of it. A starter that
    reuses saved suites beside counted pipelines only keeps its time."""
    if before == "kill":
        phase_state.open_segment(target_dir, phase_state.SEGMENT_PIPELINE)
        phase_state.open_segment(target_dir, "starter")
    elif before == "crash":
        with pytest.raises(RuntimeError):
            with phase_state.timed_segment(target_dir, phase_state.SEGMENT_PIPELINE):
                with phase_state.timed_segment(target_dir, "starter"):
                    raise RuntimeError("the guard saw the starter crash")
    with phase_state.timed_segment(target_dir, phase_state.SEGMENT_PIPELINE):
        with phase_state.timed_segment(target_dir, "starter", suites_reused=True):
            time.sleep(0.01)

    assert (phase_state.target_seconds(target_dir) is not None) == counted


def test_a_failed_attempts_parts_stay_in_the_runs_time(target_dir):
    """A coverage attempt that failed counts for nothing in the phase's own
    time, yet the run spent that time: its parts stay in the run's parts
    while its pipeline counts, beside the attempt that followed."""
    with phase_state.timed_segment(target_dir, phase_state.SEGMENT_PIPELINE):
        with phase_state.timed_segment(target_dir, "coverage"):
            phase_state.record_time(phase_state.TIME_MODEL, 0.5)
            phase_state.discard_segment("coverage")
        with phase_state.timed_segment(target_dir, "coverage"):
            phase_state.record_time(phase_state.TIME_MODEL, 0.25)

    assert phase_state.output_time_parts(target_dir, "coverage") == {"model": 0.25}
    assert phase_state.target_time_parts(target_dir)["model"] == 0.75
    assert phase_state.unknown_times(target_dir) == frozenset()


@pytest.mark.parametrize("before, unknown", [
    ("kill", {"target", "starter"}), ("crash", {"target", "starter"}),
    ("crash_alone", {"target"}), (None, set())])
def test_a_rows_unknown_times_are_written_empty(target_dir, before, unknown):
    """A row's times are empty where they are not known, never partial: the
    run's total and every part when its repetition crashed or holds work an
    uncounted pipeline left, and the starter's own time when it ran again
    with suites such a pipeline saved. A preserved row cannot refill them."""
    from testuary import results

    if before == "kill":
        phase_state.open_segment(target_dir, phase_state.SEGMENT_PIPELINE)
        phase_state.open_segment(target_dir, "starter")
    elif before in ("crash", "crash_alone"):
        with pytest.raises(RuntimeError):
            with phase_state.timed_segment(target_dir, phase_state.SEGMENT_PIPELINE):
                with phase_state.timed_segment(target_dir, "starter"):
                    raise RuntimeError("the guard saw the starter crash")
    if before != "crash_alone":
        with phase_state.timed_segment(target_dir, phase_state.SEGMENT_PIPELINE):
            with phase_state.timed_segment(target_dir, "starter", suites_reused=True):
                phase_state.record_time(phase_state.TIME_MODEL, 0.01)
    earlier = {"duration_s_target": 99.0, "starter_duration_s": 9.0, "time_model_s": 3.0,
               "time_other_s": 1.0}
    row = dict(earlier)

    cleared = results._overlay_durable_durations(row, target_dir, run_idx=0)

    emptied = {key for key, value in row.items() if value == ""}
    expected = ({"duration_s_target", "time_model_s", "time_other_s"} if "target" in unknown else set())
    expected |= {"starter_duration_s"} if "starter" in unknown else set()
    assert expected <= emptied
    assert expected <= cleared and (not unknown or emptied <= cleared)
    assert phase_state.unknown_times(target_dir) == frozenset(unknown)


def test_repeated_kills_do_not_accumulate_open_segments(target_dir):
    for _ in range(4):
        phase_state.open_segment(target_dir, "coverage")

    still_open = [s for s in _segments(target_dir) if s["outcome"] == "open"]
    assert len(still_open) == 1
    assert phase_state.output_seconds(target_dir, "coverage") is None


def test_a_reused_phase_keeps_its_own_cost_and_says_so(target_dir):
    with phase_state.timed_segment(target_dir, "coverage"):
        pass
    produced = phase_state.output_seconds(target_dir, "coverage")

    phase_state.record_restored(target_dir, "coverage")

    assert phase_state.output_seconds(target_dir, "coverage") == produced
    assert _segments(target_dir)[-1]["outcome"] == phase_state.SEGMENT_RESTORED


def test_restarting_a_completed_phase_discards_its_earlier_time(target_dir):
    """Re-running a completed phase discards the artifact its time described,
    so nothing from before the restart may reappear in the new total."""
    _started(target_dir)
    with phase_state.timed_segment(target_dir, "coverage"):
        pass
    phase_state.mark_complete(target_dir, "coverage")
    before = phase_state.output_seconds(target_dir, "coverage")
    assert before is not None

    _started(target_dir)

    assert _segments(target_dir) == []
    assert phase_state.output_seconds(target_dir, "coverage") is None


def test_a_partial_in_phase_reuse_is_flagged(target_dir):
    """--reuse-starter-suites skips generation, so the phase's time is correct
    for that run but not comparable with a fresh one."""
    with phase_state.timed_segment(target_dir, "starter", suites_reused=True):
        pass

    assert _segments(target_dir, "starter")[0]["starter_suites_reused"] is True


def test_a_phase_that_never_ran_has_no_duration_rather_than_zero(target_dir):
    assert phase_state.output_seconds(target_dir, "mutation") is None


def test_closing_a_segment_leaves_mark_complete_the_last_write(target_dir):
    """The segment records time spent and the marker records completion.
    Closing the segment must not imply completion."""
    _started(target_dir)
    with phase_state.timed_segment(target_dir, "coverage"):
        pass

    state = json.loads(
        phase_state.state_path(target_dir).read_text(encoding="utf-8"))
    assert state["coverage"]["0"]["status"] == phase_state.IN_PROGRESS

    phase_state.mark_complete(target_dir, "coverage")
    state = json.loads(
        phase_state.state_path(target_dir).read_text(encoding="utf-8"))
    assert state["coverage"]["0"]["status"] == phase_state.COMPLETE
    assert phase_state.output_seconds(target_dir, "coverage") is not None


def test_an_unknown_phase_is_rejected_on_every_entry_point(target_dir):
    for call in (
        lambda: phase_state.open_segment(target_dir, "nope"),
        lambda: phase_state.close_segment(
            target_dir, "nope", outcome=phase_state.SEGMENT_RAN, seconds=1.0),
        lambda: phase_state.record_restored(target_dir, "nope"),
    ):
        with pytest.raises(ValueError, match="unknown phase"):
            call()


def test_attributed_parts_land_on_the_segment(target_dir):
    with phase_state.timed_segment(target_dir, "coverage"):
        phase_state.record_time(phase_state.TIME_MODEL, 0.40)
        phase_state.record_time(phase_state.TIME_TEST, 0.10)
        phase_state.record_time(phase_state.TIME_MODEL, 0.20)
        time.sleep(0.02)

    seg = _segments(target_dir)[0]
    assert seg["parts"] == {"model": 0.6, "test": 0.1}
    # The model figure is the attributed one when nothing set it explicitly.
    assert seg["model_seconds"] == 0.6


def test_the_residual_is_the_total_minus_the_parts(target_dir):
    with phase_state.timed_segment(target_dir, "coverage"):
        time.sleep(0.05)
        phase_state.record_time(phase_state.TIME_MUTATION, 0.01)

    seg = _segments(target_dir)[0]
    assert seg["other_seconds"] == pytest.approx(
        round(seg["seconds"] - 0.01, 3), abs=0.002)
    assert seg["parts"]["mutation"] + seg["other_seconds"] == (
        pytest.approx(seg["seconds"], abs=0.002))


def test_the_parts_never_exceed_the_total(target_dir):
    """The invariant that catches a span which stopped being a leaf."""
    with phase_state.timed_segment(target_dir, "coverage"):
        phase_state.record_time(phase_state.TIME_TEST, 999.0)

    seg = _segments(target_dir)[0]
    assert seg["other_seconds"] == 0.0


def test_recording_outside_a_segment_is_a_no_op(target_dir):
    """The chokepoints also run outside a phase, where recording does nothing."""
    phase_state.record_time(phase_state.TIME_TEST, 5.0)

    assert phase_state.output_seconds(target_dir, "coverage") is None


def test_an_unknown_part_is_rejected_rather_than_silently_dropped(target_dir):
    with pytest.raises(ValueError, match="unknown time part"):
        phase_state.record_time("gpu", 1.0)


def test_the_active_segment_is_released_even_when_the_block_raises(target_dir):
    with pytest.raises(RuntimeError):
        with phase_state.timed_segment(target_dir, "coverage"):
            raise RuntimeError("boom")

    # A leaked ContextVar would attribute the next phase's time to a segment
    # that is already closed.
    phase_state.record_time(phase_state.TIME_TEST, 1.0)
    assert _segments(target_dir)[0].get("parts") == {}


# The chokepoints: each names the operation it owns

def test_a_span_reached_through_another_of_its_kind_is_counted_once(target_dir):
    """An engine's one-call entry opens a session and runs inside it, so the
    outer span already holds the inner one's time."""
    with phase_state.timed_segment(target_dir, "mutation"):
        with phase_state.timed_part(phase_state.TIME_MUTATION):
            time.sleep(0.05)
            with phase_state.timed_part(phase_state.TIME_MUTATION):
                time.sleep(0.05)

    seg = _segments(target_dir, "mutation")[0]
    assert list(seg["parts"]) == ["mutation"]
    assert seg["parts"]["mutation"] == pytest.approx(seg["seconds"], abs=0.03)


def test_a_span_that_raises_still_reports_what_it_spent(target_dir):
    with phase_state.timed_segment(target_dir, "mutation"):
        with pytest.raises(RuntimeError):
            with phase_state.timed_part(phase_state.TIME_MUTATION):
                time.sleep(0.05)
                raise RuntimeError("engine died")

    assert _segments(target_dir, "mutation")[0]["parts"]["mutation"] > 0


def test_an_unknown_part_is_rejected_when_the_span_opens():
    """The check happens at the ``with``, not on exit, where it would mask
    the body's own exception."""
    with pytest.raises(ValueError, match="unknown time part"):
        phase_state.timed_part("gpu")


def test_a_test_run_is_charged_to_test_and_a_measured_one_to_coverage(
        target_dir, tmp_path, monkeypatch):
    """The two run categories split on whether the call measures coverage,
    which is per-call rather than per-backend. Each is what a run in place
    measures: the runner's own wall time, and for coverage also coverage.py's
    JSON report after it. The prepared copy around it is preparation."""
    from langadapter.python.adapter import PythonAdapter as ProjectPythonBackend
    from testuary.languages.generic import GenericBackend

    runs = []
    run_suite = ProjectPythonBackend.run_suite

    def kept(self, request):
        runs.append(run_suite(self, request))
        return runs[-1]

    monkeypatch.setattr(ProjectPythonBackend, "run_suite", kept)
    project = tmp_path / "proj"
    project.mkdir()
    (project / "calc.py").write_text(
        "def add(a, b):\n    return a + b\n", encoding="utf-8")
    suite = project / "test_calc.py"
    suite.write_text(
        "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n",
        encoding="utf-8")
    backend = GenericBackend("python", (".py",))

    with phase_state.timed_segment(target_dir, "coverage"):
        tested = backend.run_test_files([str(suite)], str(project), timeout=120)
    with phase_state.timed_segment(target_dir, "mutation"):
        measured = backend.run_test_files(
            [str(suite)], str(project), timeout=120,
            measure_coverage_of=str(project / "calc.py"))

    assert tested.ok and measured.ok
    test_run, coverage_run = runs
    assert test_run.coverage_report_s is None and coverage_run.coverage_report_s > 0
    for phase, part, run, seconds in (
            ("coverage", "test", tested, test_run.wall_time_s),
            ("mutation", "coverage", measured,
             coverage_run.wall_time_s + coverage_run.coverage_report_s)):
        parts = _segments(target_dir, phase)[0]["parts"]
        assert set(parts) == {part, "preparation"}
        assert run.elapsed_s == pytest.approx(runs[0 if part == "test" else 1].wall_time_s)
        assert parts[part] == round(seconds, 3)


def test_a_mutation_session_is_charged_to_mutation(target_dir):
    """Opening a session, its runs and closing it are mutation time."""
    from langadapter.mutation import MutationRun
    from langadapter.outcomes import Outcome
    from langadapter.python.adapter import PythonAdapter
    from testuary.languages.mutation import MutationRunner

    class _Session:
        outcome = Outcome.COMPLETED

        def run(self, request):
            time.sleep(0.05)
            return MutationRun.failed(
                "src.py", Outcome.INFRASTRUCTURE_FAILURE, "nothing ran",
                failure="initialization")

        def close(self):
            time.sleep(0.05)

    class _Backend:
        mutation_session_attempts = PythonAdapter.mutation_session_attempts

        @property
        def mutations(self):
            return self

        def open_session(self, engine, **options):
            time.sleep(0.05)
            return _Session()

    with phase_state.timed_segment(target_dir, "mutation"):
        with MutationRunner(
                _Backend(), {"cosmic_ray_operators": ""}).session(1, suite_dirs=()) as session:
            measured = session.run(
                target=target_dir / "src.py", test_dir=target_dir, test_files=["t.py"],
                per_mutant_timeout=10, wall_clock_timeout=60)

    assert measured.total == 0
    assert _segments(target_dir, "mutation")[0]["parts"]["mutation"] >= 0.14


def test_every_model_invocation_is_charged_to_model():
    """Checked over the whole module, so a new graph variant is covered."""
    import ast as _ast
    from pathlib import Path as _Path

    from testuary.agent import _langgraph

    tree = _ast.parse(
        _Path(_langgraph.__file__).read_text(encoding="utf-8"))
    timed = set()
    for node in _ast.walk(tree):
        if not isinstance(node, _ast.With):
            continue
        opens = {
            getattr(item.context_expr.func, "attr", None)
            for item in node.items
            if isinstance(item.context_expr, _ast.Call)
        }
        if "timed_part" not in opens:
            continue
        for inner in _ast.walk(node):
            if isinstance(inner, _ast.Call) and getattr(
                    inner.func, "attr", None) == "invoke":
                timed.add(inner.lineno)

    invocations = {
        node.lineno for node in _ast.walk(tree)
        if isinstance(node, _ast.Call)
        and getattr(node.func, "attr", None) == "invoke"
        and any(getattr(arg, "id", None) == "msgs_in" for arg in node.args)
    }
    assert invocations, (
        "no model invocation found; the search is looking at the wrong thing")
    assert invocations <= timed, (
        "model invocation(s) outside a timed span: "
        f"{sorted(invocations - timed)}")


def test_every_backend_charges_the_runs_it_makes():
    """Checked over every backend the package ships, since a new language
    inherits nothing here: each charges its suite runs in ``run_test_files``
    or in ``_suite_run_timer``, the hook the language backend's
    ``run_test_files`` holds."""
    import ast as _ast
    from pathlib import Path as _Path

    from testuary import languages as _languages

    package = _Path(_languages.__file__).parent
    checked = 0
    for path in sorted(package.glob("*.py")):
        tree = _ast.parse(path.read_text(encoding="utf-8"))
        for node in _ast.walk(tree):
            if not isinstance(node, _ast.FunctionDef):
                continue
            if node.name not in ("run_test_files", "_suite_run_timer"):
                continue
            body = _ast.dump(node)
            if len(node.body) <= 2 and ("Ellipsis" in body or "NotImplementedError" in body):
                continue  # the abstract declaration
            checked += 1
            charged = "timed_part" in body or "measured_part" in body or (
                node.name == "run_test_files" and "_suite_run_timer" in body)
            assert charged, (
                f"{path.name}: {node.name} does not charge its run")
    assert checked, "no backend implementation found"


def test_the_pipeline_is_a_segment_key_but_not_a_resumable_unit(target_dir):
    with phase_state.timed_segment(target_dir, phase_state.SEGMENT_PIPELINE):
        pass
    assert phase_state.target_seconds(target_dir) is not None

    with pytest.raises(ValueError, match="unknown phase"):
        phase_state.mark_complete(target_dir, phase_state.SEGMENT_PIPELINE)


def test_the_target_breakdown_puts_the_work_between_phases_under_other(target_dir):
    """The phases hold the leaf time. What the pipeline spent beyond them is
    the framework's own share and is named rather than lost."""
    with phase_state.timed_segment(target_dir, phase_state.SEGMENT_PIPELINE):
        with phase_state.timed_segment(target_dir, "coverage"):
            phase_state.record_time(phase_state.TIME_MODEL, 0.05)
        time.sleep(0.10)

    parts = phase_state.target_time_parts(target_dir)
    assert parts["model"] == 0.05
    assert parts["other"] == pytest.approx(
        phase_state.target_seconds(target_dir) - 0.05, abs=0.03)
    assert parts["other"] > 0.05


def test_the_breakdown_counts_a_discarded_attempt_no_more_than_the_total_does(
        target_dir):
    with pytest.raises(RuntimeError):
        with phase_state.timed_segment(target_dir, "coverage"):
            phase_state.record_time(phase_state.TIME_MODEL, 5.0)
            raise RuntimeError("boom")
    with phase_state.timed_segment(target_dir, "coverage"):
        phase_state.record_time(phase_state.TIME_MODEL, 1.0)

    assert phase_state.output_time_parts(target_dir, "coverage") == {"model": 1.0}


def test_each_run_has_its_own_breakdown(target_dir):
    """A run's row reports its own work: another run's time is in
    its own breakdown, not added to this one."""
    with phase_state.timed_segment(target_dir, "mutation", run_idx=0):
        phase_state.record_time(phase_state.TIME_MUTATION, 1.0)
    with phase_state.timed_segment(target_dir, "mutation", run_idx=1):
        phase_state.record_time(phase_state.TIME_MUTATION, 2.0)

    assert phase_state.target_time_parts(target_dir, run_idx=0)["mutation"] == 1.0
    assert phase_state.target_time_parts(target_dir, run_idx=1)["mutation"] == 2.0


def test_the_files_shared_work_is_apart_from_every_run(target_dir):
    """The file segment spans the file's work, each run's pipeline segment
    that run's own. What the file spent outside its runs is reported once,
    and for one run it and the run's time add up to the file's."""
    with phase_state.timed_segment(target_dir, phase_state.SEGMENT_FILE):
        time.sleep(0.05)
        for run_idx in (0, 1):
            with phase_state.timed_segment(target_dir, phase_state.SEGMENT_PIPELINE, run_idx=run_idx):
                with phase_state.timed_segment(target_dir, "coverage", run_idx=run_idx):
                    phase_state.record_time(phase_state.TIME_MODEL, 0.01 * (run_idx + 1))
                time.sleep(0.02)

    first = phase_state.target_seconds(target_dir, run_idx=0)
    second = phase_state.target_seconds(target_dir, run_idx=1)
    total = phase_state.output_seconds(target_dir, phase_state.SEGMENT_FILE)
    shared = phase_state.shared_seconds(target_dir)
    assert shared >= 0.05
    assert shared == pytest.approx(total - first - second, abs=0.002)
    assert phase_state.target_time_parts(target_dir, run_idx=1)["model"] == 0.02
    assert phase_state.target_time_parts(target_dir, run_idx=1)["other"] == pytest.approx(
        second - 0.02, abs=0.002)


def test_no_file_segment_means_no_shared_figure(target_dir):
    with phase_state.timed_segment(target_dir, phase_state.SEGMENT_PIPELINE):
        pass
    assert phase_state.shared_seconds(target_dir) is None


def test_a_measured_part_records_what_the_operation_measured(target_dir):
    """A run in a prepared copy is charged the runner's own wall time, as a
    run in place is timed, not the span around it."""
    with phase_state.timed_segment(target_dir, "coverage"):
        with phase_state.measured_part(phase_state.TIME_TEST) as charge:
            with charge.timing():
                time.sleep(0.05)
                charge.observed(0.01)

    assert _segments(target_dir)[0]["parts"] == {"test": 0.01}


@pytest.mark.parametrize("raises", [False, True])
def test_a_measured_part_without_a_measurement_records_its_elapsed_time(target_dir, raises):
    # The sleep is measured with the clock the part uses, and rounded as the
    # record rounds: on Windows a sleep can end just short of its nominal length.
    with phase_state.timed_segment(target_dir, "coverage"):
        with pytest.raises(RuntimeError) if raises else contextlib.nullcontext():
            with phase_state.measured_part(phase_state.TIME_COVERAGE) as charge:
                with charge.timing():
                    started = time.perf_counter()
                    time.sleep(0.05)
                    slept = time.perf_counter() - started
                    if raises:
                        raise RuntimeError("runner died")
                    charge.observed(None)

    assert _segments(target_dir)[0]["parts"]["coverage"] >= round(slept, 3) > 0.04


def test_a_measured_part_inside_an_open_span_of_its_kind_is_counted_once(target_dir):
    with phase_state.timed_segment(target_dir, "coverage"):
        with phase_state.timed_part(phase_state.TIME_TEST):
            assert phase_state.timing_open()
            with phase_state.measured_part(phase_state.TIME_TEST) as charge:
                charge.observed(5.0)

    assert _segments(target_dir)[0]["parts"]["test"] < 5.0
    assert not phase_state.timing_open()


def test_preparation_is_a_part_of_its_own():
    assert phase_state.TIME_PREPARATION in phase_state.TIME_PARTS
    with pytest.raises(ValueError, match="unknown time part"):
        phase_state.measured_part("gpu")


def test_a_span_hands_over_what_its_operation_did_besides_its_own(target_dir):
    """A mutation run's check of where the tests import the target from is
    preparation, not mutation time."""
    with phase_state.timed_segment(target_dir, "mutation"):
        with phase_state.timed_part(phase_state.TIME_MUTATION) as span:
            time.sleep(0.1)
            span.hand_over(phase_state.TIME_PREPARATION, 0.04)
            span.hand_over(phase_state.TIME_PREPARATION, None)
    parts = _segments(target_dir, "mutation")[0]["parts"]
    assert parts["preparation"] == 0.04 and parts["mutation"] >= 0.05


def test_a_span_never_hands_over_more_than_it_held(target_dir):
    with phase_state.timed_segment(target_dir, "mutation"):
        with phase_state.timed_part(phase_state.TIME_MUTATION) as span:
            span.hand_over(phase_state.TIME_PREPARATION, 60.0)
    parts = _segments(target_dir, "mutation")[0]["parts"]
    assert parts["preparation"] < 1.0 and parts.get("mutation", 0.0) == 0.0


def test_a_span_inside_one_of_its_kind_hands_over_nothing(target_dir):
    with phase_state.timed_segment(target_dir, "mutation"):
        with phase_state.timed_part(phase_state.TIME_MUTATION):
            with phase_state.timed_part(phase_state.TIME_MUTATION) as inner:
                inner.hand_over(phase_state.TIME_PREPARATION, 0.04)
    assert set(_segments(target_dir, "mutation")[0]["parts"]) == {"mutation"}
