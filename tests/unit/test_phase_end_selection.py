"""What a phase end selects among its distinctly named candidates.

The candidates are the merged suite, the reduced suite the cover derived from
it, and the phase's own inputs. The selection orders them by score first, then
the reduced suite, then timing.
"""

import pytest

from testuary.consolidation import (
    TiedCandidateTimer,
    PhaseConsolidation,
    ScoredSuite,
    merged_scored_from_phase,
    phase_end_candidates,
    pick_phase_result,
    record_selection_audit,
    scored_from_phase,
    _PHASE_END_AXES,
)

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _suite(label, *, mutation=None, branch=None, line=None, timing=1000.0,
           green=True, tests=10):
    return ScoredSuite(
        label=label, source=f"# {label}\n", line=line, branch=branch,
        mutation=mutation, runtime_ms=timing, green=green, test_count=tests,
    )


def _consolidation(**kw):
    base = dict(
        merged_source="# merge\n", reduced_source="# reduced\n",
        reduction_valid=True,
        merged_km=_km(1, 2), reduced_km=_km(1, 2),
        line=80.0, branch=70.0, union_line=80.0, union_branch=70.0,
        runtime_ms=500.0, pool_runtime_ms=900.0,
        pool_tests=20, selected_tests=8,
    )
    base.update(kw)
    return PhaseConsolidation(**base)


class TestNamedApart:
    def test_the_two_suites_are_separate_candidates(self):
        phase = _consolidation()
        labels = [c.label for c in phase_end_candidates(phase)]
        assert labels == ["reduced", "merged"]

    @pytest.mark.parametrize("order", [1, -1])
    def test_a_full_tie_goes_to_the_reduced_suite_in_either_order(self, order):
        """A tie on every score goes to the reduced suite by the selection's
        preference, not by the order of the candidates."""
        phase = _consolidation()
        winner = pick_phase_result(phase_end_candidates(phase)[::order],
                                   required_axes=_PHASE_END_AXES)
        assert winner.label == "reduced"

    def test_a_merged_suite_that_could_not_be_reduced_leaves_only_the_merged_suite(self):
        phase = _consolidation(
            reduced_source="# merge\n", reduction_valid=False
        )
        candidates = phase_end_candidates(phase)
        assert [c.label for c in candidates if c.green] == ["merged"]

    def test_a_reduction_never_built_is_absent_with_the_phase_end_s_detail(self):
        """A reduction the phase end never built is absent, not a failing or
        incomplete suite, and its census row names no tests, hash, coverage
        or time, where the phase holds the merged suite's numbers."""
        phase = _consolidation(reduced_source=None, reduction_valid=False,
                               detail="the merged suite's per-test evidence is unavailable")
        reduced = scored_from_phase(phase)
        assert (reduced.status, reduced.status_reason) == (
            "absent", "the merged suite's per-test evidence is unavailable")
        assert (reduced.test_count, reduced.removed_tests) == (None, ())
        record_selection_audit(phase, phase_end_candidates(phase), None, required_axes=_PHASE_END_AXES)
        row = phase.selection_audit["candidates"][0]
        assert (row["status"], row["source_sha256"], row["tests"]) == ("absent", None, None)
        assert (row["line"], row["branch"], row["runtime_ms"]) == (None, None, None)

    def test_an_excluded_reduction_names_its_issue(self):
        phase = _consolidation(reduction_valid=False, selection_audit={
            "reduction_measurement_issue": "reduced_coverage_refresh_failed"})
        reduced = scored_from_phase(phase)
        assert (reduced.status, reduced.status_reason) == (
            "incomplete", "the coverage tool could not measure the reduced suite")
        phase.selection_audit["reduction_measurement_issue"] = "reduced_failing"
        assert scored_from_phase(phase).status == "failing"

    def test_an_unmeasurable_merge_leaves_no_consolidation_candidate(self):
        phase = PhaseConsolidation(detail="the merge broke")
        assert [c.label for c in phase_end_candidates(phase) if c.green] == []

    def test_the_merged_suite_carries_the_merged_measurement_not_the_reduced_one(self):
        phase = _consolidation(line=60.0, branch=50.0,
                               union_line=80.0, union_branch=70.0)
        merged = merged_scored_from_phase(phase)
        assert (merged.line, merged.branch) == (80.0, 70.0)
        assert merged.runtime_ms == 900.0
        assert merged.test_count == 20
        assert scored_from_phase(phase).line == 60.0


class TestOrder:
    @pytest.mark.parametrize("axis", ["mutation", "branch", "line"])
    def test_a_strict_win_on_any_axis_selects_that_candidate(self, axis):
        base = dict(mutation=50.0, branch=50.0, line=50.0)
        ahead = dict(base)
        ahead[axis] = 60.0
        # The winner is the slowest candidate, so timing cannot explain it.
        winner = pick_phase_result([
            _suite("reduced", timing=10.0, **base),
            _suite("merged", timing=10.0, **base),
            _suite("agent", timing=9_000.0, **ahead),
        ], required_axes=_PHASE_END_AXES)
        assert winner.label == "agent"

    def test_mutation_outranks_branch_and_line(self):
        winner = pick_phase_result([
            _suite("reduced", mutation=40.0, branch=90.0, line=90.0),
            _suite("agent", mutation=41.0, branch=10.0, line=10.0),
        ], required_axes=_PHASE_END_AXES)
        assert winner.label == "agent"

    def test_on_a_three_way_score_tie_the_reduced_suite_is_delivered(self):
        # The reduced suite is the slowest of the three, so a timing term
        # anywhere in the tie would pick something else.
        winner = pick_phase_result([
            _suite("agent", mutation=50.0, branch=50.0, line=50.0,
                   timing=10.0),
            _suite("merged", mutation=50.0, branch=50.0, line=50.0,
                   timing=20.0),
            _suite("reduced", mutation=50.0, branch=50.0, line=50.0,
                   timing=9_000.0),
        ], required_axes=_PHASE_END_AXES)
        assert winner.label == "reduced"

    def test_without_a_reduced_suite_the_tie_goes_to_timing(self):
        measured = {"merged": 800.0, "agent": 200.0}
        winner = pick_phase_result([
            _suite("merged", mutation=50.0, branch=50.0, line=50.0,
                   timing=800.0),
            _suite("agent", mutation=50.0, branch=50.0, line=50.0,
                   timing=200.0),
        ], runtime_of=lambda candidate: measured[candidate.label], required_axes=_PHASE_END_AXES)
        assert winner.label == "agent"

    def test_a_reduced_suite_that_lost_evidence_loses_to_the_merged_suite(self):
        winner = pick_phase_result([
            _suite("reduced", mutation=49.0, branch=50.0, line=50.0,
                   timing=10.0),
            _suite("merged", mutation=50.0, branch=50.0, line=50.0,
                   timing=9_000.0),
        ], required_axes=_PHASE_END_AXES)
        assert winner.label == "merged"

    def test_a_non_green_candidate_cannot_ship(self):
        winner = pick_phase_result([
            _suite("reduced", mutation=99.0, branch=99.0, line=99.0,
                   green=False),
            _suite("merged", mutation=1.0, branch=1.0, line=1.0),
        ], required_axes=_PHASE_END_AXES)
        assert winner.label == "merged"

    def test_a_mutation_gain_is_not_vetoed_by_lower_branch_coverage(self):
        winner = pick_phase_result([
            _suite("reduced", mutation=90.0, branch=10.0, line=90.0),
            _suite("baseline", mutation=50.0, branch=50.0, line=50.0),
        ], required_axes=_PHASE_END_AXES)
        assert winner.label == "reduced"


def _km(killed, total, prefix="m"):
    """A kill matrix scoring ``killed / total``, with stable mutant ids so a
    requirement can name the ones that must survive into a candidate."""
    return {"mutants": [
        {"id": f"{prefix}{i}",
         "status": "killed" if i < killed else "survived"}
        for i in range(total)
    ]}



class TestMeasuredTieBreak:
    """The timing tie-break: with no reduced suite among the tied candidates,
    their run time decides, measured rather than read off one sample."""

    def _tied(self):
        return [
            _suite("baseline", mutation=50.0, branch=50.0, line=50.0,
                   timing=10.0),
            _suite("agent", mutation=50.0, branch=50.0, line=50.0,
                   timing=9_000.0),
        ]

    def test_the_measured_time_decides_not_the_recorded_one(self):
        # The recorded times say baseline. The measurement says agent.
        measured = {"baseline": 900.0, "agent": 100.0}
        candidates = self._tied()
        winner = pick_phase_result(
            candidates, runtime_of=lambda c: measured[c.label], required_axes=_PHASE_END_AXES)
        assert winner.label == "agent"
        assert {candidate.label: candidate.runtime_ms
                for candidate in candidates} == measured

    def test_the_audit_records_how_a_timing_tie_was_resolved(self):
        candidates = self._tied()
        measured = {"baseline": 900.0, "agent": 100.0}
        winner = pick_phase_result(
            candidates, runtime_of=lambda c: measured[c.label], required_axes=_PHASE_END_AXES)
        phase = PhaseConsolidation()

        record_selection_audit(phase, candidates, winner, required_axes=_PHASE_END_AXES)

        assert phase.selection_audit["selection_basis"] == "timing_median"
        assert phase.selection_audit["top_score_labels"] == [
            "baseline", "agent"
        ]
        census = {
            candidate["label"]: candidate
            for candidate in phase.selection_audit["candidates"]
        }
        assert census["baseline"]["runtime_ms"] == 900.0
        assert census["agent"]["runtime_ms"] == 100.0
        assert phase.selection_audit["runtime_tie_break_basis"] == "median"
        assert phase.selection_audit["runtime_tie_break_candidate_order"] == [
            "baseline", "agent"
        ]

    def test_the_audit_records_reduced_preference_without_timing(self):
        candidates = [*self._tied(), _suite(
            "reduced", mutation=50.0, branch=50.0, line=50.0
        )]
        winner = pick_phase_result(candidates, required_axes=_PHASE_END_AXES)
        phase = PhaseConsolidation()

        record_selection_audit(phase, candidates, winner, required_axes=_PHASE_END_AXES)

        assert phase.selection_audit["selection_basis"] == (
            "reduced_preference"
        )
        assert phase.selection_audit["runtime_tie_break_basis"] is None
        assert phase.selection_audit["runtime_tie_break_candidate_order"] is None
        assert phase.selection_audit["top_score_labels"] == [
            "baseline", "agent", "reduced"
        ]

    def test_unmeasured_timing_is_recorded_as_json_null(self):
        candidates = [_suite(
            "reduced", mutation=50.0, branch=50.0, line=50.0,
            timing=None,
        )]
        phase = PhaseConsolidation()

        record_selection_audit(phase, candidates, candidates[0], required_axes=_PHASE_END_AXES)

        assert phase.selection_audit["candidates"][0]["runtime_ms"] is None

    def test_a_green_incomplete_candidate_is_not_fully_measured(self):
        """A candidate whose numbers a tool left incomplete is no candidate
        of the selection, so the audit does not call it fully measured."""
        incomplete = _suite("agent", mutation=50.0, branch=50.0, line=50.0)
        incomplete.status = "incomplete"
        complete = _suite("baseline", mutation=40.0, branch=40.0, line=40.0)
        phase = PhaseConsolidation()

        record_selection_audit(phase, [complete, incomplete], complete, required_axes=_PHASE_END_AXES)

        census = {c["label"]: c for c in phase.selection_audit["candidates"]}
        assert census["agent"]["fully_measured"] is False
        assert census["baseline"]["fully_measured"] is True

    def test_every_tied_candidate_is_measured_exactly_once(self):
        seen = []

        def runtime_of(c):
            seen.append(c.label)
            return 1.0 if c.label == "agent" else 2.0

        pick_phase_result(self._tied(), runtime_of=runtime_of, required_axes=_PHASE_END_AXES)
        assert sorted(seen) == ["agent", "baseline"]

    def test_a_candidate_that_is_not_tied_is_never_measured(self):
        seen = []

        def runtime_of(c):
            seen.append(c.label)
            return 1.0

        winner = pick_phase_result([
            *self._tied(),
            _suite("ahead", mutation=90.0, branch=50.0, line=50.0),
        ], runtime_of=runtime_of, required_axes=_PHASE_END_AXES)
        assert winner.label == "ahead"
        assert seen == []

    def test_the_reduced_suite_ships_without_any_measurement(self):
        seen = []

        def runtime_of(c):
            seen.append(c.label)
            return 1.0

        winner = pick_phase_result([
            *self._tied(),
            _suite("reduced", mutation=50.0, branch=50.0, line=50.0,
                   timing=9_999.0),
        ], runtime_of=runtime_of, required_axes=_PHASE_END_AXES)
        assert winner.label == "reduced"
        assert seen == []

    def test_a_measured_tie_breaks_by_input_order(self):
        winner = pick_phase_result(self._tied(), runtime_of=lambda c: 5.0, required_axes=_PHASE_END_AXES)
        assert winner.label == "baseline"

    def _timer(self, monkeypatch, results, logged):
        """A timer whose interleaved measurement gives ``results``."""
        import testuary.consolidation as consolidation

        monkeypatch.setattr(consolidation, "measure_interleaved", lambda *a, **kw: {
            "status": "measured", "rounds": [], "results": results})
        return TiedCandidateTimer(
            None, source_dir=".", target_file_path="mod.py", timeout=60,
            warmup_runs=1, benchmark_runs=3, stage_as_delivered=False,
            log=logged.append)

    def test_a_failed_timing_run_gives_no_time_and_the_input_order_decides(
            self, monkeypatch):
        """Timing only adds a time: a tied candidate whose timing run fails
        gets no time and a warning, is not rejected, and the tie is decided
        by the input order, as without a timing benchmark."""
        logged = []
        failed = {"status": "failed", "failure_kind": "test_failure",
                  "returncode": 1, "error": "1 failed"}
        runtime_of = self._timer(monkeypatch, {
            "baseline": failed,
            "agent": {"status": "measured", "samples_ms": [1.0], "median_ms": 1.0,
                      "warmup_samples_ms": []}}, logged)
        candidates = self._tied()
        winner = pick_phase_result(candidates, runtime_of=runtime_of, required_axes=_PHASE_END_AXES)
        assert winner.label == "baseline"
        assert [c.green for c in candidates] == [True, True]
        assert [c.status for c in candidates] == ["passing", "passing"]
        assert candidates[0].suite_timing["status"] == "failed"
        assert any("WARNING" in line and "'baseline'" in line for line in logged)

    def test_a_tie_without_a_timing_benchmark_records_no_times(self, monkeypatch):
        """Without a timing benchmark no tied candidate has a time: the input
        order decides, and the census records null, not zero or a tied
        candidate's provisional time."""
        import testuary.consolidation as consolidation

        monkeypatch.setattr(consolidation, "measure_interleaved", lambda *a, **kw: {
            "status": "unavailable", "error": "no benchmark"})
        runtime_of = TiedCandidateTimer(
            None, source_dir=".", target_file_path="mod.py", timeout=60,
            warmup_runs=1, benchmark_runs=3, stage_as_delivered=False, log=[].append)
        candidates = [_suite(label, mutation=50.0, branch=50.0, line=50.0, timing=time)
                      for label, time in (("baseline", 900.0), ("agent", 1200.0))]
        winner = pick_phase_result(candidates, runtime_of=runtime_of, required_axes=_PHASE_END_AXES)
        phase = PhaseConsolidation()

        record_selection_audit(phase, candidates, winner, required_axes=_PHASE_END_AXES)

        assert winner.label == "baseline"
        assert [c["runtime_ms"] for c in phase.selection_audit["candidates"]] == [None, None]
        assert phase.selection_audit["runtime_tie_break_basis"] == "input_order"

    def test_a_tie_whose_timing_runs_all_fail_still_has_a_result(self, monkeypatch):
        failed = {"status": "failed", "failure_kind": "test_failure",
                  "returncode": 1, "error": "1 failed"}
        runtime_of = self._timer(
            monkeypatch, {"baseline": failed, "agent": failed}, [])
        assert pick_phase_result(self._tied(), runtime_of=runtime_of, required_axes=_PHASE_END_AXES).label == "baseline"

    def test_an_incomplete_candidate_is_no_choice(self):
        """A candidate a tool could not measure is incomplete: even with
        scores recorded from an earlier measurement it is not chosen."""
        unmeasured = _suite("agent", mutation=90.0, branch=90.0, line=90.0)
        unmeasured.status = "incomplete"
        complete = _suite("baseline", mutation=50.0, branch=50.0, line=50.0)
        complete.status = "passing"
        assert pick_phase_result([unmeasured, complete], required_axes=_PHASE_END_AXES).label == "baseline"

    def test_an_infrastructure_benchmark_failure_still_aborts(self):
        def runtime_of(_candidate):
            raise RuntimeError("measurement infrastructure failed")

        with pytest.raises(RuntimeError, match="infrastructure failed"):
            pick_phase_result(self._tied(), runtime_of=runtime_of, required_axes=_PHASE_END_AXES)

    def test_without_a_measurement_the_tie_fails_explicitly(self):
        with pytest.raises(RuntimeError, match="no suite timing"):
            pick_phase_result(self._tied(), required_axes=_PHASE_END_AXES)



class TestADisplacedReduction:
    """A merged suite selected over a distinct reduction is only recorded as
    selected. The reduction keeps its own fields and evidence rather than
    taking the merged suite's."""

    def _merged_wins(self, **kw):
        phase = _consolidation(
            per_test_tokens={"t::a": ["L:1"], "t::b": ["L:2"]},
            kept_tests=["t::a"], selected_tests=1, pool_tests=2,
            line=60.0, branch=50.0, reduced_km=_km(1, 3, "r"),
            merged_km=_km(2, 3, "m"), **kw,
        )
        merged = _suite("merged", mutation=66.0, branch=70.0, line=80.0)
        return phase, merged

    def test_the_audit_records_that_the_reduction_was_not_selected(self):
        phase, merged = self._merged_wins()

        record_selection_audit(phase, [merged], merged, required_axes=_PHASE_END_AXES)

        assert phase.selection_audit["reduced_not_selected"] is True
        assert phase.selection_audit["merged_stats"] is not None
        assert phase.fallback_reason is None
        assert phase.fallback_candidates == []

    def test_the_reductions_own_evidence_is_left_as_measured(self):
        phase, merged = self._merged_wins()
        reduced_km = phase.reduced_km

        record_selection_audit(phase, [merged], merged, required_axes=_PHASE_END_AXES)

        assert phase.reduced_source == "# reduced\n"
        assert phase.reduced_km is reduced_km
        assert phase.kept_tests == ["t::a"]
        assert phase.selected_tests == 1
        assert (phase.line, phase.branch) == (60.0, 50.0)
        assert phase.reduction_valid is True

    def test_a_cover_that_kept_every_test_displaces_nothing(self):
        phase, merged = self._merged_wins(reduced_source="# merge\n")

        record_selection_audit(phase, [merged], merged, required_axes=_PHASE_END_AXES)

        assert "reduced_not_selected" not in phase.selection_audit
        assert phase.fallback_reason is None

    def test_a_reduction_that_won_is_not_reported_as_displaced(self):
        phase, _merged = self._merged_wins()
        reduced = _suite("reduced", mutation=66.0, branch=70.0, line=80.0)

        record_selection_audit(phase, [reduced], reduced, required_axes=_PHASE_END_AXES)

        assert "reduced_not_selected" not in phase.selection_audit

    def test_an_already_named_reason_is_kept(self):
        phase, merged = self._merged_wins(fallback_reason="cover_skipped")

        record_selection_audit(phase, [merged], merged, required_axes=_PHASE_END_AXES)

        assert phase.fallback_reason == "cover_skipped"


class TestTheDeliveredCombination:
    """The tie-break tests inject a timing function. These drive the real one
    through the real selector."""

    def _tied(self):
        return [
            _suite("baseline", mutation=50.0, branch=50.0, line=50.0,
                   timing=10.0),
            _suite("agent", mutation=50.0, branch=50.0, line=50.0,
                   timing=20.0),
        ]

    def test_the_real_benchmark_reaches_the_selector_without_raising(self):
        """The tie-break's benchmark reaches choose_best's timing with the
        scored suites themselves, whose text a benchmark runs."""
        seen = []

        class _Backend:
            def benchmark_interleaved(
                self, *, suites, warmup_runs, benchmark_runs, **kw
            ):
                from pathlib import Path

                results = {}
                for label, test_file_paths in suites:
                    paths = [Path(path) for path in test_file_paths]
                    assert paths and all(path.is_file() for path in paths)
                    assert all(path.read_text(encoding="utf-8")
                               for path in paths)
                    seen.append(label)
                    median = 100.0 if label == "agent" else 900.0
                    results[label] = {
                        "ok": True,
                        "warmup_runs": warmup_runs,
                        "timing_run_count": benchmark_runs,
                        "warmup_times_ms": [median + 50.0] * warmup_runs,
                        "times_ms": [median - 1.0, median, median + 1.0],
                        "median_ms": median,
                    }
                return {"ok": True, "suites": results, "rounds": [
                    {"round": 1, "warmup": True, "order": ["baseline", "agent"]},
                ]}

            def scratch_test_path(self, directory, source, target, **kw):
                from pathlib import Path
                path = Path(directory) / "test_probe.py"
                path.write_text(source, encoding="utf-8")
                return path

        winner = pick_phase_result(
            self._tied(),
            runtime_of=TiedCandidateTimer(
                _Backend(), source_dir=".", target_file_path="mod.py",
                timeout=60, warmup_runs=1, benchmark_runs=3,
                stage_as_delivered=False, log=lambda _m: None), required_axes=_PHASE_END_AXES,
        )
        assert sorted(seen) == ["agent", "baseline"]
        assert winner.label == "agent"
        assert winner.suite_timing == {
            "warmup_runs": 1,
            "timing_run_count": 3,
            "samples_ms": [99.0, 100.0, 101.0],
            "median_ms": 100.0,
            "order": "interleaved",
            "batch": 1,
            "rounds": [
                {"round": 1, "warmup": True, "order": ["baseline", "agent"]},
            ],
        }

    def test_a_backend_without_a_benchmark_fails_explicitly(self):
        class _Bare:
            pass

        with pytest.raises(RuntimeError, match="benchmark_interleaved"):
            pick_phase_result(
                self._tied(),
                runtime_of=TiedCandidateTimer(
                    _Bare(), source_dir=".", target_file_path="mod.py",
                    timeout=60, warmup_runs=1, benchmark_runs=3,
                    stage_as_delivered=False, log=lambda _m: None), required_axes=_PHASE_END_AXES,
            )

    def test_a_red_backend_result_gives_the_candidate_no_time(self):
        class _Red:
            def scratch_test_path(self, directory, source, target, **kw):
                from pathlib import Path
                return Path(directory) / "test_probe.py"

            def benchmark_interleaved(self, *, suites, **kw):
                return {"ok": False, "rounds": [], "suites": {
                    label: {
                        "ok": False,
                        "failure_kind": "test_failure",
                        "returncode": 1,
                        "error": "1 failed",
                    }
                    for label, _paths in suites
                }}

        runtime_of = TiedCandidateTimer(
            _Red(), source_dir=".", target_file_path="mod.py",
            timeout=60, warmup_runs=1, benchmark_runs=3,
            stage_as_delivered=False, log=lambda _m: None,
        )
        candidate = self._tied()[0]

        assert runtime_of(candidate) is None
        assert candidate.suite_timing["status"] == "failed"

    def test_a_benchmark_exception_propagates(self):
        class _Broken:
            def scratch_test_path(self, directory, source, target, **kw):
                raise OSError("no scratch")

            def benchmark_interleaved(self, **kw):
                raise AssertionError("never reached")

        with pytest.raises(OSError, match="no scratch"):
            pick_phase_result(
                self._tied(),
                runtime_of=TiedCandidateTimer(
                    _Broken(), source_dir=".", target_file_path="mod.py",
                    timeout=60, warmup_runs=1, benchmark_runs=3,
                    stage_as_delivered=False, log=lambda _m: None), required_axes=_PHASE_END_AXES,
            )



def test_a_merged_suite_that_was_never_made_green_does_not_gate_the_winner():
    """A merged suite that removing failing tests could not make green was
    never measured. It does not count as a rejected acceptance run, so the
    winner is not benchmarked before it is accepted."""
    merged = _suite("merged", mutation=100.0, branch=100.0, line=100.0)
    merged.green = False
    merged.status = "incomplete"
    reduced = _suite("reduced", mutation=50.0, branch=50.0, line=50.0)
    benchmarked = []

    winner = pick_phase_result(
        [reduced, merged],
        runtime_of=lambda candidate: benchmarked.append(candidate.label) or 1.0, required_axes=_PHASE_END_AXES,
    )

    assert winner.label == "reduced"
    assert benchmarked == []


def test_the_required_measurements_are_named_from_the_phase_ends_axes():
    """A phase end without phase_end_mutation requires branch and line only,
    and the reason it gives for selecting nothing says so."""
    from testuary.consolidation import measurement_names

    assert measurement_names(("branch", "line")) == "branch and line"
    assert measurement_names(("mutation", "branch", "line")) == "mutation, branch and line"


# Phase-end selection and its separate evidence audit record.
def _s(
    label,
    *,
    mutation=50.0,
    branch=50.0,
    line=50.0,
    green=True,
    kills=("m1",),
    covered_lines=(1,),
    covered_branches=("1->2",),
):
    return ScoredSuite(
        label=label,
        source=f"# {label}\n",
        mutation=mutation,
        branch=branch,
        line=line,
        runtime_ms=100.0,
        green=green,
        kill_matrix={
            "mutants": [
                {"id": mutant, "status": "killed"}
                for mutant in kills
            ]
        } if kills is not None else None,
        covered_lines=(
            frozenset(covered_lines) if covered_lines is not None else None
        ),
        covered_branches=(
            frozenset(covered_branches)
            if covered_branches is not None else None
        ),
    )


def test_pick_phase_result_uses_score_order_without_evidence_filtering():
    winner = pick_phase_result([
        _s("baseline", mutation=50.0, kills=("m1",)),
        _s("agent", mutation=51.0, branch=1.0, line=1.0, kills=("m2",)),
    ], required_axes=_PHASE_END_AXES)
    assert winner.label == "agent"


def test_pick_phase_result_excludes_an_incomplete_candidate():
    winner = pick_phase_result([
        _s("incomplete", mutation=None, branch=100.0, line=100.0),
        _s("complete", mutation=1.0, branch=1.0, line=1.0),
    ], required_axes=_PHASE_END_AXES)
    assert winner.label == "complete"


def test_a_candidate_whose_mutation_run_the_wall_clock_cut_is_not_selected():
    """A candidate measured alone whose mutation run the wall clock cut has a
    matrix stamped incomplete and no score. It measures no mutation axis, so
    better coverage does not select it, while a complete catalogue without a
    valid mutant still measures the axis."""
    from testuary.consolidation import scored_from_measurement
    from testuary.languages import mutation as _km

    def candidate(label, kill_matrix, coverage):
        return scored_from_measurement(
            label=label, source=f"# {label}\n", kill_matrix=kill_matrix,
            coverage={"line": coverage, "branch": coverage}, execution_name="test_target.py",
            green=True, status="passing", runtime_ms=100.0)

    cut = candidate("candidate2", {"mutants": [], _km.INCOMPLETE_KEY: {
        "reason": "untested_mutants_in_catalogue", "untested": 3, "total": 3}}, 100.0)
    no_valid_mutant = candidate("baseline", {"mutants": []}, 50.0)
    assert cut.mutation is None and no_valid_mutant.mutation is None
    assert pick_phase_result([cut, no_valid_mutant], required_axes=_PHASE_END_AXES).label == "baseline"
    assert pick_phase_result([cut], required_axes=_PHASE_END_AXES) is None


def test_a_tied_reduced_suite_wins_without_timing_measurement():
    winner = pick_phase_result([
        _s("baseline"),
        _s("reduced"),
    ], required_axes=_PHASE_END_AXES)
    assert winner.label == "reduced"


def test_record_selection_audit_counts_identity_loss_without_vetoing_winner():
    phase = PhaseConsolidation()
    baseline = _s(
        "baseline", kills=("m1",), covered_lines=(1,),
        covered_branches=("1->2",),
    )
    agent = _s(
        "agent", mutation=51.0, kills=("m2",), covered_lines=(2,),
        covered_branches=("2->3",),
    )
    winner = pick_phase_result([baseline, agent], required_axes=_PHASE_END_AXES)
    record_selection_audit(phase, [baseline, agent], winner, required_axes=_PHASE_END_AXES)

    assert winner.label == "agent"
    assert phase.selection_audit["delivered_from"] == "agent"
    assert phase.selection_audit["delivered_from_missing_from_merged_input"] == {
        "kills": 1,
        "lines": 1,
        "branches": 1,
    }


def test_record_selection_audit_names_missing_input_measurements():
    phase = PhaseConsolidation()
    baseline = _s("baseline")
    agent = _s(
        "agent", kills=None, covered_lines=None, covered_branches=None,
    )
    record_selection_audit(phase, [baseline, agent], baseline, required_axes=_PHASE_END_AXES)
    assert phase.selection_audit["input_evidence_missing_measurements"] == {
        "kills": ["agent"],
        "lines": ["agent"],
        "branches": ["agent"],
    }
    assert phase.selection_audit["delivered_from_missing_from_merged_input"] == {
        "kills": None,
        "lines": None,
        "branches": None,
    }


def test_scored_from_phase_is_incomplete_without_a_mutation_matrix():
    phase = PhaseConsolidation(
        reduced_source="def test_x():\n    pass\n",
        reduction_valid=True,
        line=91.0,
        branch=80.0,
    )
    candidate = scored_from_phase(phase)
    assert candidate.green is True
    assert candidate.mutation is None
    assert pick_phase_result([candidate], required_axes=_PHASE_END_AXES) is None


def test_the_audit_names_the_axes_the_selection_compared():
    """A coverage-only selection records the axes it compared."""
    candidates = [_suite("reduced", mutation=None, branch=50.0, line=50.0)]
    phase = PhaseConsolidation()

    record_selection_audit(phase, candidates, candidates[0],
                           required_axes=("branch", "line"))

    assert phase.selection_audit["selector"] == "branch_then_line"
    assert phase.selection_audit["active_axes"] == ["branch", "line"]


def test_a_tie_without_timing_names_the_axes_it_compared():
    """A tie the selection cannot time names the axes it compared."""
    from testuary.consolidation import choose_best
    tied = [_suite(label, branch=50.0, line=50.0) for label in ("a", "b")]
    with pytest.raises(RuntimeError, match="tie on branch and line, but no suite timing"):
        choose_best(tied, required_axes=("branch", "line"))
