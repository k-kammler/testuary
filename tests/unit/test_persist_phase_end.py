"""``persist_phase_end``, the single phase-boundary persistence point: every
outcome branch leaves the boundary's evidence and its decision record on
disk, and a write that cannot complete raises at once."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from testuary.consolidation import (
    PhaseEndDelivery,
    PhaseConsolidation,
    persist_phase_end,
)

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _phase(**kw) -> PhaseConsolidation:
    p = PhaseConsolidation()
    for k, v in kw.items():
        setattr(p, k, v)
    return p


_KM_MERGED = {"mutants": [{"id": "m1", "status": "killed"},
                         {"id": "m2", "status": "survived"}]}
_KM_RED = {"mutants": [{"id": "m1", "status": "killed"}]}


def _names(d: Path) -> list:
    return sorted(p.name for p in d.iterdir())


def test_merged_winner_persists_merged_reduced_and_record(tmp_path) -> None:
    phase = _phase(merged_km=_KM_MERGED, reduced_km=_KM_RED, pool_tests=10,
                   per_test_tokens={"t1": ["L:1"]}, kept_tests=["t1"],
                   solver="greedy", mutation_method="both")
    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="merged", delivered_km=phase.merged_km,
            delivered_tests=10))
    # The delivered suite's matrix is written under the canonical name as
    # well as its own, and the reduction's matrix whatever won.
    assert _names(tmp_path) == [
        "consolidation_record.json", "merged_coverage_matrix.json",
        "merged_kill_matrix.json", "reduced_coverage_matrix.json",
        "reduced_kill_matrix.json", "selected_kill_matrix.json"]
    assert (tmp_path / "selected_kill_matrix.json").read_bytes() == (
        tmp_path / "merged_kill_matrix.json").read_bytes()
    rec = json.loads(
        (tmp_path / "consolidation_record.json").read_text(encoding="utf-8"))
    assert rec["delivered_from"] == "merged"
    assert rec["shipped_has_matrix"] is True
    assert rec["mutation_measured"] is True
    assert rec["reduction_solver"] == "greedy"
    assert rec["consolidation_mutation_method"] == "both"


def test_a_merged_suite_that_was_not_counted_records_null_not_zero(tmp_path) -> None:
    """A phase end that never counted its merged suite records merged_tests
    as null, and a counted empty pool as 0: a missing count is never 0."""
    for phase, recorded in ((PhaseConsolidation(), None), (_phase(pool_tests=0), 0)):
        out = tmp_path / str(recorded)
        persist_phase_end(
            matrices_dir=out, target_file="pkg/mod.py", phase=phase,
            delivery=PhaseEndDelivery(winner_label="unresolved", delivered_km=None))
        rec = json.loads((out / "consolidation_record.json").read_text(encoding="utf-8"))
        assert rec["merged_tests"] == recorded


def test_coverage_sidecars_keep_merged_and_reduced_aggregates_distinct(
    tmp_path,
) -> None:
    phase = _phase(
        pool_tests=2,
        per_test_tokens={"t1": ["L:1"], "t2": ["L:2"]},
        kept_tests=["t2"],
        union_line=100.0,
        union_branch=75.0,
        union_covered_lines=[1, 2],
        union_covered_branches=["1->2"],
        line=50.0,
        branch=0.0,
        suite_covered_lines=[2],
        suite_covered_branches=[],
    )

    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="merged", delivered_km=None,
            delivered_tests=1))

    merged = json.loads(
        (tmp_path / "merged_coverage_matrix.json").read_text("utf-8")
    )
    reduced = json.loads(
        (tmp_path / "reduced_coverage_matrix.json").read_text("utf-8")
    )
    assert merged["line"] == 100.0
    assert merged["covered_statement_lines"] == [1, 2]
    assert reduced["line"] == 50.0
    assert reduced["covered_statement_lines"] == [2]


def test_an_unmeasured_union_is_written_as_unmeasured_not_as_the_reduction(
    tmp_path,
) -> None:
    """A merged suite whose aggregate coverage was not measured keeps its
    sidecar's coverage empty: the reduced suite's coverage is never written
    in its place."""
    phase = _phase(
        pool_tests=2,
        per_test_tokens={"t1": ["L:1"], "t2": ["L:2"]},
        kept_tests=["t2"],
        line=50.0,
        branch=0.0,
        suite_covered_lines=[2],
        suite_covered_branches=[],
    )

    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="merged", delivered_km=None,
            delivered_tests=1))

    merged = json.loads(
        (tmp_path / "merged_coverage_matrix.json").read_text("utf-8")
    )
    reduced = json.loads(
        (tmp_path / "reduced_coverage_matrix.json").read_text("utf-8")
    )
    assert (merged["line"], merged["branch"]) == (None, None)
    assert merged["covered_statement_lines"] is None
    assert merged["covered_branch_arcs"] is None
    assert reduced["line"] == 50.0
    assert reduced["covered_statement_lines"] == [2]


def test_aggregate_merged_coverage_persists_without_eviction_tokens(
    tmp_path,
) -> None:
    phase = _phase(
        aggregate_coverage_measured=True,
        pool_tests=3,
        union_line=75.0,
        # No branch opportunities is still a successful coverage run.
        union_branch=None,
        union_covered_lines=[1, 2, 3],
        union_covered_branches=["1->2"],
        target_total_lines=4,
        target_total_branches=1,
        fallback_reason="eviction_data_unavailable",
        fallback_candidates=["merged_unreduced"],
        fallback_union_stats={"merged_tests": 3},
        selection_audit={
            "delivered_from": "merged",
            "candidates": [{"label": "merged", "tests": 3}],
        },
    )

    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="merged", delivered_km=None,
            delivered_tests=3))

    merged = json.loads(
        (tmp_path / "merged_coverage_matrix.json").read_text("utf-8")
    )
    record = json.loads(
        (tmp_path / "consolidation_record.json").read_text("utf-8")
    )
    assert merged["line"] == 75.0
    assert merged["branch"] is None
    assert merged["target_total_statements"] == 4
    assert merged["target_total_branches"] == 1
    assert merged["per_test_tokens"] is None
    assert record["merged_tests"] == 3
    assert "merged_coverage_matrix.json" in record["artifacts"]


def test_selected_input_coverage_is_bound_to_delivered_source(tmp_path) -> None:
    target = tmp_path / "pkg" / "mod.py"
    target.parent.mkdir()
    target.write_text("def value():\n    return 1\n", encoding="utf-8")
    source = "from pkg.mod import value\n\ndef test_value():\n    assert value() == 1\n"

    persist_phase_end(
        matrices_dir=tmp_path / "matrices", target_file=str(target), phase=_phase(),
        delivery=PhaseEndDelivery(winner_label="single_suite",
            winner_source_label="suite_001", delivered_km=None, delivered_tests=1,
            delivered_source=source, delivered_coverage={
            "line": 100.0,
            "branch": 100.0,
            "covered_statement_lines": [1, 2],
            "covered_branch_arcs": [],
            "per_test_tokens": {"test_value": ["L:1"]},
        }, execution_name="test_mod.py"))

    selected = json.loads(
        (tmp_path / "matrices" / "selected_coverage_matrix.json")
        .read_text("utf-8")
    )
    record = json.loads(
        (tmp_path / "matrices" / "consolidation_record.json")
        .read_text("utf-8")
    )
    assert selected["source_sha256"] == record["selected_source_sha256"]
    assert selected["execution_name"] == "test_mod.py"
    assert "selected_coverage_matrix.json" in record["artifacts"]


@pytest.mark.parametrize("winner", ["reduced", "merged", "suite_001"])
def test_unscoped_delivered_suite_records_unavailable_construction(tmp_path, winner) -> None:
    """A caller without a generation scope names no phase or invocation, so the
    delivered suite cannot claim construction inputs. Persistence records that as
    explicitly unavailable instead of refusing its own ancestry record."""
    target = tmp_path / "pkg" / "mod.py"
    target.parent.mkdir()
    target.write_text("def value():\n    return 1\n", encoding="utf-8")
    source = "from pkg.mod import value\n\ndef test_value():\n    assert value() == 1\n"

    persist_phase_end(
        matrices_dir=tmp_path / "matrices", target_file=str(target), phase=_phase(),
        delivery=PhaseEndDelivery(winner_label=winner, delivered_km=None, delivered_tests=1,
            delivered_source=source, execution_name="test_mod.py"))

    record = json.loads(
        (tmp_path / "matrices" / "consolidation_record.json").read_text("utf-8")
    )
    ancestry = record["generation_ancestry"]
    assert ancestry["operation"] == "unavailable"
    assert ancestry["status"] == "unavailable"
    assert ancestry["phase"] is None and ancestry["parents"] == []
    assert ancestry["selected_source_sha256"] == record["selected_source_sha256"]


def test_reduced_sidecar_carries_the_derive_kept_set(tmp_path) -> None:
    """The kept set the run-time derive restricted with persists beside the
    delivered suite's kept set, so an offline replay restricts with the same
    set the run did even after a fallback widened ``kept_tests``."""
    phase = _phase(
        pool_tests=2,
        per_test_tokens={"t1": ["L:1"], "t2": ["L:2"]},
        kept_tests=["t1", "t2"],
        derive_kept_tests=["t1"],
    )
    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="merged", delivered_km=None,
            delivered_tests=2))
    reduced = json.loads(
        (tmp_path / "reduced_coverage_matrix.json").read_text("utf-8"))
    assert reduced["derive_kept_tests"] == ["t1"]
    assert reduced["kept_tests"] == ["t1", "t2"]


def test_single_suite_winner_keeps_the_losing_probe_measurable(tmp_path) -> None:
    """The agent suite is delivered without a matrix. The reduction that lost
    keeps its own matrix under its own name, and under
    the probe name earlier archives used, and no delivered-suite matrix is
    written because the delivered suite has none."""
    phase = _phase(merged_km=_KM_MERGED, reduced_km=_KM_RED, pool_tests=10,
                   per_test_tokens={"t1": ["L:1"]}, kept_tests=["t1"])
    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="agent", delivered_km=None))
    names = _names(tmp_path)
    assert "selected_kill_matrix.json" not in names
    assert "reduced_kill_matrix.json" in names
    rec = json.loads(
        (tmp_path / "consolidation_record.json").read_text(encoding="utf-8"))
    assert rec["delivered_from"] == "agent"
    assert rec["shipped_has_matrix"] is False
    assert json.loads(
        (tmp_path / "reduced_kill_matrix.json").read_text(encoding="utf-8")) == _KM_RED


def test_single_suite_record_names_the_exact_source(tmp_path) -> None:
    """A single-suite winner keeps both the stable winner category
    and the exact pool input needed to explain the decision."""
    phase = _phase(
        merged_km=_KM_MERGED,
        reduced_km=_KM_RED,
        pool_tests=10,
    )
    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="single_suite",
            winner_source_label="standalone_1", delivered_km=phase.reduced_km,
            delivered_tests=7))

    rec = json.loads(
        (tmp_path / "consolidation_record.json").read_text(encoding="utf-8")
    )
    assert rec["delivered_from"] == "single_suite"
    assert rec["winner_source_label"] == "standalone_1"


def test_a_single_suite_kept_without_a_merge_records_its_measurement(tmp_path) -> None:
    """The merge was unavailable, so the tie-break kept one measured starter
    suite: no merged matrix, but the delivered suite's own. The record says
    mutation was measured, so the phase's mutation score is not missing."""
    phase = _phase(pool_tests=4, per_test_tokens={"t1": ["L:1"]}, kept_tests=["t1"])
    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="single_suite",
            winner_source_label="suite_007", delivered_km=_KM_MERGED, delivered_tests=4))
    assert "merged_kill_matrix.json" not in _names(tmp_path)
    rec = json.loads(
        (tmp_path / "consolidation_record.json").read_text(encoding="utf-8"))
    assert rec["shipped_has_matrix"] is True
    assert rec["mutation_measured"] is True


def test_no_mutation_boundary_still_records_the_decision(tmp_path) -> None:
    """A boundary without a mutation runner, coverage-only cover. The
    sidecars and the record persist. No kill matrices are expected."""
    phase = _phase(pool_tests=5, per_test_tokens={"t1": ["L:1"]},
                   kept_tests=["t1"])
    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="reduced", delivered_km=None,
            delivered_tests=1))
    names = _names(tmp_path)
    assert "consolidation_record.json" in names
    assert "merged_coverage_matrix.json" in names
    assert "merged_kill_matrix.json" not in names
    rec = json.loads(
        (tmp_path / "consolidation_record.json").read_text(encoding="utf-8"))
    assert rec["mutation_measured"] is False


def test_separate_outcome_records_detail(tmp_path) -> None:
    """A failed merge: nothing delivered, and the record still lands with the
    reason."""
    phase = _phase(merged_km=_KM_MERGED)
    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="separate", delivered_km=None),
        detail="merge failed")
    rec = json.loads(
        (tmp_path / "consolidation_record.json").read_text(encoding="utf-8"))
    assert rec["delivered_from"] == "separate"
    assert rec["detail"] == "merge failed"
    assert "merged_kill_matrix.json" in _names(tmp_path)


_C2_KM = {"mutants": [{"id": "m1", "status": "killed"}]}
_C2_COV = {"schema_version": 1, "target_file": "pkg/mod.py",
           "line": 90.0, "branch": 80.0,
           "per_test_tokens": {"t1": ["K:m1", "L:1"]}}


def test_candidate2_sidecars_persist_when_present(tmp_path) -> None:
    """The candidate-2 standalone kill and coverage sidecars land beside the
    merged matrices and are listed in the record's ``artifacts``."""
    phase = _phase(merged_km=_KM_MERGED, reduced_km=_KM_RED, pool_tests=10,
                   per_test_tokens={"t1": ["L:1"]}, kept_tests=["t1"],
                   candidate2_km=_C2_KM, candidate2_coverage=_C2_COV)
    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="merged", delivered_km=phase.merged_km,
            delivered_tests=1))
    names = _names(tmp_path)
    assert "candidate2_kill_matrix.json" in names
    assert "candidate2_coverage_matrix.json" in names
    assert json.loads((tmp_path / "candidate2_kill_matrix.json")
                      .read_text(encoding="utf-8")) == _C2_KM
    assert json.loads((tmp_path / "candidate2_coverage_matrix.json")
                      .read_text(encoding="utf-8")) == _C2_COV
    rec = json.loads(
        (tmp_path / "consolidation_record.json").read_text(encoding="utf-8"))
    assert "candidate2_kill_matrix.json" in rec["artifacts"]
    assert "candidate2_coverage_matrix.json" in rec["artifacts"]


def test_retry_removes_only_stale_phase_end_sidecars(tmp_path) -> None:
    """An unresolved retry cannot inherit matrices from an earlier attempt."""
    stale_names = (
        "reduced_kill_matrix.json",
        "merged_kill_matrix_selective.json",
        "candidate2_kill_matrix.json",
        "starter_pool_kill_matrix.json",
        "reduced_coverage_matrix.json",
    )
    for name in stale_names:
        (tmp_path / name).write_text("{}", encoding="utf-8")
    (tmp_path / "result.py").write_text("snapshot\n", encoding="utf-8")
    (tmp_path / "pre_merge").mkdir()
    (tmp_path / "pre_merge" / "candidate.py").write_text(
        "candidate\n", encoding="utf-8"
    )

    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py",
        phase=_phase(fallback_reason="evidence_regression"),
        delivery=PhaseEndDelivery(winner_label="unresolved", delivered_km=None))

    for name in stale_names:
        assert not (tmp_path / name).exists()
    assert (tmp_path / "result.py").is_file()
    assert (tmp_path / "pre_merge" / "candidate.py").is_file()
    record = json.loads(
        (tmp_path / "consolidation_record.json").read_text("utf-8")
    )
    assert record["delivered_from"] == "unresolved"
    assert record["artifacts"] == []


def test_unreduced_fallback_flag_tracks_the_candidate_list(tmp_path) -> None:
    """The record's unreduced_fallback flag is structural: true exactly when
    the fallback candidate list names the unreduced merge. A standalone
    keep-best reason (a different fallback family) does not raise it."""
    phase = _phase(fallback_reason="reduced_failing")
    phase.fallback_candidates = ["merged_unreduced"]
    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="merged", delivered_km=None))
    record = json.loads(
        (tmp_path / "consolidation_record.json").read_text("utf-8"))
    assert record["unreduced_fallback"] is True

    other = tmp_path / "standalone"
    other.mkdir()
    phase2 = _phase(fallback_reason="standalone_kill_regression")
    phase2.fallback_candidates = ["seed_pool"]
    persist_phase_end(
        matrices_dir=other, target_file="pkg/mod.py", phase=phase2,
        delivery=PhaseEndDelivery(winner_label="seed_1", delivered_km=None))
    record2 = json.loads(
        (other / "consolidation_record.json").read_text("utf-8"))
    assert record2["unreduced_fallback"] is False


def test_none_matrices_dir_is_a_noop(tmp_path, monkeypatch) -> None:
    """Without a matrices directory nothing is written into the working
    directory."""
    monkeypatch.chdir(tmp_path)
    persist_phase_end(
        matrices_dir=None, target_file="pkg/mod.py", phase=_phase(),
        delivery=PhaseEndDelivery(winner_label="merged", delivered_km=None))
    assert list(tmp_path.iterdir()) == []


def test_consolidation_record_is_strict_json(tmp_path) -> None:
    phase = _phase(selection_audit={
        "candidates": [{"label": "reduced", "runtime_ms": None}],
    })
    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="reduced", delivered_km=None))
    raw = (tmp_path / "consolidation_record.json").read_text("utf-8")
    assert "Infinity" not in raw
    json.loads(raw, parse_constant=lambda value: (_ for _ in ()).throw(
        AssertionError(f"non-standard JSON constant: {value}")
    ))


def test_consolidation_record_rejects_unexpected_nonfinite_values(
    tmp_path,
) -> None:
    phase = _phase(selection_audit={"unexpected": float("nan")})
    with pytest.raises(ValueError, match="Out of range float values"):
        persist_phase_end(
            matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
            delivery=PhaseEndDelivery(winner_label="reduced", delivered_km=None))


def test_write_failure_raises(tmp_path) -> None:
    """A boundary that cannot persist fails at the boundary: a file occupies
    the matrices directory path, so mkdir raises."""
    blocker = tmp_path / "archive"
    blocker.write_text("in the way", encoding="utf-8")
    with pytest.raises(OSError):
        persist_phase_end(
            matrices_dir=blocker, target_file="pkg/mod.py",
            phase=_phase(merged_km=_KM_MERGED),
            delivery=PhaseEndDelivery(winner_label="merged", delivered_km=None))


# Every file describes the suite its name says, whatever won

_MERGED = 'def test_a():\n    assert 1\n\ndef test_b():\n    assert 1\n'
_REDUCTION = 'def test_a():\n    assert 1\n'


def _sha(text):
    import hashlib
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


def _merged_won(tmp_path):
    """A merge that won over a distinct reduction, persisted with its
    matrices and without the delivered source."""
    phase = _phase(
        merged_km=_KM_MERGED, reduced_km=_KM_RED,
        merged_source=_MERGED, reduced_source=_REDUCTION,
        merged_source_sha256=_sha(_MERGED),
        reduced_source_sha256=_sha(_REDUCTION),
        pool_tests=2, selected_tests=1,
        per_test_tokens={'t::a': ['L:1'], 't::b': ['L:2']},
        kept_tests=['t::a'], reduction_valid=True,
    )
    persist_phase_end(
        matrices_dir=tmp_path, target_file='pkg/mod.py', phase=phase,
        delivery=PhaseEndDelivery(winner_label='merged', delivered_km=phase.merged_km,
            delivered_tests=2, delivered_source=None, execution_name='test_mod.py'))
    return phase


def _read(tmp_path, name):
    return json.loads((tmp_path / name).read_text(encoding='utf-8'))


def test_the_reduced_files_describe_the_reduction_when_the_merge_won(
    tmp_path,
) -> None:
    """The reduction lost, and its own matrix, not the merge's, is still
    under its own name."""
    _merged_won(tmp_path)

    reduced = _read(tmp_path, 'reduced_kill_matrix.json')
    merged = _read(tmp_path, 'merged_kill_matrix.json')
    assert reduced['mutants'] == _KM_RED['mutants']
    assert merged['mutants'] == _KM_MERGED['mutants']
    assert reduced['source_sha256'] == _sha(_REDUCTION)
    assert merged['source_sha256'] == _sha(_MERGED)


def test_every_matrix_names_the_execution_it_was_measured_under(
    tmp_path,
) -> None:
    _merged_won(tmp_path)

    for name in ('reduced_kill_matrix.json', 'merged_kill_matrix.json'):
        assert _read(tmp_path, name)['execution_name'] == 'test_mod.py'


def test_the_record_keeps_the_reductions_own_identity(tmp_path) -> None:
    """The record's reduced hash is the reduction's, and the reduction is
    reported as valid. The merge's selection is recorded as the winner,
    without rewriting the reduction's fields."""
    _merged_won(tmp_path)

    record = _read(tmp_path, 'consolidation_record.json')
    assert record['delivered_from'] == 'merged'
    assert record['reduced_source_sha256'] == _sha(_REDUCTION)
    assert record['merged_source_sha256'] == _sha(_MERGED)
    assert record['reduction_valid'] is True


def test_a_matrix_already_bound_to_other_bytes_is_refused(tmp_path) -> None:
    """A stamp is never overwritten: a matrix that already names other bytes
    is a caller error, raised before the archive is written."""
    wrong = dict(_KM_RED, source_sha256=_sha('something else'))
    phase = _phase(
        reduced_km=wrong, reduced_source=_REDUCTION,
        reduced_source_sha256=_sha(_REDUCTION),
    )
    with pytest.raises(RuntimeError, match='different source bytes'):
        persist_phase_end(
            matrices_dir=tmp_path, target_file='pkg/mod.py', phase=phase,
            delivery=PhaseEndDelivery(winner_label='reduced', delivered_km=None))
    assert _names(tmp_path) == []


def test_the_reduced_sidecar_has_the_merged_sidecars_layout(tmp_path) -> None:
    """reduced_coverage_matrix.json holds the merged sidecar's keys for the
    kept tests, null for times no run measured, then the kept tests."""
    phase = _phase(per_test_tokens={"t1": ["L:1"], "t2": ["L:2"]}, kept_tests=["t1"])
    persist_phase_end(
        matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
        delivery=PhaseEndDelivery(winner_label="reduced", delivered_km=None))
    reduced = _read(tmp_path, "reduced_coverage_matrix.json")
    merged = _read(tmp_path, "merged_coverage_matrix.json")
    assert list(reduced) == list(merged) + ["kept_tests", "derive_kept_tests"]
    assert reduced["per_test_tokens"] == {"t1": ["L:1"]}
    assert reduced["per_test_runtime_ms"] is None
    assert reduced["effective_cost_us"] is None
    assert reduced["kept_tests"] == ["t1"]


def test_a_missing_sidecar_raises_before_the_record_is_written(
    tmp_path, monkeypatch,
) -> None:
    """The written set is checked before consolidation_record.json, so a
    phase end whose coverage sidecar is missing leaves no record, which a
    resume treats as incomplete."""
    from testuary import consolidation

    monkeypatch.setattr(consolidation, "persist_coverage_sidecars",
                        lambda phase, mdir, target_file: ["merged_coverage_matrix.json"])
    phase = _phase(merged_km=_KM_MERGED, per_test_tokens={"t1": ["L:1"]})
    with pytest.raises(RuntimeError, match="merged_coverage_matrix.json"):
        persist_phase_end(
            matrices_dir=tmp_path, target_file="pkg/mod.py", phase=phase,
            delivery=PhaseEndDelivery(winner_label="merged", delivered_km=phase.merged_km))
    assert not (tmp_path / "consolidation_record.json").exists()


# consolidation_record.json in the order json.dumps writes it, which is
# the order the record is built in.
_RECORD_KEYS = [
    'schema_version', 't_mono_s', 'clock_domain_id', 'target_file',
    'delivered_from', 'winner_source_label', 'generation_ancestry',
    'selected_source_sha256', 'execution_name', 'target_revision',
    'measurement_identity', 'structural_parser_identity',
    'target_validator_identity', 'merged_source_sha256',
    'reduced_source_sha256', 'merged_tests', 'delivered_tests',
    'reduction_valid', 'repaired_tests', 'evidence_repairs',
    'unreduced_fallback', 'fallback_reason', 'fallback_candidates',
    'fallback_merged_stats', 'selection_audit', 'detail',
    'derive_measure_delta', 'derive_measure_differences', 'reduction_solver',
    'reduction_solver_metadata', 'consolidation_mutation_method', 'artifacts',
    'collision_resolution', 'input_normalizations', 'mutation_measured',
    'shipped_has_matrix', 'validity_policy_version', 'validity_artifacts',
]


@pytest.mark.parametrize('outcome', ['merged', 'unresolved'])
def test_the_record_writes_its_keys_in_one_order(tmp_path, outcome) -> None:
    """The record's bytes follow its key order, the same for a winner and
    for an end without one."""
    if outcome == 'merged':
        _merged_won(tmp_path)
    else:
        persist_phase_end(
            matrices_dir=tmp_path, target_file='pkg/mod.py',
            phase=_phase(merged_km=_KM_MERGED),
            delivery=PhaseEndDelivery(winner_label='unresolved', delivered_km=None,
                execution_name='test_mod.py'),
            detail='no winner', generation_scope=None, generation_inputs=None)
    assert list(_read(tmp_path, 'consolidation_record.json')) == _RECORD_KEYS
