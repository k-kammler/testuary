"""Phase-end merging, exact reduction, fresh measurement and evidence repair.

The tests use the real Python backend and deterministic mutation runners with
fixed catalogues, so identity-level derive and remeasurement comparisons are
reproducible without Cosmic Ray.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest

from testuary import languages
from testuary import consolidation as consolidate
from testuary.consolidation import (
    InheritedMatrix,
    PhaseEndDelivery,
    ScoredSuite, consolidate_phase_end, persist_phase_end,
)
from testuary.reduction import ExactCoverError
from per_suite_double import killing_tests
from phase_end_calls import named_id, named_under_coverage, names_under_coverage, phase_end

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


TARGET = """\
def add(a, b):
    return a + b


def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0
"""


def _project(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    return languages.get_backend("python"), str(root / "calc.py"), str(root)


SUITE_A = "from calc import add\n\ndef test_add_a():\n    assert add(1, 2) == 3\n"
SUITE_B = "from calc import add\n\ndef test_add_b():\n    assert add(2, 3) == 5\n"
SUITE_SIGN = ("from calc import sign\n\ndef test_sign_all():\n"
              "    assert sign(1) == 1\n    assert sign(-1) == -1\n"
              "    assert sign(0) == 0\n")

# A FIXED catalogue: the two add tests are redundant (both kill m0, both cover
# add's line), so the cover drops one. The sign test uniquely kills m1. Surv
# always survives. A mutant is killed iff one of its killer tests is present.
_KILLERS = {
    "m0": ("test_add_a", "test_add_b"),
    "m1": ("test_sign_all",),
}


def _runner_fixed_catalogue(directory):
    pyfile = next(Path(directory).glob("test_*.py"))
    present = set(re.findall(r"def (test_\w+)", pyfile.read_text(encoding="utf-8")))
    mutants = []
    for mid, killers in _KILLERS.items():
        live = [f"{pyfile.name}::{t}" for t in killers if t in present]
        mutants.append(
            {"id": mid, "status": "killed", "killer": live[0],
             "killers": live, "killing_tests": killing_tests(live), "line": None}
            if live else
            {"id": mid, "status": "survived", "killer": None,
             "killers": [], "line": None})
    mutants.append({"id": "surv", "status": "survived", "killer": None,
                    "killers": [], "line": None})
    return {"mutants": mutants, "summary": {}}


def test_phase_end_measures_merged_covers_and_derives_equal_to_measure(tmp_path):
    backend, target, root = _project(tmp_path)
    mutation_calls = []

    def runner(directory):
        mutation_calls.append(Path(directory))
        return _runner_fixed_catalogue(directory)

    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=runner, mutation_method="both",
        log=lambda m: None,
    )
    assert res.reduced_source is not None
    ast.parse(res.reduced_source)
    # One of the two redundant add tests drops, the sign test stays.
    assert "test_sign_all" in res.reduced_source
    assert ("test_add_a" in res.reduced_source) != ("test_add_b" in res.reduced_source)
    assert res.selected_tests == 2
    # The merged and reduced matrices were measured independently.
    assert res.merged_km is not None and res.reduced_km is not None
    from testuary.languages import mutation as km
    # Reduced kills m0 (an add test) + m1 (sign), surv survives -> 2/3.
    assert km.matrix_score_percent(res.reduced_km) == 100.0 * 2 / 3
    # derive == measure (the catalogue is fixed and the runner consistent).
    assert res.derive_measure_delta is not None
    assert abs(res.derive_measure_delta) < 1e-9
    assert len(mutation_calls) == 2
    assert res.derived_reduced_km is not None
    assert res.measured_reduced_km is not None
    assert res.derive_measure_differences == []


def test_noop_cover_persists_reused_measurement_as_explicit_cross_check(
    tmp_path,
):
    backend, target, root = _project(tmp_path)
    mutation_calls = []

    def runner(directory):
        mutation_calls.append(Path(directory))
        return _runner_fixed_catalogue(directory)

    res = phase_end(
        backend,
        [SUITE_SIGN],
        target_file_path=target,
        source_dir=root,
        timeout=120,
        mutation_runner=runner,
        mutation_method="both",
        log=lambda _message: None,
    )

    assert res.reduced_source == res.merged_source
    assert len(mutation_calls) == 1
    assert res.measured_reduced_km is res.merged_km
    assert res.reduced_km is res.merged_km
    assert res.derived_reduced_km is not None
    assert res.derive_measure_delta == 0.0
    assert res.derive_measure_differences == []

    archive = tmp_path / "archive"
    persist_phase_end(
        matrices_dir=archive, target_file=target, phase=res,
        delivery=PhaseEndDelivery(winner_label="reduced", delivered_km=res.reduced_km,
            delivered_tests=res.selected_tests, delivered_source=res.reduced_source,
            execution_name=res.execution_name))
    assert (archive / "derived_reduced_kill_matrix.json").is_file()
    assert (archive / "measured_reduced_kill_matrix.json").is_file()
    merged_validity = json.loads(
        (archive / "validity_merged.json").read_text(encoding="utf-8")
    )
    derivations = json.loads(
        (archive / "validity_derivations.json").read_text(encoding="utf-8")
    )["records"]
    delivered_validity = json.loads(
        (archive / "validity_delivered.json").read_text(encoding="utf-8")
    )
    assert merged_validity["record_type"] == "direct"
    assert merged_validity["findings"] == []
    assert derivations[-1]["operation"] == "set_cover_remove_restore"
    assert derivations[-1]["parent_source_sha256"] == (
        derivations[-1]["child_source_sha256"]
    )
    assert delivered_validity["final_clean_source_sha256"] == (
        derivations[-1]["child_source_sha256"]
    )


def test_invalid_synthesized_merge_fails_before_execution(
    tmp_path, monkeypatch,
):
    backend, target, root = _project(tmp_path)
    invalid = """\
import inspect
import calc

def test_source_oracle():
    assert "return a + b" in inspect.getsource(calc.add)
"""
    executed = []
    backend_type = type(backend)
    original_run = backend_type.run_test_files
    monkeypatch.setattr(
        backend_type,
        "run_test_files",
        lambda self, *args, **kwargs: executed.append(args) or original_run(
            self, *args, **kwargs
        ),
    )

    with pytest.raises(RuntimeError, match="generated-test validity"):
        phase_end(
            backend,
            [invalid],
            target_file_path=target,
            source_dir=root,
            timeout=120,
            mutation_runner=None,
            log=lambda _message: None,
        )

    assert executed == []


def test_phase_end_uses_executor_aware_target_hash(tmp_path, monkeypatch):
    backend, target, root = _project(tmp_path)
    seen = []
    monkeypatch.setattr(
        consolidate.project, "sha256_project_file",
        lambda path: seen.append(path) or "container-source-hash",
    )

    res = phase_end(
        backend, [], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=None, log=lambda _message: None,
    )

    assert seen == [target]
    assert res.target_revision == "container-source-hash"
    assert res.measurement_identity["target_source_sha256"] == (
        "container-source-hash"
    )


def test_phase_end_fails_closed_when_target_is_unreadable(tmp_path, monkeypatch):
    backend, target, root = _project(tmp_path)
    monkeypatch.setattr(
        consolidate.project, "sha256_project_file", lambda _path: None,
    )

    with pytest.raises(OSError, match="cannot read target source"):
        phase_end(
            backend, [], target_file_path=target, source_dir=root,
            timeout=120, mutation_runner=None, log=lambda _message: None,
        )


def test_finalized_candidate_does_not_supply_stale_collision_evidence(
    tmp_path, monkeypatch,
):
    backend, target, root = _project(tmp_path)
    changed = SUITE_A + "\n# detached comment from an old cleanup\n"
    stale = {"test_add_a": ["K:stale", "L:calc:1"]}
    current = {"test_add_b": ["K:current", "L:calc:1"]}
    observed = {}

    class MergeObserved(RuntimeError):
        pass

    def capture_merge(
        _backend, _sources, _log, *, evidence=None, **_kwargs,
    ):
        observed["evidence"] = evidence
        raise MergeObserved

    monkeypatch.setattr(consolidate, "_merge_sources", capture_merge)

    with pytest.raises(MergeObserved):
        phase_end(
            backend, [changed, SUITE_B],
            target_file_path=target, source_dir=root, timeout=120,
            mutation_runner=None,
            candidate_evidence=[stale, current],
            log=lambda _message: None,
        )

    assert observed["evidence"][0] is None
    assert observed["evidence"][1] == current


def test_exact_solver_failure_is_persisted_and_raised(tmp_path, monkeypatch):
    backend, target, root = _project(tmp_path)
    failure_dir = tmp_path / "archive" / "starter"

    def fail_exact(*_args, **_kwargs):
        raise ExactCoverError(
            "optimality was not established",
            {"failure": "optimality_not_established", "tiers": []},
        )

    monkeypatch.setattr("testuary.reduction.solve_cover", fail_exact)
    with pytest.raises(ExactCoverError, match="optimality"):
        phase_end(
            backend, [SUITE_A, SUITE_B],
            target_file_path=target, source_dir=root, timeout=120,
            mutation_runner=None, solver="exact",
            solver_failure_dir=str(failure_dir), log=lambda _message: None,
        )

    payload = __import__("json").loads(
        (failure_dir / "consolidation_failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert payload["reduction_solver"] == "exact"
    metadata = payload["reduction_solver_metadata"]
    assert metadata["failure"] == "optimality_not_established"
    assert metadata["error_type"] == "ExactCoverError"
    assert isinstance(payload["t_mono_s"], float)
    assert payload["clock_domain_id"]


@pytest.mark.parametrize("times, imputed", [
    ({"test_calc.test_add_a": 3.0}, {"test_calc.test_add_b": 3.0}),
    ({}, {"test_calc.test_add_a": 1.0, "test_calc.test_add_b": 1.0}),
], ids=["some-untimed", "none-timed"])
def test_a_test_without_a_time_gets_an_explicit_cost(
    tmp_path, monkeypatch, times, imputed,
):
    """A timing table missing some tests still lets the merge be reduced: the
    solver weighs the measured times and imputes for the others the slowest
    measured time, or equal costs when none was measured."""
    backend, target, root = _project(tmp_path)

    def partly_timed(_self, *_args, **_kwargs):
        return SimpleNamespace(
            per_test_tokens={
                "test_calc.test_add_a": frozenset({"L:calc:2"}),
                "test_calc.test_add_b": frozenset({"L:calc:2"}),
            },
            per_test_time_ms=dict(times),
            suite_time_ms=3.0,
            includes_kills=False,
            target_total_lines=8,
            target_total_branches=4,
            suite_covered_lines=frozenset({1, 2, 5}),
            suite_covered_branches=frozenset(),
            first_test=None,
        )

    seen = {}

    def solver(per_test_tokens, per_test_cost_ms, **kwargs):
        seen.update(costs=dict(per_test_cost_ms),
                    imputed=dict(kwargs.get("imputed_cost_ms") or {}))
        raise ExactCoverError("the costs are what this test checks",
                              {"solver": "exact"})

    monkeypatch.setattr(
        type(backend), "collect_eviction_data", partly_timed,
    )
    monkeypatch.setattr("testuary.reduction.solve_cover", solver)
    with pytest.raises(ExactCoverError):
        phase_end(
            backend, [SUITE_A, SUITE_B], target_file_path=target,
            source_dir=root, timeout=120, mutation_runner=None,
            solver="exact", log=lambda _message: None,
        )
    assert seen == {"costs": times, "imputed": imputed}


def _phase_end_with_evidence(tmp_path, monkeypatch, change):
    """A phase end over SUITE_A and SUITE_B whose per-test evidence's covered
    lines are ``change`` of the lines it covers, and the lines the merged
    suite's own run covers."""
    import dataclasses

    backend, target, root = _project(tmp_path)
    collect = type(backend).collect_eviction_data
    measured = {}

    def changed(self, *args, **kwargs):
        data = collect(self, *args, **kwargs)
        measured.setdefault("lines", sorted(data.suite_covered_lines))
        return dataclasses.replace(data, suite_covered_lines=frozenset(
            change(sorted(data.suite_covered_lines))))

    monkeypatch.setattr(type(backend), "collect_eviction_data", changed)
    res = phase_end(
        backend, [SUITE_A, SUITE_B], target_file_path=target,
        source_dir=root, timeout=120, mutation_runner=None,
        log=lambda _message: None,
    )
    return res, measured["lines"]


def test_evidence_coverage_differing_from_the_measured_coverage_is_recorded_beside_it(
    tmp_path, monkeypatch,
):
    """Per-test evidence that covers fewer lines than the merged suite's own
    run leaves that run's coverage the merged suite's, and the record names
    both sides. The reduction still works on the evidence."""
    res, lines = _phase_end_with_evidence(tmp_path, monkeypatch, lambda lines: lines[1:])
    assert (res.line, res.union_line, res.suite_covered_lines) == (37.5, 37.5, lines)
    difference = res.selection_audit["merged_evidence_difference"]
    assert difference["measured"] == {
        "line": 37.5, "branch": res.branch, "covered_lines_only": lines[:1],
        "taken_arcs_only": []}
    assert difference["per_test_evidence"]["line"] == 25.0
    assert difference["per_test_evidence"]["covered_lines_only"] == []
    assert res.reduced_source is not None


def test_evidence_covering_other_lines_than_the_measured_coverage_is_recorded_beside_it(
    tmp_path, monkeypatch,
):
    """Equal percentages are not enough: evidence that covers other lines
    than the merged suite's own run is recorded with the lines each side
    alone covers."""
    res, lines = _phase_end_with_evidence(
        tmp_path, monkeypatch, lambda lines: [*lines[1:], max(lines) + 1000])
    assert res.suite_covered_lines == lines
    difference = res.selection_audit["merged_evidence_difference"]
    assert (difference["measured"]["line"], difference["per_test_evidence"]["line"]) == (
        37.5, 37.5)
    assert difference["measured"]["covered_lines_only"] == lines[:1]
    assert difference["per_test_evidence"]["covered_lines_only"] == [max(lines) + 1000]


@pytest.mark.parametrize("method, derived", [("both", False), ("derive", True)])
def test_a_derived_reduced_mutation_score_is_named_derived(tmp_path, method, derived) -> None:
    """Under derive the reduced suite's mutation score comes from the merged
    suite's kill matrix, and its candidate says so. Under both it is
    measured, and its candidate names nothing."""
    backend, target, root = _project(tmp_path)
    redundant = ("from calc import add, sign\n\ndef test_add_a():\n    assert add(1, 2) == 3\n"
                 "\n\ndef test_add_b():\n    assert add(2, 3) == 5\n"
                 "\n\ndef test_sign_all():\n    assert sign(1) == 1\n"
                 "    assert sign(-1) == -1\n    assert sign(0) == 0\n")
    result = phase_end(
        backend, [redundant, redundant], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=_runner_fixed_catalogue, mutation_method=method,
        log=lambda _message: None,
    )
    assert result.reduced_source is not None and result.reduced_km is not None
    reason = consolidate.scored_from_phase(result).status_reason
    assert reason == ("its mutation score is derived from the merged suite's kill matrix "
                      "for its tests, not measured" if derived else "")


def test_a_candidate_whose_evidence_disagrees_with_its_run_is_scored_on_the_run(
        tmp_path, monkeypatch):
    """The phase's own suite whose per-test evidence disagrees with its
    coverage run is scored on that run, without the evidence, and says why."""
    import dataclasses

    backend, target, root = _project(tmp_path)
    collect = type(backend).collect_eviction_data

    def one_line_short(self, *args, **kwargs):
        data = collect(self, *args, **kwargs)
        return dataclasses.replace(data, suite_covered_lines=frozenset(
            sorted(data.suite_covered_lines)[1:]))

    monkeypatch.setattr(type(backend), "collect_eviction_data", one_line_short)
    scored = consolidate.measure_candidate_standalone(
        backend, SUITE_A, target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=None, stage_as_delivered=False, log=lambda m: None)
    assert scored.status == "passing" and scored.per_test_tokens is None
    assert scored.status_reason.startswith(
        "its per-test evidence disagrees with its coverage run: line coverage")


def test_a_same_named_module_elsewhere_keeps_the_coverages_equal(tmp_path):
    """The per-test evidence reads the target's own coverage, not that of a
    same-named module below the target's directory, so it agrees with the
    checked coverage."""
    root = tmp_path / "proj"
    (root / "pkg" / "asyncio").mkdir(parents=True)
    (root / "pkg" / "__init__.py").write_text(
        "from pkg.asyncio import calc as _other\n", encoding="utf-8")
    (root / "pkg" / "calc.py").write_text(TARGET, encoding="utf-8")
    (root / "pkg" / "asyncio" / "__init__.py").write_text("", encoding="utf-8")
    (root / "pkg" / "asyncio" / "calc.py").write_text(
        "def twice(x):\n    return 2 * x\n", encoding="utf-8")
    suite = SUITE_SIGN.replace("from calc import", "from pkg.calc import")
    res = phase_end(
        languages.get_backend("python"), [suite],
        target_file_path=str(root / "pkg" / "calc.py"), source_dir=str(root),
        timeout=120, mutation_runner=None, log=lambda _message: None,
    )
    assert res.target_total_lines == 8
    assert res.suite_covered_lines == [1, 5, 6, 7, 8, 9, 10]
    assert res.target_total_branches == 4
    assert len(res.suite_covered_branches) == 4


def _module_coupled_suite(backend, tmp_path, target) -> str:
    """A suite whose single test passes only when its module is the delivered
    name for ``target``."""
    delivered = backend.target_test_path(str(tmp_path), target).name
    return (
        "from pathlib import Path\n\n"
        "from calc import add\n\n"
        "def test_module_name_is_the_delivered_one():\n"
        f"    assert Path(__file__).name == {delivered!r}\n"
        "    assert add(1, 2) == 3\n"
    )


def test_phase_end_measures_the_candidates_under_the_delivered_name(tmp_path):
    """At the mutation boundary candidates run under the delivered module name,
    so a test that depends on its module name passes the merged suite's green
    check and is not stripped."""
    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [_module_coupled_suite(backend, tmp_path, target), SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        stage_as_delivered=True,
        log=lambda m: None,
    )
    assert res.reduced_source is not None
    assert "test_module_name_is_the_delivered_one" in res.reduced_source


def test_phase_end_stages_under_the_fixed_scratch_name_by_default(tmp_path):
    """By default candidates are staged under a fixed scratch module name, so
    the module-coupled test fails there and the merged suite is no
    candidate, named by the scratch module; nothing is removed. Production
    phase ends pass ``stage_as_delivered=True``."""
    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [_module_coupled_suite(backend, tmp_path, target), SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        log=lambda m: None,
    )
    assert res.merged_green is False
    assert "test_candidate." in res.detail
    assert "test_module_name_is_the_delivered_one" in res.detail


def test_phase_end_coverage_only_without_a_runner(tmp_path):
    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=None, log=lambda m: None,
    )
    assert res.reduced_source is not None
    assert res.merged_km is None and res.reduced_km is None
    # Coverage-only: the two add tests collapse, the sign test stays.
    assert res.selected_tests == 2
    assert res.line is not None


def test_phase_end_ships_merged_unreduced_when_eviction_fails(tmp_path, monkeypatch):
    """When the per-test eviction run fails, the green merged suite with its
    measured mutation is delivered unreduced with its full kill matrix rather than
    dropping out of keep-best. The cover is an optimization, not a
    correctness gate."""
    from testuary.consolidation import (
        merged_scored_from_phase, scored_from_phase)
    from testuary.languages import mutation as km

    backend, target, root = _project(tmp_path)
    # The instrumented per-test run fails (returns None).
    monkeypatch.setattr(
        type(backend), "collect_eviction_data", lambda *a, **k: None)
    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        log=lambda m: None,
    )
    # The merged suite is delivered with its measured matrix.
    assert res.reduced_source is None
    assert res.merged_km is not None and res.reduced_km is None
    assert "eviction failed" in res.detail
    # The unreduced takeover is recorded: the union record keeps the runnable
    # test count, and token-derived fields stay null because the run failed.
    assert res.fallback_reason == "eviction_data_unavailable"
    assert res.fallback_candidates == ["merged_unreduced"]
    assert res.fallback_union_stats == {
        "merged_tests": 3, "rename_twins": None,
        "attempted_reduced_tests": None, "redundancy_ratio": None,
    }
    assert res.aggregate_coverage_measured is True
    assert res.pool_tests == 3
    assert res.target_total_lines is not None
    assert res.target_total_branches is not None
    # The full merged suite kills m0 and m1, surv survives: 2/3.
    assert km.matrix_score_percent(res.merged_km) == 100.0 * 2 / 3
    # No cover ran, so the merged suite is the only green candidate.
    assert scored_from_phase(res).green is False
    assert merged_scored_from_phase(res).green is True


def test_eviction_fallback_does_not_count_parameter_expansion(
    tmp_path, monkeypatch,
):
    backend, target, root = _project(tmp_path)
    source = (
        "import pytest\n"
        "from calc import add\n\n"
        "@pytest.mark.parametrize('left', [1, 2, 3])\n"
        "def test_add_a(left):\n"
        "    assert add(left, 1) == left + 1\n"
    )
    monkeypatch.setattr(
        type(backend), "collect_eviction_data", lambda *a, **k: None,
    )

    res = phase_end(
        backend, [source],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        log=lambda _message: None,
    )

    assert res.fallback_reason == "eviction_data_unavailable"
    assert res.pool_tests == 1
    assert res.fallback_union_stats["merged_tests"] == 1


def test_phase_end_rejects_merged_when_eviction_test_run_is_red(
    tmp_path, monkeypatch,
):
    """A test that fails only in the per-test coverage run, after the
    merged suite's passing normal run, leaves it without per-test evidence:
    the reduction is unavailable, and the merged suite stays a candidate
    with its measured numbers, as when that run fails otherwise."""
    from testuary.consolidation import merged_scored_from_phase

    backend, target, root = _project(tmp_path)

    def red_eviction(_self, *_args, failure_fn=None, **_kwargs):
        assert failure_fn is not None
        failure_fn("test_failure", "measured run exited 1: 1 failed")
        return None

    monkeypatch.setattr(
        type(backend), "collect_eviction_data", red_eviction,
    )
    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        log=lambda _message: None,
    )

    assert res.merged_source is not None
    assert res.merged_green is True
    assert "measured run exited 1: 1 failed" in res.detail
    assert res.fallback_reason == "eviction_test_failure"
    assert res.fallback_candidates == ["merged_unreduced"]
    assert merged_scored_from_phase(res).green is True
    assert merged_scored_from_phase(res).status == "passing"


def _runner_unplaceable_killers(directory):
    """Same fixed catalogue, but every killer names a test that is not in the
    suite, so the backend cannot place it (includes_kills False). The status
    still reflects which catalogue tests are present."""
    pyfile = next(Path(directory).glob("test_*.py"))
    present = set(re.findall(r"def (test_\w+)", pyfile.read_text(encoding="utf-8")))
    mutants = []
    for mid, killers in _KILLERS.items():
        if any(t in present for t in killers):
            mutants.append({"id": mid, "status": "killed",
                            "killer": f"{pyfile.name}::ghost_unmatchable",
                            "killers": [f"{pyfile.name}::ghost_unmatchable"],
                            "killing_tests": killing_tests([f"{pyfile.name}::ghost_unmatchable"]),
                            "line": None})
        else:
            mutants.append({"id": mid, "status": "survived",
                            "killer": None, "killers": [], "line": None})
    mutants.append({"id": "surv", "status": "survived", "killer": None,
                    "killers": [], "line": None})
    return {"mutants": mutants, "summary": {}}


def test_phase_end_incomplete_attribution_keeps_only_the_merge_candidate(
    tmp_path,
):
    """When killers cannot be placed (includes_kills False), reducing on
    coverage alone could drop a killer, so no reduced candidate is built and
    the measured merged suite stays available to the selector."""
    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_unplaceable_killers, mutation_method="derive",
        log=lambda m: None,
    )
    assert res.reduced_source is None
    assert res.reduced_km is None
    assert res.kept_tests is None
    assert res.selected_tests == 0
    assert res.measured_reduced_km is None
    # An unreduced takeover: attempt fields null, union fields measured.
    assert res.fallback_reason == "kills_unattributed"
    assert "could not be attributed" in res.detail
    assert res.fallback_candidates == ["merged_unreduced"]
    assert res.fallback_union_stats["merged_tests"] == res.pool_tests
    assert res.fallback_union_stats["attempted_reduced_tests"] is None
    assert res.fallback_union_stats["redundancy_ratio"] is None
    from testuary.languages import mutation as km
    assert km.matrix_score_percent(res.merged_km) == 100.0 * 2 / 3


def test_phase_end_without_a_merged_matrix_records_the_absent_reduction(
    tmp_path,
):
    """A runner returning no kill matrix for the merged suite leaves nothing to
    reduce on: the record names why, and no cover runs on coverage alone."""
    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=lambda _directory: None, mutation_method="both",
        log=lambda m: None,
    )
    assert res.merged_km is None
    assert res.fallback_reason == "merged_mutation_unmeasured"
    assert res.fallback_candidates == []
    assert res.reduced_source is None and res.kept_tests is None
    assert not res.reduction_valid
    assert res.fallback_union_stats["merged_tests"] == res.pool_tests


# The sleep gives this test a strictly positive measured cost, so the cover
# below has a unique optimum. Load can only raise it.
SUITE_EXTRA = ("import time\n\nfrom calc import add\n\n"
               "def test_add_extra():\n"
               "    time.sleep(0.05)\n"
               "    assert add(4, 4) == 8\n")


def _runner_fragile_killer(directory):
    """Composition-fragile kill: ``mfrag``'s recorded killers are test_add_a
    and test_add_extra, but the kill reproduces only when test_add_extra runs.
    The cover keeps test_add_a (sole carrier of m0's kill token) and drops
    test_add_extra, so the re-measure loses mfrag and the repair pass must
    re-add test_add_extra from the recorded carriers."""
    pyfile = next(Path(directory).glob("test_*.py"))
    present = set(re.findall(r"def (test_\w+)",
                             pyfile.read_text(encoding="utf-8")))
    live_m0 = ([f"{pyfile.name}::test_add_a"]
               if "test_add_a" in present else [])
    frag_killed = "test_add_extra" in present
    frag_killers = [f"{pyfile.name}::{t}"
                    for t in ("test_add_a", "test_add_extra") if t in present]
    mutants = [
        {"id": "m0", "operator": "op_m0", "line": 2,
         "status": "killed" if live_m0 else "survived",
         "killer": live_m0[0] if live_m0 else None, "killers": live_m0,
         "killing_tests": killing_tests(live_m0)},
        {"id": "mfrag", "operator": "op_frag", "line": 3,
         "status": "killed" if frag_killed else "survived",
         "killer": (frag_killers[0] if frag_killed and frag_killers
                    else None),
         "killers": frag_killers if frag_killed else [],
         "killing_tests": killing_tests(frag_killers if frag_killed else [])},
        {"id": "surv", "operator": "op_surv", "line": 9,
         "status": "survived", "killer": None, "killers": []},
    ]
    return {"mutants": mutants, "summary": {}}


def test_phase_end_kill_repair_readds_the_recorded_killer(tmp_path):
    """A kill the re-measured reduced suite loses is repaired: the recorded
    killer the cover dropped is re-added and the suite re-measured, so the
    delivered suite keeps the kill.

    test_add_a alone carries m0's kill token and its tokens include
    test_add_extra's, so ``{test_add_a}`` is the minimum-cost lossless cover
    under any measured timings: every other cover is a strict superset."""
    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [SUITE_A, SUITE_EXTRA],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fragile_killer, mutation_method="both",
        # The minimum-cost solver, so the cover is the argmin above. The greedy
        # heuristic ranks by tokens-per-millisecond and keeps the carrier.
        solver="exact",
        log=lambda m: None,
    )
    from testuary.languages import mutation as km
    # The merged pool kills m0 and mfrag (2/3), the repaired suite matches.
    assert km.matrix_score_percent(res.merged_km) == 100.0 * 2 / 3
    assert km.matrix_score_percent(res.reduced_km) == 100.0 * 2 / 3
    assert res.reduced_source is not None
    assert "test_add_extra" in res.reduced_source
    assert res.repaired_tests and res.repaired_tests[0].endswith(
        "test_add_extra")
    assert res.selected_tests == len(res.kept_tests) == 2
    # The persisted derive comparison describes the accepted repaired bytes,
    # while the repair record preserves the discrepancy that triggered it.
    assert res.derive_measure_delta == 0.0
    assert res.derive_measure_differences == []
    assert res.evidence_repairs[0]["missing_kills_before"]
    assert res.evidence_repairs[0]["outcome"] == "accepted"


def _runner_unattributed_clock(directory, *, line):
    """``mtime`` is a timeout kill of the merged pool at ``line`` with no
    recorded killer (a hang records none) that survives any smaller suite.
    With no kill token, repair can only escalate to line carriers."""
    pyfile = next(Path(directory).glob("test_*.py"))
    present = set(re.findall(r"def (test_\w+)",
                             pyfile.read_text(encoding="utf-8")))
    live_m0 = [f"{pyfile.name}::{t}" for t in ("test_add_a", "test_add_b")
               if t in present]
    both_present = {"test_add_a", "test_add_b"} <= present
    mutants = [
        {"id": "m0", "operator": "op_m0", "line": 2,
         "status": "killed" if live_m0 else "survived",
         "killer": live_m0[0] if live_m0 else None, "killers": live_m0,
         "killing_tests": killing_tests(live_m0)},
        {"id": "mtime", "operator": "op_time", "line": line,
         "status": "timeout" if both_present else "survived",
         "killer": None, "killers": []},
        {"id": "surv", "operator": "op_surv", "line": 9,
         "status": "survived", "killer": None, "killers": []},
    ]
    return {"mutants": mutants, "summary": {}}


def test_phase_end_escalated_repair_restores_a_composition_dependent_kill(
        tmp_path):
    """A lost kill without a recorded killer is restored by escalating to the
    tests reaching the mutated line: a repaired reduced suite is delivered, never the
    unreduced pool."""
    backend, target, root = _project(tmp_path)
    logs = []
    res = phase_end(
        backend, [SUITE_A, SUITE_B],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=lambda d: _runner_unattributed_clock(d, line=2),
        mutation_method="both", log=logs.append,
    )
    from testuary.languages import mutation as km
    assert km.matrix_score_percent(res.merged_km) == 100.0 * 2 / 3
    assert km.matrix_score_percent(res.reduced_km) == 100.0 * 2 / 3
    assert res.selected_tests == 2
    assert res.repaired_tests
    assert res.evidence_repairs[0]["missing_kills_before"]
    assert res.evidence_repairs[0]["added_tests"]
    assert res.evidence_repairs[0]["outcome"] == "accepted"
    assert not any("interference" in m for m in logs)


def test_phase_end_unrepairable_identity_loss_remains_a_scored_candidate(
    tmp_path,
):
    """Unrepairable identity loss is audited and left to score selection."""
    backend, target, root = _project(tmp_path)
    logs = []
    res = phase_end(
        backend, [SUITE_A, SUITE_B],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=lambda d: _runner_unattributed_clock(d, line=None),
        mutation_method="both", log=logs.append,
    )
    from testuary.languages import mutation as km
    assert km.matrix_score_percent(res.merged_km) == 100.0 * 2 / 3
    assert km.matrix_score_percent(res.reduced_km) == 100.0 * 1 / 3
    assert res.reduction_valid is True
    assert res.reduced_source != res.merged_source
    assert res.selected_tests == 1
    assert res.pool_tests == 2
    assert not res.repaired_tests
    assert any("no unused carrier remains" in m for m in logs)
    assert res.evidence_repairs[0]["outcome"] == "no_carrier"
    assert res.selection_audit["reduction_missing_merged_kills"] == 1
    assert res.fallback_reason is None
    assert res.selection_audit["reduction_missing_merged_kills"] == 1


SUITE_BAD = ("from calc import add\n\n"
             "def test_bad():\n    assert add(1, 1) == 999\n")


def test_a_merged_suite_with_a_failing_test_is_no_candidate_and_keeps_its_tests(tmp_path):
    """A test that fails in the merged file leaves the merged suite out of
    the phase end's choice, with the reason; no test is removed, and no
    reduction is made from it."""
    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [SUITE_A, SUITE_BAD],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="derive",
        log=lambda m: None,
    )
    assert res.merged_green is False
    assert "does not pass" in res.detail and "test_bad" in res.detail
    assert res.reduced_source is None and res.line is None


def _runner_two_phase(reduced):
    """A runner whose first call (the merged file) returns the fixed catalogue
    and whose second call (the reduced re-measure) returns ``reduced(dir)``."""
    calls = {"n": 0}

    def _run(directory):
        calls["n"] += 1
        if calls["n"] == 1:
            return _runner_fixed_catalogue(directory)
        return reduced(directory)

    return _run, calls


def test_phase_end_measure_mode_ships_the_remeasured_matrix(tmp_path):
    """Measure mode delivers the fresh re-measure of the reduced suite, not the
    derived matrix: a mutant only the re-measure reports is in res.reduced_km."""
    backend, target, root = _project(tmp_path)

    def _reduced(directory):
        cat = _runner_fixed_catalogue(directory)
        cat["mutants"].append({"id": "remeasure_only", "status": "killed",
                               "killer": None, "killers": [], "line": None})
        return cat

    runner, calls = _runner_two_phase(_reduced)
    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=runner, mutation_method="measure", log=lambda m: None,
    )
    assert calls["n"] == 2                                # merged + reduced re-measure
    assert res.measured_reduced_km is not None
    assert res.reduced_km is res.measured_reduced_km      # measure delivers the re-measure
    ids = {m["id"] for m in res.reduced_km["mutants"]}
    assert "remeasure_only" in ids                        # derive could not add it


def test_phase_end_both_ships_the_remeasure_not_the_derive(tmp_path):
    """Under the default ``both`` the delivered reduced matrix is the fresh
    re-measure, and the derived one is only the cross-check
    (derive_measure_delta)."""
    backend, target, root = _project(tmp_path)

    def _reduced(directory):
        cat = _runner_fixed_catalogue(directory)
        cat["mutants"].append({"id": "remeasure_only", "status": "killed",
                               "killer": None, "killers": [], "line": None})
        return cat

    runner, calls = _runner_two_phase(_reduced)
    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=runner, mutation_method="both", log=lambda m: None,
    )
    assert calls["n"] == 2                                # merged + reduced re-measure
    assert res.measured_reduced_km is not None
    assert res.reduced_km is res.measured_reduced_km      # delivers the re-measure
    ids = {m["id"] for m in res.reduced_km["mutants"]}
    assert "remeasure_only" in ids                        # derive could not add it
    assert res.derive_measure_delta is not None           # the cross-check still ran


def test_phase_end_both_marks_a_reduction_with_no_matrix_incomplete(
        tmp_path):
    """Under ``both``, a reduction with no matrix cannot enter selection."""
    backend, target, root = _project(tmp_path)
    runner, _calls = _runner_two_phase(lambda directory: None)  # re-measure fails
    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=runner, mutation_method="both", log=lambda m: None,
    )
    assert res.measured_reduced_km is None
    assert res.reduced_km is None
    assert res.reduction_valid is True
    assert res.fallback_reason is None
    assert res.selected_tests == 2
    assert res.selection_audit[
        "reduction_measurement_issue"
    ] == "reduced_remeasure_no_matrix"


def test_phase_end_measure_mode_marks_a_no_matrix_reduction_incomplete(tmp_path):
    """Measure mode applies the same explicit incomplete-candidate rule."""
    backend, target, root = _project(tmp_path)
    runner, _calls = _runner_two_phase(lambda directory: None)
    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=runner, mutation_method="measure", log=lambda m: None,
    )
    assert res.measured_reduced_km is None
    assert res.reduced_km is None
    assert res.reduction_valid is True
    assert res.selection_audit[
        "reduction_measurement_issue"
    ] == "reduced_remeasure_no_matrix"


def test_phase_end_empty_token_universe_constructs_no_reduced_candidate(
    tmp_path,
):
    """A green suite covering none of the target has an empty token universe,
    so the runnable merged suite stays the candidate and no empty reduced
    suite is built."""
    backend, target, root = _project(tmp_path)
    noop = "def test_noop():\n    assert True\n"
    res = phase_end(
        backend, [noop], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=None, log=lambda m: None,
    )
    assert res.merged_source is not None
    assert res.reduced_source is None
    assert res.reduced_km is None
    assert res.kept_tests is None
    assert res.selected_tests == 0
    assert res.pool_tests == 1
    assert "empty per-test token universe" in res.detail
    # No solver instance exists because there is no evidence token to cover.
    assert res.fallback_reason == "empty_cover"
    assert res.fallback_union_stats["merged_tests"] == 1
    assert res.fallback_union_stats["attempted_reduced_tests"] is None
    assert res.fallback_union_stats["redundancy_ratio"] is None


def test_phase_end_red_reduced_is_excluded_from_selection(
    tmp_path, monkeypatch,
):
    """A reduced suite that does not pass its normal run is not a candidate,
    and nothing measures it."""
    from types import SimpleNamespace
    from testuary import normal_run

    backend, target, root = _project(tmp_path)
    real_run = normal_run.run

    def _run(bk, paths, proot, **kw):
        if Path(paths[0]).parent.name == "reduced":
            return SimpleNamespace(ok=False)
        return real_run(bk, paths, proot, **kw)

    monkeypatch.setattr(normal_run, "run", _run)
    runner, calls = _runner_two_phase(lambda directory: None)
    logs = []
    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=runner, mutation_method="both", log=logs.append,
    )
    assert res.measured_reduced_km is None
    assert res.reduction_valid is False
    assert res.reduced_km is None
    assert res.selected_tests == 2
    assert any("so phase-end selection excludes it" in m for m in logs)
    assert res.fallback_reason is None
    assert res.selection_audit["reduction_measurement_issue"] == "reduced_failing"


def test_phase_end_sets_candidate2_sidecars_at_two_candidate_boundary(tmp_path):
    """At a two-candidate boundary with a mutation runner, candidate 2
    (candidates[-1]) is measured standalone and its kill matrix and coverage
    sidecar kept, so an audit can check the merge kept every standalone kill."""
    from testuary.languages import mutation as km
    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [SUITE_A, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        log=lambda m: None,
    )
    assert res.candidate2_km is not None
    # candidate 2 = SUITE_SIGN kills m1 alone (m0/surv survive standalone).
    assert km.killed_ids(res.candidate2_km) == {"m1"}
    assert res.candidate2_coverage is not None
    assert res.candidate2_coverage["schema_version"] == 1
    assert res.candidate2_coverage["target_file"] == target
    assert "per_test_tokens" in res.candidate2_coverage
    pt = res.candidate2_coverage["per_test_tokens"]
    assert any("test_sign_all" in t for t in pt)
    candidate = res.candidate2_candidate
    assert candidate is not None
    assert candidate.status == "passing"
    assert candidate.source == res.candidate2_source == SUITE_SIGN
    assert candidate.source_sha256 == hashlib.sha256(
        SUITE_SIGN.encode("utf-8")
    ).hexdigest()
    assert candidate.coverage_evidence == res.candidate2_coverage
    assert candidate.per_test_tokens


def test_phase_end_no_candidate2_for_many_suites_or_one_suite(tmp_path):
    """Starter-phase (many suites) and single-suite boundaries measure no
    candidate 2: only a two-candidate merge does."""
    backend, target, root = _project(tmp_path)
    res_many = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        log=lambda m: None,
    )
    assert res_many.candidate2_km is None
    assert res_many.candidate2_coverage is None
    res_one = phase_end(
        backend, [SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        log=lambda m: None,
    )
    assert res_one.candidate2_km is None
    assert res_one.candidate2_coverage is None


def test_phase_end_measures_candidate2_without_a_runner(tmp_path):
    """Without a mutation runner candidate 2 is still measured standalone,
    for its coverage and per-test evidence, so a coverage-only phase end
    chooses it on its own numbers; it has no kill matrix."""
    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [SUITE_A, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=None, log=lambda m: None,
    )
    assert res.candidate2_km is None
    assert res.candidate2_coverage is not None
    assert any("test_sign_all" in t for t in res.candidate2_coverage["per_test_tokens"])
    assert res.candidate2_candidate.status == "passing"
    assert res.candidate2_green and res.candidate2_source == SUITE_SIGN


def test_phase_end_candidate2_red_leaves_the_sidecars_absent(tmp_path):
    """A candidate 2 that is red standalone leaves both sidecars None; its
    failing test also fails in the merge, which is then no candidate either,
    and nothing is removed from it."""
    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [SUITE_A, SUITE_BAD],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="derive",
        log=lambda m: None,
    )
    # SUITE_BAD (candidate 2) fails standalone -> no candidate-2 sidecars.
    assert res.candidate2_km is None
    assert res.candidate2_coverage is None
    assert res.candidate2_candidate.status == "failing"
    assert res.candidate2_candidate.source == SUITE_BAD
    assert res.merged_green is False and res.reduced_source is None


def test_candidate2_infrastructure_failure_is_scored_on_its_coverage_run(
    tmp_path, monkeypatch,
):
    """A per-test coverage run that fails for any reason leaves the phase's
    own suite without per-test evidence: it is scored on its green coverage
    run, with the failure recorded, as for a language without per-test
    evidence."""
    backend, target, root = _project(tmp_path)
    backend_type = type(backend)
    collect = backend_type.collect_eviction_data

    def fail_candidate2(self, test_paths, *args, failure_fn=None, **kwargs):
        if Path(test_paths[0]).parent.name == "candidate2":
            assert failure_fn is not None
            failure_fn("infrastructure", "coverage wrapper failed")
            return None
        return collect(
            self, test_paths, *args, failure_fn=failure_fn, **kwargs,
        )

    monkeypatch.setattr(backend_type, "collect_eviction_data", fail_candidate2)
    res = phase_end(
        backend, [SUITE_A, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        log=lambda _message: None,
    )

    candidate = res.candidate2_candidate
    assert candidate is not None
    assert candidate.green is True
    assert candidate.status == "passing"
    assert candidate.line is not None and candidate.per_test_tokens is None
    assert candidate.status_reason == (
        "the per-test coverage run could not measure it: coverage wrapper failed"
    )


def test_scored_suite_rejects_a_source_hash_mismatch() -> None:
    with pytest.raises(ValueError, match="source hash"):
        ScoredSuite(
            label="candidate",
            source="def test_ok():\n    assert True\n",
            source_sha256="not-the-source-hash",
        )


def test_phase_end_persistence_rejects_shadowed_delivered_source(tmp_path):
    _backend, target, _root = _project(tmp_path)
    source = (
        "def test_value():\n    assert 1\n\n"
        "def test_value():\n    assert 2\n"
    )

    with pytest.raises(RuntimeError, match="shadowed definitions"):
        persist_phase_end(
            matrices_dir=tmp_path / "matrices", target_file=target,
            phase=consolidate.PhaseConsolidation(),
            delivery=PhaseEndDelivery(winner_label="single_suite", delivered_km=None,
                delivered_source=source))

    assert not (tmp_path / "matrices").exists()


def test_inherited_matrix_identity_mismatch_fails_before_measurement(
        tmp_path) -> None:
    backend, target, root = _project(tmp_path)
    delivered = backend.target_test_path(
        str(tmp_path), target
    ).name
    source_hash = hashlib.sha256(SUITE_A.encode("utf-8")).hexdigest()
    expected = {
        "target_source_sha256": hashlib.sha256(
            Path(target).read_bytes()
        ).hexdigest(),
        "execution_name": delivered,
        "catalogue": "full",
    }

    with pytest.raises(ValueError, match="candidate-1 inherited"):
        phase_end(
            backend,
            [SUITE_A, SUITE_SIGN],
            target_file_path=target,
            source_dir=root,
            timeout=120,
            mutation_runner=lambda _directory: pytest.fail(
                "identity mismatch must fail before mutation execution"
            ),
            stage_as_delivered=True,
            inherited1=InheritedMatrix({"mutants": []}, source_hash,
                                       delivered, {**expected, "catalogue": "selective"}),
            measurement_identity=expected,
            log=lambda _message: None,
        )


def test_exact_duplicate_candidates_and_noop_cover_reuse_matrix(
        tmp_path) -> None:
    backend, target, root = _project(tmp_path)
    delivered = backend.target_test_path(
        str(tmp_path), target
    ).name
    target_hash = hashlib.sha256(Path(target).read_bytes()).hexdigest()
    source_hash = hashlib.sha256(SUITE_A.encode("utf-8")).hexdigest()
    identity = {
        "target_source_sha256": target_hash,
        "execution_name": delivered,
        "catalogue": "full",
    }
    matrix = {
        "mutants": [{
            "id": "m0",
            "status": "killed",
            "killer": f"{delivered}::test_add_a",
            "killers": [f"{delivered}::test_add_a"],
            "killing_tests": killing_tests([f"{delivered}::test_add_a"]),
            "line": 2,
        }],
        "summary": {},
    }
    calls = {"count": 0}

    def runner(_directory):
        calls["count"] += 1
        return matrix

    result = phase_end(
        backend,
        [SUITE_A, SUITE_A],
        target_file_path=target,
        source_dir=root,
        timeout=120,
        mutation_runner=runner,
        mutation_method="both",
        stage_as_delivered=True,
        inherited1=InheritedMatrix(matrix, source_hash, delivered, identity),
        inherited2=InheritedMatrix(matrix, source_hash, delivered, identity),
        measurement_identity=identity,
        solver="exact",
        log=lambda _message: None,
    )

    assert calls["count"] == 0
    assert result.merged_source == result.reduced_source
    assert result.merged_km is matrix
    assert result.reduced_km is matrix
    assert result.solver_metadata["selected_ids"] == result.kept_tests

    matrices = tmp_path / "exact-matrices"
    matrices.mkdir()
    (matrices / "consolidation_failure.json").write_text(
        "{}", encoding="utf-8"
    )
    persist_phase_end(
        matrices_dir=matrices, target_file=target, phase=result,
        delivery=PhaseEndDelivery(winner_label="reduced", delivered_km=result.reduced_km,
            delivered_tests=result.selected_tests, delivered_source=result.reduced_source,
            execution_name=delivered))
    assert not (matrices / "consolidation_failure.json").exists()
    record = __import__("json").loads(
        (matrices / "consolidation_record.json").read_text(encoding="utf-8")
    )
    union = __import__("json").loads(
        (matrices / "merged_coverage_matrix.json").read_text(encoding="utf-8")
    )
    assert record["schema_version"] == 2
    assert record["reduction_solver_metadata"]["cbc_options"] == [
        "randomSeed 1", "randomCbcSeed 1",
    ]
    assert union["source_sha256"] == result.merged_source_sha256
    assert union["per_test_runtime_ms"] == result.per_test_cost_ms
    assert union["effective_cost_us"] == result.effective_cost_us


def test_normalization_change_invalidates_an_inherited_candidate_matrix(
        tmp_path) -> None:
    backend, target, root = _project(tmp_path)
    delivered = backend.target_test_path(
        str(tmp_path), target
    ).name
    dirty = (
        "from calc import add\n\n"
        "def test_add_a():\n    assert add(1, 2) == 999\n\n"
        "def test_add_a():\n    assert add(1, 2) == 3\n"
    )
    target_hash = hashlib.sha256(Path(target).read_bytes()).hexdigest()
    dirty_hash = hashlib.sha256(dirty.encode("utf-8")).hexdigest()
    identity = {
        "target_source_sha256": target_hash,
        "execution_name": delivered,
        "catalogue": "full",
    }
    stale = {
        "mutants": [{
            "id": "stale",
            "status": "survived",
            "killer": None,
            "killers": [],
            "line": 2,
        }],
        "summary": {},
    }
    calls = {"count": 0}

    def runner(directory):
        calls["count"] += 1
        return _runner_fixed_catalogue(directory)

    result = phase_end(
        backend,
        [dirty],
        target_file_path=target,
        source_dir=root,
        timeout=120,
        mutation_runner=runner,
        mutation_method="both",
        stage_as_delivered=True,
        inherited1=InheritedMatrix(stale, dirty_hash, delivered, identity),
        measurement_identity=identity,
        log=lambda _message: None,
    )

    assert calls["count"] >= 1
    assert result.input_normalizations[0]["name"] == "test_add_a"
    assert "== 999" not in result.merged_source
    assert all(
        mutant["id"] != "stale"
        for mutant in result.merged_km["mutants"]
    )


def test_phase_end_audits_a_kill_that_never_reproduces(tmp_path):
    """A changed kill identity does not rewrite either measured matrix."""
    calls = {"n": 0}

    def flipping_runner(directory):
        calls["n"] += 1
        km = _runner_fixed_catalogue(directory)
        pyfile = next(Path(directory).glob("test_*.py"))
        present = set(re.findall(
            r"def (test_\w+)", pyfile.read_text(encoding="utf-8")))
        flip = (
            {"id": "flip", "status": "killed",
             "killer": f"{pyfile.name}::test_sign_all",
             "killers": [f"{pyfile.name}::test_sign_all"],
             "killing_tests": killing_tests([f"{pyfile.name}::test_sign_all"]), "line": None}
            if calls["n"] == 1 and "test_sign_all" in present else
            {"id": "flip", "status": "survived", "killer": None,
             "killers": [], "line": None})
        km["mutants"].append(flip)
        return km

    backend, target, root = _project(tmp_path)
    logs = []
    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=flipping_runner, mutation_method="both",
        log=logs.append,
    )
    flip_merged = next(
        m for m in res.merged_km["mutants"] if m["id"] == "flip")
    assert flip_merged["status"] == "killed"
    assert flip_merged["killers"]
    assert not any("adjudicated unreproducible" in line for line in logs)
    assert ("test_add_a" in res.reduced_source) != (
        "test_add_b" in res.reduced_source)
    from testuary.languages import mutation as km_mod
    assert km_mod.matrix_score_percent(res.merged_km) == 75.0
    assert km_mod.matrix_score_percent(res.reduced_km) == 50.0
    assert res.selection_audit["reduction_missing_merged_kills"] == 1


def test_an_incomplete_reduced_measurement_leaves_the_merged_score(tmp_path):
    """A reduced measurement that tested no mutant is missing evidence, not
    survival, so the merged suite keeps its measured kills and score."""
    def runner(directory):
        matrix = _runner_fixed_catalogue(directory)
        if Path(directory).name == "reduced":
            for mutant in matrix["mutants"]:
                mutant.update(status="untested", killer=None, killers=[])
        return matrix

    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=runner, mutation_method="both", log=lambda m: None,
    )
    from testuary.languages import mutation as km
    assert km.matrix_score_percent(res.merged_km) == 100.0 * 2 / 3


SUITE_PREP = ("from calc import add\n\n"
              "def test_prepare():\n    assert add(0, 1) == 1\n")


def test_a_kill_that_needs_a_removed_preparatory_test_stays_a_kill(tmp_path):
    """test_sign_all kills ``prep`` only after test_prepare ran in the same
    session. The cover drops test_prepare, so the reduced suite never kills
    ``prep``. That is a dependency on a removed test, not an unreproducible
    kill, so the merged suite's measured kill stands."""
    def runner(directory):
        pyfile = next(Path(directory).glob("test_*.py"))
        present = set(re.findall(r"def (test_\w+)",
                                 pyfile.read_text(encoding="utf-8")))
        matrix = _runner_fixed_catalogue(directory)
        killed = {"test_sign_all", "test_prepare"} <= present
        matrix["mutants"].append({
            "id": "prep", "status": "killed" if killed else "survived",
            "killer": f"{pyfile.name}::test_sign_all" if killed else None,
            "killers": [f"{pyfile.name}::test_sign_all"] if killed else [],
            "killing_tests": killing_tests([f"{pyfile.name}::test_sign_all"] if killed else []),
            "line": None})
        return matrix

    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [SUITE_A, SUITE_SIGN, SUITE_PREP],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=runner, mutation_method="both", log=lambda m: None,
    )
    assert "test_prepare" not in res.reduced_source
    prep = next(m for m in res.merged_km["mutants"] if m["id"] == "prep")
    assert prep["status"] == "killed"


def test_record_carries_the_collision_resolution(tmp_path):
    """The consolidation record reports the merge's collision resolution, so
    an audit reads a cross-source drop from a record field, not the log."""
    import json

    backend, target, root = _project(tmp_path)
    colliding = ("from calc import add\n\n"
                 "def test_add_a():\n    assert add(2, 2) == 4\n")
    res = phase_end(
        backend, [SUITE_A, colliding],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="derive",
        log=lambda m: None,
    )
    assert res.collision_resolution is not None
    assert res.collision_resolution["renamed"] == 1
    assert res.collision_resolution["renamed_names"] == ["test_add_a_2"]
    assert res.collision_resolution["dropped"] == []
    mdir = tmp_path / "matrices"
    persist_phase_end(
        matrices_dir=mdir, target_file=target, phase=res,
        delivery=PhaseEndDelivery(winner_label="merged", delivered_km=res.reduced_km))
    record = json.loads(
        (mdir / "consolidation_record.json").read_text(encoding="utf-8"))
    assert record["collision_resolution"]["renamed"] == 1
    assert record["collision_resolution"]["dropped"] == []


def test_a_merged_suite_refused_by_validation_keeps_its_merge_record(tmp_path, monkeypatch):
    """What the merge did stays on record when its suite fails validation."""
    backend, target, root = _project(tmp_path)
    colliding = ("from calc import add\n\n"
                 "def test_add_a():\n    assert add(2, 2) == 4\n")
    real = consolidate._finalize_and_validate_source

    def refuse(backend, source, source_dir):
        finalized, validation, _error = real(backend, source, source_dir)
        return finalized, validation, "target validator refused: fixture"

    monkeypatch.setattr(consolidate, "_finalize_and_validate_source", refuse)
    res = phase_end(
        backend, [SUITE_A, colliding],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="derive",
        log=lambda m: None,
    )
    assert res.detail.startswith("the complete merged suite failed validation")
    assert res.collision_resolution["renamed"] == 1


def test_no_matrix_remeasure_leaves_an_incomplete_reduction(
        tmp_path):
    """A green reduction with no matrix is excluded by the outer selector."""
    backend, target, root = _project(tmp_path)
    agent_with_redundant_test = (
        SUITE_SIGN
        + "\ndef test_no_target_contact():\n    assert True\n"
    )
    calls = {"n": 0}

    def runner(directory):
        calls["n"] += 1
        # Call order at a two-candidate boundary: candidate 2 standalone, the
        # merged suite, then the reduced re-measure, which refuses.
        if calls["n"] >= 3:
            return None
        return _runner_fixed_catalogue(directory)

    res = phase_end(
        backend, [SUITE_A, agent_with_redundant_test],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=runner, mutation_method="both",
        log=lambda m: None,
    )
    assert calls["n"] >= 3                       # the re-measure ran and refused
    assert res.fallback_reason is None
    assert res.reduced_km is None
    assert res.reduction_valid is True
    assert res.selected_tests == 2
    assert "test_add_a" in res.reduced_source
    assert "test_sign_all" in res.reduced_source
    assert res.selection_audit[
        "reduction_measurement_issue"
    ] == "reduced_remeasure_no_matrix"


def test_phase_end_refresh_failure_leaves_an_incomplete_reduction(
    tmp_path, monkeypatch,
):
    """A reduction without aggregate coverage is excluded from selection."""
    from testuary import consolidation as cmod

    backend, target, root = _project(tmp_path)
    monkeypatch.setattr(
        cmod, "_refresh_reduced_coverage", lambda *a, **k: False)
    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=None, log=lambda m: None,
    )
    assert res.reduced_source != res.merged_source
    assert res.selected_tests == 2
    assert res.pool_tests == 3
    assert res.line is None
    assert res.branch is None
    assert res.fallback_reason is None
    assert res.selection_audit[
        "reduction_measurement_issue"
    ] == "reduced_coverage_refresh_failed"


def test_a_reduction_whose_coverage_refresh_failed_leaves_the_selection(
    tmp_path, monkeypatch,
):
    """A reduction whose aggregate coverage could not be measured again is
    excluded as a whole, as a red reduction is: not a valid reduction and
    no matrix of its own."""
    from testuary import consolidation as cmod

    backend, target, root = _project(tmp_path)
    monkeypatch.setattr(
        cmod, "_refresh_reduced_coverage", lambda *a, **k: False)
    runner, calls = _runner_two_phase(_runner_fixed_catalogue)
    res = phase_end(
        backend, [SUITE_A, SUITE_B, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=runner, mutation_method="measure", log=lambda m: None,
    )
    assert calls["n"] == 2                       # merged and reduced measured
    assert res.reduced_source != res.merged_source
    assert res.reduction_valid is False
    assert res.reduced_km is None
    assert (res.line, res.branch) == (None, None)
    assert res.selection_audit[
        "reduction_measurement_issue"
    ] == "reduced_coverage_refresh_failed"


def test_every_multi_source_boundary_call_passes_candidate_evidence():
    """Every PhaseEndOptions built for a phase end over more than one
    source, a PhaseEnd's or a direct consolidate_phase_end call's, names
    ``candidate_evidence``, so collision victims are chosen by evidence at
    every boundary. run_phase_end passes a PhaseEnd's options on with
    replace(end.options, ...). A single-source merge has no cross-source
    collisions."""
    phases_dir = Path(consolidate_phase_end.__code__.co_filename).parent
    offenders = []

    def keyword(node, name):
        return next((k.value for k in node.keywords if k.arg == name), None)

    for module in sorted(phases_dir.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name == "consolidate_phase_end":
                sources = node.args[1] if len(node.args) > 1 else None
                options = node.args[3] if len(node.args) > 3 else keyword(node, "options")
            elif name == "PhaseEnd":
                sources, options = keyword(node, "sources"), keyword(node, "options")
            else:
                continue
            if isinstance(sources, ast.List) and len(sources.elts) == 1:
                continue
            passed_on = (isinstance(options, ast.Call) and options.args
                         and getattr(options.args[0], "attr", "") == "options")
            keywords = {k.arg for k in getattr(options, "keywords", [])}
            if not passed_on and "candidate_evidence" not in keywords:
                offenders.append(f"{module.name}:{node.lineno}")
    assert not offenders, (
        "multi-source consolidate_phase_end call(s) without "
        f"candidate_evidence: {offenders}"
    )

# ---------------------------------------------------------------------------
# A target validator that judges nothing at the phase end ends the phase
# ---------------------------------------------------------------------------

def _is_merged(source):
    """The merge of SUITE_A and SUITE_SIGN: both tests in one source."""
    return "def test_add_a" in source and "def test_sign_all" in source


def test_a_merge_the_validator_does_not_judge_ends_the_phase(
        tmp_path, monkeypatch):
    """A validator that times out or does not start says nothing about the
    merged code, so the phase ends rather than dropping the merge as invalid."""
    from validator_fixtures import patch_validator, unjudged
    from testuary.consolidation import ValidationUnresolved

    patch_validator(monkeypatch, lambda source, real: (
        unjudged("timeout") if _is_merged(source) else real()))
    backend, target, root = _project(tmp_path)
    with pytest.raises(ValidationUnresolved, match="timeout"):
        phase_end(
            backend, [SUITE_A, SUITE_SIGN],
            target_file_path=target, source_dir=root, timeout=120,
            mutation_runner=_runner_fixed_catalogue, mutation_method="both",
            log=lambda m: None,
        )


def test_a_merge_judged_on_a_later_attempt_goes_on(tmp_path, monkeypatch):
    """A validator asked again that judges the merge on its last attempt lets
    the consolidation go on as without the failures."""
    from validator_fixtures import patch_validator, unjudged

    failures = {"left": 2}

    def judge(source, real):
        if _is_merged(source) and failures["left"]:
            failures["left"] -= 1
            return unjudged()
        return real()

    patch_validator(monkeypatch, judge)
    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [SUITE_A, SUITE_SIGN],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        log=lambda m: None,
    )
    assert failures["left"] == 0
    assert res.merged_source is not None
    assert _is_merged(res.merged_source)


def test_a_source_the_structural_parser_rejects_is_not_validated():
    """The structural parser's rejection is a verdict on the code, so the
    target validator is not asked."""
    from testuary.consolidation import _finalize_and_validate_source

    class Backend:
        def finalize_test_source(self, source):
            return source

        def check_structure(self, source):
            return SimpleNamespace(ok=False, message="unbalanced block")

        def validate_test_source(self, source, source_dir):
            raise AssertionError("the validator was asked")

    finalized, validation, error = _finalize_and_validate_source(
        Backend(), "def test_x(:\n", "root")
    assert (finalized, validation) == ("def test_x(:\n", None)
    assert error == "structural parser: unbalanced block"


@pytest.mark.parametrize("method", ["both", "derive"])
def test_a_reduction_of_the_inherited_suite_binds_its_derived_matrix_to_the_reduction(
        tmp_path, method) -> None:
    """The merge reuses the inherited suite's stored matrix, bound to the
    inherited bytes. The reduction drops a redundant test, so the derived
    matrix of the kept tests is written bound to the reduced suite, under
    both and under derive, where it is the reduced suite's matrix."""
    backend, target, root = _project(tmp_path)
    delivered = backend.target_test_path(str(tmp_path), target).name
    inherited = ("from calc import add, sign\n\ndef test_add_a():\n    assert add(1, 2) == 3\n"
                 "\n\ndef test_add_b():\n    assert add(2, 3) == 5\n"
                 "\n\ndef test_sign_all():\n    assert sign(1) == 1\n"
                 "    assert sign(-1) == -1\n    assert sign(0) == 0\n")
    inherited_hash = hashlib.sha256(inherited.encode("utf-8")).hexdigest()
    identity = {
        "target_source_sha256": hashlib.sha256(Path(target).read_bytes()).hexdigest(),
        "execution_name": delivered,
        "catalogue": "full",
    }
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / delivered).write_text(inherited, encoding="utf-8")
    # As the archive holds it: bound to the inherited bytes.
    stored = {**_runner_fixed_catalogue(staged), "source_sha256": inherited_hash,
              "execution_name": delivered}

    result = phase_end(
        backend, [inherited, inherited],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method=method,
        stage_as_delivered=True,
        inherited1=InheritedMatrix(stored, inherited_hash, delivered, identity),
        measurement_identity=identity,
        log=lambda _message: None,
    )
    assert result.merged_km is stored
    assert result.reduced_source != inherited
    reduced_hash = hashlib.sha256(result.reduced_source.encode("utf-8")).hexdigest()

    matrices = tmp_path / "matrices"
    persist_phase_end(
        matrices_dir=matrices, target_file=target, phase=result,
        delivery=PhaseEndDelivery(winner_label="reduced", delivered_km=result.reduced_km,
            delivered_tests=result.selected_tests, delivered_source=result.reduced_source,
            execution_name=delivered))
    derived = json.loads((matrices / "derived_reduced_kill_matrix.json").read_text(encoding="utf-8"))
    union = json.loads((matrices / "merged_kill_matrix.json").read_text(encoding="utf-8"))
    assert derived["source_sha256"] == reduced_hash
    assert union["source_sha256"] == inherited_hash
    # The stored matrix itself keeps the inherited binding.
    assert stored["source_sha256"] == inherited_hash


def test_a_reused_matrix_whose_reduction_keeps_every_test_is_written_as_before(
        tmp_path) -> None:
    """Without a removed test the reduced suite is the inherited one, and the
    derived matrix keeps the stored matrix's binding."""
    backend, target, root = _project(tmp_path)
    delivered = backend.target_test_path(str(tmp_path), target).name
    inherited = ("from calc import add, sign\n\ndef test_add_a():\n    assert add(1, 2) == 3\n"
                 "\n\ndef test_sign_all():\n    assert sign(1) == 1\n"
                 "    assert sign(-1) == -1\n    assert sign(0) == 0\n")
    inherited_hash = hashlib.sha256(inherited.encode("utf-8")).hexdigest()
    identity = {
        "target_source_sha256": hashlib.sha256(Path(target).read_bytes()).hexdigest(),
        "execution_name": delivered,
        "catalogue": "full",
    }
    staged = tmp_path / "staged"
    staged.mkdir()
    (staged / delivered).write_text(inherited, encoding="utf-8")
    # A version-3 matrix, as LangAdapter writes it: no upgrade reorders it.
    stored = {**_runner_fixed_catalogue(staged), "schema_version": 3,
              "source_sha256": inherited_hash, "execution_name": delivered}

    result = phase_end(
        backend, [inherited, inherited],
        target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        stage_as_delivered=True,
        inherited1=InheritedMatrix(stored, inherited_hash, delivered, identity),
        measurement_identity=identity,
        log=lambda _message: None,
    )
    assert result.reduced_source == inherited
    matrices = tmp_path / "matrices"
    persist_phase_end(
        matrices_dir=matrices, target_file=target, phase=result,
        delivery=PhaseEndDelivery(winner_label="reduced", delivered_km=result.reduced_km,
            delivered_tests=result.selected_tests, delivered_source=result.reduced_source,
            execution_name=delivered))
    derived = json.loads((matrices / "derived_reduced_kill_matrix.json").read_text(encoding="utf-8"))
    assert derived["source_sha256"] == inherited_hash
    # The stored matrix's own key order: its binding before the derivation.
    assert list(derived)[-3:] == ["source_sha256", "execution_name", "derived"]


def _red_only_under_coverage(monkeypatch, backend, test="test_add_a"):
    """The backend's runs pass, but every run under the coverage tool fails
    in ``test``."""
    from testuary.languages import TestRunResult
    from testuary.languages.generic import TestFailure

    real = type(backend).run_test_files

    def run(self, paths, source_dir, *, timeout, measure_coverage_of=None, **kwargs):
        if measure_coverage_of is None:
            return real(self, paths, source_dir, timeout=timeout, **kwargs)
        return TestRunResult(
            ok=False, returncode=1, timed_out=False, elapsed_s=0.1, collected=1,
            passed=0, failed=1, errors=0, output=f"FAILED {test}",
            line_coverage=None, branch_coverage=None,
            failures=(TestFailure(test, "failed", "fails only under the tool"),))

    monkeypatch.setattr(type(backend), "run_test_files", run)


def test_a_merged_suite_the_coverage_tool_cannot_measure_is_incomplete(
        tmp_path, monkeypatch):
    """The merged suite passes its normal run, but the coverage tool's run
    of it fails: it has no numbers, so it is no candidate, and its record
    says incomplete with the test that stopped the tool, not failing."""
    backend, target, root = _project(tmp_path)
    _red_only_under_coverage(monkeypatch, backend)
    res = phase_end(
        backend, [SUITE_A, SUITE_B], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=None, log=lambda m: None)
    assert res.merged_green is True
    assert "could not measure the merged suite" in res.detail
    assert "test_add_a" in res.detail
    scored = consolidate.merged_scored_from_phase(res)
    assert scored.status == "incomplete" and scored.status_reason == res.detail


def test_a_candidate_the_coverage_tool_cannot_measure_is_incomplete(tmp_path, monkeypatch):
    """A phase's own suite that passes its normal run but that the coverage
    tool could not measure is incomplete, with the test that stopped the
    tool, not failing."""
    backend, target, root = _project(tmp_path)
    _red_only_under_coverage(monkeypatch, backend)
    scored = consolidate.measure_candidate_standalone(
        backend, SUITE_A, target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=None, stage_as_delivered=False, log=lambda m: None)
    assert scored.green is True and scored.status == "incomplete"
    assert scored.status_reason.startswith("the coverage tool could not measure it")
    assert "test_add_a" in scored.status_reason


def test_a_candidate_whose_per_test_coverage_run_fails_is_scored_on_its_coverage_run(
        tmp_path, monkeypatch):
    """A test failing only in the per-test coverage run, after the passing
    normal run, leaves the phase's own suite without per-test evidence: it
    is scored on its coverage run, as for a language without per-test
    evidence, with the run's report recorded, not rejected."""
    backend, target, root = _project(tmp_path)

    def red_eviction(_self, *_args, failure_fn=None, **_kwargs):
        failure_fn("test_failure", "measured run exited 1: 1 failed")
        return None

    monkeypatch.setattr(type(backend), "collect_eviction_data", red_eviction)
    scored = consolidate.measure_candidate_standalone(
        backend, SUITE_A, target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=None, stage_as_delivered=False, log=lambda m: None)
    assert scored.green is True and scored.status == "passing"
    assert scored.line is not None and scored.per_test_tokens is None
    assert "measured run exited 1: 1 failed" in scored.status_reason


def test_a_candidate_of_a_backend_without_per_test_evidence_records_no_failure(
        tmp_path, monkeypatch):
    """A backend that declares no per-test evidence gives none and no
    per-test run fails: the phase's own suite is scored on its coverage
    run and its record names no failure."""
    backend, target, root = _project(tmp_path)
    supports = type(backend)._supports
    monkeypatch.setattr(type(backend), "_supports", lambda self, operation, source_dir=None: (
        operation != "tests.per_test_evidence" and supports(self, operation, source_dir)))
    scored = consolidate.measure_candidate_standalone(
        backend, SUITE_A, target_file_path=target, source_dir=root, timeout=120,
        mutation_runner=None, stage_as_delivered=False, log=lambda m: None)
    assert scored.green is True and scored.status == "passing"
    assert scored.line is not None and scored.per_test_tokens is None
    assert not scored.status_reason


def test_rename_twins_are_found_inside_a_renamed_class():
    """A twin is a copy of a test that the merge's collision rename kept
    beside the original: the test itself renamed, or its class. Ids are
    dotted in both languages. A model's own numbering is no rename, also
    with a suffix the rename uses, and a test of a renamed class that the
    original class lacks is no copy."""
    names = ["tests.test_calc.TestCalc.test_adds", "tests.test_calc.TestCalc_2.test_adds",
             "tests.test_calc.TestCalc_2.test_mul",
             "tests.test_calc.test_subtracts", "tests.test_calc.test_subtracts_2",
             "tests.test_calc.test_case_1", "tests.test_calc.test_case",
             "tests.test_calc.test_case_2",
             "demo.CalcTest.Inner.adds", "demo.CalcTest.Inner_3.adds", "demo.CalcTest.adds"]
    stats = consolidate._union_bloat_stats(
        {name: frozenset() for name in names},
        renamed_names=["TestCalc_2", "test_subtracts_2", "Inner_3"])
    assert stats["rename_twins"] == 3


def _cleanup_pending_error():
    """An error whose mutation session's cleanup failed, as a failed close
    marks it."""
    error = RuntimeError("the session did not close")
    error.mutation_cleanup_handle = object()
    return error


@pytest.mark.parametrize("error", [
    RuntimeError("boom"), NotImplementedError("no standalone measurement")])
def test_a_raising_candidate2_measurement_is_recorded_and_the_phase_goes_on(
        tmp_path, monkeypatch, error):
    """Any other error of candidate 2's standalone measurement, a
    NotImplementedError among them, becomes its status reason. Its fields
    stay empty and the merged measurement goes on."""
    backend, target, root = _project(tmp_path)

    def raising(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(consolidate, "measure_candidate_standalone", raising)
    logs = []
    res = phase_end(
        backend, [SUITE_A, SUITE_SIGN], target_file_path=target,
        source_dir=root, timeout=120, mutation_runner=None, log=logs.append)
    assert res.candidate2_status_reason == (
        f"standalone measurement raised {type(error).__name__}: {error}")
    assert (res.candidate2_km, res.candidate2_coverage, res.candidate2_source) == (
        None, None, None)
    assert res.candidate2_green is False and res.candidate2_execution_name == ""
    assert (f"[consolidate] candidate-2 standalone measurement raised ({error}), "
            "so the candidate-2 sidecars stay absent") in logs
    assert res.line is not None


@pytest.mark.parametrize("make_error", [
    lambda: ValueError("the inherited matrix is bound to other bytes"),
    _cleanup_pending_error,
])
def test_a_candidate2_value_error_or_pending_cleanup_ends_the_phase_end(
        tmp_path, monkeypatch, make_error):
    """A ValueError, such as an inherited matrix bound to other bytes, and
    an error whose mutation session's cleanup failed are raised as they
    are."""
    backend, target, root = _project(tmp_path)
    error = make_error()

    def raising(*_args, **_kwargs):
        raise error

    monkeypatch.setattr(consolidate, "measure_candidate_standalone", raising)
    with pytest.raises(type(error)) as raised:
        phase_end(
            backend, [SUITE_A, SUITE_SIGN], target_file_path=target,
            source_dir=root, timeout=120, mutation_runner=None,
            log=lambda _message: None)
    assert raised.value is error


# The removal fallback: a measurement that fails a suite after its
# passing normal run removes the tests it names at the phase end's places.

SUITE_A_SIGN = ("from calc import add, sign\n\ndef test_add_a():\n    assert add(1, 2) == 3\n\n\n"
                "def test_sign_all():\n    assert sign(1) == 1\n    assert sign(-1) == -1\n"
                "    assert sign(0) == 0\n")
SUITE_B_SIGN = SUITE_A_SIGN.replace("test_add_a", "test_add_b").replace("add(1, 2) == 3",
                                                                        "add(2, 3) == 5")


class _NamingRunner:
    """The fixed catalogue's runner, except that a suite holding ``test``
    gives no matrix and its checks name that test (``named_of``)."""

    def __init__(self, test):
        self.test, self.named = test, {}

    def __call__(self, directory):
        pyfile = next(Path(directory).glob("test_*.py"))
        held = f"def {self.test}(" in pyfile.read_text(encoding="utf-8")
        self.named[str(directory)] = (named_id(pyfile, self.test),) if held else ()
        return None if held else _runner_fixed_catalogue(directory)

    def named_of(self, directory):
        return self.named.get(str(directory), ())


def test_a_candidate_loses_the_test_its_coverage_run_names(tmp_path, monkeypatch):
    """The phase's own suite whose coverage run fails a test after its
    passing normal run loses that test with its outcome, is measured again
    without it, the merge takes it, and its changed source is archived
    while it is not delivered."""
    backend, target, root = _project(tmp_path)
    named_under_coverage(monkeypatch, backend, ["test_add_a"], outcome="error")
    res = phase_end(
        backend, [SUITE_B, SUITE_A_SIGN], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        log=lambda m: None)
    assert res.candidate2_candidate.status == "passing"
    assert "def test_add_a" not in res.candidate2_source
    assert "def test_sign_all" in res.candidate2_source
    assert "def test_add_a" not in res.merged_source
    [entry] = res.measurement_removals
    assert (entry["place"], entry["measurement"], entry["round"]) == (
        "phase_end.candidate", "coverage", 1)
    [removed] = entry["removed"]
    assert removed["test"].endswith(".test_add_a") and removed["outcome"] == "error"
    archive = tmp_path / "archive"
    persist_phase_end(
        matrices_dir=archive, target_file=target, phase=res,
        delivery=PhaseEndDelivery(winner_label="reduced", delivered_km=res.reduced_km,
            delivered_tests=res.selected_tests, delivered_source=res.reduced_source,
            execution_name=res.execution_name))
    record = json.loads((archive / "consolidation_record.json").read_text(encoding="utf-8"))
    assert record["measurement_removals"] == res.measurement_removals
    kept = archive / "candidate_standalone_1_removed.py"
    assert kept.read_text(encoding="utf-8") == res.candidate2_source
    consolidate._phase_end_dir(archive)
    assert not kept.exists()


def test_a_starter_end_of_two_suites_merges_the_second_without_its_removed_test(
        tmp_path, monkeypatch):
    """The starter phase's end with exactly two passing suites and their
    pool evidence measures the second suite alone, as every phase end with
    two candidates does. A test that measurement names is removed there,
    the merge takes the suite without it, the removal names its place, and
    the merged suite's numbers come from its own measurement, not from the
    pool evidence, which still names the removed test."""
    backend, target, root = _project(tmp_path)
    named_under_coverage(monkeypatch, backend, ["test_add_a"])
    pool = [{"test_add_b": ["L:901"]},
            {"test_add_a": ["L:902", "B:902->903"], "test_sign_all": ["L:904"]}]
    res = phase_end(
        backend, [SUITE_B, SUITE_A_SIGN], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        candidate_evidence=pool, stage_as_delivered=True, log=lambda m: None)
    assert "def test_add_a" not in res.merged_source
    assert "def test_add_b" in res.merged_source and "def test_sign_all" in res.merged_source
    [entry] = res.measurement_removals
    assert (entry["place"], entry["measurement"]) == ("phase_end.candidate", "coverage")
    [removed] = entry["removed"]
    assert removed["test"].endswith(".test_add_a")
    assert res.aggregate_coverage_measured and res.pool_tests == 2
    tokens = {token for found in res.per_test_tokens.values() for token in found}
    assert not any("test_add_a" in test for test in res.per_test_tokens)
    assert not tokens & {"L:901", "L:902", "B:902->903", "L:904"}
    assert res.merged_km is not None and "test_add_a" not in json.dumps(res.merged_km)


SUITE_REFLECTIVE_ONE = ("from calc import add\n\n\ndef helper():\n    return add(1, 2)\n\n\n"
                        "def test_one():\n    assert globals()['helper']() == 3\n")
SUITE_REFLECTIVE_TWO = ("from calc import add, sign\n\n\ndef helper():\n    return sign(1)\n\n\n"
                        "def test_two():\n    assert globals()['helper']() == 1\n\n\n"
                        "def test_removed():\n    assert add(2, 2) == 4\n")


def test_a_removed_test_leaves_the_pool_evidence_that_weighs_a_binding(tmp_path, monkeypatch):
    """Two starter suites bind helper differently and read it through
    globals(), so neither can rename it and the merge gives it to the
    suite whose tests carry the most kills in the pool evidence. In a
    language that cannot tell a name's dependent tests (Java, C) every test
    of the evidence counts. A test the second suite loses to its own
    measurement leaves its entries first, so its kills do not win the
    binding, and the first suite's helper and test stay."""
    from langadapter.python.adapter import PythonAdapter

    monkeypatch.setattr(PythonAdapter, "collision_dependents",
                        lambda self, source, name, measured: None)
    backend, target, root = _project(tmp_path)
    named_under_coverage(monkeypatch, backend, ["test_removed"])
    pool = [{"test_candidate.test_one": ["K:1", "L:2"]},
            {"test_candidate.test_two": ["L:3"],
             "test_candidate.test_removed": ["K:2", "K:3", "K:4"]}]
    res = phase_end(
        backend, [SUITE_REFLECTIVE_ONE, SUITE_REFLECTIVE_TWO], target_file_path=target,
        source_dir=root, timeout=120, candidate_evidence=pool, stage_as_delivered=True,
        log=lambda m: None)
    [entry] = res.measurement_removals
    assert entry["place"] == "phase_end.candidate"
    assert "def test_one" in res.merged_source and "return add(1, 2)" in res.merged_source
    assert "def test_removed" not in res.merged_source


def test_a_candidate_loses_the_test_its_mutation_check_names(tmp_path):
    """A mutation measurement that names a test of the phase's own suite
    removes it, recorded without an outcome, and measures the suite again."""
    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [SUITE_B, SUITE_A_SIGN], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=_NamingRunner("test_sign_all"), mutation_method="both",
        log=lambda m: None)
    [entry] = res.measurement_removals
    assert (entry["place"], entry["measurement"]) == ("phase_end.candidate", "mutation")
    [removed] = entry["removed"]
    assert removed["test"].endswith(".test_sign_all") and "outcome" not in removed
    assert res.candidate2_km is not None and "def test_sign_all" not in res.candidate2_source


def test_a_candidate_measured_again_after_a_removal_reuses_no_inherited_matrix(
        tmp_path, monkeypatch):
    """The inherited matrix bound to the suite before its removal is not
    used for the suite after it: the mutation tool measures that suite."""
    backend, target, root = _project(tmp_path)
    named_under_coverage(monkeypatch, backend, ["test_add_a"])
    delivered = backend.target_test_path(str(tmp_path), target).name
    expected = {"target_source_sha256": hashlib.sha256(Path(target).read_bytes()).hexdigest(),
                "execution_name": delivered, "catalogue": "full"}
    inherited = {"mutants": [], "summary": {}}
    res = phase_end(
        backend, [SUITE_B, SUITE_A_SIGN], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        stage_as_delivered=True, measurement_identity=expected, log=lambda m: None,
        inherited2=InheritedMatrix(
            inherited, hashlib.sha256(SUITE_A_SIGN.encode("utf-8")).hexdigest(),
            delivered, expected))
    assert res.candidate2_km is not None and res.candidate2_km is not inherited
    assert res.candidate2_km["mutants"]


def test_a_merged_suite_loses_the_test_its_coverage_run_names(tmp_path, monkeypatch):
    """The merged suite loses the test its coverage run fails: the merged
    record names the suite after the removal, validity_merged.json binds it
    as built, and the delivered reduced suite's chain runs through the
    removal and the set cover, so a reuse accepts the archive."""
    backend, target, root = _project(tmp_path)
    named_under_coverage(monkeypatch, backend, ["test_add_a"])
    res = phase_end(
        backend, [SUITE_A, SUITE_B_SIGN], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        log=lambda m: None)
    built = res.merged_validity_record["final_clean_source_sha256"]
    assert "def test_add_a" not in res.merged_source and not res.merged_incomplete
    assert res.merged_source_sha256 == hashlib.sha256(
        res.merged_source.encode("utf-8")).hexdigest() != built
    [entry] = res.measurement_removals
    assert (entry["place"], entry["measurement"], entry["round"]) == (
        "phase_end.merged", "coverage", 1)
    assert entry["removed"][0]["outcome"] == "failed"
    [record] = res.removal_derivations
    assert (record["operation"], record["parent_source_sha256"],
            record["child_source_sha256"]) == ("measurement_remove", built,
                                               res.merged_source_sha256)
    archive = tmp_path / "archive"
    persist_phase_end(
        matrices_dir=archive, target_file=target, phase=res,
        delivery=PhaseEndDelivery(winner_label="reduced", delivered_km=res.reduced_km,
            delivered_tests=res.selected_tests, delivered_source=res.reduced_source,
            execution_name=res.execution_name))
    chain = json.loads((archive / "validity_derivations.json").read_text(
        encoding="utf-8"))["records"]
    assert [item["operation"] for item in chain] == [
        "measurement_remove", "set_cover_remove_restore"]
    from testuary import evidence

    assert evidence.verify_delivered_archive(
        archive_dir=archive, source=res.reduced_source,
        identity=backend.generated_test_target_identity(target, root)) == (True, "")


#: A plain test beside a parametrized test with two cases.
SUITE_CASES = ("import pytest\nfrom calc import add\n\n"
               "def test_add_b():\n    assert add(2, 3) == 5\n\n\n"
               "@pytest.mark.parametrize('a', [1, 2])\n"
               "def test_case(a):\n    assert add(a, 0) == a\n")
#: A plain test beside a parametrized test whose cases come from a name.
SUITE_NAMED_CASES = ("import pytest\nfrom calc import add\n\nCASES = [1, 2]\n\n\n"
                     "def test_add_b():\n    assert add(2, 3) == 5\n\n\n"
                     "@pytest.mark.parametrize('a', CASES)\n"
                     "def test_case(a):\n    assert add(a, 0) == a\n")
#: A suite whose module refers to one of its tests by name.
SUITE_ALIAS = ("from calc import add\n\n"
               "def test_add_b():\n    assert add(2, 3) == 5\n\n\n"
               "def test_add_a():\n    assert add(1, 2) == 3\n\n\n"
               "CHECKS = [test_add_a]\n")


def _candidate_entries(res):
    return [entry for entry in res.measurement_removals
            if entry["place"] == "phase_end.candidate"]


def test_a_candidate_loses_a_named_case_alone_and_records_the_names_it_does_not_hold(
        tmp_path, monkeypatch):
    """A coverage run that names a case of a parametrized test, a case of a
    test the suite does not hold and a test it does not hold: the phase's
    own suite loses that case alone and keeps the test's other case, and the
    entry lists the removed case and the two names the suite does not
    hold."""
    backend, target, root = _project(tmp_path)
    names_under_coverage(monkeypatch, backend, "[1, 2]",
                         [("test_case", "2"), ("test_gone", "3"), ("test_other", None)])
    res = phase_end(
        backend, [SUITE_A, SUITE_CASES], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=None, log=lambda m: None)
    [entry] = _candidate_entries(res)
    assert [item["test"].rsplit(".", 1)[-1] for item in entry["removed"]] == ["test_case[2]"]
    assert [name.rsplit(".", 1)[-1] for name in entry["unmatched"]] == [
        "test_gone[3]", "test_other"]
    assert "detail" not in entry
    assert "def test_case" in res.candidate2_source and "[1, 2]" not in res.candidate2_source
    assert res.candidate2_candidate.status == "passing"


def test_a_named_case_whose_test_goes_whole_is_recorded_with_why(tmp_path, monkeypatch):
    """A named case of a test whose cases come from a name cannot go alone:
    the whole test goes, and the entry says why beside the removed case."""
    backend, target, root = _project(tmp_path)
    names_under_coverage(monkeypatch, backend, "def test_case(", [("test_case", "2")])
    res = phase_end(
        backend, [SUITE_A, SUITE_NAMED_CASES], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=None, log=lambda m: None)
    [entry] = _candidate_entries(res)
    assert [item["test"].rsplit(".", 1)[-1] for item in entry["removed"]] == ["test_case[2]"]
    assert "test_case is removed whole" in entry["detail"]
    assert "def test_case" not in res.candidate2_source


def test_a_suite_that_no_longer_loads_after_a_removal_is_recorded_as_failing(
        tmp_path, monkeypatch):
    """A suite that no longer loads once the named test is gone still holds
    a test: the entry after the removal says that the normal run failed,
    not that the removal left no test."""
    backend, target, root = _project(tmp_path)
    named_under_coverage(monkeypatch, backend, ["test_add_a"])
    res = phase_end(
        backend, [SUITE_SIGN, SUITE_ALIAS], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=None, log=lambda m: None)
    assert [entry.get("unnamed") for entry in _candidate_entries(res)] == [
        None, "the normal run failed after a removal"]


def test_a_merged_suite_whose_coverage_matrix_cannot_be_read_records_why(
        tmp_path, monkeypatch):
    """The merged suite's coverage run passes but leaves a matrix that
    cannot be read: the suite is incomplete as today, and its place records
    the cause in an unnamed entry."""
    backend, target, root = _project(tmp_path)
    real = type(backend).run_test_files

    def run(self, paths, source_dir, *, timeout, coverage_matrix_path=None, **kwargs):
        result = real(self, paths, source_dir, timeout=timeout,
                      coverage_matrix_path=coverage_matrix_path, **kwargs)
        if coverage_matrix_path is not None:
            Path(coverage_matrix_path).write_text("not json", encoding="utf-8")
        return result

    monkeypatch.setattr(type(backend), "run_test_files", run)
    res = phase_end(
        backend, [SUITE_A, SUITE_B], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=None, log=lambda m: None)
    assert res.merged_incomplete == (
        "the merged suite's aggregate coverage matrix could not be persisted")
    assert res.measurement_removals == [
        {"place": "phase_end.merged", "measurement": "coverage", "round": 1,
         "unnamed": "the coverage matrix could not be read"}]


def test_a_mutation_run_the_wall_clock_cut_records_a_timeout(tmp_path, monkeypatch):
    """A mutation run of the phase's own suite and of the merged suite that
    the wall clock cut gives a matrix stamped incomplete: nothing is removed,
    and each place records the timeout in an unnamed entry."""
    from testuary.languages.mutation import INCOMPLETE_KEY

    backend, target, root = _project(tmp_path)

    def runner(directory):
        matrix = _runner_fixed_catalogue(directory)
        pyfile = next(Path(directory).glob("test_*.py"))
        if "def test_sign_all(" in pyfile.read_text(encoding="utf-8"):
            matrix[INCOMPLETE_KEY] = {"reason": "wall_clock_limit"}
        return matrix

    res = phase_end(
        backend, [SUITE_A, SUITE_SIGN], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=runner, mutation_method="both", log=lambda m: None)
    assert res.measurement_removals == [
        {"place": place, "measurement": "mutation", "round": 1,
         "unnamed": "the measurement timed out"}
        for place in ("phase_end.candidate", "phase_end.merged")]


def test_a_candidate_whose_rounds_are_spent_takes_todays_branch(tmp_path, monkeypatch):
    """A candidate whose coverage run still fails after a removal stops when
    the rounds are spent: the entry after the first round's says so, and the
    candidate is left out as today."""
    from testuary import config as config_module

    real = config_module.get_agent_config
    monkeypatch.setattr(config_module, "get_agent_config", lambda name, parent=None: (
        {**real(name, parent), "failing_test_cleanup_rounds": 1}
        if name == "test_initial" else real(name, parent)))
    backend, target, root = _project(tmp_path)
    names_under_coverage(monkeypatch, backend, "def test_sign_all(", [("test_add_a", None)])
    res = phase_end(
        backend, [SUITE_B, SUITE_A_SIGN], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=None, log=lambda m: None)
    first, last = _candidate_entries(res)
    assert first["round"] == 1 and first["removed"][0]["test"].endswith(".test_add_a")
    assert last == {"place": "phase_end.candidate", "measurement": "coverage", "round": 2,
                    "unnamed": "the rounds are spent"}
    assert res.candidate2_candidate.status == "incomplete"


def test_pythons_origin_refusal_at_a_phase_end_removes_the_test_it_names(
        tmp_path, monkeypatch):
    """The mutation tool's check before the first mutant refuses the phase's
    own suite and names one of its tests: the phase end's runner keeps the
    names the fault carries, the suite loses that test, and the suite
    without it is measured with a kill matrix."""
    from testuary import measurement as metrics
    from testuary.languages.mutation import MutationInstrumentFault

    backend, target, root = _project(tmp_path)

    class Measured:
        def __init__(self, directory):
            self.matrix = _runner_fixed_catalogue(directory)

        def kill_matrix(self):
            return self.matrix

    class Session:
        def run(self, *, test_dir, test_files, **_arguments):
            if "def test_add_a(" in Path(test_files[0]).read_text(encoding="utf-8"):
                fault = MutationInstrumentFault("the origin check refused")
                fault.stopping_tests = (named_id(test_files[0], "test_add_a"),)
                raise fault
            return Measured(test_dir)

    runner = metrics.make_finishing_mutation_fn(
        session=Session(), target_file=target, source_dir=root,
        configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0)
    res = phase_end(
        backend, [SUITE_SIGN, SUITE_A_SIGN], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=runner, mutation_method="both", log=lambda m: None)
    [entry] = _candidate_entries(res)
    assert (entry["measurement"], entry["round"]) == ("mutation", 1)
    assert [item["test"].rsplit(".", 1)[-1] for item in entry["removed"]] == ["test_add_a"]
    assert res.candidate2_km is not None and "def test_add_a" not in res.candidate2_source


def test_the_reduced_suites_mutation_failure_removes_nothing(tmp_path, monkeypatch):
    """A mutation run of the reduced suite that measures nothing and names a
    test removes nothing: the reduced suite's own measurement leaves the
    selection as today, and no entry is written."""
    backend, target, root = _project(tmp_path)
    named = {}

    def runner(directory):
        if Path(directory).name != "reduced":
            return _runner_fixed_catalogue(directory)
        pyfile = next(Path(directory).glob("test_*.py"))
        named[str(directory)] = (named_id(pyfile, "test_sign_all"),)
        return None

    runner.named_of = lambda directory: named.get(str(directory), ())
    res = phase_end(
        backend, [SUITE_A, SUITE_B_SIGN], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=runner, mutation_method="both", log=lambda m: None)
    assert named and res.measured_reduced_km is None
    assert res.measurement_removals == []


def test_a_failure_that_names_no_test_removes_nothing(tmp_path, monkeypatch):
    """A coverage run that names no test removes nothing: the candidate and
    the merged suite take today's branch with today's reasons, and each
    place records its unnamed entry."""
    backend, target, root = _project(tmp_path)
    _red_only_under_coverage(monkeypatch, backend)
    res = phase_end(
        backend, [SUITE_B, SUITE_A], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=None, log=lambda m: None)
    assert res.candidate2_candidate.status == "incomplete"
    assert res.candidate2_status_reason.startswith("the coverage tool could not measure it")
    assert "could not measure the merged suite" in res.detail
    assert res.measurement_removals == [
        {"place": place, "measurement": "coverage", "round": 1,
         "unnamed": "the measurement named no test"}
        for place in ("phase_end.candidate", "phase_end.merged")]


def test_a_timed_out_coverage_run_removes_nothing(tmp_path, monkeypatch):
    """A coverage run that timed out removes none of the tests it names,
    and its entry records the timeout."""
    backend, target, root = _project(tmp_path)
    named_under_coverage(monkeypatch, backend, ["test_add_a"], timed_out=True)
    res = phase_end(
        backend, [SUITE_B, SUITE_A_SIGN], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=None, log=lambda m: None)
    assert res.candidate2_source == SUITE_A_SIGN
    assert res.candidate2_candidate.status == "incomplete"
    assert res.measurement_removals[0] == {
        "place": "phase_end.candidate", "measurement": "coverage", "round": 1,
        "unnamed": "the measurement timed out"}


def test_a_candidate_that_fails_its_normal_run_after_a_removal_is_failing(
        tmp_path, monkeypatch):
    """A candidate whose normal run fails after the removal is failing, as
    today, and the entry states it."""
    from testuary.languages import TestRunResult

    backend, target, root = _project(tmp_path)
    named_under_coverage(monkeypatch, backend, ["test_add_a"])
    real = consolidate.normal_run.run

    def run(backend_, paths, source_dir, **kwargs):
        text = Path(paths[0]).read_text(encoding="utf-8")
        if "def test_sign_all(" in text and "def test_add_a(" not in text:
            return TestRunResult(
                ok=False, returncode=1, timed_out=False, elapsed_s=0.1, collected=1,
                passed=0, failed=1, errors=0, output="FAILED",
                line_coverage=None, branch_coverage=None)
        return real(backend_, paths, source_dir, **kwargs)

    monkeypatch.setattr(consolidate.normal_run, "run", run)
    res = phase_end(
        backend, [SUITE_B, SUITE_A_SIGN], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=None, log=lambda m: None)
    assert res.candidate2_candidate.status == "failing"
    assert [entry.get("unnamed") for entry in res.measurement_removals[:2]] == [
        None, "the normal run failed after a removal"]
    assert res.measurement_removals[1]["round"] == 2


def test_an_evidence_failure_at_a_phase_end_removes_nothing(tmp_path, monkeypatch):
    """A per-test evidence run that fails naming a test at a phase end
    removes nothing: the suites are scored on their coverage runs."""
    backend, target, root = _project(tmp_path)

    def red_evidence(_self, paths, *_args, failure_fn=None, **_kwargs):
        failure_fn("test_failure", "1 failed", stopping_tests=(named_id(paths[0], "test_add_a"),))
        return None

    monkeypatch.setattr(type(backend), "collect_eviction_data", red_evidence)
    res = phase_end(
        backend, [SUITE_B, SUITE_A_SIGN], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=None, log=lambda m: None)
    assert res.measurement_removals == []
    assert res.candidate2_candidate.status == "passing"
    assert "def test_add_a" in res.candidate2_source


def test_a_phase_end_without_a_measurement_failure_records_no_removal(tmp_path):
    """Without a measurement failure nothing is removed, and the record holds
    no measurement_removals and the archive no changed source."""
    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [SUITE_A, SUITE_SIGN], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=_runner_fixed_catalogue, mutation_method="both",
        log=lambda m: None)
    archive = tmp_path / "archive"
    persist_phase_end(
        matrices_dir=archive, target_file=target, phase=res,
        delivery=PhaseEndDelivery(winner_label="reduced", delivered_km=res.reduced_km,
            delivered_tests=res.selected_tests, delivered_source=res.reduced_source,
            execution_name=res.execution_name))
    record = json.loads((archive / "consolidation_record.json").read_text(encoding="utf-8"))
    assert "measurement_removals" not in record
    assert not list(archive.glob("candidate_*_removed*"))
