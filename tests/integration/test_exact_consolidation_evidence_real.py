"""Executable exact-reduction and evidence-repair integration.

Coverage collection, per-test contexts, test removal, target validation,
repaired suite execution, aggregate coverage, exact CBC reduction,
persistence, and selection all use production code. A deterministic mutant
catalogue and one
controlled post-reduction discrepancy force the otherwise defensive repair
path for statement, branch, and killed-mutant identities in the same run.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import replace
from pathlib import Path

import pytest

from testuary import config, languages
from testuary import consolidation as cmod
from testuary.consolidation import (
    PhaseEndContext,
    PhaseEndOptions,
    PhaseEndDelivery,
    consolidate_phase_end,
    persist_phase_end,
    phase_end_candidates,
    pick_phase_result,
    record_selection_audit,
    _PHASE_END_AXES,
)


pytestmark = [pytest.mark.slow, pytest.mark.usefixtures("default_run")]


TARGET = """\
def classify(value):
    if value > 0:
        return "positive"
    return "nonpositive"
"""


def _suite(name: str) -> str:
    return (
        "from classify import classify\n\n"
        f"def {name}():\n"
        "    assert classify(1) == \"positive\"\n"
        "    assert classify(0) == \"nonpositive\"\n"
    )


def _mutation_matrix(directory: Path) -> dict:
    path = next(directory.glob("test_*.py"))
    present = sorted(set(re.findall(
        r"^def (test_\w+)", path.read_text(encoding="utf-8"), re.MULTILINE,
    )))
    killed = {"test_classify_a", "test_classify_b"} <= set(present)
    killers = [f"{path.name}::{name}" for name in present]
    return {
        "mutants": [{
            "id": "m_interaction",
            "operator": "probe/Interaction",
            "line": 2,
            "status": "killed" if killed else "survived",
            "killer": killers[0] if killed else None,
            "killers": killers if killed else [],
            "killing_tests": [
                {"file": path.name, "qualname": name, "params": None,
                 "text": f"{path.name}::{name}"}
                for name in present
            ] if killed else [],
        }],
        "summary": {},
    }


def test_exact_reduction_repairs_and_persists_all_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    project = tmp_path / "project"
    project.mkdir()
    target = project / "classify.py"
    target.write_text(TARGET, encoding="utf-8")
    backend = languages.get_backend("python")

    original_refresh = cmod._refresh_reduced_coverage
    refresh_calls = 0

    def report_one_controlled_gap(*args, **kwargs):
        nonlocal refresh_calls
        ok = original_refresh(*args, **kwargs)
        refresh_calls += 1
        if ok and refresh_calls == 1:
            result = kwargs["result"]
            assert result.suite_covered_lines
            assert result.suite_covered_branches
            result.suite_covered_lines = result.suite_covered_lines[1:]
            result.suite_covered_branches = result.suite_covered_branches[1:]
        return ok

    monkeypatch.setattr(
        cmod, "_refresh_reduced_coverage", report_one_controlled_gap,
    )
    phase = consolidate_phase_end(
        PhaseEndContext(
            backend=backend, target_file_path=str(target), source_dir=str(project),
            timeout=120, stage_as_delivered=True, log=lambda _message: None),
        [_suite("test_classify_a"), _suite("test_classify_b")],
        _mutation_matrix,
        PhaseEndOptions(
            mutation_method="both",
            reduction=replace(config.solver_settings(config.get_agent_config("test_coverage")),
                              solver="exact", solver_time_limit_seconds=120)),
    )

    assert refresh_calls == 1
    assert phase.reduction_valid is True
    assert phase.reduced_source is not None
    assert phase.merged_source is not None
    assert set(phase.kept_tests or ()) == set(phase.per_test_tokens or {})
    assert phase.repaired_tests
    repair = phase.evidence_repairs[0]
    assert repair["outcome"] == "accepted"
    assert repair["missing_lines_before"]
    assert repair["missing_branches_before"]
    assert repair["missing_kills_before"]
    assert phase.suite_covered_lines == phase.union_covered_lines
    assert phase.suite_covered_branches == phase.union_covered_branches

    metadata = phase.solver_metadata
    assert metadata is not None
    # The package default breaks ties by name and proves the choice unique.
    assert metadata["tie_break"] == "name_order"
    assert [tier["name"] for tier in metadata["tiers"]] == [
        "runtime", "test_count", "rank_sum", "rank_square_sum", "uniqueness",
    ]
    assert [tier["problem_status"] for tier in metadata["tiers"]] == [
        "Optimal", "Optimal", "Optimal", "Optimal", "Infeasible",
    ]
    assert [tier["solution_status"] for tier in metadata["tiers"][:4]] == [
        "Optimal Solution Found"] * 4
    assert metadata["uniqueness"] == "proven"
    exact_ids = set(metadata["selected_ids"])
    assert metadata["runtime_objective_us"] == sum(
        metadata["effective_cost_us"][test_id] for test_id in exact_ids
    )
    assert metadata["test_count_objective"] == len(exact_ids)

    candidates = phase_end_candidates(phase)
    winner = pick_phase_result(candidates, log=lambda _message: None, required_axes=_PHASE_END_AXES)
    assert winner is not None and winner.label == "reduced"
    record_selection_audit(phase, candidates, winner, required_axes=_PHASE_END_AXES)

    archive = tmp_path / "archive" / "starter"
    persist_phase_end(
        matrices_dir=archive, target_file=str(target), phase=phase,
        delivery=PhaseEndDelivery(winner_label=winner.label,
            winner_source_label=winner.label, delivered_km=winner.kill_matrix,
            delivered_tests=winner.test_count, delivered_source=winner.source,
            delivered_candidate=winner, execution_name=phase.execution_name))

    record = json.loads(
        (archive / "consolidation_record.json").read_text(encoding="utf-8")
    )
    assert record["delivered_from"] == "reduced"
    assert record["selection_audit"]["selection_basis"] == (
        "reduced_preference"
    )
    assert record["evidence_repairs"][0]["outcome"] == "accepted"
    assert record["reduction_solver_metadata"]["selected_ids"] == sorted(
        exact_ids
    )
    assert record["selected_source_sha256"] == hashlib.sha256(
        winner.source.encode("utf-8")
    ).hexdigest()
    for name in (
        "merged_coverage_matrix.json",
        "reduced_coverage_matrix.json",
        "merged_kill_matrix.json",
        "reduced_kill_matrix.json",
        "derived_reduced_kill_matrix.json",
        "measured_reduced_kill_matrix.json",
        "selected_coverage_matrix.json",
        "selected_kill_matrix.json",
    ):
        assert (archive / name).is_file(), name

    merged = json.loads(
        (archive / "merged_coverage_matrix.json").read_text(encoding="utf-8")
    )
    reduced = json.loads(
        (archive / "reduced_coverage_matrix.json").read_text(encoding="utf-8")
    )
    merged_tokens = set().union(*map(set, merged["per_test_tokens"].values()))
    reduced_tokens = set().union(*map(set, reduced["per_test_tokens"].values()))
    assert reduced_tokens == merged_tokens
