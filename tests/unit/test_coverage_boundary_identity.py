"""The coverage boundary's output, pinned artifact by artifact.

Most targets inherit their coverage artifacts, so the delivered suite, kept
tests, per-test token keys, killer attribution, covered lines and branches
and the consolidation record must stay exact. The boundary shares
``consolidate_phase_end`` with the others, and a change meant for one shows
up here as a moved sidecar key or a stripped test, so the pins are on
content, not only on scores.

Everything runs for real except the mutation engine, which is a runner over
a fixed mutant catalogue: a mutant's status depends only on which tests the
measured file holds.
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

from langadapter.python.adapter import PythonAdapter
from testuary import languages
from testuary import config
from testuary import project
from testuary import target_phases as orchestrator
from testuary.consolidation import PhaseEndDelivery, persist_phase_end, scored_from_phase
import pytest
from per_suite_double import killing_tests
from phase_end_calls import phase_end

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


TARGET = """\
def classify(n):
    if n > 0:
        return "pos"
    if n < 0:
        return "neg"
    return "zero"
"""

# The starter phase's delivered suite, where the coverage phase starts.
BASELINE_SUITE = (
    'from calc import classify\n'
    '\n'
    '\n'
    'def test_classify_positive():\n'
    '    assert classify(3) == "pos"\n'
)

# Two new target-touching tests and one that never reaches the target, so the
# cover has exactly one droppable test.
AGENT_SUITE_REDUNDANT = (
    'from calc import classify\n'
    '\n'
    '\n'
    'def test_classify_negative():\n'
    '    assert classify(-3) == "neg"\n'
    '\n'
    '\n'
    'def test_classify_zero():\n'
    '    assert classify(0) == "zero"\n'
    '\n'
    '\n'
    'def test_no_target_contact():\n'
    '    assert True\n'
)

# Every test holds a unique token, so the merged suite is delivered whole.
AGENT_SUITE_IRREDUCIBLE = (
    'from calc import classify\n'
    '\n'
    '\n'
    'def test_classify_negative():\n'
    '    assert classify(-3) == "neg"\n'
)

# Green alone, but the second test asserts on its own module's contents,
# which merging falsifies. Its sibling carries the same evidence.
AGENT_SUITE_MERGE_RED = (
    'from calc import classify\n'
    '\n'
    '\n'
    'def test_classify_negative():\n'
    '    assert classify(-3) == "neg"\n'
    '\n'
    '\n'
    'def test_alone_in_its_module():\n'
    '    assert "test_classify_positive" not in globals()\n'
    '    assert classify(-3) == "neg"\n'
)

# A test that fails whenever the suite runs.
AGENT_SUITE_RED = (
    'from calc import classify\n'
    '\n'
    '\n'
    'def test_classify_negative():\n'
    '    assert classify(-3) == "pos"\n'
)

# Which test kills which mutant. ``m_surv`` is killed by nothing.
_KILLERS = {
    "m_pos": ("test_classify_positive",),
    "m_neg": ("test_classify_negative", "test_alone_in_its_module"),
    "m_zero": ("test_classify_zero",),
}

TOKENS_POSITIVE = [
    "B:calc.py:-1->2", "B:calc.py:2->3", "B:calc.py:3->-1",
    "K:m_pos", "L:calc.py:2", "L:calc.py:3",
]
TOKENS_NEGATIVE = [
    "B:calc.py:-1->2", "B:calc.py:2->4", "B:calc.py:4->5", "B:calc.py:5->-1",
    "K:m_neg", "L:calc.py:2", "L:calc.py:4", "L:calc.py:5",
]
TOKENS_ZERO = [
    "B:calc.py:-1->2", "B:calc.py:2->4", "B:calc.py:4->6", "B:calc.py:6->-1",
    "K:m_zero", "L:calc.py:2", "L:calc.py:4", "L:calc.py:6",
]


def _mutant(mid: str, killers) -> dict:
    if not killers:
        return {"id": mid, "status": "survived", "killer": None,
                "killers": [], "line": None}
    named = [f"test_calc.py::{t}" for t in killers]
    return {"id": mid, "status": "killed", "killer": named[0],
            "killers": named, "killing_tests": killing_tests(named), "line": None}


def _fixed_catalogue_runner(directory):
    """A mutant is killed exactly when the measured file holds a killer."""
    pyfile = next(Path(directory).glob("test_*.py"))
    present = set(re.findall(r"def (test_\w+)",
                             pyfile.read_text(encoding="utf-8")))
    mutants = []
    for mid, killers in _KILLERS.items():
        live = [f"{pyfile.name}::{t}" for t in killers if t in present]
        mutants.append(
            {"id": mid, "status": "killed", "killer": live[0],
             "killers": live, "killing_tests": killing_tests(live), "line": None}
            if live else
            {"id": mid, "status": "survived", "killer": None,
             "killers": [], "line": None})
    mutants.append({"id": "m_surv", "status": "survived", "killer": None,
                    "killers": [], "line": None})
    return {"mutants": mutants, "summary": {}}


@contextmanager
def _fixed_catalogue_session(**_kw):
    yield _fixed_catalogue_runner


def _drive_coverage_boundary(tmp_path, monkeypatch, agent_suite: str, *,
                             before_boundary=None, loop_coverage=(None, None)):
    """Run the starter-phase boundary for real, put ``agent_suite`` on disk as
    the coverage phase's output and run the coverage boundary.
    ``before_boundary(ctx)`` runs before the coverage boundary, and
    ``loop_coverage`` is the line and branch coverage the loop reported."""
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "proj" / "calc.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    # Dataset sources are written byte for byte on every platform.
    src.write_text(TARGET, encoding="utf-8", newline="")
    # The marker resolve_source_dir walks up to.
    (src.parent / "pyproject.toml").write_text("", encoding="utf-8")
    work_tests = tmp_path / "work" / "tests"
    work_tests.mkdir(parents=True, exist_ok=True)
    for name in ("logs", "results"):
        (tmp_path / name).mkdir(exist_ok=True)
    ctx = orchestrator.TargetPhasesContext(
        log_print=lambda *a, **k: None,
        project_name="p", workspace="rank1_calc",
        top_file={"file_path": str(src), "key_functions": ["classify"]},
        canonical_path=src, work_tests=work_tests,
        results_dir=tmp_path / "results", target_logs_dir=tmp_path / "logs",
        reuse=orchestrator.ReuseFlags(),
    )
    monkeypatch.setattr(orchestrator.measurement, "union_mutation_runner",
                        _fixed_catalogue_session)

    backend = languages.get_backend("python")
    source_dir = project.resolve_source_dir(str(src))
    mut_cfg = config.get_agent_config(
        "test_mutation", parent="test_coverage"
    )
    identity = orchestrator.mutation_agent.mutation_measurement_identity(
        target_file_path=str(src),
        language="python",
        execution_name="test_calc.py",
        config=dict(mut_cfg),
        attribution=orchestrator.measurement.union_attribution(mut_cfg),
    )
    # The starter-phase end, whose evidence the coverage end may reuse.
    initial = phase_end(
        backend, [BASELINE_SUITE], target_file_path=str(src),
        source_dir=source_dir, timeout=120,
        mutation_runner=_fixed_catalogue_runner, mutation_method="both",
        stage_as_delivered=True, measurement_identity=identity,
        log=lambda m: None)
    persist_phase_end(
        matrices_dir=ctx.results_dir / ctx.workspace / "r0" / "archive" / "starter",
        target_file=str(src), phase=initial,
        delivery=PhaseEndDelivery(winner_label="single_suite",
            delivered_km=initial.reduced_km, delivered_tests=initial.selected_tests,
            delivered_source=initial.reduced_source,
            delivered_candidate=scored_from_phase(initial), execution_name="test_calc.py"))

    suite_path = work_tests / "test_calc.py"
    suite_path.write_text(agent_suite, encoding="utf-8")
    result = orchestrator.TestPipelineResult()
    result.test_file_name = suite_path.name
    result.test_result = SimpleNamespace(
        line_coverage=loop_coverage[0], branch_coverage=loop_coverage[1],
        test_count_standard=None)
    result.starter_result = SimpleNamespace(
        baseline_line_coverage=initial.line,
        baseline_branch_coverage=initial.branch,
        baseline_suite_seconds=0.1,
        selected_tests=initial.selected_tests)

    if before_boundary is not None:
        before_boundary(ctx)
    orchestrator._consolidate_coverage(ctx, result, BASELINE_SUITE)
    return SimpleNamespace(
        ctx=ctx, result=result, target=str(src), suite_path=suite_path,
        archive=ctx.results_dir / ctx.workspace / "r0" / "archive" / "coverage",
    )


def _record(run) -> dict:
    """The consolidation record, machine-specific fields checked and removed."""
    record = json.loads(
        (run.archive / "consolidation_record.json").read_text(encoding="utf-8"))
    assert record.pop("target_file") == run.target
    assert "written_at" not in record
    target_revision = hashlib.sha256(TARGET.encode("utf-8")).hexdigest()
    assert record.pop("selected_source_sha256") == hashlib.sha256(
        run.suite_path.read_bytes()
    ).hexdigest()
    assert record.pop("execution_name") == "test_calc.py"
    assert record.pop("target_revision") == target_revision
    assert float(record.pop("t_mono_s")) >= 0.0
    assert record.pop("clock_domain_id")
    assert record.pop("structural_parser_identity").startswith("tree-sitter ")
    assert record.pop("target_validator_identity").startswith("CPython ")
    assert record.pop("reduction_solver") == "exact"
    solver = record.pop("reduction_solver_metadata")
    assert solver["solver"] == "exact"
    assert solver["cbc_search"] == "serial"
    assert "threads" not in solver
    assert solver["random_seed"] == 1
    assert solver["time_limit_s"] == 120.0
    assert solver["minimum_test_runtime_ms"] == 0.001
    # The default breaks ties by name and proves the choice unique: four
    # optimal tiers, then a solve that finds no other equal cover.
    assert solver["tie_break"] == "name_order"
    assert [tier["name"] for tier in solver["tiers"]] == [
        "runtime", "test_count", "rank_sum", "rank_square_sum", "uniqueness",
    ]
    assert [tier["problem_status"] for tier in solver["tiers"]] == [
        "Optimal", "Optimal", "Optimal", "Optimal", "Infeasible",
    ]
    assert [tier["solution_status"] for tier in solver["tiers"][:4]] == [
        "Optimal Solution Found"] * 4
    assert solver["uniqueness"] == "proven"
    assert solver["equally_good_selections"] == 1
    assert solver["runtime_objective_us"] == sum(
        solver["effective_cost_us"][test_id]
        for test_id in solver["selected_ids"]
    )
    assert solver["test_count_objective"] == len(solver["selected_ids"])
    delivered_hash = hashlib.sha256(run.suite_path.read_bytes()).hexdigest()
    merged_hash = record.pop("merged_source_sha256")
    reduced_hash = record.pop("reduced_source_sha256")
    assert len(merged_hash) == 64
    assert reduced_hash == delivered_hash
    identity = record.pop("measurement_identity")
    assert identity.pop("config_sha256")
    assert identity == {
        "target_source_sha256": target_revision,
        "language": "python",
        "execution_name": "test_calc.py",
        "catalogue": "full",
        "attribution": "all",
    }
    return record


def _json(run, name: str) -> dict:
    """One archived sidecar, its machine-specific fields checked and removed."""
    payload = json.loads((run.archive / name).read_text(encoding="utf-8"))
    if "target_file" in payload:
        assert payload.pop("target_file") == run.target
    if "source_sha256" in payload:
        assert len(payload.pop("source_sha256")) == 64
    if "execution_name" in payload:
        assert payload.pop("execution_name") == "test_calc.py"
    if "structural_parser_identity" in payload:
        assert payload.pop("structural_parser_identity").startswith(
            "tree-sitter "
        )
    if "target_validator_identity" in payload:
        assert payload.pop("target_validator_identity").startswith("CPython ")
    return payload


def test_coverage_boundary_delivers_the_reduced_suite_and_its_sidecars(
    tmp_path, monkeypatch,
):
    """The merged suite beats both inputs, the cover drops the test holding no
    token and the reduced suite is delivered. Every artifact is pinned, since token
    keys and killer attribution can move while scores stay."""
    run = _drive_coverage_boundary(
        tmp_path, monkeypatch, AGENT_SUITE_REDUNDANT)

    assert run.suite_path.read_text(encoding="utf-8") == (
        'from calc import classify\n'
        '\n'
        '\n'
        'def test_classify_positive():\n'
        '    assert classify(3) == "pos"\n'
        '\n'
        '\n'
        'def test_classify_negative():\n'
        '    assert classify(-3) == "neg"\n'
        '\n'
        '\n'
        'def test_classify_zero():\n'
        '    assert classify(0) == "zero"\n'
        '\n'
        '\n'
    )
    # The loop's suite before merging is kept verbatim.
    assert (run.archive / "candidate_agent.py").read_text(
        encoding="utf-8") == AGENT_SUITE_REDUNDANT

    record = _record(run)
    selection = record.pop("selection_audit")
    # The normal run decides at the phase end, and no check as delivered is
    # recorded.
    assert "portability" not in record
    # Without the earlier generation scope and request inputs, the ancestry
    # is partial and names what is missing.
    ancestry = record.pop("generation_ancestry")
    assert (ancestry["operation"], ancestry["phase"], ancestry["status"]) == (
        "reduced", "coverage", "partial")
    assert ancestry["selected_source_sha256"] == hashlib.sha256(
        run.suite_path.read_bytes()).hexdigest()
    assert [(parent["operation"], parent["status"]) for parent in ancestry["parents"]] == [
        ("unavailable", "unavailable"), ("edited", "partial")]
    assert record == {
        "schema_version": 2,
        # The cover's suite differs from the merged suite and is delivered.
        "delivered_from": "reduced",
        "winner_source_label": None,
        "merged_tests": 4,
        "delivered_tests": 3,
        "reduction_valid": True,
            "repaired_tests": [],
            "evidence_repairs": [],
        "unreduced_fallback": False,
        "fallback_reason": None,
        "fallback_candidates": [],
        "fallback_merged_stats": None,
        "detail": "",
        "derive_measure_delta": 0.0,
        "derive_measure_differences": [],
        "consolidation_mutation_method": "both",
        "artifacts": [
            "candidate2_coverage_matrix.json",
            "candidate2_kill_matrix.json",
            "derived_reduced_kill_matrix.json",
            "measured_reduced_kill_matrix.json",
            "merged_coverage_matrix.json",
            "merged_kill_matrix.json",
            # The delivered suite's evidence, named independently of the winner.
            "reduced_coverage_matrix.json",
            "reduced_kill_matrix.json",
            "selected_coverage_matrix.json",
            "selected_kill_matrix.json",
            "validity_delivered.json",
            "validity_derivations.json",
            "validity_merged.json",
        ],
        "collision_resolution": {
            "renamed": 0, "deduped": 0, "autouse_localized": 0,
            "dropped": [], "ejected_sources": 0,
        },
        "input_normalizations": [],
        "mutation_measured": True,
        "shipped_has_matrix": True,
        "validity_policy_version": PythonAdapter.validity_policy_version,
        "validity_artifacts": [
            "validity_delivered.json",
            "validity_derivations.json",
            "validity_merged.json",
        ],
    }
    assert selection["selector"] == "mutation_then_branch_then_line"
    assert selection["active_axes"] == ["mutation", "branch", "line"]
    assert selection["delivered_from"] == "reduced"
    assert selection["selection_basis"] == "reduced_preference"
    assert selection["top_score_labels"] == ["reduced", "merged"]
    assert [candidate["label"] for candidate in selection["candidates"]] == [
        "reduced", "merged", "agent", "baseline"
    ]
    assert all(candidate["fully_measured"]
               for candidate in selection["candidates"])
    assert selection["delivered_from_missing_from_merged_input"] == {
        "kills": 0, "lines": 0, "branches": 0,
    }

    reduced_coverage = _json(run, "reduced_coverage_matrix.json")
    reduced_runtimes = reduced_coverage.pop("per_test_runtime_ms")
    assert set(reduced_runtimes) == {
        "test_calc.test_classify_negative",
        "test_calc.test_classify_positive",
        "test_calc.test_classify_zero",
    }
    assert all(value >= 0.0 for value in reduced_runtimes.values())
    effective_costs = reduced_coverage.pop("effective_cost_us")
    assert set(effective_costs) == set(reduced_runtimes)
    assert all(value >= 1 for value in effective_costs.values())
    assert reduced_coverage == {
        "schema_version": 1,
        "line": 100.0,
        "branch": 100.0,
        "target_total_statements": 6,
        "covered_statement_lines": [1, 2, 3, 4, 5, 6],
        "target_total_branches": 4,
        "covered_branch_arcs": ["2->3", "2->4", "4->5", "4->6"],
        "kept_tests": [
            "test_calc.test_classify_negative",
            "test_calc.test_classify_positive",
            "test_calc.test_classify_zero",
        ],
        "derive_kept_tests": [
            "test_calc.test_classify_negative",
            "test_calc.test_classify_positive",
            "test_calc.test_classify_zero",
        ],
        "per_test_tokens": {
            "test_calc.test_classify_negative": TOKENS_NEGATIVE,
            "test_calc.test_classify_positive": TOKENS_POSITIVE,
            "test_calc.test_classify_zero": TOKENS_ZERO,
        },
    }
    union_coverage = _json(run, "merged_coverage_matrix.json")
    union_runtimes = union_coverage.pop("per_test_runtime_ms")
    assert set(union_runtimes) == {
        "test_calc.test_no_target_contact",
        "test_calc.test_classify_negative",
        "test_calc.test_classify_positive",
        "test_calc.test_classify_zero",
    }
    assert all(value >= 0.0 for value in union_runtimes.values())
    union_effective_costs = union_coverage.pop("effective_cost_us")
    assert set(union_effective_costs) == set(union_runtimes)
    assert all(value >= 1 for value in union_effective_costs.values())
    assert union_coverage == {
        "schema_version": 1,
        "line": 100.0,
        "branch": 100.0,
        "target_total_statements": 6,
        "covered_statement_lines": [1, 2, 3, 4, 5, 6],
        "target_total_branches": 4,
        "covered_branch_arcs": ["2->3", "2->4", "4->5", "4->6"],
        "per_test_tokens": {
            "test_calc.test_no_target_contact": [],
            "test_calc.test_classify_negative": TOKENS_NEGATIVE,
            "test_calc.test_classify_positive": TOKENS_POSITIVE,
            "test_calc.test_classify_zero": TOKENS_ZERO,
        },
    }

    every_killed = {"mutants": [
        _mutant("m_pos", ("test_classify_positive",)),
        _mutant("m_neg", ("test_classify_negative",)),
        _mutant("m_zero", ("test_classify_zero",)),
        _mutant("m_surv", ()),
    ], "summary": {}}
    assert _json(run, "reduced_kill_matrix.json") == every_killed
    assert _json(run, "merged_kill_matrix.json") == every_killed

    assert run.result.consolidation_stats["coverage"] == {
        "merged_tests": 4, "delivered_tests": 3, "delivered_from": "reduced",
    }
    assert run.result.test_result.line_coverage == 100.0
    assert run.result.test_result.branch_coverage == 100.0

    # The mutation phase inherits this matrix, bound to the delivered bytes.
    handoff = json.loads(
        (run.ctx.work_tests / ".mutation_baseline_matrix.json")
        .read_text(encoding="utf-8"))
    assert handoff["matrix"] == every_killed
    assert handoff["suite_sha256"] == hashlib.sha256(
        run.suite_path.read_text(encoding="utf-8").encode("utf-8")).hexdigest()


def test_every_boundary_stages_candidates_under_the_delivered_name():
    """Every ``consolidate_phase_end`` call in the package stages candidates
    under the delivered name (its PhaseEndContext's stage_as_delivered), so
    keys agree and a test depending on its own module is judged as it is delivered.
    """
    phases_dir = Path(orchestrator.__file__).parent
    silent = []
    scratch = []
    found = 0
    for module in sorted(phases_dir.glob("*.py")):
        tree = ast.parse(module.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not isinstance(node, ast.Call):
                continue
            name = getattr(node.func, "id", getattr(node.func, "attr", ""))
            if name != "consolidate_phase_end":
                continue
            found += 1
            where = f"{module.name}:{node.lineno}"
            context = node.args[0] if node.args else None
            stated = [k for k in getattr(context, "keywords", [])
                      if k.arg == "stage_as_delivered"]
            if not stated:
                silent.append(where)
                continue
            value = stated[0].value
            if not (isinstance(value, ast.Constant) and value.value is True):
                scratch.append(where)
    assert not silent, (
        f"consolidate_phase_end call(s) without stage_as_delivered: {silent}")
    assert not scratch, (
        f"boundary/ies still staging under the scratch name: {scratch}")
    # The starter phase and the shared phase end of the coverage and
    # mutation phases (run_phase_end).
    assert found == 2, f"expected 2 boundary calls, found {found}"


def test_coverage_boundary_ships_the_whole_merged_suite_when_no_test_is_redundant(
    tmp_path, monkeypatch,
):
    """With every token unique, the cover keeps all tests, the delivered suite
    is the merged suite and the record reports a full cover, not a fallback."""
    run = _drive_coverage_boundary(
        tmp_path, monkeypatch, AGENT_SUITE_IRREDUCIBLE)

    assert run.suite_path.read_text(encoding="utf-8") == (
        'from calc import classify\n'
        '\n'
        '\n'
        'def test_classify_positive():\n'
        '    assert classify(3) == "pos"\n'
        '\n'
        '\n'
        'def test_classify_negative():\n'
        '    assert classify(-3) == "neg"\n'
    )
    record = _record(run)
    assert record["delivered_from"] == "reduced"
    assert record["reduction_valid"] is True
    assert record["selection_audit"]["selection_basis"] == (
        "reduced_preference"
    )
    assert record["merged_tests"] == record["delivered_tests"] == 2
    assert record["unreduced_fallback"] is False
    assert record["fallback_reason"] is None
    # A no-op cover reuses the merged measurement but persists both
    # cross-check roles.
    assert record["derive_measure_delta"] == 0.0
    assert record["derive_measure_differences"] == []
    assert set(record["artifacts"]) >= {
        "derived_reduced_kill_matrix.json",
        "measured_reduced_kill_matrix.json",
    }

    reduced = _json(run, "reduced_coverage_matrix.json")
    union = _json(run, "merged_coverage_matrix.json")
    assert reduced["kept_tests"] == [
        "test_calc.test_classify_negative",
        "test_calc.test_classify_positive",
    ]
    assert reduced["per_test_tokens"] == union["per_test_tokens"] == {
        "test_calc.test_classify_negative": TOKENS_NEGATIVE,
        "test_calc.test_classify_positive": TOKENS_POSITIVE,
    }
    assert reduced["covered_statement_lines"] == [1, 2, 3, 4, 5]
    assert reduced["covered_branch_arcs"] == ["2->3", "2->4", "4->5"]
    assert reduced["line"] == union["line"]
    assert reduced["branch"] == union["branch"]

    killed = {"mutants": [
        _mutant("m_pos", ("test_classify_positive",)),
        _mutant("m_neg", ("test_classify_negative",)),
        _mutant("m_zero", ()),
        _mutant("m_surv", ()),
    ], "summary": {}}
    assert _json(run, "reduced_kill_matrix.json") == killed
    assert _json(run, "merged_kill_matrix.json") == killed
    assert _json(run, "measured_reduced_kill_matrix.json") == killed
    derived = _json(run, "derived_reduced_kill_matrix.json")
    # The derived matrix keeps every verdict and names its kind, each row its
    # measured status and rule.
    assert [{k: v for k, v in row.items() if k != "derived"}
            for row in derived["mutants"]] == killed["mutants"]
    assert [row["derived"] for row in derived["mutants"]] == [
        {"measured_status": "killed", "rule": "kept_killer"},
        {"measured_status": "killed", "rule": "kept_killer"},
        {"measured_status": "survived", "rule": "not_killed"},
        {"measured_status": "survived", "rule": "not_killed"},
    ]
    assert derived["derived"]["kind"] == "kill_subset"
    assert derived["summary"] == {
        "detected": 2,
        "survived": 2,
        "timeout": 0,
        "non_viable": 0,
        "untested": 0,
        "unresolved": 0,
        "total": 4,
    }




def test_a_coverage_only_phase_end_measures_the_agent_suite(tmp_path, monkeypatch):
    """Without a mutation runner the coverage phase end measures the agent's
    suite like every candidate: one that fails is not delivered, whatever
    coverage the loop reported."""
    def coverage_only(ctx):
        monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: False)

    run = _drive_coverage_boundary(
        tmp_path, monkeypatch, AGENT_SUITE_RED, before_boundary=coverage_only,
        loop_coverage=(100.0, 100.0))
    record = json.loads(
        (run.archive / "consolidation_record.json").read_text(encoding="utf-8"))
    assert record["delivered_from"] == "baseline"
    assert 'classify(-3) == "pos"' not in run.suite_path.read_text(encoding="utf-8")


def test_coverage_boundary_leaves_a_red_merge_out_and_keeps_its_tests(
    tmp_path, monkeypatch,
):
    """A test that passes in the agent's suite but fails once merged leaves
    the merged suite out of the choice. Nothing is removed, and the agent's
    suite, which passes, is delivered as it is."""
    run = _drive_coverage_boundary(
        tmp_path, monkeypatch, AGENT_SUITE_MERGE_RED)

    assert run.suite_path.read_text(encoding="utf-8") == AGENT_SUITE_MERGE_RED
    # No reduction ran, so the record has no solver metadata to check.
    record = json.loads(
        (run.archive / "consolidation_record.json").read_text(encoding="utf-8"))
    assert record["delivered_from"] == "agent"


@pytest.mark.parametrize("stops", [
    [],
    [{"suite": "candidate2", "reason": "the unmutated suite failed on the Mull build",
      "stopping_tests": ["test_calc.py::test_a", "test_calc.py::test_b"]}],
], ids=["no stop", "one stop"])
def test_the_consolidation_record_names_the_runs_the_origin_check_stopped(tmp_path, stops):
    """The runner's stops reach the consolidation record with every stopping
    test's name. A phase end without one writes no such key."""
    src = tmp_path / "proj" / "calc.py"
    src.parent.mkdir(parents=True)
    src.write_text(TARGET, encoding="utf-8", newline="")
    (src.parent / "pyproject.toml").write_text("", encoding="utf-8")

    def runner(directory):
        return _fixed_catalogue_runner(directory)

    runner.stops = stops
    phase = phase_end(
        languages.get_backend("python"), [BASELINE_SUITE], target_file_path=str(src),
        source_dir=project.resolve_source_dir(str(src)), timeout=120,
        mutation_runner=runner, mutation_method="both", stage_as_delivered=True,
        log=lambda m: None)
    archive = tmp_path / "archive"
    persist_phase_end(
        matrices_dir=archive, target_file=str(src), phase=phase,
        delivery=PhaseEndDelivery(winner_label="single_suite",
            delivered_km=phase.reduced_km, delivered_tests=phase.selected_tests,
            delivered_source=phase.reduced_source,
            delivered_candidate=scored_from_phase(phase), execution_name="test_calc.py"))
    record = json.loads((archive / "consolidation_record.json").read_text(encoding="utf-8"))
    if stops:
        assert record["mutation_origin_stops"] == stops
    else:
        assert "mutation_origin_stops" not in record


# The agent's suite of AGENT_SUITE_MERGE_RED with one more test, which the
# coverage tool's run fails.
AGENT_SUITE_COVERAGE_RED = AGENT_SUITE_MERGE_RED + (
    '\n'
    '\n'
    'def test_classify_twice():\n'
    '    assert classify(-3) == "neg"\n'
)


def test_an_agent_suite_that_lost_a_test_reports_its_own_coverage(tmp_path, monkeypatch):
    """Where the agent's suite is chosen and the phase end removed a test its
    coverage run failed, the coverage phase delivers the suite without that
    test and reports that suite's coverage, test count and test lines, not
    the loop's."""
    from phase_end_calls import named_under_coverage

    named_under_coverage(monkeypatch, languages.get_backend("python"), ["test_classify_twice"])
    run = _drive_coverage_boundary(
        tmp_path, monkeypatch, AGENT_SUITE_COVERAGE_RED, loop_coverage=(100.0, 100.0))

    delivered = run.suite_path.read_text(encoding="utf-8")
    assert "test_classify_twice" not in delivered
    record = json.loads(
        (run.archive / "consolidation_record.json").read_text(encoding="utf-8"))
    assert record["delivered_from"] == "agent"
    assert (run.result.test_result.line_coverage,
            run.result.test_result.branch_coverage) == (pytest.approx(200 / 3), 50.0)
    assert run.result.test_result.test_count_standard == 2
    assert run.result.test_result.test_loc_standard == orchestrator.measurement.record(
        orchestrator.measurement.observe_file(
            run.suite_path, backend=languages.get_backend("python")))["code_lines"]


# The agent's suite with a comment block standing apart from code, which the
# phase end's finalized form of the suite drops.
AGENT_SUITE_DETACHED_COMMENT = (
    'from calc import classify\n'
    '\n'
    '\n'
    '# Negative inputs.\n'
    '\n'
    'def test_classify_negative():\n'
    '    assert classify(-3) == "neg"\n'
    '\n'
    '\n'
    'def test_alone_in_its_module():\n'
    '    assert "test_classify_positive" not in globals()\n'
    '    assert classify(-3) == "neg"\n'
)


def test_an_agent_suite_that_stands_keeps_the_loop_coverage(tmp_path, monkeypatch):
    """Where the agent's suite is chosen and no test of it is removed, the
    coverage phase reports the loop's coverage, also when the phase end's
    finalized form of the suite differs from the loop's file."""
    run = _drive_coverage_boundary(
        tmp_path, monkeypatch, AGENT_SUITE_DETACHED_COMMENT, loop_coverage=(100.0, 100.0))

    record = json.loads(
        (run.archive / "consolidation_record.json").read_text(encoding="utf-8"))
    assert record["delivered_from"] == "agent"
    assert (run.result.test_result.line_coverage,
            run.result.test_result.branch_coverage) == (100.0, 100.0)

def test_a_run_folder_with_removals_in_a_candidate_and_the_merged_suite_is_reusable(
        tmp_path, monkeypatch):
    """A coverage phase end whose loop suite and merged suite each lost the
    test their coverage run failed leaves a run folder that the reuse check
    of --reuse coverage accepts: the delivered suite's validity chain runs
    from the merged suite as built through the removal and the set cover."""
    from phase_end_calls import named_under_coverage

    failing = ["test_classify_twice", "test_classify_positive"]
    run = _drive_coverage_boundary(
        tmp_path, monkeypatch, AGENT_SUITE_REDUNDANT + (
            '\n\n'
            'def test_classify_twice():\n'
            '    assert classify(-3) == "neg"\n'),
        before_boundary=lambda ctx: named_under_coverage(
            monkeypatch, languages.get_backend("python"), failing))

    record = json.loads(
        (run.archive / "consolidation_record.json").read_text(encoding="utf-8"))
    assert [(entry["place"], entry["removed"][0]["test"].rsplit(".", 1)[-1])
            for entry in record["measurement_removals"]] == [
        ("phase_end.candidate", "test_classify_twice"),
        ("phase_end.merged", "test_classify_positive")]
    assert record["delivered_from"] == "reduced"
    chain = json.loads((run.archive / "validity_derivations.json").read_text(
        encoding="utf-8"))["records"]
    assert [item["operation"] for item in chain] == [
        "measurement_remove", "set_cover_remove_restore"]
    assert orchestrator._verify_restored_phase_validity(
        run.ctx, "coverage", run.suite_path) == (True, "")
    # The loop's changed suite keeps the label the selection gives it.
    assert [path.name for path in run.archive.glob("candidate_*_removed*")] == [
        "candidate_agent_removed.py"]
