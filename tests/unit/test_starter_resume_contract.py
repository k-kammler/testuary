"""Starter-phase resume rejects incomplete or mismatched handoffs."""
from __future__ import annotations

import json

import pytest

from testuary import target_phases as orchestrator

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture
def handoff(tmp_path, persist_starter_resume):
    target = tmp_path / "mod.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    ctx = orchestrator.TargetPhasesContext(
        log_print=lambda *a, **k: None,
        canonical_path=target, results_dir=tmp_path / "results",
        workspace="rank1_mod", target_logs_dir=tmp_path / "logs",
        work_tests=tmp_path / "work/tests",
        project_name="example", top_file={"file_path": str(target)},
        reuse=orchestrator.ReuseFlags(starter=True),
    )
    snapshot = persist_starter_resume(
        ctx, "from mod import f\n\ndef test_f():\n    assert f() == 1\n",
    )
    return ctx, snapshot


def test_complete_handoff_restores_accounting(handoff):
    ctx, snapshot = handoff
    result = orchestrator._restore_starter_baseline(ctx)
    assert result.baseline_file == str(snapshot)
    assert result.n_generated == result.n_green == result.requests_used == 1
    assert result.pool_tests == result.selected_tests == 1
    assert (result.input_tokens, result.cache_hit_tokens, result.output_tokens) == (100, 20, 50)
    assert result.suite_generation_attempts == {1: 1}
    assert result.generation_seconds == 2.0
    assert result.baseline_suite_seconds == 0.25
    assert result.winner == "single_suite"


@pytest.mark.parametrize("name", [
    "result.py", "validity_delivered.json", "consolidation_record.json",
    "suites/_generation.json",
])
def test_missing_handoff_artifact_is_rejected(handoff, name):
    ctx, snapshot = handoff
    (snapshot.parent / name).unlink()
    with pytest.raises(RuntimeError):
        orchestrator._restore_starter_baseline(ctx)


@pytest.mark.parametrize("artifact", ["suite", "target"])
def test_stale_handoff_identity_is_rejected(handoff, artifact):
    ctx, snapshot = handoff
    path = snapshot if artifact == "suite" else ctx.canonical_path
    path.write_text(path.read_text(encoding="utf-8") + "\n# changed source\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="validity"):
        orchestrator._restore_starter_baseline(ctx)


@pytest.mark.parametrize(("name", "key", "value"), [
    ("suites/_generation.json", "requests_used", None),
    ("suites/_generation.json", "input_tokens", -1),
    ("suites/_generation.json", "n_generated", True),
    ("suites/_generation.json", "output_tokens", "unknown"),
    ("consolidation_record.json", "merged_tests", -1),
    ("consolidation_record.json", "delivered_tests", -1),
    ("consolidation_record.json", "delivered_from", ""),
])
def test_invalid_handoff_accounting_is_rejected(handoff, name, key, value):
    ctx, snapshot = handoff
    path = snapshot.parent / name
    data = json.loads(path.read_text(encoding="utf-8"))
    data[key] = value
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(RuntimeError):
        orchestrator._restore_starter_baseline(ctx)


@pytest.mark.parametrize(("key", "value"), [("suites_passing", None), ("selected_tests", 2)])
def test_an_invalid_checkpoint_is_rejected(handoff, key, value):
    from testuary import phase_state

    ctx, _snapshot = handoff
    target_dir = ctx.results_dir / ctx.workspace
    state = phase_state.load(target_dir)
    state["starter"]["0"]["metrics"][key] = value
    phase_state.save(target_dir, state)
    with pytest.raises(RuntimeError):
        orchestrator._restore_starter_baseline(ctx)



def test_a_handoff_whose_phase_end_counted_no_merged_suite_restores(handoff):
    """A starter phase end that counted no merged suite records null
    merged_tests, and its checkpoint restores with no pool count."""
    ctx, snapshot = handoff
    path = snapshot.parent / "consolidation_record.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["merged_tests"] = None
    path.write_text(json.dumps(data), encoding="utf-8")
    assert orchestrator._restore_starter_baseline(ctx).pool_tests is None


def test_complete_handoff_restores_request_accounting(handoff):
    ctx, _snapshot = handoff
    result = orchestrator._restore_starter_baseline(ctx)
    assert result.logical_requests == 1
    assert result.unknown_usage_requests == 0


def test_a_manifest_without_request_accounting_leaves_it_unknown(handoff):
    """A manifest without the logical and unknown-usage request counts
    restores them as unknown, not zero, and the rest of the handoff."""
    ctx, snapshot = handoff
    manifest = snapshot.parent / "suites" / "_generation.json"
    data = json.loads(manifest.read_text(encoding="utf-8"))
    del data["logical_requests"]
    del data["unknown_usage_requests"]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    result = orchestrator._restore_starter_baseline(ctx)
    assert result.requests_used == 1
    assert result.logical_requests is None
    assert result.unknown_usage_requests is None
