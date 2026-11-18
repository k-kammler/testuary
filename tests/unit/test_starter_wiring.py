"""Tests for the starter suites' wiring in the pipeline: sequencing, resume,
cancellation, checkpoints, handoff to the coverage phase, CSV fields and CLI
settings."""
from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import pytest

from per_suite_double import killing_tests, per_suite_double

from offered_tools import offered_tools
import testuary
import testuary.__main__ as main_mod
from testuary import phase_state
from testuary import results as csv_writer
from testuary import config as config_module
from testuary import project
from testuary import orchestrator as main_orch
from testuary import starters
from testuary import target_phases as orchestrator
from testuary.starters import StarterResult
from phase_values import set_phase_value

# The phase code reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

# Captured before tests patch the modules, so the handoff test and the
# boundary fixture use the real implementations.
_REAL_PHASE_COVERAGE = orchestrator._phase_coverage
from testuary.consolidation import (  # noqa: E402
    PhaseEndDelivery,
    persist_phase_end as _real_persist_phase_end,
)
from agent_fixtures import logs_in  # noqa: E402
from phase_end_calls import phase_end  # noqa: E402


# Harness: a TargetPhasesContext with stubbed downstream phases. Each test
# patches the starter suites' generation.

def _ctx(tmp_path: Path, monkeypatch, *, reuse=None, hooks=None):
    src = tmp_path / "proj" / "mod.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def f(x):\n    return x\n", encoding="utf-8")
    # Project-root marker. Without it the root walk climbs out of tmp_path
    # into the enclosing repository and the phases work on that tree.
    (src.parent / "pyproject.toml").write_text("", encoding="utf-8")
    work_tests = tmp_path / "work" / "tests"
    work_tests.mkdir(parents=True, exist_ok=True)
    log_dir = tmp_path / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    results = tmp_path / "results"
    results.mkdir(parents=True, exist_ok=True)
    # Downstream phases never run for real here.
    monkeypatch.setattr(
        orchestrator, "_phase_coverage",
        lambda ctx, result, baseline=None: SimpleNamespace(
            test_file_name="test_mod.py", line_coverage=80.0,
            branch_coverage=70.0, status="ok",
        ),
    )
    monkeypatch.setattr(
        orchestrator, "_has_usable_test_file", lambda ctx, result: False,
    )
    monkeypatch.setattr(
        orchestrator, "_consolidate_coverage", lambda *args, **kwargs: None,
    )
    # The checkpoints a test observes in place of their phase_state writes.
    for name, observe in (hooks or {}).items():
        monkeypatch.setattr(orchestrator, name, observe)
    return orchestrator.TargetPhasesContext(
        log_print=lambda *a, **k: None,
        project_name="p",
        workspace="rank1_mod",
        top_file={"target_number": 1, "file_path": str(src), "key_functions": ["f"]},
        canonical_path=src,
        work_tests=work_tests,
        results_dir=results,
        target_logs_dir=log_dir,
        reuse=reuse or orchestrator.ReuseFlags(),
    )


def _ok_result(**over) -> StarterResult:
    res = StarterResult(
        status="ok", baseline_file=None, n_sets_requested=3,
        n_generated=3, n_green=2, pool_tests=10, selected_tests=6,
        baseline_line_coverage=75.0, baseline_branch_coverage=60.0,
        baseline_suite_seconds=1.5, requests_used=3,
        input_tokens=300, output_tokens=150, cache_hit_tokens=120,
        duration_s=12.0,
    )
    for k, v in over.items():
        setattr(res, k, v)
    return res


def _ok_result_for_ctx(ctx, **over) -> StarterResult:
    baseline = ctx.work_tests / "test_mod.py"
    baseline.write_text(
        "from mod import f\n\ndef test_f():\n    assert f(1) == 1\n",
        encoding="utf-8",
    )
    return _ok_result(baseline_file=str(baseline), **over)


def _persist_restorable_phase(ctx, phase: str, source: str) -> Path:
    from testuary import languages
    from testuary import evidence as validity

    archive = ctx.results_dir / ctx.workspace / "r0" / "archive" / phase
    archive.mkdir(parents=True, exist_ok=True)
    snapshot = archive / "result.py"
    snapshot.write_text(source, encoding="utf-8")
    backend = languages.get_backend("python")
    identity = backend.generated_test_target_identity(
        str(ctx.canonical_path), project.resolve_source_dir(str(ctx.canonical_path)),
    )
    validity.write_json_atomic(
        archive / "consolidation_record.json", {"delivered_from": "merged"},
    )
    validity.write_json_atomic(
        archive / "validity_delivered.json",
        validity.direct_record(
            identity=identity,
            analyzed_source=source,
            stage=f"test.{phase}.delivered",
        ),
    )
    return snapshot


def _persist_starter_boundary(ctx, baseline_source: str, runner_cm=None):
    """Write the boundary evidence the starter suites always leave: measure
    the baseline with the consolidation instrument of the coverage phase and
    persist ``archive/starter`` through the production code. ``runner_cm``
    injects a mutation runner so the baseline's kill matrix uses the test's
    vocabulary. The default follows the consolidation mutation switch."""
    import contextlib

    from testuary import languages
    from testuary.agent import mutation_agent

    target = str(ctx.canonical_path)
    backend = languages.get_backend("python")
    source_dir = project.resolve_source_dir(target)
    delivered_name = "test_mod.py"
    mut_cfg = config_module.get_agent_config(
        "test_mutation", parent="test_coverage"
    )
    identity = mutation_agent.mutation_measurement_identity(
        target_file_path=target,
        language="python",
        execution_name=delivered_name,
        config=dict(mut_cfg),
        attribution=orchestrator.measurement.union_attribution(mut_cfg),
    )
    if runner_cm is None:
        runner_cm = contextlib.nullcontext(None)
    with runner_cm as runner:
        phase = phase_end(
            backend, [baseline_source],
            target_file_path=target, source_dir=source_dir,
            timeout=120, mutation_runner=runner,
            mutation_method=config_module.consolidation_mutation_method(),
            stage_as_delivered=True,
            measurement_identity=identity,
            log=lambda m: None)
    _real_persist_phase_end(
        matrices_dir=ctx.results_dir / ctx.workspace / "r0" / "archive" / "starter",
        target_file=target, phase=phase,
        delivery=PhaseEndDelivery(winner_label="single_suite",
            delivered_km=phase.reduced_km, delivered_tests=phase.selected_tests,
            delivered_source=phase.reduced_source or baseline_source,
            execution_name=delivered_name))


# Sequencing and outcomes

def test_starter_runs_first_and_records_ok(tmp_path, monkeypatch):
    calls = []
    ctx = _ctx(tmp_path, monkeypatch)
    monkeypatch.setattr(
        starters, "run_starter_phase",
        lambda **kw: calls.append("starter") or _ok_result_for_ctx(ctx),
    )
    result = orchestrator.run_target_phases(ctx)
    assert calls == ["starter"]
    assert result.phase_outcomes["starter"] == "ok"
    assert result.starter_result is not None
    assert "starter" in result.phase_durations


def test_cancelled_zero_green_aborts_the_target(tmp_path, monkeypatch):
    monkeypatch.setattr(
        starters, "run_starter_phase",
        lambda **kw: _ok_result(
            status="cancelled_zero_passing", detail="no green tests",
        ),
    )
    coverage_ran = []
    monkeypatch.setattr(
        orchestrator, "_phase_coverage",
        lambda ctx, result: coverage_ran.append(1),
    )
    ctx = _ctx(tmp_path, monkeypatch)
    result = orchestrator.run_target_phases(ctx)
    assert result.phase_outcomes["starter"] == "failed"
    assert coverage_ran == []
    assert result.coverage_phase_failed       # main.py fails the target
    assert "no green tests" in result.phase_outcomes["starter_detail"]


def test_reuse_initial_skips_generation(tmp_path, monkeypatch, persist_starter_resume):
    called = []
    monkeypatch.setattr(
        starters, "run_starter_phase",
        lambda **kw: called.append(1) or _ok_result(),
    )
    ctx = _ctx(tmp_path, monkeypatch,
               reuse=orchestrator.ReuseFlags(starter=True))
    snapshot = persist_starter_resume(
        ctx,
        "from mod import f\n\ndef test_f():\n    assert f(1) == 1\n",
    )
    result = orchestrator.run_target_phases(ctx)
    assert called == []
    assert result.phase_outcomes["starter"] == "skipped_reused"
    # The coverage phase edits a copy in the work directory. The archive
    # snapshot stays as the starter suites delivered it.
    baseline = Path(result.starter_result.baseline_file)
    assert baseline.parent == ctx.work_tests
    assert baseline.read_bytes() == snapshot.read_bytes()
    assert result.starter_result.requests_used == 1
    assert result.starter_result.selected_tests == 1


def test_reuse_coverage_implies_starter_skip(tmp_path, monkeypatch, persist_starter_resume):
    called = []
    monkeypatch.setattr(
        starters, "run_starter_phase",
        lambda **kw: called.append(1) or _ok_result(),
    )
    ctx = _ctx(tmp_path, monkeypatch,
               reuse=orchestrator.ReuseFlags(coverage=True))
    snapshot = persist_starter_resume(
        ctx, "from mod import f\n\ndef test_f():\n    assert f(1) == 1\n",
    )
    result = orchestrator.run_target_phases(ctx)
    assert called == []
    assert result.starter_result.baseline_file == str(snapshot)
    assert result.starter_result.n_green == 1


def test_unsupported_starter_phase_fails_without_a_handoff(tmp_path, monkeypatch):
    monkeypatch.setattr(
        starters, "run_starter_phase",
        lambda **kw: (_ for _ in ()).throw(
            NotImplementedError("GenericBackend has no per-test failure report")
        ),
    )
    ctx = _ctx(tmp_path, monkeypatch)
    result = orchestrator.run_target_phases(ctx)
    assert result.phase_outcomes["starter"] == "failed"
    assert result.coverage_phase_failed


def test_resume_hooks_fire_for_starter(tmp_path, monkeypatch):
    events = []
    baseline = tmp_path / "work" / "tests" / "test_mod.py"

    def fake_starter(**kw):
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_text("def test_a():\n    assert 1\n",
                            encoding="utf-8")
        _persist_starter_boundary(
            ctx, baseline.read_text(encoding="utf-8"))
        return _ok_result(
            baseline_file=str(baseline),
            suite_line_stats={"mean": 71.0},
            suite_branch_stats={"mean": 61.0},
            suite_mutation_stats={"mean": 51.0},
        )

    monkeypatch.setattr(starters, "run_starter_phase", fake_starter)
    ctx = _ctx(
        tmp_path, monkeypatch,
        hooks=dict(
            _record_start=lambda _ctx, sp: events.append(("start", sp)),
            _record_done=lambda _ctx, sp, ok, arts, mets: events.append(
                ("done", sp, ok, dict(arts), dict(mets)),
            ),
        ),
    )
    orchestrator.run_target_phases(ctx)
    assert ("start", "starter") in events
    done = [e for e in events if e[0] == "done" and e[1] == "starter"]
    assert len(done) == 1
    _, _, ok, arts, mets = done[0]
    assert ok is True
    assert any(k.startswith("tests/") for k in arts)
    assert mets["baseline_line"] == 75.0
    assert mets["suites_passing"] == 2
    assert mets["suite_line_stats"] == {"mean": 71.0}
    assert mets["suite_branch_stats"] == {"mean": 61.0}
    assert mets["suite_mutation_stats"] == {"mean": 51.0}


def test_failed_starter_checkpoint_carries_the_reason(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(
        starters, "run_starter_phase",
        lambda **kw: _ok_result(
            status="cancelled_zero_passing", detail="empty pool",
        ),
    )
    ctx = _ctx(
        tmp_path, monkeypatch,
        hooks=dict(
            _record_start=lambda _ctx, sp: None,
            _record_done=lambda _ctx, sp, ok, arts, mets: events.append(
                (sp, ok, mets),
            ),
        ),
    )
    orchestrator.run_target_phases(ctx)
    sp, ok, mets = [e for e in events if e[0] == "starter"][0]
    assert ok is False
    assert "empty pool" in mets["reason"]


def test_baseline_handoff_reaches_coverage(tmp_path, monkeypatch):
    baseline = tmp_path / "work" / "tests" / "test_mod.py"
    received = {}

    def fake_starter(**kw):
        baseline.parent.mkdir(parents=True, exist_ok=True)
        baseline.write_text("def test_a():\n    assert 1\n",
                            encoding="utf-8")
        _persist_starter_boundary(
            ctx, baseline.read_text(encoding="utf-8"))
        return _ok_result(baseline_file=str(baseline), selected_tests=4,
                          baseline_missing_lines=(3, 4),
                          baseline_missing_branches=("2->4",))

    monkeypatch.setattr(starters, "run_starter_phase", fake_starter)
    ctx = _ctx(tmp_path, monkeypatch)
    # Restore the real _phase_coverage so its handoff construction runs and
    # intercept one level lower, at the agent entry.
    monkeypatch.setattr(
        orchestrator, "_phase_coverage", _REAL_PHASE_COVERAGE,
    )
    monkeypatch.setattr(
        orchestrator.coverage_agent, "init_logging", lambda *a, **k: None)
    monkeypatch.setattr(
        orchestrator.coverage_agent, "save_logs", lambda *a, **k: None)

    def fake_entry(project_name, top_file=None, test_output_dir=None,
                   baseline=None):
        received.update(baseline or {})
        return SimpleNamespace(test_file_name="test_mod.py", status="ok")

    monkeypatch.setattr(
        orchestrator.coverage_agent, "generate_coverage_tests", fake_entry)
    result = orchestrator.run_target_phases(ctx)
    assert received, (
        "the coverage agent was never reached, so the handoff was never "
        f"built: {result.phase_outcomes}")
    assert received["line_coverage"] == 75.0
    assert received["branch_coverage"] == 60.0
    assert received["test_count"] == 4
    assert "def test_a" in received["content"]
    assert received["missing_lines"] == (3, 4)
    assert received["missing_branches"] == ("2->4",)


# Post-coverage consolidation: merge with the baseline and lossless cover

BASELINE_SUITE = ("from mod import f\n\n"
                  "def test_negative():\n    assert f(-1) == -1\n")
AGENT_SUITE = ("from mod import f\n\n"
               "def test_positive():\n    assert f(1) == 1\n")


def _consolidation_ctx(tmp_path, monkeypatch):
    """A ctx whose starter suites write a real baseline and whose coverage
    phase replaces the file, removing the baseline test. Runs from tmp_path
    so the repository's config.ini cannot point resolve_source_dir at a
    real project."""
    monkeypatch.chdir(tmp_path)
    src = tmp_path / "proj" / "mod.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text(
        "def f(x):\n    if x > 0:\n        return 1\n    return -1\n",
        encoding="utf-8",
    )
    # The marker resolve_source_dir walks up to (any real target's
    # project has one). Measurement runs with this dir as import root.
    (src.parent / "pyproject.toml").write_text("", encoding="utf-8")
    work_tests = tmp_path / "work" / "tests"
    work_tests.mkdir(parents=True, exist_ok=True)
    baseline_path = work_tests / "test_mod.py"
    for d in ("logs", "results"):
        (tmp_path / d).mkdir(exist_ok=True)
    ctx = orchestrator.TargetPhasesContext(
        log_print=lambda *a, **k: None,
        project_name="p",
        workspace="rank1_mod",
        top_file={"target_number": 1, "file_path": str(src), "key_functions": ["f"]},
        canonical_path=src,
        work_tests=work_tests,
        results_dir=tmp_path / "results",
        target_logs_dir=tmp_path / "logs",
        reuse=orchestrator.ReuseFlags(),
    )

    def fake_starter(**kw):
        baseline_path.write_text(BASELINE_SUITE, encoding="utf-8")
        _persist_starter_boundary(ctx, BASELINE_SUITE)
        return _ok_result(baseline_file=str(baseline_path))

    def fake_coverage(ctx_, result_, baseline=None):
        # The agent rewrote the suite and lost the baseline test.
        baseline_path.write_text(AGENT_SUITE, encoding="utf-8")
        return SimpleNamespace(
            test_file_name="test_mod.py", line_coverage=50.0,
            branch_coverage=50.0, status="ok",
        )

    monkeypatch.setattr(starters, "run_starter_phase", fake_starter)
    monkeypatch.setattr(orchestrator, "_phase_coverage", fake_coverage)
    monkeypatch.setattr(
        orchestrator, "_has_usable_test_file", lambda ctx, result: False,
    )
    return ctx, baseline_path


@pytest.mark.mutation_tool
def test_consolidation_restores_dropped_baseline_test(tmp_path, monkeypatch):
    """With the real backend, consolidation restores the baseline's
    negative-branch test the agent removed, and the recorded coverage
    reflects the merged suite."""
    ctx, suite_path = _consolidation_ctx(tmp_path, monkeypatch)
    result = orchestrator.run_target_phases(ctx)
    final = suite_path.read_text(encoding="utf-8")
    assert "test_positive" in final          # the agent's work kept
    assert "test_negative" in final          # the removed test restored
    assert result.test_result.line_coverage == 100.0


def test_consolidation_failure_without_mutation_restores_the_baseline(
    tmp_path, monkeypatch,
):
    """A run without phase-end mutation whose merge cannot be measured falls back to
    keep-best over the measured candidates: the baseline, the previous
    phase's delivered suite, wins and is delivered, so the boundary never falls below it
    and never delivers the agent's unverified file."""
    import json

    from testuary import consolidation as consolidate_mod

    ctx, suite_path = _consolidation_ctx(tmp_path, monkeypatch)
    monkeypatch.setattr(config_module, "consolidation_measure_mutation", lambda: False)
    monkeypatch.setattr(
        consolidate_mod, "consolidate_phase_end",
        lambda *a, **k: consolidate_mod.PhaseConsolidation(
            detail="merge could not be measured"),
    )
    orchestrator.run_target_phases(ctx)
    record = json.loads(
        (ctx.results_dir / ctx.workspace / "r0" / "archive" / "coverage"
         / "consolidation_record.json").read_text(encoding="utf-8"))
    assert record["delivered_from"] == "baseline"
    assert suite_path.read_text(encoding="utf-8") == BASELINE_SUITE


def test_consolidation_failure_with_mutation_selects_measured_baseline(
    tmp_path, monkeypatch,
):
    """A merge failure does not erase the measured inherited candidate.

    The common selector excludes the unavailable merge and reduction, then
    selects the complete passing baseline rather than failing the later phase.
    """
    import json
    from contextlib import contextmanager

    from testuary import consolidation as consolidate_mod
    from testuary import measurement

    ctx, suite_path = _consolidation_ctx(tmp_path, monkeypatch)

    @contextmanager
    def fake_runner_cm(**_kwargs):
        def measure(_directory):
            return {"mutants": [{
                "id": "m0",
                "status": "killed",
                "killer": "test_mod.py::test_negative",
                "killers": ["test_mod.py::test_negative"],
                "killing_tests": killing_tests(["test_mod.py::test_negative"]),
            }]}
        yield measure

    monkeypatch.setattr(
        measurement, "union_mutation_runner", fake_runner_cm
    )

    def fake_starter(**_kwargs):
        suite_path.write_text(BASELINE_SUITE, encoding="utf-8")
        _persist_starter_boundary(
            ctx, BASELINE_SUITE, runner_cm=fake_runner_cm()
        )
        return _ok_result(baseline_file=str(suite_path))

    monkeypatch.setattr(starters, "run_starter_phase", fake_starter)
    monkeypatch.setattr(
        consolidate_mod, "consolidate_phase_end",
        lambda *a, **k: consolidate_mod.PhaseConsolidation(
            detail="merge could not be measured"),
    )
    orchestrator.run_target_phases(ctx)
    record = json.loads(
        (ctx.results_dir / ctx.workspace / "r0" / "archive" / "coverage"
         / "consolidation_record.json").read_text(encoding="utf-8"))
    assert record["delivered_from"] == "baseline"
    assert suite_path.read_text(encoding="utf-8") == BASELINE_SUITE


def test_no_baseline_fails_before_coverage(tmp_path, monkeypatch):
    ctx, suite_path = _consolidation_ctx(tmp_path, monkeypatch)
    monkeypatch.setattr(
        starters, "run_starter_phase",
        lambda **kw: _ok_result(baseline_file=None),
    )
    with pytest.raises(RuntimeError, match="produced no suite for the coverage phase"):
        orchestrator.run_target_phases(ctx)
    assert not suite_path.exists()


# Coverage read order under a baseline handoff

def _drive_entry(tmp_path, monkeypatch, baseline):
    """Run the real generate_coverage_tests up to run_phase, capturing
    the constructed agent and the substituted user prompt."""
    from testuary.agent import coverage_agent as ca

    src = tmp_path / "proj" / "mod.py"
    src.parent.mkdir(parents=True, exist_ok=True)
    src.write_text("def f(x):\n    return x\n", encoding="utf-8")
    captured = {}

    def fake_run_phase(self, *, project_name, user_prompt):
        captured["agent"] = self
        captured["user_prompt"] = user_prompt
        return SimpleNamespace(test_file_name="test_mod.py", status="ok")

    monkeypatch.setattr(ca.CoverageAgent, "run_phase", fake_run_phase)
    logs_in(monkeypatch, tmp_path / "logs")
    ca.generate_coverage_tests(
        "p",
        top_file={"target_number": 1, "file_path": str(src), "key_functions": ["f"]},
        test_output_dir=str(tmp_path / "tests_out"),
        baseline=baseline,
    )
    return captured


def test_baseline_switches_designated_read_and_prompt(tmp_path, monkeypatch):
    captured = _drive_entry(tmp_path, monkeypatch, baseline={
        "content": "def test_a():\n    assert 1\n",
        "line_coverage": 81.0, "branch_coverage": 72.5,
        "suite_seconds": 1.0, "test_count": 1,
    })
    agent = captured["agent"]
    assert agent.designated_read_name == "read_tests"
    prompt = captured["user_prompt"]
    assert "A green suite is already on disk in your working file" in prompt
    assert "It measured line 81.0%, branch 72.5%; goal metric (" in prompt
    assert "``read_tests``" in prompt
    # The agent's best-green was set too.
    assert agent.session.best_green.snapshot["coverage"] == 81.0


def test_a_baseline_without_branches_is_stated_without_a_branch_figure(tmp_path, monkeypatch):
    """A target without branches: the prompt names its line coverage and
    that it has no branches, and its goal is line coverage, never
    "None% branch"."""
    from testuary.agent import coverage_agent as ca

    set_phase_value(monkeypatch, ca._AGENT_CONFIG, "goal_metric", "both")
    set_phase_value(monkeypatch, ca._AGENT_CONFIG, "goal", 100)
    captured = _drive_entry(tmp_path, monkeypatch, baseline={
        "content": "def test_a():\n    assert 1\n",
        "line_coverage": 60.0, "branch_coverage": None,
        "suite_seconds": 1.0, "test_count": 1,
    })
    prompt = captured["user_prompt"]
    assert "It measured line 60.0%, no branches; goal metric (line) 60.0%, goal 100%." in prompt
    assert "100% line coverage (the target has no branches)" in prompt
    assert "None%" not in prompt


def test_the_prompt_states_the_stagnation_threshold(tmp_path, monkeypatch):
    """A gain under min_improvement_points counts as no improvement, except
    below minimum: the prompt states both."""
    from testuary.agent import coverage_agent as ca

    for key, value in {"max_measurements_without_improvement": 10, "min_improvement_points": 0.5,
                       "minimum": 75}.items():
        set_phase_value(monkeypatch, ca._AGENT_CONFIG, key, value)
    prompt = _drive_entry(tmp_path, monkeypatch, baseline={
        "content": "def test_a():\n    assert 1\n",
        "line_coverage": 81.0, "branch_coverage": 72.5,
        "suite_seconds": 1.0, "test_count": 1,
    })["user_prompt"]
    assert ("If the measurement stalls (no gain of at least 0.5\n"
            "  points across 10 consecutive measurements; while the goal metric\n"
            "  is below 75%, any gain counts)") in prompt


def test_the_baseline_measurement_is_stated_to_one_decimal(tmp_path, monkeypatch):
    """The prompt states the baseline's measured coverage to one decimal,
    floored like every percentage shown to the model, never as a 14-digit
    float."""
    from testuary.agent import coverage_agent as ca

    set_phase_value(monkeypatch, ca._AGENT_CONFIG, "goal_metric", "both")
    set_phase_value(monkeypatch, ca._AGENT_CONFIG, "goal", 100)
    prompt = _drive_entry(tmp_path, monkeypatch, baseline={
        "content": "def test_a():\n    assert 1\n",
        "line_coverage": 95.08196721311475, "branch_coverage": 100.0,
        "suite_seconds": 1.0, "test_count": 1,
    })["user_prompt"]
    assert "It measured line 95.0%, branch 100.0%; goal metric (lower of line and branch) 95.0%, goal 100%." in prompt
    assert "95.08196721311475" not in prompt


def test_missing_designated_initial_read_fails_closed(tmp_path, monkeypatch):
    captured = _drive_entry(tmp_path, monkeypatch, baseline={
        "content": "def test_a():\n    assert 1\n",
        "line_coverage": 81.0, "branch_coverage": 72.5,
        "suite_seconds": 1.0, "test_count": 1,
    })
    agent = captured["agent"]
    agent.designated_read_name = "missing_read_tool"

    with pytest.raises(RuntimeError, match="designated initial read tool"):
        offered_tools(agent)


def test_no_baseline_fails_before_coverage_model_call(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="requires a nonempty validated suite"):
        _drive_entry(tmp_path, monkeypatch, baseline=None)


# Coverage-agent starting state

def test_seed_baseline_populates_best_green():
    from testuary.agent.coverage_agent import (
        TestCoverageSession, _seed_baseline,
    )

    session = TestCoverageSession()
    _seed_baseline(session, {
        "content": "def test_a():\n    assert 1\n",
        "line_coverage": 72.5,
        "branch_coverage": 61.0,
        "suite_seconds": 2.0,
        "test_count": 7,
    })
    snap = session.best_green.snapshot
    assert snap["coverage"] == 72.5
    assert snap["branch_coverage"] == 61.0
    assert snap["test_count"] == 7
    assert snap["exec_time_ms"] == 2000.0
    assert session.best_coverage_pct == 72.5
    assert session.best_branch_coverage_pct == 61.0


def test_seed_baseline_tolerates_missing_metrics():
    from testuary.agent.coverage_agent import (
        TestCoverageSession, _seed_baseline,
    )

    session = TestCoverageSession()
    _seed_baseline(session, {"content": "x"})
    assert session.best_green.snapshot["coverage"] is None
    assert session.best_coverage_pct is None


# tests.csv columns

def _read_row(results_dir: Path) -> dict:
    with (results_dir / "tests.csv").open(encoding="utf-8-sig") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    return rows[0]


def test_csv_row_carries_starter_columns(tmp_path):
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_csv",
        target_number=1,
        file_path="proj/mod.py",
        starter_result=_ok_result(
            suite_line_stats={"mean": 70.0, "median": 71.0, "std": 5.0,
                              "min": 60.0, "max": 80.0},
            suite_branch_stats={"mean": 55.0, "median": 55.0, "std": 0.0,
                                "min": 55.0, "max": 55.0},
            n_unmeasured_coverage=2, n_unmeasured_mutation=1,
        ),
    )
    row = _read_row(tmp_path)
    assert row["starter_status"] == "ok"
    assert row["starter_suites_generated"] == "3"
    assert row["starter_suites_passing"] == "2"
    assert row["starter_suites_unmeasured_coverage"] == "2"
    assert row["starter_suites_unmeasured_mutation"] == "1"
    assert row["starter_merged_tests"] == "10"
    assert row["starter_delivered_tests"] == "6"
    assert row["starter_delivered_from"] == "merged"
    assert row["starter_baseline_line"] == "75.0"
    assert row["starter_baseline_branch"] == "60.0"
    # per-starter-suite (pre-merge) distribution columns flow through
    assert row["starter_suite_line_mean"] == "70.0"
    assert row["starter_suite_line_max"] == "80.0"
    assert row["starter_suite_branch_std"] == "0.0"
    # a metric no starter suite supplied stays blank
    assert row["starter_suite_mutation_mean"] == ""
    assert row["starter_requests"] == "3"
    assert row["starter_input_tokens"] == "300"
    assert row["starter_output_tokens"] == "150"
    assert row["starter_cache_hit_tokens"] == "120"
    assert row["starter_duration_s"] == "12.0"
    # No mutation was measured, so no method is named.
    assert row["starter_suite_mutation_method"] == ""


def test_csv_row_uses_the_actual_starter_selector_winner(tmp_path):
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_csv",
        target_number=1,
        file_path="proj/mod.py",
        starter_result=_ok_result(winner="reduced"),
    )

    assert _read_row(tmp_path)["starter_delivered_from"] == "reduced"


def test_csv_row_carries_consolidation_stage_columns(tmp_path):
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_csv",
        target_number=1,
        file_path="proj/mod.py",
        consolidation_stats={
            "coverage": {"merged_tests": 20, "delivered_tests": 12,
                         "delivered_from": "merged"},
            "mutation": {"merged_tests": 30, "delivered_tests": None,
                         "delivered_from": "merged"},
        },
    )
    row = _read_row(tmp_path)
    assert row["coverage_merged_tests"] == "20"
    assert row["coverage_delivered_tests"] == "12"
    assert row["coverage_delivered_from"] == "merged"
    assert row["mutation_merged_tests"] == "30"
    assert row["mutation_delivered_tests"] == ""   # blank when not measured


def test_csv_row_without_starter_leaves_columns_blank(tmp_path):
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_csv",
        target_number=1,
        file_path="proj/mod.py",
    )
    row = _read_row(tmp_path)
    assert row["starter_status"] == ""
    assert row["starter_suites_passing"] == ""
    assert row["starter_suites_unmeasured_coverage"] == ""


def test_csv_row_cancelled_status_recorded(tmp_path):
    csv_writer.write_tests_row(
        results_dir=tmp_path, workspace="rank1_csv",
        target_number=1,
        file_path="proj/mod.py",
        starter_result=_ok_result(status="cancelled_zero_passing",
                                  n_green=0),
    )
    row = _read_row(tmp_path)
    assert row["starter_status"] == "cancelled_zero_passing"
    assert row["starter_suites_passing"] == "0"


# CLI: reused phases

def _args(argv=None, **over):
    ns = main_mod._build_parser().parse_args(argv or [])
    main_mod._check_run_options(ns)
    for k, v in over.items():
        setattr(ns, k, v)
    return ns


def test_reuse_starter_parses():
    ns = _args(["--reuse", "starter"])
    assert ns.reuse_starter is True
    assert ns.reuse_coverage is False


def test_an_unknown_reuse_value_is_refused():
    for value in ("seeds", "initial", "metrics"):
        with pytest.raises(ValueError, match=f"--reuse: unknown value {value}. Choose from"):
            _args(["--reuse", value])


def test_reuse_takes_the_four_phase_names_only():
    with pytest.raises(ValueError, match="--reuse: unknown value test. Choose from "
                                         "starter, coverage, mutation, finishing$"):
        _args(["--reuse", "test"])
    ns = _args(["--reuse", "starter,coverage,mutation,finishing"])
    assert all(getattr(ns, f"reuse_{phase}") for phase in main_orch.PHASES)


def test_reuse_starter_and_coverage_flow_to_flags():
    """ReuseFlags wiring: --reuse starter and --reuse coverage each force
    the starter skip."""
    for argv, expect in [
        (["--reuse", "starter"], True),
        (["--reuse", "coverage"], True),
        ([], False),
    ]:
        ns = _args(argv)
        skips = {sp: False for sp in main_orch.PHASES}
        flag = (
            ns.reuse_starter or ns.reuse_coverage
            or skips["starter"]
        )
        assert flag is expect, argv


def test_phase_lists_agree_and_start_with_initial():
    assert main_orch.PHASES is phase_state.PHASES
    assert main_orch.PHASES[0] == "starter"
    assert phase_state.PHASES[0] == "starter"
    fields = set(orchestrator.ReuseFlags.__dataclass_fields__)
    assert {"starter", "coverage", "mutation", "finishing"} <= fields


def test_no_resume_skips_covers_initial():
    skips = phase_state.no_resume_skips()
    assert skips["starter"] is False
    assert set(skips) == set(main_orch.PHASES)


# Configuration settings

def test_initial_config_defaults(in_run):
    in_run()
    cfg = config_module.get_agent_config("test_initial")
    assert cfg["suite_count"] == 10
    assert cfg["refill_failed_suites"] is True
    assert cfg["temperature"] is None
    # seed_base defaults to the global [main] seed (0), the one source for
    # every seeded operation. An env override still shifts it.
    assert cfg["seed_base"] == 0
    assert int(cfg["suite_run_timeout_seconds"]) > 0
    assert cfg["reduction_solver"] in ("exact", "greedy")


def test_initial_config_env_overrides(monkeypatch, in_run):
    monkeypatch.setenv("TESTUARY_STARTER_SUITE_COUNT", "25")
    monkeypatch.setenv("TESTUARY_STARTER_REFILL_FAILED_SUITES", "false")
    monkeypatch.setenv("TESTUARY_STARTER_TEMPERATURE", "1.1")
    monkeypatch.setenv("TESTUARY_STARTER_SEED_BASE", "42")
    in_run()
    cfg = config_module.get_agent_config("test_initial")
    assert cfg["suite_count"] == 25
    assert cfg["refill_failed_suites"] is False
    assert cfg["temperature"] == 1.1
    assert cfg["seed_base"] == 42


def test_consolidation_measure_mutation_default_on_and_env_flip(monkeypatch, in_run):
    """Mutation measurement at every consolidation is on by default (the
    cover is kill-lossless). The setting's variable turns it off."""
    def measured():
        in_run()
        return config_module.consolidation_measure_mutation()

    assert measured() is True
    monkeypatch.setenv("TESTUARY_MAIN_PHASE_END_MUTATION", "false")
    assert measured() is False


def test_consolidation_mutation_method_default_both_env_and_garbage(
        monkeypatch, tmp_path, in_run):
    """The reduced-suite mutation method defaults to 'both' (derive, then a
    sanity measurement), and the setting's variable selects derive or
    measure. The settings refuse any other value from a variable or a
    settings file, naming the setting and the choices."""
    def method():
        in_run()
        return config_module.consolidation_mutation_method()

    assert method() == "both"
    monkeypatch.setenv("TESTUARY_MAIN_REDUCED_SUITE_MUTATION", "derive")
    assert method() == "derive"
    monkeypatch.setenv("TESTUARY_MAIN_REDUCED_SUITE_MUTATION", "nonsense")
    with pytest.raises(ValueError,
                       match=r"\[main\] reduced_suite_mutation: choose one of derive, measure, both"):
        method()
    monkeypatch.delenv("TESTUARY_MAIN_REDUCED_SUITE_MUTATION")
    config = tmp_path / "typo.ini"
    config.write_text("[main]\nreduced_suite_mutation = measured\n",
                      encoding="utf-8")
    monkeypatch.setenv("TESTUARY_CONFIG_FILE", str(config))
    with pytest.raises(ValueError, match=r"\[main\] reduced_suite_mutation: choose one of"):
        method()


@pytest.mark.parametrize("entry", ["api", "cli"])
def test_a_mistyped_mutation_method_stops_the_run_before_any_target(
        monkeypatch, capsys, entry, in_run):
    """The run stops before any expensive work, naming
    the value and the choices."""
    in_run.leave()
    monkeypatch.setenv("TESTUARY_MAIN_REDUCED_SUITE_MUTATION", "measurd")

    def no_phase(*_a, **_k):
        raise AssertionError("no phase may run")

    monkeypatch.setattr(main_orch, "_preflight_model_configuration", no_phase)
    try:
        if entry == "api":
            from testuary.settings import ConfigurationError
            with pytest.raises(ConfigurationError, match="reduced_suite_mutation.*choose one of"):
                main_orch.run_pipeline(_args(), None)
        else:
            # A generation request: the bare command only prints help.
            monkeypatch.setattr("sys.argv", ["testuary", "--source-files", "module.py"])
            with pytest.raises(SystemExit) as stopped:
                main_mod.main()
            assert stopped.value.code == 2
            assert "reduced_suite_mutation" in capsys.readouterr().err
    finally:
        assert config_module.active_settings() is None


@pytest.mark.parametrize("entry", ["api", "cli"])
def test_a_mistyped_context_rebuild_stops_the_run_before_any_target(
        monkeypatch, capsys, entry, in_run):
    """An unknown context rebuild policy stops the run
    before any expensive work, not at the first agent of a phase."""
    in_run.leave()
    monkeypatch.setenv("TESTUARY_MAIN_CONTEXT_REBUILD", "sometimes")

    def no_phase(*_a, **_k):
        raise AssertionError("no phase may run")

    monkeypatch.setattr(main_orch, "_preflight_model_configuration", no_phase)
    try:
        if entry == "api":
            from testuary.settings import ConfigurationError
            with pytest.raises(ConfigurationError, match="context_rebuild.*choose one of"):
                main_orch.run_pipeline(_args(), None)
        else:
            # A generation request: the bare command only prints help.
            monkeypatch.setattr("sys.argv", ["testuary", "--source-files", "module.py"])
            with pytest.raises(SystemExit) as stopped:
                main_mod.main()
            assert stopped.value.code == 2
            assert "context_rebuild" in capsys.readouterr().err
    finally:
        assert config_module.active_settings() is None


def test_suite_mutation_defaults_to_measure_and_takes_each_method(tmp_path, in_run):
    """Each starter suite is measured alone by default. Derive and both are
    accepted, and the effective value is recorded with the run's settings."""
    for value, expected in ((None, "measure"), ("derive", "derive"), ("BOTH", "both")):
        environment = {} if value is None else {"TESTUARY_STARTER_SUITE_MUTATION": value}
        in_run(config_files=[], environment=environment, project_settings=True)
        assert config_module.get_agent_config("test_initial")["suite_mutation"] == expected
        resolved_settings = config_module.resolved_test_preset()
        assert resolved_settings["sections"]["starter"]["suite_mutation"] == expected


def test_no_generation_cap_setting_exists():
    """A generation runs until the model stops: no setting caps its output."""
    cfg = config_module.get_agent_config("test_initial")
    assert "generation_max_tokens" not in cfg
    bundled = (Path(testuary.__file__).resolve().parent
               / "config.ini").read_text(encoding="utf-8")
    assert "generation_max_tokens" not in bundled


def test_starter_cleanup_bound_is_pinned_in_bundled_config():
    cfg = config_module.get_agent_config("test_initial")
    assert cfg["failing_test_cleanup_rounds"] == 3
    text = (Path(testuary.__file__).resolve().parent / "config.ini").read_text(encoding="utf-8")
    assert "failing_test_cleanup_rounds = 3" in text


def test_coverage_consolidation_persists_no_matrix_without_mutation(
    tmp_path, monkeypatch,
):
    """With consolidation mutation off, the coverage phase delivers the merged
    reduced suite but writes no kill matrix, neither the audit sidecar nor
    the mutation phase's baseline handoff."""
    ctx, suite_path = _consolidation_ctx(tmp_path, monkeypatch)
    monkeypatch.setattr(config_module, "consolidation_measure_mutation", lambda: False)
    orchestrator.run_target_phases(ctx)
    # The merge was delivered: both branch tests are present (the agent had only one).
    final = suite_path.read_text(encoding="utf-8")
    assert "test_negative" in final and "test_positive" in final
    cov_arch = ctx.results_dir / ctx.workspace / "r0" / "archive" / "coverage"
    for name in ("selected_kill_matrix.json", "reduced_kill_matrix.json"):
        assert not (cov_arch / name).exists(), name
    assert not (ctx.work_tests / ".mutation_baseline_matrix.json").exists()


def test_coverage_consolidation_persists_the_delivered_matrix_with_mutation(
    tmp_path, monkeypatch,
):
    """When mutation is measured and the reduction is delivered, the coverage phase
    persists the delivered suite's matrix under the canonical name, beside
    the reduction's own, as the audit sidecar and as the mutation phase's
    baseline handoff."""
    import json as _json
    from contextlib import contextmanager
    from testuary import measurement as test_metrics

    ctx, suite_path = _consolidation_ctx(tmp_path, monkeypatch)
    suite_path.write_text(AGENT_SUITE, encoding="utf-8")

    @contextmanager
    def fake_runner(**kw):
        def _measure(d):
            # Name each killer by whichever file the runner is pointed at (the
            # merged suite holds both branch tests in one file).
            killer_file = {}
            for fp in Path(d).glob("*.py"):
                body = fp.read_text(encoding="utf-8")
                for tname in ("test_negative", "test_positive"):
                    if f"def {tname}" in body:
                        killer_file[tname] = fp.name
            muts = []
            for mid, tname in (("m_neg", "test_negative"),
                               ("m_pos", "test_positive")):
                if tname in killer_file:
                    muts.append({"id": mid, "status": "killed",
                                 "killer": f"{killer_file[tname]}::{tname}"})
            muts.append({"id": "surv", "status": "survived", "killer": None})
            return {"mutants": muts}
        yield per_suite_double(_measure)

    monkeypatch.setattr(test_metrics, "union_mutation_runner", fake_runner)
    monkeypatch.setattr(config_module, "consolidation_measure_mutation", lambda: True)

    # The baseline's boundary evidence, measured under the same fake runner
    # so its kill vocabulary matches the instrument this test fabricates.
    _persist_starter_boundary(ctx, BASELINE_SUITE, runner_cm=fake_runner())

    result = orchestrator.TestPipelineResult()
    result.test_file_name = "test_mod.py"
    result.test_result = SimpleNamespace(
        line_coverage=50.0, branch_coverage=50.0)
    orchestrator._consolidate_coverage(ctx, result, BASELINE_SUITE)

    cov_arch = ctx.results_dir / ctx.workspace / "r0" / "archive" / "coverage"
    cov_matrix = cov_arch / "selected_kill_matrix.json"
    assert cov_matrix.exists()
    # logs/ holds only logs: the matrices moved to the archive, none under logs.
    assert not (ctx.target_logs_dir / "matrices").exists()
    km = _json.loads(cov_matrix.read_text(encoding="utf-8"))
    assert any(m["status"] == "killed" for m in km["mutants"])
    # The reduction was delivered, so its own matrix describes the same bytes.
    reduced_km = _json.loads(
        (cov_arch / "reduced_kill_matrix.json").read_text(encoding="utf-8"))
    assert reduced_km["source_sha256"] == km["source_sha256"]
    assert (ctx.work_tests / ".mutation_baseline_matrix.json").exists()
    # The merged suite's matrix before reduction is delivered beside the reduced one:
    # the score of the unreduced baseline and agent suite, backing the
    # lossless-cover claim.
    merged_matrix = cov_arch / "merged_kill_matrix.json"
    assert merged_matrix.exists()
    ukm = _json.loads(merged_matrix.read_text(encoding="utf-8"))
    assert {m["id"] for m in ukm["mutants"]} == {"m_neg", "m_pos", "surv"}
    assert sum(1 for m in ukm["mutants"] if m["status"] == "killed") == 2
    # The stage stats ride the pipeline result into the CSV columns.
    stats = result.consolidation_stats["coverage"]
    assert stats["delivered_from"] == "reduced"
    assert stats["merged_tests"] >= stats["delivered_tests"] >= 1
    # The per-test coverage sidecars back the lossless-cover claim on disk:
    # the merged suite's sidecar carries every merged test's tokens. The
    # reduced one carries the kept subset, and its tests' tokens together must
    # re-derive the merged suite's exactly.
    ucov = _json.loads(
        (cov_arch / "merged_coverage_matrix.json")
        .read_text(encoding="utf-8"))
    rcov = _json.loads(
        (cov_arch / "reduced_coverage_matrix.json")
        .read_text(encoding="utf-8"))
    assert len(ucov["per_test_tokens"]) == stats["merged_tests"]
    assert sorted(rcov["kept_tests"]) == sorted(rcov["per_test_tokens"])
    merged_tokens = {t for toks in ucov["per_test_tokens"].values() for t in toks}
    kept_tokens = {t for toks in rcov["per_test_tokens"].values() for t in toks}
    assert kept_tokens == merged_tokens




def test_a_fresh_starter_phase_is_not_flagged_as_reusing_starters(
        tmp_path, monkeypatch):
    """The flag exists so a phase that skipped generation is never averaged
    with one that paid for it. Settings values arrive as text, and every
    non-empty string is truthy, so the flag has to be read through the
    coercing accessor rather than from the raw layer."""
    import json

    from testuary import phase_state

    ctx = _ctx(tmp_path, monkeypatch)
    monkeypatch.setattr(
        starters, "run_starter_phase",
        lambda **kw: _ok_result_for_ctx(ctx),
    )
    orchestrator.run_target_phases(ctx)

    state = json.loads(
        phase_state.state_path(ctx.results_dir / ctx.workspace)
        .read_text(encoding="utf-8"))
    segments = state["starter"]["0"]["segments"]
    assert segments, "the starter phase recorded no segment"
    assert segments[-1]["starter_suites_reused"] is False


@pytest.mark.parametrize("metric", ["both", "branch"])
def test_a_target_without_branches_reads_as_judged_by_line_coverage(
        tmp_path, monkeypatch, metric):
    """The default metric (branch) and both judge a target without
    branches by line coverage. The prompt says so, states the goal metric
    with the measured value, and never asks for branch coverage."""
    from testuary.agent import coverage_agent as ca

    set_phase_value(monkeypatch, ca._AGENT_CONFIG, "goal_metric", metric)
    set_phase_value(monkeypatch, ca._AGENT_CONFIG, "goal", 100)
    prompt = _drive_entry(tmp_path, monkeypatch, baseline={
        "content": "def test_a():\n    assert 1\n",
        "line_coverage": 60.0, "branch_coverage": None,
        "suite_seconds": 1.0, "test_count": 1,
    })["user_prompt"]
    assert ("It measured line 60.0%, no branches; goal metric (line) 60.0%, "
            "goal 100%.") in prompt
    assert "100% line coverage (the target has no branches)" in prompt
    assert "line and branch coverage reach" not in prompt
    assert "its coverage reaches the goal" in prompt


@pytest.mark.parametrize("metric, clause", [
    ("line", "90% line coverage"),
    ("branch", "90% branch coverage"),
    ("either", "90% on EITHER line or branch (whichever is higher)"),
    ("both", "90% on BOTH line and branch (whichever is lower must clear the bar)"),
])
def test_each_goal_metric_value_states_its_own_goal(tmp_path, monkeypatch, capsys, metric, clause):
    """Each goal_metric value gives the model its own goal clause, and none
    is reported as unknown."""
    from testuary.agent import coverage_agent as ca

    set_phase_value(monkeypatch, ca._AGENT_CONFIG, "goal_metric", metric)
    set_phase_value(monkeypatch, ca._AGENT_CONFIG, "goal", 90)
    prompt = _drive_entry(tmp_path, monkeypatch, baseline={
        "content": "def test_a():\n    assert 1\n",
        "line_coverage": 81.0, "branch_coverage": 72.5,
        "suite_seconds": 1.0, "test_count": 1,
    })["user_prompt"]
    assert f"- Coverage goal: {clause}." in prompt
    assert "Unknown goal_metric" not in "".join(capsys.readouterr())


def test_a_fractional_goal_is_stated_as_set_in_the_coverage_task(tmp_path, monkeypatch):
    """The goal is a float setting: the task states 87.5 as set, not
    truncated."""
    from testuary.agent import coverage_agent as ca

    set_phase_value(monkeypatch, ca._AGENT_CONFIG, "goal_metric", "line")
    set_phase_value(monkeypatch, ca._AGENT_CONFIG, "goal", 87.5)
    prompt = _drive_entry(tmp_path, monkeypatch, baseline={
        "content": "def test_a():\n    assert 1\n",
        "line_coverage": 81.0, "branch_coverage": 72.5,
        "suite_seconds": 1.0, "test_count": 1,
    })["user_prompt"]
    assert "- Coverage goal: 87.5% line coverage." in prompt


def test_the_coverage_start_names_the_working_file_and_the_inherited_suite(tmp_path, monkeypatch):
    """Both loops start from a pre-filled working file and are consolidated
    with the unchanged previous suite: the coverage task says so in the
    mutation task's words."""
    from testuary.agent import coverage_agent as ca

    set_phase_value(monkeypatch, ca._AGENT_CONFIG, "goal_metric", "both")
    set_phase_value(monkeypatch, ca._AGENT_CONFIG, "goal", 100)
    captured = _drive_entry(tmp_path, monkeypatch, baseline={
        "content": "def test_a():\n    assert 1\n",
        "line_coverage": 81.0, "branch_coverage": 72.5,
        "suite_seconds": 1.0, "test_count": 1,
    })
    agent = captured["agent"]
    name = agent.test_backend().target_test_path(
        agent.deps.test_output_dir, agent.deps.target_file_path).name
    prompt = captured["user_prompt"]
    assert (f"A green suite is already on disk in your working file\n"
            f"{name}; ``read_tests`` returns it. It measured line 81.0%") in prompt
    assert ("Extend it in place.\n"
            "The inherited suite remains available for phase-end comparison, but\n"
            "consolidation does not assume every edit preserves prior evidence.") in prompt
