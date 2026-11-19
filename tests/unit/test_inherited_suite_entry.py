"""The suite a phase inherits is not checked at the phase's entry.

The phase end that delivered it chose it among suites that pass, and the
loop's start shows the model what the suite does. Nothing runs or is removed
before the loop. The next phase end reuses the inherited evidence of an
unchanged suite: it passed its normal run, and no acceptance as delivered is
recorded or needed.

The real Python backend runs every suite. Only the mutation engine is a fixed
catalogue, so a mutant's status follows from the tests present."""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import test_unchanged_suite_coverage_boundary as unchanged
from testuary import languages
from testuary import target_phases as orchestrator
from testuary.consolidation import (
    persist_phase_end,
    PhaseEndDelivery,
)
import pytest
from phase_end_calls import phase_end

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

#: A module setting that the first test raises and the second needs at its
#: default: under the measurement helper, which restores the module before
#: each test, both pass. As delivered the second fails.
TARGET = (
    "LIMIT = 0\n\n\n"
    "def classify(n):\n"
    "    if n > LIMIT:\n"
    "        return 'pos'\n"
    "    if n < LIMIT:\n"
    "        return 'neg'\n"
    "    return 'zero'\n"
)
CLEAN_SUITE = (
    "from calc import classify\n\n\n"
    "def test_positive():\n"
    "    assert classify(3) == 'pos'\n"
)


def _project(tmp_path: Path):
    root = tmp_path / "proj"
    root.mkdir(parents=True)
    (root / "calc.py").write_text(TARGET, encoding="utf-8", newline="")
    (root / "pyproject.toml").write_text("", encoding="utf-8")
    return languages.get_backend("python"), str(root / "calc.py"), str(root)


class _Counting:
    """Counts the runs as delivered the real backend makes."""

    def __init__(self, monkeypatch):
        self.delivered = 0
        backend_type = type(languages.get_backend("python"))
        real = backend_type.run_as_delivered

        def run_as_delivered(backend, *args, **kwargs):
            self.delivered += 1
            return real(backend, *args, **kwargs)

        monkeypatch.setattr(backend_type, "run_as_delivered", run_as_delivered)


# ---- the record a phase end writes -----------------------------------------

def test_a_phase_end_records_no_acceptance_as_delivered(tmp_path):
    backend, target, root = _project(tmp_path)
    res = phase_end(
        backend, [CLEAN_SUITE], target_file_path=target, source_dir=root,
        timeout=120, mutation_runner=None, stage_as_delivered=True,
        log=lambda m: None)
    archive = tmp_path / "archive"
    persist_phase_end(
        matrices_dir=archive, target_file=target, phase=res,
        delivery=PhaseEndDelivery(winner_label="merged", delivered_km=None,
            delivered_source=res.merged_source, execution_name="test_calc.py"))
    record = json.loads((archive / "consolidation_record.json").read_text(
        encoding="utf-8"))
    assert "portability" not in record


# ---- the coverage phase's entry ---------------------------------------------

def test_the_coverage_phase_checks_nothing_at_its_entry(tmp_path, monkeypatch):
    """The coverage phase starts from the suite it inherits as it is: no run as
    delivered and no test removed before the loop."""
    work_tests = tmp_path / "work" / "tests"
    work_tests.mkdir(parents=True)
    (tmp_path / "calc.py").write_text(TARGET, encoding="utf-8", newline="")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    baseline = work_tests / "test_calc.py"
    baseline.write_text(CLEAN_SUITE, encoding="utf-8")
    monkeypatch.setattr(orchestrator, "_dump_resolved_preset", lambda ctx: None)
    # The reused starter suites carry no generation records here. The
    # entry, not the reuse check, is under test.
    monkeypatch.setattr(orchestrator, "_guard_generation_reuse", lambda ctx: None)
    monkeypatch.setattr(
        orchestrator, "_restore_starter_baseline",
        lambda ctx: SimpleNamespace(
            baseline_file=str(baseline), baseline_line_coverage=50.0,
            baseline_branch_coverage=50.0, baseline_suite_seconds=0.1,
            selected_tests=1, status="ok"))
    handed = []
    monkeypatch.setattr(orchestrator, "_phase_coverage",
                        lambda ctx, result, baseline: handed.append(baseline))
    counting = _Counting(monkeypatch)
    ctx = orchestrator.TargetPhasesContext(
        log_print=lambda *a, **k: None,
        project_name="p", workspace="rank1_calc",
        top_file={"target_number": 1, "file_path": str(tmp_path / "calc.py"),
                  "key_functions": []},
        canonical_path=tmp_path / "calc.py",
        work_tests=work_tests, results_dir=tmp_path / "results",
        target_logs_dir=tmp_path / "logs",
        reuse=orchestrator.ReuseFlags(starter=True),
    )
    orchestrator.run_target_phases(ctx)
    assert counting.delivered == 0
    assert [entry["content"] for entry in handed] == [CLEAN_SUITE]
    assert baseline.read_text(encoding="utf-8") == CLEAN_SUITE


@pytest.mark.parametrize("recorded", ["these settings", "other settings"])
def test_starter_results_measured_under_other_settings_are_refused_before_the_loop(
        tmp_path, monkeypatch, recorded):
    """A reused starter phase whose delivered mutation results were measured
    under other settings, such as a run another Testuary version wrote, is
    refused before the coverage loop sends a request. Results measured under
    these settings go on to the loop."""
    work_tests = tmp_path / "work" / "tests"
    work_tests.mkdir(parents=True)
    target = tmp_path / "calc.py"
    target.write_text(TARGET, encoding="utf-8", newline="")
    (tmp_path / "pyproject.toml").write_text("", encoding="utf-8")
    baseline = work_tests / "test_calc.py"
    baseline.write_text(CLEAN_SUITE, encoding="utf-8")
    monkeypatch.setattr(orchestrator, "_dump_resolved_preset", lambda ctx: None)
    monkeypatch.setattr(orchestrator, "_guard_generation_reuse", lambda ctx: None)
    monkeypatch.setattr(
        orchestrator, "_restore_starter_baseline",
        lambda ctx: SimpleNamespace(
            baseline_file=str(baseline), baseline_line_coverage=50.0,
            baseline_branch_coverage=50.0, baseline_suite_seconds=0.1,
            selected_tests=1, status="ok"))
    # The starter archive holds a delivered kill matrix the coverage phase
    # end would inherit.
    monkeypatch.setattr(orchestrator, "_delivered_kill_matrix", lambda archive, log: {})
    loops = []
    monkeypatch.setattr(orchestrator, "_phase_coverage",
                        lambda ctx, result, baseline: loops.append(baseline))
    ctx = orchestrator.TargetPhasesContext(
        log_print=lambda *a, **k: None,
        project_name="p", workspace="rank1_calc",
        top_file={"target_number": 1, "file_path": str(target), "key_functions": []},
        canonical_path=target,
        work_tests=work_tests, results_dir=tmp_path / "results",
        target_logs_dir=tmp_path / "logs",
        reuse=orchestrator.ReuseFlags(starter=True),
    )
    mut_cfg = orchestrator.config_module.get_agent_config(
        "test_mutation", parent="test_coverage")
    identity = orchestrator.mutation_agent.mutation_measurement_identity(
        target_file_path=str(target), language="python", execution_name="test_calc.py",
        config=dict(mut_cfg), attribution=orchestrator.measurement.union_attribution(mut_cfg))
    if recorded == "other settings":
        identity["config_sha256"] = "0" * 64
    archive = orchestrator._archive_path(ctx, "starter")
    archive.mkdir(parents=True)
    (archive / "consolidation_record.json").write_text(json.dumps(
        {"execution_name": "test_calc.py", "measurement_identity": identity}), encoding="utf-8")
    if recorded == "these settings":
        orchestrator.run_target_phases(ctx)
        assert len(loops) == 1
    else:
        with pytest.raises(RuntimeError, match="measured for another target source, file name, "
                                               "catalogue or settings"):
            orchestrator.run_target_phases(ctx)
        assert loops == []


# ---- evidence reuse --------------------------------------------------------

def test_an_unchanged_suite_reuses_its_evidence(tmp_path, monkeypatch):
    """The starter suites' phase end chose the suite among suites that pass
    their normal run: the coverage phase end makes no run as delivered and
    reuses the evidence, and its record carries no acceptance."""
    counting = {}

    def start_counting(_ctx):
        counting["runs"] = _Counting(monkeypatch)

    run, work = unchanged._drive_unchanged(
        tmp_path, monkeypatch, before_boundary=start_counting)
    assert counting["runs"].delivered == 0
    assert work == unchanged.NO_WORK
    record = json.loads((run.archive / "consolidation_record.json").read_text(
        encoding="utf-8"))
    assert record["delivered_from"] == "inherited"
    assert "portability" not in record


def _with_an_older_record(ctx):
    record_path = (orchestrator._archive_dir(ctx, "starter")
                   / "consolidation_record.json")
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["portability"] = {"reject_non_portable_tests": True,
                             "report_reverse_order_failures": False,
                             "checks": {}}
    record_path.write_text(json.dumps(record), encoding="utf-8")


def test_an_older_record_of_the_check_does_not_stop_the_reuse(
        tmp_path, monkeypatch):
    """An archive written while phase ends recorded an acceptance as delivered is
    reused like any other."""
    run, work = unchanged._drive_unchanged(
        tmp_path, monkeypatch, before_boundary=_with_an_older_record)
    assert work == unchanged.NO_WORK
    record = json.loads((run.archive / "consolidation_record.json").read_text(
        encoding="utf-8"))
    assert record["delivered_from"] == "inherited"
    assert "portability" not in record
