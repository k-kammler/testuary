"""What a run will do, as displays show it: its files and runs, its three
phases with their states, steps and limits, and Finishing, built
from the run's options and the settings it resolved. The event file holds
it, and the console shows it once."""
from __future__ import annotations

import dataclasses
import json
import shutil
from types import SimpleNamespace

import pytest

import calc_language
import test_run_dirs as run_dir_tests
from testuary import models as model_source
from testuary import orchestrator
from testuary import config
from testuary.agent._base import CodingAgent
from testuary.presentation import Metric, Usage, read_completion
from testuary.presentation.terminal import (
    COMPLETION_FILE, COMPLETION_SCHEMA, FileCompletion, RunCompletion, headline,
    write_completion,
)
from testuary.presentation.events import PlannedLimit, PlannedPhase, PlannedStep, RunPlan
from testuary.settings import resolve_settings


def _args(**values):
    fields = dict(stop_after_phase=None, reuse_starter=False,
                  reuse_coverage=False, reuse_mutation=False, reuse_finishing=False,
                  runs=1, run=None)
    fields.update(values)
    return SimpleNamespace(**fields)


@pytest.fixture
def resolved(tmp_path):
    settings = resolve_settings(base_dir=tmp_path, environment={}, models={}, overrides={
        "coverage": {"goal": "85", "max_requests_per_phase": "30"},
        "mutation": {"max_requests_per_phase": "unlimited"}})
    with config.use_run_settings(settings):
        yield tmp_path


def test_a_plan_has_three_phases_their_steps_and_limits(resolved):
    files = [{"file_path": str(resolved / "src" / "calc" / "ops.py")}]
    plan = orchestrator.run_plan(_args(runs=2), files, resolved / "src")
    assert plan.files == ("calc/ops.py",) and plan.runs == ("r0", "r1")
    assert [(phase.phase, phase.label, phase.state) for phase in plan.phases] == [
        ("starter", "Starter phase", "planned"),
        ("coverage", "Coverage phase", "planned"), ("mutation", "Mutation phase", "planned")]
    loop = plan.phases[1]
    # Each graph mode's step once: Assess serves two modes.
    assert [step.step for step in loop.steps[:loop.end]] == list(dict.fromkeys(
        step for step, _ in CodingAgent.MODE_STEPS.values()))
    assert [step.step for step in loop.steps] == [
        "read", "measure", "revise", "validate", "assess", "consolidation"]
    assert (loop.cycle, loop.end) == (2, 5) and (plan.phases[2].cycle, plan.phases[2].end) == (2, 5)
    # Every phase ends with the same consolidation, a loop's after its cycle.
    assert {(phase.steps[phase.end].step, phase.steps[phase.end].label)
            for phase in plan.phases} == {("consolidation", "Consolidation")}
    # The starter suites' steps run once each, in order: no cycle, and the
    # phase end checks the chosen suite under its delivered name inside its
    # consolidation step, as the loops' phase ends do.
    assert [(step.step, step.label) for step in plan.phases[0].steps] == [
        ("generate", "Generate"), ("measure", "Measure suites"),
        ("consolidation", "Consolidation")]
    assert (plan.phases[0].cycle, plan.phases[0].end) == (2, 2)
    coverage = {limit.name: (limit.value, limit.unit) for limit in plan.phases[1].limits}
    assert coverage == {"goal": (85.0, "%"), "request limit": (30.0, "requests"),
                        "time limit": (None, "s")}
    assert dict((limit.name, limit.value) for limit in plan.phases[2].limits)["request limit"] is None
    assert plan.finishing == PlannedStep("finishing", "Finishing", "planned")
    assert plan.summary() == ("Starter phase, Coverage phase, Mutation phase, "
                              "then Finishing")


def test_a_stop_turns_the_later_phases_off_and_a_reuse_marks_its_phase(resolved):
    stopped = orchestrator.run_plan(_args(stop_after_phase="coverage", runs=3), [], resolved)
    assert [phase.state for phase in stopped.phases] == ["planned", "planned", "off"]
    assert stopped.finishing.state == "off" and stopped.runs == ("r0",)
    # A stop ends a file after its one repetition, which --run names.
    assert orchestrator.run_plan(_args(stop_after_phase="coverage", run=2), [], resolved).runs == ("r2",)
    reused = orchestrator.run_plan(_args(reuse_coverage=True, reuse_finishing=True, run=2), [], resolved)
    assert [phase.state for phase in reused.phases] == ["reused", "reused", "planned"]
    assert reused.finishing.state == "reused" and reused.runs == ("r2",)
    assert reused.summary().endswith("Mutation phase, then Finishing (reused)")


def test_a_mutation_reuse_reuses_the_phases_it_builds_on(resolved):
    # The mutation phase revised the coverage phase's suite, which revised the
    # starter phase's: the run generates nothing, and its plan says so.
    args = _args(reuse_mutation=True)
    plan = orchestrator.run_plan(args, [], resolved)
    assert [phase.state for phase in plan.phases] == ["reused", "reused", "reused"]
    assert plan.summary().endswith("Mutation phase (reused), then Finishing")
    assert orchestrator._generation_model_phases(args) == ()


def test_a_coverage_reuse_reuses_the_starter_phase_too(resolved):
    # The restored coverage file is also the starter phase's baseline: the
    # run generates no starter suites, and its plan says so.
    args = _args(reuse_coverage=True)
    plan = orchestrator.run_plan(args, [], resolved)
    assert [phase.state for phase in plan.phases] == ["reused", "reused", "planned"]
    assert orchestrator._generation_model_phases(args) == ("mutation",)
    assert orchestrator._generation_model_phases(_args(reuse_starter=True, reuse_coverage=True, reuse_mutation=True, reuse_finishing=True)) == ()


def test_a_plan_survives_its_dictionary():
    plan = RunPlan(files=("a.py",), runs=("r0",), phases=(
        PlannedPhase("coverage", "Coverage phase", "planned", (PlannedStep("revise", "Revise"),),
                     (PlannedLimit("goal", 90.0, "%"),)),),
        finishing=PlannedStep("finishing", "Finishing", "off"))
    assert RunPlan.from_dict(json.loads(json.dumps(dataclasses.asdict(plan)))) == plan
    with pytest.raises(ValueError):
        PlannedStep("revise", "Revise", "maybe")


def test_a_phase_has_a_start_a_cycle_and_a_phase_end_and_an_older_plan_no_end():
    steps = tuple(PlannedStep(step, step.title()) for step in (
        "read", "measure", "revise", "validate", "consolidation"))
    phase = PlannedPhase("coverage", "Coverage phase", steps=steps, cycle=2, end=4)
    assert ([s.step for s in phase.steps[:phase.cycle]],
            [s.step for s in phase.steps[phase.cycle:phase.end]],
            [s.step for s in phase.steps[phase.end:]]) == (
        ["read", "measure"], ["revise", "validate"], ["consolidation"])
    plan = RunPlan(files=("a.py",), runs=("r0",), phases=(phase,),
                   finishing=PlannedStep("finishing", "Finishing"))
    written = dataclasses.asdict(plan)
    assert RunPlan.from_dict(json.loads(json.dumps(written))) == plan
    # An event file written before the phase end existed reads with none.
    del written["phases"][0]["end"]
    assert RunPlan.from_dict(written).phases[0].end == len(steps)
    with pytest.raises(ValueError):
        PlannedPhase("coverage", "Coverage phase", steps=steps, cycle=3, end=2)


def test_a_runs_event_file_and_console_hold_its_plan(tmp_path, monkeypatch, capsys):
    """A whole Calc run at the console with an event file: the plan is one
    line in the file, before the run's work, and one line on the console."""
    from testuary import __main__ as cli

    events = tmp_path / "events.jsonl"
    source = run_dir_tests._Recording()
    calc_language.register()
    try:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "calc.calc").write_text(run_dir_tests.calc.SOURCE, encoding="utf-8")
        (tmp_path / "models.ini").write_text(
            "[scripted]\nprovider = ollama\nmodel = scripted\n", encoding="utf-8")
        (project / ".testuary").mkdir()
        (project / ".testuary" / "project.ini").write_text("[project]\nlanguage = calc\n",
                                                           encoding="utf-8")
        monkeypatch.chdir(project)
        monkeypatch.setenv("TESTUARY_HOME", str(tmp_path))
        monkeypatch.setenv("TESTUARY_TIMING_RUN_COUNT", "1")
        monkeypatch.setattr("sys.argv", [
            "testuary", "--model", "scripted", "--source-files", "calc.calc",
            "--stop-after-phase", "starter", "--output-dir", str(tmp_path / "out"),
            "--display", "plain", "--no-color", "--events", str(events)])
        with model_source.installed(source):
            cli.main()
        # The end-of-run summary stays in the run directory, beside summary.json.
        written = tmp_path / "out" / "results" / COMPLETION_FILE
        completion, provenance = read_completion(written)
        assert (provenance, [item.path for item in completion.files]) == ("observed", ["calc.calc"])
        assert completion.results_directory == str(tmp_path / "out" / "results")
    finally:
        calc_language.unregister()
        shutil.rmtree(tmp_path / "out", ignore_errors=True)
    records = [json.loads(line) for line in events.read_text(encoding="utf-8").splitlines()]
    kinds = [next(iter(record)) for record in records]
    assert kinds.count("plan") == 1
    plan = RunPlan.from_dict(records[kinds.index("plan")]["plan"])
    assert plan.files == ("calc.calc",)
    assert [phase.state for phase in plan.phases] == ["planned", "off", "off"]
    # Before the run's work: no phase has started yet.
    assert not any(record.get("event", {}).get("level") == "phase"
                   for record in records[:kinds.index("plan")])
    # The run delivered no complete suite: its end is a result, not a success.
    last = [record["event"] for record in records if record.get("event", {}).get("level") == "run"][-1]
    assert (last["state"], last["failure"]) == ("failed", "outcome")
    assert last["message"] == headline(completion)
    assert last["message"].startswith(
        "Run finished: 0 of 1 target with a suite that meets the minimums")
    # The display writes to standard error.
    shown = capsys.readouterr().err
    assert shown.count("Plan: Starter phase, Coverage phase (off)") == 1


def test_an_end_of_run_summary_that_cannot_be_written_is_logged(monkeypatch):
    """The console's write of completion.json warns in the log when it
    fails, as the orchestrator's own write does, and the summary is still
    shown."""
    from testuary import __main__ as cli
    from testuary.presentation import terminal as module

    def refused(*args, **kwargs):
        raise OSError("the disk is full")

    logged, shown = [], []
    monkeypatch.setattr(module, "write_completion", refused)
    monkeypatch.setattr(module, "present_completion", lambda completion, presenter: shown.append(completion))
    monkeypatch.setattr(cli, "log_print", lambda text, **kwargs: logged.append(text))
    monkeypatch.setattr("sys.argv", ["testuary"])
    cli._present_completion(RunCompletion(files=()), SimpleNamespace(provenance="observed"))
    assert logged == ["WARNING: the end-of-run summary could not be written (OSError: the disk is full)"]
    assert len(shown) == 1


def test_the_end_of_run_summary_is_written_and_read_back_without_keys(tmp_path):
    completion = RunCompletion(files=(FileCompletion(
        "calc/ops.py", "complete", reason="api_key=sk-test-123 was never sent",
        metrics=(Metric("mutation score", 87.5, "%"),), usage=Usage(logical_requests=3)),),
        results_directory=str(tmp_path / "results"), resume_command="testuary --resume")
    path = write_completion(completion, provenance="synthetic")
    assert path == tmp_path / "results" / COMPLETION_FILE
    assert "sk-test-123" not in path.read_text(encoding="utf-8")
    read, provenance = read_completion(path)
    assert provenance == "synthetic"
    assert read.files[0].metrics == completion.files[0].metrics
    assert read.files[0].usage == completion.files[0].usage
    assert read.resume_command == "testuary --resume"
    assert RunCompletion.from_dict(dataclasses.asdict(completion)) == completion
    assert write_completion(RunCompletion(files=())) is None
    (tmp_path / "other.json").write_text('{"schema_version": 9}', encoding="utf-8")
    with pytest.raises(ValueError, match="not an end-of-run summary"):
        read_completion(tmp_path / "other.json")
    (tmp_path / "empty.json").write_text(
        json.dumps({"schema_version": COMPLETION_SCHEMA}), encoding="utf-8")
    with pytest.raises(ValueError, match="not an end-of-run summary"):
        read_completion(tmp_path / "empty.json")
