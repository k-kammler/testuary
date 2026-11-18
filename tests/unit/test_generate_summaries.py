"""Generate's pre-run summary, after admission and before the first model
request, and its completion summary.

Generate passes one RunReadiness to ``present_readiness`` with
``before_run=True`` once admission and preflight passed. Only the phase body
is stubbed: the CLI, settings, admission and pipeline are real.
"""
from __future__ import annotations

import os
import subprocess
import sys

import pytest

import test_generation_reuse_interruptions as harness
from testuary import __main__ as cli, orchestrator
from testuary import exits
from testuary.orchestrator import RunRequest
from testuary import target_phases as phases
from testuary.readiness import pre_run_report
from testuary.settings import resolve_settings


@pytest.fixture(autouse=True)
def no_native_or_provider(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native process"))
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_check_canonical_against_git", lambda *a: [])


def _cli(tmp_path, monkeypatch, *extra, stop=True):
    project, _, _, case = harness.setup_case(tmp_path)
    environment = {name: value for name, value in os.environ.items()
                   if not name.startswith(("TESTUARY_", "LANGADAPTER_"))}

    def resolve(**kwargs):
        return resolve_settings(**kwargs, models=dict(case.models), environment=environment)

    monkeypatch.setattr("testuary.settings.resolve_settings", resolve)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", [
        "testuary", "generate", "--source-files", "module.py", "--model", "explicit",
        "--source-dir", str(project), "--output-dir", str(tmp_path / "output"),
        *(("--stop-after-phase", "starter") if stop else ()), "--display", "plain", "--no-color", *extra])


def test_generate_shows_the_summary_once_after_admission_and_before_generation(tmp_path, monkeypatch, capsys):
    _cli(tmp_path, monkeypatch)
    seen = []

    def body(ctx):
        seen.append(capsys.readouterr())
        return phases.TestPipelineResult(stopped_after_phase="starter")

    monkeypatch.setattr(phases, "run_target_phases", body)
    assert cli.main() == 0
    [before] = seen
    text = before.out + before.err
    assert text.count("Generation starts now") == 1
    assert "module.py" in text
    assert "Generation starts now" not in "".join(capsys.readouterr())


def test_a_refused_run_shows_no_pre_run_summary(tmp_path, monkeypatch, capsys):
    from testuary.settings import ConfigurationError

    def missing_tool(*args):
        raise ConfigurationError("required tool missing")

    _cli(tmp_path, monkeypatch)
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", missing_tool)
    monkeypatch.setattr(phases, "run_target_phases", lambda ctx: pytest.fail("generation after a refusal"))
    with pytest.raises(SystemExit):
        cli.main()
    assert "Generation starts now" not in "".join(capsys.readouterr())


def test_the_report_marks_what_admission_established_and_keeps_run_only_items(tmp_path):
    (tmp_path / "original").mkdir()
    settings = resolve_settings(
        config_files=[], base_dir=tmp_path, environment={},
        overrides={"api": {"model": "chosen"}, "project": {"source_dir": "original", "output_dir": "output"}},
        models={"chosen": {"provider": "openai", "model": "fixture", "api_url": "https://example.test/v1"}})
    report = pre_run_report(RunRequest(("a.py",)), settings)
    items = {item.key: item for item in report.items}
    for key in ("source.targets", "output.admission", "native.execution"):
        assert items[key].status == "verified" and items[key].action is None
    assert items["configuration.credentials"].tried_by == "run"
    assert report.exit_code == 0


def test_generate_ends_with_one_completion_summary(tmp_path, monkeypatch, capsys):
    _cli(tmp_path, monkeypatch, stop=False)

    def body(ctx):
        delivered = ctx.results_dir / ctx.workspace / "r0" / "tests"
        delivered.mkdir(parents=True)
        (delivered / "test_module.py").write_bytes(b"def test_add():\n    assert True\n")
        # A measurement record beside the suite is not part of it.
        (delivered / "kill_matrix.json").write_bytes(b"{}\n")
        return phases.TestPipelineResult()

    monkeypatch.setattr(phases, "run_target_phases", body)
    assert cli.main() == 0
    text = "".join(capsys.readouterr())
    assert text.count("Run finished: 1 of 1 target with a suite that meets the minimums.") == 1
    assert "Results: " in text and "run.log" in text and "Next step" not in text
    [suite] = [line for line in text.splitlines() if line.strip().startswith("Suite: ")]
    assert suite.strip().endswith("/tests/test_module.py")


def test_a_file_without_a_suite_gets_its_reason_and_the_resume_command(tmp_path, monkeypatch, capsys):
    _cli(tmp_path, monkeypatch, stop=False)
    monkeypatch.setattr(phases, "run_target_phases", lambda ctx: phases.TestPipelineResult(phase_outcomes={
        "starter": "failed", "starter_detail": "no green tests from 2 generated suites"}))
    assert cli.main() == 0
    text = "".join(capsys.readouterr())
    assert "0 of 1 target with a suite that meets the minimums, 1 without one." in text
    assert "module.py: no suite that meets the minimums: " in text
    [step] = [line for line in text.splitlines() if line.startswith("Next step: ")]
    assert "testuary generate --source-files module.py" in step and step.split(" continues")[0].endswith("--resume")
    # The completion summary is the only list of unfinished files.
    assert "Targets without a suite that meets the minimums" not in text


def test_usage_keeps_unknown_counts_unknown():
    from testuary.orchestrator import _file_usage

    known = {"starter_logical_requests": "2", "coverage_logical_requests": "3", "mutation_logical_requests": "0",
             "starter_requests": "2", "coverage_requests": "4", "mutation_requests": "0",
             "starter_input_tokens": "10", "coverage_input_tokens": "20", "mutation_input_tokens": "0",
             "starter_output_tokens": "1", "coverage_output_tokens": "2", "mutation_output_tokens": "0",
             "starter_cache_hit_tokens": "0", "coverage_cache_hit_tokens": "5", "mutation_cache_hit_tokens": "0",
             **{f"{phase}_unknown_usage_requests": "0" for phase in ("starter", "coverage", "mutation")},
             **{f"{phase}_unknown_cache_requests": "0" for phase in ("starter", "coverage", "mutation")}}
    usage = _file_usage([known])
    assert (usage.logical_requests, usage.physical_requests, usage.input_tokens,
            usage.output_tokens, usage.cached_input_tokens) == (5, 6, 30, 3, 5)
    unreported = _file_usage([{**known, "coverage_unknown_usage_requests": "1"}])
    assert unreported.input_tokens is None and unreported.output_tokens is None
    assert unreported.logical_requests == 5
    assert _file_usage([{**known, "starter_unknown_cache_requests": "2"}]).cached_input_tokens is None
    assert _file_usage([{**known, "mutation_logical_requests": ""}]).logical_requests is None


@pytest.mark.parametrize("argv, expected", [
    (["generate", "--source-files", "a.py"], ["generate", "--source-files", "a.py", "--resume"]),
    (["generate", "--source-files", "a.py", "--resume"], ["generate", "--source-files", "a.py", "--resume"]),
    (["generate", "--source-files", "a.py", "--reuse", "coverage"], None),
    (["--demo"], None),
])
def test_the_resume_command_continues_the_same_invocation(argv, expected):
    import shlex

    command = cli._resume_command(argv)
    if expected is None:
        assert command is None
    else:
        parts = command.split() if os.name == "nt" else shlex.split(command)
        assert parts == ["testuary", *expected]


def test_a_resumable_runs_completion_carries_its_resume_command(monkeypatch):
    """The completion of a run --resume continues carries the command,
    decided by whether the run can be resumed, not by the wording of the
    interrupt advice."""
    from types import SimpleNamespace

    from testuary.presentation import terminal

    shown = []
    monkeypatch.setattr(sys, "argv", ["testuary", "generate", "--source-files", "a.py"])
    monkeypatch.setattr(terminal, "write_completion", lambda completion, **_: None)
    monkeypatch.setattr(terminal, "present_completion",
                        lambda completion, **_: shown.append(completion))
    cli._present_completion(terminal.RunCompletion(files=()), SimpleNamespace(provenance=None))
    assert shown[0].resume_command is not None
    assert shown[0].resume_command.endswith("--resume")


def test_final_values_come_from_the_single_row_of_the_file(tmp_path):
    import csv

    from testuary.orchestrator import _completion_facts

    results = tmp_path / "results"
    results.mkdir()
    with open(results / "tests.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["target_number", "run", "line_coverage", "branch_coverage",
                                                    "mutation_score"])
        writer.writeheader()
        writer.writerow({"target_number": 1, "run": 0, "line_coverage": 87.5, "branch_coverage": "",
                         "mutation_score": 60})
        for run in (0, 1):
            writer.writerow({"target_number": 2, "run": run, "line_coverage": 50 + run})
    files = [("a.py", {"target_number": 1}, None, "complete", "", False),
             ("b.py", {"target_number": 2}, None, "complete", "", False)]
    facts = _completion_facts(files, results_dir=results, log_path="", runs=(0, 1))
    first, second = facts.files
    assert [(metric.name, metric.value, metric.unit, metric.basis) for metric in first.metrics] == [
        ("Line coverage", 87.5, "%", "measured"), ("Mutation score", 60.0, "%", "measured")]
    # With several runs there is no single final value to show.
    assert second.metrics == ()
    assert facts.log_path is None and facts.results_directory == str(results)
    # One run made alone shows its own row beside the others on disk.
    facts = _completion_facts(files, results_dir=results, log_path="", runs=(1,))
    assert [(metric.name, metric.value) for metric in facts.files[1].metrics] == [("Line coverage", 51.0)]


def test_an_interrupted_run_still_ends_with_its_completion_summary(tmp_path, monkeypatch, capsys):
    _cli(tmp_path, monkeypatch, "--source-files", "module.py,other.py", stop=False)
    (tmp_path / "projects" / "subject" / "other.py").write_bytes(b"def sub(a, b):\n    return a - b\n")
    started = []

    def body(ctx):
        started.append(ctx.top_file["canonical_path"])
        if len(started) == 2:
            raise KeyboardInterrupt
        return phases.TestPipelineResult()

    monkeypatch.setattr(phases, "run_target_phases", body)
    assert cli.main() == exits.INTERRUPTED
    text = "".join(capsys.readouterr())
    assert "Run stopped: 1 of 2 targets with a suite that meets the minimums" in text
    assert "1 stopped before finishing" in text
    assert "other.py: stopped before finishing" in text


@pytest.mark.parametrize("refusal", ["reuse", "strict_git"])
def test_a_later_refusal_comes_before_the_pre_run_summary(tmp_path, monkeypatch, capsys, refusal):
    if refusal == "reuse":
        _cli(tmp_path, monkeypatch)
        monkeypatch.setattr(orchestrator, "_preflight_reuse_requirements",
                            lambda *a: ["rank 1: no saved suite to reuse"])
    else:
        (tmp_path / "strict.ini").write_bytes(b"[main]\nstrict_canonical_git = true\n")
        _cli(tmp_path, monkeypatch, "--config", "strict.ini")
        monkeypatch.setattr(orchestrator, "_check_canonical_against_git",
                            lambda root, files: [(files[0], "1 line changed")])
    monkeypatch.setattr(phases, "run_target_phases", lambda ctx: pytest.fail("generation after a refusal"))
    with pytest.raises(SystemExit) as stopped:
        cli.main()
    assert stopped.value.code == 2
    assert "Generation starts now" not in "".join(capsys.readouterr())
    # A refusal leaves no run directory, so the corrected command runs.
    assert not (tmp_path / "output").exists()
    monkeypatch.setattr(orchestrator, "_preflight_reuse_requirements", lambda *a: [])
    monkeypatch.setattr(orchestrator, "_check_canonical_against_git", lambda *a: [])
    monkeypatch.setattr(phases, "run_target_phases",
                        lambda ctx: phases.TestPipelineResult(stopped_after_phase="starter"))
    assert cli.main() == 0


def test_a_requested_stop_is_neither_complete_nor_offered_a_resume(tmp_path, monkeypatch, capsys):
    _cli(tmp_path, monkeypatch)
    monkeypatch.setattr(phases, "run_target_phases",
                        lambda ctx: phases.TestPipelineResult(stopped_after_phase="starter"))
    assert cli.main() == 0
    text = "".join(capsys.readouterr())
    assert ("0 of 1 target with a suite that meets the minimums" in text
            and "1 ended after the requested phase" in text)
    assert "module.py: ended after the requested phase: stopped after starter, as requested" in text
    assert "Suite: " not in text and "Next step" not in text


def test_a_suite_below_its_floor_is_a_result_without_a_resume_offer(tmp_path, monkeypatch, capsys):
    # Resume skips such a file, so it is not offered.
    _cli(tmp_path, monkeypatch, stop=False)
    monkeypatch.setattr(phases, "run_target_phases", lambda ctx: phases.TestPipelineResult(
        below_minimum_reason="line coverage 40.0% is below the floor of 50%"))
    assert cli.main() == 0
    text = "".join(capsys.readouterr())
    assert "module.py: no suite that meets the minimums: line coverage 40.0% is below the floor of 50%" in text
    assert "Next step" not in text


def test_a_resume_skip_reads_the_floor_from_its_own_recorded_field():
    # The floor reason comes from below_minimum_reason, not from any recorded
    # error such as the coverage phase's stagnation.
    from testuary.orchestrator import _recorded_floor_reason

    floor = "line coverage 40.0% is below the floor of 50%"
    assert _recorded_floor_reason([{"below_minimum_reason": floor, "error": floor}]) == floor
    assert _recorded_floor_reason([{"below_minimum_reason": "", "error": "coverage_stagnation",
                                    "coverage_status": "success"}]) == ""
    assert _recorded_floor_reason([]) == ""


def test_the_completion_lists_the_time_limits_each_run_reached(tmp_path):
    from testuary import results as _time_limits
    from testuary.orchestrator import _completion_facts

    results = tmp_path / "results"
    results.mkdir()
    with _time_limits.collecting(results / "mod" / "r1"):
        _time_limits.note("mutation", "mutation.measurement_timeout_seconds", 600)
    files = [("mod.py", {"target_number": 1}, "mod", "complete", "", False)]
    text = ("The mutation measurement reached its time limit "
            "(mutation.measurement_timeout_seconds = 600 s). "
            "Untested mutants count as not killed.")
    several = _completion_facts(files, results_dir=results, log_path="", runs=(0, 1))
    assert several.files[0].warnings == ("r1: " + text,)
    alone = _completion_facts(files, results_dir=results, log_path="", runs=(1,))
    assert alone.files[0].warnings == (text,)
    assert _completion_facts(files, results_dir=results, log_path="", runs=(0,)).files[0].warnings == ()


def test_the_completion_names_each_loop_a_limit_stopped_before_its_goal(tmp_path):
    """A loop that ended at one of its limits gets a warning, as a time
    limit does, from the stop its phase record holds: which limit and its
    setting when the record names one, as the stop names the setting. A
    loop that reached its goal or stopped without progress gets none."""
    import json

    from testuary import phase_state
    from testuary.orchestrator import _completion_facts

    results = tmp_path / "results"
    (results / "mod").mkdir(parents=True)
    ended = "Budget exhausted: UsageLimitExceeded: "
    state = {
        "coverage": {"0": {"metrics": {"status": "success"}},
                     "1": {"metrics": {"status": "budget_exhausted",
                                       "error": ended + "max_output_tokens_per_phase (200000) exceeded "
                                                        "(used 200100)"}}},
        "mutation": {"0": {"metrics": {"status": "budget_exhausted",
                                       "error": ended + "max_requests_per_phase (20) exceeded"}},
                     "1": {"metrics": {"status": "budget_exhausted",
                                       "error": ended + "max_wall_clock_seconds_per_phase (1800s) exceeded "
                                                        "(elapsed 1801s)"}},
                     "2": {"metrics": {"status": "budget_exhausted", "error": "Budget exhausted"}},
                     "3": {"metrics": {"status": "stagnation"}}},
    }
    phase_state.state_path(results / "mod").write_text(json.dumps(state), encoding="utf-8")
    files = [("mod.py", {"target_number": 1}, "mod", "complete", "", False)]
    alone = _completion_facts(files, results_dir=results, log_path="", runs=(0,))
    assert alone.files[0].warnings == (
        "The mutation phase stopped at its request limit before reaching its goal "
        "(mutation.max_requests_per_phase = 20).",)
    several = _completion_facts(files, results_dir=results, log_path="", runs=(1, 2, 3))
    assert several.files[0].warnings == (
        "r1: The coverage phase stopped at its output token limit before reaching its goal "
        "(coverage.max_output_tokens_per_phase = 200000).",
        "r1: The mutation phase stopped at its time limit before reaching its goal "
        "(mutation.max_wall_clock_seconds_per_phase = 1800 s).",
        "r2: The mutation phase stopped at one of its limits before reaching its goal.")
    # A file that never ran has no record and no warning.
    stopped = [("mod.py", {"target_number": 1}, None, "stopped", "", True)]
    assert _completion_facts(stopped, results_dir=results, log_path="", runs=(0,)).files[0].warnings == ()

