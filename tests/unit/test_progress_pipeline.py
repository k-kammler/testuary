"""Run-level progress through the pipeline loop counts only work done.

A file without a suite, a phase stop and a reuse of every phase do not
end at 100%: finished files are not topped up and plans count only work the
run does. Only the phase body is stubbed. The file loop, plan and events are
real.
"""
from __future__ import annotations

import subprocess

import pytest

import test_generation_reuse_interruptions as harness
from testuary import orchestrator
from testuary.orchestrator import RunRequest
from testuary import target_phases as phases


@pytest.fixture(autouse=True)
def no_native_or_provider(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native process"))
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_check_canonical_against_git", lambda *a: [])


def _run(tmp_path, monkeypatch, request, body):
    _, _, _, settings = harness.setup_case(tmp_path)
    events = []
    monkeypatch.setattr(phases, "run_target_phases", body)
    status = orchestrator.run_pipeline(request, settings=settings, event_sink=events.append)
    done = sum(1 for event in events if event.level == "phase" and (
        (event.operation.startswith("phase/") and event.state == "succeeded")
        or event.operation.startswith("restore/")))
    run = [event for event in events if event.level == "run"][-1]
    files = [event for event in events if event.level == "target"]
    return status, run, files, done


def _phase(ctx, name, state):
    phases._display_phase(ctx, name, "running")
    phases._display_phase(ctx, name, state)


def test_a_file_without_a_suite_ends_short_and_says_why(tmp_path, monkeypatch):
    def body(ctx):
        _phase(ctx, "starter", "failed")
        return phases.TestPipelineResult(phase_outcomes={
            "starter": "failed", "starter_detail": "no green tests from 2 generated suites"})

    status, run, files, done = _run(tmp_path, monkeypatch, RunRequest(("module.py",)), body)
    assert status == 0
    # Four phases, Finishing the fourth.
    assert (run.completed, run.total) == (done, 4) == (0, 4)
    assert files[-1].state == "failed" and files[-1].message.startswith(
        "No suite that meets the minimums: ")
    assert files[-1].failure == "outcome"
    assert "no tests from the coverage phase" in files[-1].message


def test_a_file_whose_replies_hit_the_output_limit_says_so(tmp_path, monkeypatch):
    reason = ("the model used the whole output limit (max_output_tokens = 8192) without "
              "answering in 3 of 3 replies; raise max_output_tokens; no green tests from 0 "
              "generated suites (syntax 0, collection 0, unsalvageable 0)")

    def body(ctx):
        _phase(ctx, "starter", "failed")
        return phases.TestPipelineResult(phase_outcomes={
            "starter": "failed", "starter_detail": reason})

    status, run, files, done = _run(tmp_path, monkeypatch, RunRequest(("module.py",)), body)
    assert files[-1].state == "failed"
    assert reason in files[-1].message and "no tests from the coverage phase" not in files[-1].message


def test_a_phase_stop_plans_and_counts_one_repetition(tmp_path, monkeypatch):
    def body(ctx):
        _phase(ctx, "starter", "succeeded")
        _phase(ctx, "coverage", "succeeded")
        return phases.TestPipelineResult(stopped_after_phase="coverage")

    # A stop ends a file after its first run, so several runs are refused.
    with pytest.raises(ValueError, match="one repetition"):
        RunRequest(("module.py",), runs=3, stop_after_phase="coverage")
    request = RunRequest(("module.py",), stop_after_phase="coverage")
    status, run, files, done = _run(tmp_path, monkeypatch, request, body)
    assert status == 0
    assert (run.completed, run.total) == (done, 2) == (2, 2)
    assert files[-1].state == "succeeded"


def test_a_reuse_of_every_phase_counts_four_and_a_failed_file_is_not_topped_up(
        tmp_path, monkeypatch):
    # No saved suite exists, so the file fails before its phases; the plan and
    # the count are the subject.
    monkeypatch.setattr(orchestrator, "_preflight_reuse_requirements", lambda *a: [])
    monkeypatch.setattr(orchestrator, "_preflight_generation_reuse", lambda *a: None)

    request = RunRequest(("module.py",), reuse=("starter", "coverage", "mutation", "finishing"))
    status, run, files, done = _run(tmp_path, monkeypatch, request,
                                    lambda ctx: pytest.fail("phases without a saved suite"))
    assert status == 1
    # The reused phases count as Finishing does, so the file run plans four.
    assert (run.completed, run.total) == (done, 4) == (0, 4)
    assert files[-1].state == "failed" and files[-1].message.startswith("Target failed to execute: ")
    assert files[-1].failure == "execution"


def test_a_reuse_without_its_saved_work_is_refused_as_a_setting_is(tmp_path, monkeypatch):
    """A reuse whose saved work is missing refuses the run before any
    target, as the other refusals before the run do: a program receives
    ConfigurationError with every problem, not SystemExit."""
    from testuary.settings import ConfigurationError

    _, _, _, settings = harness.setup_case(tmp_path)
    monkeypatch.setattr(orchestrator, "_preflight_reuse_requirements",
                        lambda *a: ["rank 1: no saved suite to reuse", "rank 2: no saved suite to reuse"])
    # The starter's generation-reuse check reads the prompts through the
    # language, a native process, and is another refusal's test.
    monkeypatch.setattr(orchestrator, "_preflight_generation_reuse", lambda *a: None)
    monkeypatch.setattr(phases, "run_target_phases", lambda ctx: pytest.fail("generation after a refusal"))
    with pytest.raises(ConfigurationError, match="rank 1: no saved suite to reuse") as refused:
        orchestrator.run_pipeline(RunRequest(("module.py",), reuse=("starter",)), settings=settings)
    assert "rank 2: no saved suite to reuse" in str(refused.value)


def test_a_target_whose_location_raises_is_recorded_and_the_run_ends(tmp_path, monkeypatch):
    """An unexpected error while the loop places a target is recorded among
    the unfinished files, and the run ends with a failed target, also when
    placing the target is what raised."""
    _, _, _, settings = harness.setup_case(tmp_path)

    def locate(self, top_file):
        raise RuntimeError("cannot place the target")

    monkeypatch.setattr(orchestrator._Invocation, "locate", locate)
    monkeypatch.setattr(phases, "run_target_phases", lambda ctx: pytest.fail("a target ran"))
    assert orchestrator.run_pipeline(RunRequest(("module.py",)), settings=settings) == 1


def test_a_refused_second_run_keeps_the_running_runs_log():
    """A second run_pipeline while a run is active is refused, and it leaves
    the running run's log open."""
    import logging
    from types import SimpleNamespace

    from testuary import _progress, config
    from testuary.settings import RunSettings

    handler = logging.NullHandler()
    _progress._RUN_LOG.addHandler(handler)
    try:
        with config.use_run_settings(RunSettings()):
            with pytest.raises(RuntimeError, match="already active"):
                orchestrator.run_pipeline(SimpleNamespace(), settings=RunSettings())
        assert handler in _progress._RUN_LOG.handlers
    finally:
        _progress._RUN_LOG.removeHandler(handler)


def test_a_target_is_named_by_its_source_path_after_its_run_moved_to_the_work_copy(tmp_path):
    """A file run points the target's file_path at its work copy. The label
    the end-of-run summary and an interrupt give the file stays its path in
    the source directory, as before the run."""
    from types import SimpleNamespace

    source = tmp_path / "project"
    (source / "pkg").mkdir(parents=True)
    (source / "pkg" / "mod.py").write_text("x = 1\n", encoding="utf-8")
    invocation = orchestrator._Invocation(
        SimpleNamespace(), project_name="project", projects_root=source.resolve(), container_run=False,
        results_dir=tmp_path / "results", logs_dir=tmp_path / "logs", work_root=tmp_path / "work",
        file_count=1, reused=set())
    top_file = {"target_number": 1, "file_path": str(source / "pkg" / "mod.py")}
    assert invocation.display_file(top_file) == "pkg/mod.py"
    top_file.update(canonical_path=top_file["file_path"],
                    file_path=str(tmp_path / "work" / "rank1" / "project" / "pkg" / "mod.py"))
    assert invocation.display_file(top_file) == "pkg/mod.py"


def test_a_container_source_dir_reaches_the_target_selection_normalized(tmp_path, monkeypatch):
    """A container's source directory is a POSIX path the run keeps unresolved.
    The run normalizes it once where it builds it, so the target guard and
    the file runs compare against one root, also when the setting holds
    ".."."""
    from pathlib import PurePosixPath

    _, _, _, settings = harness.setup_case(tmp_path)
    monkeypatch.setattr(orchestrator.config_module, "get_project_config", lambda *a, **k: {
        "project_name": "repo", "source_dir": "/work/../repo", "target_repo": "/work/../repo",
        "hint_profile": "", "source_dir_override": ""})
    monkeypatch.setattr(orchestrator.config_module, "project_exec_mode", lambda: "container")
    seen = []

    def select(args, *, projects_root):
        seen.append(projects_root)
        raise RuntimeError("stop after the root")

    monkeypatch.setattr(orchestrator, "_source_file_top_files", select)
    with pytest.raises(RuntimeError, match="stop after the root"):
        orchestrator.run_pipeline(RunRequest(("module.py",)), settings=settings)
    assert seen == [PurePosixPath("/repo")]


def test_a_container_target_climbing_out_of_the_source_dir_is_refused(monkeypatch):
    """A container run keeps its paths POSIX and unresolved, so a relative
    name with ".." that leaves the source directory is refused as on the host."""
    from types import SimpleNamespace

    from testuary.settings import ConfigurationError

    monkeypatch.setattr(orchestrator.config_module, "project_exec_mode", lambda: "container")
    with pytest.raises(ConfigurationError, match="lies outside the source directory"):
        orchestrator._source_file_top_files(SimpleNamespace(source_files="../elsewhere.py"),
                                            projects_root="/work/project")


@pytest.mark.parametrize("resume", [False, True])
def test_a_target_outside_the_source_dir_is_refused_before_the_run(tmp_path, monkeypatch, resume):
    """A target outside the source directory has no place in the run's copy of
    the project: refused before any target, with or without a resume."""
    from testuary.settings import ConfigurationError

    _, _, _, settings = harness.setup_case(tmp_path)
    outside = tmp_path / "elsewhere.py"
    outside.write_text("def g():\n    return 2\n", encoding="utf-8")
    monkeypatch.setattr(phases, "run_target_phases", lambda ctx: pytest.fail("a target after a refusal"))
    with pytest.raises(ConfigurationError, match="lies outside the source directory"):
        orchestrator.run_pipeline(RunRequest((str(outside),), resume=resume), settings=settings)


def test_a_run_the_guard_saw_crash_closes_its_pipeline_as_crashed(tmp_path, monkeypatch):
    """A crash the pipeline's guard catches discards the run's work, so its
    pipeline segment closes as crashed and the run reports no time of its
    own, never the discarded work's."""
    import json

    from testuary import phase_state

    def body(ctx):
        _phase(ctx, "starter", "running")
        raise RuntimeError("the engine died")

    _, _, _, settings = harness.setup_case(tmp_path)
    monkeypatch.setattr(phases, "run_target_phases", body)
    orchestrator.run_pipeline(RunRequest(("module.py",)), settings=settings)
    states = [json.loads(path.read_text(encoding="utf-8"))
              for path in (tmp_path / "output").rglob("phase_state.json")]
    (pipeline,) = [state[phase_state.SEGMENT_PIPELINE]["0"]["segments"]
                   for state in states if phase_state.SEGMENT_PIPELINE in state]
    assert [segment["outcome"] for segment in pipeline] == [phase_state.SEGMENT_CRASHED]


def test_the_run_log_holds_the_log_lines_and_the_run_events_name_it(tmp_path, monkeypatch):
    from testuary import _progress

    def body(ctx):
        _phase(ctx, "starter", "succeeded")
        return phases.TestPipelineResult(stopped_after_phase="starter")

    _, _, _, settings = harness.setup_case(tmp_path)
    events = []
    monkeypatch.setattr(phases, "run_target_phases", body)
    request = RunRequest(("module.py",), stop_after_phase="starter")
    assert orchestrator.run_pipeline(request, settings=settings, event_sink=events.append) == 0
    run_events = [event for event in events if event.level == "run"]
    opened = next(index for index, event in enumerate(run_events) if event.log_path)
    # From the moment the log exists, every run event names it, before any file starts.
    assert all(event.log_path == run_events[opened].log_path for event in run_events[opened:])
    assert events.index(run_events[opened]) < next(i for i, e in enumerate(events) if e.level == "target")
    log = run_events[opened].log_path
    assert log.endswith("run.log") and _progress.plain_log_path() == ""
    text = open(log, encoding="utf-8").read()
    assert "testuary results version" in text and "PHASE STOP" in text


def test_the_run_log_keeps_the_lines_written_before_it_opens(tmp_path, monkeypatch):
    # Console lines from before the run directory is known, such as the
    # work directory's namespace, also reach run.log.
    def body(ctx):
        _phase(ctx, "starter", "succeeded")
        return phases.TestPipelineResult(stopped_after_phase="starter")

    _, _, _, settings = harness.setup_case(tmp_path)
    events = []
    monkeypatch.setattr(phases, "run_target_phases", body)
    monkeypatch.setattr(orchestrator.project, "resolve_work_dir_namespace", lambda: "job_fixture")
    request = RunRequest(("module.py",), stop_after_phase="starter")
    assert orchestrator.run_pipeline(request, settings=settings, event_sink=events.append) == 0
    log = next(event.log_path for event in events if event.log_path)
    lines = [line.split(" ", 1)[1] for line in open(log, encoding="utf-8").read().splitlines()]
    assert lines.count("  [main] work_dir namespace: job_fixture") == 1
    assert lines.index("  [main] work_dir namespace: job_fixture") < next(
        index for index, line in enumerate(lines) if "testuary results version" in line)
