"""The command's exit status: 1 when a target failed to execute, 0 when
every target ended in a result or a resume skip. The demonstration runs the
same pipeline and reports its failure as such."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from testuary import __main__ as cli
from testuary import orchestrator
from testuary import target_phases as phases
from testuary.demo import cli as demo_cli
from testuary.demo.scenario import Scenario


def _entry(target, reason, failed):
    return {"target_number": target, "file_path": f"/p/calc_{target}.py",
            "reason": reason, "failed": failed}


def _summary(entries):
    lines = []
    status = orchestrator._summarize_unfinished_targets(
        entries, lambda message, level=0: lines.append(message))
    return status, lines


def test_a_failure_to_execute_sets_the_exit_status():
    status, lines = _summary([_entry(1, "already complete (resume skip)", False),
                              _entry(2, "test pipeline crashed", True)])
    assert status == 1
    assert any(line.endswith(" FAILED: test pipeline crashed")
               for line in lines)
    assert any(line.endswith(" already complete (resume skip)")
               for line in lines)
    assert lines[-1] == ("1 of them failed to execute, so the command exits "
                         "with status 1.")


def test_results_and_resume_skips_leave_the_exit_status_at_zero():
    status, lines = _summary([
        _entry(1, "already complete (resume skip)", False),
        _entry(2, "coverage 41.0% below the minimum 50.0%", False),
        _entry(3, "no tests from the coverage phase", False),
    ])
    assert status == 0
    assert not any("FAILED" in line or "failed to execute" in line
                   for line in lines)
    assert _summary([]) == (0, [])


def test_a_target_with_several_reasons_is_listed_and_counted_once():
    # A refused reuse that also ended below its floor: one target, one
    # failure, as its completion record says.
    status, lines = _summary([
        _entry(1, "mutation reuse refused: the record does not bind the suite", True),
        _entry(1, "coverage 62.5% is below minimum 75%", False),
        _entry(2, "coverage 41.0% below the minimum 50.0%", False),
    ])
    assert status == 1
    assert "Targets without a suite that meets the minimums (2):" in lines
    listed = [line for line in lines if line.startswith("  target ")]
    assert len(listed) == 2
    assert listed[0].endswith(" FAILED: mutation reuse refused: the record does not bind "
                              "the suite; coverage 62.5% is below minimum 75%")
    assert lines[-2:] == [
        "1 ended without a suite that meets the minimums (a result, not a failure).",
        "1 of them failed to execute, so the command exits with status 1."]


def test_the_module_command_exits_with_the_pipeline_status(tmp_path):
    """``python -m testuary`` exits with the pipeline's status, as the
    installed script does."""
    probe = (
        "import runpy, sys\n"
        "from testuary import orchestrator\n"
        "orchestrator.run_pipeline = lambda *args, **kwargs: 1\n"
        "sys.argv = ['testuary', 'generate', '--source-files', 'module.py']\n"
        "runpy.run_module('testuary', run_name='__main__', alter_sys=True)\n"
    )
    # The suite's speed setting is not a valid user setting.
    environment = dict(os.environ)
    environment.pop("TESTUARY_TIMING_RUN_COUNT", None)
    proc = subprocess.run([sys.executable, "-c", probe], cwd=tmp_path, env=environment,
                          capture_output=True, text=True, timeout=300)
    assert proc.returncode == 1, proc.stdout[-2000:] + proc.stderr[-2000:]


def test_the_playback_raises_a_failed_run_before_it_checks_the_scenario(
        monkeypatch):
    """A failed run leaves scenario entries unused and is reported as a
    failure, not as a stale scenario."""
    monkeypatch.setattr(cli, "run_command", lambda: 1)
    with pytest.raises(demo_cli.PipelineFailure) as failure:
        demo_cli.play(Scenario(entries=[]), ["testuary"])
    assert failure.value.status == 1


def test_a_failed_demonstrated_run_exits_with_its_status(
        tmp_path, monkeypatch, capsys):
    def failed(*_args, **_kwargs):
        raise demo_cli.PipelineFailure(1)

    monkeypatch.setattr(demo_cli, "phase_modules_loaded", lambda: [])
    monkeypatch.setattr(demo_cli, "play", failed)
    # The demonstration clears these, the monkeypatch restores them.
    for name in list(os.environ):
        if name.startswith(("TESTUARY_", "LANGADAPTER_")
                           + demo_cli._TRACING_PREFIXES):
            monkeypatch.delenv(name)
    # A run reads the file this variable names, so it must exist. The demo
    # reads only its playback length from it.
    named = tmp_path / "named.ini"
    named.write_text("[main]\n", encoding="utf-8")
    monkeypatch.setenv("TESTUARY_CONFIG_FILE", str(named))
    monkeypatch.chdir(tmp_path)
    outdir = tmp_path / "out"
    parser = cli._build_parser()
    args = parser.parse_args(["--demo", "--output-dir", str(outdir)])
    assert demo_cli.main(args, parser) == 1
    out = capsys.readouterr().out
    assert "Demonstration failed: the pipeline exited with status 1." in out
    assert "Demo complete" not in out
    assert "Demonstration stopped" not in out
    record = json.loads((outdir / "DEMO.json").read_text(encoding="utf-8"))
    assert record["result"]["outcome"] == "failed"
    assert "stale" not in record["result"]


def test_runs_without_a_complete_suite_are_counted_apart_from_failures():
    status, lines = _summary([
        _entry(1, "no tests from the coverage phase", False),
        _entry(2, "coverage 41.0% below the minimum 50.0%", False),
        _entry(3, "already complete (resume skip)", False),
        _entry(4, "test pipeline crashed", True),
    ])
    assert status == 1
    assert ("2 ended without a suite that meets the minimums (a result, not a failure); "
            "1 already complete (resume skip).") in lines
    assert lines[-1] == "1 of them failed to execute, so the command exits with status 1."


def test_a_crash_is_recorded_as_its_phases_infrastructure_failure(tmp_path, default_run):
    """A phase that raises is recorded as its infrastructure failure with
    the error, in phase_state and as status crashed in its tests.csv
    column. Earlier phases' columns keep what they recorded."""
    import csv
    from types import SimpleNamespace

    from testuary import phase_state
    from testuary import results as csv_writer

    target = tmp_path / "calc.py"
    target.write_text("x = 1" + chr(10), encoding="utf-8")
    results = tmp_path / "results"
    target_dir = results / "calc"
    target_dir.mkdir(parents=True)
    csv_writer.write_tests_row(
        results_dir=results, workspace="rank1_csv", target_number=1, file_path=str(target),
        test_result=SimpleNamespace(status="ok", error=""))
    phase_state.mark_started(target_dir, "mutation", target_number=1, stem="calc",
                             target_file_path="calc.py", canonical_source_path=target)

    def crash(_ctx):
        raise ValueError("the evidence disagrees")

    assert orchestrator._run_target_phases_guarded(
        None, run_phase=crash, results_dir=results, workspace="calc", target_number=1,
        file_path=str(target), run_idx=0,
        log=lambda message, level=0: None) is None
    block = phase_state.load(target_dir)["mutation"]["0"]
    expected = ("the mutation phase crashed (infrastructure failure): "
                "ValueError: the evidence disagrees")
    assert block["status"] == phase_state.FAILED
    assert block["failure_kind"] == phase_state.INFRASTRUCTURE_FAILURE
    assert block["reason"] == expected
    with (results / "tests.csv").open(encoding="utf-8-sig", newline="") as handle:
        row, = list(csv.DictReader(handle))
    assert row["mutation_status"] == "crashed"
    assert row["coverage_status"] == "ok"
    assert row["error"] == expected


class _ProviderError(Exception):
    """A provider's refusal that echoes the key, as the raw cause a request
    error keeps."""


@pytest.mark.parametrize("reached", ["itself", "context", "cause"])
@pytest.mark.parametrize("raised", ["terminal", "exhausted"])
def test_a_crashed_request_error_leaves_its_cause_out_of_the_logs(tmp_path, monkeypatch, capsys,
                                                                  in_run, raised, reached):
    """A phase that crashes with a request error whose cause echoes the key
    leaves the key out of run.log, main.jsonl and the console: the crash
    log names the redacted error, not its raw cause. So does a crash whose
    chain holds the request error, as an error raised while it was handled
    or as the cause of another."""
    from testuary import _progress, model_requests, phase_state
    from testuary.agent._helpers import AgentLogger

    sentinel = "sk-crash-sentinel"
    in_run(overrides={"api": {"provider": "ollama", "model": "test-model"}},
           environment={"TESTUARY_OLLAMA_API_KEY": sentinel})
    cause = _ProviderError(f"invalid key {sentinel}")
    error = (model_requests.TerminalRequestError(cause) if raised == "terminal" else
             model_requests.TransportExhausted("10 consecutive transient failures", cause, 10))
    assert sentinel not in str(error)
    target = tmp_path / "calc.py"
    target.write_text("x = 1" + chr(10), encoding="utf-8")
    results = tmp_path / "results"
    (results / "calc").mkdir(parents=True)
    phase_state.mark_started(results / "calc", "coverage", target_number=1, stem="calc",
                             target_file_path="calc.py", canonical_source_path=target)
    logger = AgentLogger("main", "pipeline")
    logger.enable_streaming(str(tmp_path), filename_template="main.jsonl")
    monkeypatch.setattr(orchestrator._log, "logger", logger)
    _progress.begin_plain_log()
    _progress.open_plain_log(tmp_path / "run.log")

    outer = {"itself": None, "context": OSError("the record directory was removed"),
             "cause": RuntimeError("the phase stopped")}[reached]

    def crash(_ctx):
        if outer is None:
            raise error
        try:
            raise error
        except type(error) as handled:
            if reached == "context":
                raise outer
            raise outer from handled

    try:
        assert orchestrator._run_target_phases_guarded(
            None, run_phase=crash, results_dir=results, workspace="calc", target_number=1,
            file_path=str(target), run_idx=0, log=orchestrator.log_print) is None
    finally:
        _progress.close_plain_log()
    run_log = (tmp_path / "run.log").read_text(encoding="utf-8")
    main_log = (tmp_path / "main.jsonl").read_text(encoding="utf-8")
    shown = capsys.readouterr()
    assert "crashed unexpectedly" in run_log and "crashed unexpectedly" in main_log
    assert not [text for text in (run_log, main_log, shown.out, shown.err) if sentinel in text]
    # The chain up to the request error stays in the log.
    assert type(error).__name__ in run_log
    assert {"itself": "Traceback", "context": "During handling",
            "cause": "direct cause"}[reached] in run_log


def test_a_crash_row_counts_nothing_no_phase_recorded(tmp_path, default_run):
    """A crash before any row: the crash row's counts stay empty, never 0,
    and the half-written suite in the work directory is not counted."""
    import csv

    target = tmp_path / "calc.py"
    target.write_text("x = 1" + chr(10), encoding="utf-8")
    work = tmp_path / "work"
    work.mkdir()
    (work / "test_calc.py").write_text("def test_half():" + chr(10) + "    assert 1" + chr(10),
                                       encoding="utf-8")
    results = tmp_path / "results"
    (results / "calc").mkdir(parents=True)

    def crash(_ctx):
        raise ValueError("the evidence disagrees")

    assert orchestrator._run_target_phases_guarded(
        None, run_phase=crash, results_dir=results, workspace="calc", target_number=1,
        file_path=str(target), run_idx=0,
        log=lambda message, level=0: None) is None
    with (results / "tests.csv").open(encoding="utf-8-sig", newline="") as handle:
        row, = list(csv.DictReader(handle))
    for column in ("coverage_requests", "coverage_input_tokens", "coverage_turns",
                   "coverage_delivered_tests", "score", "coverage_status",
                   "test_count_delivered", "test_loc_delivered"):
        assert row[column] == "", column


def test_a_crash_row_carries_the_phases_completed_before_it(tmp_path, monkeypatch):
    """A crash in finishing: the row holds what the starter, coverage and
    mutation phases recorded at their ends, as a finished run's row does,
    so no reader needs phase_state.json for them, and finishing reads
    crashed."""
    import test_run_dirs as calc_runs
    from testuary import results as csv_writer
    from testuary import target_phases

    for name in ("finished", "crashed"):
        (tmp_path / name).mkdir()
    _, finished = calc_runs._run(tmp_path / "finished", monkeypatch)

    def crash(ctx, result):
        raise ValueError("the evidence disagrees")

    monkeypatch.setattr(target_phases, "_phase_finishing", crash)
    _, crashed = calc_runs._run(tmp_path / "crashed", monkeypatch, exit_code=1)
    (expected,), (row,) = calc_runs._rows(finished), calc_runs._rows(crashed)
    assert row["finishing_status"] == "crashed"
    assert row["error"].startswith("the finishing phase crashed (infrastructure failure)")
    # Finishing writes the phases' snapshot measures, the delivered files'
    # structure and the final mutation score.
    finishing = set(csv_writer._phase_finishing_columns()) | {
        "mutation_assertion_count", "mutation_parametrize_count", "mutation_fixture_count",
        "mutation_score"}
    # The consolidation record holds 0 where a finished row leaves an
    # unmeasured merged count empty; a resume shows the same 0.
    finishing.add("mutation_merged_tests")
    columns = [column for column in row if column.startswith(("starter_", "coverage_", "mutation_"))
               and not column.endswith("duration_s") and column not in finishing]
    assert row["coverage_requests"] and row["mutation_requests"] and row["starter_suites_generated"]
    assert {column: row[column] for column in columns} == {
        column: expected[column] for column in columns}



def _run_file_with_refused_reuse(tmp_path, monkeypatch, phase, reused):
    """One file's run whose phase sequence reports ``phase``'s reuse refused,
    as the restore does: the invocation and the file's tests.csv row."""
    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    target = project / "calc.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    reason = f"the restored {phase} suite does not match its validation record: changed"
    monkeypatch.setattr(orchestrator, "_restore_preserved_tests", lambda **_k: True)
    monkeypatch.setattr(orchestrator, "_run_target_phases_guarded", lambda _ctx, **_k: phases.TestPipelineResult(
        phase_outcomes={phase: "failed", f"{phase}_detail": reason}))
    output = tmp_path / "output"
    invocation = orchestrator._Invocation(
        SimpleNamespace(resume=False, runs=1, run=None, stop_after_phase=None),
        project_name="project", projects_root=project.resolve(), container_run=False,
        results_dir=output / "results", logs_dir=output / "logs", work_root=output / "_work",
        file_count=1, reused=frozenset(reused))
    orchestrator._FileRun(invocation, {"target_number": 1, "file_path": str(target)}).execute()
    with open(invocation.results_dir / "tests.csv", newline="", encoding="utf-8-sig") as handle:
        [row] = list(csv.DictReader(handle))
    return invocation, row, reason


@pytest.mark.parametrize("phase, reused", [
    ("mutation", {"mutation"}),
    ("coverage", {"starter", "coverage"}),
])
def test_a_refused_reuse_fails_the_file_and_its_row_names_the_refusal(
        tmp_path, monkeypatch, default_run, phase, reused):
    """A reuse the phase refused (its restored suite missing, not collecting,
    failing as delivered or unlike its record) is a failure to execute what
    was asked, not a result: the command exits 1, and the row reads
    reuse_refused with the reason."""
    invocation, row, reason = _run_file_with_refused_reuse(tmp_path, monkeypatch, phase, reused)

    [entry] = invocation.unfinished
    assert entry["failed"] and entry["reason"] == f"{phase} reuse refused: {reason}"
    assert orchestrator._summarize_unfinished_targets(invocation.unfinished, lambda *a, **k: None) == 1
    assert row[f"{phase}_status"] == "reuse_refused"
    assert row["error"] == reason


def test_each_target_of_a_container_run_has_its_own_tests_dir(tmp_path, monkeypatch):
    """The work directory the container mounts holds a directory of each target's
    own with its tests, as outside a container, so no target's phases see
    another target's suite."""
    from testuary import config as config_module
    from testuary import project as project_module

    project = tmp_path / "project"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    names = ("calc.py", "shapes.py")
    for name in names:
        (project / name).write_text("def f():\n    return 1\n", encoding="utf-8")
    zone = tmp_path / "zone"
    zone.mkdir()
    monkeypatch.setattr(config_module, "project_exec_mode", lambda: "container")
    monkeypatch.setattr(project_module, "project_work_dir", lambda: str(zone))
    tests_dirs = []

    def record_and_stop(self, run_label=None):
        tests_dirs.append(self.work_tests)
        return False

    monkeypatch.setattr(orchestrator._FileRun, "_copy_project", record_and_stop)
    output = tmp_path / "output"
    invocation = orchestrator._Invocation(
        SimpleNamespace(resume=False, runs=1, run=None, stop_after_phase=None),
        project_name="project", projects_root=project.resolve(), container_run=False,
        results_dir=output / "results", logs_dir=output / "logs", work_root=output / "_work",
        file_count=len(names), reused=frozenset())
    for number, name in enumerate(names, start=1):
        orchestrator._FileRun(invocation, {"target_number": number, "file_path": str(project / name)}).execute()
    assert len(tests_dirs) == 2 and tests_dirs[0] != tests_dirs[1]
    assert all(tests_dir.name == "tests" and tests_dir.parent.parent == zone for tests_dir in tests_dirs)
