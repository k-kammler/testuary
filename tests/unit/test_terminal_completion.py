"""End-of-run and pre-run summaries from authored facts, without a real run."""
from __future__ import annotations

import io

import pytest

from testuary.presentation import TerminalPresenter, present_readiness
from testuary.presentation.terminal import FileCompletion, RunCompletion, present_completion
from testuary.presentation.events import Metric, Usage
from testuary.readiness import ReadinessItem, RunReadiness


def _plain(write):
    stream = io.StringIO()
    with TerminalPresenter("plain", stream=stream) as presenter:
        write(presenter)
    return stream.getvalue().splitlines()


_DONE = FileCompletion(
    "src/calc.py", "complete", suite=("out put/tests/test_calc.py",),
    metrics=(Metric("Line coverage", 85.0, "%", goal=100.0), Metric("Mutation score", None)),
    usage=Usage(logical_requests=12, input_tokens=30000, output_tokens=4000))


def test_summary_states_each_outcome_apart_with_its_facts():
    completion = RunCompletion(
        files=(_DONE,
               FileCompletion("src/other.py", "no_complete_suite", reason="no starter suite passed"),
               FileCompletion("src/broken.py", "failed", reason="the project's tests did not run")),
        results_directory="C:/runs/out put", log_path="C:/runs/out put/run/logs/run.log",
        resume_command="testuary generate --resume --output-dir \"C:/runs/out put\"")
    lines = _plain(lambda presenter: present_completion(completion, presenter=presenter))
    assert lines[0] == ("Run finished with failures: 1 of 3 targets with a suite that meets the "
                        "minimums, 1 without one, 1 failed to execute.")
    assert lines[1:3] == ["Results: C:/runs/out put", "Log: C:/runs/out put/run/logs/run.log"]
    start = lines.index("src/calc.py: a suite that meets the minimums")
    assert lines[start + 1:start + 4] == [
        "  Suite: out put/tests/test_calc.py",
        # Reported values and states as supplied: an unmeasured score stays unavailable.
        "  Line coverage: 85% (goal 100%) [measured] | Mutation score: unavailable",
        "  Model use: 12 model requests, 30,000 input tokens, 4,000 output tokens"]
    assert "src/other.py: no suite that meets the minimums: no starter suite passed" in lines
    assert "src/broken.py: failed to execute: the project's tests did not run" in lines
    assert lines[-1] == ("Next step: testuary generate --resume --output-dir \"C:/runs/out put\" "
                         "continues the unfinished targets.")


def test_a_file_whose_model_use_was_not_reported_says_the_label_once():
    """A file whose run reported none of its model use reads "Model use: not
    reported", not the label twice."""
    completion = RunCompletion(files=(FileCompletion("src/calc.py", "no_complete_suite",
                                                     usage=Usage()),))
    lines = _plain(lambda presenter: present_completion(completion, presenter=presenter))
    assert "  Model use: not reported" in lines


def test_a_finished_run_without_unfinished_files_has_no_resume_step():
    completion = RunCompletion(files=(_DONE, FileCompletion("src/b.py", "already_complete")),
                               resume_command="testuary generate --resume")
    lines = _plain(lambda presenter: present_completion(completion, presenter=presenter))
    assert lines[0] == "Run finished: 2 of 2 targets with a suite that meets the minimums."
    assert "src/b.py: a suite that meets the minimums, recorded earlier, not run again" in lines
    assert not any(line.startswith("Next step") for line in lines)
    # Unknown values stay unsaid: no results or log line without a path.
    assert not any(line.startswith(("Results:", "Log:")) for line in lines)


def test_a_cancelled_run_says_it_stopped():
    completion = RunCompletion(files=(FileCompletion("src/a.py", "cancelled"),), cancelled=True,
                               resume_command="testuary generate --resume")
    lines = _plain(lambda presenter: present_completion(completion, presenter=presenter))
    assert lines[0] == ("Run stopped: 0 of 1 target with a suite that meets the minimums, 1 stopped before "
                        "finishing.")
    assert lines[-1] == "Next step: testuary generate --resume continues the unfinished target."


def test_a_file_stopped_on_request_is_counted_apart_and_not_offered_a_resume():
    completion = RunCompletion(
        files=(FileCompletion("src/a.py", "stopped", reason="stopped after initial, as requested"),),
        resume_command="testuary generate --resume")
    lines = _plain(lambda presenter: present_completion(completion, presenter=presenter))
    assert lines[0] == ("Run finished: 0 of 1 target with a suite that meets the minimums, 1 ended "
                        "after the requested phase.")
    assert "src/a.py: ended after the requested phase: stopped after initial, as requested" in lines
    assert not any(line.startswith("Next step") for line in lines)


def test_the_resume_step_names_only_files_a_resume_would_continue():
    below = FileCompletion("src/low.py", "no_complete_suite", reason="below the minimum",
                           resumable=False)
    broken = FileCompletion("src/broken.py", "failed", reason="crashed")
    lines = _plain(lambda presenter: present_completion(
        RunCompletion(files=(below, broken), resume_command="testuary generate --resume"),
        presenter=presenter))
    assert lines[-1] == "Next step: testuary generate --resume continues the unfinished target."
    lines = _plain(lambda presenter: present_completion(
        RunCompletion(files=(below,), resume_command="testuary generate --resume"),
        presenter=presenter))
    assert not any(line.startswith("Next step") for line in lines)
    # The same files, for a display that offers to continue the run.
    assert RunCompletion(files=(below, broken)).unfinished == (broken,)
    assert RunCompletion(files=(below,)).unfinished == ()


@pytest.mark.parametrize("values,message", [
    (dict(path="", outcome="complete"), "needs its path"),
    (dict(path="a.py", outcome="scored"), "unrecognized file outcome"),
    (dict(path="a.py", outcome="complete", metrics=("85%",)), "Metric values"),
    (dict(path="a.py", outcome="failed", resumable=1), "resumable must be True or False"),
])
def test_file_facts_are_validated(values, message):
    with pytest.raises(ValueError, match=message):
        FileCompletion(**values)


def test_only_a_completion_value_is_rendered():
    with pytest.raises(TypeError, match="RunCompletion"):
        _plain(lambda presenter: present_completion({}, presenter=presenter))


def _report(items=(), **changes):
    values = dict(scope="inspection", status="verified", exit_code=0, source_dir="/p",
                  execution="local", source_files=("a.py",), phases=("starter",),
                  model_alias="m", provider="ollama", model="m1", credentials_configured=None,
                  output_dir="/out", output_mode="fresh", limits=(), items=items)
    values.update(changes)
    return RunReadiness(**values)


def test_before_a_run_the_report_ends_by_saying_generation_starts():
    lines = _plain(lambda presenter: present_readiness(_report(), presenter=presenter,
                                                       before_run=True))
    assert lines[-1] == "Generation starts now. It sends model requests, which may cost money."
    assert "Next step" not in "\n".join(lines)
    # A problem still ends with its fix: the producer refuses the run.
    problem = ReadinessItem("configuration", True, "failed", reason="bad", action="Fix the setting.")
    lines = _plain(lambda presenter: present_readiness(
        _report((problem,), status="failed", exit_code=2), presenter=presenter, before_run=True))
    assert lines[-1] == "Next step: Fix the setting."
    with pytest.raises(TypeError, match="before_run"):
        _plain(lambda presenter: present_readiness(_report(), presenter=presenter, before_run=1))


def test_a_playback_names_no_model_key_and_says_it_costs_nothing():
    """A demonstration's playback sends no model request: its report before
    the run leaves out the model key and says that the playback is free. A
    real run keeps both."""
    key = ReadinessItem(
        "configuration.credentials", True, "unverified",
        reason="No model key is configured. A service without keys works, and a service "
               "that needs one refuses the first model request.",
        action="If the service needs a key, set TESTUARY_OLLAMA_API_KEY.", tried_by="run")
    report = _report((key,), status="unverified", credentials_configured=False)
    for mode in ("plain", "detailed"):
        stream = io.StringIO()
        with TerminalPresenter(mode, stream=stream) as presenter:
            present_readiness(report, presenter=presenter, before_run=True, playback=True)
        text = stream.getvalue()
        assert text.splitlines()[-1] == (
            "The playback starts now. It sends no model request and costs nothing.")
        assert "model key" not in text.lower() and "credentials" not in text
        assert "Not tried yet" not in text and "cost money" not in text
    lines = _plain(lambda presenter: present_readiness(report, presenter=presenter,
                                                       before_run=True))
    assert "Not tried yet:" in lines and lines[-1].startswith("Generation starts now")
    with pytest.raises(TypeError, match="playback"):
        _plain(lambda presenter: present_readiness(report, presenter=presenter, playback=1))


def test_before_a_run_the_summary_is_no_check_and_advises_no_check():
    installed = ReadinessItem("native.python.item.pytest", True, "unverified", "pytest", "8.1.1",
                              reason="installed, not yet tried", action="run the verification check to try it")
    lines = _plain(lambda presenter: present_readiness(_report((installed,)), presenter=presenter,
                                                       before_run=True))
    text = "\n".join(lines)
    assert lines[0] == "Setup before the run: no problems found."
    assert "check" not in lines[0] and "--verify" not in text
    assert lines[-1] == "Generation starts now. It sends model requests, which may cost money."


def test_before_a_run_a_nonzero_status_never_announces_the_start():
    unconfirmed = ReadinessItem("output.saved_artifacts", True, "unverified", reason="saved results")
    lines = _plain(lambda presenter: present_readiness(
        _report((unconfirmed,), status="unverified", exit_code=1), presenter=presenter,
        before_run=True))
    assert "Generation starts now" not in "\n".join(lines)
    assert lines[-1].startswith("Next step: read every observation") and "status 1" in lines[-1]


def test_a_final_summary_after_the_run_ended_does_not_restart_the_pinned_view():
    from rich.console import Console
    from testuary.presentation import DisplayEvent

    stream = io.StringIO()
    console = Console(file=stream, force_terminal=True, width=100, height=40, color_system=None,
                      highlight=False, markup=False)
    with TerminalPresenter("compact", console=console) as presenter:
        presenter(DisplayEvent("r", "all targets", "pipeline", "run", "running", 1.0, level="run",
                               completed=0, total=1, work_unit="phases done"))
        presenter(DisplayEvent("r", "all targets", "pipeline", "run", "succeeded", 2.0, level="run",
                               completed=1, total=1, work_unit="phases done", message="Run completed"))
        present_completion(RunCompletion(files=(_DONE,)), presenter=presenter)
        assert presenter._live is None
    output = stream.getvalue()
    # The summary is the last thing written: no pinned frame follows it.
    assert output.rstrip().endswith("4,000 output tokens")


def test_a_demonstration_summary_labels_its_values_synthetic():
    from testuary.presentation import DisplayEvent

    completion = RunCompletion(files=(_DONE,))

    def write(presenter):
        presenter(DisplayEvent(run_id="r", target="pipeline", phase="pipeline", operation="run",
                               state="succeeded", elapsed_s=1, provenance="synthetic"))
        present_completion(completion, presenter=presenter)

    lines = _plain(write)
    assert "  Line coverage: 85% (goal 100%) [demo] | Mutation score: unavailable" in lines
    assert not any("[measured]" in line for line in lines)


def test_a_files_warnings_follow_its_facts_and_an_older_summary_has_none():
    """Each warning of a file's runs, such as a time limit reached, is a line
    after its fallbacks. A summary written before the field reads back with
    none."""
    item = FileCompletion("calc/ops.py", "complete", fallbacks=("the timing tie-break is off",),
                          warnings=["The mutation measurement reached its time limit (20 s)."])
    assert item.warnings == ("The mutation measurement reached its time limit (20 s).",)
    lines = _plain(lambda presenter: present_completion(RunCompletion(files=(item,)), presenter=presenter))
    assert lines[-2:] == ["  Fallback: the timing tie-break is off",
                          "  Warning: The mutation measurement reached its time limit (20 s)."]
    older = RunCompletion.from_dict({"files": [{"path": "calc/ops.py", "outcome": "complete"}]})
    assert older.files[0].warnings == ()
