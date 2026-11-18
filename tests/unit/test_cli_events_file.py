"""``--events FILE`` writes the display's progress events as JSON lines for
programs that show a run. Without it the display is the only sink."""
from __future__ import annotations

import io
import json

import pytest

from testuary import __main__ as cli
from testuary import __version__, orchestrator, settings
from testuary.presentation import EVENT_FILE_SCHEMA, DisplayEvent, EventFile, Metric, TerminalPresenter
from testuary.settings import RunSettings


def _lines(path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def test_only_generation_offers_the_option():
    def options(parser):
        return {option for action in parser._actions for option in action.option_strings}

    assert "--events" in options(cli._build_parser())
    assert "--events" not in options(cli._build_parser(checking=True))


def test_without_the_option_the_display_is_the_sink(monkeypatch):
    seen = {}

    def run_pipeline(invocation, selected_languages, *, settings, event_sink):
        seen["sink"], seen["invocation"] = event_sink, invocation
        return 0

    monkeypatch.setattr("sys.argv", ["testuary", "--display", "plain", "--no-color"])
    monkeypatch.setattr(settings, "resolve_settings", lambda **kw: RunSettings())
    monkeypatch.setattr(orchestrator, "run_pipeline", run_pipeline)
    assert cli._main() == 0
    assert isinstance(seen["sink"], TerminalPresenter)
    assert not hasattr(seen["invocation"], "events")


def test_the_file_holds_each_event_the_display_shows(tmp_path, monkeypatch, capsys):
    target = tmp_path / "events.jsonl"

    def execute(args, languages):
        assert not hasattr(args, "events")
        args._progress.emit(target="calc.py", phase="coverage", operation="tool",
                            state="succeeded", provenance="recorded")
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["testuary", "--display", "plain", "--no-color",
                                     "--events", "events.jsonl"])
    monkeypatch.setattr(settings, "resolve_settings", lambda **kw: RunSettings())
    monkeypatch.setattr(orchestrator, "_run_pipeline", execute)
    assert cli._main() == 0
    header, *records = _lines(target)
    assert header == {"schema_version": 3, "testuary": __version__}
    assert all(set(record) == {"event"} for record in records)
    events = [record["event"] for record in records]
    assert [event["state"] for event in events] == ["running", "succeeded", "succeeded"]
    assert events[1]["target"] == "calc.py" and events[1]["provenance"] == "recorded"
    assert set(events[0]) == set(DisplayEvent.__dataclass_fields__)
    # The display showed the same run.
    assert "Checking run inputs" in capsys.readouterr().err


def test_the_runs_own_key_is_removed_where_no_pattern_sees_it(tmp_path, monkeypatch, capsys):
    """A key that no pattern recognizes is removed from the console and the
    event file alike, by the resolved settings' own redaction."""
    planted = "zq81Plain7731value"
    resolved = settings.resolve_settings(
        base_dir=tmp_path, use_environment=False,
        models={"m": {"provider": "openai", "model": "x", "api_url": "http://127.0.0.1:9/v1",
                      "api_key_env": "TUTORIAL_PLANTED_KEY"}},
        environment={"TUTORIAL_PLANTED_KEY": planted})
    target = tmp_path / "events.jsonl"

    def execute(args, languages):
        args._progress.emit(target="calc.py", phase="coverage", operation="request/1",
                            state="failed", message=f"API key: {planted} was refused",
                            failure="execution")
        return 0

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["testuary", "--display", "plain", "--no-color",
                                     "--events", "events.jsonl"])
    monkeypatch.setattr(settings, "resolve_settings", lambda **kw: resolved)
    monkeypatch.setattr(orchestrator, "_run_pipeline", execute)
    cli._main()
    shown = capsys.readouterr().err
    assert "was refused" in shown and planted not in shown
    assert "API key: [redacted] was refused" in shown
    written = target.read_text(encoding="utf-8")
    assert "was refused" in written and planted not in written


@pytest.mark.parametrize("make, message", [
    (lambda root: root / "missing" / "events.jsonl", "does not exist"),
    (lambda root: root, "is a directory"),
])
def test_an_unusable_path_is_refused_before_anything_runs(tmp_path, monkeypatch, capsys,
                                                          make, message):
    def never(*args, **kwargs):
        raise AssertionError("the run started")

    monkeypatch.setattr("sys.argv", ["testuary", "--events", str(make(tmp_path))])
    monkeypatch.setattr(orchestrator, "run_pipeline", never)
    with pytest.raises(SystemExit) as refused:
        cli._main()
    assert refused.value.code == 2
    assert message in capsys.readouterr().err


def test_a_refused_command_leaves_an_earlier_event_file_as_it_was(tmp_path, monkeypatch):
    earlier = tmp_path / "events.jsonl"
    earlier.write_text('{"event": "earlier"}\n', encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["testuary", "--events", str(earlier), "--demo-data", "toy"])
    with pytest.raises(SystemExit) as refused:
        cli._main()
    assert refused.value.code == 2
    assert earlier.read_text(encoding="utf-8") == '{"event": "earlier"}\n'


def test_a_failed_write_ends_the_file_not_the_run(tmp_path, monkeypatch, capsys):
    shown = []

    class Full(io.StringIO):
        def write(self, text):
            if self.getvalue():
                raise OSError(28, "No space left on device")
            return super().write(text)

    stream = Full()
    # Only the event file's module opens this stream.
    from testuary.presentation import event_file

    monkeypatch.setattr(event_file, "open", lambda *args, **kwargs: stream, raising=False)
    with EventFile(tmp_path / "events.jsonl") as events:
        sink = cli._observe(shown.append, events)
        event = DisplayEvent(run_id="r", target="calc.py", phase="coverage",
                             operation="tool", state="running", elapsed_s=0.0)
        sink(event)
        sink(event)
        # The header line only: the first event's write failed.
        assert stream.getvalue().count("\n") == 1
    assert shown == [event, event]
    assert capsys.readouterr().err.count("the event file stopped: OSError") == 1


PLANTED = ("provider said api_key=sk-test-123 for "
           "https://user:pw@example.test/v1?api_key=abc&region=eu")
SECRETS = ("sk-test-123", "user:pw", "abc&", "plain-known-value", "xyz789")


def test_credentials_are_removed_from_every_text_as_the_display_removes_them(tmp_path):
    target = tmp_path / "events.jsonl"
    event = DisplayEvent(
        run_id="r", target="calc.py", phase="coverage", operation="request/1",
        state="failed", elapsed_s=1.0, message=PLANTED + " and plain-known-value",
        metrics=(Metric("line coverage", unit="token=xyz789"),), failure="execution")
    with EventFile(target, secrets=("plain-known-value",)) as events:
        events(event)
    text = target.read_text(encoding="utf-8")
    for secret in SECRETS:
        assert secret not in text
    record = _lines(target)[1]["event"]
    assert "[redacted]" in record["message"] and "region=eu" in record["message"]
    assert record["metrics"][0]["unit"] == "token=[redacted]"
    assert record["target"] == "calc.py" and record["operation"] == "request/1"


def test_a_credential_name_after_an_underscore_is_recognized(tmp_path):
    """A credential name counts after any character but a letter or digit,
    such as the underscore in OPENAI_API_KEY, MY_TOKEN or client_secret."""
    from testuary.presentation.view import safe_text

    message = "set OPENAI_API_KEY=sk-env-1 MY_TOKEN=env-2 client_secret: env-3"
    hidden = ("set OPENAI_API_KEY=[redacted] MY_TOKEN=[redacted] "
              "client_secret: [redacted]")
    target = tmp_path / "events.jsonl"
    with EventFile(target) as events:
        events(DisplayEvent(run_id="r", target="calc.py", phase="coverage",
                            operation="request/1", state="failed", elapsed_s=1.0,
                            message=message, failure="execution"))
    assert safe_text(message) == hidden
    assert _lines(target)[1]["event"]["message"] == hidden


def test_another_program_writes_the_format_through_the_public_writer(tmp_path):
    """A program whose run a page shows as a run's, such as a web interface
    extension's job kind, writes the event file through
    testuary.presentation's writer, with the plan and completion types it
    records and the format's version."""
    from testuary import presentation

    target = tmp_path / "events.jsonl"
    with EventFile(target) as events:
        events(DisplayEvent(run_id="r", target="t", phase="pipeline", operation="run",
                            state="running", elapsed_s=0.0, message="started"))
    lines = [json.loads(line) for line in target.read_text(encoding="utf-8").splitlines()]
    assert lines[0] == {"schema_version": EVENT_FILE_SCHEMA, "testuary": __version__}
    assert EVENT_FILE_SCHEMA == 3 and lines[1]["event"]["message"] == "started"
    assert {"EventFile", "EVENT_FILE_SCHEMA", "RunPlan", "RunCompletion"} <= set(presentation.__all__)


def test_the_setup_report_and_the_end_of_the_run_are_records_of_their_own(tmp_path):
    from testuary.presentation.terminal import FileCompletion, RunCompletion
    from testuary.readiness import _configuration_error_report

    target = tmp_path / "events.jsonl"
    report = _configuration_error_report(ValueError(PLANTED + " token=xyz789"))
    completion = RunCompletion(
        files=(FileCompletion(path="calc.py", outcome="failed",
                              reason=PLANTED + " and plain-known-value"),),
        results_directory=str(tmp_path / "results"), log_path=str(tmp_path / "run.log"),
        resume_command="testuary --api-url https://example.test/v1?token=xyz789 --resume")
    with EventFile(target, secrets=("plain-known-value",)) as events:
        events.readiness(report)
        events.completion(completion)
    header, readiness, end = _lines(target)
    assert header["schema_version"] == 3
    assert readiness["readiness"]["exit_code"] == 2
    assert readiness["readiness"]["items"][0]["key"] == "configuration"
    assert end["completion"]["files"][0]["outcome"] == "failed"
    assert end["completion"]["results_directory"] == str(tmp_path / "results")
    text = target.read_text(encoding="utf-8")
    for secret in SECRETS:
        assert secret not in text
    assert text.count("[redacted]") >= 4


def test_a_refused_run_records_its_setup_report(tmp_path, monkeypatch):
    def refused(**kwargs):
        raise settings.ConfigurationError("[api] api_url https://example.test/v1?api_key=abc")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["testuary", "--display", "plain", "--no-color",
                                     "--events", "events.jsonl"])
    monkeypatch.setattr(settings, "resolve_settings", refused)
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 2
    header, readiness = _lines(tmp_path / "events.jsonl")
    assert readiness["readiness"]["status"] == "refused"
    assert "abc&" not in json.dumps(readiness) and "api_key=abc" not in json.dumps(readiness)


def test_a_run_missing_a_required_tool_is_refused_with_its_setup_report(tmp_path, monkeypatch, capsys):
    """A run whose selected language lacks a tool the workflow requires is
    refused before any phase, as a refused setting is: status 2, its setup
    report in the file with what is missing and the fix, shown on the
    console, and a last event "Setup refused" that is no failure."""
    class Backend:
        def capabilities(self, source_dir):
            raise RuntimeError("no report")

        def check_dependencies(self):
            return ["the test tool is not installed"]

    def execute(args, languages):
        orchestrator._preflight_run_dependencies({"python"}, ".")
        pytest.fail("the run went on after its dependency check")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["testuary", "--display", "plain", "--no-color",
                                     "--events", "events.jsonl"])
    monkeypatch.setattr(settings, "resolve_settings", lambda **kw: RunSettings())
    monkeypatch.setattr(orchestrator.languages, "get_backend", lambda name: Backend())
    monkeypatch.setattr(orchestrator, "_run_pipeline", execute)
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 2
    header, *records = _lines(tmp_path / "events.jsonl")
    events = [record["event"] for record in records if "event" in record]
    assert [(event["state"], event["message"], event["failure"]) for event in events] == [
        ("running", "Checking run inputs", None), ("refused", "Setup refused", None)]
    (report,) = [record["readiness"] for record in records if "readiness" in record]
    assert (report["status"], report["exit_code"]) == ("refused", 2)
    (item,) = report["items"]
    assert (item["key"], item["required"], item["status"]) == ("native.execution", True, "missing_setup")
    assert "[python] the test tool is not installed" in item["reason"]
    assert item["action"] == "Install what is missing or change the settings, then run the command again."
    shown = capsys.readouterr()
    text = shown.out + shown.err
    assert "  The tools each language needs: required tools are missing for this run:" in text
    assert "[python] the test tool is not installed" in text and "Execution:" not in text


def test_the_file_and_the_display_take_events_under_one_lock(tmp_path):
    class Lock:
        held = False

        def __enter__(self):
            self.held = True

        def __exit__(self, *exc_info):
            self.held = False

    class Presenter:
        _lock = Lock()
        seen = []

        def __call__(self, event):
            self.seen.append(self._lock.held)

    presenter = Presenter()
    written = []
    sink = cli._observe(presenter, lambda event: written.append(presenter._lock.held))
    sink(DisplayEvent(run_id="r", target="calc.py", phase="coverage", operation="tool",
                      state="running", elapsed_s=0.0))
    assert written == [True] and presenter.seen == [True]
    assert presenter._lock.held is False
