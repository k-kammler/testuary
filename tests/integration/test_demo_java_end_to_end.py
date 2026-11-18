"""Offline Java pipeline acceptance; no JDK, measurement or model service."""
from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path

import pytest

from tool_fixtures import same_code_environment

from testuary import project

pytestmark = pytest.mark.slow


def _manifest() -> dict:
    """The shipped Java scenario's manifest."""
    from testuary.demo import cli

    return cli.load_data("java")[0]


def _results(output: Path) -> Path:
    """The first run's directory of the shipped file, which holds its suite and
    phase records."""
    return project.run_dir(output / "run" / "results" / _manifest()["exemplar"]["workspace"], 0)


def _demo(output: Path, *, changed_source: bool = False, display: str = "plain",
          frame_width: int | None = None, cancel_after_starter: bool = False):
    code = """
import json, os, sys
from dataclasses import asdict
from pathlib import Path
from testuary import __main__ as entry
from testuary.demo import cli
frames = []
frame_report = Path(sys.argv[1]).with_suffix('.frames.json')
cancel_report = Path(sys.argv[1]).with_suffix('.cancel.json')
observations = []
presenters = []
cancel_after_initial = sys.argv[5] == 'cancel'
if sys.argv[4] != 'none':
    from io import StringIO
    from rich.console import Console
    from testuary import presentation
    original_presenter = presentation.TerminalPresenter
    width = int(sys.argv[4])
    class FrameRecorder(original_presenter):
        def __init__(self, mode, **kwargs):
            super().__init__(mode, console=Console(file=StringIO(), width=width,
                             height=25, force_terminal=True, color_system=None))
            assert self.mode == 'compact'
        def __call__(self, event):
            super().__call__(event)
            with self.console.capture() as capture:
                self.console.print(self.frame())
            frames.append({'phase': event.phase, 'state': event.state,
                           'provenance': event.provenance, 'elapsed_s': event.elapsed_s,
                           'target': event.target, 'operation': event.operation,
                           'message': event.message, 'total': event.total,
                           'metrics': [asdict(metric) for metric in event.metrics],
                           'usage': asdict(event.usage) if event.usage is not None else None,
                           'frame': ' '.join(capture.get().split())})
    presentation.TerminalPresenter = FrameRecorder
    os.environ['TERM'] = 'xterm'
if cancel_after_initial:
    from testuary import presentation
    from testuary import languages, models as _model_source, reduction as _cover_source
    original_backend = languages.get_backend('java')
    original_model_source = _model_source.current()
    original_cover_source = _cover_source.current()
    original_presenter = presentation.TerminalPresenter
    class CancellingPresenter(original_presenter):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.interrupted = False
            presenters.append(self)
        def __call__(self, event):
            super().__call__(event)
            observations.append({'phase': event.phase, 'state': event.state,
                                 'provenance': event.provenance})
            if (event.phase == 'starter' and event.state == 'succeeded'
                    and event.operation.startswith('phase/run-') and not self.interrupted):
                self.interrupted = True
                raise KeyboardInterrupt('cancel after durable initial evidence')
    presentation.TerminalPresenter = CancellingPresenter
os.environ.pop('JAVA_HOME', None)
os.environ['PATH'] = ''
os.environ['TESTUARY_DEMO_OUTSIDE_VALUE'] = 'not read by the demo'
original = dict(os.environ)
working_directory = Path.cwd()
if sys.argv[2] == 'changed':
    original_load = cli.load_data
    def changed(language='python'):
        manifest, files = original_load(language)
        layout = json.loads(files['project.json'])
        key = manifest['exemplar']['target']
        layout['files'][key] = layout['files'][key] + '// changed\\n'
        files['project.json'] = json.dumps(layout).encode('utf-8')
        return manifest, files
    cli.load_data = changed
sys.argv = ['testuary', '--demo', '--demo-language', 'java', '--display', sys.argv[3],
            '--no-color', '--output-dir', sys.argv[1]]
original_argv = list(sys.argv)
try:
    entry._main()
except SystemExit as exc:
    status = exc.code
except KeyboardInterrupt:
    assert cancel_after_initial
    assert presenters and any(p.interrupted for p in presenters)
    assert all(p._closed and p._live is None for p in presenters)
    assert languages.get_backend('java') is original_backend
    assert _model_source.current() is original_model_source
    assert _cover_source.current() is original_cover_source
    assert sys.argv == original_argv
    cancel_report.write_text(json.dumps({'propagated': True, 'observations': observations,
                                        'presentation_closed': True, 'hooks_restored': True}),
                             encoding='utf-8')
    status = 130
else:
    raise AssertionError('the public entry point did not exit')
assert dict(os.environ) == original
assert Path.cwd() == working_directory
if frames:
    frame_report.write_text(json.dumps(frames), encoding='utf-8')
raise SystemExit(status)
"""
    return subprocess.run(
        [sys.executable, "-B", "-c", code, str(output),
         "changed" if changed_source else "same", display,
         "none" if frame_width is None else str(frame_width),
         "cancel" if cancel_after_starter else "none"],
        cwd=output.parent, env=same_code_environment(), capture_output=True, text=True,
        encoding="utf-8", errors="replace", timeout=90)


def test_recorded_java_pipeline_plays_without_tools_and_labels_every_output(tmp_path):
    output = tmp_path / "demo"
    result = _demo(output)
    assert result.returncode == 0, result.stdout + result.stderr
    assert '\x1b' not in result.stdout + result.stderr
    events = [line for line in result.stderr.splitlines()
              if re.match(r"^\d+:\d{2}:\d{2} (?:RUNNING|WAITING|RETRYING|SUCCEEDED|FAILED|STOPPED)\b", line)]
    assert len(events) >= 2
    assert all("[demo]" in line for line in events)
    assert "RUNNING" in events[0] and "SUCCEEDED" in events[-1]
    record = json.loads((output / "DEMO.json").read_text(encoding="utf-8"))
    manifest = _manifest()
    assert record["language"] == "java"
    assert "recorded in one real run" in record["statement"]
    outcome = record["result"]
    assert outcome["outcome"] == "completed"
    assert outcome["answers"] == manifest["entries"]
    assert outcome["classes"] == manifest["classes"]
    assert outcome["processes"] == []
    assert outcome["network_connections"] == 0
    assert (output / "demo" / "audit.jsonl").read_bytes() == b""
    ledger = [json.loads(line) for line in (output / "demo" / "ledger.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert {"starter"} <= {item["segment"] for item in ledger} <= {
        "starter", "coverage", "mutation"}
    from testuary.demo import cli
    from testuary.demo.scenario import Scenario

    scenario_path = tmp_path / "scenario.json.gz"
    scenario_path.write_bytes(cli.load_data("java")[1]["scenario.json.gz"])
    assert sorted(item["kind"] for item in ledger) == sorted(
        entry.kind for entry in Scenario.load(scenario_path).entries)
    scenario_path.unlink()
    tables = list((output / "run").rglob("tests.csv"))
    assert len(tables) == 1
    with tables[0].open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(row["demo"] == "demo" for row in rows)
    for path in (output / "run").rglob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            assert "recorded in one real run" in data["testuary_demo"], path
    sources = [path for name in manifest["recorded"]["delivered_suite"]
               for path in (output / "run").rglob(name)]
    assert sources
    assert all((path.parent / "DEMO_NOTICE.txt").is_file() for path in sources)
    assert all("not read by the demo" not in value
               for item in record["overridden_settings"] for value in item.values())
    assert not any(item["setting"].startswith(("--display", "--no-color", "--demo-language"))
                   for item in record["overridden_settings"])
    assert sorted(path.name for path in tmp_path.iterdir()) == ["demo"]
    text = (output / "DEMO.md").read_text(encoding="utf-8")
    assert "recorded-v1" in text


def test_changed_java_example_stops_without_a_request_or_automatic_rebuild(tmp_path):
    output = tmp_path / "changed"
    result = _demo(output, changed_source=True)
    assert result.returncode == 3, result.stdout + result.stderr
    record = json.loads((output / "DEMO.json").read_text(encoding="utf-8"))
    assert record["result"]["outcome"] == "stale"
    assert record["result"]["processes"] == []
    assert record["result"]["network_connections"] == 0
    assert (output / "demo" / "stale.json").is_file()
    assert "no automatic recovery or live request" in result.stdout


def test_existing_java_demo_outputs_are_not_overwritten(tmp_path):
    output = tmp_path / "existing"
    output.mkdir()
    kept = output / "keep.txt"
    kept.write_bytes(b"user material\n")
    result = _demo(output)
    assert result.returncode == 2, result.stdout + result.stderr
    assert list(output.iterdir()) == [kept]
    assert kept.read_bytes() == b"user material\n"


def _scientific_projection(output: Path):
    results = _results(output)
    ledger = (output / "demo" / "ledger.jsonl").read_text(encoding="utf-8").splitlines()
    selections = {}
    for phase in ("starter", "coverage", "mutation"):
        record = json.loads((results / "archive" / phase / "consolidation_record.json").read_text(
            encoding="utf-8"))
        keys = ("delivered_from", "selected_source_sha256", "merged_source_sha256", "reduced_source_sha256",
                "execution_name", "measurement_identity", "merged_tests", "delivered_tests",
                "reduction_valid", "repaired_tests", "selection_audit")
        selections[phase] = {key: record[key] for key in keys}
    matrices = {}
    for name, keys in (("kill_matrix.json", ("summary", "mutants")),
                       ("coverage_matrix.json", ("summary", "lines", "branches"))):
        # A run that delivered no complete suite writes no matrix.
        if not (results / "tests" / name).is_file():
            continue
        record = json.loads((results / "tests" / name).read_text(encoding="utf-8"))
        matrices[name] = {key: record[key] for key in keys}
    usage = {}
    for phase in ("coverage", "mutation"):
        path = output / "run" / "logs" / "test" / results.name / f"{phase}_usage.jsonl"
        # A loop that sent no model request writes no usage log.
        if not path.is_file():
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        keys = ("request_idx", "logical_request_idx", "attempt", "outcome", "input_tokens",
                "output_tokens", "cache_hit_tokens", "cache_miss_tokens", "usage_known")
        usage[phase] = [{key: record[key] for key in keys} for record in records]
    # Saved responses sit under per-request directories named by 32 hex digits,
    # which differ per run: the names are masked, the number of files kept.
    sources = sorted(
        ("/".join("<request>" if re.fullmatch(r"[0-9a-f]{32}", part) else part
                  for part in path.relative_to(results).parts), path.read_bytes())
        for path in results.rglob("*.java"))
    return {"sources": sources, "selections": selections, "matrices": matrices,
            "usage": usage, "answers": sorted(ledger)}


def test_display_modes_preserve_suites_measurements_selections_and_requests(tmp_path):
    snapshots = []
    for display in ("plain", "compact", "detailed"):
        output = tmp_path / display
        result = _demo(output, display=display)
        assert result.returncode == 0, result.stdout + result.stderr
        assert "\x1b" not in result.stdout + result.stderr
        record = json.loads((output / "DEMO.json").read_text(encoding="utf-8"))
        assert record["result"]["answers"] == _manifest()["entries"]
        assert record["result"]["processes"] == []
        assert record["result"]["network_connections"] == 0
        # The answer ledger shows only in the detailed view, each answer's
        # class in words.
        answers = [line for line in result.stdout.splitlines() if line.startswith("[demo] ")]
        if display == "detailed":
            assert any(" recorded " in line for line in answers)
            assert not any(f" {word} " in line for line in answers for word in ("real", "synthetic"))
        else:
            assert answers == [], display
        snapshots.append(_scientific_projection(output))
    assert snapshots[0] == snapshots[1] == snapshots[2]


def test_compact_demo_frames_keep_provenance_and_plain_scientific_outputs(tmp_path):
    plain = tmp_path / "plain"
    result = _demo(plain)
    assert result.returncode == 0, result.stdout + result.stderr
    expected = _scientific_projection(plain)
    for width in (40, 100):
        output = tmp_path / f"compact-{width}"
        result = _demo(output, display="compact", frame_width=width)
        assert result.returncode == 0, result.stdout + result.stderr
        frames = json.loads(output.with_suffix(".frames.json").read_text(encoding="utf-8"))
        assert frames and frames[0]["state"] == "running" and frames[-1]["state"] == "succeeded"
        assert {frame["phase"] for frame in frames} >= {"pipeline", "starter", "coverage", "mutation", "finishing"}
        assert all(frame["provenance"] == "synthetic" for frame in frames)
        assert all("demo playback" in frame["frame"] for frame in frames)
        elapsed = [frame["elapsed_s"] for frame in frames]
        assert elapsed == sorted(elapsed)
        record = json.loads((output / "DEMO.json").read_text(encoding="utf-8"))
        assert record["result"]["answers"] == _manifest()["entries"]
        assert record["result"]["processes"] == []
        assert record["result"]["network_connections"] == 0
        assert _scientific_projection(output) == expected


def test_java_demo_observes_actual_persisted_requests_and_tool_outcomes(tmp_path):
    output = tmp_path / "producer-demo"
    result = _demo(output, display="compact", frame_width=100)
    assert result.returncode == 0, result.stdout + result.stderr
    frames = json.loads(output.with_suffix(".frames.json").read_text(encoding="utf-8"))
    assert frames and all(frame["provenance"] == "synthetic" for frame in frames)
    requests = [frame for frame in frames if frame["operation"].startswith("request/")]
    tools = [frame for frame in frames if frame["operation"].startswith("tool/")]
    # A loop that met its goal without a model request shows none.
    assert {"starter"} <= {frame["phase"] for frame in requests} <= {
        "starter", "coverage", "mutation"}
    assert tools and {frame["phase"] for frame in tools} <= {"coverage", "mutation"}
    # Every event of a file shares one identity: its selected source path.
    assert all(frame["target"] == _manifest()["exemplar"]["target"] for frame in requests + tools)
    assert all(frame["total"] is None and frame["metrics"] == [] for frame in requests + tools)
    identities = {}
    for frame in requests:
        identities.setdefault((frame["phase"], frame["operation"]), []).append(frame["state"])
    # A loop that used all its requests shows the next one refused: one
    # limit_reached event at the limit, no request sent.
    refused = [frame for frame in requests if identities[(frame["phase"], frame["operation"])]
               == ["limit_reached"]]
    assert all(frame["message"] == "Request limit reached ({0} of {0})".format(
                   frame["usage"]["logical_request_limit"])
               and frame["usage"]["logical_requests"] == frame["usage"]["logical_request_limit"]
               for frame in refused)
    assert len({frame["phase"] for frame in refused}) == len(refused)
    assert all(states == ["running", "succeeded"] for states in identities.values()
               if states != ["limit_reached"])
    for phase in ("coverage", "mutation"):
        logs = output / "run" / "logs" / "test" / _manifest()["exemplar"]["workspace"]
        path = project.run_dir(logs, 0) / f"{phase}_usage.jsonl"
        if not path.is_file():
            assert not [frame for frame in requests if frame["phase"] == phase]
            continue
        records = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
        observed = [frame for frame in requests if frame["phase"] == phase and frame["state"] == "succeeded"]
        assert len(observed) == len(records)
        assert observed[-1]["usage"]["logical_requests"] == records[-1]["logical_request_idx"]
        assert observed[-1]["usage"]["physical_requests"] == records[-1]["request_idx"]
    assert all(frame["usage"] is None for frame in tools)
    assert all(frame["usage"]["cached_input_tokens"] is None and frame["usage"]["cost_usd"] is None
               for frame in frames if frame["usage"] is not None)
    assert [frame["elapsed_s"] for frame in frames] == sorted(frame["elapsed_s"] for frame in frames)
    record = json.loads((output / "DEMO.json").read_text(encoding="utf-8"))
    assert record["result"]["outcome"] == "completed"
    assert record["result"]["answers"] == _manifest()["entries"]
    assert record["result"]["processes"] == [] and record["result"]["network_connections"] == 0
    assert (output / "demo" / "audit.jsonl").read_bytes() == b""


@pytest.mark.parametrize("display", ["plain", "compact", "detailed"])
def test_cancelled_java_demo_keeps_labeled_starter_evidence_and_refuses_overwrite(tmp_path, display):
    output = tmp_path / display
    result = _demo(output, display=display, cancel_after_starter=True)
    assert result.returncode == 130, result.stdout + result.stderr
    assert "\x1b" not in result.stdout + result.stderr
    assert "Demo complete:" not in result.stdout
    observed = json.loads(output.with_suffix(".cancel.json").read_text(encoding="utf-8"))
    assert observed["propagated"] and observed["presentation_closed"]
    assert observed["hooks_restored"]
    assert {"phase": "starter", "state": "succeeded", "provenance": "synthetic"} in observed["observations"]
    assert observed["observations"][-1] == {
        "phase": "pipeline", "state": "cancelled", "provenance": "synthetic"}
    assert not any(item["phase"] in ("coverage", "mutation", "finishing") for item in observed["observations"])
    record = json.loads((output / "DEMO.json").read_text(encoding="utf-8"))
    assert record["result"]["outcome"] == "cancelled"
    assert 0 < record["result"]["answers"] < _manifest()["entries"]
    assert record["result"]["processes"] == []
    assert record["result"]["network_connections"] == 0
    assert (output / "demo" / "audit.jsonl").read_bytes() == b""
    assert not (output / "demo" / "stale.json").exists()
    ledger = [json.loads(line) for line in (output / "demo" / "ledger.jsonl").read_text(
        encoding="utf-8").splitlines()]
    assert len(ledger) == record["result"]["answers"]
    assert {item["segment"] for item in ledger} == {"starter"}
    starter_archive = _results(output) / "archive" / "starter"
    assert (starter_archive / "consolidation_record.json").is_file()
    sources = list(starter_archive.rglob("*.java"))
    assert sources
    assert all((path.parent / "DEMO_NOTICE.txt").is_file() for path in sources)
    for path in (output / "run").rglob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            assert "recorded in one real run" in data["testuary_demo"], path
    preserved = {path.relative_to(output).as_posix(): path.read_bytes()
                 for path in output.rglob("*") if path.is_file()}
    repeated = _demo(output, display=display)
    assert repeated.returncode == 2, repeated.stdout + repeated.stderr
    assert {path.relative_to(output).as_posix(): path.read_bytes()
            for path in output.rglob("*") if path.is_file()} == preserved
