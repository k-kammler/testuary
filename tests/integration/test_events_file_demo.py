"""A demonstration with --events writes its playback's events, with the
demonstration's provenance, to a file named relative to the caller's directory."""
from __future__ import annotations

import json
import subprocess
import sys

import pytest

from tool_fixtures import same_code_environment

pytestmark = pytest.mark.slow


@pytest.mark.parametrize("language", ["python", "java"])
def test_a_demonstration_writes_its_events(tmp_path, language):
    environment = same_code_environment()
    completed = subprocess.run(
        [sys.executable, "-m", "testuary", "--demo", "--demo-language", language,
         "--demo-speed", "instant", "--output-dir", "out", "--events", "events.jsonl"],
        cwd=tmp_path, env=environment, capture_output=True, text=True, timeout=900)
    assert completed.returncode == 0, completed.stdout[-3000:] + completed.stderr[-3000:]
    header, *records = [json.loads(line) for line in
                        (tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()]
    assert header["schema_version"] == 3
    kinds = [next(iter(record)) for record in records]
    # In the display's order: the setup report before the first phase, the
    # completion after every file and before the run's final event.
    first_phase = next(index for index, record in enumerate(records)
                       if "event" in record and record["event"]["level"] == "phase")
    assert kinds.count("readiness") == 1 and kinds.index("readiness") < first_phase
    assert kinds.count("completion") == 1 and kinds[-2:] == ["completion", "event"]
    assert records[-2]["completion"]["files"][0]["outcome"] == "complete"
    assert (records[-1]["event"]["level"], records[-1]["event"]["state"]) == ("run", "succeeded")
    events = [record["event"] for record in records if "event" in record]
    assert events and all(event["provenance"] == "synthetic" for event in events)
    runs = [event for event in events if event["level"] == "run"]
    assert runs[-1]["state"] == "succeeded"
    # Every phase of the playback, each started and ended, then Finishing.
    phases = [event["phase"] for event in events if event["level"] == "phase"]
    assert phases == [phase for phase in ("starter", "coverage", "mutation") for _ in range(2)]
    finishing = [event["state"] for event in events
                 if event["operation"] == "step/finishing"]
    assert finishing == ["running", "succeeded"]
    demo = json.loads((tmp_path / "out" / "DEMO.json").read_text(encoding="utf-8"))
    assert demo["result"]["outcome"] == "completed"
