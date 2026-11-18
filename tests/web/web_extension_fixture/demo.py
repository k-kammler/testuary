"""The Play the demonstration job: Testuary's offline demonstration,
played at its configured length into the notes directory with its event file
beside it, which the job's page folds into the job's phase, and a note
about it."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

from testuary.presentation import DisplayEvent, EventFile, RunPlan
from testuary.presentation.events import PlannedPhase, PlannedStep
from web_extension_fixture import DEMO_EVENTS


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="play-demonstration")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--events", required=True)
    args = parser.parse_args(argv)
    notes_dir = Path(args.dir)
    notes_dir.mkdir(parents=True, exist_ok=True)
    # The demonstration's event file exists from the start, so the job's
    # page can fold it at once.
    (notes_dir / DEMO_EVENTS).write_text("", encoding="utf-8")
    started = time.monotonic()

    def event(state, message, operation="run", level="run", phase="play", label=""):
        return DisplayEvent(run_id="r0", target="notes", phase="pipeline" if level == "run" else phase,
                            operation=operation, state=state, elapsed_s=time.monotonic() - started,
                            message=message, level=level, label=label)

    with EventFile(args.events) as events:
        events.plan(RunPlan(files=("notes",), runs=("r0",),
                            phases=(PlannedPhase("play", "Play the demonstration"),),
                            finishing=PlannedStep("note", "Write a note")))
        events(event("running", "Playing the demonstration"))
        events(event("running", "Play the demonstration", operation="phase/run-0", level="phase"))
        status = subprocess.run([sys.executable, "-m", "testuary", "--demo", "--demo-speed", "1",
                                 "--output-dir", str(notes_dir / "demo"), "--events", str(notes_dir / DEMO_EVENTS),
                                 "--display", "plain", "--no-color"], stdin=subprocess.DEVNULL).returncode
        state = "succeeded" if status == 0 else "failed"
        events(event(state, "Play the demonstration", operation="phase/run-0", level="phase"))
        note = f"Testuary's demonstration ended with exit status {status}."
        (notes_dir / "notes.json").write_text(json.dumps({"count": 1, "notes": [note], "demo": "demo"}),
                                           encoding="utf-8")
        events(event("succeeded", "Note written", operation="step/finishing", level="step",
                     phase="finishing", label="Write a note"))
        events(event(state, "Played the demonstration" if status == 0 else note))
    return status


if __name__ == "__main__":
    raise SystemExit(main())
