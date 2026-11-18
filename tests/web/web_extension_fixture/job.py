"""The Collect notes job: it writes a notes directory's notes.json and its
run's progress in Testuary's event file format."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from testuary.presentation import DisplayEvent, EventFile, RunPlan
from testuary.presentation.events import PlannedPhase, PlannedStep


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="collect-notes")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--wait", type=float, default=0.0)
    parser.add_argument("--events", required=True)
    args = parser.parse_args(argv)
    notes_dir = Path(args.dir)
    notes_dir.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()

    def event(state, message, operation="run", level="run", phase="collect", label=""):
        return DisplayEvent(run_id="r0", target="notes", phase="pipeline" if level == "run" else phase,
                            operation=operation, state=state, elapsed_s=time.monotonic() - started,
                            message=message, level=level, label=label)

    with EventFile(args.events) as events:
        events.plan(RunPlan(files=("notes",), runs=("r0",),
                            phases=(PlannedPhase("collect", "Collect notes",
                                                 steps=(PlannedStep("write", "Write notes"),)),),
                            finishing=PlannedStep("summary", "Write the summary")))
        events(event("running", "Collecting notes"))
        events(event("running", "Collect notes", operation="phase/run-0", level="phase"))
        events(event("running", "Writing notes", operation="step/write", level="step"))
        time.sleep(args.wait)
        notes = [f"Note {number}" for number in range(1, args.count + 1)]
        (notes_dir / "notes.json").write_text(json.dumps({"count": len(notes), "notes": notes}),
                                           encoding="utf-8")
        events(event("succeeded", "Notes written", operation="step/write", level="step"))
        events(event("succeeded", "Collect notes", operation="phase/run-0", level="phase"))
        events(event("succeeded", "Summary written", operation="step/finishing", level="step",
                     phase="finishing", label="Write the summary"))
        events(event("succeeded", "Collected notes"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
