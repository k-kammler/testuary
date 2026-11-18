"""The Check notes job: whether a notes directory is ready for Collect notes,
as a setup report in Testuary's readiness format."""
from __future__ import annotations

import argparse
from pathlib import Path

from testuary.presentation import EventFile
from testuary.readiness import ReadinessItem, ReadinessLimit, RunReadiness


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="check-notes")
    parser.add_argument("--dir", required=True)
    parser.add_argument("--count", type=int, default=3)
    parser.add_argument("--events", required=True)
    args = parser.parse_args(argv)
    notes_dir = Path(args.dir)
    ready = not notes_dir.exists() or notes_dir.is_dir()
    item = ReadinessItem("notes_dir", required=True, status="verified" if ready else "missing_setup",
                         selected=str(notes_dir), reason="" if ready else "The path names a file.",
                         action=None if ready else "Name a directory for the notes.")
    report = RunReadiness(
        scope="inspection", status=item.status, exit_code=0 if ready else 1, source_dir=str(notes_dir),
        execution=None, source_files=(), phases=("collect",), model_alias=None, provider=None,
        model=None, credentials_configured=None, output_dir=str(notes_dir), output_mode="fresh",
        limits=(ReadinessLimit("collect", "count", args.count, "notes", "configured"),), items=(item,))
    with EventFile(args.events) as events:
        events.readiness(report)
    return report.exit_code


if __name__ == "__main__":
    raise SystemExit(main())
