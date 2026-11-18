"""The console records of an agent's log, as the tests read them."""
from __future__ import annotations

import json
from pathlib import Path


def console_records(path) -> list:
    """The console records of the log at ``path``, in order, one JSON object
    per line. A line that does not parse, such as one a stopped process left
    half written, is skipped."""
    records = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        try:
            records.append(json.loads(line))
        except ValueError:
            continue
    return [record for record in records if record.get("record") == "console"]
