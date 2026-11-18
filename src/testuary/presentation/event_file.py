"""The event file: what the display shows, as JSON lines, for programs
that show a run."""
from __future__ import annotations

EVENT_FILE_SCHEMA = 3
"""The event file's format version, in its header line. A change of the
format raises it. The completion record's own version is separate."""


class EventFile:
    """What the display shows, as JSON lines, for programs that show a run.
    Testuary's ``--events`` and ``run_pipeline``'s ``events`` write through
    it, and so does a program whose run a page shows as a run's, such as a
    web interface extension's job kind.

    A header line holds the format version (``schema_version``,
    ``EVENT_FILE_SCHEMA``) and the Testuary version. Each further line holds
    one record, in the order the display shows it: ``{"event": ...}`` with
    the fields of a
    ``testuary.presentation.DisplayEvent``, ``{"readiness": ...}`` with the
    setup report before the run (``RunReadiness.as_dict()``), ``{"plan": ...}``
    with what the run will do (a ``RunPlan``), and
    ``{"completion": ...}`` with the end of the run (a ``RunCompletion``).
    Every text has its credentials removed as the display removes them before
    it prints, with the same secrets. A failed write ends the file, not the
    run or its display. Events hold finite numbers only, so the JSON needs no
    NaN."""

    def __init__(self, path, *, secrets=(), clean=None) -> None:
        self._path = path
        self._secrets = tuple(secrets)
        # The display's own cleaning, so the file removes what the screen removes.
        self._clean = clean
        self._stream = None
        self._open = False

    def __enter__(self):
        from testuary import __version__

        self._stream = open(self._path, "w", encoding="utf-8", newline="\n")
        self._open = True
        self._emit({"schema_version": EVENT_FILE_SCHEMA, "testuary": __version__})
        return self

    def __exit__(self, *exc_info):
        self._stream.close()
        return False

    def __call__(self, event) -> None:
        import dataclasses

        self._emit({"event": self._safe(dataclasses.asdict(event))})

    def readiness(self, report) -> None:
        """Write the setup report ``report``, a ``RunReadiness``, as a
        ``readiness`` record."""
        self._emit({"readiness": self._safe(report.as_dict())})

    def completion(self, completion) -> None:
        """Write the end of the run ``completion``, a ``RunCompletion``, as a
        ``completion`` record."""
        import dataclasses

        self._emit({"completion": self._safe(dataclasses.asdict(completion))})

    def plan(self, plan) -> None:
        """Write what the run will do, ``plan``, a ``RunPlan``, as a ``plan``
        record."""
        import dataclasses

        self._emit({"plan": self._safe(dataclasses.asdict(plan))})

    def _safe(self, value):
        from testuary.presentation.terminal import safe_text

        if isinstance(value, str):
            if self._clean is not None:
                return self._clean(value, preserve_whitespace=True)
            return safe_text(value, self._secrets, preserve_whitespace=True)
        if isinstance(value, dict):
            return {key: self._safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [self._safe(item) for item in value]
        return value

    def _emit(self, record) -> None:
        import json
        import sys as _sys

        if not self._open:
            return
        try:
            self._stream.write(json.dumps(record, ensure_ascii=True, allow_nan=False) + "\n")
            self._stream.flush()
        except (OSError, TypeError, ValueError) as exc:
            self._open = False
            print(f"testuary: the event file stopped: {type(exc).__name__}: {exc}. "
                  "The run continues.", file=_sys.stderr, flush=True)
