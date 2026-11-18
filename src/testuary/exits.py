"""The exit statuses of Testuary's commands and what each means.

A program that starts Testuary, such as a browser interface, shows these
sentences for a command's exit status instead of writing its own, with the
state's one word and whether it is a failure, and the guides list the same
sentences (``tools/reference_pages.py`` writes them).
``CLEANUP_SECONDS`` is the longest time a stopped run takes to clean up what
it cancelled before it ends: a force stop is only needed after it.

Nothing here draws or prints, and importing this module loads no other part
of Testuary.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

#: The commands whose statuses differ: a run (``testuary`` with a project),
#: ``testuary check`` and a demonstration (``testuary --demo``).
RUN = "run"
CHECK = "check"
DEMO = "demo"

#: The exit status of a command interrupted by Ctrl+C.
INTERRUPTED = 130


@dataclass(frozen=True)
class ExitStatus:
    """One exit status of a command, the sentence that says what it means,
    its state in one word and whether that state is a failure. A refused
    setting, a problem the check found and an interruption are none."""

    status: int
    meaning: str
    word: str
    failure: bool


STATUSES: Dict[str, Tuple[ExitStatus, ...]] = {
    RUN: (
        ExitStatus(0, "The run finished. A suite below a configured minimum, a phase "
                      "that ended without a suite and a target an earlier run in this "
                      "directory had finished are results, recorded in tests.csv.",
                   "finished", False),
        ExitStatus(1, "A target failed to execute: it crashed, could not be set up, "
                      "copied, restored or delivered, or some generated starter suites "
                      "could not be checked, because running them timed out or did not "
                      "start. The summary and the run's log say why.", "failed", True),
        ExitStatus(2, "A setting or option was refused, or a required tool was "
                      "missing, before any work started.", "refused", False),
        ExitStatus(INTERRUPTED, "The run was stopped with Ctrl+C. Work that already "
                                "finished is kept, and the run can be continued in its "
                                "run directory (at the terminal: --resume).",
                   "interrupted", False),
    ),
    CHECK: (
        ExitStatus(0, "The check found no problem. Parts that only a real run can "
                      "try are listed under \"not tried yet\".", "ready", False),
        ExitStatus(1, "The check found a problem.", "problem", False),
        ExitStatus(2, "A setting was refused or a choice is ambiguous.", "refused", False),
        ExitStatus(INTERRUPTED, "The check was stopped with Ctrl+C. No setup was "
                                "changed.", "interrupted", False),
    ),
    DEMO: (
        ExitStatus(0, "The recorded run was played back completely.", "finished", False),
        ExitStatus(1, "The demonstrated run failed, or the demonstration refused a process "
                      "or a network connection a library started. What it wrote so far is "
                      "kept and marked as demonstration output. Its log is under run/logs.",
                   "failed", True),
        ExitStatus(2, "A usage error, such as a run directory that is not empty or a "
                      "language without demonstration data.", "refused", False),
        ExitStatus(3, "The recorded data do not match this Testuary version. If "
                      "playback had started, demo/stale.json names the first call "
                      "that differed.", "outdated", True),
        ExitStatus(INTERRUPTED, "The demonstration was stopped with Ctrl+C. Start it "
                                "again in a new or empty run directory.",
                   "interrupted", False),
    ),
}


def meaning(command: str, code: int) -> str:
    """The sentence for status ``code`` of ``command`` (``RUN``, ``CHECK`` or
    ``DEMO``). A status the command does not document gets a plain sentence
    that names it."""
    return status(command, code).meaning


def status(command: str, code: int) -> ExitStatus:
    """``code`` of ``command`` with its sentence, word and whether it is a
    failure. A status the command does not document ends the command
    unexpectedly: a failure that names its code."""
    for item in STATUSES[command]:
        if item.status == code:
            return item
    return ExitStatus(code, f"The command ended with status {code}.", "failed", True)


def __getattr__(name: str):
    # CLEANUP_SECONDS is read where the mutation sessions define it, only when
    # asked for, so importing this module stays light.
    if name == "CLEANUP_SECONDS":
        from testuary.languages.mutation import CANCEL_CLEANUP_WAIT_S

        return CANCEL_CLEANUP_WAIT_S
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
