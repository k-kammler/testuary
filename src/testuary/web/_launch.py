"""Run Testuary's command line in this process for one job of the web
interface: ``python <this file> <testuary arguments>``, or an extension's
job module as ``python -m`` runs it: ``python <this file> --module NAME
<arguments>``.

A job stops as Ctrl+C stops a run: on Windows the interface writes "stop"
to the job's input and a thread here raises SIGINT, and on Linux and macOS
it sends SIGINT to the process group. The end of the input means "stop"
too: the interface has ended. Testuary and its processes read the null
device as their input.
"""
from __future__ import annotations

import os
import sys

# Started by its file path, Python puts this directory first on the import
# path. It goes before anything is imported, with the absolute form of each
# relative PYTHONPATH entry (the job's directory). The working directory is not on
# the path.
_DROPPED = {os.path.normcase(os.path.dirname(os.path.abspath(__file__)))} | {
    os.path.normcase(os.path.abspath(entry))
    for entry in os.environ.get("PYTHONPATH", "").split(os.pathsep)
    if entry and not os.path.isabs(entry)}
sys.path[:] = [entry for entry in sys.path
               if entry not in ("", ".") and os.path.normcase(os.path.abspath(entry)) not in _DROPPED]

import runpy  # noqa: E402
import signal  # noqa: E402
import threading  # noqa: E402


def _follow_input(descriptor: int) -> None:
    """Raise SIGINT for each line "stop" and once more at the end of input,
    reading the descriptor itself (the shutdown would close a file object
    under a waiting thread)."""
    pending = b""
    while True:
        try:
            data = os.read(descriptor, 4096)
        except OSError:
            break
        if not data:
            break
        pending += data
        while b"\n" in pending:
            line, pending = pending.split(b"\n", 1)
            if line.strip() == b"stop":
                signal.raise_signal(signal.SIGINT)
    signal.raise_signal(signal.SIGINT)


def main(arguments=None) -> None:
    # On Linux and macOS a process inherits an ignored SIGINT, as from an
    # interface started in the background of a script. The job still stops
    # as Ctrl+C stops it, and so do the processes it starts.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    # A stop before Testuary's own Ctrl+C handling (one already waiting on
    # the input, or during its imports) ends as Testuary ends an interrupt.
    try:
        _run(arguments)
    except KeyboardInterrupt:
        from testuary.exits import INTERRUPTED

        print("Interrupted.", file=sys.stderr, flush=True)
        raise SystemExit(INTERRUPTED) from None


def _run(arguments) -> None:
    arguments = list(sys.argv[1:] if arguments is None else arguments)
    requests = os.dup(0)
    empty = os.open(os.devnull, os.O_RDONLY)
    os.dup2(empty, 0)
    os.close(empty)
    threading.Thread(target=_follow_input, args=(requests,), name="testuary-web-input",
                     daemon=True).start()
    module = "testuary"
    if arguments[:1] == ["--module"]:
        module, arguments = arguments[1], arguments[2:]
    sys.argv = [module, *arguments]
    # As `python -m testuary`: the command line's own entry runs and exits.
    runpy.run_module(module, run_name="__main__", alter_sys=True)


if __name__ == "__main__":
    main()
