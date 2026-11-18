"""The demonstration mode: the package's real pipeline on one exemplar, with
the model's responses and the measurement results taken from a scenario.

A normal run never imports this package. ``testuary --demo`` installs the
scenario's model source and backend before any phase module is imported.
Installed playback needs no results of a run. ``play`` runs it from Python.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path
from typing import Optional, Union

# Runs Testuary's command line as ``python -m testuary`` does, without the
# working directory on the module path. Its first argument is the directory this
# Testuary was imported from. That directory goes first on the module path
# only when Python does not search it already, so an installed package's
# directory stays after the standard library.
_COMMAND = ("import os, sys; root = sys.argv.pop(1); "
            "sys.path[:] = [p for p in sys.path if p not in ('', '.')]; "
            "same = lambda p: os.path.normcase(os.path.realpath(p)); "
            "sys.path[:0] = [] if any(same(p) == same(root) for p in sys.path) else [root]; "
            "from testuary.__main__ import main; sys.exit(main())")


def play(language: str = "python", *, output_dir: Union[str, os.PathLike],
         data: Optional[str] = None, speed: Union[str, float] = "auto",
         duration: Optional[float] = None) -> int:
    """Play the offline demonstration as ``testuary --demo`` does and return
    its exit status (0 when it completed, 130 when Ctrl+C stopped it).

    ``language`` selects a language's demonstration, or ``data`` names
    demonstration data in its place (as ``--demo-data``,
    without ``language``). ``output_dir`` must be new or empty. ``speed`` is
    ``auto``, ``instant`` or a factor of the configured length's speed, such
    as 2 for half as long (``--demo-speed``), and ``duration`` the paced
    playback's length in seconds (``--demo-playback-seconds``). A demonstration must
    start before any part of a run is loaded, so each call plays in a Python
    process of its own: a script can play several, one after the other, and
    Ctrl+C reaches the playback, which ends with its own cleanup."""
    arguments = ["--demo", "--output-dir", os.fspath(output_dir), "--demo-speed", str(speed)]
    if data is not None:
        if language != "python":
            raise ValueError("data names the demonstration's language. Leave language out")
        arguments += ["--demo-data", data]
    else:
        arguments += ["--demo-language", language]
    if duration is not None:
        arguments += ["--demo-playback-seconds", str(duration)]
    # The playback imports this Testuary, wherever it was imported from.
    package_root = str(Path(__file__).resolve().parents[2])
    process = subprocess.Popen([sys.executable, "-c", _COMMAND, package_root, *arguments])
    while True:
        try:
            return process.wait()
        except KeyboardInterrupt:
            # The playback received the same Ctrl+C and is cleaning up.
            continue
