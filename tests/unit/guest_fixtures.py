"""A controlled container for adapter tests of guest execution.

When the settings name a guest control helper, guest commands take the
backend's supervised route (``_linux_container.run``). This double serves it
without a container, mapping the container's source directory and work directory to
host directories. It runs preparation commands (directory test, link listing,
copy, receipt read, removal) on those directories and hands the rest to the test.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path, PurePosixPath

# No public seam serves guest commands without a container, so the double replaces the private route.
from langadapter import _linux_container

#: The guest control helper the settings name.
SUPERVISOR = "/control/python3"


def container_work_dir(work_dir: Path) -> str:
    """Where the backend mounts the work directory inside the container: a Windows
    path cannot be a container path, so it is the fixed /workzone there."""
    return "/workzone" if os.name == "nt" else work_dir.as_posix()


def controlled_guest(monkeypatch, *, container, root, source, work_dir, handle):
    """Serve guest commands for ``container`` with ``root`` (a container
    path) held in the host directory ``source``. ``handle(args, options, host)``
    answers every command but preparation's, with ``host`` mapping a
    container path to its host path. Returns the list of commands seen."""
    mount = PurePosixPath(container_work_dir(work_dir))
    calls = []

    def host(path):
        path = PurePosixPath(path)
        if path == PurePosixPath(root) or PurePosixPath(root) in path.parents:
            return source / path.relative_to(root)
        return work_dir / path.relative_to(mount)

    def run(executor, argv, *, helper_python, **options):
        assert helper_python == SUPERVISOR and executor.container == container
        args = [executor._to_container(str(arg)) for arg in argv]
        calls.append(args)
        output, code = "", 0
        if args[0] == "test" and args[1] == "-d":
            code = 0 if host(args[2]).is_dir() else 1
        elif args[0] == "find" and args[1] == "-P":
            # No links in the controlled project.
            pass
        elif args[0] == "cp" and args[1:4] == ["-R", "-P", "--"]:
            shutil.copytree(host(args[-2]), host(args[-1]), symlinks=True)
        elif args[0] == "cat":
            output = host(args[1]).read_text(encoding="utf-8")
        elif args[0] == "rm" and args[1:3] == ["-rf", "--"]:
            path = host(args[-1])
            assert path.resolve().is_relative_to(work_dir.resolve())
            if path.exists():
                shutil.rmtree(path)
        else:
            # The guest sees forwarded values at their container paths.
            env = options.get("env")
            if env is not None:
                options = dict(options, env={key: executor._to_container(str(value))
                                             for key, value in env.items()})
            return handle(args, options, host)
        return subprocess.CompletedProcess(args, code, output, "")

    monkeypatch.setenv("TESTUARY_PROJECT_WORK_DIR", str(work_dir))
    monkeypatch.setenv("TESTUARY_PROJECT_CONTAINER", container)
    monkeypatch.setenv("LANGADAPTER_HELPER_PYTHON", SUPERVISOR)
    monkeypatch.setattr(_linux_container, "run", run)
    return calls
