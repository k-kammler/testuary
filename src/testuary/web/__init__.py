"""testuary-web: a local browser interface for Testuary, started for a
directory by the ``testuary-web`` command or by ``serve``.

It shows what Testuary already offers and holds no rules of its own: every
value comes from Testuary's public interfaces and documented files, and every
action starts an existing Testuary command.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple, Union, Sequence
from testuary import __version__
import _thread
import argparse
import contextlib
import errno
import getpass
import http.client
import json
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path
from testuary.web import _jobs
# The web extension point's public names (docs/extending/web.md).
from testuary.web.extensions import TOOL_CHECK_LABEL as TOOL_CHECK_LABEL
from testuary.web.extensions import WEB_EXTENSION_API as WEB_EXTENSION_API
from testuary.web.extensions import DirType as DirType
from testuary.web.extensions import JobKind as JobKind
from testuary.web.extensions import NestedRun as NestedRun
from testuary.web.extensions import SettingsGroup as SettingsGroup
from testuary.web.extensions import WebExtension as WebExtension
from testuary.web.extensions import WebRoute as WebRoute
from testuary.web.extensions import WebTab as WebTab
from testuary.settings import _CHOICES
from testuary.web import _server


def serve(base_dir: Optional[Union[str, os.PathLike]] = None, *, port: Optional[int] = None,
          access: Optional[str] = None, open_browser: bool = True) -> int:
    """Start the web interface for ``base_dir`` as ``testuary-web`` does and
    return its exit status (0 when Ctrl+C ended it, 2 when ``base_dir`` is no
    directory, ``port`` cannot be used, ``access`` is none of the values
    below or the installed Testuary does not answer in time).

    ``base_dir`` is the directory Testuary works in (default: the current
    directory). ``port`` is the port on 127.0.0.1 (default: [web] port of
    Testuary's settings, 8765, and 0 takes a free one), ``access`` who may
    use the interface (default: [web] access, open): ``open``, every
    program on this computer that reaches the port, ``key``, only a page
    opened with the printed address, whose access key is new at each start,
    or ``stable_key``, the same with one key kept in Testuary's user
    directory. ``open_browser`` opens the printed address in the browser.
    In an SSH login (SSH_CONNECTION set) it opens none and prints the ssh -L
    line for the user's own computer instead. The call
    blocks until Ctrl+C, a SIGTERM or Ctrl+Break ends it. A job that is
    still running then stops first, as Ctrl+C stops a run, and keeps the
    work that finished. Call it from the main thread, which receives
    Ctrl+C."""

    return _start(None if base_dir is None else os.fspath(base_dir), port=port, access=access,
                  open_browser=open_browser)


class _Unavailable(Exception):
    """A start that cannot go ahead, with the message that says why."""


def _access_key(access: str) -> Optional[str]:
    """The access key of a start: none when ``open``, a new one for ``key``,
    for ``stable_key`` the one kept in Testuary's user directory
    (``web/access-key``, readable by this user only), made when there is
    none or it is damaged."""
    if access == "open":
        return None
    if access == "key":
        return secrets.token_urlsafe(32)
    from testuary.settings import file_locations

    stored = file_locations().user / "web" / "access-key"
    try:
        kept = stored.read_text(encoding="ascii").strip()
    except (OSError, UnicodeError):
        kept = ""
    if re.fullmatch(r"[A-Za-z0-9_-]{32,256}", kept):
        return kept
    key = secrets.token_urlsafe(32)
    stored.parent.mkdir(parents=True, exist_ok=True)
    written = stored.with_name(f"{stored.name}.{secrets.token_hex(4)}.tmp")
    stream = os.fdopen(os.open(written, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w",
                       encoding="ascii")
    try:
        with stream:
            stream.write(key + "\n")
        os.replace(written, stored)
    finally:
        with contextlib.suppress(OSError):
            written.unlink(missing_ok=True)
    return key


def _address(port: int, key: Optional[str]) -> str:
    """The address a start prints: plain when open, else with the access key
    in its fragment, which the browser sends to no server."""
    return f"http://127.0.0.1:{port}/" + (f"#token={key}" if key else "")


def _web_settings(base_dir: Path) -> Tuple[int, str]:
    """[web] port and access as Testuary resolves them for runs in
    ``base_dir``. It raises ValueError or OSError when its settings cannot be
    read."""
    from testuary.settings import resolve_settings

    values = resolve_settings(base_dir=base_dir).sections["web"]
    return int(values["port"]), str(values["access"])


def _chosen(base_dir: Path, *, port: Optional[int], access: Optional[str]) -> Tuple[int, str]:
    """The port and access of a start: the options given, else the
    settings. Settings that cannot be read need both options, and an
    access [web] access does not offer is refused."""
    if port is None or access is None:
        try:
            configured_port, configured_access = _web_settings(base_dir)
        except (ValueError, OSError) as exc:
            raise _Unavailable(f"Testuary's settings cannot be read ({exc}): fix them, or start "
                               "with both --port and --access") from None
        port = configured_port if port is None else port
        access = configured_access if access is None else access
    if access not in _CHOICES["access"]:
        raise _Unavailable(f"access: choose one of {', '.join(_CHOICES['access'])}")
    return port, access


def _listen(app: _server.App, port: int) -> _server.Server:
    """The server of ``app`` on ``port`` of 127.0.0.1. It raises
    _Unavailable, saying what to do, when that port cannot be used."""
    try:
        return _server.Server(app, port)
    except OSError as exc:
        if exc.errno == errno.EADDRINUSE:
            mine = [record for _, record in _records(port) if _runs(record.get("pid"))]
            if mine:
                raise _Unavailable(
                    f"port {port} on 127.0.0.1 is taken by your testuary-web for "
                    f"{mine[0].get('base_dir')} (process {mine[0].get('pid')}): start this one with "
                    f"--replace to stop that one first, stop it with testuary-web --stop --port "
                    f"{port}, or start this one on another port with --port PORT") from None
            if _testuary_web_answers(port):
                raise _Unavailable(
                    f"port {port} on 127.0.0.1 is taken by a testuary-web of another Testuary user "
                    "directory (TESTUARY_HOME) or of another user: end it with Ctrl+C where it runs, "
                    "or with --replace or --stop under its TESTUARY_HOME, or start this one on "
                    "another port with --port PORT") from None
            raise _Unavailable(
                f"port {port} on 127.0.0.1 is taken by a program that is not your testuary-web: "
                "start this one on another port with --port PORT (or [web] port in Testuary's "
                "config.ini)") from None
        raise _Unavailable(f"cannot listen on 127.0.0.1:{port} ({exc}): choose another port "
                           "with --port PORT") from None


def _testuary_web_answers(port: int) -> bool:
    """Whether a testuary-web answers on ``port``, one without a record in
    this user directory: every response of its server names it in its Server
    header, a refusal's too."""
    try:
        connection = http.client.HTTPConnection("127.0.0.1", port, timeout=3)
        try:
            connection.request("GET", "/api/state", headers={"Host": f"127.0.0.1:{port}"})
            server = connection.getresponse().getheader("Server") or ""
        finally:
            connection.close()
    except (OSError, http.client.HTTPException):
        return False
    return server.split()[:1] == [_server.Handler.server_version]


#: How long --stop and --replace wait for an interface to end: a running job
#: ends first, as Ctrl+C ends it, keeping its finished work.
_STOP_SECONDS = 600


def _running_dir() -> Path:
    """Where this user's running interfaces record themselves."""
    from testuary.settings import file_locations

    return file_locations().user / "web" / "running"


def _runs(pid) -> bool:
    return isinstance(pid, int) and pid > 0 and _jobs._alive(pid)


def _record_start(port: int, base_dir: Path, secret: str) -> Path:
    """Record this start in the user's directory, readable by this user only:
    its process, port, directory, start time and stop secret."""
    path = _running_dir() / f"{port}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    written = path.with_name(f"{path.name}.{os.getpid()}.tmp")
    written.unlink(missing_ok=True)
    record = {"pid": os.getpid(), "port": port, "base_dir": str(base_dir), "started": time.time(),
              "secret": secret}
    with os.fdopen(os.open(written, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600), "w",
                   encoding="utf-8") as stream:
        json.dump(record, stream)
    os.replace(written, path)
    return path


def _remove_record(path: Path, secret: str) -> None:
    """Remove this start's record, unless a later start wrote its own there."""
    try:
        if json.loads(path.read_text(encoding="utf-8")).get("secret") == secret:
            path.unlink()
    except (OSError, ValueError, AttributeError):
        pass


def _records(port: Optional[int] = None) -> List[Tuple[Path, dict]]:
    """This user's records of running interfaces, of ``port`` only when given."""
    running_dir = _running_dir()
    paths = [running_dir / f"{port}.json"] if port is not None else sorted(running_dir.glob("*.json"))
    found = []
    for path in paths:
        try:
            record = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if isinstance(record, dict):
            found.append((path, record))
    return found


def _ask_stop(port, secret) -> Optional[int]:
    """The status a stop request with ``secret`` gets on ``port``, None when
    nothing answers there."""
    try:
        connection = http.client.HTTPConnection("127.0.0.1", int(port), timeout=10)
        try:
            connection.request("POST", "/api/stop", body=b"{}", headers={
                "Host": f"127.0.0.1:{port}", "Content-Type": "application/json",
                _server.STOP_HEADER: str(secret)})
            return connection.getresponse().status
        finally:
            connection.close()
    except (OSError, ValueError, http.client.HTTPException):
        return None


def _stop_running(port: Optional[int] = None) -> Tuple[List[str], bool]:
    """Stop this user's running interfaces (``port``'s only, when given),
    each by the stop secret of its record, as Ctrl+C ends it. It returns the
    lines that say what happened and whether an interface did not end. A
    record whose interface no longer runs is removed. No process is
    signaled by its number, so no other program is touched."""
    records = _records(port)
    if not records:
        return [f"No testuary-web of yours runs{f' on port {port}' if port is not None else ''}."], False
    lines, failed = [], False
    for path, record in records:
        pid, where = record.get("pid"), record.get("port")
        named = f"testuary-web on port {where} (directory {record.get('base_dir')}, process {pid})"
        status = _ask_stop(where, record.get("secret"))
        if status == 202:
            deadline = time.monotonic() + _STOP_SECONDS
            while path.exists() and _runs(pid) and time.monotonic() < deadline:
                time.sleep(0.2)
            ended = not (path.exists() and _runs(pid))
            failed = failed or not ended
            lines.append(f"Stopped {named}, as Ctrl+C stops it." if ended
                         else f"Asked {named} to stop, but it has not ended within {_STOP_SECONDS} s.")
        elif not _runs(pid):
            path.unlink(missing_ok=True)
            lines.append(f"Removed the record of a testuary-web on port {where} that no longer runs.")
        else:
            failed = True
            lines.append(f"{named} did not take the stop request"
                         f"{f' (it answered {status})' if status else ''}: end it with Ctrl+C where it runs.")
    return lines, failed


def _testuary_version(python: str) -> str:
    """The installed Testuary's version line, from its public command line.
    A Testuary that does not answer in time raises _Unavailable."""
    # No color codes enter the line.
    environment = {name: value for name, value in os.environ.items() if name != "FORCE_COLOR"}
    environment.update(NO_COLOR="1", PYTHON_COLORS="0")
    seconds = 300
    try:
        done = subprocess.run(_jobs.module_command(python, "testuary", "--version"),
                              capture_output=True, text=True, encoding="utf-8",
                              errors="replace", timeout=seconds, env=environment)
    except subprocess.TimeoutExpired:
        raise _Unavailable(f"Testuary did not answer 'testuary --version' within {seconds} "
                           f"seconds: check that it runs in {python}, then start again") from None
    lines = [line.strip() for line in (done.stdout or done.stderr).splitlines() if line.strip()]
    return ", ".join(lines) or "unknown"


def _interrupt(signum, frame) -> None:
    raise KeyboardInterrupt


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Start the web interface as the ``testuary-web`` command does, with the
    options ``argv`` (the command line when None), and return its exit
    status."""
    from testuary.settings import _OptionParser

    parser = _OptionParser(
        prog="testuary-web", allow_abbrev=False,
        description="Start a local browser interface for Testuary in this directory. It "
                    "listens on 127.0.0.1 only.",
        epilog="Exit status: 0 when the interface is stopped, or when --stop ends every "
               "interface it finds, 1 when --stop finds one that does not end, 2 when the "
               "interface cannot start or an option is refused.")
    parser.add_argument("--port", type=_port, default=None,
                        help="The port on 127.0.0.1 (default: [web] port of Testuary's "
                             "settings). 0 takes a free one.")
    parser.add_argument("--access", choices=_CHOICES["access"], default=None,
                        help="Who may use the interface (default: [web] access). open: every "
                             "program on this computer that reaches the port. key: only a page "
                             "opened with the printed address, whose key is new at each start. "
                             "stable_key: the same, with one key kept in Testuary's user directory.")
    parser.add_argument("--no-browser", action="store_true",
                        help="Print the address without opening a browser.")
    parser.add_argument("--dir", type=Path, default=None,
                        help="The directory Testuary works in: its settings files, and where "
                             "demonstrations go (default: the current directory). A run started "
                             "from the pages writes by default into its project's results "
                             "directory.")
    parser.add_argument("--replace", action="store_true",
                        help="First stop your testuary-web running on this port, as Ctrl+C stops it "
                             "(a running job ends keeping its finished work), then start.")
    parser.add_argument("--stop", action="store_true",
                        help="Stop your running testuary-web interfaces, with --port only that "
                             "port's, as Ctrl+C stops them, and start none. Another program on a "
                             "port is never touched. It exits with 1 when one of them did not end.")
    parser.add_argument("--version", action="version", version=f"testuary-web {__version__}")
    options = parser.parse_args(argv)
    if options.stop:
        lines, failed = _stop_running(options.port)
        for line in lines:
            print(line)
        return 1 if failed else 0
    return _start(options.dir, port=options.port, access=options.access,
                  open_browser=not options.no_browser, replace=options.replace)


def _port(text: str) -> int:
    try:
        number = int(text)
    except ValueError:
        number = -1
    if not 0 <= number <= 65535:
        raise argparse.ArgumentTypeError(f"expected a port from 0 to 65535, not {text}")
    return number


def _tunnel_line(port: int) -> Optional[str]:
    """In an SSH login, the ssh -L line that brings ``port`` on 127.0.0.1
    here to the same port on the user's own computer, with the address,
    SSH port and login this login came in on (SSH_CONNECTION: client
    address and port, server address and port), or None outside SSH."""
    parts = os.environ.get("SSH_CONNECTION", "").split()
    if not parts:
        return None
    host, ssh_port = (parts[2], parts[3]) if len(parts) == 4 else ("THE-MACHINE", "22")
    return (f"ssh -L {port}:127.0.0.1:{port}" + ("" if ssh_port == "22" else f" -p {ssh_port}")
            + f" {getpass.getuser()}@{host}")


def _start(base_dir, *, port: Optional[int] = None, access: Optional[str] = None,
           open_browser: bool = True, replace: bool = False) -> int:
    """The interface for ``base_dir`` (default: the current directory) until
    Ctrl+C, as ``testuary-web`` and ``testuary.web.serve`` start it.
    ``replace`` first stops this user's interface on that port."""
    base_dir = (Path(base_dir) if base_dir is not None else Path.cwd()).resolve()
    if not base_dir.is_dir():
        print(f"testuary-web: {base_dir} is not a directory", file=sys.stderr)
        return 2
    try:
        port, access = _chosen(base_dir, port=port, access=access)
        if replace and port:
            for line in _stop_running(port)[0]:
                print(line, flush=True)
        version = _testuary_version(sys.executable)
        app = _server.App(base_dir, testuary_version=version, python=sys.executable,
                          token=_access_key(access))
        server = _listen(app, port)
    except _Unavailable as refused:
        print(f"testuary-web: {refused}", file=sys.stderr)
        return 2
    opened = _address(server.server_address[1], app.token)
    threading.Thread(target=server.serve_forever, name="testuary-web", daemon=True).start()
    # SIGTERM, SIGHUP and Ctrl+Break end the interface as Ctrl+C does, from
    # before it says how to stop it. Their handlers come back afterwards for
    # a caller of testuary.web.serve. A handler not set from Python reads
    # None and cannot be set back, so its signal stays as it is.
    replaced = {}
    for name in ("SIGTERM", "SIGHUP", "SIGBREAK"):
        if hasattr(signal, name):
            number = getattr(signal, name)
            previous = signal.getsignal(number)
            if previous is None:
                continue
            replaced[number] = previous
            signal.signal(number, _interrupt)
    record = None
    try:
        try:
            # testuary-web --stop and --replace find this start by its record
            # and end it by its secret, as Ctrl+C would: through SIGTERM's
            # handler where it is ours, since a start in the background of a
            # script ignores SIGINT.
            signum = signal.SIGTERM if signal.SIGTERM in replaced else signal.SIGINT
            app.stopper = lambda: _thread.interrupt_main(signum)
            try:
                record = _record_start(server.server_address[1], base_dir, app.stop_secret)
            except OSError as exc:
                print(f"testuary-web: this start could not record itself ({exc}), so "
                      "testuary-web --stop will not find it.")
            print(f"Testuary web interface for {base_dir}")
            print(f"Installed: {version}")
            print(f"Open: {opened}")
            if app.token is None:
                print("Every program on this computer that reaches this port can use it. "
                      "--access key admits only the address above.")
            # In an SSH login the browser is on the user's own computer, and
            # one opened here could be a text browser that takes over the
            # terminal. The address SSH_CONNECTION names is the one the
            # connection arrived at, which behind NAT or on a cloud machine
            # the user's computer cannot reach.
            tunnel = _tunnel_line(server.server_address[1])
            if tunnel:
                print(f"Logged in over SSH: on your own computer run  {tunnel} (or the name you log "
                      "in with)  then open the address above.")
            print("Ctrl+C stops the interface.", flush=True)
            if open_browser and not tunnel:
                webbrowser.open(opened)
            while True:
                time.sleep(0.5)
        except KeyboardInterrupt:
            pass
        job = app.jobs.active()
        if job is not None:
            _jobs._say(f"Stopping job {job.id} as Ctrl+C would. It keeps the work that finished. "
                 "Ctrl+C again ends it at once.")
            app.jobs.stop(job.id)
            try:
                while job.ended is None:
                    time.sleep(0.2)
            except KeyboardInterrupt:
                app.jobs.kill(job)
                while job.ended is None:
                    time.sleep(0.2)
        server.shutdown()
        server.server_close()
        _jobs._say("Stopped.")
        return 0
    finally:
        if record is not None:
            _remove_record(record, app.stop_secret)
        for number, handler in replaced.items():
            signal.signal(number, handler)
