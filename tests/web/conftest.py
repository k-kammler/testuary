"""Shared test helpers: a server on a free port, a small HTTP client that
controls every header, and a stand-in ``testuary`` command for job tests."""
from __future__ import annotations

import http.client
import json
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

import pytest

from testuary.web._server import App, Server

SOURCE = Path(__file__).resolve().parents[2] / "src"

STAND_IN = '''\
"""A stand-in for Testuary's command line, for the job tests."""
import json
import os
import signal
import subprocess
import sys
import time

arguments = sys.argv[1:]
if "--version" in arguments:
    print("stand-in", flush=True)
    raise SystemExit(0)
if "--events" in arguments:
    with open(arguments[arguments.index("--events") + 1], "w", encoding="utf-8") as events:
        events.write(json.dumps({"schema_version": 2, "testuary": "stand-in"}) + "\\n")
        events.write(json.dumps({"event": {
            "run_id": "r", "target": "all targets", "phase": "pipeline", "operation": "run",
            "state": "running", "elapsed_s": 0.0, "message": "stand-in", "level": "run"}}) + "\\n")
mode = os.environ.get("STAND_IN_MODE", "exit0")
if mode == "importstop":
    # Stopped while Testuary still starts: the interrupt comes before its
    # own Ctrl+C handling.
    raise KeyboardInterrupt
print("stand-in started", flush=True)
if mode == "incomplete" and "--events" in arguments:
    # A run whose one file got no suite that meets the minimums: its end-of-run summary and
    # its last event, as Testuary writes them, then status 0.
    with open(arguments[arguments.index("--events") + 1], "a", encoding="utf-8") as events:
        events.write(json.dumps({"completion": {
            "files": [{"path": "calc/ops.py", "outcome": "no_complete_suite", "reason": "no tests"}],
            "results_directory": "out", "resume_command": "testuary --resume"}}) + "\\n")
        events.write(json.dumps({"event": {
            "run_id": "r", "target": "all targets", "phase": "pipeline", "operation": "run",
            "state": "failed", "failure": "outcome", "elapsed_s": 1.0, "level": "run",
            "message": "Run finished: 0 of 1 target with a suite that meets the minimums, 1 without one."}}) + "\\n")
    raise SystemExit(0)
if mode in ("measured", "setup") and "--events" in arguments:
    # A run's setup report, as Testuary writes it before its work; then
    # (measured) its end-of-run summary with one file's values and its
    # run directory, or (setup) a run still under way.
    output = os.path.join(os.getcwd(), "out", "fake-2")
    path = arguments[arguments.index("--events") + 1]

    def metric(name, value):
        return {"name": name, "value": value, "unit": "%", "basis": "measured", "goal": None}

    with open(path, "a", encoding="utf-8") as events:
        events.write(json.dumps({"readiness": {"status": "ready", "exit_code": 0, "items": [],
                                               "model_alias": "fake", "output_dir": output}}) + "\\n")
    if mode == "measured":
        os.makedirs(os.path.join(output, "logs"), exist_ok=True)
        with open(path, "a", encoding="utf-8") as events:
            events.write(json.dumps({"completion": {
                "files": [{"path": "calc/ops.py", "outcome": "complete", "reason": "",
                           "metrics": [metric("Line coverage", 100.0), metric("Branch coverage", 100.0),
                                       metric("Mutation score", 91.8919), metric("Mutation score", None)]}],
                "results_directory": os.path.join(output, "results"), "resume_command": None}}) + "\\n")
            events.write(json.dumps({"event": {
                "run_id": "r", "target": "all targets", "phase": "pipeline", "operation": "run",
                "state": "succeeded", "elapsed_s": 1.0, "level": "run", "message": "Run finished",
                "log_path": os.path.join(output, "logs", "run.log")}}) + "\\n")
        raise SystemExit(0)
# Runs Testuary refuses before their work: a configuration, before any
# progress, and (noresults) a run whose required tools are missing,
# as a project interpreter that does not exist makes them, before its run
# directory exists. Each one's report item: key, status, reason, fix.
REFUSALS = {
    "refused": ("configuration", "refused", "[coverage] goal: expected a percentage from 0 to 100",
                "Correct it, then run the command again."),
    "noresults": ("native.execution", "missing_setup",
                  "required tools are missing for this run:\\n  [python] capability report "
                  "failed (RuntimeError: [project] python_executable: points at C:/nope/python.exe, "
                  "which does not exist)\\nNo phase ran.",
                  "Install what is missing or change the settings, then run the command again."),
}
if mode in REFUSALS and "--events" in arguments:
    # Its setup report in the event file, the console's lines, status 2.
    key, status, reason, action = REFUSALS[mode]
    with open(arguments[arguments.index("--events") + 1], "a", encoding="utf-8") as events:
        events.write(json.dumps({"readiness": {
            "scope": "inspection", "status": "refused", "exit_code": 2, "source_dir": None,
            "execution": None, "source_files": [], "phases": [], "model_alias": None, "provider": None,
            "model": None, "credentials_configured": None, "output_dir": None,
            "output_mode": "fresh", "limits": [], "items": [{
                "key": key, "required": True, "status": status, "selected": None,
                "version": None, "reason": reason, "action": action, "diagnostic_path": None,
                "tried_by": None}]}}) + "\\n")
    print("Setup before the run: 1 problem found.", file=sys.stderr, flush=True)
    raise SystemExit(2)
if mode == "results":
    # A run that writes its run directory, named by --output-dir.
    given = next(item.split("=", 1)[1] for item in arguments if item.startswith("--output-dir="))
    os.makedirs(os.path.join(given, "results"), exist_ok=True)
    raise SystemExit(1)
if mode.startswith("exit"):
    raise SystemExit(int(mode[4:]))
if mode == "stubborn":
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    # A command in a session of its own, as LangAdapter starts its commands.
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True)
    print(f"stand-in child {child.pid}", flush=True)
try:
    while True:
        time.sleep(0.05)
except KeyboardInterrupt:
    if mode == "early":
        # Stopped while Python still starts: a traceback and the system's
        # status, not Testuary's 130.
        print("Traceback (most recent call last):", file=sys.stderr, flush=True)
        print("KeyboardInterrupt", file=sys.stderr, flush=True)
        raise SystemExit(3)
    print("stand-in interrupted", flush=True)
    raise SystemExit(130)
'''


class Client:
    def __init__(self, server: Server) -> None:
        self.port = server.server_address[1]
        self.token = None

    def request(self, method, path, body=None, headers=None, host=None, token=True, expect=None):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=120)
        sent = {"Host": host or f"127.0.0.1:{self.port}"}
        if token and self.token:
            sent["X-Testuary-Web-Token"] = self.token
        data = None
        if body is not None:
            data = json.dumps(body).encode("utf-8")
            sent["Content-Type"] = "application/json"
        sent.update(headers or {})
        connection.request(method, path, body=data, headers=sent)
        response = connection.getresponse()
        try:
            # An answer whose status is not ``expect`` fails from its head,
            # before a body that may stream on is read.
            assert expect is None or response.status == expect, (path, response.status)
            payload = response.read()
        finally:
            connection.close()
        return response.status, {k.lower(): v for k, v in response.getheaders()}, payload

    def login(self, token):
        """Use the token as the page does: in the header of every request."""
        self.token = token

    def json(self, method, path, body=None, **options):
        status, _, payload = self.request(method, path, body, **options)
        return status, json.loads(payload)

    def stream(self, job_id):
        """The job's server-sent events, read until the server ends them."""
        return self.events(f"/api/jobs/{job_id}/stream")

    def events(self, path):
        """The server-sent events ``path`` answers, read until the server
        ends them."""
        status, _, payload = self.request("GET", path)
        assert status == 200, payload
        events = []
        for block in payload.decode("utf-8").split("\n\n"):
            fields = dict(line.split(": ", 1) for line in block.splitlines()
                          if ": " in line and not line.startswith(":"))
            if "event" in fields:
                events.append((fields["event"], json.loads(fields["data"])))
        return events, payload


@pytest.fixture(autouse=True)
def user_dir(tmp_path_factory, monkeypatch):
    """Testuary's user directory of the test: its models.ini, its global
    config.ini and the interface's recent projects, never the user's own,
    and outside the test's own directory, which some tests list."""
    user_dir = tmp_path_factory.mktemp("user")
    monkeypatch.setenv("TESTUARY_HOME", str(user_dir))
    try:
        from testuary import settings
    except ImportError:
        return user_dir
    if hasattr(settings, "platform_user_dir"):
        monkeypatch.setattr(settings, "platform_user_dir", lambda: user_dir)
    return user_dir


@pytest.fixture
def serve(tmp_path):
    """Start a web interface for a directory; yields (app, server, client).
    ``key`` starts it with a per-start access key, which the client sends,
    as testuary-web --access key does; without, it is open."""
    started = []

    def start(base_dir=None, *, python=sys.executable, key=True, extensions=None):
        base_dir = Path(base_dir or tmp_path / "work")
        base_dir.mkdir(parents=True, exist_ok=True)
        app = App(base_dir, testuary_version="test",
                  python=python,
                  token=secrets.token_urlsafe(32) if key else None)
        if extensions is not None:
            app.extensions = extensions
        server = Server(app, 0)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        started.append(server)
        client = Client(server)
        client.login(app.token)
        return app, server, client

    yield start
    for server in started:
        job = server.app.jobs.active()
        if job is not None:
            server.app.jobs.kill(job)
        server.shutdown()
        server.server_close()


@pytest.fixture
def stand_in(tmp_path, monkeypatch):
    """Put a stand-in ``testuary`` package first on the children's path. Its
    command line is the stand-in's, and the real package's modules, the web
    interface among them, stay importable beside it."""
    package = tmp_path / "stand-in" / "testuary"
    package.mkdir(parents=True)
    real = SOURCE / "testuary"
    (package / "__init__.py").write_bytes((
        f"__path__.append({str(real)!r})\n"
        f"exec(compile(open({str(real / '__init__.py')!r}, encoding='utf-8').read(),"
        f" {str(real / '__init__.py')!r}, 'exec'))\n").encode("utf-8"))
    (package / "__main__.py").write_text(STAND_IN, encoding="utf-8")
    paths = [str(package.parent), str(SOURCE)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))
    return monkeypatch


def page_value(names, expression, before=""):
    """The page's definitions ``names``, as app.js has them, after the stubs
    ``before``, then the value of ``expression``, awaited when it is a
    promise, evaluated by Node."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    parts = []
    for name in names:
        found = (re.search(rf"^const {name} = \{{\n.*?^\}};\n", source, re.S | re.M)
                 or re.search(rf"^const {name} = .*;\n", source, re.M)
                 or re.search(rf"^(?:async )?function {name}\(.*?^\}}\n", source, re.S | re.M))
        assert found, f"app.js defines {name}"
        parts.append(found.group(0))
    script = (before + "".join(parts)
              + f"Promise.resolve({expression}).then((value) => process.stdout.write(JSON.stringify(value)));")
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def wait_for(condition, timeout=60.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False
