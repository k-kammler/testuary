"""testuary-web --stop and --replace: each start records itself in
this user's Testuary directory (web/running/<port>.json: its process, port,
directory, start time and a stop secret) and removes the record when it ends.
--stop asks this user's running interfaces, by that secret, to end as
Ctrl+C ends them, and --replace does so on the chosen port, then starts. No
process is signaled by its number, so another program, also one on the
port, is never touched."""
from __future__ import annotations

import json
import os
import signal
import socket
import subprocess
import sys

from conftest import wait_for
from testuary.web import _jobs

STOP = signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _web(*arguments, base_dir, home=None):
    """testuary-web with ``arguments`` as a process of its own (with
    ``home`` as its Testuary user directory, else the test's), read until it
    says how its start went (or until it ends): the process and its lines."""
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
               else {"start_new_session": True})
    if home is not None:
        options["env"] = dict(os.environ, TESTUARY_HOME=str(home))
    process = subprocess.Popen(
        _jobs.module_command(sys.executable, "testuary.web", "--no-browser", "--dir", str(base_dir),
                            *arguments),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **options)
    lines = []
    for line in process.stdout:
        lines.append(line.rstrip())
        if line.startswith(("Ctrl+C", "testuary-web:")):
            break
    return process, lines


def _record(user_dir, port):
    return user_dir / "web" / "running" / f"{port}.json"


def _end(process):
    process.send_signal(STOP)
    assert process.wait(120) == 0


def test_a_start_records_itself_and_its_end_removes_the_record(tmp_path, user_dir):
    """The record names the interface's own process (on Windows a venv's
    python.exe is a launcher whose child that is), which ends with it."""
    port = _free_port()
    process, lines = _web("--port", str(port), "--access", "open", base_dir=tmp_path)
    try:
        assert lines[-1].startswith("Ctrl+C"), lines
        record = json.loads(_record(user_dir, port).read_text(encoding="utf-8"))
        assert (record["port"], record["base_dir"]) == (port, str(tmp_path.resolve()))
        assert _jobs._alive(record["pid"])
        assert len(record["secret"]) >= 32 and record["started"] > 0
        if os.name == "posix":
            assert _record(user_dir, port).stat().st_mode & 0o077 == 0
    finally:
        _end(process)
    assert not _record(user_dir, port).exists()
    assert wait_for(lambda: not _jobs._alive(record["pid"]), timeout=30)


def test_stop_ends_this_users_interface_on_its_port_as_ctrl_c_does(tmp_path, user_dir):
    port = _free_port()
    running, lines = _web("--port", str(port), "--access", "open", base_dir=tmp_path)
    assert lines[-1].startswith("Ctrl+C"), lines
    stopped = subprocess.run(_jobs.module_command(sys.executable, "testuary.web", "--stop", "--port", str(port)),
                             capture_output=True, text=True, timeout=300)
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    assert f"Stopped testuary-web on port {port}" in stopped.stdout
    assert str(tmp_path.resolve()) in stopped.stdout
    assert running.wait(120) == 0
    assert "Stopped." in running.stdout.read()
    assert not _record(user_dir, port).exists()


def test_replace_stops_the_running_one_on_the_port_then_starts(tmp_path, user_dir):
    port = _free_port()
    first, lines = _web("--port", str(port), "--access", "open", base_dir=tmp_path)
    assert lines[-1].startswith("Ctrl+C"), lines
    replaced = json.loads(_record(user_dir, port).read_text(encoding="utf-8"))["pid"]
    second, lines = _web("--port", str(port), "--access", "open", "--replace", base_dir=tmp_path)
    try:
        assert any(line.startswith(f"Stopped testuary-web on port {port}") for line in lines), lines
        assert any(line.startswith("Open: ") for line in lines), lines
        assert first.wait(120) == 0
        now = json.loads(_record(user_dir, port).read_text(encoding="utf-8"))["pid"]
        assert now != replaced and _jobs._alive(now) and not _jobs._alive(replaced)
    finally:
        _end(second)
    assert not _record(user_dir, port).exists()


def test_a_stale_record_is_removed_and_nothing_else_done(tmp_path, user_dir):
    gone = subprocess.Popen([sys.executable, "-c", "pass"])
    gone.wait()
    port = _free_port()
    record = _record(user_dir, port)
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"pid": gone.pid, "port": port, "base_dir": str(tmp_path), "started": 1.0,
                                  "secret": "s" * 43}), encoding="utf-8")
    stopped = subprocess.run(_jobs.module_command(sys.executable, "testuary.web", "--stop"),
                             capture_output=True, text=True, timeout=120)
    assert stopped.returncode == 0, stopped.stdout + stopped.stderr
    assert f"Removed the record of a testuary-web on port {port} that no longer runs" in stopped.stdout
    assert not record.exists()


def test_stop_exits_with_1_when_an_interface_does_not_end(tmp_path, user_dir):
    """A record whose process runs but whose port does not answer (its
    number taken by another program since) is not stopped, and --stop says
    so and exits with 1, so that a script does not go on as if it had."""
    port = _free_port()
    record = _record(user_dir, port)
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"pid": os.getpid(), "port": port, "base_dir": str(tmp_path),
                                  "started": 1.0, "secret": "s" * 43}), encoding="utf-8")
    stopped = subprocess.run(_jobs.module_command(sys.executable, "testuary.web", "--stop", "--port", str(port)),
                             capture_output=True, text=True, timeout=120)
    assert stopped.returncode == 1, stopped.stdout + stopped.stderr
    assert "did not take the stop request" in stopped.stdout
    assert record.exists()


def test_a_port_held_by_a_testuary_web_of_another_user_dir_is_named_so(tmp_path, user_dir):
    """An interface started with another Testuary user directory (TESTUARY_HOME)
    has no record in this one: the taken port's message names it as a
    testuary-web all the same, not as another program, and --replace
    leaves it running."""
    port = _free_port()
    other = tmp_path / "other-home"
    other.mkdir()
    running, lines = _web("--port", str(port), "--access", "open", base_dir=tmp_path, home=other)
    try:
        assert lines[-1].startswith("Ctrl+C"), lines
        refused, lines = _web("--port", str(port), "--access", "open", "--replace", base_dir=tmp_path)
        assert refused.wait(120) == 2
        assert lines[0] == f"No testuary-web of yours runs on port {port}.", lines
        assert "is taken by a testuary-web of another Testuary user directory" in lines[-1], lines
        assert "TESTUARY_HOME" in lines[-1] and running.poll() is None
    finally:
        _end(running)


def test_a_port_held_by_another_program_is_named_as_not_testuarys_and_left_alone(tmp_path, user_dir):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as other:
        other.bind(("127.0.0.1", 0))
        other.listen()
        port = other.getsockname()[1]
        process, lines = _web("--port", str(port), "--access", "open", base_dir=tmp_path)
        assert process.wait(120) == 2
        assert "is taken by a program that is not your testuary-web" in lines[-1], lines
        stopped = subprocess.run(
            _jobs.module_command(sys.executable, "testuary.web", "--stop", "--port", str(port)),
            capture_output=True, text=True, timeout=120)
        assert stopped.returncode == 0
        assert f"No testuary-web of yours runs on port {port}" in stopped.stdout
        with socket.create_connection(("127.0.0.1", port), timeout=10):
            pass


def test_the_stop_request_needs_the_records_secret(serve):
    """Only a request carrying the record's secret stops an interface. One
    started outside testuary-web's own start (here, the test's) is not
    stopped from outside at all."""
    app, server, client = serve(key=False)
    for secret in ("", "wrong", app.stop_secret[:-1]):
        status, refused = client.json("POST", "/api/stop", {}, headers={"X-Testuary-Web-Stop": secret})
        assert status == 403, secret
    status, refused = client.json("POST", "/api/stop", {}, headers={"X-Testuary-Web-Stop": app.stop_secret})
    assert status == 409 and "not stopped from outside" in refused["error"]
    assert wait_for(lambda: client.json("GET", "/api/state")[0] == 200, timeout=5)
