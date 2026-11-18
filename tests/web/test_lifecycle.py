"""A job stops as Ctrl+C stops a run, also when the interface ends, and one
job at a time runs in a directory across starts of the interface."""
from __future__ import annotations

import getpass
import http.client
import json
import os
import re
import signal
import subprocess
import sys
import time
import webbrowser
from urllib.parse import urlsplit

import pytest

from conftest import wait_for
from testuary.web import _jobs


def _wait_job(app, mode, stand_in):
    stand_in.setenv("STAND_IN_MODE", mode)
    job = app.jobs.start("demo", "Stand-in", ["--demo"], events=True)
    assert wait_for(lambda: job.stdout.exists()
                    and "stand-in started" in job.stdout.read_text(encoding="utf-8"))
    return job


def test_the_end_of_input_stops_a_job_as_ctrl_c_does(serve, stand_in):
    app, server, client = serve()
    job = _wait_job(app, "wait", stand_in)
    # The interface's end closes the job's input.
    job.process.stdin.close()
    assert wait_for(lambda: job.ended is not None, 60)
    assert (job.status, job.exit_code) == ("cancelled", 130)
    assert "stand-in interrupted" in job.stdout.read_text(encoding="utf-8")


def test_a_running_job_holds_the_dir(serve, stand_in, tmp_path):
    app, server, client = serve(tmp_path / "work")
    job = _wait_job(app, "wait", stand_in)
    held = json.loads(app.jobs.marker.read_text(encoding="utf-8"))
    assert (held["job"], held["pid"]) == (job.id, job.process.pid)
    other = _jobs.JobManager(tmp_path / "work")
    with pytest.raises(_jobs.Busy, match=f"process {job.process.pid}"):
        other.start("demo", "Second", ["--demo"])
    app.jobs.stop(job.id)
    assert wait_for(lambda: job.ended is not None, 60)
    assert not app.jobs.marker.exists()
    stand_in.setenv("STAND_IN_MODE", "exit0")
    second = other.start("demo", "Second", ["--demo"])
    assert wait_for(lambda: second.ended is not None) and second.exit_code == 0


def test_a_marker_of_ended_processes_is_removed(serve, stand_in, tmp_path):
    ended = subprocess.run([sys.executable, "-c", "import os; print(os.getpid())"],
                           capture_output=True, text=True)
    pid = int(ended.stdout)
    app, server, client = serve(tmp_path / "work")
    app.jobs.marker.parent.mkdir(parents=True, exist_ok=True)
    app.jobs.marker.write_text(json.dumps({"job": "old", "server": pid, "pid": pid}),
                               encoding="utf-8")
    job = _wait_job(app, "wait", stand_in)
    assert json.loads(app.jobs.marker.read_text(encoding="utf-8"))["job"] == job.id
    app.jobs.stop(job.id)
    assert wait_for(lambda: job.ended is not None, 60)


@pytest.fixture
def started():
    """The processes a test starts, ended with everything they started when
    the test ends, whatever its outcome."""
    processes = []
    yield processes
    for pid in processes:
        if not _jobs._alive(pid):
            continue
        if os.name == "nt":
            subprocess.run([_jobs._taskkill(), "/T", "/F", "/PID", str(pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            try:
                os.killpg(pid, signal.SIGKILL)
            except OSError:
                pass


def _start_interface(base_dir, stand_in, started, command=None):
    """The interface as a process of its own, with a stand-in job waiting:
    the process, the job's process id and the job's output file. ``command``
    starts it in place of ``python -m testuary.web``."""
    stand_in.setenv("STAND_IN_MODE", "wait")
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
               else {"start_new_session": True})
    process = subprocess.Popen(
        command or _jobs.module_command(sys.executable, "testuary.web", "--no-browser", "--dir",
                                       str(base_dir), "--port", "0", "--access", "key"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **options)
    started.append(process.pid)
    address = urlsplit(next(line for line in process.stdout
                            if line.startswith("Open: "))[len("Open: "):].strip())
    port, token = address.port, address.fragment[len("token="):]
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    connection.request("POST", "/api/demo", body=json.dumps({"pace": "instant"}), headers={
        "Host": f"127.0.0.1:{port}", "Content-Type": "application/json",
        "X-Testuary-Web-Token": token})
    response = connection.getresponse()
    job = json.loads(response.read())
    assert response.status == 201, job
    marker = base_dir / ".testuary-web" / "running.json"
    assert wait_for(lambda: marker.exists()
                    and json.loads(marker.read_text(encoding="utf-8")).get("pid"))
    pid = json.loads(marker.read_text(encoding="utf-8"))["pid"]
    started.append(pid)
    output = base_dir / ".testuary-web" / "jobs" / job["id"] / "stdout.txt"
    assert wait_for(lambda: output.exists() and "stand-in started" in output.read_text())
    return process, pid, output


def test_ending_the_interface_stops_its_job(stand_in, tmp_path, started):
    base_dir = tmp_path / "work"
    base_dir.mkdir()
    process, pid, output = _start_interface(base_dir, stand_in, started)
    # As a service manager ends it on Linux, or Ctrl+Break in its window.
    process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
    assert process.wait(120) == 0
    rest = process.stdout.read()
    assert "Stopping job" in rest and "Stopped." in rest
    assert not _jobs._alive(pid)
    assert "stand-in interrupted" in output.read_text(encoding="utf-8")


def test_serve_starts_the_interface_and_ends_as_ctrl_c_ends_it(stand_in, tmp_path, started):
    """testuary.web.serve, the Python call: it serves the directory, and the
    end of the interface stops its running job first and returns 0."""
    base_dir = tmp_path / "work"
    base_dir.mkdir()
    code = "\n".join(["import sys", "from testuary.web import serve",
                      "sys.exit(serve(sys.argv[1], port=0, access='key', open_browser=False))"])
    process, pid, output = _start_interface(base_dir, stand_in, started,
                                            [sys.executable, "-c", code, str(base_dir)])
    process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
    assert process.wait(120) == 0
    rest = process.stdout.read()
    assert "Stopping job" in rest and "Stopped." in rest
    assert not _jobs._alive(pid)
    assert "stand-in interrupted" in output.read_text(encoding="utf-8")


def _ignoring_ctrl_c(base_dir):
    """testuary-web as the background of a script starts it (``testuary-web
    &``): with SIGINT ignored, which a process started on Linux inherits."""
    code = "\n".join(["import signal, sys", "signal.signal(signal.SIGINT, signal.SIG_IGN)",
                      "from testuary.web import main",
                      "sys.exit(main(['--no-browser', '--dir', sys.argv[1], '--port', '0', "
                      "'--access', 'key']))"])
    return [sys.executable, "-c", code, str(base_dir)]


def test_a_job_stops_as_ctrl_c_stops_it_also_where_the_interface_ignores_ctrl_c(stand_in, tmp_path,
                                                                                  started):
    """The job of an interface that ignores SIGINT still takes the stop as
    Ctrl+C: on Linux it would otherwise inherit the ignored SIGINT, and the
    interface would wait for it without end."""
    base_dir = tmp_path / "work"
    base_dir.mkdir()
    process, pid, output = _start_interface(base_dir, stand_in, started, _ignoring_ctrl_c(base_dir))
    process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
    assert process.wait(120) == 0
    rest = process.stdout.read()
    assert "Stopping job" in rest and "Stopped." in rest
    assert not _jobs._alive(pid)
    assert "stand-in interrupted" in output.read_text(encoding="utf-8")


def test_stop_ends_the_interface_and_its_job_as_ctrl_c_does(stand_in, tmp_path, started):
    """testuary-web --stop stops a running job first, keeping its finished
    work, then the interface. So it does for one that ignores SIGINT, as one
    started in the background of a script does."""
    base_dir = tmp_path / "work"
    base_dir.mkdir()
    process, pid, output = _start_interface(base_dir, stand_in, started, _ignoring_ctrl_c(base_dir))
    stopped = subprocess.run(_jobs.module_command(sys.executable, "testuary.web", "--stop"),
                             capture_output=True, text=True, timeout=120)
    assert stopped.returncode == 0 and "Stopped testuary-web on port" in stopped.stdout, stopped
    assert process.wait(120) == 0
    rest = process.stdout.read()
    assert "Stopping job" in rest and "Stopped." in rest
    assert not _jobs._alive(pid)
    assert "stand-in interrupted" in output.read_text(encoding="utf-8")


def test_serve_gives_back_the_signal_handlers_it_replaced(tmp_path, monkeypatch):
    """After serve returns, a SIGTERM reaches the caller's own handler again,
    not the interface's."""
    import _thread
    import threading

    from testuary import web as web_app
    from testuary.web import serve

    monkeypatch.setattr(web_app, "_testuary_version", lambda python: "testuary")

    def own(signum, frame):
        pass

    def interrupt_when_serving():
        # Ctrl+C in the main thread once the interface has its handlers.
        deadline = time.monotonic() + 60
        while signal.getsignal(signal.SIGTERM) is not web_app._interrupt and time.monotonic() < deadline:
            time.sleep(0.05)
        _thread.interrupt_main()

    before = signal.signal(signal.SIGTERM, own)
    try:
        thread = threading.Thread(target=interrupt_when_serving)
        thread.start()
        assert serve(tmp_path, port=0, access="key", open_browser=False) == 0
        thread.join()
        assert signal.getsignal(signal.SIGTERM) is own
    finally:
        signal.signal(signal.SIGTERM, before)


def test_serve_leaves_a_handler_python_did_not_set(tmp_path, monkeypatch):
    """A handler that was not set from Python reads None: serve leaves that
    signal as it is and still stops cleanly."""
    import _thread
    import threading

    from testuary import web as web_app
    from testuary.web import serve

    monkeypatch.setattr(web_app, "_testuary_version", lambda python: "testuary")
    real = signal.getsignal
    monkeypatch.setattr(signal, "getsignal",
                        lambda number: None if number == signal.SIGTERM else real(number))
    other = getattr(signal, "SIGBREAK", None) or signal.SIGHUP

    def interrupt_when_serving():
        deadline = time.monotonic() + 60
        while real(other) is not web_app._interrupt and time.monotonic() < deadline:
            time.sleep(0.05)
        _thread.interrupt_main()

    def own(signum, frame):
        pass

    before = signal.signal(signal.SIGTERM, own)
    try:
        thread = threading.Thread(target=interrupt_when_serving)
        thread.start()
        assert serve(tmp_path, port=0, access="key", open_browser=False) == 0
        thread.join()
        assert real(signal.SIGTERM) is own
    finally:
        signal.signal(signal.SIGTERM, before)


def test_a_job_stops_when_its_interface_is_killed(stand_in, tmp_path, started):
    base_dir = tmp_path / "work"
    base_dir.mkdir()
    process, pid, output = _start_interface(base_dir, stand_in, started)
    # No handler runs: the job sees its input end and stops by itself.
    process.kill()
    process.wait(60)
    deadline = time.monotonic() + 60
    while _jobs._alive(pid) and time.monotonic() < deadline:
        time.sleep(0.1)
    assert not _jobs._alive(pid)
    assert "stand-in interrupted" in output.read_text(encoding="utf-8")


def test_the_command_names_its_version(capsys):
    from testuary.web import __version__
    from testuary import web as web_app

    with pytest.raises(SystemExit) as ended:
        web_app.main(["--version"])
    assert ended.value.code == 0
    assert capsys.readouterr().out.strip() == f"testuary-web {__version__}"


def _served(tmp_path, monkeypatch, capsys, ssh):
    """What serve prints and the addresses it opens in a browser, with
    SSH_CONNECTION set to ``ssh`` (None: not set), until Ctrl+C, which comes
    once its loop serves: after every line it prints and the browser."""
    import _thread
    import threading

    from testuary import web as web_app
    from testuary.web import serve

    monkeypatch.setattr(web_app, "_testuary_version", lambda python: "testuary")
    # A start record that takes longer than a moment, as on a slow disk.
    record_start = web_app._record_start

    def slow_record_start(*arguments):
        time.sleep(0.3)
        return record_start(*arguments)

    monkeypatch.setattr(web_app, "_record_start", slow_record_start)
    opened = []
    monkeypatch.setattr(webbrowser, "open", opened.append)
    monkeypatch.setattr(getpass, "getuser", lambda: "kim")
    if ssh is None:
        monkeypatch.delenv("SSH_CONNECTION", raising=False)
    else:
        monkeypatch.setenv("SSH_CONNECTION", ssh)

    # The loop's first sleep is the first in a start: it says the start is done.
    serving = threading.Event()

    class Clock:
        def __getattr__(self, name):
            return getattr(time, name)

        def sleep(self, seconds):
            serving.set()
            time.sleep(seconds)

    monkeypatch.setattr(web_app, "time", Clock())

    def interrupt_when_serving():
        serving.wait(60)
        _thread.interrupt_main()

    thread = threading.Thread(target=interrupt_when_serving)
    thread.start()
    assert serve(tmp_path, port=0, access="key") == 0
    thread.join()
    out = capsys.readouterr().out
    return out, re.search(r"^Open: (http://127\.0\.0\.1:(\d+)/\S*)$", out, re.M).groups(), opened


def test_over_ssh_the_interface_opens_no_browser_and_prints_the_tunnel(tmp_path, monkeypatch, capsys):
    """In an SSH login the browser that matters is on the user's own
    computer: the interface opens none on the server (where a text browser
    could take over the terminal) and prints the ssh -L line for its own
    port, the same number on both sides, from SSH_CONNECTION. That address
    is the one the connection arrived at, which behind NAT or on a cloud
    machine the user's computer cannot reach: the line also offers the name
    the user logs in with."""
    out, (address, port), opened = _served(tmp_path, monkeypatch, capsys,
                                           "203.0.113.5 50112 198.51.100.7 2222")
    assert opened == []
    assert (f"Logged in over SSH: on your own computer run  ssh -L {port}:127.0.0.1:{port} -p 2222 "
            "kim@198.51.100.7 (or the name you log in with)  then open the address above.") in out.splitlines()
    out, (address, port), opened = _served(tmp_path, monkeypatch, capsys, "203.0.113.5 50112 198.51.100.7 22")
    assert opened == []
    assert (f"Logged in over SSH: on your own computer run  ssh -L {port}:127.0.0.1:{port} "
            "kim@198.51.100.7 (or the name you log in with)  then open the address above.") in out.splitlines()


def test_outside_ssh_the_interface_opens_the_browser(tmp_path, monkeypatch, capsys):
    out, (address, port), opened = _served(tmp_path, monkeypatch, capsys, None)
    assert opened == [address]
    assert "Logged in over SSH" not in out


def _free_port():
    import socket

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _web_on(port, base_dir):
    """testuary-web as a process of its own on ``port``, until it prints how
    the start went: the process and its lines."""
    options = ({"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP} if os.name == "nt"
               else {"start_new_session": True})
    process = subprocess.Popen(
        _jobs.module_command(sys.executable, "testuary.web", "--no-browser", "--dir", str(base_dir),
                            "--port", str(port), "--access", "open"),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, **options)
    lines = []
    for line in process.stdout:
        lines.append(line.rstrip())
        if line.startswith(("Ctrl+C", "testuary-web:")):
            break
    return process, lines


def test_a_restart_takes_the_same_port_at_once(tmp_path, started):
    """After Ctrl+C ends the interface with a page's connection and a held
    stream still open, a new start on the same port succeeds at once: on
    Linux and macOS the closed connections would otherwise hold the port
    for a minute or two."""
    base_dir = tmp_path / "work"
    base_dir.mkdir()
    port = _free_port()
    first, lines = _web_on(port, base_dir)
    started.append(first.pid)
    assert any(line.startswith("Open: ") for line in lines), lines
    # Two kept-alive connections, each answered in full, as a page and its
    # stream leave them.
    page = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    page.request("GET", "/api/state", headers={"Host": f"127.0.0.1:{port}"})
    assert page.getresponse().read()
    held = http.client.HTTPConnection("127.0.0.1", port, timeout=60)
    held.request("GET", "/", headers={"Host": f"127.0.0.1:{port}"})
    assert held.getresponse().read()
    first.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
    assert first.wait(120) == 0
    # The browser closes its side once the server has gone, which leaves the
    # server's ends of those connections waiting (TIME_WAIT) on the port.
    page.close()
    held.close()
    second, lines = _web_on(port, base_dir)
    started.append(second.pid)
    if not any(line.startswith("Open: ") for line in lines):
        second.wait(120)
        pytest.fail(f"the restart on port {port} was refused: {lines}")
    second.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
    assert second.wait(120) == 0


def test_no_other_socket_shares_the_live_port(serve):
    """While the interface listens, no other socket binds its address, not
    even one that asks to reuse it (on Windows, SO_REUSEADDR would share a
    port that is not bound exclusively)."""
    import socket

    app, server, client = serve(key=False)
    sharer = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        sharer.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        with pytest.raises(OSError):
            sharer.bind(("127.0.0.1", client.port))
    finally:
        sharer.close()


def test_a_stop_right_after_the_start_line_ends_the_interface_as_ctrl_c_does(tmp_path, started):
    """The interface takes SIGTERM or Ctrl+Break as Ctrl+C from the moment it
    says how to stop it: a stop sent right after that line ends it with
    status 0 and its own last line, never by the signal's default."""
    base_dir = tmp_path / "work"
    base_dir.mkdir()
    process, lines = _web_on(_free_port(), base_dir)
    started.append(process.pid)
    assert lines[-1].startswith("Ctrl+C"), lines
    process.send_signal(signal.CTRL_BREAK_EVENT if os.name == "nt" else signal.SIGTERM)
    assert process.wait(120) == 0
    assert "Stopped." in process.stdout.read()


def test_a_testuary_that_does_not_answer_ends_the_start_with_a_line(tmp_path, monkeypatch, capsys):
    """A Testuary that does not answer its version or help texts in time ends
    the start with a testuary-web line and status 2, not a traceback."""
    from testuary import web as web_app
    from testuary.web import serve

    def slow(command, **options):
        raise subprocess.TimeoutExpired(command, options.get("timeout"))

    monkeypatch.setattr(web_app.subprocess, "run", slow)
    assert serve(tmp_path, port=0, access="key", open_browser=False) == 2
    error = capsys.readouterr().err
    assert error.startswith("testuary-web: Testuary did not answer 'testuary --version' within 300 "
                            "seconds: check that it runs in ")
    assert "Traceback" not in error
