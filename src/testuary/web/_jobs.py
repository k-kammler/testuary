"""Jobs: the Testuary commands the web interface starts.

Each job is a process of its own, started through the launcher
``testuary/web/_launch.py`` by its file path: Testuary refuses two runs in
one process and handles Ctrl+C only in a main thread. One job runs at a
time in a directory, across starts of the interface too
(``.testuary-web/running.json``). Its terminal text and progress events
go to ``.testuary-web/jobs/<id>/``, which the server streams to the page.

Stop works as Ctrl+C: SIGINT to the process group on Linux and macOS, the line
"stop" on the job's input on Windows, which the launcher turns into SIGINT.
When the interface ends, the input ends and the job stops the same way. On
Windows a job has its own console, so closing the interface's window does
not end it before its cleanup.

What a job's event file holds goes to the page from here too: its progress
in Testuary's own view model, with the end-of-run summary from Testuary's
presenter, and the check's report with the console's texts for it.
"""
from __future__ import annotations

import io
import json
import os
import re
import secrets
import signal
import subprocess
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import psutil

from testuary import _words

#: At most this much of a growing file is read at a time.
READ_CHUNK = 1024 * 1024


# Runs a module as ``python -m`` does, without the working directory on the
# module path (a project's secrets.py would replace the standard library's)
# and without the absolute form of each relative PYTHONPATH entry. Python
# 3.11's -P drops only the working directory. This works from 3.10 on.
_MODULE = ("import os, sys; _relative = {{os.path.normcase(os.path.abspath(p)) for p in "
           "os.environ.get('PYTHONPATH', '').split(os.pathsep) if p and not os.path.isabs(p)}}; "
           "sys.path[:] = [p for p in sys.path if p not in ('', '.') "
           "and os.path.normcase(os.path.abspath(p)) not in _relative]; "
           "import runpy; runpy.run_module({!r}, run_name='__main__', alter_sys=True)")


#: The job launcher, started by its file path: by module name another
#: package named testuary on the job's import path could stand in.
LAUNCHER = Path(__file__).with_name("_launch.py")


def launcher_command(python: str, *arguments: str) -> List[str]:
    """The command line running the launcher with ``arguments`` in ``python``,
    importing nothing from the launcher's or the working directory."""
    return [python, str(LAUNCHER), *arguments]


def shown_command(arguments: List[str]) -> str:
    """A job's command as a terminal runs it: ``testuary <arguments>``, or
    ``python -m NAME <arguments>`` for the launcher's ``--module NAME``."""
    if arguments[:1] == ["--module"]:
        return " ".join(["python", "-m", *arguments[1:]])
    return " ".join(["testuary", *arguments])


def module_command(python: str, module: str, *arguments: str) -> List[str]:
    """The command line running ``module`` with ``arguments`` in ``python``
    as ``python -m`` does, importing nothing from the working directory."""
    return [python, "-c", _MODULE.format(module), *arguments]


class Busy(RuntimeError):
    """Another job is still running."""


class NotAllowed(RuntimeError):
    """The job cannot do that now."""


@dataclass
class Job:
    id: str
    kind: str
    title: str
    arguments: List[str]
    job_dir: Path
    output_dir: Optional[Path]
    events: Optional[Path]
    process: subprocess.Popen = field(repr=False)
    started: float = field(default_factory=time.time)
    status: str = "running"
    exit_code: Optional[int] = None
    ended: Optional[float] = None
    stop_requested: Optional[float] = None
    forced: bool = False
    #: Testuary's cleanup time: Force stop is offered only after it.
    cleanup_seconds: Optional[float] = None
    #: The directory the command runs in: a project's directory, or the working
    #: directory.
    cwd: Optional[Path] = None
    #: How an ended run ended, from its event file (``run_end``).
    outcome: Optional[dict] = None
    #: The run directory an ended run or demonstration wrote, None when it
    #: wrote none.
    run_dir: Optional[Path] = None
    #: An ended run's or demonstration's own error or refusal, as it printed
    #: it, when it ended with a failure status.
    error: Optional[str] = None
    #: The model alias and run directory the run's setup report names
    #: (``setup_facts``), once its event file holds the report.
    setup: Optional[dict] = None
    #: Each file's values as an ended run's end-of-run summary states them
    #: (``file_values``).
    values: Optional[list] = None

    @property
    def stdout(self) -> Path:
        return self.job_dir / "stdout.txt"

    @property
    def stderr(self) -> Path:
        return self.job_dir / "stderr.txt"

    def force_after(self, now: Optional[float] = None) -> Optional[float]:
        """Seconds until Force stop is offered, or None before a stop.
        Without a cleanup time it is offered at once."""
        if self.stop_requested is None or self.ended is not None:
            return None
        now = time.time() if now is None else now
        return max(0.0, self.stop_requested + (self.cleanup_seconds or 0.0) - now)

    def view(self) -> dict:
        now = time.time()
        setup = self.setup or {}
        # The run directory a running job writes into, and the one an ended
        # job wrote. A check writes no run.
        run_dir = (None if self.kind == "check" else self.run_dir if self.ended is not None
                   else self.output_dir or setup.get("run_dir"))
        return {
            "model": setup.get("model"), "run_dir": str(run_dir) if run_dir else None,
            "values": self.values,
            "id": self.id, "kind": self.kind, "title": self.title,
            "command": shown_command(self.arguments),
            "arguments": list(self.arguments),
            "project": str(self.cwd) if self.cwd else None,
            "status": self.status, "exit_code": self.exit_code,
            "started": self.started, "ended": self.ended,
            "elapsed_s": round((self.ended or now) - self.started, 1),
            "output_dir": str(self.output_dir) if self.output_dir else None,
            "events": self.events is not None,
            "force_after_s": self.force_after(now),
            "outcome": self.outcome,
            "error": self.error,
        }


def _tail(path: Path, size: int = 65536) -> str:
    """The end of a job's output file, as text."""
    try:
        with open(path, "rb") as stream:
            stream.seek(0, os.SEEK_END)
            stream.seek(max(0, stream.tell() - size))
            return stream.read().decode("utf-8", errors="replace")
    except OSError:
        return ""


def _written(run_dir: Optional[Path]) -> Optional[Path]:
    """``run_dir`` when it exists and holds something, else None."""
    try:
        return run_dir if run_dir is not None and run_dir.is_dir() and next(run_dir.iterdir(), None) else None
    except OSError:
        return None


def _say(text: str) -> None:
    """A line for the person who started the interface. The terminal can
    be gone when the interface ends (a closed window)."""
    try:
        print(text, flush=True)
    except (OSError, ValueError):
        pass


def _alive(pid) -> bool:
    """Whether a process with this id runs."""
    return isinstance(pid, int) and pid > 0 and psutil.pid_exists(pid)


def _taskkill() -> str:
    return os.path.join(os.environ.get("SystemRoot", r"C:\Windows"), "System32", "taskkill.exe")


class JobManager:
    """Starts, watches and stops the jobs of one web interface."""

    def __init__(self, base_dir: Path, *, python: str = sys.executable,
                 cleanup_seconds: Optional[float] = None,
                 import_dir: Callable[[str], Optional[Path]] = lambda kind: None) -> None:
        self.base_dir = Path(base_dir)
        #: The directory a job of a kind imports its module from first, or
        #: None: a file extension's directory for its job kinds.
        self.import_dir = import_dir
        #: A stopped run's longest cleanup (testuary.exits), or None: Force
        #: stop is then offered at once.
        self.cleanup_seconds = cleanup_seconds
        self.store = self.base_dir / ".testuary-web" / "jobs"
        self.marker = self.base_dir / ".testuary-web" / "running.json"
        self.python = python
        self._lock = threading.Lock()
        self._jobs: Dict[str, Job] = {}

    def all(self) -> List[Job]:
        with self._lock:
            return sorted(self._jobs.values(), key=lambda job: job.started, reverse=True)

    def get(self, job_id: str) -> Optional[Job]:
        with self._lock:
            return self._jobs.get(job_id)

    def active(self) -> Optional[Job]:
        with self._lock:
            return next((job for job in self._jobs.values() if job.ended is None), None)

    def event_files(self) -> List[Path]:
        """The event files of the directory's jobs, also of earlier starts, newest
        first (a job's name begins with its start time)."""
        try:
            with os.scandir(self.store) as entries:
                names = [entry.name for entry in entries if entry.is_dir(follow_symlinks=False)]
        except OSError:
            return []
        return [self.store / name / "events.jsonl" for name in sorted(names, reverse=True)]

    def start(self, kind: str, title: str, arguments: List[str], *,
              output_dir: Optional[Path] = None, events: bool = False,
              cwd: Optional[Path] = None) -> Job:
        """Start ``testuary <arguments>``, or an extension job's module when
        the arguments begin with ``--module NAME`` (the launcher's), in
        ``cwd`` (a project's directory), else in the working directory. With
        ``events`` it writes Testuary's progress events. The kind's
        ``import_dir`` goes first on the job's import path."""
        with self._lock:
            if any(job.ended is None for job in self._jobs.values()):
                raise Busy("Another job is still running. Stop it or wait for it to end.")
            job_id = time.strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
            self._claim(job_id)
            try:
                job_dir = self.store / job_id
                job_dir.mkdir(parents=True)
                events_path = job_dir / "events.jsonl" if events else None
                command = launcher_command(self.python, *arguments)
                if events_path is not None:
                    command += ["--events", str(events_path)]
                environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
                import_dir = self.import_dir(kind)
                if import_dir is not None:
                    environment["PYTHONPATH"] = os.pathsep.join(
                        [str(import_dir), *filter(None, [os.environ.get("PYTHONPATH")])])
                options: dict = {}
                if os.name == "nt":
                    options["creationflags"] = (subprocess.CREATE_NEW_PROCESS_GROUP
                                                | subprocess.CREATE_NO_WINDOW)
                else:
                    options["start_new_session"] = True
                with open(job_dir / "stdout.txt", "wb") as out, \
                        open(job_dir / "stderr.txt", "wb") as err:
                    process = subprocess.Popen(command, cwd=str(cwd or self.base_dir),
                                               env=environment,
                                               stdin=subprocess.PIPE, stdout=out, stderr=err,
                                               **options)
            except BaseException:
                self._release(job_id)
                raise
            job = Job(job_id, kind, title, list(arguments), job_dir, output_dir, events_path,
                      process, cleanup_seconds=self.cleanup_seconds, cwd=cwd)
            self._jobs[job_id] = job
            self._mark(job)
        _say(f"Started job {job_id}: {title}")
        threading.Thread(target=self._watch, args=(job,), name=f"job-{job_id}",
                         daemon=True).start()
        return job

    def _claim(self, job_id: str) -> None:
        """Hold the directory's running-job marker for ``job_id``, refused while
        another start of the interface has a job running here. A marker whose
        processes have ended is removed."""
        self.marker.parent.mkdir(parents=True, exist_ok=True)
        for _ in range(3):
            try:
                with open(self.marker, "x", encoding="utf-8") as stream:
                    json.dump({"job": job_id, "server": os.getpid(), "pid": None,
                               "started": time.time()}, stream)
                return
            except FileExistsError:
                try:
                    held = json.loads(self.marker.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    held = {}
                pid = held.get("pid") or held.get("server")
                if _alive(pid) and pid != os.getpid():
                    raise Busy(f"A job started by another start of the interface still runs "
                               f"in this directory (process {pid}). It ends by itself. If no such "
                               f"process runs, delete {self.marker}.") from None
                try:
                    self.marker.unlink()
                except FileNotFoundError:
                    pass
        raise Busy(f"Another start of the interface holds {self.marker}.")

    def _mark(self, job: Job) -> None:
        """Name the started job's process in the marker."""
        record = {"job": job.id, "server": os.getpid(), "pid": job.process.pid,
                  "started": job.started, "command": shown_command(job.arguments)}
        written = self.marker.with_name(f"running.{job.id}.tmp")
        written.write_text(json.dumps(record), encoding="utf-8")
        os.replace(written, self.marker)

    def _release(self, job_id: str) -> None:
        try:
            if json.loads(self.marker.read_text(encoding="utf-8")).get("job") == job_id:
                self.marker.unlink()
        except (OSError, ValueError):
            pass

    def _watch(self, job: Job) -> None:
        code = job.process.wait()
        try:
            job.process.stdin.close()
        except OSError:
            pass
        self._release(job.id)
        outcome, written, error, values = None, None, None, None
        self.read_setup(job)
        if job.kind in ("run", "demo") and job.events is not None:
            try:
                outcome = run_end(job.events) if job.kind == "run" else None
            except Exception:
                outcome = None
            try:
                values = file_values(job.events)
            except Exception:
                values = None
        if job.kind in ("run", "demo"):
            # The directory the run named, else the one its log lies in. A run
            # that stopped before making it wrote none.
            output_dir = job.output_dir
            if output_dir is None and job.events is not None:
                try:
                    output_dir = run_dir(job.events)
                except Exception:
                    output_dir = None
                if output_dir is not None and job.cwd is not None:
                    output_dir = job.cwd / output_dir
            written = _written(output_dir)
            if code not in (0, 130) and not job.forced and job.stop_requested is None:
                error = run_error(_tail(job.stdout), _tail(job.stderr))
                if error is None and job.events is not None:
                    # Refused before its first progress event: the setup
                    # report the run wrote says why.
                    try:
                        error = refusal(job.events)
                    except OSError:
                        error = None
        with self._lock:
            job.outcome = outcome
            job.values = values
            job.run_dir = written
            job.error = error
            job.exit_code = code
            job.ended = time.time()
            # A job the person stopped ends as stopped whatever its status
            # (stopped while Python starts: 0xC000013A on Windows, not 130).
            job.status = ("stopped" if job.forced else "finished" if code == 0
                          else "cancelled" if code == 130 or job.stop_requested is not None
                          else "ended")
        _say(f"Job {job.id} ended with status {code}")

    def read_setup(self, job: Job) -> None:
        """Take the job's setup facts from its event file once it holds the
        setup report, which comes before the run's work: the read stops
        there."""
        if job.setup is not None or job.events is None:
            return
        try:
            found = setup_facts(job.events)
        except OSError:
            found = None
        if found is not None:
            with self._lock:
                job.setup = found

    def stop(self, job_id: str) -> Job:
        """Stop a job as Ctrl+C would: it cancels, cleans up and keeps finished
        work."""
        job = self.get(job_id)
        if job is None:
            raise NotAllowed("No such job.")
        with self._lock:
            if job.ended is not None or job.stop_requested is not None:
                return job
            job.stop_requested = time.time()
            job.status = "stopping"
        try:
            if os.name == "nt":
                job.process.stdin.write(b"stop\n")
                job.process.stdin.flush()
            else:
                os.killpg(job.process.pid, signal.SIGINT)
        except (OSError, ValueError):
            pass
        _say(f"Stop requested for job {job.id}")
        return job

    def force(self, job_id: str) -> Job:
        """End a stopped job still cleaning up after Testuary's cleanup time, with
        every process it started."""
        job = self.get(job_id)
        if job is None:
            raise NotAllowed("No such job.")
        wait = job.force_after()
        if job.ended is not None:
            return job
        if wait is None:
            raise NotAllowed("Stop the job first.")
        if wait > 0:
            raise NotAllowed(f"The run may take up to {job.cleanup_seconds:g} seconds to clean "
                             f"up. Force stop is offered in {wait:.0f} seconds.")
        self.kill(job)
        return job

    def kill(self, job: Job) -> None:
        with self._lock:
            if job.ended is not None:
                return
            job.forced = True
        if os.name == "nt":
            subprocess.run([_taskkill(), "/T", "/F", "/PID", str(job.process.pid)],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        else:
            # LangAdapter starts its commands in sessions of their own, outside
            # the job's process group. They are listed while the job's process
            # still names them as its descendants, and ended after the group,
            # each listed group leader with its group, which also holds what
            # it started after the listing.
            try:
                started = psutil.Process(job.process.pid).children(recursive=True)
            except psutil.Error:
                started = []
            try:
                os.killpg(job.process.pid, signal.SIGKILL)
            except OSError:
                pass
            for process in started:
                try:
                    if process.is_running() and os.getpgid(process.pid) == process.pid:
                        os.killpg(process.pid, signal.SIGKILL)
                    else:
                        process.kill()
                except (OSError, psutil.Error):
                    pass  # ended already, or its number now names another process
        _say(f"Job {job.id} forced to end")


def read_lines(path: Optional[Path], offset: int) -> Tuple[List[str], int]:
    """The complete lines a growing file gained after ``offset`` (at most
    READ_CHUNK bytes) and the new offset. A line still being written waits."""
    if path is None:
        return [], offset
    try:
        with open(path, "rb") as stream:
            stream.seek(offset)
            data = stream.read(READ_CHUNK)
    except FileNotFoundError:
        return [], offset
    end = data.rfind(b"\n")
    if end < 0:
        if len(data) < READ_CHUNK:
            return [], offset
        return [data.decode("utf-8", errors="replace")], offset + len(data)
    chunk = data[:end + 1]
    lines = [line.decode("utf-8", errors="replace").rstrip("\r")
             for line in chunk.split(b"\n")[:-1]]
    return lines, offset + len(chunk)


# The check's report as the page shows it: ``testuary check --json``'s
# key-free output and the console's texts for it, model names included.


def _page_steps(tabs):
    """The check's next actions in the page's words: its own controls, not
    the terminal's commands. ``tabs`` is the extensions' registry: a run
    starts on the Run tab as the header names it, and the Check page's box
    is named only while that page is Testuary's own. Otherwise the tool
    check is named as Testuary's, by no control, since a replacing page
    need not offer it under Testuary's label."""
    from testuary.presentation.readiness import NextSteps
    from testuary.web import extensions

    run_tab = tabs.tab_title("run")
    box = f'"{extensions.TOOL_CHECK_LABEL}" ticked'
    own_check = tabs.own_view("check")
    return NextSteps(
        verify=f"the check with {box}" if own_check
        else "Testuary's check that also starts each tool once and tries the project in a copy",
        check_tools=f"check again with {box}" if own_check else "check again with Testuary's tool check",
        verify_named="", detailed="the detailed report below",
        read_detailed="in the detailed report below", again="check again",
        generate=f"start a run on the {run_tab} page" if run_tab else "start a run")


#: Where Testuary's own error or refusal starts in a command's output.
_ERROR_START = re.compile(r"^(?:ERROR: |Stopped: |testuary(?: [a-z-]+)?: error: )")
#: A progress line of the plain display: a clock, then the state.
_PROGRESS_LINE = re.compile(r"^\d{2}:\d{2}:\d{2} ")
_TRACEBACK = "Traceback (most recent call last):"


def run_error(*outputs: str) -> Optional[str]:
    """A command's own error in its output: the last block starting with
    ``ERROR:``, ``Stopped:`` or the command line's ``error:``, up to a blank
    or progress line, else a traceback's exception line, or None."""
    found = None
    for text in outputs:
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if _ERROR_START.match(line):
                block = [line]
                for following in lines[index + 1:index + 12]:
                    if (not following.strip() or _PROGRESS_LINE.match(following)
                            or _ERROR_START.match(following)):
                        break
                    block.append(following)
                found = "\n".join(block)
    if found is None:
        for text in outputs:
            start = text.rfind(_TRACEBACK)
            if start >= 0:
                found = next((line for line in text[start:].splitlines()[1:]
                              if line.strip() and not line[0].isspace()), found)
    return found


def check_report(text: str) -> dict:
    """The check's JSON report, or ValueError when the output is none."""
    report = json.loads(text)
    if not isinstance(report, dict) or "items" not in report:
        raise ValueError("the check printed no report")
    return report


def check_texts(report: dict, tabs) -> dict:
    """The report as Testuary's presenter writes it, with the page's words for
    the next actions as the extensions' registry ``tabs`` shows the tabs: the
    summary ``testuary check`` prints and the ``--display detailed`` listing
    (None for a damaged report), each item's plain name and status word, and
    Testuary's label of each phase it names."""
    return {**_written_report(report, _page_steps(tabs)), "phases": phase_labels(
        [*(report.get("phases") or ()), *(limit.get("phase") for limit in report.get("limits") or ())])}


def phase_labels(phases) -> Dict[str, str]:
    """Testuary's label of each named phase, as its view model writes it."""
    from testuary.presentation.view import phase_label

    return {phase: phase_label(phase) for phase in phases if isinstance(phase, str) and phase}


def _readiness_value(report):
    """The setup report ``report`` (``RunReadiness.as_dict()``) as a
    ``RunReadiness``, or None for one this version cannot read."""
    try:
        from testuary.readiness import ReadinessItem, ReadinessLimit, RunReadiness

        return RunReadiness(**{
            **report,
            "limits": tuple(ReadinessLimit(**limit) for limit in report.get("limits") or ()),
            "items": tuple(ReadinessItem(**item) for item in report.get("items") or ())})
    except Exception:
        return None


def _written_report(report: dict, steps) -> dict:
    value = _readiness_value(report)
    if value is None:
        return {"summary": None, "detailed": None, "items": {}}
    # Each item's plain name and status word, as the summary says them.
    return {"summary": _report_text(value, "plain", steps),
            "detailed": _report_text(value, "detailed", steps),
            "items": {item.key: {"name": _words.plain_name(item),
                                 "status": _words.status_word(item.status, value.scope)}
                      for item in value.items}}


def _report_text(value, mode: str, steps) -> Optional[List[str]]:
    from rich.console import Console

    from testuary.presentation import TerminalPresenter, present_readiness

    stream = io.StringIO()
    console = Console(file=stream, width=10000, force_terminal=False, color_system=None)
    with TerminalPresenter(mode, console=console) as presenter:
        present_readiness(value, presenter=presenter, steps=steps)
    return stream.getvalue().splitlines()


# A job's progress as the console shows it: the event file goes into
# Testuary's own view model (``RunView``), and the page draws its rows, the
# end-of-run summary from Testuary's presenter (``present_completion``).


def _bar(bar) -> Optional[dict]:
    return None if bar is None else {"ratio": bar.ratio, "figures": bar.figures}


def _tone(event) -> Optional[str]:
    # A row's tone is its event's state, except that a failure the producer
    # marks as a result, not an execution failure, reads as a warning: the
    # view model's rule, which the terminal presenter uses too.
    from testuary.presentation.view import is_outcome

    if event is None:
        return None
    return "incomplete" if is_outcome(event) else event.state


def _summary(record: dict, last) -> List[str]:
    """The end-of-run summary's lines as the console writes them in plain
    mode, after the run's last observation ``last``, whose provenance they
    keep (demo values read as demo values)."""
    from testuary.presentation.terminal import RunCompletion

    return summary_lines(RunCompletion.from_dict(record), last)


def summary_lines(completion, last=None) -> List[str]:
    """A ``RunCompletion``'s plain mode lines after ``last``, without the
    command that continues the run (the page offers Continue)."""
    from dataclasses import replace

    from rich.console import Console

    from testuary.presentation import TerminalPresenter
    from testuary.presentation.terminal import present_completion

    completion = replace(completion, resume_command=None)
    stream = io.StringIO()
    console = Console(file=stream, width=10000, force_terminal=False, color_system=None)
    with TerminalPresenter("plain", console=console) as presenter:
        if last is not None:
            presenter.view.add(last)
        present_completion(completion, presenter=presenter)
    return stream.getvalue().splitlines()


def completion_lines(record: dict, last: Optional[dict] = None) -> Optional[List[str]]:
    """The end-of-run summary's lines for an event file's completion record
    after the event ``last``. None for a damaged record."""
    try:
        from testuary.presentation import DisplayEvent

        return _summary(record, DisplayEvent.from_dict(last) if last is not None else None)
    except Exception:
        return None


def record(line) -> Optional[dict]:
    """One event file line (text or bytes) as its record, a JSON object, or
    None."""
    try:
        data = json.loads(line)
    except ValueError:
        return None
    return data if isinstance(data, dict) else None


def records(path: Path):
    """Each record of an event file, in order."""
    with open(path, encoding="utf-8", errors="replace") as stream:
        for line in stream:
            data = record(line)
            if data is not None:
                yield data


def run_end(path: Path) -> Optional[dict]:
    """How an ended run's event file says it ended: whether Testuary marked it
    incomplete, its outcome in one sentence and how many files a continued
    run would finish. None without an end-of-run summary."""
    from testuary.presentation.terminal import RunCompletion, headline

    found, last = None, None
    for data in records(path):
        if isinstance(data.get("completion"), dict):
            found = data["completion"]
        event = data.get("event")
        if isinstance(event, dict) and event.get("level") == "run":
            last = event
    if found is None:
        return None
    completion = RunCompletion.from_dict(found)
    marked = bool(last) and last.get("state") == "failed" and last.get("failure") == "outcome"
    return {"incomplete": marked, "headline": headline(completion),
            "unfinished": len(completion.unfinished)}


def file_values(path: Path) -> Optional[List[dict]]:
    """Each file's values as an ended run's end-of-run summary states them,
    a percentage cut to one decimal (``percent_text``), a value it did not
    measure as None, with the summary's words for how the file ended. None
    without an end-of-run summary."""
    from testuary.presentation.terminal import RunCompletion
    from testuary.presentation.view import percent_text

    found = None
    for data in records(path):
        if isinstance(data.get("completion"), dict):
            found = data["completion"]
    if found is None:
        return None

    def text(metric) -> Optional[str]:
        if metric.value is None:
            return None
        return f"{percent_text(metric.value)}%" if metric.unit == "%" else f"{metric.value:g}{metric.unit}"

    return [{"file": item.path, "outcome": item.outcome, "outcome_text": _words.OUTCOME_WORDS[item.outcome],
             "values": [{"name": metric.name, "text": text(metric)} for metric in item.metrics]}
            for item in RunCompletion.from_dict(found).files]


def setup_facts(path: Path) -> Optional[dict]:
    """The model alias and run directory a run's setup report names, read
    up to that report, which Testuary writes before its work. None before
    it."""
    for data in records(path):
        report = data.get("readiness")
        if isinstance(report, dict):
            return {"model": report.get("model_alias") or None,
                    "run_dir": report.get("output_dir") or None}
    return None


def run_dir(path: Path) -> Optional[Path]:
    """The run directory an ended run's event file names: the directory of the
    log (``<directory>/logs/run.log``) of its latest run event naming one."""
    found = None
    for data in records(path):
        event = data.get("event")
        if isinstance(event, dict) and event.get("level") == "run" and event.get("log_path"):
            found = str(event["log_path"])
    log = Path(found) if found else None
    return log.parent.parent if log and log.name == "run.log" and log.parent.name == "logs" else None


def blocking_reason(readiness) -> Optional[str]:
    """A setup report's first blocking reason, the problem its summary lists
    first."""
    value = _readiness_value(readiness)
    return None if value is None else _words.blocking_reason(value)


def readiness_record(path: Path) -> Optional[dict]:
    """The setup report an event file holds: its first readiness record."""
    for data in records(path):
        if isinstance(data.get("readiness"), dict):
            return data["readiness"]
    return None


def refusal(path: Path) -> Optional[str]:
    """Why a run was refused before its work, from its event file's report."""
    return blocking_reason(readiness_record(path))


class Progress:
    """One job's progress, built from its event file's lines in order."""

    def __init__(self, base: Optional[Path] = None) -> None:
        from testuary.presentation import DisplayEvent
        from testuary.presentation.view import RunView

        #: The directory the job runs in: paths below it are shown relative.
        self.base = base

        self._event = DisplayEvent.from_dict
        self.view = RunView()
        #: The lines the console's log records above its pinned rows.
        self.log: List[str] = []
        #: The end-of-run summary's lines, once the event file holds it.
        self.summary: Optional[List[str]] = None
        #: The run's plan (``RunPlan``), once the event file holds it.
        self.plan = None
        #: The setup report before the run, once the event file holds it.
        self.readiness = None
        #: The latest event of each run, phase and step, for the diagram.
        self._marks: dict = {}
        #: The run's clock when each step under way began.
        self._began: dict = {}

    def add(self, line: str) -> bool:
        """Take one line of the event file. Whether the view changed."""
        record = json.loads(line)
        if "schema_version" in record:
            return False
        if "completion" in record:
            latest = list(self.view.latest().values())
            self.summary = _summary(record["completion"], latest[-1] if latest else None)
            return True
        if "readiness" in record:
            # The setup report before the run: its blocking items name why a
            # refused run did not start.
            self.readiness = record["readiness"] if isinstance(record["readiness"], dict) else None
            return False
        if "plan" in record:
            from testuary.presentation.events import RunPlan

            self.plan = RunPlan.from_dict(record["plan"])
            self.view.plan = self.plan
            return True
        fields = record.get("event")
        if not isinstance(fields, dict):
            return False
        event = self._event(fields)
        observation = self.view.add(event)
        if (event.level or "") in ("run", "phase", "step"):
            # The view model keeps only recent finished work in view: the
            # diagram keeps each phase's and step's latest state itself.
            key = (event.target, event.phase, event.operation, event.level)
            self._marks.pop(key, None)
            self._marks[key] = event
            if event.state == "running":
                self._began.setdefault(key, event.elapsed_s)
            else:
                self._began.pop(key, None)
        if observation.logged:
            self.log.append(observation.line)
        return True

    def snapshot(self, operations: int = 4) -> dict:
        """The compact view's rows as the page draws them."""
        snapshot = self.view.snapshot(operations)
        cause = self._cause(snapshot.cause)
        return {
            "empty": snapshot.empty, "title": snapshot.title, "playback": snapshot.playback,
            "layers": [{"name": layer.name, "text": layer.text, "dim": layer.dim,
                        "tone": _tone(layer.event) if layer.styled else None,
                        "bar": _bar(layer.bar), "bar_tone": _tone(layer.event),
                        **(self._path(self._log_path()) if layer.name == "Log" else {})}
                       for layer in snapshot.layers],
            "operations": [{"state": operation.state, "tone": _tone(operation.event),
                            "name": operation.name, "elapsed": operation.elapsed,
                            "bar": _bar(operation.bar), "progress": operation.progress}
                           for operation in snapshot.operations],
            "hidden_active": snapshot.hidden_active, "message": snapshot.message,
            "metrics": snapshot.metrics, "usage": snapshot.usage,
            "issues": list(snapshot.issues), "log_path": self._log_path(),
            "cause": cause, "diagram": self.diagram(),
            # A cause the model service gave, which the page links to the
            # connection test of the Models page.
            "cause_kind": ("service" if cause and cause.startswith(_words.SERVICE_FAILURE_START)
                           else None),
            # The run's clock at its newest observation: a running row's time
            # on the page follows it, not the row's own last observation.
            "clock_s": max((event.elapsed_s for event in self.view.latest().values()), default=None),
        }

    def _ended(self) -> bool:
        """Whether the run has ended: its latest run event is in an ended
        state of the view model, which is a run's last record."""
        from testuary.presentation.view import TERMINAL_STATES

        return any((event.level or "") == "run" and event.state in TERMINAL_STATES
                   for event in self._marks.values())

    def _cause(self, cause: Optional[str]) -> Optional[str]:
        """Why a failed run failed: for a run refused before its work, the setup
        report's first blocking reason, else the view model's sentence."""
        reason = blocking_reason(self.readiness)
        return reason if cause and reason else cause

    def diagram(self) -> Optional[dict]:
        """The current file's phases and steps, each in its latest observed
        state, else the plan's, with the view model's counters, and
        Finishing with what it does, the step observed in it and the
        values it measured. None before the run's plan."""
        from testuary.presentation.view import (FINISHING_PURPOSE, FINISHING, level_of, metric_text,
                                                own_finishing)

        plan = self.plan
        if plan is None:
            return None
        latest = list(self._marks.values())
        work = [event for event in latest if level_of(event) != "run"]
        target = work[-1].target if work else (plan.files[0] if plan.files else "")

        def newest(match):
            return next((event for event in reversed(latest) if event.target == target and match(event)),
                        None)

        # Once the run has ended, what it planned and never reached was not
        # run. The ended states are the view model's own (refused among them).
        from testuary.presentation.view import TERMINAL_STATES

        over = self._ended()
        phases = []
        for phase in plan.phases:
            seen = newest(lambda event: (event.level or "") == "phase" and event.phase == phase.phase)
            ended = over or (seen is not None and seen.state in TERMINAL_STATES)
            counts = self.view.step_counts(phase.phase, target)
            steps = []
            for step in phase.steps:
                observed = newest(lambda event: event.phase == phase.phase
                                  and event.operation == f"step/{step.step}")
                # A phase that ended without a step it planned did not need it.
                state = observed.state if observed else ("skipped" if ended else step.state)
                began = (self._began.get((observed.target, observed.phase, observed.operation, observed.level))
                         if observed is not None and state == "running" else None)
                # A step a limit ended names the limit (its event's message).
                reached = (observed.message if observed is not None and state == "limit_reached"
                           else None)
                steps.append({"step": step.step, "label": step.label, "state": state,
                              "count": counts["steps"].get(step.step), "since_s": began,
                              "reached": reached})
            state = seen.state if seen else ("skipped" if over and phase.state == "planned" else phase.state)
            # A phase that ended without any step of its start and cycle says
            # why, in its own end's words, such as a goal its inherited suite
            # met. Its phase end runs either way.
            quiet = steps[:phase.end] and all(item["state"] == "skipped" for item in steps[:phase.end])
            note = seen.message if quiet and seen is not None and seen.state == "succeeded" else None
            phases.append({"phase": phase.phase, "label": phase.label, "cycle": phase.cycle,
                           "end": phase.end, "state": state, "steps": steps, "note": note,
                           "count": counts["phase"], "values": counts["values"],
                           "limits": [{"name": limit.name, "value": limit.value, "unit": limit.unit}
                                      for limit in phase.limits]})
        finishing = newest(lambda event: event.operation == FINISHING)
        step = newest(lambda event: event.phase == "finishing" and level_of(event) == "step"
                      and event.operation != FINISHING)
        return {"target": target, "files": list(plan.files), "runs": list(plan.runs), "phases": phases,
                "finishing": {"step": plan.finishing.step, "label": plan.finishing.label,
                              "purpose": FINISHING_PURPOSE if own_finishing(plan.finishing.label) else None,
                              "state": finishing.state if finishing else (
                                  "skipped" if over and plan.finishing.state == "planned"
                                  else plan.finishing.state),
                              "steps": [] if step is None else [{
                                  "label": step.label, "state": step.state,
                                  "since_s": self._began.get((step.target, step.phase, step.operation, step.level))
                                  if step.state == "running" else None}],
                              "values": [metric_text(metric, finishing.provenance) for metric in finishing.metrics
                                         if metric.value is not None] if finishing else []}}

    def _path(self, path: Optional[str]) -> dict:
        """A path for the page: in full, and relative to the job's directory
        below it."""
        if not path:
            return {}
        shown = path
        if self.base is not None:
            try:
                shown = Path(path).resolve().relative_to(Path(self.base).resolve()).as_posix()
            except (OSError, ValueError):
                shown = path
        return {"path": path, "short": shown}

    def _log_path(self) -> Optional[str]:
        """The run's log, as its latest run observation names it."""
        for event in reversed(list(self.view.latest().values())):
            if (event.level or "") == "run" and event.log_path:
                return event.log_path
        return None
