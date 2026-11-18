"""The local web server: the pages, the JSON interface and the job streams.

It listens on 127.0.0.1 only and answers only requests naming it in their
Host header, which other web pages cannot use through another name for
127.0.0.1 (DNS rebinding). A change must come from this server's Origin when
the browser names one. With an access key (``--access key`` or
``stable_key``), the JSON interface and the streams also need it in the
header X-Testuary-Web-Token: the printed address gives it to the page in its
fragment, which reaches no server, and the page keeps it for this server's
address (a cookie would reach every server on 127.0.0.1). Without one, any
program on this computer that reaches the port can use the interface. The
log records actions and exit statuses only.
"""
from __future__ import annotations

import hmac
import json
import os
import re
import secrets
import socket
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib import resources
from pathlib import Path
from typing import Optional, Sequence
from urllib.parse import parse_qs, unquote, urlsplit

from testuary.setup import _DOCS
from testuary.web import _jobs, _results
from testuary.web import _data
from testuary.web import _registry
from testuary.web import extensions as web_extensions

STATIC = {
    "index.html": "text/html; charset=utf-8",
    "app.js": "text/javascript; charset=utf-8",
    "style.css": "text/css; charset=utf-8",
    "icon.svg": "image/svg+xml",
}
SECURITY_HEADERS = {
    "Content-Security-Policy": ("default-src 'none'; script-src 'self'; style-src 'self'; "
                                "img-src 'self'; connect-src 'self'; base-uri 'none'; "
                                "form-action 'none'; frame-ancestors 'none'"),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
    "Cross-Origin-Opener-Policy": "same-origin",
    "Cross-Origin-Resource-Policy": "same-origin",
}
MAX_BODY = 64 * 1024
TOKEN_HEADER = "X-Testuary-Web-Token"
#: The header of a stop request (testuary-web --stop and --replace): the
#: start's stop secret, which only its record in the user's directory holds.
STOP_HEADER = "X-Testuary-Web-Stop"
STREAM_POLL_SECONDS = 0.2
KEEPALIVE_SECONDS = 10.0
#: An extension's read-only route: its extension's id and its path.
_ROUTE = re.compile(r"/api/ext/(%s)/(%s)" % (_registry.ID.pattern, _registry.ROUTE_PATH.pattern))
_JOB = re.compile(r"/api/jobs/([0-9a-f-]{1,64})(/stream|/stop|/force|/continue)?")
# A request about one model alias: removing it or testing its connection.
_ENTRY = re.compile(r"/api/models/([^/]+)/(remove|test)")
#: The most source files one run or check names.
MAX_SOURCE_FILES = 100


#: The web interface's guide of this version, online.
GUIDE = _DOCS + "guide/web.md"


class Refused(Exception):
    """A refused request. ``field`` names the form field it concerns."""

    def __init__(self, status: int, message: str, field: Optional[str] = None) -> None:
        super().__init__(message)
        self.status = status
        self.field = field


def _text_field(body: dict, name: str, limit: int = 4096) -> str:
    value = body.get(name) or ""
    if not isinstance(value, str) or len(value) > limit or "\x00" in value:
        raise Refused(400, f"Enter text of at most {limit} characters.", name)
    return value.strip()


def _requested_files(body: dict) -> list:
    """The source files a run or check names, relative to the project's
    directory, or an empty list for the project's own [project] source_files.
    A name beginning with a dash would read as an option and is refused."""
    value = body.get("files") or []
    if isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, list) or len(value) > MAX_SOURCE_FILES:
        raise Refused(400, f"Name at most {MAX_SOURCE_FILES} source files.", "files")
    names = []
    for item in value:
        if not isinstance(item, str) or len(item) > 4096 or "\x00" in item:
            raise Refused(400, "Name each source file as text.", "files")
        if item.strip().startswith("-"):
            raise Refused(400, "A source file's name cannot begin with a dash.", "files")
        if item.strip():
            names.append(item.strip())
    return names


class App:
    """One start of the web interface: its directory, access key (None: open to
    every program that reaches the port), stop secret and jobs."""

    def __init__(self, base_dir: Path, *, testuary_version: str,
                 python: str, token: Optional[str] = None) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.token = token
        #: The secret a stop request must carry. testuary-web's start writes
        #: it into its record in the user's directory.
        self.stop_secret = secrets.token_urlsafe(32)
        #: Ends the interface as Ctrl+C does, set by testuary-web's start. An
        #: interface started otherwise (a test's, a caller's own) is not
        #: stopped from outside.
        self.stopper = None
        self.testuary_version = testuary_version
        #: Testuary's exit statuses and cleanup time (``testuary.exits``), the
        #: latter read once: reading it loads the module that defines it.
        from testuary import exits

        self.exits = exits
        self.jobs = _jobs.JobManager(self.base_dir, python=python,
                                     cleanup_seconds=float(exits.CLEANUP_SECONDS),
                                     import_dir=lambda kind: self.extensions.import_dir(kind))
        #: The web extensions the base directory's settings name, loaded
        #: once at the start.
        self.extensions = _registry.started(self.base_dir)

    def state(self) -> dict:
        active = self.jobs.active()
        if active is not None:
            # Its model and run directory, once its setup report is written.
            self.jobs.read_setup(active)
        return {"base_dir": str(self.base_dir), "testuary": self.testuary_version,
                "guide": self.extensions.guide(GUIDE), "extensions": self.extensions.public(),
                "tool_check_label": web_extensions.TOOL_CHECK_LABEL,
                "active": {**active.view(), **self._words(active)} if active else None,
                "jobs": [{**job.view(), **self._words(job)} for job in self.jobs.all()]}

    def continue_run(self, job_id: str) -> _jobs.Job:
        """The ended job again: a run with ``--resume``, as the console's next
        step names it, with the same options and directory, finishing its
        unfinished files, or an extension's job as its kind's resume says."""
        job = self.jobs.get(job_id)
        kind = self.extensions.job_kind(job.kind) if job is not None else None
        if kind is not None and kind.resume is not None and job.ended is not None:
            # An extension's job kind continues as the kind says.
            title = job.title if job.title.endswith(", continued") else f"{job.title}, continued"
            resumed = [str(item) for item in kind.resume(job.arguments[2:])]
            return self.jobs.start(job.kind, title, ["--module", kind.module, *resumed],
                                   events=True, cwd=job.cwd)
        if job is None or job.kind != "run":
            raise Refused(404, "No such run.")
        if job.ended is None or not (job.outcome or {}).get("unfinished"):
            raise _jobs.NotAllowed("This run has nothing to continue.")
        arguments = [argument for argument in job.arguments if argument != "--resume"]
        title = job.title if "--resume" in job.arguments else f"{job.title}, continued"
        return self.jobs.start("run", title, [*arguments, "--resume"], output_dir=job.output_dir,
                               events=True, cwd=job.cwd)

    def job_view(self, job: _jobs.Job) -> dict:
        view = job.view()
        view["meaning"] = self.meaning(job)
        view.update(self._words(job))
        if job.kind == "check" and job.ended is not None:
            # The check's output is its report: the page gets it parsed, with
            # the end of its error text.
            try:
                raw = job.stdout.read_text(encoding="utf-8", errors="replace")
                errors = job.stderr.read_text(encoding="utf-8", errors="replace")
            except OSError:
                raw, errors = "", ""
            try:
                view["report"] = _jobs.check_report(raw)
            except ValueError:
                view["report"] = None
            view["report_text"] = (_jobs.check_texts(view["report"], self.extensions) if view["report"]
                                   else {"summary": None, "detailed": None})
            view["errors"] = errors[-20000:]
        elif (self.extensions.job_kind(job.kind) is not None and job.ended is not None
              and job.events is not None and job.events.is_file()):
            # An extension's job reports a setup in Testuary's readiness
            # format: the page draws it as the check's report.
            report = _jobs.readiness_record(job.events)
            if report is not None:
                view["report"], view["report_text"] = report, _jobs.check_texts(report, self.extensions)
        kind = self.extensions.job_kind(job.kind)
        if kind is not None and kind.nested_runs is not None:
            view.update(_nested_runs(kind, job))
        return view

    def meaning(self, job: _jobs.Job) -> Optional[str]:
        """What the job's exit status means in Testuary's words
        (``testuary.exits``). None while it runs and after a force stop."""
        if job.exit_code is None or job.forced:
            return None
        if self.extensions.job_kind(job.kind) is not None:
            # An extension's job has no Testuary exit statuses.
            return ("The job finished." if job.exit_code == 0
                    else f"The job ended with exit status {job.exit_code}.")
        return self.exits.meaning(self._command(job), job.exit_code)

    def _words(self, job: _jobs.Job) -> dict:
        """Testuary's word for an ended job's status and whether it is a failure
        (``testuary.exits``). None while it runs and after a force stop."""
        if job.exit_code is None or job.forced:
            return {}
        # A stop the person asked for is an interruption, whatever status the
        # stopped process returned.
        code = self.exits.INTERRUPTED if job.status == "cancelled" else job.exit_code
        ended = self.exits.status(self._command(job), code)
        return {"word": ended.word, "failure": ended.failure}

    def _command(self, job: _jobs.Job) -> str:
        """The command of ``job`` as testuary.exits names it."""
        return {"check": self.exits.CHECK, "demo": self.exits.DEMO}.get(job.kind, self.exits.RUN)

    def catalogue(self) -> dict:
        """The model aliases Testuary's setup lists, from the key-free settings
        export, the default model (the one the global settings choose for a
        project that names none) and the working directory's project's own
        model, when it names one."""
        try:
            from testuary import setup
            from testuary.settings import resolve_settings

            exported = resolve_settings(base_dir=self.base_dir, project_settings=False).export()
            aliases = {entry.alias for entry in setup.models(self.base_dir)}
        except Exception:
            # No details: the check reports them, and an error text can
            # quote a configuration value.
            return {"error": "Testuary cannot read the settings of this directory. The quick "
                             "check names the problem.", "models": {}, "default_model": None,
                    "project_model": None}
        sections = exported.get("sections", {})
        models = {name: value for name, value in exported.get("models", {}).items() if name in aliases}
        project = _data.working(self.base_dir)
        try:
            own = setup.project(project).fields.get("model") if project else None
        except Exception:
            # The Projects page names a project whose settings cannot be read.
            own = None
        return {"error": None, "models": models,
                "default_model": sections.get("api", {}).get("model") or None,
                "project_model": {"name": project.name, "model": own} if own else None}

    def dirs(self, text: str) -> dict:
        """Directory names for the directory browser: the roots without a path, else
        the directories inside it, a listed project's directory marked."""
        shortcuts = {"Working directory": self.base_dir, "Home": Path.home()}
        if not text:
            return _data.dir_roots(shortcuts)
        return _data.dir_listing(text, _data.marks(self.base_dir), shortcuts)

    def results_base(self, text: str) -> Path:
        """The directory the results view reads ``text`` from: a listed project's
        directory when ``text`` lies in it, else the working directory."""
        path = Path(text)
        if path.is_absolute():
            for project_dir in _data.listed(self.base_dir):
                if _results.inside(path, project_dir):
                    return project_dir.resolve()
        return self.base_dir

    def results_project(self, text: str) -> Path:
        """The listed project a run directory ``text`` belongs to: a
        delivered test file goes only into its own project."""
        path = Path(text)
        for project_dir in _data.listed(self.base_dir):
            if path.is_absolute() and _results.inside(path, project_dir):
                return project_dir.resolve()
        raise Refused(400, "Only the results of a listed project's run go into a project.")

    def adoption(self, text: str, relative: str) -> dict:
        run_dir = _results.locate(self.results_base(text), text)
        return _results.adoption(run_dir, self.results_project(text), relative)

    def adopt(self, body: dict) -> dict:
        text = _text_field(body, "run_dir")
        run_dir = _results.locate(self.results_base(text), text)
        written = _results.adopt(run_dir, self.results_project(text), _text_field(body, "path"),
                                 _text_field(body, "test_dir"))
        return {"written": str(written)}

    def demo_settings(self) -> dict:
        """The demonstration's configured playback length in seconds from the
        key-free settings export, or None when they cannot be read."""
        try:
            from testuary.settings import resolve_settings

            sections = resolve_settings(base_dir=self.base_dir).export()["sections"]
        except Exception:
            return {"playback_seconds": None}
        return {"playback_seconds": sections.get("demo", {}).get("playback_seconds")}

    def start_demo(self, body: dict) -> _jobs.Job:
        language = body.get("language", "python")
        if language not in ("python", "java"):
            raise Refused(400, "The language is python or java.", "language")
        output = self.base_dir / f"testuary-demo-{language}-{time.strftime('%Y%m%d-%H%M%S')}"
        arguments = ["--demo", "--demo-language", language, "--output-dir", str(output),
                     "--display", "plain", "--no-color"]
        # Normal speed unless asked otherwise, over the configured or given
        # length. Testuary's "auto" would play instantly into files.
        pace = body.get("pace", "paced")
        if pace == "instant":
            arguments += ["--demo-speed", "instant"]
        elif pace == "paced":
            arguments += ["--demo-speed", "1"]
            seconds = _text_field(body, "seconds", 64)
            if seconds:
                arguments.append(f"--demo-playback-seconds={seconds}")
        else:
            raise Refused(400, "The pace is instant or paced.", "pace")
        name = "Python" if language == "python" else "Java"
        return self.jobs.start("demo", f"{name} demonstration", arguments, output_dir=output,
                               events=True)

    def start_run(self, body: dict) -> _jobs.Job:
        """A run of a listed project in its directory, as ``testuary [FILES]`` there:
        the page's files, else [project] source_files, with the page's options,
        each passed as ``--option=value`` so no text becomes an option."""
        project_dir = _data.known(self.base_dir, body.get("project_dir"))
        arguments = [*_requested_files(body), "--display", "plain", "--no-color"]
        for name, option in (("model", "--model"), ("runs", "--runs"), ("run", "--run"),
                             ("output_dir", "--output-dir")):
            value = _text_field(body, name)
            if value:
                arguments.append(f"{option}={value}")
        resume = body.get("resume", False)
        if not isinstance(resume, bool):
            raise Refused(400, "Choose whether to continue the earlier run in this directory.", "resume")
        if resume:
            # The earlier run in the same run directory, as Testuary continues it.
            arguments.append("--resume")
        given = _text_field(body, "output_dir")
        output = (project_dir / given) if given else None
        title = f"Run of {project_dir.name}" + (", continued" if resume else "")
        return self.jobs.start("run", title, arguments, output_dir=output,
                               events=True, cwd=project_dir)

    def start_check(self, body: dict) -> _jobs.Job:
        """The check of a listed project in its directory. When its run
        directory holds a saved run, the check judges what the Run page does by
        default: continuing it while files remain unfinished, else the free
        directory beside it."""
        project_dir = _data.known(self.base_dir, body.get("project_dir"))
        arguments = ["check", "--json"]
        if body.get("verify") is True:
            arguments.append("--verify")
        saved = _data.saved_run(self.base_dir, str(project_dir))
        if saved["occupied"]:
            arguments.append("--resume" if saved["unfinished"] else f"--output-dir={saved['fresh']}")
        arguments += _requested_files(body)
        return self.jobs.start("check", f"Check of {project_dir.name}", arguments, cwd=project_dir)

    def dir_view(self, kind: str, named_dir: Path) -> dict:
        """A directory an extension's directory type opens: its summary, read from
        the files the extension wrote."""
        found = self.extensions.dir_type(kind)
        if found is None:
            raise Refused(404, "No such directory type.")
        if not _detects(found, named_dir):
            raise Refused(404, f"This directory is not one of {found.title}.", "dir")
        return {"type": kind, "title": found.title, "dir": str(named_dir),
                "summary": _extension_answer("directory summary", found.summary, named_dir)}

    def dir_views(self, named_dir: Path) -> list:
        """The extensions' directory types ``named_dir`` is, which the Results page
        links to."""
        return [{"type": key, "title": item.title}
                for key, item in self.extensions.dir_types().items() if _detects(item, named_dir)]

    def start_extension_job(self, body: dict) -> _jobs.Job:
        """A job of an extension's kind: the module and arguments its
        ``command`` makes of the page's form."""
        key = body.get("kind")
        kind = self.extensions.job_kind(key) if isinstance(key, str) else None
        if kind is None:
            raise Refused(404, "No such job kind.", "kind")
        try:
            arguments = [str(item) for item in kind.command(
                {name: value for name, value in body.items() if name != "kind"})]
        except ValueError as refusal:
            raise Refused(400, str(refusal)) from None
        return self.jobs.start(key, kind.title, ["--module", kind.module, *arguments], events=True)

    def extension_route(self, extension_id: str, path: str, query: dict) -> dict:
        """An extension's read-only route: the JSON its function returns for
        the query and the directory its ``dir`` names, nothing else of the
        server."""
        route = self.extensions.route(extension_id, path)
        if route is None:
            raise Refused(404, "Not found.")
        text = _first(query, "dir")
        named_dir = _results.named_dir(self.results_base(text), text) if text else None
        return _extension_answer("route", route.read, named_dir,
                                 {name: values[0] for name, values in query.items()})

    def settings_view(self, dir_text: str, scope: str = "") -> dict:
        """Testuary's settings, then each extension's settings groups, and
        the lists of settings files the groups name for a listed project
        (``scopes``). With one of them as ``scope``, Testuary's settings as
        the extension's runs read them, from the defaults, that list and the
        environment, without the groups."""
        scopes = self.named_files(dir_text)
        try:
            named = _chosen(scopes, scope)
        except (_data.Refusal, Refused) as refusal:
            # The page shows the error in its choice, also for a scope the
            # project has no list for.
            return {"error": str(refusal), "sections": [], "files": {}, "scopes": scopes}
        view = _data.settings_view(self.base_dir, dir_text,
                                   None if named is None else [Path(file) for file in named["files"]])
        project_dir = _data.known(self.base_dir, dir_text) if dir_text else None
        for key, group in self.extensions.settings_groups().items() if named is None else ():
            view["sections"] += _group_sections(key, group, project_dir)
        return {**view, "scopes": scopes}

    def named_files(self, dir_text: object) -> list:
        """The lists of settings files the settings groups name for the
        listed project ``dir_text``, each with its group's key and title
        (``_named_entry``)."""
        from testuary.settings import file_locations

        project_dir = _data.known(self.base_dir, dir_text) if dir_text else None
        global_config = Path(file_locations(self.base_dir).config).resolve()
        return [entry for key, group in self.extensions.settings_groups().items()
                if project_dir is not None and group.testuary_settings_files is not None
                for entry in [_named_entry(key, group, project_dir, global_config)] if entry]

    def named_file(self, dir_text: object, scope: object) -> Optional[Path]:
        """The last settings file of the list a change with ``scope`` goes
        to when a settings group names one for the listed project, refused
        with the reason when it is read-only (``_chosen``)."""
        named = None if scope in _OWN_SCOPES else _chosen(self.named_files(dir_text), scope)
        if named is not None and named["read_only"]:
            raise _data.Refusal(named["read_only"])
        return None if named is None else Path(named["file"])

    def update_settings(self, body: dict) -> dict:
        """A change of Testuary's settings: in the global configuration or a
        project's settings file, else in the last file of a settings group's
        list, which the group checks (``_checked_save``)."""
        named = self.named_file(body.get("project_dir"), body.get("scope", "global"))
        if named is None:
            _data.update_settings(self.base_dir, body)
        else:
            project_dir = _data.known(self.base_dir, body.get("project_dir"))
            group = self.extensions.settings_group(body["scope"])
            _checked_save(named, lambda: _data.update_settings(self.base_dir, body, named),
                          lambda: group.testuary_settings_files(project_dir))
        return {"changed": True}

    def model_lists(self, project_dirs: Sequence[Path]) -> list:
        """The lists of model catalogue files the settings groups name for
        each project of ``project_dirs``, each with its group's key and
        title, the project's directory and name (``_named_entry``)."""
        from testuary.settings import file_locations

        own = Path(file_locations(self.base_dir).models).resolve()
        groups = [(key, group) for key, group in self.extensions.settings_groups().items()
                  if group.testuary_models_files is not None]
        projects = {str(path.resolve()): path.resolve() for path in project_dirs} if groups else {}
        return [{**entry, "project_dir": str(project), "name": project.name}
                for project in projects.values() for key, group in groups
                for entry in [_named_entry(key, group, project, own, models=True)] if entry]

    def models_view(self, dir_text: str, scope: str) -> dict:
        """The Models page: the user directory's catalogue, or with ``scope``
        the list of model catalogue files a settings group names for the
        listed project ``dir_text``, as its runs read the list (``list``),
        and the lists of every listed project (``scopes``)."""
        from testuary.settings import file_locations

        scopes = self.model_lists(_data.listed(self.base_dir))
        if scope in _OWN_SCOPES:
            catalogue = self.catalogue()
            return {**_data.models(self.base_dir), "default_model": catalogue["default_model"],
                    "project_model": catalogue["project_model"], "list": None, "scopes": scopes}
        try:
            project_dir, entry = self._model_list(dir_text, scope)
        except (_data.Refusal, Refused) as refusal:
            return {"error": str(refusal), "models": [], "providers": [], "unused": [], "file": None,
                    "list": None, "scopes": scopes}
        own = str(Path(file_locations(self.base_dir).models).resolve())
        try:
            view = _data.models(project_dir, entry["files"], **self._group_settings(project_dir, scope))
        except _data.Refusal as refusal:
            view = {"error": str(refusal), "models": [], "providers": [], "unused": [], "file": None}
        return {**view, "list": entry, "scopes": scopes, "own_file": own}

    def _model_list(self, dir_text: object, scope: object, change: bool = False):
        """The listed project ``dir_text`` and the list of model catalogue
        files ``scope`` names there (``_chosen``), refused with the reason
        for a ``change`` while its last file is read-only."""
        project_dir = _data.known(self.base_dir, dir_text)
        entry = _chosen(self.model_lists([project_dir]), scope, "list of model catalogue files")
        if entry is None:
            raise Refused(404, "Choose a list of model catalogue files.", "scope")
        if change and entry["read_only"]:
            raise _data.Refusal(entry["read_only"])
        return project_dir, entry

    def set_model(self, body: dict) -> dict:
        """A model change in the user directory's catalogue, else in the last
        file of a settings group's list, which the group checks."""
        if body.get("scope", "") in _OWN_SCOPES:
            self._save_catalogue(self._own_catalogue(), lambda: _data.set_model(self.base_dir, body))
            return {"changed": True}
        project_dir, entry = self._model_list(body.get("project_dir"), body["scope"], change=True)
        group = self.extensions.settings_group(body["scope"])
        self._save_catalogue(Path(entry["file"]), lambda: _data.set_model(project_dir, body, entry["files"]),
                             lambda: group.testuary_models_files(project_dir))
        return {"changed": True}

    def remove_model(self, alias: str, body: dict) -> dict:
        """A model's removal from the user directory's catalogue, else from
        the last file of a settings group's list, which the group checks. A
        model only an earlier file of the list defines is removed by ending
        the import (``_end_import``), when the group names a function for
        it. A section of the last file goes, a provider's included."""
        from testuary.settings import _read_ini_file

        if body.get("scope", "") in _OWN_SCOPES:
            self._save_catalogue(self._own_catalogue(), lambda: _data.remove_model(self.base_dir, alias),
                                 removed=[alias])
            return {"changed": True}
        project_dir, entry = self._model_list(body.get("project_dir"), body["scope"], change=True)
        group = self.extensions.settings_group(body["scope"])

        def ask():
            return group.testuary_models_files(project_dir)

        if alias in _read_ini_file(Path(entry["file"])):
            self._save_catalogue(Path(entry["file"]), lambda: _data.remove_model(project_dir, alias, entry["files"]),
                                 ask, [alias])
            return {"changed": True}
        if group.end_models_import is None:
            raise _data.Refusal(f"{alias} comes from an earlier file of this list, and {group.title} cannot "
                                "end its import, so Edit can replace the model but it cannot be removed.")
        variables = body.get("key_variables") or {}
        if not isinstance(variables, dict) or not all(isinstance(value, str) for value in variables.values()):
            raise _data.Refusal("Send each key variable as text, by its section.")
        return _end_import(project_dir, entry, group, alias, variables,
                           lambda path, write: self._save_catalogue(path, write, ask, [alias]))

    def test_model(self, alias: str, body: dict) -> dict:
        """A model's connection test, as runs with the user directory's
        catalogue reach the service, else as the extension's runs do with a
        settings group's list and its named settings files."""
        if body.get("scope", "") in _OWN_SCOPES:
            return _data.test_model(self.base_dir, alias)
        project_dir, entry = self._model_list(body.get("project_dir"), body["scope"])
        return _data.test_model(project_dir, alias, entry["files"], self._named_config(project_dir, body["scope"]))

    def _named_config(self, project_dir: Path, scope: str) -> Optional[list]:
        """The settings files the runs of the group ``scope`` in the project
        read in place of Testuary's own (``testuary_settings_files``), its
        last file once it exists, or None when the group names none."""
        named = next((item for item in self.named_files(str(project_dir)) if item["scope"] == scope), None)
        if named is not None and named["error"]:
            raise _data.Refusal(named["error"])
        return None if named is None else named["files"] if named["exists"] else named["files"][:-1]

    def _group_settings(self, project_dir: Path, scope: str) -> dict:
        """The keywords ``resolve_settings`` reads the runs of the group
        ``scope`` in the project with: the group's named settings files, or
        Testuary's own with the project's settings when it names none."""
        config = self._named_config(project_dir, scope)
        return {"config_files": config, "project_settings": config is None}

    def _own_catalogue(self) -> Path:
        from testuary.settings import file_locations

        return Path(file_locations(self.base_dir).models).resolve()

    def _readers(self, path: Path) -> list:
        """Every list of model catalogue files that holds ``path``, each with
        the keywords ``resolve_settings`` reads its runs' settings with:
        Testuary's own catalogue with the working directory's settings, and
        each listed project's list a settings group names, with the group's
        named settings files as the connection test reads them. A list or
        settings file a group cannot name is left out, as its page shows
        its error."""
        readers = [([str(path)], {"base_dir": self.base_dir})] if path == self._own_catalogue() else []
        for entry in self.model_lists(_data.listed(self.base_dir)):
            if entry["error"] or str(path) not in entry["files"]:
                continue
            try:
                settings = self._group_settings(Path(entry["project_dir"]), entry["scope"])
            except _data.Refusal:
                continue
            readers.append((entry["files"], {"base_dir": Path(entry["project_dir"]), **settings}))
        return readers

    def _save_catalogue(self, path: Path, write, ask=None, removed: Sequence[str] = ()) -> None:
        """``write`` the model catalogue file ``path``, then ``ask`` the group
        of its list, when given, and check every list that holds the file as
        its runs read it (``_readers``): each model that resolved before,
        but those ``removed`` that the list no longer holds, must still
        resolve (``setup._resolving``).
        Otherwise the file is put back (``_checked_save``)."""
        readers = self._readers(path)
        before = _resolving_lists(readers)

        def check():
            if ask is not None:
                ask()
            for (files, _), (was, _), (now, held) in zip(readers, before, _resolving_lists(readers)):
                broken = sorted(was - now - (set(removed) - held))
                if broken:
                    raise ValueError(f"This change would leave {', '.join(broken)} without a provider, model, "
                                     f"address or protocol for the runs that read {', '.join(files)}, so it was "
                                     "not kept.")

        _checked_save(path, write, check)

    def update_group(self, body: dict) -> dict:
        """A change of an extension's settings through its validated update."""
        key = body.get("group")
        group = self.extensions.settings_group(key) if isinstance(key, str) else None
        if group is None:
            raise Refused(404, "No such settings group.", "group")
        values, remove = body.get("values") or {}, body.get("remove") or []
        if (not isinstance(body.get("section"), str) or not isinstance(values, dict)
                or not isinstance(remove, list)
                or not all(isinstance(item, str) for item in [*values, *values.values(), *remove])):
            raise _data.Refusal("Send a section, its values as text and the keys to reset.")
        project_dir = _data.known(self.base_dir, body.get("project_dir")) if body.get("project_dir") else None
        try:
            group.update(project_dir, body["section"], values, remove)
        except ValueError as refusal:
            raise _data.Refusal(str(refusal)) from None
        return {"changed": True}


def _extension_answer(what: str, read, *arguments) -> dict:
    """What an extension's ``read`` returns for ``arguments``, checked to be
    JSON. A ``ValueError`` it raises refuses the request with its message.
    Any other error, and an answer JSON cannot hold, answers with the error,
    and the other routes and pages keep working."""
    try:
        try:
            answer = read(*arguments)
        except ValueError as refusal:
            raise Refused(400, str(refusal)) from None
        answer = dict(answer)
        json.dumps(answer, allow_nan=False)
    except Refused:
        raise
    except Exception as error:
        raise Refused(500, f"The extension's {what} failed: {type(error).__name__}: {error}") from None
    return answer


def _nested_runs(kind, job: _jobs.Job) -> dict:
    """The Testuary runs a job of an extension's kind started, as its
    ``nested_runs`` names them. A function that fails gives none and its
    error, and the rest of the job's view stays as it is."""
    try:
        runs = list(kind.nested_runs(job.arguments[2:]))
        if not all(isinstance(run, web_extensions.NestedRun) for run in runs):
            raise TypeError("it returned something other than NestedRun")
        return {"nested_runs": [{"phase": run.phase, "path": os.fspath(run.path), "title": run.title}
                                for run in runs]}
    except Exception as error:
        return {"nested_runs": [],
                "nested_runs_error": f"The extension's nested runs failed: {type(error).__name__}: {error}"}


def _detects(dir_type, named_dir: Path) -> bool:
    """Whether ``named_dir`` is of ``dir_type``. A test that fails says no."""
    try:
        return bool(dir_type.detect(named_dir))
    except Exception:
        return False


#: The scopes of Testuary's own settings files: the global and the project's.
_OWN_SCOPES = ("", "global", "project")


def _chosen(scopes: list, scope: object, what: str = "settings file") -> Optional[dict]:
    """The list of ``scopes`` that ``scope`` names, or None for Testuary's
    own files. A scope the project has no list for, and a list the group
    cannot name, are refused."""
    if scope in _OWN_SCOPES:
        return None
    entry = next((entry for entry in scopes if entry["scope"] == scope), None)
    if entry is None:
        raise Refused(404, f"This project has no such {what}.", "scope")
    if entry["error"]:
        raise _data.Refusal(entry["error"])
    return entry


def _named_entry(key: str, group, project_dir: Path, own_file: Path, models: bool = False) -> Optional[dict]:
    """The settings files ``group`` names for the project ``project_dir``, or
    with ``models`` its model catalogue files, with their links followed,
    the last one (``file``) where a change goes, or None when it names none.
    An earlier file must be a file, as runs read it, and the last one may
    not exist yet but may not be a directory. ``read_only`` says why a
    change cannot go to the last file: it is the user's own file
    (``own_file``, ``global``), Testuary's global configuration or model
    catalogue, which the page's own choice changes, or it lies outside the
    project directory and, for model catalogue files, the group's
    ``extension_user_dir``. A failing function gives its reason."""
    what = "model catalogue files" if models else "settings files"
    entry = {"scope": key, "title": group.title, "files": [], "file": None, "exists": False,
             "read_only": None, "global": False, "error": None}
    try:
        named = (group.testuary_models_files if models else group.testuary_settings_files)(project_dir)
    except Exception as error:
        return {**entry, "error": f"The extension's {what} failed: {type(error).__name__}: {error}"}
    if not named:
        return None
    paths = [(project_dir / name).resolve() for name in named]
    for index, (name, path) in enumerate(zip(named, paths)):
        if path.is_dir() or (index < len(paths) - 1 and not path.is_file()):
            return {**entry, "error": f"{group.title} names {name}, which is not a file."}
    last = paths[-1]
    return {**entry, "files": [str(path) for path in paths], "file": str(last), "exists": last.is_file(),
            "read_only": _read_only(group, named[-1], last, project_dir, own_file, models),
            "global": last == own_file, **({"end_import": group.end_models_import is not None,
                                            "in_project": last.is_relative_to(project_dir.resolve())}
                                           if models else {})}


def _read_only(group, name: object, last: Path, project_dir: Path, own_file: Path, models: bool) -> Optional[str]:
    """Why a change cannot go to the last file ``last`` of a list ``group``
    names, which it names ``name`` (``_named_entry``), or None."""
    if last == own_file:
        return ("These runs read your model catalogue last. Change it under Your model catalogue." if models
                else "These runs read your global settings last. Change them on the global settings page.")
    allowed = [project_dir.resolve()] + (
        [Path(group.extension_user_dir).resolve()] if models and group.extension_user_dir else [])
    if any(last.is_relative_to(path) for path in allowed):
        return None
    places = " and the extension's user directory" if len(allowed) > 1 else ""
    return (f"{group.title} names {name} last, which leads to {last}, outside the project directory{places}, "
            "so this page does not change it.")


def _checked_save(path: Path, write, ask) -> None:
    """``write`` the file ``path`` of a settings group's list, then ``ask``
    the group's list function again. A function that raises refuses the
    saved file: ``path`` returns to the bytes it held, or is removed when
    the write created it, and the error's message is the refusal."""
    previous = path.read_bytes() if path.is_file() else None
    write()
    try:
        ask()
    except Exception as error:
        reason = str(error) or type(error).__name__
        failure = _put_back(path, previous)
        raise _data.Refusal(reason if failure is None else
                            f"{reason} {path} could not be put back: {failure}. It holds the refused change.")


def _resolving_lists(readers: list) -> list:
    """For each list of catalogue files and its settings keywords
    (``App._readers``), the models its runs resolve now and the sections
    it holds, none for a list that cannot be read."""
    from testuary import setup
    from testuary.settings import resolve_settings

    found = []
    for files, options in readers:
        try:
            catalogue = setup._catalogue(resolve_settings(models_files=[Path(name) for name in files], **options))
            found.append((setup._resolving(catalogue, options), set(catalogue)))
        except (ValueError, OSError):
            found.append((set(), set()))
    return found


def _put_back(path: Path, previous: Optional[bytes]) -> Optional[OSError]:
    """Put ``path``, which a write just changed, back as it was: the bytes
    it held, or no file when ``previous`` is None because the write created
    it. Returns the error when that fails."""
    from testuary.settings import write_atomic

    try:
        if previous is None:
            path.unlink(missing_ok=True)
        else:
            write_atomic(path, previous.decode("utf-8"))
    except OSError as failure:
        return failure
    return None


def _end_import(project_dir: Path, entry: dict, group, alias: str, variables: dict, save) -> dict:
    """Remove ``alias``, which only an earlier file of the list ``entry``
    defines, by ending the import of the earlier files: the last file takes
    every other model of the joined list with the key ``variables``
    (``setup._ended_import_text``), ``save`` writes and checks it as any
    save (``App._save_catalogue``), the group's
    ``end_models_import`` runs, and its list must then name the last file
    alone. Otherwise the last file is put back, as ``_put_back`` can, and
    the refusal says what holds now. The answer carries a warning when the
    import ended but the function reported an error, or when the list
    cannot be read again."""
    from testuary import setup
    from testuary.settings import write_atomic

    last, earlier = Path(entry["file"]), ", ".join(entry["files"][:-1])
    try:
        text = setup._ended_import_text(setup._listed(project_dir, None, entry["files"]), alias,
                                        variables, project_dir, None)
    except setup.SetupError as error:
        raise _data.Refusal(str(error), error.problems) from None

    def ask():
        return group.testuary_models_files(project_dir)

    previous = last.read_bytes() if last.is_file() else None
    save(last, lambda: (last.parent.mkdir(parents=True, exist_ok=True), write_atomic(last, text)))
    try:
        group.end_models_import(project_dir)
        raised = None
    except Exception as error:
        raised = str(error) or type(error).__name__
    try:
        files = [(project_dir / name).resolve() for name in ask() or ()]
    except Exception as error:
        return {"changed": True, "warning": f"Testuary cannot tell whether the import of {earlier} ended: "
                                            f"{error}. {last} holds a copy of every other model, so the runs "
                                            "read the same models either way."}
    if files == [last]:
        return {"changed": True, "warning": raised and (
            f"The import of {earlier} ended, and its function reported: {raised}. {last} holds every other model.")}
    failure = _put_back(last, previous)
    raise _data.Refusal(
        f"The import of {earlier} did not end{': ' + raised if raised else ''}, and {last} could not be put "
        f"back: {failure}. {last} holds a copy of every model of {earlier}, which the runs read as "
        "overrides." if failure
        else f"The import of {earlier} did not end: {raised}. {last} is as it was, and the models of "
             f"{earlier} still apply." if raised
        else f"The import of {earlier} did not end: the runs still read {', '.join(map(str, files))}. "
             f"{last} is as it was.")


def _group_sections(key: str, group, project_dir: Optional[Path]) -> list:
    """An extension's settings group as the Settings page's sections."""
    import dataclasses

    try:
        values = group.values(project_dir)
        sections = list(group.reference(project_dir))
    except Exception as error:
        return [{"name": key, "title": group.title, "group": key, "settings": [],
                 "purpose": f"These settings could not be read: {type(error).__name__}: {error}"}]
    return [{"name": section.name, "title": section.title, "purpose": section.purpose, "group": key,
             "settings": [{**dataclasses.asdict(setting), "label": setting.label, "group": key,
                           "value": values.get((section.name, setting.name)),
                           "source": group.title, "sources": [], "active": True, "inactive": ""}
                          for setting in section.settings]}
            for section in sections]


class Handler(BaseHTTPRequestHandler):
    server_version = "testuary-web"
    sys_version = ""
    protocol_version = "HTTP/1.1"
    # A connection that sends or takes nothing for this long is closed.
    timeout = 30

    @property
    def app(self) -> App:
        return self.server.app

    def log_message(self, format, *args) -> None:
        # Requests are not logged: their addresses can carry directory names.
        pass

    # Access

    def _origins(self) -> set:
        port = self.server.server_address[1]
        return {f"127.0.0.1:{port}", f"localhost:{port}"}

    def _host_ok(self) -> bool:
        return self.headers.get("Host", "") in self._origins()

    def _token_ok(self) -> bool:
        if self.app.token is None:
            return True
        # Header values arrive decoded as Latin-1, so any byte compares.
        given = self.headers.get(TOKEN_HEADER, "").encode("utf-8")
        return hmac.compare_digest(given, self.app.token.encode("ascii"))

    def _origin_ok(self) -> bool:
        origin = self.headers.get("Origin")
        return origin is None or origin in {f"http://{host}" for host in self._origins()}

    # Responses

    def _send(self, status: int, body: bytes, content_type: str, extra=()) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        for name, value in extra:
            self.send_header(name, value)
        if self.close_connection:
            self.send_header("Connection", "close")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_error(self, code, message=None, explain=None) -> None:
        # The standard library's own refusals (an unknown method, a malformed
        # request) as plain text with the security headers.
        self.close_connection = True
        text = message or HTTPStatus(code).phrase
        self._send(code, text.encode("utf-8", "replace"), "text/plain; charset=utf-8")

    def _json(self, value, status: int = 200) -> None:
        body = json.dumps(value, ensure_ascii=True, allow_nan=False).encode("ascii")
        self._send(status, body, "application/json")

    def _refuse(self, status: int, message: str, field: Optional[str] = None,
                problems: Sequence[Sequence[str]] = ()) -> None:
        if self.path.startswith("/api/"):
            answer = {"error": message, "field": field or (problems[0][0] if problems else None)}
            if problems:
                answer["problems"] = [list(item) for item in problems]
            self._json(answer, status)
        else:
            self._send(status, message.encode("utf-8"), "text/plain; charset=utf-8")

    def _static(self, name: str) -> None:
        if name not in STATIC:
            return self._refuse(404, "Not found.")
        body = resources.files("testuary.web").joinpath("static", name).read_bytes()
        self._send(200, body, STATIC[name])

    def _extension_file(self, rest: str) -> None:
        """A view script or the guide file an extension declared, from its package."""
        extension_id, _, name = rest.partition("/")
        body = self.app.extensions.package_file(extension_id, name)
        if body is None:
            return self._refuse(404, "Not found.")
        self._send(200, body, "text/html; charset=utf-8" if name.endswith(".html")
                   else "text/javascript; charset=utf-8")

    def _length(self) -> int:
        """The body length the request states: a plain byte count up to MAX_BODY."""
        given = self.headers.get("Content-Length")
        if given is None:
            raise Refused(411, "The request must state its length.")
        if not re.fullmatch(r"[0-9]{1,12}", given.strip()):
            raise Refused(400, "The request's length is not a number of bytes.")
        length = int(given)
        if length > MAX_BODY:
            raise Refused(413, "The request is too large.")
        return length

    def _stop(self) -> None:
        """A stop request of testuary-web --stop or --replace: with the
        start's stop secret, the interface ends as Ctrl+C ends it."""
        self._discard_body()
        given = self.headers.get(STOP_HEADER, "").encode("utf-8")
        if not hmac.compare_digest(given, self.app.stop_secret.encode("ascii")):
            return self._refuse(403, "This request does not carry this interface's stop secret.")
        if self.app.stopper is None:
            return self._refuse(409, "This interface is not stopped from outside: end it where it runs.")
        self._json({"stopping": True}, 202)
        self.app.stopper()

    def _discard_body(self) -> None:
        """Read a refused request's body when its length allows, so the refusal
        reaches the client before the connection closes."""
        try:
            self.rfile.read(self._length())
        except (Refused, OSError):
            pass

    def _body(self) -> dict:
        raw = self.rfile.read(self._length())
        if not (self.headers.get("Content-Type") or "").startswith("application/json"):
            raise Refused(415, "The request must be JSON.")
        try:
            body = json.loads(raw or b"{}")
        except ValueError:
            raise Refused(400, "The request is not valid JSON.") from None
        if not isinstance(body, dict):
            raise Refused(400, "The request must be a JSON object.")
        return body

    # Routes

    def do_GET(self) -> None:
        url = urlsplit(self.path)
        if not self._host_ok():
            return self._refuse(403, "This server answers only at its own address.")
        if url.path == "/":
            return self._static("index.html")
        if url.path.startswith("/static/"):
            return self._static(url.path[len("/static/"):])
        if url.path.startswith("/ext/"):
            return self._extension_file(url.path[len("/ext/"):])
        if not self._token_ok():
            return self._refuse(403, "Open the address testuary-web printed when it started.")
        try:
            self._get(url)
        except Refused as refusal:
            self._refuse(refusal.status, str(refusal), refusal.field)
        except (_results.NotShown, _data.NotListed) as refusal:
            self._refuse(404, str(refusal))
        except _results.NotAdopted as refusal:
            self._refuse(400, str(refusal), refusal.field)
        except _data.Refusal as refusal:
            self._refuse(400, str(refusal), problems=_data.problems(refusal))

    def do_OPTIONS(self) -> None:
        # No cross-origin request is allowed, so no preflight succeeds.
        self.close_connection = True
        self._send(405, b"Not allowed.", "text/plain; charset=utf-8", (("Allow", "GET, POST"),))

    def _get(self, url) -> None:
        route = _GET_ROUTES.get(url.path)
        if route is not None:
            return self._json(route(self, parse_qs(url.query)))
        if url.path == "/api/nested-run":
            return self._nested_run(parse_qs(url.query))
        found = _ROUTE.fullmatch(url.path)
        if found:
            return self._json(self.app.extension_route(found.group(1), found.group(2),
                                                       parse_qs(url.query)))
        match = _JOB.fullmatch(url.path)
        if match and match.group(2) in (None, "/stream"):
            job = self.app.jobs.get(match.group(1))
            if job is None:
                raise Refused(404, "No such job.")
            if match.group(2) is None:
                return self._json(self.app.job_view(job))
            return self._job_stream(job)
        raise Refused(404, "Not found.")

    def _run_dir(self, query) -> Path:
        text = _first(query, "run_dir")
        return _results.locate(self.app.results_base(text), text)

    def do_POST(self) -> None:
        # A request's connection is not used again: a body with a wrong or
        # too large length stays unread.
        self.close_connection = True
        url = urlsplit(self.path)
        if not self._host_ok():
            self._discard_body()
            return self._refuse(403, "This server answers only at its own address.")
        if url.path == "/api/stop":
            return self._stop()
        if not self._token_ok() or not self._origin_ok():
            self._discard_body()
            return self._refuse(403, "This request did not come from this interface's page.")
        try:
            body = self._body()
            entry, match = _ENTRY.fullmatch(url.path), _JOB.fullmatch(url.path)
            if url.path in _POST_ROUTES:
                answer = _POST_ROUTES[url.path](self, body)
            elif entry:
                answer = _MODEL_ROUTES[entry.group(2)](self, unquote(entry.group(1)), body)
            elif match and match.group(2) in _JOB_ROUTES:
                answer = _JOB_ROUTES[match.group(2)](self, match.group(1))
            else:
                raise Refused(404, "Not found.")
            # An answer is a dict, or a dict and its status for a created job.
            return self._json(*answer) if isinstance(answer, tuple) else self._json(answer)
        except Refused as refusal:
            self._refuse(refusal.status, str(refusal), refusal.field)
        except _results.NotAdopted as refusal:
            self._refuse(400, str(refusal), refusal.field)
        except _results.NotShown as refusal:
            self._refuse(404, str(refusal))
        except _data.Refusal as refusal:
            self._refuse(400, str(refusal), problems=_data.problems(refusal))
        except _jobs.Busy as busy:
            self._refuse(409, str(busy))
        except _jobs.NotAllowed as refusal:
            self._refuse(409, str(refusal))

    # The job stream

    def _event(self, kind: str, data: str) -> None:
        self.wfile.write(f"event: {kind}\ndata: {data}\n\n".encode("utf-8"))

    def _job_stream(self, job: _jobs.Job) -> None:
        """A job's server-sent events: its event file's, its terminal text
        (``text``) and, once its process has ended, its final state
        (``end``)."""
        # A check's output is its report, which the end view carries.
        texts = () if job.kind == "check" else (("stdout", job.stdout), ("stderr", job.stderr))
        view = _jobs.Progress(job.cwd or self.app.base_dir) if job.events is not None else None
        self._stream(job.events, view, texts, lambda: job.ended is not None, lambda: self.app.job_view(job))

    def _nested_run(self, query) -> None:
        """A nested run's server-sent events (``testuaryWeb.nestedRun``): an
        event file's until its run has ended, then ``end``, or ``end`` with
        what the Results view shows of a run directory."""
        text = _first(query, "path")
        found = _results._named_run(self.app.results_base(text), text)
        if isinstance(found, Path):
            view = _jobs.Progress(self.app.base_dir)
            return self._stream(found, view, (), view._ended, dict)
        summary = _results_view(self, found)
        return self._stream(None, None, (), lambda: True, lambda: {"results": summary})

    def _stream(self, events, view, texts, ended, end) -> None:
        """Server-sent events: Testuary's view model ``view`` of the event
        file ``events`` (``view``, ``log``, ``summary``), the lines of the
        files ``texts`` (``text``), and ``end()`` (``end``) once ``ended()``
        and nothing more is read. Every connection replays them from their
        start."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        for name, value in SECURITY_HEADERS.items():
            self.send_header(name, value)
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        offsets = {"events": 0, **{name: 0 for name, _ in texts}}
        logged = 0
        summarized = False
        quiet_since = time.monotonic()
        try:
            while True:
                over = ended()
                sent = False
                lines, offsets["events"] = _jobs.read_lines(events, offsets["events"])
                read = bool(lines)
                changed = False
                for line in lines if view is not None else ():
                    try:
                        changed = view.add(line) or changed
                    except (ValueError, TypeError, KeyError):
                        # A line that is no observation stays out of the view.
                        continue
                if changed:
                    for line in view.log[logged:]:
                        self._event("log", json.dumps(line))
                    logged = len(view.log)
                    self._event("view", json.dumps(view.snapshot()))
                    if view.summary is not None and not summarized:
                        self._event("summary", json.dumps(view.summary))
                        summarized = True
                    sent = True
                for name, path in texts:
                    lines, offsets[name] = _jobs.read_lines(path, offsets[name])
                    read = read or bool(lines)
                    for line in lines:
                        self._event("text", json.dumps({"stream": name, "line": line}))
                        sent = True
                if over and not read:
                    self._event("end", json.dumps(end()))
                    self.wfile.flush()
                    return
                if sent:
                    self.wfile.flush()
                    quiet_since = time.monotonic()
                elif time.monotonic() - quiet_since > KEEPALIVE_SECONDS:
                    self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
                    quiet_since = time.monotonic()
                else:
                    time.sleep(STREAM_POLL_SECONDS)
        except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
            return


def _first(query: dict, name: str) -> str:
    """A query's first value of ``name``, else the empty text."""
    return query.get(name, [""])[0]


def _changed(_done) -> dict:
    """The answer of a change, once the call that made it returned."""
    return {"changed": True}


def _results_view(handler, run_dir: _results.Dir) -> dict:
    """A run directory's results, the listed project it belongs to (None
    outside every listed project) and the extensions' directory types it is."""
    try:
        project = str(handler.app.results_project(str(run_dir.root)))
    except Refused:
        project = None
    return {**_results.summary(run_dir, handler.app.jobs.event_files()), "project_dir": project,
            "views": handler.app.dir_views(run_dir.root)}


def _projects_view(listing: dict) -> dict:
    return {**listing, "settings": _data.project_settings(),
            "language_settings": _data.language_settings(listing["languages"])}


#: The page's reads: each path's answer, from the handler and the parsed
#: query.
_GET_ROUTES = {
    "/api/state": lambda h, q: h.app.state(),
    "/api/demo/settings": lambda h, q: h.app.demo_settings(),
    "/api/catalogue": lambda h, q: h.app.catalogue(),
    "/api/models": lambda h, q: h.app.models_view(_first(q, "project_dir"), _first(q, "scope")),
    "/api/projects": lambda h, q: _projects_view(_data.listing(h.app.base_dir)),
    "/api/project/results": lambda h, q: _results.project_runs(
        _data.known(h.app.base_dir, _first(q, "project_dir"))),
    "/api/run/saved": lambda h, q: _data.saved_run(
        h.app.base_dir, _first(q, "project_dir"), _first(q, "model"), _first(q, "output_dir")),
    "/api/project/files": lambda h, q: _data.source_files(h.app.base_dir, _first(q, "project_dir")),
    "/api/settings": lambda h, q: h.app.settings_view(_first(q, "project_dir"), _first(q, "scope")),
    "/api/dirs": lambda h, q: h.app.dirs(_first(q, "path")),
    "/api/results": lambda h, q: _results_view(h, h._run_dir(q)),
    "/api/ext/dir": lambda h, q: h.app.dir_view(_first(q, "type"), _results.named_dir(
        h.app.results_base(_first(q, "dir")), _first(q, "dir"))),
    "/api/results/adopt": lambda h, q: h.app.adoption(_first(q, "run_dir"), _first(q, "path")),
    "/api/results/file": lambda h, q: _results.read(h._run_dir(q), _first(q, "path")),
}

#: The page's changes: each path's answer, or the answer and its status,
#: from the handler and the request's body.
_POST_ROUTES = {
    "/api/demo": lambda h, b: (h.app.job_view(h.app.start_demo(b)), 201),
    "/api/check": lambda h, b: (h.app.job_view(h.app.start_check(b)), 201),
    "/api/run": lambda h, b: (h.app.job_view(h.app.start_run(b)), 201),
    "/api/models": lambda h, b: h.app.set_model(b),
    "/api/projects/forget": lambda h, b: _changed(_data.forget(b.get("project_dir"))),
    "/api/projects/clone": lambda h, b: {
        "project_dir": str(_data.clone(b.get("url"), b.get("parent_dir")))},
    "/api/project": lambda h, b: _changed(_data.set_project(h.app.base_dir, b)),
    "/api/project/setup": lambda h, b: _data.step(b),
    "/api/settings": lambda h, b: h.app.update_settings(b),
    "/api/ext/settings": lambda h, b: h.app.update_group(b),
    "/api/ext/jobs": lambda h, b: (h.app.job_view(h.app.start_extension_job(b)), 201),
    "/api/results/adopt": lambda h, b: h.app.adopt(b),
}

#: A model's actions (``/api/models/ALIAS/ACTION``), from the handler, the
#: alias and the request's body, which names a list of catalogue files by
#: its project and scope. No answer holds a stored key.
_MODEL_ROUTES = {
    "test": lambda h, alias, b: h.app.test_model(alias, b),
    "remove": lambda h, alias, b: h.app.remove_model(alias, b),
}

#: A job's actions (``/api/jobs/ID/ACTION``), from the handler and the id.
_JOB_ROUTES = {
    "/stop": lambda h, job: h.app.job_view(h.app.jobs.stop(job)),
    "/force": lambda h, job: h.app.job_view(h.app.jobs.force(job)),
    "/continue": lambda h, job: (h.app.job_view(h.app.continue_run(job)), 201),
}


class Server(ThreadingHTTPServer):
    daemon_threads = True
    # A restart on the same port must not wait for the last start's closed
    # connections (TIME_WAIT, a minute or two on Linux and macOS), so POSIX
    # reuses the address, which still refuses a second live listener there.
    # Windows' SO_REUSEADDR would let another socket share a live port: there
    # the port is bound exclusively instead, and closed connections never
    # held it.
    allow_reuse_address = os.name != "nt"

    def server_bind(self) -> None:
        if os.name == "nt":
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()

    def __init__(self, app: App, port: int = 0) -> None:
        super().__init__(("127.0.0.1", port), Handler)
        self.app = app
