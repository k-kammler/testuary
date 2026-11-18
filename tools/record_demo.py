"""Record a demonstration scenario during a real run of this release.

The run is the package's own command with the demonstration's invocation,
through the same seams the playback answers at: every model request and
every operation that starts a process. Each call goes to the real model or
operation, and its answer is stored as the playback will read it. The
machine's paths are stored as placeholders (``demo.scenario``'s path map).
The result is a data directory in the layout ``testuary --demo`` loads, its
scenario.json.gz written by ``dump``.

Each call is declared once, in ``testuary.demo.recorded``: its key, what the
run receives from a stored answer and how a real result is stored. The
recorder's backends, mutation engine, model source and cover source are the
playback's with their real counterparts, and its oracle answers each call
with the real method the declaration names, called with the run's own
arguments. A new argument of a recorded method needs a change only in the
declaration's key, and only when it changes the answer.

This development command contacts the configured model service, which may
cost money, and runs the project's tests and tools. It is not part of a
normal run or of playback. With ``--replay-model`` it contacts no model: each
model request is answered from an earlier recording of the same run by the
key the playback matches, and a request that recording does not hold stops
the run, so only the operations are recorded anew. The per-test and suite
times of each eviction measurement also come from that recording, when the
rest of the measurement matches it, so that the exact cover is the same.

A development tool of the repository, outside the package and its
distributions: ``python tools/record_demo.py --help``.
"""
from __future__ import annotations

import argparse
import contextlib
import gzip
import json
import re
import os
import shutil
import subprocess
import sys
import tempfile
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

from pydantic import PrivateAttr

from testuary import reduction
from testuary.demo import backend as playback
from testuary.demo import recorded
from testuary.demo.scenario import (
    SCHEMA_VERSION,
    SEGMENTS,
    Call,
    Entry,
    Oracle,
    PathMap,
    Scenario,
    ScenarioStop,
    canonical,
    install_paths,
    local_paths,
    neutral_paths,
)
from testuary.generation_evidence import source_sha256
from testuary.languages.generic import GenericBackend

RULE = "recorded-v1: the answer the real model or operation gave during the recorded run"
STATEMENT = (
    "DEMO, not a measurement. Every model answer and measurement result was "
    "recorded in one real run of this Testuary release. The playback runs no "
    "model, test, coverage or mutation tool, and contacts no service.")
# Project files that ship with a recording besides the target: the package
# markers of a Python target, so it imports as in the recorded run.
_PROJECT_IGNORE = shutil.ignore_patterns(
    ".git", "__pycache__", ".pytest_cache", ".mypy_cache", ".tox", ".venv",
    "venv", "target", "build", "dist", "*.egg-info", ".idea", ".vscode")


# The operations whose answers hold measured times that the exact cover's
# costs are made of: in a replay recording they come from the earlier
# recording, so that the cover, and every model request after it, is the
# same.
TIMED_KINDS = ("backend.collect_eviction_data",)
_TIME_FIELDS = ("per_test_time_ms", "suite_time_ms")


# Strings at least this long are stored once in the scenario's blob table and
# referenced from the entries, so a matrix or a suite that several answers
# carry is kept once.
_BLOB_MIN_CHARS = 2048


def dump(scenario: Scenario, path: Path) -> None:
    """Write ``scenario`` as gzip-compressed JSON, long strings stored once
    in a blob table, as ``Scenario.load`` reads it. The bytes depend only on
    the content."""
    blobs: Dict[str, str] = {}

    def pack(value: Any) -> Any:
        if isinstance(value, str) and len(value) >= _BLOB_MIN_CHARS:
            digest = source_sha256(value)
            blobs[digest] = value
            return {"$blob": digest}
        if isinstance(value, dict):
            return {k: pack(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [pack(v) for v in value]
        return value

    # Concurrent calls (the seed generations, an agent's parallel tools)
    # are answered in varying order. Playback matches by segment, kind
    # and key and keeps the order only among equal keys. A stable sort on
    # those makes the file depend only on the answers.
    ordered = sorted(
        scenario.entries,
        key=lambda entry: (SEGMENTS.index(entry.segment), entry.kind,
                           canonical(entry.key)),
    )
    entries = [pack(entry.as_json()) for entry in ordered]
    document = {
        "schema_version": SCHEMA_VERSION,
        "meta": scenario.meta,
        "entries": entries,
        "blobs": dict(sorted(blobs.items())),
    }
    raw = canonical(document).encode("utf-8")
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 keeps the compressed bytes independent of the build time.
    with open(path, "wb") as handle:
        with gzip.GzipFile(fileobj=handle, mode="wb", mtime=0,
                           filename="") as zipped:
            zipped.write(raw)


def _untimed(answer: Any) -> Any:
    """``answer`` without its measured times."""
    if isinstance(answer, dict):
        return {k: _untimed(v) for k, v in answer.items() if k not in _TIME_FIELDS}
    if isinstance(answer, list):
        return [_untimed(v) for v in answer]
    return answer


def _differences(a: Any, b: Any, prefix: str = "") -> List[str]:
    """The paths at which ``a`` and ``b`` differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b))
                for p in _differences(a.get(k), b.get(k), f"{prefix}.{k}")]
    if a != b:
        return [prefix or "."]
    return []


def _real_object(entry: recorded.RecordedCall, owner: Any, call: Call) -> Any:
    """The object whose method of the entry's name answers a recorded call
    of ``owner``: the language backend past its scenario answers, the real
    measurement of the run, or the recording's real counterpart of
    ``owner`` (``live``): its mutation runner or session, its model or its
    solver."""
    if entry.role == "backend":
        return super(playback.ScenarioOperations, owner)
    if entry.role != "measurement":
        return owner.live
    measurement = owner.runner.measurements.get(owner.run_key)
    if measurement is None:
        raise ScenarioStop(call, "the recorder has no real operation for this call")
    return measurement


class RecordingOracle(Oracle):
    """Answers each call with its real operation and records the answer:
    the real method the call's declaration (``recorded``) names, called with
    the run's arguments.

    A call made while a real operation runs (a backend operation that uses
    another one) is answered by its own real operation and not recorded: the
    playback answers the outer call and never makes the inner one."""

    def __init__(self, *, source: str, reuse: Optional[Scenario] = None,
                 **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.entries: List[Entry] = []
        self._source = source
        self._inside = threading.local()
        # The earlier recording's timing answers, by kind and key in the
        # order recorded.
        self._timings: Dict[Tuple[str, str], List[Entry]] = {}
        self.reused = 0
        self.not_reused: List[str] = []
        for entry in (reuse.entries if reuse is not None else ()):
            if entry.kind in TIMED_KINDS:
                self._timings.setdefault((entry.kind, canonical(entry.key)), []).append(entry)

    def _live(self, call: Call) -> Callable[[], Any]:
        """The real operation that answers ``call``, its answer as stored."""
        entry = call.context.get("recorded")
        if entry is None:
            raise ScenarioStop(call, "the recorder has no real operation for this call")
        arguments = call.context["arguments"]
        real = _real_object(entry, arguments.owner, call)

        def live() -> Any:
            result, collected = recorded.real_result(entry, real, arguments)
            if entry is recorded.MUTATION_RUN:
                # Its survivors' text and kill matrix are asked for by run key.
                arguments.owner.runner.measurements[recorded.run_key(call.key)] = result
            return recorded.stored_answer(entry, result, arguments, collected)

        return live

    def answer(self, call: Call) -> Any:
        live = self._live(call)
        if getattr(self._inside, "depth", 0):
            return json.loads(json.dumps(live()))
        self._inside.depth = 1
        try:
            answer = json.loads(json.dumps(live()))
        finally:
            self._inside.depth = 0
        source = self._source
        if call.kind in TIMED_KINDS:
            with self._lock:
                queue = self._timings.get((call.kind, call.key_text()))
                earlier = queue.pop(0) if queue else None
            # The measurement still runs: its timings are replaced only when
            # everything else it measured is what the earlier recording holds.
            if earlier is not None and _untimed(neutral_paths(answer)) == _untimed(earlier.answer):
                answer, source = local_paths(earlier.answer), earlier.source
                self.reused += 1
            else:
                self.not_reused.append(
                    "no earlier answer for " + call.key_text() if earlier is None else
                    "differs at " + ", ".join(_differences(
                        _untimed(earlier.answer), _untimed(neutral_paths(answer))))[:600])
        entry = Entry(self.segment, call.kind, dict(call.key), neutral_paths(answer),
                      "real", source, RULE)
        with self._lock:
            self.entries.append(entry)
        if self._on_answer is not None:
            self._on_answer(entry)
        return answer

    def _answer_entry(self, call: Call) -> Entry:
        raise NotImplementedError

    def _close_segment(self, segment: str) -> None:
        return None


# The playback's backends and mutation engine as a recording runs them: the
# recorded calls go to the recording oracle, and what the playback states
# without asking is real here. The dependency check, the project probe and
# the test setup are the real ones, and the mutation engine answers through
# the project's real runner.

class RecordingSession(playback.ScenarioMutationSession):
    """A scenario session over the recording's real session (``live``)."""

    live: Any = None

    def worker(self):
        """The scope of one agent tool call: the real session's, as a real
        run's call has, so the real operations it records run on the tool's
        thread."""
        return self.live.worker()

    def close(self) -> None:
        self.live.close()


class _RecordingRunner:
    """A scenario runner over the recording's real runner (``live``). Each
    real mutation run is kept by its run key (``measurements``), for its
    survivors' text and kill matrix."""

    session_type = RecordingSession

    def __init__(self, *args: Any, live: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.live = live
        self.measurements: Dict[str, Any] = {}

    def open(self, workers: int, *, suite_dirs=()):
        live = self.live.open(workers, suite_dirs=suite_dirs)
        session = super().open(workers, suite_dirs=suite_dirs)
        session.live = live
        return session


class RecordingRunner(_RecordingRunner, playback.ScenarioRunner):
    """Python's mutation engine in a recording."""


class RecordingPitRunner(_RecordingRunner, playback.ScenarioPitRunner):
    """PIT in a Java recording."""


class RecordingGenericRunner(_RecordingRunner, playback.ScenarioGenericRunner):
    """The mutation engine of a language registered from outside, in a
    recording."""


def _kept_facts(backend: Any, source_dir: Any = None) -> Any:
    """The real project backend, its language facts and test vocabulary kept
    on the oracle for the playback, which states them."""
    real = GenericBackend.project_backend(backend, source_dir)
    root = str(Path(source_dir or Path.cwd()).resolve())
    facts = getattr(backend._oracle, "language_facts", None) or {}
    facts[neutral_paths(root)] = playback.language_record(real)
    backend._oracle.language_facts = facts
    return real


class _RealChecks:
    """What a run checks and sets up itself: the dependencies and the test
    directory."""

    check_dependencies = GenericBackend.check_dependencies
    setup_test_env = GenericBackend.setup_test_env


class RecordingBackend(_RealChecks, playback.ScenarioBackend):
    """Python's scenario backend in a recording."""

    project_backend = _kept_facts

    def mutation_runner(self, config, *, source_dir=None, log=None):
        live = GenericBackend.mutation_runner(self, config, source_dir=source_dir, log=log)
        return RecordingRunner(self._oracle, config, log=log, live=live)


class RecordingJavaBackend(_RealChecks, playback.ScenarioJavaBackend):
    """Java's scenario backend in a recording."""

    project_backend = _kept_facts

    def mutation_runner(self, config, *, source_dir=None, log=None):
        live = GenericBackend.mutation_runner(self, config, source_dir=source_dir, log=log)
        return RecordingPitRunner(self._oracle, config,
                                  backend=self.project_backend(source_dir), log=log, live=live)


class RecordingGenericBackend(_RealChecks, playback.ScenarioGenericBackend):
    """The scenario backend of a language registered from outside, in a
    recording."""

    def mutation_runner(self, config, *, source_dir=None, log=None):
        live = GenericBackend.mutation_runner(self, config, source_dir=source_dir, log=log)
        return RecordingGenericRunner(self._oracle, config,
                                      engine=live.name, language=self.name,
                                      backend=self.project_backend(source_dir), log=log, live=live)


def recording_backend(language: str, file_extensions: Sequence[str]) -> Callable[[Any], Any]:
    """The recording's backend factory for ``language``, as the demo drivers
    call it: with the oracle only."""
    if language == "java":
        return RecordingJavaBackend
    if language == "python":
        return RecordingBackend

    def make(oracle):
        return RecordingGenericBackend(oracle, language, file_extensions)
    return make


class _ChatReply:
    """A recording's real chat model, asked as the agent asks it."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def _generate(self, messages, stop=None):
        return self.model.invoke(messages, stop=stop)


class RecordingChatModel(playback.ScriptedChatModel):
    """A scenario chat model over the recording's real chat model, with the
    tools of this copy's mode bound."""

    _real: Any = PrivateAttr(default=None)

    @property
    def live(self) -> _ChatReply:
        return _ChatReply(self._real)

    def bind_tools(self, tools: Any, *, tool_choice: Any = None,
                   **kwargs: Any) -> "RecordingChatModel":
        bound = super().bind_tools(tools, tool_choice=tool_choice, **kwargs)
        options = dict(kwargs)
        if tool_choice is not None:
            options["tool_choice"] = tool_choice
        bound._real = self._real.bind_tools(tools, **options)
        return bound


class RecordingModelSource(playback.ScenarioModelSource):
    """The scenario's model source over the recording's model (``live``):
    the configured model, or an earlier recording's answers."""

    chat_model_type = RecordingChatModel

    def __init__(self, oracle: Oracle, live: Any) -> None:
        super().__init__(oracle)
        self.live = live

    def agent_chat_model(self, phase: str) -> RecordingChatModel:
        model = super().agent_chat_model(phase)
        model._real = self.live.agent_chat_model(phase)
        return model


class RecordingCoverSource(playback.ScenarioCoverSource):
    """The scenario's cover source in a recording: the bundled CBC solves
    each problem (``reduction.exact_cover``)."""

    live = reduction


class LiveModelSource:
    """The run's configured model, asked as a normal run asks it."""

    def generate_suite(self, messages, *, temperature, seed, timeout):
        # As starters.run_starter_phase asks the configured provider: the
        # phase's model settings carry the sampling, output token limit and reasoning.
        from testuary import config, starters
        from testuary.models import get_api_config, resolve_phase_model

        model = resolve_phase_model(config.active_settings(), "starter")
        api = get_api_config()
        cfg = config.load_config("starter")
        clients = None
        if model.protocol != "ollama-native":
            from testuary.models import StarterModelClients

            clients = StarterModelClients(model, api_key=api["api_key"])
        grace = cfg.get("reasoning_grace_seconds")
        return starters._default_generate(
            api, messages, seed=seed, reasoning_grace=float(grace) if grace else None,
            model_settings=model, model_clients=clients)

    def agent_chat_model(self, phase: str):
        from testuary import config
        from testuary.models import build_resolved_chat_model, get_api_config, resolve_phase_model

        model = resolve_phase_model(config.active_settings(), phase)
        return build_resolved_chat_model(model, api_key=get_api_config()["api_key"])


class ReplayOracle(Oracle):
    """The model answers of an earlier recording, by kind and request key in
    the order recorded, whatever the segment. A request it does not hold
    stops the run: no model is asked."""

    def __init__(self, scenario: Scenario, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.queues: Dict[Tuple[str, str], List[Entry]] = {}
        for entry in scenario.entries:
            if entry.kind.startswith("model."):
                self.queues.setdefault((entry.kind, canonical(entry.key)), []).append(entry)

    def _answer_entry(self, call: Call) -> Entry:
        with self._lock:
            queue = self.queues.get((call.kind, call.key_text()))
            if not queue:
                raise ScenarioStop(call, "the earlier recording holds no answer for this "
                                         "model request, so its answer cannot be reused")
            return queue.pop(0)

    def _returned(self, entry: Entry) -> Any:
        # As the playback does: the run receives this machine's paths.
        return local_paths(entry.answer)

    def _close_segment(self, segment: str) -> None:
        return None

    def unused(self) -> int:
        """How many of the earlier recording's model answers no request took."""
        return sum(len(queue) for queue in self.queues.values())


def replay_model(scenario: Scenario) -> Tuple[Any, ReplayOracle]:
    """A model source that answers from ``scenario``'s model answers, and its
    oracle. It builds no model client."""
    oracle = ReplayOracle(scenario)
    return playback.ScenarioModelSource(oracle), oracle


#: The results-table columns a playback reproduces from the recorded run.
RESULT_COLUMNS = ("line_coverage", "branch_coverage", "mutation_score", "starter_status",
                  "coverage_status", "mutation_status", "finishing_status")


def recorded_outcome(results: Path, workspace: str) -> Dict[str, Any]:
    """What a run delivered: the suite's files with their content hashes, and
    the headline columns of its results table. A playback of the recording
    delivers the same."""
    import csv
    import hashlib

    from testuary.project import phase_artifact_path

    tests = phase_artifact_path(results / workspace, "tests", 0)
    delivered = {
        path.name: hashlib.sha256(path.read_bytes().replace(b"\r\n", b"\n")).hexdigest()
        for path in sorted(tests.iterdir())
        if path.is_file() and path.name != "DEMO_NOTICE.txt"
    } if tests.is_dir() else {}
    row: Dict[str, str] = {}
    table = results / "tests.csv"
    if table.is_file():
        with open(table, encoding="utf-8-sig", newline="") as handle:
            row = next(csv.DictReader(handle), {}) or {}
    return {"delivered_suite": delivered,
            "results": {name: row.get(name, "") for name in RESULT_COLUMNS}}


def _interpreter_paths(python: str) -> Dict[str, str]:
    """The project interpreter's own path, its site-packages directory and its
    environment's root."""
    probe = subprocess.run(
        [python, "-I", "-c", "import sys, sysconfig; print(sys.executable); "
                             "print(sysconfig.get_paths()['purelib']); print(sys.prefix)"],
        capture_output=True, text=True, timeout=60, check=True)
    executable, site, prefix = probe.stdout.strip().splitlines()[:3]
    return {"python": executable, "site": site, "env": prefix}


def _python_version(python: str) -> str:
    probe = subprocess.run([python, "-I", "-c", "import platform; print(platform.python_version())"],
                           capture_output=True, text=True, timeout=60, check=True)
    return f"Python {probe.stdout.strip()}"


@contextlib.contextmanager
def _catalogue(catalogue_dir: Optional[Path], alias: str,
               fields: Dict[str, str]) -> Iterator[None]:
    """For the duration of the run, Testuary's settings reader
    (``resolve_settings``) reads ``models.ini`` of ``catalogue_dir`` in
    memory, with the demo's alias added. The catalogue may hold a key: it is read only by Testuary's own
    settings reader, and nothing copies, prints or records it."""
    if catalogue_dir is None:
        yield
        return
    from testuary import settings

    original = settings.resolve_settings

    def resolve(*args: Any, **kwargs: Any):
        if kwargs.get("models") is None:
            models = settings._read_ini_file(catalogue_dir / "models.ini")
            models[alias] = dict(fields)
            kwargs["models"] = models
        return original(*args, **kwargs)

    settings.resolve_settings = resolve
    try:
        yield
    finally:
        settings.resolve_settings = original


@contextlib.contextmanager
def _recording_environment(workspace: Path, preset: Path) -> Iterator[None]:
    """The playback's environment, except that the model key variables stay:
    the recording contacts the model."""
    from testuary.demo.cli import own_project

    prefixes = ("TESTUARY_", "LANGADAPTER_", "LANGCHAIN_", "LANGSMITH_", "LANGFUSE_")
    original = {k: v for k, v in os.environ.items() if k.startswith(prefixes)}
    started_in = Path.cwd()
    own_project(workspace, preset)
    try:
        for name in original:
            if not name.endswith("_API_KEY"):
                del os.environ[name]
        os.environ["TESTUARY_CONFIG_FILE"] = str(preset)
        # The workspace is the run's user directory, as in the playback.
        os.environ["TESTUARY_HOME"] = str(workspace)
        os.chdir(workspace)
        yield
    finally:
        os.chdir(started_in)
        for name in list(os.environ):
            if name.startswith(prefixes):
                del os.environ[name]
        os.environ.update(original)


#: The alias fields the playback's catalogue holds. A key never is one.
ALIAS_FIELDS = ("provider", "model", "api_url")


def _alias_catalogue(alias: str, fields: Dict[str, str]) -> str:
    """The recorded alias as the playback's catalogue holds it: the
    provider, the model and the address, as given to the recorder. The
    run's own catalogue, which may hold a key, is never read here."""
    import configparser
    import io

    unknown = set(fields) - set(ALIAS_FIELDS)
    if unknown:
        raise SystemExit(f"not an alias field the demo ships: {', '.join(sorted(unknown))}")
    missing = {"provider", "model"} - set(fields)
    if missing:
        raise SystemExit(f"give the alias's {' and '.join(sorted(missing))} with --alias-field")
    out = configparser.ConfigParser(interpolation=None)
    out.optionxform = str
    out.add_section(alias)
    for key in ALIAS_FIELDS:
        if fields.get(key):
            out.set(alias, key, fields[key])
    buffer = io.StringIO()
    out.write(buffer)
    return buffer.getvalue()


def run_source(run_label: Optional[str]) -> str:
    """The recorded run's label, which every recorded answer names as its
    source and the manifest keeps: the label given, else "recorded run of
    Testuary <version>"."""
    from testuary import __version__

    return run_label or f"recorded run of Testuary {__version__}"


def record(*, language: str, project: Path, target: str, instance: str, alias: str,
           models: Optional[Path], catalogue_dir: Optional[Path],
           alias_fields: Dict[str, str], preset_text: str, options: Sequence[str], outdir: Path,
           data_dir: Path, python: Optional[str], java_home: Optional[str],
           bundle: Sequence[str], licenses: Sequence[Path], notice: str,
           run_label: Optional[str], denylist: Optional[Path] = None,
           adapter: Optional[Dict[str, str]] = None, file_extensions: Sequence[str] = (),
           preset_dir: Optional[Path] = None, dataset: Optional[str] = None,
           replay: Optional[Scenario] = None,
           recording_settings: Sequence[str] = ()) -> int:
    from testuary.demo import cli
    from testuary.demo.cli import RECORDED_LABEL, RECORDED_NOTICE, label_outputs

    if outdir.exists() and any(outdir.iterdir()):
        raise SystemExit(f"the output directory must be new or empty: {outdir}")
    _alias_catalogue(alias, alias_fields)  # refuses incomplete alias fields first
    if cli.phase_modules_loaded():
        raise SystemExit("recording must start before any phase module loads")
    workspace = outdir / "workspace"
    copy = workspace / "projects" / instance
    shutil.copytree(project, copy, ignore=_PROJECT_IGNORE)
    import configparser
    import io

    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(preset_text)
    if not parser.has_section("project"):
        parser.add_section("project")
    # The preset's prompt profile and hints file travel with the data under
    # fixed names. The shipped preset names them so, and the run reads the
    # same text from its workspace, where the playback writes it too.
    carried, carried_text = _carried_settings_files(parser, preset_dir or Path.cwd())
    found: Dict[str, Optional[str]] = {"out": str(outdir), "tmp": tempfile.gettempdir(),
                                       "home": str(Path.home()), "jdk": java_home}
    if language == "python":
        if not python:
            raise SystemExit("a Python recording needs --python, the project's interpreter")
        found.update(_interpreter_paths(python))
        parser.set("project", "runtime_version", _python_version(python))
    shipped = io.StringIO()
    parser.write(shipped)
    shipped_preset = shipped.getvalue()
    # Settings only the recording runs with, such as more mutation workers:
    # the shipped preset is written above without them.
    for item in recording_settings:
        name, equals, value = item.partition("=")
        section, dot, key = name.partition(".")
        if not (section and dot and key and equals):
            raise SystemExit(f"--recording-setting names SECTION.KEY=VALUE, not {item!r}")
        if not parser.has_section(section):
            parser.add_section(section)
        parser.set(section, key, value)
    if language == "python":
        parser.set("project", "python_executable", python)
    elif java_home:
        parser.set("project", "java_home", java_home)
    workspace.mkdir(parents=True, exist_ok=True)
    for setting, name in carried.items():
        (workspace / name).write_text(carried_text[name], encoding="utf-8", newline="\n")
        parser.set("main", setting, str(workspace / name))
    recording = io.StringIO()
    parser.write(recording)
    preset = workspace / "preset.ini"
    preset.write_text(recording.getvalue(), encoding="utf-8")
    # The workspace holds the demo's key-free alias. A run against a fake
    # model may give a key-free catalogue instead. A real catalogue is read in
    # memory from its own directory (``_catalogue``).
    if models is not None:
        shutil.copyfile(models, workspace / "models.ini")
    else:
        (workspace / "models.ini").write_text(_alias_catalogue(alias, alias_fields),
                                              encoding="utf-8")

    run_label = run_source(run_label)
    oracle = RecordingOracle(source=run_label, reuse=replay)
    invocation = ["testuary", "--config", str(preset), "--model", alias,
                  "--project-name", instance, "--source-dir", str(workspace / "projects" / instance),
                  "--source-files", target,
                  "--output-dir", str(outdir / "run"), *options]
    unregister = None
    if language == "java":
        invocation += ["--languages", "java"]
    elif language != "python":
        # A language Testuary has no code for: its adapter is registered for
        # the run, and the generic backend answers through the scenario.
        if adapter is None or not file_extensions:
            raise SystemExit(f"a {language} recording needs --adapter and --file-extension")
        unregister = playback.register(language, file_extensions, adapter)
        invocation += ["--languages", language]
    backend_class = recording_backend(language, file_extensions)
    replayed = None
    if replay is not None:
        live_model, replayed = replay_model(replay)
    else:
        live_model = LiveModelSource()
    previous = install_paths(PathMap(found))
    try:
        with _recording_environment(workspace, preset), \
                _catalogue(catalogue_dir, alias, alias_fields):
            stop = cli.play_oracle(
                oracle, invocation, language=language, backend_class=backend_class,
                model_class=lambda oracle: RecordingModelSource(oracle, live_model),
                cover_class=RecordingCoverSource)
    finally:
        install_paths(previous)
        if unregister is not None:
            unregister()
    if replay is not None:
        print(f"Timings reused from the earlier recording: {oracle.reused}, measured anew: "
              f"{len(oracle.not_reused)} {oracle.not_reused}")
    if stop is not None:
        print(f"Recording stopped: {stop}", file=sys.stderr)
        return cli.EXIT_STALE
    if replayed is not None and replayed.unused():
        print(f"Recording stopped: {replayed.unused()} model answer(s) of the earlier "
              "recording were never asked for", file=sys.stderr)
        return cli.EXIT_STALE
    data_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "project.json": json.dumps({"target": target, "files": {
            rel: (copy / rel).read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
            for rel in sorted(set([target, *bundle]))}}, indent=2, sort_keys=True) + "\n",
        "preset.ini": shipped_preset,
        "models.ini": _alias_catalogue(alias, alias_fields),
        "NOTICE.md": notice,
        **carried_text,
    }
    for license_file in licenses:
        files[license_file.name] = license_file.read_text(encoding="utf-8")
    for name, text in files.items():
        (data_dir / name).write_text(text, encoding="utf-8", newline="\n")
    dump(Scenario(oracle.entries, {"language": language, "statement": STATEMENT,
                                   "source": oracle._source}), data_dir / "scenario.json.gz")
    from testuary.demo.cli import _sha256

    names = [*files, "scenario.json.gz"]
    patterns = denylist_patterns(denylist)
    for name in names:
        found_paths = private_paths(data_dir / name, patterns)
        if found_paths:
            raise SystemExit(f"{name} matches the denylist: {', '.join(found_paths)}")
    workspace_name = target.replace("/", "_").rsplit(".", 1)[0]
    outcome = recorded_outcome(outdir / "run" / "results", workspace_name)
    manifest = {
        "schema_version": 1, "language": language, "statement": STATEMENT,
        "exemplar": {"instance": instance, "target": target, "workspace": workspace_name,
                     "run": run_label},
        "invocation": {"model": alias, "project": instance, "source_files": target,
                       "options": list(options)},
        "recorded": outcome,
        "entries": len(oracle.entries),
        "classes": {"real": len(oracle.entries)},
        "files": {name: _sha256((data_dir / name).read_bytes()) for name in sorted(names)},
    }
    if adapter is not None:
        manifest["adapter"] = dict(adapter)
        manifest["file_extensions"] = list(file_extensions)
    if carried:
        manifest["settings_files"] = dict(carried)
    if dataset and dataset != language:
        manifest["dataset"] = dataset
    if getattr(oracle, "java_facts", None) is not None:
        # Keyed by the project directory as a placeholder already. The values may
        # name machine paths too.
        manifest["java_facts"] = paths_neutral_facts(oracle.java_facts, found)
    if getattr(oracle, "language_facts", None) is not None:
        # The prompts' language facts and test vocabulary, which the playback
        # states instead of asking the project.
        manifest["language_facts"] = paths_neutral_facts(oracle.language_facts, found)
    (data_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n",
                                            encoding="utf-8", newline="\n")
    for recorded_dir in (workspace, outdir / "run"):
        if recorded_dir.is_dir():
            label_outputs(recorded_dir, label=RECORDED_LABEL, notice=RECORDED_NOTICE,
                          inputs=[workspace / name for name in carried.values()])
    print(f"Recorded {len(oracle.entries)} answers into {data_dir}.")
    return 0


#: The file name each carried setting's file has in the data.
CARRIED_NAMES = {"prompt_profile": "prompt_profile.json", "hints_file": "hints.json"}


def _carried_settings_files(parser: Any, base: Path) -> Tuple[Dict[str, str], Dict[str, str]]:
    """The settings files ``[main]`` of the preset names, by setting, with
    their text under their name in the data, and the preset changed to name
    them so. A relative path starts at ``base``, the preset's directory."""
    carried: Dict[str, str] = {}
    texts: Dict[str, str] = {}
    for setting, name in CARRIED_NAMES.items():
        value = parser.get("main", setting, fallback="").strip()
        if not value:
            continue
        source = Path(value) if Path(value).is_absolute() else base / value
        if not source.is_file():
            raise SystemExit(f"[main] {setting} names no file: {source}")
        texts[name] = source.read_bytes().replace(b"\r\n", b"\n").decode("utf-8")
        carried[setting] = name
        parser.set("main", setting, name)
    return carried, texts


def paths_neutral_facts(facts: Dict[str, Any], found: Dict[str, Optional[str]]) -> Dict[str, Any]:
    """The recorded language facts with machine paths as placeholders."""
    with_map = PathMap(found)
    return json.loads(with_map.neutral(json.dumps(facts)))


def denylist_patterns(path: Optional[Path]) -> List[re.Pattern]:
    """The patterns of a denylist file: one regular expression per line,
    matched without regard to case, comment lines starting with ``#``. The
    recorded data must contain none of them."""

    if path is None:
        return []
    patterns = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            patterns.append(re.compile(line, re.IGNORECASE))
    return patterns


def private_paths(path: Path, patterns: Sequence[re.Pattern]) -> List[str]:
    """The denylist patterns a data file matches, the compressed scenario
    read decompressed."""
    import gzip

    data = path.read_bytes()
    if path.suffix == ".gz":
        data = gzip.decompress(data)
    text = data.decode("utf-8", errors="replace")
    return sorted({pattern.pattern for pattern in patterns if pattern.search(text)})


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="python tools/record_demo.py", description=__doc__)
    parser.add_argument("--language", required=True,
                        help="python, java, or a language registered from outside, which "
                             "needs --adapter and --file-extension")
    parser.add_argument("--adapter", metavar="PACKAGE:PATH:CLASS",
                        help="the adapter of a language registered from outside: a file "
                             "inside an installed package and its class, or MODULE:CLASS")
    parser.add_argument("--file-extension", action="append", default=[],
                        help="a source file extension of that language, such as .toy")
    parser.add_argument("--project", required=True, type=Path,
                        help="the project directory; the recording copies it")
    parser.add_argument("--target", required=True, help="the source file, relative to --project")
    parser.add_argument("--instance", required=True, help="the project's name in the demo")
    parser.add_argument("--model", required=True, help="the models.ini alias the run uses")
    parser.add_argument("--replay-model", type=Path, metavar="SCENARIO",
                        help="an earlier recording's scenario.json.gz: its model answers "
                             "answer the run's model requests, matched by key, and no model "
                             "is contacted; a request it does not hold stops the recording")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--catalogue-dir", type=Path,
                        help="the directory whose models.ini (which may hold a key) Testuary's "
                             "settings reader reads in memory; nothing copies or prints it")
    source.add_argument("--models", type=Path,
                        help="a key-free models.ini for a run against a fake model")
    parser.add_argument("--alias-field", action="append", default=[], metavar="KEY=VALUE",
                        help="a field of the alias the demo ships (provider, model, api_url)")
    parser.add_argument("--preset", type=Path,
                        help="settings for the run; default: none. A prompt profile or hints "
                             "file it names, relative to its directory, travels with the data")
    parser.add_argument("--dataset", metavar="NAME",
                        help="the name testuary --demo --demo-data selects the data by, for a "
                             "variant such as tutorial; default: the language")
    parser.add_argument("--recording-setting", action="append", default=[],
                        metavar="SECTION.KEY=VALUE",
                        help="a setting only the recording runs with, not shipped in the "
                             "data's preset, such as mutation.engine_workers=6; it must "
                             "not enter what the playback matches on")
    parser.add_argument("--option", action="append", default=[],
                        help="a further option of the run, such as --display=plain")
    parser.add_argument("--python", help="the project's interpreter (Python)")
    parser.add_argument("--java-home", help="the JDK (Java)")
    parser.add_argument("--bundle", action="append", default=[],
                        help="a further project file the demo ships, relative to --project")
    parser.add_argument("--license", action="append", default=[], type=Path,
                        help="a license file the demo ships beside the project")
    parser.add_argument("--notice", required=True, type=Path, help="the data notice to ship")
    parser.add_argument("--run-label", default=None,
                        help="the recorded run's whole label, such as \"recorded run of "
                             "Testuary 0.2.0 on calc.toy, 2026-10-20\" (default: \"recorded "
                             "run of Testuary <version>\")")
    parser.add_argument("--denylist", type=Path,
                        help="a file of patterns the recorded data must not contain, "
                             "one regular expression per line")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--data-dir", required=True, type=Path)
    args = parser.parse_args(argv)
    preset_text = args.preset.read_text(encoding="utf-8") if args.preset else ""
    return record(
        language=args.language, project=args.project.resolve(), target=args.target,
        instance=args.instance, alias=args.model,
        models=args.models.resolve() if args.models else None,
        catalogue_dir=args.catalogue_dir.resolve() if args.catalogue_dir else None,
        alias_fields=dict(item.split("=", 1) for item in args.alias_field),
        preset_text=preset_text, options=args.option, outdir=args.output_dir.resolve(),
        data_dir=args.data_dir.resolve(), python=args.python, java_home=args.java_home,
        bundle=args.bundle, licenses=[path.resolve() for path in args.license],
        notice=args.notice.read_text(encoding="utf-8"), run_label=args.run_label,
        denylist=args.denylist.resolve() if args.denylist else None,
        adapter=_adapter_spec(args.adapter) if args.adapter else None,
        file_extensions=args.file_extension,
        preset_dir=args.preset.resolve().parent if args.preset else None,
        dataset=args.dataset,
        replay=Scenario.load(args.replay_model) if args.replay_model else None,
        recording_settings=args.recording_setting)


def _adapter_spec(text: str) -> Dict[str, str]:
    """``PACKAGE:PATH:CLASS`` or ``MODULE:CLASS`` as a manifest's adapter."""
    parts = text.split(":")
    if len(parts) == 3:
        return {"package": parts[0], "path": parts[1], "class": parts[2]}
    if len(parts) == 2:
        return {"module": parts[0], "class": parts[1]}
    raise SystemExit(f"--adapter names PACKAGE:PATH:CLASS or MODULE:CLASS, not {text!r}")


if __name__ == "__main__":
    raise SystemExit(main())
