"""``testuary --demo``: an offline demonstration through the real pipeline.

The package's real pipeline runs on the selected example with model
responses and measurement results from the shipped scenario. It runs with
the scenario's configuration only. Display choices still apply. Every other
option, every ``TESTUARY_*`` or ``LANGADAPTER_*`` variable and every
configuration file a normal run would read is set aside and named, with the
value used instead. Nothing is sent over the network and no test, coverage
or mutation tool runs. The set-cover solutions come from the scenario too,
so the pipeline starts no process.

Exit codes: 0 the demonstration ran and consumed the scenario exactly, 1
(the demonstrated run's own status) a target of the demonstrated run failed
to execute, or the guard refused a process or a connection a library
started, 2 a usage error (the run directory is not empty), 3 the
scenario or the shipped data do not fit this package version, 130 the
demonstration was stopped with Ctrl+C. New data come from a separate
recording, never from a retry or a live call.

Every output of a demonstration is marked as one (``label_outputs``): JSON
records and JSON-lines logs gain a field, the results table a column. Files
whose bytes are compared or recorded by hash (suites, replies, the
instrument's configuration) stay as written, and a notice file in their
directory names them. The counts go into the root manifest.
"""
from __future__ import annotations

import argparse
import configparser
import contextlib
import csv
import datetime
import hashlib
import io
import ipaddress
import json
import math
import os
import shlex
import sys
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple

if TYPE_CHECKING:
    from testuary.demo.scenario import Entry, Scenario, ScenarioStop

EXIT_OK = 0
EXIT_FAILED = 1
EXIT_USAGE = 2
EXIT_STALE = 3

# Variables that would send traces or requests elsewhere. They are cleared with the
# package's own variables before the run.
_TRACING_PREFIXES = ("LANGCHAIN_", "LANGSMITH_", "LANGFUSE_")


class DataMismatch(Exception):
    """The shipped data do not match their manifest."""


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data.replace(b"\r\n", b"\n")).hexdigest()


#: The settings that name a file, which demonstration data may carry: the
#: playback writes the file into its workspace and points the setting at it.
SETTINGS_FILES = ("prompt_profile", "hints_file")


def _dataset(manifest: Dict[str, Any]) -> str:
    """The name a manifest's data are selected by: its language, or the
    name of a variant such as tutorial."""
    return manifest.get("dataset", manifest.get("language", "python"))


def datasets() -> List[str]:
    """The demonstration data that come with Testuary: Python's, whose data are the directory
    itself, and each directory that holds a manifest, named as its data. A
    directory named as its language, such as java, is that language's
    demonstration. Any other, such as tutorial, is a variant that
    ``--demo-data`` selects."""
    root = resources.files("testuary.demo").joinpath("data")
    return ["python"] + sorted(entry.name for entry in root.iterdir()
                               if entry.is_dir() and entry.joinpath("manifest.json").is_file())


def languages() -> List[str]:
    """The languages whose demonstration data come with Testuary: Python and each
    data directory named as its language, such as java."""
    root = resources.files("testuary.demo").joinpath("data")
    found = ["python"]
    for name in datasets()[1:]:
        manifest = json.loads(root.joinpath(name, "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("language") == name:
            found.append(name)
    return found


def load_data(language: str = "python") -> Tuple[Dict[str, Any], Dict[str, bytes]]:
    """The manifest and every data file it names, each checked against the
    manifest's hash. ``language`` names the data: a language, or a variant
    such as tutorial."""
    root = resources.files("testuary.demo").joinpath("data")
    if language != "python":
        if language not in datasets():
            raise ValueError(f"unsupported demo language: {language!r}")
        root = root.joinpath(language)
    manifest = json.loads(root.joinpath("manifest.json").read_text(
        encoding="utf-8"))
    if _dataset(manifest) != language:
        raise DataMismatch("the manifest names a different demo language")
    check_recorded(manifest)
    files: Dict[str, bytes] = {}
    for name, digest in sorted(manifest["files"].items()):
        raw = root.joinpath(name).read_bytes()
        if _sha256(raw) != digest:
            raise DataMismatch(f"{name} does not match the manifest")
        files[name] = raw
    return manifest, files


def check_recorded(manifest: Dict[str, Any]) -> None:
    """Refuse data whose answers were not all recorded in a real run of this
    release: the only data a demonstration plays."""
    if set(manifest.get("classes") or {}) != {"real"}:
        raise DataMismatch("the data hold answers that were not recorded in a real "
                           "run of this release")


def settings_text(text: str) -> configparser.ConfigParser:
    """Settings text as Testuary's own settings reader reads it: no
    interpolation, keys case-folded."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(text)
    return parser


def _preset_parser(text: str) -> configparser.ConfigParser:
    """The preset to write back: as ``settings_text``, keys as written, so
    the bytes it writes keep the recorded spelling."""
    parser = configparser.ConfigParser(interpolation=None)
    parser.optionxform = str
    parser.read_string(text)
    return parser


def _preset_bytes(parser: configparser.ConfigParser) -> bytes:

    result = io.StringIO()
    parser.write(result)
    return result.getvalue().encode("utf-8")


def _preset_values(preset_text: str) -> Dict[str, Dict[str, str]]:
    parser = settings_text(preset_text)
    return {section: dict(parser[section]) for section in parser.sections()}


def _runtime_preset(raw: bytes, *, python: bool = False) -> bytes:
    """The recorded preset as the workspace runs it.

    The Python playback runs no project code, but a run still selects the
    project's interpreter before any work. The demonstration selects the one
    it runs in, explicitly, since a run never falls back to its own
    installation by itself. Otherwise the preset is the recorded one.
    """
    parser = _preset_parser(raw.decode("utf-8"))
    if not python or parser.get("project", "python_executable", fallback="").strip():
        return raw
    if not parser.has_section("project"):
        parser.add_section("project")
    parser.set("project", "python_executable", sys.executable)
    return _preset_bytes(parser)


def _check_options(parser: argparse.ArgumentParser, options: List[str]) -> None:
    """Refuse recorded command-line options that this version's command does
    not take: the data no longer fit it. Raises DataMismatch with the
    command's refusal."""
    refusal = io.StringIO()
    try:
        with contextlib.redirect_stderr(refusal):
            parser.parse_args(list(options))
    except SystemExit:
        lines = refusal.getvalue().strip().splitlines() or [""]
        raise DataMismatch(f"this version refuses the recorded options "
                           f"{' '.join(options)}: {lines[-1]}") from None


def configuration(manifest: Dict[str, Any], preset_text: str,
                  models_text: str) -> List[Tuple[str, str]]:
    """The recorded configuration the demonstration runs with, as the rows
    it prints and writes."""
    from testuary import settings

    preset = _preset_values(preset_text)
    models = _preset_values(models_text)
    defaults = settings._defaults()
    alias = manifest["invocation"]["model"]

    def value(section: str, key: str) -> Tuple[str, bool]:
        """The value the run used, and whether it is the package's default."""
        given = preset.get(section, {}).get(key)
        if given is not None:
            return given.strip(), False
        return str(defaults.get(section, {}).get(key, "?")).strip(), True

    def get(section: str, key: str) -> str:
        return value(section, key)[0]

    def shown(text: str, *keys: Tuple[str, str]) -> str:
        """``text``, marked as the default when every value in it is one."""
        return text + (" (default)" if all(value(*key)[1] for key in keys) else "")

    def yes(text: str) -> bool:
        return configparser.ConfigParser.BOOLEAN_STATES.get(text.lower(), False)

    entry = models.get(alias, {})
    service = entry.get("api_url") or (f"provider {entry['provider']}"
                                       if entry.get("provider") else "?")
    metric = {"both": "line and branch coverage", "either": "line or branch coverage",
              "line": "line coverage", "branch": "branch coverage"}
    coverage_goal = ("coverage", "goal"), ("coverage", "goal_metric")
    starter_tries = ("starter", "refill_failed_suites"), ("starter", "attempts_per_suite")
    timeout = get("mutation", "per_mutant_timeout_seconds")
    reduction = (("starter", "reduction_solver"), ("starter", "reduction_solver_time_limit_seconds"),
                 ("starter", "reduction_solver_random_seed"))
    runtime = get("project", "runtime_version")
    rows = [
        ("model", f"{alias} ({entry.get('model', '?')}, {service}, not contacted)"),
        ("target", f"{manifest['invocation']['source_files']} "
                 f"(project {manifest['exemplar']['instance']})"),
        ("runtime in the prompts", runtime or "the one the run finds"),
        ("starter suites", ", ".join((
            shown(get("starter", "suite_count"), ("starter", "suite_count")),
            shown(f"a failed one is generated again (up to {get('starter', 'attempts_per_suite')} "
                  "attempts per suite)" if yes(get("starter", "refill_failed_suites"))
                  else "a failed one is not generated again", *starter_tries),
            shown(f"temperature {get('starter', 'temperature')}", ("starter", "temperature"))))),
        ("coverage phase", ", ".join((
            shown(f"goal: {get('coverage', 'goal')}% "
                  f"{metric.get(get('coverage', 'goal_metric'), get('coverage', 'goal_metric'))}",
                  *coverage_goal),
            shown(f"request limit {get('coverage', 'max_requests_per_phase')}",
                  ("coverage", "max_requests_per_phase")),
            shown(f"stops after {get('coverage', 'max_measurements_without_improvement')} measurements without "
                  "improvement", ("coverage", "max_measurements_without_improvement"))))),
        ("mutation phase", ", ".join((
            shown(f"goal: {get('mutation', 'goal')}% mutation score", ("mutation", "goal")),
            shown(f"request limit {get('mutation', 'max_requests_per_phase')}",
                  ("mutation", "max_requests_per_phase")),
            shown("full catalogue" if get("mutation", "loop_catalogue") == "full"
                  else "selective catalogue", ("mutation", "loop_catalogue")),
            shown("time limit per mutant: from the suite's run time" if timeout == "auto"
                  else f"time limit per mutant: {timeout} s", ("mutation", "per_mutant_timeout_seconds")),
            shown(f"mutation workers: {get('mutation', 'engine_workers')}",
                  ("mutation", "engine_workers"))))),
        ("suite reduction", shown(
            "keeps the fewest, fastest tests with the same coverage and the same mutants killed "
            f"({get('starter', 'reduction_solver')} solver, at most "
            f"{get('starter', 'reduction_solver_time_limit_seconds')} s, solver seed "
            f"{get('starter', 'reduction_solver_random_seed')})", *reduction)),
        ("mutation score", shown(
            "also measured after the starter suites and the coverage phase: "
            + ("yes" if yes(get("main", "phase_end_mutation")) else "no"),
            ("main", "phase_end_mutation"))),
    ]
    if (manifest.get("language", "python") == "python"
            and not preset.get("project", {}).get("python_executable", "").strip()):
        rows.append(("interpreter", "the one this demonstration runs in, selected "
                     "explicitly (no project code runs)"))
    labels = {"prompt_profile": "prompt profile", "hints_file": "hints file"}
    for setting, name in sorted((manifest.get("settings_files") or {}).items()):
        rows.append((labels.get(setting, setting), f"{name} (carried with the data)"))
    return rows


def overridden_settings(
    args: argparse.Namespace, parser: argparse.ArgumentParser,
    environ: Dict[str, str], cwd: Path, alias: str,
) -> List[Dict[str, str]]:
    """Every setting a normal run would take from the command line, the
    environment or the working directory, which the demonstration does not
    use, with what it uses instead."""
    items: List[Dict[str, str]] = []
    instead = {
        "model": f"the recorded model alias {alias}",
        "config": "the recorded configuration shipped with the data, and only [demo] "
                  "playback_seconds is taken from these files",
    }
    from testuary.__main__ import DEMO_OPTIONS

    # Each setting as the user wrote it: an option by its first spelling, a
    # file named on the command line by its metavar (FILE), a flag alone and
    # a list as its items. A field the command line sets itself, such as a
    # phase's reuse flag, is not a setting.
    typed = {action.dest: action.option_strings[0] if action.option_strings else action.metavar
             for action in parser._actions}
    for dest, value in sorted(vars(args).items()):
        if dest not in typed or dest in DEMO_OPTIONS | {"output_dir", "display", "no_color",
                                                        "events"}:
            continue
        if value == parser.get_default(dest):
            continue
        shown = ("" if value is True else " ".join(map(str, value))
                 if isinstance(value, (list, tuple)) else str(value))
        items.append({
            "setting": f"{typed.get(dest) or '--' + dest.replace('_', '-')} {shown}".rstrip(),
            "instead": instead.get(dest, "the recorded configuration"),
        })
    explicit = bool(getattr(args, "config", None))
    named_file = not explicit and bool(environ.get("TESTUARY_CONFIG_FILE", "").strip())
    items += _environment_settings(environ, named_file,
                                   getattr(args, "demo_duration", None) is not None)
    for path, only_length in _normal_run_files(cwd, environ, explicit or named_file, args):
        if path.is_file():
            items.append({
                "setting": f"{path}",
                "instead": ("only [demo] playback_seconds is read, as the demonstration "
                            "runs in its own workspace" if only_length else
                            "not read, as the demonstration runs in its own workspace"),
            })
    return items


def _environment_settings(environ: Dict[str, str], named_file: bool,
                          length_option: bool) -> List[Dict[str, str]]:
    """The environment variables a normal run would read, with what the
    demonstration does with each instead. ``named_file`` is whether
    TESTUARY_CONFIG_FILE names the settings file, ``length_option`` whether
    --demo-playback-seconds sets the playback's length."""
    items: List[Dict[str, str]] = []
    for name in sorted(environ):
        if name == PLAYBACK_VARIABLE:
            items.append({"setting": f"environment variable {name}", "instead": (
                "not read, as --demo-playback-seconds sets the playback's length"
                if length_option else "not read, as it is empty"
                if not environ[name].strip() else "read: the playback's length")})
            continue
        if name == "TESTUARY_CONFIG_FILE" and named_file:
            items.append({"setting": f"environment variable {name}",
                          "instead": "only [demo] playback_seconds is read from the file it names"})
            continue
        if name.startswith(("TESTUARY_", "LANGADAPTER_", *_TRACING_PREFIXES)):
            items.append({
                "setting": f"environment variable {name}",
                "instead": "not read, cleared for the demonstration",
            })
    return items


def _normal_run_files(cwd: Path, environ: Dict[str, str], named: bool,
                      args: argparse.Namespace) -> List[Tuple[Path, bool]]:
    """The files a normal run in ``cwd`` with ``args`` reads, each with
    whether the demonstration reads its [demo] playback_seconds: the user
    directory's config.ini (unless a file is named in its place) and models.ini,
    from --home when given, and the project's settings file unless
    --no-project-settings."""
    from testuary.settings import project_locations

    try:
        locations = project_locations(cwd, home=getattr(args, "home", None),
                                      environment=environ)
    except (OSError, ValueError):
        return []
    files = [(locations.config, not named), (locations.models, False)]
    if not getattr(args, "no_project_settings", False):
        files.append((locations.project_settings, True))
    return files


class Audit:
    """The guard: records the processes and network connections the run
    starts, and refuses every process and every connection to a
    non-loopback address by stopping the run."""

    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []
        self.active = False

    def _refuse(self, event: Dict[str, Any], reason: str) -> None:
        from testuary.demo.scenario import Call, ScenarioStop

        event["refused"] = True
        self.active = False
        raise ScenarioStop(Call(f"guard.{event['event']}", event), reason)

    def hook(self, event: str, args: Tuple[Any, ...]) -> None:
        if not self.active:
            return
        if event == "subprocess.Popen":
            # Recorded for the stop message. The event names the executable
            # only when the caller passed one. Otherwise the program is the
            # first argument, and on Windows the whole command arrives as one
            # string.
            raw = args[1]
            if isinstance(raw, (list, tuple)):
                argv = [str(a) for a in raw]
            elif raw is None:
                argv = []
            else:
                try:
                    tokens = shlex.split(str(raw), posix=os.name != "nt")
                except ValueError:  # quoting shlex cannot read
                    tokens = str(raw).split()
                argv = [token.strip('"') for token in tokens]
            program = str(args[0]) if args[0] else (argv[0] if argv else "")
            record = {"event": event, "executable": program, "argv": argv[:4]}
            self.events.append(record)
            self._refuse(record, "the demonstration starts no process")
        elif event == "socket.connect":
            address = args[1]
            host = str(address[0] if isinstance(address, tuple) else address)
            try:
                loopback = host.lower() == "localhost" or ipaddress.ip_address(host).is_loopback
            except ValueError:
                loopback = False
            if loopback and _socketpair_connect():
                # Windows emulates socket.socketpair with a loopback
                # connection between two sockets of this process, as the
                # agents' cancel watcher uses. It reaches no network.
                return
            record = {"event": event, "address": host}
            self.events.append(record)
            if not loopback:
                self._refuse(record, "the demonstration contacts no network "
                             "address")
        elif event in ("os.system", "os.exec", "os.posix_spawn", "os.spawn"):
            # The audit arguments of os.exec, os.posix_spawn and os.spawn
            # end with the environment, which the record leaves out.
            # os.system's one argument is its command.
            record = {"event": event, "detail": repr(args[:max(1, len(args) - 1)])[:200]}
            self.events.append(record)
            self._refuse(record, "the demonstration starts no process")


def _socketpair_connect() -> bool:
    """Whether the connection being audited is socket.socketpair's own,
    which Windows makes over loopback inside this process: a frame of that
    function in the standard library's socket module, not a look-alike."""
    import socket

    standard = os.path.normcase(os.path.abspath(socket.__file__))
    frame = sys._getframe(2)
    for _ in range(8):
        if frame is None:
            return False
        code = frame.f_code
        if (code.co_name in ("socketpair", "_fallback_socketpair")
                and os.path.normcase(os.path.abspath(code.co_filename)) == standard):
            return True
        frame = frame.f_back
    return False


def _default_outdir(manifest: Dict[str, Any]) -> Path:
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y%m%dT%H%M%SZ")
    return Path.cwd() / f"testuary-demo-{manifest['exemplar']['workspace']}-{stamp}"


def _write_workspace(workspace: Path, manifest: Dict[str, Any],
                     files: Dict[str, bytes]) -> Path:
    project = workspace / "projects" / manifest["exemplar"]["instance"]
    layout = json.loads(files["project.json"].decode("utf-8"))
    for rel, text in sorted(layout["files"].items()):
        path = project / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8", newline="\n")
    preset = workspace / "preset.ini"
    python = manifest.get("language", "python") == "python"
    runtime = _runtime_preset(files["preset.ini"], python=python)
    preset.write_bytes(_with_settings_files(runtime, workspace, manifest, files))
    (workspace / "models.ini").write_bytes(files["models.ini"])
    return preset


def _with_settings_files(raw: bytes, workspace: Path, manifest: Dict[str, Any],
                         files: Dict[str, bytes]) -> bytes:
    """The preset with each settings file the data carry written into the
    workspace and named by its full path there, as the recording named the
    same file in its workspace, so the run reads the recorded profile and
    hints wherever the playback runs."""
    carried = manifest.get("settings_files") or {}
    if not carried:
        return raw
    parser = _preset_parser(raw.decode("utf-8"))
    if not parser.has_section("main"):
        parser.add_section("main")
    for setting, name in sorted(carried.items()):
        if setting not in SETTINGS_FILES:
            raise DataMismatch(f"the data carry a file for an unknown setting: {setting}")
        path = workspace / name
        path.write_bytes(files[name])
        parser.set("main", setting, str(path))
    return _preset_bytes(parser)


def own_project(workspace: Path, preset: Path) -> None:
    """Make the workspace its own project: an empty Testuary directory, named as
    the preset names it, ends the search for a project from the workspace
    upward there. A demonstration started inside a project so never reads
    that project's settings file, which its setup report sets aside."""
    from testuary import settings

    main = settings._read_ini_file(preset).get("main", {})
    name = str(main.get("testuary_dir", "")).strip() or ".testuary"
    (workspace / name).mkdir(parents=True, exist_ok=True)


@contextlib.contextmanager
def _environment_aside() -> Iterator[None]:
    """Set the environment's TESTUARY_, LANGADAPTER_ and tracing variables
    aside for the block, and restore them after it, removing any the block
    set."""
    prefixes = ("TESTUARY_", "LANGADAPTER_", *_TRACING_PREFIXES)
    original = {k: v for k, v in os.environ.items() if k.startswith(prefixes)}
    try:
        for name in original:
            del os.environ[name]
        yield
    finally:
        for name in list(os.environ):
            if name.startswith(prefixes):
                del os.environ[name]
        os.environ.update(original)


@contextlib.contextmanager
def _workspace_environment(workspace: Path, preset: Path) -> Iterator[None]:
    """Keep demo-only configuration and its working directory local to the
    run: the preset, the workspace as the user directory and as the project."""
    started_in = Path.cwd()
    own_project(workspace, preset)
    with _environment_aside():
        try:
            os.environ["TESTUARY_CONFIG_FILE"] = str(preset)
            # The workspace is the run's user directory: its models.ini is the
            # recorded one, and the user's own files are never read.
            os.environ["TESTUARY_HOME"] = str(workspace)
            os.chdir(workspace)
            yield
        finally:
            os.chdir(started_in)


@contextlib.contextmanager
def _playback_paths(outdir: Path) -> Iterator[None]:
    """The directories of this playback in place of a recording's placeholders:
    the run directory, the interpreter the demonstration runs in, its
    site-packages directory and environment, a JDK directory below the run
    directory (no JDK runs), the temporary directory and the home directory."""
    import sysconfig
    import tempfile

    from testuary.demo.scenario import PathMap, install_paths

    previous = install_paths(PathMap({
        "out": str(outdir), "python": sys.executable,
        "site": sysconfig.get_paths()["purelib"], "env": sys.prefix,
        "jdk": str(outdir / "jdk"),
        "tmp": tempfile.gettempdir(), "home": str(Path.home()),
    }))
    try:
        yield
    finally:
        install_paths(previous)


@contextlib.contextmanager
def _recorded_facts(manifest: Dict[str, Any]) -> Iterator[None]:
    """The language facts and test vocabulary the scenario's recording
    stated, for the duration of the playback."""
    from testuary.demo import backend

    previous = backend.RECORDED_LANGUAGE
    backend.RECORDED_LANGUAGE = manifest.get("language_facts")
    try:
        yield
    finally:
        backend.RECORDED_LANGUAGE = previous


def _label_outputs(outdir: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:

    # The settings files the data carry are inputs of the run, kept as read.
    inputs = [outdir / "workspace" / name
              for name in ((manifest or {}).get("settings_files") or {}).values()]
    return {name: label_outputs(outdir / name, inputs=inputs)
            for name in ("run", "workspace") if (outdir / name).is_dir()}


@contextlib.contextmanager
def _adapter(manifest: Dict[str, Any]):
    """Register the adapter of a language registered from outside, as its
    manifest names it, for the playback."""
    spec = manifest.get("adapter")
    if spec is None:
        yield
        return
    from testuary.demo import backend as generic

    undo = generic.register(manifest["language"], manifest["file_extensions"], spec)
    try:
        yield
    finally:
        undo()


def _answer_classes(classes: Dict[str, int]) -> str:
    """The answer counts in words: every answer is recorded."""
    return f"{classes.get('real', 0)} recorded"


# Playing a scenario through the package's pipeline. The model source, the
# cover source and the backend are installed for the duration of one run and
# removed afterwards. The run itself is the package's own command-line entry
# point with the exemplar's invocation. A call the scenario does not hold
# stops the run with a ScenarioStop that names it, which play returns
# instead of raising. A run that fails to execute a target raises
# PipelineFailure with the run's exit status.


class PipelineFailure(Exception):
    """The demonstrated run failed to execute a target: its exit status was
    not 0."""

    def __init__(self, status: int) -> None:
        super().__init__(f"the pipeline exited with status {status}")
        self.status = status


def phase_modules_loaded() -> List[str]:
    """Phase modules already imported: they snapshot their configuration at
    import, so a demonstration must start before any of them loads."""
    return sorted(name for name in sys.modules
                  if name in ("testuary.agent.coverage_agent",
                              "testuary.agent.mutation_agent"))


def play(
    scenario: "Scenario",
    invocation: List[str],
    *,
    on_answer: Optional[Callable[["Entry"], None]] = None,
    language: str = "python",
) -> Optional["ScenarioStop"]:
    """Run the pipeline with ``invocation`` as its command line, answering
    every model request and measurement from ``scenario``. Returns None when
    the run consumed the scenario exactly, else the stop. Raises
    ``PipelineFailure`` when the run failed to execute a target, whatever
    it left of the scenario."""
    from testuary.demo.scenario import PlaybackOracle

    oracle = PlaybackOracle(scenario, on_answer=on_answer)
    return play_oracle(oracle, invocation, language=language)


def play_oracle(oracle, invocation: List[str], *, language: str = "python",
                backend_class=None, model_class=None,
                cover_class=None) -> Optional["ScenarioStop"]:
    """The pipeline driver of a playback and of a recording. It does not
    replace the phase orchestration: ``oracle`` answers every executing
    operation. ``backend_class``, ``model_class`` and ``cover_class`` make
    the backend, the model source and the cover source from the oracle,
    the scenario's when not given.
    """
    from testuary import __main__ as entry_point
    from testuary import languages
    from testuary import models as model_source
    from testuary import reduction as cover_source
    from testuary.demo import backend as playback
    from testuary.demo.scenario import Call, ScenarioStop
    from testuary.settings import ConfigurationError

    if backend_class is None:
        if language == "java":
            backend_class = playback.ScenarioJavaBackend
        elif language == "python":
            backend_class = playback.ScenarioBackend
        else:
            from langadapter import registration

            backend_class = playback.backend_class(language,
                                                   registration(language).file_extensions)
    backend = backend_class(oracle)
    saved_argv = list(sys.argv)
    model = (model_class or playback.ScenarioModelSource)(oracle)
    cover = (cover_class or playback.ScenarioCoverSource)(oracle)
    with model_source.installed(model), cover_source.installed(cover):
        previous = languages.override_backend(language, backend)
        try:
            sys.argv = list(invocation)
            try:
                # run_command, not main: an interrupt must reach the demo as a
                # cancellation, not come back as the console's exit status 130.
                status = entry_point.run_command()
                if status:
                    raise PipelineFailure(status)
                oracle.finish()
            except ScenarioStop as stop:
                return stop
            except SystemExit as exc:
                if not isinstance(exc.__cause__, ConfigurationError):
                    raise
                # The recorded settings, aliases or options, which this
                # version refuses: the data no longer fit it.
                return ScenarioStop(
                    Call("configuration", {"invocation": list(invocation)}),
                    f"this version refuses the recorded configuration: {exc.__cause__}")
            return None
        finally:
            sys.argv = saved_argv
            languages.override_backend(language, previous)


def main(args: argparse.Namespace, parser: argparse.ArgumentParser) -> int:
    """Play the demonstration the parsed ``args`` name and return its exit
    status. ``parser`` checks the options the recorded run used."""
    from testuary.demo.scenario import Scenario, ScenarioStop

    loaded = phase_modules_loaded()
    if loaded:
        print(f"ERROR: the demonstration must start before any phase module "
              f"loads. Loaded: {', '.join(loaded)}", file=sys.stderr)
        return EXIT_USAGE
    language = getattr(args, "demo_language", "python")
    name = getattr(args, "demo_data", None) or language
    if getattr(args, "demo_data", None):
        if name not in datasets():
            print(f"ERROR: there are no demonstration data {name!r}. Testuary has: "
                  f"{', '.join(datasets())}", file=sys.stderr)
            return EXIT_USAGE
    elif language not in languages():
        print(f"ERROR: there is no demonstration of {language!r}. Testuary has: "
              f"{', '.join(languages())}", file=sys.stderr)
        return EXIT_USAGE
    try:
        playback = _playback(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return EXIT_USAGE
    try:
        manifest, files = load_data(name)
        # A variant's data are in the language their manifest names.
        language = manifest.get("language", "python")
        _runtime_preset(files["preset.ini"])
        _check_options(parser, manifest["invocation"].get("options", []))
    except (DataMismatch, OSError, KeyError, ValueError, configparser.Error) as exc:
        print(f"ERROR: the demonstration data cannot be used: {exc}",
              file=sys.stderr)
        return EXIT_STALE

    outdir = (Path(args.output_dir).resolve() if args.output_dir
              else _default_outdir(manifest))
    if not _unused_run_dir(outdir, getattr(args, "events", None)):
        print(f"ERROR: the run directory {outdir} is not empty. The "
              f"demonstration writes only into a new or empty directory.",
              file=sys.stderr)
        return EXIT_USAGE

    alias = manifest["invocation"]["model"]
    overridden = overridden_settings(args, parser, dict(os.environ),
                                     Path.cwd(), alias)
    config_rows = configuration(manifest, files["preset.ini"].decode("utf-8"),
                                files["models.ini"].decode("utf-8"))
    bar = "=" * 76
    print(bar)
    statement = manifest["statement"]
    print(statement)
    print(bar)
    print("Configuration (the recorded run's):")
    for name, value in config_rows:
        print(f"  {name:22s} {value}")
    if overridden:
        print("Ignored by the demonstration:")
        for item in overridden:
            print(f"  {item['setting']}: {item['instead']}")
    print(f"Run directory: {outdir}")
    if playback is not None:
        factor, step, seconds = playback
        print("Playback: one step per Enter press. Ctrl+C stops." if step else
              f"Playback: paced, at least about {round(seconds / factor)} seconds. "
              f"Values, records and counts are those of instant playback. Ctrl+C stops.")
    print(bar, flush=True)

    outdir.mkdir(parents=True, exist_ok=True)
    records = outdir / "demo"
    records.mkdir(exist_ok=True)
    workspace = outdir / "workspace"
    preset = _write_workspace(workspace, manifest, files)
    try:
        scenario = Scenario.load(_scenario_path(files, records))
    except (OSError, ValueError, KeyError) as exc:
        result = {"outcome": "stale", "playback": _playback_text(playback),
                  "answers": 0, "classes": {},
                  "processes": [], "network_connections": 0,
                  "labelled": _label_outputs(outdir, manifest),
                  "stale": {"reason": str(exc)}}
        _write_manifest(outdir, manifest, config_rows, overridden, result)
        print(f"ERROR: the demonstration scenario cannot be read: {exc}",
              file=sys.stderr)
        return EXIT_STALE
    ledger: List[Dict[str, Any]] = []
    display = getattr(args, "display", None)
    # The answer ledger is a detail for the detailed views. The records keep it.
    verbose_ledger = display in ("detailed", "debug")

    def on_answer(entry: Any) -> None:
        rule_id = entry.rule.split(":", 1)[0]
        ledger.append({"segment": entry.segment, "kind": entry.kind,
                       "class": entry.value_class,
                       "parts": {k: v["class"] for k, v in entry.parts.items()},
                       "rule": rule_id, "source": entry.source})
        if verbose_ledger:
            print(f"[demo] {entry.segment:8s} {'recorded':20s} "
                  f"{entry.kind} ({rule_id})", flush=True)
        if pacer is not None:
            pacer.on_answer(entry)

    # The chat model clients' libraries load before the guard: the guard
    # watches the playback, and a library may start a process when it is
    # imported (aiohttp's platform.system() runs "cmd /c ver" on Windows
    # before Python 3.12). The phase modules still load inside the run.
    import testuary._chat_models  # noqa: F401

    audit = Audit()
    sys.addaudithook(audit.hook)
    invocation = [
        "testuary", "--config", str(preset), "--model", alias,
        "--project-name", manifest["invocation"]["project"],
        "--source-dir", str(workspace / "projects" / manifest["exemplar"]["instance"]),
        "--source-files", manifest["invocation"]["source_files"],
        "--output-dir", str(outdir / "run"),
        *manifest["invocation"].get("options", []),
    ]
    if language != "python":
        invocation += ["--languages", language]
    if display is not None:
        invocation += ["--display", display]
    if getattr(args, "no_color", False):
        invocation += ["--no-color"]
    if getattr(args, "events", None):
        # Already an absolute path: the playback runs in its own directory.
        invocation += ["--events", args.events]
    failure = None
    interrupted = None
    # The pacer, and with it a stepping playback's key reader, is made right
    # before the block whose end restores the terminal's settings.
    pacer = None
    if playback is not None:
        pacing = _pacing()
        factor, step, seconds = playback
        pacer = pacing.Pacer(pacing.OBSERVATIONS[_dataset(manifest)], seconds, factor, step=step,
                             keys=pacing.keys() if step else None)
    with _paced(pacer), _workspace_environment(workspace, preset), \
            _playback_paths(outdir), _recorded_facts(manifest), _adapter(manifest):
        audit.active = True
        try:
            stop = play(scenario, invocation, on_answer=on_answer, language=language)
        except PipelineFailure as exc:
            stop, failure = None, exc
        except ScenarioStop as exc:
            stop = exc
        except BaseException as exc:
            stop, interrupted = None, exc
        finally:
            audit.active = False

    with open(records / "ledger.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for line in ledger:
            handle.write(json.dumps(line, sort_keys=True) + "\n")
    with open(records / "audit.jsonl", "w", encoding="utf-8", newline="\n") as handle:
        for event in audit.events:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
    # The guard's refusal of a process or a connection is no data stop.
    refused = stop is not None and stop.call.kind.startswith("guard.")
    classes: Dict[str, int] = {}
    for line in ledger:
        classes[line["class"]] = classes.get(line["class"], 0) + 1
    result = {
        "outcome": ("cancelled" if isinstance(interrupted, KeyboardInterrupt)
                    else "failed" if failure is not None or interrupted is not None or refused
                    else "completed" if stop is None else "stale"),
        "playback": _playback_text(playback),
        "answers": len(ledger),
        "classes": dict(sorted(classes.items())),
        "processes": sorted({Path(e["executable"]).name for e in audit.events
                             if e["event"] == "subprocess.Popen"}),
        "network_connections": sum(1 for e in audit.events
                                   if e["event"] == "socket.connect"),
    }
    if refused:
        result["refused"] = {"call": stop.call.describe(), "reason": stop.reason}
    elif stop is not None:
        stale = {"call": stop.call.describe(), "reason": stop.reason}
        (records / "stale.json").write_text(
            json.dumps(stale, indent=2) + "\n", encoding="utf-8")
        result["stale"] = stale
    result["labelled"] = _label_outputs(outdir, manifest)
    _write_manifest(outdir, manifest, config_rows, overridden, result)
    if interrupted is not None:
        raise interrupted
    print(bar)
    if failure is not None:
        print(f"Demonstration failed: {failure}. A target of the "
              f"demonstrated run failed to execute, and its log is under "
              f"{outdir / 'run' / 'logs'}.")
    elif stop is None:
        print(f"Demo complete: {len(ledger)} recorded model answers and measurement "
              f"results played back. {statement.split('.')[0]}.")
    elif refused:
        print(f"Demonstration stopped: {stop}. A library started a process or opened a "
              f"network connection during the playback, which the demonstration "
              f"refuses. The recorded data are not at fault, and the guard's record "
              f"is in {records / 'audit.jsonl'}.")
    else:
        print(f"Demonstration stopped: {stop}. The scenario does not fit this "
              f"package version. Its data require an explicit development "
              f"rebuild, and no automatic recovery or live request was attempted.")
    print(f"Run directory: {outdir}")
    print(bar, flush=True)
    if failure is not None:
        return failure.status
    if refused:
        return EXIT_FAILED
    return EXIT_OK if stop is None else EXIT_STALE


def _pacing():
    from testuary.demo import pacing

    return pacing


#: The variable that sets the playback's length, as [demo] playback_seconds.
PLAYBACK_VARIABLE = "TESTUARY_DEMO_PLAYBACK_SECONDS"


def _configured_playback(args: argparse.Namespace, cwd: Path,
                         environ: Dict[str, str]) -> Tuple[Optional[str], str]:
    """[demo] playback_seconds from the configuration files a normal run
    reads (``settings.resolve_settings``): the --config files in order, else
    the file TESTUARY_CONFIG_FILE names, else the user directory's config.ini,
    and then the project's settings file, the last one that sets it
    winning. Relative paths start from ``cwd``. Only this key is read.
    Returns the value and its source, or None when no file sets it."""
    named = list(getattr(args, "config", None) or [])
    if not named and environ.get("TESTUARY_CONFIG_FILE", "").strip():
        named = [environ["TESTUARY_CONFIG_FILE"]]
    files = [path if path.is_absolute() else cwd / path for path in map(Path, named)]
    for path in files:
        if not path.is_file():
            raise ValueError(f"configuration file is absent or not a file: {path}")
    files += [path for path, read in _normal_run_files(cwd, environ, bool(files), args)
              if read and path.is_file()]
    found: Tuple[Optional[str], str] = (None, "")
    for path in files:
        try:
            parser = settings_text(path.read_text(encoding="utf-8"))
        except (OSError, configparser.Error) as exc:
            raise ValueError(f"{path} cannot be read: {exc}") from None
        if parser.has_option("demo", "playback_seconds"):
            found = (parser.get("demo", "playback_seconds"), f"[demo] playback_seconds in {path}")
    return found


def _playback_seconds(args: argparse.Namespace, *, environ: Optional[Dict[str, str]] = None,
                      cwd: Optional[Path] = None) -> float:
    """The playback's length at speed 1, from the first that sets it:
    --demo-playback-seconds, the variable TESTUARY_DEMO_PLAYBACK_SECONDS, [demo]
    playback_seconds of the configuration files a normal run reads, then the
    packaged configuration. Raises ValueError for a value that is not a
    positive number of seconds, naming where it came from."""
    from testuary import settings

    environ = dict(os.environ) if environ is None else environ
    text = getattr(args, "demo_duration", None)
    source = "--demo-playback-seconds"
    if text is None and environ.get(PLAYBACK_VARIABLE, "").strip():
        text, source = environ[PLAYBACK_VARIABLE], PLAYBACK_VARIABLE
    if text is None:
        text, source = _configured_playback(args, Path.cwd() if cwd is None else cwd, environ)
    if text is None:
        text = settings._defaults().get("demo", {}).get("playback_seconds", "")
        source = "[demo] playback_seconds"
    try:
        seconds = float(text)
    except (TypeError, ValueError):
        seconds = math.nan
    if not math.isfinite(seconds) or seconds <= 0:
        raise ValueError(f"{source} must be a positive number of seconds: {text!r}")
    return seconds


def _playback_text(playback) -> str:
    """How DEMO.md names the playback: instant, stepped or its minimum length."""
    if playback is None:
        return "instant"
    factor, step, seconds = playback
    if step:
        return "one step per Enter press"
    return (f"paced, at least about {round(seconds / factor)} seconds "
            f"({seconds:g} s at speed {factor:g})")


def _playback(args: argparse.Namespace):
    """The playback the options ask for: None for instant, else the speed
    factor, whether to step and the length in seconds at speed 1. Raises
    ValueError for an unusable choice."""
    pacing = _pacing()
    factor = pacing.parse_speed(getattr(args, "demo_speed", "auto"))
    step = bool(getattr(args, "demo_step", False))
    if not step and (factor == 0 or (factor is None
                                     and not pacing.interactive(sys.stderr, os.environ))):
        # An instant playback reads no length, but one given on the command
        # line must still be usable.
        if getattr(args, "demo_duration", None) is not None:
            _playback_seconds(args)
        return None
    seconds = _playback_seconds(args)
    if step:
        # A person at a terminal: the display is shown there and Enter is read
        # from it. A null input device can claim to be a terminal by itself.
        try:
            keyboard = sys.stdin.isatty()
        except (AttributeError, ValueError, OSError):
            keyboard = False
        if not (keyboard and pacing.interactive(sys.stderr, os.environ)):
            raise ValueError("--demo-step needs an interactive terminal to read Enter from")
        return (factor or 1.0), True, seconds
    return ((1.0 if factor is None else factor), False, seconds)


@contextlib.contextmanager
def _paced(pacer) -> Iterator[None]:
    """Install ``pacer`` for the run. Ctrl+C ends every pause at once, in any
    thread, and then interrupts the run as it would without pacing."""
    if pacer is None:
        yield
        return
    import signal

    from testuary._progress import paced_display

    def interrupt(signum, frame):
        pacer.cancel()
        raise KeyboardInterrupt

    previous = signal.signal(signal.SIGINT, interrupt)
    try:
        with paced_display(pacer.on_event):
            yield
    finally:
        signal.signal(signal.SIGINT, previous)
        pacer.close()


def _scenario_path(files: Dict[str, bytes], records: Path) -> Path:
    path = records / "scenario.json.gz"
    path.write_bytes(files["scenario.json.gz"])
    return path


def _write_manifest(outdir: Path, manifest: Dict[str, Any],
                    config_rows: List[Tuple[str, str]],
                    overridden: List[Dict[str, str]],
                    result: Dict[str, Any]) -> None:
    record = {
        "statement": manifest["statement"],
        "language": manifest.get("language", "python"),
        "exemplar": manifest["exemplar"],
        "data_files": manifest["files"],
        "configuration": dict(config_rows),
        "overridden_settings": overridden,
        "result": result,
    }
    (outdir / "DEMO.json").write_text(json.dumps(record, indent=2) + "\n",
                                      encoding="utf-8")
    origin = manifest["exemplar"]["run"]
    rules = ("Every answer was recorded in one real run of this release "
             "(`recorded-v1`). `demo/ledger.jsonl` names each answer's source.")
    lines = [
        "# Testuary Demonstration", "", record["statement"], "",
        f"Example: `{manifest['exemplar']['instance']}`, "
        f"`{manifest['exemplar']['target']}`, {origin}.", "",
        "## Configuration", "",
    ]
    lines += [f"- {name}: {value}" for name, value in config_rows]
    if overridden:
        lines += ["", "## Ignored by the Demonstration", ""]
        lines += [f"- {item['setting']}: {item['instead']}" for item in overridden]
    lines += ["", "## Result", "", f"- outcome: {result['outcome']}",
              f"- playback: {result.get('playback', 'instant')}",
              f"- recorded answers played back: {result['answers']} "
              f"({_answer_classes(result['classes'])})",
              f"- processes started: {', '.join(result['processes']) or 'none'}",
              f"- network connections: {result['network_connections']}", "",
              "Every answer's class, rule and source is in `demo/ledger.jsonl`. "
              + rules + " " +
              "Every JSON record and JSON-lines log under `run/` and "
              "`workspace/` carries a `testuary_demo` field, and the results "
              "table a `demo` column. Suites, replies and other files whose "
              "bytes are recorded by hash are left as written, with a "
              "`DEMO_NOTICE.txt` beside them.", ""]
    (outdir / "DEMO.md").write_text("\n".join(lines), encoding="utf-8")


LABEL_FIELD = "testuary_demo"
TABLE_COLUMN = "demo"
NOTICE_NAME = "DEMO_NOTICE.txt"
# Written as they are and compared or hashed elsewhere: left unaltered.
RECORDED_LABEL = ("demo recorded in one real run of this Testuary release, not a "
                  "measurement. See DEMO.json at the output root.")
RECORDED_NOTICE = (
    "DEMO, not a measurement. These files were written by `testuary --demo` "
    "from model answers and measurement results recorded in one real run of "
    "this Testuary release. DEMO.json at the output root gives each answer's "
    "source."
)


def _label_json(path: Path, label: str) -> bool:
    text = path.read_text(encoding="utf-8")
    try:
        data = json.loads(text)
    except ValueError:
        return False
    if not isinstance(data, dict):
        return False
    data.pop(LABEL_FIELD, None)
    data = {LABEL_FIELD: label, **data}
    pretty = "\n " in text
    path.write_text(
        (json.dumps(data, indent=2) if pretty
         else json.dumps(data, separators=(",", ":"))) + "\n",
        encoding="utf-8",
    )
    return True


def _unused_run_dir(outdir: Path, events: Optional[str]) -> bool:
    """Whether ``outdir`` is a new or empty directory. The empty event file
    --events names there is the command line's own write check, which the
    run replaces when it starts, so it does not count."""
    if not outdir.exists():
        return True
    if not outdir.is_dir():
        return False
    probe = Path(events).resolve() if events else None
    return all(path.resolve() == probe and path.is_file() and path.stat().st_size == 0
               for path in outdir.iterdir())


def _label_jsonl(path: Path, label: str) -> bool:
    lines = path.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except ValueError:
            # A line that is no record takes no label, so the file is named
            # in its directory's notice as written.
            return False
        if isinstance(record, dict):
            record.pop(LABEL_FIELD, None)
            record = {LABEL_FIELD: label, **record}
        out.append(json.dumps(record, sort_keys=True, separators=(",", ":")))
    path.write_text("".join(item + "\n" for item in out), encoding="utf-8")
    return True


def _label_table(path: Path) -> bool:
    raw = path.read_bytes()
    bom = raw.startswith(b"\xef\xbb\xbf")
    text = raw.decode("utf-8-sig")
    rows = list(csv.reader(io.StringIO(text)))
    if not rows:
        return False
    rows[0].append(TABLE_COLUMN)
    for row in rows[1:]:
        row.append("demo")
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    path.write_text(buffer.getvalue(), encoding="utf-8-sig" if bom else "utf-8",
                    newline="")
    return True


def label_outputs(root: Path, *, label: str = RECORDED_LABEL,
                  notice: str = RECORDED_NOTICE, inputs: Sequence[Path] = ()) -> Dict[str, object]:
    """Label everything under ``root`` but ``inputs``, the settings files a
    run read, which stay as written. Return what was labeled and which files
    were left unaltered."""
    counts = {"json": 0, "jsonl": 0, "tables": 0, "notices": 0}
    unaltered: Dict[str, List[str]] = {}
    kept = {Path(path).resolve() for path in inputs}
    for path in sorted(root.rglob("*")):
        if not path.is_file() or path.name == NOTICE_NAME:
            continue
        if path.resolve() in kept:
            unaltered.setdefault(str(path.parent), []).append(path.name)
        elif path.suffix == ".json" and _label_json(path, label):
            counts["json"] += 1
        elif path.suffix == ".jsonl" and _label_jsonl(path, label):
            counts["jsonl"] += 1
        elif path.suffix == ".csv" and _label_table(path):
            counts["tables"] += 1
        else:
            unaltered.setdefault(str(path.parent), []).append(path.name)
    for directory, names in sorted(unaltered.items()):
        notice_path = Path(directory) / NOTICE_NAME
        notice_path.write_text(
            notice + " They are left as written, as their bytes are compared or "
            "hashed in the records or their format takes no label:\n"
            + "".join(f"  {name}\n" for name in sorted(names)),
            encoding="utf-8",
        )
        counts["notices"] += 1
    return {
        "counts": counts,
        "unaltered_files": sum(len(names) for names in unaltered.values()),
    }
