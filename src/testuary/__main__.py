import argparse
import importlib
import os
import shutil
from pathlib import Path
from typing import Optional

from testuary import project as project_module
from testuary import results
from testuary import exits
from testuary import languages
from testuary import orchestrator
from testuary.orchestrator import log_print
from testuary.settings import _CHOICES as _SETTING_CHOICES
from testuary.settings import _OptionParser


# The command line: argument parsing and the subcommands. The per-target
# flow lives in orchestrator.


def _provision_parser() -> argparse.ArgumentParser:
    return argparse.ArgumentParser(
        allow_abbrev=False,
        prog="testuary provision",
        description="Install the runtime the settings of the project in this directory ask for "
                    "(such as a Python version with its requirements, made with uv in "
                    "Testuary's cache), so that a run need not do it. May download.",
        epilog="Exit status: 0 when the runtime the settings ask for is ready, or when they "
               "ask for none, 1 when the settings cannot be read or the runtime cannot be "
               "installed, 2 when an option is refused.")


def _cmd_provision(argv) -> int:
    """Prepare and print the runtime each installed language's settings
    request for the project in this directory, as its setup context describes
    it (such as a Python version to provision). This command allows
    environment preparation to finish before generation begins."""
    import sys as _sys

    _provision_parser().parse_args(argv)
    from langadapter import names

    from testuary import config
    from testuary.settings import resolve_settings

    prepared = []
    try:
        settings = resolve_settings(load_extensions=True)
        with config.use_run_settings(settings):
            for language in names():
                described = languages.setup_context(language, settings).interpreter
                request = getattr(described, "provisioning", None)
                if request is not None:
                    prepared.append((language, project_module.provision_runtime(language, request)))
    except (ValueError, RuntimeError) as exc:
        print(exc, file=_sys.stderr)
        return 1
    for language, runtime in prepared:
        print(f"{language}: {runtime} (ready)")
    if not prepared:
        print("No runtime to install: the project's settings ask for none.")
    return 0


def _init_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        prog="testuary init", description="Copy example files: models.ini to your user directory and "
                                          "hints.json to the project. Existing files are kept.",
        epilog="Exit status: 0 when the files are copied or exist already, 1 when a packaged "
               "example is missing or a copy fails, 2 when an option is refused.")
    parser.add_argument(
        "--examples", action="store_true",
        help="Copy the extension examples (a client factory, a language-specific prompt "
             "profile and per-language hints) into ./testuary-examples instead.")
    return parser


def _cmd_init(argv=()) -> int:
    """``testuary init``: copy the bundled starters from
    ``testuary/_templates/``: ``models.ini`` into the user directory and
    ``hints.json`` into the project directory, found from the current directory
    upward as a run finds it, where a run's default hints file is read. With
    ``--examples`` it copies the extension examples instead.

    Existing files are never overwritten (the user's models.ini may hold
    real secrets). Skipped files are reported.

    Returns the exit code: 0 on success (or 'all already exist'),
    1 when a bundled template is missing or a copy fails.
    """
    import sys as _sys
    from testuary.settings import file_locations, project_locations
    from testuary.settings import _bundled_template_path

    if _init_parser().parse_args(argv).examples:
        return _copy_examples(Path.cwd() / "testuary-examples")
    models = file_locations().models
    copies = [
        ("models.ini", models),
        ("hints.json", project_locations(Path.cwd()).project / "hints.json"),
    ]
    rc = 0
    for bundled_name, target in copies:
        if target.exists():
            print(f"  skip: {target.name} already exists in {target.parent}")
            continue
        src = _bundled_template_path(bundled_name)
        if src is None:
            print(
                f"  error: bundled template {bundled_name} not found in the "
                "installed testuary package",
                file=_sys.stderr,
            )
            rc = 1
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            _write_new(target, src.read_bytes(), private=bundled_name == "models.ini")
            print(f"  wrote: {target}")
        except FileExistsError:
            print(f"  skip: {target.name} already exists in {target.parent}")
        except OSError as exc:
            print(f"  error copying {bundled_name} to {target}: {exc}", file=_sys.stderr)
            rc = 1
    if rc == 0:
        print(
            f"\nAdd a section for your model to {models}. Its name is the alias for "
            "--model. Then run testuary check --model ALIAS in your project directory."
        )
    return rc


def _write_new(target: Path, data: bytes, *, private: bool) -> None:
    """Create ``target`` with ``data``, never over an existing file, even one
    made since it was looked for (FileExistsError). A private file, the
    model catalogue into which keys are typed, is readable by its owner
    only on POSIX, from its creation on."""
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_BINARY", 0)
    descriptor = os.open(target, flags, 0o600 if private else 0o666)
    with os.fdopen(descriptor, "wb") as stream:
        stream.write(data)


def _copy_examples(target_root: Path) -> int:
    """Copy the bundled extension examples below ``target_root``. An existing
    file is left unchanged. Returns 0, or 1 when a file could not be written."""
    import sys as _sys
    from testuary.settings import _bundled_template_path

    examples = _bundled_template_path("examples")
    if examples is None:
        print("  error: the bundled examples were not found in the installed testuary package",
              file=_sys.stderr)
        return 1
    rc = 0
    for source in sorted(path for path in examples.rglob("*") if path.is_file()):
        relative = source.relative_to(examples)
        if "__pycache__" in relative.parts:
            continue
        target = target_root / relative
        if target.exists():
            print(f"  skip: {target} already exists")
            continue
        try:
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
            print(f"  wrote: {target}")
        except OSError as exc:
            print(f"  error copying {relative} to {target}: {exc}", file=_sys.stderr)
            rc = 1
    if rc == 0:
        print(f"\nSee {target_root / 'README.txt'} to check a project with the examples.")
    return rc


def _check_run_options(args) -> Optional[set]:
    """Check the run options together (``orchestrator.check_run_options``),
    set the per-phase reuse booleans the pipeline reads (``args.reuse_starter``
    .. ``args.reuse_finishing``) and return the selected languages, None for
    every registered one. Raises ValueError naming what is wrong."""
    languages = (tuple(results.parse_csv_field(args.languages))
                 if args.languages else None)
    reuse, languages = orchestrator.check_run_options(
        runs=args.runs, run=args.run, reuse=tuple(results.parse_csv_field(args.reuse or "")),
        resume=args.resume, stop_after_phase=args.stop_after_phase, languages=languages or None)
    args.reuse = ",".join(reuse) or None
    for phase in reuse:
        setattr(args, f"reuse_{phase}", True)
    return set(languages) if languages else None


def _reuse_text(text: str) -> str:
    """``--reuse``'s text, refused when it names no phase."""
    if not results.parse_csv_field(text):
        raise argparse.ArgumentTypeError(f"no phase named. Choose from {', '.join(orchestrator.PHASES)}")
    return text


def _cmd_new_language(argv) -> int:
    """LangAdapter's scaffold for a new language, written for Testuary:
    ``langadapter new-language`` with ``--for testuary`` and the same
    arguments, so a language started from Testuary lands where one started
    from LangAdapter does. Its usage and help name ``testuary
    new-language``."""
    from langadapter import cli

    argv = list(argv)
    if "--for" not in argv and not any(item.startswith("--for=") for item in argv):
        argv += ["--for", "testuary"]
    return cli.main(["new-language", *argv], prog="testuary")


def version_lines() -> str:
    """The Testuary and LangAdapter versions in use, one per line: each
    package's own ``__version__``, else its installed metadata. The package's
    own value holds for an editable install too, whose metadata keeps the
    version it was installed at."""
    from importlib import metadata

    lines = []
    for name in ("testuary", "langadapter"):
        try:
            version = getattr(importlib.import_module(name), "__version__", None)
        except ImportError:
            version = None
        if not version:
            try:
                version = metadata.version(name)
            except metadata.PackageNotFoundError:
                version = "not installed"
        lines.append(f"{name} {version}")
    return "\n".join(lines)


class _VersionAction(argparse.Action):
    """Print the versions and exit, as argparse's version action does, but
    keeping one line per package."""

    def __init__(self, option_strings, dest=argparse.SUPPRESS, default=argparse.SUPPRESS,
                 help="Print the installed Testuary and LangAdapter versions and exit."):
        super().__init__(option_strings, dest=dest, default=default, nargs=0, help=help)

    def __call__(self, parser, namespace, values, option_string=None):
        print(version_lines(), flush=True)
        parser.exit(0)


def _primary_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        allow_abbrev=False,
        prog="testuary", description="Generate and measure tests for selected targets.",
        # A command is the first word, so no option stands before one.
        usage="%(prog)s [-h] [--version]\n"
              "       %(prog)s [--demo [--demo-language LANGUAGE]] [--home DIR] [FILE ...]\n"
              "       %(prog)s COMMAND [OPTIONS]",
        epilog="generate (the default in a project) writes and measures tests, and its "
               "flags also work without 'generate'. check checks the setup without "
               "contacting a model (the quick check). setup asks for model access, what "
               "runs in parallel on this machine and a project. init copies example files. "
               "provision installs the runtime the project asks for. new-language writes a "
               "new language's file. tutorial teaches the extension points offline. "
               "Use 'testuary [FILES]' in a project to run its [project] source_files, "
               "or the targets named.",
    )
    parser.add_argument(
        "command", nargs="?", choices=("generate", "check", "init", "provision", "new-language", "setup", "tutorial"),
        help="Select a command, each described below. COMMAND --help lists its options.",
    )
    parser.add_argument("--version", action=_VersionAction)
    # The demonstration and the user directory also work without a
    # command, so the main help lists them with the help a run gives them.
    for action in _build_parser()._actions:
        if action.dest in ("demo", "demo_language", "home"):
            parser._add_action(action)
    return parser


#: The demonstration's options, by destination: check refuses them, and a
#: demonstration takes them while it names every other option it overrides.
DEMO_OPTIONS = frozenset({"demo", "demo_language", "demo_data", "demo_speed", "demo_duration", "demo_step"})


def _build_parser(*, prog: str = "testuary", checking: bool = False) -> argparse.ArgumentParser:
    """Construct the pipeline argparse parser.

    A separate function so tests and the project-command parsing can
    introspect every registered flag without running the pipeline.
    The subcommand keywords (``generate``, ``check``, ``init``, ``provision``,
    ``new-language``, ``setup``, ``tutorial``) are handled before this parser sees argv and
    so are not registered here.
    """
    parser = _OptionParser(
        prog=prog, allow_abbrev=False,
        description="Generate test suites for the selected targets with a language model. A "
                    "run asks the model for several independent test suites (the starter "
                    "suites) and combines them, improves the result in a coverage phase and a "
                    "mutation phase, and measures it in Finishing.")
    if checking:
        parser.description = ("Check the setup without contacting a model or installing anything "
                              "(the quick check). --verify also starts the tools and tries "
                              "the project in a copy.")
        parser.add_argument(
            "--verify", action="store_true",
            help="Also start the tools a run needs once and try the project in a copy as a run "
                 "does, to see that they work (the tool check). Python imports pytest, "
                 "coverage and the mutation tool with the project's interpreter and collects "
                 "the tests. Java runs java and javac, resolves the build, compiles and finds "
                 "the tests, and starts the measurement tools a run needs. The first run tries "
                 "the build of a project with its own build wrapper, which may download its "
                 "build tool. The check runs no test and contacts no model. For Java it "
                 "downloads what a run would when it is missing: what the build needs and, "
                 "unless LANGADAPTER_JAVA_TOOLS_OFFLINE is set, the JUnit launcher, JaCoCo "
                 "and PIT. "
                 "What it cannot try, and a language whose adapter has no check of its own, "
                 "is listed under Not tried yet.",
        )
        parser.add_argument("--json", dest="json_output", action="store_true",
                            help="Print the report as JSON instead of text (API keys removed).")
    parser.add_argument(
        "files", nargs="*", default=[], metavar="FILE",
        help=("Targets to run, relative to the directory the command starts in. "
              "Without them, the project's [project] source_files. --source-files "
              "names them relative to the source directory instead."),
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help=(
            "Play back one recorded run offline: every model answer and "
            "measurement comes from a recording shipped with Testuary, so it "
            "needs no model, network, project or test tools. It uses the "
            "recorded settings and lists any of yours it overrides. The run "
            "directory, the display options, the event file and the "
            "demonstration's language are yours."
        ),
    )
    parser.add_argument(
        "--demo-language", default="python", metavar="LANGUAGE",
        help="Language of the offline demonstration: python (the default), java, or "
             "another language whose demonstration data come with Testuary, such as toy. "
             "Requires --demo.",
    )
    parser.add_argument(
        "--demo-data", default=None, metavar="NAME",
        help="Demonstration data that come with Testuary, played in place of a "
             "language's own, such as a run recorded for a tutorial. Another name is "
             "refused with the names Testuary has. Requires --demo, without --demo-language.",
    )
    parser.add_argument(
        "--demo-speed", default="auto", metavar="SPEED",
        help="Pace of the demonstration's playback: auto (paced on an interactive "
             "terminal, instant when output is redirected, in tests or in continuous "
             "integration), instant, or a factor of the configured length's speed such "
             "as 2 (half as long) or 0.5 (twice as long). Pacing changes no value or "
             "record. Requires --demo.",
    )
    parser.add_argument(
        "--demo-playback-seconds", metavar="SECONDS", dest="demo_duration",
        help="Length of a paced playback in seconds, at --demo-speed 1, instead of "
             "TESTUARY_DEMO_PLAYBACK_SECONDS or [demo] playback_seconds. The "
             "demonstrations guide describes the length. Requires --demo.",
    )
    parser.add_argument(
        "--demo-step", action="store_true",
        help="Advance the demonstration's playback one step at a time when Enter is "
             "pressed, for screenshots. Requires --demo and an interactive terminal.",
    )
    if checking:
        # check refuses the demonstration's options, so its help leaves them out.
        for action in parser._actions:
            if action.dest in DEMO_OPTIONS:
                action.help = argparse.SUPPRESS
    parser.add_argument(
        "--display", choices=("auto", "compact", "detailed", "plain", "debug"), default="auto",
        help="How the run is shown: auto (compact on a terminal, plain otherwise), compact (one "
             "updating screen), plain (one line per change), detailed (also every step and the "
             "full setup report), debug (also every log line and internal event names).",
    )
    parser.add_argument(
        "--no-color", action="store_true",
        help="Disable terminal colors.",
    )
    if not checking:
        parser.add_argument(
            "--events", metavar="FILE",
            help="Also write the run's progress to FILE as JSON lines, for programs that show "
                 "a run (the terminal guide describes the records). The directory must exist, "
                 "and an existing "
                 "file is replaced. Works with --demo.",
        )
    parser.add_argument(
        "--config",
        action="append",
        help=(
            "Read settings from this INI file in place of the config.ini in "
            "your user directory. Repeat it for several files: later files win. "
            "The configuration guide describes the layers that override it."
        ),
    )
    parser.add_argument(
        "--no-project-settings",
        action="store_true",
        help=(
            "Read no project's settings file (.testuary/project.ini), here or in a "
            "directory above: for runs whose settings are all given in their own "
            "settings files. The working directory is then the project directory."
        ),
    )
    parser.add_argument(
        "--home",
        metavar="DIR",
        help=(
            "The user directory that holds models.ini and the global config.ini, "
            "in place of TESTUARY_HOME or the platform's user directory."
        ),
    )
    parser.add_argument(
        "--model",
        help="Override [api] model: a model alias of models.ini, or a model name [api] provider serves.",
    )
    parser.add_argument(
        "--proxy-port",
        type=int,
        help=(
            "Use a model server on this port of this machine: sets "
            "[api] api_url to http://localhost:<port>/v1, which wins over "
            "the endpoint of the models.ini alias and of the provider "
            "section. --api-url wins over it."
        ),
    )
    parser.add_argument(
        "--api-url",
        help=(
            "Override [api] api_url with a full base URL, for example "
            "http://localhost:8010/v1. It wins over --proxy-port and over "
            "the endpoint of the models.ini alias and of the provider "
            "section."
        ),
    )
    parser.add_argument(
        "--prompt-profile",
        help="A JSON file that changes the prompts sent to the model (examples: testuary init "
             "--examples). Read when the command starts.",
    )
    parser.add_argument(
        "--hints-file",
        dest="hints_file",
        help=(
            "Hints file to read from the working directory, in place of "
            "[main] hints_file. The hints guide says when a file is optional "
            "and what refuses a run."
        ),
    )
    parser.add_argument(
        "--reuse-starter-suites",
        dest="reuse_starter_suites",
        action="store_true",
        default=None,
        help=(
            "Run the starter phase again from the separate starter suites an "
            "earlier run saved: check and combine them again, and generate only "
            "a missing or empty suite again, which sends model requests (without "
            "saved suites, every one). --reuse starter instead keeps the "
            "combined result. Sets [starter] reuse_suites."
        ),
    )
    parser.add_argument(
        "--phase-scores",
        dest="phase_scores",
        choices=_SETTING_CHOICES["phase_scores"],
        default=None,
        help=(
            "What tests.csv reports for the delivered suites of the starter "
            "and coverage phases, read from what those phases recorded: "
            "full reports their line and branch coverage and their "
            "mutation score over the full catalogue, coverage omits the mutation "
            "score, off omits these columns. Nothing is measured again for "
            "them. Overrides [finishing] phase_scores."
        ),
    )
    parser.add_argument(
        "--project-name",
        help=(
            "The project's name in results and records (${project}). Overrides "
            "[project] name. Empty uses the source directory's name."
        ),
    )
    parser.add_argument(
        "--source-dir",
        dest="source_dir",
        help="The source directory, whose files are tested. Without it, the project directory: the one "
             "whose .testuary directory is found from the current directory upward, else the current "
             "directory. A relative path is taken from the current directory. In container mode give "
             "the absolute path inside the container.",
    )
    parser.add_argument(
        "--python-executable",
        help="The project's Python interpreter ([project] python_executable). It installs nothing.",
    )
    parser.add_argument("--java-home", help="The project's JDK home or java executable ([project] java_home).")
    parser.add_argument("--build-java-home", help="The build tool's JDK home ([project] build_java_home).")
    parser.add_argument(
        "--output-dir",
        dest="output_dir",
        help=(
            "The run directory: the results of each target and all logs go "
            "below it. A relative path is taken from the current directory. "
            "Overrides [project] output_dir, "
            "which may contain ${model} and ${project} placeholders."
        ),
    )
    parser.add_argument(
        "--python-import-root",
        help=(
            "The directory from which the project's Python package is imported "
            "([project] python_import_root). pytest still runs in the source "
            "directory, with the project's own configuration."
        ),
    )
    parser.add_argument(
        "--languages",
        help=(
            "Refuse the run if a target is in a language outside this "
            f"list (installed: {', '.join(languages.available_language_names())}). "
            "Each target's language comes from its file extension. Without this "
            "option every installed language is allowed."
        ),
    )
    parser.add_argument(
        "--targets",
        type=str,
        help=(
            "Run only these targets of the --source-files list, by position, "
            "one after another: one position (`7`), a range (`1-5`), or an "
            "open range (`5-` for the fifth to the end, `-5` for the first "
            "five). Default: the whole list. "
            "Outside container mode each target runs in its own copy of the "
            "source directory, so one target's edits cannot change the next "
            "one's measurements."
        ),
    )
    parser.add_argument(
        "--source-files",
        default=None,
        dest="source_files",
        help=(
            "Comma list of targets to generate tests for, in the "
            "order they run. A relative path is taken from the source "
            "directory. Each target must exist and be in an installed language. "
            "Without --targets the run covers the whole list."
        ),
    )
    parser.add_argument(
        "--concurrent-requests",
        dest="concurrent_requests",
        type=int,
        help=(
            "How many model requests for the starter suites run at once "
            "(concurrent requests), in place of [starter] concurrent_requests "
            "and concurrent_requests_ollama. Fewer put less load on the "
            "model server and shorten each request, which helps models "
            "that write long answers."
        ),
    )
    stop_phases = orchestrator.available_phase_names()
    parser.add_argument(
        "--stop-after-phase",
        type=lambda text: text.strip().lower(),
        metavar="{" + ",".join(stop_phases) + "}",
        default=None,
        help=(
            "Stop each target after this phase has saved its result, to try "
            f"a phase alone: {', '.join(stop_phases)}. The default runs "
            "all four phases. Gives no final scores, and each target is "
            "reported as stopped."
        ),
    )
    parser.add_argument(
        "--reuse",
        type=_reuse_text,
        default=None,
        help=(
            "Use the saved results of these phases from the earlier run in "
            "this run directory instead of running them again: "
            f"{', '.join(orchestrator.PHASES)}. The results guide says what "
            "each one reuses. Example: --reuse coverage."
        ),
    )
    # Every per-phase reuse boolean exists on args whether or not --reuse
    # names its phase, so the pipeline's reads never depend on the flags
    # given.
    parser.set_defaults(
        reuse_starter=False, reuse_coverage=False, reuse_mutation=False,
        reuse_finishing=False,
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        dest="resume",
        help=(
            "Continue an interrupted run in the same run directory: each target "
            "and repetition continues from the first phase that did not "
            "finish. Finished phases (starter, coverage, mutation, finishing) "
            "are reused if their saved results and the target's source are "
            "unchanged. Combines only with --reuse starter."
        ),
    )

    # -----------------------------------------------------------------
    # Runs. --runs N runs a target's test pipeline N times, as runs r0 to
    # r(N-1), each in its own directory results/<workspace>/r<k>/
    # (project.run_dir). --run K runs only rK.
    # -----------------------------------------------------------------
    parser.add_argument(
        "--runs", type=int, default=1, dest="runs", metavar="N",
        help=(
            "Run each target's four phases N times, as repetitions r0 to "
            "r(N-1), each from a fresh copy of the project and in its own "
            "directory. N must be 1 with --stop-after-phase."
        ),
    )
    parser.add_argument(
        "--run", type=int, default=None, dest="run", metavar="K",
        help=(
            "Run only repetition K (rK) of each target, the same one --runs "
            "makes when N is above K, for example to add a missing repetition."
        ),
    )
    return parser


def _is_check_command() -> bool:
    import sys

    return sys.argv[1:2] == ["check"]


def _resumable(argv) -> bool:
    """Whether --resume continues ``argv``: a run, neither a demonstration
    nor one whose --reuse selection goes beyond starter."""
    argv = list(argv)
    if argv[:1] in (["check"], ["init"], ["provision"], ["new-language"], ["tutorial"],
                    ["setup"]) or "--demo" in argv:
        return False
    reuse = []
    for index, item in enumerate(argv):
        # As argparse does, the last --reuse wins.
        if item == "--reuse" and index + 1 < len(argv):
            reuse = argv[index + 1].split(",")
        elif item.startswith("--reuse="):
            reuse = item.split("=", 1)[1].split(",")
    return not {unit.strip().lower() for unit in reuse if unit.strip()} - {"starter"}


def _interrupt_advice(argv) -> str:
    """What the person can do after interrupting ``argv``. Only commands that
    can continue are told how."""
    argv = list(argv)
    if argv[:1] == ["check"]:
        return "No setup was changed."
    if argv[:1] in (["init"], ["provision"], ["new-language"]):
        return "Run the command again when ready."
    if argv[:1] == ["tutorial"]:
        return "Start the tutorial again with a new or empty --output-dir."
    if "--demo" in argv:
        return ("Start the demonstration again with a new or empty --output-dir. "
                "A demonstration cannot be resumed.")
    if argv[:1] == ["setup"]:
        return "Run the command again when ready."
    if not _resumable(argv):
        return ("Finished results and saved model answers are kept. Continue with the "
                "same command (--resume does not combine with this --reuse selection).")
    return ("Finished results and saved model answers are kept. Continue with the "
            "same command and --resume.")


def main():
    """The process entry: an interrupted command ends with a short report and
    exit status 130 instead of a traceback."""
    import sys as _sys

    try:
        return run_command()
    except KeyboardInterrupt as stop:
        remains = _retry_interrupted_cleanup(stop)
        # The display has already reported the cancellation. A Python
        # traceback would add nothing for the person who pressed Ctrl+C.
        print("Stopped by you. " + _interrupt_advice(_sys.argv[1:]), file=_sys.stderr, flush=True)
        if remains is not None:
            print(remains, file=_sys.stderr, flush=True)
        return exits.INTERRUPTED
    except orchestrator._PhaseStopNotReached as exc:
        # The run's log names why the phase did not reach the stop.
        print(f"Stopped: {exc}. The run's log says why.",
              file=_sys.stderr, flush=True)
        return 1


def _retry_interrupted_cleanup(stop: BaseException) -> Optional[str]:
    """Close once more the mutation session whose own cleanup an interrupt
    cut short, as a second Ctrl+C during the first one's cleanup does: the
    interrupt carries the session as its cleanup handle. Returns a line
    naming what remains when that close fails too, else None. The exit stays
    the interrupted one either way. From here on a further Ctrl+C is
    ignored: it would cut this cleanup short, or kill the exit itself."""
    import signal
    import threading

    from testuary.languages.mutation import close_cancelled

    if threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGINT, signal.SIG_IGN)

    failures = []
    handle = getattr(stop, "mutation_cleanup_handle", None)
    if handle is not None:
        try:
            handle.close()
        except BaseException as exc:
            failures.append(exc)
    # A session the stop cancelled whose close the interrupt skipped.
    failures += close_cancelled()
    if failures:
        exc = failures[0]
        return (f"The mutation tool may still be running after the interruption "
                f"({type(exc).__name__}: {exc}). Look for leftover cosmic-ray or Java "
                "processes.")
    return None


def run_command():
    """Run the command in ``sys.argv`` and return its status.

    An interruption propagates, so an in-process caller such as the
    demonstration records it as a cancellation. ``main`` reports it for the
    process entry.
    """
    checking = _is_check_command()
    try:
        return _main()
    except Exception as exc:
        # A read-only inspection must not write a previous generation's log,
        # and a requested stop that was not reached is no program failure.
        if checking or isinstance(exc, orchestrator._PhaseStopNotReached):
            raise
        from testuary.model_requests import failure_traceback
        try:
            log_print(
                "FATAL: uncaught exception ended the run:\n"
                + failure_traceback(exc),
                level=0,
            )
        except Exception:
            pass
        raise


def _resume_command(argv):
    """The exact command that continues ``argv``, or None where --resume does
    not apply (a demonstration, or a --reuse selection beyond starter)."""
    import os
    import shlex
    import subprocess

    argv = list(argv)
    if not _resumable(argv):
        return None
    command = ["testuary", *argv] + ([] if "--resume" in argv else ["--resume"])
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def _present_plan(plan, presenter, events=None) -> None:
    """The run's plan to the event file, if any, and to the display."""
    _record(presenter, events, "plan", plan)
    show = getattr(presenter, "plan", None)
    if show is not None:
        show(plan)


def _present_completion(completion, presenter, events=None):
    from dataclasses import replace
    import sys as _sys

    from testuary.presentation.terminal import present_completion

    completion = replace(completion, resume_command=_resume_command(_sys.argv[1:]))
    try:
        from testuary.presentation.terminal import write_completion

        # With the command that continues the run, as the console shows it.
        write_completion(completion, provenance=presenter.provenance,
                         clean=getattr(presenter, "clean", None))
    except Exception as exc:
        log_print(f"WARNING: the end-of-run summary could not be written ({type(exc).__name__}: {exc})")
    _record(presenter, events, "completion", completion)
    present_completion(completion, presenter=presenter)


def _event_file_path(parser, value) -> str:
    """``--events`` as an absolute path, checked before anything runs. The
    demonstration plays its run in a directory of its own, so a relative path is
    resolved here. The write is tried by appending, so a command refused
    after this check leaves an earlier file as it was. The run replaces the
    file when it starts."""
    try:
        path = orchestrator._writable_event_file(value)
    except ValueError as exc:
        parser.error(f"--events: {exc}")
    try:
        with open(path, "a", encoding="utf-8"):
            pass
    except OSError as exc:
        parser.error(f"--events: cannot write {path}: {exc.strerror or exc}")
    return str(path)


def _display_lock(presenter):
    import threading

    return getattr(presenter, "_lock", None) or threading.RLock()


def _observe(presenter, events):
    """The run's event sink. With an event file each event is written there,
    then shown, both under the display's lock, so the file holds the events in
    the order the display shows them. Without one the display is the sink."""
    if events is None:
        return presenter
    lock = _display_lock(presenter)

    def observe(event):
        with lock:
            events(event)
            presenter(event)

    return observe


def _record(presenter, events, kind, value) -> None:
    """Write the setup report or the completion to the event file, if any,
    under the display's lock, just before the display shows it."""
    if events is not None:
        with _display_lock(presenter):
            getattr(events, kind)(value)


def _project_settings_here() -> bool:
    """Whether the directory the command starts in belongs to a project with a
    settings file, found as a run finds it."""
    from testuary.settings import project_locations

    try:
        return project_locations(Path.cwd()).project_settings.is_file()
    except (OSError, ValueError):
        return False


def _project_files(named, source_files, settings, *, start):
    """The source files a command runs, relative to the source directory, and
    whether they are the project's own list. They are the files named on
    the command line, relative to ``start``, else ``--source-files`` as
    given, else the project's ``[project] source_files``, and none when it
    lists none: the run then refuses itself, after the checks before it. A
    named file that does not exist, is a directory, lies outside the source
    directory, or is of another language than ``[project] language``, is
    refused, unless that language is unchecked: an extension the settings
    name may add it, which the tool check and a run check after loading."""
    from testuary.settings import ConfigurationError

    project = settings.sections["project"]
    if named and source_files:
        raise ConfigurationError("name the targets once: as FILE arguments or with --source-files")
    if source_files:
        return tuple(results.parse_csv_field(str(source_files))), False
    if not named:
        listed = tuple(results.parse_csv_field(str(project.get("source_files") or "")))
        return listed, bool(listed)
    root_text = str(project.get("source_dir") or "").strip()
    root = Path(root_text) if root_text else settings.project_dir
    root = (root if root.is_absolute() else settings.base_dir / root).resolve()
    language = str(project.get("language") or "").strip().lower()
    if "[project] language" in settings.unchecked:
        language = ""
    files = []
    for name in named:
        path = Path(name)
        path = (path if path.is_absolute() else Path(start) / path).resolve()
        try:
            relative = path.relative_to(root).as_posix()
        except ValueError:
            raise ConfigurationError(
                f"{name} lies outside the source directory {root}") from None
        if path.is_dir():
            raise ConfigurationError(f"{name} is a directory, not a source file")
        if not path.is_file():
            raise ConfigurationError(f"{name} does not exist")
        if language and languages.detect_language(str(path)) != language:
            raise ConfigurationError(f"{name} is not a {language} source file")
        files.append(relative)
    return tuple(files), False


def _run_cli_invocation(args, selected_languages, *, presenter, checking=False, event_sink=None,
                        events=None):
    """One resolved CLI invocation and one scoped admission-report boundary."""
    from testuary.presentation import present_readiness
    from testuary.readiness import _configuration_error_report
    from testuary.settings import ConfigurationError, resolve_settings

    settings = None
    refusal = None
    try:
        try:
            # A run and the tool check load the extensions, the quick check none.
            settings = resolve_settings(config_files=args.config, home=getattr(args, "home", None),
                                        project_settings=not getattr(args, "no_project_settings", False),
                                        load_extensions=not checking or bool(getattr(args, "verify", False)),
                                        overrides=orchestrator._from_working_dir(
                                            orchestrator._cli_settings_overrides(args)))
        except (ValueError, TypeError) as exc:
            raise ConfigurationError(str(exc)) from exc
        redact_with = getattr(presenter, "redact_with", None)
        if redact_with is not None:
            # The run's own keys are removed exactly, also where no pattern
            # recognizes them, as the check's report removes them.
            from testuary.settings import _redact_diagnostic

            resolved = settings
            redact_with(lambda text: _redact_diagnostic(text, resolved))
        if not getattr(args, "demo", False):
            files, from_project = _project_files(getattr(args, "files", None) or [], args.source_files,
                                                 settings, start=Path.cwd())
            args.source_files = files
            if from_project and not checking and len(files) > 1:
                print("\n".join([f"Project {settings.project_dir.name}: {len(files)} targets, "
                                  "each a full run:",
                                  *(f"  {index}. {path}" for index, path in enumerate(files, 1))]),
                      flush=True)
        if checking:
            try:
                request = orchestrator._run_request(args)
            except (ValueError, TypeError) as exc:
                raise ConfigurationError(str(exc)) from exc
            report = orchestrator.check(request, settings=settings, verify=args.verify)
        else:
            invocation = orchestrator._without_cli_configuration(args)
            from testuary._progress import demonstrating

            def pre_run(report):
                _record(presenter, events, "readiness", report)
                present_readiness(report, presenter=presenter, before_run=True, playback=demonstrating())

            invocation._pre_run = pre_run
            invocation._on_plan = lambda plan: _present_plan(plan, presenter, events)
            invocation._on_complete = lambda completion: _present_completion(
                completion, presenter, events)
            sink = presenter if event_sink is None else event_sink
            return orchestrator.run_pipeline(invocation, selected_languages, settings=settings, event_sink=sink)
    except ConfigurationError as exc:
        report = _configuration_error_report(
            exc, settings=settings, scope="verification" if checking and args.verify else "inspection",
            before_run=not checking)
        refusal = exc
    if checking and args.json_output:
        import json

        print(json.dumps(report.as_dict(), ensure_ascii=True, allow_nan=False))
    else:
        # A refused generate ran no check: its report is the setup before the run.
        options = {} if checking else {"before_run": True}
        if not checking:
            _record(presenter, events, "readiness", report)
        present_readiness(report, presenter=presenter, **options)
    if refusal is not None:
        raise SystemExit(report.exit_code) from refusal
    return report.exit_code


def _main():
    # Template commands parse their arguments before accessing the filesystem.
    import sys as _sys

    # Use UTF-8 for parent-process output as well as subprocess output.
    # Replacement handling prevents unsupported terminal characters from
    # interrupting a run.
    for _stream in (_sys.stdout, _sys.stderr):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, OSError):
            # A replaced stream may not be a text wrapper, and non-TTY
            # streams (pipes, files) may reject the call. Either case
            # leaves the original stream in effect.
            pass

    argv = _sys.argv[1:]
    if not argv:
        if not _project_settings_here():
            _primary_parser().print_help()
            return 2
        # In a project, the bare command runs its source files.
        argv = ["generate"]
    if argv in (["--help"], ["-h"], ["--version"]):
        _primary_parser().parse_args(argv)

    # The subcommands with their own parsers. Setup and the tutorial load
    # their modules only when chosen.
    subcommands = {
        "init": _cmd_init, "provision": _cmd_provision, "new-language": _cmd_new_language,
        "setup": lambda rest: importlib.import_module("testuary.setup").main(rest),
        "tutorial": lambda rest: importlib.import_module("testuary.tutorial").main(rest),
    }
    if argv[0] in subcommands:
        _sys.exit(subcommands[argv[0]](argv[1:]))

    explicit_generate = argv[0] == "generate"
    checking = argv[0] == "check"
    if explicit_generate or checking:
        argv = argv[1:]
    parser = _build_parser(prog="testuary check" if checking else "testuary generate" if explicit_generate else "testuary",
                           checking=checking)
    if not argv and not _project_settings_here():
        parser.print_usage()
        print("No project here and no targets named: run testuary setup, or name the targets.")
        return 2
    args = parser.parse_args(argv)
    if getattr(args, "events", None) is not None:
        args.events = _event_file_path(parser, args.events)

    demo_options = (args.demo_language != "python" or args.demo_data is not None
                    or args.demo_speed != "auto" or args.demo_duration is not None
                    or args.demo_step)
    if checking and (args.demo or demo_options):
        parser.error("check does not run a demonstration. Use --demo separately")
    if args.demo and args.demo_data is not None and args.demo_language != "python":
        parser.error("--demo-data selects the data, whose language they name. "
                     "Drop --demo-language")
    # Refused options stop before any configuration is read, with the
    # parser's status 2. check refuses them in its own report, which --json
    # prints as JSON.
    if not checking:
        try:
            selected_languages = _check_run_options(args)
        except ValueError as exc:
            parser.error(str(exc))
    if args.demo:
        # Before any configuration is read: the demonstration runs the
        # recorded configuration in its own workspace.
        from testuary.demo import cli as demo_cli
        from testuary._progress import demonstration_display

        with demonstration_display():
            _sys.exit(demo_cli.main(args, parser))

    if args.demo_language != "python":
        parser.error("--demo-language requires --demo. A normal run takes each target's "
                     "language from its file extension.")
    if args.demo_data is not None:
        parser.error("--demo-data requires --demo")
    if args.demo_speed != "auto" or args.demo_duration is not None or args.demo_step:
        parser.error("--demo-speed, --demo-playback-seconds and --demo-step require --demo")

    if checking:
        from testuary.presentation import TerminalPresenter

        if args.json_output:
            return _run_cli_invocation(args, None, presenter=None, checking=True)
        with TerminalPresenter(args.display, no_color=args.no_color) as presenter:
            return _run_cli_invocation(args, None, presenter=presenter, checking=True)

    from contextlib import nullcontext

    from testuary.presentation import EventFile, TerminalPresenter

    events_path = vars(args).pop("events", None)
    with TerminalPresenter(args.display, no_color=args.no_color) as presenter, \
            (EventFile(events_path, clean=getattr(presenter, "clean", None))
             if events_path else nullcontext()) as events:
        args._diagnostic_writer = presenter.write_diagnostic
        return _run_cli_invocation(args, selected_languages, presenter=presenter,
                                   event_sink=_observe(presenter, events), events=events)


if __name__ == "__main__":
    # The pipeline's return is the exit status, as the installed
    # ``testuary`` script uses it: 1 when a target failed to execute.
    raise SystemExit(main())
