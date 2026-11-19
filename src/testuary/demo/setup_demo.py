"""The setup wizard's demonstration: ``testuary setup --demo``.

It writes the demonstration's project into a new directory, runs the real
setup wizard there and types the wizard's answers visibly, at the paced
speed of the playback. The answers come from this module, by question key.
A key it does not know stops the demonstration as outdated, as stale
scenario data do. The machine is looked at only where that starts no process
and opens no connection: the wizard's probe answers the rest. The model list
reads as not available, and the key's environment variable as not set.

The project the wizard wrote must name the recorded run's directory, files,
language and model alias, and that alias the recorded model at its address.
Then that run is played back as ``testuary --demo`` plays it, in a directory of
its own below the output directory.
"""
from __future__ import annotations

import argparse
import contextlib
import datetime
import importlib.util
import os
import platform
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from testuary import setup as _setup
from testuary.demo import cli as demo_cli

BANNER = (
    "SETUP DEMO, not your setup. The demo types the answers below. Nothing "
    "contacts a model or starts a tool, and files are written only in the "
    "run directory.")
# Seconds at speed 1: each typed character, and the pause after an answer.
_CHARACTER = 0.045
_ANSWERED = 0.6
# Placeholders the Java demonstration names for its JDK and Maven. They are
# created as files to point at, never run: the playback replays its answers.
_JDK = "demo-tools/jdk-21"
_MAVEN = "demo-tools/maven/bin/mvn"
_POM = """<!-- Written by testuary setup --demo for the demo's project. Never built. -->
<project xmlns="http://maven.apache.org/POM/4.0.0">
  <modelVersion>4.0.0</modelVersion>
  <groupId>example</groupId>
  <artifactId>{artifact}</artifactId>
  <version>1.0</version>
</project>
"""


def _alias_fields(manifest: Dict, models_text: str) -> Dict[str, str]:
    """The fields of the scenario's model alias in its shipped, key-free
    catalogue."""
    parser = demo_cli.settings_text(models_text)
    alias = manifest["invocation"]["model"]
    return dict(parser[alias]) if parser.has_section(alias) else {}


def _model_answers(manifest: Dict, models_text: str) -> Dict[str, str]:
    """The model part's answers for the recorded scenario's alias, an Ollama
    alias without an address: Ollama Cloud. Its key is named as an
    environment variable that the demonstration leaves unset."""
    fields = _alias_fields(manifest, models_text)
    alias = manifest["invocation"]["model"]
    if fields.get("provider") != "ollama" or fields.get("api_url"):
        raise demo_cli.DataMismatch(f"the demo's model alias {alias} is not an Ollama Cloud alias")
    return {"model.provider": "ollama-cloud", "model.name": fields["model"],
            "model.alias": alias}


def _target_choice(outdir: Path, language: str, manifest: Dict) -> str:
    """The wizard's number for the recorded target among the project's
    source files, as the wizard lists them, or "" (all) when it is the only
    one."""
    project = outdir / "workspace" / "projects" / manifest["exemplar"]["instance"]
    sources = DemoProbe().source_files(project, language)
    target = manifest["invocation"]["source_files"]
    if len(sources) <= 1 or target not in sources:
        return ""
    return str(sources.index(target) + 1)


def _answers(language: str, manifest: Dict, models_text: str,
             targets: str) -> Dict[str, str]:
    """The answer to each question key. An empty answer takes the default."""
    instance = manifest["exemplar"]["instance"]
    common = {
        "model.run": "",
        "model.key_storage": "", "model.key_variable": "", "model.confirm": "",
        "machine.run": "", "machine.mutation_workers": "", "machine.starter_requests": "",
        "machine.confirm": "",
        "project.run": "", "project.folder": f"workspace/projects/{instance}",
        "project.language": "", "targets": targets, "model": "", "project.confirm": "",
    }
    common.update(_model_answers(manifest, models_text))
    if language == "java":
        return {**common, "java.jdk": _JDK, "java.build": _MAVEN}
    # The demonstration selects the interpreter it runs in, explicitly: a
    # project without an environment of its own has no default interpreter.
    return {**common, "python.import_root": "", "python.interpreter": sys.executable}


class DemoProbe(_setup.Probe):
    """The wizard's probe without processes or connections. Files and the CPU
    count are read as they are. The environment reads as empty, as the
    demonstration sets it aside."""

    def environment(self, name: str) -> str:
        return ""

    def run(self, argv: Sequence[str], timeout: float) -> Optional[Tuple[int, str]]:
        """A script in the interpreter that runs the demonstration answers
        without starting it, by what the setup asks: which of the modules
        named after the script this process cannot import, or, naming none,
        its version. Another program does not start."""
        if len(argv) < 3 or argv[0] != sys.executable or argv[1] != "-c":
            return None
        if argv[3:]:
            missing = (name for name in argv[3:] if importlib.util.find_spec(name) is None)
            return 0, " ".join(missing) + "\n"
        return 0, platform.python_version() + "\n"

    def which(self, command: str) -> str:
        return ""

    def local_ollama(self) -> bool:
        return False

    def model_list(self, provider, address, key, *, timeout: float = 15) -> Optional[List[str]]:
        return None

    def restrict_to_user(self, path: Path) -> bool:
        return False


class Outdated(RuntimeError):
    """The wizard asked something the demonstration has no answer for, or
    wrote a project that is not the recorded run's."""


class _Typist:
    """Asks the wizard's questions on the screen and types the demo's answer,
    at ``factor`` times the speed, instantly, or one answer per Enter."""

    def __init__(self, answers: Dict[str, str], playback, write: Callable[[str], None]) -> None:
        self.answers = answers
        self.write = write
        self.factor = 0.0 if playback is None else playback[0]
        self.step = playback is not None and playback[1]
        self.keys = None
        if self.step:
            from testuary.demo import pacing

            self.keys = pacing.keys()

    def _pause(self, seconds: float) -> None:
        if self.factor and not self.step:
            time.sleep(seconds / self.factor)

    def _wait_key(self) -> None:
        # Only a key pressed while the answer waits counts, as in the
        # playback's steps.
        from testuary.demo import pacing

        pacing.wait_for_key(self.keys)

    def __call__(self, question: _setup.Question) -> str:
        if question.key not in self.answers:
            raise Outdated(f"the wizard asked {question.key!r}, which the demo cannot answer")
        answer = self.answers[question.key]
        # The question as the wizard's terminal shows it.
        lines, prompt = _setup._question_lines(question)
        for line in lines:
            self.write(line + "\n")
        self.write(prompt)
        if self.step:
            self._wait_key()
        typed = "(hidden)" if question.secret and answer else answer
        for character in typed:
            self.write(character)
            self._pause(_CHARACTER)
        self.write("\n")
        self._pause(_ANSWERED)
        return answer

    def close(self) -> None:
        if self.keys is not None:
            self.keys.close()


@contextlib.contextmanager
def _working_dir(output_dir: Path):
    """Answers name paths relative to the output directory, as a person running
    the wizard there would."""
    previous = Path.cwd()
    os.chdir(output_dir)
    try:
        yield
    finally:
        os.chdir(previous)


def _prepare(outdir: Path, language: str):
    manifest, files = demo_cli.load_data(language)
    demo_cli._write_workspace(outdir / "workspace", manifest, files)
    if language == "java":
        (outdir / _JDK).mkdir(parents=True)
        (outdir / _JDK / "release").write_text('JAVA_VERSION="21"\n', encoding="utf-8")
        (outdir / _MAVEN).parent.mkdir(parents=True)
        (outdir / _MAVEN).write_text("", encoding="utf-8")
        # The project's build file, so the wizard sees a Maven project. The
        # playback replays its answers and never builds it.
        project = outdir / "workspace" / "projects" / manifest["exemplar"]["instance"]
        (project / "pom.xml").write_text(_POM.format(artifact=manifest["exemplar"]["instance"]),
                                         encoding="utf-8")
    (outdir / "DEMO_NOTICE.txt").write_text(BANNER + "\n", encoding="utf-8")
    return manifest, files["models.ini"].decode("utf-8")


def _verify(outdir: Path, result: _setup.SetupResult, manifest: Dict) -> None:
    """The written project and alias are the recorded run's, or Outdated."""
    import configparser

    def read(name: str) -> configparser.ConfigParser:
        # A file the wizard did not write reads as empty.
        path = outdir / name
        return demo_cli.settings_text(path.read_text(encoding="utf-8") if path.is_file() else "")

    invocation = manifest["invocation"]
    project_dir = (outdir / "workspace" / "projects" / manifest["exemplar"]["instance"]).resolve()
    settings = read(str(project_dir / ".testuary" / "project.ini"))
    if not result.project or not settings.has_section("project"):
        raise Outdated("the wizard wrote no project")
    section = settings["project"]
    differences = []
    if Path(result.project) != project_dir:
        differences.append(f"project directory {result.project}")
    if section.get("source_files", "") != invocation["source_files"]:
        differences.append(f"source_files {section.get('source_files')}")
    if settings.get("api", "model", fallback="") != invocation["model"]:
        differences.append(f"model {settings.get('api', 'model', fallback='')}")
    if section.get("language", "") != manifest.get("language", "python"):
        differences.append(f"language {section.get('language')}")
    if tuple(result.run) != ("testuary",):
        differences.append(f"run command {' '.join(result.run)}")
    # The alias must name the recorded model at the recorded address, which
    # is what a run contacts: the alias's own, else its provider's in the
    # catalogue or the configuration, else the packaged one. Where the
    # address is written may differ between the two.
    alias = invocation["model"]

    def contacted(dir_prefix: str) -> tuple:
        catalogue, configuration = read(dir_prefix + "models.ini"), read(dir_prefix + "config.ini")
        provider = catalogue.get(alias, "provider", fallback="")
        address = (catalogue.get(alias, "api_url", fallback="")
                   or catalogue.get(provider, "api_url", fallback="")
                   or configuration.get(provider, "api_url", fallback="")
                   or _setup._config_address(provider))
        return catalogue.get(alias, "model", fallback=""), address.rstrip("/")

    written, recorded = contacted(""), contacted("workspace/")
    if written != recorded:
        differences.append(f"[{alias}] model and address {written} instead of {recorded}")
    if differences:
        raise Outdated("the setup differs from the recorded run: " + ". ".join(differences))


def _parser() -> argparse.ArgumentParser:
    from testuary.__main__ import _build_parser

    # The playback options as the command line declares them, with its help.
    playback = {action.dest: action for action in _build_parser()._actions}
    parser = argparse.ArgumentParser(
        prog="testuary setup --demo", allow_abbrev=False,
        description="Show the setup wizard on the demo's project, then play back its run.")
    parser.add_argument("--demo", action="store_true", required=True,
                        help="Show the setup wizard's demonstration.")
    parser.add_argument("--demo-language", choices=("python", "java"), default="python",
                        help="Language of the setup's demonstration: python (the default) or "
                             "java. Other demonstrations, such as toy, and --demo-data are not "
                             "accepted here.")
    parser.add_argument("--output-dir", help=playback["output_dir"].help)
    parser.add_argument("--demo-speed", default="auto", metavar=playback["demo_speed"].metavar,
                        help=playback["demo_speed"].help)
    parser.add_argument("--demo-playback-seconds", dest="demo_duration",
                        metavar=playback["demo_duration"].metavar,
                        help=playback["demo_duration"].help)
    parser.add_argument("--demo-step", action="store_true", help=playback["demo_step"].help)
    return parser


def main(argv: Sequence[str]) -> int:
    """``testuary setup --demo [--demo-language java] [--output-dir D]``,
    with the playback options of ``testuary --demo``."""
    args = _parser().parse_args(list(argv))
    language = args.demo_language
    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    outdir = Path(args.output_dir or f"testuary-setup-demo-{language}-{stamp}").resolve()
    if outdir.exists() and (not outdir.is_dir() or any(outdir.iterdir())):
        print(f"ERROR: the run directory must be a new or empty directory: {outdir}", file=sys.stderr)
        return demo_cli.EXIT_USAGE
    try:
        playback = demo_cli._playback(args)
    except ValueError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return demo_cli.EXIT_USAGE
    outdir.mkdir(parents=True, exist_ok=True)

    def write(text: str) -> None:
        sys.stdout.write(text)
        sys.stdout.flush()

    bar = "=" * 76
    print(f"{bar}\n{BANNER}\nRun directory: {outdir}\n{bar}", flush=True)
    typist = None
    try:
        manifest, models_text = _prepare(outdir, language)
        typist = _Typist(_answers(language, manifest, models_text,
                                  _target_choice(outdir, language, manifest)), playback, write)
        with _working_dir(outdir), demo_cli._environment_aside():
            # The output directory is the setup's user directory: the user's own
            # models.ini and config.ini are never written, and their
            # environment variables are set aside, as the probe says.
            result = _setup.run_setup(typist, DemoProbe(), lambda text: write(text + "\n"), outdir,
                                      home=outdir)
        _verify(outdir, result, manifest)
    except (Outdated, demo_cli.DataMismatch) as exc:
        print(f"Setup demo stopped: {exc}. Its answers no longer fit this package version.",
              file=sys.stderr)
        return demo_cli.EXIT_STALE
    except _setup.SetupError as exc:
        print(f"Setup demo stopped: {exc}", file=sys.stderr)
        return demo_cli.EXIT_STALE
    finally:
        if typist is not None:
            typist.close()
    print(f"The quick check (testuary check in {result.project}) is left out here: it would "
          "look for the project's tools. The run the setup prepared is played back now.", flush=True)
    return _playback_run(outdir / "demo", language, args)


def _playback_run(output: Path, language: str, args: argparse.Namespace) -> int:
    from testuary import __main__ as entry
    from testuary._progress import demonstration_display

    command = ["--demo", "--output-dir", str(output), "--demo-language", language,
               "--demo-speed", args.demo_speed]
    if args.demo_duration is not None:
        command += ["--demo-playback-seconds", args.demo_duration]
    if args.demo_step:
        command.append("--demo-step")
    parser = entry._build_parser()
    with demonstration_display():
        return demo_cli.main(parser.parse_args(command), parser)
