"""``testuary tutorial``: Testuary's extension points, step by step.

The tutorial writes a small project into a new or empty directory and applies
Testuary's extension points to it one at a time: a model alias, a prompt
profile and an extension's hints. Each step explains what it adds and why,
applies it through Testuary's own code, as ``testuary setup`` or a settings
change would, shows what it adds and checks it with ``testuary check``. It
ends with the playback of a run Testuary recorded with the same model,
profile and hints. Nothing contacts a model.

The tutorial's directory is Testuary's working directory and user directory for
every command a step runs. It holds the settings files the steps write, so
the learner's own settings stay as they are.
"""
from __future__ import annotations

import sys
from importlib import resources
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple
from langadapter.tutorial import CheckResult, Step, Tutorial
from langadapter.tutorial import main as _play
from dataclasses import dataclass


PROG = "testuary tutorial"
TITLE = "Testuary tutorial: extend Testuary"
DESCRIPTION = ("Learn Testuary's extension points step by step: a model alias, a "
               "prompt profile and an extension's hints, each applied, shown and "
               "checked, without contacting a model.")

#: The alias the tutorial sets up, and the environment variable that holds
#: its key. The checks give the variable a placeholder, which is sent nowhere,
#: and name the tutorial's directory, where they run, as Testuary's user directory.
ALIAS = "tutorial_model"
KEY_VARIABLE = "TESTUARY_OLLAMA_API_KEY"
KEY_SETTING = {KEY_VARIABLE: "tutorial-placeholder", "TESTUARY_HOME": "."}

#: The answers the model step gives to the questions of ``testuary setup
#: model``, by question key. A question not named here takes its default.
MODEL_ANSWERS = {
    "model.provider": "ollama-cloud",
    "model.key_storage": "environment",
    "model.key_variable": KEY_VARIABLE,
    "model.name": "deepseek-v4.1-flash:1017",
    "model.alias": ALIAS,
    "model.confirm": "yes",
}

PROFILE = "language-profile.json"

#: The demonstration data of the run the tutorial ends with, recorded with
#: the alias's model, the profile and the hints, and the directory the playback
#: writes into.
DEMO_DATA = "tutorial"
RUN_DIR = "testuary-run"

#: The extension the hints step writes: one Python file beside the tutorial
#: project's settings file, which names it in ``[main] extensions``.
HINTS_EXTENSION_FILE = "tutorial_hints.py"
HINTS_EXTENSION_PATH = f".testuary/{HINTS_EXTENSION_FILE}"

_HINTS_EXTENSION = '''"""Hints for Testuary's Python prompts, from the Testuary tutorial."""

testuary_hints = {
    "coverage": {
        "per_language": {
            "python": "Prefer plain assert statements over assertion helpers."
        }
    }
}
'''

_PROJECT = """def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0
"""

_SAMPLE_TEST = """from calc import sign


def test_sign():
    assert sign(3) == 1
    assert sign(-3) == -1
    assert sign(0) == 0
"""


def _example(name: str) -> str:
    """A file of the examples ``testuary init --examples`` writes."""
    path = resources.files("testuary").joinpath("_templates", "examples", *name.split("/"))
    return path.read_text(encoding="utf-8")


class _Offline:
    """The setup's view of this machine for the tutorial: no local service,
    no key in the environment and no model list, so the setup contacts
    nothing."""

    def __init__(self) -> None:
        from testuary import setup

        self._probe = setup.Probe()

    def local_ollama(self) -> bool:
        return False

    def environment(self, name: str) -> str:
        return ""

    def model_list(self, provider, address: str, key: str, **options) -> None:
        return None

    def __getattr__(self, name: str):
        return getattr(self._probe, name)


def _answered(answers: Dict[str, str], transcript: List[str]) -> Callable:
    """The setup's ``ask``: each question's prepared answer, written to
    ``transcript`` as the question and its answer."""
    def ask(question) -> str:
        answer = answers.get(question.key, question.default)
        transcript.append(f"{question.prompt}: {answer}")
        return answer
    return ask


def _apply_alias(tutorial_dir: Path) -> CheckResult:
    """Set up the tutorial's alias with Testuary's setup and show the
    questions, the answers and what the setup wrote."""
    from testuary import setup

    transcript: List[str] = []
    said: List[str] = []
    try:
        result = setup.run_setup(_answered(MODEL_ANSWERS, transcript), _Offline(),
                                 said.append, tutorial_dir, part="model", home=tutorial_dir)
    except setup.SetupError as exc:
        return CheckResult(False, (f"the setup stopped: {exc}",))
    if result.alias != ALIAS:
        return CheckResult(False, ("the setup wrote no alias",) + tuple(transcript))
    written = [line for text in said for line in text.splitlines()
               if line.startswith("[") or " = " in line]
    return CheckResult(True, ("The setup's questions and the prepared answers:",
                              *transcript, "It wrote in models.ini:", *written))


def _apply_profile(tutorial_dir: Path) -> CheckResult:
    """Select the tutorial's prompt profile in the project's settings file
    with Testuary's settings update, which keeps the file's other lines, and
    show the setting. The tutorial's directory is the project of every command
    a step runs, so it gets its own Testuary directory first: a project above
    it is never changed."""
    from testuary import settings

    (tutorial_dir / ".testuary").mkdir(exist_ok=True)
    path = settings.file_locations(tutorial_dir).project_settings
    try:
        settings.update_file(path, "main", {"prompt_profile": PROFILE})
    except settings.ConfigurationError as exc:
        return CheckResult(False, tuple(str(exc).splitlines()))
    return CheckResult(True, (f"It wrote in {path.relative_to(tutorial_dir.resolve()).as_posix()}:", "[main]",
                              f"prompt_profile = {PROFILE}"))


def _apply_hints(tutorial_dir: Path) -> CheckResult:
    """Name the tutorial's extension in the project's settings file with
    Testuary's settings update, find its hints through the setting as a
    command does, validated, and show what they add to the coverage phase's
    request for Python. The tutorial's directory is the user directory, so
    the learner's own extensions stay out."""
    from testuary import hints, settings

    path = settings.file_locations(tutorial_dir).project_settings
    try:
        settings.update_file(path, "main", {"extensions": HINTS_EXTENSION_FILE})
        found = hints.extension_hints(settings.resolve_settings(base_dir=tutorial_dir,
                                                                home=tutorial_dir))
    except settings.ConfigurationError as exc:
        return CheckResult(False, tuple(str(exc).splitlines()))
    ours = [item for item in found if item.item == HINTS_EXTENSION_FILE]
    if not ours:
        return CheckResult(False, (f"the extension {HINTS_EXTENSION_FILE} holds no testuary_hints",))
    shown = path.relative_to(tutorial_dir.resolve()).as_posix()
    return CheckResult(True, (f"It wrote in {shown}:", "[main]", f"extensions = {HINTS_EXTENSION_FILE}",
                              f"Testuary loads {HINTS_EXTENSION_FILE} through the setting, and its hints "
                              "are valid.", "The coverage phase's request for Python gets:",
                              hints.request_text(ours[0], "python", "coverage")))


def _show_recorded_run(tutorial_dir: Path) -> CheckResult:
    """What the recorded run's playback used and how it ended, from the
    record it writes."""
    import json

    record = tutorial_dir / RUN_DIR / "DEMO.json"
    if not record.is_file():
        return CheckResult(False, (f"no playback record in {RUN_DIR}: the step before plays "
                                   "the recorded run",))
    data = json.loads(record.read_text(encoding="utf-8"))
    rows = data.get("configuration", {})
    result = data.get("result", {})
    lines = tuple(f"{name}: {rows[name]}" for name in ("model", "target", "prompt profile",
                                                       "hints file") if name in rows)
    if result.get("outcome") != "completed":
        return CheckResult(False, (*lines, f"The playback ended {result.get('outcome')}."))
    return CheckResult(True, (*lines, f"The playback completed with all {result.get('answers')} "
                                      "recorded answers."))


def _check_command() -> tuple:
    # The Testuary that runs the tutorial checks its files.
    return (sys.executable, "-m", "testuary", "check", "--model", ALIAS,
            "--source-dir", "project", "--python-executable", sys.executable,
            "--source-files", "calc.py")


def _point(name: str):

    return get(name)


def _introduce(name: str) -> tuple:
    """The first paragraphs of a step, from Testuary's extension-point table:
    what the extension point is, what it makes possible, where it is
    declared and its default, then what it means for runs you compare."""
    point = _point(name)
    paragraphs = [f"{point.what} It makes possible {point.enables}.",
                  f"Where: {point.declared}. Default: {point.default}."]
    if point.model_input:
        note = "It changes the text sent to the model, so keep it the same across runs you compare."
        if point.recorded:
            note += f" A run records {point.recorded}."
        paragraphs.append(note)
    return tuple(paragraphs)


def tutorial() -> Tutorial:
    """The tutorial's steps."""
    check = _check_command()
    return Tutorial(
        TITLE,
        intro=(
            "This tutorial applies Testuary's extension points to a small project, one "
            "at a time: a model alias, a prompt profile and an extension's hints. Each "
            "step explains what it adds, applies it, shows what it adds and checks it "
            "with testuary check.",
            "The settings files go into this tutorial's directory, so your own settings "
            "stay as they are. Nothing contacts a model: the checks give the alias's "
            "key variable a placeholder only while they run.",
        ),
        steps=(
            Step(
                "A small project",
                ("The checks below look at this project: one source file and one "
                 "test file.",),
                {"project/calc.py": _PROJECT, "project/test_calc.py": _SAMPLE_TEST},
                show=("project/calc.py",),
            ),
            Step(
                _point("model_alias").title,
                (*_introduce("model_alias"),
                 "Why: a run, its settings and its records name the alias, so you can "
                 "change the service in one place. The key stays in the environment, "
                 "never in a command or a record.",
                 "This step answers the questions of testuary setup model with prepared "
                 f"answers: Ollama Cloud, the key in {KEY_VARIABLE} and an alias named "
                 f"{ALIAS}. For your own service, run testuary setup model and answer "
                 "them yourself.",
                 "The check then runs testuary check with the alias. It reads the alias "
                 "and its key variable but contacts no model."),
                check=_apply_alias, command=check, env=KEY_SETTING, optional=True,
            ),
            Step(
                _point("prompt_profile").title,
                (*_introduce("prompt_profile"),
                 "This profile adds a sentence to the coverage phase's instructions. For "
                 "Python and for Java it gives two sentences in its place, the second on "
                 "how that language's tests expect an exception. The step writes it and "
                 "selects it in Testuary's settings. The check validates the profile: an unknown part "
                 "or phase stops it with the place named."),
                {PROFILE: _example(PROFILE)},
                show=(PROFILE,),
                check=_apply_profile, command=check, env=KEY_SETTING, optional=True,
            ),
            Step(
                _point("extension_hints").title,
                (*_introduce("extension_hints"),
                 "This extension adds one hint to the coverage phase's request for Python. "
                 "A run loads the extensions its settings name when it starts and validates "
                 "their hints: a problem stops it with the item named. In each request the "
                 "extensions' hints come first, then your own hints file.",
                 f"The step writes the file {HINTS_EXTENSION_PATH} and names it in the project's "
                 "settings file. In your own project, name such a file in the project's "
                 "settings file, or in the config.ini of your user directory for every "
                 "project. Nothing is installed."),
                {HINTS_EXTENSION_PATH: _HINTS_EXTENSION},
                show=(HINTS_EXTENSION_PATH,),
                check=_apply_hints, command=check, env=KEY_SETTING, optional=True,
            ),
            Step(
                "A recorded run with these extensions",
                ("Testuary recorded one real run of this release with the model the alias "
                 "names, this prompt profile and these hints, on tenacity/stop.py of the "
                 "tenacity project. This step plays it back. Every request gets its "
                 "answer from the recording: nothing contacts a model, and no test, "
                 "coverage or mutation tool runs.",
                 "Why it shows the extensions at work: the recorded answers fit only the "
                 "requests Testuary built with this profile and these hints. A request built "
                 "otherwise has no answer, and the playback stops. The recording carries the "
                 "profile and the hints, as a hints file, so the playback builds the same "
                 "requests.",
                 f"The playback writes into {RUN_DIR}. Its records name what it used and "
                 "how the run ended."),
                command=(sys.executable, "-m", "testuary", "--demo", "--demo-data", DEMO_DATA,
                         "--output-dir", RUN_DIR),
                optional=True,
            ),
            Step(
                "What the recorded run used",
                ("The playback's record, DEMO.json, names the model, the target, the prompt "
                 "profile and the hints file the run used, and how the playback ended. A "
                 "playback that completes with every recorded answer sent the requests the "
                 "recording sent.",),
                check=_show_recorded_run, optional=True,
            ),
        ),
        closing=(
            "Done. This directory holds the project, the alias in models.ini, the prompt "
            "profile and the extension with its hints that .testuary/project.ini names, "
            f"and, in {RUN_DIR}, the playback of the run recorded with them. "
            "Testuary's documentation, under Extending, describes each format in full.",
        ),
    )


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the tutorial from the command line."""
    return _play(argv, tutorial(), prog=PROG, description=DESCRIPTION)


#: A run's generation record, one per run of a phase, as the extension
#: points name it. evidence.phase_origin_path builds its path.
GENERATION_RECORD = "r<k>/archive/<phase>/generation/record.json"


@dataclass(frozen=True)
class ExtensionPoint:
    """One way to extend Testuary.

    ``settings`` are ``(section, key)`` pairs of the packaged settings and
    ``options`` command-line options. ``model_input`` says whether it changes the
    text sent to the model, so runs that are compared keep it the same.
    ``recorded`` says what a run records of it, empty when nothing beyond the
    settings. ``guide`` is the documentation page and heading.
    ``implemented_by`` names the code that serves it, as
    ``module:attribute``, where no setting or option does."""

    name: str
    title: str
    what: str
    enables: str
    declared: str
    default: str
    guide: str
    settings: Tuple[Tuple[str, str], ...] = ()
    options: Tuple[str, ...] = ()
    model_input: bool = False
    recorded: str = ""
    implemented_by: Optional[str] = None


TABLE: Tuple[ExtensionPoint, ...] = (
    ExtensionPoint(
        "model_alias", "A model alias",
        what=("An alias names a model service for Testuary: its provider, the model "
              "and the environment variable that holds the key."),
        enables=("any service testuary setup offers, and any other OpenAI-compatible "
                 "one, with no code"),
        declared=("a section of models.ini, which testuary setup model writes. A run "
                  "selects it with --model or [api] model"),
        default="none, so a run names its alias",
        guide="models-and-prompts.md#modelsini",
        settings=(("api", "model"),), options=("--model",)),
    ExtensionPoint(
        "model_provider", "A client factory",
        what=("A client factory is a Python function that returns the chat model "
              "Testuary sends its requests through, a LangChain chat model that "
              "streams and can bind tools. An alias names it with chat_model and "
              "protocol = added."),
        enables="a model service that is not OpenAI-compatible",
        declared=("the alias fields chat_model = ITEM:ATTRIBUTE, where ITEM is a file such as "
                  "my_provider.py or a module, and protocol = added"),
        default="none, since an OpenAI-compatible service needs only an alias",
        guide="extending/model-provider.md#client-factories"),
    ExtensionPoint(
        "prompt_profile", "A prompt profile",
        what=("A prompt profile changes the text Testuary sends to the model, phase by "
              "phase: it can append to, replace or remove a part, for every language "
              "or for one."),
        enables="your own instructions to the model without changing Testuary",
        declared="[main] prompt_profile, or --prompt-profile for one run",
        default="the bundled prompts",
        guide="extending/prompt-profile.md#selecting-a-profile",
        settings=(("main", "prompt_profile"),), options=("--prompt-profile",),
        model_input=True,
        recorded=("the profile's path, name, digest, a copy and the parts it "
                  f"changed, under prompt_profile in {GENERATION_RECORD}")),
    ExtensionPoint(
        "hints", "Hints",
        what=("Hints are short advice added to a phase's request, for every language, "
              "for one, or for one hint profile."),
        enables="advice for the model that fits your project",
        declared=("[main] hints_file, or --hints-file for one run. [project] "
                  "hint_profile selects a profile's cells"),
        default="hints.json in the project directory, or no hints when it is missing",
        guide="extending/hints.md#selecting-a-hints-file",
        settings=(("main", "hints_file"), ("project", "hint_profile")),
        options=("--hints-file",), model_input=True,
        recorded="the file's path, digest and a copy, under hints in the generation origin"),
    ExtensionPoint(
        "extension_hints", "Hints from an extension",
        what=("An extension's top-level name testuary_hints holds hints in the hints "
              "file's format."),
        enables="advice for the model that comes with a language or a tool, for every project",
        declared=("[main] extensions, which names the extension's file or module, such as "
                  "tutorial_hints.py"),
        default="no extension hints",
        guide="extending/hints.md#selecting-a-hints-file",
        settings=(("main", "extensions"),), model_input=True,
        recorded=("each extension's item, file and digest under extensions, and its hints' "
                  "digest and a copy under extension_hints, in the generation origin")),
    ExtensionPoint(
        "language", "A language",
        what=("An extension's top-level name langadapter_language names a language's "
              "adapter, which LangAdapter registers. Testuary runs any registered language "
              "through its generic backend, judged by Testuary's levels."),
        enables="test generation for a language Testuary has no code for",
        declared=("[main] extensions, which names the file LANGUAGE.py that testuary "
                  "new-language writes. A run selects the language with --languages"),
        default="Python and Java, and C on Linux",
        guide="extending/languages.md#what-testuary-needs",
        settings=(("main", "extensions"),), options=("--languages",)),
    ExtensionPoint(
        "language_settings", "A language's project settings",
        what=("A language's adapter declares the settings a project gives "
              "it, each with its purpose, its kind (a path, a program, a choice or "
              "text), a default and, for a choice, its choices. Python's and Java's "
              "own settings stay in Testuary's sections."),
        enables=("an extension language's runtime and tools set per project, such as "
                 "where its runtime is"),
        declared=("the adapter's project_settings, set in [project], usually in the "
                  "project's settings file .testuary/project.ini. testuary setup asks "
                  "them, and testuary check reports each by its kind"),
        default="each setting's declared default, where empty means the language finds it",
        guide="extending/languages.md#project-settings",
        recorded="with the run's resolved settings in the generation origin",
        implemented_by="testuary.languages:declared_settings"),
    ExtensionPoint(
        "web_extension", "A web interface extension",
        what=("An extension adds tabs, result views, settings groups and jobs to "
              "testuary-web, and can set the product name and rename Testuary's tabs, with "
              "its top-level name testuary_web, a testuary.web.WebExtension."),
        enables="another product's pages beside Testuary's, in one web interface",
        declared="[main] extensions, which names its file or module, such as notes/notes.py",
        default="Testuary's own tabs and pages",
        settings=(("main", "extensions"),),
        guide="extending/web.md#naming-the-extension"),
)


def get(name: str) -> ExtensionPoint:
    """The extension point named ``name``."""
    for point in TABLE:
        if point.name == name:
            return point
    raise KeyError(name)
