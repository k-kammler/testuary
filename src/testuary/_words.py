"""The words Testuary's displays share: a setup report's items, their
statuses and problems, a file's outcome and the start of a model
service's failure. The terminal's presenter, the model requests and the
web interface read them here."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from testuary.readiness import ReadinessItem, RunReadiness


# Display order only, most severe first. The producer's statuses are not combined.
ATTENTION = ("refused", "failed", "ambiguous", "missing_setup", "unsupported", "unverified")


# The producer's statuses in the summary's words. Its values stay as the
# JSON report has them.
STATUS_WORDS = {"unverified": "not tried yet", "missing_setup": "missing",
                "unsupported": "not supported", "ambiguous": "more than one found",
                "refused": "refused", "failed": "failed"}


def status_word(status: str, scope: str = "inspection") -> str:
    """A status in the summary's words: a quick check finds what it reads, a
    tool check sees that it works."""
    if status == "verified":
        return "works" if scope == "verification" else "found"
    return STATUS_WORDS.get(status, status.replace("_", " "))


def phase_words(phase: str) -> str:
    """A phase's label inside a sentence, such as "coverage phase".
    Finishing, the fourth phase's name, keeps its capital."""
    from testuary.presentation.view import phase_label

    label = phase_label(phase)
    return label if phase == "finishing" else label[:1].lower() + label[1:]


def key_parts(key: str) -> tuple[str, str, str]:
    """Display group, item kind and name taken from the producer's dotted key.

    The key is only a label: no status, requirement or next action is derived
    from it. Unrecognized shapes keep the whole key as their name.
    """
    parts = key.split(".")
    if parts[0] == "native" and len(parts) >= 4:
        return parts[1], parts[2], ".".join(parts[3:])
    if parts[0] == "native" and len(parts) == 3:
        return parts[1], "", parts[2]
    if len(parts) >= 2:
        return parts[0], "", ".".join(parts[1:])
    return key, "", ""


# Plain names for the producer's keys. A name is display text only: statuses,
# requirements and actions always come from the producer.
KEYS = {
    "configuration": "settings",
    "configuration.credentials": "the API key",
    "source.targets": "whether the targets belong to the project",
    "output.admission": "the run directory",
    "output.original_separation": "that the run directory is kept apart from the project",
    "output.saved_artifacts": "the results of an earlier run to reuse",
    "native.execution": "the tools each language needs",
}


ITEMS = {"runtime": "runtime", "jvm": "java", "compiler": "javac",
         "native_execution": "running your project's tests and measurements",
         "build_descriptor": "the build setup",
         "build_resolution": "build of your project",
         "test_compilation": "compilation of your project's tests",
         "test_discovery": "discovery of your project's tests",
         "test_collection": "collection of your project's tests",
         "project_origin": "import of your project in your tests"}


OPERATIONS = {"tests.run_suite": "running generated tests",
              "sources.prune_tests": "removing single tests from a test file",
              "sources.test_metrics": "counting the tests and assertions of a test file",
              "mutations.operators": "listing the mutation tool's operators",
              "mutations.open_session": "running the mutation tool"}


PROBLEMS = ("refused", "failed", "ambiguous", "missing_setup", "unsupported")


def language_name(key: str) -> str:
    parts = key.split(".")
    return display_name(parts[1]) if parts[0] == "native" and len(parts) >= 3 else ""


def display_name(language: str) -> str:
    """A language's name as its adapter declares it (``display_name``), else
    its key with a capital."""
    try:
        from langadapter import registration

        declared = getattr(registration(language).factory, "display_name", None)
    except Exception:
        declared = None
    return declared if isinstance(declared, str) and declared else language[:1].upper() + language[1:]


# The words a name's short forms stand for when a display writes the name in words.
SHORT_FORMS = {"dir": "directory", "dirs": "directories"}


def name_words(name: str) -> str:
    """A name in words, its short forms written out: ``notes_dir`` is notes
    directory."""
    return re.sub(r"\w+", lambda word: SHORT_FORMS.get(word.group(0), word.group(0)),
                  name.replace("_", " "))


def plain_name(item: ReadinessItem) -> str:
    """What the item is about, in words. An unknown key keeps a readable form."""
    key = item.key
    if key in KEYS:
        return KEYS[key]
    group, kind, name = key_parts(key)
    if group == "configuration" and kind == "" and "." in name:
        setting, _, phase = name.rpartition(".")
        if setting == "model":
            return f"model settings for the {phase_words(phase)}"
        if setting == "max_requests_per_phase":
            return f"request limit of the {phase_words(phase)}"
        if setting in ("extensions", "target", "unchecked"):
            return {"extensions": "extension", "unchecked": "setting"}.get(setting, setting)
    if key.startswith("native.") and kind == "operation":
        # An operation without a name of its own reads its key in words, so that no two
        # operations share a name.
        return OPERATIONS.get(name, name_words(name).replace(".", " "))
    if key.startswith("native.") and kind == "" and name == "assessment":
        if item.tried_by == "run":
            # The language has no check of its own.
            return "no check of its own (the run tries its operations)"
        return f"the {language_name(key)} check"
    if key.startswith("native.") and kind == "" and name == "runtime_selection":
        return f"the {language_name(key)} environment for the project"
    if key.startswith("native.") and kind == "item":
        text = ITEMS.get(name, name_words(name))
        if name == "build_descriptor" and item.selected:
            text += f" ({item.selected})"
        return text
    # Unknown keys: the item's own name in words, never the internal namespace.
    return name_words(name or group).replace(".", " ")


def is_problem(item: ReadinessItem) -> bool:
    """Required items the producer reports as not working.

    An unverified item is not tried yet, even when it carries an action.
    """
    return item.required and item.status in PROBLEMS


def listed_problems(items: list[ReadinessItem]) -> list[ReadinessItem]:
    """Problems to list. A language's overall assessment only when it stands alone.

    The assessment summarizes that language's items, so listing it beside them
    would repeat each problem in other words. It stays in the detailed view.
    """
    problems = [item for item in items if is_problem(item)]
    languages = {language_name(item.key) for item in problems
                 if key_parts(item.key)[1] and language_name(item.key)}
    return [item for item in problems if not (
        key_parts(item.key)[1:] == ("", "assessment") and language_name(item.key) in languages)]


def blocking_reason(report: RunReadiness) -> str | None:
    """The reason of the first problem the summary lists, the most severe
    first, or None without one."""
    return next((reason for _, reason, _ in problem_groups(listed_problems(list(report.items)))
                 if reason), None)


def problem_groups(items: list[ReadinessItem]) -> list[tuple[list[ReadinessItem], str, str | None]]:
    groups: dict[tuple, list[ReadinessItem]] = {}
    for status in ATTENTION:
        for item in items:
            if item.status == status:
                groups.setdefault((item.reason, item.action), []).append(item)
    return [(members, reason, action) for (reason, action), members in groups.items()]


#: Each outcome in words, after a file's path. The outcome values are the
#: record's and stay.
OUTCOME_WORDS = {"complete": "a suite that meets the minimums",
                 "no_complete_suite": "no suite that meets the minimums",
                 "failed": "failed to execute", "cancelled": "stopped before finishing",
                 "already_complete": "a suite that meets the minimums, recorded earlier, not run again",
                 "stopped": "ended after the requested phase"}


#: How the sentence of a request given up after its retries begins
#: (model_requests.exhausted_text). The web interface marks such a cause as
#: the model service's.
SERVICE_FAILURE_START = "The model service"
