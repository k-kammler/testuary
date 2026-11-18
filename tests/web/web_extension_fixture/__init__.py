"""A web interface extension, the extension guide's worked example: a Notes
tab in the place of Run with Results moved after it, the directory type of the
notes.json its job writes, its settings with the Testuary settings files its
runs in a project read, the job kind that writes the notes, the one that
checks a notes directory and the one that plays Testuary's demonstration as a
nested run."""
from __future__ import annotations

import configparser
import json
import os
from pathlib import Path

from testuary.settings import SectionDescription, SettingDescription
from testuary.web import DirType, JobKind, NestedRun, SettingsGroup, WebExtension, WebRoute, WebTab

SUMMARY = "notes.json"
DEMO_EVENTS = "demo-events.jsonl"
COUNT = SettingDescription("notes", "count", "How many notes Collect notes writes, from 1 to 100.",
                           "number", "3")


def is_notes(notes_dir: Path) -> bool:
    return (notes_dir / SUMMARY).is_file()


def summary(notes_dir: Path) -> dict:
    return json.loads((notes_dir / SUMMARY).read_text(encoding="utf-8"))


def _settings_file() -> Path:
    return Path(os.environ["TESTUARY_HOME"]) / "notes.ini"


def reference(project_dir):
    return [SectionDescription("notes", "The notes the Collect notes job writes.", (COUNT,))]


def values(project_dir) -> dict:
    parser = configparser.ConfigParser()
    parser.read(_settings_file(), encoding="utf-8")
    return {("notes", "count"): parser.get("notes", "count", fallback=COUNT.default)}


def update(project_dir, section, changes, remove) -> None:
    count = changes.get("count", COUNT.default)
    if section != "notes" or not count.isdigit() or not 1 <= int(count) <= 100:
        raise ValueError("count: expected a whole number from 1 to 100")
    parser = configparser.ConfigParser()
    parser.read(_settings_file(), encoding="utf-8")
    parser["notes"] = {} if "count" in remove else {"count": count}
    with open(_settings_file(), "w", encoding="utf-8") as stream:
        parser.write(stream)


def testuary_settings_files(project_dir: Path) -> list:
    return [Path(".notes") / "testuary.ini"]


def command(form) -> list:
    notes_dir = form.get("dir")
    if not isinstance(notes_dir, str) or not notes_dir.strip():
        raise ValueError("Name the directory the notes go to.")
    arguments = ["--dir", notes_dir, "--count", values(None)[("notes", "count")]]
    if form.get("wait"):
        arguments += ["--wait", str(form["wait"])]
    return arguments


def check_command(form) -> list:
    return command({"dir": form.get("dir")})


def demo_command(form) -> list:
    arguments = command(form)[:2]
    if (Path(arguments[1]) / "demo").exists():
        raise ValueError("This notes directory already holds a demonstration.")
    return arguments


def demo_runs(arguments) -> list:
    return [NestedRun("play", Path(arguments[1]) / DEMO_EVENTS, "Testuary's demonstration")]


def count(notes_dir, query) -> dict:
    if notes_dir is None or not is_notes(notes_dir):
        raise ValueError("Name a notes directory with dir.")
    return {"count": summary(notes_dir)["count"]}


def extension() -> WebExtension:
    """The notes extension, which ``testuary_web`` holds."""
    return WebExtension(
        api=1, id="notes",
        tabs=(WebTab("notes", "Notes", "notes.js", position="before:run", replaces="run"),),
        tab_positions={"results": "after:notes"},
        dir_types=(DirType("notes", "Notes", is_notes, summary, "notes.js"),),
        settings_groups=(SettingsGroup("notes", "Notes", reference, values, update,
                                       testuary_settings_files),),
        routes=(WebRoute("count", count),),
        job_kinds=(JobKind("collect", "Collect notes", "web_extension_fixture.job", command),
                   JobKind("check", "Check notes", "web_extension_fixture.check", check_command),
                   JobKind("demo", "Play the demonstration", "web_extension_fixture.demo", demo_command,
                           nested_runs=demo_runs)))


testuary_web = extension()
