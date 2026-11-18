"""Testuary's extension-point table agrees with the code: its settings,
options and guide links all exist."""
from __future__ import annotations

import argparse
import configparser
import re
from importlib import resources
from pathlib import Path

from testuary import __main__ as cli
from testuary import tutorial as extension_points

DOCS = Path(__file__).resolve().parents[2] / "docs"


def _anchors(page: Path) -> set:
    """The headings' anchors of a page, as the documentation site makes them."""
    anchors = set()
    for line in page.read_text(encoding="utf-8").splitlines():
        heading = re.match(r"#+ (.+)", line)
        if heading:
            text = re.sub(r"[^\w\- ]", "", heading.group(1).lower()).strip()
            anchors.add(re.sub(r"[ ]+", "-", text))
    return anchors


def test_the_names_are_unique_and_every_entry_is_written_out():
    names = [point.name for point in extension_points.TABLE]
    assert len(names) == len(set(names))
    for point in extension_points.TABLE:
        assert point.title and point.what.endswith(".") and point.enables
        assert point.declared and point.default and point.guide
        assert extension_points.get(point.name) is point


def test_every_setting_is_a_packaged_setting():
    settings = configparser.ConfigParser(interpolation=None)
    settings.read_string(resources.files("testuary").joinpath("config.ini")
                         .read_text(encoding="utf-8"))
    for point in extension_points.TABLE:
        for section, key in point.settings:
            assert settings.has_option(section, key), (point.name, section, key)


def test_every_option_is_on_the_command_line():
    options = {option for action in cli._build_parser()._actions
               if not isinstance(action, argparse._HelpAction)
               for option in action.option_strings}
    for point in extension_points.TABLE:
        for option in point.options:
            assert option in options, (point.name, option)


def test_every_guide_link_names_a_heading():
    for point in extension_points.TABLE:
        page, _, anchor = point.guide.partition("#")
        assert (DOCS / page).is_file(), point.guide
        assert anchor in _anchors(DOCS / page), point.guide


def test_the_code_that_serves_an_entry_exists():
    import importlib

    for point in extension_points.TABLE:
        if point.implemented_by is not None:
            module, _, attribute = point.implemented_by.partition(":")
            assert callable(getattr(importlib.import_module(module), attribute)), point.name


def test_the_generation_record_is_named_in_one_place():
    for point in extension_points.TABLE:
        assert "run_1.json" not in point.recorded, point.name
    assert extension_points.get("prompt_profile").recorded.endswith(
        extension_points.GENERATION_RECORD)


def test_the_language_default_names_the_languages_of_every_installation():
    """Python and Java are Testuary's own, and C comes with LangAdapter."""
    default = extension_points.get("language").default
    for language in ("Python", "Java", "C"):
        assert re.search(rf"\b{language}\b", default), language
