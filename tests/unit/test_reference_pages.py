"""The generated reference pages match their sources.

docs/reference/cli.md is rendered from the command-line parsers and
docs/reference/settings.md from the packaged config.ini and its comments, by
tools/reference_pages.py. After changing an option, a help text, a default or
a setting, rewrite both pages with

    python tools/reference_pages.py
"""
from __future__ import annotations

import argparse
import importlib.util
import re
from pathlib import Path
from source_tree import requires_source_tree

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("reference_pages", ROOT / "tools" / "reference_pages.py")
reference_pages = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reference_pages)

_REWRITE = "out of date: rewrite with python tools/reference_pages.py"


def test_new_language_reference_parser_is_langadapters():
    parser = reference_pages.new_language_parser()
    assert parser.prog == "testuary new-language"
    assert "--file-extension" in parser.format_help()


def test_the_cli_reference_matches_the_parsers():
    page = reference_pages.CLI_PAGE
    assert page.read_text(encoding="utf-8") == reference_pages.render_cli(), f"docs/reference/cli.md {_REWRITE}"


@requires_source_tree
def test_the_settings_reference_matches_the_packaged_defaults():
    page = reference_pages.SETTINGS_PAGE
    assert page.read_text(encoding="utf-8") == reference_pages.render_settings(), (
        f"docs/reference/settings.md {_REWRITE}")


def test_the_guides_state_the_exit_statuses_and_cleanup_time_of_the_code():
    for guide in reference_pages.GUIDES:
        text = guide.read_text(encoding="utf-8")
        assert text == reference_pages.render_regions(text), (
            f"{guide.relative_to(ROOT)} {_REWRITE}")
    regions = "".join(guide.read_text(encoding="utf-8") for guide in reference_pages.GUIDES)
    for command in ("run", "check", "demo"):
        assert f"<!-- exit-statuses: {command} -->" in regions
    assert "<!-- cleanup-seconds -->120<!-- /cleanup-seconds -->" in regions


def test_every_command_and_its_options_are_on_the_cli_page():
    text = reference_pages.render_cli()
    for name, parser in reference_pages._commands():
        assert f"## testuary {name}" in text
        for action in parser._actions:
            if action.help == argparse.SUPPRESS:
                continue  # hidden aliases stay off the page
            for option in action.option_strings:
                if option not in ("-h", "--help"):
                    assert f"`{option}`" in text, (name, option)
    # A stated default comes from the parser, not from the page.
    assert "`--display` `auto` | `compact` | `detailed` | `plain` | `debug`" in text
    assert "Default: `auto`." in text


def _named_by_section(page: str) -> dict:
    """The keys each section of the settings page names in its list, by
    the heads of its items: `key`, default ..., `a`, `b`, no default, `key`,
    refused by a run, and every other key of another section apart from
    some."""
    sections = dict(re.findall(r"^## \[(\w+)\]\n(.*?)(?=^## \[|\Z)", page, re.M | re.S))
    named = {}
    for name, text in sections.items():
        keys = set()
        for head in re.findall(r"^- (?!Every )(.*?), (?:default|no default|refused)", text, re.M):
            keys |= set(re.findall(r"`(\w+)`", head))
        named[name] = keys
    for name, text in sections.items():
        for parent, refused in re.findall(
                r"^- Every (?:other )?key of \[`\[(\w+)\]`\]\(#\w+\)(.*?), no default$", text, re.M):
            named[name] |= named[parent] - set(re.findall(r"`(\w+)`", refused))
    return named


@requires_source_tree
def test_the_settings_page_names_every_key_each_section_accepts():
    from testuary.languages import declared_settings
    from testuary.settings import _bundled_data_path, _defaults, _read_ini_file, _schema

    accepted = _schema(_defaults())
    # The page describes the built-in languages, whatever else is installed.
    accepted["project"] = {*_read_ini_file(_bundled_data_path("config.ini"))["project"],
                           *(setting.name for language in ("python", "java", "c")
                             for setting in declared_settings(language))}
    page = reference_pages.render_settings()
    named = _named_by_section(page)
    for section, keys in accepted.items():
        assert named[section] == set(keys), (section, sorted(set(keys) ^ named[section]))
    assert "- `api_url`, no default" in page
    assert re.search(r"^- `c_tests`, default empty, declared by `c`$", page, re.M)
    assert re.search(r"^- `seed`, refused by a run$", page, re.M)


def test_the_models_page_names_every_field_a_catalogue_accepts():
    from testuary import models

    page = (ROOT / "docs" / "models-and-prompts.md").read_text(encoding="utf-8")
    section = re.search(r"^### models\.ini fields\n(.*?)(?=^#)", page, re.M | re.S)
    assert section, "docs/models-and-prompts.md has no models.ini fields section"
    rows = re.findall(r"^\|(.*)\|$", section.group(1), re.M)
    named = {name for row in rows if "[limits]" not in row
             for name in re.findall(r"`(\w+)`", row)}
    accepted = {*models.REQUEST_VALUES, "protocol", "reasoning_dialect", "reasoning_template",
                "chat_model", "price_input_cache_hit", "price_input_cache_miss", "price_output",
                "provider", "model", "api_url", "deployment", "api_key", "api_key_env"}
    assert named == accepted
    for name in accepted:
        assert models.catalogue_problems({"alias": {name: "1"}}) == (), name
    assert models.catalogue_problems({"alias": {"tool_result_chars": "1000"}}) == ()


@requires_source_tree
def test_the_settings_page_links_every_page_its_descriptions_name():
    page = reference_pages.render_settings()
    unlinked = re.sub(r"\[[^\]]*\]\([^)]*\)", "", page)
    named = re.findall(r"[\w' ]*(?:guide|explanation page)[\w' ]*", unlinked)
    assert named == [], named


def test_the_setup_demo_options_are_on_the_cli_page():
    """testuary setup --demo reads its options with a parser of its own:
    the page lists each of them under testuary setup."""
    from testuary.demo import setup_demo

    text = reference_pages.render_cli()
    section = text[text.index("### testuary setup --demo"):text.index("## testuary tutorial")]
    for option in ("--demo", "--demo-language", "--output-dir", "--demo-speed",
                   "--demo-playback-seconds", "--demo-step"):
        assert f"`{option}`" in section, option
    assert "`python` | `java`" in section
    assert {action.dest for action in setup_demo._parser()._actions} >= {"demo_language", "demo_step"}


def test_the_cli_page_names_the_page_of_testuary_web():
    assert "`testuary-web` is described in the [web interface guide](../guide/web.md#start)" in (
        " ".join(reference_pages.render_cli().split()))


def test_init_provision_and_testuary_web_state_their_exit_statuses(capsys):
    """Their help ends with their exit statuses, and the page shows init's
    and provision's with their sections."""
    import pytest
    from testuary import __main__ as cli
    from testuary import web

    text = " ".join(reference_pages.render_cli().split())
    for parser in (cli._init_parser(), cli._provision_parser()):
        assert parser.epilog.startswith("Exit status: 0 when")
        assert " ".join(parser.epilog.split()) in text
    with pytest.raises(SystemExit):
        web.main(["--help"])
    help_text = " ".join(capsys.readouterr().out.split())
    assert "Exit status: 0 when the interface is stopped" in help_text
    assert "2 when the interface cannot start or an option is refused" in help_text
