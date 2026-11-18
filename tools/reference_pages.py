"""Write the generated reference pages of the documentation site.

docs/reference/cli.md is rendered from Testuary's command-line parsers, with
testuary setup --demo's options from that command's own parser and init's
and provision's exit statuses from their help, and it names the web
interface guide as the page of testuary-web.
docs/reference/settings.md is rendered from testuary.settings.reference(), the
description of every setting the packaged config.ini and its comments give,
with the declared settings of the built-in languages Python, Java and C, and
from the settings schema, whose other keys it lists with the page that
describes them.
The exit statuses and the cleanup time in four guides come from
testuary.exits. Run this after changing an option, a help text, a default,
a setting or an exit status:

    python tools/reference_pages.py

tests/unit/test_reference_pages.py fails while a committed page differs from
what this script writes.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CLI_PAGE = ROOT / "docs" / "reference" / "cli.md"
SETTINGS_PAGE = ROOT / "docs" / "reference" / "settings.md"


def _default(action: argparse.Action) -> str:
    """The default worth stating: not for switches, empty values or no value."""
    value = action.default
    if action.nargs == 0 or value in (None, False, "", [], ()) or value is argparse.SUPPRESS:
        return ""
    return f" Default: `{value}`."


def _option_lines(parser: argparse.ArgumentParser) -> list[str]:
    lines = []
    for action in parser._actions:
        if isinstance(action, argparse._HelpAction) or action.help == argparse.SUPPRESS:
            continue
        if action.option_strings:
            names = ", ".join(f"`{name}`" for name in action.option_strings)
            if action.choices:
                names += " " + " | ".join(f"`{choice}`" for choice in action.choices)
            elif action.nargs != 0:
                names += f" `{action.metavar or action.dest.upper()}`"
        else:
            names = f"`{action.metavar or action.dest}`"
            if action.choices:
                names += " " + " | ".join(f"`{choice}`" for choice in action.choices)
        help_text = " ".join(str(action.help or "").split())
        if "%(default)" in help_text:
            help_text = help_text % {"default": action.default}
        help_text = (help_text or "(no description)") + _default(action)
        lines += [f"- {names}", f"  {help_text.strip()}", ""]
    return lines


def new_language_parser() -> argparse.ArgumentParser:
    """LangAdapter's new-language arguments under Testuary's name."""
    from langadapter import cli

    commands = next(action for action in cli.parser()._actions
                    if isinstance(action, argparse._SubParsersAction))
    parser = commands.choices["new-language"]
    parser.prog = "testuary new-language"
    return parser


def _commands():
    from langadapter.tutorial import parser as tutorial_parser
    from testuary import __main__ as cli
    from testuary import tutorial
    from testuary import setup as _setup

    return (
        ("generate", cli._build_parser(prog="testuary generate")),
        ("check", cli._build_parser(prog="testuary check", checking=True)),
        ("init", cli._init_parser()),
        ("provision", cli._provision_parser()),
        ("new-language", new_language_parser()),
        ("setup", _setup._parser()),
        ("tutorial", tutorial_parser(tutorial.PROG, tutorial.DESCRIPTION)),
    )


def _setup_demo_lines() -> list[str]:
    """testuary setup --demo's options, which a parser of their own reads."""
    from testuary.demo import setup_demo

    parser = setup_demo._parser()
    return (["### testuary setup --demo", "", " ".join(parser.description.split()), ""]
            + _option_lines(parser))


def _markdown_text(text: str) -> str:
    """Help text as one Markdown paragraph. Usage brackets such as
    [--output-dir DIR] would read as reference links, so they are escaped."""
    return " ".join(text.split()).replace("[", "\\[").replace("]", "\\]")


def render_cli() -> str:
    from testuary import __main__ as cli

    primary = cli._primary_parser()
    lines = [
        "# Command line",
        "",
        "This page is generated from the command-line parsers. `testuary --help`",
        "and `testuary <command> --help` print the same text. The web interface's",
        "command `testuary-web` is described in the [web interface guide](../guide/web.md#start).",
        "",
        "## Commands",
        "",
        " ".join(primary.description.split()),
        "",
    ]
    commands = _commands()
    for name, parser in commands:
        lines.append(f"- [`testuary {name}`](#testuary-{name}): "
                     + " ".join((parser.description or "").split()))
    lines.append("")
    if primary.epilog:
        lines += [_markdown_text(primary.epilog), ""]
    # The main command's own options, such as --version.
    own = argparse.ArgumentParser(add_help=False)
    own._actions = [action for action in primary._actions if action.option_strings]
    if _option_lines(own):
        lines += ["## testuary", ""] + _option_lines(own)
    for name, parser in commands:
        lines += [f"## testuary {name}", ""]
        if parser.description:
            lines += [" ".join(parser.description.split()), ""]
        lines += _option_lines(parser)
        if parser.epilog:
            lines += [_markdown_text(parser.epilog), ""]
        if name == "setup":
            lines += _setup_demo_lines()
    return "\n".join(lines).rstrip() + "\n"


_REQUEST_CONTROLS = ("Request controls, explained under "
                     "[request controls](../models-and-prompts.md#request-controls).")

#: The keys a section accepts without a packaged default, in groups, each
#: with the page that describes them.
_UNPACKAGED = (
    (("temperature", "top_p", "seed", "max_output_tokens", "request_timeout_seconds",
      "reasoning", "reasoning_effort", "reasoning_budget_tokens", "protocol",
      "request_options", "reasoning_dialect", "reasoning_template"), _REQUEST_CONTROLS),
    (("preserve_reasoning_content", "reasoning_field_names"),
     "Explained under [`[coverage]`](#coverage)."),
    (("chat_model",), "The client factory for this service's aliases, explained in "
                      "[your own model client](../extending/model-provider.md)."),
    (("deployment",), "The label of the service deployment, explained under "
                      "[models.ini fields](../models-and-prompts.md#modelsini-fields)."),
    (("api_url",), "The address of every model, explained under "
                   "[select a model](../models-and-prompts.md#select-a-model)."),
    (("seed_base",), "The first sampling seed of the starter suites, explained under "
                     "[sampling seeds of the starter suites]"
                     "(../guide/configuration.md#sampling-seeds-of-the-starter-suites)."),
)

#: The pages the comments of config.ini name, longest name first, linked on
#: the page.
_NAMED_PAGES = (
    ("web interface guide's section Access", "../guide/web.md#access"),
    ("explanation page Measurement and recovery", "../explanation/measurement-and-recovery.md"),
    ("web interface guide", "../guide/web.md"),
    ("prompt profile guide", "../extending/prompt-profile.md"),
    ("configuration guide", "../guide/configuration.md"),
    ("demonstrations guide", "../guide/demonstrations.md"),
    ("cluster guide", "../guide/cluster.md"),
    ("hints guide", "../extending/hints.md"),
)
_NAMED_PAGE = re.compile("|".join(re.escape(name) for name, _ in _NAMED_PAGES))


def _linked(text: str) -> str:
    """``text`` with each page it names linked."""
    pages = dict(_NAMED_PAGES)
    return _NAMED_PAGE.sub(lambda match: f"[{match.group(0)}]({pages[match.group(0)]})", text)


#: Sections that take the keys of another section.
_TAKES = {"mutation": "coverage", "limits_coverage": "limits", "limits_mutation": "limits"}


def _unpackaged_lines(section, schema) -> list[str]:
    """The items of the keys ``section`` accepts beyond its packaged
    entries: the keys it takes from another section, a seed a run refuses
    and each group of ``_UNPACKAGED``, marked "no default"."""
    listed = {setting.name for setting in section.settings}
    keys = [key for key in schema[section.name] if key not in listed]
    lines = []
    parent = _TAKES.get(section.name)
    if parent and any(key in schema[parent] for key in keys):
        keys = [key for key in keys if key not in schema[parent]]
        other = "other " if listed else ""
        refused = [f"`{key}`" for key in schema[parent] if key not in schema[section.name]]
        apart = f" apart from {' and '.join(refused)}" if refused else ""
        lines +=[f"- Every {other}key of [`[{parent}]`](#{parent}){apart}, no default", ""]
    from testuary.settings import PROVIDER_SECTIONS

    if "seed" in keys and section.name in ("starter", *PROVIDER_SECTIONS):
        keys.remove("seed")
        lines += ["- `seed`, refused by a run", "  A run refuses a seed here, as "
                  "[request controls](../models-and-prompts.md#request-controls) says.", ""]
    for group, text in _UNPACKAGED:
        named = [f"`{key}`" for key in keys if key in group]
        if named:
            lines += [f"- {', '.join(named)}, no default", f"  {text}", ""]
    unexplained = sorted(set(keys) - {key for group, _ in _UNPACKAGED for key in group})
    if unexplained:
        raise KeyError(f"[{section.name}] accepts {unexplained}, which no page describes")
    return lines


def render_settings() -> str:
    from testuary.settings import _defaults, _schema, reference

    lines = [
        "# Settings",
        "",
        "This page is generated from the packaged defaults (`config.ini` in the",
        "installed package) and their comments, and from the `[project]` settings",
        "the built-in languages declare. A setting a language declares names",
        "that language. A key that a section accepts without a value in `config.ini`",
        "is listed with \"no default\" and the page that says what applies when",
        "no layer sets it. A configuration file, the environment",
        "(`TESTUARY_<SECTION>_<KEY>`) or the command line overrides these values,",
        "as the [configuration guide](../guide/configuration.md) describes. An",
        "empty default means unset. The model request controls, such as",
        "`temperature`, `reasoning` and `request_timeout_seconds`, are explained",
        "under [request controls](../models-and-prompts.md#request-controls). A",
        "variable ending in `_API_KEY` is a model service's key, never a setting.",
        "",
    ]
    schema = _schema(_defaults())
    # The built-in languages only, so that an installed language package
    # does not change the page. The [project] keys other languages declare
    # stay off it for the same reason.
    for section in reference(languages=("python", "java", "c")):
        lines += [f"## [{section.name}]", ""]
        if section.purpose:
            lines += [_linked(section.purpose), ""]
        for setting in section.settings:
            shown = f"`{setting.default}`" if setting.default else "empty"
            names = " and ".join(f"`{language}`" for language in setting.languages)
            declared = f", declared by {names}" if names else ""
            lines.append(f"- `{setting.name}`, default {shown}{declared}")
            if setting.purpose:
                lines.append(f"  {_linked(setting.purpose)}")
            lines.append("")
        if section.name != "project":
            lines += _unpackaged_lines(section, schema)
    return "\n".join(lines).rstrip() + "\n"


#: Guides with generated regions: the exit statuses of a command
#: (``<!-- exit-statuses: run -->``) and the cleanup time of a stopped run
#: (``<!-- cleanup-seconds -->``), both from ``testuary.exits``.
GUIDES = tuple(ROOT / "docs" / path for path in (
    "getting-started.md", "guide/output.md", "guide/terminal.md", "guide/demonstrations.md"))
_STATUSES = re.compile(r"(<!-- exit-statuses: (\w+) -->\n).*?(<!-- /exit-statuses -->)", re.S)
_CLEANUP = re.compile(r"(<!-- cleanup-seconds -->).*?(<!-- /cleanup-seconds -->)")


def render_regions(text: str) -> str:
    """``text`` with its generated regions written from ``testuary.exits``."""
    from testuary import exits

    def statuses(match: re.Match) -> str:
        items = "".join(f"- Status {item.status}: {item.meaning}\n"
                        for item in exits.STATUSES[match.group(2)])
        return match.group(1) + items + match.group(3)

    text = _STATUSES.sub(statuses, text)
    return _CLEANUP.sub(lambda match: f"{match.group(1)}{exits.CLEANUP_SECONDS:g}{match.group(2)}",
                        text)


def pages() -> dict[Path, str]:
    written = {CLI_PAGE: render_cli(), SETTINGS_PAGE: render_settings()}
    for guide in GUIDES:
        written[guide] = render_regions(guide.read_text(encoding="utf-8"))
    return written


if __name__ == "__main__":
    sys.path.insert(0, str(ROOT / "src"))
    for page, text in pages().items():
        page.write_bytes(text.encode("utf-8"))
        print(f"wrote {page.relative_to(ROOT)}")
