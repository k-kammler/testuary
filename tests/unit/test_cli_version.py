"""``testuary --version`` names the Testuary and LangAdapter versions in use
and exits 0 before any configuration is read."""
from __future__ import annotations

import re
import subprocess
import sys

import pytest

import langadapter
import testuary
from testuary import __main__ as cli
from testuary import settings as settings_module


def test_version_prints_both_packages_and_reads_no_configuration(monkeypatch, capsys):
    monkeypatch.setattr(settings_module, "resolve_settings",
                        lambda **kwargs: pytest.fail("configuration read"))
    monkeypatch.setattr(sys, "argv", ["testuary", "--version"])
    with pytest.raises(SystemExit) as stop:
        cli.main()
    assert stop.value.code == 0
    assert capsys.readouterr().out.splitlines() == [
        f"testuary {testuary.__version__}", f"langadapter {langadapter.__version__}"]


def test_the_help_lists_the_options_used_without_a_command(monkeypatch, capsys):
    """`testuary --help` lists the version and the options that the guides
    use without a command, with the help that `testuary generate --help`
    gives them."""
    monkeypatch.setattr(sys, "argv", ["testuary", "--help"])
    with pytest.raises(SystemExit) as stop:
        cli._main()
    assert stop.value.code == 0
    listed = re.findall(r"^  (--[\w-]+)", capsys.readouterr().out, re.M)
    assert listed == ["--version", "--demo", "--demo-language", "--home"]
    run_help = {action.dest: action.help for action in cli._build_parser()._actions}
    main_help = {action.dest: action.help for action in cli._primary_parser()._actions}
    for dest in ("demo", "demo_language", "home"):
        assert main_help[dest] == run_help[dest]


def test_the_usage_puts_no_option_before_a_command():
    """A command is the first word: options go before the files of a run
    or after the command, as in `testuary check --home DIR`."""
    assert cli._primary_parser().format_usage().splitlines() == [
        "usage: testuary [-h] [--version]",
        "       testuary [--demo [--demo-language LANGUAGE]] [--home DIR] [FILE ...]",
        "       testuary COMMAND [OPTIONS]"]


def test_the_command_line_starts_without_the_chat_log_or_message_classes():
    """Starting the command line loads neither the chat log nor LangChain's
    message classes. They load when a run reaches its model loop."""
    probe = ("import sys, testuary.__main__; "
             "print(sorted(m for m in ('testuary._chat_log', 'langchain_core.messages') "
             "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
