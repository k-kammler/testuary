"""``testuary --version`` names the Testuary and LangAdapter versions in use
and exits 0 before any configuration is read."""
from __future__ import annotations

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


def test_the_help_lists_the_version_option():
    assert "--version" in cli._primary_parser().format_help()


def test_the_command_line_starts_without_the_chat_log_or_message_classes():
    """Starting the command line loads neither the chat log nor LangChain's
    message classes; they load when a run reaches its model loop."""
    probe = ("import sys, testuary.__main__; "
             "print(sorted(m for m in ('testuary._chat_log', 'langchain_core.messages') "
             "if m in sys.modules))")
    out = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
