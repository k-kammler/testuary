"""The exit statuses and the cleanup time a program shows come from the code
that ends the command and cleans up, and the module that names them loads
nothing else."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from testuary import exits

SOURCE = Path(__file__).resolve().parents[2] / "src"


def test_every_command_can_end_interrupted():
    for command in exits.STATUSES:
        assert exits.INTERRUPTED in [item.status for item in exits.STATUSES[command]]


def test_the_demonstrations_statuses_are_its_own():
    from testuary.demo import cli as demo_cli

    statuses = [item.status for item in exits.STATUSES[exits.DEMO]]
    for status in (demo_cli.EXIT_OK, demo_cli.EXIT_USAGE, demo_cli.EXIT_STALE):
        assert status in statuses


def test_the_cleanup_time_is_where_the_sessions_define_it():
    from testuary.languages import mutation

    assert exits.CLEANUP_SECONDS == mutation.CANCEL_CLEANUP_WAIT_S


def test_each_status_has_one_sentence_and_an_unknown_one_is_named():
    for command, items in exits.STATUSES.items():
        assert len({item.status for item in items}) == len(items), command
        assert all(item.meaning.endswith(".") for item in items), command
    assert exits.meaning(exits.RUN, 2) == (
        "A setting or option was refused, or a required tool was missing, before any "
        "work started.")
    assert exits.meaning(exits.CHECK, 7) == "The command ended with status 7."


def test_a_run_missing_its_dependencies_is_refused_with_its_sentence(monkeypatch):
    """The run's dependency check refuses the run before any phase, as a
    refused setting is: its setup report names what is missing, and its
    status, 2, has a sentence that says so."""
    import pytest

    from testuary import orchestrator
    from testuary.readiness import _configuration_error_report
    from testuary.settings import ConfigurationError

    class Backend:
        def capabilities(self, source_dir):
            raise RuntimeError("no report")

        def check_dependencies(self):
            return ["the test tool is not installed"]

    monkeypatch.setattr(orchestrator.languages, "get_backend", lambda name: Backend())
    with pytest.raises(ConfigurationError) as refused:
        orchestrator._preflight_run_dependencies({"python"}, ".")
    report = _configuration_error_report(refused.value, before_run=True)
    assert (report.status, report.exit_code) == ("refused", 2)
    assert [(item.key, item.status) for item in report.items] == [("native.execution", "missing_setup")]
    assert "[python] the test tool is not installed" in report.items[0].reason
    assert "a required tool was missing" in exits.meaning(exits.RUN, report.exit_code)


def test_each_status_names_its_state_in_one_word():
    """A status display shows one word per status from here: a refused
    setting is not a failed run, nor is a problem the check found."""
    for command, items in exits.STATUSES.items():
        assert all(item.word.isalpha() and item.word.islower() for item in items), command
        assert {item.word for item in items if item.status == exits.INTERRUPTED} == {"interrupted"}
    assert [(exits.status(exits.RUN, code).word, exits.status(exits.RUN, code).failure)
            for code in (0, 1, 2, 130)] == [
        ("finished", False), ("failed", True), ("refused", False), ("interrupted", False)]
    assert [exits.status(exits.CHECK, code).word for code in (0, 1, 2)] == ["ready", "problem", "refused"]
    assert exits.status(exits.DEMO, 3).word == "outdated" and exits.status(exits.DEMO, 3).failure
    unknown = exits.status(exits.CHECK, 7)
    assert (unknown.word, unknown.failure, unknown.meaning) == (
        "failed", True, "The command ended with status 7.")


def test_importing_the_module_loads_no_other_part_of_testuary():
    probe = ("import sys; import testuary.exits; "
             "print(sorted(m for m in sys.modules if m.startswith('testuary')))")
    completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True,
                               env={"PYTHONPATH": str(SOURCE),
                                    "SYSTEMROOT": __import__("os").environ.get("SYSTEMROOT", "")},
                               timeout=60, check=True)
    assert completed.stdout.strip() == "['testuary', 'testuary.exits']"
