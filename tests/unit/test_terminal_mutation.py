"""Mutation measurement progress from the backend's public record, rendered as observations.

The records are built here with the backend's own type; they are display
inputs, not native measurements.
"""
from __future__ import annotations

import io

import pytest
from rich.console import Console

from langadapter.mutation import MutationProgress
from testuary.presentation import TerminalPresenter
from testuary.presentation.measurement import mutation_progress_ending, mutation_progress_event


@pytest.fixture(autouse=True)
def terminal_environment(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")


def _event(finished, total, elapsed_s, kind="mutants"):
    return mutation_progress_event(MutationProgress(kind, finished, total, elapsed_s - 40.0),
                                   run_id="r", target="src/calc.py", elapsed_s=elapsed_s)


def _frame(presenter):
    snapshot = Console(file=io.StringIO(), width=120, color_system=None, force_terminal=True,
                       highlight=False, markup=False)
    with snapshot.capture() as captured:
        snapshot.print(presenter.frame())
    return captured.get()


def _compact():
    console = Console(file=io.StringIO(), force_terminal=True, width=120, height=40,
                      color_system=None, highlight=False, markup=False)
    return TerminalPresenter("compact", console=console)


def test_progress_is_a_running_count_without_a_score():
    event = _event(12, 40, 52.5)
    assert (event.state, event.level, event.label) == ("running", "operation", "Mutation testing")
    assert (event.completed, event.total, event.work_unit) == (12, 40, "mutants run")
    # The run clock is the producer's; the record's own clock is not substituted.
    assert event.elapsed_s == 52.5 and event.metrics == () and event.phase == "mutation"


def test_compact_view_draws_a_bar_and_estimates_after_three_mutants_in_view():
    presenter = _compact()
    with presenter:
        presenter(_event(0, 40, 50))
        presenter(_event(2, 40, 60))
        assert "Mutation testing" in _frame(presenter) and "left" not in _frame(presenter)
        presenter(_event(4, 40, 70))
        frame = _frame(presenter)
    # 20 s for 4 mutants: about 180 s for the remaining 36.
    # A count of mutants run, without a percentage that would read as a score.
    assert "4 of 40 mutants run" in frame and " 10%  4 of 40" not in frame and "about 3 min left" in frame


def test_unknown_total_stays_unknown_and_other_kinds_keep_their_unit():
    plain = io.StringIO()
    with TerminalPresenter("plain", stream=plain) as presenter:
        presenter(_event(7, None, 47))
        presenter(_event(3, 9, 48, kind="operators"))
    lines = plain.getvalue().splitlines()
    assert "7 mutants run, total unknown" in lines[0] and "%" not in lines[0]
    assert "3 of 9 operators" in lines[1] and "%" not in lines[1]
    assert "Mutant results can still change until mutation testing ends" in lines[0]


def test_kill_verification_is_a_row_of_its_own_beside_the_measurement():
    verification = _event(3, 12, 75, kind="kill verifications")
    assert (verification.operation, verification.label, verification.work_unit) == (
        "kill-verification", "Re-checking caught mutants", "re-checked")
    assert verification.identity != _event(40, 40, 70).identity
    presenter = _compact()
    with presenter:
        presenter(_event(40, 40, 70))
        presenter(verification)
        frame = _frame(presenter)
    # The finished sweep keeps its count, and the verification counts its own.
    assert "40 of 40 mutants run" in frame and "Re-checking caught mutants" in frame and "3 of 12 re-checked" in frame
    assert mutation_progress_ending("kill verifications", "succeeded") == (
        "Re-checking caught mutants done")
    assert mutation_progress_ending("mutants", "cancelled") == "Mutation testing stopped"


def test_other_records_are_refused():
    with pytest.raises(TypeError, match="MutationProgress"):
        mutation_progress_event(object(), run_id="r", target="t", elapsed_s=1.0)


def test_plain_progress_lines_follow_ten_percent_steps_or_thirty_seconds():
    plain = io.StringIO()
    with TerminalPresenter("plain", stream=plain) as presenter:
        # One record per second, 1000 mutants: a line per ten percent, not per record.
        for second in range(25):
            presenter(_event(second * 10, 1000, 50 + second))
        # Without a total, at most one line every 30 s.
        for second in range(0, 65, 5):
            presenter(mutation_progress_event(MutationProgress("mutants", second, None, 1.0),
                                              run_id="r", target="src/other.py",
                                              elapsed_s=100 + second))
    lines = plain.getvalue().splitlines()
    known = [line for line in lines if "src/calc.py" in line]
    unknown = [line for line in lines if "src/other.py" in line]
    assert [line.split(" of 1000 ")[0].rsplit(" ", 1)[-1] for line in known] == ["0", "100", "200"]
    assert len(unknown) == 3  # at 0, 30 and 60 s
