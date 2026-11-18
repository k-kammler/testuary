"""``--runs N`` runs a file's whole test pipeline N times, as runs r0 to
r(N-1). ``--run K`` runs only rK. Each run draws its own starter seeds,
1000 apart, so at most 1000 starter suites are allowed."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from testuary import __main__ as cli
from testuary import orchestrator, settings
from testuary import config
from testuary.orchestrator import RunRequest, _invocation


def test_runs_defaults_to_one_and_takes_a_count():
    parser = cli._build_parser()
    args = parser.parse_args([])
    assert (args.runs, args.run) == (1, None)
    args = parser.parse_args(["--runs", "3"])
    assert args.runs == 3 and cli._check_run_options(args) is None
    args = parser.parse_args(["--run", "2"])
    assert args.run == 2 and cli._check_run_options(args) is None


def test_the_stop_phase_is_read_without_case_or_surrounding_spaces():
    args = cli._build_parser().parse_args(["--stop-after-phase", " Coverage "])
    assert args.stop_after_phase == "coverage" and cli._check_run_options(args) is None


@pytest.mark.parametrize("arguments, message", [
    (["--runs", "0"], "--runs must be 1 or more, not 0"),
    (["--runs", "-2"], "--runs must be 1 or more, not -2"),
    (["--run", "-1"], "--run must be a repetition number from 0, not -1"),
    (["--runs", "2", "--run", "1"], "--run K and --runs N cannot be combined"),
])
def test_a_wrong_run_selection_is_refused(arguments, message):
    args = cli._build_parser().parse_args(arguments)
    with pytest.raises(ValueError) as refused:
        cli._check_run_options(args)
    assert message in str(refused.value)


def test_the_help_shows_both_run_options_and_no_repeat():
    text = cli._build_parser().format_help()
    assert "--runs N" in text and "--run K" in text
    assert "--repeat" not in text


def test_the_request_selects_runs_or_one_run():
    assert _invocation(RunRequest(("a.py",), runs=3)).runs == 3
    request = _invocation(RunRequest(("a.py",), run=4))
    assert (request.runs, request.run) == (1, 4)
    for fields in ({"runs": 0}, {"runs": True}, {"run": -1}, {"run": "1"},
                   {"runs": 2, "run": 1}, {"runs": 2, "stop_after_phase": "coverage"}):
        with pytest.raises(ValueError):
            RunRequest(("a.py",), **fields)
    # One selected run may stop after a phase.
    RunRequest(("a.py",), run=2, stop_after_phase="coverage")


@pytest.mark.parametrize("reuse", [("starter", "coverage", "mutation", "finishing"), ("finishing",),
                                   ("starter", "finishing")])
@pytest.mark.parametrize("selection", [{"runs": 2}, {"run": 1}])
def test_a_finishing_reuse_works_with_the_first_run_alone(reuse, selection):
    with pytest.raises(ValueError, match="first repetition alone"):
        RunRequest(("a.py",), reuse=reuse, **selection)
    # The first run alone keeps its own finishing results to reuse.
    RunRequest(("a.py",), reuse=reuse)
    RunRequest(("a.py",), reuse=reuse, run=0)
    # Earlier phases are reused in every run.
    RunRequest(("a.py",), reuse=("starter", "coverage", "mutation"), **selection)


def test_the_selected_runs_are_run_numbers_from_zero():
    assert orchestrator._selected_run_indices(SimpleNamespace(runs=1, run=None)) == (0,)
    assert orchestrator._selected_run_indices(SimpleNamespace(runs=3, run=None)) == (0, 1, 2)
    assert orchestrator._selected_run_indices(SimpleNamespace(runs=1, run=4)) == (4,)


def test_each_run_draws_its_own_seeds_a_thousand_apart():
    # Run r0 sends the seeds of a single run: seed_base + i.
    assert [config.run_seed(0, 0, slot) for slot in (1, 2, 3)] == [1, 2, 3]
    assert [config.run_seed(0, 1, slot) for slot in (1, 2, 3)] == [1001, 1002, 1003]
    assert config.run_seed(7, 3, 1000) == 7 + 3000 + 1000


def test_more_starter_suites_than_one_run_has_seeds_are_refused(tmp_path):
    def resolve(count):
        return settings.resolve_settings(
            base_dir=tmp_path, use_environment=False,
            overrides={"starter": {"suite_count": str(count)}})

    assert resolve(1000).sections["starter"]["suite_count"] == "1000"
    with pytest.raises(ValueError, match="at most 1000 suites"):
        resolve(1001)
