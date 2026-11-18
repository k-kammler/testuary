"""Kill-matrix alarms for unattributed, timeout and unresolved kills.

A mutant of status killed with no named killer is one every test triggers (a
broken-source mutant or a test-side collection error). A high fraction of them
marks a broken baseline or a contaminated measurement. The alarms only read the
matrix and log. They never change a score."""
from __future__ import annotations

from testuary.languages import mutation as km


def _mutant(i, status, killers=(), line=None):
    return {"id": i, "status": status, "killers": list(killers), "line": line}


def _matrix(mutants):
    return {"mutants": mutants, "summary": {}}


def _killed_no_killer(i):
    return _mutant(i, "killed", killers=())


def _killed_by(i, killer):
    return _mutant(i, "killed", killers=[killer])


def test_alarm_fires_on_a_heavily_unattributed_matrix() -> None:
    mutants = (
        [_killed_no_killer(f"n{i}") for i in range(8)]
        + [_killed_by(f"a{i}", "t.py::x") for i in range(2)]
    )
    msg = km.unattributed_kill_alarm(_matrix(mutants))
    assert msg is not None
    assert "no named killer" in msg
    assert "8/10" in msg


def test_the_alarm_percentage_is_scaled_before_the_division() -> None:
    """23 of 40 is 57.5%, printed 58%. The fraction times 100 would print 57%."""
    mutants = (
        [_killed_no_killer(f"n{i}") for i in range(23)]
        + [_killed_by(f"a{i}", "t.py::x") for i in range(17)]
    )
    assert "23/40 kills (58%)" in km.unattributed_kill_alarm(_matrix(mutants))


def test_alarm_silent_on_a_clean_matrix() -> None:
    # One module-break among many assertion kills is normal, not suspect.
    mutants = (
        [_killed_no_killer("n0")]
        + [_killed_by(f"a{i}", "t.py::x") for i in range(9)]
    )
    assert km.unattributed_kill_alarm(_matrix(mutants)) is None


def test_alarm_silent_below_the_kill_floor() -> None:
    # A tiny all-breaking matrix does not carry enough kills to be a signal.
    mutants = [_killed_no_killer(f"n{i}") for i in range(3)]
    assert km.unattributed_kill_fraction(_matrix(mutants)) == 1.0
    assert km.unattributed_kill_alarm(_matrix(mutants)) is None


def test_timeout_alarm_fires_on_any_clock_kill() -> None:
    # A timeout verdict depends on load, so even one is worth a re-measure.
    mutants = (
        [_killed_by(f"a{i}", "t.py::x") for i in range(9)]
        + [_mutant("t0", "timeout")]
    )
    msg = km.timeout_kill_alarm(_matrix(mutants))
    assert msg is not None
    assert "1/10" in msg and "status timeout" in msg


def test_timeout_alarm_silent_without_clock_kills() -> None:
    mutants = [_killed_by(f"a{i}", "t.py::x") for i in range(5)] + [
        _mutant("s1", "survived")]
    assert km.timeout_kill_alarm(_matrix(mutants)) is None


def test_alarm_does_not_change_the_score() -> None:
    mutants = (
        [_killed_no_killer(f"n{i}") for i in range(8)]
        + [_killed_by(f"a{i}", "t.py::x") for i in range(2)]
        + [_mutant("s1", "survived"), _mutant("s2", "survived")]
    )
    m = _matrix(mutants)
    before = km.matrix_score_percent(m)
    km.unattributed_kill_alarm(m)
    after = km.matrix_score_percent(m)
    assert before == after
    assert before == 100.0 * 10 / 12


def test_unresolved_alarm_names_the_unresolved_kills() -> None:
    """A kill whose verification produced no measurement evidence is named for
    a manual re-measure and counted against the killed rows."""
    matrix = _matrix([
        {**_killed_no_killer(1), "unresolved": True},
        _killed_by(2, "t.py::test_a"),
    ])
    alarm = km.unresolved_kill_alarm(matrix)
    assert alarm is not None
    assert alarm.startswith("[mutation] 1/2 kills are UNRESOLVED")


def test_unresolved_alarm_silent_on_a_clean_matrix() -> None:
    assert km.unresolved_kill_alarm(_matrix([
        _killed_by(1, "t.py::test_a"), _mutant(2, "survived"),
    ])) is None
