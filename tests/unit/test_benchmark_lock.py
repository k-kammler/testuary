"""The benchmark lock in ``testuary.project``: on or off as the run
records ``[timing] serialize``, and its context manager."""
from __future__ import annotations

import contextlib

import pytest

from testuary import project as benchmark_lock

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


# ---------------------------------------------------------------------------
# _serialize_enabled -- the on/off resolution.
# ---------------------------------------------------------------------------

def test_serialize_enabled_default_is_true(in_run) -> None:
    """Without the variable or a setting the run records the lock on, so
    disabling it is opt-in."""
    resolved = in_run()
    assert resolved.sections["timing"]["serialize"] == "true"
    assert benchmark_lock._serialize_enabled() is True


@pytest.mark.parametrize("value, on", [
    ("on", True), ("TRUE", True), ("yes", True), ("1", True),
    ("off", False), ("False", False), ("no", False), ("0", False),
])
def test_the_lock_is_what_the_run_records(monkeypatch, in_run, value, on) -> None:
    """Each spelling the settings check takes for the variable sets the lock
    as the run records it: the lock reads the recorded value."""
    monkeypatch.setenv("TESTUARY_TIMING_SERIALIZE", value)
    resolved = in_run()
    assert resolved.sections["timing"]["serialize"] == ("true" if on else "false")
    assert benchmark_lock._serialize_enabled() is on


@pytest.mark.parametrize("value", ["", "maybe"])
def test_an_empty_or_unknown_value_refuses_the_run(monkeypatch, in_run, value) -> None:
    monkeypatch.setenv("TESTUARY_TIMING_SERIALIZE", value)
    with pytest.raises(ValueError, match=r"\[timing\] serialize: expected true or false"):
        in_run()


def test_the_lock_reads_its_setting_inside_a_run_only(in_run) -> None:
    """Outside a run the lock has no setting to read and says so, as every
    configuration read does. It is never silently on."""
    in_run.leave()
    with pytest.raises(RuntimeError, match="no run settings are active"):
        with benchmark_lock.acquire():
            pass


# ---------------------------------------------------------------------------
# acquire() context manager.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("value, on", [("true", True), ("false", False)])
def test_acquire_holds_langadapters_lock_as_the_run_records(monkeypatch, in_run, value, on) -> None:
    """``acquire()`` passes the run's ``[timing] serialize`` to LangAdapter's
    benchmark lock, which holds the block when it is on."""
    monkeypatch.setenv("TESTUARY_TIMING_SERIALIZE", value)
    in_run()
    passed = []

    @contextlib.contextmanager
    def serialized(enabled):
        passed.append(enabled)
        yield

    monkeypatch.setattr(benchmark_lock._placement, "serialized", serialized)
    with benchmark_lock.acquire():
        pass
    assert passed == [on]
