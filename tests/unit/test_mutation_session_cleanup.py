"""Consumer cleanup retains owned sessions without hiding the body failure."""
from types import SimpleNamespace

import pytest

from langadapter.mutation import MutationCleanupError
from testuary.languages.mutation import (
    MutationRunner, MutationSession, MutationSetupError, close_mutation_session,
    mutation_cleanup_pending,
)


class NativeSession:
    def __init__(self):
        self.fail = None
        self.closes = 0

    def close(self):
        self.closes += 1
        if self.fail is not None:
            raise self.fail


class NoNotesError(OSError):
    def __getattribute__(self, name):
        if name == "add_note":
            raise AttributeError(name)
        return super().__getattribute__(name)


class BrokenNotesError(OSError):
    def add_note(self, note):
        raise KeyboardInterrupt("optional note failed")


@pytest.fixture
def opened():
    native = NativeSession()
    runner = MutationRunner(None, {})
    session = MutationSession(runner, native, 2, True)
    runner.open = lambda workers, suite_dirs=(): session
    return runner, session, native


def test_successful_close_is_idempotent_and_refuses_later_measurements(opened):
    _, session, native = opened
    session.close()
    session.close()
    assert session._closed and native.closes == 1
    with pytest.raises(MutationSetupError, match="closed"):
        session.run(target=None, test_dir=None, test_files=[], per_mutant_timeout=1,
                    wall_clock_timeout=1)
    with pytest.raises(MutationSetupError, match="closed"):
        with session:
            pytest.fail("Closed session must not enter")


@pytest.mark.parametrize("primary_type", [OSError, KeyboardInterrupt, NoNotesError, BrokenNotesError])
def test_body_error_survives_cleanup_failure_with_same_recovery_handle(opened, primary_type):
    runner, session, native = opened
    primary = primary_type("original phase failure")
    cleanup = OSError("worker disposal unconfirmed")
    native.fail = cleanup
    with pytest.raises(primary_type) as caught:
        with runner.session(2, suite_dirs=()):
            raise primary
    assert caught.value is primary
    assert primary.mutation_cleanup_handle is session
    assert primary.mutation_cleanup_error is cleanup
    assert mutation_cleanup_pending(primary) and not session._closed
    assert native.closes == 1
    with pytest.raises(RuntimeError, match="retried") as refused:
        session.run(target=None, test_dir=None, test_files=[], per_mutant_timeout=1,
                    wall_clock_timeout=1)
    assert refused.value.mutation_cleanup_handle is session
    native.fail = None
    primary.mutation_cleanup_handle.close()
    session.close()
    assert session._closed and native.closes == 2


def test_successful_body_does_not_hide_failed_close(opened):
    runner, session, native = opened
    error = MutationCleanupError(native, "workers held")
    native.fail = error
    with pytest.raises(MutationCleanupError) as caught:
        with runner.session(2, suite_dirs=()):
            pass
    assert caught.value is error
    assert error.cleanup_handle is native
    assert error.mutation_cleanup_handle is session
    assert error.mutation_cleanup_error is error
    assert mutation_cleanup_pending(error) and not session._closed


def test_explicit_primary_close_preserves_the_body_object_and_failed_owner(opened):
    _, session, native = opened
    primary = KeyboardInterrupt()
    native.fail = OSError("cannot close")
    session.close(primary=primary)
    assert primary.mutation_cleanup_handle is session
    assert primary.mutation_cleanup_error is native.fail
    assert mutation_cleanup_pending(primary) and not session._closed


@pytest.mark.parametrize("body_error", [False, True])
def test_shared_close_preserves_no_argument_replay_or_registered_protocol(body_error):
    session = NativeSession()
    cleanup = OSError("replay close failed")
    session.fail = cleanup
    primary = KeyboardInterrupt() if body_error else None
    if primary is None:
        with pytest.raises(OSError) as caught:
            close_mutation_session(session)
        assert caught.value is cleanup
        error = caught.value
    else:
        close_mutation_session(session, primary=primary)
        error = primary
    assert error.mutation_cleanup_handle is session
    assert error.mutation_cleanup_error is cleanup
    assert mutation_cleanup_pending(error)
    session.fail = None
    error.mutation_cleanup_handle.close()
    assert session.closes == 2


@pytest.mark.parametrize("marker", ["preparation_setup_error",
                                  "preparation_cleanup_error"])
def test_prepared_owner_failure_refuses_setup_fallback(marker):
    error = OSError("retained prepared owner")
    setattr(error, marker, error)
    assert mutation_cleanup_pending(error)


def test_native_setup_cleanup_is_public_and_clean_setup_failure_can_fall_back():
    owner = SimpleNamespace(close=lambda: None)
    error = MutationCleanupError(owner, "cleanup unresolved")
    assert mutation_cleanup_pending(error)
    assert error.cleanup_handle is owner
    assert not mutation_cleanup_pending(MutationSetupError("setup failed, session closed"))
    assert not mutation_cleanup_pending(ValueError("unsupported engine before allocation"))
