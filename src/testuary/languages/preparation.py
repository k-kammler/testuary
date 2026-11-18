"""Prepared measurements, each in a temporary copy of the project made by
LangAdapter's ``prepare_project`` and run by a fresh adapter in that copy.
The environment-dir rule and the bytecode store come from
``langadapter.python.project``. A failed preparation or cleanup is marked
on its exception by the ``preparation_setup_error`` and
``preparation_cleanup_error`` attributes callers read, beside
``prepared_project``, the owner of the failed copy.
"""
from __future__ import annotations

import atexit
from contextlib import contextmanager, suppress
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import posixpath
import threading
import time

from langadapter import (LangAdapter, ProjectInputSelection, create, prepare_project,
                         recovery_owners, registration)
from langadapter.context import ContainerExecution
from langadapter.errors import UnsupportedLanguage
from langadapter.python.project import BytecodeStore, environment_dir
from langadapter.execution import container_work_dir_for
from langadapter.suites import Suite

from testuary import phase_state

#: This layer's own markers on an exception from a prepared copy: its failed
#: setup, its failed cleanup, and the owner of the copy.
_PREPARED_MARKERS = ("preparation_setup_error", "preparation_cleanup_error", "prepared_project")


def _host_root(context):
    execution = context.execution
    if not isinstance(execution, ContainerExecution):
        return Path(context.root).resolve()
    mapped = PurePosixPath(execution.container_work_dir or container_work_dir_for(execution.work_dir))
    try:
        relative = PurePosixPath(context.root).relative_to(mapped)
    except ValueError:
        return None
    return Path(execution.work_dir).resolve() / relative


def _project_path(value, original, scoped):
    if value is None:
        return None
    guest = isinstance(original.execution, ContainerExecution)
    if guest:
        host = Path(value)
        original_host = _host_root(original)
        if host.is_absolute() and original_host is not None:
            try:
                relative = host.resolve().relative_to(original_host.resolve())
            except ValueError:
                pass
            else:
                return str(PurePosixPath(scoped.root).joinpath(*relative.parts))
        if PureWindowsPath(value).is_absolute():
            # An unmatched host path is not a relative guest project input.
            return str(value)
    path_type = PurePosixPath if guest else Path
    path = path_type(value)
    if not path.is_absolute():
        return str(path_type(scoped.root) / path)
    original_root = path_type(original.root)
    if not guest:
        path, original_root = path.resolve(), original_root.resolve()
    try:
        return str(path_type(scoped.root) / path.relative_to(original_root))
    except ValueError:
        # An invalid or external path stays as it is for the native
        # request's validation, never posing as a project member.
        return str(path)


def _suite(suite, original, scoped, owner=None):
    old = _host_root(original)
    new = _host_root(scoped)

    def mapped(path):
        selected = Path(path).resolve()
        if old is not None and new is not None:
            try:
                return str(new / selected.relative_to(old))
            except ValueError:
                pass
        copied = owner.companion_path(str(selected)) if owner is not None else None
        return copied if copied is not None else str(selected)

    return Suite(mapped(suite.root),
                 tuple(mapped(Path(suite.root) / path) for path in suite.test_files),
                 tuple(mapped(Path(suite.root) / path) for path in suite.setup_files))


def project_inputs(context):
    """The copy plan of a measurement: the project without its runtime and
    without Testuary's own files, or None to copy the whole tree.

    The selected interpreter's environment directory inside the project (a
    project ``.venv``) is the runtime, not an input: a POSIX virtual
    environment's links cannot be copied, and copying it per measurement is
    wasted work. The active run's configuration files and the project's
    Testuary directory inside the project are left out too
    (``config.configuration_exclusions``).
    """
    from testuary import config

    excluded = []
    interpreter = context.interpreter
    if (not isinstance(context.execution, ContainerExecution) and interpreter is not None
            and interpreter.path is not None):
        env_dir = environment_dir(interpreter.path, context.root)
        if env_dir is not None:
            excluded.append(env_dir)
    root = _host_dir(context)
    if root is not None:
        excluded.extend(config.configuration_exclusions(root))
    return ProjectInputSelection(exclude=tuple(dict.fromkeys(excluded))) if excluded else None


def _host_dir(context):
    """The source directory on the host as written, or None for a guest directory
    outside the mapped work directory."""
    execution = context.execution
    if not isinstance(execution, ContainerExecution):
        return Path(os.path.abspath(context.root))
    mapped = PurePosixPath(execution.container_work_dir or container_work_dir_for(execution.work_dir))
    try:
        relative = PurePosixPath(context.root).relative_to(mapped)
    except ValueError:
        return None
    return Path(os.path.abspath(execution.work_dir)).joinpath(*relative.parts)


def _plan(context):
    plan = project_inputs(context)
    return {} if plan is None else {"inputs": plan}


#: The run's bytecode stores by work directory (None: the host's temporary
#: directory), created on first use and removed when the run ends.
_STORES = {}
_STORES_LOCK = threading.Lock()
#: The work directories whose missing store the run log has named, once each.
_UNSTORED = set()


def _fact(backend, name):
    """The class fact ``name`` of ``backend``'s adapter class, or, for a
    description of a backend, of its registered language's adapter. Off
    (LangAdapter's default) for a language without the fact."""
    if isinstance(backend, LangAdapter):
        return bool(getattr(type(backend), name, False))
    try:
        factory = registration(backend.language).factory
    except UnsupportedLanguage:
        return False
    return bool(getattr(factory, name, False))


def bytecode_store(backend):
    """The preparation option that gives a measurement the run's bytecode
    store when its language reuses compiled modules (``reuses_bytecode``,
    Python's): what runs in earlier copies compiled is written into each new
    copy, as a run in place reuses its own. {} for other languages or when
    no store can be made."""
    if not _fact(backend, "reuses_bytecode"):
        return {}
    execution = backend.context.execution
    zone = execution.work_dir if isinstance(execution, ContainerExecution) else None
    with _STORES_LOCK:
        store = _STORES.get(zone)
        if store is None:
            try:
                store = BytecodeStore.create(parent=zone)
            except OSError as error:
                # The measurement runs without the store, as before, and the
                # run log says so once per work directory, with the error's text,
                # never its path. Each later measurement tries again.
                if zone not in _UNSTORED:
                    _UNSTORED.add(zone)
                    from testuary._progress import write_plain_log

                    write_plain_log(f"[bytecode] a {backend.language} measurement runs without the "
                                    f"compiled-module store, so it compiles what it imports: "
                                    f"{type(error).__name__} ({error.strerror or 'no reason given'})")
                return {}
            if not _STORES:
                atexit.register(_close_stores)
            _STORES[zone] = store
    return {"bytecode": store}


def _close_stores():
    with _STORES_LOCK:
        for zone, store in list(_STORES.items()):
            with suppress(Exception):
                store.close()
            del _STORES[zone]


def _companions(backend, suite):
    """The suite's directory when it lies outside the project and the language
    copies such a suite (``copies_external_suite``, Python's), copied beside
    the project's copy so that a run prints the paths an in-place run
    printed. Other languages' suites (Java's) stay external. In a container
    the suite's directory in the work directory is compared with the project as the
    container sees both, so a project outside the work directory gets its suite
    copied too."""
    if suite is None or not _fact(backend, "copies_external_suite"):
        return {}
    suite_dir = _companion(backend, Path(suite.root).resolve())
    return {} if suite_dir is None else {"companions": (suite_dir,)}


def session_companions(backend, suite_dirs):
    """The companions of a mutation session's copy: each of ``suite_dirs``,
    the suite directories its caller declared, that a run copies beside the
    project (``_companions``), the outermost of nested ones."""
    if not _fact(backend, "copies_external_suite"):
        return {}
    found = list(dict.fromkeys(Path(suite_dir).resolve() for suite_dir in suite_dirs))
    outer = [suite_dir for suite_dir in found
             if not any(suite_dir != other and suite_dir.is_relative_to(other) for other in found)]
    kept = tuple(copied for copied in (_companion(backend, suite_dir) for suite_dir in outer)
                 if copied is not None)
    return {"companions": kept} if kept else {}


def _companion(backend, suite_dir):
    """``suite_dir``, resolved, when a suite in it lies outside the project and is
    copied beside the copy of the project, else None."""
    execution = backend.context.execution
    if isinstance(execution, ContainerExecution):
        zone = Path(execution.work_dir).resolve()
        if not suite_dir.is_relative_to(zone):
            return None
        mapped = PurePosixPath(execution.container_work_dir
                               or container_work_dir_for(execution.work_dir))
        guest, project = mapped.joinpath(*suite_dir.relative_to(zone).parts), PurePosixPath(
            backend.context.root)
        shared = PurePosixPath(posixpath.commonpath([str(guest), str(project)]))
        return None if shared in (guest, project) else str(suite_dir)
    host = _host_root(backend.context)
    if host is None:
        return None
    try:
        shared = Path(os.path.commonpath([str(suite_dir), str(host)]))
    except ValueError:
        # On another drive, pytest prints the suite's absolute path from the
        # original and from the copy alike.
        return None
    # A suite inside the project is in the copy already. One around the project
    # stays external (not mirrored).
    return None if shared in (suite_dir, host) else str(suite_dir)


@contextmanager
def prepared_measurement(backend, suite, *project_paths):
    """A fresh backend in a prepared copy, with the suite and
    ``project_paths`` mapped into it, for one measurement. A Python suite
    outside the project is copied beside the copy as it lay beside the
    project, so a run prints the paths an in-place run prints. Read the
    results inside the block, since the copy is removed when it ends. A
    mutation session keeps one copy for its whole life instead.
    """
    started = time.perf_counter()
    owner = _prepared_owner(backend, suite, started)
    primary = None
    spent, resumed = 0.0, None
    try:
        with owner.use() as scoped:
            measured = _measured(backend, suite, owner, scoped, project_paths)
            spent = time.perf_counter() - started
            try:
                yield measured
            finally:
                resumed = time.perf_counter()
    except BaseException as error:
        primary = error
        raise
    finally:
        try:
            _close_owner(owner, primary)
        finally:
            ended = time.perf_counter()
            _charge_preparation(ended - started if resumed is None
                                else spent + ended - resumed)


@contextmanager
def kept_measurement(backend, suite, *project_paths):
    """A prepared copy kept for several measurements. Yields a function
    whose context enters the copy for one measurement and gives what
    ``prepared_measurement`` gives. LangAdapter allows one current use of a
    copy per thread, so copies kept side by side (an interleaved timing)
    are entered one measurement at a time. The copy keeps what its runs
    leave in it (Python's bytecode). Its preparation and removal count as
    the phase's preparation time, the measurements between them do not."""
    started = time.perf_counter()
    owner = _prepared_owner(backend, suite, started)
    spent = time.perf_counter() - started

    @contextmanager
    def use():
        with owner.use() as scoped:
            yield _measured(backend, suite, owner, scoped, project_paths)

    primary = None
    try:
        yield use
    except BaseException as error:
        primary = error
        raise
    finally:
        closing = time.perf_counter()
        try:
            _close_owner(owner, primary)
        finally:
            _charge_preparation(spent + time.perf_counter() - closing)


def _prepared_owner(backend, suite, started):
    """The prepared copy of the project for ``suite``. A failed preparation
    is charged as preparation time and marked as a setup error, as is a
    failed bytecode store, which may leave the directory it made."""
    try:
        options = {**_plan(backend.context), **_companions(backend, suite)}
        try:
            options.update(bytecode_store(backend))
        except BaseException as error:
            error.preparation_setup_error = error
            raise
        return prepare_project(backend.context, **options)
    except BaseException as error:
        _charge_preparation(time.perf_counter() - started)
        if recovery_owners(error):
            error.preparation_setup_error = error
        raise


def _measured(backend, suite, owner, scoped, project_paths):
    """A fresh backend in the copy, the suite and the project paths mapped
    into it."""
    return (create(backend.language, scoped), _suite(suite, backend.context, scoped, owner),
            tuple(_project_path(path, backend.context, scoped) for path in project_paths))


def _close_owner(owner, primary):
    """Remove the copy. A cleanup failure is raised, or, when ``primary``
    is already propagating, attached to it."""
    try:
        owner.close()
    except BaseException as cleanup:
        cleanup.prepared_project = owner
        cleanup.preparation_cleanup_error = cleanup
        if primary is None:
            raise
        primary.preparation_cleanup_error = cleanup
        if hasattr(primary, "add_note"):
            with suppress(Exception):
                primary.add_note("Prepared measurement cleanup remains unresolved: " + str(cleanup))


def _charge_preparation(seconds: float) -> None:
    """A prepared measurement's own time (the copy, its bytecode and its
    removal) as the phase's preparation time, unless an open span counts it
    already."""
    if not phase_state.timing_open():
        phase_state.record_time(phase_state.TIME_PREPARATION, seconds)
