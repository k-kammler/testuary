"""Top-level orchestration for explicitly selected source files.

Prepare per-target work trees, plan resume and checkpoints, drive the test
phases, extract deliverables and report token usage. The CLI parses arguments
and calls this module. Source targets are supplied through --source-files.
"""
from __future__ import annotations

import functools
import posixpath
import shutil
import time
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple
from testuary import RESULTS_VERSION, project, phase_state, languages
from testuary import results
from testuary import config as config_module
from testuary.settings import ConfigurationError, _MissingSetup
from testuary import hints as hints_module
from testuary.agent._helpers import AgentLogger, PhaseLogger
from testuary._progress import (
    begin_plain_log,
    close_plain_log,
    diagnostic_print,
    echoes_log_line,
    open_plain_log,
    plain_log_path,
    write_plain_log,
)
from dataclasses import dataclass
from types import SimpleNamespace


if TYPE_CHECKING:
    from testuary.presentation.events import RunPlan
    from testuary.readiness import RunReadiness
    from testuary.settings import RunSettings


# The run's main log (``<outdir>/logs/main.jsonl``), installed by
# ``_run_pipeline`` once the project name and outdir are known. All phase
# banners, skip notices and summaries flow through it. Each agent keeps its
# own log file. Before it, ``log_print`` writes a line with its level to the
# plain log and prints it.
_log = PhaseLogger("main", unlogged="plain")


def set_main_logger(logger: AgentLogger) -> None:
    """Install the main log's logger (once per run, by ``_run_pipeline``)."""
    _log.logger = logger


def get_main_logger() -> Optional[AgentLogger]:
    """The main log's logger, or None before a run installs it."""
    return _log.logger


log_print = _log.log


def _echoed_log(message: str, level: int = 0) -> None:
    log_print(message, level, echo=True)


def _remove_path(path: Path) -> None:
    """Remove a file or directory if it exists."""
    if not path.exists() and not path.is_symlink():
        return
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def _reused_phases(args) -> frozenset:
    """The phases this invocation reuses from a prior run by flag: a phase
    with the phases it builds on (mutation reuse the coverage phase, whose
    suite the mutation phase revised, and coverage reuse the starter phase,
    whose baseline is the coverage file being restored), and finishing reuse
    itself."""
    reused = {phase for phase in PHASES if getattr(args, f"reuse_{phase}", False)}
    if "mutation" in reused:
        reused.add("coverage")
    if "coverage" in reused:
        reused.add("starter")
    return frozenset(reused)


def _cleanup_preserve_flags(args) -> bool:
    """Decide whether ``_cleanup_outdir`` must keep the prior test artifacts,
    from the reuse / resume CLI flags. Returns ``keep_tests``.

    The contract that prevents data loss: a flag that reuses or resumes an
    artifact must keep that artifact on disk. A fully fresh run
    requires a missing or empty output directory.

    ``keep_tests``: each target file's ``tests/`` and its ``archive/`` (the matrices
    grid and the reusable starter pool) survive, and so does ``logs/test``. Set by
    any flag that reuses / resumes the test artifacts (``--reuse
    starter`` / ``--reuse coverage`` / ``--reuse mutation`` / ``--resume``) or by ``reuse_suites`` (it
    reloads the persisted suites under ``archive/starter/suites/``).

    ``--reuse finishing`` skips measurement but consumes no saved suite on its own.
    Pair it with an artifact-reuse flag to preserve an occupied destination.
    """
    # Reusing the starter suites reloads files that live under the target file's
    # results archive (archive/starter/suites/), so the cleanup must keep that
    # tree or it would delete the suites before the starter phase reads them.
    # Read the resolved value (flag -> env -> config), so the preservation
    # holds however the reuse was requested.
    reuse_suites = bool(
        config_module.get_agent_config("test_initial")["reuse_suites"]
    )
    return bool(
        _reused_phases(args) - {"finishing"}
        or args.resume
        or reuse_suites
    )


def _check_output_route(path: Path) -> None:
    """Refuse filesystem aliases without adopting their targets for cleanup."""
    project._check_directory_route(path, purpose="output", action="Select a new --output-dir")


def _admit_output(outdir: Path, *, preserve: bool) -> Path:
    """Inspect the selected route, then return one effective output address."""
    _check_output_route(outdir)
    import os

    outdir = Path(os.path.abspath(outdir))
    if outdir.exists() and not outdir.is_dir():
        raise ConfigurationError(
            f"the run directory is not an ordinary directory: {outdir}. "
            "Select a new --output-dir"
        )
    if not preserve and outdir.exists() and next(outdir.iterdir(), None) is not None:
        raise ConfigurationError(
            f"the run directory is not empty: {outdir}. No earlier file was removed. "
            "Select a new --output-dir, or use --resume or --reuse for compatible saved work"
        )
    if preserve:
        namespace = project.resolve_work_dir_namespace()
        _check_output_route(outdir / "_work" / namespace if namespace else outdir / "_work")
    return outdir


def _check_output_outside_source(outdir: Path, projects_root: Path) -> None:
    """Refuse an output directory inside the original project, by the
    admitted effective paths, unless it lies in the project's Testuary
    directory, which no copy of the project holds."""
    if outdir.is_relative_to(projects_root):
        settings = config_module.active_settings()
        if settings is not None and outdir.is_relative_to(
                settings.project_dir / settings.main.testuary_dir):
            return

        raise ConfigurationError(
            "the run directory is inside the source directory: "
            "select --output-dir outside --source-dir"
        )


def _keep_results_out_of_version_control(outdir: Path) -> None:
    """Give the project's results directory its own ``.gitignore`` when a run
    writes below it, so that version control leaves the results out. An
    existing file is kept as it is."""
    settings = config_module.active_settings()
    if settings is None or not outdir.is_relative_to(settings.results_dir):
        return
    ignore = settings.results_dir / ".gitignore"
    if not ignore.exists():
        ignore.write_bytes(b"*\n")


def _cleanup_outdir(
    *,
    outdir: Path,
    keep_tests: bool,
) -> None:
    """Admit the destination again and clear only preserved work's transient
    tree. The cleanup removes files, so it checks the route itself, also
    after the run's first admission: a symbolic link is never a cleanup's
    authority.

    Layout reminder (``<workspace>`` names a target by its path in the
    source directory, ``project.target_workspace_name``):
      - ``<outdir>/results/<workspace>/r0/...``   per-target results of each run
      - ``<outdir>/logs/run.log``, ``main.jsonl``  orchestration logs
      - ``<outdir>/logs/test/<workspace>/...``    per-target phase logs
      - ``<outdir>/_work/<workspace>/...``        transient runtime tree

    Full run: require a missing or empty ordinary directory.
    A reuse or a resume (``keep_tests``): keep results/<workspace>/ and
        logs/test, wipe the transient work tree so each preserved suite is
        re-measured from a clean copy.
    """
    outdir = _admit_output(outdir, preserve=keep_tests)
    if not outdir.exists() or not keep_tests:
        return

    # Remove only the current execution's transient namespace. Other
    # execution namespaces remain until their owner removes them.
    job_ns = project.resolve_work_dir_namespace()
    work_root = (outdir / "_work" / job_ns) if job_ns else (outdir / "_work")
    # Preserve each target file's tests/ and its logs/test. Only the transient work
    # tree is wiped so the preserved suites are re-measured from clean copies.
    _remove_path(work_root)


def _copy_project_to_work(
    *,
    projects_root: Path,
    work_src: Path,
) -> None:
    """Mirror the selected project into the target's private copy.

    ``__pycache__`` and stray ``.pyc`` files are dropped so the target starts
    with a clean bytecode cache (otherwise stale .pyc could mask source
    edits during the first test run). A virtual environment inside the
    project (a directory holding ``pyvenv.cfg``) is left out: a run uses its
    interpreter where it lies, from the source directory.

    Copied files/directories gain owner-write permission for execution. Originals
    retain their existing bytes and permissions, including intentional read-only
    files. ``copy2`` preserves modes, so this applies only to the private copy.

    Container mode has no host-side copy: the repo already lives inside the
    container at REPO_DIR and is measured there (the target points at
    REPO_DIR/<file>), so this is a no-op.
    """
    if config_module.project_exec_mode() == "container":
        return
    import os
    # Testuary's own configuration and catalogue files (models.ini can hold
    # keys) are not inputs of the project under test, and generated tests must not find copies.
    # Entries are matched by file identity (device and inode) against the
    # files Testuary read, so neither a link nor a different letter case on a
    # case-insensitive filesystem lets a configuration file through. The
    # lexical match covers a filesystem without stable identities.
    root = os.path.abspath(projects_root)
    excluded = {os.path.normcase(os.path.join(root, *relative.split("/")))
                for relative in config_module.configuration_exclusions(root)}
    identities = set()
    for path in config_module.configuration_input_files():
        try:
            info = os.stat(path)
        except OSError:
            continue
        if info.st_ino:
            identities.add((info.st_dev, info.st_ino))
    bytecode = shutil.ignore_patterns("__pycache__", "*.pyc")

    def ignore(directory, names):
        ignored = set(bytecode(directory, names))
        # A virtual environment is no input of the project under test.
        ignored.update(name for name in names
                       if os.path.isfile(os.path.join(directory, name, "pyvenv.cfg")))
        if excluded or identities:
            here = os.path.abspath(directory)
            for name in names:
                entry = os.path.join(here, name)
                if os.path.normcase(entry) in excluded:
                    ignored.add(name)
                    continue
                try:
                    info = os.stat(entry)
                except OSError:
                    continue
                if (info.st_dev, info.st_ino) in identities:
                    ignored.add(name)
        return ignored

    shutil.copytree(projects_root, work_src, ignore=ignore)
    import stat as _stat
    user_write = _stat.S_IWUSR
    for entry in [work_src, *work_src.rglob("*")]:
        try:
            if entry.is_symlink() or not (entry.is_file() or entry.is_dir()):
                continue
            mode = entry.stat().st_mode
            if not (mode & user_write):
                entry.chmod(mode | user_write)
        except OSError:
            continue


def _check_canonical_against_git(
    projects_root: Path,
    target_files: List[Path],
) -> List[Tuple[Path, str]]:
    """Run ``git diff --quiet HEAD -- <file>`` for each file when
    ``projects_root`` is a git checkout. Returns the list of
    ``(file, summary)`` pairs that diverge from HEAD. Empty when
    everything is clean or when the directory is not a git repository.

    This is the ground-truth check: the file's authoritative version
    is the one committed in git, and any drift is a real problem
    (manual edit, stale cosmic-ray mutation, etc.). Cheap: one git
    invocation per file, no working-tree side effects.
    """
    import subprocess
    if config_module.project_exec_mode() == "container":
        # The container holds its own git checkout. There is no host-side
        # checkout to diff, and projects_root is a pure POSIX path here.
        return []
    if not (projects_root / ".git").exists():
        return []
    drift: List[Tuple[Path, str]] = []
    for f in target_files:
        try:
            rel = f.resolve().relative_to(projects_root.resolve())
        except ValueError:
            continue
        # ``git diff --quiet`` exits 0 (clean) / 1 (drift) / 128 (err).
        rc = subprocess.run(
            ["git", "-C", str(projects_root), "diff", "--quiet",
             "HEAD", "--", str(rel)],
            capture_output=True,
        ).returncode
        if rc == 1:
            lines = subprocess.run(
                ["git", "-C", str(projects_root), "diff", "--stat",
                 "HEAD", "--", str(rel)],
                capture_output=True, text=True,
            ).stdout.strip().splitlines()
            drift.append((f, lines[-1] if lines else ""))
    return drift


def _token_usage_lines(results_dir: Path) -> List[str]:
    """The run's model tokens by phase, summed from tests.csv, as log lines
    in pipeline order and a total. A phase whose rows spent no tokens did
    not call the model this run and is left out. Read from ``results_dir``,
    the ``results/`` directory of this run's run directory, so the summary is
    this run's. No lines when no row records tokens or tests.csv cannot be
    read: a run without output has nothing to sum."""
    totals: Dict[str, List[int]] = {}
    try:
        for row in results._read_csv(results_dir / "tests.csv")[1]:
            for phase in ("starter", "coverage", "mutation"):
                spent = (row.get(f"{phase}_input_tokens"), row.get(f"{phase}_output_tokens"))
                if not any(spent):
                    continue
                slot = totals.setdefault(phase, [0, 0, 0])
                slot[0] += int(spent[0] or 0)
                slot[1] += int(spent[1] or 0)
                slot[2] += 1
    except (OSError, ValueError):
        pass
    if not totals:
        return []
    lines = [f"  Test   {phase:<11}: {spent_in:>14,} in / {spent_out:>14,} out  "
             f"({rows} row{'s' if rows != 1 else ''})"
             for phase in ("starter", "coverage", "mutation") if phase in totals
             for spent_in, spent_out, rows in [totals[phase]] if spent_in or spent_out]
    return lines + ["  " + "-" * 64,
                    f"  {'Total':<21}: {sum(t[0] for t in totals.values()):>14,} in / "
                    f"{sum(t[1] for t in totals.values()):>14,} out"]


def _list_active_hint_cells(hints: Dict[str, Any],
                            hint_profile: str = "") -> List[str]:
    """The names of the cells of ``hints`` that apply to a run with
    ``hint_profile``: in each phase, a populated ``general``, the
    profile's ``per_profile`` cell, every populated ``per_language``
    cell and the profile's ``per_profile_per_language`` cells. Cells of
    another profile do not apply and are not listed.

    Format: ``<phase>.general``, ``<phase>.per_profile[<P>]``,
    ``<phase>.per_language[<L>]``,
    ``<phase>.per_profile_per_language[<P>][<L>]``.

    Returns an empty list for an empty / absent hints file so the
    caller can suppress the entire end-of-run section.
    """
    from testuary.hints import _hint_string, PHASES as HINTS_PHASES
    if not isinstance(hints, dict):
        return []
    out: List[str] = []
    for phase in HINTS_PHASES:
        block = hints.get(phase)
        if not isinstance(block, dict):
            continue
        if _hint_string(block.get("general")):
            out.append(f"{phase}.general")
        per_profile = block.get("per_profile") or {}
        if hint_profile and isinstance(per_profile, dict) and _hint_string(
                per_profile.get(hint_profile)):
            out.append(f"{phase}.per_profile[{hint_profile}]")
        per_language = block.get("per_language") or {}
        if isinstance(per_language, dict):
            for l_name, l_value in sorted(per_language.items()):
                if _hint_string(l_value):
                    out.append(f"{phase}.per_language[{l_name}]")
        per_pl = block.get("per_profile_per_language") or {}
        profile_section = (per_pl.get(hint_profile) or {}
                           if hint_profile and isinstance(per_pl, dict)
                           else {})
        if isinstance(profile_section, dict):
            for l_name in sorted(profile_section):
                if _hint_string(profile_section.get(l_name)):
                    out.append(f"{phase}.per_profile_per_language"
                               f"[{hint_profile}][{l_name}]")
    return out


def _preflight_reuse_requirements(
    args: Any,
    top_files: List[Dict[str, Any]],
    results_dir: Path,
    projects_root_resolved: Optional[Path] = None,
) -> List[str]:
    """Verify the on-disk artifacts each ``--reuse`` unit assumes are
    present, before the target loop starts.

    Returns a list of human-readable problems. An empty list means
    every requested target can run under the active reuse flags. When
    non-empty, ``_run_pipeline`` aborts with exit 2 (config error) and prints
    the list, so the operator can either re-run without the offending
    flag or narrow ``--targets`` to the runnable subset.

    The checks:

      - ``--reuse coverage`` / ``--reuse mutation``:
        each requested target's ``results/<workspace>/r0/tests/`` directory
        must exist: the target's phases restore the preserved file from that
        directory.

    ``--resume`` takes no reuse but the starter's (``check_run_options``), so
    no resumed run reaches this check.
    """
    if "coverage" not in _reused_phases(args):
        return []
    flag = "--reuse coverage" if args.reuse_coverage else "--reuse mutation"
    problems: List[str] = []
    for top_file in top_files:
        target_number = top_file["target_number"]
        canonical_path = Path(top_file["file_path"]).resolve()
        workspace = project.target_workspace_name(canonical_path, projects_root_resolved)
        target_results_dir = results_dir / workspace
        tests_dir = project.phase_artifact_path(target_results_dir, "tests", 0)

        if not tests_dir.exists():
            problems.append(
                f"target {target_number} ({canonical_path.name}): {flag} needs "
                f"saved tests at {tests_dir} but the directory "
                f"does not exist. Run without {flag} first, or "
                f"narrow --targets to leave out target {target_number}."
            )
    return problems


def _restore_preserved_tests(
    *,
    workspace: str,
    results_dir: Path,
    work_tests: Path,
    run_idx: int = 0,
) -> bool:
    """Repopulate ``work_tests`` from a previous run's
    results/<workspace>/r<k>/tests/.

    Used by --reuse coverage and --resume so the run has a
    tests directory to work from. Returns True when at least one test file
    was restored. Run ``run_idx``'s own tests are restored, else the first
    run's (``pick_phase_artifact``).
    """
    target_dir = results_dir / workspace
    try:
        src = project.pick_phase_artifact(
            target_dir, "tests", run_idx,
        )
    except FileNotFoundError:
        return False
    if not src.exists():
        return False
    work_tests.mkdir(parents=True, exist_ok=True)
    restored = False
    for entry in src.iterdir():
        target = work_tests / entry.name
        if entry.is_file():
            shutil.copy2(entry, target)
            restored = True
        elif entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
            restored = True
    return restored


def _tests_snapshot(tests_dir: Path) -> Dict[str, Tuple[bytes, float]]:
    """Every file below ``tests_dir`` by relative path, with its bytes and
    modification time, so a later run can be given the directory as it was."""
    if not tests_dir.is_dir():
        return {}
    return {path.relative_to(tests_dir).as_posix(): (path.read_bytes(), path.stat().st_mtime)
            for path in sorted(tests_dir.rglob("*")) if path.is_file()}


def _restore_tests_snapshot(snapshot: Dict[str, Tuple[bytes, float]], work_tests: Path) -> bool:
    """Write a ``_tests_snapshot`` into ``work_tests``. Returns True when at
    least one file was restored."""
    import os

    for relative, (data, mtime) in snapshot.items():
        target = work_tests / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        os.utime(target, (mtime, mtime))
    return bool(snapshot)


class _RunTimer:
    """The pipeline segment of the run in progress: started when a run
    starts, which ends the previous run's, and stopped when the file's runs
    end. A crash closes it as crashed, as a phase segment is."""

    def __init__(self) -> None:
        self._segment = None

    def start(self, target_dir: Path, run_idx: int) -> None:
        self.stop()
        segment = phase_state.timed_segment(
            target_dir, phase_state.SEGMENT_PIPELINE, run_idx=run_idx)
        # Kept only once entered: a stop that arrives while the segment
        # opens leaves nothing for stop() to close, and the stop goes on as
        # a stop, not as a crash of this bookkeeping.
        segment.__enter__()
        self._segment = segment

    def stop(self, *, crashed: bool = False) -> None:
        segment, self._segment = self._segment, None
        if segment is not None:
            failure = RuntimeError("the run ended with an error") if crashed else None
            segment.__exit__(type(failure) if failure else None, failure, None)


def _copy_first_run_archives(args, target_results_dir: Path, run_idx: int) -> None:
    """Copy into run ``run_idx``'s directory the archives of the phases this
    invocation reuses, from the first run (r0).

    A reuse takes the first run's results as the input of every run, so each
    later run's directory then holds the records of the phases its suite came
    from, and the run's phases read them where they read their own. The
    starter pool stays per run and is not copied."""
    reused = _reused_phases(args) - {"finishing"}
    first = project.run_dir(target_results_dir, 0) / "archive"
    own = project.run_dir(target_results_dir, run_idx) / "archive"
    for phase in sorted(reused):
        if (first / phase).is_dir():
            shutil.copytree(first / phase, own / phase, dirs_exist_ok=True,
                            ignore=lambda visited_dir, names: ["suites"] if Path(visited_dir) == first / "starter" else [])


def _extract_tests_deliverable(
    *,
    work_tests: Path,
    target_results_dir: Path,
    run_idx: int = 0,
) -> None:
    """Copy the target file's tests dir into the run's
    results/<workspace>/r<k>/tests/."""
    if not work_tests.exists():
        return
    result_tests_dir = project.phase_artifact_path(
        target_results_dir, "tests", run_idx,
    )
    # The delivered directory is the suite, its two eval-facing sidecars,
    # conftest.py and pytest.ini, nothing else. The runtime machinery leaves
    # dotfile droppings in the working tests dir (.evict_*,
    # .best_green_*.bak, .mutation_baseline_matrix.json, .coverage) that are
    # working state, not deliverables. Skip every dotfile on the copy so they
    # are never delivered. conftest.py / pytest.ini / *_matrix.json are not dotfiles
    # and are kept.
    shutil.copytree(
        work_tests, result_tests_dir, dirs_exist_ok=True,
        ignore=shutil.ignore_patterns(".*"),
    )


class _PhaseStopNotReached(RuntimeError):
    """An explicit diagnostic boundary failed before it was checkpointed."""


def available_phase_names() -> Tuple[str, ...]:
    """Ordered ``--stop-after-phase`` vocabulary."""
    return tuple(PHASES)


def _run_target_phases_guarded(
    ctx,
    *,
    run_phase,
    results_dir: Path,
    workspace: str,
    target_number: int,
    file_path: str,
    run_idx: int,
    log,
):
    """Run the test pipeline for one target under a crash guard.

    On an unexpected exception, log it with a traceback, mark the phase
    that raised failed with ``failure_kind=infrastructure`` in phase_state,
    write a row whose ``<phase>_status`` reads ``crashed`` to tests.csv (so
    the target is recorded, not silently lost), and return ``None`` so the
    caller skips this target
    rather than letting one target file's crash end the whole run. The designed
    phase failures (a red suite, a missing engine) return a normal
    ``TestPipelineResult`` and never reach here. Only an unhandled crash
    does.
    """
    try:
        return run_phase(ctx)
    except Exception as exc:
        from testuary.model_requests import failure_traceback
        log(
            f"ERROR: target {target_number} crashed unexpectedly in repetition {project.run_name(run_idx)} "
            f"({type(exc).__name__}: {exc}). It is recorded as crashed, and the run "
            f"continues with the next target.",
            level=0,
        )
        # A request error's raw cause can echo a credential: its traceback
        # leaves the chain out.
        log(failure_traceback(exc), level=1)
        # The phase that raised, from its record, and the crash recorded as
        # that phase's infrastructure failure: an execution failure, never
        # the method's outcome.
        target_dir = results_dir / workspace
        try:
            phase = phase_state.running_phase(target_dir, run_idx=run_idx)
        except Exception:
            phase = None
        error = (
            (f"the {phase} phase crashed" if phase is not None
             else "the test pipeline crashed outside a phase")
            + f" (infrastructure failure): {type(exc).__name__}: {exc}"
        )
        if phase is not None:
            try:
                phase_state.mark_failed(
                    target_dir, phase, reason=error, run_idx=run_idx,
                    failure_kind=phase_state.INFRASTRUCTURE_FAILURE,
                )
            except Exception:
                pass
        # The phases this run completed carry their results, rebuilt from
        # their checkpoints. The crashed phase has none.
        try:
            from testuary.target_phases import completed_phase_results

            completed = completed_phase_results(ctx) if ctx is not None else {}
        except Exception:
            completed = {}
        try:
            results.write_tests_row(
                results_dir=results_dir,
                workspace=workspace,
                target_number=target_number,
                file_path=file_path,
                # The crashed phase's status comes from ``crashed``, and a
                # value no phase recorded stays empty. The target's copy holds
                # the crashed phase's half-written suite, which is no
                # delivered suite to count.
                **completed,
                crashed=(phase, error),
                work_tests=None,
                run_idx=run_idx,
                # Overlay the crashed status onto any prior good row for this
                # (target, run_idx) rather than replacing it: a crash after an
                # earlier run wrote full coverage / mutation / Finishing
                # columns keeps those numbers instead of blanking them.
                preserve_existing=True,
            )
        except Exception:
            # The crash row is best-effort. Never let it mask the original
            # crash or stop the run from moving to the next target.
            pass
        return None


def _select_pipeline_targets(
    args: Any,
    *,
    source_targets: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Resolve ``--targets`` into the target files the phases run on.

    The candidates are the selected source files, in their order.
    ``--targets`` selects a slice of it. Without it every file runs.
    Raises ``ConfigurationError`` on an invalid or empty selection.
    """
    targets_spec = getattr(args, "targets", None)
    if targets_spec is None:
        picked = list(source_targets)
    else:
        try:
            requested_numbers = config_module.parse_targets_spec(
                targets_spec, total=len(source_targets),
            )
        except ValueError as exc:
            raise ConfigurationError(str(exc)) from exc
        picked = [
            source_targets[i - 1]
            for i in requested_numbers
            if 1 <= i <= len(source_targets)
        ]
    if not picked:
        raise ConfigurationError(f"--targets {targets_spec} selects none of the "
                                 f"{len(source_targets)} targets")
    if len(picked) > 1 or targets_spec not in (None, "1", "-1"):
        log_print(
            f"All {len(picked)} target(s) selected for tests:"
            if len(picked) == len(source_targets) else
            f"--targets {targets_spec}: {len(picked)} of {len(source_targets)} target(s) selected for tests:"
        )
        for tf in picked:
            log_print(
                f"  target {tf['target_number']:3d} {Path(tf['file_path']).name}",
                level=1,
            )
    return picked


def _target_path(name: str, projects_root, container: bool) -> Tuple[str, str]:
    """A selected file's path and its file name. In a container a POSIX path:
    pathlib on a Windows host would rewrite "/" to "\\" and prepend a drive,
    and the source lives in the container, not on this host to stat, so the
    path stays unresolved, but without "..". On the host the file must
    exist. A relative name starts from the source directory, and a target
    outside it, which has no place in the run's copy, is refused."""
    if container:
        pp = PurePosixPath(name)
        if not pp.is_absolute():
            pp = PurePosixPath(projects_root) / pp
        resolved, base_name = posixpath.normpath(str(pp)), pp.name
    else:
        p = Path(name)
        if not p.is_absolute():
            p = Path(projects_root) / p
        if p.is_dir():
            raise ConfigurationError(f"{name} is a directory, not a source file")
        if not p.is_file():
            raise ConfigurationError(
                f"source file {p} does not exist (relative names "
                f"resolve against the source directory at {projects_root})"
            )
        resolved, base_name = str(p.resolve()), p.name
    try:
        (PurePosixPath(resolved).relative_to(PurePosixPath(projects_root)) if container
         else Path(resolved).relative_to(Path(projects_root).resolve()))
    except ValueError:
        raise ConfigurationError(f"{name} lies outside the source directory {projects_root}") from None
    return resolved, base_name


def _source_file_top_files(args, *, projects_root) -> List[Dict[str, Any]]:
    """Resolve the selected files (named on the command line, with
    ``--source-files``, or by the project's ``source_files`` setting) into
    the targets the phases run on.

    The list order is the target order (the ``target_number`` column of tests.csv).
    Relative paths resolve against the source directory. A missing file or
    a file extension no installed language handles stops the run."""
    supplied = args.source_files
    names = (results.parse_csv_field(supplied) if isinstance(supplied, str)
             else list(supplied))
    if not names:
        raise ConfigurationError("no target selected")
    top_files: List[Dict[str, Any]] = []
    seen = set()
    container = config_module.project_exec_mode() == "container"
    for i, name in enumerate(names):
        resolved, base_name = _target_path(name, projects_root, container)
        if languages.detect_language(resolved) == "unknown":
            raise ConfigurationError(
                f"no installed language handles the source file {base_name}. "
                f"Installed: "
                f"{', '.join(languages.supported_file_extensions())}"
            )
        if resolved in seen:
            raise ConfigurationError(f"source file {resolved} is selected twice")
        seen.add(resolved)
        top_files.append({"target_number": i + 1, "file_path": resolved})
    # A line for the run's log: the report before the run names the files.
    # Before the run's logger exists, the console shows it only where it
    # shows log lines, as the logger does.
    message = f"{len(top_files)} target(s) selected."
    if _log.logger is not None:
        _log.logger.log(message)
    else:
        write_plain_log(message)
        if echoes_log_line(message):
            diagnostic_print(message, logged=True)
    return top_files


def _preflight_run_dependencies(lang_names, source_dir, requirements=None, groups=None) -> None:
    """Stop before any phase runs when a selected target's backend cannot do
    a family the configured workflow requires, or a run-relevant tool is
    missing, listing every gap at once. Announce and remember the fallbacks
    for the optional families it lacks (``languages.profile``). A gap refuses
    the run as missing setup: it is never silently skipped mid-run, nor
    left to crash one language while another quietly degrades.

    ``groups`` are the selected-target groups, each assessed at its own
    captured root. Without them every named language uses ``source_dir``.
    ``requirements`` is the workflow's requirement projection. Without it
    the complete workflow is required."""
    problems = []
    fallbacks = []
    selections = ([(group.language, str(group.context.root)) for group in groups]
                  if groups is not None else
                  [(lang, str(source_dir)) for lang in sorted(lang_names)])
    for lang, root in selections:
        backend = languages.get_backend(lang)
        if backend is None:
            continue
        try:
            assessment = languages.assess(backend.capabilities(source_dir=root), requirements)
            gaps = list(assessment.refusals)
            fallbacks.extend(f"  [{lang}] {f}" for f in assessment.fallbacks)
            languages.remember(lang, assessment.applied)
        except Exception as exc:
            gaps = [f"capability report failed ({type(exc).__name__}: {exc})"]
        try:
            gaps += backend.check_dependencies()
        except Exception as exc:
            gaps.append(
                f"dependency check failed ({type(exc).__name__}: {exc})")
        for g in gaps:
            problems.append(f"  [{lang}] {g}")
    if fallbacks and not problems:
        log_print(
            "Optional language adapter features this run goes without:\n"
            + "\n".join(fallbacks), level=0)
    if problems:
        raise _MissingSetup("required tools are missing for this run:\n"
                            + "\n".join(problems) + "\nNo phase ran.")


def _unresolved_suite_validation(pipeline_result) -> Optional[str]:
    """Why a target failed to execute in its starter phase: some generated
    starter suites could not be checked, because running them against the
    file or screening them for tests that read the implementation timed out
    or did not start, three times each. That says nothing about the tests.
    None otherwise."""
    starter = getattr(pipeline_result, "starter_result", None)
    count = int(getattr(starter, "n_validation_unresolved", 0) or 0)
    if not count:
        return None
    return (f"{count} generated starter suite(s) could not be checked: running them "
            f"against the file (or screening them for tests that read the "
            f"implementation) timed out or did not start, three times each. This says "
            f"nothing about the tests")


def _summarize_unfinished_targets(unfinished, log) -> int:
    """Log the targets that did not produce a complete suite, each once with
    every reason, and return the command's exit status: 1 when one of them
    failed to execute (a crash, a target that could not be set up, restored
    or delivered, a refused reuse, or a starter phase with generated suites
    that could not be checked), else 0. A resume skip and a result (a
    suite below its floor, a phase that ended without a suite) are listed
    but leave the status at 0, and ``tests.csv`` records them. A target
    with a failure among its reasons counts as failed, as in its
    completion record."""
    if not unfinished:
        return 0
    targets = {}
    for entry in unfinished:
        targets.setdefault((entry["target_number"], entry["file_path"]), []).append(entry)
    log("")
    log("=" * 60)
    log(f"Targets without a suite that meets the minimums ({len(targets)}):")
    log("=" * 60)
    failed = skipped = 0
    for (number, file_path), entries in targets.items():
        failing = any(entry["failed"] for entry in entries)
        failed += failing
        skipped += not failing and all(entry["reason"] == _RESUME_SKIP_REASON
                                       for entry in entries)
        log(
            f"  target {number:3d} "
            f"{Path(file_path).name:30s} "
            f"{'FAILED: ' if failing else ''}"
            f"{', and '.join(entry['reason'] for entry in entries)}",
            level=1,
        )
    # A target without a complete suite is a scientific outcome, counted
    # apart from a failure to execute.
    counts = [f"{count} {what}" for count, what in (
        (len(targets) - failed - skipped,
         "ended without a suite that meets the minimums (a result, not a failure)"),
        (skipped, _RESUME_SKIP_REASON),
    ) if count]
    if counts:
        log(", ".join(counts) + ".", level=0)
    if not failed:
        return 0
    log(
        f"{failed} of them failed to execute, so the command exits with "
        f"status 1.",
        level=0,
    )
    return 1


#: Each command-line option that sets a setting: its section, key and
#: attribute, and whether it names a path, which on the command line is
#: taken from the working directory when relative.
_CLI_SETTING_FIELDS = (
    ("api", "model", "model", False), ("api", "api_url", "api_url", False),
    ("project", "name", "project_name", False), ("project", "output_dir", "output_dir", True),
    ("project", "source_dir", "source_dir", True),
    ("project", "python_executable", "python_executable", True),
    ("project", "java_home", "java_home", True),
    ("project", "build_java_home", "build_java_home", True),
    ("project", "python_import_root", "python_import_root", False),
    ("main", "hints_file", "hints_file", True),
    ("main", "prompt_profile", "prompt_profile", True),
    (config_module.phase_section("finishing"), "phase_scores", "phase_scores", False),
)
_CLI_PATHS = frozenset((section, key) for section, key, _, path in _CLI_SETTING_FIELDS if path)


def _relative_path(text: str) -> bool:
    """Whether ``text`` is a relative path: not empty, and absolute for
    neither system, so a container's POSIX root stays as given."""
    return bool(text) and not (PurePosixPath(text).is_absolute() or PureWindowsPath(text).anchor)


def _from_working_dir(overrides):
    """The command line's settings with each relative path taken from the
    working directory, as each option's help says, not from the project directory
    the settings resolve from. ``run_pipeline(args)`` keeps its base."""
    taken = {section: dict(values) for section, values in overrides.items()}
    for section, key in _CLI_PATHS:
        value = str(taken.get(section, {}).get(key, "")).strip()
        if _relative_path(value):
            taken[section][key] = str(Path.cwd() / value)
    return taken


def _cli_settings_overrides(args):
    starter = config_module.phase_section("starter")
    fields = {"api": {}, "project": {}, "main": {}, starter: {},
              config_module.phase_section("finishing"): {}}
    for section, key, attr, _ in _CLI_SETTING_FIELDS:
        value = getattr(args, attr, None)
        if value is not None:
            fields[section][key] = value
    port = getattr(args, "proxy_port", None)
    if port is not None:
        if not 1 <= port <= 65535:
            raise ValueError("--proxy-port must be between 1 and 65535")
        fields["api"].setdefault("api_url", f"http://localhost:{port}/v1")
    workers = getattr(args, "concurrent_requests", None)
    if workers is not None:
        fields[starter].update(concurrent_requests=workers, concurrent_requests_ollama=workers)
    if getattr(args, "reuse_starter_suites", None) is not None:
        fields[starter]["reuse_suites"] = args.reuse_starter_suites
    return {section: values for section, values in fields.items() if values}


def _without_cli_configuration(args):
    """Keep invocation/display policy after the CLI has captured configuration."""
    import copy

    invocation = copy.copy(args)
    for attr in {field[2] for field in _CLI_SETTING_FIELDS} | {
        "config", "home", "proxy_port", "concurrent_requests", "reuse_starter_suites",
        "no_project_settings",
    }:
        if hasattr(invocation, attr):
            setattr(invocation, attr, None)
    return invocation


def check(request: "RunRequest", *, settings: "RunSettings", verify: bool = False) -> "RunReadiness":
    """Check the request's setup without generating tests, and return a
    RunReadiness without printing.

    Returns the setup report as a RunReadiness and prints nothing. With
    ``verify`` it is the tool check, which starts the tools and tries the
    project in a copy, and for a Java project downloads what a run would
    when it is missing. Without ``verify`` it downloads nothing. An item the
    check cannot try is reported as not tried: only a run tries it.
    """
    from testuary.readiness import _native_report

    return _native_report(request, settings, verify=verify)


def run_pipeline(args, selected_languages=None, *, settings=None, event_sink=None, events=None):
    """Run the targets of ``args``, a ``RunRequest``, under ``settings``, a
    ``RunSettings`` from ``resolve_settings``, in isolated run state.

    Pass ``settings`` for programmatic use, without configuration flags on
    ``args``. Command-line flags such as target selection and resume are separate.
    Overlapping calls in one process are refused, so use separate processes.
    ``event_sink`` is an optional read-only display observer. It cannot control
    requests, retries or results.
    ``events`` names an event file, which Testuary writes as ``testuary
    --events FILE`` writes its file, without a command that resumes the run.
    A relative path starts from the current directory, and an existing file is
    replaced. A call refused before its run may start writes no file.
    Before the run starts, it loads the extensions ``settings`` name and
    captures their hints.
    """
    import copy
    from contextlib import ExitStack

    from testuary import _extensions
    from testuary.settings import RunSettings, _check_unchecked, extension_items, resolve_settings

    if events is not None:
        try:
            events = _writable_event_file(events)
        except (TypeError, ValueError) as exc:
            raise ConfigurationError(f"events: {exc}") from None
    request = None
    if isinstance(args, RunRequest):
        if not isinstance(settings, RunSettings):
            raise ConfigurationError("RunRequest requires explicitly resolved RunSettings")
        if selected_languages is not None:
            raise ConfigurationError("set selected languages on RunRequest, not a separate argument")
        selected_languages = set(args.languages) if args.languages is not None else None
        request = args
        args = _invocation(args)

    try:
        overrides = _cli_settings_overrides(args)
        files = getattr(args, "config", None)
        if isinstance(files, str):
            files = [files]
        home = getattr(args, "home", None)
        if settings is not None and (files is not None or overrides or home is not None
                                     or getattr(args, "no_project_settings", False)):
            raise ValueError("settings cannot be combined with configuration flags. Resolve overrides before execution")
        if settings is None:
            settings = resolve_settings(config_files=files, overrides=overrides, home=home,
                                        project_settings=not getattr(args, "no_project_settings", False),
                                        load_extensions=True)
        # A program may resolve its settings without loading the extensions:
        # the run loads them before its work starts, checks the [project]
        # settings their languages hold and captures their hints once.
        _extensions.load(extension_items(settings))
        _check_unchecked(settings)
        captured_hints = hints_module._read_extension_hints(settings)
    except (ValueError, TypeError) as exc:
        raise ConfigurationError(str(exc)) from exc
    from testuary._progress import RunProgress, console_preferences

    event_file = None
    closing = ExitStack()

    def observe(event):
        # The event file first, then the caller's observer.
        if event_file is not None:
            event_file(event)
        if event_sink is not None:
            event_sink(event)

    sink = observe if events is not None else event_sink
    progress = RunProgress(sink) if sink is not None else None
    entered = False
    final_log = ""
    invocation = None
    try:
        with config_module.use_run_settings(settings), console_preferences(
            no_color=bool(getattr(args, "no_color", False)),
            plain=getattr(args, "display", "auto") == "plain",
            diagnostic_writer=getattr(args, "_diagnostic_writer", None),
            # A terminal display shows the run. Its debug level also shows
            # every log line, the other levels only warnings and errors.
            detail_echo=(getattr(args, "_diagnostic_writer", None) is None
                         or getattr(args, "display", "auto") == "debug"),
        ):
            config_module.current_run().extension_hints = captured_hints
            if events is not None:
                # After the run guard, so an overlapping call never replaces
                # a running call's file.
                event_file = _open_event_file(events, settings, closing)
            entered = True
            # Keep lines from before the run log opens (it needs the output
            # directory) and write them first when it does.
            begin_plain_log()
            invocation = copy.copy(args)
            invocation._progress = progress
            invocation._request = request
            invocation._events = event_file
            if progress is not None:
                progress.emit(target="all targets", phase="pipeline", operation="run",
                              state="running", message="Checking run inputs", level="run")
            status = _run_pipeline(invocation, selected_languages)
            final_log = plain_log_path()
    except BaseException as exc:
        import asyncio
        from concurrent.futures import CancelledError

        cancelled = isinstance(exc, (KeyboardInterrupt, asyncio.CancelledError, CancelledError))
        complete = getattr(invocation, "_complete_on_interrupt", None)
        if cancelled and complete is not None:
            try:
                complete()
            except Exception:
                pass  # the interruption is reported either way
        if progress is not None and entered:
            progress.emit(
                target="all targets", phase="pipeline", operation="run",
                state=("cancelled" if cancelled else "refused" if isinstance(exc, ConfigurationError) else "failed"),
                message=("Run stopped by you" if cancelled else
                         "Setup refused" if isinstance(exc, _MissingSetup) else
                         "Configuration refused" if isinstance(exc, ConfigurationError) else
                         "Run ended with an execution error"),
                level="run", log_path=plain_log_path(), **progress.run_counts(),
            )
        if event_file is not None and isinstance(exc, ConfigurationError):
            from testuary.readiness import _configuration_error_report

            event_file.readiness(_configuration_error_report(exc, settings=settings, scope="inspection",
                                                             before_run=True))
        closing.close()
        raise
    finally:
        # The final event names the log before it closes. A call the run guard
        # refused never opened one, and the running run's log stays open.
        if entered:
            close_plain_log()
    if progress is not None:
        state, message, failure = (("succeeded", "Run completed", None) if status == 0
                                   else ("failed", "Run has failed targets", None))
        completion = getattr(invocation, "_completion", None)
        if status == 0 and completion is not None and not all(
                item.outcome in ("complete", "already_complete") for item in completion.files):
            # A file without a complete suite is a result, not a success.
            from testuary.presentation.terminal import headline

            state, message, failure = "failed", headline(completion), "outcome"
        progress.emit(target="all targets", phase="pipeline", operation="run",
                      state=state, message=message, failure=failure,
                      level="run", log_path=final_log, **progress.run_counts())
    closing.close()
    return status


def _writable_event_file(value) -> Path:
    """An event file's path, checked without writing: a relative path from
    the current directory, not a directory, in a directory that exists, and a file,
    or a new file's directory, that can be written. Raises ValueError with the
    reason."""
    import errno
    import os

    path = Path(value)
    if not path.is_absolute():
        path = Path.cwd() / path
    if path.is_dir():
        raise ValueError(f"{path} is a directory")
    if not path.parent.is_dir():
        raise ValueError(f"the directory {path.parent} does not exist")
    if not os.access(path if path.exists() else path.parent, os.W_OK):
        raise ValueError(f"cannot write {path}: {os.strerror(errno.EACCES)}")
    return path


def _open_event_file(path, settings, closing):
    """The run's event file, opened into ``closing``. Each text is cleaned
    as the command line's display cleans it: the run's own credentials
    first, then the display's patterns."""
    from testuary.presentation import EventFile
    from testuary.presentation.view import safe_text
    from testuary.settings import _redact_diagnostic

    def clean(text, *, preserve_whitespace=False):
        return safe_text(_redact_diagnostic(text, settings), preserve_whitespace=preserve_whitespace)

    try:
        return closing.enter_context(EventFile(path, clean=clean))
    except OSError as exc:
        raise ConfigurationError(f"events: cannot write {path}: {exc.strerror or exc}") from exc


def _wanted(args, hook) -> bool:
    """Whether the run's setup report, plan or completion has a reader: the
    caller's ``hook`` or the run's event file."""
    return hook is not None or getattr(args, "_events", None) is not None


def _hand_on(args, kind, value, hook) -> None:
    """The run's setup report, plan or completion (``kind``) to its event
    file, if any, then to the caller's ``hook``, if set, so both receive the
    same record."""
    events = getattr(args, "_events", None)
    if events is not None:
        getattr(events, kind)(value)
    if hook is not None:
        hook(value)


_RESUME_SKIP_REASON = "already complete (resume skip)"


def _display_phases(args) -> tuple[str, ...]:
    """The generation phases each file run enters, run or restored: the run
    plan's phases and the possible request phases. Finishing, the fourth
    phase, follows them (``_counted_phases``)."""
    phases = ("starter", "coverage", "mutation")
    stop = getattr(args, "stop_after_phase", None)
    if stop in phases:
        phases = phases[:phases.index(stop) + 1]
    return phases


def _counted_phases(args) -> int:
    """The phases each file run counts in the progress totals: the
    generation phases it enters and Finishing, unless the run stops after
    an earlier phase."""
    return len(_display_phases(args)) + (getattr(args, "stop_after_phase", None) in (None, "finishing"))


def _selected_run_indices(args) -> tuple[int, ...]:
    """The runs this invocation makes of each file, as run indices (0 for
    r0): run K alone with ``--run K``, else r0 to r(N-1) with ``--runs N``."""
    run = getattr(args, "run", None)
    if run is not None:
        return (int(run),)
    return tuple(range(int(getattr(args, "runs", 1) or 1)))


def _display_repetitions(args) -> int:
    """Runs a file makes. A requested phase stop ends a file after its
    first run."""
    if getattr(args, "stop_after_phase", None):
        return 1
    return len(_selected_run_indices(args))


def _limit_value(value) -> "float | None":
    """A configured limit as a number: None for no limit (unlimited, empty or
    zero seconds)."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number > 0 else None


def run_plan(args, files, root=None) -> "RunPlan":
    """What the run will do, from its options and the settings it resolved:
    its files (relative to the source directory ``root``), its runs, its three
    phases with their states, steps and limits, and Finishing. A
    phase after --stop-after-phase is off, a reused one reused. The loops'
    steps are the graph modes' steps of their agents
    (``CodingAgent.MODE_STEPS``), each once: Assess is the step of two modes,
    with and without the edit tool after a STANDS ALONE note."""
    from testuary import starters
    from testuary.agent.coverage_agent import CoverageAgent
    from testuary.agent.mutation_agent import MutationAgent
    from testuary.presentation.events import PlannedLimit, PlannedPhase, PlannedStep, RunPlan
    from testuary.presentation.view import phase_label

    shown = _display_phases(args)
    reused = _reused_phases(args)

    def state(name: str) -> str:
        if name in reused:
            return "reused"
        return "planned" if name in shown else "off"

    def steps(agent) -> tuple:
        modes = [mode for mode in agent.MODE_STEPS
                 if (mode != agent.INITIAL_READ_MODE or agent.forces_initial_read)
                 and (mode != agent.INITIAL_MEASURE_MODE or agent.forces_initial_measurement)]
        return tuple(PlannedStep(*step)
                     for step in dict.fromkeys(agent.MODE_STEPS[mode] for mode in modes))

    # Every phase ends with the same consolidation, after a loop's cycle.
    phase_end = (PlannedStep(starters.PHASE_END, dict(starters.STEPS)[starters.PHASE_END]),)
    starter_end = [step for step, _label in starters.STEPS].index(starters.PHASE_END)

    def cycle(agent) -> int:
        # The start's modes run once. The cycle repeats after them.
        return sum(1 for step in steps(agent) if step.step in (
            agent.MODE_STEPS[agent.INITIAL_READ_MODE][0], agent.MODE_STEPS[agent.INITIAL_MEASURE_MODE][0]))

    def limits(section: str, goal: str) -> tuple:
        config = config_module.get_agent_config(section)
        return (PlannedLimit("goal", _limit_value(config.get(goal)), "%"),
                PlannedLimit("request limit", _limit_value(config.get("max_requests_per_phase")), "requests"),
                PlannedLimit("time limit", _limit_value(config.get("max_wall_clock_seconds_per_phase")), "s"))

    def name(item) -> str:
        path = PurePosixPath(str(item.get("file_path", "")).replace(chr(92), "/"))
        try:
            return path.relative_to(PurePosixPath(str(root).replace(chr(92), "/"))).as_posix()
        except ValueError:
            return path.as_posix()

    starter = config_module.get_agent_config("test_initial")
    phases = (
        PlannedPhase("starter", phase_label("starter"), state("starter"),
                     tuple(PlannedStep(step, label) for step, label in starters.STEPS),
                     limits=(PlannedLimit("starter suites", _limit_value(starter.get("suite_count")),
                                          "suites"),),
                     # The starter phase's steps run once each: no cycle.
                     cycle=starter_end, end=starter_end),
        PlannedPhase("coverage", phase_label("coverage"), state("coverage"),
                     steps(CoverageAgent) + phase_end, limits("test_coverage", "goal"),
                     cycle=cycle(CoverageAgent), end=len(steps(CoverageAgent))),
        PlannedPhase("mutation", phase_label("mutation"), state("mutation"),
                     steps(MutationAgent) + phase_end, limits("test_mutation", "goal"),
                     cycle=cycle(MutationAgent), end=len(steps(MutationAgent))),
    )
    stop = getattr(args, "stop_after_phase", None)
    finishing = PlannedStep(
        "finishing", phase_label("finishing"),
        "reused" if "finishing" in reused
        else "off" if stop in shown else "planned")
    # A stop ends a file after its first selected repetition.
    runs = (project.run_name(_selected_run_indices(args)[0]),) if stop else tuple(
        project.run_name(index) for index in _selected_run_indices(args))
    return RunPlan(files=tuple(name(item) for item in files), runs=runs, phases=phases,
                   finishing=finishing)


def _generation_model_phases(args) -> tuple[str, ...]:
    """Possible request phases. Resume or goal checks may still avoid them."""
    reused = _reused_phases(args)
    return tuple(phase for phase in _display_phases(args) if phase not in reused)


def _resolve_generation_models(args, settings):
    """Resolve captured request inputs without importing or constructing clients.

    The credential is private request input, not an exported report field.
    This common resolution also serves execution's client preflight.
    """
    from testuary.models import (
        catalogue_problems, resolve_model_identity, resolve_phase_model, resolve_provider_credential,
    )
    phases = _generation_model_phases(args)
    if not phases:
        return {}, (), None
    problems = list(catalogue_problems(settings.models))
    resolved = {}
    api_key = None
    # A problem common to several phases, such as the model selection, is
    # reported once with the phases it affects.
    phase_problems: Dict[str, List[str]] = {}
    for phase in phases:
        try:
            resolved[phase] = resolve_phase_model(settings, phase)
        except ConfigurationError as exc:
            phase_problems.setdefault(str(exc), []).append(phase)
    problems += [f"{', '.join(names)}: {message}" for message, names in phase_problems.items()]
    try:
        identity = resolve_model_identity(settings)
    except ConfigurationError:
        # Every selected phase already reports this common selection problem.
        pass
    else:
        try:
            api_key = resolve_provider_credential(
                identity.provider, settings.models.get(identity.provider, {}),
                settings.models.get(identity.model_alias, {}), settings.environment,
            )
        except ConfigurationError as exc:
            problems.append(f"credentials: {exc}")
    return resolved, tuple(problems), api_key


def _preflight_model_configuration(args) -> tuple[str, ...]:
    """Resolve request inputs, then prepare the client factories' clients before
    project work."""
    from testuary import models as _model_source
    args._prepared_models = None
    settings = config_module.active_settings()
    if settings is None or _model_source.current() is not None:
        return ()
    resolved, problems, api_key = _resolve_generation_models(args, settings)
    if problems:
        return problems
    if not resolved:
        return ()

    from testuary.models import (
        StarterModelClients, PreparedModelClients, build_resolved_chat_model,
    )

    problems = []
    prepared = PreparedModelClients(config_module.active_run_scope())
    for phase, model_settings in resolved.items():
        if model_settings.protocol != "added":
            continue
        try:
            if phase == "starter":
                cfg = config_module.get_agent_config("test_initial")
                clients = StarterModelClients(model_settings, api_key=api_key)
                seed_base = cfg["seed_base"]
                for run_idx in _selected_run_indices(args):
                    for slot in range(1, int(cfg["suite_count"]) + 1):
                        clients.get(seed=config_module.run_seed(seed_base, run_idx, slot),
                                    max_output_tokens=model_settings.values["max_output_tokens"])
                prepared.add(model_settings, clients)
            else:
                from testuary.agent import coverage_agent, mutation_agent

                cls = coverage_agent.CoverageAgent if phase == "coverage" else mutation_agent.MutationAgent
                model = build_resolved_chat_model(model_settings, api_key=api_key)
                # Tool signatures and descriptions need no target, session or
                # prompts. Preflight binds them but never executes them.
                agent = cls(prompts={}, deps=None, session=None)
                agent.preflight_tool_binding(model)
                prepared.add(model_settings, model)
        except ConfigurationError as exc:
            problems.append(f"{phase}: {exc}")
    if not problems:
        args._prepared_models = prepared or None
    return tuple(problems)


def _preflight_generation_reuse(args, top_files, results_dir, projects_root):
    """Check requested generation origins before output cleanup or work copies."""
    from testuary.evidence import current_condition, generation_only, phase_origin_path, require_origin
    from testuary import starters

    if config_module.active_settings() is None:
        return
    # A resume reuses the saved starter suites as reuse_suites does.
    suite_reuse = config_module.get_agent_config("test_initial")["reuse_suites"] or bool(args.resume)
    reused_phases = _reused_phases(args) - {"finishing"}
    if not (suite_reuse or args.resume or reused_phases):
        return
    for target in top_files:
        original = (PurePosixPath(target["file_path"]) if config_module.project_exec_mode() == "container"
                    else Path(target["file_path"]).resolve())
        target_dir = results_dir / project.target_workspace_name(original, projects_root)
        language = languages.detect_language(str(original))
        root = str(projects_root)
        reused = set(reused_phases)
        # A mutation reuse consumes only an archived mutation phase.
        if ("mutation" in reused
                and not (project.run_dir(target_dir, 0) / "archive" / "mutation").exists()):
            reused.discard("mutation")
        # A reuse consumes the first run's origins. A resume consumes, in
        # each selected run, the origins of the phases that run completed.
        consumed = {phase: {0} for phase in reused}
        if args.resume:
            for run_idx in _selected_run_indices(args):
                skipped = phase_state.resume_generation_skipped(args, run_idx)
                done, prefix, _ = phase_state.plan_test_resume(
                    target_results_dir=target_dir, rel_target=original.relative_to(projects_root),
                    canonical_path=original,
                    generation_skipped=skipped, target_number=target.get("target_number"),
                    run_idx=run_idx,
                )
                for phase in ("starter", "coverage", "mutation"):
                    if (done and phase not in skipped) or prefix[phase]:
                        consumed.setdefault(phase, set()).add(run_idx)
        warnings = []
        for phase, run_indices in consumed.items():
            condition = current_condition(phase, language, root)
            for run_idx in sorted(run_indices):
                warnings += [(phase, text) for text in
                             require_origin(phase_origin_path(target_dir, phase, run_idx), condition)]
        if suite_reuse:
            # Each run reuses its own saved starter pool.
            for run_idx in _selected_run_indices(args):
                suites = project.run_dir(target_dir, run_idx) / "archive" / "starter" / "suites"
                if starters._has_persisted_pool(suites):
                    warnings += [("starter", text) for text in require_origin(
                        suites / "generation_origin.json",
                        generation_only(current_condition("starter", language, root)))]
        for phase, text in dict.fromkeys(warnings):
            log_print(f"WARNING: {target_dir.name}, {phase} results: {text}. The provider and "
                      "model name match, so they are reused.", level=0)


class _Invocation:
    """What the files of one invocation share: its options, the project and
    the directories, the run timer, and the files that did not finish.

    ``unfinished`` holds an entry per such file for the end-of-run summary:
    {"target_number", "file_path", "reason", "failed", "below_floor"}. A
    file a resume skips as complete and a result (a suite below its floor,
    a phase that ended without a suite) are recorded too, so the summary
    shows every file's disposition. Only a failure to execute sets
    "failed", and with it the command's exit status."""

    def __init__(self, args, *, project_name: str, projects_root, container_run: bool,
                 results_dir: Path, logs_dir: Path, work_root: Path, file_count: int,
                 reused: set) -> None:
        self.args = args
        self.project_name = project_name
        # Resolved. In container mode a pure POSIX path, read and measured in
        # the container.
        self.projects_root = projects_root
        self.container_run = container_run
        self.results_dir = results_dir
        self.logs_dir = logs_dir
        self.work_root = work_root
        self.file_count = file_count
        self.reused = reused
        # Each run's pipeline segment, opened when the run starts and closed
        # when it ends, so that a run's row holds the time of that run's own
        # work.
        self.run_timer = _RunTimer()
        self.unfinished: List[Dict[str, Any]] = []

    def record_unfinished(self, target_number, file_path, reason: str, *, failed: bool,
                          below_floor: bool = False) -> None:
        self.unfinished.append({
            "target_number": target_number,
            "file_path": str(file_path),
            "reason": reason,
            "failed": failed,
            # A suite below its floor is a finished result: a resume skips it.
            "below_floor": below_floor,
        })

    def locate(self, top_file):
        """Where a target file's target sits, and what its workspace is called.

        Every target lies in the source directory, since a target outside it is
        refused before the run. The loop over the files needs the results
        dir before the target runs, so that it can time the whole of it.
        Deriving it here keeps that answer and the file run's own answer the
        same one."""
        # The target's source: a file run points file_path at its work copy
        # and keeps the source under canonical_path.
        source = top_file.get("canonical_path") or top_file["file_path"]
        if self.container_run:
            # In-container POSIX target: keep it pure so a Windows host does
            # not prepend a drive. It is read + measured in the container. The
            # host-side copy is gated off for container runs.
            canonical_path = PurePosixPath(source)
        else:
            canonical_path = Path(source).resolve()
        return (
            canonical_path,
            canonical_path.relative_to(self.projects_root),
            project.target_workspace_name(canonical_path, self.projects_root),
        )

    def display_file(self, top_file) -> str:
        # Display only: a target that cannot be located is recorded by the
        # per-file guard, not here.
        try:
            _, rel, _ = self.locate(top_file)
        except Exception:
            rel = None
        return rel.as_posix() if rel is not None else str(top_file.get("file_path"))


#: What one run of a file tells the file's loop: go on with the next run,
#: stop the runs and finish the file, or end the file at once.
_NEXT_RUN, _STOP_RUNS, _END_FILE = "next", "stop", "end"


class _FileRun:
    """One target file of an invocation: its resume plan, a fresh work copy
    of the project for each selected run (``--runs N`` or ``--run K``), the
    test pipeline, each run's tests.csv row and the delivered tests. Run
    index 0 writes into r0/, the next into r1/ (project.run_dir). A file
    that does not finish is recorded in the invocation's ``unfinished``."""

    def __init__(self, invocation: _Invocation, top_file) -> None:
        self.inv = invocation
        self.args = invocation.args
        self.top_file = top_file
        self.target_number = top_file["target_number"]
        self.canonical_path, self.rel_target, self.workspace = invocation.locate(top_file)

    def _unfinished(self, reason: str, *, failed: bool, below_floor: bool = False) -> None:
        self.inv.record_unfinished(self.target_number, self.canonical_path, reason,
                                   failed=failed, below_floor=below_floor)

    def execute(self) -> None:
        inv, target_number, canonical_path = self.inv, self.target_number, self.canonical_path
        self.work_dir = inv.work_root / self.workspace
        self.work_src = self.work_dir / inv.project_name
        self.work_tests = self.work_dir / "tests"
        work_target = self.work_src / self.rel_target

        # Container mode: the repo lives in the container, so there is no host
        # copy. The target is measured (and read) in place at REPO_DIR/<file>.
        # The generated tests live in the shared work directory, which the container
        # mounts, at the same path where that path is valid inside the
        # container, in a directory of the target's own with its tests and
        # originals, as outside a container. work_src stays unused.
        in_container = config_module.project_exec_mode() == "container"
        if in_container:
            work_dir = project.project_work_dir()
            if not work_dir:
                log_print(
                    f"ERROR: target {target_number} runs in container mode but "
                    f"TESTUARY_PROJECT_WORK_DIR is unset. Skipping it.",
                    level=0,
                )
                self._unfinished("container mode without a work directory", failed=True)
                return
            self.work_tests = Path(work_dir) / self.workspace / "tests"
            work_target = canonical_path

        self.target_results_dir = inv.results_dir / self.workspace
        self.test_log_target_dir = inv.logs_dir / "test" / self.workspace
        self.test_log_target_dir.mkdir(parents=True, exist_ok=True)

        log_print("")
        log_print("#" * 60)
        log_print(
            f"# File {target_number} / {inv.file_count} "
            f"- {canonical_path.name}  (workspace: {self.workspace})"
        )
        log_print("#" * 60)

        # Resume planning. Computed before the expensive project copy so
        # a fully-complete target is skipped without paying the copy. Each
        # selected run resumes from what it completed itself: run_plans
        # maps its run index to (complete, skips, restore_tests).
        self.run_indices = _selected_run_indices(self.args)
        self.n_tests = len(self.run_indices)
        run_plans: Dict[int, Tuple[bool, Dict[str, bool], bool]] = {}
        if self.args.resume:
            for planned_run in self.run_indices:
                run_plans[planned_run] = phase_state.plan_test_resume(
                    target_results_dir=self.target_results_dir,
                    rel_target=self.rel_target,
                    canonical_path=canonical_path,
                    generation_skipped=phase_state.resume_generation_skipped(self.args, planned_run),
                    log=log_print,
                    target_number=target_number,
                    run_idx=planned_run,
                )
            if all(plan[0] for plan in run_plans.values()):
                log_print(
                    f"Target {target_number} ({canonical_path.name}) is verifiably "
                    f"complete (all phases validated) "
                    f"and is skipped. Its CSV rows and deliverables are kept.",
                    level=1,
                )
                self._unfinished(_RESUME_SKIP_REASON, failed=False)
                return

        # A fresh copy of the project tree for the target.
        if not self._copy_project():
            return

        # Override paths so the agent only ever sees the work tree.
        self.top_file["canonical_path"] = str(canonical_path)
        self.top_file["file_path"] = str(work_target)

        # The agent's source directory: the container's repo for a container
        # target, whose host work tree does not exist there. The work tree
        # otherwise.
        target_source_dir = str(inv.projects_root if in_container else self.work_src)

        config_module.set_invocation_source_dir(target_source_dir)

        # The target's phases, per selected run (target_phases.run_target_phases): the
        # starter phase, the coverage phase, the mutation phase and
        # finishing, in one mutation engine session around the mutation
        # phase. Each run starts from a fresh copy of the project
        # source and writes into its own run directory.
        self.coverage_ever_succeeded = False
        self.last_pipeline_result = None
        self.crashed = False
        # Why the target delivered nothing, when a phase found no suite passing
        # as delivered at its entry.
        self.failure_reason = None
        # A reuse a phase refused, which is a failure, not a result.
        self.refused = False
        # The copy made above stays fresh until a run uses it.
        self.fresh_copy = True
        # With several runs, a reuse gives every run the first run's tests as
        # they were before this invocation, since run r0 replaces them with
        # its own suite before a later run starts.
        self.reused_tests = (
            _tests_snapshot(project.phase_artifact_path(self.target_results_dir, "tests", 0))
            if "coverage" in inv.reused and self.n_tests > 1 else None
        )

        for test_run_idx in self.run_indices:
            step = self._run(test_run_idx, run_plans.get(test_run_idx))
            if step == _END_FILE:
                return
            if step == _STOP_RUNS:
                break
        self._finish()

    def _copy_project(self, run_label: Optional[str] = None) -> bool:
        """Wipe the target's copy and copy the source directory again, so a run
        starts from the source's state. False, recorded, when the copy
        fails. The first copy's duration is logged because it can be slow
        for a large project on a network filesystem."""
        inv = self.inv
        _remove_path(self.work_dir)
        copy_start = time.perf_counter()
        for_run = "" if run_label is None else f" for repetition {run_label}"
        try:
            _copy_project_to_work(
                projects_root=inv.projects_root,
                work_src=self.work_src,
            )
        except Exception as exc:
            log_print(
                f"ERROR: failed to copy the source directory {inv.projects_root} "
                f"into {self.work_src}{for_run}: {type(exc).__name__}: {exc}",
                level=0,
            )
            self._unfinished(f"project-copy failed{for_run}: {type(exc).__name__}", failed=True)
            return False
        if run_label is None:
            copy_secs = time.perf_counter() - copy_start
            log_print(
                f"[copy] {inv.projects_root} -> {self.work_src} "
                f"took {copy_secs:.1f}s",
                level=1,
            )
        self.work_tests.mkdir(parents=True, exist_ok=True)
        return True

    def _extract(self, run_idx: int, run_label: Optional[str] = None) -> None:
        """Copy a run's tests into its results directory. With several runs
        each run's suite is extracted right after the run, so the next run's
        wipe does not lose it."""
        try:
            _extract_tests_deliverable(
                work_tests=self.work_tests,
                target_results_dir=self.target_results_dir,
                run_idx=run_idx,
            )
        except Exception as exc:
            of_run = "" if run_label is None else f" repetition {run_label}"
            log_print(
                f"ERROR: failed to extract tests for target "
                f"{self.target_number}{of_run}: {type(exc).__name__}: {exc}",
                level=0,
            )
            self._unfinished(
                (f"tests of repetition {run_label} not delivered" if run_label is not None
                 else "tests not delivered") + f": {type(exc).__name__}",
                failed=True)

    def _run(self, test_run_idx: int, run_plan) -> str:
        from testuary.target_phases import ReuseFlags, TargetPhasesContext, run_target_phases

        inv, args, target_number, canonical_path = self.inv, self.args, self.target_number, self.canonical_path
        reused_by_flag, n_tests = inv.reused, self.n_tests
        run_label = project.run_name(test_run_idx)
        if run_plan is not None and run_plan[0]:
            # This run completed before. Its CSV row and tests stay.
            log_print(f"Target {target_number} ({canonical_path.name}) repetition {run_label} "
                      f"is verifiably complete and kept.", level=1)
            self.coverage_ever_succeeded = True
            return _NEXT_RUN
        inv.run_timer.start(self.target_results_dir, test_run_idx)
        if not self.fresh_copy and not self._copy_project(run_label):
            return _STOP_RUNS
        self.fresh_copy = False

        run_resume_skips = run_plan[1] if run_plan is not None else phase_state.no_resume_skips()
        resumed_label = f"Target {target_number}" if n_tests == 1 else f"Target {target_number} repetition {run_label}"
        skipped = [k for k, v in run_resume_skips.items() if v]
        if skipped:
            log_print(
                f"{resumed_label}: resuming. Phases verifiably "
                f"complete and restored from disk: "
                f"{', '.join(skipped)}. Running the rest.",
                level=1,
            )
        # Resume: repopulate the target's copy with this run's
        # verifiably-complete test artifacts so the orchestrator's
        # restore-from-disk path (driven by the skip flags below) finds
        # them. Only the validated prefix was kept. The rest is
        # regenerated.
        if run_plan is not None and run_plan[2]:
            _restored = _restore_preserved_tests(
                workspace=self.workspace,
                results_dir=inv.results_dir,
                work_tests=self.work_tests,
                run_idx=test_run_idx,
            )
            if not _restored:
                log_print(
                    f"WARNING (target {target_number}): --resume found no saved tests at "
                    f"{project.phase_artifact_path(self.target_results_dir, 'tests', test_run_idx)}, "
                    f"so the later phases run again.",
                    level=1,
                )
                run_resume_skips = phase_state.no_resume_skips()

        # If the user is reusing tests from a prior run, restore them now.
        # --reuse coverage and --reuse mutation (which reuses the
        # coverage phase too) need
        # the first run's test file on disk before the orchestrator
        # decides which phase to run versus restore. The
        # ``_preflight_reuse_requirements`` check already verified
        # every requested target's preserved-tests directory exists, so
        # a restoration failure here is an unexpected disk-level
        # problem (permissions / I/O), not operator misuse.
        if "coverage" in reused_by_flag:
            if self.reused_tests is not None:
                self.work_tests.mkdir(parents=True, exist_ok=True)
                restored = _restore_tests_snapshot(self.reused_tests, self.work_tests)
            else:
                restored = _restore_preserved_tests(
                    workspace=self.workspace,
                    results_dir=inv.results_dir,
                    work_tests=self.work_tests,
                )
            if not restored:
                log_print(
                    f"ERROR (target {target_number}): saved tests at "
                    f"{project.phase_artifact_path(inv.results_dir / self.workspace, 'tests', 0)} were "
                    f"present at pre-flight but could not be restored "
                    f"(permissions / I/O). Skipping this target.",
                    level=0,
                )
                self._unfinished("preserved tests could not be restored", failed=True)
                _remove_path(self.work_dir)
                return _END_FILE
        if test_run_idx > 0:
            _copy_first_run_archives(args, self.target_results_dir, test_run_idx)

        log_print("=" * 60)
        log_print(f">>> PHASE START: test pipeline of {canonical_path.name}"
                  + (f" (repetition {run_label})" if n_tests > 1 else ""))
        log_print("=" * 60)

        run_log_dir = project.run_dir(self.test_log_target_dir, test_run_idx)
        run_log_dir.mkdir(parents=True, exist_ok=True)
        reuse = {phase: phase in reused_by_flag or run_resume_skips[phase] for phase in PHASES}
        pipeline_result = _run_target_phases_guarded(TargetPhasesContext(
            project_name=inv.project_name,
            workspace=self.workspace,
            top_file=self.top_file,
            canonical_path=canonical_path,
            work_tests=self.work_tests,
            results_dir=inv.results_dir,
            target_logs_dir=run_log_dir,
            run_idx=test_run_idx,
            resumed=frozenset(k for k, v in run_resume_skips.items() if v),
            reuse=ReuseFlags(**reuse),
            log_print=log_print,
            stop_after_phase=getattr(args, "stop_after_phase", None),
            settings=config_module.active_settings(),
            progress=getattr(args, "_progress", None),
            display_target=self.rel_target.as_posix(),
            prepared_models=getattr(args, "_prepared_models", None),
            resume=bool(getattr(args, "resume", False)),
        ),
            run_phase=run_target_phases,
            results_dir=inv.results_dir,
            workspace=self.workspace,
            target_number=target_number,
            file_path=str(canonical_path),
            run_idx=test_run_idx,
            log=log_print,
        )
        if pipeline_result is None:
            # The test pipeline crashed. The guard logged it and wrote a
            # crashed tests.csv row. Abandon this target and move on.
            self.crashed = True
            return _STOP_RUNS

        requested_stop = getattr(args, "stop_after_phase", None)
        if (
            requested_stop
            and pipeline_result.stopped_after_phase is None
        ):
            raise _PhaseStopNotReached(
                f"the {requested_stop} phase did not finish, so the run could not "
                f"stop after it"
            )

        if pipeline_result.stopped_after_phase is not None:
            log_print(
                "<<< PHASE STOP: test pipeline, "
                f"after={pipeline_result.stopped_after_phase}"
            )
            _remove_path(self.work_dir)
            return _END_FILE

        # A reuse the phase refused (its restored suite missing, not
        # collecting, failing as delivered or unlike its record) is a failure
        # to execute what was asked, not a result.
        outcomes = pipeline_result.phase_outcomes
        refused = next(((phase, outcomes.get(f"{phase}_detail") or "its saved results could not be restored")
                        for phase in ("coverage", "mutation")
                        if reuse[phase] and outcomes.get(phase) == "failed"), None)
        if refused is not None:
            self.refused = True
            self._unfinished(f"{refused[0]} reuse refused: {refused[1]}"
                             + (f" (repetition {run_label})" if n_tests > 1 else ""), failed=True)

        # A starter phase that kept the suites its validator did not
        # judge finishes the work it can, but the target failed to execute.
        unresolved = _unresolved_suite_validation(pipeline_result)
        if unresolved is not None:
            self._unfinished((f"{unresolved} (repetition {run_label})"
                              if n_tests > 1 else unresolved),
                             failed=True)

        test_result = pipeline_result.test_result
        mutation_result = pipeline_result.mutation_result
        finishing_result = pipeline_result.finishing_result

        reused_any_gen = bool(
            reused_by_flag - {"finishing"} or any(run_resume_skips.values()))

        # The coverage phase was reused or restored rather than run: a
        # row without its result then reads "reused", and otherwise no
        # coverage status stands for a phase that did not run.
        coverage_reused = bool(
            "coverage" in reused_by_flag or run_resume_skips["coverage"])
        # This file's and run's tests.csv row. After a reuse or a resume
        # it overlays the existing row instead of replacing it, so a
        # failed resume keeps a prior partially valid row's evidence.
        write_row = functools.partial(
            results.write_tests_row, results_dir=inv.results_dir, workspace=self.workspace,
            target_number=target_number, file_path=str(canonical_path), work_tests=self.work_tests,
            phase_durations=pipeline_result.phase_durations, run_idx=test_run_idx,
            preserve_existing=reused_any_gen, coverage_reused=coverage_reused, refused=refused)

        if pipeline_result.coverage_phase_failed:
            detail = pipeline_result.phase_outcomes.get(
                "coverage_detail",
            ) or pipeline_result.phase_outcomes.get(
                "starter_detail", "unknown",
            )
            from testuary.starters import OUTPUT_LIMIT_DETAIL

            if str(detail).startswith(OUTPUT_LIMIT_DETAIL):
                # Said in the run's summary, not only "no tests from the coverage phase".
                self.failure_reason = detail
            log_print(
                f"Mutation / metrics skipped for target "
                f"{target_number} repetition {run_label}: test generation "
                f"status = {detail}.",
                level=1, echo=True,
            )
            write_row(
                test_result=test_result,
                starter_result=getattr(
                    pipeline_result, "starter_result", None,
                ),
            )
            return _NEXT_RUN

        self.coverage_ever_succeeded = True
        self.last_pipeline_result = pipeline_result
        # Plumb final coverage / mutation back into test_result so the
        # CSV writer (and any downstream readers) see the
        # numbers from Finishing rather than the coverage phase's.
        if test_result is not None:
            updates: dict = {}
            if pipeline_result.final_line_coverage is not None:
                updates["line_coverage"] = pipeline_result.final_line_coverage
            if pipeline_result.final_mutation_score is not None:
                updates["mutation_score"] = pipeline_result.final_mutation_score
            if updates:
                test_result = test_result.model_copy(update=updates)

        # A phase-status summary that reflects which phases actually ran
        # versus were skipped.
        _test_parts = [f"run={run_label}"] if n_tests > 1 else []
        _test_parts.append("coverage=skipped" if "coverage" in reused_by_flag
                           else f"coverage={pipeline_result.final_line_coverage}")
        _test_parts.append("mutation=skipped" if "mutation" in reused_by_flag
                           else f"mutation={pipeline_result.final_mutation_score}")
        if "finishing" in reused_by_flag:
            _test_parts.append("metrics=skipped")
        log_print(
            "<<< PHASE END:   test pipeline, "
            + ", ".join(_test_parts)
        )

        # Stream the tests.csv row now. Coverage, mutation and Finishing
        # are all done, so it can record the coverage and mutation
        # counts and Finishing's numbers. When a phase was
        # reused (its result is None this run) or resume-skipped, the row
        # is seeded from the prior one so the recomputed metrics backfill
        # in place without blanking the skipped columns. A resume-restored
        # starter result carries complete authoritative generation and
        # consolidation accounting, so it can also create a missing CSV
        # row from scratch.
        write_row(
            test_result=None if "coverage" in reused_by_flag else test_result,
            # A starter phase reused by flag has no result this run.
            starter_result=(
                None if "starter" in reused_by_flag and not run_resume_skips["starter"]
                else pipeline_result.starter_result),
            mutation_result=mutation_result,
            finishing_result=finishing_result,
            below_minimum_reason=pipeline_result.below_minimum_reason,
            consolidation_stats=getattr(
                pipeline_result, "consolidation_stats", None),
        )

        if n_tests > 1:
            self._extract(test_run_idx, run_label)
        return _NEXT_RUN

    def _finish(self) -> None:
        """After the last run: the file's outcome and its delivered tests."""
        inv, target_number, canonical_path = self.inv, self.target_number, self.canonical_path
        # A single run's own remaining work (its extraction and cleanup) is
        # still that run's. With several runs each run extracted its suite in
        # the loop, and the rest is shared. A run the guard saw crash closes
        # as crashed: its work is discarded, and its remaining work is no
        # run's.
        inv.run_timer.stop(crashed=self.crashed)
        if self.n_tests == 1 and not self.crashed:
            inv.run_timer.start(self.target_results_dir, self.run_indices[0])

        if self.crashed:
            # The test pipeline crashed (the guard already logged it and
            # wrote the crashed tests.csv row). Record the abort and move on.
            self._unfinished("test pipeline crashed", failed=True)
            _remove_path(self.work_dir)
            return

        if not self.coverage_ever_succeeded:
            if not self.refused:
                self._unfinished(self.failure_reason or "no tests from the coverage phase",
                                 failed=False)
            _remove_path(self.work_dir)
            return

        pipeline_result = self.last_pipeline_result
        if pipeline_result is not None and pipeline_result.below_minimum_reason is not None:
            # The suite ended below the configured coverage / mutation floor.
            # The tests are still a valid deliverable: record the shortfall and
            # extract them so the target's directory holds what was produced.
            reason = pipeline_result.below_minimum_reason
            log_print("=" * 60)
            log_print(
                f"Target {target_number} ({canonical_path.name}) ended below the "
                f"configured minimum: {reason}.", echo=True,
            )
            log_print("=" * 60)
            self._unfinished(reason, failed=False, below_floor=True)

        if self.n_tests == 1:
            self._extract(self.run_indices[0])
        _remove_path(self.work_dir)


def _run_pipeline(args, selected_languages):
    """Run the test-generation pipeline for one project.

    Invoked by ``run_pipeline`` inside the run's settings and console
    scope. Returns the process exit code: 1 when a target failed to
    execute, else 0."""
    # A mistyped method or policy setting stops the run before any target:
    # it would otherwise measure differently from what was configured, or
    # stop a phase at its first agent.
    try:
        config_module.consolidation_mutation_method()
        config_module.context_rebuild()
    except ValueError as exc:
        raise ConfigurationError(f"{exc}. No phase ran.") from exc

    model_problems = _preflight_model_configuration(args)
    if model_problems:
        raise ConfigurationError("the model settings were refused before any phase ran:\n"
                                 + "\n".join(f"  - {problem}" for problem in model_problems))

    outdir = project.output_dir()
    reused_by_flag = _reused_phases(args)
    keep_tests = _cleanup_preserve_flags(args)
    outdir = _admit_output(outdir, preserve=keep_tests)

    project_config = config_module.get_project_config()
    project_name = project_config["project_name"]
    target_repo = project_config["target_repo"]
    hint_profile = project_config["hint_profile"]

    container_run = config_module.project_exec_mode() == "container"
    if container_run:
        # The source lives inside the container at REPO_DIR. Use it as the
        # projects root so the per-target target resolves against the container's
        # repo. The host project copy is skipped (see _FileRun).
        repo_dir = project_config["source_dir"]
        if not repo_dir:
            raise ConfigurationError(
                "container mode requires --source-dir or TESTUARY_PROJECT_SOURCE_DIR "
                "(the source directory inside the container)")
        # Normalized once here, so the target guard and every file run
        # compare against the same root, also when the setting holds "..".
        projects_root = PurePosixPath(posixpath.normpath(repo_dir))
    else:
        projects_root = project.resolve_project_source_path(target_repo)
        if projects_root is None:
            raise ConfigurationError(
                f"source directory not found at {target_repo}: select an existing "
                "--source-dir (the run works on its own copy of the source "
                "directory and never changes the original)")
        # Outputs and work copies must not write into the source directory.
        _check_output_outside_source(outdir, projects_root)

    results_dir = outdir / "results"
    logs_dir = outdir / "logs"
    if not getattr(args, "source_files", None):
        raise ConfigurationError(
            "no source file: name them (testuary FILE ...), or set [project] source_files "
            "in the project's settings file with testuary setup project")
    source_targets = _source_file_top_files(args, projects_root=projects_root)
    top_files = _select_pipeline_targets(args, source_targets=source_targets)
    progress = getattr(args, "_progress", None)
    if progress is not None:
        progress.plan(files=len(top_files), phases=_counted_phases(args),
                      repetitions=_display_repetitions(args))
        on_plan = getattr(args, "_on_plan", None)
        if _wanted(args, on_plan):
            try:
                _hand_on(args, "plan", run_plan(args, top_files, projects_root), on_plan)
            except Exception as exc:
                # The plan is shown, never needed: a run goes on without it.
                log_print(f"[display] no run plan: {type(exc).__name__}: {exc}", level=2)
        progress.emit(target="all targets", phase="pipeline", operation="run", level="run",
                      state="running", message="Run planned", **progress.run_counts())

    # Fail fast, before any target, when a selected target's backend lacks an
    # operation or tool the configured workflow needs. check and verify use
    # the same selection, contexts and requirements. Guest runs are assessed
    # through their guest route, and unselected languages are not assessed.
    from testuary.languages import selected_groups
    from testuary.readiness import (
        distributor_refusal, refuse_unselected_languages, session_dir_refusal, workflow_requirements,
    )

    settings = config_module.active_settings()
    groups = selected_groups([str(target["file_path"]) for target in top_files], settings)
    refuse_unselected_languages(groups, selected_languages, settings)
    requirements = workflow_requirements(args)
    _preflight_run_dependencies({group.language for group in groups}, projects_root,
                                requirements, groups)
    # A language's mutation tool refuses an unusable session directory or worker
    # setting at its first measurement, after the starter requests (Python's
    # Cosmic Ray does). Refuse it now instead, by its adapter's checks. A
    # container's directory is the container's, and its engine refuses it.
    run_languages = {group.language for group in groups}
    if not container_run and "mutations.open_session" in requirements.operations:
        refusal = (session_dir_refusal(run_languages, create=True)
                   or distributor_refusal(run_languages))
        if refusal is not None:
            raise ConfigurationError(f"{refusal}. No phase ran.")
    origin_root = (PurePosixPath(projects_root.as_posix()) if container_run else projects_root.resolve())
    _preflight_generation_reuse(args, top_files, results_dir, origin_root)
    # The remaining refusals come before any output is created, so the same
    # command runs once their cause is corrected.
    if container_run:
        # The source lives in the container at a POSIX path. Resolving it on a
        # Windows host would prepend a drive letter and flip the separators. A
        # pure POSIX path stays faithful and carries no host-fs access. The
        # tree is read and measured in the container, with no host-side copy.
        projects_root_resolved = PurePosixPath(projects_root.as_posix())
    else:
        projects_root_resolved = projects_root.resolve()

    # Pre-flight: verify each reused phase's on-disk requirements.
    # Failing here surfaces missing preserved artifacts before any LLM
    # work happens, so an operator sees the full problem list at once
    # rather than discovering it one target at a time through
    # per-target ERROR logs.
    reuse_problems = _preflight_reuse_requirements(
        args, top_files, results_dir, projects_root_resolved,
    )
    if reuse_problems:
        raise ConfigurationError(
            f"--reuse cannot be applied to {len(reuse_problems)} target(s):\n"
            + "\n".join(f"  - {problem}" for problem in reuse_problems))

    main_cfg = config_module.load_config("main")

    # Ground-truth check: when the source directory is a git checkout,
    # verify each target file matches its committed HEAD. Drift means
    # the source was modified outside git (manual edit, stale
    # mutation residue, etc.). Default is warn-and-continue. Set
    # ``[main] strict_canonical_git = true`` to abort on any drift.
    strict_canonical_git = (
        str(main_cfg.get("strict_canonical_git", "false")).lower()
        in ("true", "1", "yes")
    )
    try:
        canonical_drift = _check_canonical_against_git(
            projects_root_resolved,
            [Path(tf["file_path"]) for tf in top_files],
        )
    except Exception as _git_exc:
        log_print(
            f"WARN: git-drift check raised "
            f"({type(_git_exc).__name__}: {_git_exc}). Skipping.",
            level=1,
        )
        canonical_drift = []
    if canonical_drift:
        for path, stat in canonical_drift:
            log_print(
                f"[canonical-drift] {path} differs from git HEAD: "
                f"{stat}",
                level=0,
            )
        if strict_canonical_git:
            # A refusal, not a finished run: the command exits with status 2.
            raise ConfigurationError(
                "the targets differ from git HEAD and [main] strict_canonical_git is "
                "true. Commit or stash the changes, or set strict_canonical_git = false to "
                "test the working-tree version")
        log_print(
            "The run tests the working-tree version of these targets. To refuse a run "
            "instead, set [main] strict_canonical_git = true.",
            level=0,
        )

    # Scope transient databases and test files to the supplied execution ID.
    job_ns = project.resolve_work_dir_namespace()
    work_root = outdir / "_work" / job_ns if job_ns else outdir / "_work"
    if job_ns:
        log_print(f"[main] work_dir namespace: {job_ns}", level=1)

    # Decide what the outdir cleanup must keep from the reuse / resume flags.
    # This is the data-loss-critical gate: every flag that reuses or resumes
    # an artifact keeps it. Fresh occupied destinations are refused.
    _cleanup_outdir(
        outdir=outdir,
        keep_tests=keep_tests,
    )

    results_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    _keep_results_out_of_version_control(outdir)

    main_logger = AgentLogger(
        project_name=project_name,
        agent_name="main",
        config={"target_repo": target_repo, "hint_profile": hint_profile},
    )
    main_logger.init()
    main_logger.enable_streaming(str(logs_dir), "main.jsonl")
    set_main_logger(main_logger)
    # Every log line of the run, readable while the display shows progress.
    run_log = open_plain_log(logs_dir / "run.log")
    if progress is not None:
        progress.log_path = run_log
        progress.emit(target="all targets", phase="pipeline", operation="run", level="run",
                      state="running", message="Writing the run log", log_path=run_log,
                      **progress.run_counts())

    # The one place a result is stamped with the code state that produced it.
    log_print(f"testuary results version {RESULTS_VERSION}")
    log_print(f"Project: {project_name}  (source: {projects_root})")
    log_print(f"Output:  {outdir}")

    invocation = _Invocation(
        args, project_name=project_name, projects_root=projects_root_resolved,
        container_run=container_run, results_dir=results_dir, logs_dir=logs_dir,
        work_root=work_root, file_count=len(top_files), reused=reused_by_flag)
    # Admission and every preflight have passed: the caller's pre-run summary
    # comes before the first model request. Showing it never changes the run.
    pre_run = getattr(args, "_pre_run", None)
    if _wanted(args, pre_run):
        try:
            request = getattr(args, "_request", None) or _pre_run_request(args)
            if request is not None:
                from testuary.readiness import pre_run_report

                _hand_on(args, "readiness", pre_run_report(request, settings), pre_run)
        except Exception as exc:
            log_print(f"WARNING: the pre-run summary could not be shown ({type(exc).__name__}: {exc})")

    finished_files = []
    on_complete = getattr(args, "_on_complete", None)
    if _wanted(args, on_complete):
        def complete_on_interrupt():
            # Files that ended keep their outcome. The interrupted file and
            # those not started yet stopped before finishing.
            ended = {id(entry[1]) for entry in finished_files}
            stopped = [(invocation.display_file(top_file), top_file, None, "cancelled", "", True)
                       for top_file in top_files if id(top_file) not in ended]
            _hand_on(args, "completion", _completion_facts(
                [*finished_files, *stopped], results_dir=results_dir, log_path=run_log,
                runs=_selected_run_indices(args), cancelled=True), on_complete)

        args._complete_on_interrupt = complete_on_interrupt
    for top_file in top_files:
        file_label = invocation.display_file(top_file)
        # Before the target runs: it points file_path at the work copy.
        try:
            workspace = invocation.locate(top_file)[2]
        except Exception:
            workspace = None
        recorded = len(invocation.unfinished)
        if progress is not None:
            progress.emit(target=file_label, phase="pipeline", operation="target", level="target",
                          label=file_label, state="running", message="Target started",
                          **progress.file_counts(file_label))
        try:
            _, _, _workspace = invocation.locate(top_file)
            # Around the driver rather than inside it, so the work the file's
            # runs share (resume planning, the first project copy, engine
            # teardown) is inside something measured. Each run's own work is
            # its pipeline segment inside (the invocation's run_timer).
            with phase_state.timed_segment(
                results_dir / _workspace, phase_state.SEGMENT_FILE,
            ):
                try:
                    _FileRun(invocation, top_file).execute()
                except BaseException:
                    invocation.run_timer.stop(crashed=True)
                    raise
                invocation.run_timer.stop()
            results.refresh_tests_row_durations(
                results_dir,
                workspace=_workspace,
                target_number=int(top_file.get("target_number")),
            )
            # The summary of each run and across the runs, from tests.csv.
            try:
                results.write_file_summaries(results_dir, _workspace, int(top_file.get("target_number")))
            except OSError as exc:
                log_print(f"WARNING: the run summaries of {file_label} could not be written "
                          f"({type(exc).__name__}: {exc})", level=0)
        except Exception as exc:
            if isinstance(exc, _PhaseStopNotReached):
                raise
            from testuary.model_requests import failure_traceback
            log_print(
                f"ERROR: target {top_file.get('target_number')} crashed unexpectedly "
                f"({type(exc).__name__}: {exc}). It is recorded, and the run "
                f"continues with the next target.",
                level=0,
            )
            log_print(failure_traceback(exc), level=1)
            invocation.record_unfinished(top_file.get("target_number"),
                                         # locate's source rule, without a call that
                                         # may be what raised.
                                         top_file.get("canonical_path") or top_file.get("file_path"),
                                         f"unexpected crash: {type(exc).__name__}",
                                         failed=True)
        entries = invocation.unfinished[recorded:]
        skipped = bool(entries) and all(entry["reason"] == _RESUME_SKIP_REASON for entry in entries)
        failed = any(entry["failed"] for entry in entries)
        reasons = ", and ".join(entry["reason"] for entry in entries)
        stop = getattr(args, "stop_after_phase", None)
        if failed:
            outcome, reason, resumable = "failed", reasons, True
        elif skipped:
            # A resume skipped it as validated: report the result it recorded.
            floor = _recorded_floor_reason(_tests_rows(results_dir).get(str(top_file.get("target_number")), []))
            outcome, reason, resumable = (("no_complete_suite", floor, False) if floor
                                          else ("already_complete", "", False))
        elif entries:
            outcome, reason = "no_complete_suite", reasons
            resumable = not any(entry.get("below_floor") for entry in entries)
        elif stop:
            # A requested phase boundary delivers no suite.
            outcome, reason, resumable = "stopped", f"stopped after {stop}, as requested", False
        else:
            outcome, reason, resumable = "complete", "", False
        finished_files.append((file_label, top_file, workspace, outcome, reason, resumable))
        if progress is not None:
            if failed:
                state, message, failure = "failed", f"Target failed to execute: {reasons}", "execution"
            elif entries and not skipped:
                state, message, failure = ("failed", f"No suite that meets the minimums: {reasons}",
                                           "outcome")
            elif outcome == "no_complete_suite":
                state, message, failure = (
                    "failed", f"No suite that meets the minimums (recorded): {reason}", "outcome")
            else:
                state, message, failure = (
                    "succeeded", "Already finished by an earlier run (resume)" if skipped
                    else "Target finished", None)
            progress.emit(target=file_label, phase="pipeline", operation="target", level="target",
                          label=file_label, state=state, message=message, failure=failure,
                          **progress.file_counts(file_label, finished=True, already_complete=skipped))
            progress.report_run()

    # Per-target abortion summary. Surfaces any target that did not produce a
    # complete suite, operator-intentional skips (target already complete via
    # resume) and error skips alike, so a long multi-target run cannot hide
    # errors that scrolled past mid-run.
    # A caller that shows the completion summary replaces these console
    # lines with it. The log keeps them either way.
    exit_status = _summarize_unfinished_targets(
        invocation.unfinished, log_print if on_complete is not None else _echoed_log)

    # Aggregate token usage across every phase that produced output.
    # Reads tests.csv so an operator can compare model cost across runs /
    # model swaps at a glance without dropping into the per-phase JSONL
    # sidecars.
    token_lines = _token_usage_lines(results_dir)
    if token_lines:
        log_print("")
        log_print("=" * 60)
        log_print("Token usage (prompt / completion):")
        log_print("=" * 60)
        for line in token_lines:
            log_print(line, level=1)

    # Active model-facing hints: the cells of the hints file that apply to
    # this run's profile. A user who forgets that a hint-file edit is
    # active could misread a change in results, so the log records what
    # applied. ``per_language[L]`` and
    # ``per_profile_per_language[P][L]`` cells apply only to files whose
    # detected language is ``L``. The summary lists them without claiming
    # every file received them.
    try:
        hints_payload = hints_module.load_hints()
        active_cells = _list_active_hint_cells(hints_payload, hint_profile)
    except Exception:
        active_cells = []
    if active_cells:
        log_print("")
        log_print("=" * 60)
        log_print(
            f"Model-facing hints active ({len(active_cells)} cell"
            f"{'s' if len(active_cells) != 1 else ''}):"
        )
        log_print("=" * 60)
        for cell in active_cells:
            log_print(f"  {cell}", level=1)
        log_print(
            "  (<phase>.per_language[L] and "
            "<phase>.per_profile_per_language[P][L] apply only to targets "
            "whose detected language is L. Each cell applies only in its "
            "own phase)",
            level=1,
        )

    main_logger = get_main_logger()
    if main_logger is not None:
        main_log_path = main_logger.save(str(logs_dir), "main.jsonl")
        if main_log_path and getattr(args, "display", "auto") == "debug":
            diagnostic_print(f"Main orchestration log: {main_log_path}")
    try:
        results.write_project_summary(results_dir)
    except OSError as exc:
        log_print(f"WARNING: the project's run summary could not be written ({type(exc).__name__}: {exc})")
    completion = _completion_facts(finished_files, results_dir=results_dir, log_path=run_log,
                                   runs=_selected_run_indices(args))
    # The run's last event states its outcome from these facts.
    args._completion = completion
    try:
        from testuary._progress import _DEMONSTRATION
        from testuary.presentation.terminal import write_completion

        # The end-of-run summary stays in the run directory, for every run.
        write_completion(completion, provenance="synthetic" if _DEMONSTRATION.get() else "observed")
    except Exception as exc:
        log_print(f"WARNING: the end-of-run summary could not be written ({type(exc).__name__}: {exc})")
    if _wanted(args, on_complete):
        try:
            _hand_on(args, "completion", completion, on_complete)
        except Exception as exc:
            log_print(f"WARNING: the completion summary could not be shown ({type(exc).__name__}: {exc})")
    return exit_status


def _tests_rows(results_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    rows: Dict[str, List[Dict[str, str]]] = {}
    try:
        for row in results._read_csv(results_dir / "tests.csv")[1]:
            rows.setdefault(str(row.get("target_number", "")), []).append(row)
    except OSError:
        pass
    return rows


def _recorded_floor_reason(rows) -> str:
    """The floor reason a file's tests.csv rows record, or "" for none."""
    reasons = [row.get("below_minimum_reason") or "" for row in rows]
    return ", ".join(dict.fromkeys(reason for reason in reasons if reason.strip()))


def _number(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if number == number and number not in (float("inf"), float("-inf")) else None


def _file_usage(rows):
    """Model use summed over a file's rows. A count any row lacks is unknown,
    and tokens are unknown when a response came without reported usage."""
    from testuary.presentation import Usage

    def total(*columns):
        values = [_number(row.get(column)) for row in rows for column in columns]
        return None if not values or any(value is None for value in values) else int(sum(values))

    phases = ("starter", "coverage", "mutation")
    unreported = total(*(f"{phase}_unknown_usage_requests" for phase in phases))
    uncached = total(*(f"{phase}_unknown_cache_requests" for phase in phases))
    tokens_known = unreported == 0
    return Usage(
        logical_requests=total(*(f"{phase}_logical_requests" for phase in phases)),
        physical_requests=total("starter_requests", "coverage_requests", "mutation_requests"),
        input_tokens=total(*(f"{phase}_input_tokens" for phase in phases)) if tokens_known else None,
        output_tokens=total(*(f"{phase}_output_tokens" for phase in phases)) if tokens_known else None,
        cached_input_tokens=(total(*(f"{phase}_cache_hit_tokens" for phase in phases))
                             if tokens_known and uncached == 0 else None),
    )


def _completion_facts(finished_files, *, results_dir: Path, log_path: str,
                      runs=(0,), cancelled: bool = False):
    """The end-of-run facts for a display: each file's outcome and reason,
    the delivered suite files that exist, its measured final values from
    tests.csv (one run only) and its model use. ``runs`` are the run
    indices this invocation made (0 for r0)."""
    from testuary.presentation import Metric
    from testuary.presentation.terminal import FileCompletion, RunCompletion

    rows_by_target = _tests_rows(results_dir)
    made = {str(run_idx) for run_idx in runs}
    files = []
    for label, top_file, workspace, outcome, reason, resumable in finished_files:
        rows = [row for row in rows_by_target.get(str(top_file.get("target_number")), [])
                if str(row.get("run", "")).strip() in made]
        suite = []
        if workspace is not None and outcome != "stopped":
            for run_idx in runs:
                tests_dir = project.phase_artifact_path(results_dir / workspace, "tests", run_idx)
                if tests_dir.is_dir():
                    # The delivered test sources (with a Python suite's
                    # conftest.py), not the measurement records beside them.
                    suite.extend(path.relative_to(results_dir).as_posix()
                                 for path in sorted(tests_dir.rglob("*")) if path.is_file()
                                 and "__pycache__" not in path.parts
                                 and languages.detect_language(str(path)) != "unknown")
        metrics = []
        if len(rows) == 1:
            for column, name in (("line_coverage", "Line coverage"), ("branch_coverage", "Branch coverage"),
                                 ("mutation_score", "Mutation score")):
                value = _number(rows[0].get(column))
                if value is not None:
                    metrics.append(Metric(name, value, "%"))
        language = languages.detect_language(str(top_file.get("file_path", "")))
        fallbacks = tuple(f"{fallback.family}: {fallback.fallback}"
                          for fallback in languages.applied(language) if fallback.changes_method)
        # Each run's warnings as its summary lists them: the time limits it
        # reached and the loops a limit stopped before their goal.
        warnings = tuple(
            (f"{project.run_name(run_idx)}: " if len(runs) > 1 else "") + entry["text"]
            for run_idx in runs if workspace is not None
            for entry in results.warnings(project.run_dir(results_dir / workspace, run_idx)))
        files.append(FileCompletion(label, outcome, reason, tuple(suite), tuple(metrics),
                                    _file_usage(rows) if rows else None, resumable=resumable,
                                    fallbacks=fallbacks, warnings=warnings))
    return RunCompletion(tuple(files), results_directory=str(results_dir), log_path=log_path or None,
                         cancelled=cancelled)


# The phases in dependency order: the skip-flag keys, ReuseFlags' fields
# and the values of --reuse and --stop-after-phase.
PHASES = phase_state.PHASES


def _strings(values, name, *, normalize=False):
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must be a sequence of strings, not a single string")
    try:
        copied = tuple(values)
    except TypeError:
        raise ValueError(f"{name} must be a sequence of strings") from None
    if any(not isinstance(value, str) or not value.strip() or "\0" in value for value in copied):
        raise ValueError(f"{name} must contain nonempty strings")
    return tuple(value.strip().lower() for value in copied) if normalize else copied


def check_run_options(*, runs=1, run=None, reuse=(), resume=False, stop_after_phase=None,
                      languages=None):
    """Check a run's options together, as the command line and RunRequest
    both pass them, and return the reused phases and the languages, normalized.
    Raises ValueError naming what is wrong."""
    if type(runs) is not int or runs < 1:
        raise ValueError(f"--runs must be 1 or more, not {runs!r}")
    if run is not None:
        if type(run) is not int or run < 0:
            raise ValueError(f"--run must be a repetition number from 0, not {run!r}")
        if runs != 1:
            raise ValueError("--run K and --runs N cannot be combined: --run makes one "
                             "repetition, --runs makes N")
    if type(resume) is not bool:
        raise ValueError("resume must be a boolean")
    reuse = tuple(dict.fromkeys(_strings(reuse, "reuse", normalize=True)))
    unknown = [phase for phase in reuse if phase not in PHASES]
    if unknown:
        raise ValueError(f"--reuse: unknown value {', '.join(unknown)}. Choose from "
                         f"{', '.join(PHASES)}")
    if resume and set(reuse) - {"starter"}:
        raise ValueError("--resume works only with --reuse starter")
    if stop_after_phase is not None and stop_after_phase not in PHASES:
        raise ValueError(f"--stop-after-phase must be one of {', '.join(PHASES)}")
    if "finishing" in reuse and (runs > 1 or (run or 0) > 0):
        raise ValueError("--reuse finishing skips Finishing, which only a run itself "
                         "makes: it works with the first repetition alone")
    if stop_after_phase is not None and runs > 1:
        raise ValueError("--stop-after-phase works with one repetition: drop --runs")
    if languages is not None:
        from testuary.languages import available_language_names

        available = available_language_names()
        languages = tuple(dict.fromkeys(_strings(languages, "languages", normalize=True)))
        unknown = [name for name in languages if name not in available]
        if not languages or unknown:
            raise ValueError(f"--languages: unknown language "
                             f"{', '.join(unknown) or '(none named)'}. Installed: "
                             f"{', '.join(available)}")
    return reuse, languages


@dataclass(frozen=True)
class RunRequest:
    """Targets and execution policy for ``run_pipeline(..., settings=...)``.

    Source paths are individual strings, never comma-delimited CLI text.
    The model, the project, its runtime and the limits belong to RunSettings.

    ``runs`` runs each target's four phases that many times, as repetitions
    r0 to r(runs-1), each from a fresh copy of the project and in its own
    directory. ``run`` runs only repetition K (rK) instead, the same one
    ``runs`` makes when it is above K, so a missing repetition can be added alone.

    ``targets`` runs only some of ``source_files``, by position, as
    ``--targets`` does, such as ``"1-5"``. ``reuse`` names the phases whose
    saved results the run takes, as ``--reuse`` does, and ``resume``
    continues an interrupted run in the same run directory, as ``--resume``
    does. ``stop_after_phase`` names the phase after which each target
    stops, as ``--stop-after-phase`` does. ``languages`` lists the languages
    a target may be in, whose tools are checked before the run, as
    ``--languages`` does.
    """

    source_files: tuple[str, ...]
    targets: str | None = None
    runs: int = 1
    run: int | None = None
    reuse: tuple[str, ...] = ()
    resume: bool = False
    stop_after_phase: str | None = None
    languages: tuple[str, ...] | None = None

    def __post_init__(self):
        sources = _strings(self.source_files, "source_files")
        if not sources or len(sources) > 100:
            raise ValueError("source_files must contain between 1 and 100 targets")
        if len(set(sources)) != len(sources):
            raise ValueError("source_files contains a duplicate target")
        object.__setattr__(self, "source_files", sources)
        if self.targets is not None:
            if not isinstance(self.targets, str) or not self.targets.strip():
                raise ValueError("targets must be a nonempty file selector or None")
            from testuary.config import parse_targets_spec

            parse_targets_spec(self.targets, total=len(sources))
        reuse, languages = check_run_options(
            runs=self.runs, run=self.run, reuse=self.reuse, resume=self.resume,
            stop_after_phase=self.stop_after_phase, languages=self.languages)
        object.__setattr__(self, "reuse", reuse)
        object.__setattr__(self, "languages", languages)


def _invocation(request: RunRequest) -> SimpleNamespace:
    """One-way adapter to the existing engine, not a second configuration."""
    return SimpleNamespace(
        source_files=request.source_files, targets=request.targets,
        runs=request.runs, run=request.run, resume=request.resume,
        stop_after_phase=request.stop_after_phase,
        **{f"reuse_{name}": name in request.reuse for name in PHASES},
    )


def _run_request(args) -> RunRequest:
    """The invocation as a RunRequest. Raises ValueError or TypeError when
    the arguments do not form one."""
    return RunRequest(
        source_files=tuple(results.parse_csv_field(args.source_files)
                           if isinstance(args.source_files, str)
                           else args.source_files or ()),
        targets=args.targets, runs=args.runs, run=args.run, resume=args.resume,
        reuse=tuple(results.parse_csv_field(args.reuse or "")), stop_after_phase=args.stop_after_phase,
        languages=tuple(results.parse_csv_field(args.languages)) if args.languages else None,
    )


def _pre_run_request(args):
    """The generate invocation as a RunRequest for its pre-run summary, or
    None when the arguments do not form one (the run then refuses itself)."""
    try:
        return _run_request(args)
    except (ValueError, TypeError):
        return None
