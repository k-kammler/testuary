"""Read-only views of a run directory, from the files Testuary documents.

A directory is a run directory (``results/`` and ``logs/``) or a
demonstration's (``DEMO.json``, the run in ``run/``) below the base
directory, which is the working directory or a listed project's
directory. Every file in it can be opened, but no link or junction is
followed and no Testuary configuration file is listed or read (models.ini
can hold keys), by Testuary's own rule for project copies. A requested path is
matched name by name against the listings, never opened as given. tests.csv
is read by column name and shown as Testuary wrote it.

The views read what Testuary names: each file's run directories
(``project.run_name``), each run's ``summary.json`` and the file's summary
across its runs (``results``), phase_state.json (``phase_state``), each
phase end's ``consolidation_record.json``, the starter phase's generation
manifest, the loops' JSON Lines logs for the measurements a tool refused,
the phases' labels and the end-of-run summary.
"""
from __future__ import annotations

import csv
import io
import json
import math
import os
import threading
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Mapping, Optional, Tuple, Union

from testuary.web import _jobs

MAX_TEXT_BYTES = 2 * 1024 * 1024
MAX_TABLE_BYTES = 16 * 1024 * 1024
MAX_LISTED_FILES = 5000
MAX_ENTRIES = 50000
#: The newest job event files searched for a run's end-of-run summary, and
#: how much of each file's end is read for it.
MAX_SUMMARY_FILES = 200
SUMMARY_LINES = 50
SUMMARY_TAIL_BYTES = 4 * 1024 * 1024
#: At most this many files' readings are kept in memory.
MAX_REMEMBERED = 4096
#: The configuration files the results view never shows, in any case, also
#: as copies (a demonstration's workspace holds its models.ini).
CONFIGURATION_NAMES = frozenset({"config.ini", "models.ini", "project.ini", "projects.ini"})
# A Windows reparse point whose tag has this bit stands for another name: a
# symbolic link or a junction.
_NAME_SURROGATE = 0x20000000
# The server also opens a directory below a listed project's directory
# (App.results_base), so the refusal names both.
_OUTSIDE = "Only directories below the working directory or a listed project's directory are shown."
_NO_FILE = "The run directory shows no such file."


class NotShown(Exception):
    """The directory is not a run directory, or the file is not shown."""


@dataclass(frozen=True)
class Dir:
    root: Path
    #: The names of the run's directory below ``root``: none, or ``run`` for a
    #: demonstration.
    run: Tuple[str, ...]
    demo: bool
    #: Root-relative POSIX paths, case folded, that Testuary's rule leaves out.
    withheld: FrozenSet[str] = frozenset()

    def shows(self, relative: str) -> bool:
        """Whether the views show the file at this root-relative path."""
        name = relative.rsplit("/", 1)[-1]
        return (name.casefold() not in CONFIGURATION_NAMES
                and relative.casefold() not in self.withheld)


def _linked(entry: os.DirEntry) -> bool:
    if entry.is_symlink():
        return True
    try:
        info = entry.stat(follow_symlinks=False)
    except OSError:
        return True
    return bool(getattr(info, "st_reparse_tag", 0) & _NAME_SURROGATE)


def _child(directory: Path, name: str, *, fold_case: bool = False) -> Optional[os.DirEntry]:
    """The entry ``name`` of ``directory``, or None when absent or a link."""
    found = None
    try:
        with os.scandir(directory) as entries:
            for entry in entries:
                if entry.name == name:
                    found = entry
                    break
                if fold_case and found is None and entry.name.casefold() == name.casefold():
                    found = entry
    except OSError:
        return None
    return None if found is None or _linked(found) else found


def _parts(base: Path, text: str) -> Tuple[str, ...]:
    """The directory names of ``text`` below ``base``, checked as text first:
    opening a network or device path would contact another host."""
    if not text or "\x00" in text:
        raise NotShown("Name a run directory below the working directory or a listed project's directory.")
    given = Path(text)
    if given.drive and not (len(given.drive) == 2 and given.drive[1] == ":"):
        raise NotShown(_OUTSIDE)
    if given.is_absolute():
        try:
            given = given.relative_to(base)
        except ValueError:
            raise NotShown(_OUTSIDE) from None
    elif given.drive or given.root:
        raise NotShown(_OUTSIDE)
    if not given.parts or any(part in (".", "..") for part in given.parts):
        raise NotShown(_OUTSIDE)
    return given.parts


def _entry(run_dir: Dir, parts: Tuple[str, ...]) -> Optional[os.DirEntry]:
    """The entry at ``parts`` below the root, through directories only, or None."""
    directory, entry = run_dir.root, None
    for index, name in enumerate(parts):
        if index and not entry.is_dir(follow_symlinks=False):
            return None
        entry = _child(directory, name)
        if entry is None:
            return None
        directory = Path(entry.path)
    return entry


def _is_dir(run_dir: Dir, parts: Tuple[str, ...]) -> bool:
    entry = _entry(run_dir, parts)
    return entry is not None and entry.is_dir(follow_symlinks=False)


def _shown_file(run_dir: Dir, parts: Tuple[str, ...]) -> Optional[Path]:
    """The path of the file at ``parts`` when the views show it."""
    entry = _entry(run_dir, parts)
    if (entry is None or not entry.is_file(follow_symlinks=False)
            or not run_dir.shows("/".join(parts))):
        return None
    return Path(entry.path)


def _entries(directory: Path) -> List[os.DirEntry]:
    """The entries of ``directory``, links left out, by name."""
    try:
        with os.scandir(directory) as entries:
            return sorted((entry for entry in entries if not _linked(entry)),
                          key=lambda entry: entry.name)
    except OSError:
        return []


def _listing(run_dir: Dir, parts: Tuple[str, ...]) -> List[os.DirEntry]:
    """The entries of the directory at ``parts``, links left out, by name."""
    if parts and not _is_dir(run_dir, parts):
        return []
    return _entries(run_dir.root.joinpath(*parts))


def configuration_files(base: Path, root: Path) -> FrozenSet[str]:
    """The configuration files a copy of ``root`` leaves out by Testuary's rule
    (``config.configuration_exclusions``), under any name or hard link, as
    root-relative POSIX paths, case folded."""
    try:
        from testuary import config
        from testuary.settings import resolve_settings

        settings = resolve_settings(base_dir=base)
        excluded = config.configuration_exclusions(root, settings)
    except Exception:
        # No details: the check reports them, and an error text can
        # quote a configuration value.
        raise NotShown("Testuary cannot read the settings of the working directory, so the "
                       "results view cannot tell its configuration files apart. The quick "
                       "check names the problem.") from None
    return frozenset(path.casefold() for path in excluded)


def _walk(base: Path, text: str) -> List[str]:
    """The names of the directories ``text`` names below ``base``, each an
    existing directory with no link on the way, in their spelling on disk."""
    names: List[str] = []
    directory = base
    for name in _parts(base, text):
        entry = _child(directory, name, fold_case=os.name == "nt")
        if entry is None or not entry.is_dir(follow_symlinks=False):
            if not os.path.lexists(directory / name):
                raise NotShown(f"The directory {text} does not exist.")
            raise NotShown(f"{text} is not a directory, or a link is on the way: the results view "
                           "follows no links.")
        names.append(entry.name)
        directory = Path(entry.path)
    return names


def named_dir(base: Path, text: str) -> Path:
    """The directory ``text`` names below the base directory ``base``, which
    is the working directory or a listed project's directory, as the results
    view reaches a run directory but of any kind, such as a directory an
    extension's directory type opens."""
    return Path(base).joinpath(*_walk(Path(base), text))


def locate(base: Path, text: str) -> Dir:
    """The run directory ``text`` names: a path relative to the base
    directory ``base``, which is the working directory or a listed
    project's directory, or an absolute one inside it."""
    base = Path(base)
    names = _walk(base, text)
    working = Dir(base, (), False)

    def has(*parts: str, is_dir: bool = True) -> bool:
        entry = _entry(working, parts)
        return entry is not None and (entry.is_dir(follow_symlinks=False) if is_dir
                                      else entry.is_file(follow_symlinks=False))

    # A run's own results directory, or a demonstration's run, opens the whole.
    if len(names) > 1 and names[-1] == "results" and has(*names, "tests.csv", is_dir=False):
        names.pop()
    if len(names) > 1 and names[-1] == "run" and has(*names[:-1], "DEMO.json", is_dir=False):
        names.pop()
    root = base.joinpath(*names)
    if has(*names, "DEMO.json", is_dir=False):
        return Dir(root, ("run",), True, configuration_files(base, root))
    if has(*names, "results") and has(*names, "logs"):
        return Dir(root, (), False, configuration_files(base, root))
    raise NotShown(f"{text} is not a Testuary run directory: it holds neither the results "
                   "and logs directories of a run nor a demonstration's DEMO.json.")


def _named_run(base: Path, text: str) -> Union[Dir, Path]:
    """The run ``text`` names below the working directory ``base`` for a
    nested run: a run directory or a demonstration's directory (``locate``), else
    an event file, a regular file in a directory ``named_dir`` reaches,
    which is none of Testuary's configuration files."""
    parts = _parts(base, text)
    parent_dir = named_dir(base, os.path.join(*parts[:-1])) if len(parts) > 1 else Path(base)
    entry = _child(parent_dir, parts[-1], fold_case=os.name == "nt")
    if entry is not None and entry.is_dir(follow_symlinks=False):
        return locate(base, text)
    if entry is None or not entry.is_file(follow_symlinks=False):
        raise NotShown(f"The file {text} does not exist." if not os.path.lexists(parent_dir / parts[-1])
                       else f"{text} is a link or a special file: a nested run follows no links "
                       "and reads only files and directories.")
    if entry.name.casefold() in CONFIGURATION_NAMES:
        raise NotShown(f"{text} is a configuration file, which a nested run never reads.")
    return Path(entry.path)


def _relative(parts: Tuple[str, ...]) -> str:
    return "/".join(parts)


def _read_bytes(path: Path, limit: int) -> Tuple[int, Optional[bytes]]:
    """The file's size and, up to ``limit`` bytes, its bytes, following no
    link put in its place where the system can refuse one."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        size = os.fstat(stream.fileno()).st_size
        if size > limit:
            return size, None
        return size, stream.read(limit)


# The readers' results by path and reader, with each file's modification
# time, size and identity when read. Shared values: nothing changes them.
_READ: "OrderedDict[tuple, tuple]" = OrderedDict()
_READ_LOCK = threading.Lock()


def _remembered(path: Path, reader):
    """``reader(stream, size)`` of the file at ``path``, read again only when its
    modification time, size or identity changed, following no link put in
    its place. OSError when the file cannot be opened."""
    flags = os.O_RDONLY | getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, "rb") as stream:
        info = os.fstat(stream.fileno())
        stamp = (info.st_mtime_ns, info.st_size, info.st_ino)
        name = (str(path), reader)
        with _READ_LOCK:
            known = _READ.get(name)
            if known is not None and known[0] == stamp:
                _READ.move_to_end(name)
                return known[1]
        value = reader(stream, info.st_size)
    with _READ_LOCK:
        _READ[name] = (stamp, value)
        _READ.move_to_end(name)
        while len(_READ) > MAX_REMEMBERED:
            _READ.popitem(last=False)
    return value


def _table_reader(stream, size: int):
    if size > MAX_TABLE_BYTES:
        return [], []
    text = stream.read(MAX_TABLE_BYTES).decode("utf-8-sig", errors="replace")
    rows = list(csv.reader(io.StringIO(text, newline="")))
    if not rows:
        return [], []
    columns = rows[0]
    return columns, [dict(zip(columns, row)) for row in rows[1:] if row]


def _json_reader(stream, size: int):
    if size > MAX_TEXT_BYTES:
        return None
    try:
        return json.loads(stream.read(MAX_TEXT_BYTES).decode("utf-8"))
    except ValueError:
        return None


def _table(run_dir: Dir, parts: Tuple[str, ...]):
    path = _shown_file(run_dir, parts)
    if path is None:
        return [], []
    try:
        return _remembered(path, _table_reader)
    except OSError:
        return [], []


def _json(run_dir: Dir, parts: Tuple[str, ...]) -> Optional[dict]:
    path = _shown_file(run_dir, parts)
    if path is None:
        return None
    try:
        return _remembered(path, _json_reader)
    except OSError:
        return None


def _listed_files(run_dir: Dir, start: Tuple[str, ...] = ()) -> List[dict]:
    """The files below ``start`` that the views show, nearest first, at most
    MAX_LISTED_FILES of them, listed by path."""
    found: List[dict] = []
    if start and not _is_dir(run_dir, start):
        return found
    pending = deque([(start, run_dir.root.joinpath(*start))])
    seen = 0
    while pending and len(found) < MAX_LISTED_FILES and seen < MAX_ENTRIES:
        parts, directory = pending.popleft()
        for entry in _entries(directory):
            seen += 1
            relative = (*parts, entry.name)
            if entry.is_dir(follow_symlinks=False):
                pending.append((relative, Path(entry.path)))
            elif entry.is_file(follow_symlinks=False) and run_dir.shows(_relative(relative)):
                try:
                    size = entry.stat(follow_symlinks=False).st_size
                except OSError:
                    continue
                found.append({"path": _relative(relative), "size": size})
                if len(found) >= MAX_LISTED_FILES:
                    break
    return sorted(found, key=lambda item: item["path"])


def _suite_files(run_dir: Dir, suite: Tuple[str, ...]) -> List[dict]:
    """A delivered suite's files without Python's compiled caches."""
    return [item for item in _listed_files(run_dir, suite)
            if "__pycache__" not in item["path"].split("/") and not item["path"].endswith(".pyc")]


def _testuary():
    """Testuary's names for its result files: ``results``,
    ``phase_state.PHASES``, ``project`` and the phases' labels."""
    from testuary import phase_state, project as project_module
    from testuary import results
    from testuary.presentation.view import phase_label

    return results, phase_state.PHASES, project_module, phase_label


def _percent(value) -> Optional[dict]:
    """A summary's percentage: its number for the bars and Testuary's text
    (``percent_text``). None when not measured."""
    from testuary.presentation.view import percent_text

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    return {"value": float(value), "text": f"{percent_text(value)}%"}


def _phase_record(record: Optional[dict], phase: str, run_idx: int, words) -> Optional[dict]:
    """The phase as the file's phase_state.json records it for this run, its
    status in Testuary's ``words``, and a failed phase's failure kind."""
    entries = record.get(phase) if isinstance(record, dict) else None
    entry = entries.get(str(run_idx)) if isinstance(entries, dict) else None
    if not isinstance(entry, dict):
        return None
    status, kind = entry.get("status"), entry.get("failure_kind")
    status = status if isinstance(status, str) else None
    kind = kind if isinstance(kind, str) else None
    from testuary import results

    return {"status": status, "status_text": words(status) if status else None,
            "failure_kind": kind,
            "failure_kind_text": results.failure_kind_text(kind) if kind else None}


def _run_dirs(run_dir: Dir, parts: Tuple[str, ...], project_module) -> List[Tuple[int, str]]:
    """A file's run directories by run, as ``project.run_name`` names them."""
    found = []
    for entry in _listing(run_dir, parts):
        if entry.is_dir(follow_symlinks=False):
            index = project_module.run_idx_from_artifact_path(Path(entry.name))
            if project_module.run_name(index) == entry.name:
                found.append((index, entry.name))
    return sorted(found)


def _file_run(run_dir: Dir, file_parts: Tuple[str, ...], run_idx: int, name: str,
              record: Optional[dict], row: Optional[int], names) -> dict:
    """One run of a file: its summary's values and phase statuses, its phase
    records, its delivered suite and its row of tests.csv."""
    results, phases, project_module, phase_label = names
    summary = _json(run_dir, (*file_parts, name, results.SUMMARY))
    summary = summary if isinstance(summary, dict) else {}
    statuses = {status.rsplit("_", 1)[0]: status for status in results.STATUSES}
    entries = []
    for phase in phases:
        status = summary.get(statuses[phase]) if phase in statuses else None
        status = status if isinstance(status, str) else None
        entries.append({"phase": phase, "label": phase_label(phase), "status": status,
                        "status_text": results.outcome_text(status) if status else None,
                        "record": _phase_record(record, phase, run_idx, results.outcome_text)})
    measured = summary.get("fully_measured")
    suite = file_parts + project_module.phase_artifact_path(Path(), "tests", run_idx).parts
    files = _suite_files(run_dir, suite) if _is_dir(run_dir, suite) else None
    return {
        "name": name, "row": row, "summarized": bool(summary),
        "fully_measured": measured if isinstance(measured, bool) else None,
        "values": {value: _percent(summary.get(value)) for value in results.VALUES},
        "cut": _cut(summary),
        "phases": entries,
        "details": _details(summary),
        "warnings": _warnings(summary, results.LABELS),
        "phase_ends": [end for end in (_phase_end(run_dir, (*file_parts, name), phase, phase_label)
                                       for phase in PHASE_ENDS) if end is not None],
        "refusals": _refusals(run_dir, file_parts[-1], name, phase_label),
        "suite": _suite(run_dir, suite, files) if files is not None else None,
    }


def _suite(run_dir: Dir, suite: Tuple[str, ...], files: List[dict]) -> dict:
    """A run's delivered suite: its files, the test files Testuary delivers
    and the setup files beside them that a test can need at run time
    (Python's conftest.py)."""
    tests = [item["path"] for item in files if _delivered_test(item["path"])]
    return {"suite_dir": _relative(suite), "files": files, "tests": tests,
            "setup": _setup_files(run_dir, tests)}


def _setup_files(run_dir: Dir, tests: Iterable[str]) -> List[str]:
    """The setup files beside the delivered test files, by each language's
    own rule (``shared_setup_files``), as the run directory shows them."""
    found: List[str] = []
    for relative in tests:
        backend = _delivered_test(relative)
        parts = tuple(relative.split("/"))
        path = _shown_file(run_dir, parts)
        if backend is None or path is None:
            continue
        try:
            setup = backend.shared_setup_files(str(path.parent))
        except Exception:
            continue
        for item in setup:
            name = (*parts[:-1], Path(item).name)
            if _shown_file(run_dir, name) is not None and _relative(name) not in found:
                found.append(_relative(name))
    return found


#: The phases whose end chooses the suite the run goes on with. Each records
#: its choice in its archive directory.
PHASE_ENDS = ("starter", "coverage", "mutation")
#: A phase end's candidates in plain words. A starter suite is named by its
#: slot, as ``starters.slot_name`` writes it.
CANDIDATES = {"merged": "The combined suite",
              "reduced": "The combined suite, reduced to the tests it needs",
              "baseline": "The suite the coverage phase received",
              "agent": "The coverage phase's suite",
              "standalone_0": "The suite the mutation phase received",
              "standalone_1": "The mutation phase's suite"}
#: Why the delivered candidate was chosen, by the record's selection basis.
BASES = {"score": "it has the best score",
         "reduced_preference": "the best scores are equal and the reduced suite is preferred",
         "timing_median": "the best scores are equal and it ran fastest",
         "input_order": "the best scores are equal and the input order decides"}
#: The scores a phase end compares, in plain words.
AXES = {"mutation": "mutation score", "branch": "branch coverage", "line": "line coverage"}
#: A candidate's outcome at its phase end.
STATES = {"delivered": "chosen", "measured": "measured, not chosen",
          "incomplete": "passes, not fully measured", "failing": "does not pass",
          "absent": "not built"}


def _candidate_text(label: str) -> str:
    """A candidate's label in plain words. An unknown label as it is."""
    if label in CANDIDATES:
        return CANDIDATES[label]
    prefix, _, number = label.rpartition("_")
    return f"Starter suite {int(number)}" if prefix == "suite" and number.isdigit() else label


def _basis(audit: dict) -> Optional[str]:
    """Why the phase end chose its suite, by its record: a runtime tie-break
    whose times were equal or not measured is decided by the input order."""
    basis = audit.get("selection_basis")
    if basis == "timing_median" and audit.get("runtime_tie_break_basis") == "input_order":
        return "input_order"
    return basis


def _choice(delivered: Optional[str], chosen: Optional[str], basis, axes: List[str]) -> Optional[str]:
    """What a phase end delivered and why, from its record."""
    if delivered == "inherited":
        return ("The loop ended with the suite it received, so the phase end kept that "
                "suite and its measurements.")
    if delivered == "unresolved":
        return "No candidate passed and was fully measured, so the phase end chose no suite."
    if not chosen:
        return None
    named = _candidate_text(chosen)
    text = f"Chosen: {named[:1].lower()}{named[1:]}"
    if basis not in BASES:
        return text + "."
    order = ", then ".join(AXES[axis] for axis in axes)
    return f"{text}, because {BASES[basis]}" + (f" ({order})." if basis == "score" and order else ".")


def _freed(run_dir: Dir, run: Tuple[str, ...]) -> Optional[str]:
    """The passing starter suites that left without complete numbers, as the
    starter phase's generation manifest counts them. None when there were
    none or the manifest does not count them."""
    manifest = _json(run_dir, (*run, "archive", "starter", "suites", "_generation.json"))
    counts = [manifest.get(key) if isinstance(manifest, dict) else None
              for key in ("n_unmeasured_coverage", "n_unmeasured_mutation")]
    counts = [count if isinstance(count, int) and not isinstance(count, bool) and count > 0 else 0
              for count in counts]
    total = sum(counts)
    if not total:
        return None
    named = [f"{count} that the {tool} tool could not measure"
             for count, tool in zip(counts, ("coverage", "mutation")) if count]
    suites = "suite" if total == 1 else "suites"
    each = "It was" if total == 1 else "Each was"
    return (f"{total} passing starter {suites} left without being fully measured "
            f"({', '.join(named)}). {each} generated again while its attempts lasted, else left out.")


def _phase_end(run_dir: Dir, run: Tuple[str, ...], phase: str, phase_label) -> Optional[dict]:
    """How the phase end chose its suite, from its consolidation_record.json:
    what it delivered and why, and every candidate with its numbers and
    outcome. The starter's also names the suites that left without complete
    numbers. None without a record."""
    record = _json(run_dir, (*run, "archive", phase, "consolidation_record.json"))
    freed = _freed(run_dir, run) if phase == "starter" else None
    if not isinstance(record, dict) and freed is None:
        return None
    record = record if isinstance(record, dict) else {}
    delivered = record.get("delivered_from") if isinstance(record.get("delivered_from"), str) else None
    source = record.get("winner_source_label")
    chosen = source if delivered == "single_suite" and isinstance(source, str) else delivered
    audit = record.get("selection_audit") if isinstance(record.get("selection_audit"), dict) else {}
    axes = [axis for axis in audit.get("active_axes") or () if axis in AXES]
    candidates = []
    for item in audit.get("candidates") or ():
        if not isinstance(item, dict) or not isinstance(item.get("label"), str):
            continue
        label, status = item["label"], item.get("status")
        # The recorded status first: an excluded reduction that passed its
        # normal run records passing as False.
        state = ("delivered" if label == chosen else "measured" if item.get("fully_measured") is True
                 else status if status in ("absent", "incomplete", "failing")
                 else "incomplete" if item.get("passing") is True else "failing")
        # The reason why a candidate could not be chosen. A measured one's
        # record notes how it was measured, and its numbers say the rest.
        reason = (item.get("status_reason") or None
                  if state in ("incomplete", "failing", "absent") else None)
        candidates.append({
            "label": label, "text": _candidate_text(label), "state": state, "state_text": STATES[state],
            "tests": _amount(item.get("tests")),
            **{axis: _percent(item.get(axis)) for axis in ("line", "branch", "mutation")},
            "mutation_derived": item.get("mutation_derived") is True,
            "reason": reason.strip() if isinstance(reason, str) and reason.strip() else None})
    return {"phase": phase, "label": phase_label(phase), "delivered": chosen,
            "summary": _choice(delivered, chosen, _basis(audit), axes),
            "freed": freed, "candidates": candidates}


#: The loops' agent logs, one JSON object per line, in a run's log directory
#: (logs/test/<file>/<run>/): a measurement a tool refused is logged there
#: with the tool and the tests that stopped it.
LOOP_LOGS = {"coverage": "coverage.jsonl", "mutation": "mutation.jsonl"}
#: The tools that measure, in plain words.
TOOLS = {"coverage": "The coverage tool", "mutation": "The mutation tool"}
#: At most this much of a loop's log is read for its refused measurements.
MAX_LOG_BYTES = 64 * 1024 * 1024


def _refusal_reader(stream, size: int) -> List[dict]:
    """The console records of a JSON Lines log that name the tests a tool
    stopped on, in order. Only the lines that hold the field are parsed."""
    found = []
    for line in stream.read(MAX_LOG_BYTES).splitlines():
        if b'"stopping_tests"' not in line:
            continue
        try:
            record = json.loads(line.decode("utf-8"))
        except ValueError:
            continue
        if (isinstance(record, dict) and record.get("record") == "console"
                and isinstance(record.get("stopping_tests"), list)):
            found.append(record)
    return found


def _refusals(run_dir: Dir, workspace: str, run: str, phase_label) -> List[dict]:
    """The measurements a tool refused in the run's loops, from their logs:
    the loop, the tool and the tests that stopped it, as a sentence."""
    shown = []
    for phase, name in LOOP_LOGS.items():
        path = _shown_file(run_dir, (*run_dir.run, "logs", "test", workspace, run, name))
        if path is None:
            continue
        try:
            records = _remembered(path, _refusal_reader)
        except OSError:
            continue
        for record in records:
            tests = [test for test in record["stopping_tests"] if isinstance(test, str)]
            tool = TOOLS.get(record.get("tool"), "A tool")
            text = (f"{tool} could not measure the suite. The tests that stopped it: {', '.join(tests)}."
                    if tests else f"{tool} could not measure the suite and named no test.")
            shown.append({"phase": phase, "label": phase_label(phase), "tool": record.get("tool"),
                          "tests": tests, "text": text})
    return shown


def _delivered_test(relative: str):
    """The language backend of a suite's file when it is the test file
    Testuary delivers for a target, by the language's own rule
    (``classify_test_kind``), else None: setup files such as conftest.py
    and the records are not."""
    from langadapter import language_for

    from testuary.languages import get_backend

    name = relative.rsplit("/", 1)[-1]
    try:
        backend = get_backend(language_for(name))
    except Exception:
        return None
    return backend if backend is not None and backend.classify_test_kind(name) == "coverage" else None


def _warnings(summary: dict, labels: Mapping[str, str]) -> Optional[List[dict]]:
    """A run summary's warnings (such as a time limit reached): Testuary's
    sentence, the kind, phase and setting by their plain names and the count.
    Empty without warnings, None (not known) for a summary without the list."""
    items = summary.get("warnings")
    if not isinstance(items, list):
        return None
    named: Dict[str, str] = {}
    if items:
        from testuary import settings

        named = {f"{section.name}.{setting.name}": setting.label
                 for section in settings.reference() for setting in section.settings}
    shown = []
    for item in items:
        if not isinstance(item, dict):
            continue
        kind, phase, setting, unit, text = (
            item.get(key) if isinstance(item.get(key), str) else None
            for key in ("kind", "phase", "setting", "unit", "text"))
        count = item.get("count")
        shown.append({
            "kind": kind, "kind_label": labels.get(kind, kind) if kind else None,
            "phase": phase, "phase_label": labels.get(phase, phase) if phase else None,
            "setting": setting, "setting_label": named.get(setting) if setting else None,
            "value": _amount(item.get("value"), "s" if unit == "s" else ""), "unit": unit,
            "count": count if isinstance(count, int) and not isinstance(count, bool) else None,
            "text": text})
    return shown


def _cut(summary: dict) -> Optional[dict]:
    """A run whose mutation measurement in Finishing left mutants untested: its
    lower bound and untested mutants. None for a run with a score or no
    bound."""
    final = summary.get("finishing") if isinstance(summary.get("finishing"), dict) else {}
    bound = _percent(final.get("mutation_score_lower_bound"))
    if summary.get("mutation_score") is not None or bound is None:
        return None
    untested = final.get("mutants_untested")
    return {"lower_bound": bound,
            "untested": int(untested) if isinstance(untested, (int, float)) and not isinstance(untested, bool)
            else None}


def _amount(value, unit: str = "") -> Optional[dict]:
    """A count with separators or a time (minutes and seconds, below a minute
    seconds with one decimal). None when unknown."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        return None
    if unit == "s":
        whole = int(round(value))
        minutes, seconds = divmod(whole, 60)
        short = f"{value:g} s" if float(value).is_integer() else f"{value:.1f} s"
        text = (short if value < 60 else f"{minutes} min" if not seconds
                else f"{minutes} min {seconds} s")
        return {"value": float(value), "text": text}
    return {"value": float(value), "text": f"{int(value):,}" if float(value).is_integer() else f"{value:g}"}


def _shown(name: str, value) -> Optional[dict]:
    """A summary entry: a percentage for a score, a time for seconds, else a
    count."""
    if name.endswith(("coverage", "score", "score_selective")):
        return _percent(value)
    return _amount(value, "s" if name == "seconds" else "")


def _details(summary: dict) -> dict:
    """A run's details from its summary.json: each phase's values and use,
    finishing's measurement, the starter suites' statistics and where the time
    went, by Testuary's names (the page labels them from ``labels``)."""
    phases = summary.get("phases") if isinstance(summary.get("phases"), dict) else {}
    final = summary.get("finishing") if isinstance(summary.get("finishing"), dict) else {}
    starter = summary.get("starter") if isinstance(summary.get("starter"), dict) else {}
    time = summary.get("time") if isinstance(summary.get("time"), dict) else {}
    return {
        "phases": {phase: {name: _shown(name, value) for name, value in values.items()}
                   for phase, values in phases.items() if isinstance(values, dict)},
        "finishing": {name: _shown(name, value) for name, value in final.items()},
        "starter": {name: {statistic: _percent(value) for statistic, value in values.items()}
                    for name, values in starter.items() if isinstance(values, dict)},
        "time": {name: _amount(value, "s") for name, value in time.items()},
    }


def _statistics(item) -> Optional[dict]:
    """Statistics across files from the project's summary: the count, each
    statistic as a percentage, and the values."""
    if not isinstance(item, dict):
        return None
    values = item.get("values") if isinstance(item.get("values"), list) else []
    return {"count": item.get("count") if isinstance(item.get("count"), int) else None,
            **{key: _percent(item.get(key)) for key in ("mean", "median", "minimum", "maximum")},
            "std": _amount(item.get("std")) if item.get("std") is not None else None,
            "values": [_percent(value) for value in values if _percent(value) is not None]}


def _project(run_dir: Dir, results: Tuple[str, ...], values: Iterable[str]) -> Optional[dict]:
    """The project's summary across its files (results/summary.json), each
    value beside the starter suites' medians."""
    from testuary.results import SUMMARY

    data = _json(run_dir, (*results, SUMMARY))
    if not isinstance(data, dict):
        return None
    shown = data.get("values") if isinstance(data.get("values"), dict) else {}
    starters = data.get("starter_median") if isinstance(data.get("starter_median"), dict) else {}
    shown_values = {name: {"delivered": _statistics(shown.get(name)), "starter": _statistics(starters.get(name))}
                    for name in values}
    if "mutation_score" in shown_values:
        # The files without a score whose measurement left mutants untested.
        shown_values["mutation_score"]["lower_bound"] = _statistics(data.get("mutation_score_lower_bound"))
    return shown_values


def _across(summary: Optional[dict], values: Iterable[str]) -> Optional[dict]:
    """The file's summary across its runs as Testuary wrote it: the complete
    runs and each value's count, mean, lowest and highest."""
    if not isinstance(summary, dict):
        return None
    shown = {}
    for value in values:
        item = summary.get(value)
        item = item if isinstance(item, dict) else {}
        measured = item.get("runs")
        shown[value] = {"runs": measured if isinstance(measured, int) else None,
                        **{key: _percent(item.get(key)) for key in ("mean", "minimum", "maximum")}}
    measured_runs = summary.get("fully_measured_runs")
    starters = summary.get("starter_median") if isinstance(summary.get("starter_median"), dict) else {}
    warned = summary.get("warning_runs")
    bound = summary.get("mutation_score_lower_bound") if isinstance(
        summary.get("mutation_score_lower_bound"), dict) else {}
    return {"fully_measured_runs": measured_runs if isinstance(measured_runs, int) else None, "values": shown,
            "lower_bound": _percent(bound.get("mean")),
            "starter_median": {value: _percent(starters.get(value)) for value in values},
            "warning_runs": warned if isinstance(warned, int) and not isinstance(warned, bool) else None}


def same_path(path) -> str:
    """``path`` in one spelling for comparing paths: normalized, and in one
    case on a system that ignores case."""
    return os.path.normcase(os.path.normpath(str(path)))


def _last_lines(stream, size: int) -> List[bytes]:
    """The last SUMMARY_LINES lines of an open file, the last first, from at
    most SUMMARY_TAIL_BYTES at its end. A line that limit cuts is left out."""
    data, start, block = b"", size, 64 * 1024
    while start > 0 and len(data) < SUMMARY_TAIL_BYTES and data.count(b"\n") <= SUMMARY_LINES:
        step = min(block, start, SUMMARY_TAIL_BYTES - len(data))
        start -= step
        stream.seek(start)
        data = stream.read(step) + data
        block *= 2
    lines = data.split(b"\n")
    if start > 0:
        lines = lines[1:]
    return [line for line in reversed(lines) if line.strip()][:SUMMARY_LINES]


def _completion_reader(stream, size: int) -> Optional[tuple]:
    """The completion record among an event file's last lines and the event
    before it, whose provenance the summary keeps, or None."""
    record = None
    for line in _last_lines(stream, size):
        value = _jobs.record(line)
        if value is None:
            continue
        if record is None:
            if isinstance(value.get("completion"), dict):
                record = value["completion"]
        elif isinstance(value.get("event"), dict):
            return record, value["event"]
    return None if record is None else (record, None)


def _completion(path: Path) -> Optional[tuple]:
    try:
        return _remembered(path, _completion_reader)
    except OSError:
        return None


def _written_end(run_dir: Dir, results: Tuple[str, ...]) -> Optional[dict]:
    """The end-of-run summary Testuary keeps in the run directory
    (``results/completion.json``), as the console printed it, or None."""
    from testuary.presentation.terminal import COMPLETION_FILE, read_completion

    path = _shown_file(run_dir, (*results, COMPLETION_FILE))
    if path is None:
        return None
    try:
        completion, provenance = read_completion(path)
    except (OSError, ValueError, TypeError):
        return None
    from testuary.presentation import DisplayEvent

    last = DisplayEvent(run_id="results", target="all targets", phase="pipeline", operation="run",
                        state="succeeded", elapsed_s=0.0, level="run", provenance=provenance)
    return {"job": None, "lines": _jobs.summary_lines(completion, last)}


def _recorded_end(results_dir: Path, event_files: Iterable[Path]) -> Optional[dict]:
    """The end-of-run summary of ``results_dir`` from the newest job event
    file whose completion record names it, with the job's name: for a run
    that ended before it wrote its completion.json."""
    wanted = same_path(str(results_dir))
    for path in list(event_files)[:MAX_SUMMARY_FILES]:
        found = _completion(Path(path))
        if found is None:
            continue
        record, last = found
        named = record.get("results_directory")
        if isinstance(named, str) and same_path(named) == wanted:
            return {"job": Path(path).parent.name,
                    "lines": _jobs.completion_lines(record, last)}
    return None


def _target_order(value) -> Tuple[int, str]:
    text = str(value if value is not None else "").strip()
    return (int(text), text) if text.isdigit() else (1 << 30, text)


def _run(run_dir: Dir, rows: List[dict], event_files: Iterable[Path]) -> dict:
    """The run a run directory holds, for the dashboard: each target in tests.csv's
    order with its runs, its summary across them, and the end-of-run summary."""
    names = _testuary()
    results, _phases, project_module, _label = names
    results_parts = (*run_dir.run, "results")
    # The row of tests.csv of each target's run, found by the columns target_number and
    # run, as the result-file reference names them.
    places: Dict[Tuple[str, int], int] = {}
    for index, row in enumerate(rows):
        target_number, run = (row.get("target_number") or "").strip(), (row.get("run") or "").strip()
        if run.isdigit():
            places.setdefault((target_number, int(run)), index)
    files = []
    for entry in _listing(run_dir, results_parts):
        if not entry.is_dir(follow_symlinks=False):
            continue
        file_parts = (*results_parts, entry.name)
        record = _json(run_dir, (*file_parts, "phase_state.json"))
        record = record if isinstance(record, dict) else None
        across = _json(run_dir, (*file_parts, results.SUMMARY))
        runs = _run_dirs(run_dir, file_parts, project_module)
        if record is None and not runs and not isinstance(across, dict):
            continue
        target_number = str(record.get("target_number", "")) if record else ""
        named = record.get("target_file_path") if record else None
        if not (isinstance(named, str) and named) and isinstance(across, dict):
            named = across.get("file")
        files.append({
            "target_number": target_number, "workspace": entry.name,
            "name": named if isinstance(named, str) and named else entry.name,
            "runs": [_file_run(run_dir, file_parts, index, name, record,
                               places.get((target_number, index)), names) for index, name in runs],
            "across": _across(across, results.VALUES),
        })
    files.sort(key=lambda item: (_target_order(item["target_number"]), item["name"]))
    return {"name": run_dir.root.name, "run_dir": str(run_dir.root), "results": _relative(results_parts),
            "files": files,
            "values": [{"id": value, "name": results.LABELS[value]}
                       for value in results.VALUES],
            "labels": dict(results.LABELS),
            "project": _project(run_dir, results_parts, results.VALUES),
            "summary": (_written_end(run_dir, results_parts)
                        or _recorded_end(run_dir.root.joinpath(*results_parts), event_files))}


def latest(project: Path) -> Optional[dict]:
    """A project's newest results: the run directory whose ``results/summary.json``
    Testuary wrote last, with its files, runs and summaries. None when there
    is none. Any other failure is raised with its reason."""
    from testuary import results
    from testuary.settings import project_locations

    root = project_locations(Path(project)).results
    if not root.is_dir():
        return None
    newest, stamp = None, None
    with os.scandir(root) as entries:
        for entry in entries:
            if _linked(entry) or not entry.is_dir(follow_symlinks=False):
                continue
            written = Path(entry.path) / "results" / results.SUMMARY
            try:
                info = written.lstat()
            except FileNotFoundError:
                continue
            if written.is_symlink() or not written.is_file():
                continue
            if stamp is None or info.st_mtime > stamp:
                newest, stamp = written, info.st_mtime
    if newest is None:
        return None
    data = _remembered(newest, _json_reader)
    files = data.get("files") if isinstance(data, dict) else None
    if not isinstance(files, list):
        raise ValueError(f"{newest} is not a summary Testuary wrote, or too large to read")
    shown = []
    for item in files:
        if isinstance(item, dict):
            runs = item.get("runs")
            shown.append({"file": item.get("file") if isinstance(item.get("file"), str) else None,
                          "runs": len(runs) if isinstance(runs, list) else None,
                          **_across(item, results.VALUES)})
    return {"run_dir": str(newest.parent.parent), "written": stamp, "files": shown,
            "values": [{"id": value, "name": results.LABELS[value]} for value in results.VALUES]}


def project_runs(project: Path) -> dict:
    """A project's run directories, newest first, each with its summary's time
    and its files' fully measured runs."""
    from testuary import results
    from testuary.settings import project_locations

    root = project_locations(Path(project)).results
    if not root.is_dir():
        return {"results": str(root), "runs": []}
    found = []
    with os.scandir(root) as entries:
        for entry in entries:
            if _linked(entry) or not entry.is_dir(follow_symlinks=False):
                continue
            output = Path(entry.path)
            written = output / "results" / results.SUMMARY
            try:
                stamp = written.lstat().st_mtime if written.is_file() else None
            except OSError:
                stamp = None
            if stamp is None and not (output / "results").is_dir():
                continue
            data = None
            if stamp is not None:
                try:
                    data = _remembered(written, _json_reader)
                except OSError:
                    data = None
            files = data.get("files") if isinstance(data, dict) and isinstance(data.get("files"), list) else []
            found.append({"run_dir": str(output), "name": entry.name, "written": stamp,
                          "files": len(files),
                          "fully_measured_runs": sum(item.get("fully_measured_runs") or 0 for item in files
                                               if isinstance(item, dict))})
    found.sort(key=lambda item: -(item["written"] or 0))
    return {"results": str(root), "runs": found}


def summary(run_dir: Dir, event_files: Iterable[Path] = ()) -> dict:
    """What the results page shows of ``run_dir``: its run, tests.csv's rows,
    the run's log and every file. ``event_files`` (newest first) can hold the
    run's end-of-run summary."""
    results = (*run_dir.run, "results")
    columns, rows = _table(run_dir, (*results, "tests.csv"))
    log = (*run_dir.run, "logs", "run.log")
    view = {
        "root": str(run_dir.root), "demo": run_dir.demo, "columns": columns, "rows": rows,
        "runs": [_run(run_dir, rows, event_files)],
        "log": _relative(log) if _shown_file(run_dir, log) is not None else None,
        "files": _listed_files(run_dir),
    }
    if run_dir.demo:
        view["demo_record"] = _json(run_dir, ("DEMO.json",))
    return view


def read(run_dir: Dir, relative: str) -> dict:
    """One file of ``run_dir`` as text: ``relative`` is its path as the
    directory's listing names it."""
    if not relative or "\x00" in relative or "\\" in relative:
        raise NotShown(_NO_FILE)
    parts = tuple(relative.split("/"))
    if any(part in ("", ".", "..") for part in parts):
        raise NotShown(_NO_FILE)
    entry = _entry(run_dir, parts)
    if entry is None or not entry.is_file(follow_symlinks=False):
        raise NotShown(_NO_FILE)
    if not run_dir.shows(relative):
        raise NotShown("Testuary's configuration files are not shown in the results view.")
    try:
        size, data = _read_bytes(Path(entry.path), MAX_TEXT_BYTES)
    except OSError:
        raise NotShown(_NO_FILE) from None
    view = {"path": relative, "size": size, "text": None, "note": None}
    if data is None:
        view["note"] = "This file is too large to show here."
        return view
    try:
        if b"\x00" in data:
            raise UnicodeDecodeError("utf-8", data, 0, 1, "binary")
        view["text"] = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        view["note"] = "This file is not text."
    return view


# Copying a delivered test file into its project: only that file, as it is,
# into a directory inside the project and outside its Testuary directory, never
# over a file that exists.

#: Directories a search for a project's own tests leaves out, as the setup's
#: search for its sources does, besides hidden ones.
_NOT_TESTS = frozenset({"venv", "env", "node_modules", "build", "target", "dist", "__pycache__"})
#: The most entries that search looks at.
MAX_PROJECT_ENTRIES = 20000


class NotAdopted(Exception):
    """A copy refused. ``field`` names the form field it concerns."""

    def __init__(self, message: str, field: Optional[str] = None) -> None:
        super().__init__(message)
        self.field = field


def _delivered_file(run_dir: Dir, relative: str):
    """The delivered file at ``relative`` that goes into the project, its
    language backend and the delivered test file it goes with: a test file
    (itself), or a setup file beside one (``_setup_files``)."""
    parts = tuple(relative.split("/")) if relative and "\\" not in relative else ()
    path = _shown_file(run_dir, parts) if parts and "" not in parts and ".." not in parts else None
    if path is None:
        raise NotShown(_NO_FILE)
    backend = _delivered_test(relative)
    if backend is not None:
        return path, backend, path
    siblings = [_relative((*parts[:-1], entry.name)) for entry in _listing(run_dir, parts[:-1])]
    for test in (item for item in siblings if _delivered_test(item)):
        if relative in _setup_files(run_dir, [test]):
            return path, _delivered_test(test), _shown_file(run_dir, tuple(test.split("/")))
    raise NotAdopted(f"{parts[-1]} is not a delivered test file or a setup file beside one: only "
                     "those go into the project.")


def inside(path: Path, outer_dir: Path) -> bool:
    """Whether ``path`` lies in ``outer_dir``, by their text: a caller that
    must refuse links checks them on the way itself."""
    try:
        Path(os.path.normcase(os.path.abspath(path))).relative_to(
            os.path.normcase(os.path.abspath(outer_dir)))
    except ValueError:
        return False
    return True


def _package(backend, path: Path) -> Tuple[str, ...]:
    """The package a test file declares, as directory names: its test ids'
    prefix before the file's own name."""
    return tuple(backend.test_id_prefix(str(path)).split(".")[:-1])


def _tests_dir(project: Path, delivered: Path, backend) -> Path:
    """Where the delivered file goes by default: below the root of the
    project's own tests in the tests' language (the directory holding most of them,
    without the package path their files declare), in the delivered file's
    package path. Without such tests, the project's directory."""
    from testuary.settings import project_locations

    skipped = os.path.normcase(os.path.realpath(project_locations(project).testuary))
    counts: Dict[Path, int] = {}
    examples: Dict[Path, Path] = {}
    seen = 0
    for directory, dirs, names in os.walk(project):
        dirs[:] = [name for name in sorted(dirs) if not name.startswith(".") and name not in _NOT_TESTS
                      and os.path.normcase(os.path.realpath(os.path.join(directory, name))) != skipped
                      and not os.path.islink(os.path.join(directory, name))]
        for name in names:
            seen += 1
            path = Path(directory) / name
            if path.suffix in backend.test_file_extensions() and backend.is_test_file(path):
                counts[path.parent] = counts.get(path.parent, 0) + 1
                examples.setdefault(path.parent, path)
        if seen >= MAX_PROJECT_ENTRIES:
            break
    if not counts:
        return Path(project)
    held = max(counts, key=lambda tests_dir: (counts[tests_dir], -len(tests_dir.parts)))
    root = held
    package = _package(backend, examples[held])
    if package and held.parts[-len(package):] == package:
        root = Path(*held.parts[:-len(package)])
    return root.joinpath(*_package(backend, delivered))


def adoption(run_dir: Dir, project: Path, relative: str) -> dict:
    """What copying the delivered file at ``relative`` into ``project``
    offers: the file's name and the directory it goes to by default, the test
    file's directory for a setup file beside it."""
    path, backend, test = _delivered_file(run_dir, relative)
    return {"file": path.name, "test_dir": str(_tests_dir(Path(project), test, backend)),
            "project": str(project)}


def adopt(run_dir: Dir, project: Path, relative: str, text: str) -> Path:
    """Copy the delivered file at ``relative`` (``_delivered_file``) into the
    directory ``text`` names, relative to ``project`` or absolute inside it: the
    file as it is, into a directory made when missing, never over a file."""
    from testuary.settings import project_locations

    path, _backend, _test = _delivered_file(run_dir, relative)
    if not text.strip():
        raise NotAdopted("Name a directory of the project.", "test_dir")
    project = Path(os.path.realpath(project))
    given = Path(text.strip())
    target = Path(os.path.realpath(given if given.is_absolute() else project / given))
    if not inside(target, project):
        raise NotAdopted("Choose a directory inside the project.", "test_dir")
    if inside(target, Path(os.path.realpath(project_locations(project).testuary))):
        raise NotAdopted("The project's Testuary directory holds its settings and results: choose "
                         "another directory.", "test_dir")
    if target.exists() and not target.is_dir():
        raise NotAdopted(f"{target} is a file, not a directory.", "test_dir")
    _, data = _read_bytes(path, MAX_TEXT_BYTES)
    if data is None:
        raise NotAdopted(f"{path.name} is too large to copy here.")
    written = target / path.name
    target.mkdir(parents=True, exist_ok=True)
    try:
        with open(written, "xb") as stream:
            stream.write(data)
    except FileExistsError:
        raise NotAdopted(f"{written} exists already: nothing is written over a file. Choose "
                         "another directory, or move that file first.", "test_dir") from None
    return written
