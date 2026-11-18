"""The recoveries that make a suite load, and the words of the note on
tests that leave shared state changed.

A suite whose run gives no test outcome and did not time out gets the
backend's recoveries in turn (:func:`ensure_loadable`), each result
finalized, validated and probed again. A suite that loads only in its
recovered form is kept that way (:func:`keep_if_loadable`). The probe of a
candidate is its normal run's listing of failing tests
(:func:`scratch_probe`). The normal run's leak record, from the
language's observer, which restores nothing, reaches the model through
:func:`leak_entry` and :func:`describe_leak`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple


def leak_entry(record: Any) -> Dict[str, Any]:
    """A backend leak record as the records and the model see it: the test
    and only what it left changed."""
    entry: Dict[str, Any] = {"test": record.test}
    for name in ("added", "rebound", "removed", "xarray_options",
                 "numpy_errors", "numpy_print"):
        values = list(getattr(record, name) or ())
        if values:
            entry[name] = values
    return entry


def describe_leak(entry: Dict[str, Any]) -> str:
    """What a leaking test left changed, in words, naming at most the
    ``leak_names`` limit of names per kind."""
    from testuary.agent import _limits

    limit = _limits.value("leak_names")

    def names_of(names):
        shown, omitted = _limits.cut_items(names, limit, what="further names")
        return ", ".join(shown) + (f" {omitted}" if omitted else "")

    parts = []
    for key, label in (("rebound", "rebound"), ("added", "added"),
                       ("removed", "removed")):
        names = entry.get(key) or []
        if names:
            parts.append(f"{label} {names_of(names)} of the module under test")
    for key, label in (("xarray_options", "xarray options"),
                       ("numpy_errors", "numpy error settings"),
                       ("numpy_print", "numpy print settings")):
        names = entry.get(key) or []
        if names:
            parts.append(f"changed {label} {names_of(names)}")
    return "; ".join(parts) or "left shared state changed"


#: A probe's answer: the failing test ids (None when the run gave no test
#: outcome), whether the run timed out, and the run's outcome as reported.
Probe = Tuple[Optional[List[str]], bool, Optional[str]]


@dataclass
class Loadable:
    """The suite after the recoveries: its source, whether it loads, the
    last probe's failing ids, timeout and outcome, what each recovery
    removed, and the validation that rejected a recovered source, if any."""

    source: str
    loads: bool
    ids: Optional[List[str]]
    timed_out: bool
    outcome: Optional[str]
    removed: Dict[str, List[str]] = field(default_factory=dict)
    rejected: Any = None


def ensure_loadable(backend, source: str, source_dir: str, *, timeout: int,
                    probe: Callable[[str], Probe], validate: Callable[[str], Any],
                    log: Callable[[str], None], label: str,
                    first: Optional[Probe] = None) -> Loadable:
    """Probe ``source``. When the run gives no test outcome and did not time
    out, apply the backend's recoveries in turn, each result finalized,
    validated and probed again, until one loads. ``first`` is a probe of
    ``source`` the caller already made. A recovered source that fails
    validation ends the attempt with ``rejected`` set."""
    ids, timed_out, outcome = first if first is not None else probe(source)
    result = Loadable(source, ids is not None, ids, timed_out, outcome)
    if result.loads or timed_out:
        return result
    steps = (
        ("unresolvable_imports", "import prune",
         lambda text: backend.prune_unresolvable_imports(text, source_dir)),
        ("collection_breakers", "collection-breaker prune",
         lambda text: backend.prune_collection_breakers(text, source_dir, timeout=timeout)),
    )
    current = source
    for key, name, recover in steps:
        recovered, dropped = recover(current)
        if not dropped:
            continue
        log(f"[{label}] {name} dropped {', '.join(dropped)}")
        result.removed[key] = list(dropped)
        candidate = backend.finalize_test_source(recovered)
        validation = validate(candidate)
        if not validation.ok:
            result.source, result.rejected = candidate, validation
            return result
        current = candidate
        ids, timed_out, outcome = probe(current)
        result.source, result.ids, result.timed_out, result.outcome = (
            current, ids, timed_out, outcome)
        result.loads = ids is not None
        if result.loads or timed_out:
            break
    return result


def fewer_tests(backend, before: str, after: str) -> bool:
    """Whether ``after`` may hold fewer tests than ``before`` by the
    backend's static count. An unavailable count cannot show that no test
    was removed, so it counts as fewer."""
    counted = [backend.test_metrics(source).get("test_count") for source in (before, after)]
    return None in counted or counted[1] < counted[0]


def scratch_probe(backend, source_dir: str, target_file_path: str, *,
                  timeout: int, stage_as_delivered: bool = False,
                  ) -> Callable[[str], Probe]:
    """A probe that runs each candidate alone in a fresh directory of the work
    directory and reads its normal run's failing tests and outcome
    (:func:`testuary.normal_run.failing_ids`)."""
    from testuary import normal_run, project

    def probe(source: str) -> Probe:
        with project.scratch_dir(prefix="testuary_loadable_") as scratch_dir:
            path = backend.scratch_test_path(
                str(Path(scratch_dir)), source, target_file_path,
                stage_as_delivered=stage_as_delivered)
            path.write_text(source, encoding="utf-8")
            report: Dict[str, Any] = {}
            ids = normal_run.failing_ids(
                backend, [str(path)], source_dir, timeout=timeout,
                target=target_file_path, report=report)
        return ids, report.get("outcome") == "timed_out", report.get("outcome")

    return probe


def keep_if_loadable(backend, source: str, source_dir: str, *, timeout: int,
                     target_file_path: str, validate: Callable[[str], Any],
                     log: Callable[[str], None], label: str,
                     stage_as_delivered: bool = False,
                     ) -> Tuple[str, Dict[str, List[str]]]:
    """``source``, or its recovered form when only that loads, with what the
    recoveries removed (empty when the source was kept)."""
    result = ensure_loadable(
        backend, source, source_dir, timeout=timeout, validate=validate,
        log=log, label=label,
        probe=scratch_probe(backend, source_dir, target_file_path,
                            timeout=timeout, stage_as_delivered=stage_as_delivered))
    if result.loads and result.removed and result.rejected is None:
        return result.source, result.removed
    return source, {}
