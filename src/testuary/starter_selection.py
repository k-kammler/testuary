"""The starter suites' mutation results derived from one pooled run.

``[starter] suite_mutation`` measures each accepted suite alone
(``measure``), derives each result from one pooled run (``derive``), or
both, the measured results governing. The backend stages the pool under
names no suite shares, with the way back to each suite's own names. A
derived result is the pooled matrix restricted to the suite's tests that
ran in its own coverage run (the backend's ``restrict_kill_matrix``), not
guaranteed to equal measuring the suite alone. For suites measured both ways the record lists
each mutant whose statuses differ, with the rule behind the derived one."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Set, Tuple
from langadapter.mutation import derivation_rule
from testuary.languages import mutation
from testuary.consolidation import _covered_lines_of, _diff_body


#: Schema of the persisted derivation record.
SCHEMA = "testuary.starter_pool_derivation/1"


def stage_pool(backend: Any, suites: Mapping[str, Tuple[str, str]],
               directory: Path) -> Tuple[Any, Optional[str]]:
    """Stage the suites for one pooled run into ``directory``. ``suites``
    maps each suite to its standalone file name and text. Returns the pool
    the backend staged (Java renames its test classes) and None, or None
    and why the backend could not stage it."""
    from langadapter.sources import PoolMember

    members = [PoolMember(slot=label, name=name, source=source)
               for label, (name, source) in suites.items()]
    observation = backend.source_backend().sources.stage_pool(members)
    if not observation.completed:
        return None, (f"the starter suites cannot be pooled "
                      f"({observation.status.value}: {observation.reason})")
    observation.value.write(str(directory))
    return observation.value, None


def suite_tokens(tests: Mapping[str, Iterable[str]]) -> Dict[str, Set[str]]:
    """The suite's tests that ran in its own coverage run, each with its
    coverage tokens there."""
    return {str(test): {str(token) for token in tokens} for test, tokens in tests.items()}


def covered_lines(tokens: Mapping[str, Iterable[str]]) -> Set[int]:
    """The target lines the given tests cover, from their coverage tokens."""
    lines: Set[int] = set()
    for test_tokens in tokens.values():
        lines |= _covered_lines_of(str(token) for token in test_tokens)
    return lines


def derive_suite_matrix(pool: Any, pooled_km: Optional[dict], label: str,
                        tokens: Mapping[str, Iterable[str]]) -> Optional[dict]:
    """The suite's derived matrix (the backend's ``StagedPool.derive``), or
    None when none of its tests ran: an empty subset would still
    be credited every kill no test is named for."""
    if not pooled_km or not tokens:
        return None
    return pool.derive(pooled_km, label, covered_lines=covered_lines(tokens))


def _identity(row: Mapping[str, Any]) -> tuple:
    return (row.get("operator"), row.get("line"), _diff_body(row.get("diff")))


def _key(row: Mapping[str, Any]) -> tuple:
    """How rows of two runs match: the engine's catalogue key, else the
    stable ``mutant_id``, else the row's id, which matches across runs of
    one session only while the engine's catalogue cache is on."""
    key = row.get("key")
    if key is not None:
        return ("key", json.dumps(key, sort_keys=True, default=str))
    if row.get("mutant_id") is not None:
        return ("mutant_id", str(row["mutant_id"]))
    return ("id", str(row.get("id")))


def _killers(row: Optional[Mapping[str, Any]]) -> List[str]:
    if row is None:
        return []
    return [str(k) for k in (row.get("killers") or ([row["killer"]] if row.get("killer") else []))]


def differences(pooled_km: Optional[dict], derived_km: Optional[dict], measured_km: Optional[dict],
                *, killed: Set[Any], lines: Set[int]) -> List[Dict[str, Any]]:
    """Each mutant whose derived and measured status differ, with the rule
    behind the derived status and both runs' killers. Rows match as ``_key``
    says, an id match checked by operator, line and diff. A mutant in one
    matrix only, or an id naming two mutants, is listed with that cause and
    not compared. ``killed`` and ``lines`` are what the suite kept from the
    pooled run."""
    pooled = {_key(row): row for row in (pooled_km or {}).get("mutants") or ()}
    derived = {_key(row): row for row in (derived_km or {}).get("mutants") or ()}
    measured = {_key(row): row for row in (measured_km or {}).get("mutants") or ()}
    out: List[Dict[str, Any]] = []
    for key in sorted(set(derived) | set(measured)):
        d, m = derived.get(key), measured.get(key)
        p = pooled.get(key, d)
        entry: Dict[str, Any] = {
            "mutant_id": mutation.shown_mutant_id(d if d is not None else m),
            "derived_status": d.get("status") if d is not None else None,
            "measured_status": m.get("status") if m is not None else None,
        }
        if d is not None and m is not None and mutation.shown_mutant_id(m) != mutation.shown_mutant_id(d):
            entry["measured_mutant_id"] = mutation.shown_mutant_id(m)
        if d is None or m is None:
            entry["cause"] = "mutant_in_one_matrix"
        elif key[0] == "id" and _identity(d) != _identity(m):
            entry["cause"] = "different_mutants_under_one_id"
        elif d.get("status") == m.get("status"):
            continue
        else:
            entry["derivation_rule"] = derivation_rule(p, killed, lines.__contains__)
        entry["pooled_status"] = p.get("status") if p is not None else None
        entry["pooled_killers"] = _killers(p)
        entry["measured_killers"] = _killers(m)
        out.append(entry)
    return out


def identity(km: Optional[dict]) -> Optional[Dict[str, Any]]:
    """A run's mutant count and catalogue provenance, to check row matching."""
    if km is None:
        return None
    return {"mutants": len(km.get("mutants") or ()),
            "target_file": km.get("target_file"),
            "catalogue_provenance": km.get("catalogue_provenance")}


def counts(km: Optional[dict], outcome: Optional[str] = None) -> Optional[Dict[str, Any]]:
    """A matrix's valid and killed mutants and its score, as a standalone
    measurement scores them, or None without a matrix. A matrix stamped
    incomplete gets no counts, since its resolved mutants are no sample: it
    carries the stamp and the run's ``outcome`` when known."""
    if km is None:
        return None
    if km.get(mutation.INCOMPLETE_KEY):
        entry = {"valid": None, "killed": None, "score": None,
                 "incomplete": km[mutation.INCOMPLETE_KEY]}
        if outcome is not None:
            entry["outcome"] = outcome
        return entry
    valid = mutation.valid_ids(km)
    return {
        "valid": len(valid),
        "killed": sum(1 for row in km.get("mutants") or ()
                      if row.get("id") in valid and row.get("status") in mutation.DETECTED_STATUSES),
        "score": mutation.matrix_score_percent(km),
    }


def _mean(values: Iterable[Optional[float]]) -> Optional[float]:
    present = [value for value in values if value is not None]
    return sum(present) / len(present) if present else None


def _staged_files(pool: Any, label: str) -> Optional[List[Dict[str, Any]]]:
    if pool is None or label not in pool.slots:
        return None
    return [{"original": f.original, "path": f.path, "sha256": f.sha256,
             "renamed": [list(pair) for pair in f.renamed],
             "self_references": list(getattr(f, "self_references", ()) or ())}
            for f in pool.staged(label)]


def record(*, requested: str, used: str, reason: Optional[str],
           attribution: Mapping[str, Optional[str]],
           slots_by_label: Mapping[str, int], requested_slots: int,
           pool_names: Mapping[str, str], pooled_km: Optional[dict],
           derived: Mapping[str, Optional[dict]],
           measured: Optional[Mapping[str, dict]],
           kept: Mapping[str, Tuple[Set[Any], Set[int]]],
           seconds: Mapping[str, Optional[float]], pool: Any = None,
           pooled_limit: Optional[Mapping[str, Any]] = None,
           measured_limits: Optional[Mapping[str, Any]] = None,
           measurement_unavailable: Optional[str] = None,
           measured_outcomes: Optional[Mapping[str, str]] = None) -> Dict[str, Any]:
    """The persisted derivation record of one starter phase. ``measured``
    holds the per-suite matrices under ``both``, else None. A missing suite
    or result is None, never zero, and a missing derived result says why.
    ``kept`` holds each suite's kills and covered lines from the pooled run.
    The ``budget`` records hold each run's baseline time and per-mutant timeout: the
    pooled run's longer baseline gives it longer limits, so a timeout-only
    difference can be a different limit."""
    by_slot = {slot: label for label, slot in slots_by_label.items()}
    slots = []
    for slot in range(1, requested_slots + 1):
        label = by_slot.get(slot)
        entry: Dict[str, Any] = {"slot": slot, "accepted": label is not None}
        if label is not None:
            derived_km = derived.get(label)
            entry["pooled_name"] = pool_names.get(label)
            entry["staged_files"] = _staged_files(pool, label)
            entry["derived"] = counts(derived_km)
            if derived_km is None:
                entry["derivation_unavailable"] = (
                    reason if reason is not None
                    else "no per-test evidence for this suite" if label not in kept
                    else "no test of this suite ran in its own coverage run")
            if measured is not None:
                measured_km = measured.get(label)
                entry["measured"] = counts(measured_km, (measured_outcomes or {}).get(label))
                entry["measured_identity"] = identity(measured_km)
                entry["measured_budget"] = (measured_limits or {}).get(label)
                if derived_km is not None and measured_km is not None:
                    killed, lines = kept.get(label, (set(), set()))
                    found = differences(pooled_km, derived_km, measured_km,
                                        killed=killed, lines=lines)
                    entry["differences"] = found
                    rules: Dict[str, int] = {}
                    for item in found:
                        key = item.get("derivation_rule") or item["cause"]
                        rules[key] = rules.get(key, 0) + 1
                    entry["differences_by_rule"] = rules
        slots.append(entry)
    accepted = [entry for entry in slots if entry["accepted"]]
    pooled = counts(pooled_km)
    if pooled is not None:
        pooled.update(identity=identity(pooled_km), budget=dict(pooled_limit or {}) or None,
                      files=sorted(pool_names.values()), order="test files in name order")
    return {
        "schema": SCHEMA,
        "method_requested": requested,
        "method_used": used,
        "derivation_unavailable": reason,
        # Why the governing per-suite measurement is missing, when it is: the
        # derived values above then govern nothing.
        "measurement_unavailable": measurement_unavailable,
        "attribution": dict(attribution),
        "pooled": pooled,
        "seconds": dict(seconds),
        "mean_score": {
            "derived": _mean((entry["derived"] or {}).get("score") for entry in accepted),
            "measured": (_mean((entry.get("measured") or {}).get("score") for entry in accepted)
                         if measured is not None else None),
        },
        "slots": slots,
    }
