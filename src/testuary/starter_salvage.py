"""The salvage of a generated starter suite to a green subset: refused and
unsound cases leave, and a run must confirm the rest."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple
from testuary import evidence as validity
from testuary.generation_evidence import source_sha256
from testuary import normal_run
from testuary import loadable
from testuary.consolidation import (
    VALIDATION_ATTEMPTS, ValidationUnresolved, ValidityUnresolved, _without_tests, validate_judged)

#: The recorded disposition of a suite run that gave no per-test outcome, by
#: the run's outcome: a suite that does not compile or collect, or holds no
#: test, is judged. A run that lacked its setup, timed out or could not start
#: says nothing about the suite.
_NO_OUTCOME = {
    "collection_failed": ("judged_rejection", "the suite does not compile or collect"),
    "no_tests": ("judged_rejection", "the suite holds no test"),
    "missing_setup": ("unavailable", "the suite run lacked its setup"),
    "unsupported": ("unavailable", "the suite run is not supported here"),
    "timed_out": ("unavailable", "the suite run timed out"),
    "infrastructure_failure": ("unavailable", "the suite run failed in its infrastructure"),
    "os_error": ("unavailable", "the suite run could not start"),
}


def _no_outcome_disposition(outcome: Optional[str]) -> Tuple[str, str]:
    return _NO_OUTCOME.get(outcome or "", ("unavailable", "the suite run gave no test outcome"))


#: The recorded reason of a suite whose implementation observation lies
#: outside every test, which no test removal can take out.
_VALIDITY_OUTSIDE_TESTS = ("the generated-test validity policy rejected an implementation "
                           "observation outside every test (module level or a shared helper)")


def _validator_reason(cleanup_out: Optional[Dict[str, Any]], validation) -> None:
    """Record why the target validator rejected a candidate: a syntax error
    is a judged rejection, any other status names itself."""
    if cleanup_out is None:
        return
    status = getattr(validation, "status", None)
    if status == "syntax_error":
        cleanup_out.update(disposition_basis="judged_rejection",
                           disposition_reason="target parser rejected the source")
    else:
        cleanup_out["disposition_reason"] = f"the target validator rejected the source: {status}"


def _unsound_case_names(nodes) -> List[str]:
    """The tests that emitted unsound-state warnings, from their pytest node
    ids (``path::Class::test[id]``), as ``Class.test[id]``: a parametrized
    test's case keeps its id, so the removal can take out that case alone
    where the language can (``prune_tests``)."""
    names = set()
    for node in nodes:
        _path, separator, rest = node.partition("::")
        if not separator or not rest:
            continue
        qualname, bracket, params = rest.partition("[")
        names.add(qualname.replace("::", ".") + bracket + params)
    return sorted(names)


def _normalization_record(result) -> Dict[str, Any]:
    return {
        "changed": bool(result.changed),
        "parse_ok": bool(result.parse_ok),
        "parse_error": result.parse_error,
        "rule_version": result.rule_version,
        "before_sha256": result.before_sha256,
        "after_sha256": result.after_sha256,
        "removed_bindings": [
            asdict(binding) for binding in result.removed_bindings
        ],
    }


class _ConfirmationUnresolved(Exception):
    """The final execution did not establish a complete suite verdict."""

    def __init__(self, source, run):
        super().__init__("cleaned suite confirmation did not establish a successful execution")
        self.source = source
        self.observation = {name: getattr(run, name, None) for name in (
            "ok", "returncode", "timed_out", "elapsed_s", "collected", "passed", "failed", "errors",
            "line_coverage", "branch_coverage",
        )}


def _require_confirmed_run(source, run):
    """Do not turn a partial pass count or a failed start into a suite verdict."""
    if (getattr(run, "ok", None) is not True
            or getattr(run, "timed_out", None) is not False
            or type(getattr(run, "passed", None)) is not int or run.passed < 0
            or type(getattr(run, "failed", None)) is not int or run.failed != 0
            or type(getattr(run, "errors", None)) is not int or run.errors != 0):
        raise _ConfirmationUnresolved(source, run)


def _clean_suite(backend, source: str, target_file_path: str,
                 source_dir: str, scratch: Path, *, timeout: int,
                 log: Callable[[str], None],
                 reject_unsound: bool = True,
                 final_attempt: bool = False,
                 failing_cleanup_rounds: int = 3,
                 normalization_out: Optional[Dict[str, Any]] = None,
                 validity_out: Optional[Dict[str, Any]] = None,
                 cleanup_out: Optional[Dict[str, Any]] = None,
                 ) -> Tuple[str, Optional[str]]:
    """Salvage one suite. Unjudged parsing or execution stays unresolved.

    No new response is requested for either cause. The existing resume path
    processes the saved original again, with its original accounting.
    """
    try:
        return _salvage_suite(
            backend, source, target_file_path, source_dir, scratch,
            timeout=timeout, log=log, reject_unsound=reject_unsound,
            final_attempt=final_attempt,
            failing_cleanup_rounds=failing_cleanup_rounds,
            normalization_out=normalization_out, validity_out=validity_out,
            cleanup_out=cleanup_out,
        )
    except ValidationUnresolved as exc:
        validity_gate = isinstance(exc, ValidityUnresolved)
        judge = "the validity gate" if validity_gate else "the target validator"
        log(f"[starter] {judge} did not judge the suite: {exc}")
        if cleanup_out is not None:
            cleanup_out["validation_unresolved"] = str(exc)
            cleanup_out["unresolved_stage"] = (
                "generated_test_validity" if validity_gate else "target_validation")
        return "validation_unresolved", None
    except _ConfirmationUnresolved as exc:
        log(f"[starter] suite confirmation remains unresolved: {exc}")
        if cleanup_out is not None:
            cleanup_out.update(
                validation_unresolved=str(exc), unresolved_stage="final_suite_confirmation",
                disposition_basis="unavailable", disposition_reason=str(exc),
                confirmation={"source_sha256": source_sha256(exc.source),
                              "observation": exc.observation},
                confirmation_source=exc.source,
            )
        return "validation_unresolved", None


def _salvage_suite(backend, source: str, target_file_path: str,
                   source_dir: str, scratch: Path, *, timeout: int,
                   log: Callable[[str], None],
                   reject_unsound: bool,
                   final_attempt: bool,
                   failing_cleanup_rounds: int,
                   normalization_out: Optional[Dict[str, Any]],
                   validity_out: Optional[Dict[str, Any]],
                   cleanup_out: Optional[Dict[str, Any]],
                   ) -> Tuple[str, Optional[str]]:
    """Salvage one generated suite to a green subset, deterministically.

    The salvage runs a fixed pipeline, each stage a last resort applied only
    while the suite is not yet green:

    1. ``salvage_syntax``: bounded local repair of a truncated or malformed
       suite (``discarded_syntax`` when unrepairable).
    2. ``strip_module_level_skips``: drop module-scope skip guards that would
       skip the whole file at collection.
    3. on a collection failure, ``prune_unresolvable_imports`` then
       ``prune_collection_breakers`` (``discarded_collection`` when neither
       makes it collect).
    4. strip the tests the normal run (:mod:`testuary.normal_run`) fails for
       the configured number of rounds on ordinary generation
       attempts. On the final attempt, continue while each round removes
       another failing test (``unsalvageable`` when no passing subset can be
       obtained).
    5. ``confirm_sound``: reject a suite that runs zero tests, and strip tests
       emitting unsound-state warnings.

    Returns ``(classification, green_source_or_None)`` where the
    classification is one of ``all_passing`` / ``pruned`` /
    ``discarded_syntax`` / ``discarded_collection`` /
    ``invalid_generated_test`` / ``timeout`` / ``unsound`` /
    ``unsalvageable``. A validation the target validator
    did not judge raises ``ValidationUnresolved``, and a final run that does
    not establish a complete suite verdict raises ``_ConfirmationUnresolved``.
    A run that consumes the whole
    timeout is reported as ``timeout`` rather than a collection failure:
    the suite may be healthy but slow, and the knob for that is
    ``suite_run_timeout_seconds`` rather than salvage. A green subset still
    emitting unsound-test warnings (the run's ``unsound`` entries:
    passing-looking tests with leaked state) has the emitting tests
    stripped too. ``unsound`` is the
    discard when the warnings cannot be attributed or survive the
    strip."""
    if cleanup_out is not None:
        cleanup_out.update({
            "disposition_basis": "unavailable",
            "disposition_reason": "cleanup judgement unavailable",
            "configured_failing_test_cleanup_rounds": int(
                failing_cleanup_rounds
            ),
            "final_generation_attempt": bool(final_attempt),
            "failing_test_cleanup_rounds_used": 0,
            "warning_cleanup_rounds_used": 0,
        })
    if failing_cleanup_rounds < 1:
        raise ValueError("failing_cleanup_rounds must be a positive integer")

    salvage: Dict[str, Any] = {}
    for _ in range(VALIDATION_ATTEMPTS):
        repaired = backend.salvage_syntax(source, source_dir,
                                          report=salvage)
        if repaired is not None or salvage.get("status") != "validator_failed":
            break
    if repaired is None:
        if salvage.get("status") == "validator_failed":
            raise ValidationUnresolved(
                f"validator_failed: {salvage.get('message')}")
        if cleanup_out is not None:
            cleanup_out.update(disposition_basis="judged_rejection",
                               disposition_reason="the suite does not parse and could not be repaired")
        return "discarded_syntax", None
    if repaired != source:
        log("[starter] syntax repaired (bounded local repair)")

    # The file Testuary delivers is named for the target. A language whose
    # file name must match a declaration of the source (Java's public class)
    # gets the declaration renamed to that name, as the prompt asks for it,
    # so every staged copy and the delivered file carry the same name.
    conform = getattr(backend, "conform_delivered_name", None)
    repaired, old_name, refusal = (conform(repaired, target_file_path)
                                   if callable(conform) else (repaired, None, None))
    if refusal is not None:
        log(f"[starter] the test class {old_name} cannot be renamed to the "
            f"delivered file's name: {refusal}")
        if cleanup_out is not None:
            cleanup_out.update(
                disposition_basis="judged_rejection",
                disposition_reason=("the test class cannot be renamed to the delivered "
                                    f"file's name: {refusal}"))
        return "invalid_generated_test", None
    if old_name is not None:
        log(f"[starter] test class {old_name} renamed to the delivered file's name")
        if cleanup_out is not None:
            cleanup_out["renamed_class"] = old_name

    normalized = backend.normalize_shadowed_definitions(repaired)
    if normalization_out is not None:
        normalization_out.update(_normalization_record(normalized))
    if not normalized.parse_ok:
        if cleanup_out is not None:
            cleanup_out.update(disposition_basis="judged_rejection",
                               disposition_reason="the suite does not parse")
        return "discarded_syntax", None
    if normalized.changed:
        log(
            "[starter] removed "
            f"{len(normalized.removed_bindings)} shadowed definition(s) "
            "before collection"
        )
    target_identity = backend.generated_test_target_identity(
        target_file_path, source_dir,
    )
    analyzed_source = normalized.source
    validity_findings = backend.analyze_generated_test_validity(
        analyzed_source, target_identity,
    )
    removed_invalid_tests: List[str] = []

    def noted(outcome):
        # Why a named case's test went whole is recorded with the cleanup.
        if outcome.detail:
            log(f"[starter] {outcome.detail}")
            if cleanup_out is not None:
                cleanup_out.setdefault("prune_details", []).append(outcome.detail)

    def prune(text: str, names) -> str:
        # The language's removal.
        from testuary.languages.generic import prune_report

        outcome = prune_report(backend, text, names)
        noted(outcome)
        return outcome.source

    def invalid(basis: str, reason: str, *, removed=(), final_source=None):
        # A suite the generated-test validity policy leaves no valid
        # subset of: the findings and the recorded reason.
        if validity_out is not None:
            validity_out.update(validity.direct_record(
                identity=target_identity,
                analyzed_source=analyzed_source,
                findings=validity_findings,
                removed_tests=removed,
                final_source=final_source,
                stage="starter.generated_suite_rejected",
            ))
        if cleanup_out is not None:
            cleanup_out.update(disposition_basis=basis, disposition_reason=reason)
        log(f"[starter] {reason}")
        return "invalid_generated_test", None

    if validity_findings:
        unattributed = [
            finding for finding in validity_findings
            if not finding.enclosing_test
        ]
        if unattributed:
            # No test removal can take out a module-level or shared-helper
            # observation: the suite itself is judged.
            return invalid("judged_rejection", _VALIDITY_OUTSIDE_TESTS)
        removed_invalid_tests = sorted({
            finding.enclosing_test for finding in validity_findings
            if finding.enclosing_test
        })
        named = ", ".join(removed_invalid_tests)
        pruned = prune(analyzed_source, removed_invalid_tests)
        if pruned == analyzed_source:
            return invalid("unavailable", "the tests that break the generated-test validity "
                           f"policy could not be removed: {named}")
        normalized = backend.normalize_shadowed_definitions(pruned)
        if not normalized.parse_ok:
            return invalid("unavailable", "the suite does not parse after removing the tests "
                           f"that break the generated-test validity policy: {named}",
                           removed=removed_invalid_tests, final_source=pruned)
        residual = backend.analyze_generated_test_validity(
            normalized.source, target_identity,
        )
        if residual:
            if any(not finding.enclosing_test for finding in residual):
                return invalid("judged_rejection", f"{_VALIDITY_OUTSIDE_TESTS} after removing "
                               f"{named}", removed=removed_invalid_tests,
                               final_source=normalized.source)
            remaining = ", ".join(sorted({finding.enclosing_test for finding in residual}))
            return invalid("unavailable", "the generated-test validity policy still rejected "
                           f"{remaining} after removing {named}",
                           removed=removed_invalid_tests, final_source=normalized.source)
        log(
            "[starter] removed invalid generated test(s): "
            + ", ".join(removed_invalid_tests)
        )

    def rejected(what: str, validation):
        # The target validator rejected the candidate: a discard.
        log(f"[starter] {what}: {validation.status}: {validation.message}")
        _validator_reason(cleanup_out, validation)
        return "discarded_syntax", None

    repaired = backend.finalize_test_source(normalized.source)
    validation = validate_judged(backend, repaired, source_dir)
    if not validation.ok:
        return rejected("target parser rejected repaired candidate", validation)

    # Neutralize module-scope skips (a ``pytest.importorskip`` for an absent or
    # versionless package) before measuring: at module scope they skip the whole
    # file, which reads as green (no failures) while collecting zero tests, and
    # merged into the pool one such line skips every suite. Removing them lets the
    # dependent tests fail individually so per-test stripping handles them.
    deskipped, n_modskip = backend.strip_module_level_skips(repaired)
    if n_modskip:
        log(f"[starter] removed {n_modskip} module-level skip/importorskip "
            f"statement(s) that would skip the whole file at collection")
        repaired = backend.finalize_test_source(deskipped)
        validation = validate_judged(backend, repaired, source_dir)
        if not validation.ok:
            return rejected("target parser rejected de-skipped candidate", validation)

    scratch.mkdir(parents=True, exist_ok=True)
    suite = backend.staged_test_path(str(scratch), "candidate", repaired)
    # The outcome of the latest suite run, which says why it gave no ids.
    last_run: Dict[str, Optional[str]] = {"outcome": None}
    current = repaired

    def run_ids(src_text: str):
        # The normal run's failing tests: a fresh scratch directory's first
        # listing is the same run as a later attempt's.
        suite.write_text(src_text, encoding="utf-8")
        report: Dict[str, Any] = {}
        ids = normal_run.failing_ids(
            backend, [str(suite)], source_dir, timeout=timeout,
            target=target_file_path, report=report,
        )
        last_run["outcome"] = report.get("outcome")
        last_run["reason"] = report.get("reason")
        last_run["active"] = report.get("active_at_end")
        # The runner says whether the timeout ended the run.
        timed_out = ids is None and last_run["outcome"] == "timed_out"
        return ids, timed_out

    # The tests a run the timeout ended named as running then, removed.
    hung: List[str] = []

    def without_hangs(src_text: str, ids, timed_out):
        """A run the timeout ended that names the test then running (the
        per-test timeout journal): that test is removed and the rest run
        again, as a failing test is, at most ``failing_cleanup_rounds``
        times for the suite, since each such run costs the whole timeout.
        Returns the source and its last run's ids and timeout."""
        while (ids is None and timed_out and last_run.get("active")
               and len(hung) < failing_cleanup_rounds):
            test = last_run["active"]
            pruned = backend.finalize_test_source(prune(src_text, [test]))
            if pruned == src_text or not validate_judged(backend, pruned, source_dir).ok:
                break
            hung.append(test)
            if cleanup_out is not None:
                cleanup_out["timeout_removed"] = list(hung)
            log(f"[starter] the suite run timed out while {test} ran, so it is removed "
                "and the rest runs again")
            src_text = pruned
            ids, timed_out = run_ids(src_text)
        return src_text, ids, timed_out

    def confirm_sound(green_src: str, label: str):
        """Green is necessary, not sufficient: tests that pass while
        leaking state (the unsound-warning set) get stripped like
        failures, with one re-confirmation round. Gated on
        ``reject_unsound_async_green`` (the same knob the editing phases
        honor), so the starter pool and those phases agree when the guard is
        turned off."""
        finalized = backend.finalize_test_source(green_src)
        validation = validate_judged(backend, finalized, source_dir)
        if not validation.ok:
            return rejected("finalized candidate failed target validation", validation)
        if finalized != green_src:
            log("[starter] finalized the cleaned candidate before acceptance")
        green_src = finalized
        suite.write_text(green_src, encoding="utf-8")
        run = normal_run.run(backend, [str(suite)], source_dir,
                             timeout=timeout, target=target_file_path)
        _require_confirmed_run(green_src, run)
        # A suite with no failures but zero PASSING tests (everything skipped, or
        # nothing collected) is green by vacuity: it verifies nothing and, in the
        # pool merge, would waste a suite slot. Discard it so the refill
        # (refill_failed_suites) regenerates a real one. Checked before the
        # unsound gate so it holds when that is off.
        if run.passed == 0:
            if cleanup_out is not None:
                cleanup_out.update(disposition_basis="judged_rejection",
                                   disposition_reason="completed suite ran no passing tests")
            log("[starter] the salvaged suite runs zero tests "
                "(everything skipped or nothing collected) and is discarded")
            return "unsalvageable", None
        def accept(result_label: str, accepted_source: str):
            # Each caller has confirmed its run, with a passing test.
            if cleanup_out is not None:
                cleanup_out.update(disposition_basis="accepted_cleaned",
                                   disposition_reason="completed cleaned suite passed")
            if validity_out is not None:
                validity_out.clear()
                validity_out.update(validity.direct_record(
                    identity=target_identity,
                    analyzed_source=analyzed_source,
                    findings=validity_findings,
                    removed_tests=removed_invalid_tests,
                    final_source=accepted_source,
                    audit_flags=backend.screen_generated_test_source(
                        accepted_source, target_identity,
                    ),
                    stage="starter.accepted_suite",
                ))
            return result_label, accepted_source

        if not reject_unsound:
            return accept(label, green_src)
        if not run.unsound:
            return accept(label, green_src)
        pairs = [(test, warn) for test, warn in run.unsound if test is not None]
        names = _unsound_case_names(node for node, _ in pairs)
        def unsound(reason: str):
            if cleanup_out is not None:
                cleanup_out["disposition_reason"] = reason
            return "unsound", None

        if not names:
            log("[starter] unsound warnings with no attributable test, so the "
                "suite is discarded")
            return unsound("unsound-state warnings named no test")
        log(f"[starter] unsound-warning strip: {', '.join(names[:5])}")
        if cleanup_out is not None:
            cleanup_out["warning_cleanup_rounds_used"] = 1
        stripped = backend.finalize_test_source(
            prune(green_src, names)
        )
        validation = validate_judged(backend, stripped, source_dir)
        if not validation.ok:
            return unsound("the suite failed validation after removing the tests "
                           "with unsound-state warnings")
        ids2, timed_out2 = run_ids(stripped)
        if ids2 is None:
            reason = ("the suite run timed out" if timed_out2
                      else _no_outcome_disposition(last_run["outcome"])[1])
            return unsound(f"{reason} after removing the tests with unsound-state warnings")
        if ids2 != []:
            return unsound("tests failed after removing the tests with unsound-state warnings")
        run2 = normal_run.run(backend, [str(suite)], source_dir,
                              timeout=timeout, target=target_file_path)
        _require_confirmed_run(stripped, run2)
        if run2.passed == 0:
            if cleanup_out is not None:
                cleanup_out.update(disposition_basis="judged_rejection",
                                   disposition_reason="completed suite ran no passing tests")
            log(
                "[starter] unsound-warning cleanup left no passing tests, so the "
                "suite is discarded"
            )
            return "unsalvageable", None
        if run2.unsound:
            return unsound("unsound-state warnings remained after removing the tests "
                           "that emitted them")
        return accept("pruned", stripped)

    ids, timed_out = run_ids(current)
    current, ids, timed_out = without_hangs(current, ids, timed_out)
    before_recovery = current
    # A language without a collection step (C) names a collection failure
    # by its first failing run's reason, which a discarded attempt's record
    # keeps as the log line shows it.
    run_reason = None
    if ids is None and not timed_out:
        reason = backend.describe_collection_failure(
            [str(suite)], source_dir, timeout=timeout,
        )
        if reason is None:
            reason = run_reason = last_run["reason"]
        if reason:
            log(f"[starter] collection failed: {reason}")

        def probe(text: str):
            found, slow = run_ids(text)
            return found, slow, last_run["outcome"]

        # The backend's recoveries, as for every automatic transformation.
        recovery = loadable.ensure_loadable(
            backend, current, source_dir, timeout=timeout, probe=probe,
            validate=lambda text: validate_judged(backend, text, source_dir),
            log=log, label="starter", first=(ids, timed_out, last_run["outcome"]))
        if recovery.removed and cleanup_out is not None:
            cleanup_out["collection_recovery_removed"] = recovery.removed
        if recovery.rejected is not None:
            _validator_reason(cleanup_out, recovery.rejected)
            return "discarded_syntax", None
        current, ids, timed_out = recovery.source, recovery.ids, recovery.timed_out
    if ids is None:
        if cleanup_out is not None:
            basis, reason = _no_outcome_disposition("timed_out" if timed_out else last_run["outcome"])
            cleanup_out.update(disposition_basis=basis, disposition_reason=reason)
            if run_reason:
                cleanup_out["collection_failure_reason"] = run_reason
        if timed_out:
            return "timeout", None
        return "discarded_collection", None
    # A suite the validity policy or the collection recovery removed tests
    # from is stripped, as one whose failing tests the cleanup removes.
    first_label = ("pruned" if removed_invalid_tests or hung or (
        current != before_recovery and loadable.fewer_tests(backend, before_recovery, current))
        else "all_passing")

    rounds_used = [0]

    def settle(current: str, ids: List[str], label: str):
        """Strip the failing tests, then confirm the green suite sound."""
        if ids == []:
            return confirm_sound(current, label)
        # Ordinary attempts use the configured cleanup bound. The final
        # attempt cannot spend another model request, so keep a
        # progressively shrinking suite while every round removes at least
        # one reported failing test.
        cleanup_round = 0
        seen_sources = {current}

        def unsalvageable(reason: str):
            # The recorded reason of an ending the cleanup reached by itself.
            if cleanup_out is not None:
                cleanup_out["disposition_reason"] = reason
            return "unsalvageable", None

        while final_attempt or cleanup_round < failing_cleanup_rounds:
            # The removal step the fallback shares. The cleanup keeps its own
            # counter, progress check and records.
            stripped, pruned = _without_tests(backend, current, ids)
            noted(pruned)
            if stripped in seen_sources:
                log("[starter] failing-test cleanup made no progress")
                return unsalvageable("failing-test cleanup removed no further test")
            current = stripped
            seen_sources.add(current)
            cleanup_round += 1
            rounds_used[0] += 1
            if cleanup_out is not None:
                cleanup_out["failing_test_cleanup_rounds_used"] = (
                    rounds_used[0])
            validation = validate_judged(backend, current, source_dir)
            if not validation.ok:
                return unsalvageable(
                    f"the suite failed validation after cleanup round {cleanup_round}")
            ids, timed_out = run_ids(current)
            current, ids, timed_out = without_hangs(current, ids, timed_out)
            seen_sources.add(current)
            if ids == []:
                return confirm_sound(current, "pruned")
            if ids is None:
                basis, reason = _no_outcome_disposition(
                    "timed_out" if timed_out else last_run["outcome"])
                if timed_out:
                    if cleanup_out is not None:
                        cleanup_out.update(
                            disposition_basis=basis,
                            disposition_reason=f"{reason} after cleanup round {cleanup_round}")
                    return "timeout", None
                if cleanup_out is not None:
                    cleanup_out["disposition_basis"] = basis
                    # The compiler's text, which a run reports only where
                    # the language has no collection step.
                    if last_run["reason"]:
                        cleanup_out["collection_failure_reason"] = last_run["reason"]
                return unsalvageable(f"{reason} after cleanup round {cleanup_round}")
        return unsalvageable(
            f"failing tests remained after {cleanup_round} cleanup rounds")

    return settle(current, ids, first_label)
