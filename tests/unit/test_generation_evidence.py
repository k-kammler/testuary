"""Fixed-slot generation exports do not change generation or measurements."""
import copy
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from langadapter.sources import SuiteNormalization

from testuary import generation_evidence as evidence
from testuary import starters
from testuary import starter_salvage
from testuary.languages import SourceValidation, TestRunResult as SuiteRun
from testuary.languages.generic import PruneOutcome
from test_starter_archives import SOURCES, _ReachedConsolidation, _pool, drive as _archive_drive


ID = "0123456789abcdef0123456789abcdef:0"
ID2 = "0123456789abcdef0123456789abcdef:1"
REAL_CLEAN = starter_salvage._clean_suite


@pytest.fixture
def drive(tmp_path, monkeypatch):
    return _archive_drive.__wrapped__(tmp_path, monkeypatch)


def section(root, *, count=1):
    slots = evidence.SuiteSlots(root, count)
    slots.record(1, ID, classification="pruned", basis="accepted_cleaned",
                 original_source="original", cleaned_source="cleaned\r\n", suffix=".java")
    slots.finish()
    return {"suite_slots": slots.snapshot({1: 1})}


def test_cleaned_source_is_exact_retained_text_not_latest_file(tmp_path):
    manifest = section(tmp_path)
    slots = evidence.read_suite_slots(manifest, tmp_path)
    reference = slots["slots"][0]["accepted_source"]
    (tmp_path / "suite_001" / "cleaned.java").write_text("overwritten", encoding="utf-8")
    assert evidence.read_cleaned_source(reference, tmp_path) == "cleaned\r\n"
    assert reference["attempt_id"] == evidence.paid_attempt(ID)
    slots["slots"][0]["state"] = "unavailable"
    assert manifest["suite_slots"]["slots"][0]["state"] == "accepted"


@pytest.mark.parametrize("change", [
    lambda m: m["suite_slots"].update(version=True),
    lambda m: m["suite_slots"].update(version=2),
    lambda m: m["suite_slots"].update(requested_slots=2),
    lambda m: m["suite_slots"]["slots"][0].update(state="rejected"),
    lambda m: m["suite_slots"]["slots"][0].update(terminal_attempt=None),
    lambda m: m["suite_slots"]["slots"][0].update(attempts_used=0),
    lambda m: m["suite_slots"]["slots"][0]["accepted_source"].update(source_sha256="unknown"),
    lambda m: m["suite_slots"]["slots"][0]["attempts"][0].update(original_source_sha256="unknown"),
    lambda m: m["suite_slots"]["slots"][0]["attempts"][0].update(disposition_basis="unavailable"),
    lambda m: m["suite_slots"]["slots"][0].update(state=[]),
    lambda m: m["suite_slots"]["slots"][0].update(disposition_basis={}),
    lambda m: m["suite_slots"]["slots"][0]["terminal_attempt"].update(logical_request_idx=True),
    lambda m: m["suite_slots"]["slots"][0]["accepted_source"].update(slot=True),
])
def test_contradictory_or_unknown_evidence_refused(change, tmp_path):
    manifest = section(tmp_path)
    change(manifest)
    with pytest.raises(evidence.GenerationEvidenceError):
        evidence.read_suite_slots(manifest)


@pytest.mark.parametrize("path", ["../outside", "/absolute", "a/../b", "a\\b", "C:/file", "a//b"])
def test_unsafe_reference_refused_without_ambient_root(path, tmp_path):
    manifest = section(tmp_path)
    slot = manifest["suite_slots"]["slots"][0]
    slot["accepted_source"]["path"] = path
    slot["attempts"][0]["accepted_source"]["path"] = path
    with pytest.raises(evidence.GenerationEvidenceError, match="unsafe"):
        evidence.read_suite_slots(manifest)


def test_duplicate_paid_attempt_across_slots_refused(tmp_path):
    manifest = section(tmp_path, count=2)
    duplicate = copy.deepcopy(manifest["suite_slots"]["slots"][0])
    duplicate["slot"] = 2
    manifest["suite_slots"]["slots"][1] = duplicate
    with pytest.raises(evidence.GenerationEvidenceError):
        evidence.read_suite_slots(manifest)


def test_changed_or_missing_reference_is_not_complete(tmp_path):
    manifest = section(tmp_path)
    reference = manifest["suite_slots"]["slots"][0]["accepted_source"]
    path = tmp_path / reference["path"]
    path.write_bytes(b"changed")
    with pytest.raises(evidence.GenerationEvidenceError, match="changed"):
        evidence.read_suite_slots(manifest, tmp_path)
    path.unlink()
    with pytest.raises(evidence.GenerationEvidenceError, match="unavailable"):
        evidence.read_suite_slots(manifest, tmp_path)
    restored = evidence.SuiteSlots(tmp_path, 1, previous=manifest)
    assert restored.snapshot({1: 1})["slots"][0]["state"] == "unavailable"


def test_cyclic_generation_section_refused_as_unavailable(tmp_path):
    manifest = section(tmp_path)
    manifest["suite_slots"]["slots"][0]["reason"] = manifest
    with pytest.raises(evidence.GenerationEvidenceError, match="acyclic"):
        evidence.read_suite_slots(manifest)


def test_old_attempt_retained_after_failed_refill(tmp_path):
    slots = evidence.SuiteSlots(tmp_path, 2)
    slots.record(1, ID, classification="no_code", basis="no_code")
    slots.unavailable(1, "request_in_progress", "pending", active=True)
    slots.finish("transport failure")
    result = slots.snapshot({1: 1})
    first, second = result["slots"]
    assert first["state"] == second["state"] == "unavailable"
    assert first["terminal_attempt"] is None
    assert first["attempts"][0]["disposition_basis"] == "no_code"
    assert second["attempts_used"] == 0
    assert all(slot["terminal"] for slot in result["slots"])


def test_revalidation_retains_paid_attempt_and_refill_history(tmp_path):
    slots = evidence.SuiteSlots(tmp_path, 1)
    slots.record(1, ID, classification="no_code", basis="no_code")
    slots.record(1, ID2, classification="validation_unresolved", basis="unavailable",
                 original_source="source")
    slots.finish()
    restored = evidence.SuiteSlots(tmp_path, 1, previous={"suite_slots": slots.snapshot({1: 2})})
    assert restored.revalidation_id(1, "source") == ID2
    assert restored.revalidation_id(1, "different") is None
    restored.record(1, ID2, classification="pruned", basis="accepted_cleaned",
                    original_source="source", cleaned_source="cleaned")
    result = restored.snapshot({1: 2})["slots"][0]
    assert len(result["attempts"]) == result["attempts_used"] == 2
    assert result["attempts"][0]["disposition_basis"] == "no_code"
    assert result["terminal_attempt"] == evidence.paid_attempt(ID2)


@pytest.mark.parametrize("identity", [None, "previous:1", "unknown:0"])
def test_an_unrecognized_attempt_identity_is_never_inferred(identity, tmp_path):
    slots = evidence.SuiteSlots(tmp_path, 1)
    slots.record(1, identity, classification="all_passing", basis="accepted_cleaned",
                 original_source="original", cleaned_source="cleaned")
    result = slots.snapshot({1: 1})["slots"][0]
    assert result["state"] == "unavailable"
    assert result["terminal_attempt"] is result["accepted_source"] is None
    assert list(tmp_path.rglob("accepted")) == []


def read_manifest(drive):
    return json.loads((drive.suites / starters._GENERATION_MANIFEST).read_text("utf-8"))


def judged_cleanup(monkeypatch, drive, *, unresolved=False):
    def cleanup(backend, source, target, root, scratch, **kwargs):
        drive.cleanup_calls.append(source)
        scratch.mkdir(parents=True, exist_ok=True)
        kwargs["cleanup_out"].update(
            disposition_basis="unavailable" if unresolved else "accepted_cleaned",
            disposition_reason="target validation unresolved" if unresolved else "completed suite passed",
        )
        return ("validation_unresolved", None) if unresolved else ("pruned", source)
    monkeypatch.setattr(starter_salvage, "_clean_suite", cleanup)


@pytest.mark.parametrize("language", ["python", "java"])
def test_actual_initial_accepted_source_and_complete_reuse(language, drive, monkeypatch):
    judged_cleanup(monkeypatch, drive)
    with pytest.raises(_ReachedConsolidation):
        drive.run(language)
    manifest = read_manifest(drive)
    slot = evidence.read_suite_slots(manifest, drive.suites)["slots"][0]
    assert slot["state"] == "accepted" and slot["terminal"]
    assert evidence.read_cleaned_source(slot["accepted_source"], drive.suites) == SOURCES[language]
    assert slot["attempts_used"] == manifest["requests_used"] == manifest["logical_requests"] == 1
    assert slot["terminal_attempt"]["delivery_id"] in manifest["processed_deliveries"]
    before = {p.relative_to(drive.suites): p.read_bytes() for p in drive.suites.rglob("*") if p.is_file()}
    with pytest.raises(_ReachedConsolidation):
        drive.run(language, reuse=True)
    assert {p.relative_to(drive.suites): p.read_bytes() for p in drive.suites.rglob("*") if p.is_file()} == before
    assert len(drive.model_calls) == 1


@pytest.mark.parametrize("language", ["python", "java"])
def test_actual_revalidation_preserves_paid_identity_and_all_counters(language, drive, monkeypatch):
    judged_cleanup(monkeypatch, drive, unresolved=True)
    result = drive.run(language)
    assert result.status == "validation_unresolved"
    previous = read_manifest(drive)
    slot = evidence.read_suite_slots(previous, drive.suites)["slots"][0]
    assert slot["state"] == "unavailable" and slot["terminal"]
    judged_cleanup(monkeypatch, drive)
    with pytest.raises(_ReachedConsolidation):
        drive.run(language, reuse=True)
    current = read_manifest(drive)
    accepted = evidence.read_suite_slots(current, drive.suites)["slots"][0]
    assert accepted["state"] == "accepted"
    assert accepted["terminal_attempt"] == slot["terminal_attempt"]
    assert len(accepted["attempts"]) == 1
    for key in ("requests_used", "logical_requests", "unknown_usage_requests", "input_tokens",
                "output_tokens", "cache_hit_tokens", "generation_attempts_per_suite", "processed_deliveries"):
        assert current[key] == previous[key]
    assert len(drive.model_calls) == 1
    assert len(drive.cleanup_calls) == 2


@pytest.mark.parametrize("language", ["python", "java"])
def test_actual_unjudged_cleanup_not_exported_as_acceptance(language, drive):
    with pytest.raises(_ReachedConsolidation):
        drive.run(language)
    slot = evidence.read_suite_slots(read_manifest(drive), drive.suites)["slots"][0]
    assert slot["classification"] == "pruned"
    assert slot["state"] == "unavailable" and slot["accepted_source"] is None
    assert len(drive.model_calls) == 1


def test_actual_refill_failure_keeps_rejection_and_never_started_slots(drive):
    def request(messages, **kwargs):
        if len(drive.model_calls) == 1:
            return "no code", {"input_tokens": 4, "output_tokens": 2}
        raise ValueError("terminal request failure")
    result = drive.run("python", generate=request, configuration={
        "suite_count": 3, "attempts_per_suite": 2, "refill_failed_suites": True,
    })
    assert result.status == "failed_model_request"
    manifest = read_manifest(drive)
    assert manifest["generation_complete"] is False
    slots = evidence.read_suite_slots(manifest, drive.suites)["slots"]
    assert len(slots) == 3 and all(s["terminal"] for s in slots)
    assert slots[0]["state"] == "rejected"  # No refill was sent after the batch failed.
    assert slots[0]["attempts"][0]["disposition_basis"] == "no_code"
    assert slots[1]["state"] == slots[2]["state"] == "unavailable"
    assert slots[1]["attempts_used"] == slots[2]["attempts_used"] == 0


def test_actual_failed_refill_not_exported_as_previous_rejection(drive):
    def request(messages, **kwargs):
        if len(drive.model_calls) == 1:
            return "no code", {"input_tokens": 4, "output_tokens": 2}
        raise ValueError("terminal request failure")
    result = drive.run("python", generate=request, configuration={
        "attempts_per_suite": 2, "refill_failed_suites": True,
    })
    assert result.status == "failed_model_request"
    manifest = read_manifest(drive)
    slot = evidence.read_suite_slots(manifest, drive.suites)["slots"][0]
    assert slot["state"] == "unavailable" and slot["terminal_attempt"] is None
    assert slot["attempts"][0]["disposition_basis"] == "no_code"
    assert slot["attempts_used"] == 1
    assert manifest["requests_used"] == manifest["logical_requests"] == 2


@pytest.mark.parametrize("language", ["python", "java"])
def test_actual_source_publication_failure_preserves_saved_paid_response(language, drive, monkeypatch):
    judged_cleanup(monkeypatch, drive)
    original = evidence.SuiteSlots.record
    def fail(self, *args, **kwargs):
        raise OSError("source publication failed")
    monkeypatch.setattr(evidence.SuiteSlots, "record", fail)
    with pytest.raises(OSError, match="publication"):
        drive.run(language)
    saved = drive.suites / "suite_001" / starters._GENERATION_DELIVERY
    delivery = json.loads(saved.read_text("utf-8"))
    assert delivery["id"] not in read_manifest(drive)["processed_deliveries"]
    monkeypatch.setattr(evidence.SuiteSlots, "record", original)
    with pytest.raises(_ReachedConsolidation):
        drive.run(language, reuse=True)
    manifest = read_manifest(drive)
    slot = evidence.read_suite_slots(manifest, drive.suites)["slots"][0]
    assert slot["terminal_attempt"]["delivery_id"] == delivery["id"]
    assert slot["attempts_used"] == manifest["requests_used"] == manifest["logical_requests"] == 1
    assert not saved.exists() and len(drive.model_calls) == 1


@pytest.mark.parametrize("language", ["python", "java"])
def test_a_recheck_without_a_paid_binding_stays_unknown(language, drive, monkeypatch):
    """A response rechecked without its paid attempt binds to no attempt."""
    _pool(drive, language, ".java" if language == "java" else ".py", unresolved=True)
    judged_cleanup(monkeypatch, drive)
    with pytest.raises(_ReachedConsolidation):
        drive.run(language, reuse=True)
    slot = evidence.read_suite_slots(read_manifest(drive), drive.suites)["slots"][0]
    assert slot["state"] == "unavailable" and slot["terminal_attempt"] is None
    assert drive.model_calls == []


@pytest.mark.parametrize("ok,timed_out,passed,expected", [
    (True, False, 1, "accepted_cleaned"),
    (False, False, 1, "unavailable"),
    (False, True, 1, "unavailable"),
    (True, False, 0, "judged_rejection"),
    (False, True, 0, "unavailable"),
])
def test_actual_cleanup_basis_uses_completed_typed_observation(
        ok, timed_out, passed, expected, drive, monkeypatch):
    backend = drive.backend
    backend.salvage_syntax = lambda source, root, report: source
    backend.generated_test_target_identity = lambda *args: None
    backend.analyze_generated_test_validity = lambda *args: []
    backend.strip_module_level_skips = lambda source: (source, 0)
    backend.staged_test_path = lambda root, stem, source: drive.suites / "candidate.py"
    backend.failing_test_ids = lambda *args, **kwargs: []
    backend.screen_generated_test_source = lambda *args: []
    backend.validate_test_source = lambda source, root: SourceValidation(
        True, "valid", "", 0, 0, "fixture", evidence.source_sha256(source),
    )
    backend.run_test_files = lambda *args, **kwargs: SuiteRun(
        ok, 0 if ok else 1, timed_out, 1.0, passed, passed,
        0 if ok else 1, 0, "", None, None,
    )
    monkeypatch.setattr(starters.validity, "direct_record", lambda **kwargs: {})
    metadata = {}
    # Restore the real cleanup, which the archive fixture normally isolates.
    classification, cleaned = REAL_CLEAN(
        backend, SOURCES["python"], "target.py", ".", drive.suites,
        timeout=2, log=lambda message: None, reject_unsound=False,
        cleanup_out=metadata,
    )
    assert metadata["disposition_basis"] == expected
    confirmed = ok and not timed_out
    assert classification == ("all_passing" if confirmed and passed else
                              "unsalvageable" if confirmed else "validation_unresolved")
    assert (cleaned is not None) == (confirmed and bool(passed))
    if not confirmed:
        assert metadata["unresolved_stage"] == "final_suite_confirmation"
        assert metadata["confirmation_source"] == SOURCES["python"]


def test_actual_coverage_probe_without_judgement_is_unavailable(drive, monkeypatch):
    """A coverage probe that timed out measured nothing: the suite has no
    complete numbers, and the slot's record says the rejection is not a
    judgement."""
    judged_cleanup(monkeypatch, drive)
    drive.backend.run_test_files = lambda *args, **kwargs: SuiteRun(
        False, 1, True, 2.0, 1, 1, 0, 0, "", 0.0, None,
    )
    result = drive.run("python")
    assert result.status == "cancelled_zero_passing"
    slot = evidence.read_suite_slots(read_manifest(drive), drive.suites)["slots"][0]
    assert slot["classification"] == "coverage_unmeasured"
    assert slot["disposition_basis"] == "unavailable"


def confirmation_backend(drive, monkeypatch, runs):
    backend = drive.backend
    backend.normalize_shadowed_definitions = lambda source: SuiteNormalization(
        source, False, True, None, (), evidence.source_sha256(source),
        evidence.source_sha256(source), "fixture",
    )
    backend.salvage_syntax = lambda source, root, report: source
    backend.generated_test_target_identity = lambda *args: None
    backend.analyze_generated_test_validity = lambda *args: []
    backend.strip_module_level_skips = lambda source: (source, 0)
    backend.failing_test_ids = lambda *args, **kwargs: []
    backend.screen_generated_test_source = lambda *args: []
    backend.prune_tests = lambda source, names: source + "\n// stripped\n"
    backend.prune_tests_report = lambda source, names: PruneOutcome(
        backend.prune_tests(source, names), "")
    backend.validate_test_source = lambda source, root: SourceValidation(
        True, "valid", "", 0, 0, "fixture", evidence.source_sha256(source),
    )
    observations = iter(runs)
    def run(files, root, **kwargs):
        if "measure_coverage_of" in kwargs:
            return SuiteRun(True, 0, False, 1.0, 1, 1, 0, 0, "", 100.0, 100.0)
        return next(observations)
    backend.run_test_files = run
    monkeypatch.setattr(starters.validity, "direct_record", lambda **kwargs: {})
    monkeypatch.setattr(starter_salvage, "_clean_suite", REAL_CLEAN)


def suite_run(*, ok=True, timeout=False, passed=1, failed=0, errors=0, output="", unsound=()):
    return SuiteRun(ok, 0 if ok else 1, timeout, 1.0, 0, passed, failed, errors, output, None, None,
                    unsound=unsound)


@pytest.mark.parametrize("language", ["python", "java"])
@pytest.mark.parametrize("warnings", [False, True])
@pytest.mark.parametrize("observation", [
    suite_run(ok=False), suite_run(timeout=True), suite_run(failed=1), suite_run(errors=1),
    suite_run(ok=False, passed=0), suite_run(timeout=True, passed=0),
])
def test_actual_confirmation_failure_preserves_source_and_paid_recovery(
        language, warnings, observation, drive, monkeypatch):
    runs = ([suite_run(unsound=(("candidate::test_value", "fixture"),)), observation]
            if warnings else [observation])
    confirmation_backend(drive, monkeypatch, runs)
    first = drive.run(language)
    assert first.status == "validation_unresolved"
    before = read_manifest(drive)
    seed = drive.suites / "suite_001"
    suffix = ".py" if language == "python" else ".java"
    meta = json.loads((seed / "original.json").read_text("utf-8"))
    confirmation = meta["cleanup"]["confirmation"]
    assert meta["cleanup"]["unresolved_stage"] == "final_suite_confirmation"
    unconfirmed = (seed / confirmation["source"]).read_bytes().decode("utf-8")
    assert evidence.source_sha256(unconfirmed) == confirmation["source_sha256"]
    assert confirmation["observation"]["passed"] == observation.passed
    assert not (seed / f"cleaned{suffix}").exists()
    paid = evidence.read_suite_slots(before, drive.suites)["slots"][0]["terminal_attempt"]
    assert paid is not None and len(drive.model_calls) == 1
    # A second inconclusive confirmation retains the same original paid attempt.
    confirmation_backend(drive, monkeypatch, [observation])
    assert drive.run(language, reuse=True).status == "validation_unresolved"
    midway = read_manifest(drive)
    assert evidence.read_suite_slots(midway, drive.suites)["slots"][0]["terminal_attempt"] == paid
    confirmation_backend(drive, monkeypatch, [suite_run()])
    with pytest.raises(_ReachedConsolidation):
        drive.run(language, reuse=True)
    after = read_manifest(drive)
    slot = evidence.read_suite_slots(after, drive.suites)["slots"][0]
    assert slot["state"] == "accepted" and slot["terminal_attempt"] == paid
    for key in ("requests_used", "logical_requests", "unknown_usage_requests", "input_tokens",
                "output_tokens", "cache_hit_tokens", "generation_attempts_per_suite", "processed_deliveries",
                "transient_failures", "n_generated"):
        assert after[key] == midway[key] == before[key]
    assert len(drive.model_calls) == 1


@pytest.mark.parametrize("configuration,expected_calls", [
    ({"suite_count": 3, "cache_warmup": True, "refill_failed_suites": True}, 1),
    ({"suite_count": 2, "cache_warmup": False, "refill_failed_suites": True}, 2),
])
def test_actual_confirmation_unresolved_does_not_refill_and_keeps_green_slots(
        configuration, expected_calls, drive, monkeypatch):
    observations = [suite_run(ok=False)] + ([suite_run()] if expected_calls == 2 else [])
    confirmation_backend(drive, monkeypatch, observations)
    if expected_calls == 1:
        result = drive.run("python", configuration=configuration)
        assert result.status == "validation_unresolved"
    else:
        with pytest.raises(_ReachedConsolidation):
            drive.run("python", configuration=configuration)
    manifest = read_manifest(drive)
    slots = evidence.read_suite_slots(manifest, drive.suites)["slots"]
    assert manifest["generation_complete"] is False and manifest["n_validation_unresolved"] == 1
    assert slots[0]["disposition_basis"] == "unavailable"
    assert len(drive.model_calls) == expected_calls
    if expected_calls == 2:
        assert slots[1]["state"] == "accepted"
    else:
        assert all(slot["attempts_used"] == 0 for slot in slots[1:])


@pytest.mark.parametrize("boundary", ["source", "manifest", "delivery_removal"])
def test_actual_interruption_boundaries_keep_one_paid_attempt(boundary, drive, monkeypatch):
    judged_cleanup(monkeypatch, drive)
    write = starters.validity.write_json_atomic
    open_file, unlink = Path.open, Path.unlink
    armed = True
    def manifest_write(path, record):
        if (armed and path.name == starters._GENERATION_MANIFEST
                and record.get("processed_deliveries")):
            raise OSError("manifest publication failed")
        return write(path, record)
    def source_open(path, mode="r", *args, **kwargs):
        if armed and "accepted" in path.parts and mode == "xb":
            raise OSError("source publication failed")
        return open_file(path, mode, *args, **kwargs)
    def remove(path, *args, **kwargs):
        if armed and path.name == starters._GENERATION_DELIVERY:
            raise OSError("delivery removal failed")
        return unlink(path, *args, **kwargs)
    if boundary == "source":
        monkeypatch.setattr(Path, "open", source_open)
    elif boundary == "manifest":
        monkeypatch.setattr(starters.validity, "write_json_atomic", manifest_write)
    else:
        monkeypatch.setattr(Path, "unlink", remove)
    with pytest.raises(OSError, match="failed"):
        drive.run("python")
    saved = drive.suites / "suite_001" / starters._GENERATION_DELIVERY
    identity = json.loads(saved.read_text("utf-8"))["id"]
    before = read_manifest(drive)
    assert (identity in before["processed_deliveries"]) == (boundary == "delivery_removal")
    armed = False
    with pytest.raises(_ReachedConsolidation):
        drive.run("python", reuse=True)
    after = read_manifest(drive)
    slot = evidence.read_suite_slots(after, drive.suites)["slots"][0]
    assert slot["terminal_attempt"]["delivery_id"] == identity
    assert slot["state"] == "accepted" and not saved.exists()
    assert after["n_generated"] == after["requests_used"] == after["logical_requests"] == 1
    assert after["generation_attempts_per_suite"] == {"1": 1}
    assert len(slot["attempts"]) == len(drive.model_calls) == 1


def test_missing_confirmation_fields_are_unresolved_not_success(drive, monkeypatch):
    confirmation_backend(drive, monkeypatch, [SimpleNamespace(passed=1, output="", unsound=())])
    result = drive.run("python")
    assert result.status == "validation_unresolved"
    meta = json.loads((drive.suites / "suite_001/original.json").read_text("utf-8"))
    observation = meta["cleanup"]["confirmation"]["observation"]
    assert observation["ok"] is observation["timed_out"] is observation["errors"] is None


def test_confirmation_exception_keeps_original_error_and_unprocessed_response(drive, monkeypatch):
    error = RuntimeError("retained execution owner")
    confirmation_backend(drive, monkeypatch, [])
    def run(*args, **kwargs):
        raise error
    drive.backend.run_test_files = run
    with pytest.raises(RuntimeError) as raised:
        drive.run("python")
    assert raised.value is error
    assert (drive.suites / "suite_001" / starters._GENERATION_DELIVERY).is_file()
    assert read_manifest(drive)["processed_deliveries"] == []


@pytest.mark.parametrize("language", ["python", "java"])
def test_actual_crlf_paid_response_identity_survives_revalidation(language, drive, monkeypatch):
    source = SOURCES[language].replace("\n", "\r\n")
    reply = f"```{language}\n{source}```"
    judged_cleanup(monkeypatch, drive, unresolved=True)
    result = drive.run(language, generate=lambda *args, **kwargs: (
        reply, {"input_tokens": 10, "output_tokens": 3},
    ))
    assert result.status == "validation_unresolved"
    before = read_manifest(drive)
    paid = evidence.read_suite_slots(before, drive.suites)["slots"][0]["terminal_attempt"]
    judged_cleanup(monkeypatch, drive)
    with pytest.raises(_ReachedConsolidation):
        drive.run(language, reuse=True)
    after = read_manifest(drive)
    slot = evidence.read_suite_slots(after, drive.suites)["slots"][0]
    assert slot["terminal_attempt"] == paid
    assert evidence.read_cleaned_source(slot["accepted_source"], drive.suites) == source
    assert (drive.suites / "suite_001/response.txt").read_bytes().decode("utf-8") == reply
    assert after["requests_used"] == after["logical_requests"] == len(drive.model_calls) == 1


def removed_section(root):
    """A slot whose accepted suite ``cleaned`` a measurement's removal
    changed to ``changed``."""
    slots = evidence.SuiteSlots(root, 1)
    slots.record(1, ID, classification="pruned", basis="accepted_cleaned",
                 original_source="original", cleaned_source="cleaned", suffix=".java")
    slots.measurement_removed(1, "changed", suffix=".java")
    slots.finish()
    return slots, {"suite_slots": slots.snapshot({1: 1})}


def test_a_removal_keeps_the_slot_s_acceptance_and_names_the_changed_suite(tmp_path):
    """After a measurement's removal the slot and its attempt keep their
    accepted reference and construction, which the manifest check compares.
    The changed suite is retained under accepted/ and named in
    measurement_removed_source, and its construction is a pruned node whose
    parent is the accepted attempt's."""
    slots, manifest = removed_section(tmp_path)
    slot = evidence.read_suite_slots(manifest, tmp_path)["slots"][0]
    attempt = slot["attempts"][0]
    assert slot["accepted_source"] == attempt["accepted_source"]
    assert evidence.read_cleaned_source(slot["accepted_source"], tmp_path) == "cleaned"
    changed = slot["measurement_removed_source"]
    assert evidence.read_cleaned_source(changed, tmp_path) == "changed"
    assert changed["attempt_id"] == slot["terminal_attempt"]
    node = slots.construction_for(1, "changed", evidence.construction_scope("starter"))
    assert node["operation"] == "pruned"
    assert node["parents"] == [attempt["construction"]]


def test_a_reuse_compares_a_slot_with_its_changed_suite(tmp_path):
    """A reuse loads the slot's changed suite: it matches, and the accepted
    suite before the removal does not."""
    _slots, manifest = removed_section(tmp_path)
    restored = evidence.SuiteSlots(tmp_path, 1, previous=manifest)
    assert restored.reconcile_reused_sources({1: "changed"}) is False
    assert restored.snapshot({1: 1})["slots"][0]["state"] == "accepted"
    again = evidence.SuiteSlots(tmp_path, 1, previous=manifest)
    assert again.reconcile_reused_sources({1: "cleaned"}) is True


@pytest.mark.parametrize("change", [
    lambda slot: slot.update(state="unavailable", disposition_basis="unavailable",
                             accepted_source=None, terminal_attempt=None),
    lambda slot: slot["measurement_removed_source"].update(source_sha256="0" * 64),
    lambda slot: slot["measurement_removed_source"].update(slot=2),
], ids=["unaccepted", "changed", "other-slot"])
def test_a_changed_suite_reference_is_checked(change, tmp_path):
    _slots, manifest = removed_section(tmp_path)
    change(manifest["suite_slots"]["slots"][0])
    with pytest.raises(evidence.GenerationEvidenceError):
        evidence.read_suite_slots(manifest, tmp_path)


def test_a_withdrawn_slot_names_no_changed_suite(tmp_path):
    slots, _manifest = removed_section(tmp_path)
    slots.withdraw(1, "mutation_unmeasured", "the mutation tool measured nothing")
    assert "measurement_removed_source" not in slots.snapshot({1: 1})["slots"][0]
