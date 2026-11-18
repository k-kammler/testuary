"""The evidence-reuse functions of ``testuary.consolidation`` on a synthetic
inherited archive: which shortfalls the eligibility check names, and what
the reuse writes. The unchanged-suite boundary modules exercise the callers
for real. This module pins the functions' contract outside a run.

Only a payload bound to the delivered bytes may be copied. The reuse reads
the inherited phase's canonical files, which describe the suite that phase
delivered whatever candidate won, and verifies each file's own
``source_sha256`` rather than trusting the record.
"""
from __future__ import annotations

import hashlib
import json
import shutil

import pytest

from testuary import languages
from testuary import evidence as validity
from testuary import generation_evidence
from testuary import project
from testuary.consolidation import (
    _INHERITED_COPIES, _PHASE_END_SIDECARS, InheritedEvidence,
    InheritedPhaseEnd, inherited_phase_end_eligibility,
    reuse_inherited_phase_end,
)

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

SUITE = "from calc import f\n\n\ndef test_f():\n    assert f() == 1\n"
SUITE_SHA = hashlib.sha256(SUITE.encode("utf-8")).hexdigest()
OTHER_SHA = hashlib.sha256(b"a suite that was not delivered").hexdigest()
IDENTITY = {"target_source_sha256": "t" * 64, "language": "python",
            "execution_name": "test_calc.py", "config_sha256": "c" * 64,
            "catalogue": "full"}
KILL_MATRIX = {"schema_version": 3, "source_sha256": SUITE_SHA, "mutants": [
    {"id": "m1", "status": "killed", "operator": "op", "line": 2},
    {"id": "m2", "status": "survived", "operator": "op", "line": 3},
]}
COVERAGE = {"schema_version": 1, "source_sha256": SUITE_SHA,
            "line": 100.0, "branch": 50.0,
            "per_test_tokens": {"test_calc.py::test_f": ["L:calc.py:2"]}}

DESTINATIONS = {
    name: role
    for role, names in _INHERITED_COPIES.items() for name in names
}


@pytest.fixture
def archive(tmp_path):
    """A target, its source directory, and an inherited archive whose delivered
    evidence binds ``SUITE`` under the current validity policy."""
    target = tmp_path / "proj" / "calc.py"
    target.parent.mkdir(parents=True)
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    (tmp_path / "proj" / "pyproject.toml").write_text("", encoding="utf-8")
    root = project.resolve_source_dir(str(target))
    backend = languages.get_backend("python")
    identity = backend.generated_test_target_identity(str(target), root)

    source = tmp_path / "archive" / "starter"
    source.mkdir(parents=True)
    (source / "selected_kill_matrix.json").write_text(
        json.dumps(KILL_MATRIX, indent=2), encoding="utf-8")
    (source / "selected_coverage_matrix.json").write_text(
        json.dumps(COVERAGE, indent=2), encoding="utf-8")
    validity.write_json_atomic(
        source / "validity_delivered.json",
        validity.direct_record(
            identity=identity, analyzed_source=SUITE,
            stage="phase_end.delivered_assertion"),
    )
    record = {
        "delivered_from": "single_suite",
        "selected_source_sha256": SUITE_SHA,
        "execution_name": "test_calc.py",
        "measurement_identity": dict(IDENTITY),
        "shipped_has_matrix": True,
        "delivered_tests": 1,
        "merged_tests": 3,
        "target_revision": "r" * 64,
        "structural_parser_identity": "tree-sitter x",
        "target_validator_identity": "CPython x",
    }
    (source / "consolidation_record.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8")
    return {
        "target": str(target), "root": root, "identity": identity,
        "source": source, "destination": tmp_path / "archive" / "coverage",
        "record": record,
    }


def _eligibility(archive, **overrides):
    kwargs = dict(
        source_phase_dir=archive["source"], delivered_source=SUITE,
        execution_name="test_calc.py", measurement_identity=dict(IDENTITY),
        require_mutation=True, target_file=archive["target"],
        source_dir=archive["root"],
    )
    kwargs.update(overrides)
    return inherited_phase_end_eligibility(**kwargs)


def _rewrite_record(archive, **changes):
    record = dict(archive["record"])
    record.update(changes)
    (archive["source"] / "consolidation_record.json").write_text(
        json.dumps(record), encoding="utf-8")


def _rewrite_coverage(archive, **changes):
    payload = dict(COVERAGE)
    payload.update(changes)
    (archive["source"] / "selected_coverage_matrix.json").write_text(
        json.dumps(payload), encoding="utf-8")


# ---------------------------------------------------------------------------
# Eligibility
# ---------------------------------------------------------------------------

def test_an_inherited_version_2_matrix_is_reused_in_this_version_s_words(archive):
    """The reuse of an inherited phase end reads its stored kill matrix of
    schema version 2 in schema 3's words."""
    (archive["source"] / "selected_kill_matrix.json").write_text(
        json.dumps({**KILL_MATRIX, "schema_version": 2, "mutants": [
            {**KILL_MATRIX["mutants"][0], "status": "ok_killed"},
            {**KILL_MATRIX["mutants"][1], "status": "bad_survived"}]}), encoding="utf-8")
    evidence, reason = _eligibility(archive)
    assert reason == ""
    assert evidence.kill_matrix["schema_version"] == 3
    assert [m["status"] for m in evidence.kill_matrix["mutants"]] == ["killed", "survived"]


def test_an_inherited_matrix_of_a_later_schema_is_a_reason(archive):
    """A stored kill matrix of a schema version this version cannot read is
    a reason not to reuse the phase end, not an error."""
    (archive["source"] / "selected_kill_matrix.json").write_text(
        json.dumps({**KILL_MATRIX, "schema_version": 99}), encoding="utf-8")
    assert _eligibility(archive) == (
        None, "inherited mutation matrix has a schema version this version cannot read")


def test_complete_bound_evidence_of_the_same_bytes_is_eligible(archive):
    evidence, reason = _eligibility(archive)

    assert reason == ""
    assert isinstance(evidence, InheritedEvidence)
    assert evidence.record["selected_source_sha256"] == SUITE_SHA
    assert evidence.coverage == COVERAGE
    assert evidence.coverage_name == "selected_coverage_matrix.json"
    assert evidence.kill_matrix == KILL_MATRIX
    assert evidence.kill_matrix_name == "selected_kill_matrix.json"


def test_a_missing_record_is_a_reason(archive):
    (archive["source"] / "consolidation_record.json").unlink()

    assert _eligibility(archive) == (
        None, "inherited consolidation record missing or unreadable")


def test_an_unreadable_record_is_a_reason(archive):
    (archive["source"] / "consolidation_record.json").write_text(
        "{not json", encoding="utf-8")

    assert _eligibility(archive)[1] == (
        "inherited consolidation record missing or unreadable")


def test_a_record_that_is_not_an_object_is_a_reason(archive):
    (archive["source"] / "consolidation_record.json").write_text(
        "[]", encoding="utf-8")

    assert _eligibility(archive) == (
        None, "inherited consolidation record is not an object")


def test_different_bytes_are_a_reason(archive):
    assert _eligibility(archive, delivered_source=SUITE + "\n") == (
        None, "inherited suite bytes differ")


def test_a_different_execution_name_is_a_reason(archive):
    assert _eligibility(archive, execution_name="test_other.py") == (
        None, "inherited execution name differs")


@pytest.mark.parametrize("change", [
    {"catalogue": "selective"}, {"config_sha256": "d" * 64},
    {"target_source_sha256": "u" * 64},
])
def test_a_different_measurement_identity_is_a_reason(archive, change):
    """A selective catalogue for a full requirement, another attribution
    configuration, another target revision: each is an identity difference."""
    identity = dict(IDENTITY)
    identity.update(change)

    assert _eligibility(archive, measurement_identity=identity) == (
        None, "inherited measurement identity differs")


def test_a_missing_validity_record_is_a_reason(archive):
    (archive["source"] / "validity_delivered.json").unlink()

    assert _eligibility(archive) == (
        None,
        "inherited evidence incomplete: missing ['validity_delivered.json']")


def test_a_missing_kill_matrix_is_a_reason_when_mutation_is_required(archive):
    (archive["source"] / "selected_kill_matrix.json").unlink()

    assert _eligibility(archive) == (
        None, "inherited mutation matrix missing or unreadable")


def test_a_missing_coverage_matrix_is_a_reason(archive):
    (archive["source"] / "selected_coverage_matrix.json").unlink()

    assert _eligibility(archive) == (
        None,
        "no inherited per-test coverage evidence is bound to the suite bytes")


# A reduced_* coverage file carries the reduced candidate's hash, so when
# a starter suite, an agent suite or a merged suite is delivered instead, it is
# bound to other bytes and the reuse declines it.
def test_coverage_evidence_bound_to_another_suite_is_a_reason(archive):
    _rewrite_coverage(archive, source_sha256=OTHER_SHA, line=83.0)

    evidence, reason = _eligibility(archive)

    assert evidence is None
    assert reason == (
        "no inherited per-test coverage evidence is bound to the suite bytes")


def test_coverage_evidence_without_a_binding_is_a_reason(archive):
    """``reduced_source_sha256`` can be empty, which persists as null. An
    unstamped payload certifies nothing."""
    _rewrite_coverage(archive, source_sha256=None)

    assert _eligibility(archive)[1] == (
        "no inherited per-test coverage evidence is bound to the suite bytes")


def test_coverage_evidence_without_per_test_tokens_is_a_reason(archive):
    """The next boundary merges and the score derivation counts the per-test
    tokens, so a payload without them is incomplete."""
    _rewrite_coverage(archive, per_test_tokens=None)

    assert _eligibility(archive)[1] == (
        "no inherited per-test coverage evidence is bound to the suite bytes")


def test_unreadable_coverage_evidence_is_a_reason(archive):
    (archive["source"] / "selected_coverage_matrix.json").write_text(
        "{not json", encoding="utf-8")

    assert _eligibility(archive)[1] == (
        "no inherited per-test coverage evidence is bound to the suite bytes")


def test_the_kill_matrix_is_not_required_without_mutation(archive):
    (archive["source"] / "selected_kill_matrix.json").unlink()
    _rewrite_record(archive, shipped_has_matrix=False)

    evidence, reason = _eligibility(archive, require_mutation=False)

    assert reason == ""
    assert evidence.kill_matrix is None and evidence.kill_matrix_name is None
    assert evidence.coverage == COVERAGE


def test_a_suite_delivered_without_a_matrix_is_a_reason_when_mutation_is_required(
    archive,
):
    _rewrite_record(archive, shipped_has_matrix=False)

    assert _eligibility(archive) == (
        None, "inherited suite delivered without a mutation matrix")


def test_a_kill_matrix_bound_to_another_suite_is_a_reason(archive):
    """The canonical matrix states the source it describes. One naming other
    bytes is not evidence of these, whatever the record says."""
    matrix = dict(KILL_MATRIX, source_sha256=OTHER_SHA)
    (archive["source"] / "selected_kill_matrix.json").write_text(
        json.dumps(matrix), encoding="utf-8")

    assert _eligibility(archive) == (
        None, "inherited mutation matrix is not bound to the suite bytes")


def test_an_unstamped_kill_matrix_is_a_reason(archive):
    matrix = {key: value for key, value in KILL_MATRIX.items()
              if key != "source_sha256"}
    (archive["source"] / "selected_kill_matrix.json").write_text(
        json.dumps(matrix), encoding="utf-8")

    assert _eligibility(archive) == (
        None, "inherited mutation matrix is not bound to the suite bytes")


def test_an_unreadable_kill_matrix_is_a_reason(archive):
    (archive["source"] / "selected_kill_matrix.json").write_text(
        "[]", encoding="utf-8")

    assert _eligibility(archive) == (
        None, "inherited mutation matrix is not an object")


def test_an_unbound_validity_record_is_a_reason(archive):
    path = archive["source"] / "validity_delivered.json"
    record = validity.load_record(path)
    record["final_clean_source_sha256"] = "0" * 64
    validity.write_json_atomic(path, record)

    assert _eligibility(archive) == (
        None, "inherited validity record does not bind the suite bytes")


def test_a_stale_validity_policy_is_a_reason(archive):
    path = archive["source"] / "validity_delivered.json"
    record = validity.load_record(path)
    record["policy_version"] = "0"
    validity.write_json_atomic(path, record)

    assert _eligibility(archive) == (
        None, "inherited validity record does not bind the suite bytes")


def test_an_unreadable_validity_record_is_a_reason(archive):
    (archive["source"] / "validity_delivered.json").write_text(
        "[]", encoding="utf-8")

    evidence, reason = _eligibility(archive)

    assert evidence is None
    assert reason.startswith("inherited validity record unreadable: ")


def test_an_older_record_of_an_acceptance_as_delivered_is_no_reason(archive):
    """The inherited phase end chose only among suites that passed their
    normal run, so its evidence stands without an acceptance as delivered,
    whatever an older record says about that check."""
    for recorded in (True, False):
        _rewrite_record(archive, portability={
            "reject_non_portable_tests": recorded,
            "report_reverse_order_failures": False, "checks": {}})
        evidence, reason = _eligibility(archive)
        assert reason == "" and evidence is not None, recorded


# ---------------------------------------------------------------------------
# The reuse
# ---------------------------------------------------------------------------

def _reuse(archive, *, require_mutation=True, **overrides):
    evidence, reason = _eligibility(archive, require_mutation=require_mutation)
    assert evidence is not None, reason
    kwargs = dict(
        source_phase_dir=archive["source"], source_phase="starter",
        destination_dir=archive["destination"], evidence=evidence,
        target_file=archive["target"], source_dir=archive["root"],
        delivered_source=SUITE, execution_name="test_calc.py",
        measurement_identity=dict(IDENTITY),
        generation_scope=generation_evidence.construction_scope("coverage"),
    )
    kwargs.update(overrides)
    return reuse_inherited_phase_end(**kwargs)


def test_the_reuse_writes_the_inherited_payloads_under_every_destination_name(
    archive,
):
    inherited = _reuse(archive)

    destination = archive["destination"]
    assert sorted(p.name for p in destination.iterdir()) == sorted(
        [*DESTINATIONS, "consolidation_record.json",
         "validity_derivations.json", "validity_merged.json",
         "validity_delivered.json"])
    for name, role in DESTINATIONS.items():
        source_name = ("selected_kill_matrix.json" if role == "kill_matrix"
                       else "selected_coverage_matrix.json")
        assert (destination / name).read_bytes() == (
            archive["source"] / source_name).read_bytes(), name
    assert isinstance(inherited, InheritedPhaseEnd)
    assert inherited.kill_matrix == KILL_MATRIX
    assert inherited.coverage == COVERAGE
    assert inherited.delivered_tests == 1


def test_the_record_carries_the_provenance_and_no_measurement(archive):
    inherited = _reuse(archive)

    record = json.loads(
        (archive["destination"] / "consolidation_record.json").read_text("utf-8"))
    assert inherited.delivered_tests == record["delivered_tests"]
    assert record["delivered_from"] == "inherited"
    assert record["winner_source_label"] == "starter"
    assert record["inherited_from"] == "starter"
    assert record["measured"] is False
    assert record["measurement_calls"] == 0
    # The inherited kill matrix is the delivered suite's mutation measurement.
    assert record["mutation_measured"] is True
    assert record["shipped_has_matrix"] is True
    assert record["selected_source_sha256"] == SUITE_SHA
    assert record["merged_source_sha256"] == SUITE_SHA
    assert record["reduced_source_sha256"] == SUITE_SHA
    assert record["execution_name"] == "test_calc.py"
    assert record["measurement_identity"] == IDENTITY
    assert record["target_revision"] == "r" * 64
    assert record["structural_parser_identity"] == "tree-sitter x"
    assert record["target_validator_identity"] == "CPython x"
    # Merging identical suites gives the delivered suite, not the inherited
    # phase's own pool.
    assert record["merged_tests"] == 1
    assert record["delivered_tests"] == 1
    assert record["reduction_valid"] is None
    assert record["reduction_solver"] is None
    assert record["selection_audit"] is None
    assert "portability" not in record
    assert record["inherited_artifacts"] == {
        name: ("selected_kill_matrix.json" if role == "kill_matrix"
               else "selected_coverage_matrix.json")
        for name, role in DESTINATIONS.items()
    }
    assert record["artifacts"] == sorted(
        [*DESTINATIONS, "validity_derivations.json", "validity_merged.json",
         "validity_delivered.json"])
    assert record["validity_policy_version"] == archive["identity"].policy_version
    assert str(archive["source"]) not in json.dumps(record)


def test_the_validity_records_bind_the_bytes_for_the_destination(archive):
    _reuse(archive)

    destination = archive["destination"]
    delivered = validity.load_record(destination / "validity_delivered.json")
    merged = validity.load_record(destination / "validity_merged.json")
    assert delivered["stage"] == "phase_end.delivered_assertion"
    assert merged["stage"] == "phase_end.merged"
    for record in (delivered, merged):
        assert record["inherited_from"] == "starter"
        assert validity.record_binds_source(record, SUITE, archive["identity"])
    derivations = validity.load_record(destination / "validity_derivations.json")
    assert derivations["records"] == []
    assert derivations["inherited_from"] == "starter"
    assert validity.verify_delivered_archive(
        archive_dir=destination, source=SUITE, identity=archive["identity"],
    ) == (True, "")


def test_stale_phase_end_files_in_the_destination_are_removed(archive):
    """Files a normal phase end owns are cleared before the copy, as normal
    persistence does. Other files stay."""
    destination = archive["destination"]
    destination.mkdir(parents=True)
    # Sidecars a reusing phase end owns but never writes, so any leftover is
    # stale. The reuse writes the canonical and candidate matrices, so the
    # byte-identity test covers those.
    stale_names = ("consolidation_failure.json",
                   "merged_kill_matrix_selective.json",
                   "starter_pool_kill_matrix.json")
    for name in stale_names:
        assert name in _PHASE_END_SIDECARS
        assert name not in DESTINATIONS
        (destination / name).write_text("stale", encoding="utf-8")
    (destination / "candidate_agent.py").write_text(SUITE, encoding="utf-8")

    _reuse(archive)

    names = {p.name for p in destination.iterdir()}
    for name in stale_names:
        assert name not in names
    assert (destination / "candidate_agent.py").read_text("utf-8") == SUITE


def test_without_an_inherited_kill_matrix_no_kill_matrix_name_is_written(
    archive,
):
    (archive["source"] / "selected_kill_matrix.json").unlink()
    _rewrite_record(archive, shipped_has_matrix=False)

    inherited = _reuse(archive, require_mutation=False)

    names = {p.name for p in archive["destination"].iterdir()}
    assert not any(name.endswith("_kill_matrix.json") for name in names)
    assert inherited.kill_matrix is None
    record = json.loads(
        (archive["destination"] / "consolidation_record.json").read_text("utf-8"))
    assert record["shipped_has_matrix"] is False
    assert record["mutation_measured"] is False
    assert record["inherited_artifacts"] == {
        name: "selected_coverage_matrix.json"
        for name, role in DESTINATIONS.items() if role == "coverage"
    }


def test_the_inherited_record_has_the_phase_end_record_layout(archive):
    """The inherited record holds a measured phase end's keys in their order,
    then the four keys only a reuse has, and records nothing as having run."""
    from testuary import consolidation

    _reuse(archive)
    record = json.loads(
        (archive["destination"] / "consolidation_record.json").read_text("utf-8"))
    assert {key: record[key] for key in (
        "repaired_tests", "evidence_repairs", "unreduced_fallback",
        "fallback_reason", "fallback_candidates", "fallback_merged_stats",
        "derive_measure_delta", "derive_measure_differences",
        "reduction_solver_metadata", "consolidation_mutation_method",
        "collision_resolution", "input_normalizations",
    )} == {
        "repaired_tests": [], "evidence_repairs": [], "unreduced_fallback": False,
        "fallback_reason": None, "fallback_candidates": [],
        "fallback_merged_stats": None, "derive_measure_delta": None,
        "derive_measure_differences": None, "reduction_solver_metadata": None,
        "consolidation_mutation_method": None, "collision_resolution": None,
        "input_normalizations": [],
    }
    assert record["detail"].startswith("inherited from the starter phase: ")
    measured = consolidation._consolidation_record(
        "t.py", consolidation.PhaseConsolidation(),
        consolidation.PhaseEndDelivery(winner_label="merged", delivered_km=None),
        None, "", [], None)
    assert list(record) == list(measured) + [
        "inherited_from", "inherited_artifacts", "measured", "measurement_calls"]


def test_an_interrupted_copy_leaves_no_record(archive, monkeypatch):
    """The record is written last, so an interruption leaves a destination
    without one and the source archive unchanged."""
    before = {
        p.name: p.read_bytes() for p in archive["source"].iterdir()
    }
    real_copy = shutil.copyfile
    calls = {"n": 0}

    def failing_copy(src, dst, *args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:
            raise OSError("simulated interruption")
        return real_copy(src, dst, *args, **kwargs)

    monkeypatch.setattr(shutil, "copyfile", failing_copy)

    with pytest.raises(OSError, match="simulated interruption"):
        _reuse(archive)

    names = {p.name for p in archive["destination"].iterdir()}
    assert "consolidation_record.json" not in names
    assert not any(name.startswith("validity_") for name in names)
    assert len(names) == 2
    assert {p.name: p.read_bytes() for p in archive["source"].iterdir()} == before
