"""Construction provenance through real persistence and controlled callers."""
import copy
import json
import subprocess
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest
import httpx
from langadapter import registration

from testuary import generation_evidence as evidence
from testuary import config
from testuary import models
from testuary import project
from testuary import consolidation as consolidate
from testuary import starters
from testuary import starter_salvage
from testuary import measurement as metrics
from testuary import target_phases as orchestrator
from testuary.settings import RunSettings
from test_starter_archives import drive as _archive_drive, SOURCES
from test_resolved_starter_requests import configured


RUN = "0123456789abcdef0123456789abcdef"
OTHER = "abcdef0123456789abcdef0123456789"
BASE = "def test_base():\n    assert True\n"
EDIT = "def test_edit():\n    assert True\n"
MERGED = BASE + "\n" + EDIT


def origin_file(path, phase="starter", language="python", *, profile="original"):
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = json.dumps({"name": profile, "phases": {phase: {"append": "captured"}}})
    (path.parent / "profile.json").write_bytes(contents.encode("utf-8"))
    record = {"format": 1, "condition": {
        "format": 1, "phase": phase, "language": language,
        "model": {"provider": "fixture", "model": profile}, "request": {"temperature": 0.5},
        "prompt_composition": "a" * 64, "service_identity_known": True, "request_options_known": True,
    }, "prompt_profile": {"copy": "profile.json", "sha256": evidence.source_sha256(contents)}, "hints": None}
    path.write_text(json.dumps(record), encoding="utf-8")
    return path


def seed(source, origin, *, run=RUN, slot=1, run_idx=1):
    return evidence.construction(source, phase="starter", invocation=run, run_idx=run_idx,
                                 operation="accepted_suite", origin=origin,
                                 suite_attempt={"slot": slot, "attempt_id": evidence.paid_attempt(f"{run}:{slot}")})


def selected(source, node):
    return {"selected_source_sha256": evidence.source_sha256(source), "generation_ancestry": node}


@pytest.fixture(autouse=True)
def no_native(monkeypatch):
    def refuse(*args, **kwargs):
        pytest.fail("native process or provider was not part of this gate")
    monkeypatch.setattr(subprocess, "Popen", refuse)


@pytest.fixture
def backend(monkeypatch):
    backend = SimpleNamespace(
        normalize_shadowed_definitions=lambda source: SimpleNamespace(
            parse_ok=True, changed=False, source=source, removed_bindings=(), rule_version="fixture"),
        generated_test_target_identity=lambda *args: SimpleNamespace(policy_version="fixture"),
        analyze_generated_test_validity=lambda *args: [], screen_generated_test_source=lambda *args: [],
        target_test_path=lambda directory, target, **kwargs: Path(directory) / "test_subject.py",
        collect_eviction_data=lambda *args, **kwargs: None,
        resolve_test_collisions=lambda sources: SimpleNamespace(ejected=[]),
        test_metrics=lambda source: {"test_count": 1},
    )
    monkeypatch.setattr(consolidate.languages, "get_backend", lambda language: backend)
    monkeypatch.setattr(consolidate.validity, "direct_record", lambda **kwargs: {})
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: False)
    monkeypatch.setattr(orchestrator.mutation_agent, "mutation_measurement_identity", lambda **kwargs: {})
    return backend


def fake_phase(*args, **kwargs):
    sources = args[1]
    return consolidate.PhaseConsolidation(
        merged_source="\n".join(sources), merged_source_sha256=evidence.source_sha256("\n".join(sources)),
        reduced_source=sources[-1], reduced_source_sha256=evidence.source_sha256(sources[-1]),
        line=100.0, branch=100.0, pool_tests=len(sources), selected_tests=1,
    )


def candidate(label, source):
    return consolidate.ScoredSuite(label=label, source=source, line=100.0, branch=100.0,
                                    green=True, status="passing", test_count=1,
                                    coverage_evidence={"line": 100.0, "branch": 100.0})


@pytest.mark.parametrize("winner,parents,operation", [
    ("suite_001", 1, "selected"), ("merged", 2, "merge"), ("merged_unreduced", 2, "merge"),
    ("reduced", 2, "reduced"), ("absent", 0, "unavailable"),
])
def test_selected_inputs_exclude_losing_candidates(tmp_path, winner, parents, operation):
    origin = evidence.capture_origin(origin_file(tmp_path / "origin.json"))
    inputs = {"suite_001": seed(BASE, origin), "suite_002": seed(EDIT, origin, slot=2)}
    source = BASE if winner == "suite_001" else MERGED
    node = evidence.select_construction(source, scope=evidence.construction_scope("starter", 2, OTHER),
                                         winner=winner, inputs=inputs)
    assert len(node["parents"]) == parents and node["operation"] == operation
    assert node["basis"] == "construction_inputs"
    assert node["status"] == ("unavailable" if not parents else "complete")


@pytest.mark.parametrize("change", [
    lambda n: n.update(version=True), lambda n: n.update(version=2),
    lambda n: n.update(basis="authors"), lambda n: n.update(status="unavailable"),
    lambda n: n.update(selected_source_sha256="bad"),
    lambda n: n["suite_attempt"]["attempt_id"].update(logical_request_idx=True),
    lambda n: n["origin"]["inputs"]["prompt_profile"].update(base="../outside"),
    lambda n: n["origin"]["inputs"]["prompt_profile"].update(contents="changed"),
    lambda n: n["origin"]["condition"]["model"].update(authorization="secret"),
    lambda n: n["origin"]["condition"].update(phase="mutation"),
    lambda n: n.update(suite_attempt=None),
    lambda n: n.update(invocation=int("1" * 32)),
    lambda n: n.pop("selected_source_sha256"),
])
def test_malformed_contradictory_or_unsafe_ancestry_refused(tmp_path, change):
    node = seed(BASE, evidence.capture_origin(origin_file(tmp_path / "origin.json")))
    change(node)
    with pytest.raises(evidence.GenerationEvidenceError):
        evidence.read_ancestry(selected(BASE, node), BASE)


def test_cyclic_or_changed_parent_is_unavailable_not_complete(tmp_path):
    node = seed(BASE, evidence.capture_origin(origin_file(tmp_path / "origin.json")))
    node["parents"].append(node)
    with pytest.raises(evidence.GenerationEvidenceError):
        evidence.read_ancestry(selected(BASE, node))
    unknown = evidence.predecessor({"selected_source_sha256": "0" * 64}, phase="starter", source=BASE)
    edited = evidence.construction(EDIT, phase="coverage", invocation=OTHER, run_idx=2,
                                  operation="edited", parents=[unknown],
                                  origin=evidence.capture_origin(origin_file(tmp_path / "editing.json", "coverage")))
    assert edited["status"] == "partial" and edited["parents"][0]["status"] == "unavailable"


@pytest.mark.parametrize("operation", ["selected", "inherited"])
@pytest.mark.parametrize("known_origin", [True, False])
def test_source_preserving_reader_refuses_changed_known_parent(tmp_path, operation, known_origin):
    origin = evidence.capture_origin(origin_file(tmp_path / "origin.json")) if known_origin else None
    parent = seed(BASE, origin)
    node = evidence.construction(BASE, phase="coverage", invocation=OTHER, run_idx=2,
                                 operation=operation, parents=[parent])
    # Every node is valid on its own, only the preserved-source edge is false.
    node["parents"][0]["selected_source_sha256"] = evidence.source_sha256(EDIT)
    before = copy.deepcopy(node)
    with pytest.raises(evidence.GenerationEvidenceError):
        evidence.read_ancestry(selected(BASE, node), BASE)
    assert node == before
    with pytest.raises(evidence.GenerationEvidenceError):
        evidence.construction(BASE, phase="coverage", invocation=OTHER, run_idx=2,
                              operation=operation, parents=[seed(EDIT, origin)])


@pytest.mark.parametrize("operation", ["selected", "inherited"])
@pytest.mark.parametrize("count", [0, 2])
def test_source_preserving_reader_refuses_false_complete_parent_population(tmp_path, operation, count):
    parent = seed(BASE, evidence.capture_origin(origin_file(tmp_path / "origin.json")))
    node = evidence.construction(BASE, phase="coverage", invocation=OTHER, run_idx=2,
                                 operation=operation, parents=[parent])
    node["parents"] = [copy.deepcopy(parent) for _ in range(count)]
    with pytest.raises(evidence.GenerationEvidenceError):
        evidence.read_ancestry(selected(BASE, node), BASE)


@pytest.mark.parametrize("operation", ["selected", "inherited"])
@pytest.mark.parametrize("parent_source", [None, BASE])
def test_source_preserving_unknown_parent_stays_partial(operation, parent_source):
    parent = evidence.construction(parent_source, phase="starter", operation="unavailable")
    node = evidence.construction(BASE, phase="coverage", invocation=OTHER, run_idx=2,
                                 operation=operation, parents=[parent])
    assert evidence.read_ancestry(selected(BASE, node), BASE)["status"] == "partial"
    if parent_source is None:
        node["status"] = "complete"
        with pytest.raises(evidence.GenerationEvidenceError):
            evidence.read_ancestry(selected(BASE, node), BASE)


@pytest.mark.parametrize("operation", ["pruned", "merge", "reduced", "edited"])
def test_changed_source_transformations_keep_known_parent(tmp_path, operation):
    parent = seed(BASE, evidence.capture_origin(origin_file(tmp_path / "origin.json")))
    origin = (evidence.capture_origin(origin_file(tmp_path / "editing.json", "coverage"))
              if operation == "edited" else None)
    node = evidence.construction(EDIT, phase="coverage", invocation=OTHER, run_idx=2,
                                 operation=operation, parents=[parent], origin=origin)
    checked = evidence.read_ancestry(selected(EDIT, node), EDIT)
    assert checked["status"] == "complete" and checked["parents"] == [parent]


@pytest.mark.parametrize("change", ["unsafe", "changed", "missing", "unknown_version", "secret"])
def test_unavailable_captured_inputs_do_not_use_ambient_config(tmp_path, change):
    path = origin_file(tmp_path / "origin.json")
    data = json.loads(path.read_text("utf-8"))
    if change == "unsafe":
        data["prompt_profile"]["copy"] = "../profile.json"
    elif change == "unknown_version":
        data["format"] = 9
    elif change == "secret":
        text = json.dumps({"authorization": "secret"})
        (path.parent / "profile.json").write_text(text, encoding="utf-8")
        data["prompt_profile"]["sha256"] = evidence.source_sha256(text)
    elif change == "missing":
        (path.parent / "profile.json").unlink()
    else:
        (path.parent / "profile.json").write_text("changed", encoding="utf-8")
    path.write_text(json.dumps(data), encoding="utf-8")
    assert evidence.capture_origin(path) is None


def test_paid_revalidation_retains_original_origin_and_repeat(tmp_path):
    origin = evidence.capture_origin(origin_file(tmp_path / "origin.json"))
    slots = evidence.SuiteSlots(tmp_path / "suites", 1)
    slots.record(1, f"{RUN}:0", classification="validation_unresolved", basis="unavailable",
                 original_source=BASE, origin=origin, run_idx=1)
    slots.finish()
    restored = evidence.SuiteSlots(slots.root, 1, previous={"suite_slots": slots.snapshot({1: 1})})
    different = evidence.capture_origin(origin_file(tmp_path / "different.json", profile="later"))
    restored.record(1, f"{RUN}:0", classification="pruned", basis="accepted_cleaned",
                    original_source=BASE, cleaned_source=EDIT, origin=different, run_idx=2)
    restored.finish()
    attempt = evidence.read_suite_slots({"suite_slots": restored.snapshot({1: 1})}, slots.root)["slots"][0]["attempts"][0]
    assert attempt["construction"]["origin"] == origin
    assert attempt["construction"]["run_idx"] == 1
    assert attempt["construction"]["selected_source_sha256"] == evidence.source_sha256(EDIT)


def configure_archive_backend(drive, monkeypatch):
    backend = drive.backend
    backend.target_test_path = lambda directory, target, **kwargs: Path(directory) / (
        "CalculatorTest.java" if backend.file_extensions[0] == ".java" else "test_subject.py")
    backend.scratch_test_path = lambda directory, source, target, **kwargs: Path(directory) / (
        "CalculatorTest.java" if backend.file_extensions[0] == ".java" else "test_seed.py")
    # A run's records name the mutation engine its language's adapter offers.
    backend._adapter_class = lambda: registration(
        "java" if backend.file_extensions[0] == ".java" else "python").factory
    # No per-test evidence: each starter suite is scored on its coverage run.
    backend.collect_eviction_data = lambda *args, **kwargs: None
    backend.test_metrics = lambda source: {"test_count": 1}
    backend.resolve_test_collisions = lambda sources: SimpleNamespace(ejected=[])
    backend.normalize_shadowed_definitions = lambda source: SimpleNamespace(
        parse_ok=True, changed=False, source=source, removed_bindings=(), rule_version="fixture")
    backend.generated_test_target_identity = lambda *args: SimpleNamespace(policy_version="fixture")
    backend.analyze_generated_test_validity = lambda *args: []
    backend.screen_generated_test_source = lambda *args: []
    backend.run_test_files = lambda *args, **kwargs: SimpleNamespace(
        ok=True, timed_out=False, passed=1, failed=0, errors=0, line_coverage=100.0,
        branch_coverage=100.0, elapsed_s=1.0, missing_lines=(), missing_branches=())
    real_clean = starter_salvage._clean_suite
    def clean(*args, **kwargs):
        classification, source = real_clean(*args, **kwargs)
        kwargs["cleanup_out"].update(disposition_basis="accepted_cleaned")
        return classification, source
    monkeypatch.setattr(starter_salvage, "_clean_suite", clean)
    # The mutation settings a run always has, at the packaged default.
    monkeypatch.setattr(starters, "_starter_mutation_config", lambda **kwargs: {"cover_attribution": "full"})
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: False)
    monkeypatch.setattr(orchestrator.mutation_agent, "mutation_measurement_identity", lambda **kwargs: {})
    monkeypatch.setattr(consolidate, "consolidate_phase_end", fake_phase)
    monkeypatch.setattr(consolidate, "benchmark_tied_candidates", lambda *args, **kwargs: lambda c: 1.0)
    monkeypatch.setattr(consolidate.validity, "direct_record", lambda **kwargs: {})


@pytest.mark.parametrize("winner", ["suite_001", "merged", "reduced"])
@pytest.mark.parametrize("language", ["python", "java"])
def test_actual_initial_selection_publishes_bound_construction(tmp_path, monkeypatch, winner, language):
    drive = _archive_drive.__wrapped__(tmp_path, monkeypatch)
    configure_archive_backend(drive, monkeypatch)
    monkeypatch.setattr(consolidate, "phase_end_candidates", lambda phase: [candidate(winner, SOURCES[language])])
    monkeypatch.setattr(consolidate, "measure_candidate_standalone", lambda *args, **kwargs: candidate(winner, SOURCES[language]))
    # The starter suites tie with the injected winner, which is the fastest.
    monkeypatch.setattr(consolidate, "benchmark_tied_candidates",
                        lambda *args, **kwargs: lambda c: 0.0 if c.label == winner else 1.0)
    monkeypatch.setattr(consolidate.validity, "direct_record", lambda **kwargs: {})
    # Run the real caller with an output path, which the archive fixture omits.
    run_initial = starters.run_starter_phase
    def run(**kwargs):
        return run_initial(**kwargs, matrices_dir=str(tmp_path / "archive"), run_idx=2)
    monkeypatch.setattr(starters, "run_starter_phase", run)
    result = drive.run(language)
    record = json.loads((tmp_path / "archive/consolidation_record.json").read_text("utf-8"))
    node = evidence.read_ancestry(record, Path(result.baseline_file).read_text("utf-8"))
    assert node["run_idx"] == 2 and node["phase"] == "starter"
    assert node["operation"] == ("selected" if winner == "suite_001" else "merge" if winner == "merged" else "reduced")
    assert node["parents"][0]["suite_attempt"]["attempt_id"]["run"] == node["invocation"]
    assert Path(result.baseline_file).suffix == (".java" if language == "java" else ".py")
    assert node["status"] == "partial"  # injected generator has no provider origin
    assert len(drive.model_calls) == 1


@pytest.mark.parametrize("recover", ["fresh", "validation", "delivery", "delivery_without_inputs"])
def test_configured_initial_origin_paid_recovery_and_retained_cleaned_text(tmp_path, monkeypatch, recover):
    api_config = models.get_api_config
    drive = _archive_drive.__wrapped__(tmp_path, monkeypatch)
    configure_archive_backend(drive, monkeypatch)
    backend = drive.backend
    backend.file_extensions = (".py",)
    backend.test_file_extensions = lambda: (".py",)
    backend.test_language = lambda source_dir=None: "python"
    backend.prompt_fragments = lambda: {"lang_code_fence": "python"}
    backend.staged_test_path = lambda directory, stem, source: Path(directory) / f"{stem}.py"
    monkeypatch.setattr(models, "get_api_config", api_config)
    monkeypatch.setattr(consolidate, "phase_end_candidates", lambda phase: [candidate("merged", EDIT)])
    monkeypatch.setattr(consolidate, "measure_candidate_standalone", lambda *a, **k: candidate("suite_001", EDIT))
    states, sends = [], []
    def clean(*args, **kwargs):
        args[4].mkdir(parents=True, exist_ok=True)
        if states:
            states.pop()
            kwargs["cleanup_out"].update(disposition_basis="unavailable")
            return "validation_unresolved", None
        kwargs["cleanup_out"].update(disposition_basis="accepted_cleaned")
        return "pruned", EDIT
    monkeypatch.setattr(starter_salvage, "_clean_suite", clean)
    if recover == "validation":
        states.append("validator unavailable")
    def handle(request):
        sends.append(json.loads(request.content))
        return httpx.Response(200, text=json.dumps({
            "message": {"content": f"```python\n{BASE}```"}, "done": True,
            "prompt_eval_count": 10, "eval_count": 3,
        }) + "\n")
    @contextmanager
    def stream(method, url, **kwargs):
        with httpx.Client(transport=httpx.MockTransport(handle)) as client:
            with client.stream(method, url, **kwargs) as response:
                yield response
    monkeypatch.setattr(httpx, "stream", stream)
    root = tmp_path / "subject"
    target = root / "subject.py"
    target.write_text("target fixture", encoding="utf-8")
    settings = configured(protocol="ollama-native")
    settings = RunSettings(base_dir=tmp_path, models=settings.models, sections={
        **settings.sections, "main": {"seed": 37, "hints_file": ""},
    })
    def run(settings, run_idx):
        with config.use_run_settings(settings):
            return starters.run_starter_phase(
                language="python", target_file_path=str(target),
                source_dir=str(root), test_output_dir=str(tmp_path / "output"),
                suites_dir=str(drive.suites), matrices_dir=str(tmp_path / "archive"),
                run_idx=run_idx, log=lambda *args: None,
            )
    append = consolidate.validity.append_jsonl
    if recover in {"delivery", "delivery_without_inputs"}:
        def unavailable(path, record):
            if path.name == "request_records.jsonl":
                raise OSError("paid request journal unavailable")
            return append(path, record)
        monkeypatch.setattr(consolidate.validity, "append_jsonl", unavailable)
        from testuary import model_requests as _model_requests
        with pytest.raises(_model_requests.RecordNotWritten):
            run(settings, 1)
        monkeypatch.setattr(consolidate.validity, "append_jsonl", append)
    else:
        result = run(settings, 1)
    origin = evidence.capture_origin(drive.suites / "generation_origin.json")
    assert origin is not None
    before = json.loads((drive.suites / starters._GENERATION_MANIFEST).read_text("utf-8"))
    if recover in {"delivery", "delivery_without_inputs"}:
        delivery_path = drive.suites / "suite_001" / starters._GENERATION_DELIVERY
        delivery = json.loads(delivery_path.read_text("utf-8"))
        paid = evidence.paid_attempt(delivery["id"])
        if recover == "delivery_without_inputs":
            delivery.pop("construction_inputs")
            delivery_path.write_text(json.dumps(delivery), encoding="utf-8")
    else:
        paid = evidence.read_suite_slots(before)["slots"][0]["terminal_attempt"]
    if recover != "fresh":
        assert before["suite_slots"]["slots"][0]["disposition_basis"] == "unavailable"
        resumed = RunSettings(base_dir=tmp_path, models=settings.models, sections={
            **settings.sections, "starter": {**settings.sections["starter"], "reuse_suites": True},
        })
        result = run(resumed, 2)
    after = json.loads((drive.suites / starters._GENERATION_MANIFEST).read_text("utf-8"))
    slot = evidence.read_suite_slots(after, drive.suites)["slots"][0]
    assert slot["terminal_attempt"] == paid and len(slot["attempts"]) == 1
    assert evidence.read_cleaned_source(slot["accepted_source"], drive.suites) == EDIT
    assert slot["attempts"][0]["original_source_sha256"] == evidence.source_sha256(BASE)
    without_inputs = recover == "delivery_without_inputs"
    assert slot["attempts"][0]["construction"]["origin"] == (None if without_inputs else origin)
    assert slot["attempts"][0]["construction"]["run_idx"] == (None if without_inputs else 1)
    assert result.requests_used == result.logical_requests == len(sends) == 1
    assert result.input_tokens == 10 and result.output_tokens == 3
    record = json.loads((tmp_path / "archive/consolidation_record.json").read_text("utf-8"))
    node = evidence.read_ancestry(record, EDIT)
    assert node["status"] == ("partial" if without_inputs else "complete")
    assert node["parents"][0]["origin"] == (None if without_inputs else origin)
    assert node["parents"][0]["invocation"] == paid["run"]
    assert node["run_idx"] == (1 if recover == "fresh" else 2)


@pytest.fixture
def boundary(tmp_path, monkeypatch, backend):
    root = tmp_path / "project"
    root.mkdir()
    target = root / "subject.py"
    target.write_text("target", encoding="utf-8")
    tests = tmp_path / "work/tests"
    tests.mkdir(parents=True)
    ctx = orchestrator.TargetPhasesContext(
        log_print=lambda *a, **k: None,
        project_name="fixture", workspace="target_number", top_file={"file_path": str(target)},
        canonical_path=target, work_tests=tests,
        results_dir=tmp_path / "results", target_logs_dir=tmp_path / "logs", reuse=orchestrator.ReuseFlags(),
        run_idx=2, generation_invocation_id=OTHER)
    origin = evidence.capture_origin(origin_file(tmp_path / "initial-origin.json"))
    parent = selected(BASE, seed(BASE, origin))
    parent.update(execution_name="test_subject.py", measurement_identity={}, delivered_tests=1)
    for phase in ("starter", "coverage"):
        archive = orchestrator._archive_dir(ctx, phase)
        (archive / "consolidation_record.json").write_text(json.dumps(parent), encoding="utf-8")
    monkeypatch.setattr(project, "resolve_source_dir", lambda *args: str(root))
    monkeypatch.setattr(consolidate, "consolidate_phase_end", fake_phase)
    monkeypatch.setattr(consolidate, "benchmark_tied_candidates", lambda *args, **kwargs: lambda c: 1.0)
    monkeypatch.setattr(consolidate, "record_selection_audit", lambda *args, **kwargs: None)
    @contextmanager
    def no_runner(**kwargs):
        yield None
    monkeypatch.setattr(metrics, "union_mutation_runner", no_runner)
    settings = RunSettings(sections={"coverage": {"phase_end_evidence_reuse": False},
                                     "mutation": {"phase_end_evidence_reuse": False}}, base_dir=tmp_path)
    return SimpleNamespace(ctx=ctx, parent=parent, settings=settings)


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
@pytest.mark.parametrize("winner", ["baseline", "agent", "merged", "reduced"])
def test_actual_editing_selection_keeps_only_contributing_inputs(boundary, monkeypatch, phase, winner):
    ctx = boundary.ctx
    (ctx.work_tests / "test_subject.py").write_text(EDIT, encoding="utf-8")
    source = BASE if winner == "baseline" else EDIT if winner == "agent" else MERGED
    label = {"baseline": "standalone_0", "agent": "standalone_1"}.get(winner, winner) if phase == "mutation" else winner
    monkeypatch.setattr(consolidate, "pick_phase_result", lambda *args, **kwargs: candidate(label, source))
    origin_file(orchestrator._archive_dir(ctx, phase) / "generation/record.json", phase)
    agent = SimpleNamespace(model_requests=1, measurement_identity={})
    result = orchestrator.TestPipelineResult(test_file_name="test_subject.py", test_result=agent,
                                             mutation_result=agent, mutation_inherited_source=BASE)
    with config.use_run_settings(boundary.settings):
        if phase == "coverage":
            orchestrator._consolidate_coverage(ctx, result, BASE)
        else:
            orchestrator._consolidate_mutation(ctx, result)
    record = json.loads((orchestrator._archive_dir(ctx, phase) / "consolidation_record.json").read_text("utf-8"))
    node = evidence.read_ancestry(record, source)
    assert node["run_idx"] == 2 and node["invocation"] == OTHER and node["phase"] == phase
    assert len(node["parents"]) == (1 if winner in {"baseline", "agent"} else 2)
    if winner == "agent":
        assert node["parents"][0]["operation"] == "edited"
        assert node["parents"][0]["origin"]["condition"]["phase"] == phase
    if winner == "baseline":
        assert node["parents"][0] == boundary.parent["generation_ancestry"]
    assert node["status"] == "complete"


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
def test_actual_unchanged_reuse_retains_parent_before_overwrite(boundary, monkeypatch, phase):
    ctx = boundary.ctx
    (ctx.work_tests / "test_subject.py").write_text(BASE, encoding="utf-8")
    parent_phase = "starter" if phase == "coverage" else "coverage"
    snapshot = copy.deepcopy(boundary.parent)
    inherited = consolidate.InheritedEvidence(record=snapshot, kill_matrix=None, coverage=None,
                                              kill_matrix_name=None, coverage_name=None)
    monkeypatch.setattr(consolidate, "inherited_phase_end_eligibility", lambda **kwargs: (inherited, ""))
    result = orchestrator.TestPipelineResult(
        test_file_name="test_subject.py", test_result=SimpleNamespace(model_requests=100),
        mutation_inherited_source=BASE, mutation_result=SimpleNamespace(model_requests=100))
    settings = RunSettings(sections={"coverage": {"phase_end_evidence_reuse": True},
                                     "mutation": {"phase_end_evidence_reuse": True}}, base_dir=ctx.work_tests.parent)
    with config.use_run_settings(settings):
        if phase == "coverage":
            orchestrator._consolidate_coverage(ctx, result, BASE)
        else:
            orchestrator._consolidate_mutation(ctx, result)
    path = orchestrator._archive_dir(ctx, phase) / "consolidation_record.json"
    before = json.loads(path.read_text("utf-8"))
    origin_file(ctx.results_dir / ctx.workspace / "r0" / "archive" / parent_phase / "generation/record.json",
                parent_phase, profile="later")
    (orchestrator._archive_dir(ctx, parent_phase) / "consolidation_record.json").write_text("{}", encoding="utf-8")
    node = evidence.read_ancestry(json.loads(path.read_text("utf-8")), BASE)
    assert node == before["generation_ancestry"]
    assert node["operation"] == "inherited" and node["origin"] is None
    assert node["parents"][0] == snapshot["generation_ancestry"]
    assert node["parents"][0]["run_idx"] == 1
    assert node["parents"][0]["origin"]["condition"]["model"]["model"] == "original"


def run_boundary(boundary, monkeypatch, phase, *, source=EDIT, inherited=BASE, requests=1):
    ctx = boundary.ctx
    (ctx.work_tests / "test_subject.py").write_text(source, encoding="utf-8")
    label = "standalone_1" if phase == "mutation" else "agent"
    monkeypatch.setattr(consolidate, "pick_phase_result", lambda *args, **kwargs: candidate(label, source))
    agent = SimpleNamespace(model_requests=requests, measurement_identity={})
    result = orchestrator.TestPipelineResult(
        test_file_name="test_subject.py", test_result=agent, mutation_result=agent,
        mutation_inherited_source=inherited,
    )
    with config.use_run_settings(boundary.settings):
        if phase == "coverage":
            orchestrator._consolidate_coverage(ctx, result, inherited)
        else:
            orchestrator._consolidate_mutation(ctx, result)
    path = orchestrator._archive_dir(ctx, phase) / "consolidation_record.json"
    return evidence.read_ancestry(json.loads(path.read_text("utf-8")), source)


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
@pytest.mark.parametrize("missing", ["origin", "requests", "predecessor", "injected"])
def test_actual_editing_unknown_inputs_stay_partial(boundary, monkeypatch, phase, missing):
    path = orchestrator._archive_dir(boundary.ctx, phase) / "generation/record.json"
    if missing != "origin":
        origin_file(path, phase)
    if missing == "predecessor":
        record = copy.deepcopy(boundary.parent)
        record.pop("generation_ancestry")
        parent_phase = "starter" if phase == "coverage" else "coverage"
        (orchestrator._archive_dir(boundary.ctx, parent_phase) / "consolidation_record.json").write_text(
            json.dumps(record), encoding="utf-8")
    if missing == "injected":
        from testuary import models as _model_source
        monkeypatch.setattr(_model_source, "current", lambda: object())
    node = run_boundary(boundary, monkeypatch, phase, requests=0 if missing == "requests" else 1)
    assert node["status"] == "partial"
    edited = node["parents"][0]
    if missing == "predecessor":
        assert edited["origin"] is not None
        assert edited["parents"][0]["status"] == "unavailable"
    else:
        assert edited["origin"] is None


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
def test_actual_unchanged_consolidation_does_not_attribute_current_model(boundary, monkeypatch, phase):
    origin_file(orchestrator._archive_dir(boundary.ctx, phase) / "generation/record.json", phase, profile="current")
    node = run_boundary(boundary, monkeypatch, phase, source=BASE, requests=100)
    assert node["status"] == "complete"
    assert node["parents"] == [boundary.parent["generation_ancestry"]]


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
def test_selected_record_repeat_keeps_detached_previous_construction(boundary, monkeypatch, phase):
    ctx = boundary.ctx
    origin_file(orchestrator._archive_dir(ctx, phase) / "generation/record.json", phase, profile="repeat2")
    old = run_boundary(boundary, monkeypatch, phase)
    # The next actual construction receives the previously selected source;
    # the phase record and captured profile may both be overwritten afterwards.
    parent_phase = "starter" if phase == "coverage" else "coverage"
    record = selected(EDIT, old)
    record.update(execution_name="test_subject.py", measurement_identity={})
    boundary.ctx = replace(ctx, run_idx=3, generation_invocation_id="f" * 32)
    # The later run reads the parent record in its own run directory.
    (orchestrator._archive_dir(boundary.ctx, parent_phase) / "consolidation_record.json").write_text(
        json.dumps(record), encoding="utf-8")
    later = run_boundary(boundary, monkeypatch, phase, source=EDIT, inherited=EDIT, requests=0)
    assert later["run_idx"] == 3 and later["invocation"] == "f" * 32
    assert later["parents"] == [old]
    old["origin"] = {"changed": True}
    origin_file(orchestrator._archive_dir(ctx, phase) / "generation/record.json", phase, profile="overwritten")
    assert later["parents"][0]["origin"] is None
    assert later["parents"][0]["parents"][0]["origin"]["condition"]["model"]["model"] == "repeat2"


def test_publication_failure_cannot_publish_new_ancestry_before_its_record(boundary, monkeypatch):
    ctx = boundary.ctx
    path = orchestrator._archive_dir(ctx, "coverage") / "consolidation_record.json"
    predecessor_path = orchestrator._archive_dir(ctx, "starter") / "consolidation_record.json"
    old_bytes = predecessor_path.read_bytes()
    failure = OSError("selected record unavailable")
    write = Path.write_text
    captured = []
    def unavailable(self, text, *args, **kwargs):
        if self == path:
            captured.append(json.loads(text))
            raise failure
        return write(self, text, *args, **kwargs)
    monkeypatch.setattr(Path, "write_text", unavailable)
    origin_file(orchestrator._archive_dir(ctx, "coverage") / "generation/record.json", "coverage")
    with pytest.raises(OSError) as raised:
        run_boundary(boundary, monkeypatch, "coverage")
    assert raised.value is failure and not path.exists()
    assert predecessor_path.read_bytes() == old_bytes
    assert evidence.read_ancestry(captured[0], EDIT)["status"] == "complete"


@pytest.mark.parametrize("change", [
    lambda attempt: attempt.update(generation_run_idx=True),
    lambda attempt: attempt["construction"].update(run_idx=2),
    lambda attempt: attempt["construction"]["origin"]["condition"]["model"].update(model="different"),
    lambda attempt: attempt["generation_origin"]["condition"].update(language_policy="unknown"),
    lambda attempt: attempt["generation_origin"]["inputs"]["prompt_profile"].update(
        contents="not json", sha256=evidence.source_sha256("not json")),
])
def test_suite_reader_refuses_contradictory_ancestry_and_inputs(tmp_path, change):
    origin = evidence.capture_origin(origin_file(tmp_path / "origin.json"))
    slots = evidence.SuiteSlots(tmp_path / "suites", 1)
    slots.record(1, f"{RUN}:0", classification="pruned", basis="accepted_cleaned",
                 original_source=BASE, cleaned_source=BASE, origin=origin, run_idx=1)
    value = {"suite_slots": slots.snapshot({1: 1})}
    change(value["suite_slots"]["slots"][0]["attempts"][0])
    with pytest.raises(evidence.GenerationEvidenceError):
        evidence.read_suite_slots(value)


@pytest.mark.parametrize("change", [
    lambda d: d.pop("construction_inputs"),
    lambda d: d["construction_inputs"].update(version=True),
    lambda d: d["construction_inputs"].update(run_idx=True),
    lambda d: d["construction_inputs"]["attempt_id"].update(run=OTHER),
    lambda d: d["construction_inputs"]["origin"]["condition"].update(phase=[]),
])
def test_unprocessed_paid_input_reader_never_infers_missing_context(tmp_path, change):
    origin = evidence.capture_origin(origin_file(tmp_path / "origin.json"))
    value = {"id": f"{RUN}:0", "construction_inputs": {
        "version": 1, "attempt_id": evidence.paid_attempt(f"{RUN}:0"), "run_idx": 1, "origin": origin,
    }}
    assert evidence.read_paid_inputs(value) == {"origin": origin, "run_idx": 1}
    change(value)
    assert evidence.read_paid_inputs(value) == {"origin": None, "run_idx": None}


def test_a_first_runs_starter_records_are_accepted(tmp_path):
    """The first run is run 0: its starter slots, a starter's construction
    and an unprocessed paid reply read back with run 0."""
    origin = evidence.capture_origin(origin_file(tmp_path / "origin.json"))
    slots = evidence.SuiteSlots(tmp_path / "suites", 1)
    slots.record(1, f"{RUN}:0", classification="pruned", basis="accepted_cleaned",
                 original_source=BASE, cleaned_source=EDIT, origin=origin, run_idx=0)
    record = slots.snapshot({1: 1})
    attempt = evidence.read_suite_slots({"suite_slots": record})["slots"][0]["attempts"][0]
    assert attempt["generation_run_idx"] == 0
    node = slots.construction_for(1, EDIT, evidence.construction_scope("starter", run_idx=0))
    ancestry = evidence.read_ancestry({"generation_ancestry": node,
                                       "selected_source_sha256": evidence.source_sha256(EDIT)})
    assert ancestry["status"] == "complete"
    paid = {"id": f"{RUN}:0", "construction_inputs": {
        "version": 1, "attempt_id": evidence.paid_attempt(f"{RUN}:0"), "run_idx": 0, "origin": origin,
    }}
    assert evidence.read_paid_inputs(paid) == {"origin": origin, "run_idx": 0}


def test_a_selected_suite_binds_to_its_slots_terminal_attempt(tmp_path):
    """An attempt recorded again keeps its list position but becomes terminal,
    and the selected starter binds to it, not to the list's last attempt."""
    origin = evidence.capture_origin(origin_file(tmp_path / "origin.json"))
    slots = evidence.SuiteSlots(tmp_path / "suites", 1)
    slots.record(1, f"{RUN}:0", classification="validation_unresolved", basis="unavailable",
                 original_source=BASE, origin=origin, run_idx=1)
    slots.record(1, f"{RUN}:1", classification="no code block", basis="no_code", origin=origin, run_idx=1)
    slots.record(1, f"{RUN}:0", classification="pruned", basis="accepted_cleaned",
                 original_source=BASE, cleaned_source=EDIT, origin=origin, run_idx=1)
    scope = evidence.construction_scope("starter", run_idx=1)
    node = slots.construction_for(1, EDIT, scope)
    assert node["operation"] == "accepted_suite"
    assert node["suite_attempt"]["attempt_id"] == evidence.paid_attempt(f"{RUN}:0")


def test_a_reused_suite_with_changed_bytes_binds_as_pruned(tmp_path):
    """A reused starter whose bytes changed after its acceptance binds as a
    pruning of its accepted construction, though reuse marks the changed slot
    unavailable."""
    origin = evidence.capture_origin(origin_file(tmp_path / "origin.json"))
    slots = evidence.SuiteSlots(tmp_path / "suites", 1)
    slots.record(1, f"{RUN}:0", classification="pruned", basis="accepted_cleaned",
                 original_source=BASE, cleaned_source=EDIT, origin=origin, run_idx=1)
    pruned = EDIT + "\n# pruned\n"
    assert slots.reconcile_reused_sources({1: pruned})
    node = slots.construction_for(1, pruned, evidence.construction_scope("starter", run_idx=1))
    assert node["operation"] == "pruned"
    assert node["parents"][0]["operation"] == "accepted_suite"
