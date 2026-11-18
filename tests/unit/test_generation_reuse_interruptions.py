"""Saved request work and canonical suite reuse through the actual callers."""
import csv
import json
import os
from dataclasses import replace

import pytest

import test_generation_origin_callers as harness
from testuary import orchestrator, phase_state
from testuary import model_requests as _model_requests
from testuary import config
from testuary import project as project_module
from testuary.evidence import finishing_settings, phase_origin_path, write_origin
from testuary import target_phases as phases
from testuary.settings import ConfigurationError, RunSettings
from contextlib import contextmanager
from per_suite_double import per_suite_double
from testuary import starters
from testuary import measurement

streams = harness.streams


def setup_case(tmp_path, *, profile="", hints="", phase_fields=None, **model_fields):
    project = tmp_path / "projects/subject"
    project.mkdir(parents=True, exist_ok=True)
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    target = project / "module.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    configured = harness.settings_at(tmp_path, profile=profile, phase_fields=phase_fields, **model_fields)
    sections = {
        "api": {"model": "explicit"}, "main": {"prompt_profile": profile, "hints_file": hints, "seed": 37},
        "starter": {"suite_count": 1, "attempts_per_suite": 1, "concurrent_requests": 1,
                         "suite_run_timeout_seconds": 2, "max_output_tokens": 40, **(phase_fields or {})},
        "project": {"name": "subject", "source_dir": str(project),
                    "output_dir": str(tmp_path / "output")},
    }
    settings = RunSettings(base_dir=tmp_path, sections=sections, models=configured.models,
                           environment=dict(os.environ))
    workspace = project_module.target_workspace_name(target, project)
    ctx = phases.TargetPhasesContext(
        project_name="subject", workspace=workspace,
        top_file={"target_number": 1, "file_path": str(target)}, canonical_path=target,
        work_tests=tmp_path / "work/tests",
        results_dir=tmp_path / "output/results", target_logs_dir=tmp_path / "logs",
        reuse=phases.ReuseFlags(), settings=settings, log_print=lambda *a, **kw: None,
    )
    return project, target, ctx, settings


def request_only_pool(ctx, settings):
    # The saved starter pool of the context's own run.
    suites = project_module.run_dir(ctx.results_dir / ctx.workspace, ctx.run_idx) / "archive/starter/suites"
    suites.mkdir(parents=True)
    record = {"run": "interrupted", "slot": 1, "logical_request_idx": 1,
              "request_idx": 1, "outcome": _model_requests.TRANSIENT, "usage_known": False}
    (suites / "request_records.jsonl").write_text(json.dumps(record) + "\n", encoding="utf-8")
    with config.use_run_settings(settings):
        write_origin(suites / "generation_origin.json", phases._context_generation_condition(ctx, "starter"))
    # The generation's manifest, written before its first request.
    (suites / starters._GENERATION_MANIFEST).write_text(
        json.dumps({"generation_complete": False}), encoding="utf-8")
    assert starters._has_persisted_pool(suites)
    assert not list(suites.glob("suite_*"))
    return suites


def starter_call(ctx, suites):
    return starters.run_starter_phase(
        language="python", target_file_path=str(ctx.canonical_path),
        source_dir=str(ctx.canonical_path.parent), test_output_dir=str(ctx.work_tests),
        suites_dir=str(suites), log=lambda *a, **kw: None,
    )


def public_args(*extra):
    from testuary.__main__ import _build_parser, _check_run_options

    args = _build_parser().parse_args(["--source-files", "module.py", *extra])
    _check_run_options(args)
    return args


@pytest.mark.parametrize("entry", ["public", "context", "starter"])
@pytest.mark.parametrize("change", ["temperature", "profile", "missing-origin"])
def test_request_only_reuse_refuses_before_any_saved_work_changes(entry, change, tmp_path, monkeypatch, streams):
    profile = harness.profile_at(tmp_path)
    project, target, ctx, settings = setup_case(tmp_path, profile=profile.name,
                                              phase_fields={"reuse_suites": True})
    suites = request_only_pool(ctx, settings)
    kwargs = {"profile": profile.name, "phase_fields": {"reuse_suites": True}}
    if change == "temperature":
        kwargs["temperature"] = 0.5
    elif change == "profile":
        other = tmp_path / "changed-profile.json"
        other.write_text(json.dumps({"format": 1, "phases": {"initial": {
            "system_prompt": {"append": "different guidance"}}}}), encoding="utf-8")
        kwargs["profile"] = other.name
    else:
        (suites / "generation_origin.json").unlink()
    _, _, _, changed = setup_case(tmp_path, **kwargs)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kw: pytest.fail("cleanup before refusal"))
    monkeypatch.setattr(phases, "_phase_starter", lambda *a: pytest.fail("generation before refusal"))
    ctx = replace(ctx, settings=changed)
    monkeypatch.setattr(phases, "_record_start", lambda *a: pytest.fail("marker before refusal"))
    with pytest.raises(ConfigurationError, match="reuse refused"):
        if entry == "public":
            orchestrator.run_pipeline(public_args(), settings=changed)
        elif entry == "context":
            with config.use_run_settings(changed):
                phases.run_target_phases(ctx)
        else:
            with config.use_run_settings(changed):
                starter_call(ctx, suites)
    assert streams[0] == []
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_request_only_initial_continues_with_original_inputs_and_spent_requests(tmp_path, streams):
    profile = harness.profile_at(tmp_path)
    hints = tmp_path / "hints.json"
    hints.write_text('{"initial": {"general": "retained seed guidance"}}', encoding="utf-8")
    project, target, ctx, settings = setup_case(tmp_path, profile=profile.name,
                                              hints=hints.name,
                                              phase_fields={"reuse_suites": True})
    suites = request_only_pool(ctx, settings)
    original = {p.name: p.read_bytes() for p in suites.iterdir() if p.is_file()}
    renamed = tmp_path / "renamed-profile.json"
    renamed.write_bytes(profile.read_bytes())
    renamed_hints = tmp_path / "renamed-hints.json"
    renamed_hints.write_bytes(hints.read_bytes())
    _, _, _, changed = setup_case(tmp_path, profile=renamed.name, hints=renamed_hints.name, api_key="rotated",
                                  request_timeout_seconds=700,
                                  phase_fields={"reuse_suites": True, "max_transient_failures_per_phase": 200})
    with config.use_run_settings(changed):
        result = starter_call(ctx, suites)
    assert len(streams[0]) == 1
    assert result.requests_used == result.logical_requests == 2
    assert result.unknown_usage_requests == 1
    assert (suites / "request_records.jsonl").read_bytes().startswith(original["request_records.jsonl"])
    # The continued generation appends its records and rewrites its manifest.
    for name, data in original.items():
        if name not in ("request_records.jsonl", starters._GENERATION_MANIFEST):
            assert (suites / name).read_bytes() == data


@pytest.mark.parametrize("limit", ["max_transient_failures_per_phase", "max_consecutive_transient_failures"])
def test_request_only_initial_gives_a_resumed_session_its_own_limits(limit, tmp_path, monkeypatch, streams):
    """A resumed session counts only its own transient failures toward its
    limit. The earlier record stays and the totals hold both sessions."""
    _, _, ctx, settings = setup_case(tmp_path, phase_fields={
        "reuse_suites": True, limit: 2, "transient_first_wait_seconds": 0,
        "max_transient_wait_seconds": 0,
    })
    suites = request_only_pool(ctx, settings)
    before = (suites / "request_records.jsonl").read_bytes()
    origin = (suites / "generation_origin.json").read_bytes()
    sent = []

    def failed_transport(api_cfg, messages, **kwargs):
        sent.append(messages)
        kwargs["on_request"](kwargs["seed"], kwargs["model_settings"].values["max_output_tokens"])
        raise TimeoutError("offline connection failure")

    monkeypatch.setattr(starters, "_default_generate", failed_transport)
    with config.use_run_settings(settings):
        result = starter_call(ctx, suites)
    assert len(sent) == 2
    assert result.requests_used == 3 and result.unknown_usage_requests == 3
    assert result.status != "ok"
    assert (suites / "request_records.jsonl").read_bytes().startswith(before)
    assert (suites / "generation_origin.json").read_bytes() == origin
    assert streams[0] == []


@pytest.mark.parametrize("run_idx", [1, 2])
def test_request_only_phase_start_copies_original_origin_without_relabeling(tmp_path, streams, run_idx):
    profile = harness.profile_at(tmp_path)
    _, _, ctx, settings = setup_case(tmp_path, profile=profile.name, phase_fields={"reuse_suites": True})
    # Each run reuses its own saved starters, never another run's.
    ctx = replace(ctx, run_idx=run_idx)
    suites = request_only_pool(ctx, settings)
    before = {p.name: p.read_bytes() for p in suites.iterdir() if p.is_file()}
    renamed = tmp_path / "renamed-profile.json"
    renamed.write_bytes(profile.read_bytes())
    _, _, _, changed = setup_case(tmp_path, profile=renamed.name, api_key="rotated",
                                  phase_fields={"reuse_suites": True})
    with config.use_run_settings(changed):
        phases._start_phase(ctx, "starter")
    source = json.loads(before["generation_origin.json"])
    destination = phase_origin_path(ctx.results_dir / ctx.workspace, "starter", run_idx)
    copied = json.loads(destination.read_text())
    assert copied["settings"] == source["settings"] and copied["condition"] == source["condition"]
    assert copied["prompt_profile"]["path"] == source["prompt_profile"]["path"]
    assert (destination.parent / copied["prompt_profile"]["copy"]).read_bytes() == profile.read_bytes()
    assert {p.name: p.read_bytes() for p in suites.iterdir() if p.is_file()} == before
    assert streams[0] == []


@pytest.mark.parametrize("empty", ["directory", "empty-request-log"])
def test_empty_suite_work_is_a_fresh_target_not_unknown_saved_generation(tmp_path, streams, empty):
    _, _, ctx, settings = setup_case(tmp_path, phase_fields={"reuse_suites": True})
    suites = ctx.results_dir / ctx.workspace / "r0/archive/starter/suites"
    (suites / "suite_001").mkdir(parents=True)
    if empty == "empty-request-log":
        (suites / "request_records.jsonl").write_text("", encoding="utf-8")
    assert not starters._has_persisted_pool(suites)
    with config.use_run_settings(settings):
        result = starter_call(ctx, suites)
    assert result.requests_used == len(streams[0]) == 1
    assert (suites / "generation_origin.json").is_file()


def prepare_repeat(ctx, settings, monkeypatch, phase, *, canonical_matches, repeat_origin=True):
    ctx = replace(ctx, run_idx=1,
                  reuse=phases.ReuseFlags(coverage=True, mutation=phase == "mutation", finishing=True),
                  stop_after_phase=phase)
    target_dir = ctx.results_dir / ctx.workspace
    ctx.work_tests.mkdir(parents=True)
    canonical = ctx.work_tests / "test_module.py"
    canonical.write_text("def test_canonical():\n    assert True\n", encoding="utf-8")
    with config.use_run_settings(settings):
        for part in ("coverage", "mutation") if phase == "mutation" else ("coverage",):
            condition = phases._context_generation_condition(ctx, part)
            write_origin(phase_origin_path(target_dir, part, 0), condition)
            if repeat_origin:
                write_origin(phase_origin_path(target_dir, part, 1), condition)
    if not canonical_matches:
        changed = RunSettings(base_dir=settings.base_dir, sections=settings.sections,
                              models={"explicit": {**settings.models["explicit"], "temperature": 0.5}},
                            environment=settings.environment)
        with config.use_run_settings(changed):
            write_origin(phase_origin_path(target_dir, phase, 0), phases._context_generation_condition(ctx, phase))
    phase_state.save(target_dir, {
        "coverage": {"0": {"metrics": {"coverage": 62.5, "branch_coverage": 45.0}},
                     "1": {"metrics": {"coverage": 99.0, "branch_coverage": 98.0}}},
        "mutation": {"0": {"metrics": {"mutation_score": 37.5, "mutation_file": "test_module.py"}},
                     "1": {"metrics": {"mutation_score": 97.0, "mutation_file": "unused_repeat.py"}}},
    })
    seen = []
    monkeypatch.setattr(phases, "_verify_restored_phase_validity",
                        lambda context, part, path: seen.append((part, path.read_bytes())) or (True, ""))
    monkeypatch.setattr(phases, "_restored_test_file_collects", lambda *a: (True, ""))
    monkeypatch.setattr(phases, "_check_restored_delivered_suite", lambda *a: (True, ""))
    monkeypatch.setattr(phases, "_record_phase_restored", lambda *a: None)
    return ctx, canonical, seen


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
def test_repeat_context_rejects_mismatched_canonical_origin_before_writes(phase, tmp_path, monkeypatch, streams):
    _, _, ctx, settings = setup_case(tmp_path)
    ctx, _, _ = prepare_repeat(ctx, settings, monkeypatch, phase, canonical_matches=False)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with pytest.raises(ConfigurationError, match=f"requested {phase} reuse refused"), \
            config.use_run_settings(ctx.settings):
        phases.run_target_phases(ctx)
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before
    assert streams[0] == []


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
@pytest.mark.parametrize("repeat_origin", [False, True])
def test_repeat_context_consumes_canonical_suite_and_run_one_metrics(phase, repeat_origin, tmp_path, monkeypatch, streams):
    _, _, ctx, settings = setup_case(tmp_path)
    ctx, canonical, seen = prepare_repeat(ctx, settings, monkeypatch, phase,
                                          canonical_matches=True, repeat_origin=repeat_origin)
    if repeat_origin:
        # The unused repeat origin may describe different generation, just as its metrics do.
        other = RunSettings(base_dir=settings.base_dir, sections=settings.sections,
                            models={"explicit": {**settings.models["explicit"], "temperature": 0.5}},
                            environment=settings.environment)
        with config.use_run_settings(other):
            write_origin(phase_origin_path(ctx.results_dir / ctx.workspace, phase, 1),
                         phases._context_generation_condition(ctx, phase))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    with config.use_run_settings(ctx.settings):
        result = phases.run_target_phases(ctx)
    assert result.test_file_name == canonical.name
    assert result.test_result.line_coverage == 62.5 and result.test_result.branch_coverage == 45.0
    if phase == "mutation":
        assert result.mutation_result.mutation_score == 37.5
        assert result.mutation_result.mutation_test_file_name == canonical.name
    assert seen and all(data == canonical.read_bytes() for _, data in seen)
    assert streams[0] == []
    for relative, data in before.items():
        assert (tmp_path / relative).read_bytes() == data


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
@pytest.mark.parametrize("canonical_matches", [True, False])
def test_public_repeat_preflight_checks_only_consumed_canonical_origin(phase, canonical_matches, tmp_path, monkeypatch, streams):
    _, _, ctx, settings = setup_case(tmp_path)
    target_dir = ctx.results_dir / ctx.workspace
    with config.use_run_settings(settings):
        for part in ("starter", "coverage", "mutation") if phase == "mutation" else ("starter", "coverage"):
            write_origin(phase_origin_path(target_dir, part, 0), phases._context_generation_condition(ctx, part))
    if not canonical_matches:
        other = harness.settings_at(tmp_path, temperature=0.5)
        with config.use_run_settings(other):
            write_origin(phase_origin_path(target_dir, phase, 0), phases._context_generation_condition(ctx, phase))
        with config.use_run_settings(settings):
            write_origin(phase_origin_path(target_dir, phase, 1), phases._context_generation_condition(ctx, phase))
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    # Only generation origins are checked here, so skip the artifact preflight.
    monkeypatch.setattr(orchestrator, "_preflight_reuse_requirements", lambda *a: [])
    class ReachedCleanup(Exception):
        pass
    def reached(**kw):
        raise ReachedCleanup
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", reached)
    with pytest.raises(ReachedCleanup if canonical_matches else ConfigurationError):
        orchestrator.run_pipeline(public_args("--runs", "2", "--reuse",
                                              "coverage,mutation" if phase == "mutation" else "coverage"),
                                  settings=settings)
    assert streams[0] == []
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def completed_repeat(ctx, settings):
    target_dir = ctx.results_dir / ctx.workspace
    with config.use_run_settings(settings):
        # Each run records every phase it completed.
        for phase in ("starter", "coverage", "mutation", "finishing"):
            for run_idx in (0, 1):
                artifact = target_dir / project_module.run_name(run_idx) / "archive" / phase / "delivered.py"
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text(f"def test_{phase}_{run_idx}():\n    assert True\n", encoding="utf-8")
                phase_state.mark_started(
                    target_dir, phase, target_number=1, stem="module", run_idx=run_idx,
                    target_file_path="module.py", canonical_source_path=ctx.canonical_path,
                )
                phase_state.mark_complete(
                    target_dir, phase, run_idx=run_idx,
                    artifacts={artifact.relative_to(target_dir).as_posix(): artifact},
                    metrics={"coverage": 60 + run_idx, "mutation_score": 40 + run_idx,
                             # Finishing records the settings it measured under.
                             **({"finishing_settings": finishing_settings()}
                                if phase == "finishing" else {})},
                )
                if phase != "finishing":
                    write_origin(phase_origin_path(target_dir, phase, run_idx),
                                 phases._context_generation_condition(ctx, phase))
    fields = ["target_number", "run", *phase_state._RESUME_REQUIRED_CSV_FIELDS]
    rows = [{name: str(run) for name in fields} for run in (0, 1)]
    for row in rows:
        row["target_number"] = "1"
    with (ctx.results_dir / "tests.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    with config.use_run_settings(settings):
        for run_idx in (0, 1):
            done, prefix, restore = phase_state.plan_test_resume(
                target_results_dir=target_dir,
                rel_target=ctx.canonical_path.relative_to(ctx.canonical_path.parent),
                canonical_path=ctx.canonical_path, run_idx=run_idx,
            )
            assert done and not any(prefix.values()) and not restore
    return target_dir, rows


def change_repeat_origin(ctx, settings, phase, change):
    path = phase_origin_path(ctx.results_dir / ctx.workspace, phase, 1)
    if change == "missing":
        path.unlink()
    else:
        other = RunSettings(base_dir=settings.base_dir, sections=settings.sections,
                            models={"explicit": {**settings.models["explicit"], "temperature": 0.5}},
                            environment=settings.environment)
        with config.use_run_settings(other):
            write_origin(path, phases._context_generation_condition(ctx, phase))


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
@pytest.mark.parametrize("change", ["missing", "different"])
def test_completed_public_resume_refuses_unverified_retained_repeat(phase, change, tmp_path, monkeypatch, streams):
    _, _, ctx, settings = setup_case(tmp_path)
    completed_repeat(ctx, settings)
    change_repeat_origin(ctx, settings, phase, change)
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kw: pytest.fail("cleanup before refusal"))
    with pytest.raises(ConfigurationError, match=f"requested {phase} reuse refused"):
        orchestrator.run_pipeline(public_args("--resume", "--runs", "2"), settings=settings)
    assert streams[0] == []
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


def test_completed_public_resume_keeps_every_compatible_repeat_without_generation(tmp_path, monkeypatch, streams, capsys):
    _, _, ctx, settings = setup_case(tmp_path)
    target_dir, rows = completed_repeat(ctx, settings)
    # A skip may rewrite the checkpoint and the summaries built from tests.csv.
    rebuilt = ("phase_state.json", "summary.json")
    before = {p.relative_to(target_dir): p.read_bytes() for p in target_dir.rglob("*")
              if p.is_file() and p.name not in rebuilt}
    state = phase_state.load(target_dir)
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_check_canonical_against_git", lambda *a: [])
    monkeypatch.setattr(orchestrator, "_copy_project_to_work", lambda **kw: pytest.fail("copy on completed skip"))
    monkeypatch.setattr(phases, "run_target_phases", lambda *a: pytest.fail("generation on completed skip"))
    assert orchestrator.run_pipeline(public_args("--resume", "--runs", "2", "--no-color"), settings=settings) == 0
    assert "already complete (resume skip)" in capsys.readouterr().out
    assert streams[0] == []
    assert {p.relative_to(target_dir): p.read_bytes() for p in target_dir.rglob("*")
            if p.is_file() and p.name not in rebuilt} == before
    for phase in ("starter", "coverage", "mutation", "finishing"):
        assert phase_state.load(target_dir)[phase] == state[phase]
    with (ctx.results_dir / "tests.csv").open(encoding="utf-8-sig", newline="") as handle:
        assert list(csv.DictReader(handle)) == rows


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
@pytest.mark.parametrize("change", ["missing", "different"])
@pytest.mark.parametrize("incomplete", ["csv", "finishing"])
def test_partial_public_resume_checks_the_origins_each_run_restores(phase, change, incomplete, tmp_path, monkeypatch, streams):
    """A run restoring its own coverage and mutation refuses the resume before
    any work when their origin is changed or missing, as the first run's would."""
    _, _, ctx, settings = setup_case(tmp_path)
    target_dir, _ = completed_repeat(ctx, settings)
    change_repeat_origin(ctx, settings, phase, change)
    if incomplete == "csv":
        (ctx.results_dir / "tests.csv").unlink()
    else:
        phase_state.mark_failed(target_dir, "finishing", run_idx=1, reason="interrupted measurement")
    with config.use_run_settings(settings):
        done, prefix, restore = phase_state.plan_test_resume(
            target_results_dir=target_dir, rel_target=ctx.canonical_path.relative_to(ctx.canonical_path.parent),
            canonical_path=ctx.canonical_path, run_idx=1,
        )
    # Finishing measures again: a resume never skips it while the pipeline
    # runs, so the mutation minimum always judges a fresh final score.
    assert not done and prefix["coverage"] and prefix["mutation"] and not prefix["finishing"] and restore
    before = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    # Only generation origins are checked here, so skip the artifact preflight.
    monkeypatch.setattr(orchestrator, "_preflight_reuse_requirements", lambda *a: [])
    class ReachedCleanup(Exception):
        pass
    def reached(**kw):
        raise ReachedCleanup
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", reached)
    with pytest.raises(ConfigurationError, match=f"requested {phase} reuse refused"):
        orchestrator.run_pipeline(public_args("--resume", "--runs", "2"), settings=settings)
    assert streams[0] == []
    assert {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()} == before


# A plain --resume after a cancel in the starter phase requests no starter again.
#
# The run is cancelled during the starter suites' mutation measurement, after
# every starter was generated. The resumed run reuses the saved starters: no
# new request, each slot's spent attempts unchanged, and the request records
# counting exactly the requests that were sent.
SUITE = "from module import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def test_a_resume_after_a_cancel_in_the_starter_phase_sends_no_starter_request_again(
    tmp_path, monkeypatch, streams,
):
    _project, _target, ctx, settings = setup_case(
        tmp_path, phase_fields={"suite_count": 2, "attempts_per_suite": 2, "suite_run_timeout_seconds": 120})
    sent = []

    def generate(api_cfg, messages, **kwargs):
        sent.append(kwargs["seed"])
        if kwargs.get("on_request") is not None:
            kwargs["on_request"](kwargs["seed"], kwargs.get("max_output_tokens"))
        return f"Here is the suite.\n\n```python\n{SUITE}```\n", {"input_tokens": 10, "output_tokens": 3}

    monkeypatch.setattr(starters, "_default_generate", generate)

    @contextmanager
    def cancelled(**kw):
        raise KeyboardInterrupt
        yield  # pragma: no cover

    monkeypatch.setattr(measurement, "union_mutation_runner", cancelled)
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run_pipeline(public_args("--stop-after-phase", "starter"), settings=settings)
    assert len(sent) == 2
    starter_dir = ctx.results_dir / ctx.workspace / "r0/archive/starter/suites"
    records = (starter_dir / "request_records.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(records) == 2

    @contextmanager
    def measuring(**kw):
        yield per_suite_double(lambda d: {"mutants": [
            {"id": "m1", "operator": "core/Op", "line": 2, "diff": "m1", "status": "killed",
             "killer": "test_module.py::test_add"}]})

    monkeypatch.setattr(measurement, "union_mutation_runner", measuring)
    # Returning without a stop error: the starter phase completed.
    orchestrator.run_pipeline(public_args("--resume", "--stop-after-phase", "starter"),
                              settings=settings)
    assert len(sent) == 2
    assert len((starter_dir / "request_records.jsonl").read_text(encoding="utf-8").splitlines()) == 2
    manifest = json.loads((starter_dir / starters._GENERATION_MANIFEST).read_text(encoding="utf-8"))
    assert manifest["generation_attempts_per_suite"] == {"1": 1, "2": 1}
    # The timing says the resumed starter phase reused saved starters.
    assert _starter_segments(ctx) == [False, True]


def _starter_segments(ctx):
    from testuary import phase_state

    block = phase_state.load(ctx.results_dir / ctx.workspace)["starter"]["0"]
    return [segment["starter_suites_reused"] for segment in block["segments"]]


def test_the_timing_says_saved_starters_were_reused_only_when_they_were(
    tmp_path, monkeypatch, streams,
):
    """A resume after a cancel before the starter phase saved anything
    generates afresh, and its timing does not claim a reuse."""
    import shutil

    _project, _target, ctx, settings = setup_case(
        tmp_path, phase_fields={"suite_count": 1, "attempts_per_suite": 1, "suite_run_timeout_seconds": 120})
    starter_dir = ctx.results_dir / ctx.workspace / "r0/archive/starter/suites"
    state = {"cancel": True}

    def generate(api_cfg, messages, **kwargs):
        if state["cancel"]:
            raise KeyboardInterrupt
        if kwargs.get("on_request") is not None:
            kwargs["on_request"](kwargs["seed"], kwargs.get("max_output_tokens"))
        return f"Here is the suite.\n\n```python\n{SUITE}```\n", {"input_tokens": 10, "output_tokens": 3}

    @contextmanager
    def measuring(**kw):
        yield per_suite_double(lambda d: {"mutants": [
            {"id": "m1", "operator": "core/Op", "line": 2, "diff": "m1", "status": "killed",
             "killer": "test_module.py::test_add"}]})

    monkeypatch.setattr(starters, "_default_generate", generate)
    monkeypatch.setattr(measurement, "union_mutation_runner", measuring)
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    with pytest.raises(KeyboardInterrupt):
        orchestrator.run_pipeline(public_args("--stop-after-phase", "starter"), settings=settings)
    # As if the cancel came before the starter phase saved anything.
    shutil.rmtree(starter_dir)
    assert not starters._has_persisted_pool(starter_dir)
    state["cancel"] = False
    orchestrator.run_pipeline(public_args("--resume", "--stop-after-phase", "starter"),
                              settings=settings)
    assert _starter_segments(ctx) == [False, False]
