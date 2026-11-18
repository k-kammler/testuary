"""Typed inputs reach the existing engine without CLI-text conversion."""
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
import subprocess

import pytest

import test_generation_reuse_interruptions as harness
from testuary import orchestrator
from testuary import config
from testuary.orchestrator import RunRequest
from testuary import target_phases as phases
from testuary.settings import ConfigurationError


@pytest.fixture(autouse=True)
def no_native_or_provider(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native process"))
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_check_canonical_against_git", lambda *a: [])


def test_request_detaches_sequences_without_splitting_paths():
    sources, reuse, selected = ["a,b.py"], [" Starter "], [" Python "]
    request = RunRequest(sources, reuse=reuse, resume=True, languages=selected)
    sources[0], reuse[0], selected[0] = "changed", "test", "java"
    assert request.source_files == ("a,b.py",)
    assert request.reuse == ("starter",) and request.languages == ("python",)
    with pytest.raises(FrozenInstanceError):
        request.runs = 2


@pytest.mark.parametrize("fields", [
    {"source_files": []}, {"source_files": "one.py"}, {"source_files": [""]},
    {"source_files": ["a.py", "a.py"]}, {"source_files": [None]},
    {"source_files": ["a\0.py"]}, {"source_files": [f"a{i}.py" for i in range(101)]},
    {"runs": True}, {"runs": 0}, {"runs": 1.5}, {"run": -1}, {"run": 1.0},
    {"runs": 2, "run": 0}, {"resume": "true"},
    {"reuse": "starter"}, {"reuse": ["scaled"]}, {"reuse": [""]},
    {"reuse": ["coverage"], "resume": True}, {"reuse": ["mutation", "finishing"], "resume": True},
    {"reuse": ["test"]},
    {"languages": "python"}, {"languages": []}, {"languages": ["not_registered"]},
    {"stop_after_phase": "not_a_phase"}, {"targets": "nonsense"}, {"targets": 1}, {"targets": ""},
])
def test_request_refuses_invalid_policy_before_work(fields, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError):
        RunRequest(**{"source_files": ["one.py"], **fields})
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("extra", [[], ["--resume"], ["--reuse", "starter,mutation"]])
@pytest.mark.parametrize("outcome", [0, 1, RuntimeError("controlled body failure"), KeyboardInterrupt()])
def test_a_request_and_parsed_arguments_share_one_resolved_lifecycle(extra, outcome, tmp_path, monkeypatch):
    _, _, _, settings = harness.setup_case(tmp_path)
    parsed = harness.public_args("--runs", "2", *extra)
    reused = tuple(name for name in ("starter", "mutation") if getattr(parsed, f"reuse_{name}"))
    request = RunRequest(("module.py",), runs=2, reuse=reused, resume=parsed.resume)
    observations = []

    def execute(args, selected_languages):
        observations.append((args.runs, args.run, args.resume, args.targets,
                             args.reuse_starter, args.reuse_mutation, config.active_settings()))
        assert selected_languages is None
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome

    monkeypatch.setattr(orchestrator, "_run_pipeline", execute)
    for invocation in (parsed, request):
        if isinstance(outcome, BaseException):
            with pytest.raises(type(outcome)) as caught:
                orchestrator.run_pipeline(invocation, settings=settings)
            assert caught.value is outcome
        else:
            assert orchestrator.run_pipeline(invocation, settings=settings) == outcome
        assert config.active_settings() is None
    assert observations[0] == observations[1]
    assert request.source_files == ("module.py",) and request.targets is None


@pytest.mark.parametrize("source_name", ["module.py", "file,with,commas.py"])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_actual_typed_pipeline_preserves_path_and_original_on_every_exit(
        source_name, outcome, tmp_path, monkeypatch):
    project, target, _, settings = harness.setup_case(tmp_path)
    target.rename(project / source_name)
    target = project / source_name
    before = {path: (path.read_bytes(), path.stat().st_mode) for path in project.rglob("*") if path.is_file()}
    request = RunRequest([source_name], languages=["python"], stop_after_phase="coverage")
    calls = []
    cancelled = KeyboardInterrupt()

    def run_phase(ctx):
        assert ctx.canonical_path == target
        assert Path(ctx.top_file["file_path"]).name == source_name
        assert Path(ctx.top_file["file_path"]).read_bytes() == before[target][0]
        calls.append(ctx)
        if outcome == "failure":
            raise RuntimeError("controlled phase error")
        if outcome == "cancelled":
            raise cancelled
        return phases.TestPipelineResult(stopped_after_phase="coverage")

    monkeypatch.setattr(phases, "run_target_phases", run_phase)
    if outcome == "cancelled":
        with pytest.raises(KeyboardInterrupt) as caught:
            orchestrator.run_pipeline(request, settings=settings)
        assert caught.value is cancelled
    else:
        assert orchestrator.run_pipeline(request, settings=settings) == (1 if outcome == "failure" else 0)
    assert len(calls) == 1 and config.active_settings() is None
    assert {path: (path.read_bytes(), path.stat().st_mode) for path in before} == before
    assert request.source_files == (source_name,) and request.targets is None


@pytest.mark.parametrize("invalid", ["missing", "dictionary", "duplicate_languages"])
def test_typed_admission_refuses_implicit_settings_or_duplicate_selector(invalid, tmp_path, monkeypatch):
    _, _, _, settings = harness.setup_case(tmp_path)
    request = RunRequest(["module.py"])
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda *a: pytest.fail("body before refusal"))
    if invalid == "missing":
        settings = None
    elif invalid == "dictionary":
        settings = settings.export()
    with pytest.raises(ConfigurationError):
        orchestrator.run_pipeline(request, {"python"} if invalid == "duplicate_languages" else None,
                                  settings=settings)
    assert not (tmp_path / "output").exists()


def test_actual_typed_sequential_targets_leave_request_immutable(tmp_path, monkeypatch):
    project, _, _, settings = harness.setup_case(tmp_path)
    (project / "second.py").write_bytes(b"second target")
    request = RunRequest(["module.py", "second.py"], targets="2", stop_after_phase="coverage")
    calls = []
    monkeypatch.setattr(phases, "run_target_phases", lambda ctx: (
        calls.append(ctx.canonical_path.name) or phases.TestPipelineResult(stopped_after_phase="coverage")))
    for number in (1, 2):
        chosen = replace(settings, sections={**settings.sections, "project": {
            **settings.sections["project"], "output_dir": str(tmp_path / f"out{number}"),
        }})
        assert orchestrator.run_pipeline(request, settings=chosen) == 0
    assert calls == ["second.py", "second.py"] and request.targets == "2"
