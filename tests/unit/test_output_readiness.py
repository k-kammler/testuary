"""Inspection uses execution's output guards without acquiring or deleting."""
from dataclasses import replace
from pathlib import Path
import subprocess
import stat

import pytest

import test_generation_reuse_interruptions as harness
from test_output_admission import directory_alias as directory_alias
from testuary import orchestrator
from testuary import config
from testuary import project as project_module
from testuary.orchestrator import RunRequest
from testuary.readiness import _configuration_report, _output_report
from testuary.settings import ConfigurationError, RunSettings


@pytest.fixture(autouse=True)
def no_execution_or_cleanup(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("native launch"))
    monkeypatch.setattr(project_module, "project_runtime", lambda *a, **k: pytest.fail("runtime probe"))
    monkeypatch.setattr(orchestrator, "_remove_path", lambda *a: pytest.fail("output removal"))
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kw: pytest.fail("output cleanup"))
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)


def settings_for(tmp_path, *, output="output", source="original", fields=None, main=None):
    return RunSettings(base_dir=tmp_path, sections={
        "api": {"model": "selected"}, "main": main or {},
        "project": {"source_dir": source, "output_dir": output}, **(fields or {}),
    }, models={"selected": {"provider": "openai", "model": "fixture", "api_url": "https://example.test/v1",
                            "api_key": "output-fixture-secret"}})


def snapshot(root):
    return {str(path.relative_to(root)): (path.read_bytes(), path.stat().st_mode)
            for path in root.rglob("*") if path.is_file() and not path.is_symlink()}


def facts(tmp_path, *, settings=None, request=None, scope="inspection"):
    project = tmp_path / "original"
    project.mkdir(exist_ok=True)
    (project / "module.py").write_text("value = 1\n", encoding="utf-8")
    settings = settings or settings_for(tmp_path)
    request = request or RunRequest(("module.py",))
    return settings, request, replace(_configuration_report(request, settings), scope=scope)


@pytest.mark.parametrize("scope", ["inspection", "verification"])
def test_output_and_setup_composition_print_nothing(scope, tmp_path, capsys):
    from langadapter import Environment, Interpreter, LocalExecution, ProjectContext
    from testuary.readiness import _setup_request

    secret = "output-fixture-secret"
    settings = replace(settings_for(tmp_path),
                       sections={"api": {"model": "selected"},
                                 "project": {"name": secret, "source_dir": "original", "output_dir": "output"}})
    settings, invocation, configuration = facts(tmp_path, settings=settings, scope=scope)
    context = ProjectContext(root=str(tmp_path / "original"),
                             interpreter=Interpreter(path="not launched"),
                             execution=LocalExecution(), environment=Environment({}, inherit=False))
    before = snapshot(tmp_path), {str(path) for path in tmp_path.rglob("*")}
    first = _output_report(configuration, invocation, settings)
    report = _output_report(first, invocation, settings)
    request = _setup_request(invocation, settings, context=context, required_operations=("tests.run_suite",))
    captured = capsys.readouterr()
    assert captured.out == captured.err == "" and report == first
    assert request.context is context and request.source_files == ("module.py",)
    assert secret not in repr(report)
    assert report.status == "unverified" and report.exit_code == (0 if scope == "inspection" else 1)
    assert (snapshot(tmp_path), {str(path) for path in tmp_path.rglob("*")}) == before
    assert config.active_settings() is None


@pytest.mark.parametrize("destination", ["missing", "empty"])
@pytest.mark.parametrize("scope", ["inspection", "verification"])
def test_host_route_admission_leaves_source_membership_and_native_execution_unknown(destination, scope, tmp_path):
    settings, request, configuration = facts(tmp_path, scope=scope)
    output = tmp_path / "output"
    if destination == "empty":
        output.mkdir()
    before = snapshot(tmp_path), {str(path) for path in tmp_path.rglob("*")}
    report = _output_report(configuration, request, settings)
    assert report.output_dir == str(output)
    assert report.status == "unverified" and report.exit_code == (0 if scope == "inspection" else 1)
    items = {item.key: item for item in report.items}
    assert items["output.admission"].status == items["output.original_separation"].status == "verified"
    assert items["source.targets"].status == items["native.execution"].status == "unverified"
    assert "output.saved_artifacts" not in items
    assert (snapshot(tmp_path), {str(path) for path in tmp_path.rglob("*")}) == before
    assert config.active_settings() is None


@pytest.mark.parametrize("output, named", [("", True), ("runs/${model}", True), ("fixed", False)])
def test_without_a_model_a_run_dir_named_after_it_is_not_looked_at(output, named, tmp_path, monkeypatch):
    # A check without a model showed the default run directory as results/_.
    settings = RunSettings(base_dir=tmp_path, sections={
        "api": {"model": ""}, "project": {"source_dir": "original", "output_dir": output}}, models={})
    settings, request, configuration = facts(tmp_path, settings=settings)
    if named:
        monkeypatch.setattr(orchestrator, "_admit_output", lambda *a, **k: pytest.fail("run directory admitted"))
    report = _output_report(configuration, request, settings)
    [admission] = [item for item in report.items if item.key == "output.admission"]
    if named:
        assert report.output_dir is None and admission.status == "unverified"
        assert admission.reason == "It is named after the model alias, and no model is selected yet."
        assert not any(item.key == "output.original_separation" for item in report.items)
    else:
        assert report.output_dir == str(tmp_path / "fixed") and admission.status == "verified"


def test_a_run_dir_in_the_projects_testuary_dir_is_named_so(tmp_path):
    """The default run directory lies in the project's Testuary directory,
    inside the project directory. The check accepts it and says where it is,
    not that it lies outside the project directory."""
    settings, request, configuration = facts(
        tmp_path, settings=settings_for(tmp_path, source=".", output=".testuary/results/selected"))
    items = {item.key: item for item in _output_report(configuration, request, settings).items}
    separation = items["output.original_separation"]
    assert separation.status == "verified"
    assert "Testuary directory" in separation.reason and "outside" not in separation.reason
    settings, request, configuration = facts(tmp_path)
    items = {item.key: item for item in _output_report(configuration, request, settings).items}
    assert items["output.original_separation"].reason == "The run directory is outside the source directory."


@pytest.mark.parametrize("mode", ["resume", "starter,coverage,mutation,finishing", "starter", "coverage", "mutation", "suites"])
def test_preserved_output_does_not_certify_saved_work_or_remove_transient_files(mode, tmp_path):
    request = RunRequest(("module.py",), resume=mode == "resume",
                         reuse=() if mode in {"resume", "suites"} else tuple(mode.split(",")))
    settings = settings_for(tmp_path, fields={"starter": {"reuse_suites": mode == "suites"}})
    settings, request, configuration = facts(tmp_path, settings=settings, request=request)
    for relative in ("results/paid.py", "logs/main.jsonl", "_work/owned/transient.txt", "notes.txt"):
        path = tmp_path / "output" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"saved paid work or unrelated data\r\n")
    before = snapshot(tmp_path)
    report = _output_report(configuration, request, settings)
    items = {item.key: item for item in report.items}
    assert items["output.admission"].status == "verified"
    assert items["output.saved_artifacts"].required and items["output.saved_artifacts"].status == "unverified"
    assert items["output.saved_artifacts"].tried_by == "run" and items["output.saved_artifacts"].action
    assert snapshot(tmp_path) == before and config.active_settings() is None


@pytest.mark.parametrize("reuse", [(), ("finishing",)])
@pytest.mark.parametrize("kind", ["result", "partial", "unrelated", "file"])
def test_fresh_occupied_route_refuses_without_removal_or_native_setup(kind, reuse, tmp_path):
    settings, request, configuration = facts(tmp_path, request=RunRequest(("module.py",), reuse=reuse))
    output = tmp_path / "output"
    if kind == "file":
        output.write_bytes(b"not a directory")
    else:
        saved = output / {"result": "results/paid.py", "partial": "logs/main.jsonl", "unrelated": "note.txt"}[kind]
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b"never delete")
    before = snapshot(tmp_path)
    with pytest.raises(ConfigurationError, match="run directory is not empty|not an ordinary directory"):
        _output_report(configuration, request, settings)
    assert snapshot(tmp_path) == before and config.active_settings() is None


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("position", ["root", "ancestor", "traversal", "work", "namespace"])
def test_real_alias_routes_are_refused_in_inspection_without_foreign_writes(
        position, preserve, tmp_path, monkeypatch, directory_alias):
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "paid.txt").write_bytes(b"foreign work")
    output = tmp_path / "output"
    selected = "output"
    if position == "root":
        directory_alias(output, foreign)
    elif position in {"ancestor", "traversal"}:
        directory_alias(tmp_path / "alias", foreign)
        selected = "alias/output" if position == "ancestor" else "alias/../output"
    else:
        output.mkdir()
        if position == "work":
            directory_alias(output / "_work", foreign)
        else:
            (output / "_work").mkdir()
            directory_alias(output / "_work/current", foreign)
    monkeypatch.setattr(project_module, "resolve_work_dir_namespace", lambda: "current")
    settings = settings_for(tmp_path, output=selected)
    request = RunRequest(("module.py",), resume=preserve)
    settings, request, configuration = facts(tmp_path, settings=settings, request=request)
    before = snapshot(foreign)
    with pytest.raises(ConfigurationError, match="ordinary directory|not empty"):
        _output_report(configuration, request, settings)
    assert snapshot(foreign) == before and config.active_settings() is None


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("selected", ["original", "original/output", "ordinary/../original/output"])
def test_admitted_address_inside_original_refuses_and_preserves_original(preserve, selected, tmp_path):
    (tmp_path / "ordinary").mkdir()
    settings = settings_for(tmp_path, output=selected)
    settings, request, configuration = facts(tmp_path, settings=settings, request=RunRequest(("module.py",), resume=preserve))
    before = snapshot(tmp_path)
    with pytest.raises(ConfigurationError, match="inside the source directory|not empty"):
        _output_report(configuration, request, settings)
    assert snapshot(tmp_path) == before and config.active_settings() is None


@pytest.mark.parametrize("source", ["missing", "guest"])
def test_original_separation_remains_unknown_without_host_original_selection(source, tmp_path, monkeypatch):
    settings = settings_for(tmp_path, source="missing" if source == "missing" else "/guest/source",
                            main={"project_exec": "container"} if source == "guest" else None)
    settings, request, configuration = facts(tmp_path, settings=settings)
    if source == "guest":
        monkeypatch.setattr(project_module, "resolve_project_source_path", lambda *a: pytest.fail("host lookup of guest root"))
    report = _output_report(configuration, request, settings)
    items = {item.key: item for item in report.items}
    assert items["output.admission"].status == "verified"
    assert items["output.original_separation"].status == "unverified" and report.status == "unverified"
    # A guest project has no host location to compare, and the run requires none.
    assert items["output.original_separation"].required is (source == "missing")
    assert items["output.original_separation"].action
    assert not (tmp_path / "output").exists() and config.active_settings() is None


def test_ordinary_traversal_returns_the_effective_admitted_address(tmp_path):
    (tmp_path / "ordinary").mkdir()
    settings, request, configuration = facts(tmp_path, settings=settings_for(tmp_path, output="ordinary/../output"))
    assert configuration.output_dir == str(tmp_path / "ordinary/../output")
    report = _output_report(configuration, request, settings)
    assert report.output_dir == str(tmp_path / "output") and not (tmp_path / "output").exists()
    assert sum(item.key == "output.admission" for item in report.items) == 1
    repeated = _output_report(report, request, settings)
    assert repeated == report


@pytest.mark.parametrize("field", ["configuration", "invocation", "settings"])
def test_output_inspection_requires_typed_inputs_before_admission(field, tmp_path):
    settings, request, configuration = facts(tmp_path)
    values = dict(configuration=configuration, invocation=request, settings=settings)
    values[field] = object()
    with pytest.raises(ConfigurationError, match="output inspection requires"):
        _output_report(values["configuration"], values["invocation"], values["settings"])
    assert config.active_settings() is None and not (tmp_path / "output").exists()


def test_actual_public_generation_and_inspection_share_the_containment_rule(tmp_path, monkeypatch):
    project, _, _, settings = harness.setup_case(tmp_path)
    request = RunRequest(("module.py",))
    configuration = _configuration_report(request, settings)
    calls = []
    original = orchestrator._check_output_outside_source

    def observe(output, root):
        calls.append((output, root))
        return original(output, root)

    class AdmissionReached(Exception):
        pass

    def reached(*args):
        raise AdmissionReached

    monkeypatch.setattr(orchestrator, "_check_output_outside_source", observe)
    report = _output_report(configuration, request, settings)
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", reached)
    before = snapshot(tmp_path)
    with pytest.raises(AdmissionReached):
        orchestrator.run_pipeline(request, settings=settings)
    assert calls == [(Path(report.output_dir), project.resolve())] * 2
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("readonly", [False, True])
def test_inspection_preserves_original_file_modes(readonly, tmp_path):
    settings, request, configuration = facts(tmp_path)
    source = tmp_path / "original/module.py"
    mode = source.stat().st_mode
    if readonly:
        source.chmod(mode & ~stat.S_IWUSR)
    try:
        before = snapshot(tmp_path)
        report = _output_report(configuration, request, settings)
        assert next(item for item in report.items if item.key == "output.original_separation").status == "verified"
        assert snapshot(tmp_path) == before
    finally:
        source.chmod(mode)


@pytest.mark.parametrize("error", [PermissionError("inspection denied"), OSError(5, "inspection I/O failed"),
                                   KeyboardInterrupt(), SystemExit(7)])
def test_route_failure_and_cancellation_propagate_exactly_and_restore_settings(error, tmp_path, monkeypatch):
    settings, request, configuration = facts(tmp_path)
    before = snapshot(tmp_path)

    def fail(*a, **kw):
        raise error

    monkeypatch.setattr(orchestrator, "_admit_output", fail)
    with pytest.raises(type(error)) as caught:
        _output_report(configuration, request, settings)
    assert caught.value is error and snapshot(tmp_path) == before
    assert config.active_settings() is None and not (tmp_path / "output").exists()


def test_admitted_address_export_redacts_captured_credentials_without_changing_selection(tmp_path):
    settings, request, configuration = facts(tmp_path, settings=settings_for(tmp_path, output="output-fixture-secret"))
    report = _output_report(configuration, request, settings)
    assert "output-fixture-secret" not in str(report.as_dict())
    with config.use_run_settings(settings):
        assert project_module.output_dir() == tmp_path / "output-fixture-secret"
    assert not (tmp_path / "output-fixture-secret").exists()
