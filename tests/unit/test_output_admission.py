"""Fresh-run refusal and preserved-work cleanup through public entrypoints."""
import os
from dataclasses import replace
from pathlib import Path
import subprocess

import pytest

import test_generation_reuse_interruptions as harness
from testuary import __main__ as cli
from testuary import orchestrator
from testuary import config
from testuary import project as project_module
from testuary.orchestrator import RunRequest
from testuary.settings import ConfigurationError


class AdmissionReached(Exception):
    pass


_PROCESS = subprocess.Popen


@pytest.fixture(autouse=True)
def no_native_processes(monkeypatch):
    import subprocess

    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native process"))
    # The Python vocabulary asks the project's pytest whether it has the
    # timeout plugin, a process of its own. These tests run none: no plugin.
    from langadapter.python.adapter import PythonAdapter
    monkeypatch.setattr(PythonAdapter, "_timeout_mark", lambda self: False)


def snapshot(root):
    return {str(p.relative_to(root)): (p.read_bytes(), p.stat().st_mode)
            for p in root.rglob("*") if p.is_file() and not p.is_symlink()}


@pytest.mark.parametrize("entry", ["api", "cli"])
@pytest.mark.parametrize("kind", ["results", "partial", "unrelated", "file"])
def test_occupied_fresh_output_refuses_before_setup_or_writes(entry, kind, tmp_path, monkeypatch, capsys):
    _, _, _, settings = harness.setup_case(tmp_path)
    output = tmp_path / "output"
    if kind == "file":
        output.write_bytes(b"not a directory")
    else:
        relative = {"results": "results/target/tests/test_paid.py",
                    "partial": "logs/main.jsonl", "unrelated": "notes/keep.txt"}[kind]
        saved = output / relative
        saved.parent.mkdir(parents=True)
        saved.write_bytes(b"paid or unrelated evidence\r\n")
    before = snapshot(tmp_path)
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies",
                        lambda *a: pytest.fail("native setup before output refusal"))
    if entry == "api":
        with pytest.raises(ConfigurationError, match="run directory is not empty|not an ordinary directory"):
            orchestrator.run_pipeline(harness.public_args(), settings=settings)
    else:
        from testuary import settings as settings_module

        monkeypatch.setattr(settings_module, "resolve_settings", lambda **kw: settings)
        monkeypatch.setattr("sys.argv", ["testuary", "--source-files", "module.py", "--display", "plain"])
        with pytest.raises(SystemExit) as caught:
            cli._main()
        assert caught.value.code == 2
        assert "--output-dir" in capsys.readouterr().err
    assert snapshot(tmp_path) == before


@pytest.mark.parametrize("entry", ["api", "cleanup"])
@pytest.mark.parametrize("destination", ["missing", "empty"])
def test_fresh_missing_or_empty_output_is_admitted_without_cleanup(entry, destination, tmp_path, monkeypatch):
    _, _, _, settings = harness.setup_case(tmp_path)
    output = tmp_path / "output"
    if destination == "empty":
        output.mkdir()
    before = snapshot(tmp_path)
    monkeypatch.setattr(orchestrator, "_remove_path", lambda *a: pytest.fail("fresh removal"))
    if entry == "api":
        def reached(*args):
            raise AdmissionReached

        monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
        monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", reached)
        with pytest.raises(AdmissionReached):
            orchestrator.run_pipeline(harness.public_args(), settings=settings)
    else:
        orchestrator._cleanup_outdir(outdir=output, keep_tests=False)
    assert snapshot(tmp_path) == before
    assert output.exists() == (destination == "empty")


@pytest.mark.parametrize("entry", ["api", "cleanup"])
@pytest.mark.parametrize("preserve", [False, True])
def test_output_symlink_is_not_a_cleanup_authority(entry, preserve, tmp_path, monkeypatch):
    _, _, _, settings = harness.setup_case(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    saved = other / "paid.txt"
    saved.write_bytes(b"keep")
    output = tmp_path / "output"
    try:
        output.symlink_to(other, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation unavailable: {exc}")
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    with pytest.raises(ConfigurationError, match="ordinary directory"):
        if entry == "api":
            orchestrator.run_pipeline(harness.public_args("--resume") if preserve else harness.public_args(),
                                      settings=settings)
        else:
            orchestrator._cleanup_outdir(outdir=output, keep_tests=preserve)
    assert saved.read_bytes() == b"keep" and output.is_symlink()


@pytest.mark.parametrize("flag", ["--resume", "--reuse=starter,coverage,mutation,finishing", "--reuse=starter", "--reuse=coverage",
                                 "--reuse=mutation", "--reuse-starter-suites"])
def test_preserved_output_reaches_existing_dependency_preflight(flag, tmp_path, monkeypatch):
    _, _, _, settings = harness.setup_case(
        tmp_path, phase_fields={"reuse_suites": flag == "--reuse-starter-suites"})
    saved = tmp_path / "output/results/target/paid.txt"
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"keep")
    before = snapshot(tmp_path)

    def reached(*args):
        raise AdmissionReached

    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", reached)
    args = harness.public_args() if flag == "--reuse-starter-suites" else harness.public_args(flag)
    with pytest.raises(AdmissionReached):
        orchestrator.run_pipeline(args, settings=settings)
    assert snapshot(tmp_path) == before


def test_preserved_cleanup_only_removes_current_transient_namespace(tmp_path, monkeypatch):
    output = tmp_path / "output"
    paths = ["results/paid.txt", "logs/main.jsonl", "notes/keep.txt", "_work/peer/active.txt"]
    for relative in paths + ["_work/current/transient.txt"]:
        path = output / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(relative.encode())
    monkeypatch.setattr(project_module, "resolve_work_dir_namespace", lambda: "current")
    orchestrator._cleanup_outdir(outdir=output, keep_tests=True)
    assert not (output / "_work/current").exists()
    for relative in paths:
        assert (output / relative).read_bytes() == relative.encode()


@pytest.mark.parametrize("phase", ["starter", "mutation"])
def test_actual_archive_reuse_requests_preserving_cleanup(phase, tmp_path, monkeypatch):
    from testuary.evidence import current_condition, phase_origin_path, write_origin

    project, _, ctx, settings = harness.setup_case(tmp_path)
    target_dir = ctx.results_dir / ctx.workspace
    # A mutation reuse takes the coverage and starter phases with it, and the
    # coverage reuse restores the preserved tests.
    with config.use_run_settings(settings):
        for reused in {"starter": ("starter",), "mutation": ("starter", "coverage", "mutation")}[phase]:
            write_origin(phase_origin_path(target_dir, reused),
                         current_condition(reused, "python", str(project)))
    if phase == "mutation":
        (target_dir / "r0" / "tests").mkdir(parents=True)
    before = snapshot(tmp_path)
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)

    def reached(**kwargs):
        assert kwargs["keep_tests"] is True
        raise AdmissionReached

    monkeypatch.setattr(orchestrator, "_cleanup_outdir", reached)
    with pytest.raises(AdmissionReached):
        orchestrator.run_pipeline(harness.public_args("--reuse", {"starter": "starter"}.get(phase, phase)),
                                  settings=settings)
    assert snapshot(tmp_path) == before


def test_failed_transient_cleanup_stops_public_resume_before_logger_or_phase(tmp_path, monkeypatch):
    _, _, _, settings = harness.setup_case(tmp_path)
    saved = tmp_path / "output/results/paid.txt"
    saved.parent.mkdir(parents=True)
    saved.write_bytes(b"keep")
    transient = tmp_path / "output/_work/transient.txt"
    transient.parent.mkdir(parents=True)
    transient.write_bytes(b"still owned")
    before = snapshot(tmp_path)
    failure = PermissionError("transient cleanup refused")
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_preflight_generation_reuse", lambda *a: None)
    monkeypatch.setattr(project_module, "resolve_work_dir_namespace", lambda: "")

    def refuse(path, **kwargs):
        assert Path(path) == transient.parent
        assert not kwargs.get("ignore_errors", False)
        raise failure

    monkeypatch.setattr(orchestrator.shutil, "rmtree", refuse)
    monkeypatch.setattr(orchestrator.AgentLogger, "init", lambda *a: pytest.fail("logger before cleanup"))
    with pytest.raises(PermissionError) as caught:
        orchestrator.run_pipeline(harness.public_args("--resume"), settings=settings)
    assert caught.value is failure
    assert snapshot(tmp_path) == before


def test_direct_fresh_cleanup_cannot_delete_a_prior_run(tmp_path, monkeypatch):
    saved = tmp_path / "results/paid.txt"
    saved.parent.mkdir()
    saved.write_bytes(b"keep")
    before = snapshot(tmp_path)
    monkeypatch.setattr(orchestrator, "_remove_path", lambda *a: pytest.fail("fresh removal"))
    with pytest.raises(ConfigurationError, match="not empty"):
        orchestrator._cleanup_outdir(outdir=tmp_path, keep_tests=False)
    assert snapshot(tmp_path) == before


def test_output_is_checked_again_after_preflight(tmp_path, monkeypatch):
    _, _, _, settings = harness.setup_case(tmp_path)
    saved = tmp_path / "output/appeared.txt"
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())

    def occupy(*args):
        saved.parent.mkdir()
        saved.write_bytes(b"another writer")

    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", occupy)
    monkeypatch.setattr(orchestrator.AgentLogger, "init", lambda *a: pytest.fail("logger before second gate"))
    monkeypatch.setattr(orchestrator, "_remove_path", lambda *a: pytest.fail("fresh removal"))
    with pytest.raises(ConfigurationError, match="not empty"):
        orchestrator.run_pipeline(harness.public_args(), settings=settings)
    assert saved.read_bytes() == b"another writer"


@pytest.fixture(params=["junction"] if os.name == "nt" else ["symlink"])
def directory_alias(request, monkeypatch):
    """Windows junctions do not require symlink privilege or developer mode."""
    links = []

    def create(link, target):
        if request.param == "junction":
            command = Path(os.environ["SystemRoot"]) / "System32/cmd.exe"
            # Permit only fixture creation; native measurement remains forbidden.
            with monkeypatch.context() as setup:
                setup.setattr(subprocess, "Popen", _PROCESS)
                # cmd answers in the console code page, not the locale's ANSI
                # code page, so its output stays undecoded bytes.
                result = subprocess.run([str(command), "/c", "mklink", "/J", str(link), str(target)],
                                        capture_output=True, check=False)
            assert result.returncode == 0, (result.stdout + result.stderr).decode(errors="replace")
            assert link.lstat().st_file_attributes & 0x400
        else:
            link.symlink_to(target, target_is_directory=True)
        links.append(link)

    yield create
    for link in reversed(links):
        if link.is_symlink():
            link.unlink()
        else:
            os.rmdir(link)


@pytest.mark.parametrize("entry", ["api", "cli", "cleanup"])
@pytest.mark.parametrize("position", ["root", "work", "namespace", "ancestor"])
@pytest.mark.parametrize("flags", [["--resume"], ["--reuse", "starter"]])
def test_actual_preserved_output_refuses_alias_routes_before_removal(
        entry, position, flags, tmp_path, monkeypatch, directory_alias, capsys):
    _, _, _, settings = harness.setup_case(tmp_path)
    output = tmp_path / "output"
    foreign = tmp_path / "foreign"
    for relative in ("paid.txt", "current/paid.txt", "output/_work/current/paid.txt"):
        saved = foreign / relative
        saved.parent.mkdir(parents=True, exist_ok=True)
        saved.write_bytes(b"foreign evidence")
    if position == "root":
        directory_alias(output, foreign)
    elif position == "ancestor":
        directory_alias(tmp_path / "alias", foreign)
        output = tmp_path / "alias/output"
        from dataclasses import replace
        settings = replace(settings, sections={**settings.sections, "project": {
            **settings.sections["project"], "output_dir": str(output),
        }})
    else:
        output.mkdir()
        work = output / "_work"
        if position == "work":
            directory_alias(work, foreign)
        else:
            work.mkdir()
            directory_alias(work / "current", foreign)
    before = snapshot(foreign)
    monkeypatch.setattr(project_module, "resolve_work_dir_namespace", lambda: "current")
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies",
                        lambda *a: pytest.fail("native setup before alias refusal"))
    monkeypatch.setattr(orchestrator, "_remove_path", lambda *a: pytest.fail("alias removal"))
    if entry == "cli":
        from testuary import settings as settings_module
        monkeypatch.setattr(settings_module, "resolve_settings", lambda **kw: settings)
        monkeypatch.setattr("sys.argv", ["testuary", "--source-files", "module.py",
                                         "--display", "plain", *flags])
        with pytest.raises(SystemExit) as caught:
            cli._main()
        assert caught.value.code == 2
        assert "--output-dir" in capsys.readouterr().err
    else:
        with pytest.raises(ConfigurationError, match="ordinary directory"):
            if entry == "api":
                orchestrator.run_pipeline(harness.public_args(*flags), settings=settings)
            else:
                orchestrator._cleanup_outdir(outdir=output, keep_tests=True)
    assert snapshot(foreign) == before


def test_transient_alias_appearing_after_admission_refuses_before_cleanup(
        tmp_path, monkeypatch, directory_alias):
    _, _, _, settings = harness.setup_case(tmp_path)
    output = tmp_path / "output"
    output.mkdir()
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    saved = foreign / "paid.txt"
    saved.write_bytes(b"keep foreign")
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_generation_reuse", lambda *a: None)
    monkeypatch.setattr(project_module, "resolve_work_dir_namespace", lambda: "")
    monkeypatch.setattr(orchestrator.AgentLogger, "init", lambda *a: pytest.fail("logger before refusal"))
    monkeypatch.setattr(orchestrator, "_remove_path", lambda *a: pytest.fail("foreign removal"))
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies",
                        lambda *a: directory_alias(output / "_work", foreign))
    with pytest.raises(ConfigurationError, match="ordinary directory"):
        orchestrator.run_pipeline(harness.public_args("--resume"), settings=settings)
    assert saved.read_bytes() == b"keep foreign"


@pytest.mark.parametrize("relative", [False, True])
def test_output_selector_preserves_parent_traversal_without_inspection(relative, tmp_path, monkeypatch):
    _, _, _, settings = harness.setup_case(tmp_path)
    selected = Path("ordinary/../output")
    configured = selected if relative else tmp_path / selected
    settings = replace(settings, sections={**settings.sections, "project": {
        **settings.sections["project"], "output_dir": str(configured),
    }})
    monkeypatch.setattr(project_module, "_check_directory_route", lambda *a, **kw: pytest.fail("configuration inspected output"))
    with config.use_run_settings(settings):
        assert project_module.output_dir() == tmp_path / selected


@pytest.mark.parametrize("entry", ["typed", "parsed", "cli", "cleanup"])
@pytest.mark.parametrize("relative", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_selected_alias_before_parent_traversal_refuses_without_removal(
        entry, relative, preserve, tmp_path, monkeypatch, directory_alias, capsys):
    _, _, _, settings = harness.setup_case(tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "paid.txt").write_bytes(b"foreign evidence")
    directory_alias(tmp_path / "alias", foreign)
    selected = Path("alias/../output")
    configured = selected if relative else tmp_path / selected
    settings = replace(settings, sections={**settings.sections, "project": {
        **settings.sections["project"], "output_dir": str(configured),
    }})
    output = tmp_path / "output"
    output.mkdir()
    if preserve:
        (output / "retained.txt").write_bytes(b"retained evidence")
    before = snapshot(foreign), snapshot(output)
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: pytest.fail("native setup before refusal"))
    monkeypatch.setattr(orchestrator, "_remove_path", lambda *a: pytest.fail("alias removal"))
    flags = ["--resume"] if preserve else []
    if entry == "cli":
        from testuary import settings as settings_module

        real_resolve = settings_module.resolve_settings
        monkeypatch.setattr(settings_module, "resolve_settings", lambda **kw: real_resolve(
            **kw, base_dir=tmp_path, use_environment=False, models=settings.models))
        # A relative --output-dir is taken from the working directory.
        monkeypatch.chdir(tmp_path)
        monkeypatch.setattr("sys.argv", ["testuary", "generate", "--source-files", "module.py",
            "--project-name", "subject", "--output-dir", str(configured), "--display", "plain", *flags])
        with pytest.raises(SystemExit) as stopped:
            cli._main()
        assert stopped.value.code == 2
        assert "ordinary directory" in capsys.readouterr().err
    else:
        with pytest.raises(ConfigurationError, match="ordinary directory"):
            if entry == "cleanup":
                orchestrator._cleanup_outdir(outdir=tmp_path / selected, keep_tests=preserve)
            else:
                request = RunRequest(["module.py"], resume=preserve) if entry == "typed" else harness.public_args(*flags)
                orchestrator.run_pipeline(request, settings=settings)
    assert (snapshot(foreign), snapshot(output)) == before


@pytest.mark.parametrize("entry", ["typed", "parsed", "cli"])
@pytest.mark.parametrize("contained", [False, True])
@pytest.mark.parametrize("preserve", [False, True])
def test_admitted_parent_traversal_uses_effective_address_for_containment_and_cleanup(
        entry, contained, preserve, tmp_path, monkeypatch, capsys):
    project, _, _, settings = harness.setup_case(tmp_path)
    output = project / "output" if contained else tmp_path / "output"
    selected = tmp_path / "ordinary" / ".." / output.relative_to(tmp_path)
    (tmp_path / "ordinary").mkdir()
    output.mkdir()
    settings = replace(settings, sections={**settings.sections, "project": {
        **settings.sections["project"], "output_dir": str(selected),
    }})
    if preserve:
        (output / "retained.txt").write_bytes(b"retained evidence")
    before = snapshot(project), snapshot(output)
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_generation_reuse", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)

    def cleanup(**kwargs):
        assert not contained
        assert kwargs == {"outdir": output, "keep_tests": preserve}
        raise AdmissionReached

    monkeypatch.setattr(orchestrator, "_cleanup_outdir", cleanup)
    request = RunRequest(["module.py"], resume=preserve) if entry == "typed" else harness.public_args(*(["--resume"] if preserve else []))
    if entry == "cli":
        from testuary import settings as settings_module

        monkeypatch.setattr(settings_module, "resolve_settings", lambda **kw: settings)
        monkeypatch.setattr("sys.argv", ["testuary", "generate", "--source-files", "module.py",
            "--display", "plain", *(["--resume"] if preserve else [])])
        call = cli._main
    else:
        def call():
            return orchestrator.run_pipeline(request, settings=settings)
    expected = (SystemExit if entry == "cli" else ConfigurationError) if contained else AdmissionReached
    with pytest.raises(expected) as stopped:
        call()
    if contained and entry == "cli":
        assert stopped.value.code == 2
        assert "inside the source directory" in capsys.readouterr().err
    assert (snapshot(project), snapshot(output)) == before


@pytest.mark.parametrize("preserve", [False, True])
def test_direct_cleanup_collapses_ordinary_parent_traversal_only_after_admission(preserve, tmp_path, monkeypatch):
    selected = tmp_path / "ordinary/../output"
    (tmp_path / "ordinary").mkdir()
    output = tmp_path / "output"
    output.mkdir()
    removed = []
    monkeypatch.setattr(project_module, "resolve_work_dir_namespace", lambda: "current")
    monkeypatch.setattr(orchestrator, "_remove_path", removed.append)
    orchestrator._cleanup_outdir(outdir=selected, keep_tests=preserve)
    assert removed == ([output / "_work/current"] if preserve else [])


@pytest.mark.parametrize("entry", ["typed", "cli"])
def test_actual_run_uses_one_admitted_address_for_work_results_and_logs(entry, tmp_path, monkeypatch):
    from testuary import target_phases as phases

    project, _, _, settings = harness.setup_case(tmp_path)
    selected = tmp_path / "ordinary/../output"
    (tmp_path / "ordinary").mkdir()
    output = tmp_path / "output"
    settings = replace(settings, sections={**settings.sections, "project": {
        **settings.sections["project"], "output_dir": str(selected),
    }})
    before = snapshot(project)
    called = []

    def run_phase(ctx):
        assert ctx.results_dir == output / "results"
        for path in (ctx.work_tests.parent, ctx.target_logs_dir, ctx.results_dir):
            assert path.is_relative_to(output) and ".." not in path.parts
        called.append(ctx)
        return phases.TestPipelineResult(stopped_after_phase="coverage")

    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_check_canonical_against_git", lambda *a: [])
    monkeypatch.setattr(phases, "run_target_phases", run_phase)
    if entry == "cli":
        from testuary import settings as settings_module

        monkeypatch.setattr(settings_module, "resolve_settings", lambda **kw: settings)
        monkeypatch.setattr("sys.argv", ["testuary", "generate", "--source-files", "module.py",
            "--stop-after-phase", "coverage", "--display", "plain"])
        assert cli._main() == 0
    else:
        assert orchestrator.run_pipeline(RunRequest(["module.py"], stop_after_phase="coverage"), settings=settings) == 0
    assert len(called) == 1 and (output / "logs/main.jsonl").is_file()
    assert snapshot(project) == before
    assert config.active_settings() is None and orchestrator.get_main_logger() is None
