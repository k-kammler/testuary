"""Explicit roots stay distinct from catalogue labels and private copies."""
from dataclasses import replace
from pathlib import Path, PurePosixPath
import subprocess
import sys

import pytest

import test_generation_reuse_interruptions as harness
from test_output_admission import directory_alias  # noqa: F401 - shared platform fixture
from testuary import __main__ as cli, orchestrator, settings as settings_module
from testuary import config as config_module
from testuary import project as project_module
from testuary.orchestrator import RunRequest
from testuary import target_phases as phases
from testuary.settings import ConfigurationError, RunSettings, resolve_settings


@pytest.fixture(autouse=True)
def no_native_or_provider(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native process"))
    monkeypatch.setattr(orchestrator, "_preflight_model_configuration", lambda *a: ())
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_check_canonical_against_git", lambda *a: [])
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)


@pytest.mark.parametrize("selection", ["explicit", "label", "base"])
def test_root_label_and_work_override_are_independent(selection, tmp_path, monkeypatch):
    other = tmp_path / "other"
    other.mkdir()
    project = {"name": "label", "source_dir": "actual source", "python_import_root": "src"} \
        if selection == "explicit" else ({"name": "label"} if selection == "label" else {})
    settings = RunSettings(base_dir=tmp_path, sections={"project": project})
    # A label alone names the project; the project directory stays its root.
    expected = tmp_path / "actual source" if selection == "explicit" else tmp_path
    monkeypatch.chdir(other)
    with config_module.use_run_settings(settings):
        resolved = config_module.get_project_config()
        assert resolved["source_dir"] == resolved["target_repo"] == str(expected)
        assert resolved["source_dir_override"] == (str(expected) if selection != "base" else "")
        assert resolved["project_name"] == ("label" if selection != "base" else tmp_path.name)
        if selection == "explicit":
            assert resolved["python_import_root"] == "src"
        config_module.set_invocation_source_dir(str(tmp_path / "private"))
        redirected = config_module.get_project_config()
        assert redirected["source_dir_override"] == str(tmp_path / "private")
        assert redirected["source_dir"] == redirected["target_repo"] == str(expected)


def test_source_dir_layers_through_existing_resolver_and_environment(tmp_path):
    overlay = tmp_path / "config.ini"
    overlay.write_text("[project]\nname = profile\nsource_dir = configured\n", encoding="utf-8")
    for overrides, expected in (({}, "environment"), ({"project": {"source_dir": "explicit"}}, "explicit")):
        settings = resolve_settings(config_files=[overlay], overrides=overrides, models={},
                                    base_dir=tmp_path, environment={"TESTUARY_PROJECT_SOURCE_DIR": "environment"})
        with config_module.use_run_settings(settings):
            assert config_module.get_project_config()["source_dir"] == str(tmp_path / expected)
    settings = resolve_settings(config_files=[overlay], use_environment=False, models={}, base_dir=tmp_path)
    with config_module.use_run_settings(settings):
        assert config_module.get_project_config()["source_dir"] == str(tmp_path / "configured")


@pytest.mark.parametrize("explicit", [False, True])
def test_guest_root_selection_never_probes_host_paths(explicit, tmp_path, monkeypatch):
    settings = RunSettings(base_dir=tmp_path, sections={
        "main": {"project_exec": "container"},
        "project": {"name": "profile", "source_dir": "/chosen source" if explicit else ""},
    }, environment={"TESTUARY_PROJECT_SOURCE_DIR": "/captured"})
    with config_module.use_run_settings(settings):
        monkeypatch.setattr(Path, "resolve", lambda *a, **k: pytest.fail("host canonicalization"))
        monkeypatch.setattr(Path, "is_dir", lambda *a: pytest.fail("host existence probe"))
        monkeypatch.setattr(Path, "lstat", lambda *a: pytest.fail("host path admission"))
        config = config_module.get_project_config()
        expected = "/chosen source" if explicit else "/captured"
        assert config["source_dir"] == config["target_repo"] == config["source_dir_override"] == expected
        assert PurePosixPath(config["source_dir"]).is_absolute()


@pytest.mark.parametrize("root", ["relative", "C:\\host\\source"])
def test_guest_root_refuses_host_or_relative_address(root, tmp_path):
    settings = RunSettings(base_dir=tmp_path, sections={"main": {"project_exec": "container"},
                                                      "project": {"source_dir": root}})
    with config_module.use_run_settings(settings), pytest.raises(ConfigurationError, match="absolute POSIX"):
        config_module.get_project_config()


@pytest.mark.parametrize("language", ["python", "java"])
@pytest.mark.parametrize("explicit", [False, True])
def test_actual_guest_context_and_framework_use_the_same_selected_original(
        language, explicit, tmp_path, monkeypatch):
    from testuary.languages import language_project_context

    settings = RunSettings(base_dir=tmp_path, sections={
        "main": {"project_exec": "container"},
        "project": {"source_dir": "/selected" if explicit else "", "python_import_root": "src",
                    "python_executable": "/guest/python", "java_home": "/guest/jdk"},
    }, environment={"TESTUARY_PROJECT_SOURCE_DIR": "/captured", "TESTUARY_PROJECT_CONTAINER": "fixture",
                    "TESTUARY_PROJECT_WORK_DIR": str(tmp_path / "workzone")})
    monkeypatch.setenv("TESTUARY_PROJECT_SOURCE_DIR", "/later")
    with config_module.use_run_settings(settings):
        config_module.set_invocation_source_dir("/private/other")
        expected = "/selected" if explicit else "/captured"
        assert project_module.project_repo_dir() == expected
        assert project_module.resolve_source_dir(expected + "/src/module.py") == expected
        assert project_module._below(expected, "src") == expected + "/src"
        assert project_module.display_path(expected + "/src/module.py", expected) == "module.py"
        context = language_project_context(language)
        assert context.root == context.execution.repository_dir == expected


@pytest.mark.parametrize("entry", ["typed", "parsed", "cli"])
@pytest.mark.parametrize("label", ["", "subject"])
@pytest.mark.parametrize("outcome", ["success", "failure", "cancelled"])
def test_actual_arbitrary_root_runs_preserve_selection_and_original(
        entry, label, outcome, tmp_path, monkeypatch):
    project, target, _, original = harness.setup_case(tmp_path)
    project.rename(tmp_path / "arbitrary source")
    project, target = tmp_path / "arbitrary source", tmp_path / "arbitrary source/module.py"
    before = {path: (path.read_bytes(), path.stat().st_mode) for path in project.rglob("*") if path.is_file()}
    configured = replace(original, sections={**original.sections, "project": {
        **original.sections["project"], "name": label, "source_dir": "arbitrary source",
    }})
    other = tmp_path / "unrelated working directory"
    other.mkdir()
    monkeypatch.chdir(other)
    called, cancel = [], KeyboardInterrupt()

    def run(ctx):
        captured = config_module.get_project_config()
        assert ctx.canonical_path == target
        assert Path(ctx.top_file["file_path"]).read_bytes() == before[target][0]
        assert captured["source_dir"] == captured["target_repo"] == str(project)
        assert captured["source_dir_override"] != str(project)
        assert captured["project_name"] == (label or "arbitrary_source")
        called.append(ctx)
        if outcome == "cancelled":
            raise cancel
        if outcome == "failure":
            raise RuntimeError("controlled phase failure")
        return phases.TestPipelineResult(stopped_after_phase="coverage")

    monkeypatch.setattr(phases, "run_target_phases", run)
    monkeypatch.setattr(settings_module, "resolve_settings", lambda **kw: configured)
    request = RunRequest(["module.py"], stop_after_phase="coverage")
    args = harness.public_args("--stop-after-phase", "coverage")
    monkeypatch.setattr(sys, "argv", ["testuary", "generate", "--source-files", "module.py",
                                     "--stop-after-phase", "coverage", "--display", "plain"])
    call = (lambda: cli._main()) if entry == "cli" else (
        lambda: orchestrator.run_pipeline(request if entry == "typed" else args, settings=configured)
    )
    if outcome == "cancelled":
        with pytest.raises(KeyboardInterrupt) as caught:
            call()
        assert caught.value is cancel
    else:
        assert call() == (1 if outcome == "failure" else 0)
    assert len(called) == 1 and config_module.active_settings() is None
    assert {path: (path.read_bytes(), path.stat().st_mode) for path in before} == before


def test_derived_label_is_used_by_output_placeholders(tmp_path):
    settings = RunSettings(base_dir=tmp_path, sections={"project": {
        "source_dir": "project with spaces", "output_dir": "results/${model}/${project}"},
        "api": {"model": "example"}})
    with config_module.use_run_settings(settings):
        assert project_module.output_dir() == tmp_path / "results/example/project_with_spaces"


@pytest.mark.parametrize("entry", ["typed", "parsed"])
@pytest.mark.parametrize("reuse", [False, True])
@pytest.mark.parametrize("state", ["missing", "empty", "occupied"])
def test_in_source_output_refuses_before_any_setup_or_cleanup(entry, reuse, state, tmp_path, monkeypatch):
    project, _, _, original = harness.setup_case(tmp_path)
    saved = project / "output/paid.txt"
    if state != "missing":
        saved.parent.mkdir()
    if state == "occupied":
        saved.write_bytes(b"retained")
    settings = replace(original, sections={**original.sections, "project": {
        "source_dir": str(project), "output_dir": str(saved.parent),
    }})
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kw: pytest.fail("cleanup"))
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: pytest.fail("native setup"))
    invocation = RunRequest(["module.py"], resume=reuse) if entry == "typed" else harness.public_args(*(["--resume"] if reuse else []))
    with pytest.raises(ConfigurationError, match="not empty|inside the source directory"):
        orchestrator.run_pipeline(invocation, settings=settings)
    assert saved.exists() == (state == "occupied")
    assert saved.parent.exists() == (state != "missing")
    if state == "occupied":
        assert saved.read_bytes() == b"retained"


def test_parser_runtime_fields_map_only_existing_setting_keys():
    args = cli._build_parser().parse_args([
        "--source-dir", "source", "--python-executable", "venv/python",
        "--java-home", "project-jdk", "--build-java-home", "build-jdk",
    ])
    assert orchestrator._cli_settings_overrides(args) == {"project": {
        "source_dir": "source", "python_executable": "venv/python",
        "java_home": "project-jdk", "build_java_home": "build-jdk",
    }}


@pytest.mark.parametrize("label", ["..", ".", "../other", "nested/project", "nested\\project", "C:other", "bad\0label"])
def test_project_label_cannot_redirect_the_private_workspace(label, tmp_path, monkeypatch):
    settings = RunSettings(base_dir=tmp_path, sections={"project": {"name": label, "source_dir": "source"}})
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: pytest.fail("native setup"))
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kw: pytest.fail("cleanup"))
    with pytest.raises(ConfigurationError, match=r"\[project\] name must be a single directory name"):
        orchestrator.run_pipeline(RunRequest(["module.py"]), settings=settings)
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("entry", ["typed", "parsed", "cli"])
def test_actual_pipeline_defaults_to_captured_base_not_changed_cwd(entry, tmp_path, monkeypatch):
    project, target, _, original = harness.setup_case(tmp_path)
    settings = replace(original, base_dir=project, sections={**original.sections, "project": {
        "output_dir": str(tmp_path / "output"),
    }})
    other = tmp_path / "other"
    other.mkdir()
    monkeypatch.chdir(other)
    seen = []

    def run(ctx):
        assert ctx.canonical_path == target
        assert config_module.get_project_config()["source_dir"] == str(project)
        assert Path(ctx.top_file["file_path"]).read_bytes() == target.read_bytes()
        seen.append(ctx.project_name)
        return phases.TestPipelineResult(stopped_after_phase="coverage")

    monkeypatch.setattr(phases, "run_target_phases", run)
    if entry == "cli":
        monkeypatch.setattr(settings_module, "resolve_settings", lambda **kw: settings)
        monkeypatch.setattr(sys, "argv", ["testuary", "generate", "--source-files", "module.py",
                                         "--stop-after-phase", "coverage", "--display", "plain"])
        result = cli._main()
    else:
        result = orchestrator.run_pipeline(
            RunRequest(["module.py"], stop_after_phase="coverage") if entry == "typed" else
            harness.public_args("--stop-after-phase", "coverage"), settings=settings,
        )
    assert result == 0 and seen == ["subject"]


def test_actual_cli_runtime_selection_overrides_files_and_environment(tmp_path, monkeypatch):
    overlay = tmp_path / "settings.ini"
    overlay.write_text("[project]\nsource_dir = file-source\npython_executable = file-python\n"
                       "java_home = file-jdk\nbuild_java_home = file-build-jdk\n", encoding="utf-8")
    monkeypatch.setenv("TESTUARY_PROJECT_SOURCE_DIR", "environment-source")
    monkeypatch.setenv("TESTUARY_PROJECT_PYTHON_EXECUTABLE", "environment-python")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "argv", ["testuary", "generate", "--config", str(overlay),
                                     "--source-files", "module.py", "--source-dir", "cli-source",
                                     "--python-executable", "cli-python", "--java-home", "cli-jdk",
                                     "--build-java-home", "cli-build-jdk", "--display", "plain"])
    observed = []

    def run(args, selected_languages):
        config = config_module.get_project_config()
        observed.append({key: config[key] for key in ("source_dir", "python_executable", "java_home", "build_java_home")})
        return 0

    monkeypatch.setattr(orchestrator, "_run_pipeline", run)
    assert cli._main() == 0
    assert observed == [{key: str(tmp_path / value) for key, value in {
        "source_dir": "cli-source", "python_executable": "cli-python", "java_home": "cli-jdk",
        "build_java_home": "cli-build-jdk",
    }.items()}]
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("entry", ["typed", "parsed", "cli"])
@pytest.mark.parametrize("position", ["root", "ancestor", "dotdot"])
def test_actual_source_alias_is_refused_before_canonicalization_or_setup(
        entry, position, directory_alias, tmp_path, monkeypatch, capsys):  # noqa: F811 - pytest fixture
    _, _, _, original = harness.setup_case(tmp_path)
    foreign = tmp_path / "foreign"
    foreign.mkdir()
    (foreign / "module.py").write_bytes(b"foreign source")
    link = tmp_path / "alias"
    directory_alias(link, foreign)
    selected = link if position == "root" else (
        link / "child" if position == "ancestor" else link / ".." / "foreign"
    )
    settings = replace(original, sections={**original.sections, "project": {
        **original.sections["project"], "source_dir": str(selected),
    }})
    before = {path: (path.read_bytes(), path.stat().st_mode) for path in foreign.rglob("*") if path.is_file()}
    resolve = Path.resolve

    def do_not_adopt(path, *args, **kwargs):
        if path == selected:
            pytest.fail("selected source canonicalized before ordinary-path admission")
        return resolve(path, *args, **kwargs)

    monkeypatch.setattr(Path, "resolve", do_not_adopt)
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: pytest.fail("native setup"))
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kw: pytest.fail("cleanup"))
    if entry == "cli":
        monkeypatch.setattr(settings_module, "resolve_settings", lambda **kw: settings)
        monkeypatch.setattr(sys, "argv", ["testuary", "generate", "--source-files", "module.py", "--display", "plain"])
        with pytest.raises(SystemExit) as stopped:
            cli._main()
        assert stopped.value.code == 2 and "--source-dir" in capsys.readouterr().err
    else:
        with pytest.raises(ConfigurationError, match="source route.*ordinary directory"):
            orchestrator.run_pipeline(RunRequest(["module.py"]) if entry == "typed" else harness.public_args(), settings=settings)
    assert {path: (path.read_bytes(), path.stat().st_mode) for path in before} == before
    assert not (tmp_path / "output").exists()
