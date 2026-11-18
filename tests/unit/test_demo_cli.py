"""``testuary --demo`` without running the pipeline: labeling, the settings
it overrides, its refusals, and that a normal run never loads it."""
from __future__ import annotations

import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

from tool_fixtures import same_code_environment

from testuary import __main__ as cli
from testuary.demo import cli as demo_cli
from testuary.demo import cli as labels


def test_records_and_tables_are_labeled_and_hashed_files_left_alone(tmp_path):
    record = tmp_path / "results" / "archive" / "record.json"
    record.parent.mkdir(parents=True)
    record.write_text(json.dumps({"delivered_from": "reduced"}, indent=2), encoding="utf-8")
    log = tmp_path / "logs" / "usage.jsonl"
    log.parent.mkdir()
    log.write_text('{"a":1}\n{"b":2}\n', encoding="utf-8")
    table = tmp_path / "results" / "tests.csv"
    table.write_bytes("\ufeffrank,score\n1,99.5\n".encode("utf-8"))
    suite = tmp_path / "results" / "tests" / "test_x.py"
    suite.parent.mkdir()
    suite.write_bytes(b"def test_a():\n    pass\n")
    reply = tmp_path / "results" / "archive" / "response.txt"
    reply.write_bytes(b"```python\nx = 1\n```")

    outcome = labels.label_outputs(tmp_path)

    labeled = json.loads(record.read_text(encoding="utf-8"))
    assert list(labeled)[0] == labels.LABEL_FIELD and labeled["delivered_from"] == "reduced"
    lines = [json.loads(line) for line in log.read_text(encoding="utf-8").splitlines()]
    assert all(line[labels.LABEL_FIELD] == labels.RECORDED_LABEL for line in lines)
    raw = table.read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    header, row = raw.decode("utf-8-sig").splitlines()
    assert header.endswith(",demo") and row.endswith(",demo")
    assert suite.read_bytes() == b"def test_a():\n    pass\n"
    assert reply.read_bytes() == b"```python\nx = 1\n```"
    notice = (suite.parent / labels.NOTICE_NAME).read_text(encoding="utf-8")
    assert "DEMO, not a measurement" in notice and "test_x.py" in notice
    assert "response.txt" in (reply.parent / labels.NOTICE_NAME).read_text(
        encoding="utf-8")
    assert outcome["counts"] == {"json": 1, "jsonl": 1, "tables": 1, "notices": 2}


def test_every_output_is_labeled_or_named_in_a_notice(tmp_path):
    """A file whose format takes no label, such as a Toy source or a log,
    and a JSON-lines file with a line that is no record are left as written
    and named in their directory's notice."""
    (tmp_path / "calc.toy").write_text("check 1\n", encoding="utf-8")
    (tmp_path / "run.log").write_text("started\n", encoding="utf-8")
    broken = tmp_path / "events.jsonl"
    broken.write_text('{"a": 1}\nnot a record\n', encoding="utf-8")
    labels.label_outputs(tmp_path)
    notice = (tmp_path / labels.NOTICE_NAME).read_text(encoding="utf-8")
    assert all(name in notice for name in ("calc.toy", "run.log", "events.jsonl"))
    assert broken.read_text(encoding="utf-8") == '{"a": 1}\nnot a record\n'


def test_an_empty_event_file_of_the_run_leaves_the_run_dir_unused(tmp_path):
    """The command line's write check of --events leaves an empty file, which
    the run replaces, so it does not make the run directory used."""
    events = tmp_path / "events.jsonl"
    events.write_text("", encoding="utf-8")
    assert demo_cli._unused_run_dir(tmp_path, str(events))
    assert not demo_cli._unused_run_dir(tmp_path, None)
    events.write_text("{}\n", encoding="utf-8")
    assert not demo_cli._unused_run_dir(tmp_path, str(events))


def test_every_setting_a_normal_run_would_read_is_named(tmp_path, monkeypatch):
    from testuary import settings

    # The test's directory is the user directory of the run, with its config.ini.
    monkeypatch.setattr(settings, "platform_user_dir", lambda: tmp_path)
    parser = cli._build_parser()
    args = parser.parse_args(["--demo", "--model", "other", "--output-dir", "x"])
    (tmp_path / "config.ini").write_text("[main]\n", encoding="utf-8")
    environ = {"TESTUARY_STARTER_SUITE_COUNT": "3",
               "LANGADAPTER_MUTATION_TARGET": "/elsewhere/mod.py",
               "LANGCHAIN_TRACING_V2": "true", "HOME": "/home/x"}
    items = demo_cli.overridden_settings(args, parser, environ, tmp_path,
                                         "openai_deepseek")
    settings = [item["setting"] for item in items]
    assert "--model other" in settings
    assert "environment variable TESTUARY_STARTER_SUITE_COUNT" in settings
    assert "environment variable LANGADAPTER_MUTATION_TARGET" in settings
    assert "environment variable LANGCHAIN_TRACING_V2" in settings
    assert str(tmp_path / "config.ini") in settings
    assert not any("HOME" in s or "output-dir" in s for s in settings)
    model = next(item for item in items if item["setting"] == "--model other")
    assert "openai_deepseek" in model["instead"]


def test_a_setting_not_used_is_named_as_the_user_typed_it(tmp_path, monkeypatch):
    """An option whose stored name differs from its spelling is listed under
    the option itself, alone for a flag, and a file named on the command
    line under FILE with the file as typed, so every listed name is one the
    command accepts or shows."""
    from testuary import settings

    monkeypatch.setattr(settings, "platform_user_dir", lambda: tmp_path)
    parser = cli._build_parser()
    args = parser.parse_args(["--demo", "--reuse-starter-suites", "src/app.py"])
    items = demo_cli.overridden_settings(args, parser, {}, tmp_path, "openai_deepseek")
    settings = [item["setting"] for item in items]
    assert "--reuse-starter-suites" in settings
    assert "FILE src/app.py" in settings
    shown = {option for action in parser._actions for option in action.option_strings} | {"FILE"}
    assert all(setting.split()[0] in shown for setting in settings
               if not setting.startswith(("environment variable", "hints of the installed package",
                                          str(tmp_path))))


def test_a_reuse_selection_is_listed_as_its_option_alone(tmp_path, monkeypatch):
    """The run-option check sets each reused phase's flag before the
    demonstration starts. The list names --reuse as typed and no flag of a
    phase, since none is an option."""
    from testuary import settings

    monkeypatch.setattr(settings, "platform_user_dir", lambda: tmp_path)
    parser = cli._build_parser()
    args = parser.parse_args(["--demo", "--reuse", "starter"])
    cli._check_run_options(args)
    items = demo_cli.overridden_settings(args, parser, {}, tmp_path, "openai_deepseek")
    settings = [item["setting"] for item in items]
    assert "--reuse starter" in settings
    assert not [setting for setting in settings if setting.startswith("--reuse-")]


def test_the_files_listed_follow_home_and_no_project_settings(tmp_path, monkeypatch):
    """The demonstration names the files a normal run with these options
    reads: the user directory --home names, and no project settings file."""
    from testuary import settings

    monkeypatch.setattr(settings, "platform_user_dir", lambda: tmp_path / "user")
    home = tmp_path / "home"
    home.mkdir()
    (home / "config.ini").write_text("", encoding="utf-8")
    project = tmp_path / "project"
    (project / ".testuary").mkdir(parents=True)
    (project / ".testuary" / "project.ini").write_text("", encoding="utf-8")
    parser = cli._build_parser()
    args = parser.parse_args(["--demo", "--home", str(home), "--no-project-settings"])
    listed = [item["setting"] for item in
              demo_cli.overridden_settings(args, parser, {}, project, "openai_deepseek")]
    assert str((home / "config.ini").resolve()) in [str(Path(s).resolve()) for s in listed
                                                   if s.endswith("config.ini")]
    assert not [setting for setting in listed if setting.endswith("project.ini")]


def test_the_length_variable_is_named_as_read_only_when_it_is_read(tmp_path, monkeypatch):
    from testuary import settings

    monkeypatch.setattr(settings, "platform_user_dir", lambda: tmp_path)
    parser = cli._build_parser()
    variable = demo_cli.PLAYBACK_VARIABLE

    def instead(argv, value):
        items = demo_cli.overridden_settings(parser.parse_args(argv), parser, {variable: value},
                                             tmp_path, "openai_deepseek")
        return next(item["instead"] for item in items
                    if item["setting"] == f"environment variable {variable}")

    assert instead(["--demo"], "30") == "read: the playback's length"
    assert instead(["--demo", "--demo-playback-seconds", "20"], "30").startswith("not read")
    assert instead(["--demo"], " ") == "not read, as it is empty"


def test_an_instant_playback_reads_no_length(monkeypatch):
    """--demo-speed instant has no length, so a bad playback length set
    elsewhere does not refuse it."""
    from types import SimpleNamespace

    monkeypatch.setenv(demo_cli.PLAYBACK_VARIABLE, "not a number")
    args = SimpleNamespace(demo_speed="instant", demo_step=False, demo_duration=None, config=None)
    assert demo_cli._playback(args) is None


@pytest.mark.parametrize("fail", [False, True])
def test_demo_environment_is_restored_even_after_failure(tmp_path, monkeypatch, fail):
    monkeypatch.setenv("TESTUARY_PRIVATE_VALUE", "not for the demo")
    monkeypatch.setenv("LANGADAPTER_PRIVATE_VALUE", "not for the backend")
    monkeypatch.setenv("LANGSMITH_PRIVATE_VALUE", "not for tracing")
    monkeypatch.setenv("UNRELATED_VALUE", "retained")
    original = dict(os.environ)
    cwd = Path.cwd()
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    preset = workspace / "preset.ini"
    try:
        with demo_cli._workspace_environment(workspace, preset):
            assert Path.cwd() == workspace
            assert os.environ["TESTUARY_CONFIG_FILE"] == str(preset)
            assert "TESTUARY_PRIVATE_VALUE" not in os.environ
            assert "LANGADAPTER_PRIVATE_VALUE" not in os.environ
            assert "LANGSMITH_PRIVATE_VALUE" not in os.environ
            assert os.environ["UNRELATED_VALUE"] == "retained"
            os.environ["TESTUARY_CREATED_DURING_DEMO"] = "remove after the run"
            if fail:
                raise RuntimeError("fixture failure")
    except RuntimeError:
        assert fail
    assert Path.cwd() == cwd
    assert dict(os.environ) == original


def test_a_demonstration_inside_a_project_never_reads_its_settings_file(tmp_path):
    """A demonstration started in a project does not find that project's
    settings file: the workspace is its own project with the directory the
    preset names, so the run reads the preset only."""
    from testuary.settings import resolve_settings

    project = tmp_path / "proj"
    (project / ".testuary").mkdir(parents=True)
    (project / ".testuary" / "project.ini").write_text(
        "[starter]\nsuite_count = 5\n", encoding="utf-8")
    workspace = project / "demo" / "workspace"
    workspace.mkdir(parents=True)
    preset = workspace / "preset.ini"
    preset.write_text("[starter]\nsuite_count = 3\n", encoding="utf-8")
    entered = demo_cli._workspace_environment
    with entered(workspace, preset):
        resolved = resolve_settings(base_dir=Path.cwd())
    assert resolved.project_dir == workspace.resolve()
    assert resolved.sections["starter"]["suite_count"] == "3"
    assert "project" not in resolved._layer_files
    # A preset that names another Testuary directory gets that one.
    preset.write_text("[main]\ntestuary_dir = .demo-testuary\n", encoding="utf-8")
    with entered(workspace, preset):
        assert resolve_settings(base_dir=Path.cwd()).project_dir == workspace.resolve()
    assert (workspace / ".demo-testuary").is_dir()


@pytest.fixture
def _no_phase_loaded(monkeypatch):
    """Other tests in this process load phase modules, which makes the command
    refuse to start, so pretend none has loaded."""
    monkeypatch.setattr(demo_cli, "phase_modules_loaded", lambda: [])


def test_a_non_empty_output_directory_is_refused_and_nothing_written(
        tmp_path, capsys, _no_phase_loaded):
    outdir = tmp_path / "out"
    outdir.mkdir()
    (outdir / "kept.txt").write_text("user data", encoding="utf-8")
    parser = cli._build_parser()
    args = parser.parse_args(["--demo", "--output-dir", str(outdir)])
    assert demo_cli.main(args, parser) == demo_cli.EXIT_USAGE
    assert sorted(p.name for p in outdir.iterdir()) == ["kept.txt"]
    assert "not empty" in capsys.readouterr().err


def test_data_that_do_not_match_the_manifest_are_refused(
        tmp_path, monkeypatch, _no_phase_loaded):
    def mismatch(language):
        raise demo_cli.DataMismatch("scenario.json.gz does not match the manifest")

    monkeypatch.setattr(demo_cli, "load_data", mismatch)
    parser = cli._build_parser()
    args = parser.parse_args(["--demo", "--output-dir", str(tmp_path / "out")])
    assert demo_cli.main(args, parser) == demo_cli.EXIT_STALE
    assert not (tmp_path / "out").exists()


def test_the_shipped_data_match_their_manifest():
    manifest, files = demo_cli.load_data()
    assert set(files) == set(manifest["files"])
    assert manifest["exemplar"]["instance"] == "tenacity"
    assert set(manifest["classes"]) == {"real"}


@pytest.mark.parametrize("language", ["python", "java"])
def test_bundled_workspace_and_scenario_load_from_the_installed_packages_alone(
    tmp_path, language,
):
    """The shipped data load and write their workspace and scenario with no
    module from outside the standard library, the installed dependencies,
    Testuary and LangAdapter."""
    entries = demo_cli.load_data(language)[0]["entries"]
    script = """
import json
import site
import sys
from pathlib import Path

from testuary.demo import cli, scenario
manifest, files = cli.load_data(sys.argv[1])
workspace = Path.cwd() / 'workspace'
preset = cli._write_workspace(workspace, manifest, files)
scenario_path = Path.cwd() / 'scenario.json.gz'
scenario_path.write_bytes(files['scenario.json.gz'])
recorded = scenario.Scenario.load(scenario_path)
assert preset.is_file()
assert workspace.joinpath('projects', manifest['exemplar']['instance']).is_dir()
assert any(name.startswith('LICENSE-') for name in files)
assert files['NOTICE.md']
import langadapter
import testuary
roots = [Path(directory).resolve() for directory in (sys.base_prefix, sys.prefix, site.getusersitepackages())]
roots += [Path(testuary.__file__).resolve().parent, Path(langadapter.__file__).resolve().parent]
outside = sorted(name for name, module in list(sys.modules.items())
                 if getattr(module, '__file__', None)
                 and not any(Path(module.__file__).resolve().is_relative_to(root) for root in roots))
assert not outside, outside
print(json.dumps({'entries': len(recorded.entries), 'language': sys.argv[1]}))
"""
    environment = same_code_environment()
    environment["PATH"] = ""
    environment.pop("JAVA_HOME", None)
    child = subprocess.run([sys.executable, "-B", "-c", script, language],
                           cwd=tmp_path, env=environment, capture_output=True,
                           text=True, timeout=30)
    assert child.returncode == 0, child.stderr
    assert json.loads(child.stdout) == {"entries": entries, "language": language}


@pytest.mark.parametrize("language", ["python", "java"])
def test_materialized_demo_preset_resolves_without_changing_runtime_values(tmp_path, language):
    from testuary.settings import resolve_settings

    manifest, files = demo_cli.load_data(language)
    recorded = demo_cli._preset_values(files["preset.ini"].decode("utf-8"))
    preset = demo_cli._write_workspace(tmp_path / "workspace", manifest, files)
    materialized = demo_cli._preset_values(preset.read_text(encoding="utf-8"))
    expected = {section: dict(values) for section, values in recorded.items()}
    if language == "python":
        # A run never falls back to its own installation: the playback names
        # the interpreter it runs in, although no project code runs.
        expected.setdefault("project", {})["python_executable"] = sys.executable
    assert materialized == expected
    settings = resolve_settings(config_files=[preset], use_environment=False,
                                base_dir=preset.parent, models={})
    assert settings.main.phase_end_mutation is True
    if language == "python":
        # The recording pins the prompt's runtime line to the recorded
        # interpreter's, so a playback anywhere asks the recorded requests.
        assert settings.sections["project"]["runtime_version"] == \
            recorded["project"]["runtime_version"]
    if language == "java":
        assert preset.read_bytes() == files["preset.ini"]
    else:
        resolve_settings(overrides=recorded, use_environment=False,
                         base_dir=preset.parent, models={})
    assert demo_cli.load_data(language)[1]["preset.ini"] == files["preset.ini"]


@pytest.mark.parametrize("extra", ["[unknown_section]\nunknown = true\n",
                                 "[rig]\nout_root = unused\nunknown = true\n"])
def test_demo_preset_does_not_ignore_unknown_settings(tmp_path, extra):
    from testuary.settings import resolve_settings

    manifest, files = demo_cli.load_data("java")
    files = dict(files, **{"preset.ini": files["preset.ini"] + extra.encode("utf-8")})
    preset = demo_cli._write_workspace(tmp_path / "workspace", manifest, files)
    with pytest.raises(ValueError, match="unknown section"):
        resolve_settings(config_files=[preset], use_environment=False,
                         base_dir=preset.parent, models={})


def test_public_demo_refuses_an_unreadable_preset_before_writing_output(
        tmp_path, monkeypatch, _no_phase_loaded):
    manifest, files = demo_cli.load_data("java")
    duplicated = files["preset.ini"] + b"\n[main]\nseed = 1\n[main]\nseed = 2\n"
    files = dict(files, **{"preset.ini": duplicated})
    monkeypatch.setattr(demo_cli, "load_data", lambda language: (manifest, files))
    parser = cli._build_parser()
    output = tmp_path / "out"
    args = parser.parse_args(["--demo", "--demo-language", "java", "--output-dir", str(output)])
    assert demo_cli.main(args, parser) == demo_cli.EXIT_STALE
    assert not output.exists()


def test_a_recorded_option_this_version_refuses_is_stale_data(
        tmp_path, monkeypatch, capsys, _no_phase_loaded):
    """Data whose recorded option the command no longer takes stop before
    the run as data that no longer fit, not as a usage error."""
    manifest, files = demo_cli.load_data("java")
    invocation = dict(manifest["invocation"], options=["--no-such-option"])
    monkeypatch.setattr(demo_cli, "load_data",
                        lambda language: (dict(manifest, invocation=invocation), files))
    parser = cli._build_parser()
    output = tmp_path / "out"
    args = parser.parse_args(["--demo", "--demo-language", "java", "--output-dir", str(output)])
    assert demo_cli.main(args, parser) == demo_cli.EXIT_STALE
    assert not output.exists()
    error = capsys.readouterr().err
    assert "refuses the recorded options --no-such-option: " in error
    assert "unrecognized arguments: --no-such-option" in error


def test_java_dispatch_uses_agreed_options_and_recorded_labels(
        tmp_path, monkeypatch, _no_phase_loaded):
    called = []

    def playback(scenario, invocation, **kwargs):
        called.append((scenario, invocation, kwargs))
        assert os.environ.get("TESTUARY_PRIVATE_VALUE") is None
        assert Path.cwd().name == "workspace"
        return None

    monkeypatch.setenv("TESTUARY_PRIVATE_VALUE", "preserve without using")
    monkeypatch.setattr(demo_cli, "play", playback)
    parser = cli._build_parser()
    args = parser.parse_args(["--demo", "--output-dir", str(tmp_path / "out"),
                              "--demo-language", "java", "--display", "compact", "--no-color"])
    assert demo_cli.main(args, parser) == demo_cli.EXIT_OK
    (_, invocation, options), = called
    assert options["language"] == "java"
    assert invocation[-5:] == ["--languages", "java", "--display", "compact", "--no-color"]
    assert os.environ["TESTUARY_PRIVATE_VALUE"] == "preserve without using"
    record = json.loads((tmp_path / "out" / "DEMO.json").read_text(encoding="utf-8"))
    manifest = demo_cli.load_data("java")[0]
    assert record["language"] == "java"
    assert "recorded in one real run" in record["statement"]
    text = (tmp_path / "out" / "DEMO.md").read_text(encoding="utf-8")
    assert manifest["exemplar"]["run"] in text
    assert "Every answer was recorded in one real run of this release" in text
    target = Path(manifest["exemplar"]["target"])
    source = tmp_path / "out" / "workspace" / "projects" / manifest["exemplar"]["instance"] / target.parent
    assert target.name in (source / labels.NOTICE_NAME).read_text(encoding="utf-8")


def test_labels_preserve_source_and_replace_an_existing_label(tmp_path):
    source = tmp_path / "CalculatorTest.java"
    source.write_bytes(b"class CalculatorTest {}\n")
    record = tmp_path / "matrix.json"
    record.write_text(json.dumps({"mutants": [], labels.LABEL_FIELD: "fixture detail"}),
                      encoding="utf-8")
    labels.label_outputs(tmp_path)
    assert source.read_bytes() == b"class CalculatorTest {}\n"
    assert json.loads(record.read_text(encoding="utf-8"))[labels.LABEL_FIELD] == labels.RECORDED_LABEL
    notice = (tmp_path / labels.NOTICE_NAME).read_text(encoding="utf-8")
    assert notice.startswith(labels.RECORDED_NOTICE)


def test_data_with_answers_not_recorded_in_a_real_run_are_refused():
    """A demonstration plays only data recorded in one real run of this
    release: any other answer class makes the data not fit."""
    with pytest.raises(demo_cli.DataMismatch, match="not recorded in a real run"):
        demo_cli.check_recorded({"classes": {"real": 3, "synthetic": 1}})
    demo_cli.check_recorded({"classes": {"real": 3}})


@pytest.mark.parametrize("failure,outcome", [(KeyboardInterrupt(), "cancelled"),
                                            (OSError("fixture execution failure"), "failed")])
def test_failed_or_cancelled_demo_keeps_labeled_partial_evidence(
        tmp_path, monkeypatch, _no_phase_loaded, failure, outcome):
    output = tmp_path / "out"

    def fail_after_recording(scenario, invocation, **kwargs):
        path = output / "run" / "record.json"
        path.parent.mkdir()
        path.write_text('{"fixture":true}', encoding="utf-8")
        raise failure

    monkeypatch.setattr(demo_cli, "play", fail_after_recording)
    parser = cli._build_parser()
    args = parser.parse_args(["--demo", "--output-dir", str(output), "--demo-language", "java"])
    with pytest.raises(type(failure)):
        demo_cli.main(args, parser)
    record = json.loads((output / "DEMO.json").read_text(encoding="utf-8"))
    assert record["result"]["outcome"] == outcome
    assert json.loads((output / "run" / "record.json").read_text(
        encoding="utf-8"))[labels.LABEL_FIELD] == labels.RECORDED_LABEL
    assert (output / "demo" / "ledger.jsonl").is_file()


def test_unreadable_scenario_leaves_only_explicitly_labeled_partial_workspace(
        tmp_path, monkeypatch, _no_phase_loaded):
    from testuary.demo.scenario import Scenario

    def unreadable(path):
        raise ValueError("fixture invalid scenario")

    monkeypatch.setattr(Scenario, "load", unreadable)
    parser = cli._build_parser()
    args = parser.parse_args(["--demo", "--output-dir", str(tmp_path / "out"),
                              "--demo-language", "java"])
    assert demo_cli.main(args, parser) == demo_cli.EXIT_STALE
    record = json.loads((tmp_path / "out" / "DEMO.json").read_text(encoding="utf-8"))
    assert record["result"]["outcome"] == "stale"
    assert record["result"]["answers"] == 0
    assert "recorded in one real run" in record["statement"]


def test_a_normal_run_loads_no_demonstration_module():
    code = (
        "import sys\n"
        "import testuary.__main__, testuary.orchestrator\n"
        "from testuary import target_phases\n"
        "print(sorted(m for m in sys.modules if m.startswith('testuary.demo')))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         env=same_code_environment(), text=True, check=True)
    assert out.stdout.strip() == "[]"


def test_the_entry_point_loads_no_phase_module():
    """The demonstration starts before any phase module loads, so neither
    the entry point nor what it imports at start may load one."""
    code = (
        "import sys\n"
        "import testuary.__main__\n"
        "print(sorted(m for m in sys.modules\n"
        "             if m in ('testuary.agent.coverage_agent',\n"
        "                      'testuary.agent.mutation_agent')))\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True,
                         env=same_code_environment(), text=True, check=True)
    assert out.stdout.strip() == "[]"


def _command_line(argv):
    """The command as one string, quoted the way this platform's subprocess
    reports it: Windows passes a command line to the audit event, elsewhere
    the event carries the list and a string is POSIX-quoted."""
    if os.name == "nt":
        return subprocess.list2cmdline(argv)
    return shlex.join(argv)


def test_the_guard_refuses_every_process_the_bundled_solver_included():
    """The set-cover selections come from the scenario, so no process may
    start. A command arrives as a list or, on Windows, as one string, and the
    refusal names its program either way, even when the quoting cannot be
    split."""
    import pulp

    from testuary.demo.scenario import ScenarioStop

    solver = pulp.PULP_CBC_CMD(msg=False, _skip_v4_deprecation=True).path
    commands = [
        (_command_line([solver, str(Path("t") / "x-pulp.mps"), "-sec", "119.8"]),
         "cbc"),
        ([solver, "-version"], "cbc"),
        ("git -C x rev-parse", "git"),
        ('git -c "core.x=1', "git"),
        (["git", "status"], "git"),
    ]
    for command, program in commands:
        audit = demo_cli.Audit()
        audit.active = True
        with pytest.raises(ScenarioStop, match="starts no process"):
            audit.hook("subprocess.Popen", (None, command, None, None))
        (event,) = audit.events
        assert event["refused"] and not audit.active
        assert Path(event["executable"]).stem.lower() == program
    audit = demo_cli.Audit()
    audit.active = True
    with pytest.raises(ScenarioStop, match="starts no process"):
        audit.hook("os.system", (b"ls",))


@pytest.mark.parametrize("event, args", [
    ("os.exec", ("/bin/ls", ["ls"], {"TOKEN": "private value"})),
    ("os.posix_spawn", ("/bin/ls", ["ls"], {"TOKEN": "private value"})),
    ("os.spawn", (0, "/bin/ls", ["ls"], {"TOKEN": "private value"})),
])
def test_the_guard_records_a_process_without_its_environment(event, args):
    """audit.jsonl names the program, but not the environment values that
    the process's audit event carries."""
    from testuary.demo.scenario import ScenarioStop

    audit = demo_cli.Audit()
    audit.active = True
    with pytest.raises(ScenarioStop, match="starts no process"):
        audit.hook(event, args)
    (record,) = audit.events
    assert "/bin/ls" in record["detail"] and "private value" not in record["detail"]


def test_the_guard_refuses_every_connection_but_loopback():
    from testuary.demo.scenario import ScenarioStop

    audit = demo_cli.Audit()
    audit.active = True
    audit.hook("socket.connect", (None, ("127.0.0.1", 8000)))
    audit.hook("socket.connect", (None, ("::1", 8000, 0, 0)))
    assert not any(event.get("refused") for event in audit.events)
    with pytest.raises(ScenarioStop, match="no network"):
        audit.hook("socket.connect", (None, ("203.0.113.5", 443)))


@pytest.mark.parametrize("host", ["127.0.0.1", "127.42.0.12", "::1",
                                  "0:0:0:0:0:0:0:1", "localhost", "LOCALHOST"])
def test_the_guard_accepts_only_parsed_loopback_or_exact_localhost(host):
    audit = demo_cli.Audit()
    audit.active = True
    audit.hook("socket.connect", (None, (host, 8000)))
    assert audit.active
    assert audit.events == [{"event": "socket.connect", "address": host}]


@pytest.mark.parametrize("host", ["127.example.invalid", "127.0.0.1.example.invalid",
                                  "localhost.example.invalid", "localhost-remote",
                                  "::10", "::1:2345"])
def test_the_guard_rejects_prefix_masquerading_as_loopback(host):
    from testuary.demo.scenario import ScenarioStop

    audit = demo_cli.Audit()
    audit.active = True
    with pytest.raises(ScenarioStop, match="no network"):
        audit.hook("socket.connect", (None, (host, 8000)))
    assert not audit.active
    assert audit.events == [{"event": "socket.connect", "address": host, "refused": True}]


@pytest.mark.parametrize("language", ["python", "java"])
def test_every_workspace_alias_states_its_provider_and_the_configuration_says_so(tmp_path, language):
    manifest, files = demo_cli.load_data(language)
    demo_cli._write_workspace(tmp_path / "workspace", manifest, files)
    models = demo_cli._preset_values((tmp_path / "workspace" / "models.ini").read_text(encoding="utf-8"))
    alias = manifest["invocation"]["model"]
    assert models[alias]["provider"]
    recorded = demo_cli._preset_values(files["models.ini"].decode("utf-8"))
    rows = dict(demo_cli.configuration(manifest, files["preset.ini"].decode("utf-8"),
                                       files["models.ini"].decode("utf-8")))
    if recorded[alias].get("provider"):
        # An alias that already states its provider is copied as recorded.
        assert (tmp_path / "workspace" / "models.ini").read_bytes() == files["models.ini"]
        assert "model provider" not in rows
    else:
        assert models[alias]["provider"] == alias.partition("_")[0]
        assert rows["model provider"].startswith(f"{models[alias]['provider']} (named by")
    assert ("interpreter" in rows) is (language == "python")


def test_the_recorded_small_run_limits_show_as_its_settings_set_them():
    manifest = {"invocation": {"model": "ollama_x", "source_files": "pkg/mod.py",
                               "options": []},
                "exemplar": {"instance": "pkg"}}
    rows = dict(demo_cli.configuration(
        manifest, "[project]\nruntime_version = Python 3.12.3\n"
        "[starter]\nsuite_count = 3\n"
        "[coverage]\nmax_requests_per_phase = 20\n[mutation]\nmax_requests_per_phase = 20\n",
        "[ollama_x]\nprovider = ollama\nmodel = some-model\n"))
    assert "limits profile" not in rows
    assert rows["starter suites"].startswith("3, ")
    assert "request limit 20," in rows["coverage phase"]
    assert "request limit 20," in rows["mutation phase"]
    # Settings the profile leaves alone keep the package's values, each
    # marked as the default.
    assert rows["coverage phase"].startswith("goal: 90% branch coverage (default), ")
    assert rows["runtime in the prompts"] == "Python 3.12.3"
    assert rows["model"] == "ollama_x (some-model, provider ollama, not contacted)"


class _ReadSettings(dict):
    """A preset's values that note every (section, key) read from them."""

    def __init__(self, values, reads):
        super().__init__({name: _ReadSection(name, keys, reads) for name, keys in values.items()})
        self.reads = reads

    def get(self, section, default=None):
        return super().get(section, _ReadSection(section, {}, self.reads))


class _ReadSection(dict):
    def __init__(self, name, values, reads):
        super().__init__(values)
        self.name, self.reads = name, reads

    def get(self, key, default=None):
        self.reads.add((self.name, key))
        return super().get(key, default)


@pytest.mark.parametrize("data_set", ["python", "java", "toy", "tutorial"])
def test_every_configuration_row_reads_a_setting_the_recorded_preset_can_hold(
        monkeypatch, data_set):
    from testuary import settings

    manifest, files = demo_cli.load_data(data_set)
    preset_text = files["preset.ini"].decode("utf-8")
    reads = set()
    real = demo_cli._preset_values
    monkeypatch.setattr(demo_cli, "_preset_values",
                        lambda text: _ReadSettings(real(text), reads))
    rows = dict(demo_cli.configuration(manifest, preset_text, files["models.ini"].decode("utf-8")))
    alias = manifest["invocation"]["model"]
    settings_read = {(section, key) for section, key in reads if section != alias}
    schema = settings._schema(settings._defaults())
    # Every row reads a setting under its current name, so a renamed section
    # or key cannot fall back to the package default unseen.
    assert len(settings_read) >= 15
    for section, key in sorted(settings_read):
        assert key in schema.get(section, {}), f"[{section}] {key} is not a setting"
    # Each of them the recorded preset holds is shown with its value, or
    # with what its value means for the switches and words.
    recorded = real(preset_text)
    shown = " ".join(rows.values())
    words = {("coverage", "goal_metric"): {"both": "line and branch coverage",
                                           "either": "line or branch coverage"},
             ("mutation", "loop_catalogue"): {"full": "full catalogue"},
             ("mutation", "per_mutant_timeout_seconds"): {"auto": "from the suite's run time"},
             ("starter", "refill_failed_suites"): {"true": "generated again", "false": "not generated"},
             ("main", "phase_end_mutation"): {"true": "coverage phase: yes",
                                              "false": "coverage phase: no"}}
    for section, key in settings_read:
        if key in recorded.get(section, {}):
            value = recorded[section][key].strip()
            expected = words.get((section, key), {}).get(value.lower(), value)
            assert expected in shown, f"[{section}] {key}"


@pytest.mark.parametrize("value, shown", [("true", "yes"), ("false", "no")])
def test_the_mutation_score_row_shows_the_recorded_switch(value, shown):
    manifest = {"invocation": {"model": "m", "source_files": "a.py"},
                "exemplar": {"instance": "a"}}
    rows = dict(demo_cli.configuration(
        manifest, f"[main]\nphase_end_mutation = {value}\n",
        "[m]\nprovider = openai\nmodel = x\napi_url = http://h/v1\n"))
    assert f"coverage phase: {shown}" in rows["mutation score"]


def test_a_run_without_a_profile_names_none():
    manifest = {"invocation": {"model": "m", "source_files": "a.py"},
                "exemplar": {"instance": "a"}}
    rows = dict(demo_cli.configuration(
        manifest, "", "[m]\nprovider = openai\nmodel = x\napi_url = http://h/v1\n"))
    assert "limits profile" not in rows
    assert rows["starter suites"].startswith("10 (default), ")
    assert rows["model"] == "m (x, http://h/v1, not contacted)"


def test_the_closing_line_names_the_recorded_answers():
    assert demo_cli._answer_classes({"real": 180}) == "180 recorded"


def test_the_guard_leaves_out_socketpair_but_records_a_loopback_connection():
    import socket
    import threading

    audit = demo_cli.Audit()
    sys.addaudithook(audit.hook)
    server = socket.socket()
    server.bind(("127.0.0.1", 0))
    server.listen(1)
    threading.Thread(target=lambda: server.accept()[0].close(), daemon=True).start()
    audit.active = True
    try:
        left, right = socket.socketpair()
        left.close()
        right.close()
        assert audit.events == []
        client = socket.create_connection(server.getsockname(), timeout=5)
        client.close()
    finally:
        audit.active = False
        server.close()
    assert audit.events == [{"event": "socket.connect", "address": "127.0.0.1"}]


def test_a_look_alike_socketpair_outside_the_standard_library_is_refused(tmp_path, monkeypatch):
    from testuary.demo.scenario import ScenarioStop

    package = tmp_path / "fakepkg"
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "socket.py").write_text(
        "import socket as _socket\n\n\n"
        "def socketpair():\n"
        "    connection = _socket.socket()\n"
        "    try:\n"
        "        connection.connect(('192.0.2.1', 443))\n"
        "    finally:\n"
        "        connection.close()\n", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    from fakepkg import socket as fake

    audit = demo_cli.Audit()
    sys.addaudithook(audit.hook)
    audit.active = True
    try:
        with pytest.raises(ScenarioStop):
            fake.socketpair()
    finally:
        audit.active = False
    assert audit.events == [{"event": "socket.connect", "address": "192.0.2.1", "refused": True}]
