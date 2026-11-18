"""Real public entry and Python setup provider with native observations controlled."""
import json
from pathlib import Path
import stat
import subprocess
import sys

import pytest

from langadapter.execution import LocalExecutor
from testuary import __main__ as cli, orchestrator, project
from testuary import config
from testuary.orchestrator import RunRequest
from testuary._words import plain_name
from testuary.settings import resolve_settings


@pytest.fixture(autouse=True)
def no_generation_or_native_launch(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("check launched native code, provisioning, a model client or generation")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    for name in ("project_runtime", "provision_runtime"):
        monkeypatch.setattr(project, name, forbidden)
    for name in ("_preflight_model_configuration", "_cleanup_outdir", "run_pipeline"):
        monkeypatch.setattr(orchestrator, name, forbidden)


@pytest.mark.parametrize("readonly", [False, True], ids=("writable", "readonly"))
@pytest.mark.parametrize("verify", [False, True])
@pytest.mark.parametrize("missing", [False, True])
@pytest.mark.parametrize("mode", ["fresh", "resume", "reuse"])
def test_actual_builtin_check_api_json_and_human_preserve_partial_readiness(
        tmp_path, monkeypatch, capsys, collection_answer, verify, missing, mode, readonly):
    root = tmp_path / "source tree"
    root.mkdir()
    target = root / "subject.py"
    target.write_bytes(b"value = 1\n")
    if readonly:
        target.chmod(stat.S_IREAD)
    runtime = tmp_path / "selected runtime"
    runtime.write_bytes(b"controlled interpreter placeholder\n")
    output = tmp_path / "output tree"
    if mode != "fresh":
        output.mkdir()
        (output / "retained.txt").write_bytes(b"previous paid work\n")
    before = {path: (path.read_bytes(), path.stat().st_mode)
              for path in (target, runtime, *output.glob("*"))}
    environment = {"TESTUARY_PROJECT_WORK_DIR": str(tmp_path), "AUTH_TOKEN": "native-fixture-secret"}
    models = {"selected": {"provider": "openai", "model": "controlled-model",
                           "api_url": "https://example.test/v1", "api_key": "model-fixture-secret"}}
    overrides = {"api": {"model": "selected"}, "project": {
        "source_dir": str(root), "python_executable": str(runtime), "output_dir": str(output),
    }}
    settings = resolve_settings(config_files=[], base_dir=tmp_path, overrides=overrides,
                                models=models, environment=environment)
    request = RunRequest(("subject.py",), resume=mode == "resume",
                         reuse=("starter",) if mode == "reuse" else ())
    calls = []

    def observe(executor, argv, **options):
        # LangAdapter's executor holds its context only as a private member, so the stand-in reads it there.
        context = executor._context
        assert argv[0] == str(runtime) and options["timeout"] > 0
        assert not context.environment.inherit and dict(context.environment) == environment
        calls.append((context, tuple(argv), dict(options)))
        if "--collect-only" in argv:
            assert verify and not missing and context.root != str(root)
            return collection_answer(executor, argv, options)
        assert "env" not in options
        if argv[1:3] == ["-I", "-S"]:
            assert context.root == str(root)
            value = {"implementation": "CPython", "version": "3.10.19", "tools": {
                "pytest": None if missing else "8.1.1", "coverage": "7.6.8", "cosmic-ray": "8.4.1",
            }}
        else:
            assert verify and not missing and context.root != str(root)
            assert options["cwd"] == context.root and "-B" in argv
            assert argv[-1] in {"pytest", "coverage", "cosmic_ray"}
            assert (Path(context.root) / "subject.py").read_bytes() == before[target][0]
            value = {"status": "imported"}
        return subprocess.CompletedProcess(argv, 0, json.dumps(value), "")

    monkeypatch.setattr(LocalExecutor, "run", observe)
    monkeypatch.chdir(tmp_path)
    resolve_calls = []

    def resolve_cli(**kwargs):
        resolve_calls.append(kwargs)
        return resolve_settings(**kwargs, models=models, environment=environment)

    monkeypatch.setattr("testuary.settings.resolve_settings", resolve_cli)
    flags = ["--source-files", "subject.py", "--model", "selected", "--source-dir", str(root),
             "--python-executable", str(runtime), "--output-dir", str(output),
             "--display", "plain", "--no-color"]
    if verify:
        flags.append("--verify")
    if mode == "resume":
        flags.append("--resume")
    elif mode == "reuse":
        flags.extend(("--reuse", "starter"))

    # A verification with every tool present passes. The rest, including
    # saved output of a resume or reuse, only a run tries and validates.
    passes = verify and not missing
    expected_status = "missing_setup" if missing else "verified" if passes else "unverified"
    expected_code = 1 if missing or (verify and not passes) else 0
    expected = orchestrator.check(request, settings=settings, verify=verify)
    assert capsys.readouterr() == ("", "")
    assert expected.status == expected_status and expected.exit_code == expected_code
    monkeypatch.setattr(sys, "argv", ["testuary", "check", *flags, "--json"])
    assert cli.main() == expected_code
    rendered = capsys.readouterr()
    assert rendered.err == "" and json.loads(rendered.out) == expected.as_dict()
    monkeypatch.setattr(sys, "argv", ["testuary", "check", *flags])
    assert cli.main() == expected_code
    rendered = capsys.readouterr()
    assert rendered.out == "" and rendered.err and "\x1b" not in rendered.err
    assert "native-fixture-secret" not in rendered.err and "model-fixture-secret" not in rendered.err
    facts = {item.key: item for item in expected.items}
    assert facts["source.targets"].status == "verified"
    assert facts["output.admission"].status == facts["output.original_separation"].status == "verified"
    if mode != "fresh":
        assert facts["output.saved_artifacts"].status == "unverified"
        assert facts["output.saved_artifacts"].tried_by == "run" and facts["output.saved_artifacts"].action
    # Operations stay not tried (only a run tries them) even when the check passes.
    operation_status = "missing_setup" if missing else "unverified"
    assert facts["native.python.operation.tests.run_suite"].status == operation_status
    assert facts["native.python.operation.mutations.open_session"].status == operation_status
    # Each item has a plain name of its own, and the summary names every operation.
    names = [plain_name(item) for item in expected.items]
    assert len(set(names)) == len(names), names
    assert all(plain_name(item) in rendered.err for key, item in facts.items() if ".operation." in key)
    assert facts["native.python.item.pytest"].status == (
        "missing_setup" if missing else "verified" if verify else "unverified")
    if verify:
        assert facts["native.python.item.native_execution"].status == "unverified"
    assert expected.source_dir == str(root) and expected.output_mode == mode
    assert expected.model == "controlled-model" and expected.credentials_configured is True
    assert len(resolve_calls) == 2 and config.active_settings() is None
    assert len([argv for _, argv, _ in calls if argv[1:3] == ("-I", "-S")]) == 3
    prepared = [context for context, argv, _ in calls if argv[1:3] != ("-I", "-S")]
    assert bool(prepared) is (verify and not missing)
    assert all(not Path(context.root).exists() for context in prepared)
    assert {path: (path.read_bytes(), path.stat().st_mode) for path in before} == before
    assert output.exists() is (mode != "fresh")
