"""The ordinary guide's examples reach real parsers, not native execution."""
import ast
from pathlib import Path
import re
import shlex
import subprocess

import pytest

from testuary import __main__ as cli, orchestrator
from testuary import models as _model_protocols
from testuary import config
from testuary import project
from testuary.orchestrator import RunRequest
from testuary.settings import RunSettings


ROOT = Path(__file__).parents[2]
README = ROOT / "README.md"
# The ordinary guide's examples live on their documentation pages.
GETTING_STARTED = ROOT / "docs" / "getting-started.md"
CONFIGURATION = ROOT / "docs" / "guide" / "configuration.md"


@pytest.fixture(autouse=True)
def no_execution(monkeypatch):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **k: pytest.fail("native execution"))
    monkeypatch.setattr(_model_protocols, "build_resolved_chat_model", lambda *a, **k: pytest.fail("model construction"))
    monkeypatch.setattr(project, "project_runtime", lambda *args, **kwargs: pytest.fail("runtime provisioning"))


def test_readme_python_settings_and_typed_invocation_are_constructible(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TESTUARY_CONFIG_FILE", raising=False)
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    (tmp_path / "preset.ini").write_text("[main]\nseed = 3\n", encoding="utf-8")
    (tmp_path / "local.ini").write_text(
        "[api]\nmodel = fixture\nprovider = openai\napi_url = https://example.test/v1\n", encoding="utf-8")
    observed = []
    checks = []
    report = object()

    def run(request, *, settings):
        assert isinstance(request, RunRequest) and isinstance(settings, RunSettings)
        observed.append((request, settings))
        return 0

    def check(request, *, settings, verify=False):
        assert isinstance(request, RunRequest) and isinstance(settings, RunSettings)
        checks.append((request, settings, verify))
        return report

    monkeypatch.setattr(orchestrator, "run_pipeline", run)
    monkeypatch.setattr(orchestrator, "check", check)
    blocks = re.findall(r"^```python\n(.*?)^```", CONFIGURATION.read_text(encoding="utf-8"),
                        re.MULTILINE | re.DOTALL)
    assert len(blocks) == 3
    namespaces = []
    for block in blocks:
        namespace = {}
        exec(compile(ast.parse(block), str(CONFIGURATION), "exec"), namespace)
        namespaces.append(namespace)
    construction = next(namespace for namespace in namespaces if "layered" in namespace)
    inspection = next(namespace for namespace in namespaces if "report" in namespace)
    generation = next(namespace for namespace in namespaces if "exit_code" in namespace)
    models, problems, credential = orchestrator._resolve_generation_models(
        cli._build_parser().parse_args([]), construction["settings"])
    assert problems == () and len(models) == 3 and credential == ""
    assert construction["layered"].main.seed == 0
    assert len(checks) == 1 and checks[0][0].source_files == ("package/module.py",)
    assert checks[0][1].sections["api"]["model"] == "configured_alias" and checks[0][2] is False
    assert inspection["report"] is report
    assert len(observed) == 1 and observed[0][0].source_files == ("package/module.py",)
    assert generation["exit_code"] == 0
    assert config.active_settings() is None


def test_readme_python_and_java_quickstarts_use_the_actual_generation_parser():
    guide = GETTING_STARTED.read_text(encoding="utf-8")
    blocks = re.findall(r"^```bash\n(.*?)^```", guide, re.MULTILINE | re.DOTALL)
    # The first runs start in the project's directory, so the project's
    # settings file applies, the project directory is the source directory and the
    # results go to its .testuary directory.
    generation = [block for block in blocks if block.startswith("cd ") and "testuary generate " in block]
    assert len(generation) == 2
    values = []
    for block in generation:
        cd_line, command = block.split("\n", 1)
        assert cd_line.startswith("cd /path/to/")
        tokens = shlex.split(command.replace("\\\n", " "))
        assert tokens[:2] == ["testuary", "generate"]
        args = cli._build_parser(prog="testuary generate").parse_args(tokens[2:])
        values.append((args, orchestrator._cli_settings_overrides(args)))
    assert values[0][0].source_files == "package/module.py"
    assert values[0][1]["project"]["python_executable"] == "/path/to/project-env/bin/python"
    assert values[1][0].source_files.endswith("Calculator.java")
    assert values[1][1]["project"]["java_home"] == "/path/to/jdk"
    assert not any("source_dir" in fields["project"] or "output_dir" in fields["project"]
                   for _, fields in values)


def test_readme_check_command_uses_the_actual_inspection_parser():
    guide = GETTING_STARTED.read_text(encoding="utf-8")
    checks = [block for block in re.findall(r"^```bash\n(.*?)^```", guide, re.MULTILINE | re.DOTALL)
              if block.startswith("testuary check ")]
    for block in checks:
        tokens = shlex.split(block.replace("\\\n", " "))
        args = cli._build_parser(prog="testuary check", checking=True).parse_args(tokens[2:])
        assert args.verify is ("--verify" in tokens)
    commands = [block for block in checks if "--verify" not in block]
    assert len(commands) == 1
    tokens = shlex.split(commands[0].replace("\\\n", " "))
    assert tokens[:2] == ["testuary", "check"]
    args = cli._build_parser(prog="testuary check", checking=True).parse_args(tokens[2:])
    fields = orchestrator._cli_settings_overrides(args)
    assert args.source_files == "package/module.py" and not args.verify
    assert fields["project"]["source_dir"] == "/path/to/project"
    assert fields["project"]["python_executable"] == "/path/to/project-env/bin/python"


def test_every_landing_page_command_uses_the_actual_parser():
    blocks = re.findall(r"^```bash\n(.*?)^```", README.read_text(encoding="utf-8"),
                        re.MULTILINE | re.DOTALL)
    parsed = 0
    for block in blocks:
        tokens = shlex.split(block.replace("\\\n", " "))
        if tokens[:1] == ["testuary"] and tokens[1:2] in (["check"], ["generate"]):
            checking = tokens[1] == "check"
            cli._build_parser(prog=f"testuary {tokens[1]}", checking=checking).parse_args(tokens[2:])
            parsed += 1
    assert parsed
