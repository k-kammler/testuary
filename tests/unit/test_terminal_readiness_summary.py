"""Concise and detailed readiness through the actual check route.

The real CLI, settings resolver, check producer and Python setup provider run.
Only native observations at LocalExecutor.run are controlled. These cases test
presentation of supplied facts, not native readiness or provider capability.
"""
from io import StringIO
import json
import subprocess
import sys

import pytest
from rich.console import Console

from langadapter.execution import LocalExecutor
from testuary import __main__ as cli, orchestrator, presentation, project
from testuary.orchestrator import RunRequest
from testuary.presentation import DisplayEvent, TerminalPresenter, present_readiness
from testuary.presentation import readiness as shown_report
from testuary.readiness import ReadinessItem, ReadinessLimit, RunReadiness
from testuary.settings import resolve_settings


_PROBLEMS = ("failed", "ambiguous", "missing_setup", "unsupported")


@pytest.fixture(autouse=True)
def no_generation_or_native_launch(monkeypatch):
    monkeypatch.setenv("TERM", "xterm-256color")

    def forbidden(*args, **kwargs):
        pytest.fail("check launched native code, provisioning, a model client or generation")

    monkeypatch.setattr(subprocess, "Popen", forbidden)
    for name in ("project_runtime", "provision_runtime"):
        monkeypatch.setattr(project, name, forbidden)
    for name in ("_preflight_model_configuration", "_cleanup_outdir", "run_pipeline"):
        monkeypatch.setattr(orchestrator, name, forbidden)


def _problems(report):
    """Distinct required problems as the producer reports them.

    Unverified items are not tried yet, whatever their action. A language's
    overall assessment is listed only when none of its items is a problem.
    """
    problems = [item for item in report.items if item.required and item.status in _PROBLEMS]
    alone = not any(not item.key.endswith(".assessment") for item in problems)
    return {(item.reason, item.action) for item in problems
            if alone or not item.key.endswith(".assessment")}


def _project(tmp_path, monkeypatch, collection_answer, *, verify, missing):
    root = tmp_path / "source tree"
    root.mkdir()
    (root / "subject.py").write_bytes(b"value = 1\n")
    runtime = tmp_path / "selected runtime"
    runtime.write_bytes(b"controlled interpreter placeholder\n")
    output = tmp_path / "output tree"
    environment = {"TESTUARY_PROJECT_WORK_DIR": str(tmp_path)}
    models = {"selected": {"provider": "openai", "model": "controlled-model",
                           "api_url": "https://example.test/v1", "api_key": "model-fixture-secret"}}

    def observe(executor, argv, **options):
        assert argv[0] == str(runtime)
        # LangAdapter's executor holds its context only as a private member, so the stand-in reads it there.
        assert not executor._context.environment.inherit
        if "--collect-only" in argv:
            assert verify and not missing
            return collection_answer(executor, argv, options)
        assert "env" not in options
        if argv[1:3] == ["-I", "-S"]:
            value = {"implementation": "CPython", "version": "3.10.19", "tools": {
                "pytest": None if missing else "8.1.1", "coverage": "7.6.8", "cosmic-ray": "8.4.1"}}
        else:
            assert verify and not missing
            value = {"status": "imported"}
        return subprocess.CompletedProcess(argv, 0, json.dumps(value), "")

    monkeypatch.setattr(LocalExecutor, "run", observe)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("testuary.settings.resolve_settings", lambda **kwargs: resolve_settings(
        **kwargs, models=models, environment=environment))
    overrides = {"api": {"model": "selected"}, "project": {
        "source_dir": str(root), "python_executable": str(runtime), "output_dir": str(output)}}
    settings = resolve_settings(config_files=[], base_dir=tmp_path, overrides=overrides,
                                models=models, environment=environment)
    expected = orchestrator.check(RunRequest(("subject.py",)), settings=settings, verify=verify)
    flags = ["check", "--source-files", "subject.py", "--model", "selected",
             "--source-dir", str(root), "--python-executable", str(runtime),
             "--output-dir", str(output), "--no-color", *(("--verify",) if verify else ())]
    return (root, runtime, output), flags, expected


@pytest.mark.parametrize("missing", [False, True], ids=("tools", "missing-pytest"))
@pytest.mark.parametrize("verify", [False, True], ids=("inspect", "verify"))
@pytest.mark.parametrize("width", [40, 100])
@pytest.mark.parametrize("mode", ["compact", "detailed", "plain"])
def test_actual_check_summary_groups_blockers_and_detailed_lists_every_item(
        tmp_path, monkeypatch, capsys, collection_answer, mode, width, verify, missing):
    paths, flags, expected = _project(tmp_path, monkeypatch, collection_answer, verify=verify,
                                      missing=missing)
    stream = StringIO()

    def terminal(selected_mode, *, no_color):
        console = Console(file=stream, force_terminal=True, color_system=None, no_color=no_color,
                          width=width, height=60, highlight=False, markup=False)
        return TerminalPresenter(selected_mode, console=console)

    monkeypatch.setattr(presentation, "TerminalPresenter", terminal)
    monkeypatch.setattr(sys, "argv", ["testuary", *flags, "--display", mode])
    assert cli.main() == expected.exit_code
    captured = capsys.readouterr()
    assert captured.out == "" and captured.err == ""
    text = stream.getvalue()
    lines = text.splitlines()
    # Each path stays whole on one record, also at 40 columns.
    for path in paths:
        assert any(str(path) in line for line in lines)
    assert "model-fixture-secret" not in text and "\x1b" not in text and "READY" not in text
    assert "Next action: not reported" not in text and "Next action not reported" not in text
    # Every view ends with one next step.
    assert lines[-1].startswith("Next step: ") and text.count("Next step: ") == 1
    if mode == "detailed":
        assert f"Target: {expected.source_files[0]}" in lines
        check = "quick check" if expected.scope == "inspection" else "tool check"
        assert lines[0] == (f"Testuary {check}: {shown_report._OVERALL[expected.status]} "
                            f"(exit status {expected.exit_code})")
        for item in expected.items:
            assert any(line.startswith(f"{item.key} (") or f"({item.key}, " in line for line in lines)
        assert any(line.startswith("Limit | Coverage phase | max_requests_per_phase:") for line in lines)
        return
    # Blank lines separate the groups, and long paths stay on one line each.
    assert len(lines) <= 36
    assert f"Target: {expected.source_files[0]}" in lines
    problems = _problems(expected)
    # Raw keys, setting names, protocols and statuses stay in the detailed view.
    # The producer's own reasons and the selected paths are shown as supplied.
    own = text
    supplied = [text for item in expected.items for text in (item.reason, item.action) if text]
    for value in supplied + [str(path) for path in paths]:
        own = own.replace(value, "")
    for raw in ("native.python.", "max_requests_per_phase", "openai-chat", "Testuary |", "unverified",
                "Next action", "scope", "inspection", "certify", "native", "membership"):
        assert raw not in own
    count = f"{len(problems)} problem{'s' if len(problems) != 1 else ''} found" if problems else (
        "No problems found")
    reach = "by a quick check" if expected.scope == "inspection" else "by a tool check"
    assert lines[0].startswith(f"{count} {reach}")
    if not problems and expected.exit_code:
        # The first line agrees with the exit status. The reach follows as a sentence.
        assert lines[0].endswith(f"so this check ends with exit status {expected.exit_code}.")
        assert "no model" in lines[1]
    else:
        assert "no model" in lines[0]
    for reason, action in problems:
        assert any(" ".join(reason.split()) in " ".join(line.split()) for line in lines)
        if action:
            assert action in text
    assert "Model: controlled-model via OpenAI" in lines
    # The check reports the configured key as an item of its own, not tried yet.
    assert any(line.startswith("  the API key: ") for line in lines)
    assert not any(line.startswith("API key:") for line in lines)
    assert any(line.startswith("  Python: runtime CPython 3.10.19 (") and str(paths[1]) in line
               for line in lines)
    assert any(line.startswith("Limits: starter phase up to ") for line in lines)
    assert "Full setup report: the same command with --display detailed" in lines
    if missing and len(problems) == 1 and next(iter(problems))[1]:
        assert lines[-1] == f"Next step: {next(iter(problems))[1]}"
    elif not problems and not verify:
        assert lines[-1].startswith("Next step: testuary check --verify")
        assert "Installed but not tried:" in lines
    elif not problems:
        assert lines[-1].startswith("Next step: testuary generate with the same options")


@pytest.mark.parametrize("verify", [False, True])
def test_actual_check_json_is_unchanged_and_has_no_human_rows(tmp_path, monkeypatch, capsys,
                                                              collection_answer, verify):
    _, flags, expected = _project(tmp_path, monkeypatch, collection_answer, verify=verify,
                                  missing=False)
    monkeypatch.setattr(presentation, "TerminalPresenter",
                        lambda *a, **kw: pytest.fail("JSON output must not create a presenter"))
    monkeypatch.setattr(sys, "argv", ["testuary", *flags, "--json"])
    assert cli.main() == expected.exit_code
    captured = capsys.readouterr()
    assert captured.err == "" and json.loads(captured.out) == expected.as_dict()
    assert "Next step:" not in captured.out and "problems found" not in captured.out


def _report(items, **changes):
    values = dict(
        scope="inspection", status="failed", exit_code=1, source_dir="/work/Example  Project",
        execution="local", source_files=("src/alpha.py",), phases=("starter",),
        model_alias="example", provider="local", model="example-model",
        credentials_configured=None, output_dir="/work/out put", output_mode="fresh",
        limits=(ReadinessLimit("starter", "max_requests_per_phase", 3, "requests", "configured"),
                ReadinessLimit("starter", "max_wall_clock_seconds_per_phase", None, "seconds", "unlimited")),
        items=items)
    values.update(changes)
    return RunReadiness(**values)


_MIXED = (
    ReadinessItem("native.python.operation.tests.run_suite", True, "unverified",
                  reason="native execution was not verified"),
    ReadinessItem("native.python.operation.tests.collect_suite", True, "unverified",
                  reason="native execution was not verified"),
    ReadinessItem("native.python.item.pytest", True, "missing_setup", "pytest",
                  reason="pytest is not installed in the selected runtime",
                  action="Install pytest into /work/env one/bin/python"),
    ReadinessItem("native.python.item.runtime", True, "verified", "/work/env one/bin/python",
                  "CPython 3.10.19"),
    ReadinessItem("source.targets", True, "failed", reason="src/alpha.py is outside the root"),
    ReadinessItem("native.python.item.coverage", False, "unverified", "coverage", "7.6.8",
                  reason="metadata only", action="Run check --verify"),
    ReadinessItem("native.python.diagnostic.1", False, "unverified",
                  diagnostic_path="/work/out put/setup log.txt"),
)


def _render(report, mode, *, tty=True, width=100):
    stream = StringIO()
    if tty:
        console = Console(file=stream, force_terminal=True, color_system=None, width=width,
                          height=60, highlight=False, markup=False)
        presenter = TerminalPresenter(mode, console=console)
    else:
        presenter = TerminalPresenter(mode, stream=stream, width=width)
    with presenter:
        present_readiness(report, presenter=presenter)
    return presenter, stream.getvalue()


@pytest.mark.parametrize("provider, words", [
    ("openai", "OpenAI"), ("ollama", "Ollama"), ("vllm", "vLLM"), ("local", "local")])
def test_the_model_line_names_the_built_in_provider(provider, words):
    """The three built-in providers are named in words, and an added
    provider by its section name."""
    _presenter, text = _render(_report(_MIXED, provider=provider), "plain", tty=False)
    assert f"Model: example-model via {words}" in text


@pytest.mark.parametrize("mode", ["auto", "compact", "plain", "detailed"])
def test_redirected_output_keeps_the_requested_detail(mode):
    presenter, text = _render(_report(_MIXED), mode, tty=False)
    assert presenter.mode == "plain" and "\x1b" not in text
    keys = [item.key for item in _MIXED]
    if mode == "detailed":
        assert all(f"{key} (" in text or f"({key}, " in text for key in keys)
        assert "Limit | Starter phase | max_requests_per_phase: 3 requests" in text
        assert "[configured]" not in text
    else:
        assert not any(key in text for key in keys)
        assert "Limits: starter phase up to 3 model requests, no time limit" in text.splitlines()
        assert "max_requests_per_phase" not in text and "wall_clock" not in text


@pytest.mark.parametrize("mode", ["compact", "plain"])
def test_summary_groups_by_meaning_and_keeps_only_supplied_actions(mode):
    _, text = _render(_report(_MIXED), mode)
    assert text.splitlines() == [
        "2 problems found by a quick check, which runs nothing in your project and contacts no model.",
        "",
        "Source directory: /work/Example  Project (on this computer)",
        "Target: src/alpha.py",
        "Model: example-model via local",
        "Run directory: /work/out put (a new run)",
        "Phases: Starter phase",
        "Limits: starter phase up to 3 model requests, no time limit",
        "",
        "Found:",
        "  Python: runtime CPython 3.10.19 (/work/env one/bin/python)",
        "",
        "Installed but not tried:",
        "  Python: coverage 7.6.8 (optional)",
        "    Run check --verify",
        "  The tool check is testuary check --verify with the same options. It makes no model requests.",
        "",
        "Not tried yet:",
        "  Python: running generated tests, tests collect suite",
        "",
        "Missing or blocking:",
        "  Whether the targets belong to the project: src/alpha.py is outside the root",
        "  Python pytest: pytest is not installed in the selected runtime",
        "    Fix: Install pytest into /work/env one/bin/python",
        "",
        "More detail was saved to:",
        "  /work/out put/setup log.txt",
        "",
        "Full setup report: the same command with --display detailed",
        "Next step: fix the 2 problems above, then run the same command again.",
    ]


def test_single_problem_ends_with_its_action_and_leaves_unknown_facts_out():
    report = _report((ReadinessItem("configuration", True, "failed", reason="unknown field x",
                                    action="Correct the named configuration input and check again."),),
                     source_dir=None, execution=None, source_files=(), phases=(), limits=(),
                     model_alias=None, provider=None, model=None, output_dir=None)
    _, text = _render(report, "plain")
    assert text.splitlines() == [
        "1 problem found by a quick check: the settings were refused, so it stopped before "
        "looking at your project.",
        "",
        "Missing or blocking:",
        "  Settings: unknown field x",
        "",
        "Next step: Correct the named configuration input and check again.",
    ]


@pytest.mark.parametrize("configured,sentence", [
    (True, "API key: found, not tried by this check."),
    (False, "API key: none found."),
    # Not reported: nothing is said about the key rather than a guess.
    (None, None),
])
def test_model_and_key_are_plain_sentences_without_protocols(configured, sentence):
    items = (ReadinessItem("configuration.model.initial", True, "verified", "ollama-native",
                           reason="Captured request settings resolve"),)
    _, text = _render(_report(items, provider="ollama", model="deepseek-v4.1-flash",
                              credentials_configured=configured, status="verified", exit_code=0), "plain")
    lines = text.splitlines()
    assert "Model: deepseek-v4.1-flash via Ollama" in lines
    assert sentence in lines if sentence else "API key" not in text
    # The display adds no action of its own: no variable or file names.
    assert "TESTUARY_" not in text and "models.ini" not in text
    assert "ollama-native" not in text and "alias" not in text
    assert lines[0].startswith("No problems found by a quick check")
    assert lines[-1] == "Next step: testuary check --verify with the same options (tries the installed tools, no model requests)."


@pytest.mark.parametrize("status,action,group", [
    ("missing_setup", "Set TESTUARY_OLLAMA_API_KEY in your environment.", "Missing or blocking:"),
    ("unverified", "Nothing to do now; generation uses the key for its first model request.",
     "Not tried yet:"),
])
def test_a_producer_model_key_item_replaces_the_key_sentence(status, action, group):
    item = ReadinessItem("configuration.credentials", True, status, reason="key reason",
                         action=action)
    _, text = _render(_report((item,), provider="ollama", credentials_configured=status != "missing_setup",
                              status=status, exit_code=0), "plain")
    lines = text.splitlines()
    # One statement about the key, from the producer, never a second one that could contradict it.
    assert not any(line.startswith("API key:") for line in lines) and group in lines
    if status == "missing_setup":
        assert "  The API key: key reason" in lines and lines[-1] == f"Next step: {action}"
    else:
        start = lines.index(group)
        # The item on a line of its own says why, then the producer's step.
        assert lines[start + 1:start + 3] == ["  the API key: key reason", f"    {action}"]


def _token_limits(**configured):
    names = ("max_input_tokens_per_phase", "max_output_tokens_per_phase",
             "max_total_tokens_per_phase", "max_wall_clock_seconds_per_phase")
    return tuple(ReadinessLimit("coverage", name, configured.get(name), "tokens",
                                "configured" if name in configured else "unlimited")
                 for name in names)


@pytest.mark.parametrize("configured,expected", [
    (dict(max_input_tokens_per_phase=1000),
     "Limits: coverage phase up to 1000 input tokens, no output token limit, no total token "
     "limit, no time limit"),
    (dict(max_total_tokens_per_phase=5000),
     "Limits: coverage phase up to 5000 tokens in total, no input token limit, no output token "
     "limit, no time limit"),
    ({}, "Limits: coverage phase no token or time limit"),
])
def test_no_token_limit_is_said_only_when_every_token_limit_is_unlimited(configured, expected):
    _, text = _render(_report((), limits=_token_limits(**configured), status="verified",
                              exit_code=0), "plain")
    assert expected in text.splitlines()


@pytest.mark.parametrize("name,expected", [
    ("attempts_per_suite", "up to 1 attempt per suite"),
    ("max_requests_per_phase", "up to 1 model request"),
    ("max_input_tokens_per_phase", "up to 1 input token"),
])
def test_a_ceiling_of_one_is_singular(name, expected):
    limits = (ReadinessLimit("coverage", name, 1, "", "configured"),)
    _, text = _render(_report((), limits=limits, status="verified", exit_code=0), "plain")
    assert f"Limits: coverage phase {expected}" in text.splitlines()


def test_internal_keys_never_appear_as_item_names():
    items = (ReadinessItem("native.python.runtime_selection", True, "missing_setup",
                           reason="only Testuary's environment was found", action="Select one."),
             ReadinessItem("native.python.something_new", True, "failed", reason="new reason"),
             ReadinessItem("output.extra_check", True, "failed", reason="other reason"))
    _, text = _render(_report(items), "plain")
    lines = text.splitlines()
    assert "  The Python environment for the project: only Testuary's environment was found" in lines
    assert "  Python something new: new reason" in lines and "  Extra check: other reason" in lines
    assert "native" not in text.lower().replace("native.python", "") and "output extra" not in text


def test_a_refused_deeper_check_does_not_claim_that_tools_started():
    report = _report((ReadinessItem("configuration", True, "failed", reason="no source files",
                                    action="Name the source files."),),
                     scope="verification", source_dir=None, execution=None, source_files=(),
                     phases=(), limits=(), model=None, provider=None, model_alias=None,
                     output_dir=None, exit_code=2)
    _, text = _render(report, "plain")
    assert text.splitlines()[0] == ("1 problem found by a tool check: the settings were refused, "
                                    "so it stopped before looking at your project.")
    assert "starts tools" not in text and text.splitlines()[-1] == "Next step: Name the source files."


def test_verification_names_untried_parts_and_the_exit_status_without_inventing_problems():
    items = (ReadinessItem("native.java.item.jvm", True, "verified", "/jdk/bin/java", "21.0.11"),
             ReadinessItem("native.java.item.build_descriptor", True, "unverified", "maven",
                           reason="launcher/build resolution remains unverified"),
             ReadinessItem("native.java.operation.tests.run_suite", True, "unverified",
                           reason="native execution was not verified"),
             ReadinessItem("native.python.operation.tests.run_suite", True, "unverified",
                           reason="native execution was not verified"),
             ReadinessItem("native.python.item.coverage", False, "unverified", "coverage", "7.6.8"),
             *(ReadinessItem(f"native.{language}.item.native_execution", True, "unverified",
                             reason="not tried", action="start a run; the run tries it first")
               for language in ("java", "python")))
    _, text = _render(_report(items, scope="verification", status="unverified"), "plain")
    lines = text.splitlines()
    assert lines[:2] == [
        "No problems found by a tool check, but some required parts were not tried, so this "
        "check ends with exit status 1.",
        "A tool check contacts no model, starts each tool once and, in a copy of your project, "
        "tries its build where a run builds it and finds its tests without running one."]
    start = lines.index("Found and working:")
    assert lines[start:start + 2] == ["Found and working:", "  Java: java 21.0.11"]
    start = lines.index("Installed but not tried:")
    # A tool check already ran: no hint to run it again.
    assert lines[start:start + 3] == ["Installed but not tried:", "  Python: coverage 7.6.8 (optional)", ""]
    start = lines.index("Not tried yet:")
    assert lines[start + 1:start + 6] == [
        "  Java: the build setup (maven), running generated tests",
        "  Python: running generated tests",
        # Languages with the same untried parts and action share one line.
        "  Java, Python: running your project's tests and measurements",
        "    start a run; the run tries it first", ""]
    assert lines[-1] == ("Next step: testuary generate with the same options (without --verify). "
                         "It sends model requests, which may cost money.")
    assert "/jdk/bin/java" not in text


def test_the_items_of_the_projects_build_and_tests_read_in_plain_words():
    """The items a tool check adds when it tries the project in a copy read
    in plain words after their language, never as their keys."""
    names = {"Java": {"build_resolution": "build of your project",
                      "test_compilation": "compilation of your project's tests",
                      "test_discovery": "discovery of your project's tests"},
             "Python": {"test_collection": "collection of your project's tests",
                        "project_origin": "import of your project in your tests"}}
    items = tuple(ReadinessItem(f"native.{language.lower()}.item.{key}", True, "failed",
                                reason=f"reason {number}", action="fix it")
                  for number, (language, key) in enumerate(
                      (language, key) for language, keys in names.items() for key in keys))
    _, text = _render(_report(items, scope="verification", status="failed"), "plain")
    lines = text.splitlines()
    # The first line states the reach. Where a build wrapper keeps the build for
    # the first run, the report's items say so.
    assert lines[0] == ("5 problems found by a tool check, which contacts no model, starts each "
                        "tool once and, in a copy of your project, tries its build where a run "
                        "builds it and finds its tests without running one.")
    problems = [(language, words) for language, keys in names.items() for words in keys.values()]
    for number, (language, words) in enumerate(problems):
        assert f"  {language} {words}: reason {number}" in lines
    for keys in names.values():
        assert not any(key.replace("_", " ") in text for key in keys)


def test_a_problem_without_a_reported_fix_points_to_the_detailed_view():
    _, text = _render(_report((ReadinessItem("native.python.assessment", True, "failed",
                                             reason="provider stopped"),)), "plain")
    lines = text.splitlines()
    assert "  The Python check: provider stopped" in lines and "Fix:" not in text
    assert lines[-1] == ("Next step: read every observation with the same command and --display detailed, "
                         "then fix the problem and check again.")


def test_detailed_view_keeps_raw_names_and_ends_with_the_same_next_step():
    _, text = _render(_report(_MIXED), "detailed")
    lines = text.splitlines()
    assert lines[0] == "Testuary quick check: something required failed (exit status 1)"
    assert "pytest (native.python.item.pytest, required): missing" in lines
    assert "Fix: Install pytest into /work/env one/bin/python" in lines
    assert "Fix: not reported" not in text
    assert lines[-2:] == ["", "Next step: fix the 2 problems above, then run the same command again."]


@pytest.mark.parametrize("tty", [True, False], ids=("narrow-terminal", "redirected"))
def test_plain_records_are_not_split_inside_paths_or_actions(tty):
    target = "a/very/long/package/name/with space/Calculator.java"
    stream = StringIO()
    if tty:
        console = Console(file=stream, force_terminal=True, color_system=None, width=20,
                          height=60, highlight=False, markup=False)
        presenter = TerminalPresenter("plain", console=console)
    else:
        presenter = TerminalPresenter("compact", stream=stream, width=20)
    with presenter:
        presenter(DisplayEvent("run", target, "coverage", "phase/run-1", "failed", 3,
                               message="See /work/out put/run log.json"))
        present_readiness(_report(_MIXED), presenter=presenter)
    lines = stream.getvalue().splitlines()
    assert sum(target in line and "/work/out put/run log.json" in line for line in lines) == 1
    assert "    Fix: Install pytest into /work/env one/bin/python" in lines
    assert "Source directory: /work/Example  Project (on this computer)" in lines
