"""Layer precedence at actual starter-suite, coverage and mutation request boundaries."""
from dataclasses import replace
import json
import os
from types import SimpleNamespace

import httpx
import pytest

import test_model_protocols as protocol_harness
from agent_fixtures import logs_in
import test_resolved_starter_requests as starter_harness
from testuary import config
from testuary.models import resolve_phase_model
from testuary.agent import coverage_agent, mutation_agent
from testuary.settings import resolve_settings

streams = starter_harness.streams
transport = protocol_harness.transport

PHASES = ("starter", "coverage", "mutation")
SCENARIOS = [
    pytest.param((), {}, (.6, .6, .6), ("alias:[explicit]",) * 3, (32768,) * 3, id="alias-over-bundled"),
    pytest.param(("[coverage]\ntemperature=.2\n",), {}, (.6, .2, .2),
                 ("alias:[explicit]", "config[0]:[coverage]", "config[0]:[coverage]"),
                 (32768,) * 3, id="user-over-alias-and-mutation-inherits"),
    pytest.param(("[coverage]\ntemperature=.2\n",), {"TESTUARY_API_TEMPERATURE": ".3"},
                 (.3, .3, .3), ("environment:[api]",) * 3, (32768,) * 3, id="layer-before-section"),
    pytest.param(("[coverage]\ntemperature=.2\n",), {"TESTUARY_COVERAGE_TEMPERATURE": ".1"},
                 (.6, .1, .1), ("alias:[explicit]", "environment:[coverage]", "environment:[coverage]"),
                 (32768,) * 3, id="environment-over-phase-file"),
    pytest.param(("[api]\ntemperature=.3\n[coverage]\ntemperature=.2\n[mutation]\ntemperature=.1\n",),
                 {}, (.3, .2, .1), ("config[0]:[api]", "config[0]:[coverage]", "config[0]:[mutation]"),
                 (32768,) * 3, id="within-layer-phase-order"),
    pytest.param(("[coverage]\ntemperature=.2\n[mutation]\ntemperature=.1\n", "[api]\ntemperature=.3\n"),
                 {}, (.3, .3, .3), ("config[1]:[api]",) * 3, (32768,) * 3, id="later-file-before-section"),
    pytest.param(("[starter]\ntemperature=\n",), {}, (None, .6, .6),
                 ("config[0]:[starter]", "alias:[explicit]", "alias:[explicit]"),
                 (32768,) * 3, id="empty-clears-alias"),
    pytest.param(("[coverage]\ntemperature=.2\n",), {"TESTUARY_COVERAGE_TEMPERATURE": ""},
                 (.6, None, None), ("alias:[explicit]", "environment:[coverage]", "environment:[coverage]"),
                 (32768,) * 3, id="empty-environment-is-an-override"),
    pytest.param(("[api]\ntemperature=0\n",), {}, (0, 0, 0), ("config[0]:[api]",) * 3,
                 (32768,) * 3, id="zero-is-sent"),
    pytest.param(("[coverage]\nmax_output_tokens=8192\n",), {}, (.6, .6, .6),
                 ("alias:[explicit]",) * 3, (32768, 8192, 8192), id="independent-model-phase-caps"),
]


def settings_for(tmp_path, contents, environment, *, models=None, api_overrides=None):
    files = []
    for index, text in enumerate(contents):
        path = tmp_path / f"override_{index}.ini"
        path.write_text(text, encoding="utf-8")
        files.append(path)
    return resolve_settings(base_dir=tmp_path, config_files=files,
        environment=environment, models=models if models is not None else {
            "openai": {"api_url": "https://example.test/v1", "api_key": "fixture"},
            "explicit": {"provider": "openai", "model": "chosen", "temperature": .6,
                         "max_output_tokens": 32768},
        }, overrides={
            "api": {"model": "explicit", **(api_overrides or {})}, "main": {"hints_file": "", "seed": 37},
            "starter": {"suite_count": 1, "attempts_per_suite": 1, "concurrent_requests": 1,
                             "suite_run_timeout_seconds": 2},
            "coverage": {"max_requests_per_phase": 1, "max_loop_attempts": 0},
            "mutation": {"max_requests_per_phase": 1, "max_loop_attempts": 0},
        })


def invoke_phase(phase, tmp_path, settings, monkeypatch, *, expected_requests=1, usage_snapshot=None,
                 language="python"):
    if phase == "starter":
        result, starters_dir = starter_harness.run_worker(tmp_path, settings, language=language)
        assert result.requests_used == expected_requests and result.logical_requests == 1
        if usage_snapshot is not None:
            usage_snapshot.update(physical=result.requests_used, logical=result.logical_requests,
                                  unknown=result.unknown_usage_requests,
                                  input_tokens=result.input_tokens, output_tokens=result.output_tokens)
        return json.loads((starters_dir / "prompt_transport.jsonl").read_text().splitlines()[0])
    if not settings.environment_captured:
        # Editing entries build a real project context, which needs a captured
        # launch environment.
        settings = replace(settings, environment=dict(os.environ))
    project = tmp_path / "subject"
    project.mkdir()
    (project / ("pom.xml" if language == "java" else "pyproject.toml")).write_text("", encoding="utf-8")
    target = project / ("Calculator.java" if language == "java" else "module.py")
    target.write_text(
        "public class Calculator {\n    public static int add(int a, int b) { return a + b; }\n}\n"
        if language == "java" else "def add(a, b):\n    return a + b\n", encoding="utf-8",
    )
    output = tmp_path / "tests"
    output.mkdir()
    source = (
        "import org.junit.jupiter.api.Test;\nimport static org.junit.jupiter.api.Assertions.assertEquals;\n"
        "public class CalculatorTest {\n    @Test void adds() { assertEquals(3, Calculator.add(1, 2)); }\n}\n"
        if language == "java" else "def test_ok():\n    assert True\n"
    )
    (output / ("CalculatorTest.java" if language == "java" else "test_module.py")).write_text(source, encoding="utf-8")
    cls = coverage_agent.CoverageAgent if phase == "coverage" else mutation_agent.MutationAgent
    def finalize(agent, result, **kwargs):
        tracker = kwargs["usage_tracker"]
        if usage_snapshot is not None:
            usage_snapshot.update(physical=tracker.requests_used, logical=tracker.logical_requests_used,
                                  unknown=tracker.unknown_usage_requests,
                                  input_tokens=tracker.input_tokens_used, output_tokens=tracker.output_tokens_used)
        return SimpleNamespace(requests=tracker.requests_used)

    monkeypatch.setattr(cls, "finalize_result", finalize)
    monkeypatch.setattr(mutation_agent, "_seed_mutation_baseline_from_matrix", lambda agent: True)
    with config.use_run_settings(settings):
        logs_in(monkeypatch, tmp_path / "logs")
        if phase == "coverage":
            result = coverage_agent.generate_coverage_tests("subject", {"file_path": str(target)}, str(output),
                baseline={"content": source, "line_coverage": 50.0, "branch_coverage": 50.0,
                          "test_count": 1, "suite_seconds": 0.01})
        else:
            result = mutation_agent.generate_mutation_tests("subject", str(target), str(output), preseed={"content": source})
    assert result.requests == expected_requests
    path = tmp_path / "logs" / f"{phase}_prompt_transport.jsonl"
    return json.loads(path.read_text().splitlines()[0])


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("contents,environment,temperatures,sources,caps", SCENARIOS)
def test_precedence_is_preserved_by_real_phase_requests(
    phase, contents, environment, temperatures, sources, caps, tmp_path, monkeypatch, request,
):
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    settings = settings_for(tmp_path, contents, environment)
    index = PHASES.index(phase)
    resolved = resolve_phase_model(settings, phase)
    assert resolved.values["temperature"] == temperatures[index]
    assert resolved.sources["temperature"] == sources[index]
    assert resolved.values["max_output_tokens"] == caps[index]
    recorded = invoke_phase(phase, tmp_path, settings, monkeypatch)
    assert len(sent) == 1
    body = sent[0][1]
    assert body["model"] == "chosen"
    if temperatures[index] is None:
        assert "temperature" not in body
    else:
        assert body["temperature"] == temperatures[index]
    assert body["max_completion_tokens"] == caps[index]
    assert recorded["sent_settings"]["values"]["temperature"] == temperatures[index]
    assert recorded["sent_settings"]["sources"]["temperature"] == sources[index]
    assert recorded["sent_settings"]["values"]["max_output_tokens"] == caps[index]


@pytest.mark.parametrize("phase", PHASES)
@pytest.mark.parametrize("catalogue_temperature,alias_temperature,expected,source", [
    (None, None, .3, "provider:[openai]"),
    (.4, None, .4, "catalogue:[openai]"),
    (.4, .6, .6, "alias:[explicit]"),
])
def test_provider_defaults_catalogue_and_alias_order_reaches_each_caller(
    phase, catalogue_temperature, alias_temperature, expected, source, tmp_path, monkeypatch, request,
):
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    models = {
        "openai": {"api_url": "https://example.test/v1", "api_key": "fixture"},
        "explicit": {"provider": "openai", "model": "chosen"},
    }
    if catalogue_temperature is not None:
        models["openai"]["temperature"] = catalogue_temperature
    if alias_temperature is not None:
        models["explicit"]["temperature"] = alias_temperature
    settings = settings_for(tmp_path, ("[openai]\ntemperature=.3\n",), {}, models=models)
    recorded = invoke_phase(phase, tmp_path, settings, monkeypatch)
    assert len(sent) == 1 and sent[0][1]["temperature"] == expected
    assert recorded["sent_settings"]["values"]["temperature"] == expected
    assert recorded["sent_settings"]["sources"]["temperature"] == source


@pytest.mark.parametrize("phase", PHASES)
def test_explicit_override_beats_environment_and_lower_phase_layers_at_each_caller(phase, tmp_path, monkeypatch, request):
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    settings = settings_for(tmp_path, ("[starter]\ntemperature=.1\n[coverage]\ntemperature=.2\n",),
                            {"TESTUARY_COVERAGE_TEMPERATURE": ".3"}, api_overrides={"temperature": .9})
    recorded = invoke_phase(phase, tmp_path, settings, monkeypatch)
    assert len(sent) == 1 and sent[0][1]["temperature"] == .9
    assert recorded["sent_settings"]["values"]["temperature"] == .9
    assert recorded["sent_settings"]["sources"]["temperature"] == "overrides:[api]"


@pytest.mark.parametrize("phase", PHASES)
def test_only_model_selection_keeps_each_callers_default_request_fields(phase, tmp_path, monkeypatch, request):
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    settings = settings_for(tmp_path, (), {}, models={
        "explicit": {"provider": "openai", "model": "chosen", "api_url": "https://example.test/v1", "api_key": "fixture"},
    })
    invoke_phase(phase, tmp_path, settings, monkeypatch)
    assert len(sent) == 1
    body = sent[0][1]
    assert "temperature" not in body
    if phase == "starter":
        assert body["seed"] == 38
    else:
        assert "seed" not in body
    assert not {"top_p", "thinking", "reasoning", "reasoning_effort", "thinking_token_budget",
                "chat_template_kwargs", "max_tokens", "max_completion_tokens"}.intersection(body)


@pytest.mark.parametrize("phase,protocol", [
    (phase, protocol) for phase in PHASES for protocol in ("openai-chat", "openai-responses")
] + [("starter", "ollama-native")])
def test_captured_credential_reaches_transport_not_generated_files(
    phase, protocol, tmp_path, monkeypatch, request, capsys,
):
    sentinel = "synthetic-model-credential-not-for-output"
    headers = []
    real_transport = httpx.MockTransport

    def watched_transport(handle):
        def watched(request):
            headers.append(request.headers.get("Authorization"))
            return handle(request)
        return real_transport(watched)

    monkeypatch.setattr(httpx, "MockTransport", watched_transport)
    fixture = request.getfixturevalue("streams" if phase == "starter" else "transport")
    sent = fixture[0] if phase == "starter" else fixture
    settings = settings_for(tmp_path, (), {"FIXTURE_CREDENTIAL": sentinel}, models={
        "explicit": {"provider": "openai", "model": "chosen", "api_url": "https://example.test/v1",
                     "api_key_env": "FIXTURE_CREDENTIAL", "protocol": protocol},
    })
    invoke_phase(phase, tmp_path, settings, monkeypatch)
    assert len(sent) == 1 and headers == [f"Bearer {sentinel}"]
    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert files
    for path in files:
        assert sentinel.encode() not in path.read_bytes(), path.relative_to(tmp_path)
    output = capsys.readouterr()
    assert sentinel not in output.out and sentinel not in output.err
    assert sentinel not in repr(settings) and sentinel not in json.dumps(settings.export())
