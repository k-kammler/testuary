"""Generation origins and profiles at real invocation/phase boundaries."""
import json
import os
from types import SimpleNamespace

import pytest

import test_resolved_starter_requests as starter_harness
import test_model_protocols as protocol_harness
from agent_fixtures import logs_in
from testuary import orchestrator, phase_state
from testuary import config
from testuary import project as project_module
from testuary import prompts
from testuary.evidence import current_condition, phase_origin_path, write_origin
from testuary.agent import coverage_agent, mutation_agent
from testuary import starters
from testuary import target_phases as phases
from testuary.settings import PHASE_SECTIONS, PROVIDER_SECTIONS, ConfigurationError, RunSettings, resolve_settings

streams = starter_harness.streams
transport = protocol_harness.transport
#: The mutation settings' hashes under the shipped configuration, as the
#: measurement identities record them: the loop's agent configuration and
#: the phase end's mutation configuration.
SHIPPED_CONFIG_SHA256 = {
    "loop": "bda2343f203aaffb1766aa229ee19a4563c8df8267251b9c2faec10cd9775beb",
    "phase end": "8cb463f9d82f23cd29a2d247cf744409189af3b7393bfb2e4f708179383df65d",
}


def settings_at(root, *, profile="", phase_fields=None, **model_fields):
    # Real phase entries build project contexts, which need a captured environment.
    return RunSettings(base_dir=root, models=starter_harness.configured(**model_fields).models, sections={
        "api": {"model": "explicit"}, "main": {"prompt_profile": profile, "hints_file": "", "seed": 37},
        "starter": {"suite_count": 1, "attempts_per_suite": 1, "concurrent_requests": 1,
                         "suite_run_timeout_seconds": 2, "max_output_tokens": 40, **(phase_fields or {})},
    }, environment=dict(os.environ))


def profile_at(root, value=None):
    path = root / "profile.json"
    value = value or {"format": 1, "phases": {
        phase: {"system_prompt": {"append": f"{phase} profile guidance"}}
        for phase in ("starter", "coverage", "mutation")
    }}
    if "phases" in value:  # a profile file names the starter phase initial
        value = {**value, "phases": {"initial" if name == "starter" else name: entry
                                     for name, entry in value["phases"].items()}}
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_actual_entry_uses_captured_profile_and_records_its_origin(phase, tmp_path, monkeypatch, streams):
    profile = profile_at(tmp_path)
    project = tmp_path / "subject"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    target = project / "module.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    output = tmp_path / "tests"
    output.mkdir()
    source = "def test_ok():\n    assert True\n"
    (output / "test_module.py").write_text(source, encoding="utf-8")
    seen = []
    cls = coverage_agent.CoverageAgent if phase == "coverage" else mutation_agent.MutationAgent
    monkeypatch.setattr(cls, "run_phase", lambda agent, **kw: seen.append(agent.prompts) or SimpleNamespace())
    monkeypatch.setattr(mutation_agent, "_seed_mutation_baseline_from_matrix", lambda agent: True)
    monkeypatch.setattr(cls, "resolve_log_dir", lambda agent: tmp_path / "logs")
    with config.use_run_settings(settings_at(tmp_path, profile=profile.name)):
        original_bytes = profile.read_bytes()
        profile.write_text("invalid after capture", encoding="utf-8")
        if phase == "starter":
            starters.run_starter_phase(language="python", target_file_path=str(target),
                                       source_dir=str(project), test_output_dir=str(output),
                                      suites_dir=str(tmp_path / "suites"), log=lambda *a, **k: None)
            path = tmp_path / "suites/generation_origin.json"
            assert "starter profile guidance" in streams[0][0][1]["messages"][0]["content"]
            transport = json.loads((tmp_path / "suites/prompt_transport.jsonl").read_text().splitlines()[0])
        elif phase == "coverage":
            coverage_agent.generate_coverage_tests("subject", {"file_path": str(target)}, str(output),
                baseline={"content": source, "line_coverage": 50.0, "branch_coverage": 50.0,
                          "test_count": 1, "suite_seconds": 0.01})
            path = tmp_path / "logs/coverage_generation_origin.json"
        else:
            mutation_agent.generate_mutation_tests("subject", str(target), str(output), preseed={"content": source})
            path = tmp_path / "logs/mutation_generation_origin.json"
        origin = json.loads(path.read_text())
        if phase != "starter":
            assert f"{phase} profile guidance" in seen[0]["system_prompt"]
            assert origin["condition"] == current_condition(phase, "python", str(project), templates=seen[0]).export()
        else:
            assert transport["prompt_composition"] == origin["condition"]["prompt_composition"]
        assert (path.parent / origin["prompt_profile"]["copy"]).read_bytes() == original_bytes
        assert origin["prompt_profile"]["changes"] == [["system_prompt", "", "append"]]
    assert prompts.prompt_inputs() == (None, None)


@pytest.mark.parametrize("change", ["temperature", "profile", "missing-origin"])
def test_initial_reuse_refuses_changed_or_unknown_origin_before_paid_writes(change, tmp_path, streams):
    original = settings_at(tmp_path)
    _, suites = starter_harness.run_worker(tmp_path, original)
    if change == "missing-origin":
        (suites / "generation_origin.json").unlink()
    snapshot = {p.relative_to(tmp_path): p.read_bytes() for p in tmp_path.rglob("*") if p.is_file()}
    kwargs = {"phase_fields": {"reuse_suites": True}}
    if change == "temperature":
        kwargs["temperature"] = 0.5
    if change == "profile":
        kwargs["profile"] = profile_at(tmp_path).name
    with config.use_run_settings(settings_at(tmp_path, **kwargs)), pytest.raises(ConfigurationError, match="reuse refused"):
        starters.run_starter_phase(language="python", target_file_path=str(tmp_path / "subject/module.py"),
                                   source_dir=str(tmp_path / "subject"), test_output_dir=str(tmp_path / "output"),
                                  suites_dir=str(suites), log=lambda *a, **k: None)
    assert len(streams[0]) == 1
    for relative, data in snapshot.items():
        assert (tmp_path / relative).read_bytes() == data


def test_initial_credential_and_transport_rotation_keep_paid_records_and_origin(tmp_path, streams):
    _, suites = starter_harness.run_worker(tmp_path, settings_at(tmp_path))
    paid = (suites / "request_records.jsonl").read_bytes()
    origin = (suites / "generation_origin.json").read_bytes()
    changed = settings_at(tmp_path, api_key="rotated", request_timeout_seconds=700,
                          phase_fields={"reuse_suites": True, "max_transient_failures_per_phase": 200})
    with config.use_run_settings(changed):
        result = starters.run_starter_phase(language="python",
            target_file_path=str(tmp_path / "subject/module.py"), source_dir=str(tmp_path / "subject"),
            test_output_dir=str(tmp_path / "output"), suites_dir=str(suites), log=lambda *a, **k: None)
    assert len(streams[0]) == result.requests_used == 1
    assert (suites / "request_records.jsonl").read_bytes() == paid
    assert (suites / "generation_origin.json").read_bytes() == origin


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_real_phase_context_refuses_reuse_before_preset_or_markers(phase, tmp_path, monkeypatch, streams):
    target = tmp_path / "module.py"
    target.write_text("pass\n", encoding="utf-8")
    ctx = phases.TargetPhasesContext(log_print=lambda *a, **k: None, project_name="subject", workspace="target_number",
        top_file={"target_number": 1, "file_path": str(target)}, canonical_path=target,
        work_tests=tmp_path / "work/tests", results_dir=tmp_path / "results", target_logs_dir=tmp_path / "logs",
        reuse=phases.ReuseFlags(**{phase: True}), settings=settings_at(tmp_path))
    monkeypatch.setattr(phases, "_record_start", lambda *a: pytest.fail("marker changed before guard"))
    origin = phase_origin_path(ctx.results_dir / ctx.workspace, phase)
    origin.parent.mkdir(parents=True)
    origin.write_text('{"condition": null}', encoding="utf-8")
    old = ctx.results_dir / "resolved_settings.json"
    old.write_text("old preset", encoding="utf-8")
    monkeypatch.setattr(phases, "_phase_starter", lambda *a: pytest.fail("generation ran before guard"))
    with pytest.raises(ConfigurationError, match=f"requested {phase} reuse refused"), \
            config.use_run_settings(ctx.settings):
        phases.run_target_phases(ctx)
    assert old.read_text() == "old preset"
    assert origin.read_text() == '{"condition": null}'
    assert streams[0] == []


def test_profile_selection_precedence_and_setup_failure_release_guard(tmp_path):
    profile = profile_at(tmp_path)
    (tmp_path / "config.ini").write_text("[main]\nprompt_profile=missing.json\n", encoding="utf-8")
    settings = resolve_settings(base_dir=tmp_path, models={},
        environment={"TESTUARY_MAIN_PROMPT_PROFILE": profile.name})
    with config.use_run_settings(settings):
        assert prompts.prompt_inputs()[0].path == profile
    settings = resolve_settings(base_dir=tmp_path, models={},
        environment={"TESTUARY_MAIN_PROMPT_PROFILE": "missing-again.json"},
        overrides=orchestrator._cli_settings_overrides(SimpleNamespace(prompt_profile=profile.name)))
    with config.use_run_settings(settings):
        assert prompts.prompt_inputs()[0].path == profile
    with pytest.raises(ConfigurationError, match="absent"), config.use_run_settings(settings_at(tmp_path, profile="missing.json")):
        pytest.fail("invalid profile entered runtime")
    with config.use_run_settings(settings_at(tmp_path)):
        assert prompts.prompt_inputs()[0] is None


def test_public_reuse_refusal_precedes_output_cleanup(tmp_path, monkeypatch, streams):
    from testuary.__main__ import _build_parser, _check_run_options

    project = tmp_path / "projects/subject"
    project.mkdir(parents=True)
    target = project / "module.py"
    target.write_text("pass\n", encoding="utf-8")
    settings = settings_at(tmp_path)
    raw = {"api": {"model": "explicit"}, "main": {"seed": 37, "hints_file": ""},
           "project": {"name": "subject", "source_dir": str(project), "output_dir": str(tmp_path / "output")}}
    settings = RunSettings(base_dir=tmp_path, models=settings.models, sections=raw,
                           environment=dict(os.environ))
    workspace = project_module.target_workspace_name(target, project)
    origin = phase_origin_path(tmp_path / "output/results" / workspace, "starter")
    with config.use_run_settings(settings):
        write_origin(origin, current_condition("starter", "python", str(project)))
    args = _build_parser().parse_args(["--source-files", "module.py", "--reuse", "starter"])
    _check_run_options(args)
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_cleanup_outdir", lambda **kw: pytest.fail("cleanup preceded refusal"))
    before = origin.read_bytes()
    changed = RunSettings(base_dir=tmp_path, sections=raw,
                          models={"explicit": {**settings.models["explicit"], "temperature": 0.5}},
                          environment=dict(os.environ))
    with pytest.raises(ConfigurationError, match="temperature changed"):
        orchestrator.run_pipeline(args, settings=changed)
    assert origin.read_bytes() == before
    assert streams[0] == []


def test_the_origin_records_only_the_models_the_run_used(tmp_path):
    """The origin's settings hold the catalogue entries the run's model
    reads, its alias and its provider's section, and of the provider
    sections only that provider's. No other alias or provider, whose host
    can be private."""
    project = tmp_path / "subject"
    project.mkdir()
    models = {"explicit": settings_at(tmp_path).models["explicit"],
              "openai": {"request_timeout_seconds": 131},
              "other": {"provider": "ollama", "model": "m", "api_url": "http://private-alias.test:11434"},
              "ollama": {"api_url": "http://private-provider.test:11434"}}
    settings = RunSettings(base_dir=tmp_path, models=models, sections={
        "api": {"model": "explicit"}, "main": {"seed": 37, "hints_file": ""},
        "vllm": {"api_url": "http://private-section.test:8000/v1"}}, environment=dict(os.environ))
    origin = tmp_path / "starter_generation_origin.json"
    with config.use_run_settings(settings):
        write_origin(origin, current_condition("starter", "python", str(project)))
    recorded = json.loads(origin.read_text(encoding="utf-8"))["settings"]
    assert set(recorded["models"]) == {"explicit", "openai"}
    assert set(recorded["sections"]) & set(PROVIDER_SECTIONS) == {"openai"}
    assert "private" not in origin.read_text(encoding="utf-8")


class _Cleanup(Exception):
    """The run passed the reuse preflight and reached the output cleanup."""


@pytest.mark.parametrize("section, key, value, refused", [
    ("starter", "suite_mutation", "both", "suite_mutation"),
    ("main", "reduced_suite_mutation", "measure", "reduced_suite_mutation"),
    ("main", "phase_end_mutation", "false", "phase_end_mutation"),
    ("mutation", "cosmic_ray_operators", "wong_mathur_selective", "cosmic_ray_operators"),
    ("mutation", "per_mutant_timeout_factor", "20", "per_mutant_timeout_factor"),
    ("coverage", "report_reverse_order_failures", "true", "report_reverse_order_failures"),
    ("starter", "suite_run_timeout_seconds", "300", "suite_run_timeout_seconds"),
    ("coverage", "reduction_solver_random_seed", "7", "reduction_solver_random_seed"),
    # Worker count and distributor change waiting time, not verdicts.
    ("mutation", "engine_workers", "15", None),
    # Finishing is measured again instead (phase_state).
    ("timing", "run_count", "9", None),
])
def test_a_resume_refuses_changed_measurement_settings_before_any_work(
    section, key, value, refused, tmp_path, monkeypatch, streams,
):
    """A resume under changed measurement settings is refused before cleanup
    or any request. A change that alters no verdict passes."""
    from testuary.__main__ import _build_parser, _check_run_options

    project = tmp_path / "projects/subject"
    project.mkdir(parents=True)
    target = project / "module.py"
    target.write_text("pass\n", encoding="utf-8")
    models = settings_at(tmp_path).models
    raw = {"api": {"model": "explicit"}, "main": {"seed": 37, "hints_file": ""},
           "project": {"name": "subject", "source_dir": str(project), "output_dir": str(tmp_path / "output")},
           "mutation": {"cosmic_ray_distributor": "http"}}
    workspace = project_module.target_workspace_name(target, project)
    origins = [phase_origin_path(tmp_path / "output/results" / workspace, phase)
               for phase in ("starter", "coverage", "mutation")]
    with config.use_run_settings(RunSettings(base_dir=tmp_path, models=models, sections=raw,
                                            environment=dict(os.environ))):
        for origin, phase in zip(origins, ("starter", "coverage", "mutation")):
            write_origin(origin, current_condition(phase, "python", str(project)))
    args = _build_parser().parse_args(["--source-files", "module.py", "--reuse", "starter,coverage,mutation,finishing"])
    _check_run_options(args)
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    # The preserved artifacts are another preflight's subject.
    monkeypatch.setattr(orchestrator, "_preflight_reuse_requirements", lambda *a: [])

    def cleanup(**kw):
        raise _Cleanup()

    monkeypatch.setattr(orchestrator, "_cleanup_outdir", cleanup)
    changed_raw = {**raw, section: {**raw.get(section, {}), key: value}}
    changed = RunSettings(base_dir=tmp_path, models=models, sections=changed_raw,
                          environment=dict(os.environ))
    before = [origin.read_bytes() for origin in origins]
    if refused is None:
        with pytest.raises(_Cleanup):
            orchestrator.run_pipeline(args, settings=changed)
    else:
        with pytest.raises(ConfigurationError, match=f"measurement settings changed: .*{refused}"):
            orchestrator.run_pipeline(args, settings=changed)
    assert [origin.read_bytes() for origin in origins] == before
    assert streams[0] == []


def test_a_reuse_through_another_address_continues_with_a_warning(tmp_path, monkeypatch, streams):
    """The provider and the model name decide a reuse: a local proxy on
    another port reaches the same service, so the run goes on and says so
    once for each reused phase."""
    from testuary.__main__ import _build_parser, _check_run_options

    project = tmp_path / "projects/subject"
    project.mkdir(parents=True)
    target = project / "module.py"
    target.write_text("pass\n", encoding="utf-8")
    models = settings_at(tmp_path).models
    raw = {"api": {"model": "explicit"}, "main": {"seed": 37, "hints_file": ""},
           "project": {"name": "subject", "source_dir": str(project), "output_dir": str(tmp_path / "output")},
           "mutation": {"cosmic_ray_distributor": "http"}}
    workspace = project_module.target_workspace_name(target, project)
    with config.use_run_settings(RunSettings(base_dir=tmp_path, models=models, sections=raw,
                                            environment=dict(os.environ))):
        for phase in ("starter", "coverage", "mutation"):
            write_origin(phase_origin_path(tmp_path / "output/results" / workspace, phase),
                         current_condition(phase, "python", str(project)))
    args = _build_parser().parse_args(["--source-files", "module.py", "--reuse", "starter,coverage,mutation,finishing"])
    _check_run_options(args)
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: None)
    monkeypatch.setattr(orchestrator, "_preflight_reuse_requirements", lambda *a: [])
    messages = []
    monkeypatch.setattr(orchestrator, "log_print", lambda message, **kwargs: messages.append(message))

    def cleanup(**kw):
        raise _Cleanup()

    monkeypatch.setattr(orchestrator, "_cleanup_outdir", cleanup)
    moved = {**models["explicit"], "api_url": "http://127.0.0.1:9090/v1"}
    with pytest.raises(_Cleanup):
        orchestrator.run_pipeline(args, settings=RunSettings(
            base_dir=tmp_path, models={"explicit": moved}, sections=raw, environment=dict(os.environ)))
    warnings = [message for message in messages if "api_url" in message]
    assert sorted(message.split(" results:")[0] for message in warnings) == [
        f"WARNING: {workspace}, {phase}" for phase in ("coverage", "mutation", "starter")]
    assert all("provider and model name match" in message and "9090" not in message
               for message in warnings)
    assert streams[0] == []


def test_a_starter_pool_is_reusable_under_other_measurement_settings(tmp_path):
    """Saved starter suites can be measured another way: the pool's origin
    records only their generation, and the starter phase origin a reuse writes
    holds the measuring run's settings, which a later resume compares with."""
    from testuary.evidence import (
        copy_origin, generation_only, recorded_condition, require_origin)

    models = settings_at(tmp_path).models
    templates = {"system_prompt": "s", "user_prompt_template": "u"}

    def settings(mode, **initial_fields):
        return RunSettings(base_dir=tmp_path, models=models, environment=dict(os.environ), sections={
            "api": {"model": "explicit"}, "main": {"hints_file": "", "seed": 37},
            "starter": {"suite_mutation": mode, **initial_fields}})

    def condition(mode, **initial_fields):
        with config.use_run_settings(settings(mode, **initial_fields)):
            return current_condition("starter", "python", str(tmp_path), templates=templates)

    measured, both = condition("measure"), condition("both")
    suites = tmp_path / "suites" / "generation_origin.json"
    with config.use_run_settings(settings("measure")):
        write_origin(suites, generation_only(measured))
    # The pool keeps only how its starters were cleaned.
    assert set(recorded_condition(suites)["measurement"]) == {
        "suite_run_timeout_seconds", "failing_test_cleanup_rounds"}
    require_origin(suites, generation_only(both))
    # Starters cleaned another way are not reused.
    with pytest.raises(ConfigurationError, match="measurement settings changed: failing_test_cleanup_rounds"):
        require_origin(suites, generation_only(condition("measure", failing_test_cleanup_rounds=5)))
    origin = tmp_path / "initial_origin.json"
    copy_origin(suites, origin, both)
    assert json.loads(recorded_condition(origin)["measurement"]["suite_mutation"]) == "both"
    require_origin(origin, both)
    with pytest.raises(ConfigurationError, match="measurement settings changed: suite_mutation"):
        require_origin(origin, measured)


def test_a_record_without_measurement_settings_cannot_be_resumed(tmp_path):
    from testuary.evidence import reuse_problems

    settings = resolve_settings(base_dir=tmp_path, config_files=[], environment={},
                                models=settings_at(tmp_path).models,
                                overrides={"api": {"model": "explicit"}, "main": {"hints_file": ""}})
    with config.use_run_settings(settings):
        initial_condition = current_condition("starter", "python", str(tmp_path), templates={
            "system_prompt": "s", "user_prompt_template": "u"})
        coverage_condition = current_condition("coverage", "python", str(tmp_path), templates={
            "system_prompt": "s", "user_prompt_template": "u", "tool_messages": {}})
    assert json.loads(initial_condition.export()["measurement"]["suite_mutation"]) == "measure"
    assert "suite_mutation" not in coverage_condition.export()["measurement"]
    assert "engine_workers" not in coverage_condition.export()["measurement"]
    earlier = {k: v for k, v in coverage_condition.export().items() if k != "measurement"}
    assert reuse_problems(earlier, coverage_condition) == ("recorded measurement settings are unknown",)
    assert reuse_problems(coverage_condition.export(), coverage_condition) == ()


def _shipped_settings(tmp_path):
    return resolve_settings(base_dir=tmp_path, config_files=[], environment={},
                            models=settings_at(tmp_path).models,
                            overrides={"api": {"model": "explicit"}, "main": {"hints_file": ""}})


def _coverage_condition(language, tmp_path):
    return current_condition("coverage", language, str(tmp_path), templates={
        "system_prompt": "s", "user_prompt_template": "u", "tool_messages": {}})


@pytest.mark.parametrize("language, tool", [("python", "cosmic_ray"), ("java", "pit"), ("c", "mull")])
def test_the_records_name_the_mutation_tool_that_ran(language, tool, tmp_path):
    """Under the shipped engine setting, a run's generation condition, the
    settings its generation record holds and its resolved settings name the
    language's own mutation tool: Cosmic Ray for Python, PIT for Java and
    Mull for C."""
    origin = tmp_path / "record.json"
    with config.use_run_settings(_shipped_settings(tmp_path)):
        condition = _coverage_condition(language, tmp_path)
        write_origin(origin, condition)
        preset = config.resolved_test_preset(language=language)
    record = json.loads(origin.read_text(encoding="utf-8"))
    assert condition.export()["measurement"]["engine"] == json.dumps(tool)
    assert record["condition"]["measurement"]["engine"] == json.dumps(tool)
    assert record["settings"]["sections"]["mutation"]["engine"] == tool
    assert preset["sections"]["mutation"]["engine"] == tool


@pytest.mark.parametrize("language, recorded, refused", [
    ("java", "cosmic_ray", False), ("c", "cosmic_ray", False), ("java", "pit", False),
    ("java", "mull", True), ("python", "pit", True),
])
def test_a_recorded_engine_is_read_as_the_tool_that_ran(language, recorded, refused, tmp_path):
    """A record that names the shipped default for every language resumes
    and is reused with the language's own tool. A recorded tool that is not
    the one the run uses is refused."""
    from testuary.evidence import require_origin

    origin = tmp_path / "record.json"
    with config.use_run_settings(_shipped_settings(tmp_path)):
        condition = _coverage_condition(language, tmp_path)
        write_origin(origin, condition)
    record = json.loads(origin.read_text(encoding="utf-8"))
    record["condition"]["measurement"]["engine"] = json.dumps(recorded)
    origin.write_text(json.dumps(record), encoding="utf-8")
    if refused:
        with pytest.raises(ConfigurationError, match="measurement settings changed: engine"):
            require_origin(origin, condition)
    else:
        assert require_origin(origin, condition) == ()


def test_the_mutation_identity_hashes_the_engine_as_configured(tmp_path):
    """The mutation measurement identity hashes the mutation settings as
    configured, the shipped engine default included, for every language, so
    the recorded identities stay valid while the records name the tool that
    ran."""
    from testuary import measurement
    from testuary.agent.mutation_agent import mutation_measurement_identity

    target = tmp_path / "target.py"
    target.write_text("pass\n", encoding="utf-8")
    with config.use_run_settings(_shipped_settings(tmp_path)):
        hashes = {(name, mutation_measurement_identity(
            target_file_path=str(target), language=language, execution_name="test_target.py",
            config=chosen)["config_sha256"])
            for language in ("python", "java", "c")
            for name, chosen in (("loop", None), ("phase end", measurement.mutation_config()))}
    assert hashes == set(SHIPPED_CONFIG_SHA256.items())


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
@pytest.mark.parametrize("controls, expected", [
    ({}, {}),
    ({"temperature": 0, "top_p": 0, "seed": 9, "request_timeout_seconds": 12},
     {"temperature": 0, "top_p": 0, "seed": 9}),
    ({"reasoning": "off", "request_options": '{"top_k": 7}'},
     {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 7}),
    ({"reasoning_budget_tokens": 13}, {"thinking_token_budget": 13}),
])
def test_actual_editing_graph_records_the_profile_and_sent_settings(phase, controls, expected, tmp_path, monkeypatch, transport):
    from testuary._progress import RunProgress, observe_target

    events = []
    profile = profile_at(tmp_path)
    project = tmp_path / "subject"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    target = project / "module.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    output = tmp_path / "tests"
    output.mkdir()
    source = "def test_ok():\n    assert True\n"
    (output / "test_module.py").write_text(source, encoding="utf-8")
    cls = coverage_agent.CoverageAgent if phase == "coverage" else mutation_agent.MutationAgent
    monkeypatch.setattr(cls, "finalize_result", lambda agent, result, **kw: SimpleNamespace(
        requests=kw["usage_tracker"].requests_used))
    monkeypatch.setattr(mutation_agent, "_seed_mutation_baseline_from_matrix", lambda agent: True)
    model_fields = {"explicit": {**settings_at(tmp_path).models["explicit"], "reasoning_dialect": "vllm",
                                 "reasoning_template": "qwen"}}
    settings = RunSettings(base_dir=tmp_path, models=model_fields, sections={
        "api": {"model": "explicit"}, "main": {"hints_file": "", "prompt_profile": profile.name},
        "coverage": {"max_requests_per_phase": 1, "max_loop_attempts": 0, **controls},
        "mutation": {"max_requests_per_phase": 1, "max_loop_attempts": 0, **controls},
    }, environment=dict(os.environ))
    with config.use_run_settings(settings), observe_target(RunProgress(events.append), str(target)):
        logs_in(monkeypatch, tmp_path / "logs")
        if phase == "coverage":
            result = coverage_agent.generate_coverage_tests("subject", {"file_path": str(target)}, str(output),
                baseline={"content": source, "line_coverage": 50.0, "branch_coverage": 50.0,
                          "test_count": 1, "suite_seconds": 0.01})
        else:
            result = mutation_agent.generate_mutation_tests("subject", str(target), str(output), preseed={"content": source})
    assert result.requests == len(transport) == 1
    body = transport[0][1]
    for key, value in expected.items():
        assert body[key] == value
    assert f"{phase} profile guidance" in body["messages"][0]["content"]
    origin = json.loads((tmp_path / "logs" / f"{phase}_generation_origin.json").read_text())
    transport = json.loads((tmp_path / "logs" / f"{phase}_prompt_transport.jsonl").read_text().splitlines()[0])
    assert transport["prompt_composition"] == origin["condition"]["prompt_composition"]
    assert transport["sent_settings"]["phase"] == phase
    assert transport["sent_settings"]["model"] == body["model"]
    assert "fixture" not in json.dumps(transport["sent_settings"])
    assert events[0].state == "running" and events[0].phase == phase
    delivered = [event for event in events if event.state == "succeeded"]
    assert len(delivered) == 1
    assert delivered[0].usage.logical_requests == delivered[0].usage.physical_requests == 1
    assert delivered[0].usage.input_tokens == 2 and delivered[0].usage.output_tokens == 1
    # A loop step's observation carries no usage, only its state.
    assert all(event.total is None and event.usage.cached_input_tokens is None
               for event in events if event.level != "step")
    usage = json.loads((tmp_path / "logs" / f"{phase}_usage.jsonl").read_text().splitlines()[0])
    assert usage["model_name"] == "reported"


def test_same_composition_keeps_origin_even_when_profile_file_metadata_changes(tmp_path, streams):
    profile = profile_at(tmp_path)
    _, suites = starter_harness.run_worker(tmp_path, settings_at(tmp_path, profile=profile.name))
    origin = (suites / "generation_origin.json").read_bytes()
    payload = json.loads(profile.read_text())
    payload["_note"] = "metadata only"
    other = tmp_path / "renamed.json"
    other.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with config.use_run_settings(settings_at(tmp_path, profile=other.name, phase_fields={"reuse_suites": True})):
        starters.run_starter_phase(language="python",
            target_file_path=str(tmp_path / "subject/module.py"), source_dir=str(tmp_path / "subject"),
            test_output_dir=str(tmp_path / "output"), suites_dir=str(suites), log=lambda *a, **k: None)
    assert len(streams[0]) == 1
    assert (suites / "generation_origin.json").read_bytes() == origin


def test_origin_retains_actual_hint_bytes_after_the_file_changes(tmp_path, streams):
    hints = tmp_path / "hints.json"
    data = b'{"initial": {"general": "shared guidance"}}\n'
    hints.write_bytes(data)
    settings = RunSettings(environment=dict(os.environ), base_dir=tmp_path, models=settings_at(tmp_path).models, sections={
        "api": {"model": "explicit"}, "main": {"hints_file": hints.name},
        "starter": {"suite_count": 1, "attempts_per_suite": 1, "concurrent_requests": 1},
    })
    with config.use_run_settings(settings):
        hints.write_text("invalid after capture", encoding="utf-8")
        condition = current_condition("starter", "python", str(tmp_path))
        path = tmp_path / "origin.json"
        write_origin(path, condition)
    record = json.loads(path.read_text())
    assert record["hints"]["path"] == str(hints)
    assert (path.parent / record["hints"]["copy"]).read_bytes() == data


def test_presets_and_repeat_origins_are_not_relabeled_on_reuse(tmp_path, monkeypatch, streams):
    target = tmp_path / "module.py"
    target.write_text("pass\n", encoding="utf-8")
    ctx = phases.TargetPhasesContext(project_name="subject", workspace="target_number", run_idx=1,
        top_file={"target_number": 1, "file_path": str(target)}, canonical_path=target,
        work_tests=tmp_path / "work/tests", results_dir=tmp_path / "results", target_logs_dir=tmp_path / "logs",
        reuse=phases.ReuseFlags(coverage=True), settings=settings_at(tmp_path), log_print=lambda *a, **k: None)
    with config.use_run_settings(ctx.settings):
        condition = phases._context_generation_condition(ctx, "coverage")
        origin = phase_origin_path(ctx.results_dir / ctx.workspace, "coverage", 0)
        write_origin(origin, condition)
        before = origin.read_bytes()
        preset = ctx.results_dir / "resolved_settings.json"
        preset.write_text("original preset", encoding="utf-8")
        phases._guard_generation_reuse(ctx)
        phases._dump_resolved_preset(ctx)
        assert preset.read_text() == "original preset"
        assert origin.read_bytes() == before
        assert not phase_origin_path(ctx.results_dir / ctx.workspace, "coverage", 1).exists()
    assert streams[0] == []


@pytest.mark.parametrize("resume, kept", [(False, False), (True, True)])
def test_a_reuse_writes_its_settings_afresh_and_a_resume_keeps_them(resume, kept, tmp_path):
    """A run that reuses an earlier run's phases into a repetition states the
    settings its own phases run under. A resume keeps the statement its run
    began with."""
    target = tmp_path / "module.py"
    target.write_text("pass\n", encoding="utf-8")
    ctx = phases.TargetPhasesContext(log_print=lambda *a, **k: None, project_name="subject",
        workspace="target_number", top_file={"target_number": 1, "file_path": str(target)},
        canonical_path=target, work_tests=tmp_path / "work/tests", results_dir=tmp_path / "results",
        target_logs_dir=tmp_path / "logs", reuse=phases.ReuseFlags(coverage=True),
        settings=settings_at(tmp_path), resume=resume)
    preset = tmp_path / "results" / "target_number" / "r0" / "resolved_settings.json"
    preset.parent.mkdir(parents=True)
    preset.write_text("an earlier invocation's settings", encoding="utf-8")
    with config.use_run_settings(ctx.settings):
        phases._dump_resolved_preset(ctx)
    assert (preset.read_text(encoding="utf-8") == "an earlier invocation's settings") == kept


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_actual_checkpoints_record_condition_separately_from_metrics_and_identity(phase, tmp_path, monkeypatch):
    target = tmp_path / "module.py"
    target.write_text("pass\n", encoding="utf-8")
    target_dir = tmp_path / "results/target_number"
    monkeypatch.setattr(phase_state, "_current_resume_context", lambda target=None: {"existing": "identity"})
    with config.use_run_settings(settings_at(tmp_path)):
        condition = current_condition(phase, "python", str(tmp_path))
        write_origin(phase_origin_path(target_dir, phase, 1), condition)
    ctx = phases.TargetPhasesContext(log_print=lambda *a, **k: None, project_name="subject", workspace="target_number",
        top_file={"target_number": 1, "file_path": str(target)}, canonical_path=target,
        work_tests=tmp_path / "work/tests", results_dir=tmp_path / "results", target_logs_dir=tmp_path / "logs",
        reuse=phases.ReuseFlags(), run_idx=1, display_target="module.py")
    phases._record_start(ctx, phase)
    block = phase_state.load(target_dir)[phase]["1"]
    assert block["generation_condition"] == condition.export()
    assert block["resume_context"] == {"existing": "identity"}
    phases._record_done(ctx, phase, True, {}, {"requests": 1})
    block = phase_state.load(target_dir)[phase]["1"]
    assert block["generation_condition"] == condition.export()
    assert block["metrics"] == {"requests": 1}
    assert block["status"] == phase_state.COMPLETE


def test_suite_reuse_carries_original_profile_and_settings_into_new_phase_record(tmp_path, streams):
    profile = profile_at(tmp_path)
    suites = tmp_path / "results/target_number/r0/archive/starter/suites"
    (suites / "suite_001").mkdir(parents=True)
    (suites / "suite_001/cleaned.py").write_text("def test_saved():\n    assert True\n", encoding="utf-8")
    (suites / starters._GENERATION_MANIFEST).write_text('{"generation_complete": true}', encoding="utf-8")
    old = settings_at(tmp_path, profile=profile.name)
    with config.use_run_settings(old):
        write_origin(suites / "generation_origin.json", current_condition("starter", "python", str(tmp_path)))
    original = json.loads((suites / "generation_origin.json").read_text())
    target = tmp_path / "module.py"
    target.write_text("pass\n", encoding="utf-8")
    ctx = phases.TargetPhasesContext(project_name="subject", workspace="target_number",
        top_file={"target_number": 1, "file_path": str(target)}, canonical_path=target,
        work_tests=tmp_path / "work/tests", results_dir=tmp_path / "results", target_logs_dir=tmp_path / "logs",
        reuse=phases.ReuseFlags(), log_print=lambda *a, **k: None)
    renamed = tmp_path / "renamed.json"
    renamed.write_bytes(profile.read_bytes())
    with config.use_run_settings(settings_at(tmp_path, profile=renamed.name, request_timeout_seconds=30,
                                           phase_fields={"reuse_suites": True})):
        phases._start_phase(ctx, "starter")
    origin = phase_origin_path(ctx.results_dir / ctx.workspace, "starter")
    carried = json.loads(origin.read_text())
    assert carried["condition"] == original["condition"]
    assert carried["settings"] == original["settings"]
    assert {k: v for k, v in carried["prompt_profile"].items() if k != "copy"} == {
        k: v for k, v in original["prompt_profile"].items() if k != "copy"}
    assert (origin.parent / carried["prompt_profile"]["copy"]).read_bytes() == profile.read_bytes()
    assert streams[0] == []


def test_copied_suite_inputs_are_isolated_per_repeat(tmp_path):
    from testuary.evidence import copy_origin

    profile = profile_at(tmp_path)
    source = tmp_path / "suites/generation_origin.json"
    with config.use_run_settings(settings_at(tmp_path, profile=profile.name)):
        first = current_condition("starter", "python", str(tmp_path))
        write_origin(source, first)
        destination = phase_origin_path(tmp_path / "target_number", "starter", 0)
        copy_origin(source, destination, first)
    record = json.loads(destination.read_text())
    captured = destination.parent / record["prompt_profile"]["copy"]
    before = captured.read_bytes()
    profile_at(tmp_path, {"format": 1, "phases": {"starter": {
        "system_prompt": {"append": "different repeated generation"}}}})
    with config.use_run_settings(settings_at(tmp_path, profile=profile.name)):
        second = current_condition("starter", "python", str(tmp_path))
        write_origin(source, second)
        copy_origin(source, phase_origin_path(tmp_path / "target_number", "starter", 1), second)
    assert captured.read_bytes() == before
    assert json.loads(destination.read_text()) == record


def test_repeat_reuse_checks_the_canonical_initial_evidence_it_actually_restores(tmp_path):
    target = tmp_path / "module.py"
    target.write_text("pass\n", encoding="utf-8")
    ctx = phases.TargetPhasesContext(log_print=lambda *a, **k: None, project_name="subject", workspace="target_number", run_idx=1,
        top_file={"target_number": 1, "file_path": str(target)}, canonical_path=target,
        work_tests=tmp_path / "work/tests", results_dir=tmp_path / "results", target_logs_dir=tmp_path / "logs",
        reuse=phases.ReuseFlags(starter=True))
    with config.use_run_settings(settings_at(tmp_path)):
        origin = phase_origin_path(ctx.results_dir / ctx.workspace, "starter", 0)
        write_origin(origin, phases._context_generation_condition(ctx, "starter"))
        before = origin.read_bytes()
        phases._guard_generation_reuse(ctx)
    assert origin.read_bytes() == before
    assert not phase_origin_path(ctx.results_dir / ctx.workspace, "starter", 1).exists()


@pytest.mark.parametrize("contents", [None, "directory", "invalid", '{"format": 1, "phases": {"wrong": {}}}'])
def test_selected_invalid_profile_is_a_public_cli_error_before_pipeline_work(contents, tmp_path, monkeypatch, capsys):
    import sys
    from testuary.__main__ import _main

    if contents == "directory":
        (tmp_path / "profile.json").mkdir()
    elif contents is not None:
        (tmp_path / "profile.json").write_text(contents, encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    # A generation request: the bare command only prints help and does no work.
    monkeypatch.setattr(sys, "argv", ["testuary", "--prompt-profile", "profile.json",
                                      "--source-files", "module.py"])
    monkeypatch.setattr(orchestrator, "_preflight_run_dependencies", lambda *a: pytest.fail("invalid profile reached pipeline"))
    with pytest.raises(SystemExit) as error:
        _main()
    assert error.value.code == 2
    assert "profile" in capsys.readouterr().err
    assert not (tmp_path / "output").exists()


@pytest.mark.parametrize("phase", ["starter", "coverage", "mutation"])
def test_empty_profile_preserves_actual_composition_and_records_the_selected_file(phase, tmp_path):
    with config.use_run_settings(settings_at(tmp_path)):
        expected = current_condition(phase, "python", str(tmp_path))
    path = profile_at(tmp_path, {"format": 1, "phases": {}})
    with config.use_run_settings(settings_at(tmp_path, profile=path.name)):
        actual = current_condition(phase, "python", str(tmp_path))
        origin = tmp_path / f"{phase}.json"
        write_origin(origin, actual)
    assert actual.export() == expected.export()
    record = json.loads(origin.read_text())
    assert record["prompt_profile"]["changes"] == []
    assert (origin.parent / record["prompt_profile"]["copy"]).read_bytes() == path.read_bytes()


@pytest.mark.parametrize("phase", ["coverage", "mutation"])
@pytest.mark.parametrize("journal", ["normal", "disabled", "write-failed"])
@pytest.mark.parametrize("token_limit", [False, True])
def test_actual_editing_response_observations_do_not_promise_usage_persistence(
        phase, journal, token_limit, tmp_path, monkeypatch, transport):
    import builtins

    from testuary.agent import _langgraph as lg
    from testuary._progress import RunProgress, observe_target

    project = tmp_path / "subject"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    target = project / "module.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    output = tmp_path / "tests"
    output.mkdir()
    source = "def test_ok():\n    assert True\n"
    (output / "test_module.py").write_text(source, encoding="utf-8")
    cls = coverage_agent.CoverageAgent if phase == "coverage" else mutation_agent.MutationAgent
    monkeypatch.setattr(cls, "finalize_result", lambda agent, result, **kw: SimpleNamespace(
        requests=kw["usage_tracker"].requests_used, logical=kw["usage_tracker"].logical_requests_used))
    monkeypatch.setattr(mutation_agent, "_seed_mutation_baseline_from_matrix", lambda agent: True)
    if journal == "disabled":
        original = lg.UsageLogger.__init__

        def disabled(logger, *args, **kwargs):
            original(logger, *args, **kwargs)
            logger._path = logger._lock = None

        monkeypatch.setattr(lg.UsageLogger, "__init__", disabled)
    elif journal == "write-failed":
        def failed_open(path, *args, **kwargs):
            if str(path).endswith("_usage.jsonl"):
                raise OSError("offline journal write failure")
            return builtins.open(path, *args, **kwargs)

        monkeypatch.setattr(lg, "open", failed_open, raising=False)
    settings = RunSettings(base_dir=tmp_path, models=settings_at(tmp_path).models, sections={
        "api": {"model": "explicit"}, "main": {"hints_file": ""},
        PHASE_SECTIONS[phase]: {"max_requests_per_phase": 1, "max_loop_attempts": 0,
                                "max_output_tokens_per_phase": 1 if token_limit else 0},
    }, environment=dict(os.environ))
    events = []
    with config.use_run_settings(settings), observe_target(RunProgress(events.append), str(target)):
        logs_in(monkeypatch, tmp_path / "logs")
        if phase == "coverage":
            result = coverage_agent.generate_coverage_tests("subject", {"file_path": str(target)}, str(output),
                baseline={"content": source, "line_coverage": 50.0, "branch_coverage": 50.0,
                          "test_count": 1, "suite_seconds": 0.01})
        else:
            result = mutation_agent.generate_mutation_tests("subject", str(target), str(output), preseed={"content": source})
    assert result.requests == result.logical == len(transport) == 1
    request = [event for event in events if event.operation.startswith("request/")]
    assert [event.state for event in request] == [
        "running", "limit_reached" if token_limit else "succeeded"]
    assert "received" in request[-1].message.lower()
    assert all("recorded" not in event.message.lower() for event in request)
    journal_rows = (tmp_path / "logs" / f"{phase}_usage.jsonl").read_text().splitlines()
    assert len(journal_rows) == (1 if journal == "normal" else 0)
