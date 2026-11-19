"""A language registered from outside runs through the generic backend.

Calc (``calc_language``) provides level 1 only: suite runs, coverage, a
mutation session and the test units. The generic backend reaches each
through LangAdapter's public operations, every optional family takes its
recorded fallback, and a whole run with a scripted model delivers a measured
suite without a network request.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, ChatResult

import calc_language
from testuary import models as model_source
from testuary import languages
from testuary import config
from testuary.languages import generic
from testuary import languages as profile
from testuary.settings import resolve_settings

SOURCE = "double(x) = x * 2\nadd(a, b) = a + b\nsub(a, b) = a - b\n"
TESTS = ("test doubles: double(2) == 4\ntest adds: add(1, 2) == 3\n"
         "test subtracts: sub(5, 2) == 3\n")


@pytest.fixture
def calc(tmp_path):
    calc_language.register()
    project = tmp_path / "proj"
    project.mkdir()
    (project / "calc.calc").write_text(SOURCE, encoding="utf-8")
    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_calc.calc").write_text(TESTS + "test wrong: add(1, 1) == 3\n",
                                          encoding="utf-8")
    settings = resolve_settings(config_files=[], base_dir=tmp_path, models={}, environment={})
    try:
        with config.use_run_settings(settings):
            yield languages.get_backend("calc"), project, tests
    finally:
        calc_language.unregister()


def test_a_declared_no_import_hint_replaces_the_starter_prompts_fallback(calc, monkeypatch):
    """A language whose tests need no import line declares a sentence for it
    (``import_statement``, as the tutorial's Toy does), which the starter
    prompt shows in place of the fallback that invites an import line."""
    from testuary import starters

    backend, project, _tests = calc
    target, root = str(project / "calc.calc"), str(project)
    assert starters._import_hint("calc", target, root) == (
        "follow the language's import conventions for the target")
    hint = "none; a calc test calls the target's functions by name, without an import line"
    monkeypatch.setattr(backend._adapter_class(), "import_statement",
                        classmethod(lambda cls, target, root: hint))
    assert starters._import_hint("calc", target, root) == hint


def test_a_registered_language_without_testuary_code_gets_the_generic_backend(calc):
    backend, project, tests = calc
    assert type(backend) is generic.GenericBackend
    assert backend.target_test_path(str(tests), str(project / "calc.calc")).name == "test_calc.calc"
    assert backend.is_test_file(tests / "test_calc.calc")
    assert not backend.is_test_file(project / "calc.calc")


def test_runs_coverage_and_the_test_units_go_through_the_public_operations(calc):
    backend, project, tests = calc
    suite = [str(tests / "test_calc.calc")]
    report = {}
    assert backend.failing_test_ids(suite, str(project), timeout=60, report=report) == [
        "test_calc::wrong"]
    assert report["outcome"] == "test_failures"
    run = backend.run_test_files(suite, str(project), timeout=60,
                                 measure_coverage_of=str(project / "calc.calc"))
    assert (run.passed, run.failed, run.line_coverage) == (3, 1, 100.0)
    source = (tests / "test_calc.calc").read_text(encoding="utf-8")
    assert backend.prune_tests(source, ["test_calc::wrong"]) == TESTS
    assert backend.test_metrics(source)["test_count"] == 4


def test_a_run_carries_its_diagnostics_and_reason_detail(calc, monkeypatch):
    """A run carries the diagnostics its language states, such as the
    coverage_counts_inconsistent diagnostic of a run without coverage, and
    the lines that explain its reason."""
    from dataclasses import replace
    from langadapter.runs import Outcome, RunDiagnostic

    backend, project, tests = calc
    own = calc_language.CalcAdapter.run_suite
    unread = (RunDiagnostic("coverage_counts_inconsistent", "negative count"),)
    monkeypatch.setattr(calc_language.CalcAdapter, "run_suite", lambda self, request: replace(
        own(self, request), outcome=Outcome.INFRASTRUCTURE_FAILURE, coverage=None,
        diagnostics=unread, reason_detail=("gcov: negative count",)))
    run = backend.run_test_files([str(tests / "test_calc.calc")], str(project), timeout=60,
                                 measure_coverage_of=str(project / "calc.calc"))
    assert (run.ok, run.line_coverage, run.diagnostics, run.reason_detail) == (
        False, None, unread, ("gcov: negative count",))


def test_a_suite_that_does_not_load_reports_it(calc):
    backend, project, tests = calc
    (tests / "test_calc.calc").write_text("not a test\n", encoding="utf-8")
    report = {}
    assert backend.failing_test_ids([str(tests / "test_calc.calc")], str(project),
                                    timeout=60, report=report) is None
    assert report["outcome"] == "collection_failed"


def test_mutation_runs_through_the_languages_engine(calc):
    backend, project, tests = calc
    (tests / "test_calc.calc").write_text(TESTS, encoding="utf-8")
    runner = backend.mutation_runner({}, source_dir=str(project))
    measurement = runner.run_once(workers=1, target=project / "calc.calc", test_dir=tests,
                                  test_files=["test_calc.calc"], per_mutant_timeout=10,
                                  wall_clock_timeout=60)
    assert (measurement.killed, measurement.survived) == (2, 1)
    assert "double(x) = x + 2" in measurement.survivors_text()


def test_each_optional_family_takes_its_fallback(calc):
    backend, project, tests = calc
    source = TESTS
    assert backend.prune_unresolvable_imports(source, str(project)) == (source, [])
    assert backend.prune_collection_breakers(source, str(project), timeout=60) == (source, [])
    assert backend.strip_module_level_skips(source) == (source, 0)
    assert backend.merge_test_files(source, source) is None
    assert backend.resolve_test_collisions([source, source]).sources == (source, source)
    assert backend.normalize_shadowed_definitions(source).source == source
    assert backend.collect_eviction_data([str(tests / "test_calc.calc")], str(project),
                                         measure_coverage_of=str(project / "calc.calc"),
                                         timeout=60) is None
    identity = backend.generated_test_target_identity(str(project / "calc.calc"), str(project))
    assert identity.policy_version == generic.NO_VALIDITY_POLICY
    assert backend.analyze_generated_test_validity(source, identity) == ()
    delivered = backend.run_as_delivered([str(tests / "test_calc.calc")], str(project),
                                         timeout=60)
    assert delivered.failing is None and "no run as delivered" in delivered.reason
    timing = backend.benchmark_tests(source_dir=str(project),
                                     test_file_paths=[str(tests / "test_calc.calc")],
                                     warmup_runs=0, benchmark_runs=1, timeout=60)
    assert timing["unavailable"] and not timing["ok"]


def test_the_syntax_check_is_the_languages_validator(calc):
    backend, project, _tests = calc
    assert backend.validate_test_source(TESTS, str(project)).ok
    verdict = backend.validate_test_source("test x: 1 == 1\nbogus\n", str(project))
    assert (verdict.ok, verdict.status, verdict.line) == (False, "syntax_error", 2)


def test_the_prompt_words_come_from_the_languages_facts(calc):
    backend, project, _tests = calc
    from testuary.prompts import SLOTS
    words = backend.prompt_fragments(str(project))
    assert set(words) == set(SLOTS)
    assert words["lang_name"] == "Calc" and words["lang_test_framework"] == "calc tests"
    assert words["lang_coverage_test_file"] == "``test_<target_stem>.calc``"
    assert "NotImplemented" not in "".join(words.values())


def test_the_preflight_assesses_calc_at_level_1(calc):
    backend, project, _tests = calc
    assessment = profile.assess(backend.capabilities(str(project)))
    assert assessment.refusals == ()
    families = {f.family for f in assessment.applied}
    assert families >= {
        "kill attribution", "merging", "per-test evidence",
        "validity screening", "timing"}
    # Runs as delivered serve only the reverse-order note, off by default.
    assert "delivered runs" not in families


# ---- a whole run ---------------------------------------------------------------

FINAL_ARGS = {
    "emit_coverage_test_result": {
        "test_file_name": "test_calc.calc", "targeted_functions": ["double", "add", "sub"],
        "explanation": "The tests call each function with small integers and check results."},
    "emit_mutation_test_result": {
        "mutation_test_file_name": "test_calc.calc", "targeted_mutants": ["0", "1"],
        "explanation": "The tests call each function with small integers and check results."},
}
USAGE = {"input_tokens": 1, "output_tokens": 1, "total_tokens": 2}


class _AgentModel(BaseChatModel):
    """Reads the suite, measures it, adds one test, runs and measures again,
    then gives the final answer: each phase edits through the edit tool."""

    tools: tuple = ()

    @property
    def _llm_type(self) -> str:
        return "scripted"

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return _AgentModel(tools=tuple(getattr(t, "name", "") for t in tools))

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        done = [getattr(m, "name", "") for m in messages if getattr(m, "type", "") == "tool"]
        measure = next((n for n in ("measure_coverage", "measure_mutation")
                        if n in self.tools), None)
        final = next((n for n in self.tools if n in FINAL_ARGS), None)
        # The coverage edit covers sub, the mutation edit kills x * 2 -> x + 2.
        added = ("test subtracts: sub(5, 2) == 3" if final == "emit_coverage_test_result"
                 else "test doubles_three: double(3) == 6")
        plan = [("read_tests", {}), (measure, {}),
                ("edit_tests", {"old_text": "test doubles: double(2) == 4",
                                "new_text": "test doubles: double(2) == 4\n" + added}),
                ("run_tests", {}), (measure, {})]
        calls = []
        for index, (name, args) in enumerate(plan):
            if name and name in self.tools and done.count(name) < [
                    n for n, _ in plan[:index + 1]].count(name):
                calls = [{"name": name, "args": args, "id": f"{name}-{len(messages)}"}]
                break
        if not calls and final is not None:
            calls = [{"name": final, "args": FINAL_ARGS[final], "id": f"final-{len(messages)}"}]
        message = AIMessage(content="" if calls else "Done.", usage_metadata=USAGE,
                            tool_calls=calls)
        return ChatResult(generations=[ChatGeneration(message=message)])


class _ScriptedSource:
    def generate_suite(self, messages, *, temperature, seed, timeout):
        # The starter leaves sub uncovered and has one failing test.
        return ("```text\ntest doubles: double(2) == 4\ntest adds: add(1, 2) == 3\n"
                "test wrong: add(1, 1) == 3\n```\n", dict(USAGE))

    def agent_chat_model(self, phase):
        return _AgentModel()


def test_a_whole_run_of_a_level_1_language_delivers_a_measured_suite(tmp_path, monkeypatch):
    from testuary import __main__ as cli

    calc_language.register()
    try:
        project = tmp_path / "proj"
        project.mkdir()
        (project / "calc.calc").write_text(SOURCE, encoding="utf-8")
        (tmp_path / "models.ini").write_text(
            "[scripted]\nprovider = ollama\nmodel = scripted\n", encoding="utf-8")
        (project / ".testuary").mkdir()
        (project / ".testuary" / "project.ini").write_text("[project]\nlanguage = calc\n",
                                                           encoding="utf-8")
        # The small first run's limits, as the getting-started guide shows them.
        limits = ("max_requests_per_phase = 20\nmax_input_tokens_per_phase = 300000\n"
                  "max_output_tokens_per_phase = 200000\nmax_wall_clock_seconds_per_phase = 1800\n")
        (tmp_path / "config.ini").write_text(
            "[project]\n[starter]\nsuite_count = 3\nattempts_per_suite = 2\n"
            "[coverage]\ngoal = 90\ngoal_metric = line\n" + limits
            + "[mutation]\ngoal = 80\n" + limits, encoding="utf-8")
        monkeypatch.chdir(project)
        monkeypatch.setenv("TESTUARY_HOME", str(tmp_path))
        # Zero repetitions are refused, and Calc has no timing benchmark anyway.
        monkeypatch.setenv("TESTUARY_TIMING_RUN_COUNT", "1")
        monkeypatch.setattr(sys, "argv", [
            "testuary", "--config", str(tmp_path / "config.ini"), "--model", "scripted",
            "--project-name", "calcproj", "calc.calc",
            "--output-dir", str(tmp_path / "out"),
            "--no-color"])
        with model_source.installed(_ScriptedSource()):
            assert cli.run_command() == 0
    finally:
        calc_language.unregister()
    results = tmp_path / "out" / "results"
    delivered = sorted(results.rglob("test_calc.calc"))
    assert delivered, "no delivered suite"
    text = delivered[-1].read_text(encoding="utf-8")
    assert "test doubles" in text and "test wrong" not in text
    # Both agents' edits went through the edit tool into the delivered suite.
    assert "test subtracts" in text and "test doubles_three" in text
    (preset_path,) = results.glob("*/r0/resolved_settings.json")
    preset = json.loads(preset_path.read_text(encoding="utf-8"))
    families = {item["family"] for item in preset["fallbacks"]}
    assert {"merging", "timing", "validity screening"} <= families
    assert "delivered runs" not in families
    rows = (results / "tests.csv").read_text(encoding="utf-8")
    assert "100" in rows



def _whole_run(tmp_path, monkeypatch, *options, entry=None, source=None):
    """Exit status of one scripted Calc run over ``tmp_path``'s project with
    the guide's small limits, through ``entry`` (``run_command`` by default),
    with ``source`` as its model (``_ScriptedSource`` by default)."""
    from testuary import __main__ as cli

    calc_language.register()
    try:
        project = tmp_path / "proj"
        (project / ".testuary").mkdir(parents=True, exist_ok=True)
        (project / "calc.calc").write_text(SOURCE, encoding="utf-8")
        (tmp_path / "models.ini").write_text(
            "[scripted]\nprovider = ollama\nmodel = scripted\n", encoding="utf-8")
        (project / ".testuary" / "project.ini").write_text("[project]\nlanguage = calc\n",
                                                           encoding="utf-8")
        limits = ("max_requests_per_phase = 20\nmax_input_tokens_per_phase = 300000\n"
                  "max_output_tokens_per_phase = 200000\nmax_wall_clock_seconds_per_phase = 1800\n")
        (tmp_path / "config.ini").write_text(
            "[project]\n[starter]\nsuite_count = 3\nattempts_per_suite = 2\n"
            "[coverage]\ngoal = 90\ngoal_metric = line\n" + limits
            + "[mutation]\ngoal = 80\n" + limits, encoding="utf-8")
        monkeypatch.chdir(project)
        monkeypatch.setenv("TESTUARY_HOME", str(tmp_path))
        monkeypatch.setenv("TESTUARY_TIMING_RUN_COUNT", "1")
        monkeypatch.setattr(sys, "argv", [
            "testuary", "--config", str(tmp_path / "config.ini"), "--model", "scripted",
            "--project-name", "calcproj", "calc.calc", "--output-dir", str(tmp_path / "out"),
            "--no-color", *options])
        with model_source.installed(source or _ScriptedSource()):
            return (entry or cli.run_command)()
    finally:
        calc_language.unregister()


class _DifferentSeeds(_ScriptedSource):
    """Each starter suite is another file: Calc can neither merge them nor
    give per-test evidence."""

    SUITES = ["test doubles: double(2) == 4\ntest adds: add(1, 2) == 3\n",
              "test doubles: double(2) == 4\ntest minus: sub(5, 2) == 3\n",
              "test doubles: double(2) == 4\n"]

    def __init__(self):
        import threading

        self.lock, self.given = threading.Lock(), []

    def generate_suite(self, messages, *, temperature, seed, timeout):
        with self.lock:
            suite = self.SUITES[len(self.given) % len(self.SUITES)]
            self.given.append(suite)
        return f"```text\n{suite}```\n", dict(USAGE)


def test_a_level_1_language_keeps_its_best_starter_suite_from_different_seeds(
        tmp_path, monkeypatch):
    """Without per-test evidence each starter suite is scored on its own
    coverage run, so the best one, unreduced, is delivered."""
    source = _DifferentSeeds()
    assert _whole_run(tmp_path, monkeypatch, "--stop-after-phase", "starter",
                      source=source) == 0
    assert set(source.given) == set(_DifferentSeeds.SUITES)
    (archived,) = (tmp_path / "out" / "results").glob("*/r0/archive/starter/result.calc")
    # Either two-test suite covers two of the three lines, and the one-test suite one.
    assert archived.read_text(encoding="utf-8") in _DifferentSeeds.SUITES[:2]


def test_a_stop_while_a_run_opens_its_time_record_ends_as_a_stop(tmp_path, monkeypatch, capsys):
    """A stop (Ctrl+C or a signal) during the save of a run's opening time
    record ends the run as stopped (exit status 130), not as a crash."""
    from testuary import __main__ as cli
    from testuary import exits
    from testuary import phase_state

    real_open, real_save = phase_state.open_segment, phase_state.save
    armed = {"now": False, "fired": False}

    def open_segment(target_dir, phase, **kwargs):
        armed["now"] = phase == phase_state.SEGMENT_PIPELINE and not armed["fired"]
        try:
            return real_open(target_dir, phase, **kwargs)
        finally:
            armed["now"] = False

    def save(*args, **kwargs):
        if armed["now"]:
            armed["fired"] = True
            raise KeyboardInterrupt
        return real_save(*args, **kwargs)

    monkeypatch.setattr(phase_state, "open_segment", open_segment)
    monkeypatch.setattr(phase_state, "save", save)
    assert _whole_run(tmp_path, monkeypatch, entry=cli.main) == exits.INTERRUPTED == 130
    assert armed["fired"]
    logs = "".join(path.read_text(encoding="utf-8", errors="replace")
                   for path in (tmp_path / "out").rglob("run.log"))
    shown = capsys.readouterr()
    assert "crashed" not in logs + shown.out + shown.err
    assert "TypeError" not in logs + shown.out + shown.err


def test_a_reused_starter_phase_hands_its_suite_to_the_coverage_loop(tmp_path, monkeypatch):
    from testuary import target_phases as phases

    assert _whole_run(tmp_path, monkeypatch, "--stop-after-phase", "starter") == 0
    results = tmp_path / "out" / "results"
    (archived,) = results.glob("*/r0/archive/starter/result.calc")
    starter = archived.read_bytes()
    entered = []
    enter = phases._phase_coverage

    def spy(ctx, result, baseline):
        entered.append((Path(result.starter_result.baseline_file), Path(ctx.work_tests)))
        return enter(ctx, result, baseline)

    monkeypatch.setattr(phases, "_phase_coverage", spy)
    assert _whole_run(tmp_path, monkeypatch, "--reuse", "starter") == 0
    # The loop edits the starter suite in the work directory. The starter
    # phase's archive stays as delivered.
    [(baseline_file, work_tests)] = entered
    assert baseline_file.parent == work_tests
    assert archived.read_bytes() == starter
    with open(results / "tests.csv", encoding="utf-8-sig", newline="") as handle:
        [row] = list(csv.DictReader(handle))
    assert row["coverage_status"] == "success"
    delivered = sorted(results.rglob("test_calc.calc"))
    assert "test subtracts" in delivered[-1].read_text(encoding="utf-8")


class _CountingSource(_ScriptedSource):
    """The scripted model, noting each phase that asks it."""

    def __init__(self):
        self.asked = []

    def generate_suite(self, messages, *, temperature, seed, timeout):
        self.asked.append("starter")
        return super().generate_suite(messages, temperature=temperature, seed=seed,
                                      timeout=timeout)

    def agent_chat_model(self, phase):
        self.asked.append(phase)
        return super().agent_chat_model(phase)


def _a_run_whose_mutation_loop_changed_the_suite(tmp_path, monkeypatch):
    """The delivered suite and the coverage phase's own suite after a whole
    run, in which the mutation phase added a test."""
    assert _whole_run(tmp_path, monkeypatch) == 0
    results = tmp_path / "out" / "results"
    (coverage_suite,) = results.glob("*/r0/archive/coverage/result.calc")
    (delivered,) = results.glob("*/r0/tests/test_calc.calc")
    assert "test doubles_three" in delivered.read_text(encoding="utf-8")
    assert "test doubles_three" not in coverage_suite.read_text(encoding="utf-8")
    return delivered, coverage_suite


def test_a_coverage_reuse_starts_the_mutation_loop_from_the_coverage_loops_suite(
        tmp_path, monkeypatch):
    # The coverage phase's validation record names its own suite, not the
    # one the mutation phase delivered.
    delivered, _coverage_suite = _a_run_whose_mutation_loop_changed_the_suite(
        tmp_path, monkeypatch)
    source = _CountingSource()
    assert _whole_run(tmp_path, monkeypatch, "--reuse", "coverage", source=source) == 0
    assert source.asked and set(source.asked) == {"mutation"}
    assert "test doubles_three" in delivered.read_text(encoding="utf-8")


@pytest.mark.parametrize("stop", ["_phase_mutation", "_consolidate_mutation"])
def test_a_resume_after_the_coverage_loop_starts_the_mutation_loop_from_its_suite(
        stop, tmp_path, monkeypatch):
    # A resume reuses the completed coverage phase as a coverage reuse does.
    # The preserved suite it restores is the coverage phase's snapshot, so the
    # mutation phase starts from the same bytes whichever of them it takes,
    # whether the loop stopped before the model's edit or after it.
    from testuary import target_phases as phases

    resumed = getattr(phases, stop)

    def interrupted(*args, **kwargs):
        raise RuntimeError("the mutation phase stopped")

    monkeypatch.setattr(phases, stop, interrupted)
    assert _whole_run(tmp_path, monkeypatch) == 1
    monkeypatch.setattr(phases, stop, resumed)
    results = tmp_path / "out" / "results"
    (snapshot,) = results.glob("*/r0/archive/coverage/result.calc")
    (preserved,) = results.glob("*/r0/tests/test_calc.calc")
    suite = snapshot.read_bytes()
    assert preserved.read_bytes() == suite
    entered = []
    run_mutation = phases._phase_mutation

    def spy(ctx, result, session):
        entered.append((Path(ctx.work_tests) / "test_calc.calc").read_bytes())
        return run_mutation(ctx, result, session)

    monkeypatch.setattr(phases, "_phase_mutation", spy)
    source = _CountingSource()
    assert _whole_run(tmp_path, monkeypatch, "--resume", source=source) == 0
    assert entered == [suite]
    assert source.asked and set(source.asked) == {"mutation"}


def test_a_resume_whose_mutation_records_fail_reruns_the_loop_from_the_coverage_suite(
        tmp_path, monkeypatch):
    # A completed mutation phase moves the coverage phase's record onto the
    # suite it kept, while tests/ holds the delivered one. When only the
    # mutation phase's records no longer hold, a resume reruns that loop from
    # the coverage phase's own suite, not from the delivered one.
    from testuary import target_phases as phases

    delivered, coverage_suite = _a_run_whose_mutation_loop_changed_the_suite(
        tmp_path, monkeypatch)
    (mutation_snapshot,) = (tmp_path / "out" / "results").glob(
        "*/r0/archive/mutation/result.calc")
    mutation_snapshot.unlink()
    entered = []
    run_mutation = phases._phase_mutation

    def spy(ctx, result, session):
        entered.append((Path(ctx.work_tests) / "test_calc.calc").read_bytes())
        return run_mutation(ctx, result, session)

    monkeypatch.setattr(phases, "_phase_mutation", spy)
    assert _whole_run(tmp_path, monkeypatch, "--resume") == 0
    assert entered == [coverage_suite.read_bytes()]
    assert "test doubles_three" in delivered.read_text(encoding="utf-8")


def test_a_mutation_reuse_reuses_the_phases_it_builds_on(tmp_path, monkeypatch):
    delivered, _coverage_suite = _a_run_whose_mutation_loop_changed_the_suite(
        tmp_path, monkeypatch)
    suite = delivered.read_bytes()
    source = _CountingSource()
    assert _whole_run(tmp_path, monkeypatch, "--reuse", "mutation", source=source) == 0
    assert source.asked == []
    assert delivered.read_bytes() == suite


def test_the_languages_own_test_vocabulary_fills_its_slots(calc, monkeypatch):
    backend, project, _tests = calc
    monkeypatch.setattr(calc_language.CalcAdapter, "test_vocabulary", lambda self: {
        "test_file": "``test_<stem>.calc``", "timeout_marker": "``timeout({seconds})``",
        "load_error": "parse failure"}, raising=False)
    words = backend.prompt_fragments(str(project))
    assert words["lang_coverage_test_file"] == words["lang_mutation_test_file"] == \
        "``test_<stem>.calc``"
    assert words["lang_timeout_decorator"] == "``timeout(120)``"
    assert words["lang_collection_error"] == "parse failure"
    assert words["lang_valid_source"] == "valid Calc"


def test_only_the_delivered_name_counts_as_the_targets_file(calc):
    backend, project, tests = calc
    assert backend.classify_test_kind("test_calc.calc") == "coverage"
    assert backend.classify_test_kind("calc_test.calc") == "unknown"
    (tests / "test_failed.py").write_text("", encoding="utf-8")
    assert backend.find_target_test_file(str(tests), str(project / "other.calc")) == (
        tests / "test_calc.calc")


def test_check_accepts_a_level_1_language_and_names_its_fallbacks(tmp_path, monkeypatch, capsys):
    from testuary import __main__ as cli

    calc_language.register()
    try:
        (tmp_path / "proj").mkdir()
        (tmp_path / "proj" / "calc.calc").write_text(SOURCE, encoding="utf-8")
        (tmp_path / "models.ini").write_text(
            "[scripted]\nprovider = ollama\nmodel = scripted\n", encoding="utf-8")
        (tmp_path / "proj" / ".testuary").mkdir()
        (tmp_path / "proj" / ".testuary" / "project.ini").write_text(
            "[project]\nlanguage = calc\nsource_files = calc.calc\n", encoding="utf-8")
        (tmp_path / "config.ini").write_text("[project]\n", encoding="utf-8")
        monkeypatch.chdir(tmp_path / "proj")
        monkeypatch.setenv("TESTUARY_HOME", str(tmp_path))
        monkeypatch.setenv("TESTUARY_TIMING_RUN_COUNT", "1")
        monkeypatch.setattr(sys, "argv", [
            "testuary", "check", "--config", str(tmp_path / "config.ini"), "--model", "scripted",
            "--project-name", "calcproj", "--no-color"])
        assert cli.run_command() == 0
    finally:
        calc_language.unregister()
    captured = capsys.readouterr()
    out = captured.out + captured.err
    # Runs as delivered serve only the reverse-order note, off here.
    assert "the reverse-order note is off" not in out
    assert "The language adapter lacks merging" in out
    assert "This changes Testuary's method." in out
    assert "which Testuary needs" not in out


def test_the_generation_condition_records_the_language_package(calc):
    """A language from another package records its distribution and version in
    the generation condition, a built-in one records nothing. An unknown version
    or another package refuses a resume."""
    from testuary.evidence import language_package, reuse_problems

    assert language_package("python") is None and language_package("java") is None
    package = language_package("calc")
    assert package == {"distribution": None, "module": "calc_language", "version": None}

    from testuary.evidence import GenerationCondition

    def current(language_package):
        return GenerationCondition(
            "coverage", "calc", {"protocol": "openai", "extension_version": None}, {},
            "composition", True, True, language_package=language_package)

    recorded = {"format": 1, "service_identity_known": True, "request_options_known": True,
                "language_package": package}
    assert "the language package's version is unknown" in reuse_problems(recorded, current(package))
    changed = {**package, "distribution": "calc", "version": "1.0"}
    assert "the language package changed" in reuse_problems(recorded, current(changed))


def test_a_runtime_version_that_cannot_be_read_is_named_unknown(monkeypatch):
    """A runtime version the adapter cannot read shows as ``Python (unknown)``
    in the prompts. A language that reads no version is named alone."""
    from types import SimpleNamespace

    from langadapter.python.adapter import PythonAdapter
    from testuary import prompts as vocabulary

    settings = resolve_settings(config_files=[], base_dir=".", models={}, environment={})
    with config.use_run_settings(settings):
        assert vocabulary.runtime_line(SimpleNamespace(runtime_version=None),
                                       "Java") == "Java (unknown)"
        assert vocabulary.runtime_line(None, "Calc") == "Calc (unknown)"
        assert vocabulary.runtime_line(None, "Toy", reads_version=False) == "Toy"
        assert vocabulary.runtime_line(SimpleNamespace(runtime_version="21"),
                                       "Java") == "Java 21"
        # Only the private version reader makes the adapter report an unknown version without a process.
        monkeypatch.setattr(PythonAdapter, "_runtime_line", lambda self: None)
        words = languages.get_backend("python").prompt_fragments()
    assert words["lang_runtime_version"] == "Python (unknown)"


def test_the_failing_test_listing_names_a_parametrized_case(tmp_path, default_run):
    """A parametrized test's failing case is listed with its id, the form
    prune_tests removes alone."""
    from testuary.languages import get_backend

    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    suite = root / "test_cases.py"
    suite.write_text(
        "import pytest\nfrom calc import add\n\n"
        "@pytest.mark.parametrize(\"a,b,total\", [(1, 1, 2), (2, 2, 5)])\n"
        "def test_add(a, b, total):\n    assert add(a, b) == total\n\n"
        "def test_wrong():\n    assert add(1, 1) == 3\n", encoding="utf-8")
    ids = get_backend("python").failing_test_ids([str(suite)], str(root), timeout=120)
    assert ids == ["test_cases.test_add[2-2-5]", "test_cases.test_wrong"]


@pytest.mark.slow
def test_a_listing_the_timeout_ended_names_the_running_test(tmp_path, default_run):
    """A run the timeout ended reports the test the timeout journal names
    as running then. The run waits out its 20 s timeout, so the test is
    slow."""
    from testuary.languages import get_backend

    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    suite = root / "test_hang.py"
    suite.write_text(
        "import time\nfrom calc import add\n\n"
        "def test_ok():\n    assert add(1, 1) == 2\n\n"
        "def test_hangs():\n    time.sleep(600)\n", encoding="utf-8")
    report = {}
    ids = get_backend("python").failing_test_ids(
        [str(suite)], str(root), timeout=20, instrument_target=str(root / "calc.py"),
        report=report)
    assert ids is None
    assert report["outcome"] == "timed_out"
    assert report["active_at_end"] == "test_hang.test_hangs"



def test_a_case_id_brackets_python_and_java_parameters_once():
    """Python's case parameters come without brackets, Java's with them:
    each listing id and journal name carries one pair."""
    from types import SimpleNamespace

    from testuary.languages import generic

    adapter = SimpleNamespace(test_id_text=lambda test: "pkg.CalcTest.adds")
    assert generic.case_id(adapter, SimpleNamespace(params="[2]")) == "pkg.CalcTest.adds[2]"
    assert generic.case_id(adapter, SimpleNamespace(params="1-2")) == "pkg.CalcTest.adds[1-2]"
    assert generic.case_id(adapter, SimpleNamespace(params=None)) == "pkg.CalcTest.adds"
    assert generic.case_suffix("[2]") == "[2]"


def test_facts_are_read_from_the_class_a_factory_function_returns(calc):
    """A plain-function factory's facts, the timeout journal among them, are
    read from the class of the adapter it returns."""
    import langadapter as la

    class Declining(calc_language.CalcAdapter):
        provides_timeout_journal = False
        test_file_name = "{stem}_check{ext}"

    la.register("calc", [".calc"], lambda context: Declining(context), replace=True)
    backend = languages.get_backend("calc")
    assert backend.offers_timeout_journal() is False
    assert backend.target_test_path("tests", "calc.calc").name == "calc_check.calc"


@pytest.mark.parametrize("declared, offered", [(None, True), (True, True), (False, False)])
def test_the_timeout_journal_is_asked_where_the_adapter_provides_it(calc, monkeypatch, declared, offered):
    """An adapter that does not provide the journal (the tutorial's Toy)
    refuses a run that asks for it, so a listing would give no outcome."""
    adapter = type("Adapter", (), {} if declared is None else {"provides_timeout_journal": declared})
    backend = languages.get_backend("calc")
    monkeypatch.setattr(type(backend), "project_backend", lambda self, source_dir=None: adapter())
    assert backend.offers_timeout_journal() is offered



@pytest.mark.parametrize("declared, seeded", [(None, False), (True, True), (False, False)])
def test_a_run_as_delivered_is_seeded_where_the_adapter_states_it(calc, monkeypatch, declared,
                                                                   seeded):
    """Where the adapter states seeds_delivered_runs (Python's), its run as
    delivered takes a session seed. One that states nothing takes none."""
    adapter = type("Adapter", (), {} if declared is None else {"seeds_delivered_runs": declared})
    backend = languages.get_backend("calc")
    monkeypatch.setattr(type(backend), "project_backend", lambda self, source_dir=None: adapter())
    assert backend.seeds_delivered_runs() is seeded


def test_the_journal_and_the_seed_are_read_on_the_projects_adapter(calc, monkeypatch):
    """An adapter that states both facts on its instance (C's tests in
    Python, ``extension_mode``) is read there, not on its class."""
    class Instance:
        provides_timeout_journal = False
        seeds_delivered_runs = False

        def __init__(self):
            self.provides_timeout_journal = True
            self.seeds_delivered_runs = True

    backend = languages.get_backend("calc")
    project = Instance()
    monkeypatch.setattr(type(backend), "_adapter_class", lambda self: Instance)
    monkeypatch.setattr(type(backend), "project_backend", lambda self, source_dir=None: project)
    assert backend.offers_timeout_journal() is True
    assert backend.seeds_delivered_runs() is True


@pytest.mark.parametrize("init_text", [None, "", "from . import mod\n"])
def test_a_src_layouts_import_line_is_named_from_its_import_root(
        tmp_path, monkeypatch, in_run, init_text):
    """With [project] python_import_root = src and the project directory as
    the root, the import line is the one the module had when src was the
    root, the module named pkg.mod as the tests import it."""
    project = tmp_path / "project"
    (project / "src" / "pkg").mkdir(parents=True)
    if init_text is not None:
        (project / "src" / "pkg" / "__init__.py").write_text(init_text, encoding="utf-8")
    target = project / "src" / "pkg" / "mod.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    monkeypatch.setattr(config, "get_project_config", lambda: {"python_import_root": "src"})
    in_run()
    backend = languages.get_backend("python")
    line = backend.import_statement(str(target), str(project))
    assert line == backend._adapter_class().import_statement(str(target), str(project / "src"))
    assert "src" not in line and "pkg.mod" in line
