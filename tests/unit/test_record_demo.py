"""The demonstration recorder (tools/record_demo.py): machine paths as
placeholders, the recording oracle that answers with the real operation,
its replay mode, whose model requests are answered from an earlier
recording by key so no model is contacted, and the scenario file it
writes. Each replay recording runs in a process of its own
(_record_demo_driver.py), since a recording must start before any phase
module loads."""
from __future__ import annotations

import contextlib
import gzip
import importlib.util
import json
import os
import subprocess
import sys
from importlib import resources
from pathlib import Path

import pytest
from source_tree import requires_source_tree

from langchain_core.messages import AIMessage, HumanMessage

from testuary.demo import recorded
from testuary.demo import scenario as sc
from testuary.demo.scenario import (Call, Entry, PathMap, Scenario, ScenarioStop, install_paths,
                                    local_paths, neutral_paths)

ROOT = Path(__file__).resolve().parents[2]
_spec = importlib.util.spec_from_file_location("record_demo", ROOT / "tools" / "record_demo.py")
record_demo = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(record_demo)

pytestmark = requires_source_tree
TOY = resources.files("testuary.demo") / "data" / "toy"
DRIVER = Path(__file__).with_name("_record_demo_driver.py")


def _entry(kind, key, answer, segment="starter", value_class="real",
           rule="recorded-kill-matrix-v1: x", parts=None):
    return sc.Entry(segment=segment, kind=kind, key=key, answer=answer,
                    value_class=value_class, source="src", rule=rule,
                    parts=parts or {})


def test_a_scenario_round_trips_with_long_strings_stored_once(tmp_path):
    long_text = "x" * 5000
    entries = [
        _entry("backend.a", {"n": 1}, {"text": long_text}),
        _entry("backend.a", {"n": 2}, {"text": long_text}),
        _entry("model.seed", {"seed": 1}, {"text": "short"},
               value_class="synthetic", rule="seed-reply-fenced-block-v1: x",
               parts={"usage": {"class": "real", "source": "s",
                                "rule": "recorded-seed-usage-v1: y"}}),
    ]
    path = tmp_path / "scenario.json.gz"
    record_demo.dump(sc.Scenario(entries=entries, meta={"exemplar": "e"}), path)
    raw = json.loads(gzip.open(path).read())
    assert len(raw["blobs"]) == 1
    loaded = sc.Scenario.load(path)
    assert loaded.meta == {"exemplar": "e"}
    by_key = {json.dumps(e.key, sort_keys=True): e for e in loaded.entries}
    assert by_key['{"n": 1}'].answer == {"text": long_text}
    seed = by_key['{"seed": 1}']
    assert seed.value_class == "synthetic" and seed.parts["usage"]["class"] == "real"


def test_the_file_depends_only_on_the_answers(tmp_path):
    """The file's bytes do not depend on the order of the answers."""
    first = [_entry("model.seed", {"seed": s}, {"text": str(s)})
             for s in (3, 1, 2)]
    second = [first[1], first[2], first[0]]
    a, b = tmp_path / "a.json.gz", tmp_path / "b.json.gz"
    record_demo.dump(sc.Scenario(entries=first), a)
    record_demo.dump(sc.Scenario(entries=second), b)
    assert a.read_bytes() == b.read_bytes()


LINUX = {"out": "/home/rec/demo-out", "python": "/home/rec/env/bin/python",
         "site": "/home/rec/env/lib/python3.12/site-packages", "tmp": "/tmp",
         "home": "/home/rec"}
WINDOWS = {"out": "C:\\Users\\viewer\\demo-python", "python": "C:\\Py\\python.exe",
           "site": "C:\\Py\\Lib\\site-packages", "tmp": "C:\\Temp", "home": "C:\\Users\\viewer"}


def test_a_recording_stores_its_paths_as_placeholders():
    text = ("FAILED /home/rec/demo-out/run/_work/tests/test_stop.py::test_a - x\n"
            '  File "/home/rec/env/lib/python3.12/site-packages/tenacity/stop.py", line 3\n'
            "interpreter /home/rec/env/bin/python, cache /home/rec/.cache/x, /tmp/abc/d.txt")
    stored = PathMap(LINUX).neutral(text)
    assert "/home/rec" not in stored and "/tmp/" not in stored
    assert "{testuary:out}/run/_work/tests/test_stop.py::test_a" in stored
    assert '"{testuary:site}/tenacity/stop.py"' in stored


def test_a_playback_names_its_own_dirs_with_its_own_separators(monkeypatch):
    stored = PathMap(LINUX).neutral("see /home/rec/demo-out/run/tests/test_a.py:12")
    monkeypatch.setattr(os, "sep", "\\")
    windows = PathMap(WINDOWS)
    shown = windows.local(stored)
    assert shown == "see C:\\Users\\viewer\\demo-python\\run\\tests\\test_a.py:12"
    # Keys see the placeholders again, so they match the recording's.
    assert windows.neutral(shown) == stored
    assert windows.neutral("'C:\\\\Users\\\\viewer\\\\demo-python\\\\run\\\\x.py'") == \
        "'{testuary:out}/run/x.py'"


def test_a_path_is_replaced_only_as_a_whole_dir_name():
    assert PathMap({"tmp": "/tmp"}).neutral("/tmpfile /tmp/x /tmp") == \
        "/tmpfile {testuary:tmp}/x {testuary:tmp}"


def test_without_a_path_map_nothing_changes():
    previous = install_paths(None)
    try:
        value = {"a": ["/home/rec/x", 1]}
        assert neutral_paths(value) == value and local_paths(value) == value
    finally:
        install_paths(previous)


def test_the_run_label_is_named_once():
    """The label given is the recorded run's source as it is, never behind a
    second prefix. Without one, it names the release."""
    from testuary import __version__
    label = f"recorded run of Testuary {__version__} on calc.toy, 2026-10-20"
    assert record_demo.run_source(label) == label
    assert record_demo.run_source(None) == f"recorded run of Testuary {__version__}"


def test_a_recording_names_the_rule_the_shipped_recordings_name():
    """Every recorded entry carries the recorder's rule, so a change of its
    text changes every answer of a re-recording."""
    data = resources.files("testuary.demo") / "data"
    for data_dir in (data, data / "java", data / "toy", data / "tutorial"):
        path = Path(str(data_dir / "scenario.json.gz"))
        assert {entry.rule for entry in Scenario.load(path).entries} == {record_demo.RULE}, path


class _Owner:
    """The owner of a declared call, its own real counterpart: its operation
    returns ``answer``, or calls it."""

    def __init__(self, oracle, answer):
        self.oracle, self.live, self.answer = oracle, self, answer

    def ask(self, call):
        return self.oracle.answer(call)

    def operate(self):
        return self.answer() if callable(self.answer) else self.answer


def _ask(oracle, kind, key, answer):
    """What the run receives for a declared call keyed ``key`` whose real
    operation returns ``answer``."""
    entry = recorded.RecordedCall(kind, "operation", key=lambda a: key,
                                  read=lambda stored, a: stored,
                                  write=lambda result, a, collected: result, name="operate")
    return recorded.playback(entry, _Owner.operate)(_Owner(oracle, answer))


def test_the_recording_oracle_answers_live_and_stores_placeholders():
    previous = install_paths(PathMap(LINUX))
    try:
        oracle = record_demo.RecordingOracle(source="recorded run of a test")
        answer = _ask(oracle, "backend.run_test_files", {"files": []},
                      {"result": {"output": "/home/rec/demo-out/run/tests/test_a.py"}})
    finally:
        install_paths(previous)
    # The run receives the real answer, the scenario the placeholder.
    assert answer == {"result": {"output": "/home/rec/demo-out/run/tests/test_a.py"}}
    (entry,) = oracle.entries
    assert entry.answer == {"result": {"output": "{testuary:out}/run/tests/test_a.py"}}
    assert (entry.value_class, entry.segment) == ("real", "starter")
    assert entry.rule.startswith("recorded-v1")


def test_a_call_inside_a_live_operation_is_not_recorded():
    oracle = record_demo.RecordingOracle(source="recorded run of a test")

    def outer():
        inner = _ask(oracle, "backend.failing_test_ids", {}, {"ids": []})
        return {"data": None, "inner": inner}

    assert _ask(oracle, "backend.collect_eviction_data", {}, outer) == {
        "data": None, "inner": {"ids": []}}
    assert [entry.kind for entry in oracle.entries] == ["backend.collect_eviction_data"]


def test_a_call_without_a_real_operation_stops_the_recording():
    oracle = record_demo.RecordingOracle(source="recorded run of a test")
    with pytest.raises(ScenarioStop, match="no real operation"):
        oracle.answer(Call("backend.unknown", {}))
    assert oracle.entries == []


def test_a_recorded_call_is_answered_by_the_real_method_with_the_run_arguments(monkeypatch):
    """The recorder calls the method the declaration names with the run's own
    arguments. It collects the progress lines and stores them with the
    result, and the run still receives them."""
    from testuary.languages.generic import GenericBackend

    seen = {}

    def benchmark(self, *, source_dir, test_file_paths, warmup_runs, benchmark_runs,
                  timeout, suite_label="delivered", progress_fn=None, instrument_target=None):
        seen.update(source_dir=source_dir, timeout=timeout, suite_label=suite_label)
        progress_fn("rep 1")
        return {"ok": True, "times_ms": [1.5]}

    monkeypatch.setattr(GenericBackend, "benchmark_tests", benchmark)
    oracle = record_demo.RecordingOracle(source="recorded run of a test")
    shown = []
    stats = record_demo.RecordingBackend(oracle).benchmark_tests(
        source_dir="project", test_file_paths=[], warmup_runs=1, benchmark_runs=2,
        timeout=60, suite_label="suite", progress_fn=shown.append)
    assert stats == {"ok": True, "times_ms": [1.5]} and shown == ["rep 1"]
    assert seen == {"source_dir": "project", "timeout": 60, "suite_label": "suite"}
    (entry,) = oracle.entries
    assert entry.kind == "backend.benchmark_tests"
    assert entry.key == {**sc.suite_files([]), "warmup_runs": 1, "timing_run_count": 2,
                         "suite_label": "suite", "instrumented": False}
    assert entry.answer == {"stats": {"ok": True, "times_ms": [1.5]}, "progress": ["rep 1"]}


class _RealMeasurement:
    killed, survived, timeout, non_viable, untested, total = 1, 1, 0, 0, 0, 2
    score, truncated = 50.0, False
    reason, suite_failed, stopping_tests, lower_bound = "", False, (), False

    def survivors_text(self, max_mutants=10, max_chars=240):
        return f"survivors {max_mutants} {max_chars}"

    def timeout_rule(self):
        return None


class _RealSession:
    def __init__(self):
        self.entered = []
        self.closed = False

    @contextlib.contextmanager
    def worker(self):
        self.entered.append(True)
        yield

    def run(self, **arguments):
        self.arguments = arguments
        return _RealMeasurement()

    def close(self):
        self.closed = True


class _RealRunner:
    def __init__(self):
        self.session = _RealSession()

    def open(self, workers, *, suite_dirs):
        return self.session


def test_a_recorded_mutation_run_answers_its_survivors_text_from_the_real_measurement(tmp_path):
    """A recording's session borrows the real session's project for a tool
    call, as a real run's call does, and the survivors' text of a run comes
    from the real measurement of that run. A playback has no real session."""
    from testuary.demo import backend as playback

    tests = tmp_path / "tests"
    tests.mkdir()
    (tests / "test_x.py").write_text("def test_a():\n    pass\n", encoding="utf-8")
    target = tmp_path / "mod.py"
    target.write_text("x = 1\n", encoding="utf-8")
    oracle = record_demo.RecordingOracle(source="recorded run of a test")
    live = _RealRunner()
    session = record_demo.RecordingRunner(oracle, {"cosmic_ray_operators": ""}, live=live).open(2)
    with session.worker():
        measurement = session.run(target=target, test_dir=tests,
                                  test_files=[str(tests / "test_x.py")],
                                  per_mutant_timeout=30, wall_clock_timeout=3600)
    assert live.session.entered == [True]
    assert live.session.arguments["per_mutant_timeout"] == 30
    assert (measurement.killed, measurement.survived) == (1, 1)
    assert measurement.survivors_text(max_mutants=5, max_chars=80) == "survivors 5 80"
    session.close()
    assert live.session.closed
    run, text = oracle.entries
    assert text.key == {"run": recorded.run_key(run.key), "max_mutants": 5, "max_chars": 80}
    assert text.answer == {"text": "survivors 5 80"}
    with playback.ScenarioMutationSession(playback.ScenarioRunner(object(), {}), 1,
                                          False).worker():
        pass


class _PythonProject:
    """A project's Python adapter whose facts and vocabulary say whether
    they were asked."""

    def __init__(self, asked):
        self.asked = asked

    def language_facts(self):
        from langadapter.capabilities import LanguageFacts

        self.asked.append("language_facts")
        return LanguageFacts(
            language="Python", test_framework="pytest", code_fence="python",
            runtime_version="Python 3.12.3", test_file_patterns=("test_*.py",),
            test_function_pattern="test_*", test_class_pattern="Test*",
            timeout_marker=None)

    def test_vocabulary(self):
        self.asked.append("test_vocabulary")
        return {"timeout_marker": "", "test_dependencies": "pytest"}


@pytest.mark.usefixtures("default_run")
def test_a_toy_recording_builds_its_runner_over_the_real_one(tmp_path):
    """The recording backend of a language registered from outside builds its
    scenario runner with the language and the adapter, whose rules are those
    of the real runner it records."""
    from testuary.demo import backend as playback
    from testuary.demo import cli as demo_cli
    from testuary.settings import _defaults, phase_section

    manifest, _files = demo_cli.load_data("toy")
    config = dict(_defaults()[phase_section("mutation")])
    undo = playback.register("toy", manifest["file_extensions"], manifest["adapter"])
    try:
        oracle = record_demo.RecordingOracle(source="recorded run of a test")
        backend = record_demo.recording_backend("toy", manifest["file_extensions"])(oracle)
        runner = backend.mutation_runner(config, source_dir=str(tmp_path))
        assert runner.demo_language == "toy"
        assert runner.catalogue() == runner.live.catalogue()
        assert runner.configured_operators() == runner.live.configured_operators()
        assert runner.engine_sets_timeout() == runner.live.engine_sets_timeout()
    finally:
        undo()


@pytest.mark.usefixtures("default_run")
def test_a_recording_keeps_the_prompts_language_facts_and_the_playback_states_them(
        tmp_path, monkeypatch):
    """Reading the facts runs the project's interpreter, which a playback does
    not start, so the playback states the recorded facts."""
    from testuary import prompts as vocabulary
    from testuary.demo import backend as playback
    from testuary.languages.generic import GenericBackend

    asked = []
    monkeypatch.setattr(GenericBackend, "project_backend",
                        lambda self, source_dir=None: _PythonProject(asked))
    oracle = record_demo.RecordingOracle(source="recorded run of a test")
    kept = vocabulary.fragments(record_demo.RecordingBackend(oracle), str(tmp_path))
    assert asked and set(oracle.language_facts) == {str(tmp_path.resolve())}

    asked.clear()
    monkeypatch.setattr(playback, "RECORDED_LANGUAGE", oracle.language_facts)
    played = vocabulary.fragments(playback.ScenarioBackend(sc.PlaybackOracle(Scenario([]))),
                                  str(tmp_path))
    assert played == kept and asked == []


class _LiveChat:
    """A real chat model's stand-in: records the tools each binding holds
    and the tools each request was sent with."""

    def __init__(self, tools=(), sent=None):
        self.tools = list(tools)
        self.sent = [] if sent is None else sent

    def bind_tools(self, tools, **options):
        return _LiveChat(tools, self.sent)

    def invoke(self, messages, stop=None):
        self.sent.append(self.tools)
        return AIMessage(content="ok", id=f"r{len(self.sent)}")


class _LiveModel:
    def __init__(self):
        self.chat = _LiveChat()

    def agent_chat_model(self, phase):
        return self.chat


def test_each_binding_keeps_its_tools_in_a_recording():
    """A request carries the tools of the binding that sent it, not those
    bound last, and the requests of all bindings count as one sequence."""
    oracle = record_demo.RecordingOracle(source="recorded run of a test")
    live = _LiveModel()
    model = record_demo.RecordingModelSource(oracle, live).agent_chat_model("mutation")
    editing = model.bind_tools(["read_tests", "edit_tests"])
    reading = model.bind_tools(["read_tests"])
    request = [HumanMessage(content="go")]
    assert editing.invoke(request).id == "r1"
    reading.invoke(request)
    editing.invoke(request)
    assert live.chat.sent == [["read_tests", "edit_tests"], ["read_tests"],
                              ["read_tests", "edit_tests"]]
    assert [(entry.segment, entry.key["turn"]) for entry in oracle.entries] == [
        ("mutation", 1), ("mutation", 2), ("mutation", 3)]
    assert oracle.entries[0].answer["message"]["data"]["id"] == "r1"


def test_a_recording_solves_each_cover_with_the_run_arguments(monkeypatch):
    """The recorder solves a cover problem with the package's exact solver,
    called with the run's own costs: the measured times and the imputed
    costs apart."""
    from testuary import reduction

    tokens = {"t.py::test_a": frozenset({"L:1"}), "t.py::test_b": frozenset({"L:1", "L:2"})}
    seen = []
    real = reduction.exact_cover

    def solve(*args, **kwargs):
        seen.append(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr(reduction, "exact_cover", solve)
    oracle = record_demo.RecordingOracle(source="recorded run of a test")
    result = record_demo.RecordingCoverSource(oracle).solve_exact(
        tokens, {"t.py::test_a": 1.0}, forced_ids=frozenset(), time_limit_s=60.0,
        minimum_test_runtime_ms=0.001, random_seed=1, tie_break="solver",
        imputed_cost_ms={"t.py::test_b": 2.0})
    assert seen[0]["per_test_cost_ms"] == {"t.py::test_a": 1.0}
    assert seen[0]["imputed_cost_ms"] == {"t.py::test_b": 2.0}
    assert result.keep == ("t.py::test_b",)
    # As the solver's: the measured times apart from the imputed costs.
    assert set(result.solver_metadata["raw_cost_ms"]) == {"t.py::test_a"}
    (entry,) = oracle.entries
    assert entry.kind == "cover.exact" and entry.answer["keep"] == ["t.py::test_b"]
    assert entry.answer["solver_metadata"]["cost_basis"] == "imputed"


def test_recorded_data_are_checked_against_a_denylist_file(tmp_path):
    """The guard's patterns come from a file, so the public source names no
    private marker. A compressed scenario is read decompressed."""
    import gzip

    denylist = tmp_path / "denylist.txt"
    denylist.write_text("# comment\nsecret-host\n/private/[a-z]+\n\n", encoding="utf-8")
    patterns = record_demo.denylist_patterns(denylist)
    scenario = tmp_path / "scenario.json.gz"
    scenario.write_bytes(gzip.compress(b'{"output": "ran on SECRET-HOST in /private/area"}'))
    assert record_demo.private_paths(scenario, patterns) == ["/private/[a-z]+", "secret-host"]
    clean = tmp_path / "notice.md"
    clean.write_text("{testuary:out}/run", encoding="utf-8")
    assert record_demo.private_paths(clean, patterns) == []
    assert record_demo.denylist_patterns(None) == []


def _replay_entry(kind, key, answer, segment="initial"):
    return Entry(segment, kind, key, answer, "real", "s", "recorded-v1: x")


def test_the_replay_oracle_answers_held_requests_in_order_and_stops_on_others():
    scenario = Scenario([
        _replay_entry("model.seed", {"seed": 1}, {"text": "first"}),
        _replay_entry("model.seed", {"seed": 1}, {"text": "second"}, segment="coverage"),
        _replay_entry("backend.run_suite", {"suite": "x"}, {"outcome": "completed"})])
    oracle = record_demo.ReplayOracle(scenario)
    # Matched by kind and key whatever the segment, in the order recorded.
    assert oracle.answer(Call("model.seed", {"seed": 1}))["text"] == "first"
    assert oracle.unused() == 1
    assert oracle.answer(Call("model.seed", {"seed": 1}))["text"] == "second"
    with pytest.raises(ScenarioStop, match="holds no answer"):
        oracle.answer(Call("model.seed", {"seed": 1}))
    # Only model answers are replayed: an operation is never answered.
    with pytest.raises(ScenarioStop):
        oracle.answer(Call("backend.run_suite", {"suite": "x"}))


def test_a_replay_recording_reuses_the_earlier_timings_only_where_the_rest_matches():
    key = {"files": [["test_a.py", "h"]]}
    earlier = {"data": {"per_test_tokens": {"t.a": ["x"]}, "per_test_time_ms": {"t.a": 1.0},
                        "suite_time_ms": 5.0}, "failure": None}
    scenario = Scenario([
        Entry("initial", "backend.collect_eviction_data", key, earlier, "real", "old run",
              "recorded-v1: x"),
        Entry("initial", "backend.collect_eviction_data", key, earlier, "real", "old run",
              "recorded-v1: x")])
    oracle = record_demo.RecordingOracle(source="new run", reuse=scenario)
    same = {"data": {"per_test_tokens": {"t.a": ["x"]}, "per_test_time_ms": {"t.a": 2.5},
                     "suite_time_ms": 9.0}, "failure": None}
    # Only the times differ: the earlier answer is taken and names its run.
    assert _ask(oracle, "backend.collect_eviction_data", key, same) == earlier
    assert (oracle.entries[-1].source, oracle.reused) == ("old run", 1)
    other = {"data": {"per_test_tokens": {"t.a": ["y"]}, "per_test_time_ms": {"t.a": 2.5},
                      "suite_time_ms": 9.0}, "failure": None}
    # The evidence differs: the new measurement stands, with its own times.
    assert _ask(oracle, "backend.collect_eviction_data", key, other) == other
    assert oracle.entries[-1].source == "new run" and len(oracle.not_reused) == 1
    # Other operations are always measured anew.
    assert _ask(oracle, "backend.run_suite", {"suite": "x"}, {"elapsed_s": 3.0}) == {
        "elapsed_s": 3.0}
    assert oracle.entries[-1].source == "new run"


def _record(tmp_path, mode, *, path=None):
    """The recorder's run in a process of its own. ``path`` replaces the
    PATH it inherits. The Toy runs its tests with the "python" its adapter
    names, found on PATH, so the directory of the interpreter running these
    tests comes first there."""
    environment = dict(os.environ)
    outer = os.environ.get("PATH", "") if path is None else path
    environment["PATH"] = os.pathsep.join(item for item in (str(Path(sys.executable).parent), outer)
                                          if item)
    return subprocess.run([sys.executable, "-B", str(DRIVER), str(tmp_path), mode],
                          capture_output=True, text=True, encoding="utf-8",
                          errors="replace", timeout=600, env=environment,
                          stdin=subprocess.DEVNULL)


def test_the_recording_does_not_depend_on_a_python_on_the_tests_path(tmp_path):
    """The Toy's runtime is the "python" its adapter names, found on PATH. A
    test run outside an activated environment, as in a WSL suite, has no
    python there: the recording still finds the interpreter that runs it."""
    empty = tmp_path / "no-python"
    empty.mkdir()
    result = _record(tmp_path / "work", "operations-only", path=str(empty))
    shown = result.stdout + result.stderr
    assert "no project runtime is available" not in shown, shown[-2000:]
    assert "holds no answer for this model request" in result.stderr


def test_a_model_request_the_earlier_recording_lacks_stops_the_recording(tmp_path):
    # The earlier recording holds no model answer at all: the run's first
    # model request stops it before any model source is built, and nothing
    # is recorded.
    result = _record(tmp_path, "operations-only")
    assert result.returncode != 0, result.stdout[-2000:] + result.stderr[-2000:]
    assert "holds no answer for this model request" in result.stderr
    assert "a live model source was built" not in result.stdout + result.stderr
    assert not (tmp_path / "data").exists()


@pytest.mark.slow
def test_the_toy_re_recorded_reuses_every_model_answer_byte_for_byte(tmp_path):
    result = _record(tmp_path, "shipped")
    assert result.returncode == 0, result.stdout[-2000:] + result.stderr[-2000:]
    shipped = Scenario.load(Path(str(TOY / "scenario.json.gz")))
    new = Scenario.load(tmp_path / "data" / "scenario.json.gz")
    label = json.loads((TOY / "manifest.json").read_text(encoding="utf-8"))["exemplar"]["run"]

    def models(scenario):
        # Everything but the source, which names the recorded run's label.
        rows = [e.as_json() for e in scenario.entries if e.kind.startswith("model.")]
        assert rows
        return sorted(json.dumps({k: v for k, v in row.items() if k != "source"},
                                 sort_keys=True) for row in rows)

    assert models(new) == models(shipped)
    assert {e.source for e in new.entries} == {label}

    def keys(scenario):
        return sorted(json.dumps([e.segment, e.kind, e.key], sort_keys=True)
                      for e in scenario.entries)

    # Every call of the earlier recording is asked again, and none besides.
    assert keys(new) == keys(shipped)
