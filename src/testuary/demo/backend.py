"""The scenario's backends and mutation engine, for Python, Java and a
language registered from outside, and its model and exact reduction.

``ScenarioBackend`` is the Python backend with every operation that starts a
process answered by an oracle instead: test runs, failing-test collection,
per-test eviction data, coverage sidecars, benchmarks, the target parser,
the import and collection probes, and the mutation engine. Every operation
that executes nothing (normalization, pruning, merging, collision
resolution, validity analysis, prompt fragments, path helpers, kill
attribution) is the Python backend's own. The dependency check reports
nothing missing without probing: the Python backend's starts CBC to read its
version, and the demonstration uses no tool. As in a real run, no operation
places a setup file beside a suite. A run as delivered and a test run
that records leaks are answered like every other execution.

Each answered call is declared once in ``recorded``, and its method here is
made from the declaration. A recording (``tools/record_demo.py``) subclasses
these backends and runners: its oracle calls the real method the declaration
names, and its project probe, test setup and mutation engine are the real
ones.

``ScenarioJavaBackend`` does the same for Java: the playback states the
language facts the recording's project probe reported. A language registered
with LangAdapter from outside, such as the Toy of ``langadapter tutorial``,
runs through the generic backend (``backend_class``). Its adapter, named in
the manifest, is registered for the playback (``register``), so the run sees
the same language.

The scenario's model (``ScenarioModelSource``) answers suite generations and
agent responses. The exact reduction (``ScenarioCoverSource``) answers every
reduction the package builds from the scenario instead of solving it with
CBC, whose choice among equally good selections depends on the platform's
build. The package still builds the cover problem, and an answer is used
only if it solves it: it covers every token, keeps every forced test and
meets the recorded runtime and test-count optimum, the one CBC proved when
the scenario was recorded.
"""
from __future__ import annotations

import importlib
import importlib.util
import itertools
import json
import threading
from importlib import resources
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from langadapter.context import Interpreter, ProjectContext
from langadapter.java.adapter import JavaAdapter as ProjectJavaBackend
from langadapter.outcomes import Outcome
from langchain_core.language_models.chat_models import BaseChatModel
from pydantic import PrivateAttr

from testuary.demo import recorded
from testuary.demo.scenario import Call, Oracle, ScenarioStop, neutral_paths
from testuary.languages.generic import GenericBackend, GenericMutationRunner, _mutation_engine
from testuary.languages.mutation import (
    MutationMeasurement,
    MutationRunner,
    MutationSession,
)
from testuary.models import ModelSource
from testuary.reduction import exact_cover

#: The language facts and test vocabulary the recording stated for each
#: project directory, the directories as placeholders (``manifest.json``
#: ``language_facts``). None outside a playback of recorded data.
RECORDED_LANGUAGE: Optional[Dict[str, Dict[str, Any]]] = None


def facts_from_record(record: Dict[str, Any]):
    """The ``LanguageFacts`` a recording stored as its fields."""
    from langadapter.capabilities import LanguageFacts

    facts = dict(record)
    facts["test_file_patterns"] = tuple(facts["test_file_patterns"])
    if isinstance(facts.get("version_record"), dict):
        # The recording stores the runtime's Version as its fields.
        from langadapter.versions import Version

        facts["version_record"] = Version(**facts["version_record"])
    return LanguageFacts(**facts)


def language_record(backend) -> Dict[str, Any]:
    """What a recording keeps of a project backend for the prompts: its
    language facts and its test vocabulary. Both are model input, and
    reading them can start the project's interpreter."""
    from dataclasses import asdict

    read = getattr(backend, "test_vocabulary", None)
    return {"facts": asdict(backend.language_facts()),
            "vocabulary": dict(read()) if callable(read) else {}}


def recorded_language(root: str) -> Dict[str, Any]:
    """The recording's language facts and vocabulary for the project directory
    ``root``, which the playback states instead of asking the project."""
    key = neutral_paths(root)
    if RECORDED_LANGUAGE is None or key not in RECORDED_LANGUAGE:
        raise ScenarioStop(Call("backend.language_facts", {"root": key}),
                           "the recording holds no language facts for this project "
                           "directory. It must be regenerated for this package version")
    return RECORDED_LANGUAGE[key]


class RecordedFacts:
    """A project backend whose language facts and test vocabulary are the
    recording's, so building a prompt starts no process. Everything else
    is the project backend's."""

    def __init__(self, backend, record: Dict[str, Any]) -> None:
        self._backend = backend
        self._record = record

    def __getattr__(self, name):
        return getattr(self._backend, name)

    def language_facts(self):
        return facts_from_record(self._record["facts"])

    def test_vocabulary(self) -> Dict[str, str]:
        return dict(self._record["vocabulary"])


class ScenarioOperations:
    """Execution answers shared by every language's scenario backend: the
    recorded calls of ``recorded.BACKEND_CALLS``, each asked of the oracle.

    Source operations that run nothing stay the language backend's own,
    among them the setup hook, which places no file.
    """

    def __init__(self, oracle: Oracle, *args: Any) -> None:
        # Any other arguments go to the language backend.
        super().__init__(*args)
        self._oracle = oracle

    def ask(self, call: Call) -> Any:
        """The oracle's answer to one recorded call."""
        return self._oracle.answer(call)

    def check_dependencies(self) -> List[str]:
        """Nothing is missing: the demonstration runs no test, coverage or
        mutation tool and no solver, so none is probed."""
        return []

    def mutation_runner(self, config: Dict[str, Any], *,
                        source_dir: Optional[str] = None,
                        log: Optional[Callable[..., None]] = None,
                        ) -> "ScenarioRunner":
        return ScenarioRunner(self._oracle, config, log=log)


for _entry in recorded.BACKEND_CALLS:
    setattr(ScenarioOperations, _entry.method, recorded.backend_method(_entry))


class ScenarioBackend(ScenarioOperations, GenericBackend):
    """Python's native source operations and scripted execution answers."""

    def __init__(self, oracle: Oracle) -> None:
        super().__init__(oracle, "python", (".py",))

    def project_backend(self, source_dir=None):
        """The project's Python adapter, with the recording's language facts
        and vocabulary: reading them runs the project's interpreter, a
        process the demonstration does not start."""
        backend = GenericBackend.project_backend(self, source_dir)
        root = str(Path(source_dir or Path.cwd()).resolve())
        return RecordedFacts(backend, recorded_language(root))

    merge_test_files = recorded.backend_method(recorded.MERGE_TEST_FILES)

    def salvage_syntax(self, source, source_dir, *, report=None):
        """Python's bounded syntax repair, its validations from the scenario's
        validator answers (``validate_test_source``)."""
        from langadapter.python.source_analysis import salvage_syntax
        from langadapter.sources import SourceValidation as Validation

        def validate(candidate):
            validation = self.validate_test_source(candidate, source_dir)
            return Validation(validation.status, validation.message, validation.line, validation.col,
                              validation.validator_identity, validation.source_sha256)

        result = salvage_syntax(source, validate)
        if report is not None:
            report.update(status=result.status, repairs=result.repairs,
                          message=result.validation.message)
        return result.source


class ScenarioMeasurement(MutationMeasurement):
    """A mutation run's measurement as the scenario holds it: its counts,
    reason, stop and the per-mutant timeout the engine applied. Its
    survivors' text and kill matrix are answered for the run's key
    (``run_key``)."""

    def __init__(self, runner: "ScenarioRunner", run_key: str,
                 data: Dict[str, Any]) -> None:
        super().__init__(
            killed=int(data["killed"]), survived=int(data["survived"]),
            timeout=int(data["timeout"]), non_viable=int(data["non_viable"]),
            untested=int(data["untested"]), total=int(data["total"]),
            score=float(data["score"]), truncated=bool(data["truncated"]),
            reason=str(data["reason"]), suite_failed=bool(data["suite_failed"]),
            stopping_tests=recorded.test_ids(data["stopping_tests"]),
            lower_bound=bool(data["lower_bound"]))
        object.__setattr__(self, "runner", runner)
        object.__setattr__(self, "run_key", run_key)
        object.__setattr__(self, "applied_timeout", data["timeout_rule"])

    def ask(self, call: Call) -> Any:
        return self.runner.ask(call)

    def timeout_rule(self) -> Optional[Dict[str, Any]]:
        return self.applied_timeout

    survivors_text = recorded.playback(recorded.RENDER_SURVIVING_DIFFS,
                                       MutationMeasurement.survivors_text)
    kill_matrix = recorded.playback(recorded.CAPTURE_KILL_MATRIX, MutationMeasurement.kill_matrix)


class ScenarioMutationSession(MutationSession):
    """A session that states the configured worker pool, as a pooled
    session does, so the phases log the configuration the recorded run
    used. Nothing is pooled, and its runs are answered by the scenario."""

    def __init__(self, runner: "ScenarioRunner", workers: int, use_http: bool) -> None:
        super().__init__(runner, None, workers, use_http)

    def ask(self, call: Call) -> Any:
        return self.runner.ask(call)

    run = recorded.playback(recorded.MUTATION_RUN, MutationSession.run)
    per_suite = recorded.playback(recorded.MUTATION_PER_SUITE, MutationSession.per_suite)

    def close(self, primary: Optional[BaseException] = None) -> None:
        """Nothing is pooled, so nothing closes, also as a with block ends."""


class ScenarioSuiteRun:
    """One suite's run of a per-suite answer: its outcome, its reason and
    whether it measured anything."""

    def __init__(self, data: Dict[str, Any]) -> None:
        self.outcome = Outcome(str(data["outcome"]))
        self.reason = str(data.get("reason") or "")
        self.measured = bool(data["measured"])


class ScenarioPerSuiteRun:
    """A per-suite answer with the parts of ``PerSuiteRun`` the phases read:
    each suite's run, the implementation, and each suite's own kill matrix
    or, without a name, their union."""

    def __init__(self, answer: Dict[str, Any]) -> None:
        self.outcome = Outcome(str(answer["outcome"]))
        self.reason = str(answer.get("reason") or "")
        self.implementation = str(answer["implementation"])
        self.runs = {str(name): ScenarioSuiteRun(run)
                     for name, run in answer["runs"].items()}
        self._matrices = dict(answer["kill_matrices"])
        self._union = answer["union"]

    def kill_matrix(self, name: Optional[str] = None) -> Dict[str, Any]:
        if name is None:
            return json.loads(json.dumps(self._union))
        if name not in self._matrices:
            raise KeyError(f"no kill matrix for suite {name!r}")
        return json.loads(json.dumps(self._matrices[name]))


class ScenarioRunner(MutationRunner):
    """The mutation runs, per-suite runs, the full catalogue, the survivors'
    text and the kill matrices answered by an oracle. The configured
    catalogue follows the real rule, which executes nothing."""

    demo_language = "python"
    #: The class of the sessions ``open`` returns.
    session_type = ScenarioMutationSession

    def __init__(self, oracle: Oracle, config: Dict[str, Any], *,
                 log: Optional[Callable[..., None]] = None) -> None:
        super().__init__(None, config, log=log)
        self._oracle = oracle

    def ask(self, call: Call) -> Any:
        return self._oracle.answer(call)

    full_operators = recorded.playback(recorded.FULL_OPERATORS, MutationRunner.full_operators)

    def open(self, workers: int, *, suite_dirs=()) -> ScenarioMutationSession:
        """A scenario session: the demo stages nothing, so ``suite_dirs``
        is accepted and unused."""
        return self.session_type(
            self, workers,
            str(self.config.get("cosmic_ray_distributor", "")).strip().lower() == "http")


class ScenarioPitRunner(ScenarioRunner, GenericMutationRunner):
    """PIT's native catalogue selection with scenario session answers."""

    name = "pit"
    demo_language = "java"

    def __init__(self, oracle, config, *, backend, log=None) -> None:
        super().__init__(oracle, config, log=log)
        # The project's Java adapter answers the catalogue rules, which run nothing.
        self.backend = backend

    def open(self, workers: int, *, suite_dirs=()):
        session = super().open(workers, suite_dirs=suite_dirs)
        session.use_http = False
        return session


class ScenarioJavaBackend(ScenarioOperations, GenericBackend):
    """Uses Java's normalization, merge, pruning and prompt implementation.

    Project facts are the recording's instead of a JDK and build probe. Test
    and mutation runs are answered from the scenario as in the Python demo.
    No Python determinism or module-restoration setup is applied to Java.
    """

    def __init__(self, oracle) -> None:
        super().__init__(oracle, "java", (".java",))

    def project_backend(self, source_dir=None):
        root = str(Path(source_dir or Path.cwd()).resolve())
        context = ProjectContext(root=root, interpreter=Interpreter(path="offline-demo-no-jdk"))
        backend = self._project_backends.get(context)
        if backend is None:
            backend = ProjectJavaBackend(context)
            self._project_backends[context] = backend
        return RecordedFacts(backend, recorded_language(root))

    def salvage_syntax(self, source, source_dir, *, report=None):
        validation = self.validate_test_source(source, source_dir)
        if report is not None:
            report.update(status="valid" if validation.ok else "beyond_repair",
                          message=validation.message, repairs=0)
        return source if validation.ok else None

    strip_module_level_skips = recorded.backend_method(recorded.STRIP_MODULE_LEVEL_SKIPS)
    merge_test_files = recorded.backend_method(recorded.MERGE_TEST_FILES)
    resolve_test_collisions = recorded.backend_method(recorded.RESOLVE_TEST_COLLISIONS)

    def setup_test_env(self, test_output_dir, source_dir, target_file_path=None):
        Path(test_output_dir).mkdir(parents=True, exist_ok=True)

    def mutation_runner(self, config: dict[str, Any], *, source_dir=None,
                        log=None) -> ScenarioPitRunner:
        return ScenarioPitRunner(self._oracle, config, backend=self.project_backend(source_dir),
                                 log=log)


class ScenarioGenericRunner(ScenarioRunner, GenericMutationRunner):
    """The generic backend's mutation runs answered by the scenario, with the
    generic runner's own catalogue, named operators and per-mutant timeout
    rule, which its backend answers without running anything."""

    def __init__(self, oracle, config: Dict[str, Any], *, engine: str, language: str,
                 backend: Any, log: Optional[Callable[..., None]] = None) -> None:
        super().__init__(oracle, config, log=log)
        self.name = engine
        self.backend = backend
        # The keys of its runs hold the files this language's rule names.
        self.demo_language = language


class ScenarioGenericBackend(ScenarioOperations, GenericBackend):
    """The generic backend with the scenario's execution answers."""

    def __init__(self, oracle, name: str, file_extensions: Sequence[str]) -> None:
        GenericBackend.__init__(self, name, tuple(file_extensions))
        self._oracle = oracle

    def setup_test_env(self, test_output_dir, source_dir, target_file_path=None):
        Path(test_output_dir).mkdir(parents=True, exist_ok=True)

    def mutation_runner(self, config: Dict[str, Any], *, source_dir=None,
                        log=None) -> ScenarioGenericRunner:
        """The generic runner's mutation runs, through the engine the generic
        backend chooses (``_mutation_engine``)."""
        backend = self.project_backend(source_dir)
        engine = _mutation_engine(backend, config.get("engine"))
        if engine is None:
            raise ValueError(f"{self.name} has no mutation engine")
        return ScenarioGenericRunner(self._oracle, config, engine=engine, language=self.name,
                                     backend=backend, log=log)


class ScriptedChatModel(BaseChatModel):
    """A chat model whose responses are the oracle's answers, one per
    request, in order. Tools are accepted as bound. The responses name the
    tool calls themselves."""

    oracle: Any
    phase: str
    # The phase's request count. Each bound copy shares it, so requests
    # count across the agent modes as one sequence.
    _turns: Any = PrivateAttr(default_factory=lambda: itertools.count(1))
    _lock: Any = PrivateAttr(default_factory=threading.Lock)

    @property
    def _llm_type(self) -> str:
        return "testuary-demo-scenario"

    def bind_tools(self, tools: Any, *, tool_choice: Any = None,
                   **kwargs: Any) -> "ScriptedChatModel":
        """A copy for one of the agent modes, which shares the phase's
        request count."""
        return self.model_copy()

    def ask(self, call: Call) -> Any:
        return self.oracle.answer(call)

    def next_turn(self) -> int:
        """The number of the phase's next request."""
        with self._lock:
            return next(self._turns)

    _generate = recorded.playback(recorded.MODEL_AGENT, BaseChatModel._generate)


class ScenarioModelSource:
    """A ``models.ModelSource`` answering from an oracle."""

    #: The class of the chat models ``agent_chat_model`` returns.
    chat_model_type = ScriptedChatModel

    def __init__(self, oracle: Oracle) -> None:
        self._oracle = oracle

    def ask(self, call: Call) -> Any:
        return self._oracle.answer(call)

    generate_suite = recorded.playback(recorded.MODEL_SUITE, ModelSource.generate_suite)

    def agent_chat_model(self, phase: str) -> ScriptedChatModel:
        self._oracle.enter_segment(phase)
        return self.chat_model_type(oracle=self._oracle, phase=phase)


class ScenarioCoverSource:
    """Answers each exact reduction from the scenario. Its problems were
    recorded with every test timed, and an imputed cost is weighed as the
    solver weighs it."""

    def __init__(self, oracle: Oracle) -> None:
        self._oracle = oracle

    def ask(self, call: Call) -> Any:
        return self._oracle.answer(call)

    solve_exact = recorded.playback(recorded.EXACT_COVER, exact_cover)


def backend_class(language: str, file_extensions: Sequence[str]) -> Callable[[Any], Any]:
    """The scenario backend factory for ``language``, as the demo drivers call
    it: with the oracle only."""
    def make(oracle):
        return ScenarioGenericBackend(oracle, language, file_extensions)
    return make


def load_adapter(spec: Dict[str, str]):
    """The adapter class a manifest names: ``{"package", "path", "class"}``
    for a file inside an installed package, or ``{"module", "class"}``."""
    if "module" in spec:
        return getattr(importlib.import_module(spec["module"]), spec["class"])
    path = resources.files(spec["package"])
    # One part per join: Python 3.10's package files take one.
    for part in spec["path"].split("/"):
        path = path.joinpath(part)
    name = "demo_adapter_" + Path(spec["path"]).stem
    module_spec = importlib.util.spec_from_file_location(name, str(path))
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return getattr(module, spec["class"])


def register(language: str, file_extensions: Sequence[str], spec: Dict[str, str]):
    """Register the adapter ``spec`` names for ``language``, unless a language
    of that name is registered already. Returns the function that undoes it."""
    import langadapter as la

    if language in la.names():
        return lambda: None
    la.register(language, list(file_extensions), load_adapter(spec))
    return lambda: la.unregister(language)
