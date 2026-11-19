"""Testuary's interface to a language, its one implementation and the
results its phases read.

``LanguageBackend`` is the interface's type, with the language's name and
file extensions. ``GenericBackend`` implements it for every language
LangAdapter registers, over the adapter's public operations and facts: its
runs, coverage, per-test evidence, timing and mutation sessions, its source
operations, and the facts that name its test files and describe its tests
to the model. The result types are Testuary's. The source-analysis records
come from ``langadapter.sources``. Helpers convert a validator observation
(``validation_result``), metric observations to Testuary's file counters
(``test_counts``) and replacement sources to suite paths
(``relative_sources``).

An operation the language does not offer takes the fallback Testuary's
requirement profile (``testuary/langadapter_profile.json``) states for its family: the
method returns what the calling phase treats as absent (no per-test data,
no merge, no repair), and the run records the fallback. A missing operation
never raises out of a method a phase calls, and never reaches a
model-facing text.

No method branches on a language's name. The mutation runner and prepared
copies used here (``languages.mutation``, ``languages.preparation``) import
Python helpers from ``langadapter.python``, and a failed prepared copy is
recognized by its ``preparation_setup_error`` or
``preparation_cleanup_error`` marker.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import (
    Any,
    Dict,
    FrozenSet,
    Iterable,
    List,
    NamedTuple,
    Optional,
    Tuple,
    Mapping,
    Union,
)
from langadapter.sources import RemovedBinding, SuiteAuditFlag, SuiteNormalization, SuiteValidityFinding, TargetIdentity, SourceObservation, SourceStatus, TestMetrics  # noqa: F401  (re-exported)
import dataclasses
import fnmatch
import json
import os
import time
from pathlib import Path
from langadapter import ProjectContext, create, registration
from langadapter.mutation import Catalogue, TimeoutRule, upgrade_kill_matrix
from langadapter.outcomes import Outcome
from langadapter.runs import (
    BenchmarkRequest,
    CollectionRequest,
    PerTestEvidenceRequest,
    RunDiagnostic,
    SuiteRunRequest,
    with_detail,
)
from langadapter.sources import CollisionResolution, SourceValidation as NativeValidation
from langadapter.suites import Suite
from testuary import phase_state, config as config_module, project as benchmark_lock
from testuary.agent import _limits
from testuary.generation_evidence import source_sha256
from testuary.languages.mutation import MutationRunner
from testuary.languages.preparation import kept_measurement, prepared_measurement
from langadapter.errors import InvalidRequest


def measured_seconds(run: Any, matrix: bool) -> Optional[float]:
    """A suite run's measured time, as a run in place is timed: the
    runner's process, the reading of its coverage data and, with
    ``matrix``, the reading of the coverage tool's statements. None when
    the runner's time is unknown."""
    if run.wall_time_s is None:
        return None
    steps = (run.coverage_report_s, run.coverage_statements_s if matrix else None)
    return run.wall_time_s + sum(step for step in steps if step is not None)


class StructureResult(NamedTuple):
    """Result of a structural check of a source.

    ``ok`` is the only field a simple caller needs. ``line`` and ``col``
    (1-based, 0 when ``ok``) let a caller build an error snippet without
    parsing ``message``. ``snippet`` is the whole source text the error
    covers, which the caller cuts and quotes after ``message``, or None
    when the language gives none (``message`` is then the whole text)."""

    ok: bool
    message: str
    line: int
    col: int
    snippet: Optional[str] = None


@dataclass(frozen=True)
class SourceValidation:
    """The target parser's judgement of complete test-source bytes. A
    validator that gives no answer yields ``ok`` False with ``status``
    ``timeout`` or ``validator_error``."""

    ok: bool
    status: str
    message: str
    line: int
    col: int
    validator_identity: str
    source_sha256: str


class TestRunResult(NamedTuple):
    """Result of ``LanguageBackend.run_test_files``: one suite run, in the
    same shape for every language and framework. ``ok`` is the field most
    callers need: the suite ran and every test passed. ``returncode`` is
    for diagnostics only. ``collected`` is a hint: without the runner's
    count it is ``passed + failed + errors``. ``output`` is the combined
    stdout and stderr. The coverage fields are filled only when
    ``measure_coverage_of`` was passed and the measurement succeeded: None
    means not measured, never zero. ``missing_lines`` holds the sorted
    1-based lines of that file the suite did not execute, ``()`` when not
    measured or fully covered. ``stopping_tests`` holds the id objects of the
    run's failed and errored tests."""

    ok: bool
    returncode: int
    timed_out: bool
    elapsed_s: Optional[float]
    collected: int
    passed: int
    failed: int
    errors: int
    output: str
    line_coverage: Optional[float]
    branch_coverage: Optional[float]
    missing_lines: Tuple[int, ...] = ()
    # The target's untaken branches. Python gives each arc as "src->dst"
    # ("12->15", "12->exit" for a jump out of the function). Java's coverage
    # does not say where a branch goes, so it gives one entry per line
    # ("12->1 of 2 branches untaken"). () when not measured or fully covered.
    missing_branches: Tuple[str, ...] = ()
    # The target's covered lines and branch arcs ("a->b") from the same
    # measurement as the percentages: Python's every line the tracer
    # recorded, excluded lines included, Java's the covered statements.
    # None when not measured.
    covered_lines: Optional[Tuple[int, ...]] = None
    covered_branches: Optional[Tuple[str, ...]] = None
    # With ``record_leaks``, the tests that left the module under test or a
    # library setting changed, as the language's observer records them,
    # restoring nothing: () when none did, None when not asked or not
    # recorded (a language without the record, a run ended before
    # collection).
    leaks: Optional[Tuple["LeakRecord", ...]] = None
    # The failed and errored tests the runner's report names, in its order,
    # each with its message and, where given, its stack trace.
    failures: Tuple["TestFailure", ...] = ()
    # With ``timeout_journal``, the test the journal shows running when the
    # run ended (a timeout or a crash), as its qualified name with its
    # parameters (``TestX.test_y[3]``). An observation, not proof that this
    # test alone ends the run. None without the journal or when every
    # started test had finished.
    active_at_end: Optional[str] = None
    # The target's statement and branch totals from the same measurement as
    # the percentages, None when not measured. Java's branch total counts
    # its branches, not its untaken entries.
    total_statements: Optional[int] = None
    total_branches: Optional[int] = None
    # The warnings that mark a passing test as leaving shared state behind
    # (the language's ``unsound_test_patterns``), in the run's order: the
    # test as the run printed it, None when the run names none, and the
    # matched text.
    unsound: Tuple[Tuple[Optional[str], str], ...] = ()
    # The run's diagnostics as its language states them, in the run's order.
    diagnostics: Tuple[RunDiagnostic, ...] = ()
    # The lines that explain the run's reason as its language gives them,
    # such as a compiler's error lines.
    reason_detail: Tuple[str, ...] = ()
    # The test id objects of the run's failed and errored tests, in the order
    # of ``failures``, so a removal takes them without matching a text. The
    # demo recorder leaves them out, since they are no JSON.
    stopping_tests: Tuple[Any, ...] = ()


class TestFailure(NamedTuple):
    """A failed or errored test of a run's report: the test in the
    language's test-id text, ``failed`` or ``error``, the message and the
    stack trace ("" when the report gives none)."""

    __test__ = False  # not a test class for pytest's collection

    test: str
    outcome: str
    message: str
    trace: str = ""


class LeakRecord(NamedTuple):
    """A test that left the module under test or a library setting
    changed, as recorded after its own fixtures were torn down: the test in
    the failing-test listing's form (``classname.name`` without the
    ``[param]`` suffix) and as pytest's node id, the names of the module under test it
    added, rebound and removed, and the xarray options and numpy error and
    print settings it left changed."""

    test: str
    nodeid: str
    added: Tuple[str, ...] = ()
    rebound: Tuple[str, ...] = ()
    removed: Tuple[str, ...] = ()
    xarray_options: Tuple[str, ...] = ()
    numpy_errors: Tuple[str, ...] = ()
    numpy_print: Tuple[str, ...] = ()


def _failed_ids(run: Any) -> Tuple[Any, ...]:
    """The id objects of ``run``'s failed and errored tests, as a
    measurement's removal reads them."""
    return tuple(case.test for case in run.tests if case.outcome in ("failed", "error"))


class DeliveredCase(NamedTuple):
    """One test case of a run as delivered: the test in the failing-test
    listing's form, the case's name with its parameters, its outcome
    (``passed``, ``failed``, ``error`` or ``skipped``) and the message."""

    test: str
    name: str
    outcome: str
    message: str = ""


class DeliveredRun(NamedTuple):
    """Test files run as delivered, each alone in a directory of its own,
    with the session seed ``seed``, in the file's
    order or, with ``reverse``, its reverse. ``cases`` are in the order they
    ran, file by file. ``failing`` holds the tests with a failed or erroring
    case, or is None, with the ``reason``, when a file gave no test outcome
    (a collection failure, a crash, the timeout, no test). ``timed_out``
    says the timeout ended it, ``elapsed_s`` its wall time when known.
    ``leaks`` and ``active_at_end`` are ``TestRunResult``'s, for a run that
    asked for them, and ``unsound`` and ``stopping_tests`` are
    ``TestRunResult``'s, each file's entries in the order the files ran."""

    cases: Tuple[DeliveredCase, ...]
    failing: Optional[Tuple[str, ...]]
    reason: Optional[str]
    output: str
    reverse: bool = False
    seed: int = 1
    timed_out: bool = False
    elapsed_s: Optional[float] = None
    leaks: Optional[Tuple["LeakRecord", ...]] = None
    active_at_end: Optional[str] = None
    unsound: Tuple[Tuple[Optional[str], str], ...] = ()
    stopping_tests: Tuple[Any, ...] = ()

    @property
    def ok(self) -> bool:
        """Whether every file gave test outcomes and every case passed or
        was skipped."""
        return self.reason is None and self.failing == ()

    def passed(self, tests: Iterable[str]) -> bool:
        """Whether each of ``tests`` ran and every case of it passed: a test
        missing from the run, or skipped in it, has not passed."""
        if self.reason is not None:
            return False
        outcomes: Dict[str, List[str]] = {}
        for case in self.cases:
            outcomes.setdefault(case.test, []).append(case.outcome)
        return all(test in outcomes
                   and all(o == "passed" for o in outcomes[test])
                   for test in tests)


@dataclass(frozen=True)
class EvictionData:
    """Per-test inputs of the suite-time reducer from one instrumented,
    warmed suite run. Tokens are opaque ids, so one structure serves every
    criterion (line, branch, killed mutant) and every language.

    ``per_test_tokens`` maps each test id to its tokens, an empty set for a
    test that holds none, so the reducer can remove it. ``per_test_time_ms``
    holds the report's times: an id without one (a coverage context no
    reported test matched) has none, and the caller decides its cost. They
    are relative costs. The total against the timeout comes from
    re-measuring the reduced file, and ``suite_time_ms`` is the measured one.

    ``includes_kills`` is True when killed-mutant tokens are present, or
    when no kill names a killer (each kept test reproduces such a kill: the
    mutant breaks the target's import). Coverage-only tokens allow a
    coverage-phase reduction but not a mutation-phase one, since a removed
    test could be a mutant's sole killer."""

    per_test_tokens: Dict[str, FrozenSet[str]]
    per_test_time_ms: Dict[str, float]
    suite_time_ms: float
    includes_kills: bool
    # The target's suite coverage from the same run: statement and branch
    # totals, covered counts, and the covered lines and branch arcs ("a->b"),
    # whose sizes over the totals equal the suite's coverage. None when the
    # run gives no coverage.
    target_total_lines: Optional[int] = None
    target_covered_lines: Optional[int] = None
    target_total_branches: Optional[int] = None
    target_covered_branches: Optional[int] = None
    suite_covered_lines: Optional[FrozenSet[int]] = None
    suite_covered_branches: Optional[FrozenSet[str]] = None
    # The test the report lists first, which ran first: its time also holds
    # the run's one-time setup, paid whichever tests are kept. None when the
    # report gives no test or the order is unknown.
    first_test: Optional[str] = None


class LanguageBackend:
    """Testuary's interface to one language, one instance per registered
    language. ``GenericBackend`` implements the operations the phases call,
    so that no phase branches on the language. A registration does not
    establish that the language can run.
    """

    # ---- identity --------------------------------------------------

    name: str
    """The canonical name used everywhere (``"python"``, ``"java"``)."""

    file_extensions: Tuple[str, ...]
    """File extensions this backend claims, the first for the files it
    writes. ``get_backend`` sets them from the language's public
    registration, which ``detect_language(path)`` reads for the
    file-extension -> backend mapping. The registry guarantees one backend
    per file extension."""


#: The default name of a target's test file, when the language names none.
DEFAULT_TEST_FILE_NAME = "test_{stem}{ext}"
#: The line between two merged suites, after the language's comment prefix.
MERGE_SEPARATOR = "=== merged from the mutation-killing test suite ==="
#: The file name recorded for a target whose phase produced no test file,
#: never a target's own file.
FAILED_MARKER = "test_failed.py"
#: The policy version of a target identity when the language has no
#: generated-test validity policy (the validity screening fallback).
NO_VALIDITY_POLICY = "none"
#: The languages whose tests are written in the language itself, which
#: Testuary knows without asking the adapter. Another adapter may report
#: tests in another language (``LanguageFacts.test_language``): Python tests
#: of a C file.
_OWN_TEST_LANGUAGES = frozenset({"python", "java"})
#: The mutation engines that time the passing suite themselves, in the form
#: a mutant's tests run, and apply a requested rule to that time: Mull runs
#: each C test of a mutant in a process of its own, and Testuary's normal
#: run runs them in one program.
_OWN_RUN_RULE_ENGINES = frozenset({"mull"})
#: Recovery statuses that mean the operation gave no answer.
_NO_ANSWER = ("unsupported", "infrastructure_failure", "missing_setup", "timeout")


def case_id(adapter: Any, test: Any) -> str:
    """A test's id as the failing-test listing writes it: ``test_id_text``,
    with a parametrized case's parameters in brackets (``module.test[id]``),
    the form ``prune_tests`` removes alone where the language can."""
    return adapter.test_id_text(test) + case_suffix(getattr(test, "params", None))


def case_suffix(params: Optional[str]) -> str:
    """A case's parameters in brackets, as pytest and JUnit write them, or
    "" without parameters. Python's ``TestId.params`` hold no brackets,
    Java's hold their own."""
    if not params:
        return ""
    return params if params.startswith("[") and params.endswith("]") else f"[{params}]"


class PruneOutcome(NamedTuple):
    """What a removal of tests did: the source, and why a named case's test
    went whole ("" when none did)."""

    source: str
    detail: str


def prune_report(backend: Any, source: str, test_ids: Iterable[str]) -> PruneOutcome:
    """``backend.prune_tests_report``: the removal with the language's
    account of what it took out."""
    return backend.prune_tests_report(source, test_ids)


def _validity_unresolved(operation: str, observation: Any) -> Exception:
    """The error for an offered validity operation that gave no answer: an
    infrastructure failure of the run, never an unchecked pass."""
    from testuary.consolidation import ValidityUnresolved
    status = getattr(observation, "status", None)
    status = getattr(status, "value", status)
    reason = getattr(observation, "reason", None) or getattr(observation, "detail", None)
    return ValidityUnresolved(
        f"generated-test validity {operation} did not complete: {status}: {reason}")


class GenericMutationRunner(MutationRunner):
    """Mutation runs through the language's own engine, configured by the
    language: its catalogue from the mutation settings, its named operator
    sets, its own per-mutant timeout rule where it has one, the auto rule
    where it times the passing suite itself, and its session attempts (a
    parallel session and the fallback when it does not open)."""

    def __init__(self, backend: Any, config: Dict[str, Any], *, engine: Optional[str] = None,
                 log: Optional[Any] = None) -> None:
        super().__init__(backend, config, log=log)
        # A subclass may name its engine as its class attribute instead.
        if engine is not None:
            self.name = engine

    def _prepared_suite_inputs(self, suite, lifetime):
        """The setup files the language loads beside the suite, declared
        before staging, and those of the copy whose originals are gone."""
        from testuary.languages.preparation import _suite as copied_suite

        backend = self.backend
        current = backend.setup_files(suite.root, list(suite.files()[:len(suite.test_files)]))
        selected = dataclasses.replace(suite, setup_files=tuple(dict.fromkeys(
            (*suite.setup_files, *(str(path) for path in current)))))
        copied = copied_suite(suite, lifetime.original, lifetime.context, lifetime.owner)
        previous = backend.setup_files(copied.root, list(copied.files()[:len(copied.test_files)]))
        absent = []
        for path in previous:
            original = Path(suite.root) / Path(path).relative_to(Path(copied.root))
            if not original.exists():
                absent.append(str(original))
        return selected, tuple(absent)

    def catalogue(self) -> Catalogue:
        return self.backend.mutation_catalogue(self.config)

    def configured_operators(self) -> Optional[List[str]]:
        catalogue = self.catalogue()
        if catalogue.kind == "full":
            return None
        if catalogue.kind == "named":
            return list(self.backend.named_operators(catalogue.name))
        return list(catalogue.operators)

    def engine_sets_timeout(self) -> bool:
        """Whether the engine's own per-mutant timeout rule applies: where
        the language has one, unless ``per_mutant_timeout_seconds`` fixes a
        value. Testuary's derived per-mutant timeout has no counterpart."""
        return bool(self.backend.own_mutant_timeout_rule) and config_module.fixed_per_mutant_timeout(
            self.config.get("per_mutant_timeout_seconds")) <= 0

    def requested_timeout(self, per_mutant_timeout: int) -> Union[int, TimeoutRule, None]:
        """What a request carries as the per-mutant timeout, and for an
        engine of ``_OWN_RUN_RULE_ENGINES`` the auto rule itself, unless
        ``per_mutant_timeout_seconds`` fixes a value: the factor, the minimum
        and the maximum (``resolve_mutation_max_timeout``, none at 0)."""
        if self.name not in _OWN_RUN_RULE_ENGINES or config_module.fixed_per_mutant_timeout(
                self.config.get("per_mutant_timeout_seconds")) > 0:
            return super().requested_timeout(per_mutant_timeout)
        cap = config_module.resolve_mutation_max_timeout(self.config)
        return TimeoutRule(
            factor=float(config_module.require_config(self.config, "per_mutant_timeout_factor")),
            minimum_s=int(config_module.require_config(self.config, "min_per_mutant_timeout_seconds")),
            cap_s=cap if cap > 0 else None)


class GenericBackend(LanguageBackend):
    """Testuary's backend for a registered language over its adapter's
    public operations and facts. An adapter is kept per project context
    while the language's selected registration stays the same."""

    def __init__(self, name: str, file_extensions: Tuple[str, ...]) -> None:
        self.name = name
        self.file_extensions = tuple(file_extensions)
        self._project_backends = {}

    # ---- the language's facts ---------------------------------------------

    def _adapter_class(self):
        """The class whose attributes declare the language's facts: the
        registered factory when it is a class, else the class of the adapter
        the factory returns (a factory may be any callable)."""
        factory = registration(self.name).factory
        return factory if isinstance(factory, type) else type(self.source_backend())

    def _fact(self, name: str, default: Any = None) -> Any:
        """A fact the language's adapter declares as a class attribute."""
        value = getattr(self._adapter_class(), name, None)
        return default if value in (None, "") else value

    def language_facts(self, source_dir=None):
        """The facts the adapter reports for the project, or None. Outside a
        run (setup reads a directory's sources before any run exists) the
        source-only context reports them."""
        backend = (self.project_backend(source_dir)
                   if source_dir is not None or config_module.active_settings() is not None
                   else self._registered_backend(self._source_context()))
        facts = getattr(backend, "language_facts", None)
        return facts() if callable(facts) else None

    def test_language(self, source_dir=None) -> str:
        """The registered language the tests of the project are written in:
        the language itself for Python and Java, else as the adapter's facts
        report it (``LanguageFacts.test_language``, None for its own)."""
        if self.name in _OWN_TEST_LANGUAGES:
            return self.name
        return getattr(self.language_facts(source_dir), "test_language", None) or self.name

    def test_file_extensions(self, source_dir=None) -> Tuple[str, ...]:
        """The file extensions of the tests' files: those of the language the
        tests are written in (``test_language``)."""
        language = self.test_language(source_dir)
        if language == self.name:
            return tuple(self.file_extensions)
        return tuple(registration(language).file_extensions)

    def _test_fact(self, name: str, default: Any = None, source_dir=None) -> Any:
        """A fact of the tests (their file names, framework and the name the
        prompts give the language): the class attribute of the language the
        tests are written in (``test_language``)."""
        language = self.test_language(source_dir)
        if language == self.name:
            return self._fact(name, default)
        value = getattr(registration(language).factory, name, None)
        return default if value in (None, "") else value

    def _test_patterns(self) -> Tuple[str, ...]:
        patterns = self._test_fact("test_file_patterns")
        if patterns:
            return tuple(patterns)
        facts = self.language_facts()
        if facts is not None and facts.test_file_patterns:
            return tuple(facts.test_file_patterns)
        template = self._test_fact("test_file_name", DEFAULT_TEST_FILE_NAME)
        return tuple(template.format(stem="*", Stem="*", ext=file_extension)
                     for file_extension in self.test_file_extensions())

    def _target_pattern(self) -> str:
        """The test file name template with a wildcard for the stem: the
        names of the files Testuary delivers."""
        template = self._test_fact("test_file_name", DEFAULT_TEST_FILE_NAME)
        return template.format(stem="*", Stem="*", ext=self.test_file_extensions()[0])

    def _test_file_name(self, stem: str, source_dir=None) -> str:
        template = self._test_fact("test_file_name", DEFAULT_TEST_FILE_NAME,
                                   source_dir=source_dir)
        return template.format(stem=stem, Stem=stem[:1].upper() + stem[1:],
                               ext=self.test_file_extensions(source_dir)[0])

    # ---- the project ---------------------------------------------------------

    def _registered_backend(self, context):
        """Cache a public factory result only while its registration remains selected."""
        selected = registration(self.name)
        registrations = self.__dict__.setdefault("_backend_registrations", {})
        backends = self.__dict__.setdefault("_project_backends", {})
        if context not in backends or registrations.get(context) is not selected:
            backends[context] = create(self.name, context)
            registrations[context] = selected
        return backends[context]

    def source_backend(self):
        """Public source-only context. Structural reads do not select a target
        runtime. They read the tests' sources, so a language whose tests are
        written in another (``test_language``: Python tests of a C file)
        reads them with that language's adapter: a source-only context knows
        no project, and the target's adapter would read its own language.
        The context's directory is the run's base, else the working directory:
        setup reads a directory's sources before any run exists."""
        language = self._source_language()
        if language != self.name:
            from testuary.languages import get_backend

            return get_backend(language).source_backend()
        return self._registered_backend(self._source_context())

    def _source_context(self) -> ProjectContext:
        """The source-only context: the run's base directory, else the working
        directory, without an interpreter."""
        settings = config_module.active_settings()
        base = settings.base_dir if settings is not None else Path.cwd()
        return ProjectContext(str(base), interpreter=None)

    def _source_language(self) -> str:
        """The language the tests are written in (``test_language``), asked
        once per context: the project's in a run, else the source-only
        context's."""
        if self.name in _OWN_TEST_LANGUAGES:
            return self.name
        from testuary.languages import language_project_context

        context = (language_project_context(self.name, None)
                   if config_module.active_settings() is not None else self._source_context())
        known = self.__dict__.setdefault("_source_languages", {})
        if context not in known:
            known[context] = self.test_language()
        return known[context]

    def project_backend(self, source_dir=None):
        """One public backend per project context, rooted at the project and
        running the language's runtime."""
        from testuary.languages import language_project_context
        return self._registered_backend(language_project_context(self.name, source_dir))

    def _supports(self, operation: str, source_dir=None) -> bool:
        group, method = operation.split(".")
        operations = getattr(self.project_backend(source_dir), group, None)
        return operations is not None and operations.supports(method)

    # ---- the target's test files -------------------------------------------

    def is_test_file(self, file_path: Path) -> bool:
        """Whether ``file_path`` is a test file by this language's file
        name rules (``test_*.py``, ``*Test.java``): the language's own
        classification where it offers one, else its test file patterns.
        It reads the name, never the file system, and needs no toolchain.
        Detection by directory (a ``tests/`` ancestor) is the caller's."""
        if self._offers("is_test_file"):
            return self._source_value("is_test_file", str(file_path))
        name = Path(file_path).name
        return (Path(name).suffix.lower() in self.test_file_extensions()
                and any(fnmatch.fnmatchcase(name, pattern)
                        for pattern in self._test_patterns()))

    def target_test_path(self, test_output_dir: str, target_file_path: str) -> Path:
        """The path under ``test_output_dir`` of the test file Testuary
        delivers for the target."""
        return Path(test_output_dir) / self._test_file_name(Path(target_file_path).stem)

    def classify_test_kind(self, filename: str) -> str:
        """The kind of a test file name, the inverse of ``target_test_path``:
        ``"coverage"`` for the file Testuary delivers for a target (both
        agent phases edit it), else ``"unknown"``."""
        return ("coverage" if fnmatch.fnmatchcase(Path(filename).name, self._target_pattern())
                else "unknown")

    def find_target_test_file(self, test_output_dir: str,
                              target_file_path: str) -> Optional[Path]:
        """This target's test file in ``test_output_dir``, which other targets
        may share: the file ``target_test_path`` names (``test_<stem>.py``,
        ``<Stem>Test.java``), else the only file there of the delivered name
        pattern, ``FAILED_MARKER`` aside. None when there is none or more
        than one."""
        expected = self.target_test_path(test_output_dir, target_file_path)
        if expected.is_file():
            return expected
        directory = Path(test_output_dir)
        if not directory.is_dir():
            return None
        candidates = [path for path in sorted(directory.iterdir())
                      if path.is_file() and path.name != FAILED_MARKER
                      and fnmatch.fnmatchcase(path.name, self._target_pattern())]
        return candidates[0] if len(candidates) == 1 else None

    def scratch_test_path(self, directory: str, source: str,
                          target_file_path: str, *,
                          stage_as_delivered: bool) -> Path:
        """A path in ``directory`` to write ``source`` as a test file for a
        one-off measurement (a consolidation's merged suite, a green probe),
        so no caller hard-codes a name or file extension. With
        ``stage_as_delivered`` it is the name the phase delivers for
        ``target_file_path``, so a test that depends on its module name or
        ``__file__`` is judged as it ships. Without it, a fixed scratch stem
        makes the recorded per-test ids and killers read the same for every
        target. Both Python names match pytest's default ``test_*.py``, as
        the consolidation's green check runs pytest on the directory. Under
        a file rule (a public class) the file takes the name the source
        declares, whichever name is asked."""
        required = self.source_backend().file_rule(source)
        if required:
            return Path(directory) / f"{required}{self.test_file_extensions()[0]}"
        if stage_as_delivered:
            return self.target_test_path(directory, target_file_path)
        return Path(directory) / self._test_file_name(self.source_backend().scratch_stem)

    def staged_test_path(self, directory: str, stem: str, source: str) -> Path:
        """Where a phase writes ``source`` as a test file it stages itself
        under the name ``stem`` (a candidate it lists or probes): ``stem``
        with the language's file extension, or, under a file rule (Java's
        top-level class), the name the source declares."""
        required = self.source_backend().file_rule(source)
        return Path(directory) / f"{required or stem}{self.test_file_extensions()[0]}"

    def test_id_prefix(self, path: str) -> str:
        """The prefix of the ids of the tests in the file at ``path``, as the
        language writes it (a declared package before the stem), or the stem
        when the file cannot be read."""
        stem = Path(path).stem
        try:
            text = Path(path).read_text(encoding="utf-8", errors="replace")
        except OSError:
            return stem
        return self.source_backend().test_id_prefix(text, stem)

    def conform_delivered_name(self, source: str, target_file_path: str):
        """For a language with a file rule, ``source`` renamed by the
        language (``rename_file_unit``) to fit the file Testuary delivers
        for the target, as ``(source, old_name, refusal)``. ``old_name`` is
        None when no rename was needed. A refused rename keeps ``source``."""
        backend = self.source_backend()
        expected = self.target_test_path("", target_file_path).stem
        found = backend.file_rule(source)
        if not found or found == expected:
            return source, None, None
        renamed = backend.rename_file_unit(source, expected)
        if not renamed.completed:
            return source, found, renamed.reason or str(renamed.status.value)
        return renamed.value, found, None

    # ---- source reads a language may leave out --------------------------------
    #
    # The agents' edit and merge guards read these. Without the read, the
    # guard is skipped and the run judges the suite. An offered read that
    # fails raises.

    def _offers(self, operation: str) -> bool:
        return self.source_backend().sources.supports(operation)

    def _source_value(self, operation, *args, parse_default=None):
        observation = getattr(self.source_backend().sources, operation)(*args)
        if observation.completed:
            return observation.value
        if observation.status is SourceStatus.PARSE_ERROR and parse_default is not None:
            return parse_default
        raise RuntimeError(
            f"{self.name} sources.{operation}: {observation.status.value}: "
            f"{observation.reason}")

    def check_structure(self, source: str):
        """The language's structural check of ``source``, fit for
        source-preserving rewrites but no judgement of the target parser or
        of execution. A language without the check passes every source."""
        if not self._offers("check_structure"):
            return StructureResult(True, "", 0, 0, None)
        observation = self.source_backend().sources.check_structure(source)
        if not observation.completed:
            raise RuntimeError(
                f"{self.name} sources.check_structure: {observation.status.value}: "
                f"{observation.reason}")
        result = observation.value
        return StructureResult(result.ok, result.message, result.line, result.column,
                               getattr(result, "snippet", None))

    def structural_parser_identity(self) -> str:
        """Stable identity for the parser used by ``check_structure``."""
        if not self._offers("check_structure"):
            return f"{self.name} (no structure check)"
        observation = self.source_backend().sources.check_structure("")
        if not observation.completed or not observation.parser:
            raise RuntimeError(f"{self.name} structural parser unavailable: {observation.reason}")
        return observation.parser

    def test_metrics_observation(self, source: str) -> SourceObservation[TestMetrics]:
        """Structural counts with native availability, reason and parser provenance."""
        return self.source_backend().sources.test_metrics(source)

    def test_metrics(self, source: str) -> Dict[str, Optional[int]]:
        """Structural counts of a test file for the per-test-file CSV
        columns: ``{loc, test_count, assertion_count, fixture_count,
        parametrize_case_count}``, by the language's own meaning of each.
        An unavailable count is None, every count without the language's
        metrics. A valid empty source keeps real zeros."""
        return test_counts(self.test_metrics_observation(source))

    def skeleton(self, source: str):
        """The qualified class and function names in source order, () on a
        parse error or without the operation."""
        if not self._offers("skeleton"):
            return ()
        return self._source_value("skeleton", source, parse_default=())

    def function_fingerprints(self, source: str):
        """``{qualified_name: fingerprint}`` per function definition,
        invariant to formatting and comments, sensitive to any code or
        literal change. Empty on a parse error or without the operation."""
        if not self._offers("function_fingerprints"):
            return {}
        return dict(self._source_value("function_fingerprints", source, parse_default=()))

    def imports(self, source: str):
        """The module-like names ``source`` imports, which the graph builder
        maps to project files, () on a parse
        error or without the operation. Python gives ``x.y`` for both
        ``import x.y`` and ``from x.y import z`` (``z`` can be a name or a
        submodule), and a relative import keeps its leading dots. Java gives
        ``x.y.z`` and ``x.y.Z.m`` from ``import x.y.z`` and
        ``import static x.y.Z.m``, and a wildcard keeps its ``.*``.
        """
        if not self._offers("imports"):
            return ()
        return self._source_value("imports", source, parse_default=())

    # ---- source operations, each with its family's fallback -----------------

    def validate_test_source(self, source: str, source_dir=None) -> SourceValidation:
        """The judgement of the language's validator, else of its structure
        check. Without either the source passes, and the run reports a
        suite that does not load."""
        sources = self.project_backend(source_dir).sources
        if sources.supports("validate_source"):
            return validation_result(sources.validate_source(source), source)
        digest = source_sha256(source)
        if sources.supports("check_structure"):
            observation = sources.check_structure(source)
            if observation.completed:
                result = observation.value
                return SourceValidation(
                    bool(result.ok), "valid" if result.ok else "syntax_error",
                    result.message or "", result.line or 0, result.column or 0,
                    observation.parser or self.name, digest)
        return SourceValidation(True, "valid", "the run judges this source", 0, 0,
                                f"{self.name} run", digest)

    def _recovery(self, result, source: str) -> Tuple[str, List[str]]:
        """A recovery's source and removed items. An offered operation that
        gave no answer raises ``ValidationUnresolved``, as a failed
        validator leaves the suite unjudged."""
        if result.status in _NO_ANSWER:
            from testuary.consolidation import ValidationUnresolved
            raise ValidationUnresolved(result.detail)
        return result.source, list(result.removed)

    def prune_unresolvable_imports(self, source, source_dir):
        """Remove the top-level imports whose modules do not resolve in the
        project's environment, as ``(rewritten, removed_modules)``: the
        salvage step for a generated suite that does not load. Tests that
        used the removed names then fail one by one, for per-test removal.
        Without the operation nothing changes, and such a suite is
        discarded whole."""
        sources = self.project_backend(source_dir).sources
        if not sources.supports("prune_unresolvable_imports"):
            return source, []
        return self._recovery(sources.prune_unresolvable_imports(source), source)

    def prune_collection_breakers(self, source, source_dir, *, timeout):
        """Remove the module-scope statements that raise while the framework
        loads the suite, as ``(rewritten, removed)``: the salvage step for a
        suite whose imports resolve but whose top-level code fails. The load
        failure names the line, its statement goes, and the loop repeats
        under a small bound, returning a rewrite only once the suite loads.
        Tests that needed a removed statement then fail one by one, for
        per-test removal. Without the operation nothing is removed."""
        sources = self.project_backend(source_dir).sources
        if not sources.supports("prune_collection_breakers"):
            return source, []
        return self._recovery(
            sources.prune_collection_breakers(source, timeout_s=timeout), source)

    def salvage_syntax(self, source, source_dir, *, report=None):
        """``source`` after the language's bounded syntax repair, or None
        beyond repair (``salvage_report``: without a repair, a source that
        parses stays and one that does not gives None). ``report`` receives
        the status, repairs and message."""
        result = self.project_backend(source_dir).salvage_report(source)
        if report is not None:
            report.update(status=result.status, repairs=result.repairs,
                          message=result.message)
        return result.source

    def strip_module_level_skips(self, source):
        """Remove the module-scope skips that skip the whole file when they
        fire, as ``(rewritten, n_removed)``. Such a skip (``importorskip`` of
        an absent package) turns a suite of real tests into "no tests
        collected", and in a merged starter pool skips every suite's tests.
        Without it, the tests that needed the guarded import fail one by one,
        for per-test removal. Without the operation nothing changes."""
        return self.project_backend().skip_removal(source)

    def describe_collection_failure(self, test_paths, source_dir, *, timeout):
        """Why the framework cannot load the suite, for the salvage log when
        a suite is given up, or None for a language without a collection
        step, whose suite run names the reason (``failing_test_ids``)."""
        if not self._supports("tests.collect_suite", source_dir):
            return None
        backend = self.project_backend(source_dir)
        run = backend.tests.collect_suite(CollectionRequest(
            self._suite(backend, test_paths), timeout_s=timeout,
            **backend.request_options("collect")))
        return backend.collection_text(run)

    # Removal and finalization read the source as the project's runtime reads
    # it (Python's interpreter), so they run on the project's adapter.

    def prune_tests(self, source, remove_ids):
        """``source`` without the tests ``remove_ids`` names (this file's ids
        from ``collect_eviction_data``), by the language's conventions: a
        test function with its decorators goes when a remove id is a
        ``.``-bounded suffix of its name, a test class only when all its
        tests go (keeping a test is always sound), and module-level code
        stays. Unchanged on a parse error or when nothing matches. The operation belongs to the required family
        ``run tests``, and one that gives no answer raises
        ``ValidationUnresolved``."""
        return self.prune_tests_report(source, remove_ids).source

    def prune_tests_report(self, source, remove_ids) -> PruneOutcome:
        """``source`` without the named tests (``PruneOutcome``): a case id
        (``module.test[id]``) removes the case alone where the language
        can, and ``detail`` says why a named case's test went whole."""
        result = self.project_backend().sources.prune_tests(source, list(remove_ids))
        pruned, _ = self._recovery(result, source)
        return PruneOutcome(pruned, str(result.detail or ""))

    def finalize_test_source(self, source):
        """The language's stable form of ``source`` for validation and
        measurement, ``source`` itself without the operation."""
        sources = self.project_backend().sources
        if not sources.supports("finalize_source"):
            return source
        return self._recovery(sources.finalize_source(source), source)[0]

    def normalize_shadowed_definitions(self, source):
        """``source`` without the definitions a later binding of the same
        scope hides (``SuiteNormalization``). The language owns the binding
        rule and states it in ``rule_version``: a caller must not assume
        whether the first or last binding wins. Deterministic and
        idempotent. A source without a shadowed definition, or a language
        without the operation (``rule_version`` ``none``), is unchanged."""
        from langadapter.sources import SuiteNormalization  # noqa: F811
        if self.source_backend().sources.supports("normalize_bindings"):
            # An offered analysis that fails is an error, never "unchanged".
            return self._source_value("normalize_bindings", source)
        digest = source_sha256(source)
        return SuiteNormalization(
            source=source, changed=False, parse_ok=True, parse_error=None,
            removed_bindings=(), before_sha256=digest, after_sha256=digest,
            rule_version="none")

    def merge_test_files(self, coverage_src, mutation_src):
        """The two suites as one by the language's merge (``merge_sources``,
        checked by the project's own tools), or None when the language
        merges nothing or refuses these two. A merge that gives no answer
        (its tools could not run, timed out or lack their setup) raises:
        keeping the better single candidate would answer a failure that
        says nothing about the suites."""
        backend = self.project_backend()
        if not backend.sources.supports("merge_sources"):
            return None
        # The separator line needs the language's comment prefix.
        prefix = backend.source_merge_separator
        separator = f"{prefix} {MERGE_SEPARATOR}" if prefix else None
        result = backend.sources.merge_sources(coverage_src, mutation_src, separator=separator)
        if result.status in _NO_ANSWER:
            raise RuntimeError(
                f"{self.name} merge did not complete: {result.status}: {result.detail}")
        return result.source if result.status in ("applied", "unchanged") else None

    def resolve_test_collisions(self, sources, *, localize_autouse=False,
                                evidence=None):
        """The language's resolution of name collisions between ``sources``
        (``CollisionResolution``). A colliding name goes to the owner whose
        tests in ``evidence`` carry the most kills, then branches, then
        lines, counted over the tests that depend on the name where the
        language can tell. Without a resolver each source stays as it is,
        and the merge that follows keeps the better single candidate. A
        resolution that does not complete raises ``ValidationUnresolved``."""
        backend = self.project_backend()
        dependents = backend.collision_dependents

        def preference(name, owners):
            weights = {}
            for owner in owners:
                measured = (evidence[owner] if evidence is not None
                            and owner < len(evidence) else None) or {}
                found = dependents(sources[owner], name, measured) if measured else None
                tests = measured if found is None else found
                items = {item for test in tests for item in measured.get(test, ())}
                weights[owner] = tuple(sum(item.startswith(prefix) for item in items)
                                       for prefix in ("K:", "B:", "L:"))
            return max(owners, key=lambda owner: (weights[owner], -owners.index(owner)))

        observation = backend.collision_resolution(
            list(sources), preference=preference, localize_autouse=localize_autouse)
        if observation.status is SourceStatus.UNSUPPORTED and not backend.sources.supports(
                "resolve_collisions"):
            return CollisionResolution(sources=tuple(sources))
        if not observation.completed:
            from testuary.consolidation import ValidationUnresolved
            raise ValidationUnresolved(f"{observation.status.value}: {observation.reason}")
        return observation.value

    def generated_test_target_identity(self, target_file, source_dir):
        """The language's target identity for the validity gate. Without a
        validity policy, an identity whose policy version says so
        (``NO_VALIDITY_POLICY``): the gate is off, its records name no
        policy, and the run records the fallback (``testuary.languages``).
        An offered identity that does not complete raises
        ``ValidityUnresolved``, never a gate that is off unrecorded."""
        from langadapter.sources import TargetIdentity  # noqa: F811
        sources = self.project_backend(source_dir).sources
        if sources.supports("target_identity"):
            observation = sources.target_identity(target_file)
            if observation.completed:
                return observation.value
            raise _validity_unresolved("target_identity", observation)
        try:
            text = Path(target_file).read_text(encoding="utf-8")
        except OSError:
            text = ""
        return TargetIdentity(
            source_path=str(target_file), project_root=str(source_dir),
            module_name=Path(target_file).stem, language=self.name,
            policy_version=NO_VALIDITY_POLICY,
            target_source_sha256=source_sha256(text),
            target_source=text)

    def analyze_generated_test_validity(self, source, target_identity):
        """The policy's findings on ``source``, () without a policy. A source
        that does not parse raises SyntaxError, an analysis that does not
        complete ``ValidityUnresolved``, never a source without findings."""
        if target_identity.policy_version == NO_VALIDITY_POLICY:
            return ()
        observation = self.project_backend(target_identity.project_root).sources \
            .analyze_validity(source, target_identity)
        if observation.status is SourceStatus.PARSE_ERROR:
            raise SyntaxError(observation.reason)
        if not observation.completed:
            raise _validity_unresolved("analyze_validity", observation)
        return tuple(observation.value)

    def screen_generated_test_source(self, source, target_identity):
        """The policy's review-only signals on ``source``, () without a
        policy or screening. A source that does not parse gives one
        ``unparseable_source`` flag, and a screening that does not complete
        raises ``ValidityUnresolved``, never a source without signals."""
        if target_identity.policy_version == NO_VALIDITY_POLICY:
            return ()
        backend = self.project_backend(target_identity.project_root)
        if not callable(getattr(backend, "screen_source", None)):
            return ()
        result = backend.screen_source(source, target_identity)
        if result.status == "parse_error":
            from langadapter.sources import SuiteAuditFlag
            return (SuiteAuditFlag(
                "unparseable_source", result.detail, result.error_line,
                result.error_column, result.error_line, result.error_column),)
        if result.status != "completed":
            raise _validity_unresolved("screen_source", result)
        return tuple(result.items)

    # ---- runs ------------------------------------------------------------------
    #
    # Each run takes the language's request fields for its kind
    # (``request_options``) and setup files (``setup_files``), runs in a
    # prepared copy, and reads ids, counts and coverage as the language writes
    # them (``test_id_text``, ``run_counts``, ``coverage_texts``).

    def _suite(self, backend, test_paths, *, setup: bool = True) -> Suite:
        """The suite of ``test_paths``, rooted at their common directory,
        with the setup files the language loads for them."""
        paths = [Path(path).resolve() for path in test_paths]
        root = Path(os.path.commonpath([str(path.parent) for path in paths]))
        files = tuple(str(path) for path in paths)
        setup_files = (tuple(backend.setup_files(str(root), list(files)))
                       if setup else ())
        return Suite(str(root), files, setup_files)

    @staticmethod
    def _request(options: Dict[str, Any], targets) -> Dict[str, Any]:
        """``options`` with the module under test mapped into the copy."""
        fields = dict(options)
        if fields.get("module_under_test") is not None:
            fields["module_under_test"] = targets[-1]
        return fields

    def setup_test_env(self, test_output_dir, source_dir, target_file_path=None):
        """Create ``test_output_dir`` and have a language that prepares a
        suite directory (its setup files) prepare it. Otherwise
        the project must resolve. Raises RuntimeError when either fails."""
        Path(test_output_dir).mkdir(parents=True, exist_ok=True)
        backend = self.project_backend(source_dir)
        prepare = getattr(backend, "prepare_suite", None)
        if callable(prepare):
            result = prepare(test_output_dir, isolation_config=True)
            if result.outcome != Outcome.COMPLETED:
                raise RuntimeError(f"suite preparation failed: {result.reason}")
            return
        resolution = backend.project()
        if resolution.outcome is not Outcome.COMPLETED:
            raise RuntimeError(f"the project does not resolve: {resolution.reason}")

    def shared_setup_files(self, test_dir: str) -> Tuple[str, ...]:
        """The existing setup files in ``test_dir`` (Python's
        ``conftest.py``) a suite needs at run time that are not test files,
        so Finishing runs them beside the merged suite. () for a
        language without them. The adapters read the directory only, so the
        source-only context answers, also outside a run (the results view)."""
        return tuple(self.source_backend().shared_setup_files(test_dir))

    def failing_test_ids(self, test_paths, source_dir, *, timeout,
                         instrument_target=None, report=None,
                         measure_coverage_of=None, delivered_seed=None):
        """The failing tests' listing ids, or None when the run gave no test
        outcome. ``report`` receives the run's ``outcome``, for a run that
        does not compile or collect, of a language without a collection
        step, its ``reason`` with the compiler's lines (its
        ``reason_detail``, ``collection_errors`` of them), and, for a
        run the timeout ended, the listing id of the test the journal shows
        then running (``active_at_end``). An OSError gives None with outcome
        ``os_error``, unless the prepared copy's setup or cleanup raised it.
        With ``delivered_seed`` the run is the run as delivered with that
        session seed, the normal run of a language whose runs as delivered
        take one (``normal_run``)."""
        report = {} if report is None else report
        backend = self.project_backend(source_dir)
        suite = self._suite(backend, test_paths)
        try:
            if delivered_seed is not None:
                # The module under test makes a prepared copy's tests import
                # the copy, as the normal run's.
                module = [instrument_target] if instrument_target else []
                options = backend.request_options(
                    "delivered", session_seed=delivered_seed, reverse_order=False,
                    **({"target": instrument_target} if module else {}))
                with prepared_measurement(backend, suite, *module) as (
                        selected, staged, targets):
                    run = selected.tests.run_as_delivered(
                        SuiteRunRequest(staged, timeout_s=timeout,
                                        timeout_journal=self.offers_timeout_journal(source_dir),
                                        **(self._request(options, targets) if module
                                           else options)),
                        sources=None)
            else:
                options = backend.request_options("failing", target=instrument_target)
                # Under the coverage instrument the first target is the measured file.
                paths = ((measure_coverage_of, instrument_target) if measure_coverage_of
                         else (instrument_target,))
                with prepared_measurement(backend, suite, *paths) as (
                        selected, staged, targets):
                    covered = {"coverage_target": targets[0]} if measure_coverage_of else {}
                    run = selected.tests.run_suite(SuiteRunRequest(
                        staged, timeout_s=timeout,
                        timeout_journal=self.offers_timeout_journal(source_dir),
                        **covered, **self._request(options, targets)))
        except OSError as error:
            if any(getattr(error, name, None) is not None for name in (
                    "preparation_cleanup_error", "preparation_setup_error")):
                raise
            report["outcome"] = "os_error"
            return None
        # A completed run whose report names no test held none.
        report["outcome"] = (Outcome.NO_TESTS.value
                             if run.outcome in (Outcome.COMPLETED, Outcome.TEST_FAILURES)
                             and not run.tests else run.outcome.value)
        if (run.outcome == Outcome.COLLECTION_FAILED
                and not self._supports("tests.collect_suite", source_dir)):
            report["reason"] = with_detail(run.reason, tuple(_limits.cut_lines(
                run.reason_detail, _limits.value("collection_errors"),
                what="further errors")), None)
        if run.outcome == Outcome.TIMED_OUT and run.active_at_end is not None:
            report["active_at_end"] = case_id(selected, run.active_at_end)
        # Only a report the language trusts, with a parametrized test's
        # failing cases by their ids.
        failed = selected.listed_failures(run)
        if failed is None:
            return None
        return sorted({case_id(selected, test) for test in failed})

    def runs_as_delivered(self, source_dir=None) -> bool:
        """Whether the language runs a suite as delivered
        (``run_as_delivered``)."""
        return self._supports("tests.run_as_delivered", source_dir)

    def offers_reverse_order(self, source_dir=None) -> bool:
        """Whether the language runs a suite as delivered in reverse order,
        for the loops' reverse-order note: not where it has no runs as
        delivered or its adapter refuses that order
        (``delivered_order_refusal``: Java's, since JUnit orders a class's
        tests itself)."""
        if not self.runs_as_delivered(source_dir):
            return False
        refusal = getattr(self.project_backend(source_dir), "delivered_order_refusal", None)
        return not (callable(refusal) and refusal(True))

    def offers_timeout_journal(self, source_dir=None) -> bool:
        """Whether the language's suite runs take the per-test timeout
        journal (``SuiteRunRequest.timeout_journal``), as the project's
        adapter states in ``provides_timeout_journal``: True for one that
        states nothing."""
        return bool(getattr(self.project_backend(source_dir), "provides_timeout_journal", True))

    def seeds_delivered_runs(self, source_dir=None) -> bool:
        """Whether the language's run as delivered takes a session seed
        (``SuiteRunRequest.session_seed``), as the project's adapter states
        in ``seeds_delivered_runs``: False for one that states nothing."""
        return bool(getattr(self.project_backend(source_dir), "seeds_delivered_runs", False))

    def run_test_files(self, test_paths, source_dir, *, timeout,
                       measure_coverage_of=None, coverage_matrix_path=None,
                       instrument_target=None, fail_fast=False, tb="short",
                       record_leaks=False, timeout_journal=False):
        """Run ``test_paths`` once, the one correctness and coverage run, so
        no caller builds a test command. Timing is ``benchmark_tests``' work.

        ``test_paths`` is the exact set, a directory standing for the test
        files the framework finds in it, and ``source_dir`` the import or
        build root. Past ``timeout``, ``timed_out`` is True and ``ok`` False.
        ``measure_coverage_of`` (a source file) fills the coverage fields
        for that file, None when the measurement fails, and
        ``coverage_matrix_path`` receives that run's line and branch
        records. ``instrument_target`` names the module under test on a run
        without coverage. ``fail_fast`` and ``tb`` ("short", "no" or "long") apply
        where the language has them. ``record_leaks`` asks for
        ``TestRunResult.leaks``, ``timeout_journal`` for ``active_at_end``:
        pass it only when ``offers_timeout_journal`` is True.
        """

        backend = self.project_backend(source_dir)
        test_paths = backend.test_files([str(path) for path in test_paths])
        with phase_state.measured_part(
                phase_state.TIME_COVERAGE if measure_coverage_of
                else phase_state.TIME_TEST) as charge:
            suite = self._suite(backend, test_paths)
            module = measure_coverage_of or instrument_target
            options = backend.request_options(
                "run", target=module, traceback=tb, stop_at_first_failure=fail_fast,
                record_leaks=record_leaks)
            with prepared_measurement(backend, suite, measure_coverage_of, module) as (
                    selected, staged, targets):
                with charge.timing():
                    run = selected.tests.run_suite(SuiteRunRequest(
                        staged, timeout_s=timeout, coverage_target=targets[0],
                        timeout_journal=timeout_journal,
                        **self._request(options, targets)))
                    measured = measured_seconds(run, bool(coverage_matrix_path))
                    charge.observed(measured)
                counts = selected.run_counts(run)
                failures = tuple(
                    TestFailure(selected.test_id_text(case.test), case.outcome,
                                case.message or "", case.trace or "")
                    for case in run.tests if case.outcome in ("failed", "error"))
                texts = (selected.coverage_texts(run.coverage)
                         if run.coverage is not None else None)
                leaks = None if run.leaks is None else tuple(LeakRecord(
                    test=selected.test_id_text(leak.test), nodeid=str(leak.test),
                    added=leak.added, rebound=leak.rebound, removed=leak.removed,
                    xarray_options=leak.xarray_options, numpy_errors=leak.numpy_errors,
                    numpy_print=leak.numpy_print) for leak in run.leaks)
            coverage = run.coverage
            output = run.output
            ok = run.outcome == Outcome.COMPLETED
            if coverage_matrix_path and coverage is not None:
                started = time.perf_counter()
                payload = coverage.to_coverage_matrix()
                payload["target_file"] = measure_coverage_of
                try:
                    Path(coverage_matrix_path).write_text(
                        json.dumps(payload, indent=2), encoding="utf-8")
                except OSError as exc:
                    coverage = None
                    ok = False
                    output += f"\ncoverage matrix could not be written: {exc}"
                if measured is not None:
                    # The matrix is written inside the timed run.
                    charge.observed(measured + time.perf_counter() - started)
            return TestRunResult(
                # A run without a process exit code (an in-process runner)
                # gets the one its outcome means.
                ok=ok, returncode=(run.returncode if run.returncode is not None
                                   else 0 if run.outcome == Outcome.COMPLETED
                                   else 1 if run.outcome == Outcome.TEST_FAILURES else -1),
                timed_out=run.outcome == Outcome.TIMED_OUT, elapsed_s=run.wall_time_s,
                output=output or run.reason or "",
                collected=counts["collected"], passed=counts["passed"],
                failed=counts["failed"], errors=counts["errors"],
                line_coverage=coverage.line_percent if coverage is not None else None,
                branch_coverage=coverage.branch_percent if coverage is not None else None,
                missing_lines=coverage.missing() if coverage is not None else (),
                missing_branches=tuple(texts["missing_branches"])
                if coverage is not None else (),
                covered_lines=(tuple(texts["covered_lines"])
                               if texts["covered_lines"] is not None else None)
                if coverage is not None else None,
                covered_branches=tuple(f"{b.source}->{b.destination}"
                                       for b in coverage.taken)
                if coverage is not None else None,
                leaks=leaks, failures=failures,
                active_at_end=(
                    run.active_at_end.qualname + case_suffix(run.active_at_end.params)
                    if run.active_at_end is not None else None),
                total_statements=len(coverage.statements) if coverage is not None else None,
                total_branches=len(coverage.branches) if coverage is not None else None,
                unsound=_unsound(run), diagnostics=tuple(run.diagnostics),
                reason_detail=tuple(run.reason_detail),
                stopping_tests=_failed_ids(run))

    def run_as_delivered(self, test_paths, source_dir, *, timeout,
                         reverse=False, seed=1, sources=None, target=None,
                         fail_fast=False, tb=None, record_leaks=False,
                         timeout_journal=False):
        """Each file run alone, as a user who copied it would run it, with
        ``sources`` replacing listed test files' text. Without the
        operation, or for an order the language refuses, a run without
        outcomes whose ``reason`` says why. ``fail_fast`` and ``tb`` are
        ``run_test_files``' (``tb`` None keeps the language's own),
        ``target`` is the module under test, which makes a prepared copy's
        tests import the copy, ``record_leaks`` asks the language's
        observer, which restores nothing, for the tests that left it or a
        library setting changed, and ``timeout_journal`` for the test
        running when a timeout ended the run. The runner's own time is test
        time, as ``run_test_files``' is: the normal run's and the
        reverse-order note's runs alike."""
        if not self._supports("tests.run_as_delivered", source_dir):
            return DeliveredRun(
                cases=(), failing=None, output="", reverse=reverse, seed=seed,
                reason=f"{self.name} has no run as delivered")
        backend = self.project_backend(source_dir)
        refusal = getattr(backend, "delivered_order_refusal", None)
        refused = refusal(reverse) if callable(refusal) else None
        if refused:
            return DeliveredRun(cases=(), failing=None, output="", reverse=reverse,
                                seed=seed, reason=refused)
        suite = self._suite(backend, test_paths)
        replacements = relative_sources(suite, sources)
        wishes = {"traceback": tb} if tb is not None else {}
        if fail_fast:
            wishes["stop_at_first_failure"] = True
        if record_leaks:
            wishes["record_leaks"] = True
        if target:
            wishes["target"] = target
        options = backend.request_options("delivered", session_seed=seed,
                                          reverse_order=reverse, **wishes)
        module = [target] if target else []
        with phase_state.measured_part(phase_state.TIME_TEST) as charge, \
                prepared_measurement(backend, suite, *module) as (selected, staged, targets):
            with charge.timing():
                run = selected.tests.run_as_delivered(
                    SuiteRunRequest(staged, timeout_s=timeout,
                                    timeout_journal=bool(timeout_journal)
                                    and self.offers_timeout_journal(source_dir),
                                    **(self._request(options, targets) if module else options)),
                    sources=replacements)
                charge.observed(measured_seconds(run, False))
            cases = tuple(DeliveredCase(*selected.delivered_case(t.test, t.outcome),
                                        t.message) for t in run.tests)
            # Named as the cases are, the test's form in the failing listing.
            leaks = None if run.leaks is None else tuple(LeakRecord(
                test=selected.delivered_case(leak.test, "passed")[0], nodeid=str(leak.test),
                added=leak.added, rebound=leak.rebound, removed=leak.removed,
                xarray_options=leak.xarray_options, numpy_errors=leak.numpy_errors,
                numpy_print=leak.numpy_print) for leak in run.leaks)
        decided = run.outcome in (Outcome.COMPLETED, Outcome.TEST_FAILURES)
        return DeliveredRun(
            cases=cases,
            failing=tuple(sorted({case.test for case in cases
                                  if case.outcome in ("failed", "error")}))
            if decided else None,
            reason=None if decided
            else run.reason_text(_limits.value("run_reason_lines")) or str(run.outcome),
            output=run.output, reverse=reverse, seed=seed,
            timed_out=run.outcome == Outcome.TIMED_OUT, elapsed_s=run.wall_time_s,
            leaks=leaks, active_at_end=(
                run.active_at_end.qualname + case_suffix(run.active_at_end.params)
                if run.active_at_end is not None else None),
            unsound=_unsound(run), stopping_tests=_failed_ids(run))

    def collect_eviction_data(self, test_paths, source_dir, *,
                              measure_coverage_of, timeout, warmup=True,
                              kill_matrix_path=None, log=None, failure_fn=None):
        """Per-test evidence (``EvictionData``), or None when the language has
        none or the run fails (``failure_fn`` then gets the failure's kind
        and reason, and ``stopping_tests``, the id objects of the tests it
        names, where it names any): the reductions are then skipped. With
        ``kill_matrix_path`` the matrix's kills join the tokens when they
        are complete (``includes_kills``). Every run keeps ``warmup``
        True. False, which skips the warm-up run, is a seam for tests, and
        the demonstration's recordings key their answers by ``warmup``."""
        if not self._supports("tests.per_test_evidence", source_dir):
            if log is not None:
                log(f"[evict] {self.name} has no per-test evidence")
            return None
        backend = self.project_backend(source_dir)
        suite = self._suite(backend, test_paths)
        options = backend.request_options("evidence", target=measure_coverage_of)
        with prepared_measurement(backend, suite, measure_coverage_of) as (
                selected, staged, targets):
            result = selected.tests.per_test_evidence(PerTestEvidenceRequest(
                staged, targets[0], timeout_s=timeout, warmup=warmup,
                **self._request(options, targets)))
            text_of = selected.test_id_text
            kill_tokens = selected.kill_tokens
        if result.outcome != Outcome.COMPLETED:
            if log is not None:
                log(f"[evict] {result.reason}")
            if failure_fn is not None:
                # The tests the failure is pinned on, as id objects, where
                # the result names any. A timed-out run names none.
                named = (() if result.outcome == Outcome.TIMED_OUT
                         else tuple(getattr(result, "stopping_tests", ()) or ()))
                failure_fn("test_failure" if result.outcome == Outcome.TEST_FAILURES
                           else "infrastructure", result.reason,
                           **({"stopping_tests": named} if named else {}))
            return None
        tokens = {text_of(test.test): test.tokens() for test in result.tests}
        if self._keeps_unmatched_contexts(source_dir):
            # Kept beside the tests, without claiming to be tests.
            tokens.update({context.context: frozenset(item.token() for item in context.items)
                           for context in result.unmatched_items})
        elif result.unmatched_items and log is not None:
            log(f"[evict] {len(result.unmatched_items)} coverage context(s) name no "
                "test of the suite and are left out: "
                + ", ".join(context.context for context in result.unmatched_items))
        times = {text_of(test.test): test.time_ms for test in result.tests
                 if test.time_ms is not None}
        includes_kills = False
        if kill_matrix_path:
            try:
                matrix = upgrade_kill_matrix(
                    json.loads(Path(kill_matrix_path).read_text(encoding="utf-8")))
                kills, unmatched = kill_tokens(matrix, set(tokens))
                # A kill without a killer (the target's import fails under
                # the mutant) is reproduced by every kept test and needs no
                # token, so a matrix of only such kills is complete. An
                # unresolved kill is not one, and an unmatched killer leaves
                # the kills incomplete.
                needs_killer = any(mutant.get("killers") or mutant.get("killer")
                                   or mutant.get("unresolved")
                                   for mutant in matrix.get("mutants") or ())
                includes_kills = (bool(kills) or not needs_killer) and not unmatched
            except (OSError, ValueError, TypeError, AttributeError):
                kills = {}
            if includes_kills:
                for test, items in kills.items():
                    tokens[test] = tokens[test] | items
        coverage = result.coverage
        return EvictionData(
            per_test_tokens=tokens, per_test_time_ms=times,
            suite_time_ms=result.suite_time_ms, includes_kills=includes_kills,
            target_total_lines=len(coverage.statements) if coverage is not None else None,
            target_covered_lines=len(coverage.covered) if coverage is not None else None,
            target_total_branches=len(coverage.branches) if coverage is not None else None,
            target_covered_branches=len(coverage.taken) if coverage is not None else None,
            suite_covered_lines=frozenset(coverage.covered) if coverage is not None else None,
            suite_covered_branches=frozenset(f"{b.source}->{b.destination}"
                                             for b in coverage.taken)
            if coverage is not None else None,
            first_test=text_of(result.tests[0].test) if result.tests else None)

    def _keeps_unmatched_contexts(self, source_dir=None) -> bool:
        """Whether the language's per-test evidence keeps the contexts no
        reported test claims beside the tests: the adapter's class fact
        ``unmatched_evidence_kept``, else its ``LanguageFacts`` for the
        project. Off when neither says: such a context (a test id the
        language did not decode) is no test the reduction can keep or
        remove, and as a pseudo-test it would enter the cover at an imputed
        cost."""
        declared = getattr(self._adapter_class(), "unmatched_evidence_kept", None)
        if declared is not None:
            return bool(declared)
        facts = self.language_facts(source_dir)
        return bool(getattr(facts, "unmatched_evidence_kept", False))

    def attribute_pool_test_kills(self, kill_matrix, file_to_index, test_ids_by_index):
        """Named per-test kill tokens, when the backend supports attribution."""
        return self.project_backend().pool_kills(kill_matrix, file_to_index, test_ids_by_index)

    def capture_coverage_matrix(self, *, target, source_dir, test_paths,
                                out_path, timeout):
        """Write the coverage-matrix JSON to ``out_path``, the counterpart of
        the kill matrix: the per-line sets ``tests.csv`` cannot hold, in one
        schema for every language::

            {"schema_version": int,
             "target_file": str,
             "lines": [{"line": int, "status": "covered"|"missing"}],
             "branches": [{"from": int, "to": int, "status": "taken"|"missing"}],
             "summary": {"total_lines","covered_lines","missing_lines",
                         "covered_branches","missing_branches"}}

        A run without coverage writes nothing, and the reader shows n/a,
        as for an absent kill matrix."""
        backend = self.project_backend(str(source_dir))
        if getattr(backend, "coverage_matrix_in_place", False):
            # Measured in the project itself, with container paths as the
            # container spells them.
            guest = config_module.project_exec_mode() == "container"
            target_name = target.as_posix() if guest else str(target)
            root = Path(source_dir).as_posix() if guest else str(source_dir)
            backend = self.project_backend(root)
            suite = self._suite(backend, test_paths)
            options = backend.request_options("coverage_matrix", target=target_name)
            run = backend.tests.run_suite(SuiteRunRequest(
                suite, timeout_s=timeout, coverage_target=target_name, **options))
        else:
            target_name = str(target)
            suite = self._suite(backend, test_paths)
            options = backend.request_options("coverage_matrix", target=target_name)
            with prepared_measurement(backend, suite, target_name) as (
                    selected, staged, targets):
                run = selected.tests.run_suite(SuiteRunRequest(
                    staged, timeout_s=timeout, coverage_target=targets[0],
                    **self._request(options, targets)))
        if run.coverage is None:
            return
        payload = run.coverage.to_coverage_matrix()
        payload["target_file"] = target_name
        Path(out_path).write_text(json.dumps(payload, indent=2), encoding="utf-8")

    def benchmark_tests(self, *, source_dir, test_file_paths, warmup_runs,
                        benchmark_runs, timeout, suite_label="delivered",
                        progress_fn=None, instrument_target=None, **_unused):
        """A timing benchmark through the language's own, or a failed one
        naming the timing family's fallback."""
        if not self._supports("tests.benchmark", source_dir):
            return {"ok": False, "unavailable": True, "times_ms": [],
                    "requested_cpus": None, "allowed_cpus": None, "returncode": -1,
                    "error": f"{self.name} has no timing benchmark",
                    "failure_kind": "infrastructure"}

        def progress(event):
            if progress_fn is None:
                return
            kind = "warmup" if event.kind == "warmup" else "rep"
            progress_fn(f"[benchmark:{suite_label}] {kind} "
                        f"{event.index}/{event.total} ({event.elapsed_ms:.0f}ms)")

        backend = self.project_backend(source_dir)
        cpus = (str(config_module.load_config("timing").get("cpuset", "")).strip()
                or None) if config_module.project_exec_mode() == "container" else None
        suite = self._suite(backend, test_file_paths)
        options = backend.request_options("benchmark", target=instrument_target)
        with benchmark_lock.acquire():
            with prepared_measurement(backend, suite, instrument_target) as (
                    selected, staged, targets):
                result = selected.tests.benchmark(BenchmarkRequest(
                    suite=staged, runs=benchmark_runs, warmup_runs=warmup_runs,
                    timeout_s=timeout, serialize=False, cpus=cpus, progress=progress,
                    **self._request(options, targets)))
        stats = {"requested_cpus": result.cpus,
                 "allowed_cpus": list(result.allowed_cpus) if result.allowed_cpus else None,
                 "ok": result.outcome == Outcome.COMPLETED,
                 "times_ms": list(result.times_ms)}
        if result.outcome == Outcome.TIMED_OUT:
            return {**stats, "timed_out": True, "returncode": -1,
                    "error": f"{suite_label} suite timed out after {timeout}s"}
        if result.outcome != Outcome.COMPLETED:
            return {**stats, "returncode": result.returncode, "error": result.reason,
                    "failure_kind": "test_failure" if (
                        result.outcome == Outcome.TEST_FAILURES) else "infrastructure"}
        return {**stats, "warmup_runs": warmup_runs, "timing_run_count": benchmark_runs,
                "median_ms": result.median_ms, "mean_ms": result.mean_ms,
                "min_ms": result.min_ms, "max_ms": result.max_ms,
                "early_stopped": False, "early_stop_reason": None}

    def benchmark_interleaved(self, *, source_dir, suites, warmup_runs,
                              benchmark_runs, timeout, progress_fn=None,
                              instrument_target=None):
        """Several suites, ``(label, test_file_paths)`` pairs, timed round by
        round so a drift in the machine's load falls on all alike. Each gets
        one prepared copy before the first round, outside the timer. After
        ``warmup_runs`` warm-up rounds come ``benchmark_runs`` timed ones,
        each running every remaining suite once under the benchmark lock,
        the starting suite rotating. A suite whose tests fail leaves the
        rounds, and any other failure ends them. The result holds ``rounds``
        (number, kind and label order), ``suites`` (each label's stats as
        ``benchmark_tests`` gives them, with the discarded
        ``warmup_times_ms``) and ``ok``, or without a timing benchmark
        ``benchmark_tests``' unavailable result with no rounds."""
        import statistics
        from contextlib import ExitStack

        labels = [str(label) for label, _ in suites]
        if len(set(labels)) != len(labels):
            raise ValueError(f"interleaved benchmark labels repeat: {labels!r}")
        if not self._supports("tests.benchmark", source_dir):
            return {"ok": False, "unavailable": True, "rounds": [], "suites": {},
                    "error": f"{self.name} has no timing benchmark",
                    "failure_kind": "infrastructure"}
        backend = self.project_backend(source_dir)
        cpus = (str(config_module.load_config("timing").get("cpuset", "")).strip()
                or None) if config_module.project_exec_mode() == "container" else None
        options = backend.request_options("benchmark", target=instrument_target)
        records = {label: {"ok": False, "times_ms": [], "warmup_times_ms": [],
                           "requested_cpus": cpus, "allowed_cpus": None}
                   for label in labels}
        rounds: List[Dict[str, Any]] = []
        stopped = None
        with ExitStack() as stack:
            live = []
            for label, paths in zip(labels, (paths for _, paths in suites)):
                live.append((label, stack.enter_context(kept_measurement(
                    backend, self._suite(backend, paths), instrument_target))))
            for number in range(warmup_runs + benchmark_runs):
                if not live or stopped is not None:
                    break
                start = number % len(live)
                order = live[start:] + live[:start]
                warm = number < warmup_runs
                rounds.append({"round": number + 1, "warmup": warm,
                               "order": [item[0] for item in order]})
                for item in order:
                    label, use = item
                    with use() as (selected, staged, targets):
                        request = BenchmarkRequest(
                            suite=staged, runs=1, warmup_runs=0, timeout_s=timeout,
                            serialize=False, cpus=cpus, **self._request(options, targets))
                        with benchmark_lock.acquire():
                            result = selected.tests.benchmark(request)
                    record = records[label]
                    record["requested_cpus"] = result.cpus
                    record["allowed_cpus"] = (list(result.allowed_cpus)
                                              if result.allowed_cpus else None)
                    if result.outcome == Outcome.COMPLETED and len(result.times_ms) == 1:
                        record["warmup_times_ms" if warm else "times_ms"].append(
                            float(result.times_ms[0]))
                        continue
                    if result.outcome == Outcome.COMPLETED:
                        record.update(returncode=result.returncode,
                                      failure_kind="infrastructure",
                                      error=f"one execution of {label} returned "
                                            f"{len(result.times_ms)} times")
                    elif result.outcome == Outcome.TIMED_OUT:
                        record.update(timed_out=True, returncode=-1,
                                      error=f"{label} suite timed out after {timeout}s")
                    else:
                        record.update(returncode=result.returncode, error=result.reason,
                                      failure_kind="test_failure" if (
                                          result.outcome == Outcome.TEST_FAILURES)
                                      else "infrastructure")
                    if record.get("failure_kind") == "test_failure":
                        live.remove(item)
                        continue
                    stopped = label
                    break
                if progress_fn is not None:
                    kind = "warmup" if warm else "round"
                    index = number + 1 if warm else number + 1 - warmup_runs
                    total = warmup_runs if warm else benchmark_runs
                    progress_fn(f"[benchmark:interleaved] {kind} {index}/{total}: "
                                + ", ".join(rounds[-1]["order"]))
        for label in labels:
            record = records[label]
            if "error" in record:
                continue
            if stopped is not None:
                record.update(failure_kind="infrastructure", incomplete=True,
                              error=f"the rounds ended when {stopped} failed")
                continue
            times = record["times_ms"]
            record.update(ok=True, warmup_runs=warmup_runs, timing_run_count=benchmark_runs,
                          median_ms=statistics.median(times) if times else None,
                          mean_ms=statistics.fmean(times) if times else None,
                          min_ms=min(times) if times else None,
                          max_ms=max(times) if times else None,
                          early_stopped=False, early_stop_reason=None)
        return {"ok": all(records[label]["ok"] for label in labels),
                "rounds": rounds, "suites": records}

    def order_dependent_mock_warnings(self, source: str) -> List[str]:
        """The tests in ``source`` that assert on the order in which a
        stubbed multi-return sequence is consumed (Python: a mock
        ``side_effect`` list). A scheduler behind the code under test can
        dispatch the stubbed calls in an order the test does not control,
        so the outcome can differ between identical runs. Advisory: callers
        log the names and never remove or gate on them. [] without the
        check or for a source the parser cannot read."""
        from langadapter import SourceReadError

        warnings = getattr(self.project_backend(), "order_dependent_mock_warnings", None)
        if not callable(warnings):
            return []
        try:
            return list(warnings(source))
        except SourceReadError:
            return []

    # ---- the environment and the model's vocabulary -------------------------

    def import_statement(self, target_file_path: str, source_dir: str) -> str:
        """The language's import line for the file under test, or "". It
        depends only on the target and the directory its package is imported
        from (``source_dir`` with ``[project] python_import_root`` below it
        for a Python target, as ``project.display_path`` names the file), so
        the adapter's class method gives it without a project context."""
        from testuary import project

        root = project._below(source_dir, project._python_import_dir(target_file_path))
        return self._adapter_class().import_statement(target_file_path, root) or ""

    def code_smells(self, paths) -> Optional[int]:
        """The language's linter findings in ``paths``, or None without one."""
        return self.project_backend().code_smells([str(path) for path in paths])

    def check_dependencies(self) -> List[str]:
        """The missing external tools a run needs beyond the operations the
        capability report covers (``capabilities``), each a line naming the
        tool and how to provide it, [] when none is missing. Called once
        before any phase, so a missing tool stops the run with a complete
        list: the exact reduction solver's tools when a phase uses it, the
        same for every language."""
        return self._solver_dependencies()

    def _solver_dependencies(self) -> List[str]:
        """PuLP and its solver when the exact reduction solver is active in
        any phase, [mutation] read over [coverage] as the mutation phase reads
        it: the reduction is the same for every language."""
        import importlib.util as importlib_util


        missing: List[str] = []
        exact = any(
            str(config_module.get_agent_config(*sections)["reduction_solver"]
                ).strip().lower() == "exact"
            for sections in (("starter",), ("coverage",), ("mutation", "coverage")))
        if exact:
            if importlib_util.find_spec("pulp") is None:
                missing.append(
                    "pulp (pip install pulp): required for "
                    "reduction_solver=exact, or set reduction_solver=greedy "
                    "for the heuristic"
                )
            else:
                from testuary.project import host_solver_mismatches

                missing.extend(host_solver_mismatches())
        return missing

    def capabilities(self, source_dir: Optional[str] = None):
        """The language's capability report
        (``langadapter.capabilities.CapabilityReport``) for the project at
        ``source_dir``: which operations are available, unsupported or
        missing setup, and why. Before any phase the run assesses it against
        Testuary's requirement profile (``testuary.languages.assess``): a
        missing required family stops the run, a missing optional one takes
        its recorded fallback."""
        return self.project_backend(source_dir).probe_capabilities()

    def prompt_fragments(self, source_dir: Optional[str] = None) -> Dict[str, str]:
        """The language's vocabulary for the test-writing prompts, built
        from its facts (``testuary.prompts.fragments``): a flat
        ``{slot_name: text}`` map whose keys match the ``$slot_name``
        placeholders of the shared per-agent prompt JSON, one slot set for
        every language. With ``source_dir`` the facts can follow the
        project (the JUnit major version its build declares)."""
        from testuary.prompts import fragments
        return fragments(self, source_dir)

    def mutation_runner(self, config: Dict[str, Any], *,
                        source_dir: Optional[str] = None,
                        log: Optional[Any] = None) -> MutationRunner:
        """Mutation runs through the engine :func:`_mutation_engine` chooses
        among the language's native engines. Any other engine is refused."""
        backend = self.project_backend(source_dir)
        engines = tuple(backend.native_mutation_engines or ())
        if not engines:
            raise ValueError(f"{self.name} has no mutation engine")
        engine = _mutation_engine(backend, config.get("engine"))
        if engine is None:
            name = self._fact("display_name") or self.name.capitalize()
            valid = ", ".join(repr(engine) for engine in engines)
            raise ValueError(f"Unknown {name} mutation engine: "
                             f"{str(config.get('engine') or '').lower().strip()!r}. "
                             f"Valid: {valid} (default).")
        return GenericMutationRunner(backend, config, engine=engine, log=log)


def _mutation_engine(backend: Any, configured: Any) -> Optional[str]:
    """The mutation engine a run of ``backend``'s language uses for the
    configured ``engine`` value: that engine when it is one of the
    backend's native engines, the first of them for the setting's shipped
    default or an empty value, else None. The records of a run name this
    engine, while the measurement identities hash the value as configured."""
    from testuary.settings import _defaults, phase_section
    engines = tuple(getattr(backend, "native_mutation_engines", ()) or ())
    value = str(configured or "").lower().strip()
    if value in engines:
        return value
    shipped = str(_defaults().get(phase_section("mutation"), {}).get("engine", "")).lower().strip()
    return engines[0] if engines and value in ("", shipped) else None


def _unsound(run: Any) -> Tuple[Tuple[Optional[str], str], ...]:
    """A suite run's ``unsound_test`` diagnostics as ``TestRunResult.unsound``:
    the test as the run printed it, or None, and the matched text."""
    return tuple((entry.native_test, entry.message) for entry in run.diagnostics
                 if entry.kind == "unsound_test")


def validation_result(observation: SourceObservation[NativeValidation], source: str) -> SourceValidation:
    """``observation`` as a ``SourceValidation``: one that did not complete
    fails with its status, never judging the source."""
    if not observation.completed:
        return SourceValidation(
            False, "timeout" if observation.status is SourceStatus.TIMEOUT else "validator_error",
            observation.reason, 0, 0, observation.parser or "validator identity unavailable",
            source_sha256(source))
    result = observation.value
    return SourceValidation(
        result.valid, result.status, result.message, result.line, result.column,
        result.interpreter, result.source_sha256)


TEST_COUNTERS = {
    "loc": "code_lines", "test_count": "tests", "assertion_count": "assertions",
    "fixture_count": "fixtures", "parametrize_case_count": "parametrized_cases",
}


def test_counts(observation):
    """Testuary's file counters (``TEST_COUNTERS``) from a test-metrics
    observation, each None unless the source parsed."""
    counts = dict.fromkeys(TEST_COUNTERS)
    if observation.completed and observation.value.parsed:
        for name, field in TEST_COUNTERS.items():
            counts[name] = getattr(observation.value, field)
    return counts


def relative_sources(suite: Suite, sources: Optional[Mapping[str, str]]):
    """``sources`` (replacement text by test path) keyed by the suite-relative
    paths of its listed test files, None for None. Any other path raises
    ``InvalidRequest``."""
    if sources is None:
        return None
    if not isinstance(sources, Mapping):
        raise InvalidRequest("sources", "must map test paths to replacement text")
    root = Path(suite.root).resolve()
    declared = {item.path for item in suite.identity()[:len(suite.test_files)]}
    replacements = {}
    for name, source in sources.items():
        if not isinstance(name, str) or not isinstance(source, str):
            raise InvalidRequest("sources", "test paths and replacement text must be strings")
        path = Path(name)
        try:
            relative = (path if path.is_absolute() else root / path).resolve().relative_to(root).as_posix()
        except ValueError:
            raise InvalidRequest("sources", f"not a listed test source: {name}") from None
        if relative not in declared:
            raise InvalidRequest("sources", f"not a listed test source: {name}")
        replacements[relative] = source
    return replacements
