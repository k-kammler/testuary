"""A made-up language at Testuary's level 1, registered from outside.

A Calc source file defines one function per line::

    double(x) = x * 2
    add(a, b) = a + b

and a Calc test file holds one test per line::

    test doubles: double(2) == 4

where both sides are integer arithmetic over the project's functions. Blank
and ``#`` lines are ignored. The adapter offers only what level 1 requires:
suite runs with per-test and run outcomes, line coverage (a line is covered
when its function was called), a mutation session (each ``+``, ``-`` and
``*`` of a target line replaced) and the test units (removing and counting
single tests). Every other operation is unsupported, so a run takes each
optional family's fallback.
"""
from __future__ import annotations

import ast
import hashlib
import operator
import re
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import langadapter as la
from langadapter.capabilities import CAPABILITIES
from langadapter.mutation import (
    BY_FAILING_TEST, SURVIVED, Classification, MutationClass,
)
from langadapter.mutation import OperatorCatalogue
from langadapter.mutation import Mutant, MutationRun, mutant_key
from langadapter.mutation import MutationSession
from langadapter.runs import (
    ERROR, FAILED, PASSED, CoverageObservation, RunCounts, SuiteRun, TestOutcome,
)
from langadapter.versions import Version
from langadapter.sources import (
    SourceResult, SourceValidation, StructureCheck, TestMetrics,
)

LANGUAGE = "calc"
ENGINE = "calc-mutate"
PARSER = "calc-parser 1"
TEST_LINE = re.compile(r"^test (?P<name>[A-Za-z_]\w*):\s*(?P<expr>.+)$")
FUNCTION_LINE = re.compile(r"^(?P<name>[A-Za-z_]\w*)\((?P<args>[^)]*)\)\s*=\s*(?P<body>.+)$")
OPERATORS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
             ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
#: Each mutation operator: the operator's text and what replaces it.
SWAPS = (("+", "-"), ("-", "+"), ("*", "+"))


class _Functions:
    """The project's functions, recording which lines a run called."""

    def __init__(self, definitions: Dict[str, Tuple[List[str], str, str, int]]):
        self.definitions = definitions
        self.called: Dict[str, set] = {}

    def call(self, name: str, values: List[int]) -> int:
        if name not in self.definitions:
            raise ValueError(f"no function {name}")
        params, body, path, line = self.definitions[name]
        if len(params) != len(values):
            raise ValueError(f"{name} takes {len(params)} arguments")
        self.called.setdefault(path, set()).add(line)
        return self.evaluate(ast.parse(body, mode="eval").body, dict(zip(params, values)))

    def evaluate(self, node, names: Dict[str, int]) -> int:
        if isinstance(node, ast.Constant) and type(node.value) is int:
            return node.value
        if isinstance(node, ast.Name) and node.id in names:
            return names[node.id]
        if isinstance(node, ast.BinOp) and type(node.op) in OPERATORS:
            return OPERATORS[type(node.op)](self.evaluate(node.left, names),
                                            self.evaluate(node.right, names))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -self.evaluate(node.operand, names)
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and not node.keywords):
            return self.call(node.func.id, [self.evaluate(a, names) for a in node.args])
        raise ValueError(f"not Calc arithmetic: {ast.unparse(node)}")


def _definitions(root: Path, replaced: Optional[Tuple[Path, str]] = None):
    """Every function of the project's source files, with ``replaced``
    giving one file's text in place of what is on disk."""
    definitions = {}
    for path in sorted(root.rglob("*.calc")):
        if path.name.startswith("test_"):
            continue
        text = (replaced[1] if replaced is not None and path.resolve() == replaced[0]
                else path.read_text(encoding="utf-8"))
        for number, line in enumerate(text.splitlines(), start=1):
            match = FUNCTION_LINE.match(line.strip())
            if match:
                params = [p.strip() for p in match["args"].split(",") if p.strip()]
                definitions[match["name"]] = (params, match["body"], str(path.resolve()), number)
    return definitions


def _tests(source: str) -> List[Tuple[int, str, str]]:
    """Each test of a test file: its line, name and expression. Raises
    ValueError naming the first line that is not a test."""
    tests = []
    for number, line in enumerate(source.splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        match = TEST_LINE.match(stripped)
        if match is None:
            raise ValueError(f"line {number} is not a test line")
        tests.append((number, match["name"], match["expr"]))
    return tests


def _check(functions: _Functions, expression: str) -> Tuple[str, str]:
    try:
        tree = ast.parse(expression, mode="eval").body
        if not (isinstance(tree, ast.Compare) and len(tree.ops) == 1
                and isinstance(tree.ops[0], ast.Eq)):
            raise ValueError("a test is <left> == <right>")
        left = functions.evaluate(tree.left, {})
        right = functions.evaluate(tree.comparators[0], {})
    except (SyntaxError, ValueError, ZeroDivisionError, RecursionError) as exc:
        return ERROR, str(exc)
    return (PASSED, "") if left == right else (FAILED, f"{left} != {right}")


def _run(root: Path, suite: la.Suite, replaced=None):
    """The outcome, reason and each test's outcome of a suite, with the
    functions called."""
    functions = _Functions(_definitions(root, replaced))
    outcomes: List[TestOutcome] = []
    suite_root = Path(suite.root)
    for path in suite.files()[:len(suite.test_files)]:
        relative = Path(path).resolve().relative_to(suite_root.resolve()).as_posix()
        try:
            tests = _tests(Path(path).read_text(encoding="utf-8"))
        except ValueError as exc:
            return la.Outcome.COLLECTION_FAILED, f"{relative}: {exc}", (), functions
        for _line, name, expression in tests:
            outcome, message = _check(functions, expression)
            test = la.TestId(relative, name, text=f"{Path(relative).stem}::{name}")
            outcomes.append(TestOutcome(test, outcome, 0.0, message))
    if not outcomes:
        return la.Outcome.NO_TESTS, "no test line", (), functions
    failing = [t for t in outcomes if t.outcome in (FAILED, ERROR)]
    if failing:
        return (la.Outcome.TEST_FAILURES,
                f"{len(failing)} of {len(outcomes)} tests did not pass",
                tuple(outcomes), functions)
    return la.Outcome.COMPLETED, "", tuple(outcomes), functions


class CalcSession(MutationSession):
    """Mutants of one target, each run against the suite."""

    def __init__(self, adapter: "CalcAdapter") -> None:
        super().__init__(workers_requested=1, workers_started=1)
        self.adapter = adapter

    def measure(self, request) -> MutationRun:
        root = Path(self.adapter.context.root)
        target = Path(request.target).resolve()
        text = target.read_text(encoding="utf-8")
        relative = target.relative_to(root.resolve()).as_posix()
        mutants: List[Mutant] = []
        lines = text.splitlines()
        for number, line in enumerate(lines, start=1):
            match = FUNCTION_LINE.match(line.strip())
            if match is None:
                continue
            body_start = line.index("=") + 1
            occurrence = 0
            for position in range(body_start, len(line)):
                for old, new in SWAPS:
                    if line[position] != old:
                        continue
                    occurrence += 1
                    mutated = line[:position] + new + line[position + 1:]
                    changed = "\n".join(lines[:number - 1] + [mutated] + lines[number:]) + "\n"
                    outcome, _reason, tests, _ = _run(root, request.suite, (target, changed))
                    killers = tuple(str(t.test) for t in tests
                                    if t.outcome in (FAILED, ERROR))
                    killed = outcome is not la.Outcome.COMPLETED
                    diff = f"{relative}:{number}: {line.strip()} -> {mutated.strip()}"
                    classification = (
                        Classification("killed", MutationClass.DETECTED, BY_FAILING_TEST)
                        if killed else
                        Classification("survived", MutationClass.UNDETECTED, SURVIVED))
                    operator_name = f"swap {old} {new}"
                    row = {"id": len(mutants), "line": number, "operator": operator_name,
                           "status": classification.status,
                           "killer": killers[0] if killers else None,
                           "killers": list(killers), "diff": diff}
                    mutants.append(Mutant(
                        key=mutant_key(relative, operator_name, occurrence, number,
                                       position, diff),
                        engine_id=len(mutants), test_outcome="killed" if killed else "survived",
                        worker_outcome=None, classification=classification,
                        killers=killers, diff=diff, row=row,
                        killing_tests=tuple(t.test for t in tests
                                            if t.outcome in (FAILED, ERROR))))
        killed = sum(m.status == "killed" for m in mutants)
        summary = {"total": len(mutants), "detected": killed,
                   "survived": len(mutants) - killed, "timeout": 0, "non_viable": 0,
                   "untested": 0, "unresolved": 0}
        return MutationRun(target_file=str(target), mutants=tuple(mutants),
                           summary=summary, requested_catalogue="full",
                           requested_operators=(), outcome=la.Outcome.COMPLETED,
                           reason="" if mutants else "the catalogue has no mutant for this target")

    def release(self) -> None:
        return None


class CalcAdapter(la.LangAdapter):
    """Calc at Testuary's level 1."""

    language = LANGUAGE
    file_extensions = (".calc",)
    runtime = sys.executable
    test_file_name = "test_{stem}{ext}"
    native_mutation_engines = (ENGINE,)

    def capabilities(self) -> la.CapabilityReport:
        offered = {"suite_runs", "coverage", "mutation"}
        return la.CapabilityReport(
            language=self.language, interpreter="the adapter itself",
            outcome=la.Outcome.COMPLETED, reason="", runtime="calc 1",
            capabilities=tuple(
                la.Capability(name, la.Availability.AVAILABLE) if name in offered else
                la.Capability(name, la.Availability.UNSUPPORTED, "Calc is at level 1")
                for name in CAPABILITIES),
            tools=(), branch_coverage="none", mutation_engines=(ENGINE,))

    def language_facts(self) -> la.LanguageFacts:
        return la.LanguageFacts(
            language=self.language, test_framework="calc tests", code_fence="text",
            runtime_version="1", test_file_patterns=("test_*.calc",),
            test_function_pattern="test <name>: <left> == <right>",
            test_class_pattern=None, timeout_marker=None)

    # ---- the test units and the syntax ------------------------------------

    def check_structure(self, source: str) -> StructureCheck:
        for number, line in enumerate(source.splitlines(), start=1):
            stripped = line.strip()
            if (stripped and not stripped.startswith("#")
                    and not TEST_LINE.match(stripped) and not FUNCTION_LINE.match(stripped)):
                return StructureCheck(False, f"line {number} is neither a test nor a function",
                                      number, 1, PARSER)
        return StructureCheck(True, "", 0, 0, PARSER)

    def structural_parser_identity(self) -> str:
        return PARSER

    def validate_source(self, source: str) -> SourceValidation:
        verdict = self.check_structure(source)
        return SourceValidation(
            "valid" if verdict.ok else "syntax_error", verdict.message, verdict.line,
            verdict.column, PARSER, hashlib.sha256(source.encode("utf-8")).hexdigest())

    def test_metrics(self, source: str) -> TestMetrics:
        try:
            tests = _tests(source)
        except ValueError as exc:
            return TestMetrics(False, PARSER, detail=str(exc))
        lines = sum(1 for line in source.splitlines()
                    if line.strip() and not line.strip().startswith("#"))
        return TestMetrics(True, PARSER, code_lines=lines, tests=len(tests),
                           assertions=len(tests), fixtures=0, parametrized_cases=0)

    def prune_tests(self, source: str, remove_ids) -> SourceResult:
        names = {str(test).split("::")[-1] for test in remove_ids}
        kept, removed = [], []
        for line in source.splitlines(keepends=True):
            match = TEST_LINE.match(line.strip())
            if match and match["name"] in names:
                removed.append(match["name"])
                continue
            kept.append(line)
        return SourceResult("applied" if removed else "unchanged", "".join(kept),
                            PARSER, removed=tuple(removed))

    # ---- runs and mutation ------------------------------------------------

    def run_suite(self, request) -> SuiteRun:
        request.validate()
        started = time.monotonic()
        identity = la.ObservationIdentity(
            target=None, suite=request.suite.identity(),
            toolchain=la.ToolchainIdentity.of(Version("calc", (1,), "1", "context")),
            instrument=None, execution=la.ExecutionIdentity("local"))
        root = Path(self.context.root)
        outcome, reason, tests, functions = _run(root, request.suite)
        coverage = None
        if request.coverage_target is not None and outcome in (
                la.Outcome.COMPLETED, la.Outcome.TEST_FAILURES):
            target = Path(request.coverage_target).resolve()
            statements = tuple(number for number, line in enumerate(
                target.read_text(encoding="utf-8").splitlines(), start=1)
                if FUNCTION_LINE.match(line.strip()))
            covered = tuple(sorted(functions.called.get(str(target), set())))
            coverage = CoverageObservation(
                target=la.FileIdentity.of(target, root.resolve()), statements=statements, covered=covered, branches=(), taken=())
        counts = RunCounts(
            collected=len(tests), passed=sum(t.outcome == PASSED for t in tests),
            failed=sum(t.outcome == FAILED for t in tests),
            errors=sum(t.outcome == ERROR for t in tests), skipped=0)
        return SuiteRun(outcome=outcome, reason=reason, counts=counts, tests=tests,
                        unmatched=(), wall_time_s=time.monotonic() - started,
                        output=reason, coverage=coverage, identity=identity)

    def mutation_operators(self, engine: str = ENGINE) -> OperatorCatalogue:
        return OperatorCatalogue(engine, tuple(f"swap {a} {b}" for a, b in SWAPS))

    def open_mutation_session(self, engine: str = ENGINE, **options) -> CalcSession:
        return CalcSession(self)


def register() -> None:
    """Register Calc, once."""
    if LANGUAGE not in la.names():
        la.register(LANGUAGE, [".calc"], CalcAdapter)


def unregister() -> None:
    if LANGUAGE in la.names():
        la.unregister(LANGUAGE)
