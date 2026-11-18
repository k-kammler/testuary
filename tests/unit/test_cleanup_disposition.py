"""The recorded disposition of a starter suite the cleanup could not accept.

A suite run that gives no per-test outcome is judged when the suite does not
compile or collect, or holds no test, and stays unjudged when the run lacked
its setup, was unsupported or could not start. Each ending names its reason.
"""
from __future__ import annotations

import hashlib
import itertools
from pathlib import Path
from types import SimpleNamespace

import pytest

from testuary import languages
from testuary.languages.generic import PruneOutcome, SourceValidation
from testuary import normal_run
from testuary import starter_salvage

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")

TARGET = "def add(a, b):\n    return a + b\n"
SUITE = (
    "from calc import add\n\n"
    "def test_first():\n    assert add(1, 1) == 3\n\n"
    "def test_second():\n    assert add(1, 2) == 4\n"
)


def _clean(tmp_path, monkeypatch, outcomes, source=SUITE):
    """Clean ``source`` while the scripted runs report ``outcomes``: each is
    (ids, outcome)."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    backend = languages.get_backend("python")
    script = iter(outcomes)

    def failing_test_ids(self, paths, source_dir, *, report=None, **kwargs):
        ids, outcome = next(script)
        if report is not None:
            report["outcome"] = outcome
        return ids

    monkeypatch.setattr(type(backend), "failing_test_ids", failing_test_ids)
    cleanup = {}
    classification, _green = starter_salvage._clean_suite(
        backend, source, str(root / "calc.py"), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None, cleanup_out=cleanup,
    )
    return classification, cleanup


@pytest.mark.parametrize("outcome,basis,reason", [
    ("collection_failed", "judged_rejection", "the suite does not compile or collect"),
    ("no_tests", "judged_rejection", "the suite holds no test"),
    ("missing_setup", "unavailable", "the suite run lacked its setup"),
    ("unsupported", "unavailable", "the suite run is not supported here"),
    ("os_error", "unavailable", "the suite run could not start"),
    ("infrastructure_failure", "unavailable", "the suite run failed in its infrastructure"),
    (None, "unavailable", "the suite run gave no test outcome"),
])
def test_a_suite_run_without_an_outcome_records_why(tmp_path, monkeypatch, outcome, basis, reason):
    classification, cleanup = _clean(tmp_path, monkeypatch, [(None, outcome)] * 3)
    assert classification == "discarded_collection"
    assert (cleanup["disposition_basis"], cleanup["disposition_reason"]) == (basis, reason)


COMPILE_REASON = "the test file does not compile: t.c:3: error: unknown type name 'cbor_item'"


@pytest.mark.parametrize("described,logged,recorded", [
    (None, COMPILE_REASON, COMPILE_REASON),
    ("ModuleNotFoundError: No module named 'calc2'",
     "ModuleNotFoundError: No module named 'calc2'", None),
], ids=["without collection step", "with collection step"])
def test_a_suite_that_does_not_compile_records_the_runs_reason_without_a_collection_step(
        tmp_path, monkeypatch, described, logged, recorded):
    """C has no collection step: the run's reason and compiler lines go to
    the log line and the discarded attempt's record. Python and Java log
    their collection's text and record no run reason."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    backend = languages.get_backend("python")

    def failing_test_ids(self, paths, source_dir, *, report=None, **kwargs):
        report.update(outcome="collection_failed", reason=COMPILE_REASON)
        return None

    monkeypatch.setattr(type(backend), "failing_test_ids", failing_test_ids)
    monkeypatch.setattr(type(backend), "describe_collection_failure",
                        lambda self, *a, **k: described)
    messages, cleanup = [], {}
    classification, _green = starter_salvage._clean_suite(
        backend, SUITE, str(root / "calc.py"), str(root), tmp_path / "scratch",
        timeout=120, log=messages.append, cleanup_out=cleanup,
    )
    assert classification == "discarded_collection"
    assert f"[starter] collection failed: {logged}" in messages
    assert cleanup.get("collection_failure_reason") == recorded
    assert cleanup["disposition_reason"] == "the suite does not compile or collect"


def test_the_recorded_compiler_reason_is_the_first_failing_runs(tmp_path, monkeypatch):
    """A recovery probe that fails to compile again leaves the record with
    the first failing run's text, the text of the log line."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    backend = languages.get_backend("python")
    reasons = iter([COMPILE_REASON, "the test file does not compile: t.c:9: error: later"])

    def failing_test_ids(self, paths, source_dir, *, report=None, **kwargs):
        report.update(outcome="collection_failed", reason=next(reasons))
        return None

    monkeypatch.setattr(type(backend), "failing_test_ids", failing_test_ids)
    monkeypatch.setattr(type(backend), "describe_collection_failure",
                        lambda self, *a, **k: None)
    monkeypatch.setattr(type(backend), "prune_unresolvable_imports",
                        lambda self, source, root_: (source.replace("import os\n", ""), ["os"]))
    monkeypatch.setattr(type(backend), "prune_collection_breakers",
                        lambda self, source, root_, *, timeout: (source, []))
    messages, cleanup = [], {}
    classification, _green = starter_salvage._clean_suite(
        backend, SUITE_WITH_BREAKER, str(root / "calc.py"), str(root), tmp_path / "scratch",
        timeout=120, log=messages.append, cleanup_out=cleanup,
    )
    assert classification == "discarded_collection"
    assert f"[starter] collection failed: {COMPILE_REASON}" in messages
    assert cleanup["collection_failure_reason"] == COMPILE_REASON


@pytest.mark.parametrize("reported,recorded", [
    (COMPILE_REASON, COMPILE_REASON), (None, None),
], ids=["without collection step", "with collection step"])
def test_a_cleaned_suite_that_does_not_compile_records_the_compiler_reason(
        tmp_path, monkeypatch, reported, recorded):
    """The first run fails a test, and the suite without it does not compile.
    Without a collection step the run reports the compiler's text, which the
    discarded attempt's record keeps. A language that collects reports none
    and records nothing new."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    backend = languages.get_backend("python")
    runs = iter([(["test_cleanup_disposition_candidate.test_first"], "test_failures"),
                 (None, "collection_failed")])

    def failing_test_ids(self, paths, source_dir, *, report=None, **kwargs):
        ids, outcome = next(runs)
        report["outcome"] = outcome
        if outcome == "collection_failed" and reported:
            report["reason"] = reported
        return ids

    monkeypatch.setattr(type(backend), "failing_test_ids", failing_test_ids)
    cleanup = {}
    classification, _green = starter_salvage._clean_suite(
        backend, SUITE, str(root / "calc.py"), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None, cleanup_out=cleanup,
    )
    assert classification == "unsalvageable"
    assert cleanup["disposition_reason"] == (
        "the suite does not compile or collect after cleanup round 1")
    assert cleanup.get("collection_failure_reason") == recorded


class _FirstRun(Exception):
    """Stops the salvage at its first suite run."""


UTEST_SUITE = '#include "utest.h"\n\nUTEST(calc, adds) {\n  ASSERT_EQ(2, 1 + 1);\n}\n'
JUNIT_SUITE = (
    "import org.junit.jupiter.api.Test;\n"
    "import static org.junit.jupiter.api.Assertions.assertEquals;\n\n"
    "class CalcTest {\n    @Test\n    void adds() {\n        assertEquals(2, Calc.add(1, 1));\n    }\n}\n"
)


@pytest.mark.parametrize("language,target,source,compiled", [
    ("c", "calc.c", UTEST_SUITE, UTEST_SUITE + "\nUTEST_MAIN()\n"),
    ("python", "calc.py", SUITE, SUITE),
    ("java", "Calc.java", JUNIT_SUITE, JUNIT_SUITE),
])
def test_the_starter_source_is_finalized_before_its_first_run(
        tmp_path, monkeypatch, language, target, source, compiled):
    """The salvage finalizes the starter source through the backend before
    its first run: C's finalization appends UTEST_MAIN() to a file with
    utest.h tests and no main, and Python's and Java's sources stay as
    they are."""
    if language == "c":
        # C's parser is LangAdapter's optional "c" extra.
        pytest.importorskip("tree_sitter_c")
    root = tmp_path / "proj"
    root.mkdir()
    (root / target).write_text({
        "c": "int add(int a, int b) { return a + b; }\n",
        "python": TARGET,
        "java": "class Calc {\n    static int add(int a, int b) { return a + b; }\n}\n",
    }[language], encoding="utf-8")
    backend = languages.get_backend(language)
    staged = []

    def failing_test_ids(self, paths, source_dir, **kwargs):
        staged.append(open(paths[0], encoding="utf-8").read())
        raise _FirstRun

    def validate(self, text, source_dir=None):
        return SourceValidation(True, "valid", "", 0, 0, "fixture validator",
                                hashlib.sha256(text.encode("utf-8")).hexdigest())

    monkeypatch.setattr(type(backend), "failing_test_ids", failing_test_ids)
    # The syntax repair and the target validator need the project's
    # toolchain: the source parses as it is.
    monkeypatch.setattr(type(backend), "salvage_syntax",
                        lambda self, text, source_dir, *, report=None: text)
    monkeypatch.setattr(type(backend), "validate_test_source", validate)
    with pytest.raises(_FirstRun):
        starter_salvage._clean_suite(
            backend, source, str(root / target), str(root), tmp_path / "scratch",
            timeout=120, log=lambda _message: None, cleanup_out={})
    assert staged == [compiled]


def _removal(failures, rounds=3):
    """The removal fallback's loop over SUITE from the place's first failure,
    ``failures[0]``, while the measurements after it fail as the rest lists
    them, then pass. Returns the source, the last failure and the entries."""
    from testuary.consolidation import _remove_named_failures

    script = iter(failures[1:])
    return _remove_named_failures(
        languages.get_backend("python"), SUITE, place="phase_end.merged",
        failed=failures[0], measure=lambda source: next(script, None), rounds=rounds,
        log=lambda _line: None)


def _test_id(name):
    from langadapter.suites import TestId

    return TestId("test_calc.py", name, text=f"test_calc.py::{name}")


def test_the_fallback_removes_the_named_test_until_the_measurement_passes():
    from testuary.consolidation import _Failed

    source, failed, entries = _removal([
        _Failed("coverage", tests=(_test_id("test_first"),), outcomes=("failed",))])
    assert failed is None
    assert "def test_first" not in source and "def test_second" in source
    assert entries == [{"place": "phase_end.merged", "measurement": "coverage", "round": 1,
                        "removed": [{"test": "test_calc.test_first", "outcome": "failed"}]}]


def test_the_fallback_records_a_mutation_name_without_an_outcome():
    from testuary.consolidation import _Failed

    _source, _failed, entries = _removal([
        _Failed("mutation", tests=(_test_id("test_first"),))])
    assert entries[0]["removed"] == [{"test": "test_calc.test_first"}]


@pytest.mark.parametrize("failure,cause", [
    ({"timed_out": True, "tests": ("test_first",)}, "the measurement timed out"),
    ({"tests": ()}, "the measurement named no test"),
    ({"cause": "the normal run failed after a removal", "tests": ("test_first",)},
     "the normal run failed after a removal"),
])
def test_the_fallback_removes_nothing_without_a_usable_name(failure, cause):
    from testuary.consolidation import _Failed

    named = tuple(_test_id(name) for name in failure.pop("tests"))
    source, failed, entries = _removal([_Failed("coverage", tests=named, **failure)])
    assert source == SUITE and failed is not None
    assert entries == [{"place": "phase_end.merged", "measurement": "coverage", "round": 1,
                        "unnamed": cause}]


def test_the_fallback_records_names_the_suite_does_not_hold():
    from testuary.consolidation import _Failed

    source, failed, entries = _removal([
        _Failed("coverage", tests=(_test_id("test_first"), _test_id("test_nowhere")),
                outcomes=("failed", "error")),
        _Failed("coverage", tests=(_test_id("test_elsewhere"),), outcomes=("failed",))])
    assert "def test_first" not in source and failed is not None
    assert entries == [
        {"place": "phase_end.merged", "measurement": "coverage", "round": 1,
         "removed": [{"test": "test_calc.test_first", "outcome": "failed"}],
         "unmatched": ["test_calc.test_nowhere"]},
        {"place": "phase_end.merged", "measurement": "coverage", "round": 2,
         "unmatched": ["test_calc.test_elsewhere"],
         "unnamed": "the suite holds no named test"}]


def test_the_fallback_stops_when_its_rounds_are_spent():
    from testuary.consolidation import _Failed

    source, failed, entries = _removal([
        _Failed("coverage", tests=(_test_id("test_first"),), outcomes=("failed",)),
        _Failed("coverage", tests=(_test_id("test_second"),), outcomes=("failed",))],
        rounds=1)
    assert "def test_second" in source and failed is not None
    assert entries[-1] == {"place": "phase_end.merged", "measurement": "coverage",
                           "round": 2, "unnamed": "the rounds are spent"}


def test_a_suite_emptied_by_its_cleanup_is_judged(tmp_path, monkeypatch):
    """Every test failed and the cleanup round removed them all: the run of
    what is left holds no test."""
    classification, cleanup = _clean(tmp_path, monkeypatch, [
        (["test_cleanup_disposition_candidate.test_first",
          "test_cleanup_disposition_candidate.test_second"], "test_failures"),
        (None, "no_tests"),
    ])
    assert classification == "unsalvageable"
    assert cleanup["disposition_basis"] == "judged_rejection"
    assert cleanup["disposition_reason"].startswith("the suite holds no test after cleanup round")


def _validator_silent_without(monkeypatch, name):
    """The validator gives no answer for a source without the test ``name``,
    as under machine load, and judges the others."""
    backend = languages.get_backend("python")
    judge = type(backend).validate_test_source

    def validate(self, source, source_dir=None):
        if f"def {name}(" not in source:
            return SourceValidation(False, "timeout", "the validator timed out", 0, 0,
                                    "validator identity unavailable",
                                    hashlib.sha256(source.encode("utf-8")).hexdigest())
        return judge(self, source, source_dir)

    monkeypatch.setattr(type(backend), "validate_test_source", validate)


def test_a_cleanup_round_the_validator_does_not_answer_for_stays_unresolved(tmp_path, monkeypatch):
    """An unanswered check of the suite a cleanup round left is no rejection:
    the suite stays unresolved, so its slot asks the model for nothing new."""
    _validator_silent_without(monkeypatch, "test_first")
    classification, cleanup = _clean(tmp_path, monkeypatch, [
        (["test_cleanup_disposition_candidate.test_first"], "test_failures"),
    ])
    assert classification == "validation_unresolved"
    assert cleanup["unresolved_stage"] == "target_validation"


def test_a_hung_test_whose_removal_the_validator_does_not_answer_for_stays_unresolved(
        tmp_path, monkeypatch):
    """A run the timeout ended while test_first ran. The validator gives no
    answer for the suite without it, which leaves the suite unresolved, not
    timed out."""
    _validator_silent_without(monkeypatch, "test_first")
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    backend = languages.get_backend("python")

    def failing_test_ids(self, paths, source_dir, *, report=None, **kwargs):
        if report is not None:
            report.update(outcome="timed_out",
                          active_at_end="test_cleanup_disposition_candidate.test_first")
        return None

    monkeypatch.setattr(type(backend), "failing_test_ids", failing_test_ids)
    cleanup = {}
    classification, _green = starter_salvage._clean_suite(
        backend, SUITE, str(root / "calc.py"), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None, cleanup_out=cleanup,
    )
    assert classification == "validation_unresolved"
    assert "timeout_removed" not in cleanup


def test_a_suite_that_does_not_parse_is_judged(tmp_path, monkeypatch):
    classification, cleanup = _clean(tmp_path, monkeypatch, [], source="}\n" * 60)
    assert classification == "discarded_syntax"
    assert cleanup["disposition_basis"] == "judged_rejection"
    assert cleanup["disposition_reason"] in (
        "the suite does not parse and could not be repaired", "the suite does not parse",
        "target parser rejected the source")


def test_the_python_backend_reports_a_collection_failure(tmp_path):
    """A real pytest run of a suite that cannot be collected."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    suite = root / "test_broken.py"
    suite.write_text("import calc\ncalc.missing_name\n\ndef test_x():\n    pass\n",
                     encoding="utf-8")
    report = {}
    ids = languages.get_backend("python").failing_test_ids(
        [str(suite)], str(root), timeout=120, report=report)
    assert ids is None and report["outcome"] == "collection_failed"



def _recover(tmp_path, monkeypatch, *, breakers=None, imports=None):
    """Clean SUITE_WITH_BREAKER while the collection fails once and the
    scripted prunes remove what they name."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    backend = languages.get_backend("python")
    runs = iter([(None, "collection_failed"), ([], "completed")])

    def failing_test_ids(self, paths, source_dir, *, report=None, **kwargs):
        ids, outcome = next(runs)
        if report is not None:
            report["outcome"] = outcome
        return ids

    monkeypatch.setattr(type(backend), "failing_test_ids", failing_test_ids)
    monkeypatch.setattr(type(backend), "describe_collection_failure",
                        lambda self, *a, **k: "")
    monkeypatch.setattr(type(backend), "prune_unresolvable_imports",
                        lambda self, source, root_: (
                            (source.replace("import os\n", ""), imports) if imports else (source, [])))
    monkeypatch.setattr(type(backend), "prune_collection_breakers",
                        lambda self, source, root_, *, timeout: (
                            (source.split("def test_broken")[0], breakers) if breakers else (source, [])))
    cleanup = {}
    classification, green = starter_salvage._clean_suite(
        backend, SUITE_WITH_BREAKER, str(root / "calc.py"), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None, cleanup_out=cleanup,
    )
    return classification, green, cleanup


SUITE_WITH_BREAKER = (
    "import os\n"
    "from calc import add\n\n"
    "def test_add():\n    assert add(1, 1) == 2\n\n"
    "def test_broken():\n    assert add(2, 2) == 4\n"
)


def test_a_test_removed_by_the_collection_recovery_strips_the_suite(tmp_path, monkeypatch):
    """The backend's collection-breaker prune removed a test: the units it
    names are recorded and the accepted suite is classified pruned."""
    classification, green, cleanup = _recover(
        tmp_path, monkeypatch, breakers=["CalcTest.testBroken (lines 6-7)"])
    assert classification == "pruned"
    assert green is not None and "test_broken" not in green
    assert cleanup["collection_recovery_removed"] == {
        "collection_breakers": ["CalcTest.testBroken (lines 6-7)"]}


def test_an_import_removed_by_the_collection_recovery_keeps_all_green(tmp_path, monkeypatch):
    """Removing an unresolvable import removes no test: the suite stays
    all_passing, and the import is recorded."""
    classification, _green, cleanup = _recover(tmp_path, monkeypatch, imports=["os"])
    assert classification == "all_passing"
    assert cleanup["collection_recovery_removed"] == {"unresolvable_imports": ["os"]}


def test_a_recovered_suite_whose_test_count_is_unavailable_is_pruned(tmp_path, monkeypatch):
    """The collection recovery changed the suite, and the static count of the
    result is unavailable. No count shows that no test was removed, so the
    suite is classified pruned, as the phase ends treat the same case."""
    backend = languages.get_backend("python")
    count = type(backend).test_metrics

    def test_metrics(self, source):
        metrics = dict(count(self, source))
        if "import os" not in source:
            metrics["test_count"] = None
        return metrics

    monkeypatch.setattr(type(backend), "test_metrics", test_metrics)
    classification, _green, _cleanup = _recover(tmp_path, monkeypatch, imports=["os"])
    assert classification == "pruned"


def test_a_validity_gate_without_an_answer_leaves_the_suite_unjudged(tmp_path, monkeypatch):
    """A suite the validity gate could not judge is not accepted. It stays
    unjudged as an infrastructure failure, and its record names the gate."""
    from langadapter.python.adapter import PythonAdapter

    def broken(self, target):
        raise RuntimeError("the identity probe failed")

    monkeypatch.setattr(PythonAdapter, "target_identity", broken)
    passing = "from calc import add\n\ndef test_add():\n    assert add(1, 1) == 2\n"
    classification, cleanup = _clean(tmp_path, monkeypatch, [([], "completed")] * 3,
                                     source=passing)
    assert classification == "validation_unresolved"
    assert cleanup["unresolved_stage"] == "generated_test_validity"
    assert "target_identity did not complete" in cleanup["validation_unresolved"]
    assert cleanup["disposition_basis"] == "unavailable"


# Each ending records its own reason, never the default "cleanup judgement
# unavailable".

GREEN = "from calc import add\n\ndef test_first():\n    assert add(1, 1) == 2\n"


def _finding(test):
    from testuary.languages import SuiteValidityFinding

    return SuiteValidityFinding(
        rule_id="target-source-read", policy_version="fixture", operation="open",
        diagnostic="reads the target's source", start_line=1, start_col=0,
        end_line=1, end_col=4, enclosing_test=test)


def _clean_with(tmp_path, monkeypatch, *, outcomes=(), findings=(), prune=None,
                timeout=120, source=SUITE, validity_out=None):
    """Clean ``source`` with scripted runs (ids, outcome, seconds slept),
    scripted validity analyses ``findings`` (one list per call) and an
    optional ``prune`` in place of the backend's test removal."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    backend = languages.get_backend("python")
    script = iter(outcomes)
    analyses = iter(findings)

    def failing_test_ids(self, paths, source_dir, *, report=None, **kwargs):
        ids, outcome, slept = next(script)
        if slept:
            import time
            time.sleep(slept)
        if report is not None:
            report["outcome"] = outcome
        return ids

    monkeypatch.setattr(type(backend), "failing_test_ids", failing_test_ids)
    monkeypatch.setattr(type(backend), "analyze_generated_test_validity",
                        lambda self, source, identity: next(analyses, []))
    if prune is not None:
        monkeypatch.setattr(type(backend), "prune_tests_report",
                            lambda self, source, names: PruneOutcome(prune(source), ""))
    cleanup = {}
    classification, _green = starter_salvage._clean_suite(
        backend, source, str(root / "calc.py"), str(root), tmp_path / "scratch",
        timeout=timeout, log=lambda _message: None, cleanup_out=cleanup,
        validity_out=validity_out,
    )
    return classification, cleanup


def test_an_observation_outside_every_test_is_a_judged_rejection(tmp_path, monkeypatch):
    validity_out = {}
    classification, cleanup = _clean_with(
        tmp_path, monkeypatch, findings=[[_finding(None)]], validity_out=validity_out)
    assert classification == "invalid_generated_test"
    assert cleanup["disposition_basis"] == "judged_rejection"
    assert cleanup["disposition_reason"] == starter_salvage._VALIDITY_OUTSIDE_TESTS
    assert validity_out["stage"] == "starter.generated_suite_rejected"


def test_invalid_tests_that_cannot_be_removed_name_themselves(tmp_path, monkeypatch):
    validity_out = {}
    classification, cleanup = _clean_with(
        tmp_path, monkeypatch, findings=[[_finding("test_first")]],
        prune=lambda source: source, validity_out=validity_out)
    assert classification == "invalid_generated_test"
    assert cleanup["disposition_basis"] == "unavailable"
    assert cleanup["disposition_reason"] == (
        "the tests that break the generated-test validity policy could not be "
        "removed: test_first")
    assert len(validity_out["findings"]) == 1


def test_a_suite_that_does_not_parse_after_the_removal_says_so(tmp_path, monkeypatch):
    classification, cleanup = _clean_with(
        tmp_path, monkeypatch, findings=[[_finding("test_first")]],
        prune=lambda source: "def broken(:\n")
    assert classification == "invalid_generated_test"
    assert cleanup["disposition_basis"] == "unavailable"
    assert cleanup["disposition_reason"] == (
        "the suite does not parse after removing the tests that break the "
        "generated-test validity policy: test_first")


def test_findings_left_after_the_removal_are_recorded(tmp_path, monkeypatch):
    classification, cleanup = _clean_with(
        tmp_path, monkeypatch,
        findings=[[_finding("test_first")], [_finding("test_second")]])
    assert classification == "invalid_generated_test"
    assert cleanup["disposition_basis"] == "unavailable"
    assert cleanup["disposition_reason"] == (
        "the generated-test validity policy still rejected test_second after "
        "removing test_first")


def test_a_shared_observation_left_after_the_removal_is_judged(tmp_path, monkeypatch):
    classification, cleanup = _clean_with(
        tmp_path, monkeypatch, findings=[[_finding("test_first")], [_finding(None)]])
    assert classification == "invalid_generated_test"
    assert cleanup["disposition_basis"] == "judged_rejection"
    assert cleanup["disposition_reason"] == (
        f"{starter_salvage._VALIDITY_OUTSIDE_TESTS} after removing test_first")


def test_a_first_run_that_times_out_records_the_timeout(tmp_path, monkeypatch):
    classification, cleanup = _clean_with(
        tmp_path, monkeypatch, outcomes=[(None, "timed_out", 1.0)], timeout=1)
    assert classification == "timeout"
    assert (cleanup["disposition_basis"], cleanup["disposition_reason"]) == (
        "unavailable", "the suite run timed out")


def test_a_run_its_runner_reports_timed_out_is_a_timeout_however_long_it_took(
        tmp_path, monkeypatch):
    classification, cleanup = _clean_with(
        tmp_path, monkeypatch, outcomes=[(None, "timed_out", 0)] * 3)
    assert classification == "timeout"
    assert cleanup["disposition_reason"] == "the suite run timed out"


def test_a_run_that_used_its_time_but_failed_otherwise_is_no_timeout(tmp_path, monkeypatch):
    # Each run seems to take 1000 seconds, past its 120: the repair once read
    # the time it took instead of the runner's outcome.
    ticks = itertools.count(0, 1000)
    monkeypatch.setattr(starter_salvage, "time", SimpleNamespace(monotonic=lambda: next(ticks)),
                        raising=False)
    classification, cleanup = _clean_with(
        tmp_path, monkeypatch, outcomes=[(None, "infrastructure_failure", 0)] * 3)
    assert classification == "discarded_collection"
    assert cleanup["disposition_reason"] == "the suite run failed in its infrastructure"


def test_a_run_that_times_out_after_a_cleanup_round_records_the_round(tmp_path, monkeypatch):
    classification, cleanup = _clean_with(tmp_path, monkeypatch, timeout=1, outcomes=[
        (["test_cleanup_disposition_candidate.test_first"], "test_failures", 0),
        (None, "timed_out", 1.0),
    ])
    assert classification == "timeout"
    assert (cleanup["disposition_basis"], cleanup["disposition_reason"]) == (
        "unavailable", "the suite run timed out after cleanup round 1")


def test_unsound_warnings_without_a_test_name_record_why(tmp_path, monkeypatch):
    real_run = normal_run.run
    monkeypatch.setattr(normal_run, "run", lambda *args, **kwargs: real_run(*args, **kwargs)._replace(
        unsound=((None, "RuntimeWarning: coroutine 'f' was never awaited"),)))
    classification, cleanup = _clean_with(
        tmp_path, monkeypatch, outcomes=[([], "completed", 0)], source=GREEN)
    assert classification == "unsound"
    assert cleanup["disposition_reason"] == "unsound-state warnings named no test"


GREEN_TWO = (
    "from calc import add\n\n"
    "def test_first():\n    assert add(1, 1) == 2\n\n"
    "def test_second():\n    assert add(1, 2) == 3\n"
)


def test_a_suite_the_validity_policy_removed_tests_from_is_pruned(tmp_path, monkeypatch):
    """A suite that lost a test to the validity policy and then passed is
    pruned, not all_passing."""
    validity_out = {}
    classification, cleanup = _clean_with(
        tmp_path, monkeypatch, source=GREEN_TWO, findings=[[_finding("test_second")]],
        outcomes=[([], "completed", 0)], validity_out=validity_out)
    assert classification == "pruned"
    assert cleanup["disposition_basis"] == "accepted_cleaned"
    assert validity_out["removed_tests"] == ["test_second"]


PARAMETRIZED = (
    "import pytest\n"
    "from calc import add\n\n"
    "@pytest.mark.parametrize(\"a,b,total\", [(1, 1, 2), (2, 2, 4), (3, 3, 6)])\n"
    "def test_add(a, b, total):\n    assert add(a, b) == total\n\n"
    "def test_other():\n    assert add(0, 0) == 0\n"
)


def test_unsound_case_names_keep_a_parametrized_case():
    assert starter_salvage._unsound_case_names([
        "test_candidate.py::test_add[2-2-4]",
        "test_candidate.py::TestCalc::test_sub",
        "test_candidate.py::test_add[2-2-4]",
        "no node",
    ]) == ["TestCalc.test_sub", "test_add[2-2-4]"]


def test_an_unsound_parametrized_case_is_removed_alone(tmp_path, monkeypatch):
    """Only the case of a literal parameter list that emitted an
    unsound-state warning is removed, and the test keeps its other cases."""
    backend = languages.get_backend("python")
    real_run = normal_run.run
    entries = iter([(("test_candidate.py::test_add[2-2-4]", ""),)])
    monkeypatch.setattr(normal_run, "run", lambda *args, **kwargs: real_run(*args, **kwargs)._replace(
        unsound=next(entries, ())))
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    cleanup = {}
    classification, green = starter_salvage._clean_suite(
        backend, PARAMETRIZED, str(root / "calc.py"), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None, cleanup_out=cleanup,
    )
    assert classification == "pruned", cleanup
    assert "def test_add(" in green and "def test_other(" in green
    assert "(2, 2, 4)" not in green
    assert "(1, 1, 2)" in green and "(3, 3, 6)" in green


FAILING_CASE = (
    "import pytest\n"
    "from calc import add\n\n"
    "@pytest.mark.parametrize(\"a,b,total\", [(1, 1, 2), (2, 2, 5), (3, 3, 6)])\n"
    "def test_add(a, b, total):\n    assert add(a, b) == total\n\n"
    "def test_other():\n    assert add(0, 0) == 0\n"
)

FAILING_CASE_UNREADABLE = FAILING_CASE.replace(
    "[(1, 1, 2), (2, 2, 5), (3, 3, 6)]", "list(zip([1, 2, 3], [1, 2, 3], [2, 5, 6]))")


def _clean_real(tmp_path, source):
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    cleanup = {}
    classification, green = starter_salvage._clean_suite(
        languages.get_backend("python"), source, str(root / "calc.py"), str(root),
        tmp_path / "scratch", timeout=120, log=lambda _message: None, cleanup_out=cleanup,
    )
    return classification, green, cleanup


def test_a_failing_case_of_a_parametrized_test_is_removed_alone(tmp_path):
    """The failing-test listing names the failing case, so the cleanup
    removes that case and keeps the test's passing cases."""
    classification, green, cleanup = _clean_real(tmp_path, FAILING_CASE)
    assert classification == "pruned"
    assert "def test_add(" in green and "def test_other(" in green
    assert "(2, 2, 5)" not in green
    assert "(1, 1, 2)" in green and "(3, 3, 6)" in green
    assert "prune_details" not in cleanup


def test_a_case_that_cannot_go_alone_takes_its_test_and_says_why(tmp_path):
    """A parameter list that is not literal: the test goes whole, and the
    cleanup records the language's reason."""
    classification, green, cleanup = _clean_real(tmp_path, FAILING_CASE_UNREADABLE)
    assert classification == "pruned"
    assert "def test_add(" not in green and "def test_other(" in green
    [detail] = cleanup["prune_details"]
    assert "test_add is removed whole" in detail and "not literal" in detail


HANGING = (
    "import time\n"
    "from calc import add\n\n"
    "def test_ok():\n    assert add(1, 1) == 2\n\n"
    "def test_hangs():\n    time.sleep(600)\n"
)


def test_a_test_the_timeout_caught_running_is_removed_and_the_rest_run(
        tmp_path, monkeypatch):
    """A test the timeout journal names as running at the timeout is removed
    and recorded, and the rest run again rather than the whole suite being
    discarded. The listing of the suite with the hanging test is the
    journal's report of a run the timeout ended, and the runs after it are
    real."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    backend = languages.get_backend("python")
    real = backend.failing_test_ids

    def listing(test_paths, source_dir, *, timeout, report=None, **options):
        suite = Path(test_paths[0])
        if "def test_hangs" not in suite.read_text(encoding="utf-8"):
            return real(test_paths, source_dir, timeout=timeout, report=report, **options)
        report.update(outcome="timed_out", active_at_end=f"{suite.stem}.test_hangs")
        return None

    monkeypatch.setattr(backend, "failing_test_ids", listing)
    cleanup = {}
    classification, green = starter_salvage._clean_suite(
        backend, HANGING, str(root / "calc.py"), str(root),
        tmp_path / "scratch", timeout=20, log=lambda _message: None, cleanup_out=cleanup,
    )
    assert classification == "pruned", cleanup
    assert "def test_ok" in green and "def test_hangs" not in green
    assert [test.rsplit(".", 1)[-1] for test in cleanup["timeout_removed"]] == ["test_hangs"]
    assert cleanup["disposition_basis"] == "accepted_cleaned"


# Two tests leave one coroutine unawaited, from the same line in leave(), so
# pytest's warnings summary lists both above one warning.
LEAVING_PAIR = (
    "from calc import add\n\n\n"
    "async def pending():\n    pass\n\n\n"
    "def leave():\n    pending()\n\n\n"
    "def test_first():\n    leave()\n    assert add(1, 1) == 2\n\n\n"
    "def test_second():\n    leave()\n    assert add(1, 2) == 3\n\n\n"
    "def test_third():\n    assert add(2, 2) == 4\n"
)


def test_two_tests_that_leave_one_unsound_warning_are_both_removed(tmp_path):
    """The run names each test of the grouped warning, so both are removed
    and the suite is accepted pruned with its third test."""
    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    cleanup = {}
    classification, green = starter_salvage._clean_suite(
        languages.get_backend("python"), LEAVING_PAIR, str(root / "calc.py"), str(root),
        tmp_path / "scratch", timeout=120, log=lambda _message: None, cleanup_out=cleanup,
    )
    assert classification == "pruned", cleanup
    assert "def test_first" not in green and "def test_second" not in green
    assert "def test_third" in green


def test_the_strict_green_text_names_each_test_of_a_shared_warning(tmp_path):
    """The loops' strict-green text names both tests that left one warning,
    from the run's own record."""
    from testuary.agent import _helpers

    root = tmp_path / "proj"
    root.mkdir()
    (root / "calc.py").write_text(TARGET, encoding="utf-8")
    suite = root / "test_leaving.py"
    suite.write_text(LEAVING_PAIR, encoding="utf-8")
    run = languages.get_backend("python").run_test_files([str(suite)], str(root), timeout=120)
    text = _helpers.unsound_async_rejection(run, lambda key, **kw: kw["sources_block"])
    assert text is not None and "test_first" in text and "test_second" in text


def _result(**fields):
    from testuary.languages.generic import TestRunResult

    return TestRunResult(**{"ok": False, "returncode": 1, "timed_out": False,
                            "elapsed_s": 1.0, "collected": 0, "passed": 0, "failed": 0,
                            "errors": 0, "output": "", "line_coverage": None,
                            "branch_coverage": None, **fields})


def test_a_failing_run_is_read_with_the_tests_its_diagnostics_name():
    """The removal reads a failing coverage run's failed tests with their
    outcomes, then each test a coverage_counts_inconsistent diagnostic names
    that the run did not fail, without an outcome."""
    from langadapter.runs import RunDiagnostic
    from testuary.consolidation import _failed_run
    from testuary.languages.generic import TestFailure

    first, second = _test_id("test_first"), _test_id("test_second")
    run = _result(failed=1, failures=(TestFailure("test_first", "error", "", ""),),
                  stopping_tests=(first,), diagnostics=(
                      RunDiagnostic("coverage_counts_inconsistent", "refused", test=second),
                      RunDiagnostic("coverage_counts_inconsistent", "refused", test=first),
                      RunDiagnostic("unsound_test", "a warning", test=second)))
    failed = _failed_run("coverage", run)
    assert (failed.measurement, failed.tests, failed.outcomes) == (
        "coverage", (first, second), ("error", None))
    assert failed.timed_out is False and failed.result is run
    unpaired = _failed_run("coverage", run._replace(failures=()))
    assert unpaired.tests == (first, second) and unpaired.outcomes == ()


@pytest.mark.parametrize("fields, count, cause", [
    ({"ok": True, "passed": 2}, 2, ""),
    ({}, 0, "the removal left no test"),
    ({}, 1, "the normal run failed after a removal"),
    ({"passed": 1, "failed": 1}, 2, "the normal run failed after a removal"),
    ({"timed_out": True}, 1, "the normal run failed after a removal"),
], ids=["passes", "no-test", "does-not-load", "fails", "timed-out"])
def test_the_normal_run_after_a_removal_decides_whether_the_suite_stays(fields, count, cause):
    """The structural count of the suite after the removal, not the run's
    counts, says whether the removal left no test."""
    from testuary.consolidation import _removal_cause

    backend = SimpleNamespace(test_metrics=lambda source: {"test_count": count})
    assert _removal_cause(backend, "the suite", _result(**fields)) == cause
