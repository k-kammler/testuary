"""Low-level helpers of ``testuary.consolidation``: the collision-resolving
merge and the shared token parsers. test_consolidate_phase_end.py covers
``consolidate_phase_end`` end to end, test_phase_end_selection.py keep-best-of-N."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import hashlib

from testuary import languages
from testuary.consolidation import (
    PhaseConsolidation,
    PhaseEndContext,
    _EvidenceRepair,
    _covered_lines_of,
    _merge_sources,
    _killed_ids_of,
    _log_catalogue_diff,
    _measured_time_ms,
    _refresh_reduced_coverage,
)
from testuary.languages import SourceValidation
from testuary.languages.generic import StructureResult
import pytest

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def test_covered_lines_of_parses_line_tokens_only():
    toks = {"L:add:5", "L:add:6", "B:add:5->6", "K:m1", "noise"}
    assert _covered_lines_of(toks) == {5, 6}


def test_covered_lines_of_ignores_malformed_line_tokens():
    assert _covered_lines_of({"L:add:notanint", "L:x"}) == set()
    assert _covered_lines_of(set()) == set()


def test_killed_ids_of_parses_kill_tokens_only():
    toks = {"K:m1", "K:abc-2", "L:add:5", "B:f:1->2"}
    assert _killed_ids_of(toks) == {"m1", "abc-2"}
    assert _killed_ids_of(set()) == set()


def _merge(backend, sources, source_dir, outcome=None):
    """``_merge_sources`` as the phase end calls it, without evidence."""
    return _merge_sources(backend, sources, lambda _message: None,
                          outcome={} if outcome is None else outcome,
                          evidence=[None] * len(sources), source_dir=str(source_dir))


def test_a_measured_time_sum_does_not_depend_on_the_order_of_the_tests():
    # The 9 recorded per-test times of the Java demonstration's coverage
    # reduction, in two orders a set of test names can take. Added one by one
    # from the left, the first order gives 13.260607000000002, the second
    # 13.260607.
    times = {"a": 9.819297, "b": 0.323767, "c": 0.237622, "d": 0.145212, "e": 0.471187,
             "f": 0.327586, "g": 0.64481, "h": 0.811918, "i": 0.479208}
    first = _measured_time_ms(times, ["a", "b", "c", "d", "e", "f", "g", "h", "i"])
    second = _measured_time_ms(times, ["h", "g", "f", "c", "i", "d", "a", "b", "e"])
    assert first == second == 13.260607


def test_merge_sources_joins_two_suites_into_one(tmp_path):
    backend = languages.get_backend("python")
    a = "def test_a():\n    assert 1 == 1\n"
    b = "def test_b():\n    assert 2 == 2\n"
    merged = _merge(backend, [a, b], tmp_path)
    assert merged is not None
    assert "def test_a" in merged and "def test_b" in merged


def test_merge_sources_resolves_a_name_collision_keeping_both_bodies(tmp_path):
    backend = languages.get_backend("python")
    a = "def test_x():\n    assert 1 == 1\n"
    b = "def test_x():\n    assert 2 == 2\n"   # same name, distinct body
    merged = _merge(backend, [a, b], tmp_path)
    assert merged is not None
    # The collision is renamed, never silently dropped: both bodies survive.
    assert merged.count("assert 1 == 1") == 1
    assert merged.count("assert 2 == 2") == 1


def test_identical_sources_merge_to_the_first_byte_for_byte(tmp_path):
    backend = languages.get_backend("python")
    source = "class TestValues:\n    def test_value(self):\n        assert True\n"
    outcome = {}
    assert _merge(backend, [source, source], tmp_path, outcome) == source
    assert outcome["identical_sources"] == 1


def test_merge_sources_returns_none_when_every_source_ejects(tmp_path):
    backend = languages.get_backend("python")
    assert _merge(backend, ["def broken(:\n"], tmp_path) is None


def test_an_unavailable_merge_records_why(monkeypatch, tmp_path):
    """A merge refused because collision rewriting ejected a source records
    the source and its reason, not only in the log."""
    from langadapter.sources import CollisionResolution

    backend = languages.get_backend("python")
    a = "def test_a():\n    assert 1 == 1\n"
    b = "def test_b():\n    assert 2 == 2\n"

    def resolve(self, sources, *, localize_autouse=False, evidence=None):
        return CollisionResolution(sources=tuple(sources), ejected=(1,),
                                   eject_reasons=("the project interpreter rejects it",))

    monkeypatch.setattr(type(backend), "resolve_test_collisions", resolve)
    outcome = {}
    assert _merge(backend, [a, b], tmp_path, outcome) is None
    assert outcome["unavailable"] == (
        "collision rewriting ejected source 1: the project interpreter rejects it")


def test_log_catalogue_diff_names_disagreeing_mutants():
    derived = {"mutants": [{"id": "m1", "status": "killed"},
                           {"id": "m2", "status": "survived"}]}
    measured = {"mutants": [{"id": "m1", "status": "killed"},
                            {"id": "m2", "status": "killed"}]}
    logs: list = []
    _log_catalogue_diff(derived, measured, logs.append)
    # m1 agrees and is silent. m2 disagrees and is named with both statuses.
    assert any("m2" in m and "derived=survived" in m
               and "measured=killed" in m for m in logs)
    assert not any("m1" in m for m in logs)


def test_refresh_reduced_coverage_uses_the_fresh_aggregate_sets():
    staged: list = []

    class Backend:
        @staticmethod
        def scratch_test_path(directory, _source, target_file_path,
                              *, stage_as_delivered):
            staged.append((target_file_path, stage_as_delivered))
            return Path(directory) / f"test_{Path(target_file_path).stem}.py"

        @staticmethod
        def run_test_files(*_args, **_kwargs):
            return SimpleNamespace(
                ok=True,
                line_coverage=75.0,
                branch_coverage=50.0,
                covered_lines=(2, 4, 8),
                covered_branches=("2->4",),
            )

    result = PhaseConsolidation(
        line=100.0,
        branch=100.0,
        suite_covered_lines=[1, 2, 4, 8],
        suite_covered_branches=["1->2", "2->4"],
    )

    context = PhaseEndContext(
        backend=Backend(), target_file_path="pkg/mod.py", source_dir=".",
        timeout=10, stage_as_delivered=False, log=lambda _message: None)
    assert _refresh_reduced_coverage(
        context, source="def test_x():\n    pass\n", result=result)
    # The staging receives the target and the boundary's naming choice.
    assert staged == [("pkg/mod.py", False)]
    assert result.line == 75.0
    assert result.branch == 50.0
    assert result.suite_covered_lines == [2, 4, 8]
    assert result.suite_covered_branches == ["2->4"]


def test_evidence_repair_restores_lines_branches_and_kills():
    class Backend:
        @staticmethod
        def prune_tests(_source, remove):
            kept = sorted({"base", "line", "branch", "kill"} - set(remove))
            return "".join(f"# {test_id}\n" for test_id in kept)

        @staticmethod
        def finalize_test_source(source):
            return source

        @staticmethod
        def check_structure(_source):
            return StructureResult(True, "", 0, 0)

        @staticmethod
        def validate_test_source(source, _source_dir):
            return SourceValidation(
                True, "valid", "", 0, 0, "fake-target",
                hashlib.sha256(source.encode("utf-8")).hexdigest(),
            )

        @staticmethod
        def scratch_test_path(directory, _source, _target, *, stage_as_delivered):
            assert stage_as_delivered is True
            return Path(directory) / "test_target.py"

        @staticmethod
        def run_test_files(paths, _source_dir, **_kwargs):
            source = Path(paths[0]).read_text(encoding="utf-8")
            lines = (10,) if "# line\n" in source else ()
            branches = ("1->2",) if "# branch\n" in source else ()
            return SimpleNamespace(
                ok=True,
                line_coverage=100.0 if lines else 0.0,
                branch_coverage=100.0 if branches else 0.0,
                covered_lines=lines,
                covered_branches=branches,
            )

    def mutation_runner(directory):
        source = (Path(directory) / "test_target.py").read_text(encoding="utf-8")
        killed = "# kill\n" in source
        return {"mutants": [{
            "id": "m1",
            "operator": "op",
            "line": 20,
            "status": "killed" if killed else "survived",
            "killer": "test_target.py::test_kill" if killed else None,
        }]}

    result = PhaseConsolidation(
        mutation_method="both",
        merged_km={"mutants": [{
            "id": "m1", "operator": "op", "line": 20,
            "status": "killed", "killer": "test_target.py::test_kill",
        }]},
        reduced_km={"mutants": [{
            "id": "m1", "operator": "op", "line": 20,
            "status": "survived", "killer": None,
        }]},
        union_covered_lines=[10],
        union_covered_branches=["1->2"],
        suite_covered_lines=[],
        suite_covered_branches=[],
        line=0.0,
        branch=0.0, kept_tests=["base"],
    )
    tokens = {
        "base": frozenset(),
        "line": frozenset({"L:target.py:10"}),
        "branch": frozenset({"B:target.py:1->2"}),
        "kill": frozenset({"K:m1"}),
    }
    _EvidenceRepair(
        PhaseEndContext(backend=Backend(), target_file_path="target.py",
                        source_dir=".", timeout=10, stage_as_delivered=True,
                        log=lambda _message: None),
        mutation_runner,
        merged="# base\n# line\n# branch\n# kill\n",
        res=result,
        per_test_tokens=tokens,
        solver_cost_ms={test_id: 1.0 for test_id in tokens},
    ).run({"base"})
    kept = set(result.kept_tests)

    assert kept == {"base", "line", "branch", "kill"}
    assert result.suite_covered_lines == [10]
    assert result.suite_covered_branches == ["1->2"]
    assert result.evidence_repairs[0]["accepted"] is True
    assert result.evidence_repairs[0]["added_tests"] == [
        "branch", "kill", "line",
    ]
    assert result.derive_kept_tests == ["base", "branch", "kill", "line"]
    assert result.derived_reduced_km is not None
    assert result.measured_reduced_km is result.reduced_km
    assert result.derive_measure_delta == 0.0
    assert result.derive_measure_differences == []


def test_evidence_repair_takes_the_crash_observed_test_as_a_kills_carrier():
    """A kill without a named killer whose run ended inside a test gives that
    test a crash ``C:`` token, not a kill token. The repair still takes it as
    the kill's carrier, before a cheaper test that only covers the mutated
    line."""
    tests = ("base", "crash", "liner")

    class Backend:
        @staticmethod
        def prune_tests(_source, remove):
            return "".join(f"# {t}\n" for t in tests if t not in remove)

        @staticmethod
        def finalize_test_source(source):
            return source

        @staticmethod
        def check_structure(_source):
            return StructureResult(True, "", 0, 0)

        @staticmethod
        def validate_test_source(source, _source_dir):
            return SourceValidation(
                True, "valid", "", 0, 0, "fake-target",
                hashlib.sha256(source.encode("utf-8")).hexdigest(),
            )

        @staticmethod
        def scratch_test_path(directory, _source, _target, *, stage_as_delivered):
            return Path(directory) / "test_target.py"

        @staticmethod
        def run_test_files(_paths, _source_dir, **_kwargs):
            return SimpleNamespace(ok=True, line_coverage=100.0,
                                   branch_coverage=None, covered_lines=(),
                                   covered_branches=())

    def mutation_runner(directory):
        source = (Path(directory) / "test_target.py").read_text(encoding="utf-8")
        return {"mutants": [{
            "id": "m1", "operator": "op", "line": 20,
            "status": "killed" if "# crash\n" in source else "survived",
            "killer": None, "killers": [],
        }]}

    crashed = {"id": "m1", "operator": "op", "line": 20, "status": "killed",
               "killer": None, "killers": [],
               "active_at_end": "test_target.py::test_crash"}
    result = PhaseConsolidation(
        mutation_method="measure",
        merged_km={"mutants": [crashed]},
        reduced_km={"mutants": [{**crashed, "status": "survived"}]},
        union_covered_lines=[], union_covered_branches=[],
        suite_covered_lines=[], suite_covered_branches=[],
        line=100.0, branch=None, kept_tests=["base"],
    )
    tokens = {"base": frozenset(), "crash": frozenset({"C:m1"}),
              "liner": frozenset({"L:target.py:20"})}
    _EvidenceRepair(
        PhaseEndContext(backend=Backend(), target_file_path="target.py",
                        source_dir=".", timeout=10, stage_as_delivered=True,
                        log=lambda _message: None),
        mutation_runner, merged="# base\n# crash\n# liner\n",
        res=result, per_test_tokens=tokens,
        solver_cost_ms={"base": 1.0, "crash": 50.0, "liner": 1.0},
    ).run({"base"})
    kept = set(result.kept_tests)
    assert kept == {"base", "crash"}
    assert result.evidence_repairs[0]["added_tests"] == ["crash"]


def test_evidence_repair_records_the_round_limit_with_carriers_remaining():
    all_tests = {"base", "carrier_1", "carrier_2", "carrier_3", "carrier_4"}

    class Backend:
        @staticmethod
        def prune_tests(_source, remove):
            kept = sorted(all_tests - set(remove))
            return "".join(f"# {test_id}\n" for test_id in kept)

        @staticmethod
        def finalize_test_source(source):
            return source

        @staticmethod
        def check_structure(_source):
            return StructureResult(True, "", 0, 0)

        @staticmethod
        def validate_test_source(source, _source_dir):
            return SourceValidation(
                True, "valid", "", 0, 0, "fake-target",
                hashlib.sha256(source.encode("utf-8")).hexdigest(),
            )

        @staticmethod
        def scratch_test_path(directory, _source, _target, *, stage_as_delivered):
            return Path(directory) / "test_target.py"

        @staticmethod
        def run_test_files(*args, **_kwargs):
            return SimpleNamespace(
                ok=True,
                line_coverage=100.0,
                branch_coverage=100.0,
                covered_lines=(10,),
                covered_branches=(),
            )

    def mutation_runner(directory):
        source = (Path(directory) / "test_target.py").read_text(encoding="utf-8")
        restored = all(f"# carrier_{index}\n" in source for index in range(1, 5))
        return {"mutants": [{
            "id": "m_timeout",
            "operator": "probe/Timeout",
            "line": 10,
            "status": "timeout" if restored else "survived",
            "killer": None,
            "killers": [],
        }]}

    result = PhaseConsolidation(
        mutation_method="both",
        merged_km={"mutants": [{
            "id": "m_timeout", "operator": "probe/Timeout", "line": 10,
            "status": "timeout", "killer": None, "killers": [],
        }]},
        reduced_km={"mutants": [{
            "id": "m_timeout", "operator": "probe/Timeout", "line": 10,
            "status": "survived", "killer": None, "killers": [],
        }]},
        union_covered_lines=[10],
        union_covered_branches=[],
        suite_covered_lines=[10],
        suite_covered_branches=[],
        line=100.0,
        branch=100.0, kept_tests=["base"],
    )
    tokens = {
        "base": frozenset(),
        **{
            f"carrier_{index}": frozenset({"L:target.py:10"})
            for index in range(1, 5)
        },
    }
    _EvidenceRepair(
        PhaseEndContext(backend=Backend(), target_file_path="target.py",
                        source_dir=".", timeout=10, stage_as_delivered=True,
                        log=lambda _message: None),
        mutation_runner,
        merged="".join(f"# {test_id}\n" for test_id in sorted(all_tests)),
        res=result,
        per_test_tokens=tokens,
        solver_cost_ms={test_id: float(index) for index, test_id in enumerate(tokens)},
    ).run({"base"})
    kept = set(result.kept_tests)

    assert kept == {"base", "carrier_1", "carrier_2", "carrier_3"}
    assert [entry["outcome"] for entry in result.evidence_repairs] == [
        "accepted", "accepted", "accepted", "repair_budget_exhausted",
    ]
    assert result.evidence_repairs[-1]["missing_kills_after"]
    assert "carrier_4" not in kept


def test_order_dependent_mock_warning_names_the_positional_pattern():
    """A ``side_effect`` list of distinct returns asserted against a named
    element depends on scheduling, so the test is flagged by name (advisory)."""
    backend = languages.get_backend("python")
    source = (
        "from unittest import mock\n\n"
        "class TestOpen:\n"
        "    def test_attrs_file_used(self):\n"
        "        ds1, ds2 = object(), object()\n"
        "        m = mock.MagicMock()\n"
        "        m.side_effect = [ds1, ds2]\n"
        "        result = m()\n"
        "        assert result is ds1\n\n"
        "def test_plain_side_effect_exception_is_not_flagged():\n"
        "    m = __import__('unittest.mock', fromlist=['MagicMock'])\n"
        "    mm = m.MagicMock()\n"
        "    mm.side_effect = ValueError\n"
        "    assert mm is not None\n\n"
        "def test_list_without_positional_assertion_is_not_flagged():\n"
        "    import unittest.mock as mock\n"
        "    m = mock.MagicMock()\n"
        "    m.side_effect = [1, 2]\n"
        "    assert m() == 1\n"
    )
    assert backend.order_dependent_mock_warnings(source) == [
        "TestOpen.test_attrs_file_used"]


def test_order_dependent_mock_warning_tolerates_broken_source():
    backend = languages.get_backend("python")
    assert backend.order_dependent_mock_warnings("def broken(:\n") == []


# ---------------------------------------------------------------------------
# Why a probe was red
# ---------------------------------------------------------------------------

def test_a_timed_out_probe_is_recorded_as_a_timeout_not_as_failures():
    """A timeout and failures both arrive as ok=False, so only the record
    tells a machine problem from a suite problem."""
    from types import SimpleNamespace

    from testuary.consolidation import red_verdict

    timed_out = SimpleNamespace(
        ok=False, timed_out=True, returncode=-1,
        collected=0, passed=0, failed=0, errors=0)
    assert "timed out" in red_verdict(timed_out)

    red = SimpleNamespace(
        ok=False, timed_out=False, returncode=1,
        collected=7, passed=5, failed=2, errors=0)
    verdict = red_verdict(red)
    assert "timed out" not in verdict
    assert "failed=2" in verdict and "collected=7" in verdict


def test_a_run_that_ended_without_a_test_result_says_how_in_its_verdict():
    """A run whose test file did not compile, or whose coverage counts the
    coverage tool could not read after its tests passed, reads as such in the
    records, with its reason, not as an exit code with counts alone."""
    from langadapter.runs import RunDiagnostic

    from testuary.consolidation import red_verdict
    from testuary.languages import TestRunResult

    compiled = TestRunResult(False, 1, False, 0.1, 0, 0, 0, 0, "", None, None, diagnostics=(
        RunDiagnostic("compile_error", "the test file test_calc.c does not compile"),))
    assert red_verdict(compiled) == (
        "Verdict: exit 1 (collected=0, passed=0, failed=0, errors=0). "
        "It ended as a collection failure: the test file test_calc.c does not compile.")
    unread = TestRunResult(False, 0, False, 0.1, 2, 2, 0, 0, "", None, None, diagnostics=(
        RunDiagnostic("coverage_counts_inconsistent", "gcov read a negative count"),))
    assert red_verdict(unread) == (
        "Verdict: exit 0 (collected=2, passed=2, failed=0, errors=0). "
        "The coverage tool could not read its coverage counts: gcov read a negative count.")


def test_a_keyed_mutant_is_identified_by_file_operator_and_occurrence():
    """Two mutants of one operator on one line with the same change differ
    only by occurrence: the backend's key keeps them apart across captures,
    whatever their ids."""
    from testuary.consolidation import _killed_by_identity, _log_catalogue_diff

    def row(mid, occurrence, status="killed"):
        return {"id": mid, "status": status, "operator": "ReplaceComparison",
                "line": 7, "diff": "- a < b\n+ a <= b",
                "key": {"file": "pkg/m.py", "operator": "ReplaceComparison",
                        "occurrence": occurrence, "line": 7}}

    merged = {"mutants": [row("a1", 0), row("a2", 1)]}
    remeasure = {"mutants": [row("b1", 0), row("b2", 1, status="survived")]}
    assert len(_killed_by_identity(merged)) == 2
    assert _killed_by_identity(merged).keys() - _killed_by_identity(remeasure).keys() == {
        ("key", "pkg/m.py", "ReplaceComparison", 1)}
    assert _log_catalogue_diff(merged, remeasure, lambda _line: None) == [
        {"mutant_id": "b2", "derived_status": "killed", "measured_status": "survived"}]
    unkeyed = {k: v for k, v in row("c", 0).items() if k != "key"}
    assert set(_killed_by_identity({"mutants": [unkeyed]})) == {
        ("catalogue", "ReplaceComparison", 7, "- a < b\n+ a <= b")}


def _carrier_trial(monkeypatch, *, normal_ok, coverage_ok):
    """One carrier repair round with a fake backend whose normal run (no
    coverage tool) passes when ``normal_ok`` and whose run under the coverage
    tool passes when ``coverage_ok``; the load recovery finds nothing."""
    from testuary import consolidation

    class Backend:
        @staticmethod
        def prune_tests(_source, remove):
            return "".join(f"# {t}\n" for t in sorted({"base", "line"} - set(remove)))

        @staticmethod
        def finalize_test_source(source):
            return source

        @staticmethod
        def check_structure(_source):
            return StructureResult(True, "", 0, 0)

        @staticmethod
        def validate_test_source(source, _source_dir):
            return SourceValidation(
                True, "valid", "", 0, 0, "fake-target",
                hashlib.sha256(source.encode("utf-8")).hexdigest())

        @staticmethod
        def scratch_test_path(directory, _source, _target, *, stage_as_delivered):
            return Path(directory) / "test_target.py"

        @staticmethod
        def run_test_files(paths, _source_dir, *, measure_coverage_of=None, **_kwargs):
            ok = coverage_ok if measure_coverage_of is not None else normal_ok
            return SimpleNamespace(
                ok=ok, timed_out=False, returncode=0 if ok else 1, collected=2,
                passed=2 if ok else 1, failed=0 if ok else 1, errors=0, failures=(),
                line_coverage=100.0, branch_coverage=None, covered_lines=(10,),
                covered_branches=())

    monkeypatch.setattr(consolidation.loadable, "keep_if_loadable",
                        lambda _backend, source, *a, **kw: (source, {}))
    measured = []
    result = PhaseConsolidation(
        mutation_method="measure", merged_km={"mutants": []}, reduced_km={"mutants": []},
        union_covered_lines=[10], union_covered_branches=[], suite_covered_lines=[],
        suite_covered_branches=[], line=0.0, branch=None, kept_tests=["base"])
    _EvidenceRepair(
        PhaseEndContext(backend=Backend(), target_file_path="target.py",
                        source_dir=".", timeout=10, stage_as_delivered=False,
                        log=lambda _message: None),
        lambda directory: measured.append(directory) or {"mutants": []},
        merged="# base\n# line\n", res=result,
        per_test_tokens={"base": frozenset(), "line": frozenset({"L:target.py:10"})},
        solver_cost_ms={"base": 1.0, "line": 1.0}).run({"base"})
    kept = set(result.kept_tests)
    return kept, result.evidence_repairs[0], measured


def test_a_carrier_trial_whose_normal_run_fails_is_not_accepted(monkeypatch):
    """The normal run decides whether the trial passes: one that fails it is
    not kept, even when the coverage tool's run of it passes."""
    kept, attempt, measured = _carrier_trial(monkeypatch, normal_ok=False, coverage_ok=True)
    assert kept == {"base"}
    assert attempt["outcome"] == "not_passing"
    assert measured == []


def test_a_carrier_trial_the_coverage_tool_cannot_measure_is_incomplete(monkeypatch):
    """A trial that passes its normal run but that the coverage tool could
    not measure has no numbers: it is not kept, recorded as unmeasured, not
    as failing."""
    kept, attempt, measured = _carrier_trial(monkeypatch, normal_ok=True, coverage_ok=False)
    assert kept == {"base"}
    assert attempt["outcome"] == "coverage_measurement_incomplete"
    assert measured == []
