"""The removal fallback at the starter's places: a measurement that
fails a starter suite whose normal run passed removes the tests it names,
the suite runs and is measured again, and the attempt's cleanup record and
the slot's records say what was removed."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

import test_starter_phase as harness

pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture(autouse=True)
def _consolidation_mutation_off(monkeypatch):
    """No mutation tool starts against these synthetic suites."""
    from testuary import config

    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: False)


_PROBED = """\
from calc import add

def test_keep():
    assert add(1, 1) == 2

def test_flaky_a():
    assert add(2, 2) == 4

def test_flaky_b():
    assert add(3, 3) == 6
"""

_ONLY_FLAKY = """\
from calc import add

def test_flaky_a():
    assert add(2, 2) == 4
"""


def _coverage_tool_names(monkeypatch, *markers, timed_out=False, names=None,
                         diagnostic=False):
    """The coverage tool fails each run of a suite that holds one of the
    tests ``markers``. It fails the first one it holds with the outcome
    ``failed`` and names it, or names the tests ``names`` whether the suite
    holds them or not (none for an empty tuple), or times out with
    ``timed_out``. With ``diagnostic`` the run fails no test, and a
    coverage_counts_inconsistent diagnostic names each test instead. Every
    other run is the backend's own."""
    from langadapter.runs import RunDiagnostic
    from langadapter.suites import TestId
    from testuary import languages
    from testuary.languages.generic import TestFailure

    backend_type = type(languages.get_backend("python"))
    real = backend_type.run_test_files

    def run_test_files(self, paths, source_dir, *args, **kwargs):
        result = real(self, paths, source_dir, *args, **kwargs)
        text = "".join(Path(path).read_text(encoding="utf-8") for path in paths)
        held = [marker for marker in markers if f"def {marker}(" in text]
        if not (kwargs.get("measure_coverage_of") and held):
            return result
        name = Path(paths[0]).name
        named = held[:1] if names is None else list(names)
        tests = tuple(TestId(name, test, text=f"{name}::{test}") for test in named)
        if diagnostic:
            return result._replace(ok=False, returncode=1, line_coverage=None,
                                   branch_coverage=None, diagnostics=tuple(
                                       RunDiagnostic("coverage_counts_inconsistent", "refused",
                                                     test=test) for test in tests))
        return result._replace(
            ok=False, returncode=1, timed_out=timed_out, failed=len(named or held[:1]),
            failures=tuple(TestFailure(test, "failed", "fails under the tool", "")
                           for test in named or held[:1]),
            stopping_tests=tests, line_coverage=None, branch_coverage=None)

    monkeypatch.setattr(backend_type, "run_test_files", run_test_files)


def _slot(suites_dir, i=1):
    sdir = suites_dir / f"suite_{i:03d}"
    return sdir, json.loads((sdir / "original.json").read_text(encoding="utf-8"))


def test_the_probe_removes_the_test_its_coverage_run_fails(tmp_path, monkeypatch):
    """A suite that passes its normal run but whose probe the coverage tool
    fails on one test loses that test, runs and is probed again, and is
    accepted with its other tests. The cleanup record and the slot's
    validity derivations name the removal."""
    _coverage_tool_names(monkeypatch, "test_flaky_a")
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    res, _calls = harness._run(root, target, tests_dir, suites_dir, [_PROBED], monkeypatch,
                               refill_failed_suites=False)
    assert res.status == "ok" and res.n_green == 1 and res.n_unmeasured_coverage == 0
    sdir, meta = _slot(suites_dir)
    assert meta["classification"] != "coverage_unmeasured"
    cleaned = (sdir / "cleaned.py").read_text(encoding="utf-8")
    assert "def test_flaky_a" not in cleaned and "def test_keep" in cleaned
    [entry] = meta["cleanup"]["measurement_removals"]
    [removed] = entry.pop("removed")
    assert entry == {"place": "starter.zero_coverage_probe", "measurement": "coverage",
                     "round": 1}
    assert removed["test"].endswith("test_flaky_a") and removed["outcome"] == "failed"
    [record] = json.loads((sdir / "validity_derivations.json").read_text(
        encoding="utf-8"))["records"]
    assert (record["operation"], record["stage"]) == (
        "measurement_remove", "starter.zero_coverage_probe")
    assert record["removed_tests"] == [removed["test"]]


def test_the_probe_stops_at_the_cleanup_rounds_on_a_last_attempt(tmp_path, monkeypatch):
    """On a suite's last attempt the probe removes within
    failing_test_cleanup_rounds and then discards the attempt as today, its
    reason unchanged, the entry saying that the rounds are spent."""
    _coverage_tool_names(monkeypatch, "test_flaky_a", "test_flaky_b")
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    res, _calls = harness._run(root, target, tests_dir, suites_dir, [_PROBED], monkeypatch,
                               refill_failed_suites=False, failing_test_cleanup_rounds=1)
    assert res.status == "cancelled_zero_passing"
    _sdir, meta = _slot(suites_dir)
    assert meta["classification"] == "coverage_unmeasured"
    assert meta["cleanup"]["disposition_reason"].startswith(
        "the coverage tool could not measure the suite: ")
    first, last = meta["cleanup"]["measurement_removals"]
    assert first["round"] == 1 and first["removed"][0]["test"].endswith("test_flaky_a")
    assert last == {"place": "starter.zero_coverage_probe", "measurement": "coverage",
                    "round": 2, "unnamed": "the rounds are spent"}


@pytest.mark.parametrize("suite, options, cause", [
    (_ONLY_FLAKY, {}, "the removal left no test"),
    (_PROBED, {"names": ()}, "the measurement named no test"),
    (_PROBED, {"timed_out": True}, "the measurement timed out"),
], ids=["emptied", "unnamed", "timed-out"])
def test_a_probe_without_a_usable_removal_takes_today_s_branch(
        tmp_path, monkeypatch, suite, options, cause):
    """A removal that leaves no test, a failure that names no test and a
    timed-out run discard the attempt as coverage_unmeasured with today's
    reason, and the entry states the cause."""
    _coverage_tool_names(monkeypatch, "test_flaky_a", **options)
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    res, _calls = harness._run(root, target, tests_dir, suites_dir, [suite], monkeypatch,
                               refill_failed_suites=False)
    assert res.status == "cancelled_zero_passing"
    sdir, meta = _slot(suites_dir)
    assert meta["classification"] == "coverage_unmeasured"
    assert meta["cleanup"]["disposition_reason"].startswith(
        "the coverage tool could not measure the suite: ")
    assert meta["cleanup"]["measurement_removals"][-1]["unnamed"] == cause
    assert not (sdir / "validity_derivations.json").exists()


def test_the_probe_removes_a_test_a_coverage_diagnostic_names(tmp_path, monkeypatch):
    """A probe whose coverage run fails no test, but whose
    coverage_counts_inconsistent diagnostic names one, loses that test and
    is accepted with its other tests."""
    _coverage_tool_names(monkeypatch, "test_flaky_a", diagnostic=True)
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    res, _calls = harness._run(root, target, tests_dir, suites_dir, [_PROBED], monkeypatch,
                               refill_failed_suites=False)
    assert res.status == "ok" and res.n_green == 1
    sdir, meta = _slot(suites_dir)
    cleaned = (sdir / "cleaned.py").read_text(encoding="utf-8")
    assert "def test_flaky_a" not in cleaned and "def test_flaky_b" in cleaned
    [entry] = meta["cleanup"]["measurement_removals"]
    [removed] = entry["removed"]
    assert removed["test"].endswith("test_flaky_a") and "outcome" not in removed


@pytest.mark.parametrize("names", [("test_flaky_a", "test_ghost"), ("test_ghost",)],
                         ids=["mixed", "unknown"])
def test_the_probe_removes_the_held_names_and_records_the_unknown_ones(
        tmp_path, monkeypatch, names):
    """A probe's coverage run that names a held test and one the suite does
    not hold removes the held one and records the other as unmatched. One
    that names only unknown tests removes nothing and discards the attempt
    as today."""
    _coverage_tool_names(monkeypatch, "test_flaky_a", names=names)
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    res, _calls = harness._run(root, target, tests_dir, suites_dir, [_PROBED], monkeypatch,
                               refill_failed_suites=False)
    sdir, meta = _slot(suites_dir)
    [entry] = meta["cleanup"]["measurement_removals"]
    [unknown] = entry.pop("unmatched")
    assert unknown.endswith("test_ghost")
    if len(names) == 2:
        assert res.status == "ok" and res.n_green == 1
        [removed] = entry["removed"]
        assert removed["test"].endswith("test_flaky_a")
        assert "def test_flaky_a" not in (sdir / "cleaned.py").read_text(encoding="utf-8")
    else:
        assert res.status == "cancelled_zero_passing"
        assert meta["classification"] == "coverage_unmeasured"
        assert entry == {"place": "starter.zero_coverage_probe", "measurement": "coverage",
                         "round": 1, "unnamed": "the suite holds no named test"}


_ALPHA_FLAKY = """\
from calc import add

def test_alpha():
    assert add(1, 1) == 2

def test_flaky_a():
    assert add(2, 2) == 4
"""

_BETA_FLAKY = """\
from calc import add

def test_beta():
    assert add(2, 2) == 4

def test_flaky_b():
    assert add(3, 3) == 6
"""


def _named_in(test_dir, *markers):
    """The id objects of the tests ``markers`` that the starter suite staged
    alone in ``test_dir`` holds."""
    from langadapter.suites import TestId

    found = []
    for path in Path(test_dir).glob("*.py"):
        text = path.read_text(encoding="utf-8")
        found += [TestId(path.name, marker, text=f"{path.name}::{marker}")
                  for marker in markers if f"def {marker}(" in text]
    return tuple(found)


def _kills_one():
    """A kill matrix in which the suite kills one of two mutants."""
    return {"mutants": [{"id": "m1", "operator": "core/Op", "line": 2, "diff": "m1",
                         "status": "killed", "killer": "x.py::test_alpha"},
                        {"id": "m2", "operator": "core/Op", "line": 3, "diff": "m2",
                         "status": "survived"}]}


def _mutation_tool_names(monkeypatch, *markers):
    """The stand-in mutation tool's check before the first mutant stops a
    starter suite measured alone that holds one of the tests ``markers``,
    naming it. Every other suite kills one of two mutants. Returns the
    stand-in, whose ``per_suite.calls`` counts the per-suite requests."""
    from per_suite_double import measured_nothing

    def measure(test_dir):
        named = _named_in(test_dir, *markers)
        if Path(test_dir).name.startswith("testuary_starter") and named:
            return measured_nothing("test_failures", named=named)
        return _kills_one()

    return _mutation_tool(monkeypatch, measure)


def _mutation_tool(monkeypatch, measure):
    """The stand-in mutation tool whose runs ``measure(test_dir)`` gives,
    for the starter and the phase end. Returns the stand-in."""
    from contextlib import contextmanager
    from per_suite_double import per_suite_double
    from testuary import config
    from testuary import consolidation as consolidate_mod
    from testuary import measurement as test_metrics

    double = per_suite_double(measure)

    @contextmanager
    def fake_runner(**_kwargs):
        yield double

    monkeypatch.setattr(test_metrics, "union_mutation_runner", fake_runner)
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: True)
    monkeypatch.setattr(consolidate_mod, "benchmark_tied_candidates",
                        lambda *_args, **_kwargs: lambda _candidate: 1.0)
    return double


def _evidence_names(monkeypatch, marker, *, named=True):
    """The per-test evidence run of a starter suite measured alone that holds
    the test ``marker`` fails, naming that test unless not ``named``."""
    from testuary.languages import get_backend

    backend_type = type(get_backend("python"))
    collect = backend_type.collect_eviction_data

    def failing(self, files, *args, **kwargs):
        if harness._alone(files, f"def {marker}("):
            tests = _named_in(Path(files[0]).parent, marker) if named else ()
            kwargs["failure_fn"]("test_failure", f"{marker} failed under the coverage tool",
                                 **({"stopping_tests": tests} if tests else {}))
            return None
        return collect(self, files, *args, **kwargs)

    monkeypatch.setattr(backend_type, "collect_eviction_data", failing)


def _manifest_slot(suites_dir, i):
    return harness._suite_slots(suites_dir)[i - 1]


def test_one_pass_removes_what_the_per_suite_mutation_names_in_each_suite(
        tmp_path, monkeypatch):
    """The per-suite mutation names a test in each of two starter suites:
    both are removed in the same pass, the pool is measured once more, and
    both slots keep their acceptance and name their changed suite."""
    double = _mutation_tool_names(monkeypatch, "test_flaky_a", "test_flaky_b")
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    res, calls = harness._run(root, target, tests_dir, suites_dir, [_ALPHA_FLAKY, _BETA_FLAKY],
                              monkeypatch, matrices_dir=tmp_path / "matrices",
                              refill_failed_suites=False)
    assert res.status == "ok" and (res.n_green, res.n_unmeasured_mutation) == (2, 0)
    assert calls["n"] == 2 and double.per_suite.calls == 2
    for i, marker in ((1, "test_flaky_a"), (2, "test_flaky_b")):
        sdir, meta = _slot(suites_dir, i)
        assert f"def {marker}" not in (sdir / "cleaned.py").read_text(encoding="utf-8")
        [entry] = meta["cleanup"]["measurement_removals"]
        assert entry["place"] == "starter.per_suite_mutation" and entry["round"] == 1
        assert entry["measurement"] == "mutation"
        [removed] = entry["removed"]
        assert removed["test"].endswith(marker) and "outcome" not in removed
        slot = _manifest_slot(suites_dir, i)
        assert slot["state"] == "accepted"
        assert slot["measurement_removed_source"]["source_sha256"] != (
            slot["accepted_source"]["source_sha256"])
        [record] = json.loads((sdir / "validity_derivations.json").read_text(
            encoding="utf-8"))["records"]
        assert record["stage"] == "starter.per_suite_mutation"


def test_a_suite_the_removal_empties_is_freed_with_today_s_reason(tmp_path, monkeypatch):
    """A per-suite mutation that names every test of a suite frees its slot
    as mutation_unmeasured with today's reason, and the entry says that the
    removal left no test."""
    _mutation_tool_names(monkeypatch, "test_flaky_a")
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    res, _calls = harness._run(root, target, tests_dir, suites_dir, [_ONLY_FLAKY, _BETA_FLAKY],
                               monkeypatch, matrices_dir=tmp_path / "matrices",
                               refill_failed_suites=False)
    assert res.status == "ok" and (res.n_green, res.n_unmeasured_mutation) == (1, 1)
    _sdir, meta = _slot(suites_dir, 1)
    assert meta["classification"] == "mutation_unmeasured"
    assert meta["cleanup"]["disposition_reason"] == (
        "the mutation tool measured nothing of the suite (test_failures)")
    assert meta["cleanup"]["measurement_removals"][-1] == {
        "place": "starter.per_suite_mutation", "measurement": "mutation", "round": 2,
        "unnamed": "the removal left no test"}
    assert "measurement_removed_source" not in _manifest_slot(suites_dir, 1)


def test_the_evidence_run_s_named_test_is_removed(tmp_path, monkeypatch):
    """A starter suite whose per-test evidence run fails on a named test
    loses that test and is measured again in the next pass."""
    _evidence_names(monkeypatch, "test_flaky_a")
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    res, _calls = harness._run(root, target, tests_dir, suites_dir, [_ALPHA_FLAKY, _BETA_FLAKY],
                               monkeypatch, refill_failed_suites=False)
    assert res.status == "ok" and (res.n_green, res.n_unmeasured_coverage) == (2, 0)
    _sdir, meta = _slot(suites_dir, 1)
    [entry] = meta["cleanup"]["measurement_removals"]
    assert (entry["place"], entry["measurement"]) == ("starter.per_suite_evidence", "evidence")
    assert entry["removed"][0]["test"].endswith("test_flaky_a")


def test_an_evidence_run_that_names_no_test_frees_the_slot_with_today_s_reason(
        tmp_path, monkeypatch):
    _evidence_names(monkeypatch, "test_flaky_a", named=False)
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    res, _calls = harness._run(root, target, tests_dir, suites_dir, [_ALPHA_FLAKY, _BETA_FLAKY],
                               monkeypatch, refill_failed_suites=False)
    assert res.status == "ok" and (res.n_green, res.n_unmeasured_coverage) == (1, 1)
    _sdir, meta = _slot(suites_dir, 1)
    assert meta["classification"] == "coverage_unmeasured"
    assert meta["cleanup"]["disposition_reason"] == (
        "the coverage tool could not measure the suite alone: "
        "test_flaky_a failed under the coverage tool")
    assert meta["cleanup"]["measurement_removals"] == [{
        "place": "starter.per_suite_evidence", "measurement": "evidence", "round": 1,
        "unnamed": "the measurement named no test"}]


def test_the_evidence_run_s_retry_gives_the_names_the_removal_reads(tmp_path, monkeypatch):
    """A starter suite's evidence run that fails naming a test, and on its
    retry times out naming none, removes nothing: the slot is freed with
    today's reason, and the entry says that the measurement named no test."""
    from testuary.languages import get_backend

    backend_type = type(get_backend("python"))
    collect = backend_type.collect_eviction_data
    attempts = []

    def failing(self, files, *args, **kwargs):
        if not harness._alone(files, "def test_flaky_a("):
            return collect(self, files, *args, **kwargs)
        attempts.append(files)
        if len(attempts) == 1:
            kwargs["failure_fn"]("test_failure", "test_flaky_a failed under the coverage tool",
                                 stopping_tests=_named_in(Path(files[0]).parent, "test_flaky_a"))
        else:
            kwargs["failure_fn"]("timeout", "the evidence run timed out")
        return None

    monkeypatch.setattr(backend_type, "collect_eviction_data", failing)
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    res, _calls = harness._run(root, target, tests_dir, suites_dir, [_ALPHA_FLAKY, _BETA_FLAKY],
                               monkeypatch, refill_failed_suites=False)
    assert res.status == "ok" and (res.n_green, res.n_unmeasured_coverage) == (1, 1)
    assert len(attempts) == 2
    _sdir, meta = _slot(suites_dir, 1)
    assert meta["classification"] == "coverage_unmeasured"
    assert meta["cleanup"]["disposition_reason"] == (
        "the coverage tool could not measure the suite alone: the evidence run timed out")
    assert meta["cleanup"]["measurement_removals"] == [{
        "place": "starter.per_suite_evidence", "measurement": "evidence", "round": 1,
        "unnamed": "the measurement named no test"}]


def test_a_per_suite_request_that_measures_no_suite_writes_an_unnamed_entry(
        tmp_path, monkeypatch):
    """A per-suite mutation request in which no starter suite passes the
    tool's green check measures no suite. The suites stay as today, nothing
    is removed, and each suite's entry states the cause."""
    double = _mutation_tool(monkeypatch, lambda test_dir: (
        None if Path(test_dir).name.startswith("testuary_starter") else _kills_one()))
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    res, _calls = harness._run(root, target, tests_dir, suites_dir, [_ALPHA_FLAKY, _BETA_FLAKY],
                               monkeypatch, matrices_dir=tmp_path / "matrices",
                               refill_failed_suites=False)
    assert res.status == "ok" and res.n_green == 2 and double.per_suite.calls == 1
    for i in (1, 2):
        sdir, meta = _slot(suites_dir, i)
        assert "def test_flaky_" in (sdir / "cleaned.py").read_text(encoding="utf-8")
        assert meta["cleanup"]["measurement_removals"] == [{
            "place": "starter.per_suite_mutation", "measurement": "mutation", "round": 1,
            "unnamed": "the mutation tool measured no suite"}]
        assert _manifest_slot(suites_dir, i)["state"] == "accepted"


def test_the_pooled_mutation_s_named_test_is_not_removed(tmp_path, monkeypatch):
    """The pooled mutation run of the starter suites stops on a test it
    names: no test is removed there, and each suite is measured alone
    instead, as today."""
    def measure(test_dir):
        if Path(test_dir).name.startswith("testuary_initkills_"):
            double.named[str(test_dir)] = _named_in(test_dir, "test_flaky_a")
            return None
        return _kills_one()

    double = _mutation_tool(monkeypatch, measure)
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    res, _calls = harness._run(root, target, tests_dir, suites_dir, [_ALPHA_FLAKY, _BETA_FLAKY],
                               monkeypatch, matrices_dir=tmp_path / "matrices",
                               refill_failed_suites=False, suite_mutation="derive")
    assert res.status == "ok" and res.n_green == 2
    assert double.attributions.count("all") == 1 and double.per_suite.calls == 1
    assert [len(names) for names in double.named.values() if names] == [1]
    for i in (1, 2):
        sdir, meta = _slot(suites_dir, i)
        assert "def test_flaky_" in (sdir / "cleaned.py").read_text(encoding="utf-8")
        assert "measurement_removals" not in meta.get("cleanup", {})
        assert not (sdir / "validity_derivations.json").exists()


class _Interrupted(BaseException):
    """An interruption of the run, as a stop of its process leaves it."""


def test_a_removal_in_a_reused_slot_without_evidence_keeps_no_slot_record(
        tmp_path, monkeypatch):
    """A reused pool whose slot evidence cannot be read keeps its suites, its
    slots unavailable. A per-suite removal there acts as at any slot and its
    entry is written, and the slot names no changed suite."""
    _mutation_tool_names(monkeypatch)
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    harness._run(root, target, tests_dir, suites_dir, [_ALPHA_FLAKY, _BETA_FLAKY], monkeypatch,
                 matrices_dir=tmp_path / "matrices", refill_failed_suites=False)
    manifest_path = suites_dir / "_generation.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["suite_slots"] = {}
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    _mutation_tool_names(monkeypatch, "test_flaky_a")
    res, calls = harness._run(root, target, tmp_path / "again", suites_dir,
                              [_ALPHA_FLAKY, _BETA_FLAKY], monkeypatch,
                              matrices_dir=tmp_path / "matrices_again",
                              refill_failed_suites=False, reuse_suites=True)
    assert res.status == "ok" and calls["n"] == 0 and res.n_green == 2
    sdir, meta = _slot(suites_dir, 1)
    assert "def test_flaky_a" not in (sdir / "cleaned.py").read_text(encoding="utf-8")
    [entry] = meta["cleanup"]["measurement_removals"]
    assert entry["place"] == "starter.per_suite_mutation"
    assert entry["removed"][0]["test"].endswith("test_flaky_a")
    slot = _manifest_slot(suites_dir, 1)
    assert slot["state"] == "unavailable" and "measurement_removed_source" not in slot


def test_a_reuse_reads_a_removal_interrupted_before_the_manifest(tmp_path, monkeypatch):
    """A run stopped after a removal rewrote a suite's cleaned file and
    before the manifest named the changed suite leaves the slot's accepted
    source behind its cleaned file. The next reuse loads the changed suite
    and marks the slot as a changed reuse, and a removal there keeps no slot
    record."""
    from testuary import generation_evidence

    def interrupted(self, *args, **kwargs):
        raise _Interrupted()

    recorded = generation_evidence.SuiteSlots.measurement_removed
    _mutation_tool_names(monkeypatch, "test_flaky_a")
    monkeypatch.setattr(generation_evidence.SuiteSlots, "measurement_removed", interrupted)
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    with pytest.raises(_Interrupted):
        harness._run(root, target, tests_dir, suites_dir, [_PROBED, harness._BETA], monkeypatch,
                     matrices_dir=tmp_path / "matrices", refill_failed_suites=False)
    sdir = suites_dir / "suite_001"
    assert "def test_flaky_a" not in (sdir / "cleaned.py").read_text(encoding="utf-8")
    assert "measurement_removed_source" not in _manifest_slot(suites_dir, 1)
    monkeypatch.setattr(generation_evidence.SuiteSlots, "measurement_removed", recorded)
    _mutation_tool_names(monkeypatch, "test_flaky_b")
    res, calls = harness._run(root, target, tmp_path / "again", suites_dir,
                              [_PROBED, harness._BETA], monkeypatch,
                              matrices_dir=tmp_path / "matrices_again",
                              refill_failed_suites=False, reuse_suites=True)
    assert res.status == "ok" and calls["n"] == 0 and res.n_green == 2
    cleaned = (sdir / "cleaned.py").read_text(encoding="utf-8")
    assert "def test_flaky_b" not in cleaned and "def test_keep" in cleaned
    slot = _manifest_slot(suites_dir, 1)
    assert (slot["state"], slot["classification"]) == ("unavailable", "reused_source_changed")
    assert "measurement_removed_source" not in slot
    _sdir, meta = _slot(suites_dir, 1)
    first, second = meta["cleanup"]["measurement_removals"]
    assert first["removed"][0]["test"].endswith("test_flaky_a")
    assert second["removed"][0]["test"].endswith("test_flaky_b")


def test_a_reused_pool_loads_the_suite_a_removal_changed(tmp_path, monkeypatch):
    """A reuse of a starter pool in which a slot lost a test loads the
    changed suite without a model request. The slot is not marked as a
    changed reuse, and its construction is a pruned node whose parent is the
    accepted attempt's."""
    from testuary import generation_evidence

    _mutation_tool_names(monkeypatch, "test_flaky_a")
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    harness._run(root, target, tests_dir, suites_dir, [_ALPHA_FLAKY, _BETA_FLAKY], monkeypatch,
                 matrices_dir=tmp_path / "matrices", refill_failed_suites=False)
    res, calls = harness._run(root, target, tmp_path / "again", suites_dir,
                              [_ALPHA_FLAKY, _BETA_FLAKY], monkeypatch,
                              matrices_dir=tmp_path / "matrices_again",
                              refill_failed_suites=False, reuse_suites=True)
    assert res.status == "ok" and calls["n"] == 0 and res.n_green == 2
    slot = _manifest_slot(suites_dir, 1)
    assert slot["state"] == "accepted" and "measurement_removed_source" in slot
    manifest = json.loads((suites_dir / "_generation.json").read_text(encoding="utf-8"))
    slots = generation_evidence.SuiteSlots(suites_dir, 2, previous=manifest)
    changed = generation_evidence.read_cleaned_source(slot["measurement_removed_source"],
                                                      suites_dir)
    node = slots.construction_for(1, changed, generation_evidence.construction_scope("starter"))
    assert node["operation"] == "pruned"
    assert node["parents"] == [slot["attempts"][-1]["construction"]]


def test_a_starter_candidate_loses_the_test_its_run_under_the_delivered_name_fails(
        tmp_path, monkeypatch):
    """A starter suite measured under the delivered name, whose coverage run
    there fails one test after its passing normal run, loses that test and is
    delivered without it. Its slot keeps the suite, and the phase record and
    its validity derivations name the removal, which the reuse check accepts."""
    from langadapter.suites import TestId
    from testuary import consolidation, languages
    from testuary import evidence as validity
    from testuary.languages.generic import TestFailure

    monkeypatch.setattr(consolidation, "consolidate_phase_end",
                        lambda *a, **k: consolidation.PhaseConsolidation(detail="no merge"))
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    backend = languages.get_backend("python")
    delivered = backend.target_test_path(str(tests_dir), str(target))
    backend_type = type(backend)
    real = backend_type.run_test_files

    def run_test_files(self, paths, source_dir, *args, **kwargs):
        result = real(self, paths, source_dir, *args, **kwargs)
        path = Path(paths[0])
        if not (kwargs.get("measure_coverage_of") and path.name == delivered.name
                and path.parent != delivered.parent
                and "def test_flaky_a(" in path.read_text(encoding="utf-8")):
            return result
        return result._replace(
            ok=False, returncode=1, failed=1, line_coverage=None, branch_coverage=None,
            failures=(TestFailure("test_flaky_a", "failed", "fails under the tool", ""),),
            stopping_tests=(TestId(path.name, "test_flaky_a",
                                   text=f"{path.name}::test_flaky_a"),))

    monkeypatch.setattr(backend_type, "run_test_files", run_test_files)
    matrices = tmp_path / "matrices"
    res, _calls = harness._run(root, target, tests_dir, suites_dir, [_PROBED], monkeypatch,
                               matrices_dir=matrices, refill_failed_suites=False)
    assert res.status == "ok" and res.baseline_from_single_suite and res.selected_tests == 2
    shipped = Path(res.baseline_file).read_text(encoding="utf-8")
    assert "def test_flaky_a" not in shipped and "def test_keep" in shipped
    sdir, _meta = _slot(suites_dir)
    assert "def test_flaky_a" in (sdir / "cleaned.py").read_text(encoding="utf-8")
    record = json.loads((matrices / "consolidation_record.json").read_text(encoding="utf-8"))
    [entry] = record["measurement_removals"]
    [removed] = entry.pop("removed")
    assert entry == {"place": "starter.candidate", "measurement": "coverage", "round": 1}
    assert removed["test"].endswith("test_flaky_a") and removed["outcome"] == "failed"
    [derivation] = json.loads((matrices / "validity_derivations.json").read_text(
        encoding="utf-8"))["records"]
    assert (derivation["operation"], derivation["stage"]) == (
        "measurement_remove", "starter.candidate")
    assert derivation["child_source_sha256"] == record["selected_source_sha256"]
    assert not list(matrices.glob("candidate_*_removed*"))
    assert validity.verify_delivered_archive(
        archive_dir=matrices, source=shipped,
        identity=backend.generated_test_target_identity(str(target), str(root))) == (True, "")


def test_the_acceptance_under_the_delivered_name_removes_nothing(tmp_path, monkeypatch):
    """The chosen starter suite's acceptance run under the delivered
    filename, whose coverage run there fails a named test, excludes the
    candidate as today and removes no test."""
    from langadapter.suites import TestId
    from testuary import consolidation, languages
    from testuary.languages.generic import TestFailure

    monkeypatch.setattr(consolidation, "consolidate_phase_end",
                        lambda *_args, **_kwargs: consolidation.PhaseConsolidation(
                            detail="no merge"))
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    backend = languages.get_backend("python")
    delivered = backend.target_test_path(str(tests_dir), str(target))
    backend_type = type(backend)
    real = backend_type.run_test_files

    def run_test_files(self, paths, source_dir, *args, **kwargs):
        result = real(self, paths, source_dir, *args, **kwargs)
        path = Path(paths[0])
        if not (kwargs.get("measure_coverage_of") and path.resolve() == delivered.resolve()):
            return result
        return result._replace(
            ok=False, returncode=1, failed=1, line_coverage=None, branch_coverage=None,
            failures=(TestFailure("test_flaky_a", "failed", "fails under the tool", ""),),
            stopping_tests=(TestId(path.name, "test_flaky_a",
                                   text=f"{path.name}::test_flaky_a"),))

    monkeypatch.setattr(backend_type, "run_test_files", run_test_files)
    matrices = tmp_path / "matrices"
    res, _calls = harness._run(root, target, tests_dir, suites_dir, [_PROBED], monkeypatch,
                               matrices_dir=matrices, refill_failed_suites=False)
    assert res.status == "failed_phase_selection"
    sdir, meta = _slot(suites_dir)
    assert "def test_flaky_a" in (sdir / "cleaned.py").read_text(encoding="utf-8")
    assert "measurement_removals" not in meta.get("cleanup", {})
    record = json.loads((matrices / "consolidation_record.json").read_text(encoding="utf-8"))
    assert "measurement_removals" not in record
    assert not list(matrices.glob("candidate_*_removed*"))
