"""The results dashboard takes every value from Testuary's result files,
unchanged: each file's runs with the values and phase statuses of their
summaries, the file's summary across its runs, each phase's record and
label, and the end-of-run summary that the interface's job recorded. No file
measures the project's own tests, so none is shown for them."""
from __future__ import annotations

import dataclasses
import io
import json
import os

import pytest

from testuary.web import _jobs, _results

COLUMNS = ["target_number", "run", "file_path", "line_coverage", "branch_coverage", "mutation_score",
           "starter_status", "coverage_status", "mutation_status", "error",
           "finishing_mutation_score_lower_bound", "finishing_untested"]


@pytest.fixture(autouse=True)
def no_machine_settings(monkeypatch):
    for name in list(os.environ):
        if name.startswith("TESTUARY_"):
            monkeypatch.delenv(name)


def _row(**values):
    return ",".join(values.get(column, "") for column in COLUMNS)


def _table(root, rows):
    """tests.csv as Testuary writes it: UTF-8 with a byte order mark,
    Windows line ends."""
    table = "\r\n".join([",".join(COLUMNS), *rows]) + "\r\n"
    (root / "results" / "tests.csv").write_bytes(table.encode("utf-8-sig"))


def _summarize(root, files):
    """The run summaries Testuary writes from tests.csv, by its own writer,
    for each target's directory."""
    from testuary import results as run_summary

    for workspace, target_number in files.items():
        run_summary.write_file_summaries(root / "results", workspace, target_number)
    run_summary.write_project_summary(root / "results")


def _run(root, rows, records=None):
    """A run directory with this tests.csv, these phase records and the run
    summaries Testuary writes for them."""
    (root / "results").mkdir(parents=True)
    (root / "logs").mkdir()
    _table(root, rows)
    for workspace, record in (records or {}).items():
        (root / "results" / workspace).mkdir()
        (root / "results" / workspace / "phase_state.json").write_text(json.dumps(record),
                                                                      encoding="utf-8")
    _summarize(root, {workspace: record["target_number"] for workspace, record in (records or {}).items()})
    return root


def test_a_runs_warnings_show_with_their_plain_names(tmp_path):
    """A run's summary warns of each time limit reached, once per phase and
    setting: the page gets Testuary's sentence, the kind, phase and setting
    by their plain names, the configured value and the count. A damaged
    summary without the list is not known, not one without warnings."""
    from testuary import results as run_summary

    root = _run(tmp_path / "out", [
        _row(target_number="1", run="0", file_path="calc/ops.py", line_coverage="90"),
        _row(target_number="1", run="1", file_path="calc/ops.py", line_coverage="80"),
    ], {"calc_ops": {"target_number": 1, "target_file_path": "calc/ops.py"}})
    first = root / "results" / "calc_ops" / "r0" / run_summary.SUMMARY
    summary = json.loads(first.read_text(encoding="utf-8"))
    summary["warnings"] = [{
        "kind": "time_limit", "phase": "mutation", "setting": "mutation.measurement_timeout_seconds",
        "value": 600, "unit": "s", "count": 2,
        "text": "The mutation measurement reached its time limit (600 s) twice."}]
    first.write_text(json.dumps(summary), encoding="utf-8")
    second = root / "results" / "calc_ops" / "r1" / run_summary.SUMMARY
    summary = json.loads(second.read_text(encoding="utf-8"))
    summary["warnings"] = []
    second.write_text(json.dumps(summary), encoding="utf-8")
    across = root / "results" / "calc_ops" / run_summary.SUMMARY
    summary = json.loads(across.read_text(encoding="utf-8"))
    summary["warning_runs"] = 1
    across.write_text(json.dumps(summary), encoding="utf-8")
    (ops,) = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]["files"]
    (warning,) = ops["runs"][0]["warnings"]
    assert warning == {
        "kind": "time_limit", "kind_label": run_summary.LABELS.get("time_limit", "time_limit"),
        "phase": "mutation", "phase_label": "Mutation phase",
        "setting": "mutation.measurement_timeout_seconds",
        "setting_label": "Measurement timeout seconds", "value": {"value": 600.0, "text": "10 min"},
        "unit": "s", "count": 2, "text": "The mutation measurement reached its time limit (600 s) twice."}
    assert ops["runs"][1]["warnings"] == [] and ops["across"]["warning_runs"] == 1
    # A configured whole number of seconds reads without a decimal.
    assert (_results._amount(20.0, "s")["text"], _results._amount(1.75, "s")["text"]) == ("20 s", "1.8 s")
    # Without the list, which Testuary always writes: not known.
    summary = json.loads(first.read_text(encoding="utf-8"))
    del summary["warnings"]
    first.write_text(json.dumps(summary), encoding="utf-8")
    (ops,) = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]["files"]
    assert ops["runs"][0]["warnings"] is None


def test_a_loop_a_limit_stopped_is_a_warning_with_its_setting(tmp_path):
    """A loop that one of its limits stopped before its goal is a warning of
    its run, as the end-of-run summary prints it: the page names its kind,
    the loop, the setting by its plain name and the configured value."""
    from testuary import results as run_summary

    root = _run(tmp_path / "out", [
        _row(target_number="1", run="0", file_path="calc/ops.py", line_coverage="90"),
    ], {"calc_ops": {"target_number": 1, "target_file_path": "calc/ops.py"}})
    first = root / "results" / "calc_ops" / "r0" / run_summary.SUMMARY
    summary = json.loads(first.read_text(encoding="utf-8"))
    text = ("The mutation phase stopped at its request limit before reaching its goal "
            "(mutation.max_requests_per_phase = 20).")
    summary["warnings"] = [{
        "kind": "loop_limit", "phase": "mutation", "setting": "mutation.max_requests_per_phase",
        "value": 20, "unit": None, "count": 1, "text": text}]
    first.write_text(json.dumps(summary), encoding="utf-8")
    (ops,) = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]["files"]
    assert ops["runs"][0]["warnings"] == [{
        "kind": "loop_limit", "kind_label": "Limit reached before the goal",
        "phase": "mutation", "phase_label": "Mutation phase",
        "setting": "mutation.max_requests_per_phase",
        "setting_label": "Request limit (model requests per phase)",
        "value": {"value": 20.0, "text": "20"}, "unit": None, "count": 1, "text": text}]


def test_a_cut_mutation_measurement_shows_its_lower_bound_never_unknown(tmp_path):
    """A run whose final mutation measurement left mutants untested has no
    score: the page gets its lower bound and untested mutants, the file's
    lower bound and the project's, from Testuary's summaries."""
    _run(tmp_path / "out", [
        _row(target_number="1", run="0", file_path="calc/ops.py", line_coverage="90",
             finishing_mutation_score_lower_bound="40.5", finishing_untested="22"),
    ], {"calc_ops": {"target_number": 1, "target_file_path": "calc/ops.py"}})
    run = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]
    (ops,) = run["files"]
    assert ops["runs"][0]["values"]["mutation_score"] is None
    assert ops["runs"][0]["cut"] == {"lower_bound": {"value": 40.5, "text": "40.5%"}, "untested": 22}
    assert ops["across"]["lower_bound"] == {"value": 40.5, "text": "40.5%"}
    bound = run["project"]["mutation_score"]["lower_bound"]
    assert bound["count"] == 1 and bound["mean"]["text"] == "40.5%"


def test_each_file_and_run_shows_testuarys_summaries(tmp_path):
    from testuary import results as run_summary
    from testuary.presentation.view import phase_label

    root = _run(tmp_path / "out", [
        _row(target_number="1", run="0", file_path="calc/ops.py", line_coverage="96.72131147540983",
             mutation_score="87.71929824561404", starter_status="ok", coverage_status="success"),
        _row(target_number="1", run="1", file_path="calc/ops.py", line_coverage="100.0",
             error="coverage phase ended by stagnation"),
        _row(target_number="2", run="0", file_path="calc/io.py"),
    ], {"calc_ops": {"target_number": 1, "target_file_path": "calc/ops.py", "starter": {
        "0": {"status": "complete"}, "1": {"status": "failed", "failure_kind": "infrastructure"}}},
        "calc_io": {"target_number": 2, "target_file_path": "calc/io.py"}})
    suite = root / "results" / "calc_ops" / "r1" / "tests"
    suite.mkdir()
    test = b"def test_x():\n    pass\n"
    (suite / "test_ops.py").write_bytes(test)
    # A directory Testuary does not name for a run is not one.
    (root / "results" / "calc_ops" / "r01").mkdir()
    run = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]
    assert (run["name"], run["results"], run["summary"]) == ("out", "results", None)
    assert run["run_dir"] == str(root)
    assert run["values"] == [{"id": value, "name": value.replace("_", " ").capitalize()}
                             for value in run_summary.VALUES]
    assert [value["name"] for value in run["values"]] == [
        "Line coverage", "Branch coverage", "Mutation score"]
    ops, other = run["files"]
    assert (ops["name"], ops["workspace"], other["name"]) == ("calc/ops.py", "calc_ops", "calc/io.py")
    first, second = ops["runs"]
    assert (first["name"], first["row"], second["name"], second["row"]) == ("r0", 0, "r1", 1)
    assert first["values"]["line_coverage"] == {"value": 96.72131147540983, "text": "96.7%"}
    assert first["values"]["mutation_score"] == {"value": 87.71929824561404, "text": "87.7%"}
    # A value a run did not measure is not known, never zero.
    assert first["values"]["branch_coverage"] is None
    assert (first["summarized"], first["fully_measured"], second["fully_measured"]) == (True, True, False)
    assert [phase["phase"] for phase in first["phases"]] == ["starter", "coverage", "mutation", "finishing"]
    assert all(phase["label"] == phase_label(phase["phase"]) for phase in first["phases"])
    initial = first["phases"][0]
    assert (initial["status"], initial["record"]) == ("ok", {
        "status": "complete", "status_text": "finished", "failure_kind": None,
        "failure_kind_text": None})
    assert first["phases"][1]["status"] == "success" and first["phases"][3]["status"] is None
    # Each outcome also in Testuary's plain words.
    assert (initial["status_text"], first["phases"][1]["status_text"]) == ("done", "goal reached")
    assert second["phases"][0]["record"] == {"status": "failed", "status_text": "failed",
                                             "failure_kind": "infrastructure",
                                             "failure_kind_text": "a model service or tool problem"}
    # Testuary names each run's suite directory.
    assert first["suite"] is None
    assert second["suite"] == {"suite_dir": "results/calc_ops/r1/tests", "files": [
        {"path": "results/calc_ops/r1/tests/test_ops.py", "size": len(test)}],
        "tests": ["results/calc_ops/r1/tests/test_ops.py"], "setup": []}
    # The file's summary across its runs, as Testuary wrote it.
    written = json.loads((root / "results" / "calc_ops" / "summary.json").read_text(encoding="utf-8"))
    across = ops["across"]
    assert across["fully_measured_runs"] == written["fully_measured_runs"] == 1
    line = across["values"]["line_coverage"]
    assert line["runs"] == 2 and line["mean"]["value"] == written["line_coverage"]["mean"]
    assert (line["minimum"]["text"], line["maximum"]["text"]) == ("96.7%", "100%")
    assert across["values"]["branch_coverage"] == {"runs": 0, "mean": None, "minimum": None,
                                                   "maximum": None}


def test_a_run_without_its_summary_yet_shows_its_phase_records(tmp_path):
    root = _run(tmp_path / "out", [], {"calc_io": {"target_number": 2, "target_file_path": "calc/io.py",
                                                   "starter": {"0": {"status": "failed",
                                                                     "failure_kind": "method"}}}})
    (root / "results" / "calc_io" / "r0").mkdir()
    (item,) = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]["files"]
    assert (item["name"], item["across"]) == ("calc/io.py", None)
    (run,) = item["runs"]
    assert (run["name"], run["row"], run["summarized"], run["fully_measured"]) == ("r0", None, False, None)
    assert run["phases"][0]["record"] == {"status": "failed", "status_text": "failed",
                                          "failure_kind": "method",
                                          "failure_kind_text": "no usable suite within the limits"}
    assert all(value is None for value in run["values"].values())


def _suite(label, line=None, branch=None, mutation=None, *, tests=3, green=True, status="passing", reason=""):
    from testuary import consolidation

    return consolidation.ScoredSuite(label=label, source=f"# {label}\n", line=line, branch=branch,
                                     mutation=mutation, test_count=tests, green=green, status=status,
                                     status_reason=reason)


def _phase_end(archive, phase, candidates, winner, *, axes=("mutation", "branch", "line"), single=False):
    """A phase end's record as Testuary writes it: the census of its
    candidates (record_selection_audit) in its consolidation_record.json
    (persist_phase_end)."""
    from testuary import consolidation

    end = consolidation.PhaseConsolidation()
    consolidation.record_selection_audit(end, candidates, winner, required_axes=axes)
    (archive / phase).mkdir(parents=True)
    consolidation.persist_phase_end(
        matrices_dir=archive / phase, target_file="calc/ops.py", phase=end,
        delivery=consolidation.PhaseEndDelivery(
            winner_label="single_suite" if single else winner.label,
            winner_source_label=winner.label if single else None, delivered_km=None,
            delivered_tests=winner.test_count))


def test_a_reduction_never_built_shows_as_not_built(tmp_path):
    from testuary import consolidation

    root = _run(tmp_path / "out", [_row(target_number="1", run="0", file_path="calc/ops.py")],
                {"calc_ops": {"target_number": 1, "target_file_path": "calc/ops.py"}})
    archive = root / "results" / "calc_ops" / "r0" / "archive"
    absent = consolidation.scored_from_phase(consolidation.PhaseConsolidation(
        detail="the merged suite's per-test evidence is unavailable"))
    merged = _suite("merged", 90.0, 80.0, 70.0)
    _phase_end(archive, "coverage", [absent, merged], merged)

    (item,) = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]["files"][0]["runs"]
    (coverage,) = item["phase_ends"]
    reduced = coverage["candidates"][0]
    assert (reduced["state_text"], reduced["reason"]) == (
        "not built", "the merged suite's per-test evidence is unavailable")


def test_a_reduction_the_coverage_tool_could_not_measure_shows_as_incomplete(tmp_path):
    """A built reduction the phase end excluded because the coverage tool
    could not measure it passes, but is not fully measured, as its record's
    status says."""
    from testuary import consolidation

    root = _run(tmp_path / "out", [_row(target_number="1", run="0", file_path="calc/ops.py")],
                {"calc_ops": {"target_number": 1, "target_file_path": "calc/ops.py"}})
    archive = root / "results" / "calc_ops" / "r0" / "archive"
    excluded = consolidation.scored_from_phase(consolidation.PhaseConsolidation(
        reduced_source="def test_a():\n    assert True\n", reduction_valid=False,
        selection_audit={"reduction_measurement_issue": "reduced_coverage_refresh_failed"}))
    merged = _suite("merged", 90.0, 80.0, 70.0)
    _phase_end(archive, "coverage", [excluded, merged], merged)

    (item,) = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]["files"][0]["runs"]
    (coverage,) = item["phase_ends"]
    reduced = coverage["candidates"][0]
    assert (reduced["state_text"], reduced["reason"]) == (
        "passes, not fully measured", "the coverage tool could not measure the reduced suite")


@pytest.mark.parametrize("derived", [True, False], ids=["derived", "measured"])
def test_a_derived_reduced_mutation_score_is_marked_in_its_record_and_on_the_page(tmp_path, derived):
    """A reduced suite whose mutation score is derived from the merged
    suite's kill matrix is marked so in its record, and the page's candidate
    carries the mark. A measured score has none."""
    from testuary import consolidation

    root = _run(tmp_path / "out", [_row(target_number="1", run="0", file_path="calc/ops.py")],
                {"calc_ops": {"target_number": 1, "target_file_path": "calc/ops.py"}})
    archive = root / "results" / "calc_ops" / "r0" / "archive"
    matrix = {"mutants": [{"id": "m0", "status": "killed"}, {"id": "m1", "status": "survived"}]}
    reduced = consolidation.scored_from_phase(consolidation.PhaseConsolidation(
        reduced_source="def test_a():\n    assert True\n", reduction_valid=True, line=90.0, branch=80.0,
        reduced_km=matrix, derived_reduced_km=matrix if derived else None))
    _phase_end(archive, "coverage", [reduced, _suite("merged", 90.0, 80.0, 40.0)], reduced)

    record = json.loads((archive / "coverage" / "consolidation_record.json").read_text(encoding="utf-8"))
    census = record["selection_audit"]["candidates"][0]
    assert census.get("mutation_derived") == (True if derived else None)
    (item,) = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]["files"][0]["runs"]
    (coverage,) = item["phase_ends"]
    candidate = coverage["candidates"][0]
    assert (candidate["state"], candidate["mutation"]["text"], candidate["mutation_derived"]) == (
        "delivered", "50%", derived)


def test_the_page_labels_a_derived_mutation_score():
    """The phase-end table shows a derived mutation score with its label,
    a measured one and the other scores alone, and a missing one as not
    measured."""
    from conftest import page_value

    stub = "const el = (tag, props = {}, ...children) => ({ text: props.text, children: children.flat() });\n"
    score = {"text": "50%"}
    cases = [({"mutation": score, "mutation_derived": True}, "mutation"),
             ({"mutation": score, "mutation_derived": False}, "mutation"),
             ({"line": score, "mutation_derived": True}, "line"),
             ({"mutation": None, "mutation_derived": False}, "mutation")]
    shown = page_value(["candidateScore"], "[" + ", ".join(
        f"candidateScore({json.dumps(candidate)}, {json.dumps(axis)})" for candidate, axis in cases) + "]",
        before=stub)
    assert shown == [["50%", {"text": " (derived, not measured)", "children": []}], "50%", "50%",
                     {"text": "not measured", "children": []}]


def test_an_untimed_tie_says_the_input_order_decided(tmp_path):
    """Equal best scores whose tie-break could not time the suites are
    decided by the input order, and the page says so, not that the suite
    ran fastest."""
    root = _run(tmp_path / "out", [_row(target_number="1", run="0", file_path="calc/ops.py")],
                {"calc_ops": {"target_number": 1, "target_file_path": "calc/ops.py"}})
    archive = root / "results" / "calc_ops" / "r0" / "archive"
    baseline, agent = _suite("baseline", 90.0, 80.0, 70.0), _suite("agent", 90.0, 80.0, 70.0)
    for suite in (baseline, agent):
        suite.suite_timing = {"status": "unavailable", "error": "no timing benchmark"}
    _phase_end(archive, "coverage", [baseline, agent], baseline)

    (item,) = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]["files"][0]["runs"]
    (coverage,) = item["phase_ends"]
    assert coverage["summary"] == ("Chosen: the suite the coverage phase received, because the "
                                   "best scores are equal and the input order decides.")


def test_each_phase_end_shows_its_candidates_and_why_it_chose_its_suite(tmp_path):
    from testuary import starters
    from testuary.presentation.view import phase_label

    root = _run(tmp_path / "out", [_row(target_number="1", run="0", file_path="calc/ops.py")],
                {"calc_ops": {"target_number": 1, "target_file_path": "calc/ops.py"}})
    archive = root / "results" / "calc_ops" / "r0" / "archive"
    first, second = starters.slot_name(1), starters.slot_name(2)
    reduced = _suite("reduced", 95.0, 85.0, 75.0, tests=4)
    _phase_end(archive, "starter", [_suite(first, 80.0, 70.0, 60.0, reason="measured alone"),
                                    _suite(second, 90.0, 80.0, 70.0),
                                    _suite("merged", 95.0, 85.0, 75.0, tests=6), reduced], reduced)
    (archive / "starter" / "suites").mkdir()
    (archive / "starter" / "suites" / "_generation.json").write_text(json.dumps(
        {"n_generated": 5, "n_unmeasured_coverage": 2, "n_unmeasured_mutation": 1}), encoding="utf-8")
    output = "test_total fails:\n" + "AssertionError: 3 != 4\n" * 20
    baseline = _suite("baseline", 96.0, 86.0, 76.0)
    _phase_end(archive, "coverage", [
        baseline, _suite("agent", green=False, status="failing", reason=output),
        _suite("merged", 97.0, 87.0, None, reason="the mutation tool gave no result")], baseline)
    own = _suite("standalone_1", 98.0, 88.0, 80.0)
    _phase_end(archive, "mutation", [_suite("standalone_0", 96.0, 86.0, 76.0), own], own, single=True)

    (item,) = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]["files"][0]["runs"]
    starter, coverage, mutation = item["phase_ends"]
    assert [end["phase"] for end in item["phase_ends"]] == ["starter", "coverage", "mutation"]
    assert starter["label"] == phase_label("starter")
    assert starter["summary"] == ("Chosen: the combined suite, reduced to the tests it needs, because the best "
                                  "scores are equal and the reduced suite is preferred.")
    assert starter["freed"] == ("3 passing starter suites left without being fully measured (2 that the "
                                "coverage tool could not measure, 1 that the mutation tool could not "
                                "measure). Each was generated again while its attempts lasted, else left out.")
    assert [(c["text"], c["state_text"]) for c in starter["candidates"]] == [
        ("Starter suite 1", "measured, not chosen"), ("Starter suite 2", "measured, not chosen"),
        ("The combined suite", "measured, not chosen"),
        ("The combined suite, reduced to the tests it needs", "chosen")]
    chosen = starter["candidates"][3]
    # A measured candidate's record notes how it was measured: its numbers
    # say why it was not chosen.
    assert starter["candidates"][0]["reason"] is None
    assert (chosen["tests"]["text"], chosen["line"]["text"], chosen["mutation"]["text"]) == ("4", "95%", "75%")
    assert coverage["summary"] == ("Chosen: the suite the coverage phase received, because it has the best "
                                   "score (mutation score, then branch coverage, then line coverage).")
    assert coverage["freed"] is None
    agent, merged = coverage["candidates"][1:]
    # A candidate that is not chosen keeps the reason its record gives, a
    # test's whole output included.
    assert (agent["state_text"], agent["reason"], agent["line"]) == ("does not pass", output.strip(), None)
    assert (merged["state_text"], merged["reason"], merged["mutation"]) == (
        "passes, not fully measured", "the mutation tool gave no result", None)
    # A single suite that won is named by the candidate it was.
    assert mutation["delivered"] == "standalone_1"
    assert mutation["summary"].startswith("Chosen: the mutation phase's suite, because it has the best score")
    assert [c["state"] for c in mutation["candidates"]] == ["measured", "delivered"]


def test_a_phase_end_that_kept_its_suite_or_chose_none_says_so(tmp_path):
    root = _run(tmp_path / "out", [], {"calc_ops": {"target_number": 1, "target_file_path": "calc/ops.py"}})
    archive = root / "results" / "calc_ops" / "r0" / "archive"
    for phase, record in (("coverage", {"delivered_from": "inherited", "winner_source_label": "starter",
                                        "selection_audit": None}),
                          ("mutation", {"delivered_from": "unresolved", "selection_audit": None})):
        (archive / phase).mkdir(parents=True)
        (archive / phase / "consolidation_record.json").write_text(json.dumps(record), encoding="utf-8")
    (item,) = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]["files"][0]["runs"]
    assert [(end["phase"], end["summary"], end["candidates"]) for end in item["phase_ends"]] == [
        ("coverage", "The loop ended with the suite it received, so the phase end kept that suite "
                     "and its measurements.", []),
        ("mutation", "No candidate passed and was fully measured, so the phase end chose no suite.",
         [])]


def test_the_measurements_a_tool_refused_show_the_tests_that_stopped_it(tmp_path):
    """The loops' JSON Lines logs record a measurement a tool refused with the
    tool and the tests that stopped it; each shows as a sentence under its
    loop, in the log's order. Other lines, and a line a stopped process left
    unfinished, show nothing."""
    from testuary.presentation.view import phase_label

    root = _run(tmp_path / "out", [], {"calc_ops": {"target_number": 1, "target_file_path": "calc/ops.py"}})
    (root / "results" / "calc_ops" / "r0").mkdir()
    logs = root / "logs" / "test" / "calc_ops" / "r0"
    logs.mkdir(parents=True)
    timing = {"t_mono_s": 1.5, "clock_domain_id": "process-x"}
    lines = [{"record": "start", "project": "calc", "agent": "mutation"},
             {"record": "console", "level": 1, "message": "measured", **timing},
             {"record": "console", "level": 1, "message": "0 mutants tested", "tool": "mutation",
              "stopping_tests": ["tests/test_ops.py::test_loaded", "tests/test_ops.py::test_twice"], **timing},
             {"record": "console", "level": 1, "message": "failed under the coverage tool",
              "tool": "coverage", "stopping_tests": [], **timing}]
    (logs / "mutation.jsonl").write_text(
        "\n".join(json.dumps(line) for line in lines) + '\n{"record": "console", "stopping_te',
        encoding="utf-8")
    (logs / "coverage.jsonl").write_text(json.dumps(
        {"record": "console", "level": 1, "message": "RED when measured", "tool": "coverage",
         "stopping_tests": ["tests/test_ops.py::test_loaded"], **timing}) + "\n", encoding="utf-8")

    (item,) = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]["files"][0]["runs"]
    assert item["refusals"] == [
        {"phase": "coverage", "label": phase_label("coverage"), "tool": "coverage",
         "tests": ["tests/test_ops.py::test_loaded"],
         "text": "The coverage tool could not measure the suite. The tests that stopped it: "
                 "tests/test_ops.py::test_loaded."},
        {"phase": "mutation", "label": phase_label("mutation"), "tool": "mutation",
         "tests": ["tests/test_ops.py::test_loaded", "tests/test_ops.py::test_twice"],
         "text": "The mutation tool could not measure the suite. The tests that stopped it: "
                 "tests/test_ops.py::test_loaded, tests/test_ops.py::test_twice."},
        {"phase": "mutation", "label": phase_label("mutation"), "tool": "coverage", "tests": [],
         "text": "The coverage tool could not measure the suite and named no test."}]


def test_a_demonstration_shows_its_recorded_values(tmp_path):
    root = tmp_path / "demo"
    _run(root / "run", [_row(target_number="1", run="0", file_path="t/stop.py", line_coverage="100.0")],
         {"t_stop": {"target_number": 1, "target_file_path": "t/stop.py"}})
    (root / "DEMO.json").write_text(json.dumps({"statement": "DEMO"}), encoding="utf-8")
    run = _results.summary(_results.locate(tmp_path, "demo"))["runs"][0]
    assert run["name"] == "demo" and run["results"] == "run/results"
    (item,) = run["files"][0]["runs"]
    assert item["values"]["line_coverage"]["text"] == "100%"


def _completion_file(path, results_directory, *, completion=True, events=1):
    from testuary.presentation import DisplayEvent, Metric
    from testuary.presentation.terminal import FileCompletion, RunCompletion

    last = DisplayEvent(run_id="r", target="all targets", phase="pipeline", operation="run",
                        state="succeeded", elapsed_s=9.0, message="Run completed", level="run",
                        provenance="synthetic")
    value = RunCompletion((FileCompletion("calc/ops.py", "complete",
                                          metrics=(Metric("Mutation score", 87.7193, "%"),)),),
                          results_directory=str(results_directory))
    lines = [{"schema_version": 2, "testuary": "test"}]
    lines += [{"event": dataclasses.asdict(last)}] * events
    if completion:
        lines.append({"completion": dataclasses.asdict(value)})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")
    return path, value, last


def _console(value, last):
    """The summary as Testuary's presenter writes it for the console."""
    from rich.console import Console

    from testuary.presentation import TerminalPresenter
    from testuary.presentation.terminal import present_completion

    stream = io.StringIO()
    console = Console(file=stream, width=10000, force_terminal=False, color_system=None)
    with TerminalPresenter("plain", console=console) as presenter:
        presenter.view.add(last)
        present_completion(value, presenter=presenter)
    return stream.getvalue().splitlines()


def test_the_summary_is_the_newest_job_record_that_names_the_dir(tmp_path):
    root = _run(tmp_path / "out", [_row(target_number="1", run="0", file_path="calc/ops.py")])
    store = tmp_path / "jobs"
    older, value, last = _completion_file(store / "20261019-010000-aaaa" / "events.jsonl",
                                          root / "results")
    other = _completion_file(store / "20261019-020000-bbbb" / "events.jsonl",
                             tmp_path / "elsewhere" / "results")[0]
    newest = _completion_file(store / "20261019-030000-cccc" / "events.jsonl", root / "results",
                              completion=False)[0]
    run_dir = _results.locate(tmp_path, "out")
    end = _results.summary(run_dir, [newest, other, older])["runs"][0]["summary"]
    assert end == {"job": "20261019-010000-aaaa", "lines": _console(value, last)}
    assert any("87.7% [demo]" in line for line in end["lines"])
    assert _results.summary(run_dir, [newest, other])["runs"][0]["summary"] is None
    # A file that changes is read again: a run that has since ended.
    _completion_file(newest, root / "results")
    assert _results.summary(run_dir, [newest, older])["runs"][0]["summary"]["job"] == (
        "20261019-030000-cccc")


def test_a_summary_is_found_at_the_end_of_a_long_event_file(tmp_path, monkeypatch):
    root = _run(tmp_path / "out", [])
    path = _completion_file(tmp_path / "job" / "events.jsonl", root / "results", events=3000)[0]
    assert path.stat().st_size > 64 * 1024
    run_dir = _results.locate(tmp_path, "out")
    assert _results.summary(run_dir, [path])["runs"][0]["summary"]["lines"]
    # Only the file's end is read: a record further back is not looked for.
    monkeypatch.setattr(_results, "SUMMARY_TAIL_BYTES", 1024)
    with open(path, "a", encoding="utf-8") as stream:
        stream.write("{}\n" * 400)
    assert _results.summary(run_dir, [path])["runs"][0]["summary"] is None


def test_a_result_file_is_read_again_only_when_it_changed(tmp_path, monkeypatch):
    root = _run(tmp_path / "out", [_row(target_number="1", run="0", file_path="a.py",
                                        line_coverage="50.0")], {"a": {"target_number": 1}})
    run_dir = _results.locate(tmp_path, "out")
    sizes = []
    reader = _results._table_reader

    def counted(stream, size):
        sizes.append(size)
        return reader(stream, size)

    monkeypatch.setattr(_results, "_table_reader", counted)
    first = _results.summary(run_dir)["runs"][0]
    assert _results.summary(run_dir)["runs"][0] == first and len(sizes) == 1
    _table(root, [_row(target_number="1", run="0", file_path="a.py", line_coverage="62.5")])
    _summarize(root, {"a": 1})
    changed = _results.summary(run_dir)["runs"][0]
    assert len(sizes) == 2
    assert changed["files"][0]["runs"][0]["values"]["line_coverage"]["text"] == "62.5%"


def test_the_jobs_event_files_are_listed_newest_first(tmp_path):
    manager = _jobs.JobManager(tmp_path)
    assert manager.event_files() == []
    for name in ("20261019-010000-aaaa", "20261019-030000-cccc", "20261019-020000-bbbb"):
        (manager.store / name).mkdir(parents=True)
    assert [path.parent.name for path in manager.event_files()] == [
        "20261019-030000-cccc", "20261019-020000-bbbb", "20261019-010000-aaaa"]
    assert {path.name for path in manager.event_files()} == {"events.jsonl"}


def test_the_interface_shows_the_summary_its_job_recorded(serve, tmp_path):
    app, server, client = serve(tmp_path / "work")
    root = _run(app.base_dir / "out", [_row(target_number="1", run="0", file_path="calc/ops.py",
                                         mutation_score="87.71929824561404")],
                {"calc_ops": {"target_number": 1}})
    _completion_file(app.jobs.store / "20261019-010000-aaaa" / "events.jsonl", root / "results")
    status, view = client.json("GET", "/api/results?run_dir=out")
    assert status == 200
    (run,) = view["runs"]
    assert run["summary"]["job"] == "20261019-010000-aaaa" and run["summary"]["lines"]
    assert run["files"][0]["runs"][0]["values"]["mutation_score"]["text"] == "87.7%"


def test_the_details_and_the_project_come_from_testuarys_summaries(tmp_path):
    """Each run's details, the file's starters and the project's statistics
    are the summary files' own values, named by Testuary's labels."""
    from testuary import results as run_summary

    global COLUMNS
    saved = COLUMNS
    COLUMNS = [*saved, "coverage_logical_requests", "coverage_input_tokens", "coverage_duration_s",
               "starter_suite_line_median", "time_model_s"]
    try:
        _run(tmp_path / "out", [
            _row(target_number="1", run="0", file_path="a.py", line_coverage="80.0", coverage_logical_requests="12",
                 coverage_input_tokens="4000", coverage_duration_s="61.5", starter_suite_line_median="50.0",
                 time_model_s="70"),
            _row(target_number="2", run="0", file_path="b.py", line_coverage="90.0", starter_suite_line_median="70.0")],
            {"a": {"target_number": 1, "target_file_path": "a.py"}, "b": {"target_number": 2, "target_file_path": "b.py"}})
    finally:
        COLUMNS = saved
    run = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]
    first = run["files"][0]["runs"][0]
    coverage = first["details"]["phases"]["coverage"]
    assert coverage["logical_requests"] == {"value": 12.0, "text": "12"}
    assert coverage["input_tokens"]["text"] == "4,000" and coverage["seconds"]["text"] == "1 min 2 s"
    assert first["details"]["starter"]["line_coverage"]["median"]["text"] == "50%"
    assert first["details"]["time"]["model_requests"]["text"] == "1 min 10 s"
    assert run["labels"] == dict(run_summary.LABELS)
    assert run["files"][0]["across"]["starter_median"]["line_coverage"]["text"] == "50%"
    line = run["project"]["line_coverage"]
    assert (line["delivered"]["count"], line["delivered"]["mean"]["text"]) == (2, "85%")
    assert [value["text"] for value in line["delivered"]["values"]] == ["80%", "90%"]
    assert line["starter"]["median"]["text"] == "60%"


def test_the_end_of_run_summary_is_read_from_the_run_dir(tmp_path):
    from testuary.presentation import Metric
    from testuary.presentation.terminal import FileCompletion, RunCompletion, write_completion

    root = _run(tmp_path / "out", [_row(target_number="1", run="0", file_path="calc/ops.py")])
    write_completion(RunCompletion(files=(FileCompletion(
        "calc/ops.py", "complete", metrics=(Metric("Mutation score", 87.7193, "%"),)),),
        results_directory=str(root / "results")), provenance="synthetic")
    end = _results.summary(_results.locate(tmp_path, "out"))["runs"][0]["summary"]
    assert end["job"] is None and any("87.7% [demo]" in line for line in end["lines"])
