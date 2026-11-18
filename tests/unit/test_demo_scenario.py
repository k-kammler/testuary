"""The demonstration's scenario: its file format and the playback oracle that
answers only what the scenario holds."""
from __future__ import annotations

import json

import pytest

from testuary.demo import scenario as sc


def _entry(kind, key, answer, segment="starter", value_class="real",
           rule="recorded-kill-matrix-v1: x", parts=None):
    return sc.Entry(segment=segment, kind=kind, key=key, answer=answer,
                    value_class=value_class, source="src", rule=rule,
                    parts=parts or {})


def test_equal_keys_are_answered_in_recorded_order():
    oracle = sc.PlaybackOracle(sc.Scenario(entries=[
        _entry("backend.run", {"k": 1}, "first"),
        _entry("backend.run", {"k": 1}, "second"),
    ]))
    call = sc.Call("backend.run", {"k": 1})
    assert oracle.answer(call) == "first"
    assert oracle.answer(call) == "second"
    with pytest.raises(sc.ScenarioStop, match="holds no answer"):
        oracle.answer(call)


def test_a_call_the_scenario_does_not_hold_stops_the_run():
    oracle = sc.PlaybackOracle(sc.Scenario(entries=[
        _entry("backend.run", {"k": 1}, "answer"),
    ]))
    with pytest.raises(sc.ScenarioStop) as stop:
        oracle.answer(sc.Call("backend.run", {"k": 2}))
    assert "backend.run" in str(stop.value)
    assert "regenerated" in stop.value.reason


def test_the_stop_is_not_an_exception_phase_code_would_catch():
    assert issubclass(sc.ScenarioStop, BaseException)
    assert not issubclass(sc.ScenarioStop, Exception)


def test_answers_are_matched_within_their_segment():
    oracle = sc.PlaybackOracle(sc.Scenario(entries=[
        _entry("backend.run", {"k": 1}, "coverage answer", segment="coverage"),
    ]))
    with pytest.raises(sc.ScenarioStop):
        oracle.answer(sc.Call("backend.run", {"k": 1}))
    oracle2 = sc.PlaybackOracle(sc.Scenario(entries=[
        _entry("backend.run", {"k": 1}, "coverage answer", segment="coverage"),
    ]))
    oracle2.enter_segment("coverage")
    assert oracle2.answer(sc.Call("backend.run", {"k": 1})) == "coverage answer"


def test_a_segment_that_ends_with_unused_answers_stops_the_run():
    oracle = sc.PlaybackOracle(sc.Scenario(entries=[
        _entry("backend.run", {"k": 1}, "unused"),
    ]))
    with pytest.raises(sc.ScenarioStop, match="never asked for"):
        oracle.enter_segment("coverage")


def test_the_last_segment_is_checked_at_the_end():
    oracle = sc.PlaybackOracle(sc.Scenario(entries=[
        _entry("backend.run", {"k": 1}, "unused", segment="mutation"),
    ]))
    oracle.enter_segment("coverage")
    oracle.enter_segment("mutation")
    with pytest.raises(sc.ScenarioStop, match="never asked for"):
        oracle.finish()


def test_a_segment_the_run_passes_over_is_checked():
    """A run that skips a phase leaves that phase's answers unused, which
    stops it as a segment it left does."""
    oracle = sc.PlaybackOracle(sc.Scenario(entries=[
        _entry("backend.run", {"k": 1}, "unused", segment="coverage"),
    ]))
    with pytest.raises(sc.ScenarioStop, match="never asked for"):
        oracle.enter_segment("mutation")


def test_a_segment_after_the_last_one_entered_is_checked_at_the_end():
    oracle = sc.PlaybackOracle(sc.Scenario(entries=[
        _entry("backend.run", {"k": 1}, "unused", segment="mutation"),
    ]))
    oracle.enter_segment("coverage")
    with pytest.raises(sc.ScenarioStop, match="never asked for"):
        oracle.finish()


def test_the_run_cannot_go_back_to_an_earlier_segment():
    oracle = sc.PlaybackOracle(sc.Scenario(entries=[]))
    oracle.enter_segment("coverage")
    with pytest.raises(sc.ScenarioStop, match="earlier phase"):
        oracle.enter_segment("starter")


def test_every_answer_reaches_the_ledger_callback():
    seen = []
    oracle = sc.PlaybackOracle(
        sc.Scenario(entries=[_entry("backend.run", {"k": 1}, "a")]),
        on_answer=seen.append,
    )
    oracle.answer(sc.Call("backend.run", {"k": 1}))
    assert [entry.kind for entry in seen] == ["backend.run"]


def test_keys_ignore_line_endings_and_non_collected_files(tmp_path):
    lf, crlf = tmp_path / "a", tmp_path / "b"
    lf.mkdir(), crlf.mkdir()
    (lf / "test_x.py").write_bytes(b"def test_a():\n    pass\n")
    (crlf / "test_x.py").write_bytes(b"def test_a():\r\n    pass\r\n")
    # A sidecar the package keeps beside the tests is not measured.
    (crlf / "aggregate_coverage_matrix.json").write_text("{}")
    (crlf / ".best_passing_test_x.py.bak").write_text("old")
    assert sc.tree_files(lf) == sc.tree_files(crlf)


def test_a_kill_matrix_key_ignores_its_capture_time(tmp_path):
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    body = {"mutants": [{"id": "m", "status": "killed"}]}
    first.write_text(json.dumps({**body, "t_mono_s": 1.0, "clock_domain_id": "a"}))
    second.write_text(json.dumps({**body, "t_mono_s": 9.0, "clock_domain_id": "b"}))
    assert sc.kill_matrix_entry(first)[1] == sc.kill_matrix_entry(second)[1]
    first.write_text(json.dumps({"mutants": []}))
    assert sc.kill_matrix_entry(first)[1] != sc.kill_matrix_entry(second)[1]
