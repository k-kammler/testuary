"""Tests for the ``AgentLogger`` and ``restore_best_green`` lifecycles and
``regression_revert_decision`` in ``testuary.agent._helpers``.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from agent_log import console_records
from testuary.agent import _helpers as ah


# ---------------------------------------------------------------------------
# AgentLogger
# ---------------------------------------------------------------------------

def test_agent_logger_init_clears_state() -> None:
    """``init()`` clears buffered console logs and analyses."""
    lg = ah.AgentLogger("proj", agent_name="profiling")
    lg._console_logs.append({"x": 1})
    lg._analyses.append({"y": 2})
    lg.init()
    assert lg._console_logs == []
    assert lg._analyses == []
    assert lg._initialized is True


def test_agent_logger_log_appends_to_buffer(capsys) -> None:
    """Each ``log(msg)`` call records its level and message and prints it."""
    lg = ah.AgentLogger("proj")
    lg.init()
    lg.log("hello world", level=1)
    assert len(lg._console_logs) == 1
    entry = lg._console_logs[0]
    assert entry["message"] == "hello world"
    assert entry["level"] == 1
    assert "timestamp" not in entry
    assert isinstance(entry["t_mono_s"], float)
    assert entry["clock_domain_id"]
    captured = capsys.readouterr()
    assert "hello world" in captured.out


def _lines(path) -> list:
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def test_a_phase_log_s_start_record_holds_the_phase_s_settings(tmp_path: Path) -> None:
    """The settings a phase gives its logger before init (its model and loop
    attempts) are in the log's start record."""
    phase_log = ah.PhaseLogger("coverage")
    phase_log._config = {"model": "example", "max_loop_attempts": 3}
    phase_log.init("proj", str(tmp_path), "log_${project_name}.jsonl")
    (start,) = [row for row in _lines(tmp_path / "log_proj.jsonl") if row["record"] == "start"]
    assert start["config"] == {"model": "example", "max_loop_attempts": 3}


def test_agent_logger_debug_saves_without_printing(capsys, tmp_path: Path) -> None:
    """``debug(msg)`` goes to the saved timeline tagged ``debug`` but is not
    printed."""
    lg = ah.AgentLogger("proj")
    lg.init()
    path = lg.enable_streaming(str(tmp_path), "log.jsonl")
    lg.log("on console")
    lg.debug("verbose detail", level=2)
    captured = capsys.readouterr()
    assert "on console" in captured.out
    assert "verbose detail" not in captured.out
    entries = console_records(path)
    assert [e["message"] for e in entries] == ["on console", "verbose detail"]
    assert entries[0].get("debug") is None
    assert entries[1]["debug"] is True
    assert entries[1]["level"] == 2


def test_agent_logger_enable_streaming_writes_initial_file(
    tmp_path: Path,
) -> None:
    """``enable_streaming`` writes the log file at once, so a killed process
    still leaves one: its start record."""
    lg = ah.AgentLogger("proj", agent_name="test", config={"k": 1})
    lg.init()
    path = lg.enable_streaming(str(tmp_path), "log_${project_name}.jsonl")
    [start] = _lines(path)
    assert list(start) == ["record", "project", "agent", "clock_domain_id", "start_t_mono_s", "config"]
    assert (start["record"], start["project"], start["agent"], start["config"]) == (
        "start", "proj", "test", {"k": 1})
    assert isinstance(start["start_t_mono_s"], float)
    assert start["clock_domain_id"]
    assert console_records(path) == []


def test_agent_logger_streaming_writes_one_line_per_event(tmp_path: Path) -> None:
    """With streaming on, every event is one more line naming its record,
    with the fields the in-memory entry has, in order, and ``save`` ends the log."""
    lg = ah.AgentLogger("proj")
    lg.init()
    path = lg.enable_streaming(str(tmp_path), "log.jsonl")
    lg.log("first")
    lg.debug("second", level=1)
    lg.log_analysis("/abs/foo.py", 5.0)
    lg.save(str(tmp_path), "log.jsonl")
    rows = _lines(path)
    assert [row["record"] for row in rows] == ["start", "console", "console", "analysis", "end"]
    assert [{k: v for k, v in row.items() if k != "record"} for row in rows[1:3]] == lg._console_logs
    assert list(rows[2]) == ["record", "level", "message", "debug", "t_mono_s", "clock_domain_id"]
    assert {k: v for k, v in rows[3].items() if k != "record"} == lg._analyses[0]
    assert list(rows[4]) == ["record", "end_t_mono_s", "end_clock_domain_id"]


def test_a_logged_lines_fields_go_into_its_record(tmp_path: Path) -> None:
    """Fields given to a log line, through log_line as Agent.log passes them,
    follow its message in the line's console record."""
    lg = ah.AgentLogger("proj")
    lg.init()
    path = lg.enable_streaming(str(tmp_path), "log.jsonl")
    ah.log_line(lg, "refused", 1, tool="mutation", stopping_tests=["t.py::test_a"])
    [console] = [row for row in _lines(path) if row["record"] == "console"]
    assert list(console)[:5] == ["record", "level", "message", "tool", "stopping_tests"]
    assert (console["tool"], console["stopping_tests"]) == ("mutation", ["t.py::test_a"])

def _failing_records(monkeypatch, kind: str) -> None:
    """The records of ``kind`` cannot be written, as on a full disk."""
    emit = ah.JsonLinesHandler.emit

    def failing(handler, record) -> None:
        if record.msg.get("record") == kind:
            handler._on_error(OSError("disk full"))
        else:
            emit(handler, record)
    monkeypatch.setattr(ah.JsonLinesHandler, "emit", failing)


def test_a_failed_whole_write_keeps_the_earlier_log(monkeypatch, capsys, tmp_path: Path) -> None:
    """``save`` writes a log that was not streamed beside it and renames it
    over the path once every line is written, so a failed write leaves the
    file there as it was."""
    target = tmp_path / "log_proj.jsonl"
    target.write_text("earlier log\n", encoding="utf-8")
    _failing_records(monkeypatch, "console")
    lg = ah.AgentLogger("proj")
    lg.init()
    lg.log("first message")
    lg.save(str(tmp_path), "log_${project_name}.jsonl")
    assert target.read_text(encoding="utf-8") == "earlier log\n"
    assert "could not be written: disk full" in capsys.readouterr().out
    assert [p.name for p in tmp_path.iterdir()] == ["log_proj.jsonl"]
    monkeypatch.undo()
    lg.save(str(tmp_path), "log_${project_name}.jsonl")
    assert [row["record"] for row in _lines(target)] == ["start", "console", "end"]
    assert [p.name for p in tmp_path.iterdir()] == ["log_proj.jsonl"]


def test_agent_logger_save_writes_final_dump(tmp_path: Path) -> None:
    """``save()`` writes a log that was not streamed whole and returns ""
    when nothing was buffered."""
    lg = ah.AgentLogger("proj")
    lg.init()
    out = lg.save(str(tmp_path), "log_${project_name}.jsonl")
    assert out == ""
    assert not (tmp_path / "log_proj.jsonl").exists()
    lg.log("first message")
    out = lg.save(str(tmp_path), "log_${project_name}.jsonl")
    assert Path(out).exists()
    assert [row["record"] for row in _lines(out)] == ["start", "console", "end"]
    assert console_records(out)[0]["message"] == "first message"


def test_a_relative_log_dir_is_refused() -> None:
    """A relative directory would put the log under the installed package."""
    lg = ah.AgentLogger("proj")
    lg.init()
    with pytest.raises(ValueError, match="not an absolute path"):
        lg.enable_streaming("logs", "log.jsonl")


def test_agent_logger_log_analysis_writes_entry(tmp_path: Path) -> None:
    """``log_analysis`` records a finished run in the analyses list."""
    lg = ah.AgentLogger("proj")
    lg.init()
    lg.log_analysis(
        file_path="/abs/foo.py", duration_ms=5000.0, messages=None, error=None,
    )
    assert len(lg._analyses) == 1
    entry = lg._analyses[0]
    assert entry["file_name"] == "foo.py"
    assert "result" not in entry
    assert entry["error"] is None


def test_agent_logger_log_analysis_with_error(tmp_path: Path) -> None:
    """``log_analysis`` records the error of a failed run."""
    lg = ah.AgentLogger("proj")
    lg.init()
    lg.log_analysis("/abs/foo.py", 1000.0, messages=None, error="usage_limit_exceeded")
    assert lg._analyses[0]["error"] == "usage_limit_exceeded"


def test_agent_logger_extract_messages_handles_empty_list() -> None:
    """Empty messages produce an empty ordered conversation."""
    lg = ah.AgentLogger("proj")
    out = lg.extract_messages([])
    assert out == []


# ---------------------------------------------------------------------------
# restore_best_green
# ---------------------------------------------------------------------------

def test_restore_best_green_when_current_is_green_no_op(tmp_path: Path) -> None:
    """A green current state is not restored."""
    snap_content = "def test_x(): assert True\n"
    state = ah.BestGreenState(
        snapshot={"content": snap_content, "mutation": 80.0,
                  "coverage": 90.0, "test_count": 5,
                  "snapshot_clock_s": 1.0},
    )
    test_path = tmp_path / "test_foo.py"
    test_path.write_text("def test_x(): assert False\n", encoding="utf-8")
    restored = ah.restore_best_green(state, test_path, current_is_green=True)
    assert restored is False
    assert "False" in test_path.read_text(encoding="utf-8")


def test_restore_best_green_no_snapshot_returns_false(tmp_path: Path) -> None:
    """Without a snapshot there is nothing to restore."""
    state = ah.BestGreenState()
    test_path = tmp_path / "test_foo.py"
    test_path.write_text("anything", encoding="utf-8")
    assert ah.restore_best_green(state, test_path, current_is_green=False) is False


def test_restore_best_green_reverts_file_when_red(tmp_path: Path) -> None:
    """A red state with a differing snapshot reverts the file and sets
    ``restored_from_green``."""
    snap_content = "def test_one():\n    assert True\n"
    state = ah.BestGreenState(
        snapshot={"content": snap_content, "mutation": 80.0,
                  "coverage": 90.0, "test_count": 1,
                  "snapshot_clock_s": 1.0},
    )
    test_path = tmp_path / "test_foo.py"
    test_path.write_text("def broken(:\n    syntax error\n", encoding="utf-8")
    restored = ah.restore_best_green(state, test_path, current_is_green=False)
    assert restored is True
    assert test_path.read_text(encoding="utf-8") == snap_content
    assert state.restored_from_green is True


def test_restore_best_green_same_content_no_restore(tmp_path: Path) -> None:
    """Content that already matches the snapshot is not restored."""
    snap_content = "def test_one(): assert True\n"
    state = ah.BestGreenState(
        snapshot={"content": snap_content, "mutation": 80.0,
                  "coverage": 90.0, "test_count": 1,
                  "snapshot_clock_s": 1.0},
    )
    test_path = tmp_path / "test_foo.py"
    test_path.write_text(snap_content, encoding="utf-8")
    assert ah.restore_best_green(state, test_path, current_is_green=False) is False


def test_restore_best_green_missing_file_returns_false(tmp_path: Path) -> None:
    """A deleted test file returns False rather than raising."""
    snap_content = "x"
    state = ah.BestGreenState(
        snapshot={"content": snap_content, "mutation": 0.0,
                  "coverage": 0.0, "test_count": 0,
                  "snapshot_clock_s": 1.0},
    )
    assert ah.restore_best_green(
        state, tmp_path / "missing.py", current_is_green=False,
    ) is False


def test_restore_best_green_calls_log_fn_on_success(tmp_path: Path) -> None:
    """A restore logs a message naming the snapshot's metrics."""
    captured = []
    state = ah.BestGreenState(
        snapshot={"content": "ok\n", "mutation": 80.0,
                  "coverage": 90.0, "test_count": 3,
                  "snapshot_clock_s": 1.0},
    )
    test_path = tmp_path / "test_foo.py"
    test_path.write_text("different\n", encoding="utf-8")
    ah.restore_best_green(
        state, test_path, current_is_green=False,
        log_fn=lambda msg, level=0: captured.append((msg, level)),
    )
    assert captured
    msg = captured[0][0]
    assert "restore" in msg.lower()
    assert "80" in msg or "mutation" in msg
    assert "not the selected best-green snapshot" in msg


# ---------------------------------------------------------------------------
# regression_revert_decision -- the shared N_restore gate
# ---------------------------------------------------------------------------


def test_regression_revert_decision_gate() -> None:
    g = ah.regression_revert_decision
    # A non-regression breaks the streak.
    assert g(regression_count=5, is_regression=False, n_restore=1) == (False, 0)
    assert g(regression_count=0, is_regression=False, n_restore=2) == (False, 0)
    # Disabled (<= 0): never reverts, count left as-is.
    assert g(regression_count=3, is_regression=True, n_restore=0) == (False, 3)
    assert g(regression_count=3, is_regression=True, n_restore=-1) == (False, 3)
    # n = 1: revert on every regression, reset.
    assert g(regression_count=0, is_regression=True, n_restore=1) == (True, 0)
    # n = 2: the first regression is tolerated, the second reverts and resets.
    assert g(regression_count=0, is_regression=True, n_restore=2) == (False, 1)
    assert g(regression_count=1, is_regression=True, n_restore=2) == (True, 0)
