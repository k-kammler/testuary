"""Each run of a file, a single run too, has its own directory: r0 for the first,
r1 for the second. tests.csv names the run in its run column, the first run
keeps its checkpoint and evidence records, and no run name reaches the
model."""
from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path

from langchain_core.messages import BaseMessage

import pytest

import calc_language
import test_generic_language as calc
from testuary import models as model_source
from testuary import phase_state
from testuary import project as project_module
from testuary.evidence import phase_origin_path


def test_a_run_has_its_own_dir_named_from_r0(tmp_path, default_run):
    """A run's index is its number, from 0, as its directory names it."""
    assert project_module.run_name(0) == "r0" and project_module.run_name(2) == "r2"
    with pytest.raises(ValueError, match="counts from 0"):
        project_module.run_name(-1)
    assert project_module.run_dir(tmp_path, 0) == tmp_path / "r0"
    assert project_module.phase_artifact_path(tmp_path, "tests") == tmp_path / "r0" / "tests"
    assert project_module.phase_artifact_path(tmp_path, "tests", 1) == tmp_path / "r1" / "tests"
    assert project_module.run_idx_from_artifact_path(tmp_path / "r2") == 2
    assert project_module.run_idx_from_artifact_path(tmp_path / "tests") == 0
    assert phase_origin_path(tmp_path, "coverage") == (
        tmp_path / "r0" / "archive" / "coverage" / "generation" / "record.json")


def test_run_keys_seeds_and_the_run_selection_count_from_0(tmp_path, default_run):
    """phase_state's run keys, the starter seeds and the selected runs use
    the run's number, from 0, as directories and tests.csv do."""
    from types import SimpleNamespace

    from testuary import config, orchestrator

    source = tmp_path / "foo.py"
    source.write_text("x = 1\n", encoding="utf-8")
    phase_state.mark_started(tmp_path / "rank1_foo", "coverage", target_number=1, stem="foo",
                             target_file_path="proj/foo.py", canonical_source_path=source)
    assert list(phase_state.load(tmp_path / "rank1_foo")["coverage"]) == ["0"]
    assert config.run_seed(37, 0, 1) == 38 and config.run_seed(37, 1, 1) == 1038
    assert orchestrator._selected_run_indices(SimpleNamespace(run=None, runs=2)) == (0, 1)
    assert orchestrator._selected_run_indices(SimpleNamespace(run=3, runs=1)) == (3,)


class _Recording(calc._ScriptedSource):
    """The scripted Calc model, recording each starter seed and every text
    the agents receive."""

    def __init__(self):
        self.seeds = []
        self.texts = []

    def generate_suite(self, messages, *, temperature, seed, timeout):
        self.seeds.append(seed)
        self.texts.extend(str(message.get("content", "")) for message in messages)
        return super().generate_suite(messages, temperature=temperature, seed=seed,
                                      timeout=timeout)

    def agent_chat_model(self, phase):
        texts = self.texts

        class Model(calc._AgentModel):
            def bind_tools(self, tools, *, tool_choice=None, **kwargs):
                return Model(tools=tuple(getattr(t, "name", "") for t in tools))

            def _generate(self, messages, stop=None, run_manager=None, **kwargs):
                texts.extend(str(m.content) for m in messages if isinstance(m, BaseMessage))
                return super()._generate(messages, stop, run_manager, **kwargs)

        return Model()


def _run(tmp_path, monkeypatch, *options, exit_code=0):
    """One whole Calc run with the scripted model, ending with ``exit_code``.
    Returns the model's record and the run directory."""
    from testuary import __main__ as cli

    source = _Recording()
    calc_language.register()
    try:
        project = tmp_path / "proj"
        project.mkdir(exist_ok=True)
        (project / "calc.calc").write_text(calc.SOURCE, encoding="utf-8")
        # The user directory holds the alias, the project's settings file names
        # its language, and the run starts in the project.
        (tmp_path / "models.ini").write_text(
            "[scripted]\nprovider = ollama\nmodel = scripted\n", encoding="utf-8")
        (project / ".testuary").mkdir(exist_ok=True)
        (project / ".testuary" / "project.ini").write_text("[project]\nlanguage = calc\n",
                                                           encoding="utf-8")
        (tmp_path / "config.ini").write_text(
            "[project]\n[starter]\nsuite_count = 3\nattempts_per_suite = 2\n"
            "[coverage]\ngoal = 90\ngoal_metric = line\n"
            "[mutation]\ngoal = 80\n", encoding="utf-8")
        monkeypatch.chdir(project)
        monkeypatch.setenv("TESTUARY_HOME", str(tmp_path))
        monkeypatch.setenv("TESTUARY_TIMING_RUN_COUNT", "1")
        monkeypatch.setattr(sys, "argv", [
            "testuary", "--config", str(tmp_path / "config.ini"), "--model", "scripted",
            "--project-name", "calcproj", "--source-files", "calc.calc",
            "--output-dir", str(tmp_path / "out"), "--no-color", *options])
        with model_source.installed(source):
            assert cli.run_command() == exit_code
    finally:
        calc_language.unregister()
    return source, tmp_path / "out" / "results"


def _rows(results: Path):
    with open(results / "tests.csv", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def test_a_single_run_lands_in_r0_and_tests_csv_names_run_0(tmp_path, monkeypatch):
    source, results = _run(tmp_path, monkeypatch)
    (workspace,) = [path for path in results.iterdir() if path.is_dir()]
    run = workspace / "r0"
    assert {"archive", "resolved_settings.json", "summary.json", "tests"} <= {path.name for path in run.iterdir()}
    assert (run / "tests" / "test_calc.calc").is_file()
    assert (run / "archive" / "starter" / "suites").is_dir()
    assert (run / "archive" / "coverage" / "consolidation_record.json").is_file()
    # Nothing of the run outside its directory, but the file's checkpoint record.
    assert sorted(path.name for path in workspace.iterdir()) == ["phase_state.json", "r0", "summary.json"]
    assert not (results / "resolved_settings.json").exists()
    settings = json.loads((run / "resolved_settings.json").read_text(encoding="utf-8"))
    assert settings["sections"]["starter"]["suite_count"] == "3"
    rows = _rows(results)
    assert [row["run"] for row in rows] == ["0"] and "run_idx" not in rows[0]
    # The first run keeps the checkpoint block of a run without repetitions.
    assert phase_state.load(workspace)["coverage"].keys() == {"0"}
    logs = tmp_path / "out" / "logs" / "test" / workspace.name
    assert sorted(path.name for path in logs.iterdir()) == ["r0"]
    # A single run's seeds (seed_base + i for slot i from 1), and no run name
    # in anything the model receives.
    assert source.seeds == [1, 2, 3]
    for text in source.texts:
        assert "r0" not in Path(text).parts and "/r0/" not in text and "\\r0\\" not in text


def test_a_second_run_lands_in_r1_beside_r0(tmp_path, monkeypatch):
    _source, results = _run(tmp_path, monkeypatch, "--runs", "2")
    (workspace,) = [path for path in results.iterdir() if path.is_dir()]
    for run in ("r0", "r1"):
        assert (workspace / run / "tests" / "test_calc.calc").is_file(), run
        assert (workspace / run / "archive" / "starter").is_dir(), run
    assert sorted(row["run"] for row in _rows(results)) == ["0", "1"]
    assert not list(workspace.glob("*_run_*")) and not (workspace / "tests").exists()
    # Each run's finishing log is in its own log directory, and no log
    # beside the results is shared by the runs.
    logs = tmp_path / "out" / "logs" / "test" / workspace.name
    for run in ("r0", "r1"):
        assert (logs / run / "finishing.jsonl").is_file(), run
    assert not list((tmp_path / "out").glob("testgen_finalize_log_*"))


def _suite_bytes(run: Path):
    return {path.relative_to(run).as_posix(): path.read_bytes()
            for path in sorted((run / "tests").rglob("*")) if path.is_file()}


_MEASURED = ("line_coverage", "branch_coverage", "mutation_score",
             "test_count_delivered", "mutation_test_count")


def _timeless(texts):
    """The texts without the test runs' measured durations."""
    return sorted(re.sub(r"elapsed \d+\.\d+s", "elapsed", text) for text in texts)


def test_r0_of_several_runs_is_the_single_run_and_r1_draws_seeds_1000_on(tmp_path, monkeypatch):
    """Run r0 receives exactly the model input of a single run, sends its
    seeds and delivers its suite and measurements. Run r1 receives the same
    texts and sends seeds 1000 further on."""
    import shutil

    single, results = _run(tmp_path, monkeypatch)
    (workspace,) = [path for path in results.iterdir() if path.is_dir()]
    single_suite = _suite_bytes(workspace / "r0")
    (single_row,) = _rows(results)
    shutil.rmtree(tmp_path / "out")

    several, results = _run(tmp_path, monkeypatch, "--runs", "2")
    assert sorted(several.seeds) == [1, 2, 3, 1001, 1002, 1003]
    assert _timeless(several.texts) == _timeless(single.texts * 2)
    assert (_suite_bytes(workspace / "r0")["tests/test_calc.calc"]
            == single_suite["tests/test_calc.calc"])
    rows = {row["run"]: row for row in _rows(results)}
    assert {name: rows["0"][name] for name in _MEASURED} == {
        name: single_row[name] for name in _MEASURED}
    assert phase_state.load(workspace)["coverage"].keys() == {"0", "1"}


def test_each_run_and_the_runs_of_a_file_are_summarized_as_data(tmp_path, monkeypatch):
    _source, results = _run(tmp_path, monkeypatch, "--runs", "2")
    (workspace,) = [path for path in results.iterdir() if path.is_dir()]
    rows = {row["run"]: row for row in _rows(results)}
    runs = []
    for k in (0, 1):
        summary = json.loads((workspace / f"r{k}" / "summary.json").read_text(encoding="utf-8"))
        assert summary["run"] == f"r{k}" and summary["file"] == rows[str(k)]["file_path"]
        assert summary["fully_measured"] is True
        assert summary["line_coverage"] == float(rows[str(k)]["line_coverage"])
        assert summary["mutation_score"] == float(rows[str(k)]["mutation_score"])
        # Calc measures no branch coverage.
        assert summary["branch_coverage"] is None
        assert summary["coverage_status"] == rows[str(k)]["coverage_status"]
        runs.append(summary)
    across = json.loads((workspace / "summary.json").read_text(encoding="utf-8"))
    assert across["runs"] == ["r0", "r1"] and across["fully_measured_runs"] == 2
    lines = [run["line_coverage"] for run in runs]
    assert across["line_coverage"] == {"runs": 2, "mean": sum(lines) / 2,
                                       "minimum": min(lines), "maximum": max(lines)}
    assert across["branch_coverage"] == {"runs": 0, "mean": None, "minimum": None, "maximum": None}
    project = json.loads((results / "summary.json").read_text(encoding="utf-8"))
    assert project["files"] == [across]
    assert project["values"]["line_coverage"]["count"] == 1


def test_each_runs_row_holds_its_own_time_and_the_shared_work_apart(tmp_path, monkeypatch):
    """A run's row reports the time of that run's own work. The work
    before the first run, shared by all runs, is the file-level figure, the
    same in every row of the file and added to no run's time."""
    _source, results = _run(tmp_path, monkeypatch, "--runs", "2")
    (workspace,) = [path for path in results.iterdir() if path.is_dir()]
    rows = {row["run"]: row for row in _rows(results)}
    own = [phase_state.target_seconds(workspace, run_idx=k) for k in (0, 1)]
    assert all(seconds and seconds > 0 for seconds in own)
    for k in (0, 1):
        assert float(rows[str(k)]["duration_s_target"]) == own[k]
    shared = rows["0"]["duration_s_file_shared"]
    assert shared != "" and shared == rows["1"]["duration_s_file_shared"]
    total = phase_state.output_seconds(workspace, phase_state.SEGMENT_FILE)
    assert float(shared) + own[0] + own[1] == pytest.approx(total, abs=0.002)


def test_for_one_run_the_shared_work_and_r0_add_up_to_the_files_time(tmp_path, monkeypatch):
    _source, results = _run(tmp_path, monkeypatch)
    (workspace,) = [path for path in results.iterdir() if path.is_dir()]
    (row,) = _rows(results)
    total = phase_state.output_seconds(workspace, phase_state.SEGMENT_FILE)
    assert float(row["duration_s_file_shared"]) + float(row["duration_s_target"]) == pytest.approx(
        total, abs=0.002)
    # The shared work is not counted in r0.
    assert float(row["duration_s_target"]) == phase_state.target_seconds(workspace, run_idx=0)
    assert float(row["duration_s_file_shared"]) > 0


def test_run_k_alone_makes_only_rk(tmp_path, monkeypatch):
    source, results = _run(tmp_path, monkeypatch, "--run", "1")
    (workspace,) = [path for path in results.iterdir() if path.is_dir()]
    assert sorted(path.name for path in workspace.iterdir()) == ["phase_state.json", "r1", "summary.json"]
    assert (workspace / "r1" / "tests" / "test_calc.calc").is_file()
    assert [row["run"] for row in _rows(results)] == ["1"]
    assert phase_state.load(workspace)["coverage"].keys() == {"1"}
    # The seeds run r1 sends among several runs.
    assert source.seeds == [1001, 1002, 1003]


def test_a_resume_with_a_higher_count_adds_only_the_missing_run(tmp_path, monkeypatch):
    _first, results = _run(tmp_path, monkeypatch)
    (workspace,) = [path for path in results.iterdir() if path.is_dir()]
    suite = workspace / "r0" / "tests" / "test_calc.calc"
    kept = suite.read_bytes()
    (kept_row,) = _rows(results)

    added, results = _run(tmp_path, monkeypatch, "--resume", "--runs", "2")
    # Run r0's generation was complete and is kept: only r1 asks the model.
    # (Calc measures no branch coverage, so a resume measures r0's suite
    # again, as it does for a single run.)
    assert added.seeds == [1001, 1002, 1003]
    assert suite.read_bytes() == kept
    rows = {row["run"]: row for row in _rows(results)}
    assert sorted(rows) == ["0", "1"]
    assert {name: rows["0"][name] for name in _MEASURED} == {
        name: kept_row[name] for name in _MEASURED}
    # The resumed phases keep their outcome and duration.
    for name in ("coverage_status", "mutation_status",
                 "coverage_agent_duration_s", "mutation_agent_duration_s"):
        assert rows["0"][name] == kept_row[name], name
    assert (workspace / "r1" / "tests" / "test_calc.calc").is_file()


def test_a_finishing_reuse_is_refused_beyond_the_first_run(tmp_path, monkeypatch, capsys):
    import pytest

    _first, results = _run(tmp_path, monkeypatch)
    before = sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*"))
    for reuse in ("starter,coverage,mutation,finishing", "finishing"):
        capsys.readouterr()
        with pytest.raises(SystemExit) as stopped:
            _run(tmp_path, monkeypatch, "--reuse", reuse, "--runs", "2")
        assert stopped.value.code == 2
        assert "works with the first repetition alone" in capsys.readouterr().err
    assert sorted(path.relative_to(tmp_path).as_posix() for path in tmp_path.rglob("*")) == before


def test_a_starter_reuse_gives_every_run_the_first_runs_starter_suite(tmp_path, monkeypatch):
    _first, results = _run(tmp_path, monkeypatch, "--stop-after-phase", "starter")
    (workspace,) = [path for path in results.iterdir() if path.is_dir()]
    record = (workspace / "r0" / "archive" / "starter" / "consolidation_record.json").read_bytes()

    reused, results = _run(tmp_path, monkeypatch, "--reuse", "starter", "--runs", "2")
    # No starter suite is generated again, the second run holds the first
    # run's starter records, and each run writes its own row.
    assert reused.seeds == []
    second = workspace / "r1"
    assert (second / "archive" / "starter" / "consolidation_record.json").read_bytes() == record
    assert not (second / "archive" / "starter" / "suites").exists()
    rows = {row["run"]: row for row in _rows(results)}
    assert sorted(rows) == ["0", "1"]
    # Both runs end alike. With this scripted Calc model the coverage phase
    # after a starter reuse also fails in a single run, so the outcome itself
    # is not asserted.
    assert rows["0"]["coverage_status"] == rows["1"]["coverage_status"]


def test_a_loops_reuse_measures_the_first_runs_suite_in_every_run(tmp_path, monkeypatch):
    _first, results = _run(tmp_path, monkeypatch)
    (workspace,) = [path for path in results.iterdir() if path.is_dir()]
    suite = (workspace / "r0" / "tests" / "test_calc.calc").read_bytes()

    reused, results = _run(tmp_path, monkeypatch, "--reuse", "coverage,mutation", "--runs", "2")
    # Only finishing runs again: no model request in either run.
    assert reused.seeds == [] and reused.texts == []
    for run in ("r0", "r1"):
        assert (workspace / run / "tests" / "test_calc.calc").read_bytes() == suite, run
    rows = {row["run"]: row for row in _rows(results)}
    assert sorted(rows) == ["0", "1"]
    assert rows["1"]["mutation_score"] == rows["0"]["mutation_score"]


def test_a_reuse_gives_every_run_the_first_runs_suite(tmp_path, monkeypatch):
    """A reused coverage suite is the first run's for every run. Each later
    run gets the first run's tests and the archives of the reused phases, so
    it restores them as the first run does and runs its own mutation phase."""
    _first, results = _run(tmp_path, monkeypatch, "--stop-after-phase", "coverage")
    (workspace,) = [path for path in results.iterdir() if path.is_dir()]
    first = workspace / "r0"
    record = (first / "archive" / "coverage" / "consolidation_record.json").read_bytes()

    reused, results = _run(tmp_path, monkeypatch, "--reuse", "coverage", "--runs", "2")
    assert reused.seeds == []
    second = workspace / "r1"
    assert (second / "archive" / "coverage" / "consolidation_record.json").read_bytes() == record
    assert not (second / "archive" / "starter" / "suites").exists()
    for run in (first, second):
        assert (run / "archive" / "mutation" / "consolidation_record.json").is_file(), run.name
        assert (run / "tests" / "test_calc.calc").is_file(), run.name
    rows = {row["run"]: row for row in _rows(results)}
    assert sorted(rows) == ["0", "1"]
    assert rows["1"]["mutation_status"] == rows["0"]["mutation_status"] == "success"
