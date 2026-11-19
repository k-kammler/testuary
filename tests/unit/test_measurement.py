"""Tests for the metrics helpers without model requests: measurement records,
structural counts and settings handling."""
from __future__ import annotations

from pathlib import Path

import pytest

from agent_fixtures import run_result
from testuary import measurement as metrics
from phase_values import set_phase_value
import json
from testuary import measurement
from testuary.generation_evidence import source_sha256

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


# Dataclass defaults

def test_test_file_metrics_defaults_are_unavailable() -> None:
    """An unobserved source must not look like a measured empty suite."""
    m = metrics.TestFileMetrics(name="test_foo.py")
    assert m.loc is None
    assert m.test_count is None
    assert m.assertion_count is None
    assert m.fixture_count is None
    assert m.parametrize_case_count is None
    assert m.cyclomatic_complexity_sum is None
    assert m.cyclomatic_complexity_max is None


def test_combined_test_metrics_defaults() -> None:
    """Every measurement field defaults to None, so an unset field reads as
    not measured in the CSV instead of as a measured zero."""
    c = metrics.CombinedTestMetrics(target_file_path="src/foo.py")
    assert c.target_complexity_sum is None
    assert c.coverage_final_line is None
    assert c.test_files == []
    assert c.status == "success"     # the only non-None default


# CombinedTestMetrics.to_csv_dict

def test_to_csv_dict_flattens_optional_columns() -> None:
    """All optional fields appear in the dict (even when None) so the
    CSV writer can rely on a stable column set independent of which
    measurements ran."""
    c = metrics.CombinedTestMetrics(
        target_file_path="src/foo.py",
        target_complexity_sum=12,
        coverage_final_line=87.5,
        duration_s=9.4,
    )
    out = c.to_csv_dict()
    assert out["target_complexity_sum"] == 12
    assert out["coverage_final_line"] == 87.5
    assert out["finishing_duration_s"] == 9.4
    assert out["finishing_status"] == "success"
    # Untouched optional fields still present, value None
    assert "lint_smells_target" in out
    assert out["lint_smells_target"] is None


def test_to_csv_dict_skips_a_file_the_backend_does_not_classify() -> None:
    """Only a classified test file fills the standard columns. Another file
    cannot overwrite them."""
    c = metrics.CombinedTestMetrics(target_file_path="src/foo.py")
    c.test_files.append(metrics.TestFileMetrics(
        name="test_foo.py", loc=10, test_count=3, assertion_count=5,
    ))
    c.test_files.append(metrics.TestFileMetrics(
        name="conftest.py", loc=20, test_count=6, assertion_count=9,
    ))
    out = c.to_csv_dict()
    assert out["test_loc_delivered"] == 10
    assert out["test_count_delivered"] == 3
    assert out["assertion_count_delivered"] == 5
    assert "mutation_test_loc" not in out


def test_to_csv_dict_emits_suite_timing_columns() -> None:
    """The full-suite timing fields flatten into the CSV dict. They are
    present (value None) even when unmeasured so the column set is stable,
    and carry the values when set."""
    c = metrics.CombinedTestMetrics(target_file_path="src/foo.py")
    out = c.to_csv_dict()
    for col in (
        "suite_timing_median_ms", "suite_timing_samples_ms",
        "suite_timing_run_count", "suite_timing_warmup_count",
        "suite_timing_requested_cpus", "suite_timing_allowed_cpus",
    ):
        assert col in out and out[col] is None

    c2 = metrics.CombinedTestMetrics(
        target_file_path="src/foo.py",
        suite_timing_median_ms=123.4,
        suite_timing_samples_ms="120.0;123.4;130.0",
        suite_timing_run_count=3,
        suite_timing_warmup_count=1,
    )
    out2 = c2.to_csv_dict()
    assert out2["suite_timing_median_ms"] == 123.4
    assert out2["suite_timing_samples_ms"] == "120.0;123.4;130.0"
    assert out2["suite_timing_run_count"] == 3
    assert out2["suite_timing_warmup_count"] == 1


# Per-phase snapshots of the standard suite

def test_phase_scores_mode_default_is_full(monkeypatch) -> None:
    monkeypatch.delenv(
        "TESTUARY_FINISHING_PHASE_SCORES", raising=False,
    )
    assert metrics.phase_scores_mode() == "full"


def test_phase_scores_mode_env_override_wins(monkeypatch, in_run) -> None:
    """The CLI flag's variable sets the run's setting over the default."""
    monkeypatch.setenv("TESTUARY_FINISHING_PHASE_SCORES", "off")
    in_run()
    assert metrics.phase_scores_mode() == "off"
    monkeypatch.setenv("TESTUARY_FINISHING_PHASE_SCORES", "COVERAGE")
    in_run()
    assert metrics.phase_scores_mode() == "coverage"


def test_finishings_record_names_phase_scores_by_its_record_field(in_run) -> None:
    """Finishing's completion record names [finishing] phase_scores by its
    record field, phase_originals."""
    from testuary import evidence

    in_run(overrides={"api": {"provider": "ollama", "model": "test-model"},
                      "finishing": {"phase_scores": "coverage"}})
    recorded = evidence.finishing_settings()
    assert recorded["phase_originals"] == '"coverage"' and "phase_scores" not in recorded


def test_the_coverage_timeout_seconds_comes_from_the_configuration(
    tmp_path, monkeypatch, in_run,
) -> None:
    """The resolved test_metrics settings carry coverage_timeout_seconds
    from config.ini, so the configured value replaces the default."""
    from testuary import config

    cfg = tmp_path / "metrics.ini"
    cfg.write_text("[finishing]\ncoverage_timeout_seconds = 900\n",
                   encoding="utf-8")
    monkeypatch.setenv("TESTUARY_CONFIG_FILE", str(cfg))
    in_run()
    resolved = config.get_agent_config("test_metrics", parent="test_coverage")
    assert resolved["coverage_timeout_seconds"] == 900
    assert "scaled_coverage_timeout" not in resolved


def test_phase_scores_mode_unknown_falls_back_to_full(monkeypatch) -> None:
    monkeypatch.setenv("TESTUARY_FINISHING_PHASE_SCORES", "bogus")
    assert metrics.phase_scores_mode() == "full"


def test_phase_finishing_csv_columns_shape() -> None:
    cols = metrics.phase_finishing_csv_columns()
    for phase in ("starter", "coverage", "mutation"):
        for kind in ("line", "branch", "mutation_full", "mutation_selective"):
            assert f"{phase}_original_{kind}" in cols
    assert len(cols) == 12


def test_phase_original_path_named_off_canonical_stem() -> None:
    p = metrics.phase_original_path(
        Path("/w/originals"), "test_groupby", ".py", "coverage",
    )
    assert p.name == "test_groupby_coverage.py"
    # The phase joins with "_", never ".": pytest cannot import a dotted stem,
    # so the snapshot would silently not be collected.
    assert "." not in p.stem
    assert p.parent.name == "originals"


def test_to_csv_dict_emits_per_phase_columns() -> None:
    """A populated ``phase_finishing`` maps each snapshot's standard-suite
    coverage and mutation onto the ``{phase}_orig_*`` columns. An absent phase
    leaves its columns present and None so the schema is stable."""
    c = metrics.CombinedTestMetrics(target_file_path="src/foo.py")
    c.phase_finishing["starter"] = metrics.FinishingMetrics(
        line_coverage=55.0, branch_coverage=40.0,
        mutation_score_full=33.0, mutation_score_selective=30.0,
    )
    out = c.to_csv_dict()
    assert out["starter_original_line"] == 55.0
    assert out["starter_original_mutation_full"] == 33.0
    assert out["starter_original_mutation_selective"] == 30.0
    # Absent coverage and mutation snapshots: columns present, value None
    assert out["coverage_original_line"] is None
    assert out["mutation_original_mutation_full"] is None


# _analyze_test_file: AST-based structural counts

def test_analyze_test_file_counts_tests_and_assertions(tmp_path: Path) -> None:
    """``def test_*`` counts as a test and ``assert`` as an assertion. A
    ``pytest.raises`` with-block also counts as one assertion, the only
    assertion-shaped check pytest offers besides ``assert``."""
    f = tmp_path / "test_foo.py"
    f.write_text(
        "import pytest\n"
        "def test_one():\n"
        "    assert 1 == 1\n"
        "    assert True\n"
        "def test_two():\n"
        "    with pytest.raises(ValueError):\n"
        "        raise ValueError\n"
        "def helper():\n"           # not a test_, not counted
        "    return 1\n",
        encoding="utf-8",
    )
    m = metrics._analyze_test_file(f)
    assert m.test_count == 2
    assert m.assertion_count == 3   # 2 asserts and 1 pytest.raises


def test_code_line_count_excludes_blanks_and_comments(tmp_path) -> None:
    """loc is code lines: blank lines and pure comment lines do not
    count, indented comments included. Trailing comments on a code
    line leave the line a code line."""
    text = (
        "# banner\n"
        "\n"
        "def test_a():\n"
        "    # rationale\n"
        "    assert 1 == 1  # trailing\n"
        "\n"
        "    # === section ===\n"
    )
    path = tmp_path / "test_a.py"
    path.write_text(text, encoding="utf-8")
    assert metrics._analyze_test_file(path).loc == 2


def test_analyze_test_file_java_comments_are_not_code(tmp_path: Path) -> None:
    """Java's source-only metrics use its parser, not Python comments."""
    f = tmp_path / "FooTest.java"
    f.write_text(
        "// comment\n"
        "\n"
        "/* comment */\n"
        "class FooTest {}\n",
        encoding="utf-8",
    )
    m = metrics._analyze_test_file(f)
    assert m.test_count == 0
    assert m.loc == 1


def test_analyze_test_file_counts_fixtures(tmp_path: Path) -> None:
    """Both ``@pytest.fixture`` (qualified) and ``@fixture`` (imported)
    count, so a project that imports ``fixture`` is not undercounted."""
    f = tmp_path / "test_foo.py"
    f.write_text(
        "import pytest\n"
        "from pytest import fixture\n"
        "@pytest.fixture\n"
        "def alpha(): return 1\n"
        "@fixture\n"
        "def beta(): return 2\n"
        "@pytest.fixture(scope='session')\n"
        "def gamma(): return 3\n",
        encoding="utf-8",
    )
    m = metrics._analyze_test_file(f)
    assert m.fixture_count == 3


def test_analyze_test_file_counts_parametrize_cases(tmp_path: Path) -> None:
    """A ``@pytest.mark.parametrize("name", [a, b, c])`` decorator
    adds len(case_list) to parametrize_case_count, the row's effective
    test count."""
    f = tmp_path / "test_foo.py"
    f.write_text(
        "import pytest\n"
        "@pytest.mark.parametrize('x', [1, 2, 3])\n"
        "def test_one(x):\n"
        "    assert x > 0\n"
        "@pytest.mark.parametrize('y', [(1, 2), (3, 4)])\n"
        "def test_two(y):\n"
        "    assert y\n",
        encoding="utf-8",
    )
    m = metrics._analyze_test_file(f)
    assert m.parametrize_case_count == 5   # 3 + 2


def test_analyze_test_file_handles_syntax_error_gracefully(
    tmp_path: Path,
) -> None:
    """A test file with a SyntaxError does not crash the measurement. The
    metrics keep the filename and the parse-failure reason."""
    f = tmp_path / "test_broken.py"
    f.write_text("def test_x(:\n    syntax error\n", encoding="utf-8")
    m = metrics._analyze_test_file(f)
    assert m.name == "test_broken.py"
    assert m.test_count is None
    assert m.assertion_count is None
    assert m.source_observation["status"] == "parse_error"
    assert m.source_observation["reason"]


def test_analyze_test_file_missing_file_is_unavailable(tmp_path: Path) -> None:
    """Unreadable evidence retains its reason, not fabricated zeros."""
    m = metrics._analyze_test_file(tmp_path / "missing.py")
    assert m.name == "missing.py"
    assert m.loc is None
    assert m.source_observation["status"] == "infrastructure_failure"


def test_analyze_test_file_loc_excludes_blank_trailing_newline(
    tmp_path: Path,
) -> None:
    """A last line without a trailing newline still counts."""
    f = tmp_path / "no_trailing.py"
    f.write_text("def test_x():\n    assert True", encoding="utf-8")  # no \n
    m = metrics._analyze_test_file(f)
    assert m.loc == 2


# _bool_flag: truthiness of configuration values

def test_bool_flag_recognizes_canonical_true_values() -> None:
    """``true``, ``1`` and ``yes`` map to True, case-insensitively. Anything
    else is False."""
    for v in ("true", "TRUE", "1", "yes", "Yes"):
        assert metrics._bool_flag({"flag_x": v}, "flag_x") is True
    for v in ("false", "0", "no", "off", ""):
        assert metrics._bool_flag({"flag_x": v}, "flag_x") is False


def test_bool_flag_has_no_default_for_a_missing_switch() -> None:
    """config.ini holds every switch, so a section without the one read
    is an error, not a default."""
    with pytest.raises(KeyError):
        metrics._bool_flag({}, "not_there")


# Cyclomatic complexity comes from measurement.compute_file_complexity_stats
# (lizard). Finishing only passes its values on.


# _cov_via_backend: coverage through LanguageBackend.run_test_files. An
# unmeasurable run gives None.

def _fake_backend(monkeypatch, result):
    """Make ``metrics.languages.get_backend`` return a backend whose
    ``run_test_files`` yields ``result`` (None: no backend)."""
    from types import SimpleNamespace

    def _get_backend(_lang):
        if result is None:
            return None
        return SimpleNamespace(run_test_files=lambda *a, **kw: result)

    monkeypatch.setattr(metrics.languages, "get_backend", _get_backend)


def test_finishing_stops_before_its_engine_when_the_suite_has_no_coverage(tmp_path):
    """The delivered suite without line coverage did not run green or its
    coverage run timed out. Finishing stops before any engine work instead
    of leaving a NaN column, and the stop names the suite as this file's
    correctness verifier, both causes and the remedy."""
    suite = tmp_path / "test_foo.py"
    suite.write_text("def test_a(): pass\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match=(r"this file's correctness verifier\) "
                                            r"did not run green.*timed out.*raise its timeout")):
        metrics.run_finishing(
            target_file="foo.py", source_dir=str(tmp_path), language="python",
            merged_test_path=suite, shared_setup_paths=[], scratch_root=tmp_path / "s",
            config={}, standard_coverage=(None, None))


def test_cov_via_backend_empty_targets_returns_none(monkeypatch):
    """No test files give (None, None, False) without consulting a backend."""
    _fake_backend(monkeypatch, run_result(passed=1, line_coverage=88.0, branch_coverage=70.0))
    assert metrics._cov_via_backend("t.py", "/p", [], "combined") == (
        None, None, False,
    )


def test_cov_via_backend_ok_returns_line_branch(monkeypatch):
    """A green coverage run passes line and branch through, with
    timed_out=False."""
    _fake_backend(monkeypatch, run_result(passed=1, line_coverage=91.5, branch_coverage=64.0))
    assert metrics._cov_via_backend(
        "t.py", "/p", ["a.py"], "combined",
    ) == (91.5, 64.0, False)


def test_cov_via_backend_timeout_returns_none_with_timed_out_true(monkeypatch):
    """A timed-out run gives (None, None, True). The flag lets Finishing
    name the timeout as the reason the standard suite has no coverage."""
    _fake_backend(monkeypatch, run_result(ok=False, timed_out=True, elapsed_s=0.1, passed=1))
    assert metrics._cov_via_backend(
        "t.py", "/p", ["a.py"], "combined",
    ) == (None, None, True)


def test_a_timed_out_coverage_run_names_finishings_setting(monkeypatch, tmp_path):
    """The warning and the run's time-limit note name the setting by the
    section a run reads it from, [finishing], not its refused former name."""
    from testuary import results as time_limits

    _fake_backend(monkeypatch, run_result(ok=False, timed_out=True, elapsed_s=0.1, passed=1))
    logged = []
    monkeypatch.setattr(metrics, "log_print", lambda text, *a, **kw: logged.append(text))
    with time_limits.collecting(tmp_path / "r0"):
        assert metrics._cov_via_backend("t.py", "/p", ["a.py"], "combined") == (None, None, True)
    [warning] = time_limits.warnings(tmp_path / "r0")
    setting = "finishing.coverage_timeout_seconds"
    assert (warning["phase"], warning["setting"]) == ("finishing", setting)
    assert any(f"({setting} = " in text for text in logged), logged


def test_cov_via_backend_red_suite_returns_none(monkeypatch):
    """A non-zero exit gives None, since the number would mislead, and
    timed_out=False, since the run completed."""
    _fake_backend(monkeypatch, run_result(ok=False, returncode=1, passed=1,
                                          line_coverage=42.0, branch_coverage=70.0))
    assert metrics._cov_via_backend(
        "t.py", "/p", ["a.py"], "combined",
    ) == (None, None, False)


def test_cov_via_backend_no_backend_returns_none(monkeypatch):
    """An unsupported language (no backend) gives (None, None, False)."""
    _fake_backend(monkeypatch, None)
    assert metrics._cov_via_backend(
        "t.weird", "/p", ["a"], "combined",
    ) == (None, None, False)


# measure_combined, with every subprocess-bound helper mocked

def test_measure_combined_assembles_full_result(monkeypatch, tmp_path) -> None:
    """On a fake project layout, the target's loc and complexity, the
    per-test-file metrics and the combined coverage all land on the result."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    target = src_dir / "foo.py"
    target.write_text(
        "def f(x):\n    return x + 1\ndef g(y):\n    return y\n",
        encoding="utf-8",
    )
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    test_file = test_dir / "test_foo.py"
    test_file.write_text(
        "def test_one():\n    assert 1 == 1\n"
        "def test_two():\n    assert 2 == 2\n",
        encoding="utf-8",
    )
    # Mock the lizard-bound complexity helper
    monkeypatch.setattr(
        metrics, "compute_file_complexity_stats",
        lambda path: (5, 3),
    )
    monkeypatch.setattr(
        metrics, "_cov_via_backend",
        lambda *a, **kw: (87.5, 75.0, False),
    )
    result = metrics.measure_combined(
        target_file_path=str(target),
        test_output_dir=str(test_dir),
        source_dir=str(tmp_path),
        test_file_names=["test_foo.py"],
    )
    assert result.target_loc == 4
    assert result.target_complexity_sum == 5
    assert result.coverage_final_line == 87.5
    assert result.coverage_final_branch == 75.0
    assert len(result.test_files) == 1
    assert result.test_files[0].test_count == 2
    assert result.duration_s is not None and result.duration_s >= 0


def test_measure_combined_takes_recorded_coverage_without_a_run(monkeypatch, tmp_path) -> None:
    """Coverage the delivered suite's phase end measured is taken as it is:
    no coverage run is made for it."""
    target = tmp_path / "foo.py"
    target.write_text("def f(x):\n    return x + 1\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_foo.py").write_text("def test_one():\n    assert 1 == 1\n", encoding="utf-8")
    monkeypatch.setattr(metrics, "compute_file_complexity_stats", lambda path: (5, 3))

    def no_run(*args, **kwargs):
        raise AssertionError("a coverage run was made")

    monkeypatch.setattr(metrics, "_cov_via_backend", no_run)
    result = metrics.measure_combined(
        target_file_path=str(target), test_output_dir=str(test_dir),
        source_dir=str(tmp_path), test_file_names=["test_foo.py"],
        recorded_coverage=(92.5, 80.0),
    )
    assert (result.coverage_final_line, result.coverage_final_branch) == (92.5, 80.0)


def test_measure_combined_missing_target_skips_objective_metrics(
    monkeypatch, tmp_path,
) -> None:
    """Without the target source on disk, target_loc and complexity stay
    None, so the CSV shows them as not measured."""
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_foo.py").write_text(
        "def test_x(): assert True\n", encoding="utf-8",
    )
    monkeypatch.setattr(
        metrics, "compute_file_complexity_stats", lambda path: (None, None),
    )
    monkeypatch.setattr(
        metrics, "_cov_via_backend", lambda *a, **kw: (None, None, False),
    )
    result = metrics.measure_combined(
        target_file_path=str(tmp_path / "src" / "missing.py"),
        test_output_dir=str(test_dir),
        source_dir=str(tmp_path),
        test_file_names=["test_foo.py"],
    )
    assert result.target_loc is None
    assert result.coverage_final_line is None


def test_measure_combined_handles_missing_test_files_gracefully(
    monkeypatch, tmp_path,
) -> None:
    """Absent requested files retain unavailable structural observations."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    target = src_dir / "foo.py"
    target.write_text("def f(): pass\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    monkeypatch.setattr(metrics, "compute_file_complexity_stats", lambda p: (1, 1))
    monkeypatch.setattr(
        metrics, "_cov_via_backend", lambda *a, **kw: (None, None, False),
    )
    result = metrics.measure_combined(
        target_file_path=str(target),
        test_output_dir=str(test_dir),
        source_dir=str(tmp_path),
        test_file_names=["test_foo.py", "test_bar.py"],   # neither exists
    )
    assert len(result.test_files) == 2
    assert all(item.test_count is None for item in result.test_files)
    assert all(item.source_observation["status"] == "infrastructure_failure"
               for item in result.test_files)
    assert result.duration_s is not None


def test_measure_combined_measures_a_timed_out_suite_once(
    monkeypatch, tmp_path,
) -> None:
    """A timed-out coverage run is not retried and leaves the cell empty."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    target = src_dir / "foo.py"
    target.write_text("def f(): pass\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_foo.py").write_text("def test_a(): pass\n", encoding="utf-8")

    monkeypatch.setattr(
        metrics, "compute_file_complexity_stats", lambda p: (1, 1),
    )
    calls: list[tuple[tuple, dict]] = []

    def fake_cov(*args, **kwargs):
        calls.append((args, kwargs))
        return (None, None, True)

    monkeypatch.setattr(metrics, "_cov_via_backend", fake_cov)
    result = metrics.measure_combined(
        target_file_path=str(target),
        test_output_dir=str(test_dir),
        source_dir=str(tmp_path),
        test_file_names=["test_foo.py"],
    )
    assert len(calls) == 1
    assert result.coverage_final_line is None


# Full-suite timing: _measure_suite_time and the measure_combined switch,
# with backend.benchmark_tests mocked.

def _timing_backend(stats):
    """A backend stub whose ``benchmark_tests`` returns ``stats`` and records
    the kwargs it was called with."""
    from types import SimpleNamespace
    seen: dict = {}

    def _bench(**kwargs):
        seen.update(kwargs)
        return stats

    return SimpleNamespace(benchmark_tests=_bench), seen


def test_measure_suite_time_populates_fields(monkeypatch) -> None:
    """A green benchmark sets the median, the samples joined by semicolons, reps and
    warmup on the metrics object. The early-stop hook is passed disabled
    (None) so the measurement matches the baseline's."""
    backend, seen = _timing_backend({
        "ok": True, "median_ms": 123.0, "times_ms": [120.0, 123.0, 130.0],
        "timing_run_count": 3, "warmup_runs": 1,
        "requested_cpus": "0-3", "allowed_cpus": [0, 1, 2, 3],
    })
    set_phase_value(monkeypatch, metrics._TIMING_CONFIG, "run_count", 3)
    set_phase_value(monkeypatch, metrics._TIMING_CONFIG, "warmup_count", 1)
    result = metrics.CombinedTestMetrics(target_file_path="t.py")
    metrics._measure_suite_time(
        result, backend, ["test_foo.py"], "/proj",
    )
    assert result.suite_timing_median_ms == 123.0
    assert result.suite_timing_samples_ms == "120.0000;123.0000;130.0000"
    assert result.suite_timing_run_count == 3
    assert result.suite_timing_warmup_count == 1
    assert result.suite_timing_requested_cpus == "0-3"
    assert result.suite_timing_allowed_cpus == "0,1,2,3"
    # Parity: no early-stop option is sent, and the run and warmup counts
    # come from the module settings so they match the baseline side.
    assert "early_stop_baseline_ms" not in seen
    assert "early_stop_factor" not in seen
    assert seen["warmup_runs"] == 1
    assert seen["benchmark_runs"] == 3


def test_measure_suite_time_red_suite_leaves_fields_none(monkeypatch) -> None:
    """A non-green benchmark (ok=False) leaves every timing column None,
    like any failed measurement, and raises nothing into finishing."""
    backend, _ = _timing_backend({"ok": False, "error": "suite red"})
    result = metrics.CombinedTestMetrics(target_file_path="t.py")
    metrics._measure_suite_time(result, backend, ["test_foo.py"], "/proj")
    assert result.suite_timing_median_ms is None
    assert result.suite_timing_samples_ms is None


def test_measure_suite_time_no_benchmark_path_is_noop() -> None:
    """A backend without benchmark_tests (no timing path) leaves the columns
    None and does not raise."""
    from types import SimpleNamespace
    result = metrics.CombinedTestMetrics(target_file_path="t.py")
    metrics._measure_suite_time(
        result, SimpleNamespace(), ["test_foo.py"], "/proj",
    )
    assert result.suite_timing_median_ms is None


def test_measure_combined_timing_off_by_default_leaves_columns_empty(
    monkeypatch, tmp_path,
) -> None:
    """With the suite timing's measure off (the default), measure_combined leaves
    the suite_timing_* cells empty even when the backend could measure them.
    The real backend runs its AST and classify paths. Only benchmark_tests is
    replaced, with a body that fails the test if called."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    target = src_dir / "foo.py"
    target.write_text("def f(): pass\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_foo.py").write_text("def test_a(): pass\n", encoding="utf-8")

    set_phase_value(monkeypatch, metrics._TIMING_CONFIG, "measure", False)
    monkeypatch.setattr(metrics, "compute_file_complexity_stats", lambda p: (1, 1))
    monkeypatch.setattr(metrics, "_cov_via_backend", lambda *a, **kw: (90.0, 80.0, False))
    backend = metrics.languages.get_backend("python")
    called = {"n": 0}

    def _boom(*a, **kw):
        called["n"] += 1
        raise AssertionError("benchmark_tests called with runtime off")
    monkeypatch.setattr(type(backend), "benchmark_tests", _boom)
    monkeypatch.setattr(
        type(backend), "capture_coverage_matrix", lambda *a, **kw: None)

    result = metrics.measure_combined(
        target_file_path=str(target),
        test_output_dir=str(test_dir),
        source_dir=str(tmp_path),
        test_file_names=["test_foo.py"],
    )
    assert called["n"] == 0
    assert result.suite_timing_median_ms is None
    assert result.suite_timing_samples_ms is None
    assert result.suite_timing_run_count is None


def test_measure_combined_timing_on_records_columns(monkeypatch, tmp_path) -> None:
    """With the suite timing's measure on, measure_combined times the standard suite
    through the backend and records the timing columns. The real backend's
    benchmark_tests is replaced by a recording stub returning a green result."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    target = src_dir / "foo.py"
    target.write_text("def f(): pass\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_foo.py").write_text("def test_a(): pass\n", encoding="utf-8")

    set_phase_value(monkeypatch, metrics._TIMING_CONFIG, "measure", True)
    set_phase_value(monkeypatch, metrics._TIMING_CONFIG, "run_count", 2)
    set_phase_value(monkeypatch, metrics._TIMING_CONFIG, "warmup_count", 1)
    monkeypatch.setattr(metrics, "compute_file_complexity_stats", lambda p: (1, 1))
    monkeypatch.setattr(metrics, "_cov_via_backend", lambda *a, **kw: (90.0, 80.0, False))
    backend = metrics.languages.get_backend("python")
    seen: dict = {}

    def _bench(*a, **kwargs):
        seen.update(kwargs)
        return {"ok": True, "median_ms": 55.0, "times_ms": [50.0, 60.0],
                "timing_run_count": 2, "warmup_runs": 1}
    monkeypatch.setattr(type(backend), "benchmark_tests", _bench)
    monkeypatch.setattr(
        type(backend), "capture_coverage_matrix", lambda *a, **kw: None)

    result = metrics.measure_combined(
        target_file_path=str(target),
        test_output_dir=str(test_dir),
        source_dir=str(tmp_path),
        test_file_names=["test_foo.py"],
    )
    assert result.suite_timing_median_ms == 55.0
    assert result.suite_timing_samples_ms == "50.0000;60.0000"
    assert result.suite_timing_run_count == 2
    assert result.suite_timing_warmup_count == 1
    # The one test file is the suite that is timed.
    assert seen["test_file_paths"] == [str(test_dir / "test_foo.py")]
    assert result.status == "success"


# measure_finishing: the mutation and coverage steps are injected as
# callables, so no engine or real backend runs.


def _km(statuses: dict, operators: dict = None) -> dict:
    """Kill matrix from {mutant_id: status}. ``operators`` optionally maps
    {mutant_id: operator_name} so a selective filter can be exercised."""
    mutants = []
    for i, s in statuses.items():
        entry = {"id": i, "status": s}
        if operators is not None and i in operators:
            entry["operator"] = operators[i]
        mutants.append(entry)
    return {"mutants": mutants, "summary": {}}


def test_measure_finishing_isolates_the_suite_and_wipes_scratch(
        tmp_path: Path) -> None:
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    merged = tests_dir / "test_mod.py"
    merged.write_text("def test_a(): pass\n", encoding="utf-8")
    other = tests_dir / "test_mod_scaled.py"
    other.write_text("def test_s(): pass\n", encoding="utf-8")
    conftest = tests_dir / "conftest.py"
    conftest.write_text("# shared\n", encoding="utf-8")

    seen = []

    def run_mutation_on(test_dir):
        names = {p.name for p in Path(test_dir).iterdir()}
        seen.append(names)
        return _km({1: "killed", 2: "killed", 3: "survived"})


    scratch = tmp_path / "_finishing_scratch"
    scratch.mkdir()
    (scratch / "orphan.txt").write_text("from a crashed run", encoding="utf-8")

    m = metrics.measure_finishing(
        scratch_root=scratch,
        merged_test_path=merged,
        shared_setup_paths=[conftest],
        run_mutation_on=run_mutation_on,
        standard_coverage=(80.0, 70.0),
    )

    # The suite kills {1,2} of 3. The combined counts describe the same suite.
    assert round(m.mutation_score_full, 1) == 66.7
    assert m.mutation_measurement == "complete"
    # No selective set passed, so the selective scores stay None.
    assert m.mutation_score_selective is None
    assert (m.killed, m.survived, m.total) == (
        2, 1, 3)
    assert m.line_coverage == 80.0
    assert m.branch_coverage == 70.0
    # The suite ran isolated with its shared setup. Another file beside it
    # is not measured.
    assert seen == [{"conftest.py", "test_mod.py"}]
    assert (m.line_coverage, m.branch_coverage) == (80.0, 70.0)
    # The scratch directory is removed on the way out, with the orphan and
    # per-suite directories.
    assert not scratch.exists()


def test_measure_finishing_flags_a_heavily_unattributed_standard_matrix(
    tmp_path: Path,
) -> None:
    """A standard matrix whose kills mostly name no killing test, the
    signature of a broken baseline or contaminated run, is logged
    at finalize time. The reported score stays the matrix's cr-rate."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    merged = tests_dir / "test_mod.py"
    merged.write_text("def test_a(): pass\n", encoding="utf-8")

    # 8 kills, none naming a killer, plus 4 survivors: 100% of the kills
    # are unattributed and the count clears the alarm floor.
    contaminated = _km(
        {**{f"n{i}": "killed" for i in range(8)},
         **{f"s{i}": "survived" for i in range(4)}}
    )

    logs: list = []

    m = metrics.measure_finishing(
        scratch_root=tmp_path / "_scratch",
        merged_test_path=merged,
        shared_setup_paths=[],
        run_mutation_on=lambda _d: contaminated,
        standard_coverage=(80.0, 70.0),
        log_fn=logs.append,
    )

    assert any("no named killer" in line for line in logs), logs
    # 8 killed of 12 valid. The alarm did not touch the score.
    assert round(m.mutation_score_full, 1) == 66.7


def test_measure_finishing_flags_an_unresolved_standard_matrix(
    tmp_path: Path,
) -> None:
    """A standard matrix with an unresolved kill (a no-attribution kill the
    verification pass could not confirm) is logged at finalize time beside
    the unattributed and timeout alarms. The score stays the matrix's
    cr-rate."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    merged = tests_dir / "test_mod.py"
    merged.write_text("def test_a(): pass\n", encoding="utf-8")

    # 8 attributed kills and 4 survivors, one kill left unresolved by verification.
    km = _km(
        {**{f"n{i}": "killed" for i in range(8)},
         **{f"s{i}": "survived" for i in range(4)}}
    )
    for m in km["mutants"]:
        if m["status"] == "killed":
            m["killer"] = "tests/test_mod.py::test_a"
    km["mutants"][0]["unresolved"] = True

    logs: list = []
    metrics.measure_finishing(
        scratch_root=tmp_path / "_scratch",
        merged_test_path=merged,
        shared_setup_paths=[],
        run_mutation_on=lambda _d: km,
        standard_coverage=(80.0, 70.0),
        log_fn=logs.append,
    )

    assert any("UNRESOLVED" in line for line in logs), logs


def test_measure_finishing_flags_and_counts_timeout_kills(
    tmp_path: Path,
) -> None:
    """A clock-produced kill (timeout) depends on machine load. The
    finalize logs the timeout alarm and tests.csv records the count as
    finishing_timeout_kills, so such rows can be found and re-measured
    with a longer per-mutant timeout."""
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    merged = tests_dir / "test_mod.py"
    merged.write_text("def test_a(): pass\n", encoding="utf-8")

    with_timeout = _km({"k1": "killed", "t1": "timeout",
                        "s1": "survived"})
    logs: list = []
    m = metrics.measure_finishing(
        scratch_root=tmp_path / "_scratch",
        merged_test_path=merged,
        shared_setup_paths=[],
        run_mutation_on=lambda _d: with_timeout,
        standard_coverage=(80.0, 70.0),
        log_fn=logs.append,
    )
    assert m.timeout_kills == 1
    assert m.to_csv_dict()["finishing_timeout_kills"] == 1
    assert any("(status timeout" in line for line in logs), logs
    # The alarm never changes the score: 2 killed of 3 valid.
    assert round(m.mutation_score_full, 2) == 66.67


def test_measure_finishing_derives_selective_from_full_run(tmp_path: Path) -> None:
    """When a selective operator set is given and the matrices carry
    operators, the _selective scores derive from the same full run by
    filtering, with no extra mutation run. Only OpA is selective here."""
    merged = tmp_path / "test_mod.py"
    merged.write_text("def test_a(): pass\n", encoding="utf-8")

    # Four mutants: OpA kills both its mutants (100%). OpB kills one of two
    # (50%). Full catalogue kills 3 of 4 (75%).
    def run_mutation_on(test_dir):
        return _km(
            {1: "killed", 2: "killed", 3: "killed", 4: "survived"},
            operators={1: "OpA", 2: "OpA", 3: "OpB", 4: "OpB"},
        )

    m = metrics.measure_finishing(
        scratch_root=tmp_path / "scratch",
        merged_test_path=merged,
        shared_setup_paths=[],
        run_mutation_on=run_mutation_on,
        standard_coverage=(90.0, 80.0),
        selective_operators={"OpA"},
    )

    assert m.mutation_score_full == 75.0
    # OpA subset: both its mutants killed, 100%, distinct from the full 75%.
    assert m.mutation_score_selective == 100.0


def test_measure_finishing_no_selective_without_operators(tmp_path: Path) -> None:
    """A selective set is given but the matrix has no operator labels
    (an engine that does not record them): the selective scores stay None
    rather than scoring an empty subset."""
    merged = tmp_path / "test_mod.py"
    merged.write_text("def test_a(): pass\n", encoding="utf-8")

    def run_mutation_on(test_dir):
        return _km({1: "killed", 2: "survived"})   # no operators

    m = metrics.measure_finishing(
        scratch_root=tmp_path / "scratch",
        merged_test_path=merged,
        shared_setup_paths=[],
        run_mutation_on=run_mutation_on,
        standard_coverage=(50.0, 40.0),
        selective_operators={"OpA"},
    )

    assert m.mutation_score_full == 50.0
    assert m.mutation_score_selective is None


def test_measure_finishing_reports_the_suite_in_the_combined_fields(
        tmp_path: Path) -> None:
    merged = tmp_path / "test_mod.py"
    merged.write_text("def test_a(): pass\n", encoding="utf-8")

    def run_mutation_on(test_dir):
        return _km({1: "killed", 2: "survived"})

    m = metrics.measure_finishing(
        scratch_root=tmp_path / "scratch",
        merged_test_path=merged,
        shared_setup_paths=[],
        run_mutation_on=run_mutation_on,
        standard_coverage=(75.0, 60.0),
    )

    assert m.mutation_score_full == 50.0
    assert m.mutation_measurement == "complete"
    # The combined coverage is the suite's.
    assert m.line_coverage == 75.0


def test_measure_finishing_mutation_failure_yields_none_scores(tmp_path: Path) -> None:
    merged = tmp_path / "test_mod.py"
    merged.write_text("def test_a(): pass\n", encoding="utf-8")

    def run_mutation_on(test_dir):
        return None    # the engine run failed

    m = metrics.measure_finishing(
        scratch_root=tmp_path / "scratch",
        merged_test_path=merged,
        shared_setup_paths=[],
        run_mutation_on=run_mutation_on,
        standard_coverage=(50.0, 40.0),
    )

    assert m.mutation_score_full is None
    # The run was attempted and produced no matrix: failed, nothing counted.
    assert m.mutation_measurement == "failed"
    assert (m.killed, m.survived, m.total) == (
        None, None, None)
    # Coverage is independent of the mutation run and still recorded.
    assert m.line_coverage == 50.0


def test_measure_finishing_gives_an_unresolved_catalogue_no_score(
        tmp_path: Path) -> None:
    """A finishing measurement whose engine stamped the mutant catalogue
    unresolved (its wall clock ended the run with mutants outstanding) has
    no score, only the counts of the resolved mutants. Its per-mutant
    verdicts are not reclassified."""
    merged = tmp_path / "test_mod.py"
    merged.write_text("def test_a(): pass\n", encoding="utf-8")
    matrix = _km({1: "killed", 2: "timeout", 3: "survived"})
    matrix["incomplete"] = {"reason": "wall clock", "outstanding": 7}

    m = metrics.measure_finishing(
        scratch_root=tmp_path / "scratch",
        merged_test_path=merged,
        shared_setup_paths=[],
        run_mutation_on=lambda _d: matrix,
        standard_coverage=(60.0, 50.0),
    )

    assert m.mutation_measurement == "incomplete"
    assert m.mutation_score_full is None
    assert (m.killed, m.survived, m.total) == (
        2, 1, 3)
    row = m.to_csv_dict()
    assert row["finishing_mutation_measurement"] == "incomplete"
    assert "final_combined_mutation_score_full" not in row


def test_measure_finishing_without_a_mutation_measurement(
        tmp_path: Path) -> None:
    """The explicit opt-out measures no mutation: no status, no score."""
    merged = tmp_path / "test_mod.py"
    merged.write_text("def test_a(): pass\n", encoding="utf-8")
    m = metrics.measure_finishing(
        scratch_root=tmp_path / "scratch",
        merged_test_path=merged,
        shared_setup_paths=[],
        run_mutation_on=None,
        standard_coverage=(60.0, 50.0),
    )
    assert m.mutation_measurement is None
    assert m.mutation_score_full is None
    assert m.line_coverage == 60.0


def test_measure_finishing_cleans_scratch_even_when_callable_raises(
    tmp_path: Path,
) -> None:
    merged = tmp_path / "test_mod.py"
    merged.write_text("def test_a(): pass\n", encoding="utf-8")
    scratch = tmp_path / "scratch"

    def run_mutation_on(test_dir):
        raise RuntimeError("engine blew up")

    with pytest.raises(RuntimeError):
        metrics.measure_finishing(
            scratch_root=scratch,
            merged_test_path=merged,
            shared_setup_paths=[],
            run_mutation_on=run_mutation_on,
            standard_coverage=(None, None),
        )
    # The finally clause wiped the scratch despite the exception.
    assert not scratch.exists()


def test_combined_metrics_merges_finishing_columns() -> None:
    fin = metrics.FinishingMetrics(
        mutation_score_full=80.0,
        mutation_score_selective=72.0,
        mutation_measurement="complete",
        killed=9,
        survived=1,
        total=10,
        line_coverage=95.0,
        branch_coverage=88.0,
    )
    cm = metrics.CombinedTestMetrics(target_file_path="t.py", finishing=fin)
    d = cm.to_csv_dict()
    assert d["finishing_mutation_score_full"] == 80.0
    assert d["finishing_mutation_score_selective"] == 72.0
    assert d["finishing_mutation_measurement"] == "complete"
    assert d["finishing_total"] == 10
    assert d["finishing_line_coverage"] == 95.0


def test_combined_metrics_without_finishing_omits_final_columns() -> None:
    cm = metrics.CombinedTestMetrics(target_file_path="t.py")
    d = cm.to_csv_dict()
    assert not any(k.startswith("final_") for k in d)


# Finishing's wiring builder: the mutation callable runs the live session
# and captures the kill matrix.


@pytest.fixture
def _finishing_green(monkeypatch):
    """Make the finalize baseline green-check pass so the session/capture logic
    can be tested without a real project and suite."""
    from types import SimpleNamespace
    monkeypatch.setattr(
        metrics.languages, "get_backend",
        lambda lang: SimpleNamespace(
            file_extensions=(".py",), test_file_extensions=lambda: (".py",), shared_setup_files=lambda root: [],
            run_test_files=lambda paths, root, **kw: SimpleNamespace(
                ok=True, elapsed_s=0.1, returncode=0)))


class _Measured:
    """A measurement whose kill matrix is ``matrix`` (None: the run has
    none). ``calls`` counts the captures."""

    def __init__(self, matrix):
        self._matrix = matrix
        self.calls = 0

    def kill_matrix(self):
        self.calls += 1
        return self._matrix


class _MutSession:
    """A session whose runs all return ``measurement``. ``runs`` holds each
    run's arguments."""

    def __init__(self, measurement):
        self.measurement = measurement
        self.runs = []

    def run(self, **kw):
        self.runs.append(kw)
        return self.measurement


def test_make_finishing_mutation_fn_runs_session_and_parses(tmp_path, _finishing_green) -> None:
    matrix = {"mutants": [{"id": 1, "status": "killed"}], "summary": {}}
    measured = _Measured(matrix)
    session = _MutSession(measured)
    fn = metrics.make_finishing_mutation_fn(
        session=session,
        target_file="src/foo.py",
        source_dir="/proj",
        configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0,
    )
    test_dir = tmp_path / "delivered"
    test_dir.mkdir()
    out = fn(test_dir)
    assert out == matrix
    assert len(session.runs) == 1
    assert session.runs[0]["test_dir"] == test_dir
    assert session.runs[0]["operators"] is None
    assert measured.calls == 1
    # The baseline time and the per-mutant timeout the run was given (this
    # fake measurement states no applied timeout).
    assert fn.limit_of(test_dir) == {"baseline_seconds": 0.1, "per_mutant_timeout_seconds": 60,
                                     "per_mutant_timeout": None}
    assert fn.limit_of(tmp_path / "never") is None


def test_make_finishing_mutation_fn_run_failure_returns_none(tmp_path, _finishing_green) -> None:
    class BoomSession:
        def run(self, **kw):
            raise RuntimeError("worker pool died")

    fn = metrics.make_finishing_mutation_fn(
        session=BoomSession(), target_file="f", source_dir="r",
        configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0,
    )
    assert fn(tmp_path) is None


def test_a_run_the_origin_check_stopped_names_its_stopping_tests(tmp_path, _finishing_green) -> None:
    """The mutation tool's origin check stopped the run, a test failing
    without a mutant: the log line and the runner's stops name every
    stopping test, and the run gives no matrix."""
    from testuary.languages.mutation import MutationMeasurement

    reason = "the unmutated suite failed on the Mull build"
    stopping = ("test_floats.c::floats.width_seven", "test_floats.c::floats.half")

    class Session:
        def run(self, **kw):
            return MutationMeasurement.empty(reason=reason, suite_failed=True,
                                             stopping_tests=stopping)

    logged = []
    fn = metrics.make_finishing_mutation_fn(
        session=Session(), target_file="f", source_dir="r",
        configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0, log_fn=logged.append)
    assert fn(tmp_path) is None
    assert (f"[finishing] the mutation tool's origin check stopped {tmp_path.name}: {reason} "
            f"(stopping tests: {stopping[0]}, {stopping[1]})") in logged
    assert fn.stops == [{"suite": tmp_path.name, "reason": reason,
                         "stopping_tests": list(stopping)}]


def test_a_suite_the_origin_check_stopped_is_named_with_its_stopping_tests(
        tmp_path, _finishing_green) -> None:
    """A per-suite run: the suite the origin check stopped is named with its
    stopping tests, and a suite that measured nothing for another reason
    keeps its line."""
    from types import SimpleNamespace

    stopping = (SimpleNamespace(text="test_a.c::a.one"), SimpleNamespace(text="test_a.c::a.two"))

    class Session:
        def per_suite(self, **kw):
            run = _PerSuiteRun({"suite_001": ("test_failures", False),
                                "suite_002": ("infrastructure_failure", False)}, {})
            run.runs["suite_001"].stopping_tests = stopping
            return run

    logged = []
    fn = metrics.make_finishing_mutation_fn(
        session=Session(), target_file="f", source_dir="r",
        configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0, log_fn=logged.append)
    dirs = {name: tmp_path / name for name in ("suite_001", "suite_002")}
    for path in dirs.values():
        path.mkdir()
    out = fn.per_suite(dirs)
    text = "suite_001 reason (stopping tests: test_a.c::a.one, test_a.c::a.two)"
    assert out["stopped"] == {"suite_001": text}
    assert f"[finishing] the mutation tool's origin check stopped suite_001: {text}" in logged
    assert any("suite_002 measured nothing (infrastructure_failure" in str(line) for line in logged)
    assert out["unmeasured"] == ["suite_001", "suite_002"]
    assert fn.stops == [{"suite": "suite_001", "reason": "suite_001 reason",
                         "stopping_tests": ["test_a.c::a.one", "test_a.c::a.two"]}]


def test_the_runner_hands_on_the_tests_an_instrument_fault_names(
        tmp_path, _finishing_green) -> None:
    """At a model-silent place the runner keeps the test id objects that an
    instrument fault's run names, such as Python's origin refusal, while it
    logs the failed run as today."""
    from langadapter.suites import TestId
    from testuary.languages.mutation import MutationInstrumentFault

    named = TestId("test_a.py", "test_loaded", text="test_a.py::test_loaded")

    class Session:
        def run(self, **kw):
            fault = MutationInstrumentFault("the origin check refused")
            fault.stopping_tests = (named,)
            raise fault

    logged = []
    fn = metrics.make_finishing_mutation_fn(
        session=Session(), target_file="f", source_dir="r",
        configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0, log_fn=logged.append)
    assert fn(tmp_path) is None
    assert fn.named_of(tmp_path) == (named,)
    assert (f"[finishing] mutation run failed on {tmp_path.name}: the origin check refused"
            in logged)
    assert fn.stops == []


def test_the_runner_hands_on_a_red_origin_and_records_its_texts(
        tmp_path, _finishing_green) -> None:
    """A red origin (test_failures) hands on its test id objects, and its
    stop keeps the texts it records today."""
    from langadapter.suites import TestId
    from testuary.languages.mutation import MutationMeasurement

    named = TestId("test_a.py", "test_loaded", text="test_a.py::test_loaded")

    class Session:
        def run(self, **kw):
            return MutationMeasurement.empty(reason="red", suite_failed=True,
                                             stopping_tests=(named,))

    fn = metrics.make_finishing_mutation_fn(
        session=Session(), target_file="f", source_dir="r",
        configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0)
    assert fn(tmp_path) is None
    assert fn.named_of(tmp_path) == (named,)
    assert fn.stops == [{"suite": tmp_path.name, "reason": "red",
                         "stopping_tests": ["test_a.py::test_loaded"]}]


def test_a_per_suite_run_hands_on_the_tests_it_names(tmp_path, _finishing_green) -> None:
    """A per-suite run that measured nothing hands on the tests each suite's
    run names under each suite's own directory, and logs each suite as
    today."""
    from langadapter.suites import TestId

    named = TestId("test_a.py", "test_loaded", text="test_a.py::test_loaded")

    class Session:
        def per_suite(self, **kw):
            run = _PerSuiteRun({"suite_001": ("infrastructure_failure", False),
                                "suite_002": ("infrastructure_failure", False)}, {})
            run.runs["suite_001"].stopping_tests = (named,)
            return run

    logged = []
    fn = metrics.make_finishing_mutation_fn(
        session=Session(), target_file="f", source_dir="r",
        configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0, log_fn=logged.append)
    dirs = {name: tmp_path / name for name in ("suite_001", "suite_002")}
    for path in dirs.values():
        path.mkdir()
    out = fn.per_suite(dirs)
    assert fn.named_of(dirs["suite_001"]) == (named,)
    assert fn.named_of(dirs["suite_002"]) == ()
    assert "named" not in out
    assert any("suite_001 measured nothing (infrastructure_failure" in str(line)
               for line in logged)


def _red_green_check(monkeypatch, failing, *, timed_out=False):
    """A backend whose green check fails the directories in ``failing`` with
    their tests and passes every other one."""
    from types import SimpleNamespace

    def run_test_files(paths, root, **kw):
        tests = next((tests for name, tests in failing.items()
                      if any(name in str(path) for path in paths)), ())
        return SimpleNamespace(ok=not tests, elapsed_s=0.1, returncode=1 if tests else 0,
                               timed_out=timed_out and bool(tests), stopping_tests=tests)

    monkeypatch.setattr(
        metrics.languages, "get_backend",
        lambda lang: SimpleNamespace(
            file_extensions=(".py",), test_file_extensions=lambda: (".py",),
            shared_setup_files=lambda root: [], run_test_files=run_test_files))


def test_a_green_check_hands_on_the_tests_it_fails_unless_timed_out(
        tmp_path, monkeypatch) -> None:
    """A standalone run whose green check fails hands on the tests that check
    names, and a timed-out check names none, as a mutation run does."""
    from langadapter.suites import TestId

    named = TestId("test_a.py", "test_loaded", text="test_a.py::test_loaded")
    for timed_out, expected in ((False, (named,)), (True, ())):
        suite = tmp_path / f"suite_{timed_out}"
        suite.mkdir()
        (suite / "test_a.py").write_text("x", encoding="utf-8")
        _red_green_check(monkeypatch, {suite.name: (named,)}, timed_out=timed_out)
        fn = metrics.make_finishing_mutation_fn(
            session=object(), target_file="f", source_dir="r",
            configured_per_mutant_timeout=60, wall_clock_timeout=3600,
            per_mutant_timeout_max=900, per_mutant_timeout_min=30,
            per_mutant_timeout_factor=10.0)
        assert fn(suite) is None
        assert fn.named_of(suite) == expected


def test_a_per_suite_green_check_hands_on_the_tests_it_fails(
        tmp_path, monkeypatch) -> None:
    """Each suite whose green check fails before a per-suite run hands on the
    tests that check names, also where no suite is green and nothing runs."""
    from langadapter.suites import TestId

    named = TestId("test_a.py", "test_loaded", text="test_a.py::test_loaded")
    dirs = {name: tmp_path / name for name in ("suite_001", "suite_002")}
    for path in dirs.values():
        path.mkdir()
        (path / "test_a.py").write_text("x", encoding="utf-8")
    _red_green_check(monkeypatch, {"suite_001": (named,), "suite_002": (named,)})

    class Session:
        def per_suite(self, **kw):
            raise AssertionError("no suite is green")

    fn = metrics.make_finishing_mutation_fn(
        session=Session(), target_file="f", source_dir="r",
        configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0)
    assert fn.per_suite(dirs) is None
    assert fn.named_of(dirs["suite_001"]) == (named,)
    assert fn.named_of(dirs["suite_002"]) == (named,)


class _PerSuiteRun:
    """A per-suite result: ``runs`` maps each suite to its outcome and
    whether it measured, and a measured suite's own matrix is ``matrices``."""

    implementation = "per_suite_runs"

    def __init__(self, runs, matrices, union=None):
        from types import SimpleNamespace

        from langadapter.outcomes import Outcome

        self.runs = {name: SimpleNamespace(outcome=Outcome(outcome), measured=measured,
                                           reason=f"{name} reason")
                     for name, (outcome, measured) in runs.items()}
        self._matrices = matrices
        self._union = union

    def kill_matrix(self, name=None):
        if name is None:
            return self._union or {"mutants": [], "union": True}
        return self._matrices[name]


def test_per_suite_measurement_keeps_a_failed_suite_unmeasured(tmp_path, _finishing_green) -> None:
    # A suite whose run measured nothing has no matrix, so no score and never
    # zeros. A run cut by the wall clock keeps its matrix and its outcome.
    cut = {"mutants": [{"id": 1, "status": "killed"}]}
    logged = []

    class Session:
        def per_suite(self, **kw):
            assert [entry[0] for entry in kw["suites"]] == ["suite_001", "suite_002"]
            return _PerSuiteRun({"suite_001": ("timed_out", True),
                                 "suite_002": ("infrastructure_failure", False)},
                                {"suite_001": cut, "suite_002": {"mutants": []}})

    fn = metrics.make_finishing_mutation_fn(
        session=Session(), target_file="f", source_dir="r",
        configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0, log_fn=logged.append)
    dirs = {name: tmp_path / name for name in ("suite_001", "suite_002")}
    for path in dirs.values():
        path.mkdir()
    out = fn.per_suite(dirs)
    assert out is not None, logged
    assert out["suites"] == {"suite_001": cut}
    assert out["outcomes"] == {"suite_001": "timed_out", "suite_002": "infrastructure_failure"}
    assert out["unmeasured"] == ["suite_002"]
    assert any("suite_002 measured nothing (infrastructure_failure" in str(line) for line in logged)
    assert out["budgets"] == {name: {"baseline_seconds": 0.1, "per_mutant_timeout_seconds": 60,
                                     "per_mutant_timeout": None}
                              for name in ("suite_001", "suite_002")}


def test_a_suite_not_green_leaves_the_merged_matrix_incomplete():
    # A suite that never entered the per-suite request has no result for any
    # mutant: a row no suite killed is untested, and ``merged_matrix`` is incomplete.
    merged_matrix = {"mutants": [
        {"id": 1, "status": "killed", "suite_verdicts": {"suite_001": {"verdict": "killed", "killers": ["t"]}}},
        {"id": 2, "status": "survived", "suite_verdicts": {"suite_001": {"verdict": "survived", "killers": []}}},
    ], "summary": {}, "schema_version": 3}
    metrics._with_suites_not_measured(merged_matrix, ["suite_002"], {})
    # The killed row stays killed: only a row no suite killed is untested.
    assert [m["status"] for m in merged_matrix["mutants"]] == ["killed", "untested"]
    assert merged_matrix["mutants"][1]["suite_verdicts"]["suite_002"] == {"verdict": "untested", "killers": []}
    assert merged_matrix["incomplete"]["untested"] == 1
    from testuary.languages.mutation import matrix_score_percent
    assert matrix_score_percent(merged_matrix) is None


@pytest.mark.parametrize("verdict", ["killed", "collection_error", "non_viable", "timeout",
                                     "survived", "untested"])
def test_a_suite_not_measured_merges_as_langadapter_merges_it(verdict):
    """_with_suites_not_measured restates LangAdapter's per-suite merge for a
    suite outside the request. A union LangAdapter merges over both suites
    and one it merges over the measured suite, completed here, agree on
    each row's status and on the incomplete stamp."""
    from langadapter.mutation import (
        INCOMPLETE_KEY, UNTESTED, MutantKey, MutantVerdicts, MutationRun, PerSuiteRun,
        SuiteVerdict)
    from langadapter.outcomes import Outcome

    key = MutantKey("t.py", "op", 0, 3, 0, "0" * 64)
    measured = SuiteVerdict(verdict, ("test_t.py::test_a",) if verdict == "killed" else ())

    def union(verdicts):
        runs = {name: MutationRun.failed("t.py", Outcome.COMPLETED, "") for name in verdicts}
        return PerSuiteRun(target="t.py", outcome=Outcome.COMPLETED, reason="",
                           implementation="test", runs=runs,
                           mutants=(MutantVerdicts(key, None, verdicts),)).kill_matrix()

    both = union({"suite_001": measured, "suite_002": SuiteVerdict(UNTESTED)})
    completed = union({"suite_001": measured})
    metrics._with_suites_not_measured(completed, ["suite_002"], {})
    assert [m["status"] for m in completed["mutants"]] == [m["status"] for m in both["mutants"]]
    assert completed.get(INCOMPLETE_KEY) == both.get(INCOMPLETE_KEY)


def test_a_mutant_that_did_not_load_merges_as_langadapter_merges_it():
    """A measured suite's mutant that did not load (the native status
    load_error) beside a suite outside the request: the union LangAdapter
    merges over both suites and the one it merges over the measured suite,
    completed here from that suite's own matrix, agree on the row's status
    and on the incomplete stamp."""
    from langadapter.mutation import (
        INCOMPLETE_KEY, NON_VIABLE, UNTESTED, Classification, Mutant, MutantVerdicts,
        MutationClass, MutationRun, PerSuiteRun, SuiteVerdict, mutant_key)
    from langadapter.outcomes import Outcome

    key = mutant_key("t.c", "op", 0, 3, 0, "-a +b")
    did_not_load = Mutant(
        key=key, engine_id="m1", test_outcome=None, worker_outcome=None,
        classification=Classification(NON_VIABLE, MutationClass.DETECTED, NON_VIABLE),
        killers=(), diff="-a +b",
        row={"id": "m1", "line": 3, "operator": "op", "status": NON_VIABLE,
             "native_status": "load_error", "load_error": "the extension did not load"})

    def per_suite(verdicts):
        runs = {name: MutationRun.failed("t.c", Outcome.COMPLETED, "") for name in verdicts}
        runs["suite_001"] = MutationRun("t.c", (did_not_load,), {}, "full", ())
        return PerSuiteRun(target="t.c", outcome=Outcome.COMPLETED, reason="",
                           implementation="test", runs=runs,
                           mutants=(MutantVerdicts(key, None, verdicts),))

    both = per_suite({"suite_001": SuiteVerdict(NON_VIABLE),
                      "suite_002": SuiteVerdict(UNTESTED)}).kill_matrix()
    measured = per_suite({"suite_001": SuiteVerdict(NON_VIABLE)})
    completed = measured.kill_matrix()
    metrics._with_suites_not_measured(completed, ["suite_002"],
                                      {"suite_001": measured.kill_matrix("suite_001")})
    assert [m["status"] for m in completed["mutants"]] == [m["status"] for m in both["mutants"]]
    assert completed.get(INCOMPLETE_KEY) == both.get(INCOMPLETE_KEY)


def test_a_mutant_that_did_not_load_stays_detected_beside_a_suite_not_green(tmp_path, monkeypatch) -> None:
    """A C mutant under which the Python extension did not load (the native
    status load_error) is non_viable for its suite. The load error belongs to
    the mutant, not to a suite, so a suite left out of the request as not
    green does not make the union row untested, and the union stays
    complete, as LangAdapter merges a suite without a result."""
    import dataclasses
    from types import SimpleNamespace

    from langadapter.mutation import MutantKey

    def run_test_files(paths, root, **kw):
        green = "suite_001" in " ".join(paths)
        return SimpleNamespace(ok=green, elapsed_s=0.1, returncode=0 if green else 1)

    monkeypatch.setattr(metrics.languages, "get_backend", lambda lang: SimpleNamespace(
        file_extensions=(".py",), test_file_extensions=lambda: (".py",),
        shared_setup_files=lambda root: [], run_test_files=run_test_files))
    key = dataclasses.asdict(MutantKey("t.c", "op", 0, 3, 0, "0" * 64))
    own = {"mutants": [{"id": 1, "status": "non_viable", "native_status": "load_error",
                        "load_error": "the extension did not load", "key": key}]}
    union = {"schema_version": 3, "summary": {}, "mutants": [
        {"id": 1, "status": "non_viable", "key": key, "killer": None, "killers": [],
         "suite_verdicts": {"suite_001": {"verdict": "non_viable", "killers": []}}}]}

    class Session:
        def per_suite(self, **kw):
            assert [entry[0] for entry in kw["suites"]] == ["suite_001"]
            return _PerSuiteRun({"suite_001": ("completed", True)}, {"suite_001": own}, union)

    fn = metrics.make_finishing_mutation_fn(
        session=Session(), target_file="f", source_dir="r",
        configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0)
    dirs = {name: tmp_path / name for name in ("suite_001", "suite_002")}
    for path in dirs.values():
        path.mkdir()
        (path / "test_t.py").write_text("def test_t():\n    pass\n", encoding="utf-8")
    out = fn.per_suite(dirs)
    assert out["unmeasured"] == ["suite_002"]
    (row,) = out["union"]["mutants"]
    assert row["status"] == "non_viable"
    assert row["suite_verdicts"]["suite_002"] == {"verdict": "untested", "killers": []}
    assert "incomplete" not in out["union"] and out["union"]["summary"]["untested"] == 0


def test_per_suite_measurement_does_not_hide_a_session_without_it(tmp_path, _finishing_green) -> None:
    fn = metrics.make_finishing_mutation_fn(
        session=_MutSession(_Measured(None)), target_file="f", source_dir="r",
        configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0)
    (tmp_path / "suite_001").mkdir()
    with pytest.raises(AttributeError):
        fn.per_suite({"suite_001": tmp_path / "suite_001"})


def test_make_finishing_mutation_fn_empty_capture_returns_none(tmp_path, _finishing_green) -> None:
    # A run that measured nothing has no matrix.
    fn = metrics.make_finishing_mutation_fn(
        session=_MutSession(_Measured(None)), target_file="f", source_dir="r",
        configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0,
    )
    assert fn(tmp_path) is None


def test_finishing_mutation_measures_once_and_returns_unchanged(
    tmp_path, _finishing_green,
) -> None:
    """Finishing runs the engine once and returns the captured matrix
    verbatim: no re-measure loop, no reliability flag, no edit to the summary."""
    matrix = {
        "mutants": [
            {"id": "k0", "status": "killed"},
            {"id": "s0", "status": "survived"},
        ],
        "summary": {"total": 2, "killed": 1, "survived": 1},
    }
    eng = _Measured(matrix)
    fn = metrics.make_finishing_mutation_fn(
        session=_MutSession(eng), target_file="f", source_dir="r",
        configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0,
    )
    km = fn(tmp_path)
    assert eng.calls == 1
    assert km == matrix
    assert "mutation_unreliable" not in km["summary"]


def test_make_finishing_mutation_fn_refuses_red_baseline(tmp_path, monkeypatch) -> None:
    """A finalize whose unmutated suite is not green does not mutate:
    cosmic-ray scores every mutant killed on a broken baseline, a false 100%.
    The check returns None and never opens the session."""
    from types import SimpleNamespace
    monkeypatch.setattr(
        metrics.languages, "get_backend",
        lambda lang: SimpleNamespace(
            file_extensions=(".py",), test_file_extensions=lambda: (".py",), shared_setup_files=lambda root: [],
            run_test_files=lambda paths, root, **kw: SimpleNamespace(
                ok=False, elapsed_s=0.1, returncode=2)))   # collection error

    class MustNotRun:
        def run(self, **kw):
            raise AssertionError("must not mutate a red baseline")

    fn = metrics.make_finishing_mutation_fn(
        session=MustNotRun(),
        target_file="f", source_dir="r", configured_per_mutant_timeout=60, wall_clock_timeout=3600,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0,
    )
    assert fn(tmp_path) is None


def test_finishing_green_check_names_test_files_not_setup(tmp_path, monkeypatch):
    from types import SimpleNamespace

    test_file = tmp_path / "test_generated.py"
    test_file.write_text("def test_ok():\n    assert True\n", encoding="utf-8")
    setup_file = tmp_path / "conftest.py"
    setup_file.write_text("", encoding="utf-8")
    (tmp_path / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
    calls = []

    def green(paths, root, **kwargs):
        calls.append((paths, kwargs))
        return SimpleNamespace(ok=True, elapsed_s=0.1, returncode=0)

    monkeypatch.setattr(metrics.languages, "get_backend", lambda lang: SimpleNamespace(
        file_extensions=(".py",), test_file_extensions=lambda: (".py",), shared_setup_files=lambda root: [str(setup_file)],
        run_test_files=green))
    matrix = {"mutants": [], "summary": {}}
    session = _MutSession(_Measured(matrix))
    measure = metrics.make_finishing_mutation_fn(
        session=session, target_file="src/foo.py",
        source_dir="/proj", configured_per_mutant_timeout=60,
        wall_clock_timeout=3600, per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0)
    assert measure(tmp_path) == matrix
    assert len(calls) == 1
    assert calls[0][0] == [str(test_file)]
    assert calls[0][1]["instrument_target"] == "src/foo.py"
    # The mutation run measures the files the green check ran.
    assert session.runs[0]["test_files"] == [str(test_file)]


def test_loc_counts_code_lines_not_comment_residue():
    """Removed tests leave their comment blocks behind in a merged suite.
    loc measures the code that remains, not the residue."""
    from testuary import languages

    source = (
        "# ----------------------------\n"
        "# tests for removed_function\n"
        "# ----------------------------\n"
        "\n"
        "# rationale for a pruned test\n"
        "def test_kept():\n"
        "    assert 1 == 1\n"
    )
    m = languages.get_backend("python").test_metrics(source)
    assert m["loc"] == 2
    assert m["test_count"] == 1


@pytest.mark.parametrize("cut", [True, False])
def test_a_finishing_measurement_the_time_limit_cut_shows_its_lower_bound(tmp_path: Path, cut) -> None:
    """Cut by its time limit, the finishing measurement stays incomplete,
    with no score, and records the score counting every untested mutant as not
    killed and how many were untested. A complete one records neither."""
    merged = tmp_path / "test_mod.py"
    merged.write_text("def test_a(): pass\n", encoding="utf-8")
    matrix = _km({1: "killed", 2: "timeout", 3: "survived"}
                 | ({4: "untested"} if cut else {}))
    if cut:
        matrix["incomplete"] = {"reason": "untested_mutants_in_catalogue", "untested": 1}
    m = metrics.measure_finishing(
        scratch_root=tmp_path / "scratch", merged_test_path=merged, shared_setup_paths=[],
        run_mutation_on=lambda _d: matrix, standard_coverage=(60.0, 50.0))
    row = m.to_csv_dict()
    if cut:
        assert m.mutation_measurement == "incomplete" and m.mutation_score_full is None
        assert (row["finishing_mutation_score_lower_bound"], row["finishing_untested"]) == (50.0, 1)
    else:
        assert m.mutation_measurement == "complete"
        assert (row["finishing_mutation_score_lower_bound"], row["finishing_untested"]) == (None, None)


@pytest.mark.parametrize("phase", ["coverage", "finishing"])
def test_a_shared_measurement_its_clock_cut_is_noted_for_its_phase(
        tmp_path, _finishing_green, phase) -> None:
    """A measurement the time limit cut is one of the warnings of the phase
    that ran it, Finishing's included."""
    from testuary import results as _time_limits

    measured = _Measured({"mutants": [{"id": 1, "status": "killed"}], "summary": {}})
    measured.truncated = True
    fn = metrics.make_finishing_mutation_fn(
        session=_MutSession(measured), target_file="src/foo.py", source_dir="/proj",
        configured_per_mutant_timeout=60, wall_clock_timeout=900,
        per_mutant_timeout_max=900, per_mutant_timeout_min=30,
        per_mutant_timeout_factor=10.0)
    test_dir = tmp_path / "delivered"
    test_dir.mkdir()
    with _time_limits.collecting(tmp_path / "r0"), _time_limits.in_phase(phase):
        fn(test_dir)
    [warning] = _time_limits.warnings(tmp_path / "r0")
    assert (warning["phase"], warning["setting"], warning["value"]) == (
        phase, "mutation.measurement_timeout_seconds", 900)



# Finishing's kill-matrix persistence.
#
# An explicit output path receives the measured matrix. Omitting that path
# does not write a matrix next to the suite.
def _setup(tmp_path: Path):
    merged = tmp_path / "test_mod.py"
    merged.write_text("def test_x():\n    assert True\n", encoding="utf-8")
    scratch = tmp_path / "_finishing_scratch"
    scratch.mkdir()
    std_matrix = {
        "schema_version": 3,
        "target_file": "mod.py",
        "mutants": [
            {"id": "m1", "line": 1, "operator": "op", "status": "killed"},
            {"id": "m2", "line": 2, "operator": "op", "status": "survived"},
        ],
        "summary": {"total": 2, "detected": 1, "survived": 1,
                    "timeout": 0, "non_viable": 0, "untested": 0},
    }
    return merged, scratch, std_matrix


def test_measure_finishing_persists_without_direct_path_write(
    tmp_path: Path, monkeypatch,
) -> None:
    """The delivered matrix must use the module's atomic writer."""
    merged, scratch, std_matrix = _setup(tmp_path)
    out = tmp_path / "kill_matrix.json"

    def reject_direct_write(*_args, **_kwargs):
        raise AssertionError("kill_matrix.json used Path.write_text")

    monkeypatch.setattr(Path, "write_text", reject_direct_write)
    measurement.measure_finishing(
        scratch_root=scratch,
        merged_test_path=merged,
        shared_setup_paths=[],
        run_mutation_on=lambda _d: std_matrix,
        standard_coverage=(80.0, 70.0),
        kill_matrix_out=out,
    )

    assert json.loads(out.read_text(encoding="utf-8")) == {
        **std_matrix, "source_sha256": source_sha256(merged.read_text(encoding="utf-8")),
        "execution_name": merged.name}


def test_measure_finishing_fails_when_matrix_cannot_be_persisted(
    tmp_path: Path, monkeypatch,
) -> None:
    merged, scratch, std_matrix = _setup(tmp_path)

    def fail_atomic_write(*_args, **_kwargs):
        raise OSError("storage failure")

    monkeypatch.setattr(measurement, "_atomic_write_text", fail_atomic_write)
    with pytest.raises(RuntimeError, match="could not persist kill_matrix"):
        measurement.measure_finishing(
            scratch_root=scratch,
            merged_test_path=merged,
            shared_setup_paths=[],
            run_mutation_on=lambda _d: std_matrix,
            standard_coverage=(80.0, 70.0),
            kill_matrix_out=tmp_path / "kill_matrix.json",
        )


def test_measure_finishing_skips_persist_without_out_path(tmp_path: Path) -> None:
    merged, scratch, std_matrix = _setup(tmp_path)
    measurement.measure_finishing(
        scratch_root=scratch,
        merged_test_path=merged,
        shared_setup_paths=[],
        run_mutation_on=lambda _d: std_matrix,
        standard_coverage=(80.0, 70.0),
    )
    # With no output path, no matrix is written next to the suite.
    assert not (tmp_path / "kill_matrix.json").exists()


@pytest.mark.parametrize("inherited", [False, True])
def test_the_final_kill_matrix_names_the_suite_it_describes(tmp_path: Path, inherited) -> None:
    """kill_matrix.json carries the delivered suite's hash and file name,
    as the phase archive's matrices do, whether Finishing measured the
    matrix or took the phase end's."""
    merged, scratch, std_matrix = _setup(tmp_path)
    out = tmp_path / "kill_matrix.json"
    measurement.measure_finishing(
        scratch_root=scratch,
        merged_test_path=merged,
        shared_setup_paths=[],
        run_mutation_on=None if inherited else (lambda _d: std_matrix),
        inherited_standard_km=std_matrix if inherited else None,
        standard_coverage=(80.0, 70.0),
        kill_matrix_out=out,
    )
    written = json.loads(out.read_text(encoding="utf-8"))
    assert written["source_sha256"] == source_sha256("def test_x():\n    assert True\n")
    assert written["execution_name"] == "test_mod.py"


# Finishing's coverage matrix.
@pytest.mark.parametrize("captured", [True, False])
def test_the_final_coverage_matrix_names_the_suite_it_describes(
        monkeypatch, tmp_path: Path, captured) -> None:
    """coverage_matrix.json carries the delivered suite's hash and file
    name, as the phase archive's matrices do. When the capture writes
    none, no earlier attempt's matrix stays beside the suite."""
    src_dir = tmp_path / "src"
    src_dir.mkdir()
    target = src_dir / "foo.py"
    target.write_text("def f(): pass\n", encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir()
    (test_dir / "test_foo.py").write_text("def test_a(): pass\n", encoding="utf-8")
    matrix = test_dir / "coverage_matrix.json"
    matrix.write_text('{"lines": []}', encoding="utf-8")
    set_phase_value(monkeypatch, metrics._TIMING_CONFIG, "measure", False)
    monkeypatch.setattr(metrics, "compute_file_complexity_stats", lambda p: (1, 1))
    monkeypatch.setattr(metrics, "_cov_via_backend", lambda *a, **kw: (90.0, 80.0, False))
    backend = metrics.languages.get_backend("python")

    def capture(self, *, out_path, **_kwargs):
        if captured:
            Path(out_path).write_text('{"schema_version": 1, "lines": []}', encoding="utf-8")

    monkeypatch.setattr(type(backend), "capture_coverage_matrix", capture)
    metrics.measure_combined(
        target_file_path=str(target),
        test_output_dir=str(test_dir),
        source_dir=str(tmp_path),
        test_file_names=["test_foo.py"],
    )
    if captured:
        assert json.loads(matrix.read_text(encoding="utf-8")) == {
            "schema_version": 1, "lines": [],
            "source_sha256": source_sha256("def test_a(): pass\n"),
            "execution_name": "test_foo.py"}
    else:
        assert not matrix.exists()
