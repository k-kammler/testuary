from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib  # Python 3.11+
except ImportError:  # pragma: no cover - exercised on 3.10 hosts only
    import tomli as tomllib  # type: ignore

import pytest
from langadapter import Outcome
from packaging.requirements import Requirement

from testuary import config, languages
from testuary import project as _execution_tools
from testuary import reduction
from testuary.settings import RunSettings


ROOT = Path(__file__).resolve().parents[2]


def test_the_declared_dependencies_pin_the_execution_tool_versions(tmp_path) -> None:
    """The package depends on exactly the test runner, coverage and mutation
    tool versions LangAdapter measures this Python's projects with: the pins
    of the capability report for the Python that runs the tests."""
    settings = RunSettings(base_dir=tmp_path, environment=dict(os.environ),
                           sections={"project": {"python_executable": sys.executable}})
    with config.use_run_settings(settings):
        report = languages.get_backend("python").capabilities(str(tmp_path))
    assert report.outcome is Outcome.COMPLETED, report.reason
    version = "%d.%d" % sys.version_info[:2]
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    selected = set()
    for text in data["project"]["dependencies"]:
        requirement = Requirement(text)
        if requirement.marker is None or requirement.marker.evaluate({"python_version": version}):
            selected.add(f"{requirement.name.lower()}{requirement.specifier}")
    pins = {f"{tool.name}=={tool.pinned}" for tool in report.tools}
    assert pins and pins <= selected


def test_the_bundled_solver_reports_the_pinned_versions() -> None:
    assert _execution_tools.host_solver_versions() == {
        "PuLP": "3.3.2", "CBC": "2.10.3"}
    assert _execution_tools.host_solver_mismatches() == ()


def test_a_solver_probe_that_fails_keeps_its_reason(monkeypatch) -> None:
    # A fresh store: an earlier test may have found the real CBC's version.
    monkeypatch.setattr(reduction, "_CBC_VERSIONS", {})
    def time_out(cmd, **kwargs):
        raise subprocess.TimeoutExpired(cmd, kwargs["timeout"])

    monkeypatch.setattr(_execution_tools.subprocess, "run", time_out)
    observed = _execution_tools.host_solver_versions()["CBC"]
    assert observed.startswith("unavailable (TimeoutExpired: Command ")
    assert observed.endswith("timed out after 10 seconds)")
    (mismatch,) = _execution_tools.host_solver_mismatches()
    assert mismatch == f"CBC: expected 2.10.3, observed {observed}"


def test_a_pulp_without_the_bundled_solver_flag_is_reported(monkeypatch) -> None:
    """A PuLP other than the pinned one, whose solver class takes no
    _skip_v4_deprecation, is reported as a mismatch, not raised."""
    pulp = pytest.importorskip("pulp")

    def refuse(*args, **kwargs):
        raise TypeError("unexpected keyword argument '_skip_v4_deprecation'")

    monkeypatch.setattr(pulp, "PULP_CBC_CMD", refuse)
    observed = _execution_tools.host_solver_versions()["CBC"]
    assert observed == ("unavailable (TypeError: unexpected keyword argument "
                        "'_skip_v4_deprecation')")
    assert f"CBC: expected 2.10.3, observed {observed}" in _execution_tools.host_solver_mismatches()


def test_a_solver_that_prints_no_version_is_described(monkeypatch) -> None:
    # A fresh store: an earlier test may have found the real CBC's version.
    monkeypatch.setattr(reduction, "_CBC_VERSIONS", {})
    monkeypatch.setattr(
        _execution_tools.subprocess, "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(
            cmd, 3, stdout="\nCoin LP solver started\n", stderr=""),
    )
    assert _execution_tools.host_solver_versions()["CBC"] == (
        "unknown (exit code 3, 'Coin LP solver started')")
    monkeypatch.setattr(
        _execution_tools.subprocess, "run",
        lambda cmd, **kwargs: subprocess.CompletedProcess(cmd, -11, "", ""),
    )
    assert _execution_tools.host_solver_versions()["CBC"] == (
        "unknown (exit code -11, no output)")


# CBC version probes never read the caller's stdin.
#
# CBC reads commands from stdin: with an inherited open pipe `cbc -version`
# would wait until its timeout and the preflight would take the solver as
# missing.
def _recorder(calls):
    def run(argv, **options):
        calls.append(options)
        return subprocess.CompletedProcess(argv, 0, "Version: 2.10.12\n", "")
    return run


def test_the_reduction_probe_gives_cbc_no_stdin(monkeypatch):
    # A fresh store: an earlier test may have found the real CBC's version.
    monkeypatch.setattr(reduction, "_CBC_VERSIONS", {})
    calls = []
    monkeypatch.setattr(reduction.subprocess, "run", _recorder(calls))
    assert reduction.cbc_version("cbc-for-stdin-test") == ("2.10.12", "")
    assert calls[0]["stdin"] is subprocess.DEVNULL


def test_a_found_cbc_version_is_kept_and_a_failed_probe_is_not(monkeypatch):
    """The version a probe finds is kept for the process, so an exact
    cover's record probes once. A failed probe is tried again."""
    calls = []

    def run(argv, **options):
        calls.append(argv)
        found = len(calls) > 1
        return subprocess.CompletedProcess(
            argv, 0 if found else 1, "Version: 2.10.12\n" if found else "", "")

    monkeypatch.setattr(reduction, "_CBC_VERSIONS", {})
    monkeypatch.setattr(reduction.subprocess, "run", run)
    assert reduction.cbc_version("cbc-kept-test")[0] is None
    assert reduction.cbc_version("cbc-kept-test") == ("2.10.12", "")
    assert reduction.cbc_version("cbc-kept-test") == ("2.10.12", "")
    assert len(calls) == 2


def test_the_setup_report_and_the_solver_record_take_one_probe(monkeypatch):
    """The setup report and an exact reduction's record read CBC's version
    through one probe: the report shows a failed probe's reason, the
    record holds no version."""
    probed = []

    def probe(executable):
        probed.append(executable)
        return None, "unknown (exit code 1, 'no version')"

    monkeypatch.setattr(reduction, "cbc_version", probe)
    versions = _execution_tools.host_solver_versions()
    if versions["PuLP"] == "not installed":
        pytest.skip("no solver to probe in this environment")
    assert versions["CBC"] == "unknown (exit code 1, 'no version')"
    cover = reduction.exact_cover({"t::a": frozenset({"L:1"})}, {"t::a": 1.0})
    assert cover.solver_metadata["cbc_version"] is None
    assert len(probed) == 2


def test_the_host_solver_probe_gives_cbc_no_stdin(monkeypatch):
    # A fresh store: an earlier test may have found the real CBC's version.
    monkeypatch.setattr(reduction, "_CBC_VERSIONS", {})
    calls = []
    monkeypatch.setattr(_execution_tools.subprocess, "run", _recorder(calls))
    versions = _execution_tools.host_solver_versions()
    if versions.get("PuLP") == "not installed":
        return  # no solver to probe in this environment
    assert calls and calls[0]["stdin"] is subprocess.DEVNULL
