"""Settings helpers used by ``__main__`` and the orchestrator:
``get_api_config``, ``parse_targets_spec`` (the ``--targets`` selector),
``get_project_config`` and the per-run artifact paths
(``phase_artifact_path``, ``pick_phase_artifact``).
"""
from __future__ import annotations

from pathlib import Path

import pytest

from testuary import config as config_module
from testuary import models
from testuary import project


@pytest.fixture(autouse=True)
def _the_test_dir_is_the_user_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The user directory holds config.ini and models.ini. Here it is the
    test's own directory, so a file a test writes there is the user's."""
    from testuary import settings

    monkeypatch.setattr(settings, "platform_user_dir", lambda: tmp_path)


# get_api_config: the selected alias resolved against models.ini

def test_get_api_config_merges_the_alias_over_its_provider_defaults(
    tmp_cwd: Path, in_run,
) -> None:
    """The alias's fields lie over the bundled [vllm] section: its model
    and cap win, the provider's api_url stays."""
    (tmp_cwd / "models.ini").write_text(
        "[vllm_my_model]\nprovider = vllm\n"
        "model = my-custom-model\n"
        "max_output_tokens = 4096\n",
        encoding="utf-8",
    )
    (tmp_cwd / "config.ini").write_text("[api]\nmodel = vllm_my_model\n", encoding="utf-8")
    in_run(models=None)
    cfg = models.get_api_config()
    assert (cfg["provider"], cfg["model_alias"], cfg["model"]) == ("vllm", "vllm_my_model", "my-custom-model")
    assert cfg["max_output_tokens"] == 4096
    assert cfg["api_url"] == config_module.load_config("vllm")["api_url"]


# parse_targets_spec: the --targets selector


def test_parse_targets_spec_single_target_number() -> None:
    assert config_module.parse_targets_spec("7", total=100) == [7]
    assert config_module.parse_targets_spec("1", total=10) == [1]


def test_parse_targets_spec_closed_range_inclusive() -> None:
    """`A-B` -> inclusive range from A to B."""
    assert config_module.parse_targets_spec("1-5", total=100) == [1, 2, 3, 4, 5]
    assert config_module.parse_targets_spec("6-10", total=100) == [6, 7, 8, 9, 10]


def test_parse_targets_spec_open_range_to_end() -> None:
    """`A-` selects target A through ``total``."""
    assert config_module.parse_targets_spec("5-", total=8) == [5, 6, 7, 8]


def test_parse_targets_spec_open_range_from_start() -> None:
    """`-B` selects target 1 through B."""
    assert config_module.parse_targets_spec("-5", total=100) == [1, 2, 3, 4, 5]


@pytest.mark.parametrize("spec", ["top5", "TOP3", "Top 2", "bottom5", "BOTTOM3"])
def test_parse_targets_spec_refuses_named_forms(spec) -> None:
    """Targets are selected by number and range only; topN and bottomN are
    refused, naming the accepted forms."""
    with pytest.raises(ValueError, match="is not a position or range. Use N, A-B, A- or -B$"):
        config_module.parse_targets_spec(spec, total=100)


def test_parse_targets_spec_range_clamps_to_total() -> None:
    """A range that extends past ``total`` is clamped. No error."""
    assert config_module.parse_targets_spec("6-100", total=8) == [6, 7, 8]
    assert config_module.parse_targets_spec("5-", total=7) == [5, 6, 7]


def test_parse_targets_spec_range_entirely_above_total_returns_empty() -> None:
    """`20-30` on a 10-row CSV gives an empty list, no error. The caller
    reports the empty result."""
    assert config_module.parse_targets_spec("20-30", total=10) == []


def test_parse_targets_spec_single_above_total_returns_empty() -> None:
    assert config_module.parse_targets_spec("50", total=10) == []


def test_parse_targets_spec_empty_csv_returns_empty() -> None:
    """When ``total == 0``, every spec resolves to an empty list."""
    assert config_module.parse_targets_spec("1", total=0) == []
    assert config_module.parse_targets_spec("1-5", total=0) == []
    # Open-ended ranges give [] too.
    assert config_module.parse_targets_spec("5-", total=0) == []
    assert config_module.parse_targets_spec("-5", total=0) == []


def test_parse_targets_spec_open_range_start_above_total_returns_empty() -> None:
    """``A-`` with ``A > total`` (non-zero) gives ``[]`` instead of raising:
    ``5-`` means from target 5 onward, however many exist."""
    assert config_module.parse_targets_spec("5-", total=3) == []
    assert config_module.parse_targets_spec("10-", total=8) == []


def test_parse_targets_spec_rejects_empty_or_whitespace() -> None:
    with pytest.raises(ValueError, match="no position or range given"):
        config_module.parse_targets_spec("", total=10)
    with pytest.raises(ValueError, match="no position or range given"):
        config_module.parse_targets_spec("   ", total=10)


def test_parse_targets_spec_rejects_zero_and_negative() -> None:
    with pytest.raises(ValueError, match="1 or more, not 0"):
        config_module.parse_targets_spec("0", total=10)
    with pytest.raises(ValueError, match="1 or more, not 0"):
        config_module.parse_targets_spec("0-5", total=10)


def test_parse_targets_spec_rejects_inverted_range() -> None:
    with pytest.raises(ValueError, match="is after its end"):
        config_module.parse_targets_spec("5-3", total=10)


def test_parse_targets_spec_rejects_unrecognized_form() -> None:
    with pytest.raises(ValueError, match="is not a position or range"):
        config_module.parse_targets_spec("foo", total=10)
    with pytest.raises(ValueError, match="is not a position or range"):
        config_module.parse_targets_spec("bottom-3", total=10)


def test_parse_targets_spec_rejects_bare_dash() -> None:
    """`-` alone is ambiguous (the whole range or nothing), so it is refused."""
    with pytest.raises(ValueError, match="alone is not a range"):
        config_module.parse_targets_spec("-", total=10)


# get_project_config: the [project] settings of the project's settings file


def _project_settings(project_dir, text: str) -> None:
    (project_dir / ".testuary").mkdir(exist_ok=True)
    (project_dir / ".testuary" / "project.ini").write_bytes(text.encode("utf-8"))


def test_get_project_config_reads_the_projects_settings_file(tmp_cwd, in_run) -> None:
    """The project's settings file holds its [project] settings, with the
    files and language a run without named files takes."""
    _project_settings(tmp_cwd, "[project]\nhint_profile = runtime\nsource_files = a.py\n"
                               "language = python\n\n[api]\nmodel = chosen\n")
    in_run(base_dir=tmp_cwd, project_settings=True)
    config = config_module.get_project_config()
    assert (config["hint_profile"], config["source_files"], config["language"]) == (
        "runtime", "a.py", "python")


def test_an_unknown_project_setting_is_refused_with_the_closest_name(tmp_cwd) -> None:
    from testuary.settings import resolve_settings

    _project_settings(tmp_cwd, "[project]\nhint_profil = runtime\n")
    with pytest.raises(ValueError, match=r"project\.ini: \[project\]: unknown setting "
                                         r"'hint_profil'\. The nearest known setting is 'hint_profile'"):
        resolve_settings(base_dir=tmp_cwd, environment={})


def test_check_reports_an_unknown_project_setting_as_a_refusal(tmp_path, monkeypatch, capsys) -> None:
    """check --json refuses with exit 2 and its normal report, not a
    traceback."""
    import json
    import sys as _sys

    from testuary import __main__ as cli

    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    (tmp_path / "config.ini").write_bytes(b"[api]\nmodel = m\n")
    (tmp_path / "models.ini").write_bytes(
        b"[m]\nprovider = openai\nmodel = x\napi_url = https://example.test/v1\n")
    _project_settings(tmp_path, "[project]\nhint_profil = runtime\n")
    (tmp_path / "mod.py").write_bytes(b"x = 1\n")
    monkeypatch.setattr(_sys, "argv", ["testuary", "check", "--json", "mod.py"])
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["exit_code"] == 2
    assert any("unknown setting 'hint_profil'" in item["reason"] for item in report["items"])


HOST_REFUSAL = ("LangAdapter runs a project's commands on Windows and Linux, where it "
                "supervises them and confirms that every process a command started has "
                "ended, and on other POSIX systems such as macOS in the command's own "
                "process group, without that confirmation. wasi has no processes to run a "
                "command in. Importing LangAdapter and its source operations, which run no "
                "command, work here, with a context created without an interpreter.")


@pytest.mark.parametrize("verify", [False, True])
def test_check_reports_a_host_langadapter_refuses_as_a_refusal(tmp_path, monkeypatch, capsys, verify):
    """LangAdapter refuses a project interpreter on a host it runs no command on
    (wasi): check prints its refusal report with the reason, not a traceback."""
    import json
    import sys as _sys

    import langadapter
    from langadapter import InvalidRequest
    from testuary import __main__ as cli

    def refuse(language, request):
        raise InvalidRequest("execution", HOST_REFUSAL)

    monkeypatch.setattr(langadapter, "inspect_project", refuse)
    monkeypatch.setattr(langadapter, "verify_project", refuse)
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("TESTUARY_TIMING_RUN_COUNT", raising=False)
    monkeypatch.setenv("TESTUARY_OPENAI_API_KEY", "synthetic-fixture")
    python = _sys.executable.replace("\\", "/")
    (tmp_path / "config.ini").write_bytes(
        f"[api]\nmodel = m\n[project]\npython_executable = {python}\n".encode())
    (tmp_path / "models.ini").write_bytes(
        b"[m]\nprovider = openai\nmodel = x\napi_url = https://example.test/v1\n")
    (tmp_path / "projects.ini").write_bytes(b"[demo]\n")
    (tmp_path / "projects" / "demo").mkdir(parents=True)
    (tmp_path / "projects" / "demo" / "mod.py").write_bytes(b"x = 1\n")
    monkeypatch.setattr(_sys, "argv", ["testuary", "check", "--json", "--project-name", "demo",
                                       "--source-files", "mod.py"] + (["--verify"] if verify else []))
    with pytest.raises(SystemExit) as stopped:
        cli._main()
    assert stopped.value.code == 2
    report = json.loads(capsys.readouterr().out)
    assert report["exit_code"] == 2
    assert [item["reason"] for item in report["items"]] == [f"execution: {HOST_REFUSAL}"]


def test_get_project_config_does_not_expose_a_target_selection(tmp_cwd, in_run) -> None:
    """Target selection is a per-run CLI choice, not project metadata, so the
    project settings have no ``ranks`` key. Callers read ``args.targets``."""
    in_run()
    cfg = config_module.get_project_config()
    assert "ranks" not in cfg


# phase_artifact_path and pick_phase_artifact


def test_phase_artifact_path_places_the_first_run_in_r0(tmp_cwd: Path) -> None:
    """Run 1, a single run too, lands in r0, files and directories alike."""
    assert project.phase_artifact_path(
        tmp_cwd, "coverage.json", run_idx=0,
    ) == tmp_cwd / "r0" / "coverage.json"
    assert project.phase_artifact_path(
        tmp_cwd, "tests", run_idx=0,
    ) == tmp_cwd / "r0" / "tests"


def test_phase_artifact_path_places_later_runs_in_their_dirs(tmp_cwd: Path) -> None:
    """Run i lands in r(i-1) under the artifact's own name."""
    assert project.phase_artifact_path(
        tmp_cwd, "coverage.json", run_idx=2,
    ) == tmp_cwd / "r2" / "coverage.json"
    assert project.phase_artifact_path(
        tmp_cwd, "tests", run_idx=1,
    ) == tmp_cwd / "r1" / "tests"


def test_run_idx_from_artifact_path_inverts_run_dir() -> None:
    """A run directory's name gives its run; any other directory is the first run's,
    whatever the directories above it are called."""
    from pathlib import Path as _P
    for i in (0, 1, 7):
        assert project.run_idx_from_artifact_path(project.run_dir(_P("/out"), i)) == i
    assert project.run_idx_from_artifact_path(_P("r9")) == 9
    # A name that merely starts with r is no run directory.
    assert project.run_idx_from_artifact_path(_P("/x/results")) == 0
    # A directory above the results named like a run directory is not the run.
    assert project.run_idx_from_artifact_path(_P("/home/r2/results/tests")) == 0


def test_pick_phase_artifact_reads_the_runs_own_artifact(tmp_cwd: Path) -> None:
    """When both its own and the first run's exist, a run reads its own."""
    (tmp_cwd / "r0" / "tests").mkdir(parents=True)
    (tmp_cwd / "r1" / "tests").mkdir(parents=True)
    picked = project.pick_phase_artifact(
        tmp_cwd, "tests", current_run_idx=1,
    )
    assert picked == tmp_cwd / "r1" / "tests"


def test_pick_phase_artifact_falls_back_to_the_first_run(tmp_cwd: Path) -> None:
    """Without its own, a run reads the first run's, never another's."""
    (tmp_cwd / "r0" / "tests").mkdir(parents=True)
    (tmp_cwd / "r1" / "tests").mkdir(parents=True)
    picked = project.pick_phase_artifact(
        tmp_cwd, "tests", current_run_idx=2,
    )
    assert picked == tmp_cwd / "r0" / "tests"


def test_pick_phase_artifact_raises_when_neither_exists(
    tmp_cwd: Path,
) -> None:
    """No cascade through earlier runs: if neither the run's own nor the
    first run's artifact exists, the error names both paths."""
    with pytest.raises(FileNotFoundError, match="neither exists"):
        project.pick_phase_artifact(
            tmp_cwd, "tests", current_run_idx=2,
        )


