"""A small first run is a few lines in the project's settings file. No
quickstart limits profile, option, override or layer exists, and a run
recorded under that profile still resumes on the same limits."""
from __future__ import annotations

import configparser
import importlib.util
from importlib import resources

import pytest

from testuary import __main__ as cli, settings
from testuary import config
from testuary.evidence import current_condition, reuse_problems
from testuary.settings import resolve_settings

#: The values the quickstart profile set, which the demonstrations and the
#: tutorial recordings set through their settings.
FORMER_PROFILE = {
    "starter": {
        "suite_count": "3", "attempts_per_suite": "2", "max_output_tokens": "32768",
        "request_timeout_seconds": "600", "max_consecutive_transient_failures": "3",
        "max_transient_failures_per_phase": "6", "max_transient_hint_seconds": "300",
    },
    **{section: {
        "max_requests_per_phase": "20", "max_loop_attempts": "5", "max_output_tokens": "32768",
        "max_input_tokens_per_phase": "300000", "max_output_tokens_per_phase": "200000",
        "max_total_tokens_per_phase": "500000", "max_wall_clock_seconds_per_phase": "1800",
        "request_timeout_seconds": "600", "max_consecutive_transient_failures": "3",
        "max_transient_failures_per_phase": "6", "max_transient_hint_seconds": "300",
    } for section in ("coverage", "mutation")},
}


def _project_settings(project_dir):
    """The project's settings file in ``project_dir``, where the guide's lines go."""
    (project_dir / ".testuary").mkdir(exist_ok=True)
    return project_dir / ".testuary" / "project.ini"


def _resolve(**options):
    options.setdefault("config_files", [])
    return resolve_settings(models={}, **{"environment": {}, **options})


def _ini(sections) -> str:
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_dict(sections)
    lines = []
    for section in parser.sections():
        lines.append(f"[{section}]")
        lines.extend(f"{key} = {value}" for key, value in parser[section].items())
        lines.append("")
    return "\n".join(lines)


def test_the_command_line_has_no_limits_profile(capsys):
    with pytest.raises(SystemExit) as stopped:
        cli._build_parser().parse_args(["--limits-profile", "quickstart"])
    assert stopped.value.code == 2
    assert "--limits-profile" in capsys.readouterr().err


def test_the_package_ships_no_limits_profile():
    assert not resources.files("testuary").joinpath("profiles", "quickstart.ini").is_file()
    assert importlib.util.find_spec("testuary._limits_profile") is None


@pytest.mark.parametrize("layer", ["overrides", "file"])
def test_a_limits_profile_setting_is_an_unknown_setting(tmp_path, layer):
    options = {"overrides": {"main": {"limits_profile": "quickstart"}}}
    if layer == "file":
        _project_settings(tmp_path).write_text("[main]\nlimits_profile = quickstart\n",
                                             encoding="utf-8")
        options = {"base_dir": tmp_path, "config_files": None}
    with pytest.raises(ValueError, match="unknown setting 'limits_profile'"):
        _resolve(**options)


def test_the_former_profile_values_set_in_a_settings_file_resolve_unchanged(tmp_path):
    _project_settings(tmp_path).write_text(_ini(FORMER_PROFILE), encoding="utf-8")
    settings = _resolve(base_dir=tmp_path, config_files=None)
    for section, fields in FORMER_PROFILE.items():
        assert {key: settings.sections[section][key] for key in fields} == fields
    assert [name for name, _ in settings.configuration_layers()][0] == "defaults"
    assert not any(name.startswith("profile:") for name, _ in settings.configuration_layers())


def test_the_guides_small_run_lines_resolve_to_the_former_values(tmp_path):
    """The lines getting started shows for a first run are settings a file
    can hold, and each value is the one the profile set."""
    import re
    from pathlib import Path

    guide = (Path(__file__).resolve().parents[2] / "docs" / "getting-started.md").read_text(
        encoding="utf-8")
    section = guide.split("## Run with small limits", 1)[1].split("\n## ", 1)[0]
    lines = re.findall(r"```ini\n(.*?)```", section, re.S)[0]
    _project_settings(tmp_path).write_text(lines, encoding="utf-8")
    settings = _resolve(base_dir=tmp_path, config_files=None)
    shown = configparser.ConfigParser(interpolation=None)
    shown.read_string(lines)
    assert shown.sections() == ["starter", "coverage", "mutation"]
    for section_name in shown.sections():
        for key, value in shown[section_name].items():
            assert FORMER_PROFILE[section_name][key] == value, (section_name, key)
            assert settings.sections[section_name][key] == value


def test_a_record_naming_the_removed_profile_is_refused(tmp_path):
    """A generation record that names the removed limits profile is an
    earlier version's and is refused. The limits themselves are compared:
    equal limits resume, other limits refuse."""
    models = {"chosen": {"provider": "openai", "model": "fixture",
                         "api_url": "https://example.test/v1"}}
    settings = resolve_settings(config_files=[], base_dir=tmp_path, environment={},
                                models=models,
                                overrides={"api": {"model": "chosen"}, **FORMER_PROFILE})
    with config.use_run_settings(settings):
        condition = current_condition("coverage", "python", str(tmp_path), templates={
            "system_prompt": "s", "user_prompt_template": "u", "tool_messages": {}})
    exported = condition.export()
    assert "limits_profile" not in exported
    older = {**exported, "limits_profile": "quickstart"}
    assert "generation condition has unsupported fields" in reuse_problems(older, condition)
    assert reuse_problems(exported, condition) == ()
    # Other limits still refuse the resume.
    changed = {**exported, "limits": {**exported["limits"], "max_requests_per_phase": "21"}}
    assert any("limits changed" in problem for problem in reuse_problems(changed, condition))


def test_the_resolved_settings_name_no_limits_profile():
    with config.use_run_settings(_resolve(overrides=FORMER_PROFILE)):
        resolved = config.resolved_test_preset()
    assert "limits_profile" not in resolved["sections"]["main"]
    assert resolved["sections"]["starter"]["suite_count"] == "3"


def test_the_former_values_leave_a_reasoning_model_room_to_answer():
    """A reasoning model can spend all of an 8192-token reply on reasoning and
    give no text. The small run's lines allow 32768 per reply, a request lasts
    long enough for such a reply, and each editing loop's output cap holds
    several of them."""
    sections = _resolve(overrides=FORMER_PROFILE).sections
    for phase in ("starter", "coverage", "mutation"):
        assert int(sections[phase]["max_output_tokens"]) >= 32768
        assert float(sections[phase]["request_timeout_seconds"]) >= 600
    for phase in ("coverage", "mutation"):
        per_reply = int(sections[phase]["max_output_tokens"])
        assert int(sections[phase]["max_output_tokens_per_phase"]) >= 6 * per_reply


def test_no_default_route_caps_a_reply(tmp_path):
    """No phase caps a reply by default, and the setup wizard writes no cap."""
    import test_setup_wizard as wizard

    sections = _resolve().sections
    for phase in ("starter", "coverage", "mutation"):
        assert int(sections[phase].get("max_output_tokens") or 0) == 0
    project = tmp_path / "proj" / "src" / "pkg"
    project.mkdir(parents=True)
    (project / "core.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    wizard._run(tmp_path, {"model.name": "qwen3:8b", "project.folder": "proj"})
    for name in ("models.ini", "config.ini"):
        written = configparser.ConfigParser(interpolation=None)
        written.read(settings.platform_user_dir() / name, encoding="utf-8")
        assert not any("max_output_tokens" in written[section] for section in written.sections())


def test_a_failed_starter_slot_is_generated_again_by_default():
    """The packaged default refills a failed slot within
    attempts_per_suite."""
    starter = _resolve().sections["starter"]
    assert starter["refill_failed_suites"] == "true" and starter["attempts_per_suite"] == "4"
