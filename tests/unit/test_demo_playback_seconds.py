"""The demonstration's playback length: --demo-playback-seconds, then the variable,
then [demo] playback_seconds of the settings files a normal run reads, then
the packaged 120 seconds."""
from __future__ import annotations

import argparse

import pytest

from testuary import __main__ as cli
from testuary.demo import cli as demo_cli


@pytest.fixture(autouse=True)
def _the_test_dir_is_the_user_dir(tmp_path, monkeypatch):
    """The test's directory is the user directory of the global config.ini."""
    from testuary import settings

    monkeypatch.setattr(settings, "platform_user_dir", lambda: tmp_path)


def _args(*options: str) -> argparse.Namespace:
    return cli._build_parser().parse_args(["--demo", *options])


def test_the_packaged_length_applies_when_nothing_sets_it(tmp_path):
    assert demo_cli._playback_seconds(_args(), environ={}, cwd=tmp_path) == 120


def test_each_source_wins_over_the_ones_below_it(tmp_path):
    (tmp_path / "config.ini").write_text("[demo]\nplayback_seconds = 30\n", encoding="utf-8")
    assert demo_cli._playback_seconds(_args(), environ={}, cwd=tmp_path) == 30
    # The project's settings file is read after the global one.
    (tmp_path / ".testuary").mkdir()
    (tmp_path / ".testuary" / "project.ini").write_text("[demo]\nplayback_seconds = 35\n",
                                                        encoding="utf-8")
    assert demo_cli._playback_seconds(_args(), environ={}, cwd=tmp_path) == 35
    environ = {demo_cli.PLAYBACK_VARIABLE: "45"}
    assert demo_cli._playback_seconds(_args(), environ=environ, cwd=tmp_path) == 45
    assert demo_cli._playback_seconds(_args("--demo-playback-seconds", "60"), environ=environ,
                                      cwd=tmp_path) == 60


def test_config_files_are_read_in_order_and_replace_the_working_dir_file(tmp_path):
    (tmp_path / "config.ini").write_text("[demo]\nplayback_seconds = 30\n", encoding="utf-8")
    first = tmp_path / "first.ini"
    first.write_text("[demo]\nplayback_seconds = 70\n[main]\nseed = 5\n", encoding="utf-8")
    second = tmp_path / "second.ini"
    second.write_text("[main]\nseed = 6\n", encoding="utf-8")
    third = tmp_path / "third.ini"
    third.write_text("[demo]\nplayback_seconds = 90\n", encoding="utf-8")
    args = _args("--config", str(first), "--config", str(second))
    # A file that does not set the key leaves the earlier one's value.
    assert demo_cli._playback_seconds(args, environ={}, cwd=tmp_path) == 70
    args = _args("--config", str(first), "--config", str(third))
    assert demo_cli._playback_seconds(args, environ={}, cwd=tmp_path) == 90
    # Explicit files replace the working directory's config.ini.
    args = _args("--config", str(second))
    assert demo_cli._playback_seconds(args, environ={}, cwd=tmp_path) == 120


@pytest.mark.parametrize("value", ["0", "-5", "soon", "nan", "inf"])
def test_a_bad_value_is_refused_naming_its_source(tmp_path, value):
    with pytest.raises(ValueError, match=demo_cli.PLAYBACK_VARIABLE):
        demo_cli._playback_seconds(_args(), environ={demo_cli.PLAYBACK_VARIABLE: value},
                                   cwd=tmp_path)
    path = tmp_path / "config.ini"
    path.write_text(f"[demo]\nplayback_seconds = {value}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="playback_seconds in .*config.ini"):
        demo_cli._playback_seconds(_args(), environ={}, cwd=tmp_path)


def test_an_unreadable_config_file_is_refused_by_name(tmp_path):
    path = tmp_path / "broken.ini"
    path.write_text("[demo]\nplayback_seconds = 10\n[demo]\nplayback_seconds = 20\n",
                    encoding="utf-8")
    with pytest.raises(ValueError, match="broken.ini cannot be read"):
        demo_cli._playback_seconds(_args("--config", str(path)), environ={}, cwd=tmp_path)


def test_the_record_says_what_the_demonstration_took(tmp_path):
    parser = cli._build_parser()
    (tmp_path / "config.ini").write_text("[demo]\nplayback_seconds = 30\n", encoding="utf-8")
    environ = {demo_cli.PLAYBACK_VARIABLE: "45", "TESTUARY_STARTER_SUITE_COUNT": "3"}
    items = {item["setting"]: item["instead"] for item in demo_cli.overridden_settings(
        parser.parse_args(["--demo"]), parser, environ, tmp_path, "alias")}
    assert items[f"environment variable {demo_cli.PLAYBACK_VARIABLE}"] == "read: the playback's length"
    assert items["environment variable TESTUARY_STARTER_SUITE_COUNT"].startswith("not read")
    assert items[str(tmp_path / "config.ini")].startswith("only [demo] playback_seconds is read")
    explicit = tmp_path / "local.ini"
    explicit.write_text("[main]\nseed = 5\n", encoding="utf-8")
    items = {item["setting"]: item["instead"] for item in demo_cli.overridden_settings(
        parser.parse_args(["--demo", "--config", str(explicit)]), parser, {}, tmp_path, "alias")}
    assert items[str(tmp_path / "config.ini")].startswith("not read")
    config_item = next(value for setting, value in items.items() if setting.startswith("--config"))
    assert "only [demo] playback_seconds is taken" in config_item


def test_without_config_options_the_named_configuration_file_is_read(tmp_path):
    """As a normal run reads it: TESTUARY_CONFIG_FILE, relative to the start
    directory, replaces config.ini there, and --config replaces both."""
    (tmp_path / "config.ini").write_text("[demo]\nplayback_seconds = 30\n", encoding="utf-8")
    (tmp_path / "named.ini").write_text("[demo]\nplayback_seconds = 50\n", encoding="utf-8")
    environ = {"TESTUARY_CONFIG_FILE": "named.ini"}
    assert demo_cli._playback_seconds(_args(), environ=environ, cwd=tmp_path) == 50
    explicit = tmp_path / "explicit.ini"
    explicit.write_text("[demo]\nplayback_seconds = 70\n", encoding="utf-8")
    assert demo_cli._playback_seconds(_args("--config", str(explicit)), environ=environ,
                                      cwd=tmp_path) == 70
    with pytest.raises(ValueError, match="absent or not a file"):
        demo_cli._playback_seconds(_args(), environ={"TESTUARY_CONFIG_FILE": "missing.ini"},
                                   cwd=tmp_path)
    parser = cli._build_parser()
    items = {item["setting"]: item["instead"] for item in demo_cli.overridden_settings(
        parser.parse_args(["--demo"]), parser, environ, tmp_path, "alias")}
    assert items["environment variable TESTUARY_CONFIG_FILE"].startswith(
        "only [demo] playback_seconds is read")
    assert items[str(tmp_path / "config.ini")].startswith("not read")
