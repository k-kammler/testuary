"""Demonstration data beside the languages' own: a variant such as the
tutorial's run is selected with --demo-data, and data may carry the prompt
profile and hints file their run read, which the playback writes into its
workspace and names there."""
from __future__ import annotations

import configparser
import json
import sys
from pathlib import Path

import pytest

from testuary import __main__ as entry
from testuary.demo import cli as demo_cli
from testuary.demo import cli as labels


def _manifest(data_dir: Path, **fields) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "manifest.json").write_text(
        json.dumps({"files": {}, "classes": {"real": 1}, **fields}), encoding="utf-8")


@pytest.fixture
def data_root(tmp_path, monkeypatch):
    """A package whose data hold Python's, Java's and a variant's."""
    package = tmp_path / "package"
    data = package / "data"
    _manifest(data, language="python")
    _manifest(data / "java", language="java")
    _manifest(data / "variant", language="python", dataset="variant")
    monkeypatch.setattr(demo_cli.resources, "files", lambda name: package)
    return data


def test_a_variant_is_data_but_not_a_language(data_root):
    assert demo_cli.datasets() == ["python", "java", "variant"]
    assert demo_cli.languages() == ["python", "java"]
    manifest, files = demo_cli.load_data("variant")
    assert manifest["dataset"] == "variant" and manifest["language"] == "python" and files == {}


def test_data_whose_manifest_names_other_data_are_refused(data_root):
    _manifest(data_root / "variant", language="python")
    with pytest.raises(demo_cli.DataMismatch):
        demo_cli.load_data("variant")


def test_the_playback_writes_carried_files_and_names_them_in_its_workspace(tmp_path):
    files = {"prompt_profile.json": b'{"format": 1}\n', "hints.json": b"{}\n"}
    manifest = {"settings_files": {"prompt_profile": "prompt_profile.json",
                                   "hints_file": "hints.json"}}
    raw = b"[main]\nprompt_profile = prompt_profile.json\nhints_file = hints.json\n"
    preset = demo_cli._with_settings_files(raw, tmp_path, manifest, files)
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string(preset.decode("utf-8"))
    assert parser["main"]["prompt_profile"] == str(tmp_path / "prompt_profile.json")
    assert parser["main"]["hints_file"] == str(tmp_path / "hints.json")
    assert (tmp_path / "hints.json").read_bytes() == b"{}\n"
    # Data without carried files keep their preset as recorded.
    assert demo_cli._with_settings_files(raw, tmp_path, {}, files) == raw
    with pytest.raises(demo_cli.DataMismatch):
        demo_cli._with_settings_files(raw, tmp_path, {"settings_files": {"other": "x"}}, files)


def test_carried_files_stay_as_read_when_the_outputs_are_labeled(tmp_path):
    hints = tmp_path / "hints.json"
    hints.write_text("{}\n", encoding="utf-8")
    other = tmp_path / "record.json"
    other.write_text("{}\n", encoding="utf-8")
    labels.label_outputs(tmp_path, inputs=[hints])
    assert hints.read_text(encoding="utf-8") == "{}\n"
    assert other.read_text(encoding="utf-8") != "{}\n"
    assert "hints.json" in (tmp_path / labels.NOTICE_NAME).read_text(encoding="utf-8")


def _main(argv, monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["testuary", *argv])
    with pytest.raises(SystemExit) as stopped:
        entry._main()
    return stopped.value.code, capsys.readouterr()


def test_demo_data_needs_the_demonstration_and_no_language(monkeypatch, capsys):
    code, out = _main(["--demo-data", "tutorial"], monkeypatch, capsys)
    assert code == 2 and "--demo-data requires --demo" in out.err
    code, out = _main(["--demo", "--demo-data", "tutorial", "--demo-language", "java"],
                      monkeypatch, capsys)
    assert code == 2 and "Drop --demo-language" in out.err


def test_unknown_demo_data_are_refused_with_the_installed_ones(tmp_path, capsys, monkeypatch):
    monkeypatch.setattr(demo_cli, "phase_modules_loaded", lambda: [])
    parser = entry._build_parser()
    args = parser.parse_args(["--demo", "--demo-data", "nothing", "--output-dir", str(tmp_path / "o")])
    assert demo_cli.main(args, parser) == demo_cli.EXIT_USAGE
    err = capsys.readouterr().err
    assert "no demonstration data 'nothing'" in err and "python" in err
    assert not (tmp_path / "o").exists()
