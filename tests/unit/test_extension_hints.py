"""An extension's ``testuary_hints``, a mapping in the hints file's format.
A run captures it once at its start, puts it before the user's hints in
each prompt in the combined list's order and records it in the generation
origin with the extensions it loaded, as it does for the user's file.
"""
from __future__ import annotations

import hashlib
import json
import os
from types import SimpleNamespace

import pytest

import test_generation_origin_callers as origin_harness
from testuary import _extensions, config, hints, orchestrator
from testuary.evidence import copy_origin, current_condition, write_origin
from testuary.settings import ConfigurationError, RunSettings, extension_items

EXTENSION_HINTS = {"initial": {"per_language": {"python": "Name the module the way the package does."}}}
USER_HINTS = {"initial": {"general": "Keep each test short."}}


def _extension(path, value):
    """An extension file at ``path`` whose ``testuary_hints`` is ``value``,
    written as Python source."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"testuary_hints = {value!r}\n", encoding="utf-8")


def _settings(root, extensions="ext.py", hints_file=""):
    return RunSettings(base_dir=root, models=origin_harness.settings_at(root).models,
                       environment=dict(os.environ), sections={
                           "api": {"model": "explicit"},
                           "main": {"hints_file": hints_file, "seed": 37, "extensions": extensions}})


def _in_run(monkeypatch, settings, body):
    """What ``body`` returns inside a run that run_pipeline starts with
    ``settings``, which loads their extensions and captures their hints."""
    seen = []
    monkeypatch.setattr(orchestrator, "_run_pipeline", lambda args, languages: seen.append(body()) or 0)
    assert orchestrator.run_pipeline(SimpleNamespace(), settings=settings) == 0
    return seen[0]


def _origin(path):
    """The generation origin of a starter attempt written to ``path``, and
    its condition."""
    def body():
        condition = current_condition("starter", "python", str(path.parent.parent),
                                      templates={"system_prompt": "s", "user_prompt_template": "u"})
        write_origin(path, condition)
        return condition
    return body


def test_extension_cells_come_before_the_users(tmp_path, monkeypatch):
    _extension(tmp_path / "ext.py", EXTENSION_HINTS)
    (tmp_path / "hints.json").write_text(json.dumps(USER_HINTS), encoding="utf-8")
    composed = _in_run(monkeypatch, _settings(tmp_path, hints_file="hints.json"),
                       lambda: hints.resolve_hints(None, "python", phase="initial"))
    assert composed == "Name the module the way the package does.\n\nKeep each test short."


def test_two_extensions_in_the_combined_lists_order_and_two_files_of_one_name(tmp_path, monkeypatch):
    for name in ("a.py", "one/hints.py", "two/hints.py"):
        _extension(tmp_path / name, {"coverage": {"per_language": {"python": name}}})
    (tmp_path / "hints.json").write_text(json.dumps({"coverage": {"general": "the project's"}}),
                                         encoding="utf-8")
    composed = _in_run(monkeypatch, _settings(tmp_path, "a.py, one/hints.py, two/hints.py", "hints.json"),
                       lambda: hints.resolve_hints(None, "python", phase="coverage"))
    assert composed.split("\n\n") == ["a.py", "one/hints.py", "two/hints.py", "the project's"]


def test_extension_hints_are_captured_at_the_runs_start(tmp_path, monkeypatch):
    _extension(tmp_path / "ext.py", EXTENSION_HINTS)

    def body():
        loaded = _extensions.loaded(extension_items(config.active_settings()))
        loaded[0][3].testuary_hints = {"initial": {"general": "changed during the run"}}
        return hints.resolve_hints(None, "python", phase="initial")

    composed = _in_run(monkeypatch, _settings(tmp_path), body)
    assert composed == "Name the module the way the package does."


def test_a_mapping_json_cannot_hold_refuses_naming_the_item(tmp_path):
    (tmp_path / "ext.py").write_text("testuary_hints = {'initial': {'general': {1, 2}}}\n",
                                     encoding="utf-8")
    with pytest.raises(ConfigurationError, match=r"ext\.py: testuary_hints is not JSON: .*set.* "
                                                 r"It is named in the program's overrides$"):
        hints.extension_hints(_settings(tmp_path))


def test_a_read_only_mapping_is_captured_as_a_dict_would_be(tmp_path):
    (tmp_path / "ext.py").write_text(
        "from types import MappingProxyType\n"
        f"testuary_hints = MappingProxyType({EXTENSION_HINTS!r})\n", encoding="utf-8")
    [found] = hints.extension_hints(_settings(tmp_path))
    assert json.loads(found.contents) == EXTENSION_HINTS


def test_invalid_extension_hints_refuse_naming_the_item_and_the_cell(tmp_path, monkeypatch):
    _extension(tmp_path / "ext.py", {"initial": {"generl": "a typo in the cell"}})
    with pytest.raises(ConfigurationError, match=r"ext\.py testuary_hints: unknown cell "
                                                 r"initial\.'generl'(.|\n)*It is named in the "
                                                 r"program's overrides$"):
        _in_run(monkeypatch, _settings(tmp_path), lambda: pytest.fail("the run started"))


def test_the_origin_records_the_extensions_and_a_reuse_carries_their_hints(tmp_path, monkeypatch):
    _extension(tmp_path / "ext.py", EXTENSION_HINTS)
    origin = tmp_path / "origin" / "run_1.json"
    condition = _in_run(monkeypatch, _settings(tmp_path), _origin(origin))
    record = json.loads(origin.read_text(encoding="utf-8"))
    [extension] = record["extensions"]
    assert extension["item"] == "ext.py" and extension["file"] == str((tmp_path / "ext.py").resolve())
    assert (extension["distribution"], extension["version"]) == (None, None)
    assert len(extension["sha256"]) == 64
    [item] = record["extension_hints"]
    assert item["item"] == "ext.py"
    assert json.loads((origin.parent / item["copy"]).read_bytes()) == EXTENSION_HINTS
    reused = tmp_path / "reused" / "run_1.json"
    copy_origin(origin, reused, condition)
    [carried] = json.loads(reused.read_text(encoding="utf-8"))["extension_hints"]
    assert json.loads((reused.parent / carried["copy"]).read_bytes()) == EXTENSION_HINTS
    assert carried["sha256"] == item["sha256"]


def test_an_origin_recorded_under_package_hints_keeps_its_copies_on_reuse(tmp_path, monkeypatch):
    origin = tmp_path / "origin" / "run_1.json"
    condition = _in_run(monkeypatch, _settings(tmp_path, extensions=""), _origin(origin))
    record = json.loads(origin.read_text(encoding="utf-8"))
    copy = origin.parent / "run_1.package_hints_1.json"
    copy.write_text(json.dumps(EXTENSION_HINTS), encoding="utf-8")
    record["package_hints"] = [{"name": "toy", "distribution": "toy-hints", "path": "toy:hints.json",
                                "sha256": hashlib.sha256(copy.read_bytes()).hexdigest(),
                                "copy": copy.name}]
    origin.write_text(json.dumps(record), encoding="utf-8")
    reused = tmp_path / "reused" / "run_1.json"
    copy_origin(origin, reused, condition)
    [carried] = json.loads(reused.read_text(encoding="utf-8"))["package_hints"]
    assert json.loads((reused.parent / carried["copy"]).read_bytes()) == EXTENSION_HINTS
    again = tmp_path / "again" / "run_1.json"
    copy_origin(reused, again, condition)
    assert json.loads(again.read_text(encoding="utf-8"))["package_hints"][0]["name"] == "toy"


def test_without_extensions_the_origin_keeps_its_shape(tmp_path, monkeypatch):
    origin = tmp_path / "origin" / "run_1.json"
    _in_run(monkeypatch, _settings(tmp_path, extensions=""), _origin(origin))
    record = json.loads(origin.read_text(encoding="utf-8"))
    assert not {"extensions", "extension_hints", "package_hints"} & set(record)


def test_the_public_module_finds_what_a_run_reads(tmp_path, monkeypatch):
    _extension(tmp_path / "ext.py", EXTENSION_HINTS)
    [found] = hints.extension_hints(_settings(tmp_path))
    assert (found.item, found.file) == ("ext.py", str((tmp_path / "ext.py").resolve()))
    request = _in_run(monkeypatch, _settings(tmp_path),
                      lambda: hints.resolve_hints(None, "python", phase="initial"))
    assert hints.request_text(found, "python", "initial") == request
    assert hints.request_text(EXTENSION_HINTS, "python", "initial") == request
    assert hints.request_text(found, "java", "initial") == ""


def test_the_public_module_selects_the_profile_cells():
    profiled = {"coverage": {"general": "g", "per_profile": {"p": "pp"},
                             "per_profile_per_language": {"p": {"python": "ppl"}}}}
    assert hints.request_text(profiled, "python", "coverage") == "g"
    assert hints.request_text(profiled, "python", "coverage", hint_profile="p") == "g\n\npp\n\nppl"


def test_the_public_module_refuses_unknown_phases():
    with pytest.raises(ValueError, match="unknown phase 'initail'"):
        hints.request_text(EXTENSION_HINTS, "python", "initail")


def test_a_starter_attempts_evidence_carries_the_extension_hints(tmp_path, monkeypatch):
    from testuary.generation_evidence import capture_origin

    _extension(tmp_path / "ext.py", EXTENSION_HINTS)
    origin = tmp_path / "origin" / "run_1.json"
    _in_run(monkeypatch, _settings(tmp_path), _origin(origin))
    [captured] = capture_origin(origin)["inputs"]["extension_hints"]
    assert captured["item"] == "ext.py" and captured["base"] == "inline"
    assert json.loads(captured["contents"]) == EXTENSION_HINTS
    assert captured["sha256"] == hashlib.sha256(captured["contents"].encode("utf-8")).hexdigest()


def test_evidence_recorded_before_extension_hints_stays_valid(tmp_path, monkeypatch):
    """Evidence with only the prompt profile and the hints file is read as
    valid. An origin's package_hints are not copied into the evidence."""
    from testuary.generation_evidence import capture_origin

    origin = tmp_path / "origin" / "run_1.json"
    _in_run(monkeypatch, _settings(tmp_path, extensions=""), _origin(origin))
    record = json.loads(origin.read_text(encoding="utf-8"))
    record["package_hints"] = []
    origin.write_text(json.dumps(record), encoding="utf-8")
    assert set(capture_origin(origin)["inputs"]) == {"prompt_profile", "hints"}


def test_evidence_without_extension_hints_does_not_stand_in_for_a_run_with_them(tmp_path, monkeypatch):
    """The extensions' hints are part of the composed prompts, so a reuse
    of a starter attempt recorded without them is refused as for a changed
    hints file."""
    from testuary.evidence import reuse_problems

    _extension(tmp_path / "ext.py", EXTENSION_HINTS)

    def composed():
        return current_condition("starter", "python", str(tmp_path))

    without = _in_run(monkeypatch, _settings(tmp_path, extensions=""), composed)
    with_hints = _in_run(monkeypatch, _settings(tmp_path), composed)
    assert reuse_problems(without.export(), with_hints) == ("prompt_composition changed",)
