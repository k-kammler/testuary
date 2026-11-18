"""A profile input is captured once; it does not choose or activate a run."""
from dataclasses import FrozenInstanceError
import hashlib
import json
from pathlib import Path

import pytest

from testuary import prompts as _prompt_profiles
from testuary.prompts import ProfileSource, read_profile
from testuary.settings import ConfigurationError


@pytest.fixture
def templates():
    return {phase: {
        "system_prompt": "## Hard rules\nUse $lang_name\n",
        "user_prompt_template": "$file_path\n$source",
        "tool_messages": {"tests_timeout": "Timeout after $timeout"},
    } for phase in ("initial", "coverage", "mutation")}


def _read(path, templates, base_dir):
    return read_profile(path, templates, base_dir=base_dir,
                        languages=("python", "java"), fragments=("lang_name",))


@pytest.mark.parametrize("absolute", [False, True])
def test_profile_source_uses_the_explicit_base_after_chdir(monkeypatch, tmp_path, templates, absolute):
    base = tmp_path / "base"
    other = tmp_path / "other"
    base.mkdir()
    other.mkdir()
    selected = base / "profile.json"
    selected.write_bytes(b'{"format":1,"name":"chosen","phases":{}}\r\n')
    (other / selected.name).write_bytes(b'{"format":1,"name":"wrong","phases":{}}')
    monkeypatch.chdir(other)
    monkeypatch.setenv("TESTUARY_MAIN_PROMPT_PROFILE", str(other / selected.name))
    captured = _read(selected if absolute else selected.name, templates, base)
    assert captured.path == selected
    assert captured.profile.name == "chosen"


def test_profile_source_refuses_an_ambient_relative_base(tmp_path, templates):
    with pytest.raises(ConfigurationError, match="base_dir must be absolute"):
        _read("profile.json", templates, Path("relative"))


def test_profile_source_is_read_once_and_preserves_bytes_after_replacement(
        monkeypatch, tmp_path, templates):
    selected = tmp_path / "profile.json"
    contents = (b'{\r\n "format":1, "name":"caf\\u00e9",\r\n "phases":{'
                b'"coverage":{"system_prompt":{"append":"Captured"}}}\r\n}\r\n')
    selected.write_bytes(contents)
    original_read = Path.read_bytes
    reads = []

    def read_once(path):
        reads.append(path)
        data = original_read(path)
        path.write_bytes(b'{"format":1,"name":"later","phases":{}}')
        return data

    monkeypatch.setattr(Path, "read_bytes", read_once)
    captured = _read(selected.name, templates, tmp_path)
    selected.unlink()
    assert reads == [selected]
    assert captured.path == selected
    assert captured.contents == contents
    assert captured.sha256 == hashlib.sha256(contents).hexdigest()
    assert captured.profile.name == "caf\u00e9"
    assert captured.profile.compose("coverage", "python", templates["coverage"])[
        "system_prompt"].endswith("\n\nCaptured")
    assert reads == [selected]
    assert "Captured" not in repr(captured)


def test_profile_source_and_nested_operations_are_immutable(tmp_path, templates):
    selected = tmp_path / "profile.json"
    selected.write_text(json.dumps({"format": 1, "phases": {
        "initial": {"system_prompt": {"append": "Initial only"}},
    }}), encoding="utf-8")
    captured = _read(selected.name, templates, tmp_path)
    with pytest.raises(FrozenInstanceError):
        captured.contents = b"changed"
    with pytest.raises(FrozenInstanceError):
        captured.path = tmp_path / "other.json"
    with pytest.raises(TypeError):
        captured.profile.operations("initial", "python")["system_prompt"][""] = ("append", "Changed")
    detached = captured.profile.compose("initial", "python", templates["initial"])
    detached["system_prompt"] = "Changed"
    assert captured.profile.compose("initial", "python", templates["initial"])[
        "system_prompt"].endswith("Initial only")


@pytest.mark.parametrize("field,value,error", [
    ("path", Path("relative.json"), ValueError),
    ("contents", bytearray(b"{}"), TypeError),
    ("profile", {}, TypeError),
])
def test_profile_source_constructor_refuses_mutable_or_unvalidated_inputs(
        tmp_path, templates, field, value, error):
    selected = tmp_path / "selected.json"
    selected.write_bytes(b'{"format":1,"phases":{}}')
    captured = _read(selected.name, templates, tmp_path)
    fields = {"path": captured.path, "contents": captured.contents, "profile": captured.profile}
    fields[field] = value
    with pytest.raises(error):
        ProfileSource(**fields)


def test_profile_source_preserves_a_utf8_bom_in_recordable_bytes(tmp_path, templates):
    selected = tmp_path / "selected.json"
    contents = b'\xef\xbb\xbf{"format":1,"phases":{}}'
    selected.write_bytes(contents)
    captured = _read(selected.name, templates, tmp_path)
    assert captured.contents == contents
    assert captured.sha256 == hashlib.sha256(contents).hexdigest()
    assert captured.profile.compose("initial", "python", templates["initial"]) == templates["initial"]


@pytest.mark.parametrize("failure", [FileNotFoundError, PermissionError, IsADirectoryError])
def test_profile_source_read_failure_has_no_fallback_or_content_echo(
        monkeypatch, tmp_path, templates, failure):
    selected = tmp_path / "selected.json"

    def fail(path):
        raise failure("private filesystem detail")

    monkeypatch.setattr(Path, "read_bytes", fail)
    with pytest.raises(ConfigurationError) as caught:
        _read(selected.name, templates, tmp_path)
    message = str(caught.value)
    assert str(selected) in message
    assert "private filesystem detail" not in message
    assert "absent" in message if failure is FileNotFoundError else "cannot read" in message
    assert caught.value.__cause__ is None


@pytest.mark.parametrize("contents", [
    b"", b"private malformed input", b'{"format":1,"phases":', b"\xff",
    b'{"format":1,"phases":{},"_note":NaN}',
    b'{"format":1,"phases":{},"_note":Infinity}',
    b'{"format":1,"phases":{},"_note":-Infinity}',
])
def test_profile_source_refuses_invalid_json_without_echoing_it(tmp_path, templates, contents):
    selected = tmp_path / "selected.json"
    selected.write_bytes(contents)
    with pytest.raises(ConfigurationError, match="valid UTF-8 JSON") as caught:
        _read(selected.name, templates, tmp_path)
    assert "private malformed input" not in str(caught.value)
    assert caught.value.__cause__ is None


def test_profile_source_refuses_a_parser_depth_failure_without_echoing_it(
        monkeypatch, tmp_path, templates):
    selected = tmp_path / "selected.json"
    selected.write_bytes(b'{"format":1,"phases":{}}')

    def fail(*args, **kwargs):
        raise RecursionError("private parser detail")

    monkeypatch.setattr(_prompt_profiles.json, "loads", fail)
    with pytest.raises(ConfigurationError, match="valid UTF-8 JSON") as caught:
        _read(selected.name, templates, tmp_path)
    assert "private parser detail" not in str(caught.value)


@pytest.mark.parametrize("payload", [[], 1, "text", None])
def test_profile_source_refuses_non_object_json(tmp_path, templates, payload):
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ConfigurationError, match="JSON object"):
        _read(selected.name, templates, tmp_path)


def test_profile_source_reports_every_phase_problem_before_composition(tmp_path, templates):
    selected = tmp_path / "selected.json"
    selected.write_text(json.dumps({"format": 1, "phases": {
        "initial": {"tool_messages": {"disabled_message": {"replace": "Private text"}}},
        "coverage": {"user_prompt_template": {"replace": "$unknown_slot"}},
        "mutation": {"per_language": {"pyhton": {"system_prompt": {"append": "Private text"}}}},
    }}), encoding="utf-8")
    with pytest.raises(ConfigurationError) as caught:
        _read(selected.name, templates, tmp_path)
    message = str(caught.value)
    assert "disabled_message" in message
    assert "$unknown_slot" in message
    assert "pyhton" in message
    assert "Private text" not in message


@pytest.mark.parametrize("phase", ["initial", "coverage", "mutation"])
@pytest.mark.parametrize("language", ["python", "java"])
def test_empty_profile_source_preserves_supplied_templates_and_creates_no_outputs(
        tmp_path, templates, phase, language):
    selected = tmp_path / "selected.json"
    selected.write_bytes(b'{"format":1,"phases":{}}')
    captured = _read(selected.name, templates, tmp_path)
    assert captured.profile.compose(phase, language, templates[phase]) == templates[phase]
    assert list(tmp_path.iterdir()) == [selected]
