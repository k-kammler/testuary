"""Each atomic write goes through a temporary file of its own beside the
target, so two writers never share one, and a failed write leaves none
and reports its own error."""
import contextlib
import os
import secrets
from pathlib import Path
from types import SimpleNamespace

import pytest

from testuary import _chat_log, _progress, results, settings, web
from testuary.agent._helpers import AgentLogger
from testuary.web import _data as web_data


def _csv(tmp_path, monkeypatch):
    target = tmp_path / "tests.csv"
    return target, lambda: results._rewrite_csv(target, ["a"], [{"a": "1"}])


def _projects(tmp_path, monkeypatch):
    target = tmp_path / "web" / "projects.json"
    monkeypatch.setattr(web_data, "_store", lambda: target)
    return target, lambda: web_data._save([tmp_path])


def _access_key(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "file_locations", lambda: SimpleNamespace(user=tmp_path))
    return tmp_path / "web" / "access-key", lambda: web._access_key("stable_key")


def _chat(tmp_path, monkeypatch):
    return tmp_path / "chat.json", lambda: _chat_log.export_chat_log([], str(tmp_path), "chat.json")


def _agent_log(tmp_path, monkeypatch):
    target = tmp_path / "agent.jsonl"
    logger = AgentLogger(project_name="fixture", agent_name="main", config={})
    return target, lambda: logger._write_whole(target, [{"message": "kept"}])


WRITERS = [_csv, _projects, _access_key, _chat, _agent_log]


@pytest.mark.parametrize("writer", WRITERS)
def test_a_write_leaves_another_writers_temporary_file_alone(tmp_path, monkeypatch, writer):
    target, write = writer(tmp_path, monkeypatch)
    others = target.with_name(target.name + ".tmp")
    others.parent.mkdir(parents=True, exist_ok=True)
    others.write_text("another writer's", encoding="utf-8")
    write()
    assert target.exists() and others.read_text(encoding="utf-8") == "another writer's"
    assert sorted(path.name for path in target.parent.iterdir()) == sorted([target.name, others.name])


@pytest.mark.parametrize("writer", WRITERS)
def test_a_failed_write_leaves_no_temporary_file(tmp_path, monkeypatch, writer):
    target, write = writer(tmp_path, monkeypatch)

    def refuse(*args, **kwargs):
        raise OSError("refused")

    monkeypatch.setattr(os, "replace", refuse)
    monkeypatch.setattr(Path, "replace", refuse)
    with contextlib.suppress(OSError):
        write()
    assert not target.exists()
    assert [path.name for path in target.parent.iterdir()] == []


@pytest.mark.parametrize("writer", WRITERS)
def test_a_failed_clean_up_keeps_the_writes_own_error(tmp_path, monkeypatch, writer):
    target, write = writer(tmp_path, monkeypatch)
    unlink = Path.unlink

    def refuse(*args, **kwargs):
        raise OSError("replace refused")

    def refuse_temporary(path, missing_ok=False):
        if path.name.endswith(".tmp"):
            raise PermissionError("unlink refused")
        unlink(path, missing_ok=missing_ok)

    reports = []
    monkeypatch.setattr(os, "replace", refuse)
    monkeypatch.setattr(Path, "replace", refuse)
    monkeypatch.setattr(Path, "unlink", refuse_temporary)
    monkeypatch.setattr(_progress, "diagnostic_print", reports.append)
    try:
        write()
    except OSError as error:
        reports.append(str(error))
    assert reports and all("replace refused" in report for report in reports), reports


@pytest.mark.parametrize("writer", WRITERS)
def test_a_temporary_name_adds_13_characters_to_the_target(tmp_path, monkeypatch, writer):
    target, write = writer(tmp_path, monkeypatch)
    renamed, replace, path_replace = [], os.replace, Path.replace
    monkeypatch.setattr(os, "replace", lambda source, *rest: (
        renamed.append(Path(source).name), replace(source, *rest))[1])
    monkeypatch.setattr(Path, "replace", lambda source, *rest: (
        renamed.append(source.name), path_replace(source, *rest))[1])
    write()
    assert {len(name) - len(target.name) for name in renamed} == {13}


@pytest.mark.parametrize("writer", WRITERS)
def test_a_temporary_name_another_writer_holds_fails_the_write(tmp_path, monkeypatch, writer):
    target, write = writer(tmp_path, monkeypatch)
    monkeypatch.setattr(secrets, "token_hex", lambda size: "0" * 2 * size)
    held = target.with_name(f"{target.name}.00000000.tmp")
    held.parent.mkdir(parents=True, exist_ok=True)
    held.write_text("another writer's", encoding="utf-8")
    with contextlib.suppress(OSError):
        write()
    assert not target.exists() and held.read_text(encoding="utf-8") == "another writer's"
