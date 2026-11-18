"""The directory browser lists directory names only, on local drives, and refuses a
network or device path before opening anything."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

from testuary.web import _data
from testuary.web import _data as projects


@pytest.fixture(autouse=True)
def no_machine_settings(monkeypatch):
    for name in list(os.environ):
        if name.startswith("TESTUARY_"):
            monkeypatch.delenv(name)


def test_only_dirs_are_listed_by_name_with_projects_marked(tmp_path):
    for name in ("beta", "Alpha", "gamma"):
        (tmp_path / name).mkdir()
    (tmp_path / "notes.txt").write_bytes(b"not listed")
    (tmp_path / "secret.ini").write_bytes(b"api_key = not listed")
    view = _data.dir_listing(str(tmp_path), {"calc": tmp_path / "beta"}, {"Home": tmp_path})
    assert [item["name"] for item in view["dirs"]] == ["Alpha", "beta", "gamma"]
    assert [item["project"] for item in view["dirs"]] == [None, "calc", None]
    assert view["path"] == str(tmp_path) and view["parent"] == str(tmp_path.parent)
    assert view["crumbs"][-1] == {"name": tmp_path.name, "path": str(tmp_path)}
    assert view["shortcuts"] == [{"name": "Home", "path": str(tmp_path)}]
    assert "notes.txt" not in str(view) and "api_key" not in str(view)


def test_a_long_listing_is_cut(tmp_path, monkeypatch):
    for name in ("a", "b", "c"):
        (tmp_path / name).mkdir()
    monkeypatch.setattr(_data, "MAX_DIRS", 2)
    view = _data.dir_listing(str(tmp_path))
    assert [item["name"] for item in view["dirs"]] == ["a", "b"] and view["truncated"]


def test_the_roots_are_the_local_drives_or_slash(tmp_path):
    view = _data.dir_roots({"Working directory": tmp_path})
    names = [item["name"] for item in view["dirs"]]
    if os.name == "nt":
        assert Path(os.environ.get("SystemDrive", "C:") + "\\").drive + "\\" in names
    else:
        assert names == ["/"]
    assert view["path"] is None and view["shortcuts"] == [{"name": "Working directory",
                                                           "path": str(tmp_path)}]


def test_network_device_and_relative_paths_are_refused_unopened(monkeypatch):
    def opened(*args, **kwargs):
        raise AssertionError("a refused path was opened")

    monkeypatch.setattr(_data.os, "scandir", opened)
    texts = ["", "relative", "./here", "//server/share/project"]
    if os.name == "nt":
        texts += ["\\\\server\\share\\project", "\\\\?\\C:\\Windows", "\\\\.\\pipe\\x",
                  "C:relative", "\\rooted", "/rooted"]
    for text in texts:
        with pytest.raises(_data.NotListed):
            _data.dir_listing(text)


def test_a_missing_dir_or_a_file_is_named_plainly(tmp_path):
    (tmp_path / "file.txt").write_bytes(b"x")
    with pytest.raises(_data.NotListed, match="No such directory"):
        _data.dir_listing(str(tmp_path / "missing"))
    with pytest.raises(_data.NotListed, match="not a directory"):
        _data.dir_listing(str(tmp_path / "file.txt"))


def test_the_interface_lists_dirs_for_the_page(serve, tmp_path):
    work = tmp_path / "work"
    (work / "projects" / "calc").mkdir(parents=True)
    app, server, client = serve(work)
    # A listed project's directory is marked.
    projects.add(str(work / "projects" / "calc"))
    status, view = client.json("GET", f"/api/dirs?path={work / 'projects'}")
    assert status == 200 and view["dirs"][0]["project"] == "calc"
    status, view = client.json("GET", "/api/dirs")
    assert status == 200 and {"name": "Working directory", "path": str(app.base_dir)} in view["shortcuts"]
    status, refused = client.json("GET", "/api/dirs?path=relative")
    assert status == 404 and refused["error"]
    assert client.request("GET", "/api/dirs", token=False)[0] == 403


@pytest.mark.skipif(os.name != "nt", reason="drive letters are Windows'")
def test_a_drive_letter_of_a_network_share_is_refused_unopened(monkeypatch):
    def opened(*args, **kwargs):
        raise AssertionError("a refused path was opened")

    types = {"Z:\\": 4, "Y:\\": 5, "X:\\": 1}
    monkeypatch.setattr(_data, "_drive_type", lambda root: types.get(root.upper(), 3))
    monkeypatch.setattr(_data.os, "scandir", opened)
    for text in ("Z:\\projects", "z:/projects", "Y:\\", "X:\\directory"):
        with pytest.raises(_data.NotListed, match="Network and device paths"):
            _data.dir_listing(text)
