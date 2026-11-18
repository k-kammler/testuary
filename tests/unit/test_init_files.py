"""testuary init writes the user directory's model catalogue and the project's
starter hints file: into a user directory a fresh machine does not have yet,
never over an existing file, and the catalogue readable by its owner only
on POSIX. Provisioned environments live in the user's cache directory."""
from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from testuary import __main__ as cli
from testuary import project as project_module


@pytest.fixture
def started(tmp_path, monkeypatch):
    user = tmp_path / "fresh" / "user"
    project = tmp_path / "project"
    project.mkdir()
    monkeypatch.setenv("TESTUARY_HOME", str(user))
    monkeypatch.chdir(project)
    return user, project


def test_init_makes_a_missing_user_dir(started):
    user, project = started
    assert cli._cmd_init([]) == 0
    assert (user / "models.ini").is_file()
    assert (project / "hints.json").is_file()


def test_init_never_writes_over_an_existing_file(started, capsys):
    user, _ = started
    user.mkdir(parents=True)
    (user / "models.ini").write_text("[mine]\nprovider = ollama\nmodel = m\n", encoding="utf-8")
    assert cli._cmd_init([]) == 0
    assert (user / "models.ini").read_text(encoding="utf-8").startswith("[mine]")
    assert "skip: models.ini already exists" in capsys.readouterr().out
    # A file made after it was looked for is not replaced either.
    with pytest.raises(FileExistsError):
        cli._write_new(user / "models.ini", b"replaced", private=True)
    assert (user / "models.ini").read_text(encoding="utf-8").startswith("[mine]")


@pytest.mark.skipif(os.name != "posix", reason="POSIX file modes")
def test_the_model_catalogue_is_readable_by_its_owner_only(started):
    user, _ = started
    previous = os.umask(0o022)
    try:
        assert cli._cmd_init([]) == 0
    finally:
        os.umask(previous)
    assert stat.S_IMODE((user / "models.ini").stat().st_mode) == 0o600


def test_provisioned_environments_live_in_the_users_cache_dir(monkeypatch, tmp_path):
    import platformdirs

    asked = []

    def cache(*args, **kwargs):
        asked.append((args, kwargs))
        return str(tmp_path / "cache")

    monkeypatch.setattr(platformdirs, "user_cache_dir", cache)
    assert project_module._venv_home() == Path(tmp_path / "cache" / "venvs")
    assert asked == [(("testuary",), {"appauthor": False})]
