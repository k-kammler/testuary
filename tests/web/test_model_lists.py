"""The Models page's lists of model catalogue files a settings group names
for a project: joined as the extension's runs read them, changed only in
the last file, a model only an earlier file defines removed only by ending
the import, no key written, and every save the group refuses put back."""
from __future__ import annotations

import os
from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import page_value
from testuary import settings
from testuary.web import _registry
from testuary.web import _data as projects
from testuary.web import _server as web_server
from testuary.web.extensions import SettingsGroup, WebExtension

SHARED = ("[local_qwen]\nprovider = vllm\nmodel = qwen\n\n"
          "[cloud_coder]\nprovider = ollama\nmodel = coder\nrequest_timeout_seconds = 60\n\n"
          "[tiny]\nprovider = ollama\nmodel = tiny\n")
NOTES = "# Notes' models.\n[local_qwen]\nprovider = vllm\nmodel = qwen-2\n"
LIST = [Path("shared", "models.ini"), Path(".notes", "models.ini")]


@pytest.fixture(autouse=True)
def no_machine_settings(monkeypatch):
    for name in list(os.environ):
        if name.startswith("TESTUARY_") and name != "TESTUARY_HOME":
            monkeypatch.delenv(name)


def _project(tmp_path, notes=NOTES):
    """A listed project with shared/models.ini and .notes/models.ini."""
    project_dir = tmp_path / "calc"
    (project_dir / ".testuary").mkdir(parents=True)
    (project_dir / "shared").mkdir()
    (project_dir / "shared" / "models.ini").write_text(SHARED, encoding="utf-8")
    (project_dir / ".notes").mkdir()
    if notes is not None:
        (project_dir / ".notes" / "models.ini").write_text(notes, encoding="utf-8")
    projects.add(str(project_dir))
    return project_dir.resolve()


def _client(serve, models_files, **fields):
    """A client of a server whose extension's settings group names the
    model catalogue files ``models_files`` returns for a project."""
    import web_extension_fixture as notes

    group = SettingsGroup("notes", "Notes", notes.reference, notes.values, notes.update,
                          testuary_models_files=models_files, **fields)
    extension = WebExtension(api=1, id="notes", settings_groups=(group,))
    found = ("notes", extension, Path(notes.__file__).parent, None)
    return serve(extensions=_registry.checked([found]))[2]


def _view(client, project_dir):
    status, view = client.json("GET", f"/api/models?project_dir={quote(str(project_dir))}&scope=notes/notes")
    assert status == 200
    return view


def _models(view):
    return {entry["alias"]: (entry["model"], Path(entry["file"]).parent.name) for entry in view["models"]}


def test_the_scoped_models_query_holds_the_fields_the_extension_guide_names(serve, tmp_path, user_dir):
    """/api/models with project_dir and scope holds models, each with alias,
    provider, model and usable, and error, as docs/extending/web.md promises
    an extension's view, and a project that is not listed gives error with
    no models."""
    project_dir = _project(tmp_path)
    client = _client(serve, lambda project: LIST)
    view = _view(client, project_dir)
    assert view["error"] is None and view["models"]
    for entry in view["models"]:
        assert {"alias", "provider", "model", "usable"} <= set(entry)
        assert isinstance(entry["usable"], bool)
    status, other = client.json(
        "GET", f"/api/models?project_dir={quote(str(tmp_path / 'elsewhere'))}&scope=notes/notes")
    assert status == 200 and other["error"] and other["models"] == []

def test_the_scoped_models_read_the_groups_settings_files_not_the_projects(serve, tmp_path, user_dir):
    """A group that names its own settings files shows its list as its runs
    read it, with those files: the project's project.ini, which its runs do
    not read, does not refuse the view."""
    project_dir = _project(tmp_path)
    (project_dir / ".testuary" / "project.ini").write_text(
        "[api]\napi_url = https://project.example/v1\n", encoding="utf-8")
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "testuary.ini").write_text("[coverage]\ngoal = 80\n", encoding="utf-8")
    client = _client(serve, lambda project: LIST, testuary_settings_files=lambda project: [ext / "testuary.ini"])
    view = _view(client, project_dir)
    assert view["error"] is None
    assert _models(view)["local_qwen"] == ("qwen-2", ".notes")


def test_a_groups_list_is_shown_joined_with_each_models_file_and_its_override(serve, tmp_path, user_dir):
    project_dir = _project(tmp_path)
    client = _client(serve, lambda project: LIST)
    status, own = client.json("GET", "/api/models")
    assert (status, own["list"]) == (200, None)
    assert [(entry["scope"], entry["name"], entry["read_only"]) for entry in own["scopes"]] == [
        ("notes/notes", "calc", None)]
    view = _view(client, project_dir)
    assert (view["error"], view["file"]) == (None, str(project_dir / ".notes" / "models.ini"))
    assert _models(view) == {"local_qwen": ("qwen-2", ".notes"), "cloud_coder": ("coder", "shared"),
                             "tiny": ("tiny", "shared")}
    assert {entry["alias"]: entry["overrides"] for entry in view["models"]}["local_qwen"] == str(
        project_dir / "shared" / "models.ini")


def test_a_failing_or_missing_list_answers_in_its_choice_and_the_rest_works(serve, tmp_path, user_dir):
    project_dir = _project(tmp_path)

    def failing(project):
        raise RuntimeError("no notes models")

    client = _client(serve, failing)
    (entry,) = client.json("GET", "/api/models")[1]["scopes"]
    assert entry["error"] == "The extension's model catalogue files failed: RuntimeError: no notes models"
    view = _view(client, project_dir)
    assert (view["error"], view["models"]) == (entry["error"], [])
    client = _client(serve, lambda project: None)
    assert client.json("GET", "/api/models")[1]["scopes"] == []
    assert _view(client, project_dir)["error"] == "This project has no such list of model catalogue files."
    status, _ = client.json("POST", "/api/models", {"project_dir": str(project_dir), "scope": "notes/notes",
                                                    "alias": "tiny", "provider": "ollama", "model": "t"})
    assert status == 404


def test_a_change_overrides_an_earlier_section_and_its_removal_shows_the_earlier_one(serve, tmp_path, user_dir):
    """Without a function that ends the import, a model only an earlier file
    defines is overridden, not removed."""
    project_dir = _project(tmp_path)
    notes, shared = project_dir / ".notes" / "models.ini", project_dir / "shared" / "models.ini"
    client = _client(serve, lambda project: LIST)
    body = {"project_dir": str(project_dir), "scope": "notes/notes"}
    status, refused = client.json("POST", "/api/models/cloud_coder/remove", body)
    assert status == 400 and "cannot end its import" in refused["error"]
    assert notes.read_text(encoding="utf-8") == NOTES
    assert client.json("POST", "/api/models", {**body, "alias": "cloud_coder", "provider": "ollama",
                                               "model": "coder-2", "address": ""})[0] == 200
    assert settings._read_ini_file(notes)["cloud_coder"] == {
        "provider": "ollama", "model": "coder-2", "request_timeout_seconds": "60"}
    assert shared.read_text(encoding="utf-8") == SHARED
    assert client.json("POST", "/api/models/local_qwen/remove", body)[0] == 200
    assert _models(_view(client, project_dir))["local_qwen"] == ("qwen", "shared")


def test_removing_an_inherited_model_ends_the_import(serve, tmp_path, user_dir):
    project_dir = _project(tmp_path)
    notes = project_dir / ".notes" / "models.ini"
    read, calls = {"files": LIST}, []

    def end(project):
        calls.append(project)
        read["files"] = LIST[1:]

    client = _client(serve, lambda project: read["files"], end_models_import=end)
    assert _view(client, project_dir)["list"]["end_import"] is True
    status, answer = client.json("POST", "/api/models/cloud_coder/remove",
                                 {"project_dir": str(project_dir), "scope": "notes/notes"})
    assert (status, answer) == (200, {"changed": True, "warning": None})
    assert calls == [project_dir]
    assert settings._read_ini_file(notes) == {"local_qwen": {"provider": "vllm", "model": "qwen-2"},
                                              "tiny": {"provider": "ollama", "model": "tiny"}}
    assert _models(_view(client, project_dir)) == {"local_qwen": ("qwen-2", ".notes"), "tiny": ("tiny", ".notes")}


@pytest.mark.parametrize("case", ["raises", "raises after ending", "returns", "group refuses", "no restore"])
def test_a_failing_end_of_an_import_puts_the_last_file_back(serve, tmp_path, user_dir, monkeypatch, case):
    created = case == "raises"
    project_dir = _project(tmp_path, notes=None if created else NOTES)
    notes = project_dir / ".notes" / "models.ini"
    read, calls = {"files": LIST}, []

    def listed(project):
        if case == "group refuses" and notes.is_file() and "tiny" in notes.read_text(encoding="utf-8"):
            raise ValueError("a file in the project directory may not copy tiny")
        return read["files"]

    def end(project):
        calls.append(project)
        if case == "raises after ending":
            read["files"] = LIST[1:]
        if case != "returns":
            raise RuntimeError("settings locked")

    if case == "no restore":
        monkeypatch.setattr(web_server, "_put_back", lambda path, previous: OSError("disk full"))
    client = _client(serve, listed, end_models_import=end)
    status, answer = client.json("POST", "/api/models/cloud_coder/remove",
                                 {"project_dir": str(project_dir), "scope": "notes/notes"})
    earlier, last = project_dir / "shared" / "models.ini", notes
    expected = {
        "raises": f"The import of {earlier} did not end: settings locked. {last} is as it was, and the "
                  f"models of {earlier} still apply.",
        "returns": f"The import of {earlier} did not end: the runs still read {earlier}, {last}. {last} is "
                   "as it was.",
        "group refuses": "a file in the project directory may not copy tiny",
        "no restore": f"The import of {earlier} did not end: settings locked, and {last} could not be put "
                      f"back: disk full. {last} holds a copy of every model of {earlier}, which the runs "
                      "read as overrides."}
    if case == "raises after ending":
        assert (status, answer["warning"]) == (200, f"The import of {earlier} ended, and its function "
                                                    f"reported: settings locked. {last} holds every other model.")
        assert "tiny" in notes.read_text(encoding="utf-8")
        return
    assert (status, answer["error"]) == (400, expected[case])
    assert calls == ([] if case == "group refuses" else [project_dir])
    if created:
        assert not notes.exists()
    elif case != "no restore":
        assert notes.read_text(encoding="utf-8") == NOTES


def test_a_save_the_group_refuses_is_put_back_on_the_models_and_the_settings_page(serve, tmp_path, user_dir):
    project_dir = _project(tmp_path, notes=None)
    notes = project_dir / ".notes" / "models.ini"
    named = project_dir / ".notes" / "testuary.ini"
    named.write_text("[coverage]\ngoal = 70\n", encoding="utf-8")

    def refusing(path, word, files):
        def listed(project):
            if path.is_file() and word in path.read_text(encoding="utf-8"):
                raise ValueError(f"{path.name} may not hold {word}")
            return files
        return listed

    client = _client(serve, refusing(notes, "coder-2", LIST),
                     testuary_settings_files=refusing(named, "75", [Path(".notes", "testuary.ini")]))
    body = {"project_dir": str(project_dir), "scope": "notes/notes"}
    status, refused = client.json("POST", "/api/models", {**body, "alias": "cloud_coder", "provider": "ollama",
                                                          "model": "coder-2"})
    assert (status, refused["error"]) == (400, "models.ini may not hold coder-2")
    assert not notes.exists()
    status, refused = client.json("POST", "/api/settings", {**body, "section": "coverage", "values": {"goal": "75"}})
    assert (status, refused["error"]) == (400, "testuary.ini may not hold 75")
    assert named.read_text(encoding="utf-8") == "[coverage]\ngoal = 70\n"
    assert client.json("POST", "/api/settings", {**body, "section": "coverage", "values": {"goal": "80"}})[0] == 200
    assert "goal = 80" in named.read_text(encoding="utf-8")


def test_keys_stay_out_of_the_last_file_and_an_address_credential_is_not_copied(serve, tmp_path, user_dir):
    """The user directory's catalogue holds gateway's key, read by path. The
    extension's own file, in its user directory, takes key variables only."""
    project_dir = _project(tmp_path)
    own, home = tmp_path / "ext" / "models.ini", user_dir / "models.ini"
    catalogue = ("[gateway]\napi_url = https://gate.example/v1\napi_key = sk-home\n\n"
                 "[vault]\napi_url = https://user:secret@vault.example/v1\n\n"
                 "[cloud_coder]\nprovider = gateway\nmodel = coder\n\n"
                 "[other]\nprovider = gateway\nmodel = other\n\n[vaulted]\nprovider = vault\nmodel = v\n")
    home.write_text(catalogue, encoding="utf-8")
    client = _client(serve, lambda project: [home, own], extension_user_dir=tmp_path / "ext",
                     end_models_import=lambda project: None)
    view = _view(client, project_dir)
    assert view["error"] is None and "sk-home" not in str(view) and "secret" not in str(view)
    assert {entry["alias"]: entry["provider_key_file"] for entry in view["models"]}["cloud_coder"] == str(home)
    assert (view["import_keys"], view["import_blocked"]) == (
        [{"section": "gateway", "file": str(home)}], [{"section": "vault", "file": str(home)}])
    body = {"project_dir": str(project_dir), "scope": "notes/notes"}
    model = {**body, "alias": "cloud_coder", "provider": "gateway", "model": "coder",
             "address": "https://gate2.example/v1"}
    status, refused = client.json("POST", "/api/models", {**model, "key": "sk-new"})
    assert status == 400 and refused["problems"][0][0] == "key"
    status, refused = client.json("POST", "/api/models", model)
    assert status == 400 and refused["problems"] == [["key_variable", f"{home} holds the key of [gateway], which is "
                                                                      f"not copied into {own}: name the variable "
                                                                      "that holds it"]]
    assert client.json("POST", "/api/models", {**model, "key_variable": "GATE_KEY"})[0] == 200
    assert settings._read_ini_file(own)["gateway"] == {"api_url": "https://gate2.example/v1", "api_key_env": "GATE_KEY"}
    status, refused = client.json("POST", "/api/models/other/remove", {**body, "key_variables": {}})
    assert status == 400 and "[vault] in" in refused["error"] and "credential" in refused["error"]
    assert home.read_text(encoding="utf-8") == catalogue
    assert "sk-home" not in own.read_text(encoding="utf-8")


GATEWAY = ("[gateway]\napi_url = https://gate.example/v1\napi_key = sk-home\n\n"
           "[cloud_coder]\nprovider = gateway\nmodel = coder\n\n[other]\nprovider = gateway\nmodel = other\n\n"
           "[pinned]\nprovider = openai\nmodel = p\napi_url = https://pinned.example/v1\n")


def test_a_last_file_in_the_project_takes_no_address_key_or_key_variable(serve, tmp_path, user_dir):
    """The page learns that the last file lies in the project directory. A
    model whose earlier section sets an address or a key cannot be copied
    there, an end of the import is blocked by every such section, and a
    change that needs neither is written."""
    project_dir = _project(tmp_path, notes=None)
    notes = project_dir / ".notes" / "models.ini"
    (user_dir / "models.ini").write_text(GATEWAY, encoding="utf-8")
    client = _client(serve, lambda project: [user_dir / "models.ini", notes], end_models_import=lambda project: None)
    view = _view(client, project_dir)
    assert view["list"]["in_project"] is True
    assert {entry["alias"]: entry["key_places"] for entry in view["models"]} == {
        "cloud_coder": False, "other": False, "pinned": True}
    assert [item["section"] for item in view["import_places"]] == ["gateway", "pinned"]
    body = {"project_dir": str(project_dir), "scope": "notes/notes"}
    status, refused = client.json("POST", "/api/models", {**body, "alias": "pinned", "provider": "openai",
                                                          "model": "p2", "address": "https://pinned.example/v1"})
    assert status == 400 and refused["error"].startswith(f"{notes}: [pinned] api_url decides")
    assert sorted(path.name for path in notes.parent.iterdir()) == []
    assert client.json("POST", "/api/models", {**body, "alias": "cloud_coder", "provider": "gateway",
                                               "model": "coder-2", "address": "", "key_variable": ""})[0] == 200
    assert settings._read_ini_file(notes) == {"cloud_coder": {"provider": "gateway", "model": "coder-2"}}


def test_a_provider_section_the_last_file_overrides_is_shown_and_removed(serve, tmp_path, user_dir):
    project_dir = _project(tmp_path)
    own = tmp_path / "ext" / "models.ini"
    (user_dir / "models.ini").write_text(GATEWAY, encoding="utf-8")
    client = _client(serve, lambda project: [user_dir / "models.ini", own], extension_user_dir=tmp_path / "ext")
    body = {"project_dir": str(project_dir), "scope": "notes/notes"}
    assert client.json("POST", "/api/models", {**body, "alias": "cloud_coder", "provider": "gateway", "model": "coder",
                                               "address": "https://gate2.example/v1",
                                               "key_variable": "GATE_KEY"})[0] == 200
    view = _view(client, project_dir)
    assert view["provider_overrides"] == [{"section": "gateway", "file": str(user_dir / "models.ini")}]
    assert {entry["alias"]: entry["provider_from"] for entry in view["models"]}["other"] == str(own)
    assert client.json("POST", "/api/models/gateway/remove", body)[0] == 200
    assert "gateway" not in settings._read_ini_file(own)
    view = _view(client, project_dir)
    assert view["provider_overrides"] == []
    assert {entry["alias"]: entry["address"] for entry in view["models"]}["other"] == "https://gate.example/v1"


BASE = "[base]\nprovider = openai\nmodel = b\napi_url = https://base.example/v1\n"
CHILD = "[child]\nprovider = base\nmodel = c\n"


def test_a_change_to_your_catalogue_keeps_the_models_of_every_list_that_reads_it(serve, tmp_path, user_dir):
    """child, in the extension's own file, takes its address from base in
    the user directory's catalogue, which a group's list reads first."""
    _project(tmp_path)
    own = tmp_path / "ext" / "models.ini"
    own.parent.mkdir()
    own.write_text(CHILD, encoding="utf-8")
    (user_dir / "models.ini").write_text(BASE, encoding="utf-8")
    client = _client(serve, lambda project: [user_dir / "models.ini", own], extension_user_dir=tmp_path / "ext")
    status, refused = client.json("POST", "/api/models/base/remove", {})
    assert status == 400 and refused["error"].startswith("This change would leave child without a provider")
    assert str(own) in refused["error"]
    assert (user_dir / "models.ini").read_text(encoding="utf-8") == BASE


def test_a_lists_change_is_checked_with_the_settings_its_runs_read(serve, tmp_path, user_dir):
    """The global settings give every model an address, the group's named
    settings file does not, so without base child has none for its runs."""
    project_dir = _project(tmp_path)
    ext = tmp_path / "ext"
    ext.mkdir()
    (ext / "models.ini").write_text(BASE + "\n" + CHILD, encoding="utf-8")
    (ext / "testuary.ini").write_text("[coverage]\ngoal = 80\n", encoding="utf-8")
    (user_dir / "config.ini").write_text("[api]\napi_url = https://everywhere.example/v1\n", encoding="utf-8")
    client = _client(serve, lambda project: [ext / "models.ini"], extension_user_dir=ext,
                     testuary_settings_files=lambda project: [ext / "testuary.ini"])
    status, refused = client.json("POST", "/api/models/base/remove",
                                  {"project_dir": str(project_dir), "scope": "notes/notes"})
    assert status == 400 and "leave child without" in refused["error"]
    assert (ext / "models.ini").read_text(encoding="utf-8") == BASE + "\n" + CHILD


def test_a_last_file_outside_the_allowed_dirs_or_the_users_catalogue_is_read_only(serve, tmp_path, user_dir):
    project_dir = _project(tmp_path)
    outside = tmp_path / "elsewhere" / "models.ini"
    lists = {"outside": [Path("shared", "models.ini"), outside],
             "own": [Path("shared", "models.ini"), user_dir / "models.ini"],
             "extension": [Path("shared", "models.ini"), tmp_path / "ext" / "models.ini"]}
    for name, files in lists.items():
        client = _client(serve, lambda project, files=files: files, extension_user_dir=tmp_path / "ext")
        (entry,) = client.json("GET", "/api/models")[1]["scopes"]
        status, refused = client.json("POST", "/api/models", {
            "project_dir": str(project_dir), "scope": "notes/notes", "alias": "tiny", "provider": "ollama",
            "model": "t2"})
        if name == "extension":
            assert (entry["read_only"], status) == (None, 200)
            continue
        assert (status, refused["error"]) == (400, entry["read_only"])
        assert entry["global"] is (name == "own")
    assert not outside.exists() and not (user_dir / "models.ini").exists()
    assert entry["read_only"] is None


def test_the_models_page_links_its_choice_and_says_no_key_can_be_stored_in_a_list():
    assert page_value(["modelsLink"], "[modelsLink('C:/p q', 'notes/notes'), modelsLink()]") == [
        "#/models?project_dir=C%3A%2Fp%20q&scope=notes%2Fnotes", "#/models"]
    entry = '{"key_source": "missing", "key_variable": "K"}'
    assert page_value(["KEY_SOURCES", "keyText"], f"[keyText({entry}, null), keyText({entry}, {{}})]") == [
        "missing: set K or store a key", "missing: set K"]
