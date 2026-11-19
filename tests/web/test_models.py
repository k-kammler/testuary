"""The models page changes the model catalogue only through Testuary's own
setup code: every refusal is Testuary's, named for its field, and no
answer of the server holds a stored key. The catalogue is the one in
Testuary's user directory."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from urllib.parse import quote

import pytest

from testuary.web import _server as web_server


KEY = "canary-stored-key-0123"
MODELS = (
    "# My models.\n"
    "[ollama]\n"
    f"api_key = {KEY}\n"
    "\n"
    "# The cloud one.\n"
    "[cloud]\n"
    "provider = ollama\n"
    "model = m1\n"
)


@pytest.fixture(autouse=True)
def no_machine_settings(monkeypatch):
    for name in list(os.environ):
        if name.startswith("TESTUARY_") and name != "TESTUARY_HOME":
            monkeypatch.delenv(name)


@pytest.fixture
def models(user_dir):
    path = user_dir / "models.ini"
    path.write_text(MODELS, encoding="utf-8")
    return path


def test_the_model_list_says_a_key_is_stored_and_no_answer_holds_it(serve, models):
    app, server, client = serve()
    status, _, body = client.request("GET", "/api/models")
    assert status == 200 and KEY.encode() not in body
    view = json.loads(body)
    (entry,) = view["models"]
    assert (entry["alias"], entry["key_source"], entry["usable"]) == ("cloud", "models.ini", True)
    assert {item["kind"] for item in view["providers"]} >= {"ollama-local", "openai"}
    # No address returns the stored key: no read, and no action of a model.
    for path in web_server._GET_ROUTES:
        assert KEY.encode() not in client.request("GET", path)[2], path
    for method, sent in (("GET", None), ("POST", {})):
        status, _, body = client.request(method, "/api/models/cloud/key", sent)
        assert status == 404 and KEY.encode() not in body, method


def test_a_stored_key_is_replaced_and_an_empty_key_keeps_it(serve, models):
    app, server, client = serve()
    fields = {"alias": "cloud", "provider": "ollama", "model": "m1"}
    status, answer = client.json("POST", "/api/models", {**fields, "key": ""})
    assert status == 200 and KEY not in json.dumps(answer)
    assert f"api_key = {KEY}\n" in models.read_text(encoding="utf-8")
    status, answer = client.json("POST", "/api/models", {**fields, "key": "canary-new-key-0456"})
    assert status == 200 and "canary-new-key-0456" not in json.dumps(answer)
    text = models.read_text(encoding="utf-8")
    assert "api_key = canary-new-key-0456\n" in text and KEY not in text
    status, _, body = client.request("GET", "/api/models")
    assert json.loads(body)["models"][0]["key_source"] == "models.ini"
    assert b"canary-new-key-0456" not in body


def test_a_model_is_added_edited_and_removed_through_the_setup(serve, models):
    app, server, client = serve()
    status, _ = client.json("POST", "/api/models", {
        "alias": "local_qwen", "provider": "ollama", "model": "qwen3:8b",
        "address": "http://localhost:11434/v1"})
    assert status == 200
    text = models.read_text(encoding="utf-8")
    assert text.startswith(MODELS) and "[local_qwen]\nprovider = ollama\nmodel = qwen3:8b\n" in text
    status, _ = client.json("POST", "/api/models", {"alias": "cloud", "provider": "ollama",
                                                    "model": "m2"})
    assert status == 200
    assert "[cloud]\nprovider = ollama\nmodel = m2\n" in models.read_text(encoding="utf-8")
    status, _ = client.json("POST", "/api/models/local_qwen/remove", {})
    assert status == 200 and "[local_qwen]" not in models.read_text(encoding="utf-8")
    assert models.read_text(encoding="utf-8").startswith("# My models.\n[ollama]\n")


def test_a_hand_written_alias_the_setup_would_not_write_is_removed(serve, models):
    """An alias a hand-written catalogue holds, such as Qwen3-8B, is removed
    through the page's route, its name percent-encoded in the address."""
    models.write_text(models.read_text(encoding="utf-8")
                      + "\n[Qwen3-8B]\nprovider = ollama\nmodel = qwen3:8b\n", encoding="utf-8")
    app, server, client = serve()
    status, answer = client.json("POST", "/api/models/Qwen3%2D8B/remove", {})
    assert status == 200, answer
    assert "[Qwen3-8B]" not in models.read_text(encoding="utf-8")
    # An alias within its 128 characters whose encoded address is longer.
    alias = "Modell-" + "ä" * 30
    models.write_text(models.read_text(encoding="utf-8") + f"\n[{alias}]\nprovider = ollama\nmodel = qwen3:8b\n",
                      encoding="utf-8")
    status, answer = client.json("POST", f"/api/models/{quote(alias, safe='')}/remove", {})
    assert status == 200, answer
    assert alias not in models.read_text(encoding="utf-8")


def test_adding_a_model_never_replaces_one_that_exists(serve, models):
    """Add with an alias the catalogue holds is refused at the alias, and the
    model stays as it was. Edit (not new) still changes it."""
    app, server, client = serve()
    before = models.read_bytes()
    status, refused = client.json("POST", "/api/models", {
        "alias": "cloud", "provider": "ollama", "model": "other",
        "address": "http://localhost:8000/v1", "new": True})
    assert status == 400 and refused["field"] == "alias", refused
    assert "exists already" in refused["error"]
    assert models.read_bytes() == before
    status, _ = client.json("POST", "/api/models", {"alias": "cloud", "provider": "ollama",
                                                    "model": "m2", "new": False})
    assert status == 200


def test_a_provider_section_no_model_uses_is_listed(serve, user_dir):
    """A provider's section stays in models.ini when its last model goes:
    the setup keeps it for a later model or a run with [api] provider. The
    Models page lists such sections, so none stays out of sight, and holds
    none of their keys."""
    (user_dir / "models.ini").write_text(
        "[localfake]\napi_url = http://127.0.0.1:9/v1\n\n"
        "[tempprov]\napi_url = http://127.0.0.1:8/v1\napi_key = canary-unused-key\n\n"
        "[fake]\nprovider = localfake\nmodel = fake-model\n", encoding="utf-8")
    app, server, client = serve()
    status, _, body = client.request("GET", "/api/models")
    assert status == 200 and b"canary-unused-key" not in body
    view = json.loads(body)
    assert [entry["alias"] for entry in view["models"]] == ["fake"] and view["unused"] == ["tempprov"]
    status, _ = client.json("POST", "/api/models/fake/remove", {})
    assert status == 200
    assert client.json("GET", "/api/models")[1]["unused"] == ["localfake", "tempprov"]


def test_the_remove_question_says_what_stays_for_each_model(serve, user_dir):
    """A model of the user's own service has a provider section, which stays.
    A vLLM, OpenAI or Ollama model usually has none, so its question names
    none, and its own address goes with its section."""
    from conftest import page_value

    (user_dir / "models.ini").write_text(
        "[localfake]\napi_url = http://127.0.0.1:9/v1\n\n"
        "[fake]\nprovider = localfake\nmodel = fake-model\n\n"
        "[ollamax]\nprovider = ollama\nmodel = q\napi_url = http://localhost:11434/v1\n",
        encoding="utf-8")
    app, server, client = serve()
    entries = {entry["alias"]: entry for entry in client.json("GET", "/api/models")[1]["models"]}
    assert (entries["fake"]["provider_section"], entries["ollamax"]["provider_section"]) == (True, False)
    questions = page_value(["removeQuestion"], "%s.map(removeQuestion)" % json.dumps(
        [entries["fake"], entries["ollamax"]]))
    assert "section, localfake, stays" in questions[0]
    assert "stays" not in questions[1] and "The ollama provider has no section" in questions[1]


def test_the_models_page_names_where_models_ini_is(serve, user_dir):
    """The Models page gives models.ini's path, as Settings and Projects
    give theirs."""
    app, server, client = serve()
    view = client.json("GET", "/api/models")[1]
    assert view["file"] == str(user_dir.resolve() / "models.ini")
    # Shown short and visible, as the Overview shows the working directory.
    from conftest import SOURCE

    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "pathText(view.file, shortPath(view.file))" in source


def test_each_model_names_the_kind_of_service_it_is(serve, models, user_dir):
    """A model of the user's own OpenAI-compatible service opens as that
    kind, not as the first kind the setup offers, so an unchanged Save keeps
    its provider."""
    (user_dir / "models.ini").write_text(
        "[localfake]\napi_url = http://127.0.0.1:8123/v1\n\n"
        "[fake]\nprovider = localfake\nmodel = fake-model\n\n"
        "[local]\nprovider = ollama\nmodel = qwen3:8b\napi_url = http://localhost:11434/v1\n\n"
        "[cloud]\nprovider = ollama\nmodel = gpt-oss:120b\napi_url = https://ollama.com/v1\n",
        encoding="utf-8")
    app, server, client = serve()
    status, view = client.json("GET", "/api/models")
    kinds = {entry["alias"]: entry["kind"] for entry in view["models"]}
    assert kinds == {"fake": "openai-compatible", "local": "ollama-local", "cloud": "ollama-cloud"}
    fake = next(entry for entry in view["models"] if entry["alias"] == "fake")
    before = (user_dir / "models.ini").read_text(encoding="utf-8")
    # The form's unchanged Save: the same provider, model and address.
    status, _ = client.json("POST", "/api/models", {
        "alias": "fake", "provider": fake["provider"], "model": fake["model"],
        "address": fake["address"], "key_variable": "", "new": False})
    assert status == 200
    status, view = client.json("GET", "/api/models")
    after = next(entry for entry in view["models"] if entry["alias"] == "fake")
    assert (after["provider"], after["model"], after["address"]) == (
        "localfake", "fake-model", "http://127.0.0.1:8123/v1")
    assert "provider = localfake" in (user_dir / "models.ini").read_text(encoding="utf-8")
    assert before.count("[fake]") == 1


def test_a_refusal_is_testuarys_and_names_each_field(serve, models):
    app, server, client = serve()
    before = models.read_bytes()
    status, refused = client.json("POST", "/api/models", {"alias": "Bad-Name", "provider": "ollama",
                                                          "model": ""})
    assert status == 400
    fields = [field for field, _ in refused["problems"]]
    assert fields == ["alias", "model"] and refused["field"] == "alias"
    assert models.read_bytes() == before
    # A name the catalogue does not hold, such as .., is refused with the
    # setup's reason, and the catalogue stays as it is.
    status, refused = client.json("POST", "/api/models/../remove", {})
    assert status == 400 and "holds no model alias .." in refused["error"]
    assert models.read_bytes() == before


def test_the_global_settings_show_their_sources_and_change_through_the_validated_update(
        serve, user_dir):
    config = user_dir / "config.ini"
    config.write_text("# Mine.\n[main]\nseed = 3\n", encoding="utf-8")
    app, server, client = serve()
    status, view = client.json("GET", "/api/settings")
    assert status == 200 and view["error"] is None
    assert view["files"] == {"global": str(config.resolve()), "project": None}
    main = next(section for section in view["sections"] if section["name"] == "main")
    seed = next(setting for setting in main["settings"] if setting["name"] == "seed")
    assert (seed["value"], seed["source"]) == ("3", str(config.resolve()))
    status, _ = client.json("POST", "/api/settings", {"section": "main", "values": {"seed": "5"}})
    assert status == 200 and config.read_text(encoding="utf-8") == "# Mine.\n[main]\nseed = 5\n"
    status, refused = client.json("POST", "/api/settings", {"section": "main",
                                                            "values": {"seed": "many"}})
    assert status == 400 and "seed" in refused["error"]
    assert config.read_text(encoding="utf-8") == "# Mine.\n[main]\nseed = 5\n"
    # A goal in percent above 100 is refused, as the console refuses it.
    status, refused = client.json("POST", "/api/settings", {"section": "coverage",
                                                            "values": {"goal": "150"}})
    assert status == 400 and "expected a percentage from 0 to 100" in refused["error"]
    assert config.read_text(encoding="utf-8") == "# Mine.\n[main]\nseed = 5\n"


def test_the_catalogue_is_testuarys_export_without_keys(serve, user_dir):
    (user_dir / "models.ini").write_bytes(
        b"[openai_gpt]\nprovider = openai\nmodel = gpt-x\n"
        b"api_url = https://user:pw@api.example.test/v1?token=abc\n"
        b"api_key = canary-catalogue-key\n")
    (user_dir / "config.ini").write_bytes(b"[api]\nmodel = openai_gpt\n")
    app, server, client = serve()
    status, _, body = client.request("GET", "/api/catalogue")
    assert status == 200
    for secret in (b"canary-catalogue-key", b"user:pw", b"token=abc"):
        assert secret not in body
    data = json.loads(body)
    assert data["error"] is None
    assert data["models"]["openai_gpt"]["model"] == "gpt-x"
    assert data["default_model"] == "openai_gpt"


def test_the_catalogue_lists_only_model_aliases(serve, user_dir):
    """A provider's own section of models.ini, which an OpenAI-compatible
    service gets and which stays when its last model is removed, is no
    model: the overview and the run form list only the aliases."""
    (user_dir / "models.ini").write_bytes(
        b"[localfake]\napi_url = http://127.0.0.1:9/v1\n\n"
        b"[deadprov]\napi_url = http://127.0.0.1:8/v1\n\n"
        b"[fake]\nprovider = localfake\nmodel = fake-model\n")
    app, server, client = serve()
    status, data = client.json("GET", "/api/catalogue")
    assert status == 200 and data["error"] is None
    assert list(data["models"]) == ["fake"]


def test_unreadable_settings_give_a_plain_catalogue_error(serve, user_dir):
    (user_dir / "config.ini").write_bytes(b"[no_such_section]\nkey = 1\n")
    app, server, client = serve()
    status, data = client.json("GET", "/api/catalogue")
    assert status == 200 and "cannot read" in data["error"]
    assert data["models"] == {}


def test_a_models_connection_is_tested_as_a_run_would_reach_it(serve, user_dir, monkeypatch):
    """The test asks the service the run would ask, at the address and with
    the key the run would use, for the models it serves, which generates
    nothing. It says whether the service answered and lists the model,
    names the address, and never returns the key."""
    import fake_model

    service = fake_model.start()
    port = service.server_address[1]
    try:
        (user_dir / "models.ini").write_text(
            f"[fake]\nprovider = local\nmodel = fake-model\n\n"
            f"[other]\nprovider = local\nmodel = missing-model\n\n"
            f"[local]\napi_url = http://127.0.0.1:{port}/v1\napi_key = {fake_model.KEY}\n\n"
            "[dead]\nprovider = deadprov\nmodel = fake-model\n\n"
            "[deadprov]\napi_url = http://127.0.0.1:9/v1\n", encoding="utf-8")
        app, server, client = serve()
        status, found = client.json("POST", "/api/models/fake/test", {})
        assert status == 200, found
        assert (found["reached"], found["listed"]) == (True, True)
        assert found["text"] == f"The service at http://127.0.0.1:{port}/v1 answered and lists fake-model."
        status, other = client.json("POST", "/api/models/other/test", {})
        assert (other["reached"], other["listed"]) == (True, False)
        assert "does not list missing-model" in other["text"]
        status, dead = client.json("POST", "/api/models/dead/test", {})
        assert (dead["reached"], dead["listed"]) == (False, None)
        assert dead["text"].startswith("No answer from http://127.0.0.1:9/v1")
        assert all(fake_model.KEY not in json.dumps(answer) for answer in (found, other, dead))
    finally:
        service.shutdown()


def _flash_project(tmp_path, user_dir):
    (user_dir / "models.ini").write_bytes(
        b"[flash]\nprovider = ollama\nmodel = f\n\n[other]\nprovider = ollama\nmodel = o\n")
    project = tmp_path / "tenacity"
    (project / ".testuary").mkdir(parents=True)
    (project / ".testuary" / "project.ini").write_text(
        "[project]\nlanguage = python\n\n[api]\nmodel = flash\n", encoding="utf-8")
    return project


def test_the_default_model_is_the_global_one_and_the_projects_is_its_own(serve, user_dir, tmp_path):
    """The default model is the one the global settings choose (none here).
    The working directory's project's model is named apart."""
    app, server, client = serve(_flash_project(tmp_path, user_dir))
    for path in ("/api/catalogue", "/api/models"):
        data = client.json("GET", path)[1]
        assert data["default_model"] is None, path
        assert data["project_model"] == {"name": "tenacity", "model": "flash"}, path
    (user_dir / "config.ini").write_bytes(b"[api]\nmodel = other\n")
    data = client.json("GET", "/api/catalogue")[1]
    assert (data["default_model"], data["project_model"]["model"]) == ("other", "flash")


def test_a_dir_without_a_project_has_no_project_model(serve, user_dir, tmp_path):
    _flash_project(tmp_path, user_dir)
    app, server, client = serve(tmp_path)
    assert client.json("GET", "/api/catalogue")[1]["project_model"] is None


def _page(name, signature, calls):
    """A function of the page, evaluated by Node for each argument list."""
    import re
    import shutil
    import subprocess

    from conftest import SOURCE

    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    function = re.search(rf"^function {name}\({signature}\) {{\n.*?^}}\n", source, re.S | re.M)
    assert function, f"app.js defines {name}"
    script = (function.group(0) + "process.stdout.write(JSON.stringify(" + json.dumps(calls)
              + f".map((args) => {name}(...args))));")
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_the_run_page_names_the_model_an_empty_field_asks():
    project = {"name": "tenacity", "fields": {"model": "deepseek_v4_1_flash"}}
    plain = {"name": "calc", "fields": {}}
    assert _page("emptyModelText", "project, defaultModel", [
        [project, None], [project, "other"], [plain, "other"], [plain, None], [None, None],
    ]) == ["The project's model (deepseek_v4_1_flash)", "The project's model (deepseek_v4_1_flash)",
           "The default model (other)", "The default model", "The default model"]


def test_a_model_is_marked_default_only_when_the_global_settings_choose_it():
    working = {"name": "tenacity", "model": "flash"}
    assert _page("modelMarks", "alias, defaultModel, projectModel", [
        ["flash", None, working], ["flash", "flash", working], ["other", "flash", working],
        ["other", None, None],
    ]) == [["tenacity's model"], ["default model", "tenacity's model"], [], []]


def test_a_read_only_field_looks_unchangeable():
    """The alias of a model being edited is a read-only field: it has the
    muted background of a field that cannot change and no text cursor."""
    style = (Path(web_server.__file__).parent / "static" / "style.css").read_text(encoding="utf-8")
    rule = re.search(r"^input\[readonly\] \{([^}]*)\}", style, re.M)
    assert rule and "background: var(--disabled-bg)" in rule.group(1) and "cursor: default" in rule.group(1)
