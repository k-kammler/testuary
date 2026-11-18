"""Web interface extensions: an extension's tabs, directory types,
settings groups and job kinds, read once at the start, and what is refused."""
from __future__ import annotations

import json
import os
import re
import sys
import threading
import time
from pathlib import Path
from urllib.parse import quote

import pytest

from conftest import SOURCE, page_value, wait_for
from testuary.settings import ConfigurationError
from testuary.web import _data as projects
from testuary.web import _registry, extensions
from testuary.web._registry import Registry
from testuary.web.extensions import DirType, JobKind, SettingsGroup, WebExtension, WebRoute, WebTab

HERE = Path(__file__).resolve().parent
APP = SOURCE / "testuary" / "web" / "static" / "app.js"
#: The fixture extension as [main] extensions names it: the module
#: web_extension_fixture on the tests' import path, from a program's overrides.
NOTES = ("web_extension_fixture", "overrides", str(HERE))
FIXTURE = HERE / "web_extension_fixture"


def _found(name, made, folder=FIXTURE):
    """``made`` as a loaded module's testuary_web, its files served from
    ``folder``, for ``_registry.checked``."""
    return name, made, Path(folder), None


def test_an_installed_extension_is_found_and_its_scripts_served(serve):
    registry = _registry.load([NOTES])
    _, _, client = serve(extensions=registry)
    status, state = client.json("GET", "/api/state")
    found = state["extensions"]
    assert (status, found["refused"]) == (200, [])
    assert found["tabs"] == [{"id": "notes", "title": "Notes", "position": "before:run",
                              "replaces": "run", "script": "/ext/notes/notes.js"}]
    assert found["tab_positions"] == {"results": "after:notes"}
    assert [item["id"] for item in found["dir_types"]] == ["notes/notes"]
    assert found["job_kinds"] == [{"id": "notes/collect", "title": "Collect notes", "resume": False},
                                  {"id": "notes/check", "title": "Check notes", "resume": False},
                                  {"id": "notes/demo", "title": "Play the demonstration", "resume": False}]
    status, headers, body = client.request("GET", "/ext/notes/notes.js", token=False)
    assert status == 200 and headers["content-type"].startswith("text/javascript")
    assert b'registerView("notes"' in body
    # Only the scripts the extension declares are served.
    assert client.request("GET", "/ext/notes/__init__.py", token=False)[0] == 404


def test_without_extensions_the_header_is_testuarys(serve):
    from testuary.web._server import GUIDE

    _, _, client = serve(extensions=_registry.load([]))
    state = client.json("GET", "/api/state")[1]
    assert state["extensions"] == {"product": None, "tabs": [], "tab_titles": {}, "tab_positions": {},
                                   "hidden_tabs": [], "dir_types": [], "job_kinds": [], "refused": []}
    assert state["guide"] == GUIDE


def test_refused_extensions_name_their_reason():
    tab = WebTab("shared", "Shared", "notes.js")
    registry = _registry.checked([
        _found("later", WebExtension(api=2, id="later")),
        _found("one", WebExtension(api=1, id="one", tabs=(tab,))),
        _found("two", WebExtension(api=1, id="two", tabs=(tab,))),
        _found("takes", WebExtension(api=1, id="takes", tabs=(WebTab("results", "Mine", "notes.js"),))),
        _found("named", WebExtension(api=1, id="named", product="Named")),
        _found("renamed", WebExtension(api=1, id="renamed", product="Again",
                                       tab_titles={"nonexistent": "x"})),
        _found("second", WebExtension(api=1, id="second", product="Second")),
    ])
    reasons = dict(registry.refused)
    assert "extension API version 2" in reasons["later"]
    assert reasons["one"] == reasons["two"] == "another extension has the tab or directory type 'shared'"
    assert reasons["takes"] == "the tab 'results' takes the address of Testuary's own view"
    assert reasons["renamed"] == "it renames tabs Testuary does not have: nonexistent"
    assert reasons["second"] == "the product name is already set by named"
    public = registry.public()
    assert public["product"] == "Named" and public["tabs"] == []
    assert {extension.id for extension, _ in registry.loaded} == {"takes", "named", "second"}


def test_a_view_id_is_one_extensions_and_job_kinds_of_one_id_both_load():
    """A tab's and a directory type's id name one view: two extensions sharing
    one are refused, and one extension may use it for both. Job kinds are
    kept apart by their extension's id. One extension repeating an id is
    refused."""
    def dir_type(view):
        return DirType(view, view.title(), lambda path: False, lambda path: {}, f"{view}.js")

    def kind(title):
        return JobKind("collect", title, "json.tool", lambda form: [])

    first, second = kind("First"), kind("Second")
    registry = _registry.checked([
        _found("tabbed", WebExtension(api=1, id="tabbed", tabs=(WebTab("notes", "Notes", "notes.js"),))),
        _found("dir-typed", WebExtension(api=1, id="dir-typed", dir_types=(dir_type("notes"),))),
        _found("both", WebExtension(api=1, id="both", tabs=(WebTab("own", "Own", "own.js"),),
                                    dir_types=(dir_type("own"),))),
        _found("first", WebExtension(api=1, id="first", job_kinds=(first,))),
        _found("second", WebExtension(api=1, id="second", job_kinds=(second,))),
        _found("repeats", WebExtension(api=1, id="repeats", job_kinds=(first, second))),
        _found("tabs", WebExtension(api=1, id="tabs", tabs=(WebTab("a", "A", "a.js"),
                                                            WebTab("a", "B", "b.js")))),
    ])
    assert dict(registry.refused) == {
        "tabbed": "another extension has the tab or directory type 'notes'",
        "dir-typed": "another extension has the tab or directory type 'notes'",
        "repeats": "two of its job kinds have the id 'collect'",
        "tabs": "two of its tabs have the id 'a'"}
    assert [extension.id for extension, _ in registry.loaded] == ["both", "first", "second"]
    assert registry.job_kind("first/collect") is first and registry.job_kind("second/collect") is second
    assert [tab["id"] for tab in registry.public()["tabs"]] == ["own"]
    assert list(registry.dir_types()) == ["both/own"]


def test_added_tabs_take_their_positions_and_titles_rename_testuarys():
    testuary = [{"id": "", "title": "Overview"}, {"id": "jobs", "title": "Jobs"},
                {"id": "results", "title": "Results"}]
    added = {"tab_titles": {"results": "Test results"}, "tab_positions": {}, "hidden_tabs": [], "tabs": [
        {"id": "profile", "title": "Profiler results", "position": "first"},
        {"id": "notes", "title": "Notes", "position": "before:results"},
        {"id": "later", "title": "Later", "position": "after:nobody"}]}
    tabs = page_value(["headerTabs", "placeTab"], f"headerTabs({json.dumps(testuary)}, {json.dumps(added)})")
    assert [(tab["id"], tab["title"]) for tab in tabs] == [
        ("profile", "Profiler results"), ("", "Overview"), ("jobs", "Jobs"), ("notes", "Notes"),
        ("results", "Test results"), ("later", "Later")]


def test_tab_changes_testuary_refuses_and_a_tabs_title_in_the_header():
    """A tab replaces one of Testuary's tabs but the Overview, an extension
    moves or hides any but the Overview, and one extension replaces, moves
    or hides a tab. A guide is an address or an HTML file's name. The
    registry names a tab as the header does, and a hidden one not at all."""
    twice = WebExtension(api=1, id="twice", tabs=(WebTab("one", "One", "notes.js", replaces="check"),
                                                 WebTab("two", "Two", "notes.js", replaces="check")))

    def replacing(name, replaced, moves=None, hides=()):
        tab = WebTab(name, name.title(), "notes.js", replaces=replaced)
        return _found(name, WebExtension(api=1, id=name, tabs=(tab,), tab_positions=moves or {},
                                         hidden_tabs=hides))

    def moving(name, moves, hides=()):
        return _found(name, WebExtension(api=1, id=name, tab_positions=moves, hidden_tabs=hides))

    def guided(name, guide):
        return _found(name, WebExtension(api=1, id=name, guide=guide))

    registry = _registry.checked([
        replacing("front", ""), replacing("nowhere", "nonexistent"), replacing("mine", "run"),
        replacing("yours", "run"), _found("twice", twice), moving("top", {"": "last"}),
        moving("away", {"nonexistent": "first"}), replacing("both", "settings", {"settings": "first"}),
        replacing("over", "jobs"), moving("under", {"jobs": "first"}),
        moving("here", {"results": "first"}), moving("there", {"results": "last"}),
        moving("hidden", {}, ("",)), moving("gone", {}, ("nonexistent",)),
        moving("late", {"check": "first"}, ("check",)), replacing("covered", "settings", hides=("settings",)),
        moving("unseen", {}, ("models",)), replacing("seen", "models"),
        guided("script", "javascript:alert(1)"), guided("climbing", "../guide.html")])
    reasons = dict(registry.refused)
    assert reasons == {"front": "it replaces the Overview, which stays Testuary's",
                       "nowhere": "it replaces tabs Testuary does not have: nonexistent",
                       "twice": "two of its tabs replace the same tab",
                       "mine": "another extension has the place of the tab 'run'",
                       "yours": "another extension has the place of the tab 'run'",
                       "top": "it moves the Overview, which stays Testuary's",
                       "away": "it moves tabs Testuary does not have: nonexistent",
                       "both": "it replaces, moves or hides these tabs more than once: settings",
                       "over": "another extension has the place of the tab 'jobs'",
                       "under": "another extension has the place of the tab 'jobs'",
                       "here": "another extension has the place of the tab 'results'",
                       "there": "another extension has the place of the tab 'results'",
                       "hidden": "it hides the Overview, which stays Testuary's",
                       "gone": "it hides tabs Testuary does not have: nonexistent",
                       "late": "it replaces, moves or hides these tabs more than once: check",
                       "covered": "it replaces, moves or hides these tabs more than once: settings",
                       "unseen": "another extension has the place of the tab 'models'",
                       "seen": "another extension has the place of the tab 'models'",
                       "script": "its guide is neither an http or https address nor the name of an HTML file",
                       "climbing": "its guide is neither an http or https address nor the name of an HTML file"}
    assert registry.loaded == [] and registry.tab_title("run") == "Run"
    assert (registry.public()["tab_positions"], registry.public()["hidden_tabs"]) == ({}, [])
    renamed = _registry.checked([_found("renamed", WebExtension(api=1, id="renamed",
                                                              tab_titles={"run": "Generate"}))])
    assert (renamed.tab_title("run"), _registry.load([NOTES]).tab_title("run")) == ("Generate", "Notes")
    hiding = _registry.checked([_found("hiding", WebExtension(api=1, id="hiding", hidden_tabs=("run",)))])
    assert (hiding.tab_title("run"), hiding.public()["hidden_tabs"]) == (None, ["run"])
    assert [registry.own_view("check"), renamed.own_view("run"), hiding.own_view("run"),
            _registry.load([NOTES]).own_view("run")] == [True, True, False, False]


def test_the_guide_link_opens_the_first_extensions_guide(serve):
    """The header's Guide link opens Testuary's guide, the first extension's
    address or file of its package, or nothing, and a later guide is refused
    for that field."""
    from testuary.web._server import GUIDE

    def guided(name, guide):
        return _found(name, WebExtension(api=1, id=name, guide=guide))

    first = _registry.checked([guided("own", "https://example.org/guide"), guided("none", None)])
    assert dict(first.refused) == {"none": "the guide is already set by own"}
    assert {extension.id for extension, _ in first.loaded} == {"own", "none"}
    hidden = _registry.checked([guided("none", None), guided("own", "https://example.org/guide")])
    filed = _registry.checked([guided("filed", "guide.html")])
    assert [registry.guide(GUIDE) for registry in (_registry.load([]), first, hidden, filed)] == [
        GUIDE, "https://example.org/guide", None, "/ext/filed/guide.html"]
    _, _, client = serve(extensions=hidden)
    assert client.json("GET", "/api/state")[1]["guide"] is None


def test_the_checks_next_step_names_the_run_tab_as_the_header_shows_it(serve, tmp_path):
    import time

    from testuary.readiness import RunReadiness
    from testuary.web._jobs import Job

    ready = RunReadiness(
        scope="verification", status="verified", exit_code=0, source_dir=str(tmp_path), execution=None,
        source_files=(), phases=(), model_alias=None, provider=None, model=None, credentials_configured=None,
        output_dir=None, output_mode="fresh", limits=(), items=())
    job_dir = tmp_path / "job"
    job_dir.mkdir()
    (job_dir / "stdout.txt").write_text(json.dumps(ready.as_dict()), encoding="utf-8")
    (job_dir / "stderr.txt").write_text("", encoding="utf-8")
    job = Job("check-1", "check", "Check of work", ["check", "--json", "--verify"], job_dir, None, None,
              process=None, status="finished", exit_code=0, ended=time.time())
    steps = []
    hiding = _registry.checked([_found("hiding", WebExtension(api=1, id="hiding", hidden_tabs=("run",)))])
    for registry in (_registry.load([]), _registry.load([NOTES]), hiding):
        app, _, _ = serve(extensions=registry)
        steps += [line for line in app.job_view(job)["report_text"]["summary"] if "Next step" in line]
    assert steps == [
        "Next step: start a run on the Run page. It sends model requests, which may cost money.",
        "Next step: start a run on the Notes page. It sends model requests, which may cost money.",
        "Next step: start a run. It sends model requests, which may cost money."]


def test_a_replaced_or_hidden_tab_leaves_the_header_and_a_position_naming_it_stands_there():
    testuary = [{"id": "", "title": "Overview"}, {"id": "projects", "title": "Projects"},
                {"id": "settings", "title": "Settings"}, {"id": "run", "title": "Run"},
                {"id": "check", "title": "Check"}]

    def header(tab, hidden=()):
        added = {"tab_titles": {}, "tab_positions": {}, "hidden_tabs": list(hidden),
                 "tabs": [{"id": "own", "title": "Own run", "replaces": "run", **tab}]}
        found = page_value(["headerTabs", "placeTab"], f"headerTabs({json.dumps(testuary)}, {json.dumps(added)})")
        return [item["id"] for item in found]

    assert header({"position": "before:projects"}) == ["", "own", "projects", "settings", "check"]
    assert header({"position": "before:run"}) == ["", "projects", "settings", "own", "check"]
    # A hidden tab leaves the header too, and a position naming it stands where it stood.
    assert header({"position": "after:settings", "replaces": None}, ["run", "projects"]) == [
        "", "settings", "own", "check"]
    assert header({"position": "before:projects", "replaces": None}, ["projects"]) == [
        "", "own", "settings", "run", "check"]


def test_a_moved_tab_takes_its_position_whatever_order_the_tabs_are_declared_in():
    """A moved Testuary tab takes its position among the others, and a
    position may name a moved or an added tab, whatever order they are
    declared in. A position no pass can place goes last."""
    testuary = [{"id": "", "title": "Overview"}, {"id": "projects", "title": "Projects"},
                {"id": "settings", "title": "Settings"}, {"id": "run", "title": "Run"},
                {"id": "check", "title": "Check"}, {"id": "results", "title": "Results"}]

    def header(moves, tabs):
        added = {"tab_titles": {"results": "Test results"}, "tab_positions": moves, "hidden_tabs": [], "tabs": tabs}
        found = page_value(["headerTabs", "placeTab"], f"headerTabs({json.dumps(testuary)}, {json.dumps(added)})")
        return [(item["id"], item["title"]) for item in found]

    own = {"id": "own", "title": "Own run", "position": "before:projects", "replaces": "run"}
    later = {"id": "later", "title": "Later", "position": "after:results"}
    assert header({"results": "after:own"}, [later, own]) == [
        ("", "Overview"), ("own", "Own run"), ("results", "Test results"), ("later", "Later"),
        ("projects", "Projects"), ("settings", "Settings"), ("check", "Check")]
    assert header({"results": "after:run"}, [later]) == [
        ("", "Overview"), ("projects", "Projects"), ("settings", "Settings"), ("run", "Run"),
        ("results", "Test results"), ("later", "Later"), ("check", "Check")]
    # Two tabs naming each other, or a tab no one has, go last.
    assert header({"results": "after:later"}, [later, {"id": "lost", "title": "Lost", "position": "after:x"}]) == [
        ("", "Overview"), ("projects", "Projects"), ("settings", "Settings"), ("run", "Run"),
        ("check", "Check"), ("results", "Test results"), ("later", "Later"), ("lost", "Lost")]


OPENED_STUBS = """
const location = { hash: "#/" };
const links = ["", "notes", "run", "jobs"].map((view) => ({ dataset: { view }, current: false,
  setAttribute() { this.current = true; }, removeAttribute() { this.current = false; } }));
const document = { querySelectorAll: () => links };
"""


def test_the_replaced_tabs_address_and_texts_lead_to_the_replacing_tab():
    """The replaced tab's address opens the replacing tab, texts name it by
    its title, and the Check page's note says what Testuary's Run page does
    only while that page is Testuary's and shown. A hidden tab's address
    opens the Overview."""
    def value(added, expression):
        return page_value(["addedTab", "tabTitle", "tabShown", "earlierRunNote"],
                          f"(() => {{ globalThis.extensions = {json.dumps(added)}; "
                          "globalThis.testuaryTitles = new Map([['run', 'Run'], ['models', 'Models']]); "
                          f"return {expression}; }})()")

    empty = {"product": None, "tabs": [], "tab_titles": {}, "hidden_tabs": [], "dir_types": [], "job_kinds": []}
    notes = {**empty, "tabs": [{"id": "notes", "title": "Notes", "position": "before:run",
                                "replaces": "run"}]}
    renamed = {**empty, "tab_titles": {"models": "Model access", "run": "Generate"}}
    assert value(notes, "[addedTab('run').id, addedTab('notes').id, addedTab('check') || null,"
                        " tabTitle('run'), tabTitle('models')]") == ["notes", "notes", None, "Notes", "Models"]
    assert value(renamed, "[addedTab('run') || null, tabTitle('run'), tabTitle('models')]") == [
        None, "Generate", "Model access"]
    unfinished = {"unfinished": ["a.py"], "fresh": "out-2"}
    finished = {"unfinished": [], "fresh": "out-2"}

    def notes_of(added):
        return value(added, f"[earlierRunNote({json.dumps(unfinished)}), "
                            f"earlierRunNote({json.dumps(finished)})]")

    assert notes_of(empty) == [
        ". The check judges continuing that earlier run, as a run started on the Run page does unless you"
        " choose a new directory there.",
        ". The check judges the new directory out-2, where a run started on the Run page starts a new run."]
    assert notes_of(renamed)[1] == (". The check judges the new directory out-2, where a run started on the"
                                    " Generate page starts a new run.")
    assert notes_of(notes) == notes_of({**empty, "hidden_tabs": ["run"]}) == [
        ". The check judges continuing that earlier run.", ". The check judges the new directory out-2."]
    # The address opens the added tab for it, marked current, a hidden tab's
    # address the Overview, and the texts name tabs by tabTitle.
    def opened(added, address):
        return page_value(["addedTab", "tabShown", "addressedView", "markCurrentTab"],
                          f"(() => {{ globalThis.extensions = {json.dumps(added)}; location.hash = '{address}'; "
                          "markCurrentTab(); const { name, arg, tab } = addressedView(); "
                          "return [name, arg || null, tab ? tab.id : null, "
                          "links.filter((link) => link.current).map((link) => link.dataset.view)]; })()",
                          before=OPENED_STUBS)

    assert opened(notes, "#/run?project_dir=p") == ["run", None, "notes", ["notes"]]
    assert opened({**empty, "hidden_tabs": ["run"]}, "#/run") == ["", None, None, [""]]
    assert opened(empty, "#/job/7") == ["job", "7", None, ["jobs"]]
    source = APP.read_text(encoding="utf-8")
    assert "const parts = tab ? await extensionView(" in source
    for text in ("Test the model's connection on the ${tabTitle(\"models\")} page",
                 "Add one on the ${tabTitle(\"models\")} page",
                 "every other setting, on the ${tabTitle(\"settings\")} page.",
                 "Choose on the ${tabTitle(\"run\")} page"):
        assert text in source, text
    # No string names one of those pages in words of its own.
    assert not re.search(r"[\"`][^\"`\n]*\bon the (?:Models|Settings|Run|run) page", source)


def test_pages_name_the_product_and_renamed_tabs_by_their_new_names():
    """Where a page names the whole product it uses the product's name, an
    extension job's panel without one its kind's title, and a renamed tab
    its new title, inside a sentence with a lower-case first letter unless
    it starts with an abbreviation. Without a product name, values come
    from Testuary and its extensions."""
    def named(extensions, expression):
        return page_value(["addedTab", "tabTitle", "tabTitleInText", "productTitle",
                           "productName", "panelTitle"],
                          f"(() => {{ globalThis.extensions = {json.dumps(extensions)}; "
                          "globalThis.testuaryTitles = new Map([['results', 'Results']]); "
                          f"return {expression}; }})()")

    empty = {"product": None, "tabs": [], "tab_titles": {}, "dir_types": [], "job_kinds": []}
    product = {**empty, "product": "Example Suite", "tab_titles": {"results": "Test results"}}
    kind = {"title": "Collect notes"}
    assert named(product, "[panelTitle('Testuary', null), panelTitle('Testuary | demo playback', null),"
                          " productName(), tabTitle('results'), productTitle()]") == [
        "Example Suite", "Example Suite | demo playback", "Example Suite", "Test results", "Example Suite"]
    assert named({**empty, "job_kinds": [{"id": "notes/collect"}]},
                 f"[panelTitle('Testuary', {json.dumps(kind)}), productName(), tabTitle('results')]") == [
        "Collect notes", "Testuary and its extensions", "Results"]
    assert named(empty, "[panelTitle('Testuary', null), productName(), productTitle()]") == [
        "Testuary", "Testuary", "Testuary"]
    abbreviated = {**empty, "tab_titles": {"results": "QA results"}}
    assert [named(case, "tabTitleInText('results')") for case in (empty, product, abbreviated)] == [
        "results", "test results", "QA results"]
    # The Overview's heading and a failed page's heading name the product.
    source = APP.read_text(encoding="utf-8")
    assert 'el("h1", { text: `${productTitle()} in this directory` })' in source
    assert 'el("h1", { text: productTitle() }), el("p", { class: "panel", text: error.message })' in source
    # The links to the Results page name it by its title.
    for text in ('text: `Open ${tabTitleInText("results")}`', 'text: `Open the ${tabTitleInText("results")}`',
                 'resultsLink: (dir, text = `Open the ${tabTitleInText("results")}`)'):
        assert text in source, text
    assert not re.search(r"[\"`]Open (?:the )?results", source)


def test_every_link_to_a_testuary_tab_asks_whether_the_tab_shows():
    """Each address of a Testuary tab the page links to, outside that tab's
    own page and the helpers that build its address, asks tabShown for the
    same tab on its line or the line before, so a hidden tab's links are
    left out with their words."""
    own_pages = {"settingsLink", "modelsLink", "settingsOf", "warningList", "projectRuns", "resultsView"}
    source = APP.read_text(encoding="utf-8")
    function, before, unguarded = None, "", []
    for number, line in enumerate(source.splitlines(), 1):
        started = re.match(r"(?:async )?function (\w+)", line)
        function = started.group(1) if started else function
        for tab in re.findall(r"#/(projects|models|settings|run|check|jobs|results)\b", line):
            if (function not in own_pages and not line.lstrip().startswith("//")
                    and f'tabShown("{tab}")' not in before + line):
                unguarded.append(f"{number}: {line.strip()}")
        before = line
    assert unguarded == []
    # Each sentence keeps its own words.
    for text in ('"No model yet.", tabShown("models") && [" ", el("a"',
                 '"No project yet.", tabShown("projects") && [" ", el("a"',
                 'tabShown("results") ? "Latest results, written " : "The latest results were written "',
                 'tabShown("results") ? el("a", { href: `#/results?run_dir=${encodeURIComponent(dir)}`, text })'
                 ' : null'):
        assert text in source, text


def _guide_package(tmp_path, monkeypatch, name):
    """An importable package ``name`` with the guide file guide.html, and a
    file outside.html beside the package."""
    package = tmp_path / name
    package.mkdir()
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "guide.html").write_text("<h1>Own guide</h1>", encoding="utf-8")
    (tmp_path / "outside.html").write_text("<h1>Outside</h1>", encoding="utf-8")
    monkeypatch.syspath_prepend(str(tmp_path))
    monkeypatch.delitem(sys.modules, name, raising=False)
    return package


def test_a_guide_file_is_served_from_its_package_only(serve, tmp_path, monkeypatch):
    package = _guide_package(tmp_path, monkeypatch, "guided_package")
    registry = _registry.checked([_found("guided", WebExtension(api=1, id="guided", guide="guide.html"),
                                         folder=package)])
    _, _, client = serve(extensions=registry)
    assert client.json("GET", "/api/state")[1]["guide"] == "/ext/guided/guide.html"
    status, headers, body = client.request("GET", "/ext/guided/guide.html", token=False)
    assert (status, body) == (200, b"<h1>Own guide</h1>") and headers["content-type"].startswith("text/html")
    for path in ("/ext/guided/../outside.html", "/ext/guided/..%2Foutside.html", "/ext/guided/outside.html",
                 "/ext/guided/__init__.py"):
        assert client.request("GET", path, token=False)[0] == 404, path
    climbing = WebExtension(api=1, id="climbing", guide="../outside.html")
    assert dict(_registry.checked([_found("climbing", climbing, folder=package)]).refused) == {
        "climbing": "its guide is neither an http or https address nor the name of an HTML file"}


def test_the_tool_check_label_is_testuarys_one_value(serve, monkeypatch):
    """A replacing Check view takes the label from the state, and Testuary's
    own report quotes it: a renamed label reaches both, and the page holds
    no copy of it."""
    from testuary import web
    from testuary.web import _jobs

    _, _, client = serve(extensions=_registry.load([]))
    assert client.json("GET", "/api/state")[1]["tool_check_label"] == web.TOOL_CHECK_LABEL
    monkeypatch.setattr(extensions, "TOOL_CHECK_LABEL", "Renamed tool check")
    assert client.json("GET", "/api/state")[1]["tool_check_label"] == "Renamed tool check"
    steps = _jobs._page_steps(Registry())
    assert steps.verify == 'the check with "Renamed tool check" ticked'
    assert steps.check_tools == 'check again with "Renamed tool check" ticked'
    assert web.TOOL_CHECK_LABEL not in APP.read_text(encoding="utf-8")


def test_a_declared_file_its_package_does_not_ship_is_not_found(serve, tmp_path, monkeypatch):
    package = _guide_package(tmp_path, monkeypatch, "shipless_package")
    shipless = WebExtension(api=1, id="shipless", guide="missing.html",
                            tabs=(WebTab("absent", "Absent", "absent.js"),))
    registry = _registry.checked([_found("shipless", shipless, folder=package)])
    assert registry.refused == []
    _, _, client = serve(extensions=registry)
    for path in ("/ext/shipless/missing.html", "/ext/shipless/absent.js"):
        assert client.request("GET", path, token=False)[0] == 404, path


def test_a_guide_file_that_links_out_of_its_package_is_not_served(serve, tmp_path, monkeypatch):
    package = _guide_package(tmp_path, monkeypatch, "linked_package")
    try:
        os.symlink(tmp_path / "outside.html", package / "linked.html")
    except OSError:
        pytest.skip("this system lets the test create no links")
    registry = _registry.checked([_found("linked", WebExtension(api=1, id="linked", guide="linked.html"),
                                         folder=package)])
    _, _, client = serve(extensions=registry)
    assert client.request("GET", "/ext/linked/guide.html", token=False)[0] == 404
    assert client.request("GET", "/ext/linked/linked.html", token=False)[0] == 404


def test_a_dir_type_opens_its_dir_and_results_link_to_it(serve, tmp_path):
    run_dir = tmp_path / "work" / "profiled"
    (run_dir / "results").mkdir(parents=True)
    (run_dir / "logs").mkdir()
    (run_dir / "notes.json").write_text(json.dumps({"count": 1, "notes": ["Note 1"]}), encoding="utf-8")
    _, _, client = serve(extensions=_registry.load([NOTES]))
    status, found = client.json("GET", f"/api/ext/dir?type=notes/notes&dir={run_dir}")
    assert (status, found["summary"]) == (200, {"count": 1, "notes": ["Note 1"]})
    status, summary = client.json("GET", f"/api/results?run_dir={run_dir}")
    assert (status, summary["views"]) == (200, [{"type": "notes/notes", "title": "Notes"}])
    # The extension's own directory needs no run directory's marks.
    plain = tmp_path / "work" / "plain"
    plain.mkdir()
    (plain / "notes.json").write_text(json.dumps({"count": 0, "notes": []}), encoding="utf-8")
    assert client.json("GET", f"/api/ext/dir?type=notes/notes&dir={plain}")[1]["summary"]["count"] == 0
    (run_dir / "notes.json").unlink()
    assert client.json("GET", f"/api/ext/dir?type=notes/notes&dir={run_dir}")[0] == 404


def test_a_settings_group_changes_only_through_its_validated_update(serve, user_dir):
    _, _, client = serve(extensions=_registry.load([NOTES]))
    sections = client.json("GET", "/api/settings")[1]["sections"]
    notes = next(section for section in sections if section.get("group") == "notes/notes")
    assert [(row["name"], row["value"], row["group"]) for row in notes["settings"]] == [
        ("count", "3", "notes/notes")]
    status, refused = client.json("POST", "/api/ext/settings", {
        "group": "notes/notes", "section": "notes", "values": {"count": "500"}})
    assert (status, refused["error"]) == (400, "count: expected a whole number from 1 to 100")
    assert not (user_dir / "notes.ini").exists()
    assert client.json("POST", "/api/ext/settings", {
        "group": "notes/notes", "section": "notes", "values": {"count": "7"}})[0] == 200
    sections = client.json("GET", "/api/settings")[1]["sections"]
    notes = next(section for section in sections if section.get("group") == "notes/notes")
    assert notes["settings"][0]["value"] == "7"


def _notes_project(tmp_path, user_dir):
    """A listed project whose Notes settings name shared/testuary.ini and then
    .notes/testuary.ini, with values in both, in the project's settings file
    and in the global configuration."""
    project_dir = tmp_path / "calc"
    (project_dir / ".testuary").mkdir(parents=True)
    (project_dir / ".testuary" / "project.ini").write_text("[starter]\ntemperature = 0.2\n", encoding="utf-8")
    (project_dir / "shared").mkdir()
    (project_dir / "shared" / "testuary.ini").write_text("[coverage]\ngoal = 70\n[starter]\nsuite_count = 4\n",
                                                    encoding="utf-8")
    (project_dir / ".notes").mkdir()
    (project_dir / ".notes" / "testuary.ini").write_text("# Notes' runs.\n[coverage]\ngoal = 75\n",
                                                   encoding="utf-8")
    (user_dir / "config.ini").write_text("[mutation]\ngoal = 60\n", encoding="utf-8")
    projects.add(str(project_dir))
    return project_dir


def _rows(view):
    return {(row["section"], row["name"]): (row["value"], row["source"])
            for section in view["sections"] for row in section["settings"]}


def _notes_group(files):
    """The worked example's settings group, naming the list ``files``."""
    import web_extension_fixture as notes

    return SettingsGroup("notes", "Notes", notes.reference, notes.values, notes.update, files)


def test_a_named_list_of_settings_files_is_read_in_place_of_the_global_and_the_project_settings(
        serve, tmp_path, user_dir):
    """The settings as the extension's runs in the project read them, from
    Testuary's defaults and the named files in order, without the global
    configuration and the project's settings file. A change goes to the last
    file, and Reset takes an earlier file's value out of it again."""
    project_dir = _notes_project(tmp_path, user_dir)
    shared = project_dir.resolve() / "shared" / "testuary.ini"
    named = project_dir.resolve() / ".notes" / "testuary.ini"
    group = _notes_group(lambda project: [Path("shared", "testuary.ini"), Path(".notes", "testuary.ini")])
    _, _, client = serve(extensions=_registry.checked([
        _found("notes", WebExtension(api=1, id="notes", settings_groups=(group,)))]))
    query = f"/api/settings?project_dir={quote(str(project_dir))}"
    status, view = client.json("GET", query)
    assert (status, view["error"]) == (200, None)
    assert view["scopes"] == [{"scope": "notes/notes", "title": "Notes", "files": [str(shared), str(named)],
                               "file": str(named), "exists": True, "read_only": None, "global": False,
                               "error": None}]
    status, view = client.json("GET", f"{query}&scope=notes/notes")
    assert (status, view["error"]) == (200, None)
    assert view["files"] == {"global": None, "project": None, "named": str(named),
                             "list": [str(shared), str(named)]}
    rows = _rows(view)
    assert rows[("coverage", "goal")] == ("75", str(named))
    assert rows[("starter", "suite_count")] == ("4", str(shared))
    assert rows[("starter", "temperature")][1] == "default"
    assert rows[("mutation", "goal")] == ("80", "default")
    assert not any(section.get("group") for section in view["sections"])
    body = {"scope": "notes/notes", "project_dir": str(project_dir), "section": "coverage"}
    assert client.json("POST", "/api/settings", {**body, "values": {"goal": "80"}})[0] == 200
    assert named.read_text(encoding="utf-8") == "# Notes' runs.\n[coverage]\ngoal = 80\n"
    status, refused = client.json("POST", "/api/settings", {**body, "values": {"goal": "most"}})
    assert status == 400 and "goal" in refused["error"]
    assert named.read_text(encoding="utf-8") == "# Notes' runs.\n[coverage]\ngoal = 80\n"
    starter = {**body, "section": "starter"}
    assert client.json("POST", "/api/settings", {**starter, "values": {"suite_count": "6"}})[0] == 200
    assert "suite_count = 6" in named.read_text(encoding="utf-8")
    assert _rows(client.json("GET", f"{query}&scope=notes/notes")[1])[("starter", "suite_count")] == (
        "6", str(named))
    assert client.json("POST", "/api/settings", {**starter, "remove": ["suite_count"]})[0] == 200
    assert "suite_count" not in named.read_text(encoding="utf-8")
    assert _rows(client.json("GET", f"{query}&scope=notes/notes")[1])[("starter", "suite_count")] == (
        "4", str(shared))
    assert shared.read_text(encoding="utf-8") == "[coverage]\ngoal = 70\n[starter]\nsuite_count = 4\n"
    assert (user_dir / "config.ini").read_text(encoding="utf-8") == "[mutation]\ngoal = 60\n"
    assert (project_dir / ".testuary" / "project.ini").read_text(encoding="utf-8") == (
        "[starter]\ntemperature = 0.2\n")


def test_a_named_settings_file_that_does_not_exist_is_made_by_the_first_change(
        serve, tmp_path, user_dir):
    project_dir = _notes_project(tmp_path, user_dir)
    named = project_dir.resolve() / ".notes" / "testuary.ini"
    named.unlink()
    named.parent.rmdir()
    _, _, client = serve(extensions=_registry.load([NOTES]))
    query = f"/api/settings?project_dir={quote(str(project_dir))}"
    assert client.json("GET", query)[1]["scopes"][0]["exists"] is False
    rows = _rows(client.json("GET", f"{query}&scope=notes/notes")[1])
    assert (rows[("coverage", "goal")], rows[("mutation", "goal")]) == (("90", "default"), ("80", "default"))
    assert client.json("POST", "/api/settings", {"scope": "notes/notes", "project_dir": str(project_dir),
                                                 "section": "coverage", "values": {"goal": "75"}})[0] == 200
    assert "[coverage]\ngoal = 75\n" in named.read_text(encoding="utf-8")


def test_a_named_settings_file_in_the_project_takes_no_model_address(serve, tmp_path, user_dir):
    """A settings group's file inside the project's directory takes no
    api_url, as the extension's runs refuse it there
    (check_project_settings_file): the change is refused and the file stays
    as it is."""
    project_dir = _notes_project(tmp_path, user_dir)
    named = project_dir.resolve() / ".notes" / "testuary.ini"
    before = named.read_bytes()
    _, _, client = serve(extensions=_registry.load([NOTES]))
    status, refused = client.json("POST", "/api/settings", {
        "scope": "notes/notes", "project_dir": str(project_dir), "section": "api",
        "values": {"api_url": "http://127.0.0.1:9/v1"}})
    assert status == 400 and "api_url" in refused["error"], refused
    assert named.read_bytes() == before


def _link(link, target):
    """A link to the directory ``target``: a symbolic link, else on Windows a
    junction, which needs no privilege."""
    try:
        link.symlink_to(target, target_is_directory=True)
    except OSError:
        if os.name != "nt":
            raise
        import _winapi

        _winapi.CreateJunction(str(target), str(link))


def test_a_named_list_outside_the_project_failing_or_global_is_read_only_or_refused(
        serve, tmp_path, user_dir):
    """A last file outside the project dir, through a link or not, and
    the global configuration as the last file are read-only with the reason,
    and a change writes nothing. A dir, a missing earlier file and a
    failing function each give their error, and nothing is read or written.
    A group that names no list offers no choice."""
    project_dir = _notes_project(tmp_path, user_dir)
    outside = tmp_path / "shared"
    outside.mkdir()
    (outside / "shared.ini").write_text("[coverage]\ngoal = 10\n", encoding="utf-8")
    _link(project_dir / "linked", outside)

    def failing(project):
        raise RuntimeError("no notes settings")

    named = {"outside": lambda project: [Path("..", "shared", "shared.ini")],
             "linked": lambda project: [Path("linked", "shared.ini")],
             "global": lambda project: [user_dir / "config.ini"],
             "nested": lambda project: [Path(".notes")],
             "missing": lambda project: [Path("absent.ini"), Path(".notes", "testuary.ini")],
             "failing": failing, "none": lambda project: None}
    groups = tuple(SettingsGroup(key, key.title(), group.reference, group.values, group.update, function)
                   for key, function in named.items() for group in [_notes_group(function)])
    _, _, client = serve(extensions=_registry.checked([
        _found("notes", WebExtension(api=1, id="notes", settings_groups=groups))]))
    query = f"/api/settings?project_dir={quote(str(project_dir))}"
    status, view = client.json("GET", query)
    assert (status, view["error"]) == (200, None)
    entries = {entry["scope"]: entry for entry in view["scopes"]}
    assert {scope: entry["error"] for scope, entry in entries.items() if entry["error"]} == {
        "notes/nested": "Nested names .notes, which is not a file.",
        "notes/missing": "Missing names absent.ini, which is not a file.",
        "notes/failing": "The extension's settings files failed: RuntimeError: no notes settings"}
    assert {scope: entry["read_only"] for scope, entry in entries.items() if entry["read_only"]} == {
        f"notes/{key}": f"{key.title()} names {named[key](project_dir)[-1]} last, which leads to "
                        f"{(project_dir / named[key](project_dir)[-1]).resolve()}, outside the project directory, so this "
                        "page does not change it." for key in ("outside", "linked")} | {
        "notes/global": "These runs read your global settings last. Change them on the global settings page."}
    assert entries["notes/global"]["global"] is True
    for scope, entry in entries.items():
        status, shown = client.json("GET", f"{query}&scope={scope}")
        if entry["error"]:
            assert (status, shown["error"], shown["sections"]) == (200, entry["error"], [])
        else:
            assert (status, shown["error"]) == (200, None) and shown["sections"]
        status, refused = client.json("POST", "/api/settings", {
            "scope": scope, "project_dir": str(project_dir), "section": "coverage", "values": {"goal": "75"}})
        assert (status, refused["error"]) == (400, entry["error"] or entry["read_only"])
    assert _rows(client.json("GET", f"{query}&scope=notes/outside")[1])[("coverage", "goal")] == (
        "10", str((outside / "shared.ini").resolve()))
    assert (outside / "shared.ini").read_text(encoding="utf-8") == "[coverage]\ngoal = 10\n"
    assert (user_dir / "config.ini").read_text(encoding="utf-8") == "[mutation]\ngoal = 60\n"
    assert not (project_dir / "absent.ini").exists()
    # A scope the project has no list for answers as a refused list does, so
    # the page keeps its choice, and a change to it is refused.
    status, missing = client.json("GET", f"{query}&scope=notes/none")
    assert (status, missing["error"], missing["sections"]) == (200, "This project has no such settings file.", [])
    assert {entry["scope"] for entry in missing["scopes"]} >= {"notes/outside", "notes/global"}
    assert client.json("POST", "/api/settings", {"scope": "notes/none", "project_dir": str(project_dir),
                                                 "section": "coverage", "values": {"goal": "75"}})[0] == 404
    assert client.json("GET", "/api/settings")[1]["error"] is None


def test_the_settings_page_reads_links_and_badges_a_named_list_of_settings_files():
    project = {"project_dir": "C:/p q", "name": "calc"}
    assert page_value(["settingsRequests"], "settingsRequests('notes/notes', %s)" % json.dumps(project)) == [
        "/api/settings?project_dir=C%3A%2Fp%20q&scope=notes%2Fnotes"]
    assert page_value(["settingsLink"], "settingsLink('C:/p q', null, 'notes/notes')") == (
        "#/settings?project_dir=C%3A%2Fp%20q&scope=notes%2Fnotes")
    files = json.dumps({"global": None, "project": "P", "named": "N", "list": ["C:/a/b/c/shared.ini", "N"]})
    assert page_value(["sourceWords", "levelWords", "shortPath"],
                      "[sourceWords('N', %s).badge, sourceWords('P', %s).badge, "
                      "sourceWords('C:/a/b/c/shared.ini', %s).badge, levelWords('N', %s)]"
                      % (files, files, files, files)) == ["This file", "This project", ".../c/shared.ini",
                                                          "This file"]


def test_a_route_answers_get_with_what_the_extension_read_from_its_dir(serve, tmp_path):
    notes_dir = tmp_path / "work" / "notes"
    notes_dir.mkdir(parents=True)
    (notes_dir / "notes.json").write_text(json.dumps({"count": 2, "notes": ["a", "b"]}), encoding="utf-8")
    _, _, client = serve(extensions=_registry.load([NOTES]))
    assert client.json("GET", f"/api/ext/notes/count?dir={notes_dir}") == (200, {"count": 2})
    status, refused = client.json("GET", "/api/ext/notes/count")
    assert (status, refused["error"]) == (400, "Name a notes directory with dir.")


def _probe(calls):
    """An extension whose routes count their calls, one that fails, and a
    route and a directory type that return the directory itself, which JSON
    cannot hold."""
    def read(named_dir, query):
        calls.append((named_dir, dict(query)))
        return {"dir": str(named_dir)}

    def fail(named_dir, query):
        raise RuntimeError("the probe broke")

    return Registry([(WebExtension(
        api=1, id="probe", routes=(WebRoute("calls", read), WebRoute("deep/fails", fail),
                                   WebRoute("raw", lambda named_dir, query: {"dir": named_dir}),
                                   WebRoute("text", lambda named_dir, query: "abc")),
        dir_types=(DirType("raw", "Raw", lambda named_dir: True,
                                 lambda named_dir: {"dir": named_dir}, "raw.js"),)),
        "web_extension_fixture")])


def test_a_route_answers_only_get_and_only_with_the_key(serve):
    calls = []
    _, _, client = serve(extensions=_probe(calls))
    assert client.json("GET", "/api/ext/probe/calls?x=1") == (200, {"dir": "None"})
    assert client.json("POST", "/api/ext/probe/calls", {})[0] == 404
    assert client.request("GET", "/api/ext/probe/calls", token=False)[0] == 403
    assert calls == [(None, {"x": "1"})]


def test_a_route_reaches_only_its_prefix_and_the_dirs_results_reach(serve, tmp_path):
    calls = []
    _, _, client = serve(extensions=_probe(calls))
    for path in ("/api/ext/probe/../state", "/api/ext/probe/calls/../../state", "/api/ext/other/calls"):
        assert client.request("GET", path)[0] == 404, path
    outside = tmp_path / "outside"
    outside.mkdir()
    assert client.json("GET", f"/api/ext/probe/calls?dir={outside}")[0] == 404
    assert calls == []


def test_a_failing_route_answers_its_error_and_the_rest_works(serve):
    _, _, client = serve(extensions=_probe([]))
    status, failed = client.json("GET", "/api/ext/probe/deep/fails")
    assert (status, failed["error"]) == (500, "The extension's route failed: RuntimeError: the probe broke")
    assert client.json("GET", "/api/state")[0] == 200
    assert client.request("GET", "/")[0] == 200


def test_an_answer_json_cannot_hold_answers_its_error(serve, tmp_path):
    named_dir = tmp_path / "work" / "raw"
    named_dir.mkdir(parents=True)
    _, _, client = serve(extensions=_probe([]))
    # A Path is no JSON, and a text is no mapping: both are the extension's
    # failure, never its refusal.
    for path, what, error in ((f"/api/ext/probe/raw?dir={named_dir}", "route", "TypeError"),
                              (f"/api/ext/dir?type=probe/raw&dir={named_dir}", "directory summary", "TypeError"),
                              ("/api/ext/probe/text", "route", "ValueError")):
        status, failed = client.json("GET", path)
        assert status == 500, path
        assert failed["error"].startswith(f"The extension's {what} failed: {error}: "), failed
    assert client.json("GET", "/api/state")[0] == 200


def test_reserved_ids_and_malformed_route_paths_are_refused():
    registry = _registry.checked([
        _found("dir", WebExtension(api=1, id="dir")),
        _found("paths", WebExtension(api=1, id="paths", routes=(WebRoute("../up", dict),))),
    ])
    reasons = dict(registry.refused)
    assert reasons["dir"] == "its id 'dir' is a name of Testuary's own extension API"
    assert reasons["paths"].startswith("its routes' paths are not")


def _children_import_the_fixture(monkeypatch):
    paths = [str(HERE), str(SOURCE)] + [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join(paths))


def test_a_job_kind_runs_its_module_and_its_page_draws_the_plan(serve, tmp_path, monkeypatch):
    from testuary.web import _jobs

    _children_import_the_fixture(monkeypatch)
    app, _, client = serve(extensions=_registry.load([NOTES]))
    status, refused = client.json("POST", "/api/ext/jobs", {"kind": "notes/collect", "dir": ""})
    assert (status, refused["error"]) == (400, "Name the directory the notes go to.")
    notes_dir = tmp_path / "notes-out"
    status, job = client.json("POST", "/api/ext/jobs", {"kind": "notes/collect", "dir": str(notes_dir)})
    assert status == 201 and job["kind"] == "notes/collect"
    assert job["command"] == f"python -m web_extension_fixture.job --dir {notes_dir} --count 3"
    assert wait_for(lambda: client.json("GET", f"/api/jobs/{job['id']}")[1]["status"] != "running")
    ended = client.json("GET", f"/api/jobs/{job['id']}")[1]
    assert (ended["status"], ended["meaning"]) == ("finished", "The job finished.")
    # A kind that names no nested runs keeps its job's view as it was.
    assert "nested_runs" not in ended
    assert json.loads((notes_dir / "notes.json").read_text(encoding="utf-8"))["count"] == 3
    # The job page draws the plan the job wrote through the public writer.
    progress = _jobs.Progress()
    for line in app.jobs.get(job["id"]).events.read_text(encoding="utf-8").splitlines():
        progress.add(line)
    assert [phase["label"] for phase in progress.diagram()["phases"]] == ["Collect notes"]
    # Testuary's own Finishing says what it does, the job's last step its label (one criterion).
    from testuary.presentation.view import own_finishing

    assert progress.diagram()["finishing"]["purpose"] is None
    assert (own_finishing(""), own_finishing("Finishing"), own_finishing("Delivery")) == (True, True, False)


def test_an_extension_jobs_readiness_record_is_drawn_as_its_setup_report(serve, tmp_path, monkeypatch):
    _children_import_the_fixture(monkeypatch)
    _, _, client = serve(extensions=_registry.load([NOTES]))
    taken = tmp_path / "work" / "taken"
    taken.parent.mkdir(parents=True, exist_ok=True)
    taken.write_text("a file, not a directory", encoding="utf-8")
    status, job = client.json("POST", "/api/ext/jobs", {"kind": "notes/check", "dir": str(taken)})
    assert status == 201
    assert wait_for(lambda: client.json("GET", f"/api/jobs/{job['id']}")[1]["status"] != "running")
    ended = client.json("GET", f"/api/jobs/{job['id']}")[1]
    report, texts = ended["report"], ended["report_text"]
    assert [(item["key"], item["status"], item["reason"]) for item in report["items"]] == [
        ("notes_dir", "missing_setup", "The path names a file.")]
    assert [(limit["phase"], limit["name"], limit["value"]) for limit in report["limits"]] == [
        ("collect", "count", 3)]
    # Testuary's presenter wrote the summary, as for its own check.
    assert any("The path names a file." in line for line in texts["summary"])
    assert texts["items"]["notes_dir"]["name"] == "notes directory"
    # The page draws the report of every job that has one.
    source = APP.read_text(encoding="utf-8")
    assert "if (current.report) reportBox.replaceChildren(reportView(" in source
    assert "else parts.push(progressBox, reportBox);" in source


def _run_event(state, message):
    """A run's own observation, as its event file holds it."""
    from testuary.presentation import DisplayEvent

    return DisplayEvent(run_id="r0", target="all targets", phase="pipeline", operation="run",
                        state=state, elapsed_s=1.0, message=message, level="run")


def test_a_nested_run_follows_its_event_file_until_the_run_ends(serve, tmp_path):
    """nestedRun's route draws an event file through Testuary's view model,
    as a job's stream does, follows it while the run writes it and ends
    once its last run event has ended."""
    from testuary.presentation import EventFile, RunPlan
    from testuary.presentation.events import PlannedPhase, PlannedStep

    _, _, client = serve()
    path = tmp_path / "work" / "runs" / "events.jsonl"
    path.parent.mkdir(parents=True)
    events = EventFile(path).__enter__()
    events.plan(RunPlan(files=("a.py",), runs=("r0",), phases=(PlannedPhase("coverage", "Coverage"),),
                        finishing=PlannedStep("finishing", "Finishing")))
    events(_run_event("running", "Running"))

    def finish():
        time.sleep(1.5)
        events(_run_event("succeeded", "Run completed"))
        events.__exit__(None, None, None)

    threading.Thread(target=finish, daemon=True).start()
    found, payload = client.events(f"/api/nested-run?path={quote(str(path))}")
    views = [data for kind, data in found if kind == "view"]
    assert found[-1] == ("end", {}), payload[-2000:]
    assert (views[0]["message"], views[-1]["message"]) == ("Running", "Run completed")
    assert [phase["label"] for phase in views[-1]["diagram"]["phases"]] == ["Coverage"]
    # A relative path starts from the working directory, as Results' does.
    assert client.events("/api/nested-run?path=runs/events.jsonl")[0][-1] == ("end", {})


def test_a_nested_run_replays_an_event_file_longer_than_one_read(serve, tmp_path):
    """An ended run's event file longer than the server reads at a time is
    replayed to its last record before the end."""
    from testuary.presentation import EventFile
    from testuary.web import _jobs

    _, _, client = serve()
    path = tmp_path / "work" / "long.jsonl"
    with EventFile(path) as events:
        while path.stat().st_size <= 2 * _jobs.READ_CHUNK:
            for step in range(500):
                events(_run_event("running", f"Step {step}"))
        events(_run_event("succeeded", "Run completed"))
    found, _ = client.events("/api/nested-run?path=long.jsonl")
    views = [data for kind, data in found if kind == "view"]
    assert found[-1] == ("end", {}) and views[-1]["message"] == "Run completed"


def test_a_nested_run_dir_ends_with_what_results_shows(serve, tmp_path):
    """nestedRun's route answers a run directory with what the Results view
    shows of it, in its end."""
    _, _, client = serve()
    run = tmp_path / "work" / "out"
    (run / "results").mkdir(parents=True)
    (run / "logs").mkdir()
    found, _ = client.events("/api/nested-run?path=out")
    assert [kind for kind, _ in found] == ["end"]
    assert Path(found[0][1]["results"]["root"]).resolve() == run.resolve()
    assert found[0][1]["results"]["runs"] == client.json("GET", "/api/results?run_dir=out")[1]["runs"]


def test_a_nested_run_reads_only_what_results_reaches(serve, tmp_path):
    """nestedRun's route refuses a path outside the working directory and the
    listed projects' directories, a configuration file under any case, and a
    link or junction on the way or at the end, before it reads anything."""
    _, _, client = serve()
    work = tmp_path / "work"
    real = work / "real"
    real.mkdir()
    (real / "events.jsonl").write_text("", encoding="utf-8")
    (real / "Models.INI").write_text("[alias]\napi_key = canary\n", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()
    (outside / "events.jsonl").write_text("", encoding="utf-8")

    def refusal(path):
        # A path the route reads answers 200 and streams until its run
        # ends, so its status fails the test before any of its body.
        return client.json("GET", f"/api/nested-run?path={quote(str(path))}", expect=404)[1]["error"]

    assert "configuration file" in refusal(real / "Models.INI")
    assert "configuration file" in refusal("real/Models.INI")
    assert "Only directories below the working directory" in refusal(outside / "events.jsonl")
    assert "does not exist" in refusal(real / "missing.jsonl")
    made = []

    def link(target, path, directory):
        try:
            os.symlink(target, path, target_is_directory=directory)
        except OSError:
            return
        made.append(path)

    link(real / "events.jsonl", work / "linked.jsonl", False)
    link(real, work / "linked-dir", True)
    if os.name == "nt":
        import _winapi

        _winapi.CreateJunction(str(real), str(work / "junction"))
        made.append(work / "junction")
    if not made:
        pytest.skip("this system lets the test create neither links nor junctions")
    for path in made:
        assert "link" in refusal(path)
        if path.name != "linked.jsonl":
            assert "link" in refusal(path / "events.jsonl")


def test_a_job_kinds_failing_nested_runs_leave_its_job_working(serve, tmp_path, monkeypatch):
    """A nested_runs that raises gives no runs and its error in the job's
    view, and the job runs and ends as before."""
    import web_extension_fixture

    def broken(arguments):
        raise RuntimeError("no runs today")

    _children_import_the_fixture(monkeypatch)
    kind = JobKind("broken", "Broken runs", "web_extension_fixture.job", web_extension_fixture.command,
                   nested_runs=broken)
    registry = _registry.checked([_found("broken", WebExtension(api=1, id="broken", job_kinds=(kind,)))])
    _, _, client = serve(extensions=registry)
    status, job = client.json("POST", "/api/ext/jobs", {"kind": "broken/broken", "dir": str(tmp_path / "notes")})
    assert status == 201
    assert (job["nested_runs"], job["nested_runs_error"]) == (
        [], "The extension's nested runs failed: RuntimeError: no runs today")
    found, _ = client.stream(job["id"])
    assert found[-1][1]["status"] == "finished"
    assert [data["diagram"]["phases"][0]["label"] for kind, data in found
            if kind == "view" and data["diagram"]][-1] == "Collect notes"


def test_the_worked_examples_demonstration_is_folded_into_its_phase(serve, tmp_path, monkeypatch):
    """The guide's Play the demonstration names Testuary's demonstration as
    its phase's nested run, whose event file and directory a nested run reads,
    and refuses a notes directory that holds one."""
    _children_import_the_fixture(monkeypatch)
    monkeypatch.setenv("TESTUARY_DEMO_PLAYBACK_SECONDS", "2")
    _, _, client = serve(extensions=_registry.load([NOTES]))
    notes_dir = tmp_path / "work" / "notes"
    status, job = client.json("POST", "/api/ext/jobs", {"kind": "notes/demo", "dir": str(notes_dir)})
    assert status == 201
    events = notes_dir / "demo-events.jsonl"
    assert job["nested_runs"] == [{"phase": "play", "path": str(events), "title": "Testuary's demonstration"}]
    found, payload = client.stream(job["id"])
    assert (found[-1][1]["status"], found[-1][1]["exit_code"]) == ("finished", 0), payload[-3000:]
    played, _ = client.events(f"/api/nested-run?path={quote(str(events))}")
    views = [data for kind, data in played if kind == "view"]
    assert played[-1] == ("end", {}) and views[-1]["title"] == "Testuary | demo playback"
    shown, _ = client.events(f"/api/nested-run?path={quote(str(notes_dir / 'demo'))}")
    assert shown[-1][1]["results"]["demo"]
    assert json.loads((notes_dir / "notes.json").read_text(encoding="utf-8"))["demo"] == "demo"
    status, refused = client.json("POST", "/api/ext/jobs", {"kind": "notes/demo", "dir": str(notes_dir)})
    assert (status, refused["error"]) == (400, "This notes directory already holds a demonstration.")


def test_a_later_plan_replaces_the_plan(tmp_path):
    """A job whose first phase runs once for the project writes a plan
    without files, then a plan with its files in which that phase is
    reused."""
    from testuary.presentation import DisplayEvent, EventFile, RunPlan
    from testuary.presentation.events import PlannedPhase, PlannedStep
    from testuary.web import _jobs

    def event(target, phase, state):
        return DisplayEvent(run_id="r0", target=target, phase=phase, operation="phase/run-0",
                            state=state, elapsed_s=1.0, message=phase, level="phase")

    finishing = PlannedStep("summary", "Write the summary")
    path = tmp_path / "events.jsonl"
    with EventFile(path) as events:
        events.plan(RunPlan(files=(), runs=("r0",), phases=(PlannedPhase("profiling", "Profiling"),),
                            finishing=finishing))
        events(event("project", "profiling", "succeeded"))
        events.plan(RunPlan(files=("a.py", "b.py"), runs=("r0",), finishing=finishing, phases=(
            PlannedPhase("profiling", "Profiling", state="reused"), PlannedPhase("improve", "Improve"))))
        events(event("a.py", "improve", "running"))
    progress = _jobs.Progress()
    lines = path.read_text(encoding="utf-8").splitlines()
    for line in lines[:3]:
        progress.add(line)
    first = progress.diagram()
    assert (first["files"], [(phase["label"], phase["state"]) for phase in first["phases"]]) == (
        [], [("Profiling", "succeeded")])
    for line in lines[3:]:
        progress.add(line)
    later = progress.diagram()
    assert (later["target"], later["files"]) == ("a.py", ["a.py", "b.py"])
    assert [(phase["label"], phase["state"]) for phase in later["phases"]] == [
        ("Profiling", "reused"), ("Improve", "running")]
    assert progress.snapshot()["diagram"] == later


def test_a_view_reads_the_named_fields_of_the_projects_and_models(serve, tmp_path, user_dir):
    """The fields of /api/projects and /api/models the guide names for
    views, which keep their names within an API version."""
    project = tmp_path / "work"
    (project / ".testuary").mkdir(parents=True)
    (project / ".testuary" / "project.ini").write_text("[project]\nlanguage = python\n", encoding="utf-8")
    (user_dir / "models.ini").write_text(
        "[localfake]\napi_url = http://127.0.0.1:9/v1\n\n[fake]\nprovider = localfake\nmodel = fake-model\n",
        encoding="utf-8")
    _, _, client = serve(project)
    listed = client.json("GET", "/api/projects")[1]["projects"]
    assert [(entry["name"], entry["project_dir"], entry["working"], entry["error"]) for entry in listed] == [
        ("work", str(project.resolve()), True, None)]
    models = client.json("GET", "/api/models")[1]
    assert [(entry["alias"], entry["provider"], entry["model"], entry["usable"])
            for entry in models["models"]] == [("fake", "localfake", "fake-model", True)]
    assert (models["default_model"], models["project_model"]) == (None, None)


def test_stop_ends_an_extension_job_as_it_ends_a_run(serve, tmp_path, monkeypatch):
    _children_import_the_fixture(monkeypatch)
    app, _, client = serve(extensions=_registry.load([NOTES]))
    status, job = client.json("POST", "/api/ext/jobs", {
        "kind": "notes/collect", "dir": str(tmp_path / "slow"), "wait": "60"})
    assert status == 201
    events = app.jobs.get(job["id"]).events
    assert wait_for(lambda: events.is_file() and "Writing notes" in events.read_text(encoding="utf-8"))
    assert client.json("POST", f"/api/jobs/{job['id']}/stop", {})[0] == 200
    assert wait_for(lambda: client.json("GET", f"/api/jobs/{job['id']}")[1]["ended"] is not None)
    assert client.json("GET", f"/api/jobs/{job['id']}")[1]["status"] == "cancelled"


def test_results_links_a_directory_type_with_the_directory_its_view_reads():
    """Results' link to a directory type's view passes the directory as dir,
    which the view reads, as #/dir?type=EXTENSION/TYPE&dir=DIR."""
    assert page_value(["dirTypeAddress"], "dirTypeAddress('notes/notes', 'C:/runs/a b')") == (
        "#/dir?type=notes%2Fnotes&dir=C%3A%2Fruns%2Fa%20b")


def _extension_file(folder, name, body):
    """A file extension ``name`` in ``folder`` whose module runs ``body``."""
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text("from testuary.web import JobKind, WebExtension, WebTab\n" + body,
                               encoding="utf-8")
    return folder / name


def test_testuary_web_loads_the_extensions_its_directorys_settings_name(serve, tmp_path):
    """testuary-web started in a project loads the file its project.ini
    names, once, and serves the tab's script from the file's folder."""
    project = tmp_path / "work"
    _extension_file(project / ".testuary", "filed_ext.py", (
        "testuary_web = WebExtension(api=1, id='filed', tabs=(WebTab('filed', 'Filed', 'filed.js'),))\n"))
    (project / ".testuary" / "filed.js").write_text('registerView("filed", () => null);\n', encoding="utf-8")
    (project / ".testuary" / "project.ini").write_text("[main]\nextensions = filed_ext.py\n", encoding="utf-8")
    _, _, client = serve(project)
    found = client.json("GET", "/api/state")[1]["extensions"]
    assert (found["refused"], [tab["id"] for tab in found["tabs"]]) == ([], ["filed"])
    status, _, body = client.request("GET", "/ext/filed/filed.js", token=False)
    assert (status, body) == (200, (project / ".testuary" / "filed.js").read_bytes())


def test_an_item_that_cannot_load_or_is_no_web_extension_is_listed_and_the_rest_load(tmp_path):
    """An item that fails to import is listed with where it is named, a
    module whose testuary_web is no WebExtension is refused, a module
    without testuary_web adds nothing, and the others load."""
    named_in = str(tmp_path / "project.ini")
    _extension_file(tmp_path, "broken_ext.py", "raise RuntimeError('no such module')\n")
    _extension_file(tmp_path, "function_ext.py", "def testuary_web():\n    return None\n")
    _extension_file(tmp_path, "hints_only_ext.py", "testuary_hints = {}\n")
    _extension_file(tmp_path, "good_ext.py", "testuary_web = WebExtension(api=1, id='good')\n")
    registry = _registry.load([(name, named_in, str(tmp_path)) for name in (
        "broken_ext.py", "function_ext.py", "hints_only_ext.py", "good_ext.py")])
    reasons = dict(registry.refused)
    assert sorted(reasons) == ["broken_ext.py", "function_ext.py"]
    assert reasons["broken_ext.py"].startswith("RuntimeError: no such module.")
    assert reasons["broken_ext.py"].endswith(f"It is named in {named_in}")
    assert reasons["function_ext.py"] == "its testuary_web is not a testuary.web.WebExtension"
    assert [extension.id for extension, _ in registry.loaded] == ["good"]


def test_a_file_extensions_job_imports_its_module_from_the_files_folder(serve, tmp_path):
    """A job kind of a file extension runs its module from the file's
    folder, which is on no import path of the server's."""
    folder = tmp_path / "tools"
    _extension_file(folder, "jobbed_ext.py", (
        "testuary_web = WebExtension(api=1, id='jobbed', job_kinds=(JobKind(\n"
        "    'run', 'Run', 'jobbed_job', lambda form: ['--out', form['out']],\n"
        "    resume=lambda arguments: ['--out', arguments[1] + '.again']),))\n"))
    (folder / "jobbed_job.py").write_text(
        "import sys\nfrom pathlib import Path\n"
        "Path(sys.argv[sys.argv.index('--out') + 1]).write_text('ran', encoding='utf-8')\n",
        encoding="utf-8")
    registry = _registry.load([("jobbed_ext.py", "overrides", str(folder))])
    assert (registry.import_dir("jobbed/run"), registry.import_dir("jobbed")) == (folder, None)
    _, _, client = serve(extensions=registry)
    out = tmp_path / "ran.txt"
    status, job = client.json("POST", "/api/ext/jobs", {"kind": "jobbed/run", "out": str(out)})
    assert status == 201
    assert wait_for(lambda: client.json("GET", f"/api/jobs/{job['id']}")[1]["status"] != "running")
    assert out.read_text(encoding="utf-8") == "ran"
    status, again = client.json("POST", f"/api/jobs/{job['id']}/continue", {})
    assert status == 201
    assert wait_for(lambda: client.json("GET", f"/api/jobs/{again['id']}")[1]["status"] != "running")
    assert (tmp_path / "ran.txt.again").read_text(encoding="utf-8") == "ran"


@pytest.mark.parametrize("error", [RuntimeError("the installed package is missing config.ini"),
                                   ConfigurationError("[coverage] goal: expected 'secret-value'")])
def test_testuary_web_starts_when_its_settings_cannot_be_read(tmp_path, monkeypatch, error):
    """Settings that cannot be read or are refused load no extension, and the
    Overview says so without quoting the error, which can hold a
    configuration value."""
    def unreadable(**_):
        raise error

    monkeypatch.setattr(_registry, "resolve_settings", unreadable)
    registry = _registry.started(tmp_path)
    assert (registry.loaded, registry.refused) == ([], [(
        "[main] extensions", "Testuary cannot read the settings of this directory, so no extension "
        "loaded. The check names the problem")])


HEADER_STUBS = """
let extensions = { product: null, tabs: [{ id: "notes", title: "Notes", position: "after:results" }],
  tab_titles: { results: "Test results" }, tab_positions: {}, hidden_tabs: [], dir_types: [], job_kinds: [],
  refused: [] };
const location = { hash: "#/results?run_dir=r" };
function link(view, text) {
  return { dataset: { view }, textContent: text, current: null,
    setAttribute(name, value) { this.current = value; }, removeAttribute() { this.current = null; } };
}
function el(tag, props) { return link(props["data-view"], props.text); }
const nav = { links: [["", "Overview"], ["jobs", "Jobs"], ["results", "Results"]].map(([view, text]) => link(view, text)),
  replaceChildren(...links) { this.links = links; } };
const document = { title: "", querySelector: (selector) => selector === ".top nav" ? nav : {},
  querySelectorAll: () => nav.links };
"""


def test_a_header_built_again_keeps_each_tab_once_and_marks_the_current_one():
    """A key taken in an open tab builds the header again: Testuary's tabs
    come from the page's own list, read before anything is drawn, so each
    tab shows once, a renamed tab keeps Testuary's title, and the tab of the
    open view stays marked as the current page."""
    tabs, before = page_value(
        ["testuaryTabs", "testuaryTitles", "productName", "headerTabs", "placeTab", "addedTab", "tabShown",
         "addressedView", "markCurrentTab", "buildHeader"],
        "(() => { const before = testuaryTitles.get('results'); buildHeader(); buildHeader(); "
        "return [nav.links.map((link) => [link.dataset.view, link.textContent, link.current]), before]; })()",
        before=HEADER_STUBS)
    assert tabs == [["", "Overview", None], ["jobs", "Jobs", None], ["results", "Test results", "page"],
                    ["notes", "Notes", None]]
    assert before == "Results"


def test_a_token_from_another_tab_or_a_pasted_address_reads_the_extensions_again():
    """A token another tab stores, or one in an address pasted into this
    tab, is taken and the extensions are read again with it. Another stored
    value or another fragment changes nothing."""
    source = APP.read_text(encoding="utf-8")
    listeners = re.search(r"^let tokenStorage = null;\n.*?(?=^function tokenHeaders\()", source, re.S | re.M)
    assert listeners, "app.js takes the token before tokenHeaders"
    stubs = """
const handlers = {};
const window = { localStorage: { getItem: () => null, setItem() {} },
  addEventListener(type, handler) { handlers[type] = handler; } };
const location = { hash: "#/", pathname: "/" };
const history = { replaceState(state, title, address) { location.hash = address.slice(1); } };
let loads = 0;
function loadExtensions() { loads += 1; }
"""
    expression = "(() => {\n" + listeners.group(0) + """
  const seen = [];
  handlers.storage({ key: "other", newValue: "x" });
  seen.push([loads, token]);
  handlers.storage({ key: TOKEN_KEY, newValue: "fromtab" });
  seen.push([loads, token]);
  location.hash = "#/run"; handlers.hashchange();
  seen.push([loads, token]);
  location.hash = "#token=pasted"; handlers.hashchange();
  seen.push([loads, token, location.hash]);
  return seen;
})()"""
    assert page_value(["TOKEN_KEY", "pageToken"], expression, before=stubs) == [
        [0, None], [1, "fromtab"], [1, "fromtab"], [2, "pasted", "#/"]]
