"""The pages' texts: what each option does, the prose rules, and the
demonstration's configured playback length, read from Testuary's settings."""
from __future__ import annotations

import os

import pytest


@pytest.fixture(autouse=True)
def no_machine_settings(monkeypatch):
    for name in list(os.environ):
        if name.startswith("TESTUARY_"):
            monkeypatch.delenv(name)


def test_the_demonstration_length_is_the_users_setting(serve, user_dir, monkeypatch):
    app, server, client = serve()
    assert client.json("GET", "/api/demo/settings") == (200, {"playback_seconds": "120"})
    (user_dir / "config.ini").write_bytes(b"[demo]\nplayback_seconds = 45\n")
    assert client.json("GET", "/api/demo/settings")[1] == {"playback_seconds": "45"}
    monkeypatch.setenv("TESTUARY_DEMO_PLAYBACK_SECONDS", "30")
    assert client.json("GET", "/api/demo/settings")[1] == {"playback_seconds": "30"}
    (user_dir / "config.ini").write_bytes(b"[no_such_section]\nkey = 1\n")
    assert client.json("GET", "/api/demo/settings")[1] == {"playback_seconds": None}


def test_the_pages_say_what_their_options_do():
    """Each text says what its option or figure does: the step mode needs
    Enter, the length is a minimum, normal speed is not step mode, the check's
    tools are the selected ones, a run per file generates suites, a
    duration reads as the console's clock, Zero-Shot as the tables write it,
    and Testuary's version once."""
    from conftest import SOURCE, page_value

    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    for wrong in ("one step per key press", "plays over this many seconds", "step by step over",
                  "zero-shot", "Each file is generated and measured", "This interface: testuary-web",
                  "Also run the project's tools once", "(minutes:seconds)`"):
        assert wrong not in source, wrong
    assert "Step mode, one step per Enter press, needs a terminal: testuary --demo --demo-step" in source
    assert "The playback takes at least this many seconds." in source
    assert page_value(["hms"], "[hms(59), hms(3725), hms(null)]") == [
        "00:00:59", "01:02:05", "00:00:00"]


def test_an_empty_run_directory_takes_the_projects_setting():
    """The run form's empty run directory passes no --output-dir, so the
    project's [project] output_dir applies, and its hint says so."""
    from conftest import SOURCE

    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert "Empty: a directory for the model in the results directory" not in source
    assert "Empty: the project's [project] output_dir" in source


def test_the_page_text_follows_the_prose_rules():
    """No semicolon and no ellipsis in the page's text (the path shortener's
    ".../" aside), no rhetorical question, and only ASCII: no decorative
    symbol in the script, the style sheet or the page."""
    import re

    from conftest import SOURCE

    static = SOURCE / "testuary" / "web" / "static"
    source = (static / "app.js").read_text(encoding="utf-8")
    code = "\n".join(line for line in source.splitlines() if not line.lstrip().startswith("//"))
    literals = re.findall(r'"(?:[^"\\\n]|\\.)*"|`(?:[^`\\]|\\.)*`', code)
    assert [text for text in literals if "; " in text or text.rstrip('"`').endswith(";")] == []
    assert [text for text in literals if "..." in text.replace(".../", "")] == []
    assert "New to Testuary?" not in source
    for name in ("app.js", "style.css", "index.html"):
        assert (static / name).read_text(encoding="utf-8").isascii(), name
    assert not re.search(r'content:\s*"\\', (static / "style.css").read_text(encoding="utf-8"))


def test_the_demonstration_speeds_say_plainly_what_they_do():
    """The two speeds say what they do, without the option's factor
    ("speed 1") or the terminal. Instant still plays every step, about ten
    seconds, so it does not promise to show the run at once."""
    from conftest import SOURCE

    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    for wrong in ("speed 1", "as at a terminal", "Paced", "paced playback", "Instant: shows it at once"):
        assert wrong not in source, wrong
    assert '["paced", "Normal speed: plays the recorded run over the length below"]' in source
    assert '["instant", "Instant: plays without pauses"]' in source


def test_stop_says_what_it_stops_and_the_settings_rows_line_up():
    """A demonstration's Stop names the playback, and on the Settings page a
    row with Reset keeps its value in line with the other rows."""
    import re

    from conftest import SOURCE, page_value

    static = SOURCE / "testuary" / "web" / "static"
    hints = page_value(["STOP_HINTS"], "STOP_HINTS")
    assert hints["demo"] == "Stops the playback." and hints["check"].startswith("Stops the check.")
    assert "STOP_HINTS[current.kind]" in (static / "app.js").read_text(encoding="utf-8")
    rule = re.search(r"\.settings-group \.setting \{\s*display: grid; grid-template-columns: ([^;]+);",
                     (static / "style.css").read_text(encoding="utf-8"))
    assert rule and not rule.group(1).rstrip().endswith("auto"), rule


def test_wide_content_scrolls_in_its_box_at_phone_width():
    """A grid column that takes the free width can also shrink below its
    content: a wide table then scrolls inside its box, and the page stays as
    wide as a phone's screen. A bare 1fr column grows to its content's
    width, as the overview's latest-results table made the overview 686
    pixels wide at 390."""
    import re

    from conftest import SOURCE

    style = (SOURCE / "testuary" / "web" / "static" / "style.css").read_text(encoding="utf-8")
    columns = re.findall(r"grid-template-columns:\s*([^;]+);", style)
    assert columns
    # A share of the free width outside minmax() and repeat() is a bare one.
    bare = [value for value in columns if re.search(r"[\d.]+fr", re.sub(r"\([^()]*\)", "()", value))]
    assert not bare, bare
    # A long path in a line of text, such as the project form's settings
    # file, breaks where it must.
    code = re.search(r"^code \{([^}]*)\}", style, re.M)
    assert code and "overflow-wrap: anywhere" in code.group(1)


def test_a_stopped_demonstration_or_check_says_so_in_the_pages_words():
    """A stopped demonstration or check says what the stop meant on the page,
    as a stopped run does, not the terminal's Ctrl+C sentence."""
    from conftest import SOURCE, page_value

    meanings = page_value(["STOPPED_MEANINGS"], "STOPPED_MEANINGS")
    assert set(meanings) == {"demo", "check"}
    assert not [text for text in meanings.values() if "Ctrl+C" in text or "--" in text]
    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    assert 'current.status === "cancelled" && STOPPED_MEANINGS[current.kind]' in source


def test_a_refusal_names_its_field(serve):
    app, server, client = serve()
    assert client.json("POST", "/api/demo", {"language": "cobol"})[1]["field"] == "language"
    assert client.json("POST", "/api/demo", {"pace": "slowly"})[1]["field"] == "pace"
    assert client.json("POST", "/api/demo", {"pace": "paced", "seconds": 45})[1]["field"] == "seconds"
    assert client.json("POST", "/api/check", {"project_dir": "../etc"})[1]["field"] == "project_dir"
    assert client.json("POST", "/api/run", {"files": ["a.py"]})[1]["field"] == "project_dir"
    assert app.jobs.active() is None


def test_unreadable_settings_leave_the_default_unknown(serve, user_dir):
    # Settings Testuary refuses: the length is not known.
    (user_dir / "config.ini").write_bytes(b"[no_such_section]\nkey = 1\n")
    app, server, client = serve()
    assert client.json("GET", "/api/demo/settings") == (200, {"playback_seconds": None})


def test_every_page_links_this_versions_guide(serve):
    """The header links the web interface's guide of the installed version,
    as the setup links the settings reference of its version."""
    from conftest import SOURCE
    from testuary import __version__

    app, server, client = serve()
    status, state = client.json("GET", "/api/state")
    assert state["guide"] == f"https://github.com/k-kammler/testuary/blob/v{__version__}/docs/guide/web.md"
    page = (SOURCE / "testuary" / "web" / "static" / "index.html").read_text(encoding="utf-8")
    assert 'id="guide"' in page and 'rel="noopener noreferrer"' in page


def test_the_check_button_says_what_it_checks():
    """The Check page's button says it runs the check and sends nothing to
    a model. The projects' link to it says what it opens."""
    import re

    from conftest import SOURCE

    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    start = source.index("async function checkView(query) {")
    view = source[start:source.index("\n}\n", start)]
    label = re.search(r'jobForm\(form, "([^"]+)"', view).group(1)
    assert label == "Run the check (sends nothing to a model)"
    assert 'text: "Check the setup" }' in source and 'text: "Check it" }' not in source


def test_a_projects_edit_form_links_its_full_settings():
    """A project's Edit form, which sets its language, files and model,
    links the Settings page for that project, where coverage, mutation and
    every other setting are."""
    from conftest import SOURCE

    source = (SOURCE / "testuary" / "web" / "static" / "app.js").read_text(encoding="utf-8")
    start = source.index("function projectForm(")
    form = source[start:source.index("\n}\n", start)]
    assert 'el("a", { href: settingsLink(entry.project_dir), text: `All settings of ${entry.name}` })' in form
