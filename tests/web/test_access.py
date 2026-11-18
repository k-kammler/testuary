"""Who may use the interface: without an access key by
default, a per-start key or one kept in the user directory on request, from
--access or [web] access; the port from --port or [web] port. With every
access setting it listens on 127.0.0.1 only, answers only requests naming it
in their Host header and takes a change only from its own Origin."""
from __future__ import annotations

import os
import socket
import sys

import pytest

import testuary.web as web
from conftest import Client


@pytest.fixture(autouse=True)
def no_machine_settings(monkeypatch):
    """Testuary's settings as the user directory's files give them: the test
    session's own variables (TESTUARY_TIMING_RUN_COUNT) left out."""
    for name in list(os.environ):
        if name.startswith("TESTUARY_") and name != "TESTUARY_HOME":
            monkeypatch.delenv(name)


def test_without_a_key_the_page_and_the_interface_answer(serve):
    app, server, client = serve(key=False)
    assert app.token is None and server.server_address[0] == "127.0.0.1"
    stranger = Client(server)
    assert stranger.request("GET", "/")[0] == 200
    assert stranger.request("GET", "/api/state")[0] == 200
    # A change passes the guard and reaches the validation, which refuses cobol.
    assert stranger.request("POST", "/api/demo", {"language": "cobol"})[0] == 400
    own = {"Origin": f"http://127.0.0.1:{client.port}"}
    assert stranger.request("POST", "/api/demo", {"language": "cobol"}, headers=own)[0] == 400


@pytest.mark.parametrize("key", [False, True])
def test_a_foreign_host_is_refused_in_every_mode(serve, key):
    app, server, client = serve(key=key)
    for host in ("evil.example", f"127.0.0.1.evil.example:{client.port}", "127.0.0.1"):
        assert client.request("GET", "/api/state", host=host)[0] == 403
        assert client.request("POST", "/api/demo", {"language": "python"}, host=host)[0] == 403
        assert client.request("GET", "/", host=host)[0] == 403
    assert client.request("GET", "/api/state", host=f"localhost:{client.port}")[0] == 200
    assert app.jobs.active() is None


@pytest.mark.parametrize("key", [False, True])
def test_a_change_from_another_origin_is_refused_in_every_mode(serve, key):
    app, server, client = serve(key=key)
    for origin in ("http://evil.example", "null", f"http://127.0.0.1:{client.port + 1}",
                   f"https://127.0.0.1:{client.port}"):
        assert client.request("POST", "/api/demo", {"language": "python"},
                              headers={"Origin": origin})[0] == 403, origin
    assert app.jobs.active() is None


def test_with_a_key_only_its_page_gets_in(serve):
    app, server, client = serve(key=True)
    stranger = Client(server)
    assert stranger.request("GET", "/api/state")[0] == 403
    assert stranger.request("POST", "/api/demo", {"language": "python"})[0] == 403
    assert client.request("GET", "/api/state")[0] == 200


def test_the_access_modes_and_their_keys(user_dir):
    assert web._access_key("open") is None
    first, second = web._access_key("key"), web._access_key("key")
    assert first != second and len(first) >= 32
    assert not (user_dir / "web" / "access-key").exists()
    kept = web._access_key("stable_key")
    assert web._access_key("stable_key") == kept
    stored = user_dir / "web" / "access-key"
    assert stored.read_text(encoding="ascii").strip() == kept
    if os.name == "posix":
        assert stored.stat().st_mode & 0o077 == 0
    # A damaged file gives way to a new key.
    stored.write_text("not a key!\n", encoding="ascii")
    renewed = web._access_key("stable_key")
    assert renewed != kept and web._access_key("stable_key") == renewed


def test_the_printed_address_holds_a_key_only_when_there_is_one():
    assert web._address(8765, None) == "http://127.0.0.1:8765/"
    assert web._address(8765, "abc_-1") == "http://127.0.0.1:8765/#token=abc_-1"


def test_port_and_access_come_from_the_settings_and_the_options_win(tmp_path, user_dir):
    base_dir = tmp_path / "work"
    base_dir.mkdir()
    assert web._web_settings(base_dir) == (8765, "open")
    (user_dir / "config.ini").write_text("[web]\nport = 9000\naccess = key\n", encoding="utf-8")
    assert web._web_settings(base_dir) == (9000, "key")
    assert web._chosen(base_dir, port=0, access=None) == (0, "key")
    assert web._chosen(base_dir, port=None, access="open") == (9000, "open")


def test_an_access_the_settings_do_not_name_is_refused(tmp_path, user_dir):
    """serve(access=...) takes only the choices of [web] access: another value
    is refused, not taken as a kept key."""
    base_dir = tmp_path / "work"
    base_dir.mkdir()
    assert web.serve(base_dir, port=0, access="everyone", open_browser=False) == 2
    assert not (user_dir / "web" / "access-key").exists()


def test_settings_that_cannot_be_read_need_both_options(tmp_path, user_dir):
    base_dir = tmp_path / "work"
    base_dir.mkdir()
    (user_dir / "config.ini").write_text("[web]\nport = many\n", encoding="utf-8")
    with pytest.raises(web._Unavailable, match="--port and --access"):
        web._chosen(base_dir, port=None, access="key")
    assert web._chosen(base_dir, port=8000, access="key") == (8000, "key")


def test_a_taken_port_ends_the_start_with_a_clear_message(tmp_path):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as holder:
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        port = holder.getsockname()[1]
        app = web._server.App(tmp_path, testuary_version="test", python=sys.executable)
        with pytest.raises(web._Unavailable) as refused:
            web._listen(app, port)
    message = str(refused.value)
    assert f"port {port}" in message and "taken" in message and "--port" in message


@pytest.mark.parametrize("arguments", [["--port", "70000"], ["--port", "-1"], ["--access", "public"]])
def test_the_options_refuse_what_the_settings_refuse(arguments, capsys):
    with pytest.raises(SystemExit) as ended:
        web.main(arguments)
    assert ended.value.code == 2
