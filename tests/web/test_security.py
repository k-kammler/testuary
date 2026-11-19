"""Only this interface's page can use it: 127.0.0.1, the access token in a
header, the Host and Origin checks. Every response carries the security
headers, and no response holds the token."""
from __future__ import annotations

import json
import re
import shutil
import socket
import subprocess

import pytest

from conftest import SOURCE, Client

APP = SOURCE / "testuary" / "web" / "static" / "app.js"


def _raw(port, request: bytes):
    """Send ``request`` as it is and read until the server closes: the
    status, the headers and the body of what came back."""
    with socket.create_connection(("127.0.0.1", port), timeout=30) as connection:
        connection.sendall(request)
        data = b""
        while True:
            chunk = connection.recv(65536)
            if not chunk:
                break
            data += chunk
    head, _, body = data.partition(b"\r\n\r\n")
    lines = head.decode("latin-1").split("\r\n")
    headers = {name.lower(): value.strip() for name, value in
               (line.split(":", 1) for line in lines[1:] if ":" in line)}
    return int(lines[0].split()[1]), headers, body, data


def _post(port, headers: str, body: bytes = b"") -> bytes:
    return (f"POST /api/demo HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n{headers}\r\n"
            .encode("latin-1") + body)


def test_it_listens_on_127_0_0_1_only(serve):
    app, server, client = serve()
    assert server.server_address[0] == "127.0.0.1"


def test_the_page_needs_no_token_and_the_interface_does(serve):
    app, server, client = serve()
    stranger = Client(server)
    assert stranger.request("GET", "/")[0] == 200
    assert stranger.request("GET", "/static/app.js")[0] == 200
    assert stranger.request("GET", "/api/state")[0] == 403
    for wrong in ("wrong", app.token[:-1], app.token + "x", "t\xf6ken"):
        assert stranger.request("GET", "/api/state",
                                headers={"X-Testuary-Web-Token": wrong})[0] == 403
    assert stranger.request("GET", "/api/state",
                            headers={"X-Testuary-Web-Token": "töken".encode("utf-8")})[0] == 403
    assert client.json("GET", "/api/state")[0] == 200


def _page_tokens(steps):
    """pageToken as the page runs it, evaluated by Node: each step is a
    tab's address fragment and which storage it sees ("shared", "none" or
    "broken"). The result is each tab's token."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    source = APP.read_text(encoding="utf-8")
    function = re.search(r"^function pageToken\(hash, storage\) \{\n.*?^\}\n", source, re.S | re.M)
    assert function, "app.js defines pageToken"
    script = ('const TOKEN_KEY = "testuary-web-token";\n' + function.group(0) + """
const kept = new Map();
const storages = {
  shared: { setItem: (key, value) => kept.set(key, value), getItem: (key) => kept.has(key) ? kept.get(key) : null },
  none: null,
  broken: { setItem: () => { throw new Error("denied"); }, getItem: () => { throw new Error("denied"); } },
};
const steps = """ + json.dumps(steps) + """;
process.stdout.write(JSON.stringify(steps.map(([hash, storage]) => pageToken(hash, storages[storage]))));""")
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    return json.loads(done.stdout)


def test_every_tab_of_the_running_interface_has_the_token():
    """The printed address's token is kept for the server's origin: a new
    tab, a copied address or a bookmark of the same running interface has
    it. A later start's address replaces it."""
    assert _page_tokens([["#token=first", "shared"], ["#/run", "shared"], ["", "shared"],
                         ["#token=second", "shared"], ["#/jobs", "shared"]]) == [
        "first", "first", "first", "second", "second"]
    # Without the token, or without storage, a tab has none and says so.
    assert _page_tokens([["#/run", "shared"], ["#/run", "none"], ["#/run", "broken"],
                         ["#token=only", "broken"], ["#token=bad!", "shared"]]) == [
        None, None, None, "only", None]
    source = APP.read_text(encoding="utf-8")
    assert "pageToken(location.hash, tokenStorage)" in source and "window.localStorage" in source
    assert "sessionStorage" not in source


def test_a_new_address_in_an_open_tab_gives_its_token():
    """After a restart on the same port, a new printed address pasted into
    an open tab changes only the fragment: the page takes its token on the
    fragment's change, before the view is drawn."""
    source = APP.read_text(encoding="utf-8")
    taken = source.find('window.addEventListener("hashchange", () => {\n  if (!/^#token=/.test(location.hash)) return;')
    drawn = source.find('window.addEventListener("hashchange", render);')
    assert 0 <= taken < drawn
    assert "token = pageToken(location.hash, tokenStorage);" in source[taken:drawn]


def test_a_page_without_access_says_only_why(serve):
    """A view is drawn only after the page has access. Otherwise the page
    shows the server's one reason, once."""
    source = APP.read_text(encoding="utf-8")
    start = source.index("async function render() {")
    body = source[start:source.index("\n}\n", start)]
    assert body.index('await call("/api/state");') < body.index("await demoView()")
    app, server, client = serve(key=True)
    status, refused = Client(server).json("GET", "/api/state")
    assert status == 403 and refused["error"] == "Open the address testuary-web printed when it started."


def test_a_page_without_a_key_sends_none():
    """tokenHeaders as the page runs it, evaluated by Node: the key's header
    with a key, no header without one (an open interface)."""
    node = shutil.which("node")
    if node is None:
        pytest.skip("needs Node.js to run the page's script")
    source = APP.read_text(encoding="utf-8")
    function = re.search(r"^function tokenHeaders\(\) \{\n.*?^\}\n", source, re.S | re.M)
    assert function, "app.js defines tokenHeaders"
    script = ('const TOKEN_HEADER = "X-Testuary-Web-Token";\nlet token = null;\n' + function.group(0)
              + "const open = tokenHeaders(); token = \"abc\";\n"
              + "process.stdout.write(JSON.stringify([open, tokenHeaders()]));")
    done = subprocess.run([node, "-e", script], capture_output=True, text=True, timeout=60)
    assert done.returncode == 0, done.stderr
    assert json.loads(done.stdout) == [{}, {"X-Testuary-Web-Token": "abc"}]


def test_the_page_names_its_own_icon(serve):
    """The page names its own icon, which the server gives as it gives the
    page, so the browser asks for no /favicon.ico, which needs the token."""
    app, server, client = serve()
    stranger = Client(server)
    status, _, page = stranger.request("GET", "/")
    assert status == 200 and b'<link rel="icon" href="/static/icon.svg"' in page
    status, headers, icon = stranger.request("GET", "/static/icon.svg")
    assert status == 200 and headers["content-type"] == "image/svg+xml"
    assert icon.startswith(b"<svg") and icon.isascii()


def test_cookies_play_no_part(serve):
    app, server, client = serve()
    # The old address and a cookie with the token open nothing.
    status, headers, _ = Client(server).request("GET", f"/?token={app.token}")
    assert status == 200 and "set-cookie" not in headers
    stranger = Client(server)
    for name in ("testuary_web", f"testuary_web_{client.port}"):
        assert stranger.request("GET", "/api/state",
                                headers={"Cookie": f"{name}={app.token}"})[0] == 403
    # Another server's cookies on 127.0.0.1 do not get in the way.
    foreign = {"Cookie": 'session={"a": 1}; bad cookie=x y; other=1'}
    assert client.request("GET", "/api/state", headers=foreign)[0] == 200


def test_a_foreign_host_is_refused(serve):
    app, server, client = serve()
    for host in ("evil.example", f"127.0.0.1.evil.example:{client.port}", "127.0.0.1"):
        assert client.request("GET", "/api/state", host=host)[0] == 403
        assert client.request("POST", "/api/demo", {"language": "python"}, host=host)[0] == 403
        assert client.request("GET", "/", host=host)[0] == 403
    assert client.request("GET", "/api/state", host=f"localhost:{client.port}")[0] == 200
    assert app.jobs.active() is None


def test_a_change_needs_the_token_and_this_origin(serve):
    app, server, client = serve()
    # Passing the guard reaches the validation: an unknown language is refused there.
    assert client.request("POST", "/api/demo", {"language": "cobol"})[0] == 400
    assert client.request("POST", "/api/demo", {"language": "cobol"}, token=False)[0] == 403
    for origin in ("http://evil.example", "null", f"http://127.0.0.1:{client.port + 1}",
                   f"https://127.0.0.1:{client.port}"):
        assert client.request("POST", "/api/demo", {"language": "cobol"},
                              headers={"Origin": origin})[0] == 403, origin
    own = {"Origin": f"http://127.0.0.1:{client.port}"}
    assert client.request("POST", "/api/demo", {"language": "cobol"}, headers=own)[0] == 400
    stranger = Client(server)
    assert stranger.request("POST", "/api/demo", {"language": "python"})[0] == 403
    assert app.jobs.active() is None


def test_no_preflight_succeeds(serve):
    app, server, client = serve()
    status, headers, _ = client.request("OPTIONS", "/api/demo", headers={
        "Origin": "http://evil.example", "Access-Control-Request-Method": "POST",
        "Access-Control-Request-Headers": "x-testuary-web-token"})
    assert status == 405 and not [name for name in headers if name.startswith("access-control")]
    assert client.request("PUT", "/api/demo", {"language": "python"})[0] == 501
    assert app.jobs.active() is None


def test_request_lengths_are_checked(serve):
    app, server, client = serve()
    port = client.port
    token = f"X-Testuary-Web-Token: {app.token}\r\nContent-Type: application/json\r\n"
    for length, status in (("-1", 400), ("abc", 400), ("1e3", 400), ("70000", 413)):
        assert _raw(port, _post(port, token + f"Content-Length: {length}\r\n"))[0] == status
    assert _raw(port, _post(port, token))[0] == 411
    assert _raw(port, _post(port, token + "Content-Length: 22\r\n",
                            b'{"language": "cobol"} '))[0] == 400


def test_a_refused_request_ends_its_connection(serve):
    app, server, client = serve()
    port = client.port
    body = b'{"language": "python"}'
    refused = _post(port, f"Content-Type: application/json\r\nContent-Length: {len(body)}\r\n",
                    body)
    # The server reads the small body, answers and closes: the read ends.
    status, headers, _, data = _raw(port, refused)
    assert status == 403 and headers["connection"] == "close"
    assert data.count(b"HTTP/1.1 ") == 1
    assert app.jobs.active() is None


def test_no_response_holds_the_token(serve):
    app, server, client = serve()
    stranger = Client(server)
    answers = [client.request("GET", path) for path in (
        "/", f"/?token={app.token}", "/static/app.js", "/static/style.css", "/api/state",
        "/api/nothing", "/api/jobs/unknown", "/api/results?run_dir=nowhere")]
    answers += [stranger.request("GET", "/api/state"),
                client.request("OPTIONS", "/api/state"),
                client.request("POST", "/api/demo", {"language": "cobol"}),
                stranger.request("POST", "/api/demo", {"language": "python"})]
    for status, headers, body in answers:
        assert app.token.encode("ascii") not in body
        assert not [value for value in headers.values() if app.token in value]


def test_responses_carry_the_security_headers(serve):
    app, server, client = serve()
    answers = [client.request("GET", path)
               for path in ("/", "/static/app.js", "/api/state", "/api/nothing")]
    answers += [client.request("OPTIONS", "/api/state"),
                client.request("PUT", "/api/demo", {"language": "python"})]
    for status, headers, _ in answers:
        assert "default-src 'none'" in headers["content-security-policy"], status
        assert headers["x-content-type-options"] == "nosniff"
        assert headers["referrer-policy"] == "no-referrer"
        assert headers["cache-control"] == "no-store"


def test_only_the_packaged_pages_are_served(serve):
    app, server, client = serve()
    assert client.request("GET", "/static/app.js")[0] == 200
    for path in ("/static/../_server.py", "/static/_server.py", "/static/", "/../pyproject.toml"):
        assert client.request("GET", path)[0] == 404, path


def test_bad_requests_are_refused_plainly(serve):
    app, server, client = serve()
    status, _, _ = client.request("POST", "/api/demo", headers={"Content-Type": "text/plain"},
                                  body={"language": "python"})
    assert status == 415
    assert client.json("POST", "/api/demo", {"language": "python", "pace": "slowly"}) == (
        400, {"error": "The pace is instant or paced.", "field": "pace"})
    assert client.json("POST", "/api/check", {"project": "../etc"})[0] == 400
    assert client.json("GET", "/api/jobs/unknown-job")[0] == 404
