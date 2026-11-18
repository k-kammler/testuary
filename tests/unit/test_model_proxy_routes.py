"""Model requests through a SOCKS proxy named by the standard variables.

A cluster job may reach its model provider only through a socks5h tunnel
named by ALL_PROXY, HTTPS_PROXY or HTTP_PROXY (in either case), while
loopback services stay outside it through NO_PROXY. The provider here is a
reserved name no machine resolves (RFC 2606), so a request arrives only when
the client handed the name to the proxy, which relays it to a loopback
provider."""
from __future__ import annotations

from dataclasses import replace
import socket
import threading
from urllib.parse import urlsplit

import httpx
import openai
import pytest

import test_model_precedence_callers as caller_harness
import test_protocol_loopback_callers as loopback_callers
from test_protocol_loopback_callers import loopback  # noqa: F401 (fixture)
from testuary.models import resolve_phase_model
from testuary.models import build_resolved_chat_model
from testuary import starters
from testuary.settings import RunSettings

PROVIDER = "provider.invalid"
TOKEN = loopback_callers.TOKEN
PROXY_VARIABLES = ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY", "NO_PROXY")


def _read(conn: socket.socket, size: int) -> bytes:
    data = b""
    while len(data) < size:
        chunk = conn.recv(size - len(data))
        if not chunk:
            raise OSError("closed")
        data += chunk
    return data


def _pump(source: socket.socket, sink: socket.socket) -> None:
    try:
        while chunk := source.recv(65536):
            sink.sendall(chunk)
    except OSError:
        pass
    finally:
        try:
            sink.shutdown(socket.SHUT_WR)
        except OSError:
            pass


class Socks5:
    """A loopback SOCKS5 server without authentication. It records every
    CONNECT as (address type, host, port), relays those it has a route for
    and refuses the others as unreachable."""

    def __init__(self, routes):
        self.routes = dict(routes)
        self.connects = []
        self._server = socket.socket()
        self._server.bind(("127.0.0.1", 0))
        self._server.listen(16)
        self.url = f"socks5h://127.0.0.1:{self._server.getsockname()[1]}"
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self) -> None:
        self._server.close()

    def _accept(self) -> None:
        while True:
            try:
                conn, _ = self._server.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(conn,), daemon=True).start()

    def _serve(self, conn: socket.socket) -> None:
        with conn:
            try:
                _, methods = _read(conn, 2)
                _read(conn, methods)
                conn.sendall(b"\x05\x00")
                _, command, _, kind = _read(conn, 4)
                if kind == 3:
                    kind, host = "domain", _read(conn, _read(conn, 1)[0]).decode("ascii")
                elif kind == 1:
                    kind, host = "ipv4", socket.inet_ntoa(_read(conn, 4))
                else:
                    kind, host = "ipv6", socket.inet_ntop(socket.AF_INET6, _read(conn, 16))
                port = int.from_bytes(_read(conn, 2), "big")
                self.connects.append((kind, host, port))
                target = self.routes.get((host, port))
                if command != 1 or target is None:
                    conn.sendall(b"\x05\x04\x00\x01" + bytes(6))
                    return
                upstream = socket.create_connection(target, timeout=5)
            except OSError:
                return
            with upstream:
                conn.sendall(b"\x05\x00\x00\x01" + bytes(6))
                back = threading.Thread(target=_pump, args=(upstream, conn), daemon=True)
                back.start()
                _pump(conn, upstream)
                back.join(timeout=5)


@pytest.fixture
def socks():
    servers = []

    def start(routes=()):
        servers.append(Socks5(routes))
        return servers[-1]

    yield start
    for server in servers:
        server.close()


def use_proxy(monkeypatch, variables, url, *, no_proxy="localhost,127.0.0.1,::1"):
    """Set only ``variables`` to ``url`` and NO_PROXY as the cluster job does."""
    for name in PROXY_VARIABLES:
        monkeypatch.delenv(name, raising=False)
        monkeypatch.delenv(name.lower(), raising=False)
    for name in variables:
        monkeypatch.setenv(name, url)
    if no_proxy is not None:
        monkeypatch.setenv("NO_PROXY", no_proxy)


def at(settings: RunSettings, api_url: str) -> RunSettings:
    models = {name: {**values, "api_url": api_url} for name, values in settings.models.items()}
    return replace(settings, models=models)


@pytest.mark.parametrize("phase,protocol", loopback_callers.ROUTES)
def test_every_model_route_sends_its_requests_through_the_socks5h_proxy(
    phase, protocol, tmp_path, monkeypatch, loopback, socks,  # noqa: F811
):
    """Each phase and protocol hands the provider's name to the proxy
    unresolved, and its requests and replies pass the tunnel intact."""
    proxy = socks({(PROVIDER, 80): ("127.0.0.1", urlsplit(loopback.endpoint).port)})
    use_proxy(monkeypatch, ("ALL_PROXY",), proxy.url)
    settings = at(loopback_callers.configured(tmp_path, protocol, loopback), f"http://{PROVIDER}/v1")
    expected = 1 if phase == "starter" else 2
    record = caller_harness.invoke_phase(phase, tmp_path, settings, monkeypatch, expected_requests=expected)
    assert record["sent_settings"]["protocol"] == protocol
    assert len(loopback.requests) == expected
    assert proxy.connects and set(proxy.connects) == {("domain", PROVIDER, 80)}


CLOUD_ROUTES = {
    # A settings overlay: the starter suites use Ollama's native chat, the
    # coverage and mutation phases its OpenAI-compatible endpoint.
    "ollama-cloud": (
        {"api": {"model": "deepseek-v4.1-flash", "provider": "ollama"}},
        {"ollama": {"api_key": TOKEN}},
        "ollama.com", ("ollama-native", "openai-chat", "openai-chat"),
    ),
    # DeepSeek's own API under the alias deepseek_v4_pro.
    "deepseek-api": (
        {"api": {"model": "deepseek_v4_pro"}},
        {"deepseek": {"api_url": "https://api.deepseek.com/v1", "api_key": TOKEN},
         "deepseek_v4_pro": {"provider": "deepseek", "model": "deepseek-v4-pro"}},
        "api.deepseek.com", ("openai-chat", "openai-chat", "openai-chat"),
    ),
}


@pytest.mark.parametrize("route", CLOUD_ROUTES)
@pytest.mark.parametrize("variables", (
    ("ALL_PROXY",), ("all_proxy",), ("HTTPS_PROXY",), ("https_proxy",),
    ("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"),
), ids=("ALL_PROXY", "all_proxy", "HTTPS_PROXY", "https_proxy", "every-spelling"))
def test_the_cluster_routes_hand_their_https_provider_to_the_proxy(
    route, variables, tmp_path, monkeypatch, socks,
):
    """Every phase's client for the cluster's providers asks the proxy for
    the provider's name on port 443, under each spelling of the variables.
    The proxy refuses here, so no request leaves the machine."""
    sections, models, host, protocols = CLOUD_ROUTES[route]
    proxy = socks()
    use_proxy(monkeypatch, variables, proxy.url)
    settings = RunSettings(base_dir=tmp_path, sections=sections, models=models)
    for phase, protocol in zip(("starter", "coverage", "mutation"), protocols):
        model = resolve_phase_model(settings, phase)
        assert model.protocol == protocol
        if protocol == "ollama-native":
            with pytest.raises(httpx.ProxyError):
                starters._default_generate(
                    {"model": model.identity.model, "api_key": TOKEN}, "x",
                    seed=1, model_settings=model)
        else:
            with pytest.raises(openai.APIConnectionError):
                build_resolved_chat_model(model, api_key=TOKEN).invoke("x")
    assert proxy.connects == [("domain", host, 443)] * 3


@pytest.mark.parametrize("variable", ("NO_PROXY", "no_proxy"))
def test_a_loopback_provider_named_in_no_proxy_bypasses_the_proxy(
    variable, tmp_path, monkeypatch, loopback, socks,  # noqa: F811
):
    """With every proxy variable set, NO_PROXY's loopback names keep a
    loopback endpoint off the proxy."""
    proxy = socks()
    use_proxy(monkeypatch, ("ALL_PROXY", "HTTP_PROXY", "HTTPS_PROXY"), proxy.url, no_proxy=None)
    monkeypatch.setenv(variable, "localhost,127.0.0.1,::1")
    settings = loopback_callers.configured(tmp_path, "openai-chat", loopback)
    caller_harness.invoke_phase("starter", tmp_path, settings, monkeypatch)
    assert len(loopback.requests) == 1
    assert proxy.connects == []
