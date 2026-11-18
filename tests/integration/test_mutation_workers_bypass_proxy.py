"""Cosmic Ray's HTTP workers on loopback stay off a configured proxy.

A cluster job sets every proxy variable for its model requests. The
coordinator of the ``http`` distributor and its readiness checks talk to the
workers on 127.0.0.1, and must reach them directly even when NO_PROXY is
missing. The proxy here is a loopback listener that counts connections and
refuses each, answering the SOCKS5 greeting with "no acceptable method".

Marked ``slow``, like the other real Cosmic Ray runs.
"""
from __future__ import annotations

import socket
import threading
from pathlib import Path

import httpx
import pytest

from test_cosmic_ray_single_worker import _run
from tool_fixtures import cosmic_ray_installed

from testuary import languages

pytestmark = [pytest.mark.slow, pytest.mark.mutation_tool]

PROXY_VARIABLES = ("ALL_PROXY", "HTTPS_PROXY", "HTTP_PROXY")


def test_http_workers_measure_with_every_proxy_variable_set_and_no_no_proxy(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    in_run,
) -> None:
    if not cosmic_ray_installed():
        pytest.skip("cosmic-ray not installed; integration test requires it")
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(16)
    connections = []

    def accept() -> None:
        while True:
            try:
                conn, _ = listener.accept()
            except OSError:
                return
            connections.append(conn)
            # A refusal the client always reads the same way: a bare close
            # races a reset against an empty reply.
            conn.settimeout(5)
            try:
                conn.recv(256)
                conn.sendall(b"\x05\xff")
            except OSError:
                pass
            conn.close()

    threading.Thread(target=accept, daemon=True).start()
    proxy = f"socks5h://127.0.0.1:{listener.getsockname()[1]}"
    for name in PROXY_VARIABLES:
        monkeypatch.setenv(name, proxy)
        monkeypatch.setenv(name.lower(), proxy)
    monkeypatch.delenv("NO_PROXY", raising=False)
    monkeypatch.delenv("no_proxy", raising=False)
    # Control: a client that honors the variables sends even a loopback
    # request to the proxy here.
    with pytest.raises(httpx.TransportError):
        httpx.get("http://127.0.0.1:9/", timeout=5)
    assert len(connections) == 1
    connections.clear()
    monkeypatch.setenv("PYTHONIOENCODING", "utf-8")
    monkeypatch.setenv("PYTHONUTF8", "1")
    # The run captures the environment with the proxy variables, as a
    # cluster job's run does; its commands start with that environment.
    in_run(overrides={"api": {"provider": "ollama", "model": "test-model"}})

    work = tmp_path / "work"
    proj, test_dir = work / "proj", work / "tests"
    proj.mkdir(parents=True)
    test_dir.mkdir(parents=True)
    (proj / "arith.py").write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    (test_dir / "test_arith.py").write_text(
        "from arith import add\n\n\n"
        "def test_add():\n    assert add(2, 3) == 5\n    assert add(0, 0) == 0\n",
        encoding="utf-8",
    )
    languages.get_backend("python").setup_test_env(
        test_output_dir=str(test_dir), source_dir=str(proj),
        target_file_path=str(proj / "arith.py"),
    )
    log_messages: list[str] = []
    try:
        result = _run(
            {"cosmic_ray_operators": "wong_mathur_selective", "cosmic_ray_distributor": "http"},
            proj, proj / "arith.py", test_dir, 2,
            log=lambda msg, level=0: log_messages.append(msg))
    finally:
        listener.close()
    assert result.killed > 0 and result.score >= 70.0, (
        f"score={result.score:.1f} (killed={result.killed}, survived={result.survived})\n"
        + "\n".join(log_messages))
    assert connections == []
