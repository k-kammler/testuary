"""A loopback stand-in for an OpenAI-compatible model service: a starter
suite request gets a pytest file, and an agent request reads the tests,
measures and gives its final answer. No request leaves the machine."""
from __future__ import annotations

import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

SEED = '''```python
from calc.ops import add, sign


def test_add():
    assert add(2, 3) == 5


def test_sign():
    assert sign(4) == 1
    assert sign(-4) == -1
    assert sign(0) == 0
```'''

#: The key the service takes when one is sent.
KEY = "loopback-test-key"

#: The project the model's answers fit.
OPS = ("def add(a, b):\n    return a + b\n\n\ndef sign(x):\n    if x > 0:\n"
       "        return 1\n    if x < 0:\n        return -1\n    return 0\n")


class FakeModel(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def do_GET(self):
        # The models it serves, as an OpenAI-compatible service lists them.
        # With a key, only the right one.
        given = self.headers.get("Authorization", "")
        if given and given != f"Bearer {KEY}":
            self.send_response(401)
            self.end_headers()
            return
        body = json.dumps({"data": [{"id": "fake-model"}]}).encode("utf-8") if self.path == "/v1/models" else b""
        self.send_response(200 if body else 404)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self):
        request = json.loads(self.rfile.read(int(self.headers["Content-Length"])) or b"{}")
        names = [tool["function"]["name"] for tool in request.get("tools") or []]
        if names:
            done = sum(1 for m in request.get("messages") or [] if m.get("role") == "tool")
            plan = [n for n in ("read_tests",
                                next((n for n in names if n.startswith("measure_")), None),
                                next((n for n in names if n.startswith("emit_")), None)) if n]
            name = plan[min(done, len(plan) - 1)]
            args = ({"test_file_name": "test_ops.py", "explanation": "x",
                     "targeted_functions": ["add", "sign"]}
                    if name == "emit_coverage_test_result" else
                    {"mutation_test_file_name": "test_ops.py", "explanation": "x",
                     "targeted_mutants": []}
                    if name == "emit_mutation_test_result" else {})
            call = {"id": f"call-{done}", "type": "function",
                    "function": {"name": name, "arguments": json.dumps(args)}}
            message, finish = {"role": "assistant", "content": "", "tool_calls": [call]}, "tool_calls"
        else:
            message, finish = {"role": "assistant", "content": SEED}, "stop"
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        if request.get("stream"):
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            delta = dict(message)
            if "tool_calls" in delta:
                delta["tool_calls"] = [{"index": 0, **delta["tool_calls"][0]}]
            for chunk in ({"choices": [{"index": 0, "delta": delta, "finish_reason": None}]},
                          {"choices": [{"index": 0, "delta": {}, "finish_reason": finish}],
                           "usage": usage}):
                payload = {"id": "f", "object": "chat.completion.chunk",
                           "created": int(time.time()), "model": "fake-model", **chunk}
                self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
            self.wfile.write(b"data: [DONE]\n\n")
            return
        body = json.dumps({"id": "f", "object": "chat.completion", "created": int(time.time()),
                           "model": "fake-model", "usage": usage,
                           "choices": [{"index": 0, "message": message,
                                        "finish_reason": finish}]}).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)


def start() -> ThreadingHTTPServer:
    """The fake model service on a free port of 127.0.0.1, already serving."""
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeModel)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server
