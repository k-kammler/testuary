"""Independent processes keep their own active settings at real SDK boundaries."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import pytest

import test_protocol_loopback_callers as http_harness

loopback = http_harness.loopback

CHILD = r'''
from contextlib import contextmanager
import json
import os
from pathlib import Path
import sys
import time

import pytest

# The test modules and the helpers they share in tests/, as pytest finds them.
sys.path[:0] = [sys.argv[1], str(Path(sys.argv[1]).parent)]
import test_model_precedence_callers as harness
from testuary import config
from testuary import starters as initial
from testuary.settings import RunSettings

phase, endpoint, workspace, number, gate = sys.argv[2:]
root, gate, number = Path(workspace), Path(gate), int(number)
alias = f"alias-{number}"
settings = RunSettings(base_dir=root, environment=dict(os.environ), sections={
    "api": {"model": alias}, "main": {"hints_file": "", "seed": number * 10},
    "project": {"python_executable": sys.executable},
    "starter": {"suite_count": 1, "attempts_per_suite": 1, "concurrent_requests": 1,
                "suite_run_timeout_seconds": 2, "temperature": number / 10,
                "max_output_tokens": number * 32, "request_timeout_seconds": 5},
    "coverage": {"max_requests_per_phase": 1, "max_loop_attempts": 0, "seed": number,
                 "temperature": number / 10, "max_output_tokens": number * 32,
                 "request_timeout_seconds": 5},
    "mutation": {"max_requests_per_phase": 1, "max_loop_attempts": 0, "seed": number,
                 "temperature": number / 10, "max_output_tokens": number * 32,
                 "request_timeout_seconds": 5},
}, models={alias: {
    "provider": "openai", "model": f"model-{number}", "api_url": endpoint,
    "api_key": f"process-credential-{number}", "protocol": "openai-chat",
}})
original_scope = config.use_run_settings

@contextmanager
def overlapping_scope(resolved):
    with original_scope(resolved):
        assert config.active_settings() is settings
        (root / "active.json").write_text(json.dumps({"pid": os.getpid(), "alias": alias}), encoding="utf-8")
        deadline = time.monotonic() + 30
        while not gate.exists():
            if time.monotonic() >= deadline:
                raise TimeoutError("process fixture activation barrier expired")
            time.sleep(.01)
        yield

# Changes after capture must not become configuration overrides.
os.environ["TESTUARY_API_MODEL"] = "ambient-after-capture"
os.environ["TESTUARY_COVERAGE_TEMPERATURE"] = ".9"
os.chdir(root)
usage = {}
with pytest.MonkeyPatch.context() as patch:
    patch.setattr(config, "use_run_settings", overlapping_scope)
    record = harness.invoke_phase(phase, root, settings, patch, usage_snapshot=usage)
assert config.active_settings() is None and config.active_run_scope() is None
(root / "result.json").write_text(json.dumps({"record": record, "usage": usage}), encoding="utf-8")
'''


@pytest.mark.parametrize("phase", ("starter", "coverage", "mutation"))
def test_two_simultaneously_active_processes_send_and_record_only_their_settings(phase, tmp_path, loopback):
    gate = tmp_path / "release"
    children = []
    environments = dict(os.environ)
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "http_proxy", "https_proxy", "all_proxy"):
        environments.pop(name, None)
    environments["NO_PROXY"] = "127.0.0.1,localhost"
    environments["PYTHONDONTWRITEBYTECODE"] = "1"
    try:
        for number in (1, 2):
            workspace = tmp_path / str(number)
            workspace.mkdir()
            child = subprocess.Popen([
                sys.executable, "-c", CHILD, str(Path(__file__).parent), phase,
                loopback.endpoint, str(workspace), str(number), str(gate),
            ], cwd=tmp_path, env=environments, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            children.append(child)
        deadline = time.monotonic() + 40
        while not all((tmp_path / str(number) / "active.json").exists() for number in (1, 2)):
            for child in children:
                if child.poll() is not None:
                    output, errors = child.communicate(timeout=2)
                    pytest.fail(f"process exited before the activation barrier: {child.returncode}\n{output}\n{errors}")
            assert time.monotonic() < deadline, "two-process activation did not finish"
            time.sleep(.01)
        active = [json.loads((tmp_path / str(number) / "active.json").read_text()) for number in (1, 2)]
        # Windows virtualenv launchers may have a different PID from Python.
        assert len({item["pid"] for item in active}) == 2
        assert all(item["pid"] != os.getpid() for item in active)
        assert {item["alias"] for item in active} == {"alias-1", "alias-2"}
        assert all(child.poll() is None for child in children)
        gate.touch()
        for child in children:
            output, errors = child.communicate(timeout=40)
            assert child.returncode == 0, output + errors
        assert len(loopback.requests) == 2
        for number in (1, 2):
            result = json.loads((tmp_path / str(number) / "result.json").read_text())
            settings = result["record"]["sent_settings"]
            assert settings["model_alias"] == f"alias-{number}"
            assert settings["model"] == f"model-{number}"
            assert settings["values"]["temperature"] == number / 10
            assert settings["values"]["max_output_tokens"] == number * 32
            assert result["usage"] == {"physical": 1, "logical": 1, "unknown": 0,
                                       "input_tokens": 10, "output_tokens": 3}
            requests = [entry for entry in loopback.requests if entry[1]["model"] == f"model-{number}"]
            assert len(requests) == 1
            path, body, header = requests[0]
            assert path == "/v1/chat/completions" and header == f"Bearer process-credential-{number}"
            assert body["temperature"] == number / 10 and body["max_completion_tokens"] == number * 32
            assert body["seed"] == (number * 10 + 1 if phase == "starter" else number)
            assert f"process-credential-{number}" not in (tmp_path / str(number) / "result.json").read_text()
    finally:
        gate.touch(exist_ok=True)
        for child in children:
            if child.poll() is None:
                try:
                    child.communicate(timeout=10)
                except subprocess.TimeoutExpired:
                    if os.name == "nt":
                        subprocess.run(["taskkill", "/PID", str(child.pid), "/T", "/F"],
                                       stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5, check=False)
                    else:
                        child.kill()
                    child.communicate(timeout=5)
