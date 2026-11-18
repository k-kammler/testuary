"""The process test_coverage_stop.py stops: the coverage agent runs one of
its tools on a suite whose test takes 10 s, and the interrupt ends the
process as the command line does. ``RUNNING`` on stdout marks the tool in
progress, ``CANCELLING`` the first stop reaching it.

Usage: python _coverage_stop_child.py DIR measure_coverage|run_tests|finish
``finish`` runs the agent again in the same directories until its limit ends it.
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

SOURCE = "def add(a, b):\n    return a + b\n\n\ndef sub(a, b):\n    return a - b\n"
SLOW = ("import time\nfrom calc import add\n\n\n"
        "def test_add():\n    time.sleep(10)\n    assert add(1, 2) == 3\n")
FAST = "from calc import add\n\n\ndef test_add():\n    assert add(1, 2) == 3\n"


def main(base_dir: Path, tool: str, patch=setattr) -> int:
    """``patch`` sets each replaced attribute (a test passes its monkeypatch's)."""
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    from testuary import __main__ as cli
    from testuary import exits
    from testuary import config
    from testuary.languages import mutation as mutation_runs
    from testuary.agent import coverage_agent as tc
    from testuary.settings import RunSettings

    project = base_dir / "subject"
    output = base_dir / "tests"
    if tool != "finish":
        project.mkdir()
        (project / "pyproject.toml").write_text("", encoding="utf-8")
        (project / "calc.py").write_text(SOURCE, encoding="utf-8")
        output.mkdir()
        (output / "test_calc.py").write_text(SLOW if tool == "measure_coverage" else FAST,
                                             encoding="utf-8")

    class _Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

    def call(name, cid, **args):
        return AIMessage(content="", tool_calls=[{"name": name, "id": cid, "args": args}])

    current = (output / "test_calc.py").read_text(encoding="utf-8")
    responses = {
        "measure_coverage": [call("read_tests", "r0"), call("measure_coverage", "m0")],
        # The agent measures before it may edit, then runs the edited suite.
        "run_tests": [call("read_tests", "r0"), call("measure_coverage", "m0"),
                      call("edit_tests", "e0", old_text=current, new_text=SLOW),
                      call("run_tests", "t0")],
        "finish": [call("read_tests", "r0"), call("measure_coverage", "m0")],
    }[tool]
    model = _Model(responses=responses + [call("read_tests", "r9")])
    patch(tc.CoverageAgent, "build_chat_model", lambda agent: model)
    for name in (tool,) if tool != "finish" else ():
        real = getattr(tc.CoverageAgent, name)

        def observed(agent, *args, _real=real, **kwargs):
            if threading.current_thread() is not threading.main_thread():
                os.write(1, b"RUNNING\n")
            return _real(agent, *args, **kwargs)

        observed.__name__, observed.__doc__ = real.__name__, real.__doc__
        observed.__wrapped__ = real
        patch(tc.CoverageAgent, name, observed)
    real_cancel = mutation_runs.cancel_in_flight

    def cancel_in_flight():
        os.write(1, b"CANCELLING\n")
        return real_cancel()

    patch(mutation_runs, "cancel_in_flight", cancel_in_flight)
    settings = RunSettings(base_dir=base_dir, environment=dict(os.environ), sections={
        "api": {"model": "explicit"}, "main": {"hints_file": "", "seed": 37},
        "project": {"python_executable": sys.executable},
        "coverage": {"max_requests_per_phase": len(responses), "max_loop_attempts": 0},
    }, models={"explicit": {"provider": "openai", "model": "fixture",
                            "api_url": "https://example.test/v1", "api_key": "fixture"}})
    try:
        with config.use_run_settings(settings):
            tc.init_logging("subject", str(base_dir / "logs"), "coverage.jsonl")
            result = tc.generate_coverage_tests(
                "subject", {"file_path": str(project / "calc.py")}, str(output),
                baseline={"content": current, "line_coverage": 50.0, "branch_coverage": 50.0,
                          "test_count": 1, "suite_seconds": 10.0})
    except KeyboardInterrupt as stop:
        remains = cli._retry_interrupted_cleanup(stop)
        print("Interrupted.", file=sys.stderr, flush=True)
        if remains is not None:
            print(remains, file=sys.stderr, flush=True)
        return exits.INTERRUPTED
    print(f"FINISHED {result.status}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), sys.argv[2]))
