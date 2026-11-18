"""The process test_second_stop.py stops: the mutation agent measures in a
pooled Cosmic Ray http session with two workers, opened and closed as the
orchestrator does, and the interrupt ends the process as the command line
does. ``MEASURING`` on stdout marks the tool's measurement in flight.

Usage: python _second_stop_child.py DIR
"""
from __future__ import annotations

import os
import sys
import threading
from pathlib import Path


def main(base_dir: Path) -> int:
    # The test module's imports, as pytest finds them: tests/integration,
    # then tests/ for the helpers the test files share (python_runtime).
    here = Path(__file__).resolve().parent
    sys.path[:0] = [str(here), str(here.parent)]
    from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
    from langchain_core.messages import AIMessage

    import test_interrupted_mutation_measurement as base
    from testuary import __main__ as cli
    from testuary import exits
    from testuary import languages
    from testuary import config
    from testuary.languages import mutation as mutation_runs
    from testuary import measurement
    from testuary.agent import mutation_agent as tm
    from testuary.settings import RunSettings

    mutation_runs.MutationRunner.full_operators = lambda self: list(base.OPERATORS)
    project = base_dir / "subject"
    project.mkdir()
    (project / "pyproject.toml").write_text("", encoding="utf-8")
    (project / "calc.py").write_text(base.MUTATION_TARGET, encoding="utf-8")
    output = base_dir / "tests"
    output.mkdir()

    class _Model(FakeMessagesListChatModel):
        def bind_tools(self, tools, *, tool_choice=None, **kwargs):
            return self

    def call(name, cid, **args):
        return AIMessage(content="", tool_calls=[{"name": name, "id": cid, "args": args}])

    # Every test sleeps, so each mutant's run takes seconds.
    slow = ("import time\nfrom calc import add, sub\n\n\n"
            "def test_add():\n    time.sleep(3)\n    assert add(1, 2) == 3\n\n\n"
            "def test_sub():\n    time.sleep(3)\n    assert sub(3, 2) == 1\n")
    model = _Model(responses=[
        call("read_tests", "r0"),
        # The loop's start measures the inherited suite before any edit.
        call("measure_mutation", "m0"),
        call("edit_tests", "e0", old_text=base.MUTATION_BASELINE, new_text=slow),
        call("run_tests", "t0"),
        call("measure_mutation", "m1"),
        call("run_tests", "t1"),
    ])
    tm.MutationAgent.build_chat_model = lambda agent: model
    tm._seed_mutation_baseline_from_matrix = lambda agent: False
    real_run = mutation_runs.MutationSession.run

    def run(self, **kwargs):
        if threading.current_thread() is not threading.main_thread():
            print("MEASURING", flush=True)
        return real_run(self, **kwargs)

    mutation_runs.MutationSession.run = run
    real_cancel = mutation_runs.MutationSession.cancel

    def cancel(self):
        # The first stop reached the measurement: the second one is timed
        # from here, so it lands during the cancel's cleanup.
        os.write(1, b"CANCELLING\n")
        real_cancel(self)

    mutation_runs.MutationSession.cancel = cancel
    settings = RunSettings(base_dir=base_dir, environment=dict(os.environ), sections={
        "api": {"model": "explicit"}, "main": {"hints_file": "", "seed": 37},
        "project": {"python_executable": sys.executable},
        "mutation": {"max_requests_per_phase": 6, "max_loop_attempts": 0, "goal": 100,
                          "cosmic_ray_catalogue_cache_dir": "", "cosmic_ray_distributor": "http",
                          "engine_workers": 2, "cosmic_ray_clone_sweep_hours": 0},
    }, models={"explicit": {"provider": "openai", "model": "fixture",
                            "api_url": "https://example.test/v1", "api_key": "fixture"}})
    try:
        with config.use_run_settings(settings):
            tm.init_logging("subject", str(base_dir / "logs"), "mutation.jsonl")
            runner = languages.get_backend("python").mutation_runner(
                dict(measurement.mutation_config()), source_dir=str(project),
                log=lambda *args, **kwargs: None)
            session = runner.open(2, suite_dirs=(output,))
            tm.set_mutation_session(session)
            try:
                tm.generate_mutation_tests(
                    "subject", str(project / "calc.py"), str(output),
                    preseed={"content": base.MUTATION_BASELINE, "file_stem": "calc"},
                    coverage_baseline_pct=80.0)
            except BaseException as exc:
                mutation_runs.close_mutation_session(session, primary=exc)
                raise
            else:
                mutation_runs.close_mutation_session(session)
            finally:
                tm.set_mutation_session(None)
    except KeyboardInterrupt as stop:
        remains = cli._retry_interrupted_cleanup(stop)
        print("Interrupted.", file=sys.stderr, flush=True)
        if remains is not None:
            print(remains, file=sys.stderr, flush=True)
        return exits.INTERRUPTED
    print("NOT STOPPED", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1])))
