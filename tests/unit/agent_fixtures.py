"""Builders the coding agents' unit tests share: the two agents over
synthetic deps, a project on disk for the coverage agent, a test run's
result, a run as delivered in which the files pass and a fake backend."""
from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from agent_prompts import python_prompts

from testuary.agent import coverage_agent as tc
from testuary.agent import mutation_agent as tm
from testuary.agent._helpers import AgentLogger

#: The target's source the builders write and name when a test gives none.
SOURCE = "def f(): return 1\n"


def _streamed_log(directory, agent_name: str) -> AgentLogger:
    """A log streamed into ``directory``, as a run's phase starts one."""
    logger = AgentLogger(project_name="test", agent_name=agent_name)
    logger.init()
    logger.enable_streaming(str(directory), f"{agent_name}.jsonl")
    return logger


def coverage_agent(root, **deps) -> "tc.CoverageAgent":
    """A CoverageAgent with a fresh session over synthetic deps: the target
    ``root``/src/foo.py holding SOURCE, its tests in ``root`` and no key
    functions. ``deps`` replace any of them. Its log streams to
    ``root``/logs, where it writes its other logs."""
    fields = dict(target_file_path=str(Path(root) / "src" / "foo.py"), target_file_content=SOURCE,
                  source_dir=str(root), test_output_dir=str(root), language="python",
                  key_functions=[])
    fields.update(deps)
    return tc.CoverageAgent(prompts=python_prompts("test_coverage"),
                            deps=tc.CoverageTestDeps(**fields), session=tc.TestCoverageSession(),
                            logger=_streamed_log(Path(root) / "logs", "test_coverage"))


def mutation_agent(root, *, mutation_session=None, **deps) -> "tm.MutationAgent":
    """A MutationAgent with a fresh session over synthetic deps: the target
    ``root``/src/foo.py holding SOURCE and its test file
    test_foo_mutation.py in ``root``. ``deps`` replace any of them. Its log
    streams to ``root``/logs, where it writes its other logs."""
    fields = dict(target_file_path=str(Path(root) / "src" / "foo.py"), target_file_content=SOURCE,
                  source_dir=str(root), test_output_dir=str(root), language="python",
                  mutation_test_file_name="test_foo_mutation.py")
    fields.update(deps)
    return tm.MutationAgent(prompts=python_prompts("test_mutation"),
                            deps=tm.MutationTestDeps(**fields), session=tm.TestMutationSession(),
                            mutation_session=mutation_session,
                            logger=_streamed_log(Path(root) / "logs", "test_mutation"))


def logs_in(monkeypatch, directory) -> None:
    """Both agents log to ``directory`` as a run's phases do, for a test
    that runs an entry without starting its logging."""
    for module, name in ((tc, "test_coverage"), (tm, "test_mutation")):
        monkeypatch.setattr(module._log, "logger", _streamed_log(directory, name))


def project_agent(tmp_path: Path, *, source: str = SOURCE, test_source=None) -> "tc.CoverageAgent":
    """A CoverageAgent over src/foo.py holding ``source`` and the directory
    tests below ``tmp_path``, which holds test_foo.py with ``test_source``
    when it is given."""
    target = tmp_path / "src" / "foo.py"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(source, encoding="utf-8")
    test_dir = tmp_path / "tests"
    test_dir.mkdir(parents=True, exist_ok=True)
    if test_source is not None:
        (test_dir / "test_foo.py").write_text(test_source, encoding="utf-8")
    return coverage_agent(tmp_path, target_file_path=str(target), target_file_content=source,
                          test_output_dir=str(test_dir))


def target_agent(tmp_path: Path, *, stem: str = "target",
                 source: str = "def f(x):\n    return x + 1\n") -> "tc.CoverageAgent":
    """A CoverageAgent over ``tmp_path``/``stem``.py holding ``source``, with
    its tests in ``tmp_path`` and the key function f."""
    target = tmp_path / f"{stem}.py"
    target.write_text(source, encoding="utf-8")
    return coverage_agent(tmp_path, target_file_path=str(target), target_file_content=source,
                          key_functions=["f"])


def run_result(**fields):
    """A TestRunResult of a green run that collected nothing and measured no
    coverage, with ``fields`` replacing any value. A timed-out run has no
    elapsed time. The class is imported here so pytest does not collect it."""
    from testuary.languages import TestRunResult

    values = dict(ok=True, returncode=0, timed_out=False,
                  elapsed_s=None if fields.get("timed_out") else 0.1,
                  collected=0, passed=0, failed=0, errors=0, output="",
                  line_coverage=None, branch_coverage=None, missing_lines=())
    values.update(fields)
    return TestRunResult(**values)


def passing_delivery(test_paths, source_dir, *, timeout, **_options):
    """A run as delivered in which the files pass: the normal run runs a
    green file as delivered."""
    from testuary.languages import DeliveredRun

    return DeliveredRun(cases=(), failing=(), reason=None, output="")


def install_backend(monkeypatch, result_or_fn, counter=None) -> None:
    """Route the coverage agent's test runs through a fake run_test_files
    that answers ``result_or_fn``, a result or a function of no arguments,
    and counts its calls in ``counter["n"]`` when given. Runs as delivered
    pass, and the other reads are the real Python backend's."""
    real = tc.languages.get_backend("python")

    def run(test_paths, source_dir, *, timeout, measure_coverage_of=None,
            instrument_target=None, fail_fast=False, tb="short", record_leaks=False):
        if counter is not None:
            counter["n"] += 1
        return result_or_fn() if callable(result_or_fn) else result_or_fn

    monkeypatch.setattr(tc.languages, "get_backend", lambda lang: SimpleNamespace(
        run_test_files=run, run_as_delivered=passing_delivery,
        find_target_test_file=real.find_target_test_file,
        test_metrics_observation=real.test_metrics_observation))
