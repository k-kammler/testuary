"""Calls of the phase end in tests, by keyword: consolidate_phase_end's
context and options built from one keyword list, so a test names only the
values it sets."""
from dataclasses import asdict

from testuary import config
from testuary.consolidation import (
    PhaseEndContext,
    PhaseEndOptions,
    ReductionSettings,
    consolidate_phase_end,
)

_CONTEXT = ("target_file_path", "source_dir", "timeout", "stage_as_delivered", "log")
_REDUCTION = ("solver", "solver_time_limit_seconds", "minimum_test_runtime_ms",
              "solver_random_seed", "tie_break")


def phase_end(backend, candidates, *, mutation_runner=None, **keywords):
    """consolidate_phase_end over ``candidates`` with the context's fields,
    the suite reduction settings (the packaged ones where not named), the
    mutation method (the packaged one unless named) and the other options
    taken from ``keywords``. An unknown keyword raises TypeError."""
    def take(names):
        return {name: keywords.pop(name) for name in names if name in keywords}

    context = PhaseEndContext(backend, **take(_CONTEXT))
    reduction = ReductionSettings(**{
        **asdict(config.solver_settings(config.get_agent_config("test_coverage"))), **take(_REDUCTION)})
    keywords.setdefault("mutation_method", config.consolidation_mutation_method())
    options = PhaseEndOptions(reduction=reduction, **keywords)
    return consolidate_phase_end(context, candidates, mutation_runner, options)


def named_id(path, name):
    """The test id of test ``name`` in the file ``path``, as a run names it."""
    from pathlib import Path

    from langadapter.suites import TestId

    return TestId(Path(path).name, name, text=f"{Path(path).name}::{name}")


def names_under_coverage(monkeypatch, backend, marker, named):
    """The backend's runs pass, but a run under the coverage tool of a suite
    whose text holds ``marker`` fails, naming each of ``named``, pairs of a
    test's qualified name and its case's parameters (None for none), by its
    id with the outcome ``failed``, whether the suite holds it or not."""
    from pathlib import Path

    from langadapter.suites import TestId
    from testuary.languages import TestRunResult
    from testuary.languages.generic import TestFailure, case_suffix

    real = type(backend).run_test_files

    def run(self, paths, source_dir, *, timeout, measure_coverage_of=None, **kwargs):
        if measure_coverage_of is None or marker not in Path(paths[0]).read_text(encoding="utf-8"):
            return real(self, paths, source_dir, timeout=timeout,
                        measure_coverage_of=measure_coverage_of, **kwargs)
        name = Path(paths[0]).name
        ids = tuple(TestId(name, qualname, params, text=f"{name}::{qualname}{case_suffix(params)}")
                    for qualname, params in named)
        return TestRunResult(
            ok=False, returncode=1, timed_out=False, elapsed_s=0.1, collected=len(ids),
            passed=0, failed=len(ids), errors=0, output="FAILED",
            line_coverage=None, branch_coverage=None,
            failures=tuple(TestFailure(str(test), "failed", "fails only under the tool")
                           for test in ids),
            stopping_tests=ids)

    monkeypatch.setattr(type(backend), "run_test_files", run)


def named_under_coverage(monkeypatch, backend, failing, *, outcome="failed", timed_out=False):
    """The backend's runs pass, but a run under the coverage tool of a suite
    that holds one of the tests ``failing`` fails, naming each held one by
    its id with ``outcome``."""
    from pathlib import Path

    from testuary.languages import TestRunResult
    from testuary.languages.generic import TestFailure

    real = type(backend).run_test_files

    def run(self, paths, source_dir, *, timeout, measure_coverage_of=None, **kwargs):
        text = Path(paths[0]).read_text(encoding="utf-8")
        held = [test for test in failing if f"def {test}(" in text]
        if measure_coverage_of is None or not held:
            return real(self, paths, source_dir, timeout=timeout,
                        measure_coverage_of=measure_coverage_of, **kwargs)
        return TestRunResult(
            ok=False, returncode=1, timed_out=timed_out, elapsed_s=0.1, collected=len(held),
            passed=0, failed=len(held), errors=0, output="FAILED",
            line_coverage=None, branch_coverage=None,
            failures=tuple(TestFailure(test, outcome, "fails only under the tool")
                           for test in held),
            stopping_tests=tuple(named_id(paths[0], test) for test in held))

    monkeypatch.setattr(type(backend), "run_test_files", run)
