"""The seams the demonstration mode attaches to, each inert in a normal run:
an installed model source, an installed cover source, a backend override and
a stated runtime version."""
from __future__ import annotations

import pytest

import test_starter_phase as starter_tests
import test_model_surface_snapshot as surface
from testuary import reduction as cover_source
from testuary import models as model_source
from testuary import languages, phase_state, reduction
from testuary import config
from testuary import models
from testuary.languages import LanguageBackend
from testuary import starters
from phase_values import serve_starter_settings

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


class _Source:
    """A model source that records what it was asked for."""

    def __init__(self, suites=()):
        self.suites = list(suites)
        self.seed_calls = []
        self.phases = []

    def generate_suite(self, messages, *, temperature, seed, timeout):
        self.seed_calls.append({"seed": seed, "temperature": temperature,
                                "roles": [m["role"] for m in messages]})
        code = self.suites[len(self.seed_calls) - 1]
        return starter_tests._fenced(code), {"input_tokens": 1, "output_tokens": 1}

    def agent_chat_model(self, phase):
        self.phases.append(phase)
        return ("model for", phase)


@pytest.fixture(autouse=True)
def _empty_slot():
    model_source.uninstall()
    cover_source.uninstall()
    yield
    model_source.uninstall()
    cover_source.uninstall()


# ---------------------------------------------------------------------------
# Model source
# ---------------------------------------------------------------------------

def test_a_second_source_is_refused():
    with model_source.installed(_Source()):
        with pytest.raises(RuntimeError, match="already installed"):
            model_source.install(_Source())
    assert model_source.current() is None


def test_an_agent_takes_its_model_from_an_installed_source(tmp_path, monkeypatch):
    monkeypatch.setattr(
        models, "build_resolved_chat_model",
        lambda *a, **k: pytest.fail("the configured model must not be built"),
    )
    agent = surface._BUILDERS["test_coverage"](tmp_path)
    source = _Source()
    with model_source.installed(source):
        assert agent.build_chat_model() == ("model for", "coverage")
    assert source.phases == ["coverage"]


def test_without_a_source_an_agent_builds_its_configured_model(tmp_path, monkeypatch):
    """The run's model, resolved for the agent's phase."""
    sentinel = object()
    built = []
    monkeypatch.setattr(models, "build_resolved_chat_model",
                        lambda settings, **k: built.append(settings.phase) or sentinel)
    agent = surface._BUILDERS["test_mutation"](tmp_path)
    assert agent.build_chat_model() is sentinel
    assert built == ["mutation"]


def _drive_starter(tmp_path, monkeypatch, n):
    root, target, tests_dir, suites_dir = starter_tests._project(tmp_path)
    base_cfg = config.get_agent_config("test_initial")
    base_cfg.update({"suite_count": n, "suite_run_timeout_seconds": 120})
    serve_starter_settings(monkeypatch, base_cfg)
    monkeypatch.setattr(
        models, "get_api_config",
        lambda: {"model": "fake", "api_url": "", "api_key": "x"})
    # No mutation engine in a unit test: this is about where suites come from.
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: False)
    return starters.run_starter_phase(
        language="python", target_file_path=str(target),
        source_dir=str(root), test_output_dir=str(tests_dir),
        suites_dir=str(suites_dir), matrices_dir=None,
        log=lambda *a, **k: None,
    )


SUITE = (
    "from calc import add\n\n"
    "def test_add():\n    assert add(1, 2) == 3\n"
)


def test_the_starter_phase_takes_its_suites_from_an_installed_source(
    tmp_path, monkeypatch,
):
    monkeypatch.setattr(
        starters, "_default_generate",
        lambda *a, **k: pytest.fail("the provider must not be contacted"),
    )
    source = _Source([SUITE, SUITE])
    with model_source.installed(source):
        res = _drive_starter(tmp_path, monkeypatch, 2)
    assert res.status == "ok", res.detail
    assert len(source.seed_calls) == 2
    # Each slot asks with its own seed, as it does from the provider.
    assert len({call["seed"] for call in source.seed_calls}) == 2
    assert all(call["roles"] == ["system", "user"] for call in source.seed_calls)


def test_without_a_source_the_starter_phase_calls_the_provider(
    tmp_path, monkeypatch,
):
    calls = []

    def provider(api_cfg, messages, **kwargs):
        calls.append(kwargs["seed"])
        return starter_tests._fenced(SUITE), {"input_tokens": 1, "output_tokens": 1}

    monkeypatch.setattr(starters, "_default_generate", provider)
    res = _drive_starter(tmp_path, monkeypatch, 1)
    assert res.status == "ok", res.detail
    assert len(calls) == 1


# ---------------------------------------------------------------------------
# Cover source
# ---------------------------------------------------------------------------

class _Cover:
    """A cover source that records the problems it was asked to solve."""

    def __init__(self, result=None):
        self.result = result
        self.calls = []

    def solve_exact(self, per_test_tokens, per_test_cost_ms, **settings):
        self.calls.append((per_test_tokens, per_test_cost_ms, settings))
        return self.result


_TOKENS = {"a": frozenset({"1", "2"}), "b": frozenset({"2"}),
           "c": frozenset({"3"})}
_COSTS = {"a": 1.0, "b": 1.0, "c": 2.0}


def test_a_second_cover_source_is_refused():
    with cover_source.installed(_Cover()):
        with pytest.raises(RuntimeError, match="already installed"):
            cover_source.install(_Cover())
    assert cover_source.current() is None


def test_the_exact_reduction_takes_its_selection_from_an_installed_source(
    monkeypatch,
):
    monkeypatch.setattr(reduction, "exact_cover",
                        lambda *a, **k: pytest.fail("the solver must not run"))
    charged = []
    monkeypatch.setattr(phase_state, "record_time",
                        lambda part, seconds: charged.append(part))
    sentinel = object()
    with cover_source.installed(_Cover(sentinel)) as source:
        assert reduction.solve_cover(
            _TOKENS, _COSTS, forced_ids=frozenset({"c"}), time_limit_s=30.0,
            minimum_test_runtime_ms=0.5, random_seed=7,
        ) is sentinel
    assert source.calls == [(_TOKENS, _COSTS, {
        "forced_ids": frozenset({"c"}), "time_limit_s": 30.0,
        "minimum_test_runtime_ms": 0.5, "random_seed": 7,
        "tie_break": "name_order", "imputed_cost_ms": None,
    })]
    # Its time counts as the phase's cover time, as the solver's does.
    assert charged == [phase_state.TIME_COVER]


def test_the_greedy_reduction_ignores_an_installed_source():
    with cover_source.installed(_Cover()) as source:
        result = reduction.solve_cover(_TOKENS, _COSTS, solver="greedy")
    assert source.calls == [] and result.keep == ("a", "c")


def test_without_a_source_the_exact_reduction_runs_the_solver():
    result = reduction.solve_cover(_TOKENS, _COSTS)
    assert result.keep == ("a", "c") and result.solver == "exact"
    assert [tier["name"] for tier in result.solver_metadata["tiers"]] == [
        "runtime", "test_count", "rank_sum", "rank_square_sum", "uniqueness"]


# ---------------------------------------------------------------------------
# Backend override
# ---------------------------------------------------------------------------

def test_a_backend_override_replaces_and_restores():
    original = languages.get_backend("python")
    # The registry keeps file_extensions on its adapter. A stand-in must accept it.
    stand_in = LanguageBackend()
    previous = languages.override_backend("python", stand_in)
    try:
        assert previous is original
        assert languages.get_backend("python") is stand_in
    finally:
        languages.override_backend("python", previous)
    assert languages.get_backend("python") is original


def test_an_unregistered_language_cannot_be_overridden():
    with pytest.raises(KeyError):
        languages.override_backend("cobol", LanguageBackend())

