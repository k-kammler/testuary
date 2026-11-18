"""The display contract accepts observations, not inferred execution outcomes."""
from dataclasses import FrozenInstanceError

import pytest

from testuary.presentation import DisplayEvent, Metric, Usage


def event(**overrides):
    fields = dict(run_id="run-1", target="src/calc.py", phase="mutation",
                  operation="measure", state="running", elapsed_s=4)
    return DisplayEvent(**{**fields, **overrides})


def test_zero_and_missing_are_distinct_immutable_observations():
    observation = event(metrics=(Metric("Mutation", 0, "%"),
                                Metric("Line coverage", None, "%", "incomplete")),
                        usage=Usage(input_tokens=0))
    assert observation.metrics[0].value == 0
    assert observation.metrics[1].value is None
    assert observation.usage.input_tokens == 0
    assert observation.usage.output_tokens is None
    with pytest.raises(FrozenInstanceError):
        observation.state = "succeeded"


@pytest.mark.parametrize("overrides", [
    {"state": "perfect"}, {"provenance": "guess"}, {"elapsed_s": -1},
    {"elapsed_s": float("nan")}, {"completed": -1}, {"total": True},
    {"completed": 3, "total": 2}, {"operation": ""},
])
def test_invalid_observation_is_not_silently_normalized(overrides):
    with pytest.raises(ValueError):
        event(**overrides)


def test_unknown_work_total_and_empty_work_are_not_completion_claims():
    assert event().total is None
    assert event(completed=0, total=0).total == 0


def test_request_usage_names_do_not_conflate_logical_and_physical_work():
    usage = Usage(logical_requests=2, physical_requests=5, cached_input_tokens=None)
    assert usage.logical_requests == 2
    assert usage.physical_requests == 5
    assert usage.cached_input_tokens is None
    with pytest.raises(ValueError):
        Usage(physical_requests=-1)


def test_metric_extensions_need_no_closed_metric_registry():
    metric = Metric("Project-specific complexity", 7.5, " units")
    assert event(metrics=[metric]).metrics == (metric,)
    with pytest.raises(ValueError):
        Metric("Mutation", float("inf"))
