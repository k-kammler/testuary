"""How the starter phase handles failed model requests, through
``run_starter_phase``: which failures are retried with which waits, how a
rejected credential ends the run, what a failed request records, and the
allowance a reuse run finds exhausted.
"""
from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from starter_models import starter_settings
import test_starter_phase as harness
from testuary import model_requests as mr
from testuary import starters
from testuary import starter_salvage
from testuary import target_phases as orchestrator
from phase_values import serve_starter_settings


@pytest.fixture(autouse=True)
def _consolidation_mutation_off(monkeypatch):
    """No mutation engine can start against these synthetic projects, and the
    generation accounting under test ends before consolidation measures."""
    from testuary import config
    monkeypatch.setattr(
        config, "consolidation_measure_mutation", lambda: False,
    )


@pytest.fixture
def waits(monkeypatch):
    """Record the waits between attempts instead of spending them."""
    recorded = []
    monkeypatch.setattr(
        mr.RequestAccount, "wait", lambda self, seconds: recorded.append(seconds))
    return recorded


def _outage():
    return httpx.ConnectError("connection refused")


class _RejectedCredential(Exception):
    """Stands for a provider's 401: retrying cannot succeed."""

    status_code = 401


def _records(suites_dir):
    return [
        json.loads(line) for line in
        (Path(suites_dir) / "request_records.jsonl").read_text("utf-8").splitlines()
        if line.strip()
    ]


# ---------------------------------------------------------------------------
# Transient failures
# ---------------------------------------------------------------------------


def test_transient_failures_are_retried_within_the_slot_attempt(
    tmp_path, monkeypatch, waits, default_run,
):
    """Four outages then a usable suite spend one attempt, with doubling waits."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)

    res, record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [_outage(), _outage(), _outage(), _outage(), harness.GREEN_A]},
        monkeypatch, suite_count=1, refill_failed_suites=True,
        attempts_per_suite=4,
    )

    assert record["attempts"] == {1: 5}
    assert res.suite_generation_attempts == {1: 1}
    assert res.status == "ok"
    assert res.n_green == 1
    assert res.requests_used == 5
    assert res.logical_requests == 1
    assert res.unknown_usage_requests == 4
    assert waits == [10, 20, 40, 60]


def test_a_delivered_response_without_a_code_block_spends_the_attempt(
    tmp_path, monkeypatch, waits, default_run,
):
    """A response without a code block spends the attempt and is not retried."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    no_code = "I cannot write tests for this file."
    monkeypatch.setattr(harness, "_fenced", lambda code: code)

    res, record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [no_code, no_code]}, monkeypatch,
        suite_count=1, refill_failed_suites=True, attempts_per_suite=2,
    )

    assert record["attempts"] == {1: 2}
    assert res.suite_generation_attempts == {1: 2}
    assert res.requests_used == res.logical_requests == 2
    assert res.status == "cancelled_zero_passing"
    assert waits == []


def test_the_consecutive_ceiling_ends_the_phase_with_a_transport_error(
    tmp_path, monkeypatch, waits, default_run,
):
    """Three consecutive transient failures fail the phase with a transport
    error, not as a run without green suites."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)

    res, record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [_outage()]}, monkeypatch,
        suite_count=1, refill_failed_suites=True, attempts_per_suite=4,
        max_consecutive_transient_failures=3,
    )

    assert record["attempts"] == {1: 3}
    assert res.status == "failed_model_request"
    assert "sent again until no attempt was left" in res.detail
    assert "3 consecutive transient failures" in res.detail
    assert "ConnectError" in res.detail
    assert res.requests_used == 3
    assert res.unknown_usage_requests == 3
    assert waits == [10, 20]
    # The generation stays incomplete, and its failed requests spent no attempt.
    manifest = json.loads(
        (Path(suites_dir) / starters._GENERATION_MANIFEST).read_text("utf-8"))
    assert manifest["generation_complete"] is False
    assert manifest["generation_attempts_per_suite"] == {}
    assert manifest["requests_used"] == 3
    assert manifest["unknown_usage_requests"] == 3


def test_the_phase_total_is_shared_by_all_slots(tmp_path, monkeypatch, waits, default_run):
    """The transient failure total is shared: with each slot failing once, a
    total of two ends the phase at the second slot."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)

    res, record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [_outage(), harness.GREEN_A], 2: [_outage(), harness.GREEN_A],
         3: [_outage(), harness.GREEN_A]},
        monkeypatch, suite_count=3, refill_failed_suites=True,
        max_transient_failures_per_phase=2, cache_warmup=False,
        concurrent_requests=1,
    )

    assert res.status == "failed_model_request"
    assert "2 transient failures in this phase" in res.detail


def test_the_ceiling_is_configuration(tmp_path, monkeypatch, waits, default_run):
    """The ceiling and the waits are settings."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)

    res, record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [_outage(), _outage(), harness.GREEN_A]}, monkeypatch,
        suite_count=1, max_consecutive_transient_failures=5,
        transient_first_wait_seconds=1, max_transient_wait_seconds=1.5,
    )

    assert res.status == "ok"
    assert record["attempts"] == {1: 3}
    assert waits == [1.0, 1.5]


# ---------------------------------------------------------------------------
# Terminal failures
# ---------------------------------------------------------------------------


def test_a_terminal_failure_ends_the_phase_with_the_provider_error(
    tmp_path, monkeypatch, waits, default_run,
):
    """A rejected credential makes one request, with no retry or wait, and a
    failure status that names the error."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)

    res, record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [_RejectedCredential("invalid api key")]}, monkeypatch,
        suite_count=1, refill_failed_suites=True, attempts_per_suite=2,
    )

    assert record["attempts"] == {1: 1}
    assert res.status == "failed_model_request"
    assert res.status != "cancelled_zero_passing"
    assert "terminal model request failure" in res.detail
    assert "invalid api key" in res.detail
    assert res.requests_used == res.logical_requests == 1
    assert waits == []


def test_a_terminal_failure_stops_the_other_slots_of_the_wave(
    tmp_path, monkeypatch, waits, default_run,
):
    """A terminal failure stops the other slots before their next attempt. For
    salvage of an earlier delivery in the wave see
    test_a_response_delivered_in_the_failing_wave_is_salvaged_and_reused."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)

    res, record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [_RejectedCredential("invalid api key")],
         2: [_outage(), harness.GREEN_A], 3: [harness.GREEN_A]},
        monkeypatch, suite_count=3, cache_warmup=False, concurrent_requests=1,
    )

    assert res.status == "failed_model_request"
    # Slot 1's single request ended the phase before slots 2 and 3 started.
    assert record["attempts"] == {1: 1}


def test_a_failed_starter_phase_is_a_failed_target(tmp_path, monkeypatch, waits, default_run):
    """A failed starter phase fails the target in run_target_phases with the
    provider's error, and nothing downstream runs."""
    import test_starter_wiring as wiring
    from testuary import config
    from testuary import models

    ctx = wiring._ctx(tmp_path, monkeypatch)
    real_gac = config.get_agent_config

    def gac(name, parent=None):
        cfg = real_gac(name, parent)
        if name == "test_initial":
            cfg.update({"suite_count": 1, "suite_run_timeout_seconds": 60})
        return cfg

    monkeypatch.setattr(config, "get_agent_config", gac)
    monkeypatch.setattr(
        models, "get_api_config",
        lambda: {"model": "fake", "api_url": "", "api_key": "x"})

    def gen(prompt, *, temperature, seed, timeout):
        raise _RejectedCredential("invalid api key")

    real_run = starters.run_starter_phase

    def run(**kw):
        with models.installed(SimpleNamespace(generate_suite=gen)):
            return real_run(**kw)

    monkeypatch.setattr(starters, "run_starter_phase", run)

    result = orchestrator.run_target_phases(ctx)

    assert result.phase_outcomes["starter"] == "failed"
    assert "invalid api key" in result.phase_outcomes["starter_detail"]
    assert result.test_result is None


# ---------------------------------------------------------------------------
# Records
# ---------------------------------------------------------------------------


def test_every_request_sent_is_recorded_with_its_usage_status(
    tmp_path, monkeypatch, waits, default_run,
):
    """A dropped attempt and its retry are two requests sent of one logical
    request, the failed one recorded with unknown usage."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)

    res, _record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [_outage(), harness.GREEN_A]}, monkeypatch,
        suite_count=1, refill_failed_suites=True,
    )

    assert res.requests_used == 2
    assert res.logical_requests == 1
    assert res.unknown_usage_requests == 1
    assert (res.input_tokens, res.output_tokens) == (100, 50)

    records = _records(suites_dir)
    assert [(r["logical_request_idx"], r["attempt"], r["outcome"], r["usage_known"])
            for r in records] == [(1, 1, "transient", False), (1, 2, "delivered", True)]
    assert records[0]["error_type"] == "ConnectError"
    assert records[1]["input_tokens"] == 100
    assert "input_tokens" not in records[0]

    transport = (Path(suites_dir) / "prompt_transport.jsonl").read_text(
        "utf-8").splitlines()
    assert [json.loads(line)["request_idx"] for line in transport] == [1, 2]

    meta = json.loads(
        (Path(suites_dir) / "suite_001" / "original.json").read_text("utf-8"))
    assert meta["attempts"] == 2
    assert meta["unknown_usage_requests"] == 1
    manifest = json.loads(
        (Path(suites_dir) / starters._GENERATION_MANIFEST).read_text("utf-8"))
    assert manifest["requests_used"] == 2
    assert manifest["logical_requests"] == 1
    assert manifest["unknown_usage_requests"] == 1


def test_a_delivered_response_without_reported_usage_is_unknown_not_zero(
    tmp_path, monkeypatch, waits, default_run,
):
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    from testuary import config
    from testuary import models

    base_cfg = config.get_agent_config("test_initial")
    base_cfg.update({"suite_count": 1, "suite_run_timeout_seconds": 120})
    serve_starter_settings(monkeypatch, base_cfg)
    monkeypatch.setattr(
        models, "get_api_config",
        lambda: {"model": "fake", "api_url": "", "api_key": "x"})

    def gen(prompt, *, temperature, seed, timeout):
        # A cut stream: content but no usage report.
        return harness._fenced(harness.GREEN_A), {
            "input_tokens": 0, "output_tokens": 0, "usage_known": False}

    with models.installed(SimpleNamespace(generate_suite=gen)):
        res = starters.run_starter_phase(
            language="python", target_file_path=str(target),
            source_dir=str(root), test_output_dir=str(tests_dir),
            suites_dir=str(suites_dir), log=lambda *a, **k: None,
        )

    assert res.status == "ok"
    assert res.unknown_usage_requests == 1
    assert (res.input_tokens, res.output_tokens) == (0, 0)
    [record] = _records(suites_dir)
    assert record["usage_known"] is False


def test_context_overflow_halving_records_each_request(tmp_path, monkeypatch, waits):
    """Output halving after two context overflows makes three physical
    requests of one model request."""
    from testuary import _chat_models

    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    from testuary import config
    from testuary import models

    api = {"model": "fake", "api_url": "http://localhost:9999/v1", "api_key": "x"}
    with config.use_run_settings(starter_settings(api, max_output_tokens=16000)):
        base_cfg = config.get_agent_config("test_initial")
    base_cfg.update({"suite_count": 1, "suite_run_timeout_seconds": 120})
    serve_starter_settings(monkeypatch, base_cfg)
    monkeypatch.setattr(models, "get_api_config", lambda: api)
    limits = []

    class _Chunk:
        def __init__(self, content):
            self.content = content
            self.usage_metadata = {"input_tokens": 10, "output_tokens": 5}
            self.response_metadata = {}

        def __add__(self, other):
            return self

    class _Model:
        def __init__(self, max_tokens):
            self.max_tokens = max_tokens

        def stream(self, outbound):
            limits.append(self.max_tokens)
            if len(limits) < 3:
                raise RuntimeError(
                    "This model's context length is only 8000 tokens; you "
                    "passed 9000 input tokens")
            yield _Chunk(harness._fenced(harness.GREEN_A))

    monkeypatch.setattr(
        _chat_models, "build_chat_model", lambda **kw: _Model(kw.get("max_tokens")))

    with config.use_run_settings(starter_settings(api, max_output_tokens=16000)):
        res = starters.run_starter_phase(
            language="python", target_file_path=str(target),
            source_dir=str(root), test_output_dir=str(tests_dir),
            suites_dir=str(suites_dir), log=lambda *a, **k: None,
        )

    assert res.status == "ok"
    assert limits == [16000, 8000, 4000]
    assert res.requests_used == 3
    assert res.logical_requests == 1
    assert res.unknown_usage_requests == 2
    records = _records(suites_dir)
    assert [r["outcome"] for r in records] == [
        "context_overflow", "context_overflow", "delivered"]
    assert all(r["logical_request_idx"] == 1 for r in records)
    transport = (Path(suites_dir) / "prompt_transport.jsonl").read_text(
        "utf-8").splitlines()
    assert len(transport) == 3


# ---------------------------------------------------------------------------
# Limits across resume
# ---------------------------------------------------------------------------


def test_a_suite_reuse_run_keeps_a_spent_allowance_spent(
    tmp_path, monkeypatch, waits, default_run,
):
    """A reuse run sends nothing for an exhausted slot, its count stays spent."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    plan = {1: [harness.UNSALVAGEABLE], 2: [harness.GREEN_A]}

    first, first_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir, plan, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=2,
    )
    assert first_record["attempts"] == {1: 2, 2: 1}
    assert first.suite_generation_attempts == {1: 2, 2: 1}

    second, second_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir, plan, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=2,
        reuse_suites=True,
    )

    assert second_record["attempts"] == {}
    assert second.suite_generation_attempts == {1: 2, 2: 1}
    assert second.n_green == 1


def test_a_raised_allowance_grants_the_difference_on_a_reuse_run(
    tmp_path, monkeypatch, waits, default_run,
):
    """Raising the allowance from 2 to 4 grants two more attempts on reuse."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    plan = {1: [harness.UNSALVAGEABLE], 2: [harness.GREEN_A]}

    harness._run_per_seed(
        root, target, tests_dir, suites_dir, plan, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=2,
    )
    second, second_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir, plan, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=4,
        reuse_suites=True,
    )

    assert second_record["attempts"] == {1: 2}
    assert second.suite_generation_attempts == {1: 4, 2: 1}


def test_a_pool_archived_without_attempt_counts_is_treated_as_unspent(
    tmp_path, monkeypatch, waits, default_run,
):
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    plan = {1: [harness.UNSALVAGEABLE], 2: [harness.GREEN_A]}
    harness._run_per_seed(
        root, target, tests_dir, suites_dir, plan, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=2,
    )
    manifest = Path(suites_dir) / starters._GENERATION_MANIFEST
    data = json.loads(manifest.read_text("utf-8"))
    del data["generation_attempts_per_suite"]
    # Drop the slot evidence too, which without the counts is inconsistent.
    del data["suite_slots"]
    manifest.write_text(json.dumps(data), encoding="utf-8")
    # Such a pool has no request records with slots either.
    (Path(suites_dir) / "request_records.jsonl").unlink()
    lines = []

    def log(message, *a, **k):
        lines.append(message)

    from testuary import config
    from testuary import models
    base_cfg = config.get_agent_config("test_initial")
    base_cfg.update({"suite_count": 2, "suite_run_timeout_seconds": 120, "seed_base": 1000,
                     "refill_failed_suites": True, "attempts_per_suite": 2,
                     "reuse_suites": True})
    serve_starter_settings(monkeypatch, base_cfg)
    monkeypatch.setattr(
        models, "get_api_config",
        lambda: {"model": "fake", "api_url": "", "api_key": "x"})
    calls = []

    def gen(prompt, *, temperature, seed, timeout):
        calls.append(seed)
        return harness._fenced(harness.UNSALVAGEABLE), {
            "input_tokens": 1, "output_tokens": 1}

    with models.installed(SimpleNamespace(generate_suite=gen)):
        res = starters.run_starter_phase(
            language="python", target_file_path=str(target),
            source_dir=str(root), test_output_dir=str(tests_dir),
            suites_dir=str(suites_dir), log=log,
        )

    assert any("treated as unspent" in m for m in lines)
    # Without counts the slot's attempts restart at 1 and resend its first seed.
    assert calls == [1001, 1001001]
    assert res.suite_generation_attempts == {1: 2}


# ---------------------------------------------------------------------------
# A generation that a request failure or an interruption ended
# ---------------------------------------------------------------------------


def _manifest(suites_dir):
    return json.loads(
        (Path(suites_dir) / starters._GENERATION_MANIFEST).read_text("utf-8"))


def test_a_reuse_run_after_a_failed_phase_keeps_the_spent_attempts(
    tmp_path, monkeypatch, waits, default_run,
):
    """A phase ended by a request failure records its slots' spent attempts.
    Slot 1 spends both attempts before slot 2's refill request is rejected,
    so the reuse run sends slot 1 nothing. The rejected request spent no
    attempt, so slot 2 keeps one."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    plan = {1: [harness.UNSALVAGEABLE],
            2: [harness.UNSALVAGEABLE, _RejectedCredential("rejected")],
            3: [harness.GREEN_A]}
    first, first_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir, plan, monkeypatch,
        suite_count=3, refill_failed_suites=True, attempts_per_suite=2,
        concurrent_requests=1,
    )
    assert first.status == "failed_model_request"
    assert first_record["attempts"] == {1: 2, 2: 2, 3: 1}
    assert first.suite_generation_attempts == {1: 2, 2: 1, 3: 1}
    assert (first.requests_used, first.logical_requests) == (5, 5)
    assert first.unknown_usage_requests == 1
    manifest = _manifest(suites_dir)
    assert manifest["generation_complete"] is False
    assert manifest["generation_attempts_per_suite"] == {"1": 2, "2": 1, "3": 1}
    assert manifest["requests_used"] == 5

    second, second_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A], 2: [harness.GREEN_A], 3: [harness.GREEN_A]},
        monkeypatch, suite_count=3, refill_failed_suites=True,
        attempts_per_suite=2, reuse_suites=True,
    )

    assert second_record["attempts"] == {2: 1}
    assert second.suite_generation_attempts == {1: 2, 2: 2, 3: 1}
    assert second.status == "ok"
    assert second.n_green == 2
    # The totals cover the failed run's five requests and the reuse run's one.
    assert (second.requests_used, second.logical_requests) == (6, 6)
    assert second.unknown_usage_requests == 1
    assert second.generation_seconds >= first.generation_seconds
    manifest = _manifest(suites_dir)
    assert manifest["generation_complete"] is True
    assert manifest["generation_attempts_per_suite"] == {"1": 2, "2": 2, "3": 1}
    assert manifest["requests_used"] == 6


def test_a_response_delivered_in_the_failing_wave_is_salvaged_and_reused(
    tmp_path, monkeypatch, waits, default_run,
):
    """A suite delivered in the wave a request failure ends is salvaged. Each
    slot has one attempt here, so the reuse run keeps both salvaged suites and
    generates only the slot that got no response."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    plan = {1: [harness.GREEN_A], 2: [harness.GREEN_A],
            3: [_RejectedCredential("rejected")]}
    first, first_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir, plan, monkeypatch,
        suite_count=3, cache_warmup=False, concurrent_requests=1,
    )
    assert first.status == "failed_model_request"
    assert first_record["attempts"] == {1: 1, 2: 1, 3: 1}
    assert first.suite_generation_attempts == {1: 1, 2: 1}
    assert sorted(
        p.parent.name for p in Path(suites_dir).glob("suite_*/cleaned.py")
    ) == ["suite_001", "suite_002"]

    second, second_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A], 2: [harness.GREEN_A], 3: [harness.GREEN_A]},
        monkeypatch, suite_count=3, reuse_suites=True,
    )

    assert second_record["attempts"] == {3: 1}
    assert second.suite_generation_attempts == {1: 1, 2: 1, 3: 1}
    assert second.n_green == 3
    assert _manifest(suites_dir)["generation_complete"] is True


def test_a_failed_phase_without_a_green_suite_is_continued_not_restarted(
    tmp_path, monkeypatch, waits, default_run,
):
    """A failed generation without a green starter stays a pool to reuse, so
    slot 1's spent attempt is not drawn again."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    plan = {1: [harness.UNSALVAGEABLE], 2: [_RejectedCredential("rejected")]}
    first, _ = harness._run_per_seed(
        root, target, tests_dir, suites_dir, plan, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=1,
        cache_warmup=False, concurrent_requests=1,
    )
    assert first.status == "failed_model_request"
    assert not list(Path(suites_dir).glob("suite_*/cleaned.py"))
    assert starters._has_persisted_pool(Path(suites_dir))

    second, second_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A], 2: [harness.GREEN_A]}, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=1,
        reuse_suites=True,
    )

    assert second_record["attempts"] == {2: 1}
    assert second.suite_generation_attempts == {1: 1, 2: 1}
    assert second.n_green == 1


def test_an_interrupted_salvage_leaves_the_spent_attempts_recorded(
    tmp_path, monkeypatch, waits, default_run,
):
    """The manifest is written after each processed response, so a crash in
    the second salvage leaves both attempts recorded, marked incomplete."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    real_clean = starter_salvage._clean_suite
    cleaned = []

    def crashing_clean(*args, **kwargs):
        cleaned.append(1)
        if len(cleaned) == 2:
            raise RuntimeError("salvage crashed")
        return real_clean(*args, **kwargs)

    monkeypatch.setattr(starter_salvage, "_clean_suite", crashing_clean)

    with pytest.raises(RuntimeError, match="salvage crashed"):
        harness._run_per_seed(
            root, target, tests_dir, suites_dir,
            {1: [harness.GREEN_A], 2: [harness.GREEN_A]}, monkeypatch,
            suite_count=2, cache_warmup=False, concurrent_requests=1,
        )

    manifest = _manifest(suites_dir)
    assert manifest["generation_complete"] is False
    assert manifest["generation_attempts_per_suite"] == {"1": 1, "2": 1}
    assert manifest["requests_used"] == 2


def test_an_unreadable_manifest_is_refused_on_reuse(tmp_path):
    """A reuse that cannot read the manifest of spent attempts stops instead
    of granting every slot its allowance again."""
    suites = tmp_path / "suites"
    suites.mkdir()
    (suites / starters._GENERATION_MANIFEST).write_text(
        "{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="cannot be read"):
        starters._load_persisted_green(
            suites, {}, starters.StarterResult(), lambda *a, **k: None)


# ---------------------------------------------------------------------------
# A reuse run continues what earlier runs spent
# ---------------------------------------------------------------------------


def _crash_first_salvage(monkeypatch):
    real_clean = starter_salvage._clean_suite
    calls = []

    def crashing_clean(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise RuntimeError("salvage crashed")
        return real_clean(*args, **kwargs)

    monkeypatch.setattr(starter_salvage, "_clean_suite", crashing_clean)


def test_a_reuse_run_processes_a_saved_response_without_a_request(
    tmp_path, monkeypatch, waits, default_run,
):
    """A response is saved on arrival, so a reuse run processes one whose
    salvage crashed with the allowance spent, without a request or charge."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    _crash_first_salvage(monkeypatch)
    with pytest.raises(RuntimeError, match="salvage crashed"):
        harness._run_per_seed(
            root, target, tests_dir, suites_dir, {1: [harness.GREEN_A]},
            monkeypatch, suite_count=1, refill_failed_suites=True,
            attempts_per_suite=1,
            cache_warmup=False, concurrent_requests=1,
        )
    assert (Path(suites_dir) / "suite_001" / starters._GENERATION_DELIVERY).is_file()

    second, second_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir, {1: [harness.GREEN_A]},
        monkeypatch, suite_count=1, refill_failed_suites=True,
        attempts_per_suite=1, reuse_suites=True,
    )

    assert second_record["attempts"] == {}
    assert second.status == "ok"
    assert second.n_green == 1
    assert second.suite_generation_attempts == {1: 1}
    assert (second.requests_used, second.logical_requests) == (1, 1)
    assert not (Path(suites_dir) / "suite_001"
                / starters._GENERATION_DELIVERY).exists()


def test_a_resumed_session_starts_its_phase_total_afresh(
    tmp_path, monkeypatch, waits, default_run,
):
    """A resumed session counts only its own failures toward the phase total
    of three: one outage is retried and the slot filled. The first session's
    two stay in the records and the manifest's total."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    first, _ = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A],
         2: [_outage(), _outage(), _RejectedCredential("rejected")]},
        monkeypatch, suite_count=2, refill_failed_suites=True,
        attempts_per_suite=2, cache_warmup=False,
        concurrent_requests=1, max_transient_failures_per_phase=3,
    )
    assert first.status == "failed_model_request"

    second, second_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A], 2: [_outage(), harness.GREEN_A]}, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=2,
        reuse_suites=True, max_transient_failures_per_phase=3,
    )

    assert second_record["attempts"] == {2: 2}     # the retried outage, then the suite
    assert second.status == "ok"
    assert [r["outcome"] for r in _records(suites_dir)].count("transient") == 3
    manifest = json.loads((Path(suites_dir) / starters._GENERATION_MANIFEST).read_text("utf-8"))
    assert manifest["transient_failures"] == 3


def test_a_resumed_session_starts_a_slots_consecutive_count_afresh(
    tmp_path, monkeypatch, waits, default_run,
):
    """A resumed session restarts a slot's consecutive failure count, so after
    two failures under a limit of three, one more outage is still retried."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A],
         2: [_outage(), _outage(), _RejectedCredential("rejected")]},
        monkeypatch, suite_count=2, refill_failed_suites=True,
        attempts_per_suite=2, cache_warmup=False,
        concurrent_requests=1, max_consecutive_transient_failures=3,
    )

    second, second_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A], 2: [_outage(), harness.GREEN_A]}, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=2,
        reuse_suites=True, max_consecutive_transient_failures=3,
    )

    assert second_record["attempts"] == {2: 2}     # the retried outage, then the suite
    assert second.status == "ok"


def test_a_slot_with_a_delivered_response_is_refilled_on_resume(
    tmp_path, monkeypatch, waits, default_run,
):
    """The slot's two failures were followed by a delivered, unusable
    response. The resumed session refills it with its remaining attempts."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A],
         2: [_outage(), _outage(), harness.UNSALVAGEABLE,
             _RejectedCredential("rejected")]},
        monkeypatch, suite_count=2, refill_failed_suites=True,
        attempts_per_suite=3, cache_warmup=False,
        concurrent_requests=1, max_consecutive_transient_failures=3,
    )

    second, second_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A], 2: [_outage(), harness.GREEN_A]}, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=3,
        reuse_suites=True, max_consecutive_transient_failures=3,
    )

    assert second_record["attempts"] == {2: 2}
    assert second.status == "ok"
    assert second.suite_generation_attempts == {1: 1, 2: 2}


def test_the_reuse_totals_come_from_every_recorded_request(
    tmp_path, monkeypatch, waits, default_run,
):
    """A reuse run's totals count every recorded request sent of every
    run, failed ones included: six requests for three logical ones (slot 1's,
    slot 2's rejected and delivered ones), four without reported usage."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A],
         2: [_outage(), _outage(), _RejectedCredential("rejected")]},
        monkeypatch, suite_count=2, refill_failed_suites=True,
        attempts_per_suite=2, cache_warmup=False,
        concurrent_requests=1,
    )
    second, _ = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A], 2: [_outage(), harness.GREEN_A]}, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=2,
        reuse_suites=True,
    )

    records = _records(suites_dir)
    assert len(records) == 6
    assert (second.requests_used, second.logical_requests) == (6, 3)
    assert second.unknown_usage_requests == 4
    assert {record["slot"] for record in records} == {1, 2}
    assert len({record["run"] for record in records}) == 2


# ---------------------------------------------------------------------------
# Exhausted limits, the journal, counting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("limit", [
    {"max_transient_failures_per_phase": 3}, {"max_consecutive_transient_failures": 3},
])
def test_a_session_ended_at_a_limit_is_resumed_normally(
    tmp_path, monkeypatch, waits, limit, default_run,
):
    """A resume after a session ended at a three-failure limit requests the
    slot again and completes, and the earlier failures stay recorded."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    first, _ = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A], 2: [_outage(), _outage(), _outage()]},
        monkeypatch, suite_count=2, refill_failed_suites=True,
        attempts_per_suite=2, cache_warmup=False,
        concurrent_requests=1, **limit,
    )
    assert first.status == "failed_model_request"

    second, second_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A], 2: [harness.GREEN_A]}, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=2,
        reuse_suites=True, **limit,
    )

    assert second_record["attempts"] == {2: 1}
    assert second.status == "ok"
    outcomes = [r["outcome"] for r in _records(suites_dir)]
    assert outcomes.count("transient") == 3 and outcomes.count("delivered") == 2


def test_a_saved_response_is_processed_although_the_limits_are_exhausted(
    tmp_path, monkeypatch, waits, default_run,
):
    """The first run's salvage of slot 1's response crashed after slot 2
    exhausted the phase total. The resumed session processes the saved
    response without a request and, with its own retry counts, fills slot 2."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    _crash_first_salvage(monkeypatch)
    with pytest.raises(RuntimeError, match="salvage crashed"):
        harness._run_per_seed(
            root, target, tests_dir, suites_dir,
            {1: [harness.GREEN_A], 2: [_outage(), _outage(), _outage()]},
            monkeypatch, suite_count=2, refill_failed_suites=True,
            attempts_per_suite=2, cache_warmup=False,
            concurrent_requests=1, max_transient_failures_per_phase=3,
        )

    second, second_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A], 2: [harness.GREEN_A]}, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=2,
        reuse_suites=True, max_transient_failures_per_phase=3,
    )

    assert second_record["attempts"] == {2: 1}
    assert (Path(suites_dir) / "suite_001" / "cleaned.py").is_file()
    assert second.status == "ok"
    assert second.n_generated == 2


def test_a_record_that_cannot_be_written_stops_the_run_and_keeps_the_response(
    tmp_path, monkeypatch, waits, default_run,
):
    """A failed request-record write stops the run, keeping the delivered
    response for the reuse run to count and process: two requests, two reported."""
    from testuary import evidence as validity

    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    real_append = validity.append_jsonl
    appends = []

    def flaky_append(path, record):
        if Path(path).name == "request_records.jsonl":
            appends.append(record)
            if len(appends) == 1:
                raise OSError("injected I/O error")
        return real_append(path, record)

    monkeypatch.setattr(validity, "append_jsonl", flaky_append)
    with pytest.raises(mr.RecordNotWritten, match="injected I/O error"):
        harness._run_per_seed(
            root, target, tests_dir, suites_dir,
            {1: [harness.UNSALVAGEABLE, _RejectedCredential("stop")]},
            monkeypatch, suite_count=1, refill_failed_suites=True,
            attempts_per_suite=2,
            cache_warmup=False, concurrent_requests=1,
        )
    saved = json.loads((Path(suites_dir) / "suite_001"
                        / starters._GENERATION_DELIVERY).read_text("utf-8"))
    assert saved["unrecorded"] is True
    monkeypatch.setattr(validity, "append_jsonl", real_append)

    second, second_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir, {1: [harness.GREEN_A]},
        monkeypatch, suite_count=1, refill_failed_suites=True,
        attempts_per_suite=2, reuse_suites=True,
    )

    assert second_record["attempts"] == {1: 1}
    assert second.suite_generation_attempts == {1: 2}
    assert second.requests_used == 2
    assert second.status == "ok"


def test_an_interrupted_final_record_is_moved_aside(
    tmp_path, monkeypatch, waits, default_run,
):
    """A final record line an interruption cut short is moved aside for
    diagnosis before anything is appended, so later records stay intact."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    harness._run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [harness.GREEN_A], 2: [_RejectedCredential("stop")]}, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=2,
        cache_warmup=False, concurrent_requests=1,
    )
    journal = Path(suites_dir) / "request_records.jsonl"
    partial = b'{"phase": "initial", "run": "x", "slo'
    journal.write_bytes(journal.read_bytes() + partial)

    totals = []
    for _ in range(2):
        res, _record = harness._run_per_seed(
            root, target, tests_dir, suites_dir,
            {1: [harness.GREEN_A], 2: [_outage(), _RejectedCredential("stop")]},
            monkeypatch, suite_count=2, refill_failed_suites=True,
            attempts_per_suite=2, reuse_suites=True,
        )
        totals.append(res.requests_used)

    assert totals == [4, 6]
    assert len(_records(suites_dir)) == 6
    aside = Path(suites_dir) / "request_records.jsonl.interrupted"
    assert aside.read_bytes() == partial + b"\n"


def test_a_saved_response_counts_one_generated_suite(
    tmp_path, monkeypatch, waits, default_run,
):
    """A saved response's suite is counted once, when its processing completes
    and commits with the manifest."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    _crash_first_salvage(monkeypatch)
    with pytest.raises(RuntimeError, match="salvage crashed"):
        harness._run_per_seed(
            root, target, tests_dir, suites_dir, {1: [harness.GREEN_A]},
            monkeypatch, suite_count=1, refill_failed_suites=True,
            attempts_per_suite=1,
            cache_warmup=False, concurrent_requests=1,
        )
    assert _manifest(suites_dir)["n_generated"] == 0

    second, second_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir, {1: [harness.GREEN_A]},
        monkeypatch, suite_count=1, refill_failed_suites=True,
        attempts_per_suite=1, reuse_suites=True,
    )

    assert second_record["attempts"] == {}
    assert second.n_generated == 1
    assert second.requests_used == 1


def test_a_committed_saved_response_is_not_processed_again(
    tmp_path, monkeypatch, waits, default_run,
):
    """A saved response left behind after its processing was committed is
    removed on reuse and counted only once."""
    root, target, tests_dir, suites_dir = harness._project(tmp_path)
    first, _ = harness._run_per_seed(
        root, target, tests_dir, suites_dir, {1: [harness.GREEN_A]},
        monkeypatch, suite_count=1, refill_failed_suites=True,
        attempts_per_suite=1,
        cache_warmup=False, concurrent_requests=1,
    )
    [committed] = _manifest(suites_dir)["processed_deliveries"]
    saved = Path(suites_dir) / "suite_001" / starters._GENERATION_DELIVERY
    saved.write_text(json.dumps({"id": committed, "slot": 1, "slot_attempt": 1, "seed": 1, "text": "x"}),
                     encoding="utf-8")

    second, second_record = harness._run_per_seed(
        root, target, tests_dir, suites_dir, {1: [harness.GREEN_A]},
        monkeypatch, suite_count=1, refill_failed_suites=True,
        attempts_per_suite=1, reuse_suites=True,
    )

    assert second_record["attempts"] == {}
    assert not saved.exists()
    assert second.n_generated == first.n_generated == 1
