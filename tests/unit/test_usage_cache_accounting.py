"""Unknown token usage and unreported cache splits are never recorded as zero.

Every request records whether its usage is known. A reported cached part
(Ollama's native ``prompt_eval_cached_count``, DeepSeek's
``prompt_cache_hit_tokens``, the OpenAI ``cached_tokens`` detail) is recorded
as reported, otherwise the cache split is unknown. Covers the starter suite
generator, starter suite generation with its request journal, ledger and
manifest, the editing agents' shared model turn and the result table.
"""
from __future__ import annotations

from contextlib import contextmanager
import csv
import json

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

import test_agent_transport_failures as transport
from starter_models import generate, starter_settings

from testuary.agent import _langgraph
from testuary import _chat_models
from testuary import model_requests as mr
from testuary import results as csv_writer
from testuary import config as config_module
from testuary import models
from testuary import starters
from phase_values import serve_starter_settings


@pytest.mark.parametrize("counts, expected", [
    ((100, 5, 80), (100, 5, 80)),
    ((100, 5, None), (100, 5, None)),
    ((100, 5, 0), (100, 5, 0)),
    ((10, 1, 11), (10, 1, None)),
    ((10, 1, -1), (10, 1, None)),
    ((10, 1, "3"), (10, 1, None)),
    ((None, 5, 0), None),
    ((10, None, 0), None),
    (("10", 5, 0), None),
    ((-1, 5, 0), None),
    ((True, 5, 0), None),
    ((10.0, 5, 0), None),
])
def test_reported_usage_keeps_absent_and_invalid_counts_unknown(counts, expected):
    usage = mr.reported_usage(*counts)
    if expected is None:
        assert usage is None
    else:
        assert (usage.input_tokens, usage.output_tokens, usage.cache_hit_tokens) == expected


def _native(monkeypatch, *done_lines):
    class Response:
        def raise_for_status(self):
            pass

        def iter_lines(self):
            yield '{"message":{"content":"answer"}}'
            yield from done_lines

    @contextmanager
    def stream(*args, **kwargs):
        yield Response()

    monkeypatch.setattr(httpx, "stream", stream)
    monkeypatch.setattr(_chat_models, "build_chat_model", lambda **k: pytest.fail("wrong route"))


@pytest.mark.parametrize("done, expected", [
    ({"prompt_eval_count": 100, "prompt_eval_cached_count": 80, "eval_count": 5},
     {"input_tokens": 100, "output_tokens": 5, "cache_hit_tokens": 80,
      "cache_miss_tokens": 20, "usage_known": True}),
    ({"prompt_eval_count": 100, "eval_count": 5},
     {"input_tokens": 100, "output_tokens": 5, "cache_hit_tokens": None,
      "cache_miss_tokens": None, "usage_known": True}),
    ({"prompt_eval_count": 10, "prompt_eval_cached_count": 11, "eval_count": 1},
     {"input_tokens": 10, "output_tokens": 1, "cache_hit_tokens": None,
      "cache_miss_tokens": None, "usage_known": True}),
    ({}, {"usage_known": False}),
    ({"eval_count": 5}, {"usage_known": False}),
    ({"prompt_eval_count": "100", "eval_count": 5}, {"usage_known": False}),
])
def test_native_starter_generation_reports_only_provider_counts(done, expected, monkeypatch):
    _native(monkeypatch, json.dumps({"done": True, **done}))
    text, usage = generate(
        {"model": "m", "api_url": "https://ollama.com/v1", "api_key": "k"},
        "prompt", temperature=0.7, seed=1, timeout=60,
    )
    assert text == "answer"
    assert {key: usage[key] for key in expected} == expected
    if not expected["usage_known"]:
        assert usage["cache_hit_tokens"] is None and usage["cache_miss_tokens"] is None


def test_native_starter_generation_without_a_done_line_is_unknown(monkeypatch):
    _native(monkeypatch)
    _text, usage = generate(
        {"model": "m", "api_url": "https://ollama.com/v1", "api_key": "k"},
        "prompt", temperature=0.7, seed=1, timeout=60,
    )
    assert usage["usage_known"] is False


@pytest.mark.parametrize("chunk, expected", [
    (dict(usage_metadata={"input_tokens": 50, "output_tokens": 4, "total_tokens": 54,
                          "input_token_details": {"cache_read": 30}}), (50, 4, 30)),
    (dict(usage_metadata={"input_tokens": 50, "output_tokens": 4, "total_tokens": 54},
          response_metadata={"token_usage": {"prompt_tokens": 50, "completion_tokens": 4,
                                             "prompt_cache_hit_tokens": 7}}), (50, 4, 7)),
    (dict(usage_metadata={"input_tokens": 50, "output_tokens": 4, "total_tokens": 54},
          response_metadata={"token_usage": {"prompt_tokens": 50, "completion_tokens": 4,
                                             "prompt_tokens_details": {"cached_tokens": 0}}}),
     (50, 4, 0)),
    (dict(usage_metadata={"input_tokens": 50, "output_tokens": 4, "total_tokens": 54}),
     (50, 4, None)),
    (dict(), None),
])
def test_compatible_starter_generation_reports_only_provider_counts(chunk, expected, monkeypatch):
    class Model:
        def stream(self, messages):
            yield AIMessageChunk(content="answer", **chunk)

    monkeypatch.setattr(_chat_models, "build_chat_model", lambda **k: Model())
    monkeypatch.setattr(httpx, "stream", lambda *a, **k: pytest.fail("wrong route"))
    _text, usage = generate(
        {"model": "m", "api_url": "https://example.test/v1", "api_key": "k"},
        "prompt", temperature=0.7, seed=1, timeout=60,
    )
    if expected is None:
        assert usage["usage_known"] is False
    else:
        assert usage["usage_known"] is True
        assert (usage["input_tokens"], usage["output_tokens"], usage["cache_hit_tokens"]) == expected


def _run_starter(tmp_path, monkeypatch, done):
    root = tmp_path / "subject"
    root.mkdir()
    target = root / "module.py"
    target.write_text("def add(a, b):\n    return a + b\n", encoding="utf-8")
    api = {"model": "chosen", "api_url": "https://ollama.com/v1", "api_key": "key"}
    with config_module.use_run_settings(starter_settings(api)):
        config = config_module.get_agent_config("test_initial")
    config.update({"suite_count": 1, "attempts_per_suite": 1})
    serve_starter_settings(monkeypatch, config)
    monkeypatch.setattr(models, "get_api_config", lambda: api)
    # A reply without a code block spends the slot without any measurement.
    _native(monkeypatch, json.dumps({"done": True, **done}))
    starters_dir = tmp_path / "suites"
    with config_module.use_run_settings(starter_settings(api)):
        result = starters.run_starter_phase(
            language="python", target_file_path=str(target),
            source_dir=str(root), test_output_dir=str(tmp_path / "output"),
            suites_dir=str(starters_dir), log=lambda *a, **k: None,
        )
    records = [json.loads(line) for line in
               (starters_dir / starters._REQUEST_RECORDS).read_text("utf-8").splitlines()]
    manifest = json.loads((starters_dir / starters._GENERATION_MANIFEST).read_text("utf-8"))
    return result, records, manifest


def test_actual_starter_phase_records_a_reported_cache_split(tmp_path, monkeypatch):
    result, records, manifest = _run_starter(tmp_path, monkeypatch, {
        "prompt_eval_count": 100, "prompt_eval_cached_count": 80, "eval_count": 5})
    assert [(r["usage_known"], r["cache_usage_known"], r["cache_hit_tokens"])
            for r in records] == [(True, True, 80)]
    assert (result.input_tokens, result.cache_hit_tokens, result.unknown_cache_requests) == (100, 80, 0)
    assert (manifest["cache_hit_tokens"], manifest["unknown_cache_requests"]) == (80, 0)


def test_actual_starter_phase_keeps_an_unreported_cache_split_unknown(tmp_path, monkeypatch):
    result, records, manifest = _run_starter(tmp_path, monkeypatch, {
        "prompt_eval_count": 100, "eval_count": 5})
    assert [(r["usage_known"], r["cache_usage_known"]) for r in records] == [(True, False)]
    assert "cache_hit_tokens" not in records[0]
    assert (result.input_tokens, result.cache_hit_tokens, result.unknown_cache_requests) == (100, 0, 1)
    assert result.unknown_usage_requests == 0
    assert (manifest["cache_hit_tokens"], manifest["unknown_cache_requests"]) == (0, 1)


def test_actual_starter_phase_records_an_unreported_usage_as_unknown(tmp_path, monkeypatch):
    result, records, manifest = _run_starter(tmp_path, monkeypatch, {})
    assert [r["usage_known"] for r in records] == [False]
    assert "input_tokens" not in records[0] and "cache_usage_known" not in records[0]
    assert (result.input_tokens, result.unknown_usage_requests) == (0, 1)
    # The unknown request is counted once, as unknown usage.
    assert result.unknown_cache_requests == 0
    assert manifest["unknown_usage_requests"] == 1


def _ledger(tmp_path, *records):
    (tmp_path / starters._REQUEST_RECORDS).write_text(
        "".join(json.dumps({"slot": 1, "run": "r", "logical_request_idx": index,
                            "outcome": mr.DELIVERED, **record}) + "\n"
                for index, record in enumerate(records, start=1)),
        encoding="utf-8")
    return starters._read_generation_ledger(tmp_path)


def test_ledger_counts_flagged_cache_splits_and_leaves_unflagged_records_unknown(tmp_path):
    current = _ledger(tmp_path,
                      {"usage_known": True, "input_tokens": 10, "output_tokens": 1,
                       "cache_usage_known": True, "cache_hit_tokens": 4},
                      {"usage_known": True, "input_tokens": 10, "output_tokens": 1,
                       "cache_usage_known": False},
                      {"usage_known": False})
    assert (current.cache_hit_tokens, current.unknown_cache_requests,
            current.unknown_usage_requests) == (4, 1, 1)
    unflagged = _ledger(tmp_path, {"usage_known": True, "input_tokens": 10,
                                "output_tokens": 1, "cache_hit_tokens": 0})
    assert unflagged.cache_hit_tokens == 0
    assert unflagged.unknown_cache_requests is None


def _answer(token_usage):
    message = transport._coverage_answer()
    return message.model_copy(update={"response_metadata": {"token_usage": token_usage}})


@pytest.mark.parametrize("token_usage, cache, unknown", [
    ({"prompt_tokens": 250, "completion_tokens": 40,
      "prompt_tokens_details": {"cached_tokens": 200}}, 200, 0),
    ({"prompt_tokens": 250, "completion_tokens": 40}, 0, 1),
])
def test_actual_coverage_phase_reports_cache_splits_through_its_result(
        token_usage, cache, unknown, tmp_path, monkeypatch, default_run):
    model = transport._FailThenScript(responses=[_answer(token_usage)], failures=[])
    result, _captured, _lines, log_dir = transport._drive_run_phase(tmp_path, monkeypatch, model)
    assert (result.input_tokens, result.output_tokens) == (250, 40)
    assert (result.cache_hit_tokens, result.unknown_cache_requests) == (cache, unknown)
    delivered = [json.loads(line) for line in
                 (log_dir / "coverage_usage.jsonl").read_text("utf-8").splitlines()
                 if line.strip()]
    assert [line["outcome"] for line in delivered] == ["delivered"]
    if unknown:
        assert delivered[0]["cache_usage_known"] is False
        assert delivered[0]["cache_hit_tokens"] is None
        assert delivered[0]["cache_miss_tokens"] is None
    else:
        assert "cache_usage_known" not in delivered[0]
        assert (delivered[0]["cache_hit_tokens"], delivered[0]["cache_miss_tokens"]) == (200, 50)


def test_editing_usage_without_a_reported_split_is_unknown():
    response = AIMessage(content="", response_metadata={
        "token_usage": {"prompt_tokens": 9, "completion_tokens": 2}})
    usage = _langgraph._usage_from_response(response)
    assert (usage.input_tokens, usage.output_tokens, usage.cache_hit_tokens) == (9, 2, None)
    tracker = _langgraph.UsageTracker()
    tracker.note_tokens(usage.input_tokens, usage.output_tokens, usage.cache_hit_tokens)
    tracker.note_tokens(9, 2, 5)
    assert (tracker.input_tokens_used, tracker.cache_hit_tokens_used,
            tracker.unknown_cache_requests) == (18, 5, 1)


def test_result_table_appends_unknown_cache_columns_and_leaves_absent_blank(tmp_path, default_run):
    columns = csv_writer.TESTS_FIELDNAMES
    added = ["starter_unknown_cache_requests", "coverage_unknown_cache_requests",
             "mutation_unknown_cache_requests"]
    assert columns[-4:-1] == added
    assert columns.index("source_observations") < columns.index(added[0])
    current = starters.StarterResult(cache_hit_tokens=80, unknown_cache_requests=1)
    unflagged = starters.StarterResult(cache_hit_tokens=0, unknown_cache_requests=None)
    for target_number, result in enumerate((current, unflagged), start=1):
        csv_writer.write_tests_row(tmp_path, workspace="rank1_csv", target_number=target_number, file_path=str(tmp_path / f"m{target_number}.py"),
                                   starter_result=result)
    with (tmp_path / "tests.csv").open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert [(row["starter_cache_hit_tokens"], row["starter_unknown_cache_requests"])
            for row in rows] == [("80", "1"), ("0", "")]
