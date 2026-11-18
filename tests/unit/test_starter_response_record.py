"""Starter suite generation keeps the model's reply beside the extracted code,
for every outcome, through the real ``run_starter_phase``. A reply without a
usable block, or with a wrongly picked fence, can then still be diagnosed.
"""
from __future__ import annotations

import hashlib
import json
from types import SimpleNamespace

import test_starter_phase as harness
from testuary import config
from testuary import models
from testuary import starters
from phase_values import serve_starter_settings
import pytest

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


def _drive(tmp_path, monkeypatch, replies, **cfg_over):
    """Run the real phase with a scripted reply per starter slot.

    ``replies`` maps a slot index to the raw text the model returns, so a
    test can script a reply that carries no fenced block at all, which the
    shared harness cannot express."""
    root, target, tests_dir, starters_dir = harness._project(tmp_path)
    # The boundary runs without a mutation engine: these tests cover what the
    # generation step records.
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: False)
    base = config.get_agent_config("test_initial")
    base.update({"suite_count": len(replies), "suite_run_timeout_seconds": 120,
                 "seed_base": 1000})
    base.update(cfg_over)
    serve_starter_settings(monkeypatch, base)
    monkeypatch.setattr(
        models, "get_api_config",
        lambda: {"model": "fake", "api_url": "", "api_key": "x"})

    def gen(prompt, *, temperature, seed, timeout):
        return replies[(seed - 1000) % starters.REFILL_SEED_STRIDE], {
            "input_tokens": 10, "output_tokens": 5}

    with models.installed(SimpleNamespace(generate_suite=gen)):
        res = starters.run_starter_phase(
            language="python", target_file_path=str(target),
            source_dir=str(root), test_output_dir=str(tests_dir),
            suites_dir=str(starters_dir), log=lambda *a, **k: None,
        )
    return res, starters_dir


def _slot_dir(starters_dir, index):
    return starters_dir / f"suite_{index:03d}"


def _meta(starters_dir, index):
    return json.loads((_slot_dir(starters_dir, index) / "original.json").read_text("utf-8"))


def test_the_reply_is_kept_beside_the_extracted_code(tmp_path, monkeypatch):
    """The ordinary case: the reply carried a fenced block, the block became
    original.py, and the reply itself is on disk with its hash recorded."""
    reply = harness._fenced(harness.GREEN_A)

    res, starters_dir = _drive(tmp_path, monkeypatch, {1: reply})

    assert res.n_generated == 1
    on_disk = (_slot_dir(starters_dir, 1) / "response.txt").read_text("utf-8")
    code = (_slot_dir(starters_dir, 1) / "original.py").read_text("utf-8")
    assert on_disk == reply
    # The reply is the whole message. The extraction is what was inside it.
    assert "```" in on_disk and "```" not in code
    assert "Here is the suite." in on_disk
    meta = _meta(starters_dir, 1)
    assert meta["extracted"] is True
    assert meta["response_chars"] == len(reply)
    assert meta["response_sha256"] == hashlib.sha256(
        reply.encode("utf-8")).hexdigest()
    # Two hashes of two different things, which must not be confused.
    assert meta["response_sha256"] != meta["source_sha256"]


def test_a_reply_with_no_code_block_is_still_kept(tmp_path, monkeypatch):
    """Nothing is extracted, so there is no original.py and no source hash.
    The kept reply explains the empty slot."""
    refusal = (
        "I cannot write tests for this module without seeing its imports.\n"
    )

    res, starters_dir = _drive(tmp_path, monkeypatch, {1: refusal})

    assert res.n_generated == 0
    assert not (_slot_dir(starters_dir, 1) / "original.py").exists()
    assert (_slot_dir(starters_dir, 1) / "response.txt").read_text("utf-8") == refusal
    meta = _meta(starters_dir, 1)
    assert meta["extracted"] is False
    assert "source_sha256" not in meta
    assert meta["response_chars"] == len(refusal)
    assert meta["response_sha256"] == hashlib.sha256(
        refusal.encode("utf-8")).hexdigest()


def test_every_slot_keeps_its_own_reply(tmp_path, monkeypatch):
    """Replies are not shared or overwritten between slots, so each starter
    directory explains its own starter suite."""
    replies = {
        1: harness._fenced(harness.GREEN_A),
        2: "no block here\n",
        3: harness._fenced(harness.GREEN_A) + "\ntrailing commentary\n",
    }

    _res, starters_dir = _drive(tmp_path, monkeypatch, replies)

    for index, expected in replies.items():
        assert (_slot_dir(starters_dir, index) / "response.txt").read_text(
            "utf-8") == expected
    assert _meta(starters_dir, 2)["extracted"] is False
    assert _meta(starters_dir, 3)["response_chars"] > _meta(
        starters_dir, 1)["response_chars"]


def _drive_attempts(tmp_path, monkeypatch, replies, **cfg_over):
    """Run the real phase on one slot whose successive attempts return
    ``replies`` in order (the last one repeats)."""
    root, target, tests_dir, starters_dir = harness._project(tmp_path)
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: False)
    base = config.get_agent_config("test_initial")
    base.update({"suite_count": 1, "suite_run_timeout_seconds": 120, "seed_base": 1000,
                 "refill_failed_suites": True, "attempts_per_suite": 4})
    base.update(cfg_over)
    serve_starter_settings(monkeypatch, base)
    monkeypatch.setattr(
        models, "get_api_config",
        lambda: {"model": "fake", "api_url": "", "api_key": "x"})
    calls = []

    def gen(prompt, *, temperature, seed, timeout):
        calls.append(seed)
        return replies[min(len(calls), len(replies)) - 1], {
            "input_tokens": 10, "output_tokens": 5}

    with models.installed(SimpleNamespace(generate_suite=gen)):
        res = starters.run_starter_phase(
            language="python", target_file_path=str(target),
            source_dir=str(root), test_output_dir=str(tests_dir),
            suites_dir=str(starters_dir), log=lambda *a, **k: None,
        )
    return res, starters_dir, calls


def test_every_rejected_attempt_of_a_slot_keeps_its_files(tmp_path, monkeypatch):
    """A refill keeps the rejected attempts: the slot directory holds the
    accepted attempt, and each earlier attempt keeps its reply, code and
    record under attempts/<n>/."""
    refusal = "I cannot write these tests.\n"
    broken = harness._fenced(harness.UNSALVAGEABLE)
    accepted = harness._fenced(harness.GREEN_A)

    res, starters_dir, calls = _drive_attempts(
        tmp_path, monkeypatch, [refusal, broken, accepted])

    assert res.status == "ok" and len(calls) == 3
    slot = _slot_dir(starters_dir, 1)
    # The slot directory: the accepted attempt.
    assert (slot / "response.txt").read_text("utf-8") == accepted
    assert (slot / "original.py").read_text("utf-8") == harness.GREEN_A
    assert (slot / "cleaned.py").is_file()
    assert _meta(starters_dir, 1)["slot_attempt"] == 3
    assert _meta(starters_dir, 1)["classification"] == "all_passing"
    # Attempt 1: a reply without code.
    first = slot / starters._ATTEMPTS_DIR / "1"
    assert (first / "response.txt").read_text("utf-8") == refusal
    assert not (first / "original.py").exists()
    first_meta = json.loads((first / "original.json").read_text("utf-8"))
    assert first_meta["slot_attempt"] == 1 and first_meta["extracted"] is False
    # Attempt 2: code that could not be repaired.
    second = slot / starters._ATTEMPTS_DIR / "2"
    assert (second / "response.txt").read_text("utf-8") == broken
    assert (second / "original.py").read_text("utf-8") == harness.UNSALVAGEABLE
    second_meta = json.loads((second / "original.json").read_text("utf-8"))
    assert second_meta["slot_attempt"] == 2
    assert second_meta["classification"] not in ("all_passing", "pruned")
    assert sorted(path.name for path in (slot / starters._ATTEMPTS_DIR).iterdir()) == ["1", "2"]
    # The manifest's attempts name the same numbers, and the archived code
    # is the code each attempt's record identifies.
    manifest = json.loads((starters_dir / starters._GENERATION_MANIFEST).read_text("utf-8"))
    attempts = manifest["suite_slots"]["slots"][0]["attempts"]
    assert [item["slot_attempt"] for item in attempts] == [1, 2, 3]
    assert attempts[1]["classification"] == second_meta["classification"]
    assert attempts[1]["original_source_sha256"] == hashlib.sha256(
        (second / "original.py").read_bytes()).hexdigest()


def test_the_last_attempt_of_a_slot_without_a_suite_is_in_the_slot_dir(tmp_path, monkeypatch):
    """A slot that spends all its attempts keeps its last attempt in the
    slot directory and every earlier one under attempts/."""
    broken = harness._fenced(harness.UNSALVAGEABLE)

    _res, starters_dir, calls = _drive_attempts(
        tmp_path, monkeypatch, [broken], attempts_per_suite=3)

    assert len(calls) == 3
    slot = _slot_dir(starters_dir, 1)
    assert _meta(starters_dir, 1)["slot_attempt"] == 3
    for number in (1, 2):
        meta = json.loads((slot / starters._ATTEMPTS_DIR / str(number) / "original.json").read_text("utf-8"))
        assert meta["slot_attempt"] == number
        assert (slot / starters._ATTEMPTS_DIR / str(number) / "response.txt").read_text("utf-8") == broken


def test_an_attempt_processed_again_moves_nothing(tmp_path):
    """The same attempt's files stay. A directory without an original.json
    holds a processing an interruption cut short, which is written again."""
    slot = tmp_path / "suite_001"
    slot.mkdir()
    (slot / "original.json").write_text(json.dumps({"slot_attempt": 2}), encoding="utf-8")
    (slot / "response.txt").write_text("reply", encoding="utf-8")
    assert starters._archive_earlier_attempt(slot, 2) is None
    assert sorted(path.name for path in slot.iterdir()) == ["original.json", "response.txt"]
    (slot / "original.json").unlink()
    assert starters._archive_earlier_attempt(slot, 3) is None
    assert sorted(path.name for path in slot.iterdir()) == ["response.txt"]


def test_a_record_without_an_attempt_number_moves_nothing(tmp_path):
    """A record that names no attempt is a processing an interruption cut
    short: its files stay, and the attempt writes them again."""
    slot = tmp_path / "suite_001"
    slot.mkdir()
    (slot / "original.json").write_text(json.dumps({"attempts": 1}), encoding="utf-8")
    (slot / "original.py").write_text("code", encoding="utf-8")
    (slot / "validity.json").write_text("{}", encoding="utf-8")
    assert starters._archive_earlier_attempt(slot, 2) is None
    assert sorted(path.name for path in slot.iterdir()) == [
        "original.json", "original.py", "validity.json"]


def test_a_removal_s_validity_derivations_move_with_their_attempt(tmp_path):
    """The validity derivations that a measurement's removal wrote in the
    slot directory belong to that attempt: they move under attempts/ with its
    other files when a later attempt is processed."""
    slot = tmp_path / "suite_001"
    slot.mkdir()
    (slot / "original.json").write_text(json.dumps({"slot_attempt": 1}), encoding="utf-8")
    (slot / "validity.json").write_text("{}", encoding="utf-8")
    (slot / "validity_derivations.json").write_text("{}", encoding="utf-8")
    moved = starters._archive_earlier_attempt(slot, 2)
    assert moved == slot / starters._ATTEMPTS_DIR / "1"
    assert sorted(path.name for path in moved.iterdir()) == [
        "original.json", "validity.json", "validity_derivations.json"]
    assert not (slot / "validity_derivations.json").exists()
