"""``starters.run_starter_phase``: the starter phase end to end against the
real Python backend (pytest subprocesses, per-test measurement, set cover,
collision merging). Only the model is a stand-in, an installed model source."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import threading
import time
from types import SimpleNamespace

import pytest

from per_suite_double import killing_tests, measured_nothing, per_suite_double
from validator_fixtures import patch_validator, unjudged
from starter_models import generate

from testuary.languages.generic import GenericBackend
from testuary import models as model_source
from testuary import starters
from testuary import starter_salvage
from phase_values import serve_starter_settings

# The code under test reads the run's settings: each test runs in one.
pytestmark = pytest.mark.usefixtures("default_run")


@pytest.fixture(autouse=True)
def _consolidation_mutation_off(monkeypatch):
    """Consolidation mutation is off: no engine starts against these
    synthetic fixtures, which would fail the target. The mutation-column
    tests opt back in with their own fake runner."""
    from testuary import config
    monkeypatch.setattr(
        config, "consolidation_measure_mutation", lambda: False,
    )


def test_output_cap_unset_everywhere_lets_the_backend_decide():
    # No phase or alias cap: generation sends no output token limit.
    assert starters._resolve_output_cap({}, {}) is None
    assert starters._resolve_output_cap(
        {"max_output_tokens": "0"}, {"max_output_tokens": None}) is None


def test_output_cap_alias_only_governs():
    assert starters._resolve_output_cap(
        {"max_output_tokens": "0"}, {"max_output_tokens": 32768}) == 32768


def test_output_cap_takes_the_smaller_of_phase_and_alias():
    assert starters._resolve_output_cap(
        {"max_output_tokens": "16384"}, {"max_output_tokens": 32768}) == 16384
    assert starters._resolve_output_cap(
        {"max_output_tokens": "49152"}, {"max_output_tokens": 32768}) == 32768


def test_output_cap_phase_only_is_a_floor():
    assert starters._resolve_output_cap(
        {"max_output_tokens": "24576"}, {}) == 24576


class _Chunk:
    """A minimal AIMessageChunk: addition concatenates content and keeps the
    latest metadata, the shape ``_default_generate`` streams over."""

    def __init__(self, content, usage_metadata=None, response_metadata=None):
        self.content = content
        self.usage_metadata = usage_metadata
        self.response_metadata = response_metadata or {}

    def __add__(self, other):
        return _Chunk(
            (self.content or "") + (other.content or ""),
            other.usage_metadata or self.usage_metadata,
            other.response_metadata or self.response_metadata,
        )


def _patch_chat_model(monkeypatch, chunks):
    class _Model:
        def stream(self, _prompt):
            return iter(chunks)

    from testuary import _chat_models
    monkeypatch.setattr(_chat_models, "build_chat_model", lambda **kw: _Model())


def test_default_generate_records_deepseek_cache_hit_and_miss(monkeypatch):
    # DeepSeek's raw prompt_cache_hit_tokens, the miss is the rest.
    final = _Chunk(
        "```python\ndef test_x():\n    assert True\n```",
        usage_metadata={"input_tokens": 1000, "output_tokens": 200},
        response_metadata={"token_usage": {
            "prompt_tokens": 1000, "completion_tokens": 200,
            "prompt_cache_hit_tokens": 768}},
    )
    _patch_chat_model(monkeypatch, [final])
    _text, usage = generate(
        {"model": "deepseek", "api_url": "https://api.deepseek.com",
         "api_key": "x"},
        "prompt", temperature=0.7, seed=1, timeout=30,
        max_output_tokens=2048)
    assert usage["input_tokens"] == 1000
    assert usage["cache_hit_tokens"] == 768
    assert usage["cache_miss_tokens"] == 232


def test_default_generate_reads_openai_cached_tokens_detail(monkeypatch):
    # OpenAI shape: usage_metadata.input_token_details.cache_read.
    final = _Chunk(
        "```python\ndef test_x():\n    assert True\n```",
        usage_metadata={"input_tokens": 500, "output_tokens": 50,
                        "input_token_details": {"cache_read": 320}},
    )
    _patch_chat_model(monkeypatch, [final])
    _text, usage = generate(
        {"model": "gpt", "api_url": "https://api.openai.com", "api_key": "x"},
        "prompt", temperature=0.7, seed=1, timeout=30,
        max_output_tokens=2048)
    assert usage["cache_hit_tokens"] == 320
    assert usage["cache_miss_tokens"] == 180


def test_default_generate_unreported_cache_split_stays_unknown(monkeypatch):
    # No cache split reported: hits and misses are unknown, not zero.
    final = _Chunk(
        "```python\ndef test_x():\n    assert True\n```",
        usage_metadata={"input_tokens": 400, "output_tokens": 40},
    )
    _patch_chat_model(monkeypatch, [final])
    _text, usage = generate(
        {"model": "vllm", "api_url": "http://localhost:8000/v1",
         "api_key": "x"},
        "prompt", temperature=0.7, seed=1, timeout=30,
        max_output_tokens=2048)
    assert usage["usage_known"] is True and usage["input_tokens"] == 400
    assert usage["cache_hit_tokens"] is None and usage["cache_miss_tokens"] is None


TARGET = """\
def add(a, b):
    return a + b


def sign(x):
    if x > 0:
        return 1
    if x < 0:
        return -1
    return 0


def clamp(v, lo, hi):
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v
"""


def _shifted_clock(monkeypatch):
    """starters' monotonic clock, moved forward at once by the seconds the
    returned function is given: a step's time without waiting it out.
    Every real second still counts."""
    lock, shift = threading.Lock(), [0.0]
    monkeypatch.setattr(starters, "time", SimpleNamespace(
        monotonic=lambda: time.monotonic() + shift[0]))

    def wait(seconds: float) -> None:
        with lock:
            shift[0] += seconds
    return wait


def _project(tmp_path: Path) -> tuple:
    root = tmp_path / "proj"
    root.mkdir()
    target = root / "calc.py"
    target.write_text(TARGET, encoding="utf-8")
    tests_dir = tmp_path / "tests_out"
    suites_dir = tmp_path / "suites"
    return root, target, tests_dir, suites_dir


def _fenced(code: str) -> str:
    return f"Here is the suite.\n\n```python\n{code}```\n"


def _run(
    root,
    target,
    tests_dir,
    suites_dir,
    suites,
    monkeypatch,
    *,
    matrices_dir=None,
    generate=None,
    log=None,
    **cfg_over,
):
    """Drive run_starter_phase with canned generations, or with ``generate``,
    as an installed model source, its log lines to ``log``."""
    from testuary import config
    from testuary import models

    base_cfg = config.get_agent_config("test_initial")
    base_cfg.update({"suite_count": len(suites), "suite_run_timeout_seconds": 120})
    base_cfg.update(cfg_over)
    serve_starter_settings(monkeypatch, base_cfg)
    monkeypatch.setattr(
        models, "get_api_config",
        lambda: {"model": "fake", "api_url": "", "api_key": "x"},
    )
    calls = {"n": 0}

    def gen(prompt, *, temperature, seed, timeout):
        idx = calls["n"]
        calls["n"] += 1
        if idx < len(suites):
            item = suites[idx]
        else:
            item = suites[-1]
        if isinstance(item, Exception):
            raise item
        return _fenced(item), {"input_tokens": 100, "output_tokens": 50}

    with model_source.installed(SimpleNamespace(generate_suite=generate or gen)):
        return starters.run_starter_phase(
            language="python",
            target_file_path=str(target),
            source_dir=str(root),
            test_output_dir=str(tests_dir),
            suites_dir=str(suites_dir),
            matrices_dir=(str(matrices_dir) if matrices_dir is not None else None),
            log=log or (lambda *a, **k: None),
        ), calls


def _run_per_seed(
    root,
    target,
    tests_dir,
    suites_dir,
    plan,
    monkeypatch,
    *,
    seed_base=1000,
    agent_config_fn=None,
    **cfg_over,
):
    """Drive run_starter_phase with a per-slot script of canned generations.

    ``plan`` maps a slot to its successive attempts' outcomes (a suite source
    or an Exception to raise), the last repeating, so ``{1: [UNSALVAGEABLE]}``
    never succeeds. ``record`` counts each slot's attempts and keeps its random
    seeds and prompts. A request's slot is ``(seed - seed_base)`` modulo the
    refill stride.
    """
    from testuary import config
    from testuary import models

    suite_count = cfg_over.pop("suite_count", len(plan))
    base_cfg = (agent_config_fn or config.get_agent_config)("test_initial")
    base_cfg.update({"suite_count": suite_count, "suite_run_timeout_seconds": 120,
                     "seed_base": seed_base})
    base_cfg.update(cfg_over)
    serve_starter_settings(monkeypatch, base_cfg)
    monkeypatch.setattr(
        models, "get_api_config",
        lambda: {"model": "fake", "api_url": "", "api_key": "x"})

    record = {"attempts": {}, "suites": {}, "prompts": {}}

    def gen(prompt, *, temperature, seed, timeout):
        i = (seed - seed_base) % starters.REFILL_SEED_STRIDE
        assert i in plan, f"no scripted outcome for seed {i}"
        spent = record["attempts"].get(i, 0)
        record["attempts"][i] = spent + 1
        record["suites"].setdefault(i, []).append(seed)
        record["prompts"].setdefault(i, []).append(prompt)
        script = plan[i]
        item = script[min(spent, len(script) - 1)]
        if isinstance(item, Exception):
            raise item
        return _fenced(item), {"input_tokens": 100, "output_tokens": 50}

    with model_source.installed(SimpleNamespace(generate_suite=gen)):
        res = starters.run_starter_phase(
            language="python",
            target_file_path=str(target),
            source_dir=str(root),
            test_output_dir=str(tests_dir),
            suites_dir=str(suites_dir),
            log=lambda *a, **k: None,
        )
    return res, record


# No salvage stage turns this into a running suite: its slot stays empty.
UNSALVAGEABLE = "def broken(:\n"

GREEN_A = """\
from calc import add, sign

def test_add():
    assert add(1, 2) == 3

def test_sign_positive():
    assert sign(5) == 1
"""

GREEN_B_WITH_FAILURE = """\
from calc import sign, clamp

def test_sign_negative():
    assert sign(-3) == -1

def test_sign_zero():
    assert sign(0) == 0

def test_wrong():
    assert clamp(5, 0, 3) == 5
"""

COLLIDER = """\
from calc import add, clamp

def test_add():
    assert add(0, 0) == 0

def test_clamp_low():
    assert clamp(-1, 0, 10) == 0

def test_clamp_high():
    assert clamp(99, 0, 10) == 10
"""

# Green but covers no target line: zero_coverage, not counted toward N.
ZERO_COVERAGE = """\
def test_arithmetic_only():
    assert 1 + 1 == 2
"""


@pytest.mark.parametrize("temperature", [None, 0.0, 0.7])
def test_generation_records_optional_temperature(tmp_path, monkeypatch, temperature):
    root, target, tests_dir, suites_dir = _project(tmp_path)
    result, calls = _run(
        root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch,
        suite_count=1, temperature=temperature,
    )
    assert calls["n"] == 1
    assert result.n_generated == 1
    assert result.requests_used == 1
    metadata = json.loads((suites_dir / "suite_001/original.json").read_text())
    assert metadata["temperature"] == temperature
    assert metadata["attempts"] == 1


def test_generation_progress_is_logged_per_suite(tmp_path, monkeypatch):
    """The wave is announced at its start and each generation is logged as
    it returns, so a slow model shows progress."""
    from testuary import config
    from testuary import models

    root, target, tests_dir, suites_dir = _project(tmp_path)
    base_cfg = config.get_agent_config("test_initial")
    base_cfg.update({"suite_count": 2, "suite_run_timeout_seconds": 120})
    serve_starter_settings(monkeypatch, base_cfg)
    monkeypatch.setattr(
        models, "get_api_config",
        lambda: {"model": "fake", "api_url": "", "api_key": "x"})

    suites = [GREEN_A, GREEN_B_WITH_FAILURE]
    calls = {"n": 0}

    def gen(prompt, *, temperature, seed, timeout):
        idx = calls["n"]
        calls["n"] += 1
        return _fenced(suites[min(idx, len(suites) - 1)]), {
            "input_tokens": 1, "output_tokens": 1}

    logs: list = []
    with model_source.installed(SimpleNamespace(generate_suite=gen)):
        starters.run_starter_phase(
            language="python",
            target_file_path=str(target), source_dir=str(root),
            test_output_dir=str(tests_dir), suites_dir=str(suites_dir),
            log=logs.append,
        )
    start = [m for m in logs if m.startswith("[starter] generating ")]
    assert start and "up to" in start[0], logs
    progress = [m for m in logs if m.startswith("[starter] generated ")]
    assert len(progress) == 2, progress
    assert any("starter suite 1" in m for m in progress)
    assert any("starter suite 2" in m for m in progress)
    assert all("out_tokens=" in m for m in progress)


def test_suite_stats_summary():
    from testuary.starters import _suite_stats
    assert _suite_stats([]) == {}
    assert _suite_stats([None, None]) == {}
    one = _suite_stats([85.0])
    assert one == {"mean": 85.0, "median": 85.0, "std": 0.0,
                   "min": 85.0, "max": 85.0}
    many = _suite_stats([80.0, 90.0, 100.0, None])  # None ignored
    assert many["mean"] == 90.0
    assert many["median"] == 90.0
    assert many["min"] == 80.0 and many["max"] == 100.0
    assert many["std"] == 10.0


def test_metric_distribution_formats_all_five_stats():
    from testuary.agent import _helpers as h
    line = h.format_metric_distribution(
        "line", {"mean": 86.0, "median": 87.2, "std": 5.1,
                 "min": 71.5, "max": 92.3})
    assert ("mean=86.00" in line and "median=87.20" in line
            and "std=5.10" in line and "min=71.50" in line
            and "max=92.30" in line)
    # An empty distribution renders n/a, not a missing line.
    assert h.format_metric_distribution("mutation", {}).split() == ["mutation", "n/a"]
    assert h.format_metric_distribution("mutation", None).endswith("n/a")


def test_log_metric_distribution_emits_all_three_metrics():
    from testuary.agent import _helpers as h
    lines: list = []
    h.log_metric_distribution(
        lines.append, "starter", 24,
        {"mean": 86.0, "median": 87.0, "std": 5.0, "min": 71.5, "max": 92.3},
        {"mean": 73.8, "median": 74.0, "std": 6.0, "min": 54.7, "max": 82.8},
        {"mean": 70.0, "median": 71.0, "std": 8.0, "min": 50.0, "max": 80.0},
    )
    body = "\n".join(lines)
    assert "per-suite distribution before merge (n=24)" in body
    assert "line" in body and "branch" in body and "mutation" in body
    # All three metric rows are present even when one would be empty.
    assert sum(m in body for m in ("line", "branch", "mutation")) == 3


def test_log_phase_metrics_shows_three_metrics_and_na():
    from testuary.agent import _helpers as h
    lines: list = []
    h.log_phase_metrics(lines.append, "coverage", line=100.0, branch=98.5,
                        mutation=None, tests=81, note="consolidated |")
    out = lines[0]
    assert "line=100.00" in out and "branch=98.50" in out
    assert "mutation=n/a" in out and "tests=81" in out
    assert out.startswith("[coverage] consolidated |")


def _run_with_events(tmp_path, monkeypatch, *, warmup: bool, suite_count: int):
    """Run the starter phase recording generation start and end events."""
    import threading
    import time as _t
    from testuary import config
    from testuary import models
    root, target, tests_dir, suites_dir = _project(tmp_path)
    base_cfg = config.get_agent_config("test_initial")
    base_cfg.update({"suite_count": suite_count, "suite_run_timeout_seconds": 120,
                     "concurrent_requests": 8, "cache_warmup": warmup})
    serve_starter_settings(monkeypatch, base_cfg)
    monkeypatch.setattr(
        models, "get_api_config",
        lambda: {"model": "fake", "api_url": "", "api_key": "x"})
    events: list = []
    lock = threading.Lock()

    def gen(prompt, *, temperature, seed, timeout):
        with lock:
            events.append(("start", seed))
        # End once another generation has started, or after 2 s: a concurrent
        # wave overlaps on a loaded machine, and a lone warmup call still ends.
        deadline = _t.monotonic() + 2.0
        while _t.monotonic() < deadline:
            with lock:
                if sum(1 for e in events if e[0] == "start") > 1:
                    break
            _t.sleep(0.005)
        with lock:
            events.append(("end", seed))
        return _fenced(GREEN_A), {"input_tokens": 1, "output_tokens": 1}

    with model_source.installed(SimpleNamespace(generate_suite=gen)):
        starters.run_starter_phase(
            language="python",
            target_file_path=str(target), source_dir=str(root),
            test_output_dir=str(tests_dir), suites_dir=str(suites_dir),
            log=lambda *_: None,
        )
    return events


def test_cache_warmup_runs_first_suite_alone(tmp_path, monkeypatch):
    """With warmup on, the first generation runs alone, filling the provider's
    prompt cache, before the concurrent wave."""
    events = _run_with_events(tmp_path, monkeypatch, warmup=True, suite_count=4)
    first_end = next(i for i, e in enumerate(events) if e[0] == "end")
    starts_before_first_end = [e for e in events[:first_end] if e[0] == "start"]
    assert len(starts_before_first_end) == 1, events


def test_no_warmup_starts_the_batch_concurrently(tmp_path, monkeypatch):
    """With warmup off, the whole wave starts concurrently."""
    events = _run_with_events(tmp_path, monkeypatch, warmup=False, suite_count=4)
    first_end = next(i for i, e in enumerate(events) if e[0] == "end")
    starts_before_first_end = [e for e in events[:first_end] if e[0] == "start"]
    assert len(starts_before_first_end) > 1, events


def test_full_pipeline_selects_merges_and_measures(tmp_path, monkeypatch):
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, calls = _run(
        root, target, tests_dir, suites_dir,
        [GREEN_A, GREEN_B_WITH_FAILURE, COLLIDER], monkeypatch,
    )
    assert res.status == "ok"
    assert res.n_generated == 3
    assert res.n_green == 3                      # B salvaged by strip
    assert res.baseline_file
    baseline = Path(res.baseline_file)
    assert baseline.exists()
    text = baseline.read_text(encoding="utf-8")
    assert "test_wrong" not in text              # stripped before pooling
    assert "test_clamp_low" in text              # COLLIDER survives merge
    # the colliding test_add was deduped or renamed, never duplicated
    assert text.count("def test_add()") <= 1
    assert res.baseline_line_coverage and res.baseline_line_coverage > 90
    assert res.baseline_branch_coverage is not None
    assert res.selected_tests > 0
    assert res.pool_tests >= res.selected_tests
    assert res.requests_used >= 3
    assert res.input_tokens >= 300
    # per-seed persistence
    assert (suites_dir / "suite_001" / "original.py").exists()
    assert (suites_dir / "suite_001" / "cleaned.json").exists()
    cleaned = json.loads(
        (suites_dir / "suite_001" / "cleaned.json").read_text("utf-8"))
    assert cleaned["schema_version"] == 2
    cleaned_source = (
        suites_dir / "suite_001" / "cleaned.py"
    ).read_text(encoding="utf-8")
    assert cleaned["source_sha256"] == hashlib.sha256(
        cleaned_source.encode("utf-8")
    ).hexdigest()
    assert cleaned["normalization_rule_version"] == "python-last-binding-v2"
    assert cleaned["target_total_statements"] >= 1
    assert (
        cleaned["target_covered_statements"]
        == len(cleaned["covered_statement_lines"])
    )
    assert (
        cleaned["target_covered_branches"]
        == len(cleaned["covered_branch_arcs"])
    )
    meta = json.loads(
        (suites_dir / "suite_002" / "original.json").read_text("utf-8"))
    assert meta["classification"] == "pruned"
    original_source = (
        suites_dir / "suite_002" / "original.py"
    ).read_text(encoding="utf-8")
    assert meta["source_sha256"] == hashlib.sha256(
        original_source.encode("utf-8")
    ).hexdigest()
    assert meta["normalization"]["rule_version"] == "python-last-binding-v2"
    # per-seed (pre-merge) coverage distribution is recorded for line + branch.
    for stats in (res.suite_line_stats, res.suite_branch_stats):
        assert set(stats) == {"mean", "median", "std", "min", "max"}
        assert stats["min"] <= stats["mean"] <= stats["max"]
    # The merged suite covers at least as much as the best single suite.
    assert res.baseline_line_coverage >= res.suite_line_stats["max"] - 1e-6


def test_zero_coverage_suite_is_reclassified_not_green(tmp_path, monkeypatch):
    """A green suite that covers no target line becomes zero_coverage and
    stays out of the green pool."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(
        root, target, tests_dir, suites_dir,
        [GREEN_A, ZERO_COVERAGE], monkeypatch, refill_failed_suites=False,
    )
    assert res.n_green == 1                       # only GREEN_A covers calc
    assert res.n_zero_coverage == 1               # the arithmetic-only suite
    meta = json.loads(
        (suites_dir / "suite_002" / "original.json").read_text("utf-8"))
    assert meta["classification"] == "zero_coverage"


#: Passes in a normal run. The stand-in coverage tool below cannot run it.
UNMEASURABLE = """\
from calc import add

def test_add_unmeasurable():
    assert add(2, 2) == 4
"""


def _alone(files, marker):
    """Whether a run is the own run of the starter suite holding ``marker``,
    staged alone in the measurement's scratch directory."""
    path = Path(files[0])
    return (path.parent.name.startswith("testuary_starter_")
            and marker in path.read_text(encoding="utf-8"))


def _coverage_tool_fails_on(monkeypatch, marker, *, alone=False):
    """The coverage tool's run of a suite holding ``marker`` fails, naming its
    test. With ``alone`` only the suite's own run in the measurement fails.
    Every other run is the backend's own."""
    from testuary import languages

    backend_type = type(languages.get_backend("python"))
    real = backend_type.run_test_files

    def run_test_files(self, paths, source_dir, *args, **kwargs):
        res = real(self, paths, source_dir, *args, **kwargs)
        texts = [Path(p).read_text(encoding="utf-8") for p in paths]
        failing = _alone(paths, marker) if alone else any(marker in t for t in texts)
        if kwargs.get("measure_coverage_of") and failing:
            from testuary.languages.generic import TestFailure
            return res._replace(ok=False, returncode=1, passed=0, failed=1,
                                failures=(TestFailure(marker, "failed", "fails under the tool", ""),),
                                line_coverage=None, branch_coverage=None)
        return res

    monkeypatch.setattr(backend_type, "run_test_files", run_test_files)


def test_a_suite_the_coverage_tool_cannot_measure_frees_its_slot(tmp_path, monkeypatch):
    """A starter suite that passes but that the coverage tool cannot measure
    has no complete numbers: its slot is generated again within its attempts,
    and the record names the failing test."""
    _coverage_tool_fails_on(monkeypatch, "test_add_unmeasurable")
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(
        root, target, tests_dir, suites_dir, [UNMEASURABLE, GREEN_A], monkeypatch,
        refill_failed_suites=True, attempts_per_suite=2,
    )
    assert res.n_unmeasured_coverage == 1
    assert res.n_green == 2


def test_without_attempts_left_an_unmeasurable_suite_leaves_its_slot_empty(tmp_path, monkeypatch):
    _coverage_tool_fails_on(monkeypatch, "test_add_unmeasurable")
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(
        root, target, tests_dir, suites_dir, [GREEN_A, UNMEASURABLE], monkeypatch,
        refill_failed_suites=False,
    )
    assert res.n_green == 1 and res.n_unmeasured_coverage == 1
    meta = json.loads((suites_dir / "suite_002" / "original.json").read_text("utf-8"))
    assert meta["classification"] == "coverage_unmeasured"
    assert "test_add_unmeasurable" in meta["cleanup"]["disposition_reason"]


def test_the_starter_model_time_is_the_time_a_request_is_in_flight(tmp_path, monkeypatch):
    """The requests run in parallel, so the phase's model time is the time
    some request is in flight: not the generation's span, which also holds
    the checks of each answer."""
    from testuary import phase_state

    model = []
    real_record = phase_state.record_time

    def record(part, seconds):
        if part == phase_state.TIME_MODEL:
            model.append(seconds)
        real_record(part, seconds)

    monkeypatch.setattr(phase_state, "record_time", record)
    wait = _shifted_clock(monkeypatch)
    real_clean = starter_salvage._clean_suite

    def slow_checks(*args, **kwargs):
        wait(1.0)
        return real_clean(*args, **kwargs)

    monkeypatch.setattr(starter_salvage, "_clean_suite", slow_checks)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A, GREEN_A], monkeypatch,
                  refill_failed_suites=False)
    assert res.status == "ok"
    assert res.generation_seconds >= 2.0
    assert model and sum(model) < 1.0


def test_a_refill_that_ends_on_a_request_failure_adds_only_its_own_time(tmp_path, monkeypatch):
    """A suite the measurement frees is generated again, and that request
    fails terminally. The generation's model time is recorded once, and the
    failure adds only the refill's: the measurement in between is neither
    model time nor generation time."""
    from testuary import phase_state

    model = []
    real_record = phase_state.record_time

    def record(part, seconds):
        if part == phase_state.TIME_MODEL:
            model.append(seconds)
        real_record(part, seconds)

    monkeypatch.setattr(phase_state, "record_time", record)
    _no_coverage_evidence_of(monkeypatch, "test_add_unmeasurable", failed=False)
    _coverage_tool_fails_on(monkeypatch, "test_add_unmeasurable", alone=True)
    replies = iter([UNMEASURABLE, GREEN_A, ValueError("bad request")])
    wait = _shifted_clock(monkeypatch)

    def generate(prompt, *, temperature, seed, timeout):
        wait(0.5)
        reply = next(replies)
        if isinstance(reply, Exception):
            raise reply
        return _fenced(reply), {"input_tokens": 100, "output_tokens": 50}

    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A, GREEN_A], monkeypatch,
                  generate=generate, refill_failed_suites=True, attempts_per_suite=2)
    assert res.status == "failed_model_request"
    assert res.generation_seconds < res.duration_s
    assert len(model) == 2 and model[1] < model[0] + 0.25


def test_a_phase_end_without_mutation_gives_its_suites_no_method(tmp_path, monkeypatch):
    """The phase end measures no mutation, whatever suite_mutation asks for:
    no method applies, and each starter suite's reason says so, naming no
    derivation and no failed measurement."""
    from testuary import consolidation as consolidate_mod

    reasons = []
    real_pick = consolidate_mod.pick_phase_result

    def capture(candidates, **kwargs):
        reasons.extend(c.status_reason for c in candidates if c.label.startswith("suite_"))
        return real_pick(candidates, **kwargs)

    monkeypatch.setattr(consolidate_mod, "pick_phase_result", capture)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [_ALPHA, _BETA], monkeypatch,
                  suite_mutation="derive")
    assert res.suite_mutation_method == ""
    assert reasons and set(reasons) == {"no mutation score: the starter phase end measures no mutation"}


def test_zero_green_cancels_the_target(tmp_path, monkeypatch):
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(
        root, target, tests_dir, suites_dir,
        ["}\n" * 60, "import nope_mod_xyz\nRAISE AT IMPORT"],
        monkeypatch,
    )
    assert res.status == "cancelled_zero_passing"
    assert res.baseline_file is None
    assert res.n_discarded_syntax >= 1
    assert "no green tests" in res.detail


def _run_empty_replies(root, target, tests_dir, suites_dir, monkeypatch, *, output_tokens):
    """Every reply is empty after ``output_tokens``, with a limit of 8192."""
    from testuary import config
    from testuary import models

    base_cfg = config.get_agent_config("test_initial")
    base_cfg.update({"suite_count": 2, "suite_run_timeout_seconds": 120,
                     "max_output_tokens": 8192, "attempts_per_suite": 1})
    serve_starter_settings(monkeypatch, base_cfg)
    monkeypatch.setattr(models, "get_api_config",
                        lambda: {"model": "fake", "api_url": "", "api_key": "x"})

    def gen(prompt, *, temperature, seed, timeout):
        return "", {"input_tokens": 100, "output_tokens": output_tokens}

    with model_source.installed(SimpleNamespace(generate_suite=gen)):
        return starters.run_starter_phase(
            language="python", target_file_path=str(target),
            source_dir=str(root), test_output_dir=str(tests_dir), suites_dir=str(suites_dir),
            matrices_dir=None, log=lambda *a, **k: None,
        )


def test_replies_that_spend_the_whole_output_limit_say_so(tmp_path, monkeypatch):
    """Empty replies that spend max_output_tokens name the limit in the reason."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res = _run_empty_replies(root, target, tests_dir, suites_dir, monkeypatch,
                             output_tokens=8192)
    assert res.status == "cancelled_zero_passing"
    assert res.detail.startswith(
        "the model used the whole output limit (max_output_tokens = 8192) without "
        "answering in 2 of 2 replies. Raise max_output_tokens. No green tests from 0")


def test_an_empty_reply_below_the_output_limit_is_not_called_a_limit(tmp_path, monkeypatch):
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res = _run_empty_replies(root, target, tests_dir, suites_dir, monkeypatch,
                             output_tokens=100)
    assert res.status == "cancelled_zero_passing"
    assert res.detail.startswith("no green tests from 0")


def test_exact_mode_regenerates_until_n(tmp_path, monkeypatch):
    root, target, tests_dir, suites_dir = _project(tmp_path)
    # One of two first-wave suites is broken: exact mode tops up to 2 green.
    res, calls = _run(
        root, target, tests_dir, suites_dir,
        ["def broken(:\n", GREEN_A, COLLIDER], monkeypatch,
        suite_count=2, refill_failed_suites=True,
    )
    assert res.status == "ok"
    assert res.n_green == 2
    assert calls["n"] >= 3                       # the top-up happened
    assert res.requests_used >= 3                # failures counted


def test_exact_mode_rerolls_in_place_at_the_same_indices(tmp_path, monkeypatch):
    """A failed slot is regenerated at its own index with its own random seed,
    so the green pool stays within slots 1..suite_count, as in the zero-shot
    ablation. No ``suite_003`` appears for suite_count=2."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, calls = _run(
        root, target, tests_dir, suites_dir,
        ["def broken(:\n", GREEN_A, COLLIDER], monkeypatch,
        suite_count=2, refill_failed_suites=True,
    )
    assert res.status == "ok" and res.n_green == 2
    suites = Path(suites_dir)
    assert (suites / "suite_001").is_dir() and (suites / "suite_002").is_dir()
    assert not (suites / "suite_003").exists(), "re-roll must reuse the index"
    # Slot 1's second seed is one stride above its first, next to slot 2's.
    m1 = json.loads((suites / "suite_001" / "original.json").read_text())
    m2 = json.loads((suites / "suite_002" / "original.json").read_text())
    assert m1["seed"] - starters.REFILL_SEED_STRIDE == m2["seed"] - 1


def test_exact_mode_per_suite_limit_bounds_regeneration(tmp_path, monkeypatch):
    root, target, tests_dir, suites_dir = _project(tmp_path)
    # Every generation is unusable: each slot spends its 2 attempts and stops.
    res, calls = _run(
        root, target, tests_dir, suites_dir,
        [UNSALVAGEABLE], monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=2,
    )
    assert res.status == "cancelled_zero_passing"
    assert res.n_green == 0
    assert calls["n"] == 4                        # 2 slots x 2 attempts each
    assert res.suite_generation_attempts == {1: 2, 2: 2}


def test_suite_that_fails_twice_is_accepted_on_its_third_attempt(
    tmp_path, monkeypatch,
):
    """A slot whose first two generations are unusable enters the pool with
    its third."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, record = _run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [UNSALVAGEABLE, UNSALVAGEABLE, GREEN_A]}, monkeypatch,
        suite_count=1, refill_failed_suites=True,
    )
    assert record["attempts"] == {1: 3}
    assert res.status == "ok"
    assert res.n_green == 1
    assert res.suite_generation_attempts == {1: 3}
    assert res.baseline_file and Path(res.baseline_file).exists()


def test_a_replacement_generation_repeats_the_independent_request(
    tmp_path, monkeypatch,
):
    """A rejected starter suite is replaced by the same request, with no word
    of the rejected attempt, and a new random seed per attempt, so an
    endpoint that honors seeds does not repeat the rejected answer."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, record = _run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [UNSALVAGEABLE, UNSALVAGEABLE, GREEN_A]}, monkeypatch,
        suite_count=1, refill_failed_suites=True, attempts_per_suite=4,
    )
    assert res.status == "ok" and record["attempts"] == {1: 3}
    first, *replacements = record["prompts"][1]
    assert all(prompt == first for prompt in replacements)
    assert record["suites"][1] == [1001, 1001001, 2001001]


def test_only_the_last_generation_attempt_gets_progress_cleanup(
    tmp_path, monkeypatch,
):
    root, target, tests_dir, suites_dir = _project(tmp_path)
    real_clean_suite = starter_salvage._clean_suite
    final_attempt_flags = []

    def clean_on_last_attempt(*args, final_attempt=False, **kwargs):
        final_attempt_flags.append(final_attempt)
        if not final_attempt:
            return "unsalvageable", None
        return real_clean_suite(
            *args, final_attempt=final_attempt, **kwargs,
        )

    monkeypatch.setattr(starter_salvage, "_clean_suite", clean_on_last_attempt)
    res, record = _run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [GREEN_A]}, monkeypatch,
        suite_count=1, refill_failed_suites=True,
        attempts_per_suite=4,
    )

    assert record["attempts"] == {1: 4}
    assert final_attempt_flags == [False, False, False, True]
    assert res.status == "ok"
    assert res.n_green == 1


def test_a_slot_failing_all_its_attempts_leaves_the_others_theirs(
    tmp_path, monkeypatch,
):
    """A slot that fails all its attempts is left out and the phase delivers
    the shorter pool. The other slot still gets all its own attempts."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, record = _run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [UNSALVAGEABLE], 2: [UNSALVAGEABLE, UNSALVAGEABLE, GREEN_A]},
        monkeypatch, suite_count=2, refill_failed_suites=True,
    )
    assert record["attempts"] == {1: 4, 2: 3}
    # The exhausted slot delivers nothing, and its last attempt's
    # classification stays on record.
    meta = json.loads(
        (Path(suites_dir) / "suite_001" / "original.json").read_text("utf-8"))
    assert meta["classification"] == "discarded_collection"
    assert not (Path(suites_dir) / "suite_001" / "cleaned.py").exists()
    assert res.status == "ok"
    assert res.n_green == 1
    assert (Path(suites_dir) / "suite_002" / "cleaned.py").exists()


def test_exhausting_one_slot_does_not_reduce_another_slots_attempts(
    tmp_path, monkeypatch,
):
    """Two slots spending all their attempts leave the third slot its own
    attempts, so it turns green on its third. An attempt limit shared across slots
    would be spent before then."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, record = _run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [UNSALVAGEABLE], 2: [UNSALVAGEABLE],
         3: [UNSALVAGEABLE, UNSALVAGEABLE, GREEN_A]},
        monkeypatch, suite_count=3, refill_failed_suites=True,
    )
    assert record["attempts"] == {1: 4, 2: 4, 3: 3}
    assert res.suite_generation_attempts == {1: 4, 2: 4, 3: 3}
    assert res.status == "ok"
    assert res.n_green == 1


def test_generation_seed_of_a_slot_is_unmoved_by_other_slots_retries(
    tmp_path, monkeypatch,
):
    """An attempt's random seed depends only on its slot and attempt number:
    slot 2 draws the same seed once whether slot 1 spends one attempt or
    three."""
    from testuary import config

    real_gac = config.get_agent_config
    root, target, tests_dir, suites_dir = _project(tmp_path)

    _res_a, record_a = _run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [UNSALVAGEABLE], 2: [GREEN_A]}, monkeypatch,
        agent_config_fn=real_gac, suite_count=2, refill_failed_suites=True,
        attempts_per_suite=1,
    )
    _res_b, record_b = _run_per_seed(
        root, target, tmp_path / "tests_out_b", tmp_path / "seeds_b",
        {1: [UNSALVAGEABLE], 2: [GREEN_A]}, monkeypatch,
        agent_config_fn=real_gac, suite_count=2, refill_failed_suites=True,
        attempts_per_suite=3,
    )

    assert record_a["suites"] == {1: [1001], 2: [1002]}
    # Slot 1 re-rolls twice more, each attempt with its own seed.
    assert record_b["suites"][1] == [1001, 1001001, 2001001]
    # Slot 2 is untouched: same seed, same single draw.
    assert record_b["suites"][2] == record_a["suites"][2] == [1002]


def test_generation_manifest_records_attempts_per_suite(tmp_path, monkeypatch):
    """Each slot's attempt count is recorded in the generation manifest."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _record = _run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [UNSALVAGEABLE, GREEN_A], 2: [COLLIDER]}, monkeypatch,
        suite_count=2, refill_failed_suites=True,
    )
    assert res.status == "ok"
    manifest = json.loads(
        (Path(suites_dir) / starters._GENERATION_MANIFEST).read_text("utf-8"))
    assert manifest["generation_attempts_per_suite"] == {"1": 2, "2": 1}


def test_reuse_restores_the_recorded_attempts_per_suite(tmp_path):
    """A reused pool restores the archived attempt counts, so a later refill
    extends the record instead of replacing it."""
    suites = tmp_path / "initial_seeds"
    d = suites / "suite_001"
    d.mkdir(parents=True)
    (d / "cleaned.py").write_text(
        "def test_x():\n    assert True\n", encoding="utf-8")
    (d / "original.py").write_text("x", encoding="utf-8")
    (d / "original.json").write_text(
        json.dumps({"attempts": 1, "input_tokens": 100, "output_tokens": 20,
                    "classification": "pruned", "extracted": True}),
        encoding="utf-8")
    (suites / starters._GENERATION_MANIFEST).write_text(
        json.dumps({"requests_used": 4, "input_tokens": 300,
                    "output_tokens": 55, "cache_hit_tokens": 0,
                    "generation_seconds": 42.0, "n_generated": 2,
                    "generation_attempts_per_suite": {"1": 3, "2": 1}}),
        encoding="utf-8")
    res = starters.StarterResult()
    starters._load_persisted_green(suites, {}, res, lambda *a, **k: None)
    assert res.suite_generation_attempts == {1: 3, 2: 1}


def test_a_generation_exception_that_is_not_transport_ends_the_phase(
    tmp_path, monkeypatch,
):
    """A generation exception that is not a transport error ends the phase
    instead of being retried. Transport failures are tested in
    ``test_starter_transport_failures``."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, calls = _run(
        root, target, tests_dir, suites_dir,
        [RuntimeError("sever"), GREEN_A], monkeypatch,
        cache_warmup=False, concurrent_requests=1,
    )
    assert res.status == "failed_model_request"
    assert "RuntimeError: sever" in res.detail
    assert calls["n"] == 1


def test_relative_paths_resolve_before_subprocess_use(tmp_path, monkeypatch):
    """The phase resolves relative paths before running tests from the
    source directory, where the failing-ids run would otherwise miss the suite."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    monkeypatch.chdir(tmp_path)
    from testuary import config
    from testuary import models

    base_cfg = config.get_agent_config("test_initial")
    base_cfg.update({"suite_count": 1, "suite_run_timeout_seconds": 120})
    serve_starter_settings(monkeypatch, base_cfg)
    monkeypatch.setattr(
        models, "get_api_config",
        lambda: {"model": "fake", "api_url": "", "api_key": "x"})
    source = SimpleNamespace(generate_suite=lambda prompt, *, temperature, seed, timeout: (
        _fenced(GREEN_A), {"input_tokens": 1, "output_tokens": 1}))
    with model_source.installed(source):
        res = starters.run_starter_phase(
            language="python",
            target_file_path="proj/calc.py",
            source_dir="proj",
            test_output_dir="tests_out",
            suites_dir="suites",
            log=lambda *a, **k: None,
        )
    assert res.status == "ok"
    assert res.n_green == 1
    assert Path(res.baseline_file).is_absolute()


def test_clean_suite_repairs_mid_file_glitch(tmp_path):
    """A suite with one mid-file defect keeps its healthy tests: syntax repair
    makes it parse and the run removes what the repair broke."""
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    src = ("from calc import add, sign\n\n"
           "def test_add():\n    assert add(1, 2) == 3\n\n"
           "def glitched():\n\n"
           "def test_sign():\n    assert sign(-1) == -1\n")
    cls, green = starter_salvage._clean_suite(
        backend, src, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda m: None,
    )
    assert cls in ("all_passing", "pruned")
    assert "test_add" in green and "test_sign" in green


def test_clean_suite_normalizes_shadowed_class_method_before_collection(
        tmp_path):
    """The last method is the executable binding and no refill is needed."""
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    src = (
        "from calc import add\n\n"
        "class TestValues:\n"
        "    def test_add(self):\n"
        "        assert add(1, 1) == 999\n\n"
        "    def test_add(self):\n"
        "        assert add(1, 1) == 2\n"
    )
    record = {}
    cls, green = starter_salvage._clean_suite(
        backend, src, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None,
        normalization_out=record,
    )

    assert cls == "all_passing"
    assert green is not None
    assert green.count("def test_add") == 1
    assert "== 999" not in green
    assert record["changed"] is True
    assert record["removed_bindings"][0]["scope"] == ("TestValues",)


def test_reused_suite_is_normalized_before_revalidation(tmp_path):
    from testuary import languages

    suites = tmp_path / "suites"
    seed = suites / "suite_001"
    seed.mkdir(parents=True)
    source = (
        "def test_value():\n    assert 1\n\n"
        "def test_value():\n    assert 2\n"
    )
    (seed / "cleaned.py").write_bytes(source.encode("utf-8"))
    (suites / starters._GENERATION_MANIFEST).write_text('{"generation_complete": true}', encoding="utf-8")
    result = starters.StarterResult()
    green = {}

    starters._load_persisted_green(
        suites, green, result, lambda _message: None,
        backend=languages.get_backend("python"),
    )

    assert green[1].count("def test_value") == 1
    assert "assert 1" not in green[1]
    assert (seed / "cleaned.py").read_bytes() == green[1].encode("utf-8")


def test_clean_suite_finalizes_source_before_acceptance(tmp_path):
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    source = (
        "from calc import add\n\n"
        "def test_add():\n    assert add(1, 1) == 2\n\n"
        "# detached comment from a removed test\n"
    )

    classification, green = starter_salvage._clean_suite(
        backend, source, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None,
    )

    assert classification == "all_passing"
    assert green is not None
    assert "detached comment" not in green
    assert backend.finalize_test_source(green) == green


def test_clean_suite_executes_only_finalized_pruned_sources(
    tmp_path, monkeypatch,
):
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    source = (
        "from calc import add\n\n"
        "# comment attached to the failing test\n"
        "def test_bad():\n    assert add(1, 1) == 99\n\n"
        "def test_good():\n    assert add(1, 1) == 2\n"
    )
    observed = []
    backend_type = type(backend)
    real_failing = backend_type.failing_test_ids

    def observe(self, paths, source_dir, **kwargs):
        for path in paths:
            staged = Path(path).read_text(encoding="utf-8")
            observed.append(staged)
            assert self.finalize_test_source(staged) == staged
        return real_failing(self, paths, source_dir, **kwargs)

    monkeypatch.setattr(backend_type, "failing_test_ids", observe)
    classification, green = starter_salvage._clean_suite(
        backend, source, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None,
    )

    assert classification == "pruned"
    assert green is not None and "test_bad" not in green
    assert len(observed) >= 2


_TWO_FAILING = (
    "def test_first():\n    assert False\n\n"
    "def test_second():\n    assert False\n\n"
    "def test_good():\n    assert True\n"
)


def _clean_with(tmp_path, monkeypatch, failing, *, validate=None):
    """Clean _TWO_FAILING with the failing ids the scripted runs report."""
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    reports = iter(failing)
    monkeypatch.setattr(type(backend), "failing_test_ids",
                        lambda self, paths, source_dir, **kwargs: next(reports))
    if validate is not None:
        real = type(backend).validate_test_source
        monkeypatch.setattr(type(backend), "validate_test_source",
                            lambda self, source, root_: validate(real(self, source, root_), source))
    cleanup = {}
    classification, _green = starter_salvage._clean_suite(
        backend, _TWO_FAILING, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None, cleanup_out=cleanup,
    )
    return classification, cleanup


def test_a_cleanup_that_removes_no_further_test_says_so(tmp_path, monkeypatch):
    classification, cleanup = _clean_with(tmp_path, monkeypatch,
                                          [["test_first"], ["test_first"]])
    assert classification == "unsalvageable"
    assert cleanup["disposition_reason"] == "failing-test cleanup removed no further test"
    assert cleanup["disposition_basis"] == "unavailable"


def test_a_suite_that_fails_validation_after_cleanup_says_so(tmp_path, monkeypatch):
    def validate(result, source):
        return result if "def test_first" in source else SimpleNamespace(
            ok=False, message="invalid", status="judged")

    classification, cleanup = _clean_with(tmp_path, monkeypatch, [["test_first"]],
                                          validate=validate)
    assert classification == "unsalvageable"
    assert cleanup["disposition_reason"] == "the suite failed validation after cleanup round 1"


def test_a_suite_without_a_test_outcome_after_cleanup_says_so(tmp_path, monkeypatch):
    classification, cleanup = _clean_with(tmp_path, monkeypatch, [["test_first"], None])
    assert classification == "unsalvageable"
    # The scripted run reports no outcome, so nothing says why.
    assert cleanup["disposition_reason"] == "the suite run gave no test outcome after cleanup round 1"


def test_clean_suite_ordinary_attempt_stops_after_three_failure_rounds(
    tmp_path, monkeypatch,
):
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    source = (
        "def test_first():\n    assert False\n\n"
        "def test_second():\n    assert False\n\n"
        "def test_third():\n    assert False\n\n"
        "def test_fourth():\n    assert False\n\n"
        "def test_good():\n    assert True\n"
    )
    failures = iter([
        ["test_first"], ["test_second"], ["test_third"], ["test_fourth"],
    ])
    monkeypatch.setattr(
        type(backend), "failing_test_ids",
        lambda self, paths, source_dir, **kwargs: next(failures),
    )

    cleanup = {}
    classification, green = starter_salvage._clean_suite(
        backend, source, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None, cleanup_out=cleanup,
    )

    assert classification == "unsalvageable"
    assert green is None
    # No completed run judged the suite, so the basis stays unavailable and
    # the reason names the round bound.
    assert cleanup == {
        "disposition_basis": "unavailable",
        "disposition_reason": "failing tests remained after 3 cleanup rounds",
        "configured_failing_test_cleanup_rounds": 3,
        "final_generation_attempt": False,
        "failing_test_cleanup_rounds_used": 3,
        "warning_cleanup_rounds_used": 0,
    }


@pytest.mark.parametrize("rounds", [0, -1])
def test_clean_suite_rejects_nonpositive_failure_cleanup_bound(
    tmp_path, rounds,
):
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    with pytest.raises(ValueError, match="positive integer"):
        starter_salvage._clean_suite(
            backend,
            "def test_ok():\n    assert True\n",
            str(target),
            str(root),
            tmp_path / "scratch",
            timeout=120,
            log=lambda _message: None,
            failing_cleanup_rounds=rounds,
        )


def test_clean_suite_final_attempt_continues_while_failures_are_removed(
    tmp_path, monkeypatch,
):
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    source = (
        "def test_first():\n    assert False\n\n"
        "def test_second():\n    assert False\n\n"
        "def test_third():\n    assert False\n\n"
        "def test_good():\n    assert True\n"
    )
    failures = iter([
        ["test_first"], ["test_second"], ["test_third"], [],
    ])
    monkeypatch.setattr(
        type(backend), "failing_test_ids",
        lambda self, paths, source_dir, **kwargs: next(failures),
    )
    monkeypatch.setattr(
        type(backend), "run_test_files",
        lambda self, paths, source_dir, **kwargs: SimpleNamespace(
            ok=True, timed_out=False, passed=1, failed=0, errors=0, output="",
        ),
    )

    cleanup = {}
    classification, green = starter_salvage._clean_suite(
        backend, source, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None, final_attempt=True,
        failing_cleanup_rounds=1,
        cleanup_out=cleanup,
    )

    assert classification == "pruned"
    assert green is not None
    assert "test_first" not in green
    assert "test_second" not in green
    assert "test_third" not in green
    assert "test_good" in green
    assert cleanup["final_generation_attempt"] is True
    assert cleanup["failing_test_cleanup_rounds_used"] == 3


def test_clean_suite_prunes_direct_invalid_test_before_execution(
    tmp_path,
) -> None:
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    source = (
        "import inspect\n"
        "import calc\n\n"
        "def test_invalid_source_oracle():\n"
        "    assert 'return' in inspect.getsource(calc.add)\n\n"
        "def test_behavior():\n"
        "    assert calc.add(1, 2) == 3\n"
    )
    record = {}
    classification, green = starter_salvage._clean_suite(
        backend, source, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None, validity_out=record,
    )
    # A suite that lost a test to the validity policy is stripped.
    assert classification == "pruned"
    assert green is not None
    assert "test_invalid_source_oracle" not in green
    assert "test_behavior" in green
    assert record["removed_tests"] == ["test_invalid_source_oracle"]
    assert record["findings"][0]["rule_id"] == "target_source_inspection"


def test_clean_suite_rejects_shared_invalid_helper(tmp_path) -> None:
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    source = (
        "import inspect\n"
        "import calc\n\n"
        "SOURCE = inspect.getsource(calc.add)\n\n"
        "def test_source():\n"
        "    assert SOURCE\n"
    )
    classification, green = starter_salvage._clean_suite(
        backend, source, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None,
    )
    assert classification == "invalid_generated_test"
    assert green is None


def test_clean_suite_final_attempt_stops_when_pruning_makes_no_progress(
    tmp_path, monkeypatch,
):
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    source = "def test_good():\n    assert True\n"
    monkeypatch.setattr(
        type(backend), "failing_test_ids",
        lambda self, paths, source_dir, **kwargs: ["missing_test"],
    )

    classification, green = starter_salvage._clean_suite(
        backend, source, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None, final_attempt=True,
    )

    assert classification == "unsalvageable"
    assert green is None


def test_clean_suite_final_attempt_rejects_an_empty_subset(
    tmp_path, monkeypatch,
):
    from testuary import languages
    from testuary.languages import DeliveredRun

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    source = "def test_bad():\n    assert False\n"
    failures = iter([["test_bad"], []])
    monkeypatch.setattr(
        type(backend), "failing_test_ids",
        lambda self, paths, source_dir, **kwargs: next(failures),
    )
    monkeypatch.setattr(
        type(backend), "run_as_delivered",
        lambda self, paths, source_dir, **kwargs: DeliveredRun(
            cases=(), failing=(), reason=None, output=""),
    )

    cleanup = {}
    classification, green = starter_salvage._clean_suite(
        backend, source, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None, final_attempt=True,
        cleanup_out=cleanup,
    )

    # A completed run that passed nothing confirms an empty suite.
    assert classification == "unsalvageable"
    assert green is None
    assert cleanup["disposition_basis"] == "judged_rejection"


def test_clean_suite_final_attempt_preserves_timeout_classification(
    tmp_path, monkeypatch,
):
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    source = (
        "def test_bad():\n    assert False\n\n"
        "def test_good():\n    assert True\n"
    )
    # The cleanup round's run is reported timed out.
    runs = iter([(["test_bad"], "test_failures"), (None, "timed_out")])

    def failing_test_ids(self, paths, source_dir, *, report=None, **kwargs):
        ids, outcome = next(runs)
        if report is not None:
            report["outcome"] = outcome
        return ids

    monkeypatch.setattr(type(backend), "failing_test_ids", failing_test_ids)

    classification, green = starter_salvage._clean_suite(
        backend, source, str(target), str(root), tmp_path / "scratch",
        timeout=1, log=lambda _message: None, final_attempt=True,
    )

    assert classification == "timeout"
    assert green is None


def test_reused_suite_is_finalized_and_target_validated(tmp_path):
    from testuary import languages

    root, _, _, _ = _project(tmp_path)
    suites = tmp_path / "suites"
    seed = suites / "suite_001"
    seed.mkdir(parents=True)
    source = (
        "def test_value():\n    assert True\n\n"
        "# detached comment from an old cleanup\n"
    )
    (seed / "cleaned.py").write_bytes(source.encode("utf-8"))
    (suites / starters._GENERATION_MANIFEST).write_text('{"generation_complete": true}', encoding="utf-8")
    result = starters.StarterResult()
    green = {}

    starters._load_persisted_green(
        suites, green, result, lambda _message: None,
        backend=languages.get_backend("python"), source_dir=str(root),
    )

    assert "detached comment" not in green[1]
    assert (seed / "cleaned.py").read_bytes() == green[1].encode("utf-8")


def test_clean_suite_drops_collection_breaker_then_greens(tmp_path):
    """Module-scope code that raises at load is pruned, the test that needs it
    is removed at run time and the rest is green."""
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    src = ("from calc import add\n\n"
           "BROKEN = add(1)\n\n"
           "def test_uses_broken():\n    assert BROKEN == 2\n\n"
           "def test_fine():\n    assert add(1, 1) == 2\n")
    cls, green = starter_salvage._clean_suite(
        backend, src, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda m: None,
    )
    assert cls == "pruned"
    assert "test_fine" in green
    assert "test_uses_broken" not in green


def test_clean_suite_logs_real_collection_error(tmp_path):
    """A suite pytest cannot collect logs the actual exception, so a failed
    salvage is diagnosable."""
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    src = ("import testuary_no_such_module_abc as m\n\n"
           "def test_uses():\n    assert m.f() == 1\n")
    logs = []
    starter_salvage._clean_suite(
        backend, src, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=logs.append,
    )
    assert any("collection failed" in m and "testuary_no_such_module_abc" in m
               for m in logs), logs


@pytest.mark.parametrize("operation", ["prune_unresolvable_imports", "prune_collection_breakers"])
def test_clean_suite_does_not_discard_code_when_a_recovery_check_cannot_run(
        tmp_path, monkeypatch, operation):
    from langadapter.python.adapter import PythonAdapter as ProjectPythonBackend
    from langadapter.sources import INFRASTRUCTURE_FAILURE, SourceResult, UNCHANGED
    from testuary.languages.generic import GenericBackend

    root, target, _, _ = _project(tmp_path)
    backend = GenericBackend("python", (".py",))
    monkeypatch.setattr(backend, "failing_test_ids", lambda *a, **kw: None)
    monkeypatch.setattr(backend, "describe_collection_failure", lambda *a, **kw: "collection failed")

    def no_import_removal(self, source):
        return SourceResult(UNCHANGED, source, "target")

    def failed(self, source, **options):
        return SourceResult(INFRASTRUCTURE_FAILURE, source, "target", detail="check could not run")

    if operation == "prune_collection_breakers":
        monkeypatch.setattr(ProjectPythonBackend, "prune_unresolvable_imports", no_import_removal)
    monkeypatch.setattr(ProjectPythonBackend, operation, failed)
    cleanup = {}
    classification, source = starter_salvage._clean_suite(
        backend, "from calc import add\ndef test_a(): assert add(1, 1) == 2\n",
        str(target), str(root), tmp_path / "scratch", timeout=60,
        log=lambda _: None, cleanup_out=cleanup)
    assert classification == "validation_unresolved" and source is None
    assert "check could not run" in cleanup["validation_unresolved"]


def test_strip_module_level_skips_removes_module_scope_only():
    """The backend drops module-scope ``importorskip`` and
    ``skip(allow_module_level=True)`` but keeps a per-test importorskip."""
    from testuary import languages

    backend = languages.get_backend("python")
    src = ("import pytest\n"
           "da = pytest.importorskip('dask.array', minversion='0.0')\n"
           "pytest.importorskip('foo')\n"
           "pytest.skip('gone', allow_module_level=True)\n"
           "def test_x():\n"
           "    d = pytest.importorskip('bar')\n"
           "    assert 1\n")
    out, n = backend.strip_module_level_skips(src)
    assert n == 3
    assert "da = pytest.importorskip" not in out
    assert "pytest.importorskip('foo')" not in out
    assert "allow_module_level" not in out
    assert "pytest.importorskip('bar')" in out          # per-test one kept
    assert "def test_x" in out


def test_clean_suite_salvages_module_level_importorskip(tmp_path):
    """A module-level importorskip of an absent package would skip the whole
    file, and in the merged suite every test. Salvage removes it so the tests
    that do not need the package survive."""
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    src = ("import pytest\n"
           "from calc import add\n\n"
           "dask = pytest.importorskip('nonexistent_pkg_xyz', minversion='0.0')\n\n"
           "def test_fine():\n    assert add(1, 1) == 2\n")
    cls, green = starter_salvage._clean_suite(
        backend, src, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda m: None,
    )
    assert cls in ("all_passing", "pruned")
    assert green is not None and "test_fine" in green
    assert "importorskip" not in green


def test_clean_suite_rejects_zero_test_suite(tmp_path):
    """A suite green only because every test is skipped is unsalvageable, so
    exact mode regenerates the slot."""
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    src = ("import pytest\n\n"
           "@pytest.mark.skip(reason='x')\n"
           "def test_a():\n    assert False\n")
    cls, green = starter_salvage._clean_suite(
        backend, src, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda m: None,
    )
    assert cls == "unsalvageable"
    assert green is None


def test_measurement_pytest_isolated_from_ambient_filterwarnings(tmp_path):
    """An ancestor checkout's pytest settings stay out of measurements. A
    target without its own uses defaults anchored at its suite directory, and
    its own settings, when present, apply."""
    from testuary import languages

    # The ancestor checkout is strict.
    (tmp_path / "pyproject.toml").write_text(
        "[tool.pytest.ini_options]\n"
        "filterwarnings = [\"error\"]\n"
        "addopts = \"--strict-markers\"\n",
        encoding="utf-8")
    # The target project has no settings and is the measurement's root.
    proj = tmp_path / "target"
    proj.mkdir()
    wz = tmp_path / "results" / "workzone"
    wz.mkdir(parents=True)
    seed = wz / "test_seed.py"
    seed.write_text(
        "import warnings\n"
        "warnings.warn('pkg_resources is deprecated', DeprecationWarning)\n"
        "def test_ok():\n    assert True\n",
        encoding="utf-8")
    backend = languages.get_backend("python")
    res = backend.run_test_files([str(seed)], str(proj), timeout=120)
    assert res.ok, res.output
    assert res.passed == 1


def test_clean_suite_strips_unsound_warning_tests(tmp_path):
    """A passing test that leaks an unawaited coroutine is removed."""
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    src = ("from calc import add\n\n"
           "async def helper():\n    return add(1, 1)\n\n"
           "def test_leaky():\n"
           "    helper()\n"
           "    assert True\n\n"
           "def test_clean():\n    assert add(2, 2) == 4\n")
    cls, green = starter_salvage._clean_suite(
        backend, src, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda m: None,
    )
    assert cls == "pruned"
    assert "test_clean" in green
    assert "def test_leaky" not in green


def test_clean_suite_leaves_an_unconfirmed_empty_warning_cleanup_unresolved(
    tmp_path, monkeypatch,
):
    """When warning cleanup removes the only test and the adapter reports the
    empty run as unsuccessful without a typed empty-suite observation, the
    confirmation stays unresolved instead of a judged rejection. The confirmed
    empty case is test_clean_suite_final_attempt_rejects_an_empty_subset."""
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    monkeypatch.setattr(
        type(backend), "failing_test_ids",
        lambda self, paths, source_dir, *, timeout, instrument_target=None,
        report=None, delivered_seed=None: [],
    )
    src = (
        "async def helper():\n    return 1\n\n"
        "def test_leaky():\n"
        "    helper()\n"
        "    assert True\n"
    )
    cleanup = {}
    cls, green = starter_salvage._clean_suite(
        backend, src, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda _message: None, cleanup_out=cleanup,
    )
    assert cls == "validation_unresolved"
    assert green is None
    assert cleanup["warning_cleanup_rounds_used"] == 1
    assert cleanup["unresolved_stage"] == "final_suite_confirmation"
    assert cleanup["disposition_basis"] == "unavailable"


def test_clean_suite_keeps_unsound_when_reject_off(tmp_path):
    """With ``reject_unsound=False`` the starter phase keeps the unsound test,
    as the editing phases do under that setting."""
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")
    src = ("from calc import add\n\n"
           "async def helper():\n    return add(1, 1)\n\n"
           "def test_leaky():\n"
           "    helper()\n"
           "    assert True\n\n"
           "def test_clean():\n    assert add(2, 2) == 4\n")
    cls, green = starter_salvage._clean_suite(
        backend, src, str(target), str(root), tmp_path / "scratch",
        timeout=120, log=lambda m: None, reject_unsound=False,
    )
    assert cls == "all_passing"
    assert "def test_leaky" in green and "test_clean" in green


def test_clean_suite_timeout_classified_as_timeout(tmp_path, monkeypatch):
    """A run its runner reports timed out is a timeout, not a collection failure."""
    from testuary import languages

    root, target, _, _ = _project(tmp_path)
    backend = languages.get_backend("python")

    def timed_out_ids(self, paths, source_dir, *, timeout, instrument_target=None,
                      report=None, delivered_seed=None):
        if report is not None:
            report["outcome"] = "timed_out"
        return None

    monkeypatch.setattr(type(backend), "failing_test_ids", timed_out_ids)
    cls, green = starter_salvage._clean_suite(
        backend, "def test_a():\n    assert 1\n",
        str(target), str(root), tmp_path / "scratch",
        timeout=1, log=lambda m: None,
    )
    assert cls == "timeout"
    assert green is None


def test_import_hint_python_dotted_path(tmp_path):
    """A package that does not bind the submodule's name gets the dotted
    import of the module."""
    root = tmp_path / "proj"
    (root / "pkg" / "sub").mkdir(parents=True)
    f = root / "pkg" / "sub" / "mod.py"
    f.write_text("x = 1\n", encoding="utf-8")
    hint = starters._import_hint("python", str(f), str(root))
    assert hint == "import pkg.sub.mod as mod"


def _hint_for(tmp_path, init_text):
    """The hint for pkg/mod.py whose package __init__ holds init_text."""
    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    if init_text is not None:
        (root / "pkg" / "__init__.py").write_text(init_text, encoding="utf-8")
    target = root / "pkg" / "mod.py"
    target.write_text("def f():\n    return 1\n", encoding="utf-8")
    return starters._import_hint("python", str(target), str(root))


SAFE = 'import importlib\nmod = importlib.import_module("pkg.mod")'


@pytest.mark.parametrize("init_text", [
    "def mod(f):\n    return f\n",                   # tenacity's retry decorator
    "class mod:\n    pass\n",
    "mod = 1\n",
    "from .mod import mod\n",
    "from .other import thing as mod\n",
    "import json as mod\n",
    "try:\n    from .mod import mod\nexcept ImportError:\n    pass\n",
])
def test_a_package_that_binds_the_submodules_name_gets_the_module_by_importlib(
        tmp_path, init_text):
    """``import pkg.mod as mod`` takes ``mod`` from the package, so it would
    bind the package's member. The hint binds the module under the same name."""
    assert _hint_for(tmp_path, init_text) == SAFE


@pytest.mark.parametrize("init_text", [
    None,                                              # a namespace package
    "",
    "from .mod import f\n",
    "from . import mod\n",                             # binds the submodule itself
    "from pkg import mod\n",
    "VALUE = 1\n",
])
def test_a_package_that_does_not_bind_the_name_keeps_the_dotted_import(tmp_path, init_text):
    assert _hint_for(tmp_path, init_text) == "import pkg.mod as mod"


@pytest.mark.parametrize("init_text", [
    "from .exports import *\n",
    "def __getattr__(name):\n    raise AttributeError(name)\n",
    "def broken(:\n",
])
def test_an_undecidable_package_gets_the_module_by_importlib(tmp_path, init_text):
    """A star import or a module __getattr__ may bind the name, and an
    unreadable __init__ cannot be checked: the module-safe form is given."""
    assert _hint_for(tmp_path, init_text) == SAFE


def test_the_importlib_hint_binds_the_module_when_the_package_shadows_it(tmp_path, monkeypatch):
    """The hint run as code binds the module object, where the dotted import
    binds the package's function."""
    import importlib
    import sys
    import types

    hint = _hint_for(tmp_path, "from .mod import f as mod\n")
    monkeypatch.syspath_prepend(str(tmp_path / "proj"))
    for name in ("pkg", "pkg.mod"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    shadowed, bound = {}, {}
    exec("import pkg.mod as mod", shadowed)
    exec(hint, bound)
    assert not isinstance(shadowed["mod"], types.ModuleType)
    assert isinstance(bound["mod"], types.ModuleType) and bound["mod"].f() == 1
    importlib.invalidate_caches()


def test_import_hint_top_level_module(tmp_path):
    root = tmp_path / "proj"
    root.mkdir()
    f = root / "solo.py"
    f.write_text("x = 1\n", encoding="utf-8")
    assert starters._import_hint("python", str(f), str(root)) == "import solo"


def test_display_path_is_project_relative_posix(tmp_path):
    """The file shown to the model is project-relative with forward
    slashes, identical on every platform."""
    from testuary import project

    root = tmp_path / "proj"
    (root / "pkg").mkdir(parents=True)
    f = root / "pkg" / "mod.py"
    f.write_text("x = 1\n", encoding="utf-8")
    assert project.display_path(str(f), str(root)) == "pkg/mod.py"


def test_display_path_falls_back_to_name_outside_root(tmp_path):
    """A file outside the source directory is shown by its bare name."""
    from testuary import project

    root = tmp_path / "proj"
    root.mkdir()
    other = tmp_path / "elsewhere" / "mod.py"
    other.parent.mkdir(parents=True)
    other.write_text("x = 1\n", encoding="utf-8")
    assert project.display_path(str(other), str(root)) == "mod.py"


def test_prompt_shows_relative_target_not_absolute_path(tmp_path, monkeypatch):
    """The 'Target file:' line names the file relative to the project, and the
    absolute path never reaches the model."""
    from testuary import config
    from testuary import models

    root, target, tests_dir, suites_dir = _project(tmp_path)
    base_cfg = config.get_agent_config("test_initial")
    base_cfg.update({"suite_count": 1, "suite_run_timeout_seconds": 120})
    serve_starter_settings(monkeypatch, base_cfg)
    monkeypatch.setattr(
        models, "get_api_config",
        lambda: {"model": "fake", "api_url": "", "api_key": "x"})

    captured = {}

    def gen(prompt, *, temperature, seed, timeout):
        captured["prompt"] = prompt
        return _fenced(GREEN_A), {"input_tokens": 1, "output_tokens": 1}

    with model_source.installed(SimpleNamespace(generate_suite=gen)):
        starters.run_starter_phase(
            language="python",
            target_file_path=str(target), source_dir=str(root),
            test_output_dir=str(tests_dir), suites_dir=str(suites_dir),
            log=lambda *a, **k: None,
        )
    messages = captured["prompt"]
    assert [message["role"] for message in messages] == ["system", "user"]
    assert "Target file: calc.py" in messages[1]["content"]
    assert str(target) not in "\n".join(
        message["content"] for message in messages
    )


def test_measure_mutation_keeps_redundant_killer_in_baseline(
    tmp_path, monkeypatch,
):
    """With consolidation_measure_mutation on, a coverage-redundant test that
    alone kills a mutant stays in the delivered suite."""
    from contextlib import contextmanager
    from testuary import config
    from testuary import measurement as test_metrics

    root, target, tests_dir, suites_dir = _project(tmp_path)
    twin_a = "from calc import add\n\ndef test_alpha():\n    assert add(1, 1) == 2\n"
    twin_b = "from calc import add\n\ndef test_beta():\n    assert add(1, 1) == 2\n"

    @contextmanager
    def fake_runner(**kw):
        def _measure(d):
            # The killer is named by the file that holds the test, per-suite or
            # merged, so map each twin's test to the file found here.
            killer_file = {}
            for f in Path(d).glob("*.py"):
                body = f.read_text(encoding="utf-8")
                if "def test_alpha" in body:
                    killer_file["test_alpha"] = f.name
                if "def test_beta" in body:
                    killer_file["test_beta"] = f.name
            # A suite measured alone holds one twin, the other's mutant survives.
            return {"mutants": [
                {"id": mid, "operator": "core/Op", "line": line, "diff": mid,
                 **({"status": "killed", "killer": f"{killer_file[test]}::{test}"}
                    if test in killer_file else {"status": "survived"})}
                for mid, test, line in (("m1", "test_alpha", 1), ("m2", "test_beta", 2))
            ]}
        yield per_suite_double(_measure)

    monkeypatch.setattr(test_metrics, "union_mutation_runner", fake_runner)
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: True)
    res, _ = _run(
        root, target, tests_dir, suites_dir, [twin_a, twin_b], monkeypatch,
    )
    assert res.status == "ok"
    text = Path(res.baseline_file).read_text(encoding="utf-8")
    # Coverage-identical, but each kills a distinct mutant: both stay.
    assert "test_alpha" in text and "test_beta" in text


def test_pool_engine_cannot_start_fails_the_target(tmp_path, monkeypatch):
    """With consolidation_measure_mutation on, an engine that cannot start
    fails the target instead of reducing on coverage alone."""
    from contextlib import contextmanager
    from testuary import config
    from testuary import measurement as test_metrics

    root, target, tests_dir, suites_dir = _project(tmp_path)
    seed = "from calc import add\n\ndef test_a():\n    assert add(1, 1) == 2\n"

    @contextmanager
    def none_runner(**kw):
        yield None

    monkeypatch.setattr(test_metrics, "union_mutation_runner", none_runner)
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: True)
    with pytest.raises(RuntimeError, match="kill evidence"):
        _run(root, target, tests_dir, suites_dir, [seed], monkeypatch)


def test_each_starter_suite_is_measured_alone_as_accepted(
    tmp_path, monkeypatch,
):
    """Starter suites that would interfere as sibling modules are each staged
    alone, as accepted, and measured per suite: nothing is pruned and the
    record counts each slot's own verdicts. Per-test kill evidence uses the
    pool's mutant ids, so it compares across suites."""
    from contextlib import contextmanager
    from testuary import config
    from testuary import consolidation as consolidate_mod
    from testuary import measurement as test_metrics

    root, target, tests_dir, suites_dir = _project(tmp_path)
    writer, checker = _WRITER, _CHECKER
    staged = []

    @contextmanager
    def fake_runner(**kw):
        def _measure(d):
            files = sorted(Path(d).glob("*.py"))
            bodies = {f.name: f.read_text(encoding="utf-8") for f in files}
            staged.append(bodies)
            mutants = []
            for mid, test, line in (("m1", "test_alpha", 2), ("m2", "test_no_marker", 3)):
                holder = next((name for name, body in bodies.items() if f"def {test}" in body), None)
                row = {"id": f"{Path(d).name}-{mid}", "operator": "core/Op", "line": line, "diff": mid}
                if holder is None:
                    row["status"] = "survived"
                else:
                    row.update(status="killed", killer=f"{holder}::{test}")
                mutants.append(row)
            return {"mutants": mutants}
        yield per_suite_double(_measure)

    monkeypatch.setattr(test_metrics, "union_mutation_runner", fake_runner)
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: True)
    collision_evidence = []
    real_consolidate = consolidate_mod.consolidate_phase_end

    def capture_evidence(context, candidates, mutation_runner=None, options=None):
        collision_evidence.extend(options.candidate_evidence)
        return real_consolidate(context, candidates, mutation_runner, options)

    monkeypatch.setattr(consolidate_mod, "consolidate_phase_end", capture_evidence)
    # The two suites interfere once merged, so the merge is no candidate
    # and the suites tie. A fixed time breaks the tie by name.
    monkeypatch.setattr(
        consolidate_mod,
        "benchmark_tied_candidates",
        lambda *args, **kwargs: lambda candidate: 1.0,
    )
    matrices = tmp_path / "matrices"
    res, _ = _run(
        root, target, tests_dir, suites_dir, [writer, checker], monkeypatch,
        matrices_dir=matrices,
    )
    assert res.status == "ok"
    # Each suite was staged alone and as accepted: the observer test stays.
    alone = [bodies for bodies in staged if len(bodies) == 1]
    assert any("def test_no_marker" in body for bodies in alone for body in bodies.values())
    # Each suite kills one of the two mutants alone: 50 percent each.
    assert res.suite_mutation_stats["mean"] == 50.0
    assert res.suite_mutation_stats["min"] == res.suite_mutation_stats["max"] == 50.0
    attribution = json.loads(
        (matrices / "starter_pool_attribution.json").read_text(encoding="utf-8"))
    assert (attribution["schema"], attribution["basis"], attribution["implementation"]) == (
        "testuary.starter_pool_attribution/3", "per_suite_verdicts", "test_double")
    assert [(slot["slot"], slot["measured"], slot["valid"], slot["killed"])
            for slot in attribution["slots"]] == [(1, True, 2, 1), (2, True, 2, 1)]
    assert all(slot["verdicts"]["killed"] == 1 and slot["verdicts"]["survived"] == 1
               for slot in attribution["slots"])
    measured_kills = {
        test.rsplit(".", 1)[-1]: {t for t in tokens if t.startswith("K:")}
        for suite in collision_evidence for test, tokens in suite.items()
    }
    union_ids = {row["diff"]: row["id"] for row in json.loads(
        (matrices / "starter_pool_kill_matrix.json").read_text(encoding="utf-8"))["mutants"]}
    # The merging step's collision evidence holds kill tokens.
    assert measured_kills == {"test_alpha": {f"K:{union_ids['m1']}"}, "test_beta": set(),
                              "test_no_marker": {f"K:{union_ids['m2']}"}}
    # Collision evidence is a copy, not a new per-suite measurement.
    for cleaned in suites_dir.glob("suite_*/cleaned.json"):
        saved = json.loads(cleaned.read_text(encoding="utf-8"))
        assert not any(t.startswith("K:") for t in saved["tokens"])


def _mutation_tool_measures_nothing_of(monkeypatch, marker):
    """The stand-in mutation tool measures nothing of a starter suite holding
    ``marker`` when it runs that suite alone. Every other suite kills one of
    two mutants."""
    from contextlib import contextmanager
    from testuary import config
    from testuary import consolidation as consolidate_mod
    from testuary import measurement as test_metrics

    @contextmanager
    def fake_runner(**kw):
        def _measure(d):
            bodies = [f.read_text(encoding="utf-8") for f in Path(d).glob("*.py")]
            if Path(d).name.startswith("testuary_starter") and any(marker in b for b in bodies):
                return measured_nothing("infrastructure_failure")
            return {"mutants": [{"id": "m1", "operator": "core/Op", "line": 2, "diff": "m1",
                                 "status": "killed", "killer": "x.py::test_alpha"},
                                {"id": "m2", "operator": "core/Op", "line": 3, "diff": "m2",
                                 "status": "survived"}]}
        yield per_suite_double(_measure)

    monkeypatch.setattr(test_metrics, "union_mutation_runner", fake_runner)
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: True)
    monkeypatch.setattr(consolidate_mod, "benchmark_tied_candidates",
                        lambda *args, **kwargs: lambda candidate: 1.0)


def _suite_slots(suites_dir):
    from testuary import generation_evidence

    manifest = json.loads((suites_dir / "_generation.json").read_text(encoding="utf-8"))
    return generation_evidence.read_suite_slots(manifest, suites_dir)["slots"]


def test_a_starter_suite_the_mutation_tool_measured_nothing_of_frees_its_slot(
    tmp_path, monkeypatch,
):
    """A suite whose per-suite run measured nothing has no complete numbers:
    its slot is generated again within its attempts and the suites are
    measured again, so every accepted suite has a result."""
    _mutation_tool_measures_nothing_of(monkeypatch, "test_beta")
    root, target, tests_dir, suites_dir = _project(tmp_path)
    matrices = tmp_path / "matrices"
    res, calls = _run(root, target, tests_dir, suites_dir, [_ALPHA, _BETA, _GAMMA], monkeypatch,
                      matrices_dir=matrices, suite_count=2, refill_failed_suites=True,
                      attempts_per_suite=2)
    assert res.status == "ok"
    assert calls["n"] == 3
    assert (res.n_green, res.n_unmeasured_mutation) == (2, 1)
    attribution = json.loads((matrices / "starter_pool_attribution.json").read_text(encoding="utf-8"))
    assert [(slot["accepted"], slot["measured"]) for slot in attribution["slots"]] == [(True, True)] * 2
    slot = _suite_slots(suites_dir)[1]
    freed, accepted = slot["attempts"]
    assert slot["state"] == "accepted" and accepted["disposition_basis"] == "accepted_cleaned"
    assert (freed["classification"], freed["disposition_basis"], freed["accepted_source"]) == (
        "mutation_unmeasured", "unavailable", None)
    earlier = json.loads((suites_dir / "suite_002" / "attempts" / "1" / "original.json").read_text("utf-8"))
    assert earlier["classification"] == "mutation_unmeasured"


def test_without_attempts_left_a_suite_measured_nothing_of_leaves_its_slot_empty(
    tmp_path, monkeypatch,
):
    """Without attempts left the slot stays empty, and the other suites are
    measured again without it: the slot has no counts, and the record names
    the run's outcome."""
    _mutation_tool_measures_nothing_of(monkeypatch, "test_beta")
    root, target, tests_dir, suites_dir = _project(tmp_path)
    matrices = tmp_path / "matrices"
    res, calls = _run(root, target, tests_dir, suites_dir, [_ALPHA, _BETA], monkeypatch,
                      matrices_dir=matrices, refill_failed_suites=False)
    assert res.status == "ok"
    assert calls["n"] == 2
    assert (res.n_green, res.n_unmeasured_mutation) == (1, 1)
    assert res.suite_mutation_stats["mean"] == res.suite_mutation_stats["max"] == 50.0
    attribution = json.loads((matrices / "starter_pool_attribution.json").read_text(encoding="utf-8"))
    measured, freed = attribution["slots"]
    assert (measured["measured"], measured["killed"], measured["outcome"]) == (True, 1, "completed")
    assert freed == {"slot": 2, "accepted": False, "measured": False, "killed": None}
    meta = json.loads((suites_dir / "suite_002" / "original.json").read_text(encoding="utf-8"))
    assert meta["classification"] == "mutation_unmeasured"
    assert "infrastructure_failure" in meta["cleanup"]["disposition_reason"]
    assert _suite_slots(suites_dir)[1]["state"] == "unavailable"


def test_a_reuse_does_not_restore_a_suite_freed_at_measurement(tmp_path, monkeypatch):
    _mutation_tool_measures_nothing_of(monkeypatch, "test_beta")
    root, target, tests_dir, suites_dir = _project(tmp_path)
    _run(root, target, tests_dir, suites_dir, [_ALPHA, _BETA], monkeypatch,
         refill_failed_suites=False)
    res, calls = _run(root, target, tmp_path / "tests_out2", suites_dir, [_ALPHA, _BETA],
                      monkeypatch, reuse_suites=True, refill_failed_suites=False)
    assert calls["n"] == 0
    assert (res.status, res.n_green, res.n_unmeasured_mutation) == ("ok", 1, 1)


def _no_coverage_evidence_of(monkeypatch, marker, *, failed=True):
    """The coverage tool gives no per-test evidence for a starter suite
    holding ``marker`` when it runs that suite alone: because a test failed
    under it, or, without ``failed``, because the backend gives none."""
    from testuary.languages import get_backend

    backend_type = type(get_backend("python"))
    collect = backend_type.collect_eviction_data

    def no_evidence(self, files, *args, **kwargs):
        if _alone(files, marker):
            if failed:
                kwargs["failure_fn"]("test_failure", f"{marker} failed under the coverage tool")
            return None
        return collect(self, files, *args, **kwargs)

    monkeypatch.setattr(backend_type, "collect_eviction_data", no_evidence)


def test_a_starter_suite_without_coverage_evidence_frees_its_slot(tmp_path, monkeypatch):
    """A suite the coverage tool cannot measure alone for its per-test
    coverage has no complete numbers: its slot is generated again instead of
    the suite joining the merged suite."""
    _no_coverage_evidence_of(monkeypatch, "test_beta")
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, calls = _run(root, target, tests_dir, suites_dir, [_ALPHA, _BETA, _GAMMA], monkeypatch,
                      suite_count=2, refill_failed_suites=True, attempts_per_suite=2)
    assert res.status == "ok"
    assert calls["n"] == 3
    assert (res.n_green, res.n_unmeasured_coverage) == (2, 1)
    freed = _suite_slots(suites_dir)[1]["attempts"][0]
    assert freed["classification"] == "coverage_unmeasured"
    assert freed["reason"].endswith("test_beta failed under the coverage tool")


def test_a_starter_suite_whose_report_timed_no_test_still_ranks(tmp_path, monkeypatch):
    """Per-test evidence whose report gave no test a time has no suite time.
    The starter suite still becomes a candidate, ranked as unmeasured."""
    import dataclasses

    from testuary.languages import get_backend

    backend_type = type(get_backend("python"))
    collect = backend_type.collect_eviction_data

    def untimed(self, *args, **kwargs):
        data = collect(self, *args, **kwargs)
        if data is None:
            return None
        return dataclasses.replace(data, per_test_time_ms={}, suite_time_ms=None)

    monkeypatch.setattr(backend_type, "collect_eviction_data", untimed)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    first = "from calc import add\n\ndef test_alpha():\n    assert add(1, 1) == 2\n"
    res, _ = _run(root, target, tests_dir, suites_dir, [first], monkeypatch)
    assert res.status == "ok"
    saved = json.loads((suites_dir / "suite_001" / "cleaned.json").read_text(encoding="utf-8"))
    assert saved["suite_ms"] is None


_ALPHA = "from calc import add\n\ndef test_alpha():\n    assert add(1, 1) == 2\n"
_BETA = "from calc import add\n\ndef test_beta():\n    assert add(2, 2) == 4\n"
_GAMMA = "from calc import add\n\ndef test_gamma():\n    assert add(3, 3) == 6\n"
#: Two starter suites that pass alone and interfere when run together: the
#: writer leaves a file beside it that the checker's second test forbids.
_WRITER = (
    "import os\n"
    "from calc import add\n"
    "open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"
    " 'marker.tmp'), 'w').write('x')\n\n"
    "def test_alpha():\n    assert add(1, 1) == 2\n"
)
_CHECKER = (
    "import os\n"
    "from calc import add\n\n"
    "def test_beta():\n    assert add(2, 2) == 4\n\n"
    "def test_no_marker():\n"
    "    assert not os.path.exists(os.path.join("
    "os.path.dirname(os.path.abspath(__file__)), 'marker.tmp'))\n"
)


def _two_mutant_runner(pooled_calls, per_suite_calls, *, pooled_unnamed_kill, attributions=None,
                       pooled_first_killers=False, pooled_nothing=False):
    """A runner where test_alpha kills m1 and test_beta kills m2. With
    ``pooled_unnamed_kill``, the pooled run kills m2 with no test named, as
    a collection error would. With ``pooled_first_killers``, the pooled
    run records only each kill's first killing test. With
    ``pooled_nothing``, the pooled run gives no result."""
    from contextlib import contextmanager

    def _measure(d):
        bodies = {f.name: f.read_text(encoding="utf-8") for f in Path(d).glob("*.py")}
        pooled = Path(d).name.startswith("testuary_initkills_")
        if pooled:
            pooled_calls.append(sorted(name for name in bodies if name.startswith("test_")))
            if pooled_nothing:
                return None
        rows = []
        for mid, test, line in (("m1", "test_alpha", 2), ("m2", "test_beta", 3)):
            holder = next((name for name, body in bodies.items() if f"def {test}" in body), None)
            row = {"id": mid, "operator": "core/Op", "line": line, "diff": mid}
            if pooled and pooled_unnamed_kill and mid == "m2":
                row.update(status="killed")
            elif holder is None:
                row["status"] = "survived"
            else:
                row.update(status="killed", killer=f"{holder}::{test}")
            rows.append(row)
        if pooled and pooled_first_killers:
            return {"mutants": rows, "attribution": "first"}
        return {"mutants": rows}

    @contextmanager
    def fake_runner(**kw):
        run = per_suite_double(_measure)
        if attributions is not None:
            run.attributions = attributions
        per_suite = run.per_suite

        def counted(suites):
            per_suite_calls.append(sorted(suites))
            return per_suite(suites)

        run.per_suite = counted
        yield run

    return fake_runner


def test_a_backend_without_per_test_evidence_scores_each_starter_suite_on_its_coverage_run(
    tmp_path, monkeypatch,
):
    """A backend that declares no per-test evidence gives none: each
    starter suite stays a candidate with its coverage, scored on its own
    coverage run."""
    from testuary import consolidation as consolidate_mod
    from testuary.languages import generic

    supports = generic.GenericBackend._supports
    monkeypatch.setattr(generic.GenericBackend, "_supports", lambda self, operation, source_dir=None: (
        operation != "tests.per_test_evidence" and supports(self, operation, source_dir)))
    lines = {}
    audit = consolidate_mod.record_selection_audit

    def record_selection_audit(phase, candidates, winner, **options):
        lines.update({candidate.label: candidate.line for candidate in candidates})
        audit(phase, candidates, winner, **options)

    monkeypatch.setattr(consolidate_mod, "record_selection_audit", record_selection_audit)
    monkeypatch.setattr(consolidate_mod, "benchmark_tied_candidates",
                        lambda *args, **kwargs: lambda candidate: 1.0)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [_ALPHA, _BETA], monkeypatch)
    assert res.status == "ok"
    assert lines.get("suite_001") is not None and lines.get("suite_002") is not None


def _starter_pool_run(tmp_path, monkeypatch, method, *, pooled_unnamed_kill=False,
                      pooled_first_killers=False, pooled_nothing=False):
    from testuary import config
    from testuary import consolidation as consolidate_mod
    from testuary import measurement as test_metrics

    root, target, tests_dir, suites_dir = _project(tmp_path)
    pooled, per_suite, attributions = [], [], []
    monkeypatch.setattr(test_metrics, "union_mutation_runner",
                        _two_mutant_runner(pooled, per_suite, pooled_unnamed_kill=pooled_unnamed_kill,
                                           attributions=attributions,
                                           pooled_first_killers=pooled_first_killers,
                                           pooled_nothing=pooled_nothing))
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: True)
    monkeypatch.setattr(consolidate_mod, "benchmark_tied_candidates",
                        lambda *args, **kwargs: lambda candidate: 1.0)
    matrices = tmp_path / "matrices"
    res, _ = _run(root, target, tests_dir, suites_dir, [_ALPHA, _BETA], monkeypatch,
                  matrices_dir=matrices, suite_mutation=method)
    assert res.status == "ok"
    # Only the pooled run asks for every killing test.
    assert attributions.count("all") == len(pooled)
    return res, matrices, pooled, per_suite


def test_derive_runs_one_pooled_measurement_and_derives_each_starter_score(
    tmp_path, monkeypatch,
):
    """suite_mutation = derive: one pooled run of the starter suites,
    staged under the backend's names, scores each suite by its own tests'
    kills. No suite is measured alone, so no attribution record is written."""
    res, matrices, pooled, per_suite = _starter_pool_run(tmp_path, monkeypatch, "derive")
    assert pooled == [["test_candidate_suite_001.py", "test_candidate_suite_002.py"]] and per_suite == []
    assert res.suite_mutation_method == "derive"
    assert res.suite_mutation_stats["mean"] == 50.0
    assert not (matrices / "starter_pool_attribution.json").exists()
    record = json.loads((matrices / "starter_pool_derivation.json").read_text(encoding="utf-8"))
    assert (record["schema"], record["method_requested"], record["method_used"],
            record["derivation_unavailable"]) == ("testuary.starter_pool_derivation/1", "derive", "derive", None)
    assert [(slot["slot"], slot["pooled_name"], slot["derived"]["killed"], slot["derived"]["valid"])
            for slot in record["slots"]] == [(1, "test_candidate_suite_001.py", 1, 2),
                                             (2, "test_candidate_suite_002.py", 1, 2)]
    assert record["attribution"] == {"pooled": "all", "measured": None}
    staged = record["slots"][0]["staged_files"]
    assert [(f["original"], f["path"], f["renamed"], f["self_references"]) for f in staged] == [
        ("test_candidate.py", "test_candidate_suite_001.py", [], [])]
    assert all("measured" not in slot for slot in record["slots"])
    assert record["mean_score"] == {"derived": 50.0, "measured": None}
    derived = json.loads((matrices / "starter_pool_derived_kill_matrices.json").read_text(encoding="utf-8"))
    assert sorted(derived["suites"]) == ["suite_001", "suite_002"]
    assert not (matrices / "starter_pool_pooled_kill_matrix.json").exists()


def test_both_lets_the_measured_scores_govern_and_records_each_difference(
    tmp_path, monkeypatch,
):
    """suite_mutation = both: the measured per-suite scores are reported
    and the pooled derived ones recorded beside them. A pooled kill with no
    test named, credited to every suite, is a recorded difference, with its
    rule, for the suite that does not kill it alone."""
    res, matrices, pooled, per_suite = _starter_pool_run(
        tmp_path, monkeypatch, "both", pooled_unnamed_kill=True)
    assert len(pooled) == 1 and per_suite == [["suite_001", "suite_002"]]
    assert res.suite_mutation_method == "both"
    assert res.suite_mutation_stats["mean"] == res.suite_mutation_stats["max"] == 50.0
    assert (matrices / "starter_pool_attribution.json").exists()
    record = json.loads((matrices / "starter_pool_derivation.json").read_text(encoding="utf-8"))
    assert (record["method_requested"], record["method_used"]) == ("both", "both")
    # The measured runs record every killer too, by default (cover_attribution).
    assert record["attribution"] == {"pooled": "all", "measured": "full"}
    first, second = record["slots"]
    assert (first["derived"]["score"], first["measured"]["score"]) == (100.0, 50.0)
    assert first["differences"] == [{"mutant_id": "m2", "derived_status": "killed",
                                     "measured_status": "survived",
                                     "derivation_rule": "without_named_test",
                                     "pooled_status": "killed", "pooled_killers": [],
                                     "measured_killers": []}]
    assert first["differences_by_rule"] == {"without_named_test": 1}
    assert (second["derived"]["score"], second["measured"]["score"], second["differences"]) == (50.0, 50.0, [])
    assert record["mean_score"] == {"derived": 75.0, "measured": 50.0}
    pooled_km = json.loads((matrices / "starter_pool_pooled_kill_matrix.json").read_text(encoding="utf-8"))
    union = json.loads((matrices / "starter_pool_kill_matrix.json").read_text(encoding="utf-8"))
    assert [row.get("killer") for row in pooled_km["mutants"]][1] is None
    assert all("suite_verdicts" in row for row in union["mutants"])


def test_the_starter_count_starts_after_the_per_suite_mutation(tmp_path, monkeypatch):
    """The count of measured starter suites runs with their per-suite
    coverage: the per-suite mutation before it is one request, which shows
    its own progress, so the count does not stand at none while it runs."""
    from testuary import _progress, config
    from testuary import consolidation as consolidate_mod
    from testuary import measurement as test_metrics
    from testuary.presentation.view import STARTER_MEASUREMENT

    seen = []
    monkeypatch.setattr(_progress, "phase_observer",
                        lambda phase: lambda *args, **event: seen.append(event.get("operation")))
    monkeypatch.setattr(test_metrics, "union_mutation_runner",
                        _two_mutant_runner([], seen, pooled_unnamed_kill=False))
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: True)
    monkeypatch.setattr(consolidate_mod, "benchmark_tied_candidates",
                        lambda *args, **kwargs: lambda candidate: 1.0)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [_ALPHA, _BETA], monkeypatch)
    assert res.status == "ok"
    per_suite = next(k for k, item in enumerate(seen) if isinstance(item, list))
    assert seen.index(STARTER_MEASUREMENT) > per_suite


def test_derive_measures_each_suite_when_the_suites_fail_together(tmp_path, monkeypatch):
    """Starter suites that pass alone but fail when run together lose no
    test for the pooled run: derive measures each suite alone, and the
    record names the failing test."""
    from testuary import config
    from testuary import consolidation as consolidate_mod
    from testuary import measurement as test_metrics

    root, target, tests_dir, suites_dir = _project(tmp_path)
    pooled, per_suite = [], []
    monkeypatch.setattr(test_metrics, "union_mutation_runner",
                        _two_mutant_runner(pooled, per_suite, pooled_unnamed_kill=False))
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: True)
    monkeypatch.setattr(consolidate_mod, "benchmark_tied_candidates",
                        lambda *args, **kwargs: lambda candidate: 1.0)
    matrices = tmp_path / "matrices"
    res, _ = _run(root, target, tests_dir, suites_dir, [_WRITER, _CHECKER], monkeypatch,
                  matrices_dir=matrices, suite_mutation="derive")
    assert res.status == "ok"
    assert pooled == [] and per_suite == [["suite_001", "suite_002"]]
    assert res.suite_mutation_method == "measure"
    record = json.loads((matrices / "starter_pool_derivation.json").read_text(encoding="utf-8"))
    assert record["derivation_unavailable"].startswith(
        "the starter suites do not pass when run together")
    assert "test_no_marker" in record["derivation_unavailable"]


@pytest.mark.parametrize("method", ("derive", "both"))
def test_derive_measures_each_suite_when_the_pooled_run_names_first_killers_only(
    method, tmp_path, monkeypatch,
):
    """A pooled run that records only each kill's first killing test (Java's
    run when PIT's every-killer launch gives no report) cannot derive each
    suite's kills: each suite is measured alone, no derivation is
    recorded and the record says why."""
    res, matrices, pooled, per_suite = _starter_pool_run(
        tmp_path, monkeypatch, method, pooled_first_killers=True)
    assert len(pooled) == 1 and per_suite == [["suite_001", "suite_002"]]
    assert res.suite_mutation_method == "measure"
    record = json.loads((matrices / "starter_pool_derivation.json").read_text(encoding="utf-8"))
    assert (record["method_requested"], record["method_used"], record["derivation_unavailable"]) == (
        method, "measure", "the pooled mutation run did not record every killing test")
    assert record["attribution"]["pooled"] is None
    assert all(slot["derived"] is None for slot in record["slots"])


def test_derive_measures_each_suite_when_the_suites_cannot_run_together(
    tmp_path, monkeypatch,
):
    """When the starter suites cannot be staged into one run, derive
    measures each suite alone, reports measure as the method used and
    records why no derivation was made."""
    from testuary import starter_selection as starter_selection

    monkeypatch.setattr(starter_selection, "stage_pool", lambda *args, **kwargs: (None, "the backend cannot stage this pool"))
    res, matrices, pooled, per_suite = _starter_pool_run(tmp_path, monkeypatch, "derive")
    assert pooled == [] and per_suite == [["suite_001", "suite_002"]]
    assert res.suite_mutation_method == "measure"
    assert (matrices / "starter_pool_attribution.json").exists()
    record = json.loads((matrices / "starter_pool_derivation.json").read_text(encoding="utf-8"))
    assert (record["method_requested"], record["method_used"]) == ("derive", "measure")
    assert record["derivation_unavailable"] == "the backend cannot stage this pool"
    assert all(slot["derived"] is None for slot in record["slots"])


@pytest.mark.parametrize("method", ("measure", "both"))
def test_a_failed_per_suite_measurement_leaves_no_derived_value_governing(
    method, tmp_path, monkeypatch,
):
    """Under measure and both, a per-suite measurement that returns nothing
    leaves every starter suite unscored: none is ranked or reported by a
    mutation score and no mean is formed. Under both the derivation and the
    pooled matrix are still recorded, with the measurement marked
    unavailable. An unmeasured starter suite names no measurement scope."""
    from contextlib import contextmanager
    from testuary import config
    from testuary import consolidation as consolidate_mod
    from testuary import measurement as test_metrics

    root, target, tests_dir, suites_dir = _project(tmp_path)
    per_suite_calls = []

    def _measure(d):
        bodies = {f.name: f.read_text(encoding="utf-8") for f in Path(d).glob("*.py")}
        rows = []
        for mid, test, line in (("m1", "test_alpha", 2), ("m2", "test_beta", 3)):
            holder = next((name for name, body in bodies.items() if f"def {test}" in body), None)
            row = {"id": mid, "operator": "core/Op", "line": line, "diff": mid}
            if holder is None:
                row["status"] = "survived"
            else:
                killer = f"{holder}::{test}"
                row.update(status="killed", killer=killer,
                           killing_tests=killing_tests([killer]))
            rows.append(row)
        return {"mutants": rows}

    @contextmanager
    def fake_runner(**kw):
        def run(d, *, attribution=None):
            # The matrix records the attribution asked for, as LangAdapter's runs do.
            return {**_measure(d), "attribution": attribution}

        def per_suite(suites):
            per_suite_calls.append(sorted(suites))
            return None

        run.per_suite = per_suite
        run.named_of = lambda test_dir: ()
        yield run

    monkeypatch.setattr(test_metrics, "union_mutation_runner", fake_runner)
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: True)
    monkeypatch.setattr(consolidate_mod, "benchmark_tied_candidates",
                        lambda *args, **kwargs: lambda candidate: 1.0)
    evidence = []
    real_consolidate = consolidate_mod.consolidate_phase_end

    def capture_evidence(context, candidates, mutation_runner=None, options=None):
        evidence.extend(options.candidate_evidence)
        return real_consolidate(context, candidates, mutation_runner, options)

    monkeypatch.setattr(consolidate_mod, "consolidate_phase_end", capture_evidence)
    ranked = []
    real_pick = consolidate_mod.pick_phase_result

    scopes = []

    def capture(candidates, **kwargs):
        ranked.extend((c.label, c.mutation, c.status_reason) for c in candidates
                      if c.label.startswith("suite_"))
        scopes.extend(c.measurement_identity for c in candidates if c.label.startswith("suite_"))
        return real_pick(candidates, **kwargs)

    monkeypatch.setattr(consolidate_mod, "pick_phase_result", capture)
    matrices = tmp_path / "matrices"
    res, _ = _run(root, target, tests_dir, suites_dir, [_ALPHA, _BETA], monkeypatch,
                  matrices_dir=matrices, suite_mutation=method)
    assert per_suite_calls == [["suite_001", "suite_002"]]
    assert not res.suite_mutation_stats
    assert res.suite_mutation_method == "unavailable"
    assert scopes and all(scope == {} for scope in scopes)
    assert ranked and all(mutation is None for _, mutation, _ in ranked)
    # The merging step's collision evidence holds no derived kill.
    assert not any(token.startswith("K:") for suite in evidence if suite
                   for tokens in suite.values() for token in tokens)
    assert not any("derived" in reason for _, _, reason in ranked)
    assert not (matrices / "starter_pool_attribution.json").exists()
    assert not (matrices / "starter_pool_kill_matrix.json").exists()
    record_path = matrices / "starter_pool_derivation.json"
    pooled_path = matrices / "starter_pool_pooled_kill_matrix.json"
    if method == "measure":
        assert not record_path.exists() and not pooled_path.exists()
    else:
        assert len(json.loads(pooled_path.read_text(encoding="utf-8"))["mutants"]) == 2
        record = json.loads(record_path.read_text(encoding="utf-8"))
        assert record["measurement_unavailable"] == "the per-suite mutation measurement produced no result"
        assert record["method_used"] == "unavailable"
        assert record["mean_score"]["measured"] is None
        assert record["mean_score"]["derived"] == 50.0


def test_both_without_a_derivation_reports_measure(tmp_path, monkeypatch):
    """Under both, suites that cannot be pooled are measured, the method used
    reads measure and the record keeps the reason."""
    from testuary import starter_selection as starter_selection

    monkeypatch.setattr(starter_selection, "stage_pool",
                        lambda *args, **kwargs: (None, "the backend cannot stage this pool"))
    res, matrices, pooled, per_suite = _starter_pool_run(tmp_path, monkeypatch, "both")
    assert pooled == [] and per_suite == [["suite_001", "suite_002"]]
    assert res.suite_mutation_method == "measure"
    assert res.suite_mutation_stats["mean"] == 50.0
    record = json.loads((matrices / "starter_pool_derivation.json").read_text(encoding="utf-8"))
    assert (record["method_requested"], record["method_used"]) == ("both", "measure")
    assert record["derivation_unavailable"] == "the backend cannot stage this pool"


def test_measure_mutation_off_keeps_coverage_lossless_only(
    tmp_path, monkeypatch,
):
    """With consolidation_measure_mutation off, the runner is never opened
    and coverage-identical twins reduce to one."""
    from testuary import config
    from testuary import consolidation as consolidate_mod
    from testuary import measurement as test_metrics

    root, target, tests_dir, suites_dir = _project(tmp_path)
    twin_a = "from calc import add\n\ndef test_alpha():\n    assert add(1, 1) == 2\n"
    twin_b = "from calc import add\n\ndef test_beta():\n    assert add(1, 1) == 2\n"
    opened = []

    def boom(**kw):
        opened.append(1)
        raise AssertionError("runner must not open when the knob is off")

    monkeypatch.setattr(test_metrics, "union_mutation_runner", boom)
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: False)
    monkeypatch.setattr(
        consolidate_mod,
        "benchmark_tied_candidates",
        lambda *args, **kwargs: lambda candidate: candidate.runtime_ms,
    )
    res, _ = _run(
        root, target, tests_dir, suites_dir, [twin_a, twin_b], monkeypatch,
    )
    assert res.status == "ok"
    assert opened == []
    text = Path(res.baseline_file).read_text(encoding="utf-8")
    assert ("test_alpha" in text) != ("test_beta" in text)


def test_per_suite_measurement_retries_once_then_succeeds(tmp_path, monkeypatch):
    """Per-suite measurement retries a transient None from
    collect_eviction_data once, so a flaky first measurement is not dropped."""
    from testuary import config

    # Coverage-only keeps it fast and deterministic (no cosmic-ray).
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: False)
    real = GenericBackend.collect_eviction_data
    state = {"n": 0}

    def flaky(self, *args, **kwargs):
        state["n"] += 1
        if state["n"] == 1:                 # the seed's first attempt flakes
            kwargs["failure_fn"]("infrastructure", "the run was cut")
            return None
        return real(self, *args, **kwargs)

    monkeypatch.setattr(GenericBackend, "collect_eviction_data", flaky)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch,
                  suite_count=1)
    assert res.status == "ok"
    assert state["n"] >= 2                   # retried after the transient None
    assert res.suite_line_stats               # the seed was measured, not dropped


def test_best_suite_is_selected_when_the_merge_is_unavailable(
    tmp_path, monkeypatch,
):
    """A fully measured seed remains a normal candidate when consolidation
    cannot provide a measured reduced or merged suite."""
    from testuary import config
    from testuary import consolidation as consolidate_mod

    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: False)
    materialized = []
    real_measure = consolidate_mod.measure_candidate_standalone

    def record_measure(*args, **kwargs):
        # The materialization's place measures through this function again.
        if kwargs.get("place") is not None:
            materialized.append({"stage_as_delivered": kwargs.get("stage_as_delivered"),
                                 "place": kwargs["place"]})
        return real_measure(*args, **kwargs)

    monkeypatch.setattr(
        consolidate_mod, "measure_candidate_standalone", record_measure,
    )
    # Force the merge out of contention (reduced_source None -> non-green).
    monkeypatch.setattr(
        consolidate_mod, "consolidate_phase_end",
        lambda *a, **k: consolidate_mod.PhaseConsolidation(
            detail="forced: merge unsalvageable"))
    root, target, tests_dir, suites_dir = _project(tmp_path)
    matrices_dir = tmp_path / "matrices"
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A],
                  monkeypatch, matrices_dir=matrices_dir)
    assert res.status != "failed_fold_regression"
    assert res.baseline_from_single_suite is True
    assert res.baseline_file is not None
    record = json.loads(
        (matrices_dir / "consolidation_record.json").read_text(
            encoding="utf-8"
        )
    )
    assert record["delivered_from"] == "single_suite"
    assert record["winner_source_label"].startswith("suite_")
    selected = json.loads(
        (matrices_dir / "selected_coverage_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert selected["source_sha256"] == record["selected_source_sha256"]
    assert selected["execution_name"] == record["execution_name"]
    assert materialized == [{"stage_as_delivered": True, "place": "starter.candidate"}]


def test_a_winner_the_coverage_tool_cannot_measure_under_its_delivered_name_is_incomplete(
    tmp_path, monkeypatch,
):
    """Under the delivered filename the normal run decides whether the
    chosen suite passes. When only the coverage tool's run fails there, the
    suite is recorded incomplete with the test that stopped the tool, not
    failing, and it is no choice: without another candidate the phase has
    no result."""
    from testuary import config
    from testuary import consolidation as consolidate_mod
    from testuary.languages import TestRunResult
    from testuary.languages.generic import TestFailure

    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: False)
    monkeypatch.setattr(
        consolidate_mod, "consolidate_phase_end",
        lambda *a, **k: consolidate_mod.PhaseConsolidation(
            detail="forced: merge unsalvageable"))
    root, target, tests_dir, suites_dir = _project(tmp_path)
    real_check = consolidate_mod._check_then_measure

    def check(backend, file_paths, source_dir, **kwargs):
        if Path(tests_dir).resolve() in Path(file_paths[0]).resolve().parents:
            return TestRunResult(
                ok=False, returncode=1, timed_out=False, elapsed_s=0.1, collected=1,
                passed=0, failed=1, errors=0, output="FAILED test_a",
                line_coverage=None, branch_coverage=None,
                failures=(TestFailure("test_a", "failed", "only under the tool"),)), True
        return real_check(backend, file_paths, source_dir, **kwargs)

    monkeypatch.setattr(consolidate_mod, "_check_then_measure", check)
    recorded = {}
    real_audit = consolidate_mod.record_selection_audit

    def audit(phase, candidates, winner, **kwargs):
        recorded.update({c.label: c for c in candidates})
        return real_audit(phase, candidates, winner, **kwargs)

    monkeypatch.setattr(consolidate_mod, "record_selection_audit", audit)
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch,
                  matrices_dir=tmp_path / "matrices")
    assert res.status == "failed_phase_selection"
    suites = [c for label, c in recorded.items() if label.startswith("suite_")]
    assert len(suites) == 1
    assert suites[0].green is True and suites[0].status == "incomplete"
    assert suites[0].status_reason.startswith(
        "the coverage tool could not measure it under the delivered filename")
    assert "test_a" in suites[0].status_reason


def test_ollama_native_stays_in_default_reasoning_mode(monkeypatch):
    """No think override is sent, so every suite from one model uses the same
    mode. A reply that spends its output token limit on reasoning returns empty content,
    not a retry in another mode."""

    bodies = []

    class _FakeStream:
        def __init__(self, body):
            bodies.append(body)
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_lines(self):
            # thinking only, no content, then done (token limit reached)
            yield '{"message": {"thinking": "..."}}'
            yield '{"done": true, "eval_count": 16384}'

    def fake_stream(method, url, **kw):
        return _FakeStream(kw.get("json", {}))

    import httpx
    monkeypatch.setattr(httpx, "stream", fake_stream)
    api = {"model": "kimi-k2.6:cloud",
           "api_url": "https://ollama.com/api/chat", "api_key": "k"}
    text, usage = generate(
        api, "prompt", temperature=0.7, seed=1, timeout=60)
    assert len(bodies) == 1
    assert "think" not in bodies[0]
    assert "num_predict" not in bodies[0]["options"]
    assert text == ""
    # No prompt_eval_count: usage is unknown, not zero input tokens.
    assert usage["usage_known"] is False


def test_ollama_native_usage_without_a_cached_count_has_an_unknown_split(monkeypatch):
    """A native reply without prompt_eval_cached_count keeps its counts with
    an unknown cache split, and the content joins the streamed pieces."""

    class _FakeStream:
        def __init__(self, body): pass
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_lines(self):
            yield '{"message": {"content": "```python\\n"}}'
            yield '{"message": {"content": "def test_x():\\n    assert True\\n```"}}'
            yield ('{"done": true, "prompt_eval_count": 1200, '
                   '"eval_count": 64}')

    import httpx
    monkeypatch.setattr(httpx, "stream",
                        lambda method, url, **kw: _FakeStream(kw))
    api = {"model": "deepseek-v4-pro:cloud",
           "api_url": "https://ollama.com/api/chat", "api_key": "k"}
    text, usage = generate(
        api, "prompt", temperature=0.7, seed=1, timeout=60)
    assert "def test_x" in text
    assert usage["input_tokens"] == 1200
    assert usage["output_tokens"] == 64
    assert usage["usage_known"] is True
    assert usage["cache_hit_tokens"] is None and usage["cache_miss_tokens"] is None


def test_ollama_native_sets_num_predict_from_the_output_limit(monkeypatch):
    """On the native path max_output_tokens sets num_predict, replacing the
    endpoint default, and reasoning stays on."""

    bodies = []

    class _FakeStream:
        def __init__(self, body):
            bodies.append(body)
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def raise_for_status(self): pass
        def iter_lines(self):
            yield '{"message": {"content": "```python\\nok\\n```"}}'
            yield '{"done": true, "eval_count": 9000}'

    import httpx
    monkeypatch.setattr(
        httpx, "stream",
        lambda method, url, **kw: _FakeStream(kw.get("json", {})))
    api = {"model": "kimi-k2.6:cloud",
           "api_url": "https://ollama.com/api/chat", "api_key": "k"}
    text, usage = generate(
        api, "prompt", temperature=0.7, seed=1, timeout=60,
        max_output_tokens=262144)
    assert bodies[0]["options"]["num_predict"] == 262144
    assert "think" not in bodies[0]
    assert text == "```python\nok\n```"


def _reasoning_chunk_cls():
    class _Chunk:
        def __init__(self, content):
            self.content = content
            self.usage_metadata = {"input_tokens": 1, "output_tokens": 1}
        def __add__(self, other):
            return _Chunk(self.content + other.content)
    return _Chunk


def test_reasoning_grace_cuts_a_zero_answer_stream(monkeypatch):
    """A stream with no answer content is cut once the grace passes and
    returns empty content, so one generation cannot run to the context limit."""
    from testuary import starters as initmod
    from testuary import _chat_models

    _Chunk = _reasoning_chunk_cls()
    consumed = {"n": 0}

    class _FakeModel:
        def stream(self, prompt):
            for _ in range(1000):
                consumed["n"] += 1
                yield _Chunk("")  # reasoning only: no answer content ever

    monkeypatch.setattr(_chat_models, "build_chat_model", lambda **kw: _FakeModel())
    clock = {"t": 0.0}

    def fake_time():
        clock["t"] += 1.0
        return clock["t"]

    monkeypatch.setattr(initmod.time, "monotonic", fake_time)
    api = {"model": "m", "api_url": "http://localhost:8000/v1",
           "api_key": "k"}
    text, _usage = generate(
        api, "prompt", temperature=0.7, seed=1, timeout=60,
        reasoning_grace=3)
    assert text == ""
    assert consumed["n"] < 1000


def test_reasoning_grace_never_cuts_a_productive_stream(monkeypatch):
    """Once a stream emits answer content the grace never fires, however long
    it runs."""
    from testuary import starters as initmod
    from testuary import _chat_models

    _Chunk = _reasoning_chunk_cls()
    consumed = {"n": 0}

    class _FakeModel:
        def stream(self, prompt):
            for i in range(20):
                consumed["n"] += 1
                yield _Chunk(f"c{i} ")  # answer content present from the start

    monkeypatch.setattr(_chat_models, "build_chat_model", lambda **kw: _FakeModel())
    clock = {"t": 0.0}

    def fake_time():
        clock["t"] += 1.0
        return clock["t"]

    monkeypatch.setattr(initmod.time, "monotonic", fake_time)
    api = {"model": "m", "api_url": "http://localhost:8000/v1",
           "api_key": "k"}
    text, _usage = generate(
        api, "prompt", temperature=0.7, seed=1, timeout=60,
        reasoning_grace=3)
    assert consumed["n"] == 20
    assert "c19 " in text


def test_unbounded_stream_consumes_every_chunk(monkeypatch):
    """Without a grace the stream runs to completion."""
    from testuary import _chat_models

    _Chunk = _reasoning_chunk_cls()
    consumed = {"n": 0}

    class _FakeModel:
        def stream(self, prompt):
            for i in range(5):
                consumed["n"] += 1
                yield _Chunk(f"c{i} ")

    monkeypatch.setattr(_chat_models, "build_chat_model", lambda **kw: _FakeModel())
    api = {"model": "m", "api_url": "http://localhost:8000/v1",
           "api_key": "k"}
    text, _usage = generate(
        api, "prompt", temperature=0.7, seed=1, timeout=60)
    assert text == "c0 c1 c2 c3 c4 "
    assert consumed["n"] == 5


def test_extract_code_discards_when_closing_fence_is_missing():
    """Extraction returns a closed code block and nothing for a reply cut off
    before its closing fence or without a fence."""
    from testuary.starters import _extract_code

    truncated = (
        "Here are the tests:\n```python\n"
        "def test_a():\n    assert add(1, 2) == 3\n\n"
        "def test_b():\n    assert add(1, 2,"   # cut mid-call, no ```
    )
    assert _extract_code(truncated, "python") is None

    closed = "```python\ndef test_ok():\n    pass\n```\ntrailing prose"
    assert _extract_code(closed, "python").strip() == "def test_ok():\n    pass"

    assert _extract_code("no fence at all, just prose", "python") is None


class _FakeChunk:
    def __init__(self, content, usage):
        self.content = content
        self.usage_metadata = usage

    def __add__(self, other):
        return _FakeChunk(self.content + other.content, other.usage_metadata)


def test_is_context_overflow_classifies_gateway_rejections():
    """The overflow class is recognized from the message shape. Other
    rejections (rate limits, auth) are not retried as overflows."""
    from testuary.starters import _is_context_overflow

    overflow = Exception(
        "litellm.BadRequestError: OpenAIException - You passed 43729 "
        "input tokens and requested 30000 output tokens. However, the "
        "model's context length is only 73728 tokens")
    assert _is_context_overflow(overflow)
    assert not _is_context_overflow(Exception("429 rate limited"))
    assert not _is_context_overflow(Exception("invalid api key"))


def test_default_generate_passes_max_output_tokens(monkeypatch):
    """On the OpenAI-compatible path the output token limit is sent as max_tokens,
    so a gateway does not truncate a large suite at its default."""
    from testuary import _chat_models

    seen = {}

    class _Model:
        def stream(self, prompt):
            yield _FakeChunk("```python\ndef test_a():\n    pass\n```",
                             {"input_tokens": 3, "output_tokens": 5})

    def fake_build(**kw):
        seen.update(kw)
        return _Model()

    monkeypatch.setattr(_chat_models, "build_chat_model", fake_build)
    api = {"model": "Qwen", "api_url": "https://gateway.example/api",
           "api_key": "k"}
    text, usage = generate(
        api, "prompt", temperature=0.7, seed=1, timeout=60,
        max_output_tokens=24576)
    assert seen["max_tokens"] == 24576
    assert "def test_a" in text
    assert usage["output_tokens"] == 5


def test_default_generate_halves_the_output_limit_on_context_overflow(monkeypatch):
    """A context overflow halves the output token limit per rejection until the
    request fits, as the rejection gives no usable prompt size. A persistent
    overflow re-raises at the floor."""
    from testuary import _chat_models

    overflow_msg = (
        "OpenAIException - You passed 41000 input tokens and requested "
        "{req} output tokens. However, the model's context length is "
        "only 73728 tokens")
    limits = []

    class _Model:
        def __init__(self, output_limit, limit):
            self.output_limit, self.limit = output_limit, limit

        def stream(self, prompt):
            # The provider rejects the request, not the client's construction.
            if self.output_limit > self.limit:
                raise RuntimeError(overflow_msg.format(req=self.output_limit))
            yield _FakeChunk("```python\ndef test_a():\n    pass\n```",
                             {"input_tokens": 41000, "output_tokens": 100})

    def fake_build(**kw):
        limits.append(kw.get("max_tokens"))
        return _Model(kw.get("max_tokens"), 20000)

    monkeypatch.setattr(_chat_models, "build_chat_model", fake_build)
    api = {"model": "Qwen", "api_url": "https://gateway.example/api",
           "api_key": "k"}
    text, usage = generate(
        api, "prompt", temperature=0.7, seed=1, timeout=60,
        max_output_tokens=70000)
    assert limits == [70000, 35000, 17500]
    assert "def test_a" in text

    # A rejection that never clears re-raises at the floor.
    limits.clear()

    def always_overflow(**kw):
        limits.append(kw.get("max_tokens"))
        return _Model(kw.get("max_tokens"), 0)

    monkeypatch.setattr(_chat_models, "build_chat_model", always_overflow)
    import pytest as _pytest
    with _pytest.raises(RuntimeError, match="context length"):
        generate(
            api, "prompt", temperature=0.7, seed=1, timeout=60,
            max_output_tokens=70000)
    assert limits[-1] >= 2048 and len(limits) <= 7


def test_reuse_suites_skips_generation_and_reuses_pool(tmp_path, monkeypatch):
    """A second run with reuse_suites loads the saved starter suites without a
    model call and reports the same generation cost and time as the run that
    generated them."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res1, calls1 = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch)
    assert res1.status == "ok"
    assert calls1["n"] >= 1
    assert res1.n_green >= 1
    assert list(suites_dir.glob("suite_*/cleaned.py"))
    assert res1.requests_used > 0 and res1.input_tokens > 0
    # The generating run saves a manifest of cost and wall-clock time.
    assert (suites_dir / "_generation.json").exists()

    tests_dir2 = tmp_path / "tests_out2"
    res2, calls2 = _run(root, target, tests_dir2, suites_dir, [GREEN_A],
                        monkeypatch, reuse_suites=True)
    assert calls2["n"] == 0  # generation never ran (no model calls)
    assert res2.status == "ok"
    assert res2.n_green == res1.n_green
    assert res2.requests_used == res1.requests_used
    assert res2.input_tokens == res1.input_tokens
    assert res2.output_tokens == res1.output_tokens
    assert res2.generation_seconds == res1.generation_seconds
    # The reused suite lands at the target's coverage path.
    assert list(tests_dir2.rglob("*.py"))


def test_reuse_manifest_is_authoritative_over_the_slot_sum(tmp_path):
    """Reuse takes totals from the generation manifest, not the per-slot sum,
    which misses attempts that exact mode overwrote in place."""
    suites = tmp_path / "initial_seeds"
    for i in (1, 2):
        d = suites / f"suite_{i:03d}"
        d.mkdir(parents=True)
        (d / "cleaned.py").write_text(
            "def test_x():\n    assert True\n", encoding="utf-8")
        (d / "original.py").write_text("x", encoding="utf-8")
        (d / "original.json").write_text(
            json.dumps({"attempts": 1, "input_tokens": 100, "output_tokens": 20,
                        "cache_hit_tokens": 40,
                        "classification": "pruned", "extracted": True}),
            encoding="utf-8")
    # The two slots sum to 2 requests, 200 in, 40 out, but the run spent 3,
    # 300 and 55 over 3 generations.
    (suites / starters._GENERATION_MANIFEST).write_text(
        json.dumps({"requests_used": 3, "input_tokens": 300,
                    "output_tokens": 55, "cache_hit_tokens": 120,
                    "generation_seconds": 42.0,
                    "n_generated": 3}),
        encoding="utf-8")
    res = starters.StarterResult()
    starters._load_persisted_green(suites, {}, res, lambda *a, **k: None)
    assert res.requests_used == 3       # manifest, not the 2 summed seeds
    assert res.input_tokens == 300      # manifest, not 200
    assert res.output_tokens == 55      # manifest, not 40
    assert res.n_generated == 3         # manifest, not the 2 persisted original.py
    assert res.generation_seconds == 42.0
    assert res.cache_hit_tokens == 120   # manifest, not the 80 summed seeds


def test_reuse_without_persisted_suites_falls_back_to_fresh(tmp_path, monkeypatch):
    """reuse_suites for a target without a saved pool generates a fresh one, so
    one reuse run can mix old and new targets."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    suites_dir.mkdir(parents=True, exist_ok=True)   # empty: no seed_*/cleaned.py
    res, calls = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch,
                      reuse_suites=True)
    assert res.status == "ok"
    assert calls["n"] >= 1                              # fresh generation ran
    assert list(suites_dir.glob("suite_*/cleaned.py"))   # a fresh pool was persisted
    assert (suites_dir / starters._GENERATION_MANIFEST).exists()   # manifest written


def test_has_persisted_pool_detects_a_pool(tmp_path):
    """_has_persisted_pool is the reuse-vs-fresh switch: true once a
    generation manifest exists, which a generation writes before its first
    request."""
    suites = tmp_path / "suites"
    assert not starters._has_persisted_pool(suites)           # absent dir
    suites.mkdir()
    assert not starters._has_persisted_pool(suites)           # empty dir
    d = suites / "suite_001"
    d.mkdir()
    (d / "cleaned.py").write_text("def test():\n    assert True\n",
                                  encoding="utf-8")
    assert not starters._has_persisted_pool(suites)           # seeds without a manifest
    ran = tmp_path / "ran"
    ran.mkdir()
    (ran / starters._GENERATION_MANIFEST).write_text("{}", encoding="utf-8")
    assert starters._has_persisted_pool(ran)                 # a generation ran


def test_load_persisted_green_requires_suites(tmp_path):
    """The loader refuses an empty pool. run_starter_phase checks
    _has_persisted_pool first, so this fires only on a direct call."""
    res = starters.StarterResult()
    with pytest.raises(FileNotFoundError):
        starters._load_persisted_green(tmp_path / "absent", {}, res,
                                       lambda *a, **k: None)


def _persist_pool(suites_dir, suites, manifest):
    """Write a reuse-ready seed pool: ``suite_NNN/{cleaned.py, original.py,
    original.json}`` for each index -> source, plus the generation manifest."""
    suites_dir.mkdir(parents=True, exist_ok=True)
    for i, src in suites.items():
        d = suites_dir / f"suite_{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        (d / "cleaned.py").write_text(src, encoding="utf-8")
        (d / "original.py").write_text("x", encoding="utf-8")
        (d / "original.json").write_text(
            json.dumps({"attempts": 1, "input_tokens": 100, "output_tokens": 20,
                        "cache_hit_tokens": 0, "classification": "pruned",
                        "extracted": True}), encoding="utf-8")
    (suites_dir / starters._GENERATION_MANIFEST).write_text(
        json.dumps(manifest), encoding="utf-8")


# A module-level importorskip of an absent package skips the whole file: no
# test collected, no failure, so it reads as green but skips the merged suite.
_POISON_MODULE_SKIP = (
    "import pytest\n"
    "pytest.importorskip('absent_pkg_zzz', minversion='0.0')\n"
    "from calc import add\n\n"
    "def test_add():\n    assert add(1, 2) == 3\n")

_REPLACEMENT_GREEN = (
    "from calc import clamp\n\n"
    "def test_clamp():\n    assert clamp(5, 0, 3) == 3\n")


def test_revalidate_reused_pool_runs_nothing_for_clean_pool(tmp_path, monkeypatch):
    """Revalidation runs only suites with an unconditional module-level skip,
    so a healthy reused pool is never re-run."""
    from testuary import languages

    root, target, _, suites_dir = _project(tmp_path)
    suites_dir.mkdir(parents=True, exist_ok=True)
    backend = languages.get_backend("python")

    def boom(*_a, **_k):
        raise AssertionError("a clean pool must not be re-run")

    monkeypatch.setattr(type(backend), "run_test_files", boom)
    green = {1: GREEN_A, 2: GREEN_A}
    dropped = starters._revalidate_reused_pool(
        green, backend, str(root),
        timeout=60, log=lambda *a, **k: None)
    assert dropped == 0
    assert set(green) == {1, 2}


def test_revalidate_reused_pool_drops_empty_suite(tmp_path):
    """A reused suite that collects no test is dropped, a healthy one kept."""
    from testuary import languages

    root, target, _, suites_dir = _project(tmp_path)
    suites_dir.mkdir(parents=True, exist_ok=True)
    backend = languages.get_backend("python")
    green = {1: _POISON_MODULE_SKIP, 2: GREEN_A}
    dropped = starters._revalidate_reused_pool(
        green, backend, str(root),
        timeout=120, log=lambda *a, **k: None)
    assert dropped == 1
    assert set(green) == {2}


def test_an_unanswered_collision_check_at_the_phase_end_is_not_judged(tmp_path, monkeypatch):
    """The phase end's collision check of the green suites, whose result the
    phase does not use, gives no answer: the check is not judged and the
    phase ends with its suite. The merge's own resolution is the
    backend's."""
    from testuary import languages
    from testuary.consolidation import ValidationUnresolved

    backend_type = type(languages.get_backend("python"))
    resolve = backend_type.resolve_test_collisions

    def unanswered_check(self, sources, **kwargs):
        if not kwargs:
            raise ValidationUnresolved("timeout: the collision resolution gave no answer")
        return resolve(self, sources, **kwargs)

    monkeypatch.setattr(backend_type, "resolve_test_collisions", unanswered_check)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [_ALPHA, _BETA], monkeypatch)
    assert res.status == "ok"


def test_a_starter_run_leaves_no_scratch_dir_in_its_suites(tmp_path, monkeypatch):
    """The salvage runs each response in a scratch directory it removes, so the
    suites directory keeps only the suites' own files."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch)
    assert res.status == "ok"
    assert (suites_dir / "suite_001").is_dir()
    assert not list(suites_dir.glob("suite_*/scratch"))


def test_revalidate_reused_pool_drops_a_suite_whose_run_crashed(tmp_path, monkeypatch):
    """A re-validation run without its record, such as a crash, observes no
    count, which the language layer reports as None: the suite is dropped
    for regeneration, as one that collects no test."""
    from types import SimpleNamespace
    from testuary import languages

    root, target, _, suites_dir = _project(tmp_path)
    suites_dir.mkdir(parents=True, exist_ok=True)
    backend = languages.get_backend("python")
    crashed = SimpleNamespace(ok=False, returncode=-11, timed_out=False, collected=None, passed=None,
                              failed=None, errors=None, output="Segmentation fault")
    monkeypatch.setattr(starters.normal_run, "run", lambda *a, **k: crashed)
    green = {1: _POISON_MODULE_SKIP}
    dropped = starters._revalidate_reused_pool(
        green, backend, str(root), timeout=120, log=lambda *a, **k: None)
    assert (dropped, green) == (1, {})


def test_reuse_refills_missing_slots(tmp_path, monkeypatch):
    """Exact mode completes a reused pool shorter than suite_count: only the
    missing slots are generated and their cost is added to the totals."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    _persist_pool(suites_dir, {1: GREEN_A},
                  {"requests_used": 1, "input_tokens": 100, "output_tokens": 20,
                   "cache_hit_tokens": 0, "generation_seconds": 5.0,
                   "n_generated": 1})
    res, calls = _run(root, target, tests_dir, suites_dir, [_REPLACEMENT_GREEN],
                      monkeypatch, reuse_suites=True, suite_count=3, refill_failed_suites=True)
    assert res.status == "ok"
    assert res.n_green == 3           # 1 reused + 2 regenerated
    assert calls["n"] >= 2            # only the two missing slots hit the model
    assert res.requests_used > 1      # refill added to the restored totals


def test_the_removed_strict_reuse_variable_is_refused(monkeypatch, in_run):
    """TESTUARY_INITIAL_REUSE_STRICT is no setting: a run refuses it, as any
    unknown TESTUARY_ variable, so it cannot change how a reused pool short
    of suite_count is completed."""
    monkeypatch.setenv("TESTUARY_INITIAL_REUSE_STRICT", "1")
    with pytest.raises(ValueError, match="TESTUARY_INITIAL_REUSE_STRICT"):
        in_run()


def test_reuse_drops_and_regenerates_empty_suite(tmp_path, monkeypatch):
    """A reused suite that collects no test is dropped and its slot
    regenerated, while a healthy sibling is reused unchanged."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    _persist_pool(suites_dir, {1: _POISON_MODULE_SKIP, 2: GREEN_A},
                  {"requests_used": 2, "input_tokens": 200, "output_tokens": 40,
                   "cache_hit_tokens": 0, "generation_seconds": 10.0,
                   "n_generated": 2})
    healthy_before = (suites_dir / "suite_002" / "cleaned.py").read_text(
        encoding="utf-8")
    res, calls = _run(root, target, tests_dir, suites_dir, [_REPLACEMENT_GREEN],
                      monkeypatch, reuse_suites=True, suite_count=2, refill_failed_suites=True)
    assert res.status == "ok"
    assert res.n_green == 2
    assert calls["n"] >= 1            # the emptied slot was regenerated
    assert (suites_dir / "suite_002" / "cleaned.py").read_text(
        encoding="utf-8") == healthy_before


def test_reuse_suites_via_real_env_var_skips_generation(tmp_path, monkeypatch, in_run):
    """TESTUARY_STARTER_REUSE_SUITES=1, read through the real
    get_agent_config, makes the phase reuse the saved pool without a model
    call, so the settings reader does not drop the key."""
    from testuary import config
    # Keep the real get_agent_config: each _run_real_cfg call patches it, and
    # the second call still needs the environment-driven one.
    real_gac = config.get_agent_config
    root, target, tests_dir, suites_dir = _project(tmp_path)

    # The first run generates and saves the pool, reuse off.
    monkeypatch.delenv("TESTUARY_STARTER_REUSE_SUITES", raising=False)
    in_run()
    res1, calls1 = _run_real_cfg(real_gac, root, target, tests_dir, suites_dir,
                                 [GREEN_A], monkeypatch)
    assert res1.status == "ok" and res1.n_green >= 1 and calls1["n"] >= 1

    # Second run: the env var alone (real coercion) drives the reuse.
    monkeypatch.setenv("TESTUARY_STARTER_REUSE_SUITES", "1")
    in_run()
    res2, calls2 = _run_real_cfg(real_gac, root, target, tmp_path / "out2",
                                 suites_dir, [GREEN_A], monkeypatch)
    assert calls2["n"] == 0  # generation skipped purely from the env var
    assert res2.status == "ok"
    assert res2.n_green == res1.n_green


def _run_real_cfg(real_gac, root, target, tests_dir, suites_dir, suites, monkeypatch):
    """Like _run, but with the test_initial settings from the real
    get_agent_config. Only the API settings and the model call are stubbed."""
    from testuary import models
    base_cfg = real_gac("test_initial")
    base_cfg["suite_count"] = len(suites)
    base_cfg["suite_run_timeout_seconds"] = 120
    serve_starter_settings(monkeypatch, base_cfg)
    monkeypatch.setattr(models, "get_api_config",
                        lambda: {"model": "fake", "api_url": "", "api_key": "x"})
    calls = {"n": 0}

    def gen(prompt, *, temperature, seed, timeout):
        idx = calls["n"]
        calls["n"] += 1
        item = suites[idx] if idx < len(suites) else suites[-1]
        if isinstance(item, Exception):
            raise item
        return _fenced(item), {"input_tokens": 100, "output_tokens": 50}

    with model_source.installed(SimpleNamespace(generate_suite=gen)):
        return starters.run_starter_phase(
            language="python", target_file_path=str(target),
            source_dir=str(root), test_output_dir=str(tests_dir),
            suites_dir=str(suites_dir),
            log=lambda *a, **k: None,
        ), calls


# ---------------------------------------------------------------------------
# A merged suite that was never measured is never delivered
#
# The boundary compares a candidate's kills with the pool's by subtraction.
# An unmeasured candidate holds no kills, which would read as total loss, so
# it never reaches the comparison.
# ---------------------------------------------------------------------------

def _km(*killers):
    return {"mutants": [
        {"id": f"m{i}", "status": "killed", "killer": k}
        for i, k in enumerate(killers, start=1)
    ]}


def test_an_unmeasured_remerge_never_replaces_the_measured_merge(
        tmp_path, monkeypatch):
    """The localized-autouse re-merge probes the same sources. If its
    measurement gives no matrix, the measured merged suite is kept, so a
    residual gap does not become the whole requirement."""
    import json
    from contextlib import contextmanager

    from testuary import config
    from testuary import measurement as test_metrics

    root, target, tests_dir, suites_dir = _project(tmp_path)
    matrices = tmp_path / "matrices"
    seed_a = ("from calc import add\n\n"
              "def test_alpha():\n    assert add(1, 1) == 2\n")
    seed_b = ("from calc import add\n\n"
              "def test_beta():\n    assert add(2, 2) == 4\n")

    # Each merging measures in its own directory, the second is the re-merge's.
    consolidations: list = []

    @contextmanager
    def fake_runner(**kw):
        def _measure(d):
            files = sorted(p.name for p in Path(d).glob("*.py"))
            if len([f for f in files if f.startswith("test_seed")]) > 1:
                # The pool holds both kills, so the requirement is two.
                return _km("test_suite_001.py::test_alpha",
                           "test_suite_002.py::test_beta")
            if Path(d).name != "merged":
                return _km("test_calc.py::test_alpha")
            consolidations.append(d)
            if len(consolidations) > 1:
                # The re-merge's measurement refuses.
                return None
            # The measured merged suite has one of the two kills: the residual
            # gap that fails the first selection and starts the re-merge.
            return _km("test_calc.py::test_alpha")
        yield per_suite_double(_measure)

    monkeypatch.setattr(test_metrics, "union_mutation_runner", fake_runner)
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: True)

    res, _ = _run(root, target, tests_dir, suites_dir, [seed_a, seed_b],
                  monkeypatch, matrices_dir=str(matrices))

    assert res.status == "ok", res.detail
    record = json.loads(
        (matrices / "consolidation_record.json").read_text(encoding="utf-8"))
    # The measured merged suite is delivered, with its evidence on disk.
    assert record["mutation_measured"] is True
    assert record["shipped_has_matrix"] is True
    assert (matrices / "selected_kill_matrix.json").is_file()
    assert record["delivered_tests"] > 0 and record["merged_tests"] > 0
    # The residual is the measured shortfall, not the requirement size.
    detail = record.get("detail") or ""
    assert "missing kills=2" not in detail, detail


def test_a_measured_suite_wins_when_the_merge_has_no_kill_measurement(
        tmp_path, monkeypatch):
    """A merged suite without a complete mutation measurement is excluded,
    and a fully measured passing starter suite can still win."""
    import json
    from contextlib import contextmanager

    from testuary import config
    from testuary import measurement as test_metrics

    root, target, tests_dir, suites_dir = _project(tmp_path)
    matrices = tmp_path / "matrices"
    seed_a = ("from calc import add\n\n"
              "def test_alpha():\n    assert add(1, 1) == 2\n")

    delivered_measurements = []

    @contextmanager
    def fake_runner(**kw):
        def _measure(d):
            files = sorted(p.name for p in Path(d).glob("*.py"))
            if Path(d).name.startswith("testuary_starter"):
                # The starter suite measured alone, under its staged name.
                return _km(f"{files[0]}::test_alpha")
            if Path(d).name == "candidate2":
                delivered_measurements.append(files)
                return _km("test_calc.py::test_alpha")
            return None
        yield per_suite_double(_measure)

    monkeypatch.setattr(test_metrics, "union_mutation_runner", fake_runner)
    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: True)

    res, _ = _run(root, target, tests_dir, suites_dir, [seed_a],
                  monkeypatch, matrices_dir=str(matrices))

    assert res.status == "ok", res.status
    assert res.baseline_from_single_suite is True
    record = json.loads(
        (matrices / "consolidation_record.json").read_text(encoding="utf-8"))
    assert record["delivered_from"] == "single_suite"
    assert record["winner_source_label"] == "suite_001"
    assert record["shipped_has_matrix"] is True
    selected = json.loads(
        (matrices / "selected_coverage_matrix.json").read_text(
            encoding="utf-8"
        )
    )
    assert selected["source_sha256"] == record["selected_source_sha256"]
    assert selected["execution_name"] == "test_calc.py"
    selected_km = json.loads(
        (matrices / "selected_kill_matrix.json").read_text(encoding="utf-8"))
    assert selected_km["source_sha256"] == record["selected_source_sha256"]
    # The merged suite and its reduction have no kill measurement, and no
    # other suite's matrix is written under the reduction's name.
    assert not (matrices / "reduced_kill_matrix.json").exists()
    assert len(delivered_measurements) == 1
    assert [
        name for name in delivered_measurements[0]
        if name.startswith("test_")
    ] == ["test_calc.py"]
    assert not any(
        name.endswith("_mutation.py") for name in delivered_measurements[0]
    )
    winner = next(
        candidate for candidate in record["selection_audit"]["candidates"]
        if candidate["label"] == "suite_001"
    )
    assert winner["execution_name"] == "test_calc.py"
    assert winner["measurement_identity"]


# ---------------------------------------------------------------------------
# A target validator that judges nothing: unresolved, not a syntax discard
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("status", ["validator_error", "timeout"])
def test_a_validator_that_judges_nothing_leaves_the_suites_unresolved(
        tmp_path, monkeypatch, status):
    """A validator that times out or does not start judges nothing: the
    suites are unresolved, not syntax discards, and no slot is redrawn."""
    patch_validator(monkeypatch, lambda source, real: unjudged(status))
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, calls = _run(root, target, tests_dir, suites_dir,
                      [GREEN_A, COLLIDER], monkeypatch,
                      refill_failed_suites=True, cache_warmup=False)
    assert res.status == "validation_unresolved"
    assert res.n_validation_unresolved == 2 and res.n_discarded_syntax == 0
    assert calls["n"] == 2
    assert "the interpreter did not start" in res.detail
    for i in (1, 2):
        meta = json.loads((suites_dir / f"suite_{i:03d}" / "original.json")
                          .read_text(encoding="utf-8"))
        assert meta["classification"] == "validation_unresolved"
        assert (suites_dir / f"suite_{i:03d}" / "original.py").is_file()


def test_an_unjudged_warmup_suite_requests_no_batch(tmp_path, monkeypatch):
    """When the validator does not judge the warmup suite, the rest of the
    wave is not requested."""
    patch_validator(monkeypatch, lambda source, real: unjudged())
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, calls = _run(root, target, tests_dir, suites_dir,
                      [GREEN_A, COLLIDER, GREEN_A], monkeypatch,
                      cache_warmup=True)
    assert calls["n"] == 1
    assert res.status == "validation_unresolved"
    assert res.n_validation_unresolved == 1


def test_a_validator_that_judges_on_a_second_attempt_costs_no_request(
        tmp_path, monkeypatch):
    """The validation is repeated without drawing a new response."""
    failures = {"left": 1}

    def judge(source, real):
        if failures["left"]:
            failures["left"] -= 1
            return unjudged()
        return real()

    patch_validator(monkeypatch, judge)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, calls = _run(root, target, tests_dir, suites_dir, [GREEN_A],
                      monkeypatch, refill_failed_suites=True)
    assert res.status == "ok" and res.n_green == 1
    assert res.n_validation_unresolved == 0
    assert calls["n"] == 1


def test_a_mixed_wave_keeps_its_judged_suites(tmp_path, monkeypatch):
    """A judged suite stays in the pool, and the unjudged one is counted apart
    without its slot being redrawn."""
    patch_validator(monkeypatch, lambda source, real: (
        unjudged() if "test_clamp_low" in source else real()))
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, calls = _run(root, target, tests_dir, suites_dir,
                      [GREEN_A, COLLIDER], monkeypatch,
                      refill_failed_suites=True, cache_warmup=False)
    assert res.status == "ok" and res.n_green == 1
    assert res.n_validation_unresolved == 1 and res.n_discarded_syntax == 0
    assert calls["n"] == 2


def test_a_resume_validates_the_saved_response_without_the_model(
        tmp_path, monkeypatch):
    """A resume validates an unjudged saved response again without the model."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    with monkeypatch.context() as patch:
        patch_validator(patch, lambda source, real: unjudged())
        first, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A],
                        patch, refill_failed_suites=True)
    assert first.status == "validation_unresolved"
    res, calls = _run(root, target, tests_dir, suites_dir, [GREEN_A],
                      monkeypatch, refill_failed_suites=True,
                      reuse_suites=True)
    assert calls["n"] == 0
    assert res.status == "ok" and res.n_green == 1
    assert res.n_validation_unresolved == 0
    manifest = json.loads((suites_dir / starters._GENERATION_MANIFEST)
                          .read_text(encoding="utf-8"))
    assert manifest["n_validation_unresolved"] == 0


def test_a_reused_pool_without_slot_counts_validates_its_unresolved_suite_again(
        tmp_path, monkeypatch):
    """A pool whose manifest holds neither per-slot counts nor slot
    evidence, with no request records, treats its slots as unspent. With
    the refill on, its unresolved suite is validated again without the
    model."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    with monkeypatch.context() as patch:
        patch_validator(patch, lambda source, real: unjudged())
        first, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A],
                        patch, refill_failed_suites=True)
    assert first.status == "validation_unresolved"
    manifest_path = suites_dir / starters._GENERATION_MANIFEST
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    del manifest["generation_attempts_per_suite"]
    del manifest["suite_slots"]
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    (suites_dir / starters._REQUEST_RECORDS).write_text("", encoding="utf-8")
    res, calls = _run(root, target, tests_dir, suites_dir, [GREEN_A],
                      monkeypatch, refill_failed_suites=True,
                      reuse_suites=True)
    assert calls["n"] == 0
    assert res.status == "ok" and res.n_green == 1
    assert res.n_validation_unresolved == 0


def test_an_unresolved_validation_fails_the_target():
    """The orchestrator records it as an execution failure, so the command
    exits with 1."""
    from testuary import orchestrator as rank_driver

    def run(count):
        return SimpleNamespace(
            starter_result=SimpleNamespace(n_validation_unresolved=count))

    assert rank_driver._unresolved_suite_validation(run(0)) is None
    assert rank_driver._unresolved_suite_validation(SimpleNamespace()) is None
    assert "2 generated starter suite(s) could not be checked" in (
        rank_driver._unresolved_suite_validation(run(2)))


def test_each_attempt_of_a_slot_draws_and_records_its_own_seed(tmp_path, monkeypatch):
    """Each attempt sends its own random seed, the first the slot's seed, and
    each attempt's record keeps the seed it sent."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, record = _run_per_seed(
        root, target, tests_dir, suites_dir,
        {1: [UNSALVAGEABLE, UNSALVAGEABLE, GREEN_A], 2: [GREEN_A]}, monkeypatch,
        suite_count=2, refill_failed_suites=True, attempts_per_suite=4,
    )
    assert res.status == "ok"
    assert record["suites"] == {1: [1001, 1001001, 2001001], 2: [1002]}
    suites = Path(suites_dir)
    slot = suites / "suite_001"
    assert json.loads((slot / "original.json").read_text("utf-8"))["seed"] == 2001001
    for number, seed in ((1, 1001), (2, 1001001)):
        meta = json.loads((slot / starters._ATTEMPTS_DIR / str(number) / "original.json").read_text("utf-8"))
        assert meta["seed"] == seed
    manifest = json.loads((suites / starters._GENERATION_MANIFEST).read_text("utf-8"))
    slots = manifest["suite_slots"]["slots"]
    assert [item["seed"] for item in slots[0]["attempts"]] == [1001, 1001001, 2001001]
    assert [item["seed"] for item in slots[1]["attempts"]] == [1002]


def test_attempt_seeds_do_not_collide_across_slots_attempts_and_runs():
    """Run r of a series gets the seed_base 1000 * r. No two (run, slot,
    attempt) of up to 1000 runs and 1000 slots share a seed, and a slot's
    first attempt keeps seed_base + slot."""
    assert starters.attempt_seed(3000, 7, 1) == 3007
    drawn = {}
    for run in (0, 1, 2, 499, 998, 999):
        for slot in (1, 2, 10, 999, 1000):
            for attempt in (1, 2, 3, 4, 10):
                seed = starters.attempt_seed(1000 * run, slot, attempt)
                assert seed not in drawn, (run, slot, attempt, drawn.get(seed))
                drawn[seed] = (run, slot, attempt)


def test_an_attempts_seed_is_its_runs_slot_seed_plus_the_refill_stride():
    """First attempts send the run's slot seeds (config.run_seed, or
    seed_base + slot for a single run) and later ones add the refill stride:
    no (run, slot, attempt) of up to 1000 runs of 1000 slots shares a seed."""
    from testuary import config

    assert starters.attempt_seed(37, 4, 1) == 41
    assert starters.attempt_seed(37, 4, 1, run_idx=2) == config.run_seed(37, 2, 4) == 2041
    assert starters.attempt_seed(37, 4, 2, run_idx=2) == 2041 + starters.REFILL_SEED_STRIDE
    drawn = {}
    for run_idx in (0, 1, 2, 499, 998, 999):
        for slot in (1, 2, 10, 999, 1000):
            for attempt in (1, 2, 3, 10):
                seed = starters.attempt_seed(37, slot, attempt, run_idx)
                assert seed not in drawn, (run_idx, slot, attempt, drawn.get(seed))
                drawn[seed] = (run_idx, slot, attempt)


def _run_replies(tmp_path, monkeypatch, replies, **cfg_over):
    """Run the phase on one slot whose successive requests return the
    ``(text, usage)`` pairs of ``replies`` (the last one repeats), with no
    retry wait."""
    from testuary import model_requests
    from testuary import config
    from testuary import models

    root, target, tests_dir, suites_dir = _project(tmp_path)
    monkeypatch.setattr(model_requests.RequestAccount, "wait", lambda self, seconds: None)
    base_cfg = config.get_agent_config("test_initial")
    base_cfg.update({"suite_count": 1, "suite_run_timeout_seconds": 120, "seed_base": 1000,
                     "refill_failed_suites": True, "attempts_per_suite": 2})
    base_cfg.update(cfg_over)
    serve_starter_settings(monkeypatch, base_cfg)
    monkeypatch.setattr(models, "get_api_config",
                        lambda: {"model": "fake", "api_url": "", "api_key": "x"})
    calls = []

    def gen(prompt, *, temperature, seed, timeout):
        calls.append(seed)
        text, usage = replies[min(len(calls), len(replies)) - 1]
        return text, dict(usage)

    with model_source.installed(SimpleNamespace(generate_suite=gen)):
        res = starters.run_starter_phase(
            language="python", target_file_path=str(target),
            source_dir=str(root), test_output_dir=str(tests_dir),
            suites_dir=str(suites_dir), log=lambda *a, **k: None,
        )
    records = [json.loads(line) for line in
               (Path(suites_dir) / "request_records.jsonl").read_text("utf-8").splitlines()]
    return res, calls, records


def test_a_starter_load_reply_is_sent_again_with_its_usage_known(tmp_path, monkeypatch):
    """A reply that ended while the provider loaded the model is sent again
    without spending an attempt, and its 0 of 0 tokens are known usage."""
    load = ("", {"input_tokens": 0, "output_tokens": 0, "finish_reason": "load"})
    answer = (_fenced(GREEN_A), {"input_tokens": 100, "output_tokens": 50})
    res, calls, records = _run_replies(tmp_path, monkeypatch, [load, answer])
    assert res.status == "ok"
    assert calls == [1001, 1001]
    assert res.suite_generation_attempts == {1: 1}
    assert (res.requests_used, res.logical_requests, res.unknown_usage_requests) == (2, 1, 0)
    assert [(item["outcome"], item["usage_known"]) for item in records] == [
        ("transient", True), ("delivered", True)]
    assert (records[0]["input_tokens"], records[0]["output_tokens"]) == (0, 0)


def test_an_empty_starter_reply_that_read_the_prompt_spends_its_attempt(tmp_path, monkeypatch):
    """An empty reply that read the prompt is an answer without code: it
    spends the slot's attempt and is neither resent nor a request failure."""
    stop = ("", {"input_tokens": 812, "output_tokens": 0, "finish_reason": "stop"})
    res, calls, records = _run_replies(tmp_path, monkeypatch, [stop])
    assert calls == [1001, 1001001]
    assert res.status == "cancelled_zero_passing"
    assert res.suite_generation_attempts == {1: 2}
    assert [item["outcome"] for item in records] == ["delivered", "delivered"]
    assert res.unknown_usage_requests == 0


def test_the_zero_green_reason_lists_the_rejected_suites_it_counted():
    """The reason names, among the generated suites, the ones each check
    rejected, the suites that covered nothing among them."""
    res = starters.StarterResult(n_generated=5, n_discarded_syntax=1, n_zero_coverage=2)
    assert starters._zero_green_detail(res) == (
        "no green tests from 5 generated suites (among them 1 with a syntax "
        "error, 2 that covered nothing)")
    assert starters._zero_green_detail(starters.StarterResult()) == (
        "no green tests from 0 generated suites")


def test_what_the_starter_phase_holds_is_freed_when_the_phase_returns(tmp_path, monkeypatch):
    """What the phase's object holds goes when the phase returns, without the
    cyclic garbage collector: its request account, which refers back to the
    object, is freed at once."""
    import gc
    import weakref

    accounts = []
    real_open_records = starters._StarterPhase.open_records

    def open_records(self):
        real_open_records(self)
        accounts.append(weakref.ref(self.account))

    monkeypatch.setattr(starters._StarterPhase, "open_records", open_records)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    gc.disable()
    try:
        res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch)
        assert res.status == "ok"
        assert len(accounts) == 1 and accounts[0]() is None
    finally:
        gc.enable()


# ---------------------------------------------------------------------------
# Paths no other test reaches: each pins what the phase does today.
# ---------------------------------------------------------------------------

def test_a_language_without_an_adapter_is_refused_before_any_record(tmp_path, monkeypatch):
    """A language without an adapter is refused before the phase writes a
    record or sends a request."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    monkeypatch.setattr(starters.languages, "get_backend", lambda language: None)
    with pytest.raises(ValueError, match="no language adapter for 'python'"):
        _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch)
    assert not suites_dir.exists()


def test_a_fresh_generation_removes_a_response_an_earlier_run_left(tmp_path, monkeypatch):
    """A fresh generation starts a new ledger: a saved response an earlier
    run left unprocessed, here in a slot this run does not draw, goes."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    stale = suites_dir / "suite_002" / starters._GENERATION_DELIVERY
    stale.parent.mkdir(parents=True)
    stale.write_text("{}", encoding="utf-8")
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch)
    assert res.status == "ok"
    assert not stale.exists()


def test_a_reuse_refuses_a_saved_response_it_cannot_read(tmp_path, monkeypatch):
    """A saved response that does not parse stops the reuse, naming it."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    first, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch)
    assert first.status == "ok"
    saved = suites_dir / "suite_001" / starters._GENERATION_DELIVERY
    saved.write_text("{not json", encoding="utf-8")
    with pytest.raises(RuntimeError, match="reuse_suites is set but the saved response"):
        _run(root, target, tmp_path / "tests_out2", suites_dir, [GREEN_A], monkeypatch,
             reuse_suites=True)


def test_a_reuse_refuses_an_unresolved_suite_it_cannot_read(tmp_path, monkeypatch):
    """An unresolved suite whose saved source is gone stops the reuse that
    would validate it again, naming the slot."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    with monkeypatch.context() as patch:
        patch_validator(patch, lambda source, real: unjudged())
        first, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A],
                        patch, refill_failed_suites=True)
    assert first.status == "validation_unresolved"
    (suites_dir / "suite_001" / "original.py").unlink()
    with pytest.raises(RuntimeError, match="the unvalidated starter suite 1 cannot be read"):
        _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch,
             refill_failed_suites=True, reuse_suites=True)


def _merge_unavailable(monkeypatch):
    """The merged and reduced suites out of contention, so that the single
    starter suite is the candidate the selection delivers."""
    from testuary import config
    from testuary import consolidation as consolidate_mod

    monkeypatch.setattr(config, "consolidation_measure_mutation", lambda: False)
    monkeypatch.setattr(
        consolidate_mod, "consolidate_phase_end",
        lambda *a, **k: consolidate_mod.PhaseConsolidation(detail="forced: merge unsalvageable"))
    return consolidate_mod


def test_a_delivered_starter_suite_whose_acceptance_coverage_differs_delivers_that_coverage(
        tmp_path, monkeypatch):
    """A starter suite delivered alone whose acceptance run under the
    delivered filename measures other coverage than its selection evidence
    delivers the acceptance run's coverage, and the record keeps both
    sides."""
    _merge_unavailable(monkeypatch)
    real_materialize = starters._StarterPhase.materialize

    def materialize(self, *args):
        # The suite's evidence from its measurement under the delivered
        # filename, one point higher than the final acceptance run gives.
        real_materialize(self, *args)
        for candidate in self.all_candidates:
            evidence = candidate.coverage_evidence
            if candidate.label.startswith(starters.SLOT_PREFIX) and evidence:
                candidate.coverage_evidence = {**evidence, "line": evidence["line"] + 1.0}

    monkeypatch.setattr(starters._StarterPhase, "materialize", materialize)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    matrices = tmp_path / "matrices"
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch,
                  matrices_dir=matrices)
    assert res.status == "ok" and res.winner == "single_suite"
    record = json.loads((matrices / "consolidation_record.json").read_text(encoding="utf-8"))
    selected = json.loads(
        (matrices / "selected_coverage_matrix.json").read_text(encoding="utf-8"))
    difference = record["selection_audit"]["delivered_coverage_difference"]
    assert difference["selection"]["line"] == difference["acceptance"]["line"] + 1.0
    assert selected["line"] == difference["acceptance"]["line"] == res.baseline_line_coverage


def test_a_delivered_starter_suite_whose_acceptance_run_measured_no_coverage_stops_the_phase(
        tmp_path, monkeypatch):
    """A starter suite delivered alone whose acceptance run under the
    delivered filename passed but measured no coverage has no values to
    deliver: the phase stops with an error."""
    consolidate_mod = _merge_unavailable(monkeypatch)
    real_check = consolidate_mod._check_then_measure
    root, target, tests_dir, suites_dir = _project(tmp_path)

    def unmeasured(backend, file_paths, source_dir, **kwargs):
        run, ran = real_check(backend, file_paths, source_dir, **kwargs)
        if tests_dir.resolve() in Path(file_paths[0]).resolve().parents:
            return run._replace(line_coverage=None, branch_coverage=None, covered_lines=None,
                                covered_branches=None), ran
        return run, ran

    monkeypatch.setattr(consolidate_mod, "_check_then_measure", unmeasured)
    with pytest.raises(RuntimeError, match="final acceptance coverage disagrees"):
        _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch,
             matrices_dir=tmp_path / "matrices")


def test_a_target_without_lines_is_delivered_without_line_coverage(tmp_path, monkeypatch):
    """A target without a statement has no line coverage, as the language
    reports it: its starter suite records none, and the final acceptance
    run, which reports none too, agrees with it."""
    _merge_unavailable(monkeypatch)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    target.write_text("", encoding="utf-8")
    suite = "import calc\n\n\ndef test_the_module_imports():\n    assert calc.__name__ == 'calc'\n"
    res, _ = _run(root, target, tests_dir, suites_dir, [suite], monkeypatch,
                  matrices_dir=tmp_path / "matrices")
    assert res.status == "ok" and res.winner == "single_suite"
    assert res.suite_line_stats == {}


@pytest.mark.parametrize("passed, status, matrices", [
    (True, "incomplete", True), (False, "failing", False)])
def test_a_winner_excluded_under_the_delivered_filename_fails_the_selection(
        tmp_path, monkeypatch, passed, status, matrices):
    """A selected suite measured under the delivered filename, whose baseline
    file the coverage tool then cannot measure, is excluded as incomplete,
    and one whose baseline file fails as failing. With no other candidate
    the phase fails its selection, and the baseline it wrote goes."""
    consolidate_mod = _merge_unavailable(monkeypatch)
    real_check = consolidate_mod._check_then_measure
    excluded = []
    root, target, tests_dir, suites_dir = _project(tmp_path)

    def rejected(backend, file_paths, source_dir, **kwargs):
        run, ran = real_check(backend, file_paths, source_dir, **kwargs)
        if tests_dir.resolve() in Path(file_paths[0]).resolve().parents:
            return run._replace(ok=False), passed
        return run, ran

    monkeypatch.setattr(consolidate_mod, "_check_then_measure", rejected)
    real_audit = consolidate_mod.record_selection_audit

    def audit(phase, candidates, winner, **kwargs):
        excluded.extend(c.status for c in candidates if c.label == "suite_001")
        return real_audit(phase, candidates, winner, **kwargs)

    monkeypatch.setattr(consolidate_mod, "record_selection_audit", audit)
    matrices_dir = tmp_path / "matrices" if matrices else None
    lines = []
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch,
                  matrices_dir=matrices_dir, log=lambda line, *a, **k: lines.append(line))
    assert res.status == "failed_phase_selection" and res.baseline_file is None
    assert excluded == [status]
    # The log line names the delivered filename once, through the reason.
    [line] = [line for line in lines if "was excluded" in line]
    assert line.startswith("[starter] candidate 'suite_001' was excluded: ")
    assert line.count("under the delivered filename") == 1
    assert not list(tests_dir.rglob("*.py"))
    assert (matrices_dir is None) or (matrices_dir / "consolidation_record.json").is_file()


def test_a_second_request_failure_in_a_wave_ends_the_phase_on_the_first(tmp_path, monkeypatch):
    """Both requests of one wave end on a model request failure. The phase
    ends on the one its wave collects first, and no slot is asked again."""
    import threading

    both_sent = threading.Barrier(2, timeout=60)
    asked = []

    def generate(prompt, *, temperature, seed, timeout):
        asked.append(seed)
        both_sent.wait()
        raise RuntimeError("refused")

    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A, GREEN_A], monkeypatch,
                  generate=generate, cache_warmup=False, concurrent_requests=2)
    assert res.status == "failed_model_request"
    assert "RuntimeError: refused" in res.detail
    assert len(asked) == 2


def _progress_write_fails(monkeypatch):
    def persist_progress(self):
        raise OSError("the progress write failed")

    monkeypatch.setattr(starters._StarterPhase, "persist_progress", persist_progress)


def test_a_worker_failure_ends_the_phase_when_its_progress_write_fails_too(
        tmp_path, monkeypatch):
    """A worker failure that is no model request failure, here a response
    that cannot be saved, ends the phase with its own error, also when the
    progress write after it fails."""
    real_write = starters.validity.write_json_atomic

    def write_json_atomic(path, *args, **kwargs):
        if Path(path).name == starters._GENERATION_DELIVERY:
            raise OSError("the response could not be saved")
        return real_write(path, *args, **kwargs)

    monkeypatch.setattr(starters.validity, "write_json_atomic", write_json_atomic)
    _progress_write_fails(monkeypatch)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    with pytest.raises(OSError, match="the response could not be saved"):
        _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch)


def test_an_interrupt_during_a_wave_ends_the_phase_when_its_progress_write_fails(
        tmp_path, monkeypatch):
    """A Ctrl+C while the phase waits for a wave ends it as an interruption,
    also when the progress write after it fails."""
    def interrupted(futures):
        raise KeyboardInterrupt
        yield

    monkeypatch.setattr(starters, "_completed_in_slices", interrupted)
    _progress_write_fails(monkeypatch)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    with pytest.raises(KeyboardInterrupt):
        _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch)


def test_a_refill_for_a_suite_without_coverage_evidence_ends_on_a_request_failure(
        tmp_path, monkeypatch):
    """The slot a suite without coverage evidence frees is generated again.
    A model request failure there ends the phase."""
    _no_coverage_evidence_of(monkeypatch, "test_beta")
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, calls = _run(root, target, tests_dir, suites_dir,
                      [_ALPHA, _BETA, RuntimeError("refill refused")], monkeypatch,
                      suite_count=2, refill_failed_suites=True, attempts_per_suite=2)
    assert res.status == "failed_model_request"
    assert "RuntimeError: refill refused" in res.detail
    assert calls["n"] == 3 and res.n_unmeasured_coverage == 1


def test_a_refill_for_a_suite_the_mutation_tool_measured_nothing_of_ends_on_a_request_failure(
        tmp_path, monkeypatch):
    """The slot a suite without a mutation result frees is generated again.
    A model request failure there ends the phase."""
    _mutation_tool_measures_nothing_of(monkeypatch, "test_beta")
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, calls = _run(root, target, tests_dir, suites_dir,
                      [_ALPHA, _BETA, RuntimeError("refill refused")], monkeypatch,
                      matrices_dir=tmp_path / "matrices", suite_count=2,
                      refill_failed_suites=True, attempts_per_suite=2)
    assert res.status == "failed_model_request"
    assert "RuntimeError: refill refused" in res.detail
    assert calls["n"] == 3 and res.n_unmeasured_mutation == 1


def _ended_early(tmp_path, monkeypatch):
    """A generation without the refill that a model request failure ended
    after slot 1's response and before slot 2's."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    first, _ = _run(root, target, tests_dir, suites_dir, [_ALPHA, RuntimeError("refused")],
                    monkeypatch, cache_warmup=False, concurrent_requests=1,
                    refill_failed_suites=False)
    assert first.status == "failed_model_request"
    return root, target, tests_dir, suites_dir


def test_a_reuse_of_a_generation_that_ended_early_generates_the_slots_without_a_response(
        tmp_path, monkeypatch):
    """Reusing a generation that ended early generates each slot that never
    received a response, as the uninterrupted run would have, and keeps the
    others."""
    root, target, tests_dir, suites_dir = _ended_early(tmp_path, monkeypatch)
    res, calls = _run(root, target, tests_dir, suites_dir, [_BETA], monkeypatch, suite_count=2,
                      reuse_suites=True, cache_warmup=False, refill_failed_suites=False)
    assert res.status == "ok"
    assert calls["n"] == 1 and res.n_green == 2


def test_a_reuse_of_a_generation_that_ended_early_ends_on_a_new_request_failure(
        tmp_path, monkeypatch):
    """A model request failure while a reuse generates the slots without a
    response ends the phase again."""
    root, target, tests_dir, suites_dir = _ended_early(tmp_path, monkeypatch)
    res, calls = _run(root, target, tests_dir, suites_dir, [RuntimeError("refused again")],
                      monkeypatch, suite_count=2, reuse_suites=True, cache_warmup=False,
                      refill_failed_suites=False)
    assert res.status == "failed_model_request"
    assert "RuntimeError: refused again" in res.detail and calls["n"] == 1


def _interrupted_after_every_response(tmp_path, monkeypatch):
    """A generation of two slots that a Ctrl+C ended after both responses
    arrived and before they were processed: both are saved."""
    real_slices = starters._completed_in_slices

    def then_interrupted(futures):
        yield from real_slices(futures)
        raise KeyboardInterrupt

    root, target, tests_dir, suites_dir = _project(tmp_path)
    with monkeypatch.context() as patch:
        patch.setattr(starters, "_completed_in_slices", then_interrupted)
        with pytest.raises(KeyboardInterrupt):
            _run(root, target, tests_dir, suites_dir, [_ALPHA, _BETA], patch,
                 cache_warmup=False, concurrent_requests=2, refill_failed_suites=False)
    return root, target, tests_dir, suites_dir


def _reuse_two_slots(paths, monkeypatch):
    return _run(*paths, [_GAMMA], monkeypatch, suite_count=2, reuse_suites=True,
                cache_warmup=False, refill_failed_suites=False)


def test_a_reuse_after_an_interrupt_once_every_response_arrived_asks_the_model_nothing(
        tmp_path, monkeypatch):
    """A Ctrl+C after every response of the wave arrived ends the generation
    early with no slot left without a response. The reuse processes the
    saved responses and asks the model nothing."""
    paths = _interrupted_after_every_response(tmp_path, monkeypatch)
    res, calls = _reuse_two_slots(paths, monkeypatch)
    assert res.status == "ok"
    assert calls["n"] == 0 and res.n_green == 2


def _observed(monkeypatch):
    from testuary import _progress

    events = []
    monkeypatch.setattr(_progress, "phase_observer", lambda phase: (
        lambda **event: events.append((event["operation"], event["state"]))))
    return events


def test_a_reuse_reports_each_saved_response_it_processes(tmp_path, monkeypatch):
    """The run's observer sees each saved response the reuse processes as
    running, then succeeded, without a model request."""
    paths = _interrupted_after_every_response(tmp_path, monkeypatch)
    events = _observed(monkeypatch)
    res, calls = _reuse_two_slots(paths, monkeypatch)
    assert res.status == "ok" and calls["n"] == 0
    saved = [event for event in events if event[0].startswith("saved-response/")]
    assert saved == [("saved-response/1", "running"), ("saved-response/1", "succeeded"),
                     ("saved-response/2", "running"), ("saved-response/2", "succeeded")]


def test_an_interrupt_while_a_saved_response_is_processed_is_reported_and_ends_the_reuse(
        tmp_path, monkeypatch):
    """A Ctrl+C while the reuse processes a saved response ends the phase,
    and the observer sees that response's processing as failed. The
    response stays saved for the next reuse."""
    paths = _interrupted_after_every_response(tmp_path, monkeypatch)
    events = _observed(monkeypatch)

    def interrupted(self, *args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(starters._StarterPhase, "process_response", interrupted)
    with pytest.raises(KeyboardInterrupt):
        _reuse_two_slots(paths, monkeypatch)
    saved = [event for event in events if event[0].startswith("saved-response/")]
    assert saved == [("saved-response/1", "running"), ("saved-response/1", "failed")]
    assert (paths[3] / "suite_001" / starters._GENERATION_DELIVERY).is_file()


def test_a_suite_without_per_test_evidence_whose_coverage_run_fails_frees_its_slot(
        tmp_path, monkeypatch):
    """A suite the backend gives no per-test evidence for is measured by its
    own coverage run. When that run fails, the suite has no complete numbers:
    its slot is freed, and its record names the run's verdict."""
    _no_coverage_evidence_of(monkeypatch, "test_add_unmeasurable", failed=False)
    _coverage_tool_fails_on(monkeypatch, "test_add_unmeasurable", alone=True)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A, UNMEASURABLE], monkeypatch,
                  refill_failed_suites=False)
    assert res.status == "ok"
    assert res.n_green == 1 and res.n_unmeasured_coverage == 1
    meta = json.loads((suites_dir / "suite_002" / "original.json").read_text("utf-8"))
    assert meta["classification"] == "coverage_unmeasured"
    reason = meta["cleanup"]["disposition_reason"]
    assert reason.startswith("the coverage tool could not measure the suite alone: ")
    assert "Failing: test_add_unmeasurable." in reason


def test_a_pool_its_measurement_empties_ends_the_phase_without_a_green_suite(
        tmp_path, monkeypatch):
    """When the measurement frees every starter suite and no slot has an
    attempt left, no suite is green: the phase ends as cancelled, and its
    detail counts the suite the coverage tool could not measure."""
    _no_coverage_evidence_of(monkeypatch, "test_add_unmeasurable", failed=False)
    _coverage_tool_fails_on(monkeypatch, "test_add_unmeasurable", alone=True)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [UNMEASURABLE], monkeypatch,
                  refill_failed_suites=False)
    assert res.status == "cancelled_zero_passing"
    assert res.n_unmeasured_coverage == 1
    assert "1 that the coverage tool could not measure" in res.detail


def test_derive_measures_each_suite_when_the_pooled_run_gives_no_result(tmp_path, monkeypatch):
    """When the pooled mutation run gives no result, derive measures each
    suite alone, reports measure as the method used and records why no
    derivation was made."""
    res, matrices, pooled, per_suite = _starter_pool_run(tmp_path, monkeypatch, "derive",
                                                         pooled_nothing=True)
    assert len(pooled) == 1 and per_suite == [["suite_001", "suite_002"]]
    assert res.suite_mutation_method == "measure"
    record = json.loads((matrices / "starter_pool_derivation.json").read_text(encoding="utf-8"))
    assert (record["method_requested"], record["method_used"]) == ("derive", "measure")
    assert record["derivation_unavailable"] == "the pooled mutation run produced no result"


def test_derive_leaves_a_suite_without_per_test_evidence_underived(tmp_path, monkeypatch):
    """Under derive, a suite the backend gives no per-test evidence for has
    no tests to take from the pooled run: it gets no derived result and is
    scored on its own coverage run, and the other suite is derived."""
    _no_coverage_evidence_of(monkeypatch, "test_beta", failed=False)
    res, matrices, pooled, per_suite = _starter_pool_run(tmp_path, monkeypatch, "derive")
    assert len(pooled) == 1
    record = json.loads((matrices / "starter_pool_derivation.json").read_text(encoding="utf-8"))
    assert [slot["derived"] is None for slot in record["slots"]] == [False, True]


def test_derive_leaves_a_suite_whose_evidence_names_no_test_underived(tmp_path, monkeypatch):
    """Under derive, a suite whose per-test evidence names no test has no
    test that ran in its own coverage run: it gets no derived result, and the
    other suite is derived."""
    import dataclasses

    from testuary.languages import get_backend

    backend_type = type(get_backend("python"))
    collect = backend_type.collect_eviction_data

    def no_tests(self, files, *args, **kwargs):
        data = collect(self, files, *args, **kwargs)
        if data is not None and _alone(files, "test_beta"):
            return dataclasses.replace(data, per_test_tokens={}, per_test_time_ms={},
                                       first_test=None)
        return data

    monkeypatch.setattr(backend_type, "collect_eviction_data", no_tests)
    _res, matrices, pooled, _per_suite = _starter_pool_run(tmp_path, monkeypatch, "derive")
    assert len(pooled) == 1
    record = json.loads((matrices / "starter_pool_derivation.json").read_text(encoding="utf-8"))
    assert [slot["derived"] is None for slot in record["slots"]] == [False, True]
    assert record["slots"][1]["derivation_unavailable"] == "no test of this suite ran in its own coverage run"


@pytest.mark.parametrize("record", ["{not json", "[]"], ids=["unreadable", "no-object"])
def test_a_freed_slot_whose_suite_record_is_unreadable_or_no_object_is_refilled(
        tmp_path, monkeypatch, record):
    """A slot freed after its suite measured nothing keeps a suite record it
    cannot read, or one that holds no object, as it is, and the slot is
    generated again. No readable record names the earlier attempt, so its
    files are not moved under attempts/ and the new attempt's replace them."""
    _mutation_tool_measures_nothing_of(monkeypatch, "test_beta")
    real_freed = starters._StarterPhase.mutation_freed

    def mutation_freed(self):
        freed = real_freed(self)
        for i in freed:
            (self.suites_root / starters.slot_name(i) / "original.json").write_text(
                record, encoding="utf-8")
        return freed

    monkeypatch.setattr(starters._StarterPhase, "mutation_freed", mutation_freed)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, calls = _run(root, target, tests_dir, suites_dir, [_ALPHA, _BETA, _GAMMA], monkeypatch,
                      matrices_dir=tmp_path / "matrices", suite_count=2,
                      refill_failed_suites=True, attempts_per_suite=2)
    assert res.status == "ok"
    assert calls["n"] == 3 and (res.n_green, res.n_unmeasured_mutation) == (2, 1)
    latest = json.loads((suites_dir / "suite_002" / "original.json").read_text(encoding="utf-8"))
    assert latest["slot_attempt"] == 2
    assert not (suites_dir / "suite_002" / "attempts").exists()


@pytest.mark.parametrize("count, selected", [(7, 7), (None, 2)], ids=["counted", "uncounted"])
def test_a_materialized_starter_suite_without_a_test_count_keeps_its_scored_count(
        tmp_path, monkeypatch, count, selected):
    """A starter suite measured under the delivered filename gives the
    delivered test count. Without a count of its own, the count its scoring
    gave stays."""
    consolidate_mod = _merge_unavailable(monkeypatch)
    real_measure = consolidate_mod.measure_candidate_standalone

    def measure(*args, **kwargs):
        measured = real_measure(*args, **kwargs)
        measured.test_count = count
        return measured

    monkeypatch.setattr(consolidate_mod, "measure_candidate_standalone", measure)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch)
    assert res.status == "ok" and res.winner == "single_suite"
    assert res.selected_tests == selected


def test_a_starter_reply_without_a_usage_mapping_counts_as_unknown_usage(tmp_path, monkeypatch):
    """A generator that reports no usage mapping delivers its answer, and the
    request counts as one with unknown usage."""
    def generate(prompt, *, temperature, seed, timeout):
        return _fenced(GREEN_A), None

    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch, generate=generate)
    assert res.status == "ok"
    assert res.unknown_usage_requests == 1
    records = [json.loads(line) for line in
               (suites_dir / "request_records.jsonl").read_text("utf-8").splitlines()]
    assert [(item["outcome"], item["usage_known"]) for item in records] == [("delivered", False)]


def test_a_starter_request_of_a_model_source_records_the_source_as_its_transport(
        tmp_path, monkeypatch):
    """An installed model source answers without a phase model, so the
    transport record names the model source, as the agents' records do."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch)
    assert res.status == "ok"
    records = [json.loads(line) for line in
               (suites_dir / "prompt_transport.jsonl").read_text("utf-8").splitlines()]
    assert [record["provider_transport"] for record in records] == ["model_source"]


@pytest.mark.parametrize("mode", ["local", "container"])
def test_the_starter_phase_resolves_its_project_paths_only_on_this_computer(
        tmp_path, monkeypatch, mode):
    """Test runs start in the source directory, so on this computer the phase
    resolves the target, the source directory and the test directory it is given.
    In a container run they are the container's own paths and stay as given.
    The suites directory is on this computer in both."""
    import types
    from testuary import config
    from testuary import models

    # The execution mode as the starter phase alone sees it, so that no
    # other module runs anything in a container.
    phase_config = types.SimpleNamespace(**{name: getattr(config, name) for name in dir(config)
                                            if not name.startswith("__")})
    phase_config.project_exec_mode = lambda: mode
    monkeypatch.setattr(starters, "config", phase_config)
    monkeypatch.setattr(models, "get_api_config",
                        lambda: {"model": "fake", "api_url": "", "api_key": "x"})
    root, target, tests_dir, suites_dir = _project(tmp_path)
    monkeypatch.chdir(tmp_path)
    given = [str(path.relative_to(tmp_path)) for path in (target, root, tests_dir, suites_dir)]
    phase = starters._StarterPhase(
        language="python", target_file_path=given[0], source_dir=given[1],
        test_output_dir=given[2], suites_dir=given[3], matrices_dir=None,
        log=lambda *a, **k: None, prepared_models=None, run_idx=0,
        reuse_suites=None)
    phase.prepare()
    resolved = [str((tmp_path / path).resolve()) for path in given]
    assert [phase.target_file_path, phase.source_dir, phase.test_output_dir] == (
        given[:3] if mode == "container" else resolved[:3])
    assert phase.suites_dir == resolved[3]


def test_an_accepted_suite_whose_validity_screening_does_not_complete_stays_unresolved(
        tmp_path, monkeypatch):
    """A cleaned suite is accepted, and then the validity policy's screening
    of it does not complete. The suite stays unresolved, and its record
    names no accepted source: cleanup did not deliver one."""
    from testuary import languages
    from testuary.consolidation import ValidityUnresolved

    def screen(self, source, target_identity):
        raise ValidityUnresolved("screen_source: the screening did not complete")

    monkeypatch.setattr(type(languages.get_backend("python")), "screen_generated_test_source",
                        screen)
    root, target, tests_dir, suites_dir = _project(tmp_path)
    res, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch)
    assert res.status == "validation_unresolved"
    meta = json.loads((suites_dir / "suite_001" / "original.json").read_text("utf-8"))
    assert meta["classification"] == "validation_unresolved"
    assert (meta["cleanup"]["disposition_basis"], meta["cleanup"]["disposition_reason"]) == (
        "unavailable", "cleanup did not deliver an accepted source")


def test_a_saved_response_without_an_id_or_code_spends_its_slot_on_reuse(tmp_path, monkeypatch):
    """A saved response file that lacks its id, as a damaged file can, and
    whose text holds no code is processed by the reuse as a delivered answer
    without code: its slot stays empty, the saved file goes, and without an
    id the slot's record names no paid attempt."""
    paths = _interrupted_after_every_response(tmp_path, monkeypatch)
    saved = paths[3] / "suite_002" / starters._GENERATION_DELIVERY
    delivery = json.loads(saved.read_text(encoding="utf-8"))
    del delivery["id"]
    delivery["text"] = "no code in this answer"
    saved.write_text(json.dumps(delivery), encoding="utf-8")
    res, calls = _reuse_two_slots(paths, monkeypatch)
    assert res.status == "ok" and calls["n"] == 0 and res.n_green == 1
    assert not saved.exists()
    slot = _suite_slots(paths[3])[1]
    assert (slot["state"], slot["classification"], slot["reason"]) == (
        "unavailable", "no_code", "paid attempt identity or judgement unavailable")


def test_a_reuse_whose_manifest_is_gone_after_its_suites_loaded_restores_the_pool_without_it(
        tmp_path, monkeypatch):
    """The generation manifest is gone after the reuse loaded the archived
    suites and before it reads the manifest's records, as when another
    process removed it. The suites and the request records restore the pool,
    and the model is not asked."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    first, _ = _run(root, target, tests_dir, suites_dir, [GREEN_A], monkeypatch)
    assert first.status == "ok"
    real_load = starters._load_persisted_green

    def load(suites_root, *args, **kwargs):
        restored = real_load(suites_root, *args, **kwargs)
        (Path(suites_root) / starters._GENERATION_MANIFEST).unlink()
        return restored

    monkeypatch.setattr(starters, "_load_persisted_green", load)
    res, calls = _run(root, target, tmp_path / "tests_out2", suites_dir, [GREEN_A], monkeypatch,
                      reuse_suites=True)
    assert res.status == "ok"
    assert calls["n"] == 0 and res.n_green == 1


def test_a_reused_suite_dropped_as_empty_marks_its_slot_without_adding_generation_time(
        tmp_path, monkeypatch):
    """A reused suite that now runs no test leaves the pool. Without the
    refill nothing is generated, so the manifest only records the slot as
    unavailable and keeps the generation time."""
    root, target, tests_dir, suites_dir = _project(tmp_path)
    first, _ = _run(root, target, tests_dir, suites_dir, [_ALPHA, _BETA], monkeypatch,
                    refill_failed_suites=False)
    assert first.status == "ok" and first.n_green == 2
    manifest_path = suites_dir / starters._GENERATION_MANIFEST
    seconds = json.loads(manifest_path.read_text(encoding="utf-8"))["generation_seconds"]
    for cleaned in (suites_dir / "suite_002").glob("cleaned.*"):
        if cleaned.suffix == ".py":
            cleaned.write_text("from calc import add\n", encoding="utf-8")
    res, calls = _run(root, target, tmp_path / "tests_out2", suites_dir, [_GAMMA], monkeypatch,
                      suite_count=2, reuse_suites=True, refill_failed_suites=False)
    assert res.status == "ok"
    assert calls["n"] == 0 and res.n_green == 1
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["generation_seconds"] == seconds
    assert _suite_slots(suites_dir)[1]["state"] == "unavailable"


def test_a_starter_suite_the_origin_check_stopped_names_its_stopping_tests():
    """A suite the mutation tool's origin check stopped leaves the starter
    suites with the reason and every stopping test's name, and another
    unmeasured suite keeps its outcome."""
    first, second = starters.slot_name(1), starters.slot_name(2)
    phase = SimpleNamespace(order=[1, 2], measured={
        "unmeasured": [first, second],
        "outcomes": {first: "test_failures", second: "infrastructure_failure"},
        "stopped": {first: "the suite failed (stopping tests: t.c::a, t.c::b)"}})
    assert starters._StarterPhase.mutation_freed(phase) == {
        1: ("mutation_unmeasured", "the mutation tool's origin check stopped the suite: "
                                   "the suite failed (stopping tests: t.c::a, t.c::b)"),
        2: ("mutation_unmeasured",
            "the mutation tool measured nothing of the suite (infrastructure_failure)")}
