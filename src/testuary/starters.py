"""The starter phase: best-of-N starter suites before the coverage agent.

The first of a target's phases. N independent single-prompt generations
(no tools, no agent loop) produce candidate suites. Each is salvaged to a
green subset by deterministic means only (import prune for collection
errors, per-test strip for failures). The green pool is merged, measured,
and reduced with runtime-weighted set cover. The reduced suite, merged suite,
and individual suites then enter the common phase-end selector. Its winner is
handed to the coverage agent as a non-frozen starting suite. The coverage
agent is the repair route for anything the pool lacks. This phase never spends
LLM calls on fixing.

Per-suite artifacts persist under ``suites_dir`` (original<suffix> /
original.json / cleaned<suffix> / cleaned.json with the per-test token
sets), so every production run yields the coverage-growth curve of its
N starter suites as free post-processing. A slot directory holds its latest processed
attempt, the accepted one for an accepted slot. The files of each earlier
attempt of the slot are kept beside them in ``attempts/<n>/``, n being that
attempt's number within its slot (``_ATTEMPTS_DIR``).

Every request sent, including a failed attempt, is recorded and counted
in ``requests_used``. ``logical_requests`` counts model requests, the
opportunities the population algorithm grants. A transient transport failure
is retried for the same model request under the shared request policy and
spends no refill attempt. A terminal failure, or a retry ceiling, ends the
phase with a failure status. Parallel workers compress wall-clock only,
never the limits.
"""
from __future__ import annotations

import concurrent.futures
import contextlib
import copy
import json
import re
import statistics
import threading
import time
import uuid
from dataclasses import dataclass, field, replace
from functools import partial
from pathlib import Path, PurePosixPath
from string import Template
from typing import Any, Callable, Dict, List, Mapping, NamedTuple, Optional, Sequence, Tuple
from testuary import generation_evidence, languages, phase_state, config, hints, models, project
from testuary import evidence as validity
from testuary.generation_evidence import source_sha256
from testuary import prompts as prompts_module
from testuary import starter_selection
from testuary import normal_run
from testuary.agent import _helpers as agent_helpers
from testuary import model_requests
from testuary import models as model_source
from testuary.models import PhaseModelSettings, resolve_output_cap
from testuary import starter_salvage


@dataclass
class StarterResult:
    """What ``run_starter_phase`` hands the orchestrator and the CSV."""

    # ok | cancelled_zero_passing | validation_unresolved |
    # failed_phase_selection | failed_model_request
    status: str = "ok"
    baseline_file: Optional[str] = None
    baseline_from_single_suite: bool = False
    winner: str = ""
    n_sets_requested: int = 0
    n_generated: int = 0
    n_green: int = 0
    n_discarded_syntax: int = 0
    n_discarded_collection: int = 0
    n_unsalvageable: int = 0
    n_zero_coverage: int = 0
    # Suites that pass but that the coverage tool could not measure, or of
    # which the mutation tool measured nothing: they have no complete
    # numbers, so their slots are generated again.
    n_unmeasured_coverage: int = 0
    n_unmeasured_mutation: int = 0
    # Suites whose validation or final execution remains unresolved.
    # Their paid responses are kept and processed again on resume.
    n_validation_unresolved: int = 0
    pool_tests: int = 0
    selected_tests: int = 0
    baseline_line_coverage: Optional[float] = None
    baseline_branch_coverage: Optional[float] = None
    baseline_suite_seconds: Optional[float] = None
    # The baseline run's uncovered lines and branches, the coverage phase's
    # starting work list. None when not recorded (a marker written before
    # they were kept).
    baseline_missing_lines: Optional[Tuple[int, ...]] = None
    baseline_missing_branches: Optional[Tuple[str, ...]] = None
    # Per-suite metric distribution across the green suites BEFORE the set-cover
    # merge: each holds {mean, median, std, min, max}. This is the starting
    # point the merge improves on, the average single generated suite, so a
    # researcher can compare it to a zero-shot single-suite baseline. Empty when
    # no suite yielded the metric.
    suite_line_stats: Dict[str, float] = field(default_factory=dict)
    suite_branch_stats: Dict[str, float] = field(default_factory=dict)
    suite_mutation_stats: Dict[str, float] = field(default_factory=dict)
    #: How the starter suites' mutation results were obtained: measure (each
    #: suite alone), derive (from one pooled run) or both (measured
    #: governing). A requested derivation the run could not make reads
    #: measure, and unavailable means the per-suite measurement produced no
    #: result. Empty when no mutation was measured.
    suite_mutation_method: str = ""
    #: Physical model requests: every attempt sent, including failed ones and
    #: the requests of the context-overflow output halving.
    requests_used: int = 0
    #: Model requests, each of which obtained one delivered
    #: response or ended the phase. Equal to ``requests_used`` in a run
    #: without transport failures.
    logical_requests: int = 0
    #: Requests sent whose token usage the provider did not report. The
    #: token totals undercount by their usage, and unknown is never zero.
    unknown_usage_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    #: Cached-prefix input tokens (the provider-reported prompt-cache
    #: hits), a subset of input_tokens billed at the lower rate, summed over
    #: the requests whose provider reported the cache split.
    cache_hit_tokens: int = 0
    #: Requests with reported token usage whose cache split the provider did
    #: not report: unknown, not zero hits. None for records predating it.
    unknown_cache_requests: Optional[int] = 0
    #: Generation attempts per suite slot, keyed by slot. The refill
    #: limit applies per slot, so the per-slot count is what an analysis
    #: reads. A reused pool restores the archived counts and adds the
    #: attempts this run spent refilling.
    suite_generation_attempts: Dict[int, int] = field(default_factory=dict)
    # Wall-clock of the generation portion (the parallel suite waves), separate
    # from the whole-phase ``duration_s``. Captured directly as elapsed time, so
    # it already reflects the wave concurrency, and persisted so a reused-suite
    # run adds the original generation time back into ``duration_s`` instead of
    # reporting only the (fast) reuse load. Zero without a manifest.
    generation_seconds: float = 0.0
    duration_s: float = 0.0
    detail: str = ""


#: The starter phase's steps as displays show them, in the order they run:
#: generating the suites (each is checked as it returns) and measuring the
#: accepted ones, then the phase end: the consolidation every phase ends
#: with (merging, measuring and reducing, timing tied candidates included,
#: and checking the chosen suite under its delivered name).
STEPS = (("generate", "Generate"), ("measure", "Measure suites"),
         ("consolidation", "Consolidation"))
#: The first of the phase-end steps.
PHASE_END = "consolidation"


def _enter(step: str) -> None:
    """Show the starter phase's ``step``, one of ``STEPS``, as begun."""
    from testuary._progress import enter_step

    enter_step(step, dict(STEPS)[step])


def _suite_stats(values: List[Optional[float]]) -> Dict[str, float]:
    """Distribution summary (mean / median / std / min / max) over the per-suite
    values, ignoring None. Returns {} when no suite supplied the metric. ``std``
    is the sample standard deviation (0.0 for a single suite)."""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return {}
    return {
        "mean": statistics.fmean(vals),
        "median": statistics.median(vals),
        "std": statistics.stdev(vals) if len(vals) > 1 else 0.0,
        "min": min(vals),
        "max": max(vals),
    }


def _extract_code(text: str, fence: str) -> Optional[str]:
    """The last fenced code block (models sometimes emit prose first). A
    generation truncated before its closing fence yields no block and is
    discarded: a partial suite is not the model's output."""
    blocks = re.findall(
        rf"```{re.escape(fence)}\s*\n(.*?)```", text, flags=re.DOTALL,
    )
    if not blocks:
        blocks = re.findall(r"```\s*\n(.*?)```", text, flags=re.DOTALL)
    return blocks[-1] if blocks else None


def _import_hint(language: str, target_file_path: str,
                 source_dir: str) -> str:
    """The import line for the file under test, as the language writes it
    (its ``import_statement``: Python's dotted module, bound through
    importlib where the package binds the module's name itself). A
    language that names none gets the sentence that leaves the import to
    the prompt's language rules."""
    statement = getattr(languages.get_backend(language), "import_statement", None)
    line = statement(target_file_path, source_dir) if callable(statement) else ""
    return line or "follow the language's import conventions for the target"


def _is_context_overflow(exc: Exception) -> bool:
    """Whether a gateway rejection is the context-window-overflow class
    (``input + max_tokens <= context`` violated). The token counts in
    the message are derived from the request rather than measured (the
    reported input is the context minus the requested output tokens, not the
    prompt's real size), so the message classifies the failure but
    cannot size the retry."""
    msg = str(exc)
    return bool(re.search(r"context length is only \d+", msg)
                and re.search(r"passed \d+ input tokens", msg))


def _resolve_output_cap(phase_cfg: Dict[str, Any],
                        api_cfg: Dict[str, Any]) -> Optional[int]:
    """The suite-generation output token limit, resolved the way the editing
    agents resolve theirs: the smaller of the phase cap and the model
    alias's cap, or None when neither is set so the backend sizes the
    response from the remaining context.

    A big-context model whose alias leaves the cap unset runs uncapped
    in suite generation, the same as it does in the agent loop. An alias
    that caps output (a small-context gateway that would otherwise clamp
    the response low) carries that cap into suite generation too. The
    phase value adds a limit even when no alias cap is configured."""
    return resolve_output_cap(
        phase_cfg.get("max_output_tokens"), api_cfg.get("max_output_tokens"),
    )


#: How a phase that produced nothing says its replies hit the output limit.
OUTPUT_LIMIT_DETAIL = "the model used the whole output limit"


def _zero_green_detail(res: "StarterResult") -> str:
    """Why no starter suite is green: the generated count and, among them,
    the suites each check rejected."""
    rejected = [f"{n} {what}" for n, what in (
        (res.n_discarded_syntax, "with a syntax error"),
        (res.n_discarded_collection, "that did not load"),
        (res.n_unsalvageable, "that failed the checks"),
        (res.n_zero_coverage, "that covered nothing"),
        (res.n_unmeasured_coverage, "that the coverage tool could not measure"),
        (res.n_unmeasured_mutation, "of which the mutation tool measured nothing"),
    ) if n]
    detail = f"no green tests from {res.n_generated} generated suites"
    return detail + (f" (among them {', '.join(rejected)})" if rejected else "")


def _output_limit_detail(hits: int, replies: int, limit: int) -> str:
    """Why a phase got no suite when replies spent the whole output limit
    without text: a reasoning model can think until the limit."""
    return (f"{OUTPUT_LIMIT_DETAIL} (max_output_tokens = {limit}) without "
            f"answering in {hits} of {replies} replies. Raise max_output_tokens")


def _resolve_concurrent_requests(cfg: Dict[str, Any]) -> int:
    """How many suite generations may be in flight at once.

    The ollama provider (Ollama Cloud or an Ollama on this machine) takes
    ``concurrent_requests_ollama``, since the hosted ollama.com endpoint
    answers HTTP 429 above a handful of concurrent streams. Everything else
    takes ``concurrent_requests``: an OpenAI-compatible gateway and a
    self-hosted vLLM
    both sustain the configured parallelism, and holding them to the cautious
    number would cost hours per run for nothing. ``--concurrent-requests`` is an
    operator's explicit answer and sets both."""
    provider = str(models.get_api_config().get("provider") or "").strip().lower()
    if provider == "ollama":
        return cfg["concurrent_requests_ollama"]
    return cfg["concurrent_requests"]


#: The seed distance between two attempts of one slot. Attempt a of slot i
#: in run rk sends the run's slot seed (``config.run_seed``: seed_base +
#: 1000 * k + i) plus (a - 1) * REFILL_SEED_STRIDE, so a slot's first
#: attempt keeps the run's slot seed. The run seeds of up to 1000 runs of up
#: to 1000 slots lie below this stride, so no refill seed equals another
#: slot's, another attempt's or another run's seed.
REFILL_SEED_STRIDE = 1_000_000


def attempt_seed(seed_base: int, slot: int, attempt: int, run_idx: int = 0) -> int:
    """The sampling seed of attempt ``attempt`` (from 1) of slot ``slot`` in
    run ``run_idx`` (0 for r0). Each attempt draws its own deterministic
    seed: an endpoint that honors seeds would return a refilled slot's
    rejected answer again under the same seed."""
    return (config.run_seed(seed_base, run_idx, slot)
            + (max(1, attempt) - 1) * REFILL_SEED_STRIDE)


def _usage_fields(usage: Optional[model_requests.Usage]) -> Dict[str, Any]:
    """The usage record a generator returns: unknown token usage is marked
    unknown and an unreported cache split is None, never zero."""
    if usage is None:
        return {"input_tokens": 0, "output_tokens": 0, "cache_hit_tokens": None,
                "cache_miss_tokens": None, "usage_known": False}
    cache = usage.cache_hit_tokens
    return {"input_tokens": usage.input_tokens, "output_tokens": usage.output_tokens,
            "cache_hit_tokens": cache,
            "cache_miss_tokens": None if cache is None else usage.input_tokens - cache,
            "usage_known": True}


def _default_generate(
    api_cfg: Dict[str, str],
    messages: Sequence[Mapping[str, str]] | str,
    *,
    seed: int,
    model_settings: PhaseModelSettings,
    model_clients=None,
    reasoning_grace: Optional[float] = None,
    on_context_overflow: Optional[Callable[[BaseException], None]] = None,
    on_request: Optional[Callable[[int, Optional[int]], None]] = None,
) -> Tuple[str, Dict[str, int]]:
    """One streamed generation. The ollama-native protocol goes through the
    native chat API (the OpenAI-compatible endpoint emits no chunks during a
    reasoning model's thinking phase, and the silent connection gets
    severed on long generations). Every other protocol streams through its
    chat client from ``model_clients``. ``model_settings`` supplies the
    request: sampling, output token limit, reasoning and request options.

    On the OpenAI-compatible path the output token limit comes from the model
    alias (an unset alias means None, so the backend sizes the response
    from the remaining context). A gateway that applies a small default ceiling
    when the request omits a limit would truncate a large file's suite
    so the closing code fence is lost, so such a gateway's alias sets
    the cap. When the limit plus the prompt would exceed the model's
    context window the gateway rejects the request. The limit is halved
    per rejection until the request fits (the rejection message cannot
    size the retry, so the search is geometric), down to a floor below
    which the overflow is re-raised. ``None`` sends no limit and skips
    the halving: the backend allocates from its own remaining context.

    On the native path the output token limit sets ``num_predict``: a
    reasoning model otherwise gets the endpoint's small default limit and
    can exhaust it reasoning before emitting the suite, so the caller passes
    the model's full output token limit. Reasoning stays on. A response that
    still uses up the whole limit on reasoning has empty content and is a
    failed generation, handled by the caller.

    ``reasoning_grace`` guards against a reasoning model that never
    converges: it streams reasoning continuously and emits no answer, so
    the per-read socket timeout never fires and the generation runs to
    the context limit with empty content. When set, the stream is cut
    once this many seconds pass with no answer content yet. A generation
    that has begun emitting the suite is never cut, so a large file's
    long-but-productive generation is left to finish. A cut returns the
    empty content (a failed generation, replaced by the caller's
    fresh-seed top-up). ``None`` disables the guard.

    The returned usage carries ``usage_known``: False when the provider
    reported no usage, as after a cut stream, so the caller records the
    request as unknown rather than as zero tokens. ``on_context_overflow``
    is called with the rejection before each halved request is sent, so
    every request the halving sent is recorded. ``on_request`` observes
    the actual slot seed and output cap after client preparation, just before
    each stream starts. It does not own request accounting."""
    # Slot seeds and overflow accounting stay here.
    if isinstance(messages, str):
        messages = ({"role": "user", "content": messages},)

    if model_settings.protocol == "ollama-native":
        import httpx
        from testuary.models import native_endpoint, native_request_body

        native_url = native_endpoint(model_settings.identity.api_url)
        timeout = model_settings.values["request_timeout_seconds"]
        body = native_request_body(model_settings, messages, slot_seed=seed)
        content: List[str] = []
        usage: Dict[str, int] = {}
        if on_request is not None:
            on_request(body["options"]["seed"], body["options"].get("num_predict"))
        with httpx.stream(
            "POST", native_url,
            headers={"Authorization": f"Bearer {api_cfg['api_key']}"},
            json=body, timeout=timeout,
        ) as r:
            r.raise_for_status()
            start = time.monotonic()
            for line in r.iter_lines():
                if not line.strip():
                    continue
                d = json.loads(line)
                piece = d.get("message", {}).get("content")
                if piece:
                    content.append(piece)
                if d.get("done"):
                    # Ollama reports prompt_eval_cached_count for a served
                    # cached prefix. Absent or invalid counts stay unknown.
                    usage = _usage_fields(model_requests.reported_usage(
                        d.get("prompt_eval_count"), d.get("eval_count"),
                        d.get("prompt_eval_cached_count")))
                    if isinstance(d.get("model"), str) and d["model"]:
                        usage["response_model"] = d["model"]
                    if isinstance(d.get("done_reason"), str) and d["done_reason"]:
                        usage["finish_reason"] = d["done_reason"]
                if (reasoning_grace and not content
                        and time.monotonic() - start > reasoning_grace):
                    break
        if not usage:
            usage = _usage_fields(None)
        return "".join(content), usage

    from langchain_core.messages import HumanMessage, SystemMessage
    from testuary.agent._langgraph import _extract_cache_hit_tokens

    outbound = []
    for message in messages:
        role = message.get("role")
        content = message.get("content", "")
        if role == "system":
            outbound.append(SystemMessage(content=content))
        elif role == "user":
            outbound.append(HumanMessage(content=content))
        else:
            raise ValueError(f"unsupported starter-phase message role: {role!r}")

    def _stream(max_tokens: Optional[int]) -> Tuple[str, Dict[str, int]]:
        model = model_clients.get(seed=seed, max_output_tokens=max_tokens)
        if on_request is not None:
            on_request(seed, max_tokens)
        resp = None
        start = time.monotonic()
        for chunk in model.stream(outbound):
            resp = chunk if resp is None else resp + chunk
            answer = resp.content if isinstance(resp.content, str) else str(resp.text)
            if (reasoning_grace and not answer
                    and time.monotonic() - start > reasoning_grace):
                break
        if resp is None:
            raise RuntimeError("empty stream")
        usage_meta = getattr(resp, "usage_metadata", None)
        if not isinstance(usage_meta, dict):
            usage_meta = {}
        # Cached-prefix input tokens, billed at the lower cache-hit rate, come
        # from the raw token_usage block or usage_metadata.input_token_details.
        # Neither reported leaves the split unknown.
        answer = resp.content if isinstance(resp.content, str) else str(resp.text)
        response_usage = _usage_fields(model_requests.reported_usage(
            usage_meta.get("input_tokens"), usage_meta.get("output_tokens"),
            _extract_cache_hit_tokens(resp)))
        metadata = getattr(resp, "response_metadata", None) or {}
        reported_model = metadata.get("model_name") or metadata.get("model")
        if isinstance(reported_model, str) and reported_model:
            response_usage["response_model"] = reported_model
        finish_reason = metadata.get("finish_reason") or metadata.get("done_reason")
        if isinstance(finish_reason, str) and finish_reason:
            response_usage["finish_reason"] = finish_reason
        return answer or "", response_usage

    output_limit = model_settings.values["max_output_tokens"]
    while True:
        try:
            return _stream(output_limit)
        except Exception as exc:
            if not _is_context_overflow(exc) or output_limit is None:
                raise
            output_limit //= 2
            if output_limit < 2048:
                raise
            if model_clients is not None:
                # Construct before recording another request sent. A rejected
                # constructor is configuration failure, not a provider call.
                model_clients.get(seed=seed, max_output_tokens=output_limit)
            if on_context_overflow is not None:
                on_context_overflow(exc)


def _completed_in_slices(futures):
    """Futures as they complete, in submission order within one slice.

    The caller waits in short slices: a main thread blocked in one long wait
    on Windows does not receive Ctrl+C until the wait returns.
    """
    pending = set(futures)
    while pending:
        done, pending = concurrent.futures.wait(
            pending, timeout=model_requests.WAIT_SLICE_SECONDS,
            return_when=concurrent.futures.FIRST_COMPLETED)
        yield from sorted(done, key=futures.index)


_GENERATION_MANIFEST = "_generation.json"

# A delivered response not yet processed, with its text: written when the
# response arrives and removed once it is processed.
_GENERATION_DELIVERY = "delivery.json"

_REQUEST_RECORDS = "request_records.jsonl"


@dataclass
class _GenerationLedger:
    """What earlier runs of a starter pool spent, from their request records:
    each slot's delivered attempts, the phase's transient failures, and the
    request and token totals. The records are appended and flushed
    for every request sent, so they hold what a run spent up to an
    interruption, including responses it never processed."""

    attempts_per_slot: Dict[int, int] = field(default_factory=dict)
    transient_failures: int = 0
    requests_used: int = 0
    logical_requests: int = 0
    unknown_usage_requests: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hit_tokens: int = 0
    unknown_cache_requests: Optional[int] = 0


def _has_request_records(suites_root: Path) -> bool:
    path = suites_root / _REQUEST_RECORDS
    try:
        return path.is_file() and path.stat().st_size > 0
    except OSError:
        return False


def _repair_request_records(suites_root: Path) -> Optional[Path]:
    """Move a final line of the request records that an interruption cut
    short, or that does not parse, into ``request_records.jsonl.interrupted``
    beside them, kept for diagnosis, so that the next record starts on a line
    of its own. Returns that file when a line was moved."""
    path = suites_root / _REQUEST_RECORDS
    if not _has_request_records(suites_root):
        return None
    data = path.read_bytes()
    body = data[:-1] if data.endswith(b"\n") else data
    cut = body.rfind(b"\n") + 1
    last = body[cut:]
    if data.endswith(b"\n"):
        try:
            json.loads(last.decode("utf-8"))
            return None
        except ValueError:
            pass
    aside = path.with_name(path.name + ".interrupted")
    with aside.open("ab") as handle:
        handle.write(last + b"\n")
    with path.open("r+b") as handle:
        handle.truncate(cut)
    return aside


def _read_generation_ledger(suites_root: Path) -> Optional[_GenerationLedger]:
    """The ledger of the request records beside the suites, or ``None`` when
    there are none: the generation manifest's counts apply then. The caller
    repairs an interrupted final line first, and any unreadable line stops
    the reuse."""
    if not _has_request_records(suites_root):
        return None
    path = suites_root / _REQUEST_RECORDS
    try:
        lines = [line for line in path.read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    except OSError as exc:
        raise RuntimeError(
            f"reuse_suites is set but the request records {path} cannot be "
            f"read: {exc}") from exc
    records = []
    for number, line in enumerate(lines, start=1):
        try:
            records.append(json.loads(line))
        except ValueError as exc:
            raise RuntimeError(
                f"reuse_suites is set but line {number} of the request records "
                f"{path} cannot be read: {exc}") from exc
    if not records:
        return None
    ledger = _GenerationLedger()
    counted = set()
    for record in records:
        slot = int(record["slot"])
        outcome = record.get("outcome")
        ledger.requests_used += 1
        counted.add((record["run"], record.get("logical_request_idx")))
        if record.get("usage_known"):
            ledger.input_tokens += int(record.get("input_tokens") or 0)
            ledger.output_tokens += int(record.get("output_tokens") or 0)
            ledger.cache_hit_tokens += int(record.get("cache_hit_tokens") or 0)
            cache_known = record.get("cache_usage_known")
            if cache_known is None:
                # Without the flag the count may be a default zero: unknown.
                ledger.unknown_cache_requests = None
            elif not cache_known and ledger.unknown_cache_requests is not None:
                ledger.unknown_cache_requests += 1
        else:
            ledger.unknown_usage_requests += 1
        if outcome == model_requests.DELIVERED:
            ledger.attempts_per_slot[slot] = (
                ledger.attempts_per_slot.get(slot, 0) + 1)
        elif outcome == model_requests.TRANSIENT:
            ledger.transient_failures += 1
    ledger.logical_requests = len(counted)
    return ledger


@dataclass(frozen=True)
class _GenerationStart:
    """The totals a run's generation starts from: zero for a fresh
    generation, the restored ones for a suite-reuse run, and the monotonic
    time this run's generation began."""

    requests_used: int
    logical_requests: int
    unknown_usage_requests: int
    generation_seconds: float
    started: float

    def seconds_at(self, now: float) -> float:
        """The generation's seconds at the monotonic time ``now``: those it
        started from and this run's, to the millisecond."""
        return round(self.generation_seconds + (now - self.started), 3)


#: The model request failures that end the phase: a terminal failure or a
#: retry ceiling.
_REQUEST_FAILURES = (model_requests.TerminalRequestError, model_requests.TransportExhausted)


class _RequestSpans:
    """The spans in which a model request of the starter phase is in
    flight. The requests run in parallel, so the phase's model time is the
    time some request is in flight: the union of the spans, neither their
    sum nor the generation's whole span, which also holds the checks of
    each answer."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._spans: List[Tuple[float, float]] = []

    @contextlib.contextmanager
    def request(self):
        started = time.monotonic()
        try:
            yield
        finally:
            ended = time.monotonic()
            with self._lock:
                self._spans.append((started, ended))

    def seconds(self, since: Optional[float] = None) -> float:
        """The time some request was in flight, after ``since`` when given."""
        with self._lock:
            spans = sorted((start if since is None else max(start, since), end)
                           for start, end in self._spans
                           if since is None or end > since)
        total, reach = 0.0, None
        for start, end in spans:
            if reach is None or start > reach:
                total += end - start
                reach = end
            elif end > reach:
                total += end - reach
                reach = end
        return total


def _persist_generation_manifest(
    suites_root: Path, res: "StarterResult", *, generation_seconds: float,
    complete: bool, transient_failures: int = 0,
    processed_deliveries: Sequence[str] = (),
    suite_slots: Dict[str, Any],
) -> None:
    """Record the generation totals (cost and wall-clock) and the attempts
    each slot has spent beside the suites, so a later ``--reuse-starter-suites``
    run restores them and reports the same phase totals as the generation
    that produced the pool.

    Written after every processed response and on every exit, and marked
    complete only when the generation finished. A generation that a request
    failure or an interruption ended thus leaves each slot's spent attempts,
    which a reuse run keeps. A model request is spent when its response was
    delivered. The attempt counts also make the refill history readable
    straight off the artifacts, without reconstructing it from the logs.

    It is also the commit point of processing a response: the response's
    delivery id enters ``processed_deliveries`` in the same write that
    carries the counters it changed, so a saved response is processed and
    counted exactly once even when an interruption falls in between."""
    record = {
        "requests_used": res.requests_used,
        "logical_requests": res.logical_requests,
        "unknown_usage_requests": res.unknown_usage_requests,
        "input_tokens": res.input_tokens,
        "cache_hit_tokens": res.cache_hit_tokens,
        "unknown_cache_requests": res.unknown_cache_requests,
        "output_tokens": res.output_tokens,
        "generation_seconds": generation_seconds,
        "n_generated": res.n_generated,
        "n_discarded_syntax": res.n_discarded_syntax,
        "n_discarded_collection": res.n_discarded_collection,
        "n_unsalvageable": res.n_unsalvageable,
        "n_zero_coverage": res.n_zero_coverage,
        "n_unmeasured_coverage": res.n_unmeasured_coverage,
        "n_unmeasured_mutation": res.n_unmeasured_mutation,
        "n_validation_unresolved": res.n_validation_unresolved,
        "transient_failures": transient_failures,
        "processed_deliveries": sorted(processed_deliveries),
        "generation_attempts_per_suite": {
            str(i): n
            for i, n in sorted(res.suite_generation_attempts.items())
        },
        "generation_complete": complete,
    }
    record["suite_slots"] = suite_slots
    validity.write_json_atomic(suites_root / _GENERATION_MANIFEST, record)


def _suite_source_path(suite_dir: Path, stem: str, backend=None) -> Path:
    """The suite archive's file for ``stem``, with the language's suffix.

    These are source snapshots, not executable staging files: the backend
    still chooses the filename when a snapshot is run.
    """
    suffix = backend.test_file_extensions()[0] if backend is not None else ".py"
    return suite_dir / f"{stem}{suffix}"


def _archived_suite_text(path: Path) -> str:
    """A suite archive's source, as the exact bytes it was written with."""
    return path.read_bytes().decode("utf-8")


#: The directory in a slot directory that keeps the files of the slot's earlier
#: attempts. The slot directory's own files are those of its latest processed
#: attempt (the accepted one for an accepted slot), so every reader of them
#: reads that attempt. When a later attempt of the slot is processed, the
#: earlier attempt's files move to ``attempts/<n>/``, n being the attempt's
#: number within its slot as its ``original.json`` records it
#: (``slot_attempt``).
_ATTEMPTS_DIR = "attempts"

#: The validity derivations of the removals of tests a measurement failed in
#: a slot's suite, beside its ``validity.json``.
_SLOT_DERIVATIONS = "validity_derivations.json"

#: The classifications of an accepted suite withdrawn at measurement: its
#: slot holds no starter suite, and a reuse leaves it out of the pool.
_WITHDRAWN = ("coverage_unmeasured", "mutation_unmeasured")


def _attempt_file_names(backend=None) -> Tuple[str, ...]:
    """The files a slot directory holds for one attempt, ``original.json``
    last: it names the attempt, so a move that an interruption cut short
    leaves it in place, and the move is completed when the attempt that
    replaces them is processed again."""
    return ("response.txt", _suite_source_path(Path(), "original", backend).name,
            "validity.json", _SLOT_DERIVATIONS,
            _suite_source_path(Path(), "unconfirmed", backend).name,
            _suite_source_path(Path(), "cleaned", backend).name, "cleaned.json",
            "original.json")


def _archive_earlier_attempt(sdir: Path, slot_attempt: int,
                             backend=None) -> Optional[Path]:
    """Move the files of the slot's earlier attempt into
    ``attempts/<n>/`` before attempt ``slot_attempt`` writes its own.
    Nothing moves when the directory already holds attempt ``slot_attempt``
    (its response processed again) or when no ``original.json`` names an
    attempt: the files left then are those of a processing that an
    interruption cut short, which the attempt writes again. Returns the
    directory the files moved to."""
    meta_path = sdir / "original.json"
    if not meta_path.is_file():
        return None
    present = [name for name in _attempt_file_names(backend) if (sdir / name).is_file()]
    try:
        recorded = json.loads(meta_path.read_bytes().decode("utf-8")).get("slot_attempt")
    except (OSError, ValueError, AttributeError):
        recorded = None
    if type(recorded) is not int or recorded < 1 or recorded == slot_attempt:
        return None
    target = sdir / _ATTEMPTS_DIR / str(recorded)
    target.mkdir(parents=True, exist_ok=True)
    for name in present:
        (sdir / name).replace(target / name)
    return target


#: The start of a starter slot's name: its archive directory, its label in the
#: records and its staged test file's stem.
SLOT_PREFIX = "suite_"


def slot_name(i: int) -> str:
    """Starter slot ``i``'s name, as its directory, label and staged file use it."""
    return f"{SLOT_PREFIX}{i:03d}"


def _suite_archive_paths(
    suites_root: Path, backend=None,
) -> List[Tuple[Path, Path, Path]]:
    # Resolve the whole pool before any reader normalizes or rewrites a suite.
    return [
        (suite_dir, _suite_source_path(suite_dir, "original", backend),
         _suite_source_path(suite_dir, "cleaned", backend))
        for suite_dir in sorted(suites_root.glob(f"{SLOT_PREFIX}*"))
    ]


def _has_persisted_pool(suites_root: Path) -> bool:
    """True when the suites directory holds a generation manifest, the record of a
    generation that ran, written before its first request. A reuse request
    for a target without one (a file never run) falls back to a fresh
    generation rather than failing, so one reuse run can span already-run
    and brand-new targets. A generation that ran keeps what it spent even
    when it left no green suite."""
    return (suites_root / _GENERATION_MANIFEST).is_file()


def _load_persisted_green(suites_root: Path, green: Dict[int, str],
                          res: "StarterResult",
                          log: Callable[[str], None],
                          backend=None,
                          source_dir: str = ".",
                          unresolved: Optional[set] = None) -> bool:
    """Reuse a prior run's generated starter suites instead of generating fresh.
    Slots with unresolved validation or confirmation go into ``unresolved``
    for the caller to process again without a model request.

    Loads each ``suite_NNN/cleaned<suffix>`` (a salvaged-green suite) under
    ``suites_root`` into the pool, so the expensive model generation and salvage
    is paid once and the deterministic merge can be re-run for free. The
    counts and totals come from the generation manifest, so the phase report
    matches the original generation. Returns whether the persisted
    generation completed. One that a request failure or an interruption
    ended is continued by the caller. A pool without a generation manifest
    is a hard error: regenerating would silently re-spend the model requests
    the reuse was meant to save."""
    from testuary.consolidation import ValidationUnresolved, validate_judged

    manifest = suites_root / _GENERATION_MANIFEST
    if not manifest.is_file():
        raise FileNotFoundError(
            f"reuse_suites is set but {suites_root} holds no record of a "
            "generation: run the starter phase once before reusing its starter suites")
    loaded = 0
    for sdir, _original, cleaned in _suite_archive_paths(suites_root, backend):
        try:
            slot = int(sdir.name.split("_")[1])
        except (IndexError, ValueError):
            slot = None
        classification = None
        meta_path = sdir / "original.json"
        if meta_path.exists():
            try:
                classification = json.loads(
                    meta_path.read_text(encoding="utf-8")).get("classification")
            except (OSError, ValueError):
                classification = None
        if cleaned.exists() and classification not in (
                "validation_unresolved", *_WITHDRAWN):
            if slot is None:
                continue
            i = slot
            source = _archived_suite_text(cleaned)
            if backend is not None:
                normalized = backend.normalize_shadowed_definitions(source)
                if not normalized.parse_ok:
                    raise RuntimeError(
                        f"reused starter suite {i} is not parseable under "
                        f"{normalized.rule_version}: {normalized.parse_error}"
                    )
                finalized = backend.finalize_test_source(normalized.source)
                if normalized.changed or finalized != source:
                    log(
                        f"[starter] reused starter suite {i} contained "
                        f"{len(normalized.removed_bindings)} shadowed "
                        "definition(s) or source-finalization residue, "
                        "normalized before re-validation"
                    )
                    cleaned.write_bytes(finalized.encode("utf-8"))
                source = finalized
                # A validator that gives no answer is asked again, and no
                # answer at all stops the reuse unjudged, not as a failure.
                try:
                    validation = validate_judged(backend, source, source_dir)
                except ValidationUnresolved as exc:
                    raise type(exc)(f"reused starter suite {i}: the target validator "
                                    f"did not judge it ({exc})") from None
                if not validation.ok:
                    raise RuntimeError(
                        f"reused starter suite {i} failed target validation: "
                        f"{validation.status}: {validation.message}"
                    )
            green[i] = source
            loaded += 1
        elif classification == "validation_unresolved":
            if unresolved is not None and slot is not None:
                unresolved.add(slot)
    # The generation manifest holds the run's accumulated totals, failed
    # generations that the refill re-rolled in place included, and the
    # generation wall-clock, so duration_s includes the original generation
    # time.
    complete = True
    try:
        m = json.loads(manifest.read_text(encoding="utf-8"))
        if m.get("input_tokens") is not None:
            res.requests_used = int(m.get("requests_used") or 0)
            res.logical_requests = int(m.get("logical_requests") or 0)
            res.unknown_usage_requests = int(
                m.get("unknown_usage_requests") or 0)
            res.input_tokens = int(m.get("input_tokens") or 0)
            res.output_tokens = int(m.get("output_tokens") or 0)
            res.cache_hit_tokens = int(m.get("cache_hit_tokens") or 0)
            unknown_cache = m.get("unknown_cache_requests")
            res.unknown_cache_requests = (
                int(unknown_cache) if unknown_cache is not None else None)
        if m.get("n_generated") is not None:
            res.n_generated = int(m.get("n_generated") or 0)
        # The outcome counts of every processed response, re-rolled ones
        # included.
        for key in ("n_discarded_syntax", "n_discarded_collection",
                    "n_unsalvageable", "n_zero_coverage",
                    "n_unmeasured_coverage", "n_unmeasured_mutation",
                    "n_validation_unresolved"):
            if m.get(key) is not None:
                setattr(res, key, int(m.get(key) or 0))
        res.generation_seconds = float(m.get("generation_seconds") or 0.0)
        res.suite_generation_attempts = {
            int(k): int(v) for k, v in
            (m.get("generation_attempts_per_suite") or {}).items()
        }
        complete = m.get("generation_complete") is True
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        # The manifest holds the spent attempts. Reusing the pool
        # without them would grant every slot its allowance again.
        raise RuntimeError(
            f"reuse_suites is set but the generation manifest {manifest} "
            f"cannot be read: {type(exc).__name__}: {exc}"
        ) from exc
    if complete:
        log(f"[starter] reusing {loaded} green starter suite(s) from {suites_root.name}, "
            "with no model calls for these (a short pool is refilled below)")
    else:
        log(f"[starter] reusing {loaded} green starter suite(s) from {suites_root.name}. "
            "The generation that produced them ended early and continues "
            "with each slot's spent attempts")
    return complete


def _revalidate_reused_pool(
    green: Dict[int, str], backend, source_dir: str,
    *, timeout: int, log: Callable[[str], None],
    instrument_target: Optional[str] = None,
) -> int:
    """Drop reused suites that run no test, so the refill (refill_failed_suites)
    regenerates their slots. Returns the count dropped.

    A persisted suite is trusted as green, with one exception: an unconditional
    module-scope skip (a ``pytest.importorskip`` for an absent or versionless
    package) collects nothing yet reads as green, and merged into the pool it
    skips every suite at collection. Only suites still carrying such a skip are
    re-run, so a clean pool pays no runs and is reused byte-identically. A re-run
    that completes with zero passing tests confirms the suite is empty and drops
    it. A timeout is not a drop signal: the suite may be healthy but slow."""
    suspects = [i for i in sorted(green)
                if backend.strip_module_level_skips(green[i])[1]]
    if not suspects:
        return 0
    dropped = 0
    with project.scratch_dir(prefix="testuary_revalidate_") as scratch:
        for i in suspects:
            suite = backend.staged_test_path(str(scratch), "candidate", green[i])
            suite.write_text(green[i], encoding="utf-8")
            run = normal_run.run(backend, [str(suite)], source_dir,
                                 timeout=timeout, target=instrument_target)
            # A count the run did not observe (None, a crash) says it ran none.
            if not run.timed_out and not run.passed:
                del green[i]
                dropped += 1
                log(f"[starter] reused starter suite {i} collects no test on re-validation "
                    "(an unconditional module-level skip) and is dropped for regeneration")
    return dropped


def _restore_totals(res: StarterResult, ledger, transient: int) -> int:
    """The transient failures of earlier sessions, the larger of
    ``transient`` and the request records' (``ledger``), with each count of
    ``res`` raised to the records' where they hold more."""
    if ledger is not None:
        for slot, count in ledger.attempts_per_slot.items():
            res.suite_generation_attempts[slot] = max(count, res.suite_generation_attempts.get(slot, 0))
        res.requests_used = max(res.requests_used, ledger.requests_used)
        res.logical_requests = max(res.logical_requests or 0, ledger.logical_requests)
        res.unknown_usage_requests = max(res.unknown_usage_requests or 0, ledger.unknown_usage_requests)
        res.input_tokens = max(res.input_tokens, ledger.input_tokens)
        res.output_tokens = max(res.output_tokens, ledger.output_tokens)
        res.cache_hit_tokens = max(res.cache_hit_tokens, ledger.cache_hit_tokens)
        res.unknown_cache_requests = (
            None if res.unknown_cache_requests is None or ledger.unknown_cache_requests is None
            else max(res.unknown_cache_requests, ledger.unknown_cache_requests))
        transient = max(transient, ledger.transient_failures)
    return transient


def _or_none(convert: Callable[[Any], Any], value: Any) -> Any:
    """``convert(value)``, or None for a None ``value``."""
    return convert(value) if value is not None else None


def _starter_mutation_config() -> Dict[str, Any]:
    return config.get_agent_config("test_mutation", parent="test_coverage")


class _SlotResponse(NamedTuple):
    """A starter slot's delivered model response, as ``generate_suite``
    returns it and a resume rebuilds it from what an earlier run saved."""

    slot: int
    code: Optional[str]
    usage: Dict[str, Any]
    requests: int
    unknown_usage_requests: int
    text: Optional[str]
    delivery_id: Optional[str]


@dataclass
class _StarterPhase:
    """One run of the starter phase for one target: its inputs, and the
    settings and state its generation and its phase end share."""

    language: str
    target_file_path: str
    source_dir: str
    test_output_dir: str
    suites_dir: str
    matrices_dir: Optional[str]
    log: Callable[[str], None]
    prepared_models: Any
    run_idx: int
    reuse_suites: Optional[bool]

    def _on_context_overflow(self, exc: BaseException) -> None:
        # Only the configured generator calls this, from send(), on the thread
        # that set the request's index, extra count and attempt.
        self.in_flight.extra += 1
        self.account.note_extra_request(
            self.in_flight.index, self.in_flight.attempt, model_requests.CONTEXT_OVERFLOW, exc)

    def observe_request(self, index, state, message):
        if self.observer is not None:
            self.observer(operation=f"request/{self.run_id}/{index}", state=state, message=message)

    def observe_saved(self, i, state, message):
        if self.observer is not None:
            self.observer(operation=f"saved-response/{i}", state=state, message=message,
                          provenance="recorded")

    def record_request(self, record: model_requests.RequestRecord) -> None:
        # One line per request sent: its run, slot, attempt number, outcome
        # class and usage status. Written beside the transport record.
        with self.transport_lock:
            entry = {"phase": "starter", "run": self.run_id,
                     "slot": self.slot_of.get(record.logical_index), **record.as_dict()}
            request_idx = getattr(self.in_flight, "transport_idx", None)
            if request_idx is not None:
                entry["request_idx"] = request_idx
            if self.configured_generator and record.outcome == model_requests.DELIVERED:
                reported_model = getattr(self.in_flight, "response_model", None)
                if reported_model is not None:
                    entry["response_model"] = reported_model
            validity.append_jsonl(self.records_path, entry)

        if record.outcome in (model_requests.TERMINAL, model_requests.TRANSIENT):
            self.observe_request(
                record.logical_index, "failed" if record.outcome == model_requests.TERMINAL else "retrying",
                "Request failed" if record.outcome == model_requests.TERMINAL else "Transient request failure")

    def write_manifest(self, generation_seconds: float, complete: bool) -> None:
        _persist_generation_manifest(
            self.suites_root, self.res, generation_seconds=generation_seconds, complete=complete,
            transient_failures=self.earlier_transient["failures"] + self.account.transient_failures,
            processed_deliveries=sorted(self.processed_deliveries),
            suite_slots=self.slot_evidence.snapshot(self.res.suite_generation_attempts))
        if self.observer is not None:
            from testuary.presentation import Usage

            known = self.res.unknown_usage_requests == 0
            # Accepted green starter suites of the requested slots. Refills can
            # retry a rejected slot, so only acceptances count as done.
            accepted = sum(1 for slot in self.slot_evidence.slots if slot["state"] == "accepted")
            self.observer(operation="suite-processing", state="succeeded" if complete else "running",
                          completed=accepted, total=self.suite_count,
                          work_unit="starter suites accepted", provenance="recorded", usage=Usage(
                              logical_requests=self.res.logical_requests,
                              physical_requests=self.res.requests_used,
                              input_tokens=self.res.input_tokens if known else None,
                              output_tokens=self.res.output_tokens if known else None))

    def record_transport(self, sent_settings: Optional[Dict[str, Any]] = None) -> None:
        with self.transport_lock:
            self.transport_request_idx += 1
            record = validity.prompt_transport_record(
                phase="starter", request_idx=self.transport_request_idx, messages=self.messages,
                prompt_template_version=validity.PROMPT_TEMPLATE_VERSION, provider_transport=(
                    {"ollama-native": "ollama_native", "openai-chat": "openai_compatible",
                     "openai-responses": "openai_responses", "added": "added"}[self.phase_model.protocol]
                    if self.phase_model is not None else "model_source"))
            index = getattr(self.in_flight, "index", None)
            record.update(run=self.run_id, slot=self.slot_of.get(index), logical_request_idx=index,
                          attempt=getattr(self.in_flight, "attempt", None))
            if sent_settings is not None:
                record["sent_settings"] = sent_settings
            if self.generation_condition is not None:
                record["prompt_composition"] = self.generation_condition.prompt_composition
            validity.append_jsonl(self.transport_path, record)
            self.in_flight.transport_idx = self.transport_request_idx

        self.observe_request(getattr(self.in_flight, "index", None), "running", "Sending model request")

    def _record_sent_transport(self, seed: int, max_output_tokens: Optional[int]) -> None:
        # The default generator's request hook. That generator exists only
        # inside a run, with the run's phase model.
        self.record_transport(replace(
            self.phase_model,
            values={**self.phase_model.values, "seed": seed, "max_output_tokens": max_output_tokens},
            sources={**self.phase_model.sources, "seed": "starter:slot",
                     "max_output_tokens": "starter:effective-request-cap"}).export())

    def generate_suite(self, i: int, slot_attempt: int) -> _SlotResponse:
        """The ``_SlotResponse`` of attempt ``slot_attempt`` of slot ``i``,
        one model request. ``usage`` carries the suite's own model latency
        under ``generation_seconds`` (the per-suite time, recorded for
        analysis. The phase's parallel wall-clock is timed separately).

        A transient transport failure is retried here by the request policy.
        A terminal failure or a ceiling raises out to the wave, which ends
        the phase. A delivered response without a code block is a spent slot
        attempt, never a transport failure."""
        usage_total: Dict[str, float] = {
            "input_tokens": 0, "output_tokens": 0, "cache_hit_tokens": 0, "cache_miss_tokens": 0,
            "unknown_cache_requests": 0}
        _suite_t0 = time.monotonic()
        seed = attempt_seed(self.seed_base, i, slot_attempt, self.run_idx)
        with self.transport_lock:
            self.slot_evidence.unavailable(i, "request_in_progress", "request pending", active=True)
        if self.model_clients is not None:
            self.model_clients.get(seed=seed, max_output_tokens=self._max_out)
        index = self.account.next_logical_index()
        with self.transport_lock:
            self.slot_of[index] = i
        self.in_flight.index = index
        self.in_flight.extra = 0
        self.in_flight.transport_idx = None

        unrecorded: Optional[BaseException] = None
        try:
            outcome = self.account.execute(
                partial(self.send, seed), usage_of=self.usage_of,
                on_retry=partial(self.on_retry, i, index), logical_index=index)
        except model_requests.RecordNotWritten as exc:
            # Delivered, but its record could not be written: the response is
            # saved below, marked unrecorded, before the phase stops.
            outcome, unrecorded = exc.outcome, exc
        except (model_requests.RequestCancelled, KeyboardInterrupt):
            self.observe_request(index, "cancelled", "Model request cancelled")
            raise
        except model_requests.TransportExhausted as exc:
            self.observe_request(index, "failed", model_requests.exhausted_text(exc, self.account.service))
            raise
        except Exception:
            self.observe_request(index, "failed", "Model request ended with an error")
            raise
        finally:
            self.in_flight.index = None
        text, _ = outcome.value
        if outcome.usage is not None:
            usage_total["input_tokens"] += outcome.usage.input_tokens
            usage_total["output_tokens"] += outcome.usage.output_tokens
            cache_hit = outcome.usage.cache_hit_tokens
            if cache_hit is None:
                usage_total["unknown_cache_requests"] += 1
            else:
                usage_total["cache_hit_tokens"] += cache_hit
                usage_total["cache_miss_tokens"] += (outcome.usage.input_tokens - cache_hit)
        usage_total["generation_seconds"] = round(time.monotonic() - _suite_t0, 3)
        if self.configured_generator and getattr(self.in_flight, "response_model", None) is not None:
            usage_total["response_model"] = self.in_flight.response_model
        code = _extract_code(text, self.fence)
        # The refused requests of the output halving are requests sent for
        # this model request with unknown usage.
        extra = getattr(self.in_flight, "extra", 0)
        # The response is paid for: saved before anything else can fail, so
        # a run interrupted before processing it hands it to the next reuse
        # run instead of losing it. The file is removed once its processing
        # is committed with the manifest.
        delivery_id = f"{self.run_id}:{index}"
        validity.write_json_atomic(
            self.suites_root / slot_name(i) / _GENERATION_DELIVERY, {
                "id": delivery_id, "slot": i, "slot_attempt": slot_attempt, "seed": seed,
                "run": self.run_id, "logical_request_idx": index,
                "physical_requests": outcome.attempts + extra,
                "unknown_usage_requests": outcome.unknown_usage_requests + extra,
                "usage": usage_total, "text": text, "unrecorded": unrecorded is not None,
                "construction_inputs": {"version": 1,
                                        "attempt_id": generation_evidence.paid_attempt(delivery_id),
                                        "run_idx": self.run_idx, "origin": self.generation_origin},
            })
        if unrecorded is not None:
            self.observe_request(index, "failed", "Response preserved, request record unavailable")
            raise unrecorded
        self.observe_request(index, "succeeded", "Reply received. Processing it next.")
        return _SlotResponse(slot=i, code=code, usage=usage_total,
                             requests=outcome.attempts + extra,
                             unknown_usage_requests=outcome.unknown_usage_requests + extra,
                             text=text, delivery_id=delivery_id)

    def send(self, seed: int, attempt: int) -> Tuple[str, Dict[str, int]]:
        self.in_flight.attempt = attempt
        if not self.configured_generator:
            self.record_transport()
        with self.request_spans.request():
            value = self.gen(self.messages, temperature=self.temperature, seed=seed,
                             timeout=self.generation_timeout)
        # A reply that is no answer (a load reply, or one that read
        # nothing) is sent again under the transient rules, as in the
        # agent loops. Any other empty reply is a delivered response
        # without code, which spends the slot's attempt.
        text, usage = value
        if isinstance(usage, dict):
            reported = (model_requests.reported_usage(
                usage.get("input_tokens"), usage.get("output_tokens"), usage.get("cache_hit_tokens"))
                if usage.get("usage_known", True) else None)
            finish = usage.get("finish_reason")
            finish = finish if isinstance(finish, str) else None
            if model_requests.unanswered_reply(
                    text=text or "", tool_call=False, finish_reason=finish, usage=reported):
                raise model_requests.empty_reply_error(finish, reported)
        return value

    def usage_of(self, value: Any) -> Optional[model_requests.Usage]:
        _text, usage = value
        reported_model = usage.get("response_model") if isinstance(usage, dict) else None
        self.in_flight.response_model = (
            reported_model if isinstance(reported_model, str) and reported_model else None)
        if not isinstance(usage, dict) or not usage.get("usage_known", True):
            return None
        return model_requests.reported_usage(
            usage.get("input_tokens"), usage.get("output_tokens"), usage.get("cache_hit_tokens"))

    def on_retry(self, i: int, index: int, attempt: int, exc: BaseException, wait: float) -> None:
        self.log(f"[starter] starter suite {i} attempt {attempt} failed in transport: "
                 f"{type(exc).__name__}: {model_requests.failure_text(exc)}. Retrying in {wait:.0f}s")
        if wait > 0:
            self.observe_request(index, "waiting", f"Waiting {wait:.1f}s before retry")

    def persist_progress(self) -> None:
        # The generation manifest after each processed response, marked
        # incomplete: a run that a request failure or an interruption ends
        # leaves each slot's spent attempts and the totals so far.
        self.write_manifest(self.gen_start.seconds_at(time.monotonic()), False)

    def request_failure_result(self, exc: BaseException,
                               refill_started: Optional[float] = None) -> StarterResult:
        """End the phase on a terminal model request failure or a retry ceiling.

        The status is ``failed_model_request`` with the provider's error, never
        the zero-green status: the pool was not generated, not empty. The
        totals are those the generation started from plus every request this
        run made. The generation manifest is written marked incomplete, so a
        suite-reuse run continues the generation from each slot's spent
        attempts instead of granting them again. ``refill_started`` is the
        start of a refill after the measurement, whose generation time is
        recorded already: only the refill's time is added then."""
        self.slot_evidence.finish("generation ended on a model request failure")
        res, start, account = self.res, self.gen_start, self.account
        kind = (
            "terminal model request failure"
            if isinstance(exc, model_requests.TerminalRequestError)
            else "sent again until no attempt was left"
        )
        res.status = "failed_model_request"
        res.detail = f"{kind}: {exc}"
        res.requests_used = start.requests_used + account.physical_requests
        # A model request counts once one of its requests was sent. A slot that
        # the failure stopped before its first attempt made none.
        res.logical_requests = start.logical_requests + len(
            {record.logical_index for record in account.records})
        res.unknown_usage_requests = (
            start.unknown_usage_requests + account.unknown_usage_requests)
        now = time.monotonic()
        if refill_started is None:
            res.generation_seconds = start.seconds_at(now)
        else:
            res.generation_seconds += now - refill_started
        res.duration_s = start.seconds_at(now)
        phase_state.record_time(phase_state.TIME_MODEL,
                                self.request_spans.seconds(since=refill_started))
        self.write_manifest(res.generation_seconds, False)
        self.log(f"[starter] phase ended: {res.detail}")
        return res

    def drop_delivery(self, i: int) -> None:
        # Once its processing is committed with the manifest, the saved
        # response has served its purpose. Each caller drops it after the
        # response's check returned, which records a delivery id as
        # processed.
        (self.suites_root / slot_name(i) / _GENERATION_DELIVERY).unlink(missing_ok=True)

    def check_response(self, response: _SlotResponse, *, count_totals: bool = True,
                       generation_inputs: Optional[dict] = None, slot_attempt: int, seed: int) -> None:
        i, code, usage, attempts, unknown, text, delivery_id = response
        # A delivered response is salvaged, also when a request failure is
        # ending the phase: it was paid for, and a reuse run keeps its
        # outcome instead of drawing again. Its attempt was counted when it
        # arrived. ``count_totals`` is false for a response an interrupted
        # run saved, whose requests the restored totals already hold.
        if generation_inputs is None:
            generation_inputs = {"origin": self.generation_origin if count_totals else None,
                                 "run_idx": self.run_idx if count_totals else None}
        if count_totals:
            self.count_response(attempts, unknown, usage)
        sdir, unbound_recheck, meta = self.write_reply(response, count_totals, slot_attempt, seed)
        if code is None:
            (sdir / "original.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
            self.slot_evidence.record(
                i, delivery_id, classification="no_code",
                basis="no_code" if text is not None else "unavailable",
                reason="delivered response contained no extracted code" if text is not None
                else "response text unavailable",
                origin=generation_inputs["origin"], run_idx=generation_inputs["run_idx"],
                slot_attempt=slot_attempt, seed=seed)
            if delivery_id is not None:
                self.processed_deliveries.add(delivery_id)
            return
        original_path = _suite_source_path(sdir, "original", self.backend)
        if not (unbound_recheck and original_path.is_file()):
            original_path.write_bytes(code.encode("utf-8"))
        meta["source_sha256"] = source_sha256(code)
        classification, cleaned, cleanup_record = self.salvage(i, code, sdir, meta)
        meta["classification"] = classification
        if cleaned is None and cleanup_record.get("disposition_basis") == "accepted_cleaned":
            cleanup_record.update(disposition_basis="unavailable",
                                  disposition_reason="cleanup did not deliver an accepted source")
        (sdir / "original.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        self.tally(i, classification, cleaned, cleanup_record, sdir)
        # Counted when the processing completes, and committed with the
        # manifest together with the delivery id: an interrupted processing
        # counts nothing and runs again on reuse.
        self.slot_evidence.record(
            i, delivery_id, classification=classification,
            basis=cleanup_record.get("disposition_basis", "unavailable"),
            original_source=code, cleaned_source=cleaned,
            suffix=self.backend.test_file_extensions()[0],
            reason=cleanup_record.get("disposition_reason", "cleanup judgement unavailable"),
            origin=generation_inputs["origin"], run_idx=generation_inputs["run_idx"],
            slot_attempt=slot_attempt, seed=seed)
        self.res.n_generated += 1
        if delivery_id is not None:
            self.processed_deliveries.add(delivery_id)
        self.log(f"[starter] starter suite {i} attempt {self.suite_attempts.get(i, 0)}/"
                 f"{self.attempt_limit}: {classification}")

    def count_response(self, attempts: int, unknown: int, usage: Dict[str, float]) -> None:
        self.res.requests_used += attempts
        self.res.logical_requests += 1
        self.res.unknown_usage_requests += unknown
        self.res.input_tokens += usage["input_tokens"]
        self.res.output_tokens += usage["output_tokens"]
        self.res.cache_hit_tokens += int(usage.get("cache_hit_tokens") or 0)
        unknown_cache = usage.get("unknown_cache_requests")
        self.res.unknown_cache_requests = (
            None if unknown_cache is None or self.res.unknown_cache_requests is None
            else self.res.unknown_cache_requests + int(unknown_cache))

    def write_reply(self, response: _SlotResponse, count_totals: bool, slot_attempt: int, seed: int):
        i, code, usage, attempts, unknown, text, delivery_id = response
        sdir = self.suites_root / slot_name(i)
        sdir.mkdir(parents=True, exist_ok=True)
        # The model's reply verbatim, beside the code block extracted from
        # it. Without this a suite that produced no usable block, or whose
        # fence was picked wrongly, leaves nothing to diagnose, and the
        # phase is the only one that keeps no transcript of its own.
        # A saved response checked again without its paid attempt's binding
        # (a reused suite whose check turned unresolved) keeps the files it
        # was saved with.
        unbound_recheck = delivery_id is None and not count_totals
        if not unbound_recheck:
            # The slot's earlier attempt keeps its files under attempts/.
            _archive_earlier_attempt(sdir, slot_attempt, self.backend)
        if text is not None and not (unbound_recheck and (sdir / "response.txt").is_file()):
            (sdir / "response.txt").write_bytes(text.encode("utf-8"))
        meta = {
            "i": i, "slot_attempt": slot_attempt, "temperature": self.temperature, "seed": seed,
            "attempts": attempts, "unknown_usage_requests": unknown, **usage,
            "extracted": code is not None, "response_sha256": _or_none(source_sha256, text),
            "response_chars": _or_none(len, text)}
        return sdir, unbound_recheck, meta

    def salvage(self, i: int, code: str, sdir: Path, meta: Dict[str, Any]):
        # The salvage writes a candidate and runs it in a scratch directory it
        # removes. The per-suite artifacts above stay in the (host) suites
        # directory.
        with project.scratch_dir(prefix=f"testuary_salvage_{i:03d}_") as td:
            scratch = Path(td)
            normalization_record: Dict[str, Any] = {}
            validity_record: Dict[str, Any] = {}
            cleanup_record: Dict[str, Any] = {}
            classification, cleaned = starter_salvage._clean_suite(
                self.backend, code, self.target_file_path, self.source_dir, scratch,
                timeout=self.timeout, log=self.log,
                reject_unsound=self.cfg["reject_unsound_async_green"],
                final_attempt=(not self.exact or self.suite_attempts.get(i, 0) >= self.attempt_limit),
                failing_cleanup_rounds=self.failing_cleanup_rounds,
                normalization_out=normalization_record, validity_out=validity_record,
                cleanup_out=cleanup_record)
            confirmation_source = cleanup_record.pop("confirmation_source", None)
            if confirmation_source is not None:
                confirmation_path = _suite_source_path(sdir, "unconfirmed", self.backend)
                confirmation_path.write_bytes(confirmation_source.encode("utf-8"))
                cleanup_record["confirmation"]["source"] = confirmation_path.name
            meta["normalization"] = normalization_record
            meta["cleanup"] = cleanup_record
            if validity_record:
                validity.write_json_atomic(sdir / "validity.json", validity_record)
                meta["validity_record"] = "validity.json"
            # Derivations a processing that an interruption cut short wrote.
            (sdir / _SLOT_DERIVATIONS).unlink(missing_ok=True)
            if cleaned is not None:
                # A salvaged-green suite that exercises no line of the target is
                # not a usable suite: it adds nothing to the pool, never enters
                # the saturation curves, yet would otherwise count toward
                # refill_failed_suites. Probe its target line coverage once and
                # reclassify a zero-coverage suite. A backend that cannot
                # attribute line coverage (line_coverage is None) keeps the
                # suite: there is no signal to gate on, not a wrong value.
                cov = self.probe(scratch, cleaned)
                if not getattr(cov, "ok", False):
                    # The suite passes, but the coverage tool could not measure
                    # it: the tests its run names are removed, within the
                    # cleanup's rounds, and the suite is run and probed again.
                    cleaned, cov = self.removed_at_probe(scratch, sdir, cleaned, cov,
                                                         cleanup_record)
                if cleaned is None:
                    # Without complete numbers its slot is free again.
                    classification = "coverage_unmeasured"
                    from testuary.consolidation import red_verdict

                    judged = getattr(cov, "timed_out", None) is False
                    cleanup_record.update(
                        disposition_basis="judged_rejection" if judged else "unavailable",
                        disposition_reason=(
                            "the coverage tool could not measure the suite: " + red_verdict(cov)))
                elif cov.line_coverage is not None and cov.line_coverage <= 0.0:
                    classification, cleaned = "zero_coverage", None
                    judged = getattr(cov, "ok", None) is True and getattr(cov, "timed_out", None) is False
                    cleanup_record.update(
                        disposition_basis="judged_rejection" if judged else "unavailable",
                        disposition_reason="completed target coverage was zero" if judged
                        else "target coverage probe did not establish rejection")
        return classification, cleaned, cleanup_record

    def probe(self, scratch: Path, source: str) -> Any:
        """The zero-coverage probe's coverage run of ``source``."""
        probe = self.backend.staged_test_path(str(scratch), "candidate", source)
        probe.write_text(source, encoding="utf-8")
        return self.backend.run_test_files(
            [str(probe)], self.source_dir, timeout=self.timeout,
            measure_coverage_of=self.target_file_path)

    def removal_cause(self, source: str) -> str:
        """Why ``source``, a starter suite after a removal, cannot stay ("" when
        its normal run passes)."""
        from testuary.consolidation import _removal_cause

        with project.scratch_dir(prefix="testuary_removal_") as scratch:
            path = self.backend.staged_test_path(str(scratch), "candidate", source)
            path.write_text(source, encoding="utf-8")
            run = normal_run.run(self.backend, [str(path)], self.source_dir,
                                 timeout=self.timeout, target=self.target_file_path)
        return _removal_cause(self.backend, source, run)

    def removed_at_probe(self, scratch: Path, sdir: Path, source: str, cov: Any,
                         cleanup_record: Dict[str, Any]) -> Tuple[Optional[str], Any]:
        """``source`` without the tests that the probe's failing coverage run
        ``cov`` names, run and probed again within the cleanup's rounds,
        and the probe's last run. The source is None while the probe
        still fails. The entries go into the cleanup record."""
        from testuary.consolidation import _Failed, _failed_run, _remove_named_failures

        place = "starter.zero_coverage_probe"
        runs = [cov]

        def measure(changed: str):
            cause = self.removal_cause(changed)
            if cause:
                return _Failed("coverage", cause=cause)
            runs.append(self.probe(scratch, changed))
            return None if getattr(runs[-1], "ok", False) else _failed_run("coverage", runs[-1])

        changed, failed, entries = _remove_named_failures(
            self.backend, source, place=place,
            failed=_failed_run("coverage", cov), measure=measure,
            rounds=self.failing_cleanup_rounds, log=self.log)
        cleanup_record["measurement_removals"] = entries
        if failed is not None:
            return None, runs[-1]
        self.derive_removal(sdir, place, source, changed, entries, fresh=True)
        return changed, runs[-1]

    def derive_removal(self, sdir: Path, place: str, parent: str, child: str,
                       entries: List[Dict[str, Any]], *, fresh: bool = False) -> None:
        """The validity derivation of ``child`` from ``parent`` by a removal at
        ``place``, after the slot's earlier ones (none when ``fresh``) in its
        ``validity_derivations.json``."""
        path = sdir / _SLOT_DERIVATIONS
        records = [] if fresh or not path.is_file() else list(
            validity.load_record(path).get("records") or [])
        records.append(validity.derivation_record(
            identity=self.backend.generated_test_target_identity(
                self.target_file_path, self.source_dir),
            parent_source=parent, child_source=child, operation="measurement_remove",
            removed_tests=[item["test"] for entry in entries for item in entry.get("removed", ())],
            stage=place))
        validity.write_json_atomic(path, {"schema_version": validity.RECORD_SCHEMA_VERSION,
                                          "records": records, **phase_state.timing_fields()})

    def tally(self, i: int, classification: str, cleaned: Optional[str],
              cleanup_record: Dict[str, Any], sdir: Path) -> None:
        self.unresolved.discard(i)
        if classification == "validation_unresolved":
            self.res.n_validation_unresolved += 1
            self.unresolved.add(i)
            self.unresolved_detail[:] = [str(cleanup_record.get("validation_unresolved") or "")]
        elif classification == "discarded_syntax":
            self.res.n_discarded_syntax += 1
        elif classification == "discarded_collection":
            self.res.n_discarded_collection += 1
        elif classification in ("unsalvageable", "timeout", "unsound", "invalid_generated_test"):
            self.res.n_unsalvageable += 1
        elif classification == "zero_coverage":
            self.res.n_zero_coverage += 1
        elif classification == "coverage_unmeasured":
            self.res.n_unmeasured_coverage += 1
        else:
            self.green[i] = cleaned
            _suite_source_path(sdir, "cleaned", self.backend).write_bytes(cleaned.encode("utf-8"))

    def process_response(self, *args, **kwargs):
        # A Ctrl+C during the response's test runs (the green check, the
        # cleanup rounds) stops them and removes their scratch, as for an
        # agent's tool.
        from testuary.languages.mutation import cancel_on_interrupt, tool_scope

        with cancel_on_interrupt(), tool_scope():
            return self.check_response(*args, **kwargs)

    def reprocess_unresolved(self) -> bool:
        """Reprocess saved, unresolved responses without charging an attempt."""
        slots = sorted(self.unresolved)
        for i in slots:
            sdir = self.suites_root / slot_name(i)
            try:
                meta = json.loads((sdir / "original.json").read_text(encoding="utf-8"))
                code = _archived_suite_text(_suite_source_path(sdir, "original", self.backend))
                recorded_attempt, recorded_seed = int(meta["slot_attempt"]), int(meta["seed"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"reuse_suites is set but the unvalidated starter suite {i} cannot "
                    f"be read: {exc}") from exc
            reply = sdir / "response.txt"
            text = (reply.read_bytes().decode("utf-8") if reply.is_file() else None)
            usage = {key: meta.get(key) or 0 for key in (
                "input_tokens", "output_tokens", "cache_hit_tokens")}
            # None when the suite record lacks it: unknown, not zero.
            usage["unknown_cache_requests"] = meta.get("unknown_cache_requests")
            self.log(f"[starter] starter suite {i}: checking again the saved response with "
                     "unresolved validation or execution, with no model request")
            # process_response counts the response and its outcome again.
            self.res.n_validation_unresolved -= 1
            self.res.n_generated -= 1
            self.process_response(
                _SlotResponse(slot=i, code=code, usage=usage, requests=int(meta.get("attempts") or 0),
                              unknown_usage_requests=int(meta.get("unknown_usage_requests") or 0),
                              text=text, delivery_id=self.slot_evidence.revalidation_id(i, code)),
                count_totals=False, slot_attempt=recorded_attempt, seed=recorded_seed)
        return bool(slots)

    def process_saved_responses(self, count_totals: bool) -> None:
        """Process each response an earlier run received and saved but did
        not process, because it was interrupted in between. Its attempt was
        spent then: no model request is made and no attempt charged, unless
        no request records restored the counts (``count_totals``)."""
        for saved in sorted(self.suites_root.glob(f"{SLOT_PREFIX}*/{_GENERATION_DELIVERY}")):
            try:
                delivery = json.loads(saved.read_text(encoding="utf-8"))
                i = int(delivery["slot"])
                slot_attempt, sent_seed = int(delivery["slot_attempt"]), int(delivery["seed"])
            except (OSError, ValueError, KeyError, TypeError) as exc:
                raise RuntimeError(
                    f"reuse_suites is set but the saved response {saved} "
                    f"cannot be read: {exc}") from exc
            delivery_id = delivery.get("id")
            if delivery_id is not None and delivery_id in self.processed_deliveries:
                # Processed and committed before the interruption. Only the
                # file was left.
                saved.unlink()
                continue
            text = delivery.get("text")
            code = _extract_code(text, self.fence) if text is not None else None
            # A response whose record could not be written is in no record:
            # its attempt and requests are counted here.
            counted = count_totals or bool(delivery.get("unrecorded"))
            if counted:
                self.suite_attempts[i] = self.suite_attempts.get(i, 0) + 1
                self.res.suite_generation_attempts[i] = self.res.suite_generation_attempts.get(i, 0) + 1
            self.log(f"[starter] starter suite {i}: processing the response an interrupted "
                     "run received, with no model request")
            self.observe_saved(i, "running",
                               "Using a response saved by an interrupted run. No new model request.")
            try:
                self.process_response(
                    _SlotResponse(slot=i, code=code, usage=dict(delivery.get("usage") or {}),
                                  requests=int(delivery.get("physical_requests") or 1),
                                  unknown_usage_requests=int(delivery.get("unknown_usage_requests") or 0),
                                  text=text, delivery_id=delivery_id),
                    count_totals=counted,
                    generation_inputs=generation_evidence.read_paid_inputs(delivery),
                    slot_attempt=slot_attempt, seed=sent_seed)
            except BaseException:
                self.observe_saved(i, "failed", "Saved response processing interrupted")
                raise
            finally:
                self.write_manifest(self.res.generation_seconds or 0.0, False)
            self.drop_delivery(i)
            self.observe_saved(i, "succeeded", "Saved response processing recorded, no new request")

    def restore_reused(self):
        # Reuse a prior run's persisted suites instead of paying the model
        # generation again, then complete the pool exactly as a fresh run would:
        # drop any reused suite that runs no test, and let the refill below fill
        # every empty and never-generated slot. A generation that a request
        # failure or an interruption ended continues from its spent attempts.
        restored_complete = _load_persisted_green(
            self.suites_root, self.green, self.res, self.log, backend=self.backend,
            source_dir=self.source_dir, unresolved=self.unresolved)
        # The request records are the account of what earlier runs spent:
        # every slot's delivered attempts, including responses they never
        # processed, the transient failures the phase total bounds, and the
        # request and token totals. An interrupted final record is moved
        # aside first.
        # Each count is the larger of the records' and the manifest's, so a
        # damaged record file cannot lower what the manifest holds.
        aside = _repair_request_records(self.suites_root)
        if aside is not None:
            self.log(f"[starter] the request records ended with an interrupted "
                     f"line and are moved to {aside.name}")
        ledger = _read_generation_ledger(self.suites_root)
        manifest_record: Dict[str, Any] = {}
        if (self.suites_root / _GENERATION_MANIFEST).is_file():
            manifest_record = json.loads((self.suites_root / _GENERATION_MANIFEST).read_text(encoding="utf-8"))
        self.slot_evidence = generation_evidence.SuiteSlots(
            self.suites_root, self.suite_count, previous=manifest_record)
        self.processed_deliveries.update(manifest_record.get("processed_deliveries") or ())
        transient = _restore_totals(self.res, ledger, int(manifest_record.get("transient_failures") or 0))
        # A resumed session is a new session: its retry limits count only its
        # own failures, so an outage that ended an earlier session does not
        # refuse this one. The earlier failures stay in the request records.
        self.earlier_transient["failures"] = transient
        # A spent refill allowance stays spent across a reuse run: the gate
        # starts from the archived per-slot counts, so an interrupted run
        # receives no more draws than an uninterrupted one. A pool archived
        # without counts carries none, so its slots are treated as unspent.
        if self.res.suite_generation_attempts:
            self.suite_attempts.update(self.res.suite_generation_attempts)
        else:
            self.log("[starter] reused pool carries no per-slot attempt counts, so "
                     "its slots are treated as unspent")
        self.process_saved_responses(count_totals=ledger is None)
        reprocessed = self.reprocess_unresolved()
        n_dropped = _revalidate_reused_pool(
            self.green, self.backend, self.source_dir, timeout=self.timeout,
            log=self.log, instrument_target=self.target_file_path)
        if n_dropped:
            self.log(f"[starter] {n_dropped} reused starter suite(s) dropped as empty, and their "
                     "slots are generated again")
        reused_evidence_changed = self.slot_evidence.reconcile_reused_sources(self.green)
        self.gen_start = _GenerationStart(
            self.res.requests_used, self.res.logical_requests or 0,
            self.res.unknown_usage_requests or 0, self.res.generation_seconds or 0.0,
            time.monotonic())
        return restored_complete, reprocessed, reused_evidence_changed

    def run_wave(self, indices: List[int]) -> None:
        _enter("generate")
        nonlocal_results = []
        total = len(indices)
        # A slow reasoning model can take many minutes per generation, so
        # announce the wave before the first result lands: the phase reports
        # it is alive and how many generations are in flight, rather than
        # going silent until a generation returns.
        self.log(f"[starter] generating {total} starter suites, up to "
                 f"{min(self.concurrent_requests, total)} at a time, each reported as it returns")
        fatal: Optional[BaseException] = None
        # The attempt each slot requests: one more than it has spent.
        requested = {i: self.suite_attempts.get(i, 0) + 1 for i in indices}
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(self.concurrent_requests, total)) as ex:
            futures = [ex.submit(self.generate_suite, i, requested[i]) for i in indices]
            fatal = self.collect_wave(futures, total, nonlocal_results, fatal)

        for result in sorted(nonlocal_results, key=lambda item: item.slot):
            try:
                self.process_response(result, slot_attempt=requested[result.slot], seed=attempt_seed(
                    self.seed_base, result.slot, requested[result.slot], self.run_idx))
            finally:
                self.persist_progress()
            self.drop_delivery(result.slot)
        if fatal is not None:
            raise fatal

    def phase_seconds(self) -> float:
        # The phase's duration: the generation's and the time since it ended.
        return round(self.res.generation_seconds + (time.monotonic() - self._after_gen), 3)

    def no_green_result(self) -> StarterResult:
        # No starter suite passes: unresolved validation keeps the responses
        # for the next run, otherwise the phase is cancelled.
        if self.res.n_validation_unresolved:
            self.res.status = "validation_unresolved"
            self.res.detail = (
                f"validation or suite execution remains unresolved for "
                f"{self.res.n_validation_unresolved} of {self.res.n_generated} generated "
                f"suites ({self.unresolved_detail[0] if self.unresolved_detail else ''}). "
                f"Their responses are kept and validated again on resume")
        else:
            self.res.status = "cancelled_zero_passing"
            self.res.detail = _zero_green_detail(self.res)
            if self.output_limit_replies["hits"]:
                self.res.detail = (_output_limit_detail(self.output_limit_replies["hits"],
                                                        self.output_limit_replies["replies"], self._max_out)
                                   + ". " + self.res.detail[:1].upper() + self.res.detail[1:])
        self.res.duration_s = self.phase_seconds()
        return self.res

    def free_and_refill(self, freed: Dict[int, Tuple[str, str]]) -> Optional[StarterResult]:
        # Each freed slot's suite leaves the pool, its records say why, and
        # the slot is generated again within its attempts. The refill is
        # generation time, so the measurements' share of the phase stays
        # what they took. Returns the result of a request failure.
        for i, (classification, reason) in sorted(freed.items()):
            self.log(f"[starter] starter suite {i} leaves the pool: {reason}")
            self.green.pop(i, None)
            if classification == "coverage_unmeasured":
                self.res.n_unmeasured_coverage += 1
            else:
                self.res.n_unmeasured_mutation += 1
            self.slot_evidence.withdraw(i, classification, reason)
            meta_path = self.suites_root / slot_name(i) / "original.json"
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta = None
            if isinstance(meta, dict):
                meta["classification"] = classification
                meta.setdefault("cleanup", {}).update(
                    disposition_basis="unavailable", disposition_reason=reason)
                meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
        started = time.monotonic()
        refilled = False
        while self.exact and not self.unresolved:
            retry = [i for i in sorted(freed)
                     if i not in self.green and self.suite_attempts.get(i, 0) < self.attempt_limit]
            if not retry:
                break
            refilled = True
            try:
                self.run_wave(retry)
            except _REQUEST_FAILURES as exc:
                return self.request_failure_result(exc, started)
        self.slot_evidence.finish("generation ended without a completed response")
        if refilled:
            seconds = time.monotonic() - started
            self.res.generation_seconds += seconds
            self._after_gen += seconds
            phase_state.record_time(phase_state.TIME_MODEL, self.request_spans.seconds(since=started))
        self.write_manifest(self.res.generation_seconds, not self.unresolved)
        _enter("measure")
        return None

    def note_delivery(self, result: _SlotResponse, done: int, total: int):
        idx, code, _usage, attempts, _unknown, _text, _id = result
        # A delivered response spends the slot's attempt.
        self.suite_attempts[idx] = self.suite_attempts.get(idx, 0) + 1
        self.res.suite_generation_attempts[idx] = self.res.suite_generation_attempts.get(idx, 0) + 1
        outcome = "extracted" if code is not None else "no code block"
        self.output_limit_replies["replies"] += 1
        if (code is None and self._max_out and not (_text or "").strip()
                and _usage.get("output_tokens", 0) >= self._max_out):
            self.output_limit_replies["hits"] += 1
        self.log(f"[starter] generated {done}/{total}: starter suite {idx} "
                 f"attempt {self.suite_attempts[idx]}/{self.attempt_limit} "
                 f"({outcome}, out_tokens={_usage.get('output_tokens', 0)}, "
                 f"requests={attempts})")

    def collect_wave(self, futures: list, total: int, nonlocal_results: list, fatal: Optional[BaseException]):
        try:
            # Log each suite as its generation returns, so a wave on a slow
            # model reports progress instead of going silent until the
            # whole wave completes. Salvage of each result is logged below.
            for done, fut in enumerate(_completed_in_slices(futures), start=1):
                try:
                    result = fut.result()
                except _REQUEST_FAILURES as exc:
                    # The phase ends. The other workers stop before their
                    # next attempt, and their waits end at once.
                    if fatal is None:
                        fatal = exc
                    self.account.cancel.set()
                    continue
                except model_requests.RequestCancelled:
                    continue
                except Exception:
                    # Anything else, such as a request record that cannot
                    # be written, ends the phase: the other workers stop
                    # before their next attempt, and the manifest keeps
                    # what this run counted, including failures a record
                    # may lack. Responses already saved are processed by
                    # the next reuse run.
                    self.account.cancel.set()
                    try:
                        self.persist_progress()
                    except Exception:
                        pass  # the failure below is the one to report
                    raise
                nonlocal_results.append(result)
                self.note_delivery(result, done, total)
        except BaseException as exc:
            # An interruption (Ctrl+C) ends the phase the same way: the
            # workers' waits end at once, so leaving the executor never
            # waits out a Retry-After hint, and the manifest keeps what
            # this run counted for the resumed run.
            self.account.cancel.set()
            if not isinstance(exc, Exception):
                try:
                    self.persist_progress()
                except Exception:
                    pass  # the interruption is the one to report
            raise
        return fatal

    def generate_fresh(self, wave: List[int]):
        if self.reuse_suites:
            self.log("[starter] no persisted starter pool for this target, so reuse falls "
                     "back to a fresh generation")
        # Establish the authoritative zero-processing commit before draft suite
        # files appear. A failed first response commit must not make those drafts
        # look like completed work during recovery.
        self.write_manifest(0.0, False)
        # Cache warmup: the suites share one large prefix (system prompt + target
        # source), so a cold concurrent wave misses the provider prompt cache on
        # every request. Generate the first suite alone to populate the cache,
        # then the rest concurrently so they hit it. Only the first wave is cold.
        # The refill (refill_failed_suites) below runs warm.
        try:
            if self.cfg["cache_warmup"] and len(wave) > 1:
                self.log("[starter] cache warmup: first starter suite alone, then the batch")
                self.run_wave(wave[:1])
                if self.unresolved:
                    self.log("[starter] the first starter suite has unresolved validation "
                             "or execution, so the batch is not requested")
                else:
                    self.run_wave(wave[1:])
            else:
                self.run_wave(wave)
        except _REQUEST_FAILURES as exc:
            return self.request_failure_result(exc)

    def continue_ended(self, wave: List[int]):
        # Without the refill each slot has one attempt. A generation that
        # ended early generates every slot that never received a response,
        # as the uninterrupted run would have. A slot that did keeps it.
        pending = [i for i in wave if i not in self.green and self.suite_attempts.get(i, 0) == 0]
        if pending and not self.unresolved:
            try:
                self.run_wave(pending)
            except _REQUEST_FAILURES as exc:
                return self.request_failure_result(exc)

    def refill_failed(self, wave: List[int]):
        while self.exact and len(self.green) < self.suite_count and not self.unresolved:
            failed = [i for i in wave
                      if i not in self.green and self.suite_attempts.get(i, 0) < self.attempt_limit]
            if not failed:
                break
            try:
                self.run_wave(failed)
            except _REQUEST_FAILURES as exc:
                return self.request_failure_result(exc)

    def generate(self):
        # The suite slots are fixed at 1..suite_count: a re-roll refills a slot in place,
        # so no index is ever added or shifted, and slot i's attempt a always draws
        # attempt_seed(seed_base, i, a, run_idx) no matter how many attempts any other slot
        # spends.
        wave = list(range(1, self.suite_count + 1))
        took_reuse = self.reuse_suites and self.archived_generation
        restored_complete = True
        self.gen_start = _GenerationStart(0, 0, 0, 0.0, self.t0)
        if took_reuse:
            restored_complete, reprocessed, reused_evidence_changed = self.restore_reused()
        else:
            ended = self.generate_fresh(wave)
            if ended is not None:
                return ended

        # The refill (refill_failed_suites) fills every suite slot that is not green: a generation or
        # salvage that failed, a reused suite dropped as empty, or a slot a short
        # saved pool never held. Re-rolled in place at its own index, with the
        # attempt's own sampling seed (attempt_seed), so the green pool is always a
        # subset of the fixed set 1..suite_count, directly comparable to the
        # zero-shot prompt's suites. Each slot
        # spends its own attempts_per_suite attempts and no other slot's,
        # so a slot the model keeps failing on cannot starve the rest, and a
        # deterministic endpoint reproducing a failure runs out of that slot's
        # attempts instead of looping. A pool that ends below suite_count is what a
        # hostile file legitimately yields. Failed attempts already count toward
        # requests/tokens.
        if took_reuse and not self.exact and not restored_complete:
            ended = self.continue_ended(wave)
            if ended is not None:
                return ended
        ended = self.refill_failed(wave)
        if ended is not None:
            return ended

        # Generation wall-clock: the elapsed time up to consolidation (the parallel
        # suite waves merge into it). A fresh run (including a reuse that fell back to
        # fresh for a never-run target) times from the phase start and persists the
        # manifest. A reuse run restored the time and totals from the manifest. If it
        # refilled any slot, or continued a generation that had ended early, merge
        # that refill's wall-clock and cost in and re-persist the manifest marked
        # complete, so a later reuse restores the completed pool.
        # The waves run in parallel, so the model time this phase spent is the
        # time some request was in flight (_RequestSpans), and it is what this
        # run spent: a reuse that generated nothing spends none of it, and one
        # that refilled a slot spends only the refill.
        self._after_gen = time.monotonic()
        self.slot_evidence.finish("generation ended without a completed response")
        if (not took_reuse or self.res.requests_used > self.gen_start.requests_used
                or not restored_complete or reprocessed):
            self.res.generation_seconds = self.gen_start.seconds_at(self._after_gen)
            phase_state.record_time(phase_state.TIME_MODEL, self.request_spans.seconds())
            self.write_manifest(self.res.generation_seconds, not self.unresolved)
        elif reused_evidence_changed:
            # Metadata-only correction must not add model time or change completion.
            self.write_manifest(self.res.generation_seconds, restored_complete and not self.unresolved)
        self.res.n_green = len(self.green)

    def prepare(self):
        self.t0 = time.monotonic()
        # Test runs execute with cwd at the source directory, so the subprocess that
        # runs the tests needs absolute paths. The target, its source directory, and
        # the work directory are already absolute inside the container. Resolving them
        # on a Windows host would prepend a drive and flip the separators, so pass
        # them through unchanged in container mode.
        if config.project_exec_mode() != "container":
            self.target_file_path = str(Path(self.target_file_path).resolve())
            self.source_dir = str(Path(self.source_dir).resolve())
            self.test_output_dir = str(Path(self.test_output_dir).resolve())
        # suites_dir is a host-side directory in either mode.
        self.suites_dir = str(Path(self.suites_dir).resolve())
        self.cfg = config.get_agent_config("test_initial")
        self.failing_cleanup_rounds = self.cfg["failing_test_cleanup_rounds"]
        self.backend = languages.get_backend(self.language)
        if self.backend is None:
            raise ValueError(f"no language adapter for {self.language!r}")
        self.api_cfg = models.get_api_config()
        self.generator_source = model_source.current()
        run_settings = config.active_settings()
        # Looked up when called, as every other caller does, so a replacement of
        # models.resolve_phase_model ends where it is undone.
        self.phase_model = models.resolve_phase_model(run_settings, "starter") if (
            run_settings is not None and self.generator_source is None) else None
        # The language the tests are written in is the adapter's language level,
        # as for every agent (Python's interpreter version without its patch, a
        # stated [project] runtime_version read the same way).
        self.prompts = prompts_module.resolve_agent_prompts(
            "test_initial", self.language, source_dir=self.source_dir,
            hint_profile=hints.hint_profile())
        self.fence = self.backend.prompt_fragments().get("lang_code_fence", "")
        self.res = StarterResult(n_sets_requested=int(self.cfg["suite_count"]))
        # Per-phase header, same shape the coverage / mutation phases log: the
        # operator can read the target, language, and output dir off the boundary
        # marker without scrolling into the per-suite detail.
        self.log(f"[starter] target: {self.target_file_path}")
        self.log(f"[starter] language: {self.language}")
        # The exact path / import / runtime the prompt carries, so the operator can
        # verify they are the in-container values (the relative path resolves under
        # Docker / Apptainer, the import line is right, the runtime is the target's).
        self.log(f"[starter] prompt path: {project.display_path(self.target_file_path, self.source_dir)}")
        self.log(f"[starter] prompt import: "
                 f"{_import_hint(self.language, self.target_file_path, self.source_dir)}")
        self.log(f"[starter] prompt runtime: "
                 f"{prompts_module._language_level(self.backend, self.source_dir) or '(backend default)'}")
        self.log(f"[starter] starter suites: {self.res.n_sets_requested}")
        self.log(f"[starter] test output dir: {self.test_output_dir}")
        self._max_out = _resolve_output_cap(self.cfg, self.api_cfg)
        if self.phase_model is not None:
            self._max_out = self.phase_model.values["max_output_tokens"]
        # Replies that spent the whole output limit and gave no text. Only a
        # phase that ends without a green suite reports them.
        self.output_limit_replies = {"hits": 0, "replies": 0}
        self._grace = self.cfg.get("reasoning_grace_seconds")
        self._grace = float(self._grace) if self._grace else None
        # The request in flight on this worker thread, for the context-overflow
        # hook: each halved request is a request sent for the same slot
        # attempt and is recorded as such.
        self.in_flight = threading.local()

    def prepare_generator(self):
        # An installed model source (``models.install``) wins over the
        # configured provider.
        self.model_clients = None
        self.configured_generator = self.generator_source is None
        # The phase model is resolved only for the configured generator.
        if self.phase_model is not None and self.phase_model.protocol != "ollama-native":
            from testuary.models import StarterModelClients

            self.model_clients = (self.prepared_models.take(self.phase_model, config.active_run_scope())
                                  if self.prepared_models is not None else None)
            if self.model_clients is None:
                self.model_clients = StarterModelClients(self.phase_model, api_key=self.api_cfg["api_key"])
        self.gen = (
            self.generator_source.generate_suite if self.generator_source is not None else
            lambda messages, temperature, seed, timeout: _default_generate(
                self.api_cfg, messages, seed=seed, reasoning_grace=self._grace,
                on_context_overflow=self._on_context_overflow,
                on_request=self._record_sent_transport, model_settings=self.phase_model,
                model_clients=self.model_clients))

        source = project.read_file_content(self.target_file_path)
        user_prompt = Template(self.prompts["user_prompt_template"]).substitute(
            file_path=project.display_path(self.target_file_path, self.source_dir),
            import_hint=_import_hint(self.language, self.target_file_path, self.source_dir),
            source=source)
        system = self.prompts["system_prompt"]
        self.messages = ({"role": "system", "content": system}, {"role": "user", "content": user_prompt})

    def open_records(self):
        self.temperature = self.cfg["temperature"]
        if self.phase_model is not None:
            self.temperature = self.phase_model.values["temperature"]
        if self.temperature is not None:
            self.temperature = float(self.temperature)
        self.seed_base = self.cfg["seed_base"]
        self.timeout = int(self.cfg["suite_run_timeout_seconds"])
        self.generation_timeout = (self.phase_model.values["request_timeout_seconds"]
                                   if self.phase_model is not None else self.timeout)
        self.concurrent_requests = _resolve_concurrent_requests(self.cfg)
        self.suite_count = self.res.n_sets_requested
        self.exact = self.cfg["refill_failed_suites"]
        self.attempt_limit = int(self.cfg["attempts_per_suite"])
        # The caller's choice (a resume reuses the saved starters), else the
        # setting.
        self.reuse_suites = (self.cfg["reuse_suites"] if self.reuse_suites is None
                             else self.reuse_suites)
        self.suites_root = Path(self.suites_dir)
        from testuary.evidence import current_condition, generation_only, require_origin, write_origin

        # The starter pool records how its suites were generated. How a run
        # measures them is the starter phase origin's (generation_only).
        self.generation_condition = generation_only(
            current_condition("starter", self.language, self.source_dir, templates=self.prompts)
            if self.configured_generator else None)
        origin_path = self.suites_root / "generation_origin.json"
        self.archived_generation = _has_persisted_pool(self.suites_root)
        if self.reuse_suites and self.archived_generation:
            require_origin(origin_path, self.generation_condition)
        else:
            write_origin(origin_path, self.generation_condition)
        self.transport_path = self.suites_root / "prompt_transport.jsonl"
        self.records_path = self.suites_root / "request_records.jsonl"
        self.transport_lock = threading.Lock()
        self.transport_request_idx = 0
        if not self.reuse_suites:
            self.transport_path.parent.mkdir(parents=True, exist_ok=True)
            self.transport_path.write_text("", encoding="utf-8")
            self.records_path.write_text("", encoding="utf-8")
            # A fresh generation starts a new ledger. A response an earlier run
            # left unprocessed is not part of it.
            for stale in self.suites_root.glob(f"{SLOT_PREFIX}*/{_GENERATION_DELIVERY}"):
                stale.unlink()
        # The run and slot of every request sent, so that a reuse run can
        # recover each slot's spent attempts and retries from the records.
        self.run_id = uuid.uuid4().hex
        self.generation_scope = generation_evidence.construction_scope("starter", self.run_idx, self.run_id)
        self.generation_origin = (generation_evidence.capture_origin(origin_path)
                                  if self.generation_condition is not None else None)
        self.slot_of: Dict[int, int] = {}
        # Transient failures of earlier sessions of this phase. A resumed session
        # starts its own retry counts at zero. The manifest keeps the sum.
        self.earlier_transient = {"failures": 0}
        # The delivery ids of processed responses, committed with the manifest.
        self.processed_deliveries: set = set()
        self.slot_evidence = generation_evidence.SuiteSlots(self.suites_root, self.suite_count)

        from testuary._progress import phase_observer

        self.observer = phase_observer("starter")

        # The phase's state for requests sent again, shared by every slot's worker.
        self.request_spans = _RequestSpans()
        self.account = model_requests.RequestAccount(
            # count_retries is the agents' loop's setting, which [starter] does not take.
            model_requests.RetrySettings.from_config({**self.cfg, "count_retries": False}),
            on_record=self.record_request,
            service=model_requests.service_address(
                self.phase_model.identity.api_url if self.phase_model is not None
                else self.api_cfg.get("api_url")))

        self.green: Dict[int, str] = {}
        # Generation attempts this run has spent on each suite slot. The refill
        # limit applies per slot, so one slot running out leaves every other
        # slot's attempts untouched.
        self.suite_attempts: Dict[int, int] = {}
        # Unjudged validation or execution stops new generation. Preserve the
        # paid responses for reprocessing rather than spending refill attempts.
        self.unresolved: set = set()
        self.unresolved_detail: List[str] = []

    def phase_end_log(self, message: str) -> None:
        # The phase end's lines, under the starter phase's prefix.
        self.log(message.replace("[consolidate]", "[starter]"))

    def stage_alone(self, scratch_dir, i: int, *, write: bool = True) -> Path:
        # Slot i's suite in ``scratch_dir`` under the name its standalone run
        # gives it.
        path = self.backend.scratch_test_path(
            str(scratch_dir), self.green[i], self.target_file_path, stage_as_delivered=False)
        if write:
            path.write_text(self.green[i], encoding="utf-8")
        return path

    def measured_suites(self, done: int) -> None:
        # Display only: how many accepted suites are measured so far.
        if self.observer is not None:
            from testuary.presentation.view import STARTER_MEASUREMENT

            self.observer(operation=STARTER_MEASUREMENT,
                          state="succeeded" if done == len(self.order) else "running",
                          message="Starter suites measured", completed=done, total=len(self.order),
                          work_unit="starter suites measured")

    def derive(self, runner):
        from testuary.consolidation import red_verdict

        with project.scratch_dir(prefix="testuary_initkills_") as td:
            # Each suite's file as its standalone run stages it: the
            # backend's pool maps every staged name back to it.
            standalone: Dict[str, Tuple[str, str]] = {}
            for i in self.order:
                with project.scratch_dir(prefix="testuary_initname_") as nd:
                    spath = self.stage_alone(nd, i, write=False)
                    standalone[slot_name(i)] = (
                        Path(spath).relative_to(Path(nd)).as_posix(), self.green[i])
            self.standalone_names = {label: PurePosixPath(name).name
                                for label, (name, _source) in standalone.items()}
            self.pool, self.derivation_reason = starter_selection.stage_pool(
                self.backend, standalone, Path(td))
            if self.pool is not None:
                pool_paths = {staged.slot: Path(td) / staged.path for staged in self.pool.files}
                self.pool_names = {label: path.name for label, path in pool_paths.items()}
                # The tool runs the pool as one suite, where sibling
                # files can interfere, and measures only a pool that
                # passes there as it is: one that does not gives no
                # derivation, naming the tests that stop it, and no
                # test is removed from it.
                together = self.backend.run_test_files(
                    [str(path) for path in pool_paths.values()],
                    self.source_dir, timeout=self.evict_timeout)
                if not getattr(together, "ok", False):
                    self.derivation_reason = (
                        "the starter suites do not pass when run together: "
                        + red_verdict(together))
                else:
                    started = time.monotonic()
                    # Every killing test is recorded, so each suite's
                    # derivation sees its own kills.
                    self.pooled_km = runner(Path(td), attribution="all")
                    self.run_seconds["pooled"] = time.monotonic() - started
                    limit_of = getattr(runner, "limit_of", None)
                    self.pooled_limit = limit_of(Path(td)) if limit_of is not None else None
                    if self.pooled_km is None:
                        self.derivation_reason = "the pooled mutation run produced no result"
                    elif self.pooled_km.get("attribution") != "all":
                        # Without every killing test recorded (first killers,
                        # or a matrix that does not say), a kill is credited to
                        # one suite only, so no suite's kills can be derived.
                        self.derivation_reason = "the pooled mutation run did not record every killing test"
                        self.pooled_km = None
        if self.derivation_reason is not None:
            self.log(f"[starter] starter suites: no derived results: {self.derivation_reason}")
            if self.method == "derive":
                self.method_used = "measure"
                self.log("[starter] starter suites: measuring each suite alone instead")

    def measure_alone(self, runner):
        from testuary.consolidation import _Failed, _removed_entry

        started = time.monotonic()
        with contextlib.ExitStack() as staging:
            suite_dirs: Dict[str, Path] = {}
            for i in self.order:
                sd = staging.enter_context(
                    project.scratch_dir(prefix=f"testuary_starter{i:03d}_"))
                spath = self.stage_alone(sd, i)
                suite_dirs[slot_name(i)] = Path(sd)
                self.staged_names[slot_name(i)] = spath.name
            self.measured = runner.per_suite(suite_dirs)
            # The tests each suite's checks named, for the removal.
            self.mutation_named = {name: tuple(runner.named_of(path))
                                   for name, path in suite_dirs.items()}
        self.run_seconds["per_suite"] = time.monotonic() - started
        if self.measured is not None and self.measured["unmeasured"]:
            self.log(f"[starter] starter suites without a mutation result: "
                     f"{', '.join(self.measured['unmeasured'])}")
        if self.measured is None:
            # A request that measured no suite, such as one in which no suite
            # passes the tool's green check, keeps every suite and removes no
            # test. Each suite's entry states the cause.
            place = "starter.per_suite_mutation"
            for i in self.order:
                self.note_removals(i, [_removed_entry(
                    place, _Failed("mutation"), self.removal_rounds.get((i, place), 0) + 1,
                    unnamed="the mutation tool measured no suite")])

    def measure_coverage(self, freed: Dict[int, Tuple[str, str]]):
        # Per-suite measurement for the rarefaction distribution. A transient None
        # on a green suite is retried once. A suite the coverage tool still cannot
        # measure has no complete numbers and frees its slot. Without per-test
        # evidence a suite is scored on its own coverage run instead, as the
        # phase end scores its candidates, so it stays a candidate. It is left
        # out of the distribution and still joins the merged suite below.
        from testuary.consolidation import _Failed, _failed_run, red_verdict

        self.per_suite_data: Dict[int, Any] = {}
        self.starter_runs: Dict[int, Any] = {}
        # Each freed suite's failing measurement, for the removal.
        self.evidence_failures: Dict[int, Any] = {}
        for j, i in enumerate(self.order):
            data = None
            failure: Dict[str, Any] = {}
            for _attempt in range(2):
                # The removal reads the names of the last attempt only.
                failure.pop("stopping_tests", None)
                with project.scratch_dir(prefix="testuary_starter_") as sd:
                    spath = self.stage_alone(sd, i)
                    data = self.backend.collect_eviction_data(
                        [str(spath)], self.source_dir, measure_coverage_of=self.target_file_path,
                        timeout=self.evict_timeout, kill_matrix_path=None, log=self.log,
                        failure_fn=lambda kind, detail, stopping_tests=(): failure.update(
                            kind=kind, detail=detail,
                            **({"stopping_tests": stopping_tests} if stopping_tests else {})))
                if data is not None or not failure:
                    break
            if data is None and not failure:
                # The backend gives no per-test evidence.
                with project.scratch_dir(prefix="testuary_starter_") as sd:
                    run = self.backend.run_test_files(
                        [str(self.stage_alone(sd, i))], self.source_dir,
                        timeout=self.timeout, measure_coverage_of=self.target_file_path)
                if getattr(run, "ok", False):
                    self.starter_runs[j] = run
                else:
                    failure.update(detail=red_verdict(run), run=run)
            self.measured_suites(j + 1)
            if data is None:
                if failure:
                    freed[i] = ("coverage_unmeasured",
                                "the coverage tool could not measure the suite alone: "
                                + failure["detail"])
                    self.evidence_failures[i] = (
                        _failed_run("coverage", failure["run"]) if "run" in failure
                        else _Failed("evidence", tuple(failure.get("stopping_tests", ()))))
                continue
            self.per_suite_data[j] = data
            sdir = self.suites_root / slot_name(i)
            (sdir / "cleaned.json").write_text(json.dumps({
                "schema_version": 2, "source_sha256": source_sha256(self.green[i]),
                "normalization_rule_version": self.backend.normalize_shadowed_definitions(
                    self.green[i]).rule_version,
                "tokens": {t: sorted(v) for t, v in data.per_test_tokens.items()},
                "times_ms": data.per_test_time_ms, "suite_ms": data.suite_time_ms,
                "target_total_statements": data.target_total_lines,
                "target_covered_statements": data.target_covered_lines,
                "covered_statement_lines": _or_none(sorted, data.suite_covered_lines),
                "target_total_branches": data.target_total_branches,
                "target_covered_branches": data.target_covered_branches,
                "covered_branch_arcs": _or_none(sorted, data.suite_covered_branches),
            }), encoding="utf-8")

    def measure(self, runner):
        # How each accepted starter suite's mutation result is obtained
        # (suite_mutation): measured alone, derived from one pooled run
        # of all suites, or both, the measured results governing. A suite
        # measured alone, in one per-suite request, has verdicts equal to a
        # standalone run of it. It is staged as accepted, in its own
        # directory. The union of the per-suite results, or the pooled run
        # without them, is the starter-pool matrix. The cover does not read it
        # (the merged file is measured on its own below).
        self.method = self.cfg["suite_mutation"]
        # A suite without complete numbers (no per-test coverage, or no
        # result of the mutation tool) frees its slot: it is generated again
        # within its attempts or stays empty, and the pool is measured again,
        # so every result comes from the suites that remain.
        self.removal_rounds: Dict[Tuple[int, str], int] = {}
        while self.green:
            self.order = sorted(self.green)
            self.method_used = self.method
            self.derivation_reason: Optional[str] = None
            self.measured = None
            self.pooled_km = None
            self.staged_names: Dict[str, str] = {}
            self.pool_names: Dict[str, str] = {}
            self.standalone_names: Dict[str, str] = {}
            self.pool = None
            self.run_seconds: Dict[str, Optional[float]] = {"pooled": None, "per_suite": None}
            self.pooled_limit: Optional[Dict[str, Any]] = None
            if runner is not None and self.method in ("derive", "both"):
                self.derive(runner)
            if runner is not None and self.method_used in ("measure", "both"):
                self.measure_alone(runner)
            freed = self.mutation_freed()
            # A suite whose measurement names tests loses them and stays in
            # the pool, measured again in the next pass.
            changed = self.remove_failing(freed, "starter.per_suite_mutation",
                                          self.mutation_failure)
            if freed or changed:
                ended = self.free_and_refill(freed) if freed else None
                if ended is not None:
                    return ended
                continue

            # The count runs with the per-suite coverage below: the per-suite
            # mutation above is one request, which shows its own progress.
            self.measured_suites(0)

            self.measure_coverage(freed)
            changed = self.remove_failing(freed, "starter.per_suite_evidence",
                                          self.evidence_failures.__getitem__)
            if not freed and not changed:
                break
            ended = self.free_and_refill(freed) if freed else None
            if ended is not None:
                return ended
        if not self.green:
            return self.no_green_result()
        self.res.n_green = len(self.green)

    def mutation_freed(self):
        freed: Dict[int, Tuple[str, str]] = {}
        for i in self.order:
            if self.measured is not None and slot_name(i) in self.measured["unmeasured"]:
                outcome = (self.measured.get("outcomes") or {}).get(slot_name(i), "not green")
                stopped = (self.measured.get("stopped") or {}).get(slot_name(i))
                freed[i] = ("mutation_unmeasured",
                            f"the mutation tool's origin check stopped the suite: {stopped}"
                            if stopped else
                            f"the mutation tool measured nothing of the suite ({outcome})")
        return freed

    def mutation_failure(self, i: int) -> Any:
        """Slot ``i``'s failing per-suite mutation run, as the removal reads it."""
        from testuary.consolidation import _Failed

        name = slot_name(i)
        return _Failed("mutation", self.mutation_named.get(name, ()))

    def remove_failing(self, freed: Dict[int, Tuple[str, str]], place: str,
                       failure_of: Callable[[int], Any]) -> bool:
        """Remove from each suite in ``freed`` the tests that its failing
        measurement at ``place`` names, one round per pass, within
        the cleanup's rounds counted per suite and place. A suite whose
        normal run passes after the removal stays in the pool and leaves
        ``freed``. The others stay freed with the place's existing
        classification and reason, the entry stating the cause. Returns whether a suite changed."""
        from testuary.consolidation import _Failed, _remove_named_failures

        changed = False
        for i in sorted(freed):
            first = failure_of(i)

            def measure(child: str, kind: str = first.measurement) -> Any:
                cause = self.removal_cause(child)
                return _Failed(kind, cause=cause) if cause else None

            used = self.removal_rounds.get((i, place), 0)
            source, last, entries = _remove_named_failures(
                self.backend, self.green[i], place=place, failed=first,
                measure=measure, rounds=self.failing_cleanup_rounds - used, log=self.log)
            for entry in entries:
                entry["round"] += used
            self.removal_rounds[(i, place)] = used + sum("removed" in entry for entry in entries)
            self.note_removals(i, entries)
            if last is None:
                self.slot_changed(i, place, source, entries)
                del freed[i]
                changed = True
            else:
                self.removal_rounds = {key: count for key, count in self.removal_rounds.items()
                                       if key[0] != i}
        if changed:
            self.write_manifest(self.res.generation_seconds, not self.unresolved)
        return changed

    def note_removals(self, i: int, entries: List[Dict[str, Any]]) -> None:
        """``entries`` after the earlier ones in slot ``i``'s cleanup record."""
        meta_path = self.suites_root / slot_name(i) / "original.json"
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if isinstance(meta, dict):
            meta.setdefault("cleanup", {}).setdefault("measurement_removals", []).extend(entries)
            meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    def slot_changed(self, i: int, place: str, child: str,
                     entries: List[Dict[str, Any]]) -> None:
        """Slot ``i``'s suite after a removal at ``place``: it stays in the pool
        as ``child``, its cleaned file and the slot's record name it, and its
        validity derivation follows the slot's earlier ones."""
        sdir = self.suites_root / slot_name(i)
        self.derive_removal(sdir, place, self.green[i], child, entries)
        self.green[i] = child
        _suite_source_path(sdir, "cleaned", self.backend).write_bytes(child.encode("utf-8"))
        self.slot_evidence.measurement_removed(
            i, child, suffix=self.backend.test_file_extensions()[0])

    def scored_by_run(self, i: int, j: int):
        from testuary.consolidation import coverage_evidence_of_run, scored_from_measurement

        label = slot_name(i)
        elapsed = getattr(self.starter_runs[j], "elapsed_s", None)
        self.suite_scoreds.append(scored_from_measurement(
            label=label, source=self.green[i], kill_matrix=self.own_kms.get(label),
            coverage=coverage_evidence_of_run(self.starter_runs[j], self.target_file_path),
            execution_name=self.staged_names.get(
                label, f"test_{label}{self.backend.test_file_extensions()[0]}"),
            green=True, status="ranking_estimate",
            runtime_ms=float(elapsed) * 1000.0 if elapsed is not None else None,
            test_count=self.backend.test_metrics(self.green[i]).get("test_count"),
            status_reason=("scored on its own coverage run: the language "
                           "gives no per-test evidence")))

    def scored_by_evidence(self, i: int, data):
        from testuary.consolidation import ScoredSuite, coverage_percents
        from testuary.languages import mutation

        _line, _branch = coverage_percents(data.target_covered_lines, data.target_total_lines,
                                           data.target_covered_branches, data.target_total_branches)
        label = slot_name(i)
        suite_km = self.own_kms.get(label)
        self.suite_scoreds.append(ScoredSuite(
            label=label, source=self.green[i], line=_line, branch=_branch,
            mutation=_or_none(mutation.matrix_score_percent, suite_km),
            # A report that timed no test leaves the suite time unknown,
            # which ranks as unmeasured rather than as free.
            runtime_ms=float(data.suite_time_ms) if data.suite_time_ms is not None else None,
            green=True, status="ranking_estimate", status_reason=(
                ("measured alone under the starter-suite name, and materialized "
                 "under the delivered filename before selection when competitive")
                if self.measured is not None else
                ("derived from the pooled run of the starter suites, and measured "
                 "under the delivered filename before selection when competitive")
                if self.derived_governs else
                "no mutation score: the per-suite mutation measurement produced no result"
                if self.method_used == "unavailable" else
                "no mutation score: the starter phase end measures no mutation"),
            kill_matrix=suite_km,
            coverage_evidence={
                "line": _line, "branch": _branch, "target_total_statements": data.target_total_lines,
                "target_total_branches": data.target_total_branches,
                "covered_statement_lines": _or_none(sorted, data.suite_covered_lines),
                "covered_branch_arcs": _or_none(sorted, data.suite_covered_branches),
                "per_test_tokens": {test_id: sorted(map(str, tokens))
                                    for test_id, tokens in data.per_test_tokens.items()}},
            per_test_tokens={str(test_id): frozenset(map(str, tokens))
                             for test_id, tokens in data.per_test_tokens.items()},
            test_count=len(data.per_test_tokens),
            covered_lines=_or_none(frozenset, data.suite_covered_lines),
            covered_branches=_or_none(frozenset, data.suite_covered_branches),
            execution_name=self.staged_names.get(
                label,
                f"test_{slot_name(i)}{self.backend.test_file_extensions()[0]}")))
        self.suite_test_counts[label] = len(data.per_test_tokens)

    def collision_kills(self):
        collision_kills: Dict[int, Dict[str, set]] = {}
        if self.measured is None and self.pooled_km is not None and self.derived_governs:
            collision_kills = self.pool_kills
        elif self.km is not None:
            for j, i in enumerate(self.order):
                name = slot_name(i)
                if j not in self.per_suite_data or name not in self.staged_names:
                    continue
                own_killers = {"mutants": [
                    {"id": row.get("id"), "killers": list(verdict.get("killers") or ()),
                     "killing_tests": list(verdict.get("killing_tests") or ())}
                    for row in self.km.get("mutants") or []
                    for verdict in [(row.get("suite_verdicts") or {}).get(name) or {}]]}
                collision_kills.update(self.backend.attribute_pool_test_kills(
                    own_killers, {self.staged_names[name]: j},
                    {j: set(self.per_suite_data[j].per_test_tokens)}))
        return collision_kills

    def scored_suites(self):
        from testuary.consolidation import ScoredSuite

        collision_kills = self.collision_kills()
        self.suite_evidence = [
            ({test_id: set(tokens) | collision_kills.get(j, {}).get(test_id, set())
              for test_id, tokens in self.per_suite_data[j].per_test_tokens.items()}
             if j in self.per_suite_data else None)
            for j in range(len(self.order))]
        self.suite_scoreds: List[ScoredSuite] = []
        self.suite_test_counts: Dict[str, int] = {}
        for j, i in enumerate(self.order):
            data = self.per_suite_data.get(j)
            # measure_coverage freed every slot left without data or a run.
            if data is None:
                self.scored_by_run(i, j)
            else:
                self.scored_by_evidence(i, data)

    def derivation_record(self, runner):
        self.starter_pool_derivation = None
        if runner is not None and self.method in ("derive", "both"):
            self.starter_pool_derivation = starter_selection.record(
                requested=self.method, used=self.method_used, reason=self.derivation_reason,
                attribution={"pooled": "all" if self.pooled_km is not None else None,
                             "measured": self.attribution_mode if self.measured is not None else None},
                slots_by_label={slot_name(i): i for i in self.order},
                requested_slots=self.res.n_sets_requested, pool_names=self.pool_names,
                pooled_km=self.pooled_km, derived=self.derived_kms,
                measured=(self.measured["suites"] if self.measured is not None and self.method == "both"
                          else None),
                kept=self.kept, pool=self.pool, seconds=self.run_seconds,
                pooled_limit=self.pooled_limit,
                measured_limits=(self.measured or {}).get("budgets"),
                measured_outcomes=(self.measured or {}).get("outcomes"), measurement_unavailable=(
                    "the per-suite mutation measurement produced no result"
                    if self.method_used == "unavailable" else None))
        if self.res.suite_line_stats or self.res.suite_branch_stats:
            agent_helpers.log_metric_distribution(
                self.log, "starter", len(self.per_suite_data),
                self.res.suite_line_stats, self.res.suite_branch_stats,
                self.res.suite_mutation_stats)

    def own_results(self):
        from testuary import measurement
        from testuary.languages import mutation

        # A measured starter suite's score is its own measurement, exact by
        # definition, and governs whenever it was made. The record keeps each
        # slot's counts and verdicts. Under derive the derived scores stand,
        # recorded as derived.
        self.own_kms = (self.measured["suites"] if self.measured is not None
                   else self.derived_kms if self.derived_governs else {})
        self.starter_pool_phase_end_km = copy.deepcopy(self.km)
        self.starter_pool_attribution = None
        if self.measured is not None:
            self.res.suite_mutation_stats = _suite_stats(
                list(measurement.per_suite_scores(self.measured).values()))
            self.starter_pool_attribution = measurement.per_suite_attribution_record(
                self.measured, slots_by_name={slot_name(i): i for i in self.order},
                requested_slots=self.res.n_sets_requested, mode=self.attribution_mode)
        elif self.derived_governs and self.derived_kms:
            self.res.suite_mutation_stats = _suite_stats([
                score for score in (mutation.matrix_score_percent(m) for m in self.derived_kms.values() if m)
                if score is not None])

    def derive_suites(self):
        # Each suite's result derived from the pooled run: the pooled matrix
        # restricted to the suite's tests that ran in its own coverage run, by
        # their attributed kills and covered lines.
        self.attribution_mode = self.mut_cfg["cover_attribution"]
        self.derived_kms: Dict[str, Optional[dict]] = {}
        self.kept: Dict[str, Tuple[set, set]] = {}
        self.pool_kills: Dict[int, Dict[str, set]] = {}
        if self.pooled_km is not None and self.pool is not None:
            for j, i in enumerate(self.order):
                label = slot_name(i)
                if j not in self.per_suite_data:
                    self.derived_kms[label] = None
                    continue
                # The suite's tests that ran in its own coverage run, with
                # their coverage.
                ran = starter_selection.suite_tokens(self.per_suite_data[j].per_test_tokens)
                derived_km = starter_selection.derive_suite_matrix(self.pool, self.pooled_km, label, ran)
                self.derived_kms[label] = derived_km
                self.kept[label] = (self.pool.killed_by(self.pooled_km, label) if ran else set(),
                               starter_selection.covered_lines(ran))
                if derived_km is not None:
                    # Its killers carry the suite's standalone names, so
                    # its kills attribute to its own tests.
                    self.pool_kills[j] = self.backend.attribute_pool_test_kills(
                        derived_km, {self.standalone_names[label]: j}, {j: set(ran)}).get(j, {})

    def resolve_method(self, runner):
        # The method the reported starter scores came from: both without a
        # derivation is measure, a governing per-suite measurement that
        # produced nothing leaves none, named unavailable, and a phase end
        # that measures no mutation has none.
        if runner is None:
            self.method_used = ""
        if self.method == "both" and self.pooled_km is None and self.measured is not None:
            self.method_used = "measure"
        if self.measured is None and self.method_used in ("measure", "both"):
            self.method_used = "unavailable"
        self.res.suite_mutation_method = self.method_used
        # Derived results govern only under derive. Under measure and both a
        # per-suite measurement that produced nothing leaves the starters
        # without mutation scores: the derivation is only recorded.
        self.derived_governs = self.method_used == "derive"
        if self.measured is None and self.pool_names and self.derived_governs:
            self.staged_names = dict(self.pool_names)
        self.km = (self.measured["union"] if self.measured is not None
              else self.pooled_km if self.derived_governs else None)

        # Per-suite line / branch the merge improves on (each suite's own coverage,
        # target_covered over target_total).
        from testuary.consolidation import coverage_percents

        _percents = [coverage_percents(data.target_covered_lines, data.target_total_lines,
                                       data.target_covered_branches, data.target_total_branches)
                     for data in self.per_suite_data.values()]
        self.res.suite_line_stats = _suite_stats([line for line, _ in _percents])
        self.res.suite_branch_stats = _suite_stats([branch for _, branch in _percents])

    def materialize(self, runner, pending_suites: list, materialized_suites: set):
        from testuary.consolidation import (
            PhaseEndContext, _note_removals, measure_candidate_standalone)

        for provisional in pending_suites:
            # A removal here changes no slot: the changed suite is archived as
            # this candidate's.
            entries: list = []
            measured = measure_candidate_standalone(
                self.backend, provisional.source, target_file_path=self.target_file_path,
                source_dir=self.source_dir, timeout=self.timeout, mutation_runner=runner,
                stage_as_delivered=True, expected_measurement_identity=self.phase_measurement_identity,
                log=self.phase_end_log, place="starter.candidate", removals=entries)
            _note_removals(
                PhaseEndContext(self.backend, self.target_file_path, self.source_dir, self.timeout),
                self.phase, "starter.candidate", entries, provisional.source, measured.source,
                archive_name=f"candidate_{provisional.label}_removed{self.baseline_path.suffix}")
            measured.label = provisional.label
            materialized_suites.add(provisional.label)
            index = self.all_candidates.index(provisional)
            self.all_candidates[index] = measured
            if measured.test_count is not None:
                self.test_counts[measured.label] = measured.test_count
            self.log(f"[starter] materialized candidate {measured.label!r} under the delivered filename: "
                     f"status={measured.status}, mutation={measured.mutation}, "
                     f"branch={measured.branch}, line={measured.line}")

    def select(self, runner):
        from testuary.consolidation import (
            _check_then_measure, benchmark_tied_candidates, phase_end_axes, phase_end_candidates,
            pick_phase_result, red_verdict, selectable, top_tied)

        # The starter suites' own measurements support saturation and
        # identify which suites can still affect the phase result. A suite that reaches the
        # best-score set is measured once under the delivered filename before
        # the common selector may benchmark or select it. Most boundaries ship
        # reduced, so they pay no independent suite mutation run.
        consolidation_scoreds = phase_end_candidates(self.phase)
        self.test_counts = {
            **self.suite_test_counts, "reduced": self.phase.selected_tests,
            "merged": self.phase.pool_tests or self.phase.selected_tests}
        # Timing tied candidates is part of the consolidation.
        tie_break_runtime = benchmark_tied_candidates(
            self.backend, source_dir=self.source_dir,
            target_file_path=self.target_file_path, timeout=self.timeout, stage_as_delivered=True,
            log=self.phase_end_log)
        self.required_axes = phase_end_axes(runner)
        self.all_candidates = [*self.suite_scoreds, *consolidation_scoreds]
        self.baseline_path = self.backend.target_test_path(
            self.test_output_dir, self.target_file_path)
        self.baseline_path.parent.mkdir(parents=True, exist_ok=True)
        self.winner = None
        self.run = None
        materialized_suites: set[str] = set()
        while True:
            # The selectable candidates tied on the best score: a suite measured
            # in the pooled attribution context is materialized under the
            # delivered filename before it can enter a runtime tie-break or
            # become the delivered suite.
            top = top_tied([candidate for candidate in self.all_candidates
                            if selectable(candidate, self.required_axes)], self.required_axes)
            pending_suites = ([] if any(candidate.label == "reduced" for candidate in top) else [
                candidate for candidate in top
                if candidate.label.startswith(SLOT_PREFIX) and candidate.label not in materialized_suites])
            if pending_suites:
                self.materialize(runner, pending_suites, materialized_suites)
                continue
            self.winner = pick_phase_result(
                self.all_candidates, runtime_of=tie_break_runtime, required_axes=self.required_axes,
                log=self.phase_end_log)
            if self.winner is None or not self.winner.source:
                break
            self.baseline_path.write_text(self.winner.source, encoding="utf-8")
            # The normal run decides whether it passes under the delivered
            # filename. The coverage tool then measures it there.
            self.run, passed = _check_then_measure(
                self.backend, [self.baseline_path], self.source_dir,
                timeout=self.timeout, measure_coverage_of=self.target_file_path,
                log=self.phase_end_log)
            if getattr(self.run, "ok", False):
                break
            if passed:
                self.winner.status = "incomplete"
                self.winner.status_reason = (
                    "the coverage tool could not measure it under the delivered "
                    "filename: " + red_verdict(self.run))
            else:
                self.winner.green = False
                self.winner.status = "failing"
                self.winner.status_reason = (
                    "it fails under the delivered filename: " + red_verdict(self.run))
            self.log(f"[starter] candidate {self.winner.label!r} was excluded: "
                     f"{self.winner.status_reason}")

    def delivered_coverage(self) -> dict:
        """The delivered single suite's coverage: its selection evidence, or,
        where its acceptance run under the delivered filename measured other
        line or branch coverage, that run's, with both sides recorded in the
        selection audit. An acceptance run that measured no coverage there
        stops the phase."""
        from testuary.consolidation import coverage_disagreement, coverage_sides

        evidence, run = dict(self.winner.coverage_evidence or {}), self.run
        # Known limits: only the percentages are compared, so an acceptance
        # run that covers other lines at equal percentages delivers the
        # selection evidence's lines unrecorded. The choice among the starter
        # suites stands on their selection evidence, so a winner whose
        # acceptance run measures lower is delivered with that run's values
        # without a new choice.
        if not coverage_disagreement(evidence.get("line"), evidence.get("branch"),
                                     run.line_coverage, run.branch_coverage):
            return evidence
        if run.covered_lines is None:
            # A run that measured no coverage has none to deliver.
            raise RuntimeError(
                "the selected starter suite's final acceptance coverage "
                "disagrees with its source-bound selector evidence")
        self.phase.selection_audit["delivered_coverage_difference"] = coverage_sides(
            "selection", (evidence.get("line"), evidence.get("branch"),
                          evidence.get("covered_statement_lines"),
                          evidence.get("covered_branch_arcs")),
            "acceptance", (run.line_coverage, run.branch_coverage,
                           run.covered_lines, run.covered_branches))
        self.log("[starter] the delivered suite's acceptance run measured other coverage "
                 "than its selection, and its values are delivered")
        return {"line": run.line_coverage, "branch": run.branch_coverage,
                "target_total_statements": run.total_statements,
                "covered_statement_lines": (sorted(run.covered_lines)
                                            if run.covered_lines is not None else None),
                "target_total_branches": run.total_branches,
                "covered_branch_arcs": (sorted(run.covered_branches)
                                        if run.covered_branches is not None else None)}

    def consolidate(self, runner):
        from testuary._progress import observed_step
        from testuary.consolidation import (
            PhaseEndContext, PhaseEndOptions, consolidate_phase_end)

        with observed_step("starter", "consolidation", dict(STEPS)["consolidation"]):
            self.phase = consolidate_phase_end(
                PhaseEndContext(
                    backend=self.backend, target_file_path=self.target_file_path,
                    source_dir=self.source_dir, timeout=self.timeout,
                    stage_as_delivered=True, log=self.phase_end_log),
                [self.green[i] for i in self.order], runner,
                PhaseEndOptions(
                    mutation_method=config.consolidation_mutation_method(),
                    reduction=config.solver_settings(self.cfg),
                    candidate_evidence=self.suite_evidence,
                    measurement_identity=self.phase_measurement_identity,
                    solver_failure_dir=self.matrices_dir))
        self.phase.starter_pool_km = self.starter_pool_phase_end_km
        self.phase.starter_pool_attribution = self.starter_pool_attribution
        self.phase.starter_pool_derivation = self.starter_pool_derivation
        if self.starter_pool_derivation is not None:
            self.phase.starter_pool_derived_kms = {
                label: matrix for label, matrix in self.derived_kms.items() if matrix is not None}
            # The pooled run's own matrix, whatever happened to the
            # per-suite run. Under derive it is the starter matrix already.
            if not self.derived_governs:
                self.phase.starter_pool_pooled_km = self.pooled_km
        self.res.pool_tests = self.phase.pool_tests or sum(
            len(d.per_test_tokens) for d in self.per_suite_data.values())
        # The starter's collision check of the green suites' names. Its result
        # is not used here, so a resolution the language did not complete is
        # not judged and the phase continues. The Java demonstration's
        # recorded scenario answers this call.
        from testuary.consolidation import ValidationUnresolved

        try:
            self.backend.resolve_test_collisions([self.green[i] for i in self.order])
        except ValidationUnresolved as exc:
            self.log(f"[starter] the collision check gave no answer, not judged: {exc}")

        for candidate in self.suite_scoreds:
            candidate.target_revision = self.phase.target_revision
            # The run a starter's mutation result comes from, none when no
            # run measured it.
            candidate.measurement_identity = (
                {} if self.own_kms.get(candidate.label) is None else
                {"measurement_scope": "starter_suite_alone",
                 "target_source_sha256": self.phase.target_revision,
                 "execution_name": candidate.execution_name}
                if self.measured is not None else
                {"measurement_scope": "starter_suite_derived",
                 "target_source_sha256": self.phase.target_revision,
                 "pooled_execution_name": candidate.execution_name})

    def deliver(self):
        from testuary.consolidation import (
            PhaseEndDelivery, aggregate_coverage_matrix,
            measurement_names, persist_phase_end, record_selection_audit)

        record_selection_audit(
            self.phase, self.all_candidates, self.winner, required_axes=self.required_axes)
        if self.winner is None or not self.winner.source:
            self.res.status = "failed_phase_selection"
            self.res.detail = (
                "the starter phase produced no passing candidate with complete "
                f"{measurement_names(self.required_axes)} measurements")
            if self.matrices_dir is not None:
                persist_phase_end(
                    matrices_dir=self.matrices_dir, target_file=self.target_file_path,
                    phase=self.phase,
                    delivery=PhaseEndDelivery(winner_label="unresolved", delivered_km=None),
                    detail=self.res.detail, generation_scope=self.generation_scope)
            try:
                self.baseline_path.unlink()
            except OSError:
                pass
            self.res.duration_s = self.phase_seconds()
            return self.res
        self.res.baseline_from_single_suite = self.winner.label.startswith(SLOT_PREFIX)
        self.res.winner = "single_suite" if self.res.baseline_from_single_suite else self.winner.label
        self.res.selected_tests = self.test_counts.get(self.winner.label, self.phase.selected_tests)

        if self.matrices_dir is not None:
            single_suite_winner = self.winner.label.startswith(SLOT_PREFIX)
            selected_coverage = None
            if single_suite_winner:
                evidence = self.delivered_coverage()
                selected_coverage = aggregate_coverage_matrix(
                    target_file=self.target_file_path, source_sha256=None,
                    structural_parser_identity=self.phase.structural_parser_identity,
                    target_validator_identity=self.phase.target_validator_identity,
                    line=evidence.get("line"), branch=evidence.get("branch"),
                    target_total_statements=evidence.get("target_total_statements"),
                    covered_statement_lines=evidence.get("covered_statement_lines"),
                    target_total_branches=evidence.get("target_total_branches"),
                    covered_branch_arcs=evidence.get("covered_branch_arcs"),
                    # The final acceptance run is aggregate-only. Suite-name
                    # per-test identifiers must not be relabeled as evidence
                    # from the delivered filename.
                    per_test_tokens=None, per_test_runtime_ms=None, effective_cost_us=None)
            persist_phase_end(
                matrices_dir=self.matrices_dir, target_file=self.target_file_path,
                phase=self.phase,
                delivery=PhaseEndDelivery(
                    winner_label="single_suite" if single_suite_winner else self.winner.label,
                    winner_source_label=self.winner.label if single_suite_winner else None,
                    delivered_km=self.winner.kill_matrix,
                    delivered_tests=self.res.selected_tests,
                    delivered_source=self.winner.source,
                    delivered_coverage=selected_coverage, delivered_candidate=self.winner,
                    execution_name=self.baseline_path.name),
                detail="", generation_scope=self.generation_scope, generation_inputs={
                    slot_name(i): self.slot_evidence.construction_for(i, self.green[i], self.generation_scope)
                    for i in self.order})
        assert self.run is not None and getattr(self.run, "ok", False)
        self.res.baseline_file = str(self.baseline_path)
        self.res.baseline_line_coverage = self.run.line_coverage
        self.res.baseline_branch_coverage = self.run.branch_coverage
        self.res.baseline_suite_seconds = self.run.elapsed_s
        self.res.baseline_missing_lines = tuple(self.run.missing_lines)
        self.res.baseline_missing_branches = tuple(self.run.missing_branches)
        self.res.duration_s = self.phase_seconds()
        agent_helpers.log_phase_metrics(
            self.log, "starter", line=self.run.line_coverage, branch=self.run.branch_coverage,
            mutation=self.winner.mutation, tests=self.res.selected_tests,
            note=(f"baseline (pool={self.res.pool_tests}, "
                  f"green_suites={self.res.n_green}) |"))
        return self.res

    def measurement_context(self):
        from testuary import measurement
        from testuary.consolidation import _EVICTION_TIMEOUT_MULTIPLIER
        from testuary.agent import mutation_agent

        self.order = sorted(self.green)
        self.evict_timeout = int(self.timeout * _EVICTION_TIMEOUT_MULTIPLIER)
        self.mut_cfg = _starter_mutation_config()
        delivered_name = self.backend.target_test_path(
            self.test_output_dir, self.target_file_path).name
        self.phase_measurement_identity = mutation_agent.mutation_measurement_identity(
            target_file_path=self.target_file_path, language=self.language,
            execution_name=delivered_name, config=dict(self.mut_cfg),
            attribution=measurement.union_attribution(self.mut_cfg))
        return measurement.phase_end_runner(
            "starter", language=self.language, target_file=self.target_file_path,
            source_dir=self.source_dir, config=dict(self.mut_cfg), log_fn=self.log)


def run_starter_phase(
    *,
    language: str,
    target_file_path: str,
    source_dir: str,
    test_output_dir: str,
    suites_dir: str,
    matrices_dir: Optional[str] = None,
    log: Callable[[str], None] = print,
    prepared_models=None,
    run_idx: int = 0,
    reuse_suites: Optional[bool] = None,
) -> StarterResult:
    """Run the starter phase for one target.

    Writes the selected baseline to the target file's canonical coverage test
    path and persists the per-suite artifacts under ``suites_dir``. A pool
    that ends with ZERO green tests cancels the target
    (``status='cancelled_zero_passing'``). The orchestrator hard-fails on
    it rather than running the downstream phases against nothing."""
    sp = _StarterPhase(
        language=language, target_file_path=target_file_path, source_dir=source_dir,
        test_output_dir=test_output_dir, suites_dir=suites_dir, matrices_dir=matrices_dir,
        log=log, prepared_models=prepared_models, run_idx=run_idx,
        reuse_suites=reuse_suites)
    try:
        sp.prepare()
        sp.prepare_generator()
        sp.open_records()
        ended = sp.generate()
        if ended is not None:
            return ended

        if not sp.green:
            return sp.no_green_result()

        # The uniform post-phase consolidation: each green suite is measured on its
        # own (the pre-merge rarefaction distribution and the keep-best candidates),
        # the pool is merged into one suite, that merged file is measured
        # (coverage + the full mutation catalogue) and set-covered to a runtime-
        # minimal reduced suite. The phase-end selector then compares the individual
        # suites, merged suite, and reduced suite by the same rule as later phases.
        runner_cm = sp.measurement_context()
        from testuary.languages.mutation import cancel_on_interrupt, tool_scope

        # A Ctrl+C during the starter's measurements (green checks, coverage,
        # the mutation runs) stops them and removes their scratch.
        _enter("measure")

        with cancel_on_interrupt(), tool_scope(), runner_cm as runner:
            ended = sp.measure(runner)
            if ended is not None:
                return ended
            sp.resolve_method(runner)
            sp.derive_suites()
            sp.own_results()
            sp.derivation_record(runner)

            # Merge every green suite, measure the merged file, set-cover and derive.
            # Per-suite per-test tokens ride along, aligned with the merge's source
            # order: they inform only the victim choice when several owners of
            # one name are all un-renamable.
            # Kill evidence per test, keyed by the union rows' ids so it compares
            # across suites: each suite's killers come from its own verdicts, or
            # from the pooled run's killers, which name the suite's file.
            sp.scored_suites()
            sp.consolidate(runner)
            sp.select(runner)
            return sp.deliver()
    finally:
        # The object's account and generator refer back to it. Its references go
        # when the phase returns, so its matrices, candidates and model clients
        # are freed at once, not by the cyclic garbage collector later.
        vars(sp).clear()
