"""The agent loop's context policy, in one place.

Pure functions over LangChain message lists with no module state, so
several agent loops can use them at once. ``make_context_hook`` builds the
graph's pre-model hook from the reductions: ``dedup_file_payloads`` stubs
superseded reads, ``strip_old_thinking`` keeps only the latest turn's
reasoning and ``drop_oldest_pair_safe`` keeps a window of recent messages
with whole tool-call pairs. ``render_read`` builds a read tool's return.
"""

from __future__ import annotations

import re
from typing import Any, Callable, Dict, List, Optional, Sequence

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)

from testuary.agent._helpers import ELIDED_SENTINEL, NO_PROGRESS_BANNER, number_line, text_lines
from testuary.models import field_names
from testuary.settings import _CHOICES

#: How the injected state marker starts, so each turn replaces it.
_MARKER_SENTINEL = "[state |"

#: How a cached re-read's answer starts (``read_source_cached``, alone or
#: after its loop-breaker prefix), also behind the no-progress banner: it
#: holds no content and supersedes nothing.
CACHED_READ_PREFIXES = ("[Cached read #", "[LOOP BREAKER #")


def dedup_stub(file_key) -> str:
    """Replacement text for a superseded read, with the sentinel that
    rejects a copy-forward into a write."""
    return (
        f"[Superseded earlier read/write of {file_key}. A later "
        f"read/write replaced it; the current version is later in this "
        f"conversation. {ELIDED_SENTINEL}]"
    )

def carries_tool_calls(m: BaseMessage) -> bool:
    """Whether the API serializer emits ``tool_calls`` for this message:
    for parsed calls, malformed ones (``invalid_tool_calls``) and the raw
    ``additional_kwargs`` sidecar alike. The provider rejects an unanswered
    call, so pairing rules treat all three as calls.
    """
    if not isinstance(m, AIMessage):
        return False
    if m.tool_calls or getattr(m, "invalid_tool_calls", None):
        return True
    return bool(m.additional_kwargs.get("tool_calls"))


#: Inline reasoning a server without a reasoning parser (vLLM) leaves in
#: the content.
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)

def drop_oldest_pair_safe(
    messages: Sequence[BaseMessage],
    keep_last: int,
) -> List[BaseMessage]:
    """The system messages first, then the task prompt and the most recent
    messages, ``keep_last`` non-system messages in all.

    The task prompt, the first ``HumanMessage``, is kept ahead of the window
    when the window would cut it, so the request states the task and opens
    with a user turn. The window starts with no tool result whose call was
    dropped and ends with no unanswered tool call: the provider rejects
    both.
    """
    systems = [m for m in messages if isinstance(m, SystemMessage)]
    convo = [m for m in messages if not isinstance(m, SystemMessage)]
    task: List[BaseMessage] = []

    # None keeps every message: an unlimited history window.
    if keep_last is not None and len(convo) > max(keep_last, 0):
        start = len(convo) - max(keep_last, 0)
        first_human = next(
            (i for i, m in enumerate(convo) if isinstance(m, HumanMessage)), None)
        if first_human is not None and first_human < start and keep_last > 0:
            task = [convo[first_human]]
            start += 1
        convo = convo[start:]

    while convo and isinstance(convo[0], ToolMessage):
        convo.pop(0)
    while convo and carries_tool_calls(convo[-1]):
        convo.pop()

    return systems + task + convo


def strip_old_thinking(
    messages: Sequence[BaseMessage],
    *,
    reasoning_field_names: Optional[Sequence[str]] = None,
) -> List[BaseMessage]:
    """A copy of ``messages`` without reasoning in every ``AIMessage`` but
    the latest, which keeps it: some reasoning APIs expect it beside that
    turn's tool calls. Older reasoning would re-accumulate through the
    sidecar. Clears sidecars, native reasoning items and ``<think>`` blocks.
    """
    fields = field_names(reasoning_field_names)

    last_ai = -1
    for i, m in enumerate(messages):
        if isinstance(m, AIMessage):
            last_ai = i
    if last_ai < 0:
        return list(messages)

    out: List[BaseMessage] = []
    for i, m in enumerate(messages):
        if not isinstance(m, AIMessage) or i == last_ai:
            out.append(m)
            continue
        changed = False
        kwargs = dict(m.additional_kwargs)
        for field in fields:
            if field in kwargs:
                kwargs.pop(field, None)
                changed = True
        content = m.content
        if isinstance(content, str) and _THINK_RE.search(content):
            content = _THINK_RE.sub("", content)
            changed = True
        elif isinstance(content, list):
            retained = [block for block in content
                        if not isinstance(block, dict) or block.get("type") != "reasoning"]
            if len(retained) != len(content):
                content = retained
                changed = True
        if changed:
            out.append(
                m.model_copy(
                    update={"additional_kwargs": kwargs, "content": content}
                )
            )
        else:
            out.append(m)
    return out


def render_read(
    content: str,
    *,
    name: str,
    max_chars: Optional[int],
    start_line: int = 1,
    end_line: Optional[int] = None,
) -> str:
    """A read tool's return: the lines numbered as the edit views number
    them (``_helpers.number_line``), so ``start_line`` is a coordinate the
    model sees, and the edit tool strips those numbers from a pasted block.
    ``start_line`` and ``end_line`` are 1-based, inclusive and clamped. A
    range over ``max_chars`` (None: unlimited) returns the head that fits
    and a marker naming the next ``start_line``.
    """
    from testuary.agent import _limits
    lines = text_lines(content)
    total = len(lines)
    header = f"# {name} ({total} lines)\n"

    start = start_line if (start_line and start_line > 0) else 1
    if start > total:
        return (
            f"{header}[start_line {start} is past the end of the file "
            f"({total} lines).]"
        )
    end = end_line if (end_line and end_line >= start) else total
    end = min(end, total)

    numbered = [number_line(i, ln) for i, ln in enumerate(lines[start - 1:end], start=start)]

    kept: List[str] = []
    size = 0
    for nl in numbered:
        if max_chars is not None and kept and size + len(nl) + 1 > max_chars:
            break
        kept.append(nl)
        size += len(nl) + 1

    if len(kept) < len(numbered):
        shown_end = start + len(kept) - 1
        body = "\n".join(kept)
        marker = _limits.marker(
            len(numbered) - len(kept), "lines", name,
            f"call this read tool with start_line={shown_end + 1} for the next part")
        return f"{header}{body}\n[lines {start}-{shown_end} of {total}]\n{marker}"

    body = "\n".join(numbered)
    return f"{header}{body}\n[lines {start}-{end} of {total}]"


#: render_read's last line, the lines a read's answer shows.
_SHOWN_LINES = re.compile(r"\n\[lines (\d+)-(\d+) of (\d+)\]")


def _shown_range(start: str, end: str, total: str):
    """The (start, end) of render_read's ``[lines start-end of total]``
    line, None when they are the whole file."""
    first, last = int(start), int(end)
    return None if first == 1 and last == int(total) else (first, last)


def _covers(outer, inner) -> bool:
    """Whether ``outer`` covers ``inner``. None, the whole file, covers
    all and is covered only by itself."""
    if outer is None:
        return True
    if inner is None:
        return False
    return outer[0] <= inner[0] and outer[1] >= inner[1]


def dedup_file_payloads(
    messages: Sequence[BaseMessage],
    *,
    read_tools,
    file_key,
    stub,
) -> List[BaseMessage]:
    """Stub each read that a later read of the same file covers. A read
    covers the lines its answer shows (render_read's last line), all of
    them a whole-file read, so distinct pages coexist and a read cut at its
    size limit covers only its shown part. An answer that shows no line
    stubs nothing and stays. Only reads are stubbed, in place, so the
    message count and the tool pairs stay. Each agent names its
    ``read_tools``, ``file_key(name, args)`` (which file a call reads) and
    ``stub(file_key)``.
    """
    msgs = list(messages)

    # call_id -> file_key of each read call
    call_info = {}
    for m in msgs:
        if not isinstance(m, AIMessage):
            continue
        for tc in m.tool_calls or []:
            name = tc.get("name")
            if name in read_tools:
                args = tc.get("args") or {}
                call_info[tc.get("id")] = file_key(name, args)

    # The read returns in order: (message index, file_key, shown range).
    events = []
    for i, m in enumerate(msgs):
        if isinstance(m, ToolMessage):
            fk = call_info.get(m.tool_call_id)
            answer = str(m.content)
            if answer.startswith(NO_PROGRESS_BANNER):
                answer = answer.split("]\n\n", 1)[-1]
            shown = _SHOWN_LINES.findall(answer)
            if fk is not None and shown and not answer.startswith(CACHED_READ_PREFIXES):
                events.append((i, fk, _shown_range(*shown[-1])))

    superseded = []  # (message index, file_key) to stub
    for a_idx, (idx_a, fk_a, rng_a) in enumerate(events):
        for _, fk_b, rng_b in events[a_idx + 1:]:
            if fk_b == fk_a and _covers(rng_b, rng_a):
                superseded.append((idx_a, fk_a))
                break

    for idx, fk in superseded:
        msgs[idx] = msgs[idx].model_copy(update={"content": stub(fk)})

    return msgs


def task_clause(target_file: Optional[str]) -> str:
    """The state marker's ``file=<name>``: rebuilt every turn, it names the
    target even after the window dropped the turns that did."""
    if target_file:
        name = target_file.replace("\\", "/").rsplit("/", 1)[-1]
    else:
        name = "?"
    return f"file={name}"


#: Characters per token for the second rebuild trigger: a local estimate,
#: so two runs of a file rebuild on the same turn, as no provider count
#: (tied to cache state) would.
_CHARS_PER_TOKEN = 4


def estimated_tokens(
    messages: Sequence[BaseMessage], *, reasoning_field_names: Optional[Sequence[str]] = None,
) -> int:
    """Approximate size of an assembled prompt, in tokens."""
    total = 0
    fields = field_names(reasoning_field_names)
    for m in messages:
        content = m.content
        total += len(content if isinstance(content, str) else str(content))
        for call in getattr(m, "tool_calls", None) or ():
            total += len(str(call.get("args") or ""))
        for field in fields:
            value = m.additional_kwargs.get(field)
            if value:
                total += len(str(value))
    return total // _CHARS_PER_TOKEN


#: When the context hook rebuilds the conversation.
CONTEXT_REBUILDS = _CHOICES["context_rebuild"]


def make_context_hook(
    build_marker: Callable[[Any], str],
    *,
    system_prompt: Optional[str] = None,
    keep_last: Optional[int] = None,
    dedup: Optional[dict] = None,
    strip_thinking: bool = True,
    rebuild: str = "mode_change",
    rebuild_trigger_tokens: int = 0,
    reasoning_field_names: Optional[Sequence[str]] = None,
) -> Callable[[Any], Dict[str, Any]]:
    """The ``pre_model_hook`` of the context policy. A rebuild drops the old
    state marker, stubs superseded reads (``dedup``: the keywords of
    ``dedup_file_payloads``), strips older reasoning (``strip_thinking``),
    keeps the ``keep_last`` window (None: all) and adds the system prompt
    when none is there.

    With ``rebuild`` "mode_change" it runs on the turn ``agent_mode``
    changes and the turns between send the held result plus what the graph
    appended, so each request extends the last and the provider's prefix
    cache serves all but the tail. The mode change costs least: the bound
    tool schema, ahead of the conversation, changes then anyway. With
    "every_turn" it runs on every turn. Above ``rebuild_trigger_tokens``
    (0: off) the next turn rebuilds regardless.

    The state marker, with the live metric, is appended last on every turn,
    after the held prefix. The returned messages serve the model call only:
    the graph keeps the full trajectory and the chat log.
    """
    if rebuild not in CONTEXT_REBUILDS:
        raise ValueError(
            f"rebuild must be one of {', '.join(CONTEXT_REBUILDS)}, "
            f"not {rebuild!r}")
    fields = field_names(reasoning_field_names)
    #: The held rebuild and how many of the graph's messages produced it,
    #: owned by this hook's agent loop.
    held: Dict[str, Any] = {
        "mode": None,
        "view": None,
        "consumed": -1,
        "source_prefix": None,
    }

    def rebuilt(messages: Sequence[BaseMessage]) -> List[BaseMessage]:
        msgs: List[BaseMessage] = [
            m
            for m in messages
            if not (
                isinstance(m, SystemMessage)
                and isinstance(m.content, str)
                and m.content.startswith(_MARKER_SENTINEL)
            )
        ]
        if dedup:
            msgs = dedup_file_payloads(msgs, **dedup)
        if strip_thinking:
            msgs = strip_old_thinking(msgs, reasoning_field_names=fields)
        if keep_last is not None:
            msgs = drop_oldest_pair_safe(msgs, keep_last)
        if system_prompt and not any(isinstance(m, SystemMessage) for m in msgs):
            msgs.insert(0, SystemMessage(content=system_prompt))
        return msgs

    def extended(messages: List[BaseMessage], mode) -> Optional[List[BaseMessage]]:
        """The held rebuild plus what the graph appended since, or None to
        rebuild."""
        if rebuild != "mode_change" or held["view"] is None:
            return None
        if mode != held["mode"] or not 0 <= held["consumed"] <= len(messages):
            return None
        consumed = held["consumed"]
        # The count alone does not prove the graph extended the held prefix.
        if list(messages[:consumed]) != held["source_prefix"]:
            return None
        msgs = held["view"] + list(messages[consumed:])
        # The provider rejects an unanswered tool call. After a complete
        # tool round there is none, else a rebuild drops it.
        if msgs and carries_tool_calls(msgs[-1]):
            return None
        if rebuild_trigger_tokens and (
                estimated_tokens(msgs, reasoning_field_names=fields) >= rebuild_trigger_tokens):
            return None
        return msgs

    def hook(state: Any) -> Dict[str, Any]:
        messages: List[BaseMessage] = list(state["messages"])
        mode = state.get("agent_mode") if hasattr(state, "get") else None

        msgs = extended(messages, mode)
        if msgs is None:
            msgs = rebuilt(messages)
            held.update(
                mode=mode,
                view=msgs,
                consumed=len(messages),
                source_prefix=list(messages),
            )

        marker_text = build_marker(state)
        if marker_text:
            msgs = msgs + [SystemMessage(content=marker_text)]

        return {"messages": msgs}

    return hook
