"""Each phase's chat log: messages rendered for the JSON log, written as
they arrive, kept for each failed attempt and exported at the end."""
from __future__ import annotations

import contextlib
import json
import secrets
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage
from testuary import model_requests
from testuary.phase_state import timing_fields
from testuary.models import field_names


# Chat log export (LangChain message types).


def _extract_message_for_log(
    msg: BaseMessage, *, reasoning_field_names: Optional[Sequence[str]] = None,
) -> Dict[str, Any]:
    """Render one LangChain message as a JSON-loggable dict with the
    ``type`` and ``content`` keys the chat-log readers use and the message's
    ``id`` (LangChain assigns UUIDs, which line entries up in a followed log).

    An AI message also gets its tool calls, its invalid tool calls (error
    and arguments cut to 200 characters), the ``finish_reason``, model name,
    token usage and system fingerprint of its response metadata (a
    truncation shows as ``finish_reason='length'``), and the captured
    reasoning fields (``reasoning_field_names``), which show that the
    reasoning round trip happens. A tool message gets its tool name and
    call id.
    """
    base_id = getattr(msg, "id", None)

    if isinstance(msg, SystemMessage):
        out = {"type": "system_prompt", "content": msg.content}
        if base_id:
            out["id"] = base_id
        return out
    if isinstance(msg, HumanMessage):
        out = {"type": "user_prompt", "content": msg.content}
        if base_id:
            out["id"] = base_id
        return out
    if isinstance(msg, AIMessage):
        out = {"type": "response", "content": msg.content}
        if base_id:
            out["id"] = base_id
        if msg.tool_calls:
            out["tool_calls"] = [
                {
                    "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None),
                    "id": tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None),
                    "args": tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None),
                }
                for tc in msg.tool_calls
            ]
        # Malformed calls go out on the wire and are answered at dispatch, so
        # the log records them. Without this field an assistant message with
        # neither content nor parsed calls is unreadable.
        invalid = getattr(msg, "invalid_tool_calls", None) or []
        if invalid:
            out["invalid_tool_calls"] = [
                {
                    "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", None),
                    "id": tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None),
                    "error": (
                        str(tc.get("error"))[:200]
                        if isinstance(tc, dict) and tc.get("error") else None
                    ),
                    "args_head": (
                        str(tc.get("args"))[:200]
                        if isinstance(tc, dict) else None
                    ),
                }
                for tc in invalid
            ]
        meta = getattr(msg, "response_metadata", None) or {}
        if meta:
            # Only these keys: the full metadata can carry the whole raw response.
            slim = {
                k: meta[k] for k in (
                    "finish_reason",
                    "model_name",
                    "token_usage",
                    "system_fingerprint",
                ) if k in meta
            }
            if slim:
                out["response_metadata"] = slim
        addl = getattr(msg, "additional_kwargs", None) or {}
        if addl:
            kept = field_names(reasoning_field_names)
            slim_addl = {k: addl[k] for k in addl if k in kept}
            if slim_addl:
                out["additional_kwargs"] = slim_addl
        return out
    if isinstance(msg, ToolMessage):
        out = {
            "type": "tool_return",
            "tool_name": getattr(msg, "name", None) or getattr(msg, "tool_name", None),
            "tool_call_id": getattr(msg, "tool_call_id", None),
            "content": msg.content,
        }
        if base_id:
            out["id"] = base_id
        return out
    return {"type": type(msg).__name__, "content": str(msg)}


def make_chat_log_flusher(
    *,
    output_dir: str,
    filename: str,
    project_name: Optional[str] = None,
    extra_metadata_factory: Optional[Callable[[str], Dict[str, Any]]] = None,
    reasoning_field_names: Optional[Sequence[str]] = None,
) -> Callable[[List[BaseMessage], Optional[str], str], None]:
    """Build the progressive chat-log flusher an agent wires into
    ``on_intermediate_state``, so every agent writes its chat log the same
    way: atomically, with the ``stage`` in the metadata, and with a failed
    write printed as a warning instead of raised.

    The callable ``flush(messages, error=None, stage="in_progress")`` writes
    a fresh JSON snapshot. A first call with no messages writes the starting
    state, so the file exists from the first turn.
    ``extra_metadata_factory(stage)`` adds per-call metadata (for example the
    phase or target). An exception it raises is ignored.
    """

    def flush(
        messages: List[BaseMessage],
        error: Optional[str] = None,
        stage: str = "in_progress",
    ) -> None:
        extra: Dict[str, Any] = {"stage": stage, "framework": "langgraph"}
        if error is not None:
            extra["error"] = error
        if extra_metadata_factory is not None:
            try:
                extra.update(extra_metadata_factory(stage))
            except Exception:
                pass
        export_chat_log(
            messages or [],
            output_dir=output_dir,
            filename=filename,
            project_name=project_name,
            extra_metadata=extra,
            reasoning_field_names=reasoning_field_names,
        )

    return flush


def make_retry_chat_snapshotter(
    *,
    output_dir: str,
    filename_prefix: str,
    project_name: Optional[str] = None,
    extra_metadata_factory: Optional[Callable[[str], Dict[str, Any]]] = None,
    reasoning_field_names: Optional[Sequence[str]] = None,
) -> Callable[[int, Dict[str, Any], BaseException], None]:
    """Build the callback that keeps each failed graph attempt in full.

    The active chat log keeps this state too, since recovery does not
    shorten the history. Each ``<prefix>_pre_retry_NNN.json`` is a separate,
    atomically written and unchanging record of one failed attempt. Files are
    numbered in write order, so a request sent again and an output-validation
    retry of the same attempt never share a file. The attempt number and the
    error are in the metadata.
    """
    written = [0]

    def snapshot(
        failed_attempt: int,
        state: Dict[str, Any],
        exc: BaseException,
    ) -> None:
        stage = "pre_retry"
        written[0] += 1
        extra: Dict[str, Any] = {
            "stage": stage,
            "framework": "langgraph",
            "failed_attempt": int(failed_attempt),
            "next_attempt": int(failed_attempt) + 1,
            "error_type": type(exc).__name__,
            "error": model_requests.failure_text(exc),
        }
        if extra_metadata_factory is not None:
            try:
                extra.update(extra_metadata_factory(stage))
            except Exception:
                pass
        export_chat_log(
            list(state.get("messages") or []),
            output_dir=output_dir,
            filename=f"{filename_prefix}_pre_retry_{written[0]:03d}.json",
            project_name=project_name,
            extra_metadata=extra,
            reasoning_field_names=reasoning_field_names,
        )

    return snapshot


def export_chat_log(
    messages: List[BaseMessage],
    output_dir: str,
    filename: str,
    *,
    project_name: Optional[str] = None,
    extra_metadata: Optional[Dict[str, Any]] = None,
    reasoning_field_names: Optional[Sequence[str]] = None,
) -> Optional[str]:
    """Write a LangChain message list as JSON (metadata and rendered
    messages) to ``output_dir/filename``, atomically through a temporary
    file. A relative ``output_dir`` is under the working directory. Returns
    the path, or None after printing a warning when the write fails.
    """
    try:
        out_dir = Path(output_dir)
        if not out_dir.is_absolute():
            out_dir = Path.cwd() / out_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / filename

        rendered = [_extract_message_for_log(m, reasoning_field_names=reasoning_field_names)
                    for m in (messages or [])]
        metadata: Dict[str, Any] = {
            "message_count": len(messages) if messages else 0,
            **timing_fields(),
        }
        if project_name:
            metadata["project"] = project_name
        if extra_metadata:
            metadata.update(extra_metadata)
        payload = {"metadata": metadata, "messages": rendered}
        tmp_path = out_path.with_name(f"{out_path.name}.{secrets.token_hex(4)}.tmp")
        # A model may send a lone surrogate, which UTF-8 cannot hold: it is
        # written as the JSON escape json.loads reads back.
        f = open(tmp_path, "x", encoding="utf-8", errors="backslashreplace")
        try:
            with f:
                json.dump(payload, f, indent=2, ensure_ascii=False)
            tmp_path.replace(out_path)
        finally:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
        return str(out_path)
    except Exception as exc:
        from testuary._progress import diagnostic_print

        diagnostic_print(f"WARNING: the chat log {filename} could not be written: {exc}")
        return None
