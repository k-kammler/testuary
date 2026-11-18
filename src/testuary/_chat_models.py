"""The chat model clients the agents talk to: an OpenAI-compatible endpoint
that sends the configured sampling exactly and keeps the reasoning
sidecars."""
from __future__ import annotations

from typing import Any, Dict, Optional, Sequence
import httpx
from langchain_core.messages import AIMessage
from langchain_openai import ChatOpenAI
from pydantic import field_validator
from testuary.models import DEFAULT_FIELDS, field_names


# Chat model factory.


def build_chat_model(
    *,
    model: str,
    api_url: str,
    api_key: str,
    timeout_seconds: float,
    max_tokens: Optional[int] = None,
    extra_kwargs: Optional[Dict[str, Any]] = None,
    preserve_reasoning_content: bool = False,
    reasoning_field_names: Optional[Sequence[str]] = None,
    use_responses_api: bool = False,
    include_legacy_output_cap: bool = True,
) -> ChatOpenAI:
    """Build a ``ChatOpenAI`` for an OpenAI-compatible endpoint.

    The httpx clients carry the per-request timeout, so long reasoning
    generations do not hit LangChain's default transport timeout.
    ``max_retries=0`` leaves retries to the agents' recovery loop.
    ``extra_kwargs`` passes further options LangChain accepts (for example
    ``extra_body``).

    ``preserve_reasoning_content`` (off here, on in the agents' bundled
    settings) keeps the non-standard reasoning sidecar fields, such as the
    ``reasoning_content`` that vLLM's ``--reasoning-parser`` emits, which
    LangChain drops on the way in and never sends again. Without them,
    interleaved-thinking models lose their earlier reasoning between turns
    and multi-turn tool calling degrades (MiniMax's documentation cites an
    accuracy drop of up to 40%). The alternative is to disable the server's
    reasoning parser so ``<think>...</think>`` blocks stay in ``content``.
    The returned subclass captures the fields into ``additional_kwargs``
    and sends them again on the way out, the community workaround of
    langchain-ai/langchain#35901, which upstream declined because
    ChatOpenAI follows the OpenAI specification strictly. Its overrides of
    the private ``_create_chat_result`` and ``_get_request_payload`` can
    break silently with a langchain-openai minor version.

    ``use_responses_api`` sends requests to the Responses API instead of
    Chat Completions. It round-trips reasoning blocks natively. It is off
    by default because vLLM's Responses support is per model and
    experimental (a response of unknown shape can break the agent loop
    silently), while Chat Completions with ``preserve_reasoning_content``
    covers the supported models on a stable path. With reasoning
    preservation, native Responses items stay in ``content``
    (``output_version="responses/v1"``) and the subclass checks their
    replay as well as the chat sidecars: an item the SDK cannot replay (for
    example unencrypted reasoning with ``store=False``) fails the request
    before it is sent instead of being lost.

    ``reasoning_field_names`` names the sidecar fields, by default
    ``reasoning_content`` (vLLM's parser, DeepSeek), ``reasoning`` (Ollama
    Cloud's /v1, verified against a live response), ``reasoning_details``
    and ``thinking`` (relays). Only fields present in a response are
    captured and only fields set on an outbound ``AIMessage`` are sent, so
    extra names are harmless. LiteLLM passes ``reasoning_content`` through
    unchanged and needs no proxy-side configuration.

    Every request carries the ``temperature`` and ``top_p`` given in
    ``extra_kwargs`` exactly and omits the ones not given, whatever the
    model name (see ``_ChatOpenAIExactSampling``).
    """
    http_client = httpx.Client(timeout=httpx.Timeout(timeout_seconds))
    http_async_client = httpx.AsyncClient(timeout=httpx.Timeout(timeout_seconds))
    kwargs: Dict[str, Any] = dict(
        model=model,
        base_url=api_url,
        # An empty key becomes ``"EMPTY"``: vLLM and LiteLLM proxies accept
        # ``Bearer EMPTY``, while an empty bearer value draws a 401.
        api_key=api_key or "EMPTY",
        http_client=http_client,
        http_async_client=http_async_client,
        timeout=timeout_seconds,
        use_responses_api=use_responses_api,
        max_retries=0,
        # langchain-openai 1.2+ injects TCP keepalive socket options by
        # default and warns that HTTP_PROXY auto-detection is disabled. The
        # clients passed here carry the timeout, so the empty tuple opts out
        # of the injection. It only silences the warning.
        http_socket_options=(),
    )
    if max_tokens is not None:
        kwargs["max_tokens"] = int(max_tokens)
    if use_responses_api and preserve_reasoning_content:
        kwargs["output_version"] = "responses/v1"
    if extra_kwargs:
        kwargs.update(extra_kwargs)
    if max_tokens is not None and include_legacy_output_cap:
        # langchain-openai sends the cap as ``max_completion_tokens``, which
        # servers predating that field (DeepSeek's API) ignore, falling back
        # to a small default output. The body also gets the classic
        # ``max_tokens`` they honor, unless the caller's extra_body sets it.
        body = dict(kwargs.get("extra_body") or {})
        body.setdefault("max_tokens", int(max_tokens))
        kwargs["extra_body"] = body
    configured = {key: kwargs[key] for key in _EXACT_SAMPLING if kwargs.get(key) is not None}
    if preserve_reasoning_content:
        return _ChatOpenAIWithReasoning(
            reasoning_field_names=field_names(reasoning_field_names),
            configured_sampling=configured,
            **kwargs,
        )
    return _ChatOpenAIExactSampling(configured_sampling=configured, **kwargs)


_EXACT_SAMPLING = ("temperature", "top_p")


class _ChatOpenAIExactSampling(ChatOpenAI):
    """Send the configured temperature and top_p exactly, and nothing unset.

    langchain-openai changes temperature by model name: it removes a value
    other than 1 for gpt-5 models other than gpt-5-chat and sends 1 for o1
    models when none was configured, so the request would differ from the
    run record. This class writes the configured values into every request
    payload and removes the ones not configured. A value passed with the
    call itself is kept, and a provider that rejects a value fails the
    request with its own error. The override uses a private langchain-openai
    method, so check it against the installed version."""

    configured_sampling: Dict[str, float] = {}

    def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[override]
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        for key in _EXACT_SAMPLING:
            if key in kwargs:
                continue
            if key in self.configured_sampling:
                payload[key] = self.configured_sampling[key]
            else:
                payload.pop(key, None)
        return payload


class _ChatOpenAIWithReasoning(_ChatOpenAIExactSampling):
    """Preserve the configured chat sidecars and check native Responses replay.

    Captures the configured response fields into
    ``AIMessage.additional_kwargs`` and sends them again when a later request
    is built from those messages. A request that cannot carry them raises
    ``ValueError``: outbound chat messages that do not match the
    conversation, or Responses input that lacks a reasoning item or would
    need a chat sidecar. The overrides use private langchain-openai methods,
    so check them against the installed version (upstream issue
    langchain-ai/langchain#35901)."""

    # A pydantic field, so the base class accepts it at construction
    # instead of rejecting an extra attribute.
    reasoning_field_names: tuple[str, ...] = DEFAULT_FIELDS

    @field_validator("reasoning_field_names", mode="before")
    @classmethod
    def _validate_reasoning_fields(cls, value):
        return field_names(value)

    def _create_chat_result(self, response, generation_info=None):  # type: ignore[override]
        result = super()._create_chat_result(response, generation_info)
        choices = (
            response.choices
            if hasattr(response, "choices")
            else response.get("choices", [])
        )
        for gen, choice in zip(result.generations, choices):
            raw_msg = (
                choice.message
                if hasattr(choice, "message")
                else choice.get("message", {})
            )
            for field in self.reasoning_field_names:
                value = (
                    getattr(raw_msg, field, None)
                    if not isinstance(raw_msg, dict)
                    else raw_msg.get(field)
                )
                if value is not None:
                    gen.message.additional_kwargs[field] = value
        return result

    def _get_request_payload(self, input_, *, stop=None, **kwargs):  # type: ignore[override]
        payload = super()._get_request_payload(input_, stop=stop, **kwargs)
        in_msgs = self._convert_input(input_).to_messages()
        pending = [
            (index, {field: message.additional_kwargs[field]
                     for field in self.reasoning_field_names
                     if message.additional_kwargs.get(field) is not None})
            for index, message in enumerate(in_msgs) if isinstance(message, AIMessage)
        ]
        pending = [(index, values) for index, values in pending if values]
        if self._use_responses_api(payload):
            # SDK streaming indices are local metadata, not Responses fields.
            outbound = list(payload["input"]) if isinstance(payload.get("input"), list) else []
            for message in in_msgs:
                if not isinstance(message, AIMessage) or not isinstance(message.content, list):
                    continue
                for block in message.content:
                    if not isinstance(block, dict) or block.get("type") != "reasoning":
                        continue
                    expected = {key: value for key, value in block.items() if key != "index"}
                    if isinstance(expected.get("summary"), list):
                        expected["summary"] = [
                            {key: value for key, value in item.items() if key != "index"}
                            if isinstance(item, dict) else item for item in expected["summary"]
                        ]
                    if expected not in outbound:
                        raise ValueError("cannot preserve native Responses reasoning in the outbound input")
                    outbound.remove(expected)
            if pending:
                raise ValueError("cannot attach configured reasoning sidecars to native Responses input")
            return payload
        if not pending:
            return payload
        payload_msgs = payload.get("messages")
        if not isinstance(payload_msgs, list) or len(payload_msgs) != len(in_msgs):
            raise ValueError("cannot attach configured reasoning: outbound messages do not match the conversation")
        for index, values in pending:
            out_msg = payload_msgs[index]
            if not isinstance(out_msg, dict) or out_msg.get("role") != "assistant":
                raise ValueError("cannot attach configured reasoning: outbound assistant message is unavailable")
            out_msg.update(values)
        return payload
