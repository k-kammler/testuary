"""The chat model ``models.catalogue_chat_model`` hands out to another program.

It wraps the chat model built for a catalogue alias. Each request goes
through one ``model_requests.RequestAccount``: a transient failure is sent
again under Testuary's retry policy, every request sent is recorded in the
caller's ``request_records.jsonl`` with the caller's label, and a context
overflow raises ``model_requests.ContextOverflow``. The built model is held
privately, so the handed-out model's fields, representation and
serialization hold no credential. The model's own key is removed from every
record and error text, and its errors carry no provider error, which can
hold the request and its key.
"""
from __future__ import annotations

import dataclasses
import threading
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, List, Optional

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from pydantic import PrivateAttr

from testuary import model_requests
from testuary.agent._langgraph import _usage_from_response, is_context_window_error
from testuary.evidence import append_jsonl
from testuary.settings import ConfigurationError, _without_secrets

#: One lock for the record files of every handed-out model in this process,
#: so that each line is written whole.
_RECORDS = threading.Lock()


class RecordedChatModel(BaseChatModel):
    """A built chat model whose requests are retried and recorded.

    Its requests are sent one at a time. A terminal failure or a reached
    ceiling raises to the caller, and the next request is sent again.
    """

    _inner: BaseChatModel = PrivateAttr()
    _identity: Mapping[str, str] = PrivateAttr()
    _account: model_requests.RequestAccount = PrivateAttr()
    _records: Path = PrivateAttr()
    _label: str = PrivateAttr()
    _secrets: tuple = PrivateAttr()
    _lock: threading.Lock = PrivateAttr(default_factory=threading.Lock)

    def __init__(self, inner: BaseChatModel, *, identity: Mapping[str, str],
                 retry: model_requests.RetrySettings, records: Path, label: str,
                 secrets: Sequence[str]) -> None:
        super().__init__()
        self._inner = inner
        self._identity = dict(identity)
        self._records = records
        self._label = label
        self._secrets = tuple(secrets)
        self._account = model_requests.RequestAccount(retry, on_record=self._write)

    @property
    def _llm_type(self) -> str:
        return "testuary-catalogue"

    def __repr__(self) -> str:
        return f"RecordedChatModel({self._identity!r})"

    def _write(self, record: model_requests.RequestRecord) -> None:
        # The key goes before as_dict() cuts the message, so no part of it stays.
        message = record.error_message
        record = dataclasses.replace(
            record, error_message=None if message is None else _without_secrets(message, self._secrets))
        with _RECORDS:
            append_jsonl(self._records, {**record.as_dict(), "label": self._label})

    def _redacted(self, kind: type, error: Exception) -> Exception:
        """``error`` as a ``kind``, its text without this model's key, and
        without the provider's error as its cause."""
        copy = kind.__new__(kind)
        Exception.__init__(copy, _without_secrets(str(error), self._secrets))
        for name in ("reason", "attempt"):
            if hasattr(error, name):
                value = getattr(error, name)
                setattr(copy, name, _without_secrets(value, self._secrets)
                        if isinstance(value, str) else value)
        return copy

    def _generate(self, messages: List[BaseMessage], stop: Optional[List[str]] = None,
                  run_manager: Any = None, **kwargs: Any) -> ChatResult:
        def send(attempt: int) -> Any:
            return self._inner.invoke(messages, stop=stop, **kwargs)

        failure = None
        with self._lock:
            try:
                outcome = self._account.execute(
                    send, usage_of=_usage_from_response, terminal=is_context_window_error)
            except model_requests.TerminalRequestError as exc:
                failure = self._redacted(
                    model_requests.ContextOverflow if is_context_window_error(exc.__cause__)
                    else model_requests.TerminalRequestError, exc)
            except model_requests.TransportExhausted as exc:
                failure = self._redacted(model_requests.TransportExhausted, exc)
            except Exception as exc:
                # Any other failure, such as a record that cannot be written,
                # leaves without the errors chained to it.
                exc.__cause__ = exc.__context__ = None
                failure = exc
            finally:
                # A terminal failure or a ceiling ends a phase's requests. A
                # handed-out model's next request is sent again.
                self._account.cancel.clear()
        if failure is not None:
            # Raised outside the handler, so no provider error is its context.
            raise failure
        return ChatResult(generations=[ChatGeneration(message=outcome.value)])

    def bind_tools(self, tools: Sequence[Any], **kwargs: Any) -> Any:
        """The built model's tool binding, sent through this model's retries
        and records. LangChain's chat models bind their tools as arguments. A
        client whose binding carries none is refused."""
        options = getattr(self._inner.bind_tools(tools, **kwargs), "kwargs", None)
        if not isinstance(options, Mapping):
            raise ConfigurationError(
                f"model alias [{self._identity.get('model_alias')}]: its client's bind_tools "
                "gives no bound arguments, so its tool calls cannot go through Testuary's "
                "retries and records")
        return self.bind(**options)
