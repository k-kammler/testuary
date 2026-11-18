"""A client factory for Testuary in one file.

The alias's ``chat_model = testuary_example_chat.py:factory`` names this file
and its ``factory``. Testuary loads the file and calls ``factory`` with the
resolved constructor parameters and request options as keyword arguments,
including ``max_retries=0``: Testuary owns retries and their accounting. The
factory returns a LangChain chat model that streams and can bind tools. This
example wraps the OpenAI-compatible client. A different client maps the
parameters itself.
"""
from langchain_openai import ChatOpenAI


def factory(**parameters):
    return ChatOpenAI(stream_usage=True, **parameters)
