"""The demonstration's model: starter generations and agent responses
answered from a scenario, each request identified by its transport hashes."""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, message_to_dict

from testuary.agent import _langgraph as lg
from testuary import evidence as validity
from testuary.demo import backend as demo_model


class _Oracle:
    def __init__(self, answers):
        self.answers = list(answers)
        self.calls = []
        self.segments = []

    def answer(self, call):
        self.calls.append(call)
        return self.answers.pop(0)

    def enter_segment(self, name):
        self.segments.append(name)


def test_a_suite_request_is_identified_by_its_transport_hashes():
    messages = ({"role": "system", "content": "s"},
                {"role": "user", "content": "u"})
    oracle = _Oracle([{"text": "```python\nx = 1\n```",
                       "usage": {"input_tokens": 3, "output_tokens": 4}}])
    source = demo_model.ScenarioModelSource(oracle)
    text, usage = source.generate_suite(messages, temperature=0.7, seed=1001,
                                        timeout=60)
    assert text.startswith("```python") and usage["output_tokens"] == 4
    record = validity.prompt_transport_record(
        phase="starter", request_idx=1, messages=messages,
        prompt_template_version="v", provider_transport="t")
    assert oracle.calls[0].key == {
        "seed": 1001, "temperature": 0.7,
        "messages": record["message_sha256"]}


def test_agent_responses_come_in_order_and_enter_the_phase():
    first = AIMessage(content="", id="r1", tool_calls=[
        {"name": "read_tests", "args": {}, "id": "call_1"}])
    second = AIMessage(content="done", id="r2")
    oracle = _Oracle([{"message": message_to_dict(first)},
                      {"message": message_to_dict(second)}])
    model = demo_model.ScenarioModelSource(oracle).agent_chat_model("coverage")
    assert oracle.segments == ["coverage"]
    bound = model.bind_tools([object()])
    request = [SystemMessage(content="sys"), HumanMessage(content="go")]
    reply = bound.invoke(request)
    assert reply.id == "r1"
    assert reply.tool_calls[0]["id"] == "call_1"
    assert bound.invoke(request).content == "done"
    keys = [call.key for call in oracle.calls]
    assert [key["turn"] for key in keys] == [1, 2]
    assert keys[0]["messages"] == validity.prompt_transport_record(
        phase="", request_idx=0,
        messages=[lg._message_transport_payload(m) for m in request],
        prompt_template_version="", provider_transport="",
    )["message_sha256"]
