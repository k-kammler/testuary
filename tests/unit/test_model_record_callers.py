"""Outbound settings join observed model records at every supported SDK entry."""
import json

import pytest

import test_model_precedence_callers as caller_harness
import test_protocol_loopback_callers as http_harness
from testuary import model_requests as requests

loopback = http_harness.loopback


@pytest.mark.parametrize("phase,protocol", http_harness.ROUTES)
@pytest.mark.parametrize("reported", [None, "provider-returned-model"])
@pytest.mark.parametrize("retry", [False, True])
def test_actual_transport_and_response_records_join_without_guessing_the_returned_model(
    phase, protocol, reported, retry, tmp_path, monkeypatch, loopback,
):
    loopback.response_model = reported
    if retry:
        loopback.failures.append(503)
    expected = (1 if phase == "starter" else 2) + int(retry)
    usage = {}
    caller_harness.invoke_phase(
        phase, tmp_path, http_harness.configured(tmp_path, protocol, loopback), monkeypatch,
        expected_requests=expected, usage_snapshot=usage,
    )
    root = tmp_path / ("suites" if phase == "starter" else "logs")
    outbound_path = root / ("prompt_transport.jsonl" if phase == "starter" else f"{phase}_prompt_transport.jsonl")
    responses_path = root / ("request_records.jsonl" if phase == "starter" else f"{phase}_usage.jsonl")
    outbound = [json.loads(line) for line in outbound_path.read_text(encoding="utf-8").splitlines()]
    responses = [json.loads(line) for line in responses_path.read_text(encoding="utf-8").splitlines()]
    assert len(outbound) == len(responses) == len(loopback.requests) == expected
    assert [row["request_idx"] for row in outbound] == list(range(1, expected + 1))
    assert [row["request_idx"] for row in responses] == list(range(1, expected + 1))
    logical = 1 if phase == "starter" else 2
    assert usage == {"physical": expected, "logical": logical, "unknown": int(retry),
                     "input_tokens": logical * 10, "output_tokens": logical * 3}
    response_key = "response_model" if phase == "starter" else "model_name"
    expected_logical = [1] * (1 + int(retry)) + ([2] if phase != "starter" else [])
    for index, (sent, response, (_, body, _)) in enumerate(zip(outbound, responses, loopback.requests, strict=True)):
        assert sent["request_idx"] == response["request_idx"]
        assert sent["phase"] == phase
        settings = sent["sent_settings"]
        assert settings["phase"] == phase and settings["protocol"] == protocol
        assert settings["model"] == body["model"] == "chosen"
        assert settings["model_alias"] == "explicit" and settings["provider"] == "openai"
        assert settings["api_url"] == loopback.endpoint
        assert response["logical_request_idx"] == expected_logical[index]
        if phase == "starter":
            assert response["run"] == sent["run"]
            assert response["slot"] == sent["slot"] == 1
            assert response["logical_request_idx"] == sent["logical_request_idx"]
        failed = retry and index == 0
        assert response["outcome"] == (requests.TRANSIENT if failed else requests.DELIVERED)
        assert response["usage_known"] is not failed
        if failed or reported is None:
            assert response_key not in response
        else:
            assert response[response_key] == reported
            assert response[response_key] != settings["model"]
    assert all(http_harness.TOKEN.encode() not in path.read_bytes()
               for path in tmp_path.rglob("*") if path.is_file())
