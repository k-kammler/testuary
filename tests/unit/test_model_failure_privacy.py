"""Provider errors must not expose captured credentials in persisted diagnostics."""
import os
from dataclasses import replace
import json
import sys

import pytest

from agent_log import console_records
import test_model_precedence_callers as caller_harness
import test_protocol_loopback_callers as http_harness
from testuary import model_requests as requests
from testuary import config
from testuary.settings import RunSettings, _redact_diagnostic

loopback = http_harness.loopback


@pytest.mark.parametrize("phase,protocol", http_harness.ROUTES)
@pytest.mark.parametrize("status,exhausted", [(401, False), (503, False), (503, True)],
                         ids=["terminal", "transient-recovered", "transient-exhausted"])
def test_reflected_credential_is_removed_from_actual_provider_failure_outputs(
    phase, protocol, status, exhausted, tmp_path, monkeypatch, loopback, capsys,
):
    loopback.error_message = f"fixture denied top_p; supplied credential {http_harness.TOKEN}"
    loopback.failures.append(status)
    settings = http_harness.configured(tmp_path, protocol, loopback)
    # The starter phase keeps no trace log, so only the agents' sections set it.
    settings = replace(settings, sections={**settings.sections, **{
        name: {**settings.sections[name],
               **({"enable_trace_log": True} if name != "starter" else {}),
               **({"max_consecutive_transient_failures": 1} if exhausted else {})}
        for name in ("starter", "coverage", "mutation")
    }})
    expected = 1 if status == 401 or exhausted else (2 if phase == "starter" else 3)
    caller_harness.invoke_phase(phase, tmp_path, settings, monkeypatch, expected_requests=expected)
    assert len(loopback.requests) == expected
    output = capsys.readouterr()
    assert http_harness.TOKEN not in output.out + output.err
    files = [path for path in tmp_path.rglob("*") if path.is_file()]
    assert files
    assert all(http_harness.TOKEN.encode() not in path.read_bytes() for path in files)
    if protocol != "ollama-native":
        assert "fixture denied top_p" in output.out + "".join(path.read_text(encoding="utf-8") for path in files)


@pytest.mark.parametrize("phase,protocol", http_harness.ROUTES)
def test_embedded_endpoint_credentials_are_removed_from_actual_failure_outputs(
    phase, protocol, tmp_path, monkeypatch, loopback, capsys,
):
    password, query = "synthetic-endpoint-password", "synthetic-query-credential"
    endpoint = loopback.endpoint.replace("http://", f"http://fixture-user:{password}@") + f"?key={query}"
    loopback.failures.append(401)
    loopback.error_message = f"fixture denied top_p at {endpoint}; password {password}, query {query}"
    configured = http_harness.configured(tmp_path, protocol, loopback)
    configured = replace(configured, models={"explicit": {**configured.models["explicit"], "api_url": endpoint}})
    record = caller_harness.invoke_phase(phase, tmp_path, configured, monkeypatch, expected_requests=1)
    assert record["sent_settings"]["api_url"] == loopback.endpoint
    output = capsys.readouterr()
    for secret in (http_harness.TOKEN, "fixture-user", password, query):
        assert secret not in output.out + output.err
        assert all(secret.encode() not in path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())


@pytest.mark.parametrize("protocol", ("openai-chat", "openai-responses", "ollama-native"))
@pytest.mark.parametrize("exhausted", (False, True), ids=["recovered", "exhausted-after-retry"])
@pytest.mark.parametrize("embedded_endpoint", (False, True), ids=["key", "endpoint"])
def test_starter_retry_uses_safe_messages_in_production_console_and_saved_logger(
    protocol, exhausted, embedded_endpoint, tmp_path, monkeypatch, loopback, capsys,
):
    from testuary import orchestrator
    from testuary.agent._helpers import AgentLogger
    from testuary import starters

    password, query = "synthetic-retry-password", "synthetic-retry-query"
    settings = http_harness.configured(tmp_path, protocol, loopback)
    settings = replace(settings, sections={**settings.sections, "starter": {
        **settings.sections["starter"], "max_consecutive_transient_failures": 2,
    }})
    reflected = f"HTTP 503 unavailable; supplied credential {http_harness.TOKEN}"
    secrets = [http_harness.TOKEN]
    if embedded_endpoint:
        endpoint = loopback.endpoint.replace("http://", f"http://retry-user:{password}@") + f"?key={query}"
        settings = replace(settings, models={"explicit": {
            **settings.models["explicit"], "api_url": endpoint,
        }})
        reflected += f" at {endpoint}; password {password}; query {query}"
        secrets.extend(("retry-user", password, query))
    loopback.error_message = reflected
    loopback.failures.extend([503, 503] if exhausted else [503])
    accounts, retries, delivered, failures = [], [], [], []
    execute = requests.RequestAccount.execute

    def observe_execute(account, send, **kwargs):
        accounts.append(account)
        on_retry = kwargs["on_retry"]

        def observe_retry(attempt, cause, wait):
            original_text = str(cause)
            retries.append((attempt, cause, wait, original_text))
            on_retry(attempt, cause, wait)
            assert str(cause) == original_text

        kwargs["on_retry"] = observe_retry
        try:
            outcome = execute(account, send, **kwargs)
        except requests.TransportExhausted as failure:
            failures.append((failure, failure.__cause__, str(failure.__cause__)))
            raise
        delivered.append(outcome)
        return outcome

    monkeypatch.setattr(requests.RequestAccount, "execute", observe_execute)
    root = tmp_path / "subject"
    root.mkdir()
    target = root / "module.py"
    target.write_text("def subject():\n    return 1\n", encoding="utf-8")
    starter_dir = tmp_path / "suites"
    previous_logger = orchestrator.get_main_logger()
    with config.use_run_settings(settings), monkeypatch.context() as logging_patch:
        logger = AgentLogger("subject", agent_name="main", config={})
        logger.init()
        main_log = logger.enable_streaming(str(tmp_path / "logs"), "main.jsonl")
        logging_patch.setattr(orchestrator._log, "logger", logger)
        result = starters.run_starter_phase(
            language="python", target_file_path=str(target),
            source_dir=str(root), test_output_dir=str(tmp_path / "output"),
            suites_dir=str(starter_dir), log=orchestrator.log_print,
        )
    assert orchestrator.get_main_logger() is previous_logger
    assert len(loopback.requests) == result.requests_used == 2
    assert result.logical_requests == 1
    assert result.unknown_usage_requests == 1 + int(exhausted)
    assert (result.input_tokens, result.output_tokens) == ((0, 0) if exhausted else (10, 3))
    assert len(accounts) == len(retries) == 1
    attempt, cause, wait, original_text = retries[0]
    assert (attempt, wait) == (1, 0)
    assert requests.classify_failure(cause) == requests.TRANSIENT
    assert str(cause) == original_text
    if protocol != "ollama-native":
        assert http_harness.TOKEN in original_text
    if embedded_endpoint:
        assert password in original_text and query in original_text
    assert accounts[0].transient_failures == 1 + int(exhausted)
    if exhausted:
        assert result.status == "failed_model_request" and not delivered
        assert len(failures) == 1
        failure, original_cause, cause_text = failures[0]
        assert failure.__cause__ is original_cause and str(original_cause) == cause_text
        assert requests.classify_failure(original_cause) == requests.TRANSIENT
        assert not list(starter_dir.glob("suite_*/response.txt"))
    else:
        assert result.status == "cancelled_zero_passing" and not failures
        assert len(delivered) == 1 and delivered[0].value[0] == "no code fence"
        assert (starter_dir / "suite_001/response.txt").read_text() == "no code fence"
        assert not list(starter_dir.glob(f"suite_*/{starters._GENERATION_DELIVERY}"))
    rows = [json.loads(line) for line in (starter_dir / "request_records.jsonl").read_text().splitlines()]
    assert [row["outcome"] for row in rows] == [
        requests.TRANSIENT, requests.TRANSIENT if exhausted else requests.DELIVERED,
    ]
    assert {row["logical_request_idx"] for row in rows} == {1}
    output = capsys.readouterr()
    messages = [row["message"] for row in console_records(main_log)]
    retry_messages = [text for text in messages if "failed in transport" in text]
    assert len(retry_messages) == 1 and "Retrying in 0s" in retry_messages[0]
    assert type(cause).__name__ in retry_messages[0] and "503" in retry_messages[0]
    assert retry_messages[0] in output.out
    for secret in secrets:
        assert secret not in output.out + output.err
        assert all(secret.encode() not in path.read_bytes() for path in tmp_path.rglob("*") if path.is_file())


@pytest.mark.parametrize("source", ("catalogue", "named-environment", "provider-environment"))
def test_failure_formatting_keeps_the_original_exception_and_request_policy(source):
    key = "unique-captured-provider-secret"
    model, environment = {"provider": "openai", "model": "chosen"}, {}
    if source == "catalogue":
        model["api_key"] = key
    elif source == "named-environment":
        model["api_key_env"] = "CUSTOM_CREDENTIAL"
        environment["CUSTOM_CREDENTIAL"] = key
    else:
        environment["TESTUARY_OPENAI_API_KEY"] = key
    configured = RunSettings(models={"explicit": model}, environment=environment)

    class Failure(Exception):
        status_code = 401

    cause = Failure(f"HTTP 401 fixture denied top_p; supplied {key}")
    observed = []
    account = requests.RequestAccount(requests.RetrySettings(first_wait_s=0), on_record=observed.append)
    with config.use_run_settings(configured), pytest.raises(requests.TerminalRequestError) as error:
        account.execute(lambda attempt: (_ for _ in ()).throw(cause))
    assert error.value.__cause__ is cause and key in str(cause)
    assert requests.classify_failure(cause) == requests.TERMINAL
    assert "HTTP 401 fixture denied top_p" in str(error.value) and key not in str(error.value)
    assert account.physical_requests == account.logical_requests == account.unknown_usage_requests == 1
    assert account.transient_failures == 0 and account.cancel.is_set()
    assert len(observed) == 1 and observed[0] is account.records[0]
    assert key not in json.dumps(account.records[0].as_dict())


@pytest.mark.parametrize("key", ["quoted-'secret", 'json-"secret', "escaped-\\secret", "unicode-\u2603-secret"])
def test_error_formatting_redacts_escaped_credential_forms_without_modifying_inputs(key):
    configured = RunSettings(environment=dict(os.environ), models={"explicit": {"api_key": key}})
    text = json.dumps({"error": key}) + " " + repr(key)
    result = _redact_diagnostic(text, configured)
    assert key not in result and json.dumps(key)[1:-1] not in result and repr(key)[1:-1] not in result
    assert configured.models["explicit"]["api_key"] == key


def test_error_formatting_keeps_safe_details_when_no_settings_are_active():
    assert _redact_diagnostic("HTTP 503 retry after 2s", None) == "HTTP 503 retry after 2s"
    assert _redact_diagnostic("at https://user:password@example.test/v1?key=private", None) == "at https://example.test/v1"


@pytest.mark.parametrize("kind", ("terminal", "exhausted", "paid-journal"))
def test_actual_fatal_logger_omits_raw_cause_without_replacing_exception_or_process_hook(kind, monkeypatch):
    from testuary import __main__ as cli

    key = "unique-fatal-provider-credential"
    configured = RunSettings(environment=dict(os.environ), models={"explicit": {"api_key": key}})
    cause = RuntimeError(f"fixture denied top_p with {key}")
    paid = object()
    with config.use_run_settings(configured):
        if kind == "terminal":
            failure = requests.TerminalRequestError(cause)
        elif kind == "exhausted":
            failure = requests.TransportExhausted("one failure limit", cause, 1)
        else:
            failure = requests.RecordNotWritten(paid, cause)
    previous = sys.excepthook
    recorded = []

    def fail():
        raise failure

    monkeypatch.setattr(cli, "_main", fail)
    monkeypatch.setattr(cli, "log_print", lambda message, **kwargs: recorded.append((message, kwargs)))
    with pytest.raises(type(failure)) as propagated:
        cli.main()
    assert propagated.value is failure and failure.__cause__ is cause
    assert key in str(cause) and sys.excepthook is previous
    assert len(recorded) == 1 and recorded[0][1] == {"level": 0}
    assert key not in recorded[0][0] and "fixture denied top_p" in recorded[0][0]
    assert "Traceback" in recorded[0][0] and type(failure).__name__ in recorded[0][0]
    if kind == "paid-journal":
        assert failure.outcome is paid


def test_unrelated_failure_tracebacks_keep_their_existing_chain():
    import traceback

    try:
        try:
            raise ValueError("underlying non-provider detail")
        except ValueError as cause:
            raise RuntimeError("top-level non-provider detail") from cause
    except RuntimeError as failure:
        text = requests.failure_traceback(failure)
        printed = "".join(traceback.format_exception(type(failure), failure, failure.__traceback__))
    assert "ValueError: underlying non-provider detail" in text
    assert "RuntimeError: top-level non-provider detail" in text
    assert "direct cause" in text
    assert text == printed
