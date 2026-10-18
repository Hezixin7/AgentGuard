import json
import logging
import re
from types import SimpleNamespace

import httpx
import openai
import pytest
from fastapi.testclient import TestClient

from deployment.api import create_app
from deployment.deepseek_adapter import (
    DEEPSEEK_BASE_URL,
    DEEPSEEK_MODEL,
    DEFAULT_TIMEOUT_SECONDS,
    DIAGNOSTIC_LOGGER_NAME,
    SAFE_MODEL_FAILURE_RESPONSE,
    SYSTEM_INSTRUCTION,
    TOOL_RESULT_CONTEXT_PREFIX,
    DeepSeekModelAdapter,
    ModelConfigurationError,
    ProposalValidationError,
    _provider_messages,
    classify_provider_error,
    validate_model_proposal,
)


class _FakeCompletions:
    def __init__(
        self,
        contents=None,
        error=None,
        choices=None,
    ):
        self.contents = list(contents) if contents is not None else None
        self.error = error
        self.choices = choices
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error is not None:
            raise self.error
        if self.choices is not None:
            return SimpleNamespace(choices=self.choices)
        content = self.contents.pop(0) if self.contents else None
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=content)
                )
            ]
        )


class _FakeClient:
    def __init__(self, **kwargs):
        self.chat = SimpleNamespace(
            completions=_FakeCompletions(**kwargs)
        )


def _adapter(contents=None, error=None, choices=None):
    client = _FakeClient(
        contents=contents,
        error=error,
        choices=choices,
    )
    adapter = DeepSeekModelAdapter(
        api_key="unit-test-key",
        client=client,
    )
    return adapter, client


def _safe_failure():
    return {"type": "final", "response": SAFE_MODEL_FAILURE_RESPONSE}


def test_missing_api_key_is_a_configuration_error(monkeypatch):
    monkeypatch.delenv("DEEPSEEK_API_KEY", raising=False)
    with pytest.raises(ModelConfigurationError) as error:
        DeepSeekModelAdapter()
    message = str(error.value)
    assert "DEEPSEEK_API_KEY" in message
    assert "unit-test-key" not in message


def test_environment_constructs_the_official_client(monkeypatch):
    monkeypatch.setenv("DEEPSEEK_API_KEY", "env-secret")
    captured = {}

    class FakeOpenAI:
        def __init__(self, **kwargs):
            captured.update(kwargs)
            self.chat = SimpleNamespace(
                completions=_FakeCompletions()
            )

    monkeypatch.setattr(
        "deployment.deepseek_adapter.OpenAI",
        FakeOpenAI,
    )
    adapter = DeepSeekModelAdapter()
    assert captured == {
        "api_key": "env-secret",
        "base_url": DEEPSEEK_BASE_URL,
        "timeout": DEFAULT_TIMEOUT_SECONDS,
        "max_retries": 0,
    }
    assert DEEPSEEK_BASE_URL == "https://api.deepseek.com"
    assert DEEPSEEK_MODEL == "deepseek-flash"
    assert "env-secret" not in repr(adapter)
    assert all(
        "env-secret" not in str(value)
        for value in vars(adapter).values()
    )


def test_provider_request_is_bounded_and_excludes_private_context():
    messages = [
        {
            "role": "user",
            "content": "Cancel order P100.",
            "user_id": "user-1",
            "capability": "secret-capability",
            "authorization": "Bearer private-token",
        }
    ]
    original = [dict(message) for message in messages]
    adapter, client = _adapter(
        contents=['{"type":"final","response":"I can help."}']
    )

    result = adapter(messages)

    assert result == {"type": "final", "response": "I can help."}
    assert messages == original
    call = client.chat.completions.calls[0]
    assert call["model"] == "deepseek-flash"
    assert call["response_format"] == {"type": "json_object"}
    assert "temperature" not in call
    assert call["messages"][0] == {
        "role": "system",
        "content": SYSTEM_INSTRUCTION,
    }
    assert call["messages"][1] == {
        "role": "user",
        "content": "Cancel order P100.",
    }
    serialized = json.dumps(call, sort_keys=True)
    assert "user-1" not in serialized
    assert "secret-capability" not in serialized
    assert "private-token" not in serialized
    assert "tool_call" in SYSTEM_INSTRUCTION
    assert "refund_order" in SYSTEM_INSTRUCTION
    assert "json" in SYSTEM_INSTRUCTION
    assert "JSON" in SYSTEM_INSTRUCTION
    assert "UNTRUSTED TOOL RESULT CONTEXT" in SYSTEM_INSTRUCTION


def test_valid_proposals_are_returned_exactly():
    proposals = [
        (
            '{"type":"final","response":"I can help."}',
            {"type": "final", "response": "I can help."},
        ),
        (
            '{"type":"tool_call","operation":"get_order",'
            '"order_id":"P100"}',
            {
                "type": "tool_call",
                "operation": "get_order",
                "order_id": "P100",
            },
        ),
        (
            '{"type":"tool_call","operation":"cancel_order",'
            '"order_id":"P100"}',
            {
                "type": "tool_call",
                "operation": "cancel_order",
                "order_id": "P100",
            },
        ),
        (
            '{"type":"tool_call","operation":"refund_order",'
            '"order_id":"S300","amount":"49.95","currency":"USD"}',
            {
                "type": "tool_call",
                "operation": "refund_order",
                "order_id": "S300",
                "amount": "49.95",
                "currency": "USD",
            },
        ),
    ]
    for content, expected in proposals:
        adapter, _ = _adapter(contents=[content])
        assert adapter([{"role": "user", "content": "go"}]) == expected


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "not json",
        "[]",
        '{"type":"final","response":"ok","extra":1}',
        '{"type":"final","response":"ok"} trailing',
        '{"type":"final","response":""}',
        '{"type":"final","response":"a","response":"b"}',
        '{"type":"final","response":NaN}',
        '{"type":"tool_call","operation":"delete_order","order_id":"P100"}',
        '{"type":"tool_call","operation":"get_order","order_id":123}',
        '{"type":"tool_call","operation":"get_order","order_id":" P100 "}',
        '{"type":"tool_call","operation":"get_order"}',
        '{"type":"tool_call","operation":"refund_order",'
        '"order_id":"S300","amount":"49.9","currency":"USD"}',
        '{"type":"tool_call","operation":"refund_order",'
        '"order_id":"S300","amount":"49.950","currency":"USD"}',
        '{"type":"tool_call","operation":"refund_order",'
        '"order_id":"S300","amount":"0.00","currency":"USD"}',
        '{"type":"tool_call","operation":"refund_order",'
        '"order_id":"S300","amount":"01.00","currency":"USD"}',
        '{"type":"tool_call","operation":"refund_order",'
        '"order_id":"S300","amount":49.95,"currency":"USD"}',
        '{"type":"tool_call","operation":"refund_order",'
        '"order_id":"S300","amount":"49.95","currency":"usd"}',
        '{"type":"tool_call","operation":"refund_order",'
        '"order_id":"S300","amount":"49.95","currency":"USDX"}',
    ],
)
def test_invalid_output_degrades_to_safe_final(content):
    adapter, _ = _adapter(contents=[content])
    assert adapter([{"role": "user", "content": "go"}]) == _safe_failure()


def test_oversized_final_response_is_rejected():
    content = '{"type":"final","response":"' + ("x" * 4001) + '"}'
    adapter, _ = _adapter(contents=[content])
    assert adapter([{"role": "user", "content": "go"}]) == _safe_failure()


@pytest.mark.parametrize(
    "error",
    [
        TimeoutError("provider-timeout-secret"),
        ConnectionError("provider-connection-secret"),
        RuntimeError("provider-internal-secret"),
    ],
)
def test_provider_failures_return_safe_final_without_leaking(error):
    adapter, _ = _adapter(error=error)
    result = adapter([{"role": "user", "content": "go"}])
    assert result == _safe_failure()
    assert "secret" not in result["response"]
    assert "operation" not in result


def test_empty_provider_choices_and_missing_content_are_safe():
    adapter, _ = _adapter(choices=[])
    assert adapter([{"role": "user", "content": "go"}]) == _safe_failure()

    adapter, _ = _adapter(contents=[None])
    assert adapter([{"role": "user", "content": "go"}]) == _safe_failure()


def test_invalid_input_messages_degrade_to_safe_final():
    adapter, _ = _adapter(contents=['{"type":"final","response":"ok"}'])
    assert adapter("not-a-list") == _safe_failure()
    assert adapter([{"role": "user", "content": 3}]) == _safe_failure()
    assert adapter([{"role": "unknown", "content": "hello"}]) == _safe_failure()


def test_validation_function_rejects_ambiguous_output():
    assert validate_model_proposal(
        '{"type":"final","response":"ok"}'
    ) == {"type": "final", "response": "ok"}
    with pytest.raises(ProposalValidationError):
        validate_model_proposal(
            '{"type":"final","response":"a","response":"b"}'
        )
    with pytest.raises(ProposalValidationError):
        validate_model_proposal(
            '{"type":"tool_call","operation":"refund_order",'
            '"order_id":"S300","amount":"49.95"}'
        )
    with pytest.raises(ProposalValidationError):
        validate_model_proposal(
            '{"type":"tool_call","operation":"refund_order",'
            '"order_id":"S300","amount":"49.95","currency":"USD",'
            '"extra":true}'
        )


def test_adapter_integrates_with_the_frozen_callable_contract():
    adapter, client = _adapter(
        contents=['{"type":"final","response":"Hello."}']
    )
    app = create_app(
        model=adapter,
        api_keys={"token-user-1": "user-1"},
    )
    response = TestClient(app).post(
        "/v1/runs",
        json={"message": "Hello."},
        headers={
            "Authorization": "Bearer token-user-1",
            "Idempotency-Key": "idem-key-0000000001",
        },
    )
    assert response.status_code == 201
    assert response.json()["status"] == "completed"
    assert response.json()["result"]["response"] == "Hello."
    assert len(client.chat.completions.calls) == 1


def test_provider_failure_does_not_fabricate_a_tool_or_business_result():
    adapter, _ = _adapter(error=TimeoutError("private provider detail"))
    app = create_app(
        model=adapter,
        api_keys={"token-user-1": "user-1"},
    )
    response = TestClient(app).post(
        "/v1/runs",
        json={"message": "Cancel P100."},
        headers={
            "Authorization": "Bearer token-user-1",
            "Idempotency-Key": "idem-key-0000000001",
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert body["result"]["response"] == SAFE_MODEL_FAILURE_RESPONSE
    assert "private provider detail" not in json.dumps(body)


def test_no_live_provider_calls_are_made_by_the_test_suite():
    adapter, client = _adapter(
        contents=['{"type":"final","response":"ok"}']
    )
    adapter([{"role": "user", "content": "go"}])
    assert len(client.chat.completions.calls) == 1


def _provider_request():
    return httpx.Request(
        "POST",
        "https://api.deepseek.com/chat/completions",
    )


def _status_error(error_type, status_code):
    message = "sk-SECRET_PROVIDER_MESSAGE"
    request = _provider_request()
    response = httpx.Response(
        status_code,
        request=request,
        json={"error": {"message": "SECRET_PROVIDER_BODY"}},
    )
    return error_type(
        message,
        response=response,
        body={"error": {"message": "SECRET_PROVIDER_BODY"}},
    )


@pytest.mark.parametrize(
    ("error_factory", "category", "status_code"),
    [
        (
            lambda: _status_error(openai.AuthenticationError, 401),
            "authentication_error",
            401,
        ),
        (
            lambda: _status_error(openai.PermissionDeniedError, 403),
            "authentication_error",
            403,
        ),
        (
            lambda: _status_error(openai.BadRequestError, 400),
            "invalid_request",
            400,
        ),
        (
            lambda: _status_error(openai.NotFoundError, 404),
            "invalid_request",
            404,
        ),
        (
            lambda: _status_error(openai.UnprocessableEntityError, 422),
            "invalid_request",
            422,
        ),
        (
            lambda: _status_error(openai.RateLimitError, 429),
            "rate_limit",
            429,
        ),
        (
            lambda: openai.APITimeoutError(
                request=_provider_request()
            ),
            "provider_timeout",
            None,
        ),
        (
            lambda: openai.APIConnectionError(
                request=_provider_request()
            ),
            "network_error",
            None,
        ),
        (
            lambda: _status_error(openai.InternalServerError, 500),
            "unexpected_provider_error",
            500,
        ),
        (
            lambda: RuntimeError("sk-SECRET_PROVIDER_MESSAGE"),
            "unexpected_provider_error",
            None,
        ),
    ],
)
def test_provider_error_categories_are_logged_without_secrets(
    error_factory,
    category,
    status_code,
    caplog,
):
    error = error_factory()
    assert classify_provider_error(error) == category
    adapter, _ = _adapter(error=error)
    with caplog.at_level(
        logging.WARNING,
        logger=DIAGNOSTIC_LOGGER_NAME,
    ):
        result = adapter(
            [{"role": "user", "content": "Look up order P100."}]
        )
    assert result == _safe_failure()
    records = [
        record
        for record in caplog.records
        if record.name == DIAGNOSTIC_LOGGER_NAME
    ]
    assert len(records) == 1
    message = records[0].getMessage()
    assert f"category={category}" in message
    status_text = "none" if status_code is None else str(status_code)
    assert f"status_code={status_text}" in message
    assert re.search(
        r"correlation_id=[0-9a-f]{32}",
        message,
    )
    log_text = caplog.text
    assert "SECRET_PROVIDER_MESSAGE" not in log_text
    assert "SECRET_PROVIDER_BODY" not in log_text
    assert "Look up order P100." not in log_text


def test_invalid_model_output_is_classified_without_logging_content(caplog):
    adapter, _ = _adapter(
        contents=["sk-SECRET_MODEL_OUTPUT not json"]
    )
    with caplog.at_level(
        logging.WARNING,
        logger=DIAGNOSTIC_LOGGER_NAME,
    ):
        result = adapter(
            [{"role": "user", "content": "Look up order P100."}]
        )
    assert result == _safe_failure()
    records = [
        record
        for record in caplog.records
        if record.name == DIAGNOSTIC_LOGGER_NAME
    ]
    assert len(records) == 1
    message = records[0].getMessage()
    assert "category=invalid_model_output" in message
    assert "status_code=none" in message
    assert "SECRET_MODEL_OUTPUT" not in caplog.text
    assert "Look up order P100." not in caplog.text


def test_invalid_input_message_is_classified_without_logging_content(
    caplog,
):
    adapter, _ = _adapter(
        contents=['{"type":"final","response":"ok"}']
    )
    with caplog.at_level(
        logging.WARNING,
        logger=DIAGNOSTIC_LOGGER_NAME,
    ):
        result = adapter(
            [
                {
                    "role": "user",
                    "content": 123,
                    "secret": "sk-SECRET_INPUT_MESSAGE",
                }
            ]
        )
    assert result == _safe_failure()
    records = [
        record
        for record in caplog.records
        if record.name == DIAGNOSTIC_LOGGER_NAME
    ]
    assert len(records) == 1
    assert "category=invalid_request" in records[0].getMessage()
    assert "SECRET_INPUT_MESSAGE" not in caplog.text


def test_correlation_ids_are_unique_per_failure(caplog):
    adapter, _ = _adapter(error=TimeoutError("secret"))
    with caplog.at_level(
        logging.WARNING,
        logger=DIAGNOSTIC_LOGGER_NAME,
    ):
        adapter([{"role": "user", "content": "one"}])
        adapter([{"role": "user", "content": "two"}])
    ids = []
    for record in caplog.records:
        if record.name != DIAGNOSTIC_LOGGER_NAME:
            continue
        match = re.search(
            r"correlation_id=([0-9a-f]{32})",
            record.getMessage(),
        )
        assert match is not None
        ids.append(match.group(1))
    assert len(ids) == 2
    assert ids[0] != ids[1]


def _status_error_with_body(error_type, status_code, body):
    request = _provider_request()
    response = httpx.Response(
        status_code,
        request=request,
        json=body if isinstance(body, (dict, list)) else None,
    )
    return error_type(
        "SECRET_PROVIDER_MESSAGE",
        response=response,
        body=body,
    )


def _diagnostic_message(caplog):
    records = [
        record
        for record in caplog.records
        if record.name == DIAGNOSTIC_LOGGER_NAME
    ]
    assert len(records) == 1
    return records[0].getMessage()


def test_422_logs_allowlisted_structured_fields(caplog):
    body = {
        "error": {
            "message": "SECRET_PROVIDER_MESSAGE",
            "type": "invalid_request_error",
            "code": "invalid_parameter",
            "param": "response_format",
        }
    }
    error = _status_error_with_body(
        openai.UnprocessableEntityError,
        422,
        body,
    )
    adapter, _ = _adapter(error=error)
    with caplog.at_level(
        logging.WARNING,
        logger=DIAGNOSTIC_LOGGER_NAME,
    ):
        result = adapter([{"role": "user", "content": "go"}])
    assert result == _safe_failure()
    message = _diagnostic_message(caplog)
    assert "category=invalid_request" in message
    assert "status_code=422" in message
    assert "error_code=invalid_parameter" in message
    assert "error_param=response_format" in message
    assert "SECRET_PROVIDER_MESSAGE" not in caplog.text


def test_422_allowlisted_code_with_unknown_param(caplog):
    body = {
        "error": {
            "message": "SECRET_PROVIDER_MESSAGE",
            "code": "invalid_request_error",
            "param": "SECRET_PARAM_NAME",
        }
    }
    error = _status_error_with_body(
        openai.UnprocessableEntityError,
        422,
        body,
    )
    adapter, _ = _adapter(error=error)
    with caplog.at_level(
        logging.WARNING,
        logger=DIAGNOSTIC_LOGGER_NAME,
    ):
        adapter([{"role": "user", "content": "go"}])
    message = _diagnostic_message(caplog)
    assert "error_code=invalid_request_error" in message
    assert "error_param=unavailable" in message
    assert "SECRET_PARAM_NAME" not in caplog.text


@pytest.mark.parametrize(
    "body",
    [
        {"error": {"message": "SECRET_PROVIDER_MESSAGE"}},
        {"error": {}},
        {},
        {"error": "SECRET_PROVIDER_BODY"},
        "SECRET_PROVIDER_BODY",
        None,
    ],
)
def test_422_missing_or_malformed_structured_fields_are_unavailable(
    body,
    caplog,
):
    error = _status_error_with_body(
        openai.UnprocessableEntityError,
        422,
        body,
    )
    adapter, _ = _adapter(error=error)
    with caplog.at_level(
        logging.WARNING,
        logger=DIAGNOSTIC_LOGGER_NAME,
    ):
        adapter([{"role": "user", "content": "go"}])
    message = _diagnostic_message(caplog)
    assert "error_code=unavailable" in message
    assert "error_param=unavailable" in message
    assert "SECRET_PROVIDER" not in caplog.text


@pytest.mark.parametrize(
    ("code", "param", "expected_code", "expected_param"),
    [
        (123, "response_format", "unavailable", "response_format"),
        ("invalid_parameter", 456, "invalid_parameter", "unavailable"),
        (
            "invalid_parameter\nSECRET_CODE",
            "response_format",
            "unavailable",
            "response_format",
        ),
        (
            "invalid_parameter",
            "response_format\nSECRET_PARAM",
            "invalid_parameter",
            "unavailable",
        ),
        ("invalid_parameter", " response_format", "invalid_parameter", "unavailable"),
        ("invalid_parameter", "response_format ", "invalid_parameter", "unavailable"),
        (
            "invalid_parameter",
            "sk-SECRET_PARAM",
            "invalid_parameter",
            "unavailable",
        ),
        (
            "unknown_code_SECRET",
            "unknown_param_SECRET",
            "unavailable",
            "unavailable",
        ),
    ],
)
def test_422_malicious_or_unknown_values_are_never_logged(
    code,
    param,
    expected_code,
    expected_param,
    caplog,
):
    body = {
        "error": {
            "message": "SECRET_PROVIDER_MESSAGE",
            "code": code,
            "param": param,
        }
    }
    error = _status_error_with_body(
        openai.UnprocessableEntityError,
        422,
        body,
    )
    adapter, _ = _adapter(error=error)
    with caplog.at_level(
        logging.WARNING,
        logger=DIAGNOSTIC_LOGGER_NAME,
    ):
        adapter([{"role": "user", "content": "go"}])
    message = _diagnostic_message(caplog)
    assert f"error_code={expected_code}" in message
    assert f"error_param={expected_param}" in message
    assert "SECRET_CODE" not in caplog.text
    assert "SECRET_PARAM" not in caplog.text
    assert "SECRET_PROVIDER" not in caplog.text
    assert "\n" not in message


def test_non_422_diagnostics_keep_existing_fields(caplog):
    error = _status_error(openai.AuthenticationError, 401)
    adapter, _ = _adapter(error=error)
    with caplog.at_level(
        logging.WARNING,
        logger=DIAGNOSTIC_LOGGER_NAME,
    ):
        adapter([{"role": "user", "content": "go"}])
    message = _diagnostic_message(caplog)
    assert "category=authentication_error" in message
    assert "status_code=401" in message
    assert "error_code=" not in message
    assert "error_param=" not in message


def _debug_records(caplog):
    return [
        record
        for record in caplog.records
        if record.name == DIAGNOSTIC_LOGGER_NAME
        and "deepseek_provider_422_body" in record.getMessage()
    ]


def test_debug_provider_errors_is_off_by_default(
    monkeypatch,
    caplog,
):
    monkeypatch.delenv(
        "AGENTGUARD_DEBUG_PROVIDER_ERRORS",
        raising=False,
    )
    body = {
        "error": {
            "message": "sk-SECRETKEYVALUE12345",
            "code": "invalid_parameter",
            "param": "response_format",
            "authorization": "Bearer SECRETTOKENVALUE12345",
            "cookie": "session=SECRETCOOKIE",
            "nested": {"api_key": "SECRETAPIKEY"},
        }
    }
    error = _status_error_with_body(
        openai.UnprocessableEntityError,
        422,
        body,
    )
    adapter, _ = _adapter(error=error)
    with caplog.at_level(
        logging.WARNING,
        logger=DIAGNOSTIC_LOGGER_NAME,
    ):
        adapter([{"role": "user", "content": "go"}])
    assert _debug_records(caplog) == []
    assert "error_code=invalid_parameter" in caplog.text
    assert "error_param=response_format" in caplog.text
    assert "SECRETKEYVALUE" not in caplog.text
    assert "SECRETTOKENVALUE" not in caplog.text
    assert "SECRETCOOKIE" not in caplog.text
    assert "SECRETAPIKEY" not in caplog.text


def test_debug_provider_errors_logs_redacted_structured_body(
    monkeypatch,
    caplog,
):
    monkeypatch.setenv("AGENTGUARD_DEBUG_PROVIDER_ERRORS", "1")
    body = {
        "error": {
            "message": (
                "Invalid value. key=sk-SECRETKEYVALUE12345 "
                "Authorization: Bearer SECRETTOKENVALUE12345 "
                "session=SECRETCOOKIE\nINJECTED_LINE "
                "Look up order P100."
            ),
            "code": "invalid_parameter",
            "param": "response_format",
            "authorization": "Bearer SECRETTOKENVALUE12345",
            "cookie": "session=SECRETCOOKIE",
            "api_key": "sk-SECRETKEYVALUE12345",
            "nested": {"token": "SECRETNESTEDTOKEN"},
            "messages": [{"role": "user", "content": "SECRETMESSAGE"}],
        }
    }
    error = _status_error_with_body(
        openai.UnprocessableEntityError,
        422,
        body,
    )
    adapter, _ = _adapter(error=error)
    with caplog.at_level(
        logging.WARNING,
        logger=DIAGNOSTIC_LOGGER_NAME,
    ):
        result = adapter(
            [{"role": "user", "content": "Look up order P100."}]
        )
    assert result == _safe_failure()
    debug = _debug_records(caplog)
    assert len(debug) == 1
    message = debug[0].getMessage()
    assert "\n" not in message
    payload = json.loads(message.split(" ", 1)[1])
    assert payload["error"]["code"] == "invalid_parameter"
    assert payload["error"]["param"] == "response_format"
    rendered = json.dumps(payload, sort_keys=True)
    assert "SECRETKEYVALUE" not in rendered
    assert "SECRETTOKENVALUE" not in rendered
    assert "SECRETCOOKIE" not in rendered
    assert "SECRETNESTEDTOKEN" not in rendered
    assert "SECRETMESSAGE" not in rendered
    assert "Look up order P100." not in rendered
    assert payload["error"]["authorization"] == "[REDACTED]"
    assert payload["error"]["cookie"] == "[REDACTED]"
    assert payload["error"]["api_key"] == "[REDACTED]"
    assert payload["error"]["nested"]["token"] == "[REDACTED]"
    assert payload["error"]["messages"] == "[REDACTED]"
    assert "[REDACTED]" in payload["error"]["message"]
    assert "sk-" not in caplog.text
    assert "Bearer SECRET" not in caplog.text
    assert "Look up order P100." not in caplog.text


def test_debug_body_redacts_system_prompt_echo(
    monkeypatch,
    caplog,
):
    monkeypatch.setenv("AGENTGUARD_DEBUG_PROVIDER_ERRORS", "1")
    body = {
        "error": {
            "message": f"Rejected prompt: {SYSTEM_INSTRUCTION}",
            "code": "invalid_parameter",
            "param": "messages",
        }
    }
    error = _status_error_with_body(
        openai.UnprocessableEntityError,
        422,
        body,
    )
    adapter, _ = _adapter(error=error)
    with caplog.at_level(
        logging.WARNING,
        logger=DIAGNOSTIC_LOGGER_NAME,
    ):
        adapter([{"role": "user", "content": "go"}])
    debug = _debug_records(caplog)
    assert len(debug) == 1
    message = debug[0].getMessage()
    assert SYSTEM_INSTRUCTION not in message
    assert "[REDACTED]" in message


def test_debug_flag_does_not_log_non_422_errors(
    monkeypatch,
    caplog,
):
    monkeypatch.setenv("AGENTGUARD_DEBUG_PROVIDER_ERRORS", "1")
    error = _status_error(openai.AuthenticationError, 401)
    adapter, _ = _adapter(error=error)
    with caplog.at_level(
        logging.WARNING,
        logger=DIAGNOSTIC_LOGGER_NAME,
    ):
        adapter([{"role": "user", "content": "go"}])
    assert _debug_records(caplog) == []
    assert "SECRET_PROVIDER_BODY" not in caplog.text


@pytest.mark.parametrize("body", ["SECRET_PROVIDER_BODY", None, 123])
def test_debug_body_unavailable_for_non_structured_body(
    monkeypatch,
    caplog,
    body,
):
    monkeypatch.setenv("AGENTGUARD_DEBUG_PROVIDER_ERRORS", "1")
    error = _status_error_with_body(
        openai.UnprocessableEntityError,
        422,
        body,
    )
    adapter, _ = _adapter(error=error)
    with caplog.at_level(
        logging.WARNING,
        logger=DIAGNOSTIC_LOGGER_NAME,
    ):
        adapter([{"role": "user", "content": "go"}])
    messages = [
        record.getMessage()
        for record in caplog.records
        if record.name == DIAGNOSTIC_LOGGER_NAME
    ]
    assert any(
        message == "deepseek_provider_422_body unavailable"
        for message in messages
    )
    assert "SECRET_PROVIDER_BODY" not in caplog.text


def test_tool_messages_are_serialized_as_labeled_user_context():
    input_messages = [
        {"role": "user", "content": "Look up order P100."},
        {
            "role": "tool",
            "content": '{"outcome":"blocked","reason_code":"order_unavailable"}',
        },
    ]
    original = json.loads(json.dumps(input_messages))
    provider_messages = _provider_messages(input_messages)

    assert [message["role"] for message in provider_messages] == [
        "system",
        "user",
        "user",
    ]
    assert provider_messages[1]["content"] == "Look up order P100."
    assert provider_messages[2]["content"] == (
        TOOL_RESULT_CONTEXT_PREFIX
        + '{"outcome":"blocked","reason_code":"order_unavailable"}'
    )
    assert input_messages == original
    assert all(
        set(message) == {"role", "content"}
        for message in provider_messages
    )
    assert "tool_call_id" not in json.dumps(
        provider_messages,
        sort_keys=True,
    )


def _multi_turn_client(contents):
    fake_client = _FakeClient(contents=contents)
    adapter = DeepSeekModelAdapter(
        api_key="unit-test-key",
        client=fake_client,
    )
    app = create_app(
        model=adapter,
        api_keys={"test-bearer-token": "user-1"},
    )
    return app, fake_client, TestClient(app)


def _api_headers(key):
    return {
        "Authorization": "Bearer test-bearer-token",
        "Idempotency-Key": key,
    }


def test_multi_turn_get_order_preserves_contracts_and_business_outcome():
    app, fake, client = _multi_turn_client(
        [
            '{"type":"tool_call","operation":"get_order",'
            '"order_id":"P100"}',
            '{"type":"final","response":"Order P100 is pending."}',
        ]
    )
    response = client.post(
        "/v1/runs",
        json={"message": "Look up order P100."},
        headers=_api_headers("tool-msg-get-order-0001"),
    )
    assert response.status_code == 201
    body = response.json()
    assert set(body) == {
        "run_id",
        "status",
        "execution_state",
        "version",
        "confirmation",
        "result",
    }
    assert body["status"] == "completed"
    assert body["result"]["response"] == "Order P100 is pending."
    order = app.state.service.orchestrator._state.inspect_order("P100")
    assert order is not None
    assert order.status.value == "pending"

    assert len(fake.chat.completions.calls) == 2
    first, second = fake.chat.completions.calls
    assert [m["role"] for m in first["messages"]] == ["system", "user"]
    assert [m["role"] for m in second["messages"]] == [
        "system",
        "user",
        "user",
    ]
    assert second["messages"][2]["content"].startswith(
        TOOL_RESULT_CONTEXT_PREFIX
    )
    assert "P100" in second["messages"][2]["content"]
    assert "tool_call_id" not in json.dumps(second, sort_keys=True)

    trace = client.get(
        f"/v1/runs/{body['run_id']}/trace",
        headers=_api_headers("tool-msg-get-order-0002"),
    )
    assert trace.status_code == 200
    payload = trace.json()
    assert set(payload) == {"schema_version", "trace", "events"}
    assert any(
        event.get("event_type") == "tool_result"
        and event.get("status") == "success"
        for event in payload["events"]
    )


def test_multi_turn_cancel_after_confirmation_serializes_tool_result():
    app, fake, client = _multi_turn_client(
        [
            '{"type":"tool_call","operation":"cancel_order",'
            '"order_id":"P100"}',
            '{"type":"final","response":"Order P100 was cancelled."}',
        ]
    )
    created = client.post(
        "/v1/runs",
        json={"message": "Cancel order P100."},
        headers=_api_headers("tool-msg-cancel-0001"),
    )
    assert created.status_code == 201
    created_body = created.json()
    assert created_body["status"] == "awaiting_confirmation"
    assert created_body["confirmation"]["operation"] == "cancel_order"

    confirmed = client.post(
        f"/v1/runs/{created_body['run_id']}/confirmation",
        json={
            "decision": "approve",
            "expected_version": created_body["version"],
        },
        headers=_api_headers("tool-msg-cancel-0002"),
    )
    assert confirmed.status_code == 200
    assert confirmed.json()["status"] == "completed"
    assert confirmed.json()["result"]["response"] == (
        "Order P100 was cancelled."
    )
    order = app.state.service.orchestrator._state.inspect_order("P100")
    assert order is not None
    assert order.status.value == "cancelled"

    assert len(fake.chat.completions.calls) == 2
    second = fake.chat.completions.calls[1]
    assert [m["role"] for m in second["messages"]] == [
        "system",
        "user",
        "user",
    ]
    assert second["messages"][2]["content"].startswith(
        TOOL_RESULT_CONTEXT_PREFIX
    )
    assert "cancelled" in second["messages"][2]["content"]
    assert "tool_call_id" not in json.dumps(second, sort_keys=True)


def test_blocked_cross_user_get_order_preserves_error_context_and_denial():
    app, fake, client = _multi_turn_client(
        [
            '{"type":"tool_call","operation":"get_order",'
            '"order_id":"P200"}',
            '{"type":"final","response":"I cannot access that order."}',
        ]
    )
    response = client.post(
        "/v1/runs",
        json={"message": "Look up order P200."},
        headers=_api_headers("tool-msg-blocked-0001"),
    )
    assert response.status_code == 201
    body = response.json()
    assert body["status"] == "completed"
    assert body["result"]["response"] == "I cannot access that order."
    order = app.state.service.orchestrator._state.inspect_order("P200")
    assert order is not None
    assert order.status.value == "processing"

    assert len(fake.chat.completions.calls) == 2
    second = fake.chat.completions.calls[1]
    assert [m["role"] for m in second["messages"]] == [
        "system",
        "user",
        "user",
    ]
    context = second["messages"][2]["content"]
    assert context.startswith(TOOL_RESULT_CONTEXT_PREFIX)
    assert "order_unavailable" in context
    assert "tool_call_id" not in json.dumps(second, sort_keys=True)


def test_diagnostics_are_not_exposed_in_public_api_responses():
    adapter, _ = _adapter(contents=["not json"])
    app = create_app(
        model=adapter,
        api_keys={"token-user-1": "user-1"},
    )
    response = TestClient(app).post(
        "/v1/runs",
        json={"message": "Look up order P100."},
        headers={
            "Authorization": "Bearer token-user-1",
            "Idempotency-Key": "idem-key-0000000001",
        },
    )
    assert response.status_code == 201
    body = response.json()
    assert body["result"]["response"] == SAFE_MODEL_FAILURE_RESPONSE
    serialized = json.dumps(body, sort_keys=True)
    assert "invalid_model_output" not in serialized
    assert "correlation_id" not in serialized
    assert "deepseek_adapter_failure" not in serialized
