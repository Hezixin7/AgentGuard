import asyncio
import json

from fastapi.testclient import TestClient

from deployment.api import create_app
from deployment.body_limit import (
    REQUEST_BODY_LIMIT_BYTES,
    BodySizeLimitMiddleware,
)


def _scope(method="POST", headers=None):
    return {
        "type": "http",
        "method": method,
        "headers": headers or [],
    }


def _run_middleware(chunks, max_bytes, method="POST"):
    state = {"app_called": 0, "body": None, "sent": []}

    async def app(scope, receive, send):
        state["app_called"] += 1
        message = await receive()
        state["body"] = message.get("body")
        await send(
            {
                "type": "http.response.start",
                "status": 200,
                "headers": [],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": b"ok",
                "more_body": False,
            }
        )

    messages = [
        {
            "type": "http.request",
            "body": chunk,
            "more_body": index < len(chunks) - 1,
        }
        for index, chunk in enumerate(chunks)
    ]

    async def receive():
        return messages.pop(0)

    async def send(message):
        state["sent"].append(message)

    middleware = BodySizeLimitMiddleware(app, max_bytes=max_bytes)
    asyncio.run(middleware(_scope(method=method), receive, send))
    return state


def _response_status(state):
    return state["sent"][0]["status"]


def _response_json(state):
    return json.loads(state["sent"][1]["body"])


def test_streamed_body_without_content_length_is_rejected_before_app():
    state = _run_middleware(
        [b"a" * 10, b"b" * 10],
        max_bytes=16,
    )
    assert state["app_called"] == 0
    assert _response_status(state) == 413
    body = _response_json(state)
    assert body["error"]["code"] == "request_too_large"
    assert body["error"]["request_id"]


def test_chunked_body_is_counted_across_messages():
    state = _run_middleware(
        [b"12345", b"67890", b"abcde"],
        max_bytes=14,
    )
    assert state["app_called"] == 0
    assert _response_status(state) == 413


def test_body_at_the_limit_is_replayed_to_the_app():
    state = _run_middleware(
        [b"12345678", b"90123456"],
        max_bytes=16,
    )
    assert state["app_called"] == 1
    assert state["body"] == b"1234567890123456"
    assert _response_status(state) == 200


def test_body_below_the_limit_is_replayed_to_the_app():
    state = _run_middleware(
        [b"12345", b"67890"],
        max_bytes=16,
    )
    assert state["app_called"] == 1
    assert state["body"] == b"1234567890"
    assert _response_status(state) == 200


def test_non_limited_methods_bypass_the_body_limit():
    state = _run_middleware(
        [b"x" * 100],
        max_bytes=4,
        method="GET",
    )
    assert state["app_called"] == 1
    assert _response_status(state) == 200


def test_default_limit_is_16384_bytes():
    assert REQUEST_BODY_LIMIT_BYTES == 16_384


def _counting_model():
    calls = []

    def model(messages):
        calls.append(messages)
        return {"type": "final", "response": "ok"}

    return model, calls


def _app():
    model, calls = _counting_model()
    app = create_app(
        model=model,
        api_keys={"test-bearer-token": "user-1"},
    )
    return app, calls


def _headers(key="idem-key-0000000001"):
    return {
        "Authorization": "Bearer test-bearer-token",
        "Idempotency-Key": key,
    }


def test_oversized_http_request_cannot_invoke_the_orchestrator():
    app, calls = _app()
    client = TestClient(app)
    response = client.post(
        "/v1/runs",
        content=b"x" * (REQUEST_BODY_LIMIT_BYTES + 1),
        headers={
            **_headers(),
            "Content-Type": "application/json",
        },
    )
    assert response.status_code == 413
    assert response.json()["error"]["code"] == "request_too_large"
    assert response.headers["X-Request-ID"]
    assert calls == []


def test_valid_http_request_below_limit_still_reaches_the_app():
    app, calls = _app()
    client = TestClient(app)
    response = client.post(
        "/v1/runs",
        json={"message": "hello"},
        headers=_headers(),
    )
    assert response.status_code == 201
    assert response.json()["status"] == "completed"
    assert len(calls) == 1
