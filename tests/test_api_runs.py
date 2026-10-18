from fastapi.testclient import TestClient

from deployment.api import create_app

API_KEYS = {
    "token-user-1": "user-1",
    "token-user-2": "user-2",
}


def _model(proposal, final="Completed."):
    calls = []

    def model(messages):
        calls.append(messages)
        if any(message.get("role") == "tool" for message in messages):
            return {"type": "final", "response": final}
        return dict(proposal)

    return model, calls


def _app(proposal, final="Completed."):
    model, calls = _model(proposal, final)
    app = create_app(model=model, api_keys=API_KEYS)
    return app, TestClient(app), calls


def _headers(user="user-1", key="idem-key-0000000001"):
    return {
        "Authorization": f"Bearer token-{user}",
        "Idempotency-Key": key,
    }


def _create(client, message="Cancel my order.", headers=None):
    return client.post(
        "/v1/runs",
        json={"message": message},
        headers=headers or _headers(),
    )


def _cancel_proposal():
    return {
        "type": "tool_call",
        "operation": "cancel_order",
        "order_id": "P100",
    }


def test_create_confirmation_approve_and_trace():
    app, client, _ = _app(_cancel_proposal())
    created = _create(client)
    assert created.status_code == 201
    body = created.json()
    assert body["status"] == "awaiting_confirmation"
    assert body["execution_state"] == "normal"
    assert body["version"] == 1
    assert body["confirmation"]["operation"] == "cancel_order"
    assert body["confirmation"]["order_id"] == "P100"
    assert "operation" not in body["confirmation"]["parameters"]
    assert "confirmation_id" not in body["confirmation"]
    run_id = body["run_id"]

    fetched = client.get(
        f"/v1/runs/{run_id}",
        headers=_headers(),
    )
    assert fetched.status_code == 200
    assert fetched.json()["status"] == "awaiting_confirmation"

    paused_trace = client.get(
        f"/v1/runs/{run_id}/trace",
        headers=_headers(),
    )
    assert paused_trace.status_code == 200
    assert paused_trace.json()["trace"]["run_id"] == run_id
    assert paused_trace.json()["events"]

    decided = client.post(
        f"/v1/runs/{run_id}/confirmation",
        json={"decision": "approve", "expected_version": 1},
        headers=_headers(key="idem-key-0000000002"),
    )
    assert decided.status_code == 200
    completed = decided.json()
    assert completed["status"] == "completed"
    assert completed["execution_state"] == "normal"
    assert completed["version"] == 2
    assert completed["confirmation"] is None
    assert completed["result"]["response"] == "Completed."

    trace = client.get(
        f"/v1/runs/{run_id}/trace",
        headers=_headers(),
    )
    assert trace.status_code == 200
    assert trace.json()["trace"]["run_id"] == run_id


def test_reject_and_business_denial_are_completed():
    _, client, _ = _app(_cancel_proposal())
    run_id = _create(client).json()["run_id"]
    rejected = client.post(
        f"/v1/runs/{run_id}/confirmation",
        json={"decision": "reject", "expected_version": 1},
        headers=_headers(key="idem-key-0000000002"),
    )
    assert rejected.status_code == 200
    assert rejected.json()["status"] == "completed"
    assert rejected.json()["result"]["response"] == "Completed."

    foreign = {
        "type": "tool_call",
        "operation": "cancel_order",
        "order_id": "P200",
    }
    _, denial_client, _ = _app(foreign)
    denied = _create(denial_client)
    assert denied.status_code == 201
    assert denied.json()["status"] == "completed"
    assert denied.json()["result"]["response"] == "Completed."


def test_validation_and_authentication_errors_are_sanitized():
    _, client, _ = _app(_cancel_proposal(), final="ok")

    unauth = client.post(
        "/v1/runs",
        json={"message": "hello"},
        headers={"Idempotency-Key": "idem-key-0000000001"},
    )
    assert unauth.status_code == 401
    assert unauth.json()["error"]["code"] == "unauthenticated"

    bad_token = client.post(
        "/v1/runs",
        json={"message": "hello"},
        headers={
            "Authorization": "Bearer nope",
            "Idempotency-Key": "idem-key-0000000001",
        },
    )
    assert bad_token.status_code == 401

    injected = client.post(
        "/v1/runs",
        json={
            "message": "hello",
            "user_id": "user-2",
            "group": "baseline_a",
            "capability": "secret",
        },
        headers=_headers(),
    )
    assert injected.status_code == 422
    envelope = injected.json()["error"]
    assert envelope["code"] == "invalid_request"
    assert envelope["request_id"]
    assert "user-2" not in str(envelope)
    assert "secret" not in str(envelope)

    empty = client.post(
        "/v1/runs",
        json={"message": "   "},
        headers=_headers(),
    )
    assert empty.status_code == 422

    too_long = client.post(
        "/v1/runs",
        json={"message": "x" * 4001},
        headers=_headers(),
    )
    assert too_long.status_code == 422

    missing_key = client.post(
        "/v1/runs",
        json={"message": "hello"},
        headers={"Authorization": "Bearer token-user-1"},
    )
    assert missing_key.status_code == 422

    short_key = client.post(
        "/v1/runs",
        json={"message": "hello"},
        headers=_headers(key="short"),
    )
    assert short_key.status_code == 422

    oversized = client.post(
        "/v1/runs",
        content=b"x" * 20_000,
        headers={
            "Authorization": "Bearer token-user-1",
            "Idempotency-Key": "idem-key-0000000001",
            "Content-Type": "application/json",
        },
    )
    assert oversized.status_code == 413
    assert oversized.json()["error"]["code"] == "request_too_large"

    bad_decision = client.post(
        "/v1/runs/unknown/confirmation",
        json={"decision": "maybe", "expected_version": 0},
        headers=_headers(),
    )
    assert bad_decision.status_code == 422

    missing_version = client.post(
        "/v1/runs/unknown/confirmation",
        json={"decision": "approve"},
        headers=_headers(),
    )
    assert missing_version.status_code == 422


def test_owner_only_access():
    _, client, _ = _app(_cancel_proposal())
    run_id = _create(client).json()["run_id"]

    assert client.get(
        f"/v1/runs/{run_id}",
        headers=_headers(user="user-2"),
    ).status_code == 404
    assert client.post(
        f"/v1/runs/{run_id}/confirmation",
        json={"decision": "approve", "expected_version": 1},
        headers=_headers(user="user-2", key="idem-key-0000000002"),
    ).status_code == 404
    assert client.get(
        f"/v1/runs/{run_id}/trace",
        headers=_headers(user="user-2"),
    ).status_code == 404

    assert client.get(
        "/v1/runs/does-not-exist",
        headers=_headers(),
    ).status_code == 404


def test_idempotent_create_replay_and_conflict():
    app, client, calls = _app(_cancel_proposal())
    first = _create(client)
    second = _create(client)
    assert first.status_code == 201
    assert second.status_code == 201
    assert second.headers.get("Idempotent-Replay") == "true"
    assert first.json() == second.json()
    assert len(app.state.service._runs) == 1
    assert len(calls) == 1

    conflict = _create(client, message="A different request.")
    assert conflict.status_code == 409
    assert conflict.json()["error"]["code"] == "idempotency_key_reuse"
    assert len(app.state.service._runs) == 1


def test_confirmation_replay_stale_version_and_terminal_state():
    app, client, _ = _app(_cancel_proposal())
    run_id = _create(client).json()["run_id"]

    stale = client.post(
        f"/v1/runs/{run_id}/confirmation",
        json={"decision": "approve", "expected_version": 0},
        headers=_headers(key="idem-key-0000000002"),
    )
    assert stale.status_code == 409
    assert stale.json()["error"]["code"] == "run_version_conflict"
    assert client.get(
        f"/v1/runs/{run_id}",
        headers=_headers(),
    ).json()["version"] == 1

    original_decide = app.state.service.orchestrator.decide
    decide_calls = []

    def counted_decide(*args, **kwargs):
        decide_calls.append(1)
        return original_decide(*args, **kwargs)

    app.state.service.orchestrator.decide = counted_decide

    approved = client.post(
        f"/v1/runs/{run_id}/confirmation",
        json={"decision": "approve", "expected_version": 1},
        headers=_headers(key="idem-key-0000000002"),
    )
    assert approved.status_code == 200
    replay = client.post(
        f"/v1/runs/{run_id}/confirmation",
        json={"decision": "approve", "expected_version": 1},
        headers=_headers(key="idem-key-0000000002"),
    )
    assert replay.status_code == 200
    assert replay.headers.get("Idempotent-Replay") == "true"
    assert replay.json() == approved.json()
    assert len(decide_calls) == 1

    terminal = client.post(
        f"/v1/runs/{run_id}/confirmation",
        json={"decision": "approve", "expected_version": 2},
        headers=_headers(key="idem-key-0000000003"),
    )
    assert terminal.status_code == 409
    assert terminal.json()["error"]["code"] == (
        "run_not_awaiting_confirmation"
    )


def test_healthz_requires_no_authentication():
    _, client, _ = _app(_cancel_proposal())
    response = client.get("/healthz")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_framework_errors_use_sanitized_envelope():
    _, client, _ = _app(_cancel_proposal())
    missing = client.get("/v1/unknown")
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "not_found"
    assert missing.json()["error"]["request_id"]

    wrong_method = client.request(
        "DELETE",
        "/v1/runs",
        headers=_headers(),
    )
    assert wrong_method.status_code == 405
    assert wrong_method.json()["error"]["code"] == "method_not_allowed"
