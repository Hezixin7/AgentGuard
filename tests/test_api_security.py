import json
from pathlib import Path

from fastapi.testclient import TestClient

from deployment.api import create_app

API_KEYS = {
    "token-user-1": "user-1",
    "token-user-2": "user-2",
}


def _model():
    def model(messages):
        if any(message.get("role") == "tool" for message in messages):
            return {"type": "final", "response": "Completed."}
        return {
            "type": "tool_call",
            "operation": "cancel_order",
            "order_id": "P100",
        }

    return model


def _client():
    app = create_app(model=_model(), api_keys=API_KEYS)
    return app, TestClient(app)


def _headers(user="user-1", key="idem-key-0000000001"):
    return {
        "Authorization": f"Bearer token-{user}",
        "Idempotency-Key": key,
    }


def _create(client, message="Cancel my order."):
    return client.post(
        "/v1/runs",
        json={"message": message},
        headers=_headers(),
    )


def test_authenticated_identity_never_comes_from_request_body():
    _, client = _client()
    response = client.post(
        "/v1/runs",
        json={"message": "Cancel.", "user_id": "user-2"},
        headers=_headers(user="user-1"),
    )
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "invalid_request"


def test_public_api_has_no_evaluation_or_group_controls():
    app, client = _client()
    paths = {route.path for route in app.routes}
    assert not any("eval" in path for path in paths)
    assert "/v1/runs" in paths
    assert "/v1/runs/{run_id}" in paths
    assert "/v1/runs/{run_id}/confirmation" in paths
    assert "/v1/runs/{run_id}/trace" in paths
    assert "/healthz" in paths

    for group in ("baseline_a", "baseline_b", "agentguard"):
        response = client.post(
            "/v1/runs",
            json={"message": "hello", "group": group},
            headers=_headers(),
        )
        assert response.status_code == 422

    deployment_files = Path("deployment").glob("*.py")
    text = "\n".join(
        path.read_text(encoding="utf-8")
        for path in deployment_files
    )
    assert "for_evaluation" not in text
    assert "BASELINE_A" not in text
    assert "ExperimentGroup" not in text


def test_trace_is_owner_only_and_redacted():
    app, client = _client()
    created = _create(client)
    run_id = created.json()["run_id"]
    internal = app.state.service._runs[run_id]
    assert internal.challenge is not None
    internal_confirmation_id = internal.challenge.confirmation_id
    assert internal.thread_id is not None

    response = client.get(
        f"/v1/runs/{run_id}/trace",
        headers=_headers(),
    )
    assert response.status_code == 200
    payload = response.json()
    text = json.dumps(payload, sort_keys=True)

    assert payload["trace"]["run_id"] == run_id
    assert internal_confirmation_id not in text
    assert internal.thread_id not in text
    assert "user-1" not in text
    assert "Cancel my order." not in text
    assert "token-user-1" not in text

    for event in payload["events"]:
        if event.get("event_type") == "llm_output":
            assert event.get("raw_output") == "[REDACTED]"
        if "confirmation_id" in event:
            assert event["confirmation_id"] == "[REDACTED]"
        if "authenticated_user_id" in event:
            assert event["authenticated_user_id"] == "[REDACTED]"
        if "checked_user_id" in event:
            assert event["checked_user_id"] == "[REDACTED]"

    foreign = client.get(
        f"/v1/runs/{run_id}/trace",
        headers=_headers(user="user-2"),
    )
    assert foreign.status_code == 404


def test_public_responses_do_not_expose_internal_identifiers():
    _, client = _client()
    created = _create(client)
    body = created.json()
    text = json.dumps(body, sort_keys=True)
    assert "thread_id" not in text
    assert "confirmation_id" not in text
    assert "capability" not in text
    assert body["confirmation"]["operation"] == "cancel_order"
