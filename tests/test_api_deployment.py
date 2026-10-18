import json
from concurrent.futures import ThreadPoolExecutor

from fastapi.testclient import TestClient

from agent_graph import RunResult, RunStatus
from deployment.api import create_app
from deployment.auth import Principal
from deployment.errors import ApiError
from simulated_tools import FaultMode

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


def _headers(key="idem-key-0000000001", user="user-1"):
    return {
        "Authorization": f"Bearer token-{user}",
        "Idempotency-Key": key,
    }


def _cancel():
    return {
        "type": "tool_call",
        "operation": "cancel_order",
        "order_id": "P100",
    }


def _refund():
    return {
        "type": "tool_call",
        "operation": "refund_order",
        "order_id": "S300",
        "amount": "49.95",
        "currency": "USD",
    }


def test_timeout_after_commit_remains_committed_and_replays():
    app, client, _ = _app(_refund(), final="Refunded.")
    created = client.post(
        "/v1/runs",
        json={"message": "Refund S300."},
        headers=_headers(),
    )
    assert created.status_code == 201
    run_id = created.json()["run_id"]

    orchestrator = app.state.service.orchestrator
    original_call = orchestrator._call_tool
    injected = {"count": 0}

    def call_tool(request):
        if request.operation == "refund_order" and injected["count"] == 0:
            injected["count"] += 1
            return orchestrator._tools.refund_order(
                request.order_id,
                request.idempotency_key,
                amount=request.amount,
                currency=request.currency,
                fault=FaultMode.TIMEOUT_AFTER_COMMIT,
            )
        return original_call(request)

    orchestrator._call_tool = call_tool
    decided = client.post(
        f"/v1/runs/{run_id}/confirmation",
        json={"decision": "approve", "expected_version": 1},
        headers=_headers(key="idem-key-0000000002"),
    )
    assert decided.status_code == 200
    assert decided.json()["status"] == "completed"

    state = orchestrator._state
    refunds = state.inspect_refunds()
    assert len(refunds) == 1
    assert str(refunds[0].amount) == "49.95"
    assert state.inspect_order("S300").status.value == "refunded"
    assert len(state.inspect_idempotency_records()) == 1


def test_indeterminate_create_is_queryable_and_never_reexecuted():
    app, client, calls = _app(_cancel())
    service = app.state.service
    original_start = service.orchestrator.start
    starts = []

    def counted_start(*args, **kwargs):
        starts.append(1)
        return original_start(*args, **kwargs)

    service.orchestrator.start = counted_start

    def fail_cache(*args, **kwargs):
        raise RuntimeError("cache write failed")

    service._commit_idempotent = fail_cache
    response = client.post(
        "/v1/runs",
        json={"message": "Cancel."},
        headers=_headers(),
    )
    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "execution_outcome_indeterminate"
    run_id = error["details"]["run_id"]
    assert starts == [1]

    fetched = client.get(
        f"/v1/runs/{run_id}",
        headers=_headers(),
    )
    assert fetched.status_code == 200
    assert fetched.json()["execution_state"] == "indeterminate"
    assert fetched.json()["confirmation"] is None

    retry = client.post(
        "/v1/runs",
        json={"message": "Cancel."},
        headers=_headers(),
    )
    assert retry.status_code == 409
    assert retry.json()["error"]["code"] == "request_indeterminate"
    assert starts == [1]
    assert len(calls) == 1


def test_indeterminate_confirmation_blocks_further_decisions():
    app, client, _ = _app(_cancel())
    run_id = client.post(
        "/v1/runs",
        json={"message": "Cancel."},
        headers=_headers(),
    ).json()["run_id"]

    service = app.state.service
    original_decide = service.orchestrator.decide
    decisions = []

    def counted_decide(*args, **kwargs):
        decisions.append(1)
        return original_decide(*args, **kwargs)

    service.orchestrator.decide = counted_decide

    def fail_cache(*args, **kwargs):
        raise RuntimeError("cache write failed")

    service._commit_idempotent = fail_cache
    response = client.post(
        f"/v1/runs/{run_id}/confirmation",
        json={"decision": "approve", "expected_version": 1},
        headers=_headers(key="idem-key-0000000002"),
    )
    assert response.status_code == 500
    assert response.json()["error"]["code"] == (
        "execution_outcome_indeterminate"
    )
    assert decisions == [1]

    fetched = client.get(
        f"/v1/runs/{run_id}",
        headers=_headers(),
    )
    assert fetched.status_code == 200
    assert fetched.json()["execution_state"] == "indeterminate"

    retry = client.post(
        f"/v1/runs/{run_id}/confirmation",
        json={"decision": "reject", "expected_version": 2},
        headers=_headers(key="idem-key-0000000003"),
    )
    assert retry.status_code == 409
    assert retry.json()["error"]["code"] == "run_outcome_indeterminate"
    assert decisions == [1]


def test_concurrent_confirmations_invoke_decide_once():
    app, client, _ = _app(_cancel())
    run_id = client.post(
        "/v1/runs",
        json={"message": "Cancel."},
        headers=_headers(),
    ).json()["run_id"]

    service = app.state.service
    original_decide = service.orchestrator.decide
    decisions = []

    def counted_decide(*args, **kwargs):
        decisions.append(1)
        return original_decide(*args, **kwargs)

    service.orchestrator.decide = counted_decide
    principal = Principal("user-1")

    def submit(key):
        try:
            return service.confirm_run(
                principal,
                run_id,
                key,
                "approve",
                1,
            )
        except ApiError as error:
            return error

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(
            pool.map(
                submit,
                [
                    "idem-key-0000000010",
                    "idem-key-0000000011",
                ],
            )
        )

    successes = [
        result for result in results if not isinstance(result, ApiError)
    ]
    failures = [
        result for result in results if isinstance(result, ApiError)
    ]
    assert len(successes) == 1
    assert len(failures) == 1
    assert failures[0].status_code == 409
    assert failures[0].code in {
        "run_not_awaiting_confirmation",
        "run_version_conflict",
    }
    assert decisions == [1]


def test_expired_confirmation_is_frozen_outcome_without_execution():
    class Clock:
        def __init__(self):
            self.value = 100.0

        def __call__(self):
            return self.value

        def advance(self, seconds):
            self.value += seconds

    clock = Clock()
    model, _ = _model(_cancel(), final="Done.")
    app = create_app(
        model=model,
        api_keys=API_KEYS,
        clock=clock,
    )
    client = TestClient(app)
    created = client.post(
        "/v1/runs",
        json={"message": "Cancel."},
        headers=_headers(),
    )
    run_id = created.json()["run_id"]
    clock.advance(301)
    decided = client.post(
        f"/v1/runs/{run_id}/confirmation",
        json={"decision": "approve", "expected_version": 1},
        headers=_headers(key="idem-key-0000000002"),
    )
    assert decided.status_code == 200
    assert decided.json()["status"] == "completed"
    order = app.state.service.orchestrator._state.inspect_order("P100")
    assert order.status.value == "pending"


def test_process_restart_loses_runs():
    _, first_client, _ = _app(_cancel())
    run_id = first_client.post(
        "/v1/runs",
        json={"message": "Cancel."},
        headers=_headers(),
    ).json()["run_id"]

    _, second_client, _ = _app(_cancel())
    assert second_client.get(
        f"/v1/runs/{run_id}",
        headers=_headers(),
    ).status_code == 404
    assert second_client.post(
        f"/v1/runs/{run_id}/confirmation",
        json={"decision": "approve", "expected_version": 1},
        headers=_headers(key="idem-key-0000000002"),
    ).status_code == 404


def test_unexpected_exception_is_not_a_policy_or_failed_closed_outcome():
    app, client, _ = _app(_cancel())
    service = app.state.service

    def explode(*args, **kwargs):
        raise RuntimeError("private internal detail")

    service.orchestrator.start = explode
    response = client.post(
        "/v1/runs",
        json={"message": "Cancel."},
        headers=_headers(),
    )
    assert response.status_code == 500
    error = response.json()["error"]
    assert error["code"] == "execution_outcome_indeterminate"
    assert "private internal detail" not in json.dumps(error)
    run_id = error["details"]["run_id"]
    fetched = client.get(
        f"/v1/runs/{run_id}",
        headers=_headers(),
    )
    assert fetched.status_code == 200
    assert fetched.json()["execution_state"] == "indeterminate"
    assert fetched.json()["status"] is None


def test_frozen_failed_closed_is_surfaced_authoritatively():
    app, client, _ = _app(_cancel())
    service = app.state.service

    def failed_closed(*args, **kwargs):
        return RunResult(
            thread_id="internal-thread",
            status=RunStatus.FAILED_CLOSED,
            response="Execution could not continue.",
        )

    service.orchestrator.start = failed_closed
    response = client.post(
        "/v1/runs",
        json={"message": "Cancel."},
        headers=_headers(),
    )
    assert response.status_code == 201
    assert response.json()["status"] == "failed_closed"
    assert response.json()["result"]["response"] == (
        "Execution could not continue."
    )
