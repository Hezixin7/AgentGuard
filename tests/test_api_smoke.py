import json
import socket
import threading
import time

import httpx
import uvicorn

from deployment.api import create_app

API_KEYS = {"smoke-bearer-token": "user-1"}


def _free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _scripted_model():
    def model(messages):
        if any(message.get("role") == "tool" for message in messages):
            return {"type": "final", "response": "Completed."}
        return {
            "type": "tool_call",
            "operation": "cancel_order",
            "order_id": "P100",
        }

    return model


def _auth(key):
    return {
        "Authorization": "Bearer smoke-bearer-token",
        "Idempotency-Key": key,
    }


def test_local_uvicorn_http_smoke():
    app = create_app(model=_scripted_model(), api_keys=API_KEYS)
    port = _free_port()
    config = uvicorn.Config(
        app=app,
        host="127.0.0.1",
        port=port,
        log_level="warning",
        access_log=False,
        workers=1,
    )
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    try:
        deadline = time.monotonic() + 10
        while not server.started:
            if time.monotonic() > deadline:
                raise AssertionError("Uvicorn did not start")
            if not thread.is_alive():
                raise AssertionError("Uvicorn exited during startup")
            time.sleep(0.01)

        with httpx.Client(
            base_url=f"http://127.0.0.1:{port}",
            timeout=5.0,
        ) as client:
            health = client.get("/healthz")
            assert health.status_code == 200
            assert health.json() == {"status": "ok"}

            unauthenticated = client.post(
                "/v1/runs",
                json={"message": "Cancel P100."},
                headers={"Idempotency-Key": "smoke-idem-key-0001"},
            )
            assert unauthenticated.status_code == 401

            created = client.post(
                "/v1/runs",
                json={"message": "Cancel P100."},
                headers=_auth("smoke-idem-key-0001"),
            )
            assert created.status_code == 201
            body = created.json()
            assert body["status"] == "awaiting_confirmation"
            run_id = body["run_id"]
            version = body["version"]

            confirmed = client.post(
                f"/v1/runs/{run_id}/confirmation",
                json={
                    "decision": "approve",
                    "expected_version": version,
                },
                headers=_auth("smoke-idem-key-0002"),
            )
            assert confirmed.status_code == 200
            assert confirmed.json()["status"] == "completed"
            assert confirmed.json()["result"]["response"] == "Completed."

            trace = client.get(
                f"/v1/runs/{run_id}/trace",
                headers=_auth("smoke-idem-key-0003"),
            )
            assert trace.status_code == 200
            payload = trace.json()
            assert payload["trace"]["run_id"] == run_id
            serialized = json.dumps(payload, sort_keys=True)
            assert "user-1" not in serialized
            assert "smoke-bearer-token" not in serialized
    finally:
        server.should_exit = True
        thread.join(timeout=5)
        assert not thread.is_alive()
