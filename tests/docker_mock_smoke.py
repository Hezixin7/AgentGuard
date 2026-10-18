"""Deterministic mocked-model harness for Docker smoke verification.

Run inside the runtime image with this file bind-mounted, for example:

    docker run ... agentguard:v1 python /tmp/docker_mock_smoke.py

It uses the production create_app path but injects a deterministic model. It
makes no DeepSeek calls and does not require DEEPSEEK_API_KEY.
"""

from __future__ import annotations

import os

import uvicorn

from deployment.api import create_app

API_KEYS = {
    "smoke-token-user-1": "user-1",
    "smoke-token-user-2": "user-2",
}


def _model(messages):
    del messages
    return {"type": "final", "response": "Mock run completed."}


app = create_app(model=_model, api_keys=API_KEYS)


if __name__ == "__main__":
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=int(os.environ.get("SMOKE_PORT", "8001")),
        log_level="warning",
    )
