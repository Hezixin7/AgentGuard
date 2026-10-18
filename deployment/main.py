"""ASGI entrypoint for the AgentGuard V1 API.

Start locally with one process and one worker:

    DEEPSEEK_API_KEY=... \
    AGENTGUARD_API_KEYS='{"replace-with-a-strong-bearer-token":"user-1"}' \
    uvicorn deployment.main:app --host 127.0.0.1 --port 8000 --workers 1

Missing or malformed environment configuration fails at import/startup with a
sanitized error. Secrets are never printed by this module.
"""

from deployment.config import build_app_from_env

app = build_app_from_env()
