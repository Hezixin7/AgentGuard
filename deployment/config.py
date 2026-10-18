"""Server-side environment configuration for the deployment API."""

from __future__ import annotations

import json
import os
from typing import Any, Mapping

from fastapi import FastAPI

from deployment.api import create_app
from deployment.deepseek_adapter import (
    DeepSeekModelAdapter,
    ModelConfigurationError,
)


class StartupConfigurationError(RuntimeError):
    """Sanitized startup configuration failure."""


def load_api_keys(raw: str | None) -> dict[str, str]:
    """Parse AGENTGUARD_API_KEYS as a bearer-token-to-user-ID JSON object."""
    if not isinstance(raw, str) or not raw.strip():
        raise StartupConfigurationError(
            "AGENTGUARD_API_KEYS is required as a JSON object mapping "
            "bearer tokens to user IDs."
        )
    try:
        value = json.loads(
            raw,
            object_pairs_hook=_reject_duplicate_keys,
        )
    except (json.JSONDecodeError, TypeError, ValueError):
        raise StartupConfigurationError(
            "AGENTGUARD_API_KEYS must be a valid JSON object."
        ) from None
    if not isinstance(value, dict) or not value:
        raise StartupConfigurationError(
            "AGENTGUARD_API_KEYS must be a non-empty JSON object."
        )
    for token, user_id in value.items():
        if (
            not isinstance(token, str)
            or not token
            or not isinstance(user_id, str)
            or not user_id
        ):
            raise StartupConfigurationError(
                "AGENTGUARD_API_KEYS entries must map non-empty bearer "
                "tokens to non-empty user IDs."
            )
    return dict(value)


def build_app_from_env(
    environ: Mapping[str, str] | None = None,
) -> FastAPI:
    """Build the app from server-side environment configuration."""
    env = os.environ if environ is None else environ
    api_keys = load_api_keys(env.get("AGENTGUARD_API_KEYS"))
    deepseek_key = env.get("DEEPSEEK_API_KEY")
    if not isinstance(deepseek_key, str) or not deepseek_key.strip():
        raise StartupConfigurationError(
            "DEEPSEEK_API_KEY is required for the DeepSeek model adapter."
        )
    try:
        model = DeepSeekModelAdapter(api_key=deepseek_key)
    except ModelConfigurationError as error:
        raise StartupConfigurationError(str(error)) from None
    return create_app(model=model, api_keys=api_keys)


def _reject_duplicate_keys(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result
