"""Static server-configured bearer authentication."""

from __future__ import annotations

from dataclasses import dataclass
import hmac
from typing import Mapping

from deployment.errors import ApiError


@dataclass(frozen=True)
class Principal:
    """Trusted identity established by the API authentication layer."""

    user_id: str


class StaticBearerAuthenticator:
    """Map opaque bearer tokens to trusted user IDs.

    Tokens are configured server-side and never accepted in request bodies.
    """

    def __init__(self, api_keys: Mapping[str, str]) -> None:
        if not api_keys:
            raise ValueError("at least one API key is required")
        for token, user_id in api_keys.items():
            if not isinstance(token, str) or not token:
                raise ValueError("API keys must be non-empty strings")
            if not isinstance(user_id, str) or not user_id:
                raise ValueError("API key principals must be non-empty strings")
        self._api_keys = dict(api_keys)

    def authenticate(self, authorization: str | None) -> Principal:
        if not isinstance(authorization, str):
            raise _unauthenticated()
        parts = authorization.split()
        if len(parts) != 2 or parts[0].lower() != "bearer":
            raise _unauthenticated()
        token = parts[1]
        matched: str | None = None
        for configured, user_id in self._api_keys.items():
            if hmac.compare_digest(token, configured):
                matched = user_id
        if matched is None:
            raise _unauthenticated()
        return Principal(user_id=matched)


def _unauthenticated() -> ApiError:
    return ApiError(
        401,
        "unauthenticated",
        "Authentication is required.",
    )
