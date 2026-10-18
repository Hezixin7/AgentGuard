"""Actual incoming-byte request-size enforcement for the deployment app."""

from __future__ import annotations

import json
from typing import Any, Awaitable, Callable
from uuid import uuid4

REQUEST_BODY_LIMIT_BYTES = 16_384
_LIMITED_METHODS = {"POST", "PUT", "PATCH"}


class BodySizeLimitMiddleware:
    """Buffer request bytes up to a limit before invoking the ASGI app.

    Unlike a Content-Length check, this counts every ``http.request`` chunk,
    so requests without Content-Length and chunked/streamed bodies are covered.
    The wrapped app is never invoked for an oversized request.
    """

    def __init__(
        self,
        app: Callable[..., Awaitable[None]],
        *,
        max_bytes: int = REQUEST_BODY_LIMIT_BYTES,
    ) -> None:
        if isinstance(max_bytes, bool) or not isinstance(max_bytes, int):
            raise ValueError("max_bytes must be an integer")
        if max_bytes <= 0:
            raise ValueError("max_bytes must be positive")
        self.app = app
        self.max_bytes = max_bytes

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[[], Awaitable[dict[str, Any]]],
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        if (
            scope.get("type") != "http"
            or scope.get("method") not in _LIMITED_METHODS
        ):
            await self.app(scope, receive, send)
            return

        body = bytearray()
        while True:
            message = await receive()
            if message.get("type") == "http.disconnect":
                return
            if message.get("type") != "http.request":
                continue
            chunk = message.get("body", b"")
            if not isinstance(chunk, (bytes, bytearray)):
                chunk = b""
            if len(body) + len(chunk) > self.max_bytes:
                await self._send_413(send)
                return
            body.extend(chunk)
            if not message.get("more_body", False):
                break

        delivered = False

        async def replay_receive() -> dict[str, Any]:
            nonlocal delivered
            if not delivered:
                delivered = True
                return {
                    "type": "http.request",
                    "body": bytes(body),
                    "more_body": False,
                }
            return await receive()

        await self.app(scope, replay_receive, send)

    @staticmethod
    async def _send_413(
        send: Callable[[dict[str, Any]], Awaitable[None]],
    ) -> None:
        request_id = uuid4().hex
        payload = json.dumps(
            {
                "error": {
                    "code": "request_too_large",
                    "message": "Request body is too large.",
                    "request_id": request_id,
                    "details": None,
                }
            },
            ensure_ascii=True,
            separators=(",", ":"),
        ).encode("utf-8")
        await send(
            {
                "type": "http.response.start",
                "status": 413,
                "headers": [
                    (b"content-type", b"application/json"),
                    (b"content-length", str(len(payload)).encode("ascii")),
                    (b"x-request-id", request_id.encode("ascii")),
                ],
            }
        )
        await send(
            {
                "type": "http.response.body",
                "body": payload,
                "more_body": False,
            }
        )
