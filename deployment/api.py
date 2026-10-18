"""Minimal FastAPI adapter for AgentGuard V1.

Single process, in-memory state, AgentGuard only. No evaluation controls,
databases, queues, or multi-turn message channel are exposed.
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Mapping
from uuid import uuid4

from fastapi import Depends, FastAPI, Header, Request, Response
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

from agent_graph import AgentOrchestrator
from simulated_tools import InMemoryOrderState

from deployment.auth import Principal, StaticBearerAuthenticator
from deployment.body_limit import (
    REQUEST_BODY_LIMIT_BYTES,
    BodySizeLimitMiddleware,
)
from deployment.errors import ApiError
from deployment.registry import DeploymentService
from deployment.schemas import (
    ConfirmationRequest,
    CreateRunRequest,
    ErrorDetail,
    ErrorEnvelope,
    RunView,
    TraceResponse,
)

logger = logging.getLogger(__name__)


def create_app(
    *,
    model: Callable[[list[dict[str, str]]], dict[str, Any]],
    api_keys: Mapping[str, str],
    state: InMemoryOrderState | None = None,
    clock: Callable[[], float] | None = None,
) -> FastAPI:
    if not callable(model):
        raise TypeError("model must be callable")
    authenticator = StaticBearerAuthenticator(api_keys)
    orchestrator = AgentOrchestrator(
        state if state is not None else InMemoryOrderState(),
        model,
        clock=clock,
    )
    service = DeploymentService(orchestrator, authenticator)

    app = FastAPI(title="AgentGuard V1", version="1.0.0")
    app.state.service = service
    app.state.authenticator = authenticator
    app.add_middleware(
        BodySizeLimitMiddleware,
        max_bytes=REQUEST_BODY_LIMIT_BYTES,
    )

    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next: Any):
        request.state.request_id = uuid4().hex
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    def get_principal(
        request: Request,
        authorization: str | None = Header(default=None, alias="Authorization"),
    ) -> Principal:
        return request.app.state.authenticator.authenticate(authorization)

    @app.post(
        "/v1/runs",
        response_model=RunView,
        status_code=201,
    )
    def create_run(
        payload: CreateRunRequest,
        response: Response,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        principal: Principal = Depends(get_principal),
    ):
        result = app.state.service.create_run(
            principal,
            idempotency_key,
            payload.message,
        )
        if result.replayed:
            response.headers["Idempotent-Replay"] = "true"
        return result.body

    @app.get("/v1/runs/{run_id}", response_model=RunView)
    def get_run(
        run_id: str,
        principal: Principal = Depends(get_principal),
    ):
        result = app.state.service.get_run(principal, run_id)
        return result.body

    @app.post(
        "/v1/runs/{run_id}/confirmation",
        response_model=RunView,
    )
    def confirm_run(
        run_id: str,
        payload: ConfirmationRequest,
        response: Response,
        idempotency_key: str = Header(alias="Idempotency-Key"),
        principal: Principal = Depends(get_principal),
    ):
        result = app.state.service.confirm_run(
            principal,
            run_id,
            idempotency_key,
            payload.decision,
            payload.expected_version,
        )
        if result.replayed:
            response.headers["Idempotent-Replay"] = "true"
        return result.body

    @app.get(
        "/v1/runs/{run_id}/trace",
        response_model=TraceResponse,
    )
    def get_trace(
        run_id: str,
        principal: Principal = Depends(get_principal),
    ):
        result = app.state.service.get_trace(principal, run_id)
        return result.body

    @app.get("/healthz")
    def healthz():
        return {"status": "ok"}

    @app.exception_handler(ApiError)
    async def handle_api_error(request: Request, exc: ApiError):
        return _error_response(
            request,
            exc.status_code,
            exc.code,
            exc.message,
            exc.details,
        )

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request,
        exc: StarletteHTTPException,
    ):
        if exc.status_code == 404:
            code = "not_found"
            message = "Resource was not found."
        elif exc.status_code == 405:
            code = "method_not_allowed"
            message = "Method is not allowed."
        else:
            code = "http_error"
            message = "The request could not be processed."
        return _error_response(
            request,
            exc.status_code,
            code,
            message,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request,
        exc: RequestValidationError,
    ):
        errors = [
            {
                "loc": ".".join(str(part) for part in error.get("loc", ())),
                "type": error.get("type", "value_error"),
                "msg": error.get("msg", "Invalid value"),
            }
            for error in exc.errors()
        ]
        return _error_response(
            request,
            422,
            "invalid_request",
            "Request validation failed.",
            {"errors": errors},
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(request: Request, exc: Exception):
        logger.exception(
            "Unhandled API failure request_id=%s",
            getattr(request.state, "request_id", "unknown"),
        )
        return _error_response(
            request,
            500,
            "internal_error",
            "The request could not be processed.",
        )

    return app


def _error_response(
    request: Request,
    status_code: int,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> JSONResponse:
    request_id = getattr(request.state, "request_id", None) or uuid4().hex
    envelope = ErrorEnvelope(
        error=ErrorDetail(
            code=code,
            message=message,
            request_id=request_id,
            details=details,
        )
    )
    return JSONResponse(
        status_code=status_code,
        content=envelope.model_dump(),
        headers={"X-Request-ID": request_id},
    )
