"""Strict request and response schemas for the deployment API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

RunStatusValue = Literal[
    "awaiting_confirmation",
    "completed",
    "failed_closed",
]


class CreateRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    message: str = Field(min_length=1, max_length=4000)

    @field_validator("message")
    @classmethod
    def validate_message(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("message must not be empty")
        if "\x00" in cleaned:
            raise ValueError("message must not contain NUL")
        return cleaned


class ConfirmationRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    decision: Literal["approve", "reject"]
    expected_version: int = Field(ge=0)


class ConfirmationView(BaseModel):
    operation: str
    order_id: str
    parameters: dict[str, str]
    expires_at: str


class ResultView(BaseModel):
    response: str


class RunView(BaseModel):
    run_id: str
    status: RunStatusValue | None
    execution_state: Literal["normal", "indeterminate"]
    version: int
    confirmation: ConfirmationView | None = None
    result: ResultView | None = None


class TraceMeta(BaseModel):
    trace_id: str
    run_id: str
    scenario_id: str


class TraceResponse(BaseModel):
    schema_version: int
    trace: TraceMeta
    events: list[dict[str, Any]]


class ErrorDetail(BaseModel):
    code: str
    message: str
    request_id: str
    details: dict[str, Any] | None = None


class ErrorEnvelope(BaseModel):
    error: ErrorDetail
