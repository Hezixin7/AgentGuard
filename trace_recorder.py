"""In-memory recording and redacted JSON export for AgentGuard trace events."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import fields, is_dataclass
from enum import Enum
import json
from pathlib import Path
from typing import Any
from uuid import uuid4

from trace_events import (
    HumanConfirmationEvent,
    PolicyDecisionEvent,
    ToolCallEvent,
    ToolResultEvent,
    TraceEvent,
)

_REDACTED = "[REDACTED]"
_SENSITIVE_KEYS = {
    "accesskey",
    "accesssecret",
    "address",
    "apikey",
    "auth",
    "authentication",
    "authorization",
    "bearertoken",
    "capability",
    "checkedownerid",
    "checkeduserid",
    "confirmationcapability",
    "confirmationid",
    "confirmationtoken",
    "credential",
    "email",
    "fullname",
    "name",
    "ownerid",
    "password",
    "phone",
    "refreshtoken",
    "secret",
    "token",
    "userid",
    "authenticateduserid",
}
_SENSITIVE_SUFFIXES = (
    "accesskey",
    "address",
    "apikey",
    "authorization",
    "bearertoken",
    "capability",
    "confirmationid",
    "confirmationtoken",
    "credential",
    "email",
    "name",
    "password",
    "phone",
    "refreshtoken",
    "secret",
    "token",
    "userid",
)
_FREE_TEXT_FIELDS = {"rawoutput", "response"}
_AUDIT_FIELDS = {
    "eventtype",
    "reasoncode",
    "runid",
    "status",
    "step",
    "toolcallid",
    "toolname",
    "traceid",
}


class TraceRecorder:
    """Record one execution's events and export a redacted JSON envelope."""

    def __init__(
        self,
        *,
        scenario_id: str,
        run_id: str,
        trace_id: str | None = None,
    ) -> None:
        if not scenario_id:
            raise ValueError("scenario_id must not be empty")
        if not run_id:
            raise ValueError("run_id must not be empty")

        self.scenario_id = scenario_id
        self.run_id = run_id
        self.trace_id = trace_id or str(uuid4())
        self._events: list[TraceEvent] = []
        self._tool_call_ids: set[str] = set()

    def record(self, event: TraceEvent) -> TraceEvent:
        """Record a deep-copied event with recorder-assigned metadata."""
        if not isinstance(event, TraceEvent):
            raise TypeError("event must be a TraceEvent")
        if event.trace_id and event.trace_id != self.trace_id:
            raise ValueError("event trace_id conflicts with this recorder")
        if event.run_id and event.run_id != self.run_id:
            raise ValueError("event run_id conflicts with this recorder")

        recorded = deepcopy(event)
        self._validate_correlation(recorded)
        recorded.trace_id = self.trace_id
        recorded.run_id = self.run_id
        recorded.step = len(self._events) + 1
        self._events.append(recorded)
        if isinstance(recorded, ToolCallEvent):
            self._tool_call_ids.add(recorded.tool_call_id)
        return deepcopy(recorded)

    def get_events(self) -> list[TraceEvent]:
        """Return deep copies so callers cannot mutate recorder state."""
        return deepcopy(self._events)

    def export_json(self, path: str | Path) -> None:
        """Write the trace envelope as UTF-8 JSON with sensitive data redacted."""
        payload = {
            "schema_version": 1,
            "trace": {
                "trace_id": self.trace_id,
                "run_id": self.run_id,
                "scenario_id": self.scenario_id,
            },
            "events": [_event_to_dict(event) for event in self._events],
        }
        redacted = _redact(payload)
        content = json.dumps(
            redacted,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        Path(path).write_text(content + "\n", encoding="utf-8")

    def _validate_correlation(self, event: TraceEvent) -> None:
        if isinstance(event, ToolCallEvent):
            if not event.tool_call_id:
                raise ValueError("ToolCallEvent.tool_call_id must not be empty")
            if event.tool_call_id in self._tool_call_ids:
                raise ValueError("each ToolCallEvent must have a unique tool_call_id")
        elif isinstance(
            event,
            (PolicyDecisionEvent, HumanConfirmationEvent, ToolResultEvent),
        ):
            if not event.tool_call_id:
                raise ValueError(f"{type(event).__name__}.tool_call_id must not be empty")
            if event.tool_call_id not in self._tool_call_ids:
                raise ValueError(
                    f"{type(event).__name__} refers to an unknown tool_call_id"
                )


def _event_to_dict(event: TraceEvent) -> dict[str, Any]:
    return {
        item.name: _to_json_value(getattr(event, item.name))
        for item in fields(event)
    }


def _to_json_value(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value) and not isinstance(value, type):
        return {
            item.name: _to_json_value(getattr(value, item.name))
            for item in fields(value)
        }
    if isinstance(value, dict):
        if any(not isinstance(key, str) for key in value):
            raise TypeError("trace JSON objects must have string keys")
        return {key: _to_json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_to_json_value(item) for item in value]
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise TypeError(f"value of type {type(value).__name__} is not JSON-serializable")


def _redact(value: Any, *, field_name: str = "") -> Any:
    normalized_name = "".join(char for char in field_name.lower() if char.isalnum())
    if normalized_name in _AUDIT_FIELDS:
        return value
    if normalized_name in _FREE_TEXT_FIELDS:
        return _REDACTED
    if normalized_name in _SENSITIVE_KEYS or normalized_name.endswith(
        _SENSITIVE_SUFFIXES
    ):
        return _REDACTED
    if normalized_name == "result" and isinstance(value, str):
        return _REDACTED
    if isinstance(value, dict):
        return {
            key: _redact(item, field_name=key)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact(item) for item in value]
    return value
