"""Owner-facing redacted trace projection.

Uses the frozen TraceRecorder export so the deployment layer inherits the
existing redaction policy instead of duplicating it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
from typing import Any

from deployment.errors import ApiError


def export_redacted_trace(orchestrator: Any, thread_id: str) -> dict[str, Any]:
    if not thread_id:
        raise _trace_unavailable()
    descriptor, path_value = tempfile.mkstemp(
        prefix="agentguard-trace-",
        suffix=".json",
    )
    os.close(descriptor)
    path = Path(path_value)
    try:
        orchestrator.export_trace(thread_id, str(path))
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        raise _trace_unavailable() from None
    finally:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    if not isinstance(payload, dict):
        raise _trace_unavailable()
    if not isinstance(payload.get("schema_version"), int):
        raise _trace_unavailable()
    trace = payload.get("trace")
    events = payload.get("events")
    if not isinstance(trace, dict) or not isinstance(events, list):
        raise _trace_unavailable()
    return payload


def _trace_unavailable() -> ApiError:
    return ApiError(
        409,
        "trace_unavailable",
        "Trace is not available for this run.",
    )
