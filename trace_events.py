"""AgentGuard V1 structured audit-event contracts.

These are data contracts, not a recorder, policy engine, or security boundary.
Never treat LLM-reported argument provenance as authorization evidence.
Exporters must redact personal data, secrets, and confirmation tokens.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class EventType(str, Enum):
    LLM_OUTPUT = "llm_output"
    TOOL_CALL = "tool_call"
    POLICY_DECISION = "policy_decision"
    HUMAN_CONFIRMATION = "human_confirmation"
    TOOL_RESULT = "tool_result"
    FINAL_RESPONSE = "final_response"


class PolicyOutcome(str, Enum):
    ALLOWED = "allowed"
    BLOCKED = "blocked"
    REPLAY = "replay"


class ConfirmationStatus(str, Enum):
    REQUESTED = "requested"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    EXPIRED = "expired"
    CONSUMED = "consumed"


class ToolStatus(str, Enum):
    SUCCESS = "success"
    ERROR = "error"
    TIMEOUT = "timeout"
    REPLAYED = "replayed"


@dataclass(kw_only=True)
class TraceEvent:
    """Common metadata; TraceRecorder assigns trace_id and monotonic step."""
    event_type: EventType
    trace_id: str = ""
    run_id: str = ""  # evaluation run / experiment batch identifier
    step: int = 0
    timestamp: str = field(default_factory=utc_now_iso)


@dataclass(kw_only=True)
class LLMOutputEvent(TraceEvent):
    event_type: EventType = field(default=EventType.LLM_OUTPUT, init=False)
    raw_output: str = ""  # redact before persistence
    parsed_plan: Optional[dict[str, Any]] = None
    model: str = ""
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


@dataclass(kw_only=True)
class ToolCallEvent(TraceEvent):
    event_type: EventType = field(default=EventType.TOOL_CALL, init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    args_source: str = "unknown"  # user_text | system_query | session | unknown
    # For diagnosis ONLY; never an input to authorization.


@dataclass(kw_only=True)
class PolicyDecisionEvent(TraceEvent):
    event_type: EventType = field(default=EventType.POLICY_DECISION, init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    decision: PolicyOutcome = PolicyOutcome.BLOCKED
    reason_code: str = ""  # stable machine-readable code
    reason: str = ""  # safe human-readable explanation
    checked_user_id: str = ""  # pseudonymize in exported traces
    checked_owner_id: str = ""
    idempotency_key_ref: str = ""  # hash/reference, not raw secret


@dataclass(kw_only=True)
class HumanConfirmationEvent(TraceEvent):
    event_type: EventType = field(default=EventType.HUMAN_CONFIRMATION, init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    args: dict[str, Any] = field(default_factory=dict)
    authenticated_user_id: str = ""
    confirmation_id: str = ""  # reference only; never log bearer token
    token_expires_at: str = ""
    order_state_hash: str = ""
    status: ConfirmationStatus = ConfirmationStatus.REQUESTED
    # Backend verifies identity, operation+args, expiry, snapshot, and one-time use.


@dataclass(kw_only=True)
class ToolResultEvent(TraceEvent):
    event_type: EventType = field(default=EventType.TOOL_RESULT, init=False)
    tool_call_id: str = ""
    tool_name: str = ""
    status: ToolStatus = ToolStatus.SUCCESS
    result: Any = None  # redact before persistence
    error_type: str = ""
    idempotency_key_ref: str = ""
    replayed: bool = False
    state_changed: bool = False
    duration_ms: int = 0


@dataclass(kw_only=True)
class FinalResponseEvent(TraceEvent):
    event_type: EventType = field(default=EventType.FINAL_RESPONSE, init=False)
    response: str = ""  # redact before persistence
    cited_tool_calls: list[str] = field(default_factory=list)


# Contract notes:
# - Every event from one Agent execution shares trace_id; step increments in
#   emission order. Each attempted tool call gets its own tool_call_id, reused
#   by its policy, confirmation, and result events.
# - "replay" is a policy outcome; "replayed" is a tool-result status. A replay
#   must never cause another business side effect.
# - Confirmation is a backend-controlled, single-use capability; never log its
#   raw token. State snapshots are not substitutes for live policy re-checks.
# - Events are observations, not proof of security; tests also inspect DB state.
