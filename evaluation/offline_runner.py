"""Deterministic single-scenario execution for the offline evaluation corpus."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from decimal import Decimal
import hashlib
import json
import re
from typing import Any

from agent_graph import AgentOrchestrator, ExperimentGroup, RunResult, RunStatus
from policy_gate import HumanDecision, PolicyOutcome, SessionContext
from simulated_tools import (
    FaultMode,
    InMemoryOrderState,
    Order,
    OrderStatus,
    SimulatedOrderTools,
    ToolError,
    ToolResult,
)
from trace_events import (
    FinalResponseEvent,
    HumanConfirmationEvent,
    LLMOutputEvent,
    PolicyDecisionEvent,
    ToolCallEvent,
    ToolResultEvent,
    TraceEvent,
)

_SAFE_FAILURE_DETAIL = "Unexpected execution failure; exception message omitted."
_SAFE_TRACE_CAPTURE_FAILURE = (
    "AgentGuard trace capture failed; event payload unavailable."
)
_TRACE_CAPTURE_SCHEMA_VERSION = 1
_TRACE_SOURCE_SCHEMA_VERSION = 1
_TRACE_EVENT_TYPES = {
    "llm_output",
    "tool_call",
    "policy_decision",
    "human_confirmation",
    "tool_result",
    "final_response",
}
_SAFE_TRACE_REASON_CODES = {
    "already_refunded",
    "authorized",
    "authorized_execution_failed",
    "authentication_required",
    "confirmation_expired",
    "confirmation_invalid",
    "confirmation_mismatch",
    "confirmation_not_approved",
    "confirmation_rejected",
    "confirmation_required",
    "confirmation_reused",
    "confirmation_state_changed",
    "idempotency_conflict",
    "idempotent_replay",
    "invalid_operation",
    "invalid_request",
    "invalid_transition",
    "order_unavailable",
}
_SAFE_TRACE_ERROR_TYPES = {
    *_SAFE_TRACE_REASON_CODES,
    "timeout_before_commit",
    "timeout_after_commit",
}


@dataclass(frozen=True)
class ToolInvocation:
    operation: str
    order_id: str
    outcome: str
    fault: str | None = None
    error_code: str | None = None
    committed: bool = False
    idempotency_key_ref: str | None = None
    response_delivery_count: int = 1


@dataclass(frozen=True)
class AgentGuardTraceCapture:
    schema_version: int
    source_schema_version: int
    capture_status: str
    events: tuple[dict[str, Any], ...]
    unavailable_reason: str | None = None


@dataclass(frozen=True)
class EvaluationRunResult:
    scenario_id: str
    group: ExperimentGroup
    run_id: str
    status: RunStatus
    response_class: str
    sanitized_response: str
    tool_attempts: int
    attempted_tool_calls: tuple[dict[str, str], ...]
    tool_invocations: int
    agent_state_mutations: int
    external_state_mutations: int
    order_states: dict[str, dict[str, str]]
    refunds: tuple[dict[str, Any], ...]
    idempotency_records: tuple[dict[str, Any], ...]
    policy_decisions: tuple[dict[str, str], ...]
    confirmation_statuses: tuple[str, ...]
    tool_results: tuple[ToolInvocation, ...]
    trace_event_count: int
    baseline_a_timing_mismatch: bool
    failure: dict[str, str] | None = None
    trace_capture: AgentGuardTraceCapture | None = None


class EvaluationRunError(RuntimeError):
    """Unexpected execution failure with the observations collected so far."""

    def __init__(
        self,
        scenario_id: str,
        group: ExperimentGroup,
        failure_type: str,
        detail: str,
        partial_result: EvaluationRunResult,
    ) -> None:
        super().__init__(
            f"{scenario_id}/{group.value} failed with "
            f"{failure_type}: {detail}"
        )
        self.partial_result = partial_result
        self.failure_type = failure_type
        self.detail = detail


@dataclass
class _Observation:
    attempts: list[dict[str, Any]] = field(default_factory=list)
    invocations: list[ToolInvocation] = field(default_factory=list)
    policy_decisions: list[dict[str, str]] = field(default_factory=list)
    confirmation_statuses: list[str] = field(default_factory=list)
    agent_state_mutations: int = 0
    external_state_mutations: int = 0
    trace_event_count: int = 0


class _FakeClock:
    def __init__(self) -> None:
        self.value = 100.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


class _TraceProjectionError(RuntimeError):
    """Signals that a successful AgentGuard run cannot safely export its trace."""


class _ScriptedProposal:
    def __init__(self, proposal: dict[str, Any]) -> None:
        self._proposal = dict(proposal)
        self._sent = False

    def __call__(self, messages: list[dict[str, str]]) -> dict[str, Any]:
        if not self._sent:
            self._sent = True
            return dict(self._proposal)
        if messages and messages[-1].get("role") == "tool":
            try:
                result = json.loads(messages[-1]["content"])
            except (KeyError, json.JSONDecodeError):
                result = {}
            if "error" in result:
                response = "The requested operation could not be completed."
            elif result.get("outcome") == PolicyOutcome.BLOCKED.value:
                response = "I cannot complete that request."
            else:
                response = "The requested operation completed."
            return {"type": "final", "response": response}
        return {"type": "final", "response": "I could not complete that request."}


def run_scenario(
    scenario: Any,
    group: ExperimentGroup,
) -> EvaluationRunResult:
    """Run one validated scenario once in one experiment group.

    Scenario inputs may be a raw scenario mapping or a Stage 1/2 wrapper with
    a ``data`` mapping. Expected-oracle fields are deliberately not read.
    """
    data = _scenario_data(scenario)
    scenario_id = data["id"]
    if not isinstance(group, ExperimentGroup):
        raise TypeError("group must be an ExperimentGroup")
    _validate_runtime_fixture(data)
    run_id = f"{scenario_id}:{group.value}"
    observation = _Observation()
    clock = _FakeClock()
    state = _fresh_state(data)
    state_order_ids = tuple(item["order_id"] for item in data["initial_orders"])
    fixture_keys = {
        *(seed["idempotency_key"] for seed in data["initial_idempotency"]),
        *(
            [data["trusted_idempotency_key"]]
            if "trusted_idempotency_key" in data
            else []
        ),
    }
    sensitive_values = _collect_sensitive_values(data)
    tools = SimulatedOrderTools(state)
    _seed_idempotency_records(data, tools)
    orchestrator = (
        AgentOrchestrator.for_evaluation(
            state,
            _ScriptedProposal(data["proposals"][0]),
            clock=clock,
        )
        if group is ExperimentGroup.BASELINE_A
        else AgentOrchestrator(
            state,
            _ScriptedProposal(data["proposals"][0]),
            group=group,
            clock=clock,
        )
    )
    _install_attempt_observer(orchestrator, data, observation, group)
    _install_tool_observer(
        orchestrator._tools,
        state,
        state_order_ids,
        data,
        observation,
        fixture_keys,
    )
    if orchestrator._gate is not None:
        _install_gate_observer(orchestrator._gate, observation)

    baseline_a_mismatch = (
        group is ExperimentGroup.BASELINE_A
        and bool(data["pre_resume_state_changes"])
    )
    run_error: EvaluationRunError | None = None
    trace_capture: AgentGuardTraceCapture | None = None
    try:
        if baseline_a_mismatch:
            _apply_external_changes(data, state, observation)

        result = orchestrator.start(
            data["user_input"],
            SessionContext(data["user_id"]),
            scenario_id=scenario_id,
            run_id=run_id,
        )

        if group is not ExperimentGroup.BASELINE_A:
            if data["pre_resume_state_changes"]:
                _apply_external_changes(data, state, observation)
            if result.status is RunStatus.PAUSED:
                clock.advance(float(data["clock_advance_seconds"]))
                for decision_value in data["human_decisions"]:
                    result = _submit_fixture_decision(
                        orchestrator,
                        result,
                        data["user_id"],
                        decision_value,
                        observation,
                    )

        if group is ExperimentGroup.AGENTGUARD:
            trace_capture = _capture_agentguard_trace_safely(
                orchestrator,
                run_id,
                data,
                result.thread_id,
            )
            if trace_capture.capture_status != "captured":
                raise _TraceProjectionError(_SAFE_TRACE_CAPTURE_FAILURE)
            observation.trace_event_count = len(trace_capture.events)
        return _build_result(
            scenario_id,
            group,
            run_id,
            result,
            state,
            observation,
            state_order_ids,
            fixture_keys,
            baseline_a_mismatch,
            sensitive_values=sensitive_values,
            trace_capture=trace_capture,
        )
    except Exception as error:
        if group is ExperimentGroup.AGENTGUARD and trace_capture is None:
            trace_capture = _capture_agentguard_trace_safely(
                orchestrator,
                run_id,
                data,
                getattr(locals().get("result"), "thread_id", None),
            )
            if trace_capture.capture_status == "captured":
                observation.trace_event_count = len(trace_capture.events)

        failure_type = (
            "TraceCaptureError"
            if isinstance(error, _TraceProjectionError)
            else type(error).__name__
        )
        failure = {
            "type": failure_type,
            "detail": _SAFE_TRACE_CAPTURE_FAILURE
            if isinstance(error, _TraceProjectionError)
            else _SAFE_FAILURE_DETAIL,
        }
        if (
            group is ExperimentGroup.AGENTGUARD
            and trace_capture is not None
            and trace_capture.capture_status in {"failed", "unavailable"}
        ):
            failure["trace_capture_failure"] = _SAFE_TRACE_CAPTURE_FAILURE
        partial_result = _build_result(
            scenario_id,
            group,
            run_id,
            locals().get("result"),
            state,
            observation,
            state_order_ids,
            fixture_keys,
            baseline_a_mismatch,
            failure=failure,
            sensitive_values=sensitive_values,
            trace_capture=trace_capture,
        )
        run_error = EvaluationRunError(
            scenario_id,
            group,
            failure_type,
            failure["detail"],
            partial_result,
        )
    assert run_error is not None
    raise run_error


def _capture_agentguard_trace_safely(
    orchestrator: AgentOrchestrator,
    run_id: str,
    scenario: dict[str, Any],
    thread_id: str | None,
) -> AgentGuardTraceCapture:
    try:
        return _capture_agentguard_trace(
            orchestrator,
            run_id,
            scenario,
            thread_id,
        )
    except Exception:
        return AgentGuardTraceCapture(
            schema_version=_TRACE_CAPTURE_SCHEMA_VERSION,
            source_schema_version=_TRACE_SOURCE_SCHEMA_VERSION,
            capture_status="failed",
            events=(),
            unavailable_reason=_SAFE_TRACE_CAPTURE_FAILURE,
        )


def _capture_agentguard_trace(
    orchestrator: AgentOrchestrator,
    run_id: str,
    scenario: dict[str, Any],
    thread_id: str | None,
) -> AgentGuardTraceCapture:
    if orchestrator._group is not ExperimentGroup.AGENTGUARD:
        raise ValueError("TraceRecorder capture is AgentGuard-only")

    if thread_id is None:
        thread_id = next(
            (
                candidate_id
                for candidate_id, execution in orchestrator._executions.items()
                if execution.run_id == run_id
            ),
            None,
        )
    if thread_id is None:
        return AgentGuardTraceCapture(
            schema_version=_TRACE_CAPTURE_SCHEMA_VERSION,
            source_schema_version=_TRACE_SOURCE_SCHEMA_VERSION,
            capture_status="unavailable",
            events=(),
            unavailable_reason="execution_recorder_not_initialized",
        )

    execution = orchestrator._executions.get(thread_id)
    if execution is None or execution.recorder is None:
        return AgentGuardTraceCapture(
            schema_version=_TRACE_CAPTURE_SCHEMA_VERSION,
            source_schema_version=_TRACE_SOURCE_SCHEMA_VERSION,
            capture_status="unavailable",
            events=(),
            unavailable_reason="execution_recorder_not_initialized",
        )

    try:
        source_events = orchestrator.trace_events(thread_id)
        projected = _project_trace_events(source_events, scenario)
    except Exception:
        return AgentGuardTraceCapture(
            schema_version=_TRACE_CAPTURE_SCHEMA_VERSION,
            source_schema_version=_TRACE_SOURCE_SCHEMA_VERSION,
            capture_status="failed",
            events=(),
            unavailable_reason=_SAFE_TRACE_CAPTURE_FAILURE,
        )

    return AgentGuardTraceCapture(
        schema_version=_TRACE_CAPTURE_SCHEMA_VERSION,
        source_schema_version=_TRACE_SOURCE_SCHEMA_VERSION,
        capture_status="captured",
        events=projected,
    )


def _project_trace_events(
    events: list[TraceEvent],
    scenario: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    if not isinstance(events, list):
        raise TypeError("TraceRecorder events must be a list")

    call_refs: dict[str, str] = {}
    key_refs: dict[str, str] = {}
    projected: list[dict[str, Any]] = []
    expected_steps = list(range(1, len(events) + 1))
    observed_steps: list[int] = []
    for event in events:
        if not isinstance(event, TraceEvent):
            raise TypeError("TraceRecorder returned a non-trace event")
        event_type = event.event_type.value
        if event_type not in _TRACE_EVENT_TYPES:
            raise ValueError("TraceRecorder returned an unsupported event type")
        if not isinstance(event.step, int) or isinstance(event.step, bool):
            raise TypeError("trace event step must be an integer")
        observed_steps.append(event.step)

        base: dict[str, Any] = {
            "step": event.step,
            "event_type": event_type,
        }
        if isinstance(event, LLMOutputEvent):
            entry = {
                **base,
                "unavailable_fields": [
                    "raw_output",
                    "parsed_plan",
                    "model",
                    "provider_token_counts",
                ],
            }
        elif isinstance(event, ToolCallEvent):
            if not event.tool_call_id or event.tool_call_id in call_refs:
                raise ValueError("trace tool-call identifiers are invalid")
            call_ref = f"call-{len(call_refs) + 1:03d}"
            call_refs[event.tool_call_id] = call_ref
            arguments = _project_tool_arguments(event.args, scenario)
            entry = {
                **base,
                "tool_call_ref": call_ref,
                "operation": _project_operation(event.tool_name),
                "arguments": arguments,
                "args_source": (
                    event.args_source
                    if event.args_source
                    in {"user_text", "system_query", "session", "unknown"}
                    else "unknown"
                ),
            }
        elif isinstance(event, PolicyDecisionEvent):
            entry = {
                **base,
                "tool_call_ref": _resolve_call_ref(event.tool_call_id, call_refs),
                "operation": _project_operation(event.tool_name),
                "decision": _enum_value(event.decision, {"allowed", "blocked", "replay"}),
                "reason_code": (
                    event.reason_code
                    if event.reason_code in _SAFE_TRACE_REASON_CODES
                    else None
                ),
                "checked_identity": {
                    "user_is_actor": (
                        event.checked_user_id == scenario["user_id"]
                        if event.checked_user_id
                        else None
                    ),
                    "owner_is_actor": (
                        event.checked_owner_id == scenario["user_id"]
                        if event.checked_owner_id
                        else None
                    ),
                },
                "idempotency_key_ref": _normalize_trace_key_ref(
                    event.idempotency_key_ref,
                    key_refs,
                ),
                "reason_text_available": False,
            }
        elif isinstance(event, HumanConfirmationEvent):
            entry = {
                **base,
                "tool_call_ref": _resolve_call_ref(event.tool_call_id, call_refs),
                "operation": _project_operation(event.tool_name),
                "arguments": _project_tool_arguments(event.args, scenario),
                "authenticated_as_actor": (
                    event.authenticated_user_id == scenario["user_id"]
                    if event.authenticated_user_id
                    else None
                ),
                "status": _enum_value(
                    event.status,
                    {"requested", "confirmed", "rejected", "expired", "consumed"},
                ),
                "confirmation_id_available": False,
                "expiry_and_state_hash_available": False,
            }
        elif isinstance(event, ToolResultEvent):
            error_type = (
                event.error_type
                if event.error_type in _SAFE_TRACE_ERROR_TYPES
                else None
            )
            entry = {
                **base,
                "tool_call_ref": _resolve_call_ref(event.tool_call_id, call_refs),
                "operation": _project_operation(event.tool_name),
                "status": _enum_value(
                    event.status,
                    {"success", "error", "timeout", "replayed"},
                ),
                "error_type": error_type,
                "idempotency_key_ref": _normalize_trace_key_ref(
                    event.idempotency_key_ref,
                    key_refs,
                ),
                "replayed": event.replayed,
                "state_changed": event.state_changed,
                "result_payload_available": False,
                "response_delivery_count": None,
                "response_delivery_count_unavailable_reason": (
                    "not_present_in_trace_event_schema"
                ),
            }
        elif isinstance(event, FinalResponseEvent):
            cited_refs = []
            for call_id in event.cited_tool_calls:
                cited_refs.append(_resolve_call_ref(call_id, call_refs))
            entry = {
                **base,
                "cited_tool_call_refs": cited_refs,
                "response_text_available": False,
                "response_class": None,
                "response_class_unavailable_reason": (
                    "not_present_in_final_response_trace_event"
                ),
            }
        else:
            raise ValueError("TraceRecorder returned an unsupported event class")
        projected.append(entry)

    if observed_steps != expected_steps:
        raise ValueError("TraceRecorder event steps are not contiguous and ordered")
    return tuple(projected)


def _project_tool_arguments(
    arguments: dict[str, Any],
    scenario: dict[str, Any],
) -> dict[str, Any]:
    if not isinstance(arguments, dict):
        raise TypeError("trace tool arguments must be an object")
    allowed = {"order_id", "amount", "currency"}
    projected: dict[str, Any] = {
        "unrecognized_parameter_count": sum(
            name not in allowed for name in arguments
        )
    }
    if "order_id" in arguments:
        projected["order_target"] = _project_order_target(
            arguments["order_id"],
            scenario,
        )
    if "amount" in arguments:
        projected["amount"] = _project_amount(arguments["amount"])
    if "currency" in arguments:
        projected["currency"] = _project_currency(
            arguments["currency"],
            scenario,
        )
    return projected


def _project_order_target(value: Any, scenario: dict[str, Any]) -> dict[str, Any]:
    if (
        not isinstance(value, str)
        or not value
        or len(value) > 64
        or not value.isascii()
        or not value.replace("-", "").replace("_", "").isalnum()
    ):
        return {"classification": "invalid_target"}

    for index, order in enumerate(scenario["initial_orders"], start=1):
        if value == order["order_id"]:
            return {
                "classification": "known_target",
                "order_ref": f"order-{index:03d}",
                "ownership": (
                    "actor_owned"
                    if order["owner_id"] == scenario["user_id"]
                    else "foreign_owned"
                ),
            }
    return {"classification": "unknown_target"}


def _project_amount(value: Any) -> dict[str, str]:
    if not isinstance(value, str) or len(value) > 18 or not re.fullmatch(
        r"(?:0|[1-9][0-9]*)(?:\.[0-9]{1,2})?",
        value,
    ):
        return {"status": "invalid"}
    amount = Decimal(value)
    if not amount.is_finite() or amount <= 0:
        return {"status": "invalid"}
    return {
        "status": "valid",
        "value": f"{amount:.2f}",
    }


def _project_currency(
    value: Any,
    scenario: dict[str, Any],
) -> dict[str, str]:
    known_currencies = {
        order["currency"] for order in scenario["initial_orders"]
    }
    if (
        not isinstance(value, str)
        or not re.fullmatch(r"[A-Z]{3}", value)
        or value not in known_currencies
    ):
        return {"status": "invalid"}
    return {"status": "valid", "value": value}


def _project_operation(operation: str) -> str:
    if operation not in {"get_order", "cancel_order", "refund_order"}:
        raise ValueError("trace event has an unsupported tool operation")
    return operation


def _enum_value(value: Any, allowed: set[str]) -> str:
    normalized = getattr(value, "value", value)
    if not isinstance(normalized, str) or normalized not in allowed:
        raise ValueError("trace event contains an unsupported semantic value")
    return normalized


def _resolve_call_ref(call_id: str, call_refs: dict[str, str]) -> str:
    try:
        return call_refs[call_id]
    except KeyError:
        raise ValueError("trace event refers to an unknown tool call") from None


def _normalize_trace_key_ref(
    key_ref: str,
    key_refs: dict[str, str],
) -> str | None:
    if not key_ref:
        return None
    if key_ref not in key_refs:
        key_refs[key_ref] = f"key-{len(key_refs) + 1:03d}"
    return key_refs[key_ref]


def _scenario_data(scenario: Any) -> dict[str, Any]:
    data = scenario if isinstance(scenario, dict) else getattr(scenario, "data", None)
    if not isinstance(data, dict):
        raise TypeError("scenario must be a scenario mapping or wrapper with data")
    return data


def _validate_runtime_fixture(data: dict[str, Any]) -> None:
    required = {
        "id",
        "user_id",
        "user_input",
        "initial_orders",
        "initial_idempotency",
        "proposals",
        "human_decisions",
        "clock_advance_seconds",
        "pre_resume_state_changes",
        "fault",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"scenario fixture is missing fields: {sorted(missing)}")
    if not isinstance(data["id"], str) or not data["id"]:
        raise ValueError("scenario id must be a non-empty string")
    if len(data["proposals"]) != 1:
        raise ValueError("Stage 3A requires exactly one scripted proposal")
    proposal = data["proposals"][0]
    if not isinstance(proposal, dict) or proposal.get("type") != "tool_call":
        raise ValueError("scenario proposal must be a scripted tool_call")
    fault = data["fault"]
    if fault is not None and (
        fault.get("proposal_index") != 0
        or proposal.get("operation") != "refund_order"
    ):
        raise ValueError("Stage 3A fault must target its scripted refund proposal")


def _fresh_state(data: dict[str, Any]) -> InMemoryOrderState:
    seeded_order_ids = {
        seed["order_id"] for seed in data["initial_idempotency"]
    }
    orders = tuple(
        Order(
            order_id=item["order_id"],
            owner_id=item["owner_id"],
            total_amount=Decimal(item["total_amount"]),
            currency=item["currency"],
            status=(
                OrderStatus.SHIPPED
                if item["order_id"] in seeded_order_ids
                else OrderStatus(item["status"])
            ),
        )
        for item in data["initial_orders"]
    )
    return InMemoryOrderState(orders)


def _seed_idempotency_records(
    data: dict[str, Any],
    tools: SimulatedOrderTools,
) -> None:
    for seed in data["initial_idempotency"]:
        tools.refund_order(
            seed["order_id"],
            seed["idempotency_key"],
            amount=Decimal(seed["amount"]),
            currency=seed["currency"],
        )


def _install_attempt_observer(
    orchestrator: AgentOrchestrator,
    data: dict[str, Any],
    observation: _Observation,
    group: ExperimentGroup,
) -> None:
    trusted_key = data.get("trusted_idempotency_key")
    original_record = orchestrator._record_tool_call

    def observe_tool_call(execution: Any) -> None:
        request = execution.pending_request
        if (
            request is not None
            and request.operation == "refund_order"
            and trusted_key is not None
        ):
            execution.pending_request = replace(
                request,
                idempotency_key=trusted_key,
            )
            request = execution.pending_request
            if request.idempotency_key != trusted_key:
                raise RuntimeError("trusted idempotency key injection failed")
        original_record(execution)
        _record_attempt(observation, request, trusted_key)

    orchestrator._record_tool_call = observe_tool_call
    if group is ExperimentGroup.BASELINE_A:
        original_call = orchestrator._call_tool

        def observe_baseline_a_retry(request: Any) -> ToolResult:
            if len(observation.attempts) <= len(observation.invocations):
                _record_attempt(observation, request, trusted_key)
            return original_call(request)

        orchestrator._call_tool = observe_baseline_a_retry


def _record_attempt(
    observation: _Observation,
    request: Any,
    trusted_key: str | None,
) -> None:
    observation.attempts.append(
        {
            "operation": request.operation if request else "",
            "order_id": request.order_id if request else "",
            "idempotency_key_ref": (
                _key_reference(request.idempotency_key, trusted_key)
                if request is not None and request.idempotency_key is not None
                else ""
            ),
        }
    )


def _install_tool_observer(
    tools: SimulatedOrderTools,
    state: InMemoryOrderState,
    order_ids: tuple[str, ...],
    data: dict[str, Any],
    observation: _Observation,
    fixture_keys: set[str],
) -> None:
    fault = data["fault"]
    fault_mode = FaultMode(fault["mode"]) if fault is not None else FaultMode.NONE
    fault_applied = False
    observed_refund_key: str | None = None

    def wrap(operation: str, original: Any) -> Any:
        def invoke(order_id: str, *args: Any, **kwargs: Any) -> ToolResult:
            nonlocal fault_applied, observed_refund_key
            applied_mode: FaultMode | None = None
            if operation == "refund_order" and fault is not None and not fault_applied:
                applied_mode = fault_mode
                fault_applied = True
                kwargs["fault"] = applied_mode
            key_ref = None
            if operation == "refund_order":
                idempotency_key = args[0] if args else kwargs.get("idempotency_key")
                if isinstance(idempotency_key, str):
                    if (
                        observed_refund_key is not None
                        and observed_refund_key != idempotency_key
                    ):
                        raise RuntimeError(
                            "refund retry did not reuse its original idempotency key"
                        )
                    observed_refund_key = idempotency_key
                    key_ref = _key_reference(idempotency_key, fixture_keys)

            before = _state_signature(state, order_ids)
            try:
                result = original(order_id, *args, **kwargs)
            except ToolError as error:
                after = _state_signature(state, order_ids)
                committed = before != after
                if committed:
                    observation.agent_state_mutations += 1
                observation.invocations.append(
                    ToolInvocation(
                        operation=operation,
                        order_id=order_id,
                        outcome="error",
                        fault=applied_mode.value if applied_mode else None,
                        error_code=error.code,
                        committed=committed,
                        idempotency_key_ref=key_ref,
                    )
                )
                raise
            except Exception as error:
                after = _state_signature(state, order_ids)
                committed = before != after
                if committed:
                    observation.agent_state_mutations += 1
                observation.invocations.append(
                    ToolInvocation(
                        operation=operation,
                        order_id=order_id,
                        outcome="error",
                        fault=applied_mode.value if applied_mode else None,
                        error_code=type(error).__name__,
                        committed=committed,
                        idempotency_key_ref=key_ref,
                    )
                )
                raise
            after = _state_signature(state, order_ids)
            committed = before != after
            if committed:
                observation.agent_state_mutations += 1
            observation.invocations.append(
                ToolInvocation(
                    operation=operation,
                    order_id=order_id,
                    outcome="replayed" if result.replayed else "success",
                    fault=applied_mode.value if applied_mode else None,
                    committed=committed,
                    idempotency_key_ref=key_ref,
                    response_delivery_count=result.response_delivery_count,
                )
            )
            return result

        return invoke

    tools.get_order = wrap("get_order", tools.get_order)
    tools.cancel_order = wrap("cancel_order", tools.cancel_order)
    tools.refund_order = wrap("refund_order", tools.refund_order)


def _install_gate_observer(gate: Any, observation: _Observation) -> None:
    original = gate.handle

    def observe_handle(request: Any, session: Any, **kwargs: Any):
        before_dispatch = kwargs.get("before_dispatch")
        if before_dispatch is not None:
            def observe_dispatch(decision: Any, consumed: bool) -> None:
                before_dispatch(decision, consumed)
                if consumed:
                    observation.confirmation_statuses.append("consumed")

            kwargs["before_dispatch"] = observe_dispatch
        result = original(request, session, **kwargs)
        observation.policy_decisions.append(
            {
                "outcome": result.policy.decision.value,
                "reason_code": result.policy.reason_code,
            }
        )
        if result.challenge is not None:
            observation.confirmation_statuses.append("requested")
        return result

    gate.handle = observe_handle


def _submit_fixture_decision(
    orchestrator: AgentOrchestrator,
    result: RunResult,
    user_id: str,
    decision_value: str,
    observation: _Observation,
) -> RunResult:
    if result.challenge is None:
        return result
    decision = HumanDecision(decision_value)
    execution = orchestrator._executions[result.thread_id]
    previous_decision_count = len(observation.policy_decisions)
    previous_confirmation_count = len(observation.confirmation_statuses)
    next_result = orchestrator.decide(
        result.thread_id,
        result.challenge.confirmation_id,
        SessionContext(user_id),
        decision,
    )
    decision_status = execution.decision_status
    resumed_outcomes = {
        item["outcome"]
        for item in observation.policy_decisions[previous_decision_count:]
    }
    if decision_status == "approved" or resumed_outcomes.intersection(
        {PolicyOutcome.ALLOWED.value, PolicyOutcome.REPLAY.value}
    ):
        consumed_index = next(
            (
                index
                for index in range(
                    previous_confirmation_count,
                    len(observation.confirmation_statuses),
                )
                if observation.confirmation_statuses[index] == "consumed"
            ),
            len(observation.confirmation_statuses),
        )
        observation.confirmation_statuses.insert(consumed_index, "confirmed")
    elif decision_status == "rejected":
        observation.confirmation_statuses.append("rejected")
    elif decision_status == "expired":
        observation.confirmation_statuses.append("expired")
    elif decision_status == "stale":
        observation.confirmation_statuses.append("consumed")

    gate_result = execution.last_gate_result
    if (
        gate_result is not None
        and len(observation.policy_decisions) == previous_decision_count
    ):
        observation.policy_decisions.append(
            {
                "outcome": gate_result.policy.decision.value,
                "reason_code": gate_result.policy.reason_code,
            }
        )
    return next_result


def _apply_external_changes(
    data: dict[str, Any],
    state: InMemoryOrderState,
    observation: _Observation,
) -> None:
    for change in data["pre_resume_state_changes"]:
        state.apply_external_order_status_change(
            change["order_id"],
            OrderStatus(change["status"]),
        )
        observation.external_state_mutations += 1


def _build_result(
    scenario_id: str,
    group: ExperimentGroup,
    run_id: str,
    run_result: RunResult | None,
    state: InMemoryOrderState,
    observation: _Observation,
    state_order_ids: tuple[str, ...],
    fixture_keys: set[str],
    baseline_a_mismatch: bool,
    *,
    failure: dict[str, str] | None = None,
    sensitive_values: set[str] | None = None,
    trace_capture: AgentGuardTraceCapture | None = None,
) -> EvaluationRunResult:
    status = (
        run_result.status
        if run_result is not None
        else RunStatus.FAILED_CLOSED
    )
    response = run_result.response if run_result is not None else ""
    response_class = _response_class(status, observation)
    return EvaluationRunResult(
        scenario_id=scenario_id,
        group=group,
        run_id=run_id,
        status=status,
        response_class=response_class,
        sanitized_response=_sanitize_response(
            response,
            sensitive_values or fixture_keys,
        ),
        tool_attempts=len(observation.attempts),
        attempted_tool_calls=tuple(observation.attempts),
        tool_invocations=len(observation.invocations),
        agent_state_mutations=observation.agent_state_mutations,
        external_state_mutations=observation.external_state_mutations,
        order_states={
            order_id: {
                "owner_id": order.owner_id,
                "total_amount": str(order.total_amount),
                "currency": order.currency,
                "status": order.status.value,
            }
            for order_id in state_order_ids
            if (order := state.inspect_order(order_id)) is not None
        },
        refunds=tuple(
            {
                "refund_id": refund.refund_id,
                "order_id": refund.order_id,
                "amount": str(refund.amount),
                "currency": refund.currency,
                "status": refund.status,
            }
            for refund in state.inspect_refunds()
        ),
        idempotency_records=tuple(
            {
                "operation": record.operation,
                "order_id": record.order_id,
                "idempotency_key_ref": _key_reference(
                    record.idempotency_key,
                    fixture_keys,
                ),
                "request_fingerprint": record.request_fingerprint,
                "refund_id": record.refund.refund_id,
            }
            for record in state.inspect_idempotency_records()
        ),
        policy_decisions=tuple(observation.policy_decisions),
        confirmation_statuses=tuple(observation.confirmation_statuses),
        tool_results=tuple(observation.invocations),
        trace_event_count=observation.trace_event_count,
        baseline_a_timing_mismatch=baseline_a_mismatch,
        failure=failure,
        trace_capture=trace_capture,
    )


def _state_signature(
    state: InMemoryOrderState,
    order_ids: tuple[str, ...],
) -> tuple[Any, ...]:
    orders = tuple(
        state.inspect_order(order_id)
        for order_id in order_ids
    )
    return (
        orders,
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    )


def _response_class(status: RunStatus, observation: _Observation) -> str:
    if status is RunStatus.PAUSED:
        return "confirmation_required"
    if status is RunStatus.FAILED_CLOSED:
        return "error"
    if observation.invocations and observation.invocations[-1].outcome == "error":
        return "error"
    if (
        observation.policy_decisions
        and all(item["outcome"] == PolicyOutcome.BLOCKED.value for item in observation.policy_decisions)
        and not observation.invocations
    ):
        return "denied"
    return "completed"


def _key_reference(
    idempotency_key: str,
    fixture_keys: str | set[str] | None,
) -> str:
    if isinstance(fixture_keys, str):
        is_fixture_key = idempotency_key == fixture_keys
    elif fixture_keys is not None:
        is_fixture_key = idempotency_key in fixture_keys
    else:
        is_fixture_key = False
    if not is_fixture_key:
        return "generated"
    digest = hashlib.sha256(idempotency_key.encode("utf-8")).hexdigest()
    return f"fixture:{digest}"


def _collect_sensitive_values(data: Any) -> set[str]:
    sensitive_markers = (
        "capability",
        "confirmation_id",
        "credential",
        "idempotency_key",
        "secret",
        "token",
        "resume_marker",
    )
    values: set[str] = set()

    def visit(value: Any, field_name: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, str(key))
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, field_name)
        elif (
            isinstance(value, str)
            and value
            and any(marker in field_name.lower() for marker in sensitive_markers)
        ):
            values.add(value)

    visit(data)
    return values


def _sanitize_response(
    response: str,
    sensitive_values: set[str] | None = None,
) -> str:
    response = re.sub(r"(?i)\bbearer\s+\S+", "Bearer [REDACTED]", response)
    response = re.sub(r"\b[A-Za-z0-9_-]{40,}\b", "[REDACTED]", response)
    for value in sorted(sensitive_values or (), key=len, reverse=True):
        response = response.replace(value, "[REDACTED]")
    return response[:2000]
