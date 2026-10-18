"""Minimal LangGraph orchestration for AgentGuard V1."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
import json
import secrets
from typing import Any, Callable, TypedDict
from uuid import uuid4

from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Command, interrupt
from langchain_core.runnables import RunnableConfig

from policy_gate import (
    ConfirmationChallenge,
    ConfirmationError,
    GateResult,
    HumanDecision,
    PolicyDecision,
    PolicyGate,
    SessionContext,
    ToolRequest,
)
from simulated_tools import (
    FaultMode,
    InMemoryOrderState,
    OrderView,
    Refund,
    SimulatedOrderTools,
    ToolError,
    ToolResult,
    ToolTimeoutError,
)
from trace_events import (
    ConfirmationStatus,
    FinalResponseEvent,
    HumanConfirmationEvent,
    LLMOutputEvent,
    PolicyOutcome,
    PolicyDecisionEvent,
    ToolCallEvent,
    ToolResultEvent,
    ToolStatus,
)
from trace_recorder import TraceRecorder

MAX_TOOL_ROUNDS = 5
MAX_AUTOMATIC_TIMEOUT_RETRIES = 1


class ExperimentGroup(str, Enum):
    BASELINE_A = "baseline_a"
    BASELINE_B = "baseline_b"
    AGENTGUARD = "agentguard"


class RunStatus(str, Enum):
    PAUSED = "paused"
    COMPLETED = "completed"
    FAILED_CLOSED = "failed_closed"


class GraphState(TypedDict):
    user_message: str
    messages: list[dict[str, str]]
    proposal: dict[str, Any]
    route: str
    round_count: int
    retry_count: int
    tool_call_id: str
    pending_action: dict[str, Any]
    confirmation_challenge: dict[str, Any]
    confirmation_status: str
    response: str


@dataclass(frozen=True)
class RunResult:
    thread_id: str
    status: RunStatus
    response: str = ""
    challenge: ConfirmationChallenge | None = None


@dataclass
class _Execution:
    session: SessionContext | None
    scenario_id: str
    run_id: str
    group: ExperimentGroup
    recorder: TraceRecorder | None
    pending_request: ToolRequest | None = None
    tool_call_id: str = ""
    active_confirmation_id: str = ""
    capability: str | None = None
    challenge: ConfirmationChallenge | None = None
    decision_status: str = ""
    decision: HumanDecision | None = None
    resume_marker: str = ""
    last_gate_result: GateResult | None = None
    last_tool_result: ToolResult | None = None
    last_tool_error: ToolError | None = None
    confirmation_events: set[tuple[str, ConfirmationStatus]] = field(
        default_factory=set
    )
    cited_tool_calls: list[str] = field(default_factory=list)


class AgentOrchestrator:
    """Run a structured model proposal through the approved V1 boundaries.

    The model is a callable receiving message dictionaries and returning one
    JSON-like proposal. It receives no Python tool objects, Gate, session, or
    confirmation capability.
    """

    def __init__(
        self,
        state: InMemoryOrderState,
        model: Callable[[list[dict[str, str]]], dict[str, Any]],
        *,
        group: ExperimentGroup = ExperimentGroup.AGENTGUARD,
        clock: Callable[[], float] | None = None,
    ) -> None:
        if group is ExperimentGroup.BASELINE_A:
            raise ValueError(
                "Baseline A is available only through for_evaluation()"
            )
        self._initialize(state, model, group=group, clock=clock)

    @classmethod
    def for_evaluation(
        cls,
        state: InMemoryOrderState,
        model: Callable[[list[dict[str, str]]], dict[str, Any]],
        *,
        clock: Callable[[], float] | None = None,
    ) -> AgentOrchestrator:
        """Create an unguarded instance for trusted evaluation harnesses only.

        Never use this factory for ordinary application configuration.
        """
        orchestrator = cls.__new__(cls)
        orchestrator._initialize(
            state,
            model,
            group=ExperimentGroup.BASELINE_A,
            clock=clock,
        )
        return orchestrator

    def _initialize(
        self,
        state: InMemoryOrderState,
        model: Callable[[list[dict[str, str]]], dict[str, Any]],
        *,
        group: ExperimentGroup,
        clock: Callable[[], float] | None,
    ) -> None:
        if not isinstance(state, InMemoryOrderState):
            raise TypeError("state must be an InMemoryOrderState")
        if not callable(model):
            raise TypeError("model must be callable")
        if not isinstance(group, ExperimentGroup):
            raise TypeError("group must be an ExperimentGroup")

        self._state = state
        self._tools = SimulatedOrderTools(state)
        self._group = group
        if group is ExperimentGroup.BASELINE_A:
            self._gate = None
        else:
            gate_options = {} if clock is None else {"clock": clock}
            self._gate = PolicyGate(state, self._tools, **gate_options)
        self._model = model
        self._executions: dict[str, _Execution] = {}
        self._checkpointer = InMemorySaver()
        self._graph = self._build_graph()

    def start(
        self,
        user_message: str,
        session: SessionContext | None,
        *,
        scenario_id: str = "interactive",
        run_id: str = "run-1",
    ) -> RunResult:
        if not isinstance(user_message, str):
            raise TypeError("user_message must be a string")
        thread_id = uuid4().hex
        recorder = (
            TraceRecorder(scenario_id=scenario_id, run_id=run_id)
            if self._group is ExperimentGroup.AGENTGUARD
            else None
        )
        self._executions[thread_id] = _Execution(
            session=session,
            scenario_id=scenario_id,
            run_id=run_id,
            group=self._group,
            recorder=recorder,
        )
        state: GraphState = {
            "user_message": user_message,
            "messages": [{"role": "user", "content": user_message}],
            "proposal": {},
            "route": "",
            "round_count": 0,
            "retry_count": 0,
            "tool_call_id": "",
            "pending_action": {},
            "confirmation_challenge": {},
            "confirmation_status": "",
            "response": "",
        }
        output = self._graph.invoke(state, self._config(thread_id))
        return self._public_result(thread_id, output)

    def decide(
        self,
        thread_id: str,
        confirmation_id: str,
        session: SessionContext,
        decision: HumanDecision,
    ) -> RunResult:
        """Record an explicit decision from a trusted authenticated handler.

        This is application-side orchestration API, not an LLM tool or a
        public unauthenticated endpoint. The caller must establish the human
        decision and trusted session before invoking it.
        """
        execution = self._executions.get(thread_id)
        if execution is None or self._gate is None:
            return self._failed_closed(thread_id)
        if (
            not execution.active_confirmation_id
            or execution.active_confirmation_id != confirmation_id
            or execution.pending_request is None
            or execution.capability is None
            or execution.decision_status
        ):
            return self._failed_closed(thread_id)
        if not isinstance(decision, HumanDecision):
            return self._failed_closed(thread_id)

        try:
            self._gate.record_human_decision(
                confirmation_id,
                session,
                decision,
                capability=execution.capability,
            )
        except ConfirmationError as error:
            if error.code not in {
                "confirmation_expired",
                "confirmation_state_changed",
                "confirmation_consumed",
                "order_unavailable",
            }:
                return self._failed_closed(thread_id)
            terminal_status = (
                "expired"
                if error.code == "confirmation_expired"
                else "stale"
            )
            execution.decision_status = terminal_status
            execution.capability = None
            self._record_confirmation_status(
                execution,
                (
                    ConfirmationStatus.EXPIRED
                    if terminal_status == "expired"
                    else ConfirmationStatus.CONSUMED
                ),
            )
        else:
            execution.decision = decision
            execution.decision_status = (
                "approved" if decision is HumanDecision.APPROVE else "rejected"
            )
            self._record_confirmation_status(
                execution,
                (
                    ConfirmationStatus.CONFIRMED
                    if decision is HumanDecision.APPROVE
                    else ConfirmationStatus.REJECTED
                ),
            )

        execution.resume_marker = secrets.token_urlsafe(18)
        config = self._config(thread_id)
        output = self._graph.invoke(
            Command(
                resume={
                    "confirmation_id": confirmation_id,
                    "marker": execution.resume_marker,
                }
            ),
            config,
        )
        return self._public_result(thread_id, output)

    def trace_events(self, thread_id: str):
        execution = self._executions.get(thread_id)
        if execution is None or execution.recorder is None:
            return []
        return execution.recorder.get_events()

    def export_trace(self, thread_id: str, path: str) -> None:
        execution = self._executions.get(thread_id)
        if execution is None or execution.recorder is None:
            raise ValueError("trace is unavailable for this execution")
        execution.recorder.export_json(path)

    def close(self, thread_id: str) -> None:
        execution = self._executions.get(thread_id)
        if execution is None:
            return
        if execution.active_confirmation_id:
            raise ValueError("cannot close an execution with a pending confirmation")
        execution.capability = None
        del self._executions[thread_id]

    def _build_graph(self):
        builder = StateGraph(GraphState)
        builder.add_node("plan", self._plan)
        builder.add_node("validate", self._validate)
        builder.add_node("create_attempt", self._create_attempt)
        builder.add_node("dispatch", self._dispatch)
        builder.add_node("prepare_pause", self._prepare_pause)
        builder.add_node("await_human", self._await_human)
        builder.add_node("resume_gate", self._resume_gate)
        builder.add_node("recover_timeout", self._recover_timeout)
        builder.add_node("after_result", self._after_result)
        builder.add_node("finalize", self._finalize)

        builder.add_edge(START, "plan")
        builder.add_edge("plan", "validate")
        builder.add_conditional_edges(
            "validate",
            lambda state: state["route"],
            {
                "attempt": "create_attempt",
                "finalize": "finalize",
            },
        )
        builder.add_edge("create_attempt", "dispatch")
        builder.add_conditional_edges(
            "dispatch",
            lambda state: state["route"],
            {
                "pause": "prepare_pause",
                "recover": "recover_timeout",
                "after_result": "after_result",
            },
        )
        builder.add_edge("prepare_pause", "await_human")
        builder.add_edge("await_human", "resume_gate")
        builder.add_conditional_edges(
            "resume_gate",
            lambda state: state["route"],
            {
                "recover": "recover_timeout",
                "after_result": "after_result",
                "finalize": "finalize",
            },
        )
        builder.add_conditional_edges(
            "recover_timeout",
            lambda state: state["route"],
            {
                "pause": "prepare_pause",
                "after_result": "after_result",
            },
        )
        builder.add_conditional_edges(
            "after_result",
            lambda state: state["route"],
            {
                "plan": "plan",
                "finalize": "finalize",
            },
        )
        builder.add_edge("finalize", END)
        return builder.compile(checkpointer=self._checkpointer)

    def _plan(self, state: GraphState, config: RunnableConfig) -> dict[str, Any]:
        execution = self._execution(config)
        proposal = self._model(deepcopy(state["messages"]))
        safe_proposal = _checkpoint_safe(proposal)
        if not isinstance(safe_proposal, dict):
            safe_proposal = {}
        if execution.recorder is not None:
            audit_proposal = _safe_audit_proposal(safe_proposal)
            execution.recorder.record(
                LLMOutputEvent(
                    raw_output=_compact_json(audit_proposal),
                    parsed_plan=audit_proposal,
                )
            )
        return {"proposal": safe_proposal}

    def _validate(self, state: GraphState, config: RunnableConfig) -> dict[str, Any]:
        del config
        proposal = state["proposal"]
        if not isinstance(proposal, dict):
            return {
                "route": "finalize",
                "response": "I could not process that request.",
            }
        if proposal.get("type") == "final":
            if set(proposal) != {"type", "response"} or not isinstance(
                proposal["response"], str
            ):
                return {
                    "route": "finalize",
                    "response": "I could not process that response.",
                }
            return {"route": "finalize", "response": proposal["response"]}
        if proposal.get("type") != "tool_call":
            return {
                "route": "finalize",
                "response": "I could not process that request.",
            }
        if state["round_count"] >= MAX_TOOL_ROUNDS:
            return {
                "route": "finalize",
                "response": "I reached the supported action limit for this turn.",
            }
        operation = proposal.get("operation")
        expected_keys = {
            "get_order": {"type", "operation", "order_id"},
            "cancel_order": {"type", "operation", "order_id"},
            "refund_order": {
                "type",
                "operation",
                "order_id",
                "amount",
                "currency",
            },
        }
        if (
            not isinstance(operation, str)
            or operation not in expected_keys
            or set(proposal) != expected_keys[operation]
        ):
            return {
                "route": "finalize",
                "response": "I could not process that tool request.",
            }
        if not isinstance(proposal.get("order_id"), str) or not proposal["order_id"]:
            return {
                "route": "finalize",
                "response": "I could not process that tool request.",
            }
        if operation == "refund_order":
            if not isinstance(proposal["currency"], str):
                return {
                    "route": "finalize",
                    "response": "I could not process that refund request.",
                }
            try:
                amount = Decimal(str(proposal["amount"]))
            except (InvalidOperation, ValueError):
                return {
                    "route": "finalize",
                    "response": "I could not process that refund request.",
                }
            if not amount.is_finite() or amount <= 0:
                return {
                    "route": "finalize",
                    "response": "I could not process that refund request.",
                }
        return {"route": "attempt"}

    def _create_attempt(
        self,
        state: GraphState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        execution = self._execution(config)
        proposal = state["proposal"]
        operation = proposal["operation"]
        amount = (
            Decimal(str(proposal["amount"]))
            if operation == "refund_order"
            else None
        )
        key = secrets.token_urlsafe(24) if operation == "refund_order" else None
        request = ToolRequest(
            operation=operation,
            order_id=proposal["order_id"],
            amount=amount,
            currency=proposal.get("currency"),
            idempotency_key=key,
            args_source="unknown",
        )
        execution.pending_request = request
        execution.tool_call_id = uuid4().hex
        execution.last_gate_result = None
        execution.last_tool_result = None
        execution.last_tool_error = None
        execution.cited_tool_calls.append(execution.tool_call_id)
        self._record_tool_call(execution)
        return {
            "route": "dispatch",
            "round_count": state["round_count"] + 1,
            "retry_count": 0,
            "tool_call_id": execution.tool_call_id,
            "pending_action": _public_action(request),
            "confirmation_challenge": {},
            "confirmation_status": "",
        }

    def _dispatch(self, state: GraphState, config: RunnableConfig) -> dict[str, Any]:
        del state
        execution = self._execution(config)
        request = self._require_request(execution)
        if execution.group is ExperimentGroup.BASELINE_A:
            try:
                execution.last_tool_result = self._call_tool(request)
            except ToolError as error:
                execution.last_tool_error = error
        else:
            assert self._gate is not None
            execution.last_gate_result = self._gate.handle(
                request,
                execution.session,
                before_dispatch=lambda decision, consumed: self._record_pre_dispatch(
                    execution,
                    decision,
                    consumed,
                ),
            )
            self._record_gate_result(execution, execution.last_gate_result)
        return self._route_after_call(execution, retry_count=0)

    def _prepare_pause(
        self,
        state: GraphState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        execution = self._execution(config)
        gate_result = execution.last_gate_result
        if gate_result is None or gate_result.challenge is None or self._gate is None:
            return {
                "route": "finalize",
                "response": "Confirmation could not be prepared.",
            }
        challenge = gate_result.challenge
        try:
            execution.capability = self._gate.take_capability_handoff(
                challenge.confirmation_id
            )
        except ConfirmationError:
            return {
                "route": "finalize",
                "response": "Confirmation could not be prepared.",
            }
        execution.active_confirmation_id = challenge.confirmation_id
        execution.challenge = challenge
        execution.decision_status = ""
        execution.decision = None
        self._record_confirmation_status(execution, ConfirmationStatus.REQUESTED)
        return {
            "route": "await_human",
            "confirmation_challenge": _public_challenge(challenge),
            "confirmation_status": "requested",
        }

    def _await_human(
        self,
        state: GraphState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        resume_value = interrupt(state["confirmation_challenge"])
        execution = self._execution(config)
        if (
            not isinstance(resume_value, dict)
            or execution.decision_status
            not in {"approved", "rejected", "expired", "stale"}
            or resume_value.get("confirmation_id")
            != execution.active_confirmation_id
            or resume_value.get("marker") != execution.resume_marker
        ):
            execution.capability = None
            return {"route": "finalize", "response": "Confirmation resume failed closed."}
        return {"route": "resume_gate"}

    def _resume_gate(
        self,
        state: GraphState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        execution = self._execution(config)
        request = self._require_request(execution)
        if execution.decision_status == "expired":
            result = GateResult(
                policy=PolicyDecision(
                    PolicyOutcome.BLOCKED,
                    "confirmation_expired",
                    "Confirmation has expired",
                )
            )
        elif execution.decision_status == "stale":
            result = GateResult(
                policy=PolicyDecision(
                    PolicyOutcome.BLOCKED,
                    "confirmation_state_changed",
                    "Confirmation is no longer valid",
                )
            )
        elif self._gate is None or execution.capability is None:
            execution.capability = None
            return {"route": "finalize", "response": "Confirmation resume failed closed."}
        else:
            confirmation_consumed = False
            dispatch_returned = False

            def record_pre_dispatch(
                decision: PolicyDecision,
                consumed: bool,
            ) -> None:
                nonlocal confirmation_consumed
                confirmation_consumed = consumed
                self._record_pre_dispatch(execution, decision, consumed)

            try:
                result = self._gate.handle(
                    request,
                    execution.session,
                    confirmation_capability=execution.capability,
                    before_dispatch=record_pre_dispatch,
                )
                dispatch_returned = True
            finally:
                if confirmation_consumed and not dispatch_returned:
                    execution.capability = None
                    execution.active_confirmation_id = ""
        execution.capability = None
        execution.last_gate_result = result
        self._record_gate_result(execution, result, resumed=True)
        self._record_confirmation_status_for_result(execution, result)
        return self._route_after_call(execution, retry_count=state["retry_count"])

    def _recover_timeout(
        self,
        state: GraphState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        execution = self._execution(config)
        request = self._require_request(execution)
        if execution.group is ExperimentGroup.BASELINE_A:
            try:
                execution.last_tool_result = self._call_tool(request)
            except ToolError as error:
                execution.last_tool_error = error
            return {"route": "after_result", "retry_count": state["retry_count"] + 1}

        assert self._gate is not None
        execution.tool_call_id = uuid4().hex
        execution.last_gate_result = None
        execution.last_tool_result = None
        execution.last_tool_error = None
        execution.cited_tool_calls.append(execution.tool_call_id)
        self._record_tool_call(execution)
        result = self._gate.handle(
            request,
            execution.session,
            before_dispatch=lambda decision, consumed: self._record_pre_dispatch(
                execution,
                decision,
                consumed,
            ),
        )
        execution.last_gate_result = result
        self._record_gate_result(execution, result)
        route = self._route_after_call(
            execution,
            retry_count=state["retry_count"] + 1,
        )
        return {
            "route": route["route"],
            "retry_count": state["retry_count"] + 1,
        }

    def _after_result(
        self,
        state: GraphState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        execution = self._execution(config)
        message = self._model_result_message(execution)
        messages = [*state["messages"], {"role": "tool", "content": message}]
        if state["round_count"] >= MAX_TOOL_ROUNDS:
            return {
                "messages": messages,
                "route": "finalize",
                "response": "I reached the supported action limit for this turn.",
            }
        return {"messages": messages, "route": "plan"}

    def _finalize(
        self,
        state: GraphState,
        config: RunnableConfig,
    ) -> dict[str, Any]:
        execution = self._execution(config)
        response = state.get("response") or "I could not complete that request."
        if execution.recorder is not None:
            execution.recorder.record(
                FinalResponseEvent(
                    response=response,
                    cited_tool_calls=list(execution.cited_tool_calls),
                )
            )
        execution.capability = None
        execution.active_confirmation_id = ""
        return {"response": response, "route": "done"}

    def _route_after_call(
        self,
        execution: _Execution,
        *,
        retry_count: int,
    ) -> dict[str, Any]:
        gate_result = execution.last_gate_result
        if gate_result is not None and gate_result.challenge is not None:
            return {"route": "pause", "retry_count": retry_count}
        error = (
            gate_result.execution_error
            if gate_result is not None
            else execution.last_tool_error
        )
        if (
            isinstance(error, ToolTimeoutError)
            and retry_count < MAX_AUTOMATIC_TIMEOUT_RETRIES
        ):
            return {"route": "recover", "retry_count": retry_count}
        return {"route": "after_result", "retry_count": retry_count}

    def _record_tool_call(self, execution: _Execution) -> None:
        if execution.recorder is None or execution.pending_request is None:
            return
        request = execution.pending_request
        args: dict[str, Any] = {"order_id": request.order_id}
        if request.amount is not None:
            args["amount"] = str(request.amount)
            args["currency"] = request.currency
        execution.recorder.record(
            ToolCallEvent(
                tool_call_id=execution.tool_call_id,
                tool_name=request.operation,
                args=args,
                args_source=request.args_source,
            )
        )

    def _record_gate_result(
        self,
        execution: _Execution,
        result: GateResult,
        *,
        resumed: bool = False,
    ) -> None:
        if execution.recorder is None or execution.pending_request is None:
            return
        request = execution.pending_request
        if result.challenge is not None:
            execution.active_confirmation_id = result.challenge.confirmation_id
        if result.result is None and result.execution_error is None:
            self._record_policy_decision(execution, result.policy)
        else:
            self._record_tool_result(execution, result)

    def _record_pre_dispatch(
        self,
        execution: _Execution,
        decision: PolicyDecision,
        confirmation_consumed: bool,
    ) -> None:
        self._record_policy_decision(execution, decision)
        if confirmation_consumed:
            self._record_confirmation_status(
                execution,
                ConfirmationStatus.CONSUMED,
            )

    def _record_policy_decision(
        self,
        execution: _Execution,
        decision: PolicyDecision,
    ) -> None:
        if execution.recorder is None or execution.pending_request is None:
            return
        execution.recorder.record(
            PolicyDecisionEvent(
                tool_call_id=execution.tool_call_id,
                tool_name=execution.pending_request.operation,
                decision=decision.decision,
                reason_code=decision.reason_code,
                reason=decision.reason,
                checked_user_id=decision.checked_user_id,
                checked_owner_id=decision.checked_owner_id,
                idempotency_key_ref=decision.idempotency_key_ref,
            )
        )

    def _record_tool_result(self, execution: _Execution, result: GateResult) -> None:
        if execution.recorder is None or execution.pending_request is None:
            return
        request = execution.pending_request
        error = result.execution_error
        tool_result = result.result
        if error is not None:
            status = (
                ToolStatus.TIMEOUT
                if isinstance(error, ToolTimeoutError)
                else ToolStatus.ERROR
            )
            state_changed = (
                error.committed if isinstance(error, ToolTimeoutError) else False
            )
            result_value = None
            error_type = error.code
        elif tool_result is not None:
            status = (
                ToolStatus.REPLAYED
                if tool_result.replayed or result.policy.decision is PolicyOutcome.REPLAY
                else ToolStatus.SUCCESS
            )
            state_changed = (
                request.operation != "get_order"
                and not tool_result.replayed
                and result.policy.decision is not PolicyOutcome.REPLAY
            )
            result_value = _tool_result_value(tool_result)
            error_type = ""
        else:
            return
        execution.recorder.record(
            ToolResultEvent(
                tool_call_id=execution.tool_call_id,
                tool_name=request.operation,
                status=status,
                result=result_value,
                error_type=error_type,
                idempotency_key_ref=result.policy.idempotency_key_ref,
                replayed=bool(tool_result and tool_result.replayed),
                state_changed=state_changed,
                duration_ms=(
                    tool_result.duration_ms
                    if tool_result is not None
                    else int(error.duration_ms or 0)
                ),
            )
        )

    def _record_confirmation_status(
        self,
        execution: _Execution,
        status: ConfirmationStatus,
    ) -> None:
        if execution.recorder is None or execution.pending_request is None:
            return
        confirmation_id = execution.active_confirmation_id
        key = (confirmation_id, status)
        if key in execution.confirmation_events:
            return
        execution.confirmation_events.add(key)
        request = execution.pending_request
        args: dict[str, Any] = {"order_id": request.order_id}
        if request.amount is not None:
            args["amount"] = str(request.amount)
            args["currency"] = request.currency
        expires_at = ""
        state_hash = ""
        if execution.challenge is not None:
            expires_at = execution.challenge.expires_at
            state_hash = execution.challenge.order_state_hash
        execution.recorder.record(
            HumanConfirmationEvent(
                tool_call_id=execution.tool_call_id,
                tool_name=request.operation,
                args=args,
                authenticated_user_id=(
                    execution.session.user_id
                    if isinstance(execution.session, SessionContext)
                    else ""
                ),
                confirmation_id=confirmation_id,
                token_expires_at=expires_at,
                order_state_hash=state_hash,
                status=status,
            )
        )

    def _record_confirmation_status_for_result(
        self,
        execution: _Execution,
        result: GateResult,
    ) -> None:
        if result.policy.reason_code == "confirmation_expired":
            self._record_confirmation_status(execution, ConfirmationStatus.EXPIRED)
        elif result.policy.reason_code in {
            "confirmation_reused",
            "confirmation_state_changed",
        }:
            self._record_confirmation_status(execution, ConfirmationStatus.CONSUMED)
        elif result.policy.decision is PolicyOutcome.ALLOWED:
            self._record_confirmation_status(execution, ConfirmationStatus.CONSUMED)

    def _model_result_message(self, execution: _Execution) -> str:
        gate_result = execution.last_gate_result
        if gate_result is not None:
            if gate_result.result is not None:
                payload = {
                    "outcome": gate_result.policy.decision.value,
                    "result": _tool_result_value(gate_result.result),
                }
            elif gate_result.execution_error is not None:
                payload = {
                    "outcome": gate_result.policy.decision.value,
                    "error": gate_result.execution_error.code,
                }
            else:
                payload = {
                    "outcome": gate_result.policy.decision.value,
                    "reason_code": gate_result.policy.reason_code,
                    "reason": gate_result.policy.reason,
                }
            return _compact_json(payload)
        if execution.last_tool_error is not None:
            return _compact_json({"error": execution.last_tool_error.code})
        if execution.last_tool_result is not None:
            return _compact_json(
                {"result": _tool_result_value(execution.last_tool_result)}
            )
        return _compact_json({"error": "no_tool_result"})

    def _call_tool(self, request: ToolRequest) -> ToolResult:
        if request.operation == "get_order":
            return self._tools.get_order(request.order_id)
        if request.operation == "cancel_order":
            return self._tools.cancel_order(request.order_id)
        assert request.amount is not None
        assert request.currency is not None
        assert request.idempotency_key is not None
        return self._tools.refund_order(
            request.order_id,
            request.idempotency_key,
            amount=request.amount,
            currency=request.currency,
            fault=FaultMode.NONE,
        )

    def _require_request(self, execution: _Execution) -> ToolRequest:
        if execution.pending_request is None:
            raise RuntimeError("execution has no pending tool request")
        return execution.pending_request

    def _execution(self, config: RunnableConfig) -> _Execution:
        thread_id = config.get("configurable", {}).get("thread_id")
        execution = self._executions.get(thread_id)
        if execution is None:
            raise RuntimeError("trusted execution continuation is unavailable")
        return execution

    @staticmethod
    def _config(thread_id: str) -> dict[str, Any]:
        return {"configurable": {"thread_id": thread_id}}

    def _public_result(self, thread_id: str, state: GraphState) -> RunResult:
        if state["route"] == "await_human":
            challenge = _challenge_from_dict(state["confirmation_challenge"])
            return RunResult(thread_id, RunStatus.PAUSED, challenge=challenge)
        if state["route"] == "done":
            return RunResult(
                thread_id,
                RunStatus.COMPLETED,
                response=state.get("response", ""),
            )
        return RunResult(
            thread_id,
            RunStatus.FAILED_CLOSED,
            response=state.get("response", "Execution could not continue."),
        )

    @staticmethod
    def _failed_closed(thread_id: str) -> RunResult:
        return RunResult(
            thread_id,
            RunStatus.FAILED_CLOSED,
            response="The confirmation continuation is unavailable.",
        )


def _checkpoint_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        safe_value = {}
        for key, item in value.items():
            safe_key = str(key)
            normalized_key = "".join(
                character.lower()
                for character in safe_key
                if character.isalnum()
            )
            if any(
                marker in normalized_key
                for marker in (
                    "password",
                    "secret",
                    "token",
                    "capability",
                    "credential",
                    "authorization",
                    "bearer",
                    "idempotencykey",
                )
            ):
                safe_value[safe_key] = "[REDACTED]"
            else:
                safe_value[safe_key] = _checkpoint_safe(item)
        return safe_value
    if isinstance(value, (list, tuple)):
        return [_checkpoint_safe(item) for item in value]
    return "[unsupported]"


def _safe_audit_proposal(proposal: dict[str, Any]) -> dict[str, Any]:
    """Keep only bounded, non-free-text fields from untrusted model output."""
    proposal_type = proposal.get("type")
    if proposal_type == "final":
        return {"type": "final"}
    if proposal_type != "tool_call":
        return {"type": "invalid"}

    audit: dict[str, Any] = {"type": "tool_call"}
    operation = proposal.get("operation")
    if isinstance(operation, str) and operation in {
        "get_order",
        "cancel_order",
        "refund_order",
    }:
        audit["operation"] = operation
    order_id = proposal.get("order_id")
    if isinstance(order_id, str):
        audit["order_id"] = order_id
    amount = proposal.get("amount")
    if isinstance(amount, (str, int, float, Decimal)) and not isinstance(amount, bool):
        audit["amount"] = str(amount)
    currency = proposal.get("currency")
    if isinstance(currency, str):
        audit["currency"] = currency
    return audit


def _compact_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _public_action(request: ToolRequest) -> dict[str, Any]:
    action: dict[str, Any] = {
        "operation": request.operation,
        "order_id": request.order_id,
    }
    if request.amount is not None:
        action["amount"] = str(request.amount)
        action["currency"] = request.currency
    return action


def _public_challenge(challenge: ConfirmationChallenge) -> dict[str, Any]:
    return {
        "confirmation_id": challenge.confirmation_id,
        "operation": challenge.operation,
        "order_id": challenge.order_id,
        "parameters": [list(item) for item in challenge.parameters],
        "expires_at": challenge.expires_at,
        "order_state_hash": challenge.order_state_hash,
    }


def _challenge_from_dict(value: dict[str, Any]) -> ConfirmationChallenge:
    return ConfirmationChallenge(
        confirmation_id=value["confirmation_id"],
        operation=value["operation"],
        order_id=value["order_id"],
        parameters=tuple(tuple(item) for item in value["parameters"]),
        expires_at=value["expires_at"],
        order_state_hash=value["order_state_hash"],
    )


def _tool_result_value(result: ToolResult) -> dict[str, Any]:
    value = result.value
    if isinstance(value, OrderView):
        result_value: dict[str, Any] = {
            "order_id": value.order_id,
            "total_amount": str(value.total_amount),
            "currency": value.currency,
            "status": value.status.value,
        }
    elif isinstance(value, Refund):
        result_value = {
            "refund_id": value.refund_id,
            "order_id": value.order_id,
            "amount": str(value.amount),
            "currency": value.currency,
            "status": value.status,
        }
    else:
        raise TypeError(f"unsupported tool result value: {type(value).__name__}")
    return {
        **result_value,
        "operation": result.operation,
        "replayed": result.replayed,
        "response_delivery_count": result.response_delivery_count,
        "duration_ms": result.duration_ms,
    }
