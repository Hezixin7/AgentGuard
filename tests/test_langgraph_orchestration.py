import json
from decimal import Decimal

import pytest
from langgraph.types import Command

from agent_graph import (
    MAX_AUTOMATIC_TIMEOUT_RETRIES,
    MAX_TOOL_ROUNDS,
    AgentOrchestrator,
    ExperimentGroup,
    RunStatus,
)
from policy_gate import (
    CONFIRMATION_TTL_SECONDS,
    HumanDecision,
    SessionContext,
    ToolRequest,
)
from simulated_tools import FaultMode, InMemoryOrderState, Order, OrderStatus
from trace_events import (
    ConfirmationStatus,
    EventType,
    PolicyOutcome,
    PolicyDecisionEvent,
    ToolStatus,
)


class ScriptedModel:
    def __init__(self, *proposals):
        self.proposals = list(proposals)
        self.calls = []

    def __call__(self, messages):
        self.calls.append(messages)
        if self.proposals:
            return self.proposals.pop(0)
        return {"type": "final", "response": "Done."}


class FakeClock:
    def __init__(self):
        self.value = 10.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def _state(*orders):
    return InMemoryOrderState(orders)


def _session(user_id="user_a"):
    return SessionContext(user_id)


def _cancel(order_id="C1"):
    return {
        "type": "tool_call",
        "operation": "cancel_order",
        "order_id": order_id,
    }


def _refund(order_id="R1"):
    return {
        "type": "tool_call",
        "operation": "refund_order",
        "order_id": order_id,
        "amount": "49.95",
        "currency": "USD",
    }


def test_get_order_runs_through_gate_and_returns_order_view():
    state = _state(
        Order("O1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    model = ScriptedModel(
        {
            "type": "tool_call",
            "operation": "get_order",
            "order_id": "O1",
        },
        {"type": "final", "response": "Your order is pending."},
    )
    orchestrator = AgentOrchestrator(state, model)

    result = orchestrator.start("Check O1", _session())

    assert result.status is RunStatus.COMPLETED
    assert result.response == "Your order is pending."
    assert any(
        event.event_type is EventType.POLICY_DECISION
        and event.decision is PolicyOutcome.ALLOWED
        for event in orchestrator.trace_events(result.thread_id)
    )
    assert state.inspect_order("O1").status is OrderStatus.PENDING


def test_cross_user_access_is_blocked_before_disclosure():
    state = _state(
        Order("O1", "user_b", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    model = ScriptedModel(
        {
            "type": "tool_call",
            "operation": "get_order",
            "order_id": "O1",
        },
        {"type": "final", "response": "I cannot access that order."},
    )
    orchestrator = AgentOrchestrator(state, model)

    result = orchestrator.start("Check O1", _session("user_a"))

    assert result.status is RunStatus.COMPLETED
    assert state.inspect_order("O1").status is OrderStatus.PENDING
    assert "10" not in model.calls[-1][-1]["content"]
    assert "pending" not in model.calls[-1][-1]["content"]
    assert any(
        event.event_type is EventType.POLICY_DECISION
        and event.reason_code == "order_unavailable"
        for event in orchestrator.trace_events(result.thread_id)
    )


def test_missing_session_is_blocked_before_order_disclosure():
    state = _state(
        Order("O1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    model = ScriptedModel(
        {
            "type": "tool_call",
            "operation": "get_order",
            "order_id": "O1",
        },
        {"type": "final", "response": "I cannot access that order."},
    )
    orchestrator = AgentOrchestrator(state, model)

    result = orchestrator.start("Check O1", None)

    assert result.status is RunStatus.COMPLETED
    assert "10" not in model.calls[-1][-1]["content"]
    assert "pending" not in model.calls[-1][-1]["content"]
    assert any(
        event.event_type is EventType.POLICY_DECISION
        and event.decision is PolicyOutcome.BLOCKED
        for event in orchestrator.trace_events(result.thread_id)
    )


@pytest.mark.parametrize(
    ("proposal", "order"),
    [
        (
            _cancel(),
            Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING),
        ),
        (
            _refund(),
            Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED),
        ),
    ],
)
def test_cancel_and_refund_pause_for_explicit_confirmation(proposal, order):
    state = _state(order)
    orchestrator = AgentOrchestrator(
        state,
        ScriptedModel(proposal),
    )

    result = orchestrator.start("Do the requested operation", _session())

    assert result.status is RunStatus.PAUSED
    assert result.challenge.operation == proposal["operation"]
    assert not hasattr(result.challenge, "capability")
    assert state.inspect_order(order.order_id) == order
    assert state.inspect_refunds() == ()
    assert state.inspect_idempotency_records() == ()


def test_confirmation_hold_trace_precedes_any_execution():
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    orchestrator = AgentOrchestrator(state, ScriptedModel(_cancel()))

    paused = orchestrator.start("Cancel C1", _session())

    events = orchestrator.trace_events(paused.thread_id)
    event_types = [event.event_type for event in events]
    assert event_types.index(EventType.TOOL_CALL) < event_types.index(
        EventType.POLICY_DECISION
    )
    assert event_types.index(EventType.POLICY_DECISION) < event_types.index(
        EventType.HUMAN_CONFIRMATION
    )
    assert not any(event.event_type is EventType.TOOL_RESULT for event in events)


def test_resume_payload_without_recorded_approval_cannot_execute():
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    orchestrator = AgentOrchestrator(state, ScriptedModel(_cancel()))
    paused = orchestrator.start("Cancel C1", _session())
    checkpoint_config = orchestrator._config(paused.thread_id)

    output = orchestrator._graph.invoke(
        Command(
            resume={
                "confirmation_id": paused.challenge.confirmation_id,
                "decision": "approve",
                "marker": "forged",
            }
        ),
        checkpoint_config,
    )

    assert output["route"] == "done"
    assert output["response"] == "Confirmation resume failed closed."
    assert state.inspect_order("C1").status is OrderStatus.PENDING


def test_approved_resume_executes_once_and_preserves_tool_call_id():
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    model = ScriptedModel(_cancel(), {"type": "final", "response": "Cancelled."})
    orchestrator = AgentOrchestrator(state, model)
    paused = orchestrator.start("Cancel C1", _session())
    before_resume_calls = len(model.calls)

    result = orchestrator.decide(
        paused.thread_id,
        paused.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
    )

    assert result.status is RunStatus.COMPLETED
    assert state.inspect_order("C1").status is OrderStatus.CANCELLED
    assert len(model.calls) == before_resume_calls + 1
    events = orchestrator.trace_events(paused.thread_id)
    calls = [event for event in events if event.event_type is EventType.TOOL_CALL]
    correlated = [
        event
        for event in events
        if getattr(event, "tool_call_id", None) == calls[0].tool_call_id
    ]
    assert len(calls) == 1
    assert all(event.tool_call_id == calls[0].tool_call_id for event in correlated)
    assert sum(event.event_type is EventType.TOOL_RESULT for event in events) == 1


def test_pre_dispatch_trace_failure_cleans_consumed_confirmation(monkeypatch):
    state = _state(
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    orchestrator = AgentOrchestrator(state, ScriptedModel(_refund()))
    paused = orchestrator.start("Refund R1", _session())
    execution = orchestrator._executions[paused.thread_id]
    capability = execution.capability
    assert capability is not None
    assert execution.pending_request is not None

    refund_calls = []
    original_refund = orchestrator._tools.refund_order

    def count_refund(*args, **kwargs):
        refund_calls.append(args)
        return original_refund(*args, **kwargs)

    monkeypatch.setattr(orchestrator._tools, "refund_order", count_refund)
    recorder = execution.recorder
    assert recorder is not None
    original_record = recorder.record
    callback_error = RuntimeError("trace sink unavailable")

    def fail_on_policy_decision(event):
        if isinstance(event, PolicyDecisionEvent):
            raise callback_error
        return original_record(event)

    monkeypatch.setattr(recorder, "record", fail_on_policy_decision)
    with pytest.raises(RuntimeError) as raised:
        orchestrator.decide(
            paused.thread_id,
            paused.challenge.confirmation_id,
            _session(),
            HumanDecision.APPROVE,
        )
    assert raised.value is callback_error

    assert refund_calls == []
    assert state.inspect_order("R1").status is OrderStatus.SHIPPED
    assert state.inspect_refunds() == ()
    assert state.inspect_idempotency_records() == ()
    assert execution.capability is None
    assert execution.active_confirmation_id == ""

    replay = orchestrator._gate.handle(
        ToolRequest(
            operation="refund_order",
            order_id="R1",
            amount=Decimal("49.95"),
            currency="USD",
            idempotency_key=execution.pending_request.idempotency_key,
        ),
        _session(),
        confirmation_capability=capability,
    )
    assert replay.policy.decision is PolicyOutcome.BLOCKED
    assert replay.policy.reason_code == "confirmation_reused"
    assert refund_calls == []
    assert state.inspect_order("R1").status is OrderStatus.SHIPPED
    assert state.inspect_refunds() == ()
    assert state.inspect_idempotency_records() == ()

    monkeypatch.setattr(recorder, "record", original_record)
    orchestrator.close(paused.thread_id)
    assert paused.thread_id not in orchestrator._executions


def test_rejected_confirmation_does_not_execute():
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    orchestrator = AgentOrchestrator(
        state,
        ScriptedModel(_cancel(), {"type": "final", "response": "Not cancelled."}),
    )
    paused = orchestrator.start("Cancel C1", _session())

    result = orchestrator.decide(
        paused.thread_id,
        paused.challenge.confirmation_id,
        _session(),
        HumanDecision.REJECT,
    )

    assert result.status is RunStatus.COMPLETED
    assert state.inspect_order("C1").status is OrderStatus.PENDING
    events = orchestrator.trace_events(paused.thread_id)
    assert any(
        event.event_type is EventType.HUMAN_CONFIRMATION
        and event.status is ConfirmationStatus.REJECTED
        for event in events
    )
    assert any(
        event.event_type is EventType.POLICY_DECISION
        and event.reason_code == "confirmation_rejected"
        for event in events
    )


def test_expired_confirmation_fails_closed_without_execution():
    clock = FakeClock()
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    orchestrator = AgentOrchestrator(state, ScriptedModel(_cancel()), clock=clock)
    paused = orchestrator.start("Cancel C1", _session())
    clock.advance(CONFIRMATION_TTL_SECONDS + 1)

    result = orchestrator.decide(
        paused.thread_id,
        paused.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
    )

    assert result.status is RunStatus.COMPLETED
    assert state.inspect_order("C1").status is OrderStatus.PENDING
    assert any(
        event.event_type is EventType.HUMAN_CONFIRMATION
        and event.status is ConfirmationStatus.EXPIRED
        for event in orchestrator.trace_events(paused.thread_id)
    )


def test_missing_private_continuation_fails_closed():
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    orchestrator = AgentOrchestrator(state, ScriptedModel(_cancel()))
    paused = orchestrator.start("Cancel C1", _session())
    del orchestrator._executions[paused.thread_id]

    result = orchestrator.decide(
        paused.thread_id,
        paused.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
    )

    assert result.status is RunStatus.FAILED_CLOSED
    assert state.inspect_order("C1").status is OrderStatus.PENDING


def test_stale_confirmation_cannot_execute():
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    orchestrator = AgentOrchestrator(state, ScriptedModel(_cancel()))
    paused = orchestrator.start("Cancel C1", _session())
    orchestrator._tools.cancel_order("C1")
    before = state.inspect_order("C1")

    result = orchestrator.decide(
        paused.thread_id,
        paused.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
    )

    assert result.status is RunStatus.COMPLETED
    assert state.inspect_order("C1") == before
    assert state.inspect_refunds() == ()
    execution = orchestrator._executions[paused.thread_id]
    assert execution.capability is None
    assert execution.active_confirmation_id == ""
    events = orchestrator.trace_events(paused.thread_id)
    assert any(
        event.event_type is EventType.HUMAN_CONFIRMATION
        and event.status is ConfirmationStatus.CONSUMED
        for event in events
    )
    assert any(
        event.event_type is EventType.POLICY_DECISION
        and event.reason_code == "confirmation_state_changed"
        for event in events
    )
    assert not any(event.event_type is EventType.TOOL_RESULT for event in events)
    assert events[-1].event_type is EventType.FINAL_RESPONSE
    orchestrator.close(paused.thread_id)
    assert paused.thread_id not in orchestrator._executions


def test_authentication_failure_is_retryable_for_pending_confirmation():
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    orchestrator = AgentOrchestrator(state, ScriptedModel(_cancel()))
    paused = orchestrator.start("Cancel C1", _session())

    invalid_attempt = orchestrator.decide(
        paused.thread_id,
        paused.challenge.confirmation_id,
        _session("user_b"),
        HumanDecision.APPROVE,
    )

    assert invalid_attempt.status is RunStatus.FAILED_CLOSED
    execution = orchestrator._executions[paused.thread_id]
    assert execution.capability is not None
    assert execution.decision_status == ""
    assert state.inspect_order("C1").status is OrderStatus.PENDING

    valid_attempt = orchestrator.decide(
        paused.thread_id,
        paused.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
    )
    assert valid_attempt.status is RunStatus.COMPLETED
    assert state.inspect_order("C1").status is OrderStatus.CANCELLED


def test_timeout_before_commit_requires_fresh_confirmation_and_old_capability_fails(
    monkeypatch,
):
    state = _state(
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    orchestrator = AgentOrchestrator(
        state,
        ScriptedModel(_refund(), {"type": "final", "response": "Refunded."}),
    )
    original = orchestrator._tools.refund_order
    calls = []

    def timeout_once(order_id, key, *, amount, currency, fault=FaultMode.NONE):
        calls.append(key)
        if len(calls) == 1:
            return original(
                order_id,
                key,
                amount=amount,
                currency=currency,
                fault=FaultMode.TIMEOUT_BEFORE_COMMIT,
            )
        return original(
            order_id,
            key,
            amount=amount,
            currency=currency,
            fault=fault,
        )

    monkeypatch.setattr(orchestrator._tools, "refund_order", timeout_once)
    first = orchestrator.start("Refund R1", _session())
    first_capability = orchestrator._executions[first.thread_id].capability
    second = orchestrator.decide(
        first.thread_id,
        first.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
    )

    assert second.status is RunStatus.PAUSED
    assert state.inspect_order("R1").status is OrderStatus.SHIPPED
    assert state.inspect_refunds() == ()
    assert state.inspect_idempotency_records() == ()
    old_attempt = orchestrator.decide(
        first.thread_id,
        first.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
    )
    assert old_attempt.status is RunStatus.FAILED_CLOSED
    assert orchestrator._executions[first.thread_id].capability != first_capability

    success = orchestrator.decide(
        second.thread_id,
        second.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
    )

    assert success.status is RunStatus.COMPLETED
    assert calls[0] == calls[1]
    assert state.inspect_order("R1").status is OrderStatus.REFUNDED
    assert len(state.inspect_refunds()) == 1
    assert len(state.inspect_idempotency_records()) == 1


def test_timeout_after_commit_is_verified_by_gate_replay_without_second_commit(
    monkeypatch,
):
    state = _state(
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    orchestrator = AgentOrchestrator(
        state,
        ScriptedModel(_refund(), {"type": "final", "response": "Refund complete."}),
    )
    original = orchestrator._tools.refund_order
    calls = []

    def timeout_after_commit_once(order_id, key, *, amount, currency, fault=FaultMode.NONE):
        calls.append(key)
        if len(calls) == 1:
            return original(
                order_id,
                key,
                amount=amount,
                currency=currency,
                fault=FaultMode.TIMEOUT_AFTER_COMMIT,
            )
        return original(
            order_id,
            key,
            amount=amount,
            currency=currency,
            fault=fault,
        )

    monkeypatch.setattr(orchestrator._tools, "refund_order", timeout_after_commit_once)
    paused = orchestrator.start("Refund R1", _session())
    result = orchestrator.decide(
        paused.thread_id,
        paused.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
    )

    assert result.status is RunStatus.COMPLETED
    assert calls == [calls[0], calls[0]]
    assert state.inspect_order("R1").status is OrderStatus.REFUNDED
    assert len(state.inspect_refunds()) == 1
    assert len(state.inspect_idempotency_records()) == 1
    policy_events = [
        event
        for event in orchestrator.trace_events(paused.thread_id)
        if event.event_type is EventType.POLICY_DECISION
    ]
    assert any(event.decision is PolicyOutcome.REPLAY for event in policy_events)
    results = [
        event
        for event in orchestrator.trace_events(paused.thread_id)
        if event.event_type is EventType.TOOL_RESULT
    ]
    events = orchestrator.trace_events(paused.thread_id)
    assert [event.status for event in results] == [
        ToolStatus.TIMEOUT,
        ToolStatus.REPLAYED,
    ]
    assert results[0].state_changed is True
    assert results[1].state_changed is False
    for result in results:
        result_index = events.index(result)
        attempt_policy = max(
            index
            for index, event in enumerate(events[:result_index])
            if event.event_type is EventType.POLICY_DECISION
            and event.tool_call_id == result.tool_call_id
        )
        assert attempt_policy < result_index
        if result is results[0]:
            consumed_index = next(
                index
                for index, event in enumerate(events[:result_index])
                if event.event_type is EventType.HUMAN_CONFIRMATION
                and event.tool_call_id == result.tool_call_id
                and event.status is ConfirmationStatus.CONSUMED
            )
            assert attempt_policy < consumed_index < result_index


def test_duplicate_response_fault_commits_once(monkeypatch):
    state = _state(
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    orchestrator = AgentOrchestrator(
        state,
        ScriptedModel(_refund(), {"type": "final", "response": "Refund complete."}),
    )
    original = orchestrator._tools.refund_order

    def duplicate_response(order_id, key, *, amount, currency, fault=FaultMode.NONE):
        return original(
            order_id,
            key,
            amount=amount,
            currency=currency,
            fault=FaultMode.DUPLICATE_RESPONSE,
        )

    monkeypatch.setattr(orchestrator._tools, "refund_order", duplicate_response)
    paused = orchestrator.start("Refund R1", _session())
    result = orchestrator.decide(
        paused.thread_id,
        paused.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
    )

    assert result.status is RunStatus.COMPLETED
    assert state.inspect_order("R1").status is OrderStatus.REFUNDED
    assert len(state.inspect_refunds()) == 1
    assert len(state.inspect_idempotency_records()) == 1
    result_events = [
        event
        for event in orchestrator.trace_events(paused.thread_id)
        if event.event_type is EventType.TOOL_RESULT
    ]
    assert len(result_events) == 1
    assert result_events[0].result["response_delivery_count"] == 2
    assert result_events[0].state_changed is True


def test_retry_count_is_bounded_and_preserves_refund_key(monkeypatch):
    state = _state(
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    orchestrator = AgentOrchestrator(
        state,
        ScriptedModel(_refund(), {"type": "final", "response": "Could not confirm."}),
    )
    original = orchestrator._tools.refund_order
    keys = []

    def always_timeout_before(order_id, key, *, amount, currency, fault=FaultMode.NONE):
        keys.append(key)
        return original(
            order_id,
            key,
            amount=amount,
            currency=currency,
            fault=FaultMode.TIMEOUT_BEFORE_COMMIT,
        )

    monkeypatch.setattr(orchestrator._tools, "refund_order", always_timeout_before)
    first = orchestrator.start("Refund R1", _session())
    second = orchestrator.decide(
        first.thread_id,
        first.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
    )
    assert second.status is RunStatus.PAUSED
    finished = orchestrator.decide(
        second.thread_id,
        second.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
    )

    assert finished.status is RunStatus.COMPLETED
    assert MAX_AUTOMATIC_TIMEOUT_RETRIES == 1
    assert len(keys) == 2
    assert keys[0] == keys[1]
    assert state.inspect_order("R1").status is OrderStatus.SHIPPED
    assert state.inspect_refunds() == ()
    assert state.inspect_idempotency_records() == ()


def test_maximum_tool_rounds_are_enforced():
    state = _state(
        Order("O1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    model = ScriptedModel(
        *[
            {
                "type": "tool_call",
                "operation": "get_order",
                "order_id": "O1",
            }
            for _ in range(MAX_TOOL_ROUNDS + 2)
        ]
    )
    orchestrator = AgentOrchestrator(state, model)

    result = orchestrator.start("Check O1 repeatedly", _session())

    assert result.status is RunStatus.COMPLETED
    assert len(model.calls) == MAX_TOOL_ROUNDS
    assert sum(
        event.event_type is EventType.TOOL_CALL
        for event in orchestrator.trace_events(result.thread_id)
    ) == MAX_TOOL_ROUNDS


def test_trace_is_sequential_correlated_and_has_no_confirmation_secret(tmp_path):
    state = _state(
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    model = ScriptedModel(_refund(), {"type": "final", "response": "Refund complete."})
    orchestrator = AgentOrchestrator(state, model)
    paused = orchestrator.start("Refund R1", _session(), run_id="run-7")
    runtime = orchestrator._executions[paused.thread_id]
    raw_capability = runtime.capability
    checkpoint = orchestrator._graph.get_state(
        orchestrator._config(paused.thread_id)
    ).values

    assert raw_capability not in repr(checkpoint)
    assert raw_capability not in repr(model.calls)
    assert raw_capability not in repr(orchestrator.trace_events(paused.thread_id))

    result = orchestrator.decide(
        paused.thread_id,
        paused.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
    )
    assert result.status is RunStatus.COMPLETED

    events = orchestrator.trace_events(paused.thread_id)
    assert [event.step for event in events] == list(range(1, len(events) + 1))
    assert {event.trace_id for event in events} == {events[0].trace_id}
    assert {event.run_id for event in events} == {"run-7"}
    tool_calls = [event for event in events if event.event_type is EventType.TOOL_CALL]
    assert len(tool_calls) == 1
    correlated = [
        event
        for event in events
        if hasattr(event, "tool_call_id")
    ]
    assert all(event.tool_call_id == tool_calls[0].tool_call_id for event in correlated)
    statuses = [
        event.status
        for event in events
        if event.event_type is EventType.HUMAN_CONFIRMATION
    ]
    assert statuses == [
        ConfirmationStatus.REQUESTED,
        ConfirmationStatus.CONFIRMED,
        ConfirmationStatus.CONSUMED,
    ]
    confirmed_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type is EventType.HUMAN_CONFIRMATION
        and event.status is ConfirmationStatus.CONFIRMED
    )
    consumed_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type is EventType.HUMAN_CONFIRMATION
        and event.status is ConfirmationStatus.CONSUMED
    )
    allowed_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type is EventType.POLICY_DECISION
        and event.decision is PolicyOutcome.ALLOWED
    )
    tool_result_index = next(
        index
        for index, event in enumerate(events)
        if event.event_type is EventType.TOOL_RESULT
    )
    assert confirmed_index < allowed_index < consumed_index < tool_result_index
    assert all(
        "idempotency_key" not in event.args
        for event in events
        if event.event_type in {
            EventType.TOOL_CALL,
            EventType.HUMAN_CONFIRMATION,
        }
    )

    path = tmp_path / "agentguard.json"
    orchestrator.export_trace(paused.thread_id, str(path))
    serialized = path.read_text(encoding="utf-8")
    assert raw_capability not in serialized
    exported = json.loads(serialized)
    assert [event["step"] for event in exported["events"]] == list(
        range(1, len(events) + 1)
    )


def test_untrusted_model_secrets_are_not_checkpointed_or_traced():
    state = _state(
        Order("O1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    model = ScriptedModel(
        {
            "type": "tool_call",
            "operation": "get_order",
            "order_id": "O1",
            "confirmation_capability": "model-supplied-secret",
            "idempotency_key": "model-supplied-key",
        }
    )
    orchestrator = AgentOrchestrator(state, model)

    result = orchestrator.start("Check O1", _session())

    assert result.status is RunStatus.COMPLETED
    checkpoint = orchestrator._graph.get_state(
        orchestrator._config(result.thread_id)
    ).values
    events = orchestrator.trace_events(result.thread_id)
    assert "model-supplied-secret" not in repr(checkpoint)
    assert "model-supplied-secret" not in repr(events)
    assert "model-supplied-key" not in repr(checkpoint)
    assert "model-supplied-key" not in repr(events)
    assert result.response == "I could not process that tool request."
    llm_event = next(
        event for event in events if event.event_type is EventType.LLM_OUTPUT
    )
    assert llm_event.parsed_plan == {
        "type": "tool_call",
        "operation": "get_order",
        "order_id": "O1",
    }


def test_unknown_model_fields_are_excluded_from_exported_trace(tmp_path):
    state = _state(
        Order("O1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    marker = "PRIVATE-UNTRUSTED-MODEL-TEXT"
    orchestrator = AgentOrchestrator(
        state,
        ScriptedModel(
            {
                "type": "tool_call",
                "operation": "get_order",
                "order_id": "O1",
                "note": marker,
                "debug": {"detail": marker},
            }
        ),
    )

    result = orchestrator.start("Check O1", _session())
    output_path = tmp_path / "safe-trace.json"
    orchestrator.export_trace(result.thread_id, output_path)

    exported = output_path.read_text(encoding="utf-8")
    assert marker not in exported
    payload = json.loads(exported)
    llm_event = next(
        event for event in payload["events"] if event["event_type"] == "llm_output"
    )
    assert llm_event["parsed_plan"] == {
        "type": "tool_call",
        "operation": "get_order",
        "order_id": "O1",
    }


def test_malformed_unhashable_operation_fails_safely_and_traces_final_response():
    state = _state(
        Order("O1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    orchestrator = AgentOrchestrator(
        state,
        ScriptedModel(
            {
                "type": "tool_call",
                "operation": ["get_order"],
                "order_id": "O1",
            }
        ),
    )

    result = orchestrator.start("Check O1", _session())

    assert result.status is RunStatus.COMPLETED
    assert result.response == "I could not process that tool request."
    events = orchestrator.trace_events(result.thread_id)
    assert events[-1].event_type is EventType.FINAL_RESPONSE
    assert not any(event.event_type is EventType.TOOL_CALL for event in events)
    assert state.inspect_order("O1").status is OrderStatus.PENDING


def test_decide_is_not_exposed_in_model_messages():
    state = _state(
        Order("O1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    model = ScriptedModel({"type": "final", "response": "Done."})
    orchestrator = AgentOrchestrator(state, model)

    orchestrator.start("Check O1", _session())

    assert len(model.calls) == 1
    assert model.calls[0][0]["role"] == "user"
    assert "decide" not in repr(model.calls)
    assert "record_human_decision" not in repr(model.calls)
    assert "SessionContext" not in repr(model.calls)


def test_baseline_a_requires_explicit_evaluation_configuration():
    state = _state(
        Order("O1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    model = ScriptedModel({"type": "final", "response": "Done."})

    with pytest.raises(ValueError, match="for_evaluation"):
        AgentOrchestrator(state, model, group=ExperimentGroup.BASELINE_A)

    with pytest.raises(TypeError):
        AgentOrchestrator(
            state,
            model,
            group=ExperimentGroup.BASELINE_A,
            evaluation_only=True,
        )

    evaluation_orchestrator = AgentOrchestrator.for_evaluation(
        state,
        model,
    )
    assert evaluation_orchestrator._gate is None
    assert evaluation_orchestrator._group is ExperimentGroup.BASELINE_A


def test_baseline_a_keeps_task_two_idempotency_across_committed_timeout(
    monkeypatch,
):
    state = _state(
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    orchestrator = AgentOrchestrator.for_evaluation(
        state,
        ScriptedModel(
            _refund(),
            {"type": "final", "response": "Refund complete."},
        ),
    )
    original = orchestrator._tools.refund_order
    attempts = []

    def timeout_after_commit_once(order_id, key, *, amount, currency, fault=FaultMode.NONE):
        attempts.append(key)
        if len(attempts) == 1:
            return original(
                order_id,
                key,
                amount=amount,
                currency=currency,
                fault=FaultMode.TIMEOUT_AFTER_COMMIT,
            )
        return original(
            order_id,
            key,
            amount=amount,
            currency=currency,
            fault=fault,
        )

    monkeypatch.setattr(orchestrator._tools, "refund_order", timeout_after_commit_once)

    result = orchestrator.start("Refund R1", _session())

    assert result.status is RunStatus.COMPLETED
    assert len(attempts) == 2
    assert attempts[0] == attempts[1]
    assert state.inspect_order("R1").status is OrderStatus.REFUNDED
    assert len(state.inspect_refunds()) == 1
    assert len(state.inspect_idempotency_records()) == 1
