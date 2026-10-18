from dataclasses import replace
from decimal import Decimal

import pytest

from policy_gate import (
    CONFIRMATION_TTL_SECONDS,
    ConfirmationError,
    HumanDecision,
    PolicyGate,
    SessionContext,
    ToolRequest,
    _order_state_hash,
)
from simulated_tools import (
    AlreadyRefunded,
    FaultMode,
    InMemoryOrderState,
    InvalidTransition,
    Order,
    OrderStatus,
    SimulatedOrderTools,
    ToolTimeoutError,
)
from trace_events import PolicyOutcome


class FakeClock:
    def __init__(self):
        self.value = 100.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def _state(*orders):
    return InMemoryOrderState(orders)


def _session(user_id="user_a"):
    return SessionContext(user_id)


def _cancel_request(order_id="C1", *, args_source="user_text"):
    return ToolRequest(
        operation="cancel_order",
        order_id=order_id,
        args_source=args_source,
    )


def _refund_request(
    order_id="R1",
    *,
    key="key-1",
    amount=Decimal("49.95"),
    currency="USD",
    args_source="user_text",
):
    return ToolRequest(
        operation="refund_order",
        order_id=order_id,
        amount=amount,
        currency=currency,
        idempotency_key=key,
        args_source=args_source,
    )


def _issue_and_approve(gate, request, session):
    held = gate.handle(request, session)
    assert held.policy.decision is PolicyOutcome.BLOCKED
    assert held.policy.reason_code == "confirmation_required"
    assert held.challenge is not None
    capability = gate.take_capability_handoff(held.challenge.confirmation_id)
    gate.record_human_decision(
        held.challenge.confirmation_id,
        session,
        HumanDecision.APPROVE,
        capability=capability,
    )
    return held.challenge, capability


def test_missing_and_invalid_authentication_are_blocked():
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    gate = PolicyGate(state)

    missing = gate.handle(_cancel_request(), None)
    invalid = gate.handle(_cancel_request(), SessionContext(""))
    wrong_type = gate.handle(_cancel_request(), object())

    assert missing.policy.decision is PolicyOutcome.BLOCKED
    assert missing.policy.reason_code == "authentication_required"
    assert invalid.policy == missing.policy
    assert wrong_type.policy == missing.policy
    assert state.inspect_order("C1").status is OrderStatus.PENDING


def test_missing_and_cross_user_orders_have_indistinguishable_denials():
    state = _state(
        Order("C1", "user_b", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    gate = PolicyGate(state)

    missing = gate.handle(_cancel_request("MISSING"), _session("user_a"))
    cross_user = gate.handle(_cancel_request("C1"), _session("user_a"))

    assert missing.policy == cross_user.policy
    assert missing.policy.reason_code == "order_unavailable"
    assert missing.policy.reason == "Order unavailable"
    assert missing.policy.checked_owner_id == ""
    assert missing.result is cross_user.result is None
    assert state.inspect_order("C1").status is OrderStatus.PENDING
    assert state.inspect_refunds() == ()


def test_authorized_get_order_succeeds_without_exposing_owner():
    order = Order("O1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    state = _state(order)

    result = PolicyGate(state).handle(
        ToolRequest("get_order", "O1"),
        _session(),
    )

    assert result.policy.decision is PolicyOutcome.ALLOWED
    assert result.result.value.order_id == "O1"
    assert result.result.value.status is OrderStatus.PENDING
    assert not hasattr(result.result.value, "owner_id")
    assert state.inspect_order("O1") == order


def test_gate_rejects_tools_bound_to_a_different_authoritative_state():
    gate_state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    tools_state = _state(
        Order("C1", "user_b", Decimal("10"), "USD", OrderStatus.PENDING)
    )

    with pytest.raises(ValueError, match="same order state"):
        PolicyGate(gate_state, SimulatedOrderTools(tools_state))

    assert gate_state.inspect_order("C1").status is OrderStatus.PENDING
    assert tools_state.inspect_order("C1").status is OrderStatus.PENDING
    assert tools_state.inspect_refunds() == ()


def test_unauthorized_refund_replay_is_denied_before_idempotency_lookup():
    state = _state(
        Order("R1", "user_b", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    tools = SimulatedOrderTools(state)
    tools.refund_order(
        "R1",
        "key-1",
        amount=Decimal("49.95"),
        currency="USD",
    )
    before = (
        state.inspect_order("R1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    )

    result = PolicyGate(state, tools).handle(
        _refund_request(),
        _session("user_a"),
    )

    assert result.policy.decision is PolicyOutcome.BLOCKED
    assert result.policy.reason_code == "order_unavailable"
    assert result.result is None
    assert (
        state.inspect_order("R1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    ) == before


@pytest.mark.parametrize(
    ("tool_request", "order"),
    [
        (
            _cancel_request(),
            Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING),
        ),
        (
            _refund_request(),
            Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED),
        ),
    ],
)
def test_cancel_and_refund_require_confirmation_without_mutation(tool_request, order):
    state = _state(order)
    before = (
        state.inspect_order(order.order_id),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    )
    gate = PolicyGate(state)

    result = gate.handle(tool_request, _session())

    assert result.policy.decision is PolicyOutcome.BLOCKED
    assert result.policy.reason_code == "confirmation_required"
    assert result.result is None
    assert result.challenge is not None
    assert result.challenge.expires_at
    assert result.challenge.order_state_hash == _order_state_hash(order)
    assert not hasattr(result.challenge, "capability")
    assert (
        state.inspect_order(order.order_id),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    ) == before


def test_capability_handoff_is_secret_single_use_and_not_approval():
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    gate = PolicyGate(state)
    hold = gate.handle(_cancel_request(), _session())
    capability = gate.take_capability_handoff(hold.challenge.confirmation_id)

    assert capability not in repr(hold.challenge)
    with pytest.raises(ConfirmationError, match="capability_handoff_unavailable"):
        gate.take_capability_handoff(hold.challenge.confirmation_id)

    resumed = gate.handle(
        _cancel_request(),
        _session(),
        confirmation_capability=capability,
    )
    assert resumed.policy.decision is PolicyOutcome.BLOCKED
    assert resumed.policy.reason_code == "confirmation_not_approved"
    assert state.inspect_order("C1").status is OrderStatus.PENDING


def test_explicit_approval_is_required_before_confirmed_cancel_executes():
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    gate = PolicyGate(state)
    hold = gate.handle(_cancel_request(), _session())
    capability = gate.take_capability_handoff(hold.challenge.confirmation_id)
    gate.record_human_decision(
        hold.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
        capability=capability,
    )

    result = gate.handle(
        _cancel_request(),
        _session(),
        confirmation_capability=capability,
    )

    assert result.policy.decision is PolicyOutcome.ALLOWED
    assert result.execution_error is None
    assert result.result.value.status is OrderStatus.CANCELLED
    assert state.inspect_order("C1").status is OrderStatus.CANCELLED


def test_explicit_rejection_never_authorizes_execution():
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    gate = PolicyGate(state)
    hold = gate.handle(_cancel_request(), _session())
    capability = gate.take_capability_handoff(hold.challenge.confirmation_id)
    gate.record_human_decision(
        hold.challenge.confirmation_id,
        _session(),
        HumanDecision.REJECT,
        capability=capability,
    )

    result = gate.handle(
        _cancel_request(),
        _session(),
        confirmation_capability=capability,
    )

    assert result.policy.decision is PolicyOutcome.BLOCKED
    assert result.policy.reason_code == "confirmation_rejected"
    assert state.inspect_order("C1").status is OrderStatus.PENDING


def test_forged_capability_is_rejected():
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    gate = PolicyGate(state)
    gate.handle(_cancel_request(), _session())

    result = gate.handle(
        _cancel_request(),
        _session(),
        confirmation_capability="forged-secret",
    )

    assert result.policy.reason_code == "confirmation_invalid"
    assert state.inspect_order("C1").status is OrderStatus.PENDING


def test_confirmation_expires_and_cannot_execute():
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    clock = FakeClock()
    gate = PolicyGate(state, clock=clock)
    hold = gate.handle(_cancel_request(), _session())
    capability = gate.take_capability_handoff(hold.challenge.confirmation_id)
    gate.record_human_decision(
        hold.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
        capability=capability,
    )
    clock.advance(CONFIRMATION_TTL_SECONDS + 1)

    result = gate.handle(
        _cancel_request(),
        _session(),
        confirmation_capability=capability,
    )

    assert result.policy.reason_code == "confirmation_expired"
    assert state.inspect_order("C1").status is OrderStatus.PENDING


def test_confirmation_rejects_user_parameter_and_operation_mismatches():
    orders = (
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING),
        Order("C2", "user_a", Decimal("11"), "USD", OrderStatus.PENDING),
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED),
    )
    state = _state(*orders)
    gate = PolicyGate(state)
    hold = gate.handle(_cancel_request("C1"), _session())
    capability = gate.take_capability_handoff(hold.challenge.confirmation_id)
    gate.record_human_decision(
        hold.challenge.confirmation_id,
        _session(),
        HumanDecision.APPROVE,
        capability=capability,
    )

    parameter_mismatch = gate.handle(
        _cancel_request("C2"),
        _session(),
        confirmation_capability=capability,
    )
    operation_mismatch = gate.handle(
        _refund_request("R1"),
        _session(),
        confirmation_capability=capability,
    )
    with pytest.raises(ConfirmationError, match="confirmation_user_mismatch"):
        gate.record_human_decision(
            hold.challenge.confirmation_id,
            _session("user_b"),
            HumanDecision.APPROVE,
            capability=capability,
        )

    assert parameter_mismatch.policy.reason_code == "confirmation_mismatch"
    assert operation_mismatch.policy.reason_code == "confirmation_mismatch"
    assert state.inspect_order("C1").status is OrderStatus.PENDING
    assert state.inspect_order("C2").status is OrderStatus.PENDING
    assert state.inspect_order("R1").status is OrderStatus.SHIPPED


@pytest.mark.parametrize(
    "mismatch_kwargs",
    [
        {"amount": Decimal("48.00")},
        {"currency": "EUR"},
        {"key": "key-2"},
    ],
    ids=["amount", "currency", "idempotency_key"],
)
def test_refund_confirmation_rejects_request_binding_mismatches(mismatch_kwargs):
    state = _state(
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    gate = PolicyGate(state)
    request = _refund_request()
    _, capability = _issue_and_approve(gate, request, _session())
    before = (
        state.inspect_order("R1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    )

    result = gate.handle(
        _refund_request(**mismatch_kwargs),
        _session(),
        confirmation_capability=capability,
    )

    assert result.policy.decision is PolicyOutcome.BLOCKED
    assert result.policy.reason_code == "confirmation_mismatch"
    assert result.result is None
    assert result.execution_error is None
    assert (
        state.inspect_order("R1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    ) == before


def test_confirmation_reuse_is_rejected_after_successful_execution():
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    gate = PolicyGate(state)
    _, capability = _issue_and_approve(gate, _cancel_request(), _session())

    first = gate.handle(
        _cancel_request(),
        _session(),
        confirmation_capability=capability,
    )
    second = gate.handle(
        _cancel_request(),
        _session(),
        confirmation_capability=capability,
    )

    assert first.policy.decision is PolicyOutcome.ALLOWED
    assert second.policy.reason_code == "confirmation_reused"
    assert state.inspect_order("C1").status is OrderStatus.CANCELLED


def test_order_state_change_invalidates_confirmation():
    order = Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    state = _state(order)
    gate = PolicyGate(state)
    _, capability = _issue_and_approve(gate, _cancel_request(), _session())
    SimulatedOrderTools(state).cancel_order("C1")

    result = gate.handle(
        _cancel_request(),
        _session(),
        confirmation_capability=capability,
    )

    assert result.policy.reason_code == "confirmation_state_changed"
    assert state.inspect_order("C1").status is OrderStatus.CANCELLED


def test_external_order_status_change_invalidates_confirmation():
    order = Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    state = _state(order)
    gate = PolicyGate(state)
    _, capability = _issue_and_approve(gate, _cancel_request(), _session())
    state.apply_external_order_status_change("C1", OrderStatus.PROCESSING)

    result = gate.handle(
        _cancel_request(),
        _session(),
        confirmation_capability=capability,
    )

    assert result.policy.reason_code == "confirmation_state_changed"
    assert result.result is None
    assert state.inspect_order("C1").status is OrderStatus.PROCESSING
    assert state.inspect_refunds() == ()
    assert state.inspect_idempotency_records() == ()


def test_order_state_hash_is_stable_and_covers_all_required_fields():
    base = Order("R1", "user_a", Decimal("49.950"), "USD", OrderStatus.SHIPPED)
    equivalent_amount = replace(base, total_amount=Decimal("49.95"))

    assert _order_state_hash(base) == _order_state_hash(equivalent_amount)
    assert _order_state_hash(replace(base, order_id="R2")) != _order_state_hash(base)
    assert _order_state_hash(replace(base, owner_id="user_b")) != _order_state_hash(base)
    assert _order_state_hash(replace(base, status=OrderStatus.DELIVERED)) != _order_state_hash(base)
    assert _order_state_hash(replace(base, total_amount=Decimal("50"))) != _order_state_hash(base)
    assert _order_state_hash(replace(base, currency="EUR")) != _order_state_hash(base)


def test_exact_match_replay_is_authorized_without_confirmation_or_mutation():
    state = _state(
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    tools = SimulatedOrderTools(state)
    original = tools.refund_order(
        "R1",
        "key-1",
        amount=Decimal("49.95"),
        currency="USD",
    )
    before = (
        state.inspect_order("R1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    )

    replay = PolicyGate(state, tools).handle(_refund_request(), _session())

    assert replay.policy.decision is PolicyOutcome.REPLAY
    assert replay.result.value == original.value
    assert replay.result.replayed is True
    assert replay.challenge is None
    assert (
        state.inspect_order("R1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    ) == before


def test_same_key_conflict_and_different_key_refunded_reject_without_challenge():
    state = _state(
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    tools = SimulatedOrderTools(state)
    tools.refund_order(
        "R1",
        "key-1",
        amount=Decimal("49.95"),
        currency="USD",
    )
    gate = PolicyGate(state, tools)

    conflict = gate.handle(
        _refund_request(amount=Decimal("48")),
        _session(),
    )
    duplicate = gate.handle(
        _refund_request(key="key-2"),
        _session(),
    )

    assert conflict.policy.reason_code == "idempotency_conflict"
    assert duplicate.policy.reason_code == "already_refunded"
    assert conflict.challenge is duplicate.challenge is None
    assert len(state.inspect_refunds()) == 1
    assert len(state.inspect_idempotency_records()) == 1


def test_timeout_before_commit_requires_fresh_confirmation(monkeypatch):
    state = _state(
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    tools = SimulatedOrderTools(state)
    gate = PolicyGate(state, tools)
    request = _refund_request()
    first_challenge, first_capability = _issue_and_approve(gate, request, _session())
    real_refund_order = tools.refund_order

    def timeout_before_commit(order_id, key, *, amount, currency, fault=FaultMode.NONE):
        return real_refund_order(
            order_id,
            key,
            amount=amount,
            currency=currency,
            fault=FaultMode.TIMEOUT_BEFORE_COMMIT,
        )

    with monkeypatch.context() as patch:
        patch.setattr(tools, "refund_order", timeout_before_commit)
        result = gate.handle(
            request,
            _session(),
            confirmation_capability=first_capability,
        )

    assert result.policy.decision is PolicyOutcome.ALLOWED
    assert isinstance(result.execution_error, ToolTimeoutError)
    assert result.execution_error.committed is False
    assert result.execution_error.duration_ms >= 0
    assert state.inspect_order("R1").status is OrderStatus.SHIPPED
    assert state.inspect_refunds() == ()
    assert state.inspect_idempotency_records() == ()

    second_challenge, second_capability = _issue_and_approve(gate, request, _session())
    assert second_challenge.confirmation_id != first_challenge.confirmation_id
    success = gate.handle(
        request,
        _session(),
        confirmation_capability=second_capability,
    )
    assert success.policy.decision is PolicyOutcome.ALLOWED
    assert success.result is not None
    assert len(state.inspect_refunds()) == 1
    assert len(state.inspect_idempotency_records()) == 1


def test_timeout_before_commit_consumes_confirmation_and_retry_requires_fresh_approval(
    monkeypatch,
):
    state = _state(
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    tools = SimulatedOrderTools(state)
    gate = PolicyGate(state, tools)
    request = _refund_request()
    _, capability = _issue_and_approve(gate, request, _session())
    real_refund_order = tools.refund_order

    def timeout_before_commit(order_id, key, *, amount, currency, fault=FaultMode.NONE):
        return real_refund_order(
            order_id,
            key,
            amount=amount,
            currency=currency,
            fault=FaultMode.TIMEOUT_BEFORE_COMMIT,
        )

    with monkeypatch.context() as patch:
        patch.setattr(tools, "refund_order", timeout_before_commit)
        failed_response = gate.handle(
            request,
            _session(),
            confirmation_capability=capability,
        )

    assert failed_response.policy.decision is PolicyOutcome.ALLOWED
    assert isinstance(failed_response.execution_error, ToolTimeoutError)
    assert failed_response.execution_error.committed is False
    before_retry = (
        state.inspect_order("R1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    )
    assert before_retry[0].status is OrderStatus.SHIPPED
    assert before_retry[1] == ()
    assert before_retry[2] == ()

    retry = gate.handle(
        request,
        _session(),
        confirmation_capability=capability,
    )

    assert retry.policy.decision is PolicyOutcome.BLOCKED
    assert retry.policy.reason_code == "confirmation_reused"
    assert retry.result is None
    assert (
        state.inspect_order("R1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    ) == before_retry

    _, fresh_capability = _issue_and_approve(gate, request, _session())
    successful_retry = gate.handle(
        request,
        _session(),
        confirmation_capability=fresh_capability,
    )

    assert successful_retry.policy.decision is PolicyOutcome.ALLOWED
    assert successful_retry.result is not None
    assert successful_retry.execution_error is None
    assert state.inspect_order("R1").status is OrderStatus.REFUNDED
    assert len(state.inspect_refunds()) == 1
    assert len(state.inspect_idempotency_records()) == 1


def test_timeout_after_commit_retries_as_replay_without_another_mutation(monkeypatch):
    state = _state(
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    tools = SimulatedOrderTools(state)
    gate = PolicyGate(state, tools)
    request = _refund_request()
    _, capability = _issue_and_approve(gate, request, _session())
    real_refund_order = tools.refund_order

    def timeout_after_commit(order_id, key, *, amount, currency, fault=FaultMode.NONE):
        return real_refund_order(
            order_id,
            key,
            amount=amount,
            currency=currency,
            fault=FaultMode.TIMEOUT_AFTER_COMMIT,
        )

    with monkeypatch.context() as patch:
        patch.setattr(tools, "refund_order", timeout_after_commit)
        failed_response = gate.handle(
            request,
            _session(),
            confirmation_capability=capability,
        )

    assert failed_response.policy.decision is PolicyOutcome.ALLOWED
    assert isinstance(failed_response.execution_error, ToolTimeoutError)
    assert failed_response.execution_error.committed is True
    assert failed_response.execution_error.duration_ms >= 0
    assert state.inspect_order("R1").status is OrderStatus.REFUNDED
    assert len(state.inspect_refunds()) == 1
    assert len(state.inspect_idempotency_records()) == 1
    refund = state.inspect_refunds()[0]

    replay = gate.handle(request, _session())

    assert replay.policy.decision is PolicyOutcome.REPLAY
    assert replay.result.value == refund
    assert replay.result.replayed is True
    assert state.inspect_order("R1").status is OrderStatus.REFUNDED
    assert state.inspect_refunds() == (refund,)
    assert len(state.inspect_idempotency_records()) == 1


def test_duplicate_response_keeps_one_business_refund(monkeypatch):
    state = _state(
        Order("R1", "user_a", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    tools = SimulatedOrderTools(state)
    gate = PolicyGate(state, tools)
    request = _refund_request()
    _, capability = _issue_and_approve(gate, request, _session())
    real_refund_order = tools.refund_order

    def duplicate_response(order_id, key, *, amount, currency, fault=FaultMode.NONE):
        return real_refund_order(
            order_id,
            key,
            amount=amount,
            currency=currency,
            fault=FaultMode.DUPLICATE_RESPONSE,
        )

    monkeypatch.setattr(tools, "refund_order", duplicate_response)
    result = gate.handle(
        request,
        _session(),
        confirmation_capability=capability,
    )

    assert result.policy.decision is PolicyOutcome.ALLOWED
    assert result.result.response_delivery_count == 2
    assert result.result.replayed is False
    assert len(state.inspect_refunds()) == 1
    assert len(state.inspect_idempotency_records()) == 1


def test_authorized_typed_tool_failure_is_not_a_policy_block(monkeypatch):
    state = _state(
        Order("C1", "user_a", Decimal("10"), "USD", OrderStatus.PENDING)
    )
    gate = PolicyGate(state)
    _, capability = _issue_and_approve(gate, _cancel_request(), _session())

    def fail_during_state_commit(order):
        raise InvalidTransition("cancel", OrderStatus.PENDING)

    monkeypatch.setattr(state, "_commit_cancellation", fail_during_state_commit)
    result = gate.handle(
        _cancel_request(),
        _session(),
        confirmation_capability=capability,
    )

    assert result.policy.decision is PolicyOutcome.ALLOWED
    assert result.policy.reason_code == "authorized_execution_failed"
    assert isinstance(result.execution_error, InvalidTransition)
    assert isinstance(result.execution_error.duration_ms, int)
    assert result.execution_error.duration_ms >= 0
    assert state.inspect_order("C1").status is OrderStatus.PENDING
