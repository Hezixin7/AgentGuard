from dataclasses import FrozenInstanceError, replace
from decimal import Decimal

import pytest

from simulated_tools import (
    AlreadyRefunded,
    FaultMode,
    IdempotencyConflict,
    InMemoryOrderState,
    InvalidRequest,
    InvalidTransition,
    Order,
    OrderNotFound,
    OrderStatus,
    RefundLookupStatus,
    SimulatedOrderTools,
    ToolError,
    ToolResult,
    ToolTimeoutError,
)


def _state_with(*orders: Order) -> InMemoryOrderState:
    return InMemoryOrderState(orders)


def _tool(state: InMemoryOrderState) -> SimulatedOrderTools:
    return SimulatedOrderTools(state)


def _refund(
    tools: SimulatedOrderTools,
    *,
    order_id: str = "S1",
    key: str = "key-1",
    amount: Decimal = Decimal("49.95"),
    currency: str = "USD",
    fault: FaultMode = FaultMode.NONE,
) -> ToolResult:
    return tools.refund_order(
        order_id,
        key,
        amount=amount,
        currency=currency,
        fault=fault,
    )


def test_get_order_returns_read_only_view_without_mutating_state():
    state = InMemoryOrderState()
    tools = _tool(state)

    result = tools.get_order("P100")

    assert result.operation == "get_order"
    assert result.value.order_id == "P100"
    assert result.value.status is OrderStatus.PENDING
    assert result.duration_ms >= 0
    assert state.inspect_order("P100").status is OrderStatus.PENDING
    assert not hasattr(result.value, "owner_id")


def test_get_order_does_not_perform_ownership_authorization():
    order = Order(
        "S1",
        "user_b",
        Decimal("49.95"),
        "USD",
        OrderStatus.SHIPPED,
    )
    state = _state_with(order)
    tools = _tool(state)

    result = tools.get_order("S1")

    assert result.value.order_id == "S1"
    assert result.value.total_amount == Decimal("49.95")
    assert result.value.currency == "USD"
    assert result.value.status is OrderStatus.SHIPPED
    assert state.inspect_order("S1") == order


def test_returned_order_values_cannot_mutate_authoritative_state():
    order = Order(
        "S1",
        "user_b",
        Decimal("49.95"),
        "USD",
        OrderStatus.SHIPPED,
    )
    state = _state_with(order)
    tools = _tool(state)

    inspected_order = state.inspect_order("S1")
    returned_view = tools.get_order("S1").value

    with pytest.raises(FrozenInstanceError):
        inspected_order.status = OrderStatus.REFUNDED
    with pytest.raises(FrozenInstanceError):
        returned_view.status = OrderStatus.REFUNDED

    assert state.inspect_order("S1") == order


def test_successful_cancellation_persists_allowed_transition():
    state = InMemoryOrderState()
    tools = _tool(state)

    result = tools.cancel_order("P100")

    assert result.value.status is OrderStatus.CANCELLED
    assert state.inspect_order("P100").status is OrderStatus.CANCELLED
    assert result.duration_ms >= 0
    assert state.inspect_refunds() == ()


@pytest.mark.parametrize(
    "status",
    [
        OrderStatus.CANCELLED,
        OrderStatus.PROCESSING,
        OrderStatus.SHIPPED,
    ],
)
def test_external_status_change_applies_fixture_supported_transitions(status):
    original = Order("P1", "owner-1", Decimal("24.50"), "USD", OrderStatus.PENDING)
    state = _state_with(original)

    state.apply_external_order_status_change("P1", status)

    assert state.inspect_order("P1") == replace(original, status=status)
    assert state.inspect_refunds() == ()
    assert state.inspect_idempotency_records() == ()


@pytest.mark.parametrize(
    ("initial_status", "target_status"),
    [
        (OrderStatus.PENDING, OrderStatus.PENDING),
        (OrderStatus.PENDING, OrderStatus.DELIVERED),
        (OrderStatus.PENDING, OrderStatus.REFUNDED),
        (OrderStatus.CANCELLED, OrderStatus.PROCESSING),
        (OrderStatus.REFUNDED, OrderStatus.SHIPPED),
    ],
)
def test_external_status_change_rejects_unapproved_transitions(
    initial_status,
    target_status,
):
    state = _state_with(
        Order("P1", "owner-1", Decimal("24.50"), "USD", initial_status)
    )
    before = state.inspect_order("P1")

    with pytest.raises(InvalidTransition):
        state.apply_external_order_status_change("P1", target_status)

    assert state.inspect_order("P1") == before


def test_external_status_change_rejects_unknown_order_and_invalid_status():
    state = _state_with(
        Order("P1", "owner-1", Decimal("24.50"), "USD", OrderStatus.PENDING)
    )

    with pytest.raises(OrderNotFound):
        state.apply_external_order_status_change("missing", OrderStatus.SHIPPED)
    with pytest.raises(InvalidRequest, match="status must be an OrderStatus"):
        state.apply_external_order_status_change("P1", "shipped")

    assert state.inspect_order("P1").status is OrderStatus.PENDING


def test_external_status_change_preserves_refunds_idempotency_and_refund_sequence():
    state = _state_with(
        Order("R1", "owner-1", Decimal("49.95"), "USD", OrderStatus.SHIPPED),
        Order("P1", "owner-1", Decimal("24.50"), "USD", OrderStatus.PENDING),
    )
    tools = _tool(state)
    first_refund = _refund(tools, order_id="R1")
    refunds_before = state.inspect_refunds()
    records_before = state.inspect_idempotency_records()

    state.apply_external_order_status_change("P1", OrderStatus.SHIPPED)

    assert state.inspect_order("P1") == Order(
        "P1", "owner-1", Decimal("24.50"), "USD", OrderStatus.SHIPPED
    )
    assert state.inspect_order("R1").status is OrderStatus.REFUNDED
    assert state.inspect_refunds() == refunds_before == (first_refund.value,)
    assert state.inspect_idempotency_records() == records_before
    second_refund = _refund(tools, order_id="P1", key="key-2", amount=Decimal("24.50"))
    assert second_refund.value.refund_id == "R000002"


def test_external_status_change_cannot_desynchronize_refunded_order_records():
    state = _state_with(
        Order("R1", "owner-1", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    _refund(_tool(state), order_id="R1")
    before = (
        state.inspect_order("R1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    )

    with pytest.raises(InvalidTransition):
        state.apply_external_order_status_change("R1", OrderStatus.SHIPPED)

    assert (
        state.inspect_order("R1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    ) == before


def test_successful_full_refund_atomically_persists_refund_and_idempotency():
    order = Order("S1", "owner-1", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    state = _state_with(order)
    tools = _tool(state)

    result = _refund(tools)
    refund = result.value

    assert refund.order_id == "S1"
    assert refund.amount == Decimal("49.95")
    assert refund.currency == "USD"
    assert refund.refund_id == "R000001"
    assert result.replayed is False
    assert result.response_delivery_count == 1
    assert result.duration_ms >= 0
    assert state.inspect_order("S1").status is OrderStatus.REFUNDED
    assert state.inspect_refunds() == (refund,)
    records = state.inspect_idempotency_records()
    assert len(records) == 1
    assert records[0].operation == "refund_order"
    assert records[0].order_id == "S1"
    assert records[0].idempotency_key == "key-1"
    assert records[0].refund == refund


@pytest.mark.parametrize(
    ("status", "operation"),
    [
        (OrderStatus.SHIPPED, "cancel"),
        (OrderStatus.DELIVERED, "cancel"),
        (OrderStatus.CANCELLED, "cancel"),
        (OrderStatus.PENDING, "refund"),
        (OrderStatus.PROCESSING, "refund"),
        (OrderStatus.CANCELLED, "refund"),
    ],
)
def test_invalid_state_transitions_are_rejected_without_mutation(status, operation):
    order = Order("S1", "owner-1", Decimal("49.95"), "USD", status)
    state = _state_with(order)
    tools = _tool(state)

    if operation == "cancel":
        with pytest.raises(InvalidTransition) as captured:
            tools.cancel_order("S1")
    else:
        with pytest.raises(InvalidTransition) as captured:
            _refund(tools)

    assert captured.value.code == "invalid_transition"
    assert isinstance(captured.value.duration_ms, int)
    assert captured.value.duration_ms >= 0
    assert state.inspect_order("S1") == order
    assert state.inspect_refunds() == ()
    assert state.inspect_idempotency_records() == ()


def test_missing_order_raises_typed_error_with_duration():
    state = InMemoryOrderState()
    tools = _tool(state)

    with pytest.raises(OrderNotFound) as captured:
        tools.get_order("missing")

    assert captured.value.code == "order_not_found"
    assert isinstance(captured.value.duration_ms, int)
    assert captured.value.duration_ms >= 0


def test_same_key_same_request_replays_without_changing_state():
    state = _state_with(
        Order("S1", "owner-1", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    tools = _tool(state)
    initial = _refund(tools)
    before = (
        state.inspect_order("S1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    )

    replay = _refund(tools, amount=Decimal("49.950"))

    assert replay.value == initial.value
    assert replay.replayed is True
    assert replay.response_delivery_count == 1
    assert replay.duration_ms >= 0
    assert (
        state.inspect_order("S1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    ) == before


def test_read_only_refund_idempotency_inspection_classifies_without_mutation():
    state = _state_with(
        Order("S1", "owner-1", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    refund = _refund(_tool(state)).value
    before = (
        state.inspect_order("S1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    )

    replay = state.inspect_refund_idempotency(
        "S1",
        "key-1",
        amount=Decimal("49.950"),
        currency="USD",
    )
    conflict = state.inspect_refund_idempotency(
        "S1",
        "key-1",
        amount=Decimal("48.00"),
        currency="USD",
    )
    missing = state.inspect_refund_idempotency(
        "S1",
        "other-key",
        amount=Decimal("49.95"),
        currency="USD",
    )

    assert replay.status is RefundLookupStatus.REPLAY
    assert replay.refund == refund
    assert replay.refund is not refund
    assert conflict.status is RefundLookupStatus.CONFLICT
    assert conflict.refund is None
    assert missing.status is RefundLookupStatus.NO_RECORD
    assert missing.refund is None
    assert (
        state.inspect_order("S1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    ) == before


@pytest.mark.parametrize(
    "fault",
    [
        FaultMode.TIMEOUT_BEFORE_COMMIT,
        FaultMode.TIMEOUT_AFTER_COMMIT,
        FaultMode.DUPLICATE_RESPONSE,
    ],
)
def test_replay_does_not_retrigger_fault_injection(fault):
    state = _state_with(
        Order("S1", "owner-1", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    tools = _tool(state)
    successful = _refund(tools)
    before = (
        state.inspect_order("S1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    )

    replay = _refund(tools, fault=fault)

    assert replay.value == successful.value
    assert replay.replayed is True
    assert replay.response_delivery_count == 1
    assert (
        state.inspect_order("S1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    ) == before


def test_same_key_different_canonical_request_conflicts_without_mutation():
    state = _state_with(
        Order("S1", "owner-1", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    tools = _tool(state)
    _refund(tools)
    before = (
        state.inspect_order("S1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    )

    with pytest.raises(IdempotencyConflict) as captured:
        _refund(tools, amount=Decimal("48.00"))

    assert captured.value.code == "idempotency_conflict"
    assert (
        state.inspect_order("S1"),
        state.inspect_refunds(),
        state.inspect_idempotency_records(),
    ) == before


def test_different_key_on_refunded_order_is_rejected_without_new_records():
    state = _state_with(
        Order("S1", "owner-1", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    tools = _tool(state)
    _refund(tools)
    before_records = state.inspect_idempotency_records()

    with pytest.raises(AlreadyRefunded) as captured:
        _refund(tools, key="key-2")

    assert captured.value.code == "already_refunded"
    assert state.inspect_order("S1").status is OrderStatus.REFUNDED
    assert len(state.inspect_refunds()) == 1
    assert state.inspect_idempotency_records() == before_records


def test_timeout_before_commit_leaves_no_persisted_mutation_and_can_retry():
    state = _state_with(
        Order("S1", "owner-1", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    tools = _tool(state)
    original = state.inspect_order("S1")

    with pytest.raises(ToolTimeoutError) as captured:
        _refund(tools, fault=FaultMode.TIMEOUT_BEFORE_COMMIT)

    assert isinstance(captured.value, TimeoutError)
    assert captured.value.committed is False
    assert captured.value.fault_mode is FaultMode.TIMEOUT_BEFORE_COMMIT
    assert isinstance(captured.value.duration_ms, int)
    assert captured.value.duration_ms >= 0
    assert state.inspect_order("S1") == original
    assert state.inspect_refunds() == ()
    assert state.inspect_idempotency_records() == ()

    retry = _refund(tools)
    assert retry.replayed is False
    assert len(state.inspect_refunds()) == 1
    assert len(state.inspect_idempotency_records()) == 1


def test_timeout_after_commit_persists_refund_and_record_then_retry_replays():
    state = _state_with(
        Order("S1", "owner-1", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    tools = _tool(state)

    with pytest.raises(ToolTimeoutError) as captured:
        _refund(tools, fault=FaultMode.TIMEOUT_AFTER_COMMIT)

    assert captured.value.committed is True
    assert captured.value.fault_mode is FaultMode.TIMEOUT_AFTER_COMMIT
    assert isinstance(captured.value.duration_ms, int)
    assert captured.value.duration_ms >= 0
    assert state.inspect_order("S1").status is OrderStatus.REFUNDED
    assert len(state.inspect_refunds()) == 1
    assert len(state.inspect_idempotency_records()) == 1
    committed_refund = state.inspect_refunds()[0]
    assert state.inspect_idempotency_records()[0].refund == committed_refund

    replay = _refund(tools)
    assert replay.value == committed_refund
    assert replay.replayed is True
    assert replay.duration_ms >= 0
    assert state.inspect_order("S1").status is OrderStatus.REFUNDED
    assert state.inspect_refunds() == (committed_refund,)
    assert len(state.inspect_idempotency_records()) == 1


def test_duplicate_response_is_delivery_metadata_not_second_transaction():
    state = _state_with(
        Order("S1", "owner-1", Decimal("49.95"), "USD", OrderStatus.DELIVERED)
    )
    tools = _tool(state)

    result = _refund(tools, fault=FaultMode.DUPLICATE_RESPONSE)

    assert result.replayed is False
    assert result.response_delivery_count == 2
    assert result.duration_ms >= 0
    assert state.inspect_order("S1").status is OrderStatus.REFUNDED
    assert len(state.inspect_refunds()) == 1
    assert len(state.inspect_idempotency_records()) == 1


def test_reset_restores_default_fixture_and_clears_mutations_deterministically():
    state = InMemoryOrderState()
    tools = _tool(state)
    baseline = tuple(state.inspect_order(order_id) for order_id in ("P100", "S300"))
    _refund(tools, order_id="S300", amount=Decimal("49.95"))
    tools.cancel_order("P100")

    state.reset()

    assert tuple(state.inspect_order(order_id) for order_id in ("P100", "S300")) == baseline
    assert state.inspect_refunds() == ()
    assert state.inspect_idempotency_records() == ()
    result = _refund(_tool(state), order_id="S300", amount=Decimal("49.95"))
    assert result.value.refund_id == "R000001"


@pytest.mark.parametrize(
    ("amount", "currency"),
    [
        (Decimal("48.00"), "USD"),
        (Decimal("49.95"), "EUR"),
    ],
)
def test_non_full_refund_or_currency_mismatch_is_rejected_without_mutation(
    amount, currency
):
    state = _state_with(
        Order("S1", "owner-1", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )

    with pytest.raises(InvalidRequest):
        _refund(_tool(state), amount=amount, currency=currency)

    assert state.inspect_order("S1").status is OrderStatus.SHIPPED
    assert state.inspect_refunds() == ()
    assert state.inspect_idempotency_records() == ()


def test_tool_exception_hierarchy_and_result_contract():
    state = _state_with(
        Order("S1", "owner-1", Decimal("49.95"), "USD", OrderStatus.SHIPPED)
    )
    result = _refund(_tool(state))

    assert isinstance(result, ToolResult)
    assert result.operation == "refund_order"
    assert result.duration_ms >= 0
    assert isinstance(result.replayed, bool)
    assert isinstance(result.response_delivery_count, int)
    assert issubclass(OrderNotFound, ToolError)
    assert issubclass(InvalidTransition, ToolError)
    assert issubclass(InvalidRequest, ToolError)
    assert issubclass(IdempotencyConflict, ToolError)
    assert issubclass(AlreadyRefunded, ToolError)
    assert issubclass(ToolTimeoutError, ToolError)
    assert issubclass(ToolTimeoutError, TimeoutError)
