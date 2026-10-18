"""Deterministic in-memory order operations for AgentGuard V1.

Future orchestration must authenticate and check ownership before invoking
refund_order. Only after authorization may it rely on idempotency replay; this
module itself performs neither authentication nor ownership checks.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import json
import time
from typing import Sequence


class OrderStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    SHIPPED = "shipped"
    DELIVERED = "delivered"
    CANCELLED = "cancelled"
    REFUNDED = "refunded"


class FaultMode(str, Enum):
    NONE = "none"
    TIMEOUT_BEFORE_COMMIT = "timeout_before_commit"
    TIMEOUT_AFTER_COMMIT = "timeout_after_commit"
    DUPLICATE_RESPONSE = "duplicate_response"


class RefundLookupStatus(str, Enum):
    NO_RECORD = "no_record"
    REPLAY = "replay"
    CONFLICT = "conflict"


@dataclass(frozen=True)
class Order:
    order_id: str
    owner_id: str
    total_amount: Decimal
    currency: str
    status: OrderStatus


@dataclass(frozen=True)
class OrderView:
    order_id: str
    total_amount: Decimal
    currency: str
    status: OrderStatus


@dataclass(frozen=True)
class Refund:
    refund_id: str
    order_id: str
    amount: Decimal
    currency: str
    status: str = "completed"


@dataclass(frozen=True)
class IdempotencyRecord:
    operation: str
    order_id: str
    idempotency_key: str
    request_fingerprint: str
    refund: Refund


@dataclass(frozen=True)
class RefundIdempotencyLookup:
    status: RefundLookupStatus
    refund: Refund | None = None


@dataclass(frozen=True)
class ToolResult:
    operation: str
    value: OrderView | Refund
    duration_ms: int
    replayed: bool = False
    response_delivery_count: int = 1


class ToolError(Exception):
    """Base for expected tool rejections; duration is set by the tool call."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.duration_ms: int | None = None


class OrderNotFound(ToolError):
    def __init__(self, order_id: str) -> None:
        super().__init__("order_not_found", f"Order {order_id!r} does not exist")


class InvalidTransition(ToolError):
    def __init__(self, operation: str, status: OrderStatus) -> None:
        super().__init__(
            "invalid_transition",
            f"Cannot {operation} an order in {status.value!r} state",
        )


class InvalidRequest(ToolError):
    def __init__(self, message: str) -> None:
        super().__init__("invalid_request", message)


class IdempotencyConflict(ToolError):
    def __init__(self) -> None:
        super().__init__(
            "idempotency_conflict",
            "Idempotency key was already used for a different request",
        )


class AlreadyRefunded(ToolError):
    def __init__(self, order_id: str) -> None:
        super().__init__(
            "already_refunded",
            f"Order {order_id!r} has already been refunded",
        )


class ToolTimeoutError(ToolError, TimeoutError):
    def __init__(self, fault_mode: FaultMode, *, committed: bool) -> None:
        super().__init__(
            fault_mode.value,
            f"Simulated refund timeout ({fault_mode.value})",
        )
        self.fault_mode = fault_mode
        self.committed = committed


@dataclass(frozen=True)
class _State:
    orders: dict[str, Order]
    refunds: dict[str, Refund]
    idempotency_records: dict[tuple[str, str, str], IdempotencyRecord]
    next_refund_number: int = 1


def default_orders() -> tuple[Order, ...]:
    """Return fresh deterministic fixtures for offline tests and demos."""
    return (
        Order("P100", "user-1", Decimal("24.50"), "USD", OrderStatus.PENDING),
        Order("P200", "user-2", Decimal("31.00"), "USD", OrderStatus.PROCESSING),
        Order("S300", "user-1", Decimal("49.95"), "USD", OrderStatus.SHIPPED),
        Order("D400", "user-2", Decimal("18.75"), "USD", OrderStatus.DELIVERED),
        Order("C500", "user-1", Decimal("12.00"), "USD", OrderStatus.CANCELLED),
    )


class InMemoryOrderState:
    """Authoritative single-threaded order, refund, and idempotency state."""

    def __init__(self, orders: Sequence[Order] | None = None) -> None:
        self.reset(orders)

    def reset(self, orders: Sequence[Order] | None = None) -> None:
        initial_orders = tuple(default_orders() if orders is None else orders)
        order_map: dict[str, Order] = {}
        for order in initial_orders:
            if not isinstance(order, Order):
                raise TypeError("orders must contain Order instances")
            if not order.order_id:
                raise ValueError("order_id must not be empty")
            if order.order_id in order_map:
                raise ValueError(f"duplicate order_id: {order.order_id}")
            order_map[order.order_id] = replace(order)
        self._state = _State(
            orders=order_map,
            refunds={},
            idempotency_records={},
        )

    def apply_external_order_status_change(
        self,
        order_id: str,
        status: OrderStatus,
    ) -> None:
        """Apply an allowed external status change without resetting business state."""
        SimulatedOrderTools._validate_identifier(order_id, "order_id")
        if not isinstance(status, OrderStatus):
            raise InvalidRequest("status must be an OrderStatus")

        order = self._state.orders.get(order_id)
        if order is None:
            raise OrderNotFound(order_id)

        if status is OrderStatus.REFUNDED:
            raise InvalidTransition("externally refund", order.status)

        has_refund_records = any(
            refund.order_id == order_id for refund in self._state.refunds.values()
        ) or any(
            record.order_id == order_id
            for record in self._state.idempotency_records.values()
        )
        if has_refund_records:
            raise InvalidTransition(
                "externally change status for an order with refund records",
                order.status,
            )

        allowed_targets = {
            OrderStatus.CANCELLED,
            OrderStatus.PROCESSING,
            OrderStatus.SHIPPED,
        }
        if order.status is not OrderStatus.PENDING or status not in allowed_targets:
            raise InvalidTransition(
                f"externally change status to {status.value}",
                order.status,
            )

        new_orders = dict(self._state.orders)
        new_orders[order_id] = replace(order, status=status)
        self._state = replace(self._state, orders=new_orders)

    def inspect_order(self, order_id: str) -> Order | None:
        order = self._state.orders.get(order_id)
        return replace(order) if order is not None else None

    def inspect_refunds(self) -> tuple[Refund, ...]:
        return tuple(replace(refund) for refund in self._state.refunds.values())

    def inspect_idempotency_records(self) -> tuple[IdempotencyRecord, ...]:
        return tuple(
            replace(record, refund=replace(record.refund))
            for record in self._state.idempotency_records.values()
        )

    def inspect_refund_idempotency(
        self,
        order_id: str,
        idempotency_key: str,
        *,
        amount: Decimal,
        currency: str,
    ) -> RefundIdempotencyLookup:
        """Read-only idempotency classification using refund_order semantics."""
        SimulatedOrderTools._validate_identifier(order_id, "order_id")
        SimulatedOrderTools._validate_identifier(
            idempotency_key,
            "idempotency_key",
        )
        parsed_amount = _validate_amount(amount)
        if not isinstance(currency, str) or not currency:
            raise InvalidRequest("currency must be a non-empty string")

        record = self._find_idempotency(
            ("refund_order", order_id, idempotency_key)
        )
        if record is None:
            return RefundIdempotencyLookup(RefundLookupStatus.NO_RECORD)

        fingerprint = _request_fingerprint(
            order_id=order_id,
            amount=parsed_amount,
            currency=currency,
        )
        if record.request_fingerprint != fingerprint:
            return RefundIdempotencyLookup(RefundLookupStatus.CONFLICT)
        return RefundIdempotencyLookup(
            RefundLookupStatus.REPLAY,
            refund=replace(record.refund),
        )

    def _order(self, order_id: str) -> Order:
        order = self._state.orders.get(order_id)
        if order is None:
            raise OrderNotFound(order_id)
        return order

    def _find_idempotency(
        self,
        key: tuple[str, str, str],
    ) -> IdempotencyRecord | None:
        return self._state.idempotency_records.get(key)

    def _commit_cancellation(self, order: Order) -> Order:
        updated = replace(order, status=OrderStatus.CANCELLED)
        new_orders = dict(self._state.orders)
        new_orders[order.order_id] = updated
        self._state = replace(self._state, orders=new_orders)
        return updated

    def _commit_refund(
        self,
        order: Order,
        idempotency_key: str,
        fingerprint: str,
        amount: Decimal,
        currency: str,
    ) -> Refund:
        refund = Refund(
            refund_id=f"R{self._state.next_refund_number:06d}",
            order_id=order.order_id,
            amount=amount,
            currency=currency,
        )
        updated_order = replace(order, status=OrderStatus.REFUNDED)
        record = IdempotencyRecord(
            operation="refund_order",
            order_id=order.order_id,
            idempotency_key=idempotency_key,
            request_fingerprint=fingerprint,
            refund=refund,
        )

        new_orders = dict(self._state.orders)
        new_orders[order.order_id] = updated_order
        new_refunds = dict(self._state.refunds)
        new_refunds[refund.refund_id] = refund
        new_idempotency_records = dict(self._state.idempotency_records)
        new_idempotency_records[
            ("refund_order", order.order_id, idempotency_key)
        ] = record
        self._state = _State(
            orders=new_orders,
            refunds=new_refunds,
            idempotency_records=new_idempotency_records,
            next_refund_number=self._state.next_refund_number + 1,
        )
        return refund


class SimulatedOrderTools:
    """Public deterministic order operations; no auth, tracing, or orchestration."""

    def __init__(self, state: InMemoryOrderState) -> None:
        if not isinstance(state, InMemoryOrderState):
            raise TypeError("state must be an InMemoryOrderState")
        self._state = state

    def get_order(self, order_id: str) -> ToolResult:
        return self._invoke(
            "get_order",
            lambda: self._get_order(order_id),
        )

    def cancel_order(self, order_id: str) -> ToolResult:
        return self._invoke(
            "cancel_order",
            lambda: self._cancel_order(order_id),
        )

    def refund_order(
        self,
        order_id: str,
        idempotency_key: str,
        *,
        amount: Decimal,
        currency: str,
        fault: FaultMode = FaultMode.NONE,
    ) -> ToolResult:
        """Refund an order with idempotency and fault injection.

        Task 2 performs no authentication or ownership checks.

        Future integration ordering (Task 3/4):
        1. Authenticated identity and ownership MUST be verified before
            this method returns a cached replay.
        2. Within this method, idempotency lookup and fingerprint
            comparison run before new-refund state-transition validation,
            so a legitimate same-key replay remains possible when the
            order is already REFUNDED.
        3. A different-key refund of an already-refunded order is
            rejected with AlreadyRefunded.
        """
        return self._invoke(
            "refund_order",
            lambda: self._refund_order(
                order_id,
                idempotency_key,
                amount=amount,
                currency=currency,
                fault=fault,
            ),
        )

    def _get_order(self, order_id: str) -> ToolResult:
        self._validate_identifier(order_id, "order_id")
        order = self._state._order(order_id)
        return ToolResult(
            operation="get_order",
            value=_order_view(order),
            duration_ms=0,
        )

    def _cancel_order(self, order_id: str) -> ToolResult:
        self._validate_identifier(order_id, "order_id")
        order = self._state._order(order_id)
        if order.status not in (OrderStatus.PENDING, OrderStatus.PROCESSING):
            raise InvalidTransition("cancel", order.status)
        updated = self._state._commit_cancellation(order)
        return ToolResult(
            operation="cancel_order",
            value=_order_view(updated),
            duration_ms=0,
        )

    def _refund_order(
        self,
        order_id: str,
        idempotency_key: str,
        *,
        amount: Decimal,
        currency: str,
        fault: FaultMode,
    ) -> ToolResult:
        self._validate_identifier(order_id, "order_id")
        self._validate_identifier(idempotency_key, "idempotency_key")
        parsed_amount = _validate_amount(amount)
        if not isinstance(currency, str) or not currency:
            raise InvalidRequest("currency must be a non-empty string")
        if not isinstance(fault, FaultMode):
            raise InvalidRequest("fault must be a FaultMode")

        order = self._state._order(order_id)
        fingerprint = _request_fingerprint(
            order_id=order_id,
            amount=parsed_amount,
            currency=currency,
        )
        record_key = ("refund_order", order_id, idempotency_key)
        existing = self._state._find_idempotency(record_key)
        if existing is not None:
            if existing.request_fingerprint != fingerprint:
                raise IdempotencyConflict()
            return ToolResult(
                operation="refund_order",
                value=replace(existing.refund),
                duration_ms=0,
                replayed=True,
            )

        if parsed_amount != order.total_amount:
            raise InvalidRequest("only full refunds equal to the order total are allowed")
        if currency != order.currency:
            raise InvalidRequest("currency must match the order currency")
        if order.status == OrderStatus.REFUNDED:
            raise AlreadyRefunded(order_id)
        if order.status not in (OrderStatus.SHIPPED, OrderStatus.DELIVERED):
            raise InvalidTransition("refund", order.status)

        if fault == FaultMode.TIMEOUT_BEFORE_COMMIT:
            raise ToolTimeoutError(fault, committed=False)

        refund = self._state._commit_refund(
            order,
            idempotency_key,
            fingerprint,
            parsed_amount,
            currency,
        )

        if fault == FaultMode.TIMEOUT_AFTER_COMMIT:
            raise ToolTimeoutError(fault, committed=True)

        return ToolResult(
            operation="refund_order",
            value=replace(refund),
            duration_ms=0,
            response_delivery_count=(
                2 if fault == FaultMode.DUPLICATE_RESPONSE else 1
            ),
        )

    @staticmethod
    def _validate_identifier(value: str, name: str) -> None:
        if not isinstance(value, str) or not value:
            raise InvalidRequest(f"{name} must be a non-empty string")

    @staticmethod
    def _invoke(operation: str, call) -> ToolResult:
        started_ns = time.monotonic_ns()
        try:
            result = call()
        except ToolError as error:
            error.duration_ms = _elapsed_ms(started_ns)
            raise
        return replace(result, duration_ms=_elapsed_ms(started_ns))


def _order_view(order: Order) -> OrderView:
    return OrderView(
        order_id=order.order_id,
        total_amount=order.total_amount,
        currency=order.currency,
        status=order.status,
    )


def _validate_amount(amount: Decimal) -> Decimal:
    if isinstance(amount, bool):
        raise InvalidRequest("amount must be a positive finite Decimal")
    try:
        parsed = amount if isinstance(amount, Decimal) else Decimal(amount)
    except (InvalidOperation, TypeError, ValueError):
        raise InvalidRequest("amount must be a positive finite Decimal") from None
    if not parsed.is_finite() or parsed <= 0:
        raise InvalidRequest("amount must be a positive finite Decimal")
    return parsed


def _request_fingerprint(
    *,
    order_id: str,
    amount: Decimal,
    currency: str,
) -> str:
    canonical_amount = format(amount, "f")
    if "." in canonical_amount:
        canonical_amount = canonical_amount.rstrip("0").rstrip(".")
    request = {
        "operation": "refund_order",
        "order_id": order_id,
        "amount": canonical_amount,
        "currency": currency,
    }
    canonical_json = json.dumps(
        request,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
    return hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()


def _elapsed_ms(started_ns: int) -> int:
    return (time.monotonic_ns() - started_ns) // 1_000_000
