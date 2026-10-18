"""Synchronous deterministic authorization and confirmation for AgentGuard."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal, InvalidOperation
from enum import Enum
import hashlib
import hmac
import json
import secrets
import time
from typing import Callable
from uuid import uuid4

from simulated_tools import (
    AlreadyRefunded,
    FaultMode,
    InMemoryOrderState,
    InvalidRequest,
    InvalidTransition,
    Order,
    OrderStatus,
    Refund,
    RefundLookupStatus,
    SimulatedOrderTools,
    ToolError,
    ToolResult,
)
from trace_events import PolicyOutcome

CONFIRMATION_TTL_SECONDS = 5 * 60
_CANCELABLE_STATES = {OrderStatus.PENDING, OrderStatus.PROCESSING}
_REFUNDABLE_STATES = {OrderStatus.SHIPPED, OrderStatus.DELIVERED}


@dataclass(frozen=True)
class SessionContext:
    """Authenticated identity provided by trusted application code."""

    user_id: str


@dataclass(frozen=True)
class ToolRequest:
    operation: str
    order_id: str
    amount: Decimal | None = None
    currency: str | None = None
    idempotency_key: str | None = None
    args_source: str = "unknown"


@dataclass(frozen=True)
class PolicyDecision:
    decision: PolicyOutcome
    reason_code: str
    reason: str
    checked_user_id: str = ""
    checked_owner_id: str = ""
    idempotency_key_ref: str = ""


@dataclass(frozen=True)
class ConfirmationChallenge:
    confirmation_id: str
    operation: str
    order_id: str
    parameters: tuple[tuple[str, str], ...]
    expires_at: str
    order_state_hash: str


@dataclass(frozen=True)
class GateResult:
    policy: PolicyDecision
    result: ToolResult | None = None
    challenge: ConfirmationChallenge | None = None
    execution_error: ToolError | None = None


class HumanDecision(str, Enum):
    APPROVE = "approve"
    REJECT = "reject"


class ConfirmationError(Exception):
    """An invalid or unavailable trusted human-confirmation interaction."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


@dataclass
class _ConfirmationRecord:
    confirmation_id: str
    capability_digest: str
    user_id: str
    operation: str
    order_id: str
    request_digest: str
    order_state_hash: str
    expires_at: str
    expires_monotonic: float
    capability_handed_off: bool = False
    approved: bool = False
    rejected: bool = False
    consumed: bool = False


class PolicyGate:
    """Authorize order requests before delegating business work to Task 2.

    Trusted application code must take capability handoffs and retain the raw
    capability in private continuation state. It must never forward a
    capability to model-visible arguments, prompts, logs, or trace events.
    """

    def __init__(
        self,
        state: InMemoryOrderState,
        tools: SimulatedOrderTools | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not isinstance(state, InMemoryOrderState):
            raise TypeError("state must be an InMemoryOrderState")
        if tools is not None and not isinstance(tools, SimulatedOrderTools):
            raise TypeError("tools must be a SimulatedOrderTools instance")
        if tools is not None and tools._state is not state:
            raise ValueError("tools must use the same order state as the gate")
        self._state = state
        self._tools = tools if tools is not None else SimulatedOrderTools(state)
        self._clock = clock
        self._confirmations: dict[str, _ConfirmationRecord] = {}
        self._capability_handoffs: dict[str, str] = {}

    def handle(
        self,
        request: ToolRequest,
        session: SessionContext | None,
        *,
        confirmation_capability: str | None = None,
        before_dispatch: Callable[[PolicyDecision, bool], None] | None = None,
    ) -> GateResult:
        """Authorize a request and synchronously dispatch it when allowed.

        Trusted callers may supply ``before_dispatch`` to observe the final
        policy decision after authorization, immediately before tool execution.
        The boolean reports whether this dispatch consumes a confirmation.
        If the callback raises, the tool is not invoked.
        """
        user_id = _trusted_user_id(session)
        if user_id is None:
            return _blocked(
                "authentication_required",
                "Authenticated session required",
            )
        if not isinstance(request, ToolRequest):
            return _blocked(
                "invalid_request",
                "Invalid tool request",
                checked_user_id=user_id,
            )
        if not isinstance(request.order_id, str) or not request.order_id:
            return _blocked(
                "order_unavailable",
                "Order unavailable",
                checked_user_id=user_id,
            )

        order = self._state.inspect_order(request.order_id)
        if order is None or order.owner_id != user_id:
            return _blocked(
                "order_unavailable",
                "Order unavailable",
                checked_user_id=user_id,
            )

        idempotency_key_ref = ""
        if not isinstance(request.operation, str) or request.operation not in {
            "get_order",
            "cancel_order",
            "refund_order",
        }:
            return _blocked(
                "invalid_operation",
                "Unsupported operation",
                checked_user_id=user_id,
                checked_owner_id=order.owner_id,
            )

        try:
            canonical = _canonical_request(request)
        except ValueError as error:
            return _blocked(
                "invalid_request",
                str(error),
                checked_user_id=user_id,
                checked_owner_id=order.owner_id,
            )

        if request.operation == "get_order":
            if request.amount is not None or request.currency is not None or request.idempotency_key is not None:
                return _blocked(
                    "invalid_request",
                    "get_order accepts only order_id",
                    checked_user_id=user_id,
                    checked_owner_id=order.owner_id,
                )
            return self._dispatch(
                PolicyOutcome.ALLOWED,
                "authorized",
                "Request authorized",
                request,
                user_id,
                order.owner_id,
                idempotency_key_ref,
                before_dispatch=before_dispatch,
            )

        business_error: tuple[str, str] | None = None
        if request.operation == "cancel_order":
            if request.amount is not None or request.currency is not None or request.idempotency_key is not None:
                return _blocked(
                    "invalid_request",
                    "cancel_order accepts only order_id",
                    checked_user_id=user_id,
                    checked_owner_id=order.owner_id,
                )
            if order.status not in _CANCELABLE_STATES:
                business_error = (
                    "invalid_transition",
                    "Order cannot be cancelled in its current state",
                )
        else:
            refund_error = _validate_refund_parameters(request)
            if refund_error is not None:
                return _blocked(
                    "invalid_request",
                    refund_error,
                    checked_user_id=user_id,
                    checked_owner_id=order.owner_id,
                )
            assert request.amount is not None
            assert request.currency is not None
            assert request.idempotency_key is not None
            idempotency_key_ref = _idempotency_key_ref(
                request.order_id,
                request.idempotency_key,
            )
            lookup = self._state.inspect_refund_idempotency(
                request.order_id,
                request.idempotency_key,
                amount=request.amount,
                currency=request.currency,
            )
            if lookup.status is RefundLookupStatus.REPLAY:
                self._invalidate_matching_confirmations(user_id, canonical)
                return self._dispatch(
                    PolicyOutcome.REPLAY,
                    "idempotent_replay",
                    "Previously completed refund returned",
                    request,
                    user_id,
                    order.owner_id,
                    idempotency_key_ref,
                    before_dispatch=before_dispatch,
                )
            if lookup.status is RefundLookupStatus.CONFLICT:
                return _blocked(
                    "idempotency_conflict",
                    "Idempotency key conflicts with an earlier request",
                    checked_user_id=user_id,
                    checked_owner_id=order.owner_id,
                    idempotency_key_ref=idempotency_key_ref,
                )
            business_error = _refund_business_error(request, order)

        if confirmation_capability is None:
            if business_error is not None:
                return _blocked(
                    business_error[0],
                    business_error[1],
                    checked_user_id=user_id,
                    checked_owner_id=order.owner_id,
                    idempotency_key_ref=idempotency_key_ref,
                )
            return self._create_confirmation(
                request,
                canonical,
                order,
                user_id,
                idempotency_key_ref,
            )

        confirmation = self._find_confirmation(confirmation_capability)
        if confirmation is None:
            return _blocked(
                "confirmation_invalid",
                "Confirmation is invalid",
                checked_user_id=user_id,
                checked_owner_id=order.owner_id,
                idempotency_key_ref=idempotency_key_ref,
            )
        confirmation_error = self._validate_confirmation(
            confirmation,
            request,
            canonical,
            order,
            user_id,
        )
        if confirmation_error is not None:
            return _blocked(
                confirmation_error[0],
                confirmation_error[1],
                checked_user_id=user_id,
                checked_owner_id=order.owner_id,
                idempotency_key_ref=idempotency_key_ref,
            )
        if business_error is not None:
            return _blocked(
                business_error[0],
                business_error[1],
                checked_user_id=user_id,
                checked_owner_id=order.owner_id,
                idempotency_key_ref=idempotency_key_ref,
            )

        confirmation.consumed = True
        return self._dispatch(
            PolicyOutcome.ALLOWED,
            "authorized",
            "Request authorized",
            request,
            user_id,
            order.owner_id,
            idempotency_key_ref,
            before_dispatch=before_dispatch,
            confirmation_consumed=True,
        )

    def take_capability_handoff(self, confirmation_id: str) -> str:
        """One-time secret handoff for trusted application code only."""
        token = self._capability_handoffs.pop(confirmation_id, None)
        confirmation = self._confirmations.get(confirmation_id)
        if token is None or confirmation is None or confirmation.consumed:
            raise ConfirmationError("capability_handoff_unavailable")
        confirmation.capability_handed_off = True
        return token

    def record_human_decision(
        self,
        confirmation_id: str,
        session: SessionContext,
        decision: HumanDecision,
        *,
        capability: str,
    ) -> None:
        """Record an explicit trusted-handler decision for a pending challenge.

        Only an application handler responding to explicit human input may call
        this method. The capability proves challenge possession, not approval;
        `decision` must be supplied separately by that trusted handler.
        """
        user_id = _trusted_user_id(session)
        if user_id is None:
            raise ConfirmationError("authentication_required")
        if not isinstance(decision, HumanDecision):
            raise ConfirmationError("invalid_human_decision")
        confirmation = self._confirmations.get(confirmation_id)
        if confirmation is None:
            raise ConfirmationError("confirmation_invalid")
        if not confirmation.capability_handed_off:
            raise ConfirmationError("capability_not_handed_off")
        if not self._capability_matches(confirmation, capability):
            raise ConfirmationError("confirmation_invalid")
        if confirmation.consumed:
            raise ConfirmationError("confirmation_consumed")
        if self._clock() >= confirmation.expires_monotonic:
            confirmation.consumed = True
            raise ConfirmationError("confirmation_expired")
        if confirmation.user_id != user_id:
            raise ConfirmationError("confirmation_user_mismatch")

        order = self._state.inspect_order(confirmation.order_id)
        if order is None or order.owner_id != user_id:
            confirmation.consumed = True
            raise ConfirmationError("order_unavailable")
        if _order_state_hash(order) != confirmation.order_state_hash:
            confirmation.consumed = True
            raise ConfirmationError("confirmation_state_changed")
        if confirmation.approved or confirmation.rejected:
            raise ConfirmationError("human_decision_already_recorded")

        if decision is HumanDecision.APPROVE:
            confirmation.approved = True
        else:
            confirmation.rejected = True

    def _create_confirmation(
        self,
        request: ToolRequest,
        canonical: dict[str, str],
        order: Order,
        user_id: str,
        idempotency_key_ref: str,
    ) -> GateResult:
        confirmation_id = uuid4().hex
        capability = secrets.token_urlsafe(32)
        expires_monotonic = self._clock() + CONFIRMATION_TTL_SECONDS
        expires_at = (
            datetime.now(timezone.utc) + timedelta(seconds=CONFIRMATION_TTL_SECONDS)
        ).isoformat()
        state_hash = _order_state_hash(order)
        confirmation = _ConfirmationRecord(
            confirmation_id=confirmation_id,
            capability_digest=_token_digest(capability),
            user_id=user_id,
            operation=request.operation,
            order_id=request.order_id,
            request_digest=_canonical_digest(canonical),
            order_state_hash=state_hash,
            expires_at=expires_at,
            expires_monotonic=expires_monotonic,
        )
        self._confirmations[confirmation_id] = confirmation
        self._capability_handoffs[confirmation_id] = capability
        challenge = ConfirmationChallenge(
            confirmation_id=confirmation_id,
            operation=request.operation,
            order_id=request.order_id,
            parameters=_public_parameters(canonical),
            expires_at=expires_at,
            order_state_hash=state_hash,
        )
        return GateResult(
            policy=PolicyDecision(
                PolicyOutcome.BLOCKED,
                "confirmation_required",
                "Explicit confirmation is required",
                user_id,
                order.owner_id,
                idempotency_key_ref,
            ),
            challenge=challenge,
        )

    def _find_confirmation(self, capability: str) -> _ConfirmationRecord | None:
        if not isinstance(capability, str) or not capability:
            return None
        digest = _token_digest(capability)
        for confirmation in self._confirmations.values():
            if hmac.compare_digest(confirmation.capability_digest, digest):
                return confirmation
        return None

    @staticmethod
    def _capability_matches(
        confirmation: _ConfirmationRecord,
        capability: str,
    ) -> bool:
        if not isinstance(capability, str) or not capability:
            return False
        return hmac.compare_digest(
            confirmation.capability_digest,
            _token_digest(capability),
        )

    def _validate_confirmation(
        self,
        confirmation: _ConfirmationRecord,
        request: ToolRequest,
        canonical: dict[str, str],
        order: Order,
        user_id: str,
    ) -> tuple[str, str] | None:
        if confirmation.consumed:
            return "confirmation_reused", "Confirmation has already been consumed"
        if confirmation.rejected:
            confirmation.consumed = True
            return "confirmation_rejected", "Confirmation was rejected"
        if self._clock() >= confirmation.expires_monotonic:
            confirmation.consumed = True
            return "confirmation_expired", "Confirmation has expired"
        if not confirmation.approved:
            return (
                "confirmation_not_approved",
                "Explicit human approval has not been recorded",
            )
        if (
            confirmation.user_id != user_id
            or confirmation.operation != request.operation
            or confirmation.order_id != request.order_id
            or confirmation.request_digest != _canonical_digest(canonical)
        ):
            return "confirmation_mismatch", "Confirmation does not match this request"
        if _order_state_hash(order) != confirmation.order_state_hash:
            confirmation.consumed = True
            return "confirmation_state_changed", "Order state changed after confirmation"
        return None

    def _invalidate_matching_confirmations(
        self,
        user_id: str,
        canonical: dict[str, str],
    ) -> None:
        request_digest = _canonical_digest(canonical)
        for confirmation in self._confirmations.values():
            if (
                confirmation.user_id == user_id
                and confirmation.request_digest == request_digest
            ):
                confirmation.consumed = True
                self._capability_handoffs.pop(confirmation.confirmation_id, None)

    def _dispatch(
        self,
        outcome: PolicyOutcome,
        reason_code: str,
        reason: str,
        request: ToolRequest,
        user_id: str,
        owner_id: str,
        idempotency_key_ref: str,
        *,
        before_dispatch: Callable[[PolicyDecision, bool], None] | None = None,
        confirmation_consumed: bool = False,
    ) -> GateResult:
        decision = PolicyDecision(
            outcome,
            reason_code,
            reason,
            user_id,
            owner_id,
            idempotency_key_ref,
        )
        if before_dispatch is not None:
            before_dispatch(decision, confirmation_consumed)
        try:
            if request.operation == "get_order":
                result = self._tools.get_order(request.order_id)
            elif request.operation == "cancel_order":
                result = self._tools.cancel_order(request.order_id)
            else:
                assert request.amount is not None
                assert request.currency is not None
                assert request.idempotency_key is not None
                result = self._tools.refund_order(
                    request.order_id,
                    request.idempotency_key,
                    amount=request.amount,
                    currency=request.currency,
                    fault=FaultMode.NONE,
                )
        except ToolError as error:
            decision = PolicyDecision(
                PolicyOutcome.ALLOWED,
                "authorized_execution_failed",
                "Authorized tool execution failed",
                user_id,
                owner_id,
                idempotency_key_ref,
            )
            return GateResult(policy=decision, execution_error=error)
        return GateResult(policy=decision, result=result)


def _trusted_user_id(session: SessionContext | None) -> str | None:
    if not isinstance(session, SessionContext):
        return None
    if not isinstance(session.user_id, str) or not session.user_id.strip():
        return None
    return session.user_id


def _blocked(
    reason_code: str,
    reason: str,
    *,
    checked_user_id: str = "",
    checked_owner_id: str = "",
    idempotency_key_ref: str = "",
) -> GateResult:
    return GateResult(
        policy=PolicyDecision(
            PolicyOutcome.BLOCKED,
            reason_code,
            reason,
            checked_user_id,
            checked_owner_id,
            idempotency_key_ref,
        )
    )


def _canonical_request(request: ToolRequest) -> dict[str, str]:
    if not isinstance(request.operation, str) or request.operation not in {
        "get_order",
        "cancel_order",
        "refund_order",
    }:
        raise ValueError("Unsupported operation")
    canonical = {
        "operation": request.operation,
        "order_id": request.order_id,
    }
    if request.operation == "refund_order":
        amount = _parse_amount(request.amount)
        if not isinstance(request.currency, str) or not request.currency:
            raise ValueError("currency must be a non-empty string")
        if not isinstance(request.idempotency_key, str) or not request.idempotency_key:
            raise ValueError("idempotency_key must be a non-empty string")
        canonical.update(
            {
                "amount": _decimal_string(amount),
                "currency": request.currency,
                "idempotency_key": request.idempotency_key,
            }
        )
    return canonical


def _validate_refund_parameters(request: ToolRequest) -> str | None:
    try:
        _parse_amount(request.amount)
    except ValueError as error:
        return str(error)
    if not isinstance(request.currency, str) or not request.currency:
        return "currency must be a non-empty string"
    if not isinstance(request.idempotency_key, str) or not request.idempotency_key:
        return "idempotency_key must be a non-empty string"
    return None


def _refund_business_error(
    request: ToolRequest,
    order: Order,
) -> tuple[str, str] | None:
    assert request.amount is not None
    assert request.currency is not None
    if request.amount != order.total_amount:
        return (
            "invalid_request",
            "Only full refunds equal to the order total are allowed",
        )
    if request.currency != order.currency:
        return "invalid_request", "Currency must match the order currency"
    if order.status is OrderStatus.REFUNDED:
        return "already_refunded", "Order has already been refunded"
    if order.status not in _REFUNDABLE_STATES:
        return (
            "invalid_transition",
            "Order cannot be refunded in its current state",
        )
    return None


def _parse_amount(amount: Decimal | None) -> Decimal:
    if isinstance(amount, bool) or amount is None:
        raise ValueError("amount must be a positive finite Decimal")
    try:
        parsed = amount if isinstance(amount, Decimal) else Decimal(amount)
    except (InvalidOperation, TypeError, ValueError):
        raise ValueError("amount must be a positive finite Decimal") from None
    if not parsed.is_finite() or parsed <= 0:
        raise ValueError("amount must be a positive finite Decimal")
    return parsed


def _decimal_string(value: Decimal) -> str:
    text = format(value, "f")
    if "." in text:
        text = text.rstrip("0").rstrip(".")
    return text


def _canonical_digest(values: dict[str, str]) -> str:
    serialized = json.dumps(
        values,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _order_state_hash(order: Order) -> str:
    snapshot = {
        "order_id": order.order_id,
        "owner_id": order.owner_id,
        "status": order.status.value,
        "total_amount": _decimal_string(order.total_amount),
        "currency": order.currency,
    }
    return _canonical_digest(snapshot)


def _public_parameters(canonical: dict[str, str]) -> tuple[tuple[str, str], ...]:
    return tuple(
        (key, value)
        for key, value in sorted(canonical.items())
        if key != "idempotency_key"
    )


def _token_digest(capability: str) -> str:
    return hashlib.sha256(capability.encode("utf-8")).hexdigest()


def _idempotency_key_ref(order_id: str, key: str) -> str:
    return _canonical_digest(
        {
            "operation": "refund_order",
            "order_id": order_id,
            "idempotency_key": key,
        }
    )
