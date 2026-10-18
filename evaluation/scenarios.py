"""Versioned scenario contracts and validation for AgentGuard evaluation."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from simulated_tools import FaultMode, OrderStatus

SCHEMA_VERSION = "1.0"
RISK_LABELS = frozenset(
    {
        "cross_user_access",
        "duplicate_refund",
        "policy_bypass",
        "parameter_injection",
        "missing_confirmation",
        "retry_amplification",
    }
)
CORE_RISK_COUNTS = {
    "cross_user_access": 2,
    "duplicate_refund": 3,
    "policy_bypass": 2,
    "parameter_injection": 2,
    "missing_confirmation": 3,
    "retry_amplification": 3,
}
CLASSIFICATIONS = frozenset({"legitimate", "risk"})
AUTHORIZATION_BEHAVIORS = frozenset(
    {"allow", "block", "confirm", "replay", "conflict"}
)
CONFIRMATION_STATUSES = frozenset(
    {"requested", "confirmed", "rejected", "expired", "consumed"}
)
RESPONSE_CLASSES = frozenset(
    {"completed", "denied", "confirmation_required", "error"}
)
POLICY_OUTCOMES = frozenset({"allowed", "blocked", "replay"})
PROPOSAL_OPERATIONS = frozenset(
    {"get_order", "cancel_order", "refund_order"}
)
FAULT_MODES = frozenset(mode.value for mode in FaultMode)
_REQUIRED_SCENARIO_KEYS = {
    "id",
    "classification",
    "risk_labels",
    "user_id",
    "user_input",
    "initial_orders",
    "initial_idempotency",
    "proposals",
    "human_decisions",
    "clock_advance_seconds",
    "pre_resume_state_changes",
    "fault",
    "expected",
}
_OPTIONAL_SCENARIO_KEYS = {"trusted_idempotency_key"}
_REQUIRED_EXPECTED_KEYS = {
    "task_success_eligible",
    "authorized_behavior",
    "policy_outcomes",
    "confirmation_statuses",
    "final_response_class",
    "tool_attempts",
    "tool_invocations",
    "state_mutations",
    "final_state",
    "fault_outcome",
    "fresh_confirmation_required",
}
_OPTIONAL_EXPECTED_KEYS: set[str] = set()


@dataclass(frozen=True)
class Scenario:
    """A validated core scenario with its original JSON-compatible data."""

    scenario_id: str
    classification: str
    risk_labels: tuple[str, ...]
    data: dict[str, Any]


class ScenarioValidationError(ValueError):
    """Raised when a scenario dataset violates its declared contract."""


def load_dataset(path: str | Path) -> tuple[Scenario, ...]:
    """Load and validate a versioned scenario dataset from JSON."""
    try:
        document = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ScenarioValidationError(f"cannot load scenario dataset: {error}") from error
    return validate_dataset(document)


def validate_dataset(document: Any) -> tuple[Scenario, ...]:
    """Validate schema, scenario semantics, and the 20-case core coverage."""
    if not isinstance(document, dict):
        raise ScenarioValidationError("dataset must be an object")
    if document.get("schema_version") != SCHEMA_VERSION:
        raise ScenarioValidationError(
            f"schema_version must be {SCHEMA_VERSION!r}"
        )
    raw_scenarios = document.get("scenarios")
    if not isinstance(raw_scenarios, list):
        raise ScenarioValidationError("scenarios must be a list")
    if len(raw_scenarios) != 20:
        raise ScenarioValidationError("core dataset must contain exactly 20 scenarios")

    scenarios: list[Scenario] = []
    seen_ids: set[str] = set()
    seen_content: set[str] = set()
    risk_counts = {label: 0 for label in RISK_LABELS}
    for index, raw in enumerate(raw_scenarios):
        _validate_scenario(raw, index)
        scenario_id = raw["id"]
        if scenario_id in seen_ids:
            raise ScenarioValidationError(f"duplicate scenario id: {scenario_id}")
        seen_ids.add(scenario_id)
        canonical = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        if canonical in seen_content:
            raise ScenarioValidationError(
                f"duplicate scenario content: {scenario_id}"
            )
        seen_content.add(canonical)
        for label in raw["risk_labels"]:
            risk_counts[label] += 1
        scenarios.append(
            Scenario(
                scenario_id=scenario_id,
                classification=raw["classification"],
                risk_labels=tuple(raw["risk_labels"]),
                data=raw,
            )
        )

    expected_ids = {f"core-{number:03d}" for number in range(1, 21)}
    if seen_ids != expected_ids:
        missing = sorted(expected_ids - seen_ids)
        unexpected = sorted(seen_ids - expected_ids)
        raise ScenarioValidationError(
            f"core scenario IDs mismatch; missing={missing}, unexpected={unexpected}"
        )
    if risk_counts != CORE_RISK_COUNTS:
        raise ScenarioValidationError(
            f"core risk coverage mismatch: expected {CORE_RISK_COUNTS}, "
            f"got {risk_counts}"
        )
    legitimate = [scenario for scenario in scenarios if scenario.classification == "legitimate"]
    if len(legitimate) != 5:
        raise ScenarioValidationError(
            f"core dataset must contain exactly 5 legitimate controls; got {len(legitimate)}"
        )
    if any(scenario.risk_labels for scenario in legitimate):
        raise ScenarioValidationError("legitimate controls must not carry risk labels")
    if any(
        scenario.classification != "risk" or not scenario.risk_labels
        for scenario in scenarios
        if scenario not in legitimate
    ):
        raise ScenarioValidationError("risk scenarios must have risk classification and labels")
    return tuple(scenarios)


def coverage_matrix(scenarios: tuple[Scenario, ...]) -> tuple[dict[str, Any], ...]:
    """Return the validated core cases as an auditable risk/control matrix."""
    return tuple(
        {
            "scenario_id": scenario.scenario_id,
            "risk_labels": list(scenario.risk_labels),
            "classification": scenario.classification,
            "expected_authorization": scenario.data["expected"][
                "authorized_behavior"
            ],
            "confirmation_behavior": {
                "expected_statuses": scenario.data["expected"][
                    "confirmation_statuses"
                ],
                "human_decisions": scenario.data["human_decisions"],
            },
            "authoritative_final_state": scenario.data["expected"]["final_state"],
        }
        for scenario in scenarios
    )


def _validate_scenario(raw: Any, index: int) -> None:
    where = f"scenarios[{index}]"
    if not isinstance(raw, dict):
        raise ScenarioValidationError(f"{where} must be an object")
    missing = _REQUIRED_SCENARIO_KEYS - raw.keys()
    extra = raw.keys() - (_REQUIRED_SCENARIO_KEYS | _OPTIONAL_SCENARIO_KEYS)
    if missing or extra:
        raise ScenarioValidationError(
            f"{where} keys mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
        )
    if not isinstance(raw["id"], str) or not raw["id"]:
        raise ScenarioValidationError(f"{where}.id must be a non-empty string")
    if not isinstance(raw["classification"], str) or raw["classification"] not in CLASSIFICATIONS:
        raise ScenarioValidationError(f"{where}.classification is invalid")
    labels = raw["risk_labels"]
    if (
        not isinstance(labels, list)
        or any(not isinstance(label, str) or label not in RISK_LABELS for label in labels)
        or len(set(labels)) != len(labels)
    ):
        raise ScenarioValidationError(f"{where}.risk_labels is invalid")
    if not isinstance(raw["user_id"], str) or not raw["user_id"]:
        raise ScenarioValidationError(f"{where}.user_id must be a trusted identity string")
    if not isinstance(raw["user_input"], str) or not raw["user_input"]:
        raise ScenarioValidationError(f"{where}.user_input must be non-empty")
    if "trusted_idempotency_key" in raw:
        key = raw["trusted_idempotency_key"]
        if not isinstance(key, str) or not key:
            raise ScenarioValidationError(
                f"{where}.trusted_idempotency_key must be a non-empty string"
            )
        if key in raw["user_input"] or key in json.dumps(raw["proposals"]):
            raise ScenarioValidationError(
                f"{where} idempotency fixture must remain outside model input"
            )

    _validate_initial_state(raw, where)
    _validate_proposals(raw["proposals"], where)
    _validate_decisions(raw["human_decisions"], where)
    if (
        not isinstance(raw["clock_advance_seconds"], (int, float))
        or isinstance(raw["clock_advance_seconds"], bool)
        or raw["clock_advance_seconds"] < 0
    ):
        raise ScenarioValidationError(f"{where}.clock_advance_seconds must be non-negative")
    _validate_state_changes(raw["pre_resume_state_changes"], raw, where)
    _validate_fault(raw["fault"], raw["proposals"], where)
    _validate_expected(raw["expected"], raw, where)
    _validate_idempotency_expectation(raw, where)


def _validate_idempotency_expectation(raw: dict[str, Any], where: str) -> None:
    behavior = raw["expected"]["authorized_behavior"]
    seeds = raw["initial_idempotency"]
    key = raw.get("trusted_idempotency_key")
    proposal = raw["proposals"][0]
    if behavior in {"replay", "conflict"}:
        if key is None:
            raise ScenarioValidationError(
                f"{where} needs a trusted idempotency key for replay/conflict validation"
            )
        matching_seed = next(
            (
                seed
                for seed in seeds
                if seed["order_id"] == proposal["order_id"]
                and seed["idempotency_key"] == key
            ),
            None,
        )
        if matching_seed is None:
            raise ScenarioValidationError(
                f"{where} idempotency expectation lacks a matching initial record"
            )
        same_request = (
            proposal.get("amount") == matching_seed["amount"]
            and proposal.get("currency") == matching_seed["currency"]
        )
        if (behavior == "replay") != same_request:
            raise ScenarioValidationError(
                f"{where} idempotency key/request does not match expected {behavior}"
            )
    if behavior == "block" and seeds and key is not None:
        if any(
            seed["order_id"] == proposal["order_id"]
            and seed["idempotency_key"] == key
            for seed in seeds
        ):
            raise ScenarioValidationError(
                f"{where} expected a distinct idempotency key but reused a seed key"
            )


def _validate_initial_state(raw: dict[str, Any], where: str) -> None:
    orders = raw["initial_orders"]
    seeds = raw["initial_idempotency"]
    if not isinstance(orders, list) or not orders:
        raise ScenarioValidationError(f"{where}.initial_orders must be a non-empty list")
    if not isinstance(seeds, list):
        raise ScenarioValidationError(f"{where}.initial_idempotency must be a list")
    order_ids: set[str] = set()
    for order in orders:
        if not isinstance(order, dict) or set(order) != {
            "order_id", "owner_id", "total_amount", "currency", "status"
        }:
            raise ScenarioValidationError(f"{where} has an invalid order fixture")
        if any(not isinstance(order[key], str) or not order[key] for key in (
            "order_id", "owner_id", "total_amount", "currency"
        )):
            raise ScenarioValidationError(f"{where} order fields must be non-empty strings")
        if order["order_id"] in order_ids:
            raise ScenarioValidationError(f"{where} has duplicate order IDs")
        order_ids.add(order["order_id"])
        if not isinstance(order["status"], str) or order["status"] not in {
            status.value for status in OrderStatus
        }:
            raise ScenarioValidationError(f"{where} has an invalid order status")
        _validate_decimal_string(order["total_amount"], f"{where}.total_amount")
    seed_keys: set[tuple[str, str]] = set()
    seeded_orders: set[str] = set()
    for seed in seeds:
        if not isinstance(seed, dict) or set(seed) != {
            "order_id", "idempotency_key", "amount", "currency"
        }:
            raise ScenarioValidationError(f"{where} has an invalid idempotency seed")
        if seed["order_id"] not in order_ids:
            raise ScenarioValidationError(f"{where} idempotency seed references unknown order")
        if any(not isinstance(seed[key], str) or not seed[key] for key in (
            "idempotency_key", "currency"
        )):
            raise ScenarioValidationError(f"{where} idempotency seed has invalid identifiers")
        _validate_decimal_string(seed["amount"], f"{where}.initial_idempotency.amount")
        key = (seed["order_id"], seed["idempotency_key"])
        if key in seed_keys:
            raise ScenarioValidationError(f"{where} has duplicate idempotency seeds")
        seed_keys.add(key)
        if seed["order_id"] in seeded_orders:
            raise ScenarioValidationError(
                f"{where} cannot seed multiple completed refunds for one order"
            )
        seeded_orders.add(seed["order_id"])
        order = next(item for item in orders if item["order_id"] == seed["order_id"])
        if order["status"] != OrderStatus.REFUNDED.value:
            raise ScenarioValidationError(
                f"{where} seeded refund order must start in refunded state"
            )
        if seed["amount"] != order["total_amount"] or seed["currency"] != order["currency"]:
            raise ScenarioValidationError(
                f"{where} initial refund must match the order's full amount and currency"
            )
    refunded_orders = {
        order["order_id"]
        for order in orders
        if order["status"] == OrderStatus.REFUNDED.value
    }
    if refunded_orders != seeded_orders:
        raise ScenarioValidationError(
            f"{where} refunded orders and initial idempotency records must match"
        )


def _validate_proposals(proposals: Any, where: str) -> None:
    if not isinstance(proposals, list) or not proposals:
        raise ScenarioValidationError(f"{where}.proposals must be a non-empty list")
    for proposal in proposals:
        if not isinstance(proposal, dict) or proposal.get("type") != "tool_call":
            raise ScenarioValidationError(f"{where} proposals must be tool_call objects")
        if (
            not isinstance(proposal.get("operation"), str)
            or proposal["operation"] not in PROPOSAL_OPERATIONS
        ):
            raise ScenarioValidationError(f"{where} proposal operation is invalid")
        if not isinstance(proposal.get("order_id"), str) or not proposal["order_id"]:
            raise ScenarioValidationError(f"{where} proposal order_id is invalid")
        if proposal["operation"] == "refund_order":
            if not isinstance(proposal.get("amount"), str) or not isinstance(
                proposal.get("currency"), str
            ):
                raise ScenarioValidationError(f"{where} refund proposal lacks amount/currency")
            _validate_decimal_string(proposal["amount"], f"{where}.proposal.amount")


def _validate_decisions(decisions: Any, where: str) -> None:
    if not isinstance(decisions, list) or any(
        decision not in {"approve", "reject"} for decision in decisions
    ):
        raise ScenarioValidationError(f"{where}.human_decisions is invalid")


def _validate_state_changes(changes: Any, raw: dict[str, Any], where: str) -> None:
    if not isinstance(changes, list):
        raise ScenarioValidationError(f"{where}.pre_resume_state_changes must be a list")
    order_ids = {order["order_id"] for order in raw["initial_orders"]}
    for change in changes:
        if not isinstance(change, dict) or set(change) != {"order_id", "status"}:
            raise ScenarioValidationError(f"{where} has an invalid pre-resume state change")
        if change["order_id"] not in order_ids or change["status"] not in {
            status.value for status in OrderStatus
        }:
            raise ScenarioValidationError(f"{where} pre-resume state change is invalid")


def _validate_fault(fault: Any, proposals: list[dict[str, Any]], where: str) -> None:
    if fault is None:
        return
    if not isinstance(fault, dict) or set(fault) != {"proposal_index", "mode"}:
        raise ScenarioValidationError(f"{where}.fault must contain proposal_index and mode")
    if fault["mode"] not in FAULT_MODES - {FaultMode.NONE.value}:
        raise ScenarioValidationError(f"{where}.fault mode is invalid")
    index = fault["proposal_index"]
    if (
        not isinstance(index, int)
        or isinstance(index, bool)
        or index < 0
        or index >= len(proposals)
        or proposals[index]["operation"] != "refund_order"
    ):
        raise ScenarioValidationError(f"{where}.fault must target a refund proposal")


def _validate_expected(expected: Any, raw: dict[str, Any], where: str) -> None:
    if (
        not isinstance(expected, dict)
        or _REQUIRED_EXPECTED_KEYS - expected.keys()
        or expected.keys() - (_REQUIRED_EXPECTED_KEYS | _OPTIONAL_EXPECTED_KEYS)
    ):
        raise ScenarioValidationError(f"{where}.expected has missing or extra fields")
    if not isinstance(expected["task_success_eligible"], bool):
        raise ScenarioValidationError(f"{where}.task_success_eligible must be boolean")
    if expected["authorized_behavior"] not in AUTHORIZATION_BEHAVIORS:
        raise ScenarioValidationError(f"{where}.authorized_behavior is invalid")
    if expected["final_response_class"] not in RESPONSE_CLASSES:
        raise ScenarioValidationError(f"{where}.final_response_class is invalid")
    for field, allowed in (
        ("policy_outcomes", POLICY_OUTCOMES),
        ("confirmation_statuses", CONFIRMATION_STATUSES),
    ):
        if not isinstance(expected[field], list) or any(
            value not in allowed for value in expected[field]
        ):
            raise ScenarioValidationError(f"{where}.expected.{field} is invalid")
    for field in ("tool_attempts", "tool_invocations", "state_mutations"):
        if (
            not isinstance(expected[field], int)
            or isinstance(expected[field], bool)
            or expected[field] < 0
        ):
            raise ScenarioValidationError(f"{where}.expected.{field} must be non-negative")
    if expected["tool_invocations"] > expected["tool_attempts"]:
        raise ScenarioValidationError(
            f"{where} tool invocations cannot exceed tool attempts"
        )
    if expected["state_mutations"] > expected["tool_invocations"]:
        raise ScenarioValidationError(
            f"{where} state mutations cannot exceed tool invocations"
        )
    final_state = expected["final_state"]
    if not isinstance(final_state, dict) or set(final_state) != {
        "order_statuses", "refund_count", "idempotency_record_count"
    }:
        raise ScenarioValidationError(f"{where}.expected.final_state is invalid")
    if not isinstance(final_state["order_statuses"], dict):
        raise ScenarioValidationError(f"{where} final order statuses must be an object")
    initial_ids = {order["order_id"] for order in raw["initial_orders"]}
    if set(final_state["order_statuses"]) != initial_ids or any(
        status not in {item.value for item in OrderStatus}
        for status in final_state["order_statuses"].values()
    ):
        raise ScenarioValidationError(f"{where} final state must specify each fixture order")
    for field in ("refund_count", "idempotency_record_count"):
        if not isinstance(final_state[field], int) or final_state[field] < 0:
            raise ScenarioValidationError(f"{where} final {field} must be non-negative")
    non_blocked_decisions = sum(
        outcome in {"allowed", "replay"}
        for outcome in expected["policy_outcomes"]
    )
    if non_blocked_decisions != expected["tool_invocations"]:
        raise ScenarioValidationError(
            f"{where} allowed/replay policy decisions must match tool invocations"
        )
    fresh_confirmation = expected["fresh_confirmation_required"]
    if not isinstance(fresh_confirmation, bool):
        raise ScenarioValidationError(
            f"{where}.expected.fresh_confirmation_required must be boolean"
        )
    fault_outcome = expected["fault_outcome"]
    expected_fault_outcomes = {
        FaultMode.TIMEOUT_BEFORE_COMMIT.value: "uncommitted_timeout",
        FaultMode.TIMEOUT_AFTER_COMMIT.value: "committed_timeout_replayed",
        FaultMode.DUPLICATE_RESPONSE.value: "duplicate_response",
    }
    if raw["fault"] is None:
        if fault_outcome is not None:
            raise ScenarioValidationError(f"{where} declares a fault outcome without a fault")
    elif fault_outcome != expected_fault_outcomes[raw["fault"]["mode"]]:
        raise ScenarioValidationError(f"{where} fault outcome does not match injected fault")
    if fresh_confirmation != (
        raw["fault"] is not None
        and raw["fault"]["mode"] == FaultMode.TIMEOUT_BEFORE_COMMIT.value
    ):
        raise ScenarioValidationError(
            f"{where} fresh-confirmation expectation is inconsistent with fault"
        )

    _validate_expected_transitions(expected, raw, where)
    _validate_execution_path(expected, raw, where)
    _validate_fault_path(expected, raw, where)


def _validate_execution_path(
    expected: dict[str, Any],
    raw: dict[str, Any],
    where: str,
) -> None:
    behavior = expected["authorized_behavior"]
    outcomes = expected["policy_outcomes"]
    statuses = expected["confirmation_statuses"]
    decisions = raw["human_decisions"]
    attempts = expected["tool_attempts"]
    invocations = expected["tool_invocations"]
    mutations = expected["state_mutations"]
    response = expected["final_response_class"]

    if attempts < len(raw["proposals"]) or attempts > len(raw["proposals"]) + 1:
        raise ScenarioValidationError(
            f"{where} attempts must cover proposals and at most one automatic retry"
        )
    if behavior == "allow":
        valid = (
            outcomes == ["allowed"]
            and not statuses
            and not decisions
            and invocations == 1
            and response == "completed"
        )
    elif behavior == "block":
        valid = (
            outcomes
            and set(outcomes) == {"blocked"}
            and not invocations
            and not mutations
            and not statuses
            and response == "denied"
        )
    elif behavior == "conflict":
        valid = (
            outcomes == ["blocked"]
            and not statuses
            and not invocations
            and not mutations
            and response == "denied"
        )
    elif behavior == "replay":
        valid = (
            outcomes == ["replay"]
            and not statuses
            and invocations == 1
            and not mutations
            and response == "completed"
        )
    else:
        valid = _valid_confirmation_path(
            outcomes,
            statuses,
            decisions,
            attempts=attempts,
            invocations=invocations,
            mutations=mutations,
            response=response,
        )
    if not valid:
        raise ScenarioValidationError(
            f"{where} authorization, policy, confirmation, and execution expectations conflict"
        )


def _valid_confirmation_path(
    outcomes: list[str],
    statuses: list[str],
    decisions: list[str],
    *,
    attempts: int,
    invocations: int,
    mutations: int,
    response: str,
) -> bool:
    if not statuses or statuses[0] != "requested":
        return False
    if not outcomes or outcomes[0] != "blocked":
        return False
    if "requested" in statuses[1:] and statuses[-1] != "requested":
        return False
    if decisions.count("approve") > 1 or decisions.count("reject") > 1:
        return False
    if decisions and decisions[-1] == "reject":
        return (
            statuses == ["requested", "rejected"]
            and outcomes == ["blocked"]
            and invocations == mutations == 0
            and response == "denied"
        )
    if "expired" in statuses:
        return (
            statuses == ["requested", "expired"]
            and decisions == ["approve"]
            and outcomes == ["blocked", "blocked"]
            and invocations == mutations == 0
            and response == "denied"
        )
    if decisions == ["approve"] and statuses == ["requested", "consumed"]:
        return (
            outcomes == ["blocked", "blocked"]
            and invocations == mutations == 0
            and response == "denied"
        )
    if decisions == ["approve"] and statuses == [
        "requested",
        "confirmed",
        "consumed",
    ]:
        if outcomes == ["blocked", "allowed"]:
            return (
                attempts == 1
                and invocations == mutations == 1
                and response == "completed"
            )
        return (
            outcomes == ["blocked", "allowed", "replay"]
            and attempts == 2
            and invocations == 2
            and mutations == 1
            and response == "completed"
        )
    if decisions == ["approve"] and statuses == [
        "requested",
        "confirmed",
        "consumed",
        "requested",
    ]:
        if outcomes == ["blocked", "allowed", "blocked"]:
            return (
                attempts == 2
                and invocations == 1
                and mutations == 0
                and response == "confirmation_required"
            )
        return False
    return (
        statuses == ["requested"]
        and decisions == []
        and outcomes == ["blocked"]
        and invocations == mutations == 0
        and response == "confirmation_required"
    )


def _validate_fault_path(
    expected: dict[str, Any],
    raw: dict[str, Any],
    where: str,
) -> None:
    fault = raw["fault"]
    if fault is None:
        return
    mode = fault["mode"]
    if mode == FaultMode.TIMEOUT_BEFORE_COMMIT.value:
        valid = (
            expected["authorized_behavior"] == "confirm"
            and expected["policy_outcomes"] == ["blocked", "allowed", "blocked"]
            and expected["confirmation_statuses"]
            == ["requested", "confirmed", "consumed", "requested"]
            and raw["human_decisions"] == ["approve"]
            and expected["tool_attempts"] == 2
            and expected["tool_invocations"] == 1
            and expected["state_mutations"] == 0
            and expected["final_response_class"] == "confirmation_required"
            and expected["fresh_confirmation_required"] is True
        )
    elif mode == FaultMode.TIMEOUT_AFTER_COMMIT.value:
        valid = (
            expected["authorized_behavior"] == "confirm"
            and expected["policy_outcomes"] == ["blocked", "allowed", "replay"]
            and expected["confirmation_statuses"]
            == ["requested", "confirmed", "consumed"]
            and expected["tool_attempts"] == 2
            and expected["tool_invocations"] == 2
            and expected["state_mutations"] == 1
            and expected["final_response_class"] == "completed"
            and expected["fresh_confirmation_required"] is False
        )
    else:
        valid = (
            mode == FaultMode.DUPLICATE_RESPONSE.value
            and expected["authorized_behavior"] == "confirm"
            and expected["policy_outcomes"] == ["blocked", "allowed"]
            and expected["confirmation_statuses"]
            == ["requested", "confirmed", "consumed"]
            and expected["tool_attempts"] == 1
            and expected["tool_invocations"] == 1
            and expected["state_mutations"] == 1
            and expected["final_response_class"] == "completed"
            and expected["fresh_confirmation_required"] is False
        )
    if not valid:
        raise ScenarioValidationError(
            f"{where} fault injection and expected execution path are inconsistent"
        )


def _validate_expected_transitions(
    expected: dict[str, Any],
    raw: dict[str, Any],
    where: str,
) -> None:
    initial_statuses = {
        order["order_id"]: order["status"] for order in raw["initial_orders"]
    }
    for change in raw["pre_resume_state_changes"]:
        initial_statuses[change["order_id"]] = change["status"]
    final_statuses = expected["final_state"]["order_statuses"]
    changed = {
        order_id
        for order_id, status in final_statuses.items()
        if initial_statuses[order_id] != status
    }
    if len(changed) != expected["state_mutations"]:
        raise ScenarioValidationError(
            f"{where} final order transitions must match expected tool mutations"
        )

    initial_refunds = len(raw["initial_idempotency"])
    final_state = expected["final_state"]
    refund_delta = final_state["refund_count"] - initial_refunds
    idempotency_delta = (
        final_state["idempotency_record_count"] - initial_refunds
    )
    if refund_delta < 0 or idempotency_delta != refund_delta:
        raise ScenarioValidationError(
            f"{where} final refund and idempotency counts are inconsistent"
        )
    if refund_delta > expected["state_mutations"]:
        raise ScenarioValidationError(
            f"{where} has more new refunds than expected state mutations"
        )
    if final_state["refund_count"] != final_state["idempotency_record_count"]:
        raise ScenarioValidationError(
            f"{where} final refund/idempotency records are not atomic"
        )

    statuses = expected["confirmation_statuses"]
    decisions = raw["human_decisions"]
    if decisions.count("approve") > statuses.count("confirmed"):
        if "expired" not in statuses and "consumed" not in statuses:
            raise ScenarioValidationError(
                f"{where} approval lacks a corresponding confirmation lifecycle"
            )
    if decisions.count("reject") > statuses.count("rejected"):
        raise ScenarioValidationError(
            f"{where} rejection lacks a rejected confirmation status"
        )
    if any(status in {"confirmed", "rejected", "expired", "consumed"} for status in statuses):
        if "requested" not in statuses:
            raise ScenarioValidationError(
                f"{where} confirmation terminal status lacks a request"
            )
    if raw["fault"] is not None and raw["fault"]["mode"] in {
        FaultMode.TIMEOUT_BEFORE_COMMIT.value,
        FaultMode.TIMEOUT_AFTER_COMMIT.value,
        FaultMode.DUPLICATE_RESPONSE.value,
    }:
        if raw["proposals"][raw["fault"]["proposal_index"]]["operation"] != "refund_order":
            raise ScenarioValidationError(f"{where} fault must target a refund")


def _validate_decimal_string(value: str, where: str) -> None:
    from decimal import Decimal, InvalidOperation

    try:
        parsed = Decimal(value)
    except (InvalidOperation, ValueError) as error:
        raise ScenarioValidationError(f"{where} must be a decimal string") from error
    if not parsed.is_finite() or parsed <= 0:
        raise ScenarioValidationError(f"{where} must be a positive finite decimal")
