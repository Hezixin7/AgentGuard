"""Deterministic corruption corpus for Stage 4B Phase 2.

This module applies declared mutations to frozen records and validates the
mutation target, event reference, expected step, and causal category. It must
never import or call ``evaluation.localization``.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import replace
import json
from pathlib import Path
from typing import Any, Mapping

CORRUPTION_SCHEMA_VERSION = "1.0"
CORRUPTION_MANIFEST_VERSION = "localization-corruptions-v1"
CATEGORIES = {
    "proven_earliest",
    "observed_unproven",
    "unobservable",
}


def load_corruption_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        manifest = json.load(handle)
    if not isinstance(manifest, dict):
        raise ValueError("corruption manifest must be a JSON object")
    return manifest


def _scenario_by_id(dataset: Any, scenario_id: str) -> dict[str, Any]:
    for scenario in dataset.all_scenarios:
        if scenario["id"] == scenario_id:
            return scenario
    raise KeyError(f"unknown scenario: {scenario_id}")


def _record_by_key(batch: Any, scenario_id: str, group: str) -> Any:
    for record in batch.records:
        if record.scenario_id == scenario_id and record.group == group:
            return record
    raise KeyError(f"unknown run record: {scenario_id}/{group}")


def _require_capture(record: Any) -> Any:
    if record.observations is None:
        raise ValueError("record has no observations")
    capture = record.observations.trace_capture
    if capture is None:
        raise ValueError("record has no trace capture")
    return capture


def apply_corruption(
    case: Mapping[str, Any],
    scenario: dict[str, Any],
    record: Any,
) -> tuple[dict[str, Any], Any]:
    """Apply one declared mutation and return ``(scenario, record)``."""
    mutation = case["mutation"]
    kind = mutation["kind"]
    observations = record.observations
    if observations is None:
        raise ValueError("record has no observations")

    if kind == "trace_event_fields":
        capture = _require_capture(record)
        events = [dict(event) for event in capture.events]
        index = mutation["event_index"]
        events[index].update(mutation["fields"])
        observations = replace(
            observations,
            trace_capture=replace(capture, events=tuple(events)),
        )
    elif kind == "trace_drop_event":
        capture = _require_capture(record)
        events = [dict(event) for event in capture.events]
        index = mutation["event_index"]
        events.pop(index)
        if mutation.get("renumber", False):
            events = [
                dict(event, step=position + 1)
                for position, event in enumerate(events)
            ]
        observations = replace(
            observations,
            trace_capture=replace(capture, events=tuple(events)),
            trace_event_count=len(events),
        )
    elif kind == "trace_capture_none":
        observations = replace(
            observations,
            trace_capture=None,
            trace_event_count=0,
        )
    elif kind == "runner_policy_decisions":
        observations = replace(
            observations,
            policy_decisions=tuple(
                dict(decision) for decision in mutation["value"]
            ),
        )
    elif kind == "runner_confirmation_statuses":
        observations = replace(
            observations,
            confirmation_statuses=tuple(mutation["value"]),
        )
    elif kind == "runner_tool_result_fields":
        results = list(observations.tool_results)
        index = mutation["index"]
        results[index] = replace(results[index], **mutation["fields"])
        observations = replace(observations, tool_results=tuple(results))
    elif kind == "runner_field":
        observations = replace(
            observations,
            **{mutation["field"]: mutation["value"]},
        )
    elif kind == "runner_order_state":
        order_states = {
            order_id: dict(state)
            for order_id, state in observations.order_states.items()
        }
        order_states[mutation["order_id"]][mutation["field"]] = mutation[
            "value"
        ]
        observations = replace(observations, order_states=order_states)
    else:
        raise ValueError(f"unsupported corruption kind: {kind}")

    return scenario, replace(record, observations=observations)


def _events(record: Any) -> list[dict[str, Any]]:
    capture = _require_capture(record)
    return [dict(event) for event in capture.events]


def _steps_contiguous(events: list[dict[str, Any]]) -> bool:
    return [event["step"] for event in events] == list(
        range(1, len(events) + 1)
    )


def _calls(events: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    calls: dict[str, dict[str, Any]] = {}
    for event in events:
        if event["event_type"] == "tool_call":
            calls[event["tool_call_ref"]] = event
    return calls


def _unknown_references(events: list[dict[str, Any]]) -> list[str]:
    calls: set[str] = set()
    unknown: list[str] = []
    for event in events:
        event_type = event["event_type"]
        if event_type == "tool_call":
            calls.add(event["tool_call_ref"])
        elif event_type in {
            "policy_decision",
            "human_confirmation",
            "tool_result",
        }:
            if event.get("tool_call_ref") not in calls:
                unknown.append(event["tool_call_ref"])
    return unknown


def _result_sequence_monotonic(events: list[dict[str, Any]]) -> bool:
    ordinal = {
        event["tool_call_ref"]: index
        for index, event in enumerate(
            event for event in events if event["event_type"] == "tool_call"
        )
    }
    last = -1
    for event in events:
        if event["event_type"] != "tool_result":
            continue
        current = ordinal[event["tool_call_ref"]]
        if current <= last:
            return False
        last = current
    return True


def _unknown_final_citations(events: list[dict[str, Any]]) -> list[str]:
    calls = {
        event["tool_call_ref"]
        for event in events
        if event["event_type"] == "tool_call"
    }
    unknown: list[str] = []
    for event in events:
        if event["event_type"] != "final_response":
            continue
        for ref in event.get("cited_tool_call_refs") or []:
            if ref not in calls:
                unknown.append(ref)
    return unknown


def _sequence_conflict_indices(
    trace_values: list[Any],
    runner_values: list[Any],
) -> list[int]:
    sentinel = object()
    return [
        index
        for index in range(max(len(trace_values), len(runner_values)))
        if (
            trace_values[index]
            if index < len(trace_values)
            else sentinel
        )
        != (
            runner_values[index]
            if index < len(runner_values)
            else sentinel
        )
    ]


def _predicate_index(predicate_id: str) -> int | None:
    if "[" not in predicate_id or not predicate_id.endswith("]"):
        return None
    value = predicate_id.rsplit("[", 1)[1][:-1]
    try:
        return int(value)
    except ValueError:
        return None


def _changed_indices(
    before: list[dict[str, Any]],
    after: list[dict[str, Any]],
) -> list[int]:
    if len(before) != len(after):
        return list(range(max(len(before), len(after))))
    return [
        index
        for index, (left, right) in enumerate(zip(before, after))
        if left != right
    ]


def _validate_trace_event_change(
    case: Mapping[str, Any],
    before_events: list[dict[str, Any]],
    after_events: list[dict[str, Any]],
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    mutation = case["mutation"]
    index = mutation["event_index"]
    declared_step = mutation["declared_event_step"]
    fields = mutation["fields"]
    if index >= len(before_events):
        return {}, ["declared event index does not exist"]
    if before_events[index]["step"] != declared_step:
        errors.append("declared event step does not match the frozen event")

    changed = _changed_indices(before_events, after_events)
    changed_fields = sorted(
        key
        for key in fields
        if before_events[index].get(key) != after_events[index].get(key)
    )
    if changed != [index]:
        errors.append(f"mutation changed unexpected events: {changed}")
    if changed_fields != sorted(fields):
        errors.append(
            "declared fields did not all change: "
            f"changed={changed_fields}"
        )
    for key in ("event_type", "tool_call_ref", "operation"):
        if key not in fields and before_events[index].get(
            key
        ) != after_events[index].get(key):
            errors.append(f"undeclared structural field changed: {key}")

    summary = {
        "event_index": index,
        "event_step": before_events[index]["step"],
        "mutated_event_step": after_events[index]["step"],
        "changed_event_indices": changed,
        "changed_fields": changed_fields,
    }
    return summary, errors


def _proven_checks(
    case: Mapping[str, Any],
    before_events: list[dict[str, Any]],
    after_events: list[dict[str, Any]],
) -> tuple[dict[str, bool], list[str]]:
    errors: list[str] = []
    expected = case["expected"]
    index = case["mutation"]["event_index"]
    changed = _changed_indices(before_events, after_events)
    checks: dict[str, bool] = {
        "single_event_changed": changed == [index],
        "prior_events_unchanged": all(
            before_events[position] == after_events[position]
            for position in range(min(index, len(after_events)))
        ),
        "trace_steps_contiguous": _steps_contiguous(after_events),
        "result_references_valid": not _unknown_references(after_events),
        "result_sequence_monotonic": _result_sequence_monotonic(
            after_events
        ),
        "earliest_changed_step_matches_expected": (
            changed == [index]
            and after_events[index]["step"] == expected["step"]
        ),
    }
    if len(case["mutation"]["fields"]) > 1:
        checks["multi_field_single_event"] = changed == [index]
    for name in case.get("causal_checks", []):
        if checks.get(name) is False:
            errors.append(f"causal check failed: {name}")
    return checks, errors


def _validate_unobservable_predicate(
    predicate: str,
    scenario: Mapping[str, Any],
    record: Any,
) -> tuple[bool, str]:
    capture = record.observations.trace_capture
    events = (
        [dict(event) for event in capture.events]
        if capture is not None
        else []
    )
    if predicate == "trace_steps_not_contiguous":
        return not _steps_contiguous(events), (
            "mutated trace steps are not contiguous"
        )
    if predicate == "unknown_tool_call_reference":
        return bool(_unknown_references(events)), (
            "lifecycle event references a missing tool call"
        )
    if predicate == "unknown_final_citation":
        return bool(_unknown_final_citations(events)), (
            "final response cites an unknown tool call"
        )
    if predicate == "external_mutation_mismatch":
        expected = len(scenario["pre_resume_state_changes"])
        actual = record.observations.external_state_mutations
        return actual != expected, (
            "external mutation count differs from fixture context"
        )
    if predicate == "trace_capture_missing":
        return record.observations.trace_capture is None, (
            "trace capture is absent"
        )
    return False, f"unsupported unobservable predicate: {predicate}"


def validate_corruption_case(
    case: Mapping[str, Any],
    dataset: Any,
    batch: Any,
) -> tuple[dict[str, Any], list[str]]:
    errors: list[str] = []
    scenario = _scenario_by_id(dataset, case["base_scenario_id"])
    record = _record_by_key(
        batch,
        case["base_scenario_id"],
        case["group"],
    )
    if record.observations is None:
        return {}, ["base record has no observations"]

    before_events: list[dict[str, Any]] | None = None
    capture = record.observations.trace_capture
    if capture is not None:
        before_events = [dict(event) for event in capture.events]
    before_observations = record.observations

    _, mutated_record = apply_corruption(case, scenario, record)
    if mutated_record.observations is None:
        return {}, ["mutated record lost observations"]

    summary: dict[str, Any] = {
        "case_id": case["case_id"],
        "category": case["category"],
        "base_scenario_id": case["base_scenario_id"],
        "group": case["group"],
    }
    kind = case["mutation"]["kind"]
    after_capture = mutated_record.observations.trace_capture

    if kind == "trace_event_fields":
        after_events = [dict(event) for event in after_capture.events]
        event_summary, event_errors = _validate_trace_event_change(
            case,
            before_events or [],
            after_events,
        )
        errors.extend(event_errors)
        summary.update(event_summary)
        if case["category"] == "proven_earliest":
            checks, check_errors = _proven_checks(
                case,
                before_events or [],
                after_events,
            )
            summary["causal_checks"] = checks
            errors.extend(check_errors)
    elif kind == "trace_drop_event":
        after_events = [dict(event) for event in after_capture.events]
        index = case["mutation"]["event_index"]
        if before_events is None or index >= len(before_events):
            errors.append("declared drop index does not exist")
        else:
            summary["event_index"] = index
            summary["dropped_event_step"] = before_events[index]["step"]
        if len(after_events) != len(before_events or []) - 1:
            errors.append("drop mutation did not remove exactly one event")
        if case["mutation"].get("renumber") and not _steps_contiguous(
            after_events
        ):
            errors.append("renumbered trace steps are not contiguous")
    elif kind == "trace_capture_none":
        if after_capture is not None:
            errors.append("trace capture was not removed")
    elif kind in {
        "runner_policy_decisions",
        "runner_confirmation_statuses",
        "runner_tool_result_fields",
        "runner_field",
        "runner_order_state",
    }:
        if after_capture is not None and before_events is not None:
            after_events = [dict(event) for event in after_capture.events]
            if before_events != after_events:
                errors.append("runner mutation altered the trace")
        changed_target = True
        if kind == "runner_policy_decisions":
            changed_target = (
                before_observations.policy_decisions
                != mutated_record.observations.policy_decisions
            )
        elif kind == "runner_confirmation_statuses":
            changed_target = (
                before_observations.confirmation_statuses
                != mutated_record.observations.confirmation_statuses
            )
        elif kind == "runner_tool_result_fields":
            index = case["mutation"]["index"]
            changed_target = (
                before_observations.tool_results[index]
                != mutated_record.observations.tool_results[index]
            )
        elif kind == "runner_field":
            field = case["mutation"]["field"]
            changed_target = (
                getattr(before_observations, field)
                != getattr(mutated_record.observations, field)
            )
        elif kind == "runner_order_state":
            order_id = case["mutation"]["order_id"]
            field = case["mutation"]["field"]
            changed_target = (
                before_observations.order_states[order_id][field]
                != mutated_record.observations.order_states[order_id][
                    field
                ]
            )
        if not changed_target:
            errors.append("declared runner target did not change")

        expected_predicate = case["expected"].get("predicate_id", "")
        mutated_observations = mutated_record.observations
        if kind == "runner_policy_decisions":
            if before_events is None:
                errors.append("policy conflict requires a trace")
            else:
                trace_values = [
                    event.get("decision")
                    for event in before_events
                    if event["event_type"] == "policy_decision"
                ]
                runner_values = [
                    decision.get("outcome")
                    for decision in mutated_observations.policy_decisions
                ]
                conflicts = _sequence_conflict_indices(
                    trace_values,
                    runner_values,
                )
                summary["contradiction_indices"] = conflicts
                expected_index = _predicate_index(expected_predicate)
                if conflicts != [expected_index]:
                    errors.append(
                        "runner policy contradiction does not match "
                        f"declared index: {conflicts}"
                    )
        elif kind == "runner_confirmation_statuses":
            if before_events is None:
                errors.append("confirmation conflict requires a trace")
            else:
                trace_values = [
                    event.get("status")
                    for event in before_events
                    if event["event_type"] == "human_confirmation"
                ]
                runner_values = list(
                    mutated_observations.confirmation_statuses
                )
                conflicts = _sequence_conflict_indices(
                    trace_values,
                    runner_values,
                )
                summary["contradiction_indices"] = conflicts
                expected_index = _predicate_index(expected_predicate)
                if conflicts != [expected_index]:
                    errors.append(
                        "runner confirmation contradiction does not match "
                        f"declared index: {conflicts}"
                    )
        elif kind == "runner_tool_result_fields":
            if before_events is None:
                errors.append("tool-result conflict requires a trace")
            else:
                trace_result = next(
                    event
                    for event in before_events
                    if event["event_type"] == "tool_result"
                )
                runner_result = mutated_observations.tool_results[0]
                conflict = (
                    bool(trace_result["state_changed"])
                    != bool(runner_result.committed)
                )
                summary["contradiction_indices"] = [0] if conflict else []
                if not conflict:
                    errors.append(
                        "runner tool-result mutation did not create a "
                        "declared contradiction"
                    )
        elif kind == "runner_field" and case["mutation"]["field"] == (
            "response_class"
        ):
            expected_response = scenario["expected"]["final_response_class"]
            conflict = (
                mutated_observations.response_class != expected_response
            )
            summary["contradiction_observed"] = conflict
            if not conflict:
                errors.append(
                    "runner response class still matches the frozen oracle"
                )
        elif kind == "runner_order_state":
            order_id = case["mutation"]["order_id"]
            expected_status = scenario["expected"]["final_state"][
                "order_statuses"
            ][order_id]
            actual_status = mutated_observations.order_states[order_id][
                "status"
            ]
            conflict = actual_status != expected_status
            summary["contradiction_observed"] = conflict
            if not conflict:
                errors.append(
                    "runner order state still matches the frozen oracle"
                )
    else:
        errors.append(f"unsupported corruption kind: {kind}")

    expected = case["expected"]
    if case["category"] == "proven_earliest":
        if expected.get("step_scope") != "causal_earliest":
            errors.append("proven case must declare causal_earliest step")
        if expected.get("causal_label") != "proven_earliest":
            errors.append("proven case must declare proven_earliest")
        if expected.get("source") != "agentguard_trace":
            errors.append("proven case must use trace source")
    elif case["category"] == "observed_unproven":
        if expected.get("causal_label") != "observed_violation":
            errors.append("unproven case must declare observed_violation")
        if expected.get("step_scope") == "causal_earliest":
            errors.append("unproven case must not claim causal step")
    elif case["category"] == "unobservable":
        if expected.get("status") != "unobservable":
            errors.append("unobservable case must declare status")
        if expected.get("causal_label") != "not_establishable":
            errors.append("unobservable case must declare not_establishable")
        predicate = case.get("validation", {}).get("predicate")
        valid, reason = _validate_unobservable_predicate(
            predicate,
            scenario,
            mutated_record,
        )
        summary["unobservable_predicate"] = {
            "predicate": predicate,
            "valid": valid,
            "reason": reason,
        }
        if not valid:
            errors.append(f"unobservable predicate not established: {predicate}")

    provenance = case.get("provenance", {})
    if provenance.get("localizer_used_for_label") is not False:
        errors.append("provenance must state localizer was not used")
    if not provenance.get("label_origin"):
        errors.append("provenance.label_origin is required")
    if not provenance.get("review_status"):
        errors.append("provenance.review_status is required")

    summary["causal_defensible"] = not errors
    return summary, errors


def validate_corruption_manifest(
    manifest: Mapping[str, Any],
    dataset: Any,
    batch: Any,
) -> dict[str, Any]:
    errors: list[str] = []
    if manifest.get("schema_version") != CORRUPTION_SCHEMA_VERSION:
        errors.append("unsupported corruption schema version")
    if manifest.get("manifest_version") != CORRUPTION_MANIFEST_VERSION:
        errors.append("unsupported corruption manifest version")
    cases = manifest.get("cases")
    if not isinstance(cases, list):
        return {
            "manifest_version": manifest.get("manifest_version"),
            "case_count": 0,
            "category_counts": {},
            "cases": [],
            "errors": [*errors, "cases must be a list"],
            "valid": False,
        }

    case_ids: set[str] = set()
    category_counts: Counter[str] = Counter()
    validated: list[dict[str, Any]] = []
    for index, case in enumerate(cases):
        case_id = case.get("case_id", f"index-{index}")
        if case_id in case_ids:
            errors.append(f"duplicate case id: {case_id}")
        case_ids.add(case_id)
        category = case.get("category")
        if category not in CATEGORIES:
            errors.append(f"{case_id}: invalid category {category!r}")
        category_counts[category] += 1
        summary, case_errors = validate_corruption_case(
            case,
            dataset,
            batch,
        )
        validated.append(summary)
        errors.extend(f"{case_id}: {error}" for error in case_errors)

    approval_scope = manifest.get("approval_scope", {})
    if approval_scope.get("catalog_wide_human_signoff") is not False:
        errors.append(
            "approval scope must not claim catalog-wide human sign-off"
        )
    if (
        approval_scope.get("independent_human_review")
        != "pending_for_all_labels"
    ):
        errors.append(
            "independent human review must remain pending for all labels"
        )

    conventions = manifest.get("scoring_conventions", [])
    if not isinstance(conventions, list):
        errors.append("scoring_conventions must be a list")
    else:
        cases_by_id = {
            case.get("case_id"): case for case in cases
        }
        for convention in conventions:
            convention_id = convention.get("convention_id")
            case = cases_by_id.get(convention.get("case_id"))
            if case is None:
                errors.append(
                    f"{convention_id}: unknown scoring-convention case"
                )
                continue
            approval = convention.get("approval", {})
            if approval.get("approved_by") != "project_owner":
                errors.append(
                    f"{convention_id}: approval must name project_owner"
                )
            if approval.get("status") != "approved":
                errors.append(
                    f"{convention_id}: approval status must be approved"
                )
            canonical = convention.get("canonical_gold_finding", {})
            expected = case.get("expected", {})
            for key in (
                "predicate_id",
                "predicate_family",
                "source",
                "step",
                "causal_label",
            ):
                if canonical.get(key) != expected.get(key):
                    errors.append(
                        f"{convention_id}: canonical {key} does not match "
                        "the gold label"
                    )
            supporting = convention.get("supporting_evidence_only", [])
            if canonical.get("predicate_id") in supporting:
                errors.append(
                    f"{convention_id}: canonical finding cannot be "
                    "supporting evidence"
                )
            if not convention.get("strict_earliest_divergence_rule"):
                errors.append(
                    f"{convention_id}: strict scoring rule is required"
                )
            if convention.get("gold_changed_to_match_predictions") is not False:
                errors.append(
                    f"{convention_id}: gold must not be changed to match "
                    "predictions"
                )
            provenance = case.get("provenance", {})
            if (
                provenance.get("approved_scoring_convention")
                != convention_id
            ):
                errors.append(
                    f"{convention_id}: case provenance does not reference "
                    "the approved convention"
                )

    return {
        "manifest_version": manifest.get("manifest_version"),
        "case_count": len(cases),
        "category_counts": dict(category_counts),
        "cases": validated,
        "errors": errors,
        "valid": not errors,
    }
