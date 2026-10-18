"""Deterministic scenario-catalog expansion for Stage 2."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
import re
from typing import Any

from evaluation.scenarios import (
    SCHEMA_VERSION,
    Scenario,
    ScenarioValidationError,
    validate_dataset,
)

CATALOG_VERSION = "1.0"
GENERATOR_VERSION = "stage-2b-1"
_VARIANT_ID = re.compile(r"^(core-\d{3})-v([1-3])$")
_FIXTURE_PATHS = {
    "user_input",
    "user_id",
    "initial_orders",
    "proposals",
    "human_decisions",
    "clock_advance_seconds",
    "pre_resume_state_changes",
    "fault",
    "trusted_idempotency_key",
}
_ORACLE_PATHS = {
    "expected.authorized_behavior",
    "expected.policy_outcomes",
    "expected.confirmation_statuses",
    "expected.final_response_class",
    "expected.tool_attempts",
    "expected.tool_invocations",
    "expected.state_mutations",
    "expected.final_state",
    "expected.fault_outcome",
    "expected.fresh_confirmation_required",
}


class ExpansionValidationError(ValueError):
    """Raised for an invalid or contradictory scenario expansion catalog."""


@dataclass(frozen=True)
class ExpandedScenario:
    """A child scenario paired with its parent and deterministic provenance."""

    scenario_id: str
    parent_scenario_id: str
    construction: str
    data: dict[str, Any]
    provenance: dict[str, Any]


def load_catalog(path: str | Path) -> dict[str, Any]:
    """Read a JSON transformation catalog without generating any scenarios."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExpansionValidationError(
            f"cannot load expansion catalog: {error}"
        ) from error
    if not isinstance(value, dict):
        raise ExpansionValidationError("expansion catalog must be an object")
    return value


def expand_catalog(
    core_scenarios: tuple[Scenario, ...],
    catalog: dict[str, Any],
    *,
    allow_partial: bool = True,
) -> tuple[ExpandedScenario, ...]:
    """Apply catalog transformations to frozen Stage 1 scenarios."""
    source_snapshot = _canonical([scenario.data for scenario in core_scenarios])
    _validate_core(core_scenarios)
    if _canonical([scenario.data for scenario in core_scenarios]) != source_snapshot:
        raise ExpansionValidationError("core dataset changed during validation")
    _validate_catalog_header(catalog, allow_partial=allow_partial)

    parents = {scenario.scenario_id: scenario for scenario in core_scenarios}
    entries = catalog["entries"]
    seen_ids: set[str] = set()
    seen_transforms: set[tuple[str, str]] = set()
    seen_content = {
        _canonical(_without_id(scenario.data))
        for scenario in core_scenarios
    }
    expanded: list[ExpandedScenario] = []

    for index, entry in enumerate(entries):
        where = f"entries[{index}]"
        _validate_entry_shape(entry, where)
        match = _VARIANT_ID.fullmatch(entry["id"])
        if match is None or match.group(1) != entry["parent_scenario_id"]:
            raise ExpansionValidationError(
                f"{where} ID must be <parent_scenario_id>-v1, -v2, or -v3"
            )
        parent_id = entry["parent_scenario_id"]
        parent = parents.get(parent_id)
        if parent is None:
            raise ExpansionValidationError(f"{where} references unknown parent {parent_id}")
        if entry["id"] in seen_ids:
            raise ExpansionValidationError(f"duplicate child ID: {entry['id']}")
        seen_ids.add(entry["id"])
        transform_key = (parent_id, entry["transform_id"])
        if transform_key in seen_transforms:
            raise ExpansionValidationError(
                f"duplicate transformation for parent {parent_id}"
            )
        seen_transforms.add(transform_key)

        child_data = deepcopy(parent.data)
        changed_fixture_paths = _apply_replacements(
            child_data,
            entry["changes"],
            _FIXTURE_PATHS,
            where,
        )
        changed_oracle_paths = _apply_replacements(
            child_data,
            entry["oracle_changes"],
            _ORACLE_PATHS,
            where,
        )
        _validate_proposal_schema(child_data, where)
        _validate_final_state_counter_types(child_data, where)
        if not changed_fixture_paths:
            raise ExpansionValidationError(
                f"{where} must change at least one meaningful fixture or execution field"
            )
        oracle_changed = bool(changed_oracle_paths)
        if entry["oracle_changed"] is not oracle_changed:
            raise ExpansionValidationError(
                f"{where}.oracle_changed does not match its oracle changes"
            )
        justification = entry["oracle_change_justification"]
        if oracle_changed and (not isinstance(justification, str) or not justification.strip()):
            raise ExpansionValidationError(
                f"{where} oracle changes require a justification"
            )
        if not oracle_changed and justification is not None:
            raise ExpansionValidationError(
                f"{where} must not justify an unchanged oracle"
            )
        if child_data["risk_labels"] != parent.data["risk_labels"]:
            raise ExpansionValidationError(f"{where} may not alter inherited risk labels")
        if child_data["classification"] != parent.data["classification"]:
            raise ExpansionValidationError(
                f"{where} may not alter inherited classification"
            )
        child_data["id"] = entry["id"]
        content_signature = _canonical(_without_id(child_data))
        if content_signature in seen_content:
            raise ExpansionValidationError(
                f"{where} duplicates scenario content apart from its ID"
            )
        seen_content.add(content_signature)

        _validate_child_with_frozen_validator(
            core_scenarios,
            parent_id,
            child_data,
            where,
        )
        expanded.append(
            ExpandedScenario(
                scenario_id=entry["id"],
                parent_scenario_id=parent_id,
                construction="deterministic_variant",
                data=child_data,
                provenance={
                    "catalog_version": catalog["catalog_version"],
                    "generator_version": GENERATOR_VERSION,
                    "transform_id": entry["transform_id"],
                    "boundary": entry["boundary"],
                    "rationale": entry["rationale"],
                    "changed_fixture_paths": changed_fixture_paths,
                    "changed_oracle_paths": changed_oracle_paths,
                    "oracle_changed": oracle_changed,
                    "oracle_change_justification": justification,
                    "evaluation_layer": "deterministic_offline",
                    "random_seed": None,
                },
            )
        )

    if not allow_partial:
        expected_ids = {
            f"{parent_id}-v{number}"
            for parent_id in parents
            for number in range(1, 4)
        }
        if seen_ids != expected_ids:
            raise ExpansionValidationError(
                "complete catalog must contain exactly three variants per core scenario"
            )
    return tuple(expanded)


def _validate_core(core_scenarios: tuple[Scenario, ...]) -> None:
    if not isinstance(core_scenarios, tuple):
        raise ExpansionValidationError("core scenarios must be a tuple from load_dataset()")
    try:
        validate_dataset(
            {
                "schema_version": SCHEMA_VERSION,
                "scenarios": [deepcopy(scenario.data) for scenario in core_scenarios],
            }
        )
    except ScenarioValidationError as error:
        raise ExpansionValidationError(f"frozen core dataset is invalid: {error}") from error


def _validate_catalog_header(catalog: dict[str, Any], *, allow_partial: bool) -> None:
    required = {
        "catalog_version",
        "source_schema_version",
        "complete",
        "entries",
    }
    if set(catalog) != required:
        raise ExpansionValidationError("catalog has missing or unexpected top-level fields")
    if catalog["catalog_version"] != CATALOG_VERSION:
        raise ExpansionValidationError(
            f"catalog_version must be {CATALOG_VERSION!r}"
        )
    if catalog["source_schema_version"] != SCHEMA_VERSION:
        raise ExpansionValidationError(
            f"source_schema_version must be {SCHEMA_VERSION!r}"
        )
    if not isinstance(catalog["complete"], bool):
        raise ExpansionValidationError("catalog complete must be boolean")
    if not isinstance(catalog["entries"], list) or not catalog["entries"]:
        raise ExpansionValidationError("catalog entries must be a non-empty list")
    if catalog["complete"] and len(catalog["entries"]) != 60:
        raise ExpansionValidationError(
            "a complete catalog must contain exactly 60 deterministic variants"
        )
    if not catalog["complete"] and not allow_partial:
        raise ExpansionValidationError(
            "partial catalog rejected when allow_partial is false"
        )
    if len(catalog["entries"]) > 60:
        raise ExpansionValidationError("catalog cannot exceed 60 deterministic variants")


def _validate_entry_shape(entry: Any, where: str) -> None:
    required = {
        "id",
        "parent_scenario_id",
        "transform_id",
        "boundary",
        "rationale",
        "changes",
        "oracle_changes",
        "oracle_changed",
        "oracle_change_justification",
    }
    if not isinstance(entry, dict) or set(entry) != required:
        raise ExpansionValidationError(f"{where} has missing or unexpected fields")
    for key in ("id", "parent_scenario_id", "transform_id", "boundary", "rationale"):
        if not isinstance(entry[key], str) or not entry[key].strip():
            raise ExpansionValidationError(f"{where}.{key} must be a non-empty string")
    if not isinstance(entry["oracle_changed"], bool):
        raise ExpansionValidationError(f"{where}.oracle_changed must be boolean")
    if not isinstance(entry["changes"], list) or not entry["changes"]:
        raise ExpansionValidationError(f"{where}.changes must be a non-empty list")
    if not isinstance(entry["oracle_change_justification"], (str, type(None))):
        raise ExpansionValidationError(
            f"{where}.oracle_change_justification must be a string or null"
        )
    if "language" in entry or "review_status" in entry:
        raise ExpansionValidationError(
            f"{where} is deterministic-only; language cases are not supported in Stage 2A"
        )
    if "oracle_changes" not in entry or not isinstance(entry["oracle_changes"], list):
        raise ExpansionValidationError(f"{where}.oracle_changes must be a list")
    if entry["oracle_changed"] != bool(entry["oracle_changes"]):
        raise ExpansionValidationError(
            f"{where}.oracle_changed must reflect oracle_changes"
        )


def _apply_replacements(
    data: dict[str, Any],
    changes: list[Any],
    allowed_paths: set[str],
    where: str,
) -> list[str]:
    paths: list[str] = []
    for change in changes:
        if not isinstance(change, dict) or set(change) != {"path", "value"}:
            raise ExpansionValidationError(
                f"{where} changes must contain only path and value"
            )
        path = change["path"]
        if not isinstance(path, str) or path not in allowed_paths:
            raise ExpansionValidationError(f"{where} uses disallowed path {path!r}")
        if path in paths:
            raise ExpansionValidationError(f"{where} changes path {path!r} more than once")
        paths.append(path)
        target, key = _resolve_path(data, path)
        if target[key] == change["value"]:
            raise ExpansionValidationError(
                f"{where} declares a no-op replacement for {path!r}"
            )
        target[key] = deepcopy(change["value"])
    return paths


def _resolve_path(data: dict[str, Any], path: str) -> tuple[dict[str, Any], str]:
    if "." not in path:
        return data, path
    parent_path, key = path.rsplit(".", 1)
    target: Any = data
    for part in parent_path.split("."):
        if not isinstance(target, dict) or part not in target:
            raise ExpansionValidationError(f"replacement path {path!r} does not exist")
        target = target[part]
    if not isinstance(target, dict) or key not in target:
        raise ExpansionValidationError(f"replacement path {path!r} does not exist")
    return target, key


def _validate_final_state_counter_types(data: dict[str, Any], where: str) -> None:
    expected = data.get("expected")
    if not isinstance(expected, dict):
        return
    final_state = expected.get("final_state")
    if not isinstance(final_state, dict):
        return
    for field in ("refund_count", "idempotency_record_count"):
        if field in final_state and type(final_state[field]) is not int:
            raise ExpansionValidationError(
                f"{where}.expected.final_state.{field} must be an integer"
            )


def _validate_proposal_schema(data: dict[str, Any], where: str) -> None:
    proposal_keys = {
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
    proposals = data.get("proposals")
    if not isinstance(proposals, list):
        raise ExpansionValidationError(f"{where}.proposals must be a list")
    for index, proposal in enumerate(proposals):
        if not isinstance(proposal, dict):
            raise ExpansionValidationError(
                f"{where}.proposals[{index}] must be an object"
            )
        operation = proposal.get("operation")
        if not isinstance(operation, str) or operation not in proposal_keys:
            raise ExpansionValidationError(
                f"{where}.proposals[{index}] has an unsupported operation"
            )
        if set(proposal) != proposal_keys[operation]:
            raise ExpansionValidationError(
                f"{where}.proposals[{index}] does not match the "
                f"{operation} proposal schema"
            )


def _validate_child_with_frozen_validator(
    core_scenarios: tuple[Scenario, ...],
    parent_id: str,
    child_data: dict[str, Any],
    where: str,
) -> None:
    shadow_rows = []
    for scenario in core_scenarios:
        row = deepcopy(scenario.data)
        if scenario.scenario_id == parent_id:
            row = deepcopy(child_data)
            row["id"] = parent_id
        shadow_rows.append(row)
    try:
        validate_dataset(
            {
                "schema_version": SCHEMA_VERSION,
                "scenarios": shadow_rows,
            }
        )
    except ScenarioValidationError as error:
        raise ExpansionValidationError(
            f"{where} fails frozen per-scenario semantic validation: {error}"
        ) from error


def _without_id(data: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in data.items() if key != "id"}


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
