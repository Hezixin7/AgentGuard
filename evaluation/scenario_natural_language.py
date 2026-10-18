"""Deterministic Stage 2C natural-language scenario perturbations."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from evaluation.scenario_expansion import ExpandedScenario, expand_catalog
from evaluation.scenarios import (
    SCHEMA_VERSION,
    Scenario,
    ScenarioValidationError,
    validate_dataset,
)

CATALOG_VERSION = "1.0"
GENERATOR_VERSION = "stage-2c-1"
_EXPECTED_CORE_IDS = {f"core-{number:03d}" for number in range(1, 21)}
_EXPECTED_VARIANT_IDS = {
    f"core-{parent:03d}-v{variant}"
    for parent in range(1, 21)
    for variant in range(1, 4)
}
_EXPECTED_PERTURBATION_IDS = {
    f"core-{number:03d}-nl" for number in range(1, 21)
}
_ENTRY_KEYS = {
    "id",
    "parent_scenario_id",
    "category",
    "original_intent",
    "modified_user_input",
    "semantic_preservation_rationale",
    "oracle_inheritance",
    "oracle_change_justification",
    "human_review_status",
}


class NaturalLanguageValidationError(ValueError):
    """Raised when a Stage 2C catalog changes scenario semantics or is incomplete."""


@dataclass(frozen=True)
class NaturalLanguageScenario:
    """A natural-language child scenario paired with its parent and provenance."""

    scenario_id: str
    parent_scenario_id: str
    data: dict[str, Any]
    provenance: dict[str, Any]


@dataclass(frozen=True)
class Stage2CDataset:
    """The verified core, deterministic, and natural-language evaluation cases."""

    core_scenarios: tuple[Scenario, ...]
    deterministic_variants: tuple[ExpandedScenario, ...]
    natural_language_perturbations: tuple[NaturalLanguageScenario, ...]

    @property
    def scenario_count(self) -> int:
        return (
            len(self.core_scenarios)
            + len(self.deterministic_variants)
            + len(self.natural_language_perturbations)
        )

    @property
    def all_scenarios(self) -> tuple[dict[str, Any], ...]:
        return (
            *(scenario.data for scenario in self.core_scenarios),
            *(scenario.data for scenario in self.deterministic_variants),
            *(scenario.data for scenario in self.natural_language_perturbations),
        )


def load_perturbation_catalog(path: str | Path) -> dict[str, Any]:
    """Load a Stage 2C catalog without generating or mutating scenarios."""
    try:
        value = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise NaturalLanguageValidationError(
            f"cannot load Stage 2C perturbation catalog: {error}"
        ) from error
    if not isinstance(value, dict):
        raise NaturalLanguageValidationError(
            "Stage 2C perturbation catalog must be an object"
        )
    return value


def generate_natural_language_perturbations(
    core_scenarios: tuple[Scenario, ...],
    catalog: dict[str, Any],
) -> tuple[NaturalLanguageScenario, ...]:
    """Create one input-only, oracle-inheriting child for every frozen core case."""
    _validate_core(core_scenarios)
    _validate_catalog_header(catalog)
    parents = {scenario.scenario_id: scenario for scenario in core_scenarios}
    entries: dict[str, dict[str, Any]] = {}

    for index, entry in enumerate(catalog["entries"]):
        where = f"entries[{index}]"
        _validate_entry(entry, where)
        parent_id = entry["parent_scenario_id"]
        if parent_id not in parents:
            raise NaturalLanguageValidationError(
                f"{where} references unknown parent {parent_id}"
            )
        if entry["id"] != f"{parent_id}-nl":
            raise NaturalLanguageValidationError(
                f"{where}.id must be {parent_id}-nl"
            )
        if parent_id in entries:
            raise NaturalLanguageValidationError(
                f"duplicate natural-language parent: {parent_id}"
            )
        entries[parent_id] = entry

    if set(entries) != _EXPECTED_CORE_IDS:
        missing = sorted(_EXPECTED_CORE_IDS - set(entries))
        unexpected = sorted(set(entries) - _EXPECTED_CORE_IDS)
        raise NaturalLanguageValidationError(
            f"perturbation parents mismatch; missing={missing}, unexpected={unexpected}"
        )

    normalized_inputs: set[str] = set()
    perturbations: list[NaturalLanguageScenario] = []
    for parent in core_scenarios:
        entry = entries[parent.scenario_id]
        modified_input = entry["modified_user_input"]
        normalized_input = " ".join(modified_input.split()).casefold()
        original_input = " ".join(parent.data["user_input"].split()).casefold()
        if normalized_input == original_input:
            raise NaturalLanguageValidationError(
                f"{entry['id']} does not perturb the parent user input"
            )
        if normalized_input in normalized_inputs:
            raise NaturalLanguageValidationError(
                f"duplicate normalized user input: {entry['id']}"
            )
        normalized_inputs.add(normalized_input)

        child_data = deepcopy(parent.data)
        child_data["id"] = entry["id"]
        child_data["user_input"] = modified_input
        _validate_child(parent, core_scenarios, child_data, entry["id"])
        perturbations.append(
            NaturalLanguageScenario(
                scenario_id=entry["id"],
                parent_scenario_id=parent.scenario_id,
                data=child_data,
                provenance={
                    "catalog_version": catalog["catalog_version"],
                    "generator_version": GENERATOR_VERSION,
                    "category": entry["category"],
                    "original_intent": entry["original_intent"],
                    "semantic_preservation_rationale": entry[
                        "semantic_preservation_rationale"
                    ],
                    "oracle_inheritance": entry["oracle_inheritance"],
                    "oracle_change_justification": entry[
                        "oracle_change_justification"
                    ],
                    "human_review_status": entry["human_review_status"],
                    "evaluation_layer": "deterministic_offline",
                    "random_seed": None,
                },
            )
        )
    return tuple(perturbations)


def build_stage2c_dataset(
    core_scenarios: tuple[Scenario, ...],
    expansion_catalog: dict[str, Any],
    perturbation_catalog: dict[str, Any],
) -> Stage2CDataset:
    """Build and semantically validate the complete offline 20+60+20 dataset."""
    deterministic = expand_catalog(
        core_scenarios,
        expansion_catalog,
        allow_partial=False,
    )
    natural_language = generate_natural_language_perturbations(
        core_scenarios,
        perturbation_catalog,
    )
    dataset = Stage2CDataset(
        core_scenarios=core_scenarios,
        deterministic_variants=deterministic,
        natural_language_perturbations=natural_language,
    )
    if dataset.scenario_count != 100:
        raise NaturalLanguageValidationError(
            "Stage 2C dataset must contain exactly 100 scenarios"
        )

    all_ids = [
        scenario.scenario_id
        for scenario in (
            *dataset.core_scenarios,
            *dataset.deterministic_variants,
            *dataset.natural_language_perturbations,
        )
    ]
    if (
        set(all_ids)
        != _EXPECTED_CORE_IDS
        | _EXPECTED_VARIANT_IDS
        | _EXPECTED_PERTURBATION_IDS
        or len(all_ids) != len(set(all_ids))
    ):
        raise NaturalLanguageValidationError(
            "Stage 2C scenario IDs must comprise the stable 20+60+20 catalog"
        )

    signatures = {
        _canonical({key: value for key, value in scenario.items() if key != "id"})
        for scenario in dataset.all_scenarios
    }
    if len(signatures) != 100:
        raise NaturalLanguageValidationError(
            "Stage 2C scenarios must have unique content apart from their IDs"
        )
    return dataset


def _validate_core(core_scenarios: tuple[Scenario, ...]) -> None:
    if not isinstance(core_scenarios, tuple):
        raise NaturalLanguageValidationError(
            "core scenarios must be a tuple from load_dataset()"
        )
    try:
        validate_dataset(
            {
                "schema_version": SCHEMA_VERSION,
                "scenarios": [deepcopy(scenario.data) for scenario in core_scenarios],
            }
        )
    except ScenarioValidationError as error:
        raise NaturalLanguageValidationError(
            f"frozen core dataset is invalid: {error}"
        ) from error


def _validate_catalog_header(catalog: dict[str, Any]) -> None:
    if set(catalog) != {
        "catalog_version",
        "source_schema_version",
        "complete",
        "lifecycle_status",
        "entries",
    }:
        raise NaturalLanguageValidationError(
            "Stage 2C catalog has missing or unexpected top-level fields"
        )
    if catalog["catalog_version"] != CATALOG_VERSION:
        raise NaturalLanguageValidationError(
            f"catalog_version must be {CATALOG_VERSION!r}"
        )
    if catalog["source_schema_version"] != SCHEMA_VERSION:
        raise NaturalLanguageValidationError(
            f"source_schema_version must be {SCHEMA_VERSION!r}"
        )
    if catalog["complete"] is not True:
        raise NaturalLanguageValidationError(
            "Stage 2C catalog must be marked complete"
        )
    if catalog["lifecycle_status"] != "APPROVED AND FROZEN":
        raise NaturalLanguageValidationError(
            "Stage 2C catalog must be approved and frozen"
        )
    if not isinstance(catalog["entries"], list) or len(catalog["entries"]) != 20:
        raise NaturalLanguageValidationError(
            "Stage 2C catalog must contain exactly 20 perturbations"
        )


def _validate_entry(entry: Any, where: str) -> None:
    if not isinstance(entry, dict) or set(entry) != _ENTRY_KEYS:
        raise NaturalLanguageValidationError(
            f"{where} has missing or unexpected fields"
        )
    for key in (
        "id",
        "parent_scenario_id",
        "category",
        "original_intent",
        "modified_user_input",
        "semantic_preservation_rationale",
    ):
        if not isinstance(entry[key], str) or not entry[key].strip():
            raise NaturalLanguageValidationError(
                f"{where}.{key} must be a non-empty string"
            )
    if entry["oracle_inheritance"] != "inherit_parent":
        raise NaturalLanguageValidationError(
            f"{where} must inherit the parent oracle for an input-only perturbation"
        )
    if entry["oracle_change_justification"] is not None:
        raise NaturalLanguageValidationError(
            f"{where} cannot justify an oracle change without changing scenario semantics"
        )
    if entry["human_review_status"] != "approved":
        raise NaturalLanguageValidationError(
            f"{where}.human_review_status must be 'approved'"
        )


def _validate_child(
    parent: Scenario,
    core_scenarios: tuple[Scenario, ...],
    child_data: dict[str, Any],
    scenario_id: str,
) -> None:
    expected_data = deepcopy(parent.data)
    expected_data["id"] = scenario_id
    expected_data["user_input"] = child_data["user_input"]
    if child_data != expected_data:
        raise NaturalLanguageValidationError(
            f"{scenario_id} may change only its ID and user_input"
        )

    shadow_rows = [
        deepcopy(child_data if scenario.scenario_id == parent.scenario_id else scenario.data)
        for scenario in core_scenarios
    ]
    shadow_rows[next(
        index
        for index, scenario in enumerate(core_scenarios)
        if scenario.scenario_id == parent.scenario_id
    )]["id"] = parent.scenario_id
    try:
        validate_dataset(
            {
                "schema_version": SCHEMA_VERSION,
                "scenarios": shadow_rows,
            }
        )
    except ScenarioValidationError as error:
        raise NaturalLanguageValidationError(
            f"{scenario_id} fails frozen per-scenario semantic validation: {error}"
        ) from error


def _canonical(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
