import ast
from pathlib import Path

import pytest

from evaluation.gold_labels import (
    GROUPS,
    build_gold_label_catalog,
    load_gold_label_catalog,
    validate_gold_label_catalog,
)
from evaluation.offline_batch import compose_stage2c_dataset, run_batch

GOLD_MODULE = Path("evaluation/gold_labels.py")
GOLD_CATALOG = Path("evaluation/gold_labels_v1.json")

ELIGIBLE_BASELINE_A = {
    "core-001-v1",
    "core-001-v2",
    "core-003-v2",
    "core-005-v1",
    "core-005-v2",
    "core-006",
    "core-006-nl",
    "core-006-v3",
    "core-007",
    "core-007-nl",
    "core-007-v2",
    "core-007-v3",
    "core-008-v2",
    "core-008-v3",
    "core-009-v3",
    "core-010",
    "core-010-nl",
    "core-010-v2",
    "core-011-v2",
    "core-011-v3",
    "core-012-v2",
    "core-013",
    "core-013-nl",
    "core-013-v3",
    "core-014",
    "core-014-nl",
    "core-014-v1",
    "core-014-v3",
}

SUCCESS_BASELINE_A = {
    "core-001-v1",
    "core-005-v1",
    "core-006",
    "core-006-nl",
    "core-007",
    "core-007-nl",
    "core-011-v2",
    "core-013",
    "core-013-nl",
    "core-014-v1",
}

MUTATING_SUCCESS_BASELINE_A = {
    "core-007",
    "core-007-nl",
    "core-011-v2",
    "core-013",
    "core-013-nl",
}

NON_MUTATING_SUCCESS_BASELINE_A = {
    "core-001-v1",
    "core-005-v1",
    "core-006",
    "core-006-nl",
    "core-014-v1",
}

PARENT_COUNTS = {
    "core-001": 2,
    "core-003": 1,
    "core-005": 2,
    "core-006": 3,
    "core-007": 4,
    "core-008": 2,
    "core-009": 1,
    "core-010": 3,
    "core-011": 2,
    "core-012": 1,
    "core-013": 3,
    "core-014": 4,
}


@pytest.fixture(scope="module")
def inputs():
    dataset = compose_stage2c_dataset()
    batch = run_batch()
    catalog = build_gold_label_catalog(dataset, batch)
    static = load_gold_label_catalog(GOLD_CATALOG)
    return dataset, batch, catalog, static


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_catalog_is_schema_valid_and_covers_every_scenario_group(inputs):
    dataset, _, catalog, _ = inputs
    assert validate_gold_label_catalog(catalog, dataset) == []
    expected = {
        (scenario["id"], group)
        for scenario in dataset.all_scenarios
        for group in GROUPS
    }
    actual = {
        (label["scenario_id"], label["group"])
        for label in catalog["labels"]
    }
    assert actual == expected
    assert len(catalog["labels"]) == 300
    assert len({label["label_id"] for label in catalog["labels"]}) == 300


def test_static_core_catalog_matches_independent_builder(inputs):
    _, _, catalog, static = inputs
    built_core = {
        label["label_id"]: label
        for label in catalog["labels"]
        if not label["is_replication"]
    }
    static_core = {
        label["label_id"]: label for label in static["core_labels"]
    }
    assert len(built_core) == 60
    assert set(built_core) == set(static_core)
    for label_id, label in built_core.items():
        assert label == static_core[label_id]


def test_frozen_labels_never_claim_proven_earliest_from_oracles(inputs):
    dataset, _, catalog, _ = inputs
    proven = [
        label
        for label in catalog["labels"]
        if label["causal_label"] == "proven_earliest"
    ]
    assert proven == []
    for label in catalog["labels"]:
        finding = label.get("expected_finding")
        if finding is not None:
            assert finding["step"] is None
            assert label["label_provenance"]["step_derived_from"] is None
        assert (
            label["label_provenance"]["localizer_used_for_label"] is False
        )
    for scenario in dataset.all_scenarios:
        assert "step" not in scenario["expected"]


def test_derived_labels_are_replications_with_parent_clusters(inputs):
    dataset, _, catalog, _ = inputs
    by_parent_group: dict[tuple[str, str], int] = {}
    for label in catalog["labels"]:
        key = (label["parent_scenario_id"], label["group"])
        by_parent_group[key] = by_parent_group.get(key, 0) + 1
    assert set(by_parent_group.values()) == {5}
    assert len(by_parent_group) == 20 * len(GROUPS)

    origins = {label["label_origin"] for label in catalog["labels"]}
    assert "inherited_natural_language_replication" in origins
    assert "inherited_oracle_unchanged_replication" in origins
    assert "derived_from_child_oracle" in origins

    unchanged = [
        scenario.scenario_id
        for scenario in dataset.deterministic_variants
        if not scenario.provenance.get("oracle_changed")
    ]
    assert unchanged
    unchanged_id = unchanged[0]
    label = next(
        label
        for label in catalog["labels"]
        if label["scenario_id"] == unchanged_id
        and label["group"] == "agentguard"
    )
    assert label["label_origin"] == (
        "inherited_oracle_unchanged_replication"
    )


def test_baseline_a_eligibility_audit_matches_frozen_evidence(inputs):
    _, _, catalog, static = inputs
    audit = catalog["baseline_a_eligibility"]
    assert audit == static["baseline_a_eligibility"]
    assert audit["eligible_count"] == 28
    assert audit["success_count"] == 10
    assert audit["success_committed_true"] == 5
    assert audit["success_committed_false"] == 5
    assert audit["success_mutation_observed_true"] == 5
    assert audit["success_mutation_observed_false"] == 5
    assert audit["error_count"] == 18
    assert audit["parent_counts"] == PARENT_COUNTS

    records = {row["scenario_id"]: row for row in audit["records"]}
    assert set(records) == ELIGIBLE_BASELINE_A
    assert {
        scenario_id
        for scenario_id, row in records.items()
        if row["outcome"] in {"success", "replayed"}
    } == SUCCESS_BASELINE_A
    assert {
        scenario_id
        for scenario_id, row in records.items()
        if row["outcome"] in {"success", "replayed"}
        and row["mutation_observed"]
    } == MUTATING_SUCCESS_BASELINE_A
    assert {
        scenario_id
        for scenario_id, row in records.items()
        if row["outcome"] in {"success", "replayed"}
        and not row["mutation_observed"]
    } == NON_MUTATING_SUCCESS_BASELINE_A
    for row in records.values():
        assert row["mutation_observed"] == (
            row["committed"] and row["agent_state_mutations"] > 0
        )
        assert row["agent_state_mutations"] == (
            1 if row["mutation_observed"] else 0
        )


def test_success_outcome_alone_does_not_imply_business_mutation(inputs):
    _, _, catalog, _ = inputs
    labels = {
        label["scenario_id"]: label
        for label in catalog["labels"]
        if label["group"] == "baseline_a"
        and label["observable_label"] == "divergence"
    }
    for scenario_id in NON_MUTATING_SUCCESS_BASELINE_A:
        finding = labels[scenario_id]["expected_finding"]
        assert finding["subtype"] == "non_mutating_prohibited_execution"
        assert finding["expected_evidence"]["mutation_observed"] is False
        assert finding["expected_evidence"]["agent_state_mutations"] == 0
    for scenario_id in MUTATING_SUCCESS_BASELINE_A:
        finding = labels[scenario_id]["expected_finding"]
        assert finding["subtype"] == "mutating_prohibited_execution"
        assert finding["expected_evidence"]["mutation_observed"] is True
        assert finding["expected_evidence"]["agent_state_mutations"] == 1


def test_all_noneligible_baseline_a_labels_are_not_scored(inputs):
    _, _, catalog, _ = inputs
    baseline_a = [
        label
        for label in catalog["labels"]
        if label["group"] == "baseline_a"
    ]
    not_scored = [
        label
        for label in baseline_a
        if label["observable_label"] == "not_scored"
    ]
    assert len(baseline_a) == 100
    assert len(not_scored) == 72
    assert {
        label["scenario_id"] for label in not_scored
    } == {
        label["scenario_id"]
        for label in baseline_a
        if label["scenario_id"] not in ELIGIBLE_BASELINE_A
    }
    for label in not_scored:
        assert label["causal_label"] == "not_applicable"
        assert label["exclusion_reason"]
        assert label["diagnosis_required"] is False


def test_group_scopes_reflect_actual_instrumentation(inputs):
    _, _, catalog, _ = inputs
    scopes = catalog["group_scopes"]
    assert "trace_event_order" in scopes["agentguard"][
        "verified_predicates"
    ]
    assert "trace_event_order" in scopes["baseline_b"][
        "unverified_predicates"
    ]
    assert "policy_outcomes" in scopes["baseline_a"][
        "unverified_predicates"
    ]
    assert "confirmation_statuses" in scopes["baseline_a"][
        "unverified_predicates"
    ]
    for group in GROUPS:
        assert scopes[group]["source_scope"]
        assert scopes[group]["verified_predicates"]


def test_gold_label_module_has_no_localization_import():
    names = _imports(GOLD_MODULE)
    assert not any("localization" in name for name in names)
    assert "localize_run" not in GOLD_MODULE.read_text(encoding="utf-8")


def test_validator_rejects_prediction_derived_or_inconsistent_labels(inputs):
    _, _, catalog, _ = inputs
    invalid = {
        "schema_version": catalog["schema_version"],
        "labels": [
            {
                **catalog["labels"][0],
                "label_provenance": {
                    "localizer_used_for_label": True,
                },
            }
        ],
    }
    errors = validate_gold_label_catalog(invalid)
    assert any("localizer was not used" in error for error in errors)

    proven = {
        "schema_version": catalog["schema_version"],
        "labels": [
            {
                **catalog["labels"][0],
                "observable_label": "divergence",
                "causal_label": "proven_earliest",
                "expected_finding": {
                    "source": "private_runner_observations",
                    "step": None,
                },
            }
        ],
    }
    errors = validate_gold_label_catalog(proven)
    assert any("requires trace source" in error for error in errors)
    assert any("requires an integer step" in error for error in errors)
