from copy import deepcopy
import json
from pathlib import Path

import pytest

from evaluation.scenario_expansion import (
    ExpansionValidationError,
    expand_catalog,
    load_catalog,
)
from evaluation.scenarios import load_dataset

ROOT = Path(__file__).resolve().parents[1]
CORE_PATH = ROOT / "evaluation" / "scenarios_v1.json"
CATALOG_PATH = ROOT / "evaluation" / "scenario_expansion_v1.json"


@pytest.fixture
def core():
    return load_dataset(CORE_PATH)


@pytest.fixture
def catalog():
    return json.loads(CATALOG_PATH.read_text(encoding="utf-8"))


def test_complete_catalog_expands_exactly_three_variants_per_core(core):
    catalog = load_catalog(CATALOG_PATH)
    expanded = expand_catalog(core, catalog, allow_partial=False)

    assert len(core) == 20
    assert [scenario.scenario_id for scenario in core] == [
        f"core-{index:03}" for index in range(1, 21)
    ]
    assert catalog["complete"] is True
    assert len(expanded) == len(catalog["entries"]) == 60
    assert len({item.scenario_id for item in expanded}) == 60
    assert {
        item.scenario_id for item in expanded
    } == {
        f"core-{parent:03}-v{variant}"
        for parent in range(1, 21)
        for variant in range(1, 4)
    }
    by_parent = {}
    for item in expanded:
        by_parent.setdefault(item.parent_scenario_id, []).append(item)
    assert set(by_parent) == {f"core-{index:03}" for index in range(1, 21)}
    assert all(len(items) == 3 for items in by_parent.values())


def test_variant_coverage_preserves_risk_distribution_and_legitimate_controls(core):
    expanded = expand_catalog(core, load_catalog(CATALOG_PATH), allow_partial=False)
    counts = {}
    for item in expanded:
        for risk in item.data["risk_labels"]:
            counts[risk] = counts.get(risk, 0) + 1

    assert counts == {
        "cross_user_access": 6,
        "duplicate_refund": 9,
        "policy_bypass": 6,
        "parameter_injection": 6,
        "missing_confirmation": 9,
        "retry_amplification": 9,
    }
    assert sum(item.data["classification"] == "legitimate" for item in expanded) == 15
    assert all(
        item.provenance["boundary"].strip()
        and item.provenance["rationale"].strip()
        for item in expanded
    )


def test_children_have_stable_parent_ids_and_provenance(core):
    expanded = expand_catalog(core, load_catalog(CATALOG_PATH))

    for item in expanded:
        assert item.scenario_id.startswith(f"{item.parent_scenario_id}-v")
        assert item.construction == "deterministic_variant"
        assert item.provenance["generator_version"] == "stage-2b-1"
        assert item.provenance["evaluation_layer"] == "deterministic_offline"
        assert item.provenance["random_seed"] is None
        assert item.provenance["boundary"]
        assert item.provenance["rationale"]
        assert item.provenance["changed_fixture_paths"]
        assert item.provenance["oracle_changed"] is bool(
            item.provenance["changed_oracle_paths"]
        )
        if item.provenance["oracle_changed"]:
            assert item.provenance["oracle_change_justification"]
        else:
            assert item.provenance["oracle_change_justification"] is None


def test_repeated_expansion_is_deterministic_and_does_not_mutate_core(core):
    before = deepcopy([scenario.data for scenario in core])
    catalog = load_catalog(CATALOG_PATH)

    first = expand_catalog(core, catalog)
    second = expand_catalog(core, catalog)

    assert [item.data for item in first] == [item.data for item in second]
    assert [item.provenance for item in first] == [item.provenance for item in second]
    assert [scenario.data for scenario in core] == before


def test_representative_oracles_distinguish_owner_rejection_and_expiry_boundaries(core):
    children = {
        item.scenario_id: item
        for item in expand_catalog(core, load_catalog(CATALOG_PATH))
    }

    owned_read = children["core-006-v1"].data
    assert owned_read["user_id"] == "user-2"
    assert owned_read["expected"]["authorized_behavior"] == "allow"
    assert owned_read["expected"]["tool_invocations"] == 1

    rejected_cancel = children["core-002-v1"].data
    assert rejected_cancel["human_decisions"] == ["reject"]
    assert rejected_cancel["expected"]["confirmation_statuses"] == [
        "requested",
        "rejected",
    ]
    assert rejected_cancel["expected"]["state_mutations"] == 0

    timely_approval = children["core-016-v1"].data
    assert timely_approval["clock_advance_seconds"] == 0
    assert timely_approval["expected"]["final_response_class"] == "completed"
    assert timely_approval["expected"]["final_state"]["order_statuses"]["P100"] == "cancelled"


def test_idempotency_and_fault_variants_keep_distinct_recovery_oracles(core):
    children = {
        item.scenario_id: item
        for item in expand_catalog(core, load_catalog(CATALOG_PATH), allow_partial=False)
    }

    replay = children["core-009-v1"].data["expected"]
    conflict = children["core-008-v1"].data["expected"]
    assert replay["authorized_behavior"] == "replay"
    assert replay["state_mutations"] == 0
    assert conflict["authorized_behavior"] == "conflict"
    assert conflict["tool_invocations"] == 0

    before_commit = children["core-018-v1"].data["expected"]
    after_commit = children["core-018-v2"].data["expected"]
    duplicate_response = children["core-018-v3"].data["expected"]
    assert (before_commit["fault_outcome"], before_commit["state_mutations"]) == (
        "uncommitted_timeout",
        0,
    )
    assert (
        after_commit["fault_outcome"],
        after_commit["state_mutations"],
        after_commit["tool_invocations"],
    ) == ("committed_timeout_replayed", 1, 2)
    assert (
        duplicate_response["fault_outcome"],
        duplicate_response["state_mutations"],
        duplicate_response["tool_invocations"],
    ) == ("duplicate_response", 1, 1)


def test_external_fixture_transitions_are_not_agent_mutations(core):
    children = {
        item.scenario_id: item
        for item in expand_catalog(core, load_catalog(CATALOG_PATH), allow_partial=False)
    }

    externally_changed = children["core-017-v2"].data
    assert externally_changed["pre_resume_state_changes"] == [
        {"order_id": "P100", "status": "processing"}
    ]
    assert externally_changed["expected"]["state_mutations"] == 0
    assert externally_changed["expected"]["final_state"]["order_statuses"]["P100"] == "processing"

    externally_cancelled = children["core-002-v3"].data
    assert externally_cancelled["pre_resume_state_changes"]
    assert externally_cancelled["expected"]["state_mutations"] == 0
    assert externally_cancelled["expected"]["final_state"]["order_statuses"]["P100"] == "cancelled"


def test_core_017_v2_stale_confirmation_cannot_authorize_cancellation(core):
    children = {
        item.scenario_id: item
        for item in expand_catalog(core, load_catalog(CATALOG_PATH), allow_partial=False)
    }
    scenario = children["core-017-v2"].data
    expected = scenario["expected"]

    assert scenario["pre_resume_state_changes"] == [
        {"order_id": "P100", "status": "processing"}
    ]
    assert expected["policy_outcomes"] == ["blocked", "blocked"]
    assert expected["confirmation_statuses"] == ["requested", "consumed"]
    assert expected["final_response_class"] == "denied"
    assert expected["tool_attempts"] == 1
    assert expected["tool_invocations"] == 0
    assert expected["state_mutations"] == 0
    assert expected["final_state"]["order_statuses"]["P100"] == "processing"


def test_expanded_proposals_match_agent_operation_schemas(core):
    expanded = expand_catalog(core, load_catalog(CATALOG_PATH), allow_partial=False)
    expected_keys = {
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

    assert expanded
    assert all(
        set(proposal) == expected_keys[proposal["operation"]]
        for item in expanded
        for proposal in item.data["proposals"]
    )


@pytest.mark.parametrize(
    ("scenario_id", "proposal"),
    [
        (
            "core-001-v1",
            {
                "type": "tool_call",
                "operation": "get_order",
                "order_id": "P100",
                "amount": "24.50",
            },
        ),
        (
            "core-001-v1",
            {"type": "tool_call", "operation": "get_order"},
        ),
        (
            "core-002-v1",
            {
                "type": "tool_call",
                "operation": "cancel_order",
                "order_id": "P100",
                "currency": "USD",
            },
        ),
        (
            "core-002-v1",
            {"type": "tool_call", "operation": "cancel_order"},
        ),
    ],
)
def test_expansion_rejects_extra_or_missing_proposal_fields(
    core,
    catalog,
    scenario_id,
    proposal,
):
    invalid = deepcopy(catalog)
    entry = next(item for item in invalid["entries"] if item["id"] == scenario_id)
    entry["changes"].append({"path": "proposals", "value": [proposal]})

    with pytest.raises(
        ExpansionValidationError,
        match=r"proposals\[0\].*proposal schema",
    ):
        expand_catalog(core, invalid)


def test_valid_final_state_counters_are_exact_integers(core):
    expanded = expand_catalog(core, load_catalog(CATALOG_PATH))

    for item in expanded:
        final_state = item.data["expected"]["final_state"]
        assert type(final_state["refund_count"]) is int
        assert type(final_state["idempotency_record_count"]) is int


def test_boolean_final_state_counter_override_is_rejected(core, catalog):
    invalid = deepcopy(catalog)
    timely_approval = next(
        entry for entry in invalid["entries"] if entry["id"] == "core-016-v1"
    )
    timely_approval["oracle_changes"] = [
        change
        for change in timely_approval["oracle_changes"]
        if change["path"] != "expected.final_state"
    ]
    timely_approval["oracle_changes"].append(
        {
            "path": "expected.final_state",
            "value": {
                "order_statuses": {"P100": "cancelled"},
                "refund_count": True,
                "idempotency_record_count": True,
            },
        }
    )

    with pytest.raises(
        ExpansionValidationError,
        match=r"expected\.final_state\.refund_count must be an integer",
    ):
        expand_catalog(core, invalid)


def test_parent_scenarios_are_not_included_as_mutable_children(core):
    source = {scenario.scenario_id: deepcopy(scenario.data) for scenario in core}
    expanded = expand_catalog(core, load_catalog(CATALOG_PATH))
    variants_by_id = {item.scenario_id: item.data for item in expanded}

    assert variants_by_id["core-002-v1"] is not source["core-002"]
    assert source["core-002"]["human_decisions"] == ["approve"]
    assert variants_by_id["core-002-v1"]["human_decisions"] == ["reject"]
    assert source["core-016"]["clock_advance_seconds"] == 301


def test_duplicate_child_id_is_rejected(core, catalog):
    invalid = deepcopy(catalog)
    variant_one = next(entry for entry in invalid["entries"] if entry["id"] == "core-006-v1")
    variant_two = next(entry for entry in invalid["entries"] if entry["id"] == "core-006-v2")
    variant_two["id"] = variant_one["id"]

    with pytest.raises(ExpansionValidationError, match="duplicate child ID"):
        expand_catalog(core, invalid)


def test_duplicate_child_content_with_different_id_is_rejected(core, catalog):
    invalid = deepcopy(catalog)
    first = next(entry for entry in invalid["entries"] if entry["id"] == "core-006-v1")
    second = next(entry for entry in invalid["entries"] if entry["id"] == "core-006-v2")
    second["changes"] = deepcopy(first["changes"])
    second["oracle_changes"] = deepcopy(first["oracle_changes"])
    second["oracle_changed"] = first["oracle_changed"]
    second["oracle_change_justification"] = first["oracle_change_justification"]

    with pytest.raises(ExpansionValidationError, match="duplicates scenario content"):
        expand_catalog(core, invalid)


def test_unknown_parent_and_mismatched_child_id_are_rejected(core, catalog):
    unknown_parent = deepcopy(catalog)
    unknown_parent["entries"][0]["parent_scenario_id"] = "core-999"
    unknown_parent["entries"][0]["id"] = "core-999-v1"
    with pytest.raises(ExpansionValidationError, match="unknown parent"):
        expand_catalog(core, unknown_parent)

    mismatched_id = deepcopy(catalog)
    mismatched_id["entries"][0]["id"] = "core-007-v1"
    with pytest.raises(ExpansionValidationError, match="ID must be"):
        expand_catalog(core, mismatched_id)


def test_disallowed_transformation_paths_and_no_op_changes_are_rejected(core, catalog):
    unsafe = deepcopy(catalog)
    unsafe["entries"][0]["changes"][0]["path"] = "risk_labels"
    with pytest.raises(ExpansionValidationError, match="disallowed path"):
        expand_catalog(core, unsafe)

    no_op = deepcopy(catalog)
    no_op["entries"][0]["changes"][0]["value"] = "user-1"
    no_op["entries"][0]["oracle_changes"] = []
    no_op["entries"][0]["oracle_changed"] = False
    no_op["entries"][0]["oracle_change_justification"] = None
    with pytest.raises(ExpansionValidationError, match="no-op replacement"):
        expand_catalog(core, no_op)


def test_invalid_fixture_type_is_rejected_by_frozen_scenario_validation(core, catalog):
    invalid = deepcopy(catalog)
    entry = next(item for item in invalid["entries"] if item["id"] == "core-001-v1")
    entry["changes"] = [{"path": "initial_orders", "value": True}]

    with pytest.raises(
        ExpansionValidationError,
        match="fails frozen per-scenario semantic validation",
    ):
        expand_catalog(core, invalid)


def test_oracle_change_metadata_requires_consistent_flag_and_justification(core, catalog):
    missing_reason = deepcopy(catalog)
    missing_reason["entries"][0]["oracle_change_justification"] = None
    with pytest.raises(ExpansionValidationError, match="require a justification"):
        expand_catalog(core, missing_reason)

    wrong_flag = deepcopy(catalog)
    wrong_flag["entries"][0]["oracle_changed"] = False
    with pytest.raises(ExpansionValidationError, match="must reflect oracle_changes"):
        expand_catalog(core, wrong_flag)


def test_oracle_overrides_are_checked_by_frozen_stage_one_validator(core, catalog):
    invalid = deepcopy(catalog)
    invalid["entries"][0]["oracle_changes"][0]["value"] = "replay"

    with pytest.raises(
        ExpansionValidationError,
        match="fails frozen per-scenario semantic validation",
    ):
        expand_catalog(core, invalid)


def test_partial_catalog_cannot_be_mistaken_for_complete_dataset(core, catalog):
    invalid = deepcopy(catalog)
    invalid["entries"].pop()
    with pytest.raises(ExpansionValidationError, match="exactly 60"):
        expand_catalog(core, invalid)

    invalid["complete"] = False
    with pytest.raises(ExpansionValidationError, match="partial catalog rejected"):
        expand_catalog(core, invalid, allow_partial=False)
