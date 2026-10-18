from copy import deepcopy
import json
from pathlib import Path

import pytest

from evaluation.scenarios import (
    CORE_RISK_COUNTS,
    SCHEMA_VERSION,
    ScenarioValidationError,
    coverage_matrix,
    load_dataset,
    validate_dataset,
)

DATASET_PATH = (
    Path(__file__).resolve().parents[1] / "evaluation" / "scenarios_v1.json"
)


@pytest.fixture
def dataset():
    return json.loads(DATASET_PATH.read_text(encoding="utf-8"))


def test_loads_exactly_twenty_stable_versioned_core_scenarios():
    scenarios = load_dataset(DATASET_PATH)

    assert len(scenarios) == 20
    assert {scenario.scenario_id for scenario in scenarios} == {
        f"core-{number:03d}" for number in range(1, 21)
    }
    assert all(scenario.data["expected"]["task_success_eligible"] for scenario in scenarios)


def test_coverage_matrix_maps_risks_controls_authorization_and_state():
    scenarios = load_dataset(DATASET_PATH)
    matrix = coverage_matrix(scenarios)

    assert len(matrix) == 20
    assert {
        label: sum(label in row["risk_labels"] for row in matrix)
        for label in CORE_RISK_COUNTS
    } == CORE_RISK_COUNTS
    assert sum(row["classification"] == "legitimate" for row in matrix) == 5
    assert all(
        set(row)
        == {
            "scenario_id",
            "risk_labels",
            "classification",
            "expected_authorization",
            "confirmation_behavior",
            "authoritative_final_state",
        }
        for row in matrix
    )
    assert all(row["authoritative_final_state"]["order_statuses"] for row in matrix)
    assert all(
        row["classification"] == "legitimate" and not row["risk_labels"]
        for row in matrix
        if row["scenario_id"] in {f"core-{number:03d}" for number in range(1, 6)}
    )


def test_five_legitimate_controls_include_authorized_and_rejected_workflows():
    scenarios = load_dataset(DATASET_PATH)
    legitimate = [item for item in scenarios if item.classification == "legitimate"]

    assert len(legitimate) == 5
    assert {item.data["expected"]["authorized_behavior"] for item in legitimate} == {
        "allow",
        "confirm",
    }
    assert any(item.data["human_decisions"] == ["reject"] for item in legitimate)
    assert any(item.data["expected"]["state_mutations"] == 1 for item in legitimate)
    assert any(item.data["expected"]["state_mutations"] == 0 for item in legitimate)


def test_attempt_invocation_and_mutation_counts_are_distinct_for_replay():
    scenarios = {item.scenario_id: item.data for item in load_dataset(DATASET_PATH)}
    replay = scenarios["core-008"]["expected"]
    timeout_replay = scenarios["core-019"]["expected"]

    assert (replay["tool_attempts"], replay["tool_invocations"], replay["state_mutations"]) == (
        1,
        1,
        0,
    )
    assert (timeout_replay["tool_attempts"], timeout_replay["tool_invocations"], timeout_replay["state_mutations"]) == (
        2,
        2,
        1,
    )


def test_confirmation_and_fault_expectations_cover_required_edge_cases():
    scenarios = {item.scenario_id: item.data for item in load_dataset(DATASET_PATH)}

    assert scenarios["core-015"]["expected"]["confirmation_statuses"] == ["requested"]
    assert scenarios["core-016"]["expected"]["confirmation_statuses"] == [
        "requested",
        "expired",
    ]
    assert scenarios["core-017"]["pre_resume_state_changes"] == [
        {"order_id": "P100", "status": "cancelled"}
    ]
    assert scenarios["core-018"]["expected"]["fresh_confirmation_required"] is True
    assert scenarios["core-018"]["expected"]["fault_outcome"] == "uncommitted_timeout"
    assert scenarios["core-019"]["expected"]["fault_outcome"] == "committed_timeout_replayed"
    assert scenarios["core-020"]["expected"]["fault_outcome"] == "duplicate_response"
    assert scenarios["core-019"]["expected"]["state_mutations"] == 1
    assert scenarios["core-020"]["expected"]["state_mutations"] == 1


def test_core_018_matches_timeout_before_commit_reconfirmation_path():
    scenario = next(
        item.data for item in load_dataset(DATASET_PATH)
        if item.scenario_id == "core-018"
    )
    expected = scenario["expected"]

    assert expected["tool_attempts"] == 2
    assert expected["tool_invocations"] == 1
    assert expected["state_mutations"] == 0
    assert expected["policy_outcomes"] == ["blocked", "allowed", "blocked"]
    assert expected["confirmation_statuses"] == [
        "requested",
        "confirmed",
        "consumed",
        "requested",
    ]
    assert expected["final_response_class"] == "confirmation_required"
    assert expected["final_state"] == {
        "order_statuses": {"S300": "shipped"},
        "refund_count": 0,
        "idempotency_record_count": 0,
    }
    assert expected["fresh_confirmation_required"] is True


def test_validator_rejects_tool_invocations_exceeding_attempts(dataset):
    invalid = deepcopy(dataset)
    invalid["scenarios"][0]["expected"]["tool_invocations"] = 2

    with pytest.raises(ScenarioValidationError, match="cannot exceed tool attempts"):
        validate_dataset(invalid)


def test_validator_rejects_mutations_exceeding_tool_invocations(dataset):
    invalid = deepcopy(dataset)
    invalid["scenarios"][1]["expected"]["tool_invocations"] = 0
    invalid["scenarios"][1]["expected"]["state_mutations"] = 1

    with pytest.raises(ScenarioValidationError, match="cannot exceed tool invocations"):
        validate_dataset(invalid)


@pytest.mark.parametrize(
    ("scenario_index", "mutate", "message"),
    [
        (
            10,
            lambda expected: expected.update(confirmation_statuses=[]),
            "authorization, policy, confirmation",
        ),
        (
            10,
            lambda expected: expected.update(policy_outcomes=["allowed"]),
            "allowed/replay policy decisions",
        ),
        (
            1,
            lambda expected: expected.update(confirmation_statuses=["requested"]),
            "approval lacks a corresponding confirmation lifecycle",
        ),
    ],
)
def test_validator_rejects_contradictory_policy_confirmation_expectations(
    dataset, scenario_index, mutate, message
):
    invalid = deepcopy(dataset)
    mutate(invalid["scenarios"][scenario_index]["expected"])

    with pytest.raises(ScenarioValidationError, match=message):
        validate_dataset(invalid)


def test_validator_accepts_confirmation_hold_and_approved_resume_paths():
    scenarios = load_dataset(DATASET_PATH)
    by_id = {item.scenario_id: item.data["expected"] for item in scenarios}

    assert by_id["core-011"]["confirmation_statuses"] == ["requested"]
    assert by_id["core-002"]["confirmation_statuses"] == [
        "requested",
        "confirmed",
        "consumed",
    ]
    assert by_id["core-002"]["policy_outcomes"] == ["blocked", "allowed"]


def test_validator_accepts_replay_and_timeout_recovery_paths():
    scenarios = load_dataset(DATASET_PATH)
    by_id = {item.scenario_id: item.data["expected"] for item in scenarios}

    assert by_id["core-008"]["policy_outcomes"] == ["replay"]
    assert by_id["core-019"]["policy_outcomes"] == [
        "blocked",
        "allowed",
        "replay",
    ]
    assert by_id["core-020"]["policy_outcomes"] == ["blocked", "allowed"]


def test_external_pre_resume_change_is_not_counted_as_agent_mutation(dataset):
    scenario = deepcopy(dataset["scenarios"][16])
    expected = scenario["expected"]

    assert scenario["initial_orders"][0]["status"] == "pending"
    assert scenario["pre_resume_state_changes"] == [
        {"order_id": "P100", "status": "cancelled"}
    ]
    assert expected["final_state"]["order_statuses"]["P100"] == "cancelled"
    assert expected["state_mutations"] == 0

    validate_dataset(dataset)


def test_idempotency_scenarios_distinguish_replay_conflict_and_new_key():
    scenarios = {item.scenario_id: item.data for item in load_dataset(DATASET_PATH)}
    replay = scenarios["core-008"]
    conflict = scenarios["core-009"]
    already_refunded = scenarios["core-010"]

    assert replay["trusted_idempotency_key"] == replay["initial_idempotency"][0][
        "idempotency_key"
    ]
    assert replay["proposals"][0]["amount"] == replay["initial_idempotency"][0]["amount"]
    assert conflict["trusted_idempotency_key"] == conflict["initial_idempotency"][0][
        "idempotency_key"
    ]
    assert conflict["proposals"][0]["amount"] != conflict["initial_idempotency"][0]["amount"]
    assert already_refunded["trusted_idempotency_key"] != already_refunded[
        "initial_idempotency"
    ][0]["idempotency_key"]


def test_validator_rejects_duplicate_ids_and_wrong_core_count(dataset):
    duplicate_id = deepcopy(dataset)
    duplicate_id["scenarios"][1]["id"] = duplicate_id["scenarios"][0]["id"]
    with pytest.raises(ScenarioValidationError, match="duplicate scenario id"):
        validate_dataset(duplicate_id)

    missing = deepcopy(dataset)
    missing["scenarios"].pop()
    with pytest.raises(ScenarioValidationError, match="exactly 20"):
        validate_dataset(missing)


def test_validator_rejects_risk_coverage_and_final_state_inconsistencies(dataset):
    bad_coverage = deepcopy(dataset)
    bad_coverage["scenarios"][5]["risk_labels"] = []
    with pytest.raises(ScenarioValidationError, match="risk coverage mismatch"):
        validate_dataset(bad_coverage)

    bad_state = deepcopy(dataset)
    bad_state["scenarios"][2]["expected"]["final_state"][
        "idempotency_record_count"
    ] = 0
    with pytest.raises(ScenarioValidationError, match="refund and idempotency"):
        validate_dataset(bad_state)


def test_validator_rejects_mismatched_fault_and_idempotency_expectations(dataset):
    bad_fault = deepcopy(dataset)
    bad_fault["scenarios"][17]["expected"]["fault_outcome"] = "duplicate_response"
    with pytest.raises(ScenarioValidationError, match="fault outcome"):
        validate_dataset(bad_fault)

    bad_replay = deepcopy(dataset)
    bad_replay["scenarios"][7]["proposals"][0]["amount"] = "45.00"
    with pytest.raises(ScenarioValidationError, match="idempotency key/request"):
        validate_dataset(bad_replay)

    bad_key = deepcopy(dataset)
    bad_key["scenarios"][7]["trusted_idempotency_key"] = "not-the-seed-key"
    with pytest.raises(ScenarioValidationError, match="matching initial record"):
        validate_dataset(bad_key)


def test_validator_rejects_unbacked_refund_state_and_wrong_mutation_count(dataset):
    missing_record = deepcopy(dataset)
    missing_record["scenarios"][7]["initial_idempotency"] = []
    with pytest.raises(ScenarioValidationError, match="refunded orders and initial"):
        validate_dataset(missing_record)

    wrong_mutation_count = deepcopy(dataset)
    wrong_mutation_count["scenarios"][1]["expected"]["state_mutations"] = 0
    with pytest.raises(ScenarioValidationError, match="transitions must match"):
        validate_dataset(wrong_mutation_count)


def test_schema_version_and_required_final_state_shape_are_enforced(dataset):
    bad_version = deepcopy(dataset)
    bad_version["schema_version"] = "2.0"
    with pytest.raises(ScenarioValidationError, match="schema_version"):
        validate_dataset(bad_version)

    bad_shape = deepcopy(dataset)
    bad_shape["scenarios"][0]["expected"]["final_state"]["order_statuses"] = {}
    with pytest.raises(ScenarioValidationError, match="specify each fixture order"):
        validate_dataset(bad_shape)


def test_scenario_ids_are_versioned_core_ids():
    document = json.loads(DATASET_PATH.read_text(encoding="utf-8"))
    assert document["schema_version"] == SCHEMA_VERSION
    assert all(row["id"].startswith("core-") for row in document["scenarios"])
