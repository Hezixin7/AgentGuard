import ast
from pathlib import Path

import pytest

from evaluation.corruptions import (
    apply_corruption,
    load_corruption_manifest,
    validate_corruption_manifest,
)
from evaluation.offline_batch import compose_stage2c_dataset, run_batch

CORRUPTION_MODULE = Path("evaluation/corruptions.py")
CORRUPTION_MANIFEST = Path(
    "evaluation/localization_corruptions_v1.json"
)
PROVEN_IDS = {"P1", "P2", "P3", "P4", "P5"}
UNPROVEN_IDS = {"U1", "U2", "U3", "U4", "U5"}
UNOBSERVABLE_IDS = {"X1", "X2", "X3", "X4", "X5"}


@pytest.fixture(scope="module")
def inputs():
    dataset = compose_stage2c_dataset()
    batch = run_batch()
    manifest = load_corruption_manifest(CORRUPTION_MANIFEST)
    report = validate_corruption_manifest(manifest, dataset, batch)
    return dataset, batch, manifest, report


def _case(manifest, case_id):
    return next(
        case for case in manifest["cases"] if case["case_id"] == case_id
    )


def _record(batch, scenario_id, group):
    return next(
        record
        for record in batch.records
        if record.scenario_id == scenario_id and record.group == group
    )


def _scenario(dataset, scenario_id):
    return next(
        scenario
        for scenario in dataset.all_scenarios
        if scenario["id"] == scenario_id
    )


def _imports(path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    names = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.append(node.module or "")
    return names


def test_manifest_has_fifteen_cases_in_three_explicit_categories(inputs):
    _, _, manifest, report = inputs
    assert report["valid"] is True
    assert report["errors"] == []
    assert report["case_count"] == 15
    assert report["category_counts"] == {
        "proven_earliest": 5,
        "observed_unproven": 5,
        "unobservable": 5,
    }
    ids = {case["case_id"] for case in manifest["cases"]}
    assert ids == PROVEN_IDS | UNPROVEN_IDS | UNOBSERVABLE_IDS


def test_every_mutation_targets_an_existing_frozen_field(inputs):
    dataset, batch, manifest, report = inputs
    for case in manifest["cases"]:
        record = _record(
            batch,
            case["base_scenario_id"],
            case["group"],
        )
        scenario = _scenario(dataset, case["base_scenario_id"])
        _, mutated = apply_corruption(case, scenario, record)
        assert mutated != record, case["case_id"]
    assert all(
        case["causal_defensible"] for case in report["cases"]
    )


def test_proven_cases_have_one_earliest_changed_event(inputs):
    _, _, manifest, report = inputs
    summaries = {case["case_id"]: case for case in report["cases"]}
    for case_id in PROVEN_IDS:
        case = _case(manifest, case_id)
        summary = summaries[case_id]
        assert case["expected"]["causal_label"] == "proven_earliest"
        assert case["expected"]["step_scope"] == "causal_earliest"
        assert case["expected"]["source"] == "agentguard_trace"
        assert summary["changed_event_indices"] == [
            case["mutation"]["event_index"]
        ]
        assert summary["event_step"] == case["expected"]["step"]
        for check, value in summary["causal_checks"].items():
            assert value is True, (case_id, check)


def test_p3_multi_field_mutation_is_unambiguous(inputs):
    _, _, manifest, report = inputs
    case = _case(manifest, "P3")
    summary = next(
        item for item in report["cases"] if item["case_id"] == "P3"
    )
    assert case["mutation"]["event_index"] == 10
    assert case["mutation"]["declared_event_step"] == 11
    assert summary["changed_event_indices"] == [10]
    assert summary["changed_fields"] == [
        "error_type",
        "replayed",
        "status",
    ]
    assert summary["event_step"] == 11
    assert summary["mutated_event_step"] == 11
    assert summary["causal_checks"]["multi_field_single_event"] is True
    assert summary["causal_checks"][
        "earliest_changed_step_matches_expected"
    ] is True
    assert case["expected"]["step"] == 11
    assert case["expected"]["predicate_id"] == "fault_outcome[1]"


def test_unproven_cases_never_claim_earliest_divergence(inputs):
    _, _, manifest, report = inputs
    for case_id in UNPROVEN_IDS:
        case = _case(manifest, case_id)
        expected = case["expected"]
        assert case["category"] == "observed_unproven"
        assert expected["causal_label"] == "observed_violation"
        assert expected["step_scope"] != "causal_earliest"
        assert expected["status"] == "localized"
        assert case["provenance"]["localizer_used_for_label"] is False

    summaries = {case["case_id"]: case for case in report["cases"]}
    assert summaries["U1"]["contradiction_indices"] == [0]
    assert summaries["U2"]["contradiction_indices"] == [2]
    assert summaries["U3"]["contradiction_indices"] == [0]
    assert summaries["U4"]["contradiction_observed"] is True
    assert summaries["U5"]["contradiction_observed"] is True


def test_cross_source_cases_leave_trace_unchanged(inputs):
    dataset, batch, manifest, _ = inputs
    for case_id in {"U1", "U2", "U3"}:
        case = _case(manifest, case_id)
        record = _record(
            batch,
            case["base_scenario_id"],
            case["group"],
        )
        scenario = _scenario(dataset, case["base_scenario_id"])
        _, mutated = apply_corruption(case, scenario, record)
        assert mutated.observations.trace_capture == (
            record.observations.trace_capture
        )
        assert mutated.observations != record.observations


def test_baseline_unproven_cases_have_no_trace_chronology(inputs):
    _, batch, manifest, _ = inputs
    for case_id in {"U4", "U5"}:
        case = _case(manifest, case_id)
        record = _record(
            batch,
            case["base_scenario_id"],
            case["group"],
        )
        assert record.observations.trace_capture is None
        assert case["expected"]["source"] == "private_runner_observations"
        assert case["expected"]["step"] is None


def test_unobservable_cases_establish_their_evidence_gap(inputs):
    _, _, manifest, report = inputs
    summaries = {case["case_id"]: case for case in report["cases"]}
    for case_id in UNOBSERVABLE_IDS:
        case = _case(manifest, case_id)
        expected = case["expected"]
        summary = summaries[case_id]
        assert case["category"] == "unobservable"
        assert expected["status"] == "unobservable"
        assert expected["causal_label"] == "not_establishable"
        assert expected["reason_family"]
        predicate = summary["unobservable_predicate"]
        assert predicate["valid"] is True
        assert predicate["predicate"] == case["validation"]["predicate"]


def test_x2_drop_leaves_unknown_call_references(inputs):
    dataset, batch, manifest, _ = inputs
    case = _case(manifest, "X2")
    record = _record(
        batch,
        case["base_scenario_id"],
        case["group"],
    )
    scenario = _scenario(dataset, case["base_scenario_id"])
    _, mutated = apply_corruption(case, scenario, record)
    before = record.observations.trace_capture.events
    after = mutated.observations.trace_capture.events
    assert len(after) == len(before) - 1
    assert [event["step"] for event in after] == list(
        range(1, len(after) + 1)
    )
    call_refs = {
        event["tool_call_ref"]
        for event in after
        if event["event_type"] == "tool_call"
    }
    assert call_refs == set()
    assert any(
        event.get("tool_call_ref")
        for event in after
        if event["event_type"]
        in {"policy_decision", "human_confirmation", "tool_result"}
    )


def test_x5_removes_trace_capture(inputs):
    dataset, batch, manifest, _ = inputs
    case = _case(manifest, "X5")
    record = _record(
        batch,
        case["base_scenario_id"],
        case["group"],
    )
    scenario = _scenario(dataset, case["base_scenario_id"])
    _, mutated = apply_corruption(case, scenario, record)
    assert record.observations.trace_capture is not None
    assert mutated.observations.trace_capture is None
    assert mutated.observations.trace_event_count == 0


def test_manifest_provenance_is_prediction_independent(inputs):
    _, _, manifest, _ = inputs
    for case in manifest["cases"]:
        provenance = case["provenance"]
        assert provenance["label_origin"] == "corruption_manifest"
        assert provenance["localizer_used_for_label"] is False
        assert provenance["review_status"]
        assert provenance["rationale"]
        assert "localize_run" not in provenance["rationale"]


def test_p3_owner_approved_scoring_convention_is_recorded(inputs):
    _, _, manifest, _ = inputs
    scope = manifest["approval_scope"]
    assert scope["catalog_wide_human_signoff"] is False
    assert scope["independent_human_review"] == "pending_for_all_labels"
    assert "P3_canonical_fault_outcome_1" in (
        scope["project_owner_approved_conventions"]
    )

    convention = next(
        item
        for item in manifest["scoring_conventions"]
        if item["convention_id"] == "P3_canonical_fault_outcome_1"
    )
    assert convention["case_id"] == "P3"
    assert convention["approval"]["approved_by"] == "project_owner"
    assert convention["approval"]["status"] == "approved"
    assert convention["gold_changed_to_match_predictions"] is False
    assert convention["supporting_evidence_only"] == [
        "trace_runner_tool_result_conflict"
    ]
    assert "no strict earliest-divergence accuracy credit" in (
        convention["strict_earliest_divergence_rule"]
    )

    case = _case(manifest, "P3")
    expected = case["expected"]
    assert expected["predicate_id"] == "fault_outcome[1]"
    assert expected["source"] == "agentguard_trace"
    assert expected["step"] == 11
    assert expected["causal_label"] == "proven_earliest"
    assert convention["canonical_gold_finding"] == {
        "predicate_id": expected["predicate_id"],
        "predicate_family": expected["predicate_family"],
        "source": expected["source"],
        "step": expected["step"],
        "causal_label": expected["causal_label"],
    }
    assert case["provenance"]["approved_scoring_convention"] == (
        convention["convention_id"]
    )


def test_pending_review_status_is_preserved_for_all_labels(inputs):
    _, _, manifest, _ = inputs
    for case in manifest["cases"]:
        assert case["provenance"]["review_status"] == (
            "structurally_validated_pending_independent_human_signoff"
        )
        if case["case_id"] != "P3":
            assert "approved_scoring_convention" not in case["provenance"]
    assert manifest["review_status"] == (
        "structurally_validated_pending_independent_human_signoff"
    )


def test_corruption_module_and_manifest_do_not_use_localization():
    names = _imports(CORRUPTION_MODULE)
    assert not any("localization" in name for name in names)
    text = CORRUPTION_MODULE.read_text(encoding="utf-8")
    assert "localize_run" not in text
    manifest_text = CORRUPTION_MANIFEST.read_text(encoding="utf-8")
    assert "localize_run" not in manifest_text
    assert "evaluation.localization" not in manifest_text
