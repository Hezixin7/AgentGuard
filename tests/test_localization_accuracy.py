import copy
import hashlib
import json
from pathlib import Path

import pytest

from evaluation.corruptions import load_corruption_manifest
from evaluation.gold_labels import build_gold_label_catalog
from evaluation.localization_accuracy import (
    REPORT_PATH,
    _frozen_clean_group,
    abstention_match,
    build_accuracy_report,
    metric,
    observed_violation_match,
    render_accuracy_report,
    status_matches,
    strict_earliest_match,
    usable_finding,
)
from evaluation.offline_batch import compose_stage2c_dataset, run_batch

GOLD_PATH = Path("evaluation/gold_labels_v1.json")
MANIFEST_PATH = Path("evaluation/localization_corruptions_v1.json")


@pytest.fixture(scope="module")
def artifacts():
    dataset = compose_stage2c_dataset()
    batch = run_batch()
    manifest = load_corruption_manifest(MANIFEST_PATH)
    report = build_accuracy_report(
        dataset=dataset,
        batch=batch,
    )
    return dataset, batch, manifest, report


def _prediction(
    status="localized",
    predicate_id=None,
    source="agentguard_trace",
    step=None,
    causal="proven",
    reasons=(),
):
    finding = None
    if predicate_id is not None:
        source_ref = {"source": source}
        if step is not None:
            source_ref["step"] = step
        finding = {
            "predicate_id": predicate_id,
            "source_ref": source_ref,
            "earliest_causal_divergence": causal,
        }
    return {
        "status": status,
        "finding": finding,
        "causal_assessment": {
            "earliest_causal_divergence": causal,
        },
        "unobservability_reasons": list(reasons),
    }


def test_metric_reports_explicit_counts_and_supports_zero_denominator():
    assert metric(0, 0) == {
        "numerator": 0,
        "denominator": 0,
        "rate": None,
        "status": "not_supported",
    }
    assert metric(3, 4) == {
        "numerator": 3,
        "denominator": 4,
        "rate": 0.75,
        "status": "supported",
    }


def test_status_matches_all_three_observable_labels():
    assert status_matches("no_divergence", _prediction("no_divergence"))[0]
    assert status_matches("divergence", _prediction("localized"))[0]
    assert status_matches("unobservable", _prediction("unobservable"))[0]

    for label, wrong_status in (
        ("no_divergence", "localized"),
        ("divergence", "no_divergence"),
        ("unobservable", "localized"),
    ):
        ok, error = status_matches(label, _prediction(wrong_status))
        assert not ok
        assert error == "status_mismatch"


def test_strict_earliest_requires_family_source_step_and_proven():
    expected = {
        "predicate_id": "fault_outcome[1]",
        "source": "agentguard_trace",
        "step": 11,
        "causal_label": "proven_earliest",
    }
    ok, error = strict_earliest_match(
        expected,
        _prediction(
            predicate_id="fault_outcome[1]",
            source="agentguard_trace",
            step=11,
            causal="proven",
        ),
    )
    assert ok and error is None

    cases = [
        (
            _prediction(
                predicate_id="confirmation_statuses[1]",
                step=11,
                causal="proven",
            ),
            "wrong_predicate_family",
        ),
        (
            _prediction(
                predicate_id="fault_outcome[1]",
                source="private_runner_observations",
                step=11,
                causal="proven",
            ),
            "wrong_source",
        ),
        (
            _prediction(
                predicate_id="fault_outcome[1]",
                step=10,
                causal="proven",
            ),
            "wrong_step",
        ),
        (
            _prediction(
                predicate_id="fault_outcome[1]",
                step=11,
                causal="unproven",
            ),
            "causal_status_mismatch",
        ),
        (
            _prediction("no_divergence"),
            "status_mismatch",
        ),
        (
            _prediction("localized"),
            "missing_finding",
        ),
    ]
    for prediction, expected_error in cases:
        ok, error = strict_earliest_match(expected, prediction)
        assert not ok
        assert error == expected_error


def test_p3_cross_source_conflict_alone_gets_no_strict_credit(artifacts):
    _, _, manifest, _ = artifacts
    case = next(
        item for item in manifest["cases"] if item["case_id"] == "P3"
    )
    expected = case["expected"]
    conflict_only = _prediction(
        predicate_id="trace_runner_tool_result_conflict",
        source="agentguard_trace",
        step=11,
        causal="proven",
    )
    ok, error = strict_earliest_match(expected, conflict_only)
    assert not ok
    assert error == "wrong_predicate_family"

    canonical = _prediction(
        predicate_id="fault_outcome[1]",
        source="agentguard_trace",
        step=11,
        causal="proven",
    )
    assert strict_earliest_match(expected, canonical) == (True, None)


def test_observed_violation_requires_family_source_and_unproven(artifacts):
    _, _, _, report = artifacts
    baseline_record = report["frozen_scenarios"]["baseline_a"][
        "records"
    ][0]
    label = next(
        item
        for item in build_gold_label_catalog(
            artifacts[0],
            artifacts[1],
        )["labels"]
        if item["label_id"]
        == f"{baseline_record['scenario_id']}__baseline_a"
    )
    expected = label["expected_finding"]
    predicate = expected["accepted_predicate_ids"][0]
    actual_predicate = baseline_record["predicate_id"]
    assert actual_predicate in expected["accepted_predicate_ids"]

    ok, error = observed_violation_match(
        expected,
        _prediction(
            predicate_id=actual_predicate,
            source="private_runner_observations",
            causal="unproven",
        ),
        accepted_predicate_ids=expected["accepted_predicate_ids"],
        require_step=False,
    )
    assert ok and error is None

    cases = [
        (
            _prediction(
                predicate_id=actual_predicate,
                source="private_runner_observations",
                causal="proven",
            ),
            "causal_status_mismatch",
        ),
        (
            _prediction(
                predicate_id="unexpected_confirmation_status",
                source="private_runner_observations",
                causal="unproven",
            ),
            "wrong_predicate_family",
        ),
        (
            _prediction(
                predicate_id=actual_predicate,
                source="agentguard_trace",
                causal="unproven",
            ),
            "wrong_source",
        ),
    ]
    for prediction, expected_error in cases:
        ok, error = observed_violation_match(
            expected,
            prediction,
            accepted_predicate_ids=expected["accepted_predicate_ids"],
            require_step=False,
        )
        assert not ok
        assert error == expected_error

    u1_expected = {
        "predicate_id": "trace_runner_policy_conflict[0]",
        "source": "agentguard_trace",
        "step": 3,
    }
    ok, error = observed_violation_match(
        u1_expected,
        _prediction(
            predicate_id="trace_runner_policy_conflict[0]",
            step=4,
            causal="unproven",
        ),
        accepted_predicate_ids=["trace_runner_policy_conflict[0]"],
        require_step=True,
    )
    assert not ok
    assert error == "wrong_step"


def test_abstention_requires_status_and_reason_family():
    expected = {
        "reason_family": "agentguard_trace_unavailable",
        "accepted_reason_prefixes": ["agentguard_trace_unavailable"],
    }
    ok, error = abstention_match(
        expected,
        _prediction(
            "unobservable",
            causal="unproven",
            reasons=["agentguard_trace_unavailable:missing"],
        ),
    )
    assert ok and error is None

    ok, error = abstention_match(
        expected,
        _prediction(
            "unobservable",
            causal="unproven",
            reasons=["some_other_reason"],
        ),
    )
    assert not ok
    assert error == "wrong_reason_family"

    ok, error = abstention_match(
        expected,
        _prediction("localized", predicate_id="x", causal="unproven"),
    )
    assert not ok
    assert error == "status_mismatch"


def test_usable_diagnosis_is_independent_of_correctness():
    partial = _prediction(
        predicate_id="unexpected_confirmation_status",
        source="agentguard_trace",
        step=4,
        causal="proven",
    )
    assert usable_finding(partial) is True

    assert usable_finding(_prediction("localized")) is False
    assert (
        usable_finding(
            _prediction(
                predicate_id="x",
                source="agentguard_trace",
                causal=None,
            )
        )
        is False
    )
    assert usable_finding(_prediction("unobservable")) is False


def test_parent_cluster_weighting_requires_all_members():
    labels = [
        {
            "scenario_id": "a1",
            "parent_scenario_id": "p1",
            "observable_label": "no_divergence",
        },
        {
            "scenario_id": "a2",
            "parent_scenario_id": "p1",
            "observable_label": "no_divergence",
        },
        {
            "scenario_id": "b1",
            "parent_scenario_id": "p2",
            "observable_label": "no_divergence",
        },
        {
            "scenario_id": "b2",
            "parent_scenario_id": "p2",
            "observable_label": "no_divergence",
        },
    ]
    predictions = {
        "a1": _prediction("no_divergence"),
        "a2": _prediction("no_divergence"),
        "b1": _prediction("no_divergence"),
        "b2": _prediction("localized", predicate_id="x", causal="proven"),
    }
    result = _frozen_clean_group("agentguard", labels, predictions)
    assert result["status_accuracy"] == {
        "numerator": 1,
        "denominator": 2,
        "rate": 0.5,
        "status": "supported",
    }
    assert result["record_status_accuracy"]["numerator"] == 3
    assert result["record_status_accuracy"]["denominator"] == 4
    assert result["clean_run_false_positive"]["cluster_rate"][
        "numerator"
    ] == 1
    assert result["clean_run_false_positive"]["record_rate"][
        "numerator"
    ] == 1


def test_frozen_corpus_metrics_and_not_supported_boundaries(artifacts):
    _, _, _, report = artifacts
    frozen = report["frozen_scenarios"]

    for group in ("agentguard", "baseline_b"):
        metrics = frozen[group]
        assert metrics["status_accuracy"] == {
            "numerator": 20,
            "denominator": 20,
            "rate": 1.0,
            "status": "supported",
        }
        assert metrics["record_status_accuracy"] == {
            "numerator": 100,
            "denominator": 100,
            "rate": 1.0,
            "status": "supported",
        }
        assert metrics["clean_run_false_positive"]["cluster_rate"][
            "numerator"
        ] == 0
        assert metrics["error_categories"] == {"status": {}}
        assert metrics["strict_earliest_accuracy"]["status"] == (
            "not_supported"
        )
        assert metrics["diagnosis_coverage"]["status"] == "not_supported"
        assert metrics["abstention_accuracy"]["status"] == "not_supported"

    baseline_a = frozen["baseline_a"]
    assert baseline_a["status_accuracy"] == {
        "numerator": 28,
        "denominator": 28,
        "rate": 1.0,
        "status": "supported",
    }
    assert baseline_a["observed_violation_detection"] == {
        "numerator": 28,
        "denominator": 28,
        "rate": 1.0,
        "status": "supported",
    }
    assert baseline_a["diagnosis_coverage"] == {
        "numerator": 28,
        "denominator": 28,
        "rate": 1.0,
        "status": "supported",
    }
    assert baseline_a["parent_cluster_consistency"] == {
        "numerator": 12,
        "denominator": 12,
        "rate": 1.0,
        "status": "supported",
    }
    assert baseline_a["not_scored_count"] == 72
    assert baseline_a["not_scored_reasons"] == {
        "oracle_does_not_forbid_tool_execution": 72
    }
    assert baseline_a["strict_earliest_accuracy"]["status"] == (
        "not_supported"
    )
    assert baseline_a["abstention_accuracy"]["status"] == "not_supported"
    assert all(
        cluster["passed"] for cluster in baseline_a["parent_clusters"]
    )


def test_corruption_corpus_metrics_separate_categories(artifacts):
    _, _, _, report = artifacts
    corruption = report["corruption"]
    categories = corruption["categories"]

    assert categories["proven_earliest"]["status_accuracy"][
        "numerator"
    ] == 5
    assert categories["proven_earliest"]["strict_earliest_accuracy"] == {
        "numerator": 5,
        "denominator": 5,
        "rate": 1.0,
        "status": "supported",
    }
    assert categories["proven_earliest"]["diagnosis_coverage"][
        "numerator"
    ] == 5

    assert categories["observed_unproven"]["status_accuracy"][
        "numerator"
    ] == 5
    assert categories["observed_unproven"][
        "observed_violation_detection"
    ] == {
        "numerator": 5,
        "denominator": 5,
        "rate": 1.0,
        "status": "supported",
    }
    assert categories["observed_unproven"]["diagnosis_coverage"][
        "numerator"
    ] == 5

    assert categories["unobservable"]["status_accuracy"][
        "numerator"
    ] == 5
    assert categories["unobservable"]["abstention_accuracy"] == {
        "numerator": 5,
        "denominator": 5,
        "rate": 1.0,
        "status": "supported",
    }

    assert corruption["all_cases_status_accuracy"] == {
        "numerator": 15,
        "denominator": 15,
        "rate": 1.0,
        "status": "supported",
    }
    assert corruption["diagnosis_coverage"] == {
        "numerator": 10,
        "denominator": 10,
        "rate": 1.0,
        "status": "supported",
    }
    assert len(corruption["cases"]) == 15


def test_all_fifteen_corruption_cases_match_declared_expectations(artifacts):
    _, _, manifest, report = artifacts
    expected_by_id = {
        case["case_id"]: case for case in manifest["cases"]
    }
    for result in report["corruption"]["cases"]:
        case = expected_by_id[result["case_id"]]
        assert result["status_correct"] is True
        assert result["actual_status"] == case["expected"]["status"]
        if case["category"] == "proven_earliest":
            assert result["strict_correct"] is True
            assert result["strict_error"] is None
            assert result["actual_predicate_id"] == (
                case["expected"]["predicate_id"]
            )
            assert result["actual_source"] == case["expected"]["source"]
            assert result["actual_step"] == case["expected"]["step"]
            assert result["causal_status"] == "proven"
        elif case["category"] == "observed_unproven":
            assert result["observed_correct"] is True
            assert result["observed_error"] is None
            assert result["causal_status"] == "unproven"
        else:
            assert result["abstention_correct"] is True
            assert result["abstention_error"] is None
            assert result["usable_diagnosis"] is False


def test_p3_convention_is_enforced_and_reported(artifacts):
    _, _, _, report = artifacts
    p3 = report["corruption"]["p3_strict_convention"]
    assert p3["convention_id"] == "P3_canonical_fault_outcome_1"
    assert p3["strict_credit"] is True
    assert p3["actual_predicate_id"] == "fault_outcome[1]"
    assert p3["actual_source"] == "agentguard_trace"
    assert p3["actual_step"] == 11
    assert p3["actual_causal_status"] == "proven"
    assert p3["supporting_evidence_only"] == [
        "trace_runner_tool_result_conflict"
    ]


def test_report_is_byte_deterministic(artifacts):
    dataset, batch, _, report = artifacts
    second = build_accuracy_report(dataset=dataset, batch=batch)
    assert render_accuracy_report(report) == render_accuracy_report(second)
    assert json.dumps(report, sort_keys=True) == json.dumps(
        second, sort_keys=True
    )


def test_report_provenance_and_pending_review(artifacts):
    _, _, _, report = artifacts
    provenance = report["provenance"]
    assert provenance["review_status"] == (
        "independent_human_review_pending_for_all_labels"
    )
    assert provenance["catalog_wide_human_signoff"] is False
    assert provenance["project_owner_approved_conventions"] == [
        "P3_canonical_fault_outcome_1"
    ]
    assert provenance["static_core_labels_match_builder"] is True
    assert provenance["gold_catalog_schema_errors"] == []
    for key in (
        "gold_label_catalog_sha256",
        "corruption_manifest_sha256",
    ):
        assert len(provenance[key]) == 64
        int(provenance[key], 16)
    assert any(
        "Independent human review remains pending" in limitation
        for limitation in report["limitations"]
    )


def test_report_does_not_modify_frozen_labels(artifacts):
    dataset, batch, _, _ = artifacts
    before = hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest()
    before_manifest = hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
    build_accuracy_report(dataset=dataset, batch=batch)
    assert hashlib.sha256(GOLD_PATH.read_bytes()).hexdigest() == before
    assert (
        hashlib.sha256(MANIFEST_PATH.read_bytes()).hexdigest()
        == before_manifest
    )


def test_committed_report_artifact_matches_fresh_render(artifacts):
    _, _, _, report = artifacts
    assert REPORT_PATH.exists()
    assert REPORT_PATH.read_text(encoding="utf-8") == (
        render_accuracy_report(report)
    )
