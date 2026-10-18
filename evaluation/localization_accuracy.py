"""Deterministic localization accuracy and diagnosis coverage scoring.

Frozen inputs are loaded read-only: the Stage 2C scenarios and batch records,
the Phase 2 Step 1 gold catalog, and the corruption manifest. Scoring never
changes labels and never merges frozen and corruption corpora.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

from evaluation.corruptions import (
    apply_corruption,
    load_corruption_manifest,
)
from evaluation.gold_labels import (
    GROUPS,
    build_gold_label_catalog,
    load_gold_label_catalog,
    validate_gold_label_catalog,
)
from evaluation.localization import LOCALIZATION_SCHEMA_VERSION, localize_run
from evaluation.offline_batch import compose_stage2c_dataset, run_batch

ACCURACY_SCHEMA_VERSION = "1.0"
ACCURACY_REPORT_VERSION = "localization-accuracy-v1"
GOLD_LABELS_PATH = Path("evaluation/gold_labels_v1.json")
CORRUPTIONS_PATH = Path("evaluation/localization_corruptions_v1.json")
REPORT_PATH = Path("evaluation/reports/localization_accuracy_v1.json")

_CAUSAL_EXPECTATION = {
    "observed_violation": "unproven",
    "proven_earliest": "proven",
}


def metric(numerator: int, denominator: int) -> dict[str, Any]:
    """Return an explicit numerator/denominator metric.

    A zero denominator is reported as ``not_supported`` with ``rate=None``;
    it is never converted to a zero score.
    """
    if denominator <= 0:
        return {
            "numerator": numerator,
            "denominator": denominator,
            "rate": None,
            "status": "not_supported",
        }
    return {
        "numerator": numerator,
        "denominator": denominator,
        "rate": round(numerator / denominator, 6),
        "status": "supported",
    }


def _sha256(path: Path) -> str:
    return sha256(path.read_bytes()).hexdigest()


def _predicted_causal(prediction: Mapping[str, Any]) -> str | None:
    finding = prediction.get("finding")
    if isinstance(finding, Mapping):
        return finding.get("earliest_causal_divergence")
    assessment = prediction.get("causal_assessment")
    if isinstance(assessment, Mapping):
        return assessment.get("earliest_causal_divergence")
    return None


def _predicted_status_for(label: str) -> str:
    return {
        "no_divergence": "no_divergence",
        "divergence": "localized",
        "unobservable": "unobservable",
    }[label]


def status_matches(
    label: str,
    prediction: Mapping[str, Any],
) -> tuple[bool, str | None]:
    if label not in {"no_divergence", "divergence", "unobservable"}:
        raise ValueError(f"unsupported observable label: {label!r}")
    expected = _predicted_status_for(label)
    if prediction.get("status") != expected:
        return False, "status_mismatch"
    return True, None


def _finding_error(
    expected: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    accepted_predicate_ids: Iterable[str],
    expected_causal: str,
    require_step: bool,
) -> str | None:
    if prediction.get("status") != "localized":
        return "status_mismatch"
    finding = prediction.get("finding")
    if not isinstance(finding, Mapping):
        return "missing_finding"
    if _predicted_causal(prediction) != expected_causal:
        return "causal_status_mismatch"
    if finding.get("predicate_id") not in set(accepted_predicate_ids):
        return "wrong_predicate_family"
    source_ref = finding.get("source_ref")
    if not isinstance(source_ref, Mapping):
        return "wrong_source"
    if source_ref.get("source") != expected.get("source"):
        return "wrong_source"
    if (
        require_step
        and expected.get("step") is not None
        and source_ref.get("step") != expected.get("step")
    ):
        return "wrong_step"
    return None


def strict_earliest_match(
    expected: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Strict proven-earliest match: family, source, exact step, proven."""
    error = _finding_error(
        expected,
        prediction,
        accepted_predicate_ids=[expected["predicate_id"]],
        expected_causal="proven",
        require_step=True,
    )
    return error is None, error


def observed_violation_match(
    expected: Mapping[str, Any],
    prediction: Mapping[str, Any],
    *,
    accepted_predicate_ids: Iterable[str],
    require_step: bool,
) -> tuple[bool, str | None]:
    """Observed-violation match: family, source, and optional evidence step."""
    error = _finding_error(
        expected,
        prediction,
        accepted_predicate_ids=accepted_predicate_ids,
        expected_causal="unproven",
        require_step=require_step,
    )
    return error is None, error


def abstention_match(
    expected: Mapping[str, Any],
    prediction: Mapping[str, Any],
) -> tuple[bool, str | None]:
    """Abstention match: unobservable status and accepted reason family."""
    if prediction.get("status") != "unobservable":
        return False, "status_mismatch"
    prefixes = list(expected.get("accepted_reason_prefixes") or [])
    if not prefixes and expected.get("reason_family"):
        prefixes = [expected["reason_family"]]
    reasons = prediction.get("unobservability_reasons") or []
    if not any(
        isinstance(reason, str)
        and any(reason.startswith(prefix) for prefix in prefixes)
        for reason in reasons
    ):
        return False, "wrong_reason_family"
    return True, None


def usable_finding(prediction: Mapping[str, Any]) -> bool:
    """A diagnosis is usable when it has a finding and identifiable source.

    Usability is independent of correctness; accuracy is scored separately.
    """
    finding = prediction.get("finding")
    if prediction.get("status") != "localized":
        return False
    if not isinstance(finding, Mapping):
        return False
    if not finding.get("predicate_id"):
        return False
    source_ref = finding.get("source_ref")
    if not isinstance(source_ref, Mapping) or not source_ref:
        return False
    return _predicted_causal(prediction) in {"proven", "unproven"}


def _confusion(
    pairs: Iterable[tuple[str, str]],
) -> dict[str, dict[str, int]]:
    counts: dict[str, Counter[str]] = defaultdict(Counter)
    for gold, predicted in pairs:
        counts[gold][predicted] += 1
    return {
        gold: dict(sorted(predicted.items()))
        for gold, predicted in sorted(counts.items())
    }


def _error_counts(errors: Iterable[str | None]) -> dict[str, int]:
    return dict(
        sorted(
            Counter(
                error for error in errors if error is not None
            ).items()
        )
    )


def _frozen_clean_group(
    group: str,
    labels: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in labels:
        clusters[label["parent_scenario_id"]].append(label)

    record_pairs: list[tuple[str, str]] = []
    record_correct = 0
    false_positive_records = 0
    status_errors: list[str | None] = []
    cluster_results: list[dict[str, Any]] = []
    cluster_passes = 0
    false_positive_clusters = 0
    for parent in sorted(clusters):
        members = sorted(
            clusters[parent],
            key=lambda label: label["scenario_id"],
        )
        failed: list[str] = []
        any_localized = False
        for label in members:
            prediction = predictions[label["scenario_id"]]
            ok, error = status_matches(
                label["observable_label"],
                prediction,
            )
            status_errors.append(error)
            record_pairs.append(
                (label["observable_label"], prediction["status"])
            )
            record_correct += ok
            if prediction["status"] == "localized":
                false_positive_records += 1
                any_localized = True
            if not ok:
                failed.append(label["scenario_id"])
        cluster_pass = not failed
        cluster_passes += cluster_pass
        false_positive_clusters += any_localized
        cluster_results.append(
            {
                "parent_scenario_id": parent,
                "record_count": len(members),
                "passed": cluster_pass,
                "failed_scenario_ids": failed,
            }
        )

    denominator = len(labels)
    cluster_denominator = len(clusters)
    return {
        "primary_unit": "parent_cluster",
        "cluster_size": 5,
        "status_accuracy": metric(cluster_passes, cluster_denominator),
        "record_status_accuracy": metric(record_correct, denominator),
        "clean_run_false_positive": {
            "cluster_rate": metric(
                false_positive_clusters,
                cluster_denominator,
            ),
            "record_rate": metric(
                false_positive_records,
                denominator,
            ),
        },
        "confusion": _confusion(record_pairs),
        "error_categories": {
            "status": _error_counts(status_errors),
        },
        "cluster_confusion": {
            "pass": cluster_passes,
            "fail": cluster_denominator - cluster_passes,
        },
        "parent_clusters": cluster_results,
        "strict_earliest_accuracy": metric(0, 0),
        "diagnosis_coverage": metric(0, 0),
        "abstention_accuracy": metric(0, 0),
        "not_scored_count": 0,
        "not_scored_reasons": {},
    }


def _frozen_baseline_a(
    labels: list[dict[str, Any]],
    predictions: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    eligible = [
        label
        for label in labels
        if label["observable_label"] == "divergence"
    ]
    excluded = [
        label
        for label in labels
        if label["observable_label"] == "not_scored"
    ]
    eligible.sort(key=lambda label: label["scenario_id"])

    status_correct = 0
    observed_correct = 0
    usable = 0
    record_pairs: list[tuple[str, str]] = []
    status_errors: list[str | None] = []
    observed_errors: list[str | None] = []
    per_record: list[dict[str, Any]] = []
    clusters: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for label in eligible:
        clusters[label["parent_scenario_id"]].append(label)
        prediction = predictions[label["scenario_id"]]
        status_ok, status_error = status_matches(
            label["observable_label"],
            prediction,
        )
        expected = label["expected_finding"]
        observed_ok, observed_error = observed_violation_match(
            expected,
            prediction,
            accepted_predicate_ids=expected["accepted_predicate_ids"],
            require_step=False,
        )
        status_correct += status_ok
        observed_correct += observed_ok
        usable += usable_finding(prediction)
        record_pairs.append((label["observable_label"], prediction["status"]))
        status_errors.append(status_error)
        observed_errors.append(observed_error)
        finding = prediction.get("finding") or {}
        per_record.append(
            {
                "scenario_id": label["scenario_id"],
                "parent_scenario_id": label["parent_scenario_id"],
                "status": prediction["status"],
                "predicate_id": finding.get("predicate_id"),
                "source": (finding.get("source_ref") or {}).get("source"),
                "causal_status": _predicted_causal(prediction),
                "status_correct": status_ok,
                "observed_violation_correct": observed_ok,
                "observed_violation_error": observed_error,
                "usable_diagnosis": usable_finding(prediction),
            }
        )

    cluster_results: list[dict[str, Any]] = []
    cluster_passes = 0
    for parent in sorted(clusters):
        members = clusters[parent]
        failed = [
            item["scenario_id"]
            for item in per_record
            if item["parent_scenario_id"] == parent
            and not item["observed_violation_correct"]
        ]
        passed = not failed
        cluster_passes += passed
        cluster_results.append(
            {
                "parent_scenario_id": parent,
                "eligible_record_count": len(members),
                "passed": passed,
                "failed_scenario_ids": failed,
            }
        )

    return {
        "primary_unit": "record_for_observed_violation",
        "status_accuracy": metric(status_correct, len(eligible)),
        "observed_violation_detection": metric(
            observed_correct,
            len(eligible),
        ),
        "diagnosis_coverage": metric(usable, len(eligible)),
        "strict_earliest_accuracy": metric(0, 0),
        "abstention_accuracy": metric(0, 0),
        "parent_cluster_consistency": metric(
            cluster_passes,
            len(clusters),
        ),
        "parent_clusters": cluster_results,
        "confusion": _confusion(record_pairs),
        "error_categories": {
            "status": _error_counts(status_errors),
            "observed_violation": _error_counts(observed_errors),
        },
        "records": per_record,
        "not_scored_count": len(excluded),
        "not_scored_reasons": dict(
            sorted(
                Counter(
                    label.get("exclusion_reason") for label in excluded
                ).items()
            )
        ),
    }


def _corruption_category_report(
    results: list[dict[str, Any]],
    category: str,
) -> dict[str, Any]:
    members = [item for item in results if item["category"] == category]
    status_correct = sum(item["status_correct"] for item in members)
    confusion = _confusion(
        (item["expected_status"], item["actual_status"])
        for item in members
    )
    report: dict[str, Any] = {
        "status_accuracy": metric(status_correct, len(members)),
        "confusion": confusion,
        "cases": members,
    }
    if category == "proven_earliest":
        report["strict_earliest_accuracy"] = metric(
            sum(item["strict_correct"] for item in members),
            len(members),
        )
        report["diagnosis_coverage"] = metric(
            sum(item["usable_diagnosis"] for item in members),
            len(members),
        )
        report["error_categories"] = _error_counts(
            item["strict_error"] for item in members
        )
    elif category == "observed_unproven":
        report["observed_violation_detection"] = metric(
            sum(item["observed_correct"] for item in members),
            len(members),
        )
        report["diagnosis_coverage"] = metric(
            sum(item["usable_diagnosis"] for item in members),
            len(members),
        )
        report["error_categories"] = _error_counts(
            item["observed_error"] for item in members
        )
    elif category == "unobservable":
        report["abstention_accuracy"] = metric(
            sum(item["abstention_correct"] for item in members),
            len(members),
        )
        report["error_categories"] = _error_counts(
            item["abstention_error"] for item in members
        )
    return report


def _corruption_report(
    manifest: Mapping[str, Any],
    dataset: Any,
    batch: Any,
) -> dict[str, Any]:
    scenarios = {
        scenario["id"]: scenario for scenario in dataset.all_scenarios
    }
    records = {
        (record.scenario_id, record.group): record
        for record in batch.records
    }
    results: list[dict[str, Any]] = []
    for case in manifest["cases"]:
        scenario = scenarios[case["base_scenario_id"]]
        record = records[(case["base_scenario_id"], case["group"])]
        _, mutated = apply_corruption(case, scenario, record)
        prediction = localize_run(scenario, mutated)
        expected = case["expected"]
        category = case["category"]
        expected_status = expected["status"]
        actual_status = prediction["status"]
        status_ok = actual_status == expected_status

        strict_ok = False
        strict_error: str | None = None
        observed_ok = False
        observed_error: str | None = None
        abstention_ok = False
        abstention_error: str | None = None
        if category == "proven_earliest":
            strict_ok, strict_error = strict_earliest_match(
                expected,
                prediction,
            )
        elif category == "observed_unproven":
            observed_ok, observed_error = observed_violation_match(
                expected,
                prediction,
                accepted_predicate_ids=[expected["predicate_id"]],
                require_step=expected.get("step") is not None,
            )
        elif category == "unobservable":
            abstention_ok, abstention_error = abstention_match(
                expected,
                prediction,
            )

        finding = prediction.get("finding") or {}
        results.append(
            {
                "case_id": case["case_id"],
                "category": category,
                "base_scenario_id": case["base_scenario_id"],
                "group": case["group"],
                "expected_status": expected_status,
                "actual_status": actual_status,
                "status_correct": status_ok,
                "expected_predicate_id": expected.get("predicate_id"),
                "actual_predicate_id": finding.get("predicate_id"),
                "expected_source": expected.get("source"),
                "actual_source": (finding.get("source_ref") or {}).get(
                    "source"
                ),
                "expected_step": expected.get("step"),
                "actual_step": (finding.get("source_ref") or {}).get(
                    "step"
                ),
                "causal_status": _predicted_causal(prediction),
                "strict_correct": strict_ok,
                "strict_error": strict_error,
                "observed_correct": observed_ok,
                "observed_error": observed_error,
                "abstention_correct": abstention_ok,
                "abstention_error": abstention_error,
                "usable_diagnosis": usable_finding(prediction),
            }
        )

    results.sort(key=lambda item: item["case_id"])
    categories = {
        category: _corruption_category_report(results, category)
        for category in (
            "proven_earliest",
            "observed_unproven",
            "unobservable",
        )
    }

    divergence_results = [
        item
        for item in results
        if item["category"] in {"proven_earliest", "observed_unproven"}
    ]
    p3_result = next(
        item for item in results if item["case_id"] == "P3"
    )
    p3_convention = next(
        convention
        for convention in manifest["scoring_conventions"]
        if convention["case_id"] == "P3"
    )
    return {
        "primary_unit": "case",
        "categories": categories,
        "all_cases_status_accuracy": metric(
            sum(item["status_correct"] for item in results),
            len(results),
        ),
        "diagnosis_coverage": metric(
            sum(item["usable_diagnosis"] for item in divergence_results),
            len(divergence_results),
        ),
        "p3_strict_convention": {
            "convention_id": p3_convention["convention_id"],
            "canonical_gold_finding": p3_convention[
                "canonical_gold_finding"
            ],
            "supporting_evidence_only": p3_convention[
                "supporting_evidence_only"
            ],
            "strict_credit": p3_result["strict_correct"],
            "actual_predicate_id": p3_result["actual_predicate_id"],
            "actual_source": p3_result["actual_source"],
            "actual_step": p3_result["actual_step"],
            "actual_causal_status": p3_result["causal_status"],
        },
        "cases": results,
    }


def _frozen_report(
    catalog: Mapping[str, Any],
    dataset: Any,
    batch: Any,
) -> dict[str, Any]:
    scenarios = {
        scenario["id"]: scenario for scenario in dataset.all_scenarios
    }
    records = {
        (record.scenario_id, record.group): record
        for record in batch.records
    }
    labels_by_group = {group: [] for group in GROUPS}
    for label in catalog["labels"]:
        labels_by_group[label["group"]].append(label)

    predictions: dict[str, dict[str, dict[str, Any]]] = {
        group: {} for group in GROUPS
    }
    for group in GROUPS:
        for label in labels_by_group[group]:
            if label["observable_label"] == "not_scored":
                continue
            scenario_id = label["scenario_id"]
            predictions[group][scenario_id] = localize_run(
                scenarios[scenario_id],
                records[(scenario_id, group)],
            )

    return {
        "agentguard": _frozen_clean_group(
            "agentguard",
            labels_by_group["agentguard"],
            predictions["agentguard"],
        ),
        "baseline_b": _frozen_clean_group(
            "baseline_b",
            labels_by_group["baseline_b"],
            predictions["baseline_b"],
        ),
        "baseline_a": _frozen_baseline_a(
            labels_by_group["baseline_a"],
            predictions["baseline_a"],
        ),
    }


def build_accuracy_report(
    *,
    dataset: Any | None = None,
    batch: Any | None = None,
    gold_labels_path: str | Path = GOLD_LABELS_PATH,
    corruptions_path: str | Path = CORRUPTIONS_PATH,
) -> dict[str, Any]:
    """Build the deterministic accuracy/coverage report."""
    dataset = dataset if dataset is not None else compose_stage2c_dataset()
    batch = batch if batch is not None else run_batch()
    gold_path = Path(gold_labels_path)
    corruption_path = Path(corruptions_path)
    static_catalog = load_gold_label_catalog(gold_path)
    manifest = load_corruption_manifest(corruption_path)
    catalog = build_gold_label_catalog(dataset, batch)

    schema_errors = validate_gold_label_catalog(catalog, dataset)
    if schema_errors:
        raise ValueError(
            "gold label catalog is invalid: " + "; ".join(schema_errors)
        )
    built_core = {
        label["label_id"]: label
        for label in catalog["labels"]
        if not label["is_replication"]
    }
    static_core = {
        label["label_id"]: label
        for label in static_catalog["core_labels"]
    }
    if built_core != static_core:
        raise ValueError(
            "static core gold labels do not match the frozen label builder"
        )

    frozen = _frozen_report(catalog, dataset, batch)
    corruption = _corruption_report(manifest, dataset, batch)

    return {
        "schema_version": ACCURACY_SCHEMA_VERSION,
        "report_version": ACCURACY_REPORT_VERSION,
        "provenance": {
            "localization_schema_version": LOCALIZATION_SCHEMA_VERSION,
            "dataset_schema_version": batch.dataset_schema_version,
            "batch_schema_version": batch.batch_schema_version,
            "gold_label_catalog_path": str(gold_path),
            "gold_label_catalog_version": static_catalog[
                "catalog_version"
            ],
            "gold_label_catalog_sha256": _sha256(gold_path),
            "corruption_manifest_path": str(corruption_path),
            "corruption_manifest_version": manifest["manifest_version"],
            "corruption_manifest_sha256": _sha256(corruption_path),
            "gold_catalog_schema_errors": schema_errors,
            "static_core_labels_match_builder": True,
            "review_status": (
                "independent_human_review_pending_for_all_labels"
            ),
            "catalog_wide_human_signoff": False,
            "project_owner_approved_conventions": list(
                manifest["approval_scope"][
                    "project_owner_approved_conventions"
                ]
            ),
        },
        "frozen_scenarios": frozen,
        "corruption": corruption,
        "limitations": [
            "Independent human review remains pending for the complete label catalog.",
            "The frozen corpus contains no proven earliest-divergence labels; strict earliest-divergence accuracy is supported only by the corruption corpus.",
            "Baseline A is scored only on its 28 eligible observed-violation records; 72 records are not_scored.",
            "Derived deterministic and natural-language scenarios are replications, not independent samples; parent clusters are the primary unit.",
            "Corruption cases are synthetic and separate from frozen-scenario accuracy.",
        ],
    }


def render_accuracy_report(report: Mapping[str, Any]) -> str:
    """Render byte-stable JSON with sorted keys and no timestamps."""
    return json.dumps(report, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def write_accuracy_report(
    path: str | Path = REPORT_PATH,
    *,
    report: Mapping[str, Any] | None = None,
) -> str:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    rendered = render_accuracy_report(
        report if report is not None else build_accuracy_report()
    )
    destination.write_text(rendered, encoding="utf-8")
    return rendered
