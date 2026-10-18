"""Deterministic batch orchestration and JSON output for offline evaluation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from decimal import Decimal
from enum import Enum
import json
from pathlib import Path
import re
from typing import Any

from agent_graph import ExperimentGroup, RunStatus
from evaluation.offline_runner import (
    AgentGuardTraceCapture,
    EvaluationRunError,
    EvaluationRunResult,
    run_scenario,
)
from evaluation.scenario_expansion import (
    CATALOG_VERSION as EXPANSION_CATALOG_VERSION,
    GENERATOR_VERSION as EXPANSION_GENERATOR_VERSION,
    load_catalog as load_expansion_catalog,
)
from evaluation.scenario_natural_language import (
    CATALOG_VERSION as PERTURBATION_CATALOG_VERSION,
    GENERATOR_VERSION as PERTURBATION_GENERATOR_VERSION,
    Stage2CDataset,
    build_stage2c_dataset,
    load_perturbation_catalog,
)
from evaluation.scenarios import SCHEMA_VERSION, load_dataset

BATCH_SCHEMA_VERSION = "1.1"
_LEGACY_BATCH_SCHEMA_VERSION = "1.0"
_SUPPORTED_BATCH_SCHEMA_VERSIONS = {
    _LEGACY_BATCH_SCHEMA_VERSION,
    BATCH_SCHEMA_VERSION,
}
_TRACE_EVENT_FIELDS = {
    "llm_output": {"step", "event_type", "unavailable_fields"},
    "tool_call": {
        "step",
        "event_type",
        "tool_call_ref",
        "operation",
        "arguments",
        "args_source",
    },
    "policy_decision": {
        "step",
        "event_type",
        "tool_call_ref",
        "operation",
        "decision",
        "reason_code",
        "checked_identity",
        "idempotency_key_ref",
        "reason_text_available",
    },
    "human_confirmation": {
        "step",
        "event_type",
        "tool_call_ref",
        "operation",
        "arguments",
        "authenticated_as_actor",
        "status",
        "confirmation_id_available",
        "expiry_and_state_hash_available",
    },
    "tool_result": {
        "step",
        "event_type",
        "tool_call_ref",
        "operation",
        "status",
        "error_type",
        "idempotency_key_ref",
        "replayed",
        "state_changed",
        "result_payload_available",
        "response_delivery_count",
        "response_delivery_count_unavailable_reason",
    },
    "final_response": {
        "step",
        "event_type",
        "cited_tool_call_refs",
        "response_text_available",
        "response_class",
        "response_class_unavailable_reason",
    },
}
GROUPS = (
    ExperimentGroup.BASELINE_A,
    ExperimentGroup.BASELINE_B,
    ExperimentGroup.AGENTGUARD,
)
_SAFE_FAILURE_DETAIL = "Unexpected execution failure; exception message omitted."


@dataclass(frozen=True)
class BatchRunRecord:
    dataset_schema_version: str
    scenario_id: str
    group: str
    run_id: str
    execution_status: str
    run_outcome: str
    observations: EvaluationRunResult | None
    failure: dict[str, str] | None = None


@dataclass(frozen=True)
class EvaluationBatch:
    batch_schema_version: str
    dataset_schema_version: str
    expansion_catalog_version: str
    expansion_generator_version: str
    perturbation_catalog_version: str
    perturbation_generator_version: str
    core_scenario_count: int
    deterministic_variant_count: int
    natural_language_perturbation_count: int
    scenario_ids: tuple[str, ...]
    records: tuple[BatchRunRecord, ...]
    _sensitive_values: tuple[str, ...] = field(
        default=(),
        repr=False,
        compare=False,
    )

    @property
    def scenario_count(self) -> int:
        return len(self.scenario_ids)

    @property
    def record_count(self) -> int:
        return len(self.records)

    def to_json(self) -> str:
        validate_batch(self)
        payload = {
            "batch_schema_version": self.batch_schema_version,
            "dataset": {
                "schema_version": self.dataset_schema_version,
                "expansion_catalog_version": self.expansion_catalog_version,
                "expansion_generator_version": self.expansion_generator_version,
                "perturbation_catalog_version": self.perturbation_catalog_version,
                "perturbation_generator_version": self.perturbation_generator_version,
                "scenario_counts": {
                    "core": self.core_scenario_count,
                    "deterministic": self.deterministic_variant_count,
                    "natural_language": self.natural_language_perturbation_count,
                    "total": self.scenario_count,
                },
            },
            "records": [
                _redact_sensitive_values(
                    _record_to_dict(
                        record,
                        include_trace_capture=(
                            self.batch_schema_version == BATCH_SCHEMA_VERSION
                        ),
                    ),
                    self._sensitive_values,
                )
                for record in self.records
            ],
        }
        return json.dumps(payload, ensure_ascii=True, indent=2, sort_keys=True) + "\n"


def compose_stage2c_dataset(
    evaluation_dir: str | Path | None = None,
) -> Stage2CDataset:
    """Load frozen Stage 1/2 catalogs and compose the validated 100 scenarios."""
    source_dir = (
        Path(evaluation_dir)
        if evaluation_dir is not None
        else Path(__file__).resolve().parent
    )
    core = load_dataset(source_dir / "scenarios_v1.json")
    expansion = load_expansion_catalog(source_dir / "scenario_expansion_v1.json")
    perturbations = load_perturbation_catalog(
        source_dir / "scenario_perturbations_v1.json"
    )
    return build_stage2c_dataset(core, expansion, perturbations)


def run_batch(
    evaluation_dir: str | Path | None = None,
) -> EvaluationBatch:
    """Run every composed scenario once in each frozen experiment group."""
    dataset = compose_stage2c_dataset(evaluation_dir)
    scenarios = dataset.all_scenarios
    scenario_ids = tuple(scenario["id"] for scenario in scenarios)
    records: list[BatchRunRecord] = []
    sensitive_values = tuple(
        sorted(
            {
                value
                for scenario in scenarios
                for value in _scenario_sensitive_values(scenario)
            },
            key=len,
            reverse=True,
        )
    )

    for scenario in scenarios:
        scenario_id = scenario["id"]
        for group in GROUPS:
            run_id = f"{scenario_id}:{group.value}"
            try:
                result = run_scenario(scenario, group)
            except EvaluationRunError as error:
                partial = error.partial_result
                records.append(
                    BatchRunRecord(
                        dataset_schema_version=SCHEMA_VERSION,
                        scenario_id=scenario_id,
                        group=group.value,
                        run_id=run_id,
                        execution_status=partial.status.value,
                        run_outcome="failed",
                        observations=partial,
                        failure={
                            "type": error.failure_type,
                            "detail": _SAFE_FAILURE_DETAIL,
                        },
                    )
                )
            except Exception as error:
                records.append(
                    BatchRunRecord(
                        dataset_schema_version=SCHEMA_VERSION,
                        scenario_id=scenario_id,
                        group=group.value,
                        run_id=run_id,
                        execution_status=RunStatus.FAILED_CLOSED.value,
                        run_outcome="failed",
                        observations=None,
                        failure={
                            "type": type(error).__name__,
                            "detail": _SAFE_FAILURE_DETAIL,
                        },
                    )
                )
            else:
                records.append(
                    BatchRunRecord(
                        dataset_schema_version=SCHEMA_VERSION,
                        scenario_id=scenario_id,
                        group=group.value,
                        run_id=run_id,
                        execution_status=result.status.value,
                        run_outcome="completed",
                        observations=result,
                    )
                )

    batch = EvaluationBatch(
        batch_schema_version=BATCH_SCHEMA_VERSION,
        dataset_schema_version=SCHEMA_VERSION,
        expansion_catalog_version=EXPANSION_CATALOG_VERSION,
        expansion_generator_version=EXPANSION_GENERATOR_VERSION,
        perturbation_catalog_version=PERTURBATION_CATALOG_VERSION,
        perturbation_generator_version=PERTURBATION_GENERATOR_VERSION,
        core_scenario_count=len(dataset.core_scenarios),
        deterministic_variant_count=len(dataset.deterministic_variants),
        natural_language_perturbation_count=len(
            dataset.natural_language_perturbations
        ),
        scenario_ids=scenario_ids,
        records=tuple(records),
        _sensitive_values=sensitive_values,
    )
    validate_batch(batch)
    return batch


def validate_batch(batch: EvaluationBatch) -> None:
    """Reject incomplete, duplicated, reordered, or misidentified batch records."""
    if batch.batch_schema_version not in _SUPPORTED_BATCH_SCHEMA_VERSIONS:
        raise ValueError("unsupported batch schema version")
    canonical_scenarios = compose_stage2c_dataset().all_scenarios
    canonical_scenario_ids = tuple(
        scenario["id"] for scenario in canonical_scenarios
    )
    if (
        batch.core_scenario_count != 20
        or batch.deterministic_variant_count != 60
        or batch.natural_language_perturbation_count != 20
        or batch.dataset_schema_version != SCHEMA_VERSION
        or batch.scenario_ids != canonical_scenario_ids
    ):
        raise ValueError(
            "batch scenarios must match the canonical frozen 20+60+20 composition"
        )

    expected_records = tuple(
        (scenario_id, group.value)
        for scenario_id in canonical_scenario_ids
        for group in GROUPS
    )
    actual_records = tuple(
        (record.scenario_id, record.group) for record in batch.records
    )
    if len(batch.records) != 300 or actual_records != expected_records:
        raise ValueError(
            "batch records must cover every scenario/group pair exactly once in order"
        )

    for scenario_id, group, record in zip(
        (scenario_id for scenario_id, _ in expected_records),
        (group for _, group in expected_records),
        batch.records,
        strict=True,
    ):
        if (
            record.dataset_schema_version != batch.dataset_schema_version
            or record.run_id != f"{scenario_id}:{group}"
            or record.run_outcome not in {"completed", "failed"}
            or record.execution_status
            not in {status.value for status in RunStatus}
            or (record.run_outcome == "completed" and record.observations is None)
            or (record.run_outcome == "failed" and record.failure is None)
            or (
                record.run_outcome == "completed"
                and record.execution_status == RunStatus.FAILED_CLOSED.value
            )
            or (
                record.observations is not None
                and record.observations.status.value != record.execution_status
            )
        ):
            raise ValueError("batch record metadata or outcome is inconsistent")
        observations = record.observations
        trace_capture = observations.trace_capture if observations is not None else None
        if group != ExperimentGroup.AGENTGUARD.value and trace_capture is not None:
            raise ValueError("Baseline A/B records must not contain TraceRecorder payloads")
        if batch.batch_schema_version == _LEGACY_BATCH_SCHEMA_VERSION:
            if trace_capture is not None:
                raise ValueError("legacy 1.0 batch cannot contain trace-capture payloads")
        elif group == ExperimentGroup.AGENTGUARD.value and observations is not None:
            if trace_capture is None:
                raise ValueError("schema 1.1 AgentGuard observations require trace capture status")
            _validate_trace_capture(trace_capture, observations.trace_event_count)


def _validate_trace_capture(trace_capture: Any, trace_event_count: int) -> None:
    if not isinstance(trace_capture, AgentGuardTraceCapture):
        raise ValueError("trace-capture payload has an invalid structure")
    if (
        trace_capture.schema_version != 1
        or trace_capture.source_schema_version != 1
        or trace_capture.capture_status
        not in {"captured", "failed", "unavailable"}
        or trace_capture.unavailable_reason
        not in {
            None,
            "execution_recorder_not_initialized",
            "AgentGuard trace capture failed; event payload unavailable.",
        }
    ):
        raise ValueError("trace-capture schema metadata is invalid")
    if not isinstance(trace_capture.events, tuple):
        raise ValueError("trace events must retain their ordered tuple representation")
    if any(not isinstance(event, dict) for event in trace_capture.events):
        raise ValueError("trace event must be an object")
    if trace_capture.capture_status == "captured":
        if (
            trace_capture.unavailable_reason is not None
            or len(trace_capture.events) != trace_event_count
            or [event.get("step") for event in trace_capture.events]
            != list(range(1, trace_event_count + 1))
        ):
            raise ValueError("captured trace event count or steps are inconsistent")
    elif (
        not trace_capture.unavailable_reason
        or trace_capture.events
        or trace_event_count != 0
    ):
        raise ValueError("incomplete trace capture must state why events are unavailable")

    call_operations: dict[str, str] = {}
    for event in trace_capture.events:
        event_type = event.get("event_type")
        if (
            not isinstance(event_type, str)
            or event_type not in _TRACE_EVENT_FIELDS
            or event.keys() != _TRACE_EVENT_FIELDS[event_type]
        ):
            raise ValueError("trace event type is invalid")
        if (
            not isinstance(event.get("step"), int)
            or isinstance(event.get("step"), bool)
        ):
            raise ValueError("trace event step is invalid")
        if event_type != "llm_output" and event.get("operation") not in {
            "get_order",
            "cancel_order",
            "refund_order",
        }:
            if event_type != "final_response":
                raise ValueError("trace operation is invalid")
        if event_type == "tool_call":
            call_ref = event.get("tool_call_ref")
            operation = event.get("operation")
            expected_ref = f"call-{len(call_operations) + 1:03d}"
            if (
                not isinstance(call_ref, str)
                or call_ref != expected_ref
                or call_ref in call_operations
            ):
                raise ValueError("trace tool-call reference is invalid or duplicated")
            call_operations[call_ref] = operation
            _validate_projected_arguments(event.get("arguments"))
        elif event_type in {
            "policy_decision",
            "human_confirmation",
            "tool_result",
        }:
            _validate_prior_call_reference(event, call_operations)
            if event_type == "human_confirmation":
                _validate_projected_arguments(event.get("arguments"))
        elif event_type == "final_response":
            cited_refs = event.get("cited_tool_call_refs")
            if not isinstance(cited_refs, list):
                raise ValueError("final-response correlation references are invalid")
            if any(
                not isinstance(ref, str) or ref not in call_operations
                for ref in cited_refs
            ):
                raise ValueError(
                    "final-response citation must reference an earlier tool call"
                )


def _validate_prior_call_reference(
    event: dict[str, Any],
    call_operations: dict[str, str],
) -> None:
    call_ref = event.get("tool_call_ref")
    if not isinstance(call_ref, str) or call_ref not in call_operations:
        raise ValueError("trace event must reference an earlier tool call")
    if event.get("operation") != call_operations[call_ref]:
        raise ValueError("trace event operation does not match its tool call")


def _validate_projected_arguments(arguments: Any) -> None:
    if not isinstance(arguments, dict) or "unrecognized_parameter_count" not in arguments:
        raise ValueError("projected tool arguments have an invalid structure")
    extra_count = arguments["unrecognized_parameter_count"]
    if not isinstance(extra_count, int) or isinstance(extra_count, bool) or extra_count < 0:
        raise ValueError("unrecognized parameter count is invalid")
    if arguments.keys() - {
        "unrecognized_parameter_count",
        "order_target",
        "amount",
        "currency",
    }:
        raise ValueError("projected tool arguments contain unapproved fields")

    if "order_target" in arguments:
        target = arguments["order_target"]
        if not isinstance(target, dict):
            raise ValueError("projected order target is invalid")
        classification = target.get("classification")
        if classification in {"invalid_target", "unknown_target"}:
            if target.keys() != {"classification"}:
                raise ValueError("invalid or unknown order target contains extra data")
        elif classification == "known_target":
            if (
                target.keys() != {"classification", "order_ref", "ownership"}
                or not isinstance(target.get("order_ref"), str)
                or not re.fullmatch(r"order-[0-9]{3}", target["order_ref"])
                or target.get("ownership")
                not in {"actor_owned", "foreign_owned"}
            ):
                raise ValueError("known order target is invalid")
        else:
            raise ValueError("order target classification is invalid")

    for name in ("amount", "currency"):
        if name not in arguments:
            continue
        parameter = arguments[name]
        if not isinstance(parameter, dict):
            raise ValueError(f"projected {name} is invalid")
        if parameter == {"status": "invalid"}:
            continue
        if parameter.keys() != {"status", "value"} or parameter.get("status") != "valid":
            raise ValueError(f"projected {name} is invalid")
        value = parameter.get("value")
        if name == "amount":
            if (
                not isinstance(value, str)
                or len(value) > 21
                or not re.fullmatch(r"(?:0|[1-9][0-9]*)\.[0-9]{2}", value)
                or Decimal(value) <= 0
            ):
                raise ValueError("projected amount is not canonical")
        elif not isinstance(value, str) or not re.fullmatch(r"[A-Z]{3}", value):
            raise ValueError("projected currency is not canonical")


def _record_to_dict(
    record: BatchRunRecord,
    *,
    include_trace_capture: bool,
) -> dict[str, Any]:
    observations = (
        _json_safe(asdict(record.observations))
        if record.observations is not None
        else None
    )
    if observations is not None and not include_trace_capture:
        observations.pop("trace_capture", None)
    return {
        "dataset_schema_version": record.dataset_schema_version,
        "scenario_id": record.scenario_id,
        "group": record.group,
        "run_id": record.run_id,
        "execution_status": record.execution_status,
        "run_outcome": record.run_outcome,
        "observations": observations,
        "failure": record.failure,
    }


def _json_safe(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _scenario_sensitive_values(scenario: dict[str, Any]) -> set[str]:
    sensitive_markers = (
        "capability",
        "confirmation_id",
        "credential",
        "idempotency_key",
        "secret",
        "token",
        "resume_marker",
    )
    values: set[str] = set()

    def visit(value: Any, field_name: str = "") -> None:
        if isinstance(value, dict):
            for key, item in value.items():
                visit(item, str(key))
        elif isinstance(value, (list, tuple)):
            for item in value:
                visit(item, field_name)
        elif (
            isinstance(value, str)
            and value
            and any(marker in field_name.lower() for marker in sensitive_markers)
        ):
            values.add(value)

    visit(scenario)
    return values


def _redact_sensitive_values(value: Any, sensitive_values: tuple[str, ...]) -> Any:
    if isinstance(value, dict):
        return {
            key: _redact_sensitive_values(item, sensitive_values)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_redact_sensitive_values(item, sensitive_values) for item in value]
    if isinstance(value, str):
        for sensitive_value in sensitive_values:
            value = value.replace(sensitive_value, "[REDACTED]")
    return value
