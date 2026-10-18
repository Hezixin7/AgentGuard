from dataclasses import replace
import json

import pytest

from agent_graph import ExperimentGroup, RunStatus
from evaluation import offline_batch
from evaluation.offline_batch import compose_stage2c_dataset, run_batch
from evaluation.offline_runner import EvaluationRunError
from agent_graph import AgentOrchestrator


def test_batch_composes_exact_frozen_dataset_and_three_groups():
    dataset = compose_stage2c_dataset()
    batch = run_batch()

    assert dataset.scenario_count == 100
    assert len(dataset.core_scenarios) == 20
    assert len(dataset.deterministic_variants) == 60
    assert len(dataset.natural_language_perturbations) == 20
    assert batch.record_count == 300
    assert batch.scenario_count == 100
    assert batch.batch_schema_version == "1.1"
    assert len(
        {(record.scenario_id, record.group) for record in batch.records}
    ) == 300
    assert [record.group for record in batch.records[:3]] == [
        "baseline_a",
        "baseline_b",
        "agentguard",
    ]
    assert all(
        record.run_id == f"{record.scenario_id}:{record.group}"
        for record in batch.records
    )
    assert all(record.run_outcome == "completed" for record in batch.records)
    assert all(
        record.observations is None
        or (
            record.observations.trace_capture is None
            if record.group != "agentguard"
            else record.observations.trace_capture is not None
        )
        for record in batch.records
    )
    offline_batch.validate_batch(batch)


def test_record_order_and_json_are_semantically_reproducible():
    first = run_batch()
    second = run_batch()

    assert [record.run_id for record in first.records] == [
        record.run_id for record in second.records
    ]
    assert first.to_json() == second.to_json()


def test_schema_11_exports_trace_payloads_only_for_agentguard():
    batch = run_batch()
    payload = json.loads(batch.to_json())

    assert payload["batch_schema_version"] == "1.1"
    baseline_a = next(
        item for item in payload["records"] if item["group"] == "baseline_a"
    )
    baseline_b = next(
        item for item in payload["records"] if item["group"] == "baseline_b"
    )
    agentguard = next(
        item for item in payload["records"] if item["group"] == "agentguard"
    )
    assert baseline_a["observations"]["trace_capture"] is None
    assert baseline_b["observations"]["trace_capture"] is None
    assert agentguard["observations"]["trace_capture"]["capture_status"] == "captured"
    assert (
        len(agentguard["observations"]["trace_capture"]["events"])
        == agentguard["observations"]["trace_event_count"]
    )


def test_legacy_schema_10_without_trace_payloads_remains_valid():
    batch = run_batch()
    legacy_records = tuple(
        replace(
            record,
            observations=(
                replace(
                    record.observations,
                    trace_event_count=0,
                    trace_capture=None,
                )
                if record.observations is not None
                else None
            ),
        )
        for record in batch.records
    )
    legacy = replace(
        batch,
        batch_schema_version="1.0",
        records=legacy_records,
    )

    offline_batch.validate_batch(legacy)
    payload = json.loads(legacy.to_json())
    assert payload["batch_schema_version"] == "1.0"
    assert all(
        "trace_capture" not in record["observations"]
        for record in payload["records"]
        if record["observations"] is not None
    )


def test_schema_11_rejects_missing_or_malformed_agentguard_trace_capture():
    batch = run_batch()
    record = next(
        record
        for record in batch.records
        if record.group == "agentguard" and record.observations is not None
    )
    assert record.observations is not None

    missing = tuple(
        replace(
            item,
            observations=replace(item.observations, trace_capture=None),
        )
        if item is record
        else item
        for item in batch.records
    )
    with pytest.raises(ValueError, match="require trace capture status"):
        offline_batch.validate_batch(replace(batch, records=missing))

    capture = record.observations.trace_capture
    assert capture is not None
    events = list(capture.events)
    events[0] = {**events[0], "step": 99}
    malformed = tuple(
        replace(
            item,
            observations=replace(
                item.observations,
                trace_capture=replace(capture, events=tuple(events)),
            ),
        )
        if item is record
        else item
        for item in batch.records
    )
    with pytest.raises(ValueError, match="count or steps"):
        offline_batch.validate_batch(replace(batch, records=malformed))


def test_trace_validator_rejects_forward_call_reference():
    batch = run_batch()
    record = next(
        record
        for record in batch.records
        if record.scenario_id == "core-019"
        and record.group == "agentguard"
        and record.observations is not None
    )
    assert record.observations is not None
    capture = record.observations.trace_capture
    assert capture is not None
    events = list(capture.events)
    call_refs = [
        event["tool_call_ref"]
        for event in events
        if event["event_type"] == "tool_call"
    ]
    policy_index = next(
        index
        for index, event in enumerate(events)
        if event["event_type"] == "policy_decision"
    )
    events[policy_index] = {
        **events[policy_index],
        "tool_call_ref": call_refs[-1],
    }
    altered_capture = replace(capture, events=tuple(events))
    altered_observations = replace(
        record.observations,
        trace_capture=altered_capture,
    )
    altered_records = tuple(
        replace(item, observations=altered_observations)
        if item is record
        else item
        for item in batch.records
    )

    with pytest.raises(ValueError, match="earlier tool call"):
        offline_batch.validate_batch(replace(batch, records=altered_records))

    events = list(capture.events)
    events[policy_index] = {
        **events[policy_index],
        "tool_call_ref": "call-999",
    }
    unknown_observations = replace(
        record.observations,
        trace_capture=replace(capture, events=tuple(events)),
    )
    unknown_records = tuple(
        replace(item, observations=unknown_observations)
        if item is record
        else item
        for item in batch.records
    )
    with pytest.raises(ValueError, match="earlier tool call"):
        offline_batch.validate_batch(replace(batch, records=unknown_records))


def test_trace_validator_rejects_operation_mismatch_and_future_citation():
    batch = run_batch()
    record = next(
        record
        for record in batch.records
        if record.scenario_id == "core-019"
        and record.group == "agentguard"
        and record.observations is not None
    )
    assert record.observations is not None
    capture = record.observations.trace_capture
    assert capture is not None
    events = list(capture.events)
    policy_index = next(
        index
        for index, event in enumerate(events)
        if event["event_type"] == "policy_decision"
    )
    events[policy_index] = {
        **events[policy_index],
        "operation": "cancel_order",
    }
    mismatch_observations = replace(
        record.observations,
        trace_capture=replace(capture, events=tuple(events)),
    )
    mismatch_records = tuple(
        replace(item, observations=mismatch_observations)
        if item is record
        else item
        for item in batch.records
    )
    with pytest.raises(ValueError, match="operation does not match"):
        offline_batch.validate_batch(replace(batch, records=mismatch_records))

    events = list(capture.events)
    second_call_index = next(
        index
        for index, event in enumerate(events)
        if event["event_type"] == "tool_call"
        and event["tool_call_ref"] == "call-002"
    )
    final_index = next(
        index
        for index, event in enumerate(events)
        if event["event_type"] == "final_response"
    )
    final_event = events.pop(final_index)
    events.insert(second_call_index, final_event)
    renumbered = tuple(
        {**event, "step": step}
        for step, event in enumerate(events, start=1)
    )
    citation_observations = replace(
        record.observations,
        trace_capture=replace(capture, events=renumbered),
    )
    citation_records = tuple(
        replace(item, observations=citation_observations)
        if item is record
        else item
        for item in batch.records
    )
    with pytest.raises(ValueError, match="citation must reference an earlier"):
        offline_batch.validate_batch(replace(batch, records=citation_records))


def test_runs_have_independent_state_and_baseline_outcomes_match():
    batch = run_batch()
    records = {
        (record.scenario_id, record.group): record
        for record in batch.records
    }

    approved = records[("core-003", "baseline_b")].observations
    rejected_variant = records[("core-003-v1", "baseline_b")].observations
    assert approved is not None and rejected_variant is not None
    assert approved.order_states["S300"]["status"] == "refunded"
    assert rejected_variant.order_states["S300"]["status"] == "shipped"
    assert approved.agent_state_mutations == 1
    assert rejected_variant.agent_state_mutations == 0

    for scenario_id in batch.scenario_ids:
        baseline = records[(scenario_id, "baseline_b")].observations
        guarded = records[(scenario_id, "agentguard")].observations
        assert baseline is not None and guarded is not None
        assert baseline.status == guarded.status
        assert baseline.response_class == guarded.response_class
        assert baseline.tool_attempts == guarded.tool_attempts
        assert baseline.tool_invocations == guarded.tool_invocations
        assert baseline.agent_state_mutations == guarded.agent_state_mutations
        assert baseline.external_state_mutations == guarded.external_state_mutations
        assert baseline.order_states == guarded.order_states
        assert baseline.refunds == guarded.refunds
        assert baseline.idempotency_records == guarded.idempotency_records
        assert baseline.policy_decisions == guarded.policy_decisions
        assert baseline.confirmation_statuses == guarded.confirmation_statuses


def test_batch_continues_after_unexpected_failure_and_keeps_partial_result(
    monkeypatch,
):
    real_run_scenario = offline_batch.run_scenario

    def fail_one_pair(scenario, group):
        result = real_run_scenario(scenario, group)
        if scenario["id"] == "core-009" and group is ExperimentGroup.BASELINE_A:
            sensitive_key = scenario["trusted_idempotency_key"]
            result = replace(result, sanitized_response=sensitive_key)
            raise EvaluationRunError(
                scenario["id"],
                group,
                "RuntimeError",
                sensitive_key,
                result,
            )
        if scenario["id"] == "core-010" and group is ExperimentGroup.BASELINE_A:
            raise ValueError("unexpected value error")
        return result

    monkeypatch.setattr(offline_batch, "run_scenario", fail_one_pair)
    batch = run_batch()

    failed = next(
        record
        for record in batch.records
        if (record.scenario_id, record.group) == ("core-009", "baseline_a")
    )
    generic_failure = next(
        record
        for record in batch.records
        if (record.scenario_id, record.group) == ("core-010", "baseline_a")
    )
    assert batch.record_count == 300
    assert failed.run_outcome == "failed"
    assert failed.observations is not None
    assert failed.observations.sanitized_response == (
        "seed-key-1"
    )
    assert failed.failure == {
        "type": "RuntimeError",
        "detail": "Unexpected execution failure; exception message omitted.",
    }
    assert generic_failure.run_outcome == "failed"
    assert generic_failure.execution_status == "failed_closed"
    assert generic_failure.observations is None
    assert batch.records[30].run_outcome == "completed"
    offline_batch.validate_batch(batch)

    serialized = batch.to_json()
    assert "seed-key-1" not in serialized
    assert "[REDACTED]" in serialized


def test_batch_continues_after_required_trace_capture_failure(monkeypatch):
    original_trace_events = AgentOrchestrator.trace_events

    def fail_one_capture(self, thread_id):
        execution = self._executions[thread_id]
        if execution.run_id == "core-001:agentguard":
            raise RuntimeError("injected trace read failure")
        return original_trace_events(self, thread_id)

    monkeypatch.setattr(AgentOrchestrator, "trace_events", fail_one_capture)
    batch = run_batch()

    failed = next(
        record
        for record in batch.records
        if (record.scenario_id, record.group) == ("core-001", "agentguard")
    )
    next_record = next(
        record
        for record in batch.records
        if (record.scenario_id, record.group) == ("core-002", "baseline_a")
    )
    assert batch.record_count == 300
    assert failed.run_outcome == "failed"
    assert failed.failure == {
        "type": "TraceCaptureError",
        "detail": "Unexpected execution failure; exception message omitted.",
    }
    assert failed.observations is not None
    assert failed.observations.trace_capture is not None
    assert failed.observations.trace_capture.capture_status == "failed"
    assert next_record.run_outcome == "completed"
    offline_batch.validate_batch(batch)


def test_batch_safe_json_excludes_fixture_idempotency_values():
    batch = run_batch()
    dataset = compose_stage2c_dataset()
    sensitive_keys = {
        seed["idempotency_key"]
        for scenario in dataset.all_scenarios
        for seed in scenario["initial_idempotency"]
    }
    sensitive_keys.update(
        scenario["trusted_idempotency_key"]
        for scenario in dataset.all_scenarios
        if "trusted_idempotency_key" in scenario
    )

    output = batch.to_json()
    decoded = json.loads(output)

    assert decoded["dataset"]["scenario_counts"] == {
        "core": 20,
        "deterministic": 60,
        "natural_language": 20,
        "total": 100,
    }
    assert len(decoded["records"]) == 300
    assert all(key not in output for key in sensitive_keys)
    assert all(
        record["dataset_schema_version"] == batch.dataset_schema_version
        for record in decoded["records"]
    )


def test_baseline_a_timing_mismatch_is_retained():
    batch = run_batch()
    records = {
        (record.scenario_id, record.group): record
        for record in batch.records
    }

    baseline_a = records[("core-017", "baseline_a")].observations
    baseline_b = records[("core-017", "baseline_b")].observations
    assert baseline_a is not None and baseline_b is not None
    assert baseline_a.baseline_a_timing_mismatch is True
    assert baseline_b.baseline_a_timing_mismatch is False


def test_validation_rejects_missing_or_duplicate_record():
    batch = run_batch()

    with pytest.raises(ValueError, match="cover every scenario/group pair"):
        offline_batch.validate_batch(replace(batch, records=batch.records[:-1]))

    duplicate_records = (*batch.records[:-1], batch.records[0])
    with pytest.raises(ValueError, match="cover every scenario/group pair"):
        offline_batch.validate_batch(replace(batch, records=duplicate_records))


def test_validation_rejects_substituted_scenario_id_and_records():
    batch = run_batch()
    records = list(batch.records)
    for index in range(3):
        records[index] = replace(
            records[index],
            scenario_id="surprise",
            run_id=f"surprise:{records[index].group}",
        )
    scenario_ids = ("surprise", *batch.scenario_ids[1:])

    with pytest.raises(ValueError, match="canonical frozen"):
        offline_batch.validate_batch(
            replace(batch, scenario_ids=scenario_ids, records=tuple(records))
        )


def test_validation_rejects_invalid_execution_status():
    batch = run_batch()
    records = (
        replace(batch.records[0], execution_status="nonsense"),
        *batch.records[1:],
    )

    with pytest.raises(ValueError, match="metadata or outcome"):
        offline_batch.validate_batch(replace(batch, records=records))


def test_validation_rejects_reordered_scenario_group_records():
    batch = run_batch()
    records = list(batch.records)
    records[0], records[1] = records[1], records[0]

    with pytest.raises(ValueError, match="cover every scenario/group pair"):
        offline_batch.validate_batch(replace(batch, records=tuple(records)))


def test_validation_accepts_explicit_failed_closed_exception_record(
    monkeypatch,
):
    def fail_one_pair(scenario, group):
        if scenario["id"] == "core-001" and group is ExperimentGroup.BASELINE_A:
            raise RuntimeError("injected runner exception")
        return offline_batch.run_scenario(scenario, group)

    monkeypatch.setattr(offline_batch, "run_scenario", fail_one_pair)
    batch = run_batch()
    failed_closed = batch.records[0]

    assert failed_closed.execution_status == RunStatus.FAILED_CLOSED.value
    assert failed_closed.run_outcome == "failed"
    assert failed_closed.observations is None
    assert failed_closed.failure == {
        "type": "RuntimeError",
        "detail": "Unexpected execution failure; exception message omitted.",
    }
    offline_batch.validate_batch(batch)
