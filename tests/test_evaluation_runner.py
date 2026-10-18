import hashlib
import json
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path

import pytest

from agent_graph import AgentOrchestrator, ExperimentGroup, RunStatus
from evaluation.offline_runner import EvaluationRunError, run_scenario
from evaluation.scenario_expansion import expand_catalog, load_catalog
from evaluation.scenario_natural_language import (
    build_stage2c_dataset,
    load_perturbation_catalog,
)
from evaluation.scenarios import load_dataset
from trace_events import (
    FinalResponseEvent,
    HumanConfirmationEvent,
    LLMOutputEvent,
    ToolCallEvent,
    ToolResultEvent,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def scenarios():
    core = load_dataset(ROOT / "evaluation" / "scenarios_v1.json")
    dataset = build_stage2c_dataset(
        core,
        load_catalog(ROOT / "evaluation" / "scenario_expansion_v1.json"),
        load_perturbation_catalog(
            ROOT / "evaluation" / "scenario_perturbations_v1.json"
        ),
    )
    assert dataset.scenario_count == 100
    return {scenario["id"]: scenario for scenario in dataset.all_scenarios}


def test_authorized_read_records_attempt_and_invocation(scenarios):
    result = run_scenario(scenarios["core-001"], ExperimentGroup.BASELINE_B)

    assert result.status is RunStatus.COMPLETED
    assert result.response_class == "completed"
    assert (result.tool_attempts, result.tool_invocations) == (1, 1)
    assert result.agent_state_mutations == 0
    assert result.order_states["P100"]["status"] == "pending"
    assert result.trace_event_count == 0


def test_approved_cancellation_and_refund_commit_once(scenarios):
    cancellation = run_scenario(scenarios["core-002"], ExperimentGroup.BASELINE_B)
    refund = run_scenario(scenarios["core-003"], ExperimentGroup.AGENTGUARD)

    assert cancellation.order_states["P100"]["status"] == "cancelled"
    assert cancellation.confirmation_statuses == (
        "requested",
        "confirmed",
        "consumed",
    )
    assert cancellation.agent_state_mutations == 1
    assert refund.order_states["S300"]["status"] == "refunded"
    assert len(refund.refunds) == len(refund.idempotency_records) == 1
    assert refund.agent_state_mutations == 1
    assert refund.trace_event_count > 0


def test_cross_user_denial_does_not_invoke_tool(scenarios):
    result = run_scenario(scenarios["core-006"], ExperimentGroup.BASELINE_B)

    assert result.response_class == "denied"
    assert result.sanitized_response == "I cannot complete that request."
    assert result.tool_attempts == 1
    assert result.tool_invocations == 0
    assert result.agent_state_mutations == 0
    assert result.policy_decisions[0]["reason_code"] == "order_unavailable"


def test_missing_rejected_expired_and_stale_confirmation(scenarios):
    missing = run_scenario(scenarios["core-011"], ExperimentGroup.AGENTGUARD)
    rejected = run_scenario(scenarios["core-004"], ExperimentGroup.BASELINE_B)
    expired = run_scenario(scenarios["core-016"], ExperimentGroup.BASELINE_B)
    stale = run_scenario(scenarios["core-017"], ExperimentGroup.AGENTGUARD)

    assert missing.status is RunStatus.PAUSED
    assert missing.response_class == "confirmation_required"
    assert missing.confirmation_statuses == ("requested",)
    assert rejected.response_class == "denied"
    assert rejected.confirmation_statuses == ("requested", "rejected")
    assert rejected.tool_invocations == 0
    assert expired.response_class == "denied"
    assert expired.confirmation_statuses == ("requested", "expired")
    assert expired.tool_invocations == 0
    assert stale.response_class == "denied"
    assert stale.confirmation_statuses == ("requested", "consumed")
    assert stale.external_state_mutations == 1
    assert stale.agent_state_mutations == 0
    assert stale.order_states["P100"]["status"] == "cancelled"
    assert stale.policy_decisions[-1]["reason_code"] == "confirmation_state_changed"


def test_seeded_idempotency_replay_and_conflict_use_fixture_keys(scenarios):
    replay = run_scenario(scenarios["core-008"], ExperimentGroup.BASELINE_B)
    conflict = run_scenario(scenarios["core-009"], ExperimentGroup.BASELINE_B)

    expected_key_ref = f"fixture:{hashlib.sha256(b'seed-key-1').hexdigest()}"
    assert replay.tool_invocations == 1
    assert replay.agent_state_mutations == 0
    assert replay.idempotency_records[0]["idempotency_key_ref"] == expected_key_ref
    assert replay.tool_results[0].idempotency_key_ref == expected_key_ref
    assert replay.tool_attempts == 1
    assert conflict.response_class == "denied"
    assert conflict.tool_invocations == 0
    assert conflict.idempotency_records[0]["idempotency_key_ref"] == expected_key_ref
    assert conflict.attempted_tool_calls[0]["idempotency_key_ref"] == expected_key_ref


@pytest.mark.parametrize(
    ("scenario_id", "attempts", "invocations", "mutations", "response_class"),
    [
        ("core-018", 2, 1, 0, "confirmation_required"),
        ("core-019", 2, 2, 1, "completed"),
        ("core-020", 1, 1, 1, "completed"),
    ],
)
def test_fault_plan_and_retry_observations(
    scenarios,
    scenario_id,
    attempts,
    invocations,
    mutations,
    response_class,
):
    result = run_scenario(scenarios[scenario_id], ExperimentGroup.BASELINE_B)

    assert result.tool_attempts == attempts
    assert result.tool_invocations == invocations
    assert result.agent_state_mutations == mutations
    assert result.response_class == response_class
    assert len(result.refunds) == len(result.idempotency_records) == mutations
    if scenario_id == "core-018":
        assert result.confirmation_statuses == (
            "requested",
            "confirmed",
            "consumed",
            "requested",
        )
        assert result.tool_results[0].committed is False
    elif scenario_id == "core-019":
        assert result.tool_results[0].committed is True
        assert result.tool_results[1].outcome == "replayed"
        assert result.tool_results[0].idempotency_key_ref == (
            result.tool_results[1].idempotency_key_ref
        )
        assert result.tool_results[0].idempotency_key_ref == "generated"
    else:
        assert result.tool_results[0].fault == "duplicate_response"
        assert result.tool_results[0].response_delivery_count == 2


def test_baseline_a_timeout_retry_counts_each_attempt(scenarios):
    baseline_a = run_scenario(scenarios["core-019"], ExperimentGroup.BASELINE_A)
    baseline_b = run_scenario(scenarios["core-019"], ExperimentGroup.BASELINE_B)
    agentguard = run_scenario(scenarios["core-019"], ExperimentGroup.AGENTGUARD)

    for result in (baseline_a, baseline_b, agentguard):
        assert result.tool_attempts == 2
        assert result.tool_invocations == 2
        assert result.agent_state_mutations == 1
    assert baseline_a.attempted_tool_calls == baseline_b.attempted_tool_calls
    assert baseline_b.attempted_tool_calls == agentguard.attempted_tool_calls


def test_external_state_change_is_separate_from_baseline_a_mutation(scenarios):
    result = run_scenario(scenarios["core-017"], ExperimentGroup.BASELINE_A)

    assert result.baseline_a_timing_mismatch is True
    assert result.external_state_mutations == 1
    assert result.agent_state_mutations == 0
    assert result.order_states["P100"]["status"] == "cancelled"
    assert result.response_class == "error"
    assert result.tool_invocations == 1
    assert result.trace_event_count == 0


def test_explicit_trusted_key_is_injected_before_policy_evaluation(scenarios):
    result = run_scenario(scenarios["core-009"], ExperimentGroup.AGENTGUARD)

    assert result.tool_attempts == 1
    assert result.tool_invocations == 0
    expected = f"fixture:{hashlib.sha256(b'seed-key-1').hexdigest()}"
    assert result.idempotency_records[0]["idempotency_key_ref"] == expected
    assert result.attempted_tool_calls[0]["idempotency_key_ref"] == expected


def test_generated_keys_are_normalized_for_reproducible_outputs(scenarios):
    first = run_scenario(scenarios["core-019"], ExperimentGroup.BASELINE_B)
    second = run_scenario(scenarios["core-019"], ExperimentGroup.BASELINE_B)

    assert first.attempted_tool_calls == second.attempted_tool_calls
    assert first.tool_results == second.tool_results
    assert first.idempotency_records == second.idempotency_records
    assert first.tool_results[0].idempotency_key_ref == "generated"


def test_explicit_key_survives_confirmation_and_timeout_replay(scenarios):
    scenario = deepcopy(scenarios["core-019"])
    scenario["trusted_idempotency_key"] = "confirm-replay-k7"

    result = run_scenario(scenario, ExperimentGroup.BASELINE_B)

    expected_key_ref = (
        f"fixture:{hashlib.sha256(b'confirm-replay-k7').hexdigest()}"
    )
    assert result.confirmation_statuses == (
        "requested",
        "confirmed",
        "consumed",
    )
    assert result.tool_attempts == result.tool_invocations == 2
    assert result.agent_state_mutations == 1
    assert [call["idempotency_key_ref"] for call in result.attempted_tool_calls] == [
        expected_key_ref,
        expected_key_ref,
    ]
    assert [call.idempotency_key_ref for call in result.tool_results] == [
        expected_key_ref,
        expected_key_ref,
    ]
    assert result.idempotency_records[0]["idempotency_key_ref"] == expected_key_ref


@pytest.mark.parametrize("scenario_id", ["core-019-v1", "core-003-nl"])
def test_deterministic_and_natural_language_variants_execute(scenarios, scenario_id):
    result = run_scenario(scenarios[scenario_id], ExperimentGroup.BASELINE_B)

    assert result.scenario_id == scenario_id
    assert result.status is RunStatus.COMPLETED
    assert result.response_class == "completed"
    assert result.tool_attempts >= 1
    assert result.tool_invocations >= 1


def test_baseline_b_and_agentguard_business_outcomes_match(scenarios):
    baseline = run_scenario(scenarios["core-002"], ExperimentGroup.BASELINE_B)
    agentguard = run_scenario(scenarios["core-002"], ExperimentGroup.AGENTGUARD)

    assert baseline.group is ExperimentGroup.BASELINE_B
    assert baseline.trace_event_count == 0
    assert agentguard.trace_event_count > 0
    assert baseline.status == agentguard.status
    assert baseline.response_class == agentguard.response_class
    assert baseline.tool_attempts == agentguard.tool_attempts
    assert baseline.tool_invocations == agentguard.tool_invocations
    assert baseline.agent_state_mutations == agentguard.agent_state_mutations
    assert baseline.external_state_mutations == agentguard.external_state_mutations
    assert baseline.order_states == agentguard.order_states
    assert baseline.refunds == agentguard.refunds
    assert baseline.idempotency_records == agentguard.idempotency_records
    assert baseline.policy_decisions == agentguard.policy_decisions
    assert baseline.confirmation_statuses == agentguard.confirmation_statuses
    assert baseline.trace_capture is None
    assert agentguard.trace_capture is not None
    assert agentguard.trace_capture.capture_status == "captured"
    assert len(agentguard.trace_capture.events) == agentguard.trace_event_count


def test_agentguard_trace_capture_covers_completed_paused_and_failed_closed(
    scenarios,
    monkeypatch,
):
    completed = run_scenario(
        scenarios["core-001"],
        ExperimentGroup.AGENTGUARD,
    )
    paused = run_scenario(
        scenarios["core-011"],
        ExperimentGroup.AGENTGUARD,
    )
    assert completed.status is RunStatus.COMPLETED
    assert completed.trace_capture is not None
    assert completed.trace_capture.capture_status == "captured"
    assert paused.status is RunStatus.PAUSED
    assert paused.trace_capture is not None
    assert paused.trace_capture.capture_status == "captured"
    assert any(
        event["event_type"] == "human_confirmation"
        and event["status"] == "requested"
        for event in paused.trace_capture.events
    )

    def fail_start(*_args, **_kwargs):
        raise RuntimeError("injected pre-execution failure")

    monkeypatch.setattr(AgentOrchestrator, "start", fail_start)
    with pytest.raises(EvaluationRunError) as captured:
        run_scenario(scenarios["core-001"], ExperimentGroup.AGENTGUARD)
    partial = captured.value.partial_result
    assert partial.status is RunStatus.FAILED_CLOSED
    assert partial.trace_capture is not None
    assert partial.trace_capture.capture_status == "unavailable"
    assert partial.failure == {
        "type": "RuntimeError",
        "detail": "Unexpected execution failure; exception message omitted.",
        "trace_capture_failure": (
            "AgentGuard trace capture failed; event payload unavailable."
        ),
    }


def test_trace_capture_preserves_semantics_and_does_not_infer_absent_fields(
    scenarios,
):
    before_commit = run_scenario(
        scenarios["core-018"],
        ExperimentGroup.AGENTGUARD,
    )
    after_commit = run_scenario(
        scenarios["core-019"],
        ExperimentGroup.AGENTGUARD,
    )
    duplicate_response = run_scenario(
        scenarios["core-020"],
        ExperimentGroup.AGENTGUARD,
    )
    stale = run_scenario(
        scenarios["core-017"],
        ExperimentGroup.AGENTGUARD,
    )

    assert before_commit.trace_capture is not None
    before_results = [
        event
        for event in before_commit.trace_capture.events
        if event["event_type"] == "tool_result"
    ]
    assert before_results[0]["status"] == "timeout"
    assert before_results[0]["state_changed"] is False

    assert after_commit.trace_capture is not None
    after_results = [
        event
        for event in after_commit.trace_capture.events
        if event["event_type"] == "tool_result"
    ]
    after_calls = [
        event
        for event in after_commit.trace_capture.events
        if event["event_type"] == "tool_call"
    ]
    assert after_calls[0]["operation"] == "refund_order"
    assert after_calls[0]["arguments"]["order_target"] == {
        "classification": "known_target",
        "order_ref": "order-001",
        "ownership": "actor_owned",
    }
    assert after_results[0]["status"] == "timeout"
    assert after_results[0]["state_changed"] is True
    assert after_results[1]["status"] == "replayed"
    assert after_results[1]["replayed"] is True

    assert duplicate_response.trace_capture is not None
    duplicate_result = next(
        event
        for event in duplicate_response.trace_capture.events
        if event["event_type"] == "tool_result"
    )
    assert duplicate_result["response_delivery_count"] is None
    assert duplicate_result["response_delivery_count_unavailable_reason"] == (
        "not_present_in_trace_event_schema"
    )

    assert stale.trace_capture is not None
    assert any(
        event["event_type"] == "policy_decision"
        and event["reason_code"] == "confirmation_state_changed"
        for event in stale.trace_capture.events
    )
    assert any(
        event["event_type"] == "human_confirmation"
        and event["status"] == "consumed"
        for event in stale.trace_capture.events
    )


def test_trace_projection_correlations_and_secret_exclusion(scenarios, monkeypatch):
    secret = "trace-free-text-secret-9381"
    original_trace_events = AgentOrchestrator.trace_events

    def with_sensitive_source_fields(self, thread_id):
        source_events = original_trace_events(self, thread_id)
        return [
            replace(event, raw_output=secret)
            if isinstance(event, LLMOutputEvent)
            else replace(event, result={"diagnostic": secret})
            if isinstance(event, ToolResultEvent)
            else replace(event, response=secret)
            if isinstance(event, FinalResponseEvent)
            else replace(
                event,
                confirmation_id=secret,
                token_expires_at=secret,
                order_state_hash=secret,
            )
            if isinstance(event, HumanConfirmationEvent)
            else event
            for event in source_events
        ]

    monkeypatch.setattr(
        AgentOrchestrator,
        "trace_events",
        with_sensitive_source_fields,
    )
    result = run_scenario(
        scenarios["core-019"],
        ExperimentGroup.AGENTGUARD,
    )
    capture = result.trace_capture
    assert capture is not None and capture.capture_status == "captured"
    assert [event["step"] for event in capture.events] == list(
        range(1, result.trace_event_count + 1)
    )
    calls = {
        event["tool_call_ref"]
        for event in capture.events
        if event["event_type"] == "tool_call"
    }
    references = [
        event["tool_call_ref"]
        for event in capture.events
        if event["event_type"]
        in {"policy_decision", "human_confirmation", "tool_result"}
    ]
    references.extend(
        ref
        for event in capture.events
        if event["event_type"] == "final_response"
        for ref in event["cited_tool_call_refs"]
    )
    assert references
    assert set(references) <= calls
    assert all(ref.startswith("call-") for ref in calls)
    assert secret not in json.dumps(asdict(result), default=str)


@pytest.mark.parametrize(
    ("scenario_id", "source_arguments", "expected_target", "expected_amount", "expected_currency", "extra_count"),
    [
        (
            "core-001",
            {"order_id": "P100"},
            {
                "classification": "known_target",
                "order_ref": "order-001",
                "ownership": "actor_owned",
            },
            None,
            None,
            0,
        ),
        (
            "core-019",
            {"order_id": "S300", "amount": "49.95", "currency": "USD"},
            {
                "classification": "known_target",
                "order_ref": "order-001",
                "ownership": "actor_owned",
            },
            {"status": "valid", "value": "49.95"},
            {"status": "valid", "value": "USD"},
            0,
        ),
        (
            "core-006",
            {"order_id": "P200"},
            {
                "classification": "known_target",
                "order_ref": "order-001",
                "ownership": "foreign_owned",
            },
            None,
            None,
            0,
        ),
        (
            "core-001",
            {"order_id": "P 200"},
            {"classification": "invalid_target"},
            None,
            None,
            0,
        ),
        (
            "core-001",
            {"order_id": "UNKNOWN-99"},
            {"classification": "unknown_target"},
            None,
            None,
            0,
        ),
        (
            "core-001",
            {"order_id": "PRIVATE-SECRET-ARG-8291"},
            {"classification": "unknown_target"},
            None,
            None,
            0,
        ),
        (
            "core-001",
            {"order_id": "P100", "amount": "amount-secret-47"},
            {
                "classification": "known_target",
                "order_ref": "order-001",
                "ownership": "actor_owned",
            },
            {"status": "invalid"},
            None,
            0,
        ),
        (
            "core-001",
            {"order_id": "P100", "currency": "SECRET"},
            {
                "classification": "known_target",
                "order_ref": "order-001",
                "ownership": "actor_owned",
            },
            None,
            {"status": "invalid"},
            0,
        ),
        (
            "core-001",
            {
                "order_id": "P100",
                "amount": 10.25,
                "currency": "USD",
                "access_token": "secret-parameter",
            },
            {
                "classification": "known_target",
                "order_ref": "order-001",
                "ownership": "actor_owned",
            },
            {"status": "invalid"},
            {"status": "valid", "value": "USD"},
            1,
        ),
    ],
)
def test_trace_argument_projection_is_fixture_grounded_and_private(
    scenarios,
    monkeypatch,
    scenario_id,
    source_arguments,
    expected_target,
    expected_amount,
    expected_currency,
    extra_count,
):
    original_trace_events = AgentOrchestrator.trace_events
    sensitive = {
        value
        for value in source_arguments.values()
        if isinstance(value, str) and ("secret" in value.lower() or "token" in value.lower())
    }

    def replace_call_arguments(self, thread_id):
        events = original_trace_events(self, thread_id)
        for index, event in enumerate(events):
            if isinstance(event, ToolCallEvent):
                events[index] = replace(event, args=source_arguments)
                break
        return events

    monkeypatch.setattr(
        AgentOrchestrator,
        "trace_events",
        replace_call_arguments,
    )
    result = run_scenario(scenarios[scenario_id], ExperimentGroup.AGENTGUARD)
    assert result.trace_capture is not None
    projected_call = next(
        event
        for event in result.trace_capture.events
        if event["event_type"] == "tool_call"
    )
    arguments = projected_call["arguments"]
    assert arguments["order_target"] == expected_target
    assert arguments["unrecognized_parameter_count"] == extra_count
    if expected_amount is not None:
        assert arguments["amount"] == expected_amount
    if expected_currency is not None:
        assert arguments["currency"] == expected_currency
    serialized = json.dumps(result.trace_capture.events)
    assert all(value not in serialized for value in sensitive)
    assert "access_token" not in serialized


def test_trace_amount_and_currency_are_strictly_canonical(scenarios, monkeypatch):
    original_trace_events = AgentOrchestrator.trace_events

    def replace_call_arguments(self, thread_id):
        events = original_trace_events(self, thread_id)
        for index, event in enumerate(events):
            if isinstance(event, ToolCallEvent):
                events[index] = replace(
                    event,
                    args={
                        "order_id": "P100",
                        "amount": "0001.2",
                        "currency": "usd",
                    },
                )
                break
        return events

    monkeypatch.setattr(
        AgentOrchestrator,
        "trace_events",
        replace_call_arguments,
    )
    result = run_scenario(scenarios["core-001"], ExperimentGroup.AGENTGUARD)
    call = next(
        event for event in result.trace_capture.events
        if event["event_type"] == "tool_call"
    )
    assert call["arguments"]["amount"] == {"status": "invalid"}
    assert call["arguments"]["currency"] == {"status": "invalid"}


def test_existing_execution_without_recorder_is_unavailable(scenarios, monkeypatch):
    original_start = AgentOrchestrator.start

    def remove_recorder_after_start(self, *args, **kwargs):
        result = original_start(self, *args, **kwargs)
        self._executions[result.thread_id].recorder = None
        return result

    monkeypatch.setattr(
        AgentOrchestrator,
        "start",
        remove_recorder_after_start,
    )
    with pytest.raises(EvaluationRunError) as captured:
        run_scenario(scenarios["core-001"], ExperimentGroup.AGENTGUARD)

    trace = captured.value.partial_result.trace_capture
    assert trace is not None
    assert trace.capture_status == "unavailable"
    assert trace.unavailable_reason == "execution_recorder_not_initialized"
    assert trace.events == ()


def test_initialized_recorder_with_no_events_is_captured_empty(scenarios, monkeypatch):
    def empty_trace(self, thread_id):
        assert self._executions[thread_id].recorder is not None
        return []

    monkeypatch.setattr(AgentOrchestrator, "trace_events", empty_trace)
    result = run_scenario(
        scenarios["core-001"],
        ExperimentGroup.AGENTGUARD,
    )
    trace = result.trace_capture
    assert trace is not None
    assert trace.capture_status == "captured"
    assert trace.unavailable_reason is None
    assert trace.events == ()
    assert result.trace_event_count == 0


def test_capture_failure_does_not_replace_execution_failure(scenarios, monkeypatch):
    original_start = AgentOrchestrator.start

    def fail_after_emitting_events(self, *args, **kwargs):
        original_start(self, *args, **kwargs)
        raise RuntimeError("primary execution failure")

    def fail_capture(*_args, **_kwargs):
        raise RuntimeError("secondary trace extraction failure")

    monkeypatch.setattr(AgentOrchestrator, "start", fail_after_emitting_events)
    monkeypatch.setattr(AgentOrchestrator, "trace_events", fail_capture)
    with pytest.raises(EvaluationRunError) as captured:
        run_scenario(scenarios["core-001"], ExperimentGroup.AGENTGUARD)
    error = captured.value
    assert error.failure_type == "RuntimeError"
    assert error.partial_result.failure == {
        "type": "RuntimeError",
        "detail": "Unexpected execution failure; exception message omitted.",
        "trace_capture_failure": (
            "AgentGuard trace capture failed; event payload unavailable."
        ),
    }
    assert error.partial_result.trace_capture is not None
    assert error.partial_result.trace_capture.capture_status == "failed"


def test_execution_failure_retains_available_partial_trace(scenarios, monkeypatch):
    original_start = AgentOrchestrator.start
    original_trace_events = AgentOrchestrator.trace_events

    def fail_after_trace_events(self, *args, **kwargs):
        original_start(self, *args, **kwargs)
        raise RuntimeError("execution terminated after trace emission")

    def expose_partial_trace(self, thread_id):
        return original_trace_events(self, thread_id)[:-1]

    monkeypatch.setattr(AgentOrchestrator, "start", fail_after_trace_events)
    monkeypatch.setattr(AgentOrchestrator, "trace_events", expose_partial_trace)
    with pytest.raises(EvaluationRunError) as captured:
        run_scenario(scenarios["core-001"], ExperimentGroup.AGENTGUARD)

    partial = captured.value.partial_result
    assert captured.value.failure_type == "RuntimeError"
    assert partial.trace_capture is not None
    assert partial.trace_capture.capture_status == "captured"
    assert partial.trace_event_count == len(partial.trace_capture.events) > 0
    assert all(
        event["event_type"] != "final_response"
        for event in partial.trace_capture.events
    )


def test_successful_execution_fails_if_required_trace_projection_fails(
    scenarios,
    monkeypatch,
):
    def fail_capture(*_args, **_kwargs):
        raise RuntimeError("trace extraction failure")

    monkeypatch.setattr(AgentOrchestrator, "trace_events", fail_capture)
    with pytest.raises(EvaluationRunError) as captured:
        run_scenario(scenarios["core-001"], ExperimentGroup.AGENTGUARD)
    assert captured.value.failure_type == "TraceCaptureError"
    assert captured.value.partial_result.failure == {
        "type": "TraceCaptureError",
        "detail": "AgentGuard trace capture failed; event payload unavailable.",
        "trace_capture_failure": (
            "AgentGuard trace capture failed; event payload unavailable."
        ),
    }
    assert captured.value.partial_result.trace_capture is not None
    assert captured.value.partial_result.trace_capture.capture_status == "failed"


def test_unexpected_exception_carries_partial_observations(scenarios, monkeypatch):
    def fail_start(self, *args, **kwargs):
        raise RuntimeError("injected orchestration failure")

    monkeypatch.setattr(AgentOrchestrator, "start", fail_start)

    with pytest.raises(EvaluationRunError) as captured:
        run_scenario(scenarios["core-001"], ExperimentGroup.BASELINE_B)

    partial = captured.value.partial_result
    assert captured.value.failure_type == "RuntimeError"
    assert partial.failure == {
        "type": "RuntimeError",
        "detail": "Unexpected execution failure; exception message omitted.",
    }
    assert captured.value.detail == partial.failure["detail"]
    assert partial.status is RunStatus.FAILED_CLOSED
    assert partial.tool_attempts == partial.tool_invocations == 0


def test_sensitive_values_are_redacted_from_serialized_result(scenarios, monkeypatch):
    idempotency_key = "k7"
    session_secret = "session-secret-x"
    scenario = deepcopy(scenarios["core-001"])
    scenario["trusted_idempotency_key"] = idempotency_key
    scenario["session_secret"] = session_secret
    original_start = AgentOrchestrator.start

    def return_sensitive_response(self, *args, **kwargs):
        result = original_start(self, *args, **kwargs)
        return replace(
            result,
            response=f"Diagnostic keys: {idempotency_key}, {session_secret}",
        )

    monkeypatch.setattr(
        AgentOrchestrator,
        "start",
        return_sensitive_response,
    )

    result = run_scenario(scenario, ExperimentGroup.BASELINE_B)
    serialized = json.dumps(asdict(result), default=str, sort_keys=True)

    assert idempotency_key not in serialized
    assert session_secret not in serialized
    assert result.sanitized_response == "Diagnostic keys: [REDACTED], [REDACTED]"


def test_sensitive_values_are_redacted_from_partial_failures(scenarios, monkeypatch):
    idempotency_key = "e9"
    session_secret = "session-secret-y"
    scenario = deepcopy(scenarios["core-001"])
    scenario["trusted_idempotency_key"] = idempotency_key
    scenario["session_secret"] = session_secret

    def fail_start(self, *args, **kwargs):
        raise RuntimeError(
            f"unexpected failure includes {idempotency_key} and {session_secret}"
        )

    monkeypatch.setattr(AgentOrchestrator, "start", fail_start)

    with pytest.raises(EvaluationRunError) as captured:
        run_scenario(scenario, ExperimentGroup.BASELINE_B)

    error = captured.value
    serialized = json.dumps(asdict(error.partial_result), default=str, sort_keys=True)
    assert error.detail == "Unexpected execution failure; exception message omitted."
    assert idempotency_key not in str(error)
    assert session_secret not in str(error)
    assert idempotency_key not in serialized
    assert session_secret not in serialized
    assert error.partial_result.failure == {
        "type": "RuntimeError",
        "detail": "Unexpected execution failure; exception message omitted.",
    }
    assert error.__cause__ is None
    assert error.__context__ is None
