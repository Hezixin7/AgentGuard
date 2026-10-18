from dataclasses import replace
import json

import pytest

from evaluation.localization import localize_run
from evaluation.offline_batch import compose_stage2c_dataset, run_batch


@pytest.fixture(scope="module")
def localization_inputs():
    dataset = compose_stage2c_dataset()
    batch = run_batch()
    scenarios = {scenario["id"]: scenario for scenario in dataset.all_scenarios}
    records = {
        (record.scenario_id, record.group): record
        for record in batch.records
    }
    return scenarios, records


def _localize(localization_inputs, scenario_id, group="agentguard", record=None):
    scenarios, records = localization_inputs
    return localize_run(
        scenarios[scenario_id],
        record or records[(scenario_id, group)],
    )


def _replace_record(localization_inputs, scenario_id, group, **observation_changes):
    _, records = localization_inputs
    record = records[(scenario_id, group)]
    assert record.observations is not None
    observations = replace(record.observations, **observation_changes)
    return replace(record, observations=observations)


def _trace_events(localization_inputs, scenario_id, group="agentguard"):
    _, records = localization_inputs
    record = records[(scenario_id, group)]
    assert record.observations is not None
    capture = record.observations.trace_capture
    assert capture is not None
    return [dict(event) for event in capture.events]


def _with_events(localization_inputs, scenario_id, events, group="agentguard"):
    _, records = localization_inputs
    record = records[(scenario_id, group)]
    assert record.observations is not None
    capture = record.observations.trace_capture
    assert capture is not None
    return replace(
        record,
        observations=replace(
            record.observations,
            trace_capture=replace(capture, events=tuple(events)),
            trace_event_count=len(events),
        ),
    )


def _renumber(events):
    return [dict(event, step=index + 1) for index, event in enumerate(events)]


def test_cross_user_block_and_prohibited_execution_are_distinguished(
    localization_inputs,
):
    guarded = _localize(localization_inputs, "core-006")
    assert guarded["status"] == "no_divergence"
    assert "policy_outcomes[0]" in guarded["matched_predicates"]
    assert "confirmation_statuses:expected_absent" in (
        guarded["matched_predicates"]
    )

    baseline = _localize(
        localization_inputs,
        "core-006",
        group="baseline_a",
    )
    assert baseline["status"] == "localized"
    assert baseline["finding"]["predicate_id"] == "prohibited_tool_execution"
    assert baseline["finding"]["source_ref"] == {
        "source": "private_runner_observations",
        "field": "tool_results",
        "index": 0,
    }


def test_approved_refund_and_expected_confirmation_hold_are_not_divergences(
    localization_inputs,
):
    refund = _localize(localization_inputs, "core-003")
    assert refund["status"] == "no_divergence"

    refusal = _localize(localization_inputs, "core-004")
    assert refusal["status"] == "no_divergence"
    assert "post_rejection_terminal_block" in (
        refusal["matched_predicates"]
    )

    hold = _localize(localization_inputs, "core-015")
    assert hold["status"] == "no_divergence"
    assert "tool_invocation_count" in hold["matched_predicates"]


def test_missing_confirmation_is_localized_from_private_observation(
    localization_inputs,
):
    altered = _replace_record(
        localization_inputs,
        "core-002",
        "baseline_b",
        confirmation_statuses=(),
    )
    result = _localize(
        localization_inputs,
        "core-002",
        group="baseline_b",
        record=altered,
    )
    assert result["status"] == "localized"
    finding = result["finding"]
    assert finding["predicate_id"] == "confirmation_statuses[0]"
    assert finding["expected_predicate"] == "requested"
    assert finding["observed_semantics"] is None
    assert finding["source_ref"]["field"] == "confirmation_statuses"


def test_expected_timeout_before_commit_and_timeout_after_commit_replay(
    localization_inputs,
):
    before_commit = _localize(localization_inputs, "core-018")
    assert before_commit["status"] == "no_divergence"
    assert "fault_outcome[0]" in before_commit["matched_predicates"]

    replay = _localize(localization_inputs, "core-019")
    assert replay["status"] == "no_divergence"
    assert "fault_outcome[0]" in replay["matched_predicates"]
    assert "fault_outcome[1]" in replay["matched_predicates"]


def test_stale_confirmation_uses_separate_external_change_observation(
    localization_inputs,
):
    result = _localize(localization_inputs, "core-017")
    assert result["status"] == "no_divergence"
    assert "external_state_mutations" in result["matched_predicates"]
    assert "stale_confirmation_reason_code" in result["matched_predicates"]
    assert result["source_scope"] == "agentguard_trace_and_runner_observations"
    evidence = result["observed_evidence"]
    assert evidence["external_state_mutations"]["source"] == (
        "private_runner_observations"
    )
    assert evidence["external_state_mutations"]["observed"] == 1
    # The external transition is fixture setup; it is not claimed as a trace
    # event produced by the AgentGuard run.
    assert evidence["external_transition_trace_event"] == "unavailable"
    assert "confirmation_state_changed" in evidence["policy_reason_codes"]
    assert result["comparison_provenance"]["fixture_context_used_as_gold"] is False


def test_final_state_difference_alone_is_unobservable(localization_inputs):
    original = _localize(localization_inputs, "core-003")
    assert original["status"] == "no_divergence"
    _, records = localization_inputs
    record = records[("core-003", "agentguard")]
    assert record.observations is not None
    altered_orders = {
        order_id: dict(order)
        for order_id, order in record.observations.order_states.items()
    }
    altered_orders["S300"]["status"] = "shipped"
    altered = replace(
        record,
        observations=replace(record.observations, order_states=altered_orders),
    )

    result = _localize(
        localization_inputs,
        "core-003",
        record=altered,
    )
    assert result["status"] == "unobservable"
    assert result["finding"] is None
    assert any(
        reason.startswith(
            "authoritative_order_state_differs_without_observable_causal_event"
        )
        for reason in result["unobservability_reasons"]
    )


def test_missing_or_inconsistent_agentguard_trace_is_unobservable(
    localization_inputs,
):
    missing = _replace_record(
        localization_inputs,
        "core-006",
        "agentguard",
        trace_capture=None,
    )
    missing_result = _localize(
        localization_inputs,
        "core-006",
        record=missing,
    )
    assert missing_result["status"] == "unobservable"

    _, records = localization_inputs
    record = records[("core-006", "agentguard")]
    assert record.observations is not None
    capture = record.observations.trace_capture
    assert capture is not None
    incomplete = replace(capture, events=capture.events[:-1])
    inconsistent_record = replace(
        record,
        observations=replace(record.observations, trace_capture=incomplete),
    )
    inconsistent = _localize(
        localization_inputs,
        "core-006",
        record=inconsistent_record,
    )
    assert inconsistent["status"] == "unobservable"
    assert "agentguard_trace_capture_is_incomplete_or_inconsistent" in (
        inconsistent["unobservability_reasons"]
    )


def test_baseline_trace_payload_is_rejected_as_evidence(localization_inputs):
    _, records = localization_inputs
    baseline = records[("core-003", "baseline_b")]
    agentguard = records[("core-003", "agentguard")]
    assert baseline.observations is not None
    assert agentguard.observations is not None
    altered = replace(
        baseline,
        observations=replace(
            baseline.observations,
            trace_capture=agentguard.observations.trace_capture,
        ),
    )
    result = _localize(
        localization_inputs,
        "core-003",
        group="baseline_b",
        record=altered,
    )
    assert result["status"] == "unobservable"
    assert result["unobservability_reasons"] == [
        "baseline_trace_payload_is_not_permitted_evidence"
    ]


def test_localization_output_is_deterministic_and_schema_bounded(
    localization_inputs,
):
    first = _localize(localization_inputs, "core-019")
    second = _localize(localization_inputs, "core-019")
    assert first == second
    assert json.dumps(first, sort_keys=True) == json.dumps(second, sort_keys=True)
    assert first["status"] in {"localized", "no_divergence", "unobservable"}
    assert set(first) == {
        "localization_schema_version",
        "scenario_id",
        "group",
        "status",
        "source_scope",
        "comparison_provenance",
        "finding",
        "matched_predicates",
        "unobservability_reasons",
        "causal_assessment",
        "observed_evidence",
    }
    assert first["comparison_provenance"]["fixture_context_used_as_gold"] is False


def test_all_batch_records_use_only_the_three_supported_statuses(
    localization_inputs,
):
    scenarios, records = localization_inputs
    statuses = {
        localize_run(scenarios[record.scenario_id], record)["status"]
        for record in records.values()
    }
    assert len(records) == 300
    assert statuses <= {"localized", "no_divergence", "unobservable"}


def test_altered_trace_chronology_and_step_gaps_are_unobservable(
    localization_inputs,
):
    events = _trace_events(localization_inputs, "core-003")
    events[2], events[3] = events[3], events[2]
    reordered = _localize(
        localization_inputs,
        "core-003",
        record=_with_events(localization_inputs, "core-003", events),
    )
    assert reordered["status"] == "unobservable"
    assert "agentguard_trace_capture_is_incomplete_or_inconsistent" in (
        reordered["unobservability_reasons"]
    )

    gap_events = _trace_events(localization_inputs, "core-003")
    gap_events[2] = dict(gap_events[2], step=99)
    gap = _localize(
        localization_inputs,
        "core-003",
        record=_with_events(localization_inputs, "core-003", gap_events),
    )
    assert gap["status"] == "unobservable"
    assert "agentguard_trace_capture_is_incomplete_or_inconsistent" in (
        gap["unobservability_reasons"]
    )


def test_missing_intermediate_trace_events_are_unobservable(
    localization_inputs,
):
    dropped_policy = [
        event
        for event in _trace_events(localization_inputs, "core-003")
        if not (
            event["event_type"] == "policy_decision"
            and event.get("decision") == "blocked"
        )
    ]
    policy_result = _localize(
        localization_inputs,
        "core-003",
        record=_with_events(
            localization_inputs,
            "core-003",
            _renumber(dropped_policy),
        ),
    )
    assert policy_result["status"] == "unobservable"
    assert any(
        reason.startswith("policy_outcomes[1]_not_observed")
        for reason in policy_result["unobservability_reasons"]
    )

    dropped_call = [
        event
        for event in _trace_events(localization_inputs, "core-003")
        if event["event_type"] != "tool_call"
    ]
    call_result = _localize(
        localization_inputs,
        "core-003",
        record=_with_events(
            localization_inputs,
            "core-003",
            _renumber(dropped_call),
        ),
    )
    assert call_result["status"] == "unobservable"
    assert any(
        "call_reference_is_not_to_a_preceding_call" in reason
        for reason in call_result["unobservability_reasons"]
    )


def test_mismatched_tool_call_references_are_unobservable(
    localization_inputs,
):
    operation_events = _trace_events(localization_inputs, "core-003")
    for event in operation_events:
        if event["event_type"] == "policy_decision":
            event["operation"] = "cancel_order"
            break
    operation_result = _localize(
        localization_inputs,
        "core-003",
        record=_with_events(
            localization_inputs,
            "core-003",
            operation_events,
        ),
    )
    assert operation_result["status"] == "unobservable"
    assert (
        "policy_decision_operation_does_not_match_referenced_call"
        in operation_result["unobservability_reasons"]
    )

    citation_events = _trace_events(localization_inputs, "core-003")
    for event in citation_events:
        if event["event_type"] == "final_response":
            event["cited_tool_call_refs"] = ["call-999"]
    citation_result = _localize(
        localization_inputs,
        "core-003",
        record=_with_events(
            localization_inputs,
            "core-003",
            citation_events,
        ),
    )
    assert citation_result["status"] == "unobservable"
    assert (
        "final_response_cites_unknown_tool_call"
        in citation_result["unobservability_reasons"]
    )

    duplicate_events = _trace_events(localization_inputs, "core-003")
    first_call = next(
        event
        for event in duplicate_events
        if event["event_type"] == "tool_call"
    )
    duplicate_events.append(dict(first_call, step=len(duplicate_events) + 1))
    duplicate_result = _localize(
        localization_inputs,
        "core-003",
        record=_with_events(
            localization_inputs,
            "core-003",
            duplicate_events,
        ),
    )
    assert duplicate_result["status"] == "unobservable"
    assert "tool_call_identity_is_missing_or_ambiguous" in (
        duplicate_result["unobservability_reasons"]
    )


def test_tool_result_lifecycle_references_are_validated(
    localization_inputs,
):
    unauthorized_events = _trace_events(localization_inputs, "core-003")
    for event in unauthorized_events:
        if event["event_type"] == "policy_decision" and event.get(
            "decision"
        ) == "allowed":
            event["decision"] = "blocked"
    unauthorized = _localize(
        localization_inputs,
        "core-003",
        record=_with_events(
            localization_inputs,
            "core-003",
            unauthorized_events,
        ),
    )
    assert unauthorized["status"] == "unobservable"
    assert (
        "tool_result_has_no_preceding_authorizing_policy_decision"
        in unauthorized["unobservability_reasons"]
    )

    duplicate_result_events = _trace_events(localization_inputs, "core-003")
    first_result = next(
        event
        for event in duplicate_result_events
        if event["event_type"] == "tool_result"
    )
    duplicate_result_events.append(
        dict(first_result, step=len(duplicate_result_events) + 1)
    )
    duplicate = _localize(
        localization_inputs,
        "core-003",
        record=_with_events(
            localization_inputs,
            "core-003",
            duplicate_result_events,
        ),
    )
    assert duplicate["status"] == "unobservable"
    assert "tool_result_duplicates_referenced_call" in (
        duplicate["unobservability_reasons"]
    )

    events_without_results = [
        event
        for event in _trace_events(localization_inputs, "core-019")
        if event["event_type"] != "tool_result"
    ]
    result_events = [
        event
        for event in _trace_events(localization_inputs, "core-019")
        if event["event_type"] == "tool_result"
    ]
    last_policy_index = max(
        index
        for index, event in enumerate(events_without_results)
        if event["event_type"] == "policy_decision"
    )
    # Both calls are authorized before either result; only the result order is
    # inverted (call ordinal 1 then call ordinal 0).
    non_monotonic_events = _renumber(
        events_without_results[: last_policy_index + 1]
        + [result_events[1], result_events[0]]
        + events_without_results[last_policy_index + 1 :]
    )
    non_monotonic_result = _localize(
        localization_inputs,
        "core-019",
        record=_with_events(
            localization_inputs,
            "core-019",
            non_monotonic_events,
        ),
    )
    assert non_monotonic_result["status"] == "unobservable"
    assert "tool_result_call_order_is_not_monotonic" in (
        non_monotonic_result["unobservability_reasons"]
    )


def test_timeout_mutation_contradictions_are_localized(
    localization_inputs,
):
    uncommitted_events = _trace_events(localization_inputs, "core-018")
    for event in uncommitted_events:
        if event["event_type"] == "tool_result":
            event["state_changed"] = True
            break
    uncommitted = _localize(
        localization_inputs,
        "core-018",
        record=_with_events(
            localization_inputs,
            "core-018",
            uncommitted_events,
        ),
    )
    assert uncommitted["status"] == "localized"
    assert uncommitted["finding"]["predicate_id"] == "fault_outcome[0]"
    assert uncommitted["finding"]["earliest_causal_divergence"] == "proven"

    committed_events = _trace_events(localization_inputs, "core-019")
    for event in committed_events:
        if event["event_type"] == "tool_result":
            event["state_changed"] = False
            break
    committed = _localize(
        localization_inputs,
        "core-019",
        record=_with_events(
            localization_inputs,
            "core-019",
            committed_events,
        ),
    )
    assert committed["status"] == "localized"
    assert committed["finding"]["predicate_id"] == "fault_outcome[0]"

    replay_events = _trace_events(localization_inputs, "core-019")
    results = [
        event
        for event in replay_events
        if event["event_type"] == "tool_result"
    ]
    results[1]["status"] = "timeout"
    results[1]["replayed"] = False
    results[1]["error_type"] = "timeout_after_commit"
    replay = _localize(
        localization_inputs,
        "core-019",
        record=_with_events(
            localization_inputs,
            "core-019",
            replay_events,
        ),
    )
    assert replay["status"] == "localized"
    assert replay["finding"]["predicate_id"] == "fault_outcome[1]"


def test_timeout_semantics_are_cross_checked_with_runner_observations(
    localization_inputs,
):
    _, records = localization_inputs
    record = records[("core-018", "agentguard")]
    assert record.observations is not None
    tool_results = list(record.observations.tool_results)
    tool_results[0] = replace(tool_results[0], committed=True)
    altered = replace(
        record,
        observations=replace(
            record.observations,
            tool_results=tuple(tool_results),
        ),
    )
    result = _localize(
        localization_inputs,
        "core-018",
        record=altered,
    )
    assert result["status"] == "localized"
    assert result["finding"]["predicate_id"] == (
        "trace_runner_tool_result_conflict"
    )
    assert result["finding"]["earliest_causal_divergence"] == "unproven"
    assert result["finding"]["observed_semantics"]["runner_semantics"][
        "state_changed"
    ] is True


def test_duplicate_response_is_not_reported_as_a_trace_conflict(
    localization_inputs,
):
    result = _localize(localization_inputs, "core-020")
    assert result["status"] == "no_divergence"
    assert "duplicate_response_delivery_count" in result["matched_predicates"]


def test_baseline_b_compares_policy_and_confirmation_observations(
    localization_inputs,
):
    altered = _replace_record(
        localization_inputs,
        "core-002",
        "baseline_b",
        policy_decisions=(
            {"outcome": "blocked", "reason_code": "confirmation_required"},
            {"outcome": "blocked", "reason_code": "confirmation_required"},
        ),
    )
    result = _localize(
        localization_inputs,
        "core-002",
        group="baseline_b",
        record=altered,
    )
    assert result["status"] == "localized"
    assert result["finding"]["predicate_id"] == "policy_outcomes[1]"
    assert result["finding"]["expected_predicate"] == "allowed"
    assert result["finding"]["observed_semantics"] == "blocked"
    assert result["finding"]["earliest_causal_divergence"] == "unproven"
    assert (
        "private_observations_lack_cross_event_ordering"
        in result["finding"]["unresolved_prior_predicates"]
    )
    assert result["source_scope"] == "private_runner_observations"


def test_baseline_b_tolerates_documented_post_rejection_block(
    localization_inputs,
):
    result = _localize(
        localization_inputs,
        "core-004",
        group="baseline_b",
    )
    assert result["status"] == "no_divergence"
    assert "post_rejection_terminal_block" in result["matched_predicates"]


def test_baseline_a_retains_explicit_lifecycle_observability_limits(
    localization_inputs,
):
    result = _localize(
        localization_inputs,
        "core-002",
        group="baseline_a",
    )
    assert result["status"] == "unobservable"
    assert "policy_outcomes_not_instrumented_for_baseline_a" in (
        result["unobservability_reasons"]
    )
    assert "confirmation_statuses_not_instrumented_for_baseline_a" in (
        result["unobservability_reasons"]
    )
    assert result["causal_assessment"]["earliest_causal_divergence"] == "unproven"


def test_external_state_mutation_requires_runner_observation_consistency(
    localization_inputs,
):
    altered = _replace_record(
        localization_inputs,
        "core-017",
        "agentguard",
        external_state_mutations=0,
    )
    result = _localize(
        localization_inputs,
        "core-017",
        record=altered,
    )
    assert result["status"] == "unobservable"
    assert (
        "external_state_mutation_count_inconsistent_with_fixture_context"
        in result["unobservability_reasons"]
    )


def test_fixture_proposals_are_never_promoted_to_oracle_labels(
    localization_inputs,
):
    scenarios, records = localization_inputs
    baseline = _localize(localization_inputs, "core-003")
    altered_scenario = dict(scenarios["core-003"])
    altered_scenario["proposals"] = [
        {
            "type": "tool_call",
            "operation": "cancel_order",
            "order_id": "P100",
        }
    ]
    altered = localize_run(
        altered_scenario,
        records[("core-003", "agentguard")],
    )
    assert altered == baseline
    assert baseline["comparison_provenance"] == {
        "expected_behavior": "scenario.expected",
        "fixture_context": [
            "scenario.proposals",
            "scenario.pre_resume_state_changes",
        ],
        "fixture_context_used_as_gold": False,
    }


def test_agentguard_runner_policy_contradiction_is_unproven(
    localization_inputs,
):
    altered = _replace_record(
        localization_inputs,
        "core-002",
        "agentguard",
        policy_decisions=(
            {"outcome": "allowed", "reason_code": "authorized"},
            {"outcome": "allowed", "reason_code": "authorized"},
        ),
    )
    result = _localize(
        localization_inputs,
        "core-002",
        record=altered,
    )
    assert result["status"] == "localized"
    finding = result["finding"]
    assert finding["predicate_id"] == "trace_runner_policy_conflict[0]"
    assert finding["earliest_causal_divergence"] == "unproven"
    assert "observation_conflict_does_not_establish_causal_origin" in (
        finding["unresolved_prior_predicates"]
    )
    assert finding["expected_predicate"]["trace_observation"] == "blocked"
    assert finding["observed_semantics"]["runner_observation"] == "allowed"
    assert finding["observed_semantics"]["runner_source"] == (
        "private_runner_observations"
    )
    assert finding["observed_semantics"]["trace_source"] == "agentguard_trace"
    assert finding["source_ref"]["source"] == "agentguard_trace"

    # Control: the unaltered record still has no cross-source conflict.
    control = _localize(localization_inputs, "core-002")
    assert control["status"] == "no_divergence"


def test_agentguard_runner_extra_policy_observation_is_unproven(
    localization_inputs,
):
    record = _replace_record(
        localization_inputs,
        "core-002",
        "agentguard",
    )
    assert record.observations is not None
    altered = _replace_record(
        localization_inputs,
        "core-002",
        "agentguard",
        policy_decisions=(
            *record.observations.policy_decisions,
            {"outcome": "blocked", "reason_code": "confirmation_required"},
        ),
    )
    result = _localize(
        localization_inputs,
        "core-002",
        record=altered,
    )
    assert result["status"] == "localized"
    finding = result["finding"]
    assert finding["predicate_id"] == "trace_runner_policy_conflict[2]"
    assert finding["earliest_causal_divergence"] == "unproven"
    assert finding["expected_predicate"]["trace_observation"] is None
    assert finding["observed_semantics"]["runner_observation"] == "blocked"
    # Runner-only evidence carries no invented trace call reference.
    assert finding["source_ref"] == {
        "source": "private_runner_observations",
        "field": "policy_decisions",
        "index": 2,
    }


def test_agentguard_runner_missing_confirmation_is_unproven(
    localization_inputs,
):
    altered = _replace_record(
        localization_inputs,
        "core-002",
        "agentguard",
        confirmation_statuses=("requested", "confirmed"),
    )
    result = _localize(
        localization_inputs,
        "core-002",
        record=altered,
    )
    assert result["status"] == "localized"
    finding = result["finding"]
    assert finding["predicate_id"] == (
        "trace_runner_confirmation_conflict[2]"
    )
    assert finding["earliest_causal_divergence"] == "unproven"
    assert finding["expected_predicate"]["trace_observation"] == "consumed"
    assert finding["observed_semantics"]["runner_observation"] is None

    control = _localize(localization_inputs, "core-002")
    assert control["status"] == "no_divergence"


def test_baseline_b_observable_oracle_mismatches_are_localized(
    localization_inputs,
):
    response = _localize(
        localization_inputs,
        "core-002",
        group="baseline_b",
        record=_replace_record(
            localization_inputs,
            "core-002",
            "baseline_b",
            response_class="denied",
        ),
    )
    assert response["status"] == "localized"
    assert response["finding"]["predicate_id"] == "final_response_class"
    assert response["finding"]["earliest_causal_divergence"] == "unproven"

    order = _localize(
        localization_inputs,
        "core-002",
        group="baseline_b",
        record=_replace_record(
            localization_inputs,
            "core-002",
            "baseline_b",
            order_states={"P100": {"status": "shipped"}},
        ),
    )
    assert order["status"] == "localized"
    assert order["finding"]["predicate_id"].startswith(
        "authoritative_order_state_differs_without_observable_causal_event"
    )
    assert order["finding"]["earliest_causal_divergence"] == "unproven"

    mutations = _localize(
        localization_inputs,
        "core-002",
        group="baseline_b",
        record=_replace_record(
            localization_inputs,
            "core-002",
            "baseline_b",
            agent_state_mutations=99,
        ),
    )
    assert mutations["status"] == "localized"
    assert mutations["finding"]["predicate_id"] == (
        "agent_mutation_count_differs_without_observable_causal_event"
    )

    refunds = _localize(
        localization_inputs,
        "core-002",
        group="baseline_b",
        record=_replace_record(
            localization_inputs,
            "core-002",
            "baseline_b",
            refunds=({"order_id": "P100"},),
        ),
    )
    assert refunds["status"] == "localized"
    assert refunds["finding"]["predicate_id"] == (
        "refund_count_differs_without_observable_causal_event"
    )

    idempotency = _localize(
        localization_inputs,
        "core-003",
        group="baseline_b",
        record=_replace_record(
            localization_inputs,
            "core-003",
            "baseline_b",
            idempotency_records=(),
        ),
    )
    assert idempotency["status"] == "localized"
    assert idempotency["finding"]["predicate_id"] == (
        "idempotency_count_differs_without_observable_causal_event"
    )

    # Control: all observable oracle predicates match for the frozen record.
    control = _localize(
        localization_inputs,
        "core-002",
        group="baseline_b",
    )
    assert control["status"] == "no_divergence"
    for predicate in (
        "final_response_class",
        "agent_state_mutations",
        "order_states",
        "refund_count",
        "idempotency_record_count",
    ):
        assert predicate in control["matched_predicates"]


def test_localized_earlier_evidence_excludes_later_events(
    localization_inputs,
):
    events = _trace_events(localization_inputs, "core-019")
    for event in events:
        if event["event_type"] == "tool_result":
            event["state_changed"] = False
            break
    result = _localize(
        localization_inputs,
        "core-019",
        record=_with_events(localization_inputs, "core-019", events),
    )
    assert result["status"] == "localized"
    finding = result["finding"]
    assert finding["predicate_id"] == "fault_outcome[0]"
    assert finding["source_ref"]["step"] == 8

    earlier = finding["earlier_required_predicates_resolved"]
    assert "tool_attempt[0]" in earlier
    assert "policy_outcomes[1]" in earlier
    for later_predicate in (
        "tool_attempt[1]",
        "policy_outcomes[2]",
        "fault_outcome[1]",
        "final_response_references_valid",
        "tool_attempt_count",
        "tool_invocation_count",
    ):
        assert later_predicate not in earlier

    # matched_predicates remains the complete set of resolved predicates.
    assert "policy_outcomes[2]" in result["matched_predicates"]
    assert "fault_outcome[1]" in result["matched_predicates"]
    assert "final_response_references_valid" in result["matched_predicates"]

    # Control: the unaltered record has no divergence and no finding.
    control = _localize(localization_inputs, "core-019")
    assert control["status"] == "no_divergence"
    assert control["finding"] is None
