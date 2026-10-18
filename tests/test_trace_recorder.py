import json

import pytest

from trace_events import (
    EventType,
    FinalResponseEvent,
    HumanConfirmationEvent,
    LLMOutputEvent,
    PolicyDecisionEvent,
    PolicyOutcome,
    ToolCallEvent,
    ToolResultEvent,
    ToolStatus,
    ConfirmationStatus,
)
from trace_recorder import TraceRecorder
from unittest.mock import patch


def _read_export(tmp_path, recorder):
    path = tmp_path / "trace.json"
    recorder.export_json(path)
    return json.loads(path.read_text(encoding="utf-8"))


def test_records_sequential_events_and_exports_event_specific_fields(tmp_path):
    recorder = TraceRecorder(
        scenario_id="refund-happy-path",
        run_id="batch-1",
        trace_id="trace-1",
    )
    events = [
        LLMOutputEvent(raw_output="plan", parsed_plan={"action": "refund"}),
        ToolCallEvent(tool_call_id="call-1", tool_name="refund_order"),
        PolicyDecisionEvent(
            tool_call_id="call-1",
            tool_name="refund_order",
            decision=PolicyOutcome.ALLOWED,
            reason_code="authorized",
        ),
        HumanConfirmationEvent(
            tool_call_id="call-1",
            tool_name="refund_order",
            status=ConfirmationStatus.CONFIRMED,
        ),
        ToolResultEvent(
            tool_call_id="call-1",
            tool_name="refund_order",
            status=ToolStatus.SUCCESS,
            result={"refund_id": "r-1"},
            state_changed=True,
        ),
        FinalResponseEvent(response="Refund complete", cited_tool_calls=["call-1"]),
    ]
    for event in events:
        recorder.record(event)

    exported = _read_export(tmp_path, recorder)
    assert exported["trace"] == {
        "trace_id": "trace-1",
        "run_id": "batch-1",
        "scenario_id": "refund-happy-path",
    }
    assert [event["step"] for event in exported["events"]] == list(range(1, 7))
    assert {
        event["trace_id"] for event in exported["events"]
    } == {"trace-1"}
    assert {event["run_id"] for event in exported["events"]} == {"batch-1"}
    assert exported["events"][2]["decision"] == "allowed"
    assert exported["events"][3]["status"] == "confirmed"
    assert exported["events"][4]["state_changed"] is True
    assert exported["events"][5]["cited_tool_calls"] == ["call-1"]
    assert [event["event_type"] for event in exported["events"]] == [
        EventType.LLM_OUTPUT.value,
        EventType.TOOL_CALL.value,
        EventType.POLICY_DECISION.value,
        EventType.HUMAN_CONFIRMATION.value,
        EventType.TOOL_RESULT.value,
        EventType.FINAL_RESPONSE.value,
    ]
    assert all(
        event["tool_call_id"] == "call-1"
        for event in exported["events"][1:5]
    )


def test_assigns_trace_id_and_rejects_conflicting_metadata():
    recorder = TraceRecorder(scenario_id="scenario-1", run_id="run-1")
    recorded = recorder.record(LLMOutputEvent())

    assert recorded.trace_id == recorder.trace_id
    assert recorded.step == 1
    assert recorded.run_id == "run-1"
    with pytest.raises(ValueError, match="trace_id"):
        recorder.record(LLMOutputEvent(trace_id="another-trace"))
    with pytest.raises(ValueError, match="run_id"):
        recorder.record(LLMOutputEvent(run_id="another-run"))


def test_record_rejects_non_event_input():
    recorder = TraceRecorder(scenario_id="scenario-1", run_id="run-1")

    with pytest.raises(TypeError, match="TraceEvent"):
        recorder.record({"not": "an event"})


def test_requires_unique_call_ids_and_known_correlations():
    recorder = TraceRecorder(scenario_id="scenario-1", run_id="run-1")
    with pytest.raises(ValueError, match="unknown tool_call_id"):
        recorder.record(ToolResultEvent(tool_call_id="missing"))

    recorder.record(ToolCallEvent(tool_call_id="call-1"))
    with pytest.raises(ValueError, match="unique"):
        recorder.record(ToolCallEvent(tool_call_id="call-1"))
    with pytest.raises(ValueError, match="unknown tool_call_id"):
        recorder.record(PolicyDecisionEvent(tool_call_id="call-2"))


def test_records_timeout_after_commit_then_replay_without_extra_state_change(tmp_path):
    recorder = TraceRecorder(scenario_id="retry-after-timeout", run_id="run-1")
    recorder.record(ToolCallEvent(tool_call_id="call-timeout", tool_name="refund"))
    recorder.record(
        PolicyDecisionEvent(
            tool_call_id="call-timeout",
            decision=PolicyOutcome.ALLOWED,
        )
    )
    recorder.record(
        ToolResultEvent(
            tool_call_id="call-timeout",
            status=ToolStatus.TIMEOUT,
            error_type="TimeoutError",
            idempotency_key_ref="hash-1",
            state_changed=True,
        )
    )
    recorder.record(ToolCallEvent(tool_call_id="call-retry", tool_name="refund"))
    recorder.record(
        PolicyDecisionEvent(
            tool_call_id="call-retry",
            decision=PolicyOutcome.REPLAY,
            reason_code="idempotent_replay",
        )
    )
    recorder.record(
        ToolResultEvent(
            tool_call_id="call-retry",
            status=ToolStatus.REPLAYED,
            idempotency_key_ref="hash-1",
            replayed=True,
            state_changed=False,
        )
    )

    results = [
        event for event in _read_export(tmp_path, recorder)["events"]
        if event["event_type"] == EventType.TOOL_RESULT.value
    ]
    assert [(event["status"], event["state_changed"]) for event in results] == [
        ("timeout", True),
        ("replayed", False),
    ]
    assert results[1]["replayed"] is True
    assert results[0]["idempotency_key_ref"] == results[1]["idempotency_key_ref"]
    assert results[0]["tool_call_id"] != results[1]["tool_call_id"]


def test_serializes_blocked_policy_outcome(tmp_path):
    recorder = TraceRecorder(scenario_id="blocked-request", run_id="run-1")
    recorder.record(ToolCallEvent(tool_call_id="call-1"))
    recorder.record(
        PolicyDecisionEvent(
            tool_call_id="call-1",
            decision=PolicyOutcome.BLOCKED,
            reason_code="not_authorized",
        )
    )

    exported = _read_export(tmp_path, recorder)

    assert exported["events"][1]["decision"] == "blocked"


def test_redacts_sensitive_fields_recursively_and_preserves_audit_fields(tmp_path):
    recorder = TraceRecorder(scenario_id="sensitive-data", run_id="run-1")
    recorder.record(
        ToolCallEvent(
            tool_call_id="call-1",
            tool_name="get_order",
            args={
                "order_id": "B456",
                "customer": {
                    "email": "person@example.test",
                    "name": "Sensitive Person",
                },
                "customer_name": "张三",
                "billing_address": "Example Street 12",
                "shipping_address": "Other Street 34",
                "contact_email": "contact@example.test",
                "customer_phone": "+1-555-0199",
                "access_token": "raw-access-token",
                "authentication": "raw-authentication-secret",
                "items": [
                    {"phone": "+1-555-0100"},
                    {"sku": "item-1", "status": "packed"},
                ],
                "confirmation_token": "raw-confirmation-capability",
            },
        )
    )
    recorder.record(
        PolicyDecisionEvent(
            tool_call_id="call-1",
            tool_name="get_order",
            reason_code="owner_match",
            checked_user_id="user-secret",
        )
    )
    recorder.record(
        HumanConfirmationEvent(
            tool_call_id="call-1",
            authenticated_user_id="user-secret",
            confirmation_id="confirmation-secret",
            args={"order_id": "B456"},
        )
    )
    recorder.record(
        ToolResultEvent(
            tool_call_id="call-1",
            tool_name="get_order",
            status=ToolStatus.SUCCESS,
            result={
                "status": "packed",
                "customer_email": "person@example.test",
                "items": [{"sku": "item-1"}],
            },
        )
    )
    recorder.record(
        LLMOutputEvent(raw_output="Contains private customer data")
    )
    recorder.record(
        ToolResultEvent(
            tool_call_id="call-1",
            result="unstructured private result",
        )
    )
    recorder.record(FinalResponseEvent(response="Private response text"))

    exported_text = (tmp_path / "redacted.json")
    recorder.export_json(exported_text)
    text = exported_text.read_text(encoding="utf-8")
    data = json.loads(text)

    for secret in (
        "person@example.test",
        "Sensitive Person",
        "张三",
        "Example Street 12",
        "Other Street 34",
        "contact@example.test",
        "+1-555-0199",
        "raw-access-token",
        "+1-555-0100",
        "raw-confirmation-capability",
        "raw-authentication-secret",
        "user-secret",
        "confirmation-secret",
        "Contains private customer data",
        "unstructured private result",
        "Private response text",
    ):
        assert secret not in text

    call = data["events"][0]
    assert call["tool_name"] == "get_order"
    assert call["args"]["order_id"] == "B456"
    assert call["args"]["customer"]["email"] == "[REDACTED]"
    assert call["args"]["customer"]["name"] == "[REDACTED]"
    assert call["args"]["customer_name"] == "[REDACTED]"
    assert call["args"]["billing_address"] == "[REDACTED]"
    assert call["args"]["shipping_address"] == "[REDACTED]"
    assert call["args"]["contact_email"] == "[REDACTED]"
    assert call["args"]["customer_phone"] == "[REDACTED]"
    assert call["args"]["access_token"] == "[REDACTED]"
    assert call["args"]["authentication"] == "[REDACTED]"
    assert call["args"]["items"][0]["phone"] == "[REDACTED]"
    assert call["args"]["items"][1]["status"] == "packed"
    assert call["args"]["confirmation_token"] == "[REDACTED]"
    assert data["events"][1]["reason_code"] == "owner_match"
    assert data["events"][1]["checked_user_id"] == "[REDACTED]"
    assert data["events"][2]["confirmation_id"] == "[REDACTED]"
    assert data["events"][3]["result"]["status"] == "packed"
    assert data["events"][3]["result"]["customer_email"] == "[REDACTED]"
    assert data["events"][3]["result"]["items"] == [{"sku": "item-1"}]
    assert data["events"][0]["event_type"] == "tool_call"
    assert data["events"][0]["trace_id"] == recorder.trace_id
    assert data["events"][0]["run_id"] == "run-1"
    assert data["events"][0]["step"] == 1
    assert data["events"][0]["tool_call_id"] == "call-1"
    assert data["events"][4]["raw_output"] == "[REDACTED]"
    assert data["events"][5]["result"] == "[REDACTED]"
    assert data["events"][6]["response"] == "[REDACTED]"


def test_get_events_returns_isolated_deep_copies():
    original = ToolCallEvent(
        tool_call_id="call-1",
        args={"nested": {"values": ["original"]}},
    )
    recorder = TraceRecorder(scenario_id="scenario-1", run_id="run-1")
    recorder.record(original)

    original.args["nested"]["values"].append("mutated-original")
    first = recorder.get_events()
    first[0].args["nested"]["values"].append("mutated-copy")
    first.clear()
    second = recorder.get_events()

    assert second[0].args == {"nested": {"values": ["original"]}}
    assert second[0].step == 1


def test_export_rejects_non_json_values(tmp_path):
    recorder = TraceRecorder(scenario_id="scenario-1", run_id="run-1")
    recorder.record(
        ToolCallEvent(
            tool_call_id="call-1",
            args={"invalid": object()},
        )
    )
    with pytest.raises(TypeError, match="not JSON-serializable"):
        recorder.export_json(tmp_path / "invalid.json")


def test_export_propagates_filesystem_errors():
    recorder = TraceRecorder(scenario_id="scenario-1", run_id="run-1")

    with patch(
        "trace_recorder.Path.write_text",
        side_effect=OSError("disk full"),
    ):
        with pytest.raises(OSError, match="disk full"):
            recorder.export_json("trace.json")
