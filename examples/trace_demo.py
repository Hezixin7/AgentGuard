"""Create a small redacted AgentGuard trace export."""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from trace_events import (
    FinalResponseEvent,
    HumanConfirmationEvent,
    LLMOutputEvent,
    PolicyDecisionEvent,
    PolicyOutcome,
    ConfirmationStatus,
    ToolCallEvent,
    ToolResultEvent,
    ToolStatus,
)
from trace_recorder import TraceRecorder


def main() -> None:
    recorder = TraceRecorder(
        scenario_id="refund-timeout-and-replay",
        run_id="demo-run",
    )
    recorder.record(
        LLMOutputEvent(
            raw_output='{"tool": "refund_order", "order_id": "A123"}',
            parsed_plan={"tool": "refund_order", "order_id": "A123"},
            model="demo-model",
        )
    )
    recorder.record(
        ToolCallEvent(
            tool_call_id="refund-attempt-1",
            tool_name="refund_order",
            args={"order_id": "A123", "amount": 25},
            args_source="user_text",
        )
    )
    recorder.record(
        PolicyDecisionEvent(
            tool_call_id="refund-attempt-1",
            tool_name="refund_order",
            decision=PolicyOutcome.ALLOWED,
            reason_code="refund_eligible",
            idempotency_key_ref="sha256:demo-request",
        )
    )
    recorder.record(
        HumanConfirmationEvent(
            tool_call_id="refund-attempt-1",
            tool_name="refund_order",
            status=ConfirmationStatus.CONFIRMED,
            confirmation_id="confirmation-reference",
            args={"order_id": "A123", "amount": 25},
        )
    )
    recorder.record(
        ToolResultEvent(
            tool_call_id="refund-attempt-1",
            tool_name="refund_order",
            status=ToolStatus.TIMEOUT,
            error_type="TimeoutError",
            idempotency_key_ref="sha256:demo-request",
            state_changed=True,
        )
    )
    recorder.record(
        ToolCallEvent(
            tool_call_id="refund-attempt-2",
            tool_name="refund_order",
            args={"order_id": "A123", "amount": 25},
            args_source="session",
        )
    )
    recorder.record(
        PolicyDecisionEvent(
            tool_call_id="refund-attempt-2",
            tool_name="refund_order",
            decision=PolicyOutcome.REPLAY,
            reason_code="idempotent_replay",
            idempotency_key_ref="sha256:demo-request",
        )
    )
    recorder.record(
        ToolResultEvent(
            tool_call_id="refund-attempt-2",
            tool_name="refund_order",
            status=ToolStatus.REPLAYED,
            result={"refund_id": "refund-demo", "status": "completed"},
            idempotency_key_ref="sha256:demo-request",
            replayed=True,
            state_changed=False,
        )
    )
    recorder.record(
        FinalResponseEvent(
            response="The refund completed successfully.",
            cited_tool_calls=["refund-attempt-2"],
        )
    )

    output = Path(__file__).with_name("trace_demo.json")
    recorder.export_json(output)
    print(f"Wrote trace export to {output}")


if __name__ == "__main__":
    main()
