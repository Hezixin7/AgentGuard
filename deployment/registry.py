"""In-memory run and idempotency registry for the V1 API."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import hmac
import json
import re
import threading
from typing import Any, Mapping
from uuid import uuid4

from agent_graph import AgentOrchestrator, RunResult, RunStatus
from policy_gate import ConfirmationChallenge, HumanDecision, SessionContext

from deployment.auth import Principal
from deployment.errors import ApiError
from deployment.trace_view import export_redacted_trace

_IDEMPOTENCY_KEY = re.compile(r"^[A-Za-z0-9._:-]{16,128}$")
_API_CONTRACT_VERSION = "v1"


@dataclass
class RunRecord:
    run_id: str
    owner_user_id: str
    thread_id: str | None = None
    status: str | None = None
    execution_state: str = "normal"
    version: int = 0
    challenge: ConfirmationChallenge | None = None
    result_response: str | None = None


@dataclass
class IdempotencyRecord:
    state: str
    fingerprint: str
    run_id: str
    status_code: int | None = None
    body: dict[str, Any] | None = None


@dataclass(frozen=True)
class ServiceResult:
    status_code: int
    body: dict[str, Any]
    replayed: bool = False


class DeploymentService:
    """Synchronous, single-process service around an AgentGuard orchestrator."""

    def __init__(
        self,
        orchestrator: AgentOrchestrator,
        authenticator: Any,
    ) -> None:
        self.orchestrator = orchestrator
        self.authenticator = authenticator
        self._runs: dict[str, RunRecord] = {}
        self._idempotency: dict[tuple[str, ...], IdempotencyRecord] = {}
        self._lock = threading.RLock()

    # ------------------------------------------------------------------ create
    def create_run(
        self,
        principal: Principal,
        idempotency_key: str,
        message: str,
    ) -> ServiceResult:
        _validate_idempotency_key(idempotency_key)
        fingerprint = _fingerprint(
            {
                "api_contract_version": _API_CONTRACT_VERSION,
                "operation": "create_run",
                "message": message,
            }
        )
        namespace = _namespace(
            principal,
            "POST",
            "/v1/runs",
            "",
            idempotency_key,
        )
        with self._lock:
            replay = self._lookup_idempotency(namespace, fingerprint)
            if replay is not None:
                return replay

            run_id = uuid4().hex
            run = RunRecord(
                run_id=run_id,
                owner_user_id=principal.user_id,
                execution_state="in_progress",
            )
            self._runs[run_id] = run
            record = IdempotencyRecord(
                state="in_progress",
                fingerprint=fingerprint,
                run_id=run_id,
            )
            self._idempotency[namespace] = record
            started = False
            try:
                started = True
                result = self.orchestrator.start(
                    message,
                    SessionContext(principal.user_id),
                    scenario_id="api",
                    run_id=run_id,
                )
                self._apply_result(run, result)
                body = self._run_view(run)
                self._commit_idempotent(record, 201, body)
                self._commit_run(run)
                return ServiceResult(201, body)
            except Exception:
                if not started:
                    self._idempotency.pop(namespace, None)
                    self._runs.pop(run_id, None)
                    raise ApiError(
                        500,
                        "internal_error",
                        "The request could not be processed.",
                    ) from None
                self._mark_indeterminate(run, record)
                raise ApiError(
                    500,
                    "execution_outcome_indeterminate",
                    (
                        "Execution outcome is uncertain. Inspect the run "
                        "status; do not retry the request."
                    ),
                    {"run_id": run_id, "execution_state": "indeterminate"},
                ) from None

    # ------------------------------------------------------------------- read
    def get_run(self, principal: Principal, run_id: str) -> ServiceResult:
        with self._lock:
            run = self._owned_run(principal, run_id)
            return ServiceResult(200, self._run_view(run))

    def get_trace(
        self,
        principal: Principal,
        run_id: str,
    ) -> ServiceResult:
        with self._lock:
            run = self._owned_run(principal, run_id)
            if run.thread_id is None:
                raise ApiError(
                    409,
                    "trace_unavailable",
                    "Trace is not available for this run.",
                )
            payload = export_redacted_trace(
                self.orchestrator,
                run.thread_id,
            )
            return ServiceResult(200, payload)

    # -------------------------------------------------------------- confirm
    def confirm_run(
        self,
        principal: Principal,
        run_id: str,
        idempotency_key: str,
        decision: str,
        expected_version: int,
    ) -> ServiceResult:
        _validate_idempotency_key(idempotency_key)
        fingerprint = _fingerprint(
            {
                "api_contract_version": _API_CONTRACT_VERSION,
                "operation": "confirm_run",
                "run_id": run_id,
                "expected_version": expected_version,
                "decision": decision,
            }
        )
        namespace = _namespace(
            principal,
            "POST",
            "/v1/runs/{run_id}/confirmation",
            run_id,
            idempotency_key,
        )
        with self._lock:
            replay = self._lookup_idempotency(namespace, fingerprint)
            if replay is not None:
                return replay

            run = self._owned_run(principal, run_id)
            if run.execution_state == "indeterminate":
                raise ApiError(
                    409,
                    "run_outcome_indeterminate",
                    (
                        "The run outcome is uncertain; further decisions "
                        "are not accepted."
                    ),
                    {"run_id": run_id, "execution_state": "indeterminate"},
                )
            if (
                run.status != "awaiting_confirmation"
                or run.challenge is None
                or run.thread_id is None
            ):
                raise ApiError(
                    409,
                    "run_not_awaiting_confirmation",
                    "The run is not awaiting a human confirmation.",
                    {
                        "run_id": run_id,
                        "status": run.status,
                        "current_version": run.version,
                    },
                )
            if run.version != expected_version:
                raise ApiError(
                    409,
                    "run_version_conflict",
                    "The run state has changed; refresh the run before deciding.",
                    {
                        "run_id": run_id,
                        "status": run.status,
                        "current_version": run.version,
                    },
                )

            record = IdempotencyRecord(
                state="in_progress",
                fingerprint=fingerprint,
                run_id=run_id,
            )
            self._idempotency[namespace] = record
            started = False
            try:
                started = True
                result = self.orchestrator.decide(
                    run.thread_id,
                    run.challenge.confirmation_id,
                    SessionContext(principal.user_id),
                    HumanDecision(decision),
                )
                self._apply_result(run, result)
                body = self._run_view(run)
                self._commit_idempotent(record, 200, body)
                self._commit_run(run)
                return ServiceResult(200, body)
            except Exception:
                if not started:
                    self._idempotency.pop(namespace, None)
                    raise ApiError(
                        500,
                        "internal_error",
                        "The request could not be processed.",
                    ) from None
                self._mark_indeterminate(run, record)
                raise ApiError(
                    500,
                    "execution_outcome_indeterminate",
                    (
                        "Execution outcome is uncertain. Inspect the run "
                        "status; do not retry the request."
                    ),
                    {"run_id": run_id, "execution_state": "indeterminate"},
                ) from None

    # ------------------------------------------------------------- internals
    def _lookup_idempotency(
        self,
        namespace: tuple[str, ...],
        fingerprint: str,
    ) -> ServiceResult | None:
        record = self._idempotency.get(namespace)
        if record is None:
            return None
        if not hmac.compare_digest(record.fingerprint, fingerprint):
            raise ApiError(
                409,
                "idempotency_key_reuse",
                "Idempotency key was already used for a different request.",
            )
        if record.state == "completed":
            return ServiceResult(
                record.status_code or 200,
                record.body or {},
                replayed=True,
            )
        if record.state == "in_progress":
            raise ApiError(
                409,
                "request_in_progress",
                "The request is still in progress.",
            )
        raise ApiError(
            409,
            "request_indeterminate",
            (
                "The previous outcome is uncertain; do not retry this "
                "request."
            ),
            {
                "run_id": record.run_id,
                "execution_state": "indeterminate",
            },
        )

    def _owned_run(self, principal: Principal, run_id: str) -> RunRecord:
        run = self._runs.get(run_id)
        if run is None or run.owner_user_id != principal.user_id:
            raise ApiError(404, "run_not_found", "Run was not found.")
        return run

    def _apply_result(self, run: RunRecord, result: RunResult) -> None:
        run.thread_id = result.thread_id
        run.version += 1
        if result.status is RunStatus.PAUSED:
            if result.challenge is None:
                raise RuntimeError("paused run has no confirmation challenge")
            run.status = "awaiting_confirmation"
            run.challenge = result.challenge
            run.result_response = None
        elif result.status is RunStatus.COMPLETED:
            run.status = "completed"
            run.challenge = None
            run.result_response = result.response
        elif result.status is RunStatus.FAILED_CLOSED:
            run.status = "failed_closed"
            run.challenge = None
            run.result_response = result.response
        else:
            raise RuntimeError("unsupported frozen run status")
        run.execution_state = "normal"

    def _commit_idempotent(
        self,
        record: IdempotencyRecord,
        status_code: int,
        body: dict[str, Any],
    ) -> None:
        record.status_code = status_code
        record.body = body
        record.state = "completed"

    def _commit_run(self, run: RunRecord) -> None:
        self._runs[run.run_id] = run

    def _mark_indeterminate(
        self,
        run: RunRecord,
        record: IdempotencyRecord,
    ) -> None:
        record.state = "indeterminate"
        record.run_id = run.run_id
        run.execution_state = "indeterminate"
        run.challenge = None
        self._runs[run.run_id] = run

    @staticmethod
    def _run_view(run: RunRecord) -> dict[str, Any]:
        confirmation = None
        if (
            run.execution_state == "normal"
            and run.status == "awaiting_confirmation"
            and run.challenge is not None
        ):
            confirmation = {
                "operation": run.challenge.operation,
                "order_id": run.challenge.order_id,
                "parameters": {
                    key: value
                    for key, value in run.challenge.parameters
                    if key != "operation"
                },
                "expires_at": run.challenge.expires_at,
            }
        result = None
        if (
            run.status in {"completed", "failed_closed"}
            and run.result_response is not None
        ):
            result = {"response": run.result_response}
        return {
            "run_id": run.run_id,
            "status": run.status,
            "execution_state": run.execution_state,
            "version": run.version,
            "confirmation": confirmation,
            "result": result,
        }


def _namespace(
    principal: Principal,
    method: str,
    endpoint: str,
    run_id: str,
    key: str,
) -> tuple[str, ...]:
    return (
        principal.user_id,
        method.upper(),
        endpoint,
        run_id,
        key,
    )


def _fingerprint(payload: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        payload,
        ensure_ascii=True,
        separators=(",", ":"),
        sort_keys=True,
    )
    return sha256(canonical.encode("utf-8")).hexdigest()


def _validate_idempotency_key(value: str) -> None:
    if not isinstance(value, str) or not _IDEMPOTENCY_KEY.fullmatch(value):
        raise ApiError(
            422,
            "invalid_request",
            "Idempotency-Key is missing or invalid.",
        )
