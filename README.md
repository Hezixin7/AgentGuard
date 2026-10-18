# AgentGuard

**Status:** V1 research/portfolio implementation. **Not production-ready.**

AgentGuard is a simulated e-commerce support agent that explores a central
question in LLM application engineering:

> How do you let a model choose actions without letting it become the
> authorization boundary?

The project combines a LangGraph agent with a deterministic PolicyGate,
trusted human confirmation, refund idempotency, timeout-after-commit
handling, correlated audit traces, and a deterministic offline evaluation
framework. It compares three groups: an unguarded baseline, a policy-enforced
baseline, and a policy-enforced agent with trace auditing.

The model proposes actions. It never receives or controls the trusted
application identity, confirmation capability, tools, or enforcement logic.

## Why this project

Most agent demos focus on tool use and response quality. AgentGuard focuses on
the application layer around the model:

- **Trusted identity** comes from server-side authentication, never from model
  output or user text.
- **Tool authorization** is deterministic and re-checked against current state.
- **Irreversible actions** pause for a trusted human decision.
- **Retries** are idempotent; a timeout after commit does not create a second
  business side effect.
- **Audit evidence** is correlated, redacted, and separated from enforcement.
- **Evaluation** includes explicit unguarded and policy-enforced baselines,
  rather than reporting a single flattering metric.

## Key features

- Deterministic `PolicyGate` for ownership, canonical parameters, business
  state, refund replay/conflict handling, and confirmation.
- LangGraph pause/resume with a single-use, server-side confirmation capability.
- Atomic refund + idempotency state; timeout-before-commit and
  timeout-after-commit/replay are distinguished.
- Typed, correlated trace events with redacted JSON export.
- FastAPI deployment API with:
  - static bearer authentication and trusted principals,
  - owner-only runs and traces,
  - versioned human-confirmation decisions,
  - HTTP idempotency keys,
  - strict Pydantic schemas and sanitized errors,
  - byte-accurate request-size enforcement,
  - indeterminate-outcome protection.
- DeepSeek adapter over the OpenAI-compatible API:
  - strict local JSON proposal validation,
  - safe non-action fallback on provider/validation failure,
  - no provider-native tool IDs fabricated,
  - internal tool messages converted to clearly labeled untrusted user context.
- Deterministic offline evaluation:
  - 100 frozen scenarios,
  - three groups,
  - 300 records,
  - metrics and first-divergence localization.
- Docker packaging with a pinned base image, non-root user, one worker, health
  check, and runtime-only source copies.

## Technology stack

| Layer | Technology |
|---|---|
| Runtime | Python 3.12 |
| Agent orchestration | LangGraph |
| API | FastAPI, Pydantic, Uvicorn |
| Model provider | DeepSeek via the official OpenAI Python SDK |
| Testing | pytest |
| Packaging | Docker, pinned runtime lock |
| State | In-memory only (V1) |

No database, Redis, queue, or persistent checkpoint store is used.

## Architecture

See [docs/architecture.md](docs/architecture.md) for:

- the deployed application diagram,
- the separate offline evaluation diagram,
- the trust boundary between untrusted model proposals and deterministic
  enforcement,
- the human-confirmation pause/resume sequence,
- TraceRecorder's cross-cutting audit role.

At a high level:

```text
Client → FastAPI body limit → bearer auth → DeploymentService
       → LangGraph plan → model proposal → local validation
       → PolicyGate → simulated tools/state
       → trace event → final response or human confirmation pause
```

## Security mechanisms and trust boundaries

| Untrusted | Trusted application layer |
|---|---|
| User message | Static bearer token → `Principal(user_id)` |
| Model final response | LangGraph routing and proposal validation |
| Model tool proposal | PolicyGate authorization, ownership, idempotency, confirmation |
| Tool arguments in the proposal | Canonicalization and authoritative state checks |
| Client request body | Never supplies `user_id`, group, capability, or internal confirmation identity |

Additional properties:

- The model receives only message content; it never receives the session,
  PolicyGate, tools, or confirmation capability.
- Confirmation challenges exposed to clients omit confirmation IDs, state
  hashes, and raw capabilities. The capability remains server-side.
- A valid confirmation resume re-runs PolicyGate against current state.
- Expired, stale, reused, missing, or mismatched confirmations fail closed.
- Refund retries use the same internal idempotency key and replay the stored
  result without repeating the business side effect.
- Public deployment routes are AgentGuard-only. Baseline A and Baseline B are
  offline evaluation groups, not API modes.

## Deterministic offline evaluation

The 100-scenario evaluation is **deterministic and scripted**. It uses frozen
scenario oracles, scripted model proposals, a fake clock, and in-memory state.
It is **not** 100 live LLM calls.

| Group | Components | Task success | Confirmed unsafe execution lower bound | False-block rate |
|---|---|---:|---:|---:|
| Baseline A | Tools only | 35/100 | **43/75** | Not applicable (no policy gate) |
| Baseline B | Tools + PolicyGate | 100/100 | **0/75** | 0/13 eligible |
| AgentGuard | Baseline B + TraceRecorder | 100/100 | **0/75** | 0/13 eligible |

Additional deterministic facts:

- 100 scenarios: 20 core + 60 deterministic variants + 20 natural-language
  perturbations.
- 300 group records: 100 scenarios × 3 groups.
- Tool attempts/invocations: Baseline A 110/110; Baseline B and AgentGuard
  110/45.
- Baseline B and AgentGuard share the same enforcement logic. The equal unsafe
  and false-block results show that **TraceRecorder is not a second blocking
  layer**; it adds audit and diagnostic scope, not blocking performance.
- Variants share parent scenarios; aggregate rows are not independent
  statistical samples.
- Latency, token usage/cost, and trace-storage overhead are **not measured** in
  this evaluation.

### Diagnostic localization evaluation

The localization evaluation uses frozen gold labels and a separate synthetic
corruption corpus:

- AgentGuard and Baseline B clean-run parent-cluster status accuracy: 20/20.
- Baseline A eligible observed-violation detection: 28/28 records, 12/12
  parents.
- Corruption corpus: 5/5 proven earliest-divergence cases, 5/5 observed
  violations, and 5/5 abstentions.
- The frozen 100-scenario corpus contains no proven earliest-divergence labels,
  so strict earliest accuracy is supported only by the corruption corpus.
- **Independent human review of the complete gold-label catalog remains
  pending.** These are structurally validated labels, not human-certified
  ground truth.

## Limited live DeepSeek checks

During development, three selected end-to-end paths were exercised manually
against the live DeepSeek API. These were limited integration checks, not the
100-scenario evaluation, and they are not part of the automated pytest suite.

- No live DeepSeek calls are made by pytest.
- The provider adapter has mocked tests for proposal validation, safe fallback,
  diagnostics, and tool-message serialization.
- The 100-scenario comparison should not be described as a live LLM benchmark.

## Local setup

Prerequisites:

- Python 3.12
- A DeepSeek API key for the production entrypoint
- Docker is optional; the offline test suite does not require it

Install runtime dependencies from the lock, plus pytest for the test suite:

```bash
python3.12 -m venv .venv
.venv/bin/pip install -r deployment/requirements.lock
.venv/bin/pip install pytest
```

Configure server-side secrets (do not commit them):

```bash
export DEEPSEEK_API_KEY='<server-side DeepSeek key>'
export AGENTGUARD_API_KEYS='{"<long-random-bearer-token>":"user-1"}'
```

Start the existing ASGI entrypoint with exactly one worker:

```bash
.venv/bin/uvicorn deployment.main:app \
  --host 127.0.0.1 --port 8000 --workers 1
```

The app fails at startup if required configuration is missing or malformed.

Health check:

```bash
curl -fsS http://127.0.0.1:8000/healthz
```

## Docker deployment

Build the image:

```bash
docker build --pull -t agentguard:v1 .
```

Run with runtime secrets from a protected env file:

```bash
docker run --rm --name agentguard \
  -p 127.0.0.1:8000:8000 \
  --env-file /secure/path/agentguard.env \
  agentguard:v1
```

The container:

- runs as UID/GID 10001 (non-root),
- runs exactly one Uvicorn worker,
- exposes port 8000 internally,
- has a `/healthz` health check,
- keeps `AGENTGUARD_DEBUG_PROVIDER_ERRORS` disabled by default,
- requires a writable `/tmp` only for trace export; use a tmpfs if running
  with `--read-only`.

Manual Docker verification was performed during development: the image built,
health checks passed, UID 10001 was confirmed, one worker was observed, the
mock container passed authentication/trace checks, and the containers were
stopped. These checks are manual and limited in scope; they are not part of the
automated pytest suite.

## API examples

All state-changing requests require a fresh `Idempotency-Key` (16–128
characters from `A-Za-z0-9._:-`) and a bearer token. For the examples below,
set `AGENTGUARD_TOKEN` to the same bearer-token value configured in
`AGENTGUARD_API_KEYS`, and use the `run_id`/`version` returned by the create
request.

Create a run:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/runs \
  -H "Authorization: Bearer $AGENTGUARD_TOKEN" \
  -H "Idempotency-Key: demo-idempotency-key-0001" \
  -H "Content-Type: application/json" \
  -d '{"message":"Look up order P100."}'
```

The response may be `completed` or `awaiting_confirmation`. For a pending
confirmation, use the returned `version` and do **not** approve irreversible
actions automatically:

```bash
curl -sS -X POST http://127.0.0.1:8000/v1/runs/$RUN_ID/confirmation \
  -H "Authorization: Bearer $AGENTGUARD_TOKEN" \
  -H "Idempotency-Key: demo-idempotency-key-0002" \
  -H "Content-Type: application/json" \
  -d '{"decision":"approve","expected_version":1}'
```

Retrieve run status and the owner-only redacted trace:

```bash
curl -sS http://127.0.0.1:8000/v1/runs/$RUN_ID \
  -H "Authorization: Bearer $AGENTGUARD_TOKEN"

curl -sS http://127.0.0.1:8000/v1/runs/$RUN_ID/trace \
  -H "Authorization: Bearer $AGENTGUARD_TOKEN"
```

Unknown and non-owned run IDs return `404`. The public API does not expose a
group selector or evaluation-only controls.

## Tests

Run the complete deterministic suite:

```bash
.venv/bin/python -m pytest -q
```

At the time of writing, the suite reports **410 passed**. It covers tools,
PolicyGate, confirmations, idempotency, timeout/commit semantics, LangGraph
orchestration, traces, scenario catalogs, offline evaluation, localization,
FastAPI authentication/ownership, body limits, and the DeepSeek adapter with
mocked providers.

Focused API tests:

```bash
.venv/bin/python -m pytest tests/test_api_runs.py \
  tests/test_api_security.py tests/test_api_deployment.py \
  tests/test_api_deepseek_adapter.py tests/test_api_startup.py \
  tests/test_api_body_limit.py tests/test_api_smoke.py -q
```

Docker mock smoke harness (manual; no DeepSeek key required):

```bash
docker run --rm -d --name agentguard-mock \
  -p 127.0.0.1:8001:8001 \
  -v "$PWD/tests/docker_mock_smoke.py:/tmp/docker_mock_smoke.py:ro" \
  agentguard:v1 \
  python /tmp/docker_mock_smoke.py
```

The harness uses a deterministic injected model and test-only tokens. It is
excluded from the runtime image by `.dockerignore` and bind-mounted for the
smoke check.

## Repository layout

| Path | Purpose |
|---|---|
| `agent_graph.py` | LangGraph orchestration, model contract, confirmation pause/resume |
| `policy_gate.py` | Deterministic authorization, idempotency, confirmation |
| `simulated_tools.py` | In-memory order state and tool/fault simulation |
| `trace_events.py`, `trace_recorder.py` | Typed audit events, correlation, redaction |
| `deployment/` | FastAPI, auth, registry, body limit, DeepSeek adapter, startup |
| `evaluation/` | Scenario catalogs, offline runner, metrics, localization |
| `tests/` | Deterministic and mocked-provider tests |
| `docs/architecture.md` | Deployed and evaluation architecture diagrams |
| `Dockerfile`, `.dockerignore`, `deployment/requirements.lock` | Container packaging |

## Known limitations and V1 scope

- **In-memory, single-process state.** Runs, LangGraph checkpoints, pending
  confirmations, idempotency records, traces, and order state are lost on
  restart. One Uvicorn worker is required; horizontal scaling is unsupported.
- **No database, Redis, queue, persistent checkpoint store, or durable audit
  retention.**
- **Static bearer keys.** No OIDC, scopes, key rotation, or rate limiting.
- **No internal TLS.** Terminate TLS at a reverse proxy if deploying beyond
  localhost.
- **Offline evaluation is deterministic/scripted.** It is not a live-LLM
  benchmark at 100-scenario scale.
- **Live DeepSeek and Docker checks are limited/manual**, not automated
  production verification.
- **Latency, token usage/cost, and trace-storage overhead are not measured.**
- **Gold-label human review is pending.**
- **The project is not production-ready** and should not be presented as such.

## What this project demonstrates

For an LLM/AI-agent engineering audience, the project demonstrates:

- designing a trust boundary around a model rather than trusting model output,
- deterministic policy and confirmation enforcement,
- idempotency and timeout-after-commit reasoning,
- correlated, redacted audit evidence,
- honest baseline comparison and evaluation methodology,
- deployment-oriented API, provider-adapter, and container practices,
- explicit acknowledgment of verification gaps and V1 constraints.
