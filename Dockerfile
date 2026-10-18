# Verified multi-architecture manifest digest for python:3.12-slim-bookworm.
FROM python:3.12-slim-bookworm@sha256:34386ef0cb081344d7ec1c103ba398e6e9f64e9ab3a1509accc92a4e24a07258

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    AGENTGUARD_DEBUG_PROVIDER_ERRORS=0

WORKDIR /app

RUN groupadd --system --gid 10001 app \
 && useradd --system --uid 10001 --gid app \
      --create-home --home-dir /home/app app \
 && python -m venv /opt/venv

ENV PATH=/opt/venv/bin:$PATH

COPY deployment/requirements.lock /tmp/requirements.lock
RUN /opt/venv/bin/pip install \
      --no-cache-dir \
      --requirement /tmp/requirements.lock \
 && rm /tmp/requirements.lock

COPY --chown=app:app \
    agent_graph.py \
    policy_gate.py \
    simulated_tools.py \
    trace_events.py \
    trace_recorder.py \
    ./

COPY --chown=app:app deployment/*.py ./deployment/

USER app
EXPOSE 8000

# Trace export uses a temporary file; /tmp is writable in the base image.
# If running with --read-only, mount a writable tmpfs at /tmp.
HEALTHCHECK --interval=30s --timeout=3s --start-period=5s --retries=3 \
  CMD ["python", "-c", "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:8000/healthz', timeout=2); sys.exit(0 if r.status == 200 else 1)"]

CMD ["uvicorn", "deployment.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
