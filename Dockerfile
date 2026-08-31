# syntax=docker/dockerfile:1

# One image, three entrypoints: the agent and both MCP servers share the same
# dependency set, so the Deployments differ only by `command:`. That keeps this to one
# CI pipeline, one OCIR repository and one Flux ImagePolicy instead of three.
#
# Built for linux/arm64 only — the cluster is Oracle Ampere A1.

FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build

# The venv is copied wholesale into the runtime stage, so build tooling never ships.
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY pyproject.toml ./
COPY src ./src
RUN pip install .


FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH" \
    AGENTS_DIR=/app/agents

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app
# Profiles are data, not code: they are read at startup, so adding an agent means a
# new YAML file here rather than a code change.
COPY agents ./agents

RUN useradd --uid 10001 --no-create-home --shell /usr/sbin/nologin appuser
USER 10001

EXPOSE 8080

# Overridden per Deployment: `python -m mcp_cloudflare` / `python -m mcp_cluster`.
ENTRYPOINT ["python", "-m", "agentcore"]
