"""Read-only Kubernetes MCP server: `python -m mcp_cluster`.

Runs as its own Deployment with its own ServiceAccount. That is the point: the RBAC
lives here, so the agent pod holds no cluster credentials at all, and this pod holds
no LLM or GitHub credentials.
"""

from __future__ import annotations

import logging
import os

from mcp_cluster.tools import readiness, register
from mcp_common.server import build_server, serve

logging.basicConfig(
    format="%(message)s",
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
)


def main() -> None:
    server = build_server("mcp-cluster", "0.1.0", readiness=readiness)
    register(server)
    serve(server, port=int(os.environ.get("PORT", "8080")))


if __name__ == "__main__":
    main()
