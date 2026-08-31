"""Cloudflare MCP server: `python -m mcp_cloudflare`.

Runs as its own Deployment. The Cloudflare API token is mounted here and nowhere
else — in particular not in the agent pod.
"""

from __future__ import annotations

import logging
import os

from mcp_cloudflare.tools import readiness, register
from mcp_common.server import build_server, serve

logging.basicConfig(
    format="%(message)s",
    level=os.environ.get("LOG_LEVEL", "INFO").upper(),
)


def main() -> None:
    server = build_server("mcp-cloudflare", "0.1.0", readiness=readiness)
    register(server)
    serve(server, port=int(os.environ.get("PORT", "8080")))


if __name__ == "__main__":
    main()
