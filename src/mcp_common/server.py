"""Bootstrap shared by every MCP server we write ourselves.

The API here follows mcp SDK 2.x, which renamed a lot from 1.x. Verified against
mcp==2.1.1 with a live client/server round-trip:

  * `MCPServer` replaces `FastMCP`
  * `@server.tool(...)` builds the JSON schema from type hints
  * `server.custom_route(path, methods=[...])` mounts plain Starlette routes
  * `server.streamable_http_app(stateless_http=True)` returns the ASGI app
  * a tool that returns a dict is delivered as `structured_content`

Do not "modernise" these names against online examples: nearly every tutorial
still shows the 1.x spelling and will silently not exist here.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable

import uvicorn
from mcp.server import MCPServer
from starlette.requests import Request
from starlette.responses import JSONResponse, Response

logger = logging.getLogger(__name__)

# Readiness returns (ok, detail). Servers that talk to an upstream (Cloudflare API,
# kube-apiserver) supply one so k8s can tell "process is up" from "actually usable".
# Either sync or async: the kubernetes client is blocking, the Cloudflare one is not.
ReadinessCheck = Callable[[], tuple[bool, str] | Awaitable[tuple[bool, str]]]


def build_server(
    name: str,
    version: str,
    *,
    readiness: ReadinessCheck | None = None,
) -> MCPServer:
    """Create an MCPServer with the health endpoints k8s probes expect."""
    server = MCPServer(name=name, version=version)

    @server.custom_route("/healthz", methods=["GET"], include_in_schema=False)
    async def _healthz(_: Request) -> JSONResponse:
        # Liveness only: the process is running and the event loop is turning.
        return JSONResponse({"status": "ok", "server": name, "version": version})

    @server.custom_route("/readyz", methods=["GET"], include_in_schema=False)
    async def _readyz(_: Request) -> JSONResponse:
        if readiness is None:
            return JSONResponse({"status": "ready", "server": name})
        try:
            outcome = readiness()
            if inspect.isawaitable(outcome):
                outcome = await outcome
            ok, detail = outcome
        except Exception as exc:  # a probe must never raise
            logger.warning("readiness check raised: %s", exc)
            return JSONResponse({"status": "not-ready", "detail": str(exc)}, status_code=503)
        if not ok:
            return JSONResponse({"status": "not-ready", "detail": detail}, status_code=503)
        return JSONResponse({"status": "ready", "detail": detail})

    @server.custom_route("/metrics", methods=["GET"], include_in_schema=False)
    async def _metrics(_: Request) -> Response:
        # Every server built here gets the endpoint, not just the agent ones. For an
        # MCP server the agent counters stay at zero, but the process collectors that
        # prometheus_client registers by default — resident memory, CPU, open file
        # descriptors — are worth having on all four.
        from agentcore import metrics

        payload, content_type = metrics.render()
        return Response(payload, media_type=content_type)

    return server


def serve(server: MCPServer, *, host: str = "0.0.0.0", port: int = 8080) -> None:
    """Run the server over Streamable HTTP until killed.

    `stateless_http=True` on purpose: no session affinity, so the Deployment can be
    restarted or scaled without the agent losing a session mid-conversation. This
    matches how the official GitHub MCP server runs its own HTTP mode.
    """
    app = server.streamable_http_app(stateless_http=True, host=host)
    uvicorn.run(app, host=host, port=port, log_level="info", access_log=False)
