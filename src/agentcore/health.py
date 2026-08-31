"""Liveness and readiness endpoints.

`gemini-sre-agent` serves these from a `http.server` in a daemon thread that returns
200 unconditionally, so a wedged Telegram poller — the one failure that actually
matters — still reports Ready and k8s never restarts it.

Here the health app runs on the *same* event loop as the bot and readiness asks the
Updater whether it is really polling, so a dead poller fails the probe.
"""

from __future__ import annotations

from collections.abc import Callable

import uvicorn
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.routing import Route

ReadinessFn = Callable[[], tuple[bool, str]]


def build_health_app(*, agent: str, readiness: ReadinessFn) -> Starlette:
    async def healthz(_: Request) -> JSONResponse:
        return JSONResponse({"status": "ok", "agent": agent})

    async def readyz(_: Request) -> JSONResponse:
        try:
            ok, detail = readiness()
        except Exception as exc:  # a probe must never raise
            return JSONResponse({"status": "not-ready", "detail": str(exc)}, status_code=503)
        if not ok:
            return JSONResponse({"status": "not-ready", "detail": detail}, status_code=503)
        return JSONResponse({"status": "ready", "detail": detail})

    return Starlette(
        routes=[
            Route("/healthz", healthz),
            Route("/readyz", readyz),
            Route("/", healthz),
        ]
    )


def build_health_server(app: Starlette, *, port: int) -> uvicorn.Server:
    """A uvicorn Server to be awaited as a task on the bot's own event loop."""
    config = uvicorn.Config(
        app,
        host="0.0.0.0",  # in-cluster only: the Service is ClusterIP, never exposed
        port=port,
        log_level="warning",
        access_log=False,
    )
    return uvicorn.Server(config)
