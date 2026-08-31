"""Multi-server MCP client.

Design note — why a fresh MCP session per call:

The `Client` context manager is built on anyio task groups, whose cancel scopes are
bound to the task that opened them. Holding one open for the lifetime of a Telegram
bot and using it from whichever task happens to handle a message is a reliable way to
get "cancel scope in a different task" errors that are miserable to debug.

Every server we talk to runs in stateless mode (ours by `stateless_http=True`, the
official GitHub one because its HTTP handler is stateless), so a session carries no
value across calls. We therefore open a session per operation and keep only the
`httpx2.AsyncClient` — the expensive part, holding the connection pool and TLS
session — alive for the process. The cost is one extra `initialize` round-trip per
tool call, which against an in-cluster service is noise next to the LLM call it sits
between.

Tool names are namespaced as `<server>__<tool>` so two servers can expose a tool of
the same name, and so profile globs like `github__*` can address a whole server.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any

import httpx2
from mcp import Client
from mcp.client.streamable_http import streamable_http_client

logger = logging.getLogger(__name__)

NAMESPACE_SEP = "__"
_TOOL_CACHE_TTL = 300.0


@dataclass(frozen=True)
class MCPServerConfig:
    name: str
    url: str
    # Sent on every request. The official GitHub MCP server in http mode has no token
    # field in its own config and rejects unauthenticated requests with 401, so the
    # PAT has to travel as a per-request header from here.
    headers: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class RemoteTool:
    server: str
    name: str  # bare name as the server knows it
    qualified_name: str  # "<server>__<name>"
    description: str
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ToolResult:
    ok: bool
    text: str
    structured: Any | None = None

    def for_model(self) -> str:
        """What actually goes back into the conversation as the tool result."""
        return self.text


class MCPError(Exception):
    pass


def split_qualified(qualified_name: str) -> tuple[str, str]:
    """"github__create_issue" -> ("github", "create_issue")."""
    server, sep, tool = qualified_name.partition(NAMESPACE_SEP)
    if not sep or not tool:
        raise MCPError(f"tool name {qualified_name!r} is not namespaced as server{NAMESPACE_SEP}tool")
    return server, tool


class MCPPool:
    """Aggregates several MCP servers behind one tool namespace."""

    def __init__(self, servers: list[MCPServerConfig], *, timeout: float = 120.0) -> None:
        self._servers = {s.name: s for s in servers}
        self._timeout = timeout
        self._http: dict[str, httpx2.AsyncClient] = {}
        self._tools: list[RemoteTool] | None = None
        self._tools_at: float = 0.0
        self._lock = asyncio.Lock()

    # -- lifecycle ----------------------------------------------------------

    async def start(self) -> None:
        for name, cfg in self._servers.items():
            self._http[name] = httpx2.AsyncClient(
                headers=cfg.headers,
                timeout=httpx2.Timeout(30.0, read=self._timeout),
                follow_redirects=True,
            )

    async def close(self) -> None:
        for client in self._http.values():
            await client.aclose()
        self._http.clear()

    async def __aenter__(self) -> MCPPool:
        await self.start()
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()

    # -- internals ----------------------------------------------------------

    def _session(self, server: str):
        cfg = self._servers.get(server)
        if cfg is None:
            raise MCPError(f"unknown MCP server {server!r}; configured: {sorted(self._servers)}")
        http_client = self._http.get(server)
        if http_client is None:
            raise MCPError("MCPPool.start() was not called")
        return Client(streamable_http_client(cfg.url, http_client=http_client))

    # -- discovery ----------------------------------------------------------

    async def list_tools(self, *, force_refresh: bool = False) -> list[RemoteTool]:
        """Aggregated tool list, cached — servers do not change their tools often."""
        async with self._lock:
            fresh = self._tools is not None and (time.monotonic() - self._tools_at) < _TOOL_CACHE_TTL
            if fresh and not force_refresh:
                return self._tools or []

            collected: list[RemoteTool] = []
            for name in self._servers:
                try:
                    collected.extend(await self._list_one(name))
                except Exception as exc:  # noqa: BLE001
                    # One unreachable server must not blind the agent to the others.
                    logger.warning("mcp server %s: list_tools failed: %s", name, exc)
            self._tools = collected
            self._tools_at = time.monotonic()
            return collected

    async def _list_one(self, server: str) -> list[RemoteTool]:
        async with self._session(server) as client:
            result = await client.list_tools()
        return [
            RemoteTool(
                server=server,
                name=t.name,
                qualified_name=f"{server}{NAMESPACE_SEP}{t.name}",
                description=t.description or "",
                input_schema=t.input_schema or {"type": "object", "properties": {}},
            )
            for t in result.tools
        ]

    # -- invocation ---------------------------------------------------------

    async def call_tool(self, qualified_name: str, arguments: dict[str, Any]) -> ToolResult:
        server, tool = split_qualified(qualified_name)
        try:
            async with self._session(server) as client:
                result = await client.call_tool(tool, arguments)
        except Exception as exc:  # noqa: BLE001
            # Transport-level failure. Surface it as a tool result rather than an
            # exception so the loop can let the model react instead of dying.
            logger.warning("mcp call %s failed: %s", qualified_name, exc)
            return ToolResult(ok=False, text=f"tool transport error: {exc}")

        text = _flatten_content(result.content)
        return ToolResult(
            ok=not result.is_error,
            text=text or ("(empty result)" if not result.is_error else "(error, no detail)"),
            structured=result.structured_content,
        )


def _flatten_content(content: Any) -> str:
    """Join the content blocks of a CallToolResult into plain text for the model."""
    if not content:
        return ""
    parts: list[str] = []
    for block in content:
        text = getattr(block, "text", None)
        if text:
            parts.append(text)
            continue
        # Non-text blocks (images, resources) are summarised rather than dropped, so
        # the model at least knows something came back.
        parts.append(f"({getattr(block, 'type', 'unknown')} content block)")
    return "\n".join(parts)
