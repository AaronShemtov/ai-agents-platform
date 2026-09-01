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

Header-bound parameters — see docs/mcp-header-params.md for the full diagnosis:

A server may mark a parameter in its JSON Schema with `x-mcp-header: <name>`, meaning
the value must ALSO travel as an `Mcp-Param-<name>` HTTP header, so a gateway can see
what a call touches without parsing the body. The GitHub server does this for `owner`
and `repo`. The SDK refuses to send a request whose headers do not match, so without
this every GitHub tool taking a repository was unusable — which is most of them.
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
    # Read timeout for this server alone. A delegated agent runs its own tool loop and
    # can legitimately take minutes, where a DNS lookup should never take ten seconds.
    # None means the pool default.
    timeout: float | None = None


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


def header_params(schema: dict[str, Any], arguments: dict[str, Any]) -> dict[str, str]:
    """Arguments the server wants echoed as `Mcp-Param-<name>` headers.

    Only properties carrying `x-mcp-header` and actually present in this call.
    """
    out: dict[str, str] = {}
    for prop, spec in (schema.get("properties") or {}).items():
        if not isinstance(spec, dict):
            continue
        header = spec.get("x-mcp-header")
        if not header or prop not in arguments:
            continue
        value = arguments[prop]
        if value is None:
            continue
        out[f"Mcp-Param-{header}"] = str(value)
    return out


def explain_exception(exc: BaseException, _depth: int = 0) -> str:
    """Flatten an ExceptionGroup down to the causes that actually say something.

    anyio wraps transport failures in a task-group ExceptionGroup whose str() is
    "unhandled errors in a TaskGroup (1 sub-exception)" — which cost real debugging
    time once already. Never log or return the group itself.
    """
    if isinstance(exc, BaseExceptionGroup) and _depth < 5:
        parts = [explain_exception(e, _depth + 1) for e in exc.exceptions]
        joined = "; ".join(p for p in parts if p)
        return joined or repr(exc)
    return f"{type(exc).__name__}: {exc}"


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
                timeout=httpx2.Timeout(30.0, read=cfg.timeout or self._timeout),
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
                except Exception as exc:
                    # One unreachable server must not blind the agent to the others.
                    logger.warning(
                        "mcp server %s: list_tools failed: %s", name, explain_exception(exc)
                    )
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

    def _schema_for(self, qualified_name: str) -> dict[str, Any]:
        for tool in self._tools or []:
            if tool.qualified_name == qualified_name:
                return tool.input_schema
        return {}

    async def _invoke(self, url: str, http: httpx2.AsyncClient, tool: str, arguments):
        async with Client(streamable_http_client(url, http_client=http)) as client:
            return await client.call_tool(tool, arguments)

    async def call_tool(self, qualified_name: str, arguments: dict[str, Any]) -> ToolResult:
        server, tool = split_qualified(qualified_name)
        cfg = self._servers.get(server)
        if cfg is None:
            raise MCPError(f"unknown MCP server {server!r}; configured: {sorted(self._servers)}")

        # The schema is what tells us which arguments need Mcp-Param headers, so the
        # catalog has to exist before the first call. The loop always builds it first,
        # but a caller that does not would otherwise silently send no headers and hit
        # the same 400 this method exists to prevent.
        if self._tools is None:
            await self.list_tools()
        extra = header_params(self._schema_for(qualified_name), arguments)

        try:
            if extra:
                # These headers vary per call, so the pooled client cannot carry them;
                # a short-lived client is the price of correctness here.
                async with httpx2.AsyncClient(
                    headers={**cfg.headers, **extra},
                    timeout=httpx2.Timeout(30.0, read=cfg.timeout or self._timeout),
                    follow_redirects=True,
                ) as http:
                    result = await self._invoke(cfg.url, http, tool, arguments)
            else:
                http = self._http.get(server)
                if http is None:
                    raise MCPError("MCPPool.start() was not called")
                result = await self._invoke(cfg.url, http, tool, arguments)
        except Exception as exc:
            # Transport-level failure. Surface it as a tool result rather than an
            # exception so the loop can let the model react instead of dying.
            detail = explain_exception(exc)
            logger.warning("mcp call %s failed: %s", qualified_name, detail)
            return ToolResult(ok=False, text=f"tool transport error: {detail}")

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
