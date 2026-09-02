"""Run the agent named by AGENT_PROFILE as an MCP server: `python -m mcp_agent`.

Same image, same loop, same policy as the Telegram agent — only the front door differs.
`python -m agentcore` puts a Telegram bot in front of the loop; this puts an MCP tool in
front of it.

Two consequences of having no Telegram, both deliberate:

  * There is nobody to ask, so `approver` stays at its deny-by-default. A profile whose
    write mode is `approve` will simply be refused here. Give a worker agent a profile
    whose tools it may use outright, and leave the confirmable work to the agent that
    can actually show a button.
  * Each call is a fresh conversation. The caller passes whatever context matters in the
    task text. Sharing memory between the lead and a worker would mean deciding whose
    turn owns it, and a stateless worker sidesteps that entirely.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from agentcore.audit import AuditLog, configure_logging
from agentcore.config import Settings, get_settings
from agentcore.llm.router import build_llm
from agentcore.loop import AgentLoop
from agentcore.mcp.client import MCPPool, MCPServerConfig
from agentcore.memory import ChatMemory
from agentcore.policy import Policy
from agentcore.profiles import AgentProfile, ProfileNotFound, load_profile
from mcp_common import build_server, serve
from mcp_common.errors import tool_error

logger = logging.getLogger(__name__)


def build_mcp_servers(settings: Settings, profile_servers: list[str]) -> list[MCPServerConfig]:
    """Wire up the MCP servers this profile asks for and the deployment provides.

    Deliberately the same shape as agentcore.__main__: a worker agent reaches its tools
    exactly the way the Telegram agent does.
    """
    endpoints = settings.mcp_endpoints()
    servers: list[MCPServerConfig] = []

    for name in profile_servers:
        url = endpoints.get(name)
        if not url:
            logger.warning("profile wants MCP server %r but no URL is configured; skipping", name)
            continue

        headers: dict[str, str] = {}
        if name == "github":
            # The official GitHub MCP server in http mode has no token setting and
            # returns 401 without this header on every request.
            if not settings.github_pat:
                logger.error("mcp-github is configured but GITHUB_PAT is empty; it will 401")
            else:
                headers["Authorization"] = f"Bearer {settings.github_pat}"

        servers.append(
            MCPServerConfig(
                name=name,
                url=url,
                headers=headers,
                # A delegated agent answers only after running its own loop, so the
                # ordinary tool timeout would cut it off mid-task.
                timeout=settings.agent_tool_timeout_seconds if name == "coder" else None,
            )
        )

    return servers


class AgentService:
    """Holds the loop and starts its MCP pool on first use.

    Lazily, because `serve()` hands control to uvicorn, which owns the event loop. The
    pool has to be created inside that loop rather than before it exists.
    """

    def __init__(self, settings: Settings, profile: AgentProfile) -> None:
        self._settings = settings
        self._profile = profile
        self._pool: MCPPool | None = None
        self._loop: AgentLoop | None = None
        self._lock = asyncio.Lock()
        self._calls = 0

    @property
    def profile(self) -> AgentProfile:
        return self._profile

    async def _ready(self) -> AgentLoop:
        async with self._lock:
            if self._loop is None:
                pool = MCPPool(
                    build_mcp_servers(self._settings, self._profile.mcp_servers),
                    timeout=self._settings.tool_timeout_seconds,
                )
                await pool.start()
                self._pool = pool
                self._loop = AgentLoop(
                    profile=self._profile,
                    settings=self._settings,
                    llm=build_llm(self._settings),
                    pool=pool,
                    policy=Policy(self._settings),
                )
                logger.info(
                    "agent %s ready as an MCP server, servers=%s",
                    self._profile.id,
                    self._profile.mcp_servers,
                )
            return self._loop

    async def run(self, task: str, context: str) -> dict[str, Any]:
        loop = await self._ready()
        self._calls += 1
        # A synthetic chat id per call keeps audit lines attributable while keeping the
        # conversation itself throwaway.
        audit = AuditLog(agent=self._profile.id, chat_id=f"mcp-{self._calls}")

        prompt = f"{context.strip()}\n\n---\n\n{task.strip()}" if context.strip() else task.strip()

        result = await loop.run(
            chat=ChatMemory(chat_id=self._calls),
            user_text=prompt,
            model=self._profile.model or self._settings.model_default,
            audit=audit,
        )
        return {
            "ok": result.stopped_because == "completed",
            "answer": result.text,
            "steps": result.steps,
            "stopped_because": result.stopped_because,
            "billable_tokens": result.usage.billable,
        }


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)

    try:
        profile = load_profile(settings.agent_profile, settings.agents_dir)
    except ProfileNotFound as exc:
        raise SystemExit(str(exc)) from exc

    service = AgentService(settings, profile)
    server = build_server(f"agent-{profile.id}", "0.1.0")

    @server.tool(
        name="ask",
        description=(
            f"Delegate a task to the {profile.name} agent. {profile.description} "
            "It runs its own tool loop and returns when finished, which can take minutes "
            "on a real task — send one complete, self-contained instruction rather than a "
            "conversation. It shares no memory with you: everything it needs to know must "
            "be in `task` and `context`. It cannot ask a human for confirmation, so do not "
            "hand it work that needs approval."
        ),
    )
    async def ask(task: str, context: str = "") -> dict[str, Any]:
        if not task.strip():
            return tool_error("task пустой — опиши, что нужно сделать")
        try:
            return await service.run(task, context)
        except Exception as exc:  # a failed delegation must not take down the server
            logger.exception("delegated task failed")
            return tool_error(f"{type(exc).__name__}: {exc}")

    serve(server, port=settings.health_port)


if __name__ == "__main__":
    main()
