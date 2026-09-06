"""Entrypoint for an agent process: `python -m agentcore`.

Which agent this is comes from AGENT_PROFILE. The same image also ships the MCP
servers (`python -m mcp_cloudflare`, `python -m mcp_cluster`); the Deployment picks a
role via `command:`.

The bot and the health server share one event loop on purpose — see agentcore.health
for why running health in a separate thread makes readiness meaningless.
"""

from __future__ import annotations

import asyncio
import logging
import sys

from agentcore.audit import configure_logging
from agentcore.config import Settings, get_settings
from agentcore.health import build_health_app, build_health_server
from agentcore.llm.router import build_llm
from agentcore.loop import AgentLoop
from agentcore.mcp.client import MCPPool, MCPServerConfig
from agentcore.memory import MemoryStore
from agentcore.store import AdbStore
from agentcore.policy import Policy
from agentcore.profiles import ProfileNotFound, load_profile
from agentcore.ui import TelegramUI

logger = logging.getLogger(__name__)


def build_mcp_servers(settings: Settings, profile_servers: list[str]) -> list[MCPServerConfig]:
    """Wire up the MCP servers this profile asks for and the deployment provides."""
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
            # returns 401 without this header on *every* request.
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


async def amain() -> int:
    settings = get_settings()
    configure_logging(settings.log_level)

    try:
        profile = load_profile(settings.agent_profile, settings.agents_dir)
    except ProfileNotFound as exc:
        logger.error("%s", exc)
        return 2

    if not settings.telegram_bot_token:
        logger.error("TELEGRAM_BOT_TOKEN is not set")
        return 2
    if not settings.allowed_user_ids():
        # Fail-closed would refuse everyone anyway; say so loudly instead of silently
        # running a bot nobody can talk to.
        logger.error("TELEGRAM_ALLOWED_USERS is empty; the bot would refuse every user")
        return 2

    llm = build_llm(settings)

    pool = MCPPool(
        build_mcp_servers(settings, profile.mcp_servers),
        timeout=settings.tool_timeout_seconds,
    )
    await pool.start()

    agent_loop = AgentLoop(
        profile=profile,
        settings=settings,
        llm=llm,
        pool=pool,
        policy=Policy(settings),
    )
    # Durable memory, if it is configured. Constructed rather than connected:
    # there is no I/O here, so an unreachable database shows up as a warning on
    # the first turn instead of a pod that will not start. An agent that forgets
    # is worth more than one that refuses to run.
    store = None
    if settings.memory_enabled():
        store = AdbStore(
            base_url=settings.adb_sql_url,
            username=settings.adb_username,
            password=settings.adb_password,
            timeout_seconds=settings.adb_timeout_seconds,
        )
        logger.info("durable memory enabled as %s", settings.adb_username)
    else:
        logger.warning(
            "durable memory is not configured; history will be lost on restart"
        )

    ui = TelegramUI(
        settings=settings,
        profile=profile,
        agent_loop=agent_loop,
        memory=MemoryStore(),
        store=store,
    )

    app = ui.build_application()
    await app.initialize()
    await app.start()
    await app.updater.start_polling(drop_pending_updates=True)
    logger.info("agent %s started, polling telegram", profile.id)

    def readiness() -> tuple[bool, str]:
        updater = app.updater
        if updater is None or not updater.running:
            return False, "telegram updater is not polling"
        if not app.running:
            return False, "application is not running"
        return True, "polling"

    health = build_health_server(
        build_health_app(agent=profile.id, readiness=readiness),
        port=settings.health_port,
    )

    try:
        # Returns on SIGTERM/SIGINT — uvicorn installs the handlers.
        await health.serve()
    finally:
        logger.info("shutting down")
        await app.updater.stop()
        await app.stop()
        await app.shutdown()
        await pool.close()

    return 0


def main() -> None:
    try:
        sys.exit(asyncio.run(amain()))
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
