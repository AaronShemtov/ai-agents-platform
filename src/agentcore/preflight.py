"""Check a model before making it the default: `python -m agentcore.preflight <model>`.

Changing MODEL_DEFAULT has broken this agent three times, each time silently and
each time in a way that "does it answer?" would not have caught:

  * astra with tools on /chat/completions — HTTP 400 on the first turn that
    needed a tool, which is every turn;
  * luna replaying stored history — HTTP 400 "messages with role 'tool' must be
    a response to a preceeding message", on every message, permanently, because
    the Responses API had been tolerating a window that opened mid-turn;
  * luna and web access — no error at all. Hosted tools exist only on
    /responses, so the agent quietly stopped reading pages and started answering
    from memory, which for a famous URL looks right and for a news article is a
    confident fabrication.

The common thread is that the endpoint a model is routed to decides which
capabilities exist, and each failure was in a capability nobody thought to test.
So this exercises all three against the live deployment: the real stored
conversation, the real tool catalogue, the real internet.

Exit status is 0 only if nothing failed, so it can gate a change rather than
merely inform one.
"""

from __future__ import annotations

import asyncio
import sys
import time
from dataclasses import dataclass

from agentcore.__main__ import build_mcp_servers
from agentcore.config import Settings, get_settings
from agentcore.llm.router import build_llm
from agentcore.mcp.client import MCPPool
from agentcore.mcp.toolset import build_catalog
from agentcore.memory import ChatMemory
from agentcore.profiles import load_profile
from agentcore.store import AdbStore

PASS, FAIL, SKIP = "PASS", "FAIL", "SKIP"


@dataclass
class Check:
    name: str
    status: str
    detail: str

    def line(self) -> str:
        return f"  {self.name:<16} {self.status:<5} {self.detail}"


async def _busiest_chat(store: AdbStore, agent: str) -> int | None:
    """The conversation most likely to expose a replay problem: the longest one."""
    rows = await store._run(
        "select chat_id, count(*) as n from conversations "
        "where agent = :agent group by chat_id order by n desc fetch first 1 rows only",
        {"agent": agent},
    )
    return int(rows[0]["chat_id"]) if rows else None


async def check_history(settings: Settings, model: str, profile) -> Check:
    """Replay what is actually stored. A window that opens mid-turn is a 400."""
    if not settings.memory_enabled():
        return Check("history replay", SKIP, "no durable memory configured")

    store = AdbStore(
        base_url=settings.adb_sql_url,
        username=settings.adb_username,
        password=settings.adb_password,
        timeout_seconds=settings.adb_timeout_seconds,
    )
    chat_id = await _busiest_chat(store, profile.id)
    if chat_id is None:
        return Check("history replay", SKIP, "nothing stored yet")

    chat = ChatMemory(chat_id=chat_id)
    chat.adopt(
        await store.history(
            agent=profile.id, chat_id=chat_id, limit=settings.memory_history_messages
        )
    )
    restored = len(chat.messages)
    chat.append({"role": "user", "content": "Ответь одним словом: готово"})

    local = settings.is_local_model(model)
    prompt = (
        settings.local_system_prompt
        if local and settings.local_system_prompt
        else profile.system_prompt
    )
    budget = settings.local_max_tokens if local else settings.max_tokens_per_turn
    messages = chat.transcript(prompt, max_tokens=budget)

    started = time.monotonic()
    try:
        await build_llm(settings).complete(model=model, messages=messages)
    except Exception as exc:
        return Check(
            "history replay",
            FAIL,
            f"{type(exc).__name__}: {str(exc)[:160]}",
        )
    return Check(
        "history replay",
        PASS,
        f"chat {chat_id}, {restored} restored, {len(messages)} sent, "
        f"{time.monotonic() - started:.1f}s",
    )


async def check_tool_call(settings: Settings, model: str, profile, pool: MCPPool) -> Check:
    """Ask for something only a tool can answer, and see whether one is called."""
    if settings.is_local_model(model) and not settings.local_tools:
        return Check("tool call", SKIP, "local models are given no tools by design")

    catalog = build_catalog(await pool.list_tools(), is_allowed=profile.tool_allowed)
    if not catalog.specs:
        return Check("tool call", SKIP, "no tools in the catalogue")

    try:
        response = await build_llm(settings).complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Сколько подов в namespace agents? Вызови инструмент.",
                }
            ],
            tools=catalog.specs,
        )
    except Exception as exc:
        return Check("tool call", FAIL, f"{type(exc).__name__}: {str(exc)[:160]}")

    if not response.tool_calls:
        return Check(
            "tool call",
            FAIL,
            f"{len(catalog.specs)} tools offered, none called — answered instead",
        )
    return Check(
        "tool call",
        PASS,
        f"{len(catalog.specs)} tools offered, called {response.tool_calls[0].name}",
    )


async def check_page_read(settings: Settings, model: str) -> Check:
    """Read a page. Absence of web_search_call is the silent failure, not an error."""
    if not settings.hosted_tool_specs():
        return Check("page read", SKIP, "HOSTED_TOOLS is empty")
    if settings.is_local_model(model):
        # Not a failure, and not fixable: Ollama serves no /responses at all, and
        # Settings refuses a model listed in both MODELS_OLLAMA and
        # MODELS_RESPONSES_API. Telling anyone to add it to that list would be
        # advice that cannot be followed.
        return Check(
            "page read",
            SKIP,
            "served by Ollama, which has no /responses and so no hosted tools",
        )
    if model not in settings.responses_api_models():
        return Check(
            "page read",
            FAIL,
            "hosted tools exist only on /responses; add this model to "
            "MODELS_RESPONSES_API or it silently answers from memory",
        )

    try:
        response = await build_llm(settings).complete(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": "Открой https://example.com и процитируй его заголовок H1.",
                }
            ],
        )
    except Exception as exc:
        return Check("page read", FAIL, f"{type(exc).__name__}: {str(exc)[:160]}")

    items = [i.get("type") for i in (response.raw_message.get("_raw_output") or [])]
    if "web_search_call" not in items:
        return Check(
            "page read",
            FAIL,
            f"answered without reading: {items or 'no provider items'}",
        )
    return Check("page read", PASS, f"{items.count('web_search_call')} web calls")


def _seconds(detail: str) -> float:
    """Pull the trailing "3.8s" out of a check's detail, if it has one."""
    for word in detail.replace(",", " ").split():
        if word.endswith("s"):
            try:
                return float(word[:-1])
            except ValueError:
                continue
    return 0.0


async def run(model: str) -> int:
    settings = get_settings()
    profile = load_profile(settings.agent_profile, settings.agents_dir)

    local = settings.is_local_model(model)
    endpoint = (
        "ollama"
        if local
        else "/responses"
        if model in settings.responses_api_models()
        else "/chat/completions"
    )
    print(f"preflight for {model} (profile {profile.id}, endpoint {endpoint})")

    pool = MCPPool(
        build_mcp_servers(settings, profile.mcp_servers),
        timeout=settings.tool_timeout_seconds,
    )
    await pool.start()

    checks = [
        await check_history(settings, model, profile),
        await check_tool_call(settings, model, profile, pool),
        await check_page_read(settings, model),
    ]
    for check in checks:
        print(check.line())

    failed = [c for c in checks if c.status == FAIL]
    if failed:
        print(f"\nNOT safe as MODEL_DEFAULT: {', '.join(c.name for c in failed)}")
        return 1

    # "Nothing broke" is not the same as "this is a good default", and a verdict
    # read on its own should not blur them. A skipped check means the capability
    # is absent rather than working, which for a model answering every message
    # is the thing worth knowing.
    caveats = [f"no {c.name}" for c in checks if c.status == SKIP]
    slow = [c.name for c in checks if c.status == PASS and _seconds(c.detail) > 30]
    if slow:
        caveats.append("slow " + " and ".join(slow))

    verdict = "safe as MODEL_DEFAULT"
    if caveats:
        verdict += " — but " + ", ".join(caveats)
    print(f"\n{verdict}")
    return 0


def main() -> None:
    if len(sys.argv) != 2:
        print("usage: python -m agentcore.preflight <model>", file=sys.stderr)
        raise SystemExit(2)
    raise SystemExit(asyncio.run(run(sys.argv[1])))


if __name__ == "__main__":
    main()
