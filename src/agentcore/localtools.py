"""Tools the agent runs in this process, with no MCP server behind them.

Every other tool reaches a server over HTTP. Memory is different: the store is
already open here, holding a connection this process owns, and putting it behind
an MCP server would mean another Deployment, another copy of the database
password and a network hop to reach a table this process can already write.

They are still namespaced (`memory__remember`) and still go through the catalogue,
the profile's allow/deny globs and the policy, so from every other angle in the
loop they are ordinary tools. Only the last step differs — dispatch runs a
coroutine instead of a request.

Why tools rather than a command: a fact worth keeping usually turns up in the
middle of a sentence about something else, and a person should not have to
notice that and retype it as `/remember`. The model is already reading the
sentence; letting it call this is the difference between memory that happens and
memory that has to be operated.
"""

from __future__ import annotations

import logging

from agentcore.mcp.client import RemoteTool, ToolResult, split_qualified
from agentcore.store import AdbStore, StoreUnavailable

logger = logging.getLogger(__name__)

MEMORY_SERVER = "memory"
REMEMBER = f"{MEMORY_SERVER}__remember"
FORGET = f"{MEMORY_SERVER}__forget"

# A key is how a fact is replaced later and how a person names it to `/forget`,
# so it has to be stable and typeable — not a sentence, not a uuid.
_KEY_RULE = (
    "Короткий устойчивый ключ на латинице через дефис, например "
    "'git-identity' или 'deploy-process'. По ключу факт потом заменяется на "
    "новый и удаляется по просьбе, поэтому для одной и той же темы ключ должен "
    "быть один и тот же."
)

MEMORY_TOOLS: tuple[RemoteTool, ...] = (
    RemoteTool(
        server=MEMORY_SERVER,
        name="remember",
        qualified_name=REMEMBER,
        description=(
            "Запомнить факт о человеке или его проектах надолго — так, чтобы он "
            "был известен и в следующих разговорах, после перезапуска. "
            "Вызывай сам, без просьбы, когда узнаёшь устойчивое: как его зовут, "
            "чем он занимается, как у него принято работать, чего он просил не "
            "делать. Повторный вызов с тем же ключом заменяет факт — так его и "
            "надо исправлять, когда узнал точнее. Не надо запоминать детали "
            "текущей задачи: они и так в истории разговора."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "key": {"type": "string", "description": _KEY_RULE},
                "fact": {
                    "type": "string",
                    "description": (
                        "Сам факт, одной фразой, в третьем лице и понятный без "
                        "контекста этого разговора."
                    ),
                },
            },
            "required": ["key", "fact"],
        },
    ),
    RemoteTool(
        server=MEMORY_SERVER,
        name="forget",
        qualified_name=FORGET,
        description=(
            "Забыть один ранее запомненный факт по его ключу. Вызывай, когда "
            "человек просит забыть, или когда факт перестал быть верным и "
            "заменить его нечем. Если факт устарел, но тема осталась — лучше "
            "remember с тем же ключом."
        ),
        input_schema={
            "type": "object",
            "properties": {"key": {"type": "string", "description": _KEY_RULE}},
            "required": ["key"],
        },
    ),
)


def is_local(qualified_name: str) -> bool:
    """Whether this name dispatches in-process rather than to an MCP server."""
    try:
        server, _ = split_qualified(qualified_name)
    except Exception:
        return False
    return server == MEMORY_SERVER


def memory_tools(store: AdbStore | None) -> list[RemoteTool]:
    """The memory tools, or none at all when there is nowhere to write.

    Offering a tool that cannot work is worse than offering nothing: the model
    would spend a step calling it and a step reading the failure.
    """
    return list(MEMORY_TOOLS) if store is not None else []


async def call_local(
    qualified_name: str,
    arguments: dict,
    *,
    store: AdbStore | None,
    chat_id: int | None = None,
) -> ToolResult:
    """Run one in-process tool.

    Failures come back as `ok=False` rather than raising. The store raises on a
    failed write on purpose — a caller must not report a fact as kept when it was
    not — and this is where that becomes something the model can act on: it sees
    the failure in the transcript and tells the person, instead of the whole turn
    dying over one fact.
    """
    if store is None:
        return ToolResult(ok=False, text="память отключена: писать некуда")

    _, bare = split_qualified(qualified_name)
    key = str(arguments.get("key") or "").strip()
    if not key:
        return ToolResult(ok=False, text="нужен key — короткий ключ факта")

    try:
        if bare == "remember":
            fact = str(arguments.get("fact") or "").strip()
            if not fact:
                return ToolResult(ok=False, text="нужен fact — сам факт одной фразой")
            await store.remember(key=key, fact=fact, chat_id=chat_id)
            return ToolResult(ok=True, text=f"запомнено под ключом {key}")

        if bare == "forget":
            removed = await store.forget(key)
            return ToolResult(
                ok=True,
                text=(
                    f"забыто: {key}"
                    if removed
                    else f"нечего забывать: факта с ключом {key} не было"
                ),
            )
    except StoreUnavailable as exc:
        logger.warning("local tool %s failed: %s", qualified_name, exc)
        return ToolResult(ok=False, text=f"база памяти недоступна, факт не сохранён: {exc}")

    return ToolResult(ok=False, text=f"нет такого локального инструмента: {qualified_name}")
