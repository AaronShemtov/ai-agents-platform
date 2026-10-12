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

The other three read the agent's other conversations — the second priority behind
the one it is in; recall.py says why they are tools rather than history. They only
read, and only this agent's own rows.
"""

from __future__ import annotations

import logging
from typing import Any

from agentcore import recall
from agentcore.mcp.client import RemoteTool, ToolResult, split_qualified
from agentcore.memory import ConversationKey
from agentcore.store import AdbStore, StoreUnavailable

logger = logging.getLogger(__name__)

MEMORY_SERVER = "memory"
REMEMBER = f"{MEMORY_SERVER}__remember"
FORGET = f"{MEMORY_SERVER}__forget"
LIST_CONVERSATIONS = f"{MEMORY_SERVER}__list_conversations"
SEARCH_CONVERSATIONS = f"{MEMORY_SERVER}__search_conversations"
READ_CONVERSATION = f"{MEMORY_SERVER}__read_conversation"

# Sizes of what the reading tools hand back. A tool result is clipped at
# loop.MAX_TOOL_RESULT_CHARS, 12,000, and these keep a full answer under it.
SEARCH_DEFAULT, SEARCH_MAX = 8, 20
READ_DEFAULT, READ_MAX = 12, 30
READ_TEXT = 8_000  # spread over the messages of one read
READ_FOCUS = 2_500  # the message a search pointed at
READ_ALONE = 6_000  # that message when it is the only one asked for

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
    RemoteTool(
        server=MEMORY_SERVER,
        name="search_conversations",
        qualified_name=SEARCH_CONVERSATIONS,
        description=(
            "Найти, где в разговорах с человеком встречались слова: во всех темах "
            "группы, в General, в личном чате и в старой части этого разговора, "
            "которой нет в истории выше. Ищется часть слова без учёта регистра, "
            "поэтому давай основы без окончаний: «миграц», «кластер», «oracle». "
            "Находит сообщения, где есть хоть одно из слов; выше те, где их больше, "
            "при равенстве — свежие. Возвращает разговор, номер сообщения, дату и "
            "отрывок вокруг совпадения; дальше читай memory__read_conversation."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Слова через пробел, до шести, каждое от двух букв.",
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Сколько сообщений вернуть: по умолчанию {SEARCH_DEFAULT}, "
                        f"не больше {SEARCH_MAX}."
                    ),
                },
            },
            "required": ["query"],
        },
    ),
    RemoteTool(
        server=MEMORY_SERVER,
        name="read_conversation",
        qualified_name=READ_CONVERSATION,
        description=(
            "Прочитать разговор с человеком: его последние сообщения или кусок вокруг "
            "сообщения, найденного memory__search_conversations. Только реплики "
            "человека и твои ответы, без вызовов инструментов; длинные сообщения "
            "обрезаны, и сказано, сколько осталось."
        ),
        input_schema={
            "type": "object",
            "properties": {
                "conversation": {
                    "type": "string",
                    "description": (
                        "id разговора из поиска или memory__list_conversations, вида "
                        "«-1001234567890/78». Для темы той же группы хватит номера "
                        "темы, «78»; подойдёт и ссылка на тему."
                    ),
                },
                "around": {
                    "type": "integer",
                    "description": (
                        "Номер сообщения из поиска: покажу его и соседние. Без него — "
                        "последние сообщения разговора."
                    ),
                },
                "limit": {
                    "type": "integer",
                    "description": (
                        f"Сколько сообщений: по умолчанию {READ_DEFAULT}, не больше "
                        f"{READ_MAX}. 1 с around — это сообщение длиннее, до "
                        f"{READ_ALONE} символов."
                    ),
                },
            },
            "required": ["conversation"],
        },
    ),
    RemoteTool(
        server=MEMORY_SERVER,
        name="list_conversations",
        qualified_name=LIST_CONVERSATIONS,
        description=(
            "Перечислить все разговоры с человеком — темы группы, General, личный "
            "чат: сколько в каждом сообщений, когда он шёл, с чего начался и чем "
            "закончился. Нужен, чтобы понять, где что обсуждали, и взять id "
            "разговора для memory__read_conversation."
        ),
        input_schema={"type": "object", "properties": {}},
    ),
)

# The ones that read conversations rather than facts, by their bare names.
_RECALL = frozenset({"search_conversations", "read_conversation", "list_conversations"})


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
    agent: str | None = None,
    chat_id: int | None = None,
    thread_id: int = 0,
) -> ToolResult:
    """Run one in-process tool.

    Failures come back as `ok=False` rather than raising. The store raises on a
    failed write on purpose — a caller must not report a fact as kept when it was
    not — and this is where that becomes something the model can act on: it sees
    the failure in the transcript and tells the person, instead of the whole turn
    dying over one fact.

    `agent`, `chat_id` and `thread_id` are the conversation the turn is in. The facts
    keep the chat as where a fact came from; the reading tools need all three, to read
    this agent's rows only and to mark this conversation among the others.
    """
    if store is None:
        return ToolResult(ok=False, text="память отключена: базы нет, ни записать, ни прочитать")

    _, bare = split_qualified(qualified_name)
    if bare in _RECALL:
        return await _recall(
            bare, arguments, store=store, agent=agent, chat_id=chat_id, thread_id=thread_id
        )

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


def _number(value: Any, *, default: int, most: int) -> int:
    """A count the model asked for, kept between 1 and `most`. Whatever is not a
    number gets the default: a wrong count is not worth a failed call."""
    try:
        wanted = int(value)
    except (TypeError, ValueError):
        return default
    return max(1, min(wanted, most))


def _message_id(value: Any) -> int | None:
    """`around` as the model wrote it — 1853, 1853.0, "1853" or "[1853]" — or None
    when it wrote none. Anything else raises ValueError."""
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        raise ValueError(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float) and value.is_integer():
        return int(value)
    text = str(value).strip().strip("[]#")
    if not text.isdigit():
        raise ValueError(value)
    return int(text)


async def _recall(
    bare: str,
    arguments: dict,
    *,
    store: AdbStore,
    agent: str | None,
    chat_id: int | None,
    thread_id: int,
) -> ToolResult:
    """The three tools that read conversations: list, search, read."""
    if not agent:
        # Rows belong to an agent; reading without one would read everybody's.
        return ToolResult(ok=False, text="не знаю, чьи разговоры читать: агент не указан")
    current = ConversationKey(chat_id, thread_id) if chat_id is not None else None

    try:
        conversations = await store.conversations(agent=agent)

        if bare == "list_conversations":
            return ToolResult(ok=True, text=recall.format_list(conversations, current))

        if bare == "search_conversations":
            terms = recall.terms_of(str(arguments.get("query") or ""))
            if not terms:
                return ToolResult(
                    ok=False,
                    text="нужен query: слова для поиска, от двух букв, не только союзы и предлоги",
                )
            hits = await store.search(
                agent=agent,
                terms=terms,
                limit=_number(arguments.get("limit"), default=SEARCH_DEFAULT, most=SEARCH_MAX),
            )
            return ToolResult(
                ok=True, text=recall.format_hits(hits, conversations, current, terms)
            )

        # read_conversation
        target = recall.parse_handle(
            str(arguments.get("conversation") or ""), current_chat=chat_id
        )
        if target is None:
            return ToolResult(
                ok=False,
                text=(
                    "нужен conversation: id разговора вида «-1001234567890/78» из "
                    "memory__list_conversations или поиска, номер темы этой группы или "
                    "ссылка на тему"
                ),
            )
        try:
            around = _message_id(arguments.get("around"))
        except ValueError:
            return ToolResult(
                ok=False, text="around — номер сообщения из результатов поиска, число"
            )
        limit = _number(arguments.get("limit"), default=READ_DEFAULT, most=READ_MAX)
        messages = await store.read(
            agent=agent,
            chat_id=target[0],
            thread_id=target[1],
            around=around,
            limit=limit,
            width=max(250, min(1_500, READ_TEXT // limit)),
            focus_width=READ_ALONE if limit == 1 else READ_FOCUS,
        )
        return ToolResult(
            ok=True,
            text=recall.format_read(
                messages,
                chat_id=target[0],
                thread_id=target[1],
                conversations=conversations,
                current=current,
                around=around,
            ),
        )
    except StoreUnavailable as exc:
        logger.warning("local tool memory__%s failed: %s", bare, exc)
        # Said as a failure, never as an empty result: "nothing was found" is something
        # the model would repeat to the person as a fact about the past.
        return ToolResult(
            ok=False, text=f"база памяти недоступна, прочитать разговоры не вышло: {exc}"
        )
