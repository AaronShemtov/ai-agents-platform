"""Everything the agent keeps about the person, shown back to them.

Two views of the same rows:

  * /memories — an overview in the chat: the facts, which go into every prompt, and
    the conversations, with how many records each holds and when it ran;
  * /export — all of it as a text file, every stored row of every conversation, tool
    calls and their results included, sent to the private chat.

Keeping memory in a database rather than in the model was meant to make it possible to
read it back and delete it. /forget and /new delete; these are the reading half. The
person asked for exactly that: to see everything the bot knows about them.

Both pass audit.redact_secrets. The file goes to the private chat and nowhere else, but
it carries tool results the person never saw in Telegram, and a token a tool read is
not something to hand out as a file.
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from agentcore.audit import redact_secrets
from agentcore.recall import Atlas, handle, one_line, topic_link
from agentcore.store import AdbStore, ConversationSummary, Fact

OVERVIEW_PREVIEW = 80

ROLE = {"user": "ты", "assistant": "бот", "tool": "результат инструмента"}


def records(n: int) -> str:
    """"4 записи", "11 записей", "21 запись": the count the way it is said."""
    if n % 10 == 1 and n % 100 != 11:
        word = "запись"
    elif n % 10 in (2, 3, 4) and n % 100 not in (12, 13, 14):
        word = "записи"
    else:
        word = "записей"
    return f"{n} {word}"


def _in_order(conversations: Sequence[ConversationSummary]) -> list[ConversationSummary]:
    """The order they appeared in, as the index in the prompt has them."""
    return sorted(conversations, key=lambda c: (c.first_id, c.chat_id, c.thread_id))


def overview(facts: Sequence[Fact], conversations: Sequence[ConversationSummary]) -> str:
    """Markdown for /memories: every fact, every conversation, and how to remove each."""
    sections = ["**Что я о тебе храню**"]

    if facts:
        lines = [f"**Факты** ({len(facts)}) — идут в каждый мой ответ:"]
        lines += [
            f"• `{f.key}` — {f.fact}" + (f" ({f.updated_at})" if f.updated_at else "")
            for f in facts
        ]
        lines.append("Удалить факт: `/forget ключ`.")
    else:
        lines = [
            "**Факты**: пока ни одного. Я сохраняю их сам, когда узнаю о тебе что-то "
            "устойчивое; можно и прямо сказать «запомни …»."
        ]
    sections.append("\n".join(lines))

    if conversations:
        atlas = Atlas(conversations)
        total = sum(c.messages for c in conversations)
        lines = [
            f"**Разговоры** — {len(conversations)}, в них {records(total)}: твои сообщения, "
            "мои ответы, вызовы инструментов и их результаты."
        ]
        for c in _in_order(conversations):
            name = atlas.label(c.chat_id, c.thread_id)
            link = topic_link(c.chat_id, c.thread_id)
            line = (
                f"• {f'[{name}]({link})' if link else name} — {records(c.messages)}, "
                f"{c.first_day} … {c.last_at[:10]}"
            )
            if c.last_user:
                last = one_line(c.last_user, OVERVIEW_PREVIEW, cut=c.last_user_cut)
                line += f"; последнее твоё: «{last}»"
            if atlas.before_split(c.chat_id, c.thread_id, c.first_id):
                line += " (здесь же всё, что было до разделения по темам)"
            lines.append(line)
        lines.append("Очистить разговор: `/new` в нём.")
    else:
        lines = ["**Разговоры**: в базе ничего нет."]
    sections.append("\n".join(lines))

    sections.append(
        "Всё целиком, каждое сообщение, — файлом: /export. Пришлю в личку: в группе "
        "его увидели бы все."
    )
    return "\n\n".join(sections)


def _row(row: dict) -> str:
    """One stored row, as the file shows it."""
    role = row.get("role") or "?"
    head = f"[{row.get('id')}] {row.get('at') or ''} · {ROLE.get(role, role)}"
    parts = []
    if row.get("content"):
        parts.append(str(row["content"]))
    if row.get("tool_calls"):
        try:
            calls = json.loads(row["tool_calls"])
        except (TypeError, ValueError):
            calls = []
        for call in calls if isinstance(calls, list) else []:
            function = call.get("function") or {}
            parts.append(f"→ вызов {function.get('name', '?')} {function.get('arguments', '')}")
    return head + ":\n" + ("\n".join(parts) if parts else "(пусто)")


async def export_text(store: AdbStore, *, agent: str, now: str) -> tuple[str, int, int]:
    """The whole file, and how many conversations and rows it holds.

    Raises StoreUnavailable when the database does not answer: a file that silently
    stops halfway would read as the whole of it.
    """
    facts = await store.facts()
    conversations = await store.conversations(agent=agent)
    atlas = Atlas(conversations)
    total = sum(c.messages for c in conversations)
    out = [
        f"Что агент {agent} хранит о тебе — выгрузка {now} UTC.",
        "Oracle Autonomous Database, схема AGENTS. Пароли и токены заменены на <REDACTED>.",
        f"Фактов: {len(facts)}. Разговоров: {len(conversations)}, записей всего: {total}.",
        "",
        "===== Факты: идут в каждый ответ =====",
    ]
    out += [f"[{f.key}] {f.fact} ({f.updated_at})" for f in facts] or ["(нет)"]
    rows_written = 0
    for c in _in_order(conversations):
        link = topic_link(c.chat_id, c.thread_id)
        title = atlas.label(c.chat_id, c.thread_id)
        out += [
            "",
            f"===== {title}{f' {link}' if link else ''} · id {handle(c.chat_id, c.thread_id)} "
            f"· {records(c.messages)} · {c.first_day} … {c.last_at} =====",
        ]
        if atlas.before_split(c.chat_id, c.thread_id, c.first_id):
            out.append(
                "Здесь же всё, что было записано до разделения по темам, из всех тем вместе."
            )
        async for rows in store.dump(agent=agent, chat_id=c.chat_id, thread_id=c.thread_id):
            for row in rows:
                out += ["", _row(row)]
                rows_written += 1
    out += ["", f"===== Конец: {records(rows_written)} ====="]
    return redact_secrets("\n".join(out)), len(conversations), rows_written
