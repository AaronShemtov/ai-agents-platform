"""What the person sees when they ask what the bot keeps about them: /memories in the
chat, /export as a file.

Worth pinning: that nothing stored is left out (facts, every conversation, every row,
tool calls and results included), that it is in an order a person can follow, that it
says how to delete each part, and that a secret in it comes back redacted.
"""

from __future__ import annotations

import asyncio
import inspect
import json
from typing import Any

from agentcore import export
from agentcore.store import AdbStore, ConversationSummary, Fact

GROUP = -1001234567890
PRIVATE = 1234567
GITHUB_TOKEN = "ghp_" + "B" * 36  # a token's shape, not a token


def conv(chat_id: int, thread_id: int, first_id: int, **kw: Any) -> ConversationSummary:
    base = {
        "messages": 3,
        "first_day": "2026-10-01",
        "last_at": "2026-10-11 21:42",
        "first_user": "",
        "last_user": "",
    }
    base.update(kw)
    return ConversationSummary(chat_id=chat_id, thread_id=thread_id, first_id=first_id, **base)


class FakeStore:
    def __init__(self, facts=(), listing=(), rows=None):
        self.facts_known = list(facts)
        self.listing = list(listing)
        self.rows = rows or {}
        self.dumped: list[tuple[int, int]] = []

    async def facts(self, *, scope=None):
        return list(self.facts_known)

    async def conversations(self, *, agent):
        return list(self.listing)

    async def dump(self, *, agent, chat_id, thread_id, page=200):
        self.dumped.append((chat_id, thread_id))
        rows = self.rows.get((chat_id, thread_id), [])
        for start in range(0, len(rows), page):
            yield rows[start : start + page]


def listing() -> list[ConversationSummary]:
    """Newest first, as the store lists them."""
    return [
        conv(GROUP, 78, 1852, messages=4, last_user="загадал число 7"),
        conv(GROUP, 0, 341, messages=471, last_user="команды linux"),
        conv(PRIVATE, 0, 21, messages=505, last_user="what is the state of my cluster"),
    ]


# -- /memories ------------------------------------------------------------------


def test_the_overview_lists_every_fact_with_how_to_delete_it() -> None:
    facts = [Fact(key="git-identity", fact="Коммитит как A", scope="user", updated_at="2026-10-17")]
    out = export.overview(facts, listing())
    assert "`git-identity` — Коммитит как A (2026-10-17)" in out
    assert "/forget ключ" in out


def test_with_no_facts_the_overview_still_shows_the_conversations() -> None:
    """It used to answer "nothing is recorded about you" over a thousand messages."""
    out = export.overview([], listing())
    assert "пока ни одного" in out
    assert "**Разговоры** — 3, в них 980 записей" in out
    assert "[тема 78](https://t.me/c/1234567890/78) — 4 записи" in out
    assert "General — 471 запись" in out
    assert "личный чат — 505 записей" in out


def test_the_overview_puts_conversations_in_the_order_they_began() -> None:
    out = export.overview([], listing())
    assert out.index("личный чат") < out.index("General") < out.index("тема 78")


def test_general_is_said_to_hold_everything_from_before_the_split() -> None:
    out = export.overview([], listing())
    general = next(line for line in out.splitlines() if line.startswith("• General"))
    assert "до разделения по темам" in general


def test_the_overview_says_how_to_clear_a_conversation_and_get_everything() -> None:
    out = export.overview([], listing())
    assert "/new" in out
    assert "/export" in out


def test_a_secret_in_the_latest_message_does_not_reach_the_overview() -> None:
    out = export.overview([], [conv(PRIVATE, 0, 21, last_user=f"токен {GITHUB_TOKEN}")])
    assert GITHUB_TOKEN not in out


def test_an_empty_memory_says_so() -> None:
    out = export.overview([], [])
    assert "в базе ничего нет" in out


# -- /export --------------------------------------------------------------------


def rows() -> dict[tuple[int, int], list[dict[str, Any]]]:
    return {
        (PRIVATE, 0): [
            {"id": 21, "role": "user", "at": "2026-09-06 10:00", "content": "привет", "tool_calls": None},
            {
                "id": 22,
                "role": "assistant",
                "at": "2026-09-06 10:00",
                "content": None,
                "tool_calls": json.dumps(
                    [{"id": "c1", "type": "function",
                      "function": {"name": "cluster__get_pods", "arguments": "{\"ns\": \"agents\"}"}}]
                ),
            },
            {"id": 23, "role": "tool", "at": "2026-09-06 10:00", "content": f"pods ok {GITHUB_TOKEN}", "tool_calls": None},
            {"id": 24, "role": "assistant", "at": "2026-09-06 10:01", "content": "всё работает", "tool_calls": None},
        ],
        (GROUP, 0): [{"id": 341, "role": "user", "at": "2026-09-07 09:00", "content": "в группе", "tool_calls": None}],
        (GROUP, 78): [{"id": 1852, "role": "user", "at": "2026-10-11 21:41", "content": "в теме", "tool_calls": None}],
    }


def run_export(store: FakeStore) -> tuple[str, int, int]:
    return asyncio.run(export.export_text(store, agent="lead", now="2026-10-17 20:00"))


def test_the_file_holds_every_conversation_and_every_row() -> None:
    store = FakeStore(listing=listing(), rows=rows())
    text, conversations, written = run_export(store)
    assert (conversations, written) == (3, 6)
    for marker in ("[21]", "[22]", "[23]", "[24]", "[341]", "[1852]"):
        assert marker in text
    assert sorted(store.dumped) == sorted([(PRIVATE, 0), (GROUP, 0), (GROUP, 78)])


def test_the_file_is_in_the_order_conversations_began_and_rows_were_written() -> None:
    text, _, _ = run_export(FakeStore(listing=listing(), rows=rows()))
    assert text.index("===== личный чат") < text.index("===== General") < text.index("===== тема 78")
    assert text.index("[21]") < text.index("[22]") < text.index("[23]") < text.index("[24]")


def test_the_file_shows_who_said_it_and_what_tools_were_called() -> None:
    text, _, _ = run_export(FakeStore(listing=listing(), rows=rows()))
    assert "[21] 2026-09-06 10:00 · ты:\nпривет" in text
    assert '→ вызов cluster__get_pods {"ns": "agents"}' in text
    assert "[23] 2026-09-06 10:00 · результат инструмента:" in text
    assert "[24] 2026-09-06 10:01 · бот:\nвсё работает" in text


def test_the_file_starts_with_the_facts() -> None:
    facts = [Fact(key="git-identity", fact="Коммитит как A", scope="user", updated_at="2026-10-17")]
    text, _, _ = run_export(FakeStore(facts=facts, listing=listing(), rows=rows()))
    assert text.index("[git-identity] Коммитит как A") < text.index("===== личный чат")


def test_a_secret_a_tool_read_does_not_leave_in_the_file() -> None:
    text, _, _ = run_export(FakeStore(listing=listing(), rows=rows()))
    assert GITHUB_TOKEN not in text
    assert "pods ok <REDACTED>" in text


def test_a_file_from_an_empty_memory_says_so_rather_than_failing() -> None:
    text, conversations, written = run_export(FakeStore())
    assert (conversations, written) == (0, 0)
    assert "Фактов: 0. Разговоров: 0, записей всего: 0." in text


def test_the_fake_store_has_the_real_stores_signatures() -> None:
    for name in ("facts", "conversations", "dump"):
        real = list(inspect.signature(getattr(AdbStore, name)).parameters)
        fake = list(inspect.signature(getattr(FakeStore, name)).parameters)
        assert fake == real, name


def test_a_count_is_said_the_way_it_is_said() -> None:
    assert [export.records(n) for n in (1, 2, 4, 5, 11, 12, 21, 22, 25, 101, 111, 1084)] == [
        "1 запись", "2 записи", "4 записи", "5 записей", "11 записей", "12 записей",
        "21 запись", "22 записи", "25 записей", "101 запись", "111 записей", "1084 записи",
    ]
