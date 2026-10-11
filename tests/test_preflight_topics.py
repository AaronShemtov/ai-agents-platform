"""The preflight replays the longest conversation, and a conversation is a chat and a topic.

Counting by chat alone would pick the whole forum group and replay a window that no
topic ever sees, so the one check meant to expose a bad replay would test the wrong
thing. CI imports this module and runs nothing in it, which is why it has a test of its own.
"""

from __future__ import annotations

from agentcore.memory import ConversationKey
from agentcore.preflight import _busiest_conversation


class FakeStore:
    def __init__(self, rows: list[dict]) -> None:
        self.rows = rows
        self.statement = ""
        self.binds: dict = {}

    async def _run(self, statement: str, binds: dict | None = None) -> list[dict]:
        self.statement, self.binds = statement, binds or {}
        return self.rows


async def test_the_busiest_conversation_is_a_chat_and_a_topic() -> None:
    store = FakeStore([{"chat_id": -100123, "thread_id": 5, "n": 90}])
    assert await _busiest_conversation(store, "lead") == ConversationKey(-100123, 5)  # type: ignore[arg-type]
    assert "group by chat_id, thread_id" in store.statement
    assert store.binds == {"agent": "lead"}


async def test_an_empty_database_has_no_busiest_conversation() -> None:
    assert await _busiest_conversation(FakeStore([]), "lead") is None  # type: ignore[arg-type]
