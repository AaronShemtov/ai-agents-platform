"""A forum group is one chat with many conversations: the front door tells them apart.

Before this, history was keyed by chat id alone, so every topic of the group replayed
one shared history and the agent answered each of them with the others mixed in. The
key is now a chat and a topic, and these tests drive the real handlers with real
`Update` objects (so python-telegram-bot's own routing into a topic is exercised)
against a fake bot, a fake store and a fake loop.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

import pytest
from telegram import Update

from agentcore import recall
from agentcore.config import Settings
from agentcore.loop import AgentLoop
from agentcore.memory import MemoryStore
from agentcore.store import AdbStore, ConversationSummary, Fact, StoreUnavailable
from agentcore.ui import TelegramUI
from agentcore.ui import telegram as telegram_ui
from agentcore.ui.telegram import TOPIC_NOTE, thread_of

GROUP = -1001234567890  # a forum supergroup, as a chat id looks
PRIVATE = 1234567


class FakeBot:
    """Records what the bot is asked to do. `reply_text` reaches `send_message` here
    through python-telegram-bot, with the thread it chose."""

    defaults = None

    def __init__(self) -> None:
        self.sent: list[dict[str, Any]] = []
        self.deleted: list[dict[str, Any]] = []

    async def send_message(self, **kwargs: Any) -> SimpleNamespace:
        self.sent.append(kwargs)
        return SimpleNamespace(message_id=1000 + len(self.sent))

    async def edit_message_text(self, **kwargs: Any) -> None:
        return None

    async def delete_message(self, **kwargs: Any) -> None:
        self.deleted.append(kwargs)


class FakeStore:
    """The part of AdbStore the front door uses, keyed the way the real one is."""

    def __init__(self) -> None:
        self.rows: dict[tuple[str, int, int], list[dict[str, Any]]] = {}
        self.facts_known: list[Fact] = []
        self.clear_works = True
        self.gate: asyncio.Event | None = None  # holds every history read until set
        self.touched: list[tuple[str, int, int]] = []  # conversations by last write
        self.listing_error: Exception | None = None

    async def history(
        self, *, agent: str, chat_id: int, thread_id: int, limit: int
    ) -> list[dict[str, Any]]:
        if self.gate is not None:
            await self.gate.wait()
        return list(self.rows.get((agent, chat_id, thread_id), []))[-limit:]

    async def append(
        self, *, agent: str, chat_id: int, thread_id: int, message: dict[str, Any]
    ) -> None:
        key = (agent, chat_id, thread_id)
        self.rows.setdefault(key, []).append(message)
        if key in self.touched:
            self.touched.remove(key)
        self.touched.append(key)

    async def clear(self, *, agent: str, chat_id: int, thread_id: int) -> bool:
        if not self.clear_works:
            return False
        key = (agent, chat_id, thread_id)
        self.rows.pop(key, None)
        # A cleared conversation is gone from the listing too. Left in `touched`, the
        # listing raised KeyError on it, and the front door's broad except hid that.
        if key in self.touched:
            self.touched.remove(key)
        return True

    async def facts(self, *, scope: str | None = None) -> list[Fact]:
        return list(self.facts_known)

    async def conversations(self, *, agent: str) -> list[ConversationSummary]:
        """Newest first, as the real one orders them: the ones written to last, then
        the rows a test put there directly."""
        if self.listing_error is not None:
            raise self.listing_error
        keys = [key for key in reversed(self.touched) if key[0] == agent]
        keys += [key for key in self.rows if key[0] == agent and key not in keys]
        listing = []
        for key in keys:
            said = [m["content"] for m in self.rows[key] if m.get("role") == "user"]
            listing.append(
                ConversationSummary(
                    chat_id=key[1],
                    thread_id=key[2],
                    messages=len(self.rows[key]),
                    first_day="2026-10-11",
                    last_at="2026-10-11 21:42",
                    first_user=said[0] if said else "",
                    last_user=said[-1] if said else "",
                )
            )
        return listing


class FakeLoop:
    """Stands in for AgentLoop: remembers what each turn was given and answers."""

    max_steps = 5

    def __init__(self) -> None:
        self.runs: list[SimpleNamespace] = []
        self.gate: asyncio.Event | None = None

    async def run(self, *, chat, user_text, model, audit, progress, approver, cancel, extra_system):
        self.runs.append(
            SimpleNamespace(
                chat=chat,
                text=user_text,
                model=model,
                audit=audit,
                approver=approver,
                extra_system=extra_system,
                seen=list(chat.messages),  # what the topic knew before this turn
            )
        )
        chat.append({"role": "user", "content": user_text})
        if self.gate is not None:
            await self.gate.wait()
        chat.append({"role": "assistant", "content": f"re: {user_text}"})
        return SimpleNamespace(text=f"re: {user_text}")


def make_ui(*, store: FakeStore | None = None, recall_tools: bool = True, **settings: Any):
    """`recall_tools=False` is a profile without memory__* in its allow list."""
    loop = FakeLoop()
    store = FakeStore() if store is None else store
    profile = SimpleNamespace(
        id="lead",
        name="Lead",
        description="",
        model=None,
        tool_allowed=lambda name: recall_tools or not name.startswith("memory__"),
    )
    ui = TelegramUI(
        settings=Settings(telegram_allowed_users="42", **settings),
        profile=profile,
        agent_loop=loop,
        memory=MemoryStore(),
        store=store,
    )
    return ui, loop, store


def make_update(
    bot: FakeBot,
    text: str = "hello",
    *,
    chat_id: int = GROUP,
    thread: int | None = None,
    topic: bool = False,
    forum: bool = True,
    command: bool = False,
) -> Update:
    chat: dict[str, Any] = {
        "id": chat_id,
        "type": "supergroup" if chat_id < 0 else "private",
    }
    if chat_id < 0:
        chat["title"] = "agents"
        if forum:
            chat["is_forum"] = True
    message: dict[str, Any] = {
        "message_id": 10,
        "date": 1_700_000_000,
        "chat": chat,
        "from": {"id": 42, "is_bot": False, "first_name": "A"},
        "text": text,
    }
    if thread is not None:
        message["message_thread_id"] = thread
    if topic:
        message["is_topic_message"] = True
    if command:
        message["entities"] = [{"type": "bot_command", "offset": 0, "length": len(text.split()[0])}]
    return Update.de_json({"update_id": 1, "message": message}, bot)


def ctx_for(bot: FakeBot, *args: str) -> SimpleNamespace:
    return SimpleNamespace(bot=bot, args=list(args))


def replies(bot: FakeBot) -> list[dict[str, Any]]:
    return bot.sent


# -- which conversation a message belongs to ----------------------------------


@pytest.mark.parametrize(
    ("thread", "is_topic", "expected"),
    [
        (7, True, 7),  # a message in a forum topic
        (7, None, 0),  # a chain of replies in an ordinary group: the id names no topic
        (7, False, 0),
        (None, True, 0),  # a topic flag with no id cannot be addressed
        (None, None, 0),  # private chat, ordinary group, or the General topic of a forum
        (1, True, 0),  # General's own id is never a topic: sending to it is refused
        (2, True, 2),  # the first topic a forum creates is a topic, not General
    ],
)
def test_only_a_topic_message_names_a_topic(thread, is_topic, expected) -> None:
    message = SimpleNamespace(message_thread_id=thread, is_topic_message=is_topic)
    assert thread_of(message) == expected


def test_a_missing_message_is_no_topic() -> None:
    assert thread_of(None) == 0


# -- the memory of a topic ----------------------------------------------------


async def test_two_topics_of_one_group_do_not_share_a_history() -> None:
    ui, loop, store = make_ui()
    bot = FakeBot()
    await ui.on_message(make_update(bot, "about the database", thread=5, topic=True), ctx_for(bot))
    await ui.on_message(make_update(bot, "about the interview", thread=9, topic=True), ctx_for(bot))

    first, second = loop.runs
    assert first.chat is not second.chat
    assert (first.chat.chat_id, first.chat.thread_id) == (GROUP, 5)
    assert (second.chat.chat_id, second.chat.thread_id) == (GROUP, 9)
    # The second topic did not see what was said in the first.
    assert second.seen == []
    stored = {key: [m["content"] for m in rows] for key, rows in store.rows.items()}
    assert stored[("lead", GROUP, 5)] == ["about the database", "re: about the database"]
    assert stored[("lead", GROUP, 9)] == ["about the interview", "re: about the interview"]


async def test_a_topic_goes_on_where_it_left_off() -> None:
    ui, loop, _ = make_ui()
    bot = FakeBot()
    await ui.on_message(make_update(bot, "one", thread=5, topic=True), ctx_for(bot))
    await ui.on_message(make_update(bot, "two", thread=5, topic=True), ctx_for(bot))
    assert loop.runs[1].chat is loop.runs[0].chat
    assert [m["content"] for m in loop.runs[1].seen] == ["one", "re: one"]


async def test_history_is_restored_for_the_topic_that_has_it() -> None:
    """After a restart, a topic reloads its own rows and no one else's."""
    store = FakeStore()
    store.rows[("lead", GROUP, 5)] = [
        {"role": "user", "content": "earlier in five"},
        {"role": "assistant", "content": "answer in five"},
    ]
    store.rows[("lead", GROUP, 0)] = [{"role": "user", "content": "the old pile"}]
    ui, loop, _ = make_ui(store=store)
    bot = FakeBot()
    await ui.on_message(make_update(bot, "again", thread=5, topic=True), ctx_for(bot))
    await ui.on_message(make_update(bot, "somewhere new", thread=11, topic=True), ctx_for(bot))

    assert [m["content"] for m in loop.runs[0].seen] == ["earlier in five", "answer in five"]
    assert loop.runs[1].seen == []


async def test_the_general_topic_and_a_private_chat_are_topic_zero() -> None:
    """Where everything stored before topics were told apart sits, untouched."""
    store = FakeStore()
    store.rows[("lead", GROUP, 0)] = [{"role": "user", "content": "from before"}]
    store.rows[("lead", PRIVATE, 0)] = [{"role": "user", "content": "private, from before"}]
    ui, loop, _ = make_ui(store=store)
    bot = FakeBot()
    await ui.on_message(make_update(bot, "in general"), ctx_for(bot))
    await ui.on_message(make_update(bot, "in private", chat_id=PRIVATE), ctx_for(bot))

    assert [m["content"] for m in loop.runs[0].seen] == ["from before"]
    assert [m["content"] for m in loop.runs[1].seen] == ["private, from before"]
    assert (loop.runs[0].chat.chat_id, loop.runs[0].chat.thread_id) == (GROUP, 0)


async def test_a_reply_chain_in_an_ordinary_group_is_not_a_topic() -> None:
    ui, loop, _ = make_ui()
    bot = FakeBot()
    update = make_update(bot, "hi", thread=123, topic=False, forum=False)
    await ui.on_message(update, ctx_for(bot))
    assert loop.runs[0].chat.thread_id == 0


# -- answers land in the topic ------------------------------------------------


async def test_the_answer_lands_in_the_topic_it_was_asked_in() -> None:
    ui, _, _ = make_ui()
    bot = FakeBot()
    await ui.on_message(make_update(bot, "hello", thread=7, topic=True), ctx_for(bot))
    assert replies(bot)  # the status message and the answer
    assert {m["message_thread_id"] for m in replies(bot)} == {7}


async def test_the_general_topic_is_answered_with_no_thread() -> None:
    """Telegram refuses message_thread_id=1 for General, so there must be none."""
    ui, _, _ = make_ui()
    bot = FakeBot()
    await ui.on_message(make_update(bot, "hello"), ctx_for(bot))
    assert {m["message_thread_id"] for m in replies(bot)} == {None}


async def test_an_approval_prompt_goes_to_the_topic_of_the_request() -> None:
    """A bare send_message put it in General, where nobody was looking, and the turn
    waited out the whole approval timeout for a button nobody saw."""
    ui, loop, _ = make_ui()
    bot = FakeBot()
    ctx = ctx_for(bot)
    await ui.on_message(make_update(bot, "do it", thread=7, topic=True), ctx)
    approver = loop.runs[0].approver

    bot.sent.clear()
    asking = asyncio.create_task(approver("coder__ask", {"task": "x"}, "why"))
    await asyncio.sleep(0.05)
    assert [m["message_thread_id"] for m in bot.sent] == [7]
    for future in ui._pending.values():
        future.set_result(True)
    assert await asking is True


async def test_an_approval_prompt_in_general_carries_no_thread() -> None:
    ui, loop, _ = make_ui()
    bot = FakeBot()
    await ui.on_message(make_update(bot, "do it"), ctx_for(bot))
    approver = loop.runs[0].approver

    bot.sent.clear()
    asking = asyncio.create_task(approver("coder__ask", {"task": "x"}, "why"))
    await asyncio.sleep(0.05)
    assert "message_thread_id" not in bot.sent[0]
    for future in ui._pending.values():
        future.set_result(False)
    assert await asking is False


# -- /new, /status, /model ----------------------------------------------------


async def test_new_clears_this_topic_and_only_this_one() -> None:
    store = FakeStore()
    for topic in (0, 5, 9):
        store.rows[("lead", GROUP, topic)] = [{"role": "user", "content": f"topic {topic}"}]
    ui, loop, _ = make_ui(store=store)
    bot = FakeBot()
    await ui.on_message(make_update(bot, "go", thread=5, topic=True), ctx_for(bot))
    await ui.on_message(make_update(bot, "go", thread=9, topic=True), ctx_for(bot))

    update = make_update(bot, "/new", thread=5, topic=True, command=True)
    await ui.cmd_new(update, ctx_for(bot))

    assert ("lead", GROUP, 5) not in store.rows
    assert ("lead", GROUP, 9) in store.rows
    assert ("lead", GROUP, 0) in store.rows
    assert "этой темы" in replies(bot)[-1]["text"]
    # And in memory: the cleared topic starts empty, the other keeps going.
    await ui.on_message(make_update(bot, "again", thread=5, topic=True), ctx_for(bot))
    await ui.on_message(make_update(bot, "again", thread=9, topic=True), ctx_for(bot))
    assert loop.runs[-2].seen == []
    assert loop.runs[-1].seen != []


async def test_new_says_which_conversation_it_cleared() -> None:
    ui, _, _ = make_ui()
    bot = FakeBot()
    await ui.cmd_new(make_update(bot, "/new", thread=5, topic=True, command=True), ctx_for(bot))
    await ui.cmd_new(make_update(bot, "/new", command=True), ctx_for(bot))
    await ui.cmd_new(
        make_update(bot, "/new", chat_id=PRIVATE, command=True), ctx_for(bot)
    )
    said = [m["text"] for m in replies(bot)]
    assert "этой темы" in said[0]
    assert "темы General" in said[1]
    assert "этого чата" in said[2]


async def test_status_loads_the_topic_and_names_it() -> None:
    """Right after a restart a zero would look like lost memory."""
    store = FakeStore()
    store.rows[("lead", GROUP, 5)] = [{"role": "user", "content": "a"}] * 3
    ui, _, _ = make_ui(store=store)
    bot = FakeBot()
    await ui.cmd_status(
        make_update(bot, "/status", thread=5, topic=True, command=True), ctx_for(bot)
    )
    text = replies(bot)[-1]["text"]
    assert "этой темы: 3" in text
    assert "Тема: 5" in text


async def test_the_model_is_picked_for_the_chat_and_every_topic_uses_it() -> None:
    """/model has always applied to the whole chat; the topic did not take it over."""
    ui, loop, _ = make_ui(model_default="m1", model_allowlist="m1,m2")
    bot = FakeBot()
    await ui.cmd_model(
        make_update(bot, "/model m2", thread=5, topic=True, command=True), ctx_for(bot, "m2")
    )
    await ui.on_message(make_update(bot, "a", thread=9, topic=True), ctx_for(bot))
    await ui.on_message(make_update(bot, "b"), ctx_for(bot))
    assert [run.model for run in loop.runs] == ["m2", "m2"]


# -- what the model is told ---------------------------------------------------


async def test_every_conversation_is_told_it_comes_first_and_the_others_can_be_read() -> None:
    """A topic, General and the private chat alike: each is first priority for itself,
    and every other one is second."""
    ui, loop, _ = make_ui()
    bot = FakeBot()
    await ui.on_message(make_update(bot, "a", thread=5, topic=True), ctx_for(bot))
    await ui.on_message(make_update(bot, "b"), ctx_for(bot))
    await ui.on_message(make_update(bot, "c", chat_id=PRIVATE), ctx_for(bot))
    assert len(loop.runs) == 3
    for run in loop.runs:
        assert recall.RECALL_NOTE in run.extra_system
        assert TOPIC_NOTE not in run.extra_system


async def test_the_index_names_the_other_conversations_and_not_this_one() -> None:
    store = FakeStore()
    store.rows[("lead", GROUP, 9)] = [{"role": "user", "content": "про собеседование"}]
    store.rows[("lead", GROUP, 0)] = [{"role": "user", "content": "старое"}]
    store.rows[("lead", PRIVATE, 0)] = [{"role": "user", "content": "в личке"}]
    store.rows[("coder", GROUP, 3)] = [{"role": "user", "content": "чужое"}]
    ui, loop, _ = make_ui(store=store)
    bot = FakeBot()
    await ui.on_message(make_update(bot, "про базу", thread=5, topic=True), ctx_for(bot))
    await ui.on_message(make_update(bot, "ещё", thread=5, topic=True), ctx_for(bot))

    second = loop.runs[1].extra_system
    assert "тема 9" in second
    assert "про собеседование" in second
    assert "General" in second
    assert "личный чат" in second
    assert "тема 5" not in second  # its own history is the transcript, not a line here
    assert "тема 3" not in second  # another agent's conversations are not this agent's


async def test_a_topic_cleared_with_new_drops_out_of_the_other_topics_index() -> None:
    ui, loop, _ = make_ui()
    bot = FakeBot()
    await ui.on_message(make_update(bot, "про базу", thread=5, topic=True), ctx_for(bot))
    await ui.on_message(make_update(bot, "про другое", thread=9, topic=True), ctx_for(bot))
    assert "тема 5" in loop.runs[-1].extra_system
    await ui.cmd_new(make_update(bot, "/new", thread=5, topic=True, command=True), ctx_for(bot))
    await ui.on_message(make_update(bot, "ещё", thread=9, topic=True), ctx_for(bot))
    assert "тема 5" not in loop.runs[-1].extra_system
    assert "Других разговоров с ним пока нет." in loop.runs[-1].extra_system


async def test_the_prompt_reads_facts_then_the_note_then_the_index() -> None:
    """The note says the history below is this conversation's; after the index it
    would stand between the other conversations and nothing."""
    store = FakeStore()
    store.facts_known = [Fact(key="k", fact="a durable fact", scope="user", updated_at="")]
    store.rows[("lead", GROUP, 9)] = [{"role": "user", "content": "elsewhere"}]
    ui, loop, _ = make_ui(store=store)
    bot = FakeBot()
    await ui.on_message(make_update(bot, "a", thread=5, topic=True), ctx_for(bot))
    extra = loop.runs[0].extra_system
    assert extra.index("a durable fact") < extra.index(recall.RECALL_NOTE) < extra.index("тема 9")


async def test_without_the_tools_a_topic_is_told_it_sees_no_other_and_general_nothing() -> None:
    """A profile that cannot read the other conversations must not be told it can."""
    ui, loop, _ = make_ui(recall_tools=False)
    bot = FakeBot()
    await ui.on_message(make_update(bot, "a", thread=5, topic=True), ctx_for(bot))
    await ui.on_message(make_update(bot, "b"), ctx_for(bot))
    assert TOPIC_NOTE in loop.runs[0].extra_system
    assert TOPIC_NOTE not in loop.runs[1].extra_system
    assert all(recall.RECALL_NOTE not in run.extra_system for run in loop.runs)


async def test_without_a_store_there_is_nothing_to_read_and_nothing_is_promised() -> None:
    loop = FakeLoop()
    ui = TelegramUI(
        settings=Settings(telegram_allowed_users="42"),
        profile=SimpleNamespace(
            id="lead", name="Lead", description="", model=None, tool_allowed=lambda _name: True
        ),
        agent_loop=loop,
        memory=MemoryStore(),
        store=None,
    )
    bot = FakeBot()
    await ui.on_message(make_update(bot, "a", thread=5, topic=True), ctx_for(bot))
    assert recall.RECALL_NOTE not in loop.runs[0].extra_system
    assert TOPIC_NOTE in loop.runs[0].extra_system


async def test_a_local_model_without_tools_is_not_told_about_them() -> None:
    """AgentLoop withholds every tool from it; the note would send it looking."""
    ui, loop, _ = make_ui(
        models_ollama="qwen",
        ollama_base_url="http://ollama.invalid",
        model_default="qwen",
        model_allowlist="qwen",
    )
    bot = FakeBot()
    await ui.on_message(make_update(bot, "a", thread=5, topic=True), ctx_for(bot))
    assert loop.runs[0].model == "qwen"
    assert recall.RECALL_NOTE not in loop.runs[0].extra_system
    assert TOPIC_NOTE in loop.runs[0].extra_system


@pytest.mark.parametrize("error", [StoreUnavailable("ORDS returned HTTP 503"), KeyError("n")])
async def test_a_listing_that_fails_keeps_the_note_and_the_turn(error: Exception) -> None:
    """It runs before every turn: a failure here must not end every conversation."""
    store = FakeStore()
    store.listing_error = error
    ui, loop, _ = make_ui(store=store)
    bot = FakeBot()
    await ui.on_message(make_update(bot, "a", thread=5, topic=True), ctx_for(bot))
    assert len(loop.runs) == 1
    assert loop.runs[0].extra_system.endswith(recall.RECALL_NOTE)
    assert replies(bot)[-1]["text"] == "re: a"


# -- one task at a time per chat, as before -----------------------------------


async def test_a_busy_chat_stays_busy_for_all_of_its_topics() -> None:
    """The busy flag and /cancel were per chat and still are; only memory is split."""
    ui, loop, _ = make_ui()
    loop.gate = asyncio.Event()
    bot = FakeBot()
    first = asyncio.create_task(
        ui.on_message(make_update(bot, "long task", thread=5, topic=True), ctx_for(bot))
    )
    await asyncio.sleep(0.05)
    await ui.on_message(make_update(bot, "meanwhile", thread=9, topic=True), ctx_for(bot))
    assert "занят" in replies(bot)[-1]["text"]
    assert len(loop.runs) == 1

    loop.gate.set()
    await first


# -- the audit log ------------------------------------------------------------


def test_the_audit_log_carries_the_topic_beside_the_chat() -> None:
    from structlog.testing import capture_logs

    from agentcore.audit import AuditLog

    with capture_logs() as logs:
        audit = AuditLog(agent="lead", chat_id=GROUP, thread_id=5)
        audit.tool_call(tool="github__get_file_contents", arguments={}, decision="allow", ok=True)
    entry = logs[0]
    # Beside, not folded in: anything that filters the log by chat keeps working.
    assert entry["chat_id"] == str(GROUP)
    assert entry["thread_id"] == "5"


def test_an_audit_log_without_a_topic_says_zero() -> None:
    from structlog.testing import capture_logs

    from agentcore.audit import AuditLog

    with capture_logs() as logs:
        AuditLog(agent="lead", chat_id=1).tool_call(
            tool="github__get_file_contents", arguments={}, decision="allow", ok=True
        )
    assert logs[0]["thread_id"] == "0"


# -- /new says what happened --------------------------------------------------


async def test_new_says_so_when_the_database_would_not_forget() -> None:
    """The rows are still there and the next restart replays them: the opposite of
    what was asked, so the reply must not claim the database is clear."""
    store = FakeStore()
    store.clear_works = False
    ui, _, _ = make_ui(store=store)
    bot = FakeBot()
    await ui.cmd_new(make_update(bot, "/new", thread=5, topic=True, command=True), ctx_for(bot))
    said = replies(bot)[-1]["text"]
    assert "не вышло" in said
    assert "и в базе" not in said


async def test_new_in_general_says_it_clears_what_came_before_topics() -> None:
    """General holds everything written before topics were told apart, so clearing it
    clears all of that, not only what was said in General."""
    ui, _, _ = make_ui()
    bot = FakeBot()
    await ui.cmd_new(make_update(bot, "/new", command=True), ctx_for(bot))
    await ui.cmd_new(make_update(bot, "/new", thread=5, topic=True, command=True), ctx_for(bot))
    await ui.cmd_new(make_update(bot, "/new", chat_id=PRIVATE, command=True), ctx_for(bot))
    general, topic, private = (m["text"] for m in replies(bot))
    assert "до разделения по темам" in general
    assert "до разделения по темам" not in topic
    assert "до разделения по темам" not in private


# -- /status does not load anything -------------------------------------------


async def test_status_after_a_restart_reports_what_will_load_without_loading_it() -> None:
    store = FakeStore()
    store.rows[("lead", GROUP, 5)] = [{"role": "user", "content": "a"}] * 3
    ui, loop, _ = make_ui(store=store)
    bot = FakeBot()
    await ui.cmd_status(
        make_update(bot, "/status", thread=5, topic=True, command=True), ctx_for(bot)
    )
    assert "этой темы: 3" in replies(bot)[-1]["text"]
    # Still unloaded: the first message gets the load it was always going to get.
    await ui.on_message(make_update(bot, "go", thread=5, topic=True), ctx_for(bot))
    assert len(loop.runs[0].seen) == 3


async def test_a_status_typed_just_before_a_message_cannot_wipe_that_message() -> None:
    """The race the review reproduced: a load started by /status marked the topic as
    loaded, the message that followed ran with no history, and the late result then
    replaced the transcript, taking the new question with it."""
    store = FakeStore()
    store.rows[("lead", GROUP, 5)] = [
        {"role": "user", "content": "earlier question"},
        {"role": "assistant", "content": "earlier answer"},
    ]
    store.gate = asyncio.Event()
    ui, loop, _ = make_ui(store=store)
    bot = FakeBot()

    status = asyncio.create_task(
        ui.cmd_status(make_update(bot, "/status", thread=5, topic=True, command=True), ctx_for(bot))
    )
    await asyncio.sleep(0.02)  # the status read is in flight, held by the gate
    message = asyncio.create_task(
        ui.on_message(make_update(bot, "new question", thread=5, topic=True), ctx_for(bot))
    )
    await asyncio.sleep(0.02)
    store.gate.set()
    await asyncio.gather(status, message)

    assert [m["content"] for m in loop.runs[0].seen] == ["earlier question", "earlier answer"]
    stored = [m["content"] for m in store.rows[("lead", GROUP, 5)]]
    assert stored == ["earlier question", "earlier answer", "new question", "re: new question"]


# -- what the audit log and the prompt say about the topic --------------------


async def test_the_audit_log_names_the_topic_of_the_turn(monkeypatch) -> None:
    seen: list[dict[str, Any]] = []

    class RecordingAudit:
        def __init__(self, **kwargs: Any) -> None:
            seen.append(kwargs)

    monkeypatch.setattr(telegram_ui, "AuditLog", RecordingAudit)
    ui, _, _ = make_ui()
    bot = FakeBot()
    await ui.on_message(make_update(bot, "a", thread=7, topic=True), ctx_for(bot))
    await ui.on_message(make_update(bot, "b"), ctx_for(bot))
    assert [(a["chat_id"], a["thread_id"]) for a in seen] == [(GROUP, 7), (GROUP, 0)]


async def test_the_topic_note_comes_after_the_facts_it_refers_to() -> None:
    """The note says "факты выше"; placed before them it would point at nothing."""
    store = FakeStore()
    store.facts_known = [Fact(key="k", fact="a durable fact", scope="user", updated_at="")]
    ui, loop, _ = make_ui(store=store, recall_tools=False)
    bot = FakeBot()
    await ui.on_message(make_update(bot, "a", thread=5, topic=True), ctx_for(bot))
    extra = loop.runs[0].extra_system
    assert extra.endswith(TOPIC_NOTE)
    assert extra.index("a durable fact") < extra.index(TOPIC_NOTE)


# -- the model picker ---------------------------------------------------------


def _press(chat_id: int, data: str) -> tuple[SimpleNamespace, list[str]]:
    said: list[str] = []

    async def answer(text: str = "", **_: Any) -> None:
        said.append(f"answer:{text}")

    async def edit_message_text(text: str, **_: Any) -> None:
        said.append(f"edit:{text}")

    query = SimpleNamespace(data=data, answer=answer, edit_message_text=edit_message_text)
    update = SimpleNamespace(
        callback_query=query,
        effective_chat=SimpleNamespace(id=chat_id),
        effective_user=SimpleNamespace(id=42),
    )
    return update, said


async def test_a_picker_button_sets_the_model_for_the_chat_with_no_topic_needed() -> None:
    """The callback has no reliable topic (the message can be gone), which is why the
    model belongs to the chat."""
    ui, loop, _ = make_ui(model_default="m1", model_allowlist="m1,m2")
    update, said = _press(GROUP, "ap:model:1")
    await ui.on_callback(update, ctx_for(FakeBot()))
    assert ui._memory.model_for(GROUP) == "m2"
    assert any("m2" in line for line in said)

    bot = FakeBot()
    await ui.on_message(make_update(bot, "a", thread=9, topic=True), ctx_for(bot))
    assert loop.runs[0].model == "m2"


async def test_a_stale_picker_button_changes_nothing() -> None:
    ui, _, _ = make_ui(model_default="m1", model_allowlist="m1,m2")
    update, said = _press(GROUP, "ap:model:9")
    await ui.on_callback(update, ctx_for(FakeBot()))
    assert ui._memory.model_for(GROUP) is None
    assert said and said[0].startswith("answer:")


# -- the fakes stay honest ----------------------------------------------------


def _params(fn: Any) -> list[str]:
    return list(inspect.signature(fn).parameters)


def test_the_fake_store_has_the_real_stores_signatures() -> None:
    """A rename on the real side must fail here, not hide behind a permissive fake."""
    for name in ("history", "append", "clear", "facts", "conversations"):
        assert _params(getattr(FakeStore, name)) == _params(getattr(AdbStore, name)), name


def test_the_fake_loop_has_the_real_loops_signature() -> None:
    assert _params(FakeLoop.run) == _params(AgentLoop.run)
