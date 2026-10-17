"""The three tools that read the other conversations, from the model naming one to the
store being asked.

What matters is what the model can rely on: that the lead is offered them and no one has
to press a button for them, that they read this agent's rows from the conversation the
turn is in, that a count or an id the model gets slightly wrong still works or fails
with a sentence it can act on, and that a database that is down is never reported as a
past in which nothing was said.
"""

from __future__ import annotations

import asyncio
import inspect
from types import SimpleNamespace
from typing import Any

import pytest

from agentcore import recall
from agentcore.config import Settings
from agentcore.llm.base import LLMResponse, ToolCall, Usage
from agentcore.localtools import (
    FORGET,
    LIST_CONVERSATIONS,
    READ_ALONE,
    READ_CONVERSATION,
    READ_FOCUS,
    REMEMBER,
    SEARCH_CONVERSATIONS,
    call_local,
    memory_tools,
    read_width,
)
from agentcore.loop import AgentLoop
from agentcore.mcp.toolset import build_catalog
from agentcore.memory import ChatMemory
from agentcore.policy import Decision, Policy
from agentcore.profiles import load_profile
from agentcore.store import AdbStore, ConversationSummary, StoredText, StoreUnavailable

GROUP = -1001234567890
PRIVATE = 1234567

RECALL_TOOLS = (SEARCH_CONVERSATIONS, READ_CONVERSATION, LIST_CONVERSATIONS)


class FakeStore:
    """The part of AdbStore the reading tools use. Records what it was asked."""

    def __init__(
        self,
        *,
        listing: list[ConversationSummary] | None = None,
        hits: list[StoredText] | None = None,
        messages: list[StoredText] | None = None,
        fails: bool = False,
    ) -> None:
        self.listing = listing if listing is not None else [
            summary(GROUP, 939),
            summary(GROUP, 78),
            summary(GROUP, 0),
            summary(PRIVATE, 0),
        ]
        self.hits = hits or []
        self.messages = messages or []
        self.fails = fails
        self.asked: list[tuple[str, dict[str, Any]]] = []

    async def conversations(self, *, agent):
        self.asked.append(("conversations", {"agent": agent}))
        if self.fails:
            raise StoreUnavailable("cannot reach the database: timed out")
        return list(self.listing)

    async def search(self, *, agent, terms, limit, current=None):
        self.asked.append(
            ("search", {"agent": agent, "terms": list(terms), "limit": limit, "current": current})
        )
        return list(self.hits)

    async def read(self, *, agent, chat_id, thread_id, around, limit, width, focus_width):
        self.asked.append(
            (
                "read",
                {
                    "agent": agent,
                    "chat_id": chat_id,
                    "thread_id": thread_id,
                    "around": around,
                    "limit": limit,
                    "width": width,
                    "focus_width": focus_width,
                },
            )
        )
        return list(self.messages)

    def calls(self, name: str) -> list[dict[str, Any]]:
        return [args for asked, args in self.asked if asked == name]


def summary(chat_id: int, thread_id: int, first_user: str = "") -> ConversationSummary:
    return ConversationSummary(
        chat_id=chat_id,
        thread_id=thread_id,
        messages=4,
        first_day="2026-10-11",
        last_at="2026-10-11 21:42",
        first_user=first_user,
        last_user="",
        first_id={(GROUP, 0): 341, (PRIVATE, 0): 21}.get((chat_id, thread_id), 1800 + thread_id),
    )


def hit(id: int, chat_id: int = GROUP, thread_id: int = 78) -> StoredText:
    return StoredText(
        id=id,
        chat_id=chat_id,
        thread_id=thread_id,
        role="user",
        at="2026-10-11 21:42",
        text="загадал число 7",
        start=1,
        length=15,
    )


def call(name: str, arguments: dict[str, Any], store: Any, **where: Any):
    where = {"agent": "lead", "chat_id": GROUP, "thread_id": 939, **where}
    return asyncio.run(call_local(name, arguments, store=store, **where))


# -- offered, and to whom -----------------------------------------------------


def test_the_reading_tools_reach_the_catalogue_with_a_store() -> None:
    catalog = build_catalog(memory_tools(FakeStore()), is_allowed=lambda _name: True)
    for name in RECALL_TOOLS:
        assert catalog.resolve(name) == name


@pytest.mark.parametrize("name", RECALL_TOOLS)
def test_reading_the_conversations_needs_no_button(name: str) -> None:
    """Approving every look into the past would make the past not worth looking into."""
    assert Policy(Settings()).evaluate(name, {}).decision is Decision.ALLOW


def test_the_lead_is_offered_every_memory_tool(agents_dir) -> None:
    """Without memory__* in its allow list it was offered none: the facts prompt asked
    for memory__remember and the catalogue never held it."""
    lead = load_profile("lead", agents_dir)
    for tool in memory_tools(FakeStore()):
        assert lead.tool_allowed(tool.qualified_name), tool.qualified_name


@pytest.mark.parametrize("worker", ["coder", "reviewer"])
def test_the_workers_are_offered_none(agents_dir, worker: str) -> None:
    """They answer the lead with a throwaway transcript; there is nothing of theirs to
    read, and the lead's conversations are not theirs to read."""
    profile = load_profile(worker, agents_dir)
    for tool in memory_tools(FakeStore()):
        assert not profile.tool_allowed(tool.qualified_name), tool.qualified_name


# -- list ---------------------------------------------------------------------


def test_list_reads_this_agents_conversations_and_marks_the_current_one() -> None:
    store = FakeStore(listing=[summary(GROUP, 939), summary(GROUP, 78), summary(PRIVATE, 0)])
    result = call(LIST_CONVERSATIONS, {}, store)
    assert result.ok
    assert store.calls("conversations") == [{"agent": "lead"}]
    assert "тема 939 (этот разговор)" in result.text
    assert "тема 78 ·" in result.text
    assert "личный чат" in result.text


# -- search -------------------------------------------------------------------


def test_search_sends_the_words_and_marks_where_each_hit_is() -> None:
    store = FakeStore(listing=[summary(GROUP, 78)], hits=[hit(1853)])
    result = call(SEARCH_CONVERSATIONS, {"query": "Загадал число"}, store)
    assert result.ok
    [asked] = store.calls("search")
    assert asked == {
        "agent": "lead",
        "terms": ["загадал", "число"],
        "limit": 8,
        "current": (GROUP, 939),  # ranked after the other conversations
    }
    assert "[1853] тема 78" in result.text
    assert f"id {GROUP}/78" in result.text


@pytest.mark.parametrize(
    ("limit", "sent"), [(3, 3), ("5", 5), (500, 20), (0, 1), (-4, 1), ("много", 8), (None, 8)]
)
def test_a_search_limit_is_kept_in_range_rather_than_refused(limit: Any, sent: int) -> None:
    store = FakeStore()
    assert call(SEARCH_CONVERSATIONS, {"query": "oracle", "limit": limit}, store).ok
    assert store.calls("search")[0]["limit"] == sent


@pytest.mark.parametrize("query", ["", "   ", "и в на", None])
def test_a_search_with_nothing_to_look_for_fails_before_the_database(query: Any) -> None:
    store = FakeStore()
    result = call(SEARCH_CONVERSATIONS, {"query": query}, store)
    assert not result.ok
    assert "query" in result.text
    assert store.calls("search") == []


# -- read ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("conversation", "target"),
    [
        (f"{GROUP}/78", (GROUP, 78)),
        ("78", (GROUP, 78)),  # a topic of the group this turn is in
        ("https://t.me/c/1234567890/78", (GROUP, 78)),
        (f"{PRIVATE}/0", (PRIVATE, 0)),
        ("1", (GROUP, 0)),  # General's own id
    ],
)
def test_read_finds_the_conversation_however_the_model_names_it(conversation, target) -> None:
    store = FakeStore()
    assert call(READ_CONVERSATION, {"conversation": conversation}, store).ok
    [asked] = store.calls("read")
    assert (asked["chat_id"], asked["thread_id"]) == target
    assert asked["agent"] == "lead"


@pytest.mark.parametrize(("conversation", "target"), [("78", (GROUP, 78)), ("0", (GROUP, 0))])
def test_a_bare_topic_number_from_the_private_chat_is_a_topic_of_the_group(
    conversation: str, target: tuple[int, int]
) -> None:
    """A private chat has no topics: "78" read there was a conversation that cannot
    exist, and came back as one with nothing said in it."""
    store = FakeStore()
    assert call(READ_CONVERSATION, {"conversation": conversation}, store, chat_id=PRIVATE, thread_id=0).ok
    [asked] = store.calls("read")
    assert (asked["chat_id"], asked["thread_id"]) == target


def test_a_conversation_that_does_not_exist_is_a_failure_with_the_ones_that_do() -> None:
    store = FakeStore()
    result = call(READ_CONVERSATION, {"conversation": "4242"}, store)
    assert not result.ok
    assert "такого разговора нет" in result.text
    assert f"{GROUP}/78" in result.text  # what there is instead
    assert store.calls("read") == []


@pytest.mark.parametrize("conversation", ["", "тема 78", None, "General"])
def test_read_refuses_a_conversation_it_cannot_name(conversation: Any) -> None:
    store = FakeStore()
    result = call(READ_CONVERSATION, {"conversation": conversation}, store)
    assert not result.ok
    assert "conversation" in result.text
    assert store.calls("read") == []


@pytest.mark.parametrize(
    ("around", "sent"),
    [
        (1853, 1853),
        ("1853", 1853),
        ("[1853]", 1853),
        (1853.0, 1853),
        (None, None),
        ("", None),
        (0, None),  # how some models fill an optional integer they mean to leave out
        ("0", None),
    ],
)
def test_around_is_taken_as_a_model_writes_it(around: Any, sent: int | None) -> None:
    store = FakeStore()
    assert call(READ_CONVERSATION, {"conversation": "78", "around": around}, store).ok
    assert store.calls("read")[0]["around"] == sent


@pytest.mark.parametrize("around", ["вчера", True, 18.5, "-3", -3])
def test_an_around_that_is_not_a_message_number_is_refused(around: Any) -> None:
    store = FakeStore()
    result = call(READ_CONVERSATION, {"conversation": "78", "around": around}, store)
    assert not result.ok
    assert "around" in result.text
    assert store.calls("read") == []


def test_one_message_asked_for_alone_is_given_more_room() -> None:
    store = FakeStore()
    call(READ_CONVERSATION, {"conversation": "78", "around": 5, "limit": 1}, store)
    call(READ_CONVERSATION, {"conversation": "78", "around": 5}, store)
    call(READ_CONVERSATION, {"conversation": "78", "limit": 30}, store)
    alone, default, many = store.calls("read")
    assert alone["focus_width"] == READ_ALONE
    assert default["focus_width"] == READ_FOCUS
    assert default["limit"] == 12
    assert many["limit"] == 30
    assert many["width"] == read_width(30)


@pytest.mark.parametrize("limit", [1, 2, 5, 12, 30])
def test_a_read_of_long_messages_fits_in_one_tool_result(limit: int) -> None:
    """The loop clips a tool result at 12,000 characters; the headers count too, and
    leaving them out of the budget is how the last message of a long read was lost."""
    width = read_width(limit)
    focus = READ_ALONE if limit == 1 else READ_FOCUS
    messages = [
        StoredText(
            id=1000 + i,
            chat_id=GROUP,
            thread_id=78,
            role="assistant",
            at="2026-10-11 21:42",
            text="я" * (focus if i == 0 else width),
            start=1,
            length=20_000,
        )
        for i in range(limit)
    ]
    out = recall.format_read(
        messages,
        chat_id=GROUP,
        thread_id=78,
        conversations=FakeStore().listing,
        current=None,
        around=1000,
    )
    assert len(out) < 12_000


def test_read_marks_the_message_it_was_asked_about() -> None:
    store = FakeStore(listing=[summary(GROUP, 78)], messages=[hit(1852), hit(1853)])
    result = call(READ_CONVERSATION, {"conversation": "78", "around": 1853}, store)
    assert "→ [1853]" in result.text
    assert "→ [1852]" not in result.text


# -- failures -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "arguments"),
    [
        (LIST_CONVERSATIONS, {}),
        (SEARCH_CONVERSATIONS, {"query": "oracle"}),
        (READ_CONVERSATION, {"conversation": "78"}),
    ],
)
def test_a_database_that_is_down_is_a_failure_and_not_an_empty_past(name, arguments) -> None:
    result = call(name, arguments, FakeStore(fails=True))
    assert not result.ok
    assert "недоступна" in result.text
    assert "не нашлось" not in result.text


@pytest.mark.parametrize("name", RECALL_TOOLS)
def test_without_an_agent_nothing_is_read(name: str) -> None:
    """Rows belong to an agent; a read without one would read every agent's."""
    store = FakeStore()
    result = call(name, {"query": "oracle", "conversation": "78"}, store, agent=None)
    assert not result.ok
    assert store.asked == []


@pytest.mark.parametrize("name", RECALL_TOOLS)
def test_without_a_store_the_call_fails_instead_of_crashing(name: str) -> None:
    assert not call(name, {"query": "oracle", "conversation": "78"}, None).ok


# -- through the loop ---------------------------------------------------------


class FakeAudit:
    def turn(self, **_kwargs):
        pass

    def tool_call(self, **_kwargs):
        pass

    def denied(self, **_kwargs):
        pass


class FakeLLM:
    def __init__(self, responses):
        self.responses = iter(responses)

    async def complete(self, **_kwargs):
        return next(self.responses)


class NoPool:
    async def list_tools(self, **_kwargs):
        return []

    async def call_tool(self, name, _arguments):
        raise AssertionError(f"{name} should have been dispatched in-process")


def test_the_loop_reads_as_the_agent_from_the_conversation_the_turn_is_in() -> None:
    store = FakeStore(listing=[summary(GROUP, 78), summary(GROUP, 939)], hits=[hit(1859, thread_id=939)])
    loop = AgentLoop(
        profile=SimpleNamespace(
            id="lead",
            max_steps=5,
            max_tokens_per_turn=10_000,
            system_prompt="test",
            tool_allowed=lambda _name: True,
        ),
        settings=Settings(telegram_allowed_users="42", max_billable_tokens_per_turn=200_000),
        llm=FakeLLM(
            [
                LLMResponse(
                    content=None,
                    tool_calls=[ToolCall(id="1", name=SEARCH_CONVERSATIONS, arguments={"query": "число"})],
                    usage=Usage(prompt_tokens=10, completion_tokens=1),
                    finish_reason="tool_calls",
                ),
                LLMResponse(content="7", usage=Usage(prompt_tokens=12, completion_tokens=1), finish_reason="stop"),
            ]
        ),
        pool=NoPool(),
        policy=Policy(Settings()),
        store=store,
    )
    chat = ChatMemory(chat_id=GROUP, thread_id=939)
    result = asyncio.run(loop.run(chat=chat, user_text="какое число?", model="m", audit=FakeAudit()))

    assert result.tool_calls == 1
    assert store.calls("search")[0]["agent"] == "lead"
    tool_message = next(m for m in chat.messages if m["role"] == "tool")
    # The hit is in the topic the turn is in, so it is marked as this conversation.
    assert "тема 939 (этот разговор)" in tool_message["content"]


def _catalog_names(allowed_users: str) -> list[str]:
    loop = AgentLoop(
        profile=SimpleNamespace(
            id="lead", max_steps=5, max_tokens_per_turn=10_000, system_prompt="t",
            tool_allowed=lambda _name: True,
        ),
        settings=Settings(telegram_allowed_users=allowed_users),
        llm=FakeLLM([]),
        pool=NoPool(),
        policy=Policy(Settings()),
        store=FakeStore(),
    )
    return asyncio.run(loop.catalog()).names()


def test_with_two_people_allowed_the_conversations_cannot_be_read() -> None:
    """Rows carry no author: with a second person allowed, reading every conversation
    would read one's private chat to the other. Facts stay; they always were shared."""
    one = _catalog_names("42")
    two = _catalog_names("42,43")
    assert set(RECALL_TOOLS) <= set(one)
    assert not set(RECALL_TOOLS) & set(two)
    assert {REMEMBER, FORGET} <= set(two)


# -- the fake stays honest ----------------------------------------------------


def test_the_fake_store_has_the_real_stores_signatures() -> None:
    for name in ("conversations", "search", "read"):
        real = list(inspect.signature(getattr(AdbStore, name)).parameters)
        fake = list(inspect.signature(getattr(FakeStore, name)).parameters)
        assert fake == real, name
