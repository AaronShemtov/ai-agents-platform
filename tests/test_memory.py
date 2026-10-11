"""Transcript trimming.

The failure this guards against is subtle: dropping messages from the front can leave
a `tool` message whose parent assistant message — the one carrying the matching
`tool_calls` — has been trimmed away. The API rejects that transcript outright, and
the symptom is a 400 that appears only on long conversations.
"""

from __future__ import annotations

from typing import Any

import pytest

from agentcore.memory import ChatMemory, ConversationKey, MemoryStore, estimate_tokens


def user(text: str) -> dict[str, Any]:
    return {"role": "user", "content": text}


def assistant_calling(tool_call_id: str, name: str = "github__get_file_contents") -> dict[str, Any]:
    return {
        "role": "assistant",
        "content": None,
        "tool_calls": [
            {"id": tool_call_id, "type": "function", "function": {"name": name, "arguments": "{}"}}
        ],
    }


def tool_result(tool_call_id: str, text: str = "ok") -> dict[str, Any]:
    return {"role": "tool", "tool_call_id": tool_call_id, "content": text}


def orphaned_tool_messages(messages: list[dict[str, Any]]) -> list[str]:
    """Tool messages whose parent assistant tool_call is not in the transcript."""
    declared = {
        call["id"]
        for message in messages
        if message.get("role") == "assistant"
        for call in (message.get("tool_calls") or [])
    }
    return [
        m["tool_call_id"]
        for m in messages
        if m.get("role") == "tool" and m.get("tool_call_id") not in declared
    ]


def test_system_prompt_comes_first() -> None:
    chat = ChatMemory(chat_id=1)
    chat.append(user("hi"))
    transcript = chat.transcript("SYSTEM", max_tokens=10_000)
    assert transcript[0] == {"role": "system", "content": "SYSTEM"}


def test_short_history_is_kept_whole() -> None:
    chat = ChatMemory(chat_id=1)
    chat.append(user("hi"))
    chat.append({"role": "assistant", "content": "hello"})
    assert len(chat.transcript("S", max_tokens=10_000)) == 3


def test_trimming_never_orphans_a_tool_message() -> None:
    chat = ChatMemory(chat_id=1)
    for i in range(30):
        chat.append(user("x" * 2000))
        chat.append(assistant_calling(f"call-{i}"))
        chat.append(tool_result(f"call-{i}", "y" * 2000))
        chat.append({"role": "assistant", "content": "done"})

    transcript = chat.transcript("S", max_tokens=4_000)
    assert orphaned_tool_messages(transcript) == []


def test_trimming_actually_drops_something() -> None:
    chat = ChatMemory(chat_id=1)
    for _ in range(30):
        chat.append(user("x" * 2000))
        chat.append({"role": "assistant", "content": "y" * 2000})

    transcript = chat.transcript("S", max_tokens=4_000)
    assert len(transcript) < len(chat.messages) + 1


def test_the_newest_turn_is_kept_even_if_it_alone_exceeds_the_budget() -> None:
    # Dropping it would mean answering without knowing what was asked.
    chat = ChatMemory(chat_id=1)
    chat.append(user("x" * 100_000))
    transcript = chat.transcript("S", max_tokens=100)
    assert len(transcript) == 2


def test_trimming_keeps_the_most_recent_turns() -> None:
    chat = ChatMemory(chat_id=1)
    for i in range(20):
        chat.append(user(f"turn-{i} " + "x" * 2000))

    transcript = chat.transcript("S", max_tokens=4_000)
    assert "turn-19" in transcript[-1]["content"]


def test_estimate_tokens_counts_tool_calls_too() -> None:
    plain = [{"role": "assistant", "content": "hi"}]
    with_calls = [assistant_calling("c1")]
    assert estimate_tokens(with_calls) > estimate_tokens(plain)


def test_store_isolates_chats_and_resets_one_at_a_time() -> None:
    store = MemoryStore()
    store.get(ConversationKey(1, 0)).append(user("a"))
    store.get(ConversationKey(2, 0)).append(user("b"))
    store.reset(ConversationKey(1, 0))
    assert store.get(ConversationKey(1, 0)).messages == []
    assert len(store.get(ConversationKey(2, 0)).messages) == 1


def test_the_topics_of_one_chat_are_separate_conversations() -> None:
    """The reason the key has two parts: one forum group, one history per topic."""
    store = MemoryStore()
    store.get(ConversationKey(-100, 5)).append(user("about the database"))
    store.get(ConversationKey(-100, 9)).append(user("about the interview"))
    assert [m["content"] for m in store.get(ConversationKey(-100, 5)).messages] == [
        "about the database"
    ]
    assert [m["content"] for m in store.get(ConversationKey(-100, 9)).messages] == [
        "about the interview"
    ]


def test_resetting_one_topic_leaves_the_others() -> None:
    store = MemoryStore()
    store.get(ConversationKey(-100, 5)).append(user("a"))
    store.get(ConversationKey(-100, 9)).append(user("b"))
    store.get(ConversationKey(-100, 0)).append(user("general"))
    store.reset(ConversationKey(-100, 5))
    assert store.get(ConversationKey(-100, 5)).messages == []
    assert len(store.get(ConversationKey(-100, 9)).messages) == 1
    assert len(store.get(ConversationKey(-100, 0)).messages) == 1


def test_a_conversation_knows_its_own_chat_and_topic() -> None:
    chat = MemoryStore().get(ConversationKey(-100, 5))
    assert (chat.chat_id, chat.thread_id) == (-100, 5)


def test_a_bare_chat_id_is_refused_rather_than_becoming_a_conversation() -> None:
    """The old way of calling this. Quietly accepting it would file the messages
    under a topic-less guess, which is the mixing the key exists to stop."""
    with pytest.raises(AttributeError):
        MemoryStore().get(1)  # type: ignore[arg-type]


def test_a_transcript_built_without_a_topic_is_topic_zero() -> None:
    # The coder and the reviewer keep a throwaway transcript per call.
    assert ChatMemory(chat_id=1).thread_id == 0


def test_reset_keeps_the_model_override() -> None:
    # /new clears the conversation, not the model the user picked with /model.
    store = MemoryStore()
    store.set_model(1, "gpt-5-mini")
    store.get(ConversationKey(1, 0)).append(user("a"))
    store.reset(ConversationKey(1, 0))
    assert store.model_for(1) == "gpt-5-mini"


def test_the_model_is_picked_per_chat_and_shared_by_its_topics() -> None:
    """/model has always applied to the whole chat; moving it into the topic would
    have changed that without anyone asking."""
    store = MemoryStore()
    assert store.model_for(-100) is None
    store.set_model(-100, "gpt-5-mini")
    assert store.model_for(-100) == "gpt-5-mini"
    assert store.model_for(-200) is None
