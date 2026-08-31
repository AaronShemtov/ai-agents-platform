"""Transcript trimming.

The failure this guards against is subtle: dropping messages from the front can leave
a `tool` message whose parent assistant message — the one carrying the matching
`tool_calls` — has been trimmed away. The API rejects that transcript outright, and
the symptom is a 400 that appears only on long conversations.
"""

from __future__ import annotations

from typing import Any

from agentcore.memory import ChatMemory, MemoryStore, estimate_tokens


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
    store.get(1).append(user("a"))
    store.get(2).append(user("b"))
    store.reset(1)
    assert store.get(1).messages == []
    assert len(store.get(2).messages) == 1


def test_reset_keeps_the_model_override() -> None:
    # /new clears the conversation, not the model the user picked with /model.
    store = MemoryStore()
    chat = store.get(1)
    chat.model = "gpt-5-mini"
    chat.append(user("a"))
    store.reset(1)
    assert chat.model == "gpt-5-mini"
