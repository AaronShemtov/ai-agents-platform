"""Per-chat conversation state.

In-memory for now. That means history is lost on restart, which for a single-replica
Deployment with `strategy: Recreate` happens on every rollout. Accepted for phase 1;
a durable store (Oracle ADB via SODA REST, or OCI Object Storage — no PVC is possible
on this cluster) is phase 2.

The one genuinely tricky part is trimming. A naive "drop the oldest messages" leaves
orphaned `tool` messages whose parent assistant message — the one carrying the
matching `tool_calls` — has been dropped. The API rejects that transcript outright,
so trimming works on whole turns: a `user` message and everything it triggered.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# Rough characters-per-token. Only used to decide when to trim, so being off by 20%
# costs a little context, not correctness.
_CHARS_PER_TOKEN = 3.5


def estimate_tokens(messages: list[dict[str, Any]]) -> int:
    chars = 0
    for message in messages:
        content = message.get("content")
        if isinstance(content, str):
            chars += len(content)
        elif content:
            chars += len(str(content))
        for call in message.get("tool_calls") or []:
            chars += len(str(call))
    return int(chars / _CHARS_PER_TOKEN)


@dataclass
class ChatMemory:
    chat_id: int
    model: str | None = None  # per-chat override set by /model
    messages: list[dict[str, Any]] = field(default_factory=list)

    # How many of `messages` are already in the durable store. A turn writes
    # only what it added, rather than re-writing the transcript each time.
    persisted: int = 0

    # Whether this chat has been loaded from the durable store yet. Set once,
    # even if the load failed — a database that is down must not be asked again
    # on every message, and an agent with no memory still answers.
    hydrated: bool = False

    def append(self, message: dict[str, Any]) -> None:
        self.messages.append(message)

    def adopt(self, messages: list[dict[str, Any]]) -> None:
        """Replace the transcript with one restored from durable storage.

        `persisted` follows, because these messages are by definition already
        stored; leaving it at zero would write the whole history back on the
        next turn.
        """
        self.messages = list(messages)
        self.persisted = len(self.messages)

    def unpersisted(self) -> list[dict[str, Any]]:
        return self.messages[self.persisted :]

    def mark_persisted(self) -> None:
        self.persisted = len(self.messages)

    def reset(self) -> None:
        self.messages.clear()
        # Not a count of anything that still exists.
        self.persisted = 0

    def transcript(self, system_prompt: str, *, max_tokens: int) -> list[dict[str, Any]]:
        """System prompt plus as much recent history as fits, trimmed by whole turns."""
        system = {"role": "system", "content": system_prompt}
        budget = max_tokens - estimate_tokens([system])

        turns = _split_into_turns(self.messages)
        kept: list[list[dict[str, Any]]] = []
        used = 0
        for turn in reversed(turns):
            cost = estimate_tokens(turn)
            # Always keep the most recent turn even if it alone blows the budget —
            # dropping it would mean answering with no idea what was asked.
            if kept and used + cost > budget:
                break
            kept.append(turn)
            used += cost

        history = [msg for turn in reversed(kept) for msg in turn]
        return [system, *history]


def _split_into_turns(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Group messages into turns, each starting at a `user` message.

    Tool calls and their results always sit between two user messages, so cutting only
    at user boundaries keeps every assistant/tool pair intact.
    """
    turns: list[list[dict[str, Any]]] = []
    current: list[dict[str, Any]] = []
    for message in messages:
        if message.get("role") == "user" and current:
            turns.append(current)
            current = []
        current.append(message)
    if current:
        turns.append(current)
    return turns


class MemoryStore:
    """Chat id -> ChatMemory."""

    def __init__(self) -> None:
        self._chats: dict[int, ChatMemory] = {}

    def get(self, chat_id: int) -> ChatMemory:
        if chat_id not in self._chats:
            self._chats[chat_id] = ChatMemory(chat_id=chat_id)
        return self._chats[chat_id]

    def reset(self, chat_id: int) -> None:
        self.get(chat_id).reset()
