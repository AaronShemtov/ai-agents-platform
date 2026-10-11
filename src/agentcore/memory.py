"""Per-conversation state: one chat, or one topic of a forum chat.

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
from typing import Any, NamedTuple

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


# How many turns the start of the history is snapped to. Four is a compromise:
# one expensive re-read buys three cheap turns, and the history carried is never
# more than four turns short of the budget. See ChatMemory.transcript.
TRIM_STEP = 4


class ConversationKey(NamedTuple):
    """Which conversation a message belongs to: a chat, and a topic inside it.

    `thread_id` is the forum topic. 0 means "no topic": a private chat, an ordinary
    group, or the General topic of a forum, which Telegram addresses by leaving the
    thread out. Every row stored before topics were told apart is 0 too.

    There is no default on `thread_id`, on purpose. A call site that forgot the topic
    would land in 0 without anything failing, and 0 is the shared pile this key
    exists to stop adding to.
    """

    chat_id: int
    thread_id: int


@dataclass
class ChatMemory:
    chat_id: int
    messages: list[dict[str, Any]] = field(default_factory=list)

    # How many of `messages` are already in the durable store. A turn writes
    # only what it added, rather than re-writing the transcript each time.
    persisted: int = 0

    # Whether this chat has been loaded from the durable store yet. Set once,
    # even if the load failed — a database that is down must not be asked again
    # on every message, and an agent with no memory still answers.
    hydrated: bool = False

    # The topic of `chat_id` this transcript belongs to; see ConversationKey. Keyword
    # only and defaulted because most constructions have no topic to speak of (the
    # coder and the reviewer keep a throwaway transcript per call), and MemoryStore,
    # which builds the real ones, always passes it.
    thread_id: int = field(default=0, kw_only=True)

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

    def transcript(
        self, system_prompt: str, *, max_tokens: int, step: int = TRIM_STEP
    ) -> list[dict[str, Any]]:
        """System prompt plus as much recent history as fits, trimmed by whole turns.

        Where the history starts is snapped to a multiple of `step` turns, and
        that is the whole point. Trimming to exactly what fits moves the first
        message on every single turn once the budget is reached, and a prefix
        that changes at token one is a prefix no cache can match.

        Measured on the local box, budget 1,200: while the window held still a
        turn cost 2.9s and 11.4s; the two turns where it slid cost 30.1s and
        29.7s, all four with the same amount of context. The difference is
        re-reading 1,100 tokens that had already been read.

        It is not only a latency question. Azure bills a cached input token at a
        tenth of a fresh one, so a prefix that survives is also the cheaper one —
        and the turn where sol spent 43.1M input tokens is what that costs when
        it does not.

        Rounding up rather than down: the snapped start drops at least as many
        turns as the budget required, so the result still fits. The price is
        carrying a little less history than the budget would allow, paid once per
        `step` turns instead of a full re-read every turn.
        """
        system = {"role": "system", "content": system_prompt}
        budget = max_tokens - estimate_tokens([system])

        turns = _split_into_turns(self.messages)
        kept = 0
        used = 0
        for turn in reversed(turns):
            cost = estimate_tokens(turn)
            # Always keep the most recent turn even if it alone blows the budget —
            # dropping it would mean answering with no idea what was asked.
            if kept and used + cost > budget:
                break
            kept += 1
            used += cost

        start = len(turns) - kept
        if step > 1 and start > 0:
            # Snap forward to the next multiple of `step`, never past the last
            # turn: the newest one is the question being answered.
            start = min(-(-start // step) * step, len(turns) - 1)

        history = [msg for turn in turns[start:] for msg in turn]
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
    """Conversation -> ChatMemory, and chat -> the model picked with /model.

    The transcript belongs to a conversation (a chat, or a topic of one), the model
    to the chat: /model has always applied to the whole chat, and moving it to the
    topic would have changed that without anyone asking.
    """

    def __init__(self) -> None:
        self._chats: dict[ConversationKey, ChatMemory] = {}
        self._models: dict[int, str] = {}

    def get(self, key: ConversationKey) -> ChatMemory:
        # `key.chat_id` rather than unpacking: a bare chat id, the old way of calling
        # this, fails here instead of quietly becoming a conversation of its own.
        if key not in self._chats:
            self._chats[key] = ChatMemory(chat_id=key.chat_id, thread_id=key.thread_id)
        return self._chats[key]

    def reset(self, key: ConversationKey) -> None:
        """Forget one conversation. The model picked for its chat stays."""
        self.get(key).reset()

    def model_for(self, chat_id: int) -> str | None:
        return self._models.get(chat_id)

    def set_model(self, chat_id: int, model: str) -> None:
        self._models[chat_id] = model
