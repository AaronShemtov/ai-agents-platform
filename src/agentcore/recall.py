"""The other conversations: second priority, reached when they are needed.

A conversation's own history goes into every turn whole: telegram.py replays it and
memory.py trims it. The other conversations the agent has had with the person — the
other topics of a forum, its General topic, the private chat — do not. Replaying them
together is what made each topic answer with the rest mixed in, which is why memory
was split by topic in the first place.

The split left the opposite gap: an agent that could not say what was decided in the
next topic even when asked to look. So the rest is reachable, in two steps, neither
of which puts it in the way:

  * a note and an index in the system prompt: the history below is this conversation
    and it comes first; these are the others, a line each — what it is, how it began,
    the day it last moved. Enough to know something was discussed elsewhere, not
    enough to be mistaken for this conversation;
  * three in-process tools, in localtools.py, to list the conversations, search them
    by words and read them. What was said elsewhere arrives when it is asked for, and
    labelled with where it came from.

The index changes only when another conversation does, never from one step of a turn
to the next: it sits at the front of the prompt, and a front that moves is a prefix no
cache can match (see ChatMemory.transcript).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence

from agentcore.memory import ConversationKey
from agentcore.store import ConversationSummary, StoredText

# The prefix Telegram puts on a supergroup's chat id. A link to one of its topics uses
# the id without it: chat -1001234567890, topic 78 -> https://t.me/c/1234567890/78.
_SUPERGROUP = "-100"

# How many other conversations the index names. The rest are a tool call away, and a
# line each for every topic ever opened would cost tokens on every turn.
INDEX_LIMIT = 12
INDEX_PREVIEW = 100
LIST_PREVIEW = 160

# Search terms. Substring matching makes a two-letter term find half the table; the
# words below are the short ones a sentence is made of, so a model that passes a whole
# question still searches by what the question is about.
MAX_TERMS = 6
MIN_TERM = 2
_STOP_WORDS = frozenset(
    {
        "и", "в", "во", "на", "не", "по", "с", "со", "к", "ко", "о", "об", "от", "до",
        "за", "из", "у", "а", "но", "да", "же", "ли", "то", "что", "как", "это", "мы",
        "ты", "он", "она", "они", "я", "мне", "нас", "для", "или", "так", "там", "тут",
        "the", "a", "an", "and", "or", "of", "to", "in", "on", "for", "is", "it", "be",
    }
)
_EDGES = "\"'«»„“”‘’`()[]{}<>.,;:!?…"

ROLE = {"user": "человек", "assistant": "ты"}

RECALL_NOTE = (
    "История ниже — только этого разговора. Он главный: отвечай по нему.\n"
    "Остальные разговоры с человеком — другие темы группы, General, личный чат — "
    "второй приоритет. Их истории здесь нет, но они доступны: "
    "memory__search_conversations ищет в них по словам, memory__read_conversation "
    "показывает разговор или кусок вокруг найденного сообщения, "
    "memory__list_conversations перечисляет все. Загляни туда, когда человек ссылается "
    "на то, чего в этом разговоре нет («как договорились», «в другой теме», "
    "«помнишь»), или когда ответ зависит от прошлой работы: сначала поищи, и только "
    "если не нашёл, говори, что не знаешь. Без такой причины оттуда ничего не бери.\n"
    "Взял что-то из другого разговора — скажи откуда: тема (ссылкой, если она есть) и "
    "дата. Если там сказано иначе, чем здесь, верь этому разговору. Факты о человеке "
    "общие для всех разговоров; если этот разговор им противоречит, верь разговору."
)


# -- naming a conversation ----------------------------------------------------


def handle(chat_id: int, thread_id: int) -> str:
    """How a conversation is named to the model, and how the model names it back."""
    return f"{chat_id}/{thread_id}"


def parse_handle(text: str, *, current_chat: int | None) -> tuple[int, int] | None:
    """The conversation the model named, as (chat, topic); None if it named none.

    Three spellings: the handle as the tools print it, "-1001234567890/78"; a topic
    link, which is what the model shows people and so what it may copy back; and a
    bare topic number, meaning a topic of the chat this turn is in. Topic 1 is the
    General topic's own id and is stored as 0, like everything outside a topic.
    """
    text = (text or "").strip()
    if match := re.fullmatch(r"(-?\d+)\s*/\s*(\d+)", text):
        chat, thread = int(match[1]), int(match[2])
    elif match := re.fullmatch(r"(?:https?://)?t\.me/c/(\d+)/(\d+)/?", text):
        chat, thread = int(_SUPERGROUP + match[1]), int(match[2])
    elif current_chat is not None and re.fullmatch(r"\d+", text):
        chat, thread = current_chat, int(text)
    else:
        return None
    return chat, (thread if thread > 1 else 0)


def topic_link(chat_id: int, thread_id: int) -> str | None:
    """A t.me link that opens a forum topic for a member of the group.

    None for anything else: a private chat has no such link, and the General topic is
    not a topic Telegram can link to by number.
    """
    chat = str(chat_id)
    if thread_id <= 0 or not chat.startswith(_SUPERGROUP):
        return None
    return f"https://t.me/c/{chat[len(_SUPERGROUP):]}/{thread_id}"


def forum_chats(conversations: Iterable[ConversationSummary]) -> set[int]:
    """The chats known to have topics: those with a conversation in any topic."""
    return {c.chat_id for c in conversations if c.thread_id}


def label(chat_id: int, thread_id: int, *, forums: set[int]) -> str:
    """What a person would call the conversation."""
    if chat_id > 0:
        return "личный чат"
    if thread_id:
        return f"тема {thread_id}"
    return "General" if chat_id in forums else "группа"


def _is(current: ConversationKey | None, chat_id: int, thread_id: int) -> bool:
    return current is not None and (current.chat_id, current.thread_id) == (chat_id, thread_id)


def where(
    chat_id: int, thread_id: int, *, forums: set[int], current: ConversationKey | None
) -> str:
    """One line that names a conversation: what it is, its link, its id for the tools."""
    name = label(chat_id, thread_id, forums=forums)
    if _is(current, chat_id, thread_id):
        name += " (этот разговор)"
    link = topic_link(chat_id, thread_id)
    return " · ".join([name, *([link] if link else []), f"id {handle(chat_id, thread_id)}"])


def _one_line(text: str | None, limit: int) -> str:
    flat = " ".join((text or "").split())
    return flat if len(flat) <= limit else flat[: limit - 1].rstrip() + "…"


def _units(text: str) -> int:
    """Length the way Oracle counts a CLOB: in UTF-16 code units, so a character
    outside the Basic Multilingual Plane — most emoji — counts twice. Comparing a
    Python len() with `dbms_lob.getlength` reports a cut that never happened."""
    return len(text.encode("utf-16-le")) // 2


# -- what goes into the system prompt -------------------------------------------


def index_block(
    conversations: Sequence[ConversationSummary],
    current: ConversationKey,
    *,
    limit: int = INDEX_LIMIT,
) -> str:
    """A line for each other conversation, the most recently active first.

    Only what stays put between two turns of this conversation: how a topic began, the
    day another conversation last moved. Counts and the latest message are left to
    memory__list_conversations, because they would change the prompt's front on every
    turn of whichever conversation they describe.
    """
    others = [c for c in conversations if not _is(current, c.chat_id, c.thread_id)]
    if not others:
        return "Других разговоров с ним пока нет."
    forums = forum_chats(conversations)
    lines = []
    for c in others[:limit]:
        name = label(c.chat_id, c.thread_id, forums=forums)
        link = topic_link(c.chat_id, c.thread_id)
        parts = [name + (f" ({link})" if link else "")]
        if c.thread_id and c.first_user:
            parts.append(f"началась с «{_one_line(c.first_user, INDEX_PREVIEW)}»")
        elif c.chat_id < 0 and c.chat_id in forums:
            parts.append("общая часть группы, в ней же всё, что было до разделения по темам")
        parts.append(f"последнее сообщение {c.last_at[:10]}")
        lines.append("- " + " — ".join(parts))
    more = len(others) - limit
    tail = f"\n…и ещё {more}: полный список — memory__list_conversations." if more > 0 else ""
    return "Другие разговоры с ним, свежие сверху:\n" + "\n".join(lines) + tail


def prompt_block(
    conversations: Sequence[ConversationSummary] | None, current: ConversationKey
) -> str:
    """The note, then the index. Without a listing — the database did not answer —
    the note alone: the tools may still work by the time they are called."""
    if conversations is None:
        return RECALL_NOTE
    return RECALL_NOTE + "\n\n" + index_block(conversations, current)


# -- what the tools answer -------------------------------------------------------


def terms_of(query: str) -> list[str]:
    """The words of a query worth searching for, lower case, at most MAX_TERMS."""
    terms: list[str] = []
    for raw in (query or "").lower().split():
        word = raw.strip(_EDGES)
        if len(word) >= MIN_TERM and word not in _STOP_WORDS and word not in terms:
            terms.append(word)
    return terms[:MAX_TERMS]


def format_list(
    conversations: Sequence[ConversationSummary], current: ConversationKey | None
) -> str:
    if not conversations:
        return "Разговоров в памяти нет."
    forums = forum_chats(conversations)
    blocks = [f"Разговоров: {len(conversations)}. Свежие сверху, время UTC."]
    for c in conversations:
        lines = [
            "- " + where(c.chat_id, c.thread_id, forums=forums, current=current),
            f"  сообщений: {c.messages}, с {c.first_day} по {c.last_at}",
        ]
        if c.first_user:
            lines.append(f"  первое от человека: «{_one_line(c.first_user, LIST_PREVIEW)}»")
        if c.last_user and c.last_user != c.first_user:
            lines.append(f"  последнее от человека: «{_one_line(c.last_user, LIST_PREVIEW)}»")
        blocks.append("\n".join(lines))
    return "\n".join(blocks)


def _window(text: StoredText) -> str:
    """The excerpt, with an ellipsis on each side something was cut from."""
    body = " ".join(text.text.split())
    head = "…" if text.start > 1 else ""
    tail = "…" if text.start - 1 + _units(text.text) < text.length else ""
    return f"{head}{body}{tail}"


def format_hits(
    hits: Sequence[StoredText],
    conversations: Sequence[ConversationSummary],
    current: ConversationKey | None,
    terms: Sequence[str],
) -> str:
    if not hits:
        return (
            f"Ничего не нашлось по словам: {', '.join(terms)}. Попробуй другие слова "
            "или основы короче — ищется часть слова, без окончаний."
        )
    forums = forum_chats(conversations)
    out = [
        f"Нашлось сообщений: {len(hits)}. Сначала те, где совпало больше слов, "
        "при равенстве — свежие. Время UTC."
    ]
    for hit in hits:
        place = where(hit.chat_id, hit.thread_id, forums=forums, current=current)
        out.append(
            f"\n[{hit.id}] {place} · {hit.at} · {ROLE.get(hit.role, hit.role)}:\n"
            f"{_window(hit)}"
        )
    out.append(
        "\nПрочитать вокруг: memory__read_conversation, conversation — id разговора, "
        "around — номер сообщения в скобках."
    )
    return "\n".join(out)


def format_read(
    messages: Sequence[StoredText],
    *,
    chat_id: int,
    thread_id: int,
    conversations: Sequence[ConversationSummary],
    current: ConversationKey | None,
    around: int | None,
) -> str:
    place = where(chat_id, thread_id, forums=forum_chats(conversations), current=current)
    if not messages:
        return (
            f"{place}: сообщений нет — или такого разговора нет. "
            "Список разговоров — memory__list_conversations."
        )
    out = [f"{place} · время UTC"]
    if around is not None and all(m.id != around for m in messages):
        out.append(f"Сообщения {around} в этом разговоре нет; ниже — соседние с этим номером.")
    for message in messages:
        mark = "→ " if message.id == around else ""
        left = message.length - _units(message.text)
        cut = f"\n… [дальше ещё {left} симв.]" if left > 0 else ""
        out.append(
            f"\n{mark}[{message.id}] {message.at} · {ROLE.get(message.role, message.role)}:\n"
            f"{message.text}{cut}"
        )
    return "\n".join(out)
